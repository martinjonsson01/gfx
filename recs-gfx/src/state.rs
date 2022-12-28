use crate::camera::{Camera, CameraController, Projection};
use crate::instance::{ModelInstances, Transform, TransformRaw};
use crate::model::{DrawLight, DrawModel, Model, ModelVertex, Vertex};
use crate::shader_locations::{
    FRAGMENT_DIFFUSE_SAMPLER, FRAGMENT_DIFFUSE_TEXTURE, FRAGMENT_MATERIAL_UNIFORM,
    FRAGMENT_NORMAL_SAMPLER, FRAGMENT_NORMAL_TEXTURE,
};
use crate::state::StateError::ModelLoad;
use crate::texture::Texture;
use crate::{resources, CameraUniform, EventPropagation};
use cgmath::prelude::*;
use cgmath::{Deg, Quaternion, Vector3};
use std::time::Duration;
use thiserror::Error;
use wgpu::util::DeviceExt;
use winit::event::{ElementState, KeyboardInput, MouseButton, WindowEvent};
use winit::window::Window;

#[derive(Error, Debug)]
pub enum StateError {
    #[error("the window width can't be 0")]
    WindowWidthZero,
    #[error("the window height can't be 0")]
    WindowHeightZero,
    #[error("an adapter that matches the requirements can not be found")]
    AdapterNotFound,
    #[error("a device that matches the requirements can not be found")]
    DeviceNotFound(#[source] wgpu::RequestDeviceError),
    #[error("the surface `{surface}` is not compatible with the available adapter `{adapter}`")]
    SurfaceIncompatibleWithAdapter { surface: String, adapter: String },
    #[error("failed to load model from path `{1}`")]
    ModelLoad(#[source] resources::LoadError, String),
    #[error("failed to get output texture")]
    MissingOutputTexture(#[source] wgpu::SurfaceError),
    #[error("model handle `{0}` is invalid")]
    InvalidModelHandle(ModelHandle),
}

type StateResult<T, E = StateError> = Result<T, E>;

/// A point-light that emits light in every direction and has no area.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PointLightUniform {
    position: [f32; 3],
    _padding: u32,
    color: [f32; 3],
    _padding2: u32,
}

#[derive(Debug)]
pub(crate) struct State {
    /// The part of the [`Window`] that we draw to.
    surface: wgpu::Surface,
    /// Current GPU.
    device: wgpu::Device,
    /// Command queue to the GPU.
    queue: wgpu::Queue,
    /// Defines how the [`wgpu::Surface`] creates its underlying [`wgpu::SurfaceTexture`]s
    config: wgpu::SurfaceConfiguration,
    /// The physical size of the [`Window`]'s content area.
    pub(crate) size: winit::dpi::PhysicalSize<u32>,
    /// What color to clear the display with every frame.
    clear_color: wgpu::Color,
    /// The camera that views the scene.
    camera: Camera,
    /// A description of the viewport to project onto.
    projection: Projection,
    /// A controller for moving the camera around based on user input.
    pub(crate) camera_controller: CameraController,
    /// The uniform-representation of the camera.
    camera_uniform: CameraUniform,
    /// The camera uniform buffer that can be sent to shaders.
    camera_buffer: wgpu::Buffer,
    /// Binding the camera uniform to the shaders.
    camera_bind_group: wgpu::BindGroup,
    /// How the GPU acts on a set of data.
    render_pipeline: wgpu::RenderPipeline,
    /// How textures are laid out in memory.
    material_bind_group_layout: wgpu::BindGroupLayout,
    /// Model instances, to allow for one model to be shown multiple times with different transforms.
    instances: Vec<ModelInstances>,
    /// Used for depth-testing (z-culling) to render pixels in front of each other correctly.
    depth_texture: Texture,
    /// The models that can be rendered.
    models: Vec<Model>,
    /// The uniform-representation of the light. (currently only single light supported)
    light_uniform: PointLightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_model: Model,
    /// How the GPU acts on lights.
    light_render_pipeline: wgpu::RenderPipeline,
    /// Whether the mouse button is pressed or not.
    pub(crate) mouse_pressed: bool,
}

/// An identifier for a specific model that has been loaded into the engine.
pub type ModelHandle = usize;

/// An identifier for a group of model instances.
pub type InstancesHandle = usize;

impl State {
    pub(crate) async fn load_model(&mut self, path: &str) -> StateResult<ModelHandle> {
        let obj_model = resources::load_model(
            path,
            &self.device,
            &self.queue,
            &self.material_bind_group_layout,
        )
        .await
        .map_err(|e| ModelLoad(e, path.to_string()))?;

        let index = self.models.len();
        self.models.push(obj_model);
        Ok(index)
    }

    pub(crate) fn create_model_instances(
        &mut self,
        model: ModelHandle,
        transforms: Vec<Transform>,
    ) -> StateResult<InstancesHandle> {
        if model >= self.models.len() {
            return Err(StateError::InvalidModelHandle(model));
        }

        let instances = ModelInstances::new(&self.device, model, transforms);

        let index = self.instances.len();
        self.instances.push(instances);
        Ok(index)
    }
}

impl State {
    pub(crate) async fn new(window: &Window) -> StateResult<Self> {
        let size = window.inner_size();

        if size.width == 0 {
            return Err(StateError::WindowWidthZero);
        }
        if size.height == 0 {
            return Err(StateError::WindowHeightZero);
        }

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        };
        let adapter = instance
            .request_adapter(&adapter_options)
            .await
            .ok_or_else(|| StateError::AdapterNotFound)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None,
            )
            .await
            .map_err(StateError::DeviceNotFound)?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: *surface
                .get_supported_formats(&adapter)
                .first()
                .ok_or_else(|| StateError::SurfaceIncompatibleWithAdapter {
                    surface: format!("{surface:?}"),
                    adapter: format!("{adapter:?}"),
                })?,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox, // Fast VSync
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&device, &config);

        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Material properties.
                    wgpu::BindGroupLayoutEntry {
                        binding: FRAGMENT_MATERIAL_UNIFORM,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Diffuse texture.
                    Self::create_texture_2d_layout(FRAGMENT_DIFFUSE_TEXTURE),
                    Self::create_sampler_2d_layout(FRAGMENT_DIFFUSE_SAMPLER),
                    // Normal map.
                    Self::create_texture_2d_layout(FRAGMENT_NORMAL_TEXTURE),
                    Self::create_sampler_2d_layout(FRAGMENT_NORMAL_SAMPLER),
                ],
                label: Some("material_bind_group_layout"),
            });

        let clear_color = wgpu::Color {
            r: 0.1,
            g: 0.2,
            b: 0.3,
            a: 1.0,
        };

        let camera = Camera::new((0.0, 5.0, 10.0), Deg(-90.0), Deg(-20.0));
        let projection = Projection::new(config.width, config.height, Deg(70.0), 0.1, 100.0);
        let camera_controller = CameraController::new(7.0, 1.0);
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_projection(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[Self::create_uniform_layout(0)],
                label: Some("camera_bind_group_layout"),
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let light_uniform = PointLightUniform {
            position: [5.0, 5.0, 5.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0],
            _padding2: 0,
        };
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light VB"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let light_model_file_name = "cube.obj";
        let light_model = resources::load_model(
            light_model_file_name,
            &device,
            &queue,
            &material_bind_group_layout,
        )
        .await
        .map_err(|e| ModelLoad(e, light_model_file_name.to_string()))?;
        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[Self::create_uniform_layout(0)],
                label: Some("light_bind_group_layout"),
            });
        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &material_bind_group_layout,
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(Texture::DEPTH_FORMAT),
                &[ModelVertex::descriptor(), TransformRaw::descriptor()],
                shader,
            )
        };

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(Texture::DEPTH_FORMAT),
                &[ModelVertex::descriptor()],
                shader,
            )
        };

        let depth_texture = Texture::create_depth_texture(&device, &config, "depth_texture");

        Ok(Self {
            surface,
            device,
            queue,
            size,
            config,
            clear_color,
            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            render_pipeline,
            material_bind_group_layout,
            instances: vec![],
            depth_texture,
            models: vec![],
            light_uniform,
            light_buffer,
            light_bind_group,
            light_render_pipeline,
            light_model,
            mouse_pressed: false,
        })
    }

    fn create_uniform_layout(binding: wgpu::ShaderLocation) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn create_sampler_2d_layout(binding: wgpu::ShaderLocation) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }
    }

    fn create_texture_2d_layout(binding: wgpu::ShaderLocation) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
            },
            count: None,
        }
    }

    pub(crate) fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                Texture::create_depth_texture(&self.device, &self.config, "depth_texture");

            self.projection.resize(new_size.width, new_size.height);
        }
    }

    pub(crate) fn input(&mut self, event: &WindowEvent) -> EventPropagation {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(key),
                        state,
                        ..
                    },
                ..
            } => self.camera_controller.process_keyboard(*key, *state),
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                EventPropagation::Consume
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                EventPropagation::Consume
            }
            _ => EventPropagation::Propagate,
        }
    }

    pub(crate) fn update(&mut self, dt: Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_projection(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Animate instance rotation
        let rotation_delta = 10.0 * dt.as_secs_f32();
        for instances in &mut self.instances {
            let new_transforms = instances
                .transforms()
                .iter()
                .map(|old_transform| {
                    let rotation_axis = if old_transform.position.is_zero() {
                        Vector3::unit_z()
                    } else {
                        old_transform.position.normalize()
                    };
                    let rotation = Quaternion::from_axis_angle(rotation_axis, Deg(rotation_delta));
                    Transform {
                        rotation: old_transform.rotation * rotation,
                        ..*old_transform
                    }
                })
                .collect();
            instances.update_transforms(&self.device, new_transforms);
        }

        // Animate light rotation
        const DEGREES_PER_SECOND: f32 = 60.0;
        let old_position: Vector3<_> = self.light_uniform.position.into();
        let rotation = Quaternion::from_axis_angle(
            Vector3::unit_y(),
            Deg(DEGREES_PER_SECOND * dt.as_secs_f32()),
        );
        self.light_uniform.position = (rotation * old_position).into();
        self.queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );
    }

    pub(crate) fn render(&mut self) -> StateResult<()> {
        let output = self
            .surface
            .get_current_texture()
            .map_err(StateError::MissingOutputTexture)?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.light_render_pipeline);
            render_pass.draw_light_model(
                &self.light_model,
                &self.camera_bind_group,
                &self.light_bind_group,
            );

            render_pass.set_pipeline(&self.render_pipeline);
            for model_instances in &self.instances {
                render_pass.set_vertex_buffer(1, model_instances.buffer_slice());
                render_pass.draw_model_instanced(
                    &self.models[model_instances.model],
                    model_instances.instances(),
                    &self.camera_bind_group,
                    &self.light_bind_group,
                );
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
            // or Features::POLYGON_MODE_POINT
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        // If the pipeline will be used with a multiview render pass, this
        // indicates how many array layers the attachments will have.
        multiview: None,
    })
}
