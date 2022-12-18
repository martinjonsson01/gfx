use wgpu::Color;
use winit::event::WindowEvent;
use winit::window::Window;

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
}

impl State {
    pub(crate) async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        assert_ne!(size.width, 0, "can't render to zero-width area");
        assert_ne!(size.height, 0, "can't render to zero-height area");

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("should be able to find adapter that accepts provided options");

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
            .expect("should be able to find device that matches descriptor");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: *surface
                .get_supported_formats(&adapter)
                .first()
                .expect("should support at least one format"),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo, // VSync
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&device, &config);

        let clear_color = Color {
            r: 0.1,
            g: 0.2,
            b: 0.3,
            a: 1.0,
        };

        Self {
            surface,
            device,
            queue,
            config,
            size,
            clear_color,
        }
    }

    pub(crate) fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Returns whether an event has been fully processed.
    pub(crate) fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let cursor_x_normalized = position.x / self.size.width as f64;
                let cursor_y_normalized = position.y / self.size.height as f64;
                self.clear_color = Color {
                    r: cursor_x_normalized,
                    g: cursor_y_normalized,
                    b: cursor_x_normalized,
                    a: 1.0,
                };
                true
            }
            _ => {
                // Don't capture the event.
                false
            }
        }
    }

    pub(crate) fn update(&mut self) {}

    pub(crate) fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
