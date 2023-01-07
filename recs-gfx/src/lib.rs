//! A simple graphics engine that provides a simple API for simple graphics.

// rustc lints
#![warn(
    let_underscore,
    nonstandard_style,
    unused,
    explicit_outlives_requirements,
    meta_variable_misuse,
    missing_debug_implementations,
    missing_docs,
    non_ascii_idents,
    noop_method_call,
    pointer_structural_match,
    trivial_casts,
    trivial_numeric_casts
)]
// clippy lints
#![warn(
    clippy::cognitive_complexity,
    clippy::dbg_macro,
    clippy::if_then_some_else_none,
    clippy::print_stdout,
    clippy::rc_mutex,
    clippy::unwrap_used,
    clippy::large_enum_variant
)]

mod camera;
mod instance;
mod model;
mod resources;
mod state;
mod texture;
mod uniform;

use crate::camera::{Camera, Projection};
pub use crate::instance::Transform;
use crate::state::{InstancesHandle, ModelHandle, State, StateError};
use cgmath::{Matrix4, SquareMatrix};
use color_eyre::Report;
use crossbeam_queue::ArrayQueue;
use derivative::Derivative;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::dispatcher::DefaultGuard;
use tracing::{error, info, instrument};
use tracing_subscriber::filter::ParseError;
use winit::window::Window;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod shader_locations {
    use wgpu::ShaderLocation;

    pub const VERTEX_POSITION: ShaderLocation = 0;
    pub const VERTEX_TEXTURE_COORDINATES: ShaderLocation = 1;
    pub const VERTEX_NORMAL: ShaderLocation = 2;
    pub const VERTEX_TANGENT: ShaderLocation = 3;
    pub const VERTEX_BITANGENT: ShaderLocation = 4;

    pub const FRAGMENT_MATERIAL_UNIFORM: ShaderLocation = 0;
    pub const FRAGMENT_DIFFUSE_TEXTURE: ShaderLocation = 1;
    pub const FRAGMENT_DIFFUSE_SAMPLER: ShaderLocation = 2;
    pub const FRAGMENT_NORMAL_TEXTURE: ShaderLocation = 3;
    pub const FRAGMENT_NORMAL_SAMPLER: ShaderLocation = 4;

    pub const INSTANCE_MODEL_MATRIX_COLUMN_0: ShaderLocation = 5;
    pub const INSTANCE_MODEL_MATRIX_COLUMN_1: ShaderLocation = 6;
    pub const INSTANCE_MODEL_MATRIX_COLUMN_2: ShaderLocation = 7;
    pub const INSTANCE_MODEL_MATRIX_COLUMN_3: ShaderLocation = 8;
    pub const INSTANCE_NORMAL_MATRIX_COLUMN_0: ShaderLocation = 9;
    pub const INSTANCE_NORMAL_MATRIX_COLUMN_1: ShaderLocation = 10;
    pub const INSTANCE_NORMAL_MATRIX_COLUMN_2: ShaderLocation = 11;
}

/// A representation of the [`Camera`] that can be sent into shaders through a uniform buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    /// The world-space location of the camera.
    view_position: [f32; 4],
    /// Transformation matrix that transforms from world space to view space to clip space.
    view_projection: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_projection: Matrix4::identity().into(),
        }
    }

    pub fn update_view_projection(&mut self, camera: &Camera, projection: &Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        self.view_projection = (projection.perspective_matrix() * camera.view_matrix()).into();
    }
}

/// An error that has occurred within the graphics engine.
#[derive(Error, Debug)]
pub enum EngineError {
    /// Could not construct root Window.
    #[error("could not construct root Window")]
    MissingRootWindow(#[source] winit::error::OsError),
    /// Could not instantiate State.
    #[error("could not instantiate State")]
    StateConstruction(#[source] Box<StateError>),
    /// Failed to render frame.
    #[error("failed to render frame")]
    Rendering(#[source] Box<StateError>),
    /// Could not load model.
    #[error("could not load model `{1}`")]
    ModelLoad(#[source] Box<StateError>, PathBuf),
    /// Could not create event filter from environment variable.
    #[error("could not create event filter from environment variable")]
    EnvironmentEventFilter(#[source] ParseError),
    /// Could not create a new object.
    #[error("a new object could not be created")]
    ObjectCreation(#[source] Box<StateError>),
    /// Could not create a single new object.
    #[error("a single new object could not be created")]
    SingleObjectCreation(),
    /// The simulation thread panicked.
    #[error("the simulation thread panicked")]
    SimulationThreadPanic(),
}

/// Whether an engine operation failed or succeeded.
pub type EngineResult<T, E = EngineError> = Result<T, E>;

/// Starts the engine.
///
/// The render loop runs on the calling thread,
/// while the simulation loop is started in a new thread.
///
/// # Examples
/// ```no_run
/// # use std::error::Error;
/// use recs_gfx::{EngineError, EngineResult, GraphicsEngine};
///
/// fn init_gfx(gfx: &mut GraphicsEngine<'_>) -> EngineResult<()> {
///     gfx.load_model(std::path::Path::new("path/to/model.obj"))?;
///     Ok(())
/// }
/// # fn main() -> Result<(), Box<dyn Error>> {
///
/// recs_gfx::start(init_gfx)?;
/// #   Ok(())
/// # }
/// ```
pub fn start<Func>(initialize_gfx: Func) -> EngineResult<()>
where
    Func: FnOnce(&mut GraphicsEngine<'_>) -> EngineResult<()> + Send + Sync,
{
    // A FIFO-queue buffer introduces a slight latency, but decreases lock contention
    // as opposed to a LIFO-queue.
    const TRANSFORM_BUFFER_SIZE: usize = 2;
    let object_queue: ArrayQueue<Vec<_>> = ArrayQueue::new(TRANSFORM_BUFFER_SIZE);

    let simulation_thread = thread::spawn(move || loop {
        info!("testing");
        thread::sleep(Duration::from_secs(1))
    });

    let mut gfx = GraphicsEngine::new(&object_queue)?;
    initialize_gfx(&mut gfx)?;
    gfx.run()?;

    simulation_thread.join().map_err(|e| {
        error!("{e:?}");
        EngineError::SimulationThreadPanic()
    })
}

/// The core data structure of the rendering system.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct GraphicsEngine<'queue> {
    window: Window,
    #[derivative(Debug = "ignore")]
    egui_context: egui::Context,
    #[derivative(Debug = "ignore")]
    egui_state: egui_winit::State,
    event_loop: EventLoop<()>,
    state: State,
    last_render_time: Instant,
    object_queue: &'queue ArrayQueue<Vec<Object>>,
}

impl<'queue> GraphicsEngine<'queue> {
    /// Tries to create a new engine instance, but it may fail if there isn't hardware support.
    pub fn new(object_queue: &'queue ArrayQueue<Vec<Object>>) -> EngineResult<GraphicsEngine> {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .build(&event_loop)
            .map_err(EngineError::MissingRootWindow)?;

        let egui_context = egui::Context::default();
        let mut egui_state = egui_winit::State::new(&event_loop);
        egui_state.set_pixels_per_point(window.scale_factor() as f32);

        let state = State::new(&window).map_err(|e| EngineError::StateConstruction(Box::new(e)))?;

        Ok(Self {
            window,
            egui_context,
            egui_state,
            event_loop,
            state,
            last_render_time: Instant::now(),
            object_queue,
        })
    }

    /// Starts the graphics engine, opening a new window and rendering to it.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::error::Error;
    /// use crossbeam_queue::ArrayQueue;
    /// use recs_gfx::{EngineError, GraphicsEngine};
    ///
    /// let queue = ArrayQueue::new(1);
    /// let graphics_engine = GraphicsEngine::new(&queue)?;
    ///
    /// graphics_engine.run()?;
    /// # Ok::<(), recs_gfx::EngineError>(())
    /// ```
    #[instrument(skip(self))]
    pub fn run(mut self) -> EngineResult<()> {
        let _guard = install_tracing()?;

        self.event_loop.run(move |event, _, control_flow| {
            let result = handle_events(
                &self.window,
                &mut self.egui_context,
                &mut self.egui_state,
                &mut self.state,
                &mut self.last_render_time,
                event,
                control_flow,
            );
            if let Err(error) = result {
                let report = Report::new(error);
                error!("{report:?}");
                *control_flow = ControlFlow::ExitWithCode(1); // Non-zero exit code means error.
            }
        });
    }

    /// Loads a model into the engine.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::error::Error;
    /// use crossbeam_queue::ArrayQueue;
    /// use recs_gfx::GraphicsEngine;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let queue = ArrayQueue::new(1);
    /// let mut graphics_engine = GraphicsEngine::new(&queue)?;
    ///
    /// let model_path = std::path::Path::new("path/to/model.obj");
    /// let model_handle = graphics_engine.load_model(model_path)?;
    /// #   Ok(())
    /// # }
    /// ```
    #[instrument(skip(self))]
    pub fn load_model(&mut self, path: &Path) -> EngineResult<ModelHandle> {
        self.state
            .load_model(path)
            .map_err(|e| EngineError::ModelLoad(Box::new(e), path.to_owned()))
    }

    /// Creates an object in the world.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::error::Error;
    /// use cgmath::{Quaternion, Vector3, Zero};
    /// use crossbeam_queue::ArrayQueue;
    /// use recs_gfx::{GraphicsEngine, Transform};
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let queue = ArrayQueue::new(1);
    /// let mut graphics_engine = GraphicsEngine::new(&queue)?;
    ///
    /// let model_path = std::path::Path::new("path/to/model.obj");
    /// let model_handle = graphics_engine.load_model(model_path)?;
    ///
    /// let transform = Transform {
    ///     position: Vector3::new(0.0, 10.0, 0.0),
    ///     rotation: Quaternion::zero(),
    ///     scale: Vector3::new(1.0, 1.0, 1.0),
    /// };
    /// let object = graphics_engine.create_object(model_handle, transform)?;
    /// #   Ok(())
    /// # }
    /// ```
    #[instrument(skip(self))]
    pub fn create_object(
        &mut self,
        model: ModelHandle,
        transform: Transform,
    ) -> EngineResult<Object> {
        self.create_objects(model, vec![transform])?
            .pop()
            .ok_or_else(EngineError::SingleObjectCreation)
    }

    /// Creates multiple objects with the same model in the world.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::error::Error;
    /// use cgmath::{One, Quaternion, Vector3, Zero};
    /// use crossbeam_queue::ArrayQueue;
    /// use recs_gfx::{GraphicsEngine, Object, Transform};
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let queue = ArrayQueue::new(1);
    /// let mut graphics_engine = GraphicsEngine::new(&queue)?;
    ///
    /// let model_path = std::path::Path::new("path/to/model.obj");
    /// let model_handle = graphics_engine.load_model(model_path)?;
    ///
    /// const NUMBER_OF_TRANSFORMS: usize = 10;
    /// let transforms = (0..NUMBER_OF_TRANSFORMS)
    ///     .map(|_| Transform {
    ///         position: Vector3::zero(),
    ///         rotation: Quaternion::one(),
    ///         scale: Vector3::new(1.0, 1.0, 1.0),
    ///     })
    ///     .collect();
    /// let objects: Vec<Object> = graphics_engine.create_objects(model_handle, transforms)?;
    ///
    /// assert_eq!(objects.len(), NUMBER_OF_TRANSFORMS);
    /// #   Ok(())
    /// # }
    /// ```
    #[instrument(skip(self))]
    pub fn create_objects(
        &mut self,
        model: ModelHandle,
        transforms: Vec<Transform>,
    ) -> EngineResult<Vec<Object>> {
        let instances_group = self
            .state
            .create_model_instances(model, transforms.clone())
            .map_err(|e| EngineError::ObjectCreation(Box::new(e)))?;
        Ok(transforms
            .into_iter()
            .map(|transform| Object {
                transform,
                model,
                _instances_group: instances_group,
            })
            .collect())
    }
}

/// A graphical object that can be rendered.
#[derive(Debug, Copy, Clone)]
pub struct Object {
    /// The orientation and location.
    pub transform: Transform,
    /// Which visual representation to use.
    pub model: ModelHandle,
    /// Which group of instances it belongs to.
    _instances_group: InstancesHandle,
}

fn install_tracing() -> EngineResult<DefaultGuard> {
    use tracing_error::ErrorLayer;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

    let fmt_layer = fmt::layer().with_target(true);
    let filter_layer = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("warn"))
        .map_err(EngineError::EnvironmentEventFilter)?;

    let subscriber = tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .with(ErrorLayer::default());

    Ok(tracing::subscriber::set_default(subscriber))
}

/// Whether an event should continue to propagate, or be consumed.
#[derive(Eq, PartialEq)]
pub(crate) enum EventPropagation {
    /// Ends the propagation, the event is seen as handled.
    Consume,
    /// Continues the propagation, the event is not seen as handled.
    Propagate,
}

fn handle_events(
    window: &Window,
    egui_context: &mut egui::Context,
    egui_state: &mut egui_winit::State,
    state: &mut State,
    last_render_time: &mut Instant,
    event: Event<()>,
    control_flow: &mut ControlFlow,
) -> EngineResult<()> {
    match event {
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            window.request_redraw();
        }
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion { delta },
            ..
        } => {
            if state.mouse_pressed {
                state.camera_controller.process_mouse(delta.0, delta.1)
            }
        }
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            // Pass the winit events to the egui platform integration.
            let egui_response = egui_state.on_event(egui_context, event);

            // Let egui and state choose whether to handle the event instead of event_loop,
            // so it can override behaviors.
            // Order of handling is egui -> state.input -> event_loop
            if !egui_response.consumed && state.input(event) == EventPropagation::Propagate {
                handle_window_event(state, control_flow, event);
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            let input = egui_state.take_egui_input(window);

            let now = Instant::now();
            let delta_time = now - *last_render_time;
            *last_render_time = now;

            state.update(delta_time);

            match state.render(window, input, egui_state, egui_context) {
                Ok(_) => {}
                // Reconfigure the surface if lost.
                Err(StateError::MissingOutputTexture(wgpu::SurfaceError::Lost)) => {
                    state.resize(state.size)
                }
                // The system is out of memory, we should probably quit.
                Err(StateError::MissingOutputTexture(wgpu::SurfaceError::OutOfMemory)) => {
                    *control_flow = ControlFlow::Exit
                }
                // `SurfaceError::Outdated` occurs when the app is minimized on Windows.
                // Silently return here to prevent spamming the console with "Outdated".
                Err(StateError::MissingOutputTexture(wgpu::SurfaceError::Outdated)) => {}
                // All other surface errors (Timeout) should be resolved by the next frame.
                Err(StateError::MissingOutputTexture(error)) => eprintln!("{error:?}"),
                // Pass on any other rendering errors.
                Err(error) => return Err(EngineError::Rendering(Box::new(error)))?,
            }
        }
        _ => {}
    };
    Ok(())
}

fn handle_window_event(state: &mut State, control_flow: &mut ControlFlow, event: &WindowEvent) {
    match event {
        WindowEvent::Resized(physical_size) => {
            state.resize(*physical_size);
        }
        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => state.resize(**new_inner_size),
        WindowEvent::CloseRequested
        | WindowEvent::KeyboardInput {
            input:
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(VirtualKeyCode::Escape),
                    ..
                },
            ..
        } => *control_flow = ControlFlow::Exit,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        assert_eq!(true, true)
    }
}
