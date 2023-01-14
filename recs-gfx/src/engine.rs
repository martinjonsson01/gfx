//! The core managing module of the engine, responsible for high-level startup and error-handling.

use crate::camera::CameraController;
use crate::state::{ModelHandle, State, StateError};
use crate::time::{Time, UpdateRate};
use crate::window::{InputEvent, Windowing, WindowingCommand, WindowingError, WindowingEvent};
use crate::{Object, Transform};
use crossbeam_channel::{unbounded, Receiver, SendError, Sender};
use crossbeam_queue::ArrayQueue;
pub use ring_channel::RingSender;
use ring_channel::{ring_channel, RingReceiver};
use std::error::Error;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::thread;
use std::thread::ScopedJoinHandle;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{error, info, info_span, instrument, span, Level};
use winit::window::Window;

/// An error that has occurred within the graphics engine.
#[derive(Error, Debug)]
pub enum EngineError {
    /// Could not instantiate State.
    #[error("could not instantiate State")]
    StateConstruction(#[source] Box<StateError>),
    /// Could not create a window.
    #[error("could not create a window")]
    WindowCreation(#[source] WindowingError),
    /// Failed to render frame.
    #[error("failed to render frame")]
    Rendering(#[source] Box<StateError>),
    /// Could not load model.
    #[error("could not load model `{1}`")]
    ModelLoad(#[source] Box<StateError>, PathBuf),
    /// Could not create a new object.
    #[error("a new object could not be created")]
    ObjectCreation(#[source] Box<StateError>),
    /// Could not create a single new object.
    #[error("a single new object could not be created")]
    SingleObjectCreation(),
    /// The simulation thread panicked.
    #[error("the simulation thread panicked with error `{0}`")]
    SimulationThreadPanic(String),
    /// The rendering thread has closed, causing the simulation thread to not be able to send it data.
    #[error("the rendering thread has closed")]
    RenderingThreadClosed(#[source] SendError<Duration>),
    /// The rendering thread failed to receive rendering data from the simulation thread.
    #[error("the rendering thread failed to receive rendering data from the simulation thread")]
    RenderDataReceipt(#[source] ring_channel::RecvError),
    /// An error occurred during graphics initialization by the client.
    #[error("an error occurred during graphics initialization by the client")]
    RenderInitialization(#[source] Box<dyn Error + Send + Sync>),
    /// A critical error has occurred and the application should shut down as soon as possible.
    #[error(
        "a critical error has occurred and the application should shut down as soon as possible"
    )]
    Critical(#[source] StateError),
    /// Could not send command to windowing system.
    #[error("could not send command to windowing system")]
    EngineSendCommand(#[source] SendError<WindowingCommand>),
}

/// Whether an engine operation failed or succeeded.
pub type EngineResult<T, E = EngineError> = Result<T, E>;

const AVERAGE_FPS_SAMPLES: usize = 128;

/// The driving actor of all windowing, rendering and simulation.
#[derive(Debug)]
pub struct Engine<GfxInitFn, SimulationFn, ClientContext, RenderData> {
    initialize_gfx: GfxInitFn,
    simulate: SimulationFn,
    client_context: ClientContext,
    /// Used to pass data about what to render from the simulation thread to the render thread.
    render_data_sender: RingSender<RenderData>,
    /// Used to receive data about what to render.
    render_data_receiver: RingReceiver<RenderData>,
    /// A controller for moving the camera around based on user input.
    camera_controller: CameraController,
    /// The window management wrapper.
    windowing: Windowing<RenderData>,
    /// Used to listen to window events.
    window_event_receiver: Receiver<WindowingEvent>,
    /// Used to send commands to the windowing system.
    window_command_sender: Sender<WindowingCommand>,
    /// The current state of the renderer.
    render_state: State<RenderData>,
    /// The current time of the engine.
    time: Time,
}

const CAMERA_SPEED: f32 = 7.0;
const CAMERA_SENSITIVITY: f32 = 1.0;

/// A buffer that contains simulation results that need to be rendered.
pub type SimulationBuffer<T> = ArrayQueue<T>;

impl<GfxInitFn, SimulationFn, ClientContext, RenderData>
    Engine<GfxInitFn, SimulationFn, ClientContext, RenderData>
where
    for<'a> RenderData: IntoIterator<Item = Object> + Send + 'a,
    for<'a> ClientContext: Send + 'a,
    for<'a> GfxInitFn:
        Fn(&mut ClientContext, &mut dyn Creator) -> Result<(), Box<dyn Error + Send + Sync>> + 'a,
    for<'a> SimulationFn: FnMut(
            &mut ClientContext,
            &UpdateRate,
            &mut RingSender<RenderData>,
        ) -> Result<(), Box<dyn Error + Send + Sync>>
        + Send
        + Sync
        + 'a,
{
    /// Creates a new instance of `Engine`.
    ///
    /// The `initialize_gfx` function is called once, before beginning the render loop.
    /// The `simulate` function is called from the simulation thread every simulation tick.
    /// The `client_context` is passed to both the renderer and simulator to store data in.
    pub fn new(
        initialize_gfx: GfxInitFn,
        simulate: SimulationFn,
        client_context: ClientContext,
    ) -> EngineResult<Self> {
        let render_data_channel_capacity = NonZeroUsize::new(1).expect("1 is non-zero");
        let (render_data_sender, render_data_receiver) = ring_channel(render_data_channel_capacity);

        let (windowing, window_event_receiver, window_command_sender) =
            Windowing::new().map_err(EngineError::WindowCreation)?;
        let render_state = State::new(&windowing.window)
            .map_err(|e| EngineError::StateConstruction(Box::new(e)))?;

        Ok(Self {
            initialize_gfx,
            simulate,
            client_context,
            render_data_sender,
            render_data_receiver,
            camera_controller: CameraController::new(CAMERA_SPEED, CAMERA_SENSITIVITY),
            windowing,
            window_event_receiver,
            window_command_sender,
            render_state,
            time: Time::new(Instant::now(), AVERAGE_FPS_SAMPLES),
        })
    }

    /// Initializes and starts all state and threads, beginning the core event-loops of the program.
    pub fn start(mut self) -> EngineResult<()> {
        (self.initialize_gfx)(&mut self.client_context, &mut self.render_state)
            .map_err(EngineError::RenderInitialization)?;

        let (simulation_rate_sender, simulation_rate_receiver) = unbounded();

        thread::scope(|scope| {
            let _simulation_thread: ScopedJoinHandle<EngineResult<()>> =
                scope.spawn(move || loop {
                    let mut simulation_rate = UpdateRate::new(Instant::now(), AVERAGE_FPS_SAMPLES);
                    loop {
                        simulation_rate.update_time(Instant::now());
                        simulation_rate_sender
                            .send(simulation_rate.delta_time)
                            .map_err(EngineError::RenderingThreadClosed)
                            .expect("rendering thread should be alive");

                        // todo: handle error by sending to main thread and letting it decide
                        info_span!("sim").in_scope(|| {
                            (self.simulate)(
                                &mut self.client_context,
                                &simulation_rate,
                                &mut self.render_data_sender,
                            )
                            .expect("todo");
                        });
                    }
                });

            self.windowing.run(
                self.render_state,
                move |state, window, egui_context, egui_state| {
                    Self::main_loop(
                        &mut self.time,
                        &self.window_event_receiver,
                        &mut self.window_command_sender,
                        &simulation_rate_receiver,
                        &mut self.camera_controller,
                        &mut self.render_data_receiver,
                        state,
                    )?;
                    Self::render_loop(state, window, egui_context, egui_state)
                },
            )
        })
    }

    fn main_loop(
        time: &mut Time,
        window_event_receiver: &Receiver<WindowingEvent>,
        window_command_sender: &mut Sender<WindowingCommand>,
        simulation_rate_receiver: &Receiver<Duration>,
        camera_controller: &mut CameraController,
        render_data_receiver: &mut RingReceiver<RenderData>,
        state: &mut State<RenderData>,
    ) -> EngineResult<()> {
        time.render.update_time(Instant::now());
        time.simulation
            .update_from_delta_samples(simulation_rate_receiver.try_iter());

        for event in window_event_receiver.try_iter() {
            match event {
                WindowingEvent::Input(InputEvent::Close) => {
                    // todo: graceful shutdown of all threads
                    window_command_sender
                        .send(WindowingCommand::Quit(0))
                        .map_err(EngineError::EngineSendCommand)?;
                }
                WindowingEvent::Input(input) => {
                    camera_controller.input(&input);
                }
                WindowingEvent::Resized(new_size) => {
                    state.resize(new_size);
                }
            }
        }

        let render_rate = &time.render;
        info!("gfx {render_rate}");

        let simulation_rate = &time.simulation;
        info!("sim {simulation_rate}");

        if let Ok(render_data) = render_data_receiver.try_recv() {
            state.update(&time.render, render_data, |camera, update_rate| {
                camera_controller.update_camera(camera, update_rate.delta_time);
            });
        }

        Ok(())
    }

    fn render_loop(
        state: &mut State<RenderData>,
        window: &Window,
        egui_context: &mut egui::Context,
        egui_state: &mut egui_winit::State,
    ) -> EngineResult<()> {
        let span = span!(Level::INFO, "gfx");
        let _enter = span.enter();
        match state.render(window, egui_state, egui_context) {
            Ok(_) => Ok(()),
            // Reconfigure the surface if lost.
            Err(StateError::MissingOutputTexture(wgpu::SurfaceError::Lost)) => {
                // Resizing to same size effectively recreates the surface.
                state.resize(window.inner_size());
                Ok(())
            }
            // The system is out of memory, we should probably quit.
            Err(error)
                if matches!(
                    error,
                    StateError::MissingOutputTexture(wgpu::SurfaceError::OutOfMemory)
                ) =>
            {
                Err(EngineError::Critical(error))
            }
            // `SurfaceError::Outdated` occurs when the app is minimized on Windows.
            // Silently return here to prevent spamming the console with "Outdated".
            Err(StateError::MissingOutputTexture(wgpu::SurfaceError::Outdated)) => Ok(()),
            // All other surface errors (Timeout) should be resolved by the next frame.
            Err(StateError::MissingOutputTexture(error)) => {
                error!("{error:?}");
                Ok(())
            }
            // Pass on any other rendering errors.
            Err(error) => Err(EngineError::Rendering(Box::new(error))),
        }
    }
}

/// A way of creating objects in the renderer.
pub trait Creator {
    /// Creates multiple objects with the same model in the world.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::error::Error;
    /// use cgmath::{One, Quaternion, Vector3, Zero};
    /// use crossbeam_queue::ArrayQueue;
    /// use recs_gfx::{Object, RenderContext, Transform};
    /// use std::rc::Rc;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let queue = Rc::new(ArrayQueue::new(1));
    /// let mut graphics_engine = RenderContext::new(queue.clone())?;
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
    fn create_objects(
        &mut self,
        model: ModelHandle,
        transforms: Vec<Transform>,
    ) -> EngineResult<Vec<Object>>;
    /// Creates an object in the world.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::error::Error;
    /// use cgmath::{Quaternion, Vector3, Zero};
    /// use crossbeam_queue::ArrayQueue;
    /// use recs_gfx::{RenderContext, Transform};
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use std::rc::Rc;
    /// let queue = Rc::new(ArrayQueue::new(1));
    /// let mut graphics_engine = RenderContext::new(queue.clone())?;
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
    fn create_object(&mut self, model: ModelHandle, transform: Transform) -> EngineResult<Object>;
    /// Loads a model into the engine.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::error::Error;
    /// use crossbeam_queue::ArrayQueue;
    /// use recs_gfx::RenderContext;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use std::rc::Rc;
    /// let queue = Rc::new(ArrayQueue::new(1));
    /// let mut graphics_engine = RenderContext::new(queue.clone())?;
    ///
    /// let model_path = std::path::Path::new("path/to/model.obj");
    /// let model_handle = graphics_engine.load_model(model_path)?;
    /// #   Ok(())
    /// # }
    /// ```
    fn load_model(&mut self, path: &Path) -> EngineResult<ModelHandle>;
}

impl<RenderData> Creator for State<RenderData> {
    /// Creates multiple objects with the same model in the world.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::error::Error;
    /// use cgmath::{One, Quaternion, Vector3, Zero};
    /// use crossbeam_queue::ArrayQueue;
    /// use recs_gfx::{Object, RenderContext, Transform};
    /// use std::rc::Rc;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let queue = Rc::new(ArrayQueue::new(1));
    /// let mut graphics_engine = RenderContext::new(queue.clone())?;
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
    fn create_objects(
        &mut self,
        model: ModelHandle,
        transforms: Vec<Transform>,
    ) -> EngineResult<Vec<Object>> {
        let instances_group = self
            .create_model_instances(model, transforms.clone())
            .map_err(|e| EngineError::ObjectCreation(Box::new(e)))?;
        Ok(transforms
            .into_iter()
            .map(|transform| Object {
                transform,
                model,
                instances_group,
            })
            .collect())
    }

    /// Creates an object in the world.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::error::Error;
    /// use cgmath::{Quaternion, Vector3, Zero};
    /// use crossbeam_queue::ArrayQueue;
    /// use recs_gfx::{RenderContext, Transform};
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use std::rc::Rc;
    /// let queue = Rc::new(ArrayQueue::new(1));
    /// let mut graphics_engine = RenderContext::new(queue.clone())?;
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
    fn create_object(&mut self, model: ModelHandle, transform: Transform) -> EngineResult<Object> {
        self.create_objects(model, vec![transform])?
            .pop()
            .ok_or_else(EngineError::SingleObjectCreation)
    }

    /// Loads a model into the engine.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::error::Error;
    /// use crossbeam_queue::ArrayQueue;
    /// use recs_gfx::RenderContext;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use std::rc::Rc;
    /// let queue = Rc::new(ArrayQueue::new(1));
    /// let mut graphics_engine = RenderContext::new(queue.clone())?;
    ///
    /// let model_path = std::path::Path::new("path/to/model.obj");
    /// let model_handle = graphics_engine.load_model(model_path)?;
    /// #   Ok(())
    /// # }
    /// ```
    #[instrument(skip(self))]
    fn load_model(&mut self, path: &Path) -> EngineResult<ModelHandle> {
        self.load_model(path)
            .map_err(|e| EngineError::ModelLoad(Box::new(e), path.to_owned()))
    }
}
