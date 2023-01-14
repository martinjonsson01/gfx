use crossbeam_channel::{select, unbounded, Receiver, RecvError, SendError, Sender};
use derivative::Derivative;
use std::marker::PhantomData;
use thiserror::Error;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, Event, KeyboardInput, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use crate::engine::{EngineError, EngineResult};
use crate::state::State;
pub use winit::event::{ElementState, MouseButton, MouseScrollDelta, VirtualKeyCode};

/// An error that has occurred relating to window management.
#[derive(Error, Debug)]
pub enum WindowingError {
    /// Could not construct root Window.
    #[error("could not construct root Window")]
    MissingRootWindow(#[source] winit::error::OsError),
    /// Could not send window event to receivers.
    #[error("could not send window event to receivers")]
    WindowSendEvent(#[source] SendError<WindowingEvent>),
    /// Failed to render frame.
    #[error("failed to render frame")]
    Rendering(#[source] Box<EngineError>),
}

/// Whether a windowing operation failed or succeeded.
pub type WindowingResult<T, E = WindowingError> = Result<T, E>;

/// A window-related event such as the window having been resized.
#[derive(Debug, PartialEq)]
pub enum WindowingEvent {
    Resized(PhysicalSize<u32>),
    Input(InputEvent),
}

/// Information about an input that has been given by the user.
#[derive(Debug, PartialEq)]
pub enum InputEvent {
    Keyboard {
        key: VirtualKeyCode,
        state: ElementState,
    },
    Scroll(MouseScrollDelta),
    MouseClick {
        button: MouseButton,
        state: ElementState,
    },
    MouseMove {
        mouse_delta_x: f64,
        mouse_delta_y: f64,
    },
    Close,
}

/// An instruction to the windowing system about what action to perform.
#[derive(Debug, Eq, PartialEq)]
pub enum WindowingCommand {
    Quit(i32),
}

/// A facade to the OS' windowing system.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Windowing<RenderData> {
    pub(crate) window: Window,
    #[derivative(Debug = "ignore")]
    pub(crate) egui_context: egui::Context,
    #[derivative(Debug = "ignore")]
    pub(crate) egui_state: egui_winit::State,
    event_loop: EventLoop<()>,
    event_sender: Sender<WindowingEvent>,
    command_receiver: Receiver<WindowingCommand>,
    _phantom0: PhantomData<RenderData>,
}

impl<RenderData> Windowing<RenderData>
where
    for<'a> RenderData: 'a,
{
    /// Tries to create a new root `Window`.
    pub fn new() -> WindowingResult<(Self, Receiver<WindowingEvent>, Sender<WindowingCommand>)> {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .build(&event_loop)
            .map_err(WindowingError::MissingRootWindow)?;

        let egui_context = egui::Context::default();
        let mut egui_state = egui_winit::State::new(&event_loop);
        egui_state.set_pixels_per_point(window.scale_factor() as f32);

        let (event_sender, event_receiver) = unbounded();
        let (command_sender, command_receiver) = unbounded();

        let windowing = Self {
            window,
            egui_context,
            egui_state,
            event_loop,
            event_sender,
            command_receiver,
            _phantom0: PhantomData::default(),
        };
        Ok((windowing, event_receiver, command_sender))
    }

    pub(crate) fn run<UpdateFn>(
        self,
        mut render_state: State<RenderData>,
        mut update_loop: UpdateFn,
    ) -> !
    where
        for<'a> UpdateFn: FnMut(
                &mut State<RenderData>,
                &Window,
                &mut egui::Context,
                &mut egui_winit::State,
            ) -> EngineResult<()>
            + 'a,
    {
        let Windowing {
            mut window,
            mut egui_context,
            mut egui_state,
            event_loop,
            event_sender,
            command_receiver,
            _phantom0,
        } = self;
        event_loop.run(move |event, _, control_flow| {
            select! (
                recv(command_receiver) -> command => {
                    Self::handle_command(command, control_flow)
                }
                default => {
                    let result = Self::handle_event(
                        event,
                        &mut render_state,
                        &mut update_loop,
                        &mut window,
                        &mut egui_context,
                        &mut egui_state,
                        &event_sender,
                    );
                    Self::handle_error(result, control_flow);
                }
            )
        });
    }

    fn handle_command(
        command: Result<WindowingCommand, RecvError>,
        control_flow: &mut ControlFlow,
    ) {
        // Want a `match` rather than `if let` here so it errors out if a new command is ever added.
        #[allow(clippy::single_match)]
        match command {
            Ok(WindowingCommand::Quit(exit_code)) => {
                *control_flow = ControlFlow::ExitWithCode(exit_code)
            }
            _ => {}
        }
    }

    fn handle_event<UpdateFn>(
        event: Event<()>,
        render_state: &mut State<RenderData>,
        update_loop: &mut UpdateFn,
        window: &mut Window,
        egui_context: &mut egui::Context,
        egui_state: &mut egui_winit::State,
        event_sender: &Sender<WindowingEvent>,
    ) -> WindowingResult<()>
    where
        UpdateFn: FnMut(
            &mut State<RenderData>,
            &Window,
            &mut egui::Context,
            &mut egui_winit::State,
        ) -> EngineResult<()>,
    {
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
                event_sender
                    .send(WindowingEvent::Input(InputEvent::MouseMove {
                        mouse_delta_x: delta.0,
                        mouse_delta_y: delta.1,
                    }))
                    .map_err(WindowingError::WindowSendEvent)?;
            }
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                // Pass the winit events to the egui platform integration.
                let egui_response = egui_state.on_event(egui_context, &event);

                // Let egui choose whether to handle the event instead of event_loop,
                // so it can override behaviors.
                // Order of handling is egui -> event_loop
                if !egui_response.consumed {
                    Self::handle_window_event(event, event_sender)?;
                }
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                update_loop(render_state, window, egui_context, egui_state)
                    .map_err(|e| WindowingError::Rendering(Box::new(e)))?;
            }
            _ => {}
        };
        Ok(())
    }

    fn handle_window_event(
        event: WindowEvent,
        event_sender: &Sender<WindowingEvent>,
    ) -> WindowingResult<()> {
        match event {
            WindowEvent::Resized(physical_size) => event_sender
                .send(WindowingEvent::Resized(physical_size))
                .map_err(WindowingError::WindowSendEvent),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => event_sender
                .send(WindowingEvent::Resized(*new_inner_size))
                .map_err(WindowingError::WindowSendEvent),
            WindowEvent::CloseRequested => event_sender
                .send(WindowingEvent::Input(InputEvent::Close))
                .map_err(WindowingError::WindowSendEvent),
            WindowEvent::MouseInput { button, state, .. } => event_sender
                .send(WindowingEvent::Input(InputEvent::MouseClick {
                    button,
                    state,
                }))
                .map_err(WindowingError::WindowSendEvent),
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(virtual_keycode),
                        state,
                        ..
                    },
                ..
            } => event_sender
                .send(WindowingEvent::Input(InputEvent::Keyboard {
                    key: virtual_keycode,
                    state,
                }))
                .map_err(WindowingError::WindowSendEvent),
            _ => Ok(()),
        }
    }

    fn handle_error(result: WindowingResult<()>, control_flow: &mut ControlFlow) {
        if let Err(error) = result {
            let report = color_eyre::Report::new(error);
            tracing::error!("{report:?}");
            *control_flow = ControlFlow::ExitWithCode(1); // Non-zero exit code means error.
        }
    }
}
