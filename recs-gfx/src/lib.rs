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
    clippy::unwrap_used
)]

mod camera;
mod instance;
mod model;
mod resources;
mod state;
mod texture;

use crate::camera::{Camera, Projection};
use crate::state::{State, StateError};
use cgmath::{Matrix4, SquareMatrix};
use color_eyre::Report;
use std::time::Instant;
use thiserror::Error;
use tracing::dispatcher::DefaultGuard;
use tracing::{error, instrument};
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
    StateConstruction(#[source] StateError),
    /// Failed to render frame.
    #[error("failed to render frame")]
    Rendering(#[source] StateError),
    /// Could not create event filter from environment variable.
    #[error("could not create event filter from environment variable")]
    EnvironmentEventFilter(#[source] ParseError),
}

type Result<T, E = EngineError> = std::result::Result<T, E>;

/// Starts the graphics engine, opening a new window and rendering to it.
#[instrument]
pub async fn run() -> Result<()> {
    let _guard = install_tracing()?;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .map_err(EngineError::MissingRootWindow)?;

    let mut state = State::new(&window)
        .await
        .map_err(EngineError::StateConstruction)?;
    let mut last_render_time = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let result = handle_events(
            &window,
            &mut state,
            &mut last_render_time,
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

fn install_tracing() -> Result<DefaultGuard> {
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
    state: &mut State,
    last_render_time: &mut Instant,
    event: Event<()>,
    control_flow: &mut ControlFlow,
) -> Result<()> {
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
            // Let state choose whether to handle the event instead of event_loop,
            // so it can override behaviors.
            if state.input(event) == EventPropagation::Propagate {
                handle_window_event(state, control_flow, event);
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            let now = Instant::now();
            let delta_time = now - *last_render_time;
            *last_render_time = now;

            state.update(delta_time);

            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost.
                Err(StateError::MissingOutputTexture(wgpu::SurfaceError::Lost)) => {
                    state.resize(state.size)
                }
                // The system is out of memory, we should probably quit.
                Err(StateError::MissingOutputTexture(wgpu::SurfaceError::OutOfMemory)) => {
                    *control_flow = ControlFlow::Exit
                }
                // All other surface errors (Outdated, Timeout) should be resolved by the next frame.
                Err(StateError::MissingOutputTexture(error)) => eprintln!("{error:?}"),
                // Pass on any other rendering errors.
                Err(error) => return Err(EngineError::Rendering(error))?,
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
