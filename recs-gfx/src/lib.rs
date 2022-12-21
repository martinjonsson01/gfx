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
use crate::state::State;
use cgmath::{Matrix4, SquareMatrix};
use std::time::Instant;
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

/// Starts the graphics engine, opening a new window and rendering to it.
pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .expect("should be able to build a window");

    let mut state = State::new(&window).await;
    let mut last_render_time = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        handle_events(
            &window,
            &mut state,
            &mut last_render_time,
            event,
            control_flow,
        )
    });
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
) {
    // Use polling to run
    //*control_flow = ControlFlow::Poll;
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
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // The system is out of memory, we should probably quit.
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame.
                Err(error) => eprintln!("{error:?}"),
            }
        }
        _ => {}
    }
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
