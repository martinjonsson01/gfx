use crate::camera::Camera;
use crate::EventPropagation;
use crate::EventPropagation::{Consume, Propagate};
use cgmath::InnerSpace;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

pub struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
        }
    }

    pub(crate) fn process_events(&mut self, event: &WindowEvent) -> EventPropagation {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        Consume
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        Consume
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        Consume
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        Consume
                    }
                    VirtualKeyCode::E => {
                        self.is_up_pressed = is_pressed;
                        Consume
                    }
                    VirtualKeyCode::Q => {
                        self.is_down_pressed = is_pressed;
                        Consume
                    }
                    _ => Propagate,
                }
            }
            _ => Propagate,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        let forward = camera.target - camera.position;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.position += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.position -= forward_norm * self.speed;
        }

        if self.is_up_pressed {
            camera.position += camera.up * self.speed;
        }
        if self.is_down_pressed {
            camera.position -= camera.up * self.speed;
        }

        // Redo radius calc in case the forward/backward is pressed.
        let forward = camera.target - camera.position;
        let forward_mag = forward.magnitude();

        let right = forward_norm.cross(camera.up);

        if self.is_right_pressed {
            // Rescale the distance between the target and eye so
            // that it doesn't change. The eye therefore still
            // lies on the circle made by the target and eye.
            camera.position =
                camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.position =
                camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}
