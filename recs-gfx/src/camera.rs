use cgmath::{perspective, Matrix4, Point3, SquareMatrix, Vector3};

/// The coordinate system in wgpu is based on DirectX's and Metal's coordinate systems.
/// This means that in normalized device coordinates, the x- and y-axis span [-1, 1], with
/// z spanning [0, 1]. cgmath is built for OpenGL's coordinate system, so we need to transform it.
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub struct Camera {
    /// The world-space location of the camera.
    pub position: Point3<f32>,
    /// The world-space location of where the camera is looking.
    pub target: Point3<f32>,
    /// Which direction in world-space is up (normalized).
    pub up: Vector3<f32>,
    /// The aspect ratio of the viewport.
    pub aspect: f32,
    /// The vertical field of view of the viewport.
    pub fovy: f32,
    /// The near clipping plane.
    pub znear: f32,
    /// The far clipping plane.
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> Matrix4<f32> {
        let view = Matrix4::look_at_rh(self.position, self.target, self.up);
        let proj = perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        // See documentation for OPENGL_TO_WGPU_MATRIX.
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

/// A representation of the [`Camera`] that can be sent into shaders through a uniform buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
        }
    }

    pub fn update_view_projection(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}
