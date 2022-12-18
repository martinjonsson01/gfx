use crate::shader_locations::*;
use cgmath::{Matrix4, Quaternion, Vector3};
use std::mem::size_of;

/// The transform of an instance of a model.
pub struct Instance {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
}
impl Instance {
    pub fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (Matrix4::from_translation(self.position) * Matrix4::from(self.rotation)).into(),
        }
    }
}

/// A representation of an [`Instance`] that can be sent into shaders through a uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    model: [[f32; 4]; 4],
}
impl InstanceRaw {
    pub fn descriptor<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: INSTANCE_MODEL_MATRIX_COLUMN_0,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: INSTANCE_MODEL_MATRIX_COLUMN_1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: INSTANCE_MODEL_MATRIX_COLUMN_2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: INSTANCE_MODEL_MATRIX_COLUMN_3,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}
