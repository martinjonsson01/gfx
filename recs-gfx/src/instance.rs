use crate::shader_locations::*;
use crate::state::ModelHandle;
use cgmath::{Matrix3, Matrix4, Quaternion, Vector3};
use std::mem::size_of;
use std::ops::Range;
use wgpu::util::DeviceExt;

/// A set of instances of a specific model.
///
/// If new instances are added, [`instance_buffer`] and [`camera_bind_group`] needs to be recreated,
/// otherwise the new instances won't show up correctly.
#[derive(Debug)]
pub(crate) struct ModelInstances {
    pub model: ModelHandle,
    transforms: Vec<Transform>,
    buffer: wgpu::Buffer,
}

impl ModelInstances {
    pub fn new(device: &wgpu::Device, model: ModelHandle, transforms: Vec<Transform>) -> Self {
        let transform_data = transforms.iter().map(Transform::to_raw).collect::<Vec<_>>();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Transform Buffer"),
            contents: bytemuck::cast_slice(&transform_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            model,
            transforms,
            buffer,
        }
    }

    /// Which instance data to use.
    pub fn buffer_slice(&self) -> wgpu::BufferSlice {
        self.buffer.slice(..)
    }

    /// Which instances to render.
    pub fn instances(&self) -> Range<u32> {
        0..self.transforms.len() as u32
    }
}

/// Describes the placement and orientation of an object in the world.
#[derive(Debug)]
pub struct Transform {
    /// The 3D translation of the object.
    pub position: Vector3<f32>,
    /// The orientation of the object.
    pub rotation: Quaternion<f32>,
}
impl Transform {
    pub(crate) fn to_raw(&self) -> TransformRaw {
        let model = Matrix4::from_translation(self.position) * Matrix4::from(self.rotation);
        TransformRaw {
            model: model.into(),
            normal: Matrix3::from(self.rotation).into(),
        }
    }
}

/// A representation of an [`Transform`] that can be sent into shaders through a uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TransformRaw {
    model: [[f32; 4]; 4],
    /// Can't use model matrix to transform normals, as we only want to rotate them.
    normal: [[f32; 3]; 3],
}
impl TransformRaw {
    pub fn descriptor<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<TransformRaw>() as wgpu::BufferAddress,
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
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: INSTANCE_NORMAL_MATRIX_COLUMN_0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: INSTANCE_NORMAL_MATRIX_COLUMN_1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: INSTANCE_NORMAL_MATRIX_COLUMN_2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}
