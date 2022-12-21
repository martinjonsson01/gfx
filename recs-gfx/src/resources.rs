use crate::model::{Material, Mesh, Model, ModelVertex};
use crate::texture::Texture;
use cgmath::{Vector2, Vector3};
use std::io::{BufReader, Cursor};
use wgpu::util::DeviceExt;

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    let path = std::path::Path::new(env!("OUT_DIR"))
        .join("res")
        .join(file_name);
    let txt = std::fs::read_to_string(path)?;
    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    let path = std::path::Path::new(env!("OUT_DIR"))
        .join("res")
        .join(file_name);
    let data = std::fs::read(path)?;
    Ok(data)
}

pub async fn load_texture(
    file_name: &str,
    is_normal_map: bool,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<Texture> {
    let data = load_binary(file_name).await?;
    Texture::from_bytes(device, queue, &data, file_name, is_normal_map)
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |material_name| async move {
            let mat_text = load_string(&material_name)
                .await
                .expect("materials for cube are always present due to build script");
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        let diffuse_texture = load_texture(&m.diffuse_texture, false, device, queue).await?;
        let normal_texture = load_texture(&m.normal_texture, true, device, queue).await?;

        materials.push(Material::new(
            device,
            &m.name,
            diffuse_texture,
            normal_texture,
            layout,
        ));
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let mut vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                    // These are calculate later.
                    tangent: [0.0; 3],
                    bitangent: [0.0; 3],
                })
                .collect::<Vec<_>>();

            calculate_tangents_bitangents(&m, &mut vertices);

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{file_name:?} Vertex Buffer")),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{file_name:?} Index Buffer")),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material_index: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    Ok(Model { meshes, materials })
}

fn calculate_tangents_bitangents(m: &tobj::Model, vertices: &mut Vec<ModelVertex>) {
    let indices = &m.mesh.indices;
    let mut triangles_included = vec![0; vertices.len()];

    // Calculate tangents and bitangents.
    for triangle in indices.chunks(3) {
        let v0 = vertices[triangle[0] as usize];
        let v1 = vertices[triangle[1] as usize];
        let v2 = vertices[triangle[2] as usize];

        let pos0: Vector3<_> = v0.position.into();
        let pos1: Vector3<_> = v1.position.into();
        let pos2: Vector3<_> = v2.position.into();

        let uv0: Vector2<_> = v0.tex_coords.into();
        let uv1: Vector2<_> = v1.tex_coords.into();
        let uv2: Vector2<_> = v2.tex_coords.into();

        let edge1 = pos1 - pos0;
        let edge2 = pos2 - pos0;

        // This will give us a direction to calculate the tangent and bitangent.
        let delta_uv1 = uv1 - uv0;
        let delta_uv2 = uv2 - uv0;

        // Solving the following system of equations will give us the tangent and bitangent:
        //     edge1 = delta_uv1.x * T + delta_u.y * B
        //     edge2 = delta_uv2.x * T + delta_uv2.y * B
        let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
        let tangent = (edge1 * delta_uv2.y - edge2 * delta_uv1.y) * r;
        // We flip the bitangent to enable right-handed normal
        // maps with wgpu texture coordinate system
        let bitangent = (edge2 * delta_uv1.x - edge1 * delta_uv2.x) * -r;

        // We'll use the same tangent/bitangent for each vertex in the triangle
        for vertex in 0..3 {
            vertices[triangle[vertex] as usize].tangent =
                (tangent + Vector3::from(vertices[triangle[vertex] as usize].tangent)).into();
            vertices[triangle[vertex] as usize].bitangent =
                (bitangent + Vector3::from(vertices[triangle[vertex] as usize].bitangent)).into();
            // Used to average the tangents/bitangents
            triangles_included[triangle[vertex] as usize] += 1;
        }
    }

    // Average the tangents/bitangents
    for (i, n) in triangles_included.into_iter().enumerate() {
        let denom = 1.0 / n as f32;
        let mut v = &mut vertices[i];
        v.tangent = (cgmath::Vector3::from(v.tangent) * denom).into();
        v.bitangent = (cgmath::Vector3::from(v.bitangent) * denom).into();
    }
}
