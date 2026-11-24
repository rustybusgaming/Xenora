// SPDX-License-Identifier: GPL-3.0-or-later

//! Indexed triangle lists drawn through Metal's mesh render pipeline.
//!
//! This is the first native step of the mesh-stage work (milestone X13). An
//! authored MSL mesh function emits exactly the triangles that
//! [`NativeCompute::render_indexed`] draws with `drawIndexedPrimitives`, so
//! both paths are checked against the same independently derived images.
//! Translated RDNA 2 mesh, primitive or geometry shaders are not supported.

use std::ptr::NonNull;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLFunction,
    MTLGPUFamily, MTLLoadAction, MTLMeshRenderPipelineDescriptor, MTLPipelineOption,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineState, MTLSize,
    MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureUsage, MTLViewport,
};

use super::super::{MetalError, NativeCompute, compile_options, complete_command, find_function};
use super::{
    DEPTH_SOURCE, DepthFill, IndexBuffer, IndexedPass, MAX_PIXELS, PIXEL_FORMAT,
    validate_indexed_pass, verify_render_validation, whole,
};

/// One mesh threadgroup per triangle and one thread per corner. Each thread
/// reads its index (unpacking 16-bit indices from little-endian words) and
/// fetches vertex `index + base_vertex`, as the indexed draw does. It is
/// compiled with [`DEPTH_SOURCE`], whose fragment functions shade the pass.
const MESH_SOURCE: &str = r"
struct MeshVertex {
    float4 position [[position]];
};

struct MeshDraw {
    uint start_index;
    uint wide_indices;
    int base_vertex;
};

using MeshTriangle = metal::mesh<MeshVertex, void, 3, 1, metal::topology::triangle>;

[[mesh]] void mesh_triangle(MeshTriangle output,
                            const device float4 *positions [[buffer(0)]],
                            const device uint *indices [[buffer(1)]],
                            constant MeshDraw &draw [[buffer(2)]],
                            uint triangle [[threadgroup_position_in_grid]],
                            uint corner [[thread_index_in_threadgroup]]) {
    uint position = draw.start_index + triangle * 3u + corner;
    uint index;
    if (draw.wide_indices != 0u) {
        index = indices[position];
    } else {
        uint pair = indices[position / 2u];
        index = (position & 1u) != 0u ? pair >> 16 : pair & 0xffffu;
    }
    MeshVertex vertex;
    vertex.position = positions[int(index) + draw.base_vertex];
    output.set_vertex(corner, vertex);
    output.set_index(corner, uchar(corner));
    if (corner == 0u) {
        output.set_primitive_count(1u);
    }
}
";

/// The per-draw constants of `MeshDraw` in [`MESH_SOURCE`].
#[repr(C)]
struct MeshDrawConstants {
    start_index: u32,
    wide_indices: u32,
    base_vertex: i32,
}

impl NativeCompute {
    /// Draw an indexed pass through Metal's mesh pipeline and read back its colour.
    ///
    /// The pass and its validation are those of [`NativeCompute::render_indexed`].
    /// Each draw is one `drawMeshThreadgroups` call with one mesh threadgroup per
    /// triangle and three threads per threadgroup, without an object stage. The
    /// mesh function reads the index and vertex buffers itself and applies the
    /// draw's start index and signed base vertex. Fragments are shaded by the same
    /// functions as the indexed pass.
    ///
    /// # Errors
    ///
    /// Returns the validation errors of [`NativeCompute::render_indexed`] first.
    /// A device outside the Metal 3 GPU family then fails with a capability error
    /// (operation [`MetalError::MESH_UNAVAILABLE`]). Otherwise it fails like
    /// [`NativeCompute::render_indexed`] on unavailable BC compression or Metal
    /// setup or execution failure.
    pub fn render_mesh(&self, pass: &IndexedPass<'_>) -> Result<Vec<u8>, MetalError> {
        let [width, height] = pass.extent;
        let pixels = width
            .checked_mul(height)
            .filter(|&count| (1..=MAX_PIXELS).contains(&count))
            .ok_or_else(|| MetalError::new("render bounds", "expected 1..=4096 pixels"))?;
        validate_indexed_pass(pass)?;
        if !self.device.supportsFamily(MTLGPUFamily::Metal3) {
            return Err(MetalError::new(
                MetalError::MESH_UNAVAILABLE,
                "mesh render pipelines need a Metal 3 family device",
            ));
        }
        let bc1 = pass
            .texture
            .map(|upload| {
                let level = upload.subresource(0, 0).map_err(|error| {
                    MetalError::new("texture subresource", format!("{error:?}"))
                })?;
                self.validate_bc_extent(level.width, level.height, "BC1")?;
                self.new_bc1_texture(&level)
            })
            .transpose()?;

        autoreleasepool(|_| {
            let options = compile_options()?;
            let library = self.compile_library(&[DEPTH_SOURCE, MESH_SOURCE].concat(), &options)?;
            let mesh = find_function(&library, "mesh_triangle")?;
            let flat = self.mesh_pipeline(&mesh, &*find_function(&library, "depth_flat")?)?;
            let textured =
                self.mesh_pipeline(&mesh, &*find_function(&library, "depth_textured")?)?;
            let color = self.new_target(
                PIXEL_FORMAT,
                width,
                height,
                MTLStorageMode::Shared,
                MTLTextureUsage::RenderTarget,
            )?;
            let pass_descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
            // SAFETY: Slot zero is the only color attachment for this pass.
            let attachment = unsafe {
                pass_descriptor
                    .colorAttachments()
                    .objectAtIndexedSubscript(0)
            };
            attachment.setTexture(Some(&color));
            attachment.setLoadAction(MTLLoadAction::Clear);
            let [red, green, blue, alpha] = pass.clear_color.map(|value| f64::from(value) / 255.0);
            attachment.setClearColor(MTLClearColor {
                red,
                green,
                blue,
                alpha,
            });
            attachment.setStoreAction(MTLStoreAction::Store);
            let positions: Vec<u32> = pass
                .vertices
                .as_flattened()
                .iter()
                .map(|value| value.to_bits())
                .collect();
            let positions = self.new_buffer(&positions)?;
            let indices = self.new_buffer(&pass.indices.words())?;
            let wide_indices = u32::from(matches!(pass.indices, IndexBuffer::U32(_)));
            let command = self.queue.commandBuffer().ok_or_else(|| {
                MetalError::new("commandBuffer", "render command buffer creation failed")
            })?;
            let encoder = command
                .renderCommandEncoderWithDescriptor(&pass_descriptor)
                .ok_or_else(|| {
                    MetalError::new(
                        "renderCommandEncoderWithDescriptor",
                        "render encoder creation failed",
                    )
                })?;
            encoder.setViewport(MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: width as f64,
                height: height as f64,
                znear: 0.0,
                zfar: 1.0,
            });
            let extent = [width as f32, height as f32];
            for draw in pass.draws {
                // Validation bounds the start index by the 4096-entry index buffer.
                let constants = MeshDrawConstants {
                    start_index: draw.start_index as u32,
                    wide_indices,
                    base_vertex: draw.base_vertex,
                };
                // SAFETY: Validation proved that every index position the draw's
                // threads read lies inside the index buffer (16-bit positions read
                // word `position / 2`, which the packing pads), and that every
                // `index + base_vertex` names a float4 of the position buffer, the
                // only other memory the mesh stage reads. The constants match
                // `MeshDraw` field for field and Metal copies them, as it copies
                // the fragment constants. The texture, when bound, is a live BC1
                // texture of this device; both buffers outlive the command.
                unsafe {
                    encoder.setMeshBuffer_offset_atIndex(Some(&positions), 0, 0);
                    encoder.setMeshBuffer_offset_atIndex(Some(&indices), 0, 1);
                    encoder.setMeshBytes_length_atIndex(
                        NonNull::from(&constants).cast(),
                        size_of_val(&constants),
                        2,
                    );
                    match draw.fill {
                        DepthFill::Flat(rgba) => {
                            encoder.setRenderPipelineState(&flat);
                            let rgba = rgba.map(|value| f32::from(value) / 255.0);
                            encoder.setFragmentBytes_length_atIndex(
                                NonNull::from(&rgba).cast(),
                                size_of_val(&rgba),
                                0,
                            );
                        }
                        DepthFill::Bc1Texture => {
                            encoder.setRenderPipelineState(&textured);
                            encoder.setFragmentTexture_atIndex(bc1.as_deref(), 0);
                            encoder.setFragmentBytes_length_atIndex(
                                NonNull::from(&extent).cast(),
                                size_of_val(&extent),
                                0,
                            );
                        }
                    }
                }
                encoder.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                    MTLSize {
                        width: draw.index_count / 3,
                        height: 1,
                        depth: 1,
                    },
                    // Ignored without an object function.
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 3,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.endEncoding();
            complete_command(&command)?;

            let mut output = vec![0; pixels * 4];
            // SAFETY: GPU writes have completed, and the initialized output
            // allocation fits the entire shared texture with tightly packed rows.
            unsafe {
                color.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                    NonNull::from(output.as_mut_slice()).cast(),
                    width * 4,
                    whole(width, height),
                    0,
                );
            }
            Ok(output)
        })
    }

    fn mesh_pipeline(
        &self,
        mesh: &ProtocolObject<dyn MTLFunction>,
        fragment: &ProtocolObject<dyn MTLFunction>,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, MetalError> {
        let descriptor = MTLMeshRenderPipelineDescriptor::new();
        // SAFETY: Both functions come from the library compiled from
        // `DEPTH_SOURCE` and `MESH_SOURCE`: a mesh function and a fragment
        // function whose inputs are only the position builtin and the buffer
        // and texture the encoder binds for each draw.
        unsafe {
            descriptor.setMeshFunction(Some(mesh));
            descriptor.setFragmentFunction(Some(fragment));
        }
        // SAFETY: Slot zero is the only color attachment for this pipeline.
        unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) }
            .setPixelFormat(PIXEL_FORMAT);
        let pipeline = self
            .device
            .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| {
                MetalError::new(
                    "newRenderPipelineStateWithMeshDescriptor",
                    error.localizedDescription().to_string(),
                )
            })?;
        verify_render_validation(&pipeline)?;
        Ok(pipeline)
    }
}
