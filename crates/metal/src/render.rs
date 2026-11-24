// SPDX-License-Identifier: GPL-3.0-or-later

use std::ptr::NonNull;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::sel;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLClearColor, MTLColorWriteMask, MTLCommandBuffer,
    MTLCommandEncoder, MTLCommandQueue, MTLCompareFunction, MTLDepthStencilDescriptor,
    MTLDepthStencilState, MTLDevice, MTLFunction, MTLIndexType, MTLLoadAction, MTLOrigin,
    MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLShaderValidation, MTLSize,
    MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureUsage, MTLViewport,
};

use xenora_texture::{Bc1Subresource, Bc1Upload};

use super::{
    ComputeInput, MetalError, NativeCompute, compile_options, complete_command, find_function,
    validate_source,
};

// Indexed passes drawn through the mesh pipeline; shares this module's helpers.
#[path = "mesh.rs"]
mod mesh;

const MAX_PIXELS: usize = 4096;
const PIXEL_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA8Unorm;

impl NativeCompute {
    /// Draw one triangle with trusted MSL into a cleared RGBA8Unorm target.
    ///
    /// The entries are ordered vertex, then fragment. Vertex slot zero holds
    /// three clip-space float4 positions. Vertex slot one holds three float4
    /// RGBA colors. The stages must restrict accesses to these inputs and
    /// the single color output. Use only trusted MSL in a process with a wall
    /// deadline. The returned bytes are tightly packed in top-to-bottom rows.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite inputs, invalid source or extent, or
    /// Metal compilation, setup or completion failure.
    pub fn render_rgba8(
        &self,
        source: &str,
        entries: [&str; 2],
        vertices: &[[f32; 4]; 3],
        colors: &[[f32; 4]; 3],
        [width, height]: [usize; 2],
        clear: [u8; 4],
    ) -> Result<Vec<u8>, MetalError> {
        self.render_rgba8_sampled(
            source,
            entries,
            vertices,
            colors,
            [width, height],
            clear,
            None,
        )
    }

    /// Draw one offscreen triangle and return readback and GPU command time in seconds.
    /// Timing is absent if Metal supplies no usable command timestamps.
    pub fn render_rgba8_profiled(
        &self,
        source: &str,
        entries: [&str; 2],
        vertices: &[[f32; 4]; 3],
        colors: &[[f32; 4]; 3],
        extent: [usize; 2],
        clear: [u8; 4],
    ) -> Result<(Vec<u8>, Option<f64>), MetalError> {
        self.render_rgba8_internal(source, entries, vertices, colors, extent, clear, None)
    }

    /// Draw one triangle with trusted MSL and optional bound sampled texture into a cleared RGBA8Unorm target.
    ///
    /// When `sampled` is provided, the texture is bound to fragment texture slot 0 and
    /// the sampler to fragment sampler slot 0.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite inputs, invalid source or extent, or
    /// Metal compilation, setup or completion failure.
    #[allow(clippy::too_many_arguments)]
    pub fn render_rgba8_sampled(
        &self,
        source: &str,
        entries: [&str; 2],
        vertices: &[[f32; 4]; 3],
        colors: &[[f32; 4]; 3],
        [width, height]: [usize; 2],
        clear: [u8; 4],
        sampled: Option<(
            &crate::texture_pipeline::MetalRgba8Texture,
            &crate::sampler::MetalSampler,
        )>,
    ) -> Result<Vec<u8>, MetalError> {
        self.render_rgba8_internal(
            source,
            entries,
            vertices,
            colors,
            [width, height],
            clear,
            sampled,
        )
        .map(|(pixels, _)| pixels)
    }

    #[allow(clippy::too_many_arguments)]
    fn render_rgba8_internal(
        &self,
        source: &str,
        entries: [&str; 2],
        vertices: &[[f32; 4]; 3],
        colors: &[[f32; 4]; 3],
        [width, height]: [usize; 2],
        clear: [u8; 4],
        sampled: Option<(
            &crate::texture_pipeline::MetalRgba8Texture,
            &crate::sampler::MetalSampler,
        )>,
    ) -> Result<(Vec<u8>, Option<f64>), MetalError> {
        let pixels = width
            .checked_mul(height)
            .filter(|&count| (1..=MAX_PIXELS).contains(&count))
            .ok_or_else(|| MetalError::new("render bounds", "expected 1..=4096 pixels"))?;
        validate_source(source, &entries)?;
        if vertices
            .iter()
            .chain(colors)
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(MetalError::new(
                "render input",
                "expected finite vertex data",
            ));
        }

        autoreleasepool(|_| {
            let options = compile_options()?;
            let library = self.compile_library(source, &options)?;
            let vertex = find_function(&library, entries[0])?;
            let fragment = find_function(&library, entries[1])?;
            let descriptor = MTLRenderPipelineDescriptor::new();
            descriptor.setVertexFunction(Some(&vertex));
            descriptor.setFragmentFunction(Some(&fragment));
            // SAFETY: Slot zero is the only color attachment for this pipeline.
            unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) }
                .setPixelFormat(PIXEL_FORMAT);
            let pipeline = self
                .device
                .newRenderPipelineStateWithDescriptor_error(&descriptor)
                .map_err(|error| {
                    MetalError::new(
                        "newRenderPipelineStateWithDescriptor",
                        error.localizedDescription().to_string(),
                    )
                })?;
            verify_render_validation(&pipeline)?;

            // SAFETY: The nonzero dimensions are bounded by MAX_PIXELS, and
            // RGBA8Unorm is a renderable format on the selected Apple family.
            let texture_descriptor = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    PIXEL_FORMAT,
                    width,
                    height,
                    false,
                )
            };
            texture_descriptor.setStorageMode(MTLStorageMode::Shared);
            texture_descriptor.setUsage(MTLTextureUsage::RenderTarget);
            let texture = self
                .device
                .newTextureWithDescriptor(&texture_descriptor)
                .ok_or_else(|| {
                    MetalError::new(
                        "newTextureWithDescriptor",
                        "render target allocation failed",
                    )
                })?;
            let pass = MTLRenderPassDescriptor::renderPassDescriptor();
            // SAFETY: Slot zero is the only color attachment for this pass.
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            attachment.setTexture(Some(&texture));
            attachment.setLoadAction(MTLLoadAction::Clear);
            let [red, green, blue, alpha] = clear.map(|value| f64::from(value) / 255.0);
            attachment.setClearColor(MTLClearColor {
                red,
                green,
                blue,
                alpha,
            });
            attachment.setStoreAction(MTLStoreAction::Store);

            let positions = self.new_buffer(
                vertices
                    .map(|vertex| vertex.map(f32::to_bits))
                    .as_flattened(),
            )?;
            let colors =
                self.new_buffer(colors.map(|color| color.map(f32::to_bits)).as_flattened())?;
            let command = self.queue.commandBuffer().ok_or_else(|| {
                MetalError::new("commandBuffer", "render command buffer creation failed")
            })?;
            let encoder = command
                .renderCommandEncoderWithDescriptor(&pass)
                .ok_or_else(|| {
                    MetalError::new(
                        "renderCommandEncoderWithDescriptor",
                        "render encoder creation failed",
                    )
                })?;
            encoder.setRenderPipelineState(&pipeline);
            encoder.setViewport(MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: width as f64,
                height: height as f64,
                znear: 0.0,
                zfar: 1.0,
            });
            // SAFETY: Both buffers come from this device and each holds three
            // float4s. The trusted vertex entry may read only these records.
            // The render target format matches that of the pipeline.
            unsafe {
                encoder.setVertexBuffer_offset_atIndex(Some(&positions), 0, 0);
                encoder.setVertexBuffer_offset_atIndex(Some(&colors), 0, 1);
                if let Some((tex, samp)) = sampled {
                    encoder.setFragmentTexture_atIndex(Some(tex.raw()), 0);
                    encoder.setFragmentSamplerState_atIndex(Some(samp.raw()), 0);
                }
                encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            }
            encoder.endEncoding();
            complete_command(&command)?;

            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width,
                    height,
                    depth: 1,
                },
            };
            let mut output = vec![0; pixels * 4];
            // SAFETY: GPU writes have completed, and the initialized output
            // allocation fits the entire shared texture with tightly packed rows.
            unsafe {
                texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                    NonNull::from(output.as_mut_slice()).cast(),
                    width * 4,
                    region,
                    0,
                );
            }
            let gpu_start = command.GPUStartTime();
            let gpu_seconds = command.GPUEndTime() - gpu_start;
            let gpu_time = (gpu_start > 0.0 && gpu_seconds.is_finite() && gpu_seconds > 0.0)
                .then_some(gpu_seconds);
            Ok((output, gpu_time))
        })
    }
}

fn verify_render_validation(
    pipeline: &ProtocolObject<dyn MTLRenderPipelineState>,
) -> Result<(), MetalError> {
    if std::env::var("MTL_SHADER_VALIDATION").as_deref() != Ok("1") {
        return Ok(());
    }
    if !pipeline.respondsToSelector(sel!(shaderValidation)) {
        return Err(MetalError::new(
            "shaderValidation",
            "the runtime lacks the render pipeline state query",
        ));
    }
    // SAFETY: The selector exists and the newly created pipeline is private
    // to this thread. Its non-atomic property is read before encoding.
    if unsafe { pipeline.shaderValidation() } != MTLShaderValidation::Enabled {
        return Err(MetalError::new(
            "shaderValidation",
            "shader validation requested but inactive for the render pipeline",
        ));
    }
    Ok(())
}

const DEPTH_FORMAT: MTLPixelFormat = MTLPixelFormat::Depth32Float;
const MAX_DEPTH_DRAWS: usize = 16;

const DEPTH_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;

vertex float4 depth_vertex(uint id [[vertex_id]],
                           const device float4 *positions [[buffer(0)]]) {
    return positions[id];
}

fragment float4 depth_flat(constant float4 &color [[buffer(0)]]) {
    return color;
}

fragment float4 depth_textured(float4 position [[position]],
                               texture2d<float> image [[texture(0)]],
                               constant float2 &extent [[buffer(0)]]) {
    constexpr sampler nearest(coord::normalized, filter::nearest, address::clamp_to_edge);
    return image.sample(nearest, position.xy / extent);
}
";

const DEPTH_READBACK: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void read_depth(depth2d<float, access::read> depth [[texture(0)]],
                       device uint *output [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    uint width = depth.get_width();
    output[index] = as_type<uint>(depth.read(uint2(index % width, index / width)));
}
";

/// Comparison of a fragment's depth against the stored depth; it passes when
/// `fragment <compare> stored` holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DepthCompare {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl DepthCompare {
    fn metal(self) -> MTLCompareFunction {
        match self {
            Self::Never => MTLCompareFunction::Never,
            Self::Less => MTLCompareFunction::Less,
            Self::Equal => MTLCompareFunction::Equal,
            Self::LessEqual => MTLCompareFunction::LessEqual,
            Self::Greater => MTLCompareFunction::Greater,
            Self::NotEqual => MTLCompareFunction::NotEqual,
            Self::GreaterEqual => MTLCompareFunction::GreaterEqual,
            Self::Always => MTLCompareFunction::Always,
        }
    }
}

/// Depth test and write enable of one draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DepthState {
    pub compare: DepthCompare,
    pub write: bool,
}

impl DepthState {
    /// No depth test and no depth write: the only state allowed without a
    /// depth attachment.
    pub const DISABLED: Self = Self {
        compare: DepthCompare::Always,
        write: false,
    };
}

/// Colour source of one depth-pass draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DepthFill {
    /// A constant RGBA8 colour.
    Flat([u8; 4]),
    /// The pass's BC1 texture, sampled with nearest filtering at the pixel centre.
    Bc1Texture,
}

/// One triangle of a depth pass: three clip-space positions, colour and depth state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthDraw {
    pub positions: [[f32; 4]; 3],
    pub fill: DepthFill,
    pub depth: DepthState,
}

/// An offscreen pass of up to 16 triangles drawn in order into a cleared
/// RGBA8Unorm target with viewport depth range 0..1.
///
/// With `clear_depth`, the pass has a `Depth32Float` attachment cleared to that
/// value and stored after the pass. Without it, the pass has no depth
/// attachment and every draw must use [`DepthState::DISABLED`].
#[derive(Clone, Copy, Debug)]
pub struct DepthPass<'a> {
    pub extent: [usize; 2],
    pub clear_color: [u8; 4],
    pub clear_depth: Option<f32>,
    pub draws: &'a [DepthDraw],
    /// A validated BC1 upload whose base level of layer 0 is uploaded to a
    /// native `BC1_RGBA` texture as the original blocks, without CPU decoding.
    pub texture: Option<&'a Bc1Upload<'a>>,
}

/// Colour bytes, tightly packed top-to-bottom, and the stored depth values
/// when the pass has a depth attachment.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthOutput {
    pub color: Vec<u8>,
    pub depth: Option<Vec<f32>>,
}

impl NativeCompute {
    /// Draw a depth-tested offscreen pass and read back colour and depth.
    ///
    /// Each draw sets its own depth state before drawing, so depth compare,
    /// clear and write enable act exactly as the pass describes. The stored
    /// depth is read back by a compute kernel that returns each value as
    /// the texture read gives it, without numeric conversion.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid extent, too many draws, non-finite
    /// positions, a depth state without depth attachment, a non-finite clear
    /// depth or one outside 0..=1, a textured draw without texture, unavailable
    /// BC compression (operation `supportsBCTextureCompression`), or Metal
    /// setup or execution failure.
    pub fn render_depth(&self, pass: &DepthPass<'_>) -> Result<DepthOutput, MetalError> {
        let [width, height] = pass.extent;
        let pixels = width
            .checked_mul(height)
            .filter(|&count| (1..=MAX_PIXELS).contains(&count))
            .ok_or_else(|| MetalError::new("render bounds", "expected 1..=4096 pixels"))?;
        validate_depth_pass(pass)?;
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
            let library = self.compile_library(DEPTH_SOURCE, &options)?;
            let vertex = find_function(&library, "depth_vertex")?;
            let depth_format = if pass.clear_depth.is_some() {
                DEPTH_FORMAT
            } else {
                MTLPixelFormat::Invalid
            };
            let flat = self.depth_pipeline(
                &vertex,
                &*find_function(&library, "depth_flat")?,
                depth_format,
            )?;
            let textured = self.depth_pipeline(
                &vertex,
                &*find_function(&library, "depth_textured")?,
                depth_format,
            )?;
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
            let depth = pass
                .clear_depth
                .map(|clear| {
                    let texture = self.new_target(
                        DEPTH_FORMAT,
                        width,
                        height,
                        MTLStorageMode::Private,
                        MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
                    )?;
                    let depth_attachment = pass_descriptor.depthAttachment();
                    depth_attachment.setTexture(Some(&texture));
                    depth_attachment.setLoadAction(MTLLoadAction::Clear);
                    depth_attachment.setClearDepth(f64::from(clear));
                    depth_attachment.setStoreAction(MTLStoreAction::Store);
                    Ok::<_, MetalError>(texture)
                })
                .transpose()?;

            // A pass without draws still clears and stores both attachments.
            let words: Vec<u32> = pass
                .draws
                .iter()
                .flat_map(|draw| draw.positions.as_flattened())
                .map(|value| value.to_bits())
                .collect();
            let positions = (!words.is_empty())
                .then(|| self.new_buffer(&words))
                .transpose()?;
            // Everything fallible is resolved before encoding starts, so an
            // error never leaves an encoder without endEncoding.
            let states = pass
                .draws
                .iter()
                .map(|draw| {
                    depth
                        .as_ref()
                        .map(|_| self.depth_state(draw.depth))
                        .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?;
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
            for (index, (draw, state)) in pass.draws.iter().zip(&states).enumerate() {
                if let Some(state) = state {
                    encoder.setDepthStencilState(Some(state));
                }
                // SAFETY: The position buffer holds three float4s per draw and
                // the vertex stage reads only this draw's three. Fragment
                // constants are copied by Metal and match the declared types.
                // The texture, when bound, is a live BC1 texture of this device.
                unsafe {
                    encoder.setVertexBuffer_offset_atIndex(positions.as_deref(), index * 48, 0);
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
                    encoder.drawPrimitives_vertexStart_vertexCount(
                        MTLPrimitiveType::Triangle,
                        0,
                        3,
                    );
                }
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
            let depth = depth
                .map(|texture| {
                    self.run(
                        DEPTH_READBACK,
                        "read_depth",
                        ComputeInput::Texture(&texture, pixels),
                    )
                    .map(|profile| profile.output.into_iter().map(f32::from_bits).collect())
                })
                .transpose()?;
            Ok(DepthOutput {
                color: output,
                depth,
            })
        })
    }

    fn depth_pipeline(
        &self,
        vertex: &ProtocolObject<dyn MTLFunction>,
        fragment: &ProtocolObject<dyn MTLFunction>,
        depth_format: MTLPixelFormat,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, MetalError> {
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(vertex));
        descriptor.setFragmentFunction(Some(fragment));
        // SAFETY: Slot zero is the only color attachment for this pipeline.
        unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) }
            .setPixelFormat(PIXEL_FORMAT);
        descriptor.setDepthAttachmentPixelFormat(depth_format);
        let pipeline = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| {
                MetalError::new(
                    "newRenderPipelineStateWithDescriptor",
                    error.localizedDescription().to_string(),
                )
            })?;
        verify_render_validation(&pipeline)?;
        Ok(pipeline)
    }

    fn depth_state(
        &self,
        state: DepthState,
    ) -> Result<Retained<ProtocolObject<dyn MTLDepthStencilState>>, MetalError> {
        let descriptor = MTLDepthStencilDescriptor::new();
        descriptor.setDepthCompareFunction(state.compare.metal());
        descriptor.setDepthWriteEnabled(state.write);
        self.device
            .newDepthStencilStateWithDescriptor(&descriptor)
            .ok_or_else(|| {
                MetalError::new(
                    "newDepthStencilStateWithDescriptor",
                    "depth state creation failed",
                )
            })
    }

    fn new_target(
        &self,
        format: MTLPixelFormat,
        width: usize,
        height: usize,
        storage: MTLStorageMode,
        usage: MTLTextureUsage,
    ) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, MetalError> {
        // SAFETY: The nonzero dimensions are bounded by MAX_PIXELS, and both
        // RGBA8Unorm and Depth32Float are renderable on every Apple GPU family.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format, width, height, false,
            )
        };
        descriptor.setStorageMode(storage);
        descriptor.setUsage(usage);
        self.device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                MetalError::new(
                    "newTextureWithDescriptor",
                    "render target allocation failed",
                )
            })
    }

    fn new_bc1_texture(
        &self,
        level: &Bc1Subresource<'_>,
    ) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, MetalError> {
        let texture = self.new_target(
            MTLPixelFormat::BC1_RGBA,
            level.width,
            level.height,
            MTLStorageMode::Shared,
            MTLTextureUsage::ShaderRead,
        )?;
        // Apple documents a zero stride for a single compressed block row.
        let bytes_per_row = if level.height <= 4 {
            0
        } else {
            level.row_pitch
        };
        // SAFETY: The region spans the whole level. The level comes only from
        // `Bc1Upload::subresource`, whose layout validation guarantees that the
        // bytes hold every block row, including partial edge blocks, at this
        // block-aligned stride.
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                whole(level.width, level.height),
                0,
                NonNull::from(level.bytes).cast(),
                bytes_per_row,
            );
        }
        Ok(texture)
    }
}

const MAX_INDEXED_VERTICES: usize = 1024;
const MAX_INDICES: usize = 4096;

/// An index buffer of 16- or 32-bit unsigned indices.
#[derive(Clone, Copy, Debug)]
pub enum IndexBuffer<'a> {
    U16(&'a [u16]),
    U32(&'a [u32]),
}

impl IndexBuffer<'_> {
    fn len(self) -> usize {
        match self {
            Self::U16(indices) => indices.len(),
            Self::U32(indices) => indices.len(),
        }
    }

    fn index(self, position: usize) -> u32 {
        match self {
            Self::U16(indices) => u32::from(indices[position]),
            Self::U32(indices) => indices[position],
        }
    }

    fn restart(self) -> u32 {
        match self {
            Self::U16(_) => u32::from(u16::MAX),
            Self::U32(_) => u32::MAX,
        }
    }

    fn size(self) -> usize {
        match self {
            Self::U16(_) => 2,
            Self::U32(_) => 4,
        }
    }

    /// The indices packed into little-endian words, the last one zero-padded.
    fn words(self) -> Vec<u32> {
        match self {
            Self::U16(indices) => indices
                .chunks(2)
                .map(|pair| {
                    u32::from(pair[0]) | pair.get(1).map_or(0, |&high| u32::from(high) << 16)
                })
                .collect(),
            Self::U32(indices) => indices.to_vec(),
        }
    }
}

/// One indexed triangle-list draw: `index_count` indices from `start_index`,
/// each fetching vertex `index + base_vertex`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexedDraw {
    pub start_index: usize,
    pub index_count: usize,
    pub base_vertex: i32,
    pub fill: DepthFill,
}

/// An offscreen pass of up to 16 indexed draws, in order, into a cleared
/// RGBA8Unorm target without depth attachment.
#[derive(Clone, Copy, Debug)]
pub struct IndexedPass<'a> {
    pub extent: [usize; 2],
    pub clear_color: [u8; 4],
    /// Up to 1024 clip-space float4 positions.
    pub vertices: &'a [[f32; 4]],
    /// Up to 4096 indices shared by every draw.
    pub indices: IndexBuffer<'a>,
    pub draws: &'a [IndexedDraw],
    /// As [`DepthPass::texture`], for [`DepthFill::Bc1Texture`] draws.
    pub texture: Option<&'a Bc1Upload<'a>>,
}

impl NativeCompute {
    /// Draw an indexed offscreen pass and read back its colour.
    ///
    /// Each draw is one native `drawIndexedPrimitives` call on the pass's
    /// index buffer with the draw's start offset and signed base vertex, so
    /// Metal itself fetches the vertices. Every fetched vertex is checked
    /// against the vertex buffer first: out-of-range fetches and raw
    /// primitive-restart values are rejected, never left to the GPU. Base vertex needs no capability query: the
    /// Apple family 7 that [`NativeCompute::new`] requires supports it.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid extent, too many draws, vertices or
    /// indices, non-finite positions, an index count that is zero or not a
    /// multiple of 3, an index range past the buffer, a primitive-restart
    /// index, a fetched vertex outside the vertex buffer, a textured draw without texture, unavailable BC
    /// compression (operation `supportsBCTextureCompression`), or Metal setup
    /// or execution failure.
    pub fn render_indexed(&self, pass: &IndexedPass<'_>) -> Result<Vec<u8>, MetalError> {
        let [width, height] = pass.extent;
        let pixels = width
            .checked_mul(height)
            .filter(|&count| (1..=MAX_PIXELS).contains(&count))
            .ok_or_else(|| MetalError::new("render bounds", "expected 1..=4096 pixels"))?;
        validate_indexed_pass(pass)?;
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
            let library = self.compile_library(DEPTH_SOURCE, &options)?;
            let vertex = find_function(&library, "depth_vertex")?;
            let flat = self.depth_pipeline(
                &vertex,
                &*find_function(&library, "depth_flat")?,
                MTLPixelFormat::Invalid,
            )?;
            let textured = self.depth_pipeline(
                &vertex,
                &*find_function(&library, "depth_textured")?,
                MTLPixelFormat::Invalid,
            )?;
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
            let index_type = match pass.indices {
                IndexBuffer::U16(_) => MTLIndexType::UInt16,
                IndexBuffer::U32(_) => MTLIndexType::UInt32,
            };
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
                // SAFETY: Validation proved that every index the draw reads lies
                // inside the index buffer, that the offset is a multiple of the
                // index size, and that every `index + base_vertex` names a float4
                // of the position buffer, the only memory the vertex stage reads.
                // Fragment constants are copied by Metal and match the declared
                // types. The texture, when bound, is a live BC1 texture of this
                // device; both buffers outlive the command.
                unsafe {
                    encoder.setVertexBuffer_offset_atIndex(Some(&positions), 0, 0);
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
                    encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                        MTLPrimitiveType::Triangle,
                        draw.index_count,
                        index_type,
                        &indices,
                        draw.start_index * pass.indices.size(),
                        1,
                        draw.base_vertex as isize,
                        0,
                    );
                }
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
}

fn validate_indexed_pass(pass: &IndexedPass<'_>) -> Result<(), MetalError> {
    let invalid = |reason: String| Err(MetalError::new("indexed pass", reason));
    if pass.draws.len() > MAX_DEPTH_DRAWS {
        return invalid("expected at most 16 draws".to_owned());
    }
    if !(1..=MAX_INDEXED_VERTICES).contains(&pass.vertices.len())
        || !(1..=MAX_INDICES).contains(&pass.indices.len())
    {
        return invalid("expected 1..=1024 vertices and 1..=4096 indices".to_owned());
    }
    if pass
        .vertices
        .as_flattened()
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(MetalError::new(
            "render input",
            "expected finite vertex data",
        ));
    }
    for (number, draw) in pass.draws.iter().enumerate() {
        if draw.index_count == 0 || !draw.index_count.is_multiple_of(3) {
            return invalid(format!(
                "draw {number}: index count must be a nonzero multiple of 3"
            ));
        }
        let end = draw
            .start_index
            .checked_add(draw.index_count)
            .filter(|&end| end <= pass.indices.len())
            .ok_or_else(|| {
                MetalError::new(
                    "indexed pass",
                    format!("draw {number}: index range past the index buffer"),
                )
            })?;
        for position in draw.start_index..end {
            let index = pass.indices.index(position);
            // Metal documents 0xFFFF / 0xFFFFFFFF as the primitive-restart value
            // of indexed draws without defining it for triangle lists, so the
            // raw value is rejected whatever the base vertex.
            if index == pass.indices.restart() {
                return invalid(format!(
                    "draw {number}: index {position} is the primitive-restart value"
                ));
            }
            let vertex = i64::from(index) + i64::from(draw.base_vertex);
            if !(0..pass.vertices.len() as i64).contains(&vertex) {
                return invalid(format!(
                    "draw {number}: index {position} fetches vertex {vertex} outside the vertex buffer"
                ));
            }
        }
        if draw.fill == DepthFill::Bc1Texture && pass.texture.is_none() {
            return invalid("textured draw without texture".to_owned());
        }
    }
    Ok(())
}

fn validate_depth_pass(pass: &DepthPass<'_>) -> Result<(), MetalError> {
    if pass.draws.len() > MAX_DEPTH_DRAWS {
        return Err(MetalError::new("depth pass", "expected at most 16 draws"));
    }
    if pass
        .draws
        .iter()
        .flat_map(|draw| draw.positions.as_flattened())
        .any(|value| !value.is_finite())
    {
        return Err(MetalError::new(
            "render input",
            "expected finite vertex data",
        ));
    }
    if let Some(clear) = pass.clear_depth
        && !(0.0..=1.0).contains(&clear)
    {
        return Err(MetalError::new(
            "depth pass",
            "expected a clear depth in 0..=1",
        ));
    }
    if pass.clear_depth.is_none()
        && pass
            .draws
            .iter()
            .any(|draw| draw.depth != DepthState::DISABLED)
    {
        return Err(MetalError::new(
            "depth pass",
            "a depth test or write needs a depth attachment",
        ));
    }
    if pass.texture.is_none()
        && pass
            .draws
            .iter()
            .any(|draw| draw.fill == DepthFill::Bc1Texture)
    {
        return Err(MetalError::new(
            "depth pass",
            "textured draw without texture",
        ));
    }
    Ok(())
}

fn whole(width: usize, height: usize) -> MTLRegion {
    MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width,
            height,
            depth: 1,
        },
    }
}

/// A blend factor, as Metal defines it for the RGB and the alpha channel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlendFactor {
    Zero,
    One,
    SourceColor,
    OneMinusSourceColor,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestinationColor,
    OneMinusDestinationColor,
    DestinationAlpha,
    OneMinusDestinationAlpha,
}

impl BlendFactor {
    fn metal(self) -> MTLBlendFactor {
        match self {
            Self::Zero => MTLBlendFactor::Zero,
            Self::One => MTLBlendFactor::One,
            Self::SourceColor => MTLBlendFactor::SourceColor,
            Self::OneMinusSourceColor => MTLBlendFactor::OneMinusSourceColor,
            Self::SourceAlpha => MTLBlendFactor::SourceAlpha,
            Self::OneMinusSourceAlpha => MTLBlendFactor::OneMinusSourceAlpha,
            Self::DestinationColor => MTLBlendFactor::DestinationColor,
            Self::OneMinusDestinationColor => MTLBlendFactor::OneMinusDestinationColor,
            Self::DestinationAlpha => MTLBlendFactor::DestinationAlpha,
            Self::OneMinusDestinationAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        }
    }
}

/// A blend operation combining the weighted source and destination; `Min`
/// and `Max` take the unweighted values and ignore the factors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

impl BlendOperation {
    fn metal(self) -> MTLBlendOperation {
        match self {
            Self::Add => MTLBlendOperation::Add,
            Self::Subtract => MTLBlendOperation::Subtract,
            Self::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
            Self::Min => MTLBlendOperation::Min,
            Self::Max => MTLBlendOperation::Max,
        }
    }
}

/// Separate RGB and alpha blend equations of one draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlendState {
    pub source_rgb: BlendFactor,
    pub destination_rgb: BlendFactor,
    pub rgb_operation: BlendOperation,
    pub source_alpha: BlendFactor,
    pub destination_alpha: BlendFactor,
    pub alpha_operation: BlendOperation,
}

/// The colour channels a draw writes; the others keep the stored value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ColorWriteMask {
    pub red: bool,
    pub green: bool,
    pub blue: bool,
    pub alpha: bool,
}

impl ColorWriteMask {
    pub const ALL: Self = Self {
        red: true,
        green: true,
        blue: true,
        alpha: true,
    };

    fn metal(self) -> MTLColorWriteMask {
        [
            (self.red, MTLColorWriteMask::Red),
            (self.green, MTLColorWriteMask::Green),
            (self.blue, MTLColorWriteMask::Blue),
            (self.alpha, MTLColorWriteMask::Alpha),
        ]
        .into_iter()
        .filter(|(on, _)| *on)
        .fold(MTLColorWriteMask::None, |mask, (_, bit)| mask | bit)
    }
}

/// One flat-coloured triangle of a blend pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendDraw {
    pub positions: [[f32; 4]; 3],
    pub color: [u8; 4],
    /// `None` disables blending: the source replaces the masked channels.
    pub blend: Option<BlendState>,
    pub write_mask: ColorWriteMask,
}

/// An offscreen pass of up to 16 triangles drawn in order into a cleared
/// RGBA8Unorm target without depth attachment.
#[derive(Clone, Copy, Debug)]
pub struct BlendPass<'a> {
    pub extent: [usize; 2],
    pub clear_color: [u8; 4],
    pub draws: &'a [BlendDraw],
}

impl NativeCompute {
    /// Draw a blended, write-masked offscreen pass and read back its colour.
    ///
    /// Blend state and write mask are pipeline state in Metal, so each draw
    /// gets its own pipeline built from exactly the draw's settings.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid extent, too many draws, non-finite
    /// positions, or Metal setup or execution failure.
    pub fn render_blend(&self, pass: &BlendPass<'_>) -> Result<Vec<u8>, MetalError> {
        let [width, height] = pass.extent;
        let pixels = width
            .checked_mul(height)
            .filter(|&count| (1..=MAX_PIXELS).contains(&count))
            .ok_or_else(|| MetalError::new("render bounds", "expected 1..=4096 pixels"))?;
        if pass.draws.len() > MAX_DEPTH_DRAWS {
            return Err(MetalError::new("blend pass", "expected at most 16 draws"));
        }
        if pass
            .draws
            .iter()
            .flat_map(|draw| draw.positions.as_flattened())
            .any(|value| !value.is_finite())
        {
            return Err(MetalError::new(
                "render input",
                "expected finite vertex data",
            ));
        }

        autoreleasepool(|_| {
            let options = compile_options()?;
            let library = self.compile_library(DEPTH_SOURCE, &options)?;
            let vertex = find_function(&library, "depth_vertex")?;
            let fragment = find_function(&library, "depth_flat")?;
            let pipelines = pass
                .draws
                .iter()
                .map(|draw| self.blend_pipeline(&vertex, &fragment, draw))
                .collect::<Result<Vec<_>, _>>()?;
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
            let words: Vec<u32> = pass
                .draws
                .iter()
                .flat_map(|draw| draw.positions.as_flattened())
                .map(|value| value.to_bits())
                .collect();
            let positions = (!words.is_empty())
                .then(|| self.new_buffer(&words))
                .transpose()?;
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
            for (index, (draw, pipeline)) in pass.draws.iter().zip(&pipelines).enumerate() {
                encoder.setRenderPipelineState(pipeline);
                let rgba = draw.color.map(|value| f32::from(value) / 255.0);
                // SAFETY: The position buffer holds three float4s per draw and
                // the vertex stage reads only this draw's three. The fragment
                // constant is copied by Metal and matches the declared float4.
                unsafe {
                    encoder.setVertexBuffer_offset_atIndex(positions.as_deref(), index * 48, 0);
                    encoder.setFragmentBytes_length_atIndex(
                        NonNull::from(&rgba).cast(),
                        size_of_val(&rgba),
                        0,
                    );
                    encoder.drawPrimitives_vertexStart_vertexCount(
                        MTLPrimitiveType::Triangle,
                        0,
                        3,
                    );
                }
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

    fn blend_pipeline(
        &self,
        vertex: &ProtocolObject<dyn MTLFunction>,
        fragment: &ProtocolObject<dyn MTLFunction>,
        draw: &BlendDraw,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, MetalError> {
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(vertex));
        descriptor.setFragmentFunction(Some(fragment));
        // SAFETY: Slot zero is the only color attachment for this pipeline.
        let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        attachment.setPixelFormat(PIXEL_FORMAT);
        attachment.setWriteMask(draw.write_mask.metal());
        attachment.setBlendingEnabled(draw.blend.is_some());
        if let Some(blend) = draw.blend {
            attachment.setSourceRGBBlendFactor(blend.source_rgb.metal());
            attachment.setDestinationRGBBlendFactor(blend.destination_rgb.metal());
            attachment.setRgbBlendOperation(blend.rgb_operation.metal());
            attachment.setSourceAlphaBlendFactor(blend.source_alpha.metal());
            attachment.setDestinationAlphaBlendFactor(blend.destination_alpha.metal());
            attachment.setAlphaBlendOperation(blend.alpha_operation.metal());
        }
        let pipeline = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| {
                MetalError::new(
                    "newRenderPipelineStateWithDescriptor",
                    error.localizedDescription().to_string(),
                )
            })?;
        verify_render_validation(&pipeline)?;
        Ok(pipeline)
    }
}
