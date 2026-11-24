use std::ffi::c_void;
use std::fmt;
use std::mem::size_of_val;
use std::ptr::{NonNull, copy_nonoverlapping};
use std::time::Instant;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::sel;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction, MTLGPUFamily, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode, MTLOrigin, MTLPixelFormat, MTLRegion,
    MTLResourceOptions, MTLShaderValidation, MTLSize, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage,
};
use xenora_texture::{
    Bc1Upload, BcAlphaFormat, BcAlphaUpload, BcChannelFormat, BcChannelUpload, TiledBc1,
};

#[path = "render.rs"]
mod render;

pub use render::{
    BlendDraw, BlendFactor, BlendOperation, BlendPass, BlendState, ColorWriteMask, DepthCompare,
    DepthDraw, DepthFill, DepthOutput, DepthPass, DepthState, IndexBuffer, IndexedDraw,
    IndexedPass,
};

const MAX_THREADS_PER_GROUP: usize = 64;
const MAX_GROUPS: usize = 64;
const MAX_GRID_WORDS: usize = MAX_GROUPS * MAX_THREADS_PER_GROUP;
const MAX_SOURCE_BYTES: usize = 64 * 1024;

const BC_READBACK: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void read_bc(texture2d<float, access::read> image [[texture(0)]],
                    device uint *output [[buffer(1)]],
                    uint index [[thread_position_in_grid]]) {
    uint width = image.get_width();
    uint4 channels = uint4(round(image.read(uint2(index % width, index / width)) * 255.0f));
    output[index] = channels.r | (channels.g << 8u) | (channels.b << 16u) | (channels.a << 24u);
}
";

// Signed formats return -1.0..=1.0; each byte holds the channel times 127.
const BC_SNORM_READBACK: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void read_bc_snorm(texture2d<float, access::read> image [[texture(0)]],
                          device uint *output [[buffer(1)]],
                          uint index [[thread_position_in_grid]]) {
    uint width = image.get_width();
    float4 texel = clamp(image.read(uint2(index % width, index / width)), -1.0f, 1.0f);
    uint4 bytes = uint4(int4(round(texel * 127.0f))) & 0xffu;
    output[index] = bytes.r | (bytes.g << 8u) | (bytes.b << 16u) | (bytes.a << 24u);
}
";

enum ComputeInput<'a> {
    Words(&'a [u32]),
    Texture(&'a ProtocolObject<dyn MTLTexture>, usize),
    SampledTexture {
        texture: &'a ProtocolObject<dyn MTLTexture>,
        sampler: &'a ProtocolObject<dyn objc2_metal::MTLSamplerState>,
        coords: &'a [u32],
    },
}

impl ComputeInput<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Words(words) => words.len(),
            Self::Texture(_, pixels) => *pixels,
            Self::SampledTexture { coords, .. } => coords.len() / 2,
        }
    }
}

/// Host-provided backing bytes for one synthetic guest resource.
pub struct GuestBuffer<'a> {
    pub format: u32,
    pub bytes: &'a [u8],
}

impl GuestBuffer<'_> {
    /// The synthetic ABI's format identifier for little-endian 32-bit words.
    pub const U32_LE: u32 = 1;
}

/// The synthetic guest's selection of an input range from a resource table.
#[derive(Debug, Clone, Copy)]
pub struct GuestBufferView {
    /// Zero-based index in the supplied resource table.
    pub handle: usize,
    /// Input slot in the fixture ABI. Only slot zero accepts guest data.
    pub binding: u32,
    pub byte_offset: usize,
    pub byte_len: usize,
}

/// Failure at a named operation in the native Metal backend.
#[derive(Debug)]
pub struct MetalError {
    pub operation: &'static str,
    pub detail: String,
}

impl MetalError {
    /// Operation name of the error returned when the device lacks BC texture
    /// compression, which is a hardware capability, not an implementation fault.
    pub const BC_UNAVAILABLE: &'static str = "supportsBCTextureCompression";

    /// Operation name of the error returned when the device is outside the
    /// Metal 3 GPU family that [`NativeCompute::render_mesh`] requires.
    pub const MESH_UNAVAILABLE: &'static str = "supportsFamily(Metal3)";

    /// Whether the device lacks a required capability, as opposed to a failure
    /// in Xenora's validation or Metal execution.
    #[must_use]
    pub fn is_capability_unavailable(&self) -> bool {
        self.operation == Self::BC_UNAVAILABLE || self.operation == Self::MESH_UNAVAILABLE
    }

    pub(crate) fn new(operation: &'static str, detail: impl Into<String>) -> Self {
        Self {
            operation,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for MetalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.operation, self.detail)
    }
}

impl std::error::Error for MetalError {}

/// One compiled-and-dispatched kernel with its measured stage timings.
///
/// Durations come from monotonic host clocks for the compiler stages and from
/// the command buffer's GPU start and end timestamps for `gpu_seconds`. They
/// measure a single submission and are not a benchmark distribution.
#[derive(Debug, Clone)]
pub struct RunProfile {
    pub library_seconds: f64,
    pub pipeline_seconds: f64,
    pub gpu_seconds: f64,
    pub host_seconds: f64,
    pub output: Vec<u32>,
}

/// A native Metal device and its command queue for the fixture ABI.
pub struct NativeCompute {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl NativeCompute {
    /// Open the default Apple GPU and create a retained-resource command queue.
    ///
    /// # Errors
    ///
    /// Returns an error if discovery, the capability check, or queue creation fails.
    pub fn new() -> Result<Self, MetalError> {
        autoreleasepool(|_| {
            let device = MTLCreateSystemDefaultDevice().ok_or_else(|| {
                MetalError::new("MTLCreateSystemDefaultDevice", "no Metal device")
            })?;
            // Apple family 7 documents nonuniform threadgroups, which this ABI needs.
            if !device.supportsFamily(MTLGPUFamily::Apple7) {
                return Err(MetalError::new(
                    "supportsFamily",
                    "Apple family 7 capabilities are required",
                ));
            }
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| MetalError::new("newCommandQueue", "queue creation failed"))?;
            Ok(Self { device, queue })
        })
    }

    /// Report the selected device's registered name.
    pub fn device_name(&self) -> String {
        autoreleasepool(|_| self.device.name().to_string())
    }

    /// Access the underlying native Metal device protocol object.
    #[inline]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Access the underlying native Metal command queue protocol object.
    #[inline]
    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    /// Read an original synthetic BC1 subresource with a native Metal texture.
    ///
    /// The fixed compute kernel returns row-major RGBA8 texels. Each returned
    /// word stores red in the least significant byte and alpha in the highest.
    ///
    /// # Errors
    ///
    /// Returns an error if the subresource is invalid, exceeds the 4096-texel
    /// limit, BC texture compression is unavailable, or Metal execution fails.
    pub fn read_tiled_bc1(
        &self,
        tiled: &TiledBc1<'_>,
        layer: usize,
        mip: u32,
    ) -> Result<Vec<u32>, MetalError> {
        let (width, height) = tiled
            .mip_extent(mip)
            .map_err(|error| MetalError::new("texture subresource", format!("{error:?}")))?;
        let pixels = self.validate_bc_extent(width, height, "BC1")?;
        let blocks = tiled
            .linearize(layer, mip)
            .map_err(|error| MetalError::new("texture subresource", format!("{error:?}")))?;
        self.read_bc_blocks(
            MTLPixelFormat::BC1_RGBA,
            &blocks,
            width.div_ceil(4) * 8,
            width,
            height,
            pixels,
        )
    }

    /// Sample a native Metal RGBA8 texture using a native Metal sampler on Apple Silicon GPU.
    ///
    /// Accepts normalized float UV coordinates.
    /// Executes an authored compute fixture sampling the texture at `[[texture(0)]]` using
    /// the sampler at `[[sampler(0)]]`, reading coordinates from `[[buffer(0)]]`, and writing
    /// row-major RGBA8 texels to `[[buffer(1)]]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of sample points exceeds maximum allowable limits,
    /// or if Metal compilation, dispatch or completion fails.
    pub fn sample_rgba8(
        &self,
        texture: &crate::texture_pipeline::MetalRgba8Texture,
        sampler: &crate::sampler::MetalSampler,
        uv_coords: &[[f32; 2]],
    ) -> Result<Vec<u32>, MetalError> {
        const SAMPLER_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void sample_rgba8(texture2d<float, access::sample> tex [[texture(0)]],
                         sampler s [[sampler(0)]],
                         device const float2 *coords [[buffer(0)]],
                         device uint *output [[buffer(1)]],
                         uint id [[thread_position_in_grid]]) {
    float4 color = tex.sample(s, coords[id], level(0.0f));
    uint4 channels = uint4(round(color * 255.0f));
    output[id] = channels.r | (channels.g << 8u) | (channels.b << 16u) | (channels.a << 24u);
}
"#;
        if uv_coords.is_empty() || uv_coords.len() > MAX_GRID_WORDS {
            return Err(MetalError::new(
                "sample bounds",
                format!(
                    "expected 1..={MAX_GRID_WORDS} sample points, got {}",
                    uv_coords.len()
                ),
            ));
        }

        let words: Vec<u32> = uv_coords
            .iter()
            .flat_map(|[u, v]| [u.to_bits(), v.to_bits()])
            .collect();

        validate_source(SAMPLER_KERNEL, &["sample_rgba8"])?;
        autoreleasepool(|_| {
            self.run(
                SAMPLER_KERNEL,
                "sample_rgba8",
                ComputeInput::SampledTexture {
                    texture: texture.raw(),
                    sampler: sampler.raw(),
                    coords: &words,
                },
            )
            .map(|profile| profile.output)
        })
    }

    /// Read a selected subresource from a validated pitched BC1 upload.
    ///
    /// Output matches [`Self::read_tiled_bc1`]. Metal receives the upload's
    /// block-row stride and reads only the selected level's logical blocks.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer or mip, out-of-policy size,
    /// unavailable BC compression, or Metal setup or execution failure.
    pub fn read_bc1_upload(
        &self,
        upload: &Bc1Upload<'_>,
        layer: usize,
        mip: u32,
    ) -> Result<Vec<u32>, MetalError> {
        let level = upload
            .subresource(layer, mip)
            .map_err(|error| MetalError::new("texture subresource", format!("{error:?}")))?;
        let pixels = self.validate_bc_extent(level.width, level.height, "BC1")?;
        self.read_bc_blocks(
            MTLPixelFormat::BC1_RGBA,
            level.bytes,
            level.row_pitch,
            level.width,
            level.height,
            pixels,
        )
    }

    /// Read a selected subresource from a validated pitched BC2 or BC3 upload.
    ///
    /// The original 16-byte blocks go to a native `BC2_RGBA` or `BC3_RGBA`
    /// texture, and the output has the layout of [`Self::read_bc1_upload`].
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer or mip, out-of-policy size,
    /// unavailable BC compression, or Metal setup or execution failure.
    pub fn read_bc_alpha_upload(
        &self,
        upload: &BcAlphaUpload<'_>,
        layer: usize,
        mip: u32,
    ) -> Result<Vec<u32>, MetalError> {
        let level = upload
            .subresource(layer, mip)
            .map_err(|error| MetalError::new("texture subresource", format!("{error:?}")))?;
        let (format, name) = match upload.format() {
            BcAlphaFormat::Bc2 => (MTLPixelFormat::BC2_RGBA, "BC2"),
            BcAlphaFormat::Bc3 => (MTLPixelFormat::BC3_RGBA, "BC3"),
        };
        let pixels = self.validate_bc_extent(level.width, level.height, name)?;
        self.read_bc_blocks(
            format,
            level.bytes,
            level.row_pitch,
            level.width,
            level.height,
            pixels,
        )
    }

    /// Read a selected subresource from a validated pitched BC4 or BC5 upload.
    ///
    /// The original blocks go to a native `BC4_R*` or `BC5_RG*` texture.
    /// Unsigned output has the layout of [`Self::read_bc1_upload`]. Signed
    /// output stores each channel, times 127 and rounded, as a two's-complement
    /// byte in the same order. Metal fills missing green and blue with 0 and
    /// alpha with 1.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer or mip, out-of-policy size,
    /// unavailable BC compression (operation `supportsBCTextureCompression`),
    /// or Metal setup or execution failure.
    pub fn read_bc_channel_upload(
        &self,
        upload: &BcChannelUpload<'_>,
        layer: usize,
        mip: u32,
    ) -> Result<Vec<u32>, MetalError> {
        let level = upload
            .subresource(layer, mip)
            .map_err(|error| MetalError::new("texture subresource", format!("{error:?}")))?;
        let (format, name) = match upload.format() {
            BcChannelFormat::Bc4Unorm => (MTLPixelFormat::BC4_RUnorm, "BC4 UNORM"),
            BcChannelFormat::Bc4Snorm => (MTLPixelFormat::BC4_RSnorm, "BC4 SNORM"),
            BcChannelFormat::Bc5Unorm => (MTLPixelFormat::BC5_RGUnorm, "BC5 UNORM"),
            BcChannelFormat::Bc5Snorm => (MTLPixelFormat::BC5_RGSnorm, "BC5 SNORM"),
        };
        let pixels = self.validate_bc_extent(level.width, level.height, name)?;
        self.read_bc_blocks(
            format,
            level.bytes,
            level.row_pitch,
            level.width,
            level.height,
            pixels,
        )
    }

    fn validate_bc_extent(
        &self,
        width: usize,
        height: usize,
        name: &str,
    ) -> Result<usize, MetalError> {
        let pixels = width
            .checked_mul(height)
            .filter(|pixels| (1..=MAX_GRID_WORDS).contains(pixels))
            .ok_or_else(|| MetalError::new("texture bounds", "expected 1..=4096 texels"))?;
        if !self.device.supportsBCTextureCompression() {
            return Err(MetalError::new(
                MetalError::BC_UNAVAILABLE,
                format!("{name} is unavailable"),
            ));
        }
        Ok(pixels)
    }

    fn read_bc_blocks(
        &self,
        format: MTLPixelFormat,
        blocks: &[u8],
        row_pitch: usize,
        width: usize,
        height: usize,
        pixels: usize,
    ) -> Result<Vec<u32>, MetalError> {
        autoreleasepool(|_| {
            // SAFETY: The positive bounded dimensions and supported BC format
            // describe a 2D texture with just its base level.
            let descriptor = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    format, width, height, false,
                )
            };
            descriptor.setStorageMode(MTLStorageMode::Shared);
            descriptor.setUsage(MTLTextureUsage::ShaderRead);
            let texture = self
                .device
                .newTextureWithDescriptor(&descriptor)
                .ok_or_else(|| {
                    MetalError::new("newTextureWithDescriptor", "BC allocation failed")
                })?;
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width,
                    height,
                    depth: 1,
                },
            };
            // Apple documents a zero stride for a single compressed block row.
            let bytes_per_row = if height <= 4 { 0 } else { row_pitch };
            // SAFETY: The region spans the whole level, including any partial
            // edge block. Both callers validate that the input holds all rows
            // of complete blocks of this format at the supplied, block-aligned
            // stride.
            unsafe {
                texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    region,
                    0,
                    NonNull::from(blocks).cast(),
                    bytes_per_row,
                );
            }
            let (source, entry) =
                if format == MTLPixelFormat::BC4_RSnorm || format == MTLPixelFormat::BC5_RGSnorm {
                    (BC_SNORM_READBACK, "read_bc_snorm")
                } else {
                    (BC_READBACK, "read_bc")
                };
            self.run(source, entry, ComputeInput::Texture(&texture, pixels))
                .map(|profile| profile.output)
        })
    }

    /// Run trusted MSL with read-only words at buffer 0 and output words at buffer 1.
    ///
    /// The kernel must use one invocation per word and restrict its memory accesses
    /// to these bindings and ranges. Run this inside a child process with a wall
    /// deadline. The backend caps resources and dispatch size but does not verify
    /// the MSL, its control flow, or its termination.
    ///
    /// # Errors
    ///
    /// Returns an error for out-of-policy input sizes, compilation, resource setup,
    /// dispatch or completion failure.
    pub fn run_words(
        &self,
        source: &str,
        entry: &str,
        input: &[u32],
    ) -> Result<Vec<u32>, MetalError> {
        self.run_words_profiled(source, entry, input)
            .map(|profile| profile.output)
    }

    /// Resolve a synthetic guest input view and run the trusted fixture kernel.
    ///
    /// The validated word range is copied into its own Metal buffer at slot zero.
    /// Slot one receives a separate output buffer with the same number of words.
    /// The trusted-kernel requirements of [`Self::run_words`] also apply here.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported bindings or formats, invalid handles or
    /// ranges, or any error from [`Self::run_words`].
    pub fn run_guest_words(
        &self,
        source: &str,
        entry: &str,
        resources: &[GuestBuffer<'_>],
        view: GuestBufferView,
    ) -> Result<Vec<u32>, MetalError> {
        if view.binding != 0 {
            return Err(MetalError::new(
                "guest buffer binding",
                "expected input slot 0",
            ));
        }
        let buffer = resources
            .get(view.handle)
            .ok_or_else(|| MetalError::new("guest buffer handle", "handle is not in the table"))?;
        if buffer.format != GuestBuffer::U32_LE {
            return Err(MetalError::new("guest buffer format", "expected U32_LE"));
        }
        let word_size = size_of::<u32>();
        if !view.byte_offset.is_multiple_of(word_size)
            || !view.byte_len.is_multiple_of(word_size)
            || !(1..=MAX_GRID_WORDS).contains(&(view.byte_len / word_size))
        {
            return Err(MetalError::new(
                "guest buffer bounds",
                format!("expected an aligned range of 1..={MAX_GRID_WORDS} words"),
            ));
        }
        let bytes = view
            .byte_offset
            .checked_add(view.byte_len)
            .and_then(|end| buffer.bytes.get(view.byte_offset..end))
            .ok_or_else(|| MetalError::new("guest buffer bounds", "range exceeds the resource"))?;
        let words: Vec<_> = bytes
            .as_chunks::<{ size_of::<u32>() }>()
            .0
            .iter()
            .copied()
            .map(u32::from_le_bytes)
            .collect();
        self.run_words(source, entry, &words)
    }

    /// Run trusted MSL and report the measured compilation and execution stages.
    ///
    /// Applies the same bounds and completion checks as [`Self::run_words`].
    ///
    /// # Errors
    ///
    /// Returns an error for out-of-policy input sizes, compilation, resource setup,
    /// dispatch or completion failure.
    pub fn run_words_profiled(
        &self,
        source: &str,
        entry: &str,
        input: &[u32],
    ) -> Result<RunProfile, MetalError> {
        self.run_profiled_words(source, entry, input, None)
    }

    /// Run trusted MSL with an explicit one-dimensional threadgroup width.
    ///
    /// The grid still contains one invocation per input word. The requested width
    /// is passed to Metal even when the grid is smaller or ends in a partial group.
    /// The trusted-kernel requirements of [`Self::run_words`] apply.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::run_words_profiled`], with dispatch
    /// bounds evaluated using this width. Zero and widths above the device or
    /// pipeline limit are rejected.
    pub fn run_words_with_group_width(
        &self,
        source: &str,
        entry: &str,
        input: &[u32],
        width: usize,
    ) -> Result<RunProfile, MetalError> {
        self.run_profiled_words(source, entry, input, Some(width))
    }

    /// Compile the kernel and report the smaller of the device and pipeline width limits.
    ///
    /// Input size and threadgroup count are checked separately at dispatch.
    ///
    /// # Errors
    ///
    /// Returns an error if source validation, compilation or pipeline setup fails.
    pub fn max_group_width(&self, source: &str, entry: &str) -> Result<usize, MetalError> {
        validate_source(source, &[entry])?;
        autoreleasepool(|_| {
            let options = compile_options()?;
            let library = self.compile_library(source, &options)?;
            let function = find_function(&library, entry)?;
            let pipeline = self.compute_pipeline(&function)?;
            verify_shader_validation(&pipeline)?;
            Ok(self.pipeline_group_width_limit(&pipeline))
        })
    }

    fn run_profiled_words(
        &self,
        source: &str,
        entry: &str,
        input: &[u32],
        group_width: Option<usize>,
    ) -> Result<RunProfile, MetalError> {
        if !(1..=MAX_GRID_WORDS).contains(&input.len()) {
            return Err(MetalError::new(
                "dispatch bounds",
                format!("expected 1..={MAX_GRID_WORDS} words, got {}", input.len()),
            ));
        }
        validate_source(source, &[entry])?;
        autoreleasepool(|_| self.run_sized(source, entry, ComputeInput::Words(input), group_width))
    }

    fn run(
        &self,
        source: &str,
        entry: &str,
        input: ComputeInput<'_>,
    ) -> Result<RunProfile, MetalError> {
        self.run_sized(source, entry, input, None)
    }

    /// [`Self::run`] with an optional explicit threadgroup width; `None` keeps
    /// the default sizing.
    fn run_sized(
        &self,
        source: &str,
        entry: &str,
        input: ComputeInput<'_>,
        group_width: Option<usize>,
    ) -> Result<RunProfile, MetalError> {
        let host_start = Instant::now();
        let options = compile_options()?;
        let library_start = Instant::now();
        let library = self.compile_library(source, &options)?;
        let library_seconds = library_start.elapsed().as_secs_f64();
        let function = find_function(&library, entry)?;
        let pipeline_start = Instant::now();
        let pipeline = self.compute_pipeline(&function)?;
        let pipeline_seconds = pipeline_start.elapsed().as_secs_f64();
        verify_shader_validation(&pipeline)?;

        let limit = self.pipeline_group_width_limit(&pipeline);
        let width = match group_width {
            Some(width) if (1..=limit).contains(&width) => width,
            Some(width) => {
                return Err(MetalError::new(
                    "threadgroup width",
                    format!("expected a nonzero width at most {limit}, got {width}"),
                ));
            }
            None => MAX_THREADS_PER_GROUP.min(input.len()).min(limit),
        };
        if width == 0 || input.len().div_ceil(width) > MAX_GROUPS {
            return Err(MetalError::new(
                "dispatch bounds",
                "device or pipeline would require too many groups",
            ));
        }

        let byte_len = input.len() * size_of::<u32>();
        if byte_len > self.device.maxBufferLength() {
            return Err(MetalError::new(
                "buffer bounds",
                "fixture exceeds device buffer limit",
            ));
        }
        let input_buffer = match input {
            ComputeInput::Words(words) => Some(self.new_buffer(words)?),
            ComputeInput::SampledTexture { coords, .. } => Some(self.new_buffer(coords)?),
            ComputeInput::Texture(..) => None,
        };
        let mut words = vec![0xA5A5_A5A5; input.len()];
        let output_buffer = self.new_buffer(&words)?;
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| MetalError::new("commandBuffer", "command buffer creation failed"))?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            MetalError::new("computeCommandEncoder", "compute encoder creation failed")
        })?;
        encoder.setComputePipelineState(&pipeline);
        // SAFETY: All resources are live, initialized, and were allocated by
        // this device. The kernel uses one input at index zero and writes the
        // separate output buffer at index one. Both buffer offsets are zero.
        unsafe {
            match input {
                ComputeInput::Texture(texture, _) => {
                    encoder.setTexture_atIndex(Some(texture), 0);
                }
                ComputeInput::SampledTexture {
                    texture, sampler, ..
                } => {
                    encoder.setTexture_atIndex(Some(texture), 0);
                    encoder.setSamplerState_atIndex(Some(sampler), 0);
                }
                ComputeInput::Words(..) => {}
            }
            if let Some(buffer) = input_buffer.as_deref() {
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, 0);
            }
            encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 1);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(size(input.len()), size(width));
        encoder.endEncoding();
        complete_command(&command)?;
        let gpu_seconds = command.GPUEndTime() - command.GPUStartTime();
        let host_seconds = host_start.elapsed().as_secs_f64();
        // SAFETY: Metal copied the initialized words into distinct shared storage.
        // The GPU finished before the CPU touches that storage. Both ranges are
        // valid for byte_len bytes and don't overlap. A byte copy requires no
        // additional alignment and all u32 bit patterns are valid.
        unsafe {
            copy_nonoverlapping(
                output_buffer.contents().cast::<u8>().as_ptr(),
                words.as_mut_ptr().cast::<u8>(),
                byte_len,
            );
        }
        Ok(RunProfile {
            library_seconds,
            pipeline_seconds,
            gpu_seconds,
            host_seconds,
            output: words,
        })
    }

    fn compile_library(
        &self,
        source: &str,
        options: &MTLCompileOptions,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
        self.device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(options))
            .map_err(|error| {
                MetalError::new(
                    "newLibraryWithSource",
                    error.localizedDescription().to_string(),
                )
            })
    }

    fn compute_pipeline(
        &self,
        function: &ProtocolObject<dyn MTLFunction>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        self.device
            .newComputePipelineStateWithFunction_error(function)
            .map_err(|error| {
                MetalError::new(
                    "newComputePipelineStateWithFunction",
                    error.localizedDescription().to_string(),
                )
            })
    }

    fn pipeline_group_width_limit(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    ) -> usize {
        self.device
            .maxThreadsPerThreadgroup()
            .width
            .min(pipeline.maxTotalThreadsPerThreadgroup())
    }

    fn new_buffer(
        &self,
        words: &[u32],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let bytes = NonNull::from(words).cast::<c_void>();
        // SAFETY: Metal reads exactly size_of_val(words) bytes from the live,
        // initialized slice and makes its own copy within this call.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                bytes,
                size_of_val(words),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| MetalError::new("newBufferWithBytes", "shared buffer allocation failed"))
    }
}

fn validate_source(source: &str, entries: &[&str]) -> Result<(), MetalError> {
    if source.is_empty()
        || source.len() > MAX_SOURCE_BYTES
        || entries.iter().any(|entry| entry.is_empty())
    {
        return Err(MetalError::new(
            "source bounds",
            "expected nonempty entries and source of at most 64 KiB",
        ));
    }
    Ok(())
}

pub(crate) fn compile_options() -> Result<Retained<MTLCompileOptions>, MetalError> {
    let options = MTLCompileOptions::new();
    if !options.respondsToSelector(sel!(setMathMode:))
        || !options.respondsToSelector(sel!(setMathFloatingPointFunctions:))
    {
        return Err(MetalError::new(
            "MTLCompileOptions",
            "this Metal runtime lacks explicit safe and precise math options",
        ));
    }
    options.setMathMode(MTLMathMode::Safe);
    options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
    Ok(options)
}

fn find_function(
    library: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, MetalError> {
    library
        .newFunctionWithName(&NSString::from_str(entry))
        .ok_or_else(|| MetalError::new("newFunctionWithName", format!("missing entry {entry}")))
}

fn complete_command(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    command.commit();
    command.waitUntilCompleted();

    if command.status() != MTLCommandBufferStatus::Completed {
        let error = command.error().map_or_else(
            || format!("unexpected command status {:?}", command.status()),
            |error| error.localizedDescription().to_string(),
        );
        return Err(MetalError::new("waitUntilCompleted", error));
    }
    Ok(())
}

fn verify_shader_validation(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(), MetalError> {
    if std::env::var("MTL_SHADER_VALIDATION").as_deref() != Ok("1") {
        return Ok(());
    }

    if !pipeline.respondsToSelector(sel!(shaderValidation)) {
        return Err(MetalError::new(
            "shaderValidation",
            "the runtime lacks the pipeline state query",
        ));
    }
    // SAFETY: The selector exists and this newly created pipeline is private
    // to the current thread. The non-atomic property is read before encoding.
    if unsafe { pipeline.shaderValidation() } != MTLShaderValidation::Enabled {
        return Err(MetalError::new(
            "shaderValidation",
            "shader validation requested but inactive for the pipeline",
        ));
    }
    Ok(())
}

fn size(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}
