# Interfaces

## gfx1030 compute fixtures

The first input is little-endian gfx1030 machine code for a compute fixture with optional statically resolved forward scalar branches. The synthetic ABI starts with full Wave32 EXEC and reserves vector general-purpose register (VGPR) 0 for the grid index used by buffer accesses. Resource operands select documented synthetic bindings. The parser must require explicit target identification and validate all lengths before allocating. Console containers and AMDHSA initialization require separate contracts.

The production decoder takes borrowed code bytes. Its output records typed instructions, operand classes, original offsets and byte lengths. Unknown encodings and unsupported forms fail with a byte offset. The reference interpreter takes the original bytes and input words and performs its own decoding. It shares no production instruction definitions.

The compiler lifts decoded instructions to typed operations and verifies register initialization, supported resource offsets and termination before generating MSL. The initial subset permits only lane-indexed 32-bit buffer accesses against two disjoint buffers with equal lengths. The input is read-only. Each invocation may write only its own output word. The compiler and reference check instruction boundaries, termination placement and buffer bindings throughout the stream. Register initialization and transfer state are checked on the path selected by compile-time known forward scalar branches. Dynamic branches, workgroup barriers and cross-lane operations remain unsupported.

The Wave32 execution mask (EXEC) starts full. A supported scalar move can
assign a known 32-bit value to its low half. Each bit selects the corresponding
logical lane, numbered by the one-dimensional grid index modulo 32. Partial
masks limit vector arithmetic and moves to selected lanes. These instructions
must write an initialized destination so inactive lanes can retain their values.
Empty EXEC suppresses vector arithmetic and moves. Reachable vector memory
operations require full EXEC. Forward branches on EXEC or the vector condition
code (VCC) test only their low halves. VCC must be initialized before use.

The native backend exposes `NativeCompute::new()` and `run_words(source, entry, input)` for trusted MSL. It binds input at index 0 and output at index 1, dispatches one invocation per word and returns owned readback after completion. `run_words_profiled` returns the same output plus measured library, pipeline, GPU and host durations for the benchmark harness; those timings are single submissions, not distributions. The caller runs generated MSL in a bounded child process. Raw MSL is a trusted developer input. Errors report the operation and backend detail. Across other stages, diagnostics carry a stage, error kind, optional byte offset and human-readable context.

## Synthetic guest experiment

The separate [X1–X4 driver](../experiments/x1/README.md) loads fixed-base
PE32+ fixtures, interprets a bounded x86-64 subset and dispatches synthetic
services. X3 and X4 validate fixed-size command records from read-only
guest bytes before issuing compute or draw work. Their command formats and
Metal shaders are authored for the experiment.

The multi-function guest fixture (X6) maps PE32+ sections with separate
read, write and execute permissions. Its interpreter tracks the expected
return slot and continuation for each call. A return from the authenticated
root frame ends execution.

The named-import fixture (X7) resolves imports for one synthetic module
before guest entry. An indirect call may invoke only a resolved slot in the
guest import address table. The handler validates the guest buffer and
operates on words in checked guest memory. The command prints the guest
exit value, while its host exit code reports whether loading and execution
succeeded.

With the Metal feature, the guest compute fixture (X8) adds the
`xenora_compute` import to the same loader. RCX points to a four-byte-aligned
readable and writable three-word buffer. RDX points to a readable aligned
16-byte record containing four little-endian 32-bit fields: command 1,
handle 0, input slot 0, and word count 3. Handle 0 names the buffer supplied
on this call and is consumed after a successful dispatch. A second submission
within the same guest run fails with a stale-resource fault.
The CPU validates both spans before creating a Metal context.
It selects the original translated XOR program, waits for completion,
writes the three output words back to guest memory, and returns 3.
The guest can read the result through its existing report import.

A general guest command stream must submit validated resource handles and ranges through an explicit address translation boundary. The graphics side owns mappings and the lifetime of in-flight views. Completion and fence signaling flow back across the command boundary. Guest addresses are never host pointers.

`NativeCompute::run_guest_words` takes a host-provided immutable resource
table and a synthetic guest view. The view selects an input slot, table index,
byte offset and byte length. The backend validates slot zero, little-endian
32-bit words, alignment, bounds and dispatch size before copying the selected
range into a Metal buffer. Slot one holds a separate output buffer. The call
returns owned words after command completion or a stage-specific error.

## PE32+ fixture startup and static import reports

`xenora_loader::startup::run_synthetic_startup` takes an original, trusted
fixture, guest memory addresses, a service registry and an entry instruction
budget. It maps and rebases the image, binds imports, initializes thread-local
storage, invokes its callbacks and runs the entry point. The fixture entry point
uses a Win64-aligned stack and terminates with HLT. The result includes the loaded
guest memory and the entry execution outcome.

`xenora probe --mode static` inventories ordinary PE imports without guest
execution. `missing_imports.csv` lists unresolved keys with their import address
table relative virtual addresses. The full inventory records whether each key is registered, its implementation
status, and the canonical module behind API-set aliases. Runtime-loaded modules
and services require separate analysis. The coverage matrix groups unresolved
imports by shared startup prerequisites. Static imports do not reveal call
order. The JSON report counts import slots, DLL and symbol keys, distinct
names with DLL-scoped ordinals, and canonical registered handlers separately.
The unresolved total includes explicitly tracked unsupported APIs and other
missing handlers. Existing stubs remain visible in the full inventory.

The synthetic process registry supports the current-process pseudo handle and
its process ID. The priority boost setter and getter record and query a guest
policy. They are partial until a guest scheduler applies the policy. The process
ID lookup is limited to the current-process pseudo handle. An unsupported
handle reports `ERROR_INVALID_HANDLE` through the guest error slot. The getter
rejects invalid guest output addresses with `ERROR_NOACCESS`.

Win32 references include Microsoft's documentation for
[GetCurrentProcess](https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentprocess),
[GetProcessId](https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocessid),
[SetProcessPriorityBoost](https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocesspriorityboost),
and [GetProcessPriorityBoost](https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesspriorityboost).

### Thread handles and virtual memory

The thread object model exposes one guest thread with a stable synthetic ID.
`OpenThread` issues rights-checked handles to that thread, and `CloseHandle`
invalidates real handles. A thread pseudo handle refers to the current synthetic
thread. The priority APIs store and query the thread's base-priority setting.
Thread creation, inheritance, debugger exceptions, and scheduling remain outside
this model.

`VirtualAlloc` supports combined reserve and commit for data-only pages.
`VirtualFree` supports `MEM_RELEASE` with the exact allocation base and a
zero size. The guest memory map tracks allocation ownership and bounds live
virtual allocations to 64 MiB. Decommit, executable pages, and console-specific
allocation lifecycle remain unsupported.

References include Microsoft's
[thread handle](https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-openthread),
[thread priority](https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadpriority),
[handle close](https://learn.microsoft.com/windows/win32/api/handleapi/nf-handleapi-closehandle),
[virtual allocation](https://learn.microsoft.com/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc),
and [virtual release](https://learn.microsoft.com/windows/win32/api/memoryapi/nf-memoryapi-virtualfree) contracts.

## Synthetic tiled textures

The texture crate reads original fixtures in an array-layer-major layout. Each
layer stores a consecutive mip chain. Within each mip, row-major 2×2 tiles store
four row-major Block Compression 1 (BC1) blocks, including slots beyond the
logical edge. The reader validates the full byte length, rejects padding
coordinates and can decode each logical block to sixteen RGBA8 texels.
The decoder uses bit replication to expand 5:6:5 endpoints and rounds integer
palette interpolation down. This layout is specific to Xenora fixtures. The block encoding follows the
[Direct3D BC1 description](https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression).

`TiledBc1::linearize` strips the tile padding from one selected subresource
while retaining complete compressed blocks at logical image edges.
`NativeCompute::read_tiled_bc1` checks the pixel budget and the Metal device's
BC texture capability, then uploads the selected blocks as a single
BC1 RGBA texture. A fixed Metal compute kernel reads each texel and returns
row-major RGBA8 words. The texture path uses authored MSL and takes its
input from the host fixture.

## Persistent guest graphics resources (X9)

A separate fixed-base PE32+ test profile exposes `xenora_resource`. RCX
points to a readable, four-byte-aligned request of five little-endian
32-bit words (operation, handle, offset, length and kind). RDX carries a
guest data address for uploads and readback. Other operations require it
to be zero. Handle values encode a slot and generation. Every guest run
owns a separate eight-slot table. Slots can be reused only with a new
generation. Guest addresses and handles never expose host pointers.

| Operation | Request data |
| --- | --- |
| 1, create buffer | Length in bytes and kind 1 (compute words) or 2 (draw vertices). RDX points to initial data. |
| 2, update buffer | Handle, byte offset and length. RDX points to source bytes. |
| 3, destroy resource | Handle. A queued snapshot retains its bytes after destruction. |
| 4, queue compute | Handle, byte offset and length. The selected view binds the original translated XOR compute fixture. |
| 5, complete and read back | Exact result length and a writable RDX destination. |
| 6, queue draw | Handle, byte offset, length 96 and program ID 1 (constant color) or 2 (interpolated color). The input contains three float4 clip positions followed by three float4 colors. Readback is 4×3 row-major RGBA8. |
| 7, create texture | Length of compressed data in bytes. RDX points to eight descriptor words followed immediately by the compressed data. |
| 8, queue texture read | Handle, array layer in the offset field and mip index in the length field. Completion returns row-major RGBA8. |

The texture descriptor contains width, height, mip count, layer count,
format (1 for BC1), usage (1 for sampling), row pitch and slice pitch.
The row pitch is aligned to the eight-byte BC1 block size. Each mip
occupies one slice and uses the same pitches. Slices appear in mip order
within each layer. The whole descriptor and payload must occupy one
readable guest span. Textures are immutable. This is an original linear
upload contract for the test profile. Opcode 8 uploads the selected
subresource and uses an authored Metal texel reader.

Buffers and compressed texture payloads are limited to 4096 bytes.
Queueing captures one immutable view. Completion validates the guest
destination before submitting GPU work and waits for Metal before writing
the result. An update or destruction after queueing cannot change the
captured bytes. Guest exit drops resources and any queued view. Data persists
across guest calls in host memory and is uploaded to Metal for execution.
Persistent device allocations and GPU work in flight across guest calls
are outside this interface.

### Translated offscreen stages

`compile_graphics` decodes two original gfx1030 byte streams and links
one vertex position export, an optional four-component parameter export,
and one pixel color export. Buffer 0 supplies positions to vertex vector
registers 0–3 and buffer 1 supplies colors to registers 4–7. The pixel
stage can consume parameter zero through matching, adjacent interpolation
phase pairs. This maps to center-perspective interpolation in Metal.
The vertex stage supports a bounded set of 32-bit floating-point add and
multiply forms. The graphics ABI also supports register and finite
literal moves. Unsupported stage instructions or export combinations
fail explicitly. The Xbox graphics ABI, hardware-specific floating-point
modes, image instructions, and additional varyings remain unsupported.

## Graphics draw and blend contracts

`NativeCompute::render_indexed` issues one real `drawIndexedPrimitives` call
per draw (up to 16) on a 16- or 32-bit index buffer with a start index and a
signed base vertex, into an offscreen RGBA8 target. A draw can sample an
original BC1 texture uploaded natively. Before encoding anything it rejects
malformed draws; execution checks each draw in turn.

`NativeCompute::render_blend` draws up to 16 flat-coloured triangles in order
into an offscreen RGBA8 target. Each draw has its own colour write mask and
either blending disabled or separate RGB and alpha blend factors and
operations. Colours are compared per channel against expectations derived
independently from the public Metal blend equations.

`NativeCompute::render_depth` draws up to 16 triangles in order with an
optional `Depth32Float` attachment, a chosen clear depth and a per-draw depth
compare function. Colour and the stored depth are both read back.

These passes are checked on the GPU with Metal validation against expectations
derived independently from the public semantics; synthetic fixtures only.

`NativeCompute::render_mesh` takes the same `IndexedPass` and draws it through
Metal's mesh render pipeline instead: one `drawMeshThreadgroups` call per draw
with one mesh threadgroup per triangle and three threads, without an object
stage. An authored MSL mesh function reads the index and vertex buffers and
applies the start index and base vertex; the indexed pass's fragment functions
shade it. Validation is the indexed pass's and runs first. A device outside the
Metal 3 GPU family then fails with the capability error
`MetalError::MESH_UNAVAILABLE`. Its GPU test compares every indexed case with
the same independent expectations, but it has not yet run on Apple hardware.
Translated RDNA 2 mesh, primitive or geometry shaders are unsupported; see the
[mesh-stage research notes](research/mesh-stages.md).
