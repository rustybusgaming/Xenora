# Mesh and geometry stages (milestone X13)

Milestone X13 covers mesh, geometry and additional compute shader capabilities
with verified Metal execution. It has only started. Nothing in it has run on
Apple hardware, and no translated RDNA 2 mesh, primitive or geometry shader is
supported. Evidence classes follow the project's scheme: A is measured on Apple
hardware, B is public vendor documentation, D is a vendor disclosure or
developer report, and U is unknown.

## What exists

`NativeCompute::render_mesh` draws an `IndexedPass` through Metal's mesh render
pipeline. An authored MSL mesh function emits one triangle per mesh threadgroup
and reads the index and vertex buffers itself. The pass reuses the indexed
pass's validation and fragment functions. Its GPU test,
`mesh_draw_known_answers` in `crates/metal/tests/native/mesh.rs`, compares
every indexed case with the expected images in
`crates/metal/tests/fixtures/indexed_expected.txt`. Those images were derived
independently of both draw paths, so no mesh code produces an expected value.
The test runs with `scripts/test-gpu.sh` and has not run yet.

The Rust side type-checks for `aarch64-apple-darwin` against the pinned
`objc2-metal` 0.3.2 bindings. The MSL has not been compiled, because it was
written on a host without the Metal compiler.

## Evidence

- API surface: the pinned bindings, generated from Apple's Metal headers,
  provide `MTLMeshRenderPipelineDescriptor` with object, mesh and fragment
  functions, `newRenderPipelineStateWithMeshDescriptor`, `setMeshBuffer`,
  `setMeshBytes` and `drawMeshThreadgroups`. Their header text says the object
  thread count is ignored without an object function. This is treated as class
  B for the API's existence only, pending a direct check against Apple's
  documentation. Behaviour on any device is unmeasured.
- No geometry stage: the render pipeline descriptors in those bindings set
  vertex, object, mesh and fragment functions and no geometry function, so
  geometry shaders would need emulation (class B, same source).
- Capability gate: `render_mesh` requires `MTLGPUFamily::Metal3`. That this
  family is the right gate for mesh render pipelines on every supported Mac is
  class U until checked against Apple's Metal Feature Set Tables.
- Console behaviour: the Xbox Series X|S mesh-shader ABI is unknown (class U).
  Microsoft's Xbox graphics documentation requires NDA-program access
  (CLAIM-XBOX-GDK-ACCESS); Xenora does not use restricted material.

## Research tasks

1. Run `scripts/test-gpu.sh` on Apple Silicon, M1 and later where available,
   and record each device name and result. Fix any MSL compile errors first.
2. Confirm which GPU families support mesh render pipelines in Apple's Metal
   Feature Set Tables, and adjust the `Metal3` gate if needed.
3. Find out from public AMD RDNA 2 and LLVM AMDGPU documentation how a gfx1030
   primitive or mesh shader allocates and exports vertices and primitives,
   before any decoder or translation work. Record what remains unknown.
4. Add per-primitive data and an object stage to the native pass, each with
   independently derived expectations.
5. Design geometry-shader emulation on mesh or compute passes.
6. The additional compute capabilities depend on milestone X12 (wave
   execution, LDS and barriers).
