# Xenora documentation

## Getting started

Start with the [project README](../README.md) for scope and build commands,
then read the [interface contracts](interfaces.md). Reproduce an
[original guest test program](../experiments/x1/README.md) or run the
[independent validation suite](../tests/validation/README.md) before changing
either path. The [changelog](../CHANGELOG.md) records notable changes.

## Architecture and technical design

- [Interface contracts](interfaces.md) — Translated shader stages, guest services, native command submission, and synthetic textures.
- [Fixture memory model](adr/0003-demo-memory-model.md) and [CPU boundary](adr/0004-cpu-boundary.md) — Buffer bindings, guest address translation, and dependency direction.

## Native Metal graphics

- [Translated offscreen stages](interfaces.md#translated-offscreen-stages) — Translated vertex and pixel stages for the bounded gfx1030 subset.
- [Blend and depth passes](interfaces.md) — Native blend and depth graphics passes with per-draw state.
- [Apple GPU evidence](research/apple-gpu.md) and [host lab record](research/host-lab.md) — Observed behavior and historical floating-point results.
- [Mesh and geometry stages (X13)](research/mesh-stages.md) — The native mesh-pipeline pass, its evidence and the open research tasks for mesh, geometry and further compute stages.

## RDNA 2 decoding and shader translation

- [Instruction coverage](shaders/instruction-coverage.md) — Desktop gfx1030 instruction forms, operand limits, and CPU and GPU results.
- [Semantic coverage](shaders/semantic-coverage.md) — Generated per-family counts of identified, declared, decoded, lowered and verified gfx1030 forms.
- [SOP2 scalar ALU](shaders/scalar-alu.md) — Supported scalar add, subtract and bitwise forms, their SCC rules and sources.
- [Semantic registry](shaders/semantic-registry.md) — How to add an instruction form: registry fields, validation, generation and the independence rule.
- [VOPC integer compares](shaders/vector-compare.md) — Wave32 compare forms, the VCC/EXEC contract, runtime `s_cbranch_vccz` and the Metal mapping.
- [Metal Shading Language generation](shaders/msl.md) and [independent CPU reference](shaders/reference-interpreter.md) — Translation rules and separate CPU execution models.

## Guest CPU and runtime experiments

- [Guest experiment guide](../experiments/x1/README.md) — Bounded x86-64 execution, executable loading, experimental services, native compute, and offscreen rendering.
- [Guest CPU research](research/cpu-boundary.md) — Public specifications and unresolved console requirements.
- [Synthetic import fixtures](../tests/compat_fixtures/README.md) — Reproducible PE32+ builds and expected service behavior.

## Testing, validation, and benchmarks

- [Engineering progress](progress.md) — Verification commands, metric definitions, and open release gates.
- [Test inventory comparison](testing/test-inventory.md) — Detect missing or newly ignored Cargo tests across commits.
- [Persistent resource and translated rendering acceptance](testing/x9-acceptance.md) — Pinned upstream tests, native ARM64 results and unsupported cases.
- [Baseline acceptance](testing/phase-02-baseline.md), [Phase 01 assurance](testing/phase01-assurance.md), and [independent reviews](testing/review-log.md) — Candidate-specific results and release findings.
- [Mutation testing](testing/mutation.md) and [benchmark protocol](testing/benchmarks.md) — Test sensitivity, benchmark inputs, and timing requirements.

## Research and engineering decisions

- [Public sources](research/sources.md), [RDNA 2 notes](research/rdna2.md), and [Xbox GPU evidence](research/xbox-gpu.md) — Specifications, evidence scope, and unknown console behavior.
- [Shader inputs](research/shader-containers.md) and [risk register](risk-register.md) — Input provenance and unresolved engineering risks.
- [Synthetic integration and upstream evidence](research/synthetic-integration.md) — Audited source snapshots and remaining commercial startup prerequisites.

## Security, licensing, and contributions

- [Threat model](threat-model.md) and [security reporting](../SECURITY.md) — Execution limits and disclosure instructions.
- [Legal scope](legal.md), [dependency licensing](licenses.md), and [third-party notices](../THIRD_PARTY_NOTICES.md) — Input policy, attribution, and redistribution findings.
- [Contribution guidelines](../CONTRIBUTING.md) — Original authorship, Developer Certificate of Origin sign-off, and evidence requirements.
- [Agent task queue](coordination/agent-tasks.md) — Areas, claiming, leases, upstream sync, verification and the security model for parallel contributors and coding agents.
- [Infrastructure QA ownership](coordination/infrastructure-qa-ownership.md) — Non-overlapping paths for infrastructure and QA tooling.
- [Agent queue smoke evidence](coordination/agent-queue-smoke.md) — Live Actions results from the disposable smoke repository.
- [Infrastructure QA tooling](testing/infrastructure/README.md) — Advisory orchestration, PR validation, benchmark analysis and release checks.
