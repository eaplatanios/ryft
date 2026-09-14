# Ryft Mosaic

Experimental Mosaic GPU compilation of portable `ryft_core::kernels` definitions. The adapter owns typed GPU target
contracts, lowering, compiler payloads and communication simulation. It emits binary Mosaic source for the existing
native runtime; it does not own an executable cache, CUDA launcher or PJRT client. TPU support is not enabled.

The GPU compiler maps each parallel logical grid point to one CUDA thread block and traverses sequential grid axes
inside that block. Threads share the work for each array result and synchronize before consuming one another's
values. Ordinary dot and sum use scalar accumulation loops. This baseline does not select tensor-core instructions.

Inputs require specialized shapes and dense row-major storage. The numerical subset supports Boolean, I32, U32,
I64, U64, F32 and F64 values, with operation-specific admission. Arrays, dimension values and reference views retain
their canonical core meanings. Arithmetic operand element types must agree. Masked loads branch before touching memory and use explicit padding; stores publish
only valid lanes. Bounded loops honor their semantic iteration limit and update their carried arrays simultaneously.
Checked scalar-to-dimension conversion emits a device trap on invalid input, reported through the native execution
failure path. XLA preserves these ordered assertions with a token chain; native buffer slots account for the
input and result tokens without interpreting them as arrays. Dimension arithmetic requiring other runtime assertions is rejected before native construction.

Asynchronous copies currently require whole, statically known global source roots and shared scratch destinations,
with matching 4- or 8-byte elements. Each thread issues its assigned copies, commits a group and waits before a
block-wide publication barrier. Cooperative work between an outstanding copy and its wait is rejected. The adapter
checks emitted copy groups and barriers with its deterministic synchronization simulator; core qualification retains
responsibility for logical initialization and reference lifetimes. The baseline does not support TMA, warpgroup or
cluster communication.

The compiler reports conservative shared-memory requirements for all live source allocations and temporary arrays.
`Target` and `KernelSchedule` constrain these allocations before compilation. Generated shared allocations have
explicit native alignment. The XLA integration checks exact compute capability, pinned CUDA/PJRT compatibility and
the physical argument layout; the native compilation provider validates the installed PTX assembler and driver.
The source configuration key includes the pinned XLA/JAX revisions and Mosaic serialization versions.

Use `Compiler::module` to inspect verified MLIR, or compile a `VerifiedKernel` to obtain binary source and its exact
hash. XLA execution is configured separately through `ryft_xla::kernels::MosaicGpuEmbedding` and the `mosaic-gpu`
feature. Macro-authored vector addition, reduction and tiled matrix multiplication are in
[`experimental/src/kernels.rs`](../experimental/src/kernels.rs), alongside an explicit asynchronous-copy fixture.

Implementation status and hardware qualification are tracked in [`plan-pallas.md`](../../plan-pallas.md), Phase 14.
