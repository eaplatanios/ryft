# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added `Parameterized::try_map_parameters`, `Parameterized::try_map_named_parameters`, and
  `Parameterized::broadcast_to_parameter_structure`.
- Added a backend-neutral, typed program representation (i.e., an intermediate representation or IR) in
  `ryft_core::programs`:
  - `Program`s are immutable graphs of `Instruction`s over typed `Atom`s, organized into sealed `Region`s that model
    nested computations (e.g., branches, loop bodies, callees, and custom derivative rules), and are constructed and
    validated with `ProgramBuilder`s. `TypedProgram` exposes the flat type signature of programs and of the artifacts
    derived from them.
  - Operations declare their type inference and their effects (i.e., `Effects`, including `ReferenceEffect`s), types
    support dynamic type identities and refinements, and instructions carry source `Provenance`.
  - Mutable references with alias analysis and discharge into pure, state-passing programs.
  - Structural `ProgramStatistics`, a bounded `SpecializationCache` for retained artifacts (e.g., traced, lowered, or
    compiled programs), and per-region caching of structural transform results (i.e., `Transform` and
    `TransformCache`).
- Added `Domain`s and `Context`s in `ryft_core::contexts` as the shared protocol through which eager execution,
  staging, and transforms bind operations, along with:
  - Tracing of Rust closures into `Program`s in `ryft_core::tracing` via `trace` and `Trace::trace`, including nested
    tracing and named-axis bindings.
  - Captured runtime values in `ryft_core::captures` that are kept out of the IR through `CaptureReference`s and
    `ClosedProgram`s.
  - Interpretation in `ryft_core::interpretation` that replays programs eagerly via `Program::interpret` or through
    any other context (e.g., for restaging or transformation) via `Program::interpret_in_context`.
- Added composable program transforms driven by operation-owned rules:
  - **Batching** (i.e., vectorization) in `ryft_core::batching` via `batch` and `Batch::batch`, supporting mapped and
    replicated axes, named batch axes, and pluggable `BatchingPolicy`s.
  - **Differentiation** in `ryft_core::differentiation` via `differentiate_at`, which supports forward mode (i.e.,
    `jvp` and `linearize`), reverse mode via program transposition (i.e., `vjp`, `gradient`, and `value_and_gradient`),
    full `Jacobian` and `Hessian` materialization, holomorphic differentiation, auxiliary outputs, and
    non-differentiated captured inputs. Rematerialization (i.e., gradient checkpointing) is supported via
    `rematerialize` with configurable residual-saving and offloading policies.
  - **Partial Evaluation** in `ryft_core::partial` via `Program::partially_evaluate` and `Program::partition`,
    splitting known work from a residual program.
  - **Reference Discharge** via `Program::discharge_references`.
- Added arrays and array programs in `ryft_core::arrays`. `ArrayType`s model data types, shapes with static or symbolic
  bounded `Dimension`s, layouts, memory spaces, and shardings over `LogicalMesh`es and `DeviceMesh`es, which are
  defined in `ryft_core::arrays::sharding`. `Array` is an eager reference implementation, `ArrayReference` provides
  mutable array references with indexed and sliced views, and `ArrayIrValue` unifies arrays, dimensions, and references
  within one program.
- Added operations along with value capability traits in `ryft_core::operations`, which can be applied uniformly to
  eager values and to traced values, and which provide rules for interpretation and all of the above transforms.
  Related capabilities are also bundled for generic code (e.g., `ArithmeticOperations`, `ManipulationOperations`,
  `CollectiveOperations`, and `ArrayOperations`). The supported operations are:
  - **Constants:** `Constant`, `Zero`, `One`, `Fill`, `Iota`, `ZeroLike`, and `OneLike`, along with the dynamic-shape
    `DynamicZero`, `DynamicOne`, `DynamicFill`, and `DynamicIota`.
  - **Arithmetic:** `Sign`, `Neg`, `Abs`, `Add`, `Sub`, `SaturatingSub`, `Mul`, `Div`, `Rem`, `Pow`, `Sqrt`, and
    `Rsqrt`.
  - **Exponential, Trigonometric, & Special Functions:** `Exp`, `ExpM1`, `Log`, `Ln1p`, `LogAddExp`, `Logistic`, `Sin`,
    `Cos`, `Tan`, `Tanh`, `Atan2`, and `Erf`.
  - **Rounding & Extrema:** `Floor`, `Ceil`, `Round`, `Min`, `Max`, and `Clamp`.
  - **Comparisons & Logical:** `Compare`, `Not`, `And`, `Or`, and `Xor`.
  - **Complex Numbers:** `Complex`, `Real`, `Imaginary`, and `Conjugate`.
  - **Manipulation:** `Transpose`, `Reverse`, `Reshape`, `Broadcast`, `Pad`, `Concatenate`, `Slice`, `UpdateSlice`,
    `DynamicSlice`, `DynamicUpdateSlice`, `Gather`, `Scatter`, `Indexing`, `ConvertElementType`, `Cast`,
    `ReducePrecision`, and `TransferToMemory`, along with the dynamic-shape `DynamicReshape`, `DynamicBroadcast`,
    `DynamicPad`, `DynamicConcatenate`, `DynamicSliceWithDimensions`, `DynamicGather`, and `DynamicScatter`.
  - **Contractions, Reductions, & Sorting:** `Dot`, `RaggedDot`, `Reduce`, `ArgMax`, `ArgMin`, `Cumulative`,
    `Softmax`, `Sort`, and `TopK`.
  - **Quantization:** `BlockQuantize`, `BlockDequantize`, and `ScaledDot` for block-scaled (i.e., microscaling) formats
    such as NVFP4 and MXFP8, along with `block_scaled_dot`, which creates a differentiable block-scaled dot product for
    training.
  - **Random Number Generation:** `RngBitGenerator`, `DynamicRngBitGenerator`, and `Random`.
  - **Control Flow:** `Select`, `Condition`, `While`, and `Scan`.
  - **Dimensions:** `DimensionSize`, `DimensionFromScalar`, `DimensionToScalar`, `DimensionMin`, `DimensionMax`,
    `DimensionPow`, and `DimensionConstant`, with the remaining dimension arithmetic provided via the arithmetic
    capabilities.
  - **References:** `ReferenceNew`, `ReferenceRead`, `ReferenceWrite`, `ReferenceSwap`, `ReferenceAddUpdate`,
    `ReferenceAtomicAddUpdate`, and `ReferenceFreeze`.
  - **Sharding & Collectives:** `Reshard`, `ConstrainSharding`, `AxisIndex`, `ParallelReduce`, `ParallelVary`,
    `ParallelAllGather`, `ParallelSumScatter`, `ParallelPermute`, `ParallelAllToAll`, and `ParallelRaggedAllToAll`.
  - **Differentiation:** `StopGradient`, along with operations that support rematerialization and linear calls.
  - **Extensibility & Debugging:** `CustomCall`, `Assert`, `Print`, and `Tag`, along with composite operations that
    are created via `composite` and support custom JVP, VJP, and batching rules.
- Introduced more fine-grained error types like `ParameterError`, `DataTypeError`, `DimensionError`, `LayoutError`,
  `BroadcastingError`, `ShardingError`, `BatchingError`, and `DifferentiationError`.
- Added support for the `DataType::F6E3M2FN` and `DataType::F6E2M3FN` 6-bit microscaling floating-point data types.

## [0.0.2] - 2026-03-02

### Changed

- Significantly enhanced the `Parameterized` trait.

## [0.0.1] - 2026-02-22

### Added

- Initial release.

<!-- next-url -->
[0.0.1]: https://github.com/eaplatanios/ryft/compare/v0.0.1...HEAD
