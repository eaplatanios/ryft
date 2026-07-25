# Symbolic dimensions P0 evidence freeze

This document freezes the evidence against which the symbolic-dimension architecture cleanup is reviewed. The
immutable archive remains the behavioral authority until a later increment deliberately replaces a behavior and
records the replacement here or in the cleanup ledger.

## Revisions and environment

| Role | Revision |
|---|---|
| Immutable feature archive | `770e77d001547c72150a44843c170ea6417ab41e` |
| Archive parent | `8105cfd26817ab728bb2799c889021f240345993` |
| Reviewed integration baseline | `20eefa3085e70a44995862ff0fc9986f80158c0d` |
| Reconciled mutable remainder | `9bcc73d7093a1a001c8c0539e5307851558ad9cd` |

The archive changes 142 paths relative to its parent: 51,762 insertions and 22,370 deletions. Relative to the reviewed
integration baseline, it changes 147 paths: 52,308 insertions and 24,415 deletions. The reconciled remainder changes
141 paths relative to integration: 51,552 insertions and 23,240 deletions.

Measurements were taken on a 12-core Apple M2 Max MacBook Pro with 96 GiB of memory, macOS 26.4.1 (25E253), and
`arm64` Darwin 25.4.0. The Rust toolchain was `rustc 1.93.1 (01f6ddf758 2026-02-11)`, LLVM 21.1.8, and Cargo 1.93.1.
Git was Apple Git 2.50.1 (155).

Archive measurements used a read-only extraction at `/private/tmp/ryft-p0-archive-770e77d`. Clean builds used
independent empty target directories. Core checks used the default `ryft-core` feature set. IR and compilation
benchmarks used release builds of `ryft-xla` with `benchmarking ndarray`. Runtime compilation measurements used the CPU
backend, a `[1024]` input, three iterations, and `sin(x * x + x)`.

The reviewed integration baseline required one compatibility correction before its release benchmark feature could
compile: `tracing_v2::benchmarking` still called the removed `RegionRef::region_ref` spelling rather than `with_id`.
That one-line S1 call-site correction is part of P0 and changes no dimension semantics.

## Code and generated-size baseline

`tokei` reports code, comments, and blanks separately. The production/test split below is physical Rust lines: for
source files with inline unit tests, lines before the canonical `#[cfg(test)] mod tests` boundary are production and
lines from the module onward are tests. Standalone test directories remain tests. Generated expansion is reported
separately and is never included in hand-written source counts.

| Area | Archive files | Archive code | Archive comments | Archive blanks | Integration files | Integration code | Integration comments | Integration blanks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ryft-core/src` | 130 | 99,662 | 4,153 | 9,226 | 123 | 78,838 | 4,101 | 7,562 |
| `ryft-core/tests` | 5 | 1,706 | 114 | 204 | 3 | 1,367 | 97 | 162 |
| `ryft-xla/src` | 49 | 37,329 | 806 | 2,826 | 34 | 34,271 | 976 | 2,598 |
| `ryft-macros/src` | 12 | 4,895 | 146 | 522 | 12 | 4,766 | 146 | 514 |
| `ryft-macros-tests` | 18 | 2,677 | 24 | 490 | 18 | 2,566 | 24 | 468 |
| Python source | 12 | 2,006 | 0 | 493 | 11 | 1,897 | 0 | 457 |
| Python tests | 6 | 407 | 0 | 126 | 5 | 367 | 0 | 117 |

| Area | Archive production physical lines | Archive inline-test physical lines | Integration production physical lines | Integration inline-test physical lines |
|---|---:|---:|---:|---:|
| `ryft-core/src` | 81,201 | 46,899 | 64,453 | 39,816 |
| `ryft-xla/src` | 30,928 | 13,104 | 24,740 | 15,792 |
| `ryft-macros/src` | 4,709 | 1,782 | 4,590 | 1,762 |

The archive's three `backends/array_programs` files alone contain 4,800 pre-test physical lines: 2,382 in `mod.rs`,
1,754 in `batching.rs`, and 664 in `differentiation.rs`. This is the primary special-purpose adapter budget P2, P5,
and P6 must retire.

`cargo expand -p ryft-core --lib` produced:

| Revision | Lines | Words | Bytes | Structural tokens |
|---|---:|---:|---:|---:|
| Archive | 179,423 | 503,415 | 8,774,099 | 1,130,479 |
| Integration | 134,855 | 400,243 | 6,673,970 | 852,279 |

Structural tokens were counted by parsing the expansion with `proc_macro2`, counting every leaf token plus the opening
and closing delimiter of each group. This avoids treating formatting changes as semantic growth.

## Compile time, memory, and binary size

All measurements used `/usr/bin/time -l`.

| Revision | Clean `cargo check -p ryft-core` | Peak RSS | Immediate incremental check | Incremental peak RSS |
|---|---:|---:|---:|---:|
| Archive | 14.56 s | 858,161,152 B | 0.17 s | 66,420,736 B |
| Integration | 11.05 s | 696,647,680 B | 0.17 s | 65,388,544 B |

The integration check emits the pre-existing ambiguous `arrays` glob re-export warning assigned to P9.

The archive's clean release `ryft-xla` IR-benchmark build took 96.38 s, peaked at 3,287,367,680 B, and produced a
58,840,224-byte binary. The corrected integration binary is 57,564,144 bytes. Its observed 95.09 s build is not a
clean paired datum because dependencies were retained after the feature-gated source error; it must not be used as a
compile-time acceptance comparison.

## Graph-size baseline

Archive and corrected integration emitted byte-for-byte equivalent benchmark summaries for all runnable cases.

| Case | Surface | Instructions | Nested regions | Maximum depth | Raw IR bytes | Raw IR lines |
|---|---|---:|---:|---:|---:|---:|
| `scalar_bilinear_sin_jit` | JIT | 3 | 0 | 2 | 97 | 5 |
| `scalar_bilinear_sin_vjp_pullback` | VJP pullback | 4 | 0 | 2 | 143 | 6 |
| `scalar_quartic_plus_sin_grad` | gradient | 15 | 0 | 7 | 383 | 17 |
| `scalar_quartic_plus_sin_value_and_grad` | value and gradient | 18 | 0 | 7 | 460 | 20 |
| `shard_map_basic` | program | 1 | 1 | 1 | 405 | 11 |
| `shard_map_matmul` | program | 1 | 1 | 1 | 652 | 11 |
| `grad_around_shard_map` | program | 3 | 2 | 2 | 960 | 17 |
| `nested_shard_map` | program | 1 | 2 | 1 | 713 | 15 |
| `scalar_quartic_plus_sin_linearize_pushforward` | linearize pushforward | 11 | 0 | 7 | 924 | 21 |

`shard_map_grad_inside` exits with code 101 on both revisions. Its diagnostic is:

```text
shard_map grad-inside IR benchmark should trace the inner gradient: gradient output must be a rank-0 scalar but got
f32[2][sharding={mesh<['x'=4:manual]>, [{'x'}], varying_manual={'x'}}]
```

P6/P10 must either make the case runnable or explicitly replace it with a valid equivalent. It is not permissible to
drop the case silently.

## Runtime smoke baseline

These are three-sample smoke measurements, not statistical acceptance thresholds. P10 must rerun enough iterations to
establish useful distributions.

| Metric | Archive | Integration |
|---|---:|---:|
| Cold trace | 507,167 ns | 485,500 ns |
| Cold lower | 4,444,917 ns | 3,919,958 ns |
| Cold compile | 48,461,667 ns | 47,597,583 ns |
| Compile-cache compilation duration | 48,314,084 ns | 47,446,542 ns |
| Warm dispatch p50 / p95 | 38,833 / 42,875 ns | 11,042 / 27,583 ns |
| Enqueue p50 / p95 | 8,833 / 19,584 ns | 10,500 / 51,875 ns |
| Synchronized p50 / p95 | 24,917 / 25,375 ns | 38,334 / 49,584 ns |

## Allocation baseline

The archive's `dimension_allocation_counts` integration test passed all 11 cases with one test thread:

| Case | Allocations |
|---|---:|
| Clone dynamic shapes | 0 |
| Clone static shapes | 0 |
| Construct static array types | 0 |
| Distinct dimension identity proofs | 0 |
| Dynamic callee exact certificate | 0 |
| Dynamic callee fresh alpha-equivalence | 8,000 |
| Dynamic program tracing | 103 |
| Eager array transpose | 10 |
| Shared dimension identity proofs | 0 |
| Static callee preflight | 0 |
| Static program tracing | 98 |

The 8,000-allocation alpha-equivalence case is a regression ceiling, not a target. P1 and P10 must substantially reduce
it. P2 adds a separate zero-allocation projection test over a large eager array payload.

## Operation-family and contract inventory

The archive contains:

| Family | Variants | Classification |
|---|---:|---|
| Complete homogeneous `ArrayOperation` | 82 | Transitional array-only family containing shape and region operations |
| `ArrayPrimitiveOperation` | 56 | Correct homogeneous array-only inner family |
| `DimensionOperation` | 14 | Correct homogeneous dimension-only family |
| `MixedArrayDimensionOperation` | 30 | Genuinely mixed or region-polymorphic family |
| Outer `ArrayProgramOperation` | 3 | Array primitive, dimension primitive, or mixed storage dispatcher |

A bounded multiline syntactic inventory of explicit production impl headers found 88 `Operation<ArrayType>`
implementations, 41 `Operation<ArrayProgramType>` implementations, and eight `Operation<DimensionType>`
implementations. These counts are a growth/regression baseline, not the semantic classification: macro-generated and
blanket implementations make raw impl counts less authoritative than the concrete-payload inventory below.

The 56 array-only primitives are: zero-like, one-like, constant, abs, neg, add, sub, mul, div, sin, cos, atan2, exp,
log, sqrt, rsqrt, tanh, logistic, erf, pow, sign, floor, ceil, round, maximum, minimum, remainder, not, and, or, xor,
complex, conjugate, real, imaginary, dot, scaled dot, dot-product attention, dot-product-attention backward, sort,
collective, ppermute, axis index, transpose, scatter, update slice, dynamic update slice, compare, select, convert
element type, transfer to memory, reshard, sharding constraint, stop gradient, tag, and print.

The 14 dimension primitives are constant; add, subtract, clamped subtract, multiply, floor divide, remainder, minimum,
and maximum; compare; and equal, less-than-or-equal, divisible-by, and nonzero requirements.

The 30 mixed variants are zero, one, fill, iota, broadcast, scalar-to-dimension, dimension-to-scalar,
checked-scalar-to-dimension, dimension size, dimension compare, reshape, slice, slice scatter, dynamic slice, pad,
gather, concatenate, reduce, RNG bit generation, all-gather, psum-scatter, all-to-all, custom call, condition, while,
scan, custom JVP, custom VJP, custom-VJP tangent, and rematerialize. Condition, while, scan, custom JVP, custom VJP,
custom-VJP tangent, and rematerialize are region-polymorphic.

Twelve payloads have erroneous dual homogeneous/composite operation contracts: broadcast, concatenate, custom call,
dimension size, dynamic slice, gather, pad, reduce, reshape, RNG bit generation, slice, and slice scatter. P3 removes
their homogeneous contract.

Four generic constructor implementations overlap array-program-specific instantiations: zero, one, fill, and iota.
Their explicit destination is:

- Fully static public construction remains the homogeneous constructor.
- Transform-generated geometry uses structural zero/one or `zero_like`/`one_like` when an operand supplies it.
- Dynamic public construction uses one mixed shaped-constructor wrapper with explicit dimension operands.
- Eager, tracing, PE, batching, differentiation, and lowering all consume that one canonical signature.

Select, stop-gradient, and the test-only nullary operation are benign parameterized multi-contract payloads rather than
mixed operations. P8 preserves the parameterization without giving the same concrete payload two material contracts.

## Complete homogeneous-family consumers

Every production use of complete `ArrayOperation` has this destination:

| Consumer | Current reason | Destination |
|---|---|---|
| `backends/arrays.rs` | Reference array operation family and eager rules | Retain only as the public/reference homogeneous family; remove mixed members in P3/P8 |
| `backends/array_primitives.rs` | Lossless lift of 56 array-only primitives | Replace with generated member lift in P2 |
| `backends/array_programs/mod.rs` | Composite dispatch, replay, and projection | Replace with projected context and canonical mixed signatures in P2/P3 |
| `backends/array_programs/batching.rs` | Array delegation plus dimension recovery | P2 projection, P3 operands, P5 policy |
| `backends/array_programs/differentiation.rs` | Array delegation plus transpose recovery | P2 projection, P3 operands, P6 residuals |
| XLA experimental operations/domains/lowering | Backend dispatch and lowering | P7 typed member and mixed lowering |
| Core transform/compilation modules | Generic bounds and test harnesses | P8 generated dispatch; no shape policy in generic machinery |
| Macros and macro tests | Derived dispatch surface | P8 derive cleanup and compile fixtures |

No independent production consumer gets a replacement complete homogeneous family.

## Context views and hidden reconstruction

The archive has 270 `ArrayProgramProjection` matches in 11 files, 87 `ArrayContextView` matches in six files, and 17
`DimensionContextView` matches in four files. It also has 13 `with_dimensions`, four `with_source_array`, and three
`bind_replayed` matches. Matches inside tests and operation-builder methods were separated from context-view
construction during the audit.

| Path | Why it currently needs a view | Explicit replacement |
|---|---|---|
| XLA experimental domain replay | Bind a homogeneous operation into the outer builder | P2 zero-state projected context binds the lifted operation directly |
| Composite eager/tracing/PE dispatch | Project an outer array or dimension value | P2 borrowed projection for reads and consuming projection for ownership transfer |
| `with_source_array` tracer and partial-tracer dispatch | Recover dynamic extents from a source array side channel | P3 explicit dimension operands already present in the outer SSA graph |
| `with_dimensions` batching paths | Make transformed extents discoverable to later helper operations | P3/P5 ordinary dimension SSA operands and batching policy |
| `with_dimensions` JVP/transpose paths | Make primal extents discoverable while replaying helper operations | P3/P6 explicit operands or declared transform residuals |
| `bind_replayed` | Import homogeneous nested operations and lift results | P2 one generated inner-operation lift contract |
| `DimensionContextView` | Run host dimension arithmetic while retaining the outer graph | P2 generic projected context; eager dimension math remains host integer math |

The unrelated `ReshapeOperation::with_dimensions` builder stores `transpose_dimension_variables`; it is not a context
view. P6 deletes that witness and replaces it with ordinary transpose residuals.

## Transform and residual inventory

### Batching

The special cases in `array_programs/batching.rs` are projection/lifting, explicit dimension replication, dynamic
reshape/broadcast/slice/pad, concatenate/reduce, shape-changing collectives, RNG scan, and dimension-bearing
condition/while/scan regions. P2 owns generic projection, P3 owns signatures, and P5 owns the policies:

- Dimension values are replicated by default.
- A mapped shape authority is rejected until Ryft has an explicit ragged model.
- Region-carried dimension values are ordinary structural carries.
- No transformed operation may recover a dimension by searching arrays or a context-side vector.

### Differentiation and transposition

The special cases in `array_programs/differentiation.rs` are dimension no-tangent/objective rejection, primitive
projection, the mixed shape operations, collective inverses, and dimension-bearing control-flow regions. P6 owns the
policy that dimensions are nondifferentiable structural values and owns these explicit primal residuals:

| Transpose | Required primal information | Explicit replacement |
|---|---|---|
| Reshape | Input dynamic extents needed by the inverse reshape | Residual dimension SSA values; delete `transpose_dimension_variables` |
| Concatenate | Each input extent on the concatenation axis | Residual extent tuple used to slice the cotangent |
| Reduce/mean | Reduced extents, including the mean scale | Residual dimension values or existing explicit operands |
| Slice/dynamic slice | Original geometry, starts, and sizes | Residual start/size/extents used by scatter transpose |
| Pad | Operand/output extents and padding geometry | Residual extents used by inverse crop/interior handling |
| Gather | Operand shape and slice sizes | Residual shape values used by scatter transpose |
| Broadcast | Input extents that determine reduction axes | Existing explicit dimension operands retained as residuals |
| Shape-changing collectives | Inverse partition geometry | Existing explicit dimension operands retained as residuals |

### Partial evaluation

Dimension arithmetic, comparisons, requirements, and simplification currently live in `backends/dimensions.rs`; outer
dispatch and projected partial tracers live in `array_programs/mod.rs`. P2 moves projection into generic member
machinery. P4 retains the interval/divisibility transfer core, folds proven requirements, rejects disproven
requirements with current diagnostics, and preserves inconclusive requirements as ordered assertion effects.

## Dimension-operand collector inventory and schema

The archive has 72 `runtime_dimension_variables` matches in 13 files. The central helper is in
`operations/constants/mod.rs`, but independent collectors and copied validation exist in broadcast, reshape, slice,
slice scatter, dynamic slice, pad, gather, reduce, RNG, custom call, collectives, the reference backend, composite
eager/replay, batching, differentiation, and XLA lowering.

P3 replaces all collectors with one typed schema per canonical mixed operation:

| Operation class | Ordered dimension segment |
|---|---|
| Dynamic constructors | Dynamic output axes in output-axis order |
| Broadcast | Dynamic output axes in output-axis order |
| Reshape | Dynamic output axes only; transpose information is a transform residual |
| Slice | Dynamic starts followed by dynamic sizes in axis order |
| Slice scatter | Dynamic starts, sizes, and required operand/output extents in declared segment order |
| Dynamic slice | Dynamic sizes in axis order |
| Pad | Dynamic result extents derived from the operand and padding configuration |
| Gather | Dynamic slice sizes in index-map order |
| Concatenate | Dynamic input/result extent needed on the concatenation axis |
| Reduce/mean | Dynamic reduced input extents in axis order |
| RNG bit generation | Dynamic output axes in output-axis order |
| Shape-changing collectives | Dynamic result axes in operation-defined axis order |
| Custom call | Dynamic axes of each output, flattened by output then axis |

The schema owns count, order, kind, identity, bounds, and typed operand views. Operation inference, eager execution,
transforms, and lowering consume the same schema. P3's residual search must find no ad hoc collector consumer.

## Eager projection ownership decision

The reference `Array` stores `values: Vec<Scalar>` and its manual `Clone` deep-copies that vector. The archive's
borrowed `ArrayProgramProjection::project_array` returns an owned `A` via cloning, so ordinary eager projection can
copy the complete array payload.

P2 will implement both:

- A borrowed projection path, expressed with a GAT/reference view, for inference and eager operations that only read
  their inputs.
- A consuming projection path that transfers the member out of the storage sum when ownership is available.

P2 will not first change `Array::values` to `Arc<Vec<Scalar>>`. Borrowing fixes the projection defect without changing
reference-array storage semantics. An immutable shared payload is a separate optimization only if measurement after P2
finds unavoidable large-array clones elsewhere. The gate is zero allocations and zero `Scalar` copies when projecting
a large eager array.

## Diagnostic and prover acceptance floor

Requirement diagnostics must retain the requirement text and observed named values. Frozen examples include:

```text
left == right; observed left=12, right=8
left <= right; observed left=12, right=8
left % right == 0; observed left=12, right=5
right > 0 for divisibility; observed left=12, right=0
elements % alignment == 0; observed elements=25, alignment=8
```

Bounds failures retain the actor and interval, for example a value `3` outside `[4, 16)`. Multiple requirements fail
deterministically at the first ordered assertion. Identity closure retains actor-named messages, including duplicate
definition and unbound-reference diagnostics such as:

```text
operation `read_dimension` output defines identity batch more than once in this region
operation `reference_dimension` output type references identity length without consuming or defining it
```

P1 and P4 must capture the complete exact strings in dedicated tests before changing their owners.

Exact rendered graph and diagnostic text remains frozen in the immutable archive through these named golden tests:

- Static and direct dynamic: `test_array_program_dynamic_reshape_with_explicit_dimension_operands`,
  `test_array_program_dimension_size_tracing_repeated_readers_and_import`, and
  `test_dimension_program_tracing_rendering_and_import`.
- Derived dynamic and gateway authority: `test_array_program_data_dependent_extent_controls_multiple_bounded_outputs`
  and `test_array_program_dimension_data_gateways_tracing_and_import`.
- Control flow: `test_array_program_condition_tracing_import_and_eager_execution` and the scan/while structural-value
  differentiation tests.
- Batching: `test_array_program_batching_stages_one_composite_graph`,
  `test_array_program_batching_broadcast_preserves_explicit_dynamic_extent`, and
  `test_array_program_batching_dynamic_slice_varying_indices_uses_explicit_batch_extent`.
- Differentiation: `test_array_program_staged_dynamic_reshape_jvp_reuses_dimension_primals`,
  `test_array_program_dynamic_reshape_linearization_threads_dimension_residual_into_pullback`, and
  `test_array_program_concatenate_transpose_threads_dynamic_non_concatenated_extent`.
- Assertions: `test_dimension_requirement_tracing_rendering_and_import`,
  `test_array_program_dimension_requirement_static_failure_preserves_observed_actors`,
  `test_dimension_requirement_order_is_deterministic`, and
  `test_array_program_dimension_requirement_simplification_preserves_residual_diagnostics`.

Later phases copy these fixtures into their increments before deleting an archived owner. Behavioral equivalence means
the complete asserted string and observed-value order, not merely the same success/failure disposition.

The graph-level abstract interpreter must preserve the archive's probe disposition:

| Probe | Before `(instructions, assertions)` | After |
|---|---:|---:|
| P3 | (5, 2) | (3, 1) |
| P3 negative | (5, 2) | (5, 2) |
| P4 | (4, 2) | (2, 1) |
| P5 | (3, 3) | (2, 2) |
| P8 | (4, 2) | (2, 1) |
| P8 remainder spelling | (6, 2) | (4, 1) |
| P11 | (2, 1) | (0, 0) |
| P11 negative | (2, 1) | (2, 1) |
| P12 | (3, 1) | (0, 0) |
| P13 | (7, 2) | (3, 1) |
| Round-3 long-chain probe | (212, 61) | (60, 60) |

The acceptance tests are:

- `test_dimension_ssa_requirement_proof_probes_and_negative_controls`
- `test_dimension_ssa_congruence_transfer_handles_zero_factors_and_wide_residues`
- `test_array_program_dimension_requirement_proof_inventory`
- `test_array_program_dimension_requirement_simplification_preserves_residual_diagnostics`

Any case not proven after migration must be explicitly triaged to a runtime assertion, with the residual assertion
count recorded and justified.

## Code ownership map

| Concern | Owner after cleanup |
|---|---|
| Leaf identity, bounds, and refinements | `types` and generic program closure (`P1`) |
| Array/dimension member projection and lift | Generic program-domain machinery (`P2`) |
| Array and dimension eager values | Their reference backend modules (`P2`, with no outer-graph knowledge) |
| Dimension arithmetic and requirements | Dimension primitive operation family and abstract interpreter (`P2/P4`) |
| Mixed shape signatures and operand schemas | Owning operation modules (`P3.*`) |
| Batching policy | Batching transform (`P5`) |
| Nondifferentiability and transpose residuals | Differentiation transform (`P6`) |
| Runtime assertions and StableHLO/XLA operands | XLA backend (`P7`) |
| Operation-family dispatch generation | Derive macros (`P8`) |
| Public aliases, imports, and compatibility deletion | Core public surface (`P9`) |
| Performance and allocation acceptance | Cross-backend benchmark/test infrastructure (`P10`) |

## P0 gate result

Every known dual contract, hidden reconstruction path, transform special case, collector, and transpose witness now has
an explicit destination. The leaf-only `Dimension` decision remains deliberate: it avoids expression trees, scopes,
substitution, witnesses, and a second symbolic language. Arithmetic relationships live in ordinary dimension SSA;
requirements are proven by the graph interpreter or remain visible runtime assertions.

P1 may begin only after this evidence increment is reviewed and landed. Later increments may not weaken a frozen
behavior silently; they must update the operation migration matrix, disposition ledger, and the relevant acceptance
measurement.

The immutable archive library suite passes 1,035 tests and has one ignored test. The ignored
`test_array_program_batching_while_batch_varying_predicate_widens_carried_state` documents a real gap: composite while
batching ignores a batch-varying predicate. P5 must resolve the gap or retain an explicit rejection; it may not inherit
the silent behavior. The reviewed integration baseline passes all 913 library tests.
