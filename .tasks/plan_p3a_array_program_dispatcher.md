# P3a: Production Array-Program Dispatcher

## Objective

Introduce the smallest production operation dispatcher needed to store homogeneous array and first-class-dimension
operations in one `ArrayProgramType` program. This increment establishes only the family boundary that P2d's
`ProjectedContext` already prototyped. It does not add a mixed operation, dimension gateway, shape operation, transform
policy, or lowering rule.

This split is required by the main cleanup plan's review budget: Phase 3 must be split per operation, and combining the
outer dispatcher with `dimension_size`, two data gateways, the data conversion, and dimension comparison would exceed
roughly 800 substantive lines and cross independently compilable concerns.

## Fixed Design

- Add one `ArrayProgramOperation<A>` stored family under `backends::array_programs` with exactly two variants:
  - the existing homogeneous `ArrayOperation<A>` family; and
  - the existing homogeneous `DimensionOperation<DimensionValue>` family.
- Do not flatten either homogeneous family into per-primitive outer variants. The outer dispatcher selects a value kind;
  the inner family remains the sole primitive dispatcher for that kind.
- Use ordinary `From` implementations for both lifts and implement `OperationProjection<ArrayType>` and
  `OperationProjection<DimensionType>` only to associate each member type with its inner family.
- Implement the complete `Operation<ArrayProgramType>` contract by projecting types and region interfaces into the
  selected homogeneous family, delegating once, and lifting results back. Wrong-kind diagnostics must come from one
  shared projection path.
- Keep projection zero-state. Do not add dimensions, source arrays, identity maps, operation classification tables,
  replay, or dependency reconstruction to the dispatcher or `ProjectedContext`.
- Region-free eager interpretation may delegate through the homogeneous family. Region-carrying execution remains an
  explicit P4 deferral and must fail at the existing projected-region boundary rather than partially interpreting a
  region or silently dropping it.
- Do not rename or split `ArrayOperation` in this increment. P3's per-operation migration will remove mixed/implicit
  shape operations from that existing family before the final primitive-family rename.
- Do not add `ArrayProgramValue` batching, differentiation, transposition, or XLA policy here; those remain owned by
  P5, P6, and P7 after mixed signatures stabilize.

## Implementation

- [x] Record P2d's integration and remainder reconciliation commits in the cleanup ledger.
- [x] Add `ArrayProgramOperation<A>` beside `ArrayProgramValue<A>` with concise family-level documentation and no
      compatibility alias.
- [x] Add the two standard `From` lifts and the two `OperationProjection` associations.
- [x] Add one reusable private type/interface projection path that:
  - [x] borrows and validates every input type;
  - [x] projects every attached region input and output type while preserving effects;
  - [x] delegates `infer_region_input_types` and `infer_output_types` exactly once; and
  - [x] lifts all returned types without intermediate semantic metadata.
- [x] Forward the remainder of the `Operation<ArrayProgramType>` contract by family variant, including names, region
      declarations, rendering, effects, identity renaming, and output semantic queries.
- [x] Add region-free homogeneous interpretation without per-primitive cases. If doing so requires a value-kind-specific
      `Value` implementation tower, semantic state, or eager array payload copies beyond the existing generic context
      contract, stop and treat the P2 projection abort criterion as fired.
- [x] Point `ArrayProgramValue<A>`'s rich execution domain at the new dispatcher only after the eager delegation path is
      complete; retain the constant-only dispatch domain used for coherence.
- [x] Re-export the dispatcher through the backend module and crate facade.

## Tests

- [x] Consolidate dispatcher tests in `backends::array_programs` and cover:
  - [x] array and dimension `From` lifts;
  - [x] `OperationProjection` family selection;
  - [x] array and dimension type inference;
  - [x] canonical wrong-kind diagnostics;
  - [x] region-interface projection and effect preservation using one existing array higher-order payload;
  - [x] identity renaming for dynamic array axes and dimension values;
  - [x] region-free eager interpretation for one array primitive and one dimension primitive;
  - [x] tracing both member kinds into one outer program with no implicit operands or regions; and
  - [x] unchanged zero-allocation borrowed/consuming eager value projection.
- [x] Run focused dispatcher/projection tests.
- [x] Run `cargo check -p ryft-core -p ryft-xla`.
- [x] Run the complete `ryft-core` library and doctest suites and the `ryft-xla` library suite.
- [x] Run formatting, diff hygiene, and changed-file Clippy auditing.

## Gates

- [x] The production dispatcher contains exactly two homogeneous family variants and no mixed operation.
- [x] Adding an ordinary primitive to either inner family requires no handwritten outer dispatcher case.
- [x] No production code matches individual array or dimension primitive variants to perform projection.
- [x] No new context/value wrapper or carrier-specific `Value` implementation is introduced.
- [x] The dispatcher and projected context contain no semantic state and create no implicit dependency.
- [x] The substantive diff remains within the review budget; split again before implementation if the complete tested
      increment would materially exceed it.

## Subsequent Phase 3 Increments

After P3a lands, add one reviewable mixed boundary at a time:

1. P3b: canonical `DimensionSizeOperation` (`array -> dimension`).
2. P3c: canonical `DimensionToScalarOperation` (`dimension -> scalar array`).
3. P3d: checked `DimensionFromScalarOperation` data gateway.
4. P3e: checked `DimensionFromVectorElementOperation` data gateway.
5. P3f: dimension comparison producing an ordinary Boolean scalar array.
6. P3g onward: reshape, broadcast, and the remaining shape-carrying operations, split per operation as required by the
   main plan.

## Review

Implemented the production `ArrayProgramOperation<A>` as a two-variant family dispatcher over the existing homogeneous
array and dimension families. One private projection path validates composite inputs and region boundaries, preserves
region effects, and lifts inferred types after a single homogeneous-family delegation. The remainder of the operation
contract forwards by family, including output provenance, zero classification, effects, identity renaming, and
rendering. Region-free eager replay uses one generic helper with stack storage for nullary, unary, and binary inputs;
region-carrying execution is rejected at the projected-region boundary and remains explicitly deferred to P4.

Focused tests cover both family lifts, type projection, wrong-kind diagnostics, region interfaces and effects, array and
dimension identity renaming, eager interpretation, region rejection, and one trace containing both member kinds with
only explicit operand edges. The existing allocation integration test continues to prove zero allocations and stable
payload pointers for borrowed and consuming eager value projection.

Verification passed:

- `cargo test -p ryft-core backends::array_programs::tests --lib`: 7 passed;
- `cargo test -p ryft-core --test test_array_program_projection_allocations`: 2 passed;
- `cargo check -p ryft-core -p ryft-xla`;
- `cargo test -p ryft-core --lib`: 967 passed;
- `cargo test -p ryft-core --doc`: 53 passed, 16 ignored;
- `cargo test -p ryft-xla --lib`: 396 passed, 1 ignored; and
- `cargo fmt --all`, `git diff --check`, and changed-file Clippy attribution.

The residual audit finds exactly the two intended production dispatcher variants, no `Mixed` variant, no per-primitive
projection match, no ambient dimension/source-array/replay state, and no new context or value wrapper. Clippy retains
the repository's inherited warning backlog but reports no warning in either P3a-owned backend file. The complete diff
is below the increment's approximately 800-substantive-line review ceiling. `dimension_size`, gateways, comparisons,
mixed shape operations, transforms, region execution, and lowering remain assigned to P3b+ and P4+ exactly as listed
above.
