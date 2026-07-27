# P3b: Canonical First-Class Dimension Size

## Objective

Introduce one canonical `DimensionSizeOperation` whose only semantic contract is:

```text
array --dimension_size(axis)--> dimension
```

This increment must make an array axis extent an explicit first-class dimension SSA value in an
`ArrayProgramType` program. It must not retain or recreate the archived homogeneous
`Operation<ArrayType>` contract that returned a rank-zero integer array. Ordinary data materialization remains the
separate `DimensionToScalarOperation` increment in P3c.

P3b is deliberately limited to this one mixed boundary. Data gateways, comparisons, shape-carrying operations,
composite transform policy, and backend lowering remain in their named later increments.

## Audit Findings and Rejected Archive Behavior

- The reviewed integration tree contains no production `DimensionSizeOperation`; the only implementation is on the
  mutable archive/remainder branch.
- The archived payload implements two materially different contracts:
  - `Operation<ArrayType>` returns a rank-zero `i64` array; and
  - `Operation<ArrayProgramType>` returns a first-class dimension.
  That dual contract is the deficiency P3b removes rather than ports.
- The archived operation stores only an optional result identity and validates it directly against the type seen
  during each inference call. That rejects a valid eager replay when the staged input axis is dynamic but the concrete
  runtime array refines it to a static extent. The production operation must retain the selected declared axis
  dimension and accept a compatible refinement.
- The archived `OutputIdentityRole` hook is obsolete. Structural type-identity positions already classify a
  `DimensionType` result as a definition; when its identity occurs on the array operand, region closure recognizes it
  as a forwarded definition. Repeated readers are therefore ordinary SSA edges, not operation-specific identity
  metadata.
- The archived `RUNTIME_DIMENSION_DATA_TYPE` and rank-zero result path belong only to the data conversion and must not
  appear in P3b.
- The archived gateway operations, batching rule, differentiation/transposition shells, and XLA lowering are outside
  this increment and must not be copied into the new module.

## Fixed Design

### Public operation and capability

- Add `operations::dimensions::dimension_size`, following the same module, documentation, export, and test structure
  as the existing dimension operation modules.
- Define `DIMENSION_SIZE_OPERATION_NAME` and a nominal `DimensionSizeOperation`.
- Put the complete user-facing semantics and executable example on a `DimensionSize` capability trait. The operation
  type documentation refers to that capability instead of duplicating the explanation.
- `DimensionSize::dimension_size(axis)` returns the selected axis size in the representation chosen by its `Output`
  parameter. Composite program values return a first-class dimension in their parent carrier, while concrete array
  backends may return a host extent; neither path returns integer array data.
- Give context-carrying `Value<Type = ArrayProgramType>` implementations one blanket staging implementation. It:
  1. projects the receiver's array type with the canonical borrowed `TryFrom`;
  2. constructs `DimensionSizeOperation` from that type and the requested axis; and
  3. binds the mixed operation once through the value's dispatch context.
- Give `ProjectedValue<ArrayType, V>` the same capability with `Output = V`, so results of projected array operations
  can feed shape computation without an explicit adapter conversion or a second wrapper type.
- Use the same `DimensionSize<Output>` capability for eager backend extraction: array backends select
  `Output = usize`, while composite values select their parent first-class-dimension carrier. The backend
  implementation reads one checked host extent and does not create an SSA identity or expose a second program
  operation contract.
- Implement `DimensionSize<usize>` for the reference `Array` backend. Implement `DimensionSize` for
  `ArrayProgramValue<A>` when `A: DimensionSize<usize>`, producing `ArrayProgramValue::Dimension`.
- Do not add `DimensionSizeOperation` to `ArrayOperation<A>`; the homogeneous reference value implements only the
  shared semantic capability that the composite eager implementation consumes.

### Operation payload and inference

- Make invalid operation payloads unrepresentable: the sole public constructor takes an `&ArrayType` and an
  `Into<Axis>`, normalizes the axis immediately, and constructs all result metadata.
- Store only the semantic facts inference needs:
  - the normalized nonnegative axis;
  - the selected declared `Dimension`; and
  - the resulting `DimensionType`.
- For a dynamic selected axis, reuse its `DimensionVariable` in the result type. This makes the graph edge explicitly
  define the same extent identity referenced by the input array axis.
- For a static selected axis, create one fresh result variable with the selected dimension's exact half-open bounds.
  Reject an extent above `MAX_DIMENSION_EXTENT`; on narrower hosts, preserve the existing representable exact-maximum
  convention through `Dimension::bounds()`.
- `Operation<ArrayProgramType>::infer_output_types` must:
  1. reject attached regions and require exactly one input;
  2. require that input to be the array member through the canonical borrowed conversion;
  3. require the normalized axis to remain valid;
  4. require the actual selected dimension to refine the declared selected dimension; and
  5. return exactly the stored first-class `DimensionType`.
- The refinement check is directional: a declared dynamic axis may execute against an in-bounds static axis, while an
  unrelated dynamic identity, an out-of-bounds static extent, a changed static extent, or an invalid rank/axis is
  rejected.
- Identity renaming applies to both the selected declared dimension and the result type in one simultaneous renaming.
  Rendering uses the canonical normalized axis and no data-type field.

### Mixed outer dispatch

- Add exactly one genuinely mixed `DimensionSize(DimensionSizeOperation)` variant to
  `ArrayProgramOperation<A>`, plus its ordinary `From` lift.
- Extend each `Operation<ArrayProgramType>` dispatcher arm directly for this mixed variant. It must not pass through
  `OperationProjection`, because a mixed signature belongs to neither homogeneous member family.
- Extend eager outer interpretation with one direct mixed-operation arm. The existing homogeneous projection helper
  remains unchanged and is not generalized into mixed matching.
- Keep the outer family at three top-level variants: array family, dimension family, and dimension size. Do not flatten
  member primitives or introduce a generic `Mixed` bucket.

### Interpretation and phase boundaries

- `DimensionSizeOperation` owns its generic `InterpretableOperation` implementation, constrained by the
  `DimensionSize` value capability. The reference backend owns concrete extent extraction.
- Before evaluating, interpretation validates the operation against the runtime input type. This preserves exact
  wrong-kind/count/refinement diagnostics even when interpretation is invoked directly.
- A concrete eager result may carry an exact static result identity distinct from the staged result identity. Existing
  boundary refinements validate it against the declared input-axis fact; no witness, identity side channel, or runtime
  payload mutation is needed.
- Add ordinary `PartiallyEvaluatableOperation` support so known arrays fold through eager interpretation and unknown
  arrays residualize the same mixed instruction.
- Defer composite batching to P5, differentiation/transposition to P6, and XLA/StableHLO lowering to P7. P3b must not
  introduce temporary transform adapters, a homogeneous fallback, or a second lowering contract.

## Implementation

- [x] Update the delivery ledger to close P3a with its reviewed source, integration, and remainder reconciliation
      commits, and record this P3b branch and scope.
- [x] Add the `dimension_size` operation module, canonical exports, operation constant, capability traits, payload,
      inference, identity renaming, rendering, interpretation, and ordinary partial-evaluation contract.
- [x] Implement reference-array `DimensionSize<usize>` without allocation, payload traversal, or scalar-array
      materialization.
- [x] Implement eager `DimensionSize` for `ArrayProgramValue<A>` without cloning the array payload.
- [x] Add the mixed operation variant, lift, operation-contract dispatch, and eager interpretation arm to
      `ArrayProgramOperation<A>`.
- [x] Confirm the generic staging and concrete eager capability implementations are coherence-safe before expanding
      the test sweep. If Rust requires per-carrier capability implementations or another wrapper tower, stop and
      redesign rather than reproducing the archived adapter hierarchy.
- [x] Keep the complete substantive increment within the approximately 800-line review ceiling. Split tests from
      production only if the complete operation increment would materially exceed it.

## Tests

- [x] Add module tests covering:
  - [x] dynamic-axis construction reuses the selected input variable;
  - [x] static-axis construction creates a fresh exact-bounds result variable;
  - [x] positive and negative axis normalization and out-of-range rejection;
  - [x] portable-extent rejection;
  - [x] one array input, zero regions, and exactly one dimension output;
  - [x] wrong member kind, wrong count, attached region, invalid rank/axis, unrelated dynamic identity, incompatible
        static refinement, and out-of-bounds static refinement diagnostics;
  - [x] identity renaming of both selected-axis metadata and the result;
  - [x] canonical rendering;
  - [x] eager reference execution returns the selected extent without copying or scanning the payload;
  - [x] direct eager use on a dimension member returns the canonical wrong-kind diagnostic; and
  - [x] known-side partial evaluation folds while unknown-side partial evaluation retains one instruction.
- [x] Extend array-program dispatcher tests to cover the third mixed variant and its direct eager arm.
- [x] Add a traced composite program golden proving:
  - [x] `dimension_size` has one explicit array operand and one dimension result;
  - [x] the dynamic result reuses the selected array-axis identity;
  - [x] two readers of the same axis are accepted as forwarded definitions;
  - [x] importing/replaying the program preserves the shared identity relationship; and
  - [x] no implicit dimension dependency, scalar-array conversion, or attached region is created.
- [x] Retain the existing projection allocation tests unchanged; P3b must not weaken their zero-copy guarantees.

## Verification

- [x] Run focused `dimension_size` and array-program dispatcher tests.
- [x] Run the array-program allocation integration test.
- [x] Run `cargo check -p ryft-core -p ryft-xla`.
- [x] Run `cargo test -p ryft-core --lib`.
- [x] Run `cargo test -p ryft-core --doc`.
- [x] Run `cargo test -p ryft-xla --lib`.
- [x] Run `cargo fmt --all -- --check` and `git diff --check`.
- [x] Attribute changed-file Clippy diagnostics; do not claim the inherited workspace warning backlog as P3b work.
- [x] Record targeted residual searches proving:
  - [x] no `Operation<ArrayType>` implementation exists for `DimensionSizeOperation`;
  - [x] no rank-zero array result or `RUNTIME_DIMENSION_DATA_TYPE` path exists in P3b;
  - [x] `ArrayOperation` contains no dimension-size variant;
  - [x] the outer family contains exactly the two homogeneous families plus the one named mixed variant;
  - [x] no witness, scope, expression, source-array, replay, or implicit-dependency mechanism was introduced; and
  - [x] all remaining archived `DimensionSizeOperation` behavior is either intentionally replaced here or assigned to
        P3c/P5/P6/P7.

## Acceptance Gates

- [x] `DimensionSizeOperation` has exactly one semantic result kind everywhere: a first-class dimension. Concrete
      backend capability implementations may expose the same axis size as a host extent.
- [x] Dynamic declared inputs accept compatible static eager refinements without weakening identity validation for
      staged dynamic inputs.
- [x] The rendered graph contains one explicit array-to-dimension edge and no hidden extent reconstruction.
- [x] Eager reference execution performs constant-time shape metadata access, checked host integer construction, and
      no array payload allocation or copy.
- [x] Repeated readers and program import preserve structural identity closure without operation-specific identity
      hooks.
- [x] The implementation adds one mixed variant and one concrete backend implementation of the shared capability, not
      a parallel mixed-operation framework or carrier-specific wrapper tower.

## Review

- Added the canonical mixed `DimensionSizeOperation` and its `DimensionSize` capability. Dynamic axes forward their
  declared identity; static axes define a fresh exact-bounds dimension.
- Implemented the shared `DimensionSize<Output>` capability for the reference `Array` with `Output = usize`, using
  only shape metadata. The existing allocation test and a payload-pointer regression prove that no eager array payload
  is cloned or scanned.
- Added one explicit mixed `ArrayProgramOperation::DimensionSize` variant. Homogeneous operation projection remains
  unchanged, while eager interpretation dispatches this mixed signature directly.
- The staging capability uses one generic outer-carrier implementation plus one `ProjectedValue<ArrayType, V>` bridge
  returning `V`. Focused compilation and the traced projected-result test confirmed coherence without a per-carrier
  implementation tower or another wrapper.
- Added consolidated construction/inference/diagnostic/eager/renaming/rendering tests, a composite trace/import golden,
  the dynamic-declared/static-actual regression, and known/unknown partial-evaluation coverage.
- Verification passed:
  - focused dimension-size tests: 3 passed;
  - array-program dispatcher tests: 7 passed;
  - projection allocation integration tests: 2 passed;
  - `cargo check -p ryft-core -p ryft-xla`;
  - `cargo test -p ryft-core --lib`: 970 passed;
  - `cargo test -p ryft-core --doc`: 54 passed and 16 ignored;
  - `cargo test -p ryft-xla --lib`: 396 passed and 1 ignored;
  - formatting and diff checks.
- Clippy completed successfully. Its diagnostics are inherited backlog; none names the new module or a changed P3b
  hunk.
- Residual searches found no homogeneous array contract, runtime-dimension array data type, array-family variant,
  hidden dependency machinery, or P3b-local witness/scope/expression/source-array/replay path. The sole unrelated
  `runtime dimension expressions` match is pre-existing reshape interpretation code in `backends/arrays.rs`.
- The substantive increment is approximately 635 added/changed lines including tests, below the 800-line review
  ceiling. Data conversion, transform policy, and lowering remain explicitly deferred to P3c, P5, P6, and P7.
