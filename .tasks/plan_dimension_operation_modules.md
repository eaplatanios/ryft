# Dimension operation module ownership

## Objective

Move the first-class dimension primitive payloads out of the reference backend and into a canonical
`ryft_core::operations::dimensions` module before heterogeneous projection work begins. Model every arithmetic
primitive as its own nominal operation type, with shared behavior provided by an `ArithmeticDimensionOperation`
supertrait rather than by a runtime function-selector enum.

The resulting ownership must match the rest of `ryft-core`:

- operation payloads, type inference, generic transform behavior, and user-facing capabilities belong to
  `operations`;
- concrete eager values and their eager interpretation belong to `backends`;
- shared type and representability facts belong to `types`; and
- the reference backend's closed operation-family enum remains backend-owned.

This is a bounded ownership increment between P2b and P2c. It intentionally pulls the relevant part of Phase 9
forward because P2c and later mixed-operation work should depend on the final primitive paths rather than temporary
backend paths.

## Reviewed baseline

- Integration branch: `u/eaplatanios/dynamic-shapes`
- Reviewed plan integration: `a077fd0cdefadfa934e0fc49415a26d83c811ca5`
- Increment branch: `u/eaplatanios/increment/p2b1-dimension-operation-modules`
- P2b remainder reconciliation: `982333afccc46c04884db3d3fcd6cef44ac9d46a`

P2b currently defines two tagged primitive payloads in `backends::dimensions`:

- `DimensionArithmeticOperation`, tagged by `DimensionArithmetic`; and
- `DimensionRequirementOperation`, tagged by `DimensionRequirement`.

P2b.1 replaces the arithmetic tag with nine nominal operation types. The requirement payload remains tagged because
its predicates deliberately form one assertion operation family: all variants produce no results, have the same
effect/proof/partial-evaluation contract, and its private constructors already make unary/binary states valid.

The backend module also owns `DimensionValue`, `DimensionOperation<V>`, and `DimensionTracingContext`. Those remain
backend-owned. The closed `DimensionOperation<V>` sum is analogous to `ArrayOperation<V>` and `ScalarOperation<V>`:
it is a reference-backend dispatch family, not an individual language primitive.

## Why arithmetic uses nominal operation types

Every stored Ryft program already performs one runtime selection through its closed operation-family enum. A tagged
arithmetic payload adds a second selection:

```text
DimensionOperation::Arithmetic(DimensionArithmeticOperation { function: Add, ... })
                                  └──────── outer dispatch ────────┘
                                                                   └─ inner dispatch
```

Nominal variants flatten this to one selection:

```text
DimensionOperation::Add(DimensionAddOperation { ... })
                    └──────── one outer dispatch ────────┘
```

Capability implementations construct the concrete nominal operation before binding it. The reference eager context
then converts it into its closed operation family, so eager binding performs the same single outer-family selection
used by stored-program interpretation. Avoiding even that selection would require changing the generic `Context`
contract or introducing operation-specific eager domains, neither of which is justified here. XLA lowering resolves
the concrete primitive during compilation and emits no operation-enum dispatch into executable code.

The nine nominal payloads must not duplicate their shared contract. `ArithmeticDimensionOperation` plays the same
role for dimension arithmetic that `ElementwiseOperation` plays for array elementwise operations: concrete operation
types declare their function-specific calculation, while the supertrait centralizes common operand validation, result
metadata access, inference, and generic transform behavior.

## Why existing numeric operation payloads are not reused

Do not implement `Operation<DimensionType>` for the existing `AddOperation`, `SubOperation`, `MulOperation`, and
related unit payloads.

Those payloads are stateless because array/scalar output metadata follows ordinary promotion and broadcasting.
Dimension arithmetic must retain:

- the declared left operand type;
- the declared right operand type; and
- one fresh result `DimensionVariable` with inferred bounds.

The result identity must remain stable across repeated inference, cloning, renaming, import, and caching. A stateless
numeric payload cannot create it during `infer_output_types` without generating a different identity on repeated
calls. Existing numeric operations also carry array/scalar broadcasting and differentiation contracts that dimensions
must not inherit. Reusing them would spread conditional dimension behavior into established generic operations and
would not eliminate the need for dimension-specific metadata.

The existing `Add`, `Sub`, and related capability traits are not reused either. Their blanket implementations stage
the corresponding stateless numeric payloads. Dimension capabilities must stage the identity-bearing dimension
payloads instead.

## Target public structure

```text
ryft_core
├── operations
│   └── dimensions
│       ├── add.rs
│       ├── floor_divide.rs
│       ├── maximum.rs
│       ├── minimum.rs
│       ├── mod.rs
│       ├── multiply.rs
│       ├── power.rs
│       ├── remainder.rs
│       ├── requirement.rs
│       ├── subtract.rs
│       └── subtract_clamped.rs
├── backends
│   └── dimensions.rs
└── types
    └── dimensions.rs
```

`operations::dimensions::mod.rs` follows `operations::constants::mod.rs`: it declares one public submodule per
operation payload and explicitly re-exports each submodule's capability, operation payload, and operation-name
constant. It additionally owns the dimension-specific arithmetic supertrait and private shared implementation
machinery. `operations::mod.rs` declares and re-exports the new module so root-facade paths come from the semantic
operation owner.

### Ownership

- `operations::dimensions` owns `ArithmeticDimensionOperation` and the shared arithmetic operand/result metadata.
- Each arithmetic submodule owns its corresponding capability, nominal operation payload, result-bounds rule, and
  checked calculation.
- `operations::dimensions::requirement` owns `DimensionRequirement`, `DimensionRequirementPredicate`,
  `DimensionRequirementOperation`, and requirement abstract interpretation.
- `types::dimensions` owns `MAX_DIMENSION_EXTENT`, the shared representability fact used by values and operations.
- `backends::dimensions` owns `DimensionValue`, eager materialization, `DimensionOperation<V>`, and
  `DimensionTracingContext`.

No compatibility module or backend re-export preserves the old semantic path. All in-repo imports move directly to
the canonical operation path. Existing root-facade names remain available only through the existing
`ryft_core::operations::*` re-export.

## `ArithmeticDimensionOperation`

Define the public supertrait as an `Operation<DimensionType>` refinement. Its shared surface should remain small and
semantic:

- access the declared left and right operand types;
- return the fresh result type;
- evaluate concrete nonnegative extents using the concrete operation's statically dispatched implementation; and
- provide the common two-input validation and output-type inference implementation used by each `Operation` impl.

Like `ElementwiseOperation::infer_output_types`, the supertrait's inference helper is explicitly delegated to by each
concrete `Operation<DimensionType>` implementation. Shared operand/result storage and simultaneous identity renaming
belong to one private metadata type rather than being copied across nine structs.

Function-specific constructors remain on their nominal operation types because each one owns its result-name and
bounds-transfer rule. Function-specific evaluation remains in the concrete supertrait implementation, with no public
or private arithmetic selector enum and no inner runtime match.

Generic behavior that is truly uniform should be authored once using the supertrait and generated impl shells:

- eager interpretation adapter over `DimensionValue`;
- partial-evaluation marker behavior;
- input count and refinement validation;
- result materialization from the stored result type; and
- any later uniform batching/non-differentiability rule.

Do not force operation-specific bounds formulas or checked calculations into a match inside the supertrait.

## Per-operation capabilities

Each nominal operation receives a value-level capability in its own submodule:

| Operation | Capability | Method |
| --- | --- | --- |
| `DimensionAddOperation` | `DimensionAdd` | `add_dimension` |
| `DimensionSubtractOperation` | `DimensionSubtract` | `subtract_dimension` |
| `DimensionSubtractClampedOperation` | `DimensionSubtractClamped` | `subtract_dimension_clamped` |
| `DimensionMultiplyOperation` | `DimensionMultiply` | `multiply_dimension` |
| `DimensionPowerOperation` | `DimensionPower` | `raise_dimension_to_power` |
| `DimensionFloorDivideOperation` | `DimensionFloorDivide` | `floor_divide_dimension` |
| `DimensionRemainderOperation` | `DimensionRemainder` | `remainder_dimension` |
| `DimensionMinimumOperation` | `DimensionMinimum` | `minimum_dimension` |
| `DimensionMaximumOperation` | `DimensionMaximum` | `maximum_dimension` |

Each blanket implementation for context-carrying `Value`s constructs the corresponding concrete operation from
borrowed operand types and binds it through the value's dispatch domain. Staged values record ordinary SSA
instructions. `DimensionValue` uses the reference backend's closed operation family and therefore pays one outer enum
selection, but no arithmetic selector exists inside the selected payload.

`DimensionRequirement` remains one capability with four explicit methods:

- `require_equal`;
- `require_less_than_or_equal`;
- `require_divisible_by`; and
- `require_bounds`.

The requirement capability does not expose an `Option`-based generic method. Four explicit methods preserve valid
unary/binary construction and keep call-site diagnostics clear.

## Backend-neutral semantic boundary

The operation modules must not depend on the concrete reference backend in production code.

- Arithmetic exposes its checked extent calculation through `ArithmeticDimensionOperation`. The backend-specific
  eager adapter reads concrete extents, invokes the concrete operation implementation, and constructs
  `DimensionValue` with the operation's stored result type.
- Requirements expose crate-private extent validation on their payload. Their backend interpretation extracts extents
  and delegates to it.
- Requirement partial evaluation becomes backend-neutral by requiring resolvable constants to implement
  `Concretizable<usize>` instead of naming `DimensionValue`. Exact facts remain optional; failure to resolve or
  concretize remains conservative rather than creating a value-extraction side channel.

This leaves backend files with thin materialization adapters and prevents `operations` from importing `backends`.

## Documentation contract

Each public operation submodule follows the documentation placement requested for this increment:

- The capability trait carries the main conceptual documentation and one executable rustdoc example using
  `DimensionValue`.
- Arithmetic capability documentation explains its checked or clamped behavior, bounds implications, and relevant
  error cases.
- Requirement capability documentation explains static proof, ordered residual assertions, and observed-value
  diagnostics.
- Each capability method documents its exact operand and result behavior.
- Every operation payload has concise payload-focused documentation and refers readers to its corresponding capability
  trait for semantic details and examples.
- `ArithmeticDimensionOperation` documents only the shared program contract and why nominal payloads use it.
- `DimensionRequirementPredicate` documents only its selector role and each predicate's meaning.
- Operation-name constants refer to their owning operation.
- `operations::dimensions` explains that these are first-class dimension SSA primitives, not array elementwise integer
  operations or a parallel expression language.
- `backends::dimensions` documentation narrows to the host value, eager adapters, and reference operation family.

All rustdoc examples use the repository's hidden fallible `main` pattern and are verified as doctests. Operation
payload docstrings do not duplicate capability examples.

## Unit-test placement

Follow `.agents/unit-testing-guidelines.md` and the colocated structure used by `operations::constants`.

### Per-operation arithmetic tests

Every arithmetic submodule has one consolidated `test_<operation>` covering:

- operation name, stored operand/result metadata, and rendering;
- result bounds and fresh identity;
- the operation-specific checked calculation, including its relevant edge/error cases;
- eager interpretation;
- the public capability method; and
- exact typed `DimensionError` diagnostics owned by that primitive.

Do not repeat shared arity/refinement/renaming assertions nine times. Test the
`ArithmeticDimensionOperation` common contract centrally in `operations::dimensions::tests` with a representative
operation and exact failures. Keep the dimension arithmetic program/tracing/known-side partial-evaluation scenario in
the central module test because it verifies uniform supertrait behavior rather than one formula.

### Requirement tests

- Keep `test_dimension_requirement` for predicate construction, proof outcomes, effects, rendering, eager execution,
  observed-value diagnostics, and public capability methods.
- Keep `test_dimension_requirement_effects_and_partial_evaluation` separate because ordered DCE survival,
  deterministic first-failure order, known-side folding, and residual placement form one substantial transform
  contract.
- Keep the narrow `requirement_program` helper only if both proof/PE groups use it; otherwise inline it.

### Backend tests

Retain only tests owned by `DimensionValue`, the closed `DimensionOperation<V>` family, and concrete eager adapters.
Do not duplicate operation semantic assertions already colocated with the operation modules.

## Implementation sequence

- [x] Record P2b as landed in the cleanup ledger with integration commit
      `b671b123acbde965ab8d0ea738482a22a825a9cb` and remainder reconciliation commit
      `982333afccc46c04884db3d3fcd6cef44ac9d46a`.
- [x] Add `P2b.1` to the cleanup plan's increment catalog before P2c and note that this owner-requested move advances
      the primitive-operation portion of Phase 9 without advancing the public runtime-shape API work.
- [x] Add `MAX_DIMENSION_EXTENT` to `types::dimensions` as the single shared portable-extent limit and update the
      existing backend validation to import it.
- [x] Create `operations::dimensions` with one public submodule for each of the nine arithmetic payloads and one for
      `DimensionRequirementOperation`.
- [x] Add the public `ArithmeticDimensionOperation` supertrait, private shared operand/result metadata, common
      inference, shared identity renaming, and generic transform adapters.
- [x] Replace `DimensionArithmetic` and `DimensionArithmeticOperation` with the nine nominal operation types and
      remove the selector enum without a compatibility alias.
- [x] Implement each operation-specific result-name, bounds-transfer, and checked extent calculation without an inner
      arithmetic enum match.
- [x] Add the nine documented value-level capability traits and blanket context-carrying value implementations.
- [x] Move requirement selectors, payload, names, proof lattice, abstract facts, inference, effects, generic PE
      behavior, and semantic extent validation into `requirement.rs`.
- [x] Rename the requirement selector to `DimensionRequirementPredicate` so `DimensionRequirement` names the
      capability trait.
- [x] Generalize exact requirement constant inspection from `DimensionValue` to `Concretizable<usize>` without
      weakening conservative unknown handling or diagnostics.
- [x] Add the documented `DimensionRequirement` capability and its four explicit methods.
- [x] Replace the backend family's single `Arithmetic` variant with one variant per nominal arithmetic operation;
      retain one `Requirement` variant.
- [x] Add thin macro-generated eager interpretation adapters for `ArithmeticDimensionOperation` without duplicating
      formulas in the backend.
- [x] Leave only `DimensionValue`, eager materialization adapters, `DimensionOperation<V>`, and
      `DimensionTracingContext` in `backends::dimensions`.
- [x] Remove semantic operation re-exports from `backends::mod`; export them from `operations::dimensions` and update
      every in-repo import directly.
- [x] Move and consolidate unit tests according to the placement matrix above.
- [x] Remove the temporary module-review TODO once the backend module has its final bounded responsibility.
- [x] Update rustdoc links and module prose; do not leave duplicated semantic documentation on payload structs.

## Verification

Run each potentially expensive command with a 300-second timeout and
`CARGO_TARGET_DIR=/Users/eaplatanios/Development/Repositories/ryft-1-dimensions-target`.

- [x] `cargo fmt -p ryft-core -- --check`
- [x] `git diff --check`
- [x] Focused tests for every arithmetic operation submodule
- [x] Focused shared `ArithmeticDimensionOperation` tests
- [x] Focused requirement tests
- [x] `cargo check -p ryft-core`
- [x] `cargo test -p ryft-core --lib`
- [x] `cargo test -p ryft-core --doc`
- [x] `cargo check -p ryft-xla`
- [x] `cargo test -p ryft-xla --lib`

The macro crates are not part of the default gate because this increment does not change `Operation` or a derive
contract. Run them if implementation adds or changes a macro consumed by operation-family derivation.

## Dispatch and minimality gates

- [x] `DimensionArithmetic` and `DimensionArithmeticOperation` no longer exist.
- [x] There are exactly nine nominal arithmetic payloads and nine corresponding backend-family variants.
- [x] No public or private arithmetic selector enum remains.
- [x] Stored-program eager interpretation performs the outer `DimensionOperation` selection and then invokes a
      statically selected concrete arithmetic implementation without a second function-tag match.
- [x] Direct capabilities construct a concrete nominal payload before the reference eager context performs its one
      required outer-family selection.
- [x] Shared operand validation, inference, identity renaming, and transform adapters exist once through
      `ArithmeticDimensionOperation` and its private metadata.
- [x] Function-specific bounds and evaluation code remain in their owning operation submodules rather than one central
      match.
- [x] No production definition of a dimension primitive operation remains under `backends`.
- [x] No import or rustdoc link uses `backends::dimensions` for a semantic primitive.
- [x] `operations` production code has no dependency on `crate::backends`.
- [x] `backends::dimensions` contains no duplicate arithmetic bounds/proof/evaluation implementation.
- [x] Every remaining dimension test is classified as operation semantics, shared supertrait behavior, concrete backend
      behavior, or intentionally cross-cutting program composition; no scenario is duplicated.
- [x] The public capability methods stage ordinary SSA operations and introduce no expression tree, witness,
      substitution, alternate dimension program, or value-extraction side channel.

## Review and handoff

This revision is the second plan review on the P2b.1 branch. Production implementation begins only after the revised
plan is staged and accepted. Then:

1. advance this same increment branch from the reviewed integration state;
2. implement and verify the bounded extraction;
3. push the implementation commit;
4. stage a no-commit merge in the owner checkout for line-by-line review; and
5. only after the owner commits and pushes, reconcile the reviewed paths into the mutable remainder.

P2c begins only after P2b.1 lands, so storage-sum projection is built against the canonical nominal dimension
primitive paths.

## Review

- Replaced the tagged arithmetic payload with nine nominal operation types and one corresponding variant per primitive
  in the reference backend's closed operation family.
- Added `ArithmeticDimensionOperation` plus private shared metadata and one declarative macro so operand validation,
  inference, identity renaming, capability dispatch, and transform impl shells have one implementation.
- Kept each primitive's result naming, bounds transfer, and checked calculation in its own operation submodule; no
  arithmetic selector enum or second function-tag match remains.
- Moved requirement semantics to `operations::dimensions::requirement`, renamed its selector to
  `DimensionRequirementPredicate`, and generalized partial constant inspection to `Concretizable<usize>`.
- Reduced `backends::dimensions` to the concrete host value, eager materialization adapters, tracing alias, and closed
  family. Direct eager capabilities perform one required closed-family selection through the existing `Context`
  contract; avoiding that selection would require a broader context redesign and is intentionally out of scope.
- Added executable capability examples, one consolidated test per arithmetic module, shared inference/renaming and
  program/tracing/partial-evaluation coverage, and requirement proof/effect/partial-evaluation coverage.
- Verified formatting and diff hygiene, all 943 `ryft-core` library tests, all 53 executable `ryft-core` doctests, and
  all 396 `ryft-xla` library tests. A strict repository-wide clippy run remains blocked by 133 pre-existing warnings;
  an isolated clippy pass reported no diagnostics in the changed dimension files.
