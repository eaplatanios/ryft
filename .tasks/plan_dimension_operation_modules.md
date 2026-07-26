# Dimension operation module ownership

## Objective

Move the first-class dimension primitive payloads out of the reference backend and into a canonical
`ryft_core::operations::dimensions` module before heterogeneous projection work begins. The resulting ownership must
match the rest of `ryft-core`:

- operation payloads, type inference, generic transform behavior, and user-facing capabilities belong to
  `operations`;
- concrete eager values and their eager interpretation belong to `backends`;
- shared type and representability facts belong to `types`; and
- the reference backend's closed operation-family enum remains backend-owned.

This is a bounded ownership increment between P2b and P2c. It intentionally pulls the relevant part of Phase 9
forward because P2c and later mixed-operation work should depend on the final primitive paths rather than on temporary
backend paths.

## Reviewed baseline

- Integration branch: `u/eaplatanios/dynamic-shapes`
- Baseline commit: `b671b123acbde965ab8d0ea738482a22a825a9cb`
- Increment branch: `u/eaplatanios/increment/p2b1-dimension-operation-modules`
- P2b remainder reconciliation: `982333afccc46c04884db3d3fcd6cef44ac9d46a`

P2b currently defines two semantic primitive payloads in `backends::dimensions`:

- `DimensionArithmeticOperation`, tagged by `DimensionArithmetic`; and
- `DimensionRequirementOperation`, tagged by `DimensionRequirement`.

The same backend module also owns `DimensionValue`, `DimensionOperation<V>`, and `DimensionTracingContext`. Only the
first group is misplaced. The closed `DimensionOperation<V>` sum is analogous to `ArrayOperation<V>` and
`ScalarOperation<V>`: it is a reference-backend dispatch family, not an individual language primitive.

## Target public structure

```text
ryft_core
├── operations
│   └── dimensions
│       ├── arithmetic.rs
│       ├── requirement.rs
│       └── mod.rs
├── backends
│   └── dimensions.rs
└── types
    └── dimensions.rs
```

`operations::dimensions::mod.rs` will follow `operations::constants::mod.rs`: it declares one submodule per operation
payload and explicitly re-exports that submodule's capability, selector, operation payload, and operation-name
constants. `operations::mod.rs` will declare and re-export the new module so root-facade paths continue to come from
the semantic operation owner.

### Ownership table

| Item | Final owner | Rationale |
| --- | --- | --- |
| `DimensionArithmetic` capability | `operations::dimensions::arithmetic` | User-facing value operation |
| `DimensionArithmeticFunction` selector | `operations::dimensions::arithmetic` | Semantic selector for one payload |
| `DimensionArithmeticOperation` | `operations::dimensions::arithmetic` | Primitive payload and inference |
| Arithmetic bounds/evaluation helpers | `operations::dimensions::arithmetic` | Primitive semantics |
| `DimensionRequirement` capability | `operations::dimensions::requirement` | User-facing assertion operation |
| `DimensionRequirementPredicate` selector | `operations::dimensions::requirement` | Semantic selector for one payload |
| `DimensionRequirementOperation` | `operations::dimensions::requirement` | Primitive payload, proof, and effects |
| Requirement abstract interpretation | `operations::dimensions::requirement` | Generic primitive/PE semantics |
| `MAX_DIMENSION_EXTENT` | `types::dimensions` | Shared representability fact used by values and operations |
| `DimensionValue` | `backends::dimensions` | Concrete checked host representation |
| Eager interpretation over `DimensionValue` | `backends::dimensions` | Concrete backend behavior |
| `DimensionOperation<V>` | `backends::dimensions` | Reference backend's closed operation family |
| `DimensionTracingContext` | `backends::dimensions` | Reference backend context alias |

No compatibility module or backend re-export will preserve the old semantic path. All in-repo imports will move
directly to the canonical operation path. Existing root-facade names remain available only because
`ryft_core::operations::*` is already re-exported by `ryft_core`.

## Capability APIs and naming

The natural capability names are currently occupied by selector enums. Rename those secondary selector types rather
than inventing awkward capability names:

- enum `DimensionArithmetic` becomes `DimensionArithmeticFunction`;
- enum `DimensionRequirement` becomes `DimensionRequirementPredicate`;
- field/accessor `arithmetic` becomes `function`; and
- field/accessor `requirement` becomes `predicate`.

Backwards compatibility is not a goal for this refactor.

### `DimensionArithmetic`

Add a value-level `DimensionArithmetic` capability following `Add`, `Compare`, `Select`, and the constant
capabilities. Its required dispatch method applies one `DimensionArithmeticFunction` to `self` and a right operand.
Default convenience methods provide the explicit dimension vocabulary:

- `add_dimension`;
- `subtract_dimension`;
- `subtract_dimension_clamped`;
- `multiply_dimension`;
- `raise_dimension_to_power`;
- `floor_divide_dimension`;
- `remainder_dimension`;
- `minimum_dimension`; and
- `maximum_dimension`.

The blanket implementation for context-carrying `Value`s constructs one `DimensionArithmeticOperation` from the
operand types and binds it through the value's dispatch domain. `DimensionValue` receives the small explicit eager
implementation in `backends::dimensions`, delegating through its existing `DimensionOperation` execution domain.
This follows the established split used by array/scalar capabilities: generic tracer dispatch stays with the
operation, while the concrete eager value owns its direct implementation.

### `DimensionRequirement`

Add a value-level `DimensionRequirement` capability with four methods:

- `require_equal`;
- `require_less_than_or_equal`;
- `require_divisible_by`; and
- `require_bounds`.

The public trait does not expose an `Option`-based generic requirement method. Four explicit methods preserve the
operation payload's valid unary/binary states and keep call-site diagnostics clear. Context-carrying values bind the
corresponding `DimensionRequirementOperation`; `DimensionValue` supplies the concrete eager implementation through
its execution domain.

### Interpretation boundary

The operation modules must not depend on the concrete reference backend in production code.

- Arithmetic exposes crate-private checked extent evaluation on its semantic selector/payload. The backend-specific
  `InterpretableOperation<EagerContext<DimensionValue, _>>` implementation reads concrete extents, invokes that
  semantic evaluator, and constructs the backend value with the operation's declared result type.
- Requirements expose crate-private extent validation on the payload. Their backend interpretation extracts extents
  and delegates to it.
- Requirement partial evaluation becomes backend-neutral by requiring a resolvable constant to implement
  `Concretizable<usize>` instead of naming `DimensionValue`. Exact facts remain optional; failure to resolve a
  flowing value remains conservative rather than creating a value-extraction side channel.

This leaves backend files with thin materialization adapters and prevents `operations` from importing `backends`.

## Documentation contract

Each new submodule will follow the documentation placement requested for this increment:

- The capability trait carries the main conceptual documentation, including operand/result semantics, checked error
  cases, the distinction between checked and clamped subtraction, divisibility positivity, ordered assertion
  behavior, and how dispatch behaves for eager values versus tracers.
- Each capability trait includes an executable rustdoc example using `DimensionValue` and returns
  `Result<(), ProgramError>` through a hidden `main`.
- Each capability method documents its exact behavior and edge cases.
- Selector types document only their role in choosing behavior and each variant's meaning.
- `DimensionArithmeticOperation` and `DimensionRequirementOperation` have concise payload-focused documentation and
  explicitly refer readers to their capability trait for semantic details.
- Operation name constants refer to the selector variant and owning operation where useful.
- `operations::dimensions` receives module-level documentation explaining that these are first-class dimension SSA
  primitives, not array elementwise integer operations or a parallel expression language.
- `backends::dimensions` documentation is narrowed to the host value and reference operation family after extraction.

The operation payload docstrings will not duplicate the long capability examples.

## Unit-test placement

Follow `.agents/unit-testing-guidelines.md` and the colocated structure used by `operations::constants`.

### `operations::dimensions::arithmetic::tests`

- Consolidate normal operation behavior under `test_dimension_arithmetic`:
  selector/accessor behavior, all nine functions, inferred bounds, fresh result identity, rendering, eager
  interpretation, and the public capability methods.
- Keep a separate `test_dimension_arithmetic_errors` because checked underflow, zero divisors, portable-width
  overflow, and mismatched declared operand types form a substantial independent error contract.
- Move the dimension arithmetic tracing/program/known-side partial-evaluation scenario beside the operation it
  exercises, either into the main test when it remains readable or into one independently named program-composition
  test.
- Use exact typed `DimensionError` and rendering assertions.

### `operations::dimensions::requirement::tests`

- Keep `test_dimension_requirement` for predicate construction, proof outcomes, effects, rendering, eager execution,
  observed-value diagnostics, and public capability methods.
- Keep `test_dimension_requirement_effects_and_partial_evaluation` separate because ordered DCE survival,
  deterministic first-failure order, known-side folding, and residual placement are one substantial transform
  contract.
- Keep the narrow `requirement_program` helper only if both proof/PE groups use it; otherwise inline it.

### `backends::dimensions::tests`

- Retain only tests owned by `DimensionValue`, the closed `DimensionOperation<V>` family, and their concrete eager
  adapters.
- Do not duplicate operation semantic assertions already moved into the operation submodules.

Rustdoc examples are part of the verification gate, not illustrative unchecked snippets.

## Implementation sequence

- [ ] Record P2b as landed in the cleanup ledger with integration commit
      `b671b123acbde965ab8d0ea738482a22a825a9cb` and remainder reconciliation commit
      `982333afccc46c04884db3d3fcd6cef44ac9d46a`.
- [ ] Add `P2b.1` to the cleanup plan's increment catalog before P2c and note that this owner-requested move advances
      the primitive-operation portion of Phase 9 without advancing the public runtime-shape API work.
- [ ] Add `MAX_DIMENSION_EXTENT` to `types::dimensions` as the single shared portable-extent limit and update the
      existing backend validation to import it.
- [ ] Create `operations::dimensions::{arithmetic, requirement}` and its explicit `mod.rs` re-exports.
- [ ] Move arithmetic selectors, payloads, names, inference, renaming, bounds transfer, generic PE declaration, and
      semantic extent evaluation into `arithmetic.rs`.
- [ ] Add the documented `DimensionArithmetic` capability, blanket context-carrying value implementation, and
      concrete `DimensionValue` implementation.
- [ ] Move requirement selectors, payloads, names, proof lattice, abstract facts, inference, effects, generic PE
      behavior, and semantic extent validation into `requirement.rs`.
- [ ] Generalize exact constant inspection from `DimensionValue` to `Concretizable<usize>` without weakening
      conservative unknown handling or diagnostics.
- [ ] Add the documented `DimensionRequirement` capability, blanket context-carrying value implementation, and
      concrete `DimensionValue` implementation.
- [ ] Leave only `DimensionValue`, eager materialization adapters, `DimensionOperation<V>`, and
      `DimensionTracingContext` in `backends::dimensions`.
- [ ] Remove semantic operation re-exports from `backends::mod`; export them from `operations::dimensions` and update
      every in-repo import directly.
- [ ] Move and consolidate unit tests according to the test-placement matrix above.
- [ ] Remove the temporary module-review TODO once the backend module has its final bounded responsibility.
- [ ] Update rustdoc links and module prose; do not leave duplicated semantic documentation on payload structs.

## Verification

Run each potentially expensive command with a 300-second timeout and
`CARGO_TARGET_DIR=/Users/eaplatanios/Development/Repositories/ryft-1-dimensions-target`.

- [ ] `cargo fmt -p ryft-core -- --check`
- [ ] `git diff --check`
- [ ] Focused arithmetic module tests
- [ ] Focused requirement module tests
- [ ] `cargo check -p ryft-core`
- [ ] `cargo test -p ryft-core --lib`
- [ ] `cargo test -p ryft-core --doc`
- [ ] `cargo check -p ryft-xla`
- [ ] `cargo test -p ryft-xla --lib`

The macro crates are not part of the default gate because this increment does not change `Operation` or a derive
contract. Run them if implementation reveals a derive-bound or generated-dispatch change.

## Residual and minimality gates

- [ ] No production definition of `DimensionArithmeticOperation` or `DimensionRequirementOperation` remains under
      `backends`.
- [ ] No old selector name remains (`DimensionArithmetic` as an enum or `DimensionRequirement` as an enum).
- [ ] No import or rustdoc link uses `backends::dimensions` for a semantic primitive.
- [ ] `operations` production code has no dependency on `crate::backends`.
- [ ] `backends::dimensions` contains no duplicate arithmetic bounds/proof/evaluation implementation.
- [ ] There remains exactly one arithmetic payload, one requirement payload, and one reference-backend operation-family
      variant for each payload.
- [ ] Every remaining dimension test is classified as operation semantics, concrete backend behavior, or intentionally
      cross-cutting program composition; no scenario is duplicated across modules.
- [ ] The public capability methods stage ordinary SSA operations and introduce no expression tree, witness,
      substitution, alternate dimension program, or value-extraction side channel.

## Review and handoff

This plan is the first review gate on the P2b.1 branch. Production implementation must not begin until the owner
reviews or edits it. After approval:

1. advance this same increment branch from the reviewed integration state;
2. implement and verify the bounded extraction;
3. push the source commit;
4. stage a no-commit merge in the owner checkout for line-by-line review; and
5. only after the owner commits and pushes, reconcile the reviewed paths into the mutable remainder.

P2c begins only after P2b.1 lands, so storage-sum projection is built against the canonical dimension primitive paths.
