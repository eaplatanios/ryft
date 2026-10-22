# Transform Invariants

Read this reference before changing transform rules, replay, reference validation, sharding/manual variation, or dynamic
and ragged extents. Use the owning transform's current contracts as the source of truth.

## Replay and Validation Ownership

- In transform replay code, prefer the established `InterpretableOperation` path for tracer-valued staging unless the
  user explicitly asks for a different abstraction. If an experimental staging hook or lowering path is removed or
  superseded, delete its public trait methods, helper functions, and test-only scaffolding instead of leaving unused
  compatibility layers behind.
- Keep domain-specific construction and transform-specific execution policy out of the central `Operation` trait. Prefer
  existing operation providers and the owning transform's validated execution path; add a core hook only when a general
  operation contract requires it, rather than to simplify one domain's dispatch.
- Keep core reference validation independent of transform-specific argument roles and policies. Let callers supply
  diagnostic positions; keep tangent/cotangent roles, validation order, and primal-boundary diagnostics in
  differentiation.
- `Operation::consumed_type_identities` owns validation of its input-dependent declaration. Batching supplies logical
  per-item types and validates preservation and rule evidence; do not add a separate wrapper that rechecks the returned
  identities against those inputs.

## Batching and Manual Variation

- In non-trivial batching rules, explain logical and physical axis roles, replicated-input handling, reshape/transpose
  stages, and sharding restoration with code comments beside the corresponding steps.
- In transformation rules (e.g., batching, differentiation, transposition, or partial evaluation), give every value that
  the rule creates (e.g., a constant, `iota`, `zero`, or `one`) and then combines with existing values the placement and
  manual variation of those values. Operation-level rules and `Context::bind` assume inputs that were aligned when the
  program was staged, so they do not insert `parallel_vary` the way the value-level capabilities do. Create such a value
  with that variation directly only when it is a non-differentiable constant, and say so in a comment, because the
  transpose of the skipped `parallel_vary` is a cross-device sum. Stage any value that can carry a tangent through the
  value-level capabilities, which call `align_manual_variation`. Cover such rules with a test that runs them inside a
  manual region with varying inputs.

## Ordered Checks and Extent Dependencies

- When a transformation rule stages standalone assertions (e.g., `assert` over per-item extents) and then returns values
  those assertions validate, transport the dependency through a value-producing ordered operation that every later
  transform preserves (a `GuardOperation` bound after the lifted operation, over its outputs and every per-item extent
  array the outputs carry or a check consumed, whose outputs the rule returns and records in batch metadata), never
  through a flag such as a cast's checked mode or through witness inputs alone: partial evaluation, differentiation,
  nested batching, and identity folding erase proven checked modes on purpose. When no check is needed, keep the data
  dependencies anyway (an unchecked cast of the metadata witnessed by the outputs), so extents never depend on whether a
  check happened to be necessary. Keep batch metadata values together with their coordinates: key substitutions by the
  carrier storage they came from, never by dimension identity alone, and build relationship checks from the carrier
  occurrences a declaration actually names, each at its own coordinates (an identity-level provider only decides which
  storage an acquiring axis takes). Identify the coordinates of per-item metadata by the per-item dimensions of the axes
  it varies over (the identity a declaration names, or the static extent), never by logical positions, because carriers
  lay their axes out differently (a member and its per-item transpose). Define once when two occurrences denote the same
  extents (one identity at identical coordinates) and apply that definition in every check, placement and consumer,
  rejecting a coordinate that does not identify exactly one axis instead of guessing a correspondence. Give every
  consumer of a new batch representation one consistent policy through one helper (e.g.,
  `ArrayIrBatch::replicated_dimension_value`) before adding branches elsewhere, and validate per-item
  extents at the declared physical representation so dense and ragged paths admit the same declarations.
- Inspect both producers and consumers of these dependencies, including later partial evaluation, differentiation,
  transposition, nested batching, identity folding, and backend lowering. The detailed review inventory is in [Coverage
  and final review](review.md).
- A structurally batched region must receive every extent value and coordinate descriptor needed by its body, and retain
  the corresponding output extents. Reconstructing output metadata from input identities cannot recover checks or masks
  that the body omitted. If the region boundary does not carry this information, validate and reject the unsupported
  carriers before erasing them, including replicated carriers with outer ragged coordinates. Direct region replay may
  remain supported when it transports complete carriers. Do not inline rematerialization or custom derivative regions
  merely to bypass a boundary limitation: their execution and derivative contracts must survive the transform too.

For generic transform-policy changes, also apply the custom-policy and operation-family compatibility checks in
[Capabilities and providers](capabilities.md#domain-dispatch).
