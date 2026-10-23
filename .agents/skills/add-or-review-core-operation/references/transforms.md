# Transform Invariants

Read this reference before changing transform rules, replay, reference validation, sharding/manual variation, dynamic
extents, or explicit dense-buffer kernel contracts. Use the owning transform's current contracts as the source of truth.

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

- Preserve ordered runtime checks and the values that depend on them across partial evaluation, differentiation,
  transposition, nested batching, identity folding, and lowering. A guard whose own type relationships are proven can
  still carry an earlier assertion's ordering dependency; local type equality does not make that dependency redundant.
- Standalone transformed regions must receive every shared dynamic extent they reference, including bookkeeping inputs
  required by the batching policy. Preserve the corresponding output identities and checks when adapting their boundary.
  Do not inline rematerialization or custom derivative regions merely to bypass a boundary limitation.
- Arrays remain rectangular. Reject a shape that depends on mapped data before staging operations that would need
  per-item extents. Explicit kernels may consume dense padded buffers and ordinary size arrays under their own contracts;
  those size arrays do not become type-system or batching-carrier metadata.

For generic transform-policy changes, also apply the custom-policy and operation-family compatibility checks in
[Capabilities and providers](capabilities.md#domain-dispatch).
