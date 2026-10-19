# Coverage and Final Review

For a new operation or a full operation review, record the complete coverage inventory in the task plan before
implementation or fixing findings. For an explicitly scoped follow-up, apply this process to affected contracts and
consumers without silently expanding the requested task into a repository-wide audit. Respect read-only review requests:
report findings instead of editing implementation code.

## Coverage Inventory

Use an inventory with entries for the inspected implementation, its invariants, verification evidence, and remaining
uncertainty. Mark a case inapplicable only with a reason. Include:

- Payload, constructors/accessors, formatting, and public capability/provider APIs.
- Input/output type inference, region signatures, effects, placement, and validation contracts.
- Interpretation and eager/staged dispatch, including composite members.
- Reference discharge, identity preservation, and aliasing/access contracts where applicable.
- Partial evaluation, known/unknown values, residuals, and deferred work.
- Batching, differentiation, and transposition, including their interactions.
- Family/bundle integration, backend conversions, lowering, and execution where present.
- Module/API documentation, implementation comments, and tests against repository conventions.

Read the repository's testing guidelines and identify the existing operation and capability patterns that apply. Do not
restrict the inventory to changed lines or previously reported issues. Choose meaningful cases for each rule rather than
mechanically testing every combination.

For region-carrying operations, include known, symbolic, and unknown inputs; eager and staged execution; selected and
unselected regions; observable effects and deferred work; zero outputs and structural zeros; static, dynamic, and
bounded ragged extents; references and identity-bearing values; placement and manual variation; and nested transform
composition where supported.

Trace information across every transform boundary: values and types, region signatures, runtime extents, reference
identities, effects, derivative activity, and validation evidence. Verify both their producers and consumers; a locally
correct rule can still lose information at a boundary. Check direct context binding as well as validated program replay:
rewriting an operation must not bypass its original primitive contract. Matching primal signatures do not guarantee
matching compact derivative slots, and replaying separate regions into one scope must preserve local type identities.
Inspect supported and rejected cases, including empty, singleton, and disconnected inputs or outputs where meaningful.
Evaluate documented limitations and optimizations against their semantic consequences and the requested JAX parity;
their presence in comments is not justification for accepting them.

When a change extends how batches carry extents, inventory every metadata consumer before fixing individual failures:

- Exercise repeated occurrences of one dimension identity within one carrier, with different extent coordinates,
  capacities, and metadata order. Verify both output extents and every relationship check; an identity-level provider
  cannot substitute for a particular physical occurrence. When extent coordinates are themselves ragged, vary only
  inactive coordinate entries and verify that relationship checks still depend solely on live coordinates. Cover both
  static refinements and equality checks, plus a genuine live mismatch and an empty live domain.
- Follow the metadata into region bodies, including rematerialization, custom derivatives, and linear calls. Output
  restoration cannot repair a body that already computed on padding or skipped a per-item assertion. Compare direct
  execution with execution through the region boundary using observable live values and failing checks.
- Inspect every axis-remapping consumer for removed extent coordinates, and distinguish consuming a ragged axis from
  preserving it after removing one of its coordinates. Compare sequential reductions with a joint reduction where
  supported, and require an explicit error where surviving extents cannot be represented. When aggregating extents, mask
  the extent arrays' own live coordinate domains and cover empty participant domains; a numeric reduction identity need
  not be a valid dimension extent.
- Test equivalent extent-array layouts with permuted coordinate storage. Distinguish coordinate identity from vector
  order, while retaining explicit layout requirements at boundaries that actually impose them. Also compare full and
  singleton-broadcast extent storage over the same coordinate domain, including nested ragged coordinates and empty
  domains. Include literal and symbolically proved singleton axes on either side of equality checks and in
  outer-coordinate masks, not only scalar refinements. Audit the shared masking path used by reductions and contractions
  as well as relationship checks. Storage shape is not the semantic coordinate domain: checks must range over live
  coordinates, and metadata transport must preserve valid compact storage without using a cast as a broadcast. Follow
  transported metadata into dimension reads, participant routing, and extent aggregation too. Before a consumer treats
  metadata as rectangular coordinate data, materialize every coordinate it indexes, including omitted participant
  coordinates and empty domains; normalize proved symbolic singleton storage at that materialization boundary. Cover
  both mapped and replicated carriers, and routing that inserts a participant axis as well as routing one already
  recorded in the metadata. Execute symbolic-to-static normalization through the backend followed by singleton
  expansion: a static result annotation alone may leave dynamic storage flags that invalidate that expansion.
- Exercise replicated ragged carriers produced by ordinary operations, including collectives, through shared elementwise
  rules and broadcasting. A replicated shortcut must preserve the same extents as its mapped path. Check homogeneous
  policies as well as composite policies: unsupported consumers must reject before executing on padding or unwrapping
  logical dimensions, and policy validation must catch accidental metadata loss independently of the individual
  consumer. Compare both input orders of commutative operations, including singleton coordinate broadcasts. Validate
  semantic coordinate domains before broadcasting erases their differences, not only the final axis positions. Compose
  an explicit or dense elementwise broadcast with a later ragged operation: a check local to one merge cannot recover
  coordinate distinctions an earlier producer discarded. If the representation cannot preserve those distinctions,
  reject that transformation at the producer instead of silently changing the identity's meaning.

For ordered validation dependencies, exercise both batching-before-differentiation and differentiation-before-batching,
then partially evaluate with unknown extents and known payloads or tangents. A guard whose own relationships are proven
can still transport earlier assertions; proof of its local types does not make that ordering dependency redundant.

After a finding exposes a shared invariant, audit all producers, consumers, and sibling implementations of that
invariant in the same pass. Add regressions at the owning boundary, not only at the operation that revealed the failure.

Compare the relevant rules with official JAX implementation sources and distinguish frontend conveniences from primitive
contracts. Validate deliberate extensions on their own semantics. Use existing tests as evidence, then identify
adversarial cases those tests do not exercise. Add focused regressions for changed behavior and material coverage gaps.
Tests involving non-termination or expensive execution must fail deterministically without running an unbounded
computation.

Passing tests or repairing discovered issues does not establish review completeness. Check off a behavior only after
inspecting its code and assessing the adequacy of its evidence. After edits, verify the actual final source and affected
backend behavior; an earlier compiled binary is evidence for the source it contains, not for later changes. Report
blocked verification and intentionally unsupported behavior explicitly instead of claiming exhaustive correctness or
complete parity.

## Independent Final Pass

After implementation and verification, independently review the complete requested operation surface again for
correctness, relevant JAX parity, and all repository conventions, including module/API documentation and tests. Use a
separate reviewer against the coverage inventory and final source. Provide the requirements and evidence, without
supplying prior conclusions when they would bias the review. For complex operations, delegate focused investigation to
separate subagents while keeping one shared inventory.

Resolve confirmed findings within the authorized scope, update their verification evidence, and revisit affected
inventory entries after every fix. Repeat the affected review and verification until the requested requirements are
satisfied, or report a concrete remaining blocker. In a read-only review, report confirmed findings and the needed
verification instead of applying fixes.

The final report must state verified outcomes and remaining limitations. Passing tests, repairing known issues, or a
final pass that finds nothing further does not establish that no undiscovered defects exist. Distinguish verified JAX
parity, deliberate extensions, unsupported behavior, and unavailable verification.
