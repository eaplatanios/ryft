# First-class dimension cleanup delivery ledger

This ledger records the recoverable delivery state for
`.tasks/plan_symbolic_dimensions_architecture_cleanup.md`. Repository and remote state are authoritative when this
ledger disagrees with Git.

## Bootstrap record

- Integration branch: `u/eaplatanios/dynamic-shapes`
- Integration baseline: `8105cfd26817ab728bb2799c889021f240345993`
- Immutable archive branch: `u/eaplatanios/archive/dimensions-wip-2026-07-24`
- Immutable archive commit: `770e77d001547c72150a44843c170ea6417ab41e`
- Immutable archive tree: `4edb3eb201ab45e474c03614a33d580dab70bf67`
- Mutable remainder branch: `u/eaplatanios/wip/dimensions-remainder`
- Mutable remainder initial commit: `770e77d001547c72150a44843c170ea6417ab41e`
- Expanded pre-snapshot status: 112 tracked changes plus 29 nonignored untracked paths, for 141 entries
- Snapshot manifest: 142 paths, including the ignored cleanup plan
- Snapshot manifest SHA-256: `428782ca2768c9dfdb2f8260a2e806f8c5af0ac91ed5503df89037662dfab206`
- Archive verification: the pre-snapshot and committed manifests matched byte-for-byte
- Archived baseline build:
  `CARGO_TARGET_DIR=/Users/eaplatanios/Development/Repositories/ryft-1-dimensions-target cargo check -p ryft-core`
  completed successfully

## S0: bootstrap plan and delivery controls

- Status: landed
- Branch: `u/eaplatanios/increment/s0-bootstrap`
- Source commit: `179483f2da4770fa675021334914baba6d801186`
- Integration commit: `fbf43052ec04ea2822f3b3753883b68b0bb42c7e`
- Remainder reconciliation commit: `979d9dd171ffab30944e80d4ab666614d96cf834`
- Immutable archive unchanged: yes
- Landed: the approved cleanup plan, this delivery ledger, and only the narrow staging-worktree `git restore`
  exception in `AGENTS.md`
- Deferred: all code extraction and semantic work begins with `S1`
- Verification: archive manifest equality and the archived `ryft-core` build passed; S0 formatting and staged-diff
  checks passed before handoff
- Residual search: branch, PR, `S6`, phase-ordering, and restore-policy searches passed before handoff
- Next action: complete and review `S1`

## S1: region arena rerooting

- Status: in progress
- Branch: `u/eaplatanios/increment/s1-region-reroot`
- Source commit: pending; fill from the pushed increment in the next ledger update
- Integration commit: pending owner review
- Remainder reconciliation commit: pending integration
- Immutable archive unchanged: yes
- Landed: pending; introduce `RegionRef::reroot` and replace reconstruction from an existing `RegionRef`'s arena
- Deferred: identity-signature retention remains in `P1`; `S1` provides the metadata-preserving seam without
  introducing identity machinery
- Verification: focused `ryft-core` compilation is blocked before reaching S1 by the pre-existing integration baseline:
  the committed custom-derivatives move references its old module path and dimension APIs absent from integration
- Residual search: the S1 production change leaves only initial arena-entry `RegionRef::new` calls and its direct
  constructor error test; final classification follows the baseline repair
- Next action: land the independent baseline repair before resuming S1 verification
