---
name: upgrade-xla
description: >-
  Upgrade Ryft's OpenXLA dependency and propagate native artifact, PJRT, Protobuf, MLIR, Triton, and JAX Mosaic
  compatibility changes. Also use for native artifact, generated-binding, Protobuf, distributed-runtime bridge,
  and PJRT argument-struct maintenance.
---

# Upgrade XLA

Treat an OpenXLA upgrade as a cross-crate source and native-binary change. Follow the detailed
[Contribution checklist](../../../crates/ryft-xla-sys/README.md#contribution) and the current build configuration.
Read [AGENTS.md](../../../AGENTS.md) for workflow, Git safeguards, and shared convention routing, and the
[native rules](../../../crates/ryft-xla-sys/AGENTS.md) for C/C++ conventions. For affected bindings, also read the
[FFI conventions](../../ffi-conventions.md) and the owning crate instructions linked from the root. This skill owns
the upgrade and artifact-specific requirements.

Apply only the requested mode: a full upgrade follows the workflow below; a binding repair or artifact rebuild applies
its relevant steps at the existing pins without selecting a new revision, creating an upgrade branch, or starting
publication unless requested. Focused Protobuf, distributed-runtime bridge, or PJRT argument-struct work follows the
[relevant binding audit](references/bindings.md) and affected verification only. Reviewing or editing this skill does
not authorize an upgrade, workflow dispatch, or publication.

## Establish the Revision and Build Plan

For a requested upgrade, use the specified OpenXLA revision, or resolve the latest commit from the
[official repository](https://github.com/openxla/xla) when the user requests the latest or supplies no target. Record
the full old and new commit identifiers and verified source archive checksums in the task plan. Use exact pinned
sources throughout the audit, not current upstream `main` as a substitute for a dependency's selected version.

Inspect these sources before editing:

- `crates/ryft-xla-sys/WORKSPACE`, `.bazelversion`, `.bazelrc`, `BUILD.bazel`, and `patches/` for source pins,
  dependency initialization, toolchains, and local compatibility changes.
- `crates/ryft-xla-sys/build.rs` and `Cargo.toml` for artifact selection, checksums, binding generation, and features.
- `.github/workflows/build_ryft_xla_sys.yaml` for the current platform matrix, checks, artifact names, and publication.
- The selected OpenXLA dependency declarations for exact LLVM/MLIR, StableHLO, Triton, and toolchain revisions;
  `JAX_COMMIT` and JAX's pinned sources for Mosaic dialects, runtime contracts, and build dependencies.
- Affected downstream crate manifests, wrappers, and changelogs. At minimum inspect `ryft-pjrt` and `ryft-mlir`;
  include `ryft-xla`, `ryft-triton`, and other consumers when the dependency changes reach them.

Update `XLA_COMMIT` and `XLA_SHA256` together. Change `JAX_COMMIT` and `JAX_SHA256` only when required by the selected
upgrade, and inspect Mosaic compatibility whenever that pin changes. Synchronize the effective `rules_ml_toolchain` pin
and initialization with the new OpenXLA revision, including any explicit local override in `WORKSPACE`. The current
workspace can inherit this pin through OpenXLA's workspace initialization; do not invent a redundant override. Align
hermetic `--repo_env` pins in `.bazelrc`, including ROCm distribution settings, with that toolchain and the selected
OpenXLA configuration (e.g., `workspace3.bzl` and `third_party/gpus/rocm_configure.bzl`, or their current successors).
Repository-rule mismatches can fail before compilation. Revalidate local patches and Bazel target names against the
selected sources. Preserve Bazel sandboxing, a consistent MLIR C API target family, and registration retention as
explained in the native README.

## Start the Native Build Early

For a full upgrade, create the branch `u/eaplatanios/upgrade-xla-<commit>` using the selected OpenXLA commit, subject
to the repository's working-tree safeguards. Prepare the pin/toolchain patch and a concrete workflow plan first, then
start the native build as early as a suitable pushed revision is available. Continue binding and wrapper work while
it runs; native builds may take hours.

The workflow requires committed, pushed inputs, but `AGENTS.md` reserves staging and index curation for the user.
Do not stage, unstage, commit unrelated staged work, or use destructive Git commands to satisfy this requirement.
Use an already pushed suitable revision or a user-curated commit. If neither exists, finish the reviewable patch and
independent compatibility analysis, then explain the exact Git safeguard and request the needed user-curated commit
only when that prerequisite actually blocks dispatch. This skill does not grant an exception to the safeguard.

Before dispatch, verify the pushed revision contains the intended native inputs. The workflow is also a publisher:
its release job writes assets under `ryft-xla-sys-<XLA_COMMIT>`. Proceed when publication is within the user's
authorized upgrade/rebuild scope. If the user requested only preparation or review, stop with the concrete patch and
build plan. A second dispatch on the same branch can cancel its running build, so inspect an existing run before
restarting it.

Use the README's workflow commands with the verified repository and branch. Record the exact run ID, commit, and
release tag. Monitor that run approximately every ten minutes using brief status requests, continuing independent
work between checks. Avoid an hours-long blocking watch. Diagnose a failed run and resolve confirmed causes before
retrying; do not repeatedly dispatch unchanged builds or replace a user-specified run with another silently.

## Synchronize the Public Surface

Read [Binding and compatibility audit](references/bindings.md) before changing bindings or wrappers. Compare old and
new pinned PJRT APIs/extensions, Protobuf messages, distributed-runtime bridges, LLVM/MLIR, StableHLO, Triton, and,
when affected, JAX Mosaic. Track inspected surfaces, changes, verification, and blockers in the plan. Update safe
APIs and tests together with their raw definitions, and propagate required compatibility changes through consumers.

For MLIR dialect changes, also use [add-mlir-dialect](../add-mlir-dialect/SKILL.md) for dialect-specific wrapper,
shim, registration, and test conventions. Do not treat its use as a request to add unrelated dialects.

## Finalize Artifacts and Verify

Preserve `build.rs` resolution order: environment-provided artifact, verified download, then Bazel build where
supported. Keep checksum verification and artifact naming/URL logic explicit and current; do not enable a stale
binary to satisfy checks for new source.

When publication is in scope, wait for the exact build run to complete successfully, then verify the expected release
tag and finalized assets. Update `BuildConfiguration::precompiled_artifact_name`, `precompiled_artifact_url_prefix`,
and `precompiled_artifact_checksum` where affected. Refresh every affected published checksum from the final release
asset bytes, not an earlier run or a locally rebuilt substitute. Use
`shasum -a 256 /absolute/path/to/actual-asset.tar.gz` for each downloaded asset, accounting for the subdirectories
created by workflow-artifact downloads. Confirm that any workflow download used for verification matches the asset
actually published.

Run the native archive/source/symbol contract checks described in the README and workflow against the final artifact
and its exact pinned source roots. Do not substitute compilation for executable hardware coverage: GPU build rows
prove compilation/linking, while runtime qualification requires the matching supported environment.

After publishing or rebuilding artifacts, validate at least one downstream consumer link on **each affected platform**
against the final rebuilt artifact to catch deployment-target mismatches and stale exported symbols. When publication
is in scope, verify against the finalized published asset. Use CI for platforms unavailable locally and report any
remaining gap. Once crate code changes, run the full affected crate `--lib` unit
suites, not only smoke tests; include downstream suites whose contracts changed. Follow the repository's 300-second
local command limits and memory/concurrency safeguards. Verification must exercise the final source and binaries.

Update the changelog of every crate whose public API, wrappers, tests, or generated bindings changed, including
`ryft-mlir` and other downstream crates where applicable. When the user names a specific GitHub Actions run to await
before changing release metadata, do not mark the task complete until that exact run is `completed` and all affected
published checksums have been refreshed from finalized release assets; a failed run is a blocker, not completion.

Report selected revisions, affected crates, the build run/release, artifact verification and downstream-link evidence,
and remaining unsupported or unverified behavior. Do not claim an upgrade is complete while its required artifact,
wrapper, checksum, or platform verification work remains.
