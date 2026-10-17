# Working in Ryft

Ryft is a Rust framework for numerical computing, program transformations, and compiler/runtime integration.
Make focused, maintainable changes that fix root causes. Keep APIs small, behavior explicit, and communication concise.

## Instruction Scope and Discovery

This file owns repository-wide workflow, safeguards, and routing. Detailed conventions have one authoritative home:
shared references below, scoped crate instructions, or task-specific skills.

- Before editing a file, read any nested `AGENTS.md` files along its directory path. Do this even when starting at the
  repository root; automatic instruction loading may only cover ancestors of the initial working directory. For changes
  across crates, read each affected crate's instructions.
- Apply shared conventions throughout the repository. More specific instructions refine them for their stated scope;
  they do not waive the working-tree/index safeguards below. Follow explicit user instructions over repository defaults.
  Resolve genuine ambiguity from the owning contract rather than choosing whichever wording seems stricter.
- Read the relevant references and skills before implementation or review; a link is a routing instruction, not an
  assumption that its contents were already loaded. Consult a dependency's scoped contracts when changing callers if its
  ownership, error handling, or API conventions are relevant.

## Working-Tree and Index Safeguards

The checkout may contain long-running refactors and concurrent edits. Inspect status and relevant diffs before editing
and preserve unrelated work, including the user's staged changes.

- Never stage or unstage paths: do not use `git add`, `git rm`, `git rm --cached`, or equivalent index mutations.
  Delete files with plain `rm` when deletion is in scope; leave staging to the user.
- Never use `git stash`, `git checkout <path>`, `git reset`, or `git clean` to manage changes. Do not use `git restore`
  on tracked files except under the complete
  [staging-worktree restore exception](.agents/staging-worktree-restore-exception.md). That exception is limited to a
  clean, dedicated staging worktree and never applies to the owner checkout.
- Inspect committed versions with read-only commands such as `git show HEAD:<path>` and `git diff HEAD -- <path>`.
- Do not mutate unrelated files to work around builds broken by concurrent edits. Verify in isolation when possible,
  for example in a throwaway crate outside the workspace depending only on the crates being changed; report blockers.
- When moving or reordering code, preserve existing `TODO` comments exactly and keep each attached to its annotated code
  unless the user explicitly asks to resolve or remove it. Audit marker counts after broad mechanical moves.
- Do not manually edit generated `crates/ryft-xla-sys/src/bindings.rs` except when explicitly regenerating it as part
  of a version upgrade. Follow the upgrade skill, including its required post-generation adjustments. Avoid unrelated
  binary/editor artifacts such as `.DS_Store`.

## Workflow

1. **Understand and Plan.** Inspect the relevant implementation, tests, and current diff. For substantial changes,
   architectural decisions, or work with multiple dependent phases, write `.tasks/plan_<task_name>.md` with the intended
   behavior, scope, checkable steps, and verification. Show the plan before implementation so the user can steer it;
   this is not an approval gate unless they request one. Use the client's plan mode when appropriate and available,
   without depending on it. Small, clear edits and explicitly waived narrow follow-ups can proceed directly.
2. **Execute within Scope.** Fix reported bugs using available logs, errors, and tests without requiring hand-holding.
   Use subagents for independent research, implementation, or review when useful and available, assigning one bounded
   task to each. Reassess the plan when new evidence invalidates it; do not continue an obsolete approach.
3. **Review the Design.** Prefer existing abstractions and modules, explicit ownership, and small unsafe boundaries.
   Avoid unnecessary layers, dependencies, compatibility shims, and cleverness. Before presenting substantial work,
   check whether a simpler, more coherent solution meets the same requirements.
4. **Verify.** Inspect the final diff, run checks appropriate to the change, and resolve failures introduced by or
   directly blocking the requested work. Report unrelated failures instead of expanding the task to repair all CI.
   Keep completed plan items checked and record a short review of results and remaining limitations.
5. **Maintain Instructions Selectively.** After a durable user correction, amend its authoritative rule or skill so it
   need not be repeated. Prefer improving an existing rule over appending another. Keep one-off task facts in the task
   plan; move specialized guidance to its owner and leave a reference here.

## Verification

- Start with checks scoped to the affected crate and behavior. Include integration tests, doctests, and downstream
  checks when the changed contract warrants them; unit tests alone are not always sufficient. Check the diff before
  deciding what to run or add, and report exactly what passed, failed, or could not run.
- Start potentially expensive local verification commands with an explicit **300-second timeout per command**, unless
  the user requests a longer run. Do not leave tests or benchmarks running for many minutes by default.
- If `rustc` is killed or uses extreme memory, stop broad retries and diagnose the failure. If new generic types or
  trait-solver obligations caused it, reduce that surface before rerunning. Do not assume every kill is a type issue.
- When changing a core trait contract consumed by derive macros, run the corresponding macro integration test crate
  (e.g., `ryft-macros-tests`) in addition to the macro crate's own unit tests.
- For user-requested renames or removals, run a targeted search afterward to verify that no old identifier references
  remain in the Ryft codebase.
- Documentation-only changes need relevant link, instruction, or rustdoc checks; choose verification that can expose
  actual defects rather than running unrelated builds.

## Shared Conventions

Read each applicable reference; these contain required conventions, not optional background:

| Work                                                           | Reference                                                       |
|----------------------------------------------------------------|-----------------------------------------------------------------|
| Rust code, APIs, imports, naming, errors, lifetimes, or caches | [Rust Conventions](.agents/rust-conventions.md)                 |
| Documentation, comments, API contracts, or diagrams            | [Documentation Guidelines](.agents/documentation-guidelines.md) |
| Adding, changing, or reviewing tests                           | [Unit-Testing Guidelines](.agents/unit-testing-guidelines.md)   |
| Raw C interfaces, unsafe conversions, or safe native wrappers  | [FFI Conventions](.agents/ffi-conventions.md)                   |

Preserve architectural rationale and invariants while improving prose. Follow workspace formatting in `rustfmt.toml`
(120 columns); detailed import ownership and grouping rules are in the Rust reference.

## Task Skills

Use the relevant skill for focused maintenance and reviews as well as new implementations. Skills own their detailed
workflows; the shared references and scoped instructions still apply.

- [`add-or-review-core-operation`](.agents/skills/add-or-review-core-operation/SKILL.md): core operations, value
  capabilities/providers, transform rules or replay, reference validation, composite values, and backend integration,
  including work outside `operations/`. Also consult its core input terminology when editing programs or instructions.
- [`add-mlir-dialect`](.agents/skills/add-mlir-dialect/SKILL.md): MLIR dialect wrappers, registration, passes, and
  custom dialect C API shims; owns dialect layout, typed APIs, documentation, coverage, and native shim conventions.
- [`upgrade-xla`](.agents/skills/upgrade-xla/SKILL.md): OpenXLA dependency upgrades, native artifacts/build metadata,
  generated bindings, Protobuf synchronization, distributed-runtime bridges, PJRT API/argument synchronization,
  and release verification. Apply only the relevant maintenance workflow when a full upgrade is not requested.
