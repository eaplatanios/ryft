# Documentation Guidelines

Read this reference when adding, changing, or reviewing documentation, including module/API rustdoc and explanatory
code comments. Preserve the documented contract, technical rationale, and links when improving prose.

- Every struct, enum, trait, module, and function should have a documentation string. Enum variants should have
  documentation unless their name is self-explanatory. If an enum variant requires a documentation string, then you must
  add documentation for all variants in that enum.
- When enum variants have documentation strings, keep an empty line between adjacent documented variants even for short
  unit variants or small enums.
- Keep an empty line between adjacent documented struct fields, including fields of private implementation structs.
- Do not attach rustdoc comments (i.e., `///`) to trait implementation blocks. Put reusable public contract
  documentation on the trait or implementing type, and explain implementation mechanics and invariants with ordinary
  code comments (i.e., `//`) at the beginning of the implementation or relevant method body.
- Prefer descriptive documentation that explains semantics and edge cases and includes examples where appropriate.
- Keep function docstrings self-contained. When relevant, introduce mathematical notation naturally in each function's
  own description rather than relying on a shared notation explanation in a trait or module docstring.
- When simplifying existing documentation, preserve its architectural rationale, invariants, and relevant API links.
  Improve wording without discarding useful information merely to shorten the text.
- When documenting `ryft` behavior, prefer stating the concrete semantics directly instead of saying that the code
  "matches" another system such as JAX unless the external comparison is itself the point of the documentation.
- Link to external official documentation when relevant (e.g., for MLIR, StableHLO, PJRT, XLA, Rustonomicon, etc.).
- When linking external documentation, prefer the most precise relevant page/section instead of just using top-level
  project pages.
- On first mention of in-repo entities, use rustdoc links (e.g., ``[`Operation`]``), and use explicit paths (e.g.,
  ``[`Operation`](crate::Operation)``) when not imported in the current scope.
- For function/method argument documentation strings, use a dedicated `# Parameters` section with this exact bullet
  style: ``///   - `arg_name`: description...`` and indent wrapped lines under the bullet. You may skip this section
  entirely in cases where the arguments do not need a description or where their role is clear from the main description
  of the function itself.
- Do not end function or method rustdoc blocks with an empty `///` line immediately before the item or its attributes.
  Keep empty rustdoc lines only for internal paragraph, list, table, or code-block separation.
- Do not add documentation strings to `From` implementations or their `from` functions. Document the conversion
  semantics, including paired conversions that form a normalizing cycle between two types, in the documentation string
  of the more domain-specific type (for example, a transform-owned error enum rather than the core `ProgramError`).
- For public `unsafe` APIs, include:
  - what handle/representation is being exposed,
  - why it is unsafe, and
  - why it is still exposed (e.g., extensibility/interoperability).
- Public items that other workspace crates need (e.g., backend hooks that `ryft-xla` calls on `ryft-core` types) are
  documented public API: explain why callers need them, the invariants callers must uphold, and any caveats (e.g.,
  locking order or reentrancy), and include `# Errors` where applicable. Do not use `#[doc(hidden)]` to avoid
  documenting them. Reserve `#[doc(hidden)]` for macro-expansion support and for items that are public only for
  type-system reasons (e.g., the payload of an uninhabited variant). Make items that are only used within their own
  crate `pub(crate)` or private instead of hiding them, and never make a type public solely to name it as a `Deref`
  target or in another public signature when the owning type can expose the needed functions directly.
- Do not mark in-repo types `#[non_exhaustive]`. Workspace crates evolve together, and the attribute forces wildcard
  `_` arms in other crates, which hide newly added variants that those crates should handle (e.g., a backend lowering
  that must cover every `ArrayReferenceTransform`). Match exhaustively instead, and use the attribute only for a
  published-API compatibility reason that the user explicitly asks for.
- For callback- and threading-heavy code, explain the lifetime/ownership invariants in comments. You can refer to
  documentation strings in the core traits of the `ryft-pjrt` crate for examples of this.
- When using ASCII diagrams or Markdown tables in doc comments, align columns for source readability and indent the
  block content slightly so that the raw Rust file remains easy to scan.
- When a `rustdoc` diagram would materially improve documentation clarity, prefer a rendered Mermaid diagram via
  `aquamarine` over a large ASCII graph, while keeping any supporting tables aligned in the raw source.
- When a large `rustdoc` table needs color or layout that Markdown cannot express cleanly, prefer theme-aware raw HTML
  with colors derived from rustdoc CSS variables instead of hard-coded light-theme colors, and prefer a small local
  `<style>` block with namespaced classes over repeating large inline styles on every cell.
- Keep raw HTML and CSS in doc comments within the repo's 120-column limit by breaking declarations, tags, and table
  cells across multiple lines instead of leaving oversized single-line blocks.
- When revising one sentence inside a documentation paragraph, reread and polish the whole paragraph so the final
  wording is coherent as a unit instead of sounding locally patched.
- When editing rustdoc prose, reflow the surrounding paragraph toward the 120-column limit where the text naturally
  allows it; avoid leaving documentation lines arbitrarily short unless they are lists, code blocks, tables, links, or
  readability-driven sentence breaks.
- In documentation strings, use "function" for callable APIs, including methods, rather than distinguishing methods
  from free functions in prose.
- For arrays and types, use "shape" for an array's shape, "extent" for the size of a dimension (static or runtime), and
  "dimension" for the dimension itself. Do not introduce "geometry" or other new umbrella nouns for these concepts.
  GPU backend crates keep "geometry" only in its standard CUDA meaning (e.g., launch/grid or descriptor geometry).
