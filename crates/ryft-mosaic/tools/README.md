# Pinned GPU surface review

`check_surface.py` is a local, read-only upgrade gate. It hashes every tracked file in these explicit scopes:

- JAX `jax/experimental/mosaic/gpu`: authoring semantics and native lowering references.
- JAX `jaxlib/mosaic/dialect/gpu`: dialect types, operations, verification and serialization.
- JAX `jaxlib/mosaic/gpu`: native runtime, compiler and ABI implementation.
- XLA `xla/ffi/api`: custom-call ABI used by the runtime.
- XLA `xla/backends/gpu/runtime`: execution-side GPU integration.

The scope is deliberately conservative: a comment or implementation change still requires review. Files outside these
scopes are ignored, so this gate does not replace whole-upstream release review, LLVM/PTX compatibility checks, or
Ryft wrapper tests. Scope changes require reviewing the checker itself; snapshots with a different scope fail closed.
No network access, compiler invocation, source edits or CI publication occurs.

Use clean upstream Git checkouts at the exact commits in the corresponding old/new `ryft-xla-sys/WORKSPACE` files.
Archive the old WORKSPACE and snapshot before changing pins. Patched or extracted non-Git trees are rejected; local
Ryft patches require their existing independent review and qualification in addition to this upstream comparison.

```sh
python3 crates/ryft-mosaic/tools/check_surface.py snapshot \
  --workspace /path/to/old/WORKSPACE --jax /path/to/old/jax --xla /path/to/old/xla > before.json
python3 crates/ryft-mosaic/tools/check_surface.py snapshot \
  --workspace crates/ryft-xla-sys/WORKSPACE --jax /path/to/new/jax --xla /path/to/new/xla > after.json
python3 crates/ryft-mosaic/tools/check_surface.py diff before.json after.json > changes.json
```

Review each addition (`before: null`), removal (`after: null`) and content change. Write a decisions JSON object with
exact `before_pins` and `after_pins` copied from the snapshots, an overall `reason`, and a `files` object containing
exactly the changed paths. Each file entry contains the exact `before` and `after` hashes from the diff plus its own
nonempty `reason`, describing the compatibility decision and necessary wrapper/ABI/test changes. A changed pin with
an unchanged surface still needs an overall review reason. The tool does not create reasons or accept changes for you.

```sh
python3 crates/ryft-mosaic/tools/check_surface.py check before.json after.json decisions.json
timeout 300 python3 -m unittest discover -s crates/ryft-mosaic/tools -p 'check_surface_test.py' -v
```

Missing, extra, duplicate, stale or noncanonical paths and digests fail validation. Snapshots and decisions are review
artifacts, not authenticated proof that an arbitrary file came from a trusted reviewer. Keep them with immutable source
hashes and the existing native/serialization/sanitizer evidence. A passing diff decision is not an ABI compatibility
proof: rerun affected wrappers, source roundtrips, target assembly and actual device qualification.
