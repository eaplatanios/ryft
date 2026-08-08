# `ryft` Python Utilities

This directory hosts the Python utilities that support extracting information from JAX programs that helps us ensure
that `ryft` has feature parity with JAX along certain dimensions.

## Setup

Run the following command to create a Python virtual environment for this project and install all of its dependencies:

```bash
uv sync
```

This will generate a virtual environment in `python/.venv`.

## Scripts

Run the following command to list all prespecified JAX programs that can be inspected with other scripts:

```bash
uv run python scripts/inspect_jax_programs.py --list
```

Run the following command to render the JAXPR and StableHLO for a given JAX program:

```bash
uv run python scripts/inspect_jax_programs.py --case right_mul_4_2_transpose
```

Run the following command to compare the MLIR emitted by JAX against the MLIR emitted by `ryft`:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run python scripts/compare_reshape_mlir_with_jax.py
```

Run the following command to compare backend-neutral structural program statistics between Ryft and JAX for the
shared traced workload registry:

```bash
uv run python scripts/compare_program_statistics_with_jax.py
```

The Rust side of the comparison is the `program_statistics` binary, which can also be run directly:

```bash
cargo run -p ryft-xla --features program-statistics --bin program_statistics -- --list
```

The comparison reports per-region structural statistics (instruction, input, output, and constant counts, operation
histograms, dependency depths, and attached-region graphs) for both sides and a structural diff. These are structural
counts, not performance measurements. The two sides are different IRs, so `constant_count`, attachment labels, and
region roles are reported without being diffed; cases whose registry entry is marked comparable additionally enforce
equality of the entry region's counts, normalized operation histogram, and dependency depth. The exact per-case Rust
statistics are pinned by the binary's own tests, which are the primary structural regression guard.

Run the following command to compare Ryft and JAX runtime transform overhead for the shared AD transform benchmark
cases:

```bash
uv run python scripts/compare_transform_performance_with_jax.py --iterations 1000 --warmup 50
```

The runtime comparison uses JAX's eager transform APIs and reports the Ryft/JAX median runtime ratio for each case,
exiting with a non-zero status if any selected case exceeds the configured `--max-ratio`. Note that the Rust side of
this comparison references a `transform_benchmark` binary that is currently absent from `crates/ryft-xla`; restoring
or retiring that binary is tracked separately, and until then only the JAX side of this comparison can run.

Run the matched compilation lifecycle and asynchronous-execution comparison with:

```bash
uv run python scripts/compare_compilation_performance_with_jax.py --iterations 100 --size 1048576 \
    --output /tmp/ryft-jax-compilation.json
```

Add `--smoke` for the CI-suitable counter/invariant mode. Add `--cache-dir PATH` to give each framework an isolated
persistent-cache subdirectory. The report times trace, lower, backend compile, warm dispatch, enqueue-only execution,
and explicitly synchronized execution separately; it never interprets enqueue latency as device execution latency.
Persistent-cache benchmark mode uses zero compile-duration and entry-size write thresholds in both frameworks so the
second fresh compilation context actually measures executable restoration rather than silently recompiling.

Run the following command to verify the curated preserved historical dump corpus against the committed `syrupy`
snapshots:

```bash
uv run pytest tests/test_preserved_dump_snapshots.py
```

The committed preserved historical dump snapshots live in
`python/tests/__snapshots__/test_preserved_dump_snapshots.ambr`. The corresponding curated case registry lives in
`ryft.jax.preserved_dump_cases`, and it intentionally keeps only the unique preserved programs that we still care about
instead of mirroring the old raw artifact tree.

The reusable helpers that back the program statistics workflow live under `ryft.jax.program_statistics` and
`ryft.jax.program_statistics_cases`, the preserved historical dump registry lives under
`ryft.jax.preserved_dump_cases`, and the runtime transform benchmark helpers live under
`ryft.jax.transform_performance`.
