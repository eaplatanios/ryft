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

Run the following command to verify the shared Rust IR benchmark suite against the committed Python snapshots:

```bash
uv run python scripts/compare_benchmark_mlir_with_jax.py
```

The committed benchmark IR snapshots live under `python/tests/snapshots/ir_benchmark`.

The reusable helpers that back the benchmark snapshot workflow live under `ryft.jax.ir_analysis`,
`ryft.jax.benchmark_cases`, `ryft.jax.benchmark_snapshots`, and `ryft.jax.benchmark_parity`.
