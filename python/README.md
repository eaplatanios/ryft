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

Run the behavioral and StableHLO differential suite against the repository-pinned JAX build with:

```bash
uv run python scripts/compare_behavior_with_jax.py
```

The harness runs matched Ryft and JAX workloads, compares exact values, and compares the semantic collective subset of
their StableHLO (collective family, ordered groups, and axis attributes). It also pins intentional capability
differences: the bounded data-dependent prefix case must execute eagerly in both frameworks, stage in Ryft, and fail
JAX staging with concretization. Use `--list` to show the case registry and repeat `--case CASE_ID` to run a subset.

### Collective correctness on one host

The collective suite runs real compiled XLA programs across logical CPU PJRT devices in a single process. It works
without multiple GPUs or nodes, including on DGX Spark. Each participating CPU device receives its own local buffers;
the test waits for execution and host transfers before checking results. JAX runs in a fresh subprocess and the harness
explicitly selects its CPU backend, regardless of the host's available accelerators.

From the repository root, build the Rust emitter once, then switch to the Python utilities and run the suite:

```bash
# From the repository root:
timeout 300 cargo build -p ryft-xla --features differential-testing --bin differential_testing

# From python/:
cd python
uv sync
uv run python scripts/compare_behavior_with_jax.py --suite collectives --list
timeout 300 uv run python scripts/compare_behavior_with_jax.py \
  --suite collectives --ryft-binary ../target/debug/differential_testing
```

Without `--ryft-binary`, the harness uses `cargo run`. Each framework subprocess has a 300-second timeout; `--timeout`
sets a different budget explicitly. A cold build may need a separate build invocation before running the comparisons.
The Rust binary also accepts `--suite collectives`, `--list`, and repeated `--case` arguments for direct investigation.

The suite contains the three original collective workloads and 51 additional experiments described by the shared
[`collective_cases.json`](../crates/ryft-xla/src/bin/differential_testing/collective_cases.json) manifest. Both framework
emitters consume that manifest. Each backend is checked separately against independent NumPy reference semantics, so
matching wrong results cannot pass. The references assemble participant-local arrays directly; reverse-mode references
compute the transpose of the host linear map on basis vectors, rather than copying either backend's derivative rule.

| Coverage | Executed cases |
| --- | --- |
| Participants | 1, 2, 4, and 8 CPU devices; one-dimensional and 2-by-2 meshes |
| Group ordering | Noncontiguous groups, reversed participant order, and both mesh axes |
| Shape-changing collectives | Tiled/untiled all-gather, sum-scatter, and all-to-all; distinct split/concatenation axes |
| Other operations | Sum, mean, min, max, and product reductions; partial/cyclic permutations; shuffle; swap axes; axis index; replicated input variation |
| Transforms | Actual batching, JVP, and VJP execution for supported linear collectives, including grouped VJPs |
| Derivative checks | Participant-dependent tangent/cotangent seeds, exact host transpose references, and the adjoint inner-product identity |

Actual output shapes are checked before participant-local values are flattened. Primal cases with a common direct
lowering also compare ordered collective groups and axis attributes in StableHLO. Product reduction uses a gather plus
local product in JAX, so its cross-framework contract is numerical rather than identical primitive lowering.

These additional experiments use small static `f32` arrays. They supplement core operation tests for data types,
validation, dynamic geometry, and all-gather's invariant/reduced variation semantics; they do not establish exhaustive
coverage of those dimensions. Add cases to the shared manifest to extend the finite execution matrix.

Run the harness's own tests, including the live JAX matrix, with:

```bash
timeout 300 uv run python -m unittest tests.test_collective_testing tests.test_differential_testing
```

`parallel_ragged_all_to_all` cannot execute on XLA's CPU backend. This is also a JAX limitation: its CPU compilation
fails with `HLO opcode 'ragged-all-to-all' is not supported by XLA:CPU ThunkEmitter`. See
[JAX's ragged lowering](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/parallel.py) and
[XLA's CPU opcode dispatcher](https://github.com/openxla/xla/blob/main/xla/service/cpu/thunk_emitter.cc).
Ryft's target validation reports that restriction before compilation. Core eager/named-batching tests exercise ragged
transfer semantics and adjoints, including empty transfers, repeated source reads, grouped routing, and untouched
output-seed regions; lowering tests cover its accelerator custom-call contract and explicit CPU rejection.

CPU execution does not test CUDA/NCCL communication, hardware topology, network failures, or performance. Native
cross-process kernel execution also remains outside this suite. Those require separate integration coverage. A
single-participant ragged exchange can still perform native GPU copies, but it does not establish multi-GPU routing
correctness.

The explicitly selected CUDA suite runs eight ragged all-to-all workloads on one real GPU, including a DGX Spark.
It checks seeded output holes, empty transfers, repeated source reads, trailing row dimensions, unrelated batching,
JVPs, and VJPs of both the operand and output seed with `i32` and `u64` metadata. All transfer metadata are runtime
arguments. Ryft and JAX outputs are checked against independent NumPy copies and basis-vector adjoints, including
actual array shapes and `f32` element types. VJP cases also check the adjoint inner-product identity.

Build with CUDA 13 from the repository root, then run from `python/`:

```bash
timeout 300 cargo build -p ryft-xla --features differential-testing,cuda-13 --bin differential_testing
cd python
CUDA_VISIBLE_DEVICES=0 uv run --with 'jax[cuda13]==0.10.0' python -m ryft.jax.differential_testing \
    --suite cuda-collectives --ryft-binary ../target/debug/differential_testing
```

Omit `--ryft-binary` to let the harness build with the CUDA feature automatically. CUDA cases are excluded from default
and CPU-suite execution; missing CUDA support fails explicitly. Select individual cases with `--case`, or inspect them
with `--suite cuda-collectives --list`. CPU and CUDA workloads run in separate invocations. Each emitter has a
300-second timeout that also terminates its child processes.

Ryft's CUDA plugin also needs compatible shared libraries on its loader path. If system CUDA is older than the
plugin's build toolkit, set `LD_LIBRARY_PATH` to compatible library directories; installing JAX's CUDA extra alone
does not expose those libraries to the separate Rust process. On the Spark, the plugin required NVRTC 13.2 builtins
and NVJitLink 13.2 or newer. The eight cases passed on its GB10 with driver 580.173.02 on 2026-10-06, using isolated
CUDA libraries. Native compilation dumps contained `kRaggedAllToAll` GPU thunks.

Repeated source reads are covered in primal and JVP execution. Pinned JAX's ragged transpose overwrites overlapping
operand cotangents, so CUDA VJP parity cases use disjoint source intervals; Ryft's additive transpose has separate core
regression coverage. These single-GPU tests execute the accelerator backend and verify local transfer geometry, but
the separate distributed suite below exercises cross-process routing and NCCL communication on the same GPU.

The `cuda-distributed-collectives` suite launches two independent PJRT processes sharing CUDA device 0. It uses
[NCCL's experimental multiple-ranks-per-GPU support](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-multi-rank-gpu-enable),
available since NCCL 2.30, and CUDA MPS for concurrent progress. The loaded NCCL library must support that option;
an older system library can take precedence over the version installed with JAX. Include the compatible NCCL directory
in `LD_LIBRARY_PATH` alongside the CUDA libraries needed by Ryft. The verified Spark configuration used NCCL 2.32.3
from `site-packages/nvidia/nccl/lib`, plus compatible NVRTC and NVJitLink libraries from `nvidia/cu13/lib`.

After the CUDA build above, run from `python/`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --with 'jax[cuda13]==0.10.0' python -m ryft.jax.differential_testing \
    --suite cuda-distributed-collectives --ryft-binary ../target/debug/differential_testing \
    --worker-log-directory /tmp/ryft-distributed-logs
```

The launcher starts a private `nvidia-cuda-mps-control` daemon for each framework and shuts it down afterward. If
`CUDA_MPS_PIPE_DIRECTORY` is supplied, it uses that caller-owned daemon without stopping it. It enables
`NCCL_MULTI_RANK_GPU_ENABLE=1`, disables NVLS, limits NCCL to one CTA per rank, sets each MPS client to 45% active
threads, and disables allocator preallocation. Each framework's worker pair has a 300-second deadline, configurable
with `--timeout`; PJRT/JAX coordination has 30-second deadlines. Worker failure or timeout cancels both process groups,
including descendants. The optional log directory retains each rank's stdout, stderr, and NCCL diagnostics.

Seven cases check sum all-reduce, tiled all-gather, shape-changing all-to-all, and four ragged exchanges: asymmetric
`i32` routing, trailing rows with `u64` metadata, zero transfers, and repeated source reads. Inputs differ by rank;
ragged metadata remain runtime arguments and seeded output holes are checked. Both frameworks must agree with an
independent host reference after execution and host transfer complete. Actual local output shapes and element types
are validated, and semantic StableHLO contracts are checked. These cases share
[`distributed_collective_cases.json`](../crates/ryft-xla/src/bin/differential_testing/distributed_collective_cases.json).
Use `--suite cuda-distributed-collectives --list` or repeated `--case` options to inspect or narrow the suite. It is
excluded from default execution and runs separately from CPU and single-process CUDA cases.

All seven cases passed on the DGX Spark's GB10 with driver 580.173.02 and NCCL 2.32.3 on 2026-10-06, including a full
run with automatic MPS startup and cleanup. Native logs showed two communicator ranks on the same physical GPU,
P2P/CUMEM transport, and `kRaggedAllToAll` GPU thunks. This covers real cross-process communication and routing;
separate GPUs/nodes are still needed to validate physical interconnects, remote failures, and bandwidth.

Run the orchestration and independent-reference regressions without CUDA with:

```bash
timeout 300 uv run python -m unittest tests.test_distributed_testing tests.test_distributed_collective_testing
```

To investigate local reference state inside custom JVP rules, run:

```bash
uv run python scripts/compare_reference_linearization_with_jax.py
```

This probe reports direct JVPs, repeated linearized calls, gradients, and pushforward Jaxprs for equivalent square
functions, including pure and reference-based controls. Its JSON output records the installed JAX version and backend;
it does not assume that a particular JAX version must fail. The matching Ryft regression is
`test_custom_jvp_linearization_keeps_tangent_state_fresh_and_hoists_coefficients` in `custom_jvp.rs`.

To check loop-carried dependencies and reference ordering, run:

```bash
uv run python scripts/compare_loop_carries_with_jax.py
```

This probe checks scan and while results, zero-iteration loops, repeated pushforward calls, and reconstruction from
known and residual programs. It also reports whether explicit reference carries and dynamic-while reverse mode are
supported, separately from numerical correctness. The partition checks use the pinned JAX internal partial evaluator;
the execution checks use public APIs. Corresponding Ryft carry-chain regressions live in `scan.rs` and `while.rs`.

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
`ryft.jax.preserved_dump_cases`, the differential-testing workflow lives under `ryft.jax.differential_testing` and
`ryft.jax.differential_testing_cases`, and the runtime transform benchmark helpers live under
`ryft.jax.transform_performance`.
