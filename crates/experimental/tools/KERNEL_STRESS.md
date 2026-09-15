# Seeded native compiler stress

`kernel_stress.py` runs reproducible mutations through the actual Mosaic and cuTile compilers, native execution,
completion, and scalar-oracle checks. Each iteration is a separate process. It varies vector tile widths (32/128/256),
edge-biased and interior extents through 2049, exact integer FP32 operands, and small matrix multiplication dimensions
through 35×35×65. Matrix cases exercise the existing staged contraction loop. Partial tiles generate valid-lane masks.
Each iteration also checks a canonical source roundtrip and rejects schema, region-index, unknown-field, and truncation
mutations. This does not claim arbitrary-operation coverage, random byte mutation, or automatic shrinking.

Build the existing experimental test binary on the qualified CUDA host (the compiler requires the pinned Python
installation described in the cuTile documentation):

```sh
cargo test -p ryft-experimental --features cuda-13,mosaic-gpu,cutile --no-run
python3 crates/experimental/tools/kernel_stress.py \
  --binary target/debug/deps/ryft_experimental-<hash> --output /tmp/kernel-stress-7 \
  --backend both --seed 7 --iterations 32 --timeout 120 --time-budget 240
```

Set `RYFT_CUTILE_PYTHON` and the normal PJRT CUDA-plugin configuration as for the existing native examples. The driver
never discovers a GPU or changes compiler selection implicitly. Missing backend features, an unavailable CUDA 13
platform, compilation failure, wrong results, missing native-completion markers, and timeouts fail the campaign.
A wrong test binary that executes zero tests cannot pass. Exit 0 means all requested iterations passed; exit 1 means
one failed/timed out, and exit 2 means the campaign budget expired before the next iteration. Budget exhaustion never
counts as completion of the requested count. The process timeout kills and reaps the isolated process group, including
compiler descendants. No other process group is modified.

Every case directory is created without replacement. `replay.json` is written before launch and records binary SHA-256,
seed, index, backend, command, and exact case environment. The Rust fixture saves the generated descriptor and canonical
IR before execution, then writes each malformed input before decoding. `run.log` retains compiler/runtime diagnostics;
`campaign.jsonl` distinguishes started, passed, failed, and interrupted work. Keep the binary, dependencies and parent launch script (including plugin/archive and linker configuration)
alongside the ledger for exact replay. Only the case controls, `RYFT_CUTILE_PYTHON`, and `CUDA_VISIBLE_DEVICES` are
recorded from the environment; the driver never dumps unrelated environment values. The existing cuTile runner also
writes its artifact, manifest and executable into the case's own `artifacts` directory, whose path is recorded. The
descriptor plus the source revision identifies generator semantics; the seed alone does not promise compatibility across generator changes.

Replay one failure in a fresh directory with the same binary, seed and index:

```sh
python3 crates/experimental/tools/kernel_stress.py \
  --binary target/debug/deps/ryft_experimental-<hash> --output /tmp/kernel-replay-7 \
  --backend both --seed 7 --start-iteration 13 --iterations 1 --timeout 120 --time-budget 120
```

Use explicit longer campaign budgets for sustained qualification and record the completed count and seeds. A short
owner test or an eight-case smoke corpus is not evidence of a sustained campaign. Performance comparisons are separate:
process startup, compilation and readback make these tests unsuitable as throughput benchmarks.
