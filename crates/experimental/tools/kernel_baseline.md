# Kernel regression baselines

`kernel_baseline.py` consumes measurements exported by the existing native AOT/tuning qualification runners. It does
not compile, execute or time kernels. The producer must await completion and validate outputs before recording a
sample. Host completion plus readback is a host measurement, not a GPU hardware counter. Lowering includes adapter
work; compilation may include an existing cache hit. Keep those meanings explicit in the methodology.

The existing `test_aot_on_cuda` fixtures export reports when `RYFT_KERNEL_PROFILE_DIRECTORY` is set. Also supply
`RYFT_KERNEL_SOURCE_SHA256` from the retained source manifest and a nonempty `RYFT_KERNEL_PROFILE_ENVIRONMENT`
identifying the physical runner, GPU/driver and workload/clock controls that are not supplied by live PJRT facts.
`RYFT_KERNEL_AOT_ITERATIONS` controls completed samples per process (default 128, maximum 8192). The exporter saves
raw timings separately and contributes one process upper median, plus existing lowering/compile-or-cache durations
and exact bundle/test-binary sizes. Run each process with a fresh output directory and retain its launch script.

## Measurement schema

Each JSON report has exactly these fields:

```json
{
  "schema": 1,
  "identity": {
    "semantic_digest": "<64 lowercase SHA256 hex characters>",
    "configuration_digest": "<64 lowercase SHA256 hex characters>",
    "execution_digest": "<64 lowercase SHA256 hex characters>",
    "environment": "GPU, driver, plugin and controlled machine conditions",
    "methodology": "one process median after declared warmup; cold compilation cache"
  },
  "provenance": [
    {"source_sha256": "<64 hex characters>", "binary_sha256": "<64 hex characters>", "label": "run 1"}
  ],
  "metrics": {
    "host_completion_readback": {"unit": "ns", "samples": [100, 101, 99, 100, 100]},
    "aot_bundle": {"unit": "bytes", "samples": [4096]}
  }
}
```

The digest placeholders must be replaced with real digests. Reuse canonical semantic, binding configuration and
execution-facts identities. Source and executable hashes belong to provenance so independently built candidates can
be compared. `label` is optional; all other fields are required. Metrics may have other descriptive names but their
units must be `ns` or `bytes`. Samples are unsigned 64-bit integers. Unknown fields, duplicate JSON keys, nonfinite
numbers, oversized files and malformed hashes are errors.

Do not pool different shapes, compiler settings, cache conditions, timing boundaries or machines. The comparator
requires exact identity, metric-name and unit equality. A changed fingerprint requires a separately reviewed baseline;
it cannot silently inherit a previous budget. Keep source/build manifests beside reports for inspection.

## Collection and review

1. Declare workload, warmup, repeat count, cache policy, timing boundary and machine controls before collection. Use
   the existing runner and analysis report. Record unavailable counters as unavailable outside this metric envelope;
   never substitute zero. Synchronize completion and check numerical correctness for every measured workload.
2. Collect at least five independent process runs for timing. If each run measures many invocations, export that
   process's upper median as one sample when the chosen methodology is process medians. Preserve raw invocation
   measurements separately. Concatenating within-process samples does not create independent process runs.
3. Merge reports with identical identities. Retain the exact merged baseline bytes and its SHA256. Review an explicit
   budget file, including a reason for each metric's stability tolerance and allowed increase. Thresholds must come
   from measurement variability and product requirements; this tool supplies no release defaults.
4. Repeat the same collection for the candidate, compare, and retain reports, budget, hashes and comparison output.
   An unstable baseline or candidate fails qualification before a regression can pass. Investigate and recollect;
   do not loosen thresholds automatically or discard slow samples without a recorded methodological reason.

```sh
python3 crates/experimental/tools/kernel_baseline.py merge baseline-run-*.json > baseline.json
shasum -a 256 baseline.json
python3 crates/experimental/tools/kernel_baseline.py merge candidate-run-*.json > candidate.json
python3 crates/experimental/tools/kernel_baseline.py compare baseline.json candidate.json budget.json > comparison.json
```

The budget has exactly `schema` (1), `baseline_sha256` (the exact baseline file hash), a nonempty overall `reason`, and
`metrics`. Its metric names must exactly equal the report's. Each metric policy requires:

| Field                       | Meaning                                                                 |
|-----------------------------|-------------------------------------------------------------------------|
| `direction`                 | Exactly `"lower"`; latency and size increases are the supported case.     |
| `minimum_samples`           | Explicit count, at least 5 for `ns` and 1 for `bytes`.                    |
| `maximum_relative_spread`   | Nonnegative decimal limit on `(maximum - minimum) / upper_median`.       |
| `maximum_ratio`             | Decimal multiplier of at least 1 for the baseline upper median.          |
| `absolute_allowance`        | Unsigned integer allowance in the metric's own units.                    |
| `reason`                    | Nonempty explanation of the reviewed metric policy.                     |

The regression bound is `candidate_upper_median <= baseline_upper_median * maximum_ratio + absolute_allowance`.
Decimal policy values are compared exactly, including boundary equality. Policy numbers must be finite, use at most
38 coefficient digits and a decimal exponent between -38 and 38, and have magnitude at most 1,000,000. These limits
are checked before rational conversion to prevent tiny exponent notation from allocating enormous integers. For an even sample count, upper median
means the sorted sample at zero-based index `count / 2`. A zero median is stable only if all samples are equal to zero.
Both sides must satisfy the sample count and spread limits. This is a deterministic budget check, not a statistical
confidence interval; sample independence and machine control remain the producer's responsibility.

The CLI returns 0 on success, 1 on unstable measurements or regression, and 2 on malformed or incompatible inputs.
`merge` concatenates samples and provenance without resampling or removing outliers. Each file and merged output is
bounded to 8 MiB, and each metric has at most 100,000 samples. The baseline hash detects accidental replacement; it
does not authenticate an untrusted producer or approve budgets. A self-comparison validates collection/tooling only
and does not establish a historical regression claim.

Run owner tests with `python3 crates/experimental/tools/kernel_baseline_test.py`. Continue using
`compare_sanitizer_leaks.py` for sanitizer deltas; this tool does not replace correctness, leak, race or synchronization
qualification.
