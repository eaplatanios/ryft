#!/usr/bin/env python3
"""Merge measured kernel reports and compare an explicitly reviewed, pinned baseline."""

import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys

MAX_BYTES = 8 * 1024 * 1024
MAX_SAMPLES = 100_000
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def fields(value, expected, label):
    """Require an object with exactly the documented fields."""
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} requires exactly {', '.join(sorted(expected))}")


def nonempty(value, label):
    """Require a nonempty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def digest(value):
    """Validate a canonical SHA256 digest."""
    if not isinstance(value, str) or not DIGEST.fullmatch(value):
        raise ValueError("digest must contain 64 lowercase hexadecimal characters")


def integer(value, label, minimum=0, maximum=2**64 - 1):
    """Validate a bounded unsigned integer without accepting booleans."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")


def fraction(value, label, minimum):
    """Parse an exact decimal policy threshold."""
    if type(value) not in (int, float, Decimal):
        raise ValueError(f"{label} must be a finite number")
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError(f"{label} must be a finite number")
    coefficient = decimal.as_tuple()
    if len(coefficient.digits) > 38 or not -38 <= coefficient.exponent <= 38:
        raise ValueError(f"{label} requires at most 38 digits and an exponent in [-38, 38]")
    if decimal.copy_abs() > 1_000_000:
        raise ValueError(f"{label} must not exceed 1000000 in magnitude")
    result = Fraction(decimal)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def load(path):
    """Read bounded JSON, rejecting duplicate keys and nonfinite literals."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key `{key}`")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON number `{value}`")

    with Path(path).open("rb") as source:
        data = source.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("JSON report exceeds 8 MiB")
    try:
        value = json.loads(data, object_pairs_hook=unique, parse_constant=invalid, parse_float=Decimal)
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds decoder limit") from error
    return value, hashlib.sha256(data).hexdigest()


def validate_report(report):
    """Validate the measurement envelope; measurements remain producer-owned."""
    fields(report, ("schema", "identity", "provenance", "metrics"), "report")
    integer(report["schema"], "schema", 1, 1)
    identity = report["identity"]
    fields(identity, ("semantic_digest", "configuration_digest", "execution_digest", "environment", "methodology"),
           "identity")
    for name in ("semantic_digest", "configuration_digest", "execution_digest"):
        digest(identity[name])
    for name in ("environment", "methodology"):
        nonempty(identity[name], name)
    if not isinstance(report["provenance"], list) or not report["provenance"]:
        raise ValueError("provenance must be a nonempty list")
    for provenance in report["provenance"]:
        if not isinstance(provenance, dict) or set(provenance) not in (
                {"source_sha256", "binary_sha256"}, {"source_sha256", "binary_sha256", "label"}):
            raise ValueError("provenance requires source_sha256, binary_sha256 and optional label")
        digest(provenance["source_sha256"])
        digest(provenance["binary_sha256"])
        if "label" in provenance:
            nonempty(provenance["label"], "label")
    if not isinstance(report["metrics"], dict) or not report["metrics"]:
        raise ValueError("metrics must be a nonempty object")
    for name, metric in report["metrics"].items():
        nonempty(name, "metric name")
        fields(metric, ("unit", "samples"), name)
        if metric["unit"] not in ("ns", "bytes"):
            raise ValueError(f"metric `{name}` unit must be ns or bytes")
        samples = metric["samples"]
        if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
            raise ValueError(f"metric `{name}` requires 1 to {MAX_SAMPLES} samples")
        for sample in samples:
            integer(sample, "sample")


def compatible(left, right):
    """Require the same workload, environment, methodology and measured metrics."""
    if left["identity"] != right["identity"]:
        raise ValueError("measurement identities differ")
    if set(left["metrics"]) != set(right["metrics"]):
        raise ValueError("metric sets differ")
    for name in left["metrics"]:
        if left["metrics"][name]["unit"] != right["metrics"][name]["unit"]:
            raise ValueError(f"metric `{name}` units differ")


def merge(reports):
    """Concatenate compatible runs without changing sampling semantics."""
    if not reports:
        raise ValueError("merge requires at least one report")
    for report in reports:
        validate_report(report)
        compatible(reports[0], report)
    result = {"schema": 1, "identity": reports[0]["identity"],
              "provenance": [entry for report in reports for entry in report["provenance"]],
              "metrics": {name: {"unit": metric["unit"],
                                 "samples": [sample for report in reports
                                             for sample in report["metrics"][name]["samples"]]}
                          for name, metric in reports[0]["metrics"].items()}}
    validate_report(result)
    return result


def compare(baseline, candidate, budget, baseline_sha256):
    """Compare stable upper medians using explicit relative plus absolute budgets."""
    validate_report(baseline)
    validate_report(candidate)
    compatible(baseline, candidate)
    fields(budget, ("schema", "baseline_sha256", "reason", "metrics"), "budget")
    integer(budget["schema"], "schema", 1, 1)
    digest(budget["baseline_sha256"])
    nonempty(budget["reason"], "budget reason")
    if budget["baseline_sha256"] != baseline_sha256:
        raise ValueError("baseline SHA256 differs from reviewed budget")
    if not isinstance(budget["metrics"], dict) or set(budget["metrics"]) != set(baseline["metrics"]):
        raise ValueError("budget metric set differs")
    results = {}
    for name, metric in baseline["metrics"].items():
        policy = budget["metrics"][name]
        fields(policy, ("direction", "minimum_samples", "maximum_relative_spread", "maximum_ratio",
                        "absolute_allowance", "reason"), f"budget `{name}`")
        if policy["direction"] != "lower":
            raise ValueError("only lower-is-better metrics are supported")
        integer(policy["minimum_samples"], "minimum_samples", 5 if metric["unit"] == "ns" else 1, MAX_SAMPLES)
        integer(policy["absolute_allowance"], "absolute_allowance")
        nonempty(policy["reason"], "metric budget reason")
        spread_limit = fraction(policy["maximum_relative_spread"], "maximum_relative_spread", 0)
        ratio_limit = fraction(policy["maximum_ratio"], "maximum_ratio", 1)
        medians = []
        stable = True
        for samples in (metric["samples"], candidate["metrics"][name]["samples"]):
            median = sorted(samples)[len(samples) // 2]
            medians.append(median)
            spread = max(samples) - min(samples)
            stable &= len(samples) >= policy["minimum_samples"] and (
                spread == 0 if median == 0 else Fraction(spread, median) <= spread_limit)
        limit = medians[0] * ratio_limit + policy["absolute_allowance"]
        status = "unstable" if not stable else "pass" if medians[1] <= limit else "regression"
        results[name] = {"unit": metric["unit"], "baseline_median": medians[0], "candidate_median": medians[1],
                         "baseline_samples": len(metric["samples"]),
                         "candidate_samples": len(candidate["metrics"][name]["samples"]),
                         "maximum_candidate_median": str(limit), "status": status}
    return {"schema": 1, "baseline_sha256": baseline_sha256,
            "passed": all(result["status"] == "pass" for result in results.values()), "metrics": results}


def main():
    """Print a merged report or comparison; failures return a nonzero status."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    merging = commands.add_parser("merge")
    merging.add_argument("reports", nargs="+")
    comparing = commands.add_parser("compare")
    comparing.add_argument("baseline")
    comparing.add_argument("candidate")
    comparing.add_argument("budget")
    arguments = parser.parse_args()
    try:
        if arguments.command == "merge":
            result = merge([load(path)[0] for path in arguments.reports])
        else:
            baseline, baseline_sha256 = load(arguments.baseline)
            result = compare(baseline, load(arguments.candidate)[0], load(arguments.budget)[0], baseline_sha256)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        if len(encoded.encode()) > MAX_BYTES:
            raise ValueError("merged report exceeds 8 MiB")
        print(encoded)
        return 0 if result.get("passed", True) else 1
    except (ValueError, OSError, UnicodeError, RecursionError) as error:
        print(f"kernel baseline: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
