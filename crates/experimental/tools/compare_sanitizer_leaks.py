#!/usr/bin/env python3
"""Compares cuDNN initialization allocations in same-runtime compute-sanitizer memcheck runs.

Run the ordinary PJRT compilation control and kernel probes with identical plugin, libraries, and sanitizer options.
Only identical cuDNN initialization allocation stacks are admitted; invalid accesses and other leaks remain failures.
Keep the raw logs alongside the comparison: matching retained allocations do not establish leak-free execution.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


def parse_log(text: str) -> tuple[Counter, tuple[str, ...]]:
    """Validates a complete memcheck log and extracts allocation stacks through cuDNN initialization."""
    results = re.findall(
        r"^test result: ok\. (\d+) passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out;.*$", text, re.MULTILINE
    )
    if len(results) != 1 or int(results[0]) == 0 or len(re.findall(r"^test result:", text, re.MULTILINE)) != 1:
        raise ValueError("expected one successful nonempty Rust test run with no ignored tests")
    runtime = tuple(sorted(set(re.findall(r"StreamExecutor \[\d+\]: (.+)", text))))
    if not runtime:
        raise ValueError("missing CUDA runtime device description")

    allocations = Counter()
    size = None
    frames = []
    initialized = False
    leak_summary = None
    error_summary = None
    banner_count = 0
    for line in text.splitlines():
        if not line.startswith("========="):
            continue
        diagnostic = line.removeprefix("=========").strip()
        if not diagnostic:
            continue
        if diagnostic == "COMPUTE-SANITIZER" and banner_count == 0:
            banner_count += 1
            continue
        if banner_count != 1:
            raise ValueError("sanitizer diagnostic before tool banner")
        leak = re.fullmatch(r"Leaked (\d+) bytes at 0x[0-9a-fA-F]+", diagnostic)
        summary = re.fullmatch(r"LEAK SUMMARY: (\d+) bytes leaked in (\d+) allocations", diagnostic)
        if leak or summary:
            if leak_summary is not None or error_summary is not None:
                raise ValueError("allocation record after sanitizer summary")
            if size is not None:
                if not initialized:
                    raise ValueError("retained allocation is not attributable to `CudnnSupport::Init`")
                allocations[(size, tuple(frames))] += 1
            size = None
            frames = []
            initialized = False
            if leak:
                size = int(leak[1])
            else:
                leak_summary = (int(summary[1]), int(summary[2]))
            continue
        if diagnostic == "Saved host backtrace up to driver entry point at allocation time" and size is not None:
            continue
        frame = re.fullmatch(r"Host Frame: (.*?) \[(0x[0-9a-fA-F]+)\] in (.+)", diagnostic)
        if frame and size is not None:
            if not initialized:
                frames.append((frame[1], frame[2], Path(frame[3]).name))
                initialized = "CudnnSupport::Init()" in frame[1]
            continue
        errors = re.fullmatch(r"ERROR SUMMARY: (\d+) errors?", diagnostic)
        if errors and leak_summary is not None and error_summary is None:
            error_summary = int(errors[1])
            continue
        raise ValueError(f"unrecognized or misplaced sanitizer diagnostic: {diagnostic}")

    count = sum(allocations.values())
    total_bytes = sum(size * count for (size, _), count in allocations.items())
    if banner_count != 1 or leak_summary != (total_bytes, count) or error_summary != count:
        raise ValueError("missing, inconsistent, or non-leak sanitizer summaries")
    return allocations, runtime


def compare_logs(control: str, probe: str) -> tuple[int, int]:
    """Requires exact allocation multiplicities and matching runtime descriptions, returning count and bytes."""
    expected, control_runtime = parse_log(control)
    actual, probe_runtime = parse_log(probe)
    if control_runtime != probe_runtime:
        raise ValueError("control and probe CUDA runtime device descriptions differ")
    if expected != actual:
        raise ValueError(f"retained allocations differ: added={actual - expected}; missing={expected - actual}")
    return sum(actual.values()), sum(size * count for (size, _), count in actual.items())


def main() -> int:
    """Compares two raw memcheck logs and reports the retained upstream allocation totals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("probe", type=Path)
    arguments = parser.parse_args()
    try:
        count, size = compare_logs(arguments.control.read_text(), arguments.probe.read_text())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{count} cuDNN initialization allocations retained, {size} bytes; no additional kernel allocations retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
