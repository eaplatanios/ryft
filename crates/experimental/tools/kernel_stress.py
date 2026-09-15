#!/usr/bin/env python3
"""Run bounded, seeded native compiler mutations in isolated test processes."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

TEST = "kernels::stress::test_compiler_case_on_cuda"


def run_case(command, environment, timeout, log_path):
    """Wait for one child and terminate its entire process group on timeout."""
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            stop(process)
            return process.returncode, True
        except BaseException:
            stop(process)
            raise


def stop(process):
    """Reap only the isolated process group, including a child exiting concurrently with cancellation."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        # The complete group already exited between the deadline and termination request.
        pass
    process.wait()


def campaign(binary, output, backend, seed, iterations, start_iteration, timeout, time_budget):
    """Record every replay before execution; stop at the first failure or exhausted wall-time budget."""
    started = time.monotonic()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("binary must name an executable test binary")
    if backend not in ("mosaic", "cutile", "both"):
        raise ValueError("backend must be mosaic, cutile, or both")
    if not (0 <= seed < 2**64 and 0 <= start_iteration < 2**64 and 1 <= iterations <= 10000):
        raise ValueError("seed, iteration, or count is out of bounds")
    if start_iteration + iterations > 2**64 or timeout <= 0 or time_budget <= 0:
        raise ValueError("iteration range and time limits must be finite and positive")
    if not all(math.isfinite(value) for value in (timeout, time_budget)):
        raise ValueError("time limits must be finite")
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with binary.open("rb") as compiled:
        for chunk in iter(lambda: compiled.read(1024 * 1024), b""):
            digest.update(chunk)
    binary_digest = digest.hexdigest()
    command = [str(binary.resolve()), "--exact", TEST, "--ignored", "--nocapture", "--test-threads=1"]
    with (output / "campaign.jsonl").open("a", encoding="utf-8") as ledger:
        for iteration in range(start_iteration, start_iteration + iterations):
            remaining = time_budget - (time.monotonic() - started)
            if remaining <= 0:
                ledger.write(json.dumps({"status": "budget_exhausted", "next_iteration": iteration}) + "\n")
                return 2
            directory = output / f"seed-{seed}-iteration-{iteration}"
            directory.mkdir()  # Never overwrite an earlier reproducer.
            environment = os.environ.copy()
            environment.update({"RYFT_KERNEL_STRESS_SEED": str(seed),
                                "RYFT_KERNEL_STRESS_ITERATION": str(iteration),
                                "RYFT_KERNEL_STRESS_BACKEND": backend,
                                "RYFT_KERNEL_STRESS_CASE_DIRECTORY": str(directory.resolve()),
                                "RYFT_CUTILE_ARTIFACT_DIRECTORY": str((directory / "artifacts").resolve())})
            replay_keys = ("RYFT_CUTILE_PYTHON", "RYFT_CUTILE_ARTIFACT_DIRECTORY", "CUDA_VISIBLE_DEVICES")
            replay_environment = {key: value for key, value in environment.items()
                                  if key.startswith("RYFT_KERNEL_STRESS_") or key in replay_keys}
            record = {"schema": 1, "seed": seed, "iteration": iteration, "backend": backend,
                      "binary_sha256": binary_digest, "command": command,
                      "environment": replay_environment,
                      "timeout_seconds": min(timeout, remaining)}
            (directory / "replay.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            ledger.write(json.dumps({**record, "status": "started"}) + "\n")
            ledger.flush()
            try:
                returncode, timed_out = run_case(command, environment, min(timeout, remaining), directory / "run.log")
            except KeyboardInterrupt:
                ledger.write(json.dumps({**record, "status": "interrupted"}) + "\n")
                ledger.flush()
                raise
            marker = f"compiler stress passed: seed={seed} iteration={iteration} backend={backend}"
            if returncode == 0:
                with (directory / "run.log").open(encoding="utf-8", errors="replace") as log:
                    completed = any(line.rstrip() == marker for line in log)
                if not completed:
                    returncode = 1
            record.update(returncode=returncode,
                          status="timeout" if timed_out else "passed" if returncode == 0 else "failed")
            ledger.write(json.dumps(record) + "\n")
            ledger.flush()
            if timed_out or returncode:
                return 1
    return 0


def main():
    """Parse explicit campaign controls without starting compiler work during discovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("mosaic", "cutile", "both"), default="both")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--start-iteration", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--time-budget", type=float, default=240)
    arguments = parser.parse_args()
    raise SystemExit(campaign(**vars(arguments)))


if __name__ == "__main__":
    main()
