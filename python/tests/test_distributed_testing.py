"""Process orchestration and observation merging for two-rank CUDA testing."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ryft.jax import differential_testing as harness
from ryft.jax.differential_testing import SCHEMA


def rank_payload(value: float) -> str:
    """Returns one minimal participant-local observation with a shared module contract."""

    return json.dumps([{
        "schema": SCHEMA, "case_id": "case", "observations": {"primal": [[value]]}, "stablehlo": "module {}",
    }])


class DistributedTestingTest(unittest.TestCase):
    """Checks suite isolation, rank merging, cancellation, and private MPS ownership without requiring CUDA."""

    def test_distributed_suite_selection(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(harness.main(("--suite", "cuda-distributed-collectives", "--list")), 0)
        identifiers = output.getvalue().splitlines()
        self.assertEqual(len(identifiers), 7)
        self.assertTrue(all(identifier.startswith("cuda_distributed_") for identifier in identifiers))
        self.assertTrue(set(identifiers).isdisjoint(case.case_id for case in harness._selected_cases(())))
        with self.assertRaisesRegex(ValueError, "single-process and distributed CUDA"):
            harness._selected_cases((identifiers[0], "cuda_ragged_seed_holes_i32"))

    def test_merge_distributed_observations(self) -> None:
        records = harness._merge_distributed_observations((rank_payload(1.), rank_payload(9.)), ("case",), "test")
        self.assertEqual(records[0].observations, {"primal": ((1.,), (9.,))})
        changed = json.loads(rank_payload(9.))
        changed[0]["stablehlo"] = "module {changed}"
        with self.assertRaisesRegex(ValueError, "disagree on the module contract"):
            harness._merge_distributed_observations((rank_payload(1.), json.dumps(changed)), ("case",), "test")
        changed = json.loads(rank_payload(9.))
        changed[0]["observations"]["primal"] = [[9.], [10.]]
        with self.assertRaisesRegex(ValueError, "one local output each"):
            harness._merge_distributed_observations((rank_payload(1.), json.dumps(changed)), ("case",), "test")
        with self.assertRaisesRegex(ValueError, "rank case set"):
            harness._merge_distributed_observations((rank_payload(1.), "[]"), ("case",), "test")
        with self.assertRaisesRegex(ValueError, "JSON array"):
            harness._merge_distributed_observations((rank_payload(1.), "{}"), ("case",), "test")

    def test_module_entry_point_uses_canonical_contract_types(self) -> None:
        # Registry contracts belong to the imported module; `python -m` must dispatch through that same instance.
        with patch("ryft.jax.differential_testing.main", return_value=0) as entry_point, warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(SystemExit) as result:
                runpy.run_module("ryft.jax.differential_testing", run_name="__main__")
        self.assertEqual(result.exception.code, 0)
        entry_point.assert_called_once_with()

    def test_shared_gpu_environment(self) -> None:
        with patch.dict(os.environ, {"CUDA_MPS_PIPE_DIRECTORY": "/provided/mps"}, clear=True), \
                patch("ryft.jax.differential_testing.subprocess.run") as run:
            with harness._shared_gpu_environment() as environment:
                self.assertEqual(environment["CUDA_MPS_PIPE_DIRECTORY"], "/provided/mps")
                self.assertEqual(environment["NCCL_MULTI_RANK_GPU_ENABLE"], "1")
                self.assertEqual(environment["NCCL_NVLS_ENABLE"], "0")
                self.assertEqual(environment["NCCL_MAX_CTAS"], "1")
                self.assertEqual(environment["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
            run.assert_not_called()
        with patch.dict(os.environ, {}, clear=True), \
                patch("ryft.jax.differential_testing.shutil.which", return_value="/mps-control"), \
                patch("ryft.jax.differential_testing.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "worker failure"):
                with harness._shared_gpu_environment() as environment:
                    self.assertTrue(Path(environment["CUDA_MPS_PIPE_DIRECTORY"]).is_dir())
                    raise ValueError("worker failure")
            self.assertEqual([call.args[0] for call in run.call_args_list], [["/mps-control", "-d"], ["/mps-control"]])
            self.assertEqual(run.call_args.kwargs["input"], "quit\n")
            self.assertFalse(Path(environment["CUDA_MPS_PIPE_DIRECTORY"]).exists())
        with patch.dict(os.environ, {}, clear=True), \
                patch("ryft.jax.differential_testing.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "requires `nvidia-cuda-mps-control`"):
                with harness._shared_gpu_environment():
                    self.fail("MPS was unavailable")

    def test_shared_gpu_environment_startup_failure_attempts_shutdown(self) -> None:
        timeout = subprocess.TimeoutExpired("mps-control", 10)
        with patch.dict(os.environ, {}, clear=True), \
                patch("ryft.jax.differential_testing.shutil.which", return_value="/mps-control"), \
                patch("ryft.jax.differential_testing.subprocess.run", side_effect=[timeout, None]) as run:
            with self.assertRaisesRegex(RuntimeError, "private CUDA MPS startup failed"):
                with harness._shared_gpu_environment():
                    self.fail("startup must fail")
            self.assertEqual([call.args[0] for call in run.call_args_list], [["/mps-control", "-d"], ["/mps-control"]])
            self.assertEqual(run.call_args.kwargs["input"], "quit\n")
            self.assertFalse(Path(run.call_args.kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"]).exists())

    def test_run_distributed_emitters(self) -> None:
        program = "import sys; print(sys.argv[sys.argv.index('--distributed-worker') + 1])"
        with tempfile.TemporaryDirectory() as directory, \
                patch("ryft.jax.differential_testing._shared_gpu_environment", return_value=nullcontext(os.environ.copy())):
            payloads = harness._run_distributed_emitters(
                (sys.executable, "-c", program), Path(directory), 5, "test", Path(directory) / "logs",
            )
            self.assertEqual(payloads, ("0\n", "1\n"))
            self.assertEqual((Path(directory) / "logs/test-rank-0.stdout.log").read_text(), "0\n")
            self.assertEqual((Path(directory) / "logs/test-rank-1.stderr.log").read_text(), "")

    @unittest.skipUnless(os.name == "posix", "process-group cancellation requires POSIX")
    def test_run_distributed_emitters_failure_terminates_other_rank(self) -> None:
        program = (
            "import sys,time; rank=int(sys.argv[sys.argv.index('--distributed-worker')+1]); "
            "print('rank diagnostic '+str(rank),file=sys.stderr,flush=True); "
            "time.sleep(.2 if rank==0 else 30); sys.exit(7 if rank==0 else 0)"
        )
        start = time.monotonic()
        with tempfile.TemporaryDirectory() as directory, \
                patch("ryft.jax.differential_testing._shared_gpu_environment", return_value=nullcontext(os.environ.copy())):
            with self.assertRaisesRegex(RuntimeError, "rank 0:.*", msg="failure must identify both ranks") as error:
                harness._run_distributed_emitters((sys.executable, "-c", program), Path(directory), 5, "test")
            self.assertIn("rank diagnostic 0", str(error.exception))
            self.assertIn("rank diagnostic 1", str(error.exception))
        self.assertLess(time.monotonic() - start, 3)

    @unittest.skipUnless(os.name == "posix", "process-group cancellation requires POSIX")
    def test_run_distributed_emitters_timeout_terminates_descendants(self) -> None:
        # If descendants survive, they write markers after the launcher has returned from cancellation.
        with tempfile.TemporaryDirectory() as directory, \
                patch("ryft.jax.differential_testing._shared_gpu_environment", return_value=nullcontext(os.environ.copy())):
            marker = Path(directory) / "survivor"
            child = f"import time,pathlib; time.sleep(2); pathlib.Path({str(marker)!r}).touch()"
            program = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
            with self.assertRaisesRegex(RuntimeError, "timed out after 1 seconds"):
                harness._run_distributed_emitters((sys.executable, "-c", program), Path(directory), 1, "test")
            time.sleep(1.5)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
