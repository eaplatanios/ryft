"""Regression tests for cuTile exporter failures without a compiler or GPU installation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import export_vector_add


def arguments(directory: Path) -> argparse.Namespace:
    """Builds exporter arguments with separate artifact, metadata, and diagnostic paths."""
    return argparse.Namespace(
        output_cubin=directory / "vector_add.cubin",
        output_metadata=directory / "vector_add.json",
        diagnostics=directory / "diagnostics.json",
        gpu_code="sm_100",
        compiler_timeout_seconds=1,
        process_timeout_seconds=2,
    )


class ExportFailureTests(unittest.TestCase):
    """Preserves compiler diagnostics and previous artifacts when exporting fails."""

    def test_export_fixture_missing_dependency(self) -> None:
        """Reports the required Python package instead of leaking an import failure."""
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"cuda": None}):
                with self.assertRaises(RuntimeError) as raised:
                    export_vector_add._export_fixture(arguments(Path(directory)))
            self.assertEqual(
                str(raised.exception),
                "cuTile Python is unavailable; install a supported `cuda-tile[tileiras]` environment",
            )
            self.assertIsInstance(raised.exception.__cause__, ImportError)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_orchestrate_worker_failure(self) -> None:
        """Retains worker output and status without publishing or leaving a temporary artifact."""
        with tempfile.TemporaryDirectory() as directory:
            export_arguments = arguments(Path(directory))
            export_arguments.output_cubin.write_bytes(b"previous cubin")
            export_arguments.output_metadata.write_text("previous metadata\n")
            completed = subprocess.CompletedProcess([], 7, "compiler started\n", "compiler rejected fixture\n")
            with patch.object(export_vector_add.subprocess, "run", return_value=completed) as run:
                with self.assertRaises(RuntimeError) as raised:
                    export_vector_add._orchestrate(export_arguments)
            self.assertEqual(str(raised.exception), f"cuTile export failed; diagnostics: {export_arguments.diagnostics}")
            self.assertEqual(
                json.loads(export_arguments.diagnostics.read_text()),
                {
                    "schema_version": 1,
                    "command": run.call_args.args[0],
                    "process_timeout_seconds": 2,
                    "status": "completed",
                    "return_code": 7,
                    "stdout": "compiler started\n",
                    "stderr": "compiler rejected fixture\n",
                },
            )
            self.assertEqual(export_arguments.output_cubin.read_bytes(), b"previous cubin")
            self.assertEqual(export_arguments.output_metadata.read_text(), "previous metadata\n")
            self.assertEqual(
                sorted(path.name for path in Path(directory).iterdir()),
                ["diagnostics.json", "vector_add.cubin", "vector_add.json"],
            )

    def test_orchestrate_timeout(self) -> None:
        """Serializes partial byte or text output, including truncated UTF-8, and cleans up on timeout."""
        for stdout, stderr, expected_stdout, expected_stderr in (
            (b"compiler started\n", b"compiler diagnostic\n\xe2", "compiler started\n", "compiler diagnostic\n\ufffd"),
            ("compiler started\n", "compiler diagnostic\n", "compiler started\n", "compiler diagnostic\n"),
            (None, None, "", ""),
        ):
            with self.subTest(stdout=stdout, stderr=stderr), tempfile.TemporaryDirectory() as directory:
                export_arguments = arguments(Path(directory))
                export_arguments.output_cubin.write_bytes(b"previous cubin")
                export_arguments.output_metadata.write_text("previous metadata\n")
                timeout = subprocess.TimeoutExpired("worker", 2, output=stdout, stderr=stderr)
                with patch.object(export_vector_add.subprocess, "run", side_effect=timeout) as run:
                    with self.assertRaises(RuntimeError) as raised:
                        export_vector_add._orchestrate(export_arguments)
                self.assertEqual(
                    str(raised.exception),
                    f"cuTile export exceeded the 2-second process timeout; diagnostics: {export_arguments.diagnostics}",
                )
                self.assertIs(raised.exception.__cause__, timeout)
                self.assertEqual(
                    json.loads(export_arguments.diagnostics.read_text()),
                    {
                        "schema_version": 1,
                        "command": run.call_args.args[0],
                        "process_timeout_seconds": 2,
                        "status": "timed_out",
                        "stdout": expected_stdout,
                        "stderr": expected_stderr,
                    },
                )
                self.assertEqual(run.call_args.kwargs["timeout"], 2)
                self.assertEqual(export_arguments.output_cubin.read_bytes(), b"previous cubin")
                self.assertEqual(export_arguments.output_metadata.read_text(), "previous metadata\n")
                self.assertEqual(
                    sorted(path.name for path in Path(directory).iterdir()),
                    ["diagnostics.json", "vector_add.cubin", "vector_add.json"],
                )


if __name__ == "__main__":
    unittest.main()
