"""Regression tests for strict same-runtime cuDNN allocation comparison."""

import unittest
from collections import Counter

from compare_sanitizer_leaks import compare_logs, parse_log


def log(sizes=(9216, 16), *, caller="control", library="/control/libcudnn.so.9") -> str:
    """Builds a complete log with explicit cuDNN initialization allocation records."""
    records = []
    for size in sizes:
        records.append(f"""========= Leaked {size} bytes at 0x12345
=========     Saved host backtrace up to driver entry point at allocation time
=========         Host Frame: cuMemAlloc_v2 [0x123] in libcuda.so.1
=========         Host Frame: cudnnCreate [0x456] in {library}
=========         Host Frame: stream_executor::gpu::CudnnSupport::Init() [0x789] in plugin.so
=========         Host Frame: {caller} [0xabc] in executable
=========
""")
    return f"""========= COMPUTE-SANITIZER
running 1 test
StreamExecutor [0]: NVIDIA GB10 (Driver: 13.0; Runtime: 13.2; DNN: 9.25)
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s
{''.join(records)}========= LEAK SUMMARY: {sum(sizes)} bytes leaked in {len(sizes)} allocations
========= ERROR SUMMARY: {len(sizes)} errors
"""


class ComparisonTests(unittest.TestCase):
    """Rejects extra errors and allocation differences while allowing only installation and caller differences."""

    def test_parse_log(self) -> None:
        """Preserves sizes, multiplicities, symbols, and relative offsets through cuDNN initialization."""
        frames = (
            ("cuMemAlloc_v2", "0x123", "libcuda.so.1"),
            ("cudnnCreate", "0x456", "libcudnn.so.9"),
            ("stream_executor::gpu::CudnnSupport::Init()", "0x789", "plugin.so"),
        )
        self.assertEqual(
            parse_log(log((9216, 16, 16))),
            (
                Counter({(9216, frames): 1, (16, frames): 2}),
                ("NVIDIA GB10 (Driver: 13.0; Runtime: 13.2; DNN: 9.25)",),
            ),
        )

    def test_parse_log_rejects_non_cudnn_allocations(self) -> None:
        """Does not allow arbitrary leaks into the control baseline."""
        with self.assertRaisesRegex(ValueError, "not attributable"):
            parse_log(log().replace("CudnnSupport::Init()", "OtherAllocator::Init()"))

    def test_parse_log_rejects_invalid_access_and_extra_errors(self) -> None:
        """Rejects non-leak diagnostics and unexplained errors in the summary."""
        for text in (log() + "========= Invalid __global__ read of size 4\n",
                     log().replace("ERROR SUMMARY: 2", "ERROR SUMMARY: 3"),
                     log().replace("9232 bytes leaked", "9233 bytes leaked")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_log(text)

    def test_parse_log_rejects_incomplete_or_failed_runs(self) -> None:
        """Rejects empty, skipped, failed, truncated, and concatenated runs."""
        for text in ("", log().replace("1 passed", "0 passed"), log().replace("0 ignored", "1 ignored"),
                     log().replace("test result: ok", "test result: FAILED"),
                     log().split("========= ERROR SUMMARY:")[0], log() + log(),
                     log().replace("========= COMPUTE-SANITIZER\n", ""),
                     log().replace("StreamExecutor [0]:", "missing runtime:"),
                     log().replace("========= LEAK SUMMARY: 9232 bytes leaked in 2 allocations\n", "")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_log(text)

    def test_compare_logs(self) -> None:
        """Ignores ASLR, library installation directories, and callers after cuDNN initialization."""
        probe = log(caller="kernel", library="/probe/libcudnn.so.9").replace("at 0x12345", "at 0x98765")
        self.assertEqual(compare_logs(log(), probe), (2, 9232))
        self.assertEqual(compare_logs(log(()), log(())), (0, 0))

    def test_compare_logs_rejects_changed_allocations(self) -> None:
        """Preserves allocation multiplicities, sizes, library names, and relative offsets."""
        for probe in (log((9216, 16, 16)), log((9216, 32)), log().replace("0x456", "0x457"),
                      log(library="other.so")):
            with self.subTest(probe=probe), self.assertRaisesRegex(ValueError, "retained allocations differ"):
                compare_logs(log(), probe)

    def test_compare_logs_rejects_runtime_mismatch(self) -> None:
        """Does not compare evidence from different CUDA runtime descriptions."""
        with self.assertRaisesRegex(ValueError, "runtime device descriptions differ"):
            compare_logs(log(), log().replace("DNN: 9.25", "DNN: 9.26"))


if __name__ == "__main__":
    unittest.main()
