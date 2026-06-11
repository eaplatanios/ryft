"""Dump the StableHLO JAX emits for memory-kind transfers inside `jit`.

This is the JAX side of the parity check for ryft's `TransferToMemoryOperation` lowering: the
program below mirrors `test_transfer_to_memory_lowers_to_device_placement_annotations` in
`crates/ryft-xla/src/experimental/lowering.rs`, and the `annotate_device_placement` custom calls
it prints are asserted byte-for-byte in that test.

Two JAX-specific wrinkles make the dump non-obvious:

- JAX registers the `annotate_device_placement` lowering for the TPU and GPU platforms only; on
  CPU, `device_put` lowers as a pass-through and the transfers vanish from the module. The
  program is therefore traced and then cross-lowered for TPU, which requires no TPU hardware.
- Transfers adjacent to the function boundary are folded into `mhlo.memory_kind` argument/result
  attributes instead of custom calls, so the transfers are flanked by multiplications to keep
  them in the function interior.
"""

import jax
import jax.numpy as jnp
from jax._src.sharding_impls import TransferToMemoryKind


def f(x):
    y = x * 2.0
    y_host = jax.device_put(y, TransferToMemoryKind("pinned_host"))
    y_back = jax.device_put(y_host, TransferToMemoryKind("device"))
    return y_back * 3.0


def main() -> int:
    traced = jax.jit(f).trace(jnp.zeros((4,), jnp.float32))
    print(traced.lower(lowering_platforms=("tpu",)).as_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
