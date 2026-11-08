"""Dump the StableHLO JAX emits for memory-kind transfers inside `jit`.

This is the JAX side of the parity check for ryft's `TransferToMemoryOperation` lowering: the
program below mirrors `test_transfer_to_memory_lowers_to_device_placement_annotations` in
`crates/ryft-xla/src/operations/manipulation/memory.rs`, and the `annotate_device_placement`
custom calls it prints are asserted byte-for-byte in that test.

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


def f(x):
    y = x * x
    y_host = jax.device_put(y, jax.memory.Space.Host)
    y_back = jax.device_put(y_host, jax.memory.Space.Device)
    return y_back * y_back


def main() -> int:
    traced = jax.jit(f).trace(jnp.zeros((4,), jnp.float32))
    print(traced.lower(lowering_platforms=("tpu",)).as_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
