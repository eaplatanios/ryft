"""Compare JAX-emitted reshape MLIR against the Rust-side reshape expectations."""

from ryft.jax.reshape_parity import main


if __name__ == "__main__":
    raise SystemExit(main())
