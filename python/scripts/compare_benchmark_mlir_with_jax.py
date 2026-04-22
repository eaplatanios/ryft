"""Compare Ryft benchmark MLIR against JAX-emitted Shardy MLIR."""

from ryft.jax.benchmark_parity import main

if __name__ == "__main__":
    raise SystemExit(main())
