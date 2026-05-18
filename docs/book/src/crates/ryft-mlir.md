# `ryft-mlir`

High-level, ownership-aware Rust bindings for MLIR and the MLIR dialects used by the XLA tooling
stack — including StableHLO, `func`, `arith`, `gpu`, and others. The bindings model MLIR contexts,
operations, attributes, and types as Rust types with explicit lifetimes, and expose macro-driven
operation wrappers per dialect.

- **Crate on crates.io:** [`ryft-mlir`](https://crates.io/crates/ryft-mlir)
- **API reference:** [`docs.rs/ryft-mlir`](https://docs.rs/ryft-mlir)
- **Source:** [`crates/ryft-mlir/`](https://github.com/eaplatanios/ryft/tree/main/crates/ryft-mlir)
- **README:** [`crates/ryft-mlir/README.md`](https://github.com/eaplatanios/ryft/blob/main/crates/ryft-mlir/README.md)
