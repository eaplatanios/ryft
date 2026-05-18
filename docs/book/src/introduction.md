# The Ryft Book

Ryft is a Rust library for building machine learning systems, inspired by
[JAX](https://docs.jax.dev/en/latest/). It brings type-safe support for tracing, automatic
differentiation, and just-in-time compilation to Rust, on top of the same StableHLO / PJRT / MLIR
compiler stack that powers JAX and XLA.

This book is the long-form user guide. It covers installation, the feature-flag system,
accelerator setup, the core concepts behind ryft, and a reference for each crate in the workspace.

Looking for something else?

- **The landing page and quickstart** live at [ryft.dev](https://ryft.dev).
- **API reference** is on [docs.rs/ryft](https://docs.rs/ryft) — auto-generated from rustdoc.
- **Source** is on [GitHub](https://github.com/eaplatanios/ryft).

> **Note:** Ryft is currently a work in progress and is evolving very actively. APIs and crate
> boundaries may change without notice. The most stable surface today is the
> [`Parameterized`](https://docs.rs/ryft-core/latest/ryft_core/parameters/trait.Parameterized.html)
> API in [`ryft-core`](https://docs.rs/ryft-core).
