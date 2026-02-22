# **Ryft:** A Rust Framework for Tracing, Automatic Differentiation, and Just-In-Time Compilation

> [!WARNING]
> `ryft` is currently a work in progress and is evolving very actively. APIs and crate boundaries may change.

`ryft` is a Rust library for building machine learning systems that is inspired by
[JAX](https://docs.jax.dev/en/latest/index.html). It aims to bring type-safe support for tracing, automatic
differentiation, and just-in-time compilation for leveraging hardware accelerators to Rust. The top-level `ryft`
crate is an umbrella crate that re-exports functionality from a few different crates through a single entry point:

- **`ryft-core`:** Intended home for core tracing, automatic differentiation, JIT, and program abstractions. This crate
  is still in an early stage and should not be dependent upon. It is expected to start shaping up in the coming months.
- **`ryft-macros`:** Procedural macros used by `ryft` and `ryft-core` (e.g., parameter-related derivation macros).
- **`ryft-mlir`:** High-level, ownership-aware Rust bindings for MLIR and MLIR dialects used by XLA tooling.
- **`ryft-pjrt`:** High-level, ownership-aware Rust bindings for PJRT plugins, clients, buffers, and program execution.
- **`ryft-xla-sys`:** Low-level `-sys` bindings for XLA/MLIR/PJRT APIs, plus native artifact/toolchain wiring.

## Feature Flags

The `ryft` crate enables the `xla` feature by default which brings in the `ryft-mlir`, `ryft-pjrt`, and `ryft-xla-sys`
dependencies. Accelerator-specific features (e.g., `cuda-12`, `cuda-13`, `rocm-7`, `tpu`, `neuron`, and `metal`) are
forwarded through the crate stack (`ryft` -> `ryft-core` -> `ryft-pjrt` -> `ryft-xla-sys`). For feature semantics,
platform/runtime requirements, and artifact-loading behavior, refer to:

- **[`crates/ryft-xla-sys/README.md`](crates/ryft-xla-sys/README.md):** Reference for XLA dependencies
  and for instructions on how to configure for obtaining pre-built binaries for supported platforms.
- **[`crates/ryft-pjrt/README.md`](crates/ryft-pjrt/README.md):** Reference for our PJRT bindings.
- **[`crates/ryft-mlir/README.md`](crates/ryft-mlir/README.md):** Reference for our MLIR bindings.

## Why "Ryft"?

The name for this framework started from the idea of **Rust + Lift**: "lifting" computations through tracing so they can
be transformed for automatic differentiation and just-in-time compilation. That naturally suggested the name **`rift`**.
Since that name was already taken, I chose **`ryft`** as a close alternative with the same original inspiration.
The short, catchy spelling also matches a core goal of the project: fast & efficient execution.

#### License

<sup>
Licensed under either <a href="LICENSE-APACHE">Apache License, Version 2.0</a> or <a href="LICENSE-MIT">MIT license</a>
at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
</sub>

#### Acknowledgements

<sup>
Thanks to [RunsOn](https://runs-on.com/) for providing our GitHub Actions runners infrastructure.
</sup>
