# **Ryft Core:** Work-in-Progress Core APIs

> [!WARNING]
> `ryft` is currently a work in progress and is evolving very actively. APIs and module or crate boundaries may change.

`ryft-core` is intended to host Ryft's core abstractions for tracing, automatic differentiation, and Just-In-Time (JIT)
compilation. Today, the most complete and usable part of this crate is the
[`Parameterized`](https://docs.rs/ryft-core/latest/ryft_core/parameters/trait.Parameterized.html) API.

## Coming Soon

`ryft-core` is expected to grow into the main home for higher-level tracing, automatic differentiation,
and Just-In-Time (JIT) compilation APIs. Those pieces are still being actively developed and are not yet ready
to take a dependency on.

## Roadmap / TODOs

- [ ] Move all `"xla"`-feature-gated code under the `ryft_core::xla` module. I'm wondering if it should be `ryft_xla`.
- [ ] Link from the `MeshAxisType::Manual` documentation to the `shard_map` operation once we have it.
- [ ] Add support for an operation like `jax.type_of`.
- [ ] Add support for an operation like `jax.lax.with_sharding_constraint`.

#### License

<sup>
Licensed under either <a href="../../LICENSE-APACHE">Apache License, Version 2.0</a>
or <a href="../../LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
</sub>
