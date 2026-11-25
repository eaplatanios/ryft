# **Ryft Optimizers:** Work-in-Progress Composable Optimizers

> [!WARNING]
> `ryft` is currently a work in progress and is evolving very actively. APIs and module or crate boundaries may change.

`ryft-optimizers` hosts gradient-based optimizers for [`ryft-core`](../ryft-core) parameter structures. Its design
follows [Optax](https://optax.readthedocs.io/). An optimizer is a pure transformation from gradients and a state to
updates and a new state, and larger optimizers (e.g., AdamW) are chains of smaller ones.

Among the Ryft crates, this crate depends only on `ryft-core` and `ryft-macros`, so its optimizers run eagerly on host
arrays and inside compiled training steps on any backend.

## Example

The following example minimizes the sum of `(x - 3)²` over the elements `x` of a vector with `AdamW`, without weight
decay. Updates are added to the parameters with `apply_updates`, and the optimizer state is threaded explicitly from
one step to the next, so the same loop body can also run inside a compiled training step.

```rust
use ryft_core::{Array, FillLike, Mul, Reduce, ReductionKind, Sub, differentiate_at};
use ryft_optimizers::{AdamW, Optimizer, UpdateArguments, apply_updates};

let optimizer = AdamW::new(0.1).with_weight_decay(0.0)?;
let mut parameters = Array::vector(vec![0.0f32, 1.0])?;
let mut state = optimizer.initialize(&parameters)?;
for _ in 0..200 {
    let (_, gradients) = differentiate_at(parameters.clone()).value_and_gradient(|parameters| {
        let residuals = Sub::sub(&parameters, &parameters.fill_like(3.0f32)?)?;
        Mul::mul(&residuals, &residuals)?.reduce(&[0], ReductionKind::Sum)
    })?;
    let (updates, next_state) = optimizer.update(gradients, state, &parameters, &UpdateArguments::new())?;
    parameters = apply_updates(parameters, updates)?;
    state = next_state;
}
```

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
