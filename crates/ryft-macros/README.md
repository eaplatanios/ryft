# **Ryft Macros:** Procedural Macros for the Ryft Library

This crate currently provides three derive macros that are aimed at improving ergonomics for the `ryft` library:

- **`#[derive(Operation)]`:** Generates the base operation, interpretation, and partial-evaluation dispatchers
  for an operation enum. Optional batching, differentiation, and transposition dispatchers can be generated with
  `#[ryft(dispatch(...))]`.

- **`#[derive(Parameter)]`:** Implements the marker trait `Parameter` for the annotated type. This is mainly
  a convenience macro for leaf parameter types. It can be applied to Rust containers supported by derive macros
  (e.g., structs and enums).

- **`#[derive(Parameterized)]`:** Generates `Parameterized<P>` implementations for user-defined structs and enums:
    - Exactly one generic type parameter must be bounded by `Parameter` (as `Parameter` or `ryft::Parameter`).
    - The parameter type cannot have additional bounds beyond `Parameter`.
    - Unions are not supported.
    - Parameter fields must be owned: references and pointers to the parameter type are rejected.
    - The only supported container attribute is `#[ryft(crate = "...")]`, used to override the `ryft` crate path.
    - `#[ryft(...)]` is only valid at the container level and not on fields or variants, for example.

  The derive macro composes with `Parameterized` implementations from `ryft-core`, including:

    - `P: Parameter` (leaf parameters),
    - `PhantomData<P>`,
    - tuples where all elements are `Parameterized` (arity 1 through 12),
    - arrays `[V; N]` where `V: Parameterized<P>`, and
    - `Vec<V>` where `V: Parameterized<P>`.

  For tuple handling, mixed tuples (e.g., `(P, usize)`) are supported when nested inside a derived struct/enum field.
  However, standalone mixed tuples inside generic containers (e.g., `Vec<(P, usize)>`) are currently not supported.

## Examples

### `Operation`

`Operation` can be derived for an enum whose variants each wrap one concrete operation payload. The generated enum
delegates the operation interface to the active payload and provides conversions from each payload type:

```rust
use ryft::*;

#[derive(Clone, Debug, Operation)]
enum BackendOperation<V: Value<Type = DataType>> {
    Zero(ZeroOperation<DataType>),
    Constant(ConstantOperation<V>),
}
```

Optional transform dispatchers can be generated with
`#[ryft(dispatch(batching, differentiation, transposition))]`, as described above.

### `Parameter` and `Parameterized`

```rust
use ryft::*;

#[derive(Parameter, Clone, Copy)]
struct Weight(f32);

#[derive(Parameterized, Clone)]
struct Layer<P: Parameter> {
    weights: Vec<P>,
    bias: P,
    metadata: (usize, usize),
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
