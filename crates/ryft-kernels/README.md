# **Ryft Kernels:** Work-in-Progress Portable Kernel APIs

> [!WARNING]
> `ryft` is currently a work in progress and is evolving very actively. APIs and module or crate boundaries may change.

`ryft-kernels` hosts Ryft's backend-independent kernel subsystem: the kernel operation family, kernel definitions and
verification, host interpretation for debugging, serialization, transforms, and the `KernelCompiler` interface that
native compiler adapters implement. It also re-exports the `#[kernel]` authoring macro, which `ryft-macros` defines.
Kernel bodies are ordinary [`ryft-core`](../ryft-core) programs built from canonical array, dimension, reference, and
control-flow operations together with the kernel operations of this crate (e.g., tile loads and masked memory
accesses).

This crate has no native compiler, MLIR, PJRT, CUDA, ROCm, Mosaic, Triton, or XLA dependency. Native compilation and
execution belong to the adapter crates and the XLA integration. The `ryft` crate re-exports this crate as
`ryft::kernels`.

## Example

The `#[kernel]` macro turns an annotated function into a callable that interprets host arrays and a `definition`
function that returns the inspectable `KernelDefinition`. The following kernel adds two vectors of 5 elements in tiles
of 2 elements, so its grid has 3 programs and the last tile extends past the end of both vectors. Its loads fill the
missing element with the padding value, and the masked output publishes only the element that lies inside the output.
Crates that use the `ryft` facade write `#[ryft::kernel(...)]` without the `crate` argument.

```rust
use ryft_core::{Array, ArrayType, DataType, Dimension};
use ryft_kernels::kernel;

#[kernel(crate = "ryft_kernels", requires = left.shape()[0] == right.shape()[0])]
fn tiled_add(
    #[input(data_type = F32, rank = 1)] left: &Array,
    #[input(data_type = F32, rank = 1)] right: &Array,
    #[output(data_type = F32, shape = [left.shape()[0]], tile = [2], boundary = masked)] output: &mut Array,
) {
    let [block] = output.tile_index();
    let left_tiles = left.tiles([2]).pad(0.0);
    let right_tiles = right.tiles([2]).pad(0.0);
    output.store(left_tiles.load([block]) + right_tiles.load([block]));
}

let left = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
let right = Array::vector(vec![10.0f32, 20.0, 30.0, 40.0, 50.0]).unwrap();
assert_eq!(tiled_add(&left, &right), Ok(Array::vector(vec![11.0f32, 22.0, 33.0, 44.0, 55.0]).unwrap()));

let vector = ArrayType::new_static(DataType::F32, [5]);
let definition = tiled_add::definition(&vector, &vector).unwrap();
assert_eq!(definition.operation().grid().dimensions()[0].extent(), &Dimension::Static(3));
```

A `grid(...)` argument declares the grid explicitly, with sequential axes and one block mapping per parameter, which
expresses sequentially carried reductions such as a blocked matrix product with masked edge blocks. The crate
documentation contains that example and a description of the supported subset of Rust.

## Execution

Host interpretation (i.e., the generated callable for host `Array`s and `KernelDefinition::interpret`) first verifies
that the kernel initializes every result, accesses only valid windows, and has no conflicting writes, and it then
replays the body once per grid point. Verification evaluates a parameter mapping at every grid point unless
`BlockMapping::tiling_axes` recognizes it and that recognition names every grid axis at most once, and its
`maximum_programs` argument bounds only that enumeration. A recognized mapping that repeats a grid axis (e.g., the
diagonal element starts `(i, i)`) is still enumerated. `KernelDefinition::interpret_with_limits` and
`KernelDefinition::interpret_with_trace` also cap grid points and replay steps, and
`KernelDefinition::interpret_interleavings` explores the instruction orders of small kernels within explicit bounds.

Values staged by `ryft-xla` do not implement `KernelCall`, so a kernel enters an XLA program through its definition.
`ryft_xla::kernels::stage_kernel` binds the call together with its body into the enclosing program, and the
compilation options of the XLA session select the compiler adapter. Its `stage_kernel_with_jvp`,
`stage_kernel_with_vjp`, `stage_kernel_with_rules`, and `stage_kernel_with_fallback` variants attach explicit
derivative rules or a pure fallback program, because kernel calls have no implicit differentiation rule.

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
