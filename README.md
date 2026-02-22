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

## **Example:** Low-Level StableHLO Matrix Multiplication

The following example uses the low-level MLIR and PJRT APIs provided by `ryft::mlir` and `ryft::pjrt` to build a toy
StableHLO matrix multiplication module programmatically, compile it, and execute it on the CPU plugin. Note that this
is quite low-level and verbose. `ryft::core` will make compiling and executing programs like this a lot more
ergonomic, similar to what JAX accomplishes in Python. Updates on that crate should be coming in the next few weeks
or months.

> [!NOTE]
> If you want to run on CUDA 13 instead, enable `ryft`'s `cuda-13` feature and replace `load_cpu_plugin()`
> with `load_cuda_13_plugin()` in the example code below.

```rust
use ryft::mlir::*;
use ryft::pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
use ryft::pjrt::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // First, let us construct the StableHLO module that represents this program.
    let context = Context::new();
    let location = context.unknown_location();
    let module = context.module(location);
    let f32_type = context.float32_type();
  
    // Types of the left-hand side, right-hand side, and result tensors in our matrix multiplication.
    let lhs_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(3)], None, location).unwrap();
    let rhs_type = context.tensor_type(f32_type, &[Size::Static(3), Size::Static(2)], None, location).unwrap();
    let result_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();

    // Body of the StableHLO module.
    module.body().append_operation({
        let mut block = context.block(&[(lhs_type, location), (rhs_type, location)]);
        let lhs = block.argument(0).unwrap();
        let rhs = block.argument(1).unwrap();
        let matmul = block.append_operation(dialects::stable_hlo::dot_general(
            lhs,
            rhs,
            context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]),
            None,
            None,
            result_type,
            location,
        ));
        block.append_operation(dialects::func::r#return(&[matmul.result(0).unwrap()], location));
        dialects::func::func(
            "main",
            dialects::func::FuncAttributes {
                arguments: vec![lhs_type.into(), rhs_type.into()],
                results: vec![result_type.into()],
                ..Default::default()
            },
            block.into(),
            location,
        )
    });
    assert!(module.verify());
    let program = Program::Mlir { bytecode: module.as_operation().bytecode() };

    // Now that we have the StableHLO program, let us use PJRT to compile it and execute it.
    let plugin = load_cpu_plugin()?;
    let client = plugin.client(ClientOptions::default())?;
    let executable = client.compile(
        &program,
        &CompilationOptions {
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 1,
                ..Default::default()
            }),
            matrix_unit_operand_precision: Precision::Default as i32,
            ..Default::default()
        },
    )?;
    let device = executable.addressable_devices()?[0].clone();

    // The left-hand side tensor is set to [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].
    // The right-hand side tensor is set to [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]].
    let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let lhs_bytes = lhs.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
    let rhs_bytes = rhs.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
    let lhs_buffer = client.buffer(lhs_bytes.as_slice(), BufferType::F32, &[2, 3], None, device.clone(), None)?;
    let rhs_buffer = client.buffer(rhs_bytes.as_slice(), BufferType::F32, &[3, 2], None, device, None)?;
    let inputs = [
        ExecutionInput { buffer: lhs_buffer, donatable: false },
        ExecutionInput { buffer: rhs_buffer, donatable: false },
    ];
    let inputs = vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }];

    // The expected output of this matrix multiplication is [[58.0, 64.0], [139.0, 154.0]].
    let mut outputs = executable.execute(inputs, 0, None, None, None, None)?.remove(0);
    outputs.done.r#await()?;
    let output = outputs
        .outputs
        .remove(0)
        .copy_to_host(None)?
        .r#await()?
        .chunks_exact(4)
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(chunk);
            f32::from_ne_bytes(bytes)
        })
        .collect::<Vec<_>>();
    assert_eq!(output, vec![58.0, 64.0, 139.0, 154.0]);

    Ok(())
}
```

> [!NOTE]
> Note that this is quite low-level and verbose. `ryft::core` will make compiling and executing programs like this a 
> lot more ergonomic, similar to what JAX accomplishes in Python. Updates on that crate should be coming in the next
> few weeks or months.

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
