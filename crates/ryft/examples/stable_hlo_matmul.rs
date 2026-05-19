//! Low-level StableHLO matrix multiplication example.
//!
//! Constructs a small StableHLO module that multiplies a `2x3` matrix by a `3x2` matrix using
//! `dot_general`, compiles it through PJRT, runs it on the CPU plugin, and verifies the output. This
//! intentionally uses the low-level [`ryft::mlir`] and [`ryft::pjrt`] APIs directly; once `ryft-core`
//! lands its higher-level frontend, the same computation will be expressible with much less boilerplate.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p ryft --example stable_hlo_matmul
//! ```
//!
//! To target CUDA 13 instead of the CPU plugin, enable the `cuda-13` feature on `ryft` and replace
//! [`load_cpu_plugin`] below with [`load_cuda_13_plugin`].

use std::sync::Arc;

use ryft::mlir::*;
use ryft::pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
use ryft::pjrt::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // First, let us construct the StableHLO module that represents this program.
    let context = Context::new();
    let location = context.unknown_location();
    let module = context.module(location)?;
    let f32_type = context.float32_type();

    // Types of the left-hand side, right-hand side, and result tensors in our matrix multiplication.
    let lhs_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(3)], None, location)?;
    let rhs_type = context.tensor_type(f32_type, &[Size::Static(3), Size::Static(2)], None, location)?;
    let result_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location)?;

    // Body of the StableHLO module.
    module.body()?.append_operation({
        let mut block = context.block(&[(lhs_type, location), (rhs_type, location)]);
        let lhs = block.argument(0)?;
        let rhs = block.argument(1)?;
        let matmul = block.append_operation(dialects::stable_hlo::dot_general(
            lhs,
            rhs,
            context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0])?,
            None,
            None,
            result_type,
            location,
        )?)?;
        block.append_operation(dialects::func::r#return(&[matmul.result(0)?], location)?)?;
        dialects::func::func(
            "main",
            dialects::func::FuncAttributes {
                arguments: vec![lhs_type.into(), rhs_type.into()],
                results: vec![result_type.into()],
                ..Default::default()
            },
            block.try_into()?,
            location,
        )?
    })?;
    assert!(module.verify()?);
    let program = Program::Mlir { bytecode: module.as_operation()?.bytecode() };

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
        ExecutionInput { buffer: Arc::new(lhs_buffer), donatable: false },
        ExecutionInput { buffer: Arc::new(rhs_buffer), donatable: false },
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
