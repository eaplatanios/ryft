use ryft::mlir::*;
use ryft::pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
use ryft::pjrt::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let location = context.unknown_location();
    let module = context.module(location);
    let f32_type = context.float32_type();

    let lhs_type = context
        .tensor_type(f32_type, &[Size::Static(2), Size::Static(3)], None, location)
        .expect("invalid lhs tensor type");
    let rhs_type = context
        .tensor_type(f32_type, &[Size::Static(3), Size::Static(2)], None, location)
        .expect("invalid rhs tensor type");
    let result_type = context
        .tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location)
        .expect("invalid result tensor type");

    module.body().append_operation({
        let mut block = context.block(&[(lhs_type, location), (rhs_type, location)]);
        let lhs = block.argument(0).expect("missing lhs argument");
        let rhs = block.argument(1).expect("missing rhs argument");
        let matmul = block.append_operation(dialects::stable_hlo::dot_general(
            lhs,
            rhs,
            context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]),
            None,
            None,
            result_type,
            location,
        ));
        block.append_operation(dialects::func::r#return(
            &[matmul.result(0).expect("missing matmul result")],
            location,
        ));
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

    let mut outputs = executable
        .execute(
            vec![ExecutionDeviceInputs {
                inputs: &inputs,
                ..Default::default()
            }],
            0,
            None,
            None,
            None,
            None,
        )?
        .remove(0);
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
