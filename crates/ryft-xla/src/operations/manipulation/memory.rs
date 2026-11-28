use ryft_core::{ArrayType, TransferToMemoryOperation};
use ryft_mlir::ValueRef;

use crate::lowering::types::MemoryLoweringExtension;
use crate::{Error, LowerableOperation, Lowerer, ProgramLoweringOperation};

// TODO(eaplatanios): Review this module.

impl LowerableOperation for TransferToMemoryOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        self.destination()
            .lower_transfer(input_values, &mut lowerer.block(), lowerer.context(), lowerer.location())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{DataType, Memory, Placeholder, Sharding, TransferToMemory, Typed};

    use crate::lowering::tests::{ProgramBuilder, test_vector_type};
    use crate::operations::Operation;
    use crate::sessions::tests::domain_mesh;
    use crate::tests::{cpu_client, execution_client, values_from_bytes, values_to_bytes};
    use crate::{
        Array, ArrayConstant, ArrayTracer, CompiledFunction, Constant, ProgramLoweringExtension,
        ProgramLoweringOptions, Session, compile, trace_with_context,
    };

    use super::*;

    #[test]
    fn test_transfer_to_memory_operation_lower() {
        // A round trip to host memory and back, surrounded by computation, lowers to one `annotate_device_placement`
        // custom call per transfer, carrying the destination kind in the `_xla_buffer_placement` frontend attribute.
        // This includes the transfer back to device memory, which looks like an identity but which `HostOffloader`
        // needs to see. The program mirrors the JAX example in
        // `python/scripts/dump_transfer_to_memory_mlir_from_jax.py`, and the asserted custom calls are identical, byte
        // for byte, to the ones JAX emits for it.
        let program = trace_with_context(
            |_, x: ArrayTracer| {
                let y = x.clone() * x;
                let on_host = y.transfer_to_memory(Memory::Host { pinned: true })?;
                let back = on_host.transfer_to_memory(Memory::Device)?;
                Ok(back.clone() * back)
            },
            test_vector_type(4),
        )
        .unwrap();
        assert_eq!(
            program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                %1 = stablehlo.custom_call @annotate_device_placement(%0) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}} : (tensor<4xf32>) -> tensor<4xf32>
                %2 = stablehlo.custom_call @annotate_device_placement(%1) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "device"}} : (tensor<4xf32>) -> tensor<4xf32>
                %3 = stablehlo.multiply %2, %2 : tensor<4xf32>
                return %3 : tensor<4xf32>
              }
            }
        "#},
        );
    }

    #[test]
    fn test_transfer_to_memory_operation_lower_pullback() {
        // The pullback of a transfer moves the cotangent back to the input's source memory, which is the default
        // device space here, so it lowers to an `annotate_device_placement` custom call targeting `device`.
        let program = trace_with_context(
            |_, x: ArrayTracer| x.transfer_to_memory(Memory::Host { pinned: true }),
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        let pullback = program.transpose().unwrap();
        assert_eq!(
            pullback.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<f64> {mhlo.memory_kind = "pinned_host"}) -> tensor<f64> {
                %0 = stablehlo.custom_call @annotate_device_placement(%arg0) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "device"}} : (tensor<f64>) -> tensor<f64>
                return %0 : tensor<f64>
              }
            }
        "#},
        );
    }

    #[test]
    fn test_transfer_to_memory_operation_lower_round_trip() {
        let client = execution_client();
        let devices = client.addressable_devices().unwrap();
        let device = &devices[0];
        let device_kind = device.default_memory().unwrap().kind().unwrap().into_owned();
        let device_id = device.id().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let input_type = ArrayType::new_static(DataType::I64, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        for (memory, memory_kind) in
            [(Memory::Host { pinned: true }, "pinned_host"), (Memory::Host { pinned: false }, "unpinned_host")]
        {
            let available =
                device.addressable_memories().unwrap().iter().any(|memory| memory.kind().unwrap() == memory_kind);
            assert!(available, "execution device does not expose required memory kind `{memory_kind}`");
            let values = [i64::MIN, i64::MAX, 9_007_199_254_740_993, -9_007_199_254_740_993];
            let source = Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes(&values).as_slice(),
            )
            .unwrap();

            // A transfer whose value stays inside the program needs no boundary in host memory, so it round trips on
            // every platform.
            let round_trip: CompiledFunction<ArrayType, ArrayType> = compile(
                |input| input.transfer_to_memory(memory).unwrap().transfer_to_memory(Memory::Device).unwrap(),
                input_type.clone(),
                &engine,
                mesh.clone(),
            )
            .unwrap();
            let output = round_trip.executable().execute(&engine, source).unwrap();
            assert_eq!(output.r#type().as_ref(), &input_type);
            let buffer = output.device_shard(device_id).unwrap().buffer().unwrap();
            assert_eq!(buffer.memory().unwrap().kind().unwrap(), device_kind);
            assert_eq!(
                values_from_bytes::<i64>(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice()),
                values,
            );
        }
    }

    #[test]
    fn test_transfer_to_memory_operation_lower_host_boundaries() {
        let client = cpu_client(1);
        let devices = client.addressable_devices().unwrap();
        let device = &devices[0];
        let device_kind = device.default_memory().unwrap().kind().unwrap().into_owned();
        let device_id = device.id().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let input_type = ArrayType::new_static(DataType::I64, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let values = [i64::MIN, i64::MAX, 9_007_199_254_740_993, -9_007_199_254_740_993];
        let source =
            Array::from_host_buffer(&engine, input_type.clone(), mesh.clone(), None, values_to_bytes(&values)).unwrap();

        for (memory, memory_kind) in
            [(Memory::Host { pinned: true }, "pinned_host"), (Memory::Host { pinned: false }, "unpinned_host")]
        {
            let host_type = input_type.clone().with_memory(memory);
            // A compiled host output must occupy the requested host memory, not just carry host metadata on a device
            // buffer.
            let to_host: CompiledFunction<ArrayType, ArrayType> =
                compile(|input| input.transfer_to_memory(memory).unwrap(), input_type.clone(), &engine, mesh.clone())
                    .unwrap();
            let host = to_host.executable().execute(&engine, source.clone()).unwrap();
            assert_eq!(host.r#type().as_ref(), &host_type);
            let buffer = host.device_shard(device_id).unwrap().buffer().unwrap();
            assert_eq!(buffer.memory().unwrap().kind().unwrap(), memory_kind);
            assert_eq!(
                values_from_bytes::<i64>(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice()),
                values,
            );

            // Host placement survives an identity executable boundary, even without a transfer instruction.
            let identity: CompiledFunction<ArrayType, ArrayType> =
                compile(|input| input, host_type.clone(), &engine, mesh.clone()).unwrap();
            let host = identity.executable().execute(&engine, host).unwrap();
            assert_eq!(host.r#type().as_ref(), &host_type);
            let buffer = host.device_shard(device_id).unwrap().buffer().unwrap();
            assert_eq!(buffer.memory().unwrap().kind().unwrap(), memory_kind);
            assert_eq!(
                values_from_bytes::<i64>(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice()),
                values,
            );

            // The host input signature must permit a real host input to return to device memory.
            let to_device: CompiledFunction<ArrayType, ArrayType> =
                compile(|input| input.transfer_to_memory(Memory::Device).unwrap(), host_type, &engine, mesh.clone())
                    .unwrap();
            let output = to_device.executable().execute(&engine, host).unwrap();
            assert_eq!(output.r#type().as_ref(), &input_type);
            let buffer = output.device_shard(device_id).unwrap().buffer().unwrap();
            assert_eq!(buffer.memory().unwrap().kind().unwrap(), device_kind);
            assert_eq!(
                values_from_bytes::<i64>(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice()),
                values,
            );
        }
    }

    #[test]
    fn test_transfer_to_memory_signature_placement() {
        // A host identity has no transfer instruction, so only its signature can communicate physical placement.
        for (memory, kind) in
            [(Memory::Host { pinned: true }, "pinned_host"), (Memory::Host { pinned: false }, "unpinned_host")]
        {
            let input_type = ArrayType::new_static(DataType::I64, [2]);
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(input_type.clone().with_memory(memory));
            let program = builder
                .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap();
            let expected = indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xi64> {mhlo.memory_kind = "KIND"}) -> (tensor<2xi64> {mhlo.memory_kind = "KIND"}) {
                    return %arg0 : tensor<2xi64>
                  }
                }
            "#}.replace("KIND", kind);
            let program = program.into_unprojected::<Constant, Operation>().unwrap();
            assert_eq!(program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(), expected,);

            // An explicit transfer isolates the output check because the input stays in device memory.
            let to_host =
                trace_with_context(|_, input: ArrayTracer| input.transfer_to_memory(memory), input_type).unwrap();
            let expected_to_host = to_host.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();
            for platform in ["cuda", "rocm", "custom-plugin"] {
                let options = ProgramLoweringOptions::new().with_target_platform(Some(platform));
                assert_eq!(program.to_rendered_mlir("main", &options).unwrap(), expected);
                assert_eq!(to_host.to_rendered_mlir("main", &options).unwrap(), expected_to_host);
            }
        }
    }
}
