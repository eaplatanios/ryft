use ryft_core::{ArrayType, TransferToMemoryOperation};
use ryft_mlir::ValueRef;

use crate::lowering::types::lower_transfer_to_memory;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for TransferToMemoryOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        lower_transfer_to_memory(
            self.destination(),
            input_values,
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array as CoreArray, ArrayOperation, DataType, DomainTracer, EagerContext, Memory, Placeholder,
        ReverseModeDifferentiate, Sharding, Trace, TransferToMemory, Typed,
    };

    use crate::lowering::lower_mlir_module_for_program;
    use crate::lowering::tests::{ProgramBuilder, test_vector_type, to_mlir_module_for_plain_program};
    use crate::operations::Operation;
    use crate::sessions::tests::domain_mesh;
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{
        Array, ArrayConstant, CompiledFunction, Constant, Error, LoweringError, Session, compile,
        to_mlir_module_for_program,
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
        let (_, program) = EagerContext::<CoreArray, ArrayOperation<CoreArray>>::trace(
            |x: DomainTracer<EagerContext<CoreArray, ArrayOperation<CoreArray>>>| {
                let y = x.clone() * x;
                let on_host = y.transfer_to_memory(Memory::Host { pinned: true })?;
                let back = on_host.transfer_to_memory(Memory::Device)?;
                Ok(back.clone() * back)
            },
            test_vector_type(4),
        )
        .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
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
        // The pullback of a transfer moves the cotangent back to the operand's source memory, which is the default
        // device space here, so it lowers to an `annotate_device_placement` custom call targeting `device`.
        let (_, pullback): (CoreArray, _) = EagerContext::<CoreArray, ArrayOperation<CoreArray>>::new()
            .vjp(|x, ()| x.transfer_to_memory(Memory::Host { pinned: true }), CoreArray::scalar(2.0).unwrap(), ())
            .unwrap();
        let (pullback, _residuals) = pullback.into_transposed_parts().unwrap();
        let pullback = pullback.into_program();
        assert_eq!(
            to_mlir_module_for_plain_program(&pullback, "main").unwrap(),
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
            let host_type = input_type.clone().with_memory(memory);
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
            let output = round_trip.executable().execute(&engine, source.clone()).unwrap();
            assert_eq!(output.r#type().as_ref(), &input_type);
            let buffer = output.device_shard(device_id).unwrap().buffer().unwrap();
            assert_eq!(buffer.memory().unwrap().kind().unwrap(), device_kind);
            assert_eq!(
                values_from_bytes::<i64>(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice()),
                values,
            );

            // XLA's CUDA compiler places no program argument or result in unpinned host memory, although its plugin
            // can hold buffers there, so a program with such a boundary is rejected when it is lowered.
            if client.platform_name().unwrap().eq_ignore_ascii_case("cuda")
                && memory == (Memory::Host { pinned: false })
            {
                let to_host: Result<CompiledFunction<ArrayType, ArrayType>, Error> = compile(
                    |input| input.transfer_to_memory(memory).unwrap(),
                    input_type.clone(),
                    &engine,
                    mesh.clone(),
                );
                assert!(matches!(
                    to_host,
                    Err(Error::Lowering(LoweringError::UnsupportedBoundaryMemory { memory: actual, platform }))
                        if actual == memory && platform == "cuda",
                ));
                let identity: Result<CompiledFunction<ArrayType, ArrayType>, Error> =
                    compile(|input| input, host_type, &engine, mesh.clone());
                assert!(matches!(
                    identity,
                    Err(Error::Lowering(LoweringError::UnsupportedBoundaryMemory { memory: actual, platform }))
                        if actual == memory && platform == "cuda",
                ));
                continue;
            }

            // A compiled host result must occupy the requested host memory, not just carry host metadata on a device
            // buffer.
            let to_host: CompiledFunction<ArrayType, ArrayType> =
                compile(|input| input.transfer_to_memory(memory).unwrap(), input_type.clone(), &engine, mesh.clone())
                    .unwrap();
            let host = to_host.executable().execute(&engine, source).unwrap();
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
        // Exercise both the diagnostic homogeneous path and the compiled array IR entry path.
        for (memory, kind) in
            [(Memory::Host { pinned: true }, "pinned_host"), (Memory::Host { pinned: false }, "unpinned_host")]
        {
            let r#type = ArrayType::new_static(DataType::I64, [2]).with_memory(memory);
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(r#type.clone());
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
            assert_eq!(to_mlir_module_for_plain_program(&program, "main").unwrap(), expected);
            let program = program.into_unprojected::<Constant, Operation>().unwrap();
            let types = vec![r#type];
            assert_eq!(
                to_mlir_module_for_program(&program, &[], &types, &types, "main", None, None).unwrap(),
                expected,
            );

            // A program compiled for CUDA can place its boundary in pinned host memory only.
            let lowered =
                lower_mlir_module_for_program(&program, &[], &types, &types, "main", None, None, Some("cuda"));
            if memory == (Memory::Host { pinned: true }) {
                assert_eq!(lowered.unwrap().into_stable_hlo(), expected);
            } else {
                assert!(matches!(
                    lowered,
                    Err(Error::Lowering(LoweringError::UnsupportedBoundaryMemory { memory: actual, platform }))
                        if actual == memory && platform == "cuda",
                ));
            }
        }
    }
}
