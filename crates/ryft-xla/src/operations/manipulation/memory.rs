use ryft_core::macros::check_count;
use ryft_core::{ArrayType, Memory, TransferToMemoryOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::CustomCallApiVersion;
use ryft_mlir::{
    Attribute, Block, Context as MlirContext, Location, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

/// Returns the XLA memory kind string for `memory`, used by `mhlo.memory_kind` signature attributes and the
/// `_xla_buffer_placement` frontend attribute on `annotate_device_placement` custom calls. This mapping is owned by the
/// lowering on purpose, because the core [`Memory`] type exposes no backend vocabulary and its `Display` rendering is
/// only meant for diagnostics. This mirrors how [`Sharding`](ryft_core::Sharding) converts to MLIR through conversions
/// that the backend owns.
pub(crate) fn memory_placement_kind(memory: Memory) -> &'static str {
    match memory {
        Memory::Device => "device",
        Memory::Host { pinned: true } => "pinned_host",
        Memory::Host { pinned: false } => "unpinned_host",
    }
}

/// Applies the non-default memory placement declared by `output_type` to `value`.
pub(crate) fn annotate_output_memory<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    value: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    if output_type.memory() == Memory::Device {
        return Ok(value);
    }
    Ok(lower_transfer_to_memory(output_type.memory(), &[value], block, context, location)?[0])
}

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

/// Lowers one staged [`TransferToMemoryOperation`] to the `stablehlo.custom_call @annotate_device_placement` annotation
/// that the XLA `ConvertMemoryPlacementToInternalAnnotations` and
/// [`HostOffloader`](https://openxla.org/xla/tools_and_passes/host_offloading) passes legalize into copies annotated
/// with memory spaces. The annotation uses API version 1, `has_side_effect = true`, an empty `backend_config`, and
/// the destination kind in `_xla_buffer_placement` inside `mhlo.frontend_attributes`.
///
/// These constraints are retained on CPU as well, because reference arrays only retag their types, but native arrays
/// must honor the requested physical placement on every platform. The current XLA offloading pass treats pinned and
/// unpinned host annotations alike internally, whereas function boundary attributes and PJRT memory validation retain
/// the distinction at runtime boundaries. A plugin must expose the requested memory kind for native allocation.
///
/// Placement does not affect the MLIR tensor type, so the result type is the operand's type unchanged. Identity
/// transfers (i.e., transfers whose destination equals the operand's current space) still lower to the annotation,
/// because placement round trips are meaningful to `HostOffloader` and must not be optimized away here.
fn lower_transfer_to_memory<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    destination: Memory,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    check_count!("input", input_values, 1, ProgramError);
    let empty_backend_configuration = context.string_attribute("");
    let mut operation = stable_hlo::custom_call(
        input_values,
        "annotate_device_placement",
        true,
        Some(empty_backend_configuration.as_ref()),
        CustomCallApiVersion::Original,
        &[],
        None,
        &[],
        None,
        &[input_values[0].r#type()?],
        location,
    )?;
    operation.set_discardable_attribute(
        "mhlo.frontend_attributes",
        context.dictionary_attribute(&[context.named_attribute(
            context.identifier("_xla_buffer_placement"),
            context.string_attribute(memory_placement_kind(destination)),
        )]),
    );
    let operation = block.append_operation(operation)?;
    Ok(vec![operation.result(0).unwrap().as_ref()])
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array as CoreArray, ArrayOperation, DataType, EagerContext, Placeholder, ReverseModeDifferentiate, Sharding,
        Trace, TransferToMemory, Typed, ZeroOperation,
    };

    use crate::domains::tests::domain_mesh;
    use crate::lowering::tests::{ProgramBuilder, test_vector_type};
    use crate::lowering::to_mlir_module_for_plain_program;
    use crate::operations::Operation;
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::values::ArrayConstant;
    use crate::{Array, ArrayError, CompiledFunction, Constant, Error, Session, compile, to_mlir_module_for_program};

    use super::*;

    #[test]
    fn test_memory_placement_kind() {
        assert_eq!(memory_placement_kind(Memory::Device), "device");
        assert_eq!(memory_placement_kind(Memory::Host { pinned: true }), "pinned_host");
        assert_eq!(memory_placement_kind(Memory::Host { pinned: false }), "unpinned_host");
    }

    #[test]
    fn test_annotate_output_memory() {
        // Device outputs need no annotation, while host outputs gain one `annotate_device_placement` custom call in
        // addition to their `mhlo.memory_kind` signature attribute.
        let zero_program = |memory| {
            let mut builder = ProgramBuilder::new();
            let output_type = test_vector_type(4).with_memory(memory);
            let output =
                builder.add_instruction(ZeroOperation::new(output_type), Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], Vec::new(), vec![Placeholder])
                .unwrap()
        };
        assert_eq!(
            to_mlir_module_for_plain_program(&zero_program(Memory::Device), "main").unwrap(),
            indoc! {r#"
            module {
              func.func @main() -> tensor<4xf32> {
                %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
                return %0 : tensor<4xf32>
              }
            }
        "#},
        );
        for (memory, kind) in
            [(Memory::Host { pinned: true }, "pinned_host"), (Memory::Host { pinned: false }, "unpinned_host")]
        {
            let expected = indoc! {r#"
                module {
                  func.func @main() -> (tensor<4xf32> {mhlo.memory_kind = "KIND"}) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
                    %1 = stablehlo.custom_call @annotate_device_placement(%0) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "KIND"}} : (tensor<4xf32>) -> tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#}
            .replace("KIND", kind);
            assert_eq!(to_mlir_module_for_plain_program(&zero_program(memory), "main").unwrap(), expected);
        }
    }

    #[test]
    fn test_transfer_to_memory_operation_lower() {
        // A round trip to host memory and back, surrounded by computation, lowers to one `annotate_device_placement`
        // custom call per transfer, carrying the destination kind in the `_xla_buffer_placement` frontend attribute.
        // This includes the transfer back to device memory, which looks like an identity but which `HostOffloader`
        // needs to see. The program mirrors the JAX example in
        // `python/scripts/dump_transfer_to_memory_mlir_from_jax.py`, and the asserted custom calls are identical, byte
        // for byte, to the ones JAX emits for it.
        let (_, program) = EagerContext::<CoreArray, ArrayOperation<CoreArray>>::trace(
            |x: ryft_core::tracing::DomainTracer<EagerContext<CoreArray, ArrayOperation<CoreArray>>>| {
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
        let mesh = domain_mesh(&client, "x", 1);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new_static(DataType::I64, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        for (memory, memory_kind) in
            [(Memory::Host { pinned: true }, "pinned_host"), (Memory::Host { pinned: false }, "unpinned_host")]
        {
            let available =
                device.addressable_memories().unwrap().iter().any(|memory| memory.kind().unwrap() == memory_kind);
            let host_type = input_type.clone().with_memory(memory);
            let values = [i64::MIN, i64::MAX, 9_007_199_254_740_993, -9_007_199_254_740_993];
            // CUDA exposes pinned host buffers, but not an unpinned host allocation tier. Reject an unavailable
            // placement explicitly rather than silently allocating the buffer in device memory.
            if client.platform_name().unwrap().eq_ignore_ascii_case("cuda")
                && memory == (Memory::Host { pinned: false })
            {
                assert!(!available);
                assert!(matches!(
                    Array::from_host_buffer(&engine, host_type, mesh.clone(), values_to_bytes(&values).as_slice()),
                    Err(Error::Array(error)) if matches!(error.as_ref(),
                        ArrayError::UnsupportedMemory { device_id: actual_device, memory: actual_memory }
                            if *actual_device == device_id && *actual_memory == memory,
                    ),
                ));
                continue;
            }
            assert!(available, "execution device does not expose required memory kind `{memory_kind}`");
            let source =
                Array::from_host_buffer(&engine, input_type.clone(), mesh.clone(), values_to_bytes(&values).as_slice())
                    .unwrap();

            // A compiled host result must occupy the requested host memory, not just carry host metadata on a device
            // buffer.
            let to_host: CompiledFunction<'_, ArrayType, ArrayType> =
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
            let identity: CompiledFunction<'_, ArrayType, ArrayType> =
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
            let to_device: CompiledFunction<'_, ArrayType, ArrayType> =
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

            let round_trip: CompiledFunction<'_, ArrayType, ArrayType> = compile(
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
        }
    }
}
