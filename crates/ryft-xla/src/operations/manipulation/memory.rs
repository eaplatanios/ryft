use ryft_core::macros::check_count;
use ryft_core::{ArrayType, Memory, TransferToMemoryOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::CustomCallApiVersion;
use ryft_mlir::{
    Attribute, Block, Context as MlirContext, Location, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Returns the XLA memory-kind string for `memory`, used by `mhlo.memory_kind` signature attributes and the
/// `_xla_buffer_placement` frontend attribute on `annotate_device_placement` custom calls. This mapping is owned by the
/// lowering on purpose: core's [`Memory`] exposes no backend vocabulary (its `Display` rendering is diagnostics-only),
/// mirroring how [`Sharding`](ryft_core::Sharding) converts to MLIR through backend-owned conversions.
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
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if output_type.memory() == Memory::Device {
        return Ok(value);
    }
    Ok(lower_transfer_to_memory(output_type.memory(), &[value], block, context, location)?[0])
}

/// Lowers one staged memory transfer to the `stablehlo.custom_call @annotate_device_placement` annotation that
/// XLA's `ConvertMemoryPlacementToInternalAnnotations` and
/// [`HostOffloader`](https://openxla.org/xla/tools_and_passes/host_offloading) passes legalize into memory-space
/// annotated copies. The annotation uses API version 1, `has_side_effect = true`, an empty `backend_config`, and
/// the destination kind in `_xla_buffer_placement` inside `mhlo.frontend_attributes`.
///
/// These constraints are retained on CPU as well: reference arrays only retag their types, but native arrays must
/// honor the requested physical placement on every platform. The current XLA offloading pass treats pinned and
/// unpinned host annotations alike internally; function boundary attributes and PJRT memory validation retain the
/// distinction at runtime boundaries. A plugin must expose the requested memory kind for native allocation.
///
/// Placement does not affect the MLIR tensor type, so the result type is the operand's type unchanged. Identity
/// transfers (destination equal to the operand's current space) still lower to the annotation: placement round
/// trips are meaningful to `HostOffloader` and must not be optimized away here.
pub(crate) fn lower_transfer_to_memory<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    destination: Memory,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 1, ProgramError);
    let empty_backend_config = context.string_attribute("");
    let mut operation = stable_hlo::custom_call(
        input_values,
        "annotate_device_placement",
        true,
        Some(empty_backend_config.as_ref()),
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
    Ok(vec![operation.result(0).expect("stablehlo.custom_call should return one result").as_ref()])
}

impl LowerableOperation for TransferToMemoryOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_transfer_to_memory(
            self.destination(),
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array, ArrayOperation, DataType, EagerContext, Placeholder, ReverseModeDifferentiate, Trace, ZeroOperation,
    };

    use crate::experimental::lowering::tests::{XlaProgramBuilder, test_vector_type};
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::{XlaArrayConstant, XlaConstant, XlaOperation};
    use crate::to_mlir_module_for_program;

    use super::*;

    #[test]
    fn test_transfer_to_memory_lowers_to_device_placement_annotations() {
        use ryft_core::TransferToMemory;

        // A compute-flanked host-and-back round trip lowers to one `annotate_device_placement` custom call per
        // transfer, carrying the destination kind in the `_xla_buffer_placement` frontend attribute — including the
        // identity-looking transfer back to device memory, which `HostOffloader` needs to see. The program mirrors
        // the JAX example in `python/scripts/dump_transfer_to_memory_mlir_from_jax.py`, and the asserted custom
        // calls are byte-identical to the ones JAX emits for it.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: ryft_core::tracing::DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let y = x.clone() * x;
                let on_host = y.transfer_to_memory(Memory::Host { pinned: true })?;
                let back = on_host.transfer_to_memory(Memory::Device)?;
                Ok(back.clone() * back)
            },
            test_vector_type(4),
        )
        .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        println!("=== ryft transfer_to_memory StableHLO ===\n{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(), 2, "{stablehlo}");
        assert!(
            stablehlo.contains(
                "stablehlo.custom_call @annotate_device_placement(%0) {backend_config = \"\", has_side_effect = \
                 true, mhlo.frontend_attributes = {_xla_buffer_placement = \"pinned_host\"}} : (tensor<4xf32>) -> \
                 tensor<4xf32>",
            ),
            "{stablehlo}",
        );
        assert!(
            stablehlo.contains(
                "stablehlo.custom_call @annotate_device_placement(%1) {backend_config = \"\", has_side_effect = \
                 true, mhlo.frontend_attributes = {_xla_buffer_placement = \"device\"}} : (tensor<4xf32>) -> \
                 tensor<4xf32>",
            ),
            "{stablehlo}",
        );
    }

    #[test]
    fn test_transfer_to_memory_signature_placement() {
        // A host identity has no transfer instruction: only its signature can communicate physical placement.
        // Exercise both the diagnostic homogeneous path and the compiled composite entry path.
        for (memory, kind) in
            [(Memory::Host { pinned: true }, "pinned_host"), (Memory::Host { pinned: false }, "unpinned_host")]
        {
            let r#type = ArrayType::new_static(DataType::I64, [2]).with_memory(memory);
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(r#type.clone());
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![input],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let expected = indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xi64> {mhlo.memory_kind = "KIND"}) -> (tensor<2xi64> {mhlo.memory_kind = "KIND"}) {
                    return %arg0 : tensor<2xi64>
                  }
                }
            "#}.replace("KIND", kind);
            assert_eq!(to_mlir_module_for_plain_program(&program, "main").unwrap(), expected);
            let program = program.into_unprojected::<XlaConstant, XlaOperation>().unwrap();
            let types = vec![r#type];
            assert_eq!(
                to_mlir_module_for_program(&program, &[], &types, &types, "main", None, None).unwrap(),
                expected,
            );
        }
    }

    #[test]
    fn test_synthesized_constant_output_lowers_with_its_memory_placement() {
        for (memory, placement) in [
            (Memory::Device, None),
            (Memory::Host { pinned: true }, Some("pinned_host")),
            (Memory::Host { pinned: false }, Some("unpinned_host")),
        ] {
            let output_type = test_vector_type(4).with_memory(memory);
            let mut builder = XlaProgramBuilder::new();
            let output =
                builder.add_instruction(ZeroOperation::new(output_type), Vec::new(), Vec::new(), None).unwrap()[0];
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], Vec::new(), vec![Placeholder])
                .unwrap();

            let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert_eq!(
                stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(),
                usize::from(placement.is_some()),
                "{stablehlo}",
            );
            if let Some(placement) = placement {
                assert!(stablehlo.contains(&format!("_xla_buffer_placement = \"{placement}\"")), "{stablehlo}");
            }
        }
    }

    #[test]
    fn test_transfer_to_memory_vjp_pullback_lowers_with_a_placement_annotation() {
        use ryft_core::TransferToMemory;

        // The pullback of a transfer moves the cotangent back to the operand's source memory (the default device
        // space here), so it lowers to an `annotate_device_placement` custom call targeting `device`.
        let (_, pullback): (Array, _) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(|x, ()| x.transfer_to_memory(Memory::Host { pinned: true }), Array::scalar(2.0).unwrap(), ())
            .unwrap();
        let (pullback, _residuals) = pullback.into_transposed_parts().unwrap();
        let pullback = pullback.into_program();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(), 1, "{stablehlo}");
        assert!(stablehlo.contains("_xla_buffer_placement = \"device\""), "{stablehlo}");
    }
}
