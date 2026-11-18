use std::collections::HashSet;

use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, CUSTOM_CALL_OPERATION_NAME, CustomCallArrayAttribute, CustomCallAttribute,
    CustomCallOperation, DataType, Dimension as CoreDimension, EffectClass, Layout, ProgramError, Shape,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::{CustomCallApiVersion, CustomCallMemoryLayouts};
use ryft_mlir::{
    Attribute, AttributeRef, Block, BlockRef, Context as MlirContext, LocationRef, NamedAttributeRef,
    Operation as MlirOperation, Region as MlirRegion, Size as MlirSize, StringRef, Type, Value as MlirValue, ValueRef,
};

use crate::lowering::types::{lower_physical_storage, lower_tensor_type, lower_unplaced_constant_output};
use crate::lowering::{EffectTokens, current_or_new_token};
use crate::operations::assertions::lower_output_extent_validation;
use crate::{
    CustomCallOperationExtension, Error, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError,
    MemberLowerableOperation, RUNTIME_REQUIREMENTS_ATTRIBUTE,
};

impl LowerableOperation for CustomCallOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        for requirement in self.runtime_requirements()? {
            lowerer.require_runtime(requirement)?;
        }
        let (mut block, context, location) = (lowerer.block(), lowerer.context(), lowerer.location());
        lower_custom_call_to_mlir(
            self,
            input_values,
            input_types,
            output_types,
            lowerer.effect_tokens_mut(),
            &mut block,
            context,
            location,
        )
    }
}

impl MemberLowerableOperation<ArrayIrType> for CustomCallOperation {
    fn lower_in_parent<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        for requirement in self.runtime_requirements()? {
            lowerer.require_runtime(requirement)?;
        }
        let context = lowerer.context();
        let location = lowerer.location();
        let block = &mut lowerer.block();
        let effect_tokens = lowerer.effect_tokens_mut();
        let dynamic_output_dimension_count = self
            .output_types()
            .iter()
            .flat_map(|output_type| output_type.shape().dimensions())
            .filter(|dimension| matches!(dimension, CoreDimension::Dynamic(_)))
            .count();
        let Some(array_input_count) = input_values.len().checked_sub(dynamic_output_dimension_count) else {
            return Err(ProgramError::InvalidInputCount {
                expected: dynamic_output_dimension_count,
                actual: input_values.len(),
            }
            .into());
        };
        let (array_inputs, output_extents) = input_values.split_at(array_input_count);
        let array_input_types = input_types[..array_input_count]
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        // Type inference refines the declared dynamic output axes whose extents are exact (e.g., those of eager
        // dimension values) to static axes, which are allocated at that extent and need no runtime refinement.
        let refined_output_types = output_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        if refined_output_types.len() != self.output_types().len() {
            return Err(ProgramError::InvalidOutputCount {
                expected: self.output_types().len(),
                actual: refined_output_types.len(),
            }
            .into());
        }
        // An exact trailing extent operand refines every occurrence of its dimension identity. Validate the symbolic
        // extent operands of refined axes before omitting them, and retain the predicate to gate the foreign
        // invocation itself.
        let validation = lower_output_extent_validation(
            CUSTOM_CALL_OPERATION_NAME,
            self.output_types(),
            output_extents,
            &input_types[array_input_count..],
            &refined_output_types,
            effect_tokens,
            block,
            context,
            location,
        )?;
        let physical_input_types = array_input_types
            .iter()
            .map(|input_type| custom_call_physical_type(input_type, "input"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut physical_output_types = refined_output_types
            .iter()
            .map(|output_type| custom_call_physical_type(output_type, "output"))
            .collect::<Result<Vec<_>, _>>()?;
        for alias in self.input_output_aliases() {
            // The aliased buffer takes the input's physical bounds but retains the output's kernel layout contract.
            let layout = physical_output_types[alias.output_index()].layout().cloned();
            physical_output_types[alias.output_index()] =
                physical_input_types[alias.input_index()].clone().with_layout(layout);
        }
        // XLA custom calls accept only static operands, so each bounded dynamic array input crosses the foreign
        // boundary as its dense buffer at the inclusive bound, exactly like the results allocated above. The kernel
        // therefore sees the whole buffer at the bound, whose elements beyond the runtime extent are unspecified. An
        // aliased output reuses that static operand buffer, and its logical extent is reattached after the call.
        // Static inputs pass through unchanged.
        let physical_inputs = array_inputs
            .iter()
            .zip(&array_input_types)
            .map(|(input, input_type)| lower_physical_storage(*input, input_type, block, context, location))
            .collect::<Result<Vec<_>, _>>()?;
        let mut results = if let Some(predicate) = validation {
            // A separate assertion token cannot gate a pure or nullary foreign ABI. The conditional keeps every
            // foreign effect in the valid branch. Its false results are unreachable because validation reports an
            // execution failure, so placeholders only satisfy StableHLO's requirement for matching branch signatures.
            let token_class = custom_call_token_class(self)?;
            let entry_token =
                token_class.map(|class| current_or_new_token(class, effect_tokens, block, location)).transpose()?;
            let mut true_region = context.region();
            let true_block = true_region.append_block(context.block_with_no_arguments())?;
            {
                let mut true_block = true_block.as_ref();
                let mut true_tokens = *effect_tokens;
                let mut outputs = lower_custom_call_to_mlir(
                    self,
                    &physical_inputs,
                    &physical_input_types,
                    &physical_output_types,
                    &mut true_tokens,
                    &mut true_block,
                    context,
                    location,
                )?;
                if let Some(class) = token_class {
                    outputs.push(true_tokens.get(class).unwrap());
                }
                true_block.append_operation(stable_hlo::r#return(&outputs, location)?)?;
            }
            let mut false_region = context.region();
            let false_block = false_region.append_block(context.block_with_no_arguments())?;
            {
                let mut false_block = false_block.as_ref();
                let mut outputs = physical_output_types
                    .iter()
                    .enumerate()
                    .map(|(index, r#type)| {
                        // Aliased outputs share their input's operand buffer at the bound. Forward that buffer instead
                        // of constructing a placeholder in this unreachable branch.
                        if let Some(alias) =
                            self.input_output_aliases().iter().find(|alias| alias.output_index() == index)
                        {
                            return Ok(physical_inputs[alias.input_index()]);
                        }
                        let value = i64::from(r#type.data_type() == DataType::F8E8M0FNU);
                        Ok(lower_unplaced_constant_output(
                            std::slice::from_ref(r#type),
                            value,
                            &mut false_block,
                            context,
                            location,
                        )?
                        .remove(0))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                if let Some(token) = entry_token {
                    outputs.push(token);
                }
                false_block.append_operation(stable_hlo::r#return(&outputs, location)?)?;
            }
            let guarded = block.append_operation(stable_hlo::r#if(predicate, true_region, false_region, location)?)?;
            if let Some(class) = token_class {
                effect_tokens.set(class, guarded.result(physical_output_types.len()).unwrap().as_ref());
            }
            (0..physical_output_types.len()).map(|index| guarded.result(index).unwrap().as_ref()).collect()
        } else {
            lower_custom_call_to_mlir(
                self,
                &physical_inputs,
                &physical_input_types,
                &physical_output_types,
                effect_tokens,
                block,
                context,
                location,
            )?
        };
        let i32_scalar_type = context.tensor_type(context.signless_integer_type(32), &[], None, location)?;
        let mut output_extents = output_extents.iter();
        for (output_index, output_type) in self.output_types().iter().enumerate() {
            let mut refined_type = physical_output_types[output_index].clone();
            let refined_dimensions = refined_output_types[output_index].shape().dimensions();
            for (axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
                if !matches!(dimension, CoreDimension::Dynamic(_)) {
                    continue;
                }
                let extent = *output_extents.next().unwrap();
                let Some(dimension @ CoreDimension::Dynamic(_)) = refined_dimensions.get(axis).cloned() else {
                    continue;
                };
                let extent = block.append_operation(stable_hlo::convert(extent, i32_scalar_type, location)?)?;
                let extent = extent.result(0).unwrap().as_ref();
                let mut dimensions = refined_type.shape().dimensions().to_vec();
                dimensions[axis] = dimension;
                refined_type = refined_type.with_shape(Shape::new(dimensions));
                let refined_tensor_type = lower_tensor_type(&refined_type, context, location)?;
                let result = block.append_operation(stable_hlo::set_dimension_size(
                    results[output_index],
                    extent,
                    refined_tensor_type,
                    axis,
                    location,
                )?)?;
                results[output_index] = result.result(0).unwrap().as_ref();
            }
        }
        Ok(results)
    }
}

/// Lowers one custom call to a `stablehlo.custom_call` using the typed FFI calling convention (i.e.,
/// `api_version = 4`). Typed attributes become the `backend_config` dictionary. Array layouts, which are the declared
/// input layouts or else the input and output types' own layouts, become complete StableHLO operand and result layout
/// lists, and flat input and output aliases become StableHLO output operand aliases. A call with an ordered effect
/// class additionally consumes and produces the current token of its ordering domain, which is the ordered assertion
/// chain or the ordered I/O chain for both [`EffectClass::OrderedIo`] and [`EffectClass::DeviceOrderedIo`]. Calls in
/// that class therefore remain ordered even when their array results do not carry a data dependency, and their
/// handlers receive a trailing token argument and result. An [`EffectClass::UnorderedIo`] call is only marked
/// `has_side_effect = true`, which keeps it alive without ordering it and gives its handler the same ABI without
/// tokens as a pure call, matching JAX's `has_side_effect=True`. The XLA runtime resolves handlers through the target
/// name at execution time (e.g., handlers registered with
/// [`Client::register_ffi_handler`](ryft_pjrt::Client::register_ffi_handler)).
fn lower_custom_call_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &CustomCallOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let token_class = custom_call_token_class(operation)?;
    let has_token = token_class.is_some();
    check_count!("input", input_types, input_values.len(), ProgramError);
    // Input type layouts describe the incoming values. The kernel receives the call's declared layouts instead,
    // with omitted declarations and `None` entries requesting dense row-major storage.
    let input_types = input_types
        .iter()
        .enumerate()
        .map(|(index, input_type)| {
            input_type.clone().with_layout(operation.input_layouts().get(index).cloned().flatten())
        })
        .collect::<Vec<_>>();
    let input_types = input_types.as_slice();
    let attributes = lower_custom_call_attributes(
        operation.attributes().iter().filter(|(name, _)| name != RUNTIME_REQUIREMENTS_ATTRIBUTE),
        context,
        location,
    )?;
    let backend_config = context.dictionary_attribute(&attributes);
    let memory_layouts = Some(lower_custom_call_memory_layouts(input_types, output_types, has_token)?);
    #[cfg(feature = "mosaic-gpu")]
    let memory_layouts = if operation.target_name() == ryft_xla_sys::mlir::dialects::mosaic::gpu::MOSAIC_GPU_FFI_TARGET
    {
        let mut layouts = crate::kernels::dense_memory_layouts(input_types, output_types, "mosaic GPU")?;
        if has_token {
            layouts.operands.push(Vec::new());
            layouts.results.push(Vec::new());
        }
        Some(layouts)
    } else {
        memory_layouts
    };
    // Compiler-provided static strides require fixed layouts even when the logical array type leaves them implicit.
    let memory_layouts = if operation.attributes().iter().any(|(name, value)| {
        matches!(name.as_str(), "ryft.cuda.row_major" | "ryft.rocm.row_major")
            && *value == CustomCallAttribute::Boolean(true)
    }) {
        let mut layouts = crate::kernels::dense_memory_layouts(input_types, output_types, "kernel")?;
        if has_token {
            layouts.operands.push(Vec::new());
            layouts.results.push(Vec::new());
        }
        Some(layouts)
    } else {
        memory_layouts
    };
    let mut lowered_inputs = input_values.to_vec();
    if let Some(token_class) = token_class {
        lowered_inputs.push(current_or_new_token(token_class, effect_tokens, block, location)?);
    }
    let mut lowered_output_types = output_types
        .iter()
        .map(|output_type| lower_tensor_type(output_type, context, location).map(|r#type| r#type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    if has_token {
        lowered_output_types.push(context.stable_hlo_token_type()?.as_ref());
    }
    let output_operand_aliases = operation
        .input_output_aliases()
        .iter()
        .map(|alias| {
            let output_tuple_indices =
                if lowered_output_types.len() == 1 { Vec::new() } else { vec![alias.output_index()] };
            context.stable_hlo_output_operand_alias(output_tuple_indices.as_slice(), alias.input_index(), &[])
        })
        .collect::<Result<Vec<_>, _>>()?;
    let lowered = block.append_operation(stable_hlo::custom_call(
        lowered_inputs.as_slice(),
        operation.target_name(),
        operation.has_side_effect(),
        Some(backend_config.as_ref()),
        CustomCallApiVersion::TypedFfi,
        &[],
        memory_layouts,
        output_operand_aliases.as_slice(),
        None,
        &lowered_output_types,
        location,
    )?)?;
    if let Some(token_class) = token_class {
        effect_tokens.set(token_class, lowered.result(output_types.len()).unwrap().as_ref());
    }
    Ok((0..output_types.len()).map(|index| lowered.result(index).unwrap().as_ref()).collect())
}

/// Returns the class of the effect token that a custom call threads for its effect class. That is the ordered
/// assertion token for [`EffectClass::OrderedAssertion`], the ordered I/O token for both [`EffectClass::OrderedIo`] and
/// [`EffectClass::DeviceOrderedIo`], and no token for pure and [`EffectClass::UnorderedIo`] calls.
///
/// # Errors
///
/// Returns [`LoweringError::UnresolvedState`] for [`EffectClass::OrderedState`], whose state must be resolved before
/// lowering.
fn custom_call_token_class(operation: &CustomCallOperation) -> Result<Option<EffectClass>, Error> {
    match operation.effect_class() {
        None | Some(EffectClass::UnorderedIo) => Ok(None),
        Some(EffectClass::OrderedAssertion) => Ok(Some(EffectClass::OrderedAssertion)),
        Some(EffectClass::OrderedIo | EffectClass::DeviceOrderedIo) => Ok(Some(EffectClass::OrderedIo)),
        Some(EffectClass::OrderedState) => Err(Error::Lowering(LoweringError::UnresolvedState {
            construct: format!("{CUSTOM_CALL_OPERATION_NAME} {}", operation.target_name()),
        })),
    }
}

/// Returns the statically shaped buffer type that crosses the foreign boundary for one custom call array `input` or
/// `output`, which `role` names. Every bounded dynamic dimension is widened to its inclusive upper bound, because XLA
/// custom calls accept only static operand and result shapes.
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedOperation`] for a dynamic dimension without a finite upper bound.
fn custom_call_physical_type(r#type: &ArrayType, role: &str) -> Result<ArrayType, Error> {
    let dimensions = r#type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| match dimension {
            CoreDimension::Static(extent) => Ok(CoreDimension::Static(*extent)),
            CoreDimension::Dynamic(variable) => variable
                .bounds()
                .upper()
                .and_then(|upper| upper.checked_sub(1))
                .map(CoreDimension::Static)
                .ok_or_else(|| {
                    Error::Lowering(LoweringError::UnsupportedOperation {
                        operation: CUSTOM_CALL_OPERATION_NAME.to_string(),
                        reason: format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` {role} dimension `{variable}` needs a finite upper bound \
                             for physical buffer allocation",
                        ),
                    })
                }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(r#type.clone().with_shape(Shape::new(dimensions)))
}

/// Returns complete StableHLO custom call layout constraints, using dense row-major storage for unspecified layouts.
fn lower_custom_call_memory_layouts(
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    has_effect_token: bool,
) -> Result<CustomCallMemoryLayouts, Error> {
    let mut operands = input_types.iter().map(lower_custom_call_layout).collect::<Result<Vec<_>, _>>()?;
    let mut results = output_types.iter().map(lower_custom_call_layout).collect::<Result<Vec<_>, _>>()?;
    if has_effect_token {
        operands.push(Vec::new());
        results.push(Vec::new());
    }
    Ok(CustomCallMemoryLayouts { operands, results })
}

/// Returns the minor-to-major layout required by one custom call array type.
fn lower_custom_call_layout(r#type: &ArrayType) -> Result<Vec<usize>, Error> {
    let Some(layout) = r#type.layout() else {
        return Ok((0..r#type.rank()).rev().collect());
    };
    let Layout::Tiled(layout) = layout else {
        return Err(Error::Lowering(LoweringError::UnsupportedOperation {
            operation: CUSTOM_CALL_OPERATION_NAME.to_string(),
            reason: format!("`{CUSTOM_CALL_OPERATION_NAME}` with strided array layout `{layout}`"),
        }));
    };
    if !layout.tiles().is_empty() {
        return Err(Error::Lowering(LoweringError::UnsupportedOperation {
            operation: CUSTOM_CALL_OPERATION_NAME.to_string(),
            reason: format!("`{CUSTOM_CALL_OPERATION_NAME}` with tiled array layout `{layout}`"),
        }));
    }
    if layout.rank() != r#type.rank()
        || layout.minor_to_major().iter().any(|axis| *axis >= r#type.rank())
        || layout.minor_to_major().iter().collect::<HashSet<_>>().len() != r#type.rank()
    {
        return Err(Error::Lowering(LoweringError::UnsupportedOperation {
            operation: CUSTOM_CALL_OPERATION_NAME.to_string(),
            reason: format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` with invalid array layout `{layout}` for rank-{} type `{type}`",
                r#type.rank(),
            ),
        }));
    }
    Ok(layout.minor_to_major().to_vec())
}

/// Lowers named custom call attributes to MLIR named attributes, in declaration order. Refer to
/// [`lower_custom_call_attribute`] for the encoding of each attribute kind.
fn lower_custom_call_attributes<'o, 'c, 't: 'c>(
    attributes: impl IntoIterator<Item = &'o (String, CustomCallAttribute)>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<NamedAttributeRef<'c, 't>>, Error> {
    attributes
        .into_iter()
        .map(|(name, value)| {
            let value = lower_custom_call_attribute(value, context, location)?;
            Ok(context.named_attribute(context.identifier(name.as_str()), value))
        })
        .collect()
}

/// Lowers one typed custom call attribute to the MLIR attribute that XLA's typed FFI decodes into the matching handler
/// attribute type, as implemented in XLA's
/// [`xla/ffi/attribute_map.cc`](https://github.com/openxla/xla/blob/main/xla/ffi/attribute_map.cc). Signed integers
/// become signless integer attributes and unsigned integers become unsigned integer attributes of the same width,
/// because the FFI decodes integer signedness from the attribute type. Signed integer and floating-point arrays become
/// dense array attributes, while
/// unsigned arrays become dense elements attributes of a rank-1 unsigned tensor type, which is the only unsigned
/// array encoding that the FFI decodes. Binary data becomes a string attribute holding the exact bytes, and
/// dictionaries recurse.
fn lower_custom_call_attribute<'c, 't: 'c>(
    value: &CustomCallAttribute,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<AttributeRef<'c, 't>, Error> {
    let unsigned_tensor_type = |bit_width: usize, length: usize| {
        context.tensor_type(context.unsigned_integer_type(bit_width), &[MlirSize::Static(length)], None, location)
    };
    Ok(match value {
        CustomCallAttribute::String(string) => context.string_attribute(string.as_str()).as_ref(),
        CustomCallAttribute::Bytes(bytes) => context.string_attribute(StringRef::from(bytes.as_slice())).as_ref(),
        CustomCallAttribute::Boolean(boolean) => context.boolean_attribute(*boolean).as_ref(),
        CustomCallAttribute::I8(integer) => {
            context.integer_attribute(context.signless_integer_type(8), i64::from(*integer)).as_ref()
        }
        CustomCallAttribute::I16(integer) => {
            context.integer_attribute(context.signless_integer_type(16), i64::from(*integer)).as_ref()
        }
        CustomCallAttribute::I32(integer) => {
            context.integer_attribute(context.signless_integer_type(32), i64::from(*integer)).as_ref()
        }
        CustomCallAttribute::I64(integer) => {
            context.integer_attribute(context.signless_integer_type(64), *integer).as_ref()
        }
        CustomCallAttribute::U8(integer) => {
            context.integer_attribute(context.unsigned_integer_type(8), i64::from(*integer)).as_ref()
        }
        CustomCallAttribute::U16(integer) => {
            context.integer_attribute(context.unsigned_integer_type(16), i64::from(*integer)).as_ref()
        }
        CustomCallAttribute::U32(integer) => {
            context.integer_attribute(context.unsigned_integer_type(32), i64::from(*integer)).as_ref()
        }
        // MLIR stores integer attributes as 64-bit words and the unsigned type reinterprets their bits.
        CustomCallAttribute::U64(integer) => {
            context.integer_attribute(context.unsigned_integer_type(64), integer.cast_signed()).as_ref()
        }
        CustomCallAttribute::F32(float) => context.float_attribute(context.float32_type(), f64::from(*float)).as_ref(),
        CustomCallAttribute::F64(float) => context.float_attribute(context.float64_type(), *float).as_ref(),
        CustomCallAttribute::Array(array) => match array {
            CustomCallArrayAttribute::I8(values) => context.dense_i8_array_attribute(values)?.as_ref(),
            CustomCallArrayAttribute::I16(values) => context.dense_i16_array_attribute(values)?.as_ref(),
            CustomCallArrayAttribute::I32(values) => context.dense_i32_array_attribute(values)?.as_ref(),
            CustomCallArrayAttribute::I64(values) => context.dense_i64_array_attribute(values)?.as_ref(),
            CustomCallArrayAttribute::U8(values) => {
                context.dense_u8_elements_attribute(unsigned_tensor_type(8, values.len())?, values)?.as_ref()
            }
            CustomCallArrayAttribute::U16(values) => {
                context.dense_u16_elements_attribute(unsigned_tensor_type(16, values.len())?, values)?.as_ref()
            }
            CustomCallArrayAttribute::U32(values) => {
                context.dense_u32_elements_attribute(unsigned_tensor_type(32, values.len())?, values)?.as_ref()
            }
            CustomCallArrayAttribute::U64(values) => {
                context.dense_u64_elements_attribute(unsigned_tensor_type(64, values.len())?, values)?.as_ref()
            }
            CustomCallArrayAttribute::F32(values) => context.dense_f32_array_attribute(values)?.as_ref(),
            CustomCallArrayAttribute::F64(values) => context.dense_f64_array_attribute(values)?.as_ref(),
        },
        CustomCallAttribute::Dictionary(attributes) => {
            context.dictionary_attribute(&lower_custom_call_attributes(attributes, context, location)?).as_ref()
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayIrType, BatchAxis, Composite, CustomCall, CustomCallArrayAttribute, CustomCallAttribute,
        CustomCallBatching, CustomCallOperation, CustomCallUnrolling, DataType, Device, DeviceMesh,
        Dimension as CoreDimension, DimensionBounds, DimensionSizeOperation, DimensionType, DimensionValue,
        DimensionVariable, DomainTracer, EffectClass, LogicalMesh, Memory, MeshAxis, MeshAxisType, Placeholder,
        PrintOperation, ProjectedContext, Reduce, ReductionKind, Shape, ShardMapOptions, Sharding, ShardingDimension,
        StaticShape, StridedLayout, Tile, TileDimension, TiledLayout, Typed, TypedProgram, batch, differentiate_at,
        shard_map,
    };
    use ryft_mlir::{DialectHandle, Operation as MlirOperation, Value as MlirValue, WalkOrder, WalkResult};
    use ryft_pjrt::extensions::ffi::{
        FfiAttribute, FfiAttributes, FfiBufferType, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler,
        FfiHandlerTraits, FfiInput, FfiOutput, FfiTypeId, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Handler,
    };
    use ryft_pjrt::{BufferType, Program};

    use crate::lowering::lower_mlir_module_for_program;
    use crate::lowering::tests::{
        MixedValue, ProgramBuilder, execute_mixed_program, static_sharded_array_type, test_matrix_type,
        test_spmd_compilation_options, unproject_plain_program,
    };
    use crate::operations::ProgramBuilder as ArrayIrProgramBuilder;
    use crate::operations::assertions::ensure_assertion_handler_registered;
    use crate::sessions::tests::{domain_mesh, session_with_client};
    use crate::tests::{
        ADD_ONE_CUSTOM_CALL_TARGET, cpu_client, ensure_add_one_handler_registered, replicated_array, values_from_bytes,
        values_to_bytes,
    };
    use crate::{
        Array, ArrayConstant, ArrayTracer, Constant, ExecutionArguments, FromPjrt, KernelEmbeddingError, RuntimeError,
        RuntimeRequirement, Session, TracedProgram, Value, to_mlir_module_for_program, trace,
    };

    use super::*;

    // Returns the type of an `f32` array with the provided dimensions that is replicated over `mesh`.
    fn replicated_f32_type(mesh: &DeviceMesh, dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F32, StaticShape::new(dimensions.to_vec()).into())
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), dimensions.len()))
            .unwrap()
    }

    #[test]
    fn test_custom_call_operation_lower() {
        // An ordered I/O custom call lowers its typed attributes, complete memory layout lists, buffer alias, and one
        // hidden ordered I/O token. A second ordered I/O call consumes that token even though the first call's array
        // result is unused, while the final pure call remains free of tokens.
        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2), CoreDimension::Static(3)]));
        let column_major_type = array_type.clone().with_layout(Some(TiledLayout::new(vec![0, 1], Vec::new()).into()));
        let mut builder = ProgramBuilder::new();
        let left = builder.add_input(column_major_type.clone());
        let right = builder.add_input(array_type.clone());
        let effectful =
            CustomCallOperation::new("ryft.test.scaled_add", vec![array_type.clone(), column_major_type.clone()])
                .with_attribute("scale", 2.0)
                .with_attribute("count", 4i64)
                .with_attribute("flag", true)
                .with_attribute("label", "x")
                .with_input_layouts([column_major_type.layout().cloned(), None])
                .with_input_output_alias(0, 1)
                .unwrap()
                .with_effect_class(EffectClass::OrderedIo);
        builder.add_instruction(effectful, Vec::new(), vec![left, right], None).unwrap();
        let printed = builder
            .add_instruction(PrintOperation::<ArrayType>::new("between"), Vec::new(), vec![right], None)
            .unwrap()[0];
        let chained = builder
            .add_instruction(
                CustomCallOperation::new("ryft.test.record", vec![array_type.clone()])
                    .with_effect_class(EffectClass::OrderedIo),
                Vec::new(),
                vec![printed],
                None,
            )
            .unwrap()[0];
        let pure = CustomCallOperation::new("ryft.test.add_one", vec![array_type.clone()]);
        let output = builder.add_instruction(pure, Vec::new(), vec![chained], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
        let input_types = vec![column_major_type, array_type.clone()];
        let output_types = vec![array_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\"}, %arg1: tensor<2x3xf32>) \
                        -> tensor<2x3xf32> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:3 = stablehlo.custom_call @ryft.test.scaled_add(%arg0, %arg1, %0) {api_version = 4 : i32, \
                        backend_config = {count = 4 : i64, flag = true, label = \"x\", scale = 2.000000e+00 : \
                        f64}, has_side_effect = true, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, \
                        dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = \
                        [#stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 0, \
                        operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0, \
                        1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<2x3xf32>, tensor<2x3xf32>, \
                        !stablehlo.token) -> (tensor<2x3xf32>, tensor<2x3xf32>, !stablehlo.token)
                    %2 = stablehlo.custom_call @ryft.print(%arg1, %1#2) {api_version = 4 : i32, backend_config = \
                        {label = \"between\"}, has_side_effect = true} : (tensor<2x3xf32>, !stablehlo.token) -> \
                        !stablehlo.token
                    %3:2 = stablehlo.custom_call @ryft.test.record(%arg1, %2) {api_version = 4 : i32, \
                        backend_config = {}, has_side_effect = true, operand_layouts = [dense<[1, 0]> : \
                        tensor<2xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<[1, 0]> : \
                        tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<2x3xf32>, !stablehlo.token) -> \
                        (tensor<2x3xf32>, !stablehlo.token)
                    %4 = stablehlo.custom_call @ryft.test.add_one(%3#0) {api_version = 4 : i32, backend_config = \
                        {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>], result_layouts = \
                        [dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x3xf32>) -> tensor<2x3xf32>
                    return %4 : tensor<2x3xf32>
                  }
                }
            "},
        );

        // Byte-strided descriptors cannot be represented by StableHLO's custom call layouts, which are permutations.
        let strided_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2), CoreDimension::Static(3)]))
                .with_layout(Some(StridedLayout::new(vec![12, 4]).into()));
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(strided_type.clone());
        let output = builder
            .add_instruction(
                CustomCallOperation::new("ryft.test.strided", vec![strided_type.clone()]),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
        assert!(matches!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![strided_type.clone()],
                &vec![strided_type],
                "main",
                None,
                None,
            ),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                if reason == "strided input layout `strided{12,4}`",
        ));
        let invalid_permutation_type =
            test_matrix_type(2, 3).with_layout(Some(TiledLayout::new(vec![0, 0], Vec::new()).into()));
        assert!(matches!(
            lower_custom_call_layout(&invalid_permutation_type),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                if reason == "`custom_call` with invalid array layout `tiled{0,0}` for rank-2 type \
                   `f32[2, 3][layout=tiled{0,0}]`",
        ));
        let tiled_type = test_matrix_type(2, 3)
            .with_layout(Some(TiledLayout::new(vec![1, 0], vec![Tile::new(vec![TileDimension::Sized(2)])]).into()));
        assert!(matches!(
            lower_custom_call_layout(&tiled_type),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                if reason == "`custom_call` with tiled array layout `tiled{1,0:T(2)}`",
        ));
    }

    #[test]
    fn test_custom_call_operation_lower_attributes_layouts_and_unordered_effects() {
        // Every attribute kind lowers to the MLIR attribute that XLA's typed FFI decodes into the matching handler
        // type, a declared input layout replaces the input's own default layout, and an unordered effectful call
        // is only marked `has_side_effect` without threading a token through its handler ABI.
        let array_type = test_matrix_type(2, 3);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let operation = CustomCallOperation::new("ryft.test.configured", vec![array_type.clone()])
            .with_attribute("string", "x")
            .with_attribute("bytes", vec![0u8, 255])
            .with_attribute("boolean", true)
            .with_attribute("i8", -8i8)
            .with_attribute("i16", -16i16)
            .with_attribute("i32", -32i32)
            .with_attribute("i64", -64i64)
            .with_attribute("u8", 8u8)
            .with_attribute("u16", 16u16)
            .with_attribute("u32", 32u32)
            .with_attribute("u64", u64::MAX)
            .with_attribute("f32", 2.5f32)
            .with_attribute("f64", 3.5f64)
            .with_attribute("i8_array", vec![1i8, -2])
            .with_attribute("i16_array", vec![1i16, -2])
            .with_attribute("i32_array", vec![1i32, -2])
            .with_attribute("i64_array", vec![1i64, -2])
            .with_attribute("u8_array", CustomCallArrayAttribute::U8(vec![1, 2]))
            .with_attribute("u16_array", vec![1u16, 2])
            .with_attribute("u32_array", vec![1u32, 2])
            .with_attribute("u64_array", vec![1u64, 2])
            .with_attribute("f32_array", vec![0.5f32, 1.5])
            .with_attribute("f64_array", vec![0.25f64])
            .with_attribute(
                "dictionary",
                CustomCallAttribute::Dictionary(vec![
                    ("mode".to_string(), CustomCallAttribute::from("fast")),
                    (
                        "nested".to_string(),
                        CustomCallAttribute::Dictionary(vec![("size".to_string(), CustomCallAttribute::from(4i32))]),
                    ),
                ]),
            )
            .with_input_layouts([Some(TiledLayout::new(vec![0, 1], Vec::new()).into())])
            .with_effect_class(EffectClass::UnorderedIo);
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
        let module =
            to_mlir_module_for_program(&program, &[], &vec![array_type.clone()], &vec![array_type], "main", None, None)
                .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
                    %0 = stablehlo.custom_call @ryft.test.configured(%arg0) {api_version = 4 : i32, backend_config = {boolean = true, bytes = "\00\FF", dictionary = {mode = "fast", nested = {size = 4 : i32}}, f32 = 2.500000e+00 : f32, f32_array = array<f32: 5.000000e-01, 1.500000e+00>, f64 = 3.500000e+00 : f64, f64_array = array<f64: 2.500000e-01>, i16 = -16 : i16, i16_array = array<i16: 1, -2>, i32 = -32 : i32, i32_array = array<i32: 1, -2>, i64 = -64 : i64, i64_array = array<i64: 1, -2>, i8 = -8 : i8, i8_array = array<i8: 1, -2>, string = "x", u16 = 16 : ui16, u16_array = dense<[1, 2]> : tensor<2xui16>, u32 = 32 : ui32, u32_array = dense<[1, 2]> : tensor<2xui32>, u64 = 18446744073709551615 : ui64, u64_array = dense<[1, 2]> : tensor<2xui64>, u8 = 8 : ui8, u8_array = dense<[1, 2]> : tensor<2xui8>}, has_side_effect = true, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x3xf32>) -> tensor<2x3xf32>
                    return %0 : tensor<2x3xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_custom_call_operation_lower_binary_attributes() {
        let mut builder = ProgramBuilder::new();
        builder
            .add_instruction(
                CustomCallOperation::new("ryft.test.binary", vec![])
                    .with_attribute("module", vec![0u8, 0x80, 0xff])
                    .with_attribute("empty", Vec::<u8>::new()),
                vec![],
                vec![],
                None,
            )
            .unwrap();
        let program = builder.build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![], vec![]).unwrap();
        let program = unproject_plain_program(program);
        assert_eq!(
            to_mlir_module_for_program(
                &program,
                &[],
                &Vec::<ArrayType>::new(),
                &Vec::<ArrayType>::new(),
                "main",
                None,
                None,
            )
            .unwrap(),
            indoc! {r#"
                module {
                  func.func @main() {
                    stablehlo.custom_call @ryft.test.binary() {api_version = 4 : i32, backend_config = {empty = "", module = "\00\80\FF"}, operand_layouts = [], result_layouts = []} : () -> ()
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_custom_call_operation_lower_preserves_kernel_layout_error() {
        let input_type = test_matrix_type(2, 3);
        let output_type = input_type.clone().with_layout(Some(TiledLayout::new(vec![0, 1], Vec::new()).into()));
        let cases = [
            ("ryft.test.row_major", Some("ryft.cuda.row_major"), "kernel"),
            ("ryft.test.row_major", Some("ryft.rocm.row_major"), "kernel"),
            #[cfg(feature = "mosaic-gpu")]
            (ryft_xla_sys::mlir::dialects::mosaic::gpu::MOSAIC_GPU_FFI_TARGET, None, "mosaic GPU"),
        ];
        for (target, row_major, owner) in cases {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(input_type.clone());
            let mut operation = CustomCallOperation::new(target, vec![output_type.clone()]);
            if let Some(row_major) = row_major {
                operation = operation.with_attribute(row_major, true);
            }
            let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            let program = builder
                .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            let error = to_mlir_module_for_program(
                &unproject_plain_program(program),
                &[],
                &vec![input_type.clone()],
                &vec![output_type.clone()],
                "main",
                None,
                None,
            )
            .unwrap_err();
            assert!(matches!(
                error.downcast_custom::<KernelEmbeddingError>(),
                Some(KernelEmbeddingError::Invalid { message })
                    if message == &format!("{owner} requires untiled dense row-major array layouts"),
            ));
        }
    }

    #[test]
    fn test_custom_call_runtime_metadata_is_consumed_without_changing_native_attributes() {
        let array_type = test_matrix_type(2, 3);
        let first = RuntimeRequirement::new("test.first", 2).unwrap();
        let second = RuntimeRequirement::new("test.second", 1).unwrap();
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(array_type.clone().into());
        let operation = CustomCallOperation::new("ryft.test.runtime", vec![array_type.clone()])
            .with_attribute("scale", 2i64)
            .with_attribute(
                "nested",
                CustomCallAttribute::Dictionary(vec![(
                    RUNTIME_REQUIREMENTS_ATTRIBUTE.to_string(),
                    "foreign-metadata".into(),
                )]),
            )
            .with_runtime_requirements([second.clone(), first.clone(), first.clone()])
            .unwrap();
        let intermediate = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let operation = CustomCallOperation::new("ryft.test.runtime", vec![array_type.clone()])
            .with_runtime_requirements([second.clone()])
            .unwrap();
        let output = builder.add_instruction(operation, Vec::new(), vec![intermediate], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = lower_mlir_module_for_program(
            &program,
            &[],
            &vec![array_type.clone()],
            &vec![array_type],
            "main",
            None,
            None,
            None,
        )
        .unwrap();

        let (stable_hlo, _, _, runtime_requirements) = lowered.into_parts();
        assert_eq!(runtime_requirements.as_ref(), &[first, second]);
        assert_eq!(
            stable_hlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
                    %0 = stablehlo.custom_call @ryft.test.runtime(%arg0) {api_version = 4 : i32, backend_config = \
                        {nested = {ryft.runtime.requirements = \"foreign-metadata\"}, scale = 2 : i64}, \
                        operand_layouts = [dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : \
                        tensor<2xindex>]} : (tensor<2x3xf32>) -> tensor<2x3xf32>
                    %1 = stablehlo.custom_call @ryft.test.runtime(%0) {api_version = 4 : i32, backend_config = {}, \
                        operand_layouts = [dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : \
                        tensor<2xindex>]} : (tensor<2x3xf32>) -> tensor<2x3xf32>
                    return %1 : tensor<2x3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_custom_call_runtime_requirements_conflict_across_operations() {
        let array_type = test_matrix_type(2, 3);
        let mut builder = ArrayIrProgramBuilder::new();
        let mut output = builder.add_input(array_type.clone().into());
        for version in [1, 2] {
            let operation = CustomCallOperation::new("ryft.test.runtime", vec![array_type.clone()])
                .with_runtime_requirements([RuntimeRequirement::new("test.runtime", version).unwrap()])
                .unwrap();
            output = builder.add_instruction(operation, Vec::new(), vec![output], None).unwrap()[0];
        }
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            lower_mlir_module_for_program(
                &program, &[], &vec![array_type.clone()], &vec![array_type], "main", None, None, None,
            ),
            Err(Error::Runtime(RuntimeError::ConflictingRequirements { identifier })) if identifier == "test.runtime",
        ));
    }

    #[test]
    fn test_custom_call_operation_lower_threads_assertions() {
        let scalar = ArrayType::scalar(DataType::I64);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(scalar.clone());
        #[cfg(not(feature = "mosaic-gpu"))]
        let target = "ryft.test.assertion";
        #[cfg(feature = "mosaic-gpu")]
        let target = ryft_xla_sys::mlir::dialects::mosaic::gpu::MOSAIC_GPU_FFI_TARGET;
        let call =
            CustomCallOperation::new(target, vec![scalar.clone()]).with_effect_class(EffectClass::OrderedAssertion);
        let first = builder.add_instruction(call.clone(), vec![], vec![input], None).unwrap()[0];
        builder.add_instruction(call, vec![], vec![input], None).unwrap();
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![first], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
        let text = to_mlir_module_for_program(&program, &[], &vec![scalar.clone()], &vec![scalar], "main", None, None)
            .unwrap();
        let context = MlirContext::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        context.load_dialect(DialectHandle::stable_hlo().unwrap()).unwrap();
        let module = context.parse_module(&text).unwrap();
        assert!(module.verify().unwrap());
        let mut calls = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if operation.name().as_str() == Ok("stablehlo.custom_call") {
                calls.push(operation);
            }
            WalkResult::Advance
        });
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].operand_count(), 2);
        assert_eq!(calls[0].result_count(), 2);
        #[cfg(feature = "mosaic-gpu")]
        {
            let operands = calls[0].array_attribute("operand_layouts").unwrap();
            let results = calls[0].array_attribute("result_layouts").unwrap();
            assert_eq!(operands.len(), 2);
            assert_eq!(results.len(), 2);
            assert_eq!(operands.element(1).unwrap().to_string(), "dense<> : tensor<0xindex>");
            assert_eq!(results.element(1).unwrap().to_string(), "dense<> : tensor<0xindex>");
        }
        assert_eq!(calls[1].operand(1).unwrap().value().unwrap(), calls[0].result(1).unwrap().as_ref());
    }

    #[test]
    fn test_custom_call_operation_lower_threads_assertions_without_array_results() {
        let mut builder = ProgramBuilder::new();
        let call =
            CustomCallOperation::new("ryft.test.assertion", vec![]).with_effect_class(EffectClass::OrderedAssertion);
        builder.add_instruction(call.clone(), vec![], vec![], None).unwrap();
        builder.add_instruction(call, vec![], vec![], None).unwrap();
        let program = builder.build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![], vec![]).unwrap();
        let program = unproject_plain_program(program);
        let text = to_mlir_module_for_program(
            &program,
            &[],
            &Vec::<ArrayType>::new(),
            &Vec::<ArrayType>::new(),
            "main",
            None,
            None,
        )
        .unwrap();
        let context = MlirContext::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        context.load_dialect(DialectHandle::stable_hlo().unwrap()).unwrap();
        let module = context.parse_module(&text).unwrap();
        assert!(module.verify().unwrap());
        let mut calls = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if operation.name().as_str() == Ok("stablehlo.custom_call") {
                calls.push(operation);
            }
            WalkResult::Advance
        });
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].operand_count(), 1);
        assert_eq!(calls[0].result_count(), 1);
        assert_eq!(calls[1].operand(0).unwrap().value().unwrap(), calls[0].result(0).unwrap().as_ref());
    }

    #[test]
    fn test_custom_call_operation_lower_dynamic_alias() {
        // An array IR custom call receives its declared extent as trailing scalar SSA metadata, but only the array
        // operand enters the FFI call, as its static buffer at the bound. An alias continues to refer to that leading
        // operand, and the static aliased result is refined with the trailing logical extent after the call.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent)]));
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(dynamic_type.clone().into());
        let extent = builder
            .add_instruction(DimensionSizeOperation::new(&dynamic_type, 0).unwrap(), Vec::new(), vec![input], None)
            .unwrap()[0];
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![dynamic_type.clone()])
            .with_input_output_alias(0, 0)
            .unwrap();
        let output = builder.add_instruction(operation, Vec::new(), vec![input, extent], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![dynamic_type.clone()],
            &vec![dynamic_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<8>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<8xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %c = stablehlo.constant dense<8> : tensor<i32>
                    %3 = stablehlo.set_dimension_size %0, %c, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>, tensor<i32>) -> tensor<8xf32>
                    %4 = stablehlo.custom_call @ryft.test.dynamic(%3) {api_version = 4 : i32, backend_config = {}, operand_layouts = [dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<8xf32>) -> tensor<8xf32>
                    %5 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
                    %6 = stablehlo.set_dimension_size %4, %5, dim = 0 : (tensor<8xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %7 = stablehlo.get_dimension_size %6, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %8 = stablehlo.convert %7 : (tensor<i32>) -> tensor<i64>
                    return %6, %8 : tensor<?xf32, #stablehlo.bounds<8>>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_custom_call_operation_lower_dynamic_alias_executes_on_bound_buffer() {
        const TARGET: &str = "ryft.test.increment_bound_buffer";
        static OPERAND_LENGTH: AtomicUsize = AtomicUsize::new(0);
        static REGISTRATION: OnceLock<Result<(), ryft_pjrt::Error>> = OnceLock::new();

        /// Writes every element of the `f64` operand plus one into the result and records the operand length.
        unsafe extern "C" fn handler(raw: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
            // SAFETY: XLA provides a valid call frame, operand buffer, and result allocation for this invocation, and
            // the handler checks that the operand and result have the same `f64` type and extents.
            unsafe {
                let Ok(frame) = FfiCallFrame::from_c_api(raw) else { return std::ptr::null_mut() };
                if frame.register_metadata(FfiTypeId::default()) || frame.stage() != FfiExecutionStage::Execution {
                    return std::ptr::null_mut();
                }
                let Ok(api) = frame.api() else { return std::ptr::null_mut() };
                let (Ok(FfiInput::Buffer { buffer: input }), Ok(FfiOutput::Buffer { buffer: output })) =
                    (frame.input(0), frame.output(0))
                else {
                    return FfiError::invalid_argument("expected one operand and one result buffer").to_c_api(api);
                };
                if input.element_type() != FfiBufferType::F64
                    || output.element_type() != FfiBufferType::F64
                    || input.dimensions() != output.dimensions()
                {
                    return FfiError::invalid_argument("expected matching `f64` buffers").to_c_api(api);
                }
                let length = input.dimensions().iter().product::<i64>() as usize;
                for index in 0..length {
                    *(output.data() as *mut f64).add(index) = *(input.data() as *const f64).add(index) + 1.0;
                }
                OPERAND_LENGTH.store(length, Ordering::SeqCst);
                std::ptr::null_mut()
            }
        }

        let client = cpu_client(1);
        REGISTRATION
            .get_or_init(|| {
                client.register_ffi_handler(
                    TARGET,
                    &client.platform_name()?,
                    FfiHandler::from(handler as XLA_FFI_Handler),
                    FfiHandlerTraits::NONE,
                )
            })
            .clone()
            .unwrap();

        // A bounded dynamic aliased input reaches the handler as its static buffer at the bound, and the aliased
        // result regains the input's runtime extent after the call.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![CoreDimension::Dynamic(extent)]));
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(dynamic_type.clone().into());
        let extent = builder
            .add_instruction(DimensionSizeOperation::new(&dynamic_type, 0).unwrap(), Vec::new(), vec![input], None)
            .unwrap()[0];
        let operation = CustomCallOperation::new(TARGET, vec![dynamic_type]).with_input_output_alias(0, 0).unwrap();
        let output = builder.add_instruction(operation, Vec::new(), vec![input, extent], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let input = MixedValue::Array((1..=8).map(f64::from).collect(), vec![8]);
        assert_eq!(
            execute_mixed_program(client, &program, &[input], &[3]).unwrap(),
            vec![MixedValue::Array(vec![2.0, 3.0, 4.0], vec![3]), MixedValue::Dimension(3)],
        );
        assert_eq!(OPERAND_LENGTH.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn test_custom_call_operation_lower_rejects_unbounded_input() {
        // An unbounded dynamic input has no static buffer that could cross the foreign boundary.
        let size = DimensionVariable::new("size", DimensionBounds::unbounded());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(size)]));
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let operation = CustomCallOperation::new("ryft.test.unbounded", vec![output_type.clone()]);
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            to_mlir_module_for_program(&program, &[], &vec![input_type], &vec![output_type], "main", None, None),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                if reason
                    == "`custom_call` input dimension `size` needs a finite upper bound for physical buffer \
                        allocation",
        ));
    }

    #[test]
    fn test_custom_call_operation_lower_with_exact_output_extent() {
        // An exact extent refines the declared dynamic output axis to a static one, so the call allocates that exact
        // extent and its redundant extent operand does not refine the result.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let mut builder = ArrayIrProgramBuilder::new();
        let extent =
            builder.add_constant(Constant::Dimension(DimensionValue::new(DimensionType::from(extent), 3).unwrap()));
        let operation = CustomCallOperation::new("ryft.test.exact", vec![dynamic_type]);
        let output = builder.add_instruction(operation, Vec::new(), vec![extent], None).unwrap()[0];
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![], vec![Placeholder]).unwrap();
        let output_type = ArrayType::new_static(DataType::F32, [3]);
        assert_eq!(program.output_types(), vec![ArrayIrType::Array(output_type.clone())]);
        let module =
            to_mlir_module_for_program(&program, &[], &Vec::<ArrayType>::new(), &vec![output_type], "main", None, None)
                .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main() -> tensor<3xf32> {
                    %0 = stablehlo.custom_call @ryft.test.exact() {api_version = 4 : i32, backend_config = {}, operand_layouts = [], result_layouts = [dense<0> : tensor<1xindex>]} : () -> tensor<3xf32>
                    return %0 : tensor<3xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_custom_call_operation_lower_mixed_extents_preserve_dynamic_alias_false_branch() {
        // The mixed output needs a guard, while the independent aliased output shares the static operand buffer at the
        // bound of its dynamic input. The unreachable branch forwards that buffer, and the logical extent is
        // reattached after the conditional.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let size = DimensionVariable::new("size", DimensionBounds::new(1, Some(9)).unwrap());
        let alias_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]));
        let mixed_type = ArrayType::new(DataType::F32, Shape::new(vec![size.clone().into(), size.clone().into()]));
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(alias_type.clone().into());
        let symbolic = builder.add_input(DimensionType::from(size).into());
        let alias_extent = builder
            .add_instruction(DimensionSizeOperation::new(&alias_type, 0).unwrap(), Vec::new(), vec![input], None)
            .unwrap()[0];
        let exact = builder.add_constant(Constant::Dimension(DimensionValue::constant(3).unwrap()));
        let operation = CustomCallOperation::new("ryft.test.dynamic_guard", vec![alias_type.clone(), mixed_type])
            .with_effect_class(EffectClass::OrderedIo)
            .with_input_output_alias(0, 0)
            .unwrap();
        let outputs = builder
            .add_instruction(operation, Vec::new(), vec![input, alias_extent, symbolic, exact], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let lowered = lower_mlir_module_for_program(
            &program,
            &[],
            &vec![alias_type.clone(), ArrayType::scalar(DataType::I64)],
            &vec![alias_type, ArrayType::new_static(DataType::F32, [3, 3])],
            "main",
            None,
            None,
            None,
        )
        .unwrap();
        let (stable_hlo, signature, requires_assertion_handler, runtime_requirements) = lowered.into_parts();
        assert_eq!(runtime_requirements.as_ref(), &[RuntimeRequirement::builtin("ryft.assert")]);
        // This direct lowering fixture has no ordered boundary slot and needs no transport admission.
        assert!(!signature.has_ordered_io());
        assert!(requires_assertion_handler);
        let module = stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<i64>, %arg2: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<8>>, tensor<3x3xf32>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<8xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %c = stablehlo.constant dense<3> : tensor<i64>
                    %3 = stablehlo.compare EQ, %arg1, %c, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %4 = stablehlo.after_all  : !stablehlo.token
                    %5 = stablehlo.custom_call @ryft.assert(%3, %arg1, %4) {api_version = 4 : i32, backend_config = {actor = "custom_call", kind = "generic", label_0 = "size", label_count = "1", message = "input dimension `size` expected 3"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %c_0 = stablehlo.constant dense<8> : tensor<i32>
                    %6 = stablehlo.set_dimension_size %0, %c_0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>, tensor<i32>) -> tensor<8xf32>
                    %7 = stablehlo.after_all  : !stablehlo.token
                    %8:3 = "stablehlo.if"(%3) ({
                      %13:3 = stablehlo.custom_call @ryft.test.dynamic_guard(%6, %7) {api_version = 4 : i32, backend_config = {}, has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<8xf32>, !stablehlo.token) -> (tensor<8xf32>, tensor<3x3xf32>, !stablehlo.token)
                      stablehlo.return %13#0, %13#1, %13#2 : tensor<8xf32>, tensor<3x3xf32>, !stablehlo.token
                    }, {
                      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
                      stablehlo.return %6, %13, %7 : tensor<8xf32>, tensor<3x3xf32>, !stablehlo.token
                    }) : (tensor<i1>) -> (tensor<8xf32>, tensor<3x3xf32>, !stablehlo.token)
                    %9 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
                    %10 = stablehlo.set_dimension_size %8#0, %9, dim = 0 : (tensor<8xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %11 = stablehlo.get_dimension_size %10, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %12 = stablehlo.convert %11 : (tensor<i32>) -> tensor<i64>
                    return %10, %8#1, %12 : tensor<?xf32, #stablehlo.bounds<8>>, tensor<3x3xf32>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_custom_call_operation_lower_mixed_extents_guard_effect_tokens() {
        // The guarded call threads the token of its effect class through both conditional branches. An ordered
        // assertion call continues the chain of the extent assertion, while device ordered I/O shares the ordered I/O
        // chain.
        for (effect, expected) in [
            (
                EffectClass::OrderedAssertion,
                indoc! {r#"
                    module {
                      func.func @main(%arg0: tensor<i64>) -> tensor<3x3xf64> {
                        %c = stablehlo.constant dense<3> : tensor<i64>
                        %0 = stablehlo.compare EQ, %arg0, %c, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                        %1 = stablehlo.after_all  : !stablehlo.token
                        %2 = stablehlo.custom_call @ryft.assert(%0, %arg0, %1) {api_version = 4 : i32, backend_config = {actor = "custom_call", kind = "generic", label_0 = "size", label_count = "1", message = "input dimension `size` expected 3"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                        %3:2 = "stablehlo.if"(%0) ({
                          %4:2 = stablehlo.custom_call @ryft.test.guarded(%2) {api_version = 4 : i32, backend_config = {}, has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (!stablehlo.token) -> (tensor<3x3xf64>, !stablehlo.token)
                          stablehlo.return %4#0, %4#1 : tensor<3x3xf64>, !stablehlo.token
                        }, {
                          %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                          %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
                          stablehlo.return %4, %2 : tensor<3x3xf64>, !stablehlo.token
                        }) : (tensor<i1>) -> (tensor<3x3xf64>, !stablehlo.token)
                        return %3#0 : tensor<3x3xf64>
                      }
                    }
                "#},
            ),
            (
                EffectClass::DeviceOrderedIo,
                indoc! {r#"
                    module {
                      func.func @main(%arg0: tensor<i64>) -> tensor<3x3xf64> {
                        %c = stablehlo.constant dense<3> : tensor<i64>
                        %0 = stablehlo.compare EQ, %arg0, %c, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                        %1 = stablehlo.after_all  : !stablehlo.token
                        %2 = stablehlo.custom_call @ryft.assert(%0, %arg0, %1) {api_version = 4 : i32, backend_config = {actor = "custom_call", kind = "generic", label_0 = "size", label_count = "1", message = "input dimension `size` expected 3"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                        %3 = stablehlo.after_all  : !stablehlo.token
                        %4:2 = "stablehlo.if"(%0) ({
                          %5:2 = stablehlo.custom_call @ryft.test.guarded(%3) {api_version = 4 : i32, backend_config = {}, has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (!stablehlo.token) -> (tensor<3x3xf64>, !stablehlo.token)
                          stablehlo.return %5#0, %5#1 : tensor<3x3xf64>, !stablehlo.token
                        }, {
                          %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                          %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
                          stablehlo.return %5, %3 : tensor<3x3xf64>, !stablehlo.token
                        }) : (tensor<i1>) -> (tensor<3x3xf64>, !stablehlo.token)
                        return %4#0 : tensor<3x3xf64>
                      }
                    }
                "#},
            ),
        ] {
            let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(9)).unwrap());
            let output_type = ArrayType::new(DataType::F64, Shape::new(vec![size.clone().into(), size.clone().into()]));
            let mut builder = ArrayIrProgramBuilder::new();
            let symbolic = builder.add_input(DimensionType::from(size).into());
            let exact = builder.add_constant(Constant::Dimension(DimensionValue::constant(3).unwrap()));
            let operation = CustomCallOperation::new("ryft.test.guarded", vec![output_type]).with_effect_class(effect);
            let output = builder.add_instruction(operation, Vec::new(), vec![symbolic, exact], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            let module = to_mlir_module_for_program(
                &program,
                &[],
                &vec![ArrayType::scalar(DataType::I64)],
                &vec![ArrayType::new_static(DataType::F64, [3, 3])],
                "main",
                None,
                None,
            )
            .unwrap();
            assert_eq!(module, expected, "{effect:?}");
        }
    }

    #[test]
    fn test_custom_call_operation_lower_mixed_extents_false_branch_omits_memory_placement() {
        // The unreachable placeholder matches the unannotated result of the guarded call, even when the declared
        // output has a non-default memory placement.
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![size.clone().into(), size.clone().into()]))
            .with_memory(Memory::Host { pinned: true });
        let mut builder = ArrayIrProgramBuilder::new();
        let symbolic = builder.add_input(DimensionType::from(size).into());
        let exact = builder.add_constant(Constant::Dimension(DimensionValue::constant(3).unwrap()));
        let operation = CustomCallOperation::new("ryft.test.guarded", vec![output_type]);
        let output = builder.add_instruction(operation, Vec::new(), vec![symbolic, exact], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![ArrayType::scalar(DataType::I64)],
            &vec![ArrayType::new_static(DataType::F64, [3, 3]).with_memory(Memory::Host { pinned: true })],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i64>) -> (tensor<3x3xf64> {mhlo.memory_kind = "pinned_host"}) {
                    %c = stablehlo.constant dense<3> : tensor<i64>
                    %0 = stablehlo.compare EQ, %arg0, %c, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.assert(%0, %arg0, %1) {api_version = 4 : i32, backend_config = {actor = "custom_call", kind = "generic", label_0 = "size", label_count = "1", message = "input dimension `size` expected 3"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %3 = "stablehlo.if"(%0) ({
                      %4 = stablehlo.custom_call @ryft.test.guarded() {api_version = 4 : i32, backend_config = {}, operand_layouts = [], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : () -> tensor<3x3xf64>
                      stablehlo.return %4 : tensor<3x3xf64>
                    }, {
                      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                      %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
                      stablehlo.return %4 : tensor<3x3xf64>
                    }) : (tensor<i1>) -> tensor<3x3xf64>
                    return %3 : tensor<3x3xf64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_custom_call_operation_lower_mixed_extents_validate_before_foreign_effects() {
        const TARGET: &str = "ryft.test.validated_nullary_extent";
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        static REGISTRATION: OnceLock<Result<(), ryft_pjrt::Error>> = OnceLock::new();

        /// Records execution of the nullary foreign call and initializes its `f64` result buffers.
        unsafe extern "C" fn handler(raw: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
            // SAFETY: XLA provides a valid call frame and result allocation for this invocation.
            unsafe {
                let Ok(frame) = FfiCallFrame::from_c_api(raw) else { return std::ptr::null_mut() };
                if frame.register_metadata(FfiTypeId::default()) || frame.stage() != FfiExecutionStage::Execution {
                    return std::ptr::null_mut();
                }
                let Ok(api) = frame.api() else { return std::ptr::null_mut() };
                for output in frame.outputs() {
                    let buffer = match output {
                        Ok(FfiOutput::Buffer { buffer }) => buffer,
                        Err(error) => return error.to_c_api(api),
                    };
                    if buffer.element_type() == FfiBufferType::Token {
                        continue;
                    }
                    if buffer.element_type() != FfiBufferType::F64 {
                        return FfiError::invalid_argument("expected an `f64` result buffer").to_c_api(api);
                    }
                    let Some(length) = buffer
                        .dimensions()
                        .iter()
                        .try_fold(1usize, |length, &extent| length.checked_mul(usize::try_from(extent).ok()?))
                    else {
                        return FfiError::invalid_argument("invalid result buffer extents").to_c_api(api);
                    };
                    if length > 0 && buffer.data().is_null() {
                        return FfiError::invalid_argument("null result buffer data").to_c_api(api);
                    }
                    for index in 0..length {
                        *(buffer.data() as *mut f64).add(index) = 7.0;
                    }
                }
                CALLS.fetch_add(1, Ordering::SeqCst);
                std::ptr::null_mut()
            }
        }

        // The counting handler writes its results through host pointers, so this test always runs on the CPU client
        // rather than on the configured execution platform, whose result buffers may live in device memory.
        let client = cpu_client(1);
        ensure_assertion_handler_registered(client).unwrap();
        REGISTRATION
            .get_or_init(|| {
                client.register_ffi_handler(
                    TARGET,
                    &client.platform_name()?,
                    FfiHandler::from(handler as XLA_FFI_Handler),
                    FfiHandlerTraits::NONE,
                )
            })
            .clone()
            .unwrap();
        for exact_first in [false, true] {
            for effect in [
                None,
                Some(EffectClass::UnorderedIo),
                Some(EffectClass::OrderedAssertion),
                Some(EffectClass::OrderedIo),
                Some(EffectClass::DeviceOrderedIo),
            ] {
                let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(9)).unwrap());
                let output_type =
                    ArrayType::new(DataType::F64, Shape::new(vec![size.clone().into(), size.clone().into()]));
                let mut builder = ArrayIrProgramBuilder::new();
                let symbolic = builder.add_input(DimensionType::from(size).into());
                let exact = builder.add_constant(Constant::Dimension(DimensionValue::constant(3).unwrap()));
                let inputs = if exact_first { vec![exact, symbolic] } else { vec![symbolic, exact] };
                let operation = CustomCallOperation::new(TARGET, vec![output_type]);
                let operation = match effect {
                    Some(effect) => operation.with_effect_class(effect),
                    None => operation,
                };
                let output = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap()[0];
                let program = builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                    .unwrap();
                let before = CALLS.load(Ordering::SeqCst);
                assert_eq!(
                    execute_mixed_program(client, &program, &[MixedValue::Dimension(3)], &[]).unwrap(),
                    vec![MixedValue::Array(vec![7.0; 9], vec![3, 3])],
                );
                assert_eq!(CALLS.load(Ordering::SeqCst), before + 1);
                let error =
                    execute_mixed_program(client, &program, &[MixedValue::Dimension(4)], &[]).unwrap_err().to_string();
                assert!(error.contains("input dimension `size` expected 3"), "{error}");
                assert_eq!(CALLS.load(Ordering::SeqCst), before + 1);
            }
        }
        // A guarded call forwards its aliased input buffer from the unreachable branch. A valid extent runs the call
        // on that buffer, while an invalid one fails before the call runs.
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(9)).unwrap());
        let alias_type = ArrayType::new_static(DataType::F64, [4]);
        let mixed_type = ArrayType::new(DataType::F64, Shape::new(vec![size.clone().into(), size.clone().into()]));
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(alias_type.clone().into());
        let symbolic = builder.add_input(DimensionType::from(size).into());
        let exact = builder.add_constant(Constant::Dimension(DimensionValue::constant(3).unwrap()));
        let operation = CustomCallOperation::new(TARGET, vec![alias_type, mixed_type])
            .with_effect_class(EffectClass::OrderedIo)
            .with_input_output_alias(0, 0)
            .unwrap();
        let outputs =
            builder.add_instruction(operation, Vec::new(), vec![input, symbolic, exact], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let input = MixedValue::Array(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let before = CALLS.load(Ordering::SeqCst);
        assert_eq!(
            execute_mixed_program(client, &program, &[input.clone(), MixedValue::Dimension(3)], &[]).unwrap(),
            vec![MixedValue::Array(vec![7.0; 4], vec![4]), MixedValue::Array(vec![7.0; 9], vec![3, 3])],
        );
        assert_eq!(CALLS.load(Ordering::SeqCst), before + 1);
        let error = execute_mixed_program(client, &program, &[input, MixedValue::Dimension(4)], &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("input dimension `size` expected 3"), "{error}");
        assert_eq!(CALLS.load(Ordering::SeqCst), before + 1);
    }

    #[test]
    fn test_custom_call_operation_lower_executes_registered_ffi_handler() {
        let client = cpu_client(1);
        ensure_add_one_handler_registered(client).unwrap();
        let session = session_with_client(client);
        let mesh = domain_mesh(client, "x", 1);

        // Eager dispatch compiles a single `stablehlo.custom_call` that resolves to the registered handler at
        // execution time.
        let input =
            replicated_array(&session, &mesh, DataType::F32, &[2, 2], &values_to_bytes(&[1.5f32, 2.5, 3.5, 4.5]));
        let operation = CustomCallOperation::new(ADD_ONE_CUSTOM_CALL_TARGET, vec![replicated_f32_type(&mesh, &[2, 2])]);
        let outputs = CustomCall::custom_call(&operation, [&input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(values_from_bytes::<f32>(&outputs[0].copy_to_host().unwrap()), vec![2.5f32, 3.5, 4.5, 5.5]);

        // A typed `f64` attribute reaches the handler through the `backend_config` dictionary.
        let operation = operation.with_attribute("increment", 2.5f64);
        let outputs = CustomCall::custom_call(&operation, [&input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(values_from_bytes::<f32>(&outputs[0].copy_to_host().unwrap()), vec![4f32, 5.0, 6.0, 7.0]);

        // An aliased ordered I/O call keeps the handler ABI with only arrays while its lowering threads a hidden token.
        let input = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1.5f32, 2.5]));
        let operation = CustomCallOperation::new(ADD_ONE_CUSTOM_CALL_TARGET, vec![replicated_f32_type(&mesh, &[2])])
            .with_input_output_alias(0, 0)
            .unwrap()
            .with_effect_class(EffectClass::OrderedIo);
        let outputs = CustomCall::custom_call(&operation, [&input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(values_from_bytes::<f32>(&outputs[0].copy_to_host().unwrap()), vec![2.5f32, 3.5]);
    }

    #[test]
    fn test_custom_call_operation_lower_passes_attributes_to_ffi_handler() {
        // This handler records how XLA's typed FFI decodes every attribute of its call, which verifies that each
        // `CustomCallAttribute` kind reaches handlers with exactly its declared scalar or array element type, and that
        // bytes that are not valid UTF-8 arrive unchanged. It also rejects any token operand, because an unordered
        // effectful call uses the same ABI without tokens as a pure call.
        const TARGET: &str = "ryft.test.inspect_attributes";
        static OBSERVED_ATTRIBUTES: Mutex<Vec<String>> = Mutex::new(Vec::new());
        static REGISTRATION: OnceLock<Result<(), ryft_pjrt::Error>> = OnceLock::new();

        fn render_entry(name: &str, attribute: FfiAttribute<'_>) -> Result<String, FfiError> {
            let value = match attribute {
                FfiAttribute::Scalar { scalar } => format!("{scalar:?}"),
                FfiAttribute::String { string } => format!("{string:?}"),
                FfiAttribute::Bytes { bytes } => format!("{bytes:02x?}"),
                FfiAttribute::Array { array } => format!("{:?}", array.iter().collect::<Result<Vec<_>, _>>()?),
                FfiAttribute::Dictionary { dictionary } => render_dictionary(&dictionary)?,
            };
            Ok(format!("{name}={value}"))
        }

        fn render_dictionary(dictionary: &FfiAttributes<'_>) -> Result<String, FfiError> {
            let entries = dictionary
                .iter()
                .map(|entry| entry.and_then(|(name, attribute)| render_entry(name, attribute)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("{{{}}}", entries.join(", ")))
        }

        fn inspect(call_frame: &FfiCallFrame<'_>) -> Result<(), FfiError> {
            let inputs = call_frame.inputs().collect::<Result<Vec<_>, _>>()?;
            if inputs.len() != 1
                || inputs.iter().any(|FfiInput::Buffer { buffer }| buffer.element_type() == FfiBufferType::Token)
            {
                return Err(FfiError::invalid_argument("expected exactly one array input and no token input"));
            }
            let entries = call_frame
                .attributes()
                .map(|entry| entry.and_then(|(name, attribute)| render_entry(name, attribute)))
                .collect::<Result<Vec<_>, _>>()?;
            OBSERVED_ATTRIBUTES.lock().unwrap().push(entries.join(", "));
            Ok(())
        }

        unsafe extern "C" fn handler(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
            // SAFETY: The XLA runtime passes a call frame that is valid for the duration of this invocation.
            unsafe {
                let Ok(call_frame) = FfiCallFrame::from_c_api(call_frame) else { return std::ptr::null_mut() };
                if call_frame.register_metadata(FfiTypeId::default())
                    || call_frame.stage() != FfiExecutionStage::Execution
                {
                    return std::ptr::null_mut();
                }
                let Ok(api) = call_frame.api() else { return std::ptr::null_mut() };
                match inspect(&call_frame) {
                    Ok(()) => std::ptr::null_mut(),
                    Err(error) => error.to_c_api(api),
                }
            }
        }

        let client = cpu_client(1);
        REGISTRATION
            .get_or_init(|| {
                client.register_ffi_handler(
                    TARGET,
                    &client.platform_name()?,
                    FfiHandler::from(handler as XLA_FFI_Handler),
                    FfiHandlerTraits::NONE,
                )
            })
            .clone()
            .unwrap();
        let session = session_with_client(client);
        let mesh = domain_mesh(client, "x", 1);
        let input = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.0]));
        let operation = CustomCallOperation::new(TARGET, Vec::new())
            .with_attribute("boolean", true)
            .with_attribute("bytes", CustomCallAttribute::Bytes(vec![0x00, 0xff, 0x80]))
            .with_attribute("i8", -8i8)
            .with_attribute("i16", -16i16)
            .with_attribute("i32", -32i32)
            .with_attribute("i64", -64i64)
            .with_attribute("u8", 8u8)
            .with_attribute("u16", 16u16)
            .with_attribute("u32", 32u32)
            .with_attribute("u64", u64::MAX)
            .with_attribute("f32", 2.5f32)
            .with_attribute("f64", 3.5f64)
            .with_attribute("string", "x")
            .with_attribute("i8_array", vec![1i8, -2])
            .with_attribute("i16_array", vec![1i16, -2])
            .with_attribute("i32_array", vec![1i32, -2])
            .with_attribute("i64_array", vec![1i64, -2])
            .with_attribute("u8_array", CustomCallArrayAttribute::U8(vec![1, 2]))
            .with_attribute("u16_array", vec![1u16, 2])
            .with_attribute("u32_array", vec![1u32, 2])
            .with_attribute("u64_array", vec![1u64, 2])
            .with_attribute("f32_array", vec![0.5f32, 1.5])
            .with_attribute("f64_array", vec![0.25f64])
            .with_attribute(
                "dictionary",
                CustomCallAttribute::Dictionary(vec![("size".to_string(), CustomCallAttribute::from(4i32))]),
            )
            .with_effect_class(EffectClass::UnorderedIo);
        assert!(matches!(CustomCall::custom_call(&operation, [&input]), Ok(outputs) if outputs.is_empty()));

        // Wait for the unordered I/O before inspecting the recorded attributes.
        session.synchronize().unwrap();
        assert_eq!(
            *OBSERVED_ATTRIBUTES.lock().unwrap(),
            vec![
                "boolean=Predicate(true), bytes=[00, ff, 80], dictionary={size=I32(4)}, f32=F32(2.5), \
                 f32_array=[F32(0.5), F32(1.5)], f64=F64(3.5), f64_array=[F64(0.25)], i16=I16(-16), \
                 i16_array=[I16(1), I16(-2)], i32=I32(-32), i32_array=[I32(1), I32(-2)], i64=I64(-64), \
                 i64_array=[I64(1), I64(-2)], i8=I8(-8), i8_array=[I8(1), I8(-2)], string=\"x\", u16=U16(16), \
                 u16_array=[U16(1), U16(2)], u32=U32(32), u32_array=[U32(1), U32(2)], u64=U64(18446744073709551615), \
                 u64_array=[U64(1), U64(2)], u8=U8(8), u8_array=[U8(1), U8(2)]"
                    .to_string(),
            ],
        );
    }

    #[test]
    fn test_custom_call_operation_lower_passes_declared_input_layout_to_ffi_handler() {
        const TARGET: &str = "ryft.test.physical_f32_copy";
        static REGISTRATION: OnceLock<Result<(), ryft_pjrt::Error>> = OnceLock::new();

        // Copies the physical storage of the single `f32` input, in memory order, into the rank-1 `f32` output.
        unsafe extern "C" fn handler(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
            // SAFETY: The XLA runtime passes a call frame that is valid for the duration of this invocation, and the
            // operations below declare an output with as many elements as their input.
            unsafe {
                let Ok(call_frame) = FfiCallFrame::from_c_api(call_frame) else { return std::ptr::null_mut() };
                if call_frame.register_metadata(FfiTypeId::default())
                    || call_frame.stage() != FfiExecutionStage::Execution
                {
                    return std::ptr::null_mut();
                }
                let Ok(api) = call_frame.api() else { return std::ptr::null_mut() };
                let (Ok(FfiInput::Buffer { buffer: input }), Ok(FfiOutput::Buffer { buffer: output })) =
                    (call_frame.input(0), call_frame.output(0))
                else {
                    return FfiError::invalid_argument("expected one input and one output buffer").to_c_api(api);
                };
                let length = output.dimensions().iter().product::<i64>() as usize;
                std::ptr::copy(input.data() as *const f32, output.data() as *mut f32, length);
                std::ptr::null_mut()
            }
        }

        let client = cpu_client(1);
        REGISTRATION
            .get_or_init(|| {
                client.register_ffi_handler(
                    TARGET,
                    &client.platform_name()?,
                    FfiHandler::from(handler as XLA_FFI_Handler),
                    FfiHandlerTraits::NONE,
                )
            })
            .clone()
            .unwrap();
        let session = session_with_client(client);
        let mesh = domain_mesh(client, "x", 1);
        let values = values_to_bytes(&[1f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()));
        let row_major_input = replicated_array(&session, &mesh, DataType::F32, &[2, 3], &values);
        let column_major_input = Array::from_host_buffer(
            &session,
            replicated_f32_type(&mesh, &[2, 3]).with_layout(column_major.clone()),
            mesh.clone(),
            None,
            values.as_slice(),
        )
        .unwrap();

        // A declared input layout is the buffer contract of the kernel. The backend converts a row-major input to the
        // declared column-major layout before invoking the handler, and passes a column-major input as it is.
        let operation = CustomCallOperation::new(TARGET, vec![replicated_f32_type(&mesh, &[6])])
            .with_input_layouts([Some(column_major)]);
        let outputs = CustomCall::custom_call(&operation, [&row_major_input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(values_from_bytes::<f32>(&outputs[0].copy_to_host().unwrap()), vec![1f32, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let outputs = CustomCall::custom_call(&operation, [&column_major_input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(values_from_bytes::<f32>(&outputs[0].copy_to_host().unwrap()), vec![1f32, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Both an omitted declaration and a `None` entry request row-major storage, even for a column-major input.
        let operation = CustomCallOperation::new(TARGET, vec![replicated_f32_type(&mesh, &[6])]);
        let outputs = CustomCall::custom_call(&operation, [&column_major_input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(values_from_bytes::<f32>(&outputs[0].copy_to_host().unwrap()), vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let outputs = CustomCall::custom_call(&operation.with_input_layouts([None]), [&column_major_input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(values_from_bytes::<f32>(&outputs[0].copy_to_host().unwrap()), vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Aliasing applies to the converted row-major buffer. The handler also permits input and output overlap.
        let operation = CustomCallOperation::new(TARGET, vec![replicated_f32_type(&mesh, &[2, 3])])
            .with_input_output_alias(0, 0)
            .unwrap();
        let outputs = CustomCall::custom_call(&operation, [&column_major_input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(values_from_bytes::<f32>(&outputs[0].copy_to_host().unwrap()), vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_custom_call_operation_lower_executes_nullary_ffi_handler() {
        const TARGET: &str = "ryft.test.fill_iota";
        static REGISTRATION: OnceLock<Result<(), ryft_pjrt::Error>> = OnceLock::new();

        // Fills the single rank-1 `f32` output with `0, 1, 2, ...`.
        unsafe extern "C" fn handler(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
            // SAFETY: The XLA runtime passes a call frame that is valid for the duration of this invocation, including
            // the output allocation whose element count its dimensions describe.
            unsafe {
                let Ok(call_frame) = FfiCallFrame::from_c_api(call_frame) else { return std::ptr::null_mut() };
                if call_frame.register_metadata(FfiTypeId::default())
                    || call_frame.stage() != FfiExecutionStage::Execution
                {
                    return std::ptr::null_mut();
                }
                let Ok(api) = call_frame.api() else { return std::ptr::null_mut() };
                if call_frame.input_count() != 0 {
                    return FfiError::invalid_argument("expected no inputs").to_c_api(api);
                }
                let Ok(FfiOutput::Buffer { buffer: output }) = call_frame.output(0) else {
                    return FfiError::invalid_argument("expected one output buffer").to_c_api(api);
                };
                let length = output.dimensions().iter().product::<i64>() as usize;
                let values = std::slice::from_raw_parts_mut(output.data() as *mut f32, length);
                for (index, value) in values.iter_mut().enumerate() {
                    *value = index as f32;
                }
                std::ptr::null_mut()
            }
        }

        let client = cpu_client(1);
        REGISTRATION
            .get_or_init(|| {
                client.register_ffi_handler(
                    TARGET,
                    &client.platform_name()?,
                    FfiHandler::from(handler as XLA_FFI_Handler),
                    FfiHandlerTraits::NONE,
                )
            })
            .clone()
            .unwrap();
        let session = session_with_client(client);
        let mesh = domain_mesh(client, "x", 1);

        // A call without array inputs binds directly in the eager session and runs a source kernel that only fills its
        // output.
        let operation = CustomCallOperation::new(TARGET, vec![replicated_f32_type(&mesh, &[4])]);
        let outputs = operation.call(&session, &[]).unwrap();
        let [Value::Array(output)] = outputs.as_slice() else {
            panic!("expected one array output but got {} outputs", outputs.len());
        };
        assert_eq!(values_from_bytes::<f32>(&output.copy_to_host().unwrap()), vec![0f32, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_custom_call_operation_lower_executes_batched_ffi_handler() {
        let client = cpu_client(1);
        ensure_add_one_handler_registered(client).unwrap();
        let session = session_with_client(client);
        let mesh = domain_mesh(client, "x", 1);
        let row_type = replicated_f32_type(&mesh, &[2]);
        let input = replicated_array(
            &session,
            &mesh,
            DataType::F32,
            &[3, 2],
            &values_to_bytes(&[1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5]),
        );

        // `CustomCallBatching::BroadcastAll` materializes every operand on the batch axis, so the elementwise handler
        // receives one buffer with a leading batch axis in a single call.
        let broadcast_row_type = row_type.clone();
        let output: Array = batch(
            move |row| {
                let operation = CustomCallOperation::new(ADD_ONE_CUSTOM_CALL_TARGET, vec![broadcast_row_type])
                    .with_batching(CustomCallBatching::BroadcastAll);
                Ok(CustomCall::custom_call(&operation, [&row])?.remove(0))
            },
            input.clone(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(values_from_bytes::<f32>(&output.copy_to_host().unwrap()), vec![2.5f32, 3.5, 4.5, 5.5, 6.5, 7.5]);

        // `CustomCallBatching::Sequential` stages a `scan` that applies the kernel once per batch item. The eager
        // batching parent is a `ProjectedContext`, which lifts that `scan` and its body region into the array IR
        // family.
        for unroll in [CustomCallUnrolling::Disabled, CustomCallUnrolling::Factor(2), CustomCallUnrolling::Full] {
            let sequential_row_type = row_type.clone();
            let output: Array = batch(
                move |row| {
                    let operation = CustomCallOperation::new(ADD_ONE_CUSTOM_CALL_TARGET, vec![sequential_row_type])
                        .with_batching(CustomCallBatching::Sequential { unroll });
                    Ok(CustomCall::custom_call(&operation, [&row])?.remove(0))
                },
                input.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
            assert_eq!(
                values_from_bytes::<f32>(&output.copy_to_host().unwrap()),
                vec![2.5f32, 3.5, 4.5, 5.5, 6.5, 7.5],
            );
        }
    }

    #[test]
    fn test_custom_call_operation_lower_inside_manual_region_executes_on_cpu() {
        let client = cpu_client(2);
        let session = session_with_client(client);
        ensure_add_one_handler_registered(client).unwrap();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 2);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));

        // The custom call executes per shard inside the manual region. Its declared output type is the local shard
        // type, and the lowered `stablehlo.custom_call` appears inside the `sdy.manual_computation` body.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            let operation = CustomCallOperation::new(
                                ADD_ONE_CUSTOM_CALL_TARGET,
                                vec![local_x.r#type().into_owned()],
                            );
                            Ok(CustomCall::custom_call(&operation, std::slice::from_ref(&local_x))?.remove(0))
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap()
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.custom_call @ryft.test.add_one(%arg1) {api_version = 4 : i32, backend_config = {}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[4], sharding),
            device_mesh,
            None,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 2);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            ExecutionArguments::from_arrays(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(
                execute_arguments.as_execution_device_inputs(),
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2] and [3, 4], so the kernel produces [2, 3] and [4, 5] respectively.
        assert_eq!(outputs.len(), execution_device_ids.len());
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, [device_index as f32 * 2.0 + 2.0, device_index as f32 * 2.0 + 3.0]);
        }
    }

    #[test]
    fn test_custom_call_operation_lower_executes_under_composite_differentiation() {
        let client = cpu_client(1);
        ensure_add_one_handler_registered(client).unwrap();
        let session = session_with_client(client);
        let mesh = domain_mesh(client, "x", 1);
        let input = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1.5f32, 2.5]));

        // The bare operation rejects differentiation, so a composite supplies its rules. The primal and the forward
        // rule execute the kernel through the projection of the eager session onto arrays, and the backward rule passes
        // the cotangent through because d(x + 1)/dx is the identity.
        let operation = CustomCallOperation::new(ADD_ONE_CUSTOM_CALL_TARGET, vec![replicated_f32_type(&mesh, &[2])]);
        let function = Composite::from_custom_call(operation.clone()).with_vjp(
            move |inputs: Vec<DomainTracer<ProjectedContext<Session, ArrayType>>>| {
                Ok((CustomCall::custom_call(&operation, &inputs)?, ()))
            },
            |(), cotangents| Ok(cotangents),
        );
        let (value, gradient) = differentiate_at(input)
            .value_and_gradient(|x| function.call(vec![x]).unwrap().remove(0).reduce(&[0], ReductionKind::Sum).unwrap())
            .unwrap();
        assert_eq!(values_from_bytes::<f32>(&value.copy_to_host().unwrap()), vec![6f32]);
        assert_eq!(values_from_bytes::<f32>(&gradient.copy_to_host().unwrap()), vec![1f32, 1.0]);
    }

    #[test]
    fn test_custom_call_token_class() {
        let operation = CustomCallOperation::new("ryft.test.effect", Vec::new());
        assert_eq!(custom_call_token_class(&operation).unwrap(), None);
        assert_eq!(
            custom_call_token_class(&operation.clone().with_effect_class(EffectClass::UnorderedIo)).unwrap(),
            None,
        );
        assert_eq!(
            custom_call_token_class(&operation.clone().with_effect_class(EffectClass::OrderedAssertion)).unwrap(),
            Some(EffectClass::OrderedAssertion),
        );
        assert_eq!(
            custom_call_token_class(&operation.clone().with_effect_class(EffectClass::OrderedIo)).unwrap(),
            Some(EffectClass::OrderedIo),
        );
        assert_eq!(
            custom_call_token_class(&operation.clone().with_effect_class(EffectClass::DeviceOrderedIo)).unwrap(),
            Some(EffectClass::OrderedIo),
        );
        assert!(matches!(
            custom_call_token_class(&operation.with_effect_class(EffectClass::OrderedState)),
            Err(Error::Lowering(LoweringError::UnresolvedState { construct }))
                if construct == "custom_call ryft.test.effect",
        ));
    }

    #[test]
    fn test_custom_call_physical_type() {
        // Bounded dynamic dimensions widen to their inclusive upper bound, and static dimensions are unchanged.
        let bounded = DimensionVariable::new("size", DimensionBounds::new(0, Some(5)).unwrap());
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![bounded.into(), CoreDimension::Static(2)]));
        assert_eq!(custom_call_physical_type(&r#type, "output").unwrap(), ArrayType::new_static(DataType::F32, [4, 2]));

        let unbounded = DimensionVariable::new("size", DimensionBounds::unbounded());
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![unbounded.into()]));
        assert!(matches!(
            custom_call_physical_type(&r#type, "output"),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == CUSTOM_CALL_OPERATION_NAME
                    && reason
                        == "`custom_call` output dimension `size` needs a finite upper bound for physical buffer \
                            allocation",
        ));
    }
}
