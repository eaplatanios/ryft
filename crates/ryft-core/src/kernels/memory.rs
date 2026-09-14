//! Scoped scratch allocation and masked memory access using canonical array types, views, and reference lifetimes.

use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

use thiserror::Error;

use crate::arrays::{
    Array, ArrayAddressing, ArrayIrType, ArrayIrValue, ArrayReference, ArrayReferenceView, ArraySliceAxis, ArrayType,
    DataType, DimensionVariable,
};
use crate::contexts::{Domain, EagerContext};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::programs::{
    EffectClasses, Effects, Operation, OperationFormatter, ProgramError, ReferenceAccessMode, ReferenceEffect,
    ReferenceType, RegionInterface, Type, TypeError, TypeIdentityRenaming, Typed,
};

/// Errors encountered when declaring or accessing kernel memory.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum KernelMemoryError {
    /// An ordinary memory operation selects an invalid lane of a padded kernel window.
    #[error("kernel operation `{operation}` accesses an invalid window lane without a mask")]
    UnmaskedWindowAccess { operation: &'static str },

    /// Logical initialization and resource lifetimes must be qualified before this operation can execute.
    #[error("kernel operation `{operation}` requires a qualified invocation")]
    RequiresQualification { operation: &'static str },

    /// The requested alignment is not a positive power of two.
    #[error("scratch alignment `{alignment}` must be a positive power of two")]
    InvalidAlignment { alignment: usize },

    /// The canonical array type cannot describe statically addressable storage.
    #[error(transparent)]
    Program(#[from] ProgramError),
}

/// Canonical operation name for [`ScratchOperation`].
pub const SCRATCH_OPERATION_NAME: &str = "scratch";

/// Allocates logically uninitialized storage scoped to the enclosing kernel region. Reads require a preceding
/// definite write to every selected element; physical initialization by a reference interpreter does not establish
/// logical initialization. Reference analysis enforces the allocation's lifetime and prevents it from escaping.
///
/// The referent's canonical [`ArrayType`] carries its shape, layout, and memory placement. Alignment is a minimum byte
/// alignment required of native storage; the abstract reference interpreter does not expose physical addresses.
/// Initialized scratch uses the existing [`ReferenceNewOperation`](crate::operations::ReferenceNewOperation).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScratchOperation {
    /// Canonical type of the allocated array.
    referent: ArrayType,
    /// Required minimum native allocation alignment in bytes.
    alignment: usize,
}

impl ScratchOperation {
    /// Creates a scratch allocation with a static, addressable referent and a positive power-of-two alignment.
    pub fn new(referent: ArrayType, alignment: usize) -> Result<Self, KernelMemoryError> {
        if !alignment.is_power_of_two() {
            return Err(KernelMemoryError::InvalidAlignment { alignment });
        }
        ArrayAddressing::new(referent.clone())?;
        Ok(Self { referent, alignment })
    }

    /// Returns the canonical array type allocated by this operation.
    pub fn referent(&self) -> &ArrayType {
        &self.referent
    }

    /// Returns the minimum native allocation alignment in bytes.
    pub fn alignment(&self) -> usize {
        self.alignment
    }
}

impl Display for ScratchOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ScratchOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        SCRATCH_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        Ok(vec![ReferenceType::new(self.referent.clone()).into()])
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self { referent: self.referent.rename_identities(renaming)?, alignment: self.alignment })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("alignment", self.alignment))
    }

    fn effects(&self) -> Cow<'_, Effects> {
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Allocate { output_index: 0 }], Vec::new()).unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

/// Canonical operation name for [`MaskedLoadOperation`].
pub const MASKED_LOAD_OPERATION_NAME: &str = "masked_load";

/// Loads active reference lanes and returns `other` for inactive lanes.
/// Operands are `(reference, mask, other)`; the Boolean mask has the referent's exact shape. The remaining array
/// values have its exact type. Broadcasting and padding must be explicit. The reference itself must be valid: a mask
/// does not authorize an out-of-bounds reference view, and inactive lanes never access storage.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaskedLoadOperation;

impl Display for MaskedLoadOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(MASKED_LOAD_OPERATION_NAME)
    }
}

impl Operation for MaskedLoadOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        MASKED_LOAD_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        let referent = validate_masked_types(input_types, region_interfaces, 3, 1)?;
        Ok(vec![referent.into()])
    }

    fn effects(&self) -> Cow<'_, Effects> {
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
                vec![],
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl<O: Operation<Type = ArrayIrType>> InterpretableOperation<EagerContext<ArrayIrValue<Array>, O>>
    for MaskedLoadOperation
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, O>>>(
        &self,
        _context: &EagerContext<ArrayIrValue<Array>, O>,
        _driver: &D,
        inputs: &[ArrayIrValue<Array>],
    ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
        self.infer_output_types(&inputs.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        let [ArrayIrValue::Reference(reference), ArrayIrValue::Array(mask), ArrayIrValue::Array(other)] = inputs else {
            unreachable!()
        };
        let output = interpret_masked_memory(reference, mask, None, Some(other))?;
        Ok(output.into_iter().map(ArrayIrValue::Array).collect())
    }
}

/// Canonical operation name for [`MaskedStoreOperation`].
pub const MASKED_STORE_OPERATION_NAME: &str = "masked_store";

/// Writes active reference lanes, preserving every inactive lane.
/// Operands are `(reference, value, mask)`; the Boolean mask has the referent's exact shape. The remaining array
/// values have its exact type. Broadcasting and padding must be explicit. The reference itself must be valid: a mask
/// does not authorize an out-of-bounds reference view, and inactive lanes never access storage.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaskedStoreOperation;

impl Display for MaskedStoreOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(MASKED_STORE_OPERATION_NAME)
    }
}

impl Operation for MaskedStoreOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        MASKED_STORE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        validate_masked_types(input_types, region_interfaces, 3, 2)?;
        Ok(vec![])
    }

    fn effects(&self) -> Cow<'_, Effects> {
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Write }],
                vec![],
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl<O: Operation<Type = ArrayIrType>> InterpretableOperation<EagerContext<ArrayIrValue<Array>, O>>
    for MaskedStoreOperation
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, O>>>(
        &self,
        _context: &EagerContext<ArrayIrValue<Array>, O>,
        _driver: &D,
        inputs: &[ArrayIrValue<Array>],
    ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
        self.infer_output_types(&inputs.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        let [ArrayIrValue::Reference(reference), ArrayIrValue::Array(value), ArrayIrValue::Array(mask)] = inputs else {
            unreachable!()
        };
        let output = interpret_masked_memory(reference, mask, Some(value), None)?;
        Ok(output.into_iter().map(ArrayIrValue::Array).collect())
    }
}

/// Canonical operation name for [`MaskedSwapOperation`].
pub const MASKED_SWAP_OPERATION_NAME: &str = "masked_swap";

/// Replaces active reference lanes and returns their old values, using `other` for inactive lanes.
/// Operands are `(reference, value, mask, other)`; the Boolean mask has the referent's exact shape. The remaining
/// array values have its exact type. Broadcasting and padding must be explicit. The reference itself must be valid:
/// a mask does not authorize an out-of-bounds reference view, and inactive lanes never access storage.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaskedSwapOperation;

impl Display for MaskedSwapOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(MASKED_SWAP_OPERATION_NAME)
    }
}

impl Operation for MaskedSwapOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        MASKED_SWAP_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        let referent = validate_masked_types(input_types, region_interfaces, 4, 2)?;
        Ok(vec![referent.into()])
    }

    fn effects(&self) -> Cow<'_, Effects> {
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::ReadWrite }],
                vec![],
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl<O: Operation<Type = ArrayIrType>> InterpretableOperation<EagerContext<ArrayIrValue<Array>, O>>
    for MaskedSwapOperation
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, O>>>(
        &self,
        _context: &EagerContext<ArrayIrValue<Array>, O>,
        _driver: &D,
        inputs: &[ArrayIrValue<Array>],
    ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
        self.infer_output_types(&inputs.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        let [
            ArrayIrValue::Reference(reference),
            ArrayIrValue::Array(value),
            ArrayIrValue::Array(mask),
            ArrayIrValue::Array(other),
        ] = inputs
        else {
            unreachable!()
        };
        let output = interpret_masked_memory(reference, mask, Some(value), Some(other))?;
        Ok(output.into_iter().map(ArrayIrValue::Array).collect())
    }
}

/// Canonical operation name for [`AsyncCopyOperation`].
pub const ASYNC_COPY_OPERATION_NAME: &str = "async_copy";

/// Starts copying the first reference's selected elements into the second reference. Shapes, element types, and
/// sharding must match exactly, while canonical layouts and memory placement may differ. The returned scalar token
/// reference is a scoped completion resource: only [`WaitOperation`] may consume it, in the same region as this
/// operation.
///
/// Until that wait, the source may be read but not changed and the destination may not be accessed. Destination
/// initialization becomes available only after the wait. Qualification rejects overlapping source and destination
/// selections, conflicting pending copies, escaped tokens, and exits with outstanding copies. Target adapters decide
/// which memory placements support asynchronous transfer; the reference interpreter preserves these ordering rules.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AsyncCopyOperation;

impl Display for AsyncCopyOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(ASYNC_COPY_OPERATION_NAME)
    }
}

impl Operation for AsyncCopyOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        ASYNC_COPY_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let source = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
        let destination = <&ReferenceType<ArrayType>>::try_from(&input_types[1])?;
        if source.referent().data_type().is_token() {
            return Err(TypeError::invalid("`async_copy` cannot copy effect tokens"));
        }
        if source.referent().data_type() != destination.referent().data_type()
            || source.referent().shape() != destination.referent().shape()
            || source.referent().sharding() != destination.referent().sharding()
        {
            return Err(TypeError::invalid(format!(
                concat!(
                    "`async_copy` source type `{}` and destination type `{}` ",
                    "must have identical shapes, element types, and sharding",
                ),
                source.referent(),
                destination.referent(),
            )));
        }
        Ok(vec![ReferenceType::new(ArrayType::scalar(DataType::Token)).into()])
    }

    fn effects(&self) -> Cow<'_, Effects> {
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![
                    ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 1, mode: ReferenceAccessMode::Write },
                    ReferenceEffect::Allocate { output_index: 0 },
                ],
                vec![],
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl<C: Domain> InterpretableOperation<C> for AsyncCopyOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Err(ProgramError::custom(KernelMemoryError::RequiresQualification { operation: self.name() }))
    }
}

/// Canonical operation name for [`WaitOperation`].
pub const WAIT_OPERATION_NAME: &str = "wait";

/// Completes an [`AsyncCopyOperation`] and consumes its token reference. The source reservation is released and the
/// destination's copied elements become initialized. The token must originate from an unmatched copy in this same
/// region; a token-shaped reference created through ordinary allocation does not establish a pending copy.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct WaitOperation;

impl Display for WaitOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(WAIT_OPERATION_NAME)
    }
}

impl Operation for WaitOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        WAIT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let expected = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Token)));
        if input_types[0] != expected {
            return Err(TypeError::invalid(format!("`wait` input type `{}` must be `{expected}`", input_types[0],)));
        }
        Ok(vec![])
    }

    fn effects(&self) -> Cow<'_, Effects> {
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Consume }],
                vec![],
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl<C: Domain> InterpretableOperation<C> for WaitOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Err(ProgramError::custom(KernelMemoryError::RequiresQualification { operation: self.name() }))
    }
}

/// Validates exact referent-shaped masks and values without implicit broadcasting or padding.
fn validate_masked_types(
    input_types: &[ArrayIrType],
    region_interfaces: &[RegionInterface<ArrayIrType>],
    input_count: usize,
    mask_index: usize,
) -> Result<ArrayType, TypeError> {
    check_count!("input", input_types, input_count, TypeError);
    check_count!("region", region_interfaces, 0, TypeError);
    let reference = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
    let referent = reference.referent();
    for (index, r#type) in input_types.iter().enumerate().skip(1) {
        let array = <&ArrayType>::try_from(r#type)?;
        if index == mask_index {
            if array.data_type() != DataType::Boolean || array.shape() != referent.shape() {
                return Err(TypeError::invalid(format!(
                    "masked memory mask type `{array}` must be Boolean with referent shape `{}`",
                    referent.shape(),
                )));
            }
        } else if array != referent {
            return Err(TypeError::invalid(format!(
                "masked memory input `{index}` type `{array}` must exactly match referent type `{referent}`",
            )));
        }
    }
    Ok(referent.clone())
}

/// Replays only active lanes through canonical one-element reference views. Each swap retains the reference
/// transaction's old-value semantics; inactive lanes neither read nor modify reference storage. A replacement
/// enables writes; an `other` value requests a result and supplies its inactive lanes. The caller has checked all
/// input types and supplies at least one of these values, distinguishing load, store, and swap without a new mode.
fn interpret_masked_memory(
    reference: &ArrayReference<Array>,
    mask: &Array,
    replacement: Option<&Array>,
    other: Option<&Array>,
) -> Result<Option<Array>, ProgramError> {
    let referent = reference.r#type().referent().clone();
    let addressing = ArrayAddressing::new(referent.clone())?;
    let masks = mask.elements::<bool>()?;
    let replacement = replacement.map(Array::logical_bytes);
    let mut output = other.map(Array::logical_bytes);
    let mut index = vec![0; referent.rank()];
    let width = addressing.element_byte_width();
    for (element, active) in masks.into_iter().enumerate() {
        if active {
            let view = reference.with_transform(ArrayReferenceView::Slice {
                axes: index.iter().map(|&start| ArraySliceAxis::new(start, 1, 1)).collect(),
            })?;
            let bytes = element * width..(element + 1) * width;
            let previous = if let Some(replacement) = &replacement {
                let value = Array::from_logical_bytes(view.r#type().referent().clone(), &replacement[bytes.clone()])?;
                if output.is_some() {
                    Some(view.swap(value)?)
                } else {
                    view.write(value)?;
                    None
                }
            } else {
                Some(view.read()?)
            };
            if let Some(output) = &mut output {
                output[bytes].copy_from_slice(&previous.unwrap().logical_bytes());
            }
        }
        addressing.advance_index(&mut index);
    }
    output.map(|bytes| Array::from_logical_bytes(referent, &bytes)).transpose()
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::Context;
    use crate::kernels::operations::KernelOperation;

    use crate::arrays::{DataType, Dimension, DimensionBounds, DimensionVariable, Shape};

    use super::*;

    #[test]
    fn test_scratch_operation_new() {
        let referent = ArrayType::new_static(DataType::I32, vec![2, 3]);
        assert_eq!(ScratchOperation::new(referent.clone(), 16), Ok(ScratchOperation { referent, alignment: 16 }),);
        let empty = ArrayType::new_static(DataType::I32, vec![0]);
        assert_eq!(ScratchOperation::new(empty.clone(), 1), Ok(ScratchOperation { referent: empty, alignment: 1 }));
    }

    #[test]
    fn test_scratch_operation_new_rejects_invalid_alignment() {
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        assert_eq!(
            ScratchOperation::new(referent.clone(), 0),
            Err(KernelMemoryError::InvalidAlignment { alignment: 0 })
        );
        assert_eq!(ScratchOperation::new(referent, 3), Err(KernelMemoryError::InvalidAlignment { alignment: 3 }));
    }

    #[test]
    fn test_scratch_operation_new_rejects_dynamic_storage() {
        let dimension =
            Dimension::Dynamic(DimensionVariable::new("size", DimensionBounds::non_negative(None).unwrap()));
        let referent = ArrayType::new_static(DataType::I32, vec![2]).with_shape(Shape::new(vec![dimension]));
        let expected = ArrayAddressing::new(referent.clone()).unwrap_err();
        assert_eq!(ScratchOperation::new(referent, 4), Err(KernelMemoryError::Program(expected)));
    }

    #[test]
    fn test_scratch_operation_referent() {
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        assert_eq!(ScratchOperation::new(referent.clone(), 4).unwrap().referent(), &referent);
    }

    #[test]
    fn test_scratch_operation_alignment() {
        let operation = ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 16).unwrap();
        assert_eq!(operation.alignment(), 16);
    }

    #[test]
    fn test_scratch_operation_name() {
        let operation = ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 4).unwrap();
        assert_eq!(operation.name(), SCRATCH_OPERATION_NAME);
    }

    #[test]
    fn test_scratch_operation_infer_output_types() {
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        let operation = ScratchOperation::new(referent.clone(), 4).unwrap();
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![ReferenceType::new(referent).into()]));
    }

    #[test]
    fn test_scratch_operation_infer_output_types_rejects_nonzero_arity() {
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        let operation = ScratchOperation::new(referent.clone(), 4).unwrap();
        assert_eq!(
            operation.infer_output_types(&[referent.into()], &[]),
            Err(TypeError::invalid("expected 0 inputs but got 1")),
        );
        assert_eq!(
            operation.infer_output_types(&[], &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)]),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_scratch_operation_rename_type_identities() {
        let operation = ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 4).unwrap();
        assert_eq!(operation.rename_type_identities(&TypeIdentityRenaming::new()), Ok(operation));
    }

    #[test]
    fn test_scratch_operation_render() {
        let operation = ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 16).unwrap();
        assert_eq!(operation.to_string(), "scratch [alignment=16]");
    }

    #[test]
    fn test_scratch_operation_effects() {
        let operation = ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 4).unwrap();
        assert_eq!(
            operation.effects().as_ref(),
            &Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Allocate { output_index: 0 }], Vec::new())
                .unwrap(),
        );
    }
    #[test]
    fn test_masked_load_operation_infer_output_types() {
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        let mask = ArrayType::new_static(DataType::Boolean, vec![2]);
        assert_eq!(
            MaskedLoadOperation.infer_output_types(
                &[ReferenceType::new(referent.clone()).into(), mask.into(), referent.clone().into()],
                &[],
            ),
            Ok(vec![referent.into()]),
        );
    }

    #[test]
    fn test_masked_load_operation_effects() {
        assert_eq!(
            MaskedLoadOperation.effects().as_ref(),
            &Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
                vec![],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_masked_load_operation_interpret() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = ArrayReference::new(Array::vector(vec![10i32, 20]).unwrap());
        let mask = ArrayIrValue::Array(Array::vector(vec![true, false]).unwrap());
        let other = ArrayIrValue::Array(Array::vector(vec![-1i32, -2]).unwrap());
        assert_eq!(
            context.bind(MaskedLoadOperation, vec![], &[ArrayIrValue::Reference(reference), mask, other,]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10i32, -2]).unwrap())])
        );
    }

    #[test]
    fn test_masked_load_operation_interpret_does_not_access_inactive_lanes() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = ArrayReference::new(Array::vector(vec![10i32, 20]).unwrap());
        reference.freeze().unwrap();
        let other = ArrayIrValue::Array(Array::vector(vec![-1i32, -2]).unwrap());
        assert_eq!(
            context.bind(
                MaskedLoadOperation,
                vec![],
                &[
                    ArrayIrValue::Reference(reference.clone()),
                    ArrayIrValue::Array(Array::vector(vec![false, false]).unwrap()),
                    other.clone(),
                ]
            ),
            Ok(vec![other])
        );

        let expected = reference.read().unwrap_err();
        assert_eq!(
            context.bind(
                MaskedLoadOperation,
                vec![],
                &[
                    ArrayIrValue::Reference(reference),
                    ArrayIrValue::Array(Array::vector(vec![true, false]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![-1i32, -2]).unwrap()),
                ]
            ),
            Err(expected),
        );
    }

    #[test]
    fn test_masked_store_operation_infer_output_types() {
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        let mask = ArrayType::new_static(DataType::Boolean, vec![2]);
        assert_eq!(
            MaskedStoreOperation
                .infer_output_types(&[ReferenceType::new(referent.clone()).into(), referent.into(), mask.into(),], &[]),
            Ok(vec![])
        );
    }

    #[test]
    fn test_masked_store_operation_effects() {
        assert_eq!(
            MaskedStoreOperation.effects().as_ref(),
            &Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Write }],
                vec![],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_masked_store_operation_interpret() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let root = ArrayReference::new(Array::vector(vec![1i32, 2, 3]).unwrap());
        let view = root.with_transform(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] }).unwrap();
        assert_eq!(
            context.bind(
                MaskedStoreOperation,
                vec![],
                &[
                    ArrayIrValue::Reference(view),
                    ArrayIrValue::Array(Array::vector(vec![9i32, 8]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![false, true]).unwrap()),
                ]
            ),
            Ok(vec![])
        );
        assert_eq!(root.read(), Ok(Array::vector(vec![1i32, 2, 8]).unwrap()));
    }

    #[test]
    fn test_masked_swap_operation_infer_output_types() {
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        let mask = ArrayType::new_static(DataType::Boolean, vec![2]);
        assert_eq!(
            MaskedSwapOperation.infer_output_types(
                &[
                    ReferenceType::new(referent.clone()).into(),
                    referent.clone().into(),
                    mask.into(),
                    referent.clone().into(),
                ],
                &[]
            ),
            Ok(vec![referent.into()])
        );
    }

    #[test]
    fn test_masked_swap_operation_effects() {
        assert_eq!(
            MaskedSwapOperation.effects().as_ref(),
            &Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::ReadWrite }],
                vec![],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_masked_swap_operation_interpret() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = ArrayReference::new(Array::vector(vec![10i32, 20]).unwrap());
        assert_eq!(
            context.bind(
                MaskedSwapOperation,
                vec![],
                &[
                    ArrayIrValue::Reference(reference.clone()),
                    ArrayIrValue::Array(Array::vector(vec![3i32, 4]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![true, false]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![-1i32, -2]).unwrap()),
                ]
            ),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10i32, -2]).unwrap())])
        );
        assert_eq!(reference.read(), Ok(Array::vector(vec![3i32, 20]).unwrap()));
    }

    #[test]
    fn test_async_copy_operation_infer_output_types() {
        let reference = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, vec![2])));
        let token = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Token)));
        assert_eq!(
            AsyncCopyOperation.infer_output_types(&[reference.clone(), reference.clone()], &[]),
            Ok(vec![token.clone()])
        );
        assert_eq!(
            AsyncCopyOperation.infer_output_types(&[token.clone(), token], &[]),
            Err(TypeError::invalid("`async_copy` cannot copy effect tokens")),
        );
        let different = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, vec![3])));
        assert_eq!(
            AsyncCopyOperation.infer_output_types(&[reference, different], &[]),
            Err(TypeError::invalid(concat!(
                "`async_copy` source type `i32[2]` and destination type `i32[3]` ",
                "must have identical shapes, element types, and sharding",
            ),)),
        );
    }

    #[test]
    fn test_async_copy_operation_effects() {
        assert_eq!(
            AsyncCopyOperation.effects().as_ref(),
            &Effects::new(
                EffectClasses::NONE,
                vec![
                    ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 1, mode: ReferenceAccessMode::Write },
                    ReferenceEffect::Allocate { output_index: 0 },
                ],
                vec![],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_async_copy_operation_interpret_requires_qualification() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let source = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1i32).unwrap()));
        let destination = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0i32).unwrap()));
        assert_eq!(
            context.bind(AsyncCopyOperation, vec![], &[source, destination]),
            Err(ProgramError::custom(KernelMemoryError::RequiresQualification { operation: "async_copy" })),
        );
    }

    #[test]
    fn test_wait_operation_infer_output_types() {
        let token = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Token)));
        assert_eq!(WaitOperation.infer_output_types(&[token.clone()], &[]), Ok(vec![]));
        let invalid = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::I32)));
        assert_eq!(
            WaitOperation.infer_output_types(&[invalid.clone()], &[]),
            Err(TypeError::invalid(format!("`wait` input type `{invalid}` must be `{token}`"))),
        );
    }

    #[test]
    fn test_wait_operation_effects() {
        assert_eq!(
            WaitOperation.effects().as_ref(),
            &Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Consume }],
                vec![],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_wait_operation_interpret_requires_qualification() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let token = ArrayIrValue::Reference(ArrayReference::new(
            Array::new(ArrayType::scalar(DataType::Token), vec![]).unwrap(),
        ));
        assert_eq!(
            context.bind(WaitOperation, vec![], &[token]),
            Err(ProgramError::custom(KernelMemoryError::RequiresQualification { operation: "wait" })),
        );
    }

    #[test]
    fn test_validate_masked_types() {
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        let mask = ArrayType::new_static(DataType::Boolean, vec![2]);
        assert_eq!(
            validate_masked_types(
                &[ReferenceType::new(referent.clone()).into(), mask.into(), referent.clone().into(),],
                &[],
                3,
                1
            ),
            Ok(referent)
        );
    }

    #[test]
    fn test_validate_masked_types_rejects_arity_shape_and_value_types() {
        assert_eq!(validate_masked_types(&[], &[], 3, 1), Err(TypeError::invalid("expected 3 inputs but got 0")));
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        let mask = ArrayType::new_static(DataType::Boolean, vec![1]);
        assert_eq!(
            validate_masked_types(
                &[ReferenceType::new(referent.clone()).into(), mask.clone().into(), referent.clone().into(),],
                &[],
                3,
                1
            ),
            Err(TypeError::invalid(format!(
                "masked memory mask type `{mask}` must be Boolean with referent shape `{}`",
                referent.shape(),
            )))
        );
        let value = ArrayType::new_static(DataType::F32, vec![2]);
        let mask = ArrayType::new_static(DataType::Boolean, vec![2]);
        assert_eq!(
            validate_masked_types(
                &[ReferenceType::new(referent.clone()).into(), value.clone().into(), mask.into(),],
                &[],
                3,
                2
            ),
            Err(TypeError::invalid(format!(
                "masked memory input `1` type `{value}` must exactly match referent type `{referent}`",
            )))
        );
    }

    #[test]
    fn test_interpret_masked_memory() {
        let reference = ArrayReference::new(Array::scalar(9i32).unwrap());
        let other = Array::scalar(-1i32).unwrap();
        assert_eq!(
            interpret_masked_memory(&reference, &Array::scalar(true).unwrap(), None, Some(&other)),
            Ok(Some(Array::scalar(9i32).unwrap()))
        );
        let empty = ArrayReference::new(Array::vector(Vec::<i32>::new()).unwrap());
        let other = Array::vector(Vec::<i32>::new()).unwrap();
        assert_eq!(
            interpret_masked_memory(&empty, &Array::vector(Vec::<bool>::new()).unwrap(), None, Some(&other)),
            Ok(Some(other))
        );
    }
}
