//! Scoped scratch allocation and masked memory access using canonical array types, views, and reference lifetimes.

use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

use thiserror::Error;

use crate::arrays::{
    Array, ArrayAddressing, ArrayIrType, ArrayIrValue, ArrayReference, ArrayReferenceTransform, ArraySliceAxis, ArrayType,
    DataType, DimensionVariable,
};
use crate::contexts::{Domain, EagerContext};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::references::render_reference_access;
use crate::programs::{
    EffectClasses, Effects, Operation, OperationFormatter, ProgramError, ReferenceAccessDescriptor,
    ReferenceAccessMode, ReferenceAccessOperation, ReferenceEffect, ReferenceType, ReferenceTransform, RegionInterface,
    Type, TypeError, TypeIdentityRenaming, Typed, infer_reference_view_type,
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
            Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Allocate { output_index: 0 }]).unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

/// Canonical operation name for [`MaskedLoadOperation`].
pub const MASKED_LOAD_OPERATION_NAME: &str = "masked_load";

/// Loads active reference lanes and returns `other` for inactive lanes.
/// Base inputs are `(reference, mask, other)`, followed by the view bindings. The Boolean mask has the selected
/// referent's exact shape; the remaining array values have its exact type. Broadcasting and padding must be explicit.
/// The reference itself must be valid: a mask does not authorize an out-of-bounds reference view, and inactive lanes
/// never access storage.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MaskedLoadOperation {
    /// Refer to the documentation of [`Self::views`].
    transforms: Vec<ArrayReferenceTransform>,
}

impl MaskedLoadOperation {
    /// Creates a whole-reference access.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the views applied to the reference input.
    pub fn transforms(&self) -> &[ArrayReferenceTransform] {
        &self.transforms
    }

    /// Returns a copy with the provided views applied before the access.
    pub fn with_transforms(mut self, transforms: Vec<ArrayReferenceTransform>) -> Self {
        self.transforms = transforms;
        self
    }
}

impl Display for MaskedLoadOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
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
        let referent =
            validate_masked_types(input_types, region_interfaces, 3, 1, &self.transforms, ReferenceAccessMode::Read)?;
        Ok(vec![referent.into()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        render_reference_access(formatter, indentation, self.name(), &self.transforms)
    }

    fn effects(&self) -> Cow<'_, Effects> {
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl ReferenceAccessOperation for MaskedLoadOperation {
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        3
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
        (input_index == 0).then(|| {
            ReferenceAccessDescriptor::new(
                &self.transforms,
                3..3 + self.transforms.iter().map(|view| view.binding_count()).sum::<usize>(),
            )
        })
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<ArrayReferenceTransform>,
    ) -> Result<Self, ProgramError> {
        if input_index != 0 {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{}` has no reference access at input {input_index}",
                self.name(),
            )));
        }
        Ok(self.clone().with_transforms(transforms))
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
        let [ArrayIrValue::Reference(reference), ArrayIrValue::Array(mask), ArrayIrValue::Array(other)] = &inputs[..3]
        else {
            unreachable!()
        };
        let reference = reference.with_transforms(&self.transforms, &inputs[3..])?;
        let output = interpret_masked_memory(&reference, mask, None, Some(other))?;
        Ok(output.into_iter().map(ArrayIrValue::Array).collect())
    }
}

/// Canonical operation name for [`MaskedStoreOperation`].
pub const MASKED_STORE_OPERATION_NAME: &str = "masked_store";

/// Writes active reference lanes, preserving every inactive lane.
/// Base inputs are `(reference, value, mask)`, followed by the view bindings. The Boolean mask has the selected
/// referent's exact shape; the remaining array values have its exact type. Broadcasting and padding must be explicit.
/// The reference itself must be valid: a mask does not authorize an out-of-bounds reference view, and inactive lanes
/// never access storage.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MaskedStoreOperation {
    /// Refer to the documentation of [`Self::views`].
    transforms: Vec<ArrayReferenceTransform>,
}

impl MaskedStoreOperation {
    /// Creates a whole-reference access.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the views applied to the reference input.
    pub fn transforms(&self) -> &[ArrayReferenceTransform] {
        &self.transforms
    }

    /// Returns a copy with the provided views applied before the access.
    pub fn with_transforms(mut self, transforms: Vec<ArrayReferenceTransform>) -> Self {
        self.transforms = transforms;
        self
    }
}

impl Display for MaskedStoreOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
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
        validate_masked_types(input_types, region_interfaces, 3, 2, &self.transforms, ReferenceAccessMode::Write)?;
        Ok(vec![])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        render_reference_access(formatter, indentation, self.name(), &self.transforms)
    }

    fn effects(&self) -> Cow<'_, Effects> {
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Write }],
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl ReferenceAccessOperation for MaskedStoreOperation {
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        3
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
        (input_index == 0).then(|| {
            ReferenceAccessDescriptor::new(
                &self.transforms,
                3..3 + self.transforms.iter().map(|view| view.binding_count()).sum::<usize>(),
            )
        })
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<ArrayReferenceTransform>,
    ) -> Result<Self, ProgramError> {
        if input_index != 0 {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{}` has no reference access at input {input_index}",
                self.name(),
            )));
        }
        Ok(self.clone().with_transforms(transforms))
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
        let [ArrayIrValue::Reference(reference), ArrayIrValue::Array(value), ArrayIrValue::Array(mask)] = &inputs[..3]
        else {
            unreachable!()
        };
        let reference = reference.with_transforms(&self.transforms, &inputs[3..])?;
        let output = interpret_masked_memory(&reference, mask, Some(value), None)?;
        Ok(output.into_iter().map(ArrayIrValue::Array).collect())
    }
}

/// Canonical operation name for [`MaskedSwapOperation`].
pub const MASKED_SWAP_OPERATION_NAME: &str = "masked_swap";

/// Replaces active reference lanes and returns their old values, using `other` for inactive lanes.
/// Base inputs are `(reference, value, mask, other)`, followed by the view bindings. The Boolean mask has the
/// selected referent's exact shape; the remaining array values have its exact type. Broadcasting and padding must be
/// explicit. The reference itself must be valid: a mask does not authorize an out-of-bounds reference view, and
/// inactive lanes never access storage.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MaskedSwapOperation {
    /// Refer to the documentation of [`Self::views`].
    transforms: Vec<ArrayReferenceTransform>,
}

impl MaskedSwapOperation {
    /// Creates a whole-reference access.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the views applied to the reference input.
    pub fn transforms(&self) -> &[ArrayReferenceTransform] {
        &self.transforms
    }

    /// Returns a copy with the provided views applied before the access.
    pub fn with_transforms(mut self, transforms: Vec<ArrayReferenceTransform>) -> Self {
        self.transforms = transforms;
        self
    }
}

impl Display for MaskedSwapOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
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
        let referent =
            validate_masked_types(input_types, region_interfaces, 4, 2, &self.transforms, ReferenceAccessMode::ReadWrite)?;
        Ok(vec![referent.into()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        render_reference_access(formatter, indentation, self.name(), &self.transforms)
    }

    fn effects(&self) -> Cow<'_, Effects> {
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::ReadWrite }],
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl ReferenceAccessOperation for MaskedSwapOperation {
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        4
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
        (input_index == 0).then(|| {
            ReferenceAccessDescriptor::new(
                &self.transforms,
                4..4 + self.transforms.iter().map(|view| view.binding_count()).sum::<usize>(),
            )
        })
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<ArrayReferenceTransform>,
    ) -> Result<Self, ProgramError> {
        if input_index != 0 {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{}` has no reference access at input {input_index}",
                self.name(),
            )));
        }
        Ok(self.clone().with_transforms(transforms))
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
        ] = &inputs[..4]
        else {
            unreachable!()
        };
        let reference = reference.with_transforms(&self.transforms, &inputs[4..])?;
        let output = interpret_masked_memory(&reference, mask, Some(value), Some(other))?;
        Ok(output.into_iter().map(ArrayIrValue::Array).collect())
    }
}

/// Canonical operation name for [`AsyncCopyOperation`].
pub const ASYNC_COPY_OPERATION_NAME: &str = "async_copy";

/// Starts copying the first reference's selected elements into the second reference. Shapes, element types, and
/// sharding must match exactly, while canonical layouts and memory placement may differ. The returned scalar token
/// reference is a scoped completion resource: only [`WaitOperation`] may consume it, in the same region as this
/// operation. Source view bindings trail the two references, followed by destination view bindings.
///
/// Until that wait, the source may be read but not changed and the destination may not be accessed. Destination
/// initialization becomes available only after the wait. Qualification rejects overlapping source and destination
/// selections, conflicting pending copies, escaped tokens, and exits with outstanding copies. Target adapters decide
/// which memory placements support asynchronous transfer; the reference interpreter preserves these ordering rules.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AsyncCopyOperation {
    /// Ordered selections applied to the source reference.
    source_transforms: Vec<ArrayReferenceTransform>,

    /// Ordered selections applied to the destination reference.
    destination_transforms: Vec<ArrayReferenceTransform>,
}

impl AsyncCopyOperation {
    /// Creates a copy between whole references.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the views applied to the source reference.
    pub fn source_transforms(&self) -> &[ArrayReferenceTransform] {
        &self.source_transforms
    }

    /// Returns the views applied to the destination reference.
    pub fn destination_transforms(&self) -> &[ArrayReferenceTransform] {
        &self.destination_transforms
    }

    /// Returns a copy with the provided views applied to the source reference.
    pub fn with_source_transforms(mut self, transforms: Vec<ArrayReferenceTransform>) -> Self {
        self.source_transforms = transforms;
        self
    }

    /// Returns a copy with the provided views applied to the destination reference.
    pub fn with_destination_transforms(mut self, transforms: Vec<ArrayReferenceTransform>) -> Self {
        self.destination_transforms = transforms;
        self
    }
}

impl Display for AsyncCopyOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
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
        let source_count = self.source_transforms.iter().map(|view| view.binding_count()).sum::<usize>();
        let destination_count = self.destination_transforms.iter().map(|view| view.binding_count()).sum::<usize>();
        check_count!("input", input_types, 2 + source_count + destination_count, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let source = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
        let destination = <&ReferenceType<ArrayType>>::try_from(&input_types[1])?;
        let source = infer_reference_view_type(
            source.referent(),
            &self.source_transforms,
            &input_types[2..2 + source_count].iter().collect::<Vec<_>>(),
            ReferenceAccessMode::Read,
        )?;
        let destination = infer_reference_view_type(
            destination.referent(),
            &self.destination_transforms,
            &input_types[2 + source_count..].iter().collect::<Vec<_>>(),
            ReferenceAccessMode::Write,
        )?;
        if source.data_type().is_token() {
            return Err(TypeError::invalid("`async_copy` cannot copy effect tokens"));
        }
        if source.data_type() != destination.data_type()
            || source.shape() != destination.shape()
            || source.sharding() != destination.sharding()
        {
            return Err(TypeError::invalid(format!(
                concat!(
                    "`async_copy` source type `{}` and destination type `{}` ",
                    "must have identical shapes, element types, and sharding",
                ),
                source, destination,
            )));
        }
        Ok(vec![ReferenceType::new(ArrayType::scalar(DataType::Token)).into()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // Like the portable reference accesses, empty paths are omitted so that a whole-reference copy renders as
        // just its name.
        let operation = OperationFormatter::new(formatter, indentation, self.name())?;
        if self.source_transforms.is_empty() && self.destination_transforms.is_empty() {
            return Ok(());
        }
        operation.bracketed(|operation| {
            if !self.source_transforms.is_empty() {
                operation.list("source_transforms", &self.source_transforms)?;
            }
            if !self.destination_transforms.is_empty() {
                operation.list("destination_transforms", &self.destination_transforms)?;
            }
            Ok(())
        })
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
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl ReferenceAccessOperation for AsyncCopyOperation {
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        2
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
        let source_end = 2 + self.source_transforms.iter().map(|view| view.binding_count()).sum::<usize>();
        match input_index {
            0 => Some(ReferenceAccessDescriptor::new(&self.source_transforms, 2..source_end)),
            1 => Some(ReferenceAccessDescriptor::new(
                &self.destination_transforms,
                source_end..source_end + self.destination_transforms.iter().map(|view| view.binding_count()).sum::<usize>(),
            )),
            _ => None,
        }
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<ArrayReferenceTransform>,
    ) -> Result<Self, ProgramError> {
        match input_index {
            0 => Ok(self.clone().with_source_transforms(transforms)),
            1 => Ok(self.clone().with_destination_transforms(transforms)),
            _ => Err(ProgramError::MalformedProgram(format!(
                "operation `{}` has no reference access at input {input_index}",
                self.name(),
            ))),
        }
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
    transforms: &[ArrayReferenceTransform],
    mode: ReferenceAccessMode,
) -> Result<ArrayType, TypeError> {
    let binding_count = transforms.iter().map(|view| view.binding_count()).sum::<usize>();
    check_count!("input", input_types, input_count + binding_count, TypeError);
    check_count!("region", region_interfaces, 0, TypeError);
    let reference = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
    let referent = infer_reference_view_type(
        reference.referent(),
        transforms,
        &input_types[input_count..].iter().collect::<Vec<_>>(),
        mode,
    )?;
    for (index, r#type) in input_types[..input_count].iter().enumerate().skip(1) {
        let array = <&ArrayType>::try_from(r#type)?;
        if index == mask_index {
            if array.data_type() != DataType::Boolean || array.shape() != referent.shape() {
                return Err(TypeError::invalid(format!(
                    "masked memory mask type `{array}` must be Boolean with referent shape `{}`",
                    referent.shape(),
                )));
            }
        } else if array != &referent {
            return Err(TypeError::invalid(format!(
                "masked memory input `{index}` type `{array}` must exactly match referent type `{referent}`",
            )));
        }
    }
    Ok(referent)
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
            let view = reference.with_transform(ArrayReferenceTransform::Slice {
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
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayReferenceTransformIndex, DataType, Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::contexts::Context;
    use crate::kernels::operations::KernelOperation;

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
            &Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Allocate { output_index: 0 }],).unwrap(),
        );
    }

    #[test]
    fn test_masked_load_operation_new() {
        assert_eq!(MaskedLoadOperation::new().transforms(), &[]);
    }

    #[test]
    fn test_masked_load_operation_with_transforms() {
        let transforms = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let operation = MaskedLoadOperation::new().with_transforms(transforms.clone());
        assert_eq!(operation.transforms(), transforms);
        assert_eq!(operation.reference_access_descriptor(0).unwrap().bindings(), 3..4);
        assert!(operation.reference_access_descriptor(1).is_none());
        assert_eq!(operation.with_reference_access_transforms(0, vec![]), Ok(MaskedLoadOperation::new()));
    }

    #[test]
    fn test_masked_load_operation_infer_output_types() {
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        let mask = ArrayType::new_static(DataType::Boolean, vec![2]);
        assert_eq!(
            MaskedLoadOperation::new().infer_output_types(
                &[ReferenceType::new(referent.clone()).into(), mask.into(), referent.clone().into()],
                &[],
            ),
            Ok(vec![referent.into()]),
        );
    }

    #[test]
    fn test_masked_load_operation_render() {
        assert_eq!(MaskedLoadOperation::new().to_string(), "masked_load");
        let operation = MaskedLoadOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] },
        ]);
        assert_eq!(operation.to_string(), "masked_load [transforms=[dynamic_index(axis=0), slice(axes=[1:3])]]");
    }

    #[test]
    fn test_masked_load_operation_effects() {
        assert_eq!(
            MaskedLoadOperation::new().effects().as_ref(),
            &Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
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
            context.bind(MaskedLoadOperation::new(), vec![], &[ArrayIrValue::Reference(reference), mask, other,]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10i32, -2]).unwrap())])
        );
    }

    #[test]
    fn test_masked_load_operation_interpret_transforms() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = ArrayReference::new(Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6]).unwrap());
        let operation = MaskedLoadOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] },
        ]);
        assert_eq!(
            context.bind(
                operation,
                vec![],
                &[
                    ArrayIrValue::Reference(reference),
                    ArrayIrValue::Array(Array::vector(vec![true, false]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![-1i32, -2]).unwrap()),
                    ArrayIrValue::Array(Array::scalar(-1i64).unwrap()),
                ]
            ),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![5i32, -2]).unwrap())]),
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
                MaskedLoadOperation::new(),
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
                MaskedLoadOperation::new(),
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
    fn test_masked_store_operation_new() {
        assert_eq!(MaskedStoreOperation::new().transforms(), &[]);
    }

    #[test]
    fn test_masked_store_operation_with_transforms() {
        let transforms = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let operation = MaskedStoreOperation::new().with_transforms(transforms.clone());
        assert_eq!(operation.transforms(), transforms);
        assert_eq!(operation.reference_access_descriptor(0).unwrap().bindings(), 3..4);
        assert!(operation.reference_access_descriptor(1).is_none());
        assert_eq!(operation.with_reference_access_transforms(0, vec![]), Ok(MaskedStoreOperation::new()));
    }

    #[test]
    fn test_masked_store_operation_infer_output_types() {
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        let mask = ArrayType::new_static(DataType::Boolean, vec![2]);
        assert_eq!(
            MaskedStoreOperation::new()
                .infer_output_types(&[ReferenceType::new(referent.clone()).into(), referent.into(), mask.into(),], &[]),
            Ok(vec![])
        );
    }

    #[test]
    fn test_masked_store_operation_render() {
        assert_eq!(MaskedStoreOperation::new().to_string(), "masked_store");
        let operation = MaskedStoreOperation::new()
            .with_transforms(vec![ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(2) }]);
        assert_eq!(operation.to_string(), "masked_store [transforms=[index(axis=1, index=2)]]");
    }

    #[test]
    fn test_masked_store_operation_effects() {
        assert_eq!(
            MaskedStoreOperation::new().effects().as_ref(),
            &Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Write }],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_masked_store_operation_interpret() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let root = ArrayReference::new(Array::vector(vec![1i32, 2, 3]).unwrap());
        let operation = MaskedStoreOperation::new()
            .with_transforms(vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] }]);
        assert_eq!(
            context.bind(
                operation,
                vec![],
                &[
                    ArrayIrValue::Reference(root.clone()),
                    ArrayIrValue::Array(Array::vector(vec![9i32, 8]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![false, true]).unwrap()),
                ]
            ),
            Ok(vec![])
        );
        assert_eq!(root.read(), Ok(Array::vector(vec![1i32, 2, 8]).unwrap()));
    }

    #[test]
    fn test_masked_swap_operation_new() {
        assert_eq!(MaskedSwapOperation::new().transforms(), &[]);
    }

    #[test]
    fn test_masked_swap_operation_with_transforms() {
        let transforms = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let operation = MaskedSwapOperation::new().with_transforms(transforms.clone());
        assert_eq!(operation.transforms(), transforms);
        assert_eq!(operation.reference_access_descriptor(0).unwrap().bindings(), 4..5);
        assert!(operation.reference_access_descriptor(1).is_none());
        assert_eq!(operation.with_reference_access_transforms(0, vec![]), Ok(MaskedSwapOperation::new()));
    }

    #[test]
    fn test_masked_swap_operation_infer_output_types() {
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        let mask = ArrayType::new_static(DataType::Boolean, vec![2]);
        assert_eq!(
            MaskedSwapOperation::new().infer_output_types(
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
    fn test_masked_swap_operation_render() {
        assert_eq!(MaskedSwapOperation::new().to_string(), "masked_swap");
        let operation = MaskedSwapOperation::new()
            .with_transforms(vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }]);
        assert_eq!(operation.to_string(), "masked_swap [transforms=[dynamic_index(axis=0)]]");
    }

    #[test]
    fn test_masked_swap_operation_effects() {
        assert_eq!(
            MaskedSwapOperation::new().effects().as_ref(),
            &Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::ReadWrite }],
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
                MaskedSwapOperation::new(),
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
    fn test_masked_swap_operation_interpret_transforms() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = ArrayReference::new(Array::matrix(2, 2, vec![1i32, 2, 3, 4]).unwrap());
        let operation = MaskedSwapOperation::new()
            .with_transforms(vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }]);
        assert_eq!(
            context.bind(
                operation,
                vec![],
                &[
                    ArrayIrValue::Reference(reference.clone()),
                    ArrayIrValue::Array(Array::vector(vec![8i32, 9]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![false, true]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![-1i32, -2]).unwrap()),
                    ArrayIrValue::Array(Array::scalar(1i64).unwrap()),
                ]
            ),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![-1i32, 4]).unwrap())]),
        );
        assert_eq!(reference.read(), Ok(Array::matrix(2, 2, vec![1i32, 2, 3, 9]).unwrap()));
    }

    #[test]
    fn test_async_copy_operation_new() {
        let operation = AsyncCopyOperation::new();
        assert_eq!(operation.source_transforms(), &[]);
        assert_eq!(operation.destination_transforms(), &[]);
    }

    #[test]
    fn test_async_copy_operation_with_source_transforms() {
        let transforms = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let operation = AsyncCopyOperation::new().with_source_transforms(transforms.clone());
        assert_eq!(operation.source_transforms(), transforms);
        assert_eq!(operation.reference_access_descriptor(0).unwrap().bindings(), 2..3);
        assert_eq!(operation.reference_access_descriptor(1).unwrap().bindings(), 3..3);
    }

    #[test]
    fn test_async_copy_operation_with_destination_transforms() {
        let transforms = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let operation =
            AsyncCopyOperation::new().with_source_transforms(transforms.clone()).with_destination_transforms(transforms.clone());
        assert_eq!(operation.destination_transforms(), transforms);
        assert_eq!(operation.reference_access_descriptor(1).unwrap().bindings(), 3..4);
        let replaced = operation.with_reference_access_transforms(0, vec![]).unwrap();
        assert_eq!(replaced.reference_access_descriptor(1).unwrap().bindings(), 2..3);
    }

    #[test]
    fn test_async_copy_operation_infer_output_types() {
        let reference = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, vec![2])));
        let token = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Token)));
        assert_eq!(
            AsyncCopyOperation::new().infer_output_types(&[reference.clone(), reference.clone()], &[]),
            Ok(vec![token.clone()])
        );
        assert_eq!(
            AsyncCopyOperation::new().infer_output_types(&[token.clone(), token], &[]),
            Err(TypeError::invalid("`async_copy` cannot copy effect tokens")),
        );
        let different = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, vec![3])));
        assert_eq!(
            AsyncCopyOperation::new().infer_output_types(&[reference, different], &[]),
            Err(TypeError::invalid(concat!(
                "`async_copy` source type `i32[2]` and destination type `i32[3]` ",
                "must have identical shapes, element types, and sharding",
            ),)),
        );
    }

    #[test]
    fn test_async_copy_operation_infer_output_types_transforms() {
        let operation = AsyncCopyOperation::new()
            .with_source_transforms(vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }])
            .with_destination_transforms(vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 3, 1)] }]);
        let inputs = vec![
            ReferenceType::new(ArrayType::new_static(DataType::I32, [2, 3])).into(),
            ReferenceType::new(ArrayType::new_static(DataType::I32, [5])).into(),
            ArrayIrType::Array(ArrayType::scalar(DataType::I64)),
        ];
        assert_eq!(
            operation.infer_output_types(&inputs, &[]),
            Ok(vec![ReferenceType::new(ArrayType::scalar(DataType::Token)).into(),])
        );
        assert_eq!(
            operation.infer_output_types(&inputs[..2], &[]),
            Err(TypeError::invalid("expected 3 inputs but got 2"))
        );
    }

    #[test]
    fn test_async_copy_operation_render() {
        assert_eq!(AsyncCopyOperation::new().to_string(), "async_copy");

        // Empty paths are omitted, so a copy with only one viewed input renders only that input's path.
        let source = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let destination = vec![
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 3, 1)] },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(2) },
        ];
        assert_eq!(
            AsyncCopyOperation::new().with_source_transforms(source.clone()).to_string(),
            "async_copy [source_transforms=[dynamic_index(axis=0)]]",
        );
        assert_eq!(
            AsyncCopyOperation::new().with_destination_transforms(destination.clone()).to_string(),
            "async_copy [destination_transforms=[slice(axes=[1:4]), index(axis=0, index=2)]]",
        );

        // Long metadata wraps one field per line, indented relative to the owning instruction.
        assert_eq!(
            AsyncCopyOperation::new().with_source_transforms(source).with_destination_transforms(destination).to_string(),
            indoc! {"
                async_copy [
                    source_transforms=[dynamic_index(axis=0)],
                    destination_transforms=[slice(axes=[1:4]), index(axis=0, index=2)],
                ]"},
        );
    }

    #[test]
    fn test_async_copy_operation_effects() {
        assert_eq!(
            AsyncCopyOperation::new().effects().as_ref(),
            &Effects::new(
                EffectClasses::NONE,
                vec![
                    ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 1, mode: ReferenceAccessMode::Write },
                    ReferenceEffect::Allocate { output_index: 0 },
                ],
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
            context.bind(AsyncCopyOperation::new(), vec![], &[source, destination]),
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
                1,
                &[],
                ReferenceAccessMode::Read,
            ),
            Ok(referent)
        );
    }

    #[test]
    fn test_validate_masked_types_rejects_arity_shape_and_value_types() {
        assert_eq!(
            validate_masked_types(&[], &[], 3, 1, &[], ReferenceAccessMode::Read),
            Err(TypeError::invalid("expected 3 inputs but got 0"))
        );
        let referent = ArrayType::new_static(DataType::I32, vec![2]);
        let mask = ArrayType::new_static(DataType::Boolean, vec![1]);
        assert_eq!(
            validate_masked_types(
                &[ReferenceType::new(referent.clone()).into(), mask.clone().into(), referent.clone().into(),],
                &[],
                3,
                1,
                &[],
                ReferenceAccessMode::Read,
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
                2,
                &[],
                ReferenceAccessMode::Write,
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
