//! Checked construction and explicit decomposition of first-class ragged representations.
//!
//! These operations expose the low-level padded storage and full extent payloads used by transforms. Numerical array
//! operations still require an ordinary array projection. Construction checks live indexed extents, including sibling
//! dependencies; the structural [`RaggedValue::new`] constructor alone does not establish those numerical facts.
//!
//! Array-valued derivatives preserve every indexed occurrence while differentiating only the ordinary storage.
//! [`RaggedZeroOperation`] captures a shape carrier with Zero storage and full extent arrays, so a storage projection's
//! paired linear transpose and disconnected cotangent retain no numerical primal bytes solely to reconstruct shape.
//! Dimension-valued ragged representations have zero differential space. [`RaggedAddOperation`] checks matching live
//! extents before adding cotangent storage; ordinary addition still requires explicit storage projection.
//!
//! Batching transports aligned ordinary mapped storage axes, including the independent placement of that coordinate
//! in each extent payload. Construction broadcasts replicated storage along the batch axis when its extents or identity
//! source are mapped. Ragged addition broadcasts a replicated contribution onto the mapped storage axis of the other
//! contribution and checks its live extents against that contribution for every batch item. Two mapped contributions
//! still require equal mapped-axis positions, because a first-class ragged carrier cannot yet move its batch axis.
//! Standalone storage/new transposition cannot reconstruct all indexed checks from types alone; their JVPs carry
//! executable forward/transpose regions with explicit shape residuals.
//! Cotangents with reduced or unreduced extent placement remain unsupported because integer metadata cannot acquire
//! numerical partial-sum semantics.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::Display;
use std::num::NonZeroUsize;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayIrType, ArrayIrValue, ArrayType, ArrayTypeRefinements, DataType, DimensionType, DimensionVariable,
    RaggedDimension, RaggedType, RaggedValue, RaggedValueKind,
};
use crate::axes::Axis;
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, ResidualZeroProvider, SupportsCotangentAdd,
    TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_transposable_operation, impl_reference_dischargeable_operation};
use crate::operations::Capability;
use crate::operations::arithmetic::{Add, AddOperation};
use crate::operations::assertions::{Assert, GuardOperation};
use crate::operations::comparisons::Compare;
use crate::operations::constants::iota::Iota;
use crate::operations::constants::zero::Zero;
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::logical::{And, Not, Or};
use crate::operations::manipulation::broadcasting::Broadcast;
use crate::operations::manipulation::memory::TransferToMemory;
use crate::operations::reductions::Reduce;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    EffectClass, EffectClasses, Effects, MaybeZero, Operation, OperationFormatter, OperationProvider, ProgramError,
    RegionInterface, Type, TypeError, TypeIdentityRenaming, TypeRefinements, Typed, Value, ValueDomainDispatch,
    ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`RaggedNewOperation`].
pub const RAGGED_NEW_OPERATION_NAME: &str = "ragged_new";

/// Checks numerical extent payloads and assembles their padded storage into a first-class ragged value.
/// The first input is storage; subsequent inputs are full extent arrays in the type's canonical occurrence order.
/// Checks are ordered assertions and remain observable when the assembled value is unused.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RaggedNewOperation {
    /// Declared representation whose ordinary storage dimensions may be refined by the supplied fields.
    output_type: RaggedType,

    /// Whether a final ragged input forwards the indexed identities used by this representation.
    identity_source: bool,
}

impl RaggedNewOperation {
    /// Creates a checked construction operation for `output_type`.
    #[inline]
    pub fn new(output_type: RaggedType) -> Self {
        Self { output_type, identity_source: false }
    }

    /// Creates checked construction with a final ragged identity source after the dense fields. This input forwards
    /// existing indexed identities; it supplies no numerical proof and never suppresses payload validation.
    #[inline]
    pub fn with_identity_source(output_type: RaggedType) -> Self {
        Self { output_type, identity_source: true }
    }

    /// Returns the declared output representation.
    #[inline]
    pub fn output_type(&self) -> &RaggedType {
        &self.output_type
    }

    /// Returns whether the final input forwards the output's indexed identities.
    #[inline]
    pub fn has_identity_source(&self) -> bool {
        self.identity_source
    }
}

impl Display for RaggedNewOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for RaggedNewOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        RAGGED_NEW_OPERATION_NAME
    }

    fn defined_type_identities(&self) -> Cow<'_, [DimensionVariable]> {
        let item_definition = match self.output_type.value_kind() {
            RaggedValueKind::Array => None,
            RaggedValueKind::Dimension(dimension) => dimension.variable(),
        };
        let mut identities = Vec::with_capacity(self.output_type.ragged_dimensions().len());
        for descriptor in self.output_type.ragged_dimensions() {
            if Some(descriptor.dimension()) != item_definition && !identities.contains(descriptor.dimension()) {
                identities.push(descriptor.dimension().clone());
            }
        }
        Cow::Owned(identities)
    }

    fn consumed_type_identities(&self, input_types: &[ArrayIrType]) -> Result<Vec<DimensionVariable>, TypeError> {
        let field_count = self.output_type.extent_types().len() + 1;
        check_count!("input", input_types, field_count + usize::from(self.identity_source), TypeError);
        if self.identity_source {
            Ok(indexed_identities(<&RaggedType>::try_from(&input_types[field_count])?))
        } else {
            Ok(Vec::new())
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        let field_count = self.output_type.extent_types().len() + 1;
        check_count!("input", input_types, field_count + usize::from(self.identity_source), TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let fields = input_types[..field_count].iter().map(<&ArrayType>::try_from).collect::<Result<Vec<_>, _>>()?;
        // Compare the complete signature so repeated ordinary dimensions retain their cross-field relationships.
        let declared = std::iter::once(self.output_type.storage_type())
            .chain(self.output_type.extent_types())
            .collect::<Vec<_>>();
        ArrayTypeRefinements::new(declared, fields.iter().copied())?;
        let output = RaggedType::new(
            fields[0].clone(),
            self.output_type.value_kind().clone(),
            self.output_type.ragged_dimensions().to_vec(),
        )?;
        for (index, (expected, actual)) in output.extent_types().iter().zip(&fields[1..]).enumerate() {
            if expected != *actual {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_NEW_OPERATION_NAME}` extent input {index} must have type `{expected}` but got `{actual}`",
                )));
            }
        }
        if self.identity_source {
            let source = <&RaggedType>::try_from(&input_types[field_count])?;
            if source.value_kind() != output.value_kind()
                || output.ragged_dimensions().iter().any(|descriptor| {
                    !source.ragged_dimensions().iter().any(|other| other.dimension() == descriptor.dimension())
                })
            {
                return Err(TypeError::invalid(
                    "`ragged_new` identity source must have the same value kind and contain every output indexed identity",
                ));
            }
            ArrayIrType::validate_identity_scope([input_types[field_count].clone(), output.clone().into()])?;
        }
        Ok(vec![output.into()])
    }

    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedAssertion)))
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self { output_type: self.output_type.rename_identities(renaming)?, identity_source: self.identity_source })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("output_type", &self.output_type)?;
            if self.identity_source {
                operation.field("identity_source", true)?;
            }
            Ok(())
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free RaggedNewOperation);

impl<C: Domain<Type = ArrayIrType, Value: RaggedNew>> InterpretableOperation<C> for RaggedNewOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let field_count = self.output_type.extent_types().len() + 1;
        check_count!("input", inputs, field_count + usize::from(self.identity_source), ProgramError);
        if self.identity_source {
            self.infer_output_types(&inputs.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
            Ok(vec![inputs[0].ragged_new_with_identity_source(
                self.output_type.clone(),
                &inputs[1..field_count],
                &inputs[field_count],
            )?])
        } else {
            Ok(vec![inputs[0].ragged_new(self.output_type.clone(), &inputs[1..])?])
        }
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<RaggedNewOperation>>> PartiallyEvaluatableOperation<C>
    for RaggedNewOperation
{
}

impl<C, P> BatchableOperation<C, P> for RaggedNewOperation
where
    C: Context<Type = ArrayIrType, Operation: From<RaggedNewOperation>>,
    P: BatchingPolicy<C>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        let field_count = self.output_type.extent_types().len() + 1;
        check_count!("input", inputs, field_count + usize::from(self.identity_source), ProgramError);
        if inputs.iter().all(|input| P::batch_axis(input).is_replicated()) {
            let values = inputs.iter().map(P::value).cloned().collect::<Vec<_>>();
            let value = context.parent().bind(self.clone(), Vec::new(), &values)?.remove(0);
            return Ok(vec![P::replicated(value)].into());
        }

        // Replicated storage combined with mapped extents or a mapped identity source denotes the same padded storage
        // for every batch item, so it is broadcast along the batch axis like an unmapped input of an ordinary
        // elementwise rule. The inserted storage axis follows a mapped identity source when that position exists in
        // the packed storage (e.g., a forward-mode tangent rebuilt over its mapped primal keeps the primal's layout),
        // and leads otherwise. Extent payloads keep their own batch axes; only the storage carries the output's axis.
        let (storage, axis) = match P::batch_axis(&inputs[0]).axis() {
            Some(axis) => (inputs[0].clone(), axis),
            None => {
                let storage_type = P::value(&inputs[0]).r#type();
                let rank = <&ArrayType>::try_from(storage_type.as_ref())?.rank();
                // The identity source, when present, is the only input after the dense fields.
                let source_position = inputs
                    .get(field_count)
                    .and_then(|source| P::batch_axis(source).axis().map(|axis| (source, axis)))
                    .map(|(source, axis)| {
                        let source_type = P::value(source).r#type();
                        let source_type = <&RaggedType>::try_from(source_type.as_ref())?;
                        Ok::<_, BatchingError>(axis.normalize(source_type.storage_type().rank())?)
                    })
                    .transpose()?;
                let axis = Axis::from(source_position.filter(|&position| position <= rank).unwrap_or(0));
                (driver.align_batch_axis(context, inputs[0].clone(), axis)?, axis)
            }
        };
        let storage_type = P::value(&storage).r#type();
        let storage_type = <&ArrayType>::try_from(storage_type.as_ref())?;
        let batch_axis = P::batch_axis(&storage);
        let axis = axis.normalize(storage_type.rank())?;
        // Keep the declaration's ordinary dimension relationships while inserting the mapped storage coordinate.
        // Each extent field independently chooses whether it varies over that coordinate and where it stores it.
        let declared_storage = P::batched_type(context, &storage, &self.output_type.storage_type().clone().into())?;
        let descriptors = self
            .output_type
            .ragged_dimensions()
            .iter()
            .zip(&inputs[1..field_count])
            .map(|(descriptor, input)| {
                let mut extent_axes = descriptor
                    .extent_axes()
                    .iter()
                    .map(|&coordinate| coordinate + usize::from(coordinate >= axis))
                    .collect::<Vec<_>>();
                if let Some(extent_axis) = P::batch_axis(input).axis() {
                    let extent_type = P::value(input).r#type();
                    let extent_type = <&ArrayType>::try_from(extent_type.as_ref())?;
                    extent_axes.insert(extent_axis.normalize(extent_type.rank())?, axis);
                }
                Ok(RaggedDimension::new(
                    descriptor.axis() + usize::from(descriptor.axis() >= axis),
                    descriptor.dimension().clone(),
                    extent_axes,
                ))
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        let output_type = RaggedType::new(
            <&ArrayType>::try_from(&declared_storage)?.clone(),
            self.output_type.value_kind().clone(),
            descriptors,
        )?;
        let operation =
            if self.identity_source { Self::with_identity_source(output_type) } else { Self::new(output_type) };
        let values = std::iter::once(P::value(&storage))
            .chain(inputs[1..].iter().map(P::value))
            .cloned()
            .collect::<Vec<_>>();
        let value = context.parent().bind(operation, Vec::new(), &values)?.remove(0);
        Ok(vec![P::batch(value, batch_axis)?].into())
    }
}

impl<C> DifferentiableOperation<C> for RaggedNewOperation
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<RaggedNewOperation>
        + From<RaggedStorageOperation>
        + From<RaggedExtentsOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + ResidualZeroProvider<ArrayIrType, Operation = C::Operation>,
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = ArrayIrType, Constant = C::Constant, Operation = C::Operation>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal = context.primal().bind(self.clone(), Vec::new(), &primals)?.remove(0);
        let tangent_type = primal.r#type().tangent()?;
        let tangent = match inputs[0].tangent() {
            _ if tangent_type.is_zero_space() => MaybeZero::Zero(tangent_type),
            MaybeZero::Zero(_) => MaybeZero::Zero(tangent_type),
            MaybeZero::Value(value) => {
                // The tangent reads its extents and identity from a Zero-storage shape carrier of the primal result,
                // like the reverse rule below. Under `linearize`, the carrier, not the primal result, becomes the
                // retained residual, so the pushforward never retains numerical primal storage solely for shape.
                let source = C::Operation::capture_zero_source(context.primal(), &primal, &tangent_type)?;
                let source = context.primal_to_tangent(source)?;
                let target = <&RaggedType>::try_from(&tangent_type)?.clone();
                let mut fields = vec![value.clone()];
                for descriptor in target.ragged_dimensions() {
                    fields.push(
                        context
                            .tangent()
                            .bind(RaggedExtentsOperation::new(descriptor.axis()), Vec::new(), &[source.clone()])?
                            .remove(0),
                    );
                }
                fields.push(source);
                MaybeZero::Value(
                    context.tangent().bind(Self::with_identity_source(target), Vec::new(), &fields)?.remove(0),
                )
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }

    fn jvp_for_transpose<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal = context.primal().bind(self.clone(), Vec::new(), &primals)?.remove(0);
        let tangent_type = primal.r#type().tangent()?;
        let tangent = match inputs[0].tangent() {
            _ if tangent_type.is_zero_space() => MaybeZero::Zero(tangent_type),
            MaybeZero::Zero(_) => MaybeZero::Zero(tangent_type),
            MaybeZero::Value(tangent) => {
                let source = C::Operation::capture_zero_source(context.primal(), &primal, &tangent_type)?;
                let source = context.primal_to_tangent(source)?;
                let target = <&RaggedType>::try_from(&tangent_type)?.clone();
                let outputs = LinearCallOperation::stage(
                    context.tangent(),
                    vec![source],
                    vec![tangent.clone()],
                    move |residuals, linear| {
                        let extents = target
                            .ragged_dimensions()
                            .iter()
                            .map(|descriptor| residuals[0].ragged_extents(descriptor.axis()))
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(vec![linear[0].ragged_new_with_identity_source(target, &extents, &residuals[0])?])
                    },
                    |_residuals, cotangents| Ok(vec![cotangents[0].ragged_storage()?]),
                )?;
                MaybeZero::Value(outputs.into_iter().next().unwrap())
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

impl_non_transposable_operation!(RaggedNewOperation);

/// Explicit checked assembly of padded storage and full indexed extent arrays.
///
/// The receiver is an ordinary array member of a composite value. `extents` follows the canonical occurrence order
/// of `r#type`. Every live extent must satisfy its declared bounds, and repeated indexed identities must agree at
/// corresponding coordinates where both occurrences are live. Dimension-value storage is checked against its item
/// type as well. Inactive padding is ignored. Construction never treats a nominal type or successful structural
/// [`RaggedValue::new`] call as evidence that these numerical requirements hold.
/// Direct composite wrappers around ordinary array tracers reject unresolved ordinary shape relationships; bind this
/// operation in a composite array IR context when its declarations require such a runtime relationship check.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that composite array IR
/// values implement this capability for [`ArrayIrType`].
///
/// # Errors
///
/// Returns type errors for malformed fields and assertion failures for invalid live extents. A direct
/// [`ArrayIrValue`] wrapper returns [`ProgramError::UnsupportedOperation`] for unresolved ordinary dimension
/// relationships. Composite-native staging retains the checked operation and supports those declarations.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ArrayIrValue, ArrayType, DataType, DimensionBounds, DimensionVariable,
/// #     ProgramError, RaggedDimension, RaggedExtents, RaggedNew, RaggedStorage, RaggedType, RaggedValueKind};
/// # fn main() -> Result<(), ProgramError> {
/// let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4))?);
/// let representation = RaggedType::new(
///     ArrayType::new_static(DataType::F32, [2, 3]),
///     RaggedValueKind::Array,
///     vec![RaggedDimension::new(1, rows, vec![0])],
/// )?;
/// let storage = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f32; 6])?);
/// let extents = ArrayIrValue::Array(Array::vector(vec![1i64, 3])?);
/// let ragged = storage.ragged_new(representation, &[extents.clone()])?;
/// assert_eq!(ragged.ragged_storage()?, storage);
/// assert_eq!(ragged.ragged_extents(1)?, extents);
/// # Ok(())
/// # }
/// ```
#[capability]
pub trait RaggedNew<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Checks the provided representation fields and returns their assembled ragged value.
    fn ragged_new(&self, r#type: RaggedType, extents: &[Self]) -> Result<Self, ProgramError>;

    /// Checks and assembles the fields while forwarding indexed identities from `source`. The source does not prove
    /// numerical bounds or equality: both representations are checked, and every forwarded identity must agree at
    /// corresponding coordinates where both are live.
    fn ragged_new_with_identity_source(
        &self,
        r#type: RaggedType,
        extents: &[Self],
        source: &Self,
    ) -> Result<Self, ProgramError>;
}

impl<V> RaggedNew<ArrayIrType> for V
where
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayIrType,
            Domain: Context<Type = ArrayIrType, Operation: From<RaggedNewOperation>>,
        >,
{
    fn ragged_new(&self, r#type: RaggedType, extents: &[Self]) -> Result<Self, ProgramError> {
        let mut inputs = Vec::with_capacity(extents.len() + 1);
        inputs.push(self.clone());
        inputs.extend_from_slice(extents);
        Ok(self.domain().bind(RaggedNewOperation::new(r#type), Vec::new(), &inputs)?.remove(0))
    }

    fn ragged_new_with_identity_source(
        &self,
        r#type: RaggedType,
        extents: &[Self],
        source: &Self,
    ) -> Result<Self, ProgramError> {
        let mut inputs = Vec::with_capacity(extents.len() + 2);
        inputs.push(self.clone());
        inputs.extend_from_slice(extents);
        inputs.push(source.clone());
        Ok(self.domain().bind(RaggedNewOperation::with_identity_source(r#type), Vec::new(), &inputs)?.remove(0))
    }
}

impl<A> RaggedNew<ArrayIrType> for ArrayIrValue<A>
where
    A: Value<Type = ArrayType, Domain: Context + Iota<A>>
        + Compare
        + And
        + Or
        + Not
        + Broadcast
        + Reduce
        + Assert
        + TransferToMemory,
    <A::Domain as Domain>::Constant: From<Array>,
{
    fn ragged_new(&self, r#type: RaggedType, extents: &[Self]) -> Result<Self, ProgramError> {
        checked_new(self, r#type, extents, None)
    }

    fn ragged_new_with_identity_source(
        &self,
        r#type: RaggedType,
        extents: &[Self],
        source: &Self,
    ) -> Result<Self, ProgramError> {
        checked_new(self, r#type, extents, Some(source))
    }
}

/// Canonical operation name for [`RaggedStorageOperation`].
pub const RAGGED_STORAGE_OPERATION_NAME: &str = "ragged_storage";

/// Returns the padded ordinary array storage of a first-class ragged value. This explicit operation makes padding
/// visible; ordinary array projection remains strict and never silently drops the logical extent payloads.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct RaggedStorageOperation;

impl Display for RaggedStorageOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Operation for RaggedStorageOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        RAGGED_STORAGE_OPERATION_NAME
    }

    fn consumed_type_identities(&self, input_types: &[ArrayIrType]) -> Result<Vec<DimensionVariable>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(indexed_identities(<&RaggedType>::try_from(&input_types[0])?))
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        Ok(vec![<&RaggedType>::try_from(&input_types[0])?.storage_type().clone().into()])
    }
}

impl_reference_dischargeable_operation!(@reference_free RaggedStorageOperation);

impl<C: Domain<Type = ArrayIrType, Value: RaggedStorage>> InterpretableOperation<C> for RaggedStorageOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].ragged_storage()?])
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<RaggedStorageOperation>>> PartiallyEvaluatableOperation<C>
    for RaggedStorageOperation
{
}

impl<C, P> BatchableOperation<C, P> for RaggedStorageOperation
where
    C: Context<Type = ArrayIrType, Operation: From<RaggedStorageOperation>>,
    P: BatchingPolicy<C>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let value = context.parent().bind(*self, Vec::new(), &[P::value(&inputs[0]).clone()])?.remove(0);
        Ok(vec![P::batch(value, P::batch_axis(&inputs[0]))?].into())
    }
}

impl<C> DifferentiableOperation<C> for RaggedStorageOperation
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<RaggedStorageOperation>
        + From<RaggedNewOperation>
        + From<RaggedExtentsOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + ResidualZeroProvider<ArrayIrType, Operation = C::Operation>,
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = ArrayIrType, Constant = C::Constant, Operation = C::Operation>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = context.primal().bind(*self, Vec::new(), &[inputs[0].primal().clone()])?.remove(0);
        let tangent_type = primal.r#type().tangent()?;
        let tangent = match inputs[0].tangent() {
            _ if tangent_type.is_zero_space() => MaybeZero::Zero(tangent_type),
            MaybeZero::Zero(_) => MaybeZero::Zero(tangent_type),
            MaybeZero::Value(value) => {
                MaybeZero::Value(context.tangent().bind(*self, Vec::new(), &[value.clone()])?.remove(0))
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }

    fn jvp_for_transpose<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = context.primal().bind(*self, Vec::new(), &[inputs[0].primal().clone()])?.remove(0);
        let tangent_type = primal.r#type().tangent()?;
        let tangent = match inputs[0].tangent() {
            _ if tangent_type.is_zero_space() => MaybeZero::Zero(tangent_type),
            MaybeZero::Zero(_) => MaybeZero::Zero(tangent_type),
            MaybeZero::Value(tangent) => {
                let input_cotangent_type = inputs[0].primal().r#type().cotangent()?;
                let source =
                    C::Operation::capture_zero_source(context.primal(), inputs[0].primal(), &input_cotangent_type)?;
                let source = context.primal_to_tangent(source)?;
                let target = <&RaggedType>::try_from(&input_cotangent_type)?.clone();
                let outputs = LinearCallOperation::stage(
                    context.tangent(),
                    vec![source],
                    vec![tangent.clone()],
                    |_residuals, linear| Ok(vec![linear[0].ragged_storage()?]),
                    move |residuals, cotangents| {
                        let extents = target
                            .ragged_dimensions()
                            .iter()
                            .map(|descriptor| residuals[0].ragged_extents(descriptor.axis()))
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(vec![cotangents[0].ragged_new_with_identity_source(target, &extents, &residuals[0])?])
                    },
                )?;
                MaybeZero::Value(outputs.into_iter().next().unwrap())
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

// Its paired JVP retains indexed payloads for transposition; the standalone projection type cannot supply them.
impl_non_transposable_operation!(RaggedStorageOperation);

/// Explicit access to the padded ordinary array storage of a ragged value.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that composite array IR
/// values implement this capability for [`ArrayIrType`].
#[capability]
pub trait RaggedStorage<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns the representation's physical storage, including padding.
    ///
    /// # Errors
    ///
    /// Returns a type error when the receiver is not a ragged representation, or the context's binding error.
    fn ragged_storage(&self) -> Result<Self, ProgramError>;
}

impl<V> RaggedStorage<ArrayIrType> for V
where
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayIrType,
            Domain: Context<Type = ArrayIrType, Operation: From<RaggedStorageOperation>>,
        >,
{
    fn ragged_storage(&self) -> Result<Self, ProgramError> {
        Ok(self.domain().bind(RaggedStorageOperation, Vec::new(), std::slice::from_ref(self))?.remove(0))
    }
}

impl<A: Value<Type = ArrayType>> RaggedStorage<ArrayIrType> for ArrayIrValue<A> {
    fn ragged_storage(&self) -> Result<Self, ProgramError> {
        let Self::Ragged(value) = self else {
            return Err(<&RaggedType>::try_from(self.r#type().as_ref()).unwrap_err().into());
        };
        Ok(Self::Array(value.storage().clone()))
    }
}

/// Canonical operation name for [`RaggedExtentsOperation`].
pub const RAGGED_EXTENTS_OPERATION_NAME: &str = "ragged_extents";

/// Returns the full dense extent payload for one ragged storage-axis occurrence. The axis selects the occurrence,
/// rather than a nominal identity, because one identity can occur on more than one storage axis.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RaggedExtentsOperation {
    /// Storage axis whose occurrence is selected; negative positions are resolved against the storage rank.
    axis: Axis,
}

impl RaggedExtentsOperation {
    /// Creates an extent projection selecting a storage axis occurrence.
    pub fn new<A: Into<Axis>>(axis: A) -> Self {
        Self { axis: axis.into() }
    }

    /// Returns the selected positional storage axis.
    pub fn axis(&self) -> Axis {
        self.axis
    }

    /// Resolves the selected occurrence in a validated representation.
    fn occurrence(&self, r#type: &RaggedType) -> Result<usize, TypeError> {
        let axis = self
            .axis
            .normalize(r#type.storage_type().rank())
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        r#type.ragged_dimensions().iter().position(|descriptor| descriptor.axis() == axis).ok_or_else(|| {
            TypeError::invalid(format!(
                "`{RAGGED_EXTENTS_OPERATION_NAME}` storage axis {axis} has no ragged occurrence"
            ))
        })
    }
}

impl Display for RaggedExtentsOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for RaggedExtentsOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        RAGGED_EXTENTS_OPERATION_NAME
    }

    fn consumed_type_identities(&self, input_types: &[ArrayIrType]) -> Result<Vec<DimensionVariable>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(indexed_identities(<&RaggedType>::try_from(&input_types[0])?))
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let r#type = <&RaggedType>::try_from(&input_types[0])?;
        Ok(vec![r#type.extent_types()[self.occurrence(r#type)?].clone().into()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("axis", self.axis))
    }
}

impl_reference_dischargeable_operation!(@reference_free RaggedExtentsOperation);

impl<C: Domain<Type = ArrayIrType, Value: RaggedExtents>> InterpretableOperation<C> for RaggedExtentsOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].ragged_extents(self.axis)?])
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<RaggedExtentsOperation>>> PartiallyEvaluatableOperation<C>
    for RaggedExtentsOperation
{
}

impl<C, P> BatchableOperation<C, P> for RaggedExtentsOperation
where
    C: Context<Type = ArrayIrType, Operation: From<RaggedExtentsOperation>>,
    P: BatchingPolicy<C>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let packed_type = P::value(&inputs[0]).r#type();
        let packed_type = <&RaggedType>::try_from(packed_type.as_ref())?;
        let per_item_type = P::unbatched_type(&inputs[0]);
        let per_item_type = <&RaggedType>::try_from(per_item_type.as_ref())?;
        let occurrence = self.occurrence(per_item_type)?;
        let descriptor = &packed_type.ragged_dimensions()[occurrence];
        let batch_axis = P::batch_axis(&inputs[0])
            .axis()
            .map(|axis| axis.normalize(packed_type.storage_type().rank()))
            .transpose()?;
        let output_axis =
            batch_axis.and_then(|axis| descriptor.extent_axes().iter().position(|&coordinate| coordinate == axis));
        let value = context
            .parent()
            .bind(Self::new(descriptor.axis()), Vec::new(), &[P::value(&inputs[0]).clone()])?
            .remove(0);
        Ok(vec![P::batch(
            value,
            output_axis.map_or_else(crate::batching::BatchAxis::replicated, crate::batching::BatchAxis::from_position),
        )?]
        .into())
    }
}

impl<C> DifferentiableOperation<C> for RaggedExtentsOperation
where
    C: Context<Type = ArrayIrType, Operation: From<RaggedExtentsOperation>>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = context.primal().bind(*self, Vec::new(), &[inputs[0].primal().clone()])?.remove(0);
        Ok(vec![DifferentiationDual::new_with_zero_tangent(primal)?])
    }
}

impl_non_transposable_operation!(RaggedExtentsOperation);

/// Explicit access to the full dense extents of a ragged storage-axis occurrence.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that composite array IR
/// values implement this capability for [`ArrayIrType`].
#[capability]
pub trait RaggedExtents<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns the selected occurrence's complete extent array, retaining entries outside the live coordinates
    /// determined by its sibling occurrences.
    ///
    /// # Errors
    ///
    /// Returns a type error when the receiver is not ragged, `axis` is out of bounds, or that storage axis has no
    /// ragged occurrence. Also propagates the context's binding errors.
    fn ragged_extents<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError>;
}

impl<V> RaggedExtents<ArrayIrType> for V
where
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayIrType,
            Domain: Context<Type = ArrayIrType, Operation: From<RaggedExtentsOperation>>,
        >,
{
    fn ragged_extents<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError> {
        Ok(self
            .domain()
            .bind(RaggedExtentsOperation::new(axis), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<A: Value<Type = ArrayType>> RaggedExtents<ArrayIrType> for ArrayIrValue<A> {
    fn ragged_extents<B: Into<Axis>>(&self, axis: B) -> Result<Self, ProgramError> {
        let Self::Ragged(value) = self else {
            return Err(<&RaggedType>::try_from(self.r#type().as_ref()).unwrap_err().into());
        };
        let index = RaggedExtentsOperation::new(axis).occurrence(value.r#type().as_ref())?;
        Ok(Self::Array(value.extents()[index].clone()))
    }
}

/// Canonical operation name for [`RaggedZeroOperation`].
pub const RAGGED_ZERO_OPERATION_NAME: &str = "ragged_zero";

/// Constructs zero storage while preserving a source's complete indexed shape. The source supplies every extent
/// occurrence and ordinary runtime dimension. A Zero-element output is a shape-only carrier: its storage retains
/// shape metadata without retaining numerical primal bytes. Construction still validates the source and its extents.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RaggedZeroOperation {
    /// Requested zero representation, specialized against the source's physical dimensions at inference time.
    output_type: RaggedType,
}

impl RaggedZeroOperation {
    /// Creates a zero operation that consumes one ragged array shape source.
    #[inline]
    pub fn new(output_type: RaggedType) -> Self {
        Self { output_type }
    }

    /// Returns the declared zero representation.
    #[inline]
    pub fn output_type(&self) -> &RaggedType {
        &self.output_type
    }
}

impl Display for RaggedZeroOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for RaggedZeroOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        RAGGED_ZERO_OPERATION_NAME
    }

    fn consumed_type_identities(&self, input_types: &[ArrayIrType]) -> Result<Vec<DimensionVariable>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(indexed_identities(<&RaggedType>::try_from(&input_types[0])?))
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let source = <&RaggedType>::try_from(&input_types[0])?;
        if self.output_type.value_kind() != &RaggedValueKind::Array || source.value_kind() != &RaggedValueKind::Array {
            return Err(TypeError::invalid("`ragged_zero` requires array-kind ragged output and source"));
        }
        if source.ragged_dimensions() != self.output_type.ragged_dimensions() {
            return Err(TypeError::invalid(
                "`ragged_zero` must preserve every source indexed occurrence and its storage coordinates",
            ));
        }
        self.output_type.storage_type().validate_zero()?;
        let storage = self.output_type.storage_type().clone().with_shape(source.storage_type().shape().clone());
        let mut fields = vec![storage.into()];
        fields.extend(source.extent_types().iter().cloned().map(ArrayIrType::from));
        fields.push(input_types[0].clone());
        RaggedNewOperation::with_identity_source(self.output_type.clone()).infer_output_types(&fields, &[])
    }

    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedAssertion)))
    }

    fn is_zero(&self, output_index: usize) -> bool {
        output_index == 0
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self::new(self.output_type.rename_identities(renaming)?))
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("output_type", &self.output_type))
    }
}

impl_reference_dischargeable_operation!(@reference_free RaggedZeroOperation);

impl<C> InterpretableOperation<C> for RaggedZeroOperation
where
    C: Domain<Type = ArrayIrType> + Zero<C::Value>,
    C::Value: RaggedStorage + RaggedExtents + RaggedNew,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let output_types = self.infer_output_types(&input_types, &[])?;
        let output_type = <&RaggedType>::try_from(&output_types[0])?;
        // Eager values carry their actual physical dimensions in their types. Constructor specialization above
        // validates the declaration before the ordinary domain allocates zero storage; staging keeps this native
        // operation and its source edge intact, so no device payload is read to discover a dimension.
        let storage = context.zero(&output_type.storage_type().clone().into())?;
        let extents = output_type
            .ragged_dimensions()
            .iter()
            .map(|descriptor| inputs[0].ragged_extents(descriptor.axis()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(vec![storage.ragged_new_with_identity_source(output_type.clone(), &extents, &inputs[0])?])
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<RaggedZeroOperation>>> PartiallyEvaluatableOperation<C>
    for RaggedZeroOperation
{
}

impl<C, P> BatchableOperation<C, P> for RaggedZeroOperation
where
    C: Context<Type = ArrayIrType, Operation: From<RaggedZeroOperation>>,
    P: BatchingPolicy<C>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let source_type = P::unbatched_type(&inputs[0]);
        self.infer_output_types(std::slice::from_ref(source_type.as_ref()), &[])?;
        let source = <&RaggedType>::try_from(source_type.as_ref())?;
        // Policy lifting relates a declaration to its source carrier; it does not perform the zero operation's
        // element-representation change. Lift the target coordinates and placement using the source element type,
        // then restore the requested output representation before binding the checked parent operation.
        let prototype = RaggedType::new(
            self.output_type.storage_type().clone().with_data_type(source.storage_type().data_type()),
            self.output_type.value_kind().clone(),
            self.output_type.ragged_dimensions().to_vec(),
        )?;
        let output_type = P::batched_type(context, &inputs[0], &prototype.into())?;
        let output_type = <&RaggedType>::try_from(&output_type)?;
        let operation = Self::new(RaggedType::new(
            output_type.storage_type().clone().with_data_type(self.output_type.storage_type().data_type()),
            output_type.value_kind().clone(),
            output_type.ragged_dimensions().to_vec(),
        )?);
        let value = context.parent().bind(operation, Vec::new(), &[P::value(&inputs[0]).clone()])?.remove(0);
        Ok(vec![P::batch(value, P::batch_axis(&inputs[0]))?].into())
    }
}

impl<C> DifferentiableOperation<C> for RaggedZeroOperation
where
    C: Context<Type = ArrayIrType, Operation: From<RaggedZeroOperation>>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal = context.primal().bind(self.clone(), Vec::new(), &primals)?.remove(0);
        Ok(vec![DifferentiationDual::new_with_zero_tangent(primal)?])
    }
}

impl_non_transposable_operation!(RaggedZeroOperation);

/// Canonical operation name for [`RaggedAddOperation`].
pub const RAGGED_ADD_OPERATION_NAME: &str = "ragged_add";

/// Adds two ragged cotangent representations after checking all live extent occurrences for agreement. Indexed
/// metadata is copied rather than numerically added. The [`SupportsCotangentAdd`] trait selects this operation;
/// ordinary numerical capabilities continue to require explicit ordinary storage projection.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct RaggedAddOperation;

impl Display for RaggedAddOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(RAGGED_ADD_OPERATION_NAME)
    }
}

impl Operation for RaggedAddOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        RAGGED_ADD_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let left = <&RaggedType>::try_from(&input_types[0])?;
        let right = <&RaggedType>::try_from(&input_types[1])?;
        if left.value_kind() != &RaggedValueKind::Array || left != right {
            return Err(TypeError::invalid("`ragged_add` requires two identical array-kind ragged types"));
        }
        // Zero-space contributions have no numerical payload and need no elementwise arithmetic.
        if left.storage_type().data_type() != DataType::Zero {
            AddOperation::<ArrayType>::new()
                .infer_output_types(&[left.storage_type().clone(), right.storage_type().clone()], &[])?;
        }
        Ok(vec![input_types[0].clone()])
    }

    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedAssertion)))
    }
}

impl_reference_dischargeable_operation!(@reference_free RaggedAddOperation);

impl<C> InterpretableOperation<C> for RaggedAddOperation
where
    C: Domain<Type = ArrayIrType>,
    C::Value: RaggedStorage + RaggedExtents + RaggedNew + Add,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        let input_type = inputs[0].r#type();
        let r#type = <&RaggedType>::try_from(input_type.as_ref())?;
        let extents = r#type
            .ragged_dimensions()
            .iter()
            .map(|descriptor| inputs[0].ragged_extents(descriptor.axis()))
            .collect::<Result<Vec<_>, _>>()?;
        let left = inputs[0].ragged_storage()?.ragged_new_with_identity_source(r#type.clone(), &extents, &inputs[1])?;
        let storage = if r#type.storage_type().data_type() == DataType::Zero {
            left.ragged_storage()?
        } else {
            left.ragged_storage()?.add(&inputs[1].ragged_storage()?)?
        };
        Ok(vec![storage.ragged_new_with_identity_source(r#type.clone(), &extents, &left)?])
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<RaggedAddOperation>>> PartiallyEvaluatableOperation<C>
    for RaggedAddOperation
{
}

impl<C, P> BatchableOperation<C, P> for RaggedAddOperation
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<RaggedAddOperation>
        + From<RaggedNewOperation>
        + From<RaggedStorageOperation>
        + From<RaggedExtentsOperation>,
    P: BatchingPolicy<C>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        self.infer_output_types(
            &[P::unbatched_type(&inputs[0]).into_owned(), P::unbatched_type(&inputs[1]).into_owned()],
            &[],
        )?;

        // A replicated contribution next to a mapped one is first aligned to the mapped storage axis through the
        // policy, and then rebuilt in the mapped contribution's representation with that contribution as its identity
        // source. The rebuild takes the aligned per-item extent payloads where the mapped representation varies over
        // the batch coordinate and the shared payloads elsewhere, so both inputs of the parent addition have identical
        // types, and it checks the live extents of every batch item against the mapped contribution.
        let mut inputs = inputs.to_vec();
        match (P::batch_axis(&inputs[0]).is_replicated(), P::batch_axis(&inputs[1]).is_replicated()) {
            (false, true) => inputs[1] = align_ragged_contribution(context, driver, &inputs[1], &inputs[0])?,
            (true, false) => inputs[0] = align_ragged_contribution(context, driver, &inputs[0], &inputs[1])?,
            _ => {}
        }
        if P::batch_axis(&inputs[0]) != P::batch_axis(&inputs[1]) {
            return Err(BatchingError::UnsupportedOperation {
                message: "`ragged_add` requires both inputs to use the same mapped storage axis".into(),
            });
        }
        let value = context
            .parent()
            .bind(*self, Vec::new(), &[P::value(&inputs[0]).clone(), P::value(&inputs[1]).clone()])?
            .remove(0);
        Ok(vec![P::batch(value, P::batch_axis(&inputs[0]))?].into())
    }
}

impl<C> DifferentiableOperation<C> for RaggedAddOperation
where
    C: Context<Type = ArrayIrType, Operation: From<RaggedAddOperation>>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let primal = context
            .primal()
            .bind(*self, Vec::new(), &[inputs[0].primal().clone(), inputs[1].primal().clone()])?
            .remove(0);
        let tangent = match (inputs[0].tangent(), inputs[1].tangent()) {
            (MaybeZero::Zero(_), MaybeZero::Zero(_)) => MaybeZero::Zero(primal.r#type().tangent()?),
            (MaybeZero::Value(value), MaybeZero::Zero(_)) | (MaybeZero::Zero(_), MaybeZero::Value(value)) => {
                MaybeZero::Value(value.clone())
            }
            (MaybeZero::Value(left), MaybeZero::Value(right)) => {
                MaybeZero::Value(context.tangent().bind(*self, Vec::new(), &[left.clone(), right.clone()])?.remove(0))
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

impl<V, O> TransposableOperation<V, O> for RaggedAddOperation
where
    V: Value<Type = ArrayIrType>,
    O: Operation<Type = ArrayIrType> + SupportsCotangentAdd,
    TracingContext<V, O>: StagingContext<Type = ArrayIrType, Constant = V, Operation = O>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 2, DifferentiationError);
        for accumulator in accumulators {
            accumulator.accumulate(context, outputs[0].clone())?;
        }
        Ok(())
    }
}

/// Aligns the `replicated` input of a batched [`RaggedAddOperation`] to the mapped storage axis of `mapped` and
/// rebuilds it in the packed representation of `mapped`, which also serves as its identity source. Each extent payload
/// comes from the aligned value when the corresponding descriptor of `mapped` varies over the batch coordinate, and
/// from `replicated` itself when `mapped` shares one payload across the batch. Both inputs must have the same per-item
/// type.
fn align_ragged_contribution<C, P, D>(
    context: &BatchingContext<C, P>,
    driver: &D,
    replicated: &P::Batch,
    mapped: &P::Batch,
) -> Result<P::Batch, BatchingError>
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<RaggedNewOperation> + From<RaggedStorageOperation> + From<RaggedExtentsOperation>,
    P: BatchingPolicy<C>,
    D: BatchingDriver<C, P>,
{
    let mapped_type = P::value(mapped).r#type();
    let mapped_type = <&RaggedType>::try_from(mapped_type.as_ref())?;
    let replicated_type = P::value(replicated).r#type();
    let replicated_type = <&RaggedType>::try_from(replicated_type.as_ref())?;
    let batch_axis = P::batch_axis(mapped);
    let position = batch_axis.axis().unwrap().normalize(mapped_type.storage_type().rank())?;
    let aligned = driver.align_batch_axis(context, replicated.clone(), Axis::from(position))?;
    let parent = context.parent();
    let mut fields = parent.bind(RaggedStorageOperation, Vec::new(), std::slice::from_ref(P::value(&aligned)))?;
    for (descriptor, replicated_descriptor) in
        mapped_type.ragged_dimensions().iter().zip(replicated_type.ragged_dimensions())
    {
        let (source, axis) = match descriptor.extent_axes().contains(&position) {
            true => (P::value(&aligned), descriptor.axis()),
            false => (P::value(replicated), replicated_descriptor.axis()),
        };
        fields.extend(parent.bind(RaggedExtentsOperation::new(axis), Vec::new(), std::slice::from_ref(source))?);
    }
    fields.push(P::value(mapped).clone());
    let operation = RaggedNewOperation::with_identity_source(mapped_type.clone());
    let value = parent.bind(operation, Vec::new(), &fields)?.remove(0);
    P::batch(value, batch_axis)
}

/// Indexed identities whose associations a representation-level operation may omit from its outputs. A dimension
/// item's represented identity is indexed data too, even when the representation has no ragged storage axes.
fn indexed_identities(r#type: &RaggedType) -> Vec<DimensionVariable> {
    let mut identities = Vec::with_capacity(r#type.ragged_dimensions().len() + 1);
    for descriptor in r#type.ragged_dimensions() {
        if !identities.contains(descriptor.dimension()) {
            identities.push(descriptor.dimension().clone());
        }
    }
    if let RaggedValueKind::Dimension(dimension) = r#type.value_kind()
        && let Some(identity) = dimension.variable()
        && !identities.contains(identity)
    {
        identities.push(identity.clone());
    }
    identities
}

/// Shares structural admission and numerical checks between fresh creation and identity-forwarding reconstruction.
fn checked_new<A>(
    storage: &ArrayIrValue<A>,
    r#type: RaggedType,
    extents: &[ArrayIrValue<A>],
    source: Option<&ArrayIrValue<A>>,
) -> Result<ArrayIrValue<A>, ProgramError>
where
    A: Value<Type = ArrayType, Domain: Context + Iota<A>>
        + Compare
        + And
        + Or
        + Not
        + Broadcast
        + Reduce
        + Assert
        + TransferToMemory,
    <A::Domain as Domain>::Constant: From<Array>,
{
    let mut input_types = Vec::with_capacity(extents.len() + 1 + usize::from(source.is_some()));
    input_types.push(storage.r#type().into_owned());
    input_types.extend(extents.iter().map(|extent| extent.r#type().into_owned()));
    input_types.extend(source.map(|source| source.r#type().into_owned()));
    let operation = match source {
        None => RaggedNewOperation::new(r#type),
        Some(_) => RaggedNewOperation::with_identity_source(r#type),
    };
    let outputs = operation.infer_output_types(&input_types, &[])?;
    let field_count = operation.output_type.extent_types().len() + 1;
    let declared = std::iter::once(operation.output_type.storage_type())
        .chain(operation.output_type.extent_types())
        .collect::<Vec<_>>();
    let actual = input_types[..field_count].iter().map(<&ArrayType>::try_from).collect::<Result<Vec<_>, _>>()?;
    if !ArrayTypeRefinements::new(declared, actual)?.identities_needing_runtime_checks().is_empty() {
        return Err(ProgramError::UnsupportedOperation {
            message: "direct ragged construction cannot enforce unresolved ordinary dimension relationships; bind \
                `ragged_new` in a composite array IR context"
                .to_owned(),
        });
    }
    let r#type = <&RaggedType>::try_from(&outputs[0])?.clone();
    let storage = <ArrayIrValue<A> as ValueProjection<ArrayType>>::projected(storage)?.clone();
    let extents = extents
        .iter()
        .map(|value| <ArrayIrValue<A> as ValueProjection<ArrayType>>::projected(value).cloned())
        .collect::<Result<Vec<_>, _>>()?;
    let value = RaggedValue::new(r#type, storage, extents)?;
    let masks = validate_payloads(value.r#type().as_ref(), value.storage(), value.extents())?;
    let source = source.map(<ArrayIrValue<A> as ValueProjection<RaggedType>>::projected).transpose()?;
    if let Some(source) = source {
        // A structural source is not a numerical proof. Validate its ancestors before their liveness can justify
        // ignoring a coordinate, and retain comparisons against every occurrence of each forwarded identity.
        let source_masks = validate_payloads(source.r#type().as_ref(), source.storage(), source.extents())?;
        validate_source(&value, &masks, source, &source_masks)?;
    }
    // Direct composite values can contain array tracers. Standalone assertions must therefore be followed by a
    // value-producing ordered guard: projecting a field then cannot escape a pending extent or source check.
    let own_count = value.extents().len() + 1;
    let mut fields =
        std::iter::once(value.storage().clone()).chain(value.extents().iter().cloned()).collect::<Vec<_>>();
    if let Some(source) = source {
        fields.push(source.storage().clone());
        fields.extend_from_slice(source.extents());
    }
    let outputs = guard_array_fields(RAGGED_NEW_OPERATION_NAME, &fields)?;
    let mut outputs = outputs.into_iter();
    let storage = outputs.next().unwrap();
    let extents = outputs.take(own_count - 1).collect();
    Ok(ArrayIrValue::Ragged(RaggedValue::new(value.r#type().into_owned(), storage, extents)?))
}

/// Checks the indexed relationships of one exact-schema guard. Every carrier is independently validated before
/// its liveness masks can justify ignoring padding, including dimension-item data and single-carrier bounds.
pub(crate) fn check_guard_ragged_inputs<A>(inputs: &[&RaggedValue<A>]) -> Result<(), ProgramError>
where
    A: Value<Type = ArrayType, Domain: Context + Iota<A>>
        + Compare
        + And
        + Or
        + Not
        + Broadcast
        + Reduce
        + Assert
        + TransferToMemory,
    <A::Domain as Domain>::Constant: From<Array>,
{
    let masks = inputs
        .iter()
        .map(|value| validate_payloads(value.r#type().as_ref(), value.storage(), value.extents()))
        .collect::<Result<Vec<_>, _>>()?;
    for (index, value) in inputs.iter().enumerate() {
        let r#type = value.r#type();
        for (occurrence, (identity, axes)) in r#type.indexed_identities().enumerate() {
            let (payload, mask) = if occurrence < value.extents().len() {
                (&value.extents()[occurrence], masks[index].extents[occurrence].as_ref())
            } else {
                (value.storage(), masks[index].dimension.as_ref())
            };
            for (other_index, other) in inputs[..=index].iter().enumerate() {
                let other_type = other.r#type();
                for (other_occurrence, (other_identity, other_axes)) in other_type.indexed_identities().enumerate() {
                    if identity != other_identity || (other_index == index && other_occurrence >= occurrence) {
                        continue;
                    }
                    let (other_payload, other_mask) = if other_occurrence < other.extents().len() {
                        (&other.extents()[other_occurrence], masks[other_index].extents[other_occurrence].as_ref())
                    } else {
                        (other.storage(), masks[other_index].dimension.as_ref())
                    };
                    let correspondence = corresponding_axes(other_type.as_ref(), &other_axes, r#type.as_ref(), &axes);
                    check_equality(
                        payload,
                        mask,
                        other_payload,
                        other_mask,
                        &correspondence,
                        &format!("guarded ragged occurrences of `{identity}` must have equal live extents"),
                        "other_extent",
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// Retains all dense field and ambient-identity dependencies after an indexed check, including when its values are
/// ordinary array tracers. Callers return these outputs rather than the inputs whose checks may still be pending.
pub(crate) fn guard_array_fields<A>(boundary: &str, fields: &[A]) -> Result<Vec<A>, ProgramError>
where
    A: Value<Type = ArrayType, Domain: Context>,
{
    let types = fields.iter().map(|field| field.r#type().into_owned()).collect::<Vec<_>>();
    guard_declared_array_fields(boundary, types.clone(), fields)
}

/// Executes ordinary declared relationships before indexed liveness can use any field. Keeping every ambient
/// identity prevents this dense guard from silently changing a raw carrier's physical representation.
pub(crate) fn guard_declared_array_fields<A>(
    boundary: &str,
    declared: Vec<ArrayType>,
    fields: &[A],
) -> Result<Vec<A>, ProgramError>
where
    A: Value<Type = ArrayType, Domain: Context>,
{
    let Some(first) = fields.first() else { return Ok(Vec::new()) };
    let types = fields.iter().map(|field| field.r#type().into_owned()).collect::<Vec<_>>();
    let mut identities = Vec::new();
    for r#type in &types {
        for (_, identity) in r#type.identities() {
            if !identities.contains(identity) {
                identities.push(identity.clone());
            }
        }
    }
    let guard = GuardOperation::new(boundary, declared).with_retained_identities(identities);
    let operation = <A::Domain as Domain>::Operation::provide(guard, &types.iter().collect::<Vec<_>>())?;
    let outputs = first.domain().bind(operation, Vec::new(), fields)?;
    check_count!("output", outputs, fields.len(), ProgramError);
    Ok(outputs)
}

/// Temporary live-coordinate predicates shared by a construction's local and identity-source checks.
struct PayloadMasks<A> {
    /// Predicate over the full extent axes of each [`RaggedDimension`](crate::arrays::RaggedDimension); absence means
    /// every coordinate is live.
    extents: Vec<Option<A>>,

    /// Predicate over a dimension value's full storage shape, when it has ragged coordinates.
    dimension: Option<A>,
}

/// Checks every independent numerical requirement before exposing an assembled value. Masks are built in canonical
/// dependency order and retained only for this construction; each mask covers one occurrence's extent axes, rather than
/// storage.
fn validate_payloads<A>(r#type: &RaggedType, storage: &A, extents: &[A]) -> Result<PayloadMasks<A>, ProgramError>
where
    A: Value<Type = ArrayType, Domain: Context + Iota<A>>
        + Compare
        + And
        + Or
        + Not
        + Broadcast
        + Reduce
        + Assert
        + TransferToMemory,
    <A::Domain as Domain>::Constant: From<Array>,
{
    let mut masks: Vec<Option<A>> = Vec::with_capacity(extents.len());
    for (index, (descriptor, value)) in r#type.ragged_dimensions().iter().zip(extents).enumerate() {
        let mask = live_coordinate_mask(r#type, descriptor.extent_axes(), value, extents)?;
        check_bounds(
            value,
            &DimensionType::from(descriptor.dimension().clone()),
            mask.as_ref(),
            &format!("ragged extent on storage axis {} is outside its declared bounds", descriptor.axis()),
        )?;

        // Comparing every pair is necessary: one occurrence can be inactive where two later occurrences overlap.
        for previous in 0..index {
            let other = &r#type.ragged_dimensions()[previous];
            if other.dimension() != descriptor.dimension() {
                continue;
            }
            let axes = corresponding_axes(r#type, other.extent_axes(), r#type, descriptor.extent_axes());
            check_equality(
                value,
                mask.as_ref(),
                &extents[previous],
                masks[previous].as_ref(),
                &axes,
                &format!("ragged occurrences of `{}` must have equal live extents", descriptor.dimension()),
                "other_extent",
            )?;
        }
        masks.push(mask);
    }
    let dimension = if let RaggedValueKind::Dimension(dimension) = r#type.value_kind() {
        let extent_axes = (0..r#type.storage_type().rank()).collect::<Vec<_>>();
        let mask = live_coordinate_mask(r#type, &extent_axes, storage, extents)?;
        check_bounds(storage, dimension, mask.as_ref(), "ragged dimension value is outside its declared bounds")?;
        mask
    } else {
        None
    };
    Ok(PayloadMasks { extents: masks, dimension })
}

/// Checks the numeric equality established by forwarding each indexed identity from a source carrier. Source and
/// output masks are derived separately because changed sibling extents can change their live coordinate intersection.
fn validate_source<A>(
    value: &RaggedValue<A>,
    masks: &PayloadMasks<A>,
    source: &RaggedValue<A>,
    source_masks: &PayloadMasks<A>,
) -> Result<(), ProgramError>
where
    A: Value<Type = ArrayType> + Compare + And + Or + Not + Broadcast + Assert + TransferToMemory,
{
    let r#type = value.r#type();
    let source_type = source.r#type();
    for (index, descriptor) in r#type.ragged_dimensions().iter().enumerate() {
        for (source_index, source_descriptor) in source_type.ragged_dimensions().iter().enumerate() {
            if descriptor.dimension() != source_descriptor.dimension() {
                continue;
            }
            let axes = corresponding_axes(
                source_type.as_ref(),
                source_descriptor.extent_axes(),
                r#type.as_ref(),
                descriptor.extent_axes(),
            );
            check_equality(
                &value.extents()[index],
                masks.extents[index].as_ref(),
                &source.extents()[source_index],
                source_masks.extents[source_index].as_ref(),
                &axes,
                &format!("ragged identity source for `{}` must have equal live extents", descriptor.dimension()),
                "source_extent",
            )?;
        }
    }
    if let RaggedValueKind::Dimension(dimension) = r#type.value_kind()
        && let Some(identity) = dimension.variable()
    {
        let extent_axes = (0..r#type.storage_type().rank()).collect::<Vec<_>>();
        let source_extent_axes = (0..source_type.storage_type().rank()).collect::<Vec<_>>();
        let axes = corresponding_axes(source_type.as_ref(), &source_extent_axes, r#type.as_ref(), &extent_axes);
        check_equality(
            value.storage(),
            masks.dimension.as_ref(),
            source.storage(),
            source_masks.dimension.as_ref(),
            &axes,
            &format!("ragged identity source for `{identity}` must have equal live dimension values"),
            "source_extent",
        )?;
    }
    Ok(())
}

/// Aligns an indexed payload and its liveness, then checks equality only where both occurrences are live. Placement
/// transfers use ordinary array operations; no payload is read on the host to compare its coordinates.
fn check_equality<A>(
    value: &A,
    mask: Option<&A>,
    other: &A,
    other_mask: Option<&A>,
    axes: &[usize],
    message: &str,
    other_label: &str,
) -> Result<(), ProgramError>
where
    A: Value<Type = ArrayType> + Compare + And + Or + Not + Broadcast + Assert + TransferToMemory,
{
    let output_type = value.r#type().into_owned().with_layout(None);
    let align = |value: &A, data_type| -> Result<A, ProgramError> {
        let value = if value.r#type().memory() == output_type.memory() {
            value.clone()
        } else {
            value.transfer_to_memory(output_type.memory())?
        };
        value.broadcast(output_type.clone().with_data_type(data_type), axes)
    };
    let other = align(other, output_type.data_type())?;
    let other_mask = other_mask.map(|mask| align(mask, DataType::Boolean)).transpose()?;
    let live = match (mask, other_mask) {
        (None, None) => None,
        (None, Some(mask)) => Some(mask),
        (Some(mask), None) => Some(mask.clone()),
        (Some(mask), Some(other)) => Some(mask.and(&other)?),
    };
    assert_live(&value.equal(&other)?, live.as_ref(), message, &[("extent", value.clone()), (other_label, other)])
}

/// Maps corresponding extent axes through their extent coordinates. Structural validation proves either identical
/// explicit axes or a unique bijection of logical coordinate dimensions; physical capacities are never used as indexed
/// extent equality proof.
fn corresponding_axes(
    source_type: &RaggedType,
    source: &[usize],
    target_type: &RaggedType,
    target: &[usize],
) -> Vec<usize> {
    if source == target && source_type.logical_shape() == target_type.logical_shape() {
        return (0..source.len()).collect();
    }
    source
        .iter()
        .map(|axis| {
            target
                .iter()
                .position(|other| target_type.logical_shape()[*other] == source_type.logical_shape()[*axis])
                .unwrap()
        })
        .collect()
}

/// Computes live coordinates over the dependency closure, then existentially removes omitted coordinates. All sibling
/// predicates share the same completion: independent per-axis maxima would admit coordinate pairs that never coexist
/// in one item. Earlier sibling predicates also mask padding in their descendants' payloads before projection.
fn live_coordinate_mask<A>(
    r#type: &RaggedType,
    extent_axes: &[usize],
    target: &A,
    extents: &[A],
) -> Result<Option<A>, ProgramError>
where
    A: Value<Type = ArrayType, Domain: Iota<A>> + Compare + And + Broadcast + Reduce,
{
    if !r#type.ragged_dimensions().iter().any(|descriptor| extent_axes.contains(&descriptor.axis())) {
        return Ok(None);
    }

    let mut included = vec![false; r#type.storage_type().rank()];
    for &axis in extent_axes {
        included[axis] = true;
    }
    // Dependencies precede their consumers. Walking backward visits each newly included ancestor later in this pass.
    for descriptor in r#type.ragged_dimensions().iter().rev() {
        if included[descriptor.axis()] {
            for &axis in descriptor.extent_axes() {
                included[axis] = true;
            }
        }
    }
    let mut closure = extent_axes.to_vec();
    closure.extend((0..included.len()).filter(|axis| included[*axis] && !extent_axes.contains(axis)));
    let closure_type = project_type(r#type.storage_type(), &closure)?.with_data_type(DataType::I64);
    let mut mask: Option<A> = None;
    for (index, descriptor) in r#type.ragged_dimensions().iter().enumerate() {
        let Some(position) = closure.iter().position(|axis| *axis == descriptor.axis()) else { continue };
        let axes = descriptor
            .extent_axes()
            .iter()
            .map(|axis| closure.iter().position(|target| target == axis).unwrap())
            .collect::<Vec<_>>();
        let limits = extents[index].broadcast(closure_type.clone(), &axes)?;
        let iota_type = project_type(&closure_type, &[position])?;
        let iota = target.domain().iota(&iota_type, 0)?.broadcast(closure_type.clone(), &[position])?;
        let current = iota.less_than(&limits)?;
        mask = Some(match mask {
            None => current,
            Some(mask) => mask.and(&current)?,
        });
    }
    if closure.len() > extent_axes.len() {
        let omitted = (extent_axes.len()..closure.len()).collect::<Vec<_>>();
        mask = mask.map(|mask| mask.reduce_any(&omitted)).transpose()?;
    }
    Ok(mask)
}

/// Projects an ordinary array type onto selected dimensions without retaining a storage layout.
fn project_type(r#type: &ArrayType, axes: &[usize]) -> Result<ArrayType, TypeError> {
    let sharding = r#type
        .sharding()
        .map(|sharding| {
            sharding.with_dimensions(axes.iter().map(|axis| sharding.dimensions()[*axis].clone()).collect())
        })
        .transpose()
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    ArrayType::new(
        r#type.data_type(),
        crate::arrays::Shape::new(axes.iter().map(|axis| r#type.shape()[*axis].clone()).collect()),
    )
    .with_memory(r#type.memory())
    .with_sharding(sharding)
    .map_err(|error| TypeError::invalid(error.to_string()))
}

/// Creates a scalar literal with the payload's placement and mesh dependencies, without inspecting that payload.
fn integer_constant<A>(value: &A, integer: i64) -> Result<A, ProgramError>
where
    A: Value<Type = ArrayType, Domain: Context>,
    <A::Domain as Domain>::Constant: From<Array>,
{
    let literal = Array::scalar(integer)?;
    let literal = Array::new(value.r#type().scalar_like()?, literal.storage_bytes().to_vec())?;
    value.domain().lift(literal.into())
}

/// Validates an inclusive portable extent range, avoiding overflow at the largest representable exclusive bound.
fn check_bounds<A>(value: &A, dimension: &DimensionType, mask: Option<&A>, message: &str) -> Result<(), ProgramError>
where
    A: Value<Type = ArrayType, Domain: Context> + Compare + And + Or + Not + Assert,
    <A::Domain as Domain>::Constant: From<Array>,
{
    let (lower, maximum) = dimension.bounds().representable_extent_range()?;
    let lower = integer_constant(value, lower as i64)?;
    let maximum = integer_constant(value, maximum as i64)?;
    let condition = value.greater_than_or_equal(&lower)?.and(&value.less_than_or_equal(&maximum)?)?;
    assert_live(&condition, mask, message, &[("extent", value.clone())])
}

/// Ignores inactive padding while retaining the existing assertion diagnostics and ordered-effect semantics.
fn assert_live<A: Clone + Or + Not + Assert>(
    condition: &A,
    mask: Option<&A>,
    message: &str,
    observations: &[(&str, A)],
) -> Result<(), ProgramError> {
    let condition = match mask {
        None => condition.clone(),
        Some(mask) => mask.not()?.or(condition)?,
    };
    condition.assert_with_limit(message, observations, NonZeroUsize::new(8).unwrap())
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayOperation, Dimension, DimensionBounds, DimensionError, MAX_DIMENSION_EXTENT,
        RaggedDimension, Shape,
    };
    use crate::contexts::EagerContext;
    use crate::macros::check_operation_type_inference;
    use crate::operations::assertions::{AssertionError, AssertionFailure};
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::EmptyRegionDriver;
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    /// Composite tracer shared by the exact staged-program fixtures.
    type ArrayTracer = Tracer<TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>;

    /// Ordinary array tracer used inside the direct composite wrapper, independently of composite-native staging.
    type DenseTracer = Tracer<TracingContext<Array, ArrayOperation<Array>>>;

    /// Builds the nested representation used to distinguish live coordinates from padding.
    fn nested_type(batch: usize) -> RaggedType {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(5)).unwrap());
        RaggedType::new(
            ArrayType::new_static(DataType::F32, [batch, 3, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0]), RaggedDimension::new(2, columns, vec![0, 1])],
        )
        .unwrap()
    }

    /// Creates padded numerical storage without assigning meaning to its padding.
    fn storage(r#type: &RaggedType) -> ArrayIrValue<Array> {
        ArrayIrValue::Array(Array::from_fn_elements(r#type.storage_type().clone(), |_| Ok(17.0f32)).unwrap())
    }

    /// A flat indexed family with unequal per-item lengths, reused across AD protocol tests.
    fn differential_value() -> ArrayIrValue<Array> {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        let storage = Array::from_fn_elements::<f32, _>(r#type.storage_type().clone(), |_| Ok(2f32)).unwrap();
        ArrayIrValue::Ragged(RaggedValue::new(r#type, storage, vec![Array::vector(vec![1i64, 2]).unwrap()]).unwrap())
    }

    /// Exact cross-carrier failure for a supplied shape [0, 3] at the fixed linearization shape [1, 2].
    fn differential_shape_mismatch() -> AssertionError {
        AssertionError::FailedElements {
            message: "guarded ragged occurrences of `rows` must have equal live extents".to_owned(),
            failures: vec![
                AssertionFailure {
                    index: vec![0],
                    observations: vec![
                        ("extent".to_owned(), "1".to_owned()),
                        ("other_extent".to_owned(), "0".to_owned()),
                    ],
                },
                AssertionFailure {
                    index: vec![1],
                    observations: vec![
                        ("extent".to_owned(), "2".to_owned()),
                        ("other_extent".to_owned(), "3".to_owned()),
                    ],
                },
            ],
            omitted: 0,
        }
    }

    #[test]
    fn test_ragged_new() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        let operation = RaggedNewOperation::new(r#type.clone());
        assert_eq!(operation.name(), RAGGED_NEW_OPERATION_NAME);
        assert_eq!(operation.output_type(), &r#type);
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(operation.to_string(), "ragged_new [output_type=ragged<f32[2, 3]; 1:rows over [0]>]");

        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(storage, extents): (ArrayTracer, ArrayTracer)| {
                let value = storage.ragged_new(r#type.clone(), &[extents])?;
                Ok((value.ragged_storage()?, value.ragged_extents(-1)?))
            },
            (ArrayIrType::Array(r#type.storage_type().clone()), ArrayIrType::Array(r#type.extent_types()[0].clone())),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:i64[2] .
                let %2:ragged<f32[2, 3]; 1:rows over [0]> = ragged_new \
                        [output_type=ragged<f32[2, 3]; 1:rows over [0]>] %0 %1
                    %3:f32[2, 3] = ragged_storage %2
                    %4:i64[2] = ragged_extents [axis=-1] %2
                in (%3, %4)"},
        );
        assert_eq!(program.type_identity_signature().internal_identities(), &[rows]);
        let inputs = (storage(&r#type), ArrayIrValue::Array(Array::vector(vec![1i64, 3]).unwrap()));
        assert_eq!(program.interpret(inputs.clone()), Ok(inputs));
    }

    #[test]
    fn test_ragged_operation_identity_declarations() -> Result<(), TypeError> {
        let representation = nested_type(2);
        let identities = representation
            .ragged_dimensions()
            .iter()
            .map(|descriptor| descriptor.dimension().clone())
            .collect::<Vec<_>>();
        let mut fields = std::iter::once(representation.storage_type().clone().into())
            .chain(representation.extent_types().iter().cloned().map(Into::into))
            .collect::<Vec<ArrayIrType>>();
        let fresh = RaggedNewOperation::new(representation.clone());
        assert_eq!(fresh.defined_type_identities().as_ref(), &identities);
        assert_eq!(fresh.consumed_type_identities(&fields)?, Vec::new());

        fields.push(representation.clone().into());
        let reconstruction = RaggedNewOperation::with_identity_source(representation.clone());
        assert_eq!(reconstruction.consumed_type_identities(&fields)?, identities);
        let source = [ArrayIrType::Ragged(representation.clone())];
        let operations = [
            ArrayIrOperation::<Array>::from(RaggedStorageOperation),
            ArrayIrOperation::from(RaggedExtentsOperation::new(-1)),
            ArrayIrOperation::from(RaggedZeroOperation::new(representation)),
        ];
        for operation in operations {
            assert_eq!(operation.consumed_type_identities(&source)?, identities);
            assert_eq!(Box::new(operation).consumed_type_identities(&source)?, identities);
        }

        // A dimension item's represented identity is consumed even when there are no ragged storage axes.
        let item = DimensionVariable::new("item", DimensionBounds::unbounded());
        let dimension = RaggedType::new(
            ArrayType::new_static(DataType::I64, [2]),
            RaggedValueKind::Dimension(item.clone().into()),
            Vec::new(),
        )?;
        assert_eq!(RaggedStorageOperation.consumed_type_identities(&[dimension.into()])?, vec![item]);
        assert_eq!(
            RaggedStorageOperation.consumed_type_identities(&[]),
            Err(TypeError::invalid("expected 1 input but got 0"))
        );
        Ok(())
    }

    #[test]
    fn test_ragged_new_type_inference() {
        let r#type = nested_type(2);
        let operation = RaggedNewOperation::new(r#type.clone());
        let input_types = std::iter::once(r#type.storage_type())
            .chain(r#type.extent_types())
            .cloned()
            .map(ArrayIrType::Array)
            .collect::<Vec<_>>();
        assert_eq!(operation.infer_output_types(&input_types, &[]), Ok(vec![r#type.clone().into()]));
        assert_eq!(
            operation.infer_output_types(&input_types[..2], &[]),
            Err(TypeError::invalid("expected 3 inputs but got 2"))
        );

        // A dimension-only representation admits scalar and ordinary dense inputs without ragged descriptors.
        let scalar = RaggedType::new(
            ArrayType::scalar(DataType::I64),
            RaggedValueKind::Dimension(Dimension::Static(3).into()),
            vec![],
        )
        .unwrap();
        check_operation_type_inference!(
            operation = RaggedNewOperation::new(scalar.clone()),
            cases = [
                { input_types = [ArrayType::scalar(DataType::I64).into()], output_types = [scalar.into()] },
                {
                    input_types = [ArrayType::scalar(DataType::F32).into()],
                    error = "type f32[] does not refine declared type i64[]",
                },
            ],
        );
    }

    #[test]
    fn test_ragged_new_type_inference_dynamic_storage() {
        let batch = DimensionVariable::new("batch", DimensionBounds::non_negative(Some(5)).unwrap());
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let declared = RaggedType::new(
            ArrayType::new(DataType::F32, Shape::new(vec![batch.into(), Dimension::Static(3)])),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        let actual = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        let inputs =
            [ArrayType::new_static(DataType::F32, [2, 3]).into(), ArrayType::new_static(DataType::I64, [2]).into()];
        assert_eq!(RaggedNewOperation::new(declared).infer_output_types(&inputs, &[]), Ok(vec![actual.into()]));
    }

    #[test]
    fn test_ragged_new_interpretation() {
        let r#type = nested_type(2);
        let input = storage(&r#type);
        let rows = ArrayIrValue::Array(Array::vector(vec![1i64, 3]).unwrap());
        let columns = ArrayIrValue::Array(
            Array::from_elements(r#type.extent_types()[1].clone(), &[2i64, 999, -7, 1, 4, 0]).unwrap(),
        );
        let fields = [input.clone(), rows.clone(), columns.clone()];
        let direct = input.ragged_new(r#type.clone(), &[rows.clone(), columns.clone()]).unwrap();
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            context.bind(RaggedNewOperation::new(r#type.clone()), Vec::new(), &fields),
            Ok(vec![direct.clone()]),
        );
        assert_eq!(direct.ragged_storage(), Ok(input));
        assert_eq!(direct.ragged_extents(1), Ok(rows));
        assert_eq!(direct.ragged_extents(-1), Ok(columns));

        // The same columns become invalid when the changed row extents make padded coordinate [0, 1] live.
        let changed_rows = ArrayIrValue::Array(Array::vector(vec![3i64, 3]).unwrap());
        let error = fields[0].ragged_new(r#type, &[changed_rows, fields[2].clone()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged extent on storage axis 2 is outside its declared bounds".to_owned(),
                failures: vec![
                    AssertionFailure { index: vec![0, 1], observations: vec![("extent".to_owned(), "999".to_owned())] },
                    AssertionFailure { index: vec![0, 2], observations: vec![("extent".to_owned(), "-7".to_owned())] },
                ],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_new_interpretation_no_live_coordinates() {
        for batch in [0, 2] {
            let r#type = nested_type(batch);
            let rows = ArrayIrValue::Array(Array::vector(vec![0i64; batch]).unwrap());
            let columns =
                ArrayIrValue::Array(Array::from_fn_elements(r#type.extent_types()[1].clone(), |_| Ok(999i64)).unwrap());
            assert!(storage(&r#type).ragged_new(r#type, &[rows, columns]).is_ok());
        }
    }

    #[test]
    fn test_ragged_new_interpretation_omitted_ancestor() {
        // Columns vary over rows, while row liveness varies over a batch coordinate absent from those extent axes.
        // A row is live when at least one live batch item contains it.
        for (batch, rows, columns, expected_mask) in [
            (2, vec![1i64, 2], vec![4i64, 1, 999], vec![true, true, false]),
            (2, vec![0i64, 0], vec![999i64; 3], vec![false; 3]),
            (0, vec![], vec![999i64; 3], vec![false; 3]),
        ] {
            let rows_dimension = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
            let columns_dimension = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(5)).unwrap());
            let r#type = RaggedType::new(
                ArrayType::new_static(DataType::F32, [batch, 3, 4]),
                RaggedValueKind::Array,
                vec![
                    RaggedDimension::new(1, rows_dimension, vec![0]),
                    RaggedDimension::new(2, columns_dimension, vec![1]),
                ],
            )
            .unwrap();
            let rows = Array::vector(rows).unwrap();
            let columns = Array::vector(columns).unwrap();
            assert!(
                storage(&r#type)
                    .ragged_new(r#type.clone(), &[ArrayIrValue::Array(rows.clone()), ArrayIrValue::Array(columns)])
                    .is_ok()
            );
            if expected_mask.iter().any(|live| *live) {
                let error = storage(&r#type)
                    .ragged_new(r#type, &[rows.into(), Array::vector(vec![999i64; 3]).unwrap().into()])
                    .unwrap_err();
                assert_eq!(
                    error.downcast_custom::<AssertionError>(),
                    Some(&AssertionError::FailedElements {
                        message: "ragged extent on storage axis 2 is outside its declared bounds".to_owned(),
                        failures: expected_mask
                            .iter()
                            .enumerate()
                            .filter(|(_, live)| **live)
                            .map(|(index, _)| AssertionFailure {
                                index: vec![index],
                                observations: vec![("extent".to_owned(), "999".to_owned())],
                            })
                            .collect(),
                        omitted: 0,
                    }),
                );
            }
        }
    }

    #[test]
    fn test_ragged_new_interpretation_masked_ancestor() {
        // Omitting the row predicate before projecting columns' omitted coordinates would activate padded 999 entries.
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(5)).unwrap());
        let depth = DimensionVariable::new("depth", DimensionBounds::non_negative(Some(3)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3, 4, 2]),
            RaggedValueKind::Array,
            vec![
                RaggedDimension::new(1, rows, vec![0]),
                RaggedDimension::new(2, columns, vec![0, 1]),
                RaggedDimension::new(3, depth, vec![2]),
            ],
        )
        .unwrap();
        let rows = Array::vector(vec![1i64, 0]).unwrap();
        let columns = Array::from_elements(r#type.extent_types()[1].clone(), &[2i64, 999, 999, 999, 999, 999]).unwrap();
        let depth = Array::vector(vec![1i64, 2, 999, 999]).unwrap();
        assert!(storage(&r#type).ragged_new(r#type, &[rows.into(), columns.into(), depth.into()]).is_ok());
    }

    #[test]
    fn test_ragged_new_interpretation_correlated_siblings() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(4)).unwrap());
        let depth = DimensionVariable::new("depth", DimensionBounds::non_negative(Some(3)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3, 3, 2]),
            RaggedValueKind::Array,
            vec![
                RaggedDimension::new(1, rows, vec![0]),
                RaggedDimension::new(2, columns, vec![0]),
                RaggedDimension::new(3, depth, vec![1, 2]),
            ],
        )
        .unwrap();
        let rows = ArrayIrValue::Array(Array::vector(vec![1i64, 3]).unwrap());
        let columns = ArrayIrValue::Array(Array::vector(vec![3i64, 1]).unwrap());
        // Each marginal includes coordinate 2, but no common batch item contains (row 2, column 2).
        let depth = ArrayIrValue::Array(
            Array::from_elements(r#type.extent_types()[2].clone(), &[1i64, 2, 0, 1, 999, 999, 2, 999, 999]).unwrap(),
        );
        assert!(storage(&r#type).ragged_new(r#type.clone(), &[rows.clone(), columns.clone(), depth]).is_ok());
        let changed = ArrayIrValue::Array(
            Array::from_elements(r#type.extent_types()[2].clone(), &[1i64, 2, 0, 1, 999, 999, 999, 999, 999]).unwrap(),
        );
        let error = storage(&r#type).ragged_new(r#type, &[rows, columns, changed]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged extent on storage axis 3 is outside its declared bounds".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![2, 0],
                    observations: vec![("extent".to_owned(), "999".to_owned())],
                }],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_new_interpretation_repeated_identity() {
        let extent = DimensionVariable::new("extent", DimensionBounds::non_negative(Some(5)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, extent.clone(), vec![0]), RaggedDimension::new(2, extent, vec![0])],
        )
        .unwrap();
        let first = ArrayIrValue::Array(Array::vector(vec![1i64, 2]).unwrap());
        let other = ArrayIrValue::Array(Array::vector(vec![2i64, 1]).unwrap());
        assert!(storage(&r#type).ragged_new(r#type.clone(), &[first.clone(), first.clone()]).is_ok());
        let error = storage(&r#type).ragged_new(r#type, &[first, other]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged occurrences of `extent` must have equal live extents".to_owned(),
                failures: vec![
                    AssertionFailure {
                        index: vec![0],
                        observations: vec![
                            ("extent".to_owned(), "2".to_owned()),
                            ("other_extent".to_owned(), "1".to_owned())
                        ],
                    },
                    AssertionFailure {
                        index: vec![1],
                        observations: vec![
                            ("extent".to_owned(), "1".to_owned()),
                            ("other_extent".to_owned(), "2".to_owned())
                        ],
                    },
                ],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_new_interpretation_permuted_extent_axes() {
        let extent = DimensionVariable::new("extent", DimensionBounds::non_negative(Some(5)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3, 4, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(2, extent.clone(), vec![1, 0]), RaggedDimension::new(3, extent, vec![0, 1])],
        )
        .unwrap();
        let first = ArrayIrValue::Array(
            Array::from_elements(r#type.extent_types()[0].clone(), &[1i64, 3, 2, 4, 0, 1]).unwrap(),
        );
        let second = ArrayIrValue::Array(
            Array::from_elements(r#type.extent_types()[1].clone(), &[1i64, 2, 0, 3, 4, 1]).unwrap(),
        );
        assert!(storage(&r#type).ragged_new(r#type.clone(), &[first.clone(), second]).is_ok());
        let changed = ArrayIrValue::Array(
            Array::from_elements(r#type.extent_types()[1].clone(), &[1i64, 2, 0, 3, 4, 2]).unwrap(),
        );
        let error = storage(&r#type).ragged_new(r#type, &[first, changed]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged occurrences of `extent` must have equal live extents".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![1, 2],
                    observations: vec![
                        ("extent".to_owned(), "2".to_owned()),
                        ("other_extent".to_owned(), "1".to_owned())
                    ],
                }],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_new_interpretation_scalar_extent() {
        let extent = DimensionVariable::new("extent", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, extent, vec![])],
        )
        .unwrap();
        let extent = ArrayIrValue::Array(Array::scalar(2i64).unwrap());
        let value = storage(&r#type).ragged_new(r#type, &[extent.clone()]).unwrap();
        assert_eq!(value.ragged_extents(0), Ok(extent));
    }

    #[test]
    fn test_ragged_new_interpretation_identity_source_is_not_proof() {
        let r#type = nested_type(2);
        let ArrayIrValue::Array(storage) = storage(&r#type) else { unreachable!() };
        let rows = Array::vector(vec![1i64, 0]).unwrap();
        let columns = Array::from_elements(r#type.extent_types()[1].clone(), &[999i64; 6]).unwrap();
        // The public value constructor verifies structure only; a source of the right nominal type can be invalid.
        let source = ArrayIrValue::Ragged(
            RaggedValue::new(r#type.clone(), storage.clone(), vec![rows.clone(), columns.clone()]).unwrap(),
        );
        let error = ArrayIrValue::Array(storage)
            .ragged_new_with_identity_source(r#type.clone(), &[rows.into(), columns.into()], &source)
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged extent on storage axis 2 is outside its declared bounds".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![0, 0],
                    observations: vec![("extent".to_owned(), "999".to_owned())],
                }],
                omitted: 0,
            }),
        );
        let operation = RaggedNewOperation::with_identity_source(r#type);
        assert!(operation.has_identity_source());
        assert_eq!(
            operation.to_string(),
            indoc! {"
                ragged_new [
                    output_type=ragged<f32[2, 3, 4]; 1:rows over [0]; 2:columns over [0, 1]>,
                    identity_source=true,
                ]"},
        );
    }

    #[test]
    fn test_ragged_new_identity_source_numeric_equality() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        let source = storage(&r#type)
            .ragged_new(r#type.clone(), &[Array::vector(vec![1i64, 2]).unwrap().into()])
            .unwrap();
        let error = storage(&r#type)
            .ragged_new_with_identity_source(r#type, &[Array::vector(vec![3i64, 3]).unwrap().into()], &source)
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged identity source for `rows` must have equal live extents".to_owned(),
                failures: vec![
                    AssertionFailure {
                        index: vec![0],
                        observations: vec![
                            ("extent".to_owned(), "3".to_owned()),
                            ("source_extent".to_owned(), "1".to_owned())
                        ],
                    },
                    AssertionFailure {
                        index: vec![1],
                        observations: vec![
                            ("extent".to_owned(), "3".to_owned()),
                            ("source_extent".to_owned(), "2".to_owned())
                        ],
                    },
                ],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_new_identity_source_reordered_subset() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(5)).unwrap());
        let source_type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [3, 2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows.clone(), vec![1]), RaggedDimension::new(2, columns, vec![1])],
        )
        .unwrap();
        let extents = ArrayIrValue::Array(Array::vector(vec![1i64, 2]).unwrap());
        let source = storage(&source_type)
            .ragged_new(source_type.clone(), &[extents.clone(), Array::vector(vec![4i64, 0]).unwrap().into()])
            .unwrap();
        let output_type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        let output = storage(&output_type)
            .ragged_new_with_identity_source(output_type, &[extents.clone()], &source)
            .unwrap();
        assert_eq!(output.ragged_extents(1), Ok(extents));
    }

    #[test]
    fn test_ragged_new_identity_source_checks_ancestor_bounds() {
        let r#type = nested_type(2);
        let ArrayIrValue::Array(source_storage) = storage(&r#type) else { unreachable!() };
        let source = ArrayIrValue::Ragged(
            RaggedValue::new(
                r#type.clone(),
                source_storage,
                vec![
                    Array::vector(vec![-1i64, 0]).unwrap(),
                    Array::from_elements(r#type.extent_types()[1].clone(), &[999i64; 6]).unwrap(),
                ],
            )
            .unwrap(),
        );
        let error = storage(&r#type)
            .ragged_new_with_identity_source(
                r#type.clone(),
                &[
                    Array::vector(vec![0i64, 0]).unwrap().into(),
                    Array::from_elements(r#type.extent_types()[1].clone(), &[999i64; 6]).unwrap().into(),
                ],
                &source,
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged extent on storage axis 1 is outside its declared bounds".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![0],
                    observations: vec![("extent".to_owned(), "-1".to_owned())],
                }],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_new_identity_source_dimension_values() {
        let item = DimensionType::new("item", DimensionBounds::non_negative(Some(8)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::I64, [2]),
            RaggedValueKind::Dimension(item.clone()),
            vec![],
        )
        .unwrap();
        let source =
            ArrayIrValue::Array(Array::vector(vec![1i64, 2]).unwrap()).ragged_new(r#type.clone(), &[]).unwrap();
        let error = ArrayIrValue::Array(Array::vector(vec![3i64, 3]).unwrap())
            .ragged_new_with_identity_source(r#type, &[], &source)
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged identity source for `item` must have equal live dimension values".to_owned(),
                failures: vec![
                    AssertionFailure {
                        index: vec![0],
                        observations: vec![
                            ("extent".to_owned(), "3".to_owned()),
                            ("source_extent".to_owned(), "1".to_owned())
                        ],
                    },
                    AssertionFailure {
                        index: vec![1],
                        observations: vec![
                            ("extent".to_owned(), "3".to_owned()),
                            ("source_extent".to_owned(), "2".to_owned())
                        ],
                    },
                ],
                omitted: 0,
            }),
        );

        // The represented identity's extent axes, which span its whole storage, can also be permuted through their
        // extent coordinates.
        let source_type = RaggedType::new(
            ArrayType::new_static(DataType::I64, [2, 3]),
            RaggedValueKind::Dimension(item.clone()),
            vec![],
        )
        .unwrap();
        let source = ArrayIrValue::Array(
            Array::from_elements(source_type.storage_type().clone(), &[1i64, 2, 3, 4, 5, 6]).unwrap(),
        )
        .ragged_new(source_type, &[])
        .unwrap();
        let output_type =
            RaggedType::new(ArrayType::new_static(DataType::I64, [3, 2]), RaggedValueKind::Dimension(item), vec![])
                .unwrap();
        let output = ArrayIrValue::Array(
            Array::from_elements(output_type.storage_type().clone(), &[1i64, 4, 2, 5, 3, 6]).unwrap(),
        );
        assert_eq!(
            output.ragged_new_with_identity_source(output_type, &[], &source).unwrap().ragged_storage(),
            Ok(output)
        );
    }

    #[test]
    fn test_ragged_new_interpretation_dimension_values() {
        let unbounded = DimensionType::new("item", DimensionBounds::non_negative(None).unwrap());
        for (dimension, values) in [
            (DimensionType::from(Dimension::Static(3)), vec![3i64, 3]),
            (unbounded, vec![0i64, MAX_DIMENSION_EXTENT as i64]),
        ] {
            let r#type = RaggedType::new(
                ArrayType::new_static(DataType::I64, [2]),
                RaggedValueKind::Dimension(dimension),
                vec![],
            )
            .unwrap();
            let value = ArrayIrValue::Array(Array::vector(values).unwrap());
            assert_eq!(value.ragged_new(r#type, &[]).unwrap().ragged_storage(), Ok(value));
        }
        let scalar_type = RaggedType::new(
            ArrayType::scalar(DataType::I64),
            RaggedValueKind::Dimension(Dimension::Static(3).into()),
            vec![],
        )
        .unwrap();
        assert!(ArrayIrValue::Array(Array::scalar(3i64).unwrap()).ragged_new(scalar_type.clone(), &[]).is_ok());
        let error = ArrayIrValue::Array(Array::scalar(2i64).unwrap()).ragged_new(scalar_type, &[]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged dimension value is outside its declared bounds".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![],
                    observations: vec![("extent".to_owned(), "2".to_owned())]
                }],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_new_interpretation_masked_dimension_values() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::I64, [2, 3]),
            RaggedValueKind::Dimension(Dimension::Static(7).into()),
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        let rows = ArrayIrValue::Array(Array::vector(vec![1i64, 0]).unwrap());
        let value = ArrayIrValue::Array(
            Array::from_elements(r#type.storage_type().clone(), &[7i64, -99, -99, 999, 999, 999]).unwrap(),
        );
        assert_eq!(value.ragged_new(r#type, &[rows]).unwrap().ragged_storage(), Ok(value));
    }

    #[test]
    fn test_ragged_new_staged_dimension_live_coordinates() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let item = DimensionVariable::new("item", DimensionBounds::non_negative(Some(8)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::I64, [2, 3]),
            RaggedValueKind::Dimension(item.clone().into()),
            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(storage, extents): (ArrayTracer, ArrayTracer)| storage.ragged_new(r#type.clone(), &[extents]),
            (ArrayIrType::Array(r#type.storage_type().clone()), ArrayIrType::Array(r#type.extent_types()[0].clone())),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:i64[2, 3], %1:i64[2] .
                let %2:ragged<i64[2, 3]; item=dimension<item ∈ [0, 8)>; 1:rows over [0]> = ragged_new \
                    [output_type=ragged<i64[2, 3]; item=dimension<item ∈ [0, 8)>; 1:rows over [0]>] %0 %1
                in (%2)"},
        );
        assert_eq!(program.type_identity_signature().internal_identities(), &[rows, item]);
        let values = Array::from_elements(r#type.storage_type().clone(), &[3i64, 999, -99, 999, 999, 999]).unwrap();
        let extents = Array::vector(vec![1i64, 0]).unwrap();
        let expected = ArrayIrValue::Ragged(RaggedValue::new(r#type, values.clone(), vec![extents.clone()]).unwrap());
        assert_eq!(program.interpret((values.into(), extents.into())), Ok(expected));
    }

    #[test]
    fn test_ragged_new_checked_reconstruction_retains_validation() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(storage, extents): (ArrayTracer, ArrayTracer)| {
                let checked = storage.ragged_new(r#type.clone(), &[extents])?;
                checked.ragged_storage()?.ragged_new_with_identity_source(
                    r#type.clone(),
                    &[checked.ragged_extents(1)?],
                    &checked,
                )
            },
            (ArrayIrType::Array(r#type.storage_type().clone()), ArrayIrType::Array(r#type.extent_types()[0].clone())),
        )
        .unwrap();
        let program = program.into_simplified().unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:i64[2] .
                let %2:ragged<f32[2, 3]; 1:rows over [0]> = ragged_new \
                    [output_type=ragged<f32[2, 3]; 1:rows over [0]>] %0 %1
                    %3:f32[2, 3] = ragged_storage %2
                    %4:i64[2] = ragged_extents [axis=1] %2
                    %5:ragged<f32[2, 3]; 1:rows over [0]> = ragged_new \
                        [output_type=ragged<f32[2, 3]; 1:rows over [0]>, identity_source=true] %3 %4 %2
                in (%5)"},
        );
        let error = program
            .interpret((storage(&r#type), ArrayIrValue::Array(Array::vector(vec![999i64, 2]).unwrap())))
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged extent on storage axis 1 is outside its declared bounds".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![0],
                    observations: vec![("extent".to_owned(), "999".to_owned())],
                }],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_new_reconstruction_retains_ordinary_relationship() {
        let n = DimensionVariable::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let producer_type = RaggedType::new(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![n.clone().into(), Dimension::Static(2), Dimension::Static(3)]),
            ),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(2, rows.clone(), vec![0])],
        )
        .unwrap();
        let restricted_type = RaggedType::new(
            ArrayType::new(DataType::F32, Shape::new(vec![n.clone().into(), n.into(), Dimension::Static(3)])),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(2, rows, vec![0])],
        )
        .unwrap();
        let operation = RaggedNewOperation::with_identity_source(restricted_type.clone());
        let fields = [
            ArrayIrType::Array(producer_type.storage_type().clone()),
            ArrayIrType::Array(producer_type.extent_types()[0].clone()),
            ArrayIrType::Ragged(producer_type.clone()),
        ];
        assert_eq!(operation.infer_output_types(&fields, &[]), Ok(vec![producer_type.clone().into()]));
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(storage, extents): (ArrayTracer, ArrayTracer)| {
                let checked = storage.ragged_new(producer_type.clone(), &[extents])?;
                checked.ragged_storage()?.ragged_new_with_identity_source(
                    restricted_type.clone(),
                    &[checked.ragged_extents(2)?],
                    &checked,
                )
            },
            (
                ArrayIrType::Array(producer_type.storage_type().clone()),
                ArrayIrType::Array(producer_type.extent_types()[0].clone()),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[n, 2, 3], %1:i64[n] .
                let %2:ragged<f32[n, 2, 3]; 2:rows over [0]> = ragged_new \
                    [output_type=ragged<f32[n, 2, 3]; 2:rows over [0]>] %0 %1
                    %3:f32[n, 2, 3] = ragged_storage %2
                    %4:i64[n] = ragged_extents [axis=2] %2
                    %5:ragged<f32[n, 2, 3]; 2:rows over [0]> = ragged_new \
                        [output_type=ragged<f32[n, n, 3]; 2:rows over [0]>, identity_source=true] %3 %4 %2
                in (%5)"},
        );
        let valid_storage = ArrayIrValue::Array(
            Array::from_fn_elements(ArrayType::new_static(DataType::F32, [2, 2, 3]), |_| Ok(1f32)).unwrap(),
        );
        assert!(program.interpret((valid_storage, Array::vector(vec![1i64; 2]).unwrap().into())).is_ok());
        let invalid_storage = ArrayIrValue::Array(
            Array::from_fn_elements(ArrayType::new_static(DataType::F32, [3, 2, 3]), |_| Ok(1f32)).unwrap(),
        );
        assert_eq!(
            program.interpret((invalid_storage, Array::vector(vec![1i64; 3]).unwrap().into())),
            Err(ProgramError::Type(TypeError::from(DimensionError::InputDimensionMismatch {
                dimension: "n".to_owned(),
                expected: 3,
                actual: 2,
            }))),
        );
    }

    #[test]
    fn test_ragged_new_opaque_reconstruction() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |source: ArrayTracer| {
                source.ragged_storage()?.ragged_new_with_identity_source(
                    r#type.clone(),
                    &[source.ragged_extents(1)?],
                    &source,
                )
            },
            ArrayIrType::Ragged(r#type.clone()),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ragged<f32[2, 3]; 1:rows over [0]> .
                let %1:f32[2, 3] = ragged_storage %0
                    %2:i64[2] = ragged_extents [axis=1] %0
                    %3:ragged<f32[2, 3]; 1:rows over [0]> = ragged_new \
                        [output_type=ragged<f32[2, 3]; 1:rows over [0]>, identity_source=true] %1 %2 %0
                in (%3)"},
        );
        let ArrayIrValue::Array(storage) = storage(&r#type) else { unreachable!() };
        let invalid = ArrayIrValue::Ragged(
            RaggedValue::new(r#type, storage, vec![Array::vector(vec![999i64, 1]).unwrap()]).unwrap(),
        );
        let error = program.interpret(invalid).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged extent on storage axis 1 is outside its declared bounds".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![0],
                    observations: vec![("extent".to_owned(), "999".to_owned())],
                }],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_new_partial_evaluation_pending_checks() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(storage, extents): (ArrayTracer, ArrayTracer)| {
                storage.ragged_new(r#type.clone(), &[extents])?.ragged_storage()
            },
            (ArrayIrType::Array(r#type.storage_type().clone()), ArrayIrType::Array(r#type.extent_types()[0].clone())),
        )
        .unwrap();
        let program = program.to_flat_program();
        let known_storage = storage(&r#type);
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(known_storage.clone()),
                PartialValue::Unknown(r#type.extent_types()[0].clone().into()),
            ])
            .unwrap();
        assert!(matches!(evaluation.outputs(), [PartialEvaluationOutput::Unknown(0)]));
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:i64[2], %1:f32[2, 3] .
                let %2:ragged<f32[2, 3]; 1:rows over [0]> = ragged_new \
                        [output_type=ragged<f32[2, 3]; 1:rows over [0]>] %1 %0
                    %3:f32[2, 3] = ragged_storage %2
                in (%3)"},
        );
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            evaluation.interpret(&context, &[ArrayIrValue::Array(Array::vector(vec![1i64, 3]).unwrap())]),
            Ok(vec![known_storage]),
        );
        let error = evaluation
            .interpret(&context, &[ArrayIrValue::Array(Array::vector(vec![999i64, 3]).unwrap())])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged extent on storage axis 1 is outside its declared bounds".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![0],
                    observations: vec![("extent".to_owned(), "999".to_owned())],
                }],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_new_batching() {
        use crate::arrays::ArrayIrBatchingPolicy;
        use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer, batch};

        type Eager = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        type Batched = BatchingTracer<Eager, ArrayIrBatchingPolicy>;
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let item_type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows.clone(), vec![])],
        )
        .unwrap();
        let leading_type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        let extents = Array::vector(vec![1i64, 3]).unwrap();

        // Mapped storage with mapped extents inserts the batch coordinate into the storage and into the extent axes.
        let storage =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32, 2., 3., 4., 5., 6.]).unwrap();
        assert_eq!(
            batch(
                |(storage, extents): (Batched, Batched)| {
                    let value = storage.ragged_new(item_type.clone(), &[extents])?;
                    assert_eq!(value.batch().value().r#type().as_ref(), &ArrayIrType::Ragged(leading_type.clone()));
                    Ok((value.ragged_storage()?, value.ragged_extents(0)?))
                },
                (ArrayIrValue::Array(storage.clone()), ArrayIrValue::Array(extents.clone())),
                (BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxis::new(0),
                BatchAxisSpecification::default(),
            ),
            Ok((ArrayIrValue::Array(storage), ArrayIrValue::Array(extents.clone()))),
        );

        // Replicated storage with mapped extents denotes the same padded storage for every batch item, so it is
        // broadcast along a leading batch axis instead of being rejected.
        let shared = Array::vector(vec![1f32, 2., 3.]).unwrap();
        let broadcast =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32, 2., 3., 1., 2., 3.]).unwrap();
        assert_eq!(
            batch(
                |(storage, extents): (Batched, Batched)| {
                    let value = storage.ragged_new(item_type.clone(), &[extents])?;
                    assert_eq!(value.batch().value().r#type().as_ref(), &ArrayIrType::Ragged(leading_type.clone()));
                    Ok((value.ragged_storage()?, value.ragged_extents(0)?))
                },
                (ArrayIrValue::Array(shared.clone()), ArrayIrValue::Array(extents.clone())),
                (BatchAxis::replicated(), BatchAxis::new(0)),
                BatchAxis::new(0),
                BatchAxisSpecification::default(),
            ),
            Ok((ArrayIrValue::Array(broadcast), ArrayIrValue::Array(extents.clone()))),
        );

        // A mapped identity source places broadcast storage at its own batch axis, and the rule reports the source's
        // `columns` identity as consumed; the output does not carry it, so conservation validation would otherwise
        // reject the rule.
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(5)).unwrap());
        let source_type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [3, 2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows.clone(), vec![1]), RaggedDimension::new(2, columns, vec![1])],
        )
        .unwrap();
        let source = RaggedValue::new(
            source_type.clone(),
            Array::from_fn_elements(source_type.storage_type().clone(), |_| Ok(17f32)).unwrap(),
            vec![extents.clone(), Array::vector(vec![4i64, 0]).unwrap()],
        )
        .unwrap();
        let trailing_type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [3, 2]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows, vec![1])],
        )
        .unwrap();
        let transposed =
            Array::from_elements(ArrayType::new_static(DataType::F32, [3, 2]), &[1f32, 1., 2., 2., 3., 3.]).unwrap();
        assert_eq!(
            batch(
                |(storage, source): (Batched, Batched)| {
                    let extents = source.ragged_extents(0)?;
                    let value = storage.ragged_new_with_identity_source(item_type.clone(), &[extents], &source)?;
                    assert_eq!(value.batch().value().r#type().as_ref(), &ArrayIrType::Ragged(trailing_type.clone()));
                    Ok((value.ragged_storage()?, value.ragged_extents(0)?))
                },
                (ArrayIrValue::Array(shared), ArrayIrValue::Ragged(source)),
                (BatchAxis::replicated(), BatchAxis::new(1)),
                (BatchAxis::new(1), BatchAxis::new(0)),
                BatchAxisSpecification::default(),
            ),
            Ok((ArrayIrValue::Array(transposed), ArrayIrValue::Array(extents))),
        );

        // A non-leading storage batch axis shifts the occurrence's storage axis, and an extent payload mapped at a
        // non-leading position inserts the batch coordinate at that position of the occurrence's extent axes.
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(4)).unwrap());
        let item_type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, columns.clone(), vec![0])],
        )
        .unwrap();
        let packed_type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(2, columns, vec![0, 1])],
        )
        .unwrap();
        let storage = Array::from_fn_elements(packed_type.storage_type().clone(), |_| Ok(17f32)).unwrap();
        let extents =
            Array::from_elements(ArrayType::new_static(DataType::I64, [2, 4]), &[0i64, 1, 2, 3, 3, 2, 1, 0]).unwrap();
        assert_eq!(
            batch(
                |(storage, extents): (Batched, Batched)| {
                    let value = storage.ragged_new(item_type.clone(), &[extents])?;
                    assert_eq!(value.batch().value().r#type().as_ref(), &ArrayIrType::Ragged(packed_type.clone()));
                    Ok((value.ragged_storage()?, value.ragged_extents(1)?))
                },
                (ArrayIrValue::Array(storage.clone()), ArrayIrValue::Array(extents.clone())),
                (BatchAxis::new(1), BatchAxis::new(1)),
                BatchAxis::new(1),
                BatchAxisSpecification::default(),
            ),
            Ok((ArrayIrValue::Array(storage), ArrayIrValue::Array(extents))),
        );
    }

    #[test]
    fn test_ragged_new_direct_wrapper_retains_checks() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows, vec![])],
        )
        .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(storage, extent): (DenseTracer, DenseTracer)| {
                let value = ArrayIrValue::Array(storage).ragged_new(r#type.clone(), &[ArrayIrValue::Array(extent)])?;
                <ArrayIrValue<DenseTracer> as ValueProjection<ArrayType>>::into_projected(value.ragged_storage()?)
                    .map_err(Into::into)
            },
            (r#type.storage_type().clone(), r#type.extent_types()[0].clone()),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3], %1:i64[] .
                let %2:i64[] = const 0
                    %3:i64[] = const 3
                    %4:bool[] = compare [direction=GreaterThanOrEqual] %1 %2
                    %5:bool[] = compare [direction=LessThanOrEqual] %1 %3
                    %6:bool[] = and %4 %5
                    () = assert [
                        message=\"ragged extent on storage axis 0 is outside its declared bounds\",
                        labels=[\"extent\"],
                        failure_limit=8,
                    ] %6 %1
                    %7:f32[3], %8:i64[] = guard [boundary=\"ragged_new\", declared_types=[f32[3], i64[]]] %0 %1
                in (%7)"},
        );
        assert!(matches!(program.instructions().last().unwrap().operation(), ArrayOperation::Guard(_)));
        let storage = Array::vector(vec![1f32, 2., 3.]).unwrap();
        assert_eq!(program.interpret((storage.clone(), Array::scalar(2i64).unwrap())), Ok(storage.clone()));
        let error = program.interpret((storage.clone(), Array::scalar(999i64).unwrap())).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged extent on storage axis 0 is outside its declared bounds".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![],
                    observations: vec![("extent".to_owned(), "999".to_owned())],
                }],
                omitted: 0,
            }),
        );

        let evaluation = program
            .to_flat_program()
            .partially_evaluate(&[
                PartialValue::Known(storage.clone()),
                PartialValue::Unknown(ArrayType::scalar(DataType::I64)),
            ])
            .unwrap();
        assert!(matches!(evaluation.outputs(), [PartialEvaluationOutput::Unknown(0)]));
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:i64[], %1:f32[3] .
                let %2:i64[] = const 0
                    %3:bool[] = compare [direction=GreaterThanOrEqual] %0 %2
                    %4:i64[] = const 3
                    %5:bool[] = compare [direction=LessThanOrEqual] %0 %4
                    %6:bool[] = and %3 %5
                    () = assert [
                        message=\"ragged extent on storage axis 0 is outside its declared bounds\",
                        labels=[\"extent\"],
                        failure_limit=8,
                    ] %6 %0
                    %7:f32[3], %8:i64[] = guard [boundary=\"ragged_new\", declared_types=[f32[3], i64[]]] %1 %0
                in (%7)"},
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(evaluation.interpret(&context, &[Array::scalar(2i64).unwrap()]), Ok(vec![storage]));
        let error = evaluation.interpret(&context, &[Array::scalar(999i64).unwrap()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged extent on storage axis 0 is outside its declared bounds".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![],
                    observations: vec![("extent".to_owned(), "999".to_owned())],
                }],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_new_direct_wrapper_rejects_unresolved_ordinary_relationship() {
        let n = DimensionVariable::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let storage_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![n.clone().into(), Dimension::Static(2), Dimension::Static(3)]),
        );
        let declared = RaggedType::new(
            storage_type.clone().with_shape(Shape::new(vec![n.clone().into(), n.into(), Dimension::Static(3)])),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(2, rows, vec![0])],
        )
        .unwrap();
        let error = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(storage, extent): (DenseTracer, DenseTracer)| -> Result<DenseTracer, ProgramError> {
                let value =
                    ArrayIrValue::Array(storage).ragged_new(declared.clone(), &[ArrayIrValue::Array(extent)])?;
                <ArrayIrValue<DenseTracer> as ValueProjection<ArrayType>>::into_projected(value.ragged_storage()?)
                    .map_err(Into::into)
            },
            (storage_type, declared.extent_types()[0].clone()),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::UnsupportedOperation {
                message: "direct ragged construction cannot enforce unresolved ordinary dimension relationships; bind \
                `ragged_new` in a composite array IR context"
                    .to_owned(),
            }
        );
    }

    #[test]
    fn test_ragged_storage_and_extents_type_inference() {
        let r#type = nested_type(2);
        check_operation_type_inference!(
            operation = RaggedStorageOperation,
            cases = [
                { input_types = [r#type.clone().into()], output_types = [r#type.storage_type().clone().into()] },
                { input_types = [ArrayType::scalar(DataType::I64).into()], error = "expected ragged type but got array type" },
            ],
        );
        assert_eq!(RaggedStorageOperation.to_string(), "ragged_storage");
        let operation = RaggedExtentsOperation::new(-1);
        assert_eq!(operation.axis(), Axis::from(-1));
        assert_eq!(operation.to_string(), "ragged_extents [axis=-1]");
        assert_eq!(
            operation.infer_output_types(&[r#type.clone().into()], &[]),
            Ok(vec![r#type.extent_types()[1].clone().into()])
        );
        assert_eq!(
            RaggedExtentsOperation::new(0).infer_output_types(&[r#type.clone().into()], &[]),
            Err(TypeError::invalid("`ragged_extents` storage axis 0 has no ragged occurrence")),
        );
        assert_eq!(
            RaggedExtentsOperation::new(3).infer_output_types(&[r#type.into()], &[]),
            Err(TypeError::invalid("axis 3 is out of bounds for rank 3")),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::interpret(
                &RaggedStorageOperation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
    }

    #[test]
    fn test_ragged_extents_batching() {
        use crate::arrays::ArrayIrBatchingPolicy;
        use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer, batch};

        type Eager = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        type Batched = BatchingTracer<Eager, ArrayIrBatchingPolicy>;
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(5)).unwrap());

        // Each per-item `f32[3, 4]` value is mapped along a trailing storage axis. Its `rows` extent is a scalar per
        // item, while its `columns` extents vary over storage axis 0 and so carry the batch coordinate second.
        let packed_type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [3, 4, 2]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows, vec![2]), RaggedDimension::new(1, columns, vec![0, 2])],
        )
        .unwrap();
        let row_extents = Array::vector(vec![2i64, 3]).unwrap();
        let column_extents =
            Array::from_elements(ArrayType::new_static(DataType::I64, [3, 2]), &[4i64, 1, 0, 2, 3, 4]).unwrap();
        let value = ArrayIrValue::Ragged(
            RaggedValue::new(
                packed_type.clone(),
                Array::from_fn_elements(packed_type.storage_type().clone(), |_| Ok(17f32)).unwrap(),
                vec![row_extents.clone(), column_extents],
            )
            .unwrap(),
        );

        // Negative axes resolve against the per-item rank 2, so `-1` selects `columns` rather than the packed batch
        // axis. Its output axis is the batch coordinate's position in the extent axes, so leading materialization
        // transposes the payload into per-item rows.
        let per_item_columns =
            Array::from_elements(ArrayType::new_static(DataType::I64, [2, 3]), &[4i64, 0, 3, 1, 2, 4]).unwrap();
        assert_eq!(
            batch(
                |value: Batched| value.ragged_extents(-1),
                value.clone(),
                BatchAxis::new(2),
                BatchAxis::new(0),
                BatchAxisSpecification::default(),
            ),
            Ok(ArrayIrValue::Array(per_item_columns)),
        );
        assert_eq!(
            batch(
                |value: Batched| value.ragged_extents(-2),
                value,
                BatchAxis::new(2),
                BatchAxis::new(0),
                BatchAxisSpecification::default(),
            ),
            Ok(ArrayIrValue::Array(row_extents.clone())),
        );

        // A replicated ragged value has the same extents for every batch item, so its extents stay replicated even
        // when another input is mapped.
        let item_type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [3, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, packed_type.ragged_dimensions()[0].dimension().clone(), vec![])],
        )
        .unwrap();
        let replicated = ArrayIrValue::Ragged(
            RaggedValue::new(
                item_type.clone(),
                Array::from_fn_elements(item_type.storage_type().clone(), |_| Ok(17f32)).unwrap(),
                vec![Array::scalar(2i64).unwrap()],
            )
            .unwrap(),
        );
        assert_eq!(
            batch(
                |(value, _): (Batched, Batched)| value.ragged_extents(0),
                (replicated, ArrayIrValue::Array(row_extents)),
                (BatchAxis::replicated(), BatchAxis::new(0)),
                BatchAxis::replicated(),
                BatchAxisSpecification::default(),
            ),
            Ok(ArrayIrValue::Array(Array::scalar(2i64).unwrap())),
        );
    }

    #[test]
    fn test_ragged_zero() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(3)).unwrap());
        let source_type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows.clone(), Vec::new())],
        )
        .unwrap();
        let zero_type = RaggedType::new(
            ArrayType::new_static(DataType::Zero, [2]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows.clone(), Vec::new())],
        )
        .unwrap();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            move |source: ArrayTracer| {
                source
                    .domain()
                    .bind(RaggedZeroOperation::new(zero_type), Vec::new(), &[source.clone()])
                    .map(|mut values| values.remove(0))
            },
            ArrayIrType::Ragged(source_type.clone()),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:ragged<f32[2]; 0:rows over []> .
            let %1:ragged<zero[2]; 0:rows over []> = ragged_zero [output_type=ragged<zero[2]; 0:rows over []>] %0
            in (%1)"}
        );
        let source = ArrayIrValue::Ragged(
            RaggedValue::new(source_type, Array::vector(vec![1f32, 2.]).unwrap(), vec![Array::scalar(1i64).unwrap()])
                .unwrap(),
        );
        let output = program.interpret(source).unwrap();
        let ArrayIrValue::Ragged(output) = output else { panic!("expected ragged zero") };
        assert!(output.storage().storage_bytes().is_empty());
        assert_eq!(output.storage().r#type().shape(), &Shape::new(vec![Dimension::Static(2)]));
    }

    #[test]
    fn test_ragged_zero_type_inference() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(3)).unwrap());
        let source = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows.clone(), Vec::new())],
        )
        .unwrap();
        let output = RaggedType::new(
            ArrayType::new_static(DataType::Zero, [2]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows.clone(), Vec::new())],
        )
        .unwrap();
        check_operation_type_inference!(operation = RaggedZeroOperation::new(output.clone()), cases = [
            { input_types = [source.clone().into()], output_types = [output.into()] },
            { input_types = [], error = "expected 1 input but got 0" },
            { input_types = [ArrayType::new_static(DataType::F32, [2]).into()], error = "expected ragged type but got array type" },
        ]);
    }

    #[test]
    fn test_ragged_add() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(3)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows.clone(), Vec::new())],
        )
        .unwrap();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(left, right): (ArrayTracer, ArrayTracer)| {
                left.domain()
                    .bind(RaggedAddOperation, Vec::new(), &[left.clone(), right])
                    .map(|mut values| values.remove(0))
            },
            (ArrayIrType::Ragged(r#type.clone()), ArrayIrType::Ragged(r#type.clone())),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:ragged<f32[2]; 0:rows over []>, %1:ragged<f32[2]; 0:rows over []> .
            let %2:ragged<f32[2]; 0:rows over []> = ragged_add %0 %1
            in (%2)"}
        );
        let value = ArrayIrValue::Ragged(
            RaggedValue::new(
                r#type.clone(),
                Array::vector(vec![1f32, 2.]).unwrap(),
                vec![Array::scalar(1i64).unwrap()],
            )
            .unwrap(),
        );
        assert_eq!(
            program.interpret((value.clone(), value)),
            Ok(ArrayIrValue::Ragged(
                RaggedValue::new(r#type, Array::vector(vec![2f32, 4.]).unwrap(), vec![Array::scalar(1i64).unwrap()])
                    .unwrap()
            ))
        );
    }

    #[test]
    fn test_ragged_add_type_inference() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(3)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows.clone(), Vec::new())],
        )
        .unwrap();
        check_operation_type_inference!(operation = RaggedAddOperation, cases = [
            { input_types = [r#type.clone().into(), r#type.clone().into()], output_types = [r#type.into()] },
            { input_types = [], error = "expected 2 inputs but got 0" },
        ]);
    }

    #[test]
    fn test_ragged_zero_and_cotangent_add() {
        let value = differential_value();
        let r#type = <&RaggedType>::try_from(value.r#type().as_ref()).unwrap().clone();
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let zero = context
            .bind(RaggedZeroOperation::new(r#type.clone()), Vec::new(), &[value.clone()])
            .unwrap()
            .remove(0);
        assert_eq!(
            zero.ragged_storage().unwrap(),
            Array::from_elements(r#type.storage_type().clone(), &[0f32; 6]).unwrap().into()
        );
        assert_eq!(zero.ragged_extents(1), value.ragged_extents(1));
        let operation =
            ArrayIrOperation::<Array>::cotangent_add(&[&r#type.clone().into(), &r#type.clone().into()]).unwrap();
        assert!(matches!(operation, ArrayIrOperation::RaggedAdd(_)));
        let sum = context.bind(operation, Vec::new(), &[value.clone(), value.clone()]).unwrap().remove(0);
        assert_eq!(
            sum.ragged_storage().unwrap(),
            Array::from_elements(r#type.storage_type().clone(), &[4f32; 6]).unwrap().into()
        );
        assert_eq!(sum.ragged_extents(1), value.ragged_extents(1));
        // A nominal indexed identity never substitutes for numeric equality during accumulation.
        let changed = ArrayIrValue::Ragged(
            RaggedValue::new(
                r#type.clone(),
                Array::from_elements(r#type.storage_type().clone(), &[2f32; 6]).unwrap(),
                vec![Array::vector(vec![2i64, 2]).unwrap()],
            )
            .unwrap(),
        );
        let error = context.bind(RaggedAddOperation, Vec::new(), &[value, changed]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged identity source for `rows` must have equal live extents".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![0],
                    observations: vec![
                        ("extent".to_owned(), "1".to_owned()),
                        ("source_extent".to_owned(), "2".to_owned())
                    ]
                }],
                omitted: 0,
            })
        );
    }

    #[test]
    fn test_ragged_complex_zero_and_add() {
        use num_complex::Complex;

        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::C64, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        let elements = [Complex::new(1f32, 2.), Complex::new(-3., 4.), Complex::new(5., -6.)];
        let elements = [elements[0], elements[1], elements[2], elements[0], elements[1], elements[2]];
        let value = ArrayIrValue::Ragged(
            RaggedValue::new(
                r#type.clone(),
                Array::from_elements(r#type.storage_type().clone(), &elements).unwrap(),
                vec![Array::vector(vec![1i64, 2]).unwrap()],
            )
            .unwrap(),
        );
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let zero = context
            .bind(RaggedZeroOperation::new(r#type.clone()), Vec::new(), &[value.clone()])
            .unwrap()
            .remove(0);
        let sum = context.bind(RaggedAddOperation, Vec::new(), &[value.clone(), value.clone()]).unwrap().remove(0);
        let ArrayIrValue::Array(zero_storage) = zero.ragged_storage().unwrap() else { unreachable!() };
        let ArrayIrValue::Array(sum_storage) = sum.ragged_storage().unwrap() else { unreachable!() };
        assert_eq!(zero_storage.elements::<Complex<f32>>(), Ok(vec![Complex::new(0f32, 0.); 6]));
        assert_eq!(
            sum_storage.elements::<Complex<f32>>(),
            Ok(elements.iter().map(|element| element + element).collect())
        );
        assert_eq!(zero.ragged_extents(1), value.ragged_extents(1));
        assert_eq!(sum.ragged_extents(1), value.ragged_extents(1));
    }

    #[test]
    fn test_ragged_zero_and_add_manual_variation_and_memory() {
        use crate::arrays::{LogicalMesh, Memory, MeshAxis, MeshAxisType, Sharding};
        use crate::differentiation::DifferentiableType;

        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2).with_varying_manual_axes(["m"]).unwrap();
        let storage_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(sharding)
            .unwrap();
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type =
            RaggedType::new(storage_type.clone(), RaggedValueKind::Array, vec![RaggedDimension::new(1, rows, vec![0])])
                .unwrap();
        let extent = Array::from_elements(r#type.extent_types()[0].clone(), &[1i64, 2]).unwrap();
        let value = ArrayIrValue::Ragged(
            RaggedValue::new(
                r#type.clone(),
                Array::from_elements(storage_type.clone(), &[2f32; 6]).unwrap(),
                vec![extent.clone()],
            )
            .unwrap(),
        );
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let zero = context
            .bind(RaggedZeroOperation::new(r#type.clone()), Vec::new(), &[value.clone()])
            .unwrap()
            .remove(0);
        let sum = context.bind(RaggedAddOperation, Vec::new(), &[value.clone(), value]).unwrap().remove(0);
        assert_eq!(zero.ragged_storage(), Ok(Array::from_elements(storage_type.clone(), &[0f32; 6]).unwrap().into()));
        assert_eq!(sum.ragged_storage(), Ok(Array::from_elements(storage_type.clone(), &[4f32; 6]).unwrap().into()));
        assert_eq!(zero.ragged_extents(1), Ok(extent.clone().into()));
        assert_eq!(sum.ragged_extents(1), Ok(extent.into()));
        let tangent = ArrayIrType::Ragged(r#type.clone()).tangent().unwrap();
        let cotangent = ArrayIrType::Ragged(r#type.clone()).cotangent().unwrap();
        assert_eq!(tangent, ArrayIrType::Ragged(r#type.clone()));
        assert_eq!(cotangent, ArrayIrType::Ragged(r#type));
    }

    #[test]
    fn test_ragged_zero_shape_residual_preserves_occurrences() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3, 3, 3]),
            RaggedValueKind::Array,
            vec![
                RaggedDimension::new(1, rows, vec![0]),
                RaggedDimension::new(2, columns.clone(), vec![0, 1]),
                RaggedDimension::new(3, columns, vec![0, 1]),
            ],
        )
        .unwrap();
        let first = Array::from_elements(r#type.extent_types()[1].clone(), &[1i64, 999, -7, 2, 1, 999]).unwrap();
        let second = Array::from_elements(r#type.extent_types()[2].clone(), &[1i64, -3, 999, 2, 1, -7]).unwrap();
        let source = ArrayIrValue::Ragged(
            RaggedValue::new(
                r#type.clone(),
                Array::from_elements(r#type.storage_type().clone(), &[7f32; 54]).unwrap(),
                vec![Array::vector(vec![1i64, 2]).unwrap(), first.clone(), second.clone()],
            )
            .unwrap(),
        );
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let residual =
            ArrayIrOperation::<Array>::capture_zero_source(&context, &source, &r#type.clone().into()).unwrap();
        let ArrayIrValue::Ragged(shape) = &residual else { panic!("expected ragged shape carrier") };
        assert_eq!(shape.storage().r#type().data_type(), DataType::Zero);
        assert!(shape.storage().storage_bytes().is_empty());
        // One indexed identity has two occurrences with different padding. Neither capture nor reconstruction can
        // deduplicate them by identity: each occurrence keeps its own full payload and live-coordinate mask.
        assert_eq!(shape.extents().len(), 3);
        assert_eq!(shape.extents()[1], first);
        assert_eq!(shape.extents()[2], second);
        let zero = ArrayIrOperation::<Array>::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(r#type.clone().into()),
            [&residual],
        )
        .unwrap();
        assert_eq!(
            zero.ragged_storage().unwrap(),
            Array::from_elements(r#type.storage_type().clone(), &[0f32; 54]).unwrap().into()
        );
        for axis in [1, 2, 3] {
            assert_eq!(zero.ragged_extents(axis), source.ragged_extents(axis));
        }
    }

    #[test]
    fn test_ragged_identity_pullback_rejects_changed_seed_extents() {
        use crate::differentiation::differentiate_at;

        let primal = differential_value();
        let ArrayIrValue::Ragged(primal_value) = &primal else { unreachable!() };
        let seed = ArrayIrValue::Ragged(
            RaggedValue::new(
                primal_value.r#type().into_owned(),
                primal_value.storage().clone(),
                vec![Array::vector(vec![0i64, 3]).unwrap()],
            )
            .unwrap(),
        );
        let (_, pullback) = differentiate_at(primal.clone()).vjp(Ok).unwrap();
        assert_eq!(pullback.apply(primal.clone()), Ok(MaybeZero::Value(primal.clone())));
        assert_eq!(pullback.apply_materialized(primal.clone()), Ok(primal.clone()));
        assert_eq!(pullback.apply(seed.clone()).unwrap_err().downcast_custom(), Some(&differential_shape_mismatch()));
        assert_eq!(
            pullback.apply_materialized(seed.clone()).unwrap_err().downcast_custom(),
            Some(&differential_shape_mismatch())
        );
        assert_eq!(
            pullback.materialize_cotangents(MaybeZero::Value(seed)).unwrap_err().downcast_custom(),
            Some(&differential_shape_mismatch()),
        );
        let ArrayIrValue::Ragged(value) = &primal else { unreachable!() };
        let other_rows = DimensionVariable::new("other_rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let other_type = RaggedType::new(
            value.r#type().storage_type().clone(),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, other_rows, vec![0])],
        )
        .unwrap();
        let other = ArrayIrValue::Ragged(
            RaggedValue::new(other_type, value.storage().clone(), value.extents().to_vec()).unwrap(),
        );
        for other in [other, primal.ragged_storage().unwrap()] {
            assert_eq!(
                pullback.materialize_cotangents(MaybeZero::Value(other.clone())),
                Err(ProgramError::InvalidArgument {
                    message: format!(
                        "pullback cotangent 0 has type {} but its primal input requires cotangent type {}",
                        other.r#type(),
                        primal.r#type(),
                    ),
                }),
            );
        }
    }

    #[test]
    fn test_ragged_identity_pushforward_rejects_changed_tangent_extents() {
        use crate::differentiation::differentiate_at;

        let primal = differential_value();
        let ArrayIrValue::Ragged(primal_value) = &primal else { unreachable!() };
        let tangent = ArrayIrValue::Ragged(
            RaggedValue::new(
                primal_value.r#type().into_owned(),
                primal_value.storage().clone(),
                vec![Array::vector(vec![0i64, 3]).unwrap()],
            )
            .unwrap(),
        );
        let (_, pushforward) = differentiate_at(primal.clone()).linearize(Ok).unwrap();
        assert_eq!(pushforward.apply(primal.clone()), Ok(primal.clone()));
        assert_eq!(
            pushforward.apply(tangent.clone()).unwrap_err().downcast_custom(),
            Some(&differential_shape_mismatch())
        );
        // A dense result has no indexed output witness. Rejection must occur before executing its projection.
        let (_, projection) = differentiate_at(primal).linearize(|input| input.ragged_storage()).unwrap();
        assert_eq!(projection.apply(tangent).unwrap_err().downcast_custom(), Some(&differential_shape_mismatch()));
    }

    #[test]
    fn test_ragged_identity_jvp_rejects_changed_tangent_extents() {
        use crate::differentiation::differentiate_at;

        let primal = differential_value();
        let ArrayIrValue::Ragged(primal_value) = &primal else { unreachable!() };
        let tangent = ArrayIrValue::Ragged(
            RaggedValue::new(
                primal_value.r#type().into_owned(),
                primal_value.storage().clone(),
                vec![Array::vector(vec![0i64, 3]).unwrap()],
            )
            .unwrap(),
        );
        assert_eq!(differentiate_at(primal.clone()).jvp(primal.clone(), Ok), Ok((primal.clone(), primal.clone())));
        let mut called = false;
        let error: ProgramError = differentiate_at(primal)
            .jvp(tangent, |input| {
                called = true;
                Ok(input)
            })
            .unwrap_err()
            .into();
        assert!(!called);
        assert_eq!(error.downcast_custom(), Some(&differential_shape_mismatch()));
    }

    #[test]
    fn test_ragged_add_pullback_rejects_changed_seed_extents() {
        use crate::differentiation::differentiate_at;

        let primal = differential_value();
        let ArrayIrValue::Ragged(primal_value) = &primal else { unreachable!() };
        let seed = ArrayIrValue::Ragged(
            RaggedValue::new(
                primal_value.r#type().into_owned(),
                primal_value.storage().clone(),
                vec![Array::vector(vec![0i64, 3]).unwrap()],
            )
            .unwrap(),
        );
        let (_, pullback) = differentiate_at((primal.clone(), primal.clone()))
            .vjp(|(left, right)| {
                left.domain()
                    .bind(RaggedAddOperation, Vec::new(), &[left.clone(), right])
                    .map(|mut values| values.remove(0))
            })
            .unwrap();
        assert_eq!(pullback.apply_materialized(primal.clone()), Ok((primal.clone(), primal)));
        assert_eq!(pullback.apply(seed).unwrap_err().downcast_custom(), Some(&differential_shape_mismatch()));
    }

    #[test]
    fn test_ragged_identity_differentials_preserve_inactive_padding() {
        use crate::differentiation::differentiate_at;

        let r#type = nested_type(2);
        let primal = ArrayIrValue::Ragged(
            RaggedValue::new(
                r#type.clone(),
                Array::from_elements(r#type.storage_type().clone(), &[2f32; 24]).unwrap(),
                vec![
                    Array::vector(vec![0i64, 0]).unwrap(),
                    Array::from_elements(r#type.extent_types()[1].clone(), &[999i64; 6]).unwrap(),
                ],
            )
            .unwrap(),
        );
        let tangent = ArrayIrValue::Ragged(
            RaggedValue::new(
                r#type.clone(),
                Array::from_elements(r#type.storage_type().clone(), &[3f32; 24]).unwrap(),
                vec![
                    Array::vector(vec![0i64, 0]).unwrap(),
                    Array::from_elements(r#type.extent_types()[1].clone(), &[-7i64; 6]).unwrap(),
                ],
            )
            .unwrap(),
        );
        assert_eq!(differentiate_at(primal.clone()).jvp(tangent.clone(), Ok), Ok((primal.clone(), tangent.clone())));
        let (_, pushforward) = differentiate_at(primal.clone()).linearize(Ok).unwrap();
        assert_eq!(pushforward.apply(tangent.clone()), Ok(tangent.clone()));
        let (_, pullback) = differentiate_at(primal).vjp(Ok).unwrap();
        assert_eq!(pullback.apply_materialized(tangent.clone()), Ok(tangent));
    }

    #[test]
    fn test_ragged_storage_reverse_accumulation_and_disconnected_zero() {
        use crate::differentiation::differentiate_at;
        let first = differential_value();
        // This independent family has different extents, so borrowing the active input's zero source would be wrong.
        let ArrayIrValue::Ragged(value) = &first else { unreachable!() };
        let disconnected_type = RaggedType::new(
            value.storage().r#type().into_owned(),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(
                1,
                DimensionVariable::new("disconnected_rows", value.r#type().ragged_dimensions()[0].dimension().bounds()),
                vec![0],
            )],
        )
        .unwrap();
        let disconnected = ArrayIrValue::Ragged(
            RaggedValue::new(
                disconnected_type,
                Array::from_elements(value.storage().r#type().into_owned(), &[5f32; 6]).unwrap(),
                vec![Array::vector(vec![3i64, 0]).unwrap()],
            )
            .unwrap(),
        );
        assert_ne!(disconnected.ragged_extents(1), first.ragged_extents(1));
        let (_, pullback) = differentiate_at((first.clone(), disconnected.clone()))
            .vjp(|(first, _)| first.ragged_storage()?.add(&first.ragged_storage()?))
            .unwrap();
        // Two independent storage-projection adjoints reach the same ragged input and invoke type-directed addition.
        let program = pullback.transposed_program(&[]).unwrap().into_program();
        assert!(
            program
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayIrOperation::RaggedAdd(_)))
        );
        for residual in pullback.residuals() {
            if let ArrayIrValue::Ragged(residual) = residual {
                assert_eq!(residual.storage().r#type().data_type(), DataType::Zero);
                assert!(residual.storage().storage_bytes().is_empty());
            }
        }
        let seed = Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32; 6]).unwrap().into();
        let (active, zero) = pullback.apply_materialized(seed).unwrap();
        assert_eq!(
            active.ragged_storage().unwrap(),
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[2f32; 6]).unwrap().into()
        );
        assert_eq!(active.ragged_extents(1), first.ragged_extents(1));
        assert_eq!(
            zero.ragged_storage().unwrap(),
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[0f32; 6]).unwrap().into()
        );
        assert_eq!(zero.ragged_extents(1), disconnected.ragged_extents(1));
    }

    #[test]
    fn test_ragged_dimension_value_compact_differentials() {
        use crate::differentiation::differentiate_at;
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::I64, [2]),
            RaggedValueKind::Dimension(Dimension::Static(3).into()),
            Vec::new(),
        )
        .unwrap();
        let primal =
            ArrayIrValue::Ragged(RaggedValue::new(r#type, Array::vector(vec![3i64, 3]).unwrap(), Vec::new()).unwrap());
        let (output, pushforward) = differentiate_at(primal).linearize(|input| input.ragged_storage()).unwrap();
        assert_eq!(output, Array::vector(vec![3i64, 3]).unwrap().into());
        let (_, linear, residuals, _, _, _) = pushforward.into_parts();
        assert!(linear.input_types().is_empty());
        assert!(linear.output_types().is_empty());
        assert!(residuals.is_empty());
    }
    #[test]
    fn test_ragged_dimension_value_batched_storage_consumes_item_identity() {
        use crate::arrays::ArrayIrBatchingPolicy;
        use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer, batch};

        type Eager = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let item = DimensionVariable::new("item", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::I64, [2]),
            RaggedValueKind::Dimension(item.into()),
            Vec::new(),
        )
        .unwrap();
        let storage = Array::vector(vec![1i64, 2]).unwrap();
        let value = ArrayIrValue::Ragged(RaggedValue::new(r#type, storage.clone(), Vec::new()).unwrap());
        let output = batch(
            |item: BatchingTracer<Eager, ArrayIrBatchingPolicy>| item.ragged_storage(),
            value,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::default(),
        )
        .unwrap();
        assert_eq!(output, storage.into());
    }

    #[test]
    fn test_ragged_integer_value_compact_differentials() {
        use crate::differentiation::differentiate_at;
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::I64, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        let primal = ArrayIrValue::Ragged(
            RaggedValue::new(
                r#type.clone(),
                Array::from_elements(r#type.storage_type().clone(), &[7i64; 6]).unwrap(),
                vec![Array::vector(vec![1i64, 2]).unwrap()],
            )
            .unwrap(),
        );
        let (_, pushforward) =
            differentiate_at(primal.clone()).linearize(|input| Ok::<_, ProgramError>(input)).unwrap();
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let tangent_type = ArrayIrType::Ragged(r#type.clone()).cotangent().unwrap();
        let tangent = ArrayIrOperation::<Array>::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(tangent_type),
            [&primal],
        )
        .unwrap();
        assert_eq!(pushforward.apply(tangent.clone()), Ok(tangent.clone()));
        let ArrayIrValue::Ragged(value) = &tangent else { unreachable!() };
        let mismatch = ArrayIrValue::Ragged(
            RaggedValue::new(
                value.r#type().into_owned(),
                value.storage().clone(),
                vec![Array::vector(vec![0i64, 3]).unwrap()],
            )
            .unwrap(),
        );
        assert_eq!(
            pushforward.apply(mismatch.clone()).unwrap_err().downcast_custom(),
            Some(&differential_shape_mismatch())
        );
        let error: ProgramError = differentiate_at(primal.clone()).jvp(mismatch.clone(), Ok).unwrap_err().into();
        assert_eq!(error.downcast_custom(), Some(&differential_shape_mismatch()));
        let (_, pullback) = differentiate_at(primal.clone()).vjp(Ok).unwrap();
        assert_eq!(pullback.apply(mismatch).unwrap_err().downcast_custom(), Some(&differential_shape_mismatch()));
        assert_eq!(pullback.apply_materialized(tangent.clone()), Ok(tangent));
        let (_, linear, residuals, _, _, _) = pushforward.into_parts();
        assert!(linear.output_types().is_empty());
        assert!(linear.input_types().iter().all(DifferentiableType::is_zero_space));
        for residual in residuals {
            let ArrayIrValue::Ragged(shape) = residual else { panic!("expected indexed shape residual") };
            assert!(shape.storage().storage_bytes().is_empty());
            assert_eq!(shape.extents()[0], Array::vector(vec![1i64, 2]).unwrap());
        }
    }

    #[test]
    fn test_ragged_storage_batching_differentiation_orders() {
        use crate::arrays::{ArrayIrBatch, ArrayIrBatchingPolicy};
        use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer, batch};
        use crate::differentiation::{DifferentiationContext, differentiate_at};
        type Eager = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        type Differentiated = DifferentiationContext<Eager>;
        let primal = differential_value();
        let expected = primal.ragged_storage().unwrap();
        let (_, differentiated_batch) = differentiate_at(primal.clone())
            .jvp(primal.clone(), |input| {
                batch(
                    |item: BatchingTracer<Differentiated, ArrayIrBatchingPolicy>| item.ragged_storage(),
                    input,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::default(),
                )
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(differentiated_batch, expected);
        let batched_derivative = batch(
            |item: BatchingTracer<Eager, ArrayIrBatchingPolicy>| {
                differentiate_at(item.clone())
                    .jvp(item, |input| input.ragged_storage())
                    .map(|(_, tangent)| tangent)
                    .map_err(Into::into)
            },
            primal.clone(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::default(),
        )
        .unwrap();
        assert_eq!(batched_derivative, expected);
        let ArrayIrValue::Ragged(value) = &primal else { unreachable!() };
        let mismatch = ArrayIrValue::Ragged(
            RaggedValue::new(
                value.r#type().into_owned(),
                value.storage().clone(),
                vec![Array::vector(vec![0i64, 3]).unwrap()],
            )
            .unwrap(),
        );
        let error: ProgramError = batch(
            |item: BatchingTracer<Eager, ArrayIrBatchingPolicy>| {
                let tangent =
                    BatchingTracer::new(item.domain(), ArrayIrBatch::new(mismatch.clone(), BatchAxis::new(0))?);
                differentiate_at(item)
                    .jvp(tangent, |input| input.ragged_storage())
                    .map(|(_, tangent)| tangent)
                    .map_err(Into::into)
            },
            primal,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::default(),
        )
        .unwrap_err()
        .into();
        assert_eq!(error.downcast_custom(), Some(&differential_shape_mismatch()));
    }

    #[test]
    fn test_ragged_storage_batching_differentiation_replicated_tangent() {
        use crate::arrays::ArrayIrBatchingPolicy;
        use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer, batch};
        use crate::differentiation::differentiate_at;

        // One replicated item tangent serves every item of a mapped primal whose per-item extents all equal its own.
        // The tangent validation aligns it to the primal's mapped representation and checks each item separately.
        type Eager = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        type Batched = BatchingTracer<Eager, ArrayIrBatchingPolicy>;
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let primal = |extents: Vec<i64>| {
            ArrayIrValue::Ragged(
                RaggedValue::new(
                    RaggedType::new(
                        ArrayType::new_static(DataType::F32, [2, 3]),
                        RaggedValueKind::Array,
                        vec![RaggedDimension::new(1, rows.clone(), vec![0])],
                    )
                    .unwrap(),
                    Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap(),
                    vec![Array::vector(extents).unwrap()],
                )
                .unwrap(),
            )
        };
        let tangent = ArrayIrValue::Ragged(
            RaggedValue::new(
                RaggedType::new(
                    ArrayType::new_static(DataType::F32, [3]),
                    RaggedValueKind::Array,
                    vec![RaggedDimension::new(0, rows.clone(), Vec::new())],
                )
                .unwrap(),
                Array::vector(vec![7f32, 8., 9.]).unwrap(),
                vec![Array::scalar(2i64).unwrap()],
            )
            .unwrap(),
        );
        let batched_derivative = |primal: ArrayIrValue<Array>| {
            batch(
                |(primal, tangent): (Batched, Batched)| {
                    differentiate_at(primal)
                        .jvp(tangent, |input| input.ragged_storage())
                        .map(|(_, tangent)| tangent)
                        .map_err(Into::into)
                },
                (primal, tangent.clone()),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                BatchAxisSpecification::default(),
            )
        };
        assert_eq!(
            batched_derivative(primal(vec![2, 2])),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![7f32, 8., 9., 7., 8., 9.]).unwrap())),
        );

        // A primal item whose live extent differs from the shared tangent extent fails for that item alone.
        let error: ProgramError = batched_derivative(primal(vec![1, 2])).unwrap_err().into();
        assert_eq!(
            error.downcast_custom(),
            Some(&AssertionError::FailedElements {
                message: "guarded ragged occurrences of `rows` must have equal live extents".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![0],
                    observations: vec![
                        ("extent".to_owned(), "1".to_owned()),
                        ("other_extent".to_owned(), "2".to_owned()),
                    ],
                }],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_add_batching_replicated_contribution() {
        use crate::arrays::ArrayIrBatchingPolicy;
        use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer, batch};

        type Eager = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        type Batched = BatchingTracer<Eager, ArrayIrBatchingPolicy>;
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let mapped = |descriptor: RaggedDimension, extents: Array| {
            ArrayIrValue::Ragged(
                RaggedValue::new(
                    RaggedType::new(
                        ArrayType::new_static(DataType::F32, [2, 3]),
                        RaggedValueKind::Array,
                        vec![descriptor],
                    )
                    .unwrap(),
                    Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap(),
                    vec![extents],
                )
                .unwrap(),
            )
        };
        let replicated = ArrayIrValue::Ragged(
            RaggedValue::new(
                RaggedType::new(
                    ArrayType::new_static(DataType::F32, [3]),
                    RaggedValueKind::Array,
                    vec![RaggedDimension::new(0, rows.clone(), Vec::new())],
                )
                .unwrap(),
                Array::vector(vec![10f32, 20., 30.]).unwrap(),
                vec![Array::scalar(2i64).unwrap()],
            )
            .unwrap(),
        );
        let sum = |inputs: Vec<ArrayIrValue<Array>>, axes: Vec<BatchAxis>| {
            batch(
                |inputs: Vec<Batched>| {
                    inputs[0].domain().bind(RaggedAddOperation, Vec::new(), &inputs)?.remove(0).ragged_storage()
                },
                inputs,
                axes,
                BatchAxis::new(0),
                BatchAxisSpecification::default(),
            )
        };
        let expected = Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![11f32, 22., 33., 14., 25., 36.]).unwrap()));

        // The replicated contribution is aligned to the mapped storage axis in either input order, taking per-item
        // extents where the mapped contribution varies over the batch coordinate and its shared extent elsewhere.
        let per_item = mapped(RaggedDimension::new(1, rows.clone(), vec![0]), Array::vector(vec![2i64, 2]).unwrap());
        let shared = mapped(RaggedDimension::new(1, rows.clone(), Vec::new()), Array::scalar(2i64).unwrap());
        for mapped in [per_item, shared] {
            assert_eq!(
                sum(vec![mapped.clone(), replicated.clone()], vec![BatchAxis::new(0), BatchAxis::replicated()]),
                expected,
            );
            assert_eq!(
                sum(vec![replicated.clone(), mapped], vec![BatchAxis::replicated(), BatchAxis::new(0)]),
                expected,
            );
        }

        // Every batch item's live extents are still checked against the replicated contribution.
        let error: ProgramError = sum(
            vec![mapped(RaggedDimension::new(1, rows, vec![0]), Array::vector(vec![2i64, 1]).unwrap()), replicated],
            vec![BatchAxis::new(0), BatchAxis::replicated()],
        )
        .unwrap_err()
        .into();
        assert_eq!(
            error.downcast_custom(),
            Some(&AssertionError::FailedElements {
                message: "ragged identity source for `rows` must have equal live extents".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![1],
                    observations: vec![
                        ("extent".to_owned(), "2".to_owned()),
                        ("source_extent".to_owned(), "1".to_owned()),
                    ],
                }],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_ragged_new_batching_differentiation_orders() {
        use crate::arrays::{ArrayIrBatch, ArrayIrBatchingPolicy};
        use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer, batch};
        use crate::differentiation::differentiate_at;

        type Eager = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        type Batched = BatchingTracer<Eager, ArrayIrBatchingPolicy>;
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let item_type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows, vec![])],
        )
        .unwrap();
        let seed = Array::vector(vec![1f32, 2., 3.]).unwrap();
        let broadcast_seed =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32, 2., 3., 1., 2., 3.]).unwrap();

        // Batched forward mode with a replicated tangent: the tangent rule rebuilds the replicated storage tangent over
        // the mapped primal as its identity source, so the tangent is broadcast to the primal's batch axis.
        let storage = Array::from_fn_elements(ArrayType::new_static(DataType::F32, [2, 3]), |_| Ok(1f32)).unwrap();
        assert_eq!(
            batch(
                |(storage, tangent, extents): (Batched, Batched, Batched)| {
                    differentiate_at(storage)
                        .with_captures(extents)
                        .jvp(tangent, |storage, extents| {
                            storage.ragged_new(item_type.clone(), &[extents])?.ragged_storage()
                        })
                        .map(|(_, tangent)| tangent)
                        .map_err(ProgramError::from)
                },
                (
                    ArrayIrValue::Array(storage),
                    ArrayIrValue::Array(seed.clone()),
                    ArrayIrValue::Array(Array::vector(vec![1i64, 3]).unwrap()),
                ),
                (BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(0)),
                BatchAxis::new(0),
                BatchAxisSpecification::default(),
            ),
            Ok(ArrayIrValue::Array(broadcast_seed.clone())),
        );

        // Batched reverse mode with a replicated cotangent seed: transposing `ragged_storage` rebuilds the cotangent
        // over the mapped residual, including when two storage projections accumulate their contributions.
        let primal = differential_value();
        let ArrayIrValue::Ragged(value) = &primal else { unreachable!() };
        let doubled_seed =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[2f32, 4., 6., 2., 4., 6.]).unwrap();
        for (uses, storage) in [(1, broadcast_seed.clone()), (2, doubled_seed)] {
            let expected = (ArrayIrValue::Array(storage), ArrayIrValue::Array(value.extents()[0].clone()));
            assert_eq!(
                batch(
                    |item: Batched| {
                        let (_, pullback) = differentiate_at(item.clone())
                            .vjp(|input| {
                                let projection = input.ragged_storage()?;
                                if uses == 1 { Ok(projection) } else { projection.add(&input.ragged_storage()?) }
                            })
                            .map_err(ProgramError::from)?;
                        let seed = BatchingTracer::new(
                            item.domain(),
                            ArrayIrBatch::new(ArrayIrValue::Array(seed.clone()), BatchAxis::replicated())?,
                        );
                        let cotangent = pullback.apply_materialized(seed).map_err(ProgramError::from)?;
                        // Keep the ragged cotangent internal; public batching returns ordinary storage and extents.
                        Ok((cotangent.ragged_storage()?, cotangent.ragged_extents(0)?))
                    },
                    primal.clone(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::default(),
                ),
                Ok(expected),
            );
        }

        // Reverse mode around batching: the packed cotangent of the whole batched ragged input is the packed seed with
        // the primal's extents.
        let packed_seed =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32, 2., 3., 4., 5., 6.]).unwrap();
        let (_, pullback) = differentiate_at(primal.clone())
            .vjp(|input| {
                batch(
                    |item: BatchingTracer<_, ArrayIrBatchingPolicy>| item.ragged_storage(),
                    input,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::default(),
                )
                .map_err(ProgramError::from)
            })
            .unwrap();
        assert_eq!(
            pullback.apply_materialized(ArrayIrValue::Array(packed_seed.clone())),
            Ok(ArrayIrValue::Ragged(
                RaggedValue::new(value.r#type().into_owned(), packed_seed, value.extents().to_vec()).unwrap()
            )),
        );
    }

    #[test]
    fn test_ragged_new_differentiation_keeps_unknown_extent_checks() {
        use crate::differentiation::differentiate_at;
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let representation = RaggedType::new(
            ArrayType::new_static(DataType::F32, [3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, rows, Vec::new())],
        )
        .unwrap();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            move |(storage, extent): (ArrayTracer, ArrayTracer)| {
                let (_, tangent) = differentiate_at(storage.clone())
                    .with_captures(extent)
                    .jvp(storage, |storage, extent| {
                        storage.ragged_new(representation.clone(), &[extent])?.ragged_storage()
                    })
                    .map_err(ProgramError::from)?;
                Ok::<_, ProgramError>(tangent)
            },
            (
                ArrayIrType::from(ArrayType::new_static(DataType::F32, [3])),
                ArrayIrType::from(ArrayType::scalar(DataType::I64)),
            ),
        )
        .unwrap();
        let storage = Array::vector(vec![1f32, 2., 3.]).unwrap();
        let error = program.interpret((storage.clone().into(), Array::scalar(999i64).unwrap().into())).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "ragged extent on storage axis 0 is outside its declared bounds".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![],
                    observations: vec![("extent".to_owned(), "999".to_owned())]
                }],
                omitted: 0,
            })
        );
        let partial = program
            .to_flat_program()
            .partially_evaluate(&[
                PartialValue::Known(storage.clone().into()),
                PartialValue::Unknown(ArrayType::scalar(DataType::I64).into()),
            ])
            .unwrap();
        assert!(matches!(partial.outputs(), [PartialEvaluationOutput::Unknown(0)]));
        assert_eq!(
            program.interpret((storage.clone().into(), Array::scalar(2i64).unwrap().into())),
            Ok(storage.into())
        );
    }

    #[test]
    fn test_ragged_new_differentiation_linearize_retains_shape_carrier() {
        use crate::differentiation::differentiate_at;
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let representation = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        let storage =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32, 2., 3., 4., 5., 6.]).unwrap();
        let extents = Array::vector(vec![1i64, 2]).unwrap();
        let (output, pushforward) = differentiate_at(ArrayIrValue::Array(storage.clone()))
            .with_captures(ArrayIrValue::Array(extents.clone()))
            .linearize(|storage, extents| storage.ragged_new(representation.clone(), &[extents])?.ragged_storage())
            .unwrap();
        assert_eq!(output, storage.into());
        let tangent =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[7f32, 8., 9., 10., 11., 12.]).unwrap();
        assert_eq!(pushforward.apply(tangent.clone().into()), Ok(tangent.into()));

        // Only the extents and a Zero-storage shape carrier are retained; the numerical primal storage is not.
        let (_, linear, residuals, _, _, _) = pushforward.into_parts();
        assert_eq!(
            residuals.iter().map(|residual| residual.r#type().to_string()).collect::<Vec<_>>(),
            vec!["i64[2]".to_owned(), "ragged<zero[2, 3]; 1:rows over [0]>".to_owned()],
        );
        assert_eq!(residuals[0], ArrayIrValue::Array(extents.clone()));
        let ArrayIrValue::Ragged(carrier) = &residuals[1] else { panic!("expected a ragged shape carrier") };
        assert!(carrier.storage().storage_bytes().is_empty());
        assert_eq!(carrier.extents(), &[extents]);
        assert_eq!(
            linear.to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:i64[2], %2:ragged<zero[2, 3]; 1:rows over [0]> .
                let %3:ragged<f32[2, 3]; 1:rows over [0]> = ragged_new \
                        [output_type=ragged<f32[2, 3]; 1:rows over [0]>, identity_source=true] %0 %1 %2
                    %4:f32[2, 3] = ragged_storage %3
                in (%4)"},
        );
    }
}
