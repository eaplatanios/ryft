//! Checked construction and explicit decomposition of first-class ragged representations.
//!
//! These operations expose the low-level padded storage and full extent payloads used by transforms. Numerical array
//! operations still require an ordinary array projection. Construction checks live indexed extents, including sibling
//! dependencies; the structural [`RaggedValue::new`] constructor alone does not establish those numerical facts.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::Display;
use std::num::NonZeroUsize;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayIrType, ArrayIrValue, ArrayType, ArrayTypeRefinements, DataType, DimensionType, DimensionVariable,
    RaggedType, RaggedValue, RaggedValueKind,
};
use crate::axes::Axis;
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    DifferentiationPolicy,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_transposable_operation, impl_reference_dischargeable_operation};
use crate::operations::{
    And, Assert, Broadcast, Capability, Compare, Constant, GuardOperation, Iota, Not, Or, Reduce, TransferToMemory,
};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    EffectClass, EffectClasses, Effects, Operation, OperationFormatter, OperationProvider, ProgramError,
    RegionInterface, Type, TypeError, TypeIdentityRenaming, TypeRefinements, Typed, Value, ValueDomainDispatch,
    ValueProjection,
};

/// Canonical operation name for [`RaggedNewOperation`].
pub const RAGGED_NEW_OPERATION_NAME: &str = "ragged_new";

/// Canonical operation name for [`RaggedStorageOperation`].
pub const RAGGED_STORAGE_OPERATION_NAME: &str = "ragged_storage";

/// Canonical operation name for [`RaggedExtentsOperation`].
pub const RAGGED_EXTENTS_OPERATION_NAME: &str = "ragged_extents";

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
    pub fn new(output_type: RaggedType) -> Self {
        Self { output_type, identity_source: false }
    }

    /// Creates checked construction with a final ragged identity source after the dense fields. This input forwards
    /// existing indexed identities; it supplies no numerical proof and never suppresses payload validation.
    pub fn with_identity_source(output_type: RaggedType) -> Self {
        Self { output_type, identity_source: true }
    }

    /// Returns the declared output representation.
    pub fn output_type(&self) -> &RaggedType {
        &self.output_type
    }

    /// Returns whether the final input forwards the output's indexed identities.
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

    fn type_identity_definitions(&self) -> Cow<'_, [DimensionVariable]> {
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

/// Implements the conservative transform contracts shared by the representation primitives.
macro_rules! impl_representation_transforms {
    ($operation:ty) => {
        impl<C: Context<Type = ArrayIrType, Operation: From<$operation>>> PartiallyEvaluatableOperation<C>
            for $operation
        {
        }

        impl<C: Context<Type = ArrayIrType>, P: BatchingPolicy<C>> BatchableOperation<C, P> for $operation {
            fn batch<D: BatchingDriver<C, P>>(
                &self,
                _context: &BatchingContext<C, P>,
                _driver: &D,
                _inputs: &[P::Batch],
            ) -> Result<BatchedOutputs<C, P>, BatchingError> {
                Err(BatchingError::UnsupportedOperation {
                    message: format!("batching `{}` is not supported yet", self.name()),
                })
            }
        }

        impl_reference_dischargeable_operation!(@reference_free $operation);
        impl<C: Context<Type = ArrayIrType, Operation: From<$operation>>> DifferentiableOperation<C> for $operation {
            fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
                &self,
                _context: &DifferentiationContext<C, P>,
                _driver: &D,
                _inputs: &[DifferentiationDual<C::Value>],
            ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
                Err(ProgramError::UnsupportedOperation {
                    message: format!("differentiating `{}` is not supported yet", self.name()),
                }.into())
            }
        }
        impl_non_transposable_operation!($operation);
    };
}

impl_representation_transforms!(RaggedNewOperation);
impl_representation_transforms!(RaggedStorageOperation);
impl_representation_transforms!(RaggedExtentsOperation);

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
/// let storage = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32; 6])?);
/// let extents = ArrayIrValue::Array(Array::vector(vec![1_i64, 3])?);
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

/// Explicit access to the padded ordinary array storage of a ragged value.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that composite array IR
/// values implement this capability for [`ArrayIrType`].
#[capability]
pub trait RaggedStorage<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns the representation's physical storage, including padding.
    fn ragged_storage(&self) -> Result<Self, ProgramError>;
}

/// Explicit access to the full dense extents of a ragged storage-axis occurrence.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that composite array IR
/// values implement this capability for [`ArrayIrType`].
#[capability]
pub trait RaggedExtents<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns the selected occurrence's complete extent array, retaining entries outside the live coordinates
    /// determined by its sibling occurrences.
    fn ragged_extents<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError>;
}

impl<
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayIrType,
            Domain: Context<Type = ArrayIrType, Operation: From<RaggedNewOperation>>,
        >,
> RaggedNew<ArrayIrType> for V
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

impl<
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayIrType,
            Domain: Context<Type = ArrayIrType, Operation: From<RaggedStorageOperation>>,
        >,
> RaggedStorage<ArrayIrType> for V
{
    fn ragged_storage(&self) -> Result<Self, ProgramError> {
        Ok(self.domain().bind(RaggedStorageOperation, Vec::new(), std::slice::from_ref(self))?.remove(0))
    }
}

impl<
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayIrType,
            Domain: Context<Type = ArrayIrType, Operation: From<RaggedExtentsOperation>>,
        >,
> RaggedExtents<ArrayIrType> for V
{
    fn ragged_extents<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError> {
        Ok(self
            .domain()
            .bind(RaggedExtentsOperation::new(axis), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<
    A: Value<Type = ArrayType, Domain: Context + Constant<A, Array> + Iota<A>>
        + Compare
        + And
        + Or
        + Not
        + Broadcast
        + Reduce
        + Assert
        + TransferToMemory,
> RaggedNew<ArrayIrType> for ArrayIrValue<A>
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

/// Shares structural admission and numerical checks between fresh creation and identity-forwarding reconstruction.
fn checked_new<
    A: Value<Type = ArrayType, Domain: Context + Constant<A, Array> + Iota<A>>
        + Compare
        + And
        + Or
        + Not
        + Broadcast
        + Reduce
        + Assert
        + TransferToMemory,
>(
    storage: &ArrayIrValue<A>,
    r#type: RaggedType,
    extents: &[ArrayIrValue<A>],
    source: Option<&ArrayIrValue<A>>,
) -> Result<ArrayIrValue<A>, ProgramError> {
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
    let types = fields.iter().map(|field| field.r#type().into_owned()).collect::<Vec<_>>();
    let mut identities = Vec::new();
    for r#type in &types {
        for (_, identity) in r#type.identities() {
            if !identities.contains(identity) {
                identities.push(identity.clone());
            }
        }
    }
    let guard = GuardOperation::new(RAGGED_NEW_OPERATION_NAME, types.clone()).with_retained_identities(identities);
    let operation = <A::Domain as Domain>::Operation::provide(guard, &types.iter().collect::<Vec<_>>())?;
    let outputs = value.storage().domain().bind(operation, Vec::new(), &fields)?;
    check_count!("output", outputs, fields.len(), ProgramError);
    let mut outputs = outputs.into_iter();
    let storage = outputs.next().unwrap();
    let extents = outputs.take(own_count - 1).collect();
    Ok(ArrayIrValue::Ragged(RaggedValue::new(value.r#type().into_owned(), storage, extents)?))
}

impl<A: Value<Type = ArrayType>> RaggedStorage<ArrayIrType> for ArrayIrValue<A> {
    fn ragged_storage(&self) -> Result<Self, ProgramError> {
        let Self::Ragged(value) = self else {
            return Err(<&RaggedType>::try_from(self.r#type().as_ref()).unwrap_err().into());
        };
        Ok(Self::Array(value.storage().clone()))
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
fn validate_payloads<
    A: Value<Type = ArrayType, Domain: Constant<A, Array> + Iota<A>>
        + Compare
        + And
        + Or
        + Not
        + Broadcast
        + Reduce
        + Assert
        + TransferToMemory,
>(
    r#type: &RaggedType,
    storage: &A,
    extents: &[A],
) -> Result<PayloadMasks<A>, ProgramError> {
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
fn validate_source<A: Value<Type = ArrayType> + Compare + And + Or + Not + Broadcast + Assert + TransferToMemory>(
    value: &RaggedValue<A>,
    masks: &PayloadMasks<A>,
    source: &RaggedValue<A>,
    source_masks: &PayloadMasks<A>,
) -> Result<(), ProgramError> {
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
fn check_equality<A: Value<Type = ArrayType> + Compare + And + Or + Not + Broadcast + Assert + TransferToMemory>(
    value: &A,
    mask: Option<&A>,
    other: &A,
    other_mask: Option<&A>,
    axes: &[usize],
    message: &str,
    other_label: &str,
) -> Result<(), ProgramError> {
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
fn live_coordinate_mask<A: Value<Type = ArrayType, Domain: Iota<A>> + Compare + And + Broadcast + Reduce>(
    r#type: &RaggedType,
    extent_axes: &[usize],
    target: &A,
    extents: &[A],
) -> Result<Option<A>, ProgramError> {
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
fn integer_constant<A: Value<Type = ArrayType, Domain: Constant<A, Array>>>(
    value: &A,
    integer: i64,
) -> Result<A, ProgramError> {
    let literal = Array::scalar(integer)?;
    let literal = Array::new(value.r#type().scalar_like()?, literal.storage_bytes().to_vec())?;
    value.domain().constant(literal)
}

/// Validates an inclusive portable extent range, avoiding overflow at the largest representable exclusive bound.
fn check_bounds<A: Value<Type = ArrayType, Domain: Constant<A, Array>> + Compare + And + Or + Not + Assert>(
    value: &A,
    dimension: &DimensionType,
    mask: Option<&A>,
    message: &str,
) -> Result<(), ProgramError> {
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
    use crate::operations::{AssertionError, AssertionFailure};
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
        ArrayIrValue::Array(Array::from_fn_elements(r#type.storage_type().clone(), |_| Ok(17.0_f32)).unwrap())
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
        let inputs = (storage(&r#type), ArrayIrValue::Array(Array::vector(vec![1_i64, 3]).unwrap()));
        assert_eq!(program.interpret(inputs.clone()), Ok(inputs));
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
        let rows = ArrayIrValue::Array(Array::vector(vec![1_i64, 3]).unwrap());
        let columns = ArrayIrValue::Array(
            Array::from_elements(r#type.extent_types()[1].clone(), &[2_i64, 999, -7, 1, 4, 0]).unwrap(),
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
        let changed_rows = ArrayIrValue::Array(Array::vector(vec![3_i64, 3]).unwrap());
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
            let rows = ArrayIrValue::Array(Array::vector(vec![0_i64; batch]).unwrap());
            let columns = ArrayIrValue::Array(
                Array::from_fn_elements(r#type.extent_types()[1].clone(), |_| Ok(999_i64)).unwrap(),
            );
            assert!(storage(&r#type).ragged_new(r#type, &[rows, columns]).is_ok());
        }
    }

    #[test]
    fn test_ragged_new_interpretation_omitted_ancestor() {
        // Columns vary over rows, while row liveness varies over a batch coordinate absent from those extent axes.
        // A row is live when at least one live batch item contains it.
        for (batch, rows, columns, expected_mask) in [
            (2, vec![1_i64, 2], vec![4_i64, 1, 999], vec![true, true, false]),
            (2, vec![0_i64, 0], vec![999_i64; 3], vec![false; 3]),
            (0, vec![], vec![999_i64; 3], vec![false; 3]),
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
                    .ragged_new(r#type, &[rows.into(), Array::vector(vec![999_i64; 3]).unwrap().into()])
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
        let rows = Array::vector(vec![1_i64, 0]).unwrap();
        let columns =
            Array::from_elements(r#type.extent_types()[1].clone(), &[2_i64, 999, 999, 999, 999, 999]).unwrap();
        let depth = Array::vector(vec![1_i64, 2, 999, 999]).unwrap();
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
        let rows = ArrayIrValue::Array(Array::vector(vec![1_i64, 3]).unwrap());
        let columns = ArrayIrValue::Array(Array::vector(vec![3_i64, 1]).unwrap());
        // Each marginal includes coordinate 2, but no common batch item contains (row 2, column 2).
        let depth = ArrayIrValue::Array(
            Array::from_elements(r#type.extent_types()[2].clone(), &[1_i64, 2, 0, 1, 999, 999, 2, 999, 999]).unwrap(),
        );
        assert!(storage(&r#type).ragged_new(r#type.clone(), &[rows.clone(), columns.clone(), depth]).is_ok());
        let changed = ArrayIrValue::Array(
            Array::from_elements(r#type.extent_types()[2].clone(), &[1_i64, 2, 0, 1, 999, 999, 999, 999, 999]).unwrap(),
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
        let first = ArrayIrValue::Array(Array::vector(vec![1_i64, 2]).unwrap());
        let other = ArrayIrValue::Array(Array::vector(vec![2_i64, 1]).unwrap());
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
            Array::from_elements(r#type.extent_types()[0].clone(), &[1_i64, 3, 2, 4, 0, 1]).unwrap(),
        );
        let second = ArrayIrValue::Array(
            Array::from_elements(r#type.extent_types()[1].clone(), &[1_i64, 2, 0, 3, 4, 1]).unwrap(),
        );
        assert!(storage(&r#type).ragged_new(r#type.clone(), &[first.clone(), second]).is_ok());
        let changed = ArrayIrValue::Array(
            Array::from_elements(r#type.extent_types()[1].clone(), &[1_i64, 2, 0, 3, 4, 2]).unwrap(),
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
        let extent = ArrayIrValue::Array(Array::scalar(2_i64).unwrap());
        let value = storage(&r#type).ragged_new(r#type, &[extent.clone()]).unwrap();
        assert_eq!(value.ragged_extents(0), Ok(extent));
    }

    #[test]
    fn test_ragged_new_interpretation_identity_source_is_not_proof() {
        let r#type = nested_type(2);
        let ArrayIrValue::Array(storage) = storage(&r#type) else { unreachable!() };
        let rows = Array::vector(vec![1_i64, 0]).unwrap();
        let columns = Array::from_elements(r#type.extent_types()[1].clone(), &[999_i64; 6]).unwrap();
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
            .ragged_new(r#type.clone(), &[Array::vector(vec![1_i64, 2]).unwrap().into()])
            .unwrap();
        let error = storage(&r#type)
            .ragged_new_with_identity_source(r#type, &[Array::vector(vec![3_i64, 3]).unwrap().into()], &source)
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
        let extents = ArrayIrValue::Array(Array::vector(vec![1_i64, 2]).unwrap());
        let source = storage(&source_type)
            .ragged_new(source_type.clone(), &[extents.clone(), Array::vector(vec![4_i64, 0]).unwrap().into()])
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
                    Array::vector(vec![-1_i64, 0]).unwrap(),
                    Array::from_elements(r#type.extent_types()[1].clone(), &[999_i64; 6]).unwrap(),
                ],
            )
            .unwrap(),
        );
        let error = storage(&r#type)
            .ragged_new_with_identity_source(
                r#type.clone(),
                &[
                    Array::vector(vec![0_i64, 0]).unwrap().into(),
                    Array::from_elements(r#type.extent_types()[1].clone(), &[999_i64; 6]).unwrap().into(),
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
            ArrayIrValue::Array(Array::vector(vec![1_i64, 2]).unwrap()).ragged_new(r#type.clone(), &[]).unwrap();
        let error = ArrayIrValue::Array(Array::vector(vec![3_i64, 3]).unwrap())
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
            Array::from_elements(source_type.storage_type().clone(), &[1_i64, 2, 3, 4, 5, 6]).unwrap(),
        )
        .ragged_new(source_type, &[])
        .unwrap();
        let output_type =
            RaggedType::new(ArrayType::new_static(DataType::I64, [3, 2]), RaggedValueKind::Dimension(item), vec![])
                .unwrap();
        let output = ArrayIrValue::Array(
            Array::from_elements(output_type.storage_type().clone(), &[1_i64, 4, 2, 5, 3, 6]).unwrap(),
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
            (DimensionType::from(Dimension::Static(3)), vec![3_i64, 3]),
            (unbounded, vec![0_i64, MAX_DIMENSION_EXTENT as i64]),
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
        assert!(ArrayIrValue::Array(Array::scalar(3_i64).unwrap()).ragged_new(scalar_type.clone(), &[]).is_ok());
        let error = ArrayIrValue::Array(Array::scalar(2_i64).unwrap()).ragged_new(scalar_type, &[]).unwrap_err();
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
        let rows = ArrayIrValue::Array(Array::vector(vec![1_i64, 0]).unwrap());
        let value = ArrayIrValue::Array(
            Array::from_elements(r#type.storage_type().clone(), &[7_i64, -99, -99, 999, 999, 999]).unwrap(),
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
        let values = Array::from_elements(r#type.storage_type().clone(), &[3_i64, 999, -99, 999, 999, 999]).unwrap();
        let extents = Array::vector(vec![1_i64, 0]).unwrap();
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
            .interpret((storage(&r#type), ArrayIrValue::Array(Array::vector(vec![999_i64, 2]).unwrap())))
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
            Array::from_fn_elements(ArrayType::new_static(DataType::F32, [2, 2, 3]), |_| Ok(1_f32)).unwrap(),
        );
        assert!(program.interpret((valid_storage, Array::vector(vec![1_i64; 2]).unwrap().into())).is_ok());
        let invalid_storage = ArrayIrValue::Array(
            Array::from_fn_elements(ArrayType::new_static(DataType::F32, [3, 2, 3]), |_| Ok(1_f32)).unwrap(),
        );
        assert_eq!(
            program.interpret((invalid_storage, Array::vector(vec![1_i64; 3]).unwrap().into())),
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
            RaggedValue::new(r#type, storage, vec![Array::vector(vec![999_i64, 1]).unwrap()]).unwrap(),
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
            evaluation.interpret(&context, &[ArrayIrValue::Array(Array::vector(vec![1_i64, 3]).unwrap())]),
            Ok(vec![known_storage]),
        );
        let error = evaluation
            .interpret(&context, &[ArrayIrValue::Array(Array::vector(vec![999_i64, 3]).unwrap())])
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
        let storage = Array::vector(vec![1_f32, 2., 3.]).unwrap();
        assert_eq!(program.interpret((storage.clone(), Array::scalar(2_i64).unwrap())), Ok(storage.clone()));
        let error = program.interpret((storage.clone(), Array::scalar(999_i64).unwrap())).unwrap_err();
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
        assert_eq!(evaluation.interpret(&context, &[Array::scalar(2_i64).unwrap()]), Ok(vec![storage]));
        let error = evaluation.interpret(&context, &[Array::scalar(999_i64).unwrap()]).unwrap_err();
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
}
