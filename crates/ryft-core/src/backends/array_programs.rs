use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::axes::AxisIndexOperation;
use crate::backends::arrays::{Array, ArrayOperation, BroadcastKernel};
use crate::backends::dimensions::{DimensionOperation, DimensionValue};
use crate::contexts::{Context, EagerContext, ProjectedContext};
use crate::differentiation::{
    BroadcastDerivativeAlignment, DifferentiableOperation, DifferentiableType, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, ElementwiseDerivativeAlignment, LinearCallOperation,
    ResidualZeroProvider, TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
#[cfg(test)]
use crate::operations::collectives::CollectiveOptions;
use crate::operations::collectives::{
    ALL_GATHER_OPERATION_NAME, ALL_TO_ALL_OPERATION_NAME, AllGatherOperation, AllGatherOutputVariance,
    AllToAllOperation, CollectiveMode, PSUM_SCATTER_OPERATION_NAME, PSumScatterOperation, PpermuteOperation,
    infer_explicit_all_gather_output_types, infer_explicit_all_to_all_output_types,
    infer_explicit_psum_scatter_output_types,
};
use crate::operations::compare::{Compare, CompareOperation, ComparisonDirection};
use crate::operations::constants::{
    ConstantOperation, Iota, IotaOperation, One, OneOperation, ZERO_OPERATION_NAME, Zero, ZeroLikeOperation,
    ZeroOperation, ZeroOperationProvider, check_constructor_type_has_no_identity_references,
    infer_dynamic_constructor_output_types,
};
use crate::operations::control_flow::scan::{
    ScanInterpretation, read_scan_iteration, stacked_scan_type, write_scan_iteration,
};
use crate::operations::control_flow::{
    ConditionOperation, ScanOperation, SelectOperation, WhileOperation, WhilePredicate,
};
use crate::operations::custom_call::{CustomCall, CustomCallOperation};
use crate::operations::dimensions::{
    DimensionAddOperation, DimensionFromScalar, DimensionFromScalarOperation, DimensionMulOperation,
    DimensionSaturatingSubOperation, DimensionSize, DimensionSizeOperation, DimensionToScalar,
    DimensionToScalarOperation,
};
use crate::operations::manipulation::ConvertElementTypeOperation;
use crate::operations::manipulation::broadcasting::infer_explicit_broadcast_output_type;
use crate::operations::manipulation::{
    Broadcast, BroadcastOperation, CONCATENATE_OPERATION_NAME, Concatenate, ConcatenateOperation,
    DynamicShapeSliceOperation, DynamicSliceOperation, DynamicUpdateSliceOperation, LegacyBroadcastOperation,
    LegacyReshapeOperation, Pad, PadOperation, Reshape, ReshapeOperation, ReshapeParameters, ScatterDimensionNumbers,
    ScatterOperation, ScatterReductionKind, Slice, Transpose, TransposeOperation, UpdateSlice, UpdateSliceOperation,
};
use crate::operations::math::{AddOperation, DivOperation, MulOperation, Reduce, ReduceOperation, ReductionKind};
use crate::operations::random::{RngBitGenerator, RngBitGeneratorOperation};
use crate::parameters::Parameter;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation,
};
use crate::programs::atoms::MaybeZero;
use crate::programs::effects::Effects;
use crate::programs::identities::{TypeIdentityPosition, TypeIdentityRenaming};
use crate::programs::operations::{Operation, OperationProjection};
use crate::programs::regions::{EmptyRegionDriver, OutputRegionProvenance, RegionInterface, RegionSlot};
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{Concretizable, ProjectedValue, Value, ValueProjection};
use crate::programs::{AtomId, ProgramBuilder, ProgramError};
use crate::sharding::Sharding;
use crate::tracing::{NestedTracingContext, Tracer, TracingContext};
use crate::types::{
    ArrayProgramType, ArrayType, DataType, Dimension, DimensionBounds, DimensionError, DimensionType,
    DimensionVariable, Shape,
};

pub mod batching;
mod differentiation;

/// One axis of an exact runtime shape retained by a linearization rule.
#[derive(Copy, Clone, Debug)]
enum ExactShapeDimension {
    /// Compile-time extent that can be reconstructed as a dimension constant in either attached region.
    Static(usize),

    /// Index of the ordinary SSA residual that carries this dynamic extent.
    Residual(usize),
}

/// Exact runtime shape whose dynamic axes refer to entries in a [`LinearResiduals`] list.
#[derive(Clone, Debug)]
struct ExactShape(Vec<ExactShapeDimension>);

impl ExactShape {
    /// Builds the canonical disconnected-cotangent shape plan and records each dynamic identity's first source axis.
    fn for_residual_zero(shape: &Shape) -> (Self, Vec<(usize, DimensionVariable)>) {
        let mut first_axes = Vec::new();
        // Residual slots are assigned by first axis occurrence. Repeated uses of one dimension identity reuse that
        // slot, preserving equality between axes without retaining duplicate scalar values.
        let dimensions = shape
            .dimensions()
            .iter()
            .enumerate()
            .map(|(axis, dimension)| match dimension {
                Dimension::Static(extent) => ExactShapeDimension::Static(*extent),
                Dimension::Dynamic(variable) => {
                    let residual =
                        first_axes.iter().position(|(_, candidate)| candidate == variable).unwrap_or_else(|| {
                            let residual = first_axes.len();
                            first_axes.push((axis, variable.clone()));
                            residual
                        });
                    ExactShapeDimension::Residual(residual)
                }
            })
            .collect();
        (Self(dimensions), first_axes)
    }

    /// Materializes one first-class dimension value per axis in this shape.
    fn dimensions<A, C>(&self, context: &C, residuals: &[C::Value]) -> Result<Vec<C::Value>, ProgramError>
    where
        A: Value<Type = ArrayType>,
        C: Context<Type = ArrayProgramType, Operation: From<ArrayProgramOperation<A>>>,
    {
        self.0
            .iter()
            .map(|dimension| match dimension {
                ExactShapeDimension::Static(extent) => dimension_constant::<A, _>(context, *extent),
                ExactShapeDimension::Residual(index) => Ok(residuals[*index].clone()),
            })
            .collect()
    }

    /// Returns the residual values required by dynamic array constructors, in dynamic-axis order.
    fn dynamic_dimensions<V: Clone>(&self, residuals: &[V]) -> Vec<V> {
        // Mixed array constructors consume one operand per dynamic axis. Expand deduplicated residual slots back into
        // axis order here, so repeated identities intentionally produce repeated operands.
        self.0
            .iter()
            .filter_map(|dimension| match dimension {
                ExactShapeDimension::Static(_) => None,
                ExactShapeDimension::Residual(index) => Some(residuals[*index].clone()),
            })
            .collect()
    }
}

/// Ordered residual list used by an extent-sensitive linearization rule.
///
/// Dynamic dimensions are deduplicated by identity, so repeated axes and explicit dimension operands share one SSA
/// residual. Ordinary array residuals remain positional and are never inspected.
#[derive(Clone, Debug)]
struct LinearResiduals<V: Value<Type = ArrayProgramType>> {
    /// Residual values in linear-call operand order.
    values: Vec<V>,
}

impl<V: Value<Type = ArrayProgramType>> LinearResiduals<V> {
    /// Creates an empty residual list.
    #[inline]
    fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Returns the retained residual values.
    #[inline]
    fn values(&self) -> &[V] {
        self.values.as_slice()
    }

    /// Consumes this residual list and returns its values.
    #[inline]
    fn into_values(self) -> Vec<V> {
        self.values
    }

    /// Retains `value`, deduplicating dynamic dimension definitions by identity, and returns its residual index.
    fn retain(&mut self, value: V) -> usize {
        if let ArrayProgramType::Dimension(r#type) = value.r#type().as_ref()
            && r#type.extent().is_none()
            && let Some(index) = self.values.iter().position(|value| {
                matches!(
                    value.r#type().as_ref(),
                    ArrayProgramType::Dimension(candidate) if candidate.variable() == r#type.variable()
                )
            })
        {
            return index;
        }
        let index = self.values.len();
        self.values.push(value);
        index
    }

    /// Retains an ordered value list and returns the residual index corresponding to each source value.
    fn retain_all(&mut self, values: impl IntoIterator<Item = V>) -> Vec<usize> {
        values.into_iter().map(|value| self.retain(value)).collect()
    }

    /// Retains the exact runtime shape of `array`, reading only dynamic axes that are not already represented by a
    /// dimension residual with the same identity.
    fn retain_shape<A, C>(&mut self, context: &C, array: &V) -> Result<ExactShape, ProgramError>
    where
        A: Value<Type = ArrayType>,
        C: Context<Type = ArrayProgramType, Value = V, Operation: From<ArrayProgramOperation<A>>>,
    {
        let array_type = array.r#type();
        let array_type = <&ArrayType>::try_from(array_type.as_ref())?;
        array_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .map(|(axis, dimension)| match dimension {
                Dimension::Static(extent) => Ok(ExactShapeDimension::Static(*extent)),
                Dimension::Dynamic(variable) => {
                    if let Some(index) = self.values.iter().position(|value| {
                        matches!(
                            value.r#type().as_ref(),
                            ArrayProgramType::Dimension(r#type) if r#type.variable() == variable
                        )
                    }) {
                        return Ok(ExactShapeDimension::Residual(index));
                    }
                    let extent = context
                        .bind(
                            ArrayProgramOperation::<A>::from(DimensionSizeOperation::new(array_type, axis)?),
                            Vec::new(),
                            std::slice::from_ref(array),
                        )?
                        .remove(0);
                    Ok(ExactShapeDimension::Residual(self.retain(extent)))
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map(ExactShape)
    }
}

/// Materializes one exact first-class dimension constant in a composite array-program context.
fn dimension_constant<A, C>(context: &C, extent: usize) -> Result<C::Value, ProgramError>
where
    A: Value<Type = ArrayType>,
    C: Context<Type = ArrayProgramType, Operation: From<ArrayProgramOperation<A>>>,
{
    Ok(context
        .bind(
            ArrayProgramOperation::<A>::from(DimensionOperation::from(ConstantOperation::new(
                DimensionValue::constant(extent)?,
            ))),
            Vec::new(),
            &[],
        )?
        .remove(0))
}

// TODO(eaplatanios): Review this module.

// TODO(eaplatanios): Should we flatten `ArrayOperation` into this and rename this to `ArrayOperation`?
/// Closed [`Operation`] family for array programs that contain both ordinary arrays and first-class runtime
/// dimensions. This dispatcher preserves the homogeneous operation contracts of [`ArrayOperation`] and
/// [`DimensionOperation`]: it selects the member family, projects the composite type boundary once, delegates to that
/// family, and lifts the inferred result types back into [`ArrayProgramType`].
///
/// Operations whose signatures mix arrays and dimensions are represented as explicit variants because no homogeneous
/// member family can express such a signature. For example, [`DimensionSizeOperation`] consumes an array and produces
/// a first-class dimension without changing either homogeneous family.
#[derive(Clone, Debug)]
pub enum ArrayProgramOperation<A: Value<Type = ArrayType>> {
    /// Mixed zero constructor whose stored [`ArrayType`] defines the array result and whose dynamic dimensions are
    /// consumed as explicit first-class dimension operands, one per dynamic axis in axis order. This constructor
    /// lives at the composite-family level because its signature crosses member kinds: a homogeneous
    /// [`ArrayOperation`] cannot consume dimension operands, while the stored structural type carries identities and
    /// bounds but not the concrete runtime extents required to materialize the result.
    Zero(ZeroOperation<ArrayType>),

    /// Mixed one constructor whose stored [`ArrayType`] fully defines the output type and whose dynamic dimensions
    /// are consumed as explicit first-class dimension operands, one per dynamic axis in axis order.
    DynamicOne(OneOperation<ArrayType>),

    /// Mixed iota constructor whose stored [`ArrayType`] and iota axis define the complete output, and whose dynamic
    /// dimensions are consumed as explicit first-class dimension operands in axis order.
    DynamicIota(IotaOperation<ArrayType>),

    /// Region-free homogeneous array operation. Member control-flow operations are promoted to their direct
    /// composite carriers when an array program is unprojected.
    Array(ArrayOperation<A>),

    /// Homogeneous first-class-dimension operation.
    Dimension(DimensionOperation<DimensionValue>),

    /// Mixed comparison of two first-class dimensions that produces ordinary rank-zero Boolean array data.
    ///
    /// This variant has the precise composite member signature
    /// `(Dimension, Dimension) -> Array(Boolean scalar)`. It lives directly in [`ArrayProgramOperation`] because
    /// [`DimensionOperation`] is intentionally homogeneous: its inputs and outputs are all first-class dimensions.
    /// Storing comparison there would break that invariant because a predicate is ordinary data rather than a
    /// first-class dimension.
    ///
    /// Homogeneous array comparison remains [`ArrayProgramOperation::Array`] wrapping
    /// [`ArrayOperation::Compare`]. This variant does not permit array-dimension or dimension-array comparisons; it
    /// reuses [`CompareOperation`] for the dimension-dimension signature whose result crosses from the dimension
    /// member kind to the array member kind.
    Compare(CompareOperation<ArrayProgramType>),

    /// Mixed operation that reads an array axis as a first-class dimension.
    DimensionSize(DimensionSizeOperation),

    /// Mixed operation that converts ordinary scalar-array data into a checked first-class dimension.
    DimensionFromScalar(DimensionFromScalarOperation),

    /// Mixed operation that converts a first-class dimension into ordinary scalar-array data.
    DimensionToScalar(DimensionToScalarOperation),

    /// Mixed operation that reshapes an array using one first-class dimension operand per output axis.
    Reshape(ReshapeOperation),

    /// Mixed operation that broadcasts an array using one first-class dimension operand per output axis.
    Broadcast(BroadcastOperation),

    /// Mixed operation that concatenates array operands using one trailing result-extent operand.
    Concatenate(ConcatenateOperation),

    /// Mixed foreign-kernel call whose trailing dimension operands define its dynamic output axes.
    CustomCall(CustomCallOperation),

    /// Mixed padding operation with one explicit result-extent operand per output axis.
    Pad(PadOperation),

    /// Mixed slice whose starts and output sizes are first-class dimension operands.
    DynamicShapeSlice(DynamicShapeSliceOperation),

    /// Mixed bit generator whose trailing dimension operands define its dynamic bits-output axes.
    RngBitGenerator(RngBitGeneratorOperation<ArrayProgramType>),

    /// Mixed all-gather whose trailing dimension operands define every result axis in axis order.
    AllGather(AllGatherOperation),

    /// Mixed sum-scatter whose trailing dimension operands define every result axis in axis order.
    PSumScatter(PSumScatterOperation),

    /// Mixed all-to-all whose trailing dimension operands define every result axis in axis order.
    AllToAll(AllToAllOperation),

    /// Composite condition whose attached branches may carry arrays and first-class dimensions.
    Condition(ConditionOperation<ArrayProgramValue<A>>),

    /// Composite while loop whose condition and body may carry arrays and first-class dimensions.
    While(WhileOperation<ArrayProgramType>),

    /// Composite scan whose body may carry arrays and first-class dimensions.
    Scan(ScanOperation<ArrayProgramValue<A>>),

    /// Differentiation-owned executable linear call with ordinary trailing residual operands.
    LinearCall(LinearCallOperation<ArrayProgramType>),
}

impl<A: Value<Type = ArrayType>> Display for ArrayProgramOperation<A> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<A: Value<Type = ArrayType>> From<ArrayOperation<A>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: ArrayOperation<A>) -> Self {
        match operation {
            ArrayOperation::Zero(operation) => Self::from(operation),
            ArrayOperation::Condition(_) => Self::Condition(ConditionOperation::new()),
            ArrayOperation::While(operation) => {
                Self::While(WhileOperation::new().with_iteration_bound(operation.iteration_bound()).unwrap())
            }
            ArrayOperation::Scan(operation) => {
                let captures = operation.captures().iter().cloned().map(ArrayProgramValue::Array).collect();
                Self::Scan(operation.with_captures(captures))
            }
            operation => Self::Array(operation),
        }
    }
}

impl<A: Value<Type = ArrayType>> From<DimensionOperation<DimensionValue>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: DimensionOperation<DimensionValue>) -> Self {
        Self::Dimension(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<ConditionOperation<ArrayProgramValue<A>>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: ConditionOperation<ArrayProgramValue<A>>) -> Self {
        Self::Condition(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<WhileOperation<ArrayProgramType>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: WhileOperation<ArrayProgramType>) -> Self {
        Self::While(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<ScanOperation<ArrayProgramValue<A>>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: ScanOperation<ArrayProgramValue<A>>) -> Self {
        Self::Scan(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<CompareOperation<ArrayProgramType>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: CompareOperation<ArrayProgramType>) -> Self {
        Self::Compare(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<DimensionSizeOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: DimensionSizeOperation) -> Self {
        Self::DimensionSize(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<DimensionFromScalarOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: DimensionFromScalarOperation) -> Self {
        Self::DimensionFromScalar(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<DimensionToScalarOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: DimensionToScalarOperation) -> Self {
        Self::DimensionToScalar(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<ReshapeOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: ReshapeOperation) -> Self {
        Self::Reshape(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<BroadcastOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: BroadcastOperation) -> Self {
        Self::Broadcast(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<ConcatenateOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: ConcatenateOperation) -> Self {
        Self::Concatenate(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<CustomCallOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: CustomCallOperation) -> Self {
        Self::CustomCall(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<PadOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: PadOperation) -> Self {
        Self::Pad(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<DynamicShapeSliceOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: DynamicShapeSliceOperation) -> Self {
        Self::DynamicShapeSlice(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<RngBitGeneratorOperation<ArrayProgramType>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: RngBitGeneratorOperation<ArrayProgramType>) -> Self {
        Self::RngBitGenerator(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<AllGatherOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: AllGatherOperation) -> Self {
        Self::AllGather(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<PSumScatterOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: PSumScatterOperation) -> Self {
        Self::PSumScatter(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<AllToAllOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: AllToAllOperation) -> Self {
        Self::AllToAll(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<ZeroOperation<ArrayType>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: ZeroOperation<ArrayType>) -> Self {
        // Each zero has one canonical encoding: identity-free static zeros already belong to the homogeneous array
        // member family, and only reference-bearing dynamic output types need the mixed dimension-operand variant.
        if operation
            .r#type()
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            Self::Zero(operation)
        } else {
            Self::Array(ArrayOperation::Zero(operation))
        }
    }
}

// These residual-protocol algorithms are deliberately inherent methods duplicated by thin trait-impl delegations
// below rather than living only on the trait: the `XlaOperation` provider reuses them across operation families,
// which their `Self`-typed builder and context parameters cannot express. Do not fold them into the trait impl.
impl<A: Value<Type = ArrayType>> ArrayProgramOperation<A> {
    /// Captures the runtime extents needed to materialize a disconnected cotangent zero from the primal `source`.
    pub fn capture_zero_residuals<
        V: Value<Type = ArrayProgramType>,
        O: Operation<ArrayProgramType> + From<DimensionSizeOperation>,
    >(
        builder: &mut ProgramBuilder<V, O>,
        source: AtomId,
        r#type: &ArrayProgramType,
    ) -> Result<Vec<AtomId>, ProgramError> {
        let r#type = <&ArrayType>::try_from(r#type)?;
        let (_, first_axes) = ExactShape::for_residual_zero(r#type.shape());
        if first_axes.is_empty() {
            return Ok(Vec::new());
        }
        let source_type = builder
            .atoms()
            .get(source.index())
            .ok_or(ProgramError::UnboundAtomId { id: source })?
            .r#type()
            .into_owned();
        let source_type = <&ArrayType>::try_from(&source_type)?;
        // Reading only each identity's first source axis establishes the same deduplicated residual ordering used by
        // zero construction, including when several axes share one identity.
        first_axes
            .into_iter()
            .map(|(axis, _)| {
                Ok(builder.add_instruction(
                    DimensionSizeOperation::new(source_type, axis)?,
                    Vec::new(),
                    vec![source],
                )?[0])
            })
            .collect()
    }

    /// Captures the value-level counterpart of
    /// [`capture_zero_residuals`](Self::capture_zero_residuals) in `context`.
    pub fn capture_zero_residual_values<C>(
        context: &C,
        source: &C::Value,
        r#type: &ArrayProgramType,
    ) -> Result<Vec<C::Value>, ProgramError>
    where
        C: Context<Type = ArrayProgramType>,
        C::Operation: From<DimensionSizeOperation>,
    {
        let r#type = <&ArrayType>::try_from(r#type)?;
        let (_, first_axes) = ExactShape::for_residual_zero(r#type.shape());
        if first_axes.is_empty() {
            return Ok(Vec::new());
        }
        let source_type = source.r#type();
        let source_type = <&ArrayType>::try_from(source_type.as_ref())?;
        // Capture one concrete extent per identity from its first source axis; repeated axes reuse that value when the
        // dynamic zero's per-axis operand list is reconstructed.
        first_axes
            .into_iter()
            .map(|(axis, _)| {
                Ok(context
                    .bind(DimensionSizeOperation::new(source_type, axis)?, Vec::new(), std::slice::from_ref(source))?
                    .remove(0))
            })
            .collect()
    }

    /// Returns the canonical zero operation for `r#type` and expands one explicit extent residual per distinct dynamic
    /// identity into the mixed constructor's per-axis operand order.
    pub fn zero_operation_with_residuals<R: Clone>(
        r#type: ArrayProgramType,
        residuals: &[R],
    ) -> Result<(Self, Vec<R>), ProgramError> {
        let r#type = <&ArrayType>::try_from(&r#type)?.clone();
        let (shape, first_axes) = ExactShape::for_residual_zero(r#type.shape());
        let expected_residual_count = first_axes.len();
        if residuals.len() != expected_residual_count {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "dynamic zero expected {expected_residual_count} extent residuals but got {}",
                    residuals.len(),
                ),
            });
        }
        if first_axes.is_empty() {
            return Ok((Self::zero_operation(r#type.into())?, Vec::new()));
        }
        let operands = shape.dynamic_dimensions(residuals);
        Ok((Self::Zero(ZeroOperation::new(r#type)), operands))
    }
}

impl<A: Value<Type = ArrayType>> ZeroOperationProvider<ArrayProgramType> for ArrayProgramOperation<A> {
    fn zero_operation(r#type: ArrayProgramType) -> Result<Self, ProgramError> {
        let ArrayProgramType::Array(r#type) = r#type else {
            return Err(TypeError::invalid("cannot materialize a zero for a first-class dimension type").into());
        };
        check_constructor_type_has_no_identity_references(ZERO_OPERATION_NAME, &r#type)?;
        Ok(Self::Array(ArrayOperation::Zero(ZeroOperation::new(r#type))))
    }
}

impl<A: Value<Type = ArrayType>> ResidualZeroProvider<ArrayProgramType> for ArrayProgramOperation<A> {
    fn zero_residual_types(r#type: &ArrayProgramType) -> Vec<ArrayProgramType> {
        match r#type {
            ArrayProgramType::Array(r#type) => {
                let (_, first_axes) = ExactShape::for_residual_zero(r#type.shape());
                first_axes.into_iter().map(|(_, variable)| DimensionType::new(variable).into()).collect()
            }
            ArrayProgramType::Dimension(_) => Vec::new(),
        }
    }

    fn capture_zero_residuals<V: Value<Type = ArrayProgramType>>(
        builder: &mut ProgramBuilder<V, Self>,
        source: AtomId,
        r#type: &ArrayProgramType,
    ) -> Result<Vec<AtomId>, ProgramError> {
        ArrayProgramOperation::<A>::capture_zero_residuals(builder, source, r#type)
    }

    fn capture_zero_residual_values<C: Context<Type = ArrayProgramType, Operation = Self>>(
        context: &C,
        source: &C::Value,
        r#type: &ArrayProgramType,
    ) -> Result<Vec<C::Value>, ProgramError> {
        ArrayProgramOperation::<A>::capture_zero_residual_values(context, source, r#type)
    }

    fn zero_operation_with_residuals<R: Clone>(
        r#type: ArrayProgramType,
        residuals: &[R],
    ) -> Result<(Self, Vec<R>), ProgramError> {
        ArrayProgramOperation::<A>::zero_operation_with_residuals(r#type, residuals)
    }
}

impl<A: Value<Type = ArrayType>> From<OneOperation<ArrayType>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: OneOperation<ArrayType>) -> Self {
        // Each one has one canonical encoding: identity-free static ones already belong to the homogeneous array
        // member family, and only reference-bearing dynamic output types need the mixed dimension-operand variant.
        if operation
            .r#type()
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            Self::DynamicOne(operation)
        } else {
            Self::Array(ArrayOperation::One(operation))
        }
    }
}

impl<A: Value<Type = ArrayType>> From<IotaOperation<ArrayType>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: IotaOperation<ArrayType>) -> Self {
        if operation
            .r#type()
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            Self::DynamicIota(operation)
        } else {
            Self::Array(ArrayOperation::Iota(operation))
        }
    }
}

impl<A: Value<Type = ArrayType>> From<AddOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: AddOperation) -> Self {
        Self::Array(ArrayOperation::Add(operation))
    }
}

impl<A: Value<Type = ArrayType>> From<PpermuteOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: PpermuteOperation) -> Self {
        Self::Array(ArrayOperation::Ppermute(operation))
    }
}

impl<A: Value<Type = ArrayType>> From<LinearCallOperation<ArrayProgramType>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: LinearCallOperation<ArrayProgramType>) -> Self {
        Self::LinearCall(operation)
    }
}

impl<A: Value<Type = ArrayType>> OperationProjection<ArrayType> for ArrayProgramOperation<A> {
    type Projected = ArrayOperation<A>;
}

impl<A: Value<Type = ArrayType>> OperationProjection<DimensionType> for ArrayProgramOperation<A> {
    type Projected = DimensionOperation<DimensionValue>;
}

/// Projects the complete inference boundary for one homogeneous inner operation while preserving region effects.
fn project_operation_boundary<T: Type>(
    input_types: &[ArrayProgramType],
    region_interfaces: &[RegionInterface<ArrayProgramType>],
) -> Result<(Vec<T>, Vec<RegionInterface<T>>), TypeError>
where
    for<'t> &'t T: TryFrom<&'t ArrayProgramType, Error = TypeError>,
{
    Ok((
        input_types.iter().map(|r#type| <&T>::try_from(r#type).cloned()).collect::<Result<_, _>>()?,
        region_interfaces
            .iter()
            .map(|interface| {
                Ok(RegionInterface::new(
                    interface
                        .input_types()
                        .iter()
                        .map(|r#type| <&T>::try_from(r#type).cloned())
                        .collect::<Result<_, _>>()?,
                    interface
                        .output_types()
                        .iter()
                        .map(|r#type| <&T>::try_from(r#type).cloned())
                        .collect::<Result<_, _>>()?,
                    interface.effects(),
                ))
            })
            .collect::<Result<Vec<_>, TypeError>>()?,
    ))
}

impl<A: Value<Type = ArrayType>> Operation<ArrayProgramType> for ArrayProgramOperation<A> {
    #[inline]
    fn name(&self) -> &'static str {
        match self {
            Self::Zero(operation) => operation.name(),
            Self::DynamicOne(operation) => operation.name(),
            Self::DynamicIota(operation) => operation.name(),
            Self::Array(operation) => operation.name(),
            Self::Dimension(operation) => operation.name(),
            Self::Compare(operation) => Operation::<ArrayProgramType>::name(operation),
            Self::DimensionSize(operation) => operation.name(),
            Self::DimensionFromScalar(operation) => operation.name(),
            Self::DimensionToScalar(operation) => operation.name(),
            Self::Reshape(operation) => operation.name(),
            Self::Broadcast(operation) => operation.name(),
            Self::Concatenate(operation) => Operation::<ArrayProgramType>::name(operation),
            Self::CustomCall(operation) => Operation::<ArrayProgramType>::name(operation),
            Self::Pad(operation) => Operation::<ArrayProgramType>::name(operation),
            Self::DynamicShapeSlice(operation) => operation.name(),
            Self::RngBitGenerator(operation) => Operation::<ArrayProgramType>::name(operation),
            Self::AllGather(operation) => Operation::<ArrayType>::name(operation),
            Self::PSumScatter(operation) => Operation::<ArrayType>::name(operation),
            Self::AllToAll(operation) => Operation::<ArrayType>::name(operation),
            Self::Condition(operation) => operation.name(),
            Self::While(operation) => Operation::<ArrayProgramType>::name(operation),
            Self::Scan(operation) => operation.name(),
            Self::LinearCall(operation) => operation.name(),
        }
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        match self {
            Self::Zero(operation) => operation.region_slots(),
            Self::DynamicOne(operation) => operation.region_slots(),
            Self::DynamicIota(operation) => operation.region_slots(),
            Self::Array(operation) => operation.region_slots(),
            Self::Dimension(operation) => operation.region_slots(),
            Self::Compare(operation) => Operation::<ArrayProgramType>::region_slots(operation),
            Self::DimensionSize(operation) => operation.region_slots(),
            Self::DimensionFromScalar(operation) => operation.region_slots(),
            Self::DimensionToScalar(operation) => operation.region_slots(),
            Self::Reshape(operation) => operation.region_slots(),
            Self::Broadcast(operation) => operation.region_slots(),
            Self::Concatenate(operation) => Operation::<ArrayProgramType>::region_slots(operation),
            Self::CustomCall(operation) => Operation::<ArrayProgramType>::region_slots(operation),
            Self::Pad(operation) => Operation::<ArrayProgramType>::region_slots(operation),
            Self::DynamicShapeSlice(operation) => operation.region_slots(),
            Self::RngBitGenerator(operation) => Operation::<ArrayProgramType>::region_slots(operation),
            Self::AllGather(operation) => Operation::<ArrayType>::region_slots(operation),
            Self::PSumScatter(operation) => Operation::<ArrayType>::region_slots(operation),
            Self::AllToAll(operation) => Operation::<ArrayType>::region_slots(operation),
            Self::Condition(operation) => operation.region_slots(),
            Self::While(operation) => Operation::<ArrayProgramType>::region_slots(operation),
            Self::Scan(operation) => operation.region_slots(),
            Self::LinearCall(operation) => operation.region_slots(),
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<Option<Vec<ArrayProgramType>>>, TypeError> {
        match self {
            Self::Zero(_) | Self::DynamicOne(_) | Self::DynamicIota(_) => Ok(vec![None; region_interfaces.len()]),
            Self::Array(operation) => {
                let (input_types, region_interfaces) = project_operation_boundary(input_types, region_interfaces)?;
                Ok(operation
                    .infer_region_input_types(&input_types, &region_interfaces)?
                    .into_iter()
                    .map(|types| types.map(|types| types.into_iter().map(Into::into).collect()))
                    .collect())
            }
            Self::Dimension(operation) => {
                let (input_types, region_interfaces) = project_operation_boundary(input_types, region_interfaces)?;
                Ok(operation
                    .infer_region_input_types(&input_types, &region_interfaces)?
                    .into_iter()
                    .map(|types| types.map(|types| types.into_iter().map(Into::into).collect()))
                    .collect())
            }
            Self::Compare(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::DimensionSize(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::DimensionFromScalar(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::DimensionToScalar(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::Reshape(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::Broadcast(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::Concatenate(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::CustomCall(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::Pad(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::DynamicShapeSlice(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::RngBitGenerator(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::AllGather(_) | Self::PSumScatter(_) | Self::AllToAll(_) => Ok(vec![None; region_interfaces.len()]),
            Self::Condition(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::While(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::Scan(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::LinearCall(operation) => operation.infer_region_input_types(input_types, region_interfaces),
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<ArrayProgramType>, TypeError> {
        match self {
            Self::Zero(operation) => infer_dynamic_constructor_output_types(
                operation.name(),
                operation.r#type(),
                input_types,
                region_interfaces,
            ),
            Self::DynamicOne(operation) => infer_dynamic_constructor_output_types(
                operation.name(),
                operation.r#type(),
                input_types,
                region_interfaces,
            ),
            Self::DynamicIota(operation) => infer_dynamic_constructor_output_types(
                operation.name(),
                operation.r#type(),
                input_types,
                region_interfaces,
            ),
            Self::Array(operation) => {
                let (input_types, region_interfaces) = project_operation_boundary(input_types, region_interfaces)?;
                Ok(operation
                    .infer_output_types(&input_types, &region_interfaces)?
                    .into_iter()
                    .map(Into::into)
                    .collect())
            }
            Self::Dimension(operation) => {
                let (input_types, region_interfaces) = project_operation_boundary(input_types, region_interfaces)?;
                Ok(operation
                    .infer_output_types(&input_types, &region_interfaces)?
                    .into_iter()
                    .map(Into::into)
                    .collect())
            }
            Self::Compare(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::DimensionSize(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::DimensionFromScalar(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::DimensionToScalar(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Reshape(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Broadcast(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Concatenate(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::CustomCall(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Pad(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::DynamicShapeSlice(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::RngBitGenerator(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::AllGather(operation) => {
                check_count!("region", region_interfaces, 0, TypeError);
                infer_explicit_all_gather_output_types(operation, input_types)
            }
            Self::PSumScatter(operation) => {
                check_count!("region", region_interfaces, 0, TypeError);
                infer_explicit_psum_scatter_output_types(operation, input_types)
            }
            Self::AllToAll(operation) => {
                check_count!("region", region_interfaces, 0, TypeError);
                infer_explicit_all_to_all_output_types(operation, input_types)
            }
            Self::Condition(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::While(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Scan(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::LinearCall(operation) => operation.infer_output_types(input_types, region_interfaces),
        }
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        match self {
            Self::Zero(operation) => operation.output_region_provenance(output_index),
            Self::DynamicOne(operation) => operation.output_region_provenance(output_index),
            Self::DynamicIota(operation) => operation.output_region_provenance(output_index),
            Self::Array(operation) => operation.output_region_provenance(output_index),
            Self::Dimension(operation) => operation.output_region_provenance(output_index),
            Self::Compare(operation) => {
                Operation::<ArrayProgramType>::output_region_provenance(operation, output_index)
            }
            Self::DimensionSize(operation) => operation.output_region_provenance(output_index),
            Self::DimensionFromScalar(operation) => operation.output_region_provenance(output_index),
            Self::DimensionToScalar(operation) => operation.output_region_provenance(output_index),
            Self::Reshape(operation) => operation.output_region_provenance(output_index),
            Self::Broadcast(operation) => operation.output_region_provenance(output_index),
            Self::Concatenate(operation) => {
                Operation::<ArrayProgramType>::output_region_provenance(operation, output_index)
            }
            Self::CustomCall(operation) => {
                Operation::<ArrayProgramType>::output_region_provenance(operation, output_index)
            }
            Self::Pad(operation) => Operation::<ArrayProgramType>::output_region_provenance(operation, output_index),
            Self::DynamicShapeSlice(operation) => operation.output_region_provenance(output_index),
            Self::RngBitGenerator(operation) => {
                Operation::<ArrayProgramType>::output_region_provenance(operation, output_index)
            }
            Self::AllGather(operation) => Operation::<ArrayType>::output_region_provenance(operation, output_index),
            Self::PSumScatter(operation) => Operation::<ArrayType>::output_region_provenance(operation, output_index),
            Self::AllToAll(operation) => Operation::<ArrayType>::output_region_provenance(operation, output_index),
            Self::Condition(operation) => operation.output_region_provenance(output_index),
            Self::While(operation) => Operation::<ArrayProgramType>::output_region_provenance(operation, output_index),
            Self::Scan(operation) => operation.output_region_provenance(output_index),
            Self::LinearCall(operation) => operation.output_region_provenance(output_index),
        }
    }

    #[inline]
    fn is_zero(&self, output_index: usize) -> bool {
        match self {
            Self::Zero(operation) => operation.is_zero(output_index),
            Self::DynamicOne(operation) => operation.is_zero(output_index),
            Self::DynamicIota(operation) => operation.is_zero(output_index),
            Self::Array(operation) => operation.is_zero(output_index),
            Self::Dimension(operation) => operation.is_zero(output_index),
            Self::Compare(operation) => Operation::<ArrayProgramType>::is_zero(operation, output_index),
            Self::DimensionSize(operation) => operation.is_zero(output_index),
            Self::DimensionFromScalar(operation) => operation.is_zero(output_index),
            Self::DimensionToScalar(operation) => operation.is_zero(output_index),
            Self::Reshape(operation) => operation.is_zero(output_index),
            Self::Broadcast(operation) => operation.is_zero(output_index),
            Self::Concatenate(operation) => Operation::<ArrayProgramType>::is_zero(operation, output_index),
            Self::CustomCall(operation) => Operation::<ArrayProgramType>::is_zero(operation, output_index),
            Self::Pad(operation) => Operation::<ArrayProgramType>::is_zero(operation, output_index),
            Self::DynamicShapeSlice(operation) => operation.is_zero(output_index),
            Self::RngBitGenerator(operation) => Operation::<ArrayProgramType>::is_zero(operation, output_index),
            Self::AllGather(operation) => Operation::<ArrayType>::is_zero(operation, output_index),
            Self::PSumScatter(operation) => Operation::<ArrayType>::is_zero(operation, output_index),
            Self::AllToAll(operation) => Operation::<ArrayType>::is_zero(operation, output_index),
            Self::Condition(operation) => operation.is_zero(output_index),
            Self::While(operation) => Operation::<ArrayProgramType>::is_zero(operation, output_index),
            Self::Scan(operation) => operation.is_zero(output_index),
            Self::LinearCall(operation) => operation.is_zero(output_index),
        }
    }

    #[inline]
    fn effects(&self) -> Effects {
        match self {
            Self::Zero(operation) => operation.effects(),
            Self::DynamicOne(operation) => operation.effects(),
            Self::DynamicIota(operation) => operation.effects(),
            Self::Array(operation) => operation.effects(),
            Self::Dimension(operation) => operation.effects(),
            Self::Compare(operation) => Operation::<ArrayProgramType>::effects(operation),
            Self::DimensionSize(operation) => operation.effects(),
            Self::DimensionFromScalar(operation) => operation.effects(),
            Self::DimensionToScalar(operation) => operation.effects(),
            Self::Reshape(operation) => operation.effects(),
            Self::Broadcast(operation) => operation.effects(),
            Self::Concatenate(operation) => Operation::<ArrayProgramType>::effects(operation),
            Self::CustomCall(operation) => Operation::<ArrayProgramType>::effects(operation),
            Self::Pad(operation) => Operation::<ArrayProgramType>::effects(operation),
            Self::DynamicShapeSlice(operation) => operation.effects(),
            Self::RngBitGenerator(operation) => Operation::<ArrayProgramType>::effects(operation),
            Self::AllGather(operation) => Operation::<ArrayType>::effects(operation),
            Self::PSumScatter(operation) => Operation::<ArrayType>::effects(operation),
            Self::AllToAll(operation) => Operation::<ArrayType>::effects(operation),
            Self::Condition(operation) => operation.effects(),
            Self::While(operation) => Operation::<ArrayProgramType>::effects(operation),
            Self::Scan(operation) => operation.effects(),
            Self::LinearCall(operation) => operation.effects(),
        }
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        match self {
            Self::Zero(operation) => Ok(Self::Zero(operation.rename_type_identities(renaming)?)),
            Self::DynamicOne(operation) => Ok(Self::DynamicOne(operation.rename_type_identities(renaming)?)),
            Self::DynamicIota(operation) => Ok(Self::DynamicIota(operation.rename_type_identities(renaming)?)),
            Self::Array(operation) => Ok(Self::Array(operation.rename_type_identities(renaming)?)),
            Self::Dimension(operation) => Ok(Self::Dimension(operation.rename_type_identities(renaming)?)),
            Self::Compare(operation) => {
                Ok(Self::Compare(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::DimensionSize(operation) => Ok(Self::DimensionSize(operation.rename_type_identities(renaming)?)),
            Self::DimensionFromScalar(operation) => {
                Ok(Self::DimensionFromScalar(operation.rename_type_identities(renaming)?))
            }
            Self::DimensionToScalar(operation) => {
                Ok(Self::DimensionToScalar(operation.rename_type_identities(renaming)?))
            }
            Self::Reshape(operation) => Ok(Self::Reshape(operation.rename_type_identities(renaming)?)),
            Self::Broadcast(operation) => Ok(Self::Broadcast(operation.rename_type_identities(renaming)?)),
            Self::Concatenate(operation) => {
                Ok(Self::Concatenate(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::CustomCall(operation) => {
                Ok(Self::CustomCall(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::Pad(operation) => {
                Ok(Self::Pad(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::DynamicShapeSlice(operation) => {
                Ok(Self::DynamicShapeSlice(operation.rename_type_identities(renaming)?))
            }
            Self::RngBitGenerator(operation) => {
                Ok(Self::RngBitGenerator(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::AllGather(operation) => {
                Ok(Self::AllGather(Operation::<ArrayType>::rename_type_identities(operation, renaming)?))
            }
            Self::PSumScatter(operation) => {
                Ok(Self::PSumScatter(Operation::<ArrayType>::rename_type_identities(operation, renaming)?))
            }
            Self::AllToAll(operation) => {
                Ok(Self::AllToAll(Operation::<ArrayType>::rename_type_identities(operation, renaming)?))
            }
            Self::Condition(operation) => Ok(Self::Condition(operation.rename_type_identities(renaming)?)),
            Self::While(operation) => {
                Ok(Self::While(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::Scan(operation) => Ok(Self::Scan(operation.rename_type_identities(renaming)?)),
            Self::LinearCall(operation) => Ok(Self::LinearCall(operation.rename_type_identities(renaming)?)),
        }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(operation) => operation.render(formatter, indentation),
            Self::DynamicOne(operation) => operation.render(formatter, indentation),
            Self::DynamicIota(operation) => operation.render(formatter, indentation),
            Self::Array(operation) => operation.render(formatter, indentation),
            Self::Dimension(operation) => operation.render(formatter, indentation),
            Self::Compare(operation) => Operation::<ArrayProgramType>::render(operation, formatter, indentation),
            Self::DimensionSize(operation) => operation.render(formatter, indentation),
            Self::DimensionFromScalar(operation) => operation.render(formatter, indentation),
            Self::DimensionToScalar(operation) => operation.render(formatter, indentation),
            Self::Reshape(operation) => operation.render(formatter, indentation),
            Self::Broadcast(operation) => operation.render(formatter, indentation),
            Self::Concatenate(operation) => Operation::<ArrayProgramType>::render(operation, formatter, indentation),
            Self::CustomCall(operation) => Operation::<ArrayProgramType>::render(operation, formatter, indentation),
            Self::Pad(operation) => Operation::<ArrayProgramType>::render(operation, formatter, indentation),
            Self::DynamicShapeSlice(operation) => operation.render(formatter, indentation),
            Self::RngBitGenerator(operation) => {
                Operation::<ArrayProgramType>::render(operation, formatter, indentation)
            }
            Self::AllGather(operation) => Operation::<ArrayType>::render(operation, formatter, indentation),
            Self::PSumScatter(operation) => Operation::<ArrayType>::render(operation, formatter, indentation),
            Self::AllToAll(operation) => Operation::<ArrayType>::render(operation, formatter, indentation),
            Self::Condition(operation) => operation.render(formatter, indentation),
            Self::While(operation) => Operation::<ArrayProgramType>::render(operation, formatter, indentation),
            Self::Scan(operation) => operation.render(formatter, indentation),
            Self::LinearCall(operation) => operation.render(formatter, indentation),
        }
    }
}

/// Interprets one homogeneous operation family using its native eager domain and lifts the results back into the
/// composite value family. Common operation arities stay on the stack; only wider operations allocate an input vector.
fn interpret_homogeneous_operation<
    T: Type,
    V: Value<Type = T>,
    O: Operation<T> + InterpretableOperation<EagerContext<V, O>>,
    A: Value<Type = ArrayType>,
>(
    operation: &O,
    inputs: &[ArrayProgramValue<A>],
) -> Result<Vec<ArrayProgramValue<A>>, ProgramError>
where
    ArrayProgramValue<A>: ValueProjection<T, Projected = V>,
{
    let context = EagerContext::<V, O>::new();
    let interpret = |inputs: &[V]| operation.interpret(&context, &EmptyRegionDriver, inputs);
    let outputs = match inputs {
        [] => interpret(&[]),
        [input] => {
            let inputs = [<ArrayProgramValue<A> as ValueProjection<T>>::into_projected(input.clone())?];
            interpret(&inputs)
        }
        [left, right] => {
            let inputs = [
                <ArrayProgramValue<A> as ValueProjection<T>>::into_projected(left.clone())?,
                <ArrayProgramValue<A> as ValueProjection<T>>::into_projected(right.clone())?,
            ];
            interpret(&inputs)
        }
        inputs => {
            let inputs = inputs
                .iter()
                .cloned()
                .map(<ArrayProgramValue<A> as ValueProjection<T>>::into_projected)
                .collect::<Result<Vec<_>, _>>()?;
            interpret(&inputs)
        }
    }?;
    Ok(outputs.into_iter().map(<ArrayProgramValue<A> as ValueProjection<T>>::from_projected).collect())
}

/// Resolves one mixed constructor's explicit dimension operands into the concrete static output type required by an
/// eager backend.
fn materialize_dynamic_constructor_type<A: Value<Type = ArrayType>>(
    name: &str,
    r#type: &ArrayType,
    inputs: &[ArrayProgramValue<A>],
) -> Result<ArrayType, ProgramError> {
    let expected = r#type
        .shape()
        .dimensions()
        .iter()
        .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        .count();
    if expected == 0 {
        return Err(TypeError::invalid(format!(
            "'{name}' with static output type {type} has no dynamic dimensions; use the homogeneous nullary \
             constructor instead",
            r#type = r#type,
        ))
        .into());
    }
    let mut extents = inputs.iter();
    let dimensions = r#type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| match dimension {
            Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
            Dimension::Dynamic(variable) => {
                let extent =
                    extents.next().ok_or(ProgramError::InvalidInputCount { expected, actual: inputs.len() })?;
                let extent: &DimensionValue =
                    <ArrayProgramValue<A> as ValueProjection<DimensionType>>::projected(extent)?;
                // Eager binds skip inference and intermediate results skip boundary refinement checks, so validate
                // each runtime extent against the stored output axis's authoritative bounds before allocation.
                // Identity equality is deliberately not required because interpreted inputs may be alpha-renamed.
                if !variable.bounds().contains(extent.extent()) {
                    return Err(DimensionError::BindingOutOfBounds {
                        variable: variable.to_string(),
                        value: extent.extent(),
                        bounds: variable.bounds(),
                    }
                    .into());
                }
                Ok(Dimension::Static(extent.extent()))
            }
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    if extents.next().is_some() {
        return Err(ProgramError::InvalidInputCount { expected, actual: inputs.len() });
    }
    Ok(r#type.clone().with_shape(Shape::new(dimensions)))
}

/// Resolves and validates one shape-changing collective's complete axis-ordered result extents.
fn explicit_collective_output_extents<A: Value<Type = ArrayType>>(
    operation_name: &str,
    inputs: &[ArrayProgramValue<A>],
    expected_extents: &[usize],
) -> Result<Vec<usize>, ProgramError> {
    let expected_input_count = 1 + expected_extents.len();
    if inputs.len() != expected_input_count {
        return Err(ProgramError::InvalidInputCount { expected: expected_input_count, actual: inputs.len() });
    }
    let output_extents = inputs[1..]
        .iter()
        .map(<ArrayProgramValue<A> as ValueProjection<DimensionType>>::projected)
        .map(|result| result.map(DimensionValue::extent))
        .collect::<Result<Vec<_>, _>>()?;
    for (axis, (&expected, &actual)) in expected_extents.iter().zip(&output_extents).enumerate() {
        if actual != expected {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "'{operation_name}' output axis {axis} extent must equal observed result extent {expected} but got \
                     {actual}",
                ),
            });
        }
    }
    Ok(output_extents)
}

impl<
    A: BroadcastKernel
        + Concatenate
        + Concretizable<bool>
        + CustomCall
        + DimensionFromScalar<DimensionValue>
        + DimensionSize<usize>
        + Pad
        + Reshape
        + RngBitGenerator
        + Slice
        + UpdateSlice
        + Value<Type = ArrayType>
        + WhilePredicate,
> InterpretableOperation<EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>> for ArrayProgramOperation<A>
where
    DimensionValue: Compare<A> + DimensionToScalar<A>,
    EagerContext<A, ArrayOperation<A>>: Iota<A> + One<A> + Zero<A>,
    ArrayOperation<A>: InterpretableOperation<EagerContext<A, ArrayOperation<A>>>,
    DimensionOperation<DimensionValue>:
        InterpretableOperation<EagerContext<DimensionValue, DimensionOperation<DimensionValue>>>,
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>>>(
        &self,
        context: &EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>,
        driver: &D,
        inputs: &[ArrayProgramValue<A>],
    ) -> Result<Vec<ArrayProgramValue<A>>, ProgramError> {
        if !matches!(self, Self::Condition(_) | Self::While(_) | Self::Scan(_) | Self::LinearCall(_))
            && (!self.region_slots().is_empty() || driver.region_count() != 0)
        {
            return Err(ProgramError::MalformedProgram(format!(
                "projected operation `{}` cannot carry regions",
                self.name(),
            )));
        }
        match self {
            Self::Zero(operation) => {
                let output_type = materialize_dynamic_constructor_type(operation.name(), operation.r#type(), inputs)?;
                Ok(vec![ArrayProgramValue::Array(EagerContext::<A, ArrayOperation<A>>::new().zero(&output_type)?)])
            }
            Self::DynamicOne(operation) => {
                let output_type = materialize_dynamic_constructor_type(operation.name(), operation.r#type(), inputs)?;
                Ok(vec![ArrayProgramValue::Array(EagerContext::<A, ArrayOperation<A>>::new().one(&output_type)?)])
            }
            Self::DynamicIota(operation) => {
                let output_type = materialize_dynamic_constructor_type(operation.name(), operation.r#type(), inputs)?;
                Ok(vec![ArrayProgramValue::Array(
                    EagerContext::<A, ArrayOperation<A>>::new().iota(&output_type, operation.dimension())?,
                )])
            }
            Self::Array(operation) => interpret_homogeneous_operation(operation, inputs),
            Self::Dimension(operation) => interpret_homogeneous_operation(operation, inputs),
            Self::Compare(operation) => operation.interpret(
                &EagerContext::<ArrayProgramValue<A>, ArrayProgramOperation<A>>::new(),
                &EmptyRegionDriver,
                inputs,
            ),
            Self::DimensionSize(operation) => operation.interpret(
                &EagerContext::<ArrayProgramValue<A>, ArrayProgramOperation<A>>::new(),
                &EmptyRegionDriver,
                inputs,
            ),
            Self::DimensionFromScalar(operation) => operation.interpret(
                &EagerContext::<ArrayProgramValue<A>, ArrayProgramOperation<A>>::new(),
                &EmptyRegionDriver,
                inputs,
            ),
            Self::DimensionToScalar(operation) => operation.interpret(
                &EagerContext::<ArrayProgramValue<A>, ArrayProgramOperation<A>>::new(),
                &EmptyRegionDriver,
                inputs,
            ),
            Self::Reshape(operation) => {
                let Some((input, output_extents)) = inputs.split_first() else {
                    return Err(TypeError::invalid("'reshape' expects an array followed by its output extents").into());
                };
                let input = <ArrayProgramValue<A> as ValueProjection<ArrayType>>::projected(input)?;
                let output_shape = Shape::new(
                    output_extents
                        .iter()
                        .map(<ArrayProgramValue<A> as ValueProjection<DimensionType>>::projected)
                        .map(|result| result.map(|extent: &DimensionValue| Dimension::Static(extent.extent())))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                let mut parameters = ReshapeParameters::new(output_shape);
                if let Some(dimensions) = operation.dimensions() {
                    parameters = parameters.with_dimensions(dimensions.clone());
                }
                if let Some(output_sharding) = operation.output_sharding() {
                    parameters = parameters.with_output_sharding(output_sharding.clone());
                }
                Ok(vec![ArrayProgramValue::Array(input.reshape(parameters)?)])
            }
            Self::Broadcast(operation) => {
                let Some((input, output_extents)) = inputs.split_first() else {
                    return Err(
                        TypeError::invalid("'broadcast' expects an array followed by its output extents").into()
                    );
                };
                let input = <ArrayProgramValue<A> as ValueProjection<ArrayType>>::projected(input)?;
                let output_shape = Shape::new(
                    output_extents
                        .iter()
                        .map(<ArrayProgramValue<A> as ValueProjection<DimensionType>>::projected)
                        .map(|result| result.map(|extent: &DimensionValue| Dimension::Static(extent.extent())))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                let output_type =
                    infer_explicit_broadcast_output_type(input.r#type().as_ref(), output_shape, operation)?;
                Ok(vec![ArrayProgramValue::Array(input.broadcast_to_type(output_type, operation.output_axes())?)])
            }
            Self::Concatenate(operation) => {
                let Some((result_extent, inputs)) = inputs.split_last() else {
                    return Err(TypeError::invalid(format!(
                        "'{}' expects at least one array followed by its result extent",
                        CONCATENATE_OPERATION_NAME,
                    ))
                    .into());
                };
                if inputs.is_empty() {
                    return match result_extent {
                        ArrayProgramValue::Array(_) => Err(TypeError::invalid(format!(
                            "'{}' expects a trailing result-extent dimension",
                            CONCATENATE_OPERATION_NAME,
                        ))
                        .into()),
                        ArrayProgramValue::Dimension(_) => Err(TypeError::invalid(format!(
                            "'{}' expects at least one array before its result extent",
                            CONCATENATE_OPERATION_NAME,
                        ))
                        .into()),
                    };
                }
                let result_extent = <ArrayProgramValue<A> as ValueProjection<DimensionType>>::projected(result_extent)?;
                let inputs = inputs
                    .iter()
                    .map(<ArrayProgramValue<A> as ValueProjection<ArrayType>>::projected)
                    .collect::<Result<Vec<_>, _>>()?;
                let actual_extent = inputs.iter().try_fold(0usize, |extent, input| {
                    extent.checked_add(input.dimension_size(operation.axis())?).ok_or_else(|| {
                        ProgramError::from(TypeError::invalid(format!(
                            "'{}' result extent overflows usize",
                            CONCATENATE_OPERATION_NAME,
                        )))
                    })
                })?;
                if result_extent.extent() != actual_extent {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "'{}' result extent must equal the sum of input axis {} extents; expected {actual_extent} \
                             but got {}",
                            CONCATENATE_OPERATION_NAME,
                            operation.axis(),
                            result_extent.extent(),
                        ),
                    });
                }
                Ok(vec![ArrayProgramValue::Array(A::concatenate(inputs.iter().copied(), operation.axis())?)])
            }
            Self::CustomCall(operation) => operation.interpret(
                &EagerContext::<ArrayProgramValue<A>, ArrayProgramOperation<A>>::new(),
                &EmptyRegionDriver,
                inputs,
            ),
            Self::Pad(operation) => operation.interpret(
                &EagerContext::<ArrayProgramValue<A>, ArrayProgramOperation<A>>::new(),
                &EmptyRegionDriver,
                inputs,
            ),
            Self::DynamicShapeSlice(operation) => {
                let Some((input, bounds)) = inputs.split_first() else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
                };
                let input = <ArrayProgramValue<A> as ValueProjection<ArrayType>>::projected(input)?;
                let rank = input.r#type().rank();
                check_count!("input", inputs, 1 + 2 * rank, ProgramError);
                let starts = bounds[..rank]
                    .iter()
                    .map(<ArrayProgramValue<A> as ValueProjection<DimensionType>>::projected)
                    .map(|result| result.map(DimensionValue::extent))
                    .collect::<Result<Vec<_>, _>>()?;
                let sizes = bounds[rank..]
                    .iter()
                    .map(<ArrayProgramValue<A> as ValueProjection<DimensionType>>::projected)
                    .map(|result| result.map(DimensionValue::extent))
                    .collect::<Result<Vec<_>, _>>()?;
                let limits = starts
                    .iter()
                    .zip(&sizes)
                    .zip(operation.strides())
                    .enumerate()
                    .map(|(axis, ((start, size), stride))| {
                        let span = if *size == 0 {
                            0
                        } else {
                            size.checked_sub(1)
                                .and_then(|size| size.checked_mul(*stride))
                                .and_then(|span| span.checked_add(1))
                                .ok_or_else(|| {
                                    TypeError::invalid(format!(
                                        "'dynamic_shape_slice' span overflows usize on axis {axis}",
                                    ))
                                })?
                        };
                        let limit = start.checked_add(span).ok_or_else(|| {
                            TypeError::invalid(format!("'dynamic_shape_slice' limit overflows usize on axis {axis}",))
                        })?;
                        let input_size = input.dimension_size(axis)?;
                        if limit > input_size {
                            return Err(ProgramError::InvalidArgument {
                                message: format!(
                                    "'dynamic_shape_slice' limit {limit} exceeds input axis {axis} extent \
                                     {input_size}",
                                ),
                            });
                        }
                        Ok(limit)
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                Ok(vec![ArrayProgramValue::Array(input.slice(&starts, &limits, operation.strides())?)])
            }
            Self::RngBitGenerator(operation) => operation.interpret(
                &EagerContext::<ArrayProgramValue<A>, ArrayProgramOperation<A>>::new(),
                &EmptyRegionDriver,
                inputs,
            ),
            Self::AllGather(operation) => {
                let effective_axis_size = operation.effective_axis_size()?;
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                infer_explicit_all_gather_output_types(operation, input_types.as_slice())?;
                let Some(input) = inputs.first() else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
                };
                let input = <ArrayProgramValue<A> as ValueProjection<ArrayType>>::projected(input)?;
                let mut expected_extents =
                    (0..input.r#type().rank()).map(|axis| input.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
                match operation.options().mode() {
                    CollectiveMode::Untiled => {
                        if operation.concat_axis() > expected_extents.len() {
                            return Err(TypeError::invalid(format!(
                                "'all_gather' concat axis {} is out of bounds for rank {}",
                                operation.concat_axis(),
                                expected_extents.len(),
                            ))
                            .into());
                        }
                        expected_extents.insert(operation.concat_axis(), effective_axis_size);
                    }
                    CollectiveMode::Tiled => {
                        let rank = expected_extents.len();
                        let Some(output_extent) = expected_extents.get_mut(operation.concat_axis()) else {
                            return Err(TypeError::invalid(format!(
                                "'all_gather' concat axis {} is out of bounds for rank {rank}",
                                operation.concat_axis(),
                            ))
                            .into());
                        };
                        *output_extent = output_extent
                            .checked_mul(effective_axis_size)
                            .ok_or_else(|| TypeError::invalid("'all_gather' result extent does not fit in usize"))?;
                    }
                }
                let output_extents =
                    explicit_collective_output_extents(ALL_GATHER_OPERATION_NAME, inputs, &expected_extents)?;
                if effective_axis_size != 1 {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "cannot interpret 'all_gather' over axis '{}' of size {} without an enclosing binder",
                            operation.axis_name(),
                            effective_axis_size,
                        ),
                    });
                }
                let output = match operation.options().mode() {
                    CollectiveMode::Tiled => input.clone(),
                    CollectiveMode::Untiled => {
                        input.reshape(Shape::new(output_extents.into_iter().map(Dimension::Static).collect()))?
                    }
                };
                Ok(vec![ArrayProgramValue::Array(output)])
            }
            Self::PSumScatter(operation) => {
                let effective_axis_size = operation.effective_axis_size()?;
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                infer_explicit_psum_scatter_output_types(operation, input_types.as_slice())?;
                let Some(input) = inputs.first() else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
                };
                let input = <ArrayProgramValue<A> as ValueProjection<ArrayType>>::projected(input)?;
                let mut expected_extents =
                    (0..input.r#type().rank()).map(|axis| input.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
                if operation.scatter_axis() >= expected_extents.len() {
                    return Err(TypeError::invalid(format!(
                        "'psum_scatter' scatter axis {} is out of bounds for rank {}",
                        operation.scatter_axis(),
                        expected_extents.len(),
                    ))
                    .into());
                }
                match operation.options().mode() {
                    CollectiveMode::Untiled => {
                        if expected_extents[operation.scatter_axis()] != effective_axis_size {
                            return Err(TypeError::invalid(format!(
                                "'psum_scatter' untiled scatter axis {} size {} must equal group size \
                                 {effective_axis_size}",
                                operation.scatter_axis(),
                                expected_extents[operation.scatter_axis()],
                            ))
                            .into());
                        }
                        expected_extents.remove(operation.scatter_axis());
                    }
                    CollectiveMode::Tiled => {
                        if expected_extents[operation.scatter_axis()] % effective_axis_size != 0 {
                            return Err(TypeError::invalid(format!(
                                "'psum_scatter' scatter axis {} size {} is not divisible by group size \
                                 {effective_axis_size}",
                                operation.scatter_axis(),
                                expected_extents[operation.scatter_axis()],
                            ))
                            .into());
                        }
                        expected_extents[operation.scatter_axis()] /= effective_axis_size;
                    }
                }
                let output_extents =
                    explicit_collective_output_extents(PSUM_SCATTER_OPERATION_NAME, inputs, &expected_extents)?;
                if effective_axis_size != 1 {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "cannot interpret 'psum_scatter' over axis '{}' of size {} without an enclosing binder",
                            operation.axis_name(),
                            effective_axis_size,
                        ),
                    });
                }
                let output = match operation.options().mode() {
                    CollectiveMode::Tiled => input.clone(),
                    CollectiveMode::Untiled => {
                        input.reshape(Shape::new(output_extents.into_iter().map(Dimension::Static).collect()))?
                    }
                };
                Ok(vec![ArrayProgramValue::Array(output)])
            }
            Self::AllToAll(operation) => {
                let effective_axis_size = operation.effective_axis_size()?;
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                infer_explicit_all_to_all_output_types(operation, input_types.as_slice())?;
                let Some(input) = inputs.first() else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
                };
                let input = <ArrayProgramValue<A> as ValueProjection<ArrayType>>::projected(input)?;
                let mut expected_extents =
                    (0..input.r#type().rank()).map(|axis| input.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
                let rank = expected_extents.len();
                if operation.split_axis() >= rank || operation.concat_axis() >= rank {
                    return Err(TypeError::invalid(format!(
                        "'all_to_all' split axis {} or concat axis {} is out of bounds for rank {rank}",
                        operation.split_axis(),
                        operation.concat_axis(),
                    ))
                    .into());
                }
                match operation.options().mode() {
                    CollectiveMode::Untiled => {
                        if expected_extents[operation.split_axis()] != effective_axis_size {
                            return Err(TypeError::invalid(format!(
                                "'all_to_all' untiled split axis {} size {} must equal group size \
                                 {effective_axis_size}",
                                operation.split_axis(),
                                expected_extents[operation.split_axis()],
                            ))
                            .into());
                        }
                        expected_extents.remove(operation.split_axis());
                        expected_extents.insert(operation.concat_axis(), effective_axis_size);
                    }
                    CollectiveMode::Tiled if expected_extents[operation.split_axis()] % effective_axis_size != 0 => {
                        return Err(TypeError::invalid(format!(
                            "'all_to_all' split axis {} size {} is not divisible by group size {effective_axis_size}",
                            operation.split_axis(),
                            expected_extents[operation.split_axis()],
                        ))
                        .into());
                    }
                    CollectiveMode::Tiled if operation.split_axis() == operation.concat_axis() => {}
                    CollectiveMode::Tiled => {
                        expected_extents[operation.split_axis()] /= effective_axis_size;
                        expected_extents[operation.concat_axis()] =
                            expected_extents[operation.concat_axis()].checked_mul(effective_axis_size).ok_or_else(
                                || TypeError::invalid("'all_to_all' concatenation result extent does not fit in usize"),
                            )?;
                    }
                }
                let output_extents =
                    explicit_collective_output_extents(ALL_TO_ALL_OPERATION_NAME, inputs, &expected_extents)?;
                if effective_axis_size != 1 {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "cannot interpret 'all_to_all' over axis '{}' of size {} without an enclosing binder",
                            operation.axis_name(),
                            effective_axis_size,
                        ),
                    });
                }
                let output = match operation.options().mode() {
                    CollectiveMode::Tiled => input.clone(),
                    CollectiveMode::Untiled => {
                        input.reshape(Shape::new(output_extents.into_iter().map(Dimension::Static).collect()))?
                    }
                };
                Ok(vec![ArrayProgramValue::Array(output)])
            }
            Self::Condition(operation) => operation.interpret(context, driver, inputs),
            Self::While(operation) => operation.interpret(context, driver, inputs),
            Self::Scan(operation) => operation.interpret(context, driver, inputs),
            Self::LinearCall(operation) => operation.interpret(context, driver, inputs),
        }
    }
}

impl<
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayProgramType,
            Constant: Concretizable<bool>,
            Operation: From<ArrayProgramOperation<A>>
                           + From<ConditionOperation<C::Constant>>
                           + From<DimensionFromScalarOperation>
                           + From<DimensionToScalarOperation>
                           + From<ScanOperation<C::Constant>>
                           + From<WhileOperation<ArrayProgramType>>
                           + ZeroOperationProvider<ArrayProgramType>,
        >,
> PartiallyEvaluatableOperation<C> for ArrayProgramOperation<A>
where
    C::Value: PartialEq,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        if matches!(self, Self::Condition(_)) {
            return ConditionOperation::<C::Constant>::new().partially_evaluate(context, driver, inputs);
        }
        if let Self::Scan(operation) = self {
            let scan = ScanOperation::<C::Constant>::new(operation.carry_count(), operation.length())
                .with_reverse(operation.reverse())
                .with_unroll(operation.unroll())?;
            return scan.partially_evaluate(context, driver, inputs);
        }
        if let Self::While(operation) = self {
            return operation.partially_evaluate(context, driver, inputs);
        }
        if let Self::Reshape(operation) = self
            && operation.output_sharding().is_none()
            && driver.region_count() == 0
            && let Some(input) = inputs.first()
            && let Ok(input_type) = <&ArrayType>::try_from(input.r#type().as_ref())
            && input_type.static_shape().is_some()
            && operation.dimensions().is_none_or(|dimensions| dimensions.iter().copied().eq(0..input_type.rank()))
            && operation
                .infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?
                == vec![input.r#type().into_owned()]
        {
            // An entirely static identity reshape cannot observe its exact dimension operands. Preserve the input
            // directly so an unknown array does not leave a redundant reshape in the residual program.
            return Ok(vec![input.clone()]);
        }
        if let Self::Broadcast(operation) = self
            && operation.output_sharding().is_none()
            && driver.region_count() == 0
            && let Some(input) = inputs.first()
            && let Ok(input_type) = <&ArrayType>::try_from(input.r#type().as_ref())
            && input_type.static_shape().is_some()
            && operation.output_axes().iter().copied().eq(0..input_type.rank())
            && operation
                .infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?
                == vec![input.r#type().into_owned()]
        {
            // An entirely static identity broadcast cannot observe its exact dimension operands.
            return Ok(vec![input.clone()]);
        }
        context.fold_or_residualize(self.clone(), driver.regions().map(|region| region.to_program()).collect(), inputs)
    }
}

/// [`Value`]-level counterpart to [`ArrayProgramType`] that is used by [`Program`](crate::Program)s that may contain
/// both [`ArrayType`]-typed [`Value`]s and [`DimensionValue`]. `A` is the concrete array representation selected by the
/// owning backend. Dimensions use the common [`DimensionValue`] which is a checked host representation, so that eager
/// dimension arithmetic remains host integer work and does not allocate arrays or dispatch to device backends.
///
/// This type allows us to keep arrays and checked host-side dimensions in one storage universe while
/// [`ValueProjection`] lets homogeneous [`Operation`] machinery borrow or consume only
/// the member that it understands.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum ArrayProgramValue<A: Value<Type = ArrayType>> {
    /// Ordinary backend [`ArrayType`]-typed [`Value`].
    Array(A),

    /// Checked host-side runtime [`DimensionValue`].
    Dimension(DimensionValue),
}

impl<A: Value<Type = ArrayType>> Display for ArrayProgramValue<A> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Array(value) => Display::fmt(value, formatter),
            Self::Dimension(value) => Display::fmt(value, formatter),
        }
    }
}

// TODO(eaplatanios): Review from here onwards.

impl<A: Value<Type = ArrayType>> Typed for ArrayProgramValue<A> {
    type Type = ArrayProgramType;

    fn r#type(&self) -> Cow<'_, ArrayProgramType> {
        Cow::Owned(match self {
            Self::Array(value) => ArrayProgramType::Array(value.r#type().into_owned()),
            Self::Dimension(value) => ArrayProgramType::Dimension(value.r#type().clone()),
        })
    }
}

impl<A: Value<Type = ArrayType>> Value for ArrayProgramValue<A> {
    type DispatchDomain = EagerContext<Self>;
    type ExecutionDomain = EagerContext<Self, ArrayProgramOperation<A>>;

    #[inline]
    fn dispatch_domain(&self) -> Self::DispatchDomain {
        EagerContext::new()
    }

    #[inline]
    fn execution_domain(&self) -> Self::ExecutionDomain {
        EagerContext::new()
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        match self {
            Self::Array(value) => Ok(Self::Array(value.rename_type_identities(renaming)?)),
            Self::Dimension(value) => Ok(Self::Dimension(value.rename_type_identities(renaming)?)),
        }
    }
}

impl<A: Concretizable<bool> + Value<Type = ArrayType>> Concretizable<bool> for ArrayProgramValue<A> {
    fn concretize(&self) -> Result<bool, ProgramError> {
        match self {
            Self::Array(value) => value.concretize(),
            Self::Dimension(value) => Err(ProgramError::Concretization {
                message: format!("cannot extract a concrete boolean from a first-class dimension `{value}`"),
            }),
        }
    }
}

impl<A: Value<Type = ArrayType> + WhilePredicate> WhilePredicate for ArrayProgramValue<A> {
    fn any_true(&self) -> Result<bool, ProgramError> {
        match self {
            Self::Array(predicate) => predicate.any_true(),
            Self::Dimension(value) => Err(ProgramError::Concretization {
                message: format!("cannot use first-class dimension `{value}` as a while predicate"),
            }),
        }
    }

    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let Self::Array(predicate) = self else {
            return Err(ProgramError::Concretization {
                message: format!("cannot use first-class dimension `{self}` as a while predicate"),
            });
        };
        match (on_true, on_false) {
            (Self::Array(on_true), Self::Array(on_false)) => Ok(Self::Array(predicate.mask_select(on_true, on_false)?)),
            (Self::Dimension(on_true), Self::Dimension(on_false)) => {
                Ok(Self::Dimension(if predicate.concretize()? { on_true.clone() } else { on_false.clone() }))
            }
            _ => Err(TypeError::invalid(format!(
                "while predicate cannot select between mismatched state types {} and {}",
                on_true.r#type(),
                on_false.r#type(),
            ))
            .into()),
        }
    }
}

impl<A> ScanInterpretation<EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>> for ArrayProgramType
where
    A: Reshape + Slice + UpdateSlice + Value<Type = ArrayType>,
    EagerContext<A, ArrayOperation<A>>: Zero<A>,
{
    fn interpret_scan<D: InterpretationDriver<EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>>>(
        carry_count: usize,
        length: &Dimension,
        reverse: bool,
        context: &EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>,
        driver: &D,
        inputs: &[ArrayProgramValue<A>],
    ) -> Result<Vec<ArrayProgramValue<A>>, ProgramError> {
        let (inputs, length) = match length {
            Dimension::Static(length) => (inputs, *length),
            Dimension::Dynamic(variable) => {
                let (runtime_length, inputs) =
                    inputs.split_last().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
                let runtime_length =
                    <ArrayProgramValue<A> as ValueProjection<DimensionType>>::projected(runtime_length)?;
                if runtime_length.r#type().variable() != variable {
                    return Err(TypeError::invalid(format!(
                        "'scan' runtime length operand has type {} but scan length requires {variable}",
                        runtime_length.r#type(),
                    ))
                    .into());
                }
                (inputs, runtime_length.extent())
            }
        };
        let body = driver.region(0)?;
        let y_slice_types = body.interface().output_types()[carry_count..]
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let (initial_carries, stacks) = inputs.split_at(carry_count);
        let mut carries = initial_carries.to_vec();
        let array_context = EagerContext::<A, ArrayOperation<A>>::new();
        let mut accumulators = y_slice_types
            .iter()
            .map(|r#type| {
                let dimensions = r#type
                    .shape()
                    .dimensions()
                    .iter()
                    .map(|dimension| match dimension {
                        Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
                        Dimension::Dynamic(variable) => inputs
                            .iter()
                            .find_map(|input| match input {
                                ArrayProgramValue::Dimension(value) if value.r#type().variable() == variable => {
                                    Some(Dimension::Static(value.extent()))
                                }
                                _ => None,
                            })
                            .ok_or_else(|| {
                                TypeError::invalid(format!(
                                    "cannot eagerly allocate scan output {type} because its dynamic dimension \
                                     {variable} is not supplied as a first-class scan input",
                                    r#type = r#type,
                                ))
                            }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                array_context.zero(&stacked_scan_type(&r#type.clone().with_shape(Shape::new(dimensions)), length))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let iterations: Box<dyn Iterator<Item = usize>> =
            if reverse { Box::new((0..length).rev()) } else { Box::new(0..length) };
        for iteration in iterations {
            let mut iteration_inputs = carries.clone();
            iteration_inputs.extend(
                stacks
                    .iter()
                    .map(|stack| {
                        Ok(ArrayProgramValue::Array(read_scan_iteration(
                            <ArrayProgramValue<A> as ValueProjection<ArrayType>>::projected(stack)?,
                            iteration,
                        )?))
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?,
            );
            let mut iteration_outputs = driver.interpret_region(context, 0, iteration_inputs)?;
            check_count!("output", iteration_outputs, carry_count + y_slice_types.len(), ProgramError);
            let iteration_outputs_to_stack = iteration_outputs.split_off(carry_count);
            carries = iteration_outputs;
            for (accumulator, value) in accumulators.iter_mut().zip(iteration_outputs_to_stack) {
                let value = <ArrayProgramValue<A> as ValueProjection<ArrayType>>::into_projected(value)?;
                *accumulator = write_scan_iteration(accumulator.clone(), iteration, value)?;
            }
        }
        carries.extend(accumulators.into_iter().map(ArrayProgramValue::Array));
        Ok(carries)
    }
}

impl<A: DimensionSize<usize> + Value<Type = ArrayType>> DimensionSize for ArrayProgramValue<A> {
    fn dimension_size<AxisValue: Into<crate::Axis>>(&self, axis: AxisValue) -> Result<Self, ProgramError> {
        let array = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let input_type = array.r#type();
        let operation = DimensionSizeOperation::new(input_type.as_ref(), axis)?;
        let extent = <A as DimensionSize<usize>>::dimension_size(array, operation.axis())?;
        Ok(Self::Dimension(DimensionValue::new(operation.result_type().clone(), extent)?))
    }
}

impl<A: BroadcastKernel + DimensionSize<usize> + Value<Type = ArrayType>> Broadcast for ArrayProgramValue<A> {
    fn broadcast_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_axes: &[usize],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let output_shape = Shape::new(
            output_dimensions
                .iter()
                .map(<Self as ValueProjection<DimensionType>>::projected)
                .map(|result| result.map(|dimension| Dimension::Static(dimension.extent())))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let operation = BroadcastOperation::new(output_axes.to_vec()).with_output_sharding(output_sharding);
        let output_type = infer_explicit_broadcast_output_type(input.r#type().as_ref(), output_shape, &operation)?;
        Ok(Self::Array(input.broadcast_to_type(output_type, output_axes)?))
    }
}

impl Compare<Array> for DimensionValue {
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Array, ProgramError> {
        let result = match direction {
            ComparisonDirection::Equal => self.extent() == rhs.extent(),
            ComparisonDirection::NotEqual => self.extent() != rhs.extent(),
            ComparisonDirection::LessThan => self.extent() < rhs.extent(),
            ComparisonDirection::LessThanOrEqual => self.extent() <= rhs.extent(),
            ComparisonDirection::GreaterThan => self.extent() > rhs.extent(),
            ComparisonDirection::GreaterThanOrEqual => self.extent() >= rhs.extent(),
        };
        Ok(Array::scalar(result))
    }
}

impl DimensionToScalar<Array> for DimensionValue {
    fn to_scalar(&self) -> Result<Array, ProgramError> {
        // `DimensionValue::new` enforces the portable extent ceiling, which is no greater than `i64::MAX`.
        Ok(Array::scalar(i64::try_from(self.extent()).unwrap()))
    }
}

impl DimensionFromScalar<DimensionValue> for Array {
    fn to_dimension(&self, result: DimensionVariable) -> Result<DimensionValue, ProgramError> {
        let operation = DimensionFromScalarOperation::new(result);
        DimensionFromScalarOperation::validate_input_type(self.r#type().as_ref())?;
        let scalar = self.values()[0];
        let extent = match scalar {
            crate::backends::scalars::Scalar::I8(value) => usize::try_from(value),
            crate::backends::scalars::Scalar::I16(value) => usize::try_from(value),
            crate::backends::scalars::Scalar::I32(value) => usize::try_from(value),
            crate::backends::scalars::Scalar::I64(value) => usize::try_from(value),
            crate::backends::scalars::Scalar::U8(value) => Ok(usize::from(value)),
            crate::backends::scalars::Scalar::U16(value) => Ok(usize::from(value)),
            crate::backends::scalars::Scalar::U32(value) => usize::try_from(value),
            crate::backends::scalars::Scalar::U64(value) => usize::try_from(value),
            _ => unreachable!("dimension_from_scalar input type is validated before reading its payload"),
        }
        .map_err(|_| ProgramError::InvalidArgument {
            message: format!(
                "'{}' scalar input must be a nonnegative host-representable extent but is {scalar}",
                operation.name(),
            ),
        })?;
        Ok(DimensionValue::new(operation.result_type().clone(), extent)?)
    }
}

impl<A: Value<Type = ArrayType>> DimensionToScalar for ArrayProgramValue<A>
where
    DimensionValue: DimensionToScalar<A>,
{
    fn to_scalar(&self) -> Result<Self, ProgramError> {
        let dimension = <Self as ValueProjection<DimensionType>>::projected(self)?;
        Ok(Self::Array(<DimensionValue as DimensionToScalar<A>>::to_scalar(dimension)?))
    }
}

impl<A: DimensionFromScalar<DimensionValue> + Value<Type = ArrayType>> DimensionFromScalar for ArrayProgramValue<A> {
    fn to_dimension(&self, result: DimensionVariable) -> Result<Self, ProgramError> {
        let array = <Self as ValueProjection<ArrayType>>::projected(self)?;
        Ok(Self::Dimension(<A as DimensionFromScalar<DimensionValue>>::to_dimension(array, result)?))
    }
}

impl<A: Value<Type = ArrayType>> Compare for ArrayProgramValue<A>
where
    DimensionValue: Compare<A>,
{
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self, ProgramError> {
        let left = <Self as ValueProjection<DimensionType>>::projected(self)?;
        let right = <Self as ValueProjection<DimensionType>>::projected(rhs)?;
        Ok(Self::Array(left.compare(right, direction)?))
    }
}

impl<A: Value<Type = ArrayType>, O: Operation<ArrayProgramType>> Zero<ArrayProgramValue<A>>
    for EagerContext<ArrayProgramValue<A>, O>
where
    EagerContext<A, ArrayOperation<A>>: Zero<A>,
{
    fn zero(&self, r#type: &ArrayProgramType) -> Result<ArrayProgramValue<A>, ProgramError> {
        let array_type = <&ArrayType>::try_from(r#type)?;
        Ok(ArrayProgramValue::Array(EagerContext::<A, ArrayOperation<A>>::new().zero(array_type)?))
    }
}

impl<A: Value<Type = ArrayType>> ValueProjection<ArrayType> for ArrayProgramValue<A> {
    type Projected = A;
    type ProjectedRef<'v>
        = &'v A
    where
        Self: 'v;

    #[inline]
    fn from_projected(value: A) -> Self {
        Self::Array(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v A, TypeError>
    where
        ArrayType: 'v,
    {
        match self {
            Self::Array(value) => Ok(value),
            Self::Dimension(_) => Err(TypeError::invalid("expected array type but got dimension type")),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<A, TypeError> {
        match self {
            Self::Array(value) => Ok(value),
            Self::Dimension(_) => Err(TypeError::invalid("expected array type but got dimension type")),
        }
    }
}

impl<A: Value<Type = ArrayType>> ValueProjection<DimensionType> for ArrayProgramValue<A> {
    type Projected = DimensionValue;
    type ProjectedRef<'v>
        = &'v DimensionValue
    where
        Self: 'v;

    #[inline]
    fn from_projected(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v DimensionValue, TypeError>
    where
        DimensionType: 'v,
    {
        match self {
            Self::Array(_) => Err(TypeError::invalid("expected dimension type but got array type")),
            Self::Dimension(value) => Ok(value),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<DimensionValue, TypeError> {
        match self {
            Self::Array(_) => Err(TypeError::invalid("expected dimension type but got array type")),
            Self::Dimension(value) => Ok(value),
        }
    }
}

impl<A: Value<Type = ArrayType>> From<A> for ArrayProgramValue<A> {
    #[inline]
    fn from(value: A) -> Self {
        Self::Array(value)
    }
}

impl<A: Value<Type = ArrayType>> From<DimensionValue> for ArrayProgramValue<A> {
    #[inline]
    fn from(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::axes::NamedAxis;
    use crate::backends::array_programs::batching::{ArrayProgramBatch, ArrayProgramBatching};
    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::compilation::{
        CallRequest, CompilationDomain, CompilationTracer, CompileRequest, CompiledFunction, FlatCompilationProgram,
        JittedFunction, LoweredFunction, LoweringRequest, StageRequest, StagedFunction, try_jit,
    };
    use crate::contexts::{Context, StagingContext};
    use crate::differentiation::{DifferentiationTracer, ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::collectives::{
        AllGather, AllGatherOperation, AllToAllOperation, PSumScatter, PSumScatterOperation,
    };
    use crate::operations::constants::{ConstantOperation, ZeroOperation};
    use crate::operations::control_flow::{ConditionOperation, ScanOperation, WhileOperation};
    use crate::operations::differentiation::StopGradientOperation;
    use crate::operations::dimensions::{
        DimensionAddOperation, DimensionMulOperation, DimensionRequirementOperation, DimensionSizeOperation,
    };
    use crate::operations::manipulation::{
        Broadcast, BroadcastOperation, ConcatenateOperation, GatherDimensionNumbers, GatherOperation, PadOperation,
        ReshapeOperation, SliceOperation,
    };
    use crate::operations::math::{AddOperation, MulOperation};
    use crate::parameters::Placeholder;
    use crate::partial::PartialTracer;
    use crate::programs::AtomId;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::{Effect, Effects};
    use crate::programs::operations::OperationProjection;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType};
    use crate::tracing::{Tracer, TracingContext};
    use crate::types::{DataType, Dimension, DimensionBounds, DimensionVariable, Layout, Memory, Shape, StridedLayout};

    use super::*;

    #[test]
    fn test_array_program_explicit_collective_eager_contracts() {
        let context = EagerContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let extent = ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap());

        assert_eq!(
            context.bind(
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                PSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::tiled()),
                Vec::new(),
                &[
                    ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],)),
                    ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()),
                    ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap()),
                ],
            ),
            Ok(vec![ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],))]),
        );

        assert_eq!(
            context
                .bind(
                    AllGatherOperation::new(
                        "x".to_string(),
                        1,
                        0,
                        CollectiveOptions::tiled(),
                        AllGatherOutputVariance::Varying
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayProgramValue::Dimension(DimensionValue::constant(4).unwrap()),],
                )
                .unwrap_err()
                .to_string(),
            "'all_gather' result extent must equal input axis 0 extent 3 multiplied by axis group size 1; expected 3 \
             but got 4",
        );
        assert_eq!(
            context
                .bind(
                    AllGatherOperation::new(
                        "x".to_string(),
                        2,
                        0,
                        CollectiveOptions::tiled(),
                        AllGatherOutputVariance::Varying
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayProgramValue::Dimension(DimensionValue::constant(6).unwrap()),],
                )
                .unwrap_err(),
            ProgramError::UnsupportedOperation {
                message: "cannot interpret 'all_gather' over axis 'x' of size 2 without an enclosing binder"
                    .to_string(),
            },
        );
        assert_eq!(
            context
                .bind(
                    PSumScatterOperation::new("empty".to_string(), 0, 0, CollectiveOptions::tiled()),
                    Vec::new(),
                    &[input.clone(), extent.clone()],
                )
                .unwrap_err(),
            ProgramError::Type(TypeError::invalid("'psum_scatter' axis size must be greater than zero")),
        );

        check_operation_partial_evaluation!(
            backend = (ArrayProgramValue<Array>, ArrayProgramOperation<Array>),
            operation = AllGatherOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled(), AllGatherOutputVariance::Varying),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, extent.clone())],
                    outputs = [(@known, input.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, extent.clone()),
                    ],
                    outputs = [(@residual, input.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        let variable = DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap());
        let dimension_type = DimensionType::new(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![array, result_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let primal = ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let tangent = ArrayProgramValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let result_extent = ArrayProgramValue::Dimension(DimensionValue::new(dimension_type.clone(), 3).unwrap());
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![primal.clone(), result_extent.clone(), tangent.clone(),]),
            Ok(vec![primal, tangent]),
        );
        assert!(
            jvp.instructions()
                .iter()
                .any(|instruction| { matches!(instruction.operation(), ArrayProgramOperation::LinearCall(_)) })
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0])), result_extent])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayProgramValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
        let zero_extent = ArrayProgramValue::Dimension(DimensionValue::new(dimension_type, 0).unwrap());
        let zero_array = || {
            ArrayProgramValue::Array(Array::from_f64s(
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(0)])),
                Vec::new(),
            ))
        };
        let mut primal_outputs = linearization.primal().interpret(vec![zero_array(), zero_extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let zero_cotangent = zero_array();
        let mut pullback_inputs = vec![zero_cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![zero_cotangent]));
        assert!(matches!(
            program.transpose_with_respect_to(&[0]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "direct 'all_gather' transposition with dynamic extents requires linearization so that \
                    the primal geometry can be retained as residuals",
        ));
    }

    #[test]
    fn test_array_program_invariant_all_gather_linearization() {
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dimension_type = DimensionType::new(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Invariant,
                ),
                Vec::new(),
                vec![array, result_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let rendered_tangent = linearization.tangent().to_string();
        assert!(rendered_tangent.contains("dynamic_shape_slice"));
        assert!(rendered_tangent.contains("reshape"));
        let input = ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let extent = ArrayProgramValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayProgramValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let mut pullback_inputs = vec![cotangent];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]))]),
        );

        // A nondegenerate untiled invariant gather selects the current participant's size-one slice and reshapes
        // away the ranked participant axis.
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let array = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let participant_extent =
            builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()));
        let input_extent = builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::default(),
                    AllGatherOutputVariance::Invariant,
                ),
                Vec::new(),
                vec![array, participant_extent, input_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "direct invariant 'all_gather' transposition requires linearization so that the current \
                    participant can select its gathered chunk",
        ));
        let pullback = program.linearize().unwrap().pullback().unwrap().to_string();
        assert!(pullback.contains("axis_index [axis_name=\"x\"]"));
        assert!(pullback.contains("dimension_from_scalar"));
        assert!(pullback.contains("dimension_mul"));
        assert!(pullback.contains("dynamic_shape_slice"));
    }

    #[test]
    fn test_array_program_shape_changing_collective_linearization() {
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dimension_type = DimensionType::new(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                PSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![array, result_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let input = ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let extent = ArrayProgramValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayProgramValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));

        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let array = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let extent = builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(
                AllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![array, extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert!(linearization.tangent().to_string().contains("linear_call"));
        let input = ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayProgramValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
    }

    #[test]
    fn test_array_program_explicit_collective_tracing_import_and_rendering() {
        type TestContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let bounds = DimensionBounds::new(1, Some(5)).unwrap();
        let input_variable = DimensionVariable::new("items", bounds);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(input_variable.clone())]));
        let (_, program) = TestContext::trace_with_named_axes(
            |input| input.all_gather_tiled("devices", 0),
            ArrayProgramType::Array(input_type),
            vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2 })],
        )
        .unwrap();

        let [dimension_size, multiplied_extent, all_gather] = program.instructions() else {
            panic!("expected dimension observation, multiplication, and all-gather");
        };
        assert!(matches!(dimension_size.operation(), ArrayProgramOperation::DimensionSize(_)));
        assert!(matches!(multiplied_extent.operation(), ArrayProgramOperation::Dimension(DimensionOperation::Mul(_)),));
        assert!(matches!(all_gather.operation(), ArrayProgramOperation::AllGather(_)));
        assert_eq!(multiplied_extent.inputs()[0], dimension_size.outputs()[0]);
        assert_eq!(all_gather.inputs(), &[program.input_ids()[0], multiplied_extent.outputs()[0]]);
        let rendered = program.to_string();
        assert!(rendered.contains("dimension_size"));
        assert!(rendered.contains("dimension_mul"));
        assert!(rendered.contains("all_gather ["));
        assert!(rendered.contains("axis_name=\"devices\""));
        assert!(rendered.contains("options=Tiled"));

        let target_variable = DimensionVariable::new("target", bounds);
        let target_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target_variable)]));
        let instantiated = program
            .with_instantiated_type_identities(&[ArrayProgramType::Array(target_type.clone())])
            .unwrap()
            .into_owned();
        let mut destination = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let imported_input = destination.add_input(target_type.into());
        let imported_outputs = destination.splice_program(&instantiated, &[imported_input]).unwrap();
        let [imported_dimension_size, imported_multiplied_extent, imported_all_gather] = destination.instructions()
        else {
            panic!("expected the imported explicit collective graph");
        };
        assert_eq!(imported_dimension_size.inputs(), &[imported_input]);
        assert_eq!(imported_all_gather.inputs(), &[imported_input, imported_multiplied_extent.outputs()[0]]);
        assert_eq!(imported_all_gather.outputs(), imported_outputs.as_slice());
    }

    #[test]
    fn test_array_program_untiled_collective_retains_dynamic_extent_requirement() {
        type TestContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let input_variable = DimensionVariable::new("items", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(input_variable)]));
        let (_, program) = TestContext::trace_with_named_axes(
            |input| input.psum_scatter("devices", 0),
            ArrayProgramType::Array(input_type),
            vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2 })],
        )
        .unwrap();

        let [dimension_size, requirement, psum_scatter] = program.instructions() else {
            panic!("expected dimension observation, equality requirement, and sum-scatter");
        };
        assert!(matches!(dimension_size.operation(), ArrayProgramOperation::DimensionSize(_)));
        assert!(matches!(
            requirement.operation(),
            ArrayProgramOperation::Dimension(DimensionOperation::Requirement(_)),
        ));
        assert!(matches!(psum_scatter.operation(), ArrayProgramOperation::PSumScatter(_)));
        assert_eq!(requirement.inputs()[0], dimension_size.outputs()[0]);
        assert_eq!(psum_scatter.inputs(), &[program.input_ids()[0]]);
        assert_eq!(program.output_types(), &[ArrayProgramType::Array(ArrayType::scalar(DataType::F32))],);
    }

    /// Minimal composite compilation domain used to prove the retained-JIT contract over dimension inputs: it
    /// stages through the ordinary tracing path, "lowers" and "compiles" to the lifted flat program itself, counts
    /// backend compilations, and executes calls by eager interpretation of the compiled program.
    #[derive(Clone)]
    struct RetainedJitDomain {
        /// Number of backend compilations performed by this domain.
        compilations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    /// Compilation options of [`RetainedJitDomain`], which requires none.
    #[derive(Clone, Debug, Default, PartialEq)]
    struct RetainedJitOptions;

    impl RetainedJitDomain {
        fn new() -> Self {
            Self { compilations: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)) }
        }

        fn compilation_count(&self) -> usize {
            self.compilations.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl crate::contexts::Domain for RetainedJitDomain {
        type Type = ArrayProgramType;
        type Value = ArrayProgramValue<Array>;
        type Constant = crate::captures::CaptureReference<ArrayProgramType>;
        type Operation = ArrayProgramOperation<Array>;
    }

    impl CompilationDomain for RetainedJitDomain {
        type LoweredProgram = FlatCompilationProgram<Self>;
        type CompiledProgram = FlatCompilationProgram<Self>;
        type Options = RetainedJitOptions;
        type Error = ProgramError;

        fn stage<Request>(
            &self,
            request: Request,
        ) -> Result<StagedFunction<Self, Request::Input, Request::Output>, ProgramError>
        where
            Request: StageRequest<Self>,
        {
            request.trace(|_, output_types| Ok(output_types))
        }

        fn lower<Request>(
            &self,
            staged: Request,
        ) -> Result<LoweredFunction<Self, Request::Input, Request::Output>, ProgramError>
        where
            Request: LoweringRequest<Self>,
        {
            let program = staged.lifted_program()?.as_ref().clone();
            let output_types = staged.staged().output_types().to_vec();
            Ok(staged.into_lowered(program, output_types))
        }

        fn compile<Request>(
            &self,
            lowered: Request,
        ) -> Result<CompiledFunction<Self, Request::Input, Request::Output>, ProgramError>
        where
            Request: CompileRequest<Self>,
        {
            self.compilations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let program = lowered.lowered().lowered_program().clone();
            let output_types = lowered.lowered().output_types().to_vec();
            Ok(lowered.into_compiled(std::sync::Arc::new(program), output_types))
        }

        fn call<Request>(&self, request: Request) -> Result<Request::RuntimeOutput, ProgramError>
        where
            Request: CallRequest<Self>,
        {
            let executable = request.executable().clone();
            let outputs = executable.compiled_program().interpret_with(
                request.into_arguments(),
                |_, capture| {
                    Err(ProgramError::MalformedProgram(format!(
                        "retained-JIT test program retained capture {}",
                        capture.index(),
                    )))
                },
                |instruction, inputs| {
                    instruction.operation().interpret(
                        &EagerContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new(),
                        &EmptyRegionDriver,
                        inputs,
                    )
                },
            )?;
            Request::reconstruct(&executable, outputs)
        }
    }

    #[test]
    fn test_array_program_dynamic_zero_retained_jit_reuses_one_specialization() {
        let domain = RetainedJitDomain::new();
        let function: JittedFunction<RetainedJitDomain, _, (), ArrayProgramType, ArrayProgramType> =
            try_jit(&domain, |(), extent: CompilationTracer<RetainedJitDomain>| {
                let ArrayProgramType::Dimension(extent_type) = extent.r#type().into_owned() else {
                    return Err(ProgramError::InvalidArgument { message: "expected a dimension input".to_string() });
                };
                Ok(extent
                    .context()
                    .bind(
                        ZeroOperation::new(ArrayType::new(
                            DataType::F32,
                            Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
                        )),
                        Vec::new(),
                        std::slice::from_ref(&extent),
                    )?
                    .remove(0))
            });

        // Two calls with different runtime extents share one abstract input type, and therefore one retained trace,
        // lowering, and compiled specialization, while still producing outputs with different logical shapes. This is
        // the retained-JIT contract that would break if concrete extents ever became part of type or cache identity.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        assert_eq!(
            function.call((), ArrayProgramValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap())),
            Ok(ArrayProgramValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]))),
        );
        assert_eq!(
            function.call((), ArrayProgramValue::Dimension(DimensionValue::new(extent_type, 4).unwrap())),
            Ok(ArrayProgramValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0, 0.0]))),
        );
        assert_eq!(function.specialization_count(), 1);
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_misses, 1);
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.traces, 1);
        assert_eq!(statistics.lowerings, 1);
        assert_eq!(statistics.compilation_requests, 1);
        assert_eq!(domain.compilation_count(), 1);
    }

    #[test]
    fn test_array_program_dimension_values_share_one_abstract_type() {
        use std::hash::{BuildHasher, RandomState};

        // The retained-JIT dispatch key is built from `Typed::r#type` of each input, so dimension values with
        // different runtime extents must report one identical abstract type: a `DimensionType` is strictly identity
        // plus bounds, and concrete extents never participate in structural type equality, hashing, or display.
        // Otherwise every concrete extent would acquire its own compiled specialization, turning a runtime dynamic
        // dimension back into a static specialization parameter.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap()));
        let three = ArrayProgramValue::<Array>::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let four = ArrayProgramValue::<Array>::Dimension(DimensionValue::new(extent_type.clone(), 4).unwrap());
        assert_eq!(three.r#type().into_owned(), ArrayProgramType::Dimension(extent_type));
        assert_eq!(three.r#type().into_owned(), four.r#type().into_owned());
        assert_eq!(three.r#type().to_string(), four.r#type().to_string());
        let hasher = RandomState::new();
        assert_eq!(hasher.hash_one(three.r#type().as_ref()), hasher.hash_one(four.r#type().as_ref()));
    }

    #[test]
    fn test_array_program_value_projection() {
        let array = Array::vector((0..4096).map(|value| value as f32).collect());
        let payload = array.values().as_ptr();
        let stored = ArrayProgramValue::Array(array);

        let projected = <ArrayProgramValue<Array> as ValueProjection<ArrayType>>::projected(&stored).unwrap();
        assert_eq!(projected.values().as_ptr(), payload);
        assert_eq!(
            <ArrayProgramValue<Array> as ValueProjection<DimensionType>>::projected(&stored),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        let projected = <ArrayProgramValue<Array> as ValueProjection<ArrayType>>::into_projected(stored).unwrap();
        assert_eq!(projected.values().as_ptr(), payload);
    }

    #[test]
    fn test_array_program_dimension_projection() {
        let variable = DimensionVariable::new("extent", DimensionBounds::positive(Some(9)).unwrap());
        let dimension = DimensionValue::new(DimensionType::new(variable), 4).unwrap();
        let stored = ArrayProgramValue::<Array>::Dimension(dimension.clone());

        assert_eq!(<ArrayProgramValue<Array> as ValueProjection<DimensionType>>::projected(&stored), Ok(&dimension),);
        assert_eq!(
            <ArrayProgramValue<Array> as ValueProjection<ArrayType>>::projected(&stored),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(<ArrayProgramValue<Array> as ValueProjection<DimensionType>>::into_projected(stored), Ok(dimension),);
    }

    #[test]
    fn test_array_program_type_projection() {
        let array = ArrayType::new(DataType::F32, Shape::scalar());
        let stored = ArrayProgramType::from(array.clone());
        assert_eq!(<&ArrayType>::try_from(&stored), Ok(&array));
        assert_eq!(
            <&DimensionType>::try_from(&stored),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        let dimension =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(9)).unwrap()));
        let stored = ArrayProgramType::from(dimension.clone());
        assert_eq!(<&DimensionType>::try_from(&stored), Ok(&dimension));
        assert_eq!(
            <&ArrayType>::try_from(&stored),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
    }

    #[test]
    fn test_array_program_operation() {
        fn assert_projection<T: Type, O: Operation<T>, C: OperationProjection<T, Projected = O>>() {}

        assert_projection::<ArrayType, ArrayOperation<Array>, ArrayProgramOperation<Array>>();
        assert_projection::<DimensionType, DimensionOperation<DimensionValue>, ArrayProgramOperation<Array>>();

        let array_type = ArrayType::scalar(DataType::F32);
        let array_operation = ArrayProgramOperation::<Array>::from(ArrayOperation::Add(AddOperation));
        assert!(matches!(array_operation, ArrayProgramOperation::Array(ArrayOperation::Add(_))));
        assert_eq!(array_operation.name(), "add");
        assert_eq!(array_operation.to_string(), "add");
        assert_eq!(
            array_operation.infer_output_types(&[array_type.clone().into(), array_type.clone().into()], &[],),
            Ok(vec![array_type.clone().into()]),
        );

        // Member control-flow operations promote to their direct composite carriers. Scan promotion also lifts its
        // capture values while preserving every semantic and lowering attribute.
        assert!(matches!(
            ArrayProgramOperation::<Array>::from(ArrayOperation::Condition(ConditionOperation::new())),
            ArrayProgramOperation::Condition(_),
        ));
        let while_operation = WhileOperation::new().with_iteration_bound(7).unwrap();
        let promoted_while = ArrayProgramOperation::<Array>::from(ArrayOperation::While(while_operation.clone()));
        assert!(matches!(
            promoted_while,
            ArrayProgramOperation::While(operation)
                if operation.iteration_bound() == while_operation.iteration_bound()
        ));
        let capture = Array::vector(vec![3.0_f32, 4.0, 5.0, 6.0]);
        let scan_operation = ScanOperation::<Array>::new(1, 4)
            .with_reverse(true)
            .with_unroll(2)
            .unwrap()
            .with_captures(vec![capture.clone()]);
        let promoted_scan = ArrayProgramOperation::<Array>::from(ArrayOperation::Scan(scan_operation));
        let ArrayProgramOperation::Scan(promoted_scan) = promoted_scan else {
            panic!("expected a direct composite scan operation");
        };
        assert_eq!(promoted_scan.carry_count(), 1);
        assert_eq!(promoted_scan.length(), &Dimension::Static(4));
        assert!(promoted_scan.reverse());
        assert_eq!(promoted_scan.unroll(), 2);
        assert_eq!(promoted_scan.captures(), &[ArrayProgramValue::Array(capture)]);

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        let dimension_operation = ArrayProgramOperation::<Array>::from(DimensionOperation::Add(
            DimensionAddOperation::new(&left_type, &right_type).unwrap(),
        ));
        assert!(matches!(dimension_operation, ArrayProgramOperation::Dimension(DimensionOperation::Add(_)),));
        assert_eq!(dimension_operation.name(), "dimension_add");
        let result_types = dimension_operation
            .infer_output_types(&[left_type.clone().into(), right_type.clone().into()], &[])
            .unwrap();
        let [ArrayProgramType::Dimension(result_type)] = result_types.as_slice() else {
            panic!("expected one dimension result type");
        };
        assert_eq!(result_type.bounds(), DimensionBounds::new(2, Some(17)).unwrap());
        let requirement = ArrayProgramOperation::<Array>::from(DimensionOperation::Requirement(
            DimensionRequirementOperation::equal(&left_type, &right_type),
        ));
        assert_eq!(requirement.effects(), Effects::single(Effect::OrderedAssertion));

        // Every wrong-kind path uses the same checked type projection and therefore reports the canonical diagnostic.
        assert_eq!(
            array_operation.infer_output_types(&[left_type.clone().into(), right_type.clone().into()], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            dimension_operation.infer_output_types(&[array_type.clone().into(), array_type.clone().into()], &[]),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
        assert_eq!(
            ArrayProgramOperation::<Array>::zero_operation(left_type.clone().into()).unwrap_err(),
            ProgramError::Type(TypeError::invalid("cannot materialize a zero for a first-class dimension type")),
        );

        // The direct composite condition preserves the complete higher-order interface, including effects.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let interface = RegionInterface::new(
            vec![array_type.clone().into()],
            vec![array_type.clone().into()],
            Effects::single(Effect::OrderedIo),
        );
        let condition = ArrayProgramOperation::<Array>::Condition(ConditionOperation::new());
        assert!(matches!(condition, ArrayProgramOperation::Condition(_)));
        assert_eq!(
            condition.infer_output_types(
                &[predicate_type.into(), array_type.clone().into()],
                &[interface.clone(), interface],
            ),
            Ok(vec![array_type.clone().into()]),
        );
        assert_eq!(
            condition.infer_region_input_types(
                &[ArrayType::scalar(DataType::Boolean).into(), array_type.clone().into()],
                &[
                    RegionInterface::new(vec![array_type.clone().into()], vec![], Effects::PURE),
                    RegionInterface::new(vec![array_type.clone().into()], vec![], Effects::PURE),
                ],
            ),
            Ok(vec![None, None]),
        );
        assert_eq!(condition.region_slots(), ConditionOperation::<ArrayProgramValue<Array>>::new().region_slots());
        assert_eq!(
            condition.output_region_provenance(0),
            ConditionOperation::<ArrayProgramValue<Array>>::new().output_region_provenance(0),
        );

        // Identity-bearing zeros promote to the mixed constructor and retain ordinary identity renaming.
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let zero = ArrayProgramOperation::<Array>::from(ArrayOperation::Zero(ZeroOperation::new(dynamic_type)));
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source.clone(), target.clone()).unwrap();
        let ArrayProgramOperation::Zero(zero) = zero.rename_type_identities(&renaming).unwrap() else {
            panic!("expected a renamed mixed zero operation");
        };
        assert_eq!(zero.r#type().shape().dimensions(), &[Dimension::Dynamic(target)]);

        let static_zero = ArrayProgramOperation::<Array>::from(ZeroOperation::new(ArrayType::scalar(DataType::F32)));
        assert!(matches!(static_zero, ArrayProgramOperation::Array(ArrayOperation::Zero(_))));

        // Identity-free ones remain homogeneous, while identity-bearing ones use the explicit mixed constructor.
        let static_one = ArrayProgramOperation::<Array>::from(OneOperation::new(ArrayType::scalar(DataType::F32)));
        assert!(matches!(static_one, ArrayProgramOperation::Array(ArrayOperation::One(_))));
        let dynamic_one_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let dynamic_one = ArrayProgramOperation::<Array>::from(OneOperation::new(dynamic_one_type.clone()));
        assert!(matches!(dynamic_one, ArrayProgramOperation::DynamicOne(_)));
        assert_eq!(
            dynamic_one.infer_output_types(&[DimensionType::new(source.clone()).into()], &[]),
            Ok(vec![dynamic_one_type.into()]),
        );
        assert_eq!(
            dynamic_one.infer_output_types(&[], &[]),
            Err(TypeError::invalid(
                "'one' expects one dimension operand per dynamic output dimension (1) but got 0 operands",
            )),
        );
        let other = DimensionVariable::new("other", bounds);
        assert_eq!(
            dynamic_one.infer_output_types(&[DimensionType::new(other).into()], &[]),
            Err(TypeError::invalid(
                "'one' operand 0 has type dimension<other ∈ [1, 9)> but the output shape requires \
                 dimension<source ∈ [1, 9)>",
            )),
        );
        assert_eq!(
            dynamic_one.infer_output_types(
                &[DimensionType::new(source.clone()).into()],
                &[RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE)],
            ),
            Err(TypeError::invalid("'one' expects no regions but got 1")),
        );

        // Iota follows the same static-versus-dynamic routing while retaining and validating its varying axis.
        let static_iota = ArrayProgramOperation::<Array>::from(
            IotaOperation::new(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)])), 0).unwrap(),
        );
        assert!(matches!(static_iota, ArrayProgramOperation::Array(ArrayOperation::Iota(_))));
        let dynamic_iota_type =
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(source.clone()), Dimension::Static(2)]));
        let dynamic_iota =
            ArrayProgramOperation::<Array>::from(IotaOperation::new(dynamic_iota_type.clone(), 0).unwrap());
        assert!(matches!(dynamic_iota, ArrayProgramOperation::DynamicIota(_)));
        assert_eq!(
            dynamic_iota.infer_output_types(&[DimensionType::new(source.clone()).into()], &[]),
            Ok(vec![dynamic_iota_type.clone().into()]),
        );
        assert_eq!(
            IotaOperation::new(dynamic_iota_type, 2).unwrap_err(),
            TypeError::invalid("'iota' dimension 2 is out of bounds for rank 2"),
        );

        let renamed_left = DimensionVariable::new("renamed_left", bounds);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(left_type.variable().clone(), renamed_left.clone()).unwrap();
        let ArrayProgramOperation::Dimension(DimensionOperation::Add(add)) =
            dimension_operation.rename_type_identities(&renaming).unwrap()
        else {
            panic!("expected a renamed dimension addition operation");
        };
        assert_eq!(add.left_type().variable(), &renamed_left);
        assert_eq!(add.right_type(), &right_type);

        // A genuinely mixed operation is represented directly by the outer family rather than either homogeneous
        // member projection.
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let dimension_size =
            ArrayProgramOperation::<Array>::from(DimensionSizeOperation::new(&dynamic_type, 0).unwrap());
        assert!(matches!(dimension_size, ArrayProgramOperation::DimensionSize(_)));
        assert_eq!(dimension_size.name(), "dimension_size");
        assert_eq!(
            dimension_size.infer_output_types(&[dynamic_type.into()], &[]),
            Ok(vec![DimensionType::new(source).into()]),
        );

        // Canonical reshape derives its entire result shape from its ordered first-class dimension operand types.
        let reshape = ArrayProgramOperation::<Array>::from(ReshapeOperation::new());
        assert!(matches!(reshape, ArrayProgramOperation::Reshape(_)));
        let two = DimensionValue::constant(2).unwrap();
        let three = DimensionValue::constant(3).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(6)]));
        assert_eq!(
            reshape.infer_output_types(
                &[input_type.clone().into(), two.r#type().clone().into(), three.r#type().clone().into()],
                &[],
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]),).into()
            ]),
        );
        let output_extent =
            DimensionType::new(DimensionVariable::new("output", DimensionBounds::new(1, Some(7)).unwrap()));
        assert_eq!(
            reshape.infer_output_types(
                &[input_type.into(), output_extent.clone().into(), three.r#type().clone().into()],
                &[],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(output_extent.variable().clone()), Dimension::Static(3)]),
                )
                .into()
            ]),
        );
        assert_eq!(
            reshape.infer_output_types(&[two.r#type().clone().into()], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            reshape.infer_output_types(
                &[ArrayType::scalar(DataType::F32).into(), ArrayType::scalar(DataType::I64).into()],
                &[]
            ),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        let placed_input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])))
                .with_memory(Memory::Host { pinned: true });
        let permuted = ArrayProgramOperation::<Array>::from(ReshapeOperation::new().with_dimensions([1, 0]));
        assert_eq!(permuted.to_string(), "reshape [dimensions=[1, 0]]");
        assert_eq!(
            permuted.infer_output_types(
                &[placed_input_type.into(), DimensionValue::constant(6).unwrap().r#type().clone().into(),],
                &[],
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(6)]))
                    .with_memory(Memory::Host { pinned: true })
                    .into()
            ]),
        );
    }

    #[test]
    fn test_array_program_operation_interpretation() {
        let context = EagerContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        assert_eq!(
            context.bind(
                ArrayOperation::Add(AddOperation),
                Vec::new(),
                &[
                    ArrayProgramValue::Array(Array::vector(vec![1.0, 2.0])),
                    ArrayProgramValue::Array(Array::vector(vec![3.0, 4.0])),
                ],
            ),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![4.0, 6.0]))]),
        );

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        let operation = DimensionOperation::Add(DimensionAddOperation::new(&left_type, &right_type).unwrap());
        let result = context
            .bind(
                operation,
                Vec::new(),
                &[
                    ArrayProgramValue::Dimension(DimensionValue::new(left_type, 3).unwrap()),
                    ArrayProgramValue::Dimension(DimensionValue::new(right_type, 4).unwrap()),
                ],
            )
            .unwrap();
        let [ArrayProgramValue::Dimension(result)] = result.as_slice() else {
            panic!("expected one dimension result");
        };
        assert_eq!(result.extent(), 7);

        let reshape_input = ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let reshape = context
            .bind(
                ReshapeOperation::new(),
                Vec::new(),
                &[
                    reshape_input,
                    ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()),
                    ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap()),
                ],
            )
            .unwrap();
        assert_eq!(
            reshape,
            vec![ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],))],
        );

        let rows = DimensionValue::constant(2).unwrap();
        let columns = DimensionValue::constant(3).unwrap();
        let zero = context
            .bind(
                ZeroOperation::new(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![
                        Dimension::Dynamic(rows.r#type().variable().clone()),
                        Dimension::Dynamic(columns.r#type().variable().clone()),
                    ]),
                )),
                Vec::new(),
                &[ArrayProgramValue::Dimension(rows), ArrayProgramValue::Dimension(columns)],
            )
            .unwrap();
        assert_eq!(zero, vec![ArrayProgramValue::Array(Array::matrix(2, 3, vec![0.0_f32; 6]))]);

        let extent = DimensionValue::constant(3).unwrap();
        let one = context
            .bind(
                OneOperation::new(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(extent.r#type().variable().clone())]),
                )),
                Vec::new(),
                &[ArrayProgramValue::Dimension(extent)],
            )
            .unwrap();
        assert_eq!(one, vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 1.0, 1.0]))]);
        assert_eq!(
            context.bind(
                ArrayProgramOperation::DynamicOne(OneOperation::new(ArrayType::scalar(DataType::F32))),
                Vec::new(),
                &[],
            ),
            Err(TypeError::invalid(
                "'one' with static output type f32[] has no dynamic dimensions; use the homogeneous nullary \
                 constructor instead",
            )
            .into()),
        );

        let rows = DimensionValue::constant(2).unwrap();
        let dynamic_iota = context
            .bind(
                IotaOperation::new(
                    ArrayType::new(
                        DataType::I32,
                        Shape::new(vec![Dimension::Dynamic(rows.r#type().variable().clone()), Dimension::Static(3)]),
                    ),
                    0,
                )
                .unwrap(),
                Vec::new(),
                &[ArrayProgramValue::Dimension(rows)],
            )
            .unwrap();
        assert_eq!(
            dynamic_iota,
            vec![ArrayProgramValue::Array(
                Array::new(
                    ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]),),
                    vec![
                        Scalar::I32(0),
                        Scalar::I32(0),
                        Scalar::I32(0),
                        Scalar::I32(1),
                        Scalar::I32(1),
                        Scalar::I32(1),
                    ],
                )
                .unwrap(),
            )],
        );
        let extent_type =
            DimensionType::new(DimensionVariable::new("iota_extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let extent = ArrayProgramValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let extent_program_type = extent.r#type().into_owned();
        let output = ArrayProgramValue::Array(
            Array::new(
                ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)])),
                vec![Scalar::I32(0), Scalar::I32(1), Scalar::I32(2)],
            )
            .unwrap(),
        );
        check_operation_partial_evaluation!(
            backend = (ArrayProgramValue<Array>, ArrayProgramOperation<Array>),
            operation = ArrayProgramOperation::from(IotaOperation::new(
                ArrayType::new(
                    DataType::I32,
                    Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
                ),
                0,
            )
            .unwrap()),
            cases = [
                {
                    inputs = [(@known, extent.clone())],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = extent_program_type, replay = extent))],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );

        // A runtime extent outside the stored output axis's authoritative bounds is rejected before allocation,
        // even though eager binds skip inference: the operand's own variable admits the extent, so only the stored
        // axis's bounds can catch it. Identity equality is deliberately not required (inputs may be alpha-renamed).
        let bounded = DimensionVariable::new("bounded", DimensionBounds::new(1, Some(4)).unwrap());
        let error = context
            .bind(
                ZeroOperation::new(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(bounded.clone())]),
                )),
                Vec::new(),
                &[ArrayProgramValue::Dimension(DimensionValue::constant(5).unwrap())],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::BindingOutOfBounds {
                variable: "bounded".to_string(),
                value: 5,
                bounds: DimensionBounds::new(1, Some(4)).unwrap(),
            }),
        );
        let error = context
            .bind(
                OneOperation::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(bounded.clone())]))),
                Vec::new(),
                &[ArrayProgramValue::Dimension(DimensionValue::constant(5).unwrap())],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::BindingOutOfBounds {
                variable: "bounded".to_string(),
                value: 5,
                bounds: DimensionBounds::new(1, Some(4)).unwrap(),
            }),
        );
        let error = context
            .bind(
                IotaOperation::new(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(bounded)])), 0)
                    .unwrap(),
                Vec::new(),
                &[ArrayProgramValue::Dimension(DimensionValue::constant(5).unwrap())],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::BindingOutOfBounds {
                variable: "bounded".to_string(),
                value: 5,
                bounds: DimensionBounds::new(1, Some(4)).unwrap(),
            }),
        );

        let condition = ArrayProgramOperation::<Array>::Condition(ConditionOperation::new());
        assert_eq!(
            condition.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::MalformedProgram("condition interpretation requires a predicate input".to_string(),)),
        );
    }

    #[test]
    fn test_array_program_operation_tracing_has_only_explicit_dependencies() {
        type TestContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let context = TestContext::new();
        let array = context.input(ArrayType::scalar(DataType::F32).into());
        let array_atom = array.atom_id().unwrap();
        let array = <Tracer<TestContext> as ValueProjection<ArrayType>>::into_projected(array).unwrap();
        array.dispatch_domain().bind(AddOperation, Vec::new(), &[array.clone(), array]).unwrap();

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        let left = context.input(left_type.clone().into());
        let right = context.input(right_type.clone().into());
        let left_atom = left.atom_id().unwrap();
        let right_atom = right.atom_id().unwrap();
        let left = <Tracer<TestContext> as ValueProjection<DimensionType>>::into_projected(left).unwrap();
        let right = <Tracer<TestContext> as ValueProjection<DimensionType>>::into_projected(right).unwrap();
        left.dispatch_domain()
            .bind(DimensionAddOperation::new(&left_type, &right_type).unwrap(), Vec::new(), &[left, right])
            .unwrap();

        let builder = context.builder().borrow();
        let [array_instruction, dimension_instruction] = builder.instructions() else {
            panic!("expected one array instruction and one dimension instruction");
        };
        assert_eq!(array_instruction.inputs(), &[array_atom, array_atom]);
        assert!(array_instruction.regions().is_empty());
        assert!(matches!(array_instruction.operation(), ArrayProgramOperation::Array(ArrayOperation::Add(_))));
        assert_eq!(dimension_instruction.inputs(), &[left_atom, right_atom]);
        assert!(dimension_instruction.regions().is_empty());
        assert!(matches!(
            dimension_instruction.operation(),
            ArrayProgramOperation::Dimension(DimensionOperation::Add(_)),
        ));

        let reshape_context = TestContext::new();
        let reshape_input =
            reshape_context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(6)])).into());
        let first_extent = reshape_context.constant(ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()));
        let second_extent =
            reshape_context.constant(ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap()));
        let reshape_input_atom = reshape_input.atom_id().unwrap();
        let first_extent_atom = first_extent.atom_id().unwrap();
        let second_extent_atom = second_extent.atom_id().unwrap();
        let reshape_input = <Tracer<TestContext> as ValueProjection<ArrayType>>::into_projected(reshape_input)
            .unwrap()
            .into_value();
        let reshape_output = reshape_context
            .bind(ReshapeOperation::new(), Vec::new(), &[reshape_input, first_extent, second_extent])
            .unwrap()
            .remove(0);
        let reshape_builder = reshape_context.builder().borrow();
        let [reshape_instruction] = reshape_builder.instructions() else {
            panic!("expected one reshape instruction");
        };
        assert_eq!(reshape_instruction.inputs(), &[reshape_input_atom, first_extent_atom, second_extent_atom],);
        assert!(matches!(reshape_instruction.operation(), ArrayProgramOperation::Reshape(_)));
        drop(reshape_builder);
        let reshape_program = reshape_context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![reshape_output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            reshape_program.to_string(),
            indoc! {"
                lambda %0:f32[6] .
                let %1:dimension<2> = const
                    %2:dimension<3> = const
                    %3:f32[2, 3] = reshape %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        let zero_context = TestContext::new();
        let rows_value = DimensionValue::constant(2).unwrap();
        let columns_value = DimensionValue::constant(3).unwrap();
        let zero_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(rows_value.r#type().variable().clone()),
                Dimension::Dynamic(columns_value.r#type().variable().clone()),
            ]),
        );
        let rows = zero_context.constant(ArrayProgramValue::Dimension(rows_value));
        let columns = zero_context.constant(ArrayProgramValue::Dimension(columns_value));
        let rows_atom = rows.atom_id().unwrap();
        let columns_atom = columns.atom_id().unwrap();
        let zero_output =
            zero_context.bind(ZeroOperation::new(zero_type), Vec::new(), &[rows, columns]).unwrap().remove(0);
        let zero_builder = zero_context.builder().borrow();
        let [zero_instruction] = zero_builder.instructions() else {
            panic!("expected one shaped-zero instruction");
        };
        assert_eq!(zero_instruction.inputs(), &[rows_atom, columns_atom]);
        assert!(matches!(zero_instruction.operation(), ArrayProgramOperation::Zero(_)));
        drop(zero_builder);
        let zero_program = zero_context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![zero_output.atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            zero_program.to_string(),
            indoc! {"
                lambda  .
                let %0:dimension<2> = const
                    %1:dimension<3> = const
                    %2:f32[2, 3] = zero [type=f32[2, 3]] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        let one_context = TestContext::new();
        let extent_value = DimensionValue::constant(3).unwrap();
        let extent = one_context.constant(ArrayProgramValue::Dimension(extent_value.clone()));
        let extent_atom = extent.atom_id().unwrap();
        let one_output = one_context
            .bind(
                OneOperation::new(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(extent_value.r#type().variable().clone())]),
                )),
                Vec::new(),
                &[extent],
            )
            .unwrap()
            .remove(0);
        let one_builder = one_context.builder().borrow();
        let [one_instruction] = one_builder.instructions() else {
            panic!("expected one dynamic-one instruction");
        };
        assert_eq!(one_instruction.inputs(), &[extent_atom]);
        assert!(matches!(one_instruction.operation(), ArrayProgramOperation::DynamicOne(_)));
        drop(one_builder);
        let one_program = one_context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![one_output.atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            one_program.to_string(),
            indoc! {"
                lambda  .
                let %0:dimension<3> = const
                    %1:f32[3] = one [type=f32[3]] %0
                in (%1)
            "}
            .trim_end(),
        );

        let iota_context = TestContext::new();
        let extent_value = DimensionValue::constant(3).unwrap();
        let extent = iota_context.constant(ArrayProgramValue::Dimension(extent_value.clone()));
        let extent_atom = extent.atom_id().unwrap();
        let output = iota_context
            .bind(
                IotaOperation::new(
                    ArrayType::new(
                        DataType::I32,
                        Shape::new(vec![Dimension::Dynamic(extent_value.r#type().variable().clone())]),
                    ),
                    0,
                )
                .unwrap(),
                Vec::new(),
                &[extent],
            )
            .unwrap()
            .remove(0);
        let iota_builder = iota_context.builder().borrow();
        let [instruction] = iota_builder.instructions() else {
            panic!("expected one dynamic-iota instruction");
        };
        assert_eq!(instruction.inputs(), &[extent_atom]);
        assert!(matches!(instruction.operation(), ArrayProgramOperation::DynamicIota(_)));
        drop(iota_builder);
        let iota_program = iota_context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output.atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            iota_program.to_string(),
            indoc! {"
                lambda  .
                let %0:dimension<3> = const
                    %1:i32[3] = iota [type=i32[3], dimension=0] %0
                in (%1)
            "}
            .trim_end(),
        );

        let concatenate_context = TestContext::new();
        let left =
            concatenate_context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into());
        let right =
            concatenate_context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).into());
        let extent = concatenate_context.constant(ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap()));
        let left_atom = left.atom_id().unwrap();
        let right_atom = right.atom_id().unwrap();
        let extent_atom = extent.atom_id().unwrap();
        let output = concatenate_context
            .bind(ConcatenateOperation::new(0, 1).unwrap(), Vec::new(), &[left, right, extent])
            .unwrap()
            .remove(0);
        let concatenate_builder = concatenate_context.builder().borrow();
        let [instruction] = concatenate_builder.instructions() else {
            panic!("expected one concatenate instruction");
        };
        assert_eq!(instruction.inputs(), &[left_atom, right_atom, extent_atom]);
        assert!(matches!(instruction.operation(), ArrayProgramOperation::Concatenate(_)));
        drop(concatenate_builder);
        let concatenate_program = concatenate_context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            concatenate_program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[1] .
                let %2:dimension<3> = const
                    %3:f32[3] = concatenate [axis=0] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_program_reshape_partial_evaluation() {
        let input = ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let first_extent = ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap());
        let second_extent = ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap());
        let output = ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let input_type = input.r#type().into_owned();
        check_operation_partial_evaluation!(
            backend = (ArrayProgramValue<Array>, ArrayProgramOperation<Array>),
            operation = ReshapeOperation::new(),
            cases = [
                {
                    inputs = [
                        (@known, input.clone()),
                        (@known, first_extent.clone()),
                        (@known, second_extent.clone()),
                    ],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input_type, replay = input)),
                        (@known, first_extent),
                        (@known, second_extent),
                    ],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );

        let identity_input = ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let identity_input_type = identity_input.r#type().into_owned();
        check_operation_partial_evaluation!(
            backend = (ArrayProgramValue<Array>, ArrayProgramOperation<Array>),
            operation = ReshapeOperation::new(),
            cases = [{
                inputs = [
                    (@unknown(type = identity_input_type, replay = identity_input.clone())),
                    (@known, ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap())),
                    (@known, ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap())),
                ],
                outputs = [(@residual, identity_input)],
                residual_instructions = 0,
            }],
        );
    }

    #[test]
    fn test_array_program_shaped_zero_partial_evaluation() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let extent = ArrayProgramValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let output = ArrayProgramValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]));
        check_operation_partial_evaluation!(
            backend = (ArrayProgramValue<Array>, ArrayProgramOperation<Array>),
            operation = ZeroOperation::new(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
            )),
            cases = [
                {
                    inputs = [(@known, extent.clone())],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = extent_type.into(), replay = extent))],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_array_program_dynamic_one_partial_evaluation() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let extent = ArrayProgramValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let output = ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 1.0, 1.0]));
        check_operation_partial_evaluation!(
            backend = (ArrayProgramValue<Array>, ArrayProgramOperation<Array>),
            operation = OneOperation::new(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
            )),
            cases = [
                {
                    inputs = [(@known, extent.clone())],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = extent_type.into(), replay = extent))],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_array_program_homogeneous_differentiation_dispatch() {
        type TestContext = EagerContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let context = TestContext::new();
        let (primal, tangent) = context
            .jvp(
                |input| {
                    let context = input.context().clone();
                    let factor = context.lift(ArrayProgramValue::Array(Array::scalar(3.0_f64)))?;
                    Ok(context
                        .bind(
                            ArrayProgramOperation::<Array>::Array(ArrayOperation::Mul(MulOperation)),
                            Vec::new(),
                            &[input, factor],
                        )?
                        .remove(0))
                },
                ArrayProgramValue::Array(Array::scalar(2.0_f64)),
                ArrayProgramValue::Array(Array::scalar(4.0_f64)),
            )
            .unwrap();
        assert_eq!(primal, ArrayProgramValue::Array(Array::scalar(6.0_f64)));
        assert_eq!(tangent, ArrayProgramValue::Array(Array::scalar(12.0_f64)));

        // Reverse mode composes the same projected JVP with projected transposition. The constant factor is a known
        // replay input to the homogeneous multiply transpose rule.
        let (primal, pullback) = context
            .vjp(
                |input| {
                    let context = input.context().clone();
                    let factor = context.lift(ArrayProgramValue::Array(Array::scalar(3.0_f64)))?;
                    Ok(context
                        .bind(
                            ArrayProgramOperation::<Array>::Array(ArrayOperation::Mul(MulOperation)),
                            Vec::new(),
                            &[input, factor],
                        )?
                        .remove(0))
                },
                ArrayProgramValue::Array(Array::scalar(2.0_f64)),
            )
            .unwrap();
        assert_eq!(primal, ArrayProgramValue::Array(Array::scalar(6.0_f64)));
        assert_eq!(
            pullback.apply(ArrayProgramValue::Array(Array::scalar(5.0_f64))),
            Ok(ArrayProgramValue::Array(Array::scalar(15.0_f64))),
        );

        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let factor = builder.add_constant(ArrayProgramValue::Array(Array::scalar(3.0_f64)));
        let output = builder
            .add_instruction(
                ArrayProgramOperation::<Array>::Array(ArrayOperation::Mul(MulOperation)),
                Vec::new(),
                vec![input, factor],
            )
            .unwrap()[0];
        let program = builder
            .build::<ArrayProgramValue<Array>, ArrayProgramValue<Array>>(vec![output], Placeholder, Placeholder)
            .unwrap();
        assert_eq!(
            program
                .transpose_with_respect_to(&[0])
                .unwrap()
                .interpret(vec![ArrayProgramValue::Array(Array::scalar(5.0_f64))]),
            Ok(vec![ArrayProgramValue::Array(Array::scalar(15.0_f64))]),
        );
    }

    #[test]
    fn test_array_program_shaped_zero_differentiation() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let extent = builder.add_input(extent_type.clone().into());
        let output = builder
            .add_instruction(
                ZeroOperation::new(ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
                )),
                Vec::new(),
                vec![extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        let extent = ArrayProgramValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        assert_eq!(
            jvp.interpret(vec![extent]),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0])),
                ArrayProgramValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0])),
            ]),
        );
        assert_eq!(jvp.instructions().iter().filter(|instruction| instruction.operation().is_zero(0)).count(), 1);

        // Direct differentiation-context dispatch takes the same all-zero shortcut. Its tangent must reuse the shaped
        // primal SSA value rather than materializing a nullary zero that has no access to the runtime extent.
        type TestContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
        let context = TestContext::new();
        let extent = context.input(extent_type.clone().into());
        let extent_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let dynamic_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let (primal, tangent) = context
            .jvp(
                move |extent| {
                    let context = extent.context().clone();
                    Ok(context.bind(ZeroOperation::new(dynamic_type), Vec::new(), &[extent])?.remove(0))
                },
                extent,
                extent_tangent,
            )
            .unwrap();
        assert_eq!(primal.atom_id(), tangent.atom_id());
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one dynamic-zero instruction");
        };
        assert!(matches!(instruction.operation(), ArrayProgramOperation::Zero(_)));
    }

    #[test]
    fn test_array_program_dynamic_one_differentiation() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let extent = builder.add_input(extent_type.clone().into());
        let output = builder
            .add_instruction(
                OneOperation::new(ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
                )),
                Vec::new(),
                vec![extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        let extent = ArrayProgramValue::Dimension(DimensionValue::new(extent_type, 3).unwrap());
        assert_eq!(
            jvp.interpret(vec![extent]),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 1.0, 1.0])),
                ArrayProgramValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0])),
            ]),
        );
        assert_eq!(jvp.instructions().len(), 2);
        assert!(matches!(jvp.instructions()[0].operation(), ArrayProgramOperation::DynamicOne(_)));
        assert!(matches!(jvp.instructions()[1].operation(), ArrayProgramOperation::Zero(_)));
        assert_eq!(jvp.instructions()[0].inputs(), jvp.instructions()[1].inputs());

        // The direct transform context must likewise run the explicit rule rather than taking its all-structural-zero
        // shortcut: a nullary zero cannot recover the dynamic extent after the closure returns.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let dynamic_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let context = EagerContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let (primal, tangent) = context
            .jvp(
                move |extent| {
                    let context = extent.context().clone();
                    Ok(context.bind(OneOperation::new(dynamic_type), Vec::new(), &[extent])?.remove(0))
                },
                ArrayProgramValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
                ArrayProgramValue::Array(Array::new(ArrayType::scalar(DataType::Zero), vec![Scalar::Zero]).unwrap()),
            )
            .unwrap();
        assert_eq!(primal, ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 1.0, 1.0])));
        assert_eq!(tangent, ArrayProgramValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0])));
    }

    #[test]
    fn test_array_program_dynamic_iota_differentiation() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let extent = builder.add_input(extent_type.clone().into());
        let output = builder
            .add_instruction(
                IotaOperation::new(
                    ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())])),
                    0,
                )
                .unwrap(),
                Vec::new(),
                vec![extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        let extent = ArrayProgramValue::Dimension(DimensionValue::new(extent_type, 3).unwrap());
        assert_eq!(
            jvp.interpret(vec![extent]),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(vec![0.0_f64, 1.0, 2.0])),
                ArrayProgramValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0])),
            ]),
        );
        assert_eq!(jvp.instructions().len(), 2);
        assert!(matches!(jvp.instructions()[0].operation(), ArrayProgramOperation::DynamicIota(_)));
        assert!(matches!(jvp.instructions()[1].operation(), ArrayProgramOperation::Zero(_)));
        assert_eq!(jvp.instructions()[0].inputs(), jvp.instructions()[1].inputs());
    }

    #[test]
    fn test_array_program_dynamic_zero_alpha_renamed_instantiation() {
        let formal = DimensionVariable::new("formal", DimensionBounds::new(1, Some(5)).unwrap());
        let caller = DimensionVariable::new("caller", DimensionBounds::new(2, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let extent = builder.add_input(DimensionType::new(formal.clone()).into());
        let output = builder
            .add_instruction(
                ZeroOperation::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(formal)]))),
                Vec::new(),
                vec![extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // Genuine cross-program instantiation: deriving the caller renaming from the complete boundary signature
        // renames the whole program — including the dynamic zero's stored output type — and recloses its region
        // arena, so the instantiated payload stays consistent with the instantiated atom types.
        let caller_input = ArrayProgramType::Dimension(DimensionType::new(caller.clone()));
        let instantiated = program.with_instantiated_type_identities(std::slice::from_ref(&caller_input)).unwrap();
        assert_eq!(instantiated.input_types(), vec![caller_input]);
        assert_eq!(
            instantiated.output_types(),
            vec![ArrayProgramType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(caller.clone())]),
            ))],
        );
        let [instruction] = instantiated.instructions() else {
            panic!("expected one instantiated instruction");
        };
        let ArrayProgramOperation::Zero(instantiated_zero) = instruction.operation() else {
            panic!("expected the instantiated operation to remain a dynamic zero");
        };
        assert_eq!(
            instantiated_zero.r#type(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(caller.clone())])),
        );
        assert_eq!(
            instantiated.interpret(vec![ArrayProgramValue::Dimension(
                DimensionValue::new(DimensionType::new(caller.clone()), 3).unwrap()
            )]),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]))]),
        );

        // A boundary interpretation of the *uninstantiated* program with an alpha-renamed actual input type takes
        // the non-exact establishment path instead: the actual dimension member refines the declared one by bounds
        // alone, and the concrete static output then establishes its first fact for the declared input identity
        // through the closed identity signature.
        assert_eq!(
            program.interpret(vec![ArrayProgramValue::Dimension(
                DimensionValue::new(DimensionType::new(caller), 3).unwrap()
            )]),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]))]),
        );
    }

    #[test]
    fn test_array_program_dynamic_one_identity_instantiation() {
        let formal = DimensionVariable::new("formal", DimensionBounds::new(1, Some(5)).unwrap());
        let caller = DimensionVariable::new("caller", DimensionBounds::new(2, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let extent = builder.add_input(DimensionType::new(formal.clone()).into());
        let output = builder
            .add_instruction(
                OneOperation::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(formal)]))),
                Vec::new(),
                vec![extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let caller_input = ArrayProgramType::Dimension(DimensionType::new(caller.clone()));
        let instantiated = program.with_instantiated_type_identities(std::slice::from_ref(&caller_input)).unwrap();
        let [instruction] = instantiated.instructions() else {
            panic!("expected one instantiated instruction");
        };
        let ArrayProgramOperation::DynamicOne(instantiated_one) = instruction.operation() else {
            panic!("expected the instantiated operation to remain a dynamic one");
        };
        assert_eq!(
            instantiated_one.r#type(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(caller.clone())])),
        );
        assert_eq!(
            instantiated.interpret(vec![ArrayProgramValue::Dimension(
                DimensionValue::new(DimensionType::new(caller), 3).unwrap()
            )]),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 1.0, 1.0]))]),
        );
    }

    #[test]
    fn test_array_program_dynamic_iota_identity_instantiation() {
        let formal = DimensionVariable::new("formal", DimensionBounds::new(1, Some(5)).unwrap());
        let caller = DimensionVariable::new("caller", DimensionBounds::new(2, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let extent = builder.add_input(DimensionType::new(formal.clone()).into());
        let output = builder
            .add_instruction(
                IotaOperation::new(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(formal)])), 0)
                    .unwrap(),
                Vec::new(),
                vec![extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let caller_input = ArrayProgramType::Dimension(DimensionType::new(caller.clone()));
        let instantiated = program.with_instantiated_type_identities(std::slice::from_ref(&caller_input)).unwrap();
        let [instruction] = instantiated.instructions() else {
            panic!("expected one instantiated instruction");
        };
        let ArrayProgramOperation::DynamicIota(instantiated_iota) = instruction.operation() else {
            panic!("expected the instantiated operation to remain a dynamic iota");
        };
        assert_eq!(
            instantiated_iota.r#type(),
            &ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(caller.clone())])),
        );
        assert_eq!(
            instantiated.interpret(vec![ArrayProgramValue::Dimension(
                DimensionValue::new(DimensionType::new(caller), 3).unwrap()
            )]),
            Ok(vec![ArrayProgramValue::Array(
                Array::new(
                    ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)])),
                    vec![Scalar::I32(0), Scalar::I32(1), Scalar::I32(2)],
                )
                .unwrap(),
            )]),
        );
    }

    #[test]
    fn test_array_program_dynamic_constructor_transposition() {
        // Dynamic constructors depend on their extent operands only as non-differentiable shape inputs, so every
        // extent receives a structural-zero cotangent regardless of the output cotangent being live.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        for operation in [
            ArrayProgramOperation::<Array>::from(ZeroOperation::new(output_type.clone())),
            ArrayProgramOperation::<Array>::from(OneOperation::new(output_type.clone())),
            ArrayProgramOperation::<Array>::from(IotaOperation::new(output_type.clone(), 0).unwrap()),
        ] {
            let mut context = TracingContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
            let output_cotangent = context.input(output_type.clone().into());
            let cotangents = operation
                .transpose(
                    &mut context,
                    &EmptyRegionDriver,
                    &[PartialValue::Unknown(extent_type.clone().into())],
                    &[MaybeZero::Value(output_cotangent)],
                )
                .unwrap();
            let [cotangent] = cotangents.as_slice() else {
                panic!("expected one cotangent per operation input");
            };
            assert!(matches!(cotangent, MaybeZero::Zero(_)));
        }
    }

    #[test]
    fn test_array_program_reshape_differentiation() {
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(6)])).into());
        let first_extent = builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()));
        let second_extent = builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(ReshapeOperation::new(), Vec::new(), vec![input, first_extent, second_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 2);
        assert_eq!(
            jvp.interpret(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])),
                ArrayProgramValue::Array(Array::vector(vec![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0])),
            ]),
            Ok(vec![
                ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])),
                ArrayProgramValue::Array(Array::matrix(2, 3, vec![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0])),
            ]),
        );

        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.interpret(vec![ArrayProgramValue::Array(Array::matrix(
                2,
                3,
                vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            ))]),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0,]))]),
        );

        // The inverse cannot recover `n` from the `[2, 2*n]` output shape without division. The reshape JVP must
        // therefore retain the original source extent as an explicit residual while it still has the source array.
        let source = DimensionVariable::new("source", DimensionBounds::new(0, Some(9)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let source_extent = builder
            .add_instruction(DimensionSizeOperation::new(&input_type, 0).unwrap(), Vec::new(), vec![input])
            .unwrap()[0];
        let two_value = DimensionValue::constant(2).unwrap();
        let two_type = two_value.r#type().clone();
        let two = builder.add_constant(ArrayProgramValue::Dimension(two_value));
        let source_type = DimensionType::new(input_type.shape().dimensions()[0].variable().unwrap().clone());
        let doubled_extent = builder
            .add_instruction(
                DimensionOperation::Mul(DimensionMulOperation::new(&source_type, &two_type).unwrap()),
                Vec::new(),
                vec![source_extent, two],
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(ReshapeOperation::new(), Vec::new(), vec![input, two, doubled_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 2);
        for size in [0, 1, 3, 8] {
            let element_count = size * 4;
            let primal_values = (0..element_count).map(|value| value as f64).collect::<Vec<_>>();
            let tangent_values = (element_count..2 * element_count).map(|value| value as f64).collect::<Vec<_>>();
            assert_eq!(
                jvp.interpret(vec![
                    ArrayProgramValue::Array(Array::matrix(size, 4, primal_values.clone())),
                    ArrayProgramValue::Array(Array::matrix(size, 4, tangent_values.clone())),
                ]),
                Ok(vec![
                    ArrayProgramValue::Array(Array::matrix(2, 2 * size, primal_values)),
                    ArrayProgramValue::Array(Array::matrix(2, 2 * size, tangent_values)),
                ]),
            );
        }
        let linearization = program.linearize().unwrap();
        let rendered_primal = linearization.primal().to_string();
        let rendered_tangent = linearization.tangent().to_string();
        assert_eq!(
            rendered_primal,
            "
lambda %0:f64[source, 4] .
let %1:dimension<2> = const
    %2:dimension<source ∈ [0, 9)> = dimension_size [axis=0] %0
    %3:dimension<source * 2 ∈ [0, 17)> = dimension_mul %2 %1
    %4:f64[2, source * 2] = reshape %0 %1 %3
    %5:dimension<source ∈ [0, 9)> = dimension_size [axis=0] %0
in (%4, %3, %5)
            "
            .trim(),
        );
        assert_eq!(
            rendered_tangent,
            "
lambda %0:f64[source, 4], %1:dimension<source * 2 ∈ [0, 17)>, %2:dimension<source ∈ [0, 9)> .
let %3:dimension<2> = const
    %4:f64[2, source * 2] = linear_call [residual_count=3] %3 %1 %2 %0 [
        forward={
            lambda %0:dimension<2>, %1:dimension<source * 2 ∈ [0, 17)>, %2:dimension<source ∈ [0, 9)>, \
%3:f64[source, 4] .
            let %4:f64[2, source * 2] = reshape %3 %0 %1
            in (%4)
        },
        transpose={
            lambda %0:dimension<2>, %1:dimension<source * 2 ∈ [0, 17)>, \
%2:dimension<source ∈ [0, 9)>, %3:f64[2, source * 2] .
            let %4:dimension<4> = constant [value=4]
                %5:f64[source, 4] = reshape %3 %2 %4
            in (%5)
        },
    ]
in (%4)
            "
            .trim(),
        );
        assert_eq!(linearization.tangent().input_types()[0], ArrayProgramType::Array(input_type.tangent()));
        assert!(
            linearization
                .tangent()
                .input_types()
                .iter()
                .skip(1)
                .all(|r#type| matches!(r#type, ArrayProgramType::Dimension(_)))
        );
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayProgramValue::Array(Array::matrix(3, 4, (0..12).map(|value| value as f64).collect()))])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(residuals.len(), linearization.residual_count());
        assert_eq!(residuals.len(), 2);

        let tangent_values = (12..24).map(|value| value as f64).collect::<Vec<_>>();
        let mut tangent_inputs = vec![ArrayProgramValue::Array(Array::matrix(3, 4, tangent_values.clone()))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs.clone()),
            Ok(vec![ArrayProgramValue::Array(Array::matrix(2, 6, tangent_values))]),
        );

        // The executable linear boundary remains structural when imported, including both attached regions and every
        // residual edge. Nested forward differentiation likewise treats only the array input as differentiable.
        let mut imported_builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let imported_inputs = linearization
            .tangent()
            .input_types()
            .into_iter()
            .map(|r#type| imported_builder.add_input(r#type))
            .collect::<Vec<_>>();
        let imported_outputs =
            imported_builder.splice_program(linearization.tangent(), imported_inputs.as_slice()).unwrap();
        let imported = imported_builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                imported_outputs,
                vec![Placeholder; imported_inputs.len()],
                vec![Placeholder],
            )
            .unwrap();
        let [imported_call] = imported.instructions() else {
            panic!("expected one imported linear call");
        };
        assert!(matches!(imported_call.operation(), ArrayProgramOperation::LinearCall(_)));
        assert_eq!(imported_call.regions().len(), 2);
        assert_eq!(imported.interpret(tangent_inputs.clone()), linearization.tangent().interpret(tangent_inputs));

        let nested_jvp = linearization.tangent().jvp().unwrap();
        let mut nested_inputs =
            vec![ArrayProgramValue::Array(Array::matrix(3, 4, (12..24).map(|value| value as f64).collect()))];
        nested_inputs.extend(residuals.clone());
        nested_inputs.push(ArrayProgramValue::Array(Array::matrix(3, 4, (24..36).map(|value| value as f64).collect())));
        assert_eq!(nested_jvp.input_ids().len(), 2 + residuals.len());
        assert_eq!(
            nested_jvp.interpret(nested_inputs),
            Ok(vec![
                ArrayProgramValue::Array(Array::matrix(2, 6, (12..24).map(|value| value as f64).collect(),)),
                ArrayProgramValue::Array(Array::matrix(2, 6, (24..36).map(|value| value as f64).collect(),)),
            ]),
        );

        let mut pullback_inputs =
            vec![ArrayProgramValue::Array(Array::matrix(2, 6, (24..36).map(|value| value as f64).collect()))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::matrix(3, 4, (24..36).map(|value| value as f64).collect(),))]),
        );

        // A matching explicit output-extent operand is already the authoritative SSA value for the source axis, so
        // the residual path reuses it and does not read the source array again.
        let source = DimensionVariable::new("reused_source", DimensionBounds::new(1, Some(9)).unwrap());
        let source_type = DimensionType::new(source.clone());
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)])).into(),
        );
        let source_extent = builder.add_input(source_type.into());
        let four = builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output = builder
            .add_instruction(ReshapeOperation::new(), Vec::new(), vec![input, source_extent, four])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert!(!linearization.primal().to_string().contains("dimension_size"));
        assert!(!linearization.tangent().to_string().contains("dimension_size"));
    }

    #[test]
    fn test_dynamic_reshape_differentiation_deduplicates_repeated_permuted_extents() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Dynamic(extent)]),
        );
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let extent = builder
            .add_instruction(DimensionSizeOperation::new(&input_type, 0).unwrap(), Vec::new(), vec![input])
            .unwrap()[0];
        let output = builder
            .add_instruction(ReshapeOperation::new().with_dimensions([1, 0]), Vec::new(), vec![input, extent, extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        // Both output axes and both inverse axes use the same SSA extent. Partial evaluation carries it once even
        // though the linear call consumes it in multiple operand positions.
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(linearization.primal().to_string().matches("dimension_size").count(), 1);
        let input = ArrayProgramValue::Array(Array::matrix(3, 3, (0..9).map(|value| value as f64).collect()));
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let tangent = ArrayProgramValue::Array(Array::matrix(3, 3, (9..18).map(|value| value as f64).collect()));
        let mut tangent_inputs = vec![tangent];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::matrix(
                3,
                3,
                vec![9.0, 12.0, 15.0, 10.0, 13.0, 16.0, 11.0, 14.0, 17.0],
            ))]),
        );
        let mut pullback_inputs =
            vec![ArrayProgramValue::Array(Array::matrix(3, 3, (18..27).map(|value| value as f64).collect()))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::matrix(
                3,
                3,
                vec![18.0, 21.0, 24.0, 19.0, 22.0, 25.0, 20.0, 23.0, 26.0],
            ))]),
        );

        // The same compiled programs accept the lower-bound zero without inventing an extent tangent input.
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayProgramValue::Array(Array::matrix(0, 0, Vec::<f64>::new()))])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayProgramValue::Array(Array::matrix(0, 0, Vec::<f64>::new()))];
        tangent_inputs.extend(residuals);
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::matrix(0, 0, Vec::<f64>::new()))]),
        );
    }

    #[test]
    fn test_dynamic_reshape_differentiation_preserves_sharding_through_the_inverse() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Static(4)]))
                .with_sharding(sharding.clone())
                .unwrap();
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let extent = builder.add_input(DimensionType::new(extent).into());
        let four = builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output = builder
            .add_instruction(
                ReshapeOperation::new().with_output_sharding(sharding),
                Vec::new(),
                vec![input, extent, four],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.tangent().output_types(), vec![input_type.tangent().into()]);
        assert_eq!(linearization.pullback().unwrap().output_types(), vec![input_type.cotangent().into()]);
    }

    #[test]
    fn test_array_program_pad_differentiation() {
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output_extent = builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(8).unwrap()));
        let output = builder
            .add_instruction(
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![input, padding_value, output_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        assert_eq!(
            program.transpose_with_respect_to(&[0, 1]).unwrap().interpret(vec![ArrayProgramValue::Array(
                Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,])
            )]),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(vec![2.0_f64, 4.0, 6.0])),
                ArrayProgramValue::Array(Array::scalar(24.0_f64)),
            ]),
        );

        let source = DimensionVariable::new("source", DimensionBounds::new(0, Some(5)).unwrap());
        let result = DimensionVariable::new("result", DimensionBounds::new(3, Some(11)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let result_type = DimensionType::new(result);
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output_extent = builder.add_input(result_type.clone().into());
        let output = builder
            .add_instruction(
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![input, padding_value, output_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 2);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=2]"));
        assert!(linearization.pullback().unwrap().to_string().contains("dynamic_shape_slice [strides=[2]]"));

        let input = ArrayProgramValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0]));
        let padding_value = ArrayProgramValue::Array(Array::scalar(-1.0_f64));
        let output_extent = ArrayProgramValue::Dimension(DimensionValue::new(result_type.clone(), 8).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, padding_value, output_extent]).unwrap();
        assert_eq!(
            primal_outputs[0],
            ArrayProgramValue::Array(Array::vector(vec![-1.0_f64, 10.0, -1.0, 20.0, -1.0, 30.0, -1.0, -1.0])),
        );
        let residuals = primal_outputs.split_off(1);

        let mut tangent_inputs = vec![
            ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0])),
            ArrayProgramValue::Array(Array::scalar(4.0_f64)),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![4.0_f64, 1.0, 4.0, 2.0, 4.0, 3.0, 4.0, 4.0]))]),
        );

        let mut pullback_inputs =
            vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(vec![2.0_f64, 4.0, 6.0])),
                ArrayProgramValue::Array(Array::scalar(24.0_f64)),
            ]),
        );

        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayProgramValue::Array(Array::vector(Vec::<f64>::new())),
                ArrayProgramValue::Array(Array::scalar(-1.0_f64)),
                ArrayProgramValue::Dimension(DimensionValue::new(result_type, 3).unwrap()),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayProgramValue::Array(Array::vector(vec![-1.0_f64, -1.0, -1.0])),);
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(Vec::<f64>::new())),
                ArrayProgramValue::Array(Array::scalar(6.0_f64)),
            ]),
        );

        // Explicit pad geometry retains one output extent per physical axis, including statically typed axes. Keep a
        // static leading axis to verify that the pullback selects dynamic constructor operands from the right axis.
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(5)).unwrap());
        let padded_columns = DimensionVariable::new("padded_columns", DimensionBounds::new(3, Some(7)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(columns)]));
        let padded_columns_type = DimensionType::new(padded_columns);
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let rows = builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()));
        let output_extent = builder.add_input(padded_columns_type.clone().into());
        let output = builder
            .add_instruction(
                PadOperation::new(vec![0, 1], vec![0, 1], vec![0, 0]).unwrap(),
                Vec::new(),
                vec![input, padding_value, rows, output_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayProgramValue::Array(Array::matrix(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0])),
                ArrayProgramValue::Array(Array::scalar(-1.0_f64)),
                ArrayProgramValue::Dimension(DimensionValue::new(padded_columns_type, 4).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs =
            vec![ArrayProgramValue::Array(Array::matrix(2, 4, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayProgramValue::Array(Array::matrix(2, 2, vec![2.0_f64, 3.0, 6.0, 7.0])),
                ArrayProgramValue::Array(Array::scalar(18.0_f64)),
            ]),
        );
    }

    #[test]
    fn test_array_program_dynamic_slice_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let start = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder
            .add_instruction(
                ArrayProgramOperation::Array(ArrayOperation::DynamicSlice(DynamicSliceOperation::new(vec![2]))),
                Vec::new(),
                vec![input, start],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        assert_eq!(linearization.residual_count(), 2);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=2]"));
        assert!(linearization.pullback().unwrap().to_string().contains("dynamic_update_slice"));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
                ArrayProgramValue::Array(Array::scalar(1_i32)),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayProgramValue::Array(Array::vector(vec![2.0_f64, 3.0])));
        let residuals = primal_outputs.split_off(1);

        let mut tangent_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![9.0_f64, 10.0, 11.0, 12.0]))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![10.0_f64, 11.0]))]),
        );
        let mut pullback_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![5.0_f64, 7.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![0.0_f64, 5.0, 7.0, 0.0]))]),
        );

        let extent = DimensionVariable::new("strided_extent", DimensionBounds::new(4, Some(7)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayProgramOperation::Array(ArrayOperation::Slice(
                    SliceOperation::new(vec![0], vec![4]).with_strides(vec![2]).unwrap(),
                )),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]))])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 3.0])));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![9.0_f64, 10.0, 11.0, 12.0]))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![9.0_f64, 11.0]))]),
        );
        let mut pullback_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![5.0_f64, 7.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![5.0_f64, 0.0, 7.0, 0.0]))]),
        );
    }

    #[test]
    fn test_array_program_gather_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3), Dimension::Static(1)]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let indices = builder.add_input(indices_type.clone().into());
        let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]);
        let output = builder
            .add_instruction(
                ArrayProgramOperation::Array(ArrayOperation::Gather(operation)),
                Vec::new(),
                vec![input, indices],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        assert_eq!(linearization.residual_count(), 2);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=2]"));
        let indices = ArrayProgramValue::Array(Array::from_f64s(indices_type, vec![1.0, 1.0, 3.0]));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayProgramValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])), indices])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayProgramValue::Array(Array::vector(vec![20.0_f64, 20.0, 40.0])));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![2.0_f64, 2.0, 4.0]))]),
        );
        let mut pullback_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![2.0_f64, 3.0, 5.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![0.0_f64, 5.0, 0.0, 5.0]))]),
        );
    }

    #[test]
    fn test_array_program_reduce_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        for (kind, expected_primal, expected_tangent, expected_cotangent) in
            [(ReductionKind::Sum, 6.0, 15.0, 6.0), (ReductionKind::Mean, 2.0, 5.0, 2.0)]
        {
            let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
            let input = builder.add_input(input_type.clone().into());
            let output = builder
                .add_instruction(
                    ArrayProgramOperation::Array(ArrayOperation::Reduce(ReduceOperation::new(vec![0], kind))),
                    Vec::new(),
                    vec![input],
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let linearization = program.linearize().unwrap();

            assert_eq!(linearization.residual_count(), 1);
            assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
            let mut primal_outputs = linearization
                .primal()
                .interpret(vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]))])
                .unwrap();
            assert_eq!(primal_outputs[0], ArrayProgramValue::Array(Array::scalar(expected_primal)));
            let residuals = primal_outputs.split_off(1);
            let mut tangent_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![4.0_f64, 5.0, 6.0]))];
            tangent_inputs.extend(residuals.clone());
            assert_eq!(
                linearization.tangent().interpret(tangent_inputs),
                Ok(vec![ArrayProgramValue::Array(Array::scalar(expected_tangent))]),
            );
            let mut pullback_inputs = vec![ArrayProgramValue::Array(Array::scalar(6.0_f64))];
            pullback_inputs.extend(residuals);
            assert_eq!(
                linearization.pullback().unwrap().interpret(pullback_inputs),
                Ok(vec![ArrayProgramValue::Array(Array::vector(vec![
                    expected_cotangent,
                    expected_cotangent,
                    expected_cotangent,
                ]))]),
            );
        }

        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayProgramOperation::Array(ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum))),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayProgramValue::Array(Array::vector(Vec::<f64>::new()))])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayProgramValue::Array(Array::scalar(0.0_f64)));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayProgramValue::Array(Array::vector(Vec::<f64>::new()))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::scalar(0.0_f64))]),
        );
        let mut pullback_inputs = vec![ArrayProgramValue::Array(Array::scalar(3.0_f64))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(Vec::<f64>::new()))]),
        );

        for (kind, values, expected_primal, expected_tangent, expected_cotangent) in [
            (ReductionKind::Max, vec![1.0, 5.0, 5.0, 2.0], 5.0, 25.0, vec![0.0, 4.0, 4.0, 0.0]),
            (ReductionKind::Min, vec![1.0, 1.0, 5.0, 2.0], 1.0, 15.0, vec![4.0, 4.0, 0.0, 0.0]),
        ] {
            let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
            let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
            let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
            let input = builder.add_input(input_type.into());
            let output = builder
                .add_instruction(
                    ArrayProgramOperation::Array(ArrayOperation::Reduce(ReduceOperation::new(vec![0], kind))),
                    Vec::new(),
                    vec![input],
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let linearization = program.linearize().unwrap();

            assert_eq!(linearization.residual_count(), 2);
            let mut primal_outputs =
                linearization.primal().interpret(vec![ArrayProgramValue::Array(Array::vector(values))]).unwrap();
            assert_eq!(primal_outputs[0], ArrayProgramValue::Array(Array::scalar(expected_primal)));
            let residuals = primal_outputs.split_off(1);
            let mut tangent_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]))];
            tangent_inputs.extend(residuals.clone());
            assert_eq!(
                linearization.tangent().interpret(tangent_inputs),
                Ok(vec![ArrayProgramValue::Array(Array::scalar(expected_tangent))]),
            );
            let mut pullback_inputs = vec![ArrayProgramValue::Array(Array::scalar(8.0_f64))];
            pullback_inputs.extend(residuals);
            assert_eq!(
                linearization.pullback().unwrap().interpret(pullback_inputs),
                Ok(vec![ArrayProgramValue::Array(Array::vector(expected_cotangent))]),
            );
        }
    }

    #[test]
    fn test_array_program_scatter_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(7)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let indices = builder.add_input(indices_type.clone().into());
        let updates = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let output = builder
            .add_instruction(
                ArrayProgramOperation::Array(ArrayOperation::Scatter(ScatterOperation::new(
                    ScatterDimensionNumbers::new(Vec::new(), vec![0], vec![0]),
                    ScatterReductionKind::Add,
                ))),
                Vec::new(),
                vec![input, indices, updates],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        assert_eq!(linearization.residual_count(), 1);
        let indices = ArrayProgramValue::Array(Array::from_f64s(indices_type, vec![1.0, 3.0]));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
                indices,
                ArrayProgramValue::Array(Array::vector(vec![10.0_f64, 20.0])),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 12.0, 3.0, 24.0])));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![
            ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
            ArrayProgramValue::Array(Array::vector(vec![5.0_f64, 6.0])),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 7.0, 3.0, 10.0]))]),
        );
        let mut pullback_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])),
                ArrayProgramValue::Array(Array::vector(vec![20.0_f64, 40.0])),
            ]),
        );
    }

    #[test]
    fn test_array_program_slice_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(3, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayProgramOperation::Array(ArrayOperation::Slice(SliceOperation::new(vec![1], vec![3]))),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]))])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayProgramValue::Array(Array::vector(vec![2.0_f64, 3.0])));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![9.0_f64, 10.0, 11.0, 12.0]))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![10.0_f64, 11.0]))]),
        );
        let mut pullback_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![5.0_f64, 7.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![0.0_f64, 5.0, 7.0, 0.0]))]),
        );
    }

    #[test]
    fn test_array_program_update_slice_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(3, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let update = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let output = builder
            .add_instruction(
                ArrayProgramOperation::Array(ArrayOperation::UpdateSlice(UpdateSliceOperation::new(vec![1]))),
                Vec::new(),
                vec![input, update],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        assert_eq!(linearization.residual_count(), 0);
        let primal = vec![
            ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
            ArrayProgramValue::Array(Array::vector(vec![9.0_f64, 8.0])),
        ];
        assert_eq!(
            linearization.primal().interpret(primal),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 9.0, 8.0, 4.0]))]),
        );
        assert_eq!(
            linearization.tangent().interpret(vec![
                ArrayProgramValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])),
                ArrayProgramValue::Array(Array::vector(vec![5.0_f64, 6.0])),
            ]),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![10.0_f64, 5.0, 6.0, 40.0]))]),
        );
        assert_eq!(
            linearization
                .pullback()
                .unwrap()
                .interpret(vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0,]))]),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 0.0, 0.0, 4.0])),
                ArrayProgramValue::Array(Array::vector(vec![2.0_f64, 3.0])),
            ]),
        );
    }

    #[test]
    fn test_array_program_dynamic_update_slice_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let update = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let start = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder
            .add_instruction(
                ArrayProgramOperation::Array(ArrayOperation::DynamicUpdateSlice(DynamicUpdateSliceOperation)),
                Vec::new(),
                vec![input, update, start],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
                ArrayProgramValue::Array(Array::vector(vec![9.0_f64, 8.0])),
                ArrayProgramValue::Array(Array::scalar(1_i32)),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 9.0, 8.0, 4.0])));
        let residuals = primal_outputs.split_off(1);

        let mut tangent_inputs = vec![
            ArrayProgramValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])),
            ArrayProgramValue::Array(Array::vector(vec![5.0_f64, 6.0])),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![10.0_f64, 5.0, 6.0, 40.0]))]),
        );
        let mut pullback_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 0.0, 0.0, 4.0])),
                ArrayProgramValue::Array(Array::vector(vec![2.0_f64, 3.0])),
            ]),
        );
    }

    #[test]
    fn test_array_program_reshape_identity_instantiation() {
        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let source_dimension_type = DimensionType::new(source.clone());
        let source_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone()), Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let array = builder.add_input(source_array_type.clone().into());
        let extent = builder.add_input(source_dimension_type.into());
        let four = builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output =
            builder.add_instruction(ReshapeOperation::new(), Vec::new(), vec![array, extent, four]).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.output_types(),
            vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)]),)
                    .into()
            ],
        );

        let target = DimensionVariable::new("target", bounds);
        let target_dimension_type = DimensionType::new(target.clone());
        let target_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target.clone()), Dimension::Static(4)]));
        let instantiated = program
            .with_instantiated_type_identities(&[
                target_array_type.clone().into(),
                target_dimension_type.clone().into(),
            ])
            .unwrap()
            .into_owned();
        assert_eq!(
            instantiated.output_types(),
            vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(target.clone()), Dimension::Static(4)]),
                )
                .into()
            ],
        );

        let mut destination = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let array = destination.add_input(target_array_type.into());
        let extent = destination.add_input(target_dimension_type.into());
        let outputs = destination.splice_program(&instantiated, &[array, extent]).unwrap();
        let [instruction] = destination.instructions() else {
            panic!("expected the imported reshape instruction");
        };
        assert_eq!(instruction.inputs()[..2], [array, extent]);
        assert_eq!(instruction.outputs(), outputs.as_slice());
        assert_eq!(
            destination.atoms()[outputs[0].index()].r#type().as_ref(),
            &ArrayProgramType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(target), Dimension::Static(4)]),
            )),
        );
    }

    #[test]
    fn test_array_program_broadcast() {
        let input = ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0]));
        let first_extent = ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap());
        let second_extent = ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap());
        let expected_output = ArrayProgramValue::Array(Array::matrix(3, 2, vec![1.0_f64, 2.0, 1.0, 2.0, 1.0, 2.0]));
        let context = EagerContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        assert_eq!(
            context.bind(
                BroadcastOperation::new(vec![1]),
                Vec::new(),
                &[input.clone(), first_extent.clone(), second_extent.clone()],
            ),
            Ok(vec![expected_output.clone()]),
        );
        let eager_dynamic_type =
            DimensionType::new(DimensionVariable::new("eager_extent", DimensionBounds::new(1, Some(9)).unwrap()));
        assert_eq!(
            context.bind(
                BroadcastOperation::new(vec![1]),
                Vec::new(),
                &[
                    ArrayProgramValue::Array(Array::vector(vec![7.0_f64])),
                    ArrayProgramValue::Dimension(DimensionValue::new(eager_dynamic_type, 3).unwrap()),
                    ArrayProgramValue::Dimension(DimensionValue::constant(1).unwrap()),
                ],
            ),
            Ok(vec![ArrayProgramValue::Array(Array::matrix(3, 1, vec![7.0_f64, 7.0, 7.0]))]),
        );

        let input_type = input.r#type().into_owned();
        check_operation_partial_evaluation!(
            backend = (ArrayProgramValue<Array>, ArrayProgramOperation<Array>),
            operation = BroadcastOperation::new(vec![1]),
            cases = [
                {
                    inputs = [
                        (@known, input.clone()),
                        (@known, first_extent.clone()),
                        (@known, second_extent.clone()),
                    ],
                    outputs = [(@known, expected_output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input_type, replay = input.clone())),
                        (@known, first_extent.clone()),
                        (@known, second_extent.clone()),
                    ],
                    outputs = [(@residual, expected_output.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        let identity_input = ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0]));
        check_operation_partial_evaluation!(
            backend = (ArrayProgramValue<Array>, ArrayProgramOperation<Array>),
            operation = BroadcastOperation::new(vec![0]),
            cases = [{
                inputs = [
                    (@unknown(type = identity_input.r#type().into_owned(), replay = identity_input.clone())),
                    (@known, second_extent.clone()),
                ],
                outputs = [(@residual, identity_input)],
                residual_instructions = 0,
            }],
        );

        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let first_extent = builder.add_constant(first_extent);
        let second_extent = builder.add_constant(second_extent);
        let output = builder
            .add_instruction(BroadcastOperation::new(vec![1]), Vec::new(), vec![input, first_extent, second_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(program.instructions()[0].operation(), ArrayProgramOperation::Broadcast(_)));
        assert_eq!(program.instructions()[0].inputs(), &[input, first_extent, second_extent]);
        assert!(program.to_string().contains("broadcast [output_axes=[1]]"));

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0])),
                ArrayProgramValue::Array(Array::vector(vec![3.0_f64, 4.0])),
            ]),
            Ok(vec![
                expected_output,
                ArrayProgramValue::Array(Array::matrix(3, 2, vec![3.0_f64, 4.0, 3.0, 4.0, 3.0, 4.0])),
            ]),
        );
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.interpret(vec![ArrayProgramValue::Array(Array::matrix(
                3,
                2,
                vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            ))]),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![9.0_f64, 12.0]))]),
        );

        let dynamic_variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dynamic_extent = DimensionType::new(dynamic_variable.clone());
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(dynamic_variable)])).into());
        let extent = builder.add_input(dynamic_extent.clone().into());
        let output =
            builder.add_instruction(BroadcastOperation::new(vec![0]), Vec::new(), vec![input, extent]).unwrap()[0];
        let dynamic_program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(dynamic_program.jvp().is_ok());
        let linearization = dynamic_program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let input = ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]));
        let extent = ArrayProgramValue::Dimension(DimensionValue::new(dynamic_extent, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let tangent = ArrayProgramValue::Array(Array::vector(vec![4.0_f64, 5.0, 6.0]));
        let mut tangent_inputs = vec![tangent.clone()];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(linearization.tangent().interpret(tangent_inputs), Ok(vec![tangent.clone()]));
        let mut pullback_inputs = vec![tangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![tangent]));

        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let extent = builder.add_input(DimensionType::new(source.clone()).into());
        let one = builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(1).unwrap()));
        let output = builder
            .add_instruction(BroadcastOperation::new(vec![1]), Vec::new(), vec![input, extent, one])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let target = DimensionVariable::new("target", bounds);
        let instantiated = program
            .with_instantiated_type_identities(&[
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into(),
                DimensionType::new(target.clone()).into(),
            ])
            .unwrap()
            .into_owned();
        assert_eq!(
            instantiated.output_types(),
            vec![
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(target.clone()), Dimension::Static(1)]),
                )
                .into()
            ],
        );
        let mut destination = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = destination.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let extent = destination.add_input(DimensionType::new(target.clone()).into());
        let outputs = destination.splice_program(&instantiated, &[input, extent]).unwrap();
        let [instruction] = destination.instructions() else {
            panic!("expected the imported broadcast instruction");
        };
        assert_eq!(instruction.inputs()[..2], [input, extent]);
        assert_eq!(
            destination.atoms()[outputs[0].index()].r#type().as_ref(),
            &ArrayProgramType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(target), Dimension::Static(1)]),
            )),
        );
    }

    #[test]
    fn test_array_program_dynamic_shape_slice() -> Result<(), ProgramError> {
        let context = EagerContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input =
            ArrayProgramValue::Array(Array::matrix(3, 4, (0..12).map(|value| value as f64).collect::<Vec<_>>()));
        let dimension = |extent| Ok::<_, ProgramError>(ArrayProgramValue::Dimension(DimensionValue::constant(extent)?));
        let output = context.bind(
            DynamicShapeSliceOperation::new(2),
            Vec::new(),
            &[input, dimension(1)?, dimension(1)?, dimension(2)?, dimension(2)?],
        )?;
        assert_eq!(output, vec![ArrayProgramValue::Array(Array::matrix(2, 2, vec![5.0, 6.0, 9.0, 10.0]))]);

        Ok(())
    }

    #[test]
    fn test_array_program_broadcast_to_first_class_dimensions() {
        type TestContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let eager = ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]));
        assert_eq!(
            eager.broadcast_leading_sizes(&[2]),
            Ok(ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 1.0, 2.0, 3.0],))),
        );

        let context = TestContext::new();
        let value = context.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).into());
        let extent = context.constant(ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap()));
        assert_eq!(value.broadcast(&[extent], &[0]).unwrap().atom_id(), value.atom_id());
        assert!(context.builder().borrow().instructions().is_empty());

        // A shape-preserving axis permutation is still a real broadcast. Eager execution transposes the payload and
        // tracing retains the operation even though its input and output types are equal.
        let square_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let square = ArrayProgramValue::Array(Array::matrix(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]));
        let two = ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap());
        let expected = ArrayProgramValue::Array(Array::matrix(2, 2, vec![1.0_f64, 3.0, 2.0, 4.0]));
        assert_eq!(square.broadcast(&[two.clone(), two.clone()], &[1, 0]), Ok(expected.clone()));

        let context = TestContext::new();
        let value = context.input(square_type.into());
        let extent = context.constant(two);
        let output = value.broadcast(&[extent.clone(), extent], &[1, 0]).unwrap();
        {
            let builder = context.builder().borrow();
            let [instruction] = builder.instructions() else {
                panic!("expected one shape-preserving broadcast instruction");
            };
            let ArrayProgramOperation::Broadcast(operation) = instruction.operation() else {
                panic!("expected a broadcast instruction");
            };
            assert_eq!(operation.output_axes(), &[1, 0]);
        }
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(program.interpret(vec![square]), Ok(vec![expected]));

        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let context = TestContext::new();
        let scalar = context.input(ArrayType::scalar(DataType::F64).into());
        let extent = context.input(extent_type.clone().into());
        let output = scalar.broadcast_to(std::slice::from_ref(&extent)).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:dimension<extent \u{2208} [1, 5)> .
                let %2:f64[extent] = broadcast [output_axes=[]] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![
                ArrayProgramValue::Array(Array::scalar(2.5_f64)),
                ArrayProgramValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap()),
            ]),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![2.5_f64, 2.5, 2.5]))]),
        );
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.interpret(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0])),
                ArrayProgramValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
            ]),
            Ok(vec![ArrayProgramValue::Array(Array::scalar(6.0_f64))]),
        );

        let context = TestContext::new();
        let value = context.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let rows = context.constant(ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()));
        let columns = context.constant(ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = value.broadcast_to(&[rows, columns]).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.interpret(vec![ArrayProgramValue::Array(Array::vector(vec![7.0_f64]))]),
            Ok(vec![ArrayProgramValue::Array(Array::matrix(2, 3, vec![7.0_f64; 6]))]),
        );

        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let context = TestContext::new();
        let value = context.input(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(3)]))
                .into(),
        );
        let output = value.broadcast_leading_sizes(&[2]).unwrap();
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayProgramType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(batch), Dimension::Static(3)]),
            )),
        );
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let rendered = program.to_string();
        assert_eq!(rendered.matches("dimension_size").count(), 1);
        assert!(rendered.contains("broadcast [output_axes=[1, 2]]"));
    }

    #[test]
    fn test_array_program_dynamic_literal_fill_jvp_materializes_shaped_zero() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let extent = builder.add_input(extent_type.clone().into());
        let scalar = builder
            .add_instruction(
                ArrayOperation::from(crate::ConstantOperation::new(Array::scalar(2.5_f64))),
                Vec::new(),
                vec![],
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(BroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        let extent = ArrayProgramValue::Dimension(DimensionValue::new(extent_type, 3).unwrap());
        assert_eq!(
            jvp.interpret(vec![extent]),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(vec![2.5_f64, 2.5, 2.5])),
                ArrayProgramValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0])),
            ]),
        );
        let dynamic_zero = jvp
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayProgramOperation::Zero(_)))
            .unwrap();
        assert_eq!(dynamic_zero.inputs(), &[AtomId::new(0)]);
    }

    #[test]
    fn test_array_program_dynamic_projected_jvp_materializes_source_relative_widened_zero() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayProgramOperation::Array(ArrayOperation::StopGradient(StopGradientOperation::new())),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // The projected constant derivative uses its primal result as the runtime-shape exemplar, then widens the
        // element type to the tangent representation. No type-only dynamic zero is present in the fused JVP.
        let jvp = program.jvp().unwrap();
        assert!(jvp.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayProgramOperation::Array(ArrayOperation::ZeroLike(_))
        )));
        assert!(jvp.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayProgramOperation::Array(ArrayOperation::ConvertElementType(_))
        )));
        assert!(
            !jvp.instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayProgramOperation::Zero(_)))
        );

        let primal = Array::from_f64s(
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(3)])),
            vec![1.0, 2.0, 4.0],
        );
        let tangent = Array::vector(vec![1.0_f32, 1.0, 1.0]);
        let expected_primal = primal.clone();
        assert_eq!(
            jvp.interpret(vec![ArrayProgramValue::Array(primal), ArrayProgramValue::Array(tangent)]),
            Ok(vec![
                ArrayProgramValue::Array(expected_primal),
                ArrayProgramValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0])),
            ]),
        );
    }

    #[test]
    fn test_array_program_dynamic_disconnected_pullback_uses_explicit_extent_residual() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let dynamic_type = ArrayType::new(
            DataType::F8E8M0FNU,
            Shape::new(vec![
                Dimension::Dynamic(extent_type.variable().clone()),
                Dimension::Dynamic(extent_type.variable().clone()),
            ]),
        );
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        builder.add_input(dynamic_type.into());
        let scalar = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![scalar],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // The dynamic input is disconnected from the output, so linearization retains its observed extent as one
        // ordinary residual and the pullback feeds that residual to the mixed zero constructor.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(matches!(
            linearization.primal().instructions().last().unwrap().operation(),
            ArrayProgramOperation::DimensionSize(_)
        ));
        let pullback = linearization.pullback().unwrap();
        let zero = pullback.instructions().last().unwrap();
        assert!(matches!(zero.operation(), ArrayProgramOperation::Zero(_)));
        assert_eq!(zero.inputs(), &[AtomId::new(1), AtomId::new(1)]);
        assert_eq!(
            pullback.interpret(vec![
                ArrayProgramValue::Array(Array::scalar(2.0_f64)),
                ArrayProgramValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
            ]),
            Ok(vec![
                ArrayProgramValue::Array(Array::matrix(3, 3, vec![0.0_f32; 9])),
                ArrayProgramValue::Array(Array::scalar(2.0_f64)),
            ]),
        );
    }

    #[test]
    fn test_array_program_nested_dynamic_disconnected_pullback_uses_explicit_extent_residual() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(extent)]));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let context = TracingContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let dynamic = context.input(dynamic_type.clone().into());
        let scalar = context.input(scalar_type.clone().into());

        // Value-level reverse mode runs inside the outer trace. It saves only the disconnected array's extent, and
        // the reusable pullback consumes that dimension residual through the mixed zero constructor.
        let (_, pullback) = context.vjp(|inputs: Vec<_>| Ok(vec![inputs[1].clone()]), vec![dynamic, scalar]).unwrap();
        assert_eq!(pullback.residuals().len(), 1);
        assert!(matches!(pullback.residuals()[0].r#type().as_ref(), ArrayProgramType::Dimension(_)));
        let zero = pullback.program().instructions().last().unwrap();
        assert!(matches!(zero.operation(), ArrayProgramOperation::Zero(_)));
        assert_eq!(zero.inputs(), &[AtomId::new(1)]);

        let cotangent = context.input(scalar_type.into());
        let cotangents = pullback.apply(vec![cotangent]).unwrap();
        assert_eq!(cotangents[0].r#type().as_ref(), &ArrayProgramType::Array(dynamic_type.cotangent()));
        assert_eq!(cotangents[1].r#type().as_ref(), &ArrayProgramType::Array(ArrayType::scalar(DataType::F64)));
    }

    #[test]
    fn test_array_program_concatenate() {
        let operation = ConcatenateOperation::new(0, 1).unwrap();
        let left = ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let right = ArrayProgramValue::Array(Array::vector(vec![3.0_f32]));
        let extent = ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap());
        let output = ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let context = EagerContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();

        // Eager execution consumes the explicit extent without copying either array during member projection.
        assert_eq!(
            context.bind(operation.clone(), Vec::new(), &[left.clone(), right.clone(), extent.clone()],),
            Ok(vec![output.clone()]),
        );
        assert_eq!(
            context.bind(
                operation.clone(),
                Vec::new(),
                &[left.clone(), ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()),],
            ),
            Ok(vec![left.clone()]),
        );

        let observed_extent_type =
            DimensionType::new(DimensionVariable::new("observed", DimensionBounds::new(1, Some(9)).unwrap()));
        assert_eq!(
            ArrayProgramOperation::<Array>::from(operation.clone()).interpret(
                &context,
                &EmptyRegionDriver,
                &[
                    left.clone(),
                    right.clone(),
                    ArrayProgramValue::Dimension(DimensionValue::new(observed_extent_type, 4).unwrap()),
                ],
            ),
            Err(ProgramError::InvalidArgument {
                message: format!(
                    "'{}' result extent must equal the sum of input axis 0 extents; expected 3 but got 4",
                    CONCATENATE_OPERATION_NAME,
                ),
            }),
        );

        // Partial evaluation folds a fully known concatenate and otherwise retains exactly one operation with the
        // explicit extent edge, including when only that extent is unknown.
        check_operation_partial_evaluation!(
            backend = (ArrayProgramValue<Array>, ArrayProgramOperation<Array>),
            operation = operation.clone(),
            cases = [
                {
                    inputs = [(@known, left.clone()), (@known, right.clone()), (@known, extent.clone())],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = left.r#type().into_owned(), replay = left.clone())),
                        (@known, right.clone()),
                        (@known, extent.clone()),
                    ],
                    outputs = [(@residual, output.clone())],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, left.clone()),
                        (@known, right.clone()),
                        (@unknown(type = extent.r#type().into_owned(), replay = extent.clone())),
                    ],
                    outputs = [(@residual, output.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        // A stored dynamic program computes the trailing extent through ordinary dimension SSA and records every
        // dependency explicitly on the concatenate instruction.
        let left_variable = DimensionVariable::new("left", DimensionBounds::new(1, Some(5)).unwrap());
        let right_variable = DimensionVariable::new("right", DimensionBounds::new(1, Some(6)).unwrap());
        let left_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(left_variable.clone())]));
        let right_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(right_variable.clone())]));
        let left_size_operation = DimensionSizeOperation::new(&left_type, 0).unwrap();
        let right_size_operation = DimensionSizeOperation::new(&right_type, 0).unwrap();
        let left_size_type = left_size_operation.result_type().clone();
        let right_size_type = right_size_operation.result_type().clone();
        let add_operation = DimensionAddOperation::new(&left_size_type, &right_size_type).unwrap();
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let left_input = builder.add_input(left_type.into());
        let right_input = builder.add_input(right_type.into());
        let left_size = builder.add_instruction(left_size_operation, Vec::new(), vec![left_input]).unwrap()[0];
        let right_size = builder.add_instruction(right_size_operation, Vec::new(), vec![right_input]).unwrap()[0];
        let result_extent = builder
            .add_instruction(DimensionOperation::Add(add_operation), Vec::new(), vec![left_size, right_size])
            .unwrap()[0];
        let concatenated = builder
            .add_instruction(operation, Vec::new(), vec![left_input, right_input, result_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![concatenated],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let [left_size_instruction, right_size_instruction, add_instruction, concatenate_instruction] =
            program.instructions()
        else {
            panic!("expected two dimension reads, one dimension addition, and one concatenate");
        };
        assert_eq!(left_size_instruction.inputs(), &[left_input]);
        assert_eq!(right_size_instruction.inputs(), &[right_input]);
        assert_eq!(add_instruction.inputs(), &[left_size, right_size]);
        assert_eq!(concatenate_instruction.inputs(), &[left_input, right_input, result_extent]);
        assert!(matches!(concatenate_instruction.operation(), ArrayProgramOperation::Concatenate(_),));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[left], %1:f32[right] .
                let %2:dimension<left ∈ [1, 5)> = dimension_size [axis=0] %0
                    %3:dimension<right ∈ [1, 6)> = dimension_size [axis=0] %1
                    %4:dimension<left + right ∈ [2, 10)> = dimension_add %2 %3
                    %5:f32[left + right] = concatenate [axis=0] %0 %1 %4
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(program.interpret(vec![left, right]), Ok(vec![output.clone()]));

        // The same stored dynamic program composes dimension arithmetic with both forward differentiation and
        // batching. Its tangent retains the result extent and both operand extents as ordinary residual SSA edges.
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 4);
        let transformed_result_extent = jvp
            .instructions()
            .iter()
            .find_map(|instruction| {
                matches!(instruction.operation(), ArrayProgramOperation::Dimension(DimensionOperation::Add(_)),)
                    .then_some(instruction.outputs()[0])
            })
            .unwrap();
        assert_eq!(
            jvp.instructions()
                .iter()
                .filter_map(|instruction| match instruction.operation() {
                    ArrayProgramOperation::Concatenate(_) => instruction.inputs().last().copied(),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![transformed_result_extent],
        );
        let tangent_call = jvp
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayProgramOperation::LinearCall(_)))
            .unwrap();
        assert_eq!(tangent_call.inputs()[0], transformed_result_extent);
        assert_eq!(
            jvp.interpret(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0])),
                ArrayProgramValue::Array(Array::vector(vec![3.0_f32])),
                ArrayProgramValue::Array(Array::vector(vec![4.0_f32, 5.0])),
                ArrayProgramValue::Array(Array::vector(vec![6.0_f32])),
            ]),
            Ok(vec![output, ArrayProgramValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0])),]),
        );

        type Parent = EagerContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
        let batching_context = BatchingContext::<_, ArrayProgramBatching>::new(
            Parent::new(),
            ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("items".to_string())
        .with_axis_sharding(crate::ShardingDimension::Unconstrained);
        let batched_outputs = program
            .interpret_in_context(
                &batching_context,
                vec![
                    BatchingTracer::new(
                        batching_context.clone(),
                        ArrayProgramBatch::new(
                            ArrayProgramValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 4.0, 5.0])),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                    BatchingTracer::new(
                        batching_context.clone(),
                        ArrayProgramBatch::new(
                            ArrayProgramValue::Array(Array::matrix(2, 1, vec![3.0_f32, 6.0])),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                ],
            )
            .unwrap();
        assert_eq!(batched_outputs.len(), 1);
        assert_eq!(batched_outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            batched_outputs[0].batch().value(),
            &ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],)),
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 3);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=3]"));
        assert!(linearization.pullback().unwrap().to_string().contains("dynamic_shape_slice"));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0])),
                ArrayProgramValue::Array(Array::vector(vec![3.0_f32])),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![ArrayProgramValue::Array(Array::vector(vec![7.0_f32, 8.0, 9.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(vec![7.0_f32, 8.0])),
                ArrayProgramValue::Array(Array::vector(vec![9.0_f32])),
            ]),
        );
    }

    #[test]
    fn test_array_program_concatenate_differentiation() {
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let left = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let right = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let extent = builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(ConcatenateOperation::new(0, 1).unwrap(), Vec::new(), vec![left, right, extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        assert_eq!(
            program.jvp().unwrap().interpret(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0])),
                ArrayProgramValue::Array(Array::vector(vec![3.0_f64])),
                ArrayProgramValue::Array(Array::vector(vec![4.0_f64, 5.0])),
                ArrayProgramValue::Array(Array::vector(vec![6.0_f64])),
            ]),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0])),
                ArrayProgramValue::Array(Array::vector(vec![4.0_f64, 5.0, 6.0])),
            ]),
        );
        assert_eq!(
            program
                .transpose_with_respect_to(&[0, 1])
                .unwrap()
                .interpret(vec![ArrayProgramValue::Array(Array::vector(vec![7.0_f64, 8.0, 9.0]))]),
            Ok(vec![
                ArrayProgramValue::Array(Array::vector(vec![7.0_f64, 8.0])),
                ArrayProgramValue::Array(Array::vector(vec![9.0_f64])),
            ]),
        );
    }

    #[test]
    fn test_array_program_concatenate_identity_instantiation() {
        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let result = DimensionVariable::new("result", DimensionBounds::new(2, Some(12)).unwrap());
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let fixed_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let source_array = builder.add_input(source_array_type.into());
        let fixed_array = builder.add_input(fixed_array_type.clone().into());
        let result_extent = builder.add_input(DimensionType::new(result.clone()).into());
        let output = builder
            .add_instruction(
                ConcatenateOperation::new(0, 1).unwrap(),
                Vec::new(),
                vec![source_array, fixed_array, result_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let target = DimensionVariable::new("target", bounds);
        let target_result = DimensionVariable::new("target_result", DimensionBounds::new(2, Some(12)).unwrap());
        let target_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target.clone())]));
        let target_result_type = DimensionType::new(target_result.clone());
        let instantiated = program
            .with_instantiated_type_identities(&[
                target_array_type.clone().into(),
                fixed_array_type.clone().into(),
                target_result_type.clone().into(),
            ])
            .unwrap()
            .into_owned();
        assert_eq!(
            instantiated.output_types(),
            vec![ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target_result.clone())])).into()],
        );

        let mut destination = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let imported_source = destination.add_input(target_array_type.into());
        let imported_fixed = destination.add_input(fixed_array_type.into());
        let imported_extent = destination.add_input(target_result_type.into());
        let imported_outputs = destination
            .splice_program(&instantiated, &[imported_source, imported_fixed, imported_extent])
            .unwrap();
        let [instruction] = destination.instructions() else {
            panic!("expected the imported concatenate instruction");
        };
        assert_eq!(instruction.inputs(), &[imported_source, imported_fixed, imported_extent]);
        assert_eq!(instruction.outputs(), imported_outputs.as_slice());
        assert_eq!(
            destination.atoms()[imported_outputs[0].index()].r#type().as_ref(),
            &ArrayProgramType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(target_result)]),
            )),
        );
    }

    #[test]
    fn test_array_program_explicit_shape_vertical_slice() {
        let bounds = DimensionBounds::new(1, Some(5)).unwrap();
        let extent_variable = DimensionVariable::new("extent", bounds);
        let extent_type = DimensionType::new(extent_variable.clone());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_variable.clone())]));

        // Build one stored program in which ordinary dimension arithmetic supplies explicit reshape and broadcast
        // operands. The repeated extent edge deliberately feeds both shape operations.
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let extent = builder.add_input(extent_type.clone().into());
        let one_value = DimensionValue::constant(1).unwrap();
        let one_type = one_value.r#type().clone();
        let one = builder.add_constant(ArrayProgramValue::Dimension(one_value));
        let repeated_extent = builder
            .add_instruction(
                DimensionOperation::Mul(DimensionMulOperation::new(&extent_type, &one_type).unwrap()),
                Vec::new(),
                vec![extent, one],
            )
            .unwrap()[0];
        let two = builder
            .add_instruction(
                DimensionOperation::Add(DimensionAddOperation::new(&one_type, &one_type).unwrap()),
                Vec::new(),
                vec![one, one],
            )
            .unwrap()[0];
        let reshaped = builder
            .add_instruction(ReshapeOperation::new(), Vec::new(), vec![input, one, repeated_extent])
            .unwrap()[0];
        let output = builder
            .add_instruction(BroadcastOperation::new(vec![0, 1]), Vec::new(), vec![reshaped, two, repeated_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let [multiply_instruction, add_instruction, reshape_instruction, broadcast_instruction] =
            program.instructions()
        else {
            panic!("expected dimension arithmetic followed by reshape and broadcast");
        };
        assert_eq!(reshape_instruction.inputs(), &[input, one, multiply_instruction.outputs()[0]]);
        assert_eq!(
            broadcast_instruction.inputs(),
            &[reshape_instruction.outputs()[0], add_instruction.outputs()[0], multiply_instruction.outputs()[0]],
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:dimension<extent ∈ [1, 5)> .
                let %2:dimension<1> = const
                    %3:dimension<extent * 1 ∈ [1, 5)> = dimension_mul %1 %2
                    %4:dimension<2> = dimension_add %2 %2
                    %5:f64[1, extent * 1] = reshape %0 %2 %3
                    %6:f64[2, extent * 1] = broadcast [output_axes=[0, 1]] %5 %4 %3
                in (%6)
            "}
            .trim_end(),
        );

        let extent_value = ArrayProgramValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let input_value = ArrayProgramValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]));
        let expected = ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 1.0, 2.0, 3.0]));
        assert_eq!(program.interpret(vec![input_value.clone(), extent_value.clone()]), Ok(vec![expected.clone()]));

        // Known dimension arithmetic folds during partial evaluation while the two shape operations retain their
        // explicit extent inputs in the residual program.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(input_type.clone().into()),
                PartialValue::Known(extent_value.clone()),
            ])
            .unwrap();
        assert_eq!(evaluation.program().instructions().len(), 2);
        assert!(matches!(evaluation.program().instructions()[0].operation(), ArrayProgramOperation::Reshape(_),));
        assert!(matches!(evaluation.program().instructions()[1].operation(), ArrayProgramOperation::Broadcast(_),));
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new(),
                std::slice::from_ref(&input_value),
            ),
            Ok(vec![expected.clone()]),
        );

        // Forward differentiation replays both shape operations over the live array tangent while every dimension
        // value remains structural.
        let tangent = ArrayProgramValue::Array(Array::vector(vec![4.0_f64, 5.0, 6.0]));
        let expected_tangent = ArrayProgramValue::Array(Array::matrix(2, 3, vec![4.0_f64, 5.0, 6.0, 4.0, 5.0, 6.0]));
        assert_eq!(
            program.jvp().unwrap().interpret(vec![input_value.clone(), extent_value.clone(), tangent,]),
            Ok(vec![expected.clone(), expected_tangent]),
        );

        // Batching inserts one physical leading axis while the extent remains a replicated shape value.
        let batching_context = BatchingContext::<_, ArrayProgramBatching>::new(
            EagerContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new(),
            ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let batched_input = BatchingTracer::new(
            batching_context.clone(),
            ArrayProgramBatch::new(
                ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])),
                BatchAxis::new(0),
            )
            .unwrap(),
        );
        let batched_extent =
            BatchingTracer::new(batching_context.clone(), ArrayProgramBatch::replicated(extent_value.clone()));
        let batched_output = program
            .interpret_in_context(&batching_context, vec![batched_input, batched_extent])
            .unwrap()
            .remove(0);
        assert_eq!(batched_output.batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            batched_output.batch().value(),
            &ArrayProgramValue::Array(Array::from_f64s(
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(3)]),
                ),
                vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
            )),
        );

        // Instantiation and import rename the boundary identity while preserving the internal arithmetic result and
        // both consumers of its SSA value.
        let target_variable = DimensionVariable::new("target", bounds);
        let target_type = DimensionType::new(target_variable.clone());
        let target_array_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(target_variable.clone())]));
        let instantiated = program
            .with_instantiated_type_identities(&[target_array_type.clone().into(), target_type.clone().into()])
            .unwrap()
            .into_owned();
        let mut destination = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let imported_input = destination.add_input(target_array_type.into());
        let imported_extent = destination.add_input(target_type.into());
        let imported_outputs = destination.splice_program(&instantiated, &[imported_input, imported_extent]).unwrap();
        assert_eq!(destination.instructions().len(), 4);
        let [_, _, imported_reshape, imported_broadcast] = destination.instructions() else {
            panic!("expected the complete imported vertical slice");
        };
        assert_eq!(imported_reshape.inputs()[0], imported_input);
        assert_eq!(imported_broadcast.inputs()[0], imported_reshape.outputs()[0]);
        assert_eq!(imported_broadcast.outputs(), imported_outputs.as_slice());
    }

    #[test]
    fn test_symbolic_value_projection_preserves_ssa_identity() {
        type TestContext = TracingContext<ArrayProgramValue<Array>, ConstantOperation<ArrayProgramValue<Array>>>;

        let context = TestContext::new();
        let tracer = context.input(ArrayProgramType::Array(ArrayType::scalar(DataType::F32)));
        let atom = tracer.atom_id().unwrap();
        let projected = <Tracer<TestContext> as ValueProjection<ArrayType>>::projected(&tracer).unwrap();
        assert_eq!(projected.value().atom_id(), Ok(atom));
        let projected = <Tracer<TestContext> as ValueProjection<ArrayType>>::into_projected(tracer).unwrap();
        assert_eq!(projected.value().atom_id(), Ok(atom));
        assert_eq!(<Tracer<TestContext> as ValueProjection<ArrayType>>::from_projected(projected).atom_id(), Ok(atom),);

        fn assert_projection<V: ValueProjection<ArrayType>>() {}
        assert_projection::<PartialTracer<TestContext>>();
        assert_projection::<DifferentiationTracer<TestContext>>();
    }
}
