use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::backends::arrays::{Array, ArrayOperation};
use crate::backends::dimensions::{DimensionOperation, DimensionValue};
use crate::contexts::{Context, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{
    BroadcastDerivativeAlignment, DifferentiableOperation, DifferentiableType, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, ElementwiseDerivativeAlignment, TransposableOperation,
    TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::operations::compare::{Compare, CompareOperation, ComparisonDirection};
use crate::operations::constants::{Zero, ZeroOperation};
use crate::operations::dimensions::{
    DimensionFromScalar, DimensionFromScalarOperation, DimensionSize, DimensionSizeOperation, DimensionToScalar,
    DimensionToScalarOperation,
};
use crate::operations::manipulation::broadcasting::infer_explicit_broadcast_output_type;
use crate::operations::manipulation::{
    Broadcast, BroadcastOperation, Reshape, ReshapeOperation, ReshapeParameters, Transpose,
};
use crate::operations::math::AddOperation;
use crate::parameters::{Parameter, Placeholder};
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation,
};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::effects::Effects;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationProjection};
use crate::programs::regions::{EmptyRegionDriver, OutputRegionProvenance, RegionInterface, RegionSlot};
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{ProjectedValue, Value, ValueProjection};
use crate::sharding::Sharding;
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayProgramType, ArrayType, Dimension, DimensionType, DimensionVariable, Shape};

pub mod batching;

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
    /// Array-member zero constructor used for structural tangent and cotangent materialization.
    ///
    /// Its composite type parameter lets generic differentiation code request a zero, while outer inference rejects
    /// dimension-member result types so a zero constructor can never grant first-class shape authority.
    Zero(ZeroOperation<ArrayProgramType>),

    /// Homogeneous array operation.
    Array(ArrayOperation<A>),

    /// Homogeneous first-class-dimension operation.
    Dimension(DimensionOperation<DimensionValue>),

    /// Mixed comparison of two first-class dimensions that produces ordinary rank-zero Boolean array data.
    ///
    /// This variant has the precise composite member signature
    /// `(Dimension, Dimension) -> Array(Boolean scalar)`. It lives directly in [`ArrayProgramOperation`] because
    /// [`DimensionOperation`] is intentionally homogeneous: its inputs and outputs are all first-class dimensions.
    /// Storing comparison there would break that invariant because a predicate is ordinary data rather than shape
    /// authority.
    ///
    /// Homogeneous array comparison remains [`ArrayProgramOperation::Array`] wrapping
    /// [`ArrayOperation::Compare`]. This variant does not permit array-dimension or dimension-array comparisons; it
    /// reuses [`CompareOperation`] for the dimension-dimension signature whose result crosses from the dimension
    /// member kind to the array member kind.
    Compare(CompareOperation),

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
        Self::Array(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<DimensionOperation<DimensionValue>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: DimensionOperation<DimensionValue>) -> Self {
        Self::Dimension(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<CompareOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: CompareOperation) -> Self {
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

impl<A: Value<Type = ArrayType>> From<ZeroOperation<ArrayProgramType>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: ZeroOperation<ArrayProgramType>) -> Self {
        Self::Zero(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<AddOperation> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: AddOperation) -> Self {
        Self::Array(ArrayOperation::Add(operation))
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
            Self::Array(operation) => operation.name(),
            Self::Dimension(operation) => operation.name(),
            Self::Compare(operation) => Operation::<ArrayProgramType>::name(operation),
            Self::DimensionSize(operation) => operation.name(),
            Self::DimensionFromScalar(operation) => operation.name(),
            Self::DimensionToScalar(operation) => operation.name(),
            Self::Reshape(operation) => operation.name(),
            Self::Broadcast(operation) => operation.name(),
        }
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        match self {
            Self::Zero(operation) => operation.region_slots(),
            Self::Array(operation) => operation.region_slots(),
            Self::Dimension(operation) => operation.region_slots(),
            Self::Compare(operation) => Operation::<ArrayProgramType>::region_slots(operation),
            Self::DimensionSize(operation) => operation.region_slots(),
            Self::DimensionFromScalar(operation) => operation.region_slots(),
            Self::DimensionToScalar(operation) => operation.region_slots(),
            Self::Reshape(operation) => operation.region_slots(),
            Self::Broadcast(operation) => operation.region_slots(),
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<Option<Vec<ArrayProgramType>>>, TypeError> {
        match self {
            Self::Zero(operation) => operation.infer_region_input_types(input_types, region_interfaces),
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
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<ArrayProgramType>, TypeError> {
        match self {
            Self::Zero(operation) => {
                // Differential zeros always use the ordinary array member. Rejecting a dimension result here prevents
                // this generic constructor from bypassing the checked gateways that alone may create shape authority.
                <&ArrayType>::try_from(operation.r#type())?;
                operation.infer_output_types(input_types, region_interfaces)
            }
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
        }
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        match self {
            Self::Zero(operation) => operation.output_region_provenance(output_index),
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
        }
    }

    #[inline]
    fn is_zero(&self, output_index: usize) -> bool {
        match self {
            Self::Zero(operation) => operation.is_zero(output_index),
            Self::Array(operation) => operation.is_zero(output_index),
            Self::Dimension(operation) => operation.is_zero(output_index),
            Self::Compare(operation) => Operation::<ArrayProgramType>::is_zero(operation, output_index),
            Self::DimensionSize(operation) => operation.is_zero(output_index),
            Self::DimensionFromScalar(operation) => operation.is_zero(output_index),
            Self::DimensionToScalar(operation) => operation.is_zero(output_index),
            Self::Reshape(operation) => operation.is_zero(output_index),
            Self::Broadcast(operation) => operation.is_zero(output_index),
        }
    }

    #[inline]
    fn effects(&self) -> Effects {
        match self {
            Self::Zero(operation) => operation.effects(),
            Self::Array(operation) => operation.effects(),
            Self::Dimension(operation) => operation.effects(),
            Self::Compare(operation) => Operation::<ArrayProgramType>::effects(operation),
            Self::DimensionSize(operation) => operation.effects(),
            Self::DimensionFromScalar(operation) => operation.effects(),
            Self::DimensionToScalar(operation) => operation.effects(),
            Self::Reshape(operation) => operation.effects(),
            Self::Broadcast(operation) => operation.effects(),
        }
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        match self {
            Self::Zero(operation) => Ok(Self::Zero(operation.rename_type_identities(renaming)?)),
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
        }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(operation) => operation.render(formatter, indentation),
            Self::Array(operation) => operation.render(formatter, indentation),
            Self::Dimension(operation) => operation.render(formatter, indentation),
            Self::Compare(operation) => Operation::<ArrayProgramType>::render(operation, formatter, indentation),
            Self::DimensionSize(operation) => operation.render(formatter, indentation),
            Self::DimensionFromScalar(operation) => operation.render(formatter, indentation),
            Self::DimensionToScalar(operation) => operation.render(formatter, indentation),
            Self::Reshape(operation) => operation.render(formatter, indentation),
            Self::Broadcast(operation) => operation.render(formatter, indentation),
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

impl<A: Broadcast + DimensionFromScalar<DimensionValue> + DimensionSize<usize> + Reshape + Value<Type = ArrayType>>
    InterpretableOperation<EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>> for ArrayProgramOperation<A>
where
    DimensionValue: Compare<A> + DimensionToScalar<A>,
    EagerContext<A, ArrayOperation<A>>: Zero<A>,
    ArrayOperation<A>: InterpretableOperation<EagerContext<A, ArrayOperation<A>>>,
    DimensionOperation<DimensionValue>:
        InterpretableOperation<EagerContext<DimensionValue, DimensionOperation<DimensionValue>>>,
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>>>(
        &self,
        _context: &EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>,
        driver: &D,
        inputs: &[ArrayProgramValue<A>],
    ) -> Result<Vec<ArrayProgramValue<A>>, ProgramError> {
        if !self.region_slots().is_empty() || driver.region_count() != 0 {
            return Err(ProgramError::MalformedProgram(format!(
                "projected operation `{}` cannot carry regions",
                self.name(),
            )));
        }
        match self {
            Self::Zero(operation) => operation.interpret(
                &EagerContext::<ArrayProgramValue<A>, ArrayProgramOperation<A>>::new(),
                &EmptyRegionDriver,
                inputs,
            ),
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
                Ok(vec![ArrayProgramValue::Array(input.broadcast(output_type, operation.output_axes())?)])
            }
        }
    }
}

impl<A: Value<Type = ArrayType>, C: Context<Type = ArrayProgramType, Operation: From<ArrayProgramOperation<A>>>>
    PartiallyEvaluatableOperation<C> for ArrayProgramOperation<A>
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
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

impl<
    A: Value<Type = ArrayType>,
    C: Context<Type = ArrayProgramType, Constant: ValueProjection<ArrayType, Projected = A>>,
> DifferentiableOperation<C> for ArrayProgramOperation<A>
where
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<ArrayProgramOperation<A>> + OperationProjection<ArrayType, Projected = ArrayOperation<A>>,
    ArrayOperation<A>: DifferentiableOperation<ProjectedContext<C, ArrayType>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        if matches!(self, Self::Reshape(_) | Self::Broadcast(_)) {
            let Some((array, output_extents)) = inputs.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let mut primal_outputs = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?;
            let primal = primal_outputs.remove(0);
            let tangent = match array.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                MaybeZero::Value(array_tangent) => {
                    // Output extents are structural shape authority: replay their primal values unchanged while
                    // applying the same shape operation to the live array tangent.
                    let mut tangent_inputs = Vec::with_capacity(inputs.len());
                    tangent_inputs.push(array_tangent.clone());
                    tangent_inputs.extend(output_extents.iter().map(|extent| extent.primal().clone()));
                    MaybeZero::Value(context.bind(self.clone(), Vec::new(), tangent_inputs.as_slice())?.remove(0))
                }
            };
            return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
        }

        let Self::Array(operation) = self else {
            // Dimension-only and mixed shape-observation operations carry no differential dependence. Replaying the
            // primal through the composite context preserves their explicit SSA dependencies while structural zeros
            // prevent dimension authority from entering the tangent program.
            return Ok(context
                .bind(self.clone(), Vec::new(), &inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>())?
                .into_iter()
                .map(DifferentiationDual::new_with_zero_tangent)
                .collect());
        };

        let projected_inputs = inputs
            .iter()
            .map(|input| {
                let primal = <C::Value as ValueProjection<ArrayType>>::into_projected(input.primal().clone())?;
                let tangent = match input.tangent() {
                    MaybeZero::Zero(r#type) => MaybeZero::Zero(<&ArrayType>::try_from(r#type)?.clone()),
                    MaybeZero::Value(value) => {
                        MaybeZero::Value(<C::Value as ValueProjection<ArrayType>>::into_projected(value.clone())?)
                    }
                };
                DifferentiationDual::new(primal, tangent)
            })
            .collect::<Result<Vec<_>, TypeError>>()?;

        operation
            .jvp(&ProjectedContext::new(context.clone()), &EmptyRegionDriver, projected_inputs.as_slice())?
            .into_iter()
            .map(|output| {
                let (primal, tangent) = output.into_parts();
                let primal = <C::Value as ValueProjection<ArrayType>>::from_projected(primal);
                let tangent = match tangent {
                    MaybeZero::Zero(r#type) => MaybeZero::Zero(ArrayProgramType::Array(r#type)),
                    MaybeZero::Value(value) => {
                        MaybeZero::Value(<C::Value as ValueProjection<ArrayType>>::from_projected(value))
                    }
                };
                DifferentiationDual::new(primal, tangent)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}

impl<
    A: Value<Type = ArrayType>,
    V: Value<Type = ArrayProgramType>
        + ValueProjection<ArrayType, Projected = A>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
    O: Operation<ArrayProgramType>
        + OperationProjection<ArrayType, Projected = ArrayOperation<A>>
        + From<ArrayProgramOperation<A>>,
> TransposableOperation<V, O> for ArrayProgramOperation<A>
where
    ArrayOperation<A>: TransposableOperation<A, ArrayOperation<A>>,
    ProjectedValue<ArrayType, Tracer<TracingContext<V, O>>>:
        BroadcastDerivativeAlignment + ElementwiseDerivativeAlignment<ArrayType> + Transpose,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        if let Self::Broadcast(operation) = self {
            let Some((input, output_extents)) = inputs.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let [output] = outputs else {
                return Err(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() }.into());
            };
            let input_cotangent_type = <&ArrayType>::try_from(input.r#type().as_ref())?.cotangent();
            let extent_cotangents =
                || output_extents.iter().map(|extent| MaybeZero::Zero(extent.r#type().cotangent())).collect::<Vec<_>>();
            if input_cotangent_type
                .shape()
                .dimensions()
                .iter()
                .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
            {
                return Err(ProgramError::UnsupportedOperation {
                    message: "'broadcast' transpose with dynamic input extents requires Phase 6 dimension residuals"
                        .to_string(),
                }
                .into());
            }
            let MaybeZero::Value(output_cotangent) = output else {
                let mut cotangents = vec![MaybeZero::Zero(ArrayProgramType::Array(input_cotangent_type))];
                cotangents.extend(extent_cotangents());
                return Ok(cotangents);
            };
            let projected =
                <Tracer<TracingContext<V, O>> as ValueProjection<ArrayType>>::into_projected(output_cotangent.clone())?;
            let mut cotangents = vec![MaybeZero::Value(
                projected.unalign_cotangent_along(&input_cotangent_type, operation.output_axes())?.into_value(),
            )];
            cotangents.extend(extent_cotangents());
            return Ok(cotangents);
        }

        if let Self::Reshape(operation) = self {
            let Some((input, output_extents)) = inputs.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let [output] = outputs else {
                return Err(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() }.into());
            };
            let input_cotangent_type = <&ArrayType>::try_from(input.r#type().as_ref())?.cotangent();
            let extent_cotangents =
                || output_extents.iter().map(|extent| MaybeZero::Zero(extent.r#type().cotangent())).collect::<Vec<_>>();
            let MaybeZero::Value(output_cotangent) = output else {
                let mut cotangents = vec![MaybeZero::Zero(ArrayProgramType::Array(input_cotangent_type))];
                cotangents.extend(extent_cotangents());
                return Ok(cotangents);
            };
            let permuted_input_cotangent_type = match operation.dimensions() {
                Some(dimensions) => input_cotangent_type.transpose(dimensions)?,
                None => input_cotangent_type.clone(),
            };
            if permuted_input_cotangent_type
                .shape()
                .dimensions()
                .iter()
                .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
            {
                return Err(ProgramError::UnsupportedOperation {
                    message: "'reshape' transpose with dynamic input extents requires Phase 6 dimension residuals"
                        .to_string(),
                }
                .into());
            }

            let bridge_sharding = match (
                permuted_input_cotangent_type.sharding(),
                <&ArrayType>::try_from(output_cotangent.r#type().as_ref())?.sharding(),
            ) {
                (Some(sharding), _) => Some(sharding.clone()),
                (None, Some(sharding)) => {
                    Some(Sharding::replicated(sharding.mesh().clone(), permuted_input_cotangent_type.rank()))
                }
                (None, None) => None,
            };
            let mut inverse_operation = ReshapeOperation::new();
            if let Some(bridge_sharding) = bridge_sharding {
                inverse_operation = inverse_operation.with_output_sharding(bridge_sharding);
            }
            let mut inverse_inputs = Vec::with_capacity(permuted_input_cotangent_type.rank() + 1);
            inverse_inputs.push(output_cotangent.clone());
            for dimension in permuted_input_cotangent_type.shape().dimensions() {
                let extent = dimension.value().unwrap();
                let constant = <V as ValueProjection<DimensionType>>::from_projected(
                    DimensionValue::constant(extent).map_err(ProgramError::from)?,
                );
                inverse_inputs.push(context.constant(constant));
            }
            let mut cotangent = context
                .bind(ArrayProgramOperation::<A>::from(inverse_operation), Vec::new(), inverse_inputs.as_slice())?
                .remove(0);
            let mut projected =
                <Tracer<TracingContext<V, O>> as ValueProjection<ArrayType>>::into_projected(cotangent)?;
            if let Some(dimensions) = operation.dimensions() {
                projected = projected.transpose(dimensions.inverse()?)?;
            }
            projected = projected.unalign_cotangent(&input_cotangent_type)?;
            cotangent = projected.into_value();
            let mut cotangents = vec![MaybeZero::Value(cotangent)];
            cotangents.extend(extent_cotangents());
            return Ok(cotangents);
        }

        let Self::Array(operation) = self else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("operation `{}` is not transposable", self.name()),
            }
            .into());
        };

        // Record the homogeneous array rule in a short-lived array-only program. Replaying that program through a
        // projected view of the composite builder preserves the original SSA identities while keeping every primitive
        // transposition rule written against its native `ArrayType` contract.
        let mut rule_context = TracingContext::<A, ArrayOperation<A>>::new();
        let mut replay_inputs = Vec::new();
        let rule_inputs = inputs
            .iter()
            .map(|input| match input {
                PartialValue::Unknown(r#type) => Ok(PartialValue::Unknown(<&ArrayType>::try_from(r#type)?.clone())),
                PartialValue::Known(value) => {
                    replay_inputs.push(<Tracer<TracingContext<V, O>> as ValueProjection<ArrayType>>::into_projected(
                        value.clone(),
                    )?);
                    Ok(PartialValue::Known(
                        rule_context.input(<&ArrayType>::try_from(value.r#type().as_ref())?.clone()),
                    ))
                }
            })
            .collect::<Result<Vec<_>, TypeError>>()?;
        let rule_outputs = outputs
            .iter()
            .map(|output| match output {
                MaybeZero::Zero(r#type) => Ok(MaybeZero::Zero(<&ArrayType>::try_from(r#type)?.clone())),
                MaybeZero::Value(value) => {
                    replay_inputs.push(<Tracer<TracingContext<V, O>> as ValueProjection<ArrayType>>::into_projected(
                        value.clone(),
                    )?);
                    Ok(MaybeZero::Value(rule_context.input(<&ArrayType>::try_from(value.r#type().as_ref())?.clone())))
                }
            })
            .collect::<Result<Vec<_>, TypeError>>()?;
        let rule_cotangents =
            operation.transpose(&mut rule_context, &EmptyRegionDriver, &rule_inputs, &rule_outputs)?;
        let output_ids = rule_cotangents
            .iter()
            .filter_map(|cotangent| cotangent.as_value())
            .map(Tracer::atom_id)
            .collect::<Result<Vec<_>, _>>()?;
        let rule_program = rule_context.builder().borrow().clone().build::<Vec<A>, Vec<A>>(
            output_ids,
            vec![Placeholder; replay_inputs.len()],
            vec![Placeholder; rule_cotangents.iter().filter(|cotangent| !cotangent.is_zero()).count()],
        )?;
        let mut replay_outputs = rule_program
            .interpret_in_context(&ProjectedContext::new(context.clone()), replay_inputs)?
            .into_iter();

        rule_cotangents
            .into_iter()
            .map(|cotangent| match cotangent {
                MaybeZero::Zero(r#type) => Ok(MaybeZero::Zero(ArrayProgramType::Array(r#type))),
                MaybeZero::Value(_) => Ok(MaybeZero::Value(
                    replay_outputs
                        .next()
                        .ok_or_else(|| {
                            ProgramError::MalformedProgram(
                                "array transposition adapter omitted a live cotangent output".to_string(),
                            )
                        })?
                        .into_value(),
                )),
            })
            .collect()
    }
}

/// [`Value`]-level counterpart to [`ArrayProgramType`] that is used by [`Program`](crate::Program)s that may contain
/// both [`ArrayType`]-typed [`Value`]s and [`DimensionValue`]. `A` is the concrete array representation selected by the
/// owning backend. Dimensions use the common [`DimensionValue`] which is a checked host representation, so that eager
/// dimension arithmetic remains host integer work and does not allocate arrays or dispatch to device backends.
///
/// This type allows us to keep arrays and checked host-side dimensions in one storage universe while
/// [`ValueProjection`] lets homogeneous [`Operation`](crate::Operation) machinery borrow or consume only
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

impl<A: DimensionSize<usize> + Value<Type = ArrayType>> DimensionSize for ArrayProgramValue<A> {
    fn dimension_size<AxisValue: Into<crate::Axis>>(&self, axis: AxisValue) -> Result<Self, ProgramError> {
        let array = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let input_type = array.r#type();
        let operation = DimensionSizeOperation::new(input_type.as_ref(), axis)?;
        let extent = <A as DimensionSize<usize>>::dimension_size(array, operation.axis())?;
        Ok(Self::Dimension(DimensionValue::new(operation.result_type().clone(), extent)?))
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

    use crate::backends::arrays::Array;
    use crate::contexts::{Context, StagingContext};
    use crate::differentiation::DifferentiationTracer;
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::constants::{ConstantOperation, ZeroOperation};
    use crate::operations::control_flow::ConditionOperation;
    use crate::operations::dimensions::{DimensionAddOperation, DimensionRequirementOperation, DimensionSizeOperation};
    use crate::operations::manipulation::{BroadcastOperation, ReshapeOperation};
    use crate::operations::math::AddOperation;
    use crate::parameters::Placeholder;
    use crate::partial::PartialTracer;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::{Effect, Effects};
    use crate::programs::operations::OperationProjection;
    use crate::tracing::{Tracer, TracingContext};
    use crate::types::{DataType, Dimension, DimensionBounds, DimensionVariable, Layout, Memory, Shape, StridedLayout};

    use super::*;

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
        let dimension_zero =
            ArrayProgramOperation::<Array>::from(ZeroOperation::new(ArrayProgramType::Dimension(left_type.clone())));
        assert_eq!(
            dimension_zero.infer_output_types(&[], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );

        // Region projection preserves the complete higher-order interface, including effects, before delegating to
        // the homogeneous condition operation.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let interface = RegionInterface::new(
            vec![array_type.clone().into()],
            vec![array_type.clone().into()],
            Effects::single(Effect::OrderedIo),
        );
        let (_, projected_interfaces) =
            project_operation_boundary::<ArrayType>(&[], std::slice::from_ref(&interface)).unwrap();
        assert_eq!(projected_interfaces[0].effects(), Effects::single(Effect::OrderedIo));
        let condition = ArrayProgramOperation::<Array>::from(ArrayOperation::Condition(ConditionOperation::new()));
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
        assert_eq!(condition.region_slots(), ConditionOperation::<Array>::new().region_slots());
        assert_eq!(
            condition.output_region_provenance(0),
            ConditionOperation::<Array>::new().output_region_provenance(0),
        );

        // Identity-bearing payloads are renamed by their owning homogeneous family.
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let zero = ArrayProgramOperation::<Array>::from(ArrayOperation::Zero(ZeroOperation::new(dynamic_type)));
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source.clone(), target.clone()).unwrap();
        let ArrayProgramOperation::Array(ArrayOperation::Zero(zero)) = zero.rename_type_identities(&renaming).unwrap()
        else {
            panic!("expected a renamed array zero operation");
        };
        assert_eq!(zero.r#type().shape().dimensions(), &[Dimension::Dynamic(target)]);

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

        let condition = ArrayProgramOperation::<Array>::from(ArrayOperation::Condition(ConditionOperation::new()));
        assert_eq!(
            condition.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::MalformedProgram("projected operation `condition` cannot carry regions".to_string(),)),
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
                let %1:dimension<2: [2, 3)> = const
                    %2:dimension<3: [3, 4)> = const
                    %3:f32[2, 3] = reshape %0 %1 %2
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

        let source = DimensionVariable::new("source", DimensionBounds::new(1, Some(9)).unwrap());
        let source_type = DimensionType::new(source.clone());
        let mut builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = builder.add_input(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)])).into(),
        );
        let target_extent = builder.add_input(source_type.clone().into());
        let second_extent = builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output = builder
            .add_instruction(ReshapeOperation::new(), Vec::new(), vec![input, target_extent, second_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 4);
        assert_eq!(
            jvp.interpret(vec![
                ArrayProgramValue::Array(Array::matrix(3, 4, (0..12).map(|value| value as f64).collect(),)),
                ArrayProgramValue::Dimension(DimensionValue::new(source_type, 3).unwrap()),
                ArrayProgramValue::Array(Array::matrix(3, 4, (12..24).map(|value| value as f64).collect(),)),
                ArrayProgramValue::Array(Array::scalar(crate::Scalar::Zero)),
            ]),
            Ok(vec![
                ArrayProgramValue::Array(Array::matrix(3, 4, (0..12).map(|value| value as f64).collect(),)),
                ArrayProgramValue::Array(Array::matrix(3, 4, (12..24).map(|value| value as f64).collect(),)),
            ]),
        );
        assert!(matches!(
            program.transpose_with_respect_to(&[0]),
            Err(crate::differentiation::DifferentiationError::Program(
                ProgramError::UnsupportedOperation { message },
            )) if message == "'reshape' transpose with dynamic input extents requires Phase 6 dimension residuals",
        ));
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
        let extent = builder.add_input(dynamic_extent.into());
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
        assert!(matches!(
            dynamic_program.transpose_with_respect_to(&[0]),
            Err(crate::differentiation::DifferentiationError::Program(
                ProgramError::UnsupportedOperation { message },
            )) if message == "'broadcast' transpose with dynamic input extents requires Phase 6 dimension residuals",
        ));

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
