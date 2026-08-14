use std::fmt::Display;

use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatching, ArrayIrType, ArrayType, Dimension,
    DimensionType, DimensionValue, LinearResiduals, Shape, Sharding, ShardingDimension,
};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    ElementwiseDerivativeAlignment, LinearCallOperation, TransposableOperation, TranspositionDriver,
    transpose_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::manipulation::transposition::{Permutation, Transpose, TransposeOperation};
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation,
};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, TypeError,
    TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`DynamicReshapeOperation`].
pub const RESHAPE_OPERATION_NAME: &str = "reshape";

/// Mixed [`Operation`] that reshapes one array using one explicit first-class dimension operand per output axis.
///
/// Operand zero is the array. Every remaining operand describes the corresponding output-axis extent, in order.
/// Exact dimension types produce static axes while non-exact dimension types retain their variables as dynamic axes.
/// The operation therefore carries only reshape attributes; it does not duplicate its output shape or encode shape
/// arithmetic in its payload.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DynamicReshapeOperation {
    /// Optional permutation of the input dimensions applied before reshaping.
    dimensions: Option<Permutation>,

    /// Optional requested output [`Sharding`].
    output_sharding: Option<Sharding>,
}

impl DynamicReshapeOperation {
    /// Creates a reshape with no input permutation or requested output sharding.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns this operation with `dimensions` used to permute the input before reshaping.
    #[inline]
    pub fn with_dimensions<P: Into<Permutation>>(mut self, dimensions: P) -> Self {
        self.dimensions = Some(dimensions.into());
        self
    }

    /// Returns this operation with the requested output `sharding`.
    #[inline]
    pub fn with_output_sharding(mut self, sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = sharding.into();
        self
    }

    /// Returns the optional input-dimension permutation.
    #[inline]
    pub fn dimensions(&self) -> Option<&Permutation> {
        self.dimensions.as_ref()
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }

    /// Returns whether this operation leaves the input dimension order unchanged for an input of rank `rank`.
    #[inline]
    fn has_identity_dimensions(&self, rank: usize) -> bool {
        self.dimensions
            .as_ref()
            .is_none_or(|dimensions| dimensions.len() == rank && dimensions.iter().copied().eq(0..rank))
    }
}

impl Display for DynamicReshapeOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DynamicReshapeOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        RESHAPE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let Some((input_type, output_extent_types)) = input_types.split_first() else {
            return Err(TypeError::invalid(format!("'{RESHAPE_OPERATION_NAME}' expects an array followed by its output extents")));
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        let output_shape = Shape::new(ArrayIrType::extents(output_extent_types)?);
        Ok(vec![infer_explicit_reshape_output_type(input_type, output_shape, self)?.into()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        if self.dimensions.is_none() && self.output_sharding.is_none() {
            return formatter.write_str(RESHAPE_OPERATION_NAME);
        }
        OperationFormatter::new(formatter, indentation, RESHAPE_OPERATION_NAME)?.bracketed(|operation| {
            if let Some(dimensions) = &self.dimensions {
                operation.field("dimensions", format_args!("{:?}", dimensions.as_slice()))?;
            }
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl<C> InterpretableOperation<C> for DynamicReshapeOperation
where
    C: Domain<Type = ArrayIrType>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType> + Reshape>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        if driver.region_count() != 0 {
            return Err(TypeError::invalid(format!("expected 0 regions but got {}", driver.region_count())).into());
        }
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(TypeError::invalid(format!("'{RESHAPE_OPERATION_NAME}' expects an array followed by its output extents")).into());
        };
        let input = <C::Value as ValueProjection<ArrayType>>::into_projected(input.clone())?;
        let output_shape = Shape::new(
            output_extents
                .iter()
                .cloned()
                .map(<C::Value as ValueProjection<DimensionType>>::into_projected)
                .map(|result| result.map(|extent| Dimension::Static(extent.extent())))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let mut parameters = ReshapeParameters::new(output_shape);
        if let Some(dimensions) = self.dimensions() {
            parameters = parameters.with_dimensions(dimensions.clone());
        }
        if let Some(output_sharding) = self.output_sharding() {
            parameters = parameters.with_output_sharding(output_sharding.clone());
        }
        Ok(vec![<C::Value as ValueProjection<ArrayType>>::from_projected(input.reshape(parameters)?)])
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<DynamicReshapeOperation>>> PartiallyEvaluatableOperation<C>
    for DynamicReshapeOperation
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        if self.output_sharding().is_none()
            && driver.region_count() == 0
            && let Some(input) = inputs.first()
            && let Ok(input_type) = <&ArrayType>::try_from(input.r#type().as_ref())
            && input_type.static_shape().is_some()
            && self.has_identity_dimensions(input_type.rank())
            && self
                .infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?
                == vec![input.r#type().into_owned()]
        {
            // A static identity reshape cannot observe its exact dimension operands. Preserve the input directly so
            // an unknown array does not leave a redundant reshape in the residual program.
            return Ok(vec![input.clone()]);
        }
        context.fold_or_residualize(self.clone(), driver.regions().map(|region| region.to_program()).collect(), inputs)
    }
}

/// Batching rule for [`DynamicReshapeOperation`]. Explicit output extents remain replicated shape values. A mapped
/// input is canonicalized to a leading batch axis, and that axis is inserted into both the reshape geometry and the
/// output sharding before the mixed operation is replayed.
impl<C> BatchableOperation<C, ArrayIrBatching> for DynamicReshapeOperation
where
    C: Context<Type = ArrayIrType, Operation: From<DynamicReshapeOperation>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatching>, BatchingError> {
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        <&ArrayType>::try_from(&input.unbatched_type())?;
        if !input.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("dynamic {RESHAPE_OPERATION_NAME} does not support bounded ragged array operands"),
            });
        }
        for extent in output_extents {
            extent.validate_replicated_dimension()?;
        }

        if input.batch_axis().is_replicated() {
            return Ok(context
                .parent()
                .bind(self.clone(), Vec::new(), &inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>())?
                .into_iter()
                .map(ArrayIrBatch::replicated)
                .collect::<Vec<_>>()
                .into());
        }

        let moved_input = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(input.value().clone())?,
            input.batch_axis(),
        )?
        .move_axis(0)?;
        let moved_input = <C::Value as ValueProjection<ArrayType>>::from_projected(moved_input.into_value());

        let mut operation = Self::new();
        if let Some(dimensions) = self.dimensions() {
            let mut lifted_dimensions = Vec::with_capacity(dimensions.len() + 1);
            lifted_dimensions.push(0);
            lifted_dimensions.extend(dimensions.iter().map(|dimension| dimension + 1));
            operation = operation.with_dimensions(lifted_dimensions);
        }
        if let Some(output_sharding) = self.output_sharding() {
            operation = operation.with_output_sharding(lift_output_sharding_for_leading_batch_axis(
                output_sharding,
                context.axis_sharding().clone(),
            )?);
        }

        let mut lifted_inputs = Vec::with_capacity(inputs.len() + 1);
        lifted_inputs.push(moved_input);
        lifted_inputs.push(context.axis_extent().clone());
        lifted_inputs.extend(output_extents.iter().map(|extent| extent.value().clone()));
        Ok(context
            .parent()
            .bind(operation, Vec::new(), lifted_inputs.as_slice())?
            .into_iter()
            .map(|output| ArrayIrBatch::new(output, BatchAxis::from_position(0)))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

/// Semantic parameters accepted by [`Reshape`].
///
/// A [`Shape`] converts directly into `ReshapeParameters`, preserving the ordinary `value.reshape(shape)` spelling.
/// Callers that need an input permutation or an explicit output [`Sharding`] can construct these parameters and apply
/// the corresponding builder methods. Unlike [`ReshapeOperation`], this type contains no Intermediate Representation
/// (IR) behavior and can be consumed directly by eager backends and type inference.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReshapeParameters {
    /// Output shape of this reshape.
    output_shape: Shape,

    /// Optional permutation of the input dimensions applied before reshaping.
    dimensions: Option<Permutation>,

    /// Optional requested output [`Sharding`].
    output_sharding: Option<Sharding>,
}

impl ReshapeParameters {
    /// Creates reshape parameters with the provided output shape.
    #[inline]
    pub fn new(output_shape: impl Into<Shape>) -> Self {
        Self { output_shape: output_shape.into(), dimensions: None, output_sharding: None }
    }

    /// Returns this operation with `dimensions` used to permute the input before reshaping.
    #[inline]
    pub fn with_dimensions<P: Into<Permutation>>(mut self, dimensions: P) -> Self {
        self.dimensions = Some(dimensions.into());
        self
    }

    /// Returns this operation with the requested output `sharding`.
    #[inline]
    pub fn with_output_sharding(mut self, sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = sharding.into();
        self
    }

    /// Returns the output shape.
    #[inline]
    pub fn output_shape(&self) -> &Shape {
        &self.output_shape
    }

    /// Returns the optional input-dimension permutation.
    #[inline]
    pub fn dimensions(&self) -> Option<&Permutation> {
        self.dimensions.as_ref()
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }

    /// Returns whether this operation leaves the input dimension order unchanged for an input of rank `rank`.
    #[inline]
    fn has_identity_dimensions(&self, rank: usize) -> bool {
        self.dimensions
            .as_ref()
            .is_none_or(|dimensions| dimensions.len() == rank && dimensions.iter().copied().eq(0..rank))
    }
}

impl From<Shape> for ReshapeParameters {
    #[inline]
    fn from(shape: Shape) -> Self {
        Self::new(shape)
    }
}

/// [`Operation`] that reshapes its input array according to semantic [`ReshapeParameters`].
///
/// This is the member-family reshape primitive of the homogeneous array language: complete output geometry is carried
/// by the [`ArrayType`] metadata that [`ReshapeParameters`] describes, so the operation has exactly one operand and no
/// explicit extent edges. The input shape is recoverable from the staged input types and is therefore not duplicated
/// in the payload. It and [`BroadcastOperation`](crate::operations::manipulation::BroadcastOperation) form the
/// homogeneous baseline that [`ProjectedContext`](crate::contexts::ProjectedContext) serves, which is why transform
/// rules for mixed operations can delegate to them once operand geometry is resolved. Refer to the documentation of
/// [`Reshape`] for the underlying resolved-geometry contract.
///
/// Programs that need first-class dynamic extents stage [`DynamicReshapeOperation`] instead, which takes one explicit
/// first-class dimension operand per output axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReshapeOperation {
    /// Semantic parameters carried by this operation.
    parameters: ReshapeParameters,
}

impl ReshapeOperation {
    /// Creates a new [`ReshapeOperation`] from semantic reshape `parameters`.
    #[inline]
    pub fn new(parameters: impl Into<ReshapeParameters>) -> Self {
        Self { parameters: parameters.into() }
    }

    /// Returns the semantic parameters carried by this operation.
    #[inline]
    pub fn parameters(&self) -> &ReshapeParameters {
        &self.parameters
    }
}

impl Display for ReshapeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReshapeOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        RESHAPE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].reshape(self.parameters.clone()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as crate::Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self::new(ReshapeParameters {
            output_shape: self.parameters.output_shape().rename_type_identities(renaming),
            dimensions: self.parameters.dimensions().cloned(),
            output_sharding: self.parameters.output_sharding().cloned(),
        }))
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("shape", self.parameters.output_shape())?;
            if let Some(dimensions) = self.parameters.dimensions() {
                operation.field("dimensions", format_args!("{:?}", dimensions.as_slice()))?;
            }
            if let Some(output_sharding) = self.parameters.output_sharding() {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Reshape>> InterpretableOperation<C> for ReshapeOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].reshape(self.parameters.clone())?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ReshapeOperation where
    C::Operation: From<ReshapeOperation>
{
}

impl_differentiable_operation! {
    ReshapeOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<ReshapeOperation>,
        C::Value: Reshape,
    {
        |operation, _context, _driver, inputs| {
            // Forward-mode differentiation rule for `ReshapeOperation`. `reshape` is structural-linear, and so the
            // tangent is the same reshape applied to the operand tangent. The shared all-zero fast path handles a zero
            // operand tangent before this rule is consulted, so the operand tangent reaching here is always live.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().reshape(operation.parameters().clone())?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.reshape(operation.parameters().clone())?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<ReshapeOperation> + From<TransposeOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType> + Reshape + Transpose,
    {
        |operation, _context, _driver, inputs, outputs| {
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            let input_cotangent_type = inputs[0].r#type().cotangent();
            let permuted_input_cotangent_type = match operation.parameters().dimensions() {
                Some(dimensions) => input_cotangent_type.transpose(dimensions)?,
                None => input_cotangent_type.clone(),
            };
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    let bridge_sharding = match (
                        permuted_input_cotangent_type.sharding(),
                        cotangent.r#type().sharding(),
                    ) {
                        (Some(sharding), _) => Some(sharding.clone()),
                        (None, Some(sharding)) => Some(Sharding::replicated(
                            sharding.mesh().clone(),
                            permuted_input_cotangent_type.rank(),
                        )),
                        (None, None) => None,
                    };
                    let mut inverse_parameters = ReshapeParameters::new(permuted_input_cotangent_type.shape().clone());
                    if let Some(bridge_sharding) = bridge_sharding {
                        inverse_parameters = inverse_parameters.with_output_sharding(bridge_sharding);
                    }
                    let mut cotangent = cotangent.reshape(inverse_parameters)?;
                    if let Some(dimensions) = operation.parameters().dimensions() {
                        cotangent = cotangent.transpose(dimensions.inverse()?)?;
                    }
                    Ok(vec![MaybeZero::Value(
                        cotangent.unalign_cotangent(&input_cotangent_type)?,
                    )])
                }
                MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(input_cotangent_type)]),
            }
        }
    },
}

/// Forward-mode rule for mixed reshape. The explicit output extents are ordinary non-differentiated shape values.
/// Static input cotangent geometry replays the mixed reshape directly; dynamic geometry retains the exact input shape
/// so the linear transpose can reconstruct the inverse reshape from first-class dimension residuals.
impl<C> DifferentiableOperation<C> for DynamicReshapeOperation
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + From<DynamicReshapeOperation>
        + From<ConstantOperation<DimensionValue>>
        + OperationProjection<ArrayType, Projected: From<TransposeOperation>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let Some((array, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
        let tangent = match array.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
            MaybeZero::Value(array_tangent) => {
                let input_type = <&ArrayType>::try_from(array.primal().r#type().as_ref())?.clone();
                let input_cotangent_type = input_type.cotangent();
                let permuted_input_cotangent_type = match self.dimensions() {
                    Some(dimensions) => input_cotangent_type.transpose(dimensions)?,
                    None => input_cotangent_type.clone(),
                };
                if permuted_input_cotangent_type
                    .shape()
                    .dimensions()
                    .iter()
                    .all(|dimension| matches!(dimension, Dimension::Static(_)))
                {
                    let mut tangent_inputs = Vec::with_capacity(inputs.len());
                    tangent_inputs.push(array_tangent.clone());
                    tangent_inputs.extend(output_extents.iter().map(|extent| extent.primal().clone()));
                    MaybeZero::Value(context.bind(self.clone(), Vec::new(), tangent_inputs.as_slice())?.remove(0))
                } else {
                    // Record each distinct dynamic input extent while the source array is available. Repeated type
                    // identities reuse one residual SSA value in first-use order.
                    let mut residuals = LinearResiduals::new();
                    let output_extents =
                        residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                    let input_shape = residuals.retain_shape(context, array.primal())?;
                    let permuted_input_shape = match self.dimensions() {
                        Some(dimensions) => input_shape.transposed(dimensions),
                        None => input_shape,
                    };

                    // Both linear regions share one deterministic residual boundary. The forward region consumes the
                    // retained output extents; the transpose region consumes the retained exact input geometry.
                    let forward_operation = self.clone();
                    let forward_output_extents = output_extents.clone();
                    let transpose_operation = self.clone();
                    let transpose_target_type = input_cotangent_type.clone();
                    let transpose_permuted_type = permuted_input_cotangent_type.clone();
                    let tangent = LinearCallOperation::stage(
                        context,
                        residuals.into_values(),
                        vec![array_tangent.clone()],
                        move |residuals, linear_inputs| {
                            let mut reshape_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                            reshape_inputs.push(linear_inputs[0].clone());
                            reshape_inputs.extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                            linear_inputs[0].dispatch_domain().bind(
                                forward_operation,
                                Vec::new(),
                                reshape_inputs.as_slice(),
                            )
                        },
                        move |residuals, output_cotangents| {
                            let transpose_context = output_cotangents[0].dispatch_domain();
                            let bridge_sharding = match (
                                transpose_permuted_type.sharding(),
                                <&ArrayType>::try_from(output_cotangents[0].r#type().as_ref())?.sharding(),
                            ) {
                                (Some(sharding), _) => Some(sharding.clone()),
                                (None, Some(sharding)) => {
                                    Some(Sharding::replicated(sharding.mesh().clone(), transpose_permuted_type.rank()))
                                }
                                (None, None) => None,
                            };
                            let mut inverse_operation = DynamicReshapeOperation::new();
                            if let Some(bridge_sharding) = bridge_sharding {
                                inverse_operation = inverse_operation.with_output_sharding(bridge_sharding);
                            }
                            let mut inverse_inputs = Vec::with_capacity(transpose_permuted_type.rank() + 1);
                            inverse_inputs.push(output_cotangents[0].clone());
                            inverse_inputs.extend(permuted_input_shape.dimensions(&transpose_context, residuals)?);
                            let cotangent = transpose_context
                                .bind(inverse_operation, Vec::new(), inverse_inputs.as_slice())?
                                .remove(0);
                            let cotangent = if let Some(dimensions) = transpose_operation.dimensions() {
                                transpose_context
                                    .bind(
                                        <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                            TransposeOperation::new(dimensions.inverse()?),
                                        ),
                                        Vec::new(),
                                        std::slice::from_ref(&cotangent),
                                    )?
                                    .remove(0)
                            } else {
                                cotangent
                            };
                            let cotangent_type = cotangent.r#type();
                            let actual_type = <&ArrayType>::try_from(cotangent_type.as_ref())?;
                            if actual_type != &transpose_target_type {
                                return Err(TypeError::invalid(format!(
                                    "inverse reshape cotangent type {actual_type} does not match input cotangent type \
                                     {transpose_target_type}",
                                ))
                                .into());
                            }
                            Ok(vec![cotangent])
                        },
                    )?
                    .remove(0);
                    MaybeZero::Value(tangent)
                }
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Direct transposition rule for mixed reshape. Static input geometry delegates to the homogeneous array pullback,
/// while every explicit output extent receives a structural-zero cotangent. Dynamic input geometry requires
/// linearization so [`DifferentiableOperation::jvp`] can retain its exact extents as residuals.
impl<V, O> TransposableOperation<V, O> for DynamicReshapeOperation
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    O: Operation<Type = ArrayIrType> + OperationProjection<ArrayType>,
    <O as OperationProjection<ArrayType>>::Projected: From<ReshapeOperation>
        + From<TransposeOperation>
        + TransposableOperation<
            <V as ValueProjection<ArrayType>>::Projected,
            <O as OperationProjection<ArrayType>>::Projected,
        >,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let input_cotangent_type = <&ArrayType>::try_from(input.r#type().as_ref())?.cotangent();
        let permuted_input_cotangent_type = match self.dimensions() {
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
                message: format!(
                    "direct transposition of a dynamic '{RESHAPE_OPERATION_NAME}' requires linearization so its input \
                     extents are available as explicit residuals",
                ),
            }
            .into());
        }

        let output_type = match outputs {
            [MaybeZero::Zero(r#type)] => <&ArrayType>::try_from(r#type)?.clone(),
            [MaybeZero::Value(value)] => <&ArrayType>::try_from(value.r#type().as_ref())?.clone(),
            _ => return Err(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() }.into()),
        };
        let mut parameters = ReshapeParameters::new(output_type.shape().clone());
        if let Some(dimensions) = self.dimensions() {
            parameters = parameters.with_dimensions(dimensions.clone());
        }
        if let Some(output_sharding) = self.output_sharding() {
            parameters = parameters.with_output_sharding(output_sharding.clone());
        }
        let operation = <O as OperationProjection<ArrayType>>::Projected::from(ReshapeOperation::new(parameters));
        let mut cotangents = transpose_projected_operation(context, &operation, std::slice::from_ref(input), outputs)?;
        cotangents.extend(output_extents.iter().map(|extent| MaybeZero::Zero(extent.r#type().cotangent())));
        Ok(cotangents)
    }
}

impl<C: Context<Type = ArrayType>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for ReshapeOperation
where
    C::Value: Transpose,
    ReshapeOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let Some(_) = inputs[0].batch_axis_position() else {
            // Replicated input: there is no batch axis to thread through the reshape, so interpret it as given and
            // report the output replicated.
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into());
        };
        let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
        let moved_input = inputs[0].move_axis(0)?;
        let output_shape = self.parameters.output_shape();
        let mut lifted_output_dimensions = Vec::with_capacity(output_shape.rank() + 1);
        lifted_output_dimensions.push(Dimension::Static(axis_size));
        lifted_output_dimensions.extend_from_slice(output_shape.dimensions());
        let mut lifted_parameters = ReshapeParameters::new(Shape::new(lifted_output_dimensions));
        if let Some(dimensions) = self.parameters.dimensions() {
            let mut lifted_dimensions = Vec::with_capacity(dimensions.len() + 1);
            lifted_dimensions.push(0);
            lifted_dimensions.extend(dimensions.iter().map(|dimension| dimension + 1));
            lifted_parameters = lifted_parameters.with_dimensions(lifted_dimensions);
        }
        if let Some(output_sharding) = self.parameters.output_sharding() {
            lifted_parameters = lifted_parameters.with_output_sharding(lift_output_sharding_for_leading_batch_axis(
                output_sharding,
                ArrayBatch::sharding_for_inputs(inputs)?,
            )?);
        }
        Ok(ReshapeOperation::new(lifted_parameters)
            .interpret_with_batch_axes(context, &[moved_input], &[BatchAxis::from_position(0)])?
            .into())
    }
}

/// Inserts batching's physical leading dimension into a logical per-item output sharding.
pub(crate) fn lift_output_sharding_for_leading_batch_axis(
    output_sharding: &Sharding,
    batch_dimension: ShardingDimension,
) -> Result<Sharding, BatchingError> {
    let mut dimensions = output_sharding.dimensions().to_vec();
    dimensions.insert(0, batch_dimension.clone());
    let mut varying_manual_axes = output_sharding.varying_manual_axes().clone();
    if let ShardingDimension::Sharded(axis_names) = batch_dimension {
        for axis_name in axis_names {
            varying_manual_axes.remove(&axis_name);
        }
    }
    Sharding::new(output_sharding.mesh().clone(), dimensions)
        .and_then(|sharding| sharding.with_unreduced_axes(output_sharding.unreduced_axes().clone()))
        .and_then(|sharding| sharding.with_reduced_axes(output_sharding.reduced_axes().clone()))
        .and_then(|sharding| sharding.with_varying_manual_axes(varying_manual_axes))
        .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })
}

/// Represents the ability to reshape an array without changing its element count or row-major element order.
///
/// `t.reshape(target_shape)` reinterprets `t`'s payload under the specified target [`Shape`]. The input and target
/// shapes must have equal element counts, which the type system must be able to establish from the two shapes alone.
/// Shape-polymorphic programs whose output extents are not recoverable from anonymous dynamic [`Dimension`] values
/// stage [`DynamicReshapeOperation`] instead, which takes one explicit first-class dimension operand per output axis
/// and so expresses runtime shape arithmetic as ordinary graph values. When the input carries a [`Sharding`], singleton
/// dimensions are ignored and contiguous split/merge groups redistribute compatible mesh axes over their output
/// factors. Ambiguous dynamic, zero-sized, unconstrained, or non-contiguous placement changes require an explicit
/// output sharding. A non-identity reshape preserves the input memory space and clears explicit physical layout
/// metadata because the logical shape change does not determine a unique output storage layout.
///
/// # Examples
///
/// The following example shows how to use [`Reshape`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::Reshape;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::arrays::Array;
/// # use ryft_core::arrays::{Shape, Dimension};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Reshape a length-6 vector to a `[2, 3]` matrix while keeping the row-major payload unchanged.
/// let x = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let y = x.reshape(Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))?;
/// assert_eq!(y.to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait Reshape: Sized {
    /// Reshapes `self` according to semantic `parameters`. A [`Shape`] converts directly into
    /// [`ReshapeParameters`], so ordinary calls remain `value.reshape(shape)`.
    fn reshape<P: Into<ReshapeParameters>>(&self, parameters: P) -> Result<Self, ProgramError>;
}

impl Reshape for ArrayType {
    fn reshape<P: Into<ReshapeParameters>>(&self, parameters: P) -> Result<ArrayType, ProgramError> {
        let parameters = parameters.into();
        let permuted_input = match parameters.dimensions() {
            Some(dimensions) => self.transpose(dimensions)?,
            None => self.clone(),
        };
        let shape = parameters.output_shape().clone();
        if permuted_input.shape() != &shape {
            if shape.dimensions().iter().any(|size| matches!(size, Dimension::Dynamic(_))) {
                return Err(TypeError::invalid(
                    format!("'{RESHAPE_OPERATION_NAME}' requires explicit result-dimension operands for a dynamic output shape"),
                )
                .into());
            }
            let Some(input_elements) =
                permuted_input.element_count().map_err(|error| TypeError::invalid(error.to_string()))?
            else {
                return Err(TypeError::invalid(
                    format!("'{RESHAPE_OPERATION_NAME}' requires explicit result-dimension operands for a dynamic input shape"),
                )
                .into());
            };
            let Some(output_elements) = shape.element_count().map_err(|error| TypeError::invalid(error.to_string()))?
            else {
                return Err(TypeError::invalid(
                    format!("'{RESHAPE_OPERATION_NAME}' requires explicit result-dimension operands for a dynamic output shape"),
                )
                .into());
            };
            if input_elements != output_elements {
                return Err(TypeError::invalid(format!("'{RESHAPE_OPERATION_NAME}' changes the number of elements")).into());
            }
        }

        let sharding = match parameters.output_sharding() {
            Some(requested) => Some(validate_requested_reshape_sharding(&permuted_input, &shape, requested)?),
            None => permuted_input
                .sharding()
                .map(|sharding| infer_reshape_sharding(&permuted_input, &shape, sharding))
                .transpose()?,
        };

        if parameters.has_identity_dimensions(self.rank()) && self.shape() == &shape {
            return self.clone().with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()).into());
        }

        ArrayType::new(self.data_type(), shape)
            .with_memory(self.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError::invalid(error.to_string()).into())
    }
}

/// Infers the result of the canonical mixed reshape from its explicit output extent types.
fn infer_explicit_reshape_output_type(
    input: &ArrayType,
    output_shape: Shape,
    operation: &DynamicReshapeOperation,
) -> Result<ArrayType, TypeError> {
    let permuted_input = match operation.dimensions() {
        Some(dimensions) => input.transpose(dimensions).map_err(|error| TypeError::invalid(error.to_string()))?,
        None => input.clone(),
    };

    // Reject statically provable element-count mismatches immediately. Dynamic relationships remain explicit graph
    // facts and are checked by eager reshape semantics or the lowered dynamic reshape operation.
    if let (Some(input_elements), Some(output_elements)) =
        (permuted_input.element_count()?, output_shape.element_count()?)
        && input_elements != output_elements
    {
        return Err(TypeError::invalid(format!("'{RESHAPE_OPERATION_NAME}' changes the number of elements")));
    }

    let sharding = match operation.output_sharding() {
        Some(requested) => Some(validate_requested_reshape_sharding(&permuted_input, &output_shape, requested)?),
        None => permuted_input
            .sharding()
            .map(|sharding| infer_reshape_sharding(&permuted_input, &output_shape, sharding))
            .transpose()?,
    };

    if operation.has_identity_dimensions(input.rank()) && input.shape() == &output_shape {
        return input.clone().with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()));
    }

    ArrayType::new(input.data_type(), output_shape)
        .with_memory(input.memory())
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))
}

/// Validates an explicitly requested output sharding for a reshape.
fn validate_requested_reshape_sharding(
    input: &ArrayType,
    output_shape: &Shape,
    requested: &Sharding,
) -> Result<Sharding, TypeError> {
    if requested.rank() != output_shape.rank() {
        return Err(TypeError::invalid(format!(
            "'{RESHAPE_OPERATION_NAME}' requested output sharding rank ({}) does not match the output rank ({})",
            requested.rank(),
            output_shape.rank(),
        )));
    }
    if input.sharding().is_some_and(|input| input.mesh() != requested.mesh()) {
        return Err(TypeError::invalid(format!("'{RESHAPE_OPERATION_NAME}' requested output sharding uses a different mesh")));
    }
    if requested.references_auto_axis() {
        return Err(TypeError::invalid(
            format!("'{RESHAPE_OPERATION_NAME}' requested output sharding cannot reference auto mesh axes"),
        ));
    }
    let input_unreduced = input.sharding().map(Sharding::unreduced_axes).cloned().unwrap_or_default();
    let input_reduced = input.sharding().map(Sharding::reduced_axes).cloned().unwrap_or_default();
    let input_varying = input.sharding().map(Sharding::varying_manual_axes).cloned().unwrap_or_default();
    if requested.unreduced_axes() != &input_unreduced {
        return Err(TypeError::invalid(
            format!("'{RESHAPE_OPERATION_NAME}' requested output sharding changes the unreduced mesh axes"),
        ));
    }
    if requested.reduced_axes() != &input_reduced {
        return Err(TypeError::invalid(
            format!("'{RESHAPE_OPERATION_NAME}' requested output sharding changes the reduced mesh axes"),
        ));
    }
    if requested.varying_manual_axes() != &input_varying {
        return Err(TypeError::invalid(
            format!("'{RESHAPE_OPERATION_NAME}' requested output sharding changes the varying manual mesh axes"),
        ));
    }
    Ok(requested.clone())
}

/// Infers reshape output sharding using JAX-compatible singleton and contiguous split/merge propagation.
fn infer_reshape_sharding(input: &ArrayType, output_shape: &Shape, sharding: &Sharding) -> Result<Sharding, TypeError> {
    let input_dimensions = input
        .shape()
        .dimensions()
        .iter()
        .cloned()
        .enumerate()
        .filter(|(_, size)| *size != Dimension::Static(1))
        .collect::<Vec<_>>();
    let output_dimensions = output_shape
        .dimensions()
        .iter()
        .cloned()
        .enumerate()
        .filter(|(_, size)| *size != Dimension::Static(1))
        .collect::<Vec<_>>();
    let mut output_sharding_dimensions = vec![ShardingDimension::replicated(); output_shape.rank()];
    for axis in 0..input.rank().min(output_shape.rank()) {
        if input.dimension(axis) == Dimension::Static(1) && output_shape.dimension(axis) == Dimension::Static(1) {
            output_sharding_dimensions[axis] = sharding.dimensions()[axis].clone();
        }
    }

    if input_dimensions.iter().map(|(_, size)| size).eq(output_dimensions.iter().map(|(_, size)| size)) {
        for ((input_axis, _), (output_axis, _)) in input_dimensions.iter().zip(&output_dimensions) {
            output_sharding_dimensions[*output_axis] = sharding.dimensions()[*input_axis].clone();
        }
        return rebuild_reshape_sharding(sharding, output_sharding_dimensions);
    }

    if sharding.dimensions().iter().all(|dimension| *dimension == ShardingDimension::Replicated) {
        return rebuild_reshape_sharding(sharding, output_sharding_dimensions);
    }

    if input_dimensions.iter().any(|(_, size)| *size == Dimension::Static(0))
        || output_dimensions.iter().any(|(_, size)| *size == Dimension::Static(0))
    {
        propagate_zero_reshape_sharding(
            &input_dimensions,
            &output_dimensions,
            sharding,
            &mut output_sharding_dimensions,
        )?;
        return rebuild_reshape_sharding(sharding, output_sharding_dimensions);
    }

    let alignment_error = || TypeError::invalid(format!("'{RESHAPE_OPERATION_NAME}' could not align reshape dimension groups"));
    let mut input_start = 0usize;
    let mut output_start = 0usize;
    while input_start < input_dimensions.len() || output_start < output_dimensions.len() {
        if input_start == input_dimensions.len() || output_start == output_dimensions.len() {
            return Err(alignment_error());
        }
        let input_group_start = input_start;
        let output_group_start = output_start;
        let mut input_product = static_positive_size(input_dimensions[input_start].1.clone())?;
        let mut output_product = static_positive_size(output_dimensions[output_start].1.clone())?;
        input_start += 1;
        output_start += 1;
        while input_product != output_product {
            if input_product < output_product {
                let (_, size) = input_dimensions.get(input_start).ok_or_else(alignment_error)?;
                input_product =
                    input_product.checked_mul(static_positive_size(size.clone())?).ok_or_else(alignment_error)?;
                input_start += 1;
            } else {
                let (_, size) = output_dimensions.get(output_start).ok_or_else(alignment_error)?;
                output_product =
                    output_product.checked_mul(static_positive_size(size.clone())?).ok_or_else(alignment_error)?;
                output_start += 1;
            }
        }
        propagate_static_reshape_group(
            &input_dimensions[input_group_start..input_start],
            &output_dimensions[output_group_start..output_start],
            sharding,
            &mut output_sharding_dimensions,
        )?;
    }
    rebuild_reshape_sharding(sharding, output_sharding_dimensions)
}

/// Returns the positive static value of `size` for split/merge factorization.
fn static_positive_size(size: Dimension) -> Result<usize, TypeError> {
    match size {
        Dimension::Static(value) if value > 0 => Ok(value),
        Dimension::Static(_) | Dimension::Dynamic(_) => Err(TypeError::invalid(
            format!("'{RESHAPE_OPERATION_NAME}' requires explicit output sharding for unaligned zero or dynamic dimensions"),
        )),
    }
}

/// Propagates placement around a zero-product reshape without multiplying through zero.
fn propagate_zero_reshape_sharding(
    input_dimensions: &[(usize, Dimension)],
    output_dimensions: &[(usize, Dimension)],
    sharding: &Sharding,
    output_sharding_dimensions: &mut [ShardingDimension],
) -> Result<(), TypeError> {
    let mut prefix = 0usize;
    while input_dimensions.get(prefix).map(|(_, size)| size) == output_dimensions.get(prefix).map(|(_, size)| size) {
        let Some(((input_axis, _), (output_axis, _))) = input_dimensions.get(prefix).zip(output_dimensions.get(prefix))
        else {
            break;
        };
        output_sharding_dimensions[*output_axis] = sharding.dimensions()[*input_axis].clone();
        prefix += 1;
    }

    let mut input_end = input_dimensions.len();
    let mut output_end = output_dimensions.len();
    while input_end > prefix
        && output_end > prefix
        && input_dimensions[input_end - 1].1 == output_dimensions[output_end - 1].1
    {
        input_end -= 1;
        output_end -= 1;
        output_sharding_dimensions[output_dimensions[output_end].0] =
            sharding.dimensions()[input_dimensions[input_end].0].clone();
    }

    if input_dimensions[prefix..input_end]
        .iter()
        .any(|(axis, _)| sharding.dimensions()[*axis] != ShardingDimension::Replicated)
    {
        return Err(TypeError::invalid(
            format!("'{RESHAPE_OPERATION_NAME}' requires explicit output sharding for an ambiguous zero-sized reshape"),
        ));
    }
    if input_dimensions[prefix..input_end].iter().any(|(_, size)| matches!(size, Dimension::Dynamic(_)))
        || output_dimensions[prefix..output_end].iter().any(|(_, size)| matches!(size, Dimension::Dynamic(_)))
    {
        return Err(TypeError::invalid(
            format!("'{RESHAPE_OPERATION_NAME}' requires explicit output sharding for unaligned dynamic dimensions"),
        ));
    }
    Ok(())
}

/// Propagates one positive-static reshape group, distributing contiguous mesh axes over output factors.
fn propagate_static_reshape_group(
    input_group: &[(usize, Dimension)],
    output_group: &[(usize, Dimension)],
    sharding: &Sharding,
    output_sharding_dimensions: &mut [ShardingDimension],
) -> Result<(), TypeError> {
    if input_group.len() == 1 && output_group.len() == 1 {
        output_sharding_dimensions[output_group[0].0] = sharding.dimensions()[input_group[0].0].clone();
        return Ok(());
    }

    let mut mesh_axes = Vec::new();
    let mut saw_replicated = false;
    for (axis, _) in input_group {
        match &sharding.dimensions()[*axis] {
            ShardingDimension::Replicated => saw_replicated = true,
            ShardingDimension::Unconstrained => {
                return Err(TypeError::invalid(
                    format!("'{RESHAPE_OPERATION_NAME}' requires explicit output sharding for unconstrained dimensions"),
                ));
            }
            ShardingDimension::Sharded(axis_names) => {
                if saw_replicated {
                    return Err(TypeError::invalid(
                        format!("'{RESHAPE_OPERATION_NAME}' cannot preserve non-contiguous sharding across a merge"),
                    ));
                }
                mesh_axes.extend(axis_names.iter().cloned());
            }
        }
    }

    let mut mesh_axis_index = 0usize;
    for (output_axis, size) in output_group {
        let mut remaining = static_positive_size(size.clone())?;
        let start = mesh_axis_index;
        while remaining > 1 && mesh_axis_index < mesh_axes.len() {
            let mesh_axis = &mesh_axes[mesh_axis_index];
            let mesh_axis_size = sharding
                .mesh()
                .axis_size(mesh_axis)
                .ok_or_else(|| TypeError::invalid(format!("'{RESHAPE_OPERATION_NAME}' references unknown mesh axis '{mesh_axis}'")))?;
            if remaining % mesh_axis_size != 0 {
                return Err(TypeError::invalid(
                    format!("'{RESHAPE_OPERATION_NAME}' cannot distribute sharding across the requested split factors"),
                ));
            }
            remaining /= mesh_axis_size;
            mesh_axis_index += 1;
        }
        if mesh_axis_index > start {
            output_sharding_dimensions[*output_axis] =
                ShardingDimension::sharded(mesh_axes[start..mesh_axis_index].iter().cloned());
        }
    }
    if mesh_axis_index != mesh_axes.len() {
        return Err(TypeError::invalid(
            format!("'{RESHAPE_OPERATION_NAME}' cannot distribute all input mesh axes across the output dimensions"),
        ));
    }
    Ok(())
}

/// Rebuilds inferred reshape sharding while preserving reduction and manual-axis state.
fn rebuild_reshape_sharding(input: &Sharding, dimensions: Vec<ShardingDimension>) -> Result<Sharding, TypeError> {
    Sharding::new(input.mesh().clone(), dimensions)
        .and_then(|output| output.with_unreduced_axes(input.unreduced_axes().clone()))
        .and_then(|output| output.with_reduced_axes(input.reduced_axes().clone()))
        .and_then(|output| output.with_varying_manual_axes(input.varying_manual_axes().clone()))
        .map_err(|error| TypeError::invalid(error.to_string()))
}

/// Any context-carrying value reshapes by binding a [`ReshapeOperation`] through its own context. The
/// `From<ReshapeOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Reshape for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<ReshapeOperation>,
{
    #[inline]
    fn reshape<P: Into<ReshapeParameters>>(&self, parameters: P) -> Result<Self, ProgramError> {
        let operation = ReshapeOperation::new(parameters);
        let input_type = self.r#type().into_owned();
        let output_type = input_type.reshape(operation.parameters().clone())?;
        if operation.parameters().has_identity_dimensions(input_type.rank()) && input_type == output_type {
            return Ok(self.clone());
        }
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), std::slice::from_ref(self))?;
        Ok(outputs.remove(0))
    }
}

/// Reshapes an array using one explicit first-class dimension value per output axis.
///
/// This is the shape-polymorphic counterpart of [`Reshape`], which reads its complete output geometry from
/// [`ReshapeParameters`]. Exact dimension values describe static axes and computed dimension values describe dynamic
/// axes. Both forms bind the same [`DynamicReshapeOperation`], so runtime shape arithmetic stays an ordinary graph
/// computation instead of a type-level side condition; backend lowering chooses the appropriate static, bounded, or
/// dynamic representation from the inferred result type.
///
/// Exact host sizes can use [`DynamicReshape::dynamic_reshape_to_sizes`]:
///
/// ```rust
/// use ryft_core::operations::manipulation::DynamicReshape;
/// use ryft_core::{Array, ArrayIrValue};
///
/// let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
/// let output = input.dynamic_reshape_to_sizes(&[2, 3]).unwrap();
/// assert_eq!(output, ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));
/// ```
///
/// Computed or input dimensions remain ordinary SSA operands, which is what makes a runtime-derived output shape
/// expressible. Here a `[batch, 6]` input is reshaped so that its dynamic leading extent is read off the input while
/// its trailing extent is an exact lifted dimension. Extents derived by first-class dimension arithmetic work the
/// same way, using the [`DimensionArithmetic`](crate::DimensionArithmetic) capability (e.g.,
/// `rows.dimension_mul(&columns)?`) directly on the composite values.
///
/// ```rust
/// use ryft_core::operations::manipulation::DynamicReshape;
/// use ryft_core::arrays::{
///     ArrayIrType, ArrayType, DataType, Dimension, DimensionBounds, DimensionValue, DimensionVariable, Shape,
/// };
/// use ryft_core::{
///     Array, ArrayIrOperation, ArrayIrValue, Context, DimensionSize, StagingContext, TracingContext, Typed,
/// };
///
/// type C = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
///
/// let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
/// let input_type =
///     ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(6)]));
/// let context = C::new();
/// let input = context.input(ArrayIrType::Array(input_type));
/// let rows = input.dimension_size(0).unwrap();
/// let columns = context.lift(DimensionValue::constant(2).unwrap().into()).unwrap();
/// let depth = context.lift(DimensionValue::constant(3).unwrap().into()).unwrap();
/// let output = input.dynamic_reshape(&[rows, columns, depth]).unwrap();
/// assert_eq!(output.r#type().to_string(), "f32[batch, 2, 3]");
/// ```
pub trait DynamicReshape: Value<Type = ArrayIrType> + Sized {
    /// Reshapes `self` to the output shape described by `output_dimensions`, one first-class value per output axis.
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: Ordered output extents, one per result axis.
    fn dynamic_reshape(&self, output_dimensions: &[Self]) -> Result<Self, ProgramError> {
        self.dynamic_reshape_with_parameters(output_dimensions, None, None)
    }

    /// Reshapes `self` with an optional permutation applied to the input dimensions before the reshape and an
    /// explicit requested output sharding.
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: Ordered output extents, one per result axis.
    ///   - `dimensions`: Permutation applied to the input dimensions before reshaping, if any.
    ///   - `output_sharding`: Requested output [`Sharding`], if any.
    fn dynamic_reshape_with_parameters(
        &self,
        output_dimensions: &[Self],
        dimensions: Option<Permutation>,
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError>;

    /// Reshapes `self` to an exact static shape.
    fn dynamic_reshape_to_sizes(&self, output_sizes: &[usize]) -> Result<Self, ProgramError>
    where
        Self::DispatchDomain: Context<Type = ArrayIrType>,
        <Self::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    {
        let output_dimensions = output_sizes
            .iter()
            .map(|extent| self.dispatch_domain().lift(DimensionValue::constant(*extent)?.into()))
            .collect::<Result<Vec<_>, _>>()?;
        self.dynamic_reshape(output_dimensions.as_slice())
    }
}

impl<
    V: Value<Type = ArrayIrType, DispatchDomain: Context<Type = ArrayIrType, Operation: From<DynamicReshapeOperation>>>,
> DynamicReshape for V
{
    fn dynamic_reshape_with_parameters(
        &self,
        output_dimensions: &[Self],
        dimensions: Option<Permutation>,
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let output_shape =
            Shape::new(ArrayIrType::extents(output_dimensions.iter().map(|dimension| dimension.r#type()))?);
        let mut operation = DynamicReshapeOperation::new().with_output_sharding(output_sharding);
        if let Some(dimensions) = dimensions {
            operation = operation.with_dimensions(dimensions);
        }
        let output_type = infer_explicit_reshape_output_type(input_type, output_shape, &operation)?;

        // A static identity reshape cannot observe its exact extent operands, so it stages nothing. Dynamic geometry
        // keeps the instruction because its operands assert the runtime element-count relation.
        if operation.has_identity_dimensions(input_type.rank())
            && input_type.static_shape().is_some()
            && &output_type == input_type
        {
            return Ok(self.clone());
        }

        let mut inputs = Vec::with_capacity(output_dimensions.len() + 1);
        inputs.push(self.clone());
        inputs.extend_from_slice(output_dimensions);
        Ok(self.dispatch_domain().bind(operation, Vec::new(), inputs.as_slice())?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, DimensionVariable, Layout,
        LogicalMesh, Memory, MeshAxis, MeshAxisType, Sharding, StridedLayout,
    };
    use crate::contexts::EagerContext;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::{DimensionArithmetic, DimensionSize};
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, ProgramError, Typed};
    use crate::tracing::Trace;

    use super::*;

    #[test]
    fn test_reshape() {
        let shape = Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]);
        let operation = ReshapeOperation::new(shape.clone());

        // Operation identity and accessors.
        assert_eq!(operation.name(), RESHAPE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "reshape [shape=[2, 3]]");
        assert_eq!(operation.parameters().output_shape(), &shape);

        // Type inference validates the element count and returns the target shape.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(6)]));
        let output_type = ArrayType::new(DataType::F64, shape.clone());
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input_type.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
                {
                    input_types = [ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(5)]))],
                    error = "'reshape' changes the number of elements",
                },
            ],
        );

        // Type-level (abstract) reshaping validates the target shape and returns the output type without consuming
        // the borrowed input type.
        assert_eq!(input_type.reshape(shape.clone()), Ok(output_type.clone()));

        // Interpretation reinterprets the row-major payload under the target shape.
        let input = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // The optional dimensions permutation is applied before the row-major reshape.
        assert_eq!(
            input.reshape(ReshapeParameters::new(Shape::new(vec![Dimension::Static(6)])).with_dimensions([1, 0]),),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' permutation has length 2 but input has rank 1".to_string()
            ))),
        );
        assert_eq!(
            Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .reshape(ReshapeParameters::new(Shape::new(vec![Dimension::Static(6)])).with_dimensions([1, 0]),)
                .map(|array| array.to_f64s()),
            Ok(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
        );

        // Invalid interpreter arity reports the exact program error.
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[]
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured output shape.
        let mut builder = ProgramBuilder::<Array, ReshapeOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, Vec::new(), vec![program_input]).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[6] .
                let %1:f64[2, 3] = reshape [shape=[2, 3]] %0
                in (%1)
            "}
            .trim_end(),
        );

        // Check the standard partial-evaluation contract for both known and residual inputs.
        let input = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let expected = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            cases = [
                {
                    inputs = [(@known, input.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = input.r#type().into_owned(), replay = input.clone()))],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );

        // Check batching, forward differentiation, and the inverse-reshape pullback.
        let batched_input = Array::matrix(2, 6, (0..12).map(|value| value as f64).collect());
        let batched_output = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 3.into()])),
            (0..12).map(|value| value as f64).collect(),
        );
        check_operation_batching!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), batched_input)],
                outputs = [(@mapped(axis = 0), batched_output)],
            }],
        );
        check_operation_batching!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(
                    6,
                    2,
                    vec![0.0, 6.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0],
                ))],
                outputs = [(@mapped(axis = 0), Array::from_f64s(
                    ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 3.into()])),
                    (0..12).map(|value| value as f64).collect(),
                ))],
            }],
        );
        check_operation_batching!(
            @exact,
            operation = ReshapeOperation::new(
                ReshapeParameters::new(Shape::new(vec![6.into()])).with_dimensions([1, 0]),
            ),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::from_f64s(
                    ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 3.into()])),
                    (1..=12).map(|value| value as f64).collect(),
                ))],
                outputs = [(@mapped(axis = 0), Array::matrix(
                    2,
                    6,
                    vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 7.0, 10.0, 8.0, 11.0, 9.0, 12.0],
                ))],
            }],
        );
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 2.into()])),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0, 3.0, 4.0])],
                tangents = [Array::vector(vec![5.0, 6.0, 7.0, 8.0])],
                primal_outputs = [Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])],
                tangent_outputs = [Array::matrix(2, 2, vec![5.0, 6.0, 7.0, 8.0])],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            cases = [{
                inputs = [(@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![6.into()]))))],
                output_cotangents = [Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
                input_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = ReshapeOperation::new(
                ReshapeParameters::new(Shape::new(vec![6.into()])).with_dimensions([1, 0]),
            ),
            cases = [{
                inputs = [(@linear(type = ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![2.into(), 3.into()]),
                )))],
                output_cotangents = [Array::vector(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])],
                input_cotangents = [Array::matrix(2, 3, vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0])],
            }],
        );

        // Reshaping back to the input shape restores its complete cotangent type after the forward reshape has
        // intentionally cleared physical layout metadata.
        let layout = Layout::Strided(StridedLayout::new(vec![8]));
        let placed_input_type = ArrayType::new(DataType::F64, Shape::new(vec![6.into()]))
            .with_layout(layout)
            .with_memory(Memory::Host { pinned: true });
        let placed_output_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()]))
            .with_memory(Memory::Host { pinned: true });
        check_operation_transposition!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            cases = [{
                inputs = [(@linear(type = placed_input_type.clone()))],
                output_cotangents = [Array::from_f64s(
                    placed_output_type,
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                )],
                input_cotangents = [Array::from_f64s(
                    placed_input_type,
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                )],
            }],
        );
    }

    #[test]
    fn test_array_type_reshape() {
        // Dynamic dimensions can only be reshaped without explicit dimension operands when equality follows directly
        // from identical identity-bearing shapes. Other runtime relationships require the mixed reshape operation and
        // its explicit result-dimension operands.
        let static_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(6)]));
        let dynamic_shape = Shape::new(vec![
            Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
            Dimension::Static(3),
        ]);
        let dynamic_type = ArrayType::new(DataType::F64, dynamic_shape.clone());
        assert_eq!(
            dynamic_type.reshape(Shape::new(vec![Dimension::Static(6)])),
            Err(ProgramError::Type(TypeError::invalid(
                "'reshape' requires explicit result-dimension operands for a dynamic input shape".to_string()
            ))),
        );
        assert_eq!(
            static_type.reshape(dynamic_shape.clone()),
            Err(ProgramError::Type(TypeError::invalid(
                "'reshape' requires explicit result-dimension operands for a dynamic output shape".to_string()
            ))),
        );
        assert_eq!(
            ReshapeOperation::new(dynamic_shape.clone()).infer_output_types(std::slice::from_ref(&static_type), &[]),
            Err(TypeError::invalid(
                "'reshape' requires explicit result-dimension operands for a dynamic output shape".to_string(),
            )),
        );
        assert_eq!(
            Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(dynamic_shape),
            Err(ProgramError::Type(TypeError::invalid(
                "'reshape' requires explicit result-dimension operands for a dynamic output shape".to_string()
            ))),
        );

        // Reshaping a dynamically sized type to its own shape short-circuits as the identity.
        assert_eq!(dynamic_type.reshape(dynamic_type.shape().clone()), Ok(dynamic_type.clone()));

        // A static zero product does not justify anonymous dynamic output dimensions unless a permutation identifies
        // their runtime source. A fully static zero-sized target needs no runtime extent source.
        let trailing = DimensionVariable::new("trailing", DimensionBounds::unbounded());
        let zero_dynamic_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0), Dimension::Dynamic(trailing.clone())]));
        assert_eq!(
            zero_dynamic_type.reshape(Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(0)
            ])),
            Err(ProgramError::Type(TypeError::invalid(
                "'reshape' requires explicit result-dimension operands for a dynamic output shape".to_string()
            ))),
        );
        assert_eq!(
            zero_dynamic_type.reshape(Shape::new(vec![Dimension::Static(0)])),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)]))),
        );
        assert_eq!(
            zero_dynamic_type.reshape(
                ReshapeParameters::new(Shape::new(vec![Dimension::Dynamic(trailing.clone()), Dimension::Static(0),]))
                    .with_dimensions([1, 0]),
            ),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(trailing), Dimension::Static(0)]),)),
        );

        // A non-identity reshape preserves memory placement but clears a layout whose output strides cannot be
        // inferred from the logical target shape alone.
        let placed_type = static_type
            .clone()
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            placed_type.reshape(Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                .with_memory(Memory::Host { pinned: true })),
        );

        // Singleton insertion preserves the corresponding non-singleton dimension placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(1), Dimension::Static(8), Dimension::Static(1)])),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(1), Dimension::Static(8), Dimension::Static(1)])
            )
            .with_sharding(
                Sharding::new(
                    mesh,
                    vec![
                        ShardingDimension::replicated(),
                        ShardingDimension::sharded(["x"]),
                        ShardingDimension::replicated(),
                    ],
                )
                .unwrap(),
            )
            .unwrap())
        );

        // A singleton that stays at the same position retains its own placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        assert_eq!(input_type.reshape(input_type.shape().clone()), Ok(input_type.clone()));

        // Merging replicated axes preserves an independent unchanged sharded dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(8), Dimension::Static(2), Dimension::Static(3)]),
        )
        .with_sharding(
            Sharding::new(
                mesh.clone(),
                vec![
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(8), Dimension::Static(6)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(6)]))
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                        .unwrap(),
                )
                .unwrap())
        );

        // Splitting a replicated axis likewise preserves an unchanged sharded dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(6)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(8), Dimension::Static(2), Dimension::Static(3)])),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(8), Dimension::Static(2), Dimension::Static(3)])
            )
            .with_sharding(
                Sharding::new(
                    mesh,
                    vec![
                        ShardingDimension::sharded(["x"]),
                        ShardingDimension::replicated(),
                        ShardingDimension::replicated(),
                    ],
                )
                .unwrap(),
            )
            .unwrap())
        );

        // Reshape regroups ranked dimensions but leaves the reduction-state (unreduced/reduced) and varying-manual
        // axis sets untouched, since those describe mesh axes that do not correspond to ranked array dimensions.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(6)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["r"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(8), Dimension::Static(2), Dimension::Static(3)])),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(8), Dimension::Static(2), Dimension::Static(3)])
            )
            .with_sharding(
                Sharding::new(
                    mesh,
                    vec![
                        ShardingDimension::sharded(["x"]),
                        ShardingDimension::replicated(),
                        ShardingDimension::replicated(),
                    ],
                )
                .unwrap()
                .with_reduced_axes(["r"])
                .unwrap(),
            )
            .unwrap())
        );

        // A sharded dimension can be split when its mesh axes divide a contiguous prefix of the output factors.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(2), Dimension::Static(4)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                .with_sharding(
                    Sharding::new(
                        input_type.sharding().unwrap().mesh().clone(),
                        vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
                    )
                    .unwrap(),
                )
                .unwrap()),
        );

        // A genuinely merged sharded dimension cannot preserve its placement either.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(8)])),
            Err(ProgramError::Type(TypeError::invalid(
                "'reshape' cannot preserve non-contiguous sharding across a merge".to_string()
            ))),
        );

        // Many-to-many regrouping is supported when every participating dimension is replicated.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(6)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();

        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(4)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(4)]))
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::replicated()],)
                        .unwrap()
                        .with_varying_manual_axes(["x"])
                        .unwrap(),
                )
                .unwrap())
        );

        // A compatible merge keeps sharding from the contiguous outer prefix of the merged dimensions.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(8)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
                .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
                .unwrap()),
        );

        // Explicit output sharding can request a valid redistribution that inference would not choose.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let requested =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        assert_eq!(
            input_type.reshape(
                ReshapeParameters::new(Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                    .with_output_sharding(requested.clone()),
            ),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                .with_sharding(requested)
                .unwrap()),
        );
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_requested =
            Sharding::new(other_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        assert_eq!(
            input_type.reshape(
                ReshapeParameters::new(Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                    .with_output_sharding(other_requested),
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "'reshape' requested output sharding uses a different mesh".to_string()
            ))),
        );
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_requested =
            Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap();
        let unsharded_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        assert_eq!(
            unsharded_input.reshape(
                ReshapeParameters::new(Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                    .with_output_sharding(auto_requested),
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "'reshape' requested output sharding cannot reference auto mesh axes".to_string()
            ))),
        );
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            auto_input.reshape(Shape::new(vec![Dimension::Static(2), Dimension::Static(4)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                .with_sharding(
                    Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],)
                        .unwrap(),
                )
                .unwrap()),
        );

        // Zero-product reshapes preserve fully replicated metadata. A sharded dynamic axis is ambiguous without an
        // explicit output request, and remains available to a caller that supplies one.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated_input = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Static(0),
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
            ]),
        )
        .with_sharding(
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap()
                .with_varying_manual_axes(["x"])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            replicated_input.reshape(Shape::new(vec![Dimension::Static(0)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(0)]))
                .with_sharding(
                    Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                        .unwrap()
                        .with_varying_manual_axes(["x"])
                        .unwrap(),
                )
                .unwrap()),
        );
        let sharded_dynamic_input = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Static(0),
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
            ]),
        )
        .with_sharding(
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            sharded_dynamic_input.reshape(Shape::new(vec![Dimension::Static(0)])),
            Err(ProgramError::Type(TypeError::invalid(
                "'reshape' requires explicit output sharding for an ambiguous zero-sized reshape".to_string()
            ))),
        );
        let zero_requested = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            sharded_dynamic_input.reshape(
                ReshapeParameters::new(Shape::new(vec![Dimension::Static(0)]))
                    .with_output_sharding(zero_requested.clone()),
            ),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(0)]))
                .with_sharding(zero_requested)
                .unwrap()),
        );

        // Lifting an explicit per-item sharding moves a manual mapped axis out of the varying set and onto the new
        // physical batch dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let per_item_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap()
                .with_varying_manual_axes(["x"])
                .unwrap();
        assert_eq!(
            lift_output_sharding_for_leading_batch_axis(&per_item_sharding, ShardingDimension::sharded(["x"])),
            Ok(Sharding::new(
                mesh,
                vec![
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
            )
            .unwrap()),
        );
    }

    #[test]
    fn test_dynamic_reshape() {
        // A concrete composite value resolves every explicit extent operand and reshapes its array member directly.
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let rows = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let columns = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        assert_eq!(
            input.dynamic_reshape(&[rows, columns]).unwrap(),
            ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])),
        );
        assert_eq!(input.dynamic_reshape_to_sizes(&[3, 2]).unwrap().r#type().to_string(), "f64[3, 2]");
        assert_eq!(input.dynamic_reshape_to_sizes(&[6]).unwrap(), input);

        // A staged reshape whose output shape is runtime-derived keeps each extent an ordinary operand: the leading
        // extent is read off the input and the trailing one is first-class dimension arithmetic.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2), Dimension::Static(3)]),
        );
        let (output_type, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| {
                // Dimension arithmetic is a composite capability, so the two static extents multiply directly.
                let rows = input.dimension_size(0)?;
                let columns = input.dimension_size(1)?.dimension_mul(&input.dimension_size(2)?)?;
                input.dynamic_reshape(&[rows, columns])
            },
            ArrayIrType::Array(input_type),
        )
        .unwrap();
        assert_eq!(output_type.to_string(), "f64[batch, 6]");
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[batch, 2, 3] .
                let %1:dimension<batch ∈ [1, 9)> = dimension_size [axis=0] %0
                    %2:dimension<2> = dimension_size [axis=1] %0
                    %3:dimension<3> = dimension_size [axis=2] %0
                    %4:dimension<6> = dimension_mul %2 %3
                    %5:f64[batch, 6] = reshape %0 %1 %4
                in (%5)
            "}
            .trim_end(),
        );

        // A static identity reshape observes none of its extent operands and therefore stages no instruction.
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.dynamic_reshape_to_sizes(&[6]),
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(6)]))),
        )
        .unwrap();
        assert!(program.instructions().is_empty());
    }
}
