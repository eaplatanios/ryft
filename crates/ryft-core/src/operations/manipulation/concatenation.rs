use std::collections::BTreeSet;
use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::batching::align_array_batch;
use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatching, ArrayIrType, ArrayType, Dimension,
    DimensionOperation, DimensionType, DimensionValue, LinearResiduals, Shape, Sharding,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError, InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, ProjectedContext, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    ElementwiseDerivativeAlignment, LinearCallOperation, TransposableOperation, TranspositionDriver,
    transpose_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::zero::{Zero, ZeroOperation};
use crate::operations::dimensions::dimension_add::DimensionAddOperation;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{BroadcastOperation, LegacyBroadcast};
use crate::operations::manipulation::slicing::{DynamicShapeSliceOperation, SliceOperation};
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    Effect, Effects, MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface,
    Type, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`ConcatenateOperation`].
pub const CONCATENATE_OPERATION_NAME: &str = "concatenate";

/// [`Operation`] that joins array operands along one axis. Refer to the documentation of [`Concatenate`] for general
/// concatenation semantics.
///
/// The type parameter selects the operand contract without introducing a separate concatenation operation:
///
///   - `ConcatenateOperation<ArrayType>` accepts only the arrays being concatenated.
///   - `ConcatenateOperation<ArrayIrType>` additionally accepts one trailing first-class result extent.
///
/// Both forms carry the same normalized axis and share all concatenation semantics. The mixed form also records
/// whether its operand types prove that the explicit result extent equals the sum of the input extents. This derived
/// bit lets pure, statically proven concatenations avoid an ordered runtime assertion without retaining a duplicate
/// copy of their complete input signature. Construct mixed operations with
/// [`ConcatenateOperation::<ArrayIrType>::from_input_types`] whenever the complete signature is available. The
/// homogeneous-to-mixed [`From`] conversion remains a generic projection fallback and conservatively retains the
/// assertion because a conversion of the axis-only payload cannot prove anything about its eventual operands. The
/// proof bit intentionally participates in equality and hashing: otherwise operation-keyed deduplication could merge
/// payloads with different effects and scheduling constraints. Consequently, converting a proven mixed operation to
/// the homogeneous form and back is a sound one-way loss of provenness and produces the conservative mixed form.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConcatenateOperation<T: Type> {
    /// Axis along which the operands are joined.
    axis: usize,

    /// Whether validating the explicit result extent can fail for the mixed operand signature used at construction.
    /// This field is meaningful only when `T` is [`ArrayIrType`] and is always `false` for [`ArrayType`].
    requires_runtime_assertion: bool,

    /// Type universe that determines the operation's operand contract.
    marker: PhantomData<fn() -> T>,
}

impl ConcatenateOperation<ArrayType> {
    /// Creates a new [`ConcatenateOperation`] that joins rank-`rank` operands along `axis`.
    #[inline]
    pub fn new<A: Into<Axis>>(axis: A, rank: usize) -> Result<Self, TypeError> {
        let axis = axis.into();
        axis.normalize(rank)
            .map(|axis| Self { axis, requires_runtime_assertion: false, marker: PhantomData })
            .map_err(|_| {
                TypeError::invalid(format!(
                    "'{}' axis {axis} is out of bounds for operands of rank {rank}",
                    CONCATENATE_OPERATION_NAME,
                ))
            })
    }
}

impl ConcatenateOperation<ArrayIrType> {
    /// Creates a mixed [`ConcatenateOperation`] for the provided complete operand signature. `input_types` contains
    /// one or more leading arrays followed by the explicit result-extent dimension. The operation is pure exactly
    /// when those types prove the extent equality; otherwise it carries an ordered runtime assertion.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Axis along which the leading array operands are joined.
    ///   - `input_types`: Complete mixed input signature, including the trailing result-extent dimension.
    pub fn from_input_types<A: Into<Axis>>(axis: A, input_types: &[ArrayIrType]) -> Result<Self, TypeError> {
        let (axis, _, requires_runtime_assertion) = infer_array_ir_concatenation(input_types, axis.into())?;
        Ok(Self { axis, requires_runtime_assertion, marker: PhantomData })
    }
}

impl<T: Type> ConcatenateOperation<T> {
    /// Returns the axis along which this [`ConcatenateOperation`] joins its operands.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Renders this payload independently of its homogeneous or composite operation contract.
    fn render_operation(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CONCATENATE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("axis", self.axis))
    }
}

impl From<ConcatenateOperation<ArrayType>> for ConcatenateOperation<ArrayIrType> {
    #[inline]
    fn from(operation: ConcatenateOperation<ArrayType>) -> Self {
        Self { axis: operation.axis, requires_runtime_assertion: true, marker: PhantomData }
    }
}

impl From<ConcatenateOperation<ArrayIrType>> for ConcatenateOperation<ArrayType> {
    #[inline]
    fn from(operation: ConcatenateOperation<ArrayIrType>) -> Self {
        Self { axis: operation.axis, requires_runtime_assertion: false, marker: PhantomData }
    }
}

impl<T: Type> Display for ConcatenateOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render_operation(formatter, 0)
    }
}

impl Operation for ConcatenateOperation<ArrayIrType> {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        CONCATENATE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let (_, output_type, requires_runtime_assertion) =
            infer_array_ir_concatenation(input_types, Axis::from(self.axis))?;
        if !self.requires_runtime_assertion && requires_runtime_assertion {
            return Err(TypeError::invalid(format!(
                "'{}' was constructed for an operand signature that proves its result extent, but the provided input \
                 types require a runtime extent check",
                CONCATENATE_OPERATION_NAME,
            )));
        }
        Ok(vec![output_type.into()])
    }

    #[inline]
    fn effects(&self) -> Effects {
        if self.requires_runtime_assertion { Effects::single(Effect::OrderedAssertion) } else { Effects::PURE }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render_operation(formatter, indentation)
    }
}

impl Operation for ConcatenateOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        CONCATENATE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        match ArrayType::concatenate(input_types, self.axis) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render_operation(formatter, indentation)
    }
}

impl<C: Domain<Type = ArrayType, Value: Concatenate>> InterpretableOperation<C> for ConcatenateOperation<ArrayType> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(vec![Concatenate::concatenate(inputs, self.axis)?])
    }
}

impl<C> InterpretableOperation<C> for ConcatenateOperation<ArrayIrType>
where
    C: Domain<Type = ArrayIrType>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType> + Concatenate + DimensionSize<usize>>
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
        let Some((result_extent, inputs)) = inputs.split_last() else {
            return Err(TypeError::invalid(format!(
                "'{CONCATENATE_OPERATION_NAME}' expects at least one array followed by its result extent",
            ))
            .into());
        };
        if inputs.is_empty() {
            return match <C::Value as ValueProjection<DimensionType>>::into_projected(result_extent.clone()) {
                Err(_) => Err(TypeError::invalid(format!(
                    "'{CONCATENATE_OPERATION_NAME}' expects a trailing result-extent dimension",
                ))
                .into()),
                Ok(_) => Err(TypeError::invalid(format!(
                    "'{CONCATENATE_OPERATION_NAME}' expects at least one array before its result extent",
                ))
                .into()),
            };
        }
        let result_extent =
            <C::Value as ValueProjection<DimensionType>>::into_projected(result_extent.clone())?.extent();
        let inputs = inputs
            .iter()
            .cloned()
            .map(<C::Value as ValueProjection<ArrayType>>::into_projected)
            .collect::<Result<Vec<_>, _>>()?;
        let actual_extent = inputs.iter().try_fold(0usize, |extent, input| {
            extent.checked_add(input.dimension_size(self.axis())?).ok_or_else(|| {
                ProgramError::from(TypeError::invalid(format!(
                    "'{CONCATENATE_OPERATION_NAME}' result extent overflows usize",
                )))
            })
        })?;
        if result_extent != actual_extent {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "'{CONCATENATE_OPERATION_NAME}' result extent must equal the sum of input axis {} extents; \
                     expected {actual_extent} but got {result_extent}",
                    self.axis(),
                ),
            });
        }
        Ok(vec![<C::Value as ValueProjection<ArrayType>>::from_projected(
            <<C::Value as ValueProjection<ArrayType>>::Projected as Concatenate>::concatenate(&inputs, self.axis())?,
        )])
    }
}

impl<T: Type, C: Context<Type = T, Operation: From<ConcatenateOperation<T>>>> PartiallyEvaluatableOperation<C>
    for ConcatenateOperation<T>
where
    ConcatenateOperation<T>: Operation<Type = T>,
{
}

impl_differentiable_operation! {
    ConcatenateOperation<ArrayType>,
    jvp<C>
    where
        C: Context<Type = ArrayType, Value: Concatenate, Operation: From<ConcatenateOperation<ArrayType>>>
            + Zero<C::Value>,
    {
        |operation, context, _driver, inputs| {
            // Forward-mode rule for `ConcatenateOperation`. Concatenation is linear in every input, so its tangent
            // concatenates the input tangents along the same axis. Materialize structural zeros because concatenation
            // needs one concrete tangent per input; the shared all-zero fast path has already handled that case.
            let tangents = inputs
                .iter()
                .map(|dual| dual.tangent().clone().materialize(context))
                .collect::<Result<Vec<_>, _>>()?;
            let primal = Concatenate::concatenate(inputs.iter().map(DifferentiationDual::primal), operation.axis())?;
            let tangent = Concatenate::concatenate(&tangents, operation.axis())?;
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<SliceOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
    {
        |operation, context, _driver, inputs, outputs| {
            // Transposition rule for `ConcatenateOperation`. The forward map lays its inputs end to end, so its
            // pullback slices the output cotangent at cumulative input offsets. The concatenated input dimensions must
            // be static so those offsets are known. Symbolic-zero cotangents remain symbolic for every input.
            check_count!("output", outputs, 1, ProgramError);
            if inputs.is_empty() {
                return Err(TypeError::invalid(format!(
                    "'{}' transpose expects at least one operand but got none",
                    CONCATENATE_OPERATION_NAME,
                )).into());
            }
            let axis = operation.axis();
            match &outputs[0] {
                MaybeZero::Zero(_) => {
                    Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect())
                }
                MaybeZero::Value(cotangent) => {
                    let rank = inputs[0].r#type().rank();
                    let mut offset = 0usize;
                    let mut input_cotangents = Vec::with_capacity(inputs.len());
                    for (index, input) in inputs.iter().enumerate() {
                        let input_type = input.r#type();
                        let dimension = input_type.dimension(axis);
                        let Dimension::Static(input_axis_size) = dimension else {
                            return Err(TypeError::invalid(format!(
                                    "'{CONCATENATE_OPERATION_NAME}' transpose requires a static size along the \
                                    concatenated axis {axis} but operand {index} has size {dimension}",
                                ))
                            .into());
                        };
                        let mut start_indices = vec![0usize; rank];
                        let mut limit_indices = input_type
                            .shape()
                            .dimensions()
                            .iter()
                            .enumerate()
                            .map(|(other_axis, dimension)| {
                                dimension.value().ok_or_else(|| {
                                    TypeError::invalid(format!(
                                        "'{CONCATENATE_OPERATION_NAME}' transpose requires a static size on axis \
                                         {other_axis} but operand {index} has size {dimension}",
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        start_indices[axis] = offset;
                        limit_indices[axis] = offset + input_axis_size;
                        let slice = SliceOperation::new(start_indices, limit_indices);
                        let outputs = context.stage_operation(slice, Vec::new(), std::slice::from_ref(cotangent))?;
                        check_count!("output", outputs, 1, ProgramError);
                        let input_cotangent =
                            outputs.into_iter().next().unwrap().unalign_cotangent(&input_type.cotangent())?;
                        input_cotangents.push(MaybeZero::Value(input_cotangent));
                        offset += input_axis_size;
                    }
                    Ok(input_cotangents)
                }
            }
        }
    },
}

/// Forward-mode rule for mixed array IR concatenation. The trailing result-extent operand is an ordinary
/// non-differentiated shape value. Static input cotangent shapes replay the mixed concatenate directly; dynamic input
/// shapes are retained as explicit residuals so the transpose can slice the output cotangent at runtime offsets.
impl<C> DifferentiableOperation<C> for ConcatenateOperation<ArrayIrType>
where
    C: Context<Type = ArrayIrType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<ConcatenateOperation<ArrayIrType>>
        + From<DimensionSizeOperation>
        + From<DynamicShapeSliceOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + OperationProjection<ArrayType, Projected: From<ZeroOperation<ArrayType>>>
        + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let Some((result_extent, array_inputs)) = inputs.split_last() else {
            return Err(TypeError::invalid(format!(
                "'{CONCATENATE_OPERATION_NAME}' differentiation expects at least one array followed by its result \
                 extent",
            ))
            .into());
        };
        if array_inputs.is_empty() {
            return match result_extent.primal().r#type().as_ref() {
                ArrayIrType::Array(_) => Err(TypeError::invalid(format!(
                    "'{CONCATENATE_OPERATION_NAME}' differentiation expects a trailing result-extent dimension",
                ))
                .into()),
                ArrayIrType::Dimension(_) => Err(TypeError::invalid(format!(
                    "'{CONCATENATE_OPERATION_NAME}' differentiation expects at least one array before its result \
                     extent",
                ))
                .into()),
            };
        }

        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
        let tangent = if array_inputs.iter().all(|input| input.tangent().is_zero()) {
            MaybeZero::Zero(primal.r#type().tangent())
        } else {
            // Concatenation needs one concrete array tangent per array operand. Materialize only structural zeros;
            // the trailing result extent remains the unchanged primal shape operand.
            let projected_context = ProjectedContext::<C, ArrayType>::new(context.clone());
            let mut tangent_inputs = array_inputs
                .iter()
                .map(|input| -> Result<C::Value, DifferentiationError> {
                    let tangent = match input.tangent() {
                        MaybeZero::Zero(r#type) => MaybeZero::Zero(<&ArrayType>::try_from(r#type)?.clone()),
                        MaybeZero::Value(value) => {
                            MaybeZero::Value(<C::Value as ValueProjection<ArrayType>>::into_projected(value.clone())?)
                        }
                    };
                    Ok(<C::Value as ValueProjection<ArrayType>>::from_projected(
                        tangent.materialize(&projected_context)?,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let input_cotangent_types = array_inputs
                .iter()
                .map(|input| <&ArrayType>::try_from(input.primal().r#type().as_ref()).map(ArrayType::cotangent))
                .collect::<Result<Vec<_>, _>>()?;
            if input_cotangent_types
                .iter()
                .flat_map(|r#type| r#type.shape().dimensions())
                .all(|dimension| matches!(dimension, Dimension::Static(_)))
            {
                tangent_inputs.push(result_extent.primal().clone());
                MaybeZero::Value(context.bind(self.clone(), Vec::new(), tangent_inputs.as_slice())?.remove(0))
            } else {
                let mut residuals = LinearResiduals::new();
                let result_extent_index = residuals.retain(result_extent.primal().clone());
                let mut input_shapes = Vec::with_capacity(array_inputs.len());
                for input in array_inputs {
                    input_shapes.push(residuals.retain_shape(context, input.primal())?);
                }
                let forward_operation = self.clone();
                let transpose_axis = self.axis();
                let tangent = LinearCallOperation::stage(
                    context,
                    residuals.into_values(),
                    tangent_inputs,
                    move |residuals, linear_inputs| {
                        let mut concatenate_inputs = linear_inputs.to_vec();
                        concatenate_inputs.push(residuals[result_extent_index].clone());
                        linear_inputs[0].dispatch_domain().bind(
                            forward_operation,
                            Vec::new(),
                            concatenate_inputs.as_slice(),
                        )
                    },
                    move |residuals, output_cotangents| {
                        let transpose_context = output_cotangents[0].dispatch_domain();
                        let zero = transpose_context
                            .bind(
                                DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(0)?)),
                                Vec::new(),
                                &[],
                            )?
                            .remove(0);
                        let mut offset = zero.clone();
                        let mut cotangents = Vec::with_capacity(input_shapes.len());
                        for (input_index, input_shape) in input_shapes.iter().enumerate() {
                            let sizes = input_shape.dimensions(&transpose_context, residuals)?;
                            let mut starts = vec![zero.clone(); sizes.len()];
                            starts[transpose_axis] = offset.clone();
                            let mut slice_inputs = Vec::with_capacity(1 + 2 * sizes.len());
                            slice_inputs.push(output_cotangents[0].clone());
                            slice_inputs.extend(starts);
                            slice_inputs.extend(sizes.iter().cloned());
                            cotangents.push(
                                transpose_context
                                    .bind(
                                        DynamicShapeSliceOperation::new(sizes.len()),
                                        Vec::new(),
                                        slice_inputs.as_slice(),
                                    )?
                                    .remove(0),
                            );
                            if input_index + 1 < input_shapes.len() {
                                let offset_type = <&DimensionType>::try_from(offset.r#type().as_ref())?.clone();
                                let size_type =
                                    <&DimensionType>::try_from(sizes[transpose_axis].r#type().as_ref())?.clone();
                                offset = transpose_context
                                    .bind(
                                        DimensionOperation::Add(DimensionAddOperation::new(&offset_type, &size_type)?),
                                        Vec::new(),
                                        &[offset, sizes[transpose_axis].clone()],
                                    )?
                                    .remove(0);
                            }
                        }
                        Ok(cotangents)
                    },
                )?
                .remove(0);
                MaybeZero::Value(tangent)
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Direct transposition rule for mixed array IR concatenation. Static concatenated axes delegate to the
/// homogeneous array pullback, while the explicit result extent receives a structural-zero cotangent. Dynamic input
/// extents require linearization so they can be retained as residuals by [`DifferentiableOperation::jvp`].
impl<V, O> TransposableOperation<V, O> for ConcatenateOperation<ArrayIrType>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    O: Operation<Type = ArrayIrType> + OperationProjection<ArrayType>,
    <O as OperationProjection<ArrayType>>::Projected: From<ConcatenateOperation<ArrayType>>
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
        let Some((result_extent, array_inputs)) = inputs.split_last() else {
            return Err(TypeError::invalid(format!(
                "'{CONCATENATE_OPERATION_NAME}' transpose expects at least one array followed by its result extent",
            ))
            .into());
        };
        if array_inputs.is_empty() {
            return match result_extent.r#type().as_ref() {
                ArrayIrType::Array(_) => Err(TypeError::invalid(format!(
                    "'{CONCATENATE_OPERATION_NAME}' transpose expects a trailing result-extent dimension",
                ))
                .into()),
                ArrayIrType::Dimension(_) => Err(TypeError::invalid(format!(
                    "'{CONCATENATE_OPERATION_NAME}' transpose expects at least one array before its result extent",
                ))
                .into()),
            };
        }
        for input in array_inputs {
            let input_type = input.r#type();
            let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
            if matches!(input_type.dimension(self.axis()), Dimension::Dynamic(_)) {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "direct transposition of a dynamic '{CONCATENATE_OPERATION_NAME}' requires linearization so \
                         its input extents can be retained as residuals",
                    ),
                }
                .into());
            }
        }

        let operation = <O as OperationProjection<ArrayType>>::Projected::from(
            ConcatenateOperation::<ArrayType>::from(self.clone()),
        );
        let mut cotangents = transpose_projected_operation(context, &operation, array_inputs, outputs)?;
        cotangents.push(MaybeZero::Zero(result_extent.r#type().cotangent()));
        Ok(cotangents)
    }
}

impl<C: Context<Type = ArrayType, Value: LegacyBroadcast + Transpose>, P: ArrayBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatching<P>> for ConcatenateOperation<ArrayType>
where
    ConcatenateOperation<ArrayType>: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // Align all operands on one physical batch axis (replicated operands are broadcast to gain it via
        // `ArrayBatch::match_axis`, so each batch item concatenates its own operands), and shift the concatenated
        // axis past the inserted batch axis when the batch axis sits at or before it. When no operand is batched,
        // the operation passes through unchanged.
        if inputs.is_empty() {
            return Err(TypeError::invalid(format!(
                "'{CONCATENATE_OPERATION_NAME}' expects at least one operand but got none",
            ))
            .into());
        }
        let Some(batch_axis) = inputs.iter().find_map(ArrayBatch::batch_axis_position) else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        let materialized = inputs
            .iter()
            .map(|input| P::match_axis(context, input, Axis::from(batch_axis)))
            .collect::<Result<Vec<_>, _>>()?;
        let lifted_axis = if batch_axis <= self.axis() { self.axis() + 1 } else { self.axis() };
        ConcatenateOperation::new(lifted_axis, materialized[0].r#type().rank())?.interpret_with_batch_axes(
            context,
            materialized.as_slice(),
            &[BatchAxis::from_position(batch_axis)],
        )
    }
}

/// Batching rule for mixed [`ConcatenateOperation<ArrayIrType>`] instructions. The trailing result extent stays
/// replicated, while array operands are aligned on one physical mapped axis before concatenation.
impl<C: Context<Type = ArrayIrType>> BatchableOperation<C, ArrayIrBatching> for ConcatenateOperation<ArrayIrType>
where
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: LegacyBroadcast + Transpose + Value<Type = ArrayType>>,
    C::Operation: From<ConcatenateOperation<ArrayIrType>>
        + From<BroadcastOperation>
        + From<DimensionOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + OperationProjection<ArrayType>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        let Some((result_extent, inputs)) = inputs.split_last() else {
            return Err(TypeError::invalid(format!(
                "'{CONCATENATE_OPERATION_NAME}' expects at least one array followed by its result extent",
            ))
            .into());
        };
        if inputs.is_empty() {
            return match result_extent.unbatched_type() {
                ArrayIrType::Array(_) => Err(TypeError::invalid(format!(
                    "'{CONCATENATE_OPERATION_NAME}' expects a trailing result-extent dimension",
                ))
                .into()),
                ArrayIrType::Dimension(_) => Err(TypeError::invalid(format!(
                    "'{CONCATENATE_OPERATION_NAME}' expects at least one array before its result extent",
                ))
                .into()),
            };
        }
        // A mapped extent would authorize a different output shape for each batch item, which requires a ragged
        // representation. Concatenate therefore accepts only one replicated result extent.
        result_extent.validate_replicated_dimension()?;

        let Some(batch_axis) = inputs.iter().find_map(ArrayIrBatch::batch_axis_position) else {
            return Ok(context
                .parent()
                .bind(
                    self.clone(),
                    Vec::new(),
                    &inputs
                        .iter()
                        .chain(std::iter::once(result_extent))
                        .map(|input| input.value().clone())
                        .collect::<Vec<_>>(),
                )?
                .into_iter()
                .map(ArrayIrBatch::replicated)
                .collect());
        };

        // Align every packed array on one mapped axis. Replicated operands gain that axis using the transform's
        // declared sharding, so each batch item concatenates the corresponding per-item arrays.
        let aligned_inputs = inputs
            .iter()
            .cloned()
            .map(|input| align_array_batch(context, input, Axis::from(batch_axis)))
            .collect::<Result<Vec<_>, _>>()?;
        let lifted_axis = if batch_axis <= self.axis() { self.axis() + 1 } else { self.axis() };
        let mut lifted_inputs = aligned_inputs.into_iter().map(ArrayIrBatch::into_value).collect::<Vec<_>>();
        lifted_inputs.push(result_extent.value().clone());
        let input_types = lifted_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let operation = ConcatenateOperation::<ArrayIrType>::from_input_types(lifted_axis, &input_types)?;
        context
            .parent()
            .bind(operation, Vec::new(), lifted_inputs.as_slice())?
            .into_iter()
            .map(|output| ArrayIrBatch::new(output, BatchAxis::from_position(batch_axis)))
            .collect()
    }
}

/// Represents the ability to join one or more arrays end to end along one axis. This is the direct analogue of JAX's
/// [`lax.concatenate`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.concatenate.html) and has the semantics of
/// StableHLO's [`concatenate`](https://openxla.org/stablehlo/spec#concatenate) operation. `Self::concatenate(inputs,
/// axis)` preserves the inputs' order and returns an array whose extent along `axis` is the sum of their extents.
/// There must be at least one input. All inputs must have the same element data type, rank, memory space, and
/// dimensions other than `axis`. A single input is returned unchanged without inspecting `axis`. For multiple inputs,
/// the result preserves the common memory space, clears explicit physical layout metadata, and infers sharding,
/// reduction state, and varying-manual axes independently. A dynamic concatenated dimension remains dynamic. Its type
/// records the tight exclusive upper bound when every operand is bounded, while the operation's runtime extent is the
/// exact sum of the operand extents. Equal dynamic descriptors on non-concatenated axes express the runtime requirement
/// that those extents agree; the backend validates that requirement when executing the operation.
///
/// # Example
///
/// The following example shows how to use [`Concatenate`] in practice:
///
/// ```rust
/// # use ryft_core::backends::arrays::Array;
/// # use ryft_core::operations::manipulation::Concatenate;
/// # use ryft_core::programs::ProgramError;
/// #
/// # fn main() -> Result<(), ProgramError> {
/// let left = Array::matrix(2, 1, vec![1.0, 2.0]);
/// let right = Array::matrix(2, 2, vec![3.0, 4.0, 5.0, 6.0]);
/// let output = left.concatenate_with([&right], -1)?;
/// assert_eq!(output.to_f64s(), vec![1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait Concatenate: Sized {
    /// Joins `inputs` end-to-end along `axis`. Negative axes index from the final axis. If `inputs` contains only one
    /// value, returns that value unchanged without inspecting `axis`. Refer to the documentation of this trait for more
    /// information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Values to join, in order. There must be at least one input, and, for [`ArrayType`]d values, all
    ///     inputs must share one [`DataType`](crate::arrays::DataType) and rank and agree on every axis other than
    ///     `axis`.
    ///   - `axis`: [`Axis`] along which the inputs are joined. Negative axes index from the final axis.
    fn concatenate<'i, I: IntoIterator<Item = &'i Self>, A: Into<Axis>>(
        inputs: I,
        axis: A,
    ) -> Result<Self, ProgramError>
    where
        Self: 'i;

    /// Joins `self` followed by `others` end to end along `axis`.
    ///
    /// # Parameters
    ///
    ///   - `others`: Values to append to `self`, in order.
    ///   - `axis`: [`Axis`] along which the inputs are joined. Negative axes index from the final axis.
    #[inline]
    fn concatenate_with<'i, I: IntoIterator<Item = &'i Self>, A: Into<Axis>>(
        &'i self,
        others: I,
        axis: A,
    ) -> Result<Self, ProgramError>
    where
        Self: 'i,
    {
        Self::concatenate(std::iter::once(self).chain(others), axis)
    }
}

impl Concatenate for ArrayType {
    fn concatenate<'i, I: IntoIterator<Item = &'i Self>, A: Into<Axis>>(
        inputs: I,
        axis: A,
    ) -> Result<Self, ProgramError> {
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        let Some(first) = inputs.first() else {
            return Err(TypeError::invalid(format!(
                "'{CONCATENATE_OPERATION_NAME}' expects at least one operand but got none",
            ))
            .into());
        };
        if inputs.len() == 1 {
            return Ok((*first).clone());
        }
        let rank = first.rank();
        let axis = ConcatenateOperation::new(axis, rank)?.axis();
        let mut dimensions = first.shape().dimensions().to_vec();
        dimensions[axis] = validate_concatenation_inputs(&inputs, axis)?.map(Dimension::Static).ok_or_else(|| {
            TypeError::invalid(format!(
                "'{}' dynamic axis {axis} requires an explicit result-dimension operand",
                CONCATENATE_OPERATION_NAME,
            ))
        })?;
        ArrayType::new(first.data_type(), Shape::new(dimensions))
            .with_memory(first.memory())
            .with_sharding(infer_concatenation_sharding(&inputs)?)
            .map_err(|error| TypeError::invalid(error.to_string()).into())
    }
}

impl<V: Value<Type = ArrayType>> Concatenate for V
where
    V::DispatchDomain: Context<Type = ArrayType, Operation: From<ConcatenateOperation<ArrayType>>>,
{
    fn concatenate<'i, I: IntoIterator<Item = &'i Self>, A: Into<Axis>>(
        inputs: I,
        axis: A,
    ) -> Result<Self, ProgramError>
    where
        V: 'i,
    {
        // Any context-carrying value concatenates by binding a `ConcatenateOperation<ArrayType>` through its own
        // context. The conversion bound makes this disjoint from the eager value types (whose operation is
        // `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete
        // implementations.
        let mut inputs = inputs.into_iter().cloned().collect::<Vec<_>>();
        let Some(rank) = inputs.first().map(|input| input.r#type().rank()) else {
            return Err(TypeError::invalid(format!(
                "'{CONCATENATE_OPERATION_NAME}' expects at least one operand but got none",
            ))
            .into());
        };
        if inputs.len() == 1 {
            return Ok(inputs.pop().unwrap());
        }
        let operation = ConcatenateOperation::new(axis, rank)?;
        let first = &inputs[0];
        let mut outputs = first.dispatch_domain().bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Infers a mixed concatenation's normalized axis, output type, and runtime-assertion requirement.
fn infer_array_ir_concatenation(
    input_types: &[ArrayIrType],
    axis: Axis,
) -> Result<(usize, ArrayType, bool), TypeError> {
    let Some((result_extent, inputs)) = input_types.split_last() else {
        return Err(TypeError::invalid(format!(
            "'{}' expects at least one array followed by its result extent",
            CONCATENATE_OPERATION_NAME,
        )));
    };
    if inputs.is_empty() {
        return match result_extent {
            ArrayIrType::Array(_) => Err(TypeError::invalid(format!(
                "'{}' expects a trailing result-extent dimension",
                CONCATENATE_OPERATION_NAME,
            ))),
            ArrayIrType::Dimension(_) => Err(TypeError::invalid(format!(
                "'{}' expects at least one array before its result extent",
                CONCATENATE_OPERATION_NAME,
            ))),
        };
    }
    let inputs = inputs.iter().map(<&ArrayType>::try_from).collect::<Result<Vec<_>, _>>()?;
    let result_extent = <&DimensionType>::try_from(result_extent)?;
    let rank = inputs[0].rank();
    let axis = axis.normalize(rank).map_err(|_| {
        TypeError::invalid(format!(
            "'{}' axis {axis} is out of bounds for operands of rank {rank}",
            CONCATENATE_OPERATION_NAME,
        ))
    })?;
    let static_sum = validate_concatenation_inputs(&inputs, axis)?;
    let result_dimension = result_extent.to_dimension();
    if let Some(static_sum) = static_sum
        && result_dimension != Dimension::Static(static_sum)
    {
        return Err(TypeError::invalid(format!(
            "'{}' result extent is {} but the static input extent sum is {static_sum}",
            CONCATENATE_OPERATION_NAME, result_dimension,
        )));
    }

    let first = inputs[0];
    let mut dimensions = first.shape().dimensions().to_vec();
    dimensions[axis] = result_dimension;
    let output_shape = Shape::new(dimensions);
    let output_type = if inputs.len() == 1 && first.shape() == &output_shape {
        first.clone()
    } else {
        ArrayType::new(first.data_type(), output_shape)
            .with_memory(first.memory())
            .with_sharding(infer_concatenation_sharding(&inputs)?)
            .map_err(|error| TypeError::invalid(error.to_string()))?
    };
    Ok((axis, output_type, static_sum.is_none()))
}

/// Validates concatenation array inputs and returns their axis sum when every input axis is static.
fn validate_concatenation_inputs(inputs: &[&ArrayType], axis: usize) -> Result<Option<usize>, TypeError> {
    let first = inputs[0];
    let rank = first.rank();
    if axis >= rank {
        return Err(TypeError::invalid(format!(
            "'{CONCATENATE_OPERATION_NAME}' axis {axis} is out of bounds for operands of rank {rank}",
        )));
    }
    let mut concatenated_static = 0usize;
    let mut all_static = true;
    for (index, operand) in inputs.iter().enumerate() {
        if operand.data_type() != first.data_type() {
            return Err(TypeError::invalid(format!(
                "'{}' operands must share one data type but operand {} has data type {} \
                and operand 0 has data type {}",
                CONCATENATE_OPERATION_NAME,
                index,
                operand.data_type(),
                first.data_type(),
            )));
        }

        if operand.rank() != rank {
            return Err(TypeError::invalid(format!(
                "'{}' operands must share one rank but operand {} has rank {} and operand 0 has rank {}",
                CONCATENATE_OPERATION_NAME,
                index,
                operand.rank(),
                rank,
            )));
        }

        if operand.memory() != first.memory() {
            return Err(TypeError::invalid(format!(
                "'{}' operands must share one memory space but operand {} resides in {} and operand 0 resides in {}",
                CONCATENATE_OPERATION_NAME,
                index,
                operand.memory(),
                first.memory(),
            )));
        }

        for other_axis in 0..rank {
            let dimension = operand.dimension(other_axis);
            let first_dimension = first.dimension(other_axis);
            if other_axis != axis && dimension != first_dimension {
                return Err(TypeError::invalid(format!(
                    "'{CONCATENATE_OPERATION_NAME}' operands must agree on every axis other than {axis} but \
                         operand {index} has size {dimension} on axis {other_axis} and operand 0 has size \
                         {first_dimension}",
                )));
            }
        }

        match operand.dimension(axis) {
            Dimension::Static(size) => {
                concatenated_static = concatenated_static.checked_add(size).ok_or_else(|| {
                    TypeError::invalid(format!(
                        "'{CONCATENATE_OPERATION_NAME}' output size overflows usize on axis {axis}",
                    ))
                })?;
            }
            Dimension::Dynamic(_) => {
                all_static = false;
            }
        }
    }

    Ok(all_static.then_some(concatenated_static))
}

/// Infers the [`Sharding`] of the result of a concatenation operation using JAX's independent spatial,
/// varying-manual, unreduced, and reduced rules.
fn infer_concatenation_sharding(inputs: &[&ArrayType]) -> Result<Option<Sharding>, TypeError> {
    let varying_manual_axes = inputs
        .iter()
        .filter_map(|input| input.sharding())
        .flat_map(|sharding| sharding.varying_manual_axes().iter().cloned())
        .collect::<BTreeSet<_>>();
    let mut spatial_sharding: Option<&Sharding> = None;
    let mut unreduced_axes = None;
    let mut reduced_axes = None;
    for operand in inputs {
        let operand_varying_manual_axes =
            operand.sharding().map(Sharding::varying_manual_axes).cloned().unwrap_or_default();
        let added_varying_manual_axes =
            varying_manual_axes.difference(&operand_varying_manual_axes).cloned().collect::<BTreeSet<_>>();
        let mut normalized_reduced_axes = operand.reduced_axes().clone();
        if normalized_reduced_axes == added_varying_manual_axes {
            normalized_reduced_axes.clear();
        } else if !operand.unreduced_axes().is_disjoint(&added_varying_manual_axes)
            || !normalized_reduced_axes.is_disjoint(&added_varying_manual_axes)
        {
            return Err(TypeError::invalid(format!(
                "'{CONCATENATE_OPERATION_NAME}' cannot make operand varying over axes \
                     {added_varying_manual_axes:?} while it is reduced or unreduced over any of those axes",
            )));
        }
        unreduced_axes = merge_concatenation_axis_set(unreduced_axes, operand.unreduced_axes(), "unreduced")?;
        reduced_axes = merge_concatenation_axis_set(reduced_axes, &normalized_reduced_axes, "reduced")?;

        let Some(sharding) = operand.sharding().filter(|sharding| sharding.mesh().rank() > 0) else {
            continue;
        };

        match spatial_sharding {
            Some(reference)
                if reference.mesh() != sharding.mesh() || reference.dimensions() != sharding.dimensions() =>
            {
                return Err(TypeError::invalid(format!(
                    "'{CONCATENATE_OPERATION_NAME}' operands must be sharded identically, but got {reference} \
                         and {sharding}",
                )));
            }
            None => spatial_sharding = Some(sharding),
            Some(_) => {}
        }
    }

    let Some(spatial_sharding) = spatial_sharding else { return Ok(None) };
    Sharding::new(spatial_sharding.mesh().clone(), spatial_sharding.dimensions().to_vec())
        .and_then(|sharding| sharding.with_unreduced_axes(unreduced_axes.unwrap_or_default()))
        .and_then(|sharding| sharding.with_reduced_axes(reduced_axes.unwrap_or_default()))
        .and_then(|sharding| sharding.with_varying_manual_axes(varying_manual_axes))
        .map(Some)
        .map_err(|error| TypeError::invalid(error.to_string()))
}

/// Merges one nonempty reduced or unreduced axis set into the corresponding concatenation result state.
fn merge_concatenation_axis_set(
    current: Option<BTreeSet<String>>,
    axes: &BTreeSet<String>,
    state: &str,
) -> Result<Option<BTreeSet<String>>, TypeError> {
    if axes.is_empty() {
        return Ok(current);
    }
    match current {
        Some(reference) if &reference != axes => Err(TypeError::invalid(format!(
            "'{CONCATENATE_OPERATION_NAME}' operands must be {state} over the same nonempty axis set",
        ))),
        None => Ok(Some(axes.clone())),
        current => Ok(current),
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        DataType, DimensionBounds, DimensionValue, DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis,
        MeshAxisType, Sharding, ShardingDimension, StridedLayout,
    };
    use crate::backends::{Array, ArrayOperation};
    use crate::contexts::EagerContext;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{Effect, Effects, EmptyRegionDriver, ProgramBuilder, ProgramError, Typed};

    use super::*;

    #[test]
    fn test_explicit_concatenate_operation() {
        let operation = ConcatenateOperation::new(-2, 2).unwrap();
        let mixed_operation = ConcatenateOperation::<ArrayIrType>::from(operation.clone());
        assert_eq!(operation.axis(), 0);
        assert_eq!(mixed_operation.name(), CONCATENATE_OPERATION_NAME);
        assert_eq!(operation.to_string(), "concatenate [axis=0]");
        assert_eq!(mixed_operation.effects(), Effects::single(Effect::OrderedAssertion));
        assert_eq!(operation.effects(), Effects::PURE);
        let infer = |input_types: &[ArrayIrType]| mixed_operation.infer_output_types(input_types, &[]);

        let first_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)]));
        let second_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let four = DimensionValue::constant(4).unwrap().r#type().clone();
        let static_input_types = [first_type.clone().into(), second_type.clone().into(), four.clone().into()];
        let proven_operation = ConcatenateOperation::<ArrayIrType>::from_input_types(-2, &static_input_types).unwrap();
        assert_eq!(proven_operation.axis(), 0);
        assert_eq!(proven_operation.effects(), Effects::PURE);
        assert_ne!(proven_operation, mixed_operation);
        let round_tripped = ConcatenateOperation::<ArrayIrType>::from(ConcatenateOperation::<ArrayType>::from(
            proven_operation.clone(),
        ));
        assert_eq!(round_tripped.effects(), Effects::single(Effect::OrderedAssertion));
        assert_ne!(round_tripped, proven_operation);
        assert_eq!(
            proven_operation.infer_output_types(&static_input_types, &[]),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(2)])).into()
            ]),
        );
        assert_eq!(
            infer(&static_input_types),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(2)])).into()
            ]),
        );
        assert_eq!(
            infer(&[first_type.clone().into(), DimensionValue::constant(1).unwrap().r#type().clone().into()]),
            Ok(vec![first_type.clone().into()]),
        );

        let left = DimensionVariable::new("left", DimensionBounds::new(1, Some(5)).unwrap());
        let right = DimensionVariable::new("right", DimensionBounds::new(1, Some(6)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::positive(Some(4)).unwrap());
        let result = DimensionVariable::new("result", DimensionBounds::new(2, Some(10)).unwrap());
        let dynamic_left = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(left), Dimension::Dynamic(columns.clone())]),
        );
        let dynamic_right = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(right), Dimension::Dynamic(columns.clone())]),
        );
        let dynamic_input_types =
            [dynamic_left.clone().into(), dynamic_right.clone().into(), DimensionType::new(result.clone()).into()];
        let dynamic_operation = ConcatenateOperation::<ArrayIrType>::from_input_types(0, &dynamic_input_types).unwrap();
        assert_eq!(dynamic_operation.effects(), Effects::single(Effect::OrderedAssertion));
        assert_eq!(
            dynamic_operation.infer_output_types(&dynamic_input_types, &[]),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(result), Dimension::Dynamic(columns)]),
                )
                .into()
            ]),
        );
        assert_eq!(
            proven_operation.infer_output_types(&dynamic_input_types, &[]),
            Err(TypeError::invalid(format!(
                "'{}' was constructed for an operand signature that proves its result extent, but the provided input \
                 types require a runtime extent check",
                CONCATENATE_OPERATION_NAME,
            ))),
        );

        let placed_first = first_type
            .clone()
            .with_layout(Layout::Strided(StridedLayout::new(vec![8, 4])))
            .with_memory(Memory::Host { pinned: true });
        let placed_second = second_type
            .clone()
            .with_layout(Layout::Strided(StridedLayout::new(vec![8, 4])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            infer(&[placed_first.into(), placed_second.into(), four.clone().into()]),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(2)]))
                    .with_memory(Memory::Host { pinned: true })
                    .into()
            ]),
        );

        assert_eq!(
            infer(&[]),
            Err(TypeError::invalid(format!(
                "'{}' expects at least one array followed by its result extent",
                CONCATENATE_OPERATION_NAME,
            ))),
        );
        assert_eq!(
            infer(&[first_type.clone().into()]),
            Err(TypeError::invalid(format!(
                "'{}' expects a trailing result-extent dimension",
                CONCATENATE_OPERATION_NAME,
            ))),
        );
        assert_eq!(
            infer(&[four.clone().into()]),
            Err(TypeError::invalid(format!(
                "'{}' expects at least one array before its result extent",
                CONCATENATE_OPERATION_NAME,
            ))),
        );
        assert_eq!(
            infer(&[four.clone().into(), four.clone().into()]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            infer(&[first_type.clone().into(), four.clone().into(), four.clone().into()]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            infer(&[first_type.clone().into(), second_type.clone().into()]),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
        assert_eq!(
            infer(&[
                first_type.clone().into(),
                second_type.clone().into(),
                DimensionValue::constant(5).unwrap().r#type().clone().into(),
            ]),
            Err(TypeError::invalid(format!(
                "'{}' result extent is 5 but the static input extent sum is 4",
                CONCATENATE_OPERATION_NAME,
            ))),
        );
        assert_eq!(
            ConcatenateOperation::<ArrayIrType>::from_input_types(
                0,
                &[
                    first_type.clone().into(),
                    second_type.into(),
                    DimensionValue::constant(5).unwrap().r#type().clone().into(),
                ],
            ),
            Err(TypeError::invalid(format!(
                "'{}' result extent is 5 but the static input extent sum is 4",
                CONCATENATE_OPERATION_NAME,
            ))),
        );
        assert_eq!(
            ConcatenateOperation::<ArrayIrType>::from(ConcatenateOperation::new(1, 2).unwrap()).infer_output_types(
                &[
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).into(),
                    DimensionValue::constant(1).unwrap().r#type().clone().into(),
                ],
                &[],
            ),
            Err(TypeError::invalid(format!(
                "'{}' axis 1 is out of bounds for operands of rank 1",
                CONCATENATE_OPERATION_NAME,
            ))),
        );
    }

    #[test]
    fn test_concatenate() {
        #[derive(Debug, PartialEq)]
        struct NonCloneArray(Vec<i32>);

        impl Concatenate for NonCloneArray {
            fn concatenate<'i, I: IntoIterator<Item = &'i Self>, A: Into<Axis>>(
                inputs: I,
                _axis: A,
            ) -> Result<Self, ProgramError> {
                Ok(Self(inputs.into_iter().flat_map(|input| input.0.iter().copied()).collect()))
            }
        }

        let operation = ConcatenateOperation::new(0, 2).unwrap();

        // Operation identity and accessors.
        assert_eq!(operation.name(), CONCATENATE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "concatenate [axis=0]");
        assert_eq!(operation.axis(), 0);

        // Type inference sums static concatenated axes, preserves matching non-concatenated axes, and reports exact
        // validation errors. Until the mixed signature lands, a dynamic result extent cannot be represented without
        // manufacturing an unstable identity, so it requires an explicit result-dimension operand.
        let first_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)]));
        let second_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4), Dimension::Static(2)]));
        let dynamic_stack = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(2),
            ]),
        );
        let bounded_stack = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(4)).unwrap())),
                Dimension::Static(2),
            ]),
        );
        let fixed_slice = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)]));
        let dynamic_non_axis = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Static(1),
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
            ]),
        );
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    type = ArrayType,
                    input_types = [first_type.clone(), second_type.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    type = ArrayType,
                    input_types = [dynamic_stack, fixed_slice.clone()],
                    error = "'concatenate' dynamic axis 0 requires an explicit result-dimension operand",
                },
                {
                    type = ArrayType,
                    input_types = [bounded_stack, fixed_slice],
                    error = "'concatenate' dynamic axis 0 requires an explicit result-dimension operand",
                },
                {
                    type = ArrayType,
                    input_types = [dynamic_non_axis.clone(), dynamic_non_axis.clone()],
                    output_types = [ArrayType::new(
                        DataType::F64,
                        Shape::new(vec![Dimension::Static(2), dynamic_non_axis.shape()[1].clone()]),
                    )],
                },
                {
                    type = ArrayType,
                    input_types = [],
                    error = "'concatenate' expects at least one operand but got none",
                },
                {
                    type = ArrayType,
                    input_types = [first_type.clone(), ArrayType::scalar(DataType::F64)],
                    error = "'concatenate' operands must share one rank but operand 1 has rank 0 and operand 0 has \
                        rank 2",
                },
                {
                    type = ArrayType,
                    input_types = [
                        first_type.clone(),
                        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
                    ],
                    error = "'concatenate' operands must share one data type but operand 1 has data type f32 and \
                        operand 0 has data type f64",
                },
                {
                    type = ArrayType,
                    input_types = [
                        first_type.clone(),
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(5)])),
                    ],
                    error = "'concatenate' operands must agree on every axis other than 0 but operand 1 has size 5 \
                        on axis 1 and operand 0 has size 2",
                },
                {
                    type = ArrayType,
                    input_types = [
                        dynamic_non_axis,
                        ArrayType::new(
                            DataType::F64,
                            Shape::new(vec![
                                Dimension::Static(1),
                                Dimension::Dynamic(DimensionVariable::new(
                                    "dynamic",
                                    DimensionBounds::non_negative(Some(3)).unwrap(),
                                )),
                            ]),
                        ),
                    ],
                    error = "'concatenate' operands must agree on every axis other than 0 but operand 1 has size \
                        dynamic on axis 1 and operand 0 has size dynamic",
                },
            ],
        );
        assert_eq!(ConcatenateOperation::new(-1, 2).unwrap().axis(), 1);
        assert_eq!(
            ConcatenateOperation::new(2, 2),
            Err(TypeError::invalid("'concatenate' axis 2 is out of bounds for operands of rank 2".to_string())),
        );

        // Interpretation joins the row-major payloads along axis 0, while the sole-input fast path returns its input
        // without inspecting the axis.
        let first = Array::matrix(1, 2, vec![1.0, 2.0]);
        let second = Array::matrix(3, 2, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let output = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[first.clone(), second.clone()])
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(Array::concatenate([&first, &second], -2), Ok(output[0].clone()));
        assert_eq!(first.concatenate_with([&second], -2), Ok(output[0].clone()));
        assert_eq!(Array::concatenate([&first], 2), Ok(first.clone()));
        assert_eq!(
            NonCloneArray(vec![1]).concatenate_with([&NonCloneArray(vec![2, 3])], 0),
            Ok(NonCloneArray(vec![1, 2, 3])),
        );

        // Program rendering uses the canonical operation name and includes the captured axis.
        let mut builder = ProgramBuilder::<Array, ConcatenateOperation<ArrayType>>::new();
        let program_first = builder.add_input(first_type);
        let program_second = builder.add_input(second_type);
        let program_output =
            builder.add_instruction(operation, Vec::new(), vec![program_first, program_second]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![program_output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[1, 2], %1:f64[3, 2] .
                let %2:f64[4, 2] = concatenate [axis=0] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Check standard partial evaluation with known and residual operands.
        let first = Array::vector(vec![1.0, 2.0]);
        let second = Array::vector(vec![3.0, 4.0, 5.0]);
        let expected = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = ConcatenateOperation::new(0, 1).unwrap(),
            cases = [
                {
                    inputs = [(@known, first.clone()), (@known, second.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = first.r#type().into_owned(), replay = first.clone())),
                        (@known, second.clone()),
                    ],
                    outputs = [(@residual, expected.clone())],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@unknown(type = first.r#type().into_owned(), replay = first.clone())),
                        (@unknown(type = second.r#type().into_owned(), replay = second.clone())),
                    ],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );

        // Batching aligns mapped and replicated operands before lifting the concatenation axis.
        check_operation_batching!(
            @exact,
            operation = ConcatenateOperation::new(0, 1).unwrap(),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0])),
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![4.0, 5.0, 6.0, 7.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        4,
                        vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0],
                    ))],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0])),
                        (@replicated, Array::vector(vec![8.0, 9.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        4,
                        vec![0.0, 1.0, 8.0, 9.0, 2.0, 3.0, 8.0, 9.0],
                    ))],
                },
                {
                    inputs = [
                        (@replicated, Array::vector(vec![1.0, 2.0])),
                        (@replicated, Array::vector(vec![3.0])),
                    ],
                    outputs = [(@replicated, Array::vector(vec![1.0, 2.0, 3.0]))],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0])),
                        (@mapped(axis = 1), Array::matrix(1, 2, vec![4.0, 6.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        3,
                        vec![0.0, 1.0, 4.0, 2.0, 3.0, 6.0],
                    ))],
                },
                {
                    inputs = [
                        (@mapped(axis = 1), Array::matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0])),
                        (@mapped(axis = 1), Array::matrix(1, 2, vec![4.0, 5.0])),
                    ],
                    outputs = [(@mapped(axis = 1), Array::matrix(
                        3,
                        2,
                        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    ))],
                },
            ],
        );

        // Concatenate is linear in each operand: the JVP concatenates tangents and the pullback splits cotangents.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = ConcatenateOperation::new(0, 1).unwrap(),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0]), Array::vector(vec![3.0, 4.0, 5.0])],
                tangents = [Array::vector(vec![0.5, 1.0]), Array::vector(vec![1.5, 2.0, 2.5])],
                primal_outputs = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0])],
                tangent_outputs = [Array::vector(vec![0.5, 1.0, 1.5, 2.0, 2.5])],
                jvp = indoc! {"
                    lambda %0:f64[2], %1:f64[3], %2:f64[2], %3:f64[3] .
                    let %4:f64[5] = concatenate [axis=0] %0 %1
                        %5:f64[5] = concatenate [axis=0] %2 %3
                    in (%4, %5)
                "},
            }],
        );

        let layout = Layout::Strided(StridedLayout::new(vec![8]));
        let placed_left_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()]))
            .with_layout(layout.clone())
            .with_memory(Memory::Host { pinned: true });
        let placed_right_type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()]))
            .with_layout(layout)
            .with_memory(Memory::Host { pinned: true });
        let placed_output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![5.into()])).with_memory(Memory::Host { pinned: true });
        check_operation_transposition!(
            @exact,
            operation = ConcatenateOperation::new(0, 1).unwrap(),
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()])))),
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()])))),
                    ],
                    output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0])],
                    input_cotangents = [Array::vector(vec![1.0, 2.0]), Array::vector(vec![3.0, 4.0, 5.0])],
                    pullback = indoc! {"
                        lambda %0:f64[5] .
                        let %1:f64[2] = slice [start_indices=[0], limit_indices=[2]] %0
                            %2:f64[3] = slice [start_indices=[2], limit_indices=[5]] %0
                        in (%1, %2)
                    "},
                },
                {
                    inputs = [
                        (@linear(type = placed_left_type.clone())),
                        (@linear(type = placed_right_type.clone())),
                    ],
                    output_cotangents = [Array::from_f64s(
                        placed_output_type,
                        vec![1.0, 2.0, 3.0, 4.0, 5.0],
                    )],
                    input_cotangents = [
                        Array::from_f64s(placed_left_type, vec![1.0, 2.0]),
                        Array::from_f64s(placed_right_type, vec![3.0, 4.0, 5.0]),
                    ],
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![0.into()])))),
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()])))),
                    ],
                    output_cotangents = [Array::vector(vec![1.0, 2.0])],
                    input_cotangents = [
                        Array::from_f64s(
                            ArrayType::new(DataType::F64, Shape::new(vec![0.into()])),
                            Vec::new(),
                        ),
                        Array::vector(vec![1.0, 2.0]),
                    ],
                },
            ],
        );

        check_operation_transposition!(
            @exact,
            operation = ConcatenateOperation::new(1, 2).unwrap(),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 1.into()])))),
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into()])))),
                ],
                output_cotangents = [Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
                input_cotangents = [
                    Array::matrix(2, 1, vec![1.0, 4.0]),
                    Array::matrix(2, 2, vec![2.0, 3.0, 5.0, 6.0]),
                ],
            }],
        );

        // A dynamic non-concatenated axis cannot be reconstructed by this homogeneous static-slice rule. Composite
        // differentiation retains that runtime extent as an explicit residual. Direct homogeneous transposition
        // rejects the case instead of consulting hidden input-shape metadata.
        let columns = DimensionVariable::new("columns", DimensionBounds::unbounded());
        let left_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(columns.clone())]));
        let right_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(columns)]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(left_type);
        let right = builder.add_input(right_type);
        let output = builder
            .add_instruction(ConcatenateOperation::new(0, 2).unwrap(), Vec::new(), vec![left, right])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1]),
            Err(crate::differentiation::DifferentiationError::Program(ProgramError::Type(
                TypeError::Invalid { message },
            ))) if message
                == "'concatenate' transpose requires a static size on axis 1 but operand 0 has size columns",
        ));
    }

    #[test]
    fn test_concatenate_with_sharding() {
        let operation = ConcatenateOperation::new(0, 2).unwrap();
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("n", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let spatial =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 2);
        let row = |sharding: Option<Sharding>| {
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(2)]))
                .with_sharding(sharding)
                .unwrap()
        };

        // Identical spatial placement is preserved, while an absent placement is neutral in either operand order.
        let expected =
            row(Some(spatial.clone())).with_shape(Shape::new(vec![Dimension::Static(8), Dimension::Static(2)]));
        assert_eq!(
            operation.infer_output_types(&[row(Some(spatial.clone())), row(Some(spatial.clone()))], &[]),
            Ok(vec![expected.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(&[row(None), row(Some(spatial.clone()))], &[]),
            Ok(vec![expected.clone()]),
        );
        assert_eq!(operation.infer_output_types(&[row(Some(spatial.clone())), row(None)], &[]), Ok(vec![expected]),);
        assert_eq!(operation.infer_output_types(&[row(None), row(None)], &[]).unwrap()[0].sharding(), None);

        // Explicit, manual, and automatic placement differences all conflict; this is intentionally stricter than
        // checking only explicit mesh axes.
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual, MeshAxisType::Auto] {
            let placement_mesh = LogicalMesh::new(vec![MeshAxis::new("p", 2, axis_type).unwrap()]).unwrap();
            let placed = Sharding::new(
                placement_mesh.clone(),
                vec![ShardingDimension::sharded(["p"]), ShardingDimension::replicated()],
            )
            .unwrap();
            let unplaced = Sharding::replicated(placement_mesh, 2);
            for operands in [
                [row(Some(placed.clone())), row(Some(unplaced.clone()))],
                [row(Some(unplaced.clone())), row(Some(placed.clone()))],
            ] {
                assert!(matches!(
                    operation.infer_output_types(&operands, &[]),
                    Err(TypeError::Invalid { message }) if message.starts_with(
                        "'concatenate' operands must be sharded identically, but got ",
                    )
                ));
            }
        }

        // Empty reduction state is neutral, and each nonempty state propagates independently in either order.
        for (with_state, expected_unreduced, expected_reduced) in [
            (
                replicated.clone().with_unreduced_axes(["m"]).unwrap(),
                BTreeSet::from(["m".to_string()]),
                BTreeSet::new(),
            ),
            (replicated.clone().with_reduced_axes(["n"]).unwrap(), BTreeSet::new(), BTreeSet::from(["n".to_string()])),
        ] {
            for operands in [
                [row(Some(replicated.clone())), row(Some(with_state.clone()))],
                [row(Some(with_state.clone())), row(Some(replicated.clone()))],
            ] {
                let output = operation.infer_output_types(&operands, &[]).unwrap();
                let output_sharding = output[0].sharding().unwrap();
                assert_eq!(output_sharding.unreduced_axes(), &expected_unreduced);
                assert_eq!(output_sharding.reduced_axes(), &expected_reduced);
            }
        }

        // Distinct nonempty reduction states conflict independently and without operand-order dependence.
        for (left, right, message) in [
            (
                replicated.clone().with_unreduced_axes(["m"]).unwrap(),
                replicated.clone().with_unreduced_axes(["n"]).unwrap(),
                "'concatenate' operands must be unreduced over the same nonempty axis set",
            ),
            (
                replicated.clone().with_reduced_axes(["m"]).unwrap(),
                replicated.clone().with_reduced_axes(["n"]).unwrap(),
                "'concatenate' operands must be reduced over the same nonempty axis set",
            ),
        ] {
            for operands in [
                [row(Some(left.clone())), row(Some(right.clone()))],
                [row(Some(right.clone())), row(Some(left.clone()))],
            ] {
                assert_eq!(operation.infer_output_types(&operands, &[]), Err(TypeError::invalid(message)));
            }
        }
        assert_eq!(
            operation.infer_output_types(
                &[
                    row(Some(replicated.clone().with_unreduced_axes(["m"]).unwrap())),
                    row(Some(replicated.clone().with_reduced_axes(["m"]).unwrap())),
                ],
                &[],
            ),
            Err(TypeError::invalid("mesh axis name 'm' appears more than once")),
        );

        // The public capability performs the standard varying normalization before applying the primitive rule.
        let varying = replicated.clone().with_varying_manual_axes(["m"]).unwrap();
        for operands in [
            [row(Some(varying.clone())), row(Some(replicated.clone()))],
            [row(Some(replicated.clone())), row(Some(varying.clone()))],
            [row(Some(varying.clone())), row(None)],
        ] {
            let output = operation.infer_output_types(&operands, &[]).unwrap();
            assert_eq!(output[0].sharding().unwrap().varying_manual_axes(), varying.varying_manual_axes());
        }
        let varying_n = replicated.clone().with_varying_manual_axes(["n"]).unwrap();
        let output = operation.infer_output_types(&[row(Some(varying.clone())), row(Some(varying_n))], &[]).unwrap();
        assert_eq!(
            output[0].sharding().unwrap().varying_manual_axes(),
            &BTreeSet::from(["m".to_string(), "n".to_string()]),
        );

        // When standard normalization makes an operand varying over its complete reduced set, that reduced state is
        // consumed. A partial overlap is invalid because it cannot be represented by the standard cast.
        for operands in [
            [row(Some(replicated.clone().with_reduced_axes(["m"]).unwrap())), row(Some(varying.clone()))],
            [row(Some(varying.clone())), row(Some(replicated.clone().with_reduced_axes(["m"]).unwrap()))],
        ] {
            let output = operation.infer_output_types(&operands, &[]).unwrap();
            let output_sharding = output[0].sharding().unwrap();
            assert!(output_sharding.reduced_axes().is_empty());
            assert_eq!(output_sharding.varying_manual_axes(), varying.varying_manual_axes());
        }
        assert!(matches!(
            operation.infer_output_types(
                &[
                    row(Some(replicated.with_reduced_axes(["m", "n"]).unwrap())),
                    row(Some(varying)),
                ],
                &[],
            ),
            Err(TypeError::Invalid { message }) if message.starts_with(
                "'concatenate' cannot make operand varying over axes",
            )
        ));

        // Batching preserves explicit and manual placement on the mapped physical axis while concatenating each
        // logical batch item independently.
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let physical_sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            let physical_type =
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
                    .with_sharding(physical_sharding.clone())
                    .unwrap();
            let replicated_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)]))
                .with_sharding(Sharding::replicated(mesh, 1))
                .unwrap();
            let expected_type =
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                    .with_sharding(physical_sharding)
                    .unwrap();
            check_operation_batching!(
                @exact,
                context = EagerContext::<Array, ArrayOperation<Array>>::new(),
                driver = &EmptyRegionDriver,
                operation = ConcatenateOperation::new(0, 1).unwrap(),
                axis_size = 2,
                axis_sharding = ShardingDimension::sharded(["x"]),
                cases = [{
                    inputs = [
                        (@mapped(axis = 0), Array::from_f64s(
                            physical_type,
                            vec![1.0, 2.0, 3.0, 4.0],
                        )),
                        (@replicated, Array::from_f64s(replicated_type, vec![5.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::from_f64s(
                        expected_type,
                        vec![1.0, 2.0, 5.0, 3.0, 4.0, 5.0],
                    ))],
                }],
            );
        }
    }

    #[test]
    fn test_array_type_concatenate() {
        // The type-level capability accepts negative axes and preserves a sole input exactly.
        let left = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let right = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let expected = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]));
        assert_eq!(ArrayType::concatenate([&left, &right], -1), Ok(expected));

        let placed = left
            .with_layout(Layout::Strided(StridedLayout::new(vec![8, 16])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(ArrayType::concatenate([&placed], 2), Ok(placed.clone()));

        // Multiple inputs preserve their common memory space but clear physical layout metadata.
        let host_row = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![16, 8])))
            .with_memory(Memory::Host { pinned: true });
        let host_output = ArrayType::concatenate([&host_row, &host_row], 0).unwrap();
        assert_eq!(host_output.memory(), Memory::Host { pinned: true });
        assert_eq!(host_output.layout(), None);
        assert_eq!(
            ArrayType::concatenate(
                [
                    &host_row,
                    &ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)]))
                ],
                0,
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "'concatenate' operands must share one memory space but operand 1 resides in Device and \
                    operand 0 resides in Host[Pinned]"
                    .to_string(),
            ))),
        );

        // Invalid signed axes and output-size overflow report exact type errors.
        assert_eq!(
            ArrayType::concatenate(
                [
                    &ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)])),
                    &ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
                ],
                -3,
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "'concatenate' axis -3 is out of bounds for operands of rank 2".to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::concatenate(
                [
                    &ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(usize::MAX)])),
                    &ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])),
                ],
                0,
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "'concatenate' output size overflows usize on axis 0".to_string(),
            ))),
        );
    }
}
