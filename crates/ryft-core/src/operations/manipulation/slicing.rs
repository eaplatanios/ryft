use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy,
    ArrayIrType, ArrayIrValue, ArraySliceAxis, ArrayType, ArrayTypeRefinements, DataType, Dimension, DimensionType,
    DimensionValue, LinearResiduals, ReferenceSliceOperation, Shape, Sharding, ShardingDimension, StaticShape,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, ProjectedContext, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, ElementwiseDerivativeAlignment,
    MemberDifferentiableOperation, MemberTransposableOperation, TransposableOperation, TranspositionContext,
    TranspositionDriver, jvp_projected_operation, transpose_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation, impl_reference_dischargeable_operation};
use crate::operations::compare::Compare;
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::constants::iota::DynamicIota;
use crate::operations::constants::one_like::OneLike;
use crate::operations::constants::zero::{DynamicZero, Zero, ZeroOperation};
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_min::DimensionMin;
use crate::operations::dimensions::dimension_saturating_sub::DimensionSaturatingSub;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::dimensions::dimension_to_scalar::DimensionToScalar;
use crate::operations::manipulation::broadcasting::Broadcast;
use crate::operations::manipulation::concatenation::Concatenate;
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::gathering::{
    DynamicGather, Gather, GatherDimensionNumbers, GatherMode, GatherOptions,
};
use crate::operations::manipulation::memory::TransferToMemory;
use crate::operations::manipulation::padding::PadOperation;
use crate::operations::manipulation::reshaping::{DynamicReshape, Reshape};
use crate::operations::manipulation::scattering::{
    Scatter, ScatterDimensionNumbers, ScatterMode, ScatterOptions, ScatterReductionKind,
};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::add::{Add, AddOperation};
use crate::operations::math::mul::Mul;
use crate::operations::math::reduce::{Reduce, ReductionKind};
use crate::operations::references::{ReferenceAddUpdateOperation, ReferenceReadOperation, ReferenceWriteOperation};
use crate::operations::sharding::Reshard;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    Concretizable, EffectClass, EffectClasses, Effects, EmptyRegionDriver, MaybeZero, Operation, OperationFormatter,
    OperationProjection, OperationProvider, ProgramError, RegionInterface, Type, TypeError, Typed, Value,
    ValueProjection,
};
use crate::tracing::{NestedTracingContext, Tracer, TracingContext};

/// Canonical operation name for [`SliceOperation`].
pub const SLICE_OPERATION_NAME: &str = "slice";

/// [`Operation`] that extracts a (possibly strided) sub-array from its input using static start, limit, and stride
/// values. Refer to the documentation of [`Slice`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SliceOperation {
    /// Refer to the documentation of [`start_indices`](Self::start_indices) for more information.
    start_indices: Vec<usize>,

    /// Refer to the documentation of [`limit_indices`](Self::limit_indices) for more information.
    limit_indices: Vec<usize>,

    /// Refer to the documentation of [`strides`](Self::strides) for more information.
    strides: Vec<usize>,
}

impl SliceOperation {
    /// Creates a new [`SliceOperation`] with the provided start and limit indices and unit strides.
    /// Use [`with_strides`](Self::with_strides) to attach non-unit strides.
    #[inline]
    pub fn new(start_indices: Vec<usize>, limit_indices: Vec<usize>) -> Self {
        let strides = vec![1; start_indices.len()];
        Self { start_indices, limit_indices, strides }
    }

    /// Returns a copy of this [`SliceOperation`] with its strides set to `strides`. There must be one stride per
    /// start index, and every stride must be at least `1`; otherwise, this function returns a [`TypeError`].
    pub fn with_strides(mut self, strides: Vec<usize>) -> Result<Self, ProgramError> {
        if strides.len() != self.start_indices.len() {
            return Err(TypeError::invalid(format!(
                "`{}` `strides` has length {} but `start_indices` has length {}",
                SLICE_OPERATION_NAME,
                strides.len(),
                self.start_indices.len(),
            ))
            .into());
        }
        if let Some(axis) = strides.iter().position(|stride| *stride == 0) {
            return Err(TypeError::invalid(format!(
                "`{SLICE_OPERATION_NAME}` strides must be at least 1 but axis {axis} has stride 0",
            ))
            .into());
        }
        self.strides = strides;
        Ok(self)
    }

    /// Returns the inclusive start indices of this [`SliceOperation`], one per input axis.
    #[inline]
    pub fn start_indices(&self) -> &[usize] {
        self.start_indices.as_slice()
    }

    /// Returns the exclusive limit indices of this [`SliceOperation`], one per input axis.
    #[inline]
    pub fn limit_indices(&self) -> &[usize] {
        self.limit_indices.as_slice()
    }

    /// Returns the strides of this [`SliceOperation`], one per input axis. Every stride is at least `1`.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        self.strides.as_slice()
    }
}

impl Display for SliceOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for SliceOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        SLICE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        match input_types[0].slice(
            self.start_indices.as_slice(),
            self.limit_indices.as_slice(),
            self.strides.as_slice(),
        ) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("start_indices", format_args!("{:?}", self.start_indices))?;
            operation.field("limit_indices", format_args!("{:?}", self.limit_indices))?;
            if self.strides.iter().any(|stride| *stride != 1) {
                operation.field("strides", format_args!("{:?}", self.strides))?;
            }
            Ok(())
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free SliceOperation);

impl<C: Domain<Type = ArrayType, Value: Slice>> InterpretableOperation<C> for SliceOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].slice(
            self.start_indices.as_slice(),
            self.limit_indices.as_slice(),
            self.strides.as_slice(),
        )?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for SliceOperation where
    C::Operation: From<SliceOperation>
{
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for SliceOperation
where
    SliceOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);

        // Static or clamped windows cannot describe a changed ragged extent.
        if inputs.iter().any(|input| !input.ragged_axes().is_empty()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{SLICE_OPERATION_NAME}` does not support bounded ragged array inputs"),
            }
            .into());
        }

        // A batched input keeps its batch axis by slicing it fully, so the lifted operation inserts start index `0`,
        // limit `axis_size`, and stride `1` at the batch axis position.
        match inputs[0].batch_axis_position() {
            None => Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into()),
            Some(batch_axis) => {
                let axis_size =
                    ArrayBatch::common_batch_size(inputs)?.ok_or_else(|| ProgramError::UnsupportedOperation {
                        message: format!("`{SLICE_OPERATION_NAME}` batching requires a statically known mapped extent"),
                    })?;
                let mut start_indices = self.start_indices().to_vec();
                start_indices.insert(batch_axis, 0);
                let mut limit_indices = self.limit_indices().to_vec();
                limit_indices.insert(batch_axis, axis_size);
                let mut strides = self.strides().to_vec();
                strides.insert(batch_axis, 1);
                let lifted = SliceOperation::new(start_indices, limit_indices).with_strides(strides)?;
                Ok(lifted.interpret_with_batch_axes(context, inputs, &[BatchAxis::from_position(batch_axis)])?.into())
            }
        }
    }
}

impl_differentiable_operation! {
    SliceOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Value: Slice,
        C::Operation: From<SliceOperation>,
    {
        |operation, _context, _driver, inputs| {
            // Slicing is a linear map, and so the primal output is the slice of the input primal and the tangent is
            // the same slice of the input tangent. A zero input tangent yields a typed zero output tangent.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().slice(
                operation.start_indices(),
                operation.limit_indices(),
                operation.strides(),
            )?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.slice(
                    operation.start_indices(),
                    operation.limit_indices(),
                    operation.strides(),
                )?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType>
            + From<UpdateSliceOperation>
            + From<PadOperation<ArrayType>>
            + From<ZeroOperation<ArrayType>>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // This homogeneous rule requires a statically shaped input on both strategies. Each writes into a zero
            // of the input's cotangent type (or reconstructs its extents), and the homogeneous `ArrayType` operation
            // family owns no first-class dimension operations, so it has no constructor that an supply a runtime
            // extent. A dynamically shaped input is therefore rejected here with an exact diagnostic. Mixed
            // `ArrayIrType` programs are unaffected as the `MemberDifferentiableOperation` rule routes a dynamically
            // shaped slice into a residual-carrying `LinearCallOperation` whose transpose region rebuilds the same zero
            // from the retained exact extents.
            //
            // The forward map extracts a (possibly strided) block, so its pullback scatters the output cotangent back
            // into the positions the forward map read, with the strategy split on the strides:
            //
            //   - **Unit Strides:** Read a contiguous block, so the pullback writes the cotangent into a zero array of
            //     the input type at the same static offsets: `cotangent ↦ update_slice(zeros(input_type), cotangent,
            //     start_indices)`.
            //   - **Non-Unit Strides:** Read every `strides[d]`-th element, so the pullback pads the cotangent with a
            //     zero scalar at exactly the inverse geometry: `edge_padding_low[d] = start_indices[d]`,
            //     `interior_padding[d] = strides[d] - 1`, and `edge_padding_high[d]` covers the rest of the input
            //     extent (i.e., everything after the last element the forward slice covered). For example, slicing
            //     `[0..6)` with `start = 1` and `stride = 2` reads positions `1`, `3`, and `5`, and the pullback pads
            //     the cotangent of length `3` with `low = 1`, `interior = 1`, and `high = 0`, scattering its elements
            //     back to positions `1`, `3`, and `5` of a zero-filled length-`6` array.
            //
            // Symbolic-zero cotangents propagate unchanged.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(()),
                MaybeZero::Value(_) if !accumulators[0].is_needed() => Ok(()),
                MaybeZero::Value(cotangent) if operation.strides().iter().all(|stride| *stride == 1) => {
                    // Only the nullary zero is available in the homogeneous family, so enforce this rule's static shape
                    // contract explicitly, matching the strided strategy's own check below.
                    let input_cotangent_type = inputs[0].r#type().cotangent()?;
                    if input_cotangent_type.static_shape().is_none() {
                        return Err(TypeError::invalid(format!(
                            "`{SLICE_OPERATION_NAME}` transpose requires a static input shape but got \
                             `{input_cotangent_type}`",
                        ))
                        .into());
                    }
                    let zeros = MaybeZero::Zero(input_cotangent_type).materialize(&**context)?;
                    let outputs = context.stage_operation(
                        UpdateSliceOperation::new(operation.start_indices().to_vec()),
                        Vec::new(),
                        &[zeros, cotangent.clone()],
                    )?;
                    check_count!("output", outputs, 1, ProgramError);
                    let cotangent =
                        outputs.into_iter().next().unwrap().unalign_cotangent(&inputs[0].r#type().cotangent()?)?;
                    accumulators[0].accumulate(context, MaybeZero::Value(cotangent))
                }
                MaybeZero::Value(cotangent) => {
                    let input_type = inputs[0].r#type();
                    let mut edge_padding_low = Vec::with_capacity(input_type.rank());
                    let mut edge_padding_high = Vec::with_capacity(input_type.rank());
                    let mut interior_padding = Vec::with_capacity(input_type.rank());
                    for (axis, ((&start, &limit), &stride)) in operation
                        .start_indices()
                        .iter()
                        .zip(operation.limit_indices())
                        .zip(operation.strides())
                        .enumerate()
                    {
                        let dimension = input_type.dimension(axis);
                        let Some(input_size) = dimension.value() else {
                            return Err(TypeError::invalid(format!(
                                "`{SLICE_OPERATION_NAME}` transpose requires a static input shape but axis {axis} has \
                                 size {dimension}",
                            ))
                            .into());
                        };
                        let output_size = (limit - start).div_ceil(stride);

                        // The forward slice covered positions `start + i * stride` for `i < output_size`. Everything
                        // after the last covered position becomes high edge padding. An empty slice covered nothing,
                        // so the pullback is pure edge padding around zero interior elements.
                        let high = match output_size {
                            0 => input_size - start,
                            size => input_size - (start + (size - 1) * stride) - 1,
                        };
                        edge_padding_low.push(i64::try_from(start).map_err(|_| {
                            TypeError::invalid(format!(
                                "`{SLICE_OPERATION_NAME}` transpose start index is too large on axis {axis}",
                            ))
                        })?);
                        edge_padding_high.push(i64::try_from(high).map_err(|_| {
                            TypeError::invalid(format!(
                                "`{SLICE_OPERATION_NAME}` transpose high padding is too large on axis {axis}",
                            ))
                        })?);
                        interior_padding.push(stride - 1);
                    }
                    let zero =
                        MaybeZero::Zero(cotangent.r#type().scalar_like()?).materialize(&**context)?;
                    let outputs = context.stage_operation(
                        PadOperation::<ArrayType>::new(edge_padding_low, edge_padding_high, interior_padding)?,
                        Vec::new(),
                        &[cotangent.clone(), zero],
                    )?;
                    check_count!("output", outputs, 1, ProgramError);
                    let cotangent =
                        outputs.into_iter().next().unwrap().unalign_cotangent(&inputs[0].r#type().cotangent()?)?;
                    accumulators[0].accumulate(context, MaybeZero::Value(cotangent))
                }
            }
        }
    },
}

impl<C: Context<Type = ArrayIrType>> MemberDifferentiableOperation<C> for SliceOperation
where
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + OperationProjection<
            ArrayType,
            Projected: DifferentiableOperation<ProjectedContext<C, ArrayType>>
                           + From<PadOperation<ArrayType>>
                           + From<SliceOperation>
                           + From<UpdateSliceOperation>
                           + From<ZeroOperation<ArrayType>>,
        >,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // A dynamically shaped input retains its exact extents as ordinary residual values.
        // A static input delegates to the homogeneous projected rule.
        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };
        let input_type = <&ArrayType>::try_from(input.primal().r#type().as_ref())?.clone();
        if input_type.shape().dimensions().iter().all(|dimension| matches!(dimension, Dimension::Static(_))) {
            let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
            return jvp_projected_operation(context, &operation, inputs);
        }

        let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
        let mut primal_outputs = context.primal().bind(operation, Vec::new(), std::slice::from_ref(input.primal()))?;
        check_count!("output", primal_outputs, 1, ProgramError);
        let output_primal = primal_outputs.remove(0);
        let tangent_primal = context.primal_to_tangent(output_primal.clone())?;
        let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
        let input = &tangent_inputs[0];
        let tangent_context = context.tangent();
        let tangent = match input.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(tangent_primal.r#type().tangent()?),
            MaybeZero::Value(input_tangent) => {
                let mut residuals = LinearResiduals::new();
                let input_shape = residuals.retain_shape(tangent_context, input.primal())?;
                let forward_operation = self.clone();
                let transpose_shape = input_shape.clone();
                let transpose_input_type = input_type.cotangent()?;
                let transpose_starts = self.start_indices().to_vec();
                let transpose_strides = self.strides().to_vec();
                let tangent = LinearCallOperation::stage(
                    tangent_context,
                    residuals.into_values(),
                    vec![input_tangent.clone()],
                    move |_, linear_inputs| {
                        linear_inputs[0].dispatch_domain().bind(
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(forward_operation),
                            Vec::new(),
                            std::slice::from_ref(&linear_inputs[0]),
                        )
                    },
                    move |residuals, output_cotangents| {
                        let transpose_context = output_cotangents[0].dispatch_domain();
                        let mut output_cotangent = output_cotangents[0].clone();
                        let zero_extents = transpose_shape.dynamic_dimensions(residuals);
                        let zeros = transpose_context
                            .bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(ZeroOperation::new(
                                    transpose_input_type.clone(),
                                )),
                                Vec::new(),
                                zero_extents.as_slice(),
                            )?
                            .remove(0);
                        if transpose_strides.iter().any(|stride| *stride != 1) {
                            let padding_value = transpose_context
                                .bind(
                                    <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                        ZeroOperation::new(
                                            <&ArrayType>::try_from(output_cotangent.r#type().as_ref())?
                                                .scalar_like()?,
                                        ),
                                    ),
                                    Vec::new(),
                                    &[],
                                )?
                                .remove(0);
                            output_cotangent = transpose_context
                                .bind(
                                    <C::Operation as OperationProjection<ArrayType>>::Projected::from(PadOperation::<
                                        ArrayType,
                                    >::new(
                                        vec![0; transpose_input_type.rank()],
                                        vec![0; transpose_input_type.rank()],
                                        transpose_strides.iter().map(|stride| stride - 1).collect(),
                                    )?),
                                    Vec::new(),
                                    &[output_cotangent, padding_value],
                                )?
                                .remove(0);
                        }
                        transpose_context.bind(
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                UpdateSliceOperation::new(transpose_starts),
                            ),
                            Vec::new(),
                            &[zeros, output_cotangent],
                        )
                    },
                )?
                .remove(0);
                MaybeZero::Value(tangent)
            }
        };
        Ok(vec![DifferentiationDual::new(output_primal, tangent)?])
    }
}

impl<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>> MemberTransposableOperation<V, O>
    for SliceOperation
where
    V: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    O: From<AddOperation<ArrayIrType>>
        + From<ReferenceSliceOperation>
        + From<ReferenceAddUpdateOperation<ArrayType, ArrayIrType>>
        + OperationProjection<
            ArrayType,
            Projected: TransposableOperation<
                <V as ValueProjection<ArrayType>>::Projected,
                <O as OperationProjection<ArrayType>>::Projected,
            > + From<SliceOperation>,
        >,
{
    fn transpose_in_parent<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 1, DifferentiationError);

        if let Some(reference) = accumulators[0].reference(context)? {
            if let MaybeZero::Value(cotangent) = &outputs[0] {
                // A reference slice describes the same coordinates as the array slice, including empty selections and
                // non-unit strides. Updating its view adds only the selected entries, without padding a dense gradient
                // with zeros or reading and replacing the caller's whole buffer.
                let axes = self
                    .start_indices()
                    .iter()
                    .zip(self.limit_indices())
                    .zip(self.strides())
                    .map(|((&start, &limit), &stride)| {
                        ArraySliceAxis::new(start, (limit - start).div_ceil(stride), stride)
                    })
                    .collect();
                let reference = context.bind(ReferenceSliceOperation::new(axes), Vec::new(), &[reference])?.remove(0);
                context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[reference, cotangent.clone()])?;
            }
            return Ok(());
        }

        transpose_projected_operation(
            context,
            &<O as OperationProjection<ArrayType>>::Projected::from(self.clone()),
            inputs,
            outputs,
            accumulators,
        )
    }
}

/// Represents the ability to extract a (possibly strided) sub-array using static start, limit, and stride values. Its
/// semantics follow StableHLO's [`slice`](https://openxla.org/stablehlo/spec#slice) operation.
///
/// `t.slice(start_indices, limit_indices, strides)` returns the sub-array whose element at index `i` is the input
/// element at index `start_indices + i * strides`, with output dimension `ceil((limit_indices[d] - start_indices[d]) /
/// strides[d])` along each axis `d` (where an axis with `start_indices[d] == limit_indices[d]` is empty). All three
/// slices must have length equal to the input rank, and each axis must satisfy `start_indices[d] <= limit_indices[d] <=
/// input_dimension[d]` and `strides[d] >= 1`. Slicing accepts dynamic input extents when their declared lower bounds
/// prove that every limit lies in bounds. A slice covering the complete input with unit strides passes it through
/// unchanged. Any other output preserves the input memory space and clears explicit physical layout metadata.
///
/// # Example
///
/// The following example shows how to use [`Slice`] in practice:
///
/// ```rust
/// # use ryft_core::{Array, Slice, ProgramError};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Slice the last two elements of the second row of a 2x3 matrix.
/// // Shapes: input [2, 3] -> output [1, 2].
/// let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let output = input.slice(&[1, 1], &[2, 3], &[1, 1])?;
/// // `output` has shape [1, 2] with values [[5.0, 6.0]].
/// assert_eq!(output.to_f64s(), vec![5.0, 6.0]);
///
/// // A non-unit stride keeps every other element, selecting positions 1, 3, and 5.
/// // Shapes: input [6] -> output [3].
/// let input = Array::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// let output = input.slice(&[1], &[6], &[2])?;
/// assert_eq!(output.to_f64s(), vec![1.0, 3.0, 5.0]);
/// # Ok(())
/// # }
/// ```
pub trait Slice: Sized {
    /// Slices `self` between `start_indices` and `limit_indices` with `strides`. Refer to the documentation of this
    /// trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `start_indices`: Inclusive non-negative start for each input axis, no greater than its corresponding limit.
    ///   - `limit_indices`: Exclusive limit for each input axis, no greater than its guaranteed input extent. Equal
    ///     start and limit indices produce an empty output axis.
    ///   - `strides`: Strictly positive distance between selected elements along each axis. There must be one start,
    ///     limit, and stride per input axis; empty slices describe a scalar input.
    fn slice(&self, start_indices: &[usize], limit_indices: &[usize], strides: &[usize]) -> Result<Self, ProgramError>;

    /// Slices one axis, preserving every other axis in full. Negative axis numbers count from the end of the shape.
    /// Starts and limits use the same non-negative, exclusive-limit convention as [`Self::slice`]; they do not wrap.
    /// Other axes must have static extents so their complete limits can be supplied to the static operation.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Axis to slice, normalized against the input rank.
    ///   - `start`: Inclusive first selected index.
    ///   - `limit`: Exclusive upper bound, at most the selected input extent.
    ///   - `stride`: Strictly positive distance between selected elements.
    fn slice_axis<A: Into<Axis>>(
        &self,
        axis: A,
        start: usize,
        limit: usize,
        stride: usize,
    ) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let input_type = self.r#type();
        let axis = axis.into().normalize(input_type.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        let mut starts = vec![0; input_type.rank()];
        let mut limits = Vec::with_capacity(input_type.rank());
        for (index, dimension) in input_type.shape().dimensions().iter().enumerate() {
            if index == axis {
                limits.push(limit);
            } else if let Dimension::Static(size) = dimension {
                limits.push(*size);
            } else {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!("`slice_axis` requires a static extent on unsliced axis {index}"),
                });
            }
        }
        let mut strides = vec![1; input_type.rank()];
        starts[axis] = start;
        strides[axis] = stride;
        self.slice(&starts, &limits, &strides)
    }

    /// Selects one index along an axis, optionally retaining that axis as an extent-one dimension. Negative axis
    /// numbers count from the end; the selected index must be non-negative and in bounds. This composes
    /// [`Self::slice_axis`] with [`Reshape`] when `keep_axis` is `false` and has the same static-extent requirements.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Axis containing the selected index.
    ///   - `index`: Index to extract along that axis.
    ///   - `keep_axis`: Whether the selected axis remains in the result shape.
    fn index_axis<A: Into<Axis>>(&self, axis: A, index: usize, keep_axis: bool) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType> + Reshape,
    {
        let axis =
            axis.into().normalize(self.r#type().rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        let limit = index.checked_add(1).ok_or_else(|| TypeError::invalid("`index_axis` index overflows `usize`"))?;
        let output = self.slice_axis(axis, index, limit, 1)?;
        if keep_axis { Ok(output) } else { output.squeeze([axis]) }
    }
}

impl Slice for ArrayType {
    fn slice(
        &self,
        start_indices: &[usize],
        limit_indices: &[usize],
        strides: &[usize],
    ) -> Result<ArrayType, ProgramError> {
        let rank = self.rank();
        if start_indices.len() != rank {
            return Err(TypeError::invalid(format!(
                "`{}` `start_indices` has length {} but input has rank {}",
                SLICE_OPERATION_NAME,
                start_indices.len(),
                rank,
            ))
            .into());
        }

        if limit_indices.len() != rank {
            return Err(TypeError::invalid(format!(
                "`{}` `limit_indices` has length {} but input has rank {}",
                SLICE_OPERATION_NAME,
                limit_indices.len(),
                rank,
            ))
            .into());
        }

        if strides.len() != rank {
            return Err(TypeError::invalid(format!(
                "`{}` `strides` has length {} but input has rank {}",
                SLICE_OPERATION_NAME,
                strides.len(),
                rank,
            ))
            .into());
        }

        let mut output_dimensions = Vec::with_capacity(rank);
        for (axis, ((&start, &limit), &stride)) in
            start_indices.iter().zip(limit_indices.iter()).zip(strides.iter()).enumerate()
        {
            if stride == 0 {
                return Err(TypeError::invalid(format!(
                    "`{SLICE_OPERATION_NAME}` strides must be at least 1 but axis {axis} has stride 0",
                ))
                .into());
            }

            if start > limit {
                return Err(TypeError::invalid(format!(
                    "`{SLICE_OPERATION_NAME}` start index {start} is greater than limit index {limit} at axis {axis}",
                ))
                .into());
            }

            match self.dimension(axis) {
                Dimension::Static(size) if limit > size => {
                    return Err(TypeError::invalid(format!(
                        "`{SLICE_OPERATION_NAME}` limit index {limit} is out of bounds \
                         for axis {axis} with size {size}",
                    ))
                    .into());
                }
                Dimension::Dynamic(variable) if limit > variable.bounds().lower() => {
                    return Err(TypeError::invalid(format!(
                        "`{}` limit index {} exceeds the guaranteed minimum extent {} of dynamic axis {}",
                        SLICE_OPERATION_NAME,
                        limit,
                        variable.bounds().lower(),
                        axis,
                    ))
                    .into());
                }
                _ => {}
            }

            output_dimensions.push(Dimension::Static((limit - start).div_ceil(stride)));
        }

        if output_dimensions.as_slice() == self.shape().dimensions() {
            return Ok(self.clone());
        }

        let sharding = self.resized_sharding(&output_dimensions, SLICE_OPERATION_NAME)?;
        ArrayType::new(self.data_type(), Shape::new(output_dimensions))
            .with_memory(self.memory())
            .with_sharding(sharding)
            .map_err(|error| {
                TypeError::invalid(format!("`{SLICE_OPERATION_NAME}` output type is invalid: {error}")).into()
            })
    }
}

impl Slice for Array {
    fn slice(&self, start_indices: &[usize], limit_indices: &[usize], strides: &[usize]) -> Result<Self, ProgramError> {
        let output_type = self.r#type().slice(start_indices, limit_indices, strides)?;
        let axes = start_indices
            .iter()
            .zip(limit_indices.iter())
            .zip(strides.iter())
            .map(|((start, limit), stride)| ArraySliceAxis::new(*start, (limit - start).div_ceil(*stride), *stride))
            .collect::<Vec<_>>();
        self.copy_block(output_type, &axes)
    }
}

impl<A: Slice + Value<Type = ArrayType>> Slice for ArrayIrValue<A> {
    fn slice(&self, start_indices: &[usize], limit_indices: &[usize], strides: &[usize]) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        Ok(Self::Array(input.slice(start_indices, limit_indices, strides)?))
    }
}

impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<SliceOperation>>>> Slice
    for V
{
    fn slice(&self, start_indices: &[usize], limit_indices: &[usize], strides: &[usize]) -> Result<Self, ProgramError> {
        // Any context-carrying value slices by binding a `SliceOperation` through its own context. The
        // `From<SliceOperation>` bound makes this disjoint from the eager value types (whose context operation
        // is `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete
        // implementations.
        let output_type = self.r#type().slice(start_indices, limit_indices, strides)?;
        if output_type.eq(self.r#type().as_ref()) {
            return Ok(self.clone());
        }
        let operation =
            SliceOperation::new(start_indices.to_vec(), limit_indices.to_vec()).with_strides(strides.to_vec())?;
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), std::slice::from_ref(self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Canonical operation name for [`UpdateSliceOperation`].
pub const UPDATE_SLICE_OPERATION_NAME: &str = "update_slice";

/// [`Operation`] that overwrites a contiguous sub-array of its first input with its second input at static start
/// indices. Refer to the documentation of [`UpdateSlice`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UpdateSliceOperation {
    /// Refer to the documentation of [`start_indices`](Self::start_indices) for more information.
    start_indices: Vec<usize>,
}

impl UpdateSliceOperation {
    /// Creates a new [`UpdateSliceOperation`] with the provided start indices.
    #[inline]
    pub fn new(start_indices: Vec<usize>) -> Self {
        Self { start_indices }
    }

    /// Returns the inclusive start indices at which this [`UpdateSliceOperation`] writes the update,
    /// one per input axis.
    #[inline]
    pub fn start_indices(&self) -> &[usize] {
        self.start_indices.as_slice()
    }
}

impl Display for UpdateSliceOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for UpdateSliceOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        UPDATE_SLICE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        check_count!("input", input_types, 2, TypeError);
        match input_types[0].update_slice(&input_types[1], self.start_indices.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("start_indices", format_args!("{:?}", self.start_indices)))
    }
}

impl_reference_dischargeable_operation!(@reference_free UpdateSliceOperation);

impl<C: Domain<Type = ArrayType, Value: UpdateSlice>> InterpretableOperation<C> for UpdateSliceOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].update_slice(&inputs[1], self.start_indices.as_slice())?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for UpdateSliceOperation where
    C::Operation: From<UpdateSliceOperation>
{
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for UpdateSliceOperation
where
    C::Value: Broadcast + Transpose,
    UpdateSliceOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // The input and update inputs are aligned on one physical batch axis (replicated inputs are broadcast to gain
        // it), and the lifted operation inserts start index `0` at that axis so each batch item updates its own block.
        // Static or clamped windows cannot describe a changed ragged extent.
        if inputs.iter().any(|input| !input.ragged_axes().is_empty()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{UPDATE_SLICE_OPERATION_NAME}` does not support bounded ragged array inputs"),
            }
            .into());
        }
        check_count!("input", inputs, 2, ProgramError);
        let Some(batch_axis) = inputs.iter().find_map(ArrayBatch::batch_axis_position) else {
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into());
        };
        let input = P::match_axis(context, &inputs[0], Axis::from(batch_axis))?;
        let update = P::match_axis(context, &inputs[1], Axis::from(batch_axis))?;
        let mut start_indices = self.start_indices().to_vec();
        start_indices.insert(batch_axis, 0);
        Ok(UpdateSliceOperation::new(start_indices)
            .interpret_with_batch_axes(context, &[input, update], &[BatchAxis::from_position(batch_axis)])?
            .into())
    }
}

impl_differentiable_operation! {
    UpdateSliceOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType> + Zero<C::Value>,
        C::Value: UpdateSlice,
        C::Operation: From<UpdateSliceOperation>,
    {
        |operation, context, _driver, inputs| {
            // The operation is jointly linear in its input and update, so the tangent updates the input tangent with
            // the update tangent at the same static start indices. A zero input and update tangent yields a typed zero
            // output tangent.
            check_count!("input", inputs, 2, ProgramError);
            let input = &inputs[0];
            let update = &inputs[1];
            let primal = input.primal().update_slice(update.primal(), operation.start_indices())?;
            let tangent = if input.tangent().is_zero() && update.tangent().is_zero() {
                MaybeZero::Zero(primal.r#type().tangent()?)
            } else {
                let input_tangent = input.tangent().clone().materialize(context.tangent())?;
                let update_tangent = update.tangent().clone().materialize(context.tangent())?;
                MaybeZero::Value(input_tangent.update_slice(&update_tangent, operation.start_indices())?)
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType>
            + From<SliceOperation>
            + From<UpdateSliceOperation>
            + From<ZeroOperation<ArrayType>>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // The forward map overwrites a block of the input with the update, so its pullback splits the output
            // cotangent into two contributions: (1) the input cotangent is the cotangent with the update window zeroed
            // (i.e., `update_slice(cotangent, zeros(update_type), start_indices)`), and (2) the update cotangent is the
            // static slice of the cotangent at the update window (i.e., `slice(cotangent, start_indices, start_indices
            // + update_shape)`). A structural-zero output cotangent contributes nothing as untouched accumulators
            // default to structural zeros when the transposition context collects its cotangents.
            check_count!("input", inputs, 2, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 2, DifferentiationError);

            let MaybeZero::Value(cotangent) = &outputs[0] else {
                return Ok(());
            };

            if !accumulators[0].is_needed() && !accumulators[1].is_needed() {
                return Ok(());
            }

            // Both contributions need the update's static shape as the input cotangent zeroes a window of that shape
            // and the update cotangent slices exactly that window.
            let update_type = inputs[1].r#type();
            let update_sizes = update_type
                .shape()
                .dimensions()
                .iter()
                .enumerate()
                .map(|(axis, size)| {
                    size.value().ok_or_else(|| TypeError::invalid(format!(
                        "`{UPDATE_SLICE_OPERATION_NAME}` transpose requires a static update shape \
                         but axis {axis} has size {size}",
                    )))
                })
                .collect::<Result<Vec<_>, TypeError>>()?;

            if accumulators[0].is_needed() {
                let zeros = MaybeZero::Zero(update_type.cotangent()?).materialize(&**context)?;
                let input_cotangents = context.stage_operation(
                    UpdateSliceOperation::new(operation.start_indices().to_vec()),
                    Vec::new(),
                    &[cotangent.clone(), zeros],
                )?;
                check_count!("output", input_cotangents, 1, ProgramError);
                accumulators[0].accumulate(context, MaybeZero::Value(input_cotangents.into_iter().next().unwrap()))?;
            }

            if accumulators[1].is_needed() {
                let limit_indices = operation
                    .start_indices()
                    .iter()
                    .zip(update_sizes.iter())
                    .map(|(start, size)| start + size)
                    .collect::<Vec<_>>();
                let update_cotangents = context.stage_operation(
                    SliceOperation::new(operation.start_indices().to_vec(), limit_indices),
                    Vec::new(),
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", update_cotangents, 1, ProgramError);
                let update_cotangent =
                    update_cotangents.into_iter().next().unwrap().unalign_cotangent(&update_type.cotangent()?)?;
                accumulators[1].accumulate(context, MaybeZero::Value(update_cotangent))?;
            }

            Ok(())
        }
    },
}

impl<C: Context<Type = ArrayIrType>> MemberDifferentiableOperation<C> for UpdateSliceOperation
where
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + OperationProjection<
            ArrayType,
            Projected: DifferentiableOperation<ProjectedContext<C, ArrayType>>
                           + From<SliceOperation>
                           + From<UpdateSliceOperation>
                           + From<ZeroOperation<ArrayType>>,
        >,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let input_type = <&ArrayType>::try_from(inputs[0].primal().r#type().as_ref())?.clone();
        let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
        if input_type.static_shape().is_some() || !inputs[0].tangent().is_zero() || inputs[1].tangent().is_zero() {
            return jvp_projected_operation(context, &operation, inputs);
        }

        // Only the update tangent is live. For an input of shape `[n]`, its zero tangent still needs the runtime `n` as
        // the type alone cannot allocate it. Validate the primal first, then retain just its dynamic dimensions in a
        // linear call. The pullback extracts the updated window and does not need to retain the input's array data.
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut primals = context.primal().bind(operation, Vec::new(), &primal_inputs)?;
        check_count!("output", primals, 1, ProgramError);

        let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
        let tangent_context = context.tangent();
        let mut residuals = LinearResiduals::new();
        let input_shape = residuals.retain_shape(tangent_context, tangent_inputs[0].primal())?;
        let update_tangent = tangent_inputs[1].tangent().as_value().unwrap();
        let update_type = tangent_inputs[1].primal().r#type();
        let update_sizes = <&ArrayType>::try_from(update_type.as_ref())?
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .map(|(axis, size)| {
                size.value().ok_or_else(|| {
                    TypeError::invalid(format!(
                        "`{UPDATE_SLICE_OPERATION_NAME}` transpose requires a static update shape \
                         but axis {axis} has size {size}",
                    ))
                })
            })
            .collect::<Result<Vec<_>, TypeError>>()?;

        let limit_indices = self.start_indices.iter().zip(update_sizes).map(|(start, size)| start + size).collect();
        let transpose_operation = SliceOperation::new(self.start_indices.clone(), limit_indices);
        let forward_operation = self.clone();

        let tangent_type = input_type.tangent()?;
        let mut tangents = LinearCallOperation::stage(
            tangent_context,
            residuals.into_values(),
            vec![update_tangent.clone()],
            move |residuals, linear_inputs| {
                let context = linear_inputs[0].dispatch_domain();
                let dimensions = input_shape.dynamic_dimensions(residuals);
                let mut zeros = context.bind(
                    <C::Operation as OperationProjection<ArrayType>>::Projected::from(ZeroOperation::new(tangent_type)),
                    Vec::new(),
                    &dimensions,
                )?;
                check_count!("output", zeros, 1, ProgramError);
                context.bind(
                    <C::Operation as OperationProjection<ArrayType>>::Projected::from(forward_operation),
                    Vec::new(),
                    &[zeros.remove(0), linear_inputs[0].clone()],
                )
            },
            move |_, output_cotangents| {
                output_cotangents[0].dispatch_domain().bind(
                    <C::Operation as OperationProjection<ArrayType>>::Projected::from(transpose_operation),
                    Vec::new(),
                    output_cotangents,
                )
            },
        )?;

        check_count!("output", tangents, 1, ProgramError);
        Ok(vec![DifferentiationDual::new(primals.remove(0), MaybeZero::Value(tangents.remove(0)))?])
    }
}

/// Represents the ability to overwrite a contiguous sub-array with an update value at static start indices. This is
/// the statically indexed sibling of [`DynamicUpdateSlice`] and the transpose partner of [`Slice`]: writing a cotangent
/// block into a zero array at the slice offsets is exactly an update-slice of a zero input.
///
/// `input.update_slice(update, start_indices)` returns a value equal to `input` except that the block starting at
/// `start_indices` is replaced by `update`. The update must have the same element data type and rank as the input,
/// with static dimensions. Every axis must satisfy `start_indices[d] + update_dimension[d] <= input_dimension[d]`.
/// A dynamic input axis is accepted when its declared lower bound proves that the update fits. Starts are checked
/// during type inference and execution; they are never clamped.
///
/// Input and update must share a memory space and reduction state, and their shardings must agree on explicit mesh
/// axes. The output preserves the input shape, physical layout, and memory placement. Variation over manual mesh axes
/// from the update is included in the result because the written block may vary even when the input is invariant.
///
/// # Example
///
/// The following example shows how to use [`UpdateSlice`] in practice:
///
/// ```rust
/// # use ryft_core::{Array, UpdateSlice, ProgramError};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Overwrite the last two elements of the first row of a 2x3 matrix.
/// // Shapes: input [2, 3], update [1, 2] -> output [2, 3].
/// let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let update = Array::matrix(1, 2, vec![8.0, 9.0]).unwrap();
/// let output = input.update_slice(&update, &[0, 1])?;
/// assert_eq!(output.to_f64s(), vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait UpdateSlice: Sized {
    /// Overwrites the block of `self` starting at `start_indices` with `update`. Refer to the documentation of this
    /// trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `update`: Value written into `self`. Must have the same data type and rank as `self`, static dimensions,
    ///     and fit within `self` at the provided start indices.
    ///   - `start_indices`: Inclusive start index for each input axis at which `update` is written.
    fn update_slice(&self, update: &Self, start_indices: &[usize]) -> Result<Self, ProgramError>;
}

impl UpdateSlice for ArrayType {
    fn update_slice(&self, update: &Self, start_indices: &[usize]) -> Result<ArrayType, ProgramError> {
        validate_update_compatibility(UPDATE_SLICE_OPERATION_NAME, self, update)?;

        let rank = self.rank();
        if start_indices.len() != rank {
            return Err(TypeError::invalid(format!(
                "`{}` `start_indices` has length {} but input has rank {}",
                UPDATE_SLICE_OPERATION_NAME,
                start_indices.len(),
                rank,
            ))
            .into());
        }

        for (axis, &start) in start_indices.iter().enumerate() {
            let update_dimension = update.dimension(axis);
            let Dimension::Static(update_size) = update_dimension else {
                return Err(TypeError::invalid(format!(
                    "`{UPDATE_SLICE_OPERATION_NAME}` does not support dynamic update axis {axis} with size \
                     {update_dimension}; update shapes must be static",
                ))
                .into());
            };

            let limit = start.checked_add(update_size).ok_or_else(|| {
                TypeError::invalid(format!(
                    "`{UPDATE_SLICE_OPERATION_NAME}` update limit overflows `usize` on axis {axis}",
                ))
            })?;

            match self.dimension(axis) {
                Dimension::Static(input_size) if limit > input_size => {
                    return Err(TypeError::invalid(format!(
                        "`{UPDATE_SLICE_OPERATION_NAME}` update axis {axis} with start index {start} and size \
                         {update_size} does not fit in input size {input_size}",
                    ))
                    .into());
                }
                Dimension::Dynamic(variable) if limit > variable.bounds().lower() => {
                    return Err(TypeError::invalid(format!(
                        "`{}` update limit {} exceeds the guaranteed minimum extent {} of dynamic axis {}",
                        UPDATE_SLICE_OPERATION_NAME,
                        limit,
                        variable.bounds().lower(),
                        axis,
                    ))
                    .into());
                }
                _ => {}
            }
        }

        // The output is distributed like the input (the update is written in place). The input's placement
        // and reduction state carry through, with the update's varying-manual axes folded in.
        let sharding = update_slice_output_sharding(self, update, UPDATE_SLICE_OPERATION_NAME)?;
        self.clone().with_sharding(sharding).map_err(|error| {
            TypeError::invalid(format!("`{UPDATE_SLICE_OPERATION_NAME}` output type is invalid: {error}")).into()
        })
    }
}

impl UpdateSlice for Array {
    fn update_slice(&self, update: &Self, start_indices: &[usize]) -> Result<Self, ProgramError> {
        // Type inference preserves the input's shape, element type, memory, and physical layout; only sharding
        // metadata can change. Apply that validated metadata without broadcasting and copying the updated bytes.
        let output_type = self.r#type().update_slice(update.r#type().as_ref(), start_indices)?;
        let output = self.clone().replace_block(update, start_indices);
        Ok(Self::new_unchecked(output_type, output.shared_storage().clone()))
    }
}

impl<A: UpdateSlice + Value<Type = ArrayType>> UpdateSlice for ArrayIrValue<A> {
    fn update_slice(&self, update: &Self, start_indices: &[usize]) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let update = <Self as ValueProjection<ArrayType>>::projected(update)?;
        Ok(Self::Array(input.update_slice(update, start_indices)?))
    }
}

impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<UpdateSliceOperation>>>>
    UpdateSlice for V
{
    fn update_slice(&self, update: &Self, start_indices: &[usize]) -> Result<Self, ProgramError> {
        // Any context-carrying value updates a slice by binding an `UpdateSliceOperation` through its own context. The
        // `From<UpdateSliceOperation>` bound makes this disjoint from the eager value types (whose context operation is
        // `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete
        // implementations.
        let mut outputs = self.dispatch_domain().bind(
            UpdateSliceOperation::new(start_indices.to_vec()),
            Vec::new(),
            &[self.clone(), update.clone()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Determines how a [`DynamicSliceOperation`] over dimension-valued starts resolves its start coordinates against the
/// input's logical extents. Dimension starts are non-negative by construction, so no negative-index policy applies to
/// that form.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DynamicSliceBounds {
    /// Clamp each start so the entire requested window fits. The window itself must fit the input.
    Clamp,

    /// Reject a requested window that extends outside the input, retaining runtime assertions when needed.
    Checked,
}

/// Canonical operation name for [`DynamicSliceOperation`].
pub const DYNAMIC_SLICE_OPERATION_NAME: &str = "dynamic_slice";

/// [`Operation`] that extracts a sub-array at runtime start indices. The [`ArrayType`] form stores its sizes
/// and accepts scalar-array starts with unit strides, counting negative signed starts from the end of their axes
/// under its [`allows_negative_indices`](DynamicSliceOperation::<ArrayType>::allows_negative_indices) policy and
/// then clamping. The [`ArrayIrType`] form accepts dimension starts and sizes, positive static strides, and a
/// [`DynamicSliceBounds`] policy. The homogeneous [`MemberTransposableOperation`] rule adds to the selected block of
/// an enclosing reference accumulator without constructing a dense zero gradient. It currently reads and replaces the
/// complete referent, which may copy storage in eager execution. Value accumulators use the ordinary projected
/// transpose rule. Refer to [`DynamicSlice`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicSliceOperation<T: Type = ArrayType> {
    /// Refer to the documentation of [`sizes`](DynamicSliceOperation::<ArrayType>::sizes) for more information.
    sizes: Vec<usize>,

    /// Refer to the documentation of [`strides`](DynamicSliceOperation::<ArrayIrType>::strides) for more information.
    strides: Vec<usize>,

    /// Refer to the documentation of [`bounds`](Self::bounds) for more information.
    bounds: DynamicSliceBounds,

    /// Refer to the documentation of [`allows_negative_indices`](Self::allows_negative_indices) for more information.
    allow_negative_indices: bool,

    /// Refer to the documentation of [`requires_runtime_assertion`](Self::requires_runtime_assertion)
    /// for more information.
    requires_runtime_assertion: bool,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> DynamicSliceOperation<T> {
    /// Returns whether starts are clamped or checked against the input's logical extents
    /// for this [`DynamicSliceOperation`].
    #[inline]
    pub fn bounds(&self) -> DynamicSliceBounds {
        self.bounds
    }

    /// Returns whether a negative signed start index counts from the end of its axis. When `true` (the default), a
    /// negative start `i` on an axis of extent `d` is replaced by `i + d` once before clamping, so `-1` names the last
    /// valid origin and a start that is still negative after that single wrap clamps to zero. When `false`, negative
    /// starts are out of bounds and clamp to zero directly. Unsigned and Boolean starts cannot be negative and are
    /// unaffected. This is a semantic switch that every execution path honors, and not just a lowering hint.
    #[inline]
    pub fn allows_negative_indices(&self) -> bool {
        self.allow_negative_indices
    }

    /// Returns whether execution must validate the mixed slice window against the input's logical extents. When `true`,
    /// this operation carries an [`EffectClass::OrderedAssertion`] (i.e., an invalid window must report an error even
    /// if its result is unused). When `false`, type inference requires a proof that every admitted window fits. The
    /// homogeneous form proves that its stored sizes fit and clamps starts, so it never needs this assertion. Mixed
    /// construction is conservative until [`with_input_types`](DynamicSliceOperation::<ArrayIrType>::with_input_types)
    /// proves that the window fits.
    #[inline]
    pub fn requires_runtime_assertion(&self) -> bool {
        self.requires_runtime_assertion
    }
}

impl DynamicSliceOperation<ArrayType> {
    /// Creates a new [`DynamicSliceOperation`] with the provided slice sizes.
    #[inline]
    pub fn new(sizes: Vec<usize>) -> Self {
        Self {
            sizes,
            strides: Vec::new(),
            bounds: DynamicSliceBounds::Clamp,
            allow_negative_indices: true,
            requires_runtime_assertion: false,
            marker: PhantomData,
        }
    }

    /// Returns a copy of this [`DynamicSliceOperation`] with its negative-index policy set to `allow_negative_indices`.
    /// Refer to the documentation of [`allows_negative_indices`](Self::allows_negative_indices) for the meaning of both
    /// settings.
    #[inline]
    pub fn with_allow_negative_indices(mut self, allow_negative_indices: bool) -> Self {
        self.allow_negative_indices = allow_negative_indices;
        self
    }

    /// Returns the size of the extracted slice along each input axis for this [`DynamicSliceOperation`].
    #[inline]
    pub fn sizes(&self) -> &[usize] {
        self.sizes.as_slice()
    }
}

impl DynamicSliceOperation<ArrayIrType> {
    /// Creates a new [`DynamicSliceOperation<ArrayIrType>`] for the provided `rank`, with unit strides and checked
    /// bounds. Its inputs are the array, `rank` dimension-valued starts, and `rank` dimension-valued sizes, in that
    /// order. Even static sizes are supplied as dimension inputs. The operation conservatively requires a runtime
    /// assertion until [`with_input_types`](Self::with_input_types) proves that every possible window fits.
    #[inline]
    pub fn from_rank(rank: usize) -> Self {
        Self {
            sizes: Vec::new(),
            strides: vec![1; rank],
            bounds: DynamicSliceBounds::Checked,
            allow_negative_indices: true,
            requires_runtime_assertion: true,
            marker: PhantomData,
        }
    }

    /// Returns a copy of this [`DynamicSliceOperation`] with its strides set to `strides`. The number of strides must
    /// match the rank supplied to [`from_rank`](Self::from_rank), and every stride must be strictly positive.
    /// Otherwise, this function returns a [`TypeError`].
    pub fn with_strides(mut self, strides: Vec<usize>) -> Result<Self, TypeError> {
        if strides.len() != self.strides.len() {
            return Err(TypeError::invalid(format!(
                "`{}` `strides` has length {} but input has rank {}",
                DYNAMIC_SLICE_OPERATION_NAME,
                strides.len(),
                self.strides.len(),
            )));
        }
        if let Some(axis) = strides.iter().position(|stride| *stride == 0) {
            return Err(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` stride must be positive on axis {axis}",
            )));
        }
        self.strides = strides;
        self.requires_runtime_assertion = true;
        Ok(self)
    }

    /// Returns a copy of this [`DynamicSliceOperation`] with its bounds policy set to `bounds`. Changing the policy
    /// discards any previous proof that execution can omit its window assertion.
    #[inline]
    pub fn with_bounds(mut self, bounds: DynamicSliceBounds) -> Self {
        self.bounds = bounds;
        self.requires_runtime_assertion = true;
        self
    }

    /// Returns a copy of this [`DynamicSliceOperation`] with its assertion requirement derived from `input_types`.
    /// Each size's upper bound and each checked start's upper bound must fit the input's minimum logical extent.
    /// Type inference revalidates a discharged assertion, so replay cannot reuse a proof with wider input bounds.
    /// For example, an input extent of `10`, start in `[0, 3)`, size in `[0, 5)`, and unit stride prove that the
    /// exclusive limit is at most `6`. An input extent that may be smaller than `6` still requires a runtime check.
    /// Under [`DynamicSliceBounds::Clamp`], only the maximum window span has to fit and starts are normalized at
    /// execution time.
    #[inline]
    pub fn with_input_types(mut self, input_types: &[ArrayIrType]) -> Result<Self, TypeError> {
        self.requires_runtime_assertion = true;
        self.infer_output_types(input_types, &[])?;
        self.requires_runtime_assertion = !self.has_proven_window(input_types)?;
        Ok(self)
    }

    /// Returns the static stride applied along each sliced axis for this [`DynamicSliceOperation`].
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns whether all admitted starts and sizes fit the input's minimum extent, without assuming correlations
    /// between independent dimension identities. Call only after the input kinds and counts have been validated.
    fn has_proven_window(&self, input_types: &[ArrayIrType]) -> Result<bool, TypeError> {
        let input = <&ArrayType>::try_from(&input_types[0])?;
        for axis in 0..self.strides.len() {
            let start = <&DimensionType>::try_from(&input_types[1 + axis])?;
            let size = <&DimensionType>::try_from(&input_types[1 + self.strides.len() + axis])?;
            let Some(maximum_size) = size.bounds().upper().and_then(|upper| upper.checked_sub(1)) else {
                return Ok(false);
            };
            let span = if maximum_size == 0 {
                Some(0)
            } else {
                (maximum_size - 1).checked_mul(self.strides[axis]).and_then(|span| span.checked_add(1))
            };
            let maximum_start = match self.bounds {
                DynamicSliceBounds::Clamp => Some(0),
                DynamicSliceBounds::Checked => start.bounds().upper().and_then(|upper| upper.checked_sub(1)),
            };
            let minimum_input = match input.dimension(axis) {
                Dimension::Static(size) => size,
                Dimension::Dynamic(variable) => variable.bounds().lower(),
            };
            if maximum_start
                .zip(span)
                .and_then(|(start, span)| start.checked_add(span))
                .is_none_or(|limit| limit > minimum_input)
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Adds the slice cotangent to a zero with the original input geometry. Every update is a point, so no scatter
    /// window dimension has to be static. `dimensions` supplies the cotangent extents, reusing retained size inputs
    /// when available instead of staging new definitions of their dimension identities. Positive strides make the
    /// logical coordinates unique. Physical padding and inactive updates remain the responsibility of the existing
    /// bounded scatter lowering, rather than becoming extra logical updates here.
    fn apply_adjoint<
        V: Value<Type = ArrayIrType, DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant + DynamicIota<V>>
            + DimensionSize
            + DimensionToScalar
            + ValueProjection<ArrayType, Projected: Add + Mul + Concatenate + Scatter + TransferToMemory>
            + ValueProjection<DimensionType, Projected: Add + Mul + DimensionMin + DimensionSaturatingSub>,
    >(
        &self,
        zeros: &V,
        cotangent: &V,
        starts: &[V],
        dimensions: &[V],
    ) -> Result<V, ProgramError> {
        let context = cotangent.dispatch_domain();
        let cotangent_type = cotangent.r#type();
        let cotangent_type = <&ArrayType>::try_from(cotangent_type.as_ref())?;
        let rank = cotangent_type.rank();
        if rank == 0 {
            return Ok(cotangent.clone());
        }

        let dynamic_dimensions = cotangent_type
            .shape()
            .dimensions()
            .iter()
            .zip(dimensions)
            .filter_map(|(dimension, value)| matches!(dimension, Dimension::Dynamic(_)).then_some(value.clone()))
            .collect::<Vec<_>>();

        // Build the trailing index-vector axis directly as reshaping symbolic coordinate arrays would add an
        // unnecessary element-count proof when this pullback is specialized or batched.
        let mut query_shape = cotangent_type.shape().dimensions().to_vec();
        query_shape.push(Dimension::Static(1));
        let query_type = ArrayType::new(DataType::I64, Shape::new(query_shape)).with_memory(cotangent_type.memory());
        let one = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(1)?)?;
        let mut coordinates = Vec::with_capacity(rank);
        for axis in 0..rank {
            let step = context.dimension_constant(self.strides[axis])?;
            let start = if self.bounds == DynamicSliceBounds::Clamp {
                // `min(size, 1)` makes the span zero for empty windows without a data-dependent branch.
                let dimension = ValueProjection::<DimensionType>::into_projected(dimensions[axis].clone())?;
                let span = dimension
                    .dimension_saturating_sub(&one)?
                    .mul(&ValueProjection::<DimensionType>::into_projected(step.clone())?)?
                    .add(&dimension.dimension_min(&one)?)?;
                let start = ValueProjection::<DimensionType>::into_projected(starts[axis].clone())?;
                let size = ValueProjection::<DimensionType>::into_projected(zeros.dimension_size(axis)?)?;
                ValueProjection::<DimensionType>::from_projected(
                    start.dimension_min(&size.dimension_saturating_sub(&span)?)?,
                )
            } else {
                starts[axis].clone()
            };
            let positions = context.dynamic_iota(&query_type, axis, &dynamic_dimensions)?;
            let start = ValueProjection::<ArrayType>::into_projected(start.to_scalar()?)?
                .transfer_to_memory(cotangent_type.memory())?;
            let mut positions = ValueProjection::<ArrayType>::into_projected(positions)?;
            if self.strides[axis] != 1 {
                let step = ValueProjection::<ArrayType>::into_projected(step.to_scalar()?)?
                    .transfer_to_memory(cotangent_type.memory())?;
                positions = positions.mul(&step)?;
            }
            coordinates.push(positions.add(&start)?);
        }
        let coordinates = Concatenate::concatenate(&coordinates, rank as i64)?;
        let dimensions = ScatterDimensionNumbers::new(Vec::new(), (0..rank).collect(), (0..rank).collect());
        let options = ScatterOptions::new().with_mode(ScatterMode::PromiseInBounds).with_unique_indices(true);
        Ok(ValueProjection::<ArrayType>::from_projected(
            ValueProjection::<ArrayType>::into_projected(zeros.clone())?.scatter(
                &coordinates,
                &ValueProjection::<ArrayType>::into_projected(cotangent.clone())?,
                &dimensions,
                ScatterReductionKind::Add,
                &options,
            )?,
        ))
    }
}

impl<T: Type> Display for DynamicSliceOperation<T>
where
    Self: Operation,
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DynamicSliceOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        DYNAMIC_SLICE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        if input_types.is_empty() {
            return Err(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` expects an array input followed by its start index inputs \
                 but got no inputs",
            )));
        }
        match input_types[0].dynamic_slice(&input_types[1..], self.sizes.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("sizes", format_args!("{:?}", self.sizes))?;
            if !self.allow_negative_indices {
                operation.field("allow_negative_indices", false)?;
            }
            Ok(())
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free DynamicSliceOperation<ArrayType>);

impl<C: Domain<Type = ArrayType, Value: DynamicSlice>> InterpretableOperation<C> for DynamicSliceOperation<ArrayType> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1 + self.sizes.len(), ProgramError);
        let (input, start_indices) = inputs.split_first().unwrap();
        Ok(vec![input.dynamic_slice_with_negative_indices(
            start_indices,
            self.sizes.as_slice(),
            self.allow_negative_indices,
        )?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DynamicSliceOperation<ArrayType> where
    C::Operation: From<DynamicSliceOperation>
{
}

impl<
    C: Context<
            Type = ArrayType,
            Value: ZeroLike
                       + Broadcast
                       + Transpose
                       + Slice
                       + Reshape
                       + Reshard
                       + Concatenate
                       + Gather
                       + Compare
                       + Add
                       + Select
                       + ConvertElementType
                       + TransferToMemory
                       + OneLike
                       + Reduce,
            Operation: From<ConstantOperation<Array>>,
        >,
    P: ArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for DynamicSliceOperation<ArrayType>
where
    DynamicSliceOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // Batched starts are packed into one gather index vector per item. Mapped sources use paired gather batch
        // dimensions so each item reads only its own source. Explicit-layout inputs retain per-item expansion, which
        // extracts each input/start tuple and stacks its slice along a leading batch axis. Both paths use existing
        // capabilities in eager and tracing contexts.
        //
        // Replicated start indices keep the structural fast path: a batched input keeps its batch axis by slicing it
        // fully, so the lifted operation inserts size `axis_size` at the batch axis position and a zero start index for
        // it, derived from an existing index input via `ZeroLike` so the inserted index carries the same scalar integer
        // type. Rank-0 inputs have no index inputs to donate a zero index, but a rank-0 dynamic slice is the identity
        // map, so the batched input passes through unchanged. Static or clamped windows cannot describe a changed
        // ragged extent.
        check_count!("input", inputs, 1 + self.sizes().len(), ProgramError);
        if inputs.iter().any(|input| !input.ragged_axes().is_empty()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{DYNAMIC_SLICE_OPERATION_NAME}` does not support bounded ragged array inputs"),
            }
            .into());
        }

        let batch_axes = inputs.iter().map(|input| input.batch_axis_position()).collect::<Vec<_>>();
        let axis_size = ArrayBatch::common_batch_size(inputs)?;
        if batch_axes[1..].iter().any(Option::is_some) {
            // Rectangular windows are one gather, independent of the number of mapped starts. Keep each integer index
            // in its original type so clipping preserves unsigned extremes. Explicit layouts retain the existing
            // expansion path, which owns their layout transformations.
            if inputs[0].r#type().layout().is_none()
                && let Some(axis_size) = axis_size
            {
                let input_types = inputs.iter().map(ArrayBatch::unbatched_type).collect::<Vec<_>>();
                let output_type = self.infer_output_types(&input_types, &[])?[0].batched(
                    0,
                    Dimension::Static(axis_size),
                    context.axis_sharding().clone(),
                )?;

                // Paired batching dimensions select source item `i` with index vector `i`, without adding item numbers
                // to the integer indices. In particular, this does not narrow `u64` starts or require a batch extent to
                // fit in the index element type. The batch axis contributes no output window dimension; the index batch
                // axis supplies that leading output dimension.
                let mapped_source = batch_axes[0].is_some();
                let source = if mapped_source {
                    P::match_axis(context, &inputs[0], Axis::from(0))?.value().clone()
                } else {
                    inputs[0].value().clone()
                };

                // The gather clamps but never wraps, so mapped signed starts are wrapped first under the negative-index
                // policy, using each sliced axis extent as a value (a constant for a static axis, and a reduction over
                // the source for a dynamic one).
                let input_type = &input_types[0];
                let indices = inputs[1..]
                    .iter()
                    .enumerate()
                    .map(|(axis, input)| {
                        let starts = P::match_axis(context, input, Axis::from(0))?.value().clone();
                        let starts = if self.allow_negative_indices && wraps_negative_start(starts.r#type().data_type())
                        {
                            let extent = match input_type.dimension(axis) {
                                Dimension::Static(extent) => context
                                    .parent()
                                    .bind(ConstantOperation::new(Array::scalar(extent as i64)?), Vec::new(), &[])?
                                    .remove(0),
                                Dimension::Dynamic(_) => {
                                    dynamic_axis_extent(&source, axis + usize::from(mapped_source))?
                                }
                            };
                            wrap_negative_starts(&starts, &extent)?
                        } else {
                            starts
                        };
                        starts
                            .reshape(Shape::new(vec![Dimension::Static(axis_size), Dimension::Static(1)]))
                            .map_err(BatchingError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let indices = C::Value::concatenate(&indices, 1)?;

                let mut sizes = self.sizes.clone();
                let dimensions = if mapped_source {
                    // Empty batches need a zero window to stay within the empty source batch axis.
                    sizes.insert(0, usize::from(axis_size != 0));
                    GatherDimensionNumbers::new(
                        (1..=self.sizes.len()).collect(),
                        Vec::new(),
                        (1..=self.sizes.len()).collect(),
                    )
                    .with_batching_dimensions(vec![(0, 0)])
                } else {
                    GatherDimensionNumbers::new(
                        (1..=self.sizes.len()).collect(),
                        Vec::new(),
                        (0..self.sizes.len()).collect(),
                    )
                };

                let options = GatherOptions::new()
                    .with_mode(GatherMode::Clip)
                    .with_output_sharding(output_type.sharding().cloned());
                let output = source.gather(&indices, &dimensions, &sizes, &options)?;
                return Ok(vec![ArrayBatch::new(output, BatchAxis::new(0))?].into());
            }

            return Ok(batch_by_item_expansion(
                context,
                DYNAMIC_SLICE_OPERATION_NAME,
                self,
                inputs,
                axis_size.ok_or_else(|| ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` batching with mapped start indices requires a statically \
                         known mapped extent",
                    ),
                })?,
            )?
            .into());
        }

        let Some(batch_axis) = batch_axes[0] else {
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into());
        };

        if self.sizes().is_empty() {
            // An empty size list is an identity only for a scalar per-item input. The shortcut bypasses the
            // parent bind, so validate the original unbatched call before returning its input unchanged.
            self.infer_output_types(&inputs.iter().map(ArrayBatch::unbatched_type).collect::<Vec<_>>(), &[])?;
            return Ok(vec![inputs[0].clone()].into());
        }

        let axis_size = axis_size.ok_or_else(|| ProgramError::UnsupportedOperation {
            message: format!("`{DYNAMIC_SLICE_OPERATION_NAME}` batching requires a statically known mapped extent"),
        })?;

        let mut sizes = self.sizes().to_vec();
        sizes.insert(batch_axis, axis_size);
        let zero_index = ArrayBatch::replicated(inputs[1].value().zero_like()?);
        let mut lifted_inputs = inputs.to_vec();
        lifted_inputs.insert(1 + batch_axis, zero_index);
        Ok(DynamicSliceOperation::new(sizes)
            .interpret_with_batch_axes(context, lifted_inputs.as_slice(), &[BatchAxis::from_position(batch_axis)])?
            .into())
    }
}

impl_differentiable_operation! {
    DynamicSliceOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Value: DynamicSlice,
        C::Operation: From<DynamicSliceOperation>,
    {
        |operation, context, _driver, inputs| {
            // `dynamic_slice` is linear in the input, and the scalar start indices are non-differentiated primal input
            // edges, so the tangent slices the input tangent at the same primal start indices. A zero input tangent
            // yields a typed zero output tangent.
            let (input, start_indices) =
                inputs.split_first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
            let primal_starts = start_indices.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
            let allow_negative_indices = operation.allows_negative_indices();
            let primal = input.primal().dynamic_slice_with_negative_indices(
                &primal_starts,
                operation.sizes(),
                allow_negative_indices,
            )?;
            let tangent = match input.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => {
                    let tangent_starts = primal_starts
                        .into_iter()
                        .map(|value| context.primal_to_tangent(value))
                        .collect::<Result<Vec<_>, _>>()?;
                    MaybeZero::Value(tangent.dynamic_slice_with_negative_indices(
                        &tangent_starts,
                        operation.sizes(),
                        allow_negative_indices,
                    )?)
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType>
            + From<ZeroOperation<ArrayType>>
            + From<DynamicUpdateSliceOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // The scalar integer start indices (i.e., inputs 1 onward) have no tangent space, so in a valid pushforward
            // they are the known inputs and the sliced input (i.e., input 0) is the linear one. The forward map
            // `t ↦ dynamic_slice(t, start_indices, sizes)` transposes by scattering the output cotangent back into a
            // zero array of the input type at the same start indices (i.e. a dynamic update-slice at those indices).
            // The transpose reads the known start indices from the pullback boundary and stages an ordinary
            // `DynamicUpdateSliceOperation`, so linearization retains the indices as regular Single Static Assignment
            // (SSA) residuals. The start indices receive structural zeros, and a zero output cotangent stays a
            // structural zero.
            //
            // **Contract:** This homogeneous rule requires a statically shaped input. The update target is a zero
            // of the input's cotangent type, and the homogeneous `ArrayType` operation family owns no first-class
            // dimension operations, so it has no constructor that can supply a runtime extent for that zero. A
            // dynamically shaped input is therefore rejected here with an exact diagnostic. Mixed `ArrayIrType`
            // programs are unaffected as the `MemberDifferentiableOperation` rule routes a dynamically shaped dynamic
            // slice into a residual-carrying `LinearCallOperation` whose transpose region rebuilds the same zero from
            // the retained exact extents.
            if inputs.is_empty() {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            }
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
            if let MaybeZero::Value(cotangent) = &outputs[0] {
                if !accumulators[0].is_needed() {
                    return Ok(());
                }
                let start_indices = inputs[1..]
                    .iter()
                    .map(|input| {
                        // Integer indices have no tangent space and must be retained as known primal inputs.
                        input.as_known().cloned().ok_or_else(|| ProgramError::InvalidArgument {
                            message: format!("`{DYNAMIC_SLICE_OPERATION_NAME}` transpose requires known start indices"),
                        })
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;

                // Only the nullary zero is available in the homogeneous family, so enforce this rule's static-shape
                // contract explicitly instead of letting a dynamic input surface the constructor's own diagnostic.
                let input_cotangent_type = inputs[0].r#type().cotangent()?;
                if input_cotangent_type.static_shape().is_none() {
                    return Err(TypeError::invalid(format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` transpose requires a statically shaped input but got \
                         `{input_cotangent_type}`",
                    ))
                    .into());
                }

                let zeros = MaybeZero::Zero(input_cotangent_type).materialize(&**context)?;
                let mut inputs = Vec::with_capacity(2 + start_indices.len());
                inputs.push(zeros);
                inputs.push(cotangent.clone());
                inputs.extend(start_indices);

                // The adjoint resolves the same raw starts, so it must share the forward negative-index policy.
                let allows_negative_indices = operation.allows_negative_indices();
                let adjoint = DynamicUpdateSliceOperation::new().with_allow_negative_indices(allows_negative_indices);
                let outputs = context.stage_operation(adjoint, Vec::new(), inputs.as_slice())?;
                check_count!("output", outputs, 1, ProgramError);
                accumulators[0].accumulate(context, MaybeZero::Value(outputs.into_iter().next().unwrap()))?;
            }
            Ok(())
        }
    },
}

impl<C: Context<Type = ArrayIrType>> MemberDifferentiableOperation<C> for DynamicSliceOperation<ArrayType>
where
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + OperationProjection<
            ArrayType,
            Projected: DifferentiableOperation<ProjectedContext<C, ArrayType>>
                           + From<DynamicSliceOperation>
                           + From<DynamicUpdateSliceOperation>
                           + From<ZeroOperation<ArrayType>>,
        >,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // A dynamically shaped input retains its exact extents and scalar start indices as ordinary residual values.
        // A static input delegates to the homogeneous projected rule.
        let (input, _) = inputs.split_first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
        let input_type = <&ArrayType>::try_from(input.primal().r#type().as_ref())?.clone();
        if input_type.shape().dimensions().iter().all(|dimension| matches!(dimension, Dimension::Static(_))) {
            let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
            return jvp_projected_operation(context, &operation, inputs);
        }

        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
        let mut primal_outputs = context.primal().bind(operation, Vec::new(), primal_inputs.as_slice())?;
        check_count!("output", primal_outputs, 1, ProgramError);
        let output_primal = primal_outputs.remove(0);
        let tangent_primal = context.primal_to_tangent(output_primal.clone())?;
        let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
        let (input, start_indices) = tangent_inputs.split_first().unwrap();
        let tangent_context = context.tangent();
        let tangent = match input.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(tangent_primal.r#type().tangent()?),
            MaybeZero::Value(input_tangent) => {
                // Start indices have zero differential spaces but remain ordinary residual Single Static Assignment
                // (SSA) values because both the forward slice and its transpose need their concrete runtime values.
                let mut residuals = LinearResiduals::new();
                let start_indices = residuals.retain_all(start_indices.iter().map(|index| index.primal().clone()));
                let input_shape = residuals.retain_shape(tangent_context, input.primal())?;
                let forward_operation = self.clone();
                let forward_start_indices = start_indices.clone();
                let transpose_shape = input_shape.clone();
                let transpose_input_type = input_type.cotangent()?;
                let transpose_operation =
                    DynamicUpdateSliceOperation::new().with_allow_negative_indices(self.allows_negative_indices());
                let tangent = LinearCallOperation::stage(
                    tangent_context,
                    residuals.into_values(),
                    vec![input_tangent.clone()],
                    move |residuals, linear_inputs| {
                        let mut slice_inputs = Vec::with_capacity(1 + forward_start_indices.len());
                        slice_inputs.push(linear_inputs[0].clone());
                        slice_inputs.extend(forward_start_indices.iter().map(|index| residuals[*index].clone()));
                        linear_inputs[0].dispatch_domain().bind(
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(forward_operation),
                            Vec::new(),
                            slice_inputs.as_slice(),
                        )
                    },
                    move |residuals, output_cotangents| {
                        let transpose_context = output_cotangents[0].dispatch_domain();
                        let zero_extents = transpose_shape.dynamic_dimensions(residuals);
                        let zeros = transpose_context
                            .bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(ZeroOperation::new(
                                    transpose_input_type.clone(),
                                )),
                                Vec::new(),
                                zero_extents.as_slice(),
                            )?
                            .remove(0);
                        let mut update_inputs = Vec::with_capacity(2 + start_indices.len());
                        update_inputs.push(zeros);
                        update_inputs.push(output_cotangents[0].clone());
                        update_inputs.extend(start_indices.iter().map(|index| residuals[*index].clone()));
                        transpose_context.bind(
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(transpose_operation),
                            Vec::new(),
                            update_inputs.as_slice(),
                        )
                    },
                )?
                .remove(0);
                MaybeZero::Value(tangent)
            }
        };
        Ok(vec![DifferentiationDual::new(output_primal, tangent)?])
    }
}

impl<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>> MemberTransposableOperation<V, O>
    for DynamicSliceOperation<ArrayType>
where
    V: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    O: From<AddOperation<ArrayIrType>>
        + From<ReferenceReadOperation<ArrayType, ArrayIrType>>
        + From<ReferenceWriteOperation<ArrayType, ArrayIrType>>
        + OperationProjection<
            ArrayType,
            Projected: TransposableOperation<
                <V as ValueProjection<ArrayType>>::Projected,
                <O as OperationProjection<ArrayType>>::Projected,
            > + From<DynamicSliceOperation>
                           + From<DynamicUpdateSliceOperation>
                           + From<AddOperation<ArrayType>>,
        >,
{
    fn transpose_in_parent<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("input", inputs, 1 + self.sizes().len(), ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 1 + self.sizes().len(), DifferentiationError);

        if let Some(reference) = accumulators[0].reference(context)? {
            if let MaybeZero::Value(cotangent) = &outputs[0] {
                // Runtime indices remain ordinary residual inputs, so this typed rule can choose buffer updates when
                // transposition runs. Add to the selected block, then replace that block in the current referent.
                // This preserves prior contributions and uses the same index clamping as the forward slice without
                // constructing a full-size zero gradient. Until dynamic reference views are supported, the final write
                // still describes an update to the complete reference and eager execution may copy its complete
                // storage.
                let start_indices = inputs[1..]
                    .iter()
                    .map(|input| {
                        // Integer indices have no tangent space and must be retained as known primal inputs.
                        input.as_known().cloned().ok_or_else(|| ProgramError::InvalidArgument {
                            message: format!("`{DYNAMIC_SLICE_OPERATION_NAME}` transpose requires known start indices"),
                        })
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                let current = context
                    .bind(ReferenceReadOperation::new(), Vec::new(), std::slice::from_ref(&reference))?
                    .remove(0);
                let mut slice_inputs = vec![current.clone()];
                slice_inputs.extend(start_indices.iter().cloned());
                let selected = context
                    .bind(
                        <O as OperationProjection<ArrayType>>::Projected::from(self.clone()),
                        Vec::new(),
                        &slice_inputs,
                    )?
                    .remove(0);
                let updated = context
                    .bind(
                        <O as OperationProjection<ArrayType>>::Projected::from(AddOperation::new()),
                        Vec::new(),
                        &[selected, cotangent.clone()],
                    )?
                    .remove(0);
                let mut update_inputs = vec![current, updated];
                update_inputs.extend(start_indices);
                let transpose_operation =
                    DynamicUpdateSliceOperation::new().with_allow_negative_indices(self.allows_negative_indices());
                let updated = context
                    .bind(
                        <O as OperationProjection<ArrayType>>::Projected::from(transpose_operation),
                        Vec::new(),
                        &update_inputs,
                    )?
                    .remove(0);
                context.bind(ReferenceWriteOperation::new(), Vec::new(), &[reference, updated])?;
            }
            return Ok(());
        }

        transpose_projected_operation(
            context,
            &<O as OperationProjection<ArrayType>>::Projected::from(self.clone()),
            inputs,
            outputs,
            accumulators,
        )
    }
}

impl Operation for DynamicSliceOperation<ArrayIrType> {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        DYNAMIC_SLICE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let Some(input_type) = input_types.first() else {
            return Err(TypeError::invalid(format!("`{DYNAMIC_SLICE_OPERATION_NAME}` expects an array input")));
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        if self.strides.len() != input_type.rank() {
            return Err(TypeError::invalid(format!(
                "`{}` `strides` has length {} but input has rank {}",
                DYNAMIC_SLICE_OPERATION_NAME,
                self.strides.len(),
                input_type.rank(),
            )));
        }
        check_count!("input", input_types, 1 + 2 * input_type.rank(), TypeError);

        let starts = &input_types[1..1 + input_type.rank()];
        let sizes = &input_types[1 + input_type.rank()..];
        for (axis, (start, size)) in starts.iter().zip(sizes).enumerate() {
            let start = <&DimensionType>::try_from(start)?.bounds().lower();
            let start = if self.bounds == DynamicSliceBounds::Clamp { 0 } else { start };
            let size = <&DimensionType>::try_from(size)?.bounds().lower();

            // Bounds can disprove a slice even before its dimension inputs become concrete. Wider valid ranges
            // retain the runtime assertion because separate identities cannot prove the joint bounds relation.
            let span = if size == 0 {
                Some(0)
            } else {
                (size - 1).checked_mul(self.strides[axis]).and_then(|span| span.checked_add(1))
            };

            let limit = span.and_then(|span| start.checked_add(span)).ok_or_else(|| {
                TypeError::invalid(format!(
                    "`{DYNAMIC_SLICE_OPERATION_NAME}` minimum limit overflows `usize` on axis {axis}",
                ))
            })?;

            let maximum = match input_type.dimension(axis) {
                Dimension::Static(size) => Some(size),
                Dimension::Dynamic(variable) => variable.bounds().upper().map(|upper| upper - 1),
            };

            if maximum.is_some_and(|maximum| limit > maximum) {
                return Err(TypeError::invalid(format!(
                    "`{}` minimum limit {} exceeds maximum input extent {} on axis {}",
                    DYNAMIC_SLICE_OPERATION_NAME,
                    limit,
                    maximum.unwrap(),
                    axis,
                )));
            }
        }

        let dimensions = ArrayIrType::extents(sizes)?;
        let output_type = if dimensions.as_slice() == input_type.shape().dimensions() {
            input_type.clone()
        } else {
            ArrayType::new(input_type.data_type(), Shape::new(dimensions.clone()))
                .with_memory(input_type.memory())
                .with_sharding(input_type.resized_sharding(dimensions.as_slice(), self.name())?)
                .map_err(|error| {
                    TypeError::invalid(format!("`{DYNAMIC_SLICE_OPERATION_NAME}` output type is invalid: {error}"))
                })?
        };

        if !self.requires_runtime_assertion && !self.has_proven_window(input_types)? {
            return Err(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` was constructed without a runtime window check but these input \
                 types require one",
            )));
        }

        Ok(vec![output_type.into()])
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Owned(Effects::explicit(if self.requires_runtime_assertion {
            EffectClasses::single(EffectClass::OrderedAssertion)
        } else {
            EffectClasses::NONE
        }))
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("strides", format_args!("{:?}", self.strides))?;
            operation.field("bounds", if self.bounds == DynamicSliceBounds::Clamp { "clamp" } else { "checked" })?;
            operation.field("requires_runtime_assertion", self.requires_runtime_assertion)
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free DynamicSliceOperation<ArrayIrType>);

impl<C: Domain<Type = ArrayIrType, Value: DynamicSlice>> InterpretableOperation<C>
    for DynamicSliceOperation<ArrayIrType>
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1 + 2 * self.strides.len(), ProgramError);
        let rank = self.strides.len();
        Ok(vec![inputs[0].dynamic_slice_with_bounds(
            &inputs[1..1 + rank],
            &inputs[1 + rank..],
            self.strides(),
            self.bounds,
        )?])
    }
}

impl<C: Context<Type = ArrayIrType>> PartiallyEvaluatableOperation<C> for DynamicSliceOperation<ArrayIrType> where
    C::Operation: From<DynamicSliceOperation<ArrayIrType>>
{
}

impl<
    C: Context<
            Type = ArrayIrType,
            Operation: From<DynamicSliceOperation<ArrayIrType>> + From<ConstantOperation<DimensionValue>>,
        >,
> BatchableOperation<C, ArrayIrBatchingPolicy> for DynamicSliceOperation<ArrayIrType>
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        // This rule currently supports replicated first-class starts and sizes. Different sizes can require ragged
        // output. Different starts with shared sizes would remain rectangular but need a separate reindexing rule.
        // Insert a mapped array axis with start zero, the transform's exact extent, and unit stride. The payload's
        // stride count and the input's rank bound the arity independently as a payload built for another rank must
        // be rejected rather than indexed.
        check_count!("input", inputs, 1 + 2 * self.strides.len(), ProgramError);
        if inputs.iter().any(|input| !input.ragged_axes().is_empty()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{DYNAMIC_SLICE_OPERATION_NAME}` does not support bounded ragged array inputs"),
            }
            .into());
        }

        let (input, bounds) = inputs.split_first().unwrap();
        let unbatched_type = input.unbatched_type();
        let input_type = <&ArrayType>::try_from(&unbatched_type)?;
        check_count!("input", inputs, 1 + 2 * input_type.rank(), ProgramError);

        for bound in bounds {
            bound.validate_replicated_dimension()?;
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

        let batch_axis = input.batch_axis_position().unwrap();
        let axis_dimension = <&DimensionType>::try_from(context.axis_extent().r#type().as_ref())?.to_dimension();
        let input_dimension = <&ArrayType>::try_from(input.value().r#type().as_ref())?.dimension(batch_axis);
        if input_dimension != axis_dimension {
            return Err(BatchingError::MisalignedBatchAxes {
                message: format!(
                    "`{DYNAMIC_SLICE_OPERATION_NAME}` mapped input extent {input_dimension} does not match \
                     batching extent {axis_dimension}",
                ),
            });
        }

        let (starts, sizes) = bounds.split_at(input_type.rank());
        let zero = DimensionValue::constant(0).map_err(ProgramError::from)?;
        let mut zero = context.parent().bind(ConstantOperation::new(zero), Vec::new(), &[])?;
        check_count!("output", zero, 1, ProgramError);

        let mut packed_inputs = Vec::with_capacity(inputs.len() + 2);
        packed_inputs.push(input.value().clone());
        packed_inputs.extend(starts.iter().take(batch_axis).map(|bound| bound.value().clone()));
        packed_inputs.push(zero.remove(0));
        packed_inputs.extend(starts.iter().skip(batch_axis).map(|bound| bound.value().clone()));
        packed_inputs.extend(sizes.iter().take(batch_axis).map(|bound| bound.value().clone()));
        packed_inputs.push(context.axis_extent().clone());
        packed_inputs.extend(sizes.iter().skip(batch_axis).map(|bound| bound.value().clone()));
        let mut strides = self.strides().to_vec();
        strides.insert(batch_axis, 1);
        let operation = Self::from_rank(input_type.rank() + 1).with_strides(strides)?.with_bounds(self.bounds);
        Ok(context
            .parent()
            .bind(operation, Vec::new(), packed_inputs.as_slice())?
            .into_iter()
            .map(|output| ArrayIrBatch::new(output, BatchAxis::from_position(batch_axis)))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

impl_differentiable_operation! {
    DynamicSliceOperation<ArrayIrType>,
    jvp<C>
    where
        C: Context<Type = ArrayIrType>,
        C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
        C::Operation: From<DynamicSliceOperation<ArrayIrType>> + From<LinearCallOperation<ArrayIrType>>
            + From<DimensionSizeOperation> + From<ConstantOperation<DimensionValue>>,
        Tracer<NestedTracingContext<C>>: Value<Type = ArrayIrType, DispatchDomain = NestedTracingContext<C>>
            + DimensionSize + DimensionToScalar
            + ValueProjection<ArrayType, Projected: Add + Mul + Concatenate + Scatter + TransferToMemory>
            + ValueProjection<DimensionType, Projected: Add + Mul + DimensionMin + DimensionSaturatingSub>,
        NestedTracingContext<C>: Context<
                Type = ArrayIrType, Value = Tracer<NestedTracingContext<C>>, Operation = C::Operation,
            > + DimensionConstant + DynamicIota<Tracer<NestedTracingContext<C>>>
            + DynamicZero<Tracer<NestedTracingContext<C>>>,
    {
        |operation, context, _driver, inputs| {
            check_count!("input", inputs, 1 + 2 * operation.strides.len(), ProgramError);
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let mut primals = context.primal().bind(operation.clone(), Vec::new(), &primal_inputs)?;
            check_count!("output", primals, 1, ProgramError);
            let primal = primals.remove(0);
            let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
            let MaybeZero::Value(tangent) = tangent_inputs[0].tangent() else {
                let zero = MaybeZero::Zero(primal.r#type().tangent()?);
                return Ok(vec![DifferentiationDual::new(primal, zero)?]);
            };

            // Static geometry needs no shape residuals as the ordinary transpose can recover the destination type.
            // Keep this common case as one slice rather than introducing paired regions and coordinate arrays early.
            let input_type = tangent.r#type();
            if <&ArrayType>::try_from(input_type.as_ref())?.static_shape().is_some()
                && primal_inputs[1 + operation.strides.len()..].iter().all(|size| {
                    <&DimensionType>::try_from(size.r#type().as_ref()).is_ok_and(|size| size.extent().is_some())
                })
            {
                let mut arguments = vec![tangent.clone()];
                arguments.extend(tangent_inputs[1..].iter().map(|input| input.primal().clone()));
                let tangent = context.tangent().bind(operation.clone(), Vec::new(), &arguments)?.remove(0);
                return Ok(vec![DifferentiationDual::new(primal, MaybeZero::Value(tangent))?]);
            }

            // The result cannot reveal the original extent (e.g., slicing `[n]` to `[k]`). Retain that geometry once,
            // alongside discrete starts and sizes, so the transpose survives replay and specialization.
            let mut residuals = LinearResiduals::new();
            let bounds = residuals.retain_all(tangent_inputs[1..].iter().map(|input| input.primal().clone()));
            let input_shape = residuals.retain_shape(context.tangent(), tangent_inputs[0].primal())?;
            let input_type = tangent.r#type().cotangent()?;
            let input_type = <&ArrayType>::try_from(&input_type)?.clone();
            let forward_bounds = bounds.clone();
            let forward = operation.clone();
            let transpose = operation.clone();
            let tangent = LinearCallOperation::stage(
                context.tangent(), residuals.into_values(), vec![tangent.clone()],
                move |residuals, inputs| {
                    let mut arguments = vec![inputs[0].clone()];
                    arguments.extend(forward_bounds.iter().map(|index| residuals[*index].clone()));
                    inputs[0].dispatch_domain().bind(forward, Vec::new(), &arguments)
                },
                move |residuals, cotangents| {
                    let context = cotangents[0].dispatch_domain();
                    let zeros = context.dynamic_zero(&input_type, &input_shape.dynamic_dimensions(residuals))?;
                    let starts = bounds[..transpose.strides.len()].iter().map(|index| residuals[*index].clone())
                        .collect::<Vec<_>>();
                    let sizes = bounds[transpose.strides.len()..].iter().map(|index| residuals[*index].clone())
                        .collect::<Vec<_>>();
                    Ok(vec![transpose.apply_adjoint(&zeros, &cotangents[0], &starts, &sizes)?])
                },
            )?.remove(0);
            Ok(vec![DifferentiationDual::new(primal, MaybeZero::Value(tangent))?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayIrType>,
        O: Operation<Type = ArrayIrType>,
        Tracer<TracingContext<V, O>>: Value<Type = ArrayIrType, DispatchDomain = TracingContext<V, O>>
            + DimensionSize + DimensionToScalar
            + ValueProjection<ArrayType, Projected: Add + Mul + Concatenate + Scatter + TransferToMemory>
            + ValueProjection<DimensionType, Projected: Add + Mul + DimensionMin + DimensionSaturatingSub>,
        TracingContext<V, O>: Context<
                Type = ArrayIrType, Value = Tracer<TracingContext<V, O>>, Operation = O,
            > + DimensionConstant + DynamicIota<Tracer<TracingContext<V, O>>>
            + DynamicZero<Tracer<TracingContext<V, O>>>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            check_count!("input", inputs, 1 + 2 * operation.strides.len(), ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
            let MaybeZero::Value(cotangent) = &outputs[0] else { return Ok(()); };
            if !accumulators[0].is_needed() { return Ok(()); }
            let input_type = inputs[0].r#type().cotangent()?;
            let input_type = <&ArrayType>::try_from(&input_type)?;
            if input_type.static_shape().is_none() {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` direct transpose requires retained input extents; \
                         use linearization",
                    ),
                }.into());
            }
            let starts = inputs[1..1 + operation.strides.len()].iter().map(|input| input.as_known().cloned()
                .ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("`{DYNAMIC_SLICE_OPERATION_NAME}` transpose requires known start indices"),
                })).collect::<Result<Vec<_>, _>>()?;
            let zeros = (**context).dynamic_zero(input_type, &[])?;
            let sizes = (0..operation.strides.len()).map(|axis| cotangent.dimension_size(axis))
                .collect::<Result<Vec<_>, _>>()?;
            let cotangent = operation.apply_adjoint(&zeros, cotangent, &starts, &sizes)?;
            accumulators[0].accumulate(context, MaybeZero::Value(cotangent))
        }
    },
}

impl<O: Operation<Type = ArrayType> + From<DynamicSliceOperation>> OperationProvider<ArrayType, DynamicSliceOperation>
    for O
{
    type Operation = Self;

    fn provide(request: DynamicSliceOperation, input_types: &[&ArrayType]) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 1 + request.sizes().len(), ProgramError);
        Ok(request.into())
    }
}

impl<O: Operation<Type = ArrayIrType> + OperationProjection<ArrayType, Projected: From<DynamicSliceOperation>>>
    OperationProvider<ArrayIrType, DynamicSliceOperation> for O
{
    type Operation = Self;

    fn provide(request: DynamicSliceOperation, input_types: &[&ArrayIrType]) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 1 + request.sizes().len(), ProgramError);
        Ok(<Self as OperationProjection<ArrayType>>::Projected::from(request).into())
    }
}

/// Value capability for extracting a sub-array at runtime start indices. [`dynamic_slice`](Self::dynamic_slice)
/// extracts a window of static sizes at scalar-array starts that count from the end of their axes when negative and
/// then clamp, while [`dynamic_slice_with_dimensions`](Self::dynamic_slice_with_dimensions) and
/// [`dynamic_slice_with_bounds`](Self::dynamic_slice_with_bounds) take first-class dimension starts and sizes, so the
/// result extents may vary at runtime. The dimension-taking functions require [`Value<Type = ArrayIrType>`](Value),
/// because the mixed [`ArrayIrValue`] representation is what carries dimensions alongside arrays, and so they are
/// unavailable for homogeneous arrays and type descriptors:
///
/// ```compile_fail
/// use ryft_core::{Array, DynamicSlice};
/// let input = Array::scalar(1.0_f32).unwrap();
/// input.dynamic_slice_with_dimensions(&[], &[], &[]).unwrap();
/// ```
///
/// With dimension inputs, starts are non-negative dimensions and positive static strides select `start + i * stride`
/// for `0 <= i < size`. The [`DynamicSliceBounds`] policy decides how starts meet the input: under the default
/// [`Checked`](DynamicSliceBounds::Checked) policy, every selected element must lie within the input and an empty axis
/// permits a start at its end, while the [`Clamp`](DynamicSliceBounds::Clamp) policy moves the start so that the
/// requested window fits without shrinking it. Invalid runtime bounds remain observable even when the result is unused.
/// The output preserves the input memory space and inferred sharding; an identity slice preserves the input layout and
/// any other slice uses a fresh dense layout. Array tangents follow the same slice, and reverse-mode differentiation
/// retains the original input extents and adds cotangents at the selected coordinates, treating starts and sizes as
/// discrete metadata.
///
/// Backends may require finite bounds on dimension-sized windows (e.g., the XLA backend). That is, accepting a dynamic
/// shape in the core does not imply support for unbounded allocation or runtime byte strides for all backends.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ArrayIrValue, DimensionValue, DynamicSlice, ProgramError};
/// # fn example() -> Result<(), ProgramError> {
/// // Shapes: input [4] -> result [2]; start and size are dimension values, not tensors.
/// let input = ArrayIrValue::Array(Array::vector(vec![10i32, 20, 30, 40])?);
/// let start = ArrayIrValue::Dimension(DimensionValue::constant(1)?);
/// let size = ArrayIrValue::Dimension(DimensionValue::constant(2)?);
/// let result = input.dynamic_slice_with_dimensions(&[start], &[size], &[1])?;
/// assert_eq!(result, ArrayIrValue::Array(Array::vector(vec![20i32, 30])?));
/// # Ok(())
/// # }
/// # example().unwrap();
/// ```
pub trait DynamicSlice: Sized {
    /// Extracts a statically shaped sub-array at runtime start indices, with the semantics of StableHLO's
    /// [`dynamic_slice`](https://openxla.org/stablehlo/spec#dynamic_slice) operation. `t.dynamic_slice(start_indices,
    /// sizes)` extracts the block of shape `sizes` whose origin is given by the scalar integer values in
    /// `start_indices` (one per input axis). A negative signed start counts from the end of its axis (i.e., on an axis
    /// of extent `d`, a start `i < 0` becomes `i + d` once). Every start is then clamped so that the extracted block
    /// always lies in bounds: the effective start index along axis `d` is `clamp(0, start_indices[d],
    /// input_dimension[d] - sizes[d])`, so `-1` names the last valid origin and a start that is still negative after
    /// the single wrap clamps to zero. Unsigned and Boolean starts never wrap. The output shape is exactly `sizes` and
    /// is fully static even though the slice origin is not. Each static input axis must satisfy `sizes[d] <=
    /// input_dimension[d]`. For a [`Dimension::Dynamic`] input axis, its declared lower bound must be at least the
    /// requested size, proving that the block fits every admitted runtime extent. The input and start indices must
    /// reside in the same memory space. A slice whose sizes equal the input shape passes it through unchanged because
    /// every resolved origin is necessarily zero. Any other output preserves the input memory space and clears explicit
    /// physical layout metadata. An input carrying reduction state keeps it, provided the start indices are invariant
    /// over its reduction-state mesh axes: indexing with one routing on every device commutes with the pending sum of
    /// partial contributions.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{Array, ArrayType, DataType, DynamicSlice, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// // Extract a 1x2 block starting at row 1, column 1 of a 2x3 matrix.
    /// // Shapes: input [2, 3], row and column [] (scalars) -> output [1, 2].
    /// let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// let row = Array::from_elements::<i32>(ArrayType::scalar(DataType::I32), &[1])?;
    /// let column = Array::from_elements::<i32>(ArrayType::scalar(DataType::I32), &[1])?;
    /// let output = input.dynamic_slice(&[row, column], &[1, 2])?;
    /// // `output` has shape [1, 2] with values [[5.0, 6.0]].
    /// assert_eq!(output.to_f64s(), vec![5.0, 6.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn dynamic_slice(&self, start_indices: &[Self], sizes: &[usize]) -> Result<Self, ProgramError> {
        self.dynamic_slice_with_negative_indices(start_indices, sizes, true)
    }

    /// Extracts a statically shaped sub-array at runtime start indices like [`dynamic_slice`](Self::dynamic_slice),
    /// selecting how negative signed starts are treated. With `allow_negative_indices` set to `true`, a negative start
    /// counts from the end of its axis exactly as [`dynamic_slice`](Self::dynamic_slice) describes; with `false`,
    /// negative starts are out of bounds and clamp to zero, which is StableHLO's native rule. Unsigned and Boolean
    /// starts are unaffected either way.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{Array, DynamicSlice, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// // Shapes: input [4], start [] (scalar) -> output [2].
    /// let input = Array::vector(vec![10i32, 20, 30, 40])?;
    /// let start = Array::scalar(-1i32)?;
    ///
    /// // A negative start wraps to `3` and then clamps to the last valid origin `2`.
    /// assert_eq!(
    ///     input.dynamic_slice_with_negative_indices(&[start.clone()], &[2], true)?,
    ///     Array::vector(vec![30i32, 40])?,
    /// );
    ///
    /// // Without wrapping, the same start is out of bounds and clamps to zero.
    /// assert_eq!(
    ///     input.dynamic_slice_with_negative_indices(&[start], &[2], false)?,
    ///     Array::vector(vec![10i32, 20])?,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    fn dynamic_slice_with_negative_indices(
        &self,
        start_indices: &[Self],
        sizes: &[usize],
        allow_negative_indices: bool,
    ) -> Result<Self, ProgramError>;

    /// Extracts a window whose starts and sizes are dimension values, rejecting windows that extend outside the input
    /// (i.e., using the [`Checked`](DynamicSliceBounds::Checked) bounds policy). Refer to the documentation of
    /// [`dynamic_slice_with_bounds`](Self::dynamic_slice_with_bounds) for more information.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{Array, ArrayIrValue, DimensionValue, DynamicSlice, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// // Shapes: input [4] -> output [2]. Every second element from `1` onwards is selected.
    /// let input = ArrayIrValue::Array(Array::vector(vec![10i32, 20, 30, 40])?);
    /// let start = ArrayIrValue::Dimension(DimensionValue::constant(1)?);
    /// let size = ArrayIrValue::Dimension(DimensionValue::constant(2)?);
    /// assert_eq!(
    ///     input.dynamic_slice_with_dimensions(&[start], &[size], &[2])?,
    ///     ArrayIrValue::Array(Array::vector(vec![20i32, 40])?),
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn dynamic_slice_with_dimensions(
        &self,
        start_indices: &[Self],
        sizes: &[Self],
        strides: &[usize],
    ) -> Result<Self, ProgramError>
    where
        Self: Value<Type = ArrayIrType>,
    {
        self.dynamic_slice_with_bounds(start_indices, sizes, strides, DynamicSliceBounds::Checked)
    }

    /// Extracts a window whose starts and sizes are dimension values, resolving the starts against the input extents
    /// according to the provided [`DynamicSliceBounds`] policy. [`Clamp`](DynamicSliceBounds::Clamp) moves each start
    /// so that its window fits without changing the requested sizes, whereas [`Checked`](DynamicSliceBounds::Checked)
    /// rejects a window that extends outside the input. Both policies reject a window whose span exceeds the logical
    /// input extent.
    ///
    /// # Parameters
    ///
    ///   - `start_indices`: One non-negative dimension value per input axis, specifying its inclusive start.
    ///   - `sizes`: One dimension value per input axis, specifying the number of selected elements.
    ///   - `strides`: One strictly positive static step per input axis.
    ///   - `bounds`: Policy resolving the starts against the input extents.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{Array, ArrayIrValue, DimensionValue, DynamicSlice, DynamicSliceBounds, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// // Shapes: input [4] -> output [2]. A window of size `2` starting at `3` extends past the input.
    /// let input = ArrayIrValue::Array(Array::vector(vec![10i32, 20, 30, 40])?);
    /// let start = ArrayIrValue::Dimension(DimensionValue::constant(3)?);
    /// let size = ArrayIrValue::Dimension(DimensionValue::constant(2)?);
    ///
    /// // `Clamp` moves the start back to `2` so that the window fits.
    /// assert_eq!(
    ///     input.dynamic_slice_with_bounds(&[start.clone()], &[size.clone()], &[1], DynamicSliceBounds::Clamp)?,
    ///     ArrayIrValue::Array(Array::vector(vec![30i32, 40])?),
    /// );
    ///
    /// // `Checked` rejects the window instead.
    /// assert!(matches!(
    ///     input.dynamic_slice_with_bounds(&[start], &[size], &[1], DynamicSliceBounds::Checked),
    ///     Err(ProgramError::InvalidArgument { .. }),
    /// ));
    /// # Ok(())
    /// # }
    /// ```
    fn dynamic_slice_with_bounds(
        &self,
        start_indices: &[Self],
        sizes: &[Self],
        strides: &[usize],
        bounds: DynamicSliceBounds,
    ) -> Result<Self, ProgramError>
    where
        Self: Value<Type = ArrayIrType>;

    /// Slices one axis with host-known indices while retaining every other runtime extent. This convenience constructs
    /// integer queries and uses [`DynamicGather`], sharing its gather/scatter transformation rules.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Axis to slice; negative axes count backward from the input rank.
    ///   - `start`: Non-negative inclusive start, no larger than `limit`.
    ///   - `limit`: Exclusive limit, which must be proven within the selected axis by its declared bounds.
    ///   - `stride`: Positive distance between selected elements. Indices do not wrap or clamp.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{
    /// #     Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds,
    /// #     DimensionVariable, DynamicSlice, EagerContext, ProgramError, Shape, Trace,
    /// # };
    /// # fn main() -> Result<(), ProgramError> {
    /// // Select columns 1 and 3 of every row. The staged program retains the symbolic row extent.
    /// let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(6))?);
    /// let shape = Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(4)]);
    /// let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
    ///     |input| input.dynamic_slice_axis(1, 1, 4, 2),
    ///     ArrayIrType::Array(ArrayType::new(DataType::F64, shape)),
    /// )?;
    /// let matrix = Array::matrix(2, 4, vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])?;
    /// assert_eq!(
    ///     program.interpret(ArrayIrValue::Array(matrix))?,
    ///     ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0f64, 3.0, 5.0, 7.0])?),
    /// );
    /// # Ok(())
    /// # }
    /// ```
    fn dynamic_slice_axis<A: Into<Axis>>(
        &self,
        axis: A,
        start: usize,
        limit: usize,
        stride: usize,
    ) -> Result<Self, ProgramError>
    where
        Self: Value<Type = ArrayIrType>
            + DynamicGather
            + DimensionToScalar
            + ValueProjection<ArrayType, Projected: Add + Mul + Broadcast + TransferToMemory>,
        Self::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant + DynamicIota<Self>,
    {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let axis = axis.into().normalize(input_type.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        if stride == 0 || start > limit {
            return Err(TypeError::invalid(
                "`dynamic_slice_axis` requires a positive stride and start no greater than limit",
            )
            .into());
        }

        let minimum = match input_type.dimension(axis) {
            Dimension::Static(size) => size,
            Dimension::Dynamic(variable) => variable.bounds().lower(),
        };

        if limit > minimum {
            return Err(TypeError::invalid(format!(
                "`dynamic_slice_axis` limit {limit} exceeds the guaranteed extent {minimum} of axis {axis}"
            ))
            .into());
        }

        let count = (limit - start).div_ceil(stride);
        let query_type = ArrayType::new_static(DataType::I64, [count]).with_memory(input_type.memory());
        let context = self.dispatch_domain();
        let mut queries = context.dynamic_iota(&query_type, 0, &[])?.into_projected()?;

        // Scalar dimension literals are available in every mixed context, including compiled contexts whose array
        // constants live in capture tables. Convert and place them before ordinary integer array arithmetic.
        if stride != 1 {
            let scale = context
                .dimension_constant(stride)?
                .to_scalar()?
                .into_projected()?
                .transfer_to_memory(input_type.memory())?
                .broadcast(query_type.clone(), &[])?;
            queries = queries.mul(&scale)?;
        }

        if start != 0 {
            let offset = context
                .dimension_constant(start)?
                .to_scalar()?
                .into_projected()?
                .transfer_to_memory(input_type.memory())?
                .broadcast(query_type.clone(), &[])?;
            queries = queries.add(&offset)?;
        }

        self.dynamic_gather_axis(&Self::from_projected(queries), axis, GatherMode::PromiseInBounds)
    }

    /// Selects one non-negative index on an axis, optionally retaining that axis with size one. Untouched runtime
    /// dimensions are preserved. This composes [`Self::dynamic_slice_axis`] with [`DynamicReshape`] and has the same
    /// bounds requirements and gather/scatter differentiation behavior.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Input axis containing the index; negative axes count backward from the end.
    ///   - `index`: Non-negative coordinate that must be proven in bounds.
    ///   - `keep_axis`: Whether the selected axis remains in the output with extent one.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{
    /// #     Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds,
    /// #     DimensionVariable, DynamicSlice, EagerContext, ProgramError, Shape, Trace,
    /// # };
    /// # fn main() -> Result<(), ProgramError> {
    /// // Select column 3 of every row and drop the column axis. The staged program retains the symbolic row extent.
    /// let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(6))?);
    /// let shape = Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(4)]);
    /// let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
    ///     |input| input.dynamic_index_axis(1, 3, false),
    ///     ArrayIrType::Array(ArrayType::new(DataType::F64, shape)),
    /// )?;
    /// let matrix = Array::matrix(2, 4, vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])?;
    /// assert_eq!(
    ///     program.interpret(ArrayIrValue::Array(matrix))?,
    ///     ArrayIrValue::Array(Array::vector(vec![3.0f64, 7.0])?),
    /// );
    /// # Ok(())
    /// # }
    /// ```
    fn dynamic_index_axis<A: Into<Axis>>(&self, axis: A, index: usize, keep_axis: bool) -> Result<Self, ProgramError>
    where
        Self: Value<Type = ArrayIrType>
            + DynamicGather
            + DynamicReshape
            + DimensionSize
            + DimensionToScalar
            + ValueProjection<ArrayType, Projected: Add + Mul + Broadcast + TransferToMemory>,
        Self::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant + DynamicIota<Self>,
    {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let axis = axis.into().normalize(input_type.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        let limit = index
            .checked_add(1)
            .ok_or_else(|| TypeError::invalid("`dynamic_index_axis` index overflows `usize`"))?;
        let output = self.dynamic_slice_axis(axis, index, limit, 1)?;
        if keep_axis {
            return Ok(output);
        }
        let dimensions = (0..input_type.rank())
            .filter(|input_axis| *input_axis != axis)
            .map(|axis| self.dimension_size(axis))
            .collect::<Result<Vec<_>, _>>()?;
        output.dynamic_reshape(&dimensions)
    }

    /// Extracts `size` elements along `axis` starting at the runtime scalar `start`, keeping every other axis in full.
    /// This is [`dynamic_slice`](Self::dynamic_slice) with a zero start on every other axis, so `start` counts from the
    /// end of `axis` when negative and clamps so that the window fits, and it requires the other axes to have static
    /// extents. Unlike [`dynamic_slice_axis`](Self::dynamic_slice_axis), which takes host-known bounds proven from
    /// the declared extents, this function takes a traced start.
    ///
    /// # Parameters
    ///
    ///   - `start`: Scalar integer array giving the start along `axis`.
    ///   - `size`: Static number of elements to extract along `axis`, at most the axis extent.
    ///   - `axis`: Axis to slice; negative axes count backward from the input rank.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{Array, DynamicSlice, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// // A start of `-2` counts from the end of the column axis, so the window covers the last two columns.
    /// let matrix = Array::matrix(2, 3, vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// let window = matrix.dynamic_slice_in_axis(&Array::scalar(-2i32)?, 2, 1)?;
    /// assert_eq!(window, Array::matrix(2, 2, vec![2.0f64, 3.0, 5.0, 6.0])?);
    /// # Ok(())
    /// # }
    /// ```
    fn dynamic_slice_in_axis<A: Into<Axis>>(&self, start: &Self, size: usize, axis: A) -> Result<Self, ProgramError>
    where
        Self: Clone + Typed<Type = ArrayType> + ZeroLike,
    {
        let input_type = self.r#type();
        let axis = axis.into().normalize(input_type.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        let mut sizes = static_dimensions_for_axis_window(DYNAMIC_SLICE_OPERATION_NAME, &input_type, axis)?;
        sizes[axis] = size;
        let starts = (0..input_type.rank())
            .map(|input_axis| if input_axis == axis { Ok(start.clone()) } else { start.zero_like() })
            .collect::<Result<Vec<_>, _>>()?;
        self.dynamic_slice(&starts, &sizes)
    }

    /// Selects the element at the runtime scalar `index` along `axis`, optionally retaining that axis with extent one.
    /// This is [`dynamic_slice_in_axis`](Self::dynamic_slice_in_axis) with a window of size one, so `index` counts from
    /// the end of `axis` when negative and clamps into bounds. Unlike [`dynamic_index_axis`](Self::dynamic_index_axis),
    /// which takes a host-known index, this function takes a traced index.
    ///
    /// # Parameters
    ///
    ///   - `index`: Scalar integer array giving the position along `axis`.
    ///   - `axis`: Axis to index; negative axes count backward from the input rank.
    ///   - `keep_axis`: Whether the selected axis remains in the output with extent one.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{Array, DynamicSlice, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// // An index of `-1` selects the last column; dropping the axis leaves one element per row.
    /// let matrix = Array::matrix(2, 3, vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// assert_eq!(
    ///     matrix.dynamic_index_in_axis(&Array::scalar(-1i32)?, 1, false)?,
    ///     Array::vector(vec![3.0f64, 6.0])?,
    /// );
    /// assert_eq!(
    ///     matrix.dynamic_index_in_axis(&Array::scalar(-1i32)?, 1, true)?,
    ///     Array::matrix(2, 1, vec![3.0f64, 6.0])?,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    fn dynamic_index_in_axis<A: Into<Axis>>(&self, index: &Self, axis: A, keep_axis: bool) -> Result<Self, ProgramError>
    where
        Self: Clone + Typed<Type = ArrayType> + ZeroLike + Reshape,
    {
        let input_type = self.r#type();
        let axis = axis.into().normalize(input_type.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        let output = self.dynamic_slice_in_axis(index, 1, axis)?;
        if keep_axis {
            return Ok(output);
        }
        let mut dimensions = static_dimensions_for_axis_window(DYNAMIC_SLICE_OPERATION_NAME, &input_type, axis)?;
        dimensions.remove(axis);
        output.reshape(dimensions)
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Returns the static extents of `input_type` for a window that spans every axis other than `axis` in full, naming
/// `operation_name` in the diagnostic when another axis is dynamic.
fn static_dimensions_for_axis_window(
    operation_name: &str,
    input_type: &ArrayType,
    axis: usize,
) -> Result<Vec<usize>, TypeError> {
    input_type
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .map(|(input_axis, dimension)| match dimension {
            Dimension::Static(size) => Ok(*size),
            Dimension::Dynamic(_) if input_axis == axis => Ok(0),
            Dimension::Dynamic(_) => Err(TypeError::invalid(format!(
                "`{operation_name}` along axis {axis} requires static extents on the other axes but axis {input_axis} \
                 of `{input_type}` is dynamic",
            ))),
        })
        .collect()
}

impl DynamicSlice for ArrayType {
    fn dynamic_slice_with_negative_indices(
        &self,
        start_indices: &[Self],
        sizes: &[usize],
        _allow_negative_indices: bool,
    ) -> Result<ArrayType, ProgramError> {
        let rank = self.rank();
        if start_indices.len() != rank {
            return Err(TypeError::invalid(format!(
                "`{}` expects one start index per input axis ({}) but got {}",
                DYNAMIC_SLICE_OPERATION_NAME,
                rank,
                start_indices.len(),
            ))
            .into());
        }
        if sizes.len() != rank {
            return Err(TypeError::invalid(format!(
                "`{}` sizes has length {} but input has rank {}",
                DYNAMIC_SLICE_OPERATION_NAME,
                sizes.len(),
                rank,
            ))
            .into());
        }
        validate_start_index_types(DYNAMIC_SLICE_OPERATION_NAME, self, start_indices)?;
        for (axis, &size) in sizes.iter().enumerate() {
            // Clamping can keep the window in bounds only if it fits every possible input extent. Static axes use
            // their exact size; dynamic axes must have a lower bound at least as large as the requested window.
            match self.dimension(axis) {
                Dimension::Static(input_size) if size > input_size => {
                    return Err(TypeError::invalid(format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` size {size} is out of bounds for axis {axis} with size \
                        {input_size}",
                    ))
                    .into());
                }
                Dimension::Dynamic(variable) if size > variable.bounds().lower() => {
                    return Err(TypeError::invalid(format!(
                        "`{}` size {} exceeds the guaranteed minimum extent {} of dynamic axis {}",
                        DYNAMIC_SLICE_OPERATION_NAME,
                        size,
                        variable.bounds().lower(),
                        axis,
                    ))
                    .into());
                }
                _ => {}
            }
        }
        let output_dimensions = sizes.iter().map(|size| Dimension::Static(*size)).collect::<Vec<_>>();
        if output_dimensions.as_slice() == self.shape().dimensions() {
            return indexed_slice_output_type(self.clone(), start_indices, DYNAMIC_SLICE_OPERATION_NAME);
        }
        let sharding = self.resized_sharding(&output_dimensions, DYNAMIC_SLICE_OPERATION_NAME)?;
        let output_type = ArrayType::new(self.data_type(), Shape::new(output_dimensions))
            .with_memory(self.memory())
            .with_sharding(sharding)
            .map_err(|error| {
                TypeError::invalid(format!("`{DYNAMIC_SLICE_OPERATION_NAME}` output type is invalid: {error}"))
            })?;
        indexed_slice_output_type(output_type, start_indices, DYNAMIC_SLICE_OPERATION_NAME)
    }

    fn dynamic_slice_with_bounds(
        &self,
        _start_indices: &[Self],
        _sizes: &[Self],
        _strides: &[usize],
        _bounds: DynamicSliceBounds,
    ) -> Result<Self, ProgramError> {
        // The trait restricts this function to `Value<Type = ArrayIrType>`, which this type cannot implement.
        unreachable!("dimension inputs require a mixed array value")
    }
}

impl DynamicSlice for Array {
    fn dynamic_slice_with_negative_indices(
        &self,
        start_indices: &[Self],
        sizes: &[usize],
        allow_negative_indices: bool,
    ) -> Result<Self, ProgramError> {
        let index_types = start_indices.iter().map(|index| index.r#type().into_owned()).collect::<Vec<_>>();
        let output_type = self.r#type().dynamic_slice(&index_types, sizes)?;
        let input_shape = self.r#type().static_shape().unwrap();
        let starts = Self::clamped_start_indices(start_indices, &input_shape, sizes, allow_negative_indices);
        let axes = starts
            .iter()
            .zip(sizes)
            .map(|(start, size)| ArraySliceAxis::new(*start, *size, 1))
            .collect::<Vec<_>>();
        self.copy_block(output_type, &axes)
    }

    fn dynamic_slice_with_bounds(
        &self,
        _start_indices: &[Self],
        _sizes: &[Self],
        _strides: &[usize],
        _bounds: DynamicSliceBounds,
    ) -> Result<Self, ProgramError> {
        // The trait restricts this function to `Value<Type = ArrayIrType>`, which this type cannot implement.
        unreachable!("dimension inputs require a mixed array value")
    }
}

impl<A: DimensionSize<usize> + Slice + DynamicSlice + Value<Type = ArrayType>> DynamicSlice for ArrayIrValue<A> {
    fn dynamic_slice_with_negative_indices(
        &self,
        start_indices: &[Self],
        sizes: &[usize],
        allow_negative_indices: bool,
    ) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let start_indices = start_indices
            .iter()
            .cloned()
            .map(ValueProjection::<ArrayType>::into_projected)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::Array(input.dynamic_slice_with_negative_indices(&start_indices, sizes, allow_negative_indices)?))
    }

    fn dynamic_slice_with_bounds(
        &self,
        start_indices: &[Self],
        sizes: &[Self],
        strides: &[usize],
        policy: DynamicSliceBounds,
    ) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let rank = input.r#type().rank();
        validate_dynamic_slice_bound_counts(rank, start_indices.len(), sizes.len())?;
        let operation = DynamicSliceOperation::<ArrayIrType>::from_rank(rank)
            .with_strides(strides.to_vec())?
            .with_bounds(policy);
        let strides = operation.strides();
        // Binding every start and size to its dimension identity rejects repeated identities that denote different
        // runtime extents; the bindings are the validation.
        let mut refinements = ArrayTypeRefinements::default();
        let bounds = start_indices
            .iter()
            .chain(sizes)
            .cloned()
            .map(ValueProjection::<DimensionType>::into_projected)
            .map(|result| {
                let value = result?;
                refinements.bind(value.r#type().variable(), value.extent())?;
                Ok::<_, TypeError>(value.extent())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (starts, sizes) = bounds.split_at(rank);
        let mut starts = starts.to_vec();
        let limits = starts
            .iter_mut()
            .zip(sizes)
            .zip(strides)
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
                                "`{DYNAMIC_SLICE_OPERATION_NAME}` span overflows `usize` on axis {axis}",
                            ))
                        })?
                };
                let input_size = input.dimension_size(axis)?;
                if policy == DynamicSliceBounds::Clamp && span <= input_size {
                    *start = (*start).min(input_size - span);
                }
                let limit = start.checked_add(span).ok_or_else(|| {
                    TypeError::invalid(format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` limit overflows `usize` on axis {axis}",
                    ))
                })?;
                if limit > input_size {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "`{DYNAMIC_SLICE_OPERATION_NAME}` limit {limit} exceeds input axis {axis} extent \
                            {input_size}",
                        ),
                    });
                }
                Ok(limit)
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        Ok(<Self as ValueProjection<ArrayType>>::from_projected(input.slice(&starts, &limits, strides)?))
    }
}

impl<V: Value> DynamicSlice for V
where
    V::DispatchDomain: Context,
    <V::DispatchDomain as Domain>::Operation: From<DynamicSliceOperation<V::Type>>
        + OperationProvider<V::Type, DynamicSliceOperation, Operation = <V::DispatchDomain as Domain>::Operation>,
{
    fn dynamic_slice_with_negative_indices(
        &self,
        start_indices: &[Self],
        sizes: &[usize],
        allow_negative_indices: bool,
    ) -> Result<Self, ProgramError> {
        let mut inputs = Vec::with_capacity(1 + start_indices.len());
        inputs.push(self.clone());
        inputs.extend_from_slice(start_indices);
        let input_types = inputs.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let operation = <V::DispatchDomain as Domain>::Operation::provide(
            DynamicSliceOperation::new(sizes.to_vec()).with_allow_negative_indices(allow_negative_indices),
            &input_types.iter().collect::<Vec<_>>(),
        )?;
        // Preserve the identity-window shortcut after the operation has validated all start-index types.
        if operation.infer_output_types(&input_types, &[])? == vec![self.r#type().into_owned()] {
            return Ok(self.clone());
        }
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    fn dynamic_slice_with_bounds(
        &self,
        start_indices: &[Self],
        sizes: &[Self],
        strides: &[usize],
        policy: DynamicSliceBounds,
    ) -> Result<Self, ProgramError>
    where
        Self: Value<Type = ArrayIrType>,
    {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        validate_dynamic_slice_bound_counts(input_type.rank(), start_indices.len(), sizes.len())?;
        let operation = DynamicSliceOperation::<ArrayIrType>::from_rank(input_type.rank())
            .with_strides(strides.to_vec())?
            .with_bounds(policy);
        let mut inputs = vec![self.clone()];
        inputs.extend_from_slice(start_indices);
        inputs.extend_from_slice(sizes);
        let input_types = inputs.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let operation = operation.with_input_types(&input_types)?;
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Canonical operation name for [`DynamicUpdateSliceOperation`].
pub const DYNAMIC_UPDATE_SLICE_OPERATION_NAME: &str = "dynamic_update_slice";

/// [`Operation`] that overwrites a contiguous sub-array of its first input with its second input at start indices that
/// are computed at run time, counting negative signed starts from the end of their axes under its
/// [`allows_negative_indices`](Self::allows_negative_indices) policy and then clamping. Refer to the documentation of
/// [`DynamicUpdateSlice`] for more information.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicUpdateSliceOperation {
    /// Refer to the documentation of [`allows_negative_indices`](Self::allows_negative_indices) for more information.
    allow_negative_indices: bool,
}

impl DynamicUpdateSliceOperation {
    /// Creates a new [`DynamicUpdateSliceOperation`] that counts negative start indices from the end of their axes.
    #[inline]
    pub fn new() -> Self {
        Self { allow_negative_indices: true }
    }

    /// Returns a copy of this [`DynamicUpdateSliceOperation`] with its negative-index policy set to
    /// `allow_negative_indices`. Refer to the documentation of
    /// [`allows_negative_indices`](Self::allows_negative_indices) for the meaning of both settings.
    #[inline]
    pub fn with_allow_negative_indices(mut self, allow_negative_indices: bool) -> Self {
        self.allow_negative_indices = allow_negative_indices;
        self
    }

    /// Returns whether a negative signed start index counts from the end of its axis, with the same meaning as
    /// [`DynamicSliceOperation::allows_negative_indices`]: when `true` (the default), a negative start `i` on an axis
    /// of extent `d` is replaced by `i + d` once before clamping; when `false`, it clamps to zero directly.
    #[inline]
    pub fn allows_negative_indices(&self) -> bool {
        self.allow_negative_indices
    }
}

impl Default for DynamicUpdateSliceOperation {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Display for DynamicUpdateSliceOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DynamicUpdateSliceOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        DYNAMIC_UPDATE_SLICE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        if input_types.len() < 2 {
            return Err(TypeError::invalid(format!(
                "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` expects an array input and an update input followed by start \
                 index inputs but got {} inputs",
                input_types.len(),
            )));
        }
        match input_types[0].dynamic_update_slice(&input_types[1], &input_types[2..]) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let operation = OperationFormatter::new(formatter, indentation, self.name())?;
        // The default policy is implied so that ordinary renderings stay unchanged.
        if self.allow_negative_indices {
            return Ok(());
        }
        operation.bracketed(|operation| operation.field("allow_negative_indices", false))
    }
}

impl_reference_dischargeable_operation!(@reference_free DynamicUpdateSliceOperation);

impl<C: Domain<Type = ArrayType, Value: DynamicUpdateSlice>> InterpretableOperation<C> for DynamicUpdateSliceOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let [input, update, start_indices @ ..] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() });
        };
        check_count!("input", inputs, 2 + input.r#type().rank(), ProgramError);
        Ok(vec![input.dynamic_update_slice_with_negative_indices(
            update,
            start_indices,
            self.allow_negative_indices,
        )?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DynamicUpdateSliceOperation where
    C::Operation: From<DynamicUpdateSliceOperation>
{
}

impl<C, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>> for DynamicUpdateSliceOperation
where
    C: Context<Type = ArrayType>,
    C::Value: ZeroLike + Broadcast + Transpose + Slice + Reshape + Concatenate + Reshard,
    DynamicUpdateSliceOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // Batching rule for [`DynamicUpdateSliceOperation`].
        //
        // Replicated start indices keep the structural fast path: the input and update inputs are aligned on one
        // physical batch
        // axis (replicated inputs are broadcast to gain it), and the lifted operation inserts a zero start index for
        // that axis,
        // derived from an existing index input via [`ZeroLike`] so the inserted index carries the same scalar integer
        // type.
        // Rank-0 inputs have no index inputs to donate a zero index, but a rank-0 dynamic update-slice replaces the
        // input with
        // the update entirely, so the update input passes through unchanged.
        //
        // Batch-varying (batched) start indices cannot ride along structurally — every batch item needs its own update
        // origin
        // while the lifted operation reads one origin for all batch items — so the rule falls back to per-item
        // expansion via
        // `batch_by_item_expansion`: each batch item's input, update, and start indices are extracted (replicated
        // inputs are
        // used whole), updated per item, and restacked along a fresh leading batch axis (the result's batch axis is `0`
        // even
        // when the inputs carried their batch axes elsewhere). The expansion stages `O(batch_size)` operations and
        // behaves
        // identically in eager and tracing contexts because it only goes through the value capability traits.
        // Static or clamped windows cannot describe a changed ragged extent.
        if inputs.iter().any(|input| !input.ragged_axes().is_empty()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` does not support bounded ragged array inputs"
                ),
            }
            .into());
        }
        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }
        let batch_axes = inputs.iter().map(|input| input.batch_axis_position()).collect::<Vec<_>>();
        let axis_size = ArrayBatch::common_batch_size(inputs)?;
        if batch_axes[2..].iter().any(Option::is_some) {
            return Ok(batch_by_item_expansion(
                context,
                DYNAMIC_UPDATE_SLICE_OPERATION_NAME,
                self,
                inputs,
                axis_size.ok_or_else(|| ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` batching with mapped start indices requires a \
                         statically known mapped extent"
                    ),
                })?,
            )?
            .into());
        }
        let Some(batch_axis) = batch_axes[..2].iter().copied().flatten().next() else {
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into());
        };
        if inputs.len() == 2 {
            // No start indices means scalar replacement only after validating rank, data type, memory, and
            // placement. Otherwise this shortcut would silently accept malformed calls that the parent rejects.
            self.infer_output_types(&inputs.iter().map(ArrayBatch::unbatched_type).collect::<Vec<_>>(), &[])?;
            return Ok(vec![P::match_axis(context, &inputs[1], Axis::from(batch_axis))?].into());
        }
        let input = P::match_axis(context, &inputs[0], Axis::from(batch_axis))?;
        let update = P::match_axis(context, &inputs[1], Axis::from(batch_axis))?;
        let zero_index = ArrayBatch::replicated(inputs[2].value().zero_like()?);
        let mut lifted_inputs = vec![input, update];
        lifted_inputs.extend(inputs[2..].iter().cloned());
        lifted_inputs.insert(2 + batch_axis, zero_index);
        Ok(self
            .interpret_with_batch_axes(context, lifted_inputs.as_slice(), &[BatchAxis::from_position(batch_axis)])?
            .into())
    }
}

impl_differentiable_operation! {
    DynamicUpdateSliceOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType> + Zero<C::Value>,
        C::Operation: From<DynamicUpdateSliceOperation>,
        C::Value: DynamicUpdateSlice,
    {
        |operation, context, _driver, inputs| {
            // Forward-mode rule for [`DynamicUpdateSliceOperation`]: `dynamic_update_slice` is jointly linear in the
            // input and the update, while the scalar start indices are non-differentiated primal input edges, so the
            // tangent updates the input tangent with the update tangent at the same primal start indices. A zero input
            // and update tangent yields a typed zero output tangent.
            if inputs.len() < 2 {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
            }
            let input = &inputs[0];
            let update = &inputs[1];
            let primal_starts = inputs[2..].iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
            let allow_negative_indices = operation.allows_negative_indices();
            let primal = input.primal().dynamic_update_slice_with_negative_indices(
                update.primal(),
                &primal_starts,
                allow_negative_indices,
            )?;
            let tangent = if input.tangent().is_zero() && update.tangent().is_zero() {
                MaybeZero::Zero(primal.r#type().tangent()?)
            } else {
                let input_tangent = input.tangent().clone().materialize(context.tangent())?;
                let update_tangent = update.tangent().clone().materialize(context.tangent())?;
                MaybeZero::Value(
                    input_tangent.dynamic_update_slice_with_negative_indices(
                        &update_tangent,
                        &primal_starts
                            .into_iter()
                            .map(|value| context.primal_to_tangent(value))
                            .collect::<Result<Vec<_>, _>>()?,
                        allow_negative_indices,
                    )?,
                )
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType>
            + From<ZeroOperation<ArrayType>>
            + From<DynamicUpdateSliceOperation>
            + From<DynamicSliceOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // Partition-aware transpose rule for the primal [`DynamicUpdateSliceOperation`]. The scalar integer start
            // indices (inputs 2 onward) have no tangent space, so in a valid pushforward they are the known inputs and
            // the input and update (inputs 0 and 1) are the linear ones. The forward map `(t, u) ↦
            // dynamic_update_slice(t, u, start_indices)` splits the output cotangent into two contributions at the same
            // start indices: the input cotangent is the cotangent with the update window zeroed (a dynamic update-slice
            // writing zeros at the indices) and the update cotangent is the dynamic slice of the cotangent at the
            // update window. The transpose reads the known start indices from the pullback boundary and stages ordinary
            // dynamic slicing operations, so linearization retains the indices as regular SSA residuals. The start
            // indices receive structural zeros, and a zero output cotangent stays a structural zero.
            if inputs.len() < 2 {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
            }
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
            if let MaybeZero::Value(cotangent) = &outputs[0] {
                if !accumulators[0].is_needed() && !accumulators[1].is_needed() {
                    return Ok(());
                }
                // Both contributions need the update's static shape: the input cotangent zeroes a window of that
                // shape and the update cotangent slices exactly that window.
                let update_sizes = inputs[1].r#type()
                    .shape()
                    .dimensions()
                    .iter()
                    .enumerate()
                    .map(|(axis, size)| {
                        size.value().ok_or_else(|| TypeError::invalid(format!(
                            "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` transpose requires a static update shape \
                             but axis {axis} has size {size}"
                        )))
                    })
                    .collect::<Result<Vec<_>, TypeError>>()?;
                let start_indices = inputs[2..]
                    .iter()
                    .map(|input| {
                        // Integer indices have no tangent space and must be retained as known primal inputs.
                        input.as_known().cloned().ok_or_else(|| ProgramError::InvalidArgument {
                            message: format!("`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` transpose requires known start indices"),
                        })
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                if accumulators[0].is_needed() {
                    let zeros = MaybeZero::Zero(inputs[1].r#type().cotangent()?).materialize(&**context)?;
                    // Input cotangent: the output cotangent with the update window overwritten by zeros.
                    let mut input_cotangent_inputs = Vec::with_capacity(2 + start_indices.len());
                    input_cotangent_inputs.push(cotangent.clone());
                    input_cotangent_inputs.push(zeros);
                    input_cotangent_inputs.extend(start_indices.iter().cloned());
                    let input_cotangents = context.stage_operation(
                        DynamicUpdateSliceOperation::new()
                            .with_allow_negative_indices(operation.allows_negative_indices()),
                        Vec::new(),
                        input_cotangent_inputs.as_slice(),
                    )?;
                    check_count!("output", input_cotangents, 1, ProgramError);
                    accumulators[0]
                        .accumulate(context, MaybeZero::Value(input_cotangents.into_iter().next().unwrap()))?;
                }
                if accumulators[1].is_needed() {
                    // Update cotangent: the dynamic slice of the output cotangent at the update window.
                    let mut update_inputs = Vec::with_capacity(1 + start_indices.len());
                    update_inputs.push(cotangent.clone());
                    update_inputs.extend(start_indices);
                    let update_cotangents = context.stage_operation(
                        DynamicSliceOperation::new(update_sizes)
                            .with_allow_negative_indices(operation.allows_negative_indices()),
                        Vec::new(),
                        update_inputs.as_slice(),
                    )?;
                    check_count!("output", update_cotangents, 1, ProgramError);
                    accumulators[1].accumulate(
                        context,
                        MaybeZero::Value(
                            update_cotangents
                                .into_iter()
                                .next()
                                .unwrap()
                                .unalign_cotangent(&inputs[1].r#type().cotangent()?)?,
                        ),
                    )?;
                }
            }
            Ok(())
        }
    },
}

impl<C> MemberDifferentiableOperation<C> for DynamicUpdateSliceOperation
where
    C: Context<Type = ArrayIrType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation:
        From<DimensionSizeOperation> + From<LinearCallOperation<ArrayIrType>> + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: DifferentiableOperation<ProjectedContext<C, ArrayType>>
        + From<DynamicSliceOperation>
        + From<DynamicUpdateSliceOperation>
        + From<ZeroOperation<ArrayType>>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // A dynamic input may need exact extents to materialize a missing tangent; scalar starts also become ordinary
        // residuals. Updates have static shapes. Fully static inputs delegate to the homogeneous projected rule.
        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }
        let input = &inputs[0];
        let input_type = <&ArrayType>::try_from(input.primal().r#type().as_ref())?.clone();
        if input_type.shape().dimensions().iter().all(|dimension| matches!(dimension, Dimension::Static(_))) {
            let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(*self);
            return jvp_projected_operation(context, &operation, inputs);
        }

        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(*self);
        let mut primal_outputs = context.primal().bind(operation, Vec::new(), primal_inputs.as_slice())?;
        check_count!("output", primal_outputs, 1, ProgramError);
        let output_primal = primal_outputs.remove(0);
        let tangent_primal = context.primal_to_tangent(output_primal.clone())?;
        let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
        let input = &tangent_inputs[0];
        let update = &tangent_inputs[1];
        let start_indices = &tangent_inputs[2..];
        let tangent_context = context.tangent();
        if input.tangent().is_zero() && update.tangent().is_zero() {
            return Ok(vec![DifferentiationDual::new(
                output_primal.clone(),
                MaybeZero::Zero(tangent_primal.r#type().tangent()?),
            )?]);
        }

        // The integer starts are the ordinary primal residuals shared by the forward update and its two transpose
        // branches. Input extents are retained only when a missing input tangent must be materialized inside the
        // forward region; otherwise the output cotangent itself supplies the base geometry to the transpose.
        let mut residuals = LinearResiduals::new();
        let start_indices = residuals.retain_all(start_indices.iter().map(|index| index.primal().clone()));
        let input_is_live = !input.tangent().is_zero();
        let update_is_live = !update.tangent().is_zero();
        // The forward update and both transpose branches resolve the same raw starts under one policy.
        let allow_negative_indices = self.allows_negative_indices();
        let input_shape =
            (!input_is_live).then(|| residuals.retain_shape(tangent_context, input.primal())).transpose()?;
        let update_type = <&ArrayType>::try_from(update.primal().r#type().as_ref())?.clone();
        // Update extents were validated as static by the primal binding, so only the input can need shape residuals.
        let mut linear_values = Vec::with_capacity(usize::from(input_is_live) + usize::from(update_is_live));
        if let MaybeZero::Value(tangent) = input.tangent() {
            linear_values.push(tangent.clone());
        }
        if let MaybeZero::Value(tangent) = update.tangent() {
            linear_values.push(tangent.clone());
        }
        let forward_input_type = input_type.tangent()?;
        let forward_update_type = update_type.tangent()?;
        let forward_start_indices = start_indices.clone();
        let forward_input_shape = input_shape.clone();
        let transpose_start_indices = start_indices.clone();
        let transpose_update_type = update_type.cotangent()?;
        let update_sizes = if update_is_live {
            transpose_update_type
                .shape()
                .dimensions()
                .iter()
                .enumerate()
                .map(|(axis, size)| {
                    size.value().ok_or_else(|| {
                        TypeError::invalid(format!(
                            "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` transpose requires a static update shape \
                         but axis {axis} has size {size}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, TypeError>>()?
        } else {
            Vec::new()
        };
        let tangent = LinearCallOperation::stage(
            tangent_context,
            residuals.into_values(),
            linear_values,
            move |residuals, linear_inputs| {
                let forward_context = linear_inputs[0].dispatch_domain();
                let mut linear_index = 0;
                let input_tangent = if input_is_live {
                    let tangent = linear_inputs[linear_index].clone();
                    linear_index += 1;
                    tangent
                } else {
                    let extents = forward_input_shape.as_ref().unwrap().dynamic_dimensions(residuals);
                    forward_context
                        .bind(
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(ZeroOperation::new(
                                forward_input_type.clone(),
                            )),
                            Vec::new(),
                            extents.as_slice(),
                        )?
                        .remove(0)
                };
                let update_tangent = if update_is_live {
                    linear_inputs[linear_index].clone()
                } else {
                    forward_context
                        .bind(
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(ZeroOperation::new(
                                forward_update_type.clone(),
                            )),
                            Vec::new(),
                            &[],
                        )?
                        .remove(0)
                };
                let mut update_inputs = Vec::with_capacity(2 + forward_start_indices.len());
                update_inputs.extend([input_tangent, update_tangent]);
                update_inputs.extend(forward_start_indices.iter().map(|index| residuals[*index].clone()));
                forward_context.bind(
                    <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                        DynamicUpdateSliceOperation::new().with_allow_negative_indices(allow_negative_indices),
                    ),
                    Vec::new(),
                    update_inputs.as_slice(),
                )
            },
            move |residuals, output_cotangents| {
                let transpose_context = output_cotangents[0].dispatch_domain();
                let mut cotangents = Vec::with_capacity(usize::from(input_is_live) + usize::from(update_is_live));
                if input_is_live {
                    let update_zero = transpose_context
                        .bind(
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(ZeroOperation::new(
                                transpose_update_type.clone(),
                            )),
                            Vec::new(),
                            &[],
                        )?
                        .remove(0);
                    let mut input_cotangent_inputs = vec![output_cotangents[0].clone(), update_zero];
                    input_cotangent_inputs
                        .extend(transpose_start_indices.iter().map(|index| residuals[*index].clone()));
                    cotangents.push(
                        transpose_context
                            .bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                    DynamicUpdateSliceOperation::new()
                                        .with_allow_negative_indices(allow_negative_indices),
                                ),
                                Vec::new(),
                                input_cotangent_inputs.as_slice(),
                            )?
                            .remove(0),
                    );
                }
                if update_is_live {
                    let mut update_cotangent_inputs = vec![output_cotangents[0].clone()];
                    update_cotangent_inputs
                        .extend(transpose_start_indices.iter().map(|index| residuals[*index].clone()));
                    cotangents.push(
                        transpose_context
                            .bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                    DynamicSliceOperation::new(update_sizes)
                                        .with_allow_negative_indices(allow_negative_indices),
                                ),
                                Vec::new(),
                                update_cotangent_inputs.as_slice(),
                            )?
                            .remove(0),
                    );
                }
                Ok(cotangents)
            },
        )?
        .remove(0);
        Ok(vec![DifferentiationDual::new(output_primal, MaybeZero::Value(tangent))?])
    }
}

/// Represents the ability to overwrite a contiguous sub-array with an update value at start indices that are computed
/// at run time, with the semantics of StableHLO's
/// [`dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice) operation.
///
/// `input.dynamic_update_slice(update, start_indices)` replaces a block of `input` with `update`. A negative signed
/// start counts from the end of its axis: on an axis of extent `d`, a start `i < 0` becomes `i + d` once. The
/// effective start on axis `d` is then `clamp(0, start_indices[d], input_dimension[d] - update_dimension[d])`,
/// keeping the complete update in bounds, so `-1` places the update's last element at the end of the axis and a
/// start that is still negative after the single wrap clamps to zero. Unsigned and Boolean starts never wrap. All
/// starts must be scalar arrays with the same integer element data type, one per input axis.
///
/// The update must have the input's element data type and rank, with static dimensions. A static input axis must be
/// at least as large as its update axis. A dynamic input axis is accepted when its declared lower bound proves that
/// the update fits every admitted extent. Input, update, and starts must share a memory space. As with [`UpdateSlice`],
/// input and update must have compatible sharding and identical reduction state. The output preserves the input shape,
/// layout, and memory placement, and includes variation over manual axes contributed by the update or start indices.
/// Start indices cannot carry reduction state.
///
/// # Example
///
/// The following example shows how to use [`DynamicUpdateSlice`] in practice:
///
/// ```rust
/// # use ryft_core::{Array, ArrayType, DataType, DynamicUpdateSlice, ProgramError};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Overwrite the last two elements of the first row of a 2x3 matrix.
/// // Shapes: input [2, 3], update [1, 2], row and column [] (scalars) -> output [2, 3].
/// let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let update = Array::matrix(1, 2, vec![8.0, 9.0]).unwrap();
/// let row = Array::from_elements::<i32>(ArrayType::scalar(DataType::I32), &[0]).unwrap();
/// let column = Array::from_elements::<i32>(ArrayType::scalar(DataType::I32), &[1]).unwrap();
/// let output = input.dynamic_update_slice(&update, &[row, column])?;
/// assert_eq!(output.to_f64s(), vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait DynamicUpdateSlice: Sized {
    /// Overwrites the block of `self` starting at `start_indices` with `update`. Refer to the documentation of this
    /// trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `update`: Array written into the selected block. Its element data type and rank must match `self`; its
    ///     dimensions must be static and fit within every possible input extent.
    ///   - `start_indices`: Scalar integer arrays, one per input axis, all with the same element data type. Negative
    ///     signed values count from the end of their axis once; values beyond the last valid origin clamp to keep the
    ///     complete update in bounds.
    #[inline]
    fn dynamic_update_slice(&self, update: &Self, start_indices: &[Self]) -> Result<Self, ProgramError> {
        self.dynamic_update_slice_with_negative_indices(update, start_indices, true)
    }

    /// Overwrites the block of `self` starting at `start_indices` with `update` like
    /// [`dynamic_update_slice`](Self::dynamic_update_slice), selecting how negative signed starts are treated. With
    /// `allow_negative_indices` set to `true`, a negative start counts from the end of its axis exactly as
    /// [`dynamic_update_slice`](Self::dynamic_update_slice) describes; with `false`, negative starts are out of bounds
    /// and clamp to zero, which is StableHLO's native rule. Unsigned and Boolean starts are unaffected either way.
    fn dynamic_update_slice_with_negative_indices(
        &self,
        update: &Self,
        start_indices: &[Self],
        allow_negative_indices: bool,
    ) -> Result<Self, ProgramError>;

    /// Overwrites the block of `self` that starts at the runtime scalar `start` along `axis` and at zero along every
    /// other axis with `update`. This is [`dynamic_update_slice`](Self::dynamic_update_slice) with a zero start on
    /// every other axis, so `start` counts from the end of `axis` when negative and clamps so that the update fits,
    /// and the update's extents on the other axes need not match the input's.
    ///
    /// # Parameters
    ///
    ///   - `update`: Array written into the selected block, with the input's element data type and rank.
    ///   - `start`: Scalar integer array giving the start along `axis`.
    ///   - `axis`: Axis of the update; negative axes count backward from the input rank.
    fn dynamic_update_slice_in_axis<A: Into<Axis>>(
        &self,
        update: &Self,
        start: &Self,
        axis: A,
    ) -> Result<Self, ProgramError>
    where
        Self: Clone + Typed<Type = ArrayType> + ZeroLike,
    {
        let rank = self.r#type().rank();
        let axis = axis.into().normalize(rank).map_err(|error| TypeError::invalid(error.to_string()))?;
        let starts = (0..rank)
            .map(|input_axis| if input_axis == axis { Ok(start.clone()) } else { start.zero_like() })
            .collect::<Result<Vec<_>, _>>()?;
        self.dynamic_update_slice(update, &starts)
    }

    /// Overwrites the extent-one block of `self` at the runtime scalar `index` along `axis` with `update`, which may
    /// either carry the input's rank with extent one on `axis` or omit `axis` altogether. This is
    /// [`dynamic_update_slice_in_axis`](Self::dynamic_update_slice_in_axis) after inserting the missing axis into a
    /// rank-deficient update, so `index` counts from the end of `axis` when negative and clamps into bounds.
    ///
    /// # Parameters
    ///
    ///   - `update`: Array written at the selected position, of the input's rank or one less.
    ///   - `index`: Scalar integer array giving the position along `axis`.
    ///   - `axis`: Axis of the update; negative axes count backward from the input rank.
    fn dynamic_update_index_in_axis<A: Into<Axis>>(
        &self,
        update: &Self,
        index: &Self,
        axis: A,
    ) -> Result<Self, ProgramError>
    where
        Self: Clone + Typed<Type = ArrayType> + ZeroLike + Reshape,
    {
        let rank = self.r#type().rank();
        let axis = axis.into().normalize(rank).map_err(|error| TypeError::invalid(error.to_string()))?;
        let update_type = update.r#type();
        if update_type.rank() + 1 == rank {
            let mut dimensions = update_type.shape().dimensions().to_vec();
            dimensions.insert(axis, Dimension::Static(1));
            let update = update.reshape(Shape::new(dimensions))?;
            return self.dynamic_update_slice_in_axis(&update, index, axis);
        }
        self.dynamic_update_slice_in_axis(update, index, axis)
    }
}

impl DynamicUpdateSlice for ArrayType {
    fn dynamic_update_slice_with_negative_indices(
        &self,
        update: &Self,
        start_indices: &[Self],
        _allow_negative_indices: bool,
    ) -> Result<ArrayType, ProgramError> {
        validate_update_compatibility(DYNAMIC_UPDATE_SLICE_OPERATION_NAME, self, update)?;
        let rank = self.rank();
        if start_indices.len() != rank {
            return Err(TypeError::invalid(format!(
                "`{}` expects one start index per input axis ({}) but got {}",
                DYNAMIC_UPDATE_SLICE_OPERATION_NAME,
                rank,
                start_indices.len(),
            ))
            .into());
        }
        validate_start_index_types(DYNAMIC_UPDATE_SLICE_OPERATION_NAME, self, start_indices)?;
        for axis in 0..rank {
            let update_dimension = update.dimension(axis);
            let Dimension::Static(update_size) = update_dimension else {
                return Err(TypeError::invalid(format!(
                    "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` does not support dynamic update axis {axis} with size \
                        {update_dimension}; update shapes must be static",
                ))
                .into());
            };
            match self.dimension(axis) {
                Dimension::Static(input_size) if update_size > input_size => {
                    return Err(TypeError::invalid(format!(
                        "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` update axis {axis} has size {update_size} which \
                         exceeds input size {input_size}",
                    ))
                    .into());
                }
                Dimension::Dynamic(variable) if update_size > variable.bounds().lower() => {
                    return Err(TypeError::invalid(format!(
                        "`{}` update size {} exceeds the guaranteed minimum extent {} of dynamic axis {}",
                        DYNAMIC_UPDATE_SLICE_OPERATION_NAME,
                        update_size,
                        variable.bounds().lower(),
                        axis,
                    ))
                    .into());
                }
                _ => {}
            }
        }
        // The output is distributed like the input (the update is written in place); the input's placement
        // and reduction state carry through, with the update's varying-manual axes folded in.
        let sharding = update_slice_output_sharding(self, update, DYNAMIC_UPDATE_SLICE_OPERATION_NAME)?;
        let output_type = self.clone().with_sharding(sharding).map_err(|error| {
            TypeError::invalid(format!("`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` output type is invalid: {error}"))
        })?;
        indexed_slice_output_type(output_type, start_indices, DYNAMIC_UPDATE_SLICE_OPERATION_NAME)
    }
}

impl DynamicUpdateSlice for Array {
    fn dynamic_update_slice_with_negative_indices(
        &self,
        update: &Self,
        start_indices: &[Self],
        allow_negative_indices: bool,
    ) -> Result<Self, ProgramError> {
        let index_types = start_indices.iter().map(|index| index.r#type().into_owned()).collect::<Vec<_>>();
        let output_type = self.r#type().dynamic_update_slice(update.r#type().as_ref(), &index_types)?;
        let input_shape = self.r#type().static_shape().unwrap();
        let update_shape = update.r#type().static_shape().unwrap();
        let starts =
            Self::clamped_start_indices(start_indices, &input_shape, update_shape.dimensions(), allow_negative_indices);
        let output = self.clone().replace_block(update, starts.as_slice());
        // Type inference preserves the input's shape, element type, memory, and physical layout; only sharding
        // metadata can change. Apply that validated metadata without broadcasting and copying the updated bytes.
        Ok(Self::new_unchecked(output_type, output.shared_storage().clone()))
    }
}

impl<A: DynamicUpdateSlice + Value<Type = ArrayType>> DynamicUpdateSlice for ArrayIrValue<A> {
    fn dynamic_update_slice_with_negative_indices(
        &self,
        update: &Self,
        start_indices: &[Self],
        allow_negative_indices: bool,
    ) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let update = <Self as ValueProjection<ArrayType>>::projected(update)?;
        let start_indices = start_indices
            .iter()
            .cloned()
            .map(ValueProjection::<ArrayType>::into_projected)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::Array(input.dynamic_update_slice_with_negative_indices(
            update,
            &start_indices,
            allow_negative_indices,
        )?))
    }
}

impl<V: Value<Type = ArrayType>> DynamicUpdateSlice for V
where
    V::DispatchDomain: Context<Type = ArrayType, Operation: From<DynamicUpdateSliceOperation>>,
{
    fn dynamic_update_slice_with_negative_indices(
        &self,
        update: &Self,
        start_indices: &[Self],
        allow_negative_indices: bool,
    ) -> Result<Self, ProgramError> {
        // Any context-carrying value dynamic-update-slices by binding a [`DynamicUpdateSliceOperation`] through its own
        // context. The `From<DynamicUpdateSliceOperation>` bound makes this disjoint from the eager value types (whose
        // context operation is `ConstantOperation`), so it covers the transform tracers without conflicting with the
        // concrete implementations.
        let mut inputs = vec![self.clone(), update.clone()];
        inputs.extend(start_indices.iter().cloned());
        let operation = DynamicUpdateSliceOperation::new().with_allow_negative_indices(allow_negative_indices);
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl Array {
    /// Extracts the in-band scalar start indices of a dynamic slicing operation and resolves them against the input
    /// extents: with `allow_negative_indices`, a negative signed start on axis `d` is first replaced by
    /// `start_indices[d] + input_dimension[d]`, and every start is then clamped to
    /// `[0, input_dimension[d] - block_sizes[d]]` as StableHLO does.
    fn clamped_start_indices(
        start_indices: &[Array],
        input_shape: &StaticShape,
        block_sizes: &[usize],
        allow_negative_indices: bool,
    ) -> Vec<usize> {
        start_indices
            .iter()
            .enumerate()
            .map(|(axis, index)| {
                // Input validation guarantees a scalar integer. Preserve unsigned extremes until after clamping.
                let mut raw: i128 = index.concretize().unwrap();
                let extent = input_shape[axis] as i128;
                if allow_negative_indices && raw < 0 && wraps_negative_start(index.r#type().data_type()) {
                    raw += extent;
                }
                raw.clamp(0, extent - block_sizes[axis] as i128) as usize
            })
            .collect()
    }
}

/// Returns whether a start index of `data_type` can be negative and therefore counts from the end of its axis when a
/// dynamic slicing operation allows negative indices. Unsigned starts cannot be negative, and Boolean starts are
/// predicate carriers that never wrap.
fn wraps_negative_start(data_type: DataType) -> bool {
    !data_type.is_unsigned() && data_type != DataType::I1
}

/// Counts the negative entries of a signed start array from the end of an axis whose extent is the scalar integer
/// value `extent`, leaving the other entries unchanged, so that a clamp-only consumer such as a gather resolves them
/// like the unbatched kernel does.
fn wrap_negative_starts<V>(starts: &V, extent: &V) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + ZeroLike + Broadcast + Compare + Add + Select + ConvertElementType + TransferToMemory,
{
    let starts_type = starts.r#type().into_owned();
    let extent = extent
        .convert_element_type(starts_type.data_type())?
        .transfer_to_memory(starts_type.memory())?
        .broadcast(starts_type, &[])?;
    let negative = starts.less_than(&starts.zero_like()?)?;
    V::select(&negative, &starts.add(&extent)?, starts)
}

/// Returns the runtime extent of the dynamic `axis` of `source` as a scalar `i64` value without any dimension
/// operation, which the homogeneous array family does not have: a reduction of ones over every other axis has that
/// extent, and a second reduction of ones over the result counts it.
fn dynamic_axis_extent<V: Value<Type = ArrayType> + ConvertElementType + OneLike + Reduce>(
    source: &V,
    axis: usize,
) -> Result<V, ProgramError> {
    let other_axes = (0..source.r#type().rank()).filter(|source_axis| *source_axis != axis).collect::<Vec<_>>();
    let ones = source.convert_element_type(DataType::I64)?.one_like()?;
    Ok(ones.reduce(&other_axes, ReductionKind::Sum).one_like()?.reduce(&[0], ReductionKind::Sum))
}

/// Validates that a [`DynamicSlice`] call supplies one start index and one size per input axis, naming the list
/// that is wrong.
fn validate_dynamic_slice_bound_counts(rank: usize, start_count: usize, size_count: usize) -> Result<(), ProgramError> {
    for (name, count) in [("start index", start_count), ("size", size_count)] {
        if count != rank {
            return Err(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` expects one {name} per input axis ({rank}) but got {count}"
            ))
            .into());
        }
    }
    Ok(())
}

/// Validates the scalar integer start-index input types of a dynamic slicing operation. Each index type must be a
/// rank-0 integer type, all indices must share one integer type, and every index must reside in the input memory space.
/// The `operation_name` parameter selects the reported operation name because this helper serves both
/// [`DynamicSliceOperation`] and [`DynamicUpdateSliceOperation`].
fn validate_start_index_types(
    operation_name: &'static str,
    input_type: &ArrayType,
    index_types: &[ArrayType],
) -> Result<(), ProgramError> {
    for (index, index_type) in index_types.iter().enumerate() {
        if index_type.rank() != 0 || !index_type.data_type().is_integer() {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` start index {index} must be a scalar integer but has type `{index_type}`",
            ))
            .into());
        }
        if index_type.memory() != input_type.memory() {
            return Err(TypeError::invalid(format!(
                "`{}` input and start indices must share one memory space but start index {} resides in `{}` and the \
                 input resides in `{}`",
                operation_name,
                index,
                index_type.memory(),
                input_type.memory(),
            ))
            .into());
        }
        if let Some(sharding) = index_type.sharding() {
            if !sharding.unreduced_axes().is_empty() || !sharding.reduced_axes().is_empty() {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` start indices must not carry reduction state",
                ))
                .into());
            }
            if let Some(input_sharding) = input_type.sharding() {
                if sharding.mesh() != input_sharding.mesh() {
                    return Err(TypeError::invalid(format!(
                        "`{operation_name}` input and start indices must use the same mesh",
                    ))
                    .into());
                }
                if sharding.varying_manual_axes().iter().any(|axis| {
                    input_sharding.unreduced_axes().contains(axis) || input_sharding.reduced_axes().contains(axis)
                }) {
                    return Err(TypeError::invalid(format!(
                        "`{operation_name}` start indices must be invariant when the input carries reduction state",
                    ))
                    .into());
                }
            }
        }
        if index_type.data_type() != index_types[0].data_type() {
            return Err(TypeError::invalid(format!(
                "`{}` start indices must share one integer type but index {} has type `{}` and index 0 has type `{}`",
                operation_name, index, index_type, index_types[0],
            ))
            .into());
        }
    }
    Ok(())
}

/// Carries the distribution of dynamic start indices into the result of a slicing operation. Index values are discrete
/// control inputs: their reduction state is invalid, while variation over manual mesh axes makes the selected or
/// updated result vary over the same axes.
fn indexed_slice_output_type(
    output_type: ArrayType,
    indices: &[ArrayType],
    operation_name: &'static str,
) -> Result<ArrayType, ProgramError> {
    // Every sharded start index must share one mesh with the output and with each other, whether or not it changes the
    // output placement, so the check does not depend on index order or on whether the array is already sharded.
    let mut mesh = output_type.sharding().map(|sharding| sharding.mesh().clone());
    for index_type in indices {
        let Some(index_sharding) = index_type.sharding() else {
            continue;
        };
        match &mesh {
            Some(mesh) if mesh != index_sharding.mesh() => {
                return Err(
                    TypeError::invalid(format!("`{operation_name}` start indices must use the same mesh")).into()
                );
            }
            Some(_) => {}
            None => mesh = Some(index_sharding.mesh().clone()),
        }
    }
    // Only variation over manual axes changes the placement; a replicated index leaves an unsharded output unsharded.
    let index_varying_manual_axes = indices
        .iter()
        .filter_map(|index_type| index_type.sharding())
        .flat_map(|sharding| sharding.varying_manual_axes().iter().cloned())
        .collect::<BTreeSet<_>>();
    if index_varying_manual_axes.is_empty() {
        return Ok(output_type);
    }
    let output_sharding = match output_type.sharding() {
        Some(sharding) => sharding.clone(),
        None => Sharding::new(mesh.unwrap(), vec![ShardingDimension::Replicated; output_type.rank()])
            .map_err(|error| TypeError::invalid(format!("`{operation_name}` output sharding is invalid: {error}")))?,
    };
    let varying_manual_axes =
        output_sharding.varying_manual_axes().union(&index_varying_manual_axes).cloned().collect::<Vec<_>>();
    let sharding = output_sharding
        .with_varying_manual_axes(varying_manual_axes)
        .map_err(|error| TypeError::invalid(format!("`{operation_name}` output sharding is invalid: {error}")))?;
    output_type
        .with_sharding(Some(sharding))
        .map_err(|error| TypeError::invalid(format!("`{operation_name}` output type is invalid: {error}")).into())
}

/// Returns the output [`Sharding`] for an in-place update ([`UpdateSlice`] / [`DynamicUpdateSlice`]). Because the
/// update is written into the input without resharding, the two must agree on placement and reduction state wherever an
/// [`Explicit`](crate::arrays::MeshAxisType::Explicit) mesh axis is involved; differences confined to `Manual`/`Auto`
/// axes are tolerated (left to `shard_map` / the compiler). The output keeps the input's sharding, except that the
/// update's [`varying_manual_axes`](Sharding::varying_manual_axes) are unioned in: the written region may vary over
/// manual axes the input does not, so the result does too. An unsharded input acquires replicated placement on the
/// update's mesh when that is needed to represent the update's manual-axis variation.
fn update_slice_output_sharding(
    input: &ArrayType,
    update: &ArrayType,
    operation_name: &'static str,
) -> Result<Option<Sharding>, TypeError> {
    let input_unreduced = input.sharding().map(Sharding::unreduced_axes).cloned().unwrap_or_default();
    let input_reduced = input.sharding().map(Sharding::reduced_axes).cloned().unwrap_or_default();
    let update_unreduced = update.sharding().map(Sharding::unreduced_axes).cloned().unwrap_or_default();
    let update_reduced = update.sharding().map(Sharding::reduced_axes).cloned().unwrap_or_default();
    if input_unreduced != update_unreduced || input_reduced != update_reduced {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` input and update must carry identical reduction state",
        )));
    }
    let Some(input_sharding) = input.sharding() else {
        return update
            .sharding()
            .filter(|sharding| !sharding.varying_manual_axes().is_empty())
            .map(|sharding| {
                Sharding::replicated(sharding.mesh().clone(), input.rank())
                    .with_varying_manual_axes(sharding.varying_manual_axes().clone())
                    .map_err(|error| {
                        TypeError::invalid(format!("`{operation_name}` output sharding is invalid: {error}"))
                    })
            })
            .transpose();
    };
    let Some(update_sharding) = update.sharding() else {
        return Ok(Some(input_sharding.clone()));
    };
    if input_sharding.mesh() != update_sharding.mesh() {
        return Err(TypeError::invalid(format!("`{operation_name}` input and update must use the same mesh")));
    }
    if input_sharding.conflicts_on_explicit_axes_with(update_sharding) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` input and update must be sharded identically, but got `{input_sharding}` and \
            `{update_sharding}`"
        )));
    }
    if update_sharding.varying_manual_axes().is_subset(input_sharding.varying_manual_axes()) {
        return Ok(Some(input_sharding.clone()));
    }
    let varying_manual_axes = input_sharding
        .varying_manual_axes()
        .union(update_sharding.varying_manual_axes())
        .cloned()
        .collect::<Vec<_>>();
    input_sharding
        .clone()
        .with_varying_manual_axes(varying_manual_axes)
        .map(Some)
        .map_err(|error| TypeError::invalid(format!("`{operation_name}` output sharding is invalid: {error}")))
}

/// Validates the input/update agreement shared by the static and dynamic update slices: equal element data types, one
/// memory space, and equal ranks. Start-index arity is checked by each caller with its own wording.
fn validate_update_compatibility(
    operation_name: &'static str,
    input: &ArrayType,
    update: &ArrayType,
) -> Result<(), ProgramError> {
    if input.data_type() != update.data_type() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` input data type `{}` does not match update data type `{}`",
            input.data_type(),
            update.data_type(),
        ))
        .into());
    }
    if input.memory() != update.memory() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` input and update must share one memory space but reside in `{}` and `{}`",
            input.memory(),
            update.memory(),
        ))
        .into());
    }
    if update.rank() != input.rank() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` update has rank {} but input has rank {}",
            update.rank(),
            input.rank(),
        ))
        .into());
    }
    Ok(())
}

/// Applies a single-output `operation` independently per batch item and restacks the results along a fresh leading
/// batch axis: every input is realigned so any mapped batch axis sits at the leading physical axis, item `item` of each
/// batched input is selected with [`Slice::index_axis`] (replicated inputs are used whole), and the per-item outputs
/// are expanded with a replicated leading axis and concatenated. This is the fallback for batch-varying start
/// indices, which cannot ride along structurally, and it stages `O(axis_size)` operations because everything goes
/// through the value capability traits in both eager and tracing contexts. For an empty batch, it infers the per-item
/// output type and synthesizes the correctly typed empty packed result without interpreting a nonexistent item; this
/// requires the operation's output to have the input's rank and element type with extents within the input's, which
/// every slicing operation satisfies. Non-empty explicitly sharded mapped inputs are resharded to replicated placement
/// before item extraction, and the completed replicated result is resharded once to the context's mapped placement.
/// This avoids assigning a nontrivial sharding to the extent-one slices used internally by the expansion.
fn batch_by_item_expansion<C, O, P: ArrayExtentBatchingPolicy<C>>(
    context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
    operation_name: &'static str,
    operation: &O,
    inputs: &[ArrayBatch<C::Value>],
    axis_size: usize,
) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    C::Value: Broadcast + Transpose + Slice + Reshape + Concatenate + Reshard,
    O: Operation<Type = ArrayType> + InterpretableOperation<C>,
{
    if inputs.is_empty() {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    }
    if axis_size == 0 {
        let input_types = inputs.iter().map(ArrayBatch::unbatched_type).collect::<Vec<_>>();
        let mut output_types = operation.infer_output_types(input_types.as_slice(), &[])?;
        check_count!("output", output_types, 1, ProgramError);
        let output_type = output_types.remove(0);
        // The callers are slicing operations: their output has the input's rank and element type within its extents.
        // Reuse an empty mapped input and slice its geometry, avoiding a non-empty scalar-zero construction for
        // formats such as `F8E8M0FNU` that cannot represent zero.
        let input = P::match_axis(context, &inputs[0], Axis::from(0))?;
        if input.unbatched_type() == output_type {
            return Ok(vec![input]);
        }
        let output_shape = output_type.static_shape().ok_or_else(|| {
            TypeError::invalid(format!(
                "`{operation_name}` batching over an empty batch requires a statically known result shape"
            ))
        })?;
        let mut limits = vec![0];
        limits.extend_from_slice(output_shape.as_slice());
        let output = input.value().slice(&vec![0; limits.len()], &limits, &vec![1; limits.len()])?;
        let output_type = output_type.batched(0, Dimension::Static(0), context.axis_sharding().clone())?;
        let output = output.broadcast(output_type, &(0..limits.len()).collect::<Vec<_>>())?;
        return Ok(vec![ArrayBatch::new(output, BatchAxis::new(0))?]);
    }
    let aligned = inputs
        .iter()
        .map(|input| {
            let aligned = input.move_axis(0)?;
            let aligned_type = aligned.r#type();
            let (Some(0), Some(sharding)) = (aligned.batch_axis_position(), aligned_type.sharding()) else {
                return Ok(aligned);
            };
            if sharding.dimensions()[0] == ShardingDimension::Replicated {
                return Ok(aligned);
            }
            // Slicing one global batch item cannot retain a nontrivial Explicit placement on its new extent-one
            // dimension. Replicate the packed input once, run the expansion over replicated slices, and restore the
            // mapped placement once on the completed output accumulator.
            let mut dimensions = sharding.dimensions().to_vec();
            dimensions[0] = ShardingDimension::Replicated;
            let replicated = sharding
                .with_dimensions(dimensions)
                .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
            let value = aligned.value().reshard(&replicated);
            ArrayBatch::new(value, BatchAxis::new(0))
        })
        .collect::<Result<Vec<_>, BatchingError>>()?;
    // Concatenate singleton-axis items once. Repeated immutable updates would copy the entire eager output for
    // every item. Keep the new axis replicated until the complete batch can carry its mapped sharding.
    let items = (0..axis_size)
        .map(|item| {
            let item_inputs = aligned
                .iter()
                .map(|input| {
                    if input.batch_axis().is_replicated() {
                        return Ok(input.value().clone());
                    }
                    let input_type = input.r#type();
                    if input_type.static_shape().is_none() {
                        return Err(TypeError::invalid(format!(
                            "`{operation_name}` per-item expansion requires static batched input types but got \
                         {input_type}",
                        ))
                        .into());
                    }
                    input.value().index_axis(0, item, false)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut outputs = operation.interpret(context.parent(), &EmptyRegionDriver, item_inputs.as_slice())?;
            check_count!("output", outputs, 1, ProgramError);
            outputs.remove(0).expand_dimensions(0)
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let accumulator = C::Value::concatenate(&items, 0)?;
    let batch_dimension = context.axis_sharding();
    let accumulator = match accumulator.r#type().sharding() {
        Some(sharding) if sharding.dimensions().first() != Some(batch_dimension) => {
            let mut dimensions = sharding.dimensions().to_vec();
            dimensions[0] = batch_dimension.clone();
            let sharding = sharding
                .with_dimensions(dimensions)
                .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
            accumulator.reshard(&sharding)
        }
        _ => accumulator,
    };
    let stacked = ArrayBatch::new(accumulator, Some(0))?;
    Ok(vec![stacked])
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayReference,
        ArrayReferenceDischarge, DataType, DimensionBounds, DimensionError, DimensionType, DimensionValue,
        DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, RaggedAxis, Sharding,
        ShardingDimension, StridedLayout, f8e8m0fnu, i4,
    };
    use crate::batching::{BatchAxis, BatchingContext, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        CotangentDestination, CotangentDestinationKind, CotangentSeed, DifferentiationError, differentiate_at,
    };
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::constants::constant::Constant;
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::operations::manipulation::concatenation::Concatenate;
    use crate::operations::manipulation::conversions::ConvertElementTypeOperation;
    use crate::operations::manipulation::padding::Pad;
    use crate::operations::math::mul::MulOperation;
    use crate::operations::math::reduce::{Reduce, ReduceOperation, ReductionKind};
    use crate::operations::references::{ReferenceNew, ReferenceRead};
    use crate::parameters::Placeholder;
    use crate::programs::{
        EmptyRegionDriver, ProgramBuilder, ProgramError, ReferenceDischargeContext, ReferenceDischargeValue,
        ReferenceDischargeableOperation, Typed,
    };
    use crate::tracing::Trace;

    use super::*;

    /// Returns a `[2, 4]` batch mapped at axis `0` whose data axis is declared bounded-ragged, which no slicing rule
    /// supports.
    fn ragged_batch() -> ArrayBatch<Array> {
        let ragged_axis = RaggedAxis::new(
            1,
            Array::vector(vec![1_i64, 3]).unwrap(),
            DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap()),
            vec![0],
        );
        ArrayBatch::new(Array::matrix(2, 4, vec![1.0; 8]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![ragged_axis])
            .unwrap()
    }

    /// Lifts a scalar `i32` index constant into the trace or differentiation context that `exemplar` belongs to.
    fn index_constant<V>(exemplar: &V, value: i32) -> V
    where
        V: Value<Type = ArrayType>,
        V::DispatchDomain: Context<Constant = Array>,
    {
        exemplar.dispatch_domain().lift(Array::scalar(value).unwrap()).unwrap()
    }

    /// Traces `batch(|(input, start)| input.dynamic_slice(&[start], &[2]), ...)` over a source of `input_type` mapped
    /// at `input_axis` and `i32[size]` start indices mapped at axis `0`, returning the staged program's rendering.
    fn trace_batched_dynamic_slice(input_type: ArrayType, input_axis: BatchAxis, size: usize) -> String {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(input, starts)| {
                batch(
                    |(input, start)| input.dynamic_slice(&[start], &[2]),
                    (input, starts),
                    (input_axis, BatchAxis::new(0)),
                    BatchAxis::new(0),
                    None,
                )
                .map_err(ProgramError::from)
            },
            (input_type, ArrayType::new_static(DataType::I32, [size])),
        )
        .unwrap();
        program.to_string()
    }

    /// Returns a batch-varying scalar integer index batch carrying one start index per batch item, mapped at axis `0`.
    fn batch_varying_indices(values: Vec<i32>) -> ArrayBatch<Array> {
        let length = values.len();
        let value = Array::from_elements::<i32>(
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(length)])),
            &values,
        )
        .unwrap();
        ArrayBatch::new(value, Some(0)).unwrap()
    }

    #[test]
    fn test_slice() {
        let operation = SliceOperation::new(vec![1, 1], vec![2, 3]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), SLICE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "slice [start_indices=[1, 1], limit_indices=[2, 3]]");
        assert_eq!(operation.start_indices(), &[1, 1]);
        assert_eq!(operation.limit_indices(), &[2, 3]);

        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        // Program rendering uses the canonical operation name and includes the captured indices.
        let mut builder = ProgramBuilder::<Array, SliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, Vec::new(), vec![program_input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[1, 2] = slice [start_indices=[1, 1], limit_indices=[2, 3]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_slice_with_strides() {
        // Strided operations carry their strides through the builder, accessors, rendering, and inference: the output
        // dimension per axis is `ceil((limit - start) / stride)`.
        let strided = SliceOperation::new(vec![1], vec![6]).with_strides(vec![2]).unwrap();
        assert_eq!(strided.strides(), &[2]);
        assert_eq!(format!("{strided}"), "slice [start_indices=[1], limit_indices=[6], strides=[2]]");
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(6)]));
        assert_eq!(
            strided.infer_output_types(std::slice::from_ref(&vector_type), &[]),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))]),
        );
    }

    #[test]
    fn test_slice_type_inference() {
        let operation = SliceOperation::new(vec![1, 1], vec![2, 3]);
        // Type inference validates the slice bounds and returns the sliced type, and the type-level (abstract)
        // capability backs it without consuming the borrowed input type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)]));
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(3),
            ]),
        );
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
                    input_types = [ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))],
                    error = format!("`{SLICE_OPERATION_NAME}` `start_indices` has length 2 but input has rank 1"),
                },
                {
                    input_types = [dynamic_type],
                    error = format!(
                        "`{SLICE_OPERATION_NAME}` limit index 2 exceeds the guaranteed minimum extent 0 of dynamic \
                         axis 0"
                    ),
                },
            ],
        );
        assert_eq!(input_type.slice(&[1, 1], &[2, 3], &[1, 1]), Ok(output_type.clone()));

        // Malformed index lists and out-of-bounds windows report precise operation errors.
        check_operation_type_inference!(
            operation = SliceOperation::new(vec![0, 0], vec![2]),
            cases = [{
                input_types = [input_type.clone()],
                error = format!("`{SLICE_OPERATION_NAME}` `limit_indices` has length 1 but input has rank 2"),
            }],
        );
        check_operation_type_inference!(
            operation = SliceOperation::new(vec![2, 0], vec![1, 3]),
            cases = [{
                input_types = [input_type.clone()],
                error = format!("`{SLICE_OPERATION_NAME}` start index 2 is greater than limit index 1 at axis 0"),
            }],
        );
        check_operation_type_inference!(
            operation = SliceOperation::new(vec![0, 0], vec![2, 4]),
            cases = [{
                input_types = [input_type.clone()],
                error = format!("`{SLICE_OPERATION_NAME}` limit index 4 is out of bounds for axis 1 with size 3"),
            }],
        );

        // Stride validation belongs to the constructor and to the type-level capability, which reject a stride list of
        // the wrong length and zero strides before inference sees them.
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2, 3]).with_strides(vec![2]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{SLICE_OPERATION_NAME}` `strides` has length 1 but `start_indices` has length 2"
            )))),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2, 3]).with_strides(vec![1, 0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{SLICE_OPERATION_NAME}` strides must be at least 1 but axis 1 has stride 0"
            )))),
        );
        assert_eq!(
            input_type.slice(&[0, 0], &[2, 3], &[1]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{SLICE_OPERATION_NAME}` `strides` has length 1 but input has rank 2"
            )))),
        );
        assert_eq!(
            input_type.slice(&[0, 0], &[2, 3], &[1, 0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{SLICE_OPERATION_NAME}` strides must be at least 1 but axis 1 has stride 0"
            )))),
        );

        // A slice operation cannot own nested regions.
        assert_eq!(
            SliceOperation::new(vec![], vec![]).infer_output_types(
                &[ArrayType::scalar(DataType::F32)],
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_slice_reference_discharge() {
        // Replay preserves the complete slicing payload and its output type. Shared replay and reference rejection
        // are covered by the reference-discharge macro tests.
        let expected = SliceOperation::new(vec![0, 0], vec![2, 3]).with_strides(vec![1, 2]).unwrap();
        let operation = ArrayIrOperation::Array(ArrayOperation::Slice(expected.clone()));
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = ReferenceDischargeContext::<_, ArrayReferenceDischarge>::new(trace.clone());
        let inputs = [ReferenceDischargeValue::Value(trace.input(ArrayType::new_static(DataType::F64, [2, 3]).into()))];
        let outputs = operation.discharge_references(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        let ReferenceDischargeValue::Value(output) = &outputs[0] else {
            panic!("expected a value carrier but got {}", outputs[0]);
        };
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 2])));
        let builder = trace.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        let ArrayIrOperation::Array(ArrayOperation::Slice(staged)) = builder.instructions()[0].operation() else {
            panic!("expected a staged slice");
        };
        assert_eq!(staged, &expected);
    }

    #[test]
    fn test_slice_interpretation() {
        let operation = SliceOperation::new(vec![1, 1], vec![2, 3]);
        let output_type = ArrayType::new_static(DataType::F64, [1, 2]);
        // Interpretation copies the selected block out of the row-major payload.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![5.0, 6.0]);

        // Empty slices produce empty payloads and rank-0 slices pass through.
        let empty = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .slice(&[1, 1], &[1, 3], &[1, 1])
            .unwrap();
        assert_eq!(empty.to_f64s(), Vec::<f64>::new());
        let scalar = Array::scalar(42.0).unwrap().slice(&[], &[], &[]).unwrap();
        assert_eq!(scalar.to_f64s(), vec![42.0]);

        // A slice can cross concatenation boundaries and retain interior padding in the resulting dense array.
        let left = Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[1.0_f32, 2.0]).unwrap();
        let right = Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[3.0_f32]).unwrap();
        let padding = Array::from_elements(ArrayType::scalar(DataType::F32), &[-1.0_f32]).unwrap();
        let padded = Array::concatenate([&left, &right], 0).unwrap().pad(&padding, &[1], &[1], &[1]).unwrap();
        assert_eq!(
            padded.slice(&[1], &[6], &[1]),
            Ok(Array::from_elements(ArrayType::new_static(DataType::F32, [5]), &[1.0_f32, -1.0, 2.0, -1.0, 3.0])
                .unwrap()),
        );

        let strided = SliceOperation::new(vec![1], vec![6]).with_strides(vec![2]).unwrap();

        // Strided interpretation keeps the elements at `start + i * stride`.
        let vector = Array::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let strided_output = strided
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&vector))
            .unwrap();
        assert_eq!(strided_output[0].to_f64s(), vec![1.0, 3.0, 5.0]);

        // A stride larger than the sliced extent keeps a single element, and `start == limit` keeps none.
        let single = Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap().slice(&[1], &[4], &[5]).unwrap();
        assert_eq!(single.to_f64s(), vec![1.0]);
        let strided_empty = Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap().slice(&[2], &[2], &[2]).unwrap();
        assert_eq!(*strided_empty.r#type(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)])));
        assert_eq!(strided_empty.to_f64s(), Vec::<f64>::new());
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
    }

    #[test]
    fn test_slice_partial_evaluation() {
        // Check standard partial evaluation with known and residual inputs.
        let input = Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let expected = Array::vector(vec![1.0, 2.0]).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = SliceOperation::new(vec![1], vec![3]),
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
    }

    #[test]
    fn test_slice_batching() {
        // Batching slices each item without slicing the mapped axis.
        check_operation_batching!(
            @exact,
            operation = SliceOperation::new(vec![1], vec![3]),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        4,
                        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    ).unwrap())],
                    outputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 2.0, 5.0, 6.0]).unwrap())],
                },
                {
                    inputs = [(@replicated, Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap())],
                    outputs = [(@replicated, Array::vector(vec![1.0, 2.0]).unwrap())],
                },
            ],
        );

        // Bounded ragged inputs are rejected before any window arithmetic, and the arity is validated right after.
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        assert!(matches!(
            SliceOperation::new(vec![1], vec![3]).batch(&context, &EmptyRegionDriver, &[ragged_batch()]),
            Err(BatchingError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!("`{SLICE_OPERATION_NAME}` does not support bounded ragged array inputs"),
        ));
        assert_eq!(
            SliceOperation::new(vec![1], vec![3]).batch(&context, &EmptyRegionDriver, &[]).unwrap_err(),
            BatchingError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
    }

    #[test]
    fn test_slice_batching_sharding() {
        // Static slice limits cannot encode a symbolic mapped extent; reject it instead of unwrapping a size.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let extent = trace.input(DimensionType::from(items.clone()).into());
        let input = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(items), Dimension::Static(3)])).into(),
        );
        let input = <_ as ValueProjection<ArrayType>>::into_projected(input).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace),
            extent,
        );
        assert_eq!(
            SliceOperation::new(vec![0], vec![2])
                .batch(&context, &EmptyRegionDriver, &[ArrayBatch::new(input.clone(), BatchAxis::new(0)).unwrap()],)
                .unwrap_err(),
            BatchingError::DynamicBatchAxis { r#type: Box::new(input.r#type().into_owned()), axis: Axis::from(0) },
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        // The full input is [2 (batch), 4]: the batch axis is replicated and the data axis is sharded over `x`.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                    .unwrap(),
            )
            .unwrap();
        // Each batch item slices its `x`-sharded [4] vector to [2] (2 is divisible by the `x` mesh-axis size, so the
        // slice keeps the sharding); batching restores the replicated batch axis, so the staged slice's output stays
        // sharded.
        let (output_type, _program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x| Ok(batch(|item| item.slice(&[0], &[2], &[1]), x, BatchAxis::new(0), BatchAxis::new(0), None).unwrap()),
            input_type,
        )
        .unwrap();
        assert_eq!(
            output_type.sharding().unwrap().dimensions(),
            &[ShardingDimension::Replicated, ShardingDimension::sharded(["x"])],
        );
    }

    #[test]
    fn test_slice_differentiation() {
        // Static slicing is linear; check both the JVP and the unit- and non-unit-stride pullbacks.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = SliceOperation::new(vec![1], vec![3]),
            cases = [{
                primals = [Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap()],
                tangents = [Array::vector(vec![4.0, 5.0, 6.0, 7.0]).unwrap()],
                primal_outputs = [Array::vector(vec![1.0, 2.0]).unwrap()],
                tangent_outputs = [Array::vector(vec![5.0, 6.0]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_slice_differentiation_array_ir() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(3, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Slice(SliceOperation::new(vec![1], vec![3]))),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        // The dynamically shaped input routes through one residual-carrying linear call whose single residual is the
        // retained input extent, so the transpose region can rebuild the zero it writes the cotangent into.
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:dimension<extent ∈ [3, 6)> .
                let %2:f64[2] = linear_call [residual_count=1] %1 %0 [
                    forward={
                        lambda %0:dimension<extent ∈ [3, 6)>, %1:f64[extent] .
                        let %2:f64[2] = slice [start_indices=[1], limit_indices=[3]] %1
                        in (%2)
                    },
                    transpose={
                        lambda %0:dimension<extent ∈ [3, 6)>, %1:f64[2] .
                        let %2:f64[extent] = zero [type=f64[extent]] %0
                            %3:f64[extent] = update_slice [start_indices=[1]] %2 %1
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap())])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0]).unwrap()));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 10.0, 11.0, 12.0]).unwrap())];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 11.0]).unwrap())]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![5.0_f64, 7.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 7.0, 0.0]).unwrap())]),
        );
    }

    #[test]
    fn test_slice_differentiation_array_ir_strides() {
        let extent = DimensionVariable::new("strided_extent", DimensionBounds::new(4, Some(7)).unwrap());
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]))
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let concrete_type = input_type.clone().with_shape(Shape::new(vec![4.into()]));
        let output_type = concrete_type.clone().with_shape(Shape::new(vec![2.into()]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Slice(
                    SliceOperation::new(vec![0], vec![4]).with_strides(vec![2]).unwrap(),
                )),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(
                Array::from_elements(concrete_type.clone(), &[1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
            )])
            .unwrap();
        assert_eq!(
            primal_outputs[0],
            ArrayIrValue::Array(Array::from_elements(output_type.clone(), &[1.0_f64, 3.0]).unwrap())
        );
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(
            Array::from_elements(concrete_type.clone(), &[9.0_f64, 10.0, 11.0, 12.0]).unwrap(),
        )];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::from_elements(output_type.clone(), &[9.0_f64, 11.0]).unwrap())]),
        );
        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::from_elements(output_type.clone(), &[5.0_f64, 7.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(concrete_type.clone(), &[5.0_f64, 0.0, 7.0, 0.0]).unwrap()
            )]),
        );
    }

    #[test]
    fn test_slice_transposition() {
        check_operation_transposition!(
            @exact,
            operation = SliceOperation::new(vec![1], vec![3]),
            cases = [{
                inputs = [(@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![4.into()]))))],
                output_cotangents = [Array::vector(vec![5.0, 7.0]).unwrap()],
                input_cotangents = [Array::vector(vec![0.0, 5.0, 7.0, 0.0]).unwrap()],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = SliceOperation::new(vec![1], vec![6]).with_strides(vec![2]).unwrap(),
            cases = [
                {
                    inputs = [(@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![6.into()]))))],
                    output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0]).unwrap()],
                    input_cotangents = [Array::vector(vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0]).unwrap()],
                },
                {
                    inputs = [(@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![6.into()]))
                        .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
                        .with_memory(Memory::Host { pinned: true })))],
                    output_cotangents = [Array::from_elements::<f64>(
                        ArrayType::new(DataType::F64, Shape::new(vec![3.into()]))
                            .with_memory(Memory::Host { pinned: true }),
                        &[1.0, 2.0, 3.0],
                    ).unwrap()],
                    input_cotangents = [Array::from_elements::<f64>(
                        ArrayType::new(DataType::F64, Shape::new(vec![6.into()]))
                            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
                            .with_memory(Memory::Host { pinned: true }),
                        &[0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
                    ).unwrap()],
                },
            ],
        );

        // Inverse strided padding must preserve distributed cotangent dependencies on its internal zero scalar.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [6])
            .with_sharding(Sharding::replicated(mesh, 1).with_unreduced_axes(["m"]).unwrap())
            .unwrap();
        let input_cotangent_type = input_type.cotangent().unwrap();
        let output_cotangent_type = input_cotangent_type.clone().with_shape(Shape::new(vec![3.into()]));
        check_operation_transposition!(
            @exact,
            operation = SliceOperation::new(vec![1], vec![6]).with_strides(vec![2]).unwrap(),
            cases = [{
                inputs = [(@linear(type = input_type))],
                output_cotangents = [Array::from_elements(output_cotangent_type, &[1_f64, 2., 3.]).unwrap()],
                input_cotangents = [Array::from_elements(input_cotangent_type, &[0_f64, 1., 0., 2., 0., 3.]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_slice_transposition_accumulator_count() {
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let inputs = [PartialValue::Unknown(ArrayType::new_static(DataType::F64, [5]))];
        let outputs = [MaybeZero::Zero(ArrayType::new_static(DataType::F64, [2]))];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        let operation = SliceOperation::new(vec![1], vec![3]);

        // Even a structural-zero cotangent must validate its boundary before skipping the update.
        assert_eq!(
            operation.transpose(&mut context, &EmptyRegionDriver, &inputs, &outputs, &[]),
            Err(DifferentiationError::InvalidAccumulatorCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            operation.transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &outputs,
                &[accumulators[0].clone(), accumulators[0].clone()],
            ),
            Err(DifferentiationError::InvalidAccumulatorCount { expected: 1, actual: 2 }),
        );
    }

    #[test]
    fn test_slice_transposition_dynamic_input() {
        let elements = DimensionVariable::new("elements", DimensionBounds::new(4, Some(8)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(elements)]));

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(dynamic_type.clone());
        let output = builder
            .add_instruction(SliceOperation::new(vec![0], vec![2]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap_err(),
            TypeError::invalid(format!(
                "`{SLICE_OPERATION_NAME}` transpose requires a static input shape but got `f64[elements]`"
            ))
            .into(),
        );

        // The strided strategy reports the offending axis because it derives its padding from every input extent.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(dynamic_type);
        let output = builder
            .add_instruction(
                SliceOperation::new(vec![0], vec![4]).with_strides(vec![2]).unwrap(),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap_err(),
            TypeError::invalid(format!(
                "`{SLICE_OPERATION_NAME}` transpose requires a static input shape but axis 0 has size elements"
            ))
            .into(),
        );
    }

    #[test]
    fn test_slice_transposition_array_ir() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [5]).into());
        let first = builder
            .add_instruction(
                ArrayOperation::Slice(SliceOperation::new(vec![1], vec![4])),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let second = builder
            .add_instruction(
                ArrayOperation::Slice(SliceOperation::new(vec![2], vec![5])),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![first, second],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert!(pullback.output_ids().is_empty());
        // Both overlapping slices update the supplied buffer directly through views. There is no dense zero, pad, or
        // full-gradient result, and the second invocation adds another contribution without resetting the buffer.
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3], %2:ref<f64[5]> .
                let %3:ref<f64[3]> = reference_slice [axes=[ArraySliceAxis { start: 2, size: 3, stride: 1 }]] %2
                    () = reference_add_update %3 %1
                    %4:ref<f64[3]> = reference_slice [axes=[ArraySliceAxis { start: 1, size: 3, stride: 1 }]] %2
                    () = reference_add_update %4 %0
                in ()
            "}
            .trim_end(),
        );
        let buffer = ArrayIrValue::Array(Array::vector(vec![10.0_f64; 5]).unwrap()).reference_new().unwrap();
        let seeds = [
            ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![4.0_f64, 5.0, 6.0]).unwrap()),
        ];
        assert_eq!(pullback.interpret(vec![seeds[0].clone(), seeds[1].clone(), buffer.clone()]), Ok(vec![]));
        assert_eq!(
            buffer.read(),
            Ok(ArrayIrValue::Array(Array::vector(vec![10.0_f64, 11.0, 16.0, 18.0, 16.0]).unwrap()))
        );
        assert_eq!(pullback.interpret(vec![seeds[0].clone(), seeds[1].clone(), buffer.clone()]), Ok(vec![]));
        assert_eq!(
            buffer.read(),
            Ok(ArrayIrValue::Array(Array::vector(vec![10.0_f64, 12.0, 22.0, 26.0, 22.0]).unwrap()))
        );

        let returned = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            returned.interpret(seeds.to_vec()),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 1.0, 6.0, 8.0, 6.0]).unwrap())]),
        );
    }

    #[test]
    fn test_array_type_slice() {
        let host_input = ArrayType::new_static(DataType::F32, [4]).with_memory(Memory::Host { pinned: true });
        assert_eq!(host_input.slice(&[0], &[2], &[1]).unwrap().memory(), Memory::Host { pinned: true });
        let laid_out = host_input.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(laid_out.slice(&[0], &[4], &[1]), Ok(laid_out.clone()));

        // Resizing keeps explicit placement only when the resulting dimension remains evenly divisible.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let sharding = sharding.with_unreduced_axes(["m"]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [4, 4]).with_sharding(sharding.clone()).unwrap();
        assert_eq!(input.slice(&[0, 0], &[2, 4], &[1, 1]).unwrap().sharding(), Some(&sharding));
        assert_eq!(
            input.slice(&[0, 0], &[3, 4], &[1, 1]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{SLICE_OPERATION_NAME}` on a dimension sharded over explicit mesh axes requires the output size (3) \
                 at axis 0 to be divisible by the mesh-axis product (2)"
            ))))
        );
    }

    #[test]
    fn test_array_slice() {
        // Rank-3 slice exercises the row-major odometer across non-contiguous blocks.
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        );
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = Array::from_elements::<f64>(input_type.clone(), &values.clone())
            .unwrap()
            .slice(&[0, 1, 2], &[2, 3, 4], &[1, 1, 1])
            .unwrap();
        assert_eq!(
            *output.r#type(),
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)])
            ),
        );
        assert_eq!(output.to_f64s(), vec![6.0, 7.0, 10.0, 11.0, 18.0, 19.0, 22.0, 23.0]);

        // Strided slicing walks the row-major odometer with per-axis steps: rows with stride 2 and columns with stride
        // 3 keep elements at indices (0, 0), (0, 3), (1, 0), and (1, 3) of a 2x3x4 input's last two axes.
        let strided = Array::from_elements::<f64>(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
            ),
            &(0..24).map(|value| value as f64).collect::<Vec<_>>(),
        )
        .unwrap()
        .slice(&[0, 0, 0], &[2, 3, 4], &[2, 2, 3])
        .unwrap();
        assert_eq!(
            *strided.r#type(),
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(1), Dimension::Static(2), Dimension::Static(2)])
            ),
        );
        assert_eq!(strided.to_f64s(), vec![0.0, 3.0, 8.0, 11.0]);
    }

    #[test]
    fn test_array_slice_layouts() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(vector.slice(&[1], &[5], &[2]).unwrap(), Array::vector(vec![2.0, 4.0]).unwrap());

        // Static slicing traverses the logical coordinates of a reversed source layout.
        let input_type =
            ArrayType::new_static(DataType::U16, [5]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let vector = Array::from_elements(input_type, &[1u16, 2, 3, 4, 5]).unwrap();
        assert_eq!(vector.slice(&[1], &[5], &[2]).unwrap().elements::<u16>(), Ok(vec![2, 4]));
    }

    #[test]
    fn test_array_ir_value_slice() {
        let input = ArrayIrValue::Array(Array::vector(vec![10i32, 20, 30]).unwrap());
        assert_eq!(input.slice(&[1], &[2], &[1]), Ok(ArrayIrValue::Array(Array::vector(vec![20i32]).unwrap())));
        let wrong = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(1).unwrap());
        assert!(matches!(wrong.slice(&[], &[], &[]), Err(ProgramError::Type(error))
            if error == TypeError::invalid("expected array type but got dimension type")));
    }

    #[test]
    fn test_array_slice_axis() {
        let input = Array::matrix(2, 3, vec![10i32, 20, 30, 40, 50, 60]).unwrap();
        assert_eq!(input.slice_axis(-1, 0, 3, 2), Ok(Array::matrix(2, 2, vec![10i32, 30, 40, 60]).unwrap()));
        assert_eq!(input.slice_axis(0, 1, 2, 1), Ok(Array::matrix(1, 3, vec![40i32, 50, 60]).unwrap()));
        assert!(matches!(input.slice_axis(1, 0, 3, 0), Err(ProgramError::Type(error))
        if error == TypeError::invalid(format!(
            "`{SLICE_OPERATION_NAME}` strides must be at least 1 but axis 1 has stride 0"
        ))));

        // Every unsliced axis must have a static extent because the static operation needs its complete limit.
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("rows", DimensionBounds::new(1, Some(4)).unwrap())),
                Dimension::Static(3),
            ]),
        );
        assert!(matches!(
            dynamic_type.slice_axis(1, 0, 2, 1),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`slice_axis` requires a static extent on unsliced axis 0",
        ));
    }

    #[test]
    fn test_array_index_axis() {
        let input = Array::matrix(2, 3, vec![10i32, 20, 30, 40, 50, 60]).unwrap();
        assert_eq!(input.index_axis(-1, 1, false), Ok(Array::vector(vec![20i32, 50]).unwrap()));
        assert_eq!(input.index_axis(-1, 1, true), Ok(Array::matrix(2, 1, vec![20i32, 50]).unwrap()));
        assert!(matches!(input.index_axis(0, usize::MAX, false), Err(ProgramError::Type(error))
            if error == TypeError::invalid("`index_axis` index overflows `usize`")));
    }

    #[test]
    fn test_update_slice() {
        let operation = UpdateSliceOperation::new(vec![0, 1]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), UPDATE_SLICE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "update_slice [start_indices=[0, 1]]");
        assert_eq!(operation.start_indices(), &[0, 1]);

        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)]));
        // Program rendering uses the canonical operation name and includes the captured indices.
        let mut builder = ProgramBuilder::<Array, UpdateSliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_update = builder.add_input(update_type);
        let program_output =
            builder.add_instruction(operation, Vec::new(), vec![program_input, program_update], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![program_output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[1, 2] .
                let %2:f64[2, 3] = update_slice [start_indices=[0, 1]] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_update_slice_type_inference() {
        let operation = UpdateSliceOperation::new(vec![0, 1]);
        // Type inference validates that the update fits and returns the input type, and the type-level (abstract)
        // capability backs it without consuming the borrowed input type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)]));
        let dynamic_update_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(2),
            ]),
        );
        let dynamic_input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(4)).unwrap())),
                Dimension::Static(3),
            ]),
        );
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input_type.clone(), update_type.clone()],
                    output_types = [input_type.clone()],
                },
                {
                    input_types = [],
                    error = "expected 2 inputs but got 0",
                },
                {
                    input_types = [
                        input_type.clone(),
                        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)])),
                    ],
                    error = format!(
                        "`{UPDATE_SLICE_OPERATION_NAME}` input data type `f64` does not match update data type `f32`"
                    ),
                },
                {
                    input_types = [input_type.clone(), ArrayType::new(DataType::F64, Shape::new(vec![2.into()]))],
                    error = format!("`{UPDATE_SLICE_OPERATION_NAME}` update has rank 1 but input has rank 2"),
                },
                {
                    input_types = [input_type.clone(), dynamic_update_type],
                    error = format!(
                        "`{UPDATE_SLICE_OPERATION_NAME}` does not support dynamic update axis 0 with size dynamic; \
                         update shapes must be static"
                    ),
                },
                {
                    input_types = [dynamic_input_type, update_type.clone()],
                    error = format!(
                        "`{UPDATE_SLICE_OPERATION_NAME}` update limit 1 exceeds the guaranteed minimum extent 0 of \
                         dynamic axis 0"
                    ),
                },
            ],
        );
        assert_eq!(input_type.update_slice(&update_type, &[0, 1]), Ok(input_type.clone()));

        // Malformed start indices and windows that do not fit report precise operation errors.
        check_operation_type_inference!(
            operation = UpdateSliceOperation::new(vec![0, usize::MAX]),
            cases = [{
                input_types = [input_type.clone(), update_type.clone()],
                error = format!("`{UPDATE_SLICE_OPERATION_NAME}` update limit overflows `usize` on axis 1"),
            }],
        );
        check_operation_type_inference!(
            operation = UpdateSliceOperation::new(vec![0]),
            cases = [{
                input_types = [input_type.clone(), update_type.clone()],
                error = format!("`{UPDATE_SLICE_OPERATION_NAME}` `start_indices` has length 1 but input has rank 2"),
            }],
        );
        check_operation_type_inference!(
            operation = UpdateSliceOperation::new(vec![0, 2]),
            cases = [{
                input_types = [input_type.clone(), update_type.clone()],
                error = format!(
                    "`{UPDATE_SLICE_OPERATION_NAME}` update axis 1 with start index 2 and size 2 does not fit in \
                     input size 3"
                ),
            }],
        );

        // A slice operation cannot own nested regions.
        assert_eq!(
            UpdateSliceOperation::new(vec![])
                .infer_output_types(&[], &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)]),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_update_slice_reference_discharge() {
        // Replay preserves the complete slicing payload and its output type. Shared replay and reference rejection
        // are covered by the reference-discharge macro tests.
        let expected = UpdateSliceOperation::new(vec![0, 1]);
        let operation = ArrayIrOperation::Array(ArrayOperation::UpdateSlice(expected.clone()));
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = ReferenceDischargeContext::<_, ArrayReferenceDischarge>::new(trace.clone());
        let inputs = [
            ReferenceDischargeValue::Value(trace.input(ArrayType::new_static(DataType::F64, [2, 3]).into())),
            ReferenceDischargeValue::Value(trace.input(ArrayType::new_static(DataType::F64, [1, 2]).into())),
        ];
        let outputs = operation.discharge_references(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        let ReferenceDischargeValue::Value(output) = &outputs[0] else {
            panic!("expected a value carrier but got {}", outputs[0]);
        };
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 3])));
        let builder = trace.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        let ArrayIrOperation::Array(ArrayOperation::UpdateSlice(staged)) = builder.instructions()[0].operation() else {
            panic!("expected a staged update_slice");
        };
        assert_eq!(staged, &expected);
    }

    #[test]
    fn test_update_slice_interpretation() {
        // Applying output sharding metadata preserves the non-dense layout and the untouched input values.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type =
            ArrayType::new_static(DataType::I32, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![-4])));
        let update_type = ArrayType::new_static(DataType::I32, [1])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let input = Array::from_elements(input_type.clone(), &[1_i32, 2, 3]).unwrap();
        let update = Array::from_elements(update_type, &[9_i32]).unwrap();
        let output = input.update_slice(&update, &[1]).unwrap();
        let expected_type = input_type
            .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        assert_eq!(output, Array::from_elements(expected_type, &[1_i32, 9, 3]).unwrap());
        assert_eq!(input.elements::<i32>(), Ok(vec![1, 2, 3]));

        let operation = UpdateSliceOperation::new(vec![0, 1]);
        let input_type = ArrayType::new_static(DataType::F64, [2, 3]);
        // Interpretation overwrites the selected block of the row-major payload.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let update = Array::matrix(1, 2, vec![8.0, 9.0]).unwrap();
        let output = operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[input, update]).unwrap();
        assert_eq!(*output[0].r#type(), input_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);

        // Rank-0 updates replace the input entirely.
        let scalar = Array::scalar(1.0).unwrap().update_slice(&Array::scalar(7.0).unwrap(), &[]).unwrap();
        assert_eq!(scalar.to_f64s(), vec![7.0]);
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
    }

    #[test]
    fn test_update_slice_partial_evaluation() {
        // Check standard partial evaluation with known and residual inputs.
        let input = Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let update = Array::vector(vec![8.0, 9.0]).unwrap();
        let expected = Array::vector(vec![0.0, 8.0, 9.0, 3.0]).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = UpdateSliceOperation::new(vec![1]),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, update.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, update.clone()),
                    ],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_update_slice_batching() {
        // Batching aligns mapped and replicated inputs before applying the update independently to each item.
        check_operation_batching!(
            @exact,
            operation = UpdateSliceOperation::new(vec![1]),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::matrix(
                            2,
                            4,
                            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                        ).unwrap()),
                        (@replicated, Array::vector(vec![9.0, 9.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        4,
                        vec![0.0, 9.0, 9.0, 3.0, 4.0, 9.0, 9.0, 7.0],
                    ).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap()),
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![8.0, 8.0, 9.0, 9.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        4,
                        vec![0.0, 8.0, 8.0, 3.0, 0.0, 9.0, 9.0, 3.0],
                    ).unwrap())],
                },
            ],
        );

        // Bounded ragged inputs are rejected before any alignment, and the arity is validated right after.
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        let update = ArrayBatch::replicated(Array::vector(vec![9.0, 9.0]).unwrap());
        assert!(matches!(
            UpdateSliceOperation::new(vec![1]).batch(&context, &EmptyRegionDriver, &[ragged_batch(), update.clone()]),
            Err(BatchingError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!("`{UPDATE_SLICE_OPERATION_NAME}` does not support bounded ragged array inputs"),
        ));
        assert_eq!(
            UpdateSliceOperation::new(vec![1]).batch(&context, &EmptyRegionDriver, &[update]).unwrap_err(),
            BatchingError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
    }

    #[test]
    fn test_update_slice_batching_sharding() {
        // A mapped input sharded over an explicit mesh axis keeps its placement through both the static and the dynamic
        // update rule: the replicated update is aligned to the mapped input before each item writes its own block.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let explicit_sharding = Sharding::new(
            explicit_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [2, 4]).with_sharding(explicit_sharding.clone()).unwrap();
        let update_type = ArrayType::new_static(DataType::F64, [2])
            .with_sharding(Sharding::replicated(explicit_mesh, 1))
            .unwrap();
        let context =
            BatchingContext::new(EagerContext::<Array>::new(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));
        let input = ArrayBatch::new(
            Array::from_elements::<f64>(input_type, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let update = ArrayBatch::replicated(Array::from_elements::<f64>(update_type, &[9.0, 9.0]).unwrap());
        let outputs = UpdateSliceOperation::new(vec![1])
            .batch(&context, &EmptyRegionDriver, &[input.clone(), update.clone()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().sharding(), Some(&explicit_sharding));
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0, 9.0, 9.0, 3.0, 4.0, 9.0, 9.0, 7.0]);
        let outputs = DynamicUpdateSliceOperation::new()
            .batch(
                &context,
                &EmptyRegionDriver,
                &[input, update, ArrayBatch::replicated(Array::scalar(1_i32).unwrap())],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().sharding(), Some(&explicit_sharding));
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0, 9.0, 9.0, 3.0, 4.0, 9.0, 9.0, 7.0]);

        // A mapped input varying over a manual mesh axis keeps both its placement and its variation.
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let manual_sharding = Sharding::new(
            manual_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap()
        .with_varying_manual_axes(["x"])
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [2, 4]).with_sharding(manual_sharding.clone()).unwrap();
        let update_type = ArrayType::new_static(DataType::F64, [2])
            .with_sharding(Sharding::replicated(manual_mesh, 1))
            .unwrap();
        let input = ArrayBatch::new(
            Array::from_elements::<f64>(input_type, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let update = ArrayBatch::replicated(Array::from_elements::<f64>(update_type, &[9.0, 9.0]).unwrap());
        let outputs = UpdateSliceOperation::new(vec![1])
            .batch(&context, &EmptyRegionDriver, &[input.clone(), update.clone()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().sharding(), Some(&manual_sharding));
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0, 9.0, 9.0, 3.0, 4.0, 9.0, 9.0, 7.0]);
        let outputs = DynamicUpdateSliceOperation::new()
            .batch(
                &context,
                &EmptyRegionDriver,
                &[input, update, ArrayBatch::replicated(Array::scalar(1_i32).unwrap())],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().sharding(), Some(&manual_sharding));
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0, 9.0, 9.0, 3.0, 4.0, 9.0, 9.0, 7.0]);
    }

    #[test]
    fn test_update_slice_differentiation() {
        // Static update-slice is jointly linear in the input and update.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = UpdateSliceOperation::new(vec![1]),
            cases = [{
                primals = [
                    Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
                    Array::vector(vec![8.0, 9.0]).unwrap(),
                ],
                tangents = [
                    Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                    Array::vector(vec![5.0, 6.0]).unwrap(),
                ],
                primal_outputs = [Array::vector(vec![0.0, 8.0, 9.0, 3.0]).unwrap()],
                tangent_outputs = [Array::vector(vec![1.0, 5.0, 6.0, 4.0]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_update_slice_differentiation_dynamic_input() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(3, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let update = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::UpdateSlice(UpdateSliceOperation::new(vec![1]))),
                Vec::new(),
                vec![input, update],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        assert_eq!(linearization.residual_count(), 0);
        let primal = vec![
            ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![9.0_f64, 8.0]).unwrap()),
        ];
        assert_eq!(
            linearization.primal().interpret(primal),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 9.0, 8.0, 4.0]).unwrap())]),
        );
        assert_eq!(
            linearization.tangent().interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![5.0_f64, 6.0]).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 5.0, 6.0, 40.0]).unwrap())]),
        );
        assert_eq!(
            linearization
                .pullback()
                .unwrap()
                .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0,]).unwrap())]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 0.0, 0.0, 4.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0]).unwrap()),
            ]),
        );

        // With only the input tangent live, the missing update tangent has a static shape. Forward evaluation may
        // retain its materialized zero, but needs no runtime input dimensions to construct it.
        let input_only = program.linearize_with_respect_to(&[0]).unwrap();
        let mut outputs = input_only
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![9.0_f64, 8.0]).unwrap()),
            ])
            .unwrap();
        assert_eq!(outputs.remove(0), ArrayIrValue::Array(Array::vector(vec![1.0_f64, 9.0, 8.0, 4.0]).unwrap()));
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap())];
        tangent_inputs.extend(outputs.clone());
        assert_eq!(
            input_only.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 0.0, 0.0, 40.0]).unwrap())])
        );
        let mut cotangents = vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap())];
        cotangents.extend(outputs);
        assert_eq!(
            input_only.pullback().unwrap().interpret(cotangents),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 0.0, 0.0, 4.0]).unwrap())])
        );

        // With only the update tangent live, retain the runtime input extent to construct its missing zero tangent.
        // Reuse the same transformed program at two extents to catch accidental specialization to the first shape.
        let update_only = program.linearize_with_respect_to(&[1]).unwrap();
        assert_eq!(update_only.residual_count(), 1);
        let mut outputs = update_only
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![9.0_f64, 8.0]).unwrap()),
            ])
            .unwrap();
        assert_eq!(outputs.remove(0), ArrayIrValue::Array(Array::vector(vec![1.0_f64, 9.0, 8.0, 4.0]).unwrap()));
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![5.0_f64, 6.0]).unwrap())];
        tangent_inputs.extend(outputs.clone());
        assert_eq!(
            update_only.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 6.0, 0.0]).unwrap())])
        );
        let pullback = update_only.pullback().unwrap();
        let mut cotangents = vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap())];
        cotangents.extend(outputs);
        assert_eq!(
            pullback.interpret(cotangents),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0]).unwrap())])
        );

        let mut outputs = update_only
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![9.0_f64, 8.0]).unwrap()),
            ])
            .unwrap();
        assert_eq!(outputs.remove(0), ArrayIrValue::Array(Array::vector(vec![1.0_f64, 9.0, 8.0, 4.0, 5.0]).unwrap()));
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![5.0_f64, 6.0]).unwrap())];
        tangent_inputs.extend(outputs.clone());
        assert_eq!(
            update_only.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 6.0, 0.0, 0.0]).unwrap())])
        );
        let mut cotangents = vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap())];
        cotangents.extend(outputs);
        assert_eq!(
            pullback.interpret(cotangents),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0]).unwrap())])
        );
    }

    #[test]
    fn test_update_slice_transposition() {
        check_operation_transposition!(
            @exact,
            operation = UpdateSliceOperation::new(vec![1]),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![4.into()])))),
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()])))),
                ],
                output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
                input_cotangents = [
                    Array::vector(vec![1.0, 0.0, 0.0, 4.0]).unwrap(),
                    Array::vector(vec![2.0, 3.0]).unwrap(),
                ],
            }],
        );

        // Slicing the output cotangent back to the update restores the update's complete layout-bearing type.
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![4.into()])).with_memory(Memory::Host { pinned: true });
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        check_operation_transposition!(
            @exact,
            operation = UpdateSliceOperation::new(vec![1]),
            cases = [{
                inputs = [(@linear(type = input_type.clone())), (@linear(type = update_type.clone()))],
                output_cotangents = [Array::from_elements::<f64>(input_type.clone(), &[1.0, 2.0, 3.0, 4.0]).unwrap()],
                input_cotangents = [
                    Array::from_elements::<f64>(input_type, &[1.0, 0.0, 0.0, 4.0]).unwrap(),
                    Array::from_elements::<f64>(update_type, &[2.0, 3.0]).unwrap(),
                ],
            }],
        );
    }

    #[test]
    fn test_update_slice_transposition_zero_cotangent() {
        // A structural-zero output cotangent contributes nothing: the rule returns before staging anything and leaves
        // both accumulators at their structural-zero defaults. The same holds when neither cotangent is needed.
        let operation = UpdateSliceOperation::new(vec![1]);
        let input_type = ArrayType::new_static(DataType::F64, [4]);
        let update_type = ArrayType::new_static(DataType::F64, [2]);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut transpose = TranspositionContext::new(context.clone());
        let inputs = [PartialValue::Unknown(input_type.clone()), PartialValue::Unknown(update_type.clone())];
        let accumulators = transpose.cotangent_accumulators(&inputs, &[]).unwrap();
        let zero_outputs = [MaybeZero::Zero(input_type.cotangent().unwrap())];
        operation
            .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &zero_outputs, &accumulators)
            .unwrap();
        let cotangents = transpose.take_cotangents(&accumulators).unwrap();
        assert_eq!(cotangents.len(), 2);
        assert!(cotangents[0].is_zero());
        assert!(cotangents[1].is_zero());
        assert!(context.builder().borrow().instructions().is_empty());
        let outputs = [MaybeZero::Value(context.input(input_type.cotangent().unwrap()))];
        let unneeded = transpose.cotangent_accumulators(&inputs, &[false, false]).unwrap();
        operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &unneeded).unwrap();
        assert!(context.builder().borrow().instructions().is_empty());

        // Each contribution is staged only when its accumulator is needed: the input cotangent zeroes the update
        // window and the update cotangent slices it.
        let input_only = transpose.cotangent_accumulators(&inputs, &[true, false]).unwrap();
        operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &input_only).unwrap();
        let builder = context.builder();
        assert_eq!(
            builder
                .borrow()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().to_string())
                .collect::<Vec<_>>(),
            vec!["zero [type=f64[2]]", "update_slice [start_indices=[1]]"],
        );
        let cotangents = transpose.take_cotangents(&input_only).unwrap();
        assert!(!cotangents[0].is_zero());
        assert!(cotangents[1].is_zero());
        let update_only = transpose.cotangent_accumulators(&inputs, &[false, true]).unwrap();
        operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &update_only).unwrap();
        assert_eq!(
            builder
                .borrow()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().to_string())
                .collect::<Vec<_>>(),
            vec![
                "zero [type=f64[2]]",
                "update_slice [start_indices=[1]]",
                "slice [start_indices=[1], limit_indices=[3]]",
            ],
        );
        let cotangents = transpose.take_cotangents(&update_only).unwrap();
        assert!(cotangents[0].is_zero());
        assert!(!cotangents[1].is_zero());

        // Arity is validated before any cotangent is inspected.
        assert_eq!(
            operation
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs[..1], &outputs, &accumulators)
                .unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &[], &accumulators).unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            operation
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &accumulators[..1])
                .unwrap_err(),
            DifferentiationError::InvalidAccumulatorCount { expected: 2, actual: 1 },
        );
    }

    #[test]
    fn test_array_type_update_slice() {
        let host_input = ArrayType::new_static(DataType::F32, [4]).with_memory(Memory::Host { pinned: true });
        let host_update = ArrayType::new_static(DataType::F32, [2]).with_memory(Memory::Host { pinned: true });
        assert_eq!(host_input.update_slice(&host_update, &[0]).unwrap().memory(), Memory::Host { pinned: true });
        assert_eq!(
            host_input.update_slice(&ArrayType::new_static(DataType::F32, [2]), &[0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{UPDATE_SLICE_OPERATION_NAME}` input and update must share one memory space but reside in \
                 `Host[Pinned]` and `Device`"
            ))))
        );

        // Resizing keeps explicit placement only when the resulting dimension remains evenly divisible.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let input = ArrayType::new_static(DataType::F32, [4, 4]).with_sharding(sharding.clone()).unwrap();
        let matching = ArrayType::new_static(DataType::F32, [2, 4]).with_sharding(sharding.clone()).unwrap();
        let conflicting =
            ArrayType::new_static(DataType::F32, [2, 4]).with_sharding(Sharding::replicated(mesh, 2)).unwrap();
        assert_eq!(input.update_slice(&matching, &[0, 0]).unwrap().sharding(), Some(&sharding));
        assert_eq!(
            input.update_slice(&conflicting, &[0, 0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{UPDATE_SLICE_OPERATION_NAME}` input and update must be sharded identically, but got `{}` and `{}`",
                input.sharding().unwrap(),
                conflicting.sharding().unwrap(),
            ))))
        );
        // A varying manual update changes the dependency metadata while retaining explicit placement.
        let varying = matching.clone().with_sharding(sharding.with_varying_manual_axes(["m"]).unwrap()).unwrap();
        assert_eq!(input.update_slice(&varying, &[0, 0]).unwrap().sharding(), varying.sharding());
    }

    #[test]
    fn test_array_update_slice() {
        // Rank-3 slice exercises the row-major odometer across non-contiguous blocks.
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        );
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        // The matching update-slice writes the block back into place.
        let update = Array::from_elements::<f64>(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
            ),
            &[-6.0, -7.0, -10.0, -11.0, -18.0, -19.0, -22.0, -23.0],
        )
        .unwrap();
        let updated =
            Array::from_elements::<f64>(input_type, &values).unwrap().update_slice(&update, &[0, 1, 2]).unwrap();
        assert_eq!(
            updated.to_f64s(),
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -6.0, -7.0, 8.0, 9.0, -10.0, -11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                -18.0, -19.0, 20.0, 21.0, -22.0, -23.0,
            ],
        );
    }

    #[test]
    fn test_array_update_slice_layouts() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(
            vector.update_slice(&Array::vector(vec![10.0, 20.0]).unwrap(), &[1]).unwrap(),
            Array::vector(vec![1.0, 10.0, 20.0, 4.0, 5.0]).unwrap(),
        );

        // Updating traverses arbitrary source and update layouts while preserving the destination layout.
        let input_type =
            ArrayType::new_static(DataType::U16, [5]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let vector = Array::from_elements(input_type.clone(), &[1u16, 2, 3, 4, 5]).unwrap();
        let update_type =
            ArrayType::new_static(DataType::U16, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let update = Array::from_elements(update_type, &[10u16, 20]).unwrap();
        let updated = vector.update_slice(&update, &[1]).unwrap();
        assert_eq!(updated.r#type().as_ref(), &input_type);
        assert_eq!(updated.elements::<u16>(), Ok(vec![1, 10, 20, 4, 5]));
        assert_eq!(updated.storage_bytes(), [5, 0, 4, 0, 20, 0, 10, 0, 1, 0]);
    }

    #[test]
    fn test_array_update_slice_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated]).unwrap();
        let varying = sharding.clone().with_varying_manual_axes(["m"]).unwrap();
        let input = Array::from_elements(
            ArrayType::new_static(DataType::I32, [3]).with_sharding(sharding).unwrap(),
            &[10i32, 20, 30],
        )
        .unwrap();
        let update = Array::from_elements(
            ArrayType::new_static(DataType::I32, [1]).with_sharding(varying.clone()).unwrap(),
            &[40i32],
        )
        .unwrap();
        let expected = Array::from_elements(
            ArrayType::new_static(DataType::I32, [3]).with_sharding(varying.clone()).unwrap(),
            &[10i32, 40, 30],
        )
        .unwrap();
        assert_eq!(input.update_slice(&update, &[1]), Ok(expected.clone()));
        assert_eq!(input.dynamic_update_slice(&update, &[Array::scalar(1i32).unwrap()]), Ok(expected));

        // A discrete start can vary over a manual axis even when both array inputs are invariant.
        let index = Array::from_elements(
            ArrayType::scalar(DataType::I32)
                .with_sharding(Sharding::new(mesh, vec![]).unwrap().with_varying_manual_axes(["m"]).unwrap())
                .unwrap(),
            &[1i32],
        )
        .unwrap();
        assert_eq!(
            input.dynamic_slice(std::slice::from_ref(&index), &[1]).unwrap().r#type().sharding(),
            Some(&varying)
        );
        assert_eq!(
            input
                .dynamic_update_slice(&Array::vector(vec![40i32]).unwrap(), &[index])
                .unwrap()
                .r#type()
                .sharding(),
            Some(&varying)
        );

        // An unsharded base is invariant. A varying update still makes the written block vary over its mesh.
        let plain = Array::vector(vec![10i32, 20, 30]).unwrap();
        let expected = Array::from_elements(
            ArrayType::new_static(DataType::I32, [3]).with_sharding(varying).unwrap(),
            &[10i32, 40, 30],
        )
        .unwrap();
        assert_eq!(plain.update_slice(&update, &[1]), Ok(expected.clone()));
        assert_eq!(plain.dynamic_update_slice(&update, &[Array::scalar(1i32).unwrap()]), Ok(expected));
    }

    #[test]
    fn test_array_ir_value_update_slice() {
        let input = ArrayIrValue::Array(Array::vector(vec![10i32, 20, 30]).unwrap());
        let update = ArrayIrValue::Array(Array::vector(vec![40i32]).unwrap());
        assert_eq!(
            input.update_slice(&update, &[1]),
            Ok(ArrayIrValue::Array(Array::vector(vec![10i32, 40, 30]).unwrap()))
        );
    }

    #[test]
    fn test_dynamic_slice() {
        let operation = DynamicSliceOperation::new(vec![1, 2]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), DYNAMIC_SLICE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "dynamic_slice [sizes=[1, 2]]");
        assert_eq!(operation.sizes(), &[1, 2]);
        assert!(operation.allows_negative_indices());

        // The default negative-index policy is implied by the rendering; only the clamp-only policy is shown.
        let clamping = operation.clone().with_allow_negative_indices(false);
        assert!(!clamping.allows_negative_indices());
        assert_eq!(format!("{clamping}"), "dynamic_slice [sizes=[1, 2], allow_negative_indices=false]");

        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let index_type = ArrayType::scalar(DataType::I32);
        // Program rendering uses the canonical operation name and includes the captured sizes.
        let mut builder = ProgramBuilder::<Array, DynamicSliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_index_0 = builder.add_input(index_type.clone());
        let program_index_1 = builder.add_input(index_type);
        let program_output = builder
            .add_instruction(operation, Vec::new(), vec![program_input, program_index_0, program_index_1], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![program_output], vec![Placeholder, Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:i32[], %2:i32[] .
                let %3:f64[1, 2] = dynamic_slice [sizes=[1, 2]] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dynamic_slice_type_inference() {
        let operation = DynamicSliceOperation::new(vec![1, 2]);
        // Type inference validates the sizes and index input types and returns the statically shaped output.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let index_type = ArrayType::scalar(DataType::I32);
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)]));
        let dynamic_input_type = |bounds: DimensionBounds| {
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", bounds)), Dimension::Static(3)]),
            )
        };
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input_type.clone(), index_type.clone(), index_type.clone()],
                    output_types = [output_type.clone()],
                },
                // A dynamic input axis is accepted when its minimum extent proves that the static result window always
                // fits, whether or not the axis is bounded above. The static axis 1 still validates `2 <= 3`.
                {
                    input_types = [
                        dynamic_input_type(DimensionBounds::new(1, None).unwrap()),
                        index_type.clone(),
                        index_type.clone(),
                    ],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [
                        dynamic_input_type(DimensionBounds::new(1, Some(2)).unwrap()),
                        index_type.clone(),
                        index_type.clone(),
                    ],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [
                        dynamic_input_type(DimensionBounds::non_negative(Some(1)).unwrap()),
                        index_type.clone(),
                        index_type.clone(),
                    ],
                    error = format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` size 1 exceeds the guaranteed minimum extent 0 of dynamic \
                         axis 0"
                    ),
                },
                {
                    input_types = [],
                    error = format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` expects an array input followed by its start index inputs \
                         but got no inputs"
                    ),
                },
                {
                    input_types = [input_type.clone(), index_type.clone()],
                    error = format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` expects one start index per input axis (2) but got 1"
                    ),
                },
                {
                    input_types = [input_type.clone(), ArrayType::scalar(DataType::F64), index_type.clone()],
                    error = format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` start index 0 must be a scalar integer but has type `f64[]`"
                    ),
                },
                {
                    input_types = [
                        input_type.clone(),
                        ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2)])),
                        index_type.clone(),
                    ],
                    error = format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` start index 0 must be a scalar integer but has type `i32[2]`"
                    ),
                },
                {
                    input_types = [input_type.clone(), index_type.clone(), ArrayType::scalar(DataType::I64)],
                    error = format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` start indices must share one integer type but index 1 has \
                         type `i64[]` and index 0 has type `i32[]`"
                    ),
                },
            ],
        );
        assert_eq!(
            input_type.dynamic_slice(&[index_type.clone(), index_type.clone()], &[1, 2]),
            Ok(output_type.clone()),
        );

        // Malformed size lists and windows that do not fit report precise operation errors.
        check_operation_type_inference!(
            operation = DynamicSliceOperation::new(vec![1]),
            cases = [{
                input_types = [input_type.clone(), index_type.clone(), index_type.clone()],
                error = format!("`{DYNAMIC_SLICE_OPERATION_NAME}` sizes has length 1 but input has rank 2"),
            }],
        );
        check_operation_type_inference!(
            operation = DynamicSliceOperation::new(vec![1, 4]),
            cases = [{
                input_types = [input_type.clone(), index_type.clone(), index_type.clone()],
                error = format!("`{DYNAMIC_SLICE_OPERATION_NAME}` size 4 is out of bounds for axis 1 with size 3"),
            }],
        );

        // A slice operation cannot own nested regions.
        assert_eq!(
            DynamicSliceOperation::new(vec![])
                .infer_output_types(&[], &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)]),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_dynamic_slice_reference_discharge() {
        // Replay preserves the complete slicing payload and its output type. Shared replay and reference rejection
        // are covered by the reference-discharge macro tests.
        let expected = DynamicSliceOperation::new(vec![1, 2]);
        let operation = ArrayIrOperation::Array(ArrayOperation::DynamicSlice(expected.clone()));
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = ReferenceDischargeContext::<_, ArrayReferenceDischarge>::new(trace.clone());
        let inputs = [
            ReferenceDischargeValue::Value(trace.input(ArrayType::new_static(DataType::F64, [2, 3]).into())),
            ReferenceDischargeValue::Value(trace.input(ArrayType::scalar(DataType::I32).into())),
            ReferenceDischargeValue::Value(trace.input(ArrayType::scalar(DataType::I32).into())),
        ];
        let outputs = operation.discharge_references(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        let ReferenceDischargeValue::Value(output) = &outputs[0] else {
            panic!("expected a value carrier but got {}", outputs[0]);
        };
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(ArrayType::new_static(DataType::F64, [1, 2])));
        let builder = trace.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        let ArrayIrOperation::Array(ArrayOperation::DynamicSlice(staged)) = builder.instructions()[0].operation()
        else {
            panic!("expected a staged dynamic_slice");
        };
        assert_eq!(staged, &expected);
    }

    #[test]
    fn test_dynamic_slice_interpretation() {
        let operation = DynamicSliceOperation::new(vec![1, 2]);
        let output_type = ArrayType::new_static(DataType::F64, [1, 2]);
        // Interpretation extracts the block at the in-band start indices.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = operation
            .interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[input.clone(), Array::scalar(1_i32).unwrap(), Array::scalar(1_i32).unwrap()],
            )
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![5.0, 6.0]);

        // Out-of-bounds start indices resolve like the capability documents: the row start `5` clamps to the last
        // valid origin `1`, and the column start `-2` counts from the end of the extent-3 axis to `1`.
        let clamped = operation
            .interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[input.clone(), Array::scalar(5_i32).unwrap(), Array::scalar(-2_i32).unwrap()],
            )
            .unwrap();
        assert_eq!(clamped[0].to_f64s(), vec![5.0, 6.0]);
        // Without the negative-index policy, the column start `-2` is out of bounds and clamps to zero.
        let clamped = DynamicSliceOperation::new(vec![1, 2])
            .with_allow_negative_indices(false)
            .interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[input.clone(), Array::scalar(5_i32).unwrap(), Array::scalar(-2_i32).unwrap()],
            )
            .unwrap();
        assert_eq!(clamped[0].to_f64s(), vec![4.0, 5.0]);
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 0 }),
        );
    }

    #[test]
    fn test_dynamic_slice_partial_evaluation() {
        // Partial evaluation folds known starts and residualizes the read when the input remains unknown.
        let input = Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let start = Array::scalar(1_i32).unwrap();
        let expected = Array::vector(vec![1.0, 2.0]).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = DynamicSliceOperation::new(vec![2]),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, start.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, start.clone()),
                    ],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_dynamic_slice_batching() {
        // Index-varying empty batches preserve formats with no zero encoding without materializing a scalar zero.
        let input = ArrayBatch::replicated(
            Array::from_elements::<f8e8m0fnu>(
                ArrayType::new_static(DataType::F8E8M0FNU, [3]),
                &[f8e8m0fnu::from_f32(1.0).unwrap(); 3],
            )
            .unwrap(),
        );
        let indices = ArrayBatch::new(
            Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, [0]), &[]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(&BatchingContext::new(EagerContext::<Array>::new(), 0), &EmptyRegionDriver, &[input, indices])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[0].value(),
            &Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [0, 2]), vec![]).unwrap()
        );

        let start = Array::scalar(1_i32).unwrap();
        // Replicated starts lift by inserting a zero start for the mapped axis.
        check_operation_batching!(
            @exact,
            operation = DynamicSliceOperation::new(vec![2]),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::matrix(
                        2,
                        4,
                        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    ).unwrap()),
                    (@replicated, start.clone()),
                ],
                outputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 2.0, 5.0, 6.0]).unwrap())],
            }],
        );

        // Bounded ragged inputs are rejected before any window arithmetic, and a missing or extra start index is
        // rejected against the operation's own size count right after.
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        let start = ArrayBatch::replicated(start);
        assert!(matches!(
            DynamicSliceOperation::new(vec![2]).batch(&context, &EmptyRegionDriver, &[ragged_batch(), start.clone()]),
            Err(BatchingError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!("`{DYNAMIC_SLICE_OPERATION_NAME}` does not support bounded ragged array inputs"),
        ));
        let input = ArrayBatch::replicated(Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap());
        assert_eq!(
            DynamicSliceOperation::new(vec![2])
                .batch(&context, &EmptyRegionDriver, &[input.clone()])
                .unwrap_err(),
            BatchingError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            DynamicSliceOperation::new(vec![2])
                .batch(&context, &EmptyRegionDriver, &[input, start.clone(), start])
                .unwrap_err(),
            BatchingError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 3 }),
        );
    }

    #[test]
    fn test_dynamic_slice_batching_scalar_shortcut() {
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        let operation = DynamicSliceOperation::new(vec![]);
        let scalars = ArrayBatch::new(Array::vector(vec![1.0, 2.0]).unwrap(), BatchAxis::new(0)).unwrap();
        let outputs = operation.batch(&context, &EmptyRegionDriver, &[scalars.clone()]).unwrap().into_parts().0;
        assert_eq!(outputs, vec![scalars]);

        // An empty size list cannot turn a per-item vector into a scalar identity operation.
        let vectors =
            ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), BatchAxis::new(0))
                .unwrap();
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[vectors]).unwrap_err(),
            BatchingError::Type(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` expects one start index per input axis (1) but got 0"
            ))),
        );
    }

    #[test]
    fn test_dynamic_slice_batching_with_mapped_indices() {
        // Mapped start indices over a replicated input share one gather: item 0 reads `x[0..2]` and item 1 reads
        // `x[2..4]`, with a leading output batch axis.
        let uniform = ArrayBatch::replicated(Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap());
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[uniform, batch_varying_indices(vec![0, 2])],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().shape().dimensions(), &[Dimension::Static(2), Dimension::Static(2)]);
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0, 1.0, 2.0, 3.0]);

        // Signed and unsigned extremes clamp before narrowing; neither negative wrapping nor signed reinterpretation of
        // `u64::MAX` is part of this raw-index contract. An empty index batch yields an empty stack of windows.
        let shared = Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let both_ends = Array::matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            batch(
                |(input, start)| input.dynamic_slice(&[start], &[2]),
                (shared.clone(), Array::vector(vec![i64::MIN, i64::MAX]).unwrap()),
                (BatchAxis::replicated(), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            ),
            Ok(both_ends.clone()),
        );
        assert_eq!(
            batch(
                |(input, start)| input.dynamic_slice(&[start], &[2]),
                (shared.clone(), Array::vector(vec![0_u64, u64::MAX]).unwrap()),
                (BatchAxis::replicated(), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            ),
            Ok(both_ends),
        );
        assert_eq!(
            batch(
                |(input, start)| input.dynamic_slice(&[start], &[2]),
                (shared, Array::vector(Vec::<i32>::new()).unwrap()),
                (BatchAxis::replicated(), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            ),
            Ok(Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0, 2]), &[]).unwrap()),
        );

        // Paired source/index batching keeps the original integer range. Each extreme selects the clamped end of its
        // own row; item positions never enter the index element type. An empty paired batch gathers a zero-extent
        // window from the empty source batch axis.
        let paired = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2, 4]),
            &(0..8).map(|value| value as f32).collect::<Vec<_>>(),
        )
        .unwrap();
        let own_rows = Array::matrix(2, 2, vec![0.0_f32, 1.0, 6.0, 7.0]).unwrap();
        assert_eq!(
            batch(
                |(input, start)| input.dynamic_slice(&[start], &[2]),
                (paired.clone(), Array::vector(vec![i64::MIN, i64::MAX]).unwrap()),
                (BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            ),
            Ok(own_rows.clone()),
        );
        assert_eq!(
            batch(
                |(input, start)| input.dynamic_slice(&[start], &[2]),
                (paired, Array::vector(vec![0_u64, u64::MAX]).unwrap()),
                (BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            ),
            Ok(own_rows),
        );
        assert_eq!(
            batch(
                |(input, start)| input.dynamic_slice(&[start], &[2]),
                (
                    Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 4]), &[]).unwrap(),
                    Array::vector(Vec::<i32>::new()).unwrap(),
                ),
                (BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            ),
            Ok(Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 2]), &[]).unwrap()),
        );

        // Mapped signed starts count from the end before the clipping gather, exactly as the unbatched kernel resolves
        // each item: `-1` reads the last window, `-9` stays negative after one wrap and clamps to the first window,
        // and `3` clamps to the last valid origin. The clamp-only policy resolves both negative starts to zero.
        let vector = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            batch(
                |(input, start)| input.dynamic_slice(&[start], &[2]),
                (vector.clone(), Array::vector(vec![-1_i32, -9, 3]).unwrap()),
                (BatchAxis::replicated(), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            ),
            Ok(Array::matrix(3, 2, vec![3.0, 4.0, 1.0, 2.0, 3.0, 4.0]).unwrap()),
        );
        assert_eq!(
            batch(
                |(input, start)| input.dynamic_slice_with_negative_indices(&[start], &[2], false),
                (vector, Array::vector(vec![-1_i32, -9, 3]).unwrap()),
                (BatchAxis::replicated(), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            ),
            Ok(Array::matrix(3, 2, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0]).unwrap()),
        );

        // Mixed replicated/mapped coordinates form one index vector per window; the full-width second coordinate
        // clamps to zero. Item 0's row start `-1` counts from the end to row 2, as the unbatched kernel resolves it,
        // and item 1 reads row 2 directly; an empty result window keeps the batch axis and the full-width column
        // extent.
        let grid = Array::matrix(3, 3, (0..9).map(f64::from).collect()).unwrap();
        assert_eq!(
            batch(
                |(input, row, column)| input.dynamic_slice(&[row, column], &[1, 3]),
                (grid.clone(), Array::vector(vec![-1_i32, 2]).unwrap(), Array::scalar(99_i32).unwrap()),
                (BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            ),
            Ok(Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 3]), &[6.0, 7.0, 8.0, 6.0, 7.0, 8.0],)
                .unwrap()),
        );
        assert_eq!(
            batch(
                |(input, row, column)| input.dynamic_slice(&[row, column], &[0, 3]),
                (grid, Array::vector(vec![-1_i32, 2]).unwrap(), Array::scalar(99_i32).unwrap()),
                (BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            ),
            Ok(Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 0, 3]), &[]).unwrap()),
        );

        // The gather keeps mapped placement and host memory for a shared source and for a paired mapped source.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mapped_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let index_type = ArrayType::new_static(DataType::I32, [2])
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let indices =
            ArrayBatch::new(Array::from_elements(index_type, &[0_i32, 2]).unwrap(), BatchAxis::new(0)).unwrap();
        let context =
            BatchingContext::new(EagerContext::<Array>::new(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));
        let shared_type = ArrayType::new_static(DataType::F64, [4])
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let shared = ArrayBatch::replicated(Array::from_elements(shared_type, &[0_f64, 1.0, 2.0, 3.0]).unwrap());
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(&context, &EmptyRegionDriver, &[shared, indices.clone()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].r#type().memory(), Memory::Host { pinned: true });
        assert_eq!(outputs[0].r#type().sharding(), Some(&mapped_sharding));
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0, 1.0, 2.0, 3.0]);
        let paired_type = ArrayType::new_static(DataType::F64, [2, 4])
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(mapped_sharding.clone())
            .unwrap();
        let paired = ArrayBatch::new(
            Array::from_elements(paired_type, &[0_f64, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(&context, &EmptyRegionDriver, &[paired, indices])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].r#type().memory(), Memory::Host { pinned: true });
        assert_eq!(outputs[0].r#type().sharding(), Some(&mapped_sharding));
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0, 1.0, 2.0, 3.0]);

        // A batched input pairs item `i` of the input with item `i` of the indices; item 1's start index 3 is
        // clamped to 2 so the extracted block stays in bounds.
        let input = ArrayBatch::new(
            Array::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[input, batch_varying_indices(vec![1, 3])],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![1.0, 2.0, 6.0, 7.0]);

        // An input batched on a non-leading axis is realigned to the fresh leading batch axis first: the physical `[4,
        // 2]` input carries per-item vectors `[0, 1, 2, 3]` and `[4, 5, 6, 7]` along axis 1.
        let trailing = ArrayBatch::new(
            Array::matrix(4, 2, vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]).unwrap(),
            BatchAxis::new(1),
        )
        .unwrap();
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[trailing, batch_varying_indices(vec![1, 2])],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![1.0, 2.0, 6.0, 7.0]);
    }

    #[test]
    fn test_dynamic_slice_batching_nested() {
        // Both maps supply independent starts while sharing one input and a fixed two-element window.
        let output = batch(
            |(input, starts)| {
                Ok(batch(
                    |(input, start)| input.dynamic_slice(&[start], &[2]),
                    (input, starts),
                    (BatchAxis::replicated(), BatchAxis::new(0)),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            (Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap(), Array::matrix(2, 2, vec![0_i32, 1, 2, 3]).unwrap()),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            output,
            Ok(Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2, 2]),
                &[1.0_f32, 2.0, 2.0, 3.0, 3.0, 4.0, 3.0, 4.0],
            )
            .unwrap()),
        );

        // Nested maps pair each index with its own source at both levels. The inner gather's batching dimensions must
        // be shifted when the outer map adds its source and index axes.
        let output = batch(
            |(input, starts)| {
                Ok(batch(
                    |(input, start)| input.dynamic_slice(&[start], &[2]),
                    (input, starts),
                    (BatchAxis::new(0), BatchAxis::new(0)),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            (
                Array::from_elements(
                    ArrayType::new_static(DataType::F32, [2, 2, 4]),
                    &(0..16).map(|value| value as f32).collect::<Vec<_>>(),
                )
                .unwrap(),
                Array::matrix(2, 2, vec![0_i32, 1, 2, 3]).unwrap(),
            ),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output.r#type().static_shape().unwrap().as_slice(), &[2, 2, 2]);
        assert_eq!(output.to_f64s(), vec![0., 1., 5., 6., 10., 11., 14., 15.]);
    }

    #[test]
    fn test_dynamic_slice_batching_under_tracing() {
        // A shared source with mapped starts wraps the signed starts by the static axis extent, then stages one
        // clipping gather over the packed index vectors, independent of the mapped length: every item reads its own
        // window of the same vector.
        assert_eq!(
            trace_batched_dynamic_slice(ArrayType::new_static(DataType::F32, [4]), BatchAxis::replicated(), 2),
            indoc! {"
                lambda %0:f32[4], %1:i32[2] .
                let %2:i64[] = constant [value=4]
                    %3:i32[] = convert_element_type [data_type=i32] %2
                    %4:i32[] = transfer_to_memory [destination=Device] %3
                    %5:i32[2] = broadcast [output_type=i32[2], output_axes=[]] %4
                    %6:i32[2] = zero_like %1
                    %7:bool[2] = compare [direction=LessThan] %1 %6
                    %8:i32[2] = add %1 %5
                    %9:i32[2] = select %7 %8 %1
                    %10:i32[2, 1] = reshape [shape=[2, 1]] %9
                    %11:f32[2, 2] = gather [
                        dimensions=(offset=[1], collapsed_slice=[], start_index_map=[0], batching=[]),
                        slice_sizes=[2],
                        mode=clip,
                    ] %0 %10
                in (%11)
            "}
            .trim_end(),
        );
        assert_eq!(
            trace_batched_dynamic_slice(ArrayType::new_static(DataType::F32, [4]), BatchAxis::replicated(), 256),
            indoc! {"
                lambda %0:f32[4], %1:i32[256] .
                let %2:i64[] = constant [value=4]
                    %3:i32[] = convert_element_type [data_type=i32] %2
                    %4:i32[] = transfer_to_memory [destination=Device] %3
                    %5:i32[256] = broadcast [output_type=i32[256], output_axes=[]] %4
                    %6:i32[256] = zero_like %1
                    %7:bool[256] = compare [direction=LessThan] %1 %6
                    %8:i32[256] = add %1 %5
                    %9:i32[256] = select %7 %8 %1
                    %10:i32[256, 1] = reshape [shape=[256, 1]] %9
                    %11:f32[256, 2] = gather [
                        dimensions=(offset=[1], collapsed_slice=[], start_index_map=[0], batching=[]),
                        slice_sizes=[2],
                        mode=clip,
                    ] %0 %10
                in (%11)
            "}
            .trim_end(),
        );

        // A mapped source pairs item `i` with index vector `i` through the gather's batching dimensions, again as one
        // gather independent of the mapped length.
        assert_eq!(
            trace_batched_dynamic_slice(ArrayType::new_static(DataType::F32, [2, 4]), BatchAxis::new(0), 2),
            indoc! {"
                lambda %0:f32[2, 4], %1:i32[2] .
                let %2:i64[] = constant [value=4]
                    %3:i32[] = convert_element_type [data_type=i32] %2
                    %4:i32[] = transfer_to_memory [destination=Device] %3
                    %5:i32[2] = broadcast [output_type=i32[2], output_axes=[]] %4
                    %6:i32[2] = zero_like %1
                    %7:bool[2] = compare [direction=LessThan] %1 %6
                    %8:i32[2] = add %1 %5
                    %9:i32[2] = select %7 %8 %1
                    %10:i32[2, 1] = reshape [shape=[2, 1]] %9
                    %11:f32[2, 2] = gather [
                        dimensions=(offset=[1], collapsed_slice=[], start_index_map=[1], batching=[(0, 0)]),
                        slice_sizes=[1, 2],
                        mode=clip,
                    ] %0 %10
                in (%11)
            "}
            .trim_end(),
        );
        assert_eq!(
            trace_batched_dynamic_slice(ArrayType::new_static(DataType::F32, [256, 4]), BatchAxis::new(0), 256),
            indoc! {"
                lambda %0:f32[256, 4], %1:i32[256] .
                let %2:i64[] = constant [value=4]
                    %3:i32[] = convert_element_type [data_type=i32] %2
                    %4:i32[] = transfer_to_memory [destination=Device] %3
                    %5:i32[256] = broadcast [output_type=i32[256], output_axes=[]] %4
                    %6:i32[256] = zero_like %1
                    %7:bool[256] = compare [direction=LessThan] %1 %6
                    %8:i32[256] = add %1 %5
                    %9:i32[256] = select %7 %8 %1
                    %10:i32[256, 1] = reshape [shape=[256, 1]] %9
                    %11:f32[256, 2] = gather [
                        dimensions=(offset=[1], collapsed_slice=[], start_index_map=[1], batching=[(0, 0)]),
                        slice_sizes=[1, 2],
                        mode=clip,
                    ] %0 %10
                in (%11)
            "}
            .trim_end(),
        );

        // Explicit layouts deliberately retain per-item expansion because gather clears layout metadata whereas
        // slicing preserves it: each item's start is extracted, its window sliced, and the windows restacked.
        let shared_layout_type =
            ArrayType::new_static(DataType::F32, [4]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(
            trace_batched_dynamic_slice(shared_layout_type, BatchAxis::replicated(), 2),
            indoc! {"
                lambda %0:f32[4][layout=strided{4}], %1:i32[2] .
                let %2:i32[1] = slice [start_indices=[0], limit_indices=[1]] %1
                    %3:i32[] = reshape [shape=[]] %2
                    %4:f32[2] = dynamic_slice [sizes=[2]] %0 %3
                    %5:f32[1, 2] = reshape [shape=[1, 2]] %4
                    %6:i32[1] = slice [start_indices=[1], limit_indices=[2]] %1
                    %7:i32[] = reshape [shape=[]] %6
                    %8:f32[2] = dynamic_slice [sizes=[2]] %0 %7
                    %9:f32[1, 2] = reshape [shape=[1, 2]] %8
                    %10:f32[2, 2] = concatenate [axis=0] %5 %9
                in (%10)
            "}
            .trim_end(),
        );
        let mapped_layout_type =
            ArrayType::new_static(DataType::F32, [2, 4]).with_layout(Layout::Strided(StridedLayout::new(vec![16, 4])));
        assert_eq!(
            trace_batched_dynamic_slice(mapped_layout_type, BatchAxis::new(0), 2),
            indoc! {"
                lambda %0:f32[2, 4][layout=strided{16,4}], %1:i32[2] .
                let %2:f32[1, 4] = slice [start_indices=[0, 0], limit_indices=[1, 4]] %0
                    %3:f32[4] = reshape [shape=[4]] %2
                    %4:i32[1] = slice [start_indices=[0], limit_indices=[1]] %1
                    %5:i32[] = reshape [shape=[]] %4
                    %6:f32[2] = dynamic_slice [sizes=[2]] %3 %5
                    %7:f32[1, 2] = reshape [shape=[1, 2]] %6
                    %8:f32[1, 4] = slice [start_indices=[1, 0], limit_indices=[2, 4]] %0
                    %9:f32[4] = reshape [shape=[4]] %8
                    %10:i32[1] = slice [start_indices=[1], limit_indices=[2]] %1
                    %11:i32[] = reshape [shape=[]] %10
                    %12:f32[2] = dynamic_slice [sizes=[2]] %9 %11
                    %13:f32[1, 2] = reshape [shape=[1, 2]] %12
                    %14:f32[2, 2] = concatenate [axis=0] %7 %13
                in (%14)
            "}
            .trim_end(),
        );

        // vmap-under-tracing composition: each batch item extracts a window of the differentiated vector at its own
        // start index, so the batching rule must stage the shared gather and its transpose. With `starts = [1, 2]` over
        // `x = [1, 2, 3, 4]` the batch items read `[x1, x2]` and `[x2, x3]`, so `f(x) = sum(stack * w)` with `w = [[1,
        // 2], [3, 4]]` is `f = x1 + 2 * x2 + 3 * x2 + 4 * x3` and the gradient is `[0, 1, 5, 4]`.
        let (value, gradient) = differentiate_at(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap())
            .value_and_gradient(|x| {
                let context = x.context().clone();
                let starts = context
                    .lift(
                        Array::from_elements::<i32>(
                            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2)])),
                            &[1, 2],
                        )
                        .unwrap(),
                    )
                    .unwrap();
                let stacked = batch(
                    |(item, start)| item.dynamic_slice(&[start], &[2]),
                    (x, starts),
                    (BatchAxis::replicated(), BatchAxis::new(0)),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap();
                let weights = context.lift(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()).unwrap();
                (stacked * weights).reduce(&[0, 1], ReductionKind::Sum)
            })
            .unwrap();
        // f = 1 * 2 + 2 * 3 + 3 * 3 + 4 * 4 = 33.
        assert_abs_diff_eq!(value.to_f64s()[0], 33.0, epsilon = 1e-9);
        assert_eq!(gradient.to_f64s(), vec![0.0, 1.0, 5.0, 4.0]);
    }

    #[test]
    fn test_dynamic_slice_differentiation() {
        // Forward mode through `f(x) = dynamic_slice(x, [1], [2])` exercises the captured-index dynamic slice under
        // batched basis tangents.
        let jacobian = differentiate_at(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap())
            .jacobian_forward(|x| {
                let start = index_constant(&x, 1);
                Ok(x.dynamic_slice(&[start], &[2]).unwrap())
            })
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(block.value().to_f64s(), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

        // Finite differences of `sum(dynamic_slice(x)²)` agree with the reverse-mode rule at an in-bounds start and at
        // a start that clamps to the last valid origin. The integer start is a fixed constant, never a perturbed input.
        check_gradient!(
            |x| {
                let start = index_constant(&x, 1);
                let window = x.dynamic_slice(&[start], &[2])?;
                Ok((window.clone() * window).reduce(&[0], ReductionKind::Sum))
            },
            at = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            step = 1e-3,
            tolerance = 1e-6,
        );
        check_gradient!(
            |x| {
                let start = index_constant(&x, 9);
                let window = x.dynamic_slice(&[start], &[2])?;
                Ok((window.clone() * window).reduce(&[0], ReductionKind::Sum))
            },
            at = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            step = 1e-3,
            tolerance = 1e-6,
        );

        // A negative start counts from the end in the forward pass and in its transpose alike: `-1` names origin 3,
        // which the size-2 window clamps to 2, so the Jacobian selects the last two elements.
        let jacobian = differentiate_at(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap())
            .jacobian_forward(|x| {
                let start = index_constant(&x, -1);
                Ok(x.dynamic_slice(&[start], &[2]).unwrap())
            })
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.value().to_f64s(), vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        for start in [-1, -9] {
            check_gradient!(
                |x| {
                    let start = index_constant(&x, start);
                    let window = x.dynamic_slice(&[start], &[2])?;
                    Ok((window.clone() * window).reduce(&[0], ReductionKind::Sum))
                },
                at = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                step = 1e-3,
                tolerance = 1e-6,
            );
        }
    }

    #[test]
    fn test_dynamic_slice_differentiation_pullback_batching_orders() {
        // Batching a pullback and pulling back a batched function must preserve the same indexed linear map.
        let input = Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let batched_pullback = batch(
            |value| {
                let seed = value.context().constant(Array::vector(vec![2.0_f32]).unwrap())?;
                let (_, pullback) = differentiate_at(value).vjp(|value| {
                    let start = value.context().constant(Array::scalar(1_i32).unwrap())?;
                    value.dynamic_slice(&[start], &[1])
                })?;
                pullback.apply(seed)
            },
            input.clone(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(batched_pullback, Ok(Array::matrix(2, 3, vec![0.0_f32, 2.0, 0.0, 0.0, 2.0, 0.0]).unwrap()));

        let (_, pullback) = differentiate_at(input)
            .vjp(|value| {
                Ok(batch(
                    |value| {
                        let start = value.context().constant(Array::scalar(1_i32).unwrap())?;
                        value.dynamic_slice(&[start], &[1])
                    },
                    value,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            })
            .unwrap();
        assert_eq!(
            pullback.apply(Array::matrix(2, 1, vec![2.0_f32, 2.0]).unwrap()),
            Ok(Array::matrix(2, 3, vec![0.0_f32, 2.0, 0.0, 0.0, 2.0, 0.0]).unwrap()),
        );
    }

    #[test]
    fn test_dynamic_slice_differentiation_pullback_higher_order() {
        // Squaring before indexing leaves the primal value as a runtime coefficient of the pullback. Its derivative
        // must survive expanding the indexed backward rule, in both forward-over-reverse and reverse-over-reverse.
        let (gradient, tangent) = differentiate_at(Array::vector(vec![3.0_f32, 5.0, 7.0]).unwrap())
            .jvp(Array::vector(vec![1.0_f32; 3]).unwrap(), |value| {
                let seed = value.context().constant(Array::vector(vec![1.0_f32]).unwrap())?;
                let (_, pullback) = differentiate_at(value).vjp(|value| {
                    let start = value.context().constant(Array::scalar(1_i32).unwrap())?;
                    (value.clone() * value).dynamic_slice(&[start], &[1])
                })?;
                pullback.apply(seed)
            })
            .unwrap();
        assert_eq!(gradient, Array::vector(vec![0.0_f32, 10.0, 0.0]).unwrap());
        assert_eq!(tangent, Array::vector(vec![0.0_f32, 2.0, 0.0]).unwrap());

        let (_, pullback) = differentiate_at(Array::vector(vec![3.0_f32, 5.0, 7.0]).unwrap())
            .vjp(|value| {
                let seed = value.context().constant(Array::vector(vec![1.0_f32]).unwrap())?;
                let (_, pullback) = differentiate_at(value).vjp(|value| {
                    let start = value.context().constant(Array::scalar(1_i32).unwrap())?;
                    (value.clone() * value).dynamic_slice(&[start], &[1])
                })?;
                pullback.apply(seed)
            })
            .unwrap();
        assert_eq!(
            pullback.apply(Array::vector(vec![1.0_f32; 3]).unwrap()),
            Ok(Array::vector(vec![0.0_f32, 2.0, 0.0]).unwrap())
        );
    }

    #[test]
    fn test_dynamic_slice_differentiation_array_ir() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let start = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::DynamicSlice(DynamicSliceOperation::new(vec![2]))),
                Vec::new(),
                vec![input, start],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        // The dynamically shaped input routes through one linear call whose residuals are the scalar start and the
        // retained input extent; its transpose region writes the cotangent into a zero rebuilt from that extent.
        assert_eq!(linearization.residual_count(), 2);
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:i32[], %2:dimension<extent ∈ [2, 6)> .
                let %3:f64[2] = linear_call [residual_count=2] %1 %2 %0 [
                    forward={
                        lambda %0:i32[], %1:dimension<extent ∈ [2, 6)>, %2:f64[extent] .
                        let %3:f64[2] = dynamic_slice [sizes=[2]] %2 %0
                        in (%3)
                    },
                    transpose={
                        lambda %0:i32[], %1:dimension<extent ∈ [2, 6)>, %2:f64[2] .
                        let %3:f64[extent] = zero [type=f64[extent]] %1
                            %4:f64[extent] = dynamic_update_slice %3 %2 %0
                        in (%4)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[2], %1:i32[], %2:dimension<extent ∈ [2, 6)> .
                let %3:f64[extent] = linear_call [residual_count=2] %1 %2 %0 [
                    forward={
                        lambda %0:i32[], %1:dimension<extent ∈ [2, 6)>, %2:f64[2] .
                        let %3:f64[extent] = zero [type=f64[extent]] %1
                            %4:f64[extent] = dynamic_update_slice %3 %2 %0
                        in (%4)
                    },
                    transpose={
                        lambda %0:i32[], %1:dimension<extent ∈ [2, 6)>, %2:f64[extent] .
                        let %3:f64[2] = dynamic_slice [sizes=[2]] %2 %0
                        in (%3)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()),
                ArrayIrValue::Array(Array::scalar(1_i32).unwrap()),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0]).unwrap()));
        let residuals = primal_outputs.split_off(1);

        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 10.0, 11.0, 12.0]).unwrap())];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 11.0]).unwrap())]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![5.0_f64, 7.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 7.0, 0.0]).unwrap())]),
        );
    }

    #[test]
    fn test_dynamic_slice_transposition() {
        // Slice a [1, 2] block at start (1, 1) of a [2, 3] input: the input is linear and the scalar start indices are
        // the known inputs. The sliced output and its cotangent have shape [1, 2].
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let cotangent = Array::matrix(1, 2, vec![5.0, 7.0]).unwrap();
        let sizes = vec![1, 2];

        check_operation_transposition!(
            @exact,
            operation = DynamicSliceOperation::new(sizes),
            cases = [{
                inputs = [
                    (@linear(type = input_type)),
                    (@known, Array::scalar(1_i32).unwrap()),
                    (@known, Array::scalar(1_i32).unwrap()),
                ],
                output_cotangents = [cotangent],
                input_cotangents = [Array::matrix(2, 3, vec![0.0, 0.0, 0.0, 0.0, 5.0, 7.0]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_dynamic_slice_transposition_zero_cotangent() {
        // A structural-zero output cotangent contributes nothing: the rule returns before reading the start indices
        // and leaves the input accumulator at its structural-zero default. The same holds when no cotangent is needed.
        let operation = DynamicSliceOperation::new(vec![2]);
        let input_type = ArrayType::new_static(DataType::F64, [4]);
        let output_type = ArrayType::new_static(DataType::F64, [2]);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut transpose = TranspositionContext::new(context.clone());
        let start = context.lift(Array::scalar(1_i32).unwrap()).unwrap();
        let inputs = [PartialValue::Unknown(input_type.clone()), PartialValue::Known(start)];
        let accumulators = transpose.cotangent_accumulators(&inputs, &[]).unwrap();
        let zero_outputs = [MaybeZero::Zero(output_type.cotangent().unwrap())];
        operation
            .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &zero_outputs, &accumulators)
            .unwrap();
        let cotangents = transpose.take_cotangents(&accumulators).unwrap();
        assert_eq!(cotangents.len(), 2);
        assert!(cotangents[0].is_zero());
        assert!(cotangents[1].is_zero());
        assert!(context.builder().borrow().instructions().is_empty());
        let outputs = [MaybeZero::Value(context.input(output_type.cotangent().unwrap()))];
        // Unknown starts and a dynamic input would fail the nonzero, needed path. Neither is inspected when the
        // input cotangent is unneeded, so this still stages no instructions.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(8)).unwrap());
        let unneeded_inputs = [
            PartialValue::Unknown(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]))),
            PartialValue::Unknown(ArrayType::scalar(DataType::I32)),
        ];
        let unneeded = transpose.cotangent_accumulators(&unneeded_inputs, &[false, false]).unwrap();
        operation
            .transpose(&mut transpose, &EmptyRegionDriver, &unneeded_inputs, &outputs, &unneeded)
            .unwrap();
        assert!(context.builder().borrow().instructions().is_empty());

        // A needed input cotangent writes the output cotangent into a zero of the input type at the known start.
        let needed = transpose.cotangent_accumulators(&inputs, &[]).unwrap();
        operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &needed).unwrap();
        let builder = context.builder();
        assert_eq!(
            builder
                .borrow()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().to_string())
                .collect::<Vec<_>>(),
            vec!["zero [type=f64[4]]", "dynamic_update_slice"],
        );
        let cotangents = transpose.take_cotangents(&needed).unwrap();
        assert!(!cotangents[0].is_zero());
        assert!(cotangents[1].is_zero());

        // Integer starts have no tangent space, so a start that reaches the rule as a linear input is rejected.
        let unknown_start =
            [PartialValue::Unknown(input_type), PartialValue::Unknown(ArrayType::scalar(DataType::I32))];
        let accumulators = transpose.cotangent_accumulators(&unknown_start, &[]).unwrap();
        assert!(matches!(
            operation.transpose(&mut transpose, &EmptyRegionDriver, &unknown_start, &outputs, &accumulators),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == format!("`{DYNAMIC_SLICE_OPERATION_NAME}` transpose requires known start indices"),
        ));

        // Arity is validated before any cotangent is inspected.
        assert_eq!(
            operation.transpose(&mut transpose, &EmptyRegionDriver, &[], &outputs, &accumulators).unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &[], &accumulators).unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            operation
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &accumulators[..1])
                .unwrap_err(),
            DifferentiationError::InvalidAccumulatorCount { expected: 2, actual: 1 },
        );
    }

    #[test]
    fn test_dynamic_slice_transposition_dynamic_input() {
        let elements = DimensionVariable::new("elements", DimensionBounds::new(4, Some(8)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(elements)]));

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(dynamic_type);
        let start = builder.add_input(ArrayType::scalar(DataType::I32));
        let output = builder
            .add_instruction(DynamicSliceOperation::new(vec![2]), Vec::new(), vec![input, start], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap_err(),
            TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` transpose requires a statically shaped input but got `f64[elements]`"
            ))
            .into(),
        );
    }

    #[test]
    fn test_dynamic_slice_transposition_array_ir() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [5]).into());
        let start = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder
            .add_instruction(
                ArrayOperation::DynamicSlice(DynamicSliceOperation::new(vec![2])),
                Vec::new(),
                vec![input, start],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert!(pullback.output_ids().is_empty());
        // The buffer's selected block is read, incremented, and written back through a dynamic update at the same
        // runtime start; no dense zero gradient is constructed.
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:ref<f64[5]>, %2:i32[] .
                let %3:f64[5] = reference_read %1
                    %4:f64[2] = dynamic_slice [sizes=[2]] %3 %2
                    %5:f64[2] = add %4 %0
                    %6:f64[5] = dynamic_update_slice %3 %5 %2
                    () = reference_write %1 %6
                in ()
            "}
            .trim_end(),
        );

        // Different runtime starts share one transformed program, including the forward operation's clamping past the
        // end and its wrapping of a negative start. Repeated calls add to the prepopulated buffer instead of resetting
        // earlier contributions.
        let buffer = ArrayIrValue::Array(Array::vector(vec![10.0_f64; 5]).unwrap()).reference_new().unwrap();
        let seed = ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0]).unwrap());
        assert_eq!(
            pullback.interpret(vec![seed.clone(), buffer.clone(), ArrayIrValue::Array(Array::scalar(1_i32).unwrap())]),
            Ok(vec![]),
        );
        assert_eq!(
            buffer.read(),
            Ok(ArrayIrValue::Array(Array::vector(vec![10.0_f64, 12.0, 13.0, 10.0, 10.0]).unwrap()))
        );
        assert_eq!(
            pullback.interpret(vec![seed.clone(), buffer.clone(), ArrayIrValue::Array(Array::scalar(20_i32).unwrap())]),
            Ok(vec![]),
        );
        assert_eq!(
            buffer.read(),
            Ok(ArrayIrValue::Array(Array::vector(vec![10.0_f64, 12.0, 13.0, 12.0, 13.0]).unwrap()))
        );
        assert_eq!(
            pullback.interpret(vec![seed.clone(), buffer.clone(), ArrayIrValue::Array(Array::scalar(-1_i32).unwrap())]),
            Ok(vec![]),
        );
        assert_eq!(
            buffer.read(),
            Ok(ArrayIrValue::Array(Array::vector(vec![10.0_f64, 12.0, 13.0, 14.0, 16.0]).unwrap()))
        );

        // The same retained rule returns a dense value when requested, while Ignore constructs no scratch buffer and
        // emits no arithmetic at all.
        let returned = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            returned.interpret(vec![seed, ArrayIrValue::Array(Array::scalar(1_i32).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 2.0, 3.0, 0.0, 0.0]).unwrap())]),
        );
        let ignored = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Ignore]).unwrap();
        assert!(ignored.instructions().is_empty());
        assert!(ignored.output_ids().is_empty());
    }

    #[test]
    fn test_dynamic_slice_transposition_array_ir_batching() {
        let destination = ArrayReference::new(Array::matrix(2, 3, vec![10.0_f32; 6]).unwrap());
        let result = batch(
            |(value, destination)| {
                let seed = value.context().lift(ArrayIrValue::Array(Array::vector(vec![2.0_f32]).unwrap()))?;
                let (_, pullback) = differentiate_at(value).vjp(|value| {
                    let start = value.context().constant(ArrayIrValue::Array(Array::scalar(1_i32).unwrap()))?;
                    Ok(value
                        .context()
                        .bind(
                            ArrayOperation::DynamicSlice(DynamicSliceOperation::new(vec![1])),
                            Vec::new(),
                            &[value.clone(), start],
                        )?
                        .remove(0))
                })?;
                pullback.apply_with_destinations(
                    CotangentSeed::Value(seed),
                    CotangentDestination::Reference(destination.clone()),
                )?;
                destination.read()
            },
            (
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                ArrayIrValue::Reference(destination.clone()),
            ),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        );
        // Each mapped buffer keeps its existing contents outside the selected coordinate, and batching preserves
        // additive updates to the selected coordinate rather than sharing one member's temporary storage.
        assert_eq!(
            result,
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![10.0_f32, 12.0, 10.0, 10.0, 12.0, 10.0]).unwrap()))
        );
        assert_eq!(destination.read(), Ok(Array::matrix(2, 3, vec![10.0_f32, 12.0, 10.0, 10.0, 12.0, 10.0]).unwrap()));
    }

    #[test]
    fn test_dynamic_slice_transposition_array_ir_higher_order() {
        let (output, pullback) = differentiate_at(ArrayIrValue::Array(Array::vector(vec![3.0_f32, 5.0, 7.0]).unwrap()))
            .vjp(|value| {
                let initial =
                    value.context().constant(ArrayIrValue::Array(Array::vector(vec![10.0_f32; 3]).unwrap()))?;
                let destination = initial.reference_new()?;
                let start = value.context().constant(ArrayIrValue::Array(Array::scalar(1_i32).unwrap()))?;
                let selected = value
                    .context()
                    .bind(
                        ArrayOperation::DynamicSlice(DynamicSliceOperation::new(vec![1])),
                        Vec::new(),
                        &[value.clone(), start],
                    )?
                    .remove(0);
                let seed = selected
                    .context()
                    .bind(
                        ArrayOperation::from(MulOperation::<ArrayType>::new()),
                        Vec::new(),
                        &[selected.clone(), selected.clone()],
                    )?
                    .remove(0);
                let (_, pullback) = differentiate_at(value).vjp(|value| {
                    let start = value.context().constant(ArrayIrValue::Array(Array::scalar(1_i32).unwrap()))?;
                    Ok(value
                        .context()
                        .bind(
                            ArrayOperation::DynamicSlice(DynamicSliceOperation::new(vec![1])),
                            Vec::new(),
                            &[value.clone(), start],
                        )?
                        .remove(0))
                })?;
                pullback.apply_with_destinations(
                    CotangentSeed::Value(seed),
                    CotangentDestination::Reference(destination.clone()),
                )?;
                destination.read()
            })
            .unwrap();
        assert_eq!(output, ArrayIrValue::Array(Array::vector(vec![10.0_f32, 35.0, 10.0]).unwrap()));
        // The inner slice receives the caller's buffer directly. Its seed depends on the differentiated value, so the
        // outer pullback must differentiate the emitted buffer read/update/write operations as well.
        assert_eq!(
            pullback.apply(ArrayIrValue::Array(Array::vector(vec![1.0_f32; 3]).unwrap())),
            Ok(ArrayIrValue::Array(Array::vector(vec![0.0_f32, 10.0, 0.0]).unwrap())),
        );
    }

    #[test]
    fn test_array_type_dynamic_slice() {
        let host_input = ArrayType::new_static(DataType::F32, [4]).with_memory(Memory::Host { pinned: true });
        let host_index = ArrayType::scalar(DataType::I32).with_memory(Memory::Host { pinned: true });
        assert_eq!(
            host_input.dynamic_slice(std::slice::from_ref(&host_index), &[2]).unwrap().memory(),
            Memory::Host { pinned: true }
        );
        let laid_out = host_input.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(laid_out.dynamic_slice(std::slice::from_ref(&host_index), &[4]), Ok(laid_out.clone()));
        assert_eq!(
            host_input.dynamic_slice(&[ArrayType::scalar(DataType::I32)], &[2]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` input and start indices must share one memory space but start \
                 index 0 resides in `Device` and the input resides in `Host[Pinned]`"
            ))))
        );

        // Resizing keeps explicit placement only when the resulting dimension remains evenly divisible.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let sharding = sharding.with_unreduced_axes(["m"]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [4, 4]).with_sharding(sharding.clone()).unwrap();
        let starts = [ArrayType::scalar(DataType::I32), ArrayType::scalar(DataType::I32)];
        assert_eq!(input.dynamic_slice(&starts, &[2, 4]).unwrap().sharding(), Some(&sharding));
        assert_eq!(
            input.dynamic_slice(&starts, &[3, 4]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` on a dimension sharded over explicit mesh axes requires the output \
                 size (3) at axis 0 to be divisible by the mesh-axis product (2)"
            ))))
        );
    }

    #[test]
    fn test_array_type_dynamic_slice_index_placement() {
        // A replicated start index contributes no manual-axis variation, so an unsharded input stays unsharded, for
        // both callers of the shared index-placement rule.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new_static(DataType::F32, [4]);
        let update = ArrayType::new_static(DataType::F32, [2]);
        let replicated_index =
            ArrayType::scalar(DataType::I32).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        assert_eq!(input.dynamic_slice(std::slice::from_ref(&replicated_index), &[2]), Ok(update.clone()));
        assert_eq!(input.dynamic_update_slice(&update, std::slice::from_ref(&replicated_index)), Ok(input.clone()));

        // A start index varying over a manual axis makes the output vary over it: an unsharded output acquires a
        // replicated placement on the index's mesh carrying that variation.
        let varying_index = ArrayType::scalar(DataType::I32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let acquired = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
        assert_eq!(
            input.dynamic_slice(std::slice::from_ref(&varying_index), &[2]).unwrap().sharding(),
            Some(&acquired),
        );
        assert_eq!(
            input.dynamic_update_slice(&update, std::slice::from_ref(&varying_index)).unwrap().sharding(),
            Some(&acquired),
        );

        // An input that is already sharded keeps its placement and reduction state and only gains the variation.
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["m"]).unwrap();
        let sharded_input = input.clone().with_sharding(unreduced.clone()).unwrap();
        let sharded_update = update.clone().with_sharding(unreduced.clone()).unwrap();
        let combined = unreduced.with_varying_manual_axes(["x"]).unwrap();
        assert_eq!(
            sharded_input.dynamic_slice(std::slice::from_ref(&varying_index), &[2]).unwrap().sharding(),
            Some(&combined),
        );
        assert_eq!(
            sharded_input
                .dynamic_update_slice(&sharded_update, std::slice::from_ref(&varying_index))
                .unwrap()
                .sharding(),
            Some(&combined),
        );

        // Every sharded start index must use one mesh, whether or not it contributes variation and in either order,
        // even when the array itself is unsharded.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let other_replicated_index =
            ArrayType::scalar(DataType::I32).with_sharding(Sharding::replicated(other_mesh.clone(), 0)).unwrap();
        let matrix = ArrayType::new_static(DataType::F32, [4, 4]);
        let block = ArrayType::new_static(DataType::F32, [2, 2]);
        let same_mesh_error = Err(ProgramError::Type(TypeError::invalid(format!(
            "`{DYNAMIC_SLICE_OPERATION_NAME}` start indices must use the same mesh"
        ))));
        let same_mesh_update_error = Err(ProgramError::Type(TypeError::invalid(format!(
            "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` start indices must use the same mesh"
        ))));
        assert_eq!(
            matrix.dynamic_slice(&[replicated_index.clone(), other_replicated_index.clone()], &[2, 2]),
            same_mesh_error,
        );
        assert_eq!(
            matrix.dynamic_slice(&[varying_index.clone(), other_replicated_index.clone()], &[2, 2]),
            same_mesh_error,
        );
        assert_eq!(
            matrix.dynamic_slice(&[other_replicated_index.clone(), varying_index.clone()], &[2, 2]),
            same_mesh_error,
        );
        assert_eq!(
            matrix.dynamic_update_slice(&block, &[replicated_index, other_replicated_index.clone()]),
            same_mesh_update_error,
        );
        assert_eq!(
            matrix.dynamic_update_slice(&block, &[varying_index.clone(), other_replicated_index.clone()]),
            same_mesh_update_error,
        );
        assert_eq!(
            matrix.dynamic_update_slice(&block, &[other_replicated_index, varying_index]),
            same_mesh_update_error,
        );
    }

    #[test]
    fn test_array_dynamic_slice() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        // Dynamic start indices clamp so the block stays in bounds.
        let start = [Array::scalar(4i64).unwrap()];
        assert_eq!(vector.dynamic_slice(&start, &[2]).unwrap(), Array::vector(vec![4.0, 5.0]).unwrap());
        // Index decoding is typed and supports sub-byte integers directly; a negative start counts from the end.
        let start = [Array::scalar(i4::new(-1).unwrap()).unwrap()];
        assert_eq!(vector.dynamic_slice(&start, &[2]).unwrap(), Array::vector(vec![4.0, 5.0]).unwrap());
        assert_eq!(
            vector.dynamic_slice_with_negative_indices(&start, &[2], false).unwrap(),
            Array::vector(vec![1.0, 2.0]).unwrap(),
        );

        // A nonscalar start must fail before any indexing takes place.
        assert_eq!(
            Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap()
                .dynamic_slice(&[Array::scalar(0_i32).unwrap(), Array::vector(vec![1.0, 2.0]).unwrap()], &[1, 2]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` start index 1 must be a scalar integer but has type `f64[2]`"
            )))),
        );
    }

    #[test]
    fn test_array_dynamic_slice_unsigned_extreme() {
        let input = Array::vector(vec![1_i32, 2, 3]).unwrap();
        assert_eq!(
            input.dynamic_slice(&[Array::scalar(u64::MAX).unwrap()], &[1]),
            Ok(Array::vector(vec![3_i32]).unwrap())
        );
        assert_eq!(
            input.dynamic_slice(&[Array::scalar(i64::MIN).unwrap()], &[1]),
            Ok(Array::vector(vec![1_i32]).unwrap())
        );
    }

    #[test]
    fn test_array_ir_value_dynamic_slice() {
        let input = ArrayIrValue::Array(Array::vector(vec![10i32, 20, 30]).unwrap());
        let start = ArrayIrValue::Array(Array::scalar(1i32).unwrap());
        assert_eq!(input.dynamic_slice(&[start], &[1]), Ok(ArrayIrValue::Array(Array::vector(vec![20i32]).unwrap())));
    }

    #[test]
    fn test_dynamic_slice_array_ir() {
        let operation = DynamicSliceOperation::<ArrayIrType>::from_rank(2).with_strides(vec![1, 2]).unwrap();

        // Operation identity, accessors, and the conservative constructor assertion effect. Unit strides are always rendered.
        assert_eq!(operation.name(), DYNAMIC_SLICE_OPERATION_NAME);
        assert_eq!(operation.strides(), &[1, 2]);
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(
            format!("{operation}"),
            "dynamic_slice [strides=[1, 2], bounds=checked, requires_runtime_assertion=true]"
        );
        assert_eq!(
            format!("{}", DynamicSliceOperation::<ArrayIrType>::from_rank(1)),
            "dynamic_slice [strides=[1], bounds=checked, requires_runtime_assertion=true]"
        );

        // The constructor rejects a stride list of the wrong length and zero strides.
        assert_eq!(
            DynamicSliceOperation::<ArrayIrType>::from_rank(2).with_strides(vec![1]),
            Err(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` `strides` has length 1 but input has rank 2"
            ))),
        );
        assert_eq!(
            DynamicSliceOperation::<ArrayIrType>::from_rank(2).with_strides(vec![1, 0]),
            Err(TypeError::invalid(format!("`{DYNAMIC_SLICE_OPERATION_NAME}` stride must be positive on axis 1"))),
        );

        // Program rendering lists the array input followed by the first-class starts and sizes, and the result takes
        // its extents from the size inputs.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, DynamicSliceOperation<ArrayIrType>>::new();
        let program_input = builder.add_input(ArrayType::new_static(DataType::F64, [4, 6]).into());
        let program_start_0 = builder.add_input(DimensionValue::constant(0).unwrap().r#type().into_owned().into());
        let program_start_1 = builder.add_input(DimensionValue::constant(1).unwrap().r#type().into_owned().into());
        let program_size_0 = builder.add_input(DimensionType::from(rows).into());
        let program_size_1 = builder.add_input(DimensionValue::constant(2).unwrap().r#type().into_owned().into());
        let program_output = builder
            .add_instruction(
                operation,
                Vec::new(),
                vec![program_input, program_start_0, program_start_1, program_size_0, program_size_1],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, ArrayIrValue<Array>>(
                vec![program_output],
                vec![Placeholder; 5],
                Placeholder,
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[4, 6], %1:dimension<0>, %2:dimension<1>, %3:dimension<rows ∈ [1, 4)>, %4:dimension<2> .
                let %5:f64[rows, 2] = dynamic_slice [strides=[1, 2], bounds=checked, requires_runtime_assertion=true] %0 %1 %2 %3 %4
                in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dynamic_slice_array_ir_with_input_types() {
        let input_types = vec![
            ArrayType::new_static(DataType::F32, [4]).into(),
            DimensionValue::constant(1).unwrap().r#type().into_owned().into(),
            DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
        ];
        let operation = DynamicSliceOperation::<ArrayIrType>::from_rank(1).with_input_types(&input_types).unwrap();
        assert!(!operation.requires_runtime_assertion());
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        assert!(operation.clone().with_strides(vec![1]).unwrap().requires_runtime_assertion());
        assert!(operation.clone().with_bounds(DynamicSliceBounds::Clamp).requires_runtime_assertion());
        let mut wider = input_types.clone();
        wider[1] = DimensionType::new("start", DimensionBounds::new(0, Some(4)).unwrap()).into();
        assert_eq!(
            operation.infer_output_types(&wider, &[]),
            Err(TypeError::invalid(
                "`dynamic_slice` was constructed without a runtime window check but these input types require one"
                    .to_string(),
            ))
        );
        let clamped = operation.with_bounds(DynamicSliceBounds::Clamp).with_input_types(&wider).unwrap();
        assert!(!clamped.requires_runtime_assertion());
        assert_eq!(clamped.bounds(), DynamicSliceBounds::Clamp);
    }

    #[test]
    fn test_dynamic_slice_array_ir_type_inference() {
        // An identity slice passes the input through, keeping its explicit layout.
        let input =
            ArrayType::new_static(DataType::I32, [4]).with_layout(Layout::Strided(StridedLayout::new(vec![-4])));
        let start = DimensionValue::constant(0).unwrap().r#type().into_owned();
        let size = DimensionValue::constant(4).unwrap().r#type().into_owned();
        assert_eq!(
            DynamicSliceOperation::<ArrayIrType>::from_rank(1)
                .infer_output_types(&[input.clone().into(), start.into(), size.into()], &[]),
            Ok(vec![input.into()]),
        );

        // Declared bounds disprove a slice before its dimensions become concrete: the minimum limit is checked against
        // the static extent, or against the exclusive upper bound of a dynamic extent, while an unbounded dynamic axis
        // leaves the check to execution. A variable-sized result takes the size's dimension.
        let input = ArrayType::new_static(DataType::I32, [4]);
        let dimension =
            |extent: usize| ArrayIrType::Dimension(DimensionValue::constant(extent).unwrap().r#type().into_owned());
        let bounded = DimensionVariable::new("bounded", DimensionBounds::new(1, Some(4)).unwrap());
        let bounded_input = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(bounded)]));
        let unbounded = DimensionVariable::new("unbounded", DimensionBounds::new(1, None).unwrap());
        let unbounded_input = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(unbounded)]));
        let count = DimensionVariable::new("count", DimensionBounds::new(0, Some(3)).unwrap());
        check_operation_type_inference!(
            operation = DynamicSliceOperation::<ArrayIrType>::from_rank(1),
            cases = [
                {
                    input_types = [ArrayIrType::Array(bounded_input.clone()), dimension(1), dimension(2)],
                    output_types = [ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2]))],
                },
                {
                    input_types = [ArrayIrType::Array(unbounded_input), dimension(1), dimension(9)],
                    output_types = [ArrayIrType::Array(ArrayType::new_static(DataType::I32, [9]))],
                },
                {
                    input_types = [
                        ArrayIrType::Array(input.clone()),
                        dimension(0),
                        ArrayIrType::Dimension(DimensionType::from(count.clone())),
                    ],
                    output_types = [ArrayIrType::Array(ArrayType::new(
                        DataType::I32,
                        Shape::new(vec![Dimension::Dynamic(count)]),
                    ))],
                },
                {
                    input_types = [ArrayIrType::Array(input.clone()), dimension(3), dimension(2)],
                    error = format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` minimum limit 5 exceeds maximum input extent 4 on \
                         axis 0"
                    ),
                },
                {
                    input_types = [ArrayIrType::Array(bounded_input), dimension(2), dimension(3)],
                    error = format!(
                        "`{DYNAMIC_SLICE_OPERATION_NAME}` minimum limit 5 exceeds maximum input extent 3 on \
                         axis 0"
                    ),
                },
                {
                    input_types = [
                        ArrayIrType::Array(input.clone()),
                        ArrayIrType::Dimension(DimensionType::new(
                            "huge",
                            DimensionBounds::new(usize::MAX, None).unwrap(),
                        )),
                        dimension(2),
                    ],
                    error = format!("`{DYNAMIC_SLICE_OPERATION_NAME}` minimum limit overflows `usize` on axis 0"),
                },
                {
                    input_types = [],
                    error = format!("`{DYNAMIC_SLICE_OPERATION_NAME}` expects an array input"),
                },
                {
                    input_types = [dimension(0), dimension(0), dimension(1)],
                    error = "expected array type but got dimension type",
                },
                {
                    input_types = [ArrayIrType::Array(input.clone()), dimension(0)],
                    error = "expected 3 inputs but got 2",
                },
                {
                    input_types = [ArrayIrType::Array(input.clone()), ArrayIrType::Array(input.clone()), dimension(1)],
                    error = "expected dimension type but got array type",
                },
            ],
        );
        // The payload's stride count must match the input rank before any bound is inspected.
        check_operation_type_inference!(
            operation = DynamicSliceOperation::<ArrayIrType>::from_rank(2),
            cases = [{
                input_types = [ArrayIrType::Array(input), dimension(0), dimension(0), dimension(1), dimension(1)],
                error = format!("`{DYNAMIC_SLICE_OPERATION_NAME}` `strides` has length 2 but input has rank 1"),
            }],
        );
    }

    #[test]
    fn test_dynamic_slice_array_ir_reference_discharge() {
        // Replay preserves the complete slicing payload and its output type. Shared replay and reference rejection
        // are covered by the reference-discharge macro tests.
        let expected = DynamicSliceOperation::<ArrayIrType>::from_rank(1).with_strides(vec![2]).unwrap();
        let operation = ArrayIrOperation::DynamicSlice(expected.clone());
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = ReferenceDischargeContext::<_, ArrayReferenceDischarge>::new(trace.clone());
        let inputs = [
            ReferenceDischargeValue::Value(trace.input(ArrayType::new_static(DataType::I32, [4]).into())),
            ReferenceDischargeValue::Value(
                trace.input(DimensionValue::constant(1).unwrap().r#type().into_owned().into()),
            ),
            ReferenceDischargeValue::Value(
                trace.input(DimensionValue::constant(2).unwrap().r#type().into_owned().into()),
            ),
        ];
        let outputs = operation.discharge_references(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        let ReferenceDischargeValue::Value(output) = &outputs[0] else {
            panic!("expected a value carrier but got {}", outputs[0]);
        };
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])));
        let builder = trace.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        let ArrayIrOperation::DynamicSlice(staged) = builder.instructions()[0].operation() else {
            panic!("expected a staged mixed dynamic slice");
        };
        assert_eq!(staged, &expected);
    }

    #[test]
    fn test_dynamic_slice_array_ir_interpretation() {
        let input_type =
            ArrayType::new_static(DataType::I32, [4]).with_layout(Layout::Strided(StridedLayout::new(vec![-4])));
        let input = ArrayIrValue::Array(Array::from_elements(input_type.clone(), &[10i32, 20, 30, 40]).unwrap());
        let start = ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap());
        let size = ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap());
        let output = input.dynamic_slice_with_dimensions(&[start], &[size], &[1]).unwrap();
        assert_eq!(output, input);
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(input_type));

        let input = ArrayIrValue::Array(Array::vector(vec![10i32, 20, 30, 40]).unwrap());
        let start = ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap());
        let size = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            input.dynamic_slice_with_dimensions(std::slice::from_ref(&start), std::slice::from_ref(&size), &[2]),
            Ok(ArrayIrValue::Array(Array::vector(vec![20i32, 40]).unwrap())),
        );
        let end = ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap());
        let zero = ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap());
        assert_eq!(
            input.dynamic_slice_with_dimensions(std::slice::from_ref(&end), &[zero], &[1]),
            Ok(ArrayIrValue::Array(Array::vector(Vec::<i32>::new()).unwrap())),
        );
        assert!(matches!(
            input.dynamic_slice_with_dimensions(&[end], &[size.clone()], &[1]),
            Err(ProgramError::InvalidArgument { message })
                if message == format!("`{DYNAMIC_SLICE_OPERATION_NAME}` limit 6 exceeds input axis 0 extent 4"),
        ));

        // Bound counts are validated against the input rank, naming the list that is wrong, and the span and limit
        // arithmetic reports overflow instead of wrapping.
        assert_eq!(
            input.dynamic_slice_with_dimensions(&[start.clone()], &[], &[1]),
            Err(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` expects one size per input axis (1) but got 0"
            ))
            .into()),
        );
        assert_eq!(
            input.dynamic_slice_with_dimensions(&[], &[size.clone()], &[1]),
            Err(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` expects one start index per input axis (1) but got 0"
            ))
            .into()),
        );
        let huge = ArrayIrValue::Dimension(DimensionValue::constant(i64::MAX as usize).unwrap());
        assert_eq!(
            input.dynamic_slice_with_dimensions(std::slice::from_ref(&start), std::slice::from_ref(&huge), &[3]),
            Err(TypeError::invalid(format!("`{DYNAMIC_SLICE_OPERATION_NAME}` span overflows `usize` on axis 0")).into()),
        );
        assert_eq!(
            input.dynamic_slice_with_dimensions(std::slice::from_ref(&huge), std::slice::from_ref(&huge), &[2]),
            Err(TypeError::invalid(format!("`{DYNAMIC_SLICE_OPERATION_NAME}` limit overflows `usize` on axis 0"))
                .into()),
        );

        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input =
            ArrayIrValue::Array(Array::matrix(3, 4, (0..12).map(|value| value as f64).collect::<Vec<_>>()).unwrap());
        let dimension = |extent| ArrayIrValue::Dimension(DimensionValue::constant(extent).unwrap());
        let output = context
            .bind(
                DynamicSliceOperation::<ArrayIrType>::from_rank(2),
                Vec::new(),
                &[input.clone(), dimension(1), dimension(1), dimension(2), dimension(2)],
            )
            .unwrap();
        assert_eq!(output, vec![ArrayIrValue::Array(Array::matrix(2, 2, vec![5.0, 6.0, 9.0, 10.0]).unwrap())]);
        assert_eq!(
            DynamicSliceOperation::<ArrayIrType>::from_rank(2).interpret(&context, &EmptyRegionDriver, &[input]),
            Err(ProgramError::InvalidInputCount { expected: 5, actual: 1 }),
        );

        // Repeated dimension identities must agree even when calling the eager capability without a program boundary.
        let extent_type = DimensionType::new("extent", DimensionBounds::new(1, Some(4)).unwrap());
        let start = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 1).unwrap());
        let size = ArrayIrValue::Dimension(DimensionValue::new(extent_type, 2).unwrap());
        assert_eq!(
            ArrayIrValue::Array(Array::vector(vec![1_i32, 2, 3, 4]).unwrap()).dynamic_slice_with_dimensions(
                &[start],
                &[size],
                &[1]
            ),
            Err(ProgramError::Type(
                DimensionError::InputDimensionMismatch { dimension: "extent".to_string(), expected: 1, actual: 2 }
                    .into()
            )),
        );
    }

    #[test]
    fn test_dynamic_slice_array_ir_interpretation_data_dependent_size() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let mask = builder.add_input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)])).into());
        let values = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)])).into());
        let mask = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::from(ConvertElementTypeOperation::<ArrayType>::new(
                    DataType::I64,
                    false,
                ))),
                Vec::new(),
                vec![mask],
                None,
            )
            .unwrap()[0];
        let count = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::from(ReduceOperation::new(vec![0], ReductionKind::Sum))),
                Vec::new(),
                vec![mask],
                None,
            )
            .unwrap()[0];
        let count_variable = DimensionVariable::new("count", DimensionBounds::new(0, Some(5)).unwrap());
        let count = builder
            .add_instruction(DimensionFromScalarOperation::new(count_variable.clone()), Vec::new(), vec![count], None)
            .unwrap()[0];
        let start = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let output = builder
            .add_instruction(
                DynamicSliceOperation::<ArrayIrType>::from_rank(1),
                Vec::new(),
                vec![values, start, count],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // The count remains ordinary scalar SSA until the checked gateway defines one fresh internal identity. The
        // slice consumes that first-class dimension directly, so staging needs neither a concrete count nor an
        // input-boundary refinement for `count`.
        assert!(program.type_identity_signature().input_identities().is_empty());
        assert!(program.type_identity_signature().internal_identities().contains(&count_variable));
        assert_eq!(
            program.output_types(),
            vec![ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(count_variable)]),
            ))],
        );
        let [_, _, gateway, _] = program.instructions() else {
            panic!("expected convert, reduce, dimension gateway, and dynamic slice instructions");
        };
        assert!(matches!(gateway.operation(), ArrayIrOperation::DimensionFromScalar(_)));

        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![true, false, true, false]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0]).unwrap())]),
        );
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![false, false, false, false]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap())]),
        );
    }

    #[test]
    fn test_dynamic_slice_array_ir_partial_evaluation() {
        let input = ArrayIrValue::Array(Array::vector(vec![10_i32, 20, 30, 40]).unwrap());
        let start = ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap());
        let size = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let expected = ArrayIrValue::Array(Array::vector(vec![20_i32, 30]).unwrap());
        // Known bounds execute their assertion when folded; an unknown array retains the checked operation.
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = DynamicSliceOperation::<ArrayIrType>::from_rank(1),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, start.clone()), (@known, size.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, start), (@known, size)],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_dynamic_slice_array_ir_batching() {
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let input = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 4, vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap()),
            BatchAxis::new(0),
        )
        .unwrap();
        let start = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()));
        let size = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));

        let outputs = DynamicSliceOperation::<ArrayIrType>::from_rank(1)
            .batch(&context, &EmptyRegionDriver, &[input, start.clone(), size.clone()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[0].value(),
            &ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 5.0, 6.0]).unwrap()),
        );

        // A replicated array input takes the fast path: the operation binds unchanged in the parent context and its
        // result stays replicated.
        let replicated =
            ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![0.0_f32, 1.0, 2.0, 3.0]).unwrap()));
        let outputs = DynamicSliceOperation::<ArrayIrType>::from_rank(1)
            .batch(&context, &EmptyRegionDriver, &[replicated, start.clone(), size.clone()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value(), &ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()));

        // The payload's stride count and the input's rank bound the arity independently: a payload built for another
        // rank is rejected rather than indexed.
        let vector =
            ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0_f32; 8]).unwrap()), BatchAxis::new(0))
                .unwrap();
        assert_eq!(
            DynamicSliceOperation::<ArrayIrType>::from_rank(1)
                .batch(&context, &EmptyRegionDriver, &[vector.clone(), start.clone()])
                .unwrap_err(),
            BatchingError::Program(ProgramError::InvalidInputCount { expected: 3, actual: 2 }),
        );
        assert_eq!(
            DynamicSliceOperation::<ArrayIrType>::from_rank(2)
                .batch(
                    &context,
                    &EmptyRegionDriver,
                    &[vector.clone(), start.clone(), start.clone(), size.clone(), size.clone()],
                )
                .unwrap_err(),
            BatchingError::Program(ProgramError::InvalidInputCount { expected: 3, actual: 5 }),
        );

        // First-class starts and sizes must remain replicated because per-item slice geometry would be ragged.
        let mapped_size =
            ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![1_i64, 2]).unwrap()), BatchAxis::new(0)).unwrap();
        assert!(matches!(
            DynamicSliceOperation::<ArrayIrType>::from_rank(1).batch(&context, &EmptyRegionDriver, &[vector, start, mapped_size]),
            Err(BatchingError::Type(error))
                if error == TypeError::invalid("expected dimension type but got array type"),
        ));

        // The transform's extent must describe the complete mapped axis, rather than cropping extra batch items.
        let input =
            ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(3, 4, vec![1_f32; 12]).unwrap()), BatchAxis::new(0))
                .unwrap();
        let start = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let size = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()));
        assert!(matches!(
            DynamicSliceOperation::<ArrayIrType>::from_rank(1).batch(&context, &EmptyRegionDriver, &[input, start, size]),
            Err(BatchingError::MisalignedBatchAxes { message })
                if message == format!(
                    "`{DYNAMIC_SLICE_OPERATION_NAME}` mapped input extent 3 does not match batching extent 2"
                ),
        ));

        // A rectangular result must not silently discard a per-item extent contract.
        let ragged = RaggedAxis::new(
            1,
            ArrayIrValue::Array(Array::vector(vec![1_i64, 3]).unwrap()),
            DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap()),
            vec![0],
        );
        let input =
            ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 4, vec![1_f32; 8]).unwrap()), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![ragged])
                .unwrap();
        let start = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let size = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()));
        assert!(matches!(
            DynamicSliceOperation::<ArrayIrType>::from_rank(1).batch(&context, &EmptyRegionDriver, &[input, start, size]),
            Err(BatchingError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "`{DYNAMIC_SLICE_OPERATION_NAME}` does not support bounded ragged array inputs"
                ),
        ));
    }

    #[test]
    fn test_dynamic_slice_array_ir_differentiation() {
        let dimension = |extent| ArrayIrValue::Dimension(DimensionValue::constant(extent).unwrap());
        // The slice geometry is discrete, but the array input remains linear: JVP applies the same runtime slice to the
        // primal and tangent instead of treating the complete mixed operation as a constant.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)])).into());
        let start = builder.add_constant(dimension(1));
        let size = builder.add_constant(dimension(2));
        let output = builder
            .add_instruction(
                DynamicSliceOperation::<ArrayIrType>::from_rank(1),
                Vec::new(),
                vec![input, start, size],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:dimension<1> = const 1
                    %3:dimension<2> = const 2
                    %4:f64[2] = dynamic_slice [strides=[1], bounds=checked, requires_runtime_assertion=true] %0 %2 %3
                    %5:f64[2] = dynamic_slice [strides=[1], bounds=checked, requires_runtime_assertion=true] %1 %2 %3
                in (%4, %5)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![20.0_f64, 30.0]).unwrap()),
            ]),
        );

        // A direct JVP keeps a dynamic structural zero symbolic and still stages the primal bounds assertion.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(6)).unwrap());
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = trace.input(ArrayType::new(DataType::F64, Shape::new(vec![extent.into()])).into());
        let start = trace.input(DimensionValue::constant(1).unwrap().r#type().into_owned().into());
        let size = trace.input(DimensionValue::constant(1).unwrap().r#type().into_owned().into());
        let inputs = [input, start, size]
            .into_iter()
            .map(|value| DifferentiationDual::new_with_zero_tangent(value).unwrap())
            .collect::<Vec<_>>();
        let outputs = DynamicSliceOperation::<ArrayIrType>::from_rank(1)
            .jvp(&DifferentiationContext::fused(trace.clone()), &EmptyRegionDriver, &inputs)
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].tangent().is_zero());
        assert_eq!(
            outputs[0].tangent().r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new_static(DataType::F64, [1]))
        );
        assert_eq!(trace.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_dynamic_slice_array_ir_transposition() {
        let dimension = |extent| ArrayIrValue::Dimension(DimensionValue::constant(extent).unwrap());
        // The transpose inserts each cotangent at its selected input coordinate.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)])).into());
        let start = builder.add_constant(dimension(1));
        let size = builder.add_constant(dimension(2));
        let output = builder
            .add_instruction(
                DynamicSliceOperation::<ArrayIrType>::from_rank(1),
                Vec::new(),
                vec![input, start, size],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let transpose = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transpose.interpret(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0]).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 10.0, 20.0, 0.0]).unwrap())]),
        );
    }

    #[test]
    fn test_dynamic_slice_array_ir_transposition_runtime_geometry() {
        // Both source axes and both result axes vary, including zero extents. Point updates avoid imposing a
        // static window on either axis, and residual input dimensions restore the exact specialized source shape.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(0, Some(6)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(0, Some(7)).unwrap());
        let height = DimensionVariable::new("height", DimensionBounds::new(0, Some(4)).unwrap());
        let width = DimensionVariable::new("width", DimensionBounds::new(0, Some(4)).unwrap());
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = context.input(ArrayType::new(DataType::F64, Shape::new(vec![rows.into(), columns.into()])).into());
        let height = context.input(DimensionType::from(height).into());
        let width = context.input(DimensionType::from(width).into());
        let zero = context.dimension_constant(0).unwrap();
        let output = input
            .dynamic_slice_with_dimensions(&[zero.clone(), zero], &[height.clone(), width.clone()], &[2, 2])
            .unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        let pullback = linearization.pullback().unwrap();
        let primal = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::matrix(3, 4, vec![1.0_f64; 12]).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
            ])
            .unwrap();
        let mut arguments = vec![ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap())];
        arguments.extend_from_slice(&primal[1..]);
        // Differentiating the pullback again must retain its paired slice map, including possibly empty source
        // bounds. A generic gather transpose would require a positive lower bound on each destination axis.
        let second = pullback.linearize_with_respect_to(&[0]).unwrap();
        let forward_again = second.pullback().unwrap();
        let mut second_outputs = second.primal().interpret(arguments.clone()).unwrap();
        let mut second_arguments =
            vec![ArrayIrValue::Array(Array::matrix(3, 4, (0..12).map(|value| value as f64).collect()).unwrap())];
        second_arguments.extend(second_outputs.split_off(1));
        assert_eq!(
            forward_again.interpret(second_arguments),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 2, vec![0.0_f64, 2.0, 8.0, 10.0]).unwrap(),)])
        );
        assert_eq!(
            pullback.interpret(arguments),
            Ok(vec![ArrayIrValue::Array(
                Array::matrix(3, 4, vec![1.0_f64, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0]).unwrap(),
            )])
        );
        // Batch cotangent seeds while keeping the runtime geometry shared across mapped items.
        let mut batched_arguments = vec![ArrayIrValue::Array(
            Array::from_elements(
                ArrayType::new_static(DataType::F64, [2, 2, 2]),
                &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
            .unwrap(),
        )];
        batched_arguments.extend_from_slice(&primal[1..]);
        let mut axes = vec![BatchAxis::replicated(); batched_arguments.len()];
        axes[0] = BatchAxis::new(0);
        let batched = batch(
            |inputs| pullback.interpret_in_context(&inputs[0].context().clone(), inputs),
            batched_arguments,
            axes,
            vec![BatchAxis::new(0)],
            None,
        )
        .unwrap();
        assert_eq!(
            batched,
            vec![ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new_static(DataType::F64, [2, 3, 4]),
                    &[
                        1.0_f64, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 7.0, 0.0, 8.0, 0.0
                    ],
                )
                .unwrap()
            )]
        );
        let primal = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::matrix(0, 4, Vec::<f64>::new()).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
            ])
            .unwrap();
        let mut arguments = vec![ArrayIrValue::Array(Array::matrix(0, 2, Vec::<f64>::new()).unwrap())];
        arguments.extend_from_slice(&primal[1..]);
        assert_eq!(
            pullback.interpret(arguments),
            Ok(vec![ArrayIrValue::Array(Array::matrix(0, 4, Vec::<f64>::new()).unwrap(),)])
        );
    }

    #[test]
    fn test_dynamic_slice_array_ir_dynamic_slice_axis() {
        // An empty selection of an empty axis stages a program that passes the empty array through.
        let empty = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F64, [0]), &[] as &[f64]).unwrap(),
        );
        let (_, empty_program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.dynamic_slice_axis(0, 0, 0, 1),
            empty.r#type().into_owned(),
        )
        .unwrap();
        assert_eq!(empty_program.interpret(empty.clone()).unwrap(), empty);

        // Host-known start, limit, and stride select columns 1 and 3 of every row while the symbolic row extent is
        // retained, so one staged program serves inputs with 4 and with 5 rows.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(4, Some(6)).unwrap());
        let input_type = ArrayIrType::Array(ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(4)]),
        ));
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.dynamic_slice_axis(1, 1, 4, 2),
            input_type.clone(),
        )
        .unwrap();
        let four_rows = Array::from_elements(
            ArrayType::new_static(DataType::F64, [4, 4]),
            &(0..16).map(|value| value as f64).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(
            program.interpret(ArrayIrValue::Array(four_rows)),
            Ok(ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new_static(DataType::F64, [4, 2]),
                    &[1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
                )
                .unwrap()
            )),
        );
        let five_rows = Array::from_elements(
            ArrayType::new_static(DataType::F64, [5, 4]),
            &(0..20).map(|value| value as f64).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(
            program.interpret(ArrayIrValue::Array(five_rows)),
            Ok(ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new_static(DataType::F64, [5, 2]),
                    &[1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0],
                )
                .unwrap()
            )),
        );

        // Indices neither wrap nor clamp: a zero stride, an inverted window, and a limit beyond the guaranteed extent
        // are rejected before any query is staged.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::F64, [2, 4]).into());
        let window_error =
            Err(TypeError::invalid("`dynamic_slice_axis` requires a positive stride and start no greater than limit")
                .into());
        assert_eq!(input.dynamic_slice_axis(1, 1, 4, 0), window_error);
        assert_eq!(input.dynamic_slice_axis(1, 3, 1, 1), window_error);
        assert_eq!(
            input.dynamic_slice_axis(1, 0, 5, 1),
            Err(TypeError::invalid("`dynamic_slice_axis` limit 5 exceeds the guaranteed extent 4 of axis 1").into()),
        );
        assert!(trace.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_dynamic_slice_array_ir_dynamic_index_axis() {
        // Selecting column 1 of every row drops the axis and retains the symbolic row extent, so one staged program
        // serves inputs with 4 and with 5 rows.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(4, Some(6)).unwrap());
        let input_type = ArrayIrType::Array(ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(4)]),
        ));
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.dynamic_index_axis(1, 1, false),
            input_type,
        )
        .unwrap();
        let four_rows = Array::from_elements(
            ArrayType::new_static(DataType::F64, [4, 4]),
            &(0..16).map(|value| value as f64).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(
            program.interpret(ArrayIrValue::Array(four_rows)),
            Ok(ArrayIrValue::Array(Array::vector(vec![1.0, 5.0, 9.0, 13.0]).unwrap())),
        );
        let five_rows = Array::from_elements(
            ArrayType::new_static(DataType::F64, [5, 4]),
            &(0..20).map(|value| value as f64).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(
            program.interpret(ArrayIrValue::Array(five_rows)),
            Ok(ArrayIrValue::Array(Array::vector(vec![1.0, 5.0, 9.0, 13.0, 17.0]).unwrap())),
        );

        // Keeping the axis retains it with extent one, and an index at `usize::MAX` has no exclusive limit.
        let (_, keep_program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.dynamic_index_axis(-1, 1, true),
            ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 3])),
        )
        .unwrap();
        let input = ArrayIrValue::Array(Array::matrix(2, 3, vec![10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap());
        assert_eq!(
            keep_program.interpret(input),
            Ok(ArrayIrValue::Array(Array::matrix(2, 1, vec![20.0_f64, 50.0]).unwrap())),
        );
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::F64, [2, 3]).into());
        assert_eq!(
            input.dynamic_index_axis(0, usize::MAX, false),
            Err(TypeError::invalid("`dynamic_index_axis` index overflows `usize`").into()),
        );
        assert!(trace.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_dynamic_slice_in_axis() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        // The traced start counts from the end of the sliced axis and clamps so that the window fits, while every
        // other axis is kept in full through a zero start.
        assert_eq!(
            matrix.dynamic_slice_in_axis(&Array::scalar(-2_i32).unwrap(), 2, 1).unwrap(),
            Array::matrix(2, 2, vec![2.0, 3.0, 5.0, 6.0]).unwrap(),
        );
        assert_eq!(
            matrix.dynamic_slice_in_axis(&Array::scalar(5_i32).unwrap(), 2, -1).unwrap(),
            Array::matrix(2, 2, vec![2.0, 3.0, 5.0, 6.0]).unwrap(),
        );
        assert_eq!(
            matrix.dynamic_slice_in_axis(&Array::scalar(-9_i32).unwrap(), 2, 1).unwrap(),
            Array::matrix(2, 2, vec![1.0, 2.0, 4.0, 5.0]).unwrap(),
        );
        assert_eq!(
            matrix.dynamic_slice_in_axis(&Array::scalar(-1_i32).unwrap(), 1, 0).unwrap(),
            Array::matrix(1, 3, vec![4.0, 5.0, 6.0]).unwrap(),
        );

        // Tracing stages the zero starts from the traced start and one ordinary dynamic slice.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(input, start)| input.dynamic_slice_in_axis(&start, 2, 1),
            (ArrayType::new_static(DataType::F32, [2, 3]), ArrayType::scalar(DataType::I32)),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:i32[] .
                let %2:i32[] = zero_like %1
                    %3:f32[2, 2] = dynamic_slice [sizes=[2, 2]] %0 %2 %1
                in (%3)
            "}
            .trim_end(),
        );

        // Every other axis must be static because the window spans it in full.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("rows", DimensionBounds::new(1, Some(4)).unwrap())),
                Dimension::Static(3),
            ]),
        );
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace(
                |(input, start)| input.dynamic_slice_in_axis(&start, 2, 1),
                (dynamic_type.clone(), ArrayType::scalar(DataType::I32)),
            )
            .map(|_| ()),
            Err(TypeError::invalid(format!(
                "`dynamic_slice` along axis 1 requires static extents on the other axes but axis 0 of \
                 `{dynamic_type}` is dynamic",
            ))
            .into()),
        );
    }

    #[test]
    fn test_dynamic_index_in_axis() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        // The traced index counts from the end and clamps into bounds; the selected axis is squeezed unless kept.
        assert_eq!(
            matrix.dynamic_index_in_axis(&Array::scalar(-1_i32).unwrap(), 1, false).unwrap(),
            Array::vector(vec![3.0, 6.0]).unwrap(),
        );
        assert_eq!(
            matrix.dynamic_index_in_axis(&Array::scalar(-1_i32).unwrap(), 1, true).unwrap(),
            Array::matrix(2, 1, vec![3.0, 6.0]).unwrap(),
        );
        assert_eq!(
            matrix.dynamic_index_in_axis(&Array::scalar(7_i32).unwrap(), -2, false).unwrap(),
            Array::vector(vec![4.0, 5.0, 6.0]).unwrap(),
        );
    }

    #[test]
    fn test_dynamic_update_slice() {
        let operation = DynamicUpdateSliceOperation::new();

        // Operation identity.
        assert_eq!(operation.name(), DYNAMIC_UPDATE_SLICE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "dynamic_update_slice");
        assert!(operation.allows_negative_indices());

        // The default negative-index policy is implied by the rendering; only the clamp-only policy is shown.
        let clamping = operation.with_allow_negative_indices(false);
        assert!(!clamping.allows_negative_indices());
        assert_eq!(format!("{clamping}"), "dynamic_update_slice [allow_negative_indices=false]");

        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)]));
        let index_type = ArrayType::scalar(DataType::I32);
        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Array, DynamicUpdateSliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_update = builder.add_input(update_type);
        let program_index_0 = builder.add_input(index_type.clone());
        let program_index_1 = builder.add_input(index_type);
        let program_output = builder
            .add_instruction(
                operation,
                Vec::new(),
                vec![program_input, program_update, program_index_0, program_index_1],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(
                vec![program_output],
                vec![Placeholder, Placeholder, Placeholder, Placeholder],
                Placeholder,
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[1, 2], %2:i32[], %3:i32[] .
                let %4:f64[2, 3] = dynamic_update_slice %0 %1 %2 %3
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dynamic_update_slice_type_inference() {
        let operation = DynamicUpdateSliceOperation::new();
        // Type inference validates the update and index input types and returns the input type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)]));
        let index_type = ArrayType::scalar(DataType::I32);
        let dynamic_update_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(2),
            ]),
        );
        let dynamic_input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(3),
            ]),
        );
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    input_types = [input_type.clone(), update_type.clone(), index_type.clone(), index_type.clone()],
                    output_types = [input_type.clone()],
                },
                {
                    input_types = [input_type.clone()],
                    error = format!(
                        "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` expects an array input and an update input followed \
                         by start index inputs but got 1 inputs"
                    ),
                },
                {
                    input_types = [input_type.clone(), update_type.clone(), index_type.clone()],
                    error = format!(
                        "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` expects one start index per input axis (2) but got 1"
                    ),
                },
                {
                    input_types = [
                        input_type.clone(),
                        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)])),
                        index_type.clone(),
                        index_type.clone(),
                    ],
                    error = format!(
                        "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` input data type `f64` does not match update data \
                         type `f32`"
                    ),
                },
                {
                    input_types = [
                        input_type.clone(),
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])),
                        index_type.clone(),
                        index_type.clone(),
                    ],
                    error = format!("`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` update has rank 1 but input has rank 2"),
                },
                // Dynamic update extents stay a forward rejection: no transform rule accepts them either.
                {
                    input_types = [input_type.clone(), dynamic_update_type, index_type.clone(), index_type.clone()],
                    error = format!(
                        "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` does not support dynamic update axis 0 with size \
                         dynamic; update shapes must be static"
                    ),
                },
                {
                    input_types = [
                        input_type.clone(),
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(4)])),
                        index_type.clone(),
                        index_type.clone(),
                    ],
                    error = format!(
                        "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` update axis 1 has size 4 which exceeds input size 3"
                    ),
                },
                {
                    input_types = [dynamic_input_type, update_type.clone(), index_type.clone(), index_type.clone()],
                    error = format!(
                        "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` update size 1 exceeds the guaranteed minimum extent \
                         0 of dynamic axis 0"
                    ),
                },
                {
                    input_types = [
                        input_type.clone(),
                        update_type.clone(),
                        ArrayType::scalar(DataType::F64),
                        index_type.clone(),
                    ],
                    error = format!(
                        "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` start index 0 must be a scalar integer but has type \
                         `f64[]`"
                    ),
                },
            ],
        );
        assert_eq!(
            input_type.dynamic_update_slice(&update_type, &[index_type.clone(), index_type.clone()]),
            Ok(input_type.clone()),
        );

        // A slice operation cannot own nested regions.
        assert_eq!(
            DynamicUpdateSliceOperation::new()
                .infer_output_types(&[], &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)]),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_dynamic_update_slice_reference_discharge() {
        // Replay preserves the complete slicing payload and its output type. Shared replay and reference rejection
        // are covered by the reference-discharge macro tests.
        let expected = DynamicUpdateSliceOperation::new();
        let operation = ArrayIrOperation::Array(ArrayOperation::DynamicUpdateSlice(expected.clone()));
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = ReferenceDischargeContext::<_, ArrayReferenceDischarge>::new(trace.clone());
        let inputs = [
            ReferenceDischargeValue::Value(trace.input(ArrayType::new_static(DataType::F64, [2, 3]).into())),
            ReferenceDischargeValue::Value(trace.input(ArrayType::new_static(DataType::F64, [1, 2]).into())),
            ReferenceDischargeValue::Value(trace.input(ArrayType::scalar(DataType::I32).into())),
            ReferenceDischargeValue::Value(trace.input(ArrayType::scalar(DataType::I32).into())),
        ];
        let outputs = operation.discharge_references(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        let ReferenceDischargeValue::Value(output) = &outputs[0] else {
            panic!("expected a value carrier but got {}", outputs[0]);
        };
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 3])));
        let builder = trace.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        let ArrayIrOperation::Array(ArrayOperation::DynamicUpdateSlice(staged)) = builder.instructions()[0].operation()
        else {
            panic!("expected a staged dynamic_update_slice");
        };
        assert_eq!(staged, &expected);
    }

    #[test]
    fn test_dynamic_update_slice_interpretation() {
        // Applying output sharding metadata preserves the non-dense layout and the untouched input values.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type =
            ArrayType::new_static(DataType::I32, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![-4])));
        let update_type = ArrayType::new_static(DataType::I32, [1])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let input = Array::from_elements(input_type.clone(), &[1_i32, 2, 3]).unwrap();
        let update = Array::from_elements(update_type, &[9_i32]).unwrap();
        let output = input.dynamic_update_slice(&update, &[Array::scalar(1_i32).unwrap()]).unwrap();
        let expected_type = input_type
            .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        assert_eq!(output, Array::from_elements(expected_type, &[1_i32, 9, 3]).unwrap());
        assert_eq!(input.elements::<i32>(), Ok(vec![1, 2, 3]));

        let operation = DynamicUpdateSliceOperation::new();
        let input_type = ArrayType::new_static(DataType::F64, [2, 3]);
        // Interpretation overwrites the block at the in-band start indices.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let update = Array::matrix(1, 2, vec![8.0, 9.0]).unwrap();
        let output = operation
            .interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[input.clone(), update.clone(), Array::scalar(0_i32).unwrap(), Array::scalar(1_i32).unwrap()],
            )
            .unwrap();
        assert_eq!(*output[0].r#type(), input_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);

        // Out-of-bounds start indices clamp per StableHLO semantics: the effective start index along axis `d` is
        // `clamp(0, start_indices[d], input_dimension[d] - update_dimension[d])`.
        let clamped = operation
            .interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[input.clone(), update.clone(), Array::scalar(5_i32).unwrap(), Array::scalar(-3_i32).unwrap()],
            )
            .unwrap();
        assert_eq!(clamped[0].to_f64s(), vec![1.0, 2.0, 3.0, 8.0, 9.0, 6.0]);
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
    }

    #[test]
    fn test_dynamic_update_slice_partial_evaluation() {
        // Partial evaluation folds known updates and residualizes an unknown input with captured start indices.
        let input = Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let update = Array::vector(vec![8.0, 9.0]).unwrap();
        let start = Array::scalar(1_i32).unwrap();
        let expected = Array::vector(vec![0.0, 8.0, 9.0, 3.0]).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = DynamicUpdateSliceOperation::new(),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, update.clone()), (@known, start.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, update.clone()),
                        (@known, start.clone()),
                    ],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_dynamic_update_slice_batching() {
        let start = Array::scalar(1_i32).unwrap();
        let update = Array::vector(vec![8.0, 9.0]).unwrap();
        // Replicated starts align the input and update on one mapped axis.
        check_operation_batching!(
            @exact,
            operation = DynamicUpdateSliceOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::matrix(
                        2,
                        4,
                        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    ).unwrap()),
                    (@replicated, update),
                    (@replicated, start),
                ],
                outputs = [(@mapped(axis = 0), Array::matrix(
                    2,
                    4,
                    vec![0.0, 8.0, 9.0, 3.0, 4.0, 8.0, 9.0, 7.0],
                ).unwrap())],
            }],
        );

        // Bounded ragged inputs are rejected before any alignment, and a missing update is rejected right after.
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        let update = ArrayBatch::replicated(Array::vector(vec![8.0, 9.0]).unwrap());
        assert!(matches!(
            DynamicUpdateSliceOperation::new().batch(
                &context,
                &EmptyRegionDriver,
                &[ragged_batch(), update.clone(), ArrayBatch::replicated(Array::scalar(1_i32).unwrap())],
            ),
            Err(BatchingError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` does not support bounded ragged array inputs"
                ),
        ));
        let input = ArrayBatch::replicated(Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap());
        assert_eq!(
            DynamicUpdateSliceOperation::new().batch(&context, &EmptyRegionDriver, &[input]).unwrap_err(),
            BatchingError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
    }

    #[test]
    fn test_dynamic_update_slice_batching_scalar_shortcut() {
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        let scalars = ArrayBatch::new(Array::vector(vec![1.0, 2.0]).unwrap(), BatchAxis::new(0)).unwrap();
        let updates = ArrayBatch::new(Array::vector(vec![7.0, 8.0]).unwrap(), BatchAxis::new(0)).unwrap();
        let outputs = DynamicUpdateSliceOperation::new()
            .batch(&context, &EmptyRegionDriver, &[scalars.clone(), updates.clone()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs, vec![updates]);

        // Missing starts are invalid for per-item vectors even when both arrays have a mapped axis.
        let vectors =
            ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), BatchAxis::new(0))
                .unwrap();
        let updates = ArrayBatch::new(Array::matrix(2, 1, vec![7.0, 8.0]).unwrap(), BatchAxis::new(0)).unwrap();
        assert_eq!(
            DynamicUpdateSliceOperation::new()
                .batch(&context, &EmptyRegionDriver, &[vectors, updates])
                .unwrap_err(),
            BatchingError::Type(TypeError::invalid(format!(
                "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` expects one start index per input axis (1) but got 0"
            ))),
        );

        // Actual scalar replacement still validates the input/update element types before returning the update.
        let integer_updates = ArrayBatch::new(Array::vector(vec![7_i32, 8]).unwrap(), BatchAxis::new(0)).unwrap();
        assert_eq!(
            DynamicUpdateSliceOperation::new()
                .batch(&context, &EmptyRegionDriver, &[scalars, integer_updates])
                .unwrap_err(),
            BatchingError::Type(TypeError::invalid(format!(
                "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` input data type `f64` does not match update data type `i32`"
            ))),
        );
    }

    #[test]
    fn test_dynamic_update_slice_batching_expands_batch_varying_indices() {
        // A scalar update still has to be repeated when only the overwritten scalar input is mapped.
        let outputs = DynamicUpdateSliceOperation::new()
            .batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[
                    ArrayBatch::new(Array::vector(vec![1.0, 2.0]).unwrap(), BatchAxis::new(0)).unwrap(),
                    ArrayBatch::replicated(Array::scalar(9.0f64).unwrap()),
                ],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::vector(vec![9.0, 9.0]).unwrap());

        // A batched update with batch-varying start indices over a replicated input expands per item: item 0 writes
        // `[9, 9]` at offset 0 and item 1 writes `[8, 8]` at offset 2 of the shared input.
        let uniform_input = ArrayBatch::replicated(Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap());
        let update =
            ArrayBatch::new(Array::matrix(2, 2, vec![9.0, 9.0, 8.0, 8.0]).unwrap(), BatchAxis::new(0)).unwrap();
        let outputs = DynamicUpdateSliceOperation::new()
            .batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[uniform_input, update, batch_varying_indices(vec![0, 2])],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().shape().dimensions(), &[Dimension::Static(2), Dimension::Static(4)]);
        assert_eq!(outputs[0].value().to_f64s(), vec![9.0, 9.0, 2.0, 3.0, 0.0, 1.0, 8.0, 8.0]);

        // A batched input with a replicated update writes the same block at each batch item's own offset.
        let input = ArrayBatch::new(
            Array::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let uniform_update = ArrayBatch::replicated(Array::vector(vec![9.0, 9.0]).unwrap());
        let outputs = DynamicUpdateSliceOperation::new()
            .batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[input, uniform_update, batch_varying_indices(vec![1, 0])],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0, 9.0, 9.0, 3.0, 9.0, 9.0, 6.0, 7.0]);
    }

    #[test]
    fn test_dynamic_update_slice_differentiation() {
        // Composing JVP and transposition must retain the captured start index: the input gradient is the output
        // cotangent with the update window zeroed, while the update gradient is that window of the cotangent.
        let (value, (input_gradient, update_gradient)) = differentiate_at((
            Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            Array::vector(vec![7.0, 8.0]).unwrap(),
        ))
        .value_and_gradient(|(x, update)| {
            let start = index_constant(&x, 1);
            x.dynamic_update_slice(&update, &[start]).unwrap().reduce(&[0], ReductionKind::Sum)
        })
        .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 20.0, epsilon = 1e-9);
        assert_eq!(input_gradient.to_f64s(), vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(update_gradient.to_f64s(), vec![1.0, 1.0]);

        // Finite differences of `sum(dynamic_update_slice(x, u)²)` agree with the reverse-mode rule for the input
        // (with the update captured) and for the update (with the input captured), at an in-bounds start and at a
        // start that clamps to the last valid origin. The integer start is a fixed constant, never a perturbed input.
        check_gradient!(
            |input, update| {
                let start = index_constant(&input, 1);
                let updated = input.dynamic_update_slice(&update, &[start])?;
                Ok((updated.clone() * updated).reduce(&[0], ReductionKind::Sum))
            },
            at = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            with = Array::vector(vec![7.0, 8.0]).unwrap(),
            step = 1e-3,
            tolerance = 1e-6,
        );
        check_gradient!(
            |update, input| {
                let start = index_constant(&input, 1);
                let updated = input.dynamic_update_slice(&update, &[start])?;
                Ok((updated.clone() * updated).reduce(&[0], ReductionKind::Sum))
            },
            at = Array::vector(vec![7.0, 8.0]).unwrap(),
            with = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            step = 1e-3,
            tolerance = 1e-6,
        );
        check_gradient!(
            |input, update| {
                let start = index_constant(&input, 9);
                let updated = input.dynamic_update_slice(&update, &[start])?;
                Ok((updated.clone() * updated).reduce(&[0], ReductionKind::Sum))
            },
            at = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            with = Array::vector(vec![7.0, 8.0]).unwrap(),
            step = 1e-3,
            tolerance = 1e-6,
        );

        // A negative start counts from the end in the forward update and in both transpose branches: `-1` names
        // origin 3, which the two-element update clamps to 2, so the input gradient zeroes the last two elements.
        let (value, (input_gradient, update_gradient)) = differentiate_at((
            Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            Array::vector(vec![7.0, 8.0]).unwrap(),
        ))
        .value_and_gradient(|(x, update)| {
            let start = index_constant(&x, -1);
            x.dynamic_update_slice(&update, &[start]).unwrap().reduce(&[0], ReductionKind::Sum)
        })
        .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 18.0, epsilon = 1e-9);
        assert_eq!(input_gradient.to_f64s(), vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(update_gradient.to_f64s(), vec![1.0, 1.0]);
        check_gradient!(
            |input, update| {
                let start = index_constant(&input, -1);
                let updated = input.dynamic_update_slice(&update, &[start])?;
                Ok((updated.clone() * updated).reduce(&[0], ReductionKind::Sum))
            },
            at = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            with = Array::vector(vec![7.0, 8.0]).unwrap(),
            step = 1e-3,
            tolerance = 1e-6,
        );
        check_gradient!(
            |update, input| {
                let start = index_constant(&input, 9);
                let updated = input.dynamic_update_slice(&update, &[start])?;
                Ok((updated.clone() * updated).reduce(&[0], ReductionKind::Sum))
            },
            at = Array::vector(vec![7.0, 8.0]).unwrap(),
            with = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            step = 1e-3,
            tolerance = 1e-6,
        );

        // A shared input contributes once per mapped update outside that item's clamped overwrite window.
        let (_, (input_gradient, update_gradient)) = differentiate_at((
            Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap(),
            Array::matrix(2, 2, vec![7.0_f32, 8.0, 9.0, 10.0]).unwrap(),
        ))
        .value_and_gradient(|(input, updates)| {
            let starts = input.context().constant(Array::vector(vec![1_i32, 3]).unwrap()).unwrap();
            batch(
                |(input, updates, start)| input.dynamic_update_slice(&updates, &[start]),
                (input, updates, starts),
                (BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            )
            .unwrap()
            .reduce(&[0, 1], ReductionKind::Sum)
        })
        .unwrap();
        assert_eq!(input_gradient, Array::vector(vec![2.0_f32, 1.0, 0.0, 1.0]).unwrap());
        assert_eq!(update_gradient, Array::matrix(2, 2, vec![1.0_f32; 4]).unwrap());
    }

    #[test]
    fn test_dynamic_update_slice_differentiation_array_ir() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let update = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let start = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::DynamicUpdateSlice(DynamicUpdateSliceOperation::new())),
                Vec::new(),
                vec![input, update, start],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        // Both array tangents are live, so the only residual is the scalar start: the output cotangent itself supplies
        // the base geometry to the transpose region, and the static update shape needs no retained extent.
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:f64[2], %2:i32[] .
                let %3:f64[extent] = linear_call [residual_count=1] %2 %0 %1 [
                    forward={
                        lambda %0:i32[], %1:f64[extent], %2:f64[2] .
                        let %3:f64[extent] = dynamic_update_slice %1 %2 %0
                        in (%3)
                    },
                    transpose={
                        lambda %0:i32[], %1:f64[extent] .
                        let %2:f64[2] = zero [type=f64[2]]
                            %3:f64[extent] = dynamic_update_slice %1 %2 %0
                            %4:f64[2] = dynamic_slice [sizes=[2]] %1 %0
                        in (%3, %4)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![9.0_f64, 8.0]).unwrap()),
                ArrayIrValue::Array(Array::scalar(1_i32).unwrap()),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![1.0_f64, 9.0, 8.0, 4.0]).unwrap()));
        let residuals = primal_outputs.split_off(1);

        let mut tangent_inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![5.0_f64, 6.0]).unwrap()),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 5.0, 6.0, 40.0]).unwrap())]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 0.0, 0.0, 4.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_dynamic_update_slice_transposition() {
        // Update a [1, 2] block at start (0, 1) of a [2, 3] input: the input and update are linear and the scalar start
        // indices are the known inputs. The output and its cotangent have shape [2, 3].
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)]));
        let cotangent = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        check_operation_transposition!(
            @exact,
            operation = DynamicUpdateSliceOperation::new(),
            cases = [{
                inputs = [
                    (@linear(type = input_type)),
                    (@linear(type = update_type)),
                    (@known, Array::scalar(0_i32).unwrap()),
                    (@known, Array::scalar(1_i32).unwrap()),
                ],
                output_cotangents = [cotangent],
                input_cotangents = [
                    Array::matrix(2, 3, vec![1.0, 0.0, 0.0, 4.0, 5.0, 6.0]).unwrap(),
                    Array::matrix(1, 2, vec![2.0, 3.0]).unwrap(),
                ],
            }],
        );

        // Dynamic update-slice restores the layout-bearing update cotangent after its dynamic slice.
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![4.into()])).with_memory(Memory::Host { pinned: true });
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        let start = Array::from_elements::<i32>(
            ArrayType::scalar(DataType::I32).with_memory(Memory::Host { pinned: true }),
            &[1],
        )
        .unwrap();
        check_operation_transposition!(
            @exact,
            operation = DynamicUpdateSliceOperation::new(),
            cases = [{
                inputs = [
                    (@linear(type = input_type.clone())),
                    (@linear(type = update_type.clone())),
                    (@known, start),
                ],
                output_cotangents = [Array::from_elements::<f64>(input_type.clone(), &[1.0, 2.0, 3.0, 4.0]).unwrap()],
                input_cotangents = [
                    Array::from_elements::<f64>(input_type, &[1.0, 0.0, 0.0, 4.0]).unwrap(),
                    Array::from_elements::<f64>(update_type, &[2.0, 3.0]).unwrap(),
                ],
            }],
        );
    }

    #[test]
    fn test_dynamic_update_slice_transposition_zero_cotangent() {
        // A structural-zero output cotangent contributes nothing, and when neither array cotangent is needed the rule
        // returns before extracting the update shape or reading the start indices.
        let operation = DynamicUpdateSliceOperation::new();
        let input_type = ArrayType::new_static(DataType::F64, [4]);
        let update_type = ArrayType::new_static(DataType::F64, [2]);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut transpose = TranspositionContext::new(context.clone());
        let start = context.lift(Array::scalar(1_i32).unwrap()).unwrap();
        let inputs = [
            PartialValue::Unknown(input_type.clone()),
            PartialValue::Unknown(update_type.clone()),
            PartialValue::Known(start),
        ];
        let accumulators = transpose.cotangent_accumulators(&inputs, &[]).unwrap();
        let zero_outputs = [MaybeZero::Zero(input_type.cotangent().unwrap())];
        operation
            .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &zero_outputs, &accumulators)
            .unwrap();
        let cotangents = transpose.take_cotangents(&accumulators).unwrap();
        assert_eq!(cotangents.len(), 3);
        assert!(cotangents.iter().all(MaybeZero::is_zero));
        assert!(context.builder().borrow().instructions().is_empty());
        let outputs = [MaybeZero::Value(context.input(input_type.cotangent().unwrap()))];
        // Unknown starts would fail the needed path, but no contribution needs to read them here.
        let unneeded_inputs = [
            PartialValue::Unknown(input_type.clone()),
            PartialValue::Unknown(update_type.clone()),
            PartialValue::Unknown(ArrayType::scalar(DataType::I32)),
        ];
        let unneeded = transpose.cotangent_accumulators(&unneeded_inputs, &[false, false, false]).unwrap();
        operation
            .transpose(&mut transpose, &EmptyRegionDriver, &unneeded_inputs, &outputs, &unneeded)
            .unwrap();
        assert!(context.builder().borrow().instructions().is_empty());

        // A requested contribution requires known indices and statically sized update windows.
        let needed = transpose.cotangent_accumulators(&unneeded_inputs, &[true, true, false]).unwrap();
        assert!(matches!(
            operation.transpose(&mut transpose, &EmptyRegionDriver, &unneeded_inputs, &outputs, &needed),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == format!("`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` transpose requires known start indices"),
        ));
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(4)).unwrap());
        let dynamic_inputs = [
            PartialValue::Unknown(input_type.clone()),
            PartialValue::Unknown(ArrayType::new(DataType::F64, Shape::new(vec![extent.into()]))),
            inputs[2].clone(),
        ];
        let needed = transpose.cotangent_accumulators(&dynamic_inputs, &[true, true, false]).unwrap();
        assert_eq!(
            operation.transpose(&mut transpose, &EmptyRegionDriver, &dynamic_inputs, &outputs, &needed).unwrap_err(),
            DifferentiationError::Program(TypeError::invalid(format!(
                "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` transpose requires a static update shape but axis 0 has size extent"
            )).into()),
        );

        // Each contribution is staged only when its accumulator is needed: the input cotangent zeroes the update
        // window at the known start and the update cotangent dynamically slices it.
        let input_only = transpose.cotangent_accumulators(&inputs, &[true, false, false]).unwrap();
        operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &input_only).unwrap();
        let builder = context.builder();
        assert_eq!(
            builder
                .borrow()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().to_string())
                .collect::<Vec<_>>(),
            vec!["zero [type=f64[2]]", "dynamic_update_slice"],
        );
        let cotangents = transpose.take_cotangents(&input_only).unwrap();
        assert!(!cotangents[0].is_zero());
        assert!(cotangents[1].is_zero());
        let update_only = transpose.cotangent_accumulators(&inputs, &[false, true, false]).unwrap();
        operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &update_only).unwrap();
        assert_eq!(
            builder
                .borrow()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().to_string())
                .collect::<Vec<_>>(),
            vec!["zero [type=f64[2]]", "dynamic_update_slice", "dynamic_slice [sizes=[2]]"],
        );
        let cotangents = transpose.take_cotangents(&update_only).unwrap();
        assert!(cotangents[0].is_zero());
        assert!(!cotangents[1].is_zero());

        // Arity is validated before any cotangent is inspected.
        assert_eq!(
            operation
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs[..1], &outputs, &accumulators)
                .unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &[], &accumulators).unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            operation
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &accumulators[..2])
                .unwrap_err(),
            DifferentiationError::InvalidAccumulatorCount { expected: 3, actual: 2 },
        );
    }

    #[test]
    fn test_array_type_dynamic_update_slice() {
        let host_input = ArrayType::new_static(DataType::F32, [4]).with_memory(Memory::Host { pinned: true });
        let host_update = ArrayType::new_static(DataType::F32, [2]).with_memory(Memory::Host { pinned: true });
        let host_index = ArrayType::scalar(DataType::I32).with_memory(Memory::Host { pinned: true });
        assert_eq!(
            host_input.dynamic_update_slice(&host_update, std::slice::from_ref(&host_index)).unwrap().memory(),
            Memory::Host { pinned: true }
        );
        // The update and every start index must share the input's memory space.
        assert_eq!(
            host_input.dynamic_update_slice(&ArrayType::new_static(DataType::F32, [2]), &[host_index]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` input and update must share one memory space but reside in \
                 `Host[Pinned]` and `Device`"
            ))))
        );
        assert_eq!(
            host_input.dynamic_update_slice(&host_update, &[ArrayType::scalar(DataType::I32)]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` input and start indices must share one memory space but \
                 start index 0 resides in `Device` and the input resides in `Host[Pinned]`"
            ))))
        );

        // Resizing keeps explicit placement only when the resulting dimension remains evenly divisible.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let input = ArrayType::new_static(DataType::F32, [4, 4]).with_sharding(sharding.clone()).unwrap();
        let starts = [ArrayType::scalar(DataType::I32), ArrayType::scalar(DataType::I32)];
        let matching = ArrayType::new_static(DataType::F32, [2, 4]).with_sharding(sharding.clone()).unwrap();
        let conflicting =
            ArrayType::new_static(DataType::F32, [2, 4]).with_sharding(Sharding::replicated(mesh, 2)).unwrap();
        assert_eq!(input.dynamic_update_slice(&matching, &starts).unwrap().sharding(), Some(&sharding));
        assert_eq!(
            input.dynamic_update_slice(&conflicting, &starts),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` input and update must be sharded identically, but got `{}` \
                 and `{}`",
                input.sharding().unwrap(),
                conflicting.sharding().unwrap(),
            ))))
        );
    }

    #[test]
    fn test_dynamic_update_slice_in_axis() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        // The traced start counts from the end of the updated axis and clamps so that the update fits, while the
        // update's extent on the other axis need not match the input's.
        let update = Array::matrix(1, 2, vec![8.0, 9.0]).unwrap();
        assert_eq!(
            matrix.dynamic_update_slice_in_axis(&update, &Array::scalar(-1_i32).unwrap(), 1).unwrap(),
            Array::matrix(2, 3, vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        assert_eq!(
            matrix.dynamic_update_slice_in_axis(&update, &Array::scalar(-9_i32).unwrap(), -1).unwrap(),
            Array::matrix(2, 3, vec![8.0, 9.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        assert_eq!(
            matrix.dynamic_update_slice_in_axis(&update, &Array::scalar(-1_i32).unwrap(), 0).unwrap(),
            Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 8.0, 9.0, 6.0]).unwrap(),
        );
    }

    #[test]
    fn test_dynamic_update_index_in_axis() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        // A rank-deficient update gains the indexed axis with extent one; a full-rank update is written as is.
        assert_eq!(
            matrix
                .dynamic_update_index_in_axis(
                    &Array::vector(vec![7.0, 8.0]).unwrap(),
                    &Array::scalar(-1_i32).unwrap(),
                    1
                )
                .unwrap(),
            Array::matrix(2, 3, vec![1.0, 2.0, 7.0, 4.0, 5.0, 8.0]).unwrap(),
        );
        assert_eq!(
            matrix
                .dynamic_update_index_in_axis(
                    &Array::matrix(2, 1, vec![7.0, 8.0]).unwrap(),
                    &Array::scalar(9_i32).unwrap(),
                    -1,
                )
                .unwrap(),
            Array::matrix(2, 3, vec![1.0, 2.0, 7.0, 4.0, 5.0, 8.0]).unwrap(),
        );
        assert_eq!(
            matrix
                .dynamic_update_index_in_axis(
                    &Array::vector(vec![7.0, 8.0, 9.0]).unwrap(),
                    &Array::scalar(-2_i32).unwrap(),
                    0
                )
                .unwrap(),
            Array::matrix(2, 3, vec![7.0, 8.0, 9.0, 4.0, 5.0, 6.0]).unwrap(),
        );
    }

    #[test]
    fn test_array_dynamic_update_slice() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        // Dynamic start indices clamp so the block stays in bounds.
        let start = [Array::scalar(4i64).unwrap()];
        assert_eq!(
            vector.dynamic_update_slice(&Array::vector(vec![10.0, 20.0]).unwrap(), &start).unwrap(),
            Array::vector(vec![1.0, 2.0, 3.0, 10.0, 20.0]).unwrap(),
        );

        // Updating also validates the complete start vector before accessing the payload.
        assert_eq!(
            Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap()
                .dynamic_update_slice(&Array::matrix(1, 2, vec![8.0, 9.0]).unwrap(), &[Array::scalar(0_i32).unwrap()]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` expects one start index per input axis (2) but got 1"
            )))),
        );
    }

    #[test]
    fn test_array_ir_value_dynamic_update_slice() {
        let input = ArrayIrValue::Array(Array::vector(vec![10i32, 20, 30]).unwrap());
        let update = ArrayIrValue::Array(Array::vector(vec![40i32]).unwrap());
        assert_eq!(
            input.dynamic_update_slice(&update, &[ArrayIrValue::Array(Array::scalar(1i32).unwrap())]),
            Ok(ArrayIrValue::Array(Array::vector(vec![10i32, 40, 30]).unwrap()))
        );
    }

    #[test]
    fn test_clamped_start_indices() {
        let input_shape = StaticShape::new(vec![4, 3]);
        // In-bounds starts pass through, and starts past the last valid origin clamp to `extent - size`, so the block
        // always stays in bounds.
        assert_eq!(
            Array::clamped_start_indices(
                &[Array::scalar(1_i32).unwrap(), Array::scalar(0_i32).unwrap()],
                &input_shape,
                &[2, 3],
                true,
            ),
            vec![1, 0],
        );
        assert_eq!(
            Array::clamped_start_indices(
                &[Array::scalar(3_i32).unwrap(), Array::scalar(9_i32).unwrap()],
                &input_shape,
                &[2, 1],
                true,
            ),
            vec![2, 2],
        );
        // A negative signed start counts from the end of its axis once and then clamps: `-1` on the extent-4 axis names
        // origin 3, which the size-2 window clamps to 2, while `-7` stays negative after one wrap and clamps to zero.
        assert_eq!(
            Array::clamped_start_indices(
                &[Array::scalar(-1_i32).unwrap(), Array::scalar(-7_i32).unwrap()],
                &input_shape,
                &[2, 1],
                true,
            ),
            vec![2, 0],
        );
        assert_eq!(
            Array::clamped_start_indices(
                &[Array::scalar(-1_i32).unwrap(), Array::scalar(-1_i32).unwrap()],
                &input_shape,
                &[1, 3],
                true,
            ),
            vec![3, 0],
        );
        // Without the policy, negative starts are out of bounds and clamp to zero directly.
        assert_eq!(
            Array::clamped_start_indices(
                &[Array::scalar(-1_i32).unwrap(), Array::scalar(-7_i32).unwrap()],
                &input_shape,
                &[2, 1],
                false,
            ),
            vec![0, 0],
        );
        // Signed and unsigned extremes are decoded exactly: the unsigned maximum clamps without any signed
        // reinterpretation, and the signed minimum stays negative after one wrap and clamps to zero under either
        // policy.
        for allow_negative_indices in [true, false] {
            assert_eq!(
                Array::clamped_start_indices(
                    &[Array::scalar(u64::MAX).unwrap(), Array::scalar(i64::MIN).unwrap()],
                    &input_shape,
                    &[1, 1],
                    allow_negative_indices,
                ),
                vec![3, 0],
            );
        }
        // A full-extent window always starts at zero, and a rank-0 slice has no starts.
        assert_eq!(
            Array::clamped_start_indices(
                &[Array::scalar(3_i32).unwrap(), Array::scalar(-3_i32).unwrap()],
                &input_shape,
                &[4, 3],
                true,
            ),
            vec![0, 0],
        );
        assert_eq!(Array::clamped_start_indices(&[], &StaticShape::scalar(), &[], true), Vec::<usize>::new());
    }

    #[test]
    fn test_validate_start_index_types() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("n", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [4]);
        let index = ArrayType::scalar(DataType::I32);

        // Scalar integers of one type in the input's memory space pass, including an empty index list.
        assert_eq!(validate_start_index_types(DYNAMIC_SLICE_OPERATION_NAME, &input, &[]), Ok(()));
        assert_eq!(
            validate_start_index_types(DYNAMIC_SLICE_OPERATION_NAME, &input, &[index.clone(), index.clone()]),
            Ok(()),
        );

        // Each index must be a rank-0 integer in the input's memory space; the caller's operation name is reported.
        assert_eq!(
            validate_start_index_types(
                DYNAMIC_SLICE_OPERATION_NAME,
                &input,
                &[ArrayType::new_static(DataType::I32, [1])]
            ),
            Err(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` start index 0 must be a scalar integer but has type `i32[1]`"
            ))
            .into()),
        );
        assert_eq!(
            validate_start_index_types(
                DYNAMIC_UPDATE_SLICE_OPERATION_NAME,
                &input,
                &[ArrayType::scalar(DataType::F32)]
            ),
            Err(TypeError::invalid(format!(
                "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` start index 0 must be a scalar integer but has type `f32[]`"
            ))
            .into()),
        );
        assert_eq!(
            validate_start_index_types(
                DYNAMIC_SLICE_OPERATION_NAME,
                &input,
                &[index.clone().with_memory(Memory::Host { pinned: true })],
            ),
            Err(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` input and start indices must share one memory space but start \
                 index 0 resides in `Host[Pinned]` and the input resides in `Device`"
            ))
            .into()),
        );

        // Indices are discrete control inputs: they cannot carry reduction state, must use the input's mesh, and must
        // be invariant over the input's reduction-state axes. An invariant index over an unreduced input passes.
        let unreduced_index = index
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_unreduced_axes(["m"]).unwrap())
            .unwrap();
        let reduced_index = index
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_reduced_axes(["m"]).unwrap())
            .unwrap();
        let reduction_state_error = Err(TypeError::invalid(format!(
            "`{DYNAMIC_SLICE_OPERATION_NAME}` start indices must not carry reduction state"
        ))
        .into());
        assert_eq!(
            validate_start_index_types(DYNAMIC_SLICE_OPERATION_NAME, &input, &[unreduced_index]),
            reduction_state_error
        );
        assert_eq!(
            validate_start_index_types(DYNAMIC_SLICE_OPERATION_NAME, &input, &[reduced_index]),
            reduction_state_error
        );
        let sharded_input = input.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let other_index = index.clone().with_sharding(Sharding::replicated(other_mesh, 0)).unwrap();
        assert_eq!(
            validate_start_index_types(DYNAMIC_SLICE_OPERATION_NAME, &sharded_input, &[other_index]),
            Err(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` input and start indices must use the same mesh"
            ))
            .into()),
        );
        let unreduced_input = input
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["m"]).unwrap())
            .unwrap();
        let varying_index = index
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        assert_eq!(
            validate_start_index_types(DYNAMIC_SLICE_OPERATION_NAME, &unreduced_input, &[varying_index]),
            Err(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` start indices must be invariant when the input carries reduction \
                 state"
            ))
            .into()),
        );
        let replicated_index = index.clone().with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        assert_eq!(
            validate_start_index_types(DYNAMIC_SLICE_OPERATION_NAME, &unreduced_input, &[replicated_index]),
            Ok(()),
        );

        // All indices share the first index's integer type.
        assert_eq!(
            validate_start_index_types(
                DYNAMIC_SLICE_OPERATION_NAME,
                &input,
                &[index, ArrayType::scalar(DataType::I64)]
            ),
            Err(TypeError::invalid(format!(
                "`{DYNAMIC_SLICE_OPERATION_NAME}` start indices must share one integer type but index 1 has type \
                 `i64[]` and index 0 has type `i32[]`"
            ))
            .into()),
        );
    }

    #[test]
    fn test_indexed_slice_output_type() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let output = ArrayType::new_static(DataType::F32, [2]);
        let index = ArrayType::scalar(DataType::I32);
        let replicated_index = index.clone().with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let varying_index = index
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let other_index = index.clone().with_sharding(Sharding::replicated(other_mesh, 0)).unwrap();

        // Unsharded and replicated indices leave an unsharded output untouched; only variation over manual axes places
        // it, as a replicated sharding on the indices' mesh carrying that variation.
        assert_eq!(
            indexed_slice_output_type(output.clone(), &[index], DYNAMIC_SLICE_OPERATION_NAME),
            Ok(output.clone())
        );
        assert_eq!(
            indexed_slice_output_type(output.clone(), &[replicated_index.clone()], DYNAMIC_SLICE_OPERATION_NAME),
            Ok(output.clone()),
        );
        let acquired = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
        assert_eq!(
            indexed_slice_output_type(output.clone(), &[varying_index.clone()], DYNAMIC_SLICE_OPERATION_NAME),
            Ok(output.clone().with_sharding(acquired).unwrap()),
        );

        // A sharded output keeps its placement and reduction state and unions in the indices' variation.
        let unreduced = Sharding::replicated(mesh, 1).with_unreduced_axes(["m"]).unwrap();
        let sharded_output = output.clone().with_sharding(unreduced.clone()).unwrap();
        assert_eq!(
            indexed_slice_output_type(
                sharded_output.clone(),
                &[replicated_index.clone()],
                DYNAMIC_SLICE_OPERATION_NAME
            ),
            Ok(sharded_output.clone()),
        );
        assert_eq!(
            indexed_slice_output_type(sharded_output.clone(), &[varying_index.clone()], DYNAMIC_SLICE_OPERATION_NAME),
            Ok(output.clone().with_sharding(unreduced.with_varying_manual_axes(["x"]).unwrap()).unwrap()),
        );

        // Every sharded index must use one mesh, shared with a sharded output, independent of index order and of
        // whether the index contributes variation.
        let same_mesh_error =
            Err(TypeError::invalid(format!("`{DYNAMIC_SLICE_OPERATION_NAME}` start indices must use the same mesh"))
                .into());
        assert_eq!(
            indexed_slice_output_type(sharded_output, &[other_index.clone()], DYNAMIC_SLICE_OPERATION_NAME),
            same_mesh_error,
        );
        assert_eq!(
            indexed_slice_output_type(
                output.clone(),
                &[replicated_index.clone(), other_index.clone()],
                DYNAMIC_SLICE_OPERATION_NAME,
            ),
            same_mesh_error,
        );
        assert_eq!(
            indexed_slice_output_type(
                output.clone(),
                &[other_index.clone(), replicated_index],
                DYNAMIC_SLICE_OPERATION_NAME
            ),
            same_mesh_error,
        );
        assert_eq!(
            indexed_slice_output_type(output, &[other_index, varying_index], DYNAMIC_UPDATE_SLICE_OPERATION_NAME),
            Err(TypeError::invalid(format!(
                "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` start indices must use the same mesh"
            ))
            .into()),
        );
    }

    #[test]
    fn test_update_slice_output_sharding() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [4, 4]);
        let update = ArrayType::new_static(DataType::F32, [2, 4]);

        // Unsharded inputs stay unsharded unless the update varies over manual axes, in which case the output acquires
        // a replicated placement on the update's mesh carrying that variation. Reduction state must agree first.
        assert_eq!(update_slice_output_sharding(&input, &update, UPDATE_SLICE_OPERATION_NAME), Ok(None));
        let replicated_update = update.clone().with_sharding(Sharding::replicated(mesh.clone(), 2)).unwrap();
        assert_eq!(update_slice_output_sharding(&input, &replicated_update, UPDATE_SLICE_OPERATION_NAME), Ok(None));
        let varying = Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["m"]).unwrap();
        let varying_update = update.clone().with_sharding(varying.clone()).unwrap();
        assert_eq!(
            update_slice_output_sharding(&input, &varying_update, UPDATE_SLICE_OPERATION_NAME),
            Ok(Some(varying.clone())),
        );
        let unreduced_input = input
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 2).with_unreduced_axes(["m"]).unwrap())
            .unwrap();
        assert_eq!(
            update_slice_output_sharding(&unreduced_input, &update, DYNAMIC_UPDATE_SLICE_OPERATION_NAME),
            Err(TypeError::invalid(format!(
                "`{DYNAMIC_UPDATE_SLICE_OPERATION_NAME}` input and update must carry identical reduction state"
            ))),
        );

        // A sharded input keeps its sharding for an unsharded update and for an update whose manual-axis variation it
        // already covers; new variation is unioned in. Meshes must agree and explicit placements must not conflict.
        let sharded =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let sharded_input = input.clone().with_sharding(sharded.clone()).unwrap();
        assert_eq!(
            update_slice_output_sharding(&sharded_input, &update, UPDATE_SLICE_OPERATION_NAME),
            Ok(Some(sharded.clone())),
        );
        let sharded_varying = sharded.clone().with_varying_manual_axes(["m"]).unwrap();
        let sharded_varying_input = input.clone().with_sharding(sharded_varying.clone()).unwrap();
        let sharded_varying_update = update.clone().with_sharding(sharded_varying.clone()).unwrap();
        assert_eq!(
            update_slice_output_sharding(&sharded_varying_input, &sharded_varying_update, UPDATE_SLICE_OPERATION_NAME),
            Ok(Some(sharded_varying.clone())),
        );
        assert_eq!(
            update_slice_output_sharding(&sharded_input, &sharded_varying_update, UPDATE_SLICE_OPERATION_NAME),
            Ok(Some(sharded_varying)),
        );
        let other_update = update.clone().with_sharding(Sharding::replicated(other_mesh, 2)).unwrap();
        assert_eq!(
            update_slice_output_sharding(&sharded_input, &other_update, UPDATE_SLICE_OPERATION_NAME),
            Err(TypeError::invalid(format!("`{UPDATE_SLICE_OPERATION_NAME}` input and update must use the same mesh"))),
        );
        let conflicting =
            Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]).unwrap();
        let conflicting_update = update.with_sharding(conflicting.clone()).unwrap();
        assert_eq!(
            update_slice_output_sharding(&sharded_input, &conflicting_update, UPDATE_SLICE_OPERATION_NAME),
            Err(TypeError::invalid(format!(
                "`{UPDATE_SLICE_OPERATION_NAME}` input and update must be sharded identically, but got `{sharded}` \
                 and `{conflicting}`"
            ))),
        );
    }

    #[test]
    fn test_batch_by_item_expansion() {
        // No inputs is an arity error.
        let empty_context = BatchingContext::new(EagerContext::<Array>::new(), 0);
        assert_eq!(
            batch_by_item_expansion(
                &empty_context,
                DYNAMIC_SLICE_OPERATION_NAME,
                &DynamicSliceOperation::new(vec![2]),
                &[],
                0
            )
            .unwrap_err(),
            BatchingError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // An empty batch is synthesized from the operation's inferred per-item output type without interpreting a
        // nonexistent item: an identity window returns the empty aligned input itself, and a smaller window slices
        // that input's geometry instead of constructing a scalar zero (explicit layouts do not survive the alignment).
        let indices = ArrayBatch::new(
            Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, [0]), &[]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let plain = ArrayBatch::replicated(Array::vector(vec![0.0, 1.0, 2.0, 3.0]).unwrap());
        let outputs = batch_by_item_expansion(
            &empty_context,
            DYNAMIC_SLICE_OPERATION_NAME,
            &DynamicSliceOperation::new(vec![4]),
            &[plain, indices.clone()],
            0,
        )
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].unbatched_type(), ArrayType::new_static(DataType::F64, [4]));
        assert_eq!(outputs[0].r#type().static_shape().unwrap().as_slice(), &[0, 4]);
        let layout_type =
            ArrayType::new_static(DataType::F64, [4]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let laid_out = ArrayBatch::replicated(Array::from_elements(layout_type, &[0.0, 1.0, 2.0, 3.0]).unwrap());
        let outputs = batch_by_item_expansion(
            &empty_context,
            DYNAMIC_SLICE_OPERATION_NAME,
            &DynamicSliceOperation::new(vec![2]),
            &[laid_out, indices],
            0,
        )
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].unbatched_type(), ArrayType::new_static(DataType::F64, [2]));
        assert_eq!(outputs[0].r#type().static_shape().unwrap().as_slice(), &[0, 2]);
        assert_eq!(outputs[0].value().to_f64s(), Vec::<f64>::new());

        // A non-empty batch pairs item `i` of every batched input, using replicated inputs whole, and stacks the
        // per-item results along a fresh leading axis.
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        let input = ArrayBatch::new(
            Array::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let outputs = batch_by_item_expansion(
            &context,
            DYNAMIC_SLICE_OPERATION_NAME,
            &DynamicSliceOperation::new(vec![2]),
            &[input, batch_varying_indices(vec![1, 3])],
            2,
        )
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 2, vec![1.0, 2.0, 6.0, 7.0]).unwrap());

        // A singleton scalar item still gains a leading mapped axis.
        let singleton_context = BatchingContext::new(EagerContext::<Array>::new(), 1);
        let outputs = batch_by_item_expansion(
            &singleton_context,
            SLICE_OPERATION_NAME,
            &SliceOperation::new(vec![], vec![]),
            &[ArrayBatch::replicated(Array::scalar(7.0).unwrap())],
            1,
        )
        .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::vector(vec![7.0]).unwrap());

        // Explicitly sharded mapped inputs are replicated once before item extraction and the completed accumulator is
        // resharded once to the context's mapped placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mapped_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let context = BatchingContext::new(trace.clone(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));
        let input = ArrayBatch::new(
            trace.input(ArrayType::new_static(DataType::F64, [2, 4]).with_sharding(mapped_sharding.clone()).unwrap()),
            BatchAxis::new(0),
        )
        .unwrap();
        let indices = ArrayBatch::new(
            trace.input(
                ArrayType::new_static(DataType::I32, [2])
                    .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
                    .unwrap(),
            ),
            BatchAxis::new(0),
        )
        .unwrap();
        let outputs = batch_by_item_expansion(
            &context,
            DYNAMIC_SLICE_OPERATION_NAME,
            &DynamicSliceOperation::new(vec![2]),
            &[input, indices],
            2,
        )
        .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().sharding(), Some(&mapped_sharding));
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].value().atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}], \
                    %1:i32[2][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                let %2:f64[2, 4][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] = reshard \
                    [sharding={mesh<['x'=2:explicit]>, [{}, {}]}] %0
                    %3:i32[2][sharding={mesh<['x'=2:explicit]>, [{}]}] = reshard [sharding={mesh<['x'=2:explicit]>, \
                    [{}]}] %1
                    %4:f64[1, 4][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] = slice [start_indices=[0, 0], \
                    limit_indices=[1, 4]] %2
                    %5:f64[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = reshape [shape=[4]] %4
                    %6:i32[1][sharding={mesh<['x'=2:explicit]>, [{}]}] = slice [start_indices=[0], limit_indices=[1]] %3
                    %7:i32[][sharding={mesh<['x'=2:explicit]>, []}] = reshape [shape=[]] %6
                    %8:f64[2][sharding={mesh<['x'=2:explicit]>, [{}]}] = dynamic_slice [sizes=[2]] %5 %7
                    %9:f64[1, 2][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] = reshape [shape=[1, 2]] %8
                    %10:f64[1, 4][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] = slice [start_indices=[1, 0], \
                    limit_indices=[2, 4]] %2
                    %11:f64[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = reshape [shape=[4]] %10
                    %12:i32[1][sharding={mesh<['x'=2:explicit]>, [{}]}] = slice [start_indices=[1], \
                    limit_indices=[2]] %3
                    %13:i32[][sharding={mesh<['x'=2:explicit]>, []}] = reshape [shape=[]] %12
                    %14:f64[2][sharding={mesh<['x'=2:explicit]>, [{}]}] = dynamic_slice [sizes=[2]] %11 %13
                    %15:f64[1, 2][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] = reshape [shape=[1, 2]] %14
                    %16:f64[2, 2][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] = concatenate [axis=0] %9 %15
                    %17:f64[2, 2][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] = reshard \
                    [sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] %16
                in (%17)
            "}
            .trim(),
        );

        // A static output window does not make item extraction valid: slicing away the mapped axis still needs
        // static extents for the remaining input axes. Reject that case before staging the item slice.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(4)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let dynamic = ArrayBatch::new(
            trace.input(ArrayType::new(DataType::F64, Shape::new(vec![2.into(), Dimension::Dynamic(extent)]))),
            BatchAxis::new(0),
        )
        .unwrap();
        let index = ArrayBatch::replicated(trace.input(ArrayType::scalar(DataType::I32)));
        let context = BatchingContext::new(trace, 2);
        assert_eq!(
            batch_by_item_expansion(
                &context,
                DYNAMIC_SLICE_OPERATION_NAME,
                &DynamicSliceOperation::new(vec![1]),
                &[dynamic, index],
                2,
            )
            .unwrap_err(),
            BatchingError::Program(
                TypeError::invalid(format!(
                    "`{DYNAMIC_SLICE_OPERATION_NAME}` per-item expansion requires static batched input types but got \
                     f64[2, extent]",
                ))
                .into(),
            ),
        );
    }
}
