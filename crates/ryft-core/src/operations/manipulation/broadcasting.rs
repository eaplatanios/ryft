use std::fmt::Display;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch,
    ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType, ArrayTypeRefinements, Dimension, DimensionType,
    DimensionValue, Layout, LinearResiduals, RaggedAxis, Shape, Sharding, ShardingDimension, TiledLayout,
};
use crate::axes::Axis;
use crate::batching::{BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    BroadcastDerivativeAlignment, CotangentAccumulator, DifferentiableOperation, DifferentiableType,
    DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
    TransposableOperation, TranspositionContext, TranspositionDriver, transpose_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation, impl_reference_dischargeable_operation};
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::constants::zero::ZeroOperation;
use crate::operations::constants::zero_like::ZeroLikeOperation;
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::conversions::ConvertElementTypeOperation;
use crate::operations::manipulation::reshaping::{
    DynamicReshapeOperation, ReshapeOperation, lift_output_sharding_for_leading_batch_axis,
};
use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
use crate::operations::math::add::AddOperation;
use crate::operations::math::reduce::{Reduce, ReduceOperation, ReductionKind};
use crate::operations::sharding::ReshardOperation;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation,
};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, Type, TypeError,
    TypeIdentityPosition, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{NestedTracingContext, Tracer, TracingContext};

// TODO(eaplatanios): Review this module.

/// Canonical operation name shared by [`BroadcastOperation`] and [`DynamicBroadcastOperation`].
pub const BROADCAST_OPERATION_NAME: &str = "broadcast";

/// [`Operation`] that performs general N-dimensional broadcasting over the homogeneous array language.
///
/// This is the member-family broadcast primitive: complete output geometry is carried by the [`ArrayType`] metadata
/// stored in the operation payload, so the operation has exactly one operand and no explicit extent edges. It and
/// [`ReshapeOperation`] form the homogeneous baseline that [`ProjectedContext`](crate::contexts::ProjectedContext)
/// serves, which is why transform rules for mixed operations can delegate to them once operand geometry is resolved.
/// Refer to the documentation of [`Broadcast`] for the underlying resolved-geometry contract.
///
/// Programs that need first-class dynamic extents stage [`DynamicBroadcastOperation`] instead, which takes one
/// explicit first-class dimension operand per output axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BroadcastOperation {
    /// Output [`ArrayType`].
    output_type: ArrayType,

    /// Vector that contains, for each input axis `i`, the output axis that it maps to.
    output_axes: Vec<usize>,
}

impl BroadcastOperation {
    /// Creates a new [`BroadcastOperation`] with the supplied output type and output axes.
    #[inline]
    pub fn new(output_type: ArrayType, output_axes: Vec<usize>) -> Self {
        Self { output_type, output_axes }
    }

    /// Returns the output [`ArrayType`] of this [`BroadcastOperation`].
    #[inline]
    pub fn output_type(&self) -> &ArrayType {
        &self.output_type
    }

    /// Returns the output axes of this [`BroadcastOperation`]. The resulting slice contains, for each input axis,
    /// the output axis that it maps to.
    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        self.output_axes.as_slice()
    }
}

impl Display for BroadcastOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for BroadcastOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        BROADCAST_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        match input_types[0].broadcast(self.output_type.clone(), self.output_axes.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self { output_type: self.output_type.rename_identities(renaming)?, output_axes: self.output_axes.clone() })
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("output_type", &self.output_type)?;
            operation.field("output_axes", format_args!("{:?}", self.output_axes))
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Broadcast>> InterpretableOperation<C> for BroadcastOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // In a validated homogeneous broadcast, every dynamic output axis maps the identical input dimension.
        // Replay may refine that input to a concrete extent (including one), so recover that extent without
        // inventing a replication count for an unmapped axis or changing a still-symbolic input.
        check_count!("input", inputs, 1, ProgramError);
        let input_type = inputs[0].r#type();
        let mut dimensions = self.output_type.shape().dimensions().to_vec();
        for (input_axis, &output_axis) in self.output_axes().iter().enumerate() {
            if let (Some(Dimension::Static(extent)), Some(dimension @ Dimension::Dynamic(_))) =
                (input_type.shape().dimensions().get(input_axis), dimensions.get_mut(output_axis))
            {
                *dimension = Dimension::Static(*extent);
            }
        }
        let output_type = self.output_type.clone().with_shape(Shape::new(dimensions));
        let mut refinements = ArrayTypeRefinements::default();
        ArrayTypeRefinements::visit_dynamic_to_static_refinements(
            &self.output_type,
            &output_type,
            |variable, extent| refinements.bind(variable, extent),
        )?;
        Ok(vec![inputs[0].broadcast(output_type, self.output_axes())?])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<BroadcastOperation>>> PartiallyEvaluatableOperation<C>
    for BroadcastOperation
{
}

impl<C: Context<Type = ArrayType, Value: Broadcast>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for BroadcastOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        _context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        self.infer_output_types(&[inputs[0].unbatched_type()], &[])?;
        match inputs[0].batch_axis_position() {
            None => {
                // A replicated input has no mapped axis to lift, so the original broadcast remains replicated.
                let mut dimensions = self.output_type().shape().dimensions().to_vec();
                let ragged_axes = inputs[0]
                    .ragged_axes()
                    .iter()
                    .map(|ragged_axis| {
                        let axis = self.output_axes()[ragged_axis.axis()];
                        dimensions[axis] = inputs[0].value().r#type().dimension(ragged_axis.axis());
                        RaggedAxis::new(
                            axis,
                            ragged_axis.extents().clone(),
                            ragged_axis.dimension().clone(),
                            ragged_axis.extent_axes().iter().map(|axis| self.output_axes()[*axis]).collect(),
                        )
                    })
                    .collect::<Vec<_>>();
                let output_type = self.output_type().clone().with_shape(Shape::new(dimensions));
                let output_value = inputs[0].value().broadcast(output_type, self.output_axes())?;
                Ok(vec![ArrayBatch::replicated(output_value).with_ragged_axes(ragged_axes)?].into())
            }
            Some(batch_axis) => {
                // Insert the mapped axis at the same physical output position and shift every existing broadcast-axis
                // mapping at or after that position around it.
                let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
                let mut output_type =
                    self.output_type().with_inserted_dimension(batch_axis, Dimension::Static(axis_size))?;
                let mut output_axes = self
                    .output_axes()
                    .iter()
                    .map(|&output_axis| if output_axis >= batch_axis { output_axis + 1 } else { output_axis })
                    .collect::<Vec<_>>();
                output_axes.insert(batch_axis, batch_axis);
                let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
                let output_sharding = self.output_type().sharding().cloned();
                let input_mesh = inputs.iter().find_map(|input| {
                    input.batch_axis_position()?;
                    input.r#type().sharding().map(|sharding| sharding.mesh().clone())
                });
                let output_sharding = match (output_sharding, input_mesh) {
                    (Some(sharding), _) => Some(sharding),
                    (None, Some(mesh)) if !matches!(&axis_sharding, ShardingDimension::Replicated) => {
                        Some(Sharding::replicated(mesh, self.output_type().rank()))
                    }
                    (None, None) => None,
                    (None, Some(_)) => None,
                };
                output_type.sharding =
                    output_sharding.map(|sharding| sharding.batched(batch_axis, axis_sharding)).transpose()?;
                // Broadcast packed storage using its physical extents, then remap the logical ragged geometry.
                let mut dimensions = output_type.shape().dimensions().to_vec();
                let ragged_axes = inputs[0]
                    .ragged_axes()
                    .iter()
                    .map(|ragged_axis| {
                        let axis = output_axes[ragged_axis.axis()];
                        dimensions[axis] = inputs[0].value().r#type().dimension(ragged_axis.axis());
                        RaggedAxis::new(
                            axis,
                            ragged_axis.extents().clone(),
                            ragged_axis.dimension().clone(),
                            ragged_axis.extent_axes().iter().map(|axis| output_axes[*axis]).collect(),
                        )
                    })
                    .collect::<Vec<_>>();
                output_type = output_type.with_shape(Shape::new(dimensions));
                let output_value = inputs[0].value().broadcast(output_type, output_axes.as_slice())?;
                Ok(vec![
                    ArrayBatch::new(output_value, BatchAxis::from_position(batch_axis))?
                        .with_ragged_axes(ragged_axes)?,
                ]
                .into())
            }
        }
    }
}

impl_differentiable_operation! {
    BroadcastOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType, Value: Broadcast, Operation: From<BroadcastOperation>>,
    {
        |operation, _context, _driver, inputs| {
            // Forward-mode differentiation rule for `BroadcastOperation`. Broadcasting is structural-linear, so
            // tangent follows the same axis mapping as the primal. A structural-zero input tangent remains structural
            // and acquires the primal output's tangent type.
            check_count!("input", inputs, 1, ProgramError);
            let primal =
                inputs[0].primal().broadcast(operation.output_type().clone(), operation.output_axes())?;
            let tangent_type = primal.r#type().tangent()?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(tangent_type),
                MaybeZero::Value(tangent) => {
                    MaybeZero::Value(tangent.broadcast(tangent_type, operation.output_axes())?)
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType>
            + From<AddOperation<ArrayType>>
            + From<BroadcastOperation>
            + From<ConvertElementTypeOperation<ArrayType>>
            + From<ReduceOperation>
            + From<TransposeOperation>
            + From<ReshapeOperation>
            + From<ReshardOperation>
            + From<ZeroLikeOperation<ArrayType>>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // Transposition rule for `BroadcastOperation`. The pullback of a broadcast is a sum-reduction over
            // output axis the input was replicated along (i.e., the axes of the target type that are not named in
            // `output_axes`, plus the mapped axes whose input extent is `1` stretched to a larger target extent).
            // After the reduction, the surviving axes are reordered into input-axis order when `output_axes` is not
            // monotonically increasing, and stretched unit axes are restored with a reshape so the cotangent matches
            // the input type exactly. Symbolic-zero cotangents propagate unchanged, and an input with no cotangent
            // space receives the structural zero of that space.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            let input_cotangent_type = inputs[0].r#type().cotangent()?;
            if input_cotangent_type.is_zero_space() {
                return Ok(());
            }
            let MaybeZero::Value(cotangent) = &outputs[0] else {
                return Ok(());
            };
            {
                let contribution = MaybeZero::Value(
                    cotangent.unalign_cotangent_along(&input_cotangent_type, operation.output_axes())?,
                );
                accumulators[0].accumulate(context, contribution)?;
                Ok(())
            }
        }
    },
}

/// Replicates an array along new axes or expands axes of size one to a requested output shape. Input axis `i` maps to
/// output axis `output_axes[i]`; axes need not stay in increasing order, but each input axis must map to a distinct
/// output axis. A mapped extent must equal its input extent or expand a static size-one axis to a static size. New
/// axes must have static sizes. An existing dynamic extent can pass through unchanged; replication into a dynamic
/// extent requires [`DynamicBroadcast`] so that its size is available as a value.
///
/// Broadcasting preserves the element type and memory space. The primitive accepts the complete output [`ArrayType`],
/// including its layout and sharding. The convenience functions infer output sharding from the input axis mapping and
/// clear the physical layout when the shape changes, since replication does not determine an output storage layout.
/// An identity broadcast with unchanged metadata passes the input through unchanged.
///
/// [`Broadcast`] fills the same role for [`BroadcastOperation`] that [`std::ops::Add`] and [`std::ops::Neg`] fill for
/// their corresponding arithmetic [`Operation`]s. It also operates on [`ArrayType`] to validate output geometry without
/// computing array values. The geometry is stored in the operation rather than supplied through dimension operands,
/// allowing mixed-operation transform rules to use this capability once their operand extents have been resolved.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, Broadcast, ProgramError, StaticShape};
/// # fn main() -> Result<(), ProgramError> {
/// let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
/// let output = input.broadcast_to(StaticShape::new(vec![2, 3]))?;
/// assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
/// assert_eq!(input.broadcast_leading([2])?, output);
/// # Ok(())
/// # }
/// ```
pub trait Broadcast: Sized {
    /// Replicates or reorders the axes of `self` into the supplied output type. Mapped axes must have equal extents,
    /// except that a static input extent of one may expand to any static output extent, including zero. Unmapped
    /// output axes replicate the entire input and must have static extents. Invalid mappings, incompatible extents,
    /// or a change of element type or memory space yield a [`TypeError`].
    ///
    /// # Parameters
    ///
    ///   - `output_type`: Complete result type, including shape, layout, sharding, and memory space. Its element type
    ///     and memory space must match the input. Use [`Broadcast::broadcast_to`] to infer this metadata from a shape.
    ///   - `output_axes`: Output axis receiving each input axis, in input-axis order. There must be exactly one entry
    ///     per input axis, with distinct values in `0..output_type.rank()`. Axes absent from this mapping are new axes.
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError>;

    /// Broadcasts `self` to a shape by aligning input axes with the trailing output axes. Each aligned extent must
    /// match or expand a static size-one input extent. A smaller output rank or an incompatible extent yields a
    /// [`TypeError`]. Element type and memory space are preserved; input sharding follows the aligned axes and new
    /// leading axes are replicated. An unchanged shape returns the input with its existing layout.
    ///
    /// # Parameters
    ///
    ///   - `output_shape`: Desired result shape. For example, broadcasting `[3]` to `[2, 3]` repeats the input as two
    ///     rows. Dynamic extents may only preserve identical input extents; use [`DynamicBroadcast::dynamic_broadcast_to`]
    ///     when a replication count is supplied by a dimension value.
    #[inline]
    fn broadcast_to<S: Into<Shape>>(&self, output_shape: S) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        self.broadcast_to_with_output_sharding(output_shape, None)
    }

    /// Broadcasts to `output_shape` with trailing-axis alignment and an optional complete output sharding. Shape,
    /// element type, memory, and layout behavior follow [`Self::broadcast_to`]. An explicit sharding is applied even
    /// when the shape is unchanged; invalid sharding ranks return a [`TypeError`].
    ///
    /// # Parameters
    ///
    ///   - `output_shape`: Desired result shape, including unchanged trailing input dimensions.
    ///   - `output_sharding`: Requested result distribution. `None` maps the input sharding onto the aligned axes and
    ///     replicates new axes; it does not remove existing sharding. `Some(sharding)` replaces that inferred result.
    fn broadcast_to_with_output_sharding<S: Into<Shape>>(
        &self,
        output_shape: S,
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let input_type = self.r#type();
        let output_shape = output_shape.into();
        let offset = output_shape.rank().checked_sub(input_type.rank()).ok_or_else(|| {
            TypeError::invalid(format!(
                "cannot broadcast rank-{} input to {} output dimensions",
                input_type.rank(),
                output_shape.rank(),
            ))
        })?;
        let output_axes = (offset..output_shape.rank()).collect::<Vec<_>>();
        let output_sharding = match output_sharding {
            Some(sharding) => Some(sharding),
            None => input_type
                .sharding()
                .map(|sharding| sharding.with_broadcasted_dimensions(output_shape.rank(), &output_axes))
                .transpose()
                .map_err(|error| TypeError::invalid(error.to_string()))?,
        };
        let output_type = if input_type.shape() == &output_shape {
            input_type.into_owned()
        } else {
            ArrayType::new(input_type.data_type(), output_shape).with_memory(input_type.memory())
        }
        .with_sharding(output_sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))?;
        self.broadcast(output_type, &output_axes)
    }

    /// Replicates `self` along new leading axes, preserving all existing axes and their extents. This is equivalent
    /// to [`Broadcast::broadcast_to`] with the leading shape followed by the input shape. An empty leading shape
    /// leaves the input unchanged. New axes have replicated sharding; existing sharding follows the shifted axes.
    ///
    /// # Parameters
    ///
    ///   - `leading_shape`: Sizes of the axes to prepend, in output-axis order. For example, prepending `[2, 4]` to a
    ///     vector of size three produces shape `[2, 4, 3]`. These sizes must be static; use
    ///     [`DynamicBroadcast::dynamic_broadcast_leading`] for leading dimensions supplied as values.
    #[inline]
    fn broadcast_leading<S: Into<Vec<usize>>>(&self, leading_shape: S) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        self.broadcast_leading_with_output_sharding(leading_shape, None)
    }

    /// Prepends static replication axes with an optional complete output sharding. Existing extents remain unchanged,
    /// and an empty leading shape can still change the sharding. Other semantics follow [`Self::broadcast_leading`].
    ///
    /// # Parameters
    ///
    ///   - `leading_shape`: Sizes of the new leading axes, in output-axis order.
    ///   - `output_sharding`: Distribution for the full result rank, including existing axes. `None` preserves mapped
    ///     input sharding and replicates the new axes, as in [`Self::broadcast_to_with_output_sharding`].
    fn broadcast_leading_with_output_sharding<S: Into<Vec<usize>>>(
        &self,
        leading_shape: S,
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let mut dimensions = leading_shape.into().into_iter().map(Dimension::Static).collect::<Vec<_>>();
        dimensions.extend_from_slice(self.r#type().shape().dimensions());
        self.broadcast_to_with_output_sharding(Shape::new(dimensions), output_sharding)
    }

    /// Broadcasts every input to their common shape, preserving input order and each input's element type and memory
    /// space. Only shapes are combined: this function performs no numeric promotion. Each result's layout and sharding
    /// follow [`Self::broadcast_to`]. Empty input returns an empty vector, and incompatible shapes return a [`TypeError`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{Array, ArrayType, Broadcast, DataType, ProgramError};
    /// let inputs = [
    ///     Array::from_elements(ArrayType::new_static(DataType::I32, [2, 1]), &[1_i32, 2])?,
    ///     Array::from_elements(ArrayType::new_static(DataType::F32, [3]), &[3_f32, 4.0, 5.0])?,
    /// ];
    /// let outputs = Array::broadcast_arrays(&inputs)?;
    /// assert_eq!(outputs[0].elements::<i32>()?, vec![1, 1, 1, 2, 2, 2]);
    /// assert_eq!(outputs[1].elements::<f32>()?, vec![3.0, 4.0, 5.0, 3.0, 4.0, 5.0]);
    /// # Ok::<(), ProgramError>(())
    /// ```
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Values to align by their trailing axes. Dimension sizes must match or expand a size-one axis.
    ///     Use [`DynamicBroadcast::dynamic_broadcast_arrays`] when replication needs runtime dimension values.
    fn broadcast_arrays(inputs: &[Self]) -> Result<Vec<Self>, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let shapes = inputs.iter().map(|input| input.r#type().shape().clone()).collect::<Vec<_>>();
        let output_shape: Shape = crate::arrays::Broadcastable::broadcasted(&shapes)
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        inputs.iter().map(|input| input.broadcast_to(output_shape.clone())).collect()
    }
}

impl Broadcast for ArrayType {
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<ArrayType, ProgramError> {
        if self.data_type() != output_type.data_type() {
            return Err(TypeError::invalid(format!(
                "broadcasting input data type `{}` does not match output data type `{}`",
                self.data_type(),
                output_type.data_type(),
            ))
            .into());
        }

        if self.memory() != output_type.memory() {
            return Err(TypeError::invalid(format!(
                "broadcasting input memory {} does not match output memory {}",
                self.memory(),
                output_type.memory(),
            ))
            .into());
        }

        let input_rank = self.rank();
        let output_rank = output_type.rank();
        if output_axes.len() != input_rank {
            return Err(TypeError::invalid(format!(
                "broadcasting output axes has length {} but input has rank {}",
                output_axes.len(),
                input_rank,
            ))
            .into());
        }

        let mut seen = vec![false; output_rank];
        for (input_axis, &output_axis) in output_axes.iter().enumerate() {
            if output_axis >= output_rank {
                return Err(TypeError::invalid(format!(
                    "broadcasting `output_axes[{}] = {}` is out of bounds for output rank {}",
                    input_axis, output_axis, output_rank,
                ))
                .into());
            }
            if seen[output_axis] {
                return Err(TypeError::invalid(format!(
                    "broadcasting output axes map two input axes to output axis {output_axis}",
                ))
                .into());
            }
            seen[output_axis] = true;

            let input_dimension = self.dimension(input_axis);
            let output_dimension = output_type.dimension(output_axis);
            match (input_dimension, output_dimension.clone()) {
                // Identical sizes always map through, including identical dynamic sizes.
                (input_dimension, output_dimension) if input_dimension == output_dimension => {}
                // A singleton dynamic bound proves the same extent even when a residual has refined it to static.
                (Dimension::Static(size), Dimension::Dynamic(variable))
                    if variable.bounds() == Dimension::Static(size).bounds() => {}
                // A static size-1 input dimension is replicated to match any static output extent. Expanding it
                // into a dynamic output dimension is unsupported because the replication count is unknown.
                (Dimension::Static(1), Dimension::Static(_)) => {}
                (Dimension::Static(1), Dimension::Dynamic(_)) => {
                    return Err(TypeError::invalid(format!(
                        "broadcasting cannot expand input axis {} of size 1 into dynamic output size {}",
                        input_axis, output_dimension,
                    ))
                    .into());
                }
                (Dimension::Static(input_size), Dimension::Static(output_size)) => {
                    return Err(TypeError::invalid(format!(
                        "broadcasting input axis {} has size {}, which is neither {} nor 1",
                        input_axis, input_size, output_size,
                    ))
                    .into());
                }
                // All remaining combinations pair a dynamic size with a mismatched size on the other side.
                (input_dimension, output_dimension) => {
                    return Err(TypeError::invalid(format!(
                        "broadcasting input axis {input_axis} has size {input_dimension} but the output has size \
                            {output_dimension}; a dynamic dimension only broadcasts to an identical dynamic dimension",
                    ))
                    .into());
                }
            }
        }

        // Output axes that no input axis maps to replicate the input along that axis, which requires a known
        // replication count and is therefore unsupported for dynamic output dimensions.
        for (output_axis, mapped) in seen.iter().enumerate() {
            let output_dimension = output_type.dimension(output_axis);
            if !mapped && matches!(output_dimension, Dimension::Dynamic(_)) {
                return Err(TypeError::invalid(format!(
                    "broadcasting cannot replicate the input into unmapped dynamic output axis {} of size {}",
                    output_axis, output_dimension,
                ))
                .into());
            }
        }
        validate_broadcast_output_layout(&output_type)?;
        Ok(output_type)
    }
}

impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<BroadcastOperation>>>>
    Broadcast for V
{
    #[inline]
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        let output_type = self.r#type().broadcast(output_type, output_axes)?;
        if self.r#type().as_ref() == &output_type && output_axes.iter().copied().eq(0..output_type.rank()) {
            return Ok(self.clone());
        }
        let mut outputs = self.dispatch_domain().bind(
            BroadcastOperation::new(output_type, output_axes.to_vec()),
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Mixed [`Operation`] that broadcasts one array using one explicit first-class dimension operand per output axis.
///
/// Operand zero is the array. Every remaining operand describes the corresponding output-axis extent, in order.
/// Exact dimension types produce static axes while non-exact dimension types retain their variables as dynamic axes.
/// `output_axes[input_axis]` names the output axis to which that input axis maps.
/// Refer to [`DynamicBroadcast`] for the underlying array broadcasting semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicBroadcastOperation {
    /// Vector that contains, for each input axis, the output axis to which it maps.
    output_axes: Vec<usize>,

    /// Optional requested output [`Sharding`].
    output_sharding: Option<Sharding>,

    /// Optional requested physical output layout.
    output_layout: Option<Layout>,
}

impl DynamicBroadcastOperation {
    /// Creates a broadcast with the supplied input-to-output axis mapping.
    #[inline]
    pub fn new(output_axes: Vec<usize>) -> Self {
        Self { output_axes, output_sharding: None, output_layout: None }
    }

    /// Returns the output axes. The resulting slice contains, for each input axis, the output axis to which it maps.
    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        self.output_axes.as_slice()
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }

    /// Returns the requested physical output layout, if any.
    #[inline]
    pub fn output_layout(&self) -> Option<&Layout> {
        self.output_layout.as_ref()
    }

    /// Returns this operation with the requested output `sharding`.
    #[inline]
    pub fn with_output_sharding(mut self, sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = sharding.into();
        self
    }

    /// Returns this operation with the requested physical output `layout`.
    ///
    /// Layout rank and structure are validated during type inference. Storage geometry depending on dynamic extents
    /// is validated once their runtime values are available. Without an explicit layout, shape-changing broadcasts
    /// use the default layout; identity broadcasts preserve the input layout. Mapped batching supports tiled layouts
    /// by placing the batch axis outside each item's storage. Explicit strided layouts currently support replicated
    /// batching only, because a leading batch stride can depend on runtime output sizes.
    #[inline]
    pub fn with_output_layout(mut self, layout: impl Into<Option<Layout>>) -> Self {
        self.output_layout = layout.into();
        self
    }
}

impl Display for DynamicBroadcastOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DynamicBroadcastOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        BROADCAST_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let Some((input_type, output_extent_types)) = input_types.split_first() else {
            return Err(TypeError::invalid(format!(
                "`{BROADCAST_OPERATION_NAME}` expects an array followed by its output extents"
            )));
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        let output_shape = Shape::new(ArrayIrType::extents(output_extent_types)?);
        Ok(vec![infer_explicit_broadcast_output_type(input_type, output_shape, self)?.into()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, BROADCAST_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("output_axes", format_args!("{:?}", self.output_axes))?;
            if let Some(output_layout) = &self.output_layout {
                operation.field("output_layout", output_layout)?;
            }
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free DynamicBroadcastOperation);

impl<C> InterpretableOperation<C> for DynamicBroadcastOperation
where
    C: Domain<Type = ArrayIrType>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType> + Broadcast>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(TypeError::invalid(format!(
                "`{BROADCAST_OPERATION_NAME}` expects an array followed by its output extents"
            ))
            .into());
        };
        let input = <C::Value as ValueProjection<ArrayType>>::into_projected(input.clone())?;
        let mut refinements = ArrayTypeRefinements::default();
        let output_shape = Shape::new(
            output_extents
                .iter()
                .cloned()
                .map(<C::Value as ValueProjection<DimensionType>>::into_projected)
                .map(|result| {
                    let extent = result?;
                    // Shared dimension identities denote one size across all output axes.
                    refinements.bind(extent.r#type().variable(), extent.extent())?;
                    Ok(Dimension::Static(extent.extent()))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?,
        );
        let output_type = infer_explicit_broadcast_output_type(input.r#type().as_ref(), output_shape, self)?;
        Ok(vec![<C::Value as ValueProjection<ArrayType>>::from_projected(
            input.broadcast(output_type, self.output_axes())?,
        )])
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<DynamicBroadcastOperation>>> PartiallyEvaluatableOperation<C>
    for DynamicBroadcastOperation
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
            && self.output_axes().iter().copied().eq(0..input_type.rank())
            && self
                .infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?
                == vec![input.r#type().into_owned()]
        {
            // A static identity broadcast cannot observe its exact dimension operands. Preserve the input directly
            // so an unknown array does not leave a redundant broadcast in the residual program.
            return Ok(vec![input.clone()]);
        }
        context.fold_or_residualize(self.clone(), driver.regions().map(|region| region.to_program()).collect(), inputs)
    }
}

/// Batching rule for [`DynamicBroadcastOperation`]. A mapped input is canonicalized to a leading batch axis, which is
/// represented in both the lifted output extents and the input-to-output axis mapping. A mapped output extent uses its
/// declared finite bound as physical packed storage and records its per-item extent vector as transform-owned ragged
/// metadata; replicated extents retain their existing first-class representation.
impl<C> BatchableOperation<C, ArrayIrBatchingPolicy> for DynamicBroadcastOperation
where
    C: Context<Type = ArrayIrType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    C::Operation: From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let input_type = input.unbatched_type();
        let input_rank = <&ArrayType>::try_from(&input_type)?.rank();
        let output_rank = output_extents.len();
        // Validate the mapping before shifting its indices for the leading batch axis.
        if self.output_axes().len() != input_rank {
            return Err(TypeError::invalid(format!(
                "broadcasting output axes has length {} but input has rank {}",
                self.output_axes().len(),
                input_rank,
            ))
            .into());
        }
        let mut mapped_output_axes = vec![false; output_rank];
        for (input_axis, &output_axis) in self.output_axes().iter().enumerate() {
            if output_axis >= output_rank {
                return Err(TypeError::invalid(format!(
                    "broadcasting `output_axes[{}] = {}` is out of bounds for output rank {}",
                    input_axis, output_axis, output_rank,
                ))
                .into());
            }
            if mapped_output_axes[output_axis] {
                return Err(TypeError::invalid(format!(
                    "broadcasting output axes map two input axes to output axis {output_axis}",
                ))
                .into());
            }
            mapped_output_axes[output_axis] = true;
        }
        // Eager replay can concretize ordinary array dimensions while their dimension inputs retain declared
        // dynamic types. Parent binding validates that runtime geometry; only ragged identities must be checked
        // here, before replacing their logical dimensions with physical packed extents.
        for ragged_axis in input.ragged_axes() {
            let input_axis = ragged_axis.axis()
                - usize::from(input.batch_axis_position().is_some_and(|axis| axis < ragged_axis.axis()));
            let output_axis =
                self.output_axes().get(input_axis).copied().ok_or_else(|| BatchingError::InvalidBatchMetadata {
                    message: format!("ragged broadcast input axis {input_axis} has no output-axis mapping"),
                })?;
            let extent = output_extents.get(output_axis).ok_or_else(|| BatchingError::InvalidBatchMetadata {
                message: format!("ragged broadcast output axis {output_axis} has no dimension input"),
            })?;
            let extent_type = extent.unbatched_type();
            let extent_type = <&DimensionType>::try_from(&extent_type)?;
            if extent_type.variable() != ragged_axis.dimension() {
                return Err(BatchingError::InvalidBatchMetadata {
                    message: format!(
                        "ragged broadcast input dimension `{}` does not match output dimension `{}`",
                        ragged_axis.dimension(),
                        extent_type.variable(),
                    ),
                });
            }
        }
        let ragged_extents = output_extents
            .iter()
            .enumerate()
            .filter_map(|(axis, extent)| extent.mapped_dimension_extents().map(|extents| (axis, extent, extents)))
            .collect::<Vec<_>>();
        for extent in output_extents {
            if extent.mapped_dimension_extents().is_none() {
                extent.validate_replicated_dimension()?;
            }
        }

        if input.batch_axis().is_replicated() && ragged_extents.is_empty() {
            let mut packed_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
            let mut ragged_axes = Vec::with_capacity(input.ragged_axes().len());
            let packed_type = input.value().r#type();
            let packed_type = <&ArrayType>::try_from(packed_type.as_ref())?;
            for ragged_axis in input.ragged_axes() {
                let axis = self.output_axes()[ragged_axis.axis()];
                let Dimension::Static(physical_extent) = packed_type.dimension(ragged_axis.axis()) else {
                    return Err(BatchingError::InvalidBatchMetadata {
                        message: "ragged broadcast storage requires a static physical extent".to_owned(),
                    });
                };
                packed_inputs[axis + 1] = context.parent().dimension_constant(physical_extent)?;
                ragged_axes.push(RaggedAxis::new(
                    axis,
                    ragged_axis.extents().clone(),
                    ragged_axis.dimension().clone(),
                    ragged_axis.extent_axes().iter().map(|axis| self.output_axes()[*axis]).collect(),
                ));
            }
            let mut outputs = context.parent().bind(self.clone(), Vec::new(), packed_inputs.as_slice())?;
            check_count!("output", outputs, 1, ProgramError);
            return Ok(vec![ArrayIrBatch::replicated(outputs.remove(0)).with_ragged_axes(ragged_axes)?].into());
        }

        if matches!(self.output_layout(), Some(Layout::Strided(_))) {
            return Err(BatchingError::InvalidBatchMetadata {
                message: "mapped dynamic broadcasting with an explicit strided output layout requires a runtime batch stride, which is not supported".to_owned(),
            });
        }
        let moved_input = driver.align_batch_axis(context, input.clone(), Axis::from(0))?;

        let mut lifted_output_axes = Vec::with_capacity(self.output_axes().len() + 1);
        lifted_output_axes.push(0);
        lifted_output_axes.extend(self.output_axes().iter().map(|axis| axis + 1));
        let mut operation = Self::new(lifted_output_axes);
        if let Some(Layout::Tiled(layout)) = self.output_layout() {
            // Keep each item's tiled storage together by making the leading batch axis most major.
            let mut minor_to_major = layout.minor_to_major().iter().map(|axis| axis + 1).collect::<Vec<_>>();
            minor_to_major.push(0);
            operation =
                operation.with_output_layout(Layout::Tiled(TiledLayout::new(minor_to_major, layout.tiles().to_vec())));
        }
        if let Some(output_sharding) = self.output_sharding() {
            operation = operation.with_output_sharding(lift_output_sharding_for_leading_batch_axis(
                output_sharding,
                context.axis_sharding().clone(),
            )?);
        }

        let mut lifted_inputs = Vec::with_capacity(inputs.len() + 1);
        lifted_inputs.push(moved_input.value().clone());
        lifted_inputs.push(context.axis_extent().clone());
        for (axis, extent) in output_extents.iter().enumerate() {
            if extent.mapped_dimension_extents().is_some() {
                let unbatched_type = extent.unbatched_type();
                let extent_type = <&DimensionType>::try_from(&unbatched_type)?;
                let physical_extent =
                    extent_type.bounds().upper().and_then(|upper| upper.checked_sub(1)).ok_or_else(|| {
                        BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "ragged broadcast dimension {} requires a finite, nonempty declared upper bound",
                                extent_type.variable(),
                            ),
                        }
                    })?;
                lifted_inputs.push(context.parent().dimension_constant(physical_extent)?);
            } else if let Some(ragged_axis) = moved_input
                .ragged_axes()
                .iter()
                .find(|ragged_axis| self.output_axes()[ragged_axis.axis() - 1] == axis)
            {
                // Existing ragged geometry also uses packed extents when the output dimension is replicated.
                let packed_type = moved_input.value().r#type();
                let packed_type = <&ArrayType>::try_from(packed_type.as_ref())?;
                let Dimension::Static(physical_extent) = packed_type.dimension(ragged_axis.axis()) else {
                    return Err(BatchingError::InvalidBatchMetadata {
                        message: "ragged broadcast storage requires a static physical extent".to_owned(),
                    });
                };
                lifted_inputs.push(context.parent().dimension_constant(physical_extent)?);
            } else {
                lifted_inputs.push(extent.value().clone());
            }
        }
        let mut outputs = context.parent().bind(operation, Vec::new(), lifted_inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);

        let mut ragged_axes = moved_input
            .ragged_axes()
            .iter()
            .cloned()
            .map(|ragged_axis| {
                let axis = if ragged_axis.axis() == 0 { 0 } else { self.output_axes()[ragged_axis.axis() - 1] + 1 };
                let extent_axes = ragged_axis
                    .extent_axes()
                    .iter()
                    .map(|axis| if *axis == 0 { 0 } else { self.output_axes()[*axis - 1] + 1 })
                    .collect();
                RaggedAxis::new(axis, ragged_axis.extents().clone(), ragged_axis.dimension().clone(), extent_axes)
            })
            .collect::<Vec<_>>();
        for (axis, extent, extents) in ragged_extents {
            let unbatched_type = extent.unbatched_type();
            let extent_type = <&DimensionType>::try_from(&unbatched_type)?;
            let axis = axis + 1;
            // Explicit output dimensions own the output's ragged metadata. A mapped input can already carry
            // this same axis; replace that entry rather than recording the axis twice.
            ragged_axes.retain(|ragged_axis| ragged_axis.axis() != axis);
            ragged_axes.push(RaggedAxis::new(axis, extents.clone(), extent_type.variable().clone(), vec![0]));
        }
        let output = ArrayIrBatch::new(outputs.remove(0), BatchAxis::from_position(0))?;
        Ok(vec![output.with_ragged_axes(ragged_axes)?].into())
    }
}

/// Forward-mode rule for mixed broadcast. The explicit output extents are ordinary non-differentiated shape values.
/// Exact input geometry replays the mixed broadcast directly; dynamic input geometry retains its exact extents so the
/// linear transpose can reduce, reorder, and rebind the input cotangent using first-class dimension residuals.
impl<C> DifferentiableOperation<C> for DynamicBroadcastOperation
where
    C: Context<Type = ArrayIrType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<DynamicBroadcastOperation>
        + From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + From<DynamicReshapeOperation>
        + From<ZeroOperation<ArrayType>>
        + From<ConstantOperation<DimensionValue>>
        + OperationProjection<ArrayType, Projected: From<ReduceOperation> + From<TransposeOperation>>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let destinations = context;
        let context = destinations.primal();
        let Some(_) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut primal_outputs = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?;
        check_count!("output", primal_outputs, 1, ProgramError);
        let output_primal = primal_outputs.remove(0);
        let primal = destinations.primal_to_tangent(output_primal.clone())?;
        let tangent_inputs = destinations.dual_primal_to_tangent(inputs)?;
        let inputs = tangent_inputs.as_slice();
        let (array, output_extents) = inputs.split_first().unwrap();
        let context = destinations.tangent();
        // Tangent promotion can change element width, so derive the layout from the tangent type rather than
        // replaying byte strides captured for the primal.
        let tangent_type = primal.r#type().tangent()?;
        let tangent_operation =
            self.clone().with_output_layout(<&ArrayType>::try_from(&tangent_type)?.layout().cloned());
        let tangent = match array.tangent() {
            MaybeZero::Zero(_) => {
                if tangent_type.identities().any(|(position, _)| position == TypeIdentityPosition::Reference) {
                    let array_tangent_type = <&ArrayType>::try_from(&tangent_type)?.clone();
                    let dynamic_extents = array_tangent_type
                        .shape()
                        .dimensions()
                        .iter()
                        .zip(output_extents)
                        .filter_map(|(dimension, extent)| {
                            matches!(dimension, Dimension::Dynamic(_)).then(|| extent.primal().clone())
                        })
                        .collect::<Vec<_>>();
                    let mut outputs =
                        context.bind(ZeroOperation::new(array_tangent_type), Vec::new(), dynamic_extents.as_slice())?;
                    check_count!("output", outputs, 1, ProgramError);
                    MaybeZero::Value(outputs.remove(0))
                } else {
                    MaybeZero::Zero(tangent_type)
                }
            }
            MaybeZero::Value(array_tangent) => {
                let input_cotangent_type = <&ArrayType>::try_from(array.primal().r#type().as_ref())?.cotangent()?;
                if input_cotangent_type
                    .shape()
                    .dimensions()
                    .iter()
                    .all(|dimension| matches!(dimension, Dimension::Static(_)))
                {
                    let mut tangent_inputs = Vec::with_capacity(inputs.len());
                    tangent_inputs.push(array_tangent.clone());
                    tangent_inputs.extend(output_extents.iter().map(|extent| extent.primal().clone()));
                    let mut outputs = context.bind(tangent_operation, Vec::new(), tangent_inputs.as_slice())?;
                    check_count!("output", outputs, 1, ProgramError);
                    MaybeZero::Value(outputs.remove(0))
                } else {
                    let mut residuals = LinearResiduals::new();
                    let output_extents =
                        residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                    let input_shape = residuals.retain_shape(context, array.primal())?;
                    let forward_operation = tangent_operation;
                    let forward_output_extents = output_extents.clone();
                    let transpose_output_axes = self.output_axes().to_vec();
                    let transpose_target_type = input_cotangent_type.clone();
                    let mut tangent_outputs = LinearCallOperation::stage(
                        context,
                        residuals.into_values(),
                        vec![array_tangent.clone()],
                        move |residuals, linear_inputs| {
                            let mut broadcast_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                            broadcast_inputs.push(linear_inputs[0].clone());
                            broadcast_inputs
                                .extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                            linear_inputs[0].dispatch_domain().bind(
                                forward_operation,
                                Vec::new(),
                                broadcast_inputs.as_slice(),
                            )
                        },
                        move |residuals, output_cotangents| {
                            let transpose_context = output_cotangents[0].dispatch_domain();
                            let mut contribution =
                                <Tracer<NestedTracingContext<C>> as ValueProjection<ArrayType>>::into_projected(
                                    output_cotangents[0].clone(),
                                )?;
                            let contribution_type = contribution.r#type().into_owned();
                            if transpose_output_axes.len() != transpose_target_type.rank()
                                || transpose_output_axes.iter().any(|axis| *axis >= contribution_type.rank())
                            {
                                return Err(TypeError::invalid(format!(
                                    "cannot unalign cotangent type {contribution_type} to input cotangent type \
                                     {transpose_target_type} using output axes {transpose_output_axes:?}",
                                ))
                                .into());
                            }
                            let mut kept_axes = Vec::with_capacity(transpose_target_type.rank());
                            for (target_axis, output_axis) in transpose_output_axes.iter().copied().enumerate() {
                                let target_dimension = transpose_target_type.dimension(target_axis);
                                let output_dimension = contribution_type.dimension(output_axis);
                                if target_dimension == output_dimension {
                                    kept_axes.push((target_axis, output_axis));
                                } else if target_dimension != Dimension::Static(1) {
                                    return Err(TypeError::invalid(format!(
                                        "cannot unalign cotangent axis {output_axis} of size {output_dimension} to \
                                         input axis {target_axis} of size {target_dimension}",
                                    ))
                                    .into());
                                }
                            }
                            let reduce_axes = (0..contribution_type.rank())
                                .filter(|axis| kept_axes.iter().all(|(_, output_axis)| output_axis != axis))
                                .collect::<Vec<_>>();
                            if !reduce_axes.is_empty() {
                                contribution = contribution.reduce(reduce_axes.as_slice(), ReductionKind::Sum);
                            }
                            let mut kept_axes_by_output = kept_axes.clone();
                            kept_axes_by_output.sort_by_key(|(_, output_axis)| *output_axis);
                            let permutation = kept_axes
                                .iter()
                                .map(|kept| kept_axes_by_output.iter().position(|candidate| candidate == kept).unwrap())
                                .collect::<Vec<_>>();
                            if permutation.iter().enumerate().any(|(axis, position)| axis != *position) {
                                contribution = Transpose::transpose(&contribution, permutation)?;
                            }
                            let contribution = contribution.into_value();

                            // Rebind exact input geometry through ordinary dimension operands. This prevents a
                            // metadata-only identity from standing in for runtime geometry at the linear boundary.
                            let contribution_type = contribution.r#type().into_owned();
                            let mut exact_inputs = Vec::with_capacity(transpose_target_type.rank() + 1);
                            exact_inputs.push(contribution);
                            exact_inputs.extend(input_shape.dimensions(&transpose_context, residuals)?);
                            let contribution_type = <&ArrayType>::try_from(&contribution_type)?;
                            if contribution_type.shape() == transpose_target_type.shape() {
                                transpose_context.bind(
                                    DynamicBroadcastOperation::new((0..transpose_target_type.rank()).collect())
                                        .with_output_sharding(transpose_target_type.sharding().cloned())
                                        .with_output_layout(transpose_target_type.layout().cloned()),
                                    Vec::new(),
                                    exact_inputs.as_slice(),
                                )
                            } else {
                                let mut outputs = transpose_context.bind(
                                    DynamicReshapeOperation::new()
                                        .with_output_sharding(transpose_target_type.sharding().cloned()),
                                    Vec::new(),
                                    exact_inputs.as_slice(),
                                )?;
                                check_count!("output", outputs, 1, ProgramError);
                                // Reshaping restores geometry but clears layout. Reapply the input cotangent's
                                // complete storage metadata at the linear boundary.
                                if transpose_target_type.layout().is_some() {
                                    exact_inputs[0] = outputs.remove(0);
                                    transpose_context.bind(
                                        DynamicBroadcastOperation::new((0..transpose_target_type.rank()).collect())
                                            .with_output_sharding(transpose_target_type.sharding().cloned())
                                            .with_output_layout(transpose_target_type.layout().cloned()),
                                        Vec::new(),
                                        exact_inputs.as_slice(),
                                    )
                                } else {
                                    Ok(outputs)
                                }
                            }
                        },
                    )?;
                    check_count!("output", tangent_outputs, 1, ProgramError);
                    MaybeZero::Value(tangent_outputs.remove(0))
                }
            }
        };
        Ok(vec![DifferentiationDual::new(output_primal, tangent)?])
    }
}

/// Direct transposition rule for mixed broadcast. Static input geometry delegates to the homogeneous array pullback,
/// while every explicit output extent receives a structural-zero cotangent. Dynamic input geometry requires
/// linearization so [`DifferentiableOperation::jvp`] can retain its exact extents as residuals.
impl<V, O> TransposableOperation<V, O> for DynamicBroadcastOperation
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    O: Operation<Type = ArrayIrType> + From<AddOperation<ArrayIrType>> + OperationProjection<ArrayType>,
    <O as OperationProjection<ArrayType>>::Projected: From<BroadcastOperation>
        + TransposableOperation<
            <V as ValueProjection<ArrayType>>::Projected,
            <O as OperationProjection<ArrayType>>::Projected,
        >,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);

        let Some((input, _output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let input_cotangent_type = <&ArrayType>::try_from(input.r#type().as_ref())?.cotangent()?;
        if input_cotangent_type
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "direct transposition of a dynamic `{BROADCAST_OPERATION_NAME}` requires linearization so its \
                     input extents can be retained as residuals",
                ),
            }
            .into());
        }

        let output_type = match outputs {
            [MaybeZero::Zero(r#type)] => <&ArrayType>::try_from(r#type)?.clone(),
            [MaybeZero::Value(value)] => <&ArrayType>::try_from(value.r#type().as_ref())?.clone(),
            _ => return Err(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() }.into()),
        };
        let operation = <O as OperationProjection<ArrayType>>::Projected::from(BroadcastOperation::new(
            output_type,
            self.output_axes().to_vec(),
        ));
        // Dimension operands do not receive cotangents; forward only the array operands' handles.
        transpose_projected_operation(context, &operation, std::slice::from_ref(input), outputs, &accumulators[..1])
    }
}

/// Replicates an array using dimension values to specify its output shape. Input axis `i` maps to output axis
/// `output_axes[i]`; mapped extents must match or expand an input extent of one. Unmapped output axes replicate the
/// entire input. Unlike [`Broadcast`], this capability can replicate into dynamic extents because each output size is
/// supplied as an operand, rather than only described by type metadata. The receiver must be an array value and the
/// output sizes must be dimension values, even though both use the same [`ArrayIrType`] value language.
///
/// Exact dimension types describe static axes, while non-exact dimension types identify dynamic axes and their bounds.
/// Both forms bind [`DynamicBroadcastOperation`]; backend lowering chooses a static, bounded, or dynamic representation
/// from the inferred result type. Broadcasting preserves element type and memory space. By default, sharding follows
/// the input-to-output axis mapping and new axes are replicated; callers may request output sharding explicitly.
/// Changed geometry clears the physical layout. An identity broadcast with unchanged metadata passes through its input.
///
/// [`DynamicBroadcast`] fills the same role for [`DynamicBroadcastOperation`] that [`std::ops::Add`] and [`std::ops::Neg`]
/// fill for their corresponding arithmetic [`Operation`]s.
///
/// # Examples
///
/// Exact host sizes can use [`DynamicBroadcast::dynamic_broadcast_to_sizes`]:
///
/// ```rust
/// # use ryft_core::{Array, ArrayIrValue, DynamicBroadcast, ProgramError};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
/// let output = input.dynamic_broadcast_to_sizes(&[2, 3])?;
/// assert_eq!(
///     output,
///     ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap()),
/// );
/// # Ok(())
/// # }
/// ```
///
/// Computed or input dimensions remain ordinary SSA operands:
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType, DimensionBounds, DimensionType,
/// #     DimensionVariable, DynamicBroadcast, ProgramError, StagingContext, TracingContext, Typed,
/// # };
/// #
/// # fn main() -> Result<(), ProgramError> {
/// type C = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
///
/// let context = C::new();
/// let scalar = context.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
/// let extent = context.input(ArrayIrType::Dimension(DimensionType::new(DimensionVariable::new(
///     "extent",
///     DimensionBounds::new(1, Some(9))?,
/// ))));
/// let output = scalar.dynamic_broadcast_to(&[extent])?;
/// assert_eq!(output.r#type().to_string(), "f32[extent]");
/// # Ok(())
/// # }
/// ```
pub trait DynamicBroadcast: Value<Type = ArrayIrType> + Sized {
    /// Replicates or reorders `self` into the shape supplied by dimension values. Each mapped input extent must equal
    /// its output extent or be statically one; new axes may have static or dynamic extents. Sharding follows the axis
    /// mapping and new axes are replicated. Invalid value kinds, axis mappings, or incompatible extents yield a
    /// [`TypeError`]. Use [`DynamicBroadcast::dynamic_broadcast_to`] for the usual trailing-axis alignment.
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: One dimension value per output axis, in output-axis order. Exact dimension types yield
    ///     static extents; other dimension types retain their dynamic identity and bounds in the result. These values
    ///     are operands of the staged operation, so runtime sizes remain available during execution.
    ///   - `output_axes`: Output axis receiving each input axis, in input-axis order. Its length must equal the input
    ///     rank and its entries must be distinct and less than `output_dimensions.len()`. For a vector, `[1]` maps its
    ///     sole axis to the columns of a matrix; the unmapped row axis repeats that vector.
    fn dynamic_broadcast(&self, output_dimensions: &[Self], output_axes: &[usize]) -> Result<Self, ProgramError> {
        self.dynamic_broadcast_with_output_sharding(output_dimensions, output_axes, None)
    }

    /// Broadcasts `self` using explicit output dimensions and an optional output sharding. Shape and axis validation
    /// are the same as for [`DynamicBroadcast::dynamic_broadcast`]. Use this function when the result needs a specified
    /// distribution instead of the distribution inferred by mapping the input sharding onto the output axes.
    /// Broadcasting preserves the input element type and memory space; the requested sharding must match the
    /// output rank. Invalid value kinds, extents, mappings, or sharding yield a [`TypeError`].
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: One dimension value per output axis, in output-axis order. These determine the complete
    ///     output shape, including axes corresponding to unchanged input extents. A scalar broadcast into a matrix
    ///     therefore still supplies both row and column dimensions.
    ///   - `output_axes`: Distinct output axis indices, one per input axis. An empty mapping broadcasts a scalar;
    ///     `[1]` broadcasts a vector along a matrix's rows. Mapped extents must match or expand an input extent of one.
    ///   - `output_sharding`: Requested distribution of the output axes. `Some(sharding)` supplies the complete result
    ///     sharding; `None` infers it from the input, preserving its mesh and reduction state, mapping its axis entries,
    ///     and making newly introduced axes replicated. `None` does not remove an existing input sharding.
    fn dynamic_broadcast_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_axes: &[usize],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError>;

    /// Broadcasts `self` by aligning its axes with the trailing output axes. For example, an input of shape `[3]`
    /// broadcast to dimensions `[2, 3]` becomes two identical rows. Input extents must match the aligned output extents
    /// or be statically one. A smaller target rank or incompatible extents yield a [`TypeError`]. Sharding is inferred
    /// as in [`DynamicBroadcast::dynamic_broadcast`].
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: Dimension values describing the complete desired shape, including unchanged trailing
    ///     axes. Use [`DynamicBroadcast::dynamic_broadcast_to_sizes`] when every desired size is a host integer.
    fn dynamic_broadcast_to(&self, output_dimensions: &[Self]) -> Result<Self, ProgramError> {
        self.dynamic_broadcast_to_with_output_sharding(output_dimensions, None)
    }

    /// Broadcasts to explicit dimension values using trailing-axis alignment and optional output sharding. Dimensions
    /// and validation follow [`Self::dynamic_broadcast_to`]; distribution follows
    /// [`Self::dynamic_broadcast_with_output_sharding`], including when the result shape is unchanged.
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: One dimension value per output axis, including unchanged input axes.
    ///   - `output_sharding`: Complete requested result distribution, or `None` to infer it from the input mapping.
    fn dynamic_broadcast_to_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        let r#type = self.r#type();
        let input_type = <&ArrayType>::try_from(r#type.as_ref())?;
        let offset = output_dimensions.len().checked_sub(input_type.rank()).ok_or_else(|| {
            TypeError::invalid(format!(
                "cannot broadcast rank-{} input to {} output dimensions",
                input_type.rank(),
                output_dimensions.len(),
            ))
        })?;
        let output_axes = (0..input_type.rank()).map(|axis| axis + offset).collect::<Vec<_>>();
        self.dynamic_broadcast_with_output_sharding(output_dimensions, output_axes.as_slice(), output_sharding)
    }

    /// Replicates `self` along new leading axes while preserving the extents of its existing axes. Existing dynamic
    /// extents are obtained from the input with [`DimensionSize::dimension_size`], so callers only supply the new axes.
    /// For example, prepending dimensions `[2, 4]` to a vector of size three produces shape `[2, 4, 3]`. Existing sharding
    /// follows the shifted axes and new axes are replicated. An empty leading shape preserves the input geometry.
    ///
    /// # Parameters
    ///
    ///   - `leading_dimensions`: Dimension values for the new leading axes, in output-axis order. Both exact and dynamic
    ///     dimensions are accepted. These must be dimension values, not scalar array values.
    fn dynamic_broadcast_leading(&self, leading_dimensions: &[Self]) -> Result<Self, ProgramError>
    where
        Self: DimensionSize,
        Self::DispatchDomain: Context<Type = ArrayIrType>,
        Self::DispatchDomain: DimensionConstant,
    {
        self.dynamic_broadcast_leading_with_output_sharding(leading_dimensions, None)
    }

    /// Prepends dimension-valued replication axes with optional output sharding. Existing dimensions are read from
    /// `self`, including runtime sizes, as in [`Self::dynamic_broadcast_leading`].
    ///
    /// # Parameters
    ///
    ///   - `leading_dimensions`: Dimension values for the new leading axes, in output-axis order.
    ///   - `output_sharding`: Complete result distribution, including existing axes. `None` maps existing input
    ///     sharding and replicates new axes; an explicit distribution is applied even for an empty leading shape.
    fn dynamic_broadcast_leading_with_output_sharding(
        &self,
        leading_dimensions: &[Self],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError>
    where
        Self: DimensionSize,
        Self::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant,
    {
        let r#type = self.r#type();
        let input_type = <&ArrayType>::try_from(r#type.as_ref())?;
        let mut output_dimensions = Vec::with_capacity(leading_dimensions.len() + input_type.rank());
        output_dimensions.extend_from_slice(leading_dimensions);
        for (axis, dimension) in input_type.shape().dimensions().iter().enumerate() {
            output_dimensions.push(match dimension {
                Dimension::Static(extent) => self.dispatch_domain().dimension_constant(*extent)?,
                Dimension::Dynamic(_) => self.dimension_size(axis)?,
            });
        }
        let output_axes = (0..input_type.rank()).map(|axis| axis + leading_dimensions.len()).collect::<Vec<_>>();
        self.dynamic_broadcast_with_output_sharding(
            output_dimensions.as_slice(),
            output_axes.as_slice(),
            output_sharding,
        )
    }

    /// Broadcasts `self` to host-provided sizes using trailing-axis alignment. This creates exact dimension constants
    /// in the receiver's dispatch context and delegates to [`DynamicBroadcast::dynamic_broadcast_to`]. The staged
    /// operation therefore has explicit dimension operands even though the output shape is statically known.
    /// Invalid ranks or incompatible extents yield a [`TypeError`].
    ///
    /// # Parameters
    ///
    ///   - `output_sizes`: Complete desired shape as nonnegative host sizes, including unchanged trailing axes.
    ///     For example, `[2, 3]` repeats a vector of size three into two rows. Zero-sized axes are allowed when the
    ///     corresponding input extent is zero or one, or when the axis is newly introduced.
    fn dynamic_broadcast_to_sizes(&self, output_sizes: &[usize]) -> Result<Self, ProgramError>
    where
        Self::DispatchDomain: Context<Type = ArrayIrType>,
        Self::DispatchDomain: DimensionConstant,
    {
        let output_dimensions = output_sizes
            .iter()
            .map(|extent| self.dispatch_domain().dimension_constant(*extent))
            .collect::<Result<Vec<_>, _>>()?;
        self.dynamic_broadcast_to(output_dimensions.as_slice())
    }

    /// Replicates `self` along new leading axes whose sizes are known on the host. This creates exact dimension
    /// constants and delegates to [`DynamicBroadcast::dynamic_broadcast_leading`], preserving existing dynamic extents
    /// through dimension-size operands. Passing an empty slice preserves the input geometry.
    ///
    /// # Parameters
    ///
    ///   - `leading_sizes`: Host sizes of the new axes, in output-axis order. For example, `[2]` repeats the whole input
    ///     twice along a new first axis. Existing axes retain their extents and sharding; new axes are replicated.
    fn dynamic_broadcast_leading_sizes(&self, leading_sizes: &[usize]) -> Result<Self, ProgramError>
    where
        Self: DimensionSize,
        Self::DispatchDomain: Context<Type = ArrayIrType>,
        Self::DispatchDomain: DimensionConstant,
    {
        let context = self.dispatch_domain();
        let leading_dimensions = leading_sizes
            .iter()
            .map(|extent| context.dimension_constant(*extent))
            .collect::<Result<Vec<_>, _>>()?;
        self.dynamic_broadcast_leading(leading_dimensions.as_slice())
    }

    /// Broadcasts array inputs to a common shape while preserving their order and individual element types. Dynamic
    /// result dimensions are read from an input that carries the corresponding dimension identity; static dimensions
    /// become exact dimension constants. No numeric promotion is performed. Empty input returns an empty vector.
    ///
    /// Shape compatibility follows [`Broadcast::broadcast_arrays`]. Invalid array kinds or incompatible shapes return
    /// a [`TypeError`]. The inputs must belong to compatible dispatch contexts so their dimension values can be shared.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{Array, ArrayIrValue, ArrayType, DataType, DynamicBroadcast, ProgramError};
    /// let inputs = [
    ///     ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::I32), &[7_i32])?),
    ///     ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[1_f32, 2.0])?),
    /// ];
    /// let outputs = ArrayIrValue::dynamic_broadcast_arrays(&inputs)?;
    /// assert_eq!(
    ///     outputs[0],
    ///     ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[7_i32, 7])?),
    /// );
    /// # Ok::<(), ProgramError>(())
    /// ```
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Array values to align by trailing axes. Shared dynamic sizes must have identical dimension
    ///     identities; distinct unknown dimensions are not assumed to have equal runtime values.
    fn dynamic_broadcast_arrays(inputs: &[Self]) -> Result<Vec<Self>, ProgramError>
    where
        Self: DimensionSize,
        Self::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant,
    {
        let Some(first) = inputs.first() else {
            return Ok(Vec::new());
        };
        let shapes = inputs
            .iter()
            .map(|input| Ok(<&ArrayType>::try_from(input.r#type().as_ref())?.shape().clone()))
            .collect::<Result<Vec<_>, TypeError>>()?;
        let output_shape: Shape = crate::arrays::Broadcastable::broadcasted(&shapes)
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let output_dimensions = output_shape
            .dimensions()
            .iter()
            .map(|dimension| match dimension {
                Dimension::Static(extent) => first.dispatch_domain().dimension_constant(*extent),
                Dimension::Dynamic(_) => {
                    // A common dynamic dimension always originates in at least one input. Read its runtime size
                    // instead of replacing the dimension by a bound or a metadata-only shape.
                    let (input, axis) = inputs
                        .iter()
                        .zip(&shapes)
                        .find_map(|(input, shape)| {
                            shape
                                .dimensions()
                                .iter()
                                .position(|candidate| candidate == dimension)
                                .map(|axis| (input, axis))
                        })
                        .unwrap();
                    input.dimension_size(axis)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        inputs.iter().map(|input| input.dynamic_broadcast_to(&output_dimensions)).collect()
    }
}

impl<A: Value<Type = ArrayType> + DimensionSize<usize> + Broadcast> DynamicBroadcast for ArrayIrValue<A> {
    fn dynamic_broadcast_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_axes: &[usize],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let mut refinements = ArrayTypeRefinements::default();
        let output_shape = Shape::new(
            output_dimensions
                .iter()
                .map(<Self as ValueProjection<DimensionType>>::projected)
                .map(|result| {
                    let dimension = result?;
                    refinements.bind(dimension.r#type().variable(), dimension.extent())?;
                    Ok(Dimension::Static(dimension.extent()))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?,
        );
        let operation = DynamicBroadcastOperation::new(output_axes.to_vec()).with_output_sharding(output_sharding);
        let output_type = infer_explicit_broadcast_output_type(input.r#type().as_ref(), output_shape, &operation)?;
        Ok(Self::Array(input.broadcast(output_type, output_axes)?))
    }
}

impl<V: Value<Type = ArrayIrType>> DynamicBroadcast for V
where
    V::DispatchDomain: Context<Type = ArrayIrType, Operation: From<DynamicBroadcastOperation>>,
{
    #[inline]
    fn dynamic_broadcast_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_axes: &[usize],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let output_shape =
            Shape::new(ArrayIrType::extents(output_dimensions.iter().map(|dimension| dimension.r#type()))?);
        let operation = DynamicBroadcastOperation::new(output_axes.to_vec()).with_output_sharding(output_sharding);
        let output_type = infer_explicit_broadcast_output_type(input_type, output_shape, &operation)?;
        if input_type == &output_type && output_axes.iter().copied().eq(0..input_type.rank()) {
            return Ok(self.clone());
        }

        let mut inputs = Vec::with_capacity(output_dimensions.len() + 1);
        inputs.push(self.clone());
        inputs.extend_from_slice(output_dimensions);
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Infers the result of the canonical mixed broadcast from its explicit output extent types.
fn infer_explicit_broadcast_output_type(
    input: &ArrayType,
    output_shape: Shape,
    operation: &DynamicBroadcastOperation,
) -> Result<ArrayType, TypeError> {
    let output_rank = output_shape.rank();
    if operation.output_axes().len() != input.rank() {
        return Err(TypeError::invalid(format!(
            "broadcasting output axes has length {} but input has rank {}",
            operation.output_axes().len(),
            input.rank(),
        )));
    }

    let mut mapped_output_axes = vec![false; output_rank];
    for (input_axis, &output_axis) in operation.output_axes().iter().enumerate() {
        if output_axis >= output_rank {
            return Err(TypeError::invalid(format!(
                "broadcasting `output_axes[{}] = {}` is out of bounds for output rank {}",
                input_axis, output_axis, output_rank,
            )));
        }
        if mapped_output_axes[output_axis] {
            return Err(TypeError::invalid(format!(
                "broadcasting output axes map two input axes to output axis {output_axis}",
            )));
        }
        mapped_output_axes[output_axis] = true;

        let input_dimension = input.dimension(input_axis);
        let output_dimension = output_shape.dimensions()[output_axis].clone();
        match (input_dimension, output_dimension) {
            // Equal axes preserve their extent, including the identity of an equal dynamic extent.
            (input_dimension, output_dimension) if input_dimension == output_dimension => {}
            // The explicit output extent operand supplies the runtime replication count that the metadata-only
            // homogeneous operation lacks, so a unit input may expand to either a static or dynamic extent.
            (Dimension::Static(1), _) => {}
            (Dimension::Static(input_extent), Dimension::Static(output_extent)) => {
                return Err(TypeError::invalid(format!(
                    "broadcasting input axis {} has size {}, which is neither {} nor 1",
                    input_axis, input_extent, output_extent,
                )));
            }
            (input_dimension, output_dimension) => {
                return Err(TypeError::invalid(format!(
                    "broadcasting input axis {input_axis} has size {input_dimension} but the output has size \
                        {output_dimension}; a dynamic dimension only broadcasts to an identical dynamic dimension",
                )));
            }
        }
    }

    let output_sharding = match operation.output_sharding() {
        Some(sharding) => Some(sharding.clone()),
        None => input
            .sharding()
            .map(|sharding| sharding.with_broadcasted_dimensions(output_rank, operation.output_axes()))
            .transpose()
            .map_err(|error| TypeError::invalid(error.to_string()))?,
    };

    let mut output_type =
        if input.shape() == &output_shape && operation.output_axes().iter().copied().eq(0..input.rank()) {
            input.clone()
        } else {
            ArrayType::new(input.data_type(), output_shape).with_memory(input.memory())
        }
        .with_sharding(output_sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    if let Some(layout) = operation.output_layout() {
        output_type = output_type.with_layout(layout.clone());
    }
    validate_broadcast_output_layout(&output_type)?;
    Ok(output_type)
}

/// Validates broadcast storage metadata using known lower bounds for dynamic extents. Runtime extents can expose
/// additional stride aliasing or storage overflow, which interpretation validates against the resolved geometry.
fn validate_broadcast_output_layout(output_type: &ArrayType) -> Result<(), TypeError> {
    if output_type.layout().is_some() {
        let dimensions = output_type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| match dimension {
                Dimension::Static(extent) => *extent,
                Dimension::Dynamic(variable) => variable.bounds().lower(),
            })
            .collect::<Vec<_>>();
        ArrayAddressing::new(
            output_type.clone().with_shape(Shape::new(dimensions.into_iter().map(Dimension::Static).collect())),
        )
        .map_err(|error| match error {
            ProgramError::Type(error) => error,
            error => TypeError::invalid(error.to_string()),
        })?;
    }
    Ok(())
}

impl Broadcast for Array {
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        let r#type = self.r#type().broadcast(output_type, output_axes)?;
        let Some(target_shape) = r#type.static_shape() else {
            return Err(
                TypeError::invalid(format!("cannot materialize a value of dynamically sized type {}", r#type)).into()
            );
        };
        if &r#type == self.r#type().as_ref() && output_axes.iter().copied().eq(0..r#type.rank()) {
            return Ok(self.clone());
        }
        let input_shape = self.r#type().static_shape().unwrap();
        let input_rank = input_shape.rank();
        let target_rank = target_shape.rank();
        let output_count = Self::materialized_element_count(&r#type)?;
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(r#type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        // Structural-zero elements need no storage, regardless of the logical replication count.
        if bytes.is_empty() {
            return Ok(Self::new_unchecked(r#type, Arc::new(bytes)));
        }
        let mut target_index = vec![0usize; target_rank];
        let mut input_index = vec![0usize; input_rank];
        for output_flat in 0..output_count {
            for input_axis in 0..input_rank {
                let target_axis = output_axes[input_axis];
                input_index[input_axis] = if input_shape[input_axis] == 1 { 0 } else { target_index[target_axis] };
            }
            bytes[output_addressing.byte_range_for_flat_index(output_flat)]
                .copy_from_slice(&self.storage_bytes()[input_addressing.byte_range_unchecked(&input_index)]);
            output_addressing.advance_index(&mut target_index);
        }
        Ok(Self::new_unchecked(r#type, Arc::new(bytes)))
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::fmt::Debug;
    use std::marker::PhantomData;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, DimensionError,
        DimensionValue, DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, Sharding,
        ShardingDimension, StridedLayout, f8e8m0fnu,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{TransposableOperation, differentiate_at};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::{Parameter, Placeholder};
    use crate::partial::PartialValue;
    use crate::programs::{
        BindingRegionDriver, EffectClasses, EmptyRegionDriver, ProgramBuilder, ProgramError, Provenance,
        ProvenanceScope, Typed,
    };
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_broadcast() {
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), BROADCAST_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "broadcast [output_type=f64[2, 3], output_axes=[1]]");
        assert_eq!(*operation.output_type(), output_type);
        assert_eq!(operation.output_axes(), &[1]);

        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        // Program rendering uses the canonical operation name and includes the captured metadata.
        let mut builder = ProgramBuilder::<Array, BroadcastOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output =
            builder.add_instruction(operation.clone(), Vec::new(), vec![program_input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[1]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_broadcast_type_inference() {
        // Singleton symbolic extents retain their identity when restoring a refined cotangent signature.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(5)).unwrap());
        let output = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        assert_eq!(
            BroadcastOperation::new(output.clone(), vec![0])
                .infer_output_types(&[ArrayType::new_static(DataType::F64, [4])], &[]),
            Ok(vec![output]),
        );

        // Broadcasting is regionless even when its input and output metadata are otherwise valid.
        assert_eq!(
            BroadcastOperation::new(ArrayType::scalar(DataType::F32), vec![]).infer_output_types(
                &[ArrayType::scalar(DataType::F32)],
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
        assert_eq!(
            BroadcastOperation::new(
                ArrayType::new_static(DataType::F32, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![2]))),
                vec![],
            )
            .infer_output_types(&[ArrayType::scalar(DataType::F32)], &[]),
            Err(TypeError::invalid(
                "strided layout stride 2 on axis 0 is smaller than the 4-byte span occupied by more minor axes and may alias array elements",
            )),
        );
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);
        // Type inference validates the axis mapping and returns the target type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
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
                    input_types = [ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))],
                    error = "broadcasting input data type `f32` does not match output data type `f64`",
                },
                {
                    input_types = [output_type.clone()],
                    error = "broadcasting output axes has length 1 but input has rank 2",
                },
                {
                    input_types = [ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))],
                    error = "broadcasting input axis 0 has size 2, which is neither 3 nor 1",
                },
            ],
        );
    }

    #[test]
    fn test_array_type_broadcast() {
        // Type-level broadcasting validates arbitrary mappings without consuming the input type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(input_type.broadcast(output_type.clone(), &[1]), Ok(output_type.clone()));
        assert_eq!(
            input_type.broadcast(output_type.clone().with_memory(Memory::Host { pinned: true }), &[1]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input memory Device does not match output memory Host[Pinned]".to_string(),
            ))),
        );
        assert_eq!(
            input_type.broadcast(output_type.clone(), &[2]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting `output_axes[0] = 2` is out of bounds for output rank 2".to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(3)]))
                .broadcast(output_type, &[1, 1]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting output axes map two input axes to output axis 1".to_string(),
            ))),
        );

        // A dynamic input dimension maps only to the identical dynamic dimension, while replicated output axes must
        // remain static because their replication counts need to be known.
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(3)]));
        let output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2), Dimension::Static(3)]),
        );
        assert_eq!(input_type.broadcast(output_type.clone(), &[0, 2]), Ok(output_type));

        let unbounded = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("input", DimensionBounds::unbounded()))]),
        );
        assert_eq!(
            unbounded.broadcast(
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                        "output",
                        DimensionBounds::non_negative(Some(4)).unwrap(),
                    ))]),
                ),
                &[0],
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input axis 0 has size input but the output has size output; a dynamic dimension \
                    only broadcasts to an identical dynamic dimension"
                    .to_string(),
            ))),
        );
        assert_eq!(
            unbounded.broadcast(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])), &[0],),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input axis 0 has size input but the output has size 3; a dynamic dimension \
                    only broadcasts to an identical dynamic dimension"
                    .to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).broadcast(unbounded.clone(), &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting cannot expand input axis 0 of size 1 into dynamic output size input".to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).broadcast(unbounded, &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input axis 0 has size 3 but the output has size input; a dynamic dimension \
                    only broadcasts to an identical dynamic dimension"
                    .to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).broadcast(
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![
                        Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                        Dimension::Static(3),
                    ]),
                ),
                &[1],
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting cannot replicate the input into unmapped dynamic output axis 0 of size dynamic"
                    .to_string(),
            ))),
        );
    }

    #[test]
    fn test_broadcast_interpretation() {
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);
        // Interpretation replicates the payload along the added axis.
        let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        let output = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // Invalid interpreter arity reports the exact program error.
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
    fn test_broadcast_interpretation_refines_mapped_dynamic_dimensions() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(3)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.clone().into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let operation = BroadcastOperation::new(output_type.clone(), vec![0]);
        let context = EagerContext::<Array>::new();
        for elements in [&[1f32][..], &[1f32, 2.][..]] {
            let input = Array::from_elements(ArrayType::new_static(DataType::F32, [elements.len()]), elements).unwrap();
            assert_eq!(
                operation.interpret(&context, &EmptyRegionDriver, &[input]),
                Ok(vec![
                    Array::from_elements(
                        output_type.clone().with_shape(Shape::new(vec![elements.len().into()])),
                        elements,
                    )
                    .unwrap()
                ]),
            );
        }

        // The refined dimensions must respect their declared bounds and repeated-identity equality.
        let input = Array::from_elements(ArrayType::new_static(DataType::F32, [3]), &[1f32, 2., 3.]).unwrap();
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[input]),
            Err(ProgramError::Type(
                DimensionError::BindingOutOfBounds {
                    variable: "extent".to_string(),
                    value: 3,
                    bounds: extent.bounds(),
                }
                .into()
            )),
        );
        let repeated_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.clone().into(), extent.into()]));
        let input = Array::from_elements(ArrayType::new_static(DataType::F32, [1, 2]), &[1f32, 2.]).unwrap();
        assert_eq!(
            BroadcastOperation::new(repeated_type, vec![0, 1]).interpret(&context, &EmptyRegionDriver, &[input]),
            Err(ProgramError::Type(
                DimensionError::InputDimensionMismatch { dimension: "extent".to_string(), expected: 1, actual: 2 }
                    .into()
            )),
        );
    }

    #[test]
    fn test_broadcast_interpretation_axis_mappings_and_metadata() {
        // Arbitrary axis mappings replicate the payload along every unmapped target axis.
        let target = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(2)]),
        );
        let output = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap().broadcast(target, &[0, 2]).unwrap();
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);

        // Static unit axes stretch to the target extent, and empty target dimensions produce empty payloads.
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let output = Array::matrix(1, 3, vec![1.0, 2.0, 3.0]).unwrap().broadcast(target, &[0, 1]).unwrap();
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0), Dimension::Static(2)]));
        let output = Array::vector(vec![1.0, 2.0]).unwrap().broadcast(target, &[1]).unwrap();
        assert_eq!(output.to_f64s(), Vec::<f64>::new());

        // The eager backend primitive preserves leading and right-aligned axis mappings.
        let output = Array::vector(vec![1.0, 2.0, 3.0])
            .unwrap()
            .broadcast(ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])), &[1])
            .unwrap();
        assert_eq!(*output.r#type(), ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])));
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let output = Array::scalar(7.0)
            .unwrap()
            .broadcast(ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])), &[])
            .unwrap();
        assert_eq!(output.to_f64s(), vec![7.0; 6]);
        let output = Array::vector(vec![10.0, 20.0, 30.0])
            .unwrap()
            .broadcast(ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])), &[1])
            .unwrap();
        assert_eq!(output.to_f64s(), vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
        assert_eq!(
            Array::scalar(1.0)
                .unwrap()
                .broadcast(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)])), &[],),
            Ok(Array::from_elements::<f64>(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)])), &[])
                .unwrap()),
        );

        // Explicit output types carry placement metadata; an exact identity preserves the complete type.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        let input = Array::from_elements::<f64>(input_type.clone(), &[1.0, 2.0, 3.0]).unwrap();
        let identity = input.broadcast(input_type.clone(), &[0]).unwrap();
        assert_eq!(*identity.r#type(), input_type);
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                    .unwrap(),
            )
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        let output = input.broadcast(output_type, &[1]).unwrap();
        assert_eq!(output.r#type().memory(), Memory::Host { pinned: true });
        assert_eq!(output.r#type().layout(), None);
        assert_eq!(
            output.r#type().sharding(),
            Some(
                &Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],)
                    .unwrap(),
            ),
        );

        // Oversized target shapes fail through checked element-count arithmetic instead of panicking or wrapping.
        assert_eq!(
            Array::scalar(1.0).unwrap().broadcast(
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2)]),),
                &[],
            ),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "shape [{}, 2] element count does not fit in usize",
                usize::MAX,
            )))),
        );
    }

    #[test]
    fn test_broadcast_partial_evaluation() {
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);
        // Check standard partial evaluation with known and residual operands.
        let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        let expected = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = operation.clone(),
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
    fn test_broadcast_batching() {
        // Batching preserves replicated inputs and lifts mapped inputs through the explicit output-axis mapping.
        check_operation_batching!(
            @exact,
            operation = BroadcastOperation::new(
                ArrayType::new(DataType::F64, Shape::new(vec![3.into(), 4.into()])),
                vec![0],
            ),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@replicated, Array::vector(vec![1.0, 2.0, 3.0]).unwrap())],
                    outputs = [(@replicated, Array::matrix(
                        3,
                        4,
                        vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                    ).unwrap())],
                },
                {
                    inputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        3,
                        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    ).unwrap())],
                    outputs = [(@mapped(axis = 0), Array::from_elements::<f64>(
                        ArrayType::new(
                            DataType::F64,
                            Shape::new(vec![2.into(), 3.into(), 4.into()]),
                        ),
                        &[
                            1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
                            3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0,
                            6.0, 6.0,
                        ],
                    ).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_broadcast_batching_ragged() {
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = Array::from_elements(ArrayType::new_static(DataType::I64, [2]), &[1i64, 3]).unwrap();
        let input = ArrayBatch::new(
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32, 0., 0., 2., 3., 4.]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap()
        .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), size.clone(), vec![0])])
        .unwrap();
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![2.into(), size.clone().into()]));
        let output = BroadcastOperation::new(output_type.clone(), vec![1])
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(output.unbatched_type(), output_type);
        assert_eq!(output.ragged_axes(), &[RaggedAxis::new(2, extents.clone(), size.clone(), vec![0])]);
        assert_eq!(
            output.into_value(),
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2, 3]),
                &[1f32, 0., 0., 1., 0., 0., 2., 3., 4., 2., 3., 4.],
            )
            .unwrap()
        );

        // A nonleading mapped axis changes both the ragged-axis position and the extent-axis mapping.
        let input = ArrayBatch::new(
            Array::from_elements(ArrayType::new_static(DataType::F32, [3, 2]), &[1f32, 2., 0., 3., 0., 4.]).unwrap(),
            BatchAxis::new(1),
        )
        .unwrap()
        .with_ragged_axes(vec![RaggedAxis::new(0, extents.clone(), size.clone(), vec![1])])
        .unwrap();
        let output = BroadcastOperation::new(output_type, vec![1])
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(output.batch_axis(), BatchAxis::new(1));
        assert_eq!(output.ragged_axes(), &[RaggedAxis::new(2, extents.clone(), size.clone(), vec![1])]);
        assert_eq!(
            output.into_value(),
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2, 3]),
                &[1f32, 0., 0., 2., 3., 4., 1., 0., 0., 2., 3., 4.],
            )
            .unwrap()
        );

        // Replicated ragged arrays retain their metadata through an arbitrary axis permutation.
        let input = ArrayBatch::replicated(
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32, 0., 0., 2., 3., 4.]).unwrap(),
        )
        .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), size.clone(), vec![0])])
        .unwrap();
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![size.clone().into(), 2.into()]));
        let output = BroadcastOperation::new(output_type.clone(), vec![1, 0])
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(output.batch_axis(), BatchAxis::replicated());
        assert_eq!(output.unbatched_type(), output_type);
        assert_eq!(output.ragged_axes(), &[RaggedAxis::new(0, extents, size, vec![1])]);
        assert_eq!(
            output.into_value(),
            Array::from_elements(ArrayType::new_static(DataType::F32, [3, 2]), &[1f32, 2., 0., 3., 0., 4.],).unwrap()
        );
    }

    #[test]
    fn test_broadcast_batching_sharding() {
        // Explicit mapped-axis sharding remains attached to the lifted batch dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let logical_output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(4)]))
                .with_sharding(Sharding::replicated(mesh.clone(), 2))
                .unwrap();
        let expected_output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
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
        .unwrap();

        check_operation_batching!(
            @exact,
            operation = BroadcastOperation::new(logical_output_type, vec![0]),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::from_elements::<f64>(
                    input_type,
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ).unwrap())],
                outputs = [(@mapped(axis = 0), Array::from_elements::<f64>(expected_output_type, &[
                        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                        4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0,
                    ]).unwrap())],
            }],
        );

        // A mapped physical axis retains its mesh placement even when the logical operation result has no sharding.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let expected_output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
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
        .unwrap();
        check_operation_batching!(
            @exact,
            operation = BroadcastOperation::new(
                ArrayType::new(DataType::F64, Shape::new(vec![3.into(), 4.into()])),
                vec![0],
            ),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::from_elements::<f64>(
                    input_type,
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ).unwrap())],
                outputs = [(@mapped(axis = 0), Array::from_elements::<f64>(expected_output_type, &[
                        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                        4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0,
                    ]).unwrap())],
            }],
        );

        // A non-monotonic axis mapping keeps an explicitly sharded mapped axis at the beginning, middle, and end of
        // the physical result, even though the logical output itself has no sharding annotation.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        for batch_axis in 0..3 {
            let mut input_dimensions = vec![Dimension::Static(2), Dimension::Static(3)];
            input_dimensions.insert(batch_axis, Dimension::Static(2));
            let mut input_sharding = vec![ShardingDimension::replicated(); 3];
            input_sharding[batch_axis] = ShardingDimension::sharded(["x"]);
            let input_type = ArrayType::new(DataType::F64, Shape::new(input_dimensions))
                .with_sharding(Sharding::new(mesh.clone(), input_sharding).unwrap())
                .unwrap();
            let input = ArrayBatch::new(
                Array::from_elements::<f64>(input_type, &[0.0; 12]).unwrap(),
                BatchAxis::from_position(batch_axis),
            )
            .unwrap();

            let output = BroadcastOperation::new(
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
                vec![1, 0],
            )
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
            let mut output_dimensions = vec![Dimension::Static(3), Dimension::Static(2)];
            output_dimensions.insert(batch_axis, Dimension::Static(2));
            let mut output_sharding = vec![ShardingDimension::replicated(); 3];
            output_sharding[batch_axis] = ShardingDimension::sharded(["x"]);
            assert_eq!(
                output.r#type().as_ref(),
                &ArrayType::new(DataType::F64, Shape::new(output_dimensions))
                    .with_sharding(Sharding::new(mesh.clone(), output_sharding).unwrap())
                    .unwrap(),
            );
            assert_eq!(output.batch_axis(), BatchAxis::from_position(batch_axis));
        }
    }

    #[test]
    fn test_broadcast_differentiation() {
        // Broadcasting is structural-linear meaning that its JVP broadcasts both the primal and its tangent.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = BroadcastOperation::new(
                ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into()])),
                vec![1],
            ),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0]).unwrap()],
                tangents = [Array::vector(vec![3.0, 4.0]).unwrap()],
                primal_outputs = [Array::matrix(2, 2, vec![1.0, 2.0, 1.0, 2.0]).unwrap()],
                tangent_outputs = [Array::matrix(2, 2, vec![3.0, 4.0, 3.0, 4.0]).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[2], %1:f64[2] .
                    let %2:f64[2, 2] = broadcast [output_type=f64[2, 2], output_axes=[1]] %0
                        %3:f64[2, 2] = broadcast [output_type=f64[2, 2], output_axes=[1]] %1
                    in (%2, %3)
                "},
            }],
        );
    }

    #[test]
    fn test_broadcast_differentiation_promoted_tangents() {
        // Differentiation derives the tangent target from the primal output, preserving promoted tangent types.
        let primal_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2)]));
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let primal_output_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let tangent_output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let (primal, tangent) = differentiate_at(
            Array::from_elements::<f8e8m0fnu>(
                primal_type,
                &[2.0, 4.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
            )
            .unwrap(),
        )
        .jvp(Array::from_elements::<f32>(tangent_type, &[1.0, 3.0]).unwrap(), |value| {
            value.broadcast(primal_output_type.clone(), &[1])
        })
        .unwrap();
        assert_eq!(primal.r#type().as_ref(), &primal_output_type);
        assert_eq!(tangent.r#type().as_ref(), &tangent_output_type);
        assert_eq!(tangent.to_f64s(), vec![1.0, 3.0, 1.0, 3.0]);
    }

    #[test]
    fn test_broadcast_transposition() {
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);
        // The pullback sums the output cotangent over every newly replicated output axis.
        check_operation_transposition!(
            @exact,
            operation = operation,
            cases = [{
                inputs = [(@linear(type = ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Static(3)]),
                )))],
                output_cotangents = [Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()],
                input_cotangents = [Array::vector(vec![5.0, 7.0, 9.0]).unwrap()],
                pullback = indoc! {"
                    lambda %0:f64[2, 3] .
                    let %1:f64[3] = reduce_sum [axes=[0]] %0
                    in (%1)
                "},
            }],
        );
    }

    #[test]
    fn test_broadcast_transposition_permuted_axes() {
        // Input axis 0 (size 2) maps to output axis 2 and input axis 1 (size 3) maps to output axis 0, so the pullback
        // must sum over output axis 1 and swap the surviving axes back into input order.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(3), Dimension::Static(4), Dimension::Static(2)]),
        );
        check_operation_transposition!(
            @exact,
            operation = BroadcastOperation::new(output_type.clone(), vec![2, 0]),
            cases = [{
                inputs = [(@linear(type = input_type))],
                output_cotangents = [Array::from_elements::<f64>(
                    output_type,
                    &(0..24).map(|value| value as f64).collect::<Vec<_>>(),
                ).unwrap()],
                input_cotangents = [Array::matrix(2, 3, vec![12.0, 44.0, 76.0, 16.0, 48.0, 80.0]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_broadcast_transposition_expanded_unit_axis() {
        // Input axis 0 has extent 1 stretched to 2 in the target, so the pullback sums over it and restores the unit
        // axis with a reshape.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        check_operation_transposition!(
            @exact,
            operation = BroadcastOperation::new(output_type, vec![0, 1]),
            cases = [{
                inputs = [(@linear(type = input_type))],
                output_cotangents = [Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()],
                input_cotangents = [Array::matrix(1, 3, vec![5.0, 7.0, 9.0]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_broadcast_transposition_symbolic_zero() {
        // Symbolic-zero cotangents remain symbolic and acquire the input's promoted cotangent type.
        let input_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(3)]));
        let output_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);
        let input_cotangent_type = input_type.cotangent().unwrap();
        let output_cotangent_type = output_type.cotangent().unwrap();

        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let contributions = {
            let mut rule_context = TranspositionContext::new(context.clone());
            let rule_inputs = &[PartialValue::Unknown(input_type)];
            let accumulators = rule_context.cotangent_accumulators(rule_inputs, &[]).unwrap();
            operation
                .transpose(
                    &mut rule_context,
                    &EmptyRegionDriver,
                    rule_inputs,
                    &[MaybeZero::Zero(output_cotangent_type)],
                    &accumulators,
                )
                .unwrap();
            rule_context.take_cotangents(&accumulators).unwrap()
        };
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &input_cotangent_type);
    }

    #[test]
    fn test_broadcast_transposition_integer_input() {
        // Inputs with no cotangent space receive the structural zero of that space rather than being rejected.
        let input_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let contributions = {
            let mut rule_context = TranspositionContext::new(context.clone());
            let rule_inputs = &[PartialValue::Unknown(input_type.clone())];
            let accumulators = rule_context.cotangent_accumulators(rule_inputs, &[]).unwrap();
            operation
                .transpose(
                    &mut rule_context,
                    &EmptyRegionDriver,
                    rule_inputs,
                    &[MaybeZero::Zero(output_type.cotangent().unwrap())],
                    &accumulators,
                )
                .unwrap();
            rule_context.take_cotangents(&accumulators).unwrap()
        };
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &input_type.cotangent().unwrap());
    }

    #[test]
    fn test_broadcast_invalid_output_count() {
        /// Value wrapper that dispatches through a deliberately malformed context in either array language.
        #[derive(Clone, Debug)]
        struct DispatchValue<V: Value, O: Clone + Debug> {
            value: V,
            output_count: usize,
            operation: PhantomData<O>,
        }

        impl<V: Value, O: Clone + Debug> Display for DispatchValue<V, O> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "{}", self.value)
            }
        }

        impl<V: Value, O: Clone + Debug> Parameter for DispatchValue<V, O> {}

        impl<V: Value, O: Clone + Debug> Typed for DispatchValue<V, O> {
            type Type = V::Type;

            fn r#type(&self) -> Cow<'_, Self::Type> {
                self.value.r#type()
            }
        }

        impl<V: Value, O: Debug + Operation<Type = V::Type>> Value for DispatchValue<V, O> {
            type DispatchDomain = InvalidOutputContext<V, O>;
            type ExecutionDomain = InvalidOutputContext<V, O>;

            fn dispatch_domain(&self) -> Self::DispatchDomain {
                InvalidOutputContext(self.output_count, PhantomData)
            }

            fn execution_domain(&self) -> Self::ExecutionDomain {
                self.dispatch_domain()
            }
        }

        /// Context that accepts valid broadcast inputs but violates the single-output contract.
        #[derive(Clone)]
        struct InvalidOutputContext<V, O>(usize, PhantomData<(V, O)>);

        impl<V: Value, O: Debug + Operation<Type = V::Type>> Domain for InvalidOutputContext<V, O> {
            type Type = V::Type;
            type Value = DispatchValue<V, O>;
            type Constant = V;
            type Operation = O;
        }

        impl<V: Value, O: Debug + Operation<Type = V::Type>> Context for InvalidOutputContext<V, O> {
            fn lift(&self, value: V) -> Result<Self::Value, ProgramError> {
                Ok(DispatchValue { value, output_count: self.0, operation: PhantomData })
            }

            fn bind<Operation: Into<O>, D: BindingRegionDriver<V, O>>(
                &self,
                _operation: Operation,
                _driver: D,
                inputs: &[Self::Value],
            ) -> Result<Vec<Self::Value>, ProgramError> {
                Ok(vec![inputs[0].clone(); self.0])
            }

            fn is_eager(&self) -> bool {
                true
            }

            fn provenance(&self) -> Provenance {
                Provenance::unknown()
            }

            fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, _origin: Provenance, function: F) -> R {
                function()
            }

            fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, _scope: ProvenanceScope, function: F) -> R {
                function()
            }
        }

        let array = Array::from_elements(ArrayType::scalar(DataType::I32), &[7_i32]).unwrap();
        // Exercise both missing and extra results without involving larger operation families.
        for output_count in [0, 2] {
            let context = InvalidOutputContext::<Array, BroadcastOperation>(output_count, PhantomData);
            let input = context.lift(array.clone()).unwrap();
            assert!(matches!(
                input.broadcast(ArrayType::new_static(DataType::I32, [2]), &[]),
                Err(ProgramError::InvalidOutputCount { expected: 1, actual }) if actual == output_count,
            ));
            let context =
                InvalidOutputContext::<ArrayIrValue<Array>, DynamicBroadcastOperation>(output_count, PhantomData);
            let input = context.lift(ArrayIrValue::Array(array.clone())).unwrap();
            let extent = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
            assert!(matches!(
                input.dynamic_broadcast(&[extent], &[]),
                Err(ProgramError::InvalidOutputCount { expected: 1, actual }) if actual == output_count,
            ));
        }
    }

    #[test]
    fn test_broadcast_broadcast_to() {
        let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            input.broadcast_to(Shape::new(vec![2.into(), 3.into()])),
            Ok(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap()),
        );
        assert_eq!(
            Array::scalar(7.0).unwrap().broadcast_to(Shape::new(vec![2.into(), 3.into()])),
            Ok(Array::matrix(2, 3, vec![7.0; 6]).unwrap()),
        );
        assert_eq!(input.broadcast_to(Shape::new(vec![3.into()])), Ok(input.clone()));
        assert_eq!(
            input.broadcast_to(Shape::new(vec![0.into(), 3.into()])),
            Ok(Array::matrix(0, 3, Vec::<f64>::new()).unwrap()),
        );
    }

    #[test]
    fn test_broadcast_broadcast_to_incompatible_shape() {
        let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            input.broadcast_to(Shape::new(vec![2.into()])),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input axis 0 has size 3, which is neither 2 nor 1",
            ))),
        );
        assert_eq!(
            input.broadcast_to(Shape::new(Vec::new())),
            Err(ProgramError::Type(TypeError::invalid("cannot broadcast rank-1 input to 0 output dimensions"))),
        );
    }

    #[test]
    fn test_broadcast_broadcast_to_metadata() {
        // Shape-preserving calls retain the physical layout; inserted axes project the sharding and retain memory.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        let input = Array::from_elements::<f64>(input_type.clone(), &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(input.broadcast_to(Shape::new(vec![3.into()])).unwrap().r#type().as_ref(), &input_type);
        let expected_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]).unwrap(),
            )
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(input.broadcast_to(Shape::new(vec![2.into(), 3.into()])).unwrap().r#type().as_ref(), &expected_type);
    }

    #[test]
    fn test_broadcast_broadcast_to_with_output_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let input = ArrayType::new_static(DataType::I32, [2]);
        let expected = input.clone().with_sharding(sharding.clone()).unwrap();
        assert_eq!(
            input.broadcast_to_with_output_sharding(Shape::new(vec![2.into()]), Some(sharding)),
            Ok(expected.clone())
        );
        assert_eq!(expected.broadcast_to_with_output_sharding(Shape::new(vec![2.into()]), None), Ok(expected));
        assert_eq!(
            input.broadcast_to_with_output_sharding(Shape::new(vec![3.into()]), None),
            Err(TypeError::invalid("broadcasting input axis 0 has size 2, which is neither 3 nor 1",).into())
        );
        assert_eq!(
            input.broadcast_to_with_output_sharding(Shape::new(vec![2.into()]), Some(Sharding::replicated(mesh, 2))),
            Err(TypeError::invalid("sharding rank (2) does not match array rank (1)").into()),
        );
    }

    #[test]
    fn test_broadcast_broadcast_leading() {
        let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(input.broadcast_leading([2]), Ok(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap()));
        assert_eq!(input.broadcast_leading([0]), Ok(Array::matrix(0, 3, Vec::<f64>::new()).unwrap()));
        assert_eq!(input.broadcast_leading([]), Ok(input));
        assert_eq!(
            Array::scalar(4.0).unwrap().broadcast_leading([2, 2]),
            Ok(Array::matrix(2, 2, vec![4.0; 4]).unwrap())
        );
    }

    #[test]
    fn test_broadcast_broadcast_leading_preserves_dynamic_dimensions() {
        // Leading replication has a static count while the existing input extent keeps its symbolic identity.
        let extent = DimensionVariable::new("extent", DimensionBounds::unbounded());
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone())])));
        let output = input.broadcast_leading([2]).unwrap();
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(extent)])),
        );
        assert_eq!(input.broadcast_leading([]).unwrap().atom_id(), input.atom_id());
    }

    #[test]
    fn test_broadcast_broadcast_leading_with_output_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]).unwrap();
        let input = Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[3_i32, 7]).unwrap();
        let output_type = ArrayType::new_static(DataType::I32, [2, 2]).with_sharding(sharding.clone()).unwrap();
        assert_eq!(
            input.broadcast_leading_with_output_sharding([2], Some(sharding)),
            Ok(Array::from_elements(output_type, &[3_i32, 7, 3, 7]).unwrap()),
        );
    }

    #[test]
    fn test_broadcast_broadcast_arrays() {
        let integers = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 1]), &[1_i32, 2]).unwrap();
        let floats = Array::from_elements(ArrayType::new_static(DataType::F32, [3]), &[3_f32, 4.0, 5.0]).unwrap();
        // Common shape inference must not promote either payload to the other's element type.
        assert_eq!(
            Array::broadcast_arrays(&[integers.clone(), floats]),
            Ok(vec![
                Array::from_elements(ArrayType::new_static(DataType::I32, [2, 3]), &[1_i32, 1, 1, 2, 2, 2]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[3_f32, 4.0, 5.0, 3.0, 4.0, 5.0])
                    .unwrap(),
            ]),
        );
        assert_eq!(Array::broadcast_arrays(&[]), Ok(Vec::new()));
        assert_eq!(Array::broadcast_arrays(std::slice::from_ref(&integers)), Ok(vec![integers]));
        let scalar = Array::from_elements(ArrayType::scalar(DataType::I32), &[7_i32]).unwrap();
        let empty = Array::from_elements(ArrayType::new_static(DataType::F32, [0]), &[] as &[f32]).unwrap();
        assert_eq!(
            Array::broadcast_arrays(&[scalar, empty.clone()]),
            Ok(vec![Array::from_elements(ArrayType::new_static(DataType::I32, [0]), &[] as &[i32]).unwrap(), empty]),
        );
        let shapes = [ArrayType::new_static(DataType::I32, [2]), ArrayType::new_static(DataType::F32, [3])];
        assert_eq!(
            ArrayType::broadcast_arrays(&shapes),
            Err(TypeError::invalid("failed to broadcast shape `[2]` to shape `[3]`").into()),
        );
    }

    #[test]
    fn test_dynamic_broadcast() {
        let operation = DynamicBroadcastOperation::new(vec![1]);
        assert_eq!(operation.name(), BROADCAST_OPERATION_NAME);
        assert_eq!(operation.to_string(), "broadcast [output_axes=[1]]");

        assert_eq!(operation.output_axes(), &[1]);
        assert_eq!(operation.output_sharding(), None);
        assert_eq!(operation.output_layout(), None);
        let layout = Layout::Strided(StridedLayout::new(vec![24, 8]));
        assert_eq!(operation.with_output_layout(layout.clone()).output_layout(), Some(&layout));
    }

    #[test]
    fn test_dynamic_broadcast_type_inference() {
        let operation = DynamicBroadcastOperation::new(vec![1]);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_memory(Memory::Host { pinned: true });
        let two = DimensionValue::constant(2).unwrap();
        let dynamic_extent =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap()));
        assert_eq!(
            operation.infer_output_types(
                &[input_type.into(), two.r#type().into_owned().into(), dynamic_extent.clone().into(),],
                &[],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(dynamic_extent.variable().clone()),]),
                )
                .with_memory(Memory::Host { pinned: true })
                .into()
            ]),
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        let three = DimensionValue::constant(3).unwrap();
        assert_eq!(
            operation.infer_output_types(
                &[input_type.into(), two.r#type().into_owned().into(), three.r#type().into_owned().into()],
                &[],
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                    .with_sharding(
                        Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],)
                            .unwrap(),
                    )
                    .unwrap()
                    .with_memory(Memory::Host { pinned: true })
                    .into(),
            ]),
        );

        assert_eq!(
            DynamicBroadcastOperation::new(vec![0, 0]).infer_output_types(
                &[
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(1)]),).into(),
                    two.r#type().into_owned().into(),
                ],
                &[],
            ),
            Err(TypeError::invalid("broadcasting output axes map two input axes to output axis 0")),
        );
        assert_eq!(
            operation.infer_output_types(&[two.r#type().into_owned().into()], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::scalar(DataType::F32).into(), ArrayType::scalar(DataType::I64).into()],
                &[],
            ),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
    }

    #[test]
    fn test_dynamic_broadcast_type_inference_layout() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let layout = Layout::Strided(StridedLayout::new(vec![8]));
        let input_types = [ArrayType::scalar(DataType::F32).into(), DimensionType::new(extent.clone()).into()];
        assert_eq!(
            DynamicBroadcastOperation::new(vec![])
                .with_output_layout(layout.clone())
                .infer_output_types(&input_types, &[]),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]))
                    .with_layout(layout)
                    .into()
            ]),
        );
        assert_eq!(
            DynamicBroadcastOperation::new(vec![])
                .with_output_layout(Layout::Strided(StridedLayout::new(vec![8, 4])))
                .infer_output_types(&input_types, &[]),
            Err(TypeError::invalid("strided layout rank 2 does not match array rank 1")),
        );
        assert_eq!(
            DynamicBroadcastOperation::new(vec![])
                .with_output_layout(Layout::Tiled(TiledLayout::new(vec![1], vec![])))
                .infer_output_types(&input_types, &[]),
            Err(TypeError::invalid("tiled layout minor-to-major dimensions must be a permutation of 0..1")),
        );
    }

    #[test]
    fn test_dynamic_broadcast_type_inference_identity_instantiation() {
        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let extent = builder.add_input(DimensionType::new(source.clone()).into());
        let one = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()));
        let output = builder
            .add_instruction(DynamicBroadcastOperation::new(vec![1]), Vec::new(), vec![input, extent, one], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
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
        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = destination.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let extent = destination.add_input(DimensionType::new(target.clone()).into());
        let outputs = destination.splice_program(&instantiated, &[input, extent]).unwrap();
        let [instruction] = destination.instructions() else {
            panic!("expected the imported broadcast instruction");
        };
        assert_eq!(instruction.inputs()[..2], [input, extent]);
        assert_eq!(
            destination.atoms()[outputs[0].index()].r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(target), Dimension::Static(1)]),
            )),
        );
    }

    #[test]
    fn test_dynamic_broadcast_interpretation() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let inputs = [
            ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        ];
        assert_eq!(
            DynamicBroadcastOperation::new(vec![1]).interpret(&context, &EmptyRegionDriver, &inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap())]),
        );
        assert_eq!(
            DynamicBroadcastOperation::new(vec![1]).interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::Type(TypeError::invalid("`broadcast` expects an array followed by its output extents"))),
        );
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0]).unwrap());
        let first_extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let second_extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let expected_output = ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f64, 2.0, 1.0, 2.0, 1.0, 2.0]).unwrap());
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            context.bind(
                DynamicBroadcastOperation::new(vec![1]),
                Vec::new(),
                &[input.clone(), first_extent.clone(), second_extent.clone()],
            ),
            Ok(vec![expected_output.clone()]),
        );
        let eager_dynamic_type =
            DimensionType::new(DimensionVariable::new("eager_extent", DimensionBounds::new(1, Some(9)).unwrap()));
        assert_eq!(
            context.bind(
                DynamicBroadcastOperation::new(vec![1]),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::vector(vec![7.0_f64]).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::new(eager_dynamic_type, 3).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::matrix(3, 1, vec![7.0_f64, 7.0, 7.0]).unwrap())]),
        );
    }

    #[test]
    fn test_dynamic_broadcast_interpretation_layout() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let inputs = [
            ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        ];
        let layout = Layout::Strided(StridedLayout::new(vec![-8]));
        assert_eq!(
            DynamicBroadcastOperation::new(vec![]).with_output_layout(layout.clone()).interpret(
                &context,
                &EmptyRegionDriver,
                &inputs
            ),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new_static(DataType::F32, [3]).with_layout(layout),
                    &[2.0f32, 2.0, 2.0],
                )
                .unwrap()
            )]),
        );
        assert_eq!(
            DynamicBroadcastOperation::new(vec![])
                .with_output_layout(Layout::Strided(StridedLayout::new(vec![2])))
                .interpret(&context, &EmptyRegionDriver, &inputs),
            Err(ProgramError::Type(TypeError::invalid(
                "strided layout stride 2 on axis 0 is smaller than the 4-byte span occupied by more minor axes and may alias array elements",
            ))),
        );
    }

    #[test]
    fn test_dynamic_broadcast_interpretation_repeated_dimensions() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let dimension = DimensionType::new(DimensionVariable::new("size", DimensionBounds::unbounded()));
        let input = ArrayIrValue::Array(Array::scalar(1.0f32).unwrap());
        let two = ArrayIrValue::Dimension(DimensionValue::new(dimension.clone(), 2).unwrap());
        let three = ArrayIrValue::Dimension(DimensionValue::new(dimension, 3).unwrap());
        let operation = DynamicBroadcastOperation::new(Vec::new());
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[input.clone(), two.clone(), two.clone()]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [2, 2]), &[1.0f32; 4]).unwrap()
            )]),
        );
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[input.clone(), two.clone(), three]),
            Err(ProgramError::Type(
                DimensionError::InputDimensionMismatch { dimension: "size".to_owned(), expected: 2, actual: 3 }.into()
            )),
        );
        // Equal diagnostic names do not equate independently created dimension identities.
        let other = DimensionType::new(DimensionVariable::new("size", DimensionBounds::unbounded()));
        let three = ArrayIrValue::Dimension(DimensionValue::new(other, 3).unwrap());
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[input, two, three]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1.0f32; 6]).unwrap()
            )]),
        );
        let dimension = DimensionType::new(DimensionVariable::new("size", DimensionBounds::unbounded()));
        let input = ArrayIrValue::Array(Array::scalar(1.0f32).unwrap());
        let two = ArrayIrValue::Dimension(DimensionValue::new(dimension.clone(), 2).unwrap());
        let three = ArrayIrValue::Dimension(DimensionValue::new(dimension, 3).unwrap());
        assert_eq!(
            input.dynamic_broadcast_with_output_sharding(&[two.clone(), two.clone()], &[], None),
            Ok(ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [2, 2]), &[1.0f32; 4]).unwrap()
            )),
        );
        assert_eq!(
            input.dynamic_broadcast_with_output_sharding(&[two, three], &[], None),
            Err(ProgramError::Type(
                DimensionError::InputDimensionMismatch { dimension: "size".to_owned(), expected: 2, actual: 3 }.into()
            )),
        );
    }

    #[test]
    fn test_dynamic_broadcast_partial_evaluation() {
        let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let two = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let three = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let output = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap());
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = DynamicBroadcastOperation::new(vec![1]),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, two.clone()), (@known, three.clone())],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, two.clone()),
                        (@known, three.clone()),
                    ],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0]).unwrap());
        let first_extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let second_extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let expected_output = ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f64, 2.0, 1.0, 2.0, 1.0, 2.0]).unwrap());
        let input_type = input.r#type().into_owned();
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = DynamicBroadcastOperation::new(vec![1]),
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

        let identity_input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0]).unwrap());
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = DynamicBroadcastOperation::new(vec![0]),
            cases = [{
                inputs = [
                    (@unknown(type = identity_input.r#type().into_owned(), replay = identity_input.clone())),
                    (@known, second_extent.clone()),
                ],
                outputs = [(@residual, identity_input)],
                residual_instructions = 0,
            }],
        );
    }

    #[test]
    fn test_dynamic_broadcast_partial_evaluation_identity() {
        // Exact dimension operands make an identity independent of the unknown array payload.
        let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = DynamicBroadcastOperation::new(vec![0]),
            cases = [{
                inputs = [(@unknown(type = input.r#type().into_owned(), replay = input.clone())), (@known, extent)],
                outputs = [(@residual, input)],
                residual_instructions = 0,
            }],
        );
    }

    #[test]
    fn test_dynamic_broadcast_batching() {
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let inputs = [
            ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 1, vec![1.0, 2.0]).unwrap()), BatchAxis::new(0))
                .unwrap(),
            ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())),
        ];
        let output = DynamicBroadcastOperation::new(vec![0])
            .batch(&context, &EmptyRegionDriver, &inputs)
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        assert_eq!(
            output.into_value(),
            ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap())
        );

        // Malformed mappings are rejected before lifting can overflow or index a nonexistent axis.
        assert!(matches!(
            DynamicBroadcastOperation::new(vec![usize::MAX]).batch(&context, &EmptyRegionDriver, &inputs),
            Err(BatchingError::Type(TypeError::Invalid { message }))
                if message == format!(
                    "broadcasting `output_axes[0] = {}` is out of bounds for output rank 1", usize::MAX,
                ),
        ));
        assert!(matches!(
            DynamicBroadcastOperation::new(vec![]).batch(&context, &EmptyRegionDriver, &inputs),
            Err(BatchingError::Type(TypeError::Invalid { message }))
                if message == "broadcasting output axes has length 0 but input has rank 1",
        ));
        let duplicate_inputs = [
            ArrayIrBatch::new(
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F32, [2, 1, 1]), &[1f32, 2.]).unwrap(),
                ),
                BatchAxis::new(0),
            )
            .unwrap(),
            inputs[1].clone(),
            inputs[1].clone(),
        ];
        assert!(matches!(
            DynamicBroadcastOperation::new(vec![0, 0]).batch(&context, &EmptyRegionDriver, &duplicate_inputs),
            Err(BatchingError::Type(TypeError::Invalid { message }))
                if message == "broadcasting output axes map two input axes to output axis 0",
        ));
    }

    #[test]
    fn test_dynamic_broadcast_batching_ragged() {
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(4)).unwrap());
        let extents =
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I64, [2]), &[1i64, 3]).unwrap());
        let value = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32, 0., 0., 2., 3., 4.]).unwrap(),
        );
        let input = ArrayIrBatch::new(value.clone(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), size.clone(), vec![0])])
            .unwrap();
        let dimension =
            ArrayIrBatch::mapped_dimension(extents.clone(), BatchAxis::new(0), DimensionType::new(size.clone()))
                .unwrap();

        // Distinct logical dimensions cannot become equal merely because their packed bounds match.
        let other_size = DimensionVariable::new("other_size", DimensionBounds::new(0, Some(4)).unwrap());
        let other_dimension =
            ArrayIrBatch::mapped_dimension(extents.clone(), BatchAxis::new(0), DimensionType::new(other_size)).unwrap();
        assert!(matches!(
            DynamicBroadcastOperation::new(vec![0]).batch(
                &context, &EmptyRegionDriver, &[input.clone(), other_dimension],
            ),
            Err(BatchingError::InvalidBatchMetadata { message })
                if message == "ragged broadcast input dimension `size` does not match output dimension `other_size`",
        ));

        // Identity broadcasting retains a single entry for the existing ragged axis.
        let output = DynamicBroadcastOperation::new(vec![0])
            .batch(&context, &EmptyRegionDriver, &[input.clone(), dimension.clone()])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(output.ragged_axes(), &[RaggedAxis::new(1, extents.clone(), size.clone(), vec![0])]);
        assert_eq!(output.value(), &value);

        // An inserted axis shifts both the packed axis and its logical metadata.
        let two = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let output = DynamicBroadcastOperation::new(vec![1])
            .batch(&context, &EmptyRegionDriver, &[input, two, dimension])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(output.ragged_axes(), &[RaggedAxis::new(2, extents, size.clone(), vec![0])]);
        assert_eq!(
            output.unbatched_type(),
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![2.into(), size.clone().into()])),)
        );
        assert_eq!(
            output.into_value(),
            ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new_static(DataType::F32, [2, 2, 3]),
                    &[1f32, 0., 0., 1., 0., 0., 2., 3., 4., 2., 3., 4.],
                )
                .unwrap()
            )
        );

        // Replicated ragged geometry may index another ordinary axis rather than the mapped batch axis.
        let extents =
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I64, [2]), &[3i64, 3]).unwrap());
        let input = ArrayIrBatch::replicated(ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32, 2., 3., 4., 5., 6.]).unwrap(),
        ))
        .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), size.clone(), vec![0])])
        .unwrap();
        let dimension = ArrayIrBatch::replicated(ArrayIrValue::Dimension(
            DimensionValue::new(DimensionType::new(size.clone()), 3).unwrap(),
        ));
        // A replicated dimension still describes packed ragged storage when the array has a mapped batch axis.
        let mapped_input = ArrayIrBatch::new(input.value().clone(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(input.ragged_axes().to_vec())
            .unwrap();
        let mapped_output = DynamicBroadcastOperation::new(vec![0])
            .batch(&context, &EmptyRegionDriver, &[mapped_input, dimension.clone()])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(mapped_output.batch_axis(), BatchAxis::new(0));
        assert_eq!(mapped_output.ragged_axes(), input.ragged_axes());
        assert_eq!(mapped_output.value(), input.value());

        let two = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let output = DynamicBroadcastOperation::new(vec![1, 0])
            .batch(&context, &EmptyRegionDriver, &[input, dimension, two])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(output.batch_axis(), BatchAxis::replicated());
        assert_eq!(output.ragged_axes(), &[RaggedAxis::new(0, extents, size.clone(), vec![1])]);
        assert_eq!(
            output.unbatched_type(),
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![size.into(), 2.into()])),)
        );
        assert_eq!(
            output.into_value(),
            ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [3, 2]), &[1f32, 4., 2., 5., 3., 6.],)
                    .unwrap()
            )
        );
    }

    #[test]
    fn test_dynamic_broadcast_batching_layout() {
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let inputs = [
            ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap()), BatchAxis::new(0))
                .unwrap(),
            ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())),
        ];
        let output = DynamicBroadcastOperation::new(vec![])
            .with_output_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![])))
            .batch(&context, &EmptyRegionDriver, &inputs)
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(
            output.into_value(),
            ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new_static(DataType::F32, [2, 3])
                        .with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]))),
                    &[1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0],
                )
                .unwrap()
            )
        );
        assert!(matches!(
            DynamicBroadcastOperation::new(vec![])
                .with_output_layout(Layout::Strided(StridedLayout::new(vec![4])))
                .batch(&context, &EmptyRegionDriver, &inputs),
            Err(BatchingError::InvalidBatchMetadata { message })
                if message == "mapped dynamic broadcasting with an explicit strided output layout requires a runtime batch stride, which is not supported",
        ));
    }

    #[test]
    fn test_dynamic_broadcast_batching_preserves_declared_dynamic_extent() {
        // A mapped array input canonicalizes to a leading batch axis, and the declared output extents cross the
        // batching rule untouched: the lifted broadcast consumes the transform's own batch extent followed by exactly
        // the original extent operands. A declared *dynamic* extent makes that forwarding observable, because a rule
        // that reconstructed output geometry from the operand type would have to stage its own `dimension_size` read.
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_extent = trace.input(DimensionType::new(columns.clone()).into());
        let input = trace
            .input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)])).into());
        let declared_extent_id = declared_extent.atom_id().unwrap();
        let input_id = input.atom_id().unwrap();
        let axis_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let axis_extent_id = axis_extent.atom_id().unwrap();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), axis_extent);

        let [output] = DynamicBroadcastOperation::new(vec![0])
            .batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayIrBatch::new(input, BatchAxis::new(0)).unwrap(), ArrayIrBatch::replicated(declared_extent)],
            )
            .unwrap()
            .into_parts()
            .0
            .try_into()
            .unwrap();
        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        assert_eq!(
            output.value().r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(columns)]),
            )),
        );
        let output_id = output.into_value().atom_id().unwrap();

        // The mapped axis already sits at position zero, so no axis move is staged and the single lifted broadcast
        // forwards the declared extent operand by identity.
        let builder = trace.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected exactly one lifted broadcast instruction");
        };
        assert_eq!(instruction.inputs(), &[input_id, axis_extent_id, declared_extent_id]);
        drop(builder);

        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output_id],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<columns ∈ [1, 9)>, %1:f32[2, 1] .
                let %2:dimension<2> = const 2
                    %3:f32[2, columns] = broadcast [output_axes=[0, 1]] %1 %2 %0
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
                ArrayIrValue::Array(
                    Array::from_elements::<f32>(
                        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)])),
                        &[1.0, 2.0]
                    )
                    .unwrap()
                ),
            ]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements::<f32>(
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
                    &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
                )
                .unwrap()
            )]),
        );
    }

    #[test]
    fn test_dynamic_broadcast_differentiation() {
        let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let tangent = ArrayIrValue::Array(Array::vector(vec![4.0, 5.0, 6.0]).unwrap());
        let (output, output_tangent) =
            differentiate_at(input).jvp(tangent, |value| value.dynamic_broadcast_to_sizes(&[2, 3])).unwrap();
        assert_eq!(output, ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap()));
        assert_eq!(
            output_tangent,
            ArrayIrValue::Array(Array::matrix(2, 3, vec![4.0, 5.0, 6.0, 4.0, 5.0, 6.0]).unwrap())
        );
        let first_extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let second_extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let expected_output = ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f64, 2.0, 1.0, 2.0, 1.0, 2.0]).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let first_extent = builder.add_constant(first_extent);
        let second_extent = builder.add_constant(second_extent);
        let output = builder
            .add_instruction(
                DynamicBroadcastOperation::new(vec![1]),
                Vec::new(),
                vec![input, first_extent, second_extent],
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
        assert!(matches!(program.instructions()[0].operation(), ArrayIrOperation::Broadcast(_)));
        assert_eq!(program.instructions()[0].inputs(), &[input, first_extent, second_extent]);
        assert!(program.to_string().contains("broadcast [output_axes=[1]]"));

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![3.0_f64, 4.0]).unwrap()),
            ]),
            Ok(vec![
                expected_output,
                ArrayIrValue::Array(Array::matrix(3, 2, vec![3.0_f64, 4.0, 3.0, 4.0, 3.0, 4.0]).unwrap()),
            ]),
        );
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.interpret(vec![ArrayIrValue::Array(
                Array::matrix(3, 2, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],).unwrap()
            )]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 12.0]).unwrap())]),
        );

        let dynamic_variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dynamic_extent = DimensionType::new(dynamic_variable.clone());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(dynamic_variable)])).into());
        let extent = builder.add_input(dynamic_extent.clone().into());
        let output = builder
            .add_instruction(DynamicBroadcastOperation::new(vec![0]), Vec::new(), vec![input, extent], None)
            .unwrap()[0];
        let dynamic_program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(dynamic_program.jvp().is_ok());
        let linearization = dynamic_program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dynamic_extent, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let tangent = ArrayIrValue::Array(Array::vector(vec![4.0_f64, 5.0, 6.0]).unwrap());
        let mut tangent_inputs = vec![tangent.clone()];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(linearization.tangent().interpret(tangent_inputs), Ok(vec![tangent.clone()]));
        let mut pullback_inputs = vec![tangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![tangent]));
    }

    #[test]
    fn test_dynamic_broadcast_differentiation_promoted_layout() {
        let input = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F8E8M0FNU, [1]), &[f8e8m0fnu::from_f64(2.).unwrap()])
                .unwrap(),
        );
        let tangent =
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[1f32]).unwrap());
        let (output, tangent) = differentiate_at(input)
            .jvp(tangent, |value| {
                let context = value.dispatch_domain();
                let extent = context.dimension_constant(2)?;
                Ok(context
                    .bind(
                        DynamicBroadcastOperation::new(vec![0])
                            .with_output_layout(Layout::Strided(StridedLayout::new(vec![1]))),
                        Vec::new(),
                        &[value, extent],
                    )?
                    .remove(0))
            })
            .unwrap();
        assert_eq!(
            output,
            ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new_static(DataType::F8E8M0FNU, [2])
                        .with_layout(Layout::Strided(StridedLayout::new(vec![1]))),
                    &[f8e8m0fnu::from_f64(2.).unwrap(); 2],
                )
                .unwrap()
            )
        );
        assert_eq!(
            tangent,
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[1f32, 1.],).unwrap())
        );
    }

    #[test]
    fn test_dynamic_broadcast_differentiation_dynamic_input_layout() {
        let size = DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![size.clone().into(), 1.into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4, 4])));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let dimension = builder.add_input(DimensionType::new(size).into());
        let three = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(
                DynamicBroadcastOperation::new(vec![0, 1]),
                Vec::new(),
                vec![input, dimension, three],
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
        let linearization = program.linearize().unwrap();
        let pullback = linearization.pullback().unwrap();
        assert_eq!(pullback.outputs().next().unwrap().r#type().as_ref(), &ArrayIrType::Array(input_type));
        let concrete_type =
            ArrayType::new_static(DataType::F32, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![4, 4])));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::from_elements(concrete_type.clone(), &[1f32, 2.]).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(primal_outputs.len() - linearization.residual_count());
        let mut pullback_inputs = vec![ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32, 2., 3., 4., 5., 6.]).unwrap(),
        )];
        pullback_inputs.extend(residuals);
        assert_eq!(
            pullback.interpret(pullback_inputs).unwrap()[0],
            ArrayIrValue::Array(Array::from_elements(concrete_type, &[6f32, 15.]).unwrap(),)
        );
    }

    #[test]
    fn test_dynamic_broadcast_transposition() {
        let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let (output, pullback) =
            differentiate_at(input).vjp(|value| value.dynamic_broadcast_to_sizes(&[2, 3])).unwrap();
        assert_eq!(output, ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap()));
        assert_eq!(
            pullback.apply(ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())),
            Ok(ArrayIrValue::Array(Array::vector(vec![5.0, 7.0, 9.0]).unwrap())),
        );
    }

    #[test]
    fn test_dynamic_broadcast_dynamic_broadcast_with_output_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2);
        let operation = DynamicBroadcastOperation::new(vec![1]).with_output_sharding(sharding.clone());
        assert_eq!(operation.output_axes(), &[1]);
        assert_eq!(operation.output_sharding(), Some(&sharding));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let dimensions = [
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        ];
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()]))
            .with_sharding(sharding.clone())
            .unwrap();
        assert_eq!(
            input.dynamic_broadcast_with_output_sharding(&dimensions, &[1], Some(sharding)),
            Ok(ArrayIrValue::Array(Array::from_elements::<f64>(output_type, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap())),
        );
    }

    #[test]
    fn test_dynamic_broadcast_dynamic_broadcast_to() {
        let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let dimensions = [
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        ];
        assert_eq!(
            input.dynamic_broadcast_to(&dimensions),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap())),
        );
    }

    #[test]
    fn test_dynamic_broadcast_to_first_class_dimensions() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let eager = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]).unwrap());
        assert_eq!(
            eager.dynamic_broadcast_leading_sizes(&[2]),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 1.0, 2.0, 3.0],).unwrap())),
        );

        let context = TestContext::new();
        let value = context.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).into());
        let extent = context.constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        assert_eq!(value.dynamic_broadcast(&[extent], &[0]).unwrap().atom_id(), value.atom_id());
        assert!(context.builder().borrow().instructions().is_empty());

        // A shape-preserving axis permutation is still a real broadcast. Eager execution transposes the payload and
        // tracing retains the operation even though its input and output types are equal.
        let square_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let square = ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap());
        let two = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let expected = ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap());
        assert_eq!(square.dynamic_broadcast(&[two.clone(), two.clone()], &[1, 0]), Ok(expected.clone()));

        let context = TestContext::new();
        let value = context.input(square_type.into());
        let extent = context.constant(two);
        let output = value.dynamic_broadcast(&[extent.clone(), extent], &[1, 0]).unwrap();
        {
            let builder = context.builder().borrow();
            let [instruction] = builder.instructions() else {
                panic!("expected one shape-preserving broadcast instruction");
            };
            let ArrayIrOperation::Broadcast(operation) = instruction.operation() else {
                panic!("expected a broadcast instruction");
            };
            assert_eq!(operation.output_axes(), &[1, 0]);
        }
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
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
        let output = scalar.dynamic_broadcast_to(std::slice::from_ref(&extent)).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
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
                ArrayIrValue::Array(Array::scalar(2.5_f64).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![2.5_f64, 2.5, 2.5]).unwrap())]),
        );
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(6.0_f64).unwrap())]),
        );

        let context = TestContext::new();
        let value = context.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let rows = context.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let columns = context.constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = value.dynamic_broadcast_to(&[rows, columns]).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(Array::vector(vec![7.0_f64]).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![7.0_f64; 6]).unwrap())]),
        );

        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let context = TestContext::new();
        let value = context.input(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(3)]))
                .into(),
        );
        let output = value.dynamic_broadcast_leading_sizes(&[2]).unwrap();
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(batch), Dimension::Static(3)]),
            )),
        );
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
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
    fn test_dynamic_broadcast_dynamic_broadcast_to_exemplar() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(5)).unwrap());
        let value = context.input(ArrayType::scalar(DataType::F32).into());
        let exemplar = context.input(ArrayType::new(DataType::F32, Shape::new(vec![size.into(), 2.into()])).into());
        // Filling an exemplar's geometry is an ordinary broadcast with sizes read from that exemplar.
        let dimensions = [exemplar.dimension_size(0).unwrap(), exemplar.dimension_size(1).unwrap()];
        let output = value.dynamic_broadcast_to(&dimensions).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let value = ArrayIrValue::Array(Array::scalar(3f32).unwrap());
        assert_eq!(
            program.interpret(vec![value.clone(), ArrayIrValue::Array(Array::matrix(2, 2, vec![0f32; 4]).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 2, vec![3f32; 4]).unwrap())])
        );
        assert_eq!(
            program.interpret(vec![value, ArrayIrValue::Array(Array::matrix(0, 2, Vec::<f32>::new()).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::matrix(0, 2, Vec::<f32>::new()).unwrap())])
        );
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::scalar(3f32).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 2, vec![0f32; 4]).unwrap()),
                ArrayIrValue::Array(Array::scalar(5f32).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 2, vec![0f32; 4]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(2, 2, vec![3f32; 4]).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 2, vec![5f32; 4]).unwrap()),
            ])
        );
    }

    #[test]
    fn test_dynamic_broadcast_dynamic_broadcast_to_with_output_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1);
        let input =
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[3_i32, 7]).unwrap());
        let dimensions = [ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())];
        let output_type = ArrayType::new_static(DataType::I32, [2]).with_sharding(sharding.clone()).unwrap();
        let expected = ArrayIrValue::Array(Array::from_elements(output_type, &[3_i32, 7]).unwrap());
        assert_eq!(input.dynamic_broadcast_to_with_output_sharding(&dimensions, Some(sharding)), Ok(expected.clone()));
        assert_eq!(expected.dynamic_broadcast_to_with_output_sharding(&dimensions, None), Ok(expected));
    }

    #[test]
    fn test_dynamic_broadcast_dynamic_broadcast_leading() {
        let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let dimensions = [ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())];
        assert_eq!(
            input.dynamic_broadcast_leading(&dimensions),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap())),
        );
        assert_eq!(input.dynamic_broadcast_leading(&[]), Ok(input));
    }

    #[test]
    fn test_dynamic_broadcast_dynamic_broadcast_leading_with_output_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2);
        let input =
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[3_i32, 7]).unwrap());
        let dimensions = [ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())];
        let output_type = ArrayType::new_static(DataType::I32, [2, 2]).with_sharding(sharding.clone()).unwrap();
        assert_eq!(
            input.dynamic_broadcast_leading_with_output_sharding(&dimensions, Some(sharding)),
            Ok(ArrayIrValue::Array(Array::from_elements(output_type, &[3_i32, 7, 3, 7]).unwrap())),
        );
    }

    #[test]
    fn test_dynamic_broadcast_dynamic_broadcast_to_sizes() {
        let input = ArrayIrValue::Array(Array::scalar(7.0).unwrap());
        assert_eq!(
            input.dynamic_broadcast_to_sizes(&[2, 3]),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![7.0; 6]).unwrap())),
        );
        assert_eq!(
            input.dynamic_broadcast_to_sizes(&[0]),
            Ok(ArrayIrValue::Array(Array::vector(Vec::<f64>::new()).unwrap()))
        );
    }

    #[test]
    fn test_dynamic_broadcast_dynamic_broadcast_leading_sizes() {
        let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        assert_eq!(
            input.dynamic_broadcast_leading_sizes(&[2]),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap())),
        );
        assert_eq!(input.dynamic_broadcast_leading_sizes(&[]), Ok(input));
    }

    #[test]
    fn test_dynamic_broadcast_dynamic_broadcast_arrays() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(5)).unwrap());
        let scalar = context.input(ArrayType::scalar(DataType::I32).into());
        let vector = context.input(ArrayType::new(DataType::F32, Shape::new(vec![size.into()])).into());
        let outputs = Tracer::dynamic_broadcast_arrays(&[scalar, vector]).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                outputs.iter().map(|output| output.atom_id().unwrap()).collect(),
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let scalar = ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::I32), &[7_i32]).unwrap());
        let vector = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[1_f32, 2.0]).unwrap(),
        );
        assert_eq!(
            program.interpret(vec![scalar.clone(), vector.clone()]),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[7_i32, 7]).unwrap()
                ),
                vector
            ]),
        );
        // Runtime zero extents remain zero, rather than becoming the allocation bound.
        let empty = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [0]), &[] as &[f32]).unwrap(),
        );
        assert_eq!(
            program.interpret(vec![scalar, empty.clone()]),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::I32, [0]), &[] as &[i32]).unwrap()
                ),
                empty
            ]),
        );
        assert_eq!(ArrayIrValue::<Array>::dynamic_broadcast_arrays(&[]), Ok(Vec::new()));
    }

    #[test]
    fn test_validate_broadcast_output_layout() {
        assert_eq!(validate_broadcast_output_layout(&ArrayType::new_static(DataType::F32, [3])), Ok(()));
        assert_eq!(
            validate_broadcast_output_layout(
                &ArrayType::new_static(DataType::F32, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            ),
            Ok(()),
        );
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(5)).unwrap());
        assert_eq!(
            validate_broadcast_output_layout(
                &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]))
                    .with_layout(Layout::Strided(StridedLayout::new(vec![2])))
            ),
            Err(TypeError::invalid(
                "strided layout stride 2 on axis 0 is smaller than the 4-byte span occupied by more minor axes and may alias array elements",
            )),
        );
    }

    #[test]
    fn test_array_broadcast() {
        let vector = Array::vector(vec![1.0, 2.0]).unwrap();
        let output_type = ArrayType::new_static(DataType::F64, [3, 2]);
        let broadcast = Broadcast::broadcast(&vector, output_type.clone(), &[1]).unwrap();
        assert_eq!(broadcast.r#type().into_owned(), output_type);
        assert_eq!(broadcast.to_f64s(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        // Broadcasting reads a reversed input layout and writes the output's requested physical layout, retaining
        // zero in its holes.
        let input_type =
            ArrayType::new_static(DataType::U16, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let input = Array::from_elements(input_type, &[0x1122u16, 0x3344]).unwrap();
        let output_type =
            ArrayType::new_static(DataType::U16, [2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![6, 2])));
        let broadcast = input.broadcast(output_type.clone(), &[1]).unwrap();
        assert_eq!(broadcast.r#type().as_ref(), &output_type);
        assert_eq!(broadcast.elements::<u16>(), Ok(vec![0x1122, 0x3344, 0x1122, 0x3344]));
        assert_eq!(broadcast.storage_bytes(), [0x22, 0x11, 0x44, 0x33, 0, 0, 0x22, 0x11, 0x44, 0x33]);

        // Deliberately malformed concrete values with dynamic types fail through the structured materialization
        // diagnostic before either the identity fast path or static-shape payload logic can accept or panic on them.
        let dynamic = DimensionVariable::new("dynamic", DimensionBounds::unbounded());
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(dynamic.clone()), Dimension::Dynamic(dynamic)]),
        );
        let dynamic = Array::with_unchecked_type(
            dynamic_type.clone(),
            [1.0f64, 2.0, 3.0, 4.0].into_iter().flat_map(f64::to_le_bytes).collect(),
        );
        for output_axes in [vec![0, 1], vec![1, 0]] {
            assert!(matches!(
                dynamic.broadcast(dynamic_type.clone(), output_axes.as_slice()),
                Err(ProgramError::Type(TypeError::Invalid { message }))
                    if message == "cannot materialize a value of dynamically sized type f64[dynamic, dynamic]",
            ));
        }
    }
    #[test]
    fn test_array_broadcast_structural_zero() {
        let input = Array::new(ArrayType::scalar(DataType::Zero), vec![]).unwrap();
        let output_type = ArrayType::new_static(DataType::Zero, [usize::MAX]);
        let output = input.broadcast(output_type.clone(), &[]).unwrap();
        assert_eq!(output.r#type().as_ref(), &output_type);
        assert!(output.storage_bytes().is_empty());
    }
}
