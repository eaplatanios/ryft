//! Contains the named-axis [`AllToAllOperation`], which exchanges chunks between the participants along a named
//! axis, together with its interpretation, partial-evaluation, batching, forward-mode differentiation, and
//! transposition rules.

// TODO(eaplatanios): Review this module.

use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
use crate::arrays::{
    ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType, ArrayType, Dimension,
    DimensionOperation, DimensionType, DimensionValue, DimensionVariable, MeshAxisType, Shape, Sharding,
};
use crate::axes::{NamedAxes, NamedAxis};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    MemberBatchableOperation,
};
use crate::contexts::{Context, Domain, ProjectedContext};
use crate::differentiation::{
    DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
    MemberDifferentiableOperation,
};
use crate::interpretation::{InterpretationDriver, MemberInterpretableOperation};
use crate::macros::check_count;
use crate::operations::arithmetic::{Div, Mul, Rem};
use crate::operations::assertions::Assert;
use crate::operations::collectives::parallel_vary::ParallelVary;
use crate::operations::collectives::{
    CollectiveArrayExtentBatchingPolicy, CollectiveExtent, CollectiveMode, CollectiveOptions, collective_input_extents,
    define_linear_collective_operation, explicit_collective_inputs, forward_explicit_collective,
    forward_shape_changing_collective, impl_differentiable_linear_collective_operation,
    impl_shape_changing_collective_member_operation, infer_explicit_shape_changing_collective_output_type,
    infer_linear_collective_operation_output_type, jvp_shape_changing_collective_with_adjoint, resolve_named_axis_size,
    validate_explicit_collective_output_extents,
};
use crate::operations::comparisons::Compare;
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_max::DimensionMax;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{DynamicBroadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::programs::{
    MemberOperation, Operation, OperationProjection, ProgramError, ProjectedValue, RegionInterface, TypeError,
    TypeIdentityRenaming, Typed, Value, ValueProjection,
};

/// Canonical operation name for [`AllToAllOperation`].
pub const ALL_TO_ALL_OPERATION_NAME: &str = "all_to_all";

define_linear_collective_operation!(
    /// [`Operation`] that exchanges chunks between participants along a named axis. Within each ordered participant
    /// group, every sender splits its input along `split_axis`; receiver `i` gets chunk `i` from every sender, in group
    /// order. The [`CollectiveMode`] of its [`CollectiveOptions`] determines the shape over a group of `n` participants:
    ///
    ///   - [`CollectiveMode::Untiled`] requires extent `n` at `split_axis`, removes that input axis, and inserts extent
    ///     `n` at `concat_axis` in the output. Each receiver gets one slice from each sender along the inserted axis.
    ///   - [`CollectiveMode::Tiled`] requires the split extent to be divisible by `n`. It divides that extent by `n` and
    ///     multiplies the concatenation extent by `n`; when the axes coincide, the shape is unchanged.
    ///
    /// Both modes preserve rank and element data type. This is the analogue of
    /// [`jax.lax.all_to_all`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_to_all.html); tiled exchanges lower
    /// directly to StableHLO's [`all_to_all`](https://openxla.org/stablehlo/spec#all_to_all), while untiled exchanges
    /// insert and remove singleton dimensions around it. The collective is linear; its transpose swaps the split and
    /// concatenation axes and retains the mode and ordered participant groups.
    ///
    /// Over a manual mesh axis, an exchange can give the receivers different values, so its input must vary over that
    /// axis (refer to [`ParallelVary`]) and its output varies over it too. [`AllToAll::all_to_all_with_options`] makes an
    /// invariant input varying automatically. Mesh exchanges reject inputs with pending cross-device sums. Type
    /// inference in the homogeneous array family requires static extents; the composite array/dimension family uses
    /// explicit result extents, with runtime assertions for dynamic split divisibility and untiled split size.
    ///
    /// A matching `batch` level consumes the named axis with a local reshape/transpose block exchange. Batch item `i`
    /// receives every item's chunk `i`, in sender order. A replicated input is broadcast before the exchange, since
    /// receivers can still get different chunks. Participant groups are unsupported at a matching level. A local
    /// exchange preserves enclosing mesh variation and pending sums, even when its axis name shadows a mesh axis.
    /// Outside any binder, a single-participant tiled exchange is the identity; untiled mode relocates its size-one
    /// split axis to the concatenation position.
    ///
    /// Bounded ragged inputs are rejected. One extent per item does not determine how each sender partitions its live
    /// prefix among receivers; that requires the explicit offsets and per-destination sizes of
    /// [`RaggedAllToAllOperation`](crate::operations::collectives::RaggedAllToAllOperation).
    AllToAllOperation,
    ALL_TO_ALL_OPERATION_NAME,
    fields = {
        /// Axis of the input that is split into one chunk per participant.
        split_axis: usize,

        /// Axis of the output along which the received chunks are concatenated.
        concat_axis: usize,

        /// Shared rank and participant-group semantics.
        options: CollectiveOptions,
    },
    check_array_types = [@no_unreduced],
    infer_output_type = |operation, input_type, dimensions| {
        infer_all_to_all_output_type(operation, input_type, dimensions, true)
    },
    interpret<C> where C::Value: Reshape {
        |operation, input| {
            // A single participant exchanges chunks only with itself. Untiled mode removes the size-one split axis and
            // inserts a size-one concatenation axis, which a reshape to the inferred output type expresses, while tiled
            // mode leaves the shape unchanged.
            match operation.options.mode {
                CollectiveMode::Tiled => Ok(input.clone()),
                CollectiveMode::Untiled => {
                    let output_type = operation.infer_output_types(&[input.r#type().into_owned()], &[])?.remove(0);
                    input.reshape_with_output_sharding(output_type.shape().clone(), output_type.sharding().cloned())
                }
            }
        }
    },
);

impl AllToAllOperation {
    /// Returns the axis of the input that is split into one chunk per participant.
    #[inline]
    pub fn split_axis(&self) -> usize {
        self.split_axis
    }

    /// Returns the axis of the output along which the received chunks are concatenated.
    #[inline]
    pub fn concat_axis(&self) -> usize {
        self.concat_axis
    }

    /// Returns the shared rank and participant-group semantics.
    #[inline]
    pub fn options(&self) -> &CollectiveOptions {
        &self.options
    }

    /// Returns the participant count used for result-shape arithmetic.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if the axis size or participant groups violate [`CollectiveOptions`] requirements.
    #[inline]
    pub fn effective_axis_size(&self) -> Result<usize, TypeError> {
        self.options.effective_axis_size(ALL_TO_ALL_OPERATION_NAME, self.axis_size)
    }
}

// Batching rule for [`AllToAllOperation`]. A matching `batch` level consumes the mapped batch axis with a
// reshape/transpose block exchange: the per-item `split_axis` is split into `(b, d_p / b)` chunks, the chunk axis is
// swapped with the leading batch axis (so the batch axis indexes the *receiving* item), and the sender axis is then
// merged item-major into the per-item `concat_axis` — batch item `i` receives every item's chunk `i`, concatenated
// along `concat_axis`. A non-matching level forwards the collective to the parent context, unchanged for a replicated
// input (through `BatchingContext::forward_to_parent`) and with its array axes shifted past the batch axis for a mapped
// one.
impl<C, P: CollectiveArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>> for AllToAllOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<AllToAllOperation>,
    <C as Domain>::Value: Transpose,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        if let Some(ragged_axis) = inputs.iter().find_map(|input| input.ragged_axes().first()) {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`all_to_all` cannot route bounded ragged dimension `{}` without explicit per-destination \
                     offsets and sizes; use `ragged_all_to_all`",
                    ragged_axis.dimension(),
                ),
            });
        }
        if context.axis_name() != Some(self.axis_name.as_str()) {
            return forward_shape_changing_collective(context, self, inputs, |batch_axis| {
                let (split_axis, output_batch_axis) =
                    self.options.mode.forwarded_split_axes(self.split_axis, batch_axis);
                let (concat_axis, output_batch_axis) =
                    self.options.mode.forwarded_concat_axes(self.concat_axis, output_batch_axis);
                let operation =
                    Self::new(self.axis_name.clone(), self.axis_size, split_axis, concat_axis, self.options.clone());
                (operation, output_batch_axis)
            });
        }
        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };
        let input_type = input.unbatched_type();
        let dimensions = input_type
            .static_shape()
            .ok_or_else(|| TypeError::invalid("`all_to_all` does not support dynamically shaped inputs"))?;
        let output_type = infer_all_to_all_output_type(self, &input_type, dimensions.dimensions().to_vec(), false)?;
        let output_extents = output_type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| P::collective_extent_from_dimension(context, dimension))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(vec![batch_all_to_all_matching_axis::<C, P>(
            self,
            context,
            input,
            input_type.rank(),
            output_extents,
            output_type.sharding().cloned(),
        )?]
        .into())
    }
}

// Transpose rule for [`AllToAllOperation`]: the chunk exchange is its own adjoint with the split and concatenation
// axes swapped.
impl_differentiable_linear_collective_operation! {
    AllToAllOperation,
    transpose = |operation| -> AllToAllOperation {
        AllToAllOperation::new(
            operation.axis_name.clone(),
            operation.axis_size,
            operation.concat_axis,
            operation.split_axis,
            operation.options.clone(),
        )
    },
}

impl_shape_changing_collective_member_operation!(AllToAllOperation, infer_explicit_all_to_all_output_types);

// Batching rule for explicit-extent [`AllToAllOperation`]. Dimension SSA supplies its temporary split and merge
// shapes directly, while matching-axis array mechanics reuse the homogeneous collective kernel.
impl<C> MemberBatchableOperation<C, ArrayIrBatchingPolicy> for AllToAllOperation
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<AllToAllOperation>
                           + From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: Assert
        + DimensionSize
        + DynamicBroadcast
        + ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>
        + ValueProjection<DimensionType>,
    <C::Value as ValueProjection<DimensionType>>::Projected:
        Compare<C::Value> + DimensionMax + Rem + Div + Mul + Value<Type = DimensionType>,
{
    fn batch_in_parent<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        let (array, output_extents) = explicit_collective_inputs(inputs)?;
        if let Some(ragged_axis) = array.ragged_axes().first() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`all_to_all` cannot route bounded ragged dimension `{}` without explicit per-destination \
                     offsets and sizes; use `ragged_all_to_all`",
                    ragged_axis.dimension(),
                ),
            });
        }
        validate_explicit_collective_output_extents(output_extents)?;
        let logical_input_types = inputs.iter().map(|input| input.unbatched_type().clone()).collect::<Vec<_>>();
        // Validate local geometry before remapping axes, but leave mesh semantics to the level that handles the
        // exchange. A non-matching level can forward into another batch that shadows the manual mesh axis.
        let mut logical_output_types = infer_explicit_all_to_all_output_types_with_mesh_axis_semantics(
            self,
            logical_input_types.as_slice(),
            false,
        )?;
        let logical_output_type = <&ArrayType>::try_from(&logical_output_types.remove(0))?.clone();

        if context.axis_name() != Some(self.axis_name()) {
            if array.batch_axis().is_replicated() {
                return Ok(forward_explicit_collective(self.clone(), context, array, output_extents, None)?.into());
            }
            let input_batch_axis = array.batch_axis_position().unwrap();
            let (physical_split_axis, output_batch_axis) =
                self.options().mode().forwarded_split_axes(self.split_axis(), input_batch_axis);
            let (physical_concat_axis, output_batch_axis) =
                self.options().mode().forwarded_concat_axes(self.concat_axis(), output_batch_axis);
            let operation = Self::new(
                self.axis_name().to_string(),
                self.axis_size(),
                physical_split_axis,
                physical_concat_axis,
                self.options().clone(),
            );
            return Ok(forward_explicit_collective(
                operation,
                context,
                array,
                output_extents,
                Some(output_batch_axis),
            )?
            .into());
        }

        let array = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(array.value().clone())?,
            array.batch_axis(),
        )?;
        let output_extents = output_extents
            .iter()
            .map(|extent| <C::Value as ValueProjection<DimensionType>>::into_projected(extent.value().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let projected_context =
            BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                ProjectedContext::new(context.parent().clone()),
                context.axis_extent().clone(),
            )
            .with_axis_name(context.axis_name().map(str::to_string))
            .with_axis_sharding(context.axis_sharding().clone());
        let output = batch_all_to_all_matching_axis::<_, DynamicArrayExtentBatchingPolicy>(
            self,
            &projected_context,
            &array,
            array.unbatched_type().rank(),
            output_extents,
            logical_output_type.sharding().cloned(),
        )?;
        let batch_axis = output.batch_axis();
        Ok(ArrayIrBatch::new(<C::Value as ValueProjection<ArrayType>>::from_projected(output.into_value()), batch_axis)
            .map(|output| vec![output])?
            .into())
    }
}

// Mixed array IR JVP for all-to-all. Explicit output extents are retained as ordinary residual values, and the
// transposed linear region swaps the split and concatenation axes.
impl<C> MemberDifferentiableOperation<C> for AllToAllOperation
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<AllToAllOperation>
        + From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + From<ConstantOperation<DimensionValue>>
        + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        jvp_shape_changing_collective_with_adjoint(self, self.adjoint()?, context, inputs)
    }
}

/// Represents the ability to exchange chunks between participants of a named axis by staging an [`AllToAllOperation`].
/// Refer to that operation for the tiling, grouping, variation, and transformation semantics. Dynamic result extents
/// are staged as first-class dimension values, and runtime assertions validate dynamic split extents.
///
/// # Example
///
/// Each row sends its first half to batch item zero and its second half to batch item one:
///
/// ```
/// # use ryft_core::operations::collectives::AllToAll;
/// # use ryft_core::{
/// #     Array, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, BatchAxis, BatchAxisSpecification,
/// #     BatchingTracer, EagerContext, batch,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rows = ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?);
/// let received = batch(
///     |row: BatchingTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, ArrayIrBatchingPolicy>| {
///         row.all_to_all_tiled("rows", 0, 0)
///     },
///     rows,
///     BatchAxis::new(0),
///     BatchAxis::new(0),
///     BatchAxisSpecification::named("rows"),
/// )?;
/// assert_eq!(received, ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0])?));
/// # Ok(())
/// # }
/// ```
pub trait AllToAll: Sized {
    /// Exchanges single slices, removing `split_axis` from the input and inserting the sender axis at `concat_axis`
    /// in the output. The split extent must equal the participant count, and rank is preserved.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name bound by an enclosing batch level or manual region.
    ///   - `split_axis`: Input axis split into one slice per receiver.
    ///   - `concat_axis`: Output position at which sender slices are stacked after removing `split_axis`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`AllToAll::all_to_all_with_options`].
    #[inline]
    fn all_to_all(&self, axis_name: &str, split_axis: usize, concat_axis: usize) -> Result<Self, ProgramError> {
        self.all_to_all_with_options(axis_name, split_axis, concat_axis, CollectiveOptions::default())
    }

    /// Exchanges equal contiguous chunks, dividing the extent of `split_axis` and multiplying that of `concat_axis`
    /// by the participant count. Coincident axes preserve the shape. The split extent must be divisible by that count.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name bound by an enclosing batch level or manual region.
    ///   - `split_axis`: Input axis split into one chunk per receiver.
    ///   - `concat_axis`: Array axis along which received chunks are concatenated in sender order.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`AllToAll::all_to_all_with_options`].
    #[inline]
    fn all_to_all_tiled(&self, axis_name: &str, split_axis: usize, concat_axis: usize) -> Result<Self, ProgramError> {
        self.all_to_all_with_options(axis_name, split_axis, concat_axis, CollectiveOptions::new(CollectiveMode::Tiled))
    }

    /// Exchanges chunks with the tiling mode and ordered participant groups of `options`. Over a manual mesh axis,
    /// an invariant input is first made varying through [`ParallelVary`].
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name bound by an enclosing batch level or manual region.
    ///   - `split_axis`: Input axis split into one slice or chunk per receiver.
    ///   - `concat_axis`: Output axis holding the received slices or chunks, as selected by `options`.
    ///   - `options`: Tiling mode and optional ordered participant groups.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Axis`] wrapping [`AxisError::UnboundAxisName`](crate::axes::AxisError) when no binder
    /// binds `axis_name`, and a [`ProgramError`] for invalid axes, groups, or split geometry, pending cross-device sums,
    /// or an overflowing concatenation extent. A batch level that binds the collective axis rejects participant
    /// groups and bounded ragged inputs.
    fn all_to_all_with_options(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError>;
}

// Composite values stage the array followed by one result extent per axis. Only a manual mesh binder introduces
// variation; a named batch that shadows the same mesh-axis name performs its own local exchange.
impl<V> AllToAll for V
where
    V: Value<Type = ArrayIrType>
        + Assert
        + DimensionSize<V>
        + ValueProjection<DimensionType>
        + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
    V::DispatchDomain: Context<Type = ArrayIrType> + NamedAxes,
    V::DispatchDomain: DimensionConstant,
    <V::DispatchDomain as Domain>::Operation: From<AllToAllOperation>,
    <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + Compare<V> + Rem + Div + Mul,
    ProjectedValue<ArrayType, V>: ParallelVary,
{
    fn all_to_all_with_options(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let effective_axis_size = options.effective_axis_size(ALL_TO_ALL_OPERATION_NAME, axis_size)?;
        let mut input = self.clone();
        if matches!(context.named_axis(axis_name), Some(NamedAxis::Mesh { .. })) {
            let array = ValueProjection::<ArrayType>::into_projected(self.clone())?;
            if !array.r#type().unreduced_axes().is_empty() {
                return Err(TypeError::invalid("`all_to_all` does not support unreduced inputs").into());
            }
            if !array.r#type().sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis_name)) {
                input = <V as ValueProjection<ArrayType>>::from_projected(array.parallel_vary(axis_name)?);
            }
        }
        let operation =
            AllToAllOperation::new(axis_name.to_string(), axis_size, split_axis, concat_axis, options.clone());
        let mut output_extents = collective_input_extents(&input)?;
        let rank = output_extents.len();
        if split_axis >= rank || concat_axis >= rank {
            return Err(TypeError::invalid(format!(
                "`all_to_all` split axis {split_axis} or concat axis {concat_axis} is out of bounds for rank {rank}",
            ))
            .into());
        }
        match options.mode {
            CollectiveMode::Untiled => {
                output_extents[split_axis].require_equal(&context, effective_axis_size)?;
                output_extents.remove(split_axis);
                output_extents.insert(concat_axis, CollectiveExtent::Static(effective_axis_size));
            }
            CollectiveMode::Tiled if split_axis == concat_axis => {
                output_extents[split_axis].require_divisible(&context, effective_axis_size)?;
            }
            CollectiveMode::Tiled => {
                let split_extent = output_extents[split_axis].divided(&context, effective_axis_size)?;
                let concat_extent = output_extents[concat_axis].multiplied(&context, effective_axis_size)?;
                output_extents[split_axis] = split_extent;
                output_extents[concat_axis] = concat_extent;
            }
        };
        let output_extents =
            output_extents.into_iter().map(|extent| extent.stage(&context)).collect::<Result<Vec<_>, _>>()?;
        let inputs = std::iter::once(input).chain(output_extents).collect::<Vec<_>>();
        let mut outputs = context.bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<V> AllToAll for ProjectedValue<ArrayType, V>
where
    V: AllToAll + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
{
    fn all_to_all_with_options(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        self.value()
            .all_to_all_with_options(axis_name, split_axis, concat_axis, options)?
            .into_projected()
            .map_err(Into::into)
    }
}

/// Convenience untiled all-to-all that exchanges one ranked array axis with a named axis.
pub trait ParallelSwapAxes: AllToAll {
    /// Swaps `axis` with `axis_name` over the full named axis. The ranked axis must have the participant count as its
    /// extent. This is [`AllToAll::all_to_all`] with identical split and concatenation positions.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name bound by an enclosing batch level or manual region.
    ///   - `axis`: Ranked axis to exchange with the named axis.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`AllToAll::all_to_all_with_options`].
    #[inline]
    fn parallel_swap_axes(&self, axis_name: &str, axis: usize) -> Result<Self, ProgramError> {
        self.all_to_all(axis_name, axis, axis)
    }

    /// Swaps `axis` with `axis_name` within the provided ordered participant groups. The ranked axis extent must
    /// equal the common group size, and senders are stacked in the order specified by their group.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name bound by an enclosing batch level or manual region.
    ///   - `axis`: Ranked axis to exchange with the named axis.
    ///   - `axis_index_groups`: Ordered equal-sized partition of the named-axis participant coordinates.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`AllToAll::all_to_all_with_options`].
    #[inline]
    fn parallel_swap_axes_with_axis_index_groups(
        &self,
        axis_name: &str,
        axis: usize,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError> {
        self.all_to_all_with_options(
            axis_name,
            axis,
            axis,
            CollectiveOptions::default().with_axis_index_groups(axis_index_groups),
        )
    }
}

impl<V: AllToAll> ParallelSwapAxes for V {}

/// Infers a statically shaped all-to-all result. A matching named batch performs a local exchange and preserves
/// enclosing mesh variation and pending sums; canonical inference validates the manual mesh exchange instead.
fn infer_all_to_all_output_type(
    operation: &AllToAllOperation,
    input_type: &ArrayType,
    dimensions: Vec<usize>,
    apply_mesh_axis_semantics: bool,
) -> Result<ArrayType, TypeError> {
    let effective_axis_size = operation.effective_axis_size()?;
    let mut output_dimensions = dimensions;
    let rank = output_dimensions.len();
    if operation.split_axis >= rank || operation.concat_axis >= rank {
        return Err(TypeError::invalid(format!(
            "`all_to_all` split axis {} or concat axis {} is out of bounds for rank {rank}",
            operation.split_axis, operation.concat_axis,
        )));
    }
    let output_type = if operation.options.mode == CollectiveMode::Untiled {
        if output_dimensions[operation.split_axis] != effective_axis_size {
            return Err(TypeError::invalid(format!(
                "`all_to_all` untiled split axis {} size {} must equal group size {}",
                operation.split_axis, output_dimensions[operation.split_axis], effective_axis_size,
            )));
        }
        input_type
            .without_dimension(operation.split_axis)?
            .0
            .with_inserted_dimension(operation.concat_axis, Dimension::Static(effective_axis_size))?
    } else {
        if output_dimensions[operation.split_axis] % effective_axis_size != 0 {
            return Err(TypeError::invalid(format!(
                "`all_to_all` split axis {} size {} is not divisible by group size {}",
                operation.split_axis, output_dimensions[operation.split_axis], effective_axis_size,
            )));
        }
        output_dimensions[operation.split_axis] /= effective_axis_size;
        output_dimensions[operation.concat_axis] =
            output_dimensions[operation.concat_axis].checked_mul(effective_axis_size).ok_or_else(|| {
                TypeError::invalid("`all_to_all` concatenation result extent does not fit in usize".to_string())
            })?;
        infer_linear_collective_operation_output_type(ALL_TO_ALL_OPERATION_NAME, input_type, output_dimensions)?
    };
    all_to_all_output_type(operation, input_type, output_type, apply_mesh_axis_semantics)
}

/// Validates a mesh exchange's manual variation. A local named batch preserves its input's mesh state, including
/// pending sums, because it performs only local array rearrangement even when it shadows a manual mesh axis.
fn all_to_all_output_type(
    operation: &AllToAllOperation,
    input_type: &ArrayType,
    output_type: ArrayType,
    apply_mesh_axis_semantics: bool,
) -> Result<ArrayType, TypeError> {
    if apply_mesh_axis_semantics
        && let Some(sharding) = input_type.sharding()
        && sharding.mesh().axis_type(operation.axis_name()) == Some(MeshAxisType::Manual)
        && !sharding.varying_manual_axes().contains(operation.axis_name())
    {
        return Err(TypeError::invalid(format!(
            "`all_to_all` input must vary over manual axis `{}`; pass an invariant value through `parallel_vary` \
             first so that the exchanged output is typed as varying",
            operation.axis_name(),
        )));
    }
    Ok(output_type)
}

/// Infers an all-to-all in the composite array/dimension family, whose array input is followed by one explicit
/// extent per output axis. Known extents are checked here; dynamic extents are checked by the capability's assertions.
pub(crate) fn infer_explicit_all_to_all_output_types(
    operation: &AllToAllOperation,
    input_types: &[ArrayIrType],
) -> Result<Vec<ArrayIrType>, TypeError> {
    infer_explicit_all_to_all_output_types_with_mesh_axis_semantics(operation, input_types, true)
}

/// Infers explicit-extent outputs with the manual mesh semantics required by the calling binder.
fn infer_explicit_all_to_all_output_types_with_mesh_axis_semantics(
    operation: &AllToAllOperation,
    input_types: &[ArrayIrType],
    apply_mesh_axis_semantics: bool,
) -> Result<Vec<ArrayIrType>, TypeError> {
    let effective_axis_size = operation.effective_axis_size()?;
    let Some(input_type) = input_types.first() else {
        return Err(TypeError::invalid("`all_to_all` expects an array followed by its output extents"));
    };
    let input_type = <&ArrayType>::try_from(input_type)?;
    if operation.options.mode == CollectiveMode::Untiled {
        let Some(input_extent) = input_type.shape().dimensions().get(operation.split_axis) else {
            return Err(TypeError::invalid(format!(
                "`all_to_all` split axis {} is out of bounds for rank {}",
                operation.split_axis,
                input_type.rank(),
            )));
        };
        if let Dimension::Static(input_extent) = input_extent
            && *input_extent != effective_axis_size
        {
            return Err(TypeError::invalid(format!(
                "`all_to_all` untiled split axis {} size {input_extent} must equal group size {effective_axis_size}",
                operation.split_axis,
            )));
        }
        let output_type = input_type
            .without_dimension(operation.split_axis)?
            .0
            .with_inserted_dimension(operation.concat_axis, Dimension::Static(effective_axis_size))?;
        let mut output_types = infer_explicit_shape_changing_collective_output_type(
            ALL_TO_ALL_OPERATION_NAME,
            !apply_mesh_axis_semantics,
            input_types,
            output_type,
            &[operation.concat_axis],
            |output_extents| {
                let output_extent = &output_extents[operation.concat_axis];
                if output_extent != &Dimension::Static(effective_axis_size) {
                    return Err(TypeError::invalid(format!(
                        "`all_to_all` inserted output axis {} extent must equal axis group size \
                         {effective_axis_size} but got {output_extent}",
                        operation.concat_axis,
                    )));
                }
                Ok(())
            },
        )?;
        let output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
        return Ok(vec![all_to_all_output_type(operation, input_type, output_type, apply_mesh_axis_semantics)?.into()]);
    }
    if operation.split_axis == operation.concat_axis {
        let Some(input_extent) = input_type.shape().dimensions().get(operation.split_axis) else {
            return Err(TypeError::invalid(format!(
                "`all_to_all` split axis {} is out of bounds for rank {}",
                operation.split_axis,
                input_type.rank(),
            )));
        };
        if let Dimension::Static(input_extent) = input_extent
            && *input_extent % effective_axis_size != 0
        {
            return Err(TypeError::invalid(format!(
                "`all_to_all` split axis {} size {input_extent} is not divisible by group size \
                 {effective_axis_size}",
                operation.split_axis,
            )));
        }
        let mut output_types = infer_explicit_shape_changing_collective_output_type(
            ALL_TO_ALL_OPERATION_NAME,
            !apply_mesh_axis_semantics,
            input_types,
            input_type.clone(),
            &[],
            |_| Ok(()),
        )?;
        let output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
        return Ok(vec![all_to_all_output_type(operation, input_type, output_type, apply_mesh_axis_semantics)?.into()]);
    }
    if operation.split_axis >= input_type.rank() || operation.concat_axis >= input_type.rank() {
        return Err(TypeError::invalid(format!(
            "`all_to_all` split axis {} or concat axis {} is out of bounds for rank {}",
            operation.split_axis,
            operation.concat_axis,
            input_type.rank(),
        )));
    }
    if let Dimension::Static(input_extent) = &input_type.shape().dimensions()[operation.split_axis]
        && *input_extent % effective_axis_size != 0
    {
        return Err(TypeError::invalid(format!(
            "`all_to_all` split axis {} size {input_extent} is not divisible by group size {effective_axis_size}",
            operation.split_axis,
        )));
    }
    let expected_concat_extent = match input_type.shape().dimensions()[operation.concat_axis] {
        Dimension::Static(input_extent) => Some(
            input_extent
                .checked_mul(effective_axis_size)
                .ok_or_else(|| TypeError::invalid("`all_to_all` concatenation result extent does not fit in usize"))?,
        ),
        Dimension::Dynamic(_) => None,
    };
    let mut dimensions = input_type.shape().dimensions().to_vec();
    dimensions[operation.split_axis] = Dimension::Static(0);
    dimensions[operation.concat_axis] = Dimension::Static(0);
    let sharding = input_type.resized_sharding(dimensions.as_slice(), ALL_TO_ALL_OPERATION_NAME)?;
    let mut base_output_type =
        ArrayType::new(input_type.data_type(), Shape::new(dimensions)).with_memory(input_type.memory());
    base_output_type.sharding = sharding;
    let mut output_types = infer_explicit_shape_changing_collective_output_type(
        ALL_TO_ALL_OPERATION_NAME,
        !apply_mesh_axis_semantics,
        input_types,
        base_output_type,
        &[operation.split_axis, operation.concat_axis],
        |output_extents| {
            if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                (&input_type.shape().dimensions()[operation.split_axis], &output_extents[operation.split_axis])
            {
                let expected = *input_extent / effective_axis_size;
                if *output_extent != expected {
                    return Err(TypeError::invalid(format!(
                        "`all_to_all` split result extent must equal input axis {} extent {input_extent} divided by \
                         group size {effective_axis_size}; expected {expected} but got {output_extent}",
                        operation.split_axis,
                    )));
                }
            }
            if let (Some(expected), Dimension::Static(output_extent)) =
                (expected_concat_extent, &output_extents[operation.concat_axis])
            {
                let input_extent = &input_type.shape().dimensions()[operation.concat_axis];
                if *output_extent != expected {
                    return Err(TypeError::invalid(format!(
                        "`all_to_all` concat result extent must equal input axis {} extent {input_extent} multiplied \
                         by group size {effective_axis_size}; expected {expected} but got {output_extent}",
                        operation.concat_axis,
                    )));
                }
            }
            Ok(())
        },
    )?;
    let mut output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
    // Placeholder zeros cannot establish the sharding constraints of the actual result dimensions.
    output_type.sharding = input_type.resized_sharding(output_type.shape().dimensions(), ALL_TO_ALL_OPERATION_NAME)?;
    if output_type.shape() == input_type.shape() {
        output_type = output_type.with_layout(input_type.layout().cloned());
    }
    Ok(vec![all_to_all_output_type(operation, input_type, output_type, apply_mesh_axis_semantics)?.into()])
}

/// Applies the matching-axis all-to-all batching semantics over the policy-selected extent representation.
fn batch_all_to_all_matching_axis<C, P>(
    operation: &AllToAllOperation,
    context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
    input: &ArrayBatch<C::Value>,
    logical_input_rank: usize,
    output_extents: Vec<P::ShapeExtent>,
    output_sharding: Option<Sharding>,
) -> Result<ArrayBatch<C::Value>, BatchingError>
where
    C: Context<Type = ArrayType>,
    C::Value: Transpose,
    P: CollectiveArrayExtentBatchingPolicy<C>,
{
    if operation.options.axis_index_groups.is_some() {
        return Err(BatchingError::UnsupportedOperation {
            message: "`all_to_all` axis index groups are not supported when a batch transform binds the collective \
                      axis"
                .to_string(),
        });
    }
    if operation.split_axis >= logical_input_rank || operation.concat_axis >= logical_input_rank {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "`all_to_all` split axis {} or concat axis {} is out of bounds for rank {logical_input_rank}",
                operation.split_axis, operation.concat_axis,
            ),
        });
    }

    let axis_extent =
        P::collective_axis_extent(context, ALL_TO_ALL_OPERATION_NAME, &operation.axis_name, operation.axis_size)?;

    let (input_extents, chunk_extent) = match operation.options.mode {
        CollectiveMode::Untiled => {
            let mut input_extents = output_extents.clone();
            input_extents.remove(operation.concat_axis);
            input_extents.insert(operation.split_axis, axis_extent.clone());
            (input_extents, P::collective_extent_constant(context, 1)?)
        }
        CollectiveMode::Tiled if operation.split_axis == operation.concat_axis => {
            let axis_extent =
                P::require_divisible_collective_extents(context, &output_extents[operation.split_axis], &axis_extent)?;
            (output_extents.clone(), output_extents[operation.split_axis].div(&axis_extent)?)
        }
        CollectiveMode::Tiled => {
            let axis_extent =
                P::require_divisible_collective_extents(context, &output_extents[operation.concat_axis], &axis_extent)?;
            let mut input_extents = output_extents.clone();
            input_extents[operation.split_axis] = output_extents[operation.split_axis].mul(&axis_extent)?;
            input_extents[operation.concat_axis] = output_extents[operation.concat_axis].div(&axis_extent)?;
            (input_extents, output_extents[operation.split_axis].clone())
        }
    };
    let input = P::match_collective_axis(context, input, input_extents.as_slice())?;
    let mut split_extents = Vec::with_capacity(input_extents.len() + 2);
    split_extents.push(axis_extent.clone());
    split_extents.extend(input_extents.iter().cloned());
    split_extents[operation.split_axis + 1] = axis_extent.clone();
    split_extents.insert(operation.split_axis + 2, chunk_extent);
    let split = P::reshape_collective(context, input.into_value(), split_extents.as_slice(), None)?;
    let exchanged = split.swap_axes(0, operation.split_axis + 1)?;
    let received = match operation.options.mode {
        CollectiveMode::Untiled => {
            let mut squeezed_extents = Vec::with_capacity(input_extents.len() + 1);
            squeezed_extents.push(axis_extent.clone());
            squeezed_extents.extend(input_extents);
            P::reshape_collective(context, exchanged, squeezed_extents.as_slice(), None)?
                .move_axis(operation.split_axis + 1, operation.concat_axis + 1)?
        }
        CollectiveMode::Tiled => exchanged.move_axis(operation.split_axis + 1, operation.concat_axis + 1)?,
    };
    let mut physical_output_extents = Vec::with_capacity(output_extents.len() + 1);
    physical_output_extents.push(axis_extent);
    physical_output_extents.extend(output_extents);
    let physical_output_sharding = output_sharding
        .map(|sharding| sharding.with_leading_batch_axis(context.axis_sharding().clone()))
        .transpose()?;
    let output =
        P::reshape_collective(context, received, physical_output_extents.as_slice(), physical_output_sharding)?;
    ArrayBatch::new(output, BatchAxis::from_position(0))
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayType,
        DataType, Dimension, DimensionBounds, DimensionType, DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis,
        MeshAxisType, RaggedAxis, Shape, ShardingDimension, StridedLayout,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingContext, BatchingTracer, batch};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{check_gradient, check_operation_type_inference};
    use crate::operations::manipulation::slicing::Slice;
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::partial::{
        PartialEvaluationContext, PartialEvaluationOutput, PartialEvaluationValue, PartialValue,
        PartiallyEvaluatableOperation,
    };
    use crate::programs::{EmptyRegionDriver, Program, ProgramBuilder, ProgramError};
    use crate::tracing::TracingContext;

    use super::*;

    /// Builds a single-instruction homogeneous all-to-all program.
    fn all_to_all_program(
        operation: AllToAllOperation,
        input_type: ArrayType,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type);
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap().to_vec();
        builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Applies the matching-axis rule under an eager two-participant batch named `"x"`.
    fn batch_all_to_all(
        operation: &AllToAllOperation,
        input: ArrayBatch<Array>,
    ) -> Result<Vec<ArrayBatch<Array>>, BatchingError> {
        let context = BatchingContext::<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>::new(
            EagerContext::new(),
            2,
        )
        .with_axis_name("x".to_string());
        Ok(operation.batch(&context, &EmptyRegionDriver, &[input])?.into_parts().0)
    }

    #[test]
    fn test_all_to_all() {
        let operation = AllToAllOperation::new("x".to_string(), 4, 0, 1, CollectiveOptions::tiled());
        assert_eq!(operation.name(), ALL_TO_ALL_OPERATION_NAME);
        assert_eq!(operation.split_axis(), 0);
        assert_eq!(operation.concat_axis(), 1);
        assert_eq!(operation.options(), &CollectiveOptions::tiled());
        assert_eq!(operation.effective_axis_size(), Ok(4));
        let grouped = AllToAllOperation::new(
            "x".to_string(),
            4,
            1,
            0,
            CollectiveOptions::default().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
        );
        assert_eq!(grouped.effective_axis_size(), Ok(2));
    }

    #[test]
    fn test_all_to_all_type_inference() {
        check_operation_type_inference!(
            operation = AllToAllOperation::new("x".to_string(), 4, 0, 1, CollectiveOptions::tiled()),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [8, 3])],
                    output_types = [ArrayType::new_static(DataType::F32, [2, 12])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [6, 3])],
                    error = "`all_to_all` split axis 0 size 6 is not divisible by group size 4",
                },
            ],
        );
        check_operation_type_inference!(
            operation = AllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::default()),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::Boolean, [3, 2])],
                    output_types = [ArrayType::new_static(DataType::Boolean, [2, 3])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [3, 4])],
                    error = "`all_to_all` untiled split axis 1 size 4 must equal group size 2",
                },
            ],
        );
        check_operation_type_inference!(
            operation = AllToAllOperation::new("x".to_string(), 4, 0, 1,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]])),
            cases = [{
                input_types = [ArrayType::new_static(DataType::C64, [6, 3])],
                output_types = [ArrayType::new_static(DataType::C64, [3, 6])],
            }],
        );
    }

    #[test]
    fn test_all_to_all_type_inference_explicit_extents() {
        let operation = AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
        let split = DimensionVariable::new("split", DimensionBounds::unbounded());
        let concat = DimensionVariable::new("concat", DimensionBounds::unbounded());
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &operation,
                &[
                    ArrayType::new_static(DataType::F32, [4, 3]).into(),
                    DimensionType::from(split.clone()).into(),
                    DimensionType::from(concat.clone()).into(),
                ]
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![split.clone().into(), concat.clone().into()])).into()
            ]),
        );

        // A known invalid split extent is rejected even when both result extents are dynamic.
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &operation,
                &[
                    ArrayType::new_static(DataType::F32, [3, 3]).into(),
                    DimensionType::from(split.clone()).into(),
                    DimensionType::from(concat.clone()).into(),
                ]
            ),
            Err(TypeError::invalid("`all_to_all` split axis 0 size 3 is not divisible by group size 2")),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &operation,
                &[
                    ArrayType::new_static(DataType::F32, [4, usize::MAX]).into(),
                    DimensionType::from(split).into(),
                    DimensionType::from(concat).into(),
                ]
            ),
            Err(TypeError::invalid("`all_to_all` concatenation result extent does not fit in usize")),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &operation,
                &[
                    ArrayType::new_static(DataType::F32, [4, 3]).into(),
                    DimensionValue::constant(1).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(6).unwrap().r#type().into_owned().into(),
                ]
            ),
            Err(TypeError::invalid(
                "`all_to_all` split result extent must equal input axis 0 extent 4 divided by group size 2; \
                 expected 2 but got 1",
            )),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &operation,
                &[
                    ArrayType::new_static(DataType::F32, [4, 3]).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(5).unwrap().r#type().into_owned().into(),
                ]
            ),
            Err(TypeError::invalid(
                "`all_to_all` concat result extent must equal input axis 1 extent 3 multiplied by group size 2; \
                 expected 6 but got 5",
            )),
        );

        // Untiled geometry retains unaffected dynamic dimensions and inserts a statically known sender axis.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::default()),
                &[
                    ArrayType::new(DataType::F32, Shape::new(vec![length.clone().into(), 2.into()])).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionType::from(length.clone()).into(),
                ],
            ),
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![2.into(), length.into()])).into()]),
        );
    }

    #[test]
    fn test_all_to_all_type_inference_metadata() {
        let input = ArrayType::new_static(DataType::F32, [4, 3])
            .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])))
            .with_memory(Memory::Host { pinned: true });
        let operation = AllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::tiled());
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input), &[]), Ok(vec![input.clone()]));
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &operation,
                &[
                    input.clone().into(),
                    DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ]
            ),
            Ok(vec![input.into()]),
        );

        // Sharding must fit the actual split result, rather than the placeholder zero used during inference.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["devices"]), ShardingDimension::Replicated])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()),
                &[
                    input.into(),
                    DimensionValue::constant(1).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(6).unwrap().r#type().into_owned().into(),
                ],
            ),
            Err(TypeError::invalid(
                "`all_to_all` on a dimension sharded over explicit mesh axes requires the output size (1) at axis 0 \
                 to be divisible by the mesh-axis product (2)",
            )),
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2);
        let input = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(sharding.clone()).unwrap();
        let operation = AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [{
                input_types = [input],
                error = "`all_to_all` input must vary over manual axis `x`; pass an invariant value through \
                         `parallel_vary` first so that the exchanged output is typed as varying",
            }],
        );
        let varying = sharding.clone().with_varying_manual_axes(["x"]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(varying.clone()).unwrap();
        let output = ArrayType::new_static(DataType::F32, [1, 6]).with_sharding(varying).unwrap();
        assert_eq!(operation.infer_output_types(&[input], &[]), Ok(vec![output]));
        check_operation_type_inference!(
            operation = operation,
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [2, 3])
                    .with_sharding(sharding.with_unreduced_axes(["x"]).unwrap()).unwrap()],
                error = "`all_to_all` does not support unreduced inputs",
            }],
        );
    }

    #[test]
    fn test_all_to_all_interpretation() {
        // A single participant exchanges chunks only with itself, so tiled mode is the identity, while untiled mode
        // removes the size-one split axis and inserts a size-one concatenation axis. Any larger axis has no per-item
        // semantics outside an enclosing binder.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let input = Array::matrix(1, 2, vec![1.0, 2.0]).unwrap();
        assert_eq!(
            AllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::tiled()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            AllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::default()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![Array::matrix(2, 1, vec![1.0, 2.0]).unwrap()]),
        );
        assert_eq!(
            AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot interpret `all_to_all` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_all_to_all_partial_evaluation() {
        // A degenerate untiled exchange folds a known input through the corresponding rank-preserving reshape.
        let input = Array::matrix(1, 3, vec![1.0f32, 2.0, 3.0]).unwrap();
        let program = all_to_all_program(
            AllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::default()),
            ArrayType::new_static(DataType::F32, [1, 3]),
        );
        assert_eq!(
            program.partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap().outputs(),
            &[PartialEvaluationOutput::Known(Array::matrix(3, 1, vec![1.0f32, 2.0, 3.0]).unwrap())],
        );

        // An exchange with other participants has no eager per-item value and therefore residualizes.
        let operation = AllToAllOperation::new("x".to_string(), 3, 1, 0, CollectiveOptions::tiled());
        let program = all_to_all_program(operation.clone(), ArrayType::new_static(DataType::F32, [1, 3]));
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input)]).unwrap();
        assert!(evaluation.outputs()[0].is_unknown());
        assert_eq!(evaluation.program().to_string(), program.to_string());

        // A staging parent can retain a known tracer by staging the collective in the parent trace.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = PartialEvaluationValue::known(trace.input(ArrayType::new_static(DataType::F32, [1, 3])));
        let outputs = operation
            .partially_evaluate(&PartialEvaluationContext::new(trace), &EmptyRegionDriver, &[input])
            .unwrap();
        assert!(outputs[0].is_known());
        assert_eq!(outputs[0].r#type().as_ref(), &ArrayType::new_static(DataType::F32, [3, 1]));
    }

    #[test]
    fn test_all_to_all_batching() {
        // Replicated senders still route destination-specific chunks: each destination receives its chunk twice.
        let tiled = AllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled());
        let output = batch_all_to_all(&tiled, ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()))
            .unwrap()
            .remove(0);
        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        assert_eq!(output.value(), &Array::matrix(2, 4, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]).unwrap());

        // Removing the split axis and inserting the sender axis must work on either side of the other array axis.
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [2, 2, 3]),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let output = batch_all_to_all(
            &AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::default()),
            ArrayBatch::new(input, BatchAxis::new(0)).unwrap(),
        )
        .unwrap()
        .remove(0);
        assert_eq!(output.value().r#type().as_ref(), &ArrayType::new_static(DataType::F64, [2, 3, 2]));
        assert_eq!(output.value().to_f64s(), vec![1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0]);

        // Here the physical mapped axis follows both logical array axes, and the split axis follows the concat axis.
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [3, 2, 2]),
            &[1.0, 7.0, 4.0, 10.0, 2.0, 8.0, 5.0, 11.0, 3.0, 9.0, 6.0, 12.0],
        )
        .unwrap();
        let output = batch_all_to_all(
            &AllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::default()),
            ArrayBatch::new(input, BatchAxis::new(2)).unwrap(),
        )
        .unwrap()
        .remove(0);
        assert_eq!(output.value().r#type().as_ref(), &ArrayType::new_static(DataType::F64, [2, 2, 3]));
        assert_eq!(output.value().to_f64s(), vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0]);

        // Participant groups belong to a mesh exchange and are unsupported when this batch binds the named axis.
        assert_eq!(
            batch_all_to_all(
                &AllToAllOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    0,
                    CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]])
                ),
                ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]).unwrap()),
            ),
            Err(BatchingError::UnsupportedOperation {
                message:
                    "`all_to_all` axis index groups are not supported when a batch transform binds the collective axis"
                        .to_string(),
            }),
        );
    }

    #[test]
    fn test_all_to_all_batching_same_axes() {
        // Block exchange with `split_axis == concat_axis == 0`: each item splits its vector into two chunks and
        // receives its own chunk index from every item, concatenated item-major. With items `[1, 2, 3, 4]` and
        // `[5, 6, 7, 8]`, item 0 receives `[1, 2, 5, 6]` and item 1 receives `[3, 4, 7, 8]`, matching the verified
        // cross-device `shard_map` execution semantics of StableHLO's `all_to_all`.
        let x = Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let output: ArrayIrValue<Array> = batch(
            |item: BatchingTracer<
                EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                ArrayIrBatchingPolicy,
            >| { item.all_to_all_tiled("x", 0, 0) },
            ArrayIrValue::Array(x),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayIrType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]),
            )),
        );
        let ArrayIrValue::Array(output) = output else {
            panic!("`all_to_all` must preserve the array member kind");
        };
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_all_to_all_batching_distinct_axes() {
        // Distinct split and concatenation axes over per-item `[2, 2]` matrices: each item splits its rows across
        // the items and receives its own row index from every item, concatenated item-major along the columns. With
        // item 0 = `[[1, 2], [3, 4]]` and item 1 = `[[5, 6], [7, 8]]`, item 0 receives `[[1, 2, 5, 6]]` and item 1
        // receives `[[3, 4, 7, 8]]` (per-item shape `[1, 4]`).
        let x = Array::from_elements::<f64>(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
            ),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let output: ArrayIrValue<Array> = batch(
            |item: BatchingTracer<
                EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                ArrayIrBatchingPolicy,
            >| { item.all_to_all_tiled("x", 0, 1) },
            ArrayIrValue::Array(x),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayIrType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(1), Dimension::Static(4)]),
            )),
        );
        let ArrayIrValue::Array(output) = output else {
            panic!("`all_to_all` must preserve the array member kind");
        };
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_all_to_all_batching_forwards_untiled_axes() {
        // A non-matching batch level shifts both array axes around its mapped dimension. Removing the split axis
        // can move that mapped dimension, and insertion at the concat axis can move it a second time.
        let context = BatchingContext::<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>::new(
            EagerContext::new(),
            2,
        )
        .with_axis_name("items".to_string());
        let operation = AllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::default());
        for (input_shape, input_batch_axis, output_shape, output_batch_axis) in
            [([2, 1, 3], 0, [2, 3, 1], 0), ([1, 2, 3], 1, [2, 3, 1], 0), ([1, 3, 2], 2, [3, 1, 2], 2)]
        {
            let input = Array::from_elements::<f32>(
                ArrayType::new_static(DataType::F32, input_shape),
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .unwrap();
            let output = operation
                .batch(
                    &context,
                    &EmptyRegionDriver,
                    &[ArrayBatch::new(input, BatchAxis::new(input_batch_axis)).unwrap()],
                )
                .unwrap()
                .into_parts()
                .0
                .remove(0);
            assert_eq!(output.batch_axis(), BatchAxis::new(output_batch_axis));
            assert_eq!(output.value().r#type().as_ref(), &ArrayType::new_static(DataType::F32, output_shape));
            assert_eq!(output.value().to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
    }

    #[test]
    fn test_all_to_all_batching_shadows_manual_axis() {
        // The inner batch named `x` exchanges local chunks independently of the manual mesh axis also named `x`.
        // Local exchange preserves the manual mesh variance and any pending mesh reductions.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 3);
        for sharding in [sharding.clone(), sharding.with_unreduced_axes(["x"]).unwrap()] {
            let input_type = ArrayType::new_static(DataType::F32, [2, 2, 3]).with_sharding(sharding.clone()).unwrap();
            let expected_type = ArrayType::new_static(DataType::F32, [2, 1, 6]).with_sharding(sharding).unwrap();
            let (output_type, _) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| {
                    let context = BatchingContext::<_, ArrayBatchingPolicy>::new(input.dispatch_domain(), 2)
                        .with_axis_name("x".to_string());
                    let input = ArrayBatch::new(input, BatchAxis::new(0))?;
                    let operation = AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
                    let mut outputs = operation.batch(&context, &EmptyRegionDriver, &[input])?.into_parts().0;
                    Ok(outputs.remove(0).into_value())
                },
                input_type.clone(),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap();
            assert_eq!(output_type, expected_type);

            let (output_type, _) =
                TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                    |input| {
                        batch(
                            |item| item.all_to_all_tiled("x", 0, 1),
                            input,
                            BatchAxis::new(0),
                            BatchAxis::new(0),
                            BatchAxisSpecification::named("x"),
                        )
                        .map_err(Into::into)
                    },
                    ArrayIrType::Array(input_type),
                    vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
                )
                .unwrap();
            assert_eq!(output_type, ArrayIrType::Array(expected_type));
        }
    }

    #[test]
    fn test_all_to_all_batching_shadows_manual_axis_through_unrelated_batch() {
        // The unrelated inner batch forwards to the outer batch named `x`. That outer batch shadows the mesh axis,
        // so forwarding must leave mesh validation to the level that handles the exchange.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 4);
        for sharding in [sharding.clone(), sharding.with_unreduced_axes(["x"]).unwrap()] {
            let input_type =
                ArrayType::new_static(DataType::F32, [2, 2, 2, 3]).with_sharding(sharding.clone()).unwrap();
            let expected_type = ArrayType::new_static(DataType::F32, [2, 2, 1, 6]).with_sharding(sharding).unwrap();
            let (output_type, _) =
                TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                    |input| {
                        batch(
                            |item| {
                                batch(
                                    |item| item.all_to_all_tiled("x", 0, 1),
                                    item,
                                    BatchAxis::new(0),
                                    BatchAxis::new(0),
                                    BatchAxisSpecification::named("y"),
                                )
                                .map_err(Into::into)
                            },
                            input,
                            BatchAxis::new(0),
                            BatchAxis::new(0),
                            BatchAxisSpecification::named("x"),
                        )
                        .map_err(Into::into)
                    },
                    ArrayIrType::Array(input_type),
                    vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
                )
                .unwrap();
            assert_eq!(output_type, ArrayIrType::Array(expected_type));
        }
    }

    #[test]
    fn test_all_to_all_batching_forwards_manual_axis_validation() {
        // When no matching batch shadows `x`, the parent's mesh binding must still reject invariant inputs and
        // pending mesh sums. Bind the operation directly so the capability cannot insert `parallel_vary` first.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 3);
        for (sharding, expected_message) in [
            (
                sharding.clone(),
                "`all_to_all` input must vary over manual axis `x`; pass an invariant value through `parallel_vary` \
                 first so that the exchanged output is typed as varying",
            ),
            (sharding.with_unreduced_axes(["x"]).unwrap(), "`all_to_all` does not support unreduced inputs"),
        ] {
            let input_type = ArrayType::new_static(DataType::F32, [2, 2, 3]).with_sharding(sharding).unwrap();
            let error = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| {
                    let parent = input.dispatch_domain();
                    let context =
                        BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), parent.dimension_constant(2)?)
                            .with_axis_name("y".to_string());
                    let input = ArrayIrBatch::new(input, BatchAxis::new(0))?;
                    let split_extent = ArrayIrBatch::replicated(parent.dimension_constant(1)?);
                    let concat_extent = ArrayIrBatch::replicated(parent.dimension_constant(6)?);
                    let operation = AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
                    let mut outputs = operation
                        .batch_in_parent(&context, &EmptyRegionDriver, &[input, split_extent, concat_extent])?
                        .into_parts()
                        .0;
                    Ok(outputs.remove(0).into_value())
                },
                ArrayIrType::Array(input_type),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap_err();
            assert!(matches!(
                error,
                ProgramError::Type(TypeError::Invalid { message }) if message == expected_message,
            ));
        }
    }

    #[test]
    fn test_all_to_all_batching_ragged() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 4, vec![1.0f32; 8]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(
                1,
                Array::vector(vec![2i32, 4]).unwrap(),
                variable.clone(),
                vec![0],
            )])
            .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());

        assert_eq!(
            AllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled()).batch(
                &context,
                &EmptyRegionDriver,
                &[input],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`all_to_all` cannot route bounded ragged dimension `length` without explicit \
                          per-destination offsets and sizes; use `ragged_all_to_all`"
                    .to_string(),
            }),
        );

        let extents = ArrayIrValue::Array(Array::vector(vec![2i32, 4]).unwrap());
        let input =
            ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0f32; 8]).unwrap()), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
                .unwrap();
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("x".to_string());
        let output_extent =
            ArrayIrBatch::mapped_dimension(extents, BatchAxis::new(0), DimensionType::from(variable)).unwrap();
        assert_eq!(
            AllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled()).batch_in_parent(
                &context,
                &EmptyRegionDriver,
                &[input, output_extent],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`all_to_all` cannot route bounded ragged dimension `length` without explicit \
                          per-destination offsets and sizes; use `ragged_all_to_all`"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_all_to_all_batching_dynamic_extents() -> Result<(), ProgramError> {
        // Distinct-axis all-to-all derives its temporary pre-exchange shape from the supplied result extents and the
        // mapped extent using ordinary dimension arithmetic; it never reads the source array shape.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::from(batch.clone()).into());
        let input_split = DimensionVariable::new("input_split", DimensionBounds::new(1, Some(65))?);
        let input_concat = DimensionVariable::new("input_concat", DimensionBounds::new(1, Some(65))?);
        let output_split = DimensionVariable::new("output_split", DimensionBounds::new(1, Some(65))?);
        let output_concat = DimensionVariable::new("output_concat", DimensionBounds::new(1, Some(129))?);
        let input = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![
                    Dimension::Dynamic(batch),
                    Dimension::Dynamic(input_split),
                    Dimension::Dynamic(input_concat),
                ]),
            )
            .into(),
        );
        let output_split = trace.input(DimensionType::from(output_split).into());
        let output_concat = trace.input(DimensionType::from(output_concat).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent)
            .with_axis_name("items".to_string());
        let [output] = context
            .bind(
                ArrayIrOperation::AllToAll(AllToAllOperation::new(
                    "items".to_string(),
                    4,
                    0,
                    1,
                    CollectiveOptions::tiled(),
                )),
                Vec::new(),
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(input, BatchAxis::new(0))?),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(output_split)),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(output_concat)),
                ],
            )?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output.batch().value().atom_id()?],
            vec![Placeholder; 4],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, input_split, input_concat], \
                    %2:dimension<output_split ∈ [1, 65)>, %3:dimension<output_concat ∈ [1, 129)> .
                let %4:dimension<4> = constant [value=4]
                    %5:bool[] = compare [direction=Equal] %0 %4
                    () = assert [
                        message=\"collective axis extent must match the participant count\",
                        labels=[\"extent\", \"participants\"],
                    ] %5 %0 %4
                    %6:dimension<0> = constant [value=0]
                    %7:dimension<output_concat % batch ∈ [0, 8)> = dimension_rem %3 %0
                    %8:bool[] = compare [direction=Equal] %7 %6
                    () = assert [
                        message=\"collective extent must be divisible by the participant count\",
                        labels=[\"extent\", \"divisor\"],
                    ] %8 %3 %0
                    %9:dimension<output_split * batch ∈ [1, 513)> = dimension_mul %2 %0
                    %10:dimension<output_concat / batch ∈ [0, 129)> = dimension_div %3 %0
                    %11:f32[batch, batch, output_split, output_concat / batch] = reshape %1 %0 %0 %2 %10
                    %12:f32[batch, batch, output_split, output_concat / batch] = transpose [permutation=[1, 0, 2, 3]] %11
                    %13:f32[batch, output_split, batch, output_concat / batch] = transpose [permutation=[0, 2, 1, 3]] %12
                    %14:f32[batch, output_split, output_concat] = reshape %13 %0 %2 %3
                in (%14)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_all_to_all_differentiation() {
        // The homogeneous JVP applies exactly the same exchange to the primal and live tangent.
        let program = all_to_all_program(
            AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()),
            ArrayType::new_static(DataType::F32, [4, 3]),
        );
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.instructions().len(), 2);
        assert_eq!(jvp.instructions()[0].operation(), jvp.instructions()[1].operation());
        assert!(matches!(jvp.instructions()[0].operation(), ArrayOperation::AllToAll(_)));
        assert_eq!(jvp.output_types(), vec![ArrayType::new_static(DataType::F32, [2, 6]); 2]);

        // Distinct destination and sender weights catch an incorrectly ordered exchange or pullback. The physical
        // mapped axis follows the split and concat axes, so differentiation must also retain their logical positions.
        check_gradient!(
            |inputs| {
                let exchanged = batch(
                    |item| {
                        let operation = AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
                        let mut outputs = item.dispatch_domain().bind(operation, Vec::new(), &[item])?;
                        Ok::<_, ProgramError>(outputs.remove(0))
                    },
                    inputs,
                    BatchAxis::new(2),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?;
                let first = exchanged.slice(&[0, 0, 0], &[1, 1, 2], &[1, 1, 1])?;
                let second = exchanged.slice(&[0, 0, 2], &[1, 1, 4], &[1, 1, 1])?;
                let third = exchanged.slice(&[1, 0, 0], &[2, 1, 2], &[1, 1, 1])?;
                let fourth = exchanged.slice(&[1, 0, 2], &[2, 1, 4], &[1, 1, 1])?;
                let weighted = first
                    + second.clone()
                    + second
                    + third.clone()
                    + third.clone()
                    + third
                    + fourth.clone()
                    + fourth.clone()
                    + fourth.clone()
                    + fourth;
                weighted.reduce(&[0, 1, 2], ReductionKind::Sum)
            },
            at = Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [2, 2, 2]),
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
            .unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );

        // A composite untiled exchange retains its dynamic input extent so the pullback restores the original axes.
        let variable = DimensionVariable::new("length", DimensionBounds::new(1, Some(9)).unwrap());
        let dimension_type = DimensionType::from(variable.clone());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable), Dimension::Static(1)])).into(),
        );
        let one = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()));
        let extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                AllToAllOperation::new("x".to_string(), 1, 1, 0, CollectiveOptions::default()),
                Vec::new(),
                vec![array, one, extent],
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
        let input = ArrayIrValue::Array(Array::matrix(3, 1, vec![1.0f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::matrix(1, 3, vec![1.0f32, 2.0, 3.0]).unwrap()));
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::matrix(1, 3, vec![4.0f32, 5.0, 6.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(3, 1, vec![4.0f32, 5.0, 6.0]).unwrap())]),
        );
    }

    #[test]
    fn test_all_to_all_differentiation_with_zero_tangent() {
        // Structural zeros use the inferred result shape and stage no linear call or tangent exchange.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let first = trace.input(DimensionValue::constant(3).unwrap().r#type().into_owned().into());
        let second = trace.input(DimensionValue::constant(2).unwrap().r#type().into_owned().into());
        let inputs = [input, first, second]
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let output = AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::default())
            .jvp_in_parent(&DifferentiationContext::fused(trace.clone()), &EmptyRegionDriver, &inputs)
            .unwrap()
            .remove(0);
        assert!(output.tangent().is_zero());
        assert_eq!(
            output.primal().r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 2]))
        );
        assert_eq!(trace.builder().borrow().instructions().len(), 1);
        assert!(matches!(trace.builder().borrow().instructions()[0].operation(), ArrayIrOperation::AllToAll(_)));
    }

    #[test]
    fn test_all_to_all_differentiation_shadows_manual_axis() {
        // An inner batch named `x` performs a local exchange despite an enclosing mesh axis with that name.
        // Differentiation must retain structural zeros and preserve invariant or unreduced mesh state.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 3);
        for sharding in [sharding.clone(), sharding.with_unreduced_axes(["x"]).unwrap()] {
            let input_type = ArrayType::new_static(DataType::F32, [2, 2, 3]).with_sharding(sharding.clone()).unwrap();
            let expected_type = ArrayType::new_static(DataType::F32, [2, 1, 6]).with_sharding(sharding).unwrap();
            let (output_type, _) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| {
                    let batch_context = BatchingContext::<_, ArrayBatchingPolicy>::new(input.dispatch_domain(), 2)
                        .with_axis_name("x".to_string());
                    let item = BatchingTracer::new(batch_context.clone(), ArrayBatch::new(input, BatchAxis::new(0))?);
                    let context = DifferentiationContext::fused(batch_context);
                    let item =
                        DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(item)?, context.clone());
                    let operation = AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
                    let mut outputs = context.bind(operation, Vec::new(), &[item])?;
                    let output = outputs.remove(0);
                    assert!(output.tangent().is_zero());
                    Ok(output.primal().clone().into_batch().into_value())
                },
                input_type.clone(),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap();
            assert_eq!(output_type, expected_type);

            // The composite capability stages result extents with structural-zero tangents and uses the same binder.
            let (output_type, _) =
                TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                    |input| {
                        let axis_extent = input.dispatch_domain().dimension_constant(2)?;
                        let batch_context =
                            BatchingContext::<_, ArrayIrBatchingPolicy>::new(input.dispatch_domain(), axis_extent)
                                .with_axis_name("x".to_string());
                        let item =
                            BatchingTracer::new(batch_context.clone(), ArrayIrBatch::new(input, BatchAxis::new(0))?);
                        let context = DifferentiationContext::fused(batch_context);
                        let item =
                            DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(item)?, context);
                        let output = item.all_to_all_tiled("x", 0, 1)?;
                        assert!(output.tangent().is_zero());
                        Ok(output.primal().clone().into_batch().into_value())
                    },
                    ArrayIrType::Array(input_type),
                    vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
                )
                .unwrap();
            assert_eq!(output_type, ArrayIrType::Array(expected_type));
        }
    }

    #[test]
    fn test_all_to_all_transposition() {
        // Both modes invert by swapping the split and concat axes, retaining the participant grouping.
        for options in [
            CollectiveOptions::tiled(),
            CollectiveOptions::default(),
            CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
            CollectiveOptions::default().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
        ] {
            let axis_size = if options.axis_index_groups().is_some() { 4 } else { 2 };
            let program = all_to_all_program(
                AllToAllOperation::new("x".to_string(), axis_size, 0, 1, options.clone()),
                ArrayType::new_static(DataType::F32, [2, 3]),
            );
            let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
            assert_eq!(transposed.instructions().len(), 1);
            assert_eq!(
                transposed.instructions()[0].operation(),
                &ArrayOperation::AllToAll(AllToAllOperation::new("x".to_string(), axis_size, 1, 0, options)),
            );
            assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());
        }
    }

    #[test]
    fn test_parallel_swap_axes_parallel_swap_axes() {
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_swap_axes("x", 0),
            ArrayIrType::Array(input_type),
            vec![(
                "x".to_string(),
                NamedAxis::Mesh {
                    mesh: LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
                    axis: 0,
                    size: 2,
                },
            )],
        )
        .unwrap();

        let all_to_all = program.instructions().last().unwrap();
        let ArrayIrOperation::AllToAll(operation) = all_to_all.operation() else {
            panic!("parallel_swap_axes must compose the canonical all-to-all operation");
        };
        assert_eq!(operation.split_axis(), 0);
        assert_eq!(operation.concat_axis(), 0);
        assert_eq!(operation.options(), &CollectiveOptions::default());
        assert_eq!(all_to_all.inputs().len(), 3);
        let variation = program
            .instructions()
            .iter()
            .find(|instruction| {
                matches!(instruction.operation(), ArrayIrOperation::Array(ArrayOperation::ParallelVary(_)))
            })
            .unwrap();
        assert_eq!(all_to_all.inputs()[0], variation.outputs()[0]);
        let output_type = program.output_types().remove(0);
        assert!(
            <&ArrayType>::try_from(&output_type)
                .unwrap()
                .sharding()
                .unwrap()
                .varying_manual_axes()
                .contains("x")
        );
    }
}
