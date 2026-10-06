use std::fmt::Display;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType,
    DataType, Dimension, DimensionOperation, DimensionType, DimensionValue, DimensionVariable, LogicalMesh, Shape,
    Sharding,
};
use crate::axes::{AxisError, NamedAxes, NamedAxis};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    MemberBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy, MemberDifferentiableOperation, TransposableOperation,
    TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver, MemberInterpretableOperation};
use crate::macros::check_count;
use crate::operations::Capability;
use crate::operations::arithmetic::{AddOperation, Div, Mul, Rem};
use crate::operations::assertions::Assert;
use crate::operations::collectives::parallel_all_gather::{
    ParallelAllGatherOperation, ParallelAllGatherOutputVariance,
};
use crate::operations::collectives::parallel_vary::{PARALLEL_VARY_OPERATION_NAME, ParallelVary};
use crate::operations::collectives::{
    CollectiveArrayExtentBatchingPolicy, CollectiveMode, CollectiveOptions, LinearCollectiveOperation,
    ShapeChangingCollectiveBatching, ShapeChangingCollectiveOperation, ShapeChangingCollectiveValue,
    infer_array_ir_shape_changing_collective_output_type, infer_linear_collective_operation_output_type,
    resolve_named_axis_size, validate_manual_mesh_input,
};
use crate::operations::comparisons::Compare;
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_max::DimensionMax;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{DynamicBroadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::reductions::{Reduce, ReductionKind};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, MemberOperation, Operation, OperationFormatter, OperationProjection, ProgramError, ProjectedValue,
    RegionInterface, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`ParallelSumScatterOperation`].
pub const PARALLEL_SUM_SCATTER_OPERATION_NAME: &str = "parallel_sum_scatter";

/// [`Operation`] that sums every participant's input across the named axis and scatters the sum along `scatter_axis`,
/// so that every participant receives only its own chunk of the sum. This is the analogue of JAX's
/// [`jax.lax.psum_scatter`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.psum_scatter.html) and of StableHLO's
/// [`reduce_scatter`](https://openxla.org/stablehlo/spec#reduce_scatter) with a sum reduction. The [`CollectiveMode`]
/// of its options selects the output shape over a group of `n` participants:
///
///   - [`CollectiveMode::Untiled`] requires `scatter_axis` to have extent `n` and removes it, so that participant
///     `i` receives row `i` of the sum.
///   - [`CollectiveMode::Tiled`] requires the extent of `scatter_axis` to be divisible by `n` and divides it by
///     `n`, so that participant `i` receives the `i`-th contiguous chunk of the sum.
///
/// Inputs must be numeric (or structural zeros). Cross-device sums accumulate in the input element type. A matching
/// `batch` level uses [`ReductionKind::Sum`] along its local participant axis; when interpreted eagerly on [`Array`],
/// this widens narrow floating-point accumulation to `f32` before rounding the result back to the input element type.
/// Its rounding can therefore differ from cross-device execution. Participant groups (refer to [`CollectiveOptions`])
/// restrict the sum and the scatter to each group.
///
/// A sum-scatter over a manual mesh axis is created by [`with_mesh`](Self::with_mesh), and
/// [`ParallelSumScatter::parallel_sum_scatter_with_options`] supplies the mesh automatically from the enclosing manual
/// region. Every participant of such a sum-scatter receives a different chunk, so its input must vary over the axis
/// (refer to [`ParallelVary`]) and its output varies over it as well. An input that is instead unreduced over the
/// operation's own axis (e.g., the cotangent of a reduced [`ParallelAllGatherOperation`] result) has its pending
/// cross-device sum completed by the exchange, and its output varies over the axis too. An ordinary sum-scatter
/// carries no mesh and preserves the input's mesh state, even when its input carries a manual mesh axis with the same
/// name, because a `batch` level whose axis name shadows that mesh axis may bind it instead. The collective is linear,
/// and its transpose is a [`ParallelAllGatherOperation`] with the same mode, axis, participant groups, and mesh.
/// The gather is reduced when the input is unreduced over the scattered manual axis, and varying otherwise. Pending
/// sums over unrelated mesh axes are preserved. Outside any binder, the single participant of a degenerate axis keeps
/// its value, with the size-one scatter axis removed in untiled mode.
///
/// A matching `batch` level consumes the mapped batch axis of an ordinary sum-scatter by summing over it and mapping
/// the scattered chunks back onto it, so that batch item `i` receives chunk `i` of the sum and a value that is the
/// same for every item counts once per item. A matching level rejects participant groups and sum-scatters over a
/// manual mesh axis, and every `batch` level rejects bounded ragged inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelSumScatterOperation {
    /// Axis name referenced by this collective.
    axis_name: String,

    /// Number of participants along the named axis, resolved from the active [`NamedAxes`] environment
    /// when the operation is staged.
    axis_size: usize,

    /// Axis of the input along which the summed result is scattered across the participants.
    scatter_axis: usize,

    /// Shared rank and participant-group semantics.
    options: CollectiveOptions,

    /// Refer to the documentation of [`mesh`](Self::mesh) for more information.
    mesh: Option<LogicalMesh>,
}

impl ParallelSumScatterOperation {
    /// Creates a new [`ParallelSumScatterOperation`] over the axis with the provided name and resolved axis size.
    #[inline]
    pub fn new(axis_name: String, axis_size: usize, scatter_axis: usize, options: CollectiveOptions) -> Self {
        Self { axis_name, axis_size, scatter_axis, options, mesh: None }
    }

    /// Returns this [`ParallelSumScatterOperation`] configured to sum and scatter over a manual axis of `mesh`. The
    /// input must vary over [`axis_name`](Self::axis_name) on that mesh, or be unreduced over it, and the axis size
    /// must equal [`axis_size`](Self::axis_size). Type inference validates these requirements.
    /// [`ParallelSumScatter::parallel_sum_scatter_with_options`] supplies the mesh automatically
    /// from the enclosing manual region.
    #[inline]
    pub fn with_mesh(mut self, mesh: LogicalMesh) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Returns the axis name referenced by this collective.
    #[inline]
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }

    /// Returns the number of participants along the named axis.
    #[inline]
    pub fn axis_size(&self) -> usize {
        self.axis_size
    }

    /// Returns the axis of the input along which the summed result is scattered across the participants.
    #[inline]
    pub fn scatter_axis(&self) -> usize {
        self.scatter_axis
    }

    /// Returns the shared rank and participant-group semantics.
    #[inline]
    pub fn options(&self) -> &CollectiveOptions {
        &self.options
    }

    /// Returns the logical mesh whose manual axis this [`ParallelSumScatterOperation`] sums and scatters over, or
    /// [`None`] for an ordinary sum-scatter, whose named axis may be bound by any enclosing binder. Only a sum-scatter
    /// over a manual mesh axis validates and updates the manual variation and pending sums of its input.
    #[inline]
    pub fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

    /// Infers the statically shaped output, applying mesh-axis reduction and variance semantics to a sum-scatter
    /// over a manual mesh axis.
    fn infer_static_output_type(&self, input_type: &ArrayType, dimensions: Vec<usize>) -> Result<ArrayType, TypeError> {
        let effective_axis_size = self.effective_axis_size()?;
        let output_type = match self.options.mode {
            CollectiveMode::Untiled => {
                let Some(dimension) = dimensions.get(self.scatter_axis) else {
                    return Err(TypeError::invalid(format!(
                        "`{}` scatter axis {} is out of bounds for rank {}",
                        PARALLEL_SUM_SCATTER_OPERATION_NAME,
                        self.scatter_axis,
                        dimensions.len(),
                    )));
                };
                if *dimension != effective_axis_size {
                    return Err(TypeError::invalid(format!(
                        "`{}` untiled scatter axis {} size {} must equal group size {}",
                        PARALLEL_SUM_SCATTER_OPERATION_NAME, self.scatter_axis, dimension, effective_axis_size,
                    )));
                }
                input_type.without_dimension(self.scatter_axis)?.0
            }
            CollectiveMode::Tiled => {
                let mut output_dimensions = dimensions;
                let Some(dimension) = output_dimensions.get_mut(self.scatter_axis) else {
                    return Err(TypeError::invalid(format!(
                        "`{}` scatter axis {} is out of bounds for rank {}",
                        PARALLEL_SUM_SCATTER_OPERATION_NAME,
                        self.scatter_axis,
                        output_dimensions.len(),
                    )));
                };
                if *dimension % effective_axis_size != 0 {
                    return Err(TypeError::invalid(format!(
                        "`{}` scatter axis {} size {} is not divisible by group size {}",
                        PARALLEL_SUM_SCATTER_OPERATION_NAME, self.scatter_axis, *dimension, effective_axis_size,
                    )));
                }
                *dimension /= effective_axis_size;
                infer_linear_collective_operation_output_type(
                    PARALLEL_SUM_SCATTER_OPERATION_NAME,
                    input_type,
                    output_dimensions,
                )?
            }
        };
        self.finalize_output_type(input_type, output_type)
    }

    /// Validates the element data type of a sum-scatter input and, over a manual mesh axis, its manual variation,
    /// applying the reduction-state transition to the shape-only `output_type` shared by the static and array IR
    /// inference paths. An ordinary sum-scatter performs no mesh exchange, even when its axis name shadows a mesh axis,
    /// so it preserves pending mesh sums and variance. Over a manual mesh axis, an input that is unreduced over the
    /// scattered axis is the cotangent of a reduced all-gather result, so the sum-scatter consumes that pending
    /// reduction and returns a value that varies over the axis. Pending reductions over other axes are preserved.
    fn finalize_output_type(&self, input_type: &ArrayType, output_type: ArrayType) -> Result<ArrayType, TypeError> {
        let data_type = input_type.data_type();
        if !data_type.is_numeric() && data_type != DataType::Zero {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` requires numeric inputs but got `{data_type}`",
            )));
        }

        // An ordinary sum-scatter performs only the exchange of its binder. It neither consumes nor introduces
        // mesh-axis reduction or variation state, including when its axis name shadows an enclosing mesh axis.
        let Some(mesh) = &self.mesh else {
            return Ok(output_type);
        };

        let axis_name = self.axis_name();
        validate_manual_mesh_input(
            PARALLEL_SUM_SCATTER_OPERATION_NAME,
            axis_name,
            Some(self.axis_size),
            mesh,
            input_type,
        )?;

        let sharding = input_type.sharding().unwrap();
        if !input_type.unreduced_axes().contains(axis_name) {
            // Every participant of a manual mesh axis receives a different chunk, so an input that is still invariant
            // over that axis would yield an output whose type wrongly claims that it is invariant.
            if !sharding.varying_manual_axes().contains(axis_name) {
                return Err(TypeError::invalid(format!(
                    "`{}` input must vary over manual axis `{}`; pass an invariant \
                     value through `{}` first so that every copy is counted",
                    PARALLEL_SUM_SCATTER_OPERATION_NAME, axis_name, PARALLEL_VARY_OPERATION_NAME,
                )));
            }
            return Ok(output_type);
        }

        // Participant groups sum only within each group, so a grouped exchange would leave part of the pending sum
        // over the axis uncompleted while typing its output as complete.
        if self.options.axis_index_groups.is_some() {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` with axis index groups cannot complete a pending sum over \
                 manual axis `{axis_name}`",
            )));
        }

        // Complete only the participating axis's pending sum. Independent pending sums commute with this exchange
        // and stay pending; the shape-only output type already preserves their sharding state.
        let mut varying_axes = sharding.varying_manual_axes().clone();
        varying_axes.insert(self.axis_name().to_string());
        let output_sharding = output_type.sharding().unwrap().clone();
        Ok(output_type.with_sharding(
            output_sharding
                .with_unreduced_axes(
                    input_type.unreduced_axes().iter().filter(|axis| axis.as_str() != axis_name).cloned(),
                )
                .and_then(|sharding| sharding.with_varying_manual_axes(varying_axes))
                .map_err(TypeError::from)?,
        )?)
    }
}

impl Display for ParallelSumScatterOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ParallelSumScatterOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        PARALLEL_SUM_SCATTER_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        let input_type = self.check_input(input_types, region_interfaces)?;

        // Result-shape arithmetic in the homogeneous array family requires static extents.
        // Dynamic geometry uses explicit result extents in the composite array/dimension family.
        let Some(shape) = input_type.static_shape() else {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` does not support dynamically shaped inputs",
            )));
        };

        Ok(vec![self.infer_static_output_type(input_type, shape.dimensions().to_vec())?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, PARALLEL_SUM_SCATTER_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
            operation.field("axis_size", self.axis_size)?;
            operation.field("scatter_axis", format_args!("{:?}", &self.scatter_axis))?;
            operation.field("options", format_args!("{:?}", &self.options))?;
            if let Some(mesh) = &self.mesh {
                operation.field("mesh", mesh)?;
            }
            Ok(())
        })
    }
}

impl LinearCollectiveOperation for ParallelSumScatterOperation {
    type Adjoint = ParallelAllGatherOperation;

    #[inline]
    fn axis_name(&self) -> &str {
        &self.axis_name
    }

    #[inline]
    fn axis_size(&self) -> usize {
        self.axis_size
    }

    #[inline]
    fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

    #[inline]
    fn effective_axis_size(&self) -> Result<usize, TypeError> {
        self.options.effective_axis_size(PARALLEL_SUM_SCATTER_OPERATION_NAME, self.axis_size)
    }

    fn adjoint(&self, input_type: &ArrayType) -> Result<ParallelAllGatherOperation, ProgramError> {
        // Completing a pending sum has a reduced cotangent, whereas varying inputs keep varying cotangents.
        // Ordinary batch-bound collectives preserve mesh state even when their name shadows a manual mesh axis.
        let output_variance = if self.mesh.is_some() && input_type.unreduced_axes().contains(self.axis_name()) {
            ParallelAllGatherOutputVariance::Reduced
        } else {
            ParallelAllGatherOutputVariance::Varying
        };
        let adjoint = ParallelAllGatherOperation::new(
            self.axis_name.clone(),
            self.axis_size,
            self.scatter_axis,
            self.options.clone(),
            output_variance,
        );
        Ok(match &self.mesh {
            Some(mesh) => adjoint.with_mesh(mesh.clone()),
            None => adjoint,
        })
    }

    #[inline]
    fn adapt_to_batch_axis(&self, input_batch_axis: usize) -> (Self, usize) {
        let (scatter_axis, output_batch_axis) =
            self.options.mode.forwarded_split_axes(self.scatter_axis, input_batch_axis);
        (Self { scatter_axis, ..self.clone() }, output_batch_axis)
    }
}

impl ShapeChangingCollectiveOperation for ParallelSumScatterOperation {
    #[inline]
    fn collective_options(&self) -> &CollectiveOptions {
        &self.options
    }

    fn infer_array_ir_output_types(&self, input_types: &[ArrayIrType]) -> Result<Vec<ArrayIrType>, TypeError> {
        let effective_axis_size = self.effective_axis_size()?;
        let Some(input_type) = input_types.first() else {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` expects an array followed by its output extents",
            )));
        };

        let input_type = <&ArrayType>::try_from(input_type)?;
        if self.options.mode == CollectiveMode::Untiled {
            let Some(input_extent) = input_type.shape().dimensions().get(self.scatter_axis) else {
                return Err(TypeError::invalid(format!(
                    "`{}` scatter axis {} is out of bounds for rank {}",
                    PARALLEL_SUM_SCATTER_OPERATION_NAME,
                    self.scatter_axis,
                    input_type.rank(),
                )));
            };

            if let Dimension::Static(input_extent) = input_extent
                && *input_extent != effective_axis_size
            {
                return Err(TypeError::invalid(format!(
                    "`{}` untiled scatter axis {} size {} must equal group size {}",
                    PARALLEL_SUM_SCATTER_OPERATION_NAME, self.scatter_axis, input_extent, effective_axis_size,
                )));
            }

            let base_output_type = input_type.without_dimension(self.scatter_axis)?.0;
            let mut output_types = infer_array_ir_shape_changing_collective_output_type(
                PARALLEL_SUM_SCATTER_OPERATION_NAME,
                input_types,
                base_output_type,
                &[],
                |_| Ok(()),
            )?;
            let output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
            return Ok(vec![self.finalize_output_type(input_type, output_type)?.into()]);
        }

        if self.scatter_axis >= input_type.rank() {
            return Err(TypeError::invalid(format!(
                "`{}` scatter axis {} is out of bounds for rank {}",
                PARALLEL_SUM_SCATTER_OPERATION_NAME,
                self.scatter_axis,
                input_type.rank(),
            )));
        }

        if let Dimension::Static(input_extent) = &input_type.shape().dimensions()[self.scatter_axis]
            && *input_extent % effective_axis_size != 0
        {
            return Err(TypeError::invalid(format!(
                "`{}` scatter axis {} size {} is not divisible by group size {}",
                PARALLEL_SUM_SCATTER_OPERATION_NAME, self.scatter_axis, input_extent, effective_axis_size,
            )));
        }

        let mut dimensions = input_type.shape().dimensions().to_vec();
        dimensions[self.scatter_axis] = Dimension::Static(0);
        let sharding = input_type.resized_sharding(dimensions.as_slice(), PARALLEL_SUM_SCATTER_OPERATION_NAME)?;
        let mut base_output_type =
            ArrayType::new(input_type.data_type(), Shape::new(dimensions)).with_memory(input_type.memory());
        base_output_type.sharding = sharding;
        let mut output_types = infer_array_ir_shape_changing_collective_output_type(
            PARALLEL_SUM_SCATTER_OPERATION_NAME,
            input_types,
            base_output_type,
            &[self.scatter_axis],
            |output_extents| {
                let rank = input_type.rank();
                let Some(input_extent) = input_type.shape().dimensions().get(self.scatter_axis) else {
                    return Err(TypeError::invalid(format!(
                        "`{}` scatter axis {} is out of bounds for rank {}",
                        PARALLEL_SUM_SCATTER_OPERATION_NAME, self.scatter_axis, rank,
                    )));
                };

                if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                    (input_extent, &output_extents[self.scatter_axis])
                {
                    let expected = *input_extent / effective_axis_size;
                    if *output_extent != expected {
                        return Err(TypeError::invalid(format!(
                            "`{}` result extent must equal input axis {} extent {} divided by axis group size {}; \
                             expected {} but got {}",
                            PARALLEL_SUM_SCATTER_OPERATION_NAME,
                            self.scatter_axis,
                            input_extent,
                            effective_axis_size,
                            expected,
                            output_extent,
                        )));
                    }
                }
                Ok(())
            },
        )?;

        let mut output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();

        // Check the actual result geometry: the placeholder zero above cannot establish explicit-sharding divisibility.
        output_type.sharding =
            input_type.resized_sharding(output_type.shape().dimensions(), PARALLEL_SUM_SCATTER_OPERATION_NAME)?;
        if output_type.shape() == input_type.shape() {
            output_type = output_type.with_layout(input_type.layout().cloned());
        }

        Ok(vec![self.finalize_output_type(input_type, output_type)?.into()])
    }
}

impl<C: Context<Type = ArrayType, Value: Reduce + Transpose>> ShapeChangingCollectiveBatching<C>
    for ParallelSumScatterOperation
{
    fn batch_matching_axis<P: CollectiveArrayExtentBatchingPolicy<C>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        input: &ArrayBatch<C::Value>,
        output_extents: Vec<P::ShapeExtent>,
        output_sharding: Option<Sharding>,
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        // The batching rules infer the output type first, so the scatter axis is known to be within the input rank.
        if self.options.axis_index_groups.is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` axis index groups are not supported \
                     when a batch transform binds the collective axis",
                ),
            });
        }

        let axis_extent =
            P::collective_axis_extent(context, PARALLEL_SUM_SCATTER_OPERATION_NAME, &self.axis_name, self.axis_size)?;

        let mut input_extents = output_extents.clone();
        match self.options.mode {
            CollectiveMode::Untiled => input_extents.insert(self.scatter_axis, axis_extent.clone()),
            CollectiveMode::Tiled => {
                input_extents[self.scatter_axis] = output_extents[self.scatter_axis].mul(&axis_extent)?;
            }
        }

        let input = P::match_collective_axis(context, input, input_extents.as_slice())?;
        let summed = input.into_value().reduce(&[0], ReductionKind::Sum);
        let scattered = match self.options.mode {
            CollectiveMode::Untiled => summed?.move_axis(self.scatter_axis, 0)?,
            CollectiveMode::Tiled => {
                let mut split_extents = output_extents.clone();
                split_extents.insert(self.scatter_axis, axis_extent.clone());
                P::reshape_collective(context, summed?, split_extents.as_slice(), None)?
                    .move_axis(self.scatter_axis, 0)?
            }
        };

        let mut physical_output_extents = Vec::with_capacity(output_extents.len() + 1);
        physical_output_extents.push(axis_extent);
        physical_output_extents.extend(output_extents);
        let physical_output_sharding = output_sharding
            .map(|sharding| sharding.with_leading_batch_axis(context.axis_sharding().clone()))
            .transpose()?;
        let output =
            P::reshape_collective(context, scattered, physical_output_extents.as_slice(), physical_output_sharding)?;
        ArrayBatch::new(output, BatchAxis::from_position(0))
    }
}

impl<C: Domain<Type = ArrayType, Value: Reshape>> InterpretableOperation<C> for ParallelSumScatterOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // Eager binding does not infer output types, so interpretation validates the shared input contract and the
        // operation payload before applying the degenerate-axis rule.
        check_count!("input", inputs, 1, ProgramError);
        self.check_degenerate_interpretation()?;
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let output_type = self.infer_output_types(&input_types, &[])?.remove(0);
        let input = &inputs[0];

        // A single participant sums only its own value. Untiled mode removes the size-one scatter axis,
        // which a reshape to the inferred output type expresses, while tiled mode leaves the shape unchanged.
        Ok(vec![match self.options.mode {
            CollectiveMode::Tiled => input.clone(),
            CollectiveMode::Untiled => {
                input.reshape_with_output_sharding(output_type.shape().clone(), output_type.sharding().cloned())?
            }
        }])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelSumScatterOperation>>> PartiallyEvaluatableOperation<C>
    for ParallelSumScatterOperation
{
}

impl<
    C: Context<Type = ArrayType, Value: Reduce + Transpose, Operation: From<ParallelSumScatterOperation>>,
    P: CollectiveArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for ParallelSumScatterOperation
{
    #[inline]
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // A matching `batch` level consumes the mapped batch axis by summing over it and re-mapping the chunks of the
        // per-item `scatter_axis` onto it: the sum's `scatter_axis` is split into `(b, d_s / b)` chunks and the new
        // chunk axis becomes the output batch axis, so batch item `i` receives chunk `i` of the sum. A non-matching
        // level forwards the collective to the parent context, unchanged for a replicated input and with its array axes
        // shifted past the batch axis for a mapped one.
        self.shape_changing_collective_batch(context, inputs)
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelSumScatterOperation>>> DifferentiableOperation<C>
    for ParallelSumScatterOperation
{
    #[inline]
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        self.linear_collective_jvp(context, inputs)
    }
}

impl<
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<AddOperation<ArrayType>> + From<ParallelAllGatherOperation>,
> TransposableOperation<V, O> for ParallelSumScatterOperation
{
    #[inline]
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // The adjoint all-gather is reduced or varying so that it restores the input cotangent's reduction state.
        self.linear_collective_transpose(context, inputs, outputs, accumulators)
    }
}

impl MemberOperation<ArrayIrType> for ParallelSumScatterOperation {
    #[inline]
    fn infer_parent_region_input_types(
        &self,
        _input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
        Ok(vec![None; region_interfaces.len()])
    }

    #[inline]
    fn infer_parent_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        self.infer_array_ir_output_types(input_types)
    }

    #[inline]
    fn rename_parent_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<DimensionVariable>,
    ) -> Result<Self, TypeError> {
        self.rename_type_identities(renaming)
    }
}

impl<
    C: Domain<
            Type = ArrayIrType,
            Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType> + DimensionSize<usize> + Reshape>
                       + ValueProjection<DimensionType, Projected = DimensionValue>,
        >,
> MemberInterpretableOperation<C> for ParallelSumScatterOperation
{
    #[inline]
    fn interpret_in_parent<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        self.shape_changing_collective_interpret_in_parent::<C>(inputs)
    }
}

impl<
    C: Context<
            Type = ArrayIrType,
            Value: Assert
                       + DimensionSize
                       + DynamicBroadcast
                       + ValueProjection<ArrayType, Projected: Reduce + Transpose + Value<Type = ArrayType>>
                       + ValueProjection<
                DimensionType,
                Projected: Compare<C::Value> + DimensionMax + Rem + Div + Mul + Value<Type = DimensionType>,
            >,
            Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Operation: From<ParallelSumScatterOperation>
                           + From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
> MemberBatchableOperation<C, ArrayIrBatchingPolicy> for ParallelSumScatterOperation
{
    #[inline]
    fn batch_in_parent<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        self.shape_changing_collective_batch_in_parent(context, inputs)
    }
}

impl<
    C: Context<
            Type = ArrayIrType,
            Operation: From<ParallelAllGatherOperation>
                           + From<DimensionSizeOperation>
                           + From<LinearCallOperation<ArrayIrType>>
                           + From<ParallelSumScatterOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
        >,
> MemberDifferentiableOperation<C> for ParallelSumScatterOperation
{
    #[inline]
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The transposed linear region restores the input cotangent's reduction state, so completing a pending sum
        // uses a reduced, rather than varying, all-gather.
        self.shape_changing_collective_jvp_in_parent(context, inputs)
    }
}

/// Represents the ability to sum values across the participants of a named axis and scatter the sum, so that every
/// participant receives only its own chunk, by staging a [`ParallelSumScatterOperation`]. This is the analogue of JAX's
/// [`jax.lax.psum_scatter`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.psum_scatter.html), whose default
/// `tiled = False` corresponds to [`ParallelSumScatter::parallel_sum_scatter`] and whose `tiled = True` corresponds to
/// [`ParallelSumScatter::parallel_sum_scatter_tiled`]. Refer to [`ParallelSumScatterOperation`] for the semantics and
/// transformation rules.
///
/// Over a manual mesh axis, an input that neither varies over the axis nor carries a pending cross-device sum
/// over it is first made varying through [`ParallelVary`], so that every device's copy is counted, as with
/// [`ParallelReduce::parallel_reduce`](super::ParallelReduce::parallel_reduce). The output extents are staged
/// as explicit extent values, and a runtime assertion checks every extent that is not statically known.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// Sum two rows elementwise and give each batch item half of the summed row:
///
/// ```
/// # use ryft_core::operations::collectives::ParallelSumScatter;
/// # use ryft_core::{
/// #     Array, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, BatchAxis, BatchAxisSpecification,
/// #     BatchingTracer, EagerContext, batch,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rows = ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])?);
/// let chunks = batch(
///     |row: BatchingTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, ArrayIrBatchingPolicy>| {
///         row.parallel_sum_scatter_tiled("rows", 0)
///     },
///     rows,
///     BatchAxis::new(0),
///     BatchAxis::new(0),
///     BatchAxisSpecification::named("rows"),
/// )?;
/// assert_eq!(chunks, ArrayIrValue::Array(Array::matrix(2, 2, vec![11.0, 22.0, 33.0, 44.0])?));
/// # Ok(())
/// # }
/// ```
#[capability]
pub trait ParallelSumScatter<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns the sum of this value across the participants of the named axis `axis_name`, scattered along
    /// `scatter_axis`. The extent of `scatter_axis` must equal the number of participants, and the axis is removed,
    /// so that participant `i` receives row `i` of the sum.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `scatter_axis`: Axis of this value along which the sum is scattered.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`ParallelSumScatter::parallel_sum_scatter_with_options`].
    #[inline]
    fn parallel_sum_scatter(&self, axis_name: &str, scatter_axis: usize) -> Result<Self, ProgramError> {
        self.parallel_sum_scatter_with_options(axis_name, scatter_axis, CollectiveOptions::default())
    }

    /// Returns the sum of this value across the participants of the named axis `axis_name`, scattered in equal
    /// contiguous chunks along `scatter_axis`. The extent of `scatter_axis` must be divisible by the number of
    /// participants, and participant `i` receives chunk `i` of the sum.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `scatter_axis`: Axis of this value along which the sum is scattered.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`ParallelSumScatter::parallel_sum_scatter_with_options`].
    #[inline]
    fn parallel_sum_scatter_tiled(&self, axis_name: &str, scatter_axis: usize) -> Result<Self, ProgramError> {
        self.parallel_sum_scatter_with_options(axis_name, scatter_axis, CollectiveOptions::new(CollectiveMode::Tiled))
    }

    /// Returns the sum of this value across the participants of the named axis `axis_name`, scattered along
    /// `scatter_axis` with the tiling mode and participant groups of `options`.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `scatter_axis`: Axis of this value along which the sum is scattered.
    ///   - `options`: [`CollectiveMode`] and optional participant groups of the collective.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Axis`] error wrapping [`AxisError::UnboundAxisName`] when no enclosing binder binds
    /// `axis_name`, and a [`ProgramError`] if `scatter_axis` is out of bounds, if its extent does not fit the tiling
    /// mode, if the participant groups are invalid, if this value is not numeric, or if its manual mesh state does
    /// not satisfy the input variation or pending-sum contract.
    fn parallel_sum_scatter_with_options(
        &self,
        axis_name: &str,
        scatter_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError>;
}

impl ParallelSumScatter<ArrayType> for Array {
    // A concrete `Array` never executes inside an axis binder, because the values under a `batch` level or inside
    // a manual region are tracers, so every axis name is unbound for it.

    #[inline]
    fn parallel_sum_scatter_with_options(
        &self,
        axis_name: &str,
        _scatter_axis: usize,
        _options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        Err(AxisError::UnboundAxisName { name: axis_name.to_string() }.into())
    }
}

impl<
    V: ShapeChangingCollectiveValue<
            DispatchDomain: Context<Value = V, Operation: From<ParallelSumScatterOperation>> + NamedAxes,
        > + ParallelVary,
> ParallelSumScatter<ArrayType> for V
{
    fn parallel_sum_scatter_with_options(
        &self,
        axis_name: &str,
        scatter_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        // Homogeneous values opt into direct staging, while projected values retain composite extent delegation.
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        options.effective_axis_size(PARALLEL_SUM_SCATTER_OPERATION_NAME, axis_size)?;
        let mut input = self.clone();
        let mut operation = ParallelSumScatterOperation::new(axis_name.to_string(), axis_size, scatter_axis, options);
        if let Some(NamedAxis::Mesh { mesh, .. }) = context.named_axis(axis_name) {
            if !input.r#type().sharding().is_some_and(|sharding| {
                sharding.varying_manual_axes().contains(axis_name) || sharding.unreduced_axes().contains(axis_name)
            }) {
                input = input.parallel_vary(axis_name)?;
            }
            operation = operation.with_mesh(mesh);
        }
        let mut outputs = context.bind(operation, Vec::new(), &[input])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<A: Value<Type = ArrayType> + ParallelSumScatter<ArrayType>> ParallelSumScatter<ArrayIrType> for ArrayIrValue<A> {
    fn parallel_sum_scatter_with_options(
        &self,
        axis_name: &str,
        scatter_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        // A concrete composite value performs the collective through its array member.
        let array = ValueProjection::<ArrayType>::into_projected(self.clone())?;
        Ok(<Self as ValueProjection<ArrayType>>::from_projected(array.parallel_sum_scatter_with_options(
            axis_name,
            scatter_axis,
            options,
        )?))
    }
}

impl<
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<Type = ArrayIrType, Operation: From<ParallelSumScatterOperation>>
                                + NamedAxes
                                + DimensionConstant,
        > + Assert
        + DimensionSize<V>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType> + Div + Rem + Compare<V>>
        + ValueProjection<ArrayType, Projected: ParallelVary>,
> ParallelSumScatter<ArrayIrType> for V
{
    fn parallel_sum_scatter_with_options(
        &self,
        axis_name: &str,
        scatter_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        // A composite value binds a `ParallelSumScatterOperation` through its own context, followed by one explicit
        // extent value per output axis, which also asserts at runtime that dynamic extents fit the tiling mode. Over a
        // manual mesh axis, an input that neither varies over the axis nor is unreduced over it is first made varying
        // through its array view, exactly as JAX's `psum_scatter` does, so that every device's copy is counted.
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let effective_axis_size = options.effective_axis_size(PARALLEL_SUM_SCATTER_OPERATION_NAME, axis_size)?;
        let mut input = self.clone();
        let mut operation =
            ParallelSumScatterOperation::new(axis_name.to_string(), axis_size, scatter_axis, options.clone());
        if let Some(NamedAxis::Mesh { mesh, .. }) = context.named_axis(axis_name) {
            let array = ValueProjection::<ArrayType>::into_projected(self.clone())?;
            if !array.r#type().sharding().is_some_and(|sharding| {
                sharding.varying_manual_axes().contains(axis_name) || sharding.unreduced_axes().contains(axis_name)
            }) {
                input = <V as ValueProjection<ArrayType>>::from_projected(array.parallel_vary(axis_name)?);
            }
            operation = operation.with_mesh(mesh);
        }
        let input_type = input.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let rank = input_type.rank();
        if scatter_axis >= rank {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` scatter axis {scatter_axis} is out of bounds for rank {rank}",
            ))
            .into());
        }

        // Untiled scatter consumes its selected axis. Only a dynamic extent needs to be observed and checked;
        // operation type inference checks the static input geometry when the collective is bound below.
        if options.mode == CollectiveMode::Untiled
            && matches!(input_type.shape().dimensions()[scatter_axis], Dimension::Dynamic(_))
        {
            let extent = ValueProjection::<DimensionType>::into_projected(input.dimension_size(scatter_axis)?)?;
            let participants =
                ValueProjection::<DimensionType>::into_projected(context.dimension_constant(effective_axis_size)?)?;
            extent.equal(&participants)?.assert(
                "collective axis extent must match the participant count",
                &[
                    ("extent", ValueProjection::<DimensionType>::from_projected(extent)),
                    ("participants", ValueProjection::<DimensionType>::from_projected(participants)),
                ],
            )?;
        }

        let mut output_extents = (0..rank)
            .filter(|axis| options.mode != CollectiveMode::Untiled || *axis != scatter_axis)
            .map(|axis| input.dimension_size(axis))
            .collect::<Result<Vec<_>, _>>()?;
        if options.mode == CollectiveMode::Tiled {
            let extent = ValueProjection::<DimensionType>::into_projected(output_extents[scatter_axis].clone())?;
            let participants =
                ValueProjection::<DimensionType>::into_projected(context.dimension_constant(effective_axis_size)?)?;
            if matches!(input_type.shape().dimensions()[scatter_axis], Dimension::Dynamic(_)) {
                let zero = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(0)?)?;
                extent.rem(&participants)?.equal(&zero)?.assert(
                    "collective extent must be divisible by the participant count",
                    &[
                        ("extent", ValueProjection::<DimensionType>::from_projected(extent.clone())),
                        ("divisor", ValueProjection::<DimensionType>::from_projected(participants.clone())),
                    ],
                )?;
            }
            output_extents[scatter_axis] = ValueProjection::<DimensionType>::from_projected(extent.div(&participants)?);
        }

        let inputs = std::iter::once(input).chain(output_extents).collect::<Vec<_>>();
        let mut outputs = context.bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<V: ParallelSumScatter<ArrayIrType> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>>
    ParallelSumScatter<ArrayType> for ProjectedValue<ArrayType, V>
{
    #[inline]
    fn parallel_sum_scatter_with_options(
        &self,
        axis_name: &str,
        scatter_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        self.value()
            .parallel_sum_scatter_with_options(axis_name, scatter_axis, options)?
            .into_projected()
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use half::f16;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayOperation, DimensionBounds, Layout, Memory, MeshAxis, MeshAxisType, RaggedAxis,
        ShardingDimension, StridedLayout,
    };
    use crate::batching::{BatchAxisSpecification, BatchingTracer, batch};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{DifferentiableType, DifferentiationTracer, transpose_mixed_operation};
    use crate::macros::{check_gradient, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::assertions::AssertionError;
    use crate::operations::collectives::tests::{batch_collective, collective_program};
    use crate::operations::manipulation::slicing::Slice;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationOutput, PartialEvaluationValue};
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};

    use super::*;

    /// Creates a manual mesh whose axis `"x"` has two participants and whose axis `"y"` has one.
    fn manual_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    /// Checks that transposing a tiled sum-scatter over the two-participant manual mesh axis `"x"` of an `f32[8]`
    /// input with the provided `sharding` stages the `expected` program in both the homogeneous array family and
    /// the composite array/dimension family, so that both representations restore the same complete input cotangent
    /// reduction state. The composite family must additionally give the explicit output extent a structural-zero
    /// cotangent.
    fn check_parallel_sum_scatter_transposition_reduction_state(sharding: Sharding, expected: &str) {
        let operation = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled())
            .with_mesh(sharding.mesh().clone());
        let input_type = ArrayType::new_static(DataType::F32, [8]).with_sharding(sharding).unwrap();
        let output_type = operation.infer_output_types(std::slice::from_ref(&input_type), &[]).unwrap().remove(0);

        // Both representations must restore the complete input cotangent state, including a reduced `x` when the
        // sum-scatter completes its pending sum, and pending or reduced state on independent manual axes.
        let program = collective_program(operation.clone(), input_type.clone());
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(transposed.to_string(), expected);

        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let output_cotangent = context.input(output_type.cotangent().unwrap().into());
        let mut context = TranspositionContext::new(context);
        let inputs = [
            PartialValue::Unknown(input_type.clone().into()),
            PartialValue::Unknown(DimensionValue::constant(4).unwrap().r#type().into_owned().into()),
        ];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        transpose_mixed_operation(
            &mut context,
            &operation,
            &inputs,
            &[MaybeZero::Value(output_cotangent)],
            &accumulators,
        )
        .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        let [MaybeZero::Value(array), MaybeZero::Zero(_)] = cotangents.as_slice() else {
            panic!("expected an array cotangent and a structural-zero extent cotangent");
        };
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![array.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(program.to_string(), expected);
    }

    #[test]
    fn test_parallel_sum_scatter() {
        let operation = ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::tiled());
        assert_eq!(operation.name(), PARALLEL_SUM_SCATTER_OPERATION_NAME);
        assert_eq!(operation.axis_name(), "x");
        assert_eq!(operation.axis_size(), 4);
        assert_eq!(operation.scatter_axis(), 1);
        assert_eq!(operation.options(), &CollectiveOptions::tiled());
        assert_eq!(operation.mesh(), None);
        assert_eq!(operation.effective_axis_size(), Ok(4));
        assert_eq!(
            operation.to_string(),
            "parallel_sum_scatter [axis_name=\"x\", axis_size=4, scatter_axis=1, options=Tiled]",
        );

        // Participant groups restrict the effective axis size to the size of one group.
        let grouped = ParallelSumScatterOperation::new(
            "x".to_string(),
            4,
            0,
            CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
        );
        assert_eq!(grouped.effective_axis_size(), Ok(2));
    }

    #[test]
    fn test_parallel_sum_scatter_with_mesh() {
        // A sum-scatter over a manual mesh axis records and renders its mesh.
        let mesh_operation = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled())
            .with_mesh(manual_mesh());
        assert_eq!(mesh_operation.mesh(), Some(&manual_mesh()));
        assert_eq!(
            mesh_operation.to_string(),
            indoc! {"
                parallel_sum_scatter [
                    axis_name=\"x\",
                    axis_size=2,
                    scatter_axis=0,
                    options=Tiled,
                    mesh=['x'=2:manual, 'y'=1:manual],
                ]"
            },
        );
        assert_ne!(mesh_operation, ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()));
    }

    #[test]
    fn test_parallel_sum_scatter_type_inference() {
        // A tiled sum-scatter divides the scatter axis by the group size, while an untiled one requires the scatter
        // axis to have exactly that size and removes it.
        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new("x".to_string(), 4, 0, CollectiveOptions::tiled()),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [8])],
                    output_types = [ArrayType::new_static(DataType::F32, [2])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [0])],
                    output_types = [ArrayType::new_static(DataType::F32, [0])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::Zero, [8])],
                    output_types = [ArrayType::new_static(DataType::Zero, [2])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [6])],
                    error = "`parallel_sum_scatter` scatter axis 0 size 6 is not divisible by group size 4",
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "`parallel_sum_scatter` scatter axis 0 is out of bounds for rank 0",
                },
                {
                    input_types = [ArrayType::new_static(DataType::Boolean, [8])],
                    error = "`parallel_sum_scatter` requires numeric inputs but got `bool`",
                },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default()),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [3, 4])],
                    output_types = [ArrayType::new_static(DataType::F32, [3])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4, 3])],
                    error = "`parallel_sum_scatter` untiled scatter axis 1 size 3 must equal group size 4",
                },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new(
                "x".to_string(),
                4,
                0,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
            ),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [6])],
                output_types = [ArrayType::new_static(DataType::F32, [3])],
            }],
        );
    }

    #[test]
    fn test_parallel_sum_scatter_type_inference_metadata() {
        let operation = ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled());
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(operation.infer_output_types(&[input_type.clone()], &[]), Ok(vec![input_type.clone()]));
        assert_eq!(
            operation.infer_parent_output_types(
                &[input_type.clone().into(), DimensionValue::constant(2).unwrap().r#type().into_owned().into()],
                &[],
            ),
            Ok(vec![input_type.into()]),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_type_inference_manual_mesh() {
        // Over a manual mesh axis, every participant receives a different chunk, so an invariant input is rejected and
        // a varying input keeps its variation. An input that is unreduced over the operation's own axis has its pending
        // sum completed and its output varies over the axis, while unrelated pending sums are preserved. The input
        // must carry the operation's mesh.
        let sharding = Sharding::replicated(manual_mesh(), 1);
        let varying = sharding.clone().with_varying_manual_axes(["x"]).unwrap();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled())
                .with_mesh(manual_mesh()),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4]).with_sharding(varying.clone()).unwrap()],
                    output_types = [ArrayType::new_static(DataType::F32, [2]).with_sharding(varying.clone()).unwrap()],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4])
                        .with_sharding(sharding.clone().with_unreduced_axes(["x"]).unwrap())
                        .unwrap()],
                    output_types = [ArrayType::new_static(DataType::F32, [2]).with_sharding(varying.clone()).unwrap()],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap()],
                    error = "`parallel_sum_scatter` input must vary over manual axis `x`; pass an invariant value \
                             through `parallel_vary` first so that every copy is counted",
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4])
                        .with_sharding(sharding.clone().with_unreduced_axes(["y"]).unwrap())
                        .unwrap()],
                    error = "`parallel_sum_scatter` input must vary over manual axis `x`; pass an invariant value \
                             through `parallel_vary` first so that every copy is counted",
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4])
                        .with_sharding(sharding.clone().with_unreduced_axes(["x", "y"]).unwrap()).unwrap()],
                    output_types = [ArrayType::new_static(DataType::F32, [2])
                        .with_sharding(varying.clone().with_unreduced_axes(["y"]).unwrap()).unwrap()],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4])],
                    error = "`parallel_sum_scatter` input must carry a mesh containing manual axis `x`",
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4]).with_sharding(
                        Sharding::replicated(other_mesh, 1).with_varying_manual_axes(["x"]).unwrap()).unwrap()],
                    error = "`parallel_sum_scatter` input mesh does not match the operation mesh",
                },
            ],
        );

        // Participant groups sum only within each group, so a grouped sum-scatter accepts varying inputs but cannot
        // complete a pending sum over the whole axis.
        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]),
            )
            .with_mesh(manual_mesh()),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4]).with_sharding(varying.clone()).unwrap()],
                    output_types = [ArrayType::new_static(DataType::F32, [4]).with_sharding(varying.clone()).unwrap()],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4])
                        .with_sharding(sharding.clone().with_unreduced_axes(["x"]).unwrap())
                        .unwrap()],
                    error = "`parallel_sum_scatter` with axis index groups cannot complete a pending sum over manual \
                             axis `x`",
                },
            ],
        );

        // The mesh axis must be manual, and its size must equal the axis size of the operation.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled())
                .with_mesh(explicit_mesh),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [4]).with_sharding(varying.clone()).unwrap()],
                error = "`parallel_sum_scatter` mesh axis `x` must be manual",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new("x".to_string(), 4, 0, CollectiveOptions::tiled())
                .with_mesh(manual_mesh()),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [4]).with_sharding(varying.clone()).unwrap()],
                error = "`parallel_sum_scatter` axis size 4 does not match the size of manual mesh axis `x`",
            }],
        );
    }

    #[test]
    fn test_parallel_sum_scatter_type_inference_preserves_manual_mesh_state() {
        let sharding = Sharding::replicated(manual_mesh(), 1);
        // An ordinary sum-scatter preserves the mesh state of its input, including invariance and pending sums over a
        // manual mesh axis with the same name, because a `batch` level that shadows that mesh axis may bind it.
        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap()],
                    output_types = [ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap()],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4])
                        .with_sharding(sharding.clone().with_unreduced_axes(["x"]).unwrap())
                        .unwrap()],
                    output_types = [ArrayType::new_static(DataType::F32, [2])
                        .with_sharding(sharding.with_unreduced_axes(["x"]).unwrap())
                        .unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_parallel_sum_scatter_type_inference_array_ir() {
        // The composite family follows each array input with one explicit extent per output axis, checks every extent
        // that is statically known, and keeps dynamic extents.
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [2, 4, 3]).into(),
                        DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                        DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                    ],
                    &[],
                ),
            Ok(vec![ArrayType::new_static(DataType::F32, [2, 3]).into()]),
        );
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 2, 1, CollectiveOptions::default())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [3, 2, 4]).into(),
                        DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                        DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    ],
                    &[],
                ),
            Ok(vec![ArrayType::new_static(DataType::F32, [3, 4]).into()]),
        );
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [2, 5]).into(),
                        DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    ],
                    &[],
                ),
            Err(TypeError::invalid("`parallel_sum_scatter` untiled scatter axis 1 size 5 must equal group size 4")),
        );
        assert_eq!(
            ParallelSumScatterOperation::new(
                "x".to_string(),
                4,
                0,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
            )
            .infer_parent_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [6]).into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
                &[],
            ),
            Ok(vec![ArrayType::new_static(DataType::F32, [3]).into()]),
        );
        let input_axis = DimensionVariable::new("input", DimensionBounds::new(1, Some(17)).unwrap());
        let output_axis = DimensionVariable::new("split", DimensionBounds::new(1, Some(9)).unwrap());
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled())
                .infer_parent_output_types(
                    &[
                        ArrayType::new(
                            DataType::F32,
                            Shape::new(vec![Dimension::Dynamic(input_axis), Dimension::Static(3)]),
                        )
                        .into(),
                        ArrayIrType::Dimension(DimensionType::from(output_axis.clone())),
                        DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                    ],
                    &[],
                ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(output_axis), Dimension::Static(3)]))
                    .into(),
            ]),
        );

        // A known invalid input extent remains invalid when the caller supplies a dynamic output extent.
        let output_extent = DimensionVariable::new("output", DimensionBounds::unbounded());
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [3]).into(),
                        ArrayIrType::Dimension(DimensionType::from(output_extent)),
                    ],
                    &[],
                ),
            Err(TypeError::invalid("`parallel_sum_scatter` scatter axis 0 size 3 is not divisible by group size 2")),
        );

        // Validate explicit sharding against the final result extent rather than the temporary shape placeholder.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            ParallelSumScatterOperation::new("y".to_string(), 4, 0, CollectiveOptions::tiled())
                .infer_parent_output_types(
                    &[input_type.into(), DimensionValue::constant(1).unwrap().r#type().into_owned().into()],
                    &[],
                ),
            Err(TypeError::invalid(
                "`parallel_sum_scatter` on a dimension sharded over explicit mesh axes requires the output size \
                 (1) at axis 0 to be divisible by the mesh-axis product (2)",
            )),
        );

        // A statically supplied tiled extent must describe the actual scattered chunk size.
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [6]).into(),
                        DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    ],
                    &[],
                ),
            Err(TypeError::invalid(
                "`parallel_sum_scatter` result extent must equal input axis 0 extent 6 divided by axis group size 2; \
                 expected 3 but got 4",
            )),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_interpretation() {
        // Outside any binder, a sum-scatter whose instances each combine a single participant sums only its own value:
        // tiled mode is the identity, while untiled mode removes the size-one scatter axis. Participant groups that
        // hold one participant each qualify even over a larger axis, whereas an ungrouped larger axis has no per-item
        // semantics outside an enclosing binder.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let input = Array::matrix(1, 3, vec![1.0f32, 2.0, 3.0]).unwrap();
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::default()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()]),
        );
        let singleton_groups = vec![vec![0], vec![1]];
        assert_eq!(
            ParallelSumScatterOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled().with_axis_index_groups(singleton_groups.clone()),
            )
            .interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            ParallelSumScatterOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::default().with_axis_index_groups(singleton_groups),
            )
            .interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Ok(vec![Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()]),
        );
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 2, 1, CollectiveOptions::default()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_sum_scatter` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_interpretation_array_ir() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();

        // A single-participant sum-scatter interprets the array input with its explicit extents in the same way as the
        // homogeneous family.
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        assert_eq!(
            context.bind(
                ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())],
            ),
            Ok(vec![input]),
        );

        // A sum-scatter over a larger axis whose participant groups each hold a single participant also has per-item
        // semantics, because every instance sums only its own value.
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap());
        assert_eq!(
            context.bind(
                ParallelSumScatterOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]),
                ),
                Vec::new(),
                &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())],
            ),
            Ok(vec![input]),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_partial_evaluation() {
        // A single participant folds known inputs and residualizes unknown inputs; untiled mode removes the axis.
        check_operation_partial_evaluation!(
            operation = ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::default()),
            inputs = [Array::matrix(1, 2, vec![1.0f32, 2.0]).unwrap()],
            expected = Array::vector(vec![1.0f32, 2.0]).unwrap(),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_partial_evaluation_staging_parent() {
        // A known input under a staging parent stays known, because the operation is staged into the parent trace.
        let operation = ParallelSumScatterOperation::new("x".to_string(), 2, 1, CollectiveOptions::tiled());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = PartialEvaluationValue::known(trace.input(ArrayType::new_static(DataType::F32, [1, 2])));
        let outputs = operation
            .partially_evaluate(&PartialEvaluationContext::new(trace.clone()), &EmptyRegionDriver, &[input])
            .unwrap();
        assert!(outputs[0].is_known());
        assert_eq!(outputs[0].r#type().as_ref(), &ArrayType::new_static(DataType::F32, [1, 1]));
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].as_known().unwrap().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[1, 2] .
                let %1:f32[1, 1] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=1, options=Tiled] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_sum_scatter_partial_evaluation_residualizes_known_input() {
        // A known input over a larger axis under an eager parent residualizes the operation, which has no per-item
        // value, so the residual program is the source program itself.
        let input = Array::matrix(1, 2, vec![1.0f32, 2.0]).unwrap();
        let operation = ParallelSumScatterOperation::new("x".to_string(), 2, 1, CollectiveOptions::tiled());
        let program = collective_program(operation, ArrayType::new_static(DataType::F32, [1, 2]));
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input)]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[1, 2] .
                let %1:f32[1, 1] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=1, options=Tiled] %0
                in (%1)"
            },
        );
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
    }

    #[test]
    fn test_parallel_sum_scatter_batching() {
        // A level that binds the axis sums over its mapped axis and maps the scattered chunks back onto it, so that
        // item `i` receives chunk `i` of the sum: a contiguous chunk in tiled mode and row `i` in untiled mode.
        let tiled = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled());
        let untiled = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::default());
        assert_eq!(
            batch_collective(
                &tiled,
                "x",
                2,
                ArrayBatch::new(
                    Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]).unwrap(),
                    BatchAxis::new(0),
                )
                .unwrap(),
            ),
            Ok(vec![
                ArrayBatch::new(Array::matrix(2, 2, vec![11.0, 22.0, 33.0, 44.0]).unwrap(), BatchAxis::new(0)).unwrap()
            ]),
        );
        assert_eq!(
            batch_collective(
                &untiled,
                "x",
                2,
                ArrayBatch::new(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap(),
            ),
            Ok(vec![ArrayBatch::new(Array::vector(vec![4.0, 6.0]).unwrap(), BatchAxis::new(0)).unwrap()]),
        );

        // A replicated input holds the same value for every item, so every item's copy is counted in the sum.
        assert_eq!(
            batch_collective(&tiled, "x", 2, ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]).unwrap())),
            Ok(vec![ArrayBatch::new(Array::matrix(2, 1, vec![2.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap()]),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_batching_widens_accumulation() {
        // Eager matching batches use the ordinary local sum kernel, whose narrow floating-point accumulator is
        // `f32`. The cancellation retains a unit contribution that element-type accumulation could round away.
        let narrow = Array::matrix(
            3,
            3,
            vec![
                f16::from_f32(2048.0),
                f16::ZERO,
                f16::ZERO,
                f16::ONE,
                f16::ZERO,
                f16::ZERO,
                f16::from_f32(-2048.0),
                f16::ZERO,
                f16::ZERO,
            ],
        )
        .unwrap();
        assert_eq!(
            batch_collective(
                &ParallelSumScatterOperation::new("x".to_string(), 3, 0, CollectiveOptions::default()),
                "x",
                3,
                ArrayBatch::new(narrow, BatchAxis::new(0)).unwrap(),
            ),
            Ok(vec![
                ArrayBatch::new(Array::vector(vec![f16::ONE, f16::ZERO, f16::ZERO]).unwrap(), BatchAxis::new(0))
                    .unwrap()
            ]),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_batching_forwards_unbound_axis() {
        // A level that binds another axis forwards a tiled sum-scatter to its parent, shifting the scatter axis past
        // the mapped axis.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 3).with_axis_name("y".to_string());
        let input =
            ArrayBatch::new(trace.input(ArrayType::new_static(DataType::F32, [3, 4])), BatchAxis::new(0)).unwrap();
        let outputs = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled())
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0;
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].value().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3, 4] .
                let %1:f32[3, 2] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=1, options=Tiled] %0
                in (%1)"
            },
        );

        // In untiled mode, the forwarded sum-scatter also tracks a trailing mapped axis across the removed scatter
        // axis, which moves the mapped axis to the front of the output.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 3).with_axis_name("y".to_string());
        let input =
            ArrayBatch::new(trace.input(ArrayType::new_static(DataType::F32, [2, 3])), BatchAxis::new(1)).unwrap();
        let outputs = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::default())
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0;
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].value().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:f32[3] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=0, options=Untiled] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_sum_scatter_batching_rejects_unsupported_inputs() {
        // A level that binds the axis rejects participant groups, because its local sum spans every batch item.
        let grouped = ParallelSumScatterOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]),
        );
        assert_eq!(
            batch_collective(
                &grouped,
                "x",
                2,
                ArrayBatch::new(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap(),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_sum_scatter` axis index groups are not supported when a batch transform binds the \
                          collective axis"
                    .to_string(),
            }),
        );

        // A sum-scatter over a manual mesh axis cannot be consumed by a level that binds a batch axis with its name.
        let tiled = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled());
        assert_eq!(
            batch_collective(
                &tiled.clone().with_mesh(manual_mesh()),
                "x",
                2,
                ArrayBatch::new(Array::matrix(2, 4, vec![1.0; 8]).unwrap(), BatchAxis::new(0)).unwrap(),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_sum_scatter` over a manual mesh axis cannot bind a named batch axis".to_string(),
            }),
        );

        // A matching level rejects bounded ragged inputs in both families, and the composite family does so before it
        // reads the mapped extents.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let ragged = ArrayBatch::new(Array::matrix(2, 4, vec![1.0; 8]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![2i32, 4]).unwrap(), length.clone(), vec![0])])
            .unwrap();
        assert_eq!(
            batch_collective(&tiled, "x", 2, ragged),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_sum_scatter` does not support bounded ragged dimension `length` on input 0"
                    .to_string(),
            }),
        );
        let extents = ArrayIrValue::Array(Array::vector(vec![2i32, 4]).unwrap());
        let input =
            ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0f32; 8]).unwrap()), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), length.clone(), vec![0])])
                .unwrap();
        let output_extent =
            ArrayIrBatch::mapped_dimension(extents, BatchAxis::new(0), DimensionType::from(length)).unwrap();
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("x".to_string());
        assert_eq!(
            tiled.batch_in_parent(&context, &EmptyRegionDriver, &[input, output_extent]),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_sum_scatter` does not support bounded ragged dimension `length` on input 0"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_batching_shadows_manual_axis() {
        // An inner named batch binds `x` independently of a manual mesh axis with the same name. Its local sums
        // preserve the mesh invariance of the input.
        let mesh = manual_mesh();
        let sharding = Sharding::replicated(mesh.clone(), 2);
        let expected = indoc! {"
            lambda %0:f32[2, 4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}]}] .
            let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] = reduce [kind=sum, axes=[0]] %0
                %2:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}]}] = reshape [shape=[2, 2]] %1
            in (%2)"
        };
        let expected_array_ir = indoc! {"
            lambda %0:f32[2, 4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}]}] .
            let %1:dimension<2> = constant [value=2]
                %2:dimension<4> = constant [value=4]
                %3:dimension<2> = constant [value=2]
                %4:dimension<2> = dimension_div %2 %3
                %5:dimension<2> = constant [value=2]
                %6:bool[] = const true
                %7:dimension<4> = dimension_mul %4 %1
                %8:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] = reduce [kind=sum, axes=[0]] %0
                %9:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}]}] = reshape %8 %1 %4
                %10:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}]}] = \
                    reshape [output_sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}]}] %9 %1 %4
            in (%10)"
        };
        let input_type = ArrayType::new_static(DataType::F32, [2, 4]).with_sharding(sharding.clone()).unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(sharding).unwrap();
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| {
                let context = BatchingContext::<_, ArrayBatchingPolicy>::new(input.dispatch_domain(), 2)
                    .with_axis_name("x".to_string());
                let input = ArrayBatch::new(input, BatchAxis::new(0))?;
                let operation = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled());
                let mut outputs = operation.batch(&context, &EmptyRegionDriver, &[input])?.into_parts().0;
                Ok(outputs.remove(0).into_value())
            },
            input_type.clone(),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(output_type, expected_type);
        assert_eq!(program.to_string(), expected);

        // The composite family's explicit output extents use the same local batching rule.
        let (output_type, program) =
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| {
                    batch(
                        |item| item.parallel_sum_scatter_tiled("x", 0),
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
        assert_eq!(program.to_string(), expected_array_ir);
    }

    #[test]
    fn test_parallel_sum_scatter_batching_shadows_manual_axis_unreduced() {
        // An inner named batch binds `x` independently of a manual mesh axis with the same name. Its local sums
        // preserve the pending sum over the shadowed manual mesh axis.
        let mesh = manual_mesh();
        let sharding = Sharding::replicated(mesh.clone(), 2).with_unreduced_axes(["x"]).unwrap();
        let expected = indoc! {"
            lambda %0:f32[2, 4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}], unreduced={'x'}}] .
            let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], unreduced={'x'}}] = \
                    reduce [kind=sum, axes=[0]] %0
                %2:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}], unreduced={'x'}}] = \
                    reshape [shape=[2, 2]] %1
            in (%2)"
        };
        let expected_array_ir = indoc! {"
            lambda %0:f32[2, 4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}], unreduced={'x'}}] .
            let %1:dimension<2> = constant [value=2]
                %2:dimension<4> = constant [value=4]
                %3:dimension<2> = constant [value=2]
                %4:dimension<2> = dimension_div %2 %3
                %5:dimension<2> = constant [value=2]
                %6:bool[] = const true
                %7:dimension<4> = dimension_mul %4 %1
                %8:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], unreduced={'x'}}] = \
                    reduce [kind=sum, axes=[0]] %0
                %9:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}], unreduced={'x'}}] = \
                    reshape %8 %1 %4
                %10:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}], unreduced={'x'}}] = \
                    reshape [output_sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}], unreduced={'x'}}] %9 %1 %4
            in (%10)"
        };
        let input_type = ArrayType::new_static(DataType::F32, [2, 4]).with_sharding(sharding.clone()).unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(sharding).unwrap();
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| {
                let context = BatchingContext::<_, ArrayBatchingPolicy>::new(input.dispatch_domain(), 2)
                    .with_axis_name("x".to_string());
                let input = ArrayBatch::new(input, BatchAxis::new(0))?;
                let operation = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled());
                let mut outputs = operation.batch(&context, &EmptyRegionDriver, &[input])?.into_parts().0;
                Ok(outputs.remove(0).into_value())
            },
            input_type.clone(),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(output_type, expected_type);
        assert_eq!(program.to_string(), expected);

        // The composite family's explicit output extents use the same local batching rule.
        let (output_type, program) =
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| {
                    batch(
                        |item| item.parallel_sum_scatter_tiled("x", 0),
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
        assert_eq!(program.to_string(), expected_array_ir);
    }

    #[test]
    fn test_parallel_sum_scatter_differentiation() {
        // The collective is linear, so the tangent rides the same sum-scatter as the primal.
        let operation = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled());
        assert_eq!(
            collective_program(operation, ArrayType::new_static(DataType::F32, [4])).jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[2] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=0, options=Tiled] %0
                    %3:f32[2] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=0, options=Tiled] %1
                in (%2, %3)"
            },
        );
    }

    #[test]
    fn test_parallel_sum_scatter_differentiation_array_ir_dynamic_extent() {
        // The composite family linearizes a sum-scatter with an explicit extent into a linear call whose pullback, over
        // a single participant, returns the output cotangent unchanged.
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dimension_type = DimensionType::from(variable.clone());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array =
            builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)])).into());
        let extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![array, extent],
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
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:dimension<extent ∈ [1, 9)> .
                let %2:f32[extent] = parallel_sum_scatter [axis_name=\"x\", axis_size=1, scatter_axis=0, \
                    options=Tiled] %0 %1
                in (%2, %1)"
            },
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:dimension<extent ∈ [1, 9)> .
                let %2:f32[extent] = linear_call [residual_count=1] %1 %0 [
                    forward={
                        lambda %0:dimension<extent ∈ [1, 9)>, %1:f32[extent] .
                        let %2:f32[extent] = parallel_sum_scatter [axis_name=\"x\", axis_size=1, scatter_axis=0, \
                            options=Tiled] %1 %0
                        in (%2)
                    },
                    transpose={
                        lambda %0:dimension<extent ∈ [1, 9)>, %1:f32[extent] .
                        let %2:f32[extent] = parallel_all_gather [
                            axis_name=\"x\",
                            axis_size=1,
                            concat_axis=0,
                            options=Tiled,
                            output_variance=Varying,
                        ] %1 %0
                        in (%2)
                    },
                ]
                in (%2)"
            },
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:dimension<extent ∈ [1, 9)> .
                let %2:f32[extent] = linear_call [residual_count=1] %1 %0 [
                    forward={
                        lambda %0:dimension<extent ∈ [1, 9)>, %1:f32[extent] .
                        let %2:f32[extent] = parallel_all_gather [
                            axis_name=\"x\",
                            axis_size=1,
                            concat_axis=0,
                            options=Tiled,
                            output_variance=Varying,
                        ] %1 %0
                        in (%2)
                    },
                    transpose={
                        lambda %0:dimension<extent ∈ [1, 9)>, %1:f32[extent] .
                        let %2:f32[extent] = parallel_sum_scatter [axis_name=\"x\", axis_size=1, scatter_axis=0, \
                            options=Tiled] %1 %0
                        in (%2)
                    },
                ]
                in (%2)"
            },
        );
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input.clone(), extent]).unwrap();
        assert_eq!(primal_outputs[0], input);
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap());
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
    }

    #[test]
    fn test_parallel_sum_scatter_differentiation_weighted_chunks() {
        // Distinct weights on the participants' output chunks verify that the pullback gathers each chunk's
        // cotangent to every summand, including when the physical batch axis follows the scatter axis.
        check_gradient!(
            |inputs| {
                let scattered = batch(
                    |item| {
                        let operation =
                            ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled());
                        let mut outputs = item.dispatch_domain().bind(operation, Vec::new(), &[item])?;
                        Ok::<_, ProgramError>(outputs.remove(0))
                    },
                    inputs,
                    BatchAxis::new(1),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?;
                let first = scattered.slice(&[0, 0], &[1, 2], &[1, 1])?;
                let second = scattered.slice(&[1, 0], &[2, 2], &[1, 1])?;
                let weighted = first + second.clone() + second;
                weighted.reduce(&[0, 1], ReductionKind::Sum)
            },
            at = Array::matrix(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_parallel_sum_scatter_differentiation_shadows_manual_axis() {
        // Differentiation inside an inner named batch that shadows the manual mesh axis `x` keeps a structural-zero
        // tangent and lets the batch binder resolve the shadowed name instead of rejecting the primal in standalone
        // operation inference. The local sums preserve the mesh invariance of the input.
        let mesh = manual_mesh();
        let sharding = Sharding::replicated(mesh.clone(), 2);
        let expected = indoc! {"
            lambda %0:f32[2, 4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}]}] .
            let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] = reduce [kind=sum, axes=[0]] %0
                %2:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}]}] = reshape [shape=[2, 2]] %1
            in (%2)"
        };
        let input_type = ArrayType::new_static(DataType::F32, [2, 4]).with_sharding(sharding.clone()).unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(sharding).unwrap();
        let (differentiated_output_type, program) =
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| {
                    let batch_context = BatchingContext::<_, ArrayBatchingPolicy>::new(input.dispatch_domain(), 2)
                        .with_axis_name("x".to_string());
                    let item = BatchingTracer::new(batch_context.clone(), ArrayBatch::new(input, BatchAxis::new(0))?);
                    let context = DifferentiationContext::fused(batch_context);
                    let item =
                        DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(item)?, context.clone());
                    let operation = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled());
                    let mut outputs = context.bind(operation, Vec::new(), &[item])?;
                    let output = outputs.remove(0);
                    assert!(output.tangent().is_zero());
                    Ok(output.primal().clone().into_batch().into_value())
                },
                input_type.clone(),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap();
        assert_eq!(differentiated_output_type, expected_type);
        assert_eq!(program.to_string(), expected);
    }

    #[test]
    fn test_parallel_sum_scatter_differentiation_shadows_manual_axis_unreduced() {
        // Differentiation inside an inner named batch that shadows the manual mesh axis `x` keeps a structural-zero
        // tangent and lets the batch binder resolve the shadowed name instead of rejecting the primal in standalone
        // operation inference. The local sums preserve the pending sum over the shadowed manual mesh axis.
        let mesh = manual_mesh();
        let sharding = Sharding::replicated(mesh.clone(), 2).with_unreduced_axes(["x"]).unwrap();
        let expected = indoc! {"
            lambda %0:f32[2, 4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}], unreduced={'x'}}] .
            let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], unreduced={'x'}}] = \
                    reduce [kind=sum, axes=[0]] %0
                %2:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}, {}], unreduced={'x'}}] = \
                    reshape [shape=[2, 2]] %1
            in (%2)"
        };
        let input_type = ArrayType::new_static(DataType::F32, [2, 4]).with_sharding(sharding.clone()).unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(sharding).unwrap();
        let (differentiated_output_type, program) =
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| {
                    let batch_context = BatchingContext::<_, ArrayBatchingPolicy>::new(input.dispatch_domain(), 2)
                        .with_axis_name("x".to_string());
                    let item = BatchingTracer::new(batch_context.clone(), ArrayBatch::new(input, BatchAxis::new(0))?);
                    let context = DifferentiationContext::fused(batch_context);
                    let item =
                        DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(item)?, context.clone());
                    let operation = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled());
                    let mut outputs = context.bind(operation, Vec::new(), &[item])?;
                    let output = outputs.remove(0);
                    assert!(output.tangent().is_zero());
                    Ok(output.primal().clone().into_batch().into_value())
                },
                input_type.clone(),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap();
        assert_eq!(differentiated_output_type, expected_type);
        assert_eq!(program.to_string(), expected);
    }

    #[test]
    fn test_parallel_sum_scatter_transposition() {
        // A sum-scatter hands every participant its chunk of the summed cotangents, so its transpose gathers the output
        // cotangent into a varying all-gather, and transposing that recovers the sum-scatter.
        let program = collective_program(
            ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()),
            ArrayType::new_static(DataType::F32, [8]),
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();

        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[8] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ] %0
                in (%1)"
            },
        );
        assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());

        // Untiled scattering removes the axis; its adjoint reinstates that axis at the original position.
        let untiled = collective_program(
            ParallelSumScatterOperation::new("x".to_string(), 2, 1, CollectiveOptions::default()),
            ArrayType::new_static(DataType::F32, [3, 2, 4]),
        );
        let transposed = untiled.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[3, 4] .
                let %1:f32[3, 2, 4] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concat_axis=1,
                    options=Untiled,
                    output_variance=Varying,
                ] %0
                in (%1)"
            },
        );
        assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), untiled.to_string());
    }

    #[test]
    fn test_parallel_sum_scatter_transposition_array_ir_extent() {
        // The composite family transposes the array input through the same all-gather and gives the explicit extent a
        // structural-zero cotangent.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let output_cotangent = context.input(ArrayType::new_static(DataType::F32, [3]).into());
        let mut context = TranspositionContext::new(context);
        let inputs = [
            PartialValue::Unknown(ArrayType::new_static(DataType::F32, [3]).into()),
            PartialValue::Unknown(DimensionValue::constant(3).unwrap().r#type().into_owned().into()),
        ];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        transpose_mixed_operation(
            &mut context,
            &ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
            &inputs,
            &[MaybeZero::Value(output_cotangent)],
            &accumulators,
        )
        .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        let [MaybeZero::Value(array), MaybeZero::Zero(_)] = cotangents.as_slice() else {
            panic!("expected an array cotangent and a structural-zero extent cotangent");
        };
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![array.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:f32[3] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=1,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_sum_scatter_transposition_preserves_reduction_state() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        // Restore varying and reduced cotangents on `x`, preserving independent state on `y`.
        check_parallel_sum_scatter_transposition_reduction_state(
            replicated.clone().with_varying_manual_axes(["x"]).unwrap(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[8][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                    mesh=['x'=2:manual, 'y'=2:manual],
                ] %0
                in (%1)"
            },
        );
        check_parallel_sum_scatter_transposition_reduction_state(
            replicated.clone().with_unreduced_axes(["x"]).unwrap(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[8][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], reduced={'x'}}] = \
                    parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Reduced,
                    mesh=['x'=2:manual, 'y'=2:manual],
                ] %0
                in (%1)"
            },
        );
        check_parallel_sum_scatter_transposition_reduction_state(
            replicated.clone().with_unreduced_axes(["x", "y"]).unwrap(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], reduced={'y'}, \
                    varying_manual={'x'}}] .
                let %1:f32[8][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], reduced={'x', 'y'}}] = \
                    parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Reduced,
                    mesh=['x'=2:manual, 'y'=2:manual],
                ] %0
                in (%1)"
            },
        );
        check_parallel_sum_scatter_transposition_reduction_state(
            replicated.clone().with_unreduced_axes(["x"]).unwrap().with_reduced_axes(["y"]).unwrap(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], unreduced={'y'}, \
                    varying_manual={'x'}}] .
                let %1:f32[8][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], unreduced={'y'}, reduced={'x'}}] \
                    = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Reduced,
                    mesh=['x'=2:manual, 'y'=2:manual],
                ] %0
                in (%1)"
            },
        );
        check_parallel_sum_scatter_transposition_reduction_state(
            replicated.clone().with_varying_manual_axes(["x"]).unwrap().with_unreduced_axes(["y"]).unwrap(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], reduced={'y'}, \
                    varying_manual={'x'}}] .
                let %1:f32[8][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], reduced={'y'}, \
                    varying_manual={'x'}}] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                    mesh=['x'=2:manual, 'y'=2:manual],
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_sum_scatter_transposition_composes_reduction_state() {
        // Scattering over separate axes completes each pending sum independently. Transposition reconstructs the
        // doubly reduced cotangent by composing reduced gathers in the reverse order, in both array representations.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let input_type = ArrayType::new_static(DataType::F32, [8])
            .with_sharding(replicated.clone().with_unreduced_axes(["x", "y"]).unwrap())
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type.clone());
        let scatter_x =
            ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()).with_mesh(mesh.clone());
        let intermediate = builder.add_instruction(scatter_x.clone(), Vec::new(), vec![input], None).unwrap()[0];
        let scatter_y =
            ParallelSumScatterOperation::new("y".to_string(), 2, 0, CollectiveOptions::tiled()).with_mesh(mesh);
        let output = builder.add_instruction(scatter_y.clone(), Vec::new(), vec![intermediate], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.output_types(),
            vec![
                ArrayType::new_static(DataType::F32, [2])
                    .with_sharding(replicated.with_varying_manual_axes(["x", "y"]).unwrap())
                    .unwrap(),
            ],
        );
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], reduced={'y'}, \
                    varying_manual={'x'}}] = parallel_all_gather [
                    axis_name=\"y\",
                    axis_size=2,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Reduced,
                    mesh=['x'=2:manual, 'y'=2:manual],
                ] %0
                    %2:f32[8][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], reduced={'x', 'y'}}] = \
                        parallel_all_gather [
                        axis_name=\"x\",
                        axis_size=2,
                        concat_axis=0,
                        options=Tiled,
                        output_variance=Reduced,
                        mesh=['x'=2:manual, 'y'=2:manual],
                    ] %1
                in (%2)"
            },
        );
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let intermediate_extent = builder.add_constant(DimensionValue::constant(4).unwrap().into());
        let output_extent = builder.add_constant(DimensionValue::constant(2).unwrap().into());
        let intermediate =
            builder.add_instruction(scatter_x, Vec::new(), vec![input, intermediate_extent], None).unwrap()[0];
        let output =
            builder.add_instruction(scatter_y, Vec::new(), vec![intermediate, output_extent], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] .
                let %1:dimension<2> = const 2
                    %2:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], reduced={'y'}, \
                        varying_manual={'x'}}] = parallel_all_gather [
                        axis_name=\"y\",
                        axis_size=2,
                        concat_axis=0,
                        options=Tiled,
                        output_variance=Reduced,
                        mesh=['x'=2:manual, 'y'=2:manual],
                    ] %0
                    %3:dimension<4> = const 4
                    %4:f32[8][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], reduced={'x', 'y'}}] = \
                        parallel_all_gather [
                        axis_name=\"x\",
                        axis_size=2,
                        concat_axis=0,
                        options=Tiled,
                        output_variance=Reduced,
                        mesh=['x'=2:manual, 'y'=2:manual],
                    ] %2
                in (%4)"
            },
        );
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter() {
        // Untiled mode removes the scatter axis and gives batch item `i` element `i` of the sum, including when the
        // mapped input axis is not leading, in both representations.
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    item.parallel_sum_scatter("x", 0)
                },
                Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 4.0]).unwrap(),
                BatchAxis::new(1),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::vector(vec![4.0, 6.0]).unwrap()),
        );
        assert_eq!(
            batch(
                |item: BatchingTracer<
                    EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                    ArrayIrBatchingPolicy,
                >| { item.parallel_sum_scatter("x", 0) },
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 4.0]).unwrap()),
                BatchAxis::new(1),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(ArrayIrValue::Array(Array::vector(vec![4.0, 6.0]).unwrap())),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter_tiled() {
        // Tiled mode gives batch item `i` the `i`-th contiguous chunk of the sum and preserves a non-leading mapped
        // axis, in both representations.
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    item.parallel_sum_scatter_tiled("x", 0)
                },
                Array::matrix(4, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]).unwrap(),
                BatchAxis::new(1),
                BatchAxis::new(1),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::matrix(2, 2, vec![11.0, 33.0, 22.0, 44.0]).unwrap()),
        );
        assert_eq!(
            batch(
                |item: BatchingTracer<
                    EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                    ArrayIrBatchingPolicy,
                >| { item.parallel_sum_scatter_tiled("x", 0) },
                ArrayIrValue::Array(Array::matrix(4, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]).unwrap()),
                BatchAxis::new(1),
                BatchAxis::new(1),
                BatchAxisSpecification::named("x"),
            ),
            Ok(ArrayIrValue::Array(Array::matrix(2, 2, vec![11.0, 33.0, 22.0, 44.0]).unwrap())),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter_with_options() {
        // Participant groups of two set the group size that the static untiled scatter extent must match, so no
        // runtime assertion is staged, and the remaining static extent is staged as a constant extent value.
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| {
                input.parallel_sum_scatter_with_options(
                    "x",
                    0,
                    CollectiveOptions::default().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
                )
            },
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 3])),
            vec![("x".to_string(), NamedAxis::Batched { size: Some(4) })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:dimension<3> = constant [value=3]
                    %2:f32[3] = parallel_sum_scatter [
                        axis_name=\"x\",
                        axis_size=4,
                        scatter_axis=0,
                        options=CollectiveOptions { mode: Untiled, axis_index_groups: [[0, 2], [3, 1]] },
                    ] %0 %1
                in (%2)"
            },
        );
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter_with_options_unbound_axis() {
        // A different bound name does not authorize the requested collective axis.
        assert_eq!(
            batch(
                |item: BatchingTracer<
                    EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                    ArrayIrBatchingPolicy,
                >| item.parallel_sum_scatter_with_options("y", 0, CollectiveOptions::default()),
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Err::<ArrayIrValue<Array>, _>(BatchingError::Axis(AxisError::UnboundAxisName { name: "y".to_string() })),
        );

        // Concrete arrays, and concrete composite values through their array members, are never inside an axis
        // binder, so every axis name is unbound for them.
        assert_eq!(
            Array::vector(vec![1.0, 2.0]).unwrap().parallel_sum_scatter_with_options(
                "x",
                0,
                CollectiveOptions::default(),
            ),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
        assert_eq!(
            ArrayIrValue::Array(Array::vector(vec![1.0, 2.0]).unwrap()).parallel_sum_scatter_with_options(
                "x",
                0,
                CollectiveOptions::default(),
            ),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter_with_options_invalid_axis_index_groups() {
        // Participant groups are validated against the resolved axis size before an operation is staged.
        let error = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| {
                input.parallel_sum_scatter_with_options(
                    "x",
                    0,
                    CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![0]]),
                )
            },
            ArrayType::new_static(DataType::F32, [4]),
            vec![("x".to_string(), NamedAxis::Batched { size: Some(2) })],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::Type(TypeError::invalid(
                "`parallel_sum_scatter` axis index groups contain participant 0 more than once",
            )),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter_with_options_manual_mesh() {
        // Over a manual mesh axis, the enclosing manual region supplies the mesh of the staged operation, and a varying
        // input already represents one contribution per participant.
        let mesh = manual_mesh();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_sum_scatter_with_options("x", 0, CollectiveOptions::tiled()),
            ArrayType::new_static(DataType::F32, [4])
                .with_sharding(sharding.clone().with_varying_manual_axes(["x"]).unwrap())
                .unwrap(),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_sum_scatter [
                    axis_name=\"x\",
                    axis_size=2,
                    scatter_axis=0,
                    options=Tiled,
                    mesh=['x'=2:manual, 'y'=1:manual],
                ] %0
                in (%1)"
            },
        );

        // An invariant input is first made varying through `parallel_vary`, so that every participant contributes its
        // copy.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_sum_scatter_with_options("x", 0, CollectiveOptions::tiled()),
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap(),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_vary [axis_name=\"x\"] %0
                    %2:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_sum_scatter [
                        axis_name=\"x\",
                        axis_size=2,
                        scatter_axis=0,
                        options=Tiled,
                        mesh=['x'=2:manual, 'y'=1:manual],
                    ] %1
                in (%2)"
            },
        );

        // An input that is unreduced over the axis is not made varying, because the sum-scatter completes its pending
        // sum and its output varies over the axis.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_sum_scatter_with_options("x", 0, CollectiveOptions::tiled()),
            ArrayType::new_static(DataType::F32, [4])
                .with_sharding(sharding.with_unreduced_axes(["x"]).unwrap())
                .unwrap(),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], unreduced={'x'}}] .
                let %1:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_sum_scatter [
                    axis_name=\"x\",
                    axis_size=2,
                    scatter_axis=0,
                    options=Tiled,
                    mesh=['x'=2:manual, 'y'=1:manual],
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter_with_options_array_ir_manual_mesh() {
        // The composite family supplies the mesh in the same way and stages the output extent explicitly. A varying
        // input already represents one contribution per participant.
        let mesh = manual_mesh();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_sum_scatter_with_options("x", 0, CollectiveOptions::tiled()),
            ArrayIrType::Array(
                ArrayType::new_static(DataType::F32, [4])
                    .with_sharding(sharding.clone().with_varying_manual_axes(["x"]).unwrap())
                    .unwrap(),
            ),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] .
                let %1:dimension<4> = constant [value=4]
                    %2:dimension<2> = constant [value=2]
                    %3:dimension<2> = dimension_div %1 %2
                    %4:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_sum_scatter [
                        axis_name=\"x\",
                        axis_size=2,
                        scatter_axis=0,
                        options=Tiled,
                        mesh=['x'=2:manual, 'y'=1:manual],
                    ] %0 %3
                in (%4)"
            },
        );

        // An invariant input is first made varying through its array view, so that every participant contributes its
        // copy.
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_sum_scatter_with_options("x", 0, CollectiveOptions::tiled()),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap()),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_vary [axis_name=\"x\"] %0
                    %2:dimension<4> = constant [value=4]
                    %3:dimension<2> = constant [value=2]
                    %4:dimension<2> = dimension_div %2 %3
                    %5:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_sum_scatter [
                        axis_name=\"x\",
                        axis_size=2,
                        scatter_axis=0,
                        options=Tiled,
                        mesh=['x'=2:manual, 'y'=1:manual],
                    ] %1 %4
                in (%5)"
            },
        );

        // An input that is unreduced over the axis is not made varying, because the sum-scatter completes its pending
        // sum and its output varies over the axis.
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_sum_scatter_with_options("x", 0, CollectiveOptions::tiled()),
            ArrayIrType::Array(
                ArrayType::new_static(DataType::F32, [4])
                    .with_sharding(sharding.with_unreduced_axes(["x"]).unwrap())
                    .unwrap(),
            ),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], unreduced={'x'}}] .
                let %1:dimension<4> = constant [value=4]
                    %2:dimension<2> = constant [value=2]
                    %3:dimension<2> = dimension_div %1 %2
                    %4:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_sum_scatter [
                        axis_name=\"x\",
                        axis_size=2,
                        scatter_axis=0,
                        options=Tiled,
                        mesh=['x'=2:manual, 'y'=1:manual],
                    ] %0 %3
                in (%4)"
            },
        );
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter_with_options_array_ir_dynamic_manual_mesh() {
        // An unsharded dynamic input is placed on the manual mesh and made varying, and its dynamic tiled extent
        // acquires a runtime divisibility assertion.
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(items)]));
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_sum_scatter_with_options("x", 0, CollectiveOptions::tiled()),
            ArrayIrType::Array(input_type),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: manual_mesh() })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[items] .
                let %1:f32[items][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] = broadcast [
                    output_type=f32[items][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}],
                    output_axes=[0],
                ] %0
                    %2:f32[items][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %1
                    %3:dimension<items ∈ [1, 5)> = dimension_size [axis=0] %2
                    %4:dimension<2> = constant [value=2]
                    %5:dimension<0> = constant [value=0]
                    %6:dimension<items % 2 ∈ [0, 2)> = dimension_rem %3 %4
                    %7:bool[] = compare [direction=Equal] %6 %5
                    () = assert [
                        message=\"collective extent must be divisible by the participant count\",
                        labels=[\"extent\", \"divisor\"],
                    ] %7 %3 %4
                    %8:dimension<items / 2 ∈ [0, 3)> = dimension_div %3 %4
                    %9:f32[items / 2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_sum_scatter [
                        axis_name=\"x\",
                        axis_size=2,
                        scatter_axis=0,
                        options=Tiled,
                        mesh=['x'=2:manual, 'y'=1:manual],
                    ] %2 %8
                in (%9)"
            },
        );
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter_with_options_invalid_static_extents() {
        // Static scatter extents are checked against the group size before an operation is staged.
        let error = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| {
                input.parallel_sum_scatter_with_options(
                    "x",
                    0,
                    CollectiveOptions::new(CollectiveMode::Untiled)
                        .with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
                )
            },
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 3])),
            vec![("x".to_string(), NamedAxis::Batched { size: Some(4) })],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::Type(TypeError::invalid(
                "`parallel_sum_scatter` untiled scatter axis 0 size 3 must equal group size 2",
            )),
        );
        let error = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| {
                input.parallel_sum_scatter_with_options(
                    "x",
                    0,
                    CollectiveOptions::new(CollectiveMode::Tiled).with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
                )
            },
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 3])),
            vec![("x".to_string(), NamedAxis::Batched { size: Some(4) })],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::Type(TypeError::invalid(
                "`parallel_sum_scatter` scatter axis 0 size 3 is not divisible by group size 2",
            )),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter_with_options_dynamic_untiled_extents() {
        // A dynamic untiled scatter extent is asserted at execution to equal the group size rather than the enclosing
        // axis size.
        let dimension = DimensionVariable::new("length", DimensionBounds::new(1, Some(9)).unwrap());
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| {
                input.parallel_sum_scatter_with_options(
                    "x",
                    0,
                    CollectiveOptions::new(CollectiveMode::Untiled)
                        .with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
                )
            },
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![dimension.into(), Dimension::Static(3)]))),
            vec![("x".to_string(), NamedAxis::Batched { size: Some(4) })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[length, 3] .
                let %1:dimension<length ∈ [1, 9)> = dimension_size [axis=0] %0
                    %2:dimension<2> = constant [value=2]
                    %3:bool[] = compare [direction=Equal] %1 %2
                    () = assert [
                        message=\"collective axis extent must match the participant count\",
                        labels=[\"extent\", \"participants\"],
                    ] %3 %1 %2
                    %4:dimension<3> = constant [value=3]
                    %5:f32[3] = parallel_sum_scatter [
                        axis_name=\"x\",
                        axis_size=4,
                        scatter_axis=0,
                        options=CollectiveOptions { mode: Untiled, axis_index_groups: [[0, 2], [3, 1]] },
                    ] %0 %4
                in (%5)"
            },
        );
        let error = program.interpret(ArrayIrValue::Array(Array::matrix(3, 3, vec![1f32; 9]).unwrap())).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "collective axis extent must match the participant count".to_string(),
                observations: vec![
                    ("extent".to_string(), "3".to_string()),
                    ("participants".to_string(), "2".to_string()),
                ],
            }),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter_with_options_dynamic_tiled_extents() {
        // A dynamic tiled scatter extent is asserted at execution to be divisible by the group size rather than the
        // enclosing axis size.
        let dimension = DimensionVariable::new("length", DimensionBounds::new(1, Some(9)).unwrap());
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| {
                input.parallel_sum_scatter_with_options(
                    "x",
                    0,
                    CollectiveOptions::new(CollectiveMode::Tiled).with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
                )
            },
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![dimension.into(), Dimension::Static(3)]))),
            vec![("x".to_string(), NamedAxis::Batched { size: Some(4) })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[length, 3] .
                let %1:dimension<length ∈ [1, 9)> = dimension_size [axis=0] %0
                    %2:dimension<3> = constant [value=3]
                    %3:dimension<2> = constant [value=2]
                    %4:dimension<0> = constant [value=0]
                    %5:dimension<length % 2 ∈ [0, 2)> = dimension_rem %1 %3
                    %6:bool[] = compare [direction=Equal] %5 %4
                    () = assert [
                        message=\"collective extent must be divisible by the participant count\",
                        labels=[\"extent\", \"divisor\"],
                    ] %6 %1 %3
                    %7:dimension<length / 2 ∈ [0, 5)> = dimension_div %1 %3
                    %8:f32[length / 2, 3] = parallel_sum_scatter [
                        axis_name=\"x\",
                        axis_size=4,
                        scatter_axis=0,
                        options=CollectiveOptions { mode: Tiled, axis_index_groups: [[0, 2], [3, 1]] },
                    ] %0 %7 %2
                in (%8)"
            },
        );
        let error = program.interpret(ArrayIrValue::Array(Array::matrix(3, 3, vec![1f32; 9]).unwrap())).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "collective extent must be divisible by the participant count".to_string(),
                observations: vec![("extent".to_string(), "3".to_string()), ("divisor".to_string(), "2".to_string())],
            }),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter_with_options_projected_value() {
        // A projected array view of a composite value stages the sum-scatter through its composite value, so that the
        // output extent is staged as an explicit extent value.
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| {
                let array = ValueProjection::<ArrayType>::into_projected(input)?;
                Ok(array.parallel_sum_scatter_with_options("x", 0, CollectiveOptions::tiled())?.into_value())
            },
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
            vec![("x".to_string(), NamedAxis::Batched { size: Some(2) })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:dimension<4> = constant [value=4]
                    %2:dimension<2> = constant [value=2]
                    %3:dimension<2> = dimension_div %1 %2
                    %4:f32[2] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=0, options=Tiled] %0 %3
                in (%4)"
            },
        );
    }
}
