//! Contains the named-axis [`ParallelAllGatherOperation`], which concatenates every participant's input across a named
//! axis, together with its interpretation, partial-evaluation, batching, forward-mode differentiation, and
//! transposition rules.

// TODO(eaplatanios): Review this module.

use std::fmt::Display;

use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrContext, ArrayIrType,
    ArrayIrValue, ArrayType, Dimension, DimensionOperation, DimensionType, DimensionValue, DimensionVariable,
    LinearResiduals, LogicalMesh, RaggedAxis, Shape, Sharding,
};
use crate::axes::{AxisError, NamedAxes, NamedAxis};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    MemberBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, MemberDifferentiableOperation,
    TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver, MemberInterpretableOperation};
use crate::macros::check_count;
use crate::operations::arithmetic::{AddOperation, Div, Mul, Rem};
use crate::operations::assertions::Assert;
use crate::operations::comparisons::Compare;
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
use crate::operations::dimensions::dimension_max::DimensionMax;
use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{DynamicBroadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::slicing::DynamicSliceOperation;
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, MemberOperation, Operation, OperationFormatter, OperationProjection, ProgramError, ProjectedValue,
    RegionInterface, Type, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

use super::axis_index::AxisIndexOperation;
use super::parallel_sum_scatter::ParallelSumScatterOperation;
use super::parallel_vary::{PARALLEL_VARY_OPERATION_NAME, ParallelVary, ParallelVaryOperation};
use super::{
    CollectiveArrayExtentBatchingPolicy, CollectiveMode, CollectiveOptions, LinearCollectiveOperation,
    ShapeChangingCollectiveBatching, ShapeChangingCollectiveOperation, ShapeChangingCollectiveValue,
    infer_array_ir_shape_changing_collective_output_type, infer_linear_collective_operation_output_type,
    resolve_named_axis_size, validate_manual_mesh_input,
};

/// Named-axis variance carried by an all-gather result.
///
/// This is an operation option rather than parallel type metadata. Type inference maps it onto the canonical
/// [`Sharding::varying_manual_axes`](crate::arrays::Sharding::varying_manual_axes) and
/// [`Sharding::reduced_axes`](crate::arrays::Sharding::reduced_axes) sets.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ParallelAllGatherOutputVariance {
    /// The result continues to vary across the gathered manual mesh axis.
    #[default]
    Varying,

    /// The result is invariant across the gathered manual mesh axis.
    Invariant,

    /// The result records the gathered manual mesh axis as reduced.
    Reduced,
}

/// Canonical operation name for [`ParallelAllGatherOperation`].
pub const PARALLEL_ALL_GATHER_OPERATION_NAME: &str = "parallel_all_gather";

/// [`Operation`] that concatenates every participant's input along `concat_axis` across the named axis, so every
/// participant receives the full concatenation — the analogue of
/// [JAX's `all_gather`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_gather.html) with `tiled = True`
/// and [StableHLO's `all_gather`](https://openxla.org/stablehlo/spec#all_gather). The output extends
/// `concat_axis` by the axis size; all other dimensions are unchanged. The collective is linear and its
/// transpose depends on the requested output variance: varying results use [`ParallelSumScatterOperation`],
/// invariant results select the current participant's chunk locally, and reduced results use sum-scatter while
/// consuming the cotangent's unreduced-axis state. A matching `batch` level consumes the mapped batch axis by
/// merging it item-major into `concat_axis`, replicating the gathered value across the batch items.
///
/// Untiled batching co-moves bounded ragged metadata with the gathered value: the named participant axis becomes
/// an ordinary output axis and is added to each participant-varying extent array's `extent_axes` mapping. Tiled
/// gathering of a ragged carrier is rejected because fusing the participant and concatenation axes can make live
/// chunks non-prefix-shaped, which one [`RaggedAxis`] cannot represent faithfully.
///
/// An all-gather over a manual mesh axis is created by [`with_mesh`](Self::with_mesh), and
/// [`ParallelAllGather::parallel_all_gather_with_options`] supplies the mesh automatically from the enclosing
/// manual region. Its input must vary over the axis (refer to [`ParallelVary`]), and its output variance selects
/// the manual variation of the result. An ordinary all-gather carries no mesh and preserves the input's mesh state,
/// even when its input carries a manual mesh axis with the same name, because a `batch` level whose axis name
/// shadows that mesh axis may bind it instead. A matching `batch` level rejects all-gathers over a manual mesh
/// axis, and only those support reduced output variance.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelAllGatherOperation {
    /// Axis name referenced by this collective.
    axis_name: String,

    /// Number of participants along the named axis, resolved from the active [`NamedAxes`] environment
    /// when the operation is staged.
    axis_size: usize,

    /// Axis of the input along which the participants' values are concatenated.
    concat_axis: usize,

    /// Shared rank and participant-group semantics.
    options: CollectiveOptions,

    /// Named-axis variance of the result.
    output_variance: ParallelAllGatherOutputVariance,

    /// Refer to the documentation of [`mesh`](Self::mesh) for more information.
    mesh: Option<LogicalMesh>,
}

impl ParallelAllGatherOperation {
    /// Creates a new [`ParallelAllGatherOperation`] over the axis with the provided name and resolved axis size.
    #[inline]
    pub fn new(
        axis_name: String,
        axis_size: usize,
        concat_axis: usize,
        options: CollectiveOptions,
        output_variance: ParallelAllGatherOutputVariance,
    ) -> Self {
        Self { axis_name, axis_size, concat_axis, options, output_variance, mesh: None }
    }

    /// Returns this [`ParallelAllGatherOperation`] configured to gather over a manual axis of `mesh`. The input must
    /// vary over [`axis_name`](Self::axis_name) on that mesh, whose size must equal [`axis_size`](Self::axis_size).
    /// Type inference validates these requirements. [`ParallelAllGather::parallel_all_gather_with_options`] supplies
    /// the mesh automatically from the enclosing manual region.
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

    /// Returns the axis of the input along which the participants' values are concatenated.
    #[inline]
    pub fn concat_axis(&self) -> usize {
        self.concat_axis
    }

    /// Returns the shared rank and participant-group semantics.
    #[inline]
    pub fn options(&self) -> &CollectiveOptions {
        &self.options
    }

    /// Returns the named-axis variance of the result.
    #[inline]
    pub fn output_variance(&self) -> ParallelAllGatherOutputVariance {
        self.output_variance
    }

    /// Returns the logical mesh whose manual axis this [`ParallelAllGatherOperation`] gathers over, or [`None`] for an
    /// ordinary all-gather, whose named axis may be bound by any enclosing binder. Only an all-gather over a manual
    /// mesh axis validates and updates the manual variation of its input.
    #[inline]
    pub fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

    /// Applies an all-gather's named-axis variance transition to the canonical sharding metadata of the shape-only
    /// `output_type` shared by the static and array IR inference paths. An ordinary all-gather carries no mesh and
    /// preserves the mesh state of its input, even when its input carries a manual mesh axis with the same name,
    /// because a `batch` level whose axis name shadows that mesh axis may bind it instead. Over a manual mesh axis, the
    /// input must vary over the axis, and the output variance selects whether the result keeps varying over it,
    /// becomes invariant over it, or records it as reduced.
    fn finalize_output_type(&self, input_type: &ArrayType, mut output_type: ArrayType) -> Result<ArrayType, TypeError> {
        let axis_name = self.axis_name();
        let Some(mesh) = &self.mesh else {
            if self.output_variance == ParallelAllGatherOutputVariance::Reduced {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` with reduced output variance requires a manual mesh axis",
                )));
            }
            return Ok(output_type);
        };

        validate_manual_mesh_input(
            PARALLEL_ALL_GATHER_OPERATION_NAME,
            axis_name,
            Some(self.axis_size),
            mesh,
            input_type,
        )?;

        // Gathering across this pending sum would change its reduction semantics, but independent sums commute.
        let input_sharding = input_type.sharding().unwrap();
        if input_sharding.unreduced_axes().contains(axis_name) {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` does not support unreduced inputs",
            )));
        }

        // Gathering an input that is still invariant over the axis would concatenate identical copies under a type that
        // cannot tell them apart from per-participant values, and its transpose would produce a varying cotangent.
        if !input_sharding.varying_manual_axes().contains(axis_name) {
            return Err(TypeError::invalid(format!(
                "`{}` input must vary over manual axis `{}`; pass an invariant value \
                 through `{}` first so that every copy is gathered",
                PARALLEL_ALL_GATHER_OPERATION_NAME, axis_name, PARALLEL_VARY_OPERATION_NAME,
            )));
        }

        let mut varying_axes = input_sharding.varying_manual_axes().clone();
        let mut reduced_axes = input_sharding.reduced_axes().clone();
        match self.output_variance {
            ParallelAllGatherOutputVariance::Varying => {}
            ParallelAllGatherOutputVariance::Invariant => {
                varying_axes.remove(axis_name);
            }
            ParallelAllGatherOutputVariance::Reduced => {
                varying_axes.remove(axis_name);
                reduced_axes.insert(axis_name.to_string());
            }
        }

        // The shape-only output type preserves the input sharding, which exists here.
        let output_sharding = output_type.sharding().unwrap().clone();
        output_type.sharding = Some(
            output_sharding
                .with_varying_manual_axes(varying_axes)
                .and_then(|sharding| sharding.with_reduced_axes(reduced_axes))
                .map_err(TypeError::from)?,
        );
        Ok(output_type)
    }
}

impl LinearCollectiveOperation for ParallelAllGatherOperation {
    type Adjoint = ParallelSumScatterOperation;

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

    fn effective_axis_size(&self) -> Result<usize, TypeError> {
        if self.output_variance != ParallelAllGatherOutputVariance::Varying && self.options.axis_index_groups.is_some()
        {
            return Err(TypeError::invalid(
                "`parallel_all_gather` axis index groups are not supported with invariant or reduced output variance"
                    .to_string(),
            ));
        }
        self.options.effective_axis_size(PARALLEL_ALL_GATHER_OPERATION_NAME, self.axis_size)
    }

    fn adjoint(&self, _input_type: &ArrayType) -> Result<ParallelSumScatterOperation, ProgramError> {
        // A varying all-gather is the adjoint of a sum-scatter with the same mode, axis, participant groups, and mesh,
        // and so is a reduced one, whose unreduced cotangent the sum-scatter consumes. An invariant all-gather instead
        // needs the residual-aware composite adjoint, because its pullback depends on participant-indexed geometry.
        if self.output_variance == ParallelAllGatherOutputVariance::Invariant {
            return Err(ProgramError::UnsupportedOperation {
                message:
                    "direct transposition of invariant `parallel_all_gather` cannot represent the participant-indexed \
                          slice; linearize so that the current participant can select its gathered chunk"
                        .to_string(),
            });
        }
        let adjoint = ParallelSumScatterOperation::new(
            self.axis_name.clone(),
            self.axis_size,
            self.concat_axis,
            self.options.clone(),
        );
        Ok(match &self.mesh {
            Some(mesh) => adjoint.with_mesh(mesh.clone()),
            None => adjoint,
        })
    }

    #[inline]
    fn adapt_to_batch_axis(&self, input_batch_axis: usize) -> (Self, usize) {
        let (concat_axis, output_batch_axis) =
            self.options.mode.forwarded_concatenation_axes(self.concat_axis, input_batch_axis);
        (Self { concat_axis, ..self.clone() }, output_batch_axis)
    }
}

impl ShapeChangingCollectiveOperation for ParallelAllGatherOperation {
    #[inline]
    fn collective_options(&self) -> &CollectiveOptions {
        &self.options
    }

    fn infer_array_ir_output_types(&self, input_types: &[ArrayIrType]) -> Result<Vec<ArrayIrType>, TypeError> {
        let effective_axis_size = self.effective_axis_size()?;
        let Some(input_type) = input_types.first() else {
            return Err(TypeError::invalid("`parallel_all_gather` expects an array followed by its output extents"));
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        let base_output_type = match self.options.mode {
            CollectiveMode::Untiled => {
                input_type.with_inserted_dimension(self.concat_axis, Dimension::Static(effective_axis_size))?
            }
            CollectiveMode::Tiled => {
                if self.concat_axis >= input_type.rank() {
                    return Err(TypeError::invalid(format!(
                        "`parallel_all_gather` concat axis {} is out of bounds for rank {}",
                        self.concat_axis,
                        input_type.rank(),
                    )));
                }
                let mut dimensions = input_type.shape().dimensions().to_vec();
                dimensions[self.concat_axis] = Dimension::Static(0);
                let sharding =
                    input_type.resized_sharding(dimensions.as_slice(), PARALLEL_ALL_GATHER_OPERATION_NAME)?;
                let mut output_type =
                    ArrayType::new(input_type.data_type(), Shape::new(dimensions)).with_memory(input_type.memory());
                output_type.sharding = sharding;
                output_type
            }
        };
        let mut output_types = infer_array_ir_shape_changing_collective_output_type(
            PARALLEL_ALL_GATHER_OPERATION_NAME,
            input_types,
            base_output_type,
            &[self.concat_axis],
            |output_extents| {
                match self.options.mode {
                    CollectiveMode::Untiled => {
                        let output_extent = &output_extents[self.concat_axis];
                        if output_extent != &Dimension::Static(effective_axis_size) {
                            return Err(TypeError::invalid(format!(
                                "`parallel_all_gather` inserted output axis {} extent must equal axis group size \
                                 {effective_axis_size} but got {output_extent}",
                                self.concat_axis,
                            )));
                        }
                    }
                    CollectiveMode::Tiled => {
                        let input_extent = &input_type.shape().dimensions()[self.concat_axis];
                        let output_extent = &output_extents[self.concat_axis];
                        if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                            (input_extent, output_extent)
                        {
                            let expected = input_extent.checked_mul(effective_axis_size).ok_or_else(|| {
                                TypeError::invalid(
                                    "`parallel_all_gather` result extent does not fit in usize".to_string(),
                                )
                            })?;
                            if *output_extent != expected {
                                return Err(TypeError::invalid(format!(
                                    "`parallel_all_gather` result extent must equal input axis {} extent \
                                     {input_extent} multiplied by axis group size {effective_axis_size}; expected \
                                     {expected} but got {output_extent}",
                                    self.concat_axis,
                                )));
                            }
                        }
                    }
                }
                Ok(())
            },
        )?;
        let mut output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
        if self.options.mode == CollectiveMode::Tiled {
            // The placeholder zero cannot prove divisibility of the actual result by its explicit mesh placement.
            output_type.sharding =
                input_type.resized_sharding(output_type.shape().dimensions(), PARALLEL_ALL_GATHER_OPERATION_NAME)?;
            if output_type.shape() == input_type.shape() {
                output_type = output_type.with_layout(input_type.layout().cloned());
            }
        }
        Ok(vec![self.finalize_output_type(input_type, output_type)?.into()])
    }
}

impl<C: Context<Type = ArrayType, Value: Transpose>> ShapeChangingCollectiveBatching<C> for ParallelAllGatherOperation {
    fn batch_matching_axis<P: CollectiveArrayExtentBatchingPolicy<C>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        input: &ArrayBatch<C::Value>,
        output_extents: Vec<P::ShapeExtent>,
        output_sharding: Option<Sharding>,
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        let logical_input_rank = input.unbatched_type().rank();
        if self.options.axis_index_groups.is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: "`parallel_all_gather` axis index groups are not supported when a batch transform binds the \
                     collective axis"
                    .to_string(),
            });
        }
        if self.output_variance == ParallelAllGatherOutputVariance::Reduced {
            return Err(BatchingError::UnsupportedOperation {
                message:
                    "`parallel_all_gather` with reduced output variance is not supported when a batch transform binds \
                     the collective axis"
                        .to_string(),
            });
        }
        let axis_extent =
            P::collective_axis_extent(context, PARALLEL_ALL_GATHER_OPERATION_NAME, &self.axis_name, self.axis_size)?;

        let axis_is_out_of_bounds = match self.options.mode {
            CollectiveMode::Untiled => self.concat_axis > logical_input_rank,
            CollectiveMode::Tiled => self.concat_axis >= logical_input_rank,
        };
        if axis_is_out_of_bounds {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`parallel_all_gather` concat axis {} is out of bounds for rank {logical_input_rank}",
                    self.concat_axis,
                ),
            });
        }

        let mut input_extents = output_extents.clone();
        match self.options.mode {
            CollectiveMode::Untiled => {
                input_extents.remove(self.concat_axis);
            }
            CollectiveMode::Tiled => {
                input_extents[self.concat_axis] =
                    P::divide_extents_exactly(context, &output_extents[self.concat_axis], &axis_extent)?;
            }
        }
        let input = P::match_collective_axis(context, input, input_extents.as_slice())?;
        let moved = input.into_value().move_axis(0, self.concat_axis)?;
        let gathered = P::reshape_collective(context, moved, output_extents.as_slice(), output_sharding)?;
        Ok(ArrayBatch::replicated(gathered))
    }
}

impl Display for ParallelAllGatherOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ParallelAllGatherOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        PARALLEL_ALL_GATHER_OPERATION_NAME
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
                "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` does not support dynamically shaped inputs",
            )));
        };

        let effective_axis_size = self.effective_axis_size()?;
        let output_type = match self.options.mode {
            CollectiveMode::Untiled => {
                input_type.with_inserted_dimension(self.concat_axis, Dimension::Static(effective_axis_size))?
            }
            CollectiveMode::Tiled => {
                let mut output_dimensions = shape.dimensions().to_vec();
                let Some(dimension) = output_dimensions.get_mut(self.concat_axis) else {
                    return Err(TypeError::invalid(format!(
                        "`parallel_all_gather` concat axis {} is out of bounds for rank {}",
                        self.concat_axis,
                        output_dimensions.len(),
                    )));
                };
                *dimension = dimension.checked_mul(effective_axis_size).ok_or_else(|| {
                    TypeError::invalid("`parallel_all_gather` result extent does not fit in usize".to_string())
                })?;
                infer_linear_collective_operation_output_type(
                    PARALLEL_ALL_GATHER_OPERATION_NAME,
                    input_type,
                    output_dimensions,
                )?
            }
        };
        Ok(vec![self.finalize_output_type(input_type, output_type)?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, PARALLEL_ALL_GATHER_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
            operation.field("axis_size", self.axis_size)?;
            operation.field("concat_axis", format_args!("{:?}", &self.concat_axis))?;
            operation.field("options", format_args!("{:?}", &self.options))?;
            operation.field("output_variance", format_args!("{:?}", &self.output_variance))?;
            if let Some(mesh) = &self.mesh {
                operation.field("mesh", mesh)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Reshape>> InterpretableOperation<C> for ParallelAllGatherOperation {
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

        // A single participant gathers only its own value. Untiled mode inserts a size-one gathered axis, which a
        // reshape to the inferred output type expresses, while tiled mode leaves the shape unchanged.
        Ok(vec![match self.options.mode {
            CollectiveMode::Tiled => input.clone(),
            CollectiveMode::Untiled => {
                input.reshape_with_output_sharding(output_type.shape().clone(), output_type.sharding().cloned())?
            }
        }])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelAllGatherOperation>>> PartiallyEvaluatableOperation<C>
    for ParallelAllGatherOperation
{
}

// Batching rule for [`ParallelAllGatherOperation`]. A matching `batch` level consumes the mapped batch axis by
// materializing the gather: the batch axis is transposed to sit immediately before the per-item `concat_axis` and
// merged into it, laying the gathered chunks out item-major (item 0's chunk first), which matches the tiled StableHLO
// `all_gather` ordering. Every batch item sees the same gathered value, so the output is replicated. A non-matching
// level forwards the collective to the parent context, unchanged for a replicated input (through
// `BatchingContext::forward_to_parent`) and with its array axes shifted past the batch axis for a mapped one.
impl<
    C: Context<Type = ArrayType, Value: Transpose, Operation: From<ParallelAllGatherOperation>>,
    P: CollectiveArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for ParallelAllGatherOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            ArrayBatch::reject_ragged_inputs(self, inputs)?;
            return context.forward_collective(self, inputs);
        }
        self.reject_mesh_form()?;
        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };
        if self.options.mode == CollectiveMode::Tiled && !input.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: "tiled `parallel_all_gather` cannot represent participant-specific bounded ragged extents \
                          after the participant and concatenation axes are fused"
                    .to_string(),
            });
        }
        let input_type = if input.ragged_axes().is_empty() {
            input.unbatched_type()
        } else {
            input.value().r#type().unbatched(input.batch_axis())?
        };
        let (output_type, output_extents) = context.infer_collective_output_type_and_extents(self, &input_type)?;
        let input_batch_axis = input.batch_axis_position();
        let ragged_axes = input.ragged_axes().to_vec();
        let mut output = self.batch_matching_axis(context, input, output_extents, output_type.sharding().cloned())?;
        if !ragged_axes.is_empty() {
            let ragged_axes =
                gathered_ragged_axes::<C, P>(self, context, ragged_axes, input_batch_axis, input_type.rank())?;
            output = output.with_ragged_axes(ragged_axes)?;
        }
        Ok(vec![output].into())
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelAllGatherOperation>>> DifferentiableOperation<C>
    for ParallelAllGatherOperation
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
    O: Operation<Type = ArrayType> + From<AddOperation<ArrayType>> + From<ParallelSumScatterOperation>,
> TransposableOperation<V, O> for ParallelAllGatherOperation
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
        self.linear_collective_transpose(context, inputs, outputs, accumulators)
    }
}

impl MemberOperation<ArrayIrType> for ParallelAllGatherOperation {
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
> MemberInterpretableOperation<C> for ParallelAllGatherOperation
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

// Batching rule for array IR [`ParallelAllGatherOperation`]. The logical result extents remain ordinary
// replicated dimension SSA inputs; matching-axis batching delegates its array mechanics to the homogeneous collective
// kernel.
impl<C> MemberBatchableOperation<C, ArrayIrBatchingPolicy> for ParallelAllGatherOperation
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<ParallelAllGatherOperation>
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
        let Some((array, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        <&ArrayType>::try_from(&array.unbatched_type())?;
        let logical_input_types = inputs.iter().map(|input| input.unbatched_type().clone()).collect::<Vec<_>>();
        let mut logical_output_types = self.infer_array_ir_output_types(logical_input_types.as_slice())?;
        let logical_output_type = <&ArrayType>::try_from(&logical_output_types.remove(0))?.clone();

        if context.axis_name() != Some(self.axis_name()) {
            ArrayIrBatch::reject_ragged_inputs(self, inputs)?;
            // A result extent describes the shape shared by every batch item, so it must be replicated.
            for output_extent in output_extents {
                output_extent.validate_replicated_dimension()?;
            }
            return Ok(context.forward_collective(self, array, output_extents)?.into());
        }

        self.reject_mesh_form()?;

        if self.options().mode() == CollectiveMode::Tiled && !array.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: "tiled `parallel_all_gather` cannot represent participant-specific bounded ragged extents \
                          after the participant and concatenation axes are fused"
                    .to_string(),
            });
        }

        let input_batch_axis = array.batch_axis_position();
        let ragged_axes = array
            .ragged_axes()
            .iter()
            .map(|ragged_axis| {
                let logical_axis =
                    ragged_axis.axis() - usize::from(input_batch_axis.is_some_and(|axis| axis < ragged_axis.axis()));
                let output_axis = logical_axis + usize::from(logical_axis >= self.concat_axis());
                let output_extent = &output_extents[output_axis];
                let output_extent_type = output_extent.unbatched_type();
                let output_extent_type = <&DimensionType>::try_from(&output_extent_type)?;
                if output_extent_type.variable() != ragged_axis.dimension() {
                    return Err(BatchingError::InvalidBatchMetadata {
                        message: format!(
                            "untiled `parallel_all_gather` output axis {output_axis} carries dimension `{}` instead of \
                             bounded ragged dimension `{}`",
                            output_extent_type.variable(),
                            ragged_axis.dimension(),
                        ),
                    });
                }
                let extents = if let Some(input_batch_axis) = input_batch_axis {
                    let Some(extents) = output_extent.mapped_dimension_extents() else {
                        return Err(BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "untiled `parallel_all_gather` output axis {output_axis} must carry mapped extents for \
                                 bounded ragged dimension `{}`",
                                ragged_axis.dimension(),
                            ),
                        });
                    };
                    let expected_extent_axis = ragged_axis
                        .extent_axes()
                        .iter()
                        .position(|axis| *axis == input_batch_axis)
                        .map(BatchAxis::from_position)
                        .ok_or_else(|| BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "bounded ragged dimension `{}` does not carry extents for the mapped input axis",
                                ragged_axis.dimension(),
                            ),
                        })?;
                    if output_extent.batch_axis() != expected_extent_axis {
                        return Err(BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "untiled `parallel_all_gather` output axis {output_axis} maps bounded ragged extents \
                                 on {} instead of {expected_extent_axis}",
                                output_extent.batch_axis(),
                            ),
                        });
                    }
                    extents.clone()
                } else {
                    output_extent.validate_replicated_dimension()?;
                    if !ragged_axis.extent_axes().is_empty() {
                        return Err(BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "replicated bounded ragged dimension `{}` must carry scalar extents",
                                ragged_axis.dimension(),
                            ),
                        });
                    }
                    ragged_axis.extents().clone()
                };
                // The packed capacity comes from the input axis. A dimension variable's exclusive upper bound only
                // constrains logical extents and may be looser than that physical capacity.
                let physical_extent = array.value().dimension_size(ragged_axis.axis())?;
                Ok((
                    output_axis,
                    physical_extent,
                    RaggedAxis::new(
                        ragged_axis.axis(),
                        <C::Value as ValueProjection<ArrayType>>::into_projected(extents)?,
                        ragged_axis.dimension().clone(),
                        ragged_axis.extent_axes().to_vec(),
                    ),
                ))
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        let array = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(array.value().clone())?,
            array.batch_axis(),
        )?;
        let input_rank = array.unbatched_type().rank();
        let projected_context = context.array_projection();
        let output_extents = output_extents
            .iter()
            .enumerate()
            .map(|(axis, extent)| {
                if let Some((_, physical_extent, _)) =
                    ragged_axes.iter().find(|(ragged_output_axis, _, _)| *ragged_output_axis == axis)
                {
                    return Ok(<C::Value as ValueProjection<DimensionType>>::into_projected(physical_extent.clone())?);
                }
                if extent.mapped_dimension_extents().is_some() {
                    let extent_type = extent.unbatched_type();
                    let extent_type = <&DimensionType>::try_from(&extent_type)?;
                    return Err(BatchingError::InvalidBatchMetadata {
                        message: format!(
                            "untiled `parallel_all_gather` output axis {axis} has mapped dimension `{}` without a \
                             matching bounded ragged input axis",
                            extent_type.variable(),
                        ),
                    });
                }
                extent.validate_replicated_dimension()?;
                Ok(<C::Value as ValueProjection<DimensionType>>::into_projected(extent.value().clone())?)
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        let ragged_axes = ragged_axes.into_iter().map(|(_, _, ragged_axis)| ragged_axis).collect::<Vec<_>>();
        let mut output = self.batch_matching_axis::<DynamicArrayExtentBatchingPolicy>(
            &projected_context,
            &array,
            output_extents,
            logical_output_type.sharding().cloned(),
        )?;
        if !ragged_axes.is_empty() {
            let ragged_axes = gathered_ragged_axes::<_, DynamicArrayExtentBatchingPolicy>(
                self,
                &projected_context,
                ragged_axes,
                input_batch_axis,
                input_rank,
            )?;
            output = output.with_ragged_axes(ragged_axes)?;
        }
        let ragged_axes = output
            .ragged_axes()
            .iter()
            .map(|ragged_axis| {
                RaggedAxis::new(
                    ragged_axis.axis(),
                    <C::Value as ValueProjection<ArrayType>>::from_projected(ragged_axis.extents().clone()),
                    ragged_axis.dimension().clone(),
                    ragged_axis.extent_axes().to_vec(),
                )
            })
            .collect();
        let output =
            ArrayIrBatch::replicated(<C::Value as ValueProjection<ArrayType>>::from_projected(output.into_value()))
                .with_ragged_axes(ragged_axes)?;
        Ok(vec![output].into())
    }
}

impl ParallelAllGather<ArrayType> for Array {
    // A concrete `Array` never executes inside an axis binder, because the values under a `batch` level or inside
    // a manual region are tracers, so every axis name is unbound for it.

    #[inline]
    fn parallel_all_gather_with_options(
        &self,
        axis_name: &str,
        _concat_axis: usize,
        _options: CollectiveOptions,
        _output_variance: ParallelAllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        Err(AxisError::UnboundAxisName { name: axis_name.to_string() }.into())
    }
}

// A concrete composite value performs the collective through its array member.
impl<A: Value<Type = ArrayType> + ParallelAllGather<ArrayType>> ParallelAllGather<ArrayIrType> for ArrayIrValue<A> {
    fn parallel_all_gather_with_options(
        &self,
        axis_name: &str,
        concat_axis: usize,
        options: CollectiveOptions,
        output_variance: ParallelAllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        let array = ValueProjection::<ArrayType>::into_projected(self.clone())?;
        Ok(<Self as ValueProjection<ArrayType>>::from_projected(array.parallel_all_gather_with_options(
            axis_name,
            concat_axis,
            options,
            output_variance,
        )?))
    }
}

// Mixed array IR JVP for all-gather. Explicit output extents are retained as ordinary residual values, and an
// invariant result uses participant-indexed slicing in its transposed linear region.
impl<C> MemberDifferentiableOperation<C> for ParallelAllGatherOperation
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<ParallelAllGatherOperation>
        + From<DimensionFromScalarOperation>
        + From<DimensionSizeOperation>
        + From<DynamicSliceOperation<ArrayIrType>>
        + From<LinearCallOperation<ArrayIrType>>
        + From<ParallelSumScatterOperation>
        + From<DynamicReshapeOperation>
        + From<ConstantOperation<DimensionValue>>
        + OperationProjection<ArrayType>
        + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<AxisIndexOperation> + From<ParallelVaryOperation>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        if self.output_variance() == ParallelAllGatherOutputVariance::Invariant {
            return jvp_invariant_parallel_all_gather(self, context, inputs);
        }
        self.shape_changing_collective_jvp_in_parent(context, inputs)
    }
}

/// Stages an all-gather with first-class dynamic tiled extents and rank-changing untiled semantics.
///
/// The type-family parameter defaults to this value's type, so that homogeneous array values and composite array
/// values share the same call syntax.
pub trait ParallelAllGather<T: Type = <Self as Typed>::Type>: Typed<Type = T> + Sized {
    /// Stacks participants along a new axis at `concat_axis`, producing an output that varies across `axis_name`.
    #[inline]
    fn parallel_all_gather(&self, axis_name: &str, concat_axis: usize) -> Result<Self, ProgramError> {
        self.parallel_all_gather_with_options(
            axis_name,
            concat_axis,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        )
    }

    /// Concatenates participants into the existing `concat_axis`, producing an output that varies across
    /// `axis_name`.
    #[inline]
    fn parallel_all_gather_tiled(&self, axis_name: &str, concat_axis: usize) -> Result<Self, ProgramError> {
        self.parallel_all_gather_with_options(
            axis_name,
            concat_axis,
            CollectiveOptions::new(CollectiveMode::Tiled),
            ParallelAllGatherOutputVariance::Varying,
        )
    }

    /// Gathers participants using explicit shape, grouping, and output-variance semantics.
    fn parallel_all_gather_with_options(
        &self,
        axis_name: &str,
        concat_axis: usize,
        options: CollectiveOptions,
        output_variance: ParallelAllGatherOutputVariance,
    ) -> Result<Self, ProgramError>;
}

// A composite value binds a `ParallelAllGatherOperation` through its own context, followed by one explicit extent
// value per output axis. Over a manual mesh axis, the operation records the mesh, and an input that does not vary over
// the axis is first made varying through its array view, exactly as JAX's `all_gather` and `all_gather_invariant` do.
impl<V> ParallelAllGather<ArrayIrType> for V
where
    V: Value<Type = ArrayIrType>
        + Assert
        + DimensionSize<V>
        + ValueProjection<DimensionType>
        + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
    V::DispatchDomain: Context<Type = ArrayIrType> + NamedAxes,
    V::DispatchDomain: DimensionConstant,
    <V::DispatchDomain as Domain>::Operation: From<ParallelAllGatherOperation>,
    <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + Mul,
    ProjectedValue<ArrayType, V>: ParallelVary,
{
    fn parallel_all_gather_with_options(
        &self,
        axis_name: &str,
        concat_axis: usize,
        options: CollectiveOptions,
        output_variance: ParallelAllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let effective_axis_size = options.effective_axis_size(PARALLEL_ALL_GATHER_OPERATION_NAME, axis_size)?;
        if output_variance != ParallelAllGatherOutputVariance::Varying && options.axis_index_groups.is_some() {
            return Err(TypeError::invalid(
                "`parallel_all_gather` axis index groups are not supported with invariant or reduced output variance"
                    .to_string(),
            )
            .into());
        }
        let mut input = self.clone();
        let mut operation = ParallelAllGatherOperation::new(
            axis_name.to_string(),
            axis_size,
            concat_axis,
            options.clone(),
            output_variance,
        );
        if let Some(NamedAxis::Mesh { mesh, .. }) = context.named_axis(axis_name) {
            let array = ValueProjection::<ArrayType>::into_projected(self.clone())?;
            if array.r#type().unreduced_axes().contains(axis_name) {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` does not support unreduced inputs",
                ))
                .into());
            }
            if !array.r#type().sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis_name)) {
                input = <V as ValueProjection<ArrayType>>::from_projected(array.parallel_vary(axis_name)?);
            }
            operation = operation.with_mesh(mesh);
        }
        let input_type = input.r#type();
        let rank = <&ArrayType>::try_from(input_type.as_ref())?.rank();
        if concat_axis > rank || (options.mode == CollectiveMode::Tiled && concat_axis == rank) {
            return Err(TypeError::invalid(format!(
                "`parallel_all_gather` concat axis {concat_axis} is out of bounds for rank {rank}",
            ))
            .into());
        }
        let mut output_extents = (0..rank).map(|axis| input.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
        let participants = context.dimension_constant(effective_axis_size)?;
        match options.mode {
            CollectiveMode::Untiled => output_extents.insert(concat_axis, participants),
            CollectiveMode::Tiled => {
                let extent = ValueProjection::<DimensionType>::into_projected(output_extents[concat_axis].clone())?;
                let participants = ValueProjection::<DimensionType>::into_projected(participants)?;
                output_extents[concat_axis] =
                    ValueProjection::<DimensionType>::from_projected(extent.mul(&participants)?);
            }
        }
        let inputs = std::iter::once(input).chain(output_extents).collect::<Vec<_>>();
        let mut outputs = context.bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<V> ParallelAllGather<ArrayType> for ProjectedValue<ArrayType, V>
where
    V: ParallelAllGather<ArrayIrType> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
{
    fn parallel_all_gather_with_options(
        &self,
        axis_name: &str,
        concat_axis: usize,
        options: CollectiveOptions,
        output_variance: ParallelAllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        self.value()
            .parallel_all_gather_with_options(axis_name, concat_axis, options, output_variance)?
            .into_projected()
            .map_err(Into::into)
    }
}

// Homogeneous values opt into direct staging, while projected values retain composite extent delegation.
impl<V> ParallelAllGather<ArrayType> for V
where
    V: ShapeChangingCollectiveValue + ParallelVary,
    V::DispatchDomain: Context<Value = V, Operation: From<ParallelAllGatherOperation>> + NamedAxes,
{
    fn parallel_all_gather_with_options(
        &self,
        axis_name: &str,
        concat_axis: usize,
        options: CollectiveOptions,
        output_variance: ParallelAllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        options.effective_axis_size(PARALLEL_ALL_GATHER_OPERATION_NAME, axis_size)?;
        if output_variance != ParallelAllGatherOutputVariance::Varying && options.axis_index_groups.is_some() {
            return Err(TypeError::invalid(
                "`parallel_all_gather` axis index groups are not supported with invariant or reduced output variance"
                    .to_string(),
            )
            .into());
        }
        let mut input = self.clone();
        let mut operation =
            ParallelAllGatherOperation::new(axis_name.to_string(), axis_size, concat_axis, options, output_variance);
        if let Some(NamedAxis::Mesh { mesh, .. }) = context.named_axis(axis_name) {
            if input.r#type().unreduced_axes().contains(axis_name) {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` does not support unreduced inputs",
                ))
                .into());
            }
            if !input.r#type().sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis_name)) {
                input = input.parallel_vary(axis_name)?;
            }
            operation = operation.with_mesh(mesh);
        }
        let mut outputs = context.bind(operation, Vec::new(), &[input])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Applies the mixed array IR JVP for invariant all-gather. Its transpose selects the current participant's
/// gathered chunk using the retained input geometry and reshapes an untiled size-one participant axis away. Over a
/// manual mesh axis, the invariant output cotangent is first made varying over that axis, because every participant
/// selects a different chunk of it.
fn jvp_invariant_parallel_all_gather<C, P: DifferentiationPolicy<C>>(
    operation: &ParallelAllGatherOperation,
    context: &DifferentiationContext<C, P>,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<ParallelAllGatherOperation>
        + From<DimensionFromScalarOperation>
        + From<DimensionSizeOperation>
        + From<DynamicSliceOperation<ArrayIrType>>
        + From<LinearCallOperation<ArrayIrType>>
        + From<DynamicReshapeOperation>
        + From<ConstantOperation<DimensionValue>>
        + OperationProjection<ArrayType>
        + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<AxisIndexOperation> + From<ParallelVaryOperation>,
{
    let Some((array, _)) = inputs.split_first() else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    let primal = context.primal().bind(operation.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
    let tangent = match array.tangent() {
        MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
        MaybeZero::Value(array_tangent) => {
            let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
            let (array, output_extents) = tangent_inputs.split_first().unwrap();
            let context = context.tangent();
            let mut residuals = LinearResiduals::new();
            let output_extents = residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
            let input_shape = residuals.retain_shape(context, array.primal())?;
            let forward_operation = operation.clone();
            let forward_output_extents = output_extents.clone();
            let transpose_operation = operation.clone();
            let transpose_target_type = <&ArrayType>::try_from(array.primal().r#type().as_ref())?.cotangent()?;
            let tangent = LinearCallOperation::stage(
                context,
                residuals.into_values(),
                vec![array_tangent.clone()],
                move |residuals, linear_inputs| {
                    let mut collective_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                    collective_inputs.push(linear_inputs[0].clone());
                    collective_inputs.extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                    linear_inputs[0].dispatch_domain().bind(forward_operation, Vec::new(), collective_inputs.as_slice())
                },
                move |residuals, output_cotangents| {
                    let transpose_context = output_cotangents[0].dispatch_domain();
                    let input_dimensions = input_shape.dimensions(&transpose_context, residuals)?;
                    let output_cotangent_type = output_cotangents[0].r#type();
                    let output_cotangent_type = <&ArrayType>::try_from(output_cotangent_type.as_ref())?;
                    let output_rank = output_cotangent_type.rank();
                    let zero = transpose_context
                        .bind(
                            DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(0)?)),
                            Vec::new(),
                            &[],
                        )?
                        .remove(0);
                    let chunk_extent = match transpose_operation.options().mode() {
                        CollectiveMode::Tiled => input_dimensions[transpose_operation.concat_axis()].clone(),
                        CollectiveMode::Untiled => transpose_context
                            .bind(
                                DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(1)?)),
                                Vec::new(),
                                &[],
                            )?
                            .remove(0),
                    };
                    let start = if transpose_operation.axis_size() == 1 {
                        zero.clone()
                    } else {
                        let mut axis_index_operation =
                            AxisIndexOperation::new(transpose_operation.axis_name().to_string());
                        if let Some(mesh) = transpose_operation.mesh() {
                            axis_index_operation = axis_index_operation.with_mesh(mesh.clone());
                        }
                        let axis_index = transpose_context.bind_array(axis_index_operation, &[])?;
                        let axis_index_variable = DimensionVariable::new(
                            format!("{}_index", transpose_operation.axis_name()),
                            crate::arrays::DimensionBounds::non_negative(Some(transpose_operation.axis_size()))?,
                        );
                        let axis_index = transpose_context
                            .bind(
                                DimensionFromScalarOperation::new(axis_index_variable),
                                Vec::new(),
                                std::slice::from_ref(&axis_index),
                            )?
                            .remove(0);
                        let axis_index_type = <&DimensionType>::try_from(axis_index.r#type().as_ref())?.clone();
                        let chunk_extent_type = <&DimensionType>::try_from(chunk_extent.r#type().as_ref())?.clone();
                        transpose_context
                            .bind(
                                DimensionOperation::Mul(DimensionMulOperation::new(
                                    &axis_index_type,
                                    &chunk_extent_type,
                                )?),
                                Vec::new(),
                                &[axis_index, chunk_extent.clone()],
                            )?
                            .remove(0)
                    };
                    let mut starts = vec![zero; output_rank];
                    starts[transpose_operation.concat_axis()] = start;
                    let mut slice_sizes = input_dimensions.clone();
                    if transpose_operation.options().mode() == CollectiveMode::Untiled {
                        slice_sizes.insert(transpose_operation.concat_axis(), chunk_extent);
                    }
                    // Over a manual mesh axis, the output cotangent is invariant across the gathered axis, while
                    // every participant selects a different chunk of it, so the selected chunk varies over that axis.
                    // The cotangent can carry a tangent itself (e.g., under nested differentiation), and so it is
                    // made varying with a real `parallel_vary` transition, whose transpose is the cross-device sum,
                    // rather than by retyping the selected chunk.
                    let output_cotangent = match transpose_operation.mesh() {
                        Some(_)
                            if !output_cotangent_type.sharding().is_some_and(|sharding| {
                                sharding.varying_manual_axes().contains(transpose_operation.axis_name())
                            }) =>
                        {
                            transpose_context.bind_array(
                                ParallelVaryOperation::new(transpose_operation.axis_name().to_string()),
                                std::slice::from_ref(&output_cotangents[0]),
                            )?
                        }
                        _ => output_cotangents[0].clone(),
                    };
                    let mut slice_inputs = Vec::with_capacity(1 + 2 * output_rank);
                    slice_inputs.push(output_cotangent);
                    slice_inputs.extend(starts);
                    slice_inputs.extend(slice_sizes);
                    let selected = transpose_context
                        .bind(
                            DynamicSliceOperation::<ArrayIrType>::from_rank(output_rank),
                            Vec::new(),
                            slice_inputs.as_slice(),
                        )?
                        .remove(0);
                    let mut reshape_inputs = Vec::with_capacity(1 + input_dimensions.len());
                    reshape_inputs.push(selected);
                    reshape_inputs.extend(input_dimensions);
                    transpose_context.bind(
                        DynamicReshapeOperation::new().with_output_sharding(transpose_target_type.sharding().cloned()),
                        Vec::new(),
                        reshape_inputs.as_slice(),
                    )
                },
            )?
            .remove(0);
            MaybeZero::Value(tangent)
        }
    };
    Ok(vec![DifferentiationDual::new(primal, tangent)?])
}

/// Relocates bounded-ragged metadata through a matching untiled all-gather.
fn gathered_ragged_axes<C, P>(
    operation: &ParallelAllGatherOperation,
    context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
    ragged_axes: Vec<RaggedAxis<C::Value>>,
    input_batch_axis: Option<usize>,
    input_rank: usize,
) -> Result<Vec<RaggedAxis<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    P: CollectiveArrayExtentBatchingPolicy<C>,
{
    if let Some(input_batch_axis) = input_batch_axis {
        return Ok(ragged_axes
            .into_iter()
            .map(|ragged_axis| ragged_axis.moved(input_batch_axis, 0).moved(0, operation.concat_axis))
            .collect());
    }

    let output_axes = (1..=input_rank).collect::<Vec<_>>();
    ragged_axes
        .into_iter()
        .map(|ragged_axis| {
            if !ragged_axis.extent_axes().is_empty() {
                return Err(BatchingError::UnsupportedOperation {
                    message: "untiled `parallel_all_gather` requires replicated ragged inputs to carry scalar extents"
                        .to_string(),
                });
            }
            let extents =
                P::match_axis(context, &ArrayBatch::replicated(ragged_axis.extents().clone()), 0.into())?.into_value();
            let ragged_axis = ragged_axis.broadcasted(output_axes.as_slice()).moved(0, operation.concat_axis);
            Ok(RaggedAxis::new(
                ragged_axis.axis(),
                extents,
                ragged_axis.dimension().clone(),
                vec![operation.concat_axis],
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayType,
        DataType, Dimension, DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Layout, LogicalMesh,
        Memory, MeshAxis, MeshAxisType, RaggedAxis, Shape, Sharding, ShardingDimension, StridedLayout,
    };
    use crate::axes::{AxisError, NamedAxis};
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingContext, BatchingTracer, batch};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiationContext, DifferentiationDual, DifferentiationError, MemberDifferentiableOperation,
    };

    use crate::interpretation::InterpretableOperation;

    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, MemberOperation, ProgramBuilder, ProgramError};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_parallel_all_gather() {
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            4,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        );
        assert_eq!(operation.axis_name(), "x");
        assert_eq!(operation.axis_size(), 4);
        assert_eq!(operation.concat_axis(), 0);
        assert_eq!(operation.name(), PARALLEL_ALL_GATHER_OPERATION_NAME);
        assert_eq!(
            operation.to_string(),
            indoc! {r#"
                parallel_all_gather [
                    axis_name="x",
                    axis_size=4,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ]
            "#}
            .trim_end(),
        );
    }

    #[test]
    fn test_parallel_all_gather_type_inference() {
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            4,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        );
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2])],
                    output_types = [ArrayType::new_static(DataType::F32, [8])],
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "`parallel_all_gather` concat axis 0 is out of bounds for rank 0",
                },
                {
                    input_types = [ArrayType::new(
                        DataType::F32,
                        Shape::new(vec![Dimension::Dynamic(
                            DimensionVariable::new("dynamic", DimensionBounds::unbounded()),
                        )],),
                    )],
                    error = "`parallel_all_gather` does not support dynamically shaped inputs",
                },
            ],
        );

        // An all-gather over a manual mesh axis records and renders its mesh. Its input must vary over the axis on the
        // operation's mesh, and its output variance selects whether the result keeps varying over the axis.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let varying = sharding.clone().with_varying_manual_axes(["x"]).unwrap();
        let with_sharding = |extent: usize, sharding: &Sharding| {
            ArrayType::new_static(DataType::F32, [extent]).with_sharding(sharding.clone()).unwrap()
        };
        let mesh_gather = |output_variance| {
            ParallelAllGatherOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled(), output_variance)
                .with_mesh(mesh.clone())
        };
        let operation = mesh_gather(ParallelAllGatherOutputVariance::Varying);
        assert_eq!(operation.mesh(), Some(&mesh));
        assert_eq!(
            operation.to_string(),
            indoc! {r#"
                parallel_all_gather [
                    axis_name="x",
                    axis_size=2,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                    mesh=['x'=2:manual],
                ]
            "#}
            .trim_end(),
        );
        let other_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        check_operation_type_inference!(
            operation = operation,
            cases = [
                { input_types = [with_sharding(2, &varying)], output_types = [with_sharding(4, &varying)] },
                {
                    input_types = [with_sharding(2, &sharding)],
                    error = "`parallel_all_gather` input must vary over manual axis `x`; pass an invariant value \
                             through `parallel_vary` first so that every copy is gathered",
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2])],
                    error = "`parallel_all_gather` input must carry a mesh containing manual axis `x`",
                },
                {
                    input_types = [with_sharding(
                        2,
                        &Sharding::replicated(other_mesh, 1).with_varying_manual_axes(["x"]).unwrap(),
                    )],
                    error = "`parallel_all_gather` input mesh does not match the operation mesh",
                },
            ],
        );
        check_operation_type_inference!(
            operation = mesh_gather(ParallelAllGatherOutputVariance::Invariant),
            cases = [{ input_types = [with_sharding(2, &varying)], output_types = [with_sharding(4, &sharding)] }],
        );

        // The mesh axis must be manual, and its size must equal the axis size of the operation.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .with_mesh(explicit_mesh),
            cases = [{
                input_types = [with_sharding(2, &varying)],
                error = "`parallel_all_gather` mesh axis `x` must be manual",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                4,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .with_mesh(mesh.clone()),
            cases = [{
                input_types = [with_sharding(2, &varying)],
                error = "`parallel_all_gather` axis size 4 does not match the size of manual mesh axis `x`",
            }],
        );

        // An ordinary all-gather preserves the mesh state of its input, including invariance over a manual mesh axis
        // with the same name, because a `batch` level that shadows that mesh axis may bind it.
        check_operation_type_inference!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            ),
            cases = [{ input_types = [with_sharding(2, &sharding)], output_types = [with_sharding(4, &sharding)] }],
        );
    }

    #[test]
    fn test_parallel_all_gather_type_inference_output_variance() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_sharding = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [3]).with_sharding(varying_sharding).unwrap();

        let infer = |output_variance| {
            ParallelAllGatherOperation::new("x".to_string(), 2, 0, CollectiveOptions::default(), output_variance)
                .with_mesh(mesh.clone())
                .infer_array_ir_output_types(&[
                    ArrayIrType::Array(input.clone()),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ])
        };
        let varying = infer(ParallelAllGatherOutputVariance::Varying).unwrap();
        let varying = <&ArrayType>::try_from(&varying[0]).unwrap();
        assert_eq!(varying.sharding().unwrap().varying_manual_axes(), &["x".to_string()].into_iter().collect());
        assert!(varying.sharding().unwrap().reduced_axes().is_empty());

        let invariant = infer(ParallelAllGatherOutputVariance::Invariant).unwrap();
        let invariant = <&ArrayType>::try_from(&invariant[0]).unwrap();
        assert!(invariant.sharding().unwrap().varying_manual_axes().is_empty());
        assert!(invariant.sharding().unwrap().reduced_axes().is_empty());

        let reduced = infer(ParallelAllGatherOutputVariance::Reduced).unwrap();
        let reduced = <&ArrayType>::try_from(&reduced[0]).unwrap();
        assert!(reduced.sharding().unwrap().varying_manual_axes().is_empty());
        assert_eq!(reduced.sharding().unwrap().reduced_axes(), &["x".to_string()].into_iter().collect());

        // The cotangent of a reduced gather result is unreduced. Sum-scatter consumes exactly that marker and restores
        // the varying input-cotangent state without a second reduce-scatter operation type.
        let reduced_cotangent = reduced.cotangent().unwrap();
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::default())
                .with_mesh(mesh)
                .infer_parent_output_types(
                    &[reduced_cotangent.into(), DimensionValue::constant(3).unwrap().r#type().into_owned().into(),],
                    &[],
                ),
            Ok(vec![input.cotangent().unwrap().into()]),
        );
        // Reduced output variance requires a manual mesh axis.
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Reduced,
            )
            .infer_output_types(&[ArrayType::new_static(DataType::F32, [2])], &[]),
            Err(TypeError::invalid("`parallel_all_gather` with reduced output variance requires a manual mesh axis")),
        );
    }

    #[test]
    fn test_parallel_all_gather_type_inference_array_ir_untiled() {
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                4,
                1,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .infer_array_ir_output_types(&[
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])).into(),
                DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
            ],),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]),
                )
                .into()
            ],),
        );
    }

    #[test]
    fn test_parallel_all_gather_type_inference_array_ir_dynamic() {
        let input_axis = DimensionVariable::new("input", DimensionBounds::new(1, Some(17)).unwrap());
        let concat_result = DimensionVariable::new("concat", DimensionBounds::new(2, Some(33)).unwrap());
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(input_axis.clone()), Dimension::Static(3)]),
        );

        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .infer_array_ir_output_types(&[
                input_type.clone().into(),
                ArrayIrType::Dimension(DimensionType::from(concat_result.clone())),
                DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
            ],),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(concat_result.clone()), Dimension::Static(3)]),
                )
                .into()
            ],),
        );
        let exact_six = DimensionValue::constant(6).unwrap().r#type().into_owned();
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .infer_array_ir_output_types(&[ArrayType::new_static(DataType::F32, [3]).into(), exact_six.into()]),
            Ok(vec![ArrayType::new_static(DataType::F32, [6]).into()]),
        );
        let exact_five = DimensionValue::constant(5).unwrap().r#type().into_owned();
        assert_eq!(
            ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                ).infer_array_ir_output_types(
                &[ArrayType::new_static(DataType::F32, [3]).into(), exact_five.into()],
            ),
            Err(TypeError::invalid(
                "`parallel_all_gather` result extent must equal input axis 0 extent 3 multiplied by axis group size 2; \
                 expected 6 \
                 but got 5"
                    .to_string(),
            ),),
        );
    }

    #[test]
    fn test_parallel_all_gather_type_inference_array_ir_metadata() {
        let operation = ParallelAllGatherOperation::new(
            "participants".to_string(),
            1,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        );
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let dimension = DimensionVariable::new("input", DimensionBounds::new(0, Some(9)).unwrap());
        let input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(dimension.clone())]))
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["devices"])]).unwrap())
            .unwrap()
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_memory(Memory::Host { pinned: true });

        // Check the final extent, even when the input dimension is symbolic and cannot establish exact geometry.
        assert_eq!(
            operation.infer_array_ir_output_types(&[
                input.clone().into(),
                DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
            ],),
            Err(TypeError::invalid(
                "`parallel_all_gather` on a dimension sharded over explicit mesh axes requires the output size (3) \
                 at axis 0 to be divisible by the mesh-axis product (2)",
            ),),
        );
        let outputs = operation
            .infer_array_ir_output_types(&[
                input.clone().into(),
                DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
            ])
            .unwrap();
        let output = <&ArrayType>::try_from(&outputs[0]).unwrap();
        assert_eq!(output.sharding(), input.sharding());
        assert_eq!(output.memory(), input.memory());
        assert!(output.layout().is_none());

        // An unchanged symbolic shape retains layout just like the homogeneous identity case.
        assert_eq!(
            operation.infer_array_ir_output_types(&[input.clone().into(), DimensionType::from(dimension).into()]),
            Ok(vec![input.into()]),
        );
        let input = ArrayType::new_static(DataType::F32, [4])
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            operation.infer_array_ir_output_types(&[
                input.clone().into(),
                DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
            ],),
            Ok(vec![input.into()]),
        );
    }

    #[test]
    fn test_parallel_all_gather_type_inference_array_ir_group_size() {
        let grouped = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        let result_extent = DimensionValue::constant(6).unwrap().r#type().into_owned();
        assert_eq!(
            ParallelAllGatherOperation::new("x".to_string(), 4, 0, grouped, ParallelAllGatherOutputVariance::Varying)
                .infer_array_ir_output_types(&[ArrayType::new_static(DataType::F32, [3]).into(), result_extent.into()]),
            Ok(vec![ArrayType::new_static(DataType::F32, [6]).into()]),
        );
    }

    #[test]
    fn test_parallel_all_gather_type_inference_array_ir_unchanged_extents() {
        let exact_two = DimensionValue::constant(2).unwrap().r#type().into_owned();
        let exact_three = DimensionValue::constant(3).unwrap().r#type().into_owned();
        let exact_four = DimensionValue::constant(4).unwrap().r#type().into_owned();
        let exact_five = DimensionValue::constant(5).unwrap().r#type().into_owned();
        // Inserting an axis preserves the extents already projected into the base output type on either side.
        let gather = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            1,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        );
        assert_eq!(
            gather.infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [3, 4]).into(),
                exact_three.clone().into(),
                exact_two.clone().into(),
                exact_four.clone().into(),
            ],),
            Ok(vec![ArrayType::new_static(DataType::F32, [3, 2, 4]).into()]),
        );
        assert_eq!(
            gather.infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [3, 4]).into(),
                exact_three.clone().into(),
                exact_two.clone().into(),
                exact_five.into(),
            ],),
            Err(TypeError::invalid("`parallel_all_gather` output axis 2 extent 5 must equal unchanged extent 4")),
        );
    }

    #[test]
    fn test_parallel_all_gather_type_inference_preserves_unrelated_pending_sums() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(
                Sharding::replicated(mesh.clone(), 1)
                    .with_varying_manual_axes(["x"])
                    .unwrap()
                    .with_unreduced_axes(["y"])
                    .unwrap(),
            )
            .unwrap();
        let expected = input.clone().with_shape(Shape::new(vec![Dimension::Static(8)]));
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        )
        .with_mesh(mesh.clone());
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input), &[]), Ok(vec![expected.clone()]));
        assert_eq!(
            operation.infer_array_ir_output_types(&[
                input.clone().into(),
                DimensionValue::constant(8).unwrap().r#type().into_owned().into(),
            ],),
            Ok(vec![expected.clone().into()]),
        );

        // Normalizing an invariant input over x must retain the independent pending sum over y.
        let invariant = input
            .clone()
            .with_sharding(input.sharding().unwrap().clone().with_varying_manual_axes(Vec::<String>::new()).unwrap())
            .unwrap();
        let (output, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_all_gather_tiled("x", 0),
            ArrayIrType::Array(invariant),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(output, ArrayIrType::Array(expected));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], unreduced={'y'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], unreduced={'y'}, \
                varying_manual={'x'}}] = parallel_vary [axis_name=\"x\"] %0
                    %2:dimension<4> = constant [value=4]
                    %3:dimension<2> = constant [value=2]
                    %4:dimension<8> = dimension_mul %2 %3
                    %5:f32[8][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], unreduced={'y'}, \
                varying_manual={'x'}}] = parallel_all_gather [
                        axis_name=\"x\",
                        axis_size=2,
                        concat_axis=0,
                        options=Tiled,
                        output_variance=Varying,
                        mesh=['x'=2:manual, 'y'=2:manual],
                    ] %1 %4
                in (%5)"
            },
        );

        // Gathering over a pending sum on the participating axis remains invalid.
        let pending = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            operation.infer_output_types(&[pending], &[]),
            Err(TypeError::invalid("`parallel_all_gather` does not support unreduced inputs")),
        );
    }

    #[test]
    fn test_parallel_all_gather_interpretation() {
        // A single-participant axis is degenerate: the gather concatenates exactly one input, so interpretation is the
        // identity.
        let outputs = ParallelAllGatherOperation::new(
            "x".to_string(),
            1,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        )
        .interpret(
            &EagerContext::<Array, ArrayOperation<Array>>::new(),
            &EmptyRegionDriver,
            &[Array::vector(vec![1.0, 2.0]).unwrap()],
        )
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].elements::<f64>().unwrap(), vec![1.0, 2.0]);

        // An untiled gather over a single participant inserts its size-one gathered axis.
        let outputs = ParallelAllGatherOperation::new(
            "x".to_string(),
            1,
            0,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        )
        .interpret(
            &EagerContext::<Array, ArrayOperation<Array>>::new(),
            &EmptyRegionDriver,
            &[Array::vector(vec![1.0, 2.0]).unwrap()],
        )
        .unwrap();
        assert_eq!(outputs, vec![Array::matrix(1, 2, vec![1.0, 2.0]).unwrap()]);

        // Any larger axis has no per-item semantics: the other participants do not exist outside an enclosing binder.
        let error = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        )
        .interpret(
            &EagerContext::<Array, ArrayOperation<Array>>::new(),
            &EmptyRegionDriver,
            &[Array::vector(vec![1.0, 2.0]).unwrap()],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ProgramError::UnsupportedOperation { message }
                if message == "cannot interpret `parallel_all_gather` over axis `x` of size 2 without an enclosing \
                    binder",
        ),);
    }

    #[test]
    fn test_parallel_all_gather_interpretation_array_ir() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());

        assert_eq!(
            context.bind(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context
                .bind(
                    ParallelAllGatherOperation::new(
                        "x".to_string(),
                        1,
                        0,
                        CollectiveOptions::tiled(),
                        ParallelAllGatherOutputVariance::Varying,
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()),],
                )
                .unwrap_err(),
            ProgramError::InvalidArgument {
                message: "`parallel_all_gather` output axis 0 extent must equal observed result extent 3 but got 4"
                    .to_string()
            },
        );
        assert_eq!(
            context
                .bind(
                    ParallelAllGatherOperation::new(
                        "x".to_string(),
                        2,
                        0,
                        CollectiveOptions::tiled(),
                        ParallelAllGatherOutputVariance::Varying,
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(6).unwrap()),],
                )
                .unwrap_err(),
            ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_all_gather` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            },
        );
    }

    #[test]
    fn test_parallel_all_gather_partial_evaluation() {
        let input = Array::vector(vec![1f32, 2.0]).unwrap();
        check_operation_partial_evaluation!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                1,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            ),
            inputs = [input.clone()],
            expected = input,
        );
    }

    #[test]
    fn test_parallel_all_gather_partial_evaluation_array_ir() {
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());

        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                1,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            ),
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
    }

    #[test]
    fn test_parallel_all_gather_batching() {
        // The batch binds the axis `"x"` that the `parallel_all_gather` names, so the matching batching rule consumes
        // the mapped axis: every item receives the item-major concatenation of all items along `concat_axis`,
        // replicated across the batch. With items `[1, 2]` and `[3, 4]` the gathered value is `[1, 2, 3, 4]`, matching
        // the verified cross-device `shard_map` execution semantics of the tiled StableHLO `all_gather`.
        let output: ArrayIrValue<Array> = batch(
            |item: BatchingTracer<
                EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                ArrayIrBatchingPolicy,
            >| { item.parallel_all_gather_tiled("x", 0) },
            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)]))),
        );
        let ArrayIrValue::Array(output) = output else {
            panic!("`parallel_all_gather` must preserve the array member kind");
        };
        assert_eq!(output.elements::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_parallel_all_gather_batching_untiled() {
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let input =
            ArrayBatch::new(Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap();
        let gathered = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            1,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        )
        .batch(&context, &EmptyRegionDriver, &[input])
        .unwrap()
        .into_parts()
        .0;
        assert_eq!(gathered[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(gathered[0].value(), &Array::matrix(2, 2, vec![1.0f32, 3.0, 2.0, 4.0]).unwrap());
    }

    #[test]
    fn test_parallel_all_gather_batching_replicated() {
        // A replicated input at a matching level is first materialized as `axis_size` identical batch items, so the
        // gather degenerates to the item-major concatenation of that many copies of the shared value.
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let outputs = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        )
        .batch(&context, &EmptyRegionDriver, &[ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]).unwrap())])
        .unwrap()
        .into_parts()
        .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().elements::<f64>().unwrap(), vec![1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_parallel_all_gather_batching_tiled_ragged() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![1.0f32; 6]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1i32, 3]).unwrap(), variable, vec![0])])
            .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());

        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .batch(&context, &EmptyRegionDriver, &[input]),
            Err(BatchingError::UnsupportedOperation {
                message: "tiled `parallel_all_gather` cannot represent participant-specific bounded ragged extents \
                          after the participant and concatenation axes are fused"
                    .to_string(),
            },),
        );
    }

    #[test]
    fn test_parallel_all_gather_batching_untiled_ragged() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input =
            ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 0.0, 0.0, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(
                    1,
                    Array::vector(vec![1i32, 3]).unwrap(),
                    variable.clone(),
                    vec![0],
                )])
                .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        );

        let output = operation.batch(&context, &EmptyRegionDriver, &[input]).unwrap().into_parts().0.remove(0);

        assert_eq!(output.batch_axis(), BatchAxis::replicated());
        assert_eq!(output.value().elements::<f64>().unwrap(), vec![1.0, 0.0, 0.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            output.ragged_axes(),
            &[RaggedAxis::new(1, Array::vector(vec![1i32, 3]).unwrap(), variable, vec![0])],
        );
    }

    #[test]
    fn test_parallel_all_gather_batching_array_ir_ragged() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(8)).unwrap());
        let extents = ArrayIrValue::Array(Array::vector(vec![1i32, 3]).unwrap());
        let input = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 0.0, 0.0, 2.0, 3.0, 4.0]).unwrap()),
            BatchAxis::new(0),
        )
        .unwrap()
        .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
        .unwrap();
        let extent =
            |value| ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(value).unwrap()));
        let ragged_extent =
            ArrayIrBatch::mapped_dimension(extents.clone(), BatchAxis::new(0), DimensionType::from(variable.clone()))
                .unwrap();
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("x".to_string());
        let untiled = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        );
        let output = untiled
            .batch_in_parent(&context, &EmptyRegionDriver, &[input.clone(), extent(2), ragged_extent])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(
            output.value(),
            &ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 0.0, 0.0, 2.0, 3.0, 4.0]).unwrap()),
        );
        assert_eq!(output.ragged_axes(), &[RaggedAxis::new(1, extents, variable.clone(), vec![0])]);

        let tiled = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        );
        assert_eq!(
            tiled.batch_in_parent(&context, &EmptyRegionDriver, &[input, extent(6)]),
            Err(BatchingError::UnsupportedOperation {
                message: "tiled `parallel_all_gather` cannot represent participant-specific bounded ragged extents \
                          after the participant and concatenation axes are fused"
                    .to_string(),
            },),
        );
    }

    #[test]
    fn test_parallel_all_gather_batching_untiled_replicated_ragged() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input = ArrayBatch::replicated(Array::vector(vec![1.0f32, 2.0, 0.0]).unwrap())
            .with_ragged_axes(vec![RaggedAxis::new(0, Array::scalar(2i32).unwrap(), variable.clone(), Vec::new())])
            .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        );

        let output = operation.batch(&context, &EmptyRegionDriver, &[input]).unwrap().into_parts().0.remove(0);

        assert_eq!(output.batch_axis(), BatchAxis::replicated());
        assert_eq!(output.value().elements::<f32>().unwrap(), vec![1.0, 2.0, 0.0, 1.0, 2.0, 0.0]);
        assert_eq!(
            output.ragged_axes(),
            &[RaggedAxis::new(1, Array::vector(vec![2i32, 2]).unwrap(), variable, vec![0])],
        );
    }

    #[test]
    fn test_parallel_all_gather_batching_dynamic_extent_replicated() -> Result<(), ProgramError> {
        // Matching-axis collective batching consumes a complete logical result shape. A replicated input is
        // materialized along the mapped axis from those extents, dynamic unchanged axes keep their boundary-provided
        // identity, and the rule introduces no metadata read from the source array.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::from(batch).into());
        let sequence = DimensionVariable::new("sequence", DimensionBounds::new(1, Some(17))?);
        let width = DimensionVariable::new("width", DimensionBounds::new(1, Some(33))?);
        let gathered = DimensionVariable::new("gathered", DimensionBounds::new(1, Some(65))?);
        let input = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(sequence), Dimension::Dynamic(width.clone())]),
            )
            .into(),
        );
        let gathered_extent = trace.input(DimensionType::from(gathered).into());
        let width_extent = trace.input(DimensionType::from(width).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent)
            .with_axis_name("items".to_string());
        let [output] = context
            .bind(
                ArrayIrOperation::ParallelAllGather(ParallelAllGatherOperation::new(
                    "items".to_string(),
                    4,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                )),
                Vec::new(),
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(input)),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(gathered_extent)),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(width_extent)),
                ],
            )?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output.batch().value().atom_id()?],
            vec![Placeholder; 4],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[sequence, width], %2:dimension<gathered ∈ [1, 65)>, \
                    %3:dimension<width ∈ [1, 33)> .
                let %4:dimension<4> = constant [value=4]
                    %5:bool[] = compare [direction=Equal] %0 %4
                    () = assert [
                        message=\"collective axis extent must match the participant count\",
                        labels=[\"extent\", \"participants\"],
                    ] %5 %0 %4
                    %6:dimension<0> = constant [value=0]
                    %7:dimension<gathered % batch ∈ [0, 8)> = dimension_rem %2 %0
                    %8:bool[] = compare [direction=Equal] %7 %6
                    () = assert [
                        message=\"collective extent must be divisible by the participant count\",
                        labels=[\"extent\", \"divisor\"],
                    ] %8 %2 %0
                    %9:dimension<gathered / batch ∈ [0, 65)> = dimension_div %2 %0
                    %10:f32[gathered / batch, width] = reshape %1 %9 %3
                    %11:f32[batch, gathered / batch, width] = broadcast [output_axes=[1, 2]] %10 %0 %9 %3
                    %12:f32[gathered, width] = reshape %11 %2 %3
                in (%12)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_parallel_all_gather_batching_dynamic_extent_forwarding() -> Result<(), ProgramError> {
        // A collective over a different named axis is forwarded as the same mixed operation. Only its physical axis
        // index and complete result shape are lifted around the current mapped axis, without reading the source shape.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::from(batch.clone()).into());
        let logical_extent = DimensionVariable::new("logical", DimensionBounds::new(1, Some(17))?);
        let result_extent = DimensionVariable::new("result", DimensionBounds::new(1, Some(33))?);
        let input = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(logical_extent), Dimension::Dynamic(batch), Dimension::Static(3)]),
            )
            .into(),
        );
        let result_extent = trace.input(DimensionType::from(result_extent).into());
        let width_extent = trace.input(DimensionValue::constant(3)?.r#type().into_owned().into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent)
            .with_axis_name("outer".to_string());
        let [output] = context
            .bind(
                ArrayIrOperation::ParallelAllGather(ParallelAllGatherOperation::new(
                    "inner".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                )),
                Vec::new(),
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(input, BatchAxis::new(1))?),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(result_extent)),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(width_extent)),
                ],
            )?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(1));
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output.batch().value().atom_id()?],
            vec![Placeholder; 4],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[logical, batch, 3], %2:dimension<result ∈ [1, 33)>, \
                    %3:dimension<3> .
                let %4:f32[result, batch, 3] = parallel_all_gather [
                    axis_name=\"inner\",
                    axis_size=2,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ] %1 %2 %0 %3
                in (%4)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_parallel_all_gather_batching_array_ir_replicated_ragged() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input = ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 0.0]).unwrap()))
            .with_ragged_axes(vec![RaggedAxis::new(
                0,
                ArrayIrValue::Array(Array::scalar(2i32).unwrap()),
                variable.clone(),
                Vec::new(),
            )])
            .unwrap();
        let extent =
            |value| ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(value).unwrap()));
        let ragged_extent = ArrayIrBatch::replicated(ArrayIrValue::Dimension(
            DimensionValue::new(DimensionType::from(variable.clone()), 2).unwrap(),
        ));
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("x".to_string());
        let output = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        )
        .batch_in_parent(&context, &EmptyRegionDriver, &[input, extent(2), ragged_extent])
        .unwrap()
        .into_parts()
        .0
        .remove(0);

        assert_eq!(
            output.value(),
            &ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 0.0, 1.0, 2.0, 0.0]).unwrap()),
        );
        assert_eq!(
            output.ragged_axes(),
            &[RaggedAxis::new(1, ArrayIrValue::Array(Array::vector(vec![2i32, 2]).unwrap()), variable, vec![0])],
        );
    }

    #[test]
    fn test_parallel_all_gather_differentiation_member_rule() -> Result<(), ProgramError> {
        // A live tangent through a dynamically shaped mixed collective stages one residual-aware linear call directly
        // through the payload's member JVP rule.
        let variable = DimensionVariable::new("items", DimensionBounds::new(1, Some(9))?);
        let dimension_type = DimensionType::from(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let primal = context.input(array_type.clone().into());
        let tangent = context.input(array_type.into());
        let extent = context.input(dimension_type.into());
        let extent_tangent_type = extent.r#type().tangent()?;
        let outputs = ParallelAllGatherOperation::new(
            "x".to_string(),
            1,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        )
        .jvp_in_parent(
            &DifferentiationContext::fused(context.clone()),
            &EmptyRegionDriver,
            &[
                DifferentiationDual::new(primal, MaybeZero::Value(tangent))?,
                DifferentiationDual::new(extent, MaybeZero::Zero(extent_tangent_type))?,
            ],
        )?;
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(_)));
        let output_ids = vec![outputs[0].primal().atom_id()?, outputs[0].tangent().as_value().unwrap().atom_id()?];
        let program = context.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            output_ids,
            vec![Placeholder; 3],
            vec![Placeholder; 2],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[items], %1:f32[items], %2:dimension<items ∈ [1, 9)> .
                let %3:f32[items] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=1,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ] %0 %2
                    %4:f32[items] = linear_call [residual_count=1] %2 %1 [
                        forward={
                            lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items] .
                            let %2:f32[items] = parallel_all_gather [
                                axis_name=\"x\",
                                axis_size=1,
                                concat_axis=0,
                                options=Tiled,
                                output_variance=Varying,
                            ] %1 %0
                            in (%2)
                        },
                        transpose={
                            lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items] .
                            let %2:f32[items] = parallel_sum_scatter [axis_name=\"x\", axis_size=1, scatter_axis=0, \
                options=Tiled] %1 %0
                            in (%2)
                        },
                    ]
                in (%3, %4)"
            },
        );

        Ok(())
    }

    #[test]
    fn test_parallel_all_gather_differentiation_array_ir_jvp() {
        let variable = DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap());
        let dimension_type = DimensionType::from(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![array, result_extent],
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
        let primal = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let tangent = ArrayIrValue::Array(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap());
        let result_extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 3).unwrap());
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![primal.clone(), result_extent.clone(), tangent.clone(),]),
            Ok(vec![primal, tangent]),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:dimension<extent ∈ [0, 9)> .
                let %2:f32[extent] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=1,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ] %0 %1
                in (%2)"
            },
        );
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:dimension<extent ∈ [0, 9)>, %2:f32[extent] .
                let %3:f32[extent] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=1,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ] %0 %1
                    %4:f32[extent] = linear_call [residual_count=1] %1 %2 [
                        forward={
                            lambda %0:dimension<extent ∈ [0, 9)>, %1:f32[extent] .
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
                            lambda %0:dimension<extent ∈ [0, 9)>, %1:f32[extent] .
                            let %2:f32[extent] = parallel_sum_scatter [axis_name=\"x\", axis_size=1, scatter_axis=0, \
                options=Tiled] %1 %0
                            in (%2)
                        },
                    ]
                in (%3, %4)"
            },
        );
    }

    #[test]
    fn test_parallel_all_gather_differentiation_array_ir_invariant() {
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dimension_type = DimensionType::from(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Invariant,
                ),
                Vec::new(),
                vec![array, result_extent],
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
            program.to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:dimension<extent ∈ [1, 9)> .
                let %2:f32[extent] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=1,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Invariant,
                ] %0 %1
                in (%2)"
            },
        );
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:dimension<extent ∈ [1, 9)> .
                let %2:f32[extent] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=1,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Invariant,
                ] %0 %1
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
                        let %2:f32[extent] = parallel_all_gather [
                            axis_name=\"x\",
                            axis_size=1,
                            concat_axis=0,
                            options=Tiled,
                            output_variance=Invariant,
                        ] %1 %0
                        in (%2)
                    },
                    transpose={
                        lambda %0:dimension<extent ∈ [1, 9)>, %1:f32[extent] .
                        let %2:dimension<0> = constant [value=0]
                            %3:f32[extent] = dynamic_slice [strides=[1], bounds=checked, \
                requires_runtime_assertion=true] %1 %2 %0
                            %4:f32[extent] = reshape %3 %0
                        in (%4)
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
                        let %2:dimension<0> = constant [value=0]
                            %3:f32[extent] = dynamic_slice [strides=[1], bounds=checked, \
                requires_runtime_assertion=true] %1 %2 %0
                            %4:f32[extent] = reshape %3 %0
                        in (%4)
                    },
                    transpose={
                        lambda %0:dimension<extent ∈ [1, 9)>, %1:f32[extent] .
                        let %2:f32[extent] = parallel_all_gather [
                            axis_name=\"x\",
                            axis_size=1,
                            concat_axis=0,
                            options=Tiled,
                            output_variance=Invariant,
                        ] %1 %0
                        in (%2)
                    },
                ]
                in (%2)"
            },
        );
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap());
        let mut pullback_inputs = vec![cotangent];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap())]),
        );

        // A nondegenerate untiled invariant gather selects the current participant's size-one slice and reshapes
        // away the ranked participant axis.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let participant_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let input_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::default(),
                    ParallelAllGatherOutputVariance::Invariant,
                ),
                Vec::new(),
                vec![array, participant_extent, input_extent],
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
        // The mixed boundary delegates its array contribution to the homogeneous all-gather rule, so the invariant
        // guard that rule owns is what rejects direct transposition here.
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "direct transposition of invariant `parallel_all_gather` cannot represent the \
                    participant-indexed \
                    slice; linearize so that the current participant can select its gathered chunk",
        ),);
        let pullback = program.linearize().unwrap().pullback().unwrap().to_string();
        assert_eq!(
            pullback,
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:dimension<2> = const 2
                    %2:dimension<3> = const 3
                    %3:f32[3] = linear_call [residual_count=2] %1 %2 %0 [
                        forward={
                            lambda %0:dimension<2>, %1:dimension<3>, %2:f32[2, 3] .
                            let %3:u64[] = axis_index [axis_name=\"x\"]
                                %4:dimension<x_index ∈ [0, 2)> = dimension_from_scalar [bounds=[0, 2)] %3
                                %5:dimension<0> = constant [value=0]
                                %6:dimension<1> = constant [value=1]
                                %7:dimension<3> = constant [value=3]
                                %8:f32[1, 3] = dynamic_slice [strides=[1, 1], bounds=checked, \
                requires_runtime_assertion=true] %2 %4 %5 %6 %7
                                %9:f32[3] = reshape %8 %7
                            in (%9)
                        },
                        transpose={
                            lambda %0:dimension<2>, %1:dimension<3>, %2:f32[3] .
                            let %3:f32[2, 3] = parallel_all_gather [
                                axis_name=\"x\",
                                axis_size=2,
                                concat_axis=0,
                                options=Untiled,
                                output_variance=Invariant,
                            ] %2 %0 %1
                            in (%3)
                        },
                    ]
                in (%3)"
            },
        );
    }

    #[test]
    fn test_parallel_all_gather_differentiation_array_ir_dynamic_extent() {
        let variable = DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap());
        let dimension_type = DimensionType::from(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![array, result_extent],
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
        let result_extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 3).unwrap());
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:dimension<extent ∈ [0, 9)> .
                let %2:f32[extent] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=1,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ] %0 %1
                in (%2)"
            },
        );
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:dimension<extent ∈ [0, 9)> .
                let %2:f32[extent] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=1,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ] %0 %1
                in (%2, %1)"
            },
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:dimension<extent ∈ [0, 9)> .
                let %2:f32[extent] = linear_call [residual_count=1] %1 %0 [
                    forward={
                        lambda %0:dimension<extent ∈ [0, 9)>, %1:f32[extent] .
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
                        lambda %0:dimension<extent ∈ [0, 9)>, %1:f32[extent] .
                        let %2:f32[extent] = parallel_sum_scatter [axis_name=\"x\", axis_size=1, scatter_axis=0, \
                options=Tiled] %1 %0
                        in (%2)
                    },
                ]
                in (%2)"
            },
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:dimension<extent ∈ [0, 9)> .
                let %2:f32[extent] = linear_call [residual_count=1] %1 %0 [
                    forward={
                        lambda %0:dimension<extent ∈ [0, 9)>, %1:f32[extent] .
                        let %2:f32[extent] = parallel_sum_scatter [axis_name=\"x\", axis_size=1, scatter_axis=0, \
                options=Tiled] %1 %0
                        in (%2)
                    },
                    transpose={
                        lambda %0:dimension<extent ∈ [0, 9)>, %1:f32[extent] .
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

        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()), result_extent])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap());
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
        let zero_extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 0).unwrap());
        let zero_array = ArrayIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![zero_array.clone(), zero_extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let zero_cotangent = zero_array.clone();
        let mut pullback_inputs = vec![zero_cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![zero_cotangent]));
    }

    #[test]
    fn test_parallel_all_gather_differentiation_array_ir_invariant_manual_mesh() {
        // Inside a manual region, the output cotangent of an invariant gather is invariant across the gathered axis,
        // while every participant selects a different chunk of it. The pullback therefore varies the cotangent over
        // that axis with a real `parallel_vary` transition before slicing it at the checked participant index, so that
        // the selected chunk has the input cotangent's variation.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| {
                input.parallel_all_gather_with_options(
                    "x",
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Invariant,
                )
            },
            ArrayIrType::Array(input_type),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        let pullback = program.to_flat_program().linearize().unwrap().pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{}]}], %1:dimension<4> .
                let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = linear_call \
                    [residual_count=1] %1 %0 [
                    forward={
                        lambda %0:dimension<4>, %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}]}] .
                        let %2:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = axis_index \
                            [axis_name=\"x\", mesh=['x'=2:manual]]
                            %3:dimension<x_index ∈ [0, 2)> = dimension_from_scalar [bounds=[0, 2)] %2
                            %4:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary \
                                [axis_name=\"x\"] %1
                            %5:dimension<2> = constant [value=2]
                            %6:dimension<x_index * 2 ∈ [0, 3)> = dimension_mul %3 %5
                            %7:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = dynamic_slice \
                                [strides=[1], bounds=checked, requires_runtime_assertion=true] %4 %6 %5
                            %8:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reshape \
                                [output_sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] %7 %5
                        in (%8)
                    },
                    transpose={
                        lambda %0:dimension<4>, %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %2:f32[4][sharding={mesh<['x'=2:manual]>, [{}]}] = parallel_all_gather [
                            axis_name=\"x\",
                            axis_size=2,
                            concat_axis=0,
                            options=Tiled,
                            output_variance=Invariant,
                            mesh=['x'=2:manual],
                        ] %1 %0
                        in (%2)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_parallel_all_gather_transposition() {
        // A tiled all-gather is the adjoint of a sum-scatter over the same axis and dimension, so the pullback stages
        // a `parallel_sum_scatter` on the output cotangent with the gather's concat axis as its scatter axis.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [2]));
        let output = builder
            .add_instruction(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {r#"
                lambda %0:f32[4] .
                let %1:f32[2] = parallel_sum_scatter [axis_name="x", axis_size=2, scatter_axis=0, options=Tiled] %0
                in (%1)
            "#}
            .trim_end(),
        );

        let groups = vec![vec![0, 2], vec![3, 1]];
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [2]));
        let output = builder
            .add_instruction(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    4,
                    0,
                    CollectiveOptions::tiled().with_axis_index_groups(groups.clone()),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[2] = parallel_sum_scatter [
                    axis_name=\"x\",
                    axis_size=4,
                    scatter_axis=0,
                    options=CollectiveOptions { mode: Tiled, axis_index_groups: [[0, 2], [3, 1]] },
                ] %0
                in (%1)"
            },
        );

        // Reduced output variance swaps to an unreduced cotangent type. The same sum-scatter operation, over the same
        // mesh, consumes that state and returns the original varying input cotangent.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Reduced,
                )
                .with_mesh(mesh.clone()),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{}], unreduced={'x'}}] .
                let %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_sum_scatter \
                [axis_name=\"x\", axis_size=2, scatter_axis=0, options=Tiled, mesh=['x'=2:manual]] %0
                in (%1)"
            },
        );
        assert_eq!(pullback.output_types(), vec![input_type.cotangent().unwrap()]);
    }

    #[test]
    fn test_parallel_all_gather_transposition_array_ir_dynamic_extent() {
        let variable = DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap());
        let dimension_type = DimensionType::from(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![array, result_extent],
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
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "direct `parallel_all_gather` transposition with runtime-dependent type metadata \
                    requires \
                    linearization so that the relevant primal information can be retained as residuals",
        ),);
    }

    #[test]
    fn test_parallel_all_gather_parallel_all_gather() {
        // Homogeneous array values stage the static-shape operation without explicit result extents, making an
        // invariant input varying first, and gather over a `batch` level so that every batch item receives all items.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_all_gather_tiled("x", 0),
            invariant_type,
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3][sharding={mesh<['x'=2:manual]>, [{}]}] .
                let %1:f32[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_vary [axis_name=\"x\"] %0
                    %2:f32[6][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_all_gather [
                        axis_name=\"x\",
                        axis_size=2,
                        concat_axis=0,
                        options=Tiled,
                        output_variance=Varying,
                        mesh=['x'=2:manual],
                    ] %1
                in (%2)"
            },
        );
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    item.parallel_all_gather("x", 0)
                },
                Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::from_elements(
                ArrayType::new_static(DataType::F64, [2, 2, 2]),
                &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            )
            .unwrap(),),
        );

        // Concrete arrays, and concrete composite values through their array members, are never inside an axis
        // binder, so every axis name is unbound for them.
        assert_eq!(
            Array::vector(vec![1.0, 2.0]).unwrap().parallel_all_gather("x", 0),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
        assert_eq!(
            ArrayIrValue::Array(Array::vector(vec![1.0, 2.0]).unwrap()).parallel_all_gather("x", 0),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
    }

    #[test]
    fn test_parallel_all_gather_parallel_all_gather_unbound_axis() {
        // The batch binds only the axis `"i"`, but the `parallel_all_gather` names `"x"`, which no enclosing transform
        // binds. Axis-size resolution fails fast at staging time with `AxisError::UnboundAxisName` rather than silently
        // acting as identity.
        let result: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |item: BatchingTracer<
                EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                ArrayIrBatchingPolicy,
            >| { item.parallel_all_gather_tiled("x", 0) },
            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("i"),
        );
        assert_eq!(result.unwrap_err(), BatchingError::Axis(AxisError::UnboundAxisName { name: "x".to_string() }));
    }

    #[test]
    fn test_parallel_all_gather_parallel_all_gather_tracing_import() {
        let bounds = DimensionBounds::new(1, Some(5)).unwrap();
        let input_variable = DimensionVariable::new("items", bounds);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["devices"]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(input_variable.clone())]))
            .with_sharding(sharding.clone())
            .unwrap();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_all_gather_tiled("devices", 0),
            ArrayIrType::Array(input_type),
            vec![("devices".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2 })],
        )
        .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[items][sharding={mesh<['devices'=2:manual]>, [{}], varying_manual={'devices'}}] .
                let %1:dimension<items ∈ [1, 5)> = dimension_size [axis=0] %0
                    %2:dimension<2> = constant [value=2]
                    %3:dimension<items * 2 ∈ [2, 9)> = dimension_mul %1 %2
                    %4:f32[items * 2][sharding={mesh<['devices'=2:manual]>, [{}], varying_manual={'devices'}}] = \
                parallel_all_gather [
                        axis_name=\"devices\",
                        axis_size=2,
                        concat_axis=0,
                        options=Tiled,
                        output_variance=Varying,
                        mesh=['devices'=2:manual],
                    ] %0 %3
                in (%4)"
            },
        );

        let target_variable = DimensionVariable::new("target", bounds);
        let target_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target_variable)]))
            .with_sharding(sharding)
            .unwrap();
        let instantiated = program
            .with_instantiated_type_identities(&[ArrayIrType::Array(target_type.clone())])
            .unwrap()
            .into_owned();
        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let imported_input = destination.add_input(target_type.into());
        let imported_outputs = destination.splice_program(&instantiated, &[imported_input]).unwrap();
        let imported = destination
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                imported_outputs,
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            imported.to_string(),
            indoc! {"
                lambda %0:f32[target][sharding={mesh<['devices'=2:manual]>, [{}], varying_manual={'devices'}}] .
                let %1:dimension<target ∈ [1, 5)> = dimension_size [axis=0] %0
                    %2:dimension<2> = constant [value=2]
                    %3:dimension<items * 2 ∈ [2, 9)> = dimension_mul %1 %2
                    %4:f32[items * 2][sharding={mesh<['devices'=2:manual]>, [{}], varying_manual={'devices'}}] = \
                parallel_all_gather [
                        axis_name=\"devices\",
                        axis_size=2,
                        concat_axis=0,
                        options=Tiled,
                        output_variance=Varying,
                        mesh=['devices'=2:manual],
                    ] %0 %3
                in (%4)"
            },
        );
    }
}
