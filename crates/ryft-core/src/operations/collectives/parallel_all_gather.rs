use std::fmt::Display;

use ryft_macros::capability;

use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrContext, ArrayIrType,
    ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds, DimensionOperation, DimensionType, DimensionValue,
    DimensionVariable, LinearResiduals, LogicalMesh, RaggedAxis, Shape, Sharding,
};
use crate::axes::{Axis, AxisError, NamedAxes, NamedAxis};
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
use crate::operations::Capability;
use crate::operations::arithmetic::{AddOperation, Div, Mul, Rem};
use crate::operations::assertions::Assert;
use crate::operations::collectives::axis_index::AxisIndexOperation;
use crate::operations::collectives::parallel_sum_scatter::ParallelSumScatterOperation;
use crate::operations::collectives::parallel_vary::{
    PARALLEL_VARY_OPERATION_NAME, ParallelVary, ParallelVaryOperation,
};
use crate::operations::collectives::{
    CollectiveArrayExtentBatchingPolicy, CollectiveMode, CollectiveOptions, LinearCollectiveOperation,
    ShapeChangingCollectiveBatching, ShapeChangingCollectiveOperation, ShapeChangingCollectiveValue,
    infer_array_ir_shape_changing_collective_output_type, infer_linear_collective_operation_output_type,
    resolve_named_axis_size, validate_manual_mesh_input,
};
use crate::operations::comparisons::Compare;
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::constants::zero::ZeroOperation;
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
use crate::operations::dimensions::dimension_max::DimensionMax;
use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{BroadcastOperation, DynamicBroadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape, ReshapeOperation};
use crate::operations::manipulation::slicing::DynamicSliceOperation;
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, MemberOperation, Operation, OperationFormatter, OperationProjection, ProgramError, ProjectedValue,
    RegionInterface, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Named axis variance carried by the result of a [`ParallelAllGatherOperation`], which corresponds to the `to`
/// argument of JAX's [`jax.lax.all_gather`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_gather.html)
/// (i.e., `"varying"`, `"invarying"`, and `"reduced"`).
///
/// This is an operation option rather than parallel type metadata. Type inference maps it onto the canonical
/// [`Sharding::varying_manual_axes`] and [`Sharding::reduced_axes`] sets of a result over a manual mesh axis.
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

/// [`Operation`] that gathers every participant's input across the named axis, so that
/// every participant receives all inputs in participant order. This is the Ryft analogue of JAX's
/// [`jax.lax.all_gather`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_gather.html) and of StableHLO's
/// [`all_gather`](https://openxla.org/stablehlo/spec#all_gather). The [`CollectiveMode`] of its options selects the
/// output shape over a group of `n` participants:
///
///   - [`CollectiveMode::Untiled`] inserts a new axis of extent `n` at `concatenation_axis`, so that participant `i`'s
///     input becomes row `i` of that axis.
///   - [`CollectiveMode::Tiled`] multiplies the extent of the existing `concatenation_axis` by `n`, so that participant
///     `i`'s input becomes the `i`-th contiguous chunk of that axis.
///
/// All other dimensions are unchanged. Participant groups (refer to [`CollectiveOptions`]) restrict the gather to each
/// group, whose listed order determines the chunk order, and are supported only with varying output variance. The
/// collective is linear. A varying result transposes to a [`ParallelSumScatterOperation`] with the same mode, axis,
/// participant groups, and mesh, and so does a reduced result, whose unreduced cotangent carries the pending sum that
/// the sum-scatter completes. An invariant result instead transposes by selecting the current participant's chunk of
/// the output cotangent at its [`AxisIndexOperation`] index, which needs no communication.
///
/// An all-gather over a manual mesh axis is created by [`with_mesh`](Self::with_mesh), and
/// [`ParallelAllGather::parallel_all_gather_with_options`] supplies the mesh automatically from the enclosing manual
/// region. Its input must vary over the axis (refer to [`ParallelVary`]) and must not carry reduction state (i.e., be
/// unreduced or reduced) over it, its [`ParallelAllGatherOutputVariance`] selects the manual variation of the result,
/// and reduction state over unrelated mesh axes is preserved. The transpose of an invariant result first makes its
/// invariant output cotangent varying over the axis, because every participant selects a different chunk of it. An
/// ordinary all-gather carries no mesh and preserves the mesh state of its input, even when that input carries a manual
/// mesh axis with the same name, because a `batch` level whose axis name shadows that mesh axis may bind it instead.
/// Its output variance therefore selects only its transpose, and it rejects reduced output variance, which records
/// state on a manual mesh axis. Outside any binder, the single participant of a degenerate ordinary all-gather keeps
/// its value, with a size-one axis inserted in untiled mode, while a manual mesh all-gather always requires its binder,
/// because a local reshape cannot perform its variance transition.
///
/// A matching `batch` level consumes the mapped batch axis of an ordinary all-gather by moving it to
/// `concatenation_axis` and, in tiled mode, merging it into the existing axis in batch item order, so that every batch
/// item receives the same gathered value and the output is replicated. Untiled gathering co-moves bounded ragged
/// metadata with the gathered value: the participant axis becomes an ordinary output axis that is added to the
/// `extent_axes` of each [`RaggedAxis`] whose extents vary across the batch items. Tiled gathering of a bounded ragged
/// input is rejected, because fusing the participant and concatenation axes can make the live elements of a chunk
/// non-prefix-shaped, which one [`RaggedAxis`] cannot represent. A matching level also rejects participant groups and
/// all-gathers over a manual mesh axis, and every other `batch` level rejects bounded ragged inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelAllGatherOperation {
    /// Axis name referenced by this collective.
    axis_name: String,

    /// Number of participants along the named axis, resolved from the active [`NamedAxes`] environment when the
    /// operation is staged.
    axis_size: usize,

    /// Axis at which participants are stacked in untiled mode, or along which inputs are concatenated in tiled mode.
    concatenation_axis: usize,

    /// [`CollectiveOptions`] of this [`ParallelAllGatherOperation`].
    options: CollectiveOptions,

    /// Named axis variance of the result.
    output_variance: ParallelAllGatherOutputVariance,

    /// Refer to the documentation of [`mesh`](Self::mesh) for more information.
    mesh: Option<LogicalMesh>,
}

impl ParallelAllGatherOperation {
    /// Creates a new [`ParallelAllGatherOperation`] over the axis with the provided name and resolved axis size.
    /// Construction preserves the supplied options while type inference validates the geometry, groups, and mesh state.
    #[inline]
    pub fn new(
        axis_name: String,
        axis_size: usize,
        concatenation_axis: usize,
        options: CollectiveOptions,
        output_variance: ParallelAllGatherOutputVariance,
    ) -> Self {
        Self { axis_name, axis_size, concatenation_axis, options, output_variance, mesh: None }
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

    /// Returns the stacking position in untiled mode or the existing concatenation axis in tiled mode.
    #[inline]
    pub fn concatenation_axis(&self) -> usize {
        self.concatenation_axis
    }

    /// Returns the [`CollectiveOptions`] of this [`ParallelAllGatherOperation`].
    #[inline]
    pub fn options(&self) -> &CollectiveOptions {
        &self.options
    }

    /// Returns the named axis variance of the result.
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

    /// Rejects an input that carries reduction state (i.e., pending or completed sum) over the gathered manual mesh
    /// axis, because gathering across that sum would change its reduction semantics. Independent sums over other mesh
    /// axes commute with the gather and are preserved.
    fn validate_input_reduction_state(&self, input_type: &ArrayType) -> Result<(), TypeError> {
        let axis_name = &self.axis_name;
        if input_type.unreduced_axes().contains(axis_name) || input_type.reduced_axes().contains(axis_name) {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` input must not carry reduction state \
                 over manual axis `{axis_name}`",
            )));
        }
        Ok(())
    }

    /// Rejects a concatenation axis that is out of bounds for the output of an input with rank `input_rank`. An untiled
    /// gather inserts its participant axis, so its concatenation axis is a position in an output whose rank is one more
    /// than `input_rank`, while a tiled gather concatenates along an existing axis.
    fn validate_concatenation_axis(&self, input_rank: usize) -> Result<(), TypeError> {
        let output_rank = input_rank + usize::from(self.options.mode == CollectiveMode::Untiled);
        if self.concatenation_axis >= output_rank {
            return Err(TypeError::invalid(format!(
                "`{}` concatenation axis {} is out of bounds for output rank {}",
                PARALLEL_ALL_GATHER_OPERATION_NAME, self.concatenation_axis, output_rank,
            )));
        }
        Ok(())
    }

    /// Rejects local interpretation of a manual mesh all-gather, whose binder owns execution and output variance.
    /// A local reshape cannot change manual variance or reduction state, even for a single participant.
    fn validate_local_interpretation(&self) -> Result<(), ProgramError> {
        if self.mesh.is_some() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "cannot interpret `{}` over manual mesh axis `{}` without an enclosing binder",
                    PARALLEL_ALL_GATHER_OPERATION_NAME, self.axis_name,
                ),
            });
        }
        Ok(())
    }

    /// Applies an all-gather's named-axis variance transition to the canonical sharding metadata of the shape-only
    /// `output_type` shared by the static and array IR inference paths. An ordinary all-gather carries no mesh and
    /// preserves the mesh state of its input, even when its input carries a manual mesh axis with the same name,
    /// because a `batch` level whose axis name shadows that mesh axis may bind it instead. Over a manual mesh axis,
    /// the input must vary over the axis, and the output variance selects whether the result keeps varying over it,
    /// becomes invariant over it, or records it as reduced.
    fn finalize_output_type(&self, input_type: &ArrayType, output_type: ArrayType) -> Result<ArrayType, TypeError> {
        let axis_name = self.axis_name.as_str();
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

        // Reduction state is diagnosed before the variation check below, whose diagnostic suggests a `parallel_vary`
        // transition that would also reject it.
        self.validate_input_reduction_state(input_type)?;
        let input_sharding = input_type.sharding().unwrap();

        // Gathering an input that is still invariant over the axis would concatenate identical copies under a type that
        // cannot tell them apart from per-participant values, and its transpose would produce a varying cotangent.
        if !input_sharding.varying_manual_axes().contains(axis_name) {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` input must vary over manual axis `{axis_name}`; pass an \
                 invariant value through `{PARALLEL_VARY_OPERATION_NAME}` first so that every copy is gathered",
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
        let output_sharding = output_type
            .sharding()
            .unwrap()
            .clone()
            .with_varying_manual_axes(varying_axes)?
            .with_reduced_axes(reduced_axes)?;
        Ok(output_type.with_sharding(output_sharding)?)
    }

    /// Relocates the bounded ragged metadata of the input of an untiled all-gather whose named axis the `batch` level
    /// of `context` binds. A mapped input's per-item extents follow its batch axis to the inserted participant axis,
    /// while a replicated input's scalar extents are broadcast along that axis, which becomes their only extent axis.
    fn gathered_ragged_axes<C: Context<Type = ArrayType>, P: CollectiveArrayExtentBatchingPolicy<C>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        ragged_axes: Vec<RaggedAxis<C::Value>>,
        input_batch_axis: Option<usize>,
        input_rank: usize,
    ) -> Result<Vec<RaggedAxis<C::Value>>, BatchingError> {
        if let Some(input_batch_axis) = input_batch_axis {
            return Ok(ragged_axes
                .into_iter()
                .map(|ragged_axis| ragged_axis.moved(input_batch_axis, 0).moved(0, self.concatenation_axis))
                .collect());
        }

        let output_axes = (1..=input_rank).collect::<Vec<_>>();
        ragged_axes
            .into_iter()
            .map(|ragged_axis| {
                if !ragged_axis.extent_axes().is_empty() {
                    return Err(BatchingError::UnsupportedOperation {
                        message: format!(
                            "untiled `{PARALLEL_ALL_GATHER_OPERATION_NAME}` requires replicated ragged inputs to carry \
                             scalar extents",
                        ),
                    });
                }
                let extents = P::match_axis(context, &ArrayBatch::replicated(ragged_axis.extents().clone()), 0.into())?
                    .into_value();
                let ragged_axis = ragged_axis.broadcasted(output_axes.as_slice()).moved(0, self.concatenation_axis);
                Ok(RaggedAxis::new(
                    ragged_axis.axis(),
                    extents,
                    ragged_axis.dimension().clone(),
                    vec![self.concatenation_axis],
                ))
            })
            .collect()
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
        let input_type = self.validate_input(input_types, region_interfaces)?;
        let effective_axis_size = self.effective_axis_size()?;

        // Result-shape arithmetic in the homogeneous array family requires static extents. Dynamic geometry uses
        // explicit result extents in the composite array/dimension family.
        let Some(shape) = input_type.static_shape() else {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` does not support dynamically shaped inputs",
            )));
        };

        self.validate_concatenation_axis(input_type.rank())?;
        let output_type = match self.options.mode {
            CollectiveMode::Untiled => {
                input_type.with_inserted_dimension(self.concatenation_axis, Dimension::Static(effective_axis_size))?
            }
            CollectiveMode::Tiled => {
                let mut output_dimensions = shape.dimensions().to_vec();
                let dimension = &mut output_dimensions[self.concatenation_axis];
                *dimension = dimension.checked_mul(effective_axis_size).ok_or_else(|| {
                    TypeError::invalid(format!(
                        "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` result extent does not fit in `usize`",
                    ))
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
            operation.field("concatenation_axis", self.concatenation_axis)?;
            operation.field("options", format_args!("{:?}", self.options.mode()))?;
            if let Some(axis_index_groups) = self.options.axis_index_groups() {
                operation.field("axis_index_groups", format_args!("{axis_index_groups:?}"))?;
            }
            operation.field("output_variance", format_args!("{:?}", self.output_variance))?;
            if let Some(mesh) = &self.mesh {
                operation.field("mesh", mesh)?;
            }
            Ok(())
        })
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

    #[inline]
    fn effective_axis_size(&self) -> Result<usize, TypeError> {
        if self.output_variance != ParallelAllGatherOutputVariance::Varying && self.options.axis_index_groups.is_some()
        {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` axis index groups are not supported with invariant or reduced \
                 output variance",
            )));
        }
        self.options.effective_axis_size(PARALLEL_ALL_GATHER_OPERATION_NAME, self.axis_size)
    }

    fn adjoint(&self, _input_type: &ArrayType) -> Result<ParallelSumScatterOperation, ProgramError> {
        // A varying all-gather is the adjoint of a sum-scatter with the same mode, axis, participant groups, and mesh,
        // and so is a reduced one, whose unreduced cotangent the sum-scatter consumes. An invariant all-gather has no
        // adjoint collective, because its transpose selects a participant-indexed chunk locally, so its transposition
        // rules never request one.
        if self.output_variance == ParallelAllGatherOutputVariance::Invariant {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "invariant `{PARALLEL_ALL_GATHER_OPERATION_NAME}` has no adjoint collective because its transpose \
                     selects the current participant's chunk",
                ),
            });
        }
        let adjoint = ParallelSumScatterOperation::new(
            self.axis_name.clone(),
            self.axis_size,
            self.concatenation_axis,
            self.options.clone(),
        );
        Ok(match &self.mesh {
            Some(mesh) => adjoint.with_mesh(mesh.clone()),
            None => adjoint,
        })
    }

    #[inline]
    fn adapt_to_batch_axis(&self, input_batch_axis: usize) -> (Self, usize) {
        let (concatenation_axis, output_batch_axis) =
            self.options.mode.forwarded_concatenation_axes(self.concatenation_axis, input_batch_axis);
        (Self { concatenation_axis, ..self.clone() }, output_batch_axis)
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
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` expects an array followed by its output extents",
            )));
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        self.validate_concatenation_axis(input_type.rank())?;

        let base_output_type = match self.options.mode {
            CollectiveMode::Untiled => {
                input_type.with_inserted_dimension(self.concatenation_axis, Dimension::Static(effective_axis_size))?
            }
            CollectiveMode::Tiled => {
                let mut dimensions = input_type.shape().dimensions().to_vec();
                dimensions[self.concatenation_axis] = Dimension::Static(0);
                let sharding =
                    input_type.resized_sharding(dimensions.as_slice(), PARALLEL_ALL_GATHER_OPERATION_NAME)?;
                ArrayType::new(input_type.data_type(), Shape::new(dimensions))
                    .with_memory(input_type.memory())
                    .with_sharding(sharding)?
            }
        };

        let mut output_types = infer_array_ir_shape_changing_collective_output_type(
            PARALLEL_ALL_GATHER_OPERATION_NAME,
            input_types,
            base_output_type,
            &[self.concatenation_axis],
            |output_extents| {
                match self.options.mode {
                    CollectiveMode::Untiled => {
                        let output_extent = &output_extents[self.concatenation_axis];
                        if output_extent != &Dimension::Static(effective_axis_size) {
                            return Err(TypeError::invalid(format!(
                                "`{}` inserted output axis {} extent must equal axis group size {} but got {}",
                                PARALLEL_ALL_GATHER_OPERATION_NAME,
                                self.concatenation_axis,
                                effective_axis_size,
                                output_extent,
                            )));
                        }
                    }
                    CollectiveMode::Tiled => {
                        let input_extent = &input_type.shape().dimensions()[self.concatenation_axis];
                        let output_extent = &output_extents[self.concatenation_axis];
                        if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                            (input_extent, output_extent)
                        {
                            let expected = input_extent.checked_mul(effective_axis_size).ok_or_else(|| {
                                TypeError::invalid(format!(
                                    "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` result extent does not fit in `usize`",
                                ))
                            })?;
                            if *output_extent != expected {
                                return Err(TypeError::invalid(format!(
                                    "`{}` result extent must equal input axis {} extent {} multiplied by axis \
                                     group size {}; expected {} but got {}",
                                    PARALLEL_ALL_GATHER_OPERATION_NAME,
                                    self.concatenation_axis,
                                    input_extent,
                                    effective_axis_size,
                                    expected,
                                    output_extent,
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
            let sharding =
                input_type.resized_sharding(output_type.shape().dimensions(), PARALLEL_ALL_GATHER_OPERATION_NAME)?;
            output_type = output_type.with_sharding(sharding)?;
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
        // The batching rules reject all-gathers over a manual mesh axis, which reduced output variance requires,
        // and infer the output type first, so the concatenation axis is known to be within bounds for the input rank.
        if self.options.axis_index_groups.is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` axis index groups are not supported when a batch transform \
                     binds the collective axis",
                ),
            });
        }

        let axis_extent =
            P::collective_axis_extent(context, PARALLEL_ALL_GATHER_OPERATION_NAME, &self.axis_name, self.axis_size)?;
        let mut input_extents = output_extents.clone();
        match self.options.mode {
            CollectiveMode::Untiled => {
                input_extents.remove(self.concatenation_axis);
            }
            CollectiveMode::Tiled => {
                input_extents[self.concatenation_axis] =
                    P::divide_extents_exactly(context, &output_extents[self.concatenation_axis], &axis_extent)?;
            }
        }

        let input = P::match_collective_axis(context, input, input_extents.as_slice())?;
        let moved = input.into_value().move_axis(0, self.concatenation_axis)?;
        let gathered = P::reshape_collective(context, moved, output_extents.as_slice(), output_sharding)?;
        Ok(ArrayBatch::replicated(gathered))
    }
}

impl<C: Domain<Type = ArrayType, Value: Reshape>> InterpretableOperation<C> for ParallelAllGatherOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // Eager binding does not infer output types, so interpretation validates the shared input contract
        // and the operation payload before applying the degenerate axis rule.
        check_count!("input", inputs, 1, ProgramError);
        self.validate_local_interpretation()?;
        self.validate_degenerate_interpretation()?;

        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let output_type = self.infer_output_types(&input_types, &[])?.remove(0);
        let input = &inputs[0];

        // A single participant gathers only its own value. Untiled mode inserts a size-one gathered axis,
        // which a reshape to the inferred output type expresses, while tiled mode leaves the shape unchanged.
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
        // A matching `batch` level consumes the mapped batch axis by materializing the gather: the batch axis moves to
        // the per-item `concatenation_axis` and, in tiled mode, merges into it item-major (i.e., item 0's chunk first),
        // which matches the StableHLO `all_gather` chunk order. Every batch item sees the same gathered value, so the
        // output is replicated, and an untiled gather also relocates the bounded ragged metadata of its input. A
        // non-matching level forwards the collective to the parent context, unchanged for a replicated input and with
        // its array axes shifted past the batch axis for a mapped one. This rule cannot reuse the shared
        // `shape_changing_collective_batch` rule, which rejects every bounded ragged input.
        if context.axis_name() != Some(self.axis_name.as_str()) {
            ArrayBatch::reject_ragged_inputs(self, inputs)?;
            return context.forward_collective(self, inputs);
        }
        self.reject_mesh_form()?;
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        if self.options.mode == CollectiveMode::Tiled && !input.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "tiled `{PARALLEL_ALL_GATHER_OPERATION_NAME}` cannot represent participant-specific bounded ragged \
                     extents after the participant and concatenation axes are fused",
                ),
            });
        }

        // The gather moves the packed per-item array, so its geometry comes from the packed type rather than from the
        // logical type, which restores the dynamic dimension of every bounded ragged axis that the static-shape type
        // inference cannot consume.
        let input_type = input.value().r#type().unbatched(input.batch_axis())?;
        let (output_type, output_extents) = context.infer_collective_output_type_and_extents(self, &input_type)?;
        let input_batch_axis = input.batch_axis_position();
        let ragged_axes = input.ragged_axes().to_vec();
        let mut output = self.batch_matching_axis(context, input, output_extents, output_type.sharding().cloned())?;
        if !ragged_axes.is_empty() {
            let ragged_axes = self.gathered_ragged_axes(context, ragged_axes, input_batch_axis, input_type.rank())?;
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
    O: Operation<Type = ArrayType>
        + From<ZeroOperation<ArrayType>>
        + From<AddOperation<ArrayType>>
        + From<BroadcastOperation>
        + From<ReshapeOperation>
        + From<DynamicSliceOperation>
        + From<AxisIndexOperation>
        + From<ParallelVaryOperation>
        + From<ParallelSumScatterOperation>,
> TransposableOperation<V, O> for ParallelAllGatherOperation
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // A varying or reduced result transposes to its adjoint sum-scatter. An invariant result has no adjoint
        // collective: every participant holds the complete output cotangent, so the input cotangent of participant `i`
        // is chunk `i` of that cotangent, which this rule selects locally, as JAX's `all_gather_invariant` transpose
        // does. A known input and a structural-zero output cotangent contribute nothing.
        if self.output_variance != ParallelAllGatherOutputVariance::Invariant {
            return self.linear_collective_transpose(context, inputs, outputs, accumulators);
        }

        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 1, DifferentiationError);

        let MaybeZero::Value(cotangent) = &outputs[0] else {
            return Ok(());
        };

        if inputs[0].is_known() {
            return Ok(());
        }

        // The homogeneous family has no dimension values, so the selected chunk needs static extents. A dynamically
        // shaped input can only occur in a mixed program, whose linearization stages the composite rule instead.
        let input_cotangent_type = inputs[0].r#type().cotangent()?;
        let participants_type =
            input_cotangent_type.with_inserted_dimension(self.concatenation_axis, Dimension::Static(self.axis_size))?;
        let Some(participants_shape) = participants_type.static_shape() else {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_ALL_GATHER_OPERATION_NAME}` transpose requires a statically shaped input but got \
                 `{input_cotangent_type}`",
            ))
            .into());
        };

        let mut sizes = participants_shape.dimensions().to_vec();
        sizes[self.concatenation_axis] = 1;

        // Over a manual mesh axis, the output cotangent is invariant across the gathered axis, while every participant
        // selects a different chunk of it, so the selected chunk varies over that axis. The cotangent can carry a
        // tangent itself (e.g., under nested differentiation), and so it is made varying with a real `parallel_vary`
        // transition, whose transpose is the cross-device sum, rather than by retyping the selected chunk.
        let mut cotangent = cotangent.clone();
        if self.mesh.is_some()
            && !cotangent
                .r#type()
                .sharding()
                .is_some_and(|sharding| sharding.varying_manual_axes().contains(self.axis_name.as_str()))
        {
            let operation = ParallelVaryOperation::new(self.axis_name.clone());
            cotangent = context.bind(O::from(operation), Vec::new(), &[cotangent])?.remove(0);
        }

        // A tiled gather concatenates the participants' chunks along `concatenation_axis`, so splitting that axis into
        // a participant axis followed by the chunk axis yields the untiled view, whose participant axis is then sliced
        // at the current participant's index and removed.
        if self.options.mode == CollectiveMode::Tiled {
            let operation = ReshapeOperation::new(participants_type.shape().clone())
                .with_output_sharding(participants_type.sharding().cloned());
            cotangent = context.bind(O::from(operation), Vec::new(), &[cotangent])?.remove(0);
        }

        // The start indices must vary over the same manual mesh axes as the cotangent that they slice, which is the
        // variation of the input cotangent. The participant index varies at most over the gathered manual mesh axis,
        // so it is placed on the mesh of the cotangent if necessary and takes the remaining axes through staged
        // `parallel_vary` transitions. The zero start indices are non-differentiable constants, so they are created
        // directly with that variation. A degenerate axis selects row zero instead of reading its participant index,
        // so that the pullback of a degenerate ordinary all-gather remains interpretable without an enclosing binder.
        let index_sharding = input_cotangent_type
            .sharding()
            .filter(|sharding| !sharding.varying_manual_axes().is_empty())
            .map(|sharding| {
                Sharding::replicated(sharding.mesh().clone(), 0)
                    .with_varying_manual_axes(sharding.varying_manual_axes().clone())
            })
            .transpose()
            .map_err(TypeError::from)?;

        let start = if self.axis_size == 1 {
            None
        } else {
            let mut operation = AxisIndexOperation::new(self.axis_name.clone());
            if let Some(mesh) = &self.mesh {
                operation = operation.with_mesh(mesh.clone());
            }
            let mut start = context.bind(O::from(operation), Vec::new(), &[])?.remove(0);
            if let Some(index_sharding) = &index_sharding {
                if start.r#type().sharding().is_none() {
                    let mesh_type = ArrayType::scalar(DataType::U64)
                        .with_sharding(Sharding::replicated(index_sharding.mesh().clone(), 0))
                        .map_err(TypeError::from)?;
                    let operation = BroadcastOperation::new(mesh_type, Vec::new());
                    start = context.bind(O::from(operation), Vec::new(), &[start])?.remove(0);
                }
                let start_axes = start
                    .r#type()
                    .sharding()
                    .map(|sharding| sharding.varying_manual_axes().clone())
                    .unwrap_or_default();
                for axis in index_sharding.varying_manual_axes().difference(&start_axes) {
                    let operation = ParallelVaryOperation::new(axis.clone());
                    start = context.bind(O::from(operation), Vec::new(), &[start])?.remove(0);
                }
            }
            Some(start)
        };

        let zero = ZeroOperation::new(
            ArrayType::scalar(DataType::U64).with_sharding(index_sharding).map_err(TypeError::from)?,
        );

        let zero = context.bind(O::from(zero), Vec::new(), &[])?.remove(0);
        let start = start.unwrap_or_else(|| zero.clone());
        let mut slice_inputs = vec![zero; 1 + sizes.len()];
        slice_inputs[0] = cotangent;
        slice_inputs[1 + self.concatenation_axis] = start;
        let selected = context
            .bind(O::from(DynamicSliceOperation::new(sizes)), Vec::new(), slice_inputs.as_slice())?
            .remove(0);
        let operation = ReshapeOperation::new(input_cotangent_type.shape().clone())
            .with_output_sharding(input_cotangent_type.sharding().cloned());
        let contribution = context.bind(O::from(operation), Vec::new(), &[selected])?.remove(0);
        accumulators[0].accumulate(context, MaybeZero::Value(contribution))?;
        Ok(())
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
        self.validate_local_interpretation()?;
        self.shape_changing_collective_interpret_in_parent::<C>(inputs)
    }
}

impl<
    C: Context<
            Type = ArrayIrType,
            Value: Assert
                       + DimensionSize
                       + DynamicBroadcast
                       + ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>
                       + ValueProjection<
                DimensionType,
                Projected: Value<Type = DimensionType> + Mul + Div + Rem + Compare<C::Value> + DimensionMax,
            >,
            Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Operation: From<ConstantOperation<DimensionValue>>
                           + From<ParallelAllGatherOperation>
                           + From<DimensionSizeOperation>
                           + From<DynamicBroadcastOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
> MemberBatchableOperation<C, ArrayIrBatchingPolicy> for ParallelAllGatherOperation
{
    fn batch_in_parent<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        // The explicit result extents remain ordinary replicated dimension inputs, except that the extent of an axis
        // that carries a mapped bounded ragged dimension carries that dimension's per-item extents. A matching level
        // delegates its array mechanics to the homogeneous `batch_matching_axis` kernel over the array projection of
        // its parent and relocates the bounded ragged metadata like the homogeneous rule. This rule cannot reuse the
        // shared `shape_changing_collective_batch_in_parent` rule, which rejects every bounded ragged input.
        let Some((array, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };

        // Infer the per-item result type before lifting any physical axes. This validates the per-item contract in
        // both branches below and supplies the output sharding that the matching-axis kernel assigns.
        let logical_input_types = inputs.iter().map(|input| input.unbatched_type().clone()).collect::<Vec<_>>();
        let mut logical_output_types = self.infer_array_ir_output_types(logical_input_types.as_slice())?;
        let logical_output_type = <&ArrayType>::try_from(&logical_output_types.remove(0))?.clone();

        // A level that does not bind the gathered axis forwards the collective to its parent with its array axes
        // shifted past the mapped axis, like JAX's ordinary `all_gather` batcher. Bounded ragged inputs are rejected
        // there, because the participants' per-item extents would have to be gathered alongside the array, which the
        // forwarded collective does not do.
        if context.axis_name() != Some(self.axis_name.as_str()) {
            ArrayIrBatch::reject_ragged_inputs(self, inputs)?;

            // A result extent describes the shape shared by every batch item, so it must be replicated.
            for output_extent in output_extents {
                output_extent.validate_replicated_dimension()?;
            }
            return Ok(context.forward_collective(self, array, output_extents)?.into());
        }

        // A matching level consumes the collective by rearranging its own batch items, which cannot stand in for the
        // communication across a manual mesh axis that a mesh collective describes.
        self.reject_mesh_form()?;

        // Tiling fuses the participant axis into the concatenation axis, which places the padded chunks of all batch
        // items back to back, so the live elements of the result would no longer form the prefix that one
        // `RaggedAxis` describes.
        if self.options.mode == CollectiveMode::Tiled && !array.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "tiled `{PARALLEL_ALL_GATHER_OPERATION_NAME}` cannot represent participant-specific bounded ragged \
                     extents after the participant and concatenation axes are fused",
                ),
            });
        }

        // Every bounded ragged input axis is matched against the explicit result extent of the output axis that it
        // becomes. The per-item type inference above already requires that extent to carry the same bounded ragged
        // dimension, because the gather leaves that axis unchanged. A mapped input supplies its per-item extents
        // through a mapped result extent, while a replicated input supplies its scalar extents through a replicated
        // one. Each ragged axis yields its output axis, the packed capacity of its input axis, and its metadata over
        // the array projection.
        let input_batch_axis = array.batch_axis_position();
        let ragged_axes = array
            .ragged_axes()
            .iter()
            .map(|ragged_axis| {
                // Removing the mapped batch axis from the physical ragged axis yields the per-item input axis, which
                // the untiled gather shifts by one when it follows the inserted participant axis.
                let logical_axis =
                    ragged_axis.axis() - usize::from(input_batch_axis.is_some_and(|axis| axis < ragged_axis.axis()));
                let output_axis = logical_axis + usize::from(logical_axis >= self.concatenation_axis);
                let output_extent = &output_extents[output_axis];
                let extents = if let Some(input_batch_axis) = input_batch_axis {
                    let Some(extents) = output_extent.mapped_dimension_extents() else {
                        return Err(BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "untiled `{}` output axis {} must carry mapped extents for bounded \
                                 ragged dimension `{}`",
                                PARALLEL_ALL_GATHER_OPERATION_NAME,
                                output_axis,
                                ragged_axis.dimension(),
                            ),
                        });
                    };

                    // The per-item extents of a mapped input are indexed by the extent axes of its ragged metadata,
                    // so the mapped result extent must be batched at the position of the input batch axis among them.
                    let expected_extent_axis = ragged_axis
                        .extent_axes()
                        .iter()
                        .position(|axis| *axis == input_batch_axis)
                        .map(BatchAxis::from_position)
                        .ok_or_else(|| BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "untiled `{}` bounded ragged dimension `{}` does not carry extents \
                                 for the mapped input axis",
                                PARALLEL_ALL_GATHER_OPERATION_NAME,
                                ragged_axis.dimension(),
                            ),
                        })?;

                    if output_extent.batch_axis() != expected_extent_axis {
                        return Err(BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "untiled `{}` output axis {} maps bounded ragged extents on {} instead of {}",
                                PARALLEL_ALL_GATHER_OPERATION_NAME,
                                output_axis,
                                output_extent.batch_axis(),
                                expected_extent_axis,
                            ),
                        });
                    }
                    extents.clone()
                } else {
                    output_extent.validate_replicated_dimension()?;
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

        // The homogeneous kernel performs the gather over the array projection of the parent context, so the array
        // and its result extents are projected onto their array and dimension families.
        let array = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(array.value().clone())?,
            array.batch_axis(),
        )?;

        // The kernel reshapes the packed array, so every bounded ragged output axis uses the packed capacity of its
        // input axis rather than its per-item extents. Every other result extent describes the shape shared by all
        // batch items and must be replicated.
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
                extent.validate_replicated_dimension()?;
                Ok(<C::Value as ValueProjection<DimensionType>>::into_projected(extent.value().clone())?)
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;

        // Gather the packed batch items with the homogeneous kernel, whose result is replicated because every batch
        // item receives the same gathered value, and then relocate the bounded ragged metadata onto the gathered axes.
        let ragged_axes = ragged_axes.into_iter().map(|(_, _, ragged_axis)| ragged_axis).collect::<Vec<_>>();
        let mut output = self.batch_matching_axis::<DynamicArrayExtentBatchingPolicy>(
            &projected_context,
            &array,
            output_extents,
            logical_output_type.sharding().cloned(),
        )?;

        if !ragged_axes.is_empty() {
            let ragged_axes =
                self.gathered_ragged_axes(&projected_context, ragged_axes, input_batch_axis, input_rank)?;
            output = output.with_ragged_axes(ragged_axes)?;
        }

        // Embed the result and its bounded ragged metadata back into the composite family.
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

impl<
    C: Context<
            Type = ArrayIrType,
            Operation: From<ConstantOperation<DimensionValue>>
                           + From<DimensionFromScalarOperation>
                           + From<DimensionSizeOperation>
                           + From<DynamicReshapeOperation>
                           + From<DynamicSliceOperation<ArrayIrType>>
                           + From<LinearCallOperation<ArrayIrType>>
                           + From<ParallelAllGatherOperation>
                           + From<ParallelSumScatterOperation>
                           + OperationProjection<
                ArrayType,
                Projected: From<AxisIndexOperation> + From<ParallelVaryOperation>,
            > + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
        >,
> MemberDifferentiableOperation<C> for ParallelAllGatherOperation
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // A varying or reduced result transposes to its adjoint sum-scatter, which the shared rule stages with
        // the explicit output extents and the exact input shape as residuals. An invariant result has no adjoint
        // collective, so its tangent is instead one linear call whose transpose selects the current participant's
        // chunk of the output cotangent locally, like the homogeneous transposition rule.
        if self.output_variance != ParallelAllGatherOutputVariance::Invariant {
            return self.shape_changing_collective_jvp_in_parent(context, inputs);
        }

        // The composite inputs are the gathered array followed by one explicit extent per output axis. Only the array
        // can carry a live tangent, because the extents are integer-valued dimensions.
        let Some((array, _)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };

        // The primal output is the same invariant gather applied to the primal inputs. A structurally zero array
        // tangent stays symbolic, retyped to the output tangent type because the gather changes the shape.
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal = context.primal().bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
        let tangent = match array.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
            MaybeZero::Value(array_tangent) => {
                // The linear call is staged in the tangent context, so the primal values that its regions need are
                // first transferred there. Splitting cannot fail, because the inputs were checked to be non-empty.
                let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
                let (array, output_extents) = tangent_inputs.split_first().unwrap();
                let context = context.tangent();

                // The two regions of the linear call are traced as separate programs whose only inputs are the call's
                // residuals followed by its linear inputs or output cotangents, so every primal value that they read
                // must be retained as a residual. The forward region needs the explicit output extents, while the
                // transpose region needs the exact runtime shape of the primal input to size and reshape its selected
                // chunk. `retain_all` returns the residual slot of each extent, and `retain_shape` returns a plan that
                // rebuilds every input dimension inside a region. That plan retains nothing for static axes and reuses
                // an already-retained extent with the same dimension variable (e.g., an output extent of an axis that
                // the gather leaves unchanged), so only the remaining dynamic axes bind `dimension_size` reads.
                let mut residuals = LinearResiduals::new();
                let output_extents = residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                let input_shape = residuals.retain_shape(context, array.primal())?;

                // Each region closure owns its copy of the operation and of the residual slots that it reads. The
                // input cotangent type supplies the sharding that the transpose region restores on its result.
                let forward_operation = self.clone();
                let forward_output_extents = output_extents.clone();
                let transpose_operation = self.clone();
                let transpose_target_type = <&ArrayType>::try_from(array.primal().r#type().as_ref())?.cotangent()?;
                let tangent = LinearCallOperation::stage(
                    context,
                    residuals.into_values(),
                    vec![array_tangent.clone()],
                    move |residuals, linear_inputs| {
                        // The forward region applies the same invariant gather to the tangent, with the explicit output
                        // extents read from the region's residual inputs.
                        let mut collective_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                        collective_inputs.push(linear_inputs[0].clone());
                        collective_inputs.extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                        linear_inputs[0].dispatch_domain().bind(
                            forward_operation,
                            Vec::new(),
                            collective_inputs.as_slice(),
                        )
                    },
                    move |residuals, output_cotangents| {
                        // The transpose region maps the output cotangent to the input cotangent by selecting the
                        // current participant's chunk of it along the concatenation axis, which needs no communication
                        // because every participant holds the complete invariant output cotangent. All extents and
                        // offsets are first-class dimension values, so that they remain exact for dynamically shaped
                        // inputs.

                        // Rebuild the exact input dimensions from the retained shape, staging static extents as
                        // dimension constants in the transpose context.
                        let transpose_context = output_cotangents[0].dispatch_domain();
                        let input_dimensions = input_shape.dimensions(&transpose_context, residuals)?;
                        let output_cotangent_type = output_cotangents[0].r#type();
                        let output_cotangent_type = <&ArrayType>::try_from(output_cotangent_type.as_ref())?;
                        let output_rank = output_cotangent_type.rank();

                        // Every participant contributes one chunk along the concatenation axis. A tiled gather
                        // concatenates chunks whose extent is the input extent along that axis, while an untiled
                        // gather stacks one size-one row per participant along its inserted participant axis.
                        let zero = transpose_context
                            .bind(
                                DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(0)?)),
                                Vec::new(),
                                &[],
                            )?
                            .remove(0);
                        let chunk_extent = match transpose_operation.options().mode() {
                            CollectiveMode::Tiled => input_dimensions[transpose_operation.concatenation_axis()].clone(),
                            CollectiveMode::Untiled => transpose_context
                                .bind(
                                    DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(1)?)),
                                    Vec::new(),
                                    &[],
                                )?
                                .remove(0),
                        };

                        // The chunk of participant `i` starts at offset `i * chunk_extent`. A degenerate axis has only
                        // participant zero, so its offset is zero and needs no `axis_index`, which keeps the pullback
                        // interpretable without an enclosing binder. Otherwise, the participant index is read through
                        // `axis_index` (on the operation's mesh for a manual mesh axis) and converted into a dimension
                        // whose variable is bounded by the axis size, so that the offset becomes a symbolic dimension
                        // product (e.g., `x_index * 2`) with known bounds, which the dynamic slice below checks.
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
                                DimensionBounds::non_negative(Some(transpose_operation.axis_size()))?,
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

                        // Every other axis starts at zero and spans its full input extent. An untiled output cotangent
                        // also has the participant axis, along which the slice keeps one size-one row.
                        let mut starts = vec![zero; output_rank];
                        starts[transpose_operation.concatenation_axis()] = start;
                        let mut slice_sizes = input_dimensions.clone();
                        if transpose_operation.options().mode() == CollectiveMode::Untiled {
                            slice_sizes.insert(transpose_operation.concatenation_axis(), chunk_extent);
                        }

                        // Over a manual mesh axis, the output cotangent is invariant across the gathered axis, while
                        // every participant selects a different chunk of it, so the selected chunk varies over that
                        // axis. The cotangent can carry a tangent itself (e.g., under nested differentiation), and so
                        // it is made varying with a real `parallel_vary` transition, whose transpose is the
                        // cross-device sum, rather than by retyping the selected chunk.
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

                        // The composite `dynamic_slice` consumes the sliced array followed by all start offsets and
                        // then all slice extents, with one of each per array axis.
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

                        // Reshaping to the exact input dimensions removes the size-one participant axis of an untiled
                        // chunk and leaves a tiled chunk's shape unchanged, while assigning the input cotangent's
                        // sharding to the result in both modes.
                        let mut reshape_inputs = Vec::with_capacity(1 + input_dimensions.len());
                        reshape_inputs.push(selected);
                        reshape_inputs.extend(input_dimensions);
                        transpose_context.bind(
                            DynamicReshapeOperation::new()
                                .with_output_sharding(transpose_target_type.sharding().cloned()),
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
}

/// Represents the ability to gather values across the participants of a named axis, so that every participant receives
/// all of them in participant order, by staging a [`ParallelAllGatherOperation`]. This is the Ryft analogue of JAX's
/// [`jax.lax.all_gather`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_gather.html), whose default
/// `tiled = False` corresponds to [`ParallelAllGather::parallel_all_gather`], whose `tiled = True` corresponds to
/// [`ParallelAllGather::parallel_all_gather_tiled`], and whose `to` argument corresponds to the
/// [`ParallelAllGatherOutputVariance`] of [`ParallelAllGather::parallel_all_gather_with_options`].
/// Refer to the documentation of [`ParallelAllGatherOperation`] for the semantics and transformation rules.
///
/// Over a manual mesh axis, an input that does not vary over the axis is first made varying through [`ParallelVary`],
/// so that every device's copy is gathered. Homogeneous array values require static extents, while composite array IR
/// values stage one explicit extent value per output axis and so also support dynamic extents.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// Gather two rows, so that every batch item receives both of them:
///
/// ```
/// # use ryft_core::{
/// #     Array, ArrayBatchingPolicy, ArrayOperation, BatchAxis, BatchAxisSpecification, BatchingTracer, EagerContext,
/// #     ParallelAllGather, batch,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rows = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])?;
/// let gathered = batch(
///     |row: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
///         row.parallel_all_gather_tiled("rows", 0)
///     },
///     rows,
///     BatchAxis::new(0),
///     BatchAxis::replicated(),
///     BatchAxisSpecification::named("rows"),
/// )?;
/// assert_eq!(gathered, Array::vector(vec![1.0, 2.0, 3.0, 4.0])?);
/// # Ok(())
/// # }
/// ```
#[capability]
pub trait ParallelAllGather<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns the values of the participants of the named axis `axis_name`, stacked along a new axis that is inserted
    /// at `concatenation_axis`, so that participant `i`'s value becomes row `i` of that axis. Over a manual mesh axis,
    /// the result keeps varying over `axis_name`.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `concatenation_axis`: Position of the inserted axis in the result, whose rank is one more than the rank
    ///     of this value. Negative positions count from the end, so `-1` appends a new trailing axis.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`ParallelAllGather::parallel_all_gather_with_options`].
    #[inline]
    fn parallel_all_gather<ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        concatenation_axis: ConcatenationAxis,
    ) -> Result<Self, ProgramError> {
        self.parallel_all_gather_with_options(
            axis_name,
            concatenation_axis,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        )
    }

    /// Returns the values of the participants of the named axis `axis_name`, concatenated along the existing
    /// `concatenation_axis`, so that participant `i`'s value becomes chunk `i` of that axis. Over a manual mesh axis,
    /// the result keeps varying over `axis_name`.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `concatenation_axis`: Axis of this value along which the values are concatenated.
    ///     Negative axes count from the end.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`ParallelAllGather::parallel_all_gather_with_options`].
    #[inline]
    fn parallel_all_gather_tiled<ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        concatenation_axis: ConcatenationAxis,
    ) -> Result<Self, ProgramError> {
        self.parallel_all_gather_with_options(
            axis_name,
            concatenation_axis,
            CollectiveOptions::new(CollectiveMode::Tiled),
            ParallelAllGatherOutputVariance::Varying,
        )
    }

    /// Returns the values of the participants of the named axis `axis_name`, gathered along `concatenation_axis` with
    /// the tiling mode and participant groups of `options` and with the manual variation that `output_variance`
    /// selects.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `concatenation_axis`: Position of the inserted axis in the result in untiled mode, or axis of this value
    ///     along which the values are concatenated in tiled mode. Negative axes count from the end of the result in
    ///     untiled mode, whose rank is one more than the rank of this value, and from the end of this value in tiled
    ///     mode.
    ///   - `options`: [`CollectiveMode`] and optional participant groups of the collective.
    ///   - `output_variance`: Manual variation of the result over a manual mesh axis. An ordinary all-gather
    ///     preserves the mesh state of this value.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Axis`] error wrapping [`AxisError::UnboundAxisName`] when no enclosing binder binds
    /// `axis_name` or [`AxisError::OutOfBounds`] when `concatenation_axis` is out of bounds, and a [`ProgramError`]
    /// if the participant groups are invalid or combined with invariant or reduced output variance, if reduced output
    /// variance is requested for an axis that is not a manual mesh axis, or if this value carries reduction state
    /// (i.e., is unreduced or reduced) over the gathered manual mesh axis.
    fn parallel_all_gather_with_options<ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        concatenation_axis: ConcatenationAxis,
        options: CollectiveOptions,
        output_variance: ParallelAllGatherOutputVariance,
    ) -> Result<Self, ProgramError>;
}

impl ParallelAllGather<ArrayType> for Array {
    #[inline]
    fn parallel_all_gather_with_options<ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        _concatenation_axis: ConcatenationAxis,
        _options: CollectiveOptions,
        _output_variance: ParallelAllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        // A concrete `Array` never executes inside an axis binder, because the values under a `batch` level or inside
        // a manual region are tracers, so every axis name is unbound for it.
        Err(AxisError::UnboundAxisName { name: axis_name.to_string() }.into())
    }
}

impl<A: Value<Type = ArrayType> + ParallelAllGather<ArrayType>> ParallelAllGather<ArrayIrType> for ArrayIrValue<A> {
    #[inline]
    fn parallel_all_gather_with_options<ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        concatenation_axis: ConcatenationAxis,
        options: CollectiveOptions,
        output_variance: ParallelAllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        // A concrete composite value performs the collective through its array member.
        let array = ValueProjection::<ArrayType>::into_projected(self.clone())?;
        Ok(<Self as ValueProjection<ArrayType>>::from_projected(array.parallel_all_gather_with_options(
            axis_name,
            concatenation_axis,
            options,
            output_variance,
        )?))
    }
}

impl<V: ParallelAllGather<ArrayIrType> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>>
    ParallelAllGather<ArrayType> for ProjectedValue<ArrayType, V>
{
    #[inline]
    fn parallel_all_gather_with_options<ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        concatenation_axis: ConcatenationAxis,
        options: CollectiveOptions,
        output_variance: ParallelAllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        self.value()
            .parallel_all_gather_with_options(axis_name, concatenation_axis, options, output_variance)?
            .into_projected()
            .map_err(Into::into)
    }
}

impl<
    V: ShapeChangingCollectiveValue<
            DispatchDomain: Context<Value = V, Operation: From<ParallelAllGatherOperation>> + NamedAxes,
        > + ParallelVary,
> ParallelAllGather<ArrayType> for V
{
    fn parallel_all_gather_with_options<ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        concatenation_axis: ConcatenationAxis,
        options: CollectiveOptions,
        output_variance: ParallelAllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        // Homogeneous values opt into direct staging, while projected values retain composite extent delegation. Over
        // a manual mesh axis, an input that does not vary over the axis is first made varying, so that every device's
        // copy is gathered, while reduction state over the axis is rejected first, so that it is reported as an
        // all-gather error rather than as a `parallel_vary` error. An untiled gather inserts an axis, so its
        // concatenation axis is a position in the result, whose rank is one more than the rank of the input.
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let result_rank = self.r#type().rank() + usize::from(options.mode == CollectiveMode::Untiled);
        let concatenation_axis = concatenation_axis.into().normalize(result_rank)?;
        let mut operation = ParallelAllGatherOperation::new(
            axis_name.to_string(),
            axis_size,
            concatenation_axis,
            options,
            output_variance,
        );
        operation.effective_axis_size()?;
        let mut input = self.clone();
        if let Some(NamedAxis::Mesh { mesh, .. }) = context.named_axis(axis_name) {
            operation.validate_input_reduction_state(self.r#type().as_ref())?;
            if !self.r#type().sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis_name)) {
                input = input.parallel_vary(axis_name)?;
            }
            operation = operation.with_mesh(mesh);
        }
        let mut outputs = context.bind(operation, Vec::new(), &[input])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<Type = ArrayIrType, Operation: From<ParallelAllGatherOperation>>
                                + DimensionConstant
                                + NamedAxes,
        > + DimensionSize<V>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType> + Mul>
        + ValueProjection<ArrayType, Projected: ParallelVary>,
> ParallelAllGather<ArrayIrType> for V
{
    fn parallel_all_gather_with_options<ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        concatenation_axis: ConcatenationAxis,
        options: CollectiveOptions,
        output_variance: ParallelAllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        // A composite value binds a `ParallelAllGatherOperation` through its own context, followed by one explicit
        // extent value per output axis. Over a manual mesh axis, the operation records the mesh, and an input that
        // does not vary over the axis is first made varying through its array view, so that every device's copy is
        // gathered, while reduction state over the axis is rejected first, so that it is reported as an all-gather
        // error rather than as a `parallel_vary` error. An untiled gather inserts an axis, so its concatenation axis
        // is a position in the result, whose rank is one more than the rank of the input. The axis is normalized
        // before the explicit result extents are derived from the input extents.
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let rank = input_type.rank();
        let concatenation_axis =
            concatenation_axis.into().normalize(rank + usize::from(options.mode == CollectiveMode::Untiled))?;
        let mut operation = ParallelAllGatherOperation::new(
            axis_name.to_string(),
            axis_size,
            concatenation_axis,
            options,
            output_variance,
        );

        let effective_axis_size = operation.effective_axis_size()?;
        let mut input = self.clone();
        if let Some(NamedAxis::Mesh { mesh, .. }) = context.named_axis(axis_name) {
            operation.validate_input_reduction_state(input_type)?;
            if !input_type.sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis_name)) {
                let array = ValueProjection::<ArrayType>::into_projected(self.clone())?;
                input = <V as ValueProjection<ArrayType>>::from_projected(array.parallel_vary(axis_name)?);
            }
            operation = operation.with_mesh(mesh);
        }

        let mut output_extents = (0..rank).map(|axis| input.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
        let participants = context.dimension_constant(effective_axis_size)?;
        match operation.options.mode {
            CollectiveMode::Untiled => output_extents.insert(concatenation_axis, participants),
            CollectiveMode::Tiled => {
                let extent =
                    ValueProjection::<DimensionType>::into_projected(output_extents[concatenation_axis].clone())?;
                let participants = ValueProjection::<DimensionType>::into_projected(participants)?;
                output_extents[concatenation_axis] =
                    ValueProjection::<DimensionType>::from_projected(extent.mul(&participants)?);
            }
        }
        let inputs = std::iter::once(input).chain(output_extents).collect::<Vec<_>>();
        let mut outputs = context.bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayOperation, Layout, Memory, MeshAxis, MeshAxisType, ShardingDimension, StridedLayout,
    };
    use crate::batching::{BatchAxisSpecification, BatchingTracer, batch};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::differentiate_at;
    use crate::macros::{
        check_gradient, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::operations::collectives::tests::{batch_collective, collective_program};
    use crate::operations::manipulation::slicing::Slice;
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::partial::PartialEvaluationOutput;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};

    use super::*;

    #[test]
    fn test_parallel_all_gather_output_variance_default() {
        // Varying is the default, which is also the output variance that `parallel_all_gather` and
        // `parallel_all_gather_tiled` stage.
        assert_eq!(ParallelAllGatherOutputVariance::default(), ParallelAllGatherOutputVariance::Varying);
    }

    #[test]
    fn test_parallel_all_gather() {
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            4,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        );
        assert_eq!(operation.name(), PARALLEL_ALL_GATHER_OPERATION_NAME);
        assert_eq!(operation.axis_name(), "x");
        assert_eq!(operation.axis_size(), 4);
        assert_eq!(operation.concatenation_axis(), 0);
        assert_eq!(operation.options(), &CollectiveOptions::tiled());
        assert_eq!(operation.output_variance(), ParallelAllGatherOutputVariance::Varying);
        assert_eq!(operation.mesh(), None);
        assert_eq!(operation.effective_axis_size(), Ok(4));
        assert_eq!(
            operation.to_string(),
            indoc! {r#"
                parallel_all_gather [
                    axis_name="x",
                    axis_size=4,
                    concatenation_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ]
            "#}
            .trim_end(),
        );

        // Participant groups restrict the effective axis size to the size of one group, but only a varying result can
        // be gathered within groups.
        let grouped = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                4,
                0,
                grouped.clone(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .effective_axis_size(),
            Ok(2),
        );
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                4,
                0,
                grouped.clone(),
                ParallelAllGatherOutputVariance::Invariant,
            )
            .effective_axis_size(),
            Err(TypeError::invalid(
                "`parallel_all_gather` axis index groups are not supported with invariant or reduced output variance",
            )),
        );
        assert_eq!(
            ParallelAllGatherOperation::new("x".to_string(), 4, 0, grouped, ParallelAllGatherOutputVariance::Reduced)
                .effective_axis_size(),
            Err(TypeError::invalid(
                "`parallel_all_gather` axis index groups are not supported with invariant or reduced output variance",
            )),
        );
    }

    #[test]
    fn test_parallel_all_gather_with_mesh() {
        // An all-gather over a manual mesh axis records and renders its mesh.
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        );
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let mesh_operation = operation.clone().with_mesh(mesh.clone());
        assert_eq!(mesh_operation.mesh(), Some(&mesh));
        assert_eq!(
            mesh_operation.to_string(),
            indoc! {r#"
                parallel_all_gather [
                    axis_name="x",
                    axis_size=2,
                    concatenation_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                    mesh=['x'=2:manual],
                ]
            "#}
            .trim_end(),
        );
        assert_ne!(mesh_operation, operation);
    }

    #[test]
    fn test_parallel_all_gather_type_inference() {
        // A tiled all-gather multiplies its concatenation axis by the axis size, while an untiled one inserts a new
        // axis of that size at its concatenation axis. Static result-shape arithmetic requires statically shaped
        // inputs.
        check_operation_type_inference!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                4,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            ),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2])],
                    output_types = [ArrayType::new_static(DataType::F32, [8])],
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "`parallel_all_gather` concatenation axis 0 is out of bounds for output rank 0",
                },
                {
                    input_types = [ArrayType::new(
                        DataType::F32,
                        Shape::new(vec![Dimension::Dynamic(
                            DimensionVariable::new("dynamic", DimensionBounds::unbounded()),
                        )]),
                    )],
                    error = "`parallel_all_gather` does not support dynamically shaped inputs",
                },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                4,
                1,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Varying,
            ),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2, 3])],
                    output_types = [ArrayType::new_static(DataType::F32, [2, 4, 3])],
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "`parallel_all_gather` concatenation axis 1 is out of bounds for output rank 1",
                },
            ],
        );

        // Participant groups combined with invariant output variance are rejected before the input shape is examined.
        check_operation_type_inference!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                4,
                0,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 1], vec![2, 3]]),
                ParallelAllGatherOutputVariance::Invariant,
            ),
            cases = [{
                input_types = [ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                        "dynamic",
                        DimensionBounds::unbounded(),
                    ))]),
                )],
                error = "`parallel_all_gather` axis index groups are not supported with invariant or reduced output \
                         variance",
            }],
        );

        // An all-gather over a manual mesh axis requires its input to vary over that axis on the operation's mesh.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let varying = sharding.clone().with_varying_manual_axes(["x"]).unwrap();
        let with_sharding = |extent: usize, sharding: &Sharding| {
            ArrayType::new_static(DataType::F32, [extent]).with_sharding(sharding.clone()).unwrap()
        };
        let other_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        check_operation_type_inference!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .with_mesh(mesh.clone()),
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
        let input = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();

        // Over a manual mesh axis, the output variance selects whether the gathered result keeps varying over the axis,
        // becomes invariant over it, or records it as reduced. Both type families apply the same transition.
        let varying = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let invariant = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.clone(), 2))
            .unwrap();
        let reduced = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.clone(), 2).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .with_mesh(mesh.clone()),
            cases = [{ input_types = [input.clone()], output_types = [varying.clone()] }],
        );
        check_operation_type_inference!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Invariant,
            )
            .with_mesh(mesh.clone()),
            cases = [{ input_types = [input.clone()], output_types = [invariant.clone()] }],
        );
        check_operation_type_inference!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Reduced,
            )
            .with_mesh(mesh.clone()),
            cases = [{ input_types = [input.clone()], output_types = [reduced.clone()] }],
        );
        let array_ir_input_types = [
            ArrayIrType::Array(input),
            DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
            DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
        ];
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .with_mesh(mesh.clone())
            .infer_array_ir_output_types(&array_ir_input_types),
            Ok(vec![varying.into()]),
        );
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Invariant,
            )
            .with_mesh(mesh.clone())
            .infer_array_ir_output_types(&array_ir_input_types),
            Ok(vec![invariant.into()]),
        );
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Reduced,
            )
            .with_mesh(mesh)
            .infer_array_ir_output_types(&array_ir_input_types),
            Ok(vec![reduced.into()]),
        );

        // Reduced output variance records state on a manual mesh axis, so an ordinary gather rejects it.
        check_operation_type_inference!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Reduced,
            ),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [2])],
                error = "`parallel_all_gather` with reduced output variance requires a manual mesh axis",
            }],
        );
    }

    #[test]
    fn test_parallel_all_gather_type_inference_reduction_state() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        )
        .with_mesh(mesh.clone());
        let with_sharding = |extent: usize, sharding: &Sharding| {
            ArrayType::new_static(DataType::F32, [extent]).with_sharding(sharding.clone()).unwrap()
        };
        let sharding = Sharding::replicated(mesh, 1);
        let unrelated_pending_sum =
            sharding.clone().with_varying_manual_axes(["x"]).unwrap().with_unreduced_axes(["y"]).unwrap();

        // Gathering over `x` commutes with an independent pending sum over `y`, while gathering over a pending or
        // completed sum on `x` itself would change its reduction semantics.
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [with_sharding(4, &unrelated_pending_sum)],
                    output_types = [with_sharding(8, &unrelated_pending_sum)],
                },
                {
                    input_types = [with_sharding(4, &sharding.clone().with_unreduced_axes(["x"]).unwrap())],
                    error = "`parallel_all_gather` input must not carry reduction state over manual axis `x`",
                },
                {
                    input_types = [with_sharding(4, &sharding.with_reduced_axes(["x"]).unwrap())],
                    error = "`parallel_all_gather` input must not carry reduction state over manual axis `x`",
                },
            ],
        );

        // The composite family applies the same transition.
        assert_eq!(
            operation.infer_array_ir_output_types(&[
                with_sharding(4, &unrelated_pending_sum).into(),
                DimensionValue::constant(8).unwrap().r#type().into_owned().into(),
            ]),
            Ok(vec![with_sharding(8, &unrelated_pending_sum).into()]),
        );
    }

    #[test]
    fn test_parallel_all_gather_type_inference_array_ir() {
        // A tiled gather takes its result extent from its explicit extent input. A symbolic extent is taken as given,
        // while a static one must equal the static input extent multiplied by the axis group size.
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        );
        let input_axis = DimensionVariable::new("input", DimensionBounds::new(1, Some(17)).unwrap());
        let concatenation_result = DimensionVariable::new("concatenation", DimensionBounds::new(2, Some(33)).unwrap());
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(input_axis.clone()), Dimension::Static(3)]),
        );
        assert_eq!(
            operation.infer_array_ir_output_types(&[
                input_type.clone().into(),
                ArrayIrType::Dimension(DimensionType::from(concatenation_result.clone())),
                DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
            ]),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(concatenation_result.clone()), Dimension::Static(3)]),
                )
                .into()
            ]),
        );
        let exact_six = DimensionValue::constant(6).unwrap().r#type().into_owned();
        assert_eq!(
            operation
                .infer_array_ir_output_types(&[ArrayType::new_static(DataType::F32, [3]).into(), exact_six.into()]),
            Ok(vec![ArrayType::new_static(DataType::F32, [6]).into()]),
        );
        let exact_five = DimensionValue::constant(5).unwrap().r#type().into_owned();
        assert_eq!(
            operation
                .infer_array_ir_output_types(&[ArrayType::new_static(DataType::F32, [3]).into(), exact_five.into()]),
            Err(TypeError::invalid(
                "`parallel_all_gather` result extent must equal input axis 0 extent 3 multiplied by axis group size 2; \
                 expected 6 but got 5",
            )),
        );

        // The array input must be present, and a tiled gather concatenates into an existing axis.
        assert_eq!(
            operation.infer_array_ir_output_types(&[]),
            Err(TypeError::invalid("`parallel_all_gather` expects an array followed by its output extents")),
        );
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                1,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [3]).into(),
                DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
            ]),
            Err(TypeError::invalid("`parallel_all_gather` concatenation axis 1 is out of bounds for output rank 1")),
        );

        // An untiled gather inserts a participant axis whose explicit extent must equal the axis group size, while the
        // extents on either side of it must equal the unchanged input extents.
        let untiled = ParallelAllGatherOperation::new(
            "x".to_string(),
            4,
            1,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        );
        let extent =
            |value: usize| -> ArrayIrType { DimensionValue::constant(value).unwrap().r#type().into_owned().into() };
        assert_eq!(
            untiled.infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [2, 3]).into(),
                extent(2),
                extent(4),
                extent(3),
            ]),
            Ok(vec![ArrayType::new_static(DataType::F32, [2, 4, 3]).into()]),
        );
        assert_eq!(
            untiled.infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [2, 3]).into(),
                extent(2),
                extent(3),
                extent(3),
            ]),
            Err(TypeError::invalid(
                "`parallel_all_gather` inserted output axis 1 extent must equal axis group size 4 but got 3",
            )),
        );
        assert_eq!(
            untiled.infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [2, 3]).into(),
                extent(2),
                extent(4),
                extent(5),
            ]),
            Err(TypeError::invalid("`parallel_all_gather` output axis 2 extent 5 must equal unchanged extent 3")),
        );
        assert_eq!(
            untiled.infer_array_ir_output_types(&[ArrayType::scalar(DataType::F32).into(), extent(4)]),
            Err(TypeError::invalid("`parallel_all_gather` concatenation axis 1 is out of bounds for output rank 1")),
        );

        // A grouped gather multiplies the concatenation extent by the size of one participant group rather than the
        // axis size.
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                4,
                0,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
                ParallelAllGatherOutputVariance::Varying,
            )
            .infer_array_ir_output_types(&[ArrayType::new_static(DataType::F32, [3]).into(), extent(6)]),
            Ok(vec![ArrayType::new_static(DataType::F32, [6]).into()]),
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
            ]),
            Err(TypeError::invalid(
                "`parallel_all_gather` on a dimension sharded over explicit mesh axes requires the output size (3) \
                 at axis 0 to be divisible by the mesh-axis product (2)",
            )),
        );

        // A changed shape keeps the sharding and memory space of the input but drops its physical layout.
        assert_eq!(
            operation.infer_array_ir_output_types(&[
                input.clone().into(),
                DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
            ]),
            Ok(vec![
                ArrayType::new_static(DataType::F32, [4])
                    .with_sharding(input.sharding().unwrap().clone())
                    .unwrap()
                    .with_memory(Memory::Host { pinned: true })
                    .into(),
            ]),
        );

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
            ]),
            Ok(vec![input.into()]),
        );
    }

    #[test]
    fn test_parallel_all_gather_type_inference_array_ir_identity_instantiation() {
        // Staging through the composite capability derives the result extent from the input's symbolic extent.
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
                        concatenation_axis=0,
                        options=Tiled,
                        output_variance=Varying,
                        mesh=['devices'=2:manual],
                    ] %0 %3
                in (%4)"
            },
        );

        // Splicing the program instantiated at another input identity renames that identity on the input and on its
        // `dimension_size` read, while the gathered result keeps the derived `items * 2` extent.
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
                        concatenation_axis=0,
                        options=Tiled,
                        output_variance=Varying,
                        mesh=['devices'=2:manual],
                    ] %0 %3
                in (%4)"
            },
        );
    }

    #[test]
    fn test_parallel_all_gather_interpretation() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let input = Array::vector(vec![1.0, 2.0]).unwrap();

        // A single-participant axis is degenerate: a tiled gather concatenates exactly one input, so interpretation is
        // the identity, while an untiled gather inserts its size-one gathered axis.
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                1,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                1,
                0,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Ok(vec![Array::matrix(1, 2, vec![1.0, 2.0]).unwrap()]),
        );

        // Any larger axis has no per-item semantics: the other participants do not exist outside an enclosing binder.
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_all_gather` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_all_gather_interpretation_manual_mesh() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let input = Array::from_elements(input_type, &[1f32, 2.0]).unwrap();

        // Even a single manual participant requires its binder. Returning the input would silently keep its varying
        // metadata for invariant or reduced results, while a reshape cannot implement those variance transitions.
        for mode in [CollectiveMode::Untiled, CollectiveMode::Tiled] {
            for variance in [
                ParallelAllGatherOutputVariance::Varying,
                ParallelAllGatherOutputVariance::Invariant,
                ParallelAllGatherOutputVariance::Reduced,
            ] {
                let operation =
                    ParallelAllGatherOperation::new("x".to_string(), 1, 0, CollectiveOptions::new(mode), variance)
                        .with_mesh(mesh.clone());
                assert_eq!(
                    operation.interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
                    Err(ProgramError::UnsupportedOperation {
                        message: "cannot interpret `parallel_all_gather` over manual mesh axis `x` without an \
                                  enclosing binder"
                            .to_string(),
                    }),
                );
            }
        }
    }

    #[test]
    fn test_parallel_all_gather_interpretation_array_ir() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());

        // A single-participant gather returns its array member once its explicit result extent matches the observed
        // result extent.
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
            context.bind(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap())],
            ),
            Err(ProgramError::InvalidArgument {
                message: "`parallel_all_gather` output axis 0 extent must equal observed result extent 3 but got 4"
                    .to_string(),
            }),
        );

        // Any larger axis requires an enclosing binder.
        assert_eq!(
            context.bind(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(6).unwrap())],
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_all_gather` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_all_gather_interpretation_array_ir_manual_mesh() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let input = ArrayIrValue::Array(Array::from_elements(input_type, &[1f32, 2.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let participant_extent = ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap());

        // Explicit extents do not provide a manual execution binder or authorize dropping the output variance change.
        for mode in [CollectiveMode::Untiled, CollectiveMode::Tiled] {
            let inputs = match mode {
                CollectiveMode::Untiled => vec![input.clone(), participant_extent.clone(), extent.clone()],
                CollectiveMode::Tiled => vec![input.clone(), extent.clone()],
            };
            for variance in [
                ParallelAllGatherOutputVariance::Varying,
                ParallelAllGatherOutputVariance::Invariant,
                ParallelAllGatherOutputVariance::Reduced,
            ] {
                let operation =
                    ParallelAllGatherOperation::new("x".to_string(), 1, 0, CollectiveOptions::new(mode), variance)
                        .with_mesh(mesh.clone());
                assert_eq!(
                    context.bind(operation, Vec::new(), inputs.as_slice()),
                    Err(ProgramError::UnsupportedOperation {
                        message: "cannot interpret `parallel_all_gather` over manual mesh axis `x` without an \
                                  enclosing binder"
                            .to_string(),
                    }),
                );
            }
        }
    }

    #[test]
    fn test_parallel_all_gather_partial_evaluation() {
        // A single participant folds known inputs and residualizes unknown ones; untiled mode inserts a size-one axis.
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
            expected = input.clone(),
        );
        check_operation_partial_evaluation!(
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                1,
                1,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Varying,
            ),
            inputs = [input],
            expected = Array::matrix(2, 1, vec![1f32, 2.0]).unwrap(),
        );
    }

    #[test]
    fn test_parallel_all_gather_partial_evaluation_residualizes_known_input() {
        // A known input over a larger axis under an eager parent residualizes the operation, which needs the values of
        // the other participants, so the residual program is the source program itself.
        let input = Array::vector(vec![1f32, 2.0]).unwrap();
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        );
        let program = collective_program(operation, ArrayType::new_static(DataType::F32, [2]));
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input)]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[4] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concatenation_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ] %0
                in (%1)"
            },
        );
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
    }

    #[test]
    fn test_parallel_all_gather_partial_evaluation_manual_mesh() {
        // A known manual input remains residual because local execution cannot perform its variance transition.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let program = collective_program(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                1,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Invariant,
            )
            .with_mesh(mesh),
            input_type.clone(),
        );
        let input = Array::from_elements(input_type, &[1f32, 2.0]).unwrap();
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input)]).unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[2][sharding={mesh<['x'=1:manual]>, [{}]}] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=1,
                    concatenation_axis=0,
                    options=Tiled,
                    output_variance=Invariant,
                    mesh=['x'=1:manual],
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_all_gather_partial_evaluation_array_ir() {
        // Known inputs fold to the gathered array, while an unknown array input with a known extent stages the gather
        // as one residual instruction.
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
        // the mapped axis: every item receives the item-major concatenation of all items along `concatenation_axis`,
        // replicated across the batch, so the input of item `i` occupies the `i`-th chunk of that axis. With items
        // `[1, 2]` and `[3, 4]`, every item receives `[1, 2, 3, 4]`.
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    item.parallel_all_gather_tiled("x", 0)
                },
                Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::replicated(),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
        );

        // An untiled gather instead stacks the batch items along a new axis at `concatenation_axis`, here after each
        // item's only axis.
        assert_eq!(
            batch_collective(
                &ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    1,
                    CollectiveOptions::default(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                "x",
                2,
                ArrayBatch::new(Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap(),
            ),
            Ok(vec![ArrayBatch::replicated(Array::matrix(2, 2, vec![1.0f32, 3.0, 2.0, 4.0]).unwrap())]),
        );

        // A replicated input is first materialized as `axis_size` identical batch items, so the gather concatenates
        // that many copies of the shared value.
        assert_eq!(
            batch_collective(
                &ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                "x",
                2,
                ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]).unwrap()),
            ),
            Ok(vec![ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 1.0, 2.0]).unwrap())]),
        );
    }

    #[test]
    fn test_parallel_all_gather_batching_ragged() {
        // Untiled gathering co-moves bounded ragged metadata with the gathered value. The per-item extents of a mapped
        // input follow its batch axis onto the inserted participant axis, which becomes their extent axis.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input =
            ArrayBatch::new(Array::matrix(2, 3, vec![1.0f32, 0.0, 0.0, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(
                    1,
                    Array::vector(vec![1i32, 3]).unwrap(),
                    variable.clone(),
                    vec![0],
                )])
                .unwrap();
        assert_eq!(
            batch_collective(
                &ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::default(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                "x",
                2,
                input.clone(),
            ),
            Ok(vec![
                ArrayBatch::replicated(Array::matrix(2, 3, vec![1.0f32, 0.0, 0.0, 2.0, 3.0, 4.0]).unwrap())
                    .with_ragged_axes(vec![RaggedAxis::new(
                        1,
                        Array::vector(vec![1i32, 3]).unwrap(),
                        variable.clone(),
                        vec![0],
                    )])
                    .unwrap(),
            ]),
        );

        // Inserting the participant axis after the ragged axis moves both the data and the extent axis.
        assert_eq!(
            batch_collective(
                &ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    1,
                    CollectiveOptions::default(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                "x",
                2,
                input,
            ),
            Ok(vec![
                ArrayBatch::replicated(Array::matrix(3, 2, vec![1.0f32, 2.0, 0.0, 3.0, 0.0, 4.0]).unwrap())
                    .with_ragged_axes(vec![RaggedAxis::new(
                        0,
                        Array::vector(vec![1i32, 3]).unwrap(),
                        variable.clone(),
                        vec![1],
                    )])
                    .unwrap(),
            ]),
        );

        // A replicated input carries scalar extents, which the gather broadcasts along the participant axis.
        assert_eq!(
            batch_collective(
                &ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::default(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                "x",
                2,
                ArrayBatch::replicated(Array::vector(vec![1.0f32, 2.0, 0.0]).unwrap())
                    .with_ragged_axes(vec![RaggedAxis::new(
                        0,
                        Array::scalar(2i32).unwrap(),
                        variable.clone(),
                        Vec::new(),
                    )])
                    .unwrap(),
            ),
            Ok(vec![
                ArrayBatch::replicated(Array::matrix(2, 3, vec![1.0f32, 2.0, 0.0, 1.0, 2.0, 0.0]).unwrap())
                    .with_ragged_axes(vec![RaggedAxis::new(
                        1,
                        Array::vector(vec![2i32, 2]).unwrap(),
                        variable,
                        vec![0],
                    )])
                    .unwrap(),
            ]),
        );
    }

    #[test]
    fn test_parallel_all_gather_batching_rejects_unsupported_inputs() {
        let tiled = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        );
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let ragged = ArrayBatch::new(Array::matrix(2, 3, vec![1.0f32; 6]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1i32, 3]).unwrap(), variable, vec![0])])
            .unwrap();

        // Tiled gathering fuses the participant and concatenation axes, which can make live chunks non-prefix-shaped,
        // so a level that binds the axis rejects bounded ragged inputs.
        assert_eq!(
            batch_collective(&tiled, "x", 2, ragged.clone()),
            Err(BatchingError::UnsupportedOperation {
                message: "tiled `parallel_all_gather` cannot represent participant-specific bounded ragged extents \
                          after the participant and concatenation axes are fused"
                    .to_string(),
            }),
        );

        // A level that binds another axis rejects bounded ragged inputs instead of forwarding them to its parent.
        assert_eq!(
            batch_collective(&tiled, "y", 2, ragged),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_all_gather` does not support bounded ragged dimension `length` on input 0"
                    .to_string(),
            }),
        );

        // A level that binds the axis rejects participant groups.
        let grouped = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]),
            ParallelAllGatherOutputVariance::Varying,
        );
        assert_eq!(
            batch_collective(
                &grouped,
                "x",
                2,
                ArrayBatch::new(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap(),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_all_gather` axis index groups are not supported when a batch transform binds the \
                          collective axis"
                    .to_string(),
            }),
        );

        // An all-gather over a manual mesh axis cannot be consumed by a level that binds a batch axis with its name.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        assert_eq!(
            batch_collective(
                &tiled.with_mesh(mesh),
                "x",
                2,
                ArrayBatch::new(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap(),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_all_gather` over a manual mesh axis cannot bind a named batch axis".to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_all_gather_batching_forwards_unbound_axis() {
        // A level that binds another axis forwards the gather to its parent, shifting the concatenation axis past the
        // mapped axis, which keeps its position in the tiled result.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input =
            ArrayBatch::new(trace.input(ArrayType::new_static(DataType::F32, [3, 2])), BatchAxis::new(0)).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 3).with_axis_name("y".to_string());
        let outputs = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        )
        .batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input))
        .unwrap()
        .into_parts()
        .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
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
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3, 2] .
                let %1:f32[3, 4] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concatenation_axis=1,
                    options=Tiled,
                    output_variance=Varying,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_all_gather_batching_shadows_manual_axis() {
        // An inner named batch binds `x` independently of a manual mesh axis with the same name. The ordinary gather
        // merges the batch items into its concatenation axis and preserves the mesh invariance of its input.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.clone(), 2))
            .unwrap();
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| {
                let context = BatchingContext::<_, ArrayBatchingPolicy>::new(input.dispatch_domain(), 2)
                    .with_axis_name("x".to_string());
                let input = ArrayBatch::new(input, BatchAxis::new(0))?;
                let operation = ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                );
                let mut outputs = operation.batch(&context, &EmptyRegionDriver, &[input])?.into_parts().0;
                Ok(outputs.remove(0).into_value())
            },
            input_type,
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(
            output_type,
            ArrayType::new_static(DataType::F32, [6]).with_sharding(Sharding::replicated(mesh, 1)).unwrap(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}]}] .
                let %1:f32[6][sharding={mesh<['x'=2:manual]>, [{}]}] = \
                    reshape [shape=[6], output_sharding={mesh<['x'=2:manual]>, [{}]}] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_all_gather_batching_array_ir() {
        // The composite family's matching batching rule gathers the batch items exactly like the homogeneous one and
        // preserves the array member kind of its result.
        assert_eq!(
            batch(
                |item: BatchingTracer<
                    EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                    ArrayIrBatchingPolicy,
                >| { item.parallel_all_gather_tiled("x", 0) },
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
                BatchAxis::new(0),
                BatchAxis::replicated(),
                BatchAxisSpecification::named("x"),
            ),
            Ok(ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap())),
        );
    }

    #[test]
    fn test_parallel_all_gather_batching_array_ir_replicated_dynamic_extents() {
        // Matching-axis collective batching consumes a complete logical result shape. A replicated input is
        // materialized along the mapped axis from those extents, dynamic unchanged axes keep their boundary-provided
        // identity, and the rule introduces no metadata read from the source array.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(batch).into());
        let sequence = DimensionVariable::new("sequence", DimensionBounds::new(1, Some(17)).unwrap());
        let width = DimensionVariable::new("width", DimensionBounds::new(1, Some(33)).unwrap());
        let gathered = DimensionVariable::new("gathered", DimensionBounds::new(1, Some(65)).unwrap());
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
            )
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.batch().value().atom_id().unwrap()],
                vec![Placeholder; 4],
                vec![Placeholder],
            )
            .unwrap();
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
    }

    #[test]
    fn test_parallel_all_gather_batching_array_ir_ragged() {
        // A mapped ragged input carries its per-item extents as a mapped result extent, which an untiled gather moves
        // onto the participant axis.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = ArrayIrValue::Array(Array::vector(vec![1i32, 3]).unwrap());
        let extent =
            |value| ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(value).unwrap()));
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("x".to_string());
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        );
        assert_eq!(
            operation
                .batch_in_parent(
                    &context,
                    &EmptyRegionDriver,
                    &[
                        ArrayIrBatch::new(
                            ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 0.0, 0.0, 2.0, 3.0, 4.0]).unwrap()),
                            BatchAxis::new(0),
                        )
                        .unwrap()
                        .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
                        .unwrap(),
                        extent(2),
                        ArrayIrBatch::mapped_dimension(
                            extents.clone(),
                            BatchAxis::new(0),
                            DimensionType::from(variable.clone()),
                        )
                        .unwrap(),
                    ],
                )
                .unwrap()
                .into_parts()
                .0,
            vec![
                ArrayIrBatch::replicated(ArrayIrValue::Array(
                    Array::matrix(2, 3, vec![1.0f32, 0.0, 0.0, 2.0, 3.0, 4.0]).unwrap(),
                ))
                .with_ragged_axes(vec![RaggedAxis::new(1, extents, variable.clone(), vec![0])])
                .unwrap(),
            ],
        );

        // A replicated ragged input carries a replicated result extent of its bounded ragged dimension, and the gather
        // broadcasts its scalar extents along the participant axis.
        assert_eq!(
            operation
                .batch_in_parent(
                    &context,
                    &EmptyRegionDriver,
                    &[
                        ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 0.0]).unwrap()))
                            .with_ragged_axes(vec![RaggedAxis::new(
                                0,
                                ArrayIrValue::Array(Array::scalar(2i32).unwrap()),
                                variable.clone(),
                                Vec::new(),
                            )])
                            .unwrap(),
                        extent(2),
                        ArrayIrBatch::replicated(ArrayIrValue::Dimension(
                            DimensionValue::new(DimensionType::from(variable.clone()), 2).unwrap(),
                        )),
                    ],
                )
                .unwrap()
                .into_parts()
                .0,
            vec![
                ArrayIrBatch::replicated(ArrayIrValue::Array(
                    Array::matrix(2, 3, vec![1.0f32, 2.0, 0.0, 1.0, 2.0, 0.0]).unwrap(),
                ))
                .with_ragged_axes(vec![RaggedAxis::new(
                    1,
                    ArrayIrValue::Array(Array::vector(vec![2i32, 2]).unwrap()),
                    variable,
                    vec![0],
                )])
                .unwrap(),
            ],
        );
    }

    #[test]
    fn test_parallel_all_gather_batching_array_ir_rejects_unsupported_inputs() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = ArrayIrValue::Array(Array::vector(vec![1i32, 3]).unwrap());
        let ragged =
            ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f32; 6]).unwrap()), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
                .unwrap();
        let mapped_extent =
            ArrayIrBatch::mapped_dimension(extents, BatchAxis::new(0), DimensionType::from(variable.clone())).unwrap();
        let extent =
            |value| ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(value).unwrap()));
        let context = |axis_name: &str| {
            BatchingContext::new(
                EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
            )
            .with_axis_name(axis_name.to_string())
        };
        let untiled = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        );
        let tiled = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        );

        // Tiled gathering fuses the participant and concatenation axes, which can make live chunks non-prefix-shaped,
        // so a level that binds the axis rejects bounded ragged inputs.
        assert_eq!(
            tiled.batch_in_parent(&context("x"), &EmptyRegionDriver, &[ragged.clone(), extent(6)]),
            Err(BatchingError::UnsupportedOperation {
                message: "tiled `parallel_all_gather` cannot represent participant-specific bounded ragged extents \
                          after the participant and concatenation axes are fused"
                    .to_string(),
            }),
        );

        // A level that binds another axis rejects bounded ragged inputs instead of forwarding them to its parent.
        assert_eq!(
            untiled.batch_in_parent(&context("y"), &EmptyRegionDriver, &[ragged.clone(), extent(2), mapped_extent]),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_all_gather` does not support bounded ragged dimension `length` on input 0"
                    .to_string(),
            }),
        );

        // A mapped ragged input must supply its per-item extents through a mapped result extent.
        assert_eq!(
            untiled.batch_in_parent(
                &context("x"),
                &EmptyRegionDriver,
                &[
                    ragged,
                    extent(2),
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(
                        DimensionValue::new(DimensionType::from(variable), 3).unwrap(),
                    )),
                ],
            ),
            Err(BatchingError::InvalidBatchMetadata {
                message: "untiled `parallel_all_gather` output axis 1 must carry mapped extents for bounded ragged \
                          dimension `length`"
                    .to_string(),
            }),
        );

        // An all-gather over a manual mesh axis cannot be consumed by a level that binds a batch axis with its name.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            tiled.with_mesh(mesh).batch_in_parent(
                &context("x"),
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::replicated(ArrayIrValue::Array(
                        Array::from_elements(varying_type, &[1f32, 2.0]).unwrap(),
                    )),
                    extent(4),
                ],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_all_gather` over a manual mesh axis cannot bind a named batch axis".to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_all_gather_batching_array_ir_forwards_unbound_axis() {
        // A collective over a different named axis is forwarded as the same mixed operation. Only its physical axis
        // index and complete result shape are lifted around the current mapped axis, without reading the source shape.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(batch.clone()).into());
        let logical_extent = DimensionVariable::new("logical", DimensionBounds::new(1, Some(17)).unwrap());
        let result_extent = DimensionVariable::new("result", DimensionBounds::new(1, Some(33)).unwrap());
        let input = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(logical_extent), Dimension::Dynamic(batch), Dimension::Static(3)]),
            )
            .into(),
        );
        let result_extent = trace.input(DimensionType::from(result_extent).into());
        let width_extent = trace.input(DimensionValue::constant(3).unwrap().r#type().into_owned().into());
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
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(input, BatchAxis::new(1)).unwrap()),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(result_extent)),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(width_extent)),
                ],
            )
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(1));
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.batch().value().atom_id().unwrap()],
                vec![Placeholder; 4],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[logical, batch, 3], %2:dimension<result ∈ [1, 33)>, \
                    %3:dimension<3> .
                let %4:f32[result, batch, 3] = parallel_all_gather [
                    axis_name=\"inner\",
                    axis_size=2,
                    concatenation_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ] %1 %2 %0 %3
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_parallel_all_gather_differentiation() {
        // The collective is linear, so the tangent rides the same all-gather as the primal.
        let program = collective_program(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            ),
            ArrayType::new_static(DataType::F32, [2]),
        );
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:f32[4] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concatenation_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ] %0
                    %3:f32[4] = parallel_all_gather [
                        axis_name=\"x\",
                        axis_size=2,
                        concatenation_axis=0,
                        options=Tiled,
                        output_variance=Varying,
                    ] %1
                in (%2, %3)"
            },
        );
    }

    #[test]
    fn test_parallel_all_gather_differentiation_weighted_gradient() {
        // Distinct weights on the gathered chunks verify that the pullback routes the cotangent of every chunk back to
        // the batch item that contributed it, including when the physical batch axis follows the concatenation axis.
        check_gradient!(
            |inputs| {
                let gathered = batch(
                    |item| item.parallel_all_gather_tiled("x", 0),
                    inputs,
                    BatchAxis::new(1),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("x"),
                )?;
                let first = gathered.slice(&[0], &[2], &[1])?;
                let second = gathered.slice(&[2], &[4], &[1])?;
                let weighted = first + second.clone() + second;
                weighted.reduce(&[0], ReductionKind::Sum)
            },
            at = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );
        check_gradient!(
            |inputs| {
                let gathered = batch(
                    |item| item.parallel_all_gather("x", 1),
                    inputs,
                    BatchAxis::new(1),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("x"),
                )?;
                let first = gathered.slice(&[0, 0], &[2, 1], &[1, 1])?;
                let second = gathered.slice(&[0, 1], &[2, 2], &[1, 1])?;
                let weighted = first + second.clone() + second;
                weighted.reduce(&[0, 1], ReductionKind::Sum)
            },
            at = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_parallel_all_gather_differentiation_invariant() {
        // Inside a level that binds the axis, every item gathers the same complete value, and the pullback of the
        // invariant gather selects each item's own chunk of the cotangent, so the gradient of `sum(g * g)` is `2 * x`
        // in both modes.
        let inputs = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    differentiate_at(item)
                        .gradient(|item| {
                            let gathered = item.parallel_all_gather_with_options(
                                "x",
                                0,
                                CollectiveOptions::tiled(),
                                ParallelAllGatherOutputVariance::Invariant,
                            )?;
                            gathered.mul(&gathered)?.reduce(&[0], ReductionKind::Sum)
                        })
                        .map_err(ProgramError::from)
                },
                inputs.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::matrix(2, 2, vec![2.0, 4.0, 6.0, 8.0]).unwrap()),
        );
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    differentiate_at(item)
                        .gradient(|item| {
                            let gathered = item.parallel_all_gather_with_options(
                                "x",
                                1,
                                CollectiveOptions::default(),
                                ParallelAllGatherOutputVariance::Invariant,
                            )?;
                            gathered.mul(&gathered)?.reduce(&[0, 1], ReductionKind::Sum)
                        })
                        .map_err(ProgramError::from)
                },
                inputs,
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::matrix(2, 2, vec![2.0, 4.0, 6.0, 8.0]).unwrap()),
        );
    }

    #[test]
    fn test_parallel_all_gather_differentiation_array_ir() {
        // The composite JVP of a varying gather stages one linear call that retains the explicit result extent as its
        // only residual, applies the same gather to the tangent, and transposes to the matching sum-scatter, while the
        // structurally zero extent tangent stages nothing.
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
                    concatenation_axis=0,
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
                    concatenation_axis=0,
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
                            concatenation_axis=0,
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
                            concatenation_axis=0,
                            options=Tiled,
                            output_variance=Varying,
                        ] %1 %0
                        in (%2)
                    },
                ]
                in (%2)"
            },
        );

        // Over a single participant, the JVP passes both the primal and the tangent through unchanged, and the
        // pullback returns the output cotangent unchanged, including for an empty dynamic extent.
        let primal = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let tangent = ArrayIrValue::Array(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap());
        assert_eq!(
            program.jvp().unwrap().interpret(vec![primal.clone(), result_extent.clone(), tangent.clone()]),
            Ok(vec![primal, tangent]),
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
    fn test_parallel_all_gather_differentiation_array_ir_invariant() {
        // A degenerate invariant gather linearizes into a linear call whose transpose selects the only participant's
        // chunk at offset zero instead of staging a sum-scatter.
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
                    concatenation_axis=0,
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
                    concatenation_axis=0,
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
                            concatenation_axis=0,
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
                            concatenation_axis=0,
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

        // The linearized pullback of a nondegenerate untiled invariant gather selects the current participant's
        // size-one slice and reshapes away the participant axis.
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
        // The mixed boundary delegates its static array contribution to the homogeneous all-gather rule, which selects
        // the same participant's row of the output cotangent.
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:dimension<2> = const 2
                    %2:dimension<3> = const 3
                    %3:u64[] = axis_index [axis_name=\"x\"]
                    %4:u64[] = zero [type=u64[]]
                    %5:f32[1, 3] = dynamic_slice [sizes=[1, 3]] %0 %3 %4
                    %6:f32[3] = reshape [shape=[3]] %5
                in (%6)"
            },
        );
        assert_eq!(
            program.linearize().unwrap().pullback().unwrap().to_string(),
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
                                concatenation_axis=0,
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
                            concatenation_axis=0,
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
        // Outside any binder, the pullback of a degenerate all-gather returns its only participant's chunk, both
        // through the adjoint sum-scatter of a varying result and through the local selection of an invariant one.
        check_operation_transposition!(
            @exact,
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                1,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            ),
            cases = [{
                inputs = [(@linear(type = ArrayType::new_static(DataType::F64, [2])))],
                output_cotangents = [Array::vector(vec![3.0f64, 5.0]).unwrap()],
                input_cotangents = [Array::vector(vec![3.0f64, 5.0]).unwrap()],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = ParallelAllGatherOperation::new(
                "x".to_string(),
                1,
                1,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Invariant,
            ),
            cases = [{
                inputs = [(@linear(type = ArrayType::new_static(DataType::F64, [2])))],
                output_cotangents = [Array::matrix(2, 1, vec![3.0f64, 5.0]).unwrap()],
                input_cotangents = [Array::vector(vec![3.0f64, 5.0]).unwrap()],
            }],
        );

        // A tiled all-gather is the adjoint of a sum-scatter over the same axis and dimension, so the pullback stages
        // a `parallel_sum_scatter` on the output cotangent with the gather's concatenation axis as its scatter axis.
        let program = collective_program(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying,
            ),
            ArrayType::new_static(DataType::F32, [2]),
        );
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

        // The adjoint sum-scatter keeps the participant groups of the gather.
        let program = collective_program(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                4,
                0,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
                ParallelAllGatherOutputVariance::Varying,
            ),
            ArrayType::new_static(DataType::F32, [2]),
        );
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[2] = parallel_sum_scatter [
                    axis_name=\"x\",
                    axis_size=4,
                    scatter_axis=0,
                    options=Tiled,
                    axis_index_groups=[[0, 2], [3, 1]],
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
        let program = collective_program(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Reduced,
            )
            .with_mesh(mesh.clone()),
            input_type.clone(),
        );
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

        // A varying result over a manual mesh axis transposes to the sum-scatter over the same mesh, which keeps the
        // untiled mode and scatter axis of the gather and returns the varying input cotangent.
        let program = collective_program(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                1,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .with_mesh(mesh),
            input_type.clone(),
        );
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[2, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                let %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_sum_scatter \
                [axis_name=\"x\", axis_size=2, scatter_axis=1, options=Untiled, mesh=['x'=2:manual]] %0
                in (%1)"
            },
        );
        assert_eq!(pullback.output_types(), vec![input_type.cotangent().unwrap()]);

        // An invariant result transposes without communication: a tiled gather splits the participant axis off its
        // output cotangent, which is then sliced at the current participant's index and reshaped back to the input.
        let program = collective_program(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Invariant,
            ),
            ArrayType::new_static(DataType::F32, [2]),
        );
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[2, 2] = reshape [shape=[2, 2]] %0
                    %2:u64[] = axis_index [axis_name=\"x\"]
                    %3:u64[] = zero [type=u64[]]
                    %4:f32[1, 2] = dynamic_slice [sizes=[1, 2]] %1 %2 %3
                    %5:f32[2] = reshape [shape=[2]] %4
                in (%5)"
            },
        );

        // An invariant result has no adjoint collective, because its transpose selects a chunk locally instead.
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Invariant,
            )
            .adjoint(&ArrayType::new_static(DataType::F32, [2])),
            Err(ProgramError::UnsupportedOperation {
                message: "invariant `parallel_all_gather` has no adjoint collective because its transpose selects the \
                          current participant's chunk"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_all_gather_transposition_invariant_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();

        // Over a manual mesh axis, the invariant output cotangent is first made varying, because every participant
        // selects a different chunk of it, and the selected chunk has the varying input cotangent type.
        let program = collective_program(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Invariant,
            )
            .with_mesh(mesh.clone()),
            input_type.clone(),
        );
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_vary [axis_name=\"x\"] %0
                    %2:f32[2, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = reshape [
                        shape=[2, 2],
                        output_sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}},
                    ] %1
                    %3:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
                    axis_index [axis_name=\"x\", mesh=['x'=2:manual]]
                    %4:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
                    zero [type=u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]]
                    %5:f32[1, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                    dynamic_slice [sizes=[1, 2]] %2 %3 %4
                    %6:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                    reshape [shape=[2], output_sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] %5
                in (%6)"
            },
        );
        assert_eq!(pullback.output_types(), vec![input_type.cotangent().unwrap()]);

        // The start indices of an invariant transpose must vary over the same manual axes as the cotangent that they
        // slice. Over a manual mesh axis, the participant index varies only over the gathered axis, so it is also made
        // varying over the independent axis `y` of an input that varies over both axes.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x", "y"]).unwrap())
            .unwrap();
        let program = collective_program(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Invariant,
            )
            .with_mesh(mesh.clone()),
            input_type.clone(),
        );
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'y'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] = \
                    parallel_vary [axis_name=\"x\"] %0
                    %2:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x', 'y'}}] = \
                    reshape [
                        shape=[2, 2],
                        output_sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x', 'y'}},
                    ] %1
                    %3:u64[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'x'}}] = \
                    axis_index [axis_name=\"x\", mesh=['x'=2:manual, 'y'=2:manual]]
                    %4:u64[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'x', 'y'}}] = \
                    parallel_vary [axis_name=\"y\"] %3
                    %5:u64[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'x', 'y'}}] = zero [
                        type=u64[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'x', 'y'}}],
                    ]
                    %6:f32[1, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x', 'y'}}] = \
                    dynamic_slice [sizes=[1, 2]] %2 %4 %5
                    %7:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] = \
                    reshape [
                        shape=[2],
                        output_sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}},
                    ] %6
                in (%7)"
            },
        );
        assert_eq!(pullback.output_types(), vec![input_type.cotangent().unwrap()]);

        // An ordinary gather whose input varies over an enclosing manual mesh axis places the unsharded participant
        // index of its binder on that mesh before making it vary over that axis.
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["y"]).unwrap())
            .unwrap();
        let program = collective_program(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Invariant,
            ),
            input_type.clone(),
        );
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'y'}}] .
                let %1:u64[] = axis_index [axis_name=\"x\"]
                    %2:u64[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, []}] = broadcast [
                        output_type=u64[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, []}],
                        output_axes=[],
                    ] %1
                    %3:u64[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'y'}}] = \
                    parallel_vary [axis_name=\"y\"] %2
                    %4:u64[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'y'}}] = zero [
                        type=u64[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'y'}}],
                    ]
                    %5:f32[1, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'y'}}] = \
                    dynamic_slice [sizes=[1, 2]] %0 %3 %4
                    %6:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'y'}}] = reshape [
                        shape=[2],
                        output_sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'y'}},
                    ] %5
                in (%6)"
            },
        );
        assert_eq!(pullback.output_types(), vec![input_type.cotangent().unwrap()]);

        // A single manual participant selects its only chunk with a zero start index that varies like the cotangent.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let program = collective_program(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                1,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Invariant,
            )
            .with_mesh(mesh),
            input_type.clone(),
        );
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:manual]>, [{}]}] .
                let %1:f32[2][sharding={mesh<['x'=1:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_vary [axis_name=\"x\"] %0
                    %2:f32[1, 2][sharding={mesh<['x'=1:manual]>, [{}, {}], varying_manual={'x'}}] = reshape [
                        shape=[1, 2],
                        output_sharding={mesh<['x'=1:manual]>, [{}, {}], varying_manual={'x'}},
                    ] %1
                    %3:u64[][sharding={mesh<['x'=1:manual]>, [], varying_manual={'x'}}] = \
                    zero [type=u64[][sharding={mesh<['x'=1:manual]>, [], varying_manual={'x'}}]]
                    %4:f32[1, 2][sharding={mesh<['x'=1:manual]>, [{}, {}], varying_manual={'x'}}] = \
                    dynamic_slice [sizes=[1, 2]] %2 %3 %3
                    %5:f32[2][sharding={mesh<['x'=1:manual]>, [{}], varying_manual={'x'}}] = \
                    reshape [shape=[2], output_sharding={mesh<['x'=1:manual]>, [{}], varying_manual={'x'}}] %4
                in (%5)"
            },
        );
        assert_eq!(pullback.output_types(), vec![input_type.cotangent().unwrap()]);
    }

    #[test]
    fn test_parallel_all_gather_transposition_array_ir_rejects_dynamic_extent() {
        // Direct transposition cannot recover a runtime-dependent result extent, so the composite family requires
        // linearization, which retains that extent as a residual.
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
                    requires linearization so that the relevant primal information can be retained as residuals",
        ));
    }

    #[test]
    fn test_parallel_all_gather_parallel_all_gather() {
        // Gathering over a `batch` level stacks the batch items along a new axis, so every batch item receives all
        // items.
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
            .unwrap()),
        );
    }

    #[test]
    fn test_parallel_all_gather_parallel_all_gather_tiled() {
        // Inside a manual region, homogeneous array values stage the static-shape operation without explicit result
        // extents, and an invariant input is made varying first so that every copy is gathered.
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
                        concatenation_axis=0,
                        options=Tiled,
                        output_variance=Varying,
                        mesh=['x'=2:manual],
                    ] %1
                in (%2)"
            },
        );
    }

    #[test]
    fn test_parallel_all_gather_parallel_all_gather_tiled_preserves_unrelated_pending_sums() {
        // The composite capability makes an invariant input varying over the gathered axis `x` without dropping a
        // pending sum over the independent `y` axis.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["y"]).unwrap();
        let (output, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_all_gather_tiled("x", 0),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap()),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        assert_eq!(
            output,
            ArrayIrType::Array(
                ArrayType::new_static(DataType::F32, [8])
                    .with_sharding(sharding.with_varying_manual_axes(["x"]).unwrap())
                    .unwrap(),
            ),
        );
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
                        concatenation_axis=0,
                        options=Tiled,
                        output_variance=Varying,
                        mesh=['x'=2:manual, 'y'=2:manual],
                    ] %1 %4
                in (%5)"
            },
        );
    }

    #[test]
    fn test_parallel_all_gather_parallel_all_gather_with_options() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let varying_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(sharding.clone().with_varying_manual_axes(["x"]).unwrap())
            .unwrap();

        // Invariant output variance stages a gather over the manual mesh axis whose result no longer varies over it.
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| {
                input.parallel_all_gather_with_options(
                    "x",
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Invariant,
                )
            },
            varying_type.clone(),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::new_static(DataType::F32, [6]).with_sharding(sharding.clone()).unwrap());
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[6][sharding={mesh<['x'=2:manual]>, [{}]}] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concatenation_axis=0,
                    options=Tiled,
                    output_variance=Invariant,
                    mesh=['x'=2:manual],
                ] %0
                in (%1)"
            },
        );

        // Reduced output variance records the gathered manual mesh axis as reduced, which requires a manual mesh axis.
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| {
                input.parallel_all_gather_with_options(
                    "x",
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Reduced,
                )
            },
            varying_type.clone(),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(
            output_type,
            ArrayType::new_static(DataType::F32, [6])
                .with_sharding(sharding.clone().with_reduced_axes(["x"]).unwrap())
                .unwrap(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[6][sharding={mesh<['x'=2:manual]>, [{}], reduced={'x'}}] = parallel_all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concatenation_axis=0,
                    options=Tiled,
                    output_variance=Reduced,
                    mesh=['x'=2:manual],
                ] %0
                in (%1)"
            },
        );
        assert!(matches!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| {
                    input.parallel_all_gather_with_options(
                        "x",
                        0,
                        CollectiveOptions::tiled(),
                        ParallelAllGatherOutputVariance::Reduced,
                    )
                },
                ArrayType::new_static(DataType::F32, [3]),
                vec![("x".to_string(), NamedAxis::Batched { size: Some(2) })],
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`parallel_all_gather` with reduced output variance requires a manual mesh axis",
        ));

        // Participant groups are rejected with invariant or reduced output variance before anything is staged.
        assert!(matches!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| {
                    input.parallel_all_gather_with_options(
                        "x",
                        0,
                        CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]),
                        ParallelAllGatherOutputVariance::Invariant,
                    )
                },
                varying_type.clone(),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`parallel_all_gather` axis index groups are not supported with invariant or reduced \
                    output variance",
        ));

        // A pending or completed sum over the manual mesh axis cannot be gathered, and both value families report it
        // as an all-gather error before staging the `parallel_vary` transition that would also reject it.
        for reduction_state in
            [sharding.clone().with_unreduced_axes(["x"]).unwrap(), sharding.clone().with_reduced_axes(["x"]).unwrap()]
        {
            let input_type = ArrayType::new_static(DataType::F32, [3]).with_sharding(reduction_state).unwrap();
            assert!(matches!(
                TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                    |input| {
                        input.parallel_all_gather_with_options(
                            "x",
                            0,
                            CollectiveOptions::tiled(),
                            ParallelAllGatherOutputVariance::Varying,
                        )
                    },
                    input_type.clone(),
                    vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
                ),
                Err(ProgramError::Type(TypeError::Invalid { message }))
                    if message == "`parallel_all_gather` input must not carry reduction state over manual axis `x`",
            ));
            assert!(matches!(
                TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                    |input| {
                        input.parallel_all_gather_with_options(
                            "x",
                            0,
                            CollectiveOptions::tiled(),
                            ParallelAllGatherOutputVariance::Varying,
                        )
                    },
                    ArrayIrType::Array(input_type),
                    vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
                ),
                Err(ProgramError::Type(TypeError::Invalid { message }))
                    if message == "`parallel_all_gather` input must not carry reduction state over manual axis `x`",
            ));
        }

        // Composite values reject participant groups with invariant output variance before anything is staged too.
        assert!(matches!(
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| {
                    input.parallel_all_gather_with_options(
                        "x",
                        0,
                        CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]),
                        ParallelAllGatherOutputVariance::Invariant,
                    )
                },
                ArrayIrType::Array(varying_type.clone()),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`parallel_all_gather` axis index groups are not supported with invariant or reduced \
                    output variance",
        ));

        // Composite values normalize the concatenation axis before they derive the explicit result extents, and a tiled
        // gather concatenates along an existing axis, so its axis must be smaller than the rank of the input.
        assert!(matches!(
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| {
                    input.parallel_all_gather_with_options(
                        "x",
                        1,
                        CollectiveOptions::tiled(),
                        ParallelAllGatherOutputVariance::Varying,
                    )
                },
                ArrayIrType::Array(varying_type),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
            ),
            Err(ProgramError::Axis(AxisError::OutOfBounds { axis, rank: 1 })) if axis == Axis::from(1),
        ));
    }

    #[test]
    fn test_parallel_all_gather_parallel_all_gather_with_options_negative_axes() {
        // Negative concatenation axes count from the end of the result. An untiled gather inserts an axis, so its
        // result has one more axis than the input and `-1` appends a new trailing axis, while a tiled gather
        // concatenates along an existing axis. Homogeneous and composite values normalize the axis alike.
        let input_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let named_axes = || vec![("x".to_string(), NamedAxis::Batched { size: Some(2) })];
        let homogeneous = |concatenation_axis: i32, options: CollectiveOptions| {
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                move |input| {
                    input.parallel_all_gather_with_options(
                        "x",
                        concatenation_axis,
                        options,
                        ParallelAllGatherOutputVariance::Varying,
                    )
                },
                input_type.clone(),
                named_axes(),
            )
            .map(|(_, program)| program.to_string())
        };
        let composite = |concatenation_axis: i32, options: CollectiveOptions| {
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                move |input| {
                    input.parallel_all_gather_with_options(
                        "x",
                        concatenation_axis,
                        options,
                        ParallelAllGatherOutputVariance::Varying,
                    )
                },
                ArrayIrType::Array(input_type.clone()),
                named_axes(),
            )
            .map(|(_, program)| program.to_string())
        };
        let untiled = CollectiveOptions::default;
        let tiled = CollectiveOptions::tiled;
        assert_eq!(homogeneous(-1, untiled()).unwrap(), homogeneous(2, untiled()).unwrap());
        assert_eq!(homogeneous(-3, untiled()).unwrap(), homogeneous(0, untiled()).unwrap());
        assert_eq!(homogeneous(-1, tiled()).unwrap(), homogeneous(1, tiled()).unwrap());
        assert_eq!(composite(-1, untiled()).unwrap(), composite(2, untiled()).unwrap());
        assert_eq!(composite(-2, tiled()).unwrap(), composite(0, tiled()).unwrap());
        assert_eq!(
            homogeneous(-4, untiled()),
            Err(ProgramError::Axis(AxisError::OutOfBounds { axis: Axis::from(-4), rank: 3 })),
        );
        assert_eq!(
            composite(-3, tiled()),
            Err(ProgramError::Axis(AxisError::OutOfBounds { axis: Axis::from(-3), rank: 2 })),
        );
    }

    #[test]
    fn test_parallel_all_gather_parallel_all_gather_with_options_unbound_axis() {
        // Axis-size resolution fails at staging time with `AxisError::UnboundAxisName` rather than silently acting as
        // the identity. Here, the batch binds only the axis `"i"`, while the gather names `"x"`.
        assert_eq!(
            batch(
                |item: BatchingTracer<
                    EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                    ArrayIrBatchingPolicy,
                >| { item.parallel_all_gather_tiled("x", 0) },
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
                BatchAxis::new(0),
                BatchAxis::replicated(),
                BatchAxisSpecification::named("i"),
            ),
            Err(BatchingError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
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
    fn test_parallel_all_gather_parallel_all_gather_with_options_projected_value() {
        // A projected array view of a composite value stages the all-gather through its composite value, so that the
        // output extent is staged as an explicit extent value.
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| {
                let array = ValueProjection::<ArrayType>::into_projected(input)?;
                Ok(array
                    .parallel_all_gather_with_options(
                        "x",
                        0,
                        CollectiveOptions::tiled(),
                        ParallelAllGatherOutputVariance::Varying,
                    )?
                    .into_value())
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
                    %3:dimension<8> = dimension_mul %1 %2
                    %4:f32[8] = parallel_all_gather [
                        axis_name=\"x\",
                        axis_size=2,
                        concatenation_axis=0,
                        options=Tiled,
                        output_variance=Varying,
                    ] %0 %3
                in (%4)"
            },
        );
    }
}
