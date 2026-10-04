//! Contains the named-axis [`ParallelAllToAllOperation`], which exchanges chunks between the participants along a named
//! axis, together with its interpretation, partial-evaluation, batching, forward-mode differentiation, and
//! transposition rules.

// TODO(eaplatanios): Review this module.

use std::fmt::Display;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType, ArrayType, Dimension,
    DimensionOperation, DimensionType, DimensionValue, DimensionVariable, LogicalMesh, Shape, Sharding,
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
use crate::operations::arithmetic::{AddOperation, Div, Mul, Rem};
use crate::operations::assertions::Assert;
use crate::operations::collectives::parallel_vary::{PARALLEL_VARY_OPERATION_NAME, ParallelVary};
use crate::operations::collectives::{
    CollectiveArrayExtentBatchingPolicy, CollectiveExtent, CollectiveMode, CollectiveOptions,
    LinearCollectiveOperation, ShapeChangingCollectiveKernel, ShapeChangingCollectiveOperation,
    check_manual_mesh_input, collective_input_extents, infer_array_ir_shape_changing_collective_output_type,
    infer_linear_collective_operation_output_type, resolve_named_axis_size,
};
use crate::operations::comparisons::Compare;
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_max::DimensionMax;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{DynamicBroadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, MemberOperation, Operation, OperationFormatter, OperationProjection, ProgramError, ProjectedValue,
    RegionInterface, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`ParallelAllToAllOperation`].
pub const PARALLEL_ALL_TO_ALL_OPERATION_NAME: &str = "parallel_all_to_all";

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
/// An exchange over a manual mesh axis is created by [`with_mesh`](Self::with_mesh), and
/// [`ParallelAllToAll::parallel_all_to_all_with_options`] supplies the mesh automatically from the enclosing manual
/// region, making an invariant input varying first. Such an exchange can give the receivers different values, so
/// its input must vary over the axis (refer to [`ParallelVary`]) and its output varies over it too. A pending sum
/// over that axis is rejected; sums over unrelated manual axes are preserved. An ordinary exchange carries no mesh
/// and preserves the input's mesh variation and pending sums, even when its input carries a manual mesh axis with
/// the same name, because a `batch` level whose axis name shadows that mesh axis may bind it instead. Type
/// inference in the homogeneous array family requires static extents; the composite array/dimension family uses
/// explicit result extents, with runtime assertions for dynamic split divisibility and untiled split size.
///
/// A matching `batch` level consumes the named axis of an ordinary exchange with a local reshape/transpose block
/// exchange. Batch item `i` receives every item's chunk `i`, in sender order. A replicated input is broadcast
/// before the exchange, since receivers can still get different chunks. Participant groups and exchanges over a
/// manual mesh axis are unsupported at a matching level. Outside any binder, a single-participant tiled exchange is
/// the identity; untiled mode relocates its size-one split axis to the concatenation position.
///
/// Bounded ragged inputs are rejected. One extent per item does not determine how each sender partitions its live
/// prefix among receivers; that requires the explicit offsets and per-destination sizes of
/// [`ParallelRaggedAllToAllOperation`](crate::operations::collectives::ParallelRaggedAllToAllOperation).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelAllToAllOperation {
    /// Axis name referenced by this collective.
    axis_name: String,

    /// Number of participants along the named axis, resolved from the active [`NamedAxes`] environment
    /// when the operation is staged.
    axis_size: usize,

    /// Axis of the input that is split into one chunk per participant.
    split_axis: usize,

    /// Axis of the output along which the received chunks are concatenated.
    concat_axis: usize,

    /// Shared rank and participant-group semantics.
    options: CollectiveOptions,

    /// Refer to the documentation of [`mesh`](Self::mesh) for more information.
    mesh: Option<LogicalMesh>,
}

impl ParallelAllToAllOperation {
    /// Creates a new [`ParallelAllToAllOperation`] over the axis with the provided name and resolved axis size.
    #[inline]
    pub fn new(
        axis_name: String,
        axis_size: usize,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Self {
        Self { axis_name, axis_size, split_axis, concat_axis, options, mesh: None }
    }

    /// Returns this [`ParallelAllToAllOperation`] configured to exchange chunks over a manual axis of `mesh`. The input
    /// must vary over [`axis_name`](Self::axis_name) on that mesh, whose size must equal
    /// [`axis_size`](Self::axis_size), and must not carry a pending cross-device sum over that axis. Sums over
    /// unrelated manual axes are preserved. Type inference validates these requirements.
    /// [`ParallelAllToAll::parallel_all_to_all_with_options`] supplies the mesh automatically from the enclosing manual
    /// region.
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

    /// Returns the logical mesh whose manual axis this [`ParallelAllToAllOperation`] exchanges chunks over, or
    /// [`None`] for an ordinary exchange, whose named axis may be bound by any enclosing binder. Only an exchange over
    /// a manual mesh axis validates the manual variation and pending sums of its input.
    #[inline]
    pub fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

    /// Validates the manual variation and pending sums of an exchange over a manual mesh axis, given the shape-only
    /// `output_type` shared by the static and array IR inference paths. An ordinary exchange preserves its input's mesh
    /// state, including pending sums, because a `batch` level that binds its axis performs only local array
    /// rearrangement, even when its axis name shadows a manual mesh axis.
    fn finalize_output_type(&self, input_type: &ArrayType, output_type: ArrayType) -> Result<ArrayType, TypeError> {
        let Some(mesh) = &self.mesh else {
            return Ok(output_type);
        };

        let axis_name = self.axis_name();
        let sharding = check_manual_mesh_input(
            PARALLEL_ALL_TO_ALL_OPERATION_NAME,
            axis_name,
            Some(self.axis_size),
            mesh,
            input_type,
        )?;

        // Exchanging chunks cannot complete a pending sum over the same axis. Sums over unrelated axes commute with the
        // exchange and retain their pending state.
        if sharding.unreduced_axes().contains(axis_name) {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_ALL_TO_ALL_OPERATION_NAME}` does not support unreduced inputs",
            )));
        }

        if !sharding.varying_manual_axes().contains(axis_name) {
            return Err(TypeError::invalid(format!(
                "`{}` input must vary over manual axis `{}`; pass an invariant value through `{}` \
                 first so that the exchanged output is typed as varying",
                PARALLEL_ALL_TO_ALL_OPERATION_NAME, axis_name, PARALLEL_VARY_OPERATION_NAME,
            )));
        }

        Ok(output_type)
    }
}

impl LinearCollectiveOperation for ParallelAllToAllOperation {
    type Adjoint = ParallelAllToAllOperation;

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
        self.options.effective_axis_size(PARALLEL_ALL_TO_ALL_OPERATION_NAME, self.axis_size)
    }

    fn adjoint(&self, _input_type: &ArrayType) -> Result<ParallelAllToAllOperation, ProgramError> {
        // The chunk exchange is its own adjoint with the split and concatenation axes swapped, over the same axis,
        // participant groups, and mesh.
        Ok(ParallelAllToAllOperation { split_axis: self.concat_axis, concat_axis: self.split_axis, ..self.clone() })
    }

    #[inline]
    fn forwarded(&self, batch_axis: usize) -> (Self, usize) {
        let (split_axis, output_batch_axis) = self.options.mode.forwarded_split_axes(self.split_axis, batch_axis);
        let (concat_axis, output_batch_axis) =
            self.options.mode.forwarded_concatenation_axes(self.concat_axis, output_batch_axis);
        (Self { split_axis, concat_axis, ..self.clone() }, output_batch_axis)
    }
}

impl ShapeChangingCollectiveOperation for ParallelAllToAllOperation {
    #[inline]
    fn options(&self) -> &CollectiveOptions {
        &self.options
    }

    fn infer_array_ir_output_types(&self, input_types: &[ArrayIrType]) -> Result<Vec<ArrayIrType>, TypeError> {
        let effective_axis_size = self.effective_axis_size()?;
        let Some(input_type) = input_types.first() else {
            return Err(TypeError::invalid("`parallel_all_to_all` expects an array followed by its output extents"));
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        if self.options.mode == CollectiveMode::Untiled {
            let Some(input_extent) = input_type.shape().dimensions().get(self.split_axis) else {
                return Err(TypeError::invalid(format!(
                    "`parallel_all_to_all` split axis {} is out of bounds for rank {}",
                    self.split_axis,
                    input_type.rank(),
                )));
            };
            if let Dimension::Static(input_extent) = input_extent
                && *input_extent != effective_axis_size
            {
                return Err(TypeError::invalid(format!(
                    "`parallel_all_to_all` untiled split axis {} size {input_extent} must equal group size \
                     {effective_axis_size}",
                    self.split_axis,
                )));
            }
            let output_type = input_type
                .without_dimension(self.split_axis)?
                .0
                .with_inserted_dimension(self.concat_axis, Dimension::Static(effective_axis_size))?;
            let mut output_types = infer_array_ir_shape_changing_collective_output_type(
                PARALLEL_ALL_TO_ALL_OPERATION_NAME,
                input_types,
                output_type,
                &[self.concat_axis],
                |output_extents| {
                    let output_extent = &output_extents[self.concat_axis];
                    if output_extent != &Dimension::Static(effective_axis_size) {
                        return Err(TypeError::invalid(format!(
                            "`parallel_all_to_all` inserted output axis {} extent must equal axis group size \
                             {effective_axis_size} but got {output_extent}",
                            self.concat_axis,
                        )));
                    }
                    Ok(())
                },
            )?;
            let output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
            return Ok(vec![self.finalize_output_type(input_type, output_type)?.into()]);
        }
        if self.split_axis == self.concat_axis {
            let Some(input_extent) = input_type.shape().dimensions().get(self.split_axis) else {
                return Err(TypeError::invalid(format!(
                    "`parallel_all_to_all` split axis {} is out of bounds for rank {}",
                    self.split_axis,
                    input_type.rank(),
                )));
            };
            if let Dimension::Static(input_extent) = input_extent
                && *input_extent % effective_axis_size != 0
            {
                return Err(TypeError::invalid(format!(
                    "`parallel_all_to_all` split axis {} size {input_extent} is not divisible by group size \
                     {effective_axis_size}",
                    self.split_axis,
                )));
            }
            let mut output_types = infer_array_ir_shape_changing_collective_output_type(
                PARALLEL_ALL_TO_ALL_OPERATION_NAME,
                input_types,
                input_type.clone(),
                &[],
                |_| Ok(()),
            )?;
            let output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
            return Ok(vec![self.finalize_output_type(input_type, output_type)?.into()]);
        }
        if self.split_axis >= input_type.rank() || self.concat_axis >= input_type.rank() {
            return Err(TypeError::invalid(format!(
                "`parallel_all_to_all` split axis {} or concat axis {} is out of bounds for rank {}",
                self.split_axis,
                self.concat_axis,
                input_type.rank(),
            )));
        }
        if let Dimension::Static(input_extent) = &input_type.shape().dimensions()[self.split_axis]
            && *input_extent % effective_axis_size != 0
        {
            return Err(TypeError::invalid(format!(
                "`parallel_all_to_all` split axis {} size {input_extent} is not divisible by group size \
                 {effective_axis_size}",
                self.split_axis,
            )));
        }
        let expected_concat_extent = match input_type.shape().dimensions()[self.concat_axis] {
            Dimension::Static(input_extent) => {
                Some(input_extent.checked_mul(effective_axis_size).ok_or_else(|| {
                    TypeError::invalid("`parallel_all_to_all` concatenation result extent does not fit in usize")
                })?)
            }
            Dimension::Dynamic(_) => None,
        };
        let mut dimensions = input_type.shape().dimensions().to_vec();
        dimensions[self.split_axis] = Dimension::Static(0);
        dimensions[self.concat_axis] = Dimension::Static(0);
        let sharding = input_type.resized_sharding(dimensions.as_slice(), PARALLEL_ALL_TO_ALL_OPERATION_NAME)?;
        let mut base_output_type =
            ArrayType::new(input_type.data_type(), Shape::new(dimensions)).with_memory(input_type.memory());
        base_output_type.sharding = sharding;
        let mut output_types = infer_array_ir_shape_changing_collective_output_type(
            PARALLEL_ALL_TO_ALL_OPERATION_NAME,
            input_types,
            base_output_type,
            &[self.split_axis, self.concat_axis],
            |output_extents| {
                if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                    (&input_type.shape().dimensions()[self.split_axis], &output_extents[self.split_axis])
                {
                    let expected = *input_extent / effective_axis_size;
                    if *output_extent != expected {
                        return Err(TypeError::invalid(format!(
                            "`parallel_all_to_all` split result extent must equal input axis {} extent {input_extent} \
                             divided by group size {effective_axis_size}; expected {expected} but got {output_extent}",
                            self.split_axis,
                        )));
                    }
                }
                if let (Some(expected), Dimension::Static(output_extent)) =
                    (expected_concat_extent, &output_extents[self.concat_axis])
                {
                    let input_extent = &input_type.shape().dimensions()[self.concat_axis];
                    if *output_extent != expected {
                        return Err(TypeError::invalid(format!(
                            "`parallel_all_to_all` concat result extent must equal input axis {} extent {input_extent} \
                             multiplied by group size {effective_axis_size}; expected {expected} but got \
                             {output_extent}",
                            self.concat_axis,
                        )));
                    }
                }
                Ok(())
            },
        )?;
        let mut output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
        // Placeholder zeros cannot establish the sharding constraints of the actual result dimensions.
        output_type.sharding =
            input_type.resized_sharding(output_type.shape().dimensions(), PARALLEL_ALL_TO_ALL_OPERATION_NAME)?;
        if output_type.shape() == input_type.shape() {
            output_type = output_type.with_layout(input_type.layout().cloned());
        }
        Ok(vec![self.finalize_output_type(input_type, output_type)?.into()])
    }

    fn ragged_input_error(&self, dimension: &DimensionVariable, _input_index: usize) -> BatchingError {
        // One extent per item does not determine how each sender partitions its live prefix among the receivers, which
        // requires the explicit offsets and per-destination sizes of a ragged all-to-all instead.
        BatchingError::UnsupportedOperation {
            message: format!(
                "`parallel_all_to_all` cannot route bounded ragged dimension `{dimension}` without explicit \
                 per-destination offsets and sizes; use `parallel_ragged_all_to_all`",
            ),
        }
    }
}

impl<C: Context<Type = ArrayType, Value: Transpose>> ShapeChangingCollectiveKernel<C> for ParallelAllToAllOperation {
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
                message: "`parallel_all_to_all` axis index groups are not supported when a batch transform binds the \
                     collective axis"
                    .to_string(),
            });
        }
        if self.split_axis >= logical_input_rank || self.concat_axis >= logical_input_rank {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`parallel_all_to_all` split axis {} or concat axis {} is out of bounds for rank \
                     {logical_input_rank}",
                    self.split_axis, self.concat_axis,
                ),
            });
        }

        let axis_extent =
            P::collective_axis_extent(context, PARALLEL_ALL_TO_ALL_OPERATION_NAME, &self.axis_name, self.axis_size)?;

        let (input_extents, chunk_extent) = match self.options.mode {
            CollectiveMode::Untiled => {
                let mut input_extents = output_extents.clone();
                input_extents.remove(self.concat_axis);
                input_extents.insert(self.split_axis, axis_extent.clone());
                (input_extents, P::collective_extent_constant(context, 1)?)
            }
            CollectiveMode::Tiled if self.split_axis == self.concat_axis => {
                let axis_extent =
                    P::require_divisible_collective_extents(context, &output_extents[self.split_axis], &axis_extent)?;
                (output_extents.clone(), output_extents[self.split_axis].div(&axis_extent)?)
            }
            CollectiveMode::Tiled => {
                let axis_extent =
                    P::require_divisible_collective_extents(context, &output_extents[self.concat_axis], &axis_extent)?;
                let mut input_extents = output_extents.clone();
                input_extents[self.split_axis] = output_extents[self.split_axis].mul(&axis_extent)?;
                input_extents[self.concat_axis] = output_extents[self.concat_axis].div(&axis_extent)?;
                (input_extents, output_extents[self.split_axis].clone())
            }
        };
        let input = P::match_collective_axis(context, input, input_extents.as_slice())?;
        let mut split_extents = Vec::with_capacity(input_extents.len() + 2);
        split_extents.push(axis_extent.clone());
        split_extents.extend(input_extents.iter().cloned());
        split_extents[self.split_axis + 1] = axis_extent.clone();
        split_extents.insert(self.split_axis + 2, chunk_extent);
        let split = P::reshape_collective(context, input.into_value(), split_extents.as_slice(), None)?;
        let exchanged = split.swap_axes(0, self.split_axis + 1)?;
        let received = match self.options.mode {
            CollectiveMode::Untiled => {
                let mut squeezed_extents = Vec::with_capacity(input_extents.len() + 1);
                squeezed_extents.push(axis_extent.clone());
                squeezed_extents.extend(input_extents);
                P::reshape_collective(context, exchanged, squeezed_extents.as_slice(), None)?
                    .move_axis(self.split_axis + 1, self.concat_axis + 1)?
            }
            CollectiveMode::Tiled => exchanged.move_axis(self.split_axis + 1, self.concat_axis + 1)?,
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
}

impl Display for ParallelAllToAllOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ParallelAllToAllOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        PARALLEL_ALL_TO_ALL_OPERATION_NAME
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
                "`{PARALLEL_ALL_TO_ALL_OPERATION_NAME}` does not support dynamically shaped inputs",
            )));
        };

        let dimensions = shape.dimensions().to_vec();
        let effective_axis_size = self.effective_axis_size()?;
        let mut output_dimensions = dimensions;
        let rank = output_dimensions.len();
        if self.split_axis >= rank || self.concat_axis >= rank {
            return Err(TypeError::invalid(format!(
                "`parallel_all_to_all` split axis {} or concat axis {} is out of bounds for rank {rank}",
                self.split_axis, self.concat_axis,
            )));
        }
        let output_type = if self.options.mode == CollectiveMode::Untiled {
            if output_dimensions[self.split_axis] != effective_axis_size {
                return Err(TypeError::invalid(format!(
                    "`parallel_all_to_all` untiled split axis {} size {} must equal group size {}",
                    self.split_axis, output_dimensions[self.split_axis], effective_axis_size,
                )));
            }
            input_type
                .without_dimension(self.split_axis)?
                .0
                .with_inserted_dimension(self.concat_axis, Dimension::Static(effective_axis_size))?
        } else {
            if output_dimensions[self.split_axis] % effective_axis_size != 0 {
                return Err(TypeError::invalid(format!(
                    "`parallel_all_to_all` split axis {} size {} is not divisible by group size {}",
                    self.split_axis, output_dimensions[self.split_axis], effective_axis_size,
                )));
            }
            output_dimensions[self.split_axis] /= effective_axis_size;
            output_dimensions[self.concat_axis] =
                output_dimensions[self.concat_axis].checked_mul(effective_axis_size).ok_or_else(|| {
                    TypeError::invalid(
                        "`parallel_all_to_all` concatenation result extent does not fit in usize".to_string(),
                    )
                })?;
            infer_linear_collective_operation_output_type(
                PARALLEL_ALL_TO_ALL_OPERATION_NAME,
                input_type,
                output_dimensions,
            )?
        };
        Ok(vec![self.finalize_output_type(input_type, output_type)?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, PARALLEL_ALL_TO_ALL_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
            operation.field("axis_size", self.axis_size)?;
            operation.field("split_axis", format_args!("{:?}", &self.split_axis))?;
            operation.field("concat_axis", format_args!("{:?}", &self.concat_axis))?;
            operation.field("options", format_args!("{:?}", &self.options))?;
            if let Some(mesh) = &self.mesh {
                operation.field("mesh", mesh)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Reshape>> InterpretableOperation<C> for ParallelAllToAllOperation {
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

        // A single participant exchanges chunks only with itself. Untiled mode removes the size-one split axis and
        // inserts a size-one concatenation axis, which a reshape to the inferred output type expresses, while tiled
        // mode leaves the shape unchanged.
        Ok(vec![match self.options.mode {
            CollectiveMode::Tiled => input.clone(),
            CollectiveMode::Untiled => {
                input.reshape_with_output_sharding(output_type.shape().clone(), output_type.sharding().cloned())?
            }
        }])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelAllToAllOperation>>> PartiallyEvaluatableOperation<C>
    for ParallelAllToAllOperation
{
}

// Batching rule for [`ParallelAllToAllOperation`]. A matching `batch` level consumes the mapped batch axis with a
// reshape/transpose block exchange: the per-item `split_axis` is split into `(b, d_p / b)` chunks, the chunk axis is
// swapped with the leading batch axis (so the batch axis indexes the *receiving* item), and the sender axis is then
// merged item-major into the per-item `concat_axis` — batch item `i` receives every item's chunk `i`, concatenated
// along `concat_axis`. A non-matching level forwards the collective to the parent context, unchanged for a replicated
// input (through `BatchingContext::forward_to_parent`) and with its array axes shifted past the batch axis for a mapped
// one.
impl<
    C: Context<Type = ArrayType, Value: Transpose, Operation: From<ParallelAllToAllOperation>>,
    P: CollectiveArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for ParallelAllToAllOperation
{
    #[inline]
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        self.shape_changing_collective_batch(context, inputs)
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelAllToAllOperation>>> DifferentiableOperation<C>
    for ParallelAllToAllOperation
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
    O: Operation<Type = ArrayType> + From<AddOperation<ArrayType>> + From<ParallelAllToAllOperation>,
> TransposableOperation<V, O> for ParallelAllToAllOperation
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

impl MemberOperation<ArrayIrType> for ParallelAllToAllOperation {
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
> MemberInterpretableOperation<C> for ParallelAllToAllOperation
{
    #[inline]
    fn interpret_in_parent<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        self.shape_changing_collective_interpret::<C>(inputs)
    }
}

// Batching rule for array IR [`ParallelAllToAllOperation`]. Dimension SSA supplies its temporary split and merge
// shapes directly, while matching-axis array mechanics reuse the homogeneous collective kernel.
impl<
    C: Context<
            Type = ArrayIrType,
            Value: Assert
                       + DimensionSize
                       + DynamicBroadcast
                       + ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>
                       + ValueProjection<
                DimensionType,
                Projected: Compare<C::Value> + DimensionMax + Rem + Div + Mul + Value<Type = DimensionType>,
            >,
            Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Operation: From<ParallelAllToAllOperation>
                           + From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
> MemberBatchableOperation<C, ArrayIrBatchingPolicy> for ParallelAllToAllOperation
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

impl ParallelAllToAll for Array {
    // A concrete `Array` never executes inside an axis binder, because the values under a `batch` level or inside
    // a manual region are tracers, so every axis name is unbound for it.

    #[inline]
    fn parallel_all_to_all_with_options(
        &self,
        axis_name: &str,
        _split_axis: usize,
        _concat_axis: usize,
        _options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        Err(AxisError::UnboundAxisName { name: axis_name.to_string() }.into())
    }
}

// Mixed array IR JVP for all-to-all. Explicit output extents are retained as ordinary residual values, and the
// transposed linear region swaps the split and concatenation axes.
impl<C> MemberDifferentiableOperation<C> for ParallelAllToAllOperation
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<ParallelAllToAllOperation>
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
        self.shape_changing_collective_jvp(context, inputs)
    }
}

/// Represents the ability to exchange chunks between participants of a named axis by staging an
/// [`ParallelAllToAllOperation`]. Refer to that operation for the tiling, grouping, variation, and transformation
/// semantics. Dynamic result extents are staged as first-class dimension values, and runtime assertions validate
/// dynamic split extents.
///
/// # Example
///
/// Each row sends its first half to batch item zero and its second half to batch item one:
///
/// ```
/// # use ryft_core::operations::collectives::ParallelAllToAll;
/// # use ryft_core::{
/// #     Array, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, BatchAxis, BatchAxisSpecification,
/// #     BatchingTracer, EagerContext, batch,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rows = ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?);
/// let received = batch(
///     |row: BatchingTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, ArrayIrBatchingPolicy>| {
///         row.parallel_all_to_all_tiled("rows", 0, 0)
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
pub trait ParallelAllToAll: Sized {
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
    /// Returns the errors of [`ParallelAllToAll::parallel_all_to_all_with_options`].
    #[inline]
    fn parallel_all_to_all(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
    ) -> Result<Self, ProgramError> {
        self.parallel_all_to_all_with_options(axis_name, split_axis, concat_axis, CollectiveOptions::default())
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
    /// Returns the errors of [`ParallelAllToAll::parallel_all_to_all_with_options`].
    #[inline]
    fn parallel_all_to_all_tiled(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
    ) -> Result<Self, ProgramError> {
        self.parallel_all_to_all_with_options(
            axis_name,
            split_axis,
            concat_axis,
            CollectiveOptions::new(CollectiveMode::Tiled),
        )
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
    fn parallel_all_to_all_with_options(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError>;
}

// Composite values stage the array followed by one result extent per axis. Only a manual mesh binder records its mesh
// on the operation and introduces variation; a named batch that shadows the same mesh-axis name performs its own local
// exchange.
impl<V> ParallelAllToAll for V
where
    V: Value<Type = ArrayIrType>
        + Assert
        + DimensionSize<V>
        + ValueProjection<DimensionType>
        + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
    V::DispatchDomain: Context<Type = ArrayIrType> + NamedAxes,
    V::DispatchDomain: DimensionConstant,
    <V::DispatchDomain as Domain>::Operation: From<ParallelAllToAllOperation>,
    <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + Compare<V> + Rem + Div + Mul,
    ProjectedValue<ArrayType, V>: ParallelVary,
{
    fn parallel_all_to_all_with_options(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let effective_axis_size = options.effective_axis_size(PARALLEL_ALL_TO_ALL_OPERATION_NAME, axis_size)?;
        let mut input = self.clone();
        let mut operation =
            ParallelAllToAllOperation::new(axis_name.to_string(), axis_size, split_axis, concat_axis, options.clone());
        if let Some(NamedAxis::Mesh { mesh, .. }) = context.named_axis(axis_name) {
            let array = ValueProjection::<ArrayType>::into_projected(self.clone())?;
            if array.r#type().unreduced_axes().contains(axis_name) {
                return Err(TypeError::invalid("`parallel_all_to_all` does not support unreduced inputs").into());
            }
            if !array.r#type().sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis_name)) {
                input = <V as ValueProjection<ArrayType>>::from_projected(array.parallel_vary(axis_name)?);
            }
            operation = operation.with_mesh(mesh);
        }
        let mut output_extents = collective_input_extents(&input)?;
        let rank = output_extents.len();
        if split_axis >= rank || concat_axis >= rank {
            return Err(TypeError::invalid(format!(
                "`parallel_all_to_all` split axis {split_axis} or concat axis {concat_axis} is out of bounds for rank \
                 {rank}",
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

impl<V> ParallelAllToAll for ProjectedValue<ArrayType, V>
where
    V: ParallelAllToAll + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
{
    fn parallel_all_to_all_with_options(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        self.value()
            .parallel_all_to_all_with_options(axis_name, split_axis, concat_axis, options)?
            .into_projected()
            .map_err(Into::into)
    }
}

/// Convenience untiled all-to-all that exchanges one ranked array axis with a named axis.
pub trait ParallelSwapAxes: ParallelAllToAll {
    /// Swaps `axis` with `axis_name` over the full named axis. The ranked axis must have the participant count as its
    /// extent. This is [`ParallelAllToAll::parallel_all_to_all`] with identical split and concatenation positions.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name bound by an enclosing batch level or manual region.
    ///   - `axis`: Ranked axis to exchange with the named axis.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`ParallelAllToAll::parallel_all_to_all_with_options`].
    #[inline]
    fn parallel_swap_axes(&self, axis_name: &str, axis: usize) -> Result<Self, ProgramError> {
        self.parallel_all_to_all(axis_name, axis, axis)
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
    /// Returns the errors of [`ParallelAllToAll::parallel_all_to_all_with_options`].
    #[inline]
    fn parallel_swap_axes_with_axis_index_groups(
        &self,
        axis_name: &str,
        axis: usize,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError> {
        self.parallel_all_to_all_with_options(
            axis_name,
            axis,
            axis,
            CollectiveOptions::default().with_axis_index_groups(axis_index_groups),
        )
    }
}

impl<V: ParallelAllToAll> ParallelSwapAxes for V {}

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
    fn parallel_all_to_all_program(
        operation: ParallelAllToAllOperation,
        input_type: ArrayType,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type);
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap().to_vec();
        builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Applies the matching-axis rule under an eager two-participant batch named `"x"`.
    fn batch_parallel_all_to_all(
        operation: &ParallelAllToAllOperation,
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
    fn test_parallel_all_to_all() {
        let operation = ParallelAllToAllOperation::new("x".to_string(), 4, 0, 1, CollectiveOptions::tiled());
        assert_eq!(operation.name(), PARALLEL_ALL_TO_ALL_OPERATION_NAME);
        assert_eq!(operation.split_axis(), 0);
        assert_eq!(operation.concat_axis(), 1);
        assert_eq!(operation.options(), &CollectiveOptions::tiled());
        assert_eq!(operation.mesh(), None);
        assert_eq!(operation.effective_axis_size(), Ok(4));

        // An exchange over a manual mesh axis records and renders its mesh.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let mesh_operation = operation.clone().with_mesh(mesh.clone());
        assert_eq!(mesh_operation.mesh(), Some(&mesh));
        assert_eq!(
            mesh_operation.to_string(),
            indoc! {"
                parallel_all_to_all [
                    axis_name=\"x\",
                    axis_size=4,
                    split_axis=0,
                    concat_axis=1,
                    options=Tiled,
                    mesh=['x'=4:manual],
                ]"
            },
        );
        assert_ne!(mesh_operation, operation);

        let grouped = ParallelAllToAllOperation::new(
            "x".to_string(),
            4,
            1,
            0,
            CollectiveOptions::default().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
        );
        assert_eq!(grouped.effective_axis_size(), Ok(2));
    }

    #[test]
    fn test_parallel_all_to_all_type_inference() {
        check_operation_type_inference!(
            operation = ParallelAllToAllOperation::new("x".to_string(), 4, 0, 1, CollectiveOptions::tiled()),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [8, 3])],
                    output_types = [ArrayType::new_static(DataType::F32, [2, 12])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [6, 3])],
                    error = "`parallel_all_to_all` split axis 0 size 6 is not divisible by group size 4",
                },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelAllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::default()),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::Boolean, [3, 2])],
                    output_types = [ArrayType::new_static(DataType::Boolean, [2, 3])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [3, 4])],
                    error = "`parallel_all_to_all` untiled split axis 1 size 4 must equal group size 2",
                },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelAllToAllOperation::new("x".to_string(), 4, 0, 1,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]])),
            cases = [{
                input_types = [ArrayType::new_static(DataType::C64, [6, 3])],
                output_types = [ArrayType::new_static(DataType::C64, [3, 6])],
            }],
        );
    }

    #[test]
    fn test_parallel_all_to_all_array_ir_type_inference() {
        let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
        let split = DimensionVariable::new("split", DimensionBounds::unbounded());
        let concat = DimensionVariable::new("concat", DimensionBounds::unbounded());
        assert_eq!(
            operation.infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [4, 3]).into(),
                DimensionType::from(split.clone()).into(),
                DimensionType::from(concat.clone()).into(),
            ]),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![split.clone().into(), concat.clone().into()])).into()
            ]),
        );

        // A known invalid split extent is rejected even when both result extents are dynamic.
        assert_eq!(
            operation.infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [3, 3]).into(),
                DimensionType::from(split.clone()).into(),
                DimensionType::from(concat.clone()).into(),
            ]),
            Err(TypeError::invalid("`parallel_all_to_all` split axis 0 size 3 is not divisible by group size 2")),
        );
        assert_eq!(
            operation.infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [4, usize::MAX]).into(),
                DimensionType::from(split).into(),
                DimensionType::from(concat).into(),
            ]),
            Err(TypeError::invalid("`parallel_all_to_all` concatenation result extent does not fit in usize")),
        );
        assert_eq!(
            operation.infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [4, 3]).into(),
                DimensionValue::constant(1).unwrap().r#type().into_owned().into(),
                DimensionValue::constant(6).unwrap().r#type().into_owned().into(),
            ]),
            Err(TypeError::invalid(
                "`parallel_all_to_all` split result extent must equal input axis 0 extent 4 divided by group size 2; \
                 expected 2 but got 1",
            )),
        );
        assert_eq!(
            operation.infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [4, 3]).into(),
                DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                DimensionValue::constant(5).unwrap().r#type().into_owned().into(),
            ]),
            Err(TypeError::invalid(
                "`parallel_all_to_all` concat result extent must equal input axis 1 extent 3 multiplied by group size \
                 2; expected 6 but got 5",
            )),
        );

        // Untiled geometry retains unaffected dynamic dimensions and inserts a statically known sender axis.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::default())
                .infer_array_ir_output_types(&[
                    ArrayType::new(DataType::F32, Shape::new(vec![length.clone().into(), 2.into()])).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionType::from(length.clone()).into(),
                ]),
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![2.into(), length.into()])).into()]),
        );
    }

    #[test]
    fn test_parallel_all_to_all_type_inference_metadata() {
        let input = ArrayType::new_static(DataType::F32, [4, 3])
            .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])))
            .with_memory(Memory::Host { pinned: true });
        let operation = ParallelAllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::tiled());
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input), &[]), Ok(vec![input.clone()]));
        assert_eq!(
            operation.infer_array_ir_output_types(&[
                input.clone().into(),
                DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
            ]),
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
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled())
                .infer_array_ir_output_types(&[
                    input.into(),
                    DimensionValue::constant(1).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(6).unwrap().r#type().into_owned().into(),
                ]),
            Err(TypeError::invalid(
                "`parallel_all_to_all` on a dimension sharded over explicit mesh axes requires the output size (1) at \
                 axis 0 to be divisible by the mesh-axis product (2)",
            )),
        );

        // An exchange over a manual mesh axis keeps a varying input varying, but rejects an invariant input, which
        // would wrongly type the exchanged output as invariant, and an input with a pending cross-device sum. The
        // input must carry the operation's mesh.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 2);
        let varying = sharding.clone().with_varying_manual_axes(["x"]).unwrap();
        let unreduced = sharding.clone().with_unreduced_axes(["x"]).unwrap();
        let with_sharding = |dimensions: [usize; 2], sharding: &Sharding| {
            ArrayType::new_static(DataType::F32, dimensions).with_sharding(sharding.clone()).unwrap()
        };
        let other_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
        check_operation_type_inference!(
            operation = operation.clone().with_mesh(mesh.clone()),
            cases = [
                { input_types = [with_sharding([2, 3], &varying)], output_types = [with_sharding([1, 6], &varying)] },
                {
                    input_types = [with_sharding([2, 3], &sharding)],
                    error = "`parallel_all_to_all` input must vary over manual axis `x`; pass an invariant value \
                             through `parallel_vary` first so that the exchanged output is typed as varying",
                },
                {
                    input_types = [with_sharding([2, 3], &unreduced)],
                    error = "`parallel_all_to_all` does not support unreduced inputs",
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2, 3])],
                    error = "`parallel_all_to_all` input must carry a mesh containing manual axis `x`",
                },
                {
                    input_types = [with_sharding(
                        [2, 3],
                        &Sharding::replicated(other_mesh, 2).with_varying_manual_axes(["x"]).unwrap(),
                    )],
                    error = "`parallel_all_to_all` input mesh does not match the operation mesh",
                },
            ],
        );
        assert_eq!(
            operation.clone().with_mesh(mesh.clone()).infer_array_ir_output_types(&[
                with_sharding([2, 3], &unreduced).into(),
                DimensionValue::constant(1).unwrap().r#type().into_owned().into(),
                DimensionValue::constant(6).unwrap().r#type().into_owned().into(),
            ]),
            Err(TypeError::invalid("`parallel_all_to_all` does not support unreduced inputs")),
        );

        // The mesh axis must be manual, and its size must equal the axis size of the operation.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = operation.clone().with_mesh(explicit_mesh),
            cases = [{
                input_types = [with_sharding([2, 3], &varying)],
                error = "`parallel_all_to_all` mesh axis `x` must be manual",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelAllToAllOperation::new("x".to_string(), 4, 0, 1, CollectiveOptions::tiled())
                .with_mesh(mesh),
            cases = [{
                input_types = [with_sharding([4, 3], &varying)],
                error = "`parallel_all_to_all` axis size 4 does not match the size of manual mesh axis `x`",
            }],
        );

        // An ordinary exchange preserves the mesh state of its input, including invariance and pending sums over a
        // manual mesh axis with the same name, because a `batch` level that shadows that mesh axis may bind it.
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                { input_types = [with_sharding([2, 3], &sharding)], output_types = [with_sharding([1, 6], &sharding)] },
                {
                    input_types = [with_sharding([2, 3], &unreduced)],
                    output_types = [with_sharding([1, 6], &unreduced)],
                },
            ],
        );
    }

    #[test]
    fn test_parallel_all_to_all_preserves_unrelated_pending_sums() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled())
            .with_mesh(mesh.clone());
        for pending_axis in ["x", "y"] {
            let sharding = Sharding::replicated(mesh.clone(), 2).with_unreduced_axes([pending_axis]).unwrap();
            let invariant = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(sharding.clone()).unwrap();
            let sharding =
                if pending_axis == "y" { sharding.with_varying_manual_axes(["x"]).unwrap() } else { sharding };
            let input_type = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(sharding.clone()).unwrap();
            let output_type = ArrayType::new_static(DataType::F32, [1, 6]).with_sharding(sharding).unwrap();
            let expected = if pending_axis == "y" {
                Ok(vec![output_type])
            } else {
                Err(TypeError::invalid("`parallel_all_to_all` does not support unreduced inputs"))
            };
            // Exchanging chunks along `x` preserves a sum over independent `y` in both type representations.
            assert_eq!(operation.infer_output_types(&[input_type.clone()], &[]), expected);
            assert_eq!(
                operation.infer_array_ir_output_types(&[
                    input_type.into(),
                    DimensionValue::constant(1).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(6).unwrap().r#type().into_owned().into(),
                ]),
                expected.clone().map(|outputs| outputs.into_iter().map(ArrayIrType::Array).collect()),
            );
            // The public capability also makes an invariant input varying over `x` without dropping pending `y`.
            assert_eq!(
                TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                    |input| input.parallel_all_to_all_tiled("x", 0, 1),
                    ArrayIrType::Array(invariant),
                    vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
                )
                .map(|(output, _)| output),
                expected.map(|mut outputs| ArrayIrType::Array(outputs.remove(0))).map_err(ProgramError::Type),
            );
        }
    }

    #[test]
    fn test_parallel_all_to_all_interpretation() {
        // A single participant exchanges chunks only with itself, so tiled mode is the identity, while untiled mode
        // removes the size-one split axis and inserts a size-one concatenation axis. Any larger axis has no per-item
        // semantics outside an enclosing binder.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let input = Array::matrix(1, 2, vec![1.0, 2.0]).unwrap();
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::tiled()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::default()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![Array::matrix(2, 1, vec![1.0, 2.0]).unwrap()]),
        );
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_all_to_all` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_all_to_all_partial_evaluation() {
        // A degenerate untiled exchange folds a known input through the corresponding rank-preserving reshape.
        let input = Array::matrix(1, 3, vec![1.0f32, 2.0, 3.0]).unwrap();
        let program = parallel_all_to_all_program(
            ParallelAllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::default()),
            ArrayType::new_static(DataType::F32, [1, 3]),
        );
        assert_eq!(
            program.partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap().outputs(),
            &[PartialEvaluationOutput::Known(Array::matrix(3, 1, vec![1.0f32, 2.0, 3.0]).unwrap())],
        );

        // An exchange with other participants has no eager per-item value and therefore residualizes.
        let operation = ParallelAllToAllOperation::new("x".to_string(), 3, 1, 0, CollectiveOptions::tiled());
        let program = parallel_all_to_all_program(operation.clone(), ArrayType::new_static(DataType::F32, [1, 3]));
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
    fn test_parallel_all_to_all_batching() {
        // Replicated senders still route destination-specific chunks: each destination receives its chunk twice.
        let tiled = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled());
        let output =
            batch_parallel_all_to_all(&tiled, ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()))
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
        let output = batch_parallel_all_to_all(
            &ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::default()),
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
        let output = batch_parallel_all_to_all(
            &ParallelAllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::default()),
            ArrayBatch::new(input, BatchAxis::new(2)).unwrap(),
        )
        .unwrap()
        .remove(0);
        assert_eq!(output.value().r#type().as_ref(), &ArrayType::new_static(DataType::F64, [2, 2, 3]));
        assert_eq!(output.value().to_f64s(), vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0]);

        // Participant groups belong to a mesh exchange and are unsupported when this batch binds the named axis.
        assert_eq!(
            batch_parallel_all_to_all(
                &ParallelAllToAllOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    0,
                    CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]])
                ),
                ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]).unwrap()),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_all_to_all` axis index groups are not supported when a batch transform binds the \
                     collective axis"
                    .to_string(),
            }),
        );

        // An exchange over a manual mesh axis cannot be consumed by a level that binds a batch axis with its name.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        assert_eq!(
            batch_parallel_all_to_all(
                &tiled.with_mesh(mesh),
                ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]).unwrap()),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_all_to_all` over a manual mesh axis cannot bind a named batch axis".to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_all_to_all_batching_same_axes() {
        // Block exchange with `split_axis == concat_axis == 0`: each item splits its vector into two chunks and
        // receives its own chunk index from every item, concatenated item-major. With items `[1, 2, 3, 4]` and
        // `[5, 6, 7, 8]`, item 0 receives `[1, 2, 5, 6]` and item 1 receives `[3, 4, 7, 8]`, matching the verified
        // cross-device `shard_map` execution semantics of StableHLO's `all_to_all`.
        let x = Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let output: ArrayIrValue<Array> = batch(
            |item: BatchingTracer<
                EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                ArrayIrBatchingPolicy,
            >| { item.parallel_all_to_all_tiled("x", 0, 0) },
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
            panic!("`parallel_all_to_all` must preserve the array member kind");
        };
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_parallel_all_to_all_batching_distinct_axes() {
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
            >| { item.parallel_all_to_all_tiled("x", 0, 1) },
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
            panic!("`parallel_all_to_all` must preserve the array member kind");
        };
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_parallel_all_to_all_batching_forwards_untiled_axes() {
        // A non-matching batch level shifts both array axes around its mapped dimension. Removing the split axis
        // can move that mapped dimension, and insertion at the concat axis can move it a second time.
        let context = BatchingContext::<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>::new(
            EagerContext::new(),
            2,
        )
        .with_axis_name("items".to_string());
        let operation = ParallelAllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::default());
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
    fn test_parallel_all_to_all_batching_shadows_manual_axis() {
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
                    let operation =
                        ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
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
                            |item| item.parallel_all_to_all_tiled("x", 0, 1),
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
    fn test_parallel_all_to_all_batching_shadows_manual_axis_through_unrelated_batch() {
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
                                    |item| item.parallel_all_to_all_tiled("x", 0, 1),
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
    fn test_parallel_all_to_all_batching_forwards_manual_axis_validation() {
        // An exchange over a manual mesh axis forwarded through an unrelated batch level rejects invariant inputs and
        // pending mesh sums at that level, before forwarding. Bind the operation directly so the capability cannot
        // insert `parallel_vary` first.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 3);
        for (sharding, expected_message) in [
            (
                sharding.clone(),
                "`parallel_all_to_all` input must vary over manual axis `x`; pass an invariant value through \
                 `parallel_vary` first so that the exchanged output is typed as varying",
            ),
            (sharding.with_unreduced_axes(["x"]).unwrap(), "`parallel_all_to_all` does not support unreduced inputs"),
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
                    let operation =
                        ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled())
                            .with_mesh(mesh.clone());
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
            assert_eq!(
                error.downcast_custom::<BatchingError>(),
                Some(&BatchingError::Type(TypeError::invalid(expected_message))),
            );
        }
    }

    #[test]
    fn test_parallel_all_to_all_batching_ragged() {
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
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled()).batch(
                &context,
                &EmptyRegionDriver,
                &[input],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_all_to_all` cannot route bounded ragged dimension `length` without explicit \
                          per-destination offsets and sizes; use `parallel_ragged_all_to_all`"
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
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled()).batch_in_parent(
                &context,
                &EmptyRegionDriver,
                &[input, output_extent],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_all_to_all` cannot route bounded ragged dimension `length` without explicit \
                          per-destination offsets and sizes; use `parallel_ragged_all_to_all`"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_all_to_all_batching_dynamic_extents() -> Result<(), ProgramError> {
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
                ArrayIrOperation::ParallelAllToAll(ParallelAllToAllOperation::new(
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
    fn test_parallel_all_to_all_differentiation() {
        // The homogeneous JVP applies exactly the same exchange to the primal and live tangent.
        let program = parallel_all_to_all_program(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()),
            ArrayType::new_static(DataType::F32, [4, 3]),
        );
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.instructions().len(), 2);
        assert_eq!(jvp.instructions()[0].operation(), jvp.instructions()[1].operation());
        assert!(matches!(jvp.instructions()[0].operation(), ArrayOperation::ParallelAllToAll(_)));
        assert_eq!(jvp.output_types(), vec![ArrayType::new_static(DataType::F32, [2, 6]); 2]);

        // Distinct destination and sender weights catch an incorrectly ordered exchange or pullback. The physical
        // mapped axis follows the split and concat axes, so differentiation must also retain their logical positions.
        check_gradient!(
            |inputs| {
                let exchanged = batch(
                    |item| {
                        let operation =
                            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
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
                ParallelAllToAllOperation::new("x".to_string(), 1, 1, 0, CollectiveOptions::default()),
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
    fn test_parallel_all_to_all_differentiation_with_zero_tangent() {
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
        let output = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::default())
            .jvp_in_parent(&DifferentiationContext::fused(trace.clone()), &EmptyRegionDriver, &inputs)
            .unwrap()
            .remove(0);
        assert!(output.tangent().is_zero());
        assert_eq!(
            output.primal().r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 2]))
        );
        assert_eq!(trace.builder().borrow().instructions().len(), 1);
        assert!(matches!(
            trace.builder().borrow().instructions()[0].operation(),
            ArrayIrOperation::ParallelAllToAll(_)
        ));
    }

    #[test]
    fn test_parallel_all_to_all_differentiation_shadows_manual_axis() {
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
                    let operation =
                        ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
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
                        let output = item.parallel_all_to_all_tiled("x", 0, 1)?;
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
    fn test_parallel_all_to_all_transposition() {
        // Both modes invert by swapping the split and concat axes, retaining the participant grouping.
        for options in [
            CollectiveOptions::tiled(),
            CollectiveOptions::default(),
            CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
            CollectiveOptions::default().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
        ] {
            let axis_size = if options.axis_index_groups().is_some() { 4 } else { 2 };
            let program = parallel_all_to_all_program(
                ParallelAllToAllOperation::new("x".to_string(), axis_size, 0, 1, options.clone()),
                ArrayType::new_static(DataType::F32, [2, 3]),
            );
            let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
            assert_eq!(transposed.instructions().len(), 1);
            assert_eq!(
                transposed.instructions()[0].operation(),
                &ArrayOperation::ParallelAllToAll(ParallelAllToAllOperation::new(
                    "x".to_string(),
                    axis_size,
                    1,
                    0,
                    options
                )),
            );
            assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());
        }

        // An exchange over a manual mesh axis transposes over the same mesh.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
        let program = parallel_all_to_all_program(
            operation.clone().with_mesh(mesh.clone()),
            ArrayType::new_static(DataType::F32, [2, 3])
                .with_sharding(Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["x"]).unwrap())
                .unwrap(),
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.instructions()[0].operation(),
            &ArrayOperation::ParallelAllToAll(
                ParallelAllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::tiled()).with_mesh(mesh),
            ),
        );
        assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());
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

        let parallel_all_to_all = program.instructions().last().unwrap();
        let ArrayIrOperation::ParallelAllToAll(operation) = parallel_all_to_all.operation() else {
            panic!("parallel_swap_axes must compose the canonical all-to-all operation");
        };
        assert_eq!(operation.split_axis(), 0);
        assert_eq!(operation.concat_axis(), 0);
        assert_eq!(operation.options(), &CollectiveOptions::default());
        assert_eq!(
            operation.mesh(),
            Some(&LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap()),
        );
        assert_eq!(parallel_all_to_all.inputs().len(), 3);
        let variation = program
            .instructions()
            .iter()
            .find(|instruction| {
                matches!(instruction.operation(), ArrayIrOperation::Array(ArrayOperation::ParallelVary(_)))
            })
            .unwrap();
        assert_eq!(parallel_all_to_all.inputs()[0], variation.outputs()[0]);
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
