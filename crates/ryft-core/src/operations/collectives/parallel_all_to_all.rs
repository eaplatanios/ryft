use std::fmt::Display;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType,
    Dimension, DimensionOperation, DimensionType, DimensionValue, DimensionVariable, LogicalMesh, Shape, Sharding,
};
use crate::axes::{Axis, AxisError, NamedAxes, NamedAxis};
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
use crate::operations::collectives::parallel_ragged_all_to_all::PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME;
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
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, MemberOperation, Operation, OperationFormatter, OperationProjection, ProgramError, ProjectedValue,
    RegionInterface, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`ParallelAllToAllOperation`].
pub const PARALLEL_ALL_TO_ALL_OPERATION_NAME: &str = "parallel_all_to_all";

/// [`Operation`] that exchanges chunks between participants along a named axis. Within each ordered participant group,
/// every sender splits its input along `split_axis` and receiver `i` gets chunk `i` from every sender, in group order.
/// The [`CollectiveMode`] of its [`CollectiveOptions`] determines the shape over a group of `n` participants:
///
///   - [`CollectiveMode::Untiled`] requires extent `n` at `split_axis`, removes that input axis, and inserts extent `n`
///     at `concatenation_axis` in the output. Each receiver gets one slice from each sender along the inserted axis.
///   - [`CollectiveMode::Tiled`] requires the split extent to be divisible by `n`. It divides that extent by `n` and
///     multiplies the concatenation extent by `n`. When the axes coincide, the shape is unchanged.
///
/// Both modes preserve rank and element data type. This is the Ryft analogue of JAX's
/// [`jax.lax.all_to_all`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_to_all.html). Tiled exchanges lower
/// directly to StableHLO's [`all_to_all`](https://openxla.org/stablehlo/spec#all_to_all), while untiled exchanges
/// insert and remove singleton dimensions around it. The collective is linear; its transpose swaps the split and
/// concatenation axes and retains the mode and ordered participant groups.
///
/// An exchange over a manual mesh axis is created by [`with_mesh`](Self::with_mesh), and
/// [`ParallelAllToAll::parallel_all_to_all_with_options`] supplies the mesh automatically from the enclosing manual
/// region, making an invariant input varying first. Such an exchange can give the receivers different values, so its
/// input must vary over the axis (refer to [`ParallelVary`]) and its output varies over it too. A pending sum over that
/// axis is rejected; sums over unrelated manual axes are preserved. An ordinary exchange carries no mesh and preserves
/// the input's mesh variation and pending sums, even when its input carries a manual mesh axis with the same name,
/// because a `batch` level whose axis name shadows that mesh axis may bind it instead. Type inference in the
/// homogeneous array family requires static extents; the composite array/dimension family uses explicit result
/// extents, with runtime assertions for dynamic split divisibility and untiled split size.
///
/// A matching `batch` level consumes the named axis of an ordinary exchange with a local reshape/transpose block
/// exchange. Batch item `i` receives every item's chunk `i`, in sender order. A replicated input is broadcast before
/// the exchange, since receivers can still get different chunks. Participant groups and exchanges over a manual mesh
/// axis are unsupported at a matching level. Outside any binder, a single-participant tiled exchange is the identity;
/// untiled mode relocates its size-one split axis to the concatenation position.
///
/// Bounded ragged inputs are rejected. One extent per item does not determine how each sender partitions
/// its live prefix among receivers; that requires the explicit offsets and per-destination sizes of
/// [`ParallelRaggedAllToAllOperation`](crate::ParallelRaggedAllToAllOperation).
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
    concatenation_axis: usize,

    /// [`CollectiveOptions`] of this [`ParallelAllToAllOperation`].
    options: CollectiveOptions,

    /// Refer to the documentation of [`mesh`](Self::mesh) for more information.
    mesh: Option<LogicalMesh>,
}

impl ParallelAllToAllOperation {
    /// Creates a new [`ParallelAllToAllOperation`] over the axis with the provided name and resolved axis size.
    /// Construction preserves the supplied axes and options, while type inference validates the geometry, groups,
    /// and mesh state. Unlike the [`ParallelAllToAll`] functions, which accept negative axes, this constructor takes
    /// non-negative axis positions.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of the axis whose participants exchange chunks.
    ///   - `axis_size`: Number of participants along `axis_name`, resolved when the operation is staged.
    ///   - `split_axis`: Axis of the input that is split into one slice or chunk per receiver.
    ///   - `concatenation_axis`: Axis of the output that holds the received slices or chunks in the order of their
    ///     senders. In untiled mode it is the position at which the new sender axis is inserted after `split_axis` is
    ///     removed, and in tiled mode it is the existing axis along which the received chunks are concatenated.
    ///   - `options`: [`CollectiveMode`] and optional participant groups of the collective (refer to the
    ///     documentation of [`CollectiveOptions::with_axis_index_groups`] for how the groups route the data).
    #[inline]
    pub fn new(
        axis_name: String,
        axis_size: usize,
        split_axis: usize,
        concatenation_axis: usize,
        options: CollectiveOptions,
    ) -> Self {
        Self { axis_name, axis_size, split_axis, concatenation_axis, options, mesh: None }
    }

    /// Returns this [`ParallelAllToAllOperation`] configured to exchange chunks over a manual axis of `mesh`. The input
    /// must vary over [`axis_name`](Self::axis_name) on that mesh, whose size must equal [`axis_size`](Self::axis_size)
    /// and must not carry a pending cross-device sum over that axis. Sums over unrelated manual axes are preserved.
    /// Type inference validates these requirements. [`ParallelAllToAll::parallel_all_to_all_with_options`] supplies
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

    /// Returns the axis of the input that is split into one chunk per participant.
    #[inline]
    pub fn split_axis(&self) -> usize {
        self.split_axis
    }

    /// Returns the axis of the output along which the received chunks are concatenated.
    #[inline]
    pub fn concatenation_axis(&self) -> usize {
        self.concatenation_axis
    }

    /// Returns the [`CollectiveOptions`] of this [`ParallelAllToAllOperation`].
    #[inline]
    pub fn options(&self) -> &CollectiveOptions {
        &self.options
    }

    /// Returns the logical mesh whose manual axis this [`ParallelAllToAllOperation`] exchanges chunks over, or [`None`]
    /// for an ordinary exchange, whose named axis may be bound by any enclosing binder. Only an exchange over a manual
    /// mesh axis validates the manual variation and pending sums of its input.
    #[inline]
    pub fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

    /// Validates the manual variation and pending sums of an exchange over a manual mesh axis, given the shape-only
    /// `output_type` shared by the static and array IR inference paths. An ordinary exchange preserves its input's
    /// mesh state, including pending sums, because a `batch` level that binds its axis performs only local array
    /// rearrangement, even when its axis name shadows a manual mesh axis.
    fn finalize_output_type(&self, input_type: &ArrayType, output_type: ArrayType) -> Result<ArrayType, TypeError> {
        let Some(mesh) = &self.mesh else {
            return Ok(output_type);
        };

        let axis_name = self.axis_name();
        validate_manual_mesh_input(
            PARALLEL_ALL_TO_ALL_OPERATION_NAME,
            axis_name,
            Some(self.axis_size),
            mesh,
            input_type,
        )?;

        // Exchanging chunks cannot complete a pending sum over the same axis. Sums over unrelated axes commute
        // with the exchange and retain their pending state.
        let sharding = input_type.sharding().unwrap();
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
        let input_type = self.validate_input(input_types, region_interfaces)?;

        // Result shape arithmetic in the homogeneous array family requires static extents.
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
        if self.split_axis >= rank || self.concatenation_axis >= rank {
            return Err(TypeError::invalid(format!(
                "`{}` split axis {} or concatenation axis {} is out of bounds for rank {}",
                PARALLEL_ALL_TO_ALL_OPERATION_NAME, self.split_axis, self.concatenation_axis, rank,
            )));
        }

        let output_type = if self.options.mode == CollectiveMode::Untiled {
            if output_dimensions[self.split_axis] != effective_axis_size {
                return Err(TypeError::invalid(format!(
                    "`{}` untiled split axis {} size {} must equal group size {}",
                    PARALLEL_ALL_TO_ALL_OPERATION_NAME,
                    self.split_axis,
                    output_dimensions[self.split_axis],
                    effective_axis_size,
                )));
            }
            input_type
                .without_dimension(self.split_axis)?
                .0
                .with_inserted_dimension(self.concatenation_axis, Dimension::Static(effective_axis_size))?
        } else {
            if output_dimensions[self.split_axis] % effective_axis_size != 0 {
                return Err(TypeError::invalid(format!(
                    "`{}` split axis {} size {} is not divisible by group size {}",
                    PARALLEL_ALL_TO_ALL_OPERATION_NAME,
                    self.split_axis,
                    output_dimensions[self.split_axis],
                    effective_axis_size,
                )));
            }

            output_dimensions[self.split_axis] /= effective_axis_size;
            output_dimensions[self.concatenation_axis] =
                output_dimensions[self.concatenation_axis].checked_mul(effective_axis_size).ok_or_else(|| {
                    TypeError::invalid(format!(
                        "`{PARALLEL_ALL_TO_ALL_OPERATION_NAME}` concatenation result extent does not fit in usize",
                    ))
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
            operation.field("concatenation_axis", format_args!("{:?}", &self.concatenation_axis))?;
            operation.field("options", format_args!("{:?}", self.options.mode()))?;
            if let Some(axis_index_groups) = self.options.axis_index_groups() {
                operation.field("axis_index_groups", format_args!("{axis_index_groups:?}"))?;
            }
            if let Some(mesh) = &self.mesh {
                operation.field("mesh", mesh)?;
            }
            Ok(())
        })
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

    #[inline]
    fn adjoint(&self, _input_type: &ArrayType) -> Result<ParallelAllToAllOperation, ProgramError> {
        // The chunk exchange is its own adjoint with the split and concatenation axes swapped,
        // over the same axis, participant groups, and mesh.
        Ok(ParallelAllToAllOperation {
            split_axis: self.concatenation_axis,
            concatenation_axis: self.split_axis,
            ..self.clone()
        })
    }

    #[inline]
    fn adapt_to_batch_axis(&self, input_batch_axis: usize) -> (Self, usize) {
        let (split_axis, output_batch_axis) = self.options.mode.forwarded_split_axes(self.split_axis, input_batch_axis);
        let (concatenation_axis, output_batch_axis) =
            self.options.mode.forwarded_concatenation_axes(self.concatenation_axis, output_batch_axis);
        (Self { split_axis, concatenation_axis, ..self.clone() }, output_batch_axis)
    }
}

impl ShapeChangingCollectiveOperation for ParallelAllToAllOperation {
    #[inline]
    fn collective_options(&self) -> &CollectiveOptions {
        &self.options
    }

    fn infer_array_ir_output_types(&self, input_types: &[ArrayIrType]) -> Result<Vec<ArrayIrType>, TypeError> {
        let effective_axis_size = self.effective_axis_size()?;
        let Some(input_type) = input_types.first() else {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_ALL_TO_ALL_OPERATION_NAME}` expects an array followed by its output extents",
            )));
        };

        let input_type = <&ArrayType>::try_from(input_type)?;
        if self.options.mode == CollectiveMode::Untiled {
            let Some(input_extent) = input_type.shape().dimensions().get(self.split_axis) else {
                return Err(TypeError::invalid(format!(
                    "`{}` split axis {} is out of bounds for rank {}",
                    PARALLEL_ALL_TO_ALL_OPERATION_NAME,
                    self.split_axis,
                    input_type.rank(),
                )));
            };

            if let Dimension::Static(input_extent) = input_extent
                && *input_extent != effective_axis_size
            {
                return Err(TypeError::invalid(format!(
                    "`{}` untiled split axis {} size {} must equal group size {}",
                    PARALLEL_ALL_TO_ALL_OPERATION_NAME, self.split_axis, input_extent, effective_axis_size,
                )));
            }

            let output_type = input_type
                .without_dimension(self.split_axis)?
                .0
                .with_inserted_dimension(self.concatenation_axis, Dimension::Static(effective_axis_size))?;

            let mut output_types = infer_array_ir_shape_changing_collective_output_type(
                PARALLEL_ALL_TO_ALL_OPERATION_NAME,
                input_types,
                output_type,
                &[self.concatenation_axis],
                |output_extents| {
                    let output_extent = &output_extents[self.concatenation_axis];
                    if output_extent != &Dimension::Static(effective_axis_size) {
                        return Err(TypeError::invalid(format!(
                            "`{}` inserted output axis {} extent must equal axis group size {} but got {}",
                            PARALLEL_ALL_TO_ALL_OPERATION_NAME,
                            self.concatenation_axis,
                            effective_axis_size,
                            output_extent,
                        )));
                    }
                    Ok(())
                },
            )?;

            let output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
            return Ok(vec![self.finalize_output_type(input_type, output_type)?.into()]);
        }

        if self.split_axis == self.concatenation_axis {
            let Some(input_extent) = input_type.shape().dimensions().get(self.split_axis) else {
                return Err(TypeError::invalid(format!(
                    "`{}` split axis {} is out of bounds for rank {}",
                    PARALLEL_ALL_TO_ALL_OPERATION_NAME,
                    self.split_axis,
                    input_type.rank(),
                )));
            };

            if let Dimension::Static(input_extent) = input_extent
                && *input_extent % effective_axis_size != 0
            {
                return Err(TypeError::invalid(format!(
                    "`{}` split axis {} size {} is not divisible by group size {}",
                    PARALLEL_ALL_TO_ALL_OPERATION_NAME, self.split_axis, input_extent, effective_axis_size,
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

        if self.split_axis >= input_type.rank() || self.concatenation_axis >= input_type.rank() {
            return Err(TypeError::invalid(format!(
                "`{}` split axis {} or concatenation axis {} is out of bounds for rank {}",
                PARALLEL_ALL_TO_ALL_OPERATION_NAME,
                self.split_axis,
                self.concatenation_axis,
                input_type.rank(),
            )));
        }

        if let Dimension::Static(input_extent) = &input_type.shape().dimensions()[self.split_axis]
            && *input_extent % effective_axis_size != 0
        {
            return Err(TypeError::invalid(format!(
                "`{}` split axis {} size {} is not divisible by group size {}",
                PARALLEL_ALL_TO_ALL_OPERATION_NAME, self.split_axis, input_extent, effective_axis_size,
            )));
        }

        let expected_concatenation_extent = match input_type.shape().dimensions()[self.concatenation_axis] {
            Dimension::Static(input_extent) => {
                Some(input_extent.checked_mul(effective_axis_size).ok_or_else(|| {
                    TypeError::invalid(format!(
                        "`{PARALLEL_ALL_TO_ALL_OPERATION_NAME}` concatenation result extent does not fit in usize",
                    ))
                })?)
            }
            Dimension::Dynamic(_) => None,
        };

        let mut dimensions = input_type.shape().dimensions().to_vec();
        dimensions[self.split_axis] = Dimension::Static(0);
        dimensions[self.concatenation_axis] = Dimension::Static(0);
        let sharding = input_type.resized_sharding(dimensions.as_slice(), PARALLEL_ALL_TO_ALL_OPERATION_NAME)?;
        let base_output_type = ArrayType::new(input_type.data_type(), Shape::new(dimensions))
            .with_memory(input_type.memory())
            .with_sharding(sharding)?;
        let mut output_types = infer_array_ir_shape_changing_collective_output_type(
            PARALLEL_ALL_TO_ALL_OPERATION_NAME,
            input_types,
            base_output_type,
            &[self.split_axis, self.concatenation_axis],
            |output_extents| {
                if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                    (&input_type.shape().dimensions()[self.split_axis], &output_extents[self.split_axis])
                {
                    let expected = *input_extent / effective_axis_size;
                    if *output_extent != expected {
                        return Err(TypeError::invalid(format!(
                            "`{PARALLEL_ALL_TO_ALL_OPERATION_NAME}` split result extent must equal input axis {} \
                             extent {input_extent} divided by group size {effective_axis_size}; expected {expected} \
                             but got {output_extent}",
                            self.split_axis,
                        )));
                    }
                }

                if let (Some(expected), Dimension::Static(output_extent)) =
                    (expected_concatenation_extent, &output_extents[self.concatenation_axis])
                {
                    let input_extent = &input_type.shape().dimensions()[self.concatenation_axis];
                    if *output_extent != expected {
                        return Err(TypeError::invalid(format!(
                            "`{}` concatenation result extent must equal input axis {} extent {} multiplied \
                             by group size {}; expected {} but got {}",
                            PARALLEL_ALL_TO_ALL_OPERATION_NAME,
                            self.concatenation_axis,
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

        // Placeholder zeros cannot establish the sharding constraints of the actual result dimensions.
        let sharding =
            input_type.resized_sharding(output_type.shape().dimensions(), PARALLEL_ALL_TO_ALL_OPERATION_NAME)?;
        output_type = output_type.with_sharding(sharding)?;
        if output_type.shape() == input_type.shape() {
            output_type = output_type.with_layout(input_type.layout().cloned());
        }

        Ok(vec![self.finalize_output_type(input_type, output_type)?.into()])
    }

    #[inline]
    fn unsupported_ragged_input_error(&self, dimension: &DimensionVariable, _input_index: usize) -> BatchingError {
        // One extent per item does not determine how each sender partitions its live prefix among the receivers,
        // which requires the explicit offsets and per-destination sizes of a ragged all-to-all instead.
        BatchingError::UnsupportedOperation {
            message: format!(
                "`{PARALLEL_ALL_TO_ALL_OPERATION_NAME}` cannot route bounded ragged dimension `{dimension}` \
                 without explicit per-destination offsets and sizes; use `{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}`",
            ),
        }
    }
}

impl<C: Context<Type = ArrayType, Value: Transpose>> ShapeChangingCollectiveBatching<C> for ParallelAllToAllOperation {
    fn batch_matching_axis<P: CollectiveArrayExtentBatchingPolicy<C>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        input: &ArrayBatch<C::Value>,
        output_extents: Vec<P::ShapeExtent>,
        output_sharding: Option<Sharding>,
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        // Materialize the exchange locally: batch items initially index senders, and chunk indices along the split
        // axis index receivers. Swapping these two axes makes output batch item `i` contain chunk `i` from every
        // sender.
        let logical_input_rank = input.unbatched_type().rank();
        if self.options.axis_index_groups.is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{PARALLEL_ALL_TO_ALL_OPERATION_NAME}` axis index groups are not supported \
                     when a batch transform binds the collective axis",
                ),
            });
        }

        if self.split_axis >= logical_input_rank || self.concatenation_axis >= logical_input_rank {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{}` split axis {} or concatenation axis {} is out of bounds for rank {}",
                    PARALLEL_ALL_TO_ALL_OPERATION_NAME, self.split_axis, self.concatenation_axis, logical_input_rank,
                ),
            });
        }

        let axis_extent =
            P::collective_axis_extent(context, PARALLEL_ALL_TO_ALL_OPERATION_NAME, &self.axis_name, self.axis_size)?;

        // Recover each sender's input shape from the inferred per-receiver output extents. The extent policy keeps
        // this calculation shared between static shapes and symbolic shapes that need runtime divisibility checks.
        let (input_extents, chunk_extent) = match self.options.mode {
            CollectiveMode::Untiled => {
                // Undo insertion of the sender axis and restore the receiver axis removed from each input.
                // Each receiver takes one slice, represented below as a size-one chunk axis.
                let mut input_extents = output_extents.clone();
                input_extents.remove(self.concatenation_axis);
                input_extents.insert(self.split_axis, axis_extent.clone());
                (input_extents, P::collective_extent_constant(context, 1)?)
            }
            CollectiveMode::Tiled if self.split_axis == self.concatenation_axis => {
                // Splitting and concatenating along the same axis preserves its total extent, but each receiver
                // still takes only one participant's share of that axis from each sender.
                let chunk_extent = P::divide_extents_exactly(context, &output_extents[self.split_axis], &axis_extent)?;
                (output_extents.clone(), chunk_extent)
            }
            CollectiveMode::Tiled => {
                // Undo concatenation along the received axis and splitting along the sent axis.
                // The output split extent is already the size of a chunk sent to one receiver.
                let mut input_extents = output_extents.clone();
                input_extents[self.concatenation_axis] =
                    P::divide_extents_exactly(context, &output_extents[self.concatenation_axis], &axis_extent)?;
                input_extents[self.split_axis] = output_extents[self.split_axis].mul(&axis_extent)?;
                (input_extents, output_extents[self.split_axis].clone())
            }
        };

        // Put the sender batch axis first, broadcasting replicated inputs to all senders when necessary. Even equal
        // sender inputs can give different receiver outputs, since each receiver selects a different chunk.
        let input = P::match_collective_axis(context, input, input_extents.as_slice())?;

        // Factor the logical split axis into `[receiver, chunk]`, with every logical axis offset by the leading sender
        // axis. Intermediate reshapes omit sharding; the final reshape installs the inferred output placement.
        let mut split_extents = Vec::with_capacity(input_extents.len() + 2);
        split_extents.push(axis_extent.clone());
        split_extents.extend(input_extents.iter().cloned());
        split_extents[self.split_axis + 1] = axis_extent.clone();
        split_extents.insert(self.split_axis + 2, chunk_extent);
        let split = P::reshape_collective(context, input.into_value(), split_extents.as_slice(), None)?;

        // The receiver axis becomes the leading batch axis, while the old sender axis moves next to the chunk axis.
        let exchanged = split.swap_axes(0, self.split_axis + 1)?;
        let received = match self.options.mode {
            CollectiveMode::Untiled => {
                // Remove the size-one chunk axis, then move the sender axis to the output's concatenation position.
                let mut squeezed_extents = Vec::with_capacity(input_extents.len() + 1);
                squeezed_extents.push(axis_extent.clone());
                squeezed_extents.extend(input_extents);
                P::reshape_collective(context, exchanged, squeezed_extents.as_slice(), None)?
                    .move_axis(self.split_axis + 1, self.concatenation_axis + 1)?
            }
            CollectiveMode::Tiled => {
                // Place senders immediately before the concatenation axis so the final reshape merges their chunks
                // in sender order. The extra leading receiver axis accounts for the `+ 1` offsets in both modes.
                exchanged.move_axis(self.split_axis + 1, self.concatenation_axis + 1)?
            }
        };

        // Restore the inferred per-receiver output shape and sharding, prepending the batching context's receiver
        // axis to both. The result stays mapped because different receivers can receive different values.
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

impl<C: Domain<Type = ArrayType, Value: Reshape>> InterpretableOperation<C> for ParallelAllToAllOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // Eager binding does not infer output types, so interpretation validates the shared input contract
        // and the operation payload before applying the degenerate-axis rule.
        check_count!("input", inputs, 1, ProgramError);
        self.validate_degenerate_interpretation()?;
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
        // A matching `batch` level consumes the mapped batch axis with a reshape/transpose block exchange: the per-item
        // `split_axis` is split into `(b, d_p / b)` chunks, the chunk axis is swapped with the leading batch axis (so
        // the batch axis indexes the *receiving* item), and the sender axis is then merged item-major into the per-item
        // `concatenation_axis` (batch item `i` receives every item's chunk `i`, concatenated along
        // `concatenation_axis`). A non-matching level forwards the collective to the parent context,
        // unchanged for a replicated input (through `BatchingContext::forward_to_parent`) and with
        // its array axes shifted past the batch axis for a mapped one.
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

impl<
    C: Context<
            Type = ArrayIrType,
            Operation: From<ConstantOperation<DimensionValue>>
                           + From<ParallelAllToAllOperation>
                           + From<DimensionSizeOperation>
                           + From<LinearCallOperation<ArrayIrType>>
                           + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
        >,
> MemberDifferentiableOperation<C> for ParallelAllToAllOperation
{
    #[inline]
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // Explicit output extents are retained as ordinary residual values, and the
        // transposed linear region swaps the split and concatenation axes.
        self.shape_changing_collective_jvp_in_parent(context, inputs)
    }
}

/// Represents the ability to exchange chunks between participants of a named axis by staging a
/// [`ParallelAllToAllOperation`]. Refer to that operation for the tiling, grouping, variation, and transformation
/// semantics. Dynamic result extents are staged as first-class dimension values, and runtime assertions validate
/// dynamic split extents.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// Each row sends its first half to batch item zero and its second half to batch item one:
///
/// ```
/// # use ryft_core::{
/// #     Array, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, BatchAxis, BatchAxisSpecification,
/// #     BatchingTracer, EagerContext, ParallelAllToAll, batch,
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
#[capability]
pub trait ParallelAllToAll<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Exchanges single slices between the participants of the named axis `axis_name`. The extent of `split_axis`
    /// must equal the number of participants, and participant `r` receives slice `r` along `split_axis` of the value
    /// of every participant. The received slices drop `split_axis` and are stacked in the order of their senders
    /// along a new axis inserted at `concatenation_axis`, so the rank is preserved.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `split_axis`: Input axis split into one slice per receiver. Negative axes count from the end.
    ///   - `concatenation_axis`: Output position at which sender slices are stacked after removing `split_axis`.
    ///     Negative positions count from the end of the output, whose rank equals the rank of this value.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`ParallelAllToAll::parallel_all_to_all_with_options`].
    #[inline]
    fn parallel_all_to_all<SplitAxis: Into<Axis>, ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        split_axis: SplitAxis,
        concatenation_axis: ConcatenationAxis,
    ) -> Result<Self, ProgramError> {
        self.parallel_all_to_all_with_options(axis_name, split_axis, concatenation_axis, CollectiveOptions::default())
    }

    /// Exchanges equal contiguous chunks between the participants of the named axis `axis_name`. The extent of
    /// `split_axis` must be divisible by the number of participants `n`, and participant `r` receives chunk `r` along
    /// `split_axis` of the value of every participant. The received chunks are concatenated in the order of their
    /// senders along `concatenation_axis`, so the extent of `split_axis` is divided by `n` and that of
    /// `concatenation_axis` is multiplied by `n`. Coincident axes therefore preserve the shape.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `split_axis`: Input axis split into one chunk per receiver. Negative axes count from the end.
    ///   - `concatenation_axis`: Array axis along which received chunks are concatenated in sender order. Negative axes
    ///     count from the end.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`ParallelAllToAll::parallel_all_to_all_with_options`].
    #[inline]
    fn parallel_all_to_all_tiled<SplitAxis: Into<Axis>, ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        split_axis: SplitAxis,
        concatenation_axis: ConcatenationAxis,
    ) -> Result<Self, ProgramError> {
        self.parallel_all_to_all_with_options(
            axis_name,
            split_axis,
            concatenation_axis,
            CollectiveOptions::new(CollectiveMode::Tiled),
        )
    }

    /// Exchanges slices or chunks between the participants of the named axis `axis_name`,
    /// like [`parallel_all_to_all`](Self::parallel_all_to_all) in untiled mode and like
    /// [`parallel_all_to_all_tiled`](Self::parallel_all_to_all_tiled) in tiled mode, within the participant groups
    /// of `options`. Over a manual mesh axis, an invariant input is first made varying through [`ParallelVary`].
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `split_axis`: Input axis split into one slice or chunk per receiver. Negative axes count from the end.
    ///   - `concatenation_axis`: Output axis holding the received slices or chunks, as selected by `options`.
    ///     Negative axes count from the end of the output, whose rank equals the rank of this value in both modes.
    ///   - `options`: [`CollectiveMode`] and optional participant groups of the collective (refer to the
    ///     documentation of [`CollectiveOptions::with_axis_index_groups`] for how the groups route the data).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Axis`] wrapping [`AxisError::UnboundAxisName`](crate::AxisError::UnboundAxisName)
    /// when no binder binds `axis_name` or [`AxisError::OutOfBounds`](crate::AxisError::OutOfBounds) when an axis is
    /// out of bounds, and a [`ProgramError`] for invalid groups or split geometry, pending cross-device sums, or an
    /// overflowing concatenation extent. A batch level that binds the collective axis rejects participant groups and
    /// bounded ragged inputs.
    fn parallel_all_to_all_with_options<SplitAxis: Into<Axis>, ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        split_axis: SplitAxis,
        concatenation_axis: ConcatenationAxis,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError>;

    /// Swaps `axis` with `axis_name` over the full named axis, exchanging one ranked array axis with the named axis.
    /// The ranked axis must have the participant count as its extent. Index `j` along `axis` of the result of
    /// participant `i` holds index `i` along `axis` of the value of participant `j`. This is
    /// [`Self::parallel_all_to_all`] with identical split and concatenation positions.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `axis`: Ranked axis to exchange with the named axis. Negative axes count from the end.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::parallel_all_to_all_with_options`].
    #[inline]
    fn parallel_swap_axes<SwappedAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        axis: SwappedAxis,
    ) -> Result<Self, ProgramError> {
        let axis = axis.into();
        self.parallel_all_to_all(axis_name, axis, axis)
    }

    /// Swaps `axis` with `axis_name` within the provided ordered participant groups. The ranked axis extent must
    /// equal the common group size, and positions within each group replace axis indices: index `q` along `axis` of
    /// the result of the member at position `p` holds index `p` along `axis` of the value of the member at position
    /// `q`.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `axis`: Ranked axis to exchange with the named axis. Negative axes count from the end.
    ///   - `axis_index_groups`: Participant groups, each listing axis indices of `axis_name`, that
    ///     together partition all of its indices into groups of equal size (refer to the documentation
    ///     of [`CollectiveOptions::with_axis_index_groups`] for how the groups route the data).
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::parallel_all_to_all_with_options`].
    #[inline]
    fn parallel_swap_axes_with_axis_index_groups<SwappedAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        axis: SwappedAxis,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError> {
        let axis = axis.into();
        self.parallel_all_to_all_with_options(
            axis_name,
            axis,
            axis,
            CollectiveOptions::default().with_axis_index_groups(axis_index_groups),
        )
    }
}

impl ParallelAllToAll<ArrayType> for Array {
    #[inline]
    fn parallel_all_to_all_with_options<SplitAxis: Into<Axis>, ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        _split_axis: SplitAxis,
        _concatenation_axis: ConcatenationAxis,
        _options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        // A concrete `Array` never executes inside an axis binder, because the values under a `batch` level or inside
        // a manual region are tracers, so every axis name is unbound for it.
        Err(AxisError::UnboundAxisName { name: axis_name.to_string() }.into())
    }
}

impl<A: Value<Type = ArrayType> + ParallelAllToAll<ArrayType>> ParallelAllToAll<ArrayIrType> for ArrayIrValue<A> {
    fn parallel_all_to_all_with_options<SplitAxis: Into<Axis>, ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        split_axis: SplitAxis,
        concatenation_axis: ConcatenationAxis,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        // A concrete composite value performs the collective through its array member.
        let array = ValueProjection::<ArrayType>::into_projected(self.clone())?;
        Ok(<Self as ValueProjection<ArrayType>>::from_projected(array.parallel_all_to_all_with_options(
            axis_name,
            split_axis,
            concatenation_axis,
            options,
        )?))
    }
}

impl<V> ParallelAllToAll<ArrayIrType> for V
where
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<Type = ArrayIrType, Operation: From<ParallelAllToAllOperation>>
                                + NamedAxes
                                + DimensionConstant,
        > + Assert
        + DimensionSize<V>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType> + Mul + Div + Rem + Compare<V>>
        + ValueProjection<ArrayType, Projected: ParallelVary>,
{
    fn parallel_all_to_all_with_options<SplitAxis: Into<Axis>, ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        split_axis: SplitAxis,
        concatenation_axis: ConcatenationAxis,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        // Composite values stage the array followed by one result extent per axis. Only a manual mesh binder records
        // its mesh on the operation and introduces variation; a named batch that shadows the same mesh-axis name
        // performs its own local exchange.
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let effective_axis_size = options.effective_axis_size(PARALLEL_ALL_TO_ALL_OPERATION_NAME, axis_size)?;
        let self_type = self.r#type();
        let rank = <&ArrayType>::try_from(self_type.as_ref())?.rank();
        let split_axis = split_axis.into().normalize(rank)?;
        let concatenation_axis = concatenation_axis.into().normalize(rank)?;
        let mut input = self.clone();
        let mut operation = ParallelAllToAllOperation::new(
            axis_name.to_string(),
            axis_size,
            split_axis,
            concatenation_axis,
            options.clone(),
        );

        if let Some(NamedAxis::Mesh { mesh, .. }) = context.named_axis(axis_name) {
            let array = ValueProjection::<ArrayType>::into_projected(self.clone())?;
            if array.r#type().unreduced_axes().contains(axis_name) {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_ALL_TO_ALL_OPERATION_NAME}` does not support unreduced inputs",
                ))
                .into());
            }
            if !array.r#type().sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis_name)) {
                input = <V as ValueProjection<ArrayType>>::from_projected(array.parallel_vary(axis_name)?);
            }
            operation = operation.with_mesh(mesh);
        }

        let input_type = input.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;

        // Untiled exchange replaces the split axis. Its static extent needs no instruction. A dynamic one still
        // needs an equality assertion before it is discarded. Operation type inference checks static geometry.
        if options.mode == CollectiveMode::Untiled
            && matches!(input_type.shape().dimensions()[split_axis], Dimension::Dynamic(_))
        {
            let extent = ValueProjection::<DimensionType>::into_projected(input.dimension_size(split_axis)?)?;
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
            .filter(|axis| options.mode != CollectiveMode::Untiled || *axis != split_axis)
            .map(|axis| input.dimension_size(axis))
            .collect::<Result<Vec<_>, _>>()?;
        match options.mode {
            CollectiveMode::Untiled => {
                output_extents.insert(concatenation_axis, context.dimension_constant(effective_axis_size)?);
            }
            CollectiveMode::Tiled => {
                // Coincident axes preserve the shape. Only a dynamic split then needs a participant constant
                // and divisibility assertion. Distinct axes additionally use ordinary dimension division and
                // multiplication.
                if split_axis != concatenation_axis
                    || matches!(input_type.shape().dimensions()[split_axis], Dimension::Dynamic(_))
                {
                    let extent = ValueProjection::<DimensionType>::into_projected(output_extents[split_axis].clone())?;
                    let participants = ValueProjection::<DimensionType>::into_projected(
                        context.dimension_constant(effective_axis_size)?,
                    )?;
                    if matches!(input_type.shape().dimensions()[split_axis], Dimension::Dynamic(_)) {
                        let zero = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(0)?)?;
                        extent.rem(&participants)?.equal(&zero)?.assert(
                            "collective extent must be divisible by the participant count",
                            &[
                                ("extent", ValueProjection::<DimensionType>::from_projected(extent.clone())),
                                ("divisor", ValueProjection::<DimensionType>::from_projected(participants.clone())),
                            ],
                        )?;
                    }

                    if split_axis != concatenation_axis {
                        let concatenation_extent = ValueProjection::<DimensionType>::into_projected(
                            output_extents[concatenation_axis].clone(),
                        )?;
                        output_extents[split_axis] =
                            ValueProjection::<DimensionType>::from_projected(extent.div(&participants)?);
                        output_extents[concatenation_axis] =
                            ValueProjection::<DimensionType>::from_projected(concatenation_extent.mul(&participants)?);
                    }
                }
            }
        }

        let inputs = std::iter::once(input).chain(output_extents).collect::<Vec<_>>();
        let mut outputs = context.bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<V: ParallelAllToAll<ArrayIrType> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>>
    ParallelAllToAll<ArrayType> for ProjectedValue<ArrayType, V>
{
    #[inline]
    fn parallel_all_to_all_with_options<SplitAxis: Into<Axis>, ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        split_axis: SplitAxis,
        concatenation_axis: ConcatenationAxis,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        self.value()
            .parallel_all_to_all_with_options(axis_name, split_axis, concatenation_axis, options)?
            .into_projected()
            .map_err(Into::into)
    }
}

impl<
    V: ShapeChangingCollectiveValue<
            DispatchDomain: Context<Value = V, Operation: From<ParallelAllToAllOperation>> + NamedAxes,
        > + ParallelVary,
> ParallelAllToAll<ArrayType> for V
{
    fn parallel_all_to_all_with_options<SplitAxis: Into<Axis>, ConcatenationAxis: Into<Axis>>(
        &self,
        axis_name: &str,
        split_axis: SplitAxis,
        concatenation_axis: ConcatenationAxis,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        // Homogeneous values opt into direct staging, while projected values retain composite extent delegation.
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        options.effective_axis_size(PARALLEL_ALL_TO_ALL_OPERATION_NAME, axis_size)?;
        let rank = self.r#type().rank();
        let split_axis = split_axis.into().normalize(rank)?;
        let concatenation_axis = concatenation_axis.into().normalize(rank)?;
        let mut input = self.clone();
        let mut operation =
            ParallelAllToAllOperation::new(axis_name.to_string(), axis_size, split_axis, concatenation_axis, options);
        if let Some(NamedAxis::Mesh { mesh, .. }) = context.named_axis(axis_name) {
            if input.r#type().unreduced_axes().contains(axis_name) {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_ALL_TO_ALL_OPERATION_NAME}` does not support unreduced inputs",
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayOperation, DataType, DimensionBounds, Layout, Memory, MeshAxis, MeshAxisType,
        RaggedAxis, ShardingDimension, StridedLayout,
    };
    use crate::batching::{BatchAxisSpecification, BatchingTracer, batch};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::DifferentiationTracer;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference,
    };
    use crate::operations::assertions::AssertionError;
    use crate::operations::collectives::tests::{batch_collective, collective_program};
    use crate::operations::manipulation::slicing::Slice;
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationOutput, PartialEvaluationValue};
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};

    use super::*;

    #[test]
    fn test_parallel_all_to_all() {
        let operation = ParallelAllToAllOperation::new("x".to_string(), 4, 0, 1, CollectiveOptions::tiled());
        assert_eq!(operation.name(), PARALLEL_ALL_TO_ALL_OPERATION_NAME);
        assert_eq!(operation.axis_name(), "x");
        assert_eq!(operation.axis_size(), 4);
        assert_eq!(operation.split_axis(), 0);
        assert_eq!(operation.concatenation_axis(), 1);
        assert_eq!(operation.options(), &CollectiveOptions::tiled());
        assert_eq!(operation.mesh(), None);
        assert_eq!(
            operation.to_string(),
            "parallel_all_to_all [axis_name=\"x\", axis_size=4, split_axis=0, concatenation_axis=1, options=Tiled]",
        );
        assert_eq!(operation.effective_axis_size(), Ok(4));

        // Ordered participant groups exchange chunks within each group, so the common group size replaces the axis
        // size as the participant count.
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
    fn test_parallel_all_to_all_with_mesh() {
        // An exchange over a manual mesh axis records and renders its mesh, which distinguishes it from the same
        // exchange without a mesh.
        let operation = ParallelAllToAllOperation::new("x".to_string(), 4, 0, 1, CollectiveOptions::tiled());
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
                    concatenation_axis=1,
                    options=Tiled,
                    mesh=['x'=4:manual],
                ]"
            },
        );
        assert_ne!(mesh_operation, operation);
    }

    #[test]
    fn test_parallel_all_to_all_type_inference() {
        // Tiled exchanges divide the split extent and multiply the concatenation extent by the participant count, so
        // they require static, in-bounds, divisible extents and a concatenation extent that does not overflow.
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
                {
                    input_types = [ArrayType::new_static(DataType::F32, [4, usize::MAX])],
                    error = "`parallel_all_to_all` concatenation result extent does not fit in usize",
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [8])],
                    error = "`parallel_all_to_all` split axis 0 or concatenation axis 1 is out of bounds for rank 1",
                },
                {
                    input_types = [ArrayType::new(
                        DataType::F32,
                        Shape::new(vec![
                            DimensionVariable::new("length", DimensionBounds::unbounded()).into(),
                            Dimension::Static(3),
                        ]),
                    )],
                    error = "`parallel_all_to_all` does not support dynamically shaped inputs",
                },
            ],
        );

        // Untiled exchanges replace the split axis, whose extent must equal the participant count, with a sender axis
        // at the concatenation position, for any element data type.
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

        // Participant groups use the common group size, rather than the axis size, as the participant count.
        check_operation_type_inference!(
            operation = ParallelAllToAllOperation::new(
                "x".to_string(),
                4,
                0,
                1,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
            ),
            cases = [{
                input_types = [ArrayType::new_static(DataType::C64, [6, 3])],
                output_types = [ArrayType::new_static(DataType::C64, [3, 6])],
            }],
        );
    }

    #[test]
    fn test_parallel_all_to_all_type_inference_metadata() {
        // A shape-preserving exchange retains the complete input type, including its layout and memory space, in both
        // type representations.
        let input = ArrayType::new_static(DataType::F32, [4, 3])
            .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])))
            .with_memory(Memory::Host { pinned: true });
        let operation = ParallelAllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::tiled());
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input), &[]), Ok(vec![input.clone()]));
        assert_eq!(
            operation.infer_parent_output_types(
                &[
                    input.clone().into(),
                    DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
                &[]
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
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled())
                .infer_parent_output_types(
                    &[
                        input.into(),
                        DimensionValue::constant(1).unwrap().r#type().into_owned().into(),
                        DimensionValue::constant(6).unwrap().r#type().into_owned().into(),
                    ],
                    &[]
                ),
            Err(TypeError::invalid(
                "`parallel_all_to_all` on a dimension sharded over explicit mesh axes requires the output size (1) at \
                 axis 0 to be divisible by the mesh-axis product (2)",
            )),
        );
    }

    #[test]
    fn test_parallel_all_to_all_type_inference_manual_mesh() {
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
            operation.clone().with_mesh(mesh.clone()).infer_parent_output_types(
                &[
                    with_sharding([2, 3], &unreduced).into(),
                    DimensionValue::constant(1).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(6).unwrap().r#type().into_owned().into(),
                ],
                &[]
            ),
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
    fn test_parallel_all_to_all_type_inference_preserves_unrelated_pending_sums() {
        // Exchanging chunks along `x` preserves a pending sum over the independent `y` axis in both type
        // representations.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled())
            .with_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh, 2)
            .with_unreduced_axes(["y"])
            .unwrap()
            .with_varying_manual_axes(["x"])
            .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(sharding.clone()).unwrap();
        let output_type = ArrayType::new_static(DataType::F32, [1, 6]).with_sharding(sharding).unwrap();
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input_type), &[]), Ok(vec![output_type.clone()]));
        assert_eq!(
            operation.infer_parent_output_types(
                &[
                    input_type.into(),
                    DimensionValue::constant(1).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(6).unwrap().r#type().into_owned().into(),
                ],
                &[]
            ),
            Ok(vec![output_type.into()]),
        );
    }

    #[test]
    fn test_parallel_all_to_all_type_inference_array_ir() {
        // Explicit dimension inputs supply the tiled result extents.
        let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
        let split = DimensionVariable::new("split", DimensionBounds::unbounded());
        let concatenation = DimensionVariable::new("concatenation", DimensionBounds::unbounded());
        assert_eq!(
            operation.infer_parent_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [4, 3]).into(),
                    DimensionType::from(split.clone()).into(),
                    DimensionType::from(concatenation.clone()).into(),
                ],
                &[]
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![split.clone().into(), concatenation.clone().into()]))
                    .into()
            ]),
        );

        // A known invalid split extent is rejected even when both result extents are dynamic.
        assert_eq!(
            operation.infer_parent_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [3, 3]).into(),
                    DimensionType::from(split.clone()).into(),
                    DimensionType::from(concatenation.clone()).into(),
                ],
                &[]
            ),
            Err(TypeError::invalid("`parallel_all_to_all` split axis 0 size 3 is not divisible by group size 2")),
        );
        assert_eq!(
            operation.infer_parent_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [4, usize::MAX]).into(),
                    DimensionType::from(split).into(),
                    DimensionType::from(concatenation).into(),
                ],
                &[]
            ),
            Err(TypeError::invalid("`parallel_all_to_all` concatenation result extent does not fit in usize")),
        );

        // Static result extents must equal the extents implied by the input and the participant count.
        assert_eq!(
            operation.infer_parent_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [4, 3]).into(),
                    DimensionValue::constant(1).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(6).unwrap().r#type().into_owned().into(),
                ],
                &[]
            ),
            Err(TypeError::invalid(
                "`parallel_all_to_all` split result extent must equal input axis 0 extent 4 divided by group size 2; \
                 expected 2 but got 1",
            )),
        );
        assert_eq!(
            operation.infer_parent_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [4, 3]).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(5).unwrap().r#type().into_owned().into(),
                ],
                &[]
            ),
            Err(TypeError::invalid(
                "`parallel_all_to_all` concatenation result extent must equal input axis 1 extent 3 multiplied by \
                 group size 2; expected 6 but got 5",
            )),
        );

        // A missing array input and out-of-bounds axes are rejected before any result extent is read.
        assert_eq!(
            operation.infer_parent_output_types(&[], &[]),
            Err(TypeError::invalid("`parallel_all_to_all` expects an array followed by its output extents")),
        );
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 2, CollectiveOptions::tiled())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [4, 3]).into(),
                        DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                        DimensionValue::constant(6).unwrap().r#type().into_owned().into(),
                    ],
                    &[]
                ),
            Err(TypeError::invalid(
                "`parallel_all_to_all` split axis 0 or concatenation axis 2 is out of bounds for rank 2"
            )),
        );

        // Coincident tiled axes preserve the shape but still validate the split axis and its divisibility.
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 2, 2, CollectiveOptions::tiled())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [4, 3]).into(),
                        DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                        DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                    ],
                    &[]
                ),
            Err(TypeError::invalid("`parallel_all_to_all` split axis 2 is out of bounds for rank 2")),
        );
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [3, 3]).into(),
                        DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                        DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                    ],
                    &[]
                ),
            Err(TypeError::invalid("`parallel_all_to_all` split axis 0 size 3 is not divisible by group size 2")),
        );
    }

    #[test]
    fn test_parallel_all_to_all_type_inference_array_ir_dynamic() {
        // A tiled exchange of a dynamically shaped composite input takes its result extents from the explicit dimension
        // inputs, while coincident axes preserve the input type, including its dynamic extent.
        let input_axis = DimensionVariable::new("input", DimensionBounds::new(1, Some(17)).unwrap());
        let split_result = DimensionVariable::new("split", DimensionBounds::new(1, Some(9)).unwrap());
        let concatenation_result = DimensionVariable::new("concatenation", DimensionBounds::new(2, Some(33)).unwrap());
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(input_axis.clone()), Dimension::Static(3)]),
        );

        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled())
                .infer_parent_output_types(
                    &[
                        input_type.clone().into(),
                        ArrayIrType::Dimension(DimensionType::from(split_result.clone())),
                        ArrayIrType::Dimension(DimensionType::from(concatenation_result.clone())),
                    ],
                    &[]
                ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(split_result), Dimension::Dynamic(concatenation_result),]),
                )
                .into()
            ]),
        );
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled())
                .infer_parent_output_types(
                    &[
                        ArrayIrType::Array(input_type.clone()),
                        ArrayIrType::Dimension(DimensionType::from(input_axis)),
                        DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                    ],
                    &[]
                ),
            Ok(vec![input_type.into()]),
        );
    }

    #[test]
    fn test_parallel_all_to_all_type_inference_array_ir_untiled() {
        // Untiled exchanges remove the split axis and insert a statically known sender axis at the concatenation
        // position, which may follow, precede, or coincide with the split axis.
        let exact_two = DimensionValue::constant(2).unwrap().r#type().into_owned();
        let exact_three = DimensionValue::constant(3).unwrap().r#type().into_owned();
        let exact_four = DimensionValue::constant(4).unwrap().r#type().into_owned();
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 2, CollectiveOptions::default())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [2, 3, 4]).into(),
                        exact_three.into(),
                        exact_four.into(),
                        exact_two.into(),
                    ],
                    &[]
                ),
            Ok(vec![ArrayType::new_static(DataType::F32, [3, 4, 2]).into()]),
        );
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 4, 1, 0, CollectiveOptions::default())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [2, 4, 3]).into(),
                        DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                        DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                        DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                    ],
                    &[]
                ),
            Ok(vec![ArrayType::new_static(DataType::F32, [4, 2, 3]).into()]),
        );
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 4, 1, 1, CollectiveOptions::default())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [2, 4, 3]).into(),
                        DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                        DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                        DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                    ],
                    &[]
                ),
            Ok(vec![ArrayType::new_static(DataType::F32, [2, 4, 3]).into()]),
        );

        // Untiled geometry retains unaffected dynamic dimensions.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::default())
                .infer_parent_output_types(
                    &[
                        ArrayType::new(DataType::F32, Shape::new(vec![length.clone().into(), 2.into()])).into(),
                        DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                        DimensionType::from(length.clone()).into(),
                    ],
                    &[]
                ),
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![2.into(), length.into()])).into()]),
        );

        // The split axis must exist and have the participant count as its extent, and the inserted sender axis must
        // have that count as its explicit result extent.
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 2, 0, CollectiveOptions::default())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [3, 2]).into(),
                        DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                        DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                    ],
                    &[]
                ),
            Err(TypeError::invalid("`parallel_all_to_all` split axis 2 is out of bounds for rank 2")),
        );
        let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::default());
        assert_eq!(
            operation.infer_parent_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [3, 4]).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
                &[]
            ),
            Err(TypeError::invalid("`parallel_all_to_all` untiled split axis 1 size 4 must equal group size 2")),
        );
        assert_eq!(
            operation.infer_parent_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [3, 2]).into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
                &[]
            ),
            Err(TypeError::invalid(
                "`parallel_all_to_all` inserted output axis 0 extent must equal axis group size 2 but got 3",
            )),
        );
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
    fn test_parallel_all_to_all_interpretation_validation() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();

        // Eager binding validates the shared participant count before it can return an identity value.
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 0, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::Type(TypeError::invalid("`parallel_all_to_all` axis size must be greater than zero"))),
        );

        // The tiled identity rule must also honor the operation-specific axis validation.
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 1, 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_all_to_all` split axis 1 or concatenation axis 0 is out of bounds for rank 1",
            ))),
        );
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Ok(vec![input.clone()]),
        );

        // Outside any binder, only collectives whose instances each combine a single participant are defined. Groups
        // with one participant each therefore qualify even over a larger axis, unlike an ungrouped exchange over it.
        let singleton_groups = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]);
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, singleton_groups),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_all_to_all` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_all_to_all_interpretation_array_ir() {
        // Composite values apply the same degenerate rules to their array input after validating the explicit result
        // extents against the observed input.
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());

        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    0,
                    CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]),
                ),
                Vec::new(),
                &[input.clone(), extent],
            ),
            Ok(vec![input]),
        );
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::tiled()),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())]),
        );

        // A degenerate untiled exchange relocates its size-one split axis to the concatenation position.
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::default()),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::matrix(1, 3, vec![1.0f32, 2.0, 3.0]).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::matrix(3, 1, vec![1.0f32, 2.0, 3.0]).unwrap())]),
        );
    }

    #[test]
    fn test_parallel_all_to_all_partial_evaluation() {
        // A degenerate untiled exchange folds a known input through the corresponding rank-preserving reshape and
        // residualizes an unknown one.
        check_operation_partial_evaluation!(
            operation = ParallelAllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::default()),
            inputs = [Array::matrix(1, 3, vec![1.0f32, 2.0, 3.0]).unwrap()],
            expected = Array::matrix(3, 1, vec![1.0f32, 2.0, 3.0]).unwrap(),
        );

        // An exchange with other participants has no eager per-item value and therefore residualizes.
        let input = Array::matrix(1, 3, vec![1.0f32, 2.0, 3.0]).unwrap();
        let operation = ParallelAllToAllOperation::new("x".to_string(), 3, 1, 0, CollectiveOptions::tiled());
        let program = collective_program(operation.clone(), ArrayType::new_static(DataType::F32, [1, 3]));
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input)]).unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[1, 3] .
                let %1:f32[3, 1] = parallel_all_to_all [axis_name=\"x\", axis_size=3, split_axis=1, \
                concatenation_axis=0, options=Tiled] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_all_to_all_partial_evaluation_staging_parent() {
        // A staging parent can retain a known tracer by staging the collective in the parent trace.
        let operation = ParallelAllToAllOperation::new("x".to_string(), 3, 1, 0, CollectiveOptions::tiled());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = PartialEvaluationValue::known(trace.input(ArrayType::new_static(DataType::F32, [1, 3])));
        let outputs = operation
            .partially_evaluate(&PartialEvaluationContext::new(trace.clone()), &EmptyRegionDriver, &[input])
            .unwrap();
        assert!(outputs[0].is_known());
        assert_eq!(outputs[0].r#type().as_ref(), &ArrayType::new_static(DataType::F32, [3, 1]));

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
                lambda %0:f32[1, 3] .
                let %1:f32[3, 1] = parallel_all_to_all [axis_name=\"x\", axis_size=3, split_axis=1, \
                concatenation_axis=0, options=Tiled] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_all_to_all_batching() {
        // Replicated senders still route destination-specific chunks: each destination receives its chunk twice.
        let tiled = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled());
        let output =
            batch_collective(&tiled, "x", 2, ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()))
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
        let output = batch_collective(
            &ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::default()),
            "x",
            2,
            ArrayBatch::new(input, BatchAxis::new(0)).unwrap(),
        )
        .unwrap()
        .remove(0);
        assert_eq!(output.value().r#type().as_ref(), &ArrayType::new_static(DataType::F64, [2, 3, 2]));
        assert_eq!(
            output.value().elements::<f64>().unwrap(),
            vec![1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0],
        );

        // Here the physical mapped axis follows both logical array axes, and the split axis follows the concatenation
        // axis.
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [3, 2, 2]),
            &[1.0, 7.0, 4.0, 10.0, 2.0, 8.0, 5.0, 11.0, 3.0, 9.0, 6.0, 12.0],
        )
        .unwrap();
        let output = batch_collective(
            &ParallelAllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::default()),
            "x",
            2,
            ArrayBatch::new(input, BatchAxis::new(2)).unwrap(),
        )
        .unwrap()
        .remove(0);
        assert_eq!(output.value().r#type().as_ref(), &ArrayType::new_static(DataType::F64, [2, 2, 3]));
        assert_eq!(
            output.value().elements::<f64>().unwrap(),
            vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
        );

        // Participant groups belong to a mesh exchange and are unsupported when this batch binds the named axis.
        assert_eq!(
            batch_collective(
                &ParallelAllToAllOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    0,
                    CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]),
                ),
                "x",
                2,
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
            batch_collective(
                &tiled.with_mesh(mesh),
                "x",
                2,
                ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]).unwrap()),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_all_to_all` over a manual mesh axis cannot bind a named batch axis".to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_all_to_all_batching_untiled_coincident_axes() {
        // Coincident untiled axes exchange the per-item split axis with the batch axis, so batch item `i` receives
        // element `i` of every item, in sender order.
        let output = batch_collective(
            &ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::default()),
            "x",
            2,
            ArrayBatch::new(Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap(),
        )
        .unwrap()
        .remove(0);
        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        assert_eq!(output.value(), &Array::matrix(2, 2, vec![1.0f32, 3.0, 2.0, 4.0]).unwrap());
    }

    #[test]
    fn test_parallel_all_to_all_batching_rejects_ragged_inputs() {
        // A matching level rejects bounded ragged inputs in both array families, because one extent per item does not
        // determine how each sender partitions its live prefix among the receivers.
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
        assert_eq!(
            batch_collective(
                &ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled()),
                "x",
                2,
                input,
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
    fn test_parallel_all_to_all_batching_forwards_unbound_axis() {
        // A non-matching batch level shifts both array axes around its mapped dimension. Removing the split axis can
        // move that mapped dimension, and insertion at the concatenation axis can move it a second time: a leading
        // mapped dimension stays in front of both array axes, removing the split axis in front of the mapped dimension
        // moves that dimension to the front, and a trailing mapped dimension stays behind both array axes.
        let array = |dimensions: [usize; 3]| {
            Array::from_elements::<f32>(
                ArrayType::new_static(DataType::F32, dimensions),
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .unwrap()
        };
        check_operation_batching!(
            @exact,
            operation = ParallelAllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::default()),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@mapped(axis = 0), array([2, 1, 3]))],
                    outputs = [(@mapped(axis = 0), array([2, 3, 1]))],
                },
                {
                    inputs = [(@mapped(axis = 1), array([1, 2, 3]))],
                    outputs = [(@mapped(axis = 0), array([2, 3, 1]))],
                },
                {
                    inputs = [(@mapped(axis = 2), array([1, 3, 2]))],
                    outputs = [(@mapped(axis = 2), array([3, 1, 2]))],
                },
            ],
        );
    }

    #[test]
    fn test_parallel_all_to_all_batching_forwards_unbound_axis_validates_manual_inputs() {
        // An exchange over a manual mesh axis forwarded through an unrelated batch level rejects invariant inputs and
        // pending mesh sums at that level, before forwarding. Bind the operation directly so the capability cannot
        // insert `parallel_vary` first.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 3);
        let forward = |input_sharding: Sharding| {
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| {
                    let parent = input.dispatch_domain();
                    let context =
                        BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), parent.dimension_constant(2)?)
                            .with_axis_name("y".to_string());
                    let input = ArrayIrBatch::new(input, BatchAxis::new(0))?;
                    let split_extent = ArrayIrBatch::replicated(parent.dimension_constant(1)?);
                    let concatenation_extent = ArrayIrBatch::replicated(parent.dimension_constant(6)?);
                    let operation =
                        ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled())
                            .with_mesh(mesh.clone());
                    let mut outputs = operation
                        .batch_in_parent(&context, &EmptyRegionDriver, &[input, split_extent, concatenation_extent])?
                        .into_parts()
                        .0;
                    Ok(outputs.remove(0).into_value())
                },
                ArrayIrType::Array(
                    ArrayType::new_static(DataType::F32, [2, 2, 3]).with_sharding(input_sharding).unwrap(),
                ),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap_err()
        };

        // An invariant input would type the exchanged output as invariant.
        assert_eq!(
            forward(sharding.clone()).downcast_custom::<BatchingError>(),
            Some(&BatchingError::Type(TypeError::invalid(
                "`parallel_all_to_all` input must vary over manual axis `x`; pass an invariant value through \
                 `parallel_vary` first so that the exchanged output is typed as varying",
            ))),
        );

        // A pending sum over the exchanged mesh axis is rejected.
        assert_eq!(
            forward(sharding.with_unreduced_axes(["x"]).unwrap()).downcast_custom::<BatchingError>(),
            Some(&BatchingError::Type(TypeError::invalid("`parallel_all_to_all` does not support unreduced inputs"))),
        );
    }

    #[test]
    fn test_parallel_all_to_all_batching_shadows_manual_axis() {
        // The inner batch named `x` exchanges local chunks independently of the manual mesh axis also named `x`, so the
        // exchange neither rejects nor makes varying an input with a pending sum over the shadowed mesh axis, and the
        // local exchange preserves that pending sum.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 3).with_unreduced_axes(["x"]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2, 2, 3]).with_sharding(sharding.clone()).unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [2, 1, 6]).with_sharding(sharding).unwrap();
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| {
                let context = BatchingContext::<_, ArrayBatchingPolicy>::new(input.dispatch_domain(), 2)
                    .with_axis_name("x".to_string());
                let input = ArrayBatch::new(input, BatchAxis::new(0))?;
                let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
                let mut outputs = operation.batch(&context, &EmptyRegionDriver, &[input])?.into_parts().0;
                Ok(outputs.remove(0).into_value())
            },
            input_type.clone(),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(output_type, expected_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], unreduced={'x'}}] .
                let %1:f32[2, 2, 1, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                        reshape [shape=[2, 2, 1, 3]] %0
                    %2:f32[2, 2, 1, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                        transpose [permutation=[1, 0, 2, 3]] %1
                    %3:f32[2, 1, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                        transpose [permutation=[0, 2, 1, 3]] %2
                    %4:f32[2, 1, 6][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], unreduced={'x'}}] = \
                        reshape [
                        shape=[2, 1, 6],
                        output_sharding={mesh<['x'=2:manual]>, [{}, {}, {}], unreduced={'x'}},
                    ] %3
                in (%4)"
            },
        );

        // The composite family's explicit result extents use the same local batching rule.
        let (output_type, program) =
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
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], unreduced={'x'}}] .
                let %1:dimension<2> = constant [value=2]
                    %2:dimension<2> = constant [value=2]
                    %3:dimension<3> = constant [value=3]
                    %4:dimension<2> = constant [value=2]
                    %5:dimension<1> = dimension_div %2 %4
                    %6:dimension<6> = dimension_mul %3 %4
                    %7:dimension<2> = constant [value=2]
                    %8:bool[] = const true
                    %9:dimension<3> = dimension_div %6 %1
                    %10:f32[2, 2, 1, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                        reshape %0 %1 %1 %5 %9
                    %11:f32[2, 2, 1, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                transpose [permutation=[1, 0, 2, 3]] %10
                    %12:f32[2, 1, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                transpose [permutation=[0, 2, 1, 3]] %11
                    %13:f32[2, 1, 6][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], unreduced={'x'}}] = \
                        reshape [output_sharding={mesh<['x'=2:manual]>, [{}, {}, {}], unreduced={'x'}}] %12 %1 %5 %6
                in (%13)"
            },
        );
    }

    #[test]
    fn test_parallel_all_to_all_batching_shadows_manual_axis_through_unrelated_batch() {
        // The unrelated inner batch forwards to the outer batch named `x`. That outer batch shadows the mesh axis,
        // so forwarding must leave mesh validation to the level that handles the exchange, which accepts and
        // preserves the pending sum over the shadowed mesh axis.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 4).with_unreduced_axes(["x"]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2, 2, 2, 3]).with_sharding(sharding.clone()).unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [2, 2, 1, 6]).with_sharding(sharding).unwrap();
        let (output_type, program) =
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
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] .
                let %1:dimension<2> = constant [value=2]
                    %2:dimension<2> = constant [value=2]
                    %3:dimension<2> = constant [value=2]
                    %4:dimension<3> = constant [value=3]
                    %5:dimension<2> = constant [value=2]
                    %6:dimension<1> = dimension_div %3 %5
                    %7:dimension<6> = dimension_mul %4 %5
                    %8:dimension<2> = constant [value=2]
                    %9:bool[] = const true
                    %10:dimension<3> = dimension_div %7 %1
                    %11:f32[2, 2, 2, 1, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}, {}], unreduced={'x'}}] = \
                reshape %0 %1 %2 %1 %6 %10
                    %12:f32[2, 2, 2, 1, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}, {}], unreduced={'x'}}] = \
                transpose [permutation=[2, 1, 0, 3, 4]] %11
                    %13:f32[2, 2, 1, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}, {}], unreduced={'x'}}] = \
                transpose [permutation=[0, 1, 3, 2, 4]] %12
                    %14:f32[2, 2, 1, 6][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                        reshape [\
                            output_sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}\
                        ] %13 %1 %2 %6 %7
                in (%14)"
            },
        );
    }

    #[test]
    fn test_parallel_all_to_all_batching_array_ir_coincident_axes() {
        // Block exchange with `split_axis == concatenation_axis == 0`: each item splits its vector into two chunks, and
        // item `i` receives chunk `i` of every item, concatenated in sender order. With items `[1, 2, 3, 4]` and `[5,
        // 6, 7, 8]`, item 0 receives `[1, 2, 5, 6]` and item 1 receives `[3, 4, 7, 8]`.
        let input = Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let output: ArrayIrValue<Array> = batch(
            |item: BatchingTracer<
                EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                ArrayIrBatchingPolicy,
            >| { item.parallel_all_to_all_tiled("x", 0, 0) },
            ArrayIrValue::Array(input),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output,
            ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]).unwrap()),
        );
    }

    #[test]
    fn test_parallel_all_to_all_batching_array_ir_distinct_axes() {
        // Distinct split and concatenation axes over per-item `[2, 2]` matrices: each item splits its rows across
        // the items and receives its own row index from every item, concatenated item-major along the columns. With
        // item 0 = `[[1, 2], [3, 4]]` and item 1 = `[[5, 6], [7, 8]]`, item 0 receives `[[1, 2, 5, 6]]` and item 1
        // receives `[[3, 4, 7, 8]]` (per-item shape `[1, 4]`).
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [2, 2, 2]),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let output: ArrayIrValue<Array> = batch(
            |item: BatchingTracer<
                EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                ArrayIrBatchingPolicy,
            >| { item.parallel_all_to_all_tiled("x", 0, 1) },
            ArrayIrValue::Array(input),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output,
            ArrayIrValue::Array(
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [2, 1, 4]),
                    &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0],
                )
                .unwrap(),
            ),
        );
    }

    #[test]
    fn test_parallel_all_to_all_batching_array_ir_dynamic_extents() {
        // Distinct-axis all-to-all derives its temporary pre-exchange shape from the supplied result extents and the
        // mapped extent using ordinary dimension arithmetic; it never reads the source array shape.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch_dimension = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(batch_dimension.clone()).into());
        let input_split = DimensionVariable::new("input_split", DimensionBounds::new(1, Some(65)).unwrap());
        let input_concatenation =
            DimensionVariable::new("input_concatenation", DimensionBounds::new(1, Some(65)).unwrap());
        let output_split = DimensionVariable::new("output_split", DimensionBounds::new(1, Some(65)).unwrap());
        let output_concatenation =
            DimensionVariable::new("output_concatenation", DimensionBounds::new(1, Some(129)).unwrap());
        let input = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![
                    Dimension::Dynamic(batch_dimension),
                    Dimension::Dynamic(input_split),
                    Dimension::Dynamic(input_concatenation),
                ]),
            )
            .into(),
        );
        let output_split = trace.input(DimensionType::from(output_split).into());
        let output_concatenation = trace.input(DimensionType::from(output_concatenation).into());
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
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(input, BatchAxis::new(0)).unwrap()),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(output_split)),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(output_concatenation)),
                ],
            )
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
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
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, input_split, input_concatenation], \
                    %2:dimension<output_split ∈ [1, 65)>, %3:dimension<output_concatenation ∈ [1, 129)> .
                let %4:dimension<4> = constant [value=4]
                    %5:bool[] = compare [direction=Equal] %0 %4
                    () = assert [
                        message=\"collective axis extent must match the participant count\",
                        labels=[\"extent\", \"participants\"],
                    ] %5 %0 %4
                    %6:dimension<0> = constant [value=0]
                    %7:dimension<output_concatenation % batch ∈ [0, 8)> = dimension_rem %3 %0
                    %8:bool[] = compare [direction=Equal] %7 %6
                    () = assert [
                        message=\"collective extent must be divisible by the participant count\",
                        labels=[\"extent\", \"divisor\"],
                    ] %8 %3 %0
                    %9:dimension<output_concatenation / batch ∈ [0, 129)> = dimension_div %3 %0
                    %10:dimension<output_split * batch ∈ [1, 513)> = dimension_mul %2 %0
                    %11:f32[batch, batch, output_split, output_concatenation / batch] = reshape %1 %0 %0 %2 %9
                    %12:f32[batch, batch, output_split, output_concatenation / batch] = \
                        transpose [permutation=[1, 0, 2, 3]] %11
                    %13:f32[batch, output_split, batch, output_concatenation / batch] = \
                        transpose [permutation=[0, 2, 1, 3]] %12
                    %14:f32[batch, output_split, output_concatenation] = reshape %13 %0 %2 %3
                in (%14)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_parallel_all_to_all_differentiation() {
        // The homogeneous JVP applies exactly the same exchange to the primal and live tangent.
        let program = collective_program(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()),
            ArrayType::new_static(DataType::F32, [4, 3]),
        );
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[4, 3], %1:f32[4, 3] .
                let %2:f32[2, 6] = parallel_all_to_all [\
                        axis_name=\"x\", \
                        axis_size=2, \
                        split_axis=0, \
                        concatenation_axis=1, \
                        options=Tiled\
                    ] %0
                    %3:f32[2, 6] = parallel_all_to_all [\
                        axis_name=\"x\", \
                        axis_size=2, \
                        split_axis=0, \
                        concatenation_axis=1, \
                        options=Tiled\
                    ] %1
                in (%2, %3)"
            },
        );
    }

    #[test]
    fn test_parallel_all_to_all_differentiation_weighted_gradient() {
        // Distinct destination and sender weights catch an incorrectly ordered exchange or pullback. The physical
        // mapped axis follows the split and concatenation axes, so differentiation must also retain their logical
        // positions.
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
    }

    #[test]
    fn test_parallel_all_to_all_differentiation_shadows_manual_axis() {
        // An inner batch named `x` performs a local exchange despite an enclosing mesh axis with that name.
        // Differentiation must retain structural zeros and accept and preserve the pending sum over the shadowed mesh
        // axis.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 3).with_unreduced_axes(["x"]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2, 2, 3]).with_sharding(sharding.clone()).unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [2, 1, 6]).with_sharding(sharding).unwrap();
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| {
                let batch_context = BatchingContext::<_, ArrayBatchingPolicy>::new(input.dispatch_domain(), 2)
                    .with_axis_name("x".to_string());
                let item = BatchingTracer::new(batch_context.clone(), ArrayBatch::new(input, BatchAxis::new(0))?);
                let context = DifferentiationContext::fused(batch_context);
                let item =
                    DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(item)?, context.clone());
                let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
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
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], unreduced={'x'}}] .
                let %1:f32[2, 2, 1, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                    reshape [shape=[2, 2, 1, 3]] %0
                    %2:f32[2, 2, 1, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                transpose [permutation=[1, 0, 2, 3]] %1
                    %3:f32[2, 1, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                transpose [permutation=[0, 2, 1, 3]] %2
                    %4:f32[2, 1, 6][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], unreduced={'x'}}] = reshape [
                        shape=[2, 1, 6],
                        output_sharding={mesh<['x'=2:manual]>, [{}, {}, {}], unreduced={'x'}},
                    ] %3
                in (%4)"
            },
        );

        // The composite capability stages result extents with structural-zero tangents and uses the same binder.
        let (output_type, program) =
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| {
                    let axis_extent = input.dispatch_domain().dimension_constant(2)?;
                    let batch_context =
                        BatchingContext::<_, ArrayIrBatchingPolicy>::new(input.dispatch_domain(), axis_extent)
                            .with_axis_name("x".to_string());
                    let item = BatchingTracer::new(batch_context.clone(), ArrayIrBatch::new(input, BatchAxis::new(0))?);
                    let context = DifferentiationContext::fused(batch_context);
                    let item = DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(item)?, context);
                    let output = item.parallel_all_to_all_tiled("x", 0, 1)?;
                    assert!(output.tangent().is_zero());
                    Ok(output.primal().clone().into_batch().into_value())
                },
                ArrayIrType::Array(input_type),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(expected_type));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], unreduced={'x'}}] .
                let %1:dimension<2> = constant [value=2]
                    %2:dimension<2> = constant [value=2]
                    %3:dimension<3> = constant [value=3]
                    %4:dimension<2> = constant [value=2]
                    %5:dimension<1> = dimension_div %2 %4
                    %6:dimension<6> = dimension_mul %3 %4
                    %7:dimension<2> = constant [value=2]
                    %8:bool[] = const true
                    %9:dimension<3> = dimension_div %6 %1
                    %10:f32[2, 2, 1, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                        reshape %0 %1 %1 %5 %9
                    %11:f32[2, 2, 1, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                transpose [permutation=[1, 0, 2, 3]] %10
                    %12:f32[2, 1, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}, {}], unreduced={'x'}}] = \
                transpose [permutation=[0, 2, 1, 3]] %11
                    %13:f32[2, 1, 6][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], unreduced={'x'}}] = \
                        reshape [output_sharding={mesh<['x'=2:manual]>, [{}, {}, {}], unreduced={'x'}}] %12 %1 %5 %6
                in (%13)"
            },
        );
    }

    #[test]
    fn test_parallel_all_to_all_differentiation_array_ir_zero_tangent() {
        // Structural zeros use the inferred result shape and stage no linear call or tangent exchange.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let split_extent = trace.input(DimensionValue::constant(3).unwrap().r#type().into_owned().into());
        let concatenation_extent = trace.input(DimensionValue::constant(2).unwrap().r#type().into_owned().into());
        let inputs = [input, split_extent, concatenation_extent]
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
            &ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 2])),
        );
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.primal().atom_id().unwrap()],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:dimension<3>, %2:dimension<2> .
                let %3:f32[3, 2] = parallel_all_to_all [\
                    axis_name=\"x\", \
                    axis_size=2, \
                    split_axis=0, \
                    concatenation_axis=1, \
                    options=Untiled\
                ] %0 %1 %2
                in (%3)"
            },
        );
    }

    #[test]
    fn test_parallel_all_to_all_differentiation_array_ir_linearization() {
        // A composite exchange linearizes into a linear call that retains its explicit result extent as a residual,
        // and whose transpose applies the adjoint exchange to the output cotangent.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(
                ParallelAllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![array, extent],
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
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:dimension<3> = const 3
                    %2:f32[3] = parallel_all_to_all [\
                        axis_name=\"x\", \
                        axis_size=1, \
                        split_axis=0, \
                        concatenation_axis=0, \
                        options=Tiled\
                    ] %0 %1
                in (%2)"
            },
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:dimension<3> = const 3
                    %2:f32[3] = linear_call [residual_count=1] %1 %0 [
                        forward={
                            lambda %0:dimension<3>, %1:f32[3] .
                            let %2:f32[3] = parallel_all_to_all [\
                                axis_name=\"x\", \
                                axis_size=1, \
                                split_axis=0, \
                                concatenation_axis=0, \
                                options=Tiled\
                            ] %1 %0
                            in (%2)
                        },
                        transpose={
                            lambda %0:dimension<3>, %1:f32[3] .
                            let %2:dimension<3> = constant [value=3]
                                %3:f32[3] = parallel_all_to_all [\
                                    axis_name=\"x\", \
                                    axis_size=1, \
                                    split_axis=0, \
                                    concatenation_axis=0, \
                                    options=Tiled\
                                ] %1 %2
                            in (%3)
                        },
                    ]
                in (%2)"
            },
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:dimension<3> = const 3
                    %2:f32[3] = linear_call [residual_count=1] %1 %0 [
                        forward={
                            lambda %0:dimension<3>, %1:f32[3] .
                            let %2:dimension<3> = constant [value=3]
                                %3:f32[3] = parallel_all_to_all [\
                                    axis_name=\"x\", \
                                    axis_size=1, \
                                    split_axis=0, \
                                    concatenation_axis=0, \
                                    options=Tiled\
                                ] %1 %2
                            in (%3)
                        },
                        transpose={
                            lambda %0:dimension<3>, %1:f32[3] .
                            let %2:f32[3] = parallel_all_to_all [\
                                axis_name=\"x\", \
                                axis_size=1, \
                                split_axis=0, \
                                concatenation_axis=0, \
                                options=Tiled\
                            ] %1 %0
                            in (%2)
                        },
                    ]
                in (%2)"
            },
        );
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap());
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
    }

    #[test]
    fn test_parallel_all_to_all_differentiation_array_ir_dynamic_extent() {
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
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[length, 1], %1:dimension<length ∈ [1, 9)> .
                let %2:dimension<1> = const 1
                    %3:f32[1, length] = parallel_all_to_all [\
                        axis_name=\"x\", \
                        axis_size=1, \
                        split_axis=1, \
                        concatenation_axis=0, \
                        options=Untiled\
                    ] %0 %2 %1
                in (%3, %1)"
            },
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[length, 1], %1:dimension<length ∈ [1, 9)> .
                let %2:dimension<1> = const 1
                    %3:f32[1, length] = linear_call [residual_count=2] %2 %1 %0 [
                        forward={
                            lambda %0:dimension<1>, %1:dimension<length ∈ [1, 9)>, %2:f32[length, 1] .
                            let %3:f32[1, length] = parallel_all_to_all [\
                                axis_name=\"x\", \
                                axis_size=1, \
                                split_axis=1, \
                                concatenation_axis=0, \
                                options=Untiled\
                            ] %2 %0 %1
                            in (%3)
                        },
                        transpose={
                            lambda %0:dimension<1>, %1:dimension<length ∈ [1, 9)>, %2:f32[1, length] .
                            let %3:dimension<1> = constant [value=1]
                                %4:f32[length, 1] = parallel_all_to_all [\
                                    axis_name=\"x\", \
                                    axis_size=1, \
                                    split_axis=0, \
                                    concatenation_axis=1, \
                                    options=Untiled\
                                ] %2 %1 %3
                            in (%4)
                        },
                    ]
                in (%3)"
            },
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[1, length], %1:dimension<length ∈ [1, 9)> .
                let %2:dimension<1> = const 1
                    %3:f32[length, 1] = linear_call [residual_count=2] %2 %1 %0 [
                        forward={
                            lambda %0:dimension<1>, %1:dimension<length ∈ [1, 9)>, %2:f32[1, length] .
                            let %3:dimension<1> = constant [value=1]
                                %4:f32[length, 1] = parallel_all_to_all [\
                                    axis_name=\"x\", \
                                    axis_size=1, \
                                    split_axis=0, \
                                    concatenation_axis=1, \
                                    options=Untiled\
                                ] %2 %1 %3
                            in (%4)
                        },
                        transpose={
                            lambda %0:dimension<1>, %1:dimension<length ∈ [1, 9)>, %2:f32[length, 1] .
                            let %3:f32[1, length] = parallel_all_to_all [\
                                axis_name=\"x\", \
                                axis_size=1, \
                                split_axis=1, \
                                concatenation_axis=0, \
                                options=Untiled\
                            ] %2 %0 %1
                            in (%3)
                        },
                    ]
                in (%3)"
            },
        );

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
    fn test_parallel_all_to_all_transposition() {
        // Both modes transpose by swapping the split and concatenation axes, and transposing twice restores the
        // original program.
        let program = collective_program(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()),
            ArrayType::new_static(DataType::F32, [2, 3]),
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[1, 6] .
                let %1:f32[2, 3] = parallel_all_to_all [\
                    axis_name=\"x\", \
                    axis_size=2, \
                    split_axis=1, \
                    concatenation_axis=0, \
                    options=Tiled\
                ] %0
                in (%1)"
            },
        );
        assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());

        let program = collective_program(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::default()),
            ArrayType::new_static(DataType::F32, [2, 3]),
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[3, 2] .
                let %1:f32[2, 3] = parallel_all_to_all [\
                    axis_name=\"x\", \
                    axis_size=2, \
                    split_axis=1, \
                    concatenation_axis=0, \
                    options=Untiled\
                ] %0
                in (%1)"
            },
        );
        assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());

        // The transpose retains the ordered participant groups in both modes.
        let program = collective_program(
            ParallelAllToAllOperation::new(
                "x".to_string(),
                4,
                0,
                1,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
            ),
            ArrayType::new_static(DataType::F32, [2, 3]),
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[1, 6] .
                let %1:f32[2, 3] = parallel_all_to_all [
                    axis_name=\"x\",
                    axis_size=4,
                    split_axis=1,
                    concatenation_axis=0,
                    options=Tiled,
                    axis_index_groups=[[0, 2], [3, 1]],
                ] %0
                in (%1)"
            },
        );
        assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());

        let program = collective_program(
            ParallelAllToAllOperation::new(
                "x".to_string(),
                4,
                0,
                1,
                CollectiveOptions::default().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
            ),
            ArrayType::new_static(DataType::F32, [2, 3]),
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[3, 2] .
                let %1:f32[2, 3] = parallel_all_to_all [
                    axis_name=\"x\",
                    axis_size=4,
                    split_axis=1,
                    concatenation_axis=0,
                    options=Untiled,
                    axis_index_groups=[[0, 2], [3, 1]],
                ] %0
                in (%1)"
            },
        );
        assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());

        // An exchange over a manual mesh axis transposes over the same mesh.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
        let program = collective_program(
            operation.clone().with_mesh(mesh.clone()),
            ArrayType::new_static(DataType::F32, [2, 3])
                .with_sharding(Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["x"]).unwrap())
                .unwrap(),
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[1, 6][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                let %1:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                parallel_all_to_all [
                    axis_name=\"x\",
                    axis_size=2,
                    split_axis=1,
                    concatenation_axis=0,
                    options=Tiled,
                    mesh=['x'=2:manual],
                ] %0
                in (%1)"
            },
        );
        assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());
    }

    #[test]
    fn test_parallel_all_to_all_infer_parent_region_input_types() {
        // The exchange has no regions, so it requests no region input types from its composite parent.
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled())
                .infer_parent_region_input_types(&[ArrayType::new_static(DataType::F32, [4, 3]).into()], &[]),
            Ok(Vec::new()),
        );
    }

    #[test]
    fn test_parallel_all_to_all_rename_parent_type_identities() {
        // The exchange records no type identities of its own, because its result extents are explicit inputs of its
        // composite parent, so renaming the identities of that parent leaves it unchanged.
        let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled());
        let mut renaming = TypeIdentityRenaming::new();
        renaming
            .insert(
                DimensionVariable::new("source", DimensionBounds::unbounded()),
                DimensionVariable::new("target", DimensionBounds::unbounded()),
            )
            .unwrap();
        assert_eq!(operation.rename_parent_type_identities(&renaming), Ok(operation));
    }

    #[test]
    fn test_parallel_all_to_all_parallel_all_to_all() {
        // An untiled exchange over a `batch` level swaps the per-item split axis with the batch axis, so batch item
        // `j` receives element `j` of every item.
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    item.parallel_all_to_all("x", 0, 0)
                },
                Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 4.0]).unwrap()),
        );

        // Concrete arrays, and concrete composite values through their array members, are never inside an axis
        // binder, so every axis name is unbound for them.
        assert_eq!(
            Array::vector(vec![1.0, 2.0]).unwrap().parallel_all_to_all("x", 0, 0),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
        assert_eq!(
            ArrayIrValue::Array(Array::vector(vec![1.0, 2.0]).unwrap()).parallel_all_to_all("x", 0, 0),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
    }

    #[test]
    fn test_parallel_all_to_all_parallel_all_to_all_tiled() {
        // Homogeneous array values stage the static-shape operation without explicit result extents. Over a manual
        // mesh axis, a varying input is exchanged directly and the staged operation records the mesh.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_all_to_all_tiled("x", 0, 0),
            varying_type,
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_all_to_all [
                    axis_name=\"x\",
                    axis_size=2,
                    split_axis=0,
                    concatenation_axis=0,
                    options=Tiled,
                    mesh=['x'=2:manual],
                ] %0
                in (%1)"
            },
        );

        // Over a `batch` level, batch item `j` receives chunk `j` of every item, concatenated in sender order.
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    item.parallel_all_to_all_tiled("x", 0, 0)
                },
                Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::matrix(2, 4, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]).unwrap()),
        );
    }

    #[test]
    fn test_parallel_all_to_all_parallel_all_to_all_with_options() {
        // The staged exchange records the ordered participant groups of `options`, and its geometry uses their common
        // group size.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let options = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        let varying_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_all_to_all_with_options("x", 0, 0, options.clone()),
            varying_type.clone(),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 4, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=4:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[4][sharding={mesh<['x'=4:manual]>, [{}], varying_manual={'x'}}] = parallel_all_to_all [
                    axis_name=\"x\",
                    axis_size=4,
                    split_axis=0,
                    concatenation_axis=0,
                    options=Tiled,
                    axis_index_groups=[[0, 2], [3, 1]],
                    mesh=['x'=4:manual],
                ] %0
                in (%1)"
            },
        );

        // Over a manual mesh axis, an invariant input is first made varying, so the exchanged output is varying.
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_all_to_all_with_options("x", 0, 0, options.clone()),
                ArrayType::new_static(DataType::F32, [4])
                    .with_sharding(Sharding::replicated(mesh.clone(), 1))
                    .unwrap(),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 4, mesh: mesh.clone() })],
            )
            .map(|(output_type, _)| output_type),
            Ok(varying_type),
        );

        // A pending cross-device sum over the exchanged manual axis is rejected before anything is staged.
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_all_to_all_with_options("x", 0, 0, options.clone()),
                ArrayType::new_static(DataType::F32, [4])
                    .with_sharding(Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
                    .unwrap(),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 4, mesh })],
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid("`parallel_all_to_all` does not support unreduced inputs"))),
        );

        // A traced value outside any binder of the axis name is rejected in both array families.
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_all_to_all_with_options("x", 0, 0, options.clone()),
                ArrayType::new_static(DataType::F32, [4]),
                Vec::new(),
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
        assert_eq!(
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_all_to_all_with_options("x", 0, 0, options.clone()),
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
                Vec::new(),
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
    }

    #[test]
    fn test_parallel_all_to_all_parallel_all_to_all_with_options_negative_axes() {
        // Negative split and concatenation axes count from the end. An exchange preserves the rank, so both axes are
        // normalized against the rank of the input in both modes. Homogeneous and composite values normalize the axes
        // alike, and composite values do so before they stage any result extent.
        let named_axes = || vec![("x".to_string(), NamedAxis::Batched { size: Some(2) })];
        let homogeneous =
            |input_type: ArrayType, split_axis: i32, concatenation_axis: i32, options: CollectiveOptions| {
                TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                    move |input| input.parallel_all_to_all_with_options("x", split_axis, concatenation_axis, options),
                    input_type,
                    named_axes(),
                )
                .map(|(_, program)| program.to_string())
            };
        let composite =
            |input_type: ArrayType, split_axis: i32, concatenation_axis: i32, options: CollectiveOptions| {
                TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                    move |input| input.parallel_all_to_all_with_options("x", split_axis, concatenation_axis, options),
                    ArrayIrType::Array(input_type),
                    named_axes(),
                )
                .map(|(_, program)| program.to_string())
            };
        let untiled_type = || ArrayType::new_static(DataType::F32, [3, 4, 2]);
        let tiled_type = || ArrayType::new_static(DataType::F32, [4, 6]);
        let untiled = CollectiveOptions::default;
        let tiled = CollectiveOptions::tiled;
        assert_eq!(
            homogeneous(untiled_type(), -1, -3, untiled()).unwrap(),
            homogeneous(untiled_type(), 2, 0, untiled()).unwrap(),
        );
        assert_eq!(
            homogeneous(tiled_type(), -2, -1, tiled()).unwrap(),
            homogeneous(tiled_type(), 0, 1, tiled()).unwrap(),
        );
        assert_eq!(
            composite(untiled_type(), -1, -3, untiled()).unwrap(),
            composite(untiled_type(), 2, 0, untiled()).unwrap(),
        );
        assert_eq!(composite(tiled_type(), -2, -1, tiled()).unwrap(), composite(tiled_type(), 0, 1, tiled()).unwrap());
        assert_eq!(
            homogeneous(untiled_type(), -4, 0, untiled()),
            Err(ProgramError::Axis(AxisError::OutOfBounds { axis: Axis::from(-4), rank: 3 })),
        );
        assert_eq!(
            composite(tiled_type(), 0, -3, tiled()),
            Err(ProgramError::Axis(AxisError::OutOfBounds { axis: Axis::from(-3), rank: 2 })),
        );
    }

    #[test]
    fn test_parallel_all_to_all_parallel_all_to_all_with_options_preserves_unrelated_pending_sums() {
        // The composite capability makes an invariant input varying over the exchanged axis `x` without dropping a
        // pending sum over the independent `y` axis, but rejects a pending sum over `x` itself.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 2).with_unreduced_axes(["y"]).unwrap();
        assert_eq!(
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_all_to_all_with_options("x", 0, 1, CollectiveOptions::tiled()),
                ArrayIrType::Array(
                    ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(sharding.clone()).unwrap(),
                ),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .map(|(output_type, _)| output_type),
            Ok(ArrayIrType::Array(
                ArrayType::new_static(DataType::F32, [1, 6])
                    .with_sharding(sharding.with_varying_manual_axes(["x"]).unwrap())
                    .unwrap(),
            )),
        );
        assert_eq!(
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_all_to_all_with_options("x", 0, 1, CollectiveOptions::tiled()),
                ArrayIrType::Array(
                    ArrayType::new_static(DataType::F32, [2, 3])
                        .with_sharding(Sharding::replicated(mesh.clone(), 2).with_unreduced_axes(["x"]).unwrap())
                        .unwrap(),
                ),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid("`parallel_all_to_all` does not support unreduced inputs"))),
        );
    }

    #[test]
    fn test_parallel_all_to_all_parallel_all_to_all_with_options_array_ir_extents() {
        // Static group-local requirements are rejected while staging, while dynamic ones remain runtime assertions,
        // including when coincident axes preserve the shape.
        let untiled = CollectiveOptions::default().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        let tiled = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        let static_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 3]));
        let dynamic_type = ArrayIrType::Array(ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                DimensionVariable::new("length", DimensionBounds::new(1, Some(9)).unwrap()).into(),
                Dimension::Static(3),
            ]),
        ));

        // An untiled split extent must equal the group size.
        assert_eq!(
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_all_to_all_with_options("x", 0, 1, untiled.clone()),
                static_type.clone(),
                vec![("x".to_string(), NamedAxis::Batched { size: Some(4) })],
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_all_to_all` untiled split axis 0 size 3 must equal group size 2",
            ))),
        );
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_all_to_all_with_options("x", 0, 1, untiled.clone()),
            dynamic_type.clone(),
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
                    %5:dimension<2> = constant [value=2]
                    %6:f32[3, 2] = parallel_all_to_all [
                        axis_name=\"x\",
                        axis_size=4,
                        split_axis=0,
                        concatenation_axis=1,
                        options=Untiled,
                        axis_index_groups=[[0, 2], [3, 1]],
                    ] %0 %4 %5
                in (%6)"
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

        // A tiled split extent must be divisible by the group size.
        assert_eq!(
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_all_to_all_with_options("x", 0, 1, tiled.clone()),
                static_type.clone(),
                vec![("x".to_string(), NamedAxis::Batched { size: Some(4) })],
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_all_to_all` split axis 0 size 3 is not divisible by group size 2",
            ))),
        );
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_all_to_all_with_options("x", 0, 1, tiled.clone()),
            dynamic_type.clone(),
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
                    %8:dimension<6> = dimension_mul %2 %3
                    %9:f32[length / 2, 6] = parallel_all_to_all [
                        axis_name=\"x\",
                        axis_size=4,
                        split_axis=0,
                        concatenation_axis=1,
                        options=Tiled,
                        axis_index_groups=[[0, 2], [3, 1]],
                    ] %0 %7 %8
                in (%9)"
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

        // Coincident tiled axes preserve the shape but still require a divisible split extent.
        assert_eq!(
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_all_to_all_with_options("x", 0, 0, tiled.clone()),
                static_type,
                vec![("x".to_string(), NamedAxis::Batched { size: Some(4) })],
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_all_to_all` split axis 0 size 3 is not divisible by group size 2",
            ))),
        );
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_all_to_all_with_options("x", 0, 0, tiled.clone()),
            dynamic_type,
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
                    %7:f32[length, 3] = parallel_all_to_all [
                        axis_name=\"x\",
                        axis_size=4,
                        split_axis=0,
                        concatenation_axis=0,
                        options=Tiled,
                        axis_index_groups=[[0, 2], [3, 1]],
                    ] %0 %1 %2
                in (%7)"
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

        // An untiled exchange omits its consumed static input axis from the explicit result extents.
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_all_to_all_with_options("x", 0, 1, CollectiveOptions::default()),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 3])),
            vec![("x".to_string(), NamedAxis::Batched { size: Some(2) })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:dimension<3> = constant [value=3]
                    %2:dimension<2> = constant [value=2]
                    %3:f32[3, 2] = parallel_all_to_all [\
                        axis_name=\"x\", \
                        axis_size=2, \
                        split_axis=0, \
                        concatenation_axis=1, \
                        options=Untiled\
                    ] %0 %1 %2
                in (%3)"
            },
        );

        // Coincident static tiled axes need no extent arithmetic.
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_all_to_all_with_options("x", 0, 0, CollectiveOptions::tiled()),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4, 3])),
            vec![("x".to_string(), NamedAxis::Batched { size: Some(2) })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 3] .
                let %1:dimension<4> = constant [value=4]
                    %2:dimension<3> = constant [value=3]
                    %3:f32[4, 3] = parallel_all_to_all [\
                        axis_name=\"x\", \
                        axis_size=2, \
                        split_axis=0, \
                        concatenation_axis=0, \
                        options=Tiled\
                    ] %0 %1 %2
                in (%3)"
            },
        );
    }

    #[test]
    fn test_parallel_all_to_all_parallel_all_to_all_with_options_projected_value() {
        // A projected array view of a composite value stages the exchange through its composite value, so that the
        // result extents are staged as explicit extent values.
        let (output_type, program) =
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| {
                    let array = ValueProjection::<ArrayType>::into_projected(input)?;
                    Ok(array.parallel_all_to_all_with_options("x", 0, 1, CollectiveOptions::tiled())?.into_value())
                },
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4, 3])),
                vec![("x".to_string(), NamedAxis::Batched { size: Some(2) })],
            )
            .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 6])));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 3] .
                let %1:dimension<4> = constant [value=4]
                    %2:dimension<3> = constant [value=3]
                    %3:dimension<2> = constant [value=2]
                    %4:dimension<2> = dimension_div %1 %3
                    %5:dimension<6> = dimension_mul %2 %3
                    %6:f32[2, 6] = parallel_all_to_all [\
                        axis_name=\"x\", \
                        axis_size=2, \
                        split_axis=0, \
                        concatenation_axis=1, \
                        options=Tiled\
                    ] %0 %4 %5
                in (%6)"
            },
        );
    }

    #[test]
    fn test_parallel_all_to_all_parallel_swap_axes() {
        // Over a manual mesh axis, an unsharded invariant input is placed on the mesh and made varying before the
        // exchange, which stages identical split and concatenation positions.
        let input_type = ArrayType::new_static(DataType::F32, [2, 3]);
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
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}]}] = broadcast [
                    output_type=f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}]}],
                    output_axes=[0, 1],
                ] %0
                    %2:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %1
                    %3:dimension<3> = constant [value=3]
                    %4:dimension<2> = constant [value=2]
                    %5:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                parallel_all_to_all [
                        axis_name=\"x\",
                        axis_size=2,
                        split_axis=0,
                        concatenation_axis=0,
                        options=Untiled,
                        mesh=['x'=2:manual],
                    ] %2 %4 %3
                in (%5)"
            },
        );

        // Over a `batch` level, batch item `j` receives element `j` of every item along the swapped axis.
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    item.parallel_swap_axes("x", 0)
                },
                Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 4.0]).unwrap()),
        );

        // Negative axes count from the end, so `-1` swaps the trailing axis, which is the only axis of each item here.
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    item.parallel_swap_axes("x", -1)
                },
                Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 4.0]).unwrap()),
        );
    }

    #[test]
    fn test_parallel_all_to_all_parallel_swap_axes_with_axis_index_groups() {
        // The swapped ranked axis must have the common group size, rather than the full named-axis size, as its
        // extent, and the staged exchange records the ordered participant groups.
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_swap_axes_with_axis_index_groups("x", 0, vec![vec![0, 2], vec![3, 1]]),
            ArrayType::new_static(DataType::F32, [2, 3]),
            vec![("x".to_string(), NamedAxis::Batched { size: Some(4) })],
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::new_static(DataType::F32, [2, 3]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:f32[2, 3] = parallel_all_to_all [
                    axis_name=\"x\",
                    axis_size=4,
                    split_axis=0,
                    concatenation_axis=0,
                    options=Untiled,
                    axis_index_groups=[[0, 2], [3, 1]],
                ] %0
                in (%1)"
            },
        );

        // Negative axes count from the end, so `-2` swaps the leading axis of each rank-2 value.
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_swap_axes_with_axis_index_groups("x", -2, vec![vec![0, 2], vec![3, 1]]),
                ArrayType::new_static(DataType::F32, [2, 3]),
                vec![("x".to_string(), NamedAxis::Batched { size: Some(4) })],
            )
            .unwrap()
            .1
            .to_string(),
            program.to_string(),
        );

        // A swapped axis whose extent is the full named-axis size, rather than the group size, is rejected.
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_swap_axes_with_axis_index_groups("x", 0, vec![vec![0, 2], vec![3, 1]]),
                ArrayType::new_static(DataType::F32, [4, 3]),
                vec![("x".to_string(), NamedAxis::Batched { size: Some(4) })],
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_all_to_all` untiled split axis 0 size 4 must equal group size 2",
            ))),
        );
    }
}
