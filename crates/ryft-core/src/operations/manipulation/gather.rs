use std::collections::BTreeSet;
use std::fmt::Display;

use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{DifferentiableType, DifferentiationError, TransposableOperation, TranspositionDriver};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::constants::ZeroOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::sharding::{LogicalMesh, MeshAxisType, Sharding, ShardingDimension};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::operations::custom_derivatives::CustomVjpResidual;
use crate::types::{ArrayType, Shape, Size};

use super::scatter::{LinearScatterAddOperation, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind};
use super::slicing::is_integer;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`GatherOperation`].
pub const GATHER_OPERATION_NAME: &str = "gather";

/// Out-of-bounds index handling for [`gather`](Gather) and [`scatter`](super::scatter::Scatter), mirroring JAX's
/// [`GatherScatterMode`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.GatherScatterMode.html). The mode does
/// not affect the output [`Type`](crate::programs::types::Type) — only how a start index that would read or write outside the
/// operand is treated at execution time. It is shared by both operations (gather and scatter both reference it; the
/// scatter combiner kind lives in [`super::scatter`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum GatherScatterMode {
    /// The caller promises every index is in bounds; out-of-bounds behavior is undefined (and gradients are wrong if
    /// the promise is violated). This is the default and lowers directly to the bare StableHLO operation.
    #[default]
    PromiseInBounds,

    /// Each start index is clamped so the whole window stays in bounds.
    Clip,

    /// A window that would fall partly out of bounds is dropped: gather fills it with a fill value, scatter discards
    /// the update.
    FillOrDrop,
}

impl GatherScatterMode {
    /// Returns the canonical lowercase name of this mode.
    pub fn name(self) -> &'static str {
        match self {
            Self::PromiseInBounds => "promise_in_bounds",
            Self::Clip => "clip",
            Self::FillOrDrop => "fill_or_drop",
        }
    }
}

impl Display for GatherScatterMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Specification of how the index operand and the sliced windows map onto the operand and output axes of a
/// [`gather`](Gather), mirroring StableHLO's
/// [`gather`](https://openxla.org/stablehlo/spec#gather) dimension numbers and JAX's
/// [`GatherDimensionNumbers`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.GatherDimensionNumbers.html).
///
/// The index vector dimension is implicit and always the last axis of the indices operand (JAX's convention): the
/// indices operand has shape `[batch..., index_vector]`, where each length-`index_vector` slice is one start-index
/// vector whose components map onto operand axes through [`start_index_map`](Self::start_index_map). To gather with a
/// scalar index per query, give the indices a trailing size-1 axis.
///
/// The output rank is `offset_dimensions.len() + indices.rank() - 1`. Each output axis named in
/// [`offset_dimensions`](Self::offset_dimensions) carries one sliced window axis (in operand-axis order, skipping the
/// collapsed and batching axes); the remaining output axes carry the indices' batch axes in order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct GatherDimensionNumbers {
    /// Output axes that hold the sliced window (the "offset" axes), in ascending order. Their count equals the number
    /// of operand axes that are neither collapsed nor batching.
    offset_dimensions: Vec<usize>,

    /// Operand axes whose slice size is `1` and that are removed from the output, in ascending order.
    collapsed_slice_dimensions: Vec<usize>,

    /// For each component of a start-index vector (the last axis of the indices operand), the operand axis it indexes
    /// into. Its length equals the extent of the indices' index vector dimension.
    start_index_map: Vec<usize>,

    /// Operand axes that are batched against [`start_indices_batching_dimensions`](Self::start_indices_batching_dimensions),
    /// aligned 1:1, in ascending order. Each has slice size at most `1`.
    operand_batching_dimensions: Vec<usize>,

    /// Indices axes (other than the index vector dimension) that align 1:1 with
    /// [`operand_batching_dimensions`](Self::operand_batching_dimensions).
    start_indices_batching_dimensions: Vec<usize>,
}

impl GatherDimensionNumbers {
    /// Creates gather dimension numbers from explicit axis lists. The batching axis lists default to empty; use
    /// [`with_batching_dimensions`](Self::with_batching_dimensions) to set them.
    #[inline]
    pub fn new(
        offset_dimensions: Vec<usize>,
        collapsed_slice_dimensions: Vec<usize>,
        start_index_map: Vec<usize>,
    ) -> Self {
        Self {
            offset_dimensions,
            collapsed_slice_dimensions,
            start_index_map,
            operand_batching_dimensions: Vec::new(),
            start_indices_batching_dimensions: Vec::new(),
        }
    }

    /// Attaches the operand/indices batching axis pair (aligned 1:1).
    #[inline]
    pub fn with_batching_dimensions(
        mut self,
        operand_batching_dimensions: Vec<usize>,
        start_indices_batching_dimensions: Vec<usize>,
    ) -> Self {
        self.operand_batching_dimensions = operand_batching_dimensions;
        self.start_indices_batching_dimensions = start_indices_batching_dimensions;
        self
    }

    /// Returns the output offset axes.
    #[inline]
    pub fn offset_dimensions(&self) -> &[usize] {
        &self.offset_dimensions
    }

    /// Returns the collapsed (size-1, removed) operand axes.
    #[inline]
    pub fn collapsed_slice_dimensions(&self) -> &[usize] {
        &self.collapsed_slice_dimensions
    }

    /// Returns the start-index-to-operand-axis map.
    #[inline]
    pub fn start_index_map(&self) -> &[usize] {
        &self.start_index_map
    }

    /// Returns the operand batching axes.
    #[inline]
    pub fn operand_batching_dimensions(&self) -> &[usize] {
        &self.operand_batching_dimensions
    }

    /// Returns the indices batching axes.
    #[inline]
    pub fn start_indices_batching_dimensions(&self) -> &[usize] {
        &self.start_indices_batching_dimensions
    }
}

impl Display for GatherDimensionNumbers {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "(offset={:?}, collapsed_slice={:?}, start_index_map={:?}, operand_batching={:?}, \
             start_indices_batching={:?})",
            self.offset_dimensions,
            self.collapsed_slice_dimensions,
            self.start_index_map,
            self.operand_batching_dimensions,
            self.start_indices_batching_dimensions,
        )
    }
}

/// [`Operation`] that reads slices ("windows") out of an operand at positions named by an integer index operand,
/// assembling them into a new array. Refer to the documentation of [`Gather`] for the full semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatherOperation {
    /// Dimension numbers mapping the index operand and sliced windows onto the operand and output axes.
    dimensions: GatherDimensionNumbers,

    /// Size of the sliced window along each operand axis (length equals the operand rank).
    slice_sizes: Vec<usize>,

    /// Out-of-bounds index handling.
    mode: GatherScatterMode,

    /// Whether the caller guarantees the index vectors are sorted (a lowering hint only).
    indices_are_sorted: bool,

    /// Whether the caller guarantees the gathered windows do not overlap (a lowering hint only).
    unique_indices: bool,

    /// Optional requested output [`Sharding`], used when the inferred placement is ambiguous (see
    /// [`Self::with_output_sharding`]).
    output_sharding: Option<Sharding>,
}

impl GatherOperation {
    /// Creates a new [`GatherOperation`] with the provided dimension numbers and per-operand-axis slice sizes. The
    /// mode defaults to [`GatherScatterMode::PromiseInBounds`] and both index hints default to `false`; use the
    /// chained `with_*` builders to override them.
    #[inline]
    pub fn new(dimensions: GatherDimensionNumbers, slice_sizes: Vec<usize>) -> Self {
        Self {
            dimensions,
            slice_sizes,
            mode: GatherScatterMode::PromiseInBounds,
            indices_are_sorted: false,
            unique_indices: false,
            output_sharding: None,
        }
    }

    /// Sets the out-of-bounds index handling mode.
    #[inline]
    pub fn with_mode(mut self, mode: GatherScatterMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the sorted-indices lowering hint.
    #[inline]
    pub fn with_indices_are_sorted(mut self, indices_are_sorted: bool) -> Self {
        self.indices_are_sorted = indices_are_sorted;
        self
    }

    /// Sets the unique-indices lowering hint.
    #[inline]
    pub fn with_unique_indices(mut self, unique_indices: bool) -> Self {
        self.unique_indices = unique_indices;
        self
    }

    /// Requests `output_sharding` for the result. The gather sharding rule replicates the operand axes named by
    /// [`GatherDimensionNumbers::start_index_map`] (and the index vector axis); when that leaves the output placement
    /// ambiguous — for example because a sliced operand axis is sharded over an explicit mesh axis — a requested
    /// output sharding resolves it, bypassing inference. This mirrors `dot`/`reduce`'s `with_output_sharding`.
    #[inline]
    pub fn with_output_sharding(mut self, output_sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns the dimension numbers.
    #[inline]
    pub fn dimensions(&self) -> &GatherDimensionNumbers {
        &self.dimensions
    }

    /// Returns the per-operand-axis slice sizes.
    #[inline]
    pub fn slice_sizes(&self) -> &[usize] {
        &self.slice_sizes
    }

    /// Returns the out-of-bounds index handling mode.
    #[inline]
    pub fn mode(&self) -> GatherScatterMode {
        self.mode
    }

    /// Returns the sorted-indices hint.
    #[inline]
    pub fn indices_are_sorted(&self) -> bool {
        self.indices_are_sorted
    }

    /// Returns the unique-indices hint.
    #[inline]
    pub fn unique_indices(&self) -> bool {
        self.unique_indices
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
}

impl Display for GatherOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for GatherOperation {
    #[inline]
    fn name(&self) -> &'static str {
        GATHER_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        match input_types[0].gather(&input_types[1], self) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("dimensions", &self.dimensions)?;
            operation.field("slice_sizes", format_args!("{:?}", self.slice_sizes))?;
            if self.mode != GatherScatterMode::PromiseInBounds {
                operation.field("mode", self.mode)?;
            }
            if self.indices_are_sorted {
                operation.field("indices_are_sorted", self.indices_are_sorted)?;
            }
            if self.unique_indices {
                operation.field("unique_indices", self.unique_indices)?;
            }
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

/// Value-level gather capability: the receiver-style entry point for staging or executing [`GatherOperation`].
///
/// This is the direct analogue of the StableHLO [`gather`](https://openxla.org/stablehlo/spec#gather) operation and
/// JAX's [`lax.gather`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html). The receiver is the operand
/// (the data source); `indices` is a separate integer-typed value whose last axis holds each start-index vector. The
/// output assembles the sliced windows according to `operation`'s [`GatherDimensionNumbers`]; see that type for the
/// shape rule and the implicit index-vector-dimension convention.
///
/// # Example
///
/// ```rust
/// # use ryft_core::operations::manipulation::{Gather, GatherDimensionNumbers, GatherOperation};
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// # use ryft_core::types::{ArrayType, DataType};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Take rows 0 and 2 of a 3x2 matrix: each query is a scalar row index, so the indices have shape [2, 1]
/// // (two queries, one index component each) and the gathered window is a full row (slice sizes [1, 2]).
/// let operand = Array::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
/// let indices = Array::new(
///     ArrayType::new(DataType::I32, ryft_core::types::Shape::new(vec![
///         ryft_core::types::Size::Static(2),
///         ryft_core::types::Size::Static(1),
///     ])),
///     vec![0.0, 2.0],
/// );
/// let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
/// let operation = GatherOperation::new(dimensions, vec![1, 2]);
/// let rows = operand.gather(&indices, &operation)?;
/// // `rows` has shape [2, 2] holding rows 0 and 2: [[0, 1], [4, 5]].
/// assert_eq!(rows.values, vec![0.0, 1.0, 4.0, 5.0]);
/// # Ok(())
/// # }
/// ```
pub trait Gather: Sized {
    /// Gathers windows out of `self` (the operand) at the positions named by `indices`, according to `operation`.
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError>;
}

impl Gather for ArrayType {
    /// Type-level gather: validates the dimension numbers and slice sizes against the operand and indices types and
    /// computes the output shape and placement. Mirrors JAX's `_gather_shape_rule`/`_gather_sharding_rule`.
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError> {
        let operand = self;
        let dimensions = operation.dimensions();
        let slice_sizes = operation.slice_sizes();
        let operand_rank = operand.rank();
        let indices_rank = indices.rank();

        if indices_rank == 0 {
            return Err(TypeError {
                message: format!(
                    "'{GATHER_OPERATION_NAME}' indices must have rank at least 1 (the trailing index vector)"
                ),
            }
            .into());
        }
        if !is_integer(indices.data_type()) {
            return Err(TypeError {
                message: format!("'{GATHER_OPERATION_NAME}' indices must be integer-typed but have type {indices}"),
            }
            .into());
        }
        let index_vector_dimension = indices_rank - 1;
        let Size::Static(index_vector_extent) = indices.dimension(index_vector_dimension as isize) else {
            return Err(TypeError {
                message: format!("'{GATHER_OPERATION_NAME}' indices index vector dimension must have a static extent"),
            }
            .into());
        };

        // Output rank, and the constituent operand-axis classification.
        let output_rank = dimensions.offset_dimensions().len() + indices_rank - 1;
        validate_sorted_unique_in_range(
            GATHER_OPERATION_NAME,
            "offset_dimensions",
            dimensions.offset_dimensions(),
            output_rank,
        )?;
        validate_sorted_unique_in_range(
            GATHER_OPERATION_NAME,
            "collapsed_slice_dimensions",
            dimensions.collapsed_slice_dimensions(),
            operand_rank,
        )?;
        validate_sorted_unique_in_range(
            GATHER_OPERATION_NAME,
            "operand_batching_dimensions",
            dimensions.operand_batching_dimensions(),
            operand_rank,
        )?;

        if dimensions.start_index_map().len() != index_vector_extent {
            return Err(TypeError {
                message: format!(
                    "'{GATHER_OPERATION_NAME}' start_index_map has length {} but the index vector extent is \
                     {index_vector_extent}",
                    dimensions.start_index_map().len(),
                ),
            }
            .into());
        }
        validate_unique_in_range(GATHER_OPERATION_NAME, "start_index_map", dimensions.start_index_map(), operand_rank)?;

        if dimensions.start_indices_batching_dimensions().len() != dimensions.operand_batching_dimensions().len() {
            return Err(TypeError {
                message: format!(
                    "'{GATHER_OPERATION_NAME}' operand and start-indices batching dimensions must align 1:1, but got {} \
                     and {}",
                    dimensions.operand_batching_dimensions().len(),
                    dimensions.start_indices_batching_dimensions().len(),
                ),
            }
            .into());
        }
        for &dimension in dimensions.start_indices_batching_dimensions() {
            if dimension >= indices_rank || dimension == index_vector_dimension {
                return Err(TypeError {
                    message: format!(
                        "'{GATHER_OPERATION_NAME}' start_indices_batching_dimensions entry {dimension} is out of range \
                         or names the index vector dimension"
                    ),
                }
                .into());
            }
        }

        // The collapsed, batching, and start-index-map axis sets must be mutually disjoint where required.
        let collapsed: BTreeSet<usize> = dimensions.collapsed_slice_dimensions().iter().copied().collect();
        let operand_batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        if collapsed.intersection(&operand_batching).next().is_some() {
            return Err(TypeError {
                message: format!(
                    "'{GATHER_OPERATION_NAME}' collapsed_slice_dimensions and operand_batching_dimensions must be \
                     disjoint"
                ),
            }
            .into());
        }

        // Slice sizes: one per operand axis; size 1 on collapsed axes; size at most 1 on batching axes; within the
        // operand extent when that extent is static.
        if slice_sizes.len() != operand_rank {
            return Err(TypeError {
                message: format!(
                    "'{GATHER_OPERATION_NAME}' slice_sizes has length {} but the operand has rank {operand_rank}",
                    slice_sizes.len(),
                ),
            }
            .into());
        }
        for (axis, &size) in slice_sizes.iter().enumerate() {
            if let Size::Static(extent) = operand.dimension(axis as isize) {
                if size > extent {
                    return Err(TypeError {
                        message: format!(
                            "'{GATHER_OPERATION_NAME}' slice size {size} at axis {axis} exceeds the operand extent \
                             {extent}"
                        ),
                    }
                    .into());
                }
            }
            if collapsed.contains(&axis) && size != 1 {
                return Err(TypeError {
                    message: format!(
                        "'{GATHER_OPERATION_NAME}' collapsed slice dimension {axis} must have slice size 1 but has {size}"
                    ),
                }
                .into());
            }
            if operand_batching.contains(&axis) && size > 1 {
                return Err(TypeError {
                    message: format!(
                        "'{GATHER_OPERATION_NAME}' operand batching dimension {axis} must have slice size at most 1 but \
                         has {size}"
                    ),
                }
                .into());
            }
        }

        let offset_count = operand_rank - collapsed.len() - operand_batching.len();
        if dimensions.offset_dimensions().len() != offset_count {
            return Err(TypeError {
                message: format!(
                    "'{GATHER_OPERATION_NAME}' offset_dimensions has length {} but the operand has {offset_count} \
                     non-collapsed, non-batching axes",
                    dimensions.offset_dimensions().len(),
                ),
            }
            .into());
        }

        // Batch-dimension extents must match between operand and indices.
        for (&operand_axis, &indices_axis) in
            dimensions.operand_batching_dimensions().iter().zip(dimensions.start_indices_batching_dimensions())
        {
            if operand.dimension(operand_axis as isize) != indices.dimension(indices_axis as isize) {
                return Err(TypeError {
                    message: format!(
                        "'{GATHER_OPERATION_NAME}' batching dimensions must have equal extents, but operand axis \
                         {operand_axis} and indices axis {indices_axis} differ"
                    ),
                }
                .into());
            }
        }

        // Output shape: offset positions take the (non-collapsed, non-batching) operand window sizes in operand-axis
        // order; the remaining positions take the indices' batch axes (every axis but the index vector) in order.
        let operand_offset_axes: Vec<usize> = (0..operand_rank)
            .filter(|axis| !collapsed.contains(axis) && !operand_batching.contains(axis))
            .collect();
        let batch_query_sizes: Vec<Size> = (0..indices_rank)
            .filter(|axis| *axis != index_vector_dimension)
            .map(|axis| indices.dimension(axis as isize))
            .collect();
        let offset_position: BTreeSet<usize> = dimensions.offset_dimensions().iter().copied().collect();
        let mut offset_iterator = operand_offset_axes.iter();
        let mut batch_iterator = batch_query_sizes.iter();
        let output_dimensions: Vec<Size> = (0..output_rank)
            .map(|position| {
                if offset_position.contains(&position) {
                    let &operand_axis = offset_iterator.next().expect("offset axis count was validated");
                    Size::Static(slice_sizes[operand_axis])
                } else {
                    *batch_iterator.next().expect("batch axis count was validated")
                }
            })
            .collect();

        // Output sharding (JAX's `_gather_sharding_rule`). The operand axes named by `start_index_map`, the collapsed
        // axes, and the indices' index vector axis must be replicated (gated to explicit mesh axes; manual/auto
        // placements pass through). The output then inherits the operand's window-axis placements at the offset
        // positions and the indices' batch-axis placements elsewhere, carrying the operand's reduction state. When the
        // placement is ambiguous (a sliced axis is sharded over an explicit axis), a requested `output_sharding`
        // resolves it, mirroring `dot`/`reduce`.
        let operand_sharding = operand.sharding();
        let indices_sharding = indices.sharding();
        let sharding = if let Some(requested) = operation.output_sharding() {
            if requested.rank() != output_rank {
                return Err(TypeError {
                    message: format!(
                        "'{GATHER_OPERATION_NAME}' output sharding rank ({}) does not match the output rank \
                         ({output_rank})",
                        requested.rank(),
                    ),
                }
                .into());
            }
            if references_auto_axis(requested) {
                return Err(TypeError {
                    message: format!("'{GATHER_OPERATION_NAME}' output sharding cannot reference auto mesh axes"),
                }
                .into());
            }
            Some(requested.clone())
        } else if let Some(mesh) = resolve_mesh(operand_sharding, indices_sharding)? {
            // The operand axes that drive start indices (start_index_map ∪ collapsed) and the indices' index vector
            // axis must be replicated over explicit mesh axes; otherwise the placement is ambiguous and an output
            // sharding is required.
            let replicated_operand_axes: BTreeSet<usize> = dimensions
                .start_index_map()
                .iter()
                .chain(dimensions.collapsed_slice_dimensions())
                .copied()
                .collect();
            if let Some(sharding) = operand_sharding {
                for &axis in &replicated_operand_axes {
                    if dimension_has_explicit_axis(&mesh, &sharding.dimensions()[axis]) {
                        return Err(TypeError {
                            message: format!(
                                "'{GATHER_OPERATION_NAME}' operand axis {axis} is indexed by the start indices and must \
                                 be replicated over explicit mesh axes; request an explicit output sharding to resolve \
                                 placement"
                            ),
                        }
                        .into());
                    }
                }
            }
            if let Some(sharding) = indices_sharding {
                if dimension_has_explicit_axis(&mesh, &sharding.dimensions()[index_vector_dimension]) {
                    return Err(TypeError {
                        message: format!(
                            "'{GATHER_OPERATION_NAME}' indices index vector dimension must be replicated over explicit \
                             mesh axes"
                        ),
                    }
                    .into());
                }
            }

            // Propagate placement: offset positions inherit the operand window axes; the remaining positions inherit
            // the indices' batch axes (every axis but the index vector), in order.
            let indices_batch_axes: Vec<usize> =
                (0..indices.rank()).filter(|axis| *axis != index_vector_dimension).collect();
            let mut offset_iterator = operand_offset_axes.iter();
            let mut batch_iterator = indices_batch_axes.iter();
            let placement: Vec<ShardingDimension> = (0..output_rank)
                .map(|position| {
                    if offset_position.contains(&position) {
                        let &operand_axis = offset_iterator.next().expect("offset axis count was validated");
                        operand_sharding
                            .map(|sharding| sharding.dimensions()[operand_axis].clone())
                            .unwrap_or(ShardingDimension::Replicated)
                    } else {
                        let &indices_axis = batch_iterator.next().expect("batch axis count was validated");
                        indices_sharding
                            .map(|sharding| sharding.dimensions()[indices_axis].clone())
                            .unwrap_or(ShardingDimension::Replicated)
                    }
                })
                .collect();

            // Carry the operand's reduction state: gather selects elements, so a value unreduced/reduced over a mesh
            // axis stays so (selection commutes with a pending cross-device sum), mirroring slice.
            let (unreduced_axes, reduced_axes, varying_manual_axes) = match operand_sharding {
                Some(sharding) => (
                    sharding.unreduced_axes().iter().cloned().collect::<Vec<_>>(),
                    sharding.reduced_axes().iter().cloned().collect::<Vec<_>>(),
                    sharding.varying_manual_axes().iter().cloned().collect::<Vec<_>>(),
                ),
                None => (Vec::new(), Vec::new(), Vec::new()),
            };
            let sharding =
                Sharding::with_manual_axes(mesh, placement, unreduced_axes, reduced_axes, varying_manual_axes)
                    .map_err(|error| TypeError {
                        message: format!("'{GATHER_OPERATION_NAME}' output sharding construction failed: {error}"),
                    })?;
            Some(sharding.without_auto_axes())
        } else {
            None
        };
        ArrayType::new(operand.data_type(), Shape::new(output_dimensions))
            .with_memory(operand.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

/// Any context-carrying value gathers by binding a [`GatherOperation`] through its own context. The
/// `From<GatherOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Gather for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<GatherOperation>,
{
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError> {
        let mut outputs =
            self.dispatch_domain().bind(operation.clone(), Vec::new(), &[self.clone(), indices.clone()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Domain<Type = ArrayType, Value: Gather>> InterpretableOperation<C> for GatherOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].gather(&inputs[1], self)?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for GatherOperation where
    C::Operation: From<GatherOperation>
{
}

/// Returns whether `dimension` is sharded over at least one explicit mesh axis of `mesh` (the explicit-axis gate used
/// by the dot/reduce/slice sharding rules). Shared with [`super::scatter`].
pub(crate) fn dimension_has_explicit_axis(mesh: &LogicalMesh, dimension: &ShardingDimension) -> bool {
    matches!(dimension, ShardingDimension::Sharded(axis_names)
        if axis_names.iter().any(|name| mesh.axis_type(name) == Some(MeshAxisType::Explicit)))
}

/// Returns whether `sharding` references any auto mesh axis in a placement or reduction-state set. Shared with
/// [`super::scatter`].
pub(crate) fn references_auto_axis(sharding: &Sharding) -> bool {
    let placement_auto = sharding.dimensions().iter().any(|dimension| {
        matches!(dimension, ShardingDimension::Sharded(axis_names)
            if axis_names.iter().any(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Auto)))
    });
    let set_auto =
        |axes: &BTreeSet<String>| axes.iter().any(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Auto));
    placement_auto || set_auto(sharding.unreduced_axes()) || set_auto(sharding.reduced_axes())
}

/// Resolves the common mesh of two optional shardings, erroring on a mesh mismatch. Returns `None` when neither side
/// is sharded.
fn resolve_mesh(
    operand_sharding: Option<&Sharding>,
    indices_sharding: Option<&Sharding>,
) -> Result<Option<LogicalMesh>, TypeError> {
    match (operand_sharding, indices_sharding) {
        (None, None) => Ok(None),
        (Some(left), Some(right)) => {
            if left.mesh() != right.mesh() {
                return Err(TypeError {
                    message: format!("'{GATHER_OPERATION_NAME}' operand and indices shardings must use the same mesh"),
                });
            }
            Ok(Some(left.mesh().clone()))
        }
        (Some(left), None) => Ok(Some(left.mesh().clone())),
        (None, Some(right)) => Ok(Some(right.mesh().clone())),
    }
}

/// Validates that `axes` is strictly ascending (sorted and unique) and that every entry is in `0..bound`. Shared with
/// [`super::scatter`].
pub(crate) fn validate_sorted_unique_in_range(
    operation_name: &'static str,
    field: &str,
    axes: &[usize],
    bound: usize,
) -> Result<(), TypeError> {
    for window in axes.windows(2) {
        if window[0] >= window[1] {
            return Err(TypeError {
                message: format!("'{operation_name}' {field} must be sorted and unique but got {axes:?}"),
            });
        }
    }
    if let Some(&axis) = axes.iter().find(|&&axis| axis >= bound) {
        return Err(TypeError {
            message: format!("'{operation_name}' {field} entry {axis} is out of range for bound {bound}"),
        });
    }
    Ok(())
}

/// Validates that every entry of `axes` is unique and in `0..bound` (order not required). Shared with
/// [`super::scatter`].
pub(crate) fn validate_unique_in_range(
    operation_name: &'static str,
    field: &str,
    axes: &[usize],
    bound: usize,
) -> Result<(), TypeError> {
    let mut seen = BTreeSet::new();
    for &axis in axes {
        if axis >= bound {
            return Err(TypeError {
                message: format!("'{operation_name}' {field} entry {axis} is out of range for bound {bound}"),
            });
        }
        if !seen.insert(axis) {
            return Err(TypeError { message: format!("'{operation_name}' {field} must be unique but got {axes:?}") });
        }
    }
    Ok(())
}

// TODO(eaplatanios): Should this be renamed to something that's not about "linearity"? This is about captured primals.
/// Captured-index gather linear operation: the linear map `t ↦ gather(t, indices; dimensions)` over the tangent (or
/// cotangent) of the gathered operand.
///
/// It is the captured-index linear map emitted by the JVP of [`GatherOperation`]: the
/// integer index operand is a primal value captured at linearization time as a residual factor (it has no tangent
/// space, so the map is linear in the single tangent operand), and its transpose is the dual scatter-add. The single
/// operation input is the gathered operand's tangent; the captured `indices` factor supplies the gather's index
/// operand during type inference.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearGatherOperation<F> {
    /// Underlying [`GatherOperation`] describing the gather geometry.
    operation: GatherOperation,

    /// Captured integer index operand factor.
    indices: F,
}

impl<F> LinearGatherOperation<F> {
    /// Creates a new [`LinearGatherOperation`] from the underlying gather and the captured index factor.
    #[inline]
    pub fn new(operation: GatherOperation, indices: F) -> Self {
        Self { operation, indices }
    }

    /// Returns the underlying [`GatherOperation`] describing the gather geometry.
    #[inline]
    pub fn operation(&self) -> &GatherOperation {
        &self.operation
    }

    /// Returns the captured integer index operand factor.
    #[inline]
    pub fn indices(&self) -> &F {
        &self.indices
    }
}

impl<F: Value<Type = ArrayType>> Display for LinearGatherOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<F: Value<Type = ArrayType>> Operation<ArrayType> for LinearGatherOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        GATHER_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        self.operation
            .infer_output_types(&[input_types[0].clone(), self.indices.r#type().into_owned()], &[])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let _ = indentation;
        formatter.write_str(self.name())
    }
}

impl<F, C> InterpretableOperation<C> for LinearGatherOperation<F>
where
    C: Domain<Type = ArrayType, Value: Gather>,
    F: CustomVjpResidual<C::Value>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].gather(&self.indices().residual_value()?, self.operation())?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate) for a [`LinearGatherOperation`].
impl<F: Value<Type = ArrayType>, C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C>
    for LinearGatherOperation<F>
where
    C::Operation: From<LinearGatherOperation<F>>,
{
}

/// Transpose rule for the captured-index gather. The forward linear map
/// `t ↦ gather(t, indices)` has, as its adjoint, the dual scatter-add that writes the output cotangent back into a
/// zero operand at the gathered windows: the scatter geometry mirrors the gather axis-for-axis and the captured
/// indices carry over. Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O, F: Value<Type = ArrayType>> TransposableOperation<V, O> for LinearGatherOperation<F>
where
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<LinearScatterAddOperation<F>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent().unwrap())]),
            MaybeZero::Value(cotangent) => {
                let zeros = MaybeZero::Zero(inputs[0].r#type().cotangent().unwrap()).materialize(context)?;
                let dimensions = self.operation().dimensions();
                let scatter_dimensions = ScatterDimensionNumbers::new(
                    dimensions.offset_dimensions().to_vec(),
                    dimensions.collapsed_slice_dimensions().to_vec(),
                    dimensions.start_index_map().to_vec(),
                )
                .with_batching_dimensions(
                    dimensions.operand_batching_dimensions().to_vec(),
                    dimensions.start_indices_batching_dimensions().to_vec(),
                );
                let scatter_operation = ScatterOperation::new(scatter_dimensions, ScatterReductionKind::Add)
                    .with_mode(self.operation().mode())
                    .with_indices_are_sorted(self.operation().indices_are_sorted())
                    .with_unique_indices(self.operation().unique_indices());
                let outputs = context.stage_operation(
                    LinearScatterAddOperation::new(scatter_operation, self.indices().clone()),
                    Vec::new(),
                    &[zeros, cotangent.clone()],
                )?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![MaybeZero::Value(outputs.into_iter().next().unwrap())])
            }
        }
    }
}

/// Partition-aware transpose rule for the primal [`GatherOperation`]. The integer index operand (operand 1) has no
/// tangent space, so in a valid pushforward it is the known operand and the gathered operand (operand 0) is the
/// linear one. The forward map `t ↦ gather(t, indices)` has, as its adjoint, the dual scatter-add that writes the
/// output cotangent back into a zero operand at the gathered windows: the scatter geometry mirrors the gather
/// axis-for-axis. This reproduces the captured-index [`LinearGatherOperation`] transpose rule, reading the indices
/// from the pullback through `operand_values` and staging a primal additive [`ScatterOperation`] instead of folding
/// the indices into a captured factor. The indices receive a structural zero, and a zero output cotangent stays a
/// structural zero.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for GatherOperation
where
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<ScatterOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(vec![
                MaybeZero::Zero(inputs[0].r#type().cotangent().unwrap()),
                MaybeZero::Zero(inputs[1].r#type().into_owned()),
            ]),
            MaybeZero::Value(cotangent) => {
                // The indices are the known operand; the dispatch guarantees a `Known` operand carries its pullback
                // value, so read the tracer directly.
                let indices = inputs[1]
                    .as_known()
                    .expect("dispatch guarantees a known operand carries its pullback value")
                    .clone();
                let zeros = MaybeZero::Zero(inputs[0].r#type().cotangent().unwrap()).materialize(context)?;
                let scatter_dimensions = ScatterDimensionNumbers::new(
                    self.dimensions().offset_dimensions().to_vec(),
                    self.dimensions().collapsed_slice_dimensions().to_vec(),
                    self.dimensions().start_index_map().to_vec(),
                )
                .with_batching_dimensions(
                    self.dimensions().operand_batching_dimensions().to_vec(),
                    self.dimensions().start_indices_batching_dimensions().to_vec(),
                );
                let scatter_operation = ScatterOperation::new(scatter_dimensions, ScatterReductionKind::Add)
                    .with_mode(self.mode())
                    .with_indices_are_sorted(self.indices_are_sorted())
                    .with_unique_indices(self.unique_indices());
                let outputs =
                    context.stage_operation(scatter_operation, Vec::new(), &[zeros, indices, cotangent.clone()])?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![
                    MaybeZero::Value(outputs.into_iter().next().unwrap()),
                    MaybeZero::Zero(inputs[1].r#type().into_owned()),
                ])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::DataType;

    use super::*;

    fn indices_type(dimensions: Vec<usize>) -> ArrayType {
        ArrayType::new(DataType::I32, Shape::new(dimensions.into_iter().map(Size::Static).collect()))
    }

    fn float_type(dimensions: Vec<usize>) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(dimensions.into_iter().map(Size::Static).collect()))
    }

    #[test]
    fn test_gather_take_rows_inference_and_rendering() {
        // Take whole rows of a [3, 2] matrix indexed by a [2, 1] index array: offset axis 1 carries the row (slice
        // sizes [1, 2]); axis 0 (the collapsed row axis) is driven by the start index.
        let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
        let operation = GatherOperation::new(dimensions, vec![1, 2]);
        assert_eq!(operation.name(), GATHER_OPERATION_NAME);
        assert_eq!(operation.slice_sizes(), &[1, 2]);

        let operand = float_type(vec![3, 2]);
        let indices = indices_type(vec![2, 1]);
        let output = operation.infer_output_types(&[operand.clone(), indices.clone()], &[]).unwrap();
        assert_eq!(output, vec![float_type(vec![2, 2])]);

        assert_eq!(
            format!("{operation}"),
            concat!(
                "gather [\n",
                "    dimensions=(offset=[1], collapsed_slice=[0], start_index_map=[0], operand_batching=[], ",
                "start_indices_batching=[]),\n",
                "    slice_sizes=[1, 2],\n",
                "]",
            ),
        );
    }

    #[test]
    fn test_gather_rejects_invalid_dimension_numbers() {
        let operand = float_type(vec![3, 2]);
        let indices = indices_type(vec![2, 1]);

        // Non-integer indices are rejected.
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        assert!(operation.infer_output_types(&[operand.clone(), float_type(vec![2, 1])], &[]).is_err());

        // start_index_map length must equal the index vector extent (here 1, not 2).
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0, 1]), vec![1, 2]);
        assert!(operation.infer_output_types(&[operand.clone(), indices.clone()], &[]).is_err());

        // A collapsed axis must have slice size 1.
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![2, 2]);
        assert!(operation.infer_output_types(&[operand.clone(), indices.clone()], &[]).is_err());

        // offset_dimensions count must equal the non-collapsed, non-batching operand axes (here 1).
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1, 2], vec![0], vec![0]), vec![1, 2]);
        assert!(operation.infer_output_types(&[operand, indices], &[]).is_err());
    }

    #[test]
    fn test_gather_propagates_and_replicates_sharding() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        // Operand [4, 2] sharded only on the feature axis (axis 1); axis 0 (indexed by the start index) is replicated.
        let operand = float_type(vec![4, 2])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])])
                    .unwrap(),
            )
            .unwrap();
        let indices = indices_type(vec![3, 1]);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        // Output [3, 2]: the query axis (from the indices) is replicated, the feature axis keeps `y`.
        let output = operation.infer_output_types(&[operand, indices], &[]).unwrap();
        assert_eq!(
            output[0].sharding().unwrap().dimensions(),
            &[ShardingDimension::Replicated, ShardingDimension::sharded(["y"])],
        );

        // Sharding the start-indexed operand axis over an explicit mesh axis is ambiguous without an output sharding.
        let operand = float_type(vec![4, 2])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let indices = indices_type(vec![3, 1]);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        assert!(operation.infer_output_types(&[operand, indices], &[]).is_err());
    }

    #[test]
    fn test_gather_eager_modes() {
        use crate::tests::TestArray;

        // Gather scalars from [10, 20, 30, 40] at positions 1 and 5; position 5 is out of bounds (last valid is 3).
        let dimensions = GatherDimensionNumbers::new(vec![], vec![0], vec![0]);
        let indices = TestArray::new(indices_type(vec![2, 1]), vec![1.0, 5.0]);
        let run = |mode| {
            TestArray::vector(vec![10.0, 20.0, 30.0, 40.0])
                .gather(&indices, &GatherOperation::new(dimensions.clone(), vec![1]).with_mode(mode))
                .unwrap()
                .values
        };
        // Clip and promise-in-bounds clamp the out-of-bounds index to the last valid start.
        assert_eq!(run(GatherScatterMode::Clip), vec![20.0, 40.0]);
        assert_eq!(run(GatherScatterMode::PromiseInBounds), vec![20.0, 40.0]);
        // Fill-or-drop fills the out-of-bounds query with zero.
        assert_eq!(run(GatherScatterMode::FillOrDrop), vec![20.0, 0.0]);
    }

    /// Minimal operation enum hosting the primal [`GatherOperation`] (the forward gather) and the primal
    /// [`ScatterOperation`] (its staged scatter-add adjoint) plus the structural `zero` and `add` operations the
    /// transpose pass needs. The `Constant` variant carries the value parameter `V` so the [`Operation`] derive can
    /// infer the primary type. [`TransposableOperation`] is hand-written rather than derived because the primal
    /// [`ScatterOperation`] adjoint target has no transpose rule (it only ever appears in the pullback, never as a
    /// forward instruction being transposed); the derived all-variant dispatcher would require one.
    #[derive(Clone, Debug, ryft_macros::Operation)]
    enum TestGatherOperation<V: Value<Type = ArrayType>> {
        Zero(ZeroOperation<ArrayType>),
        Constant(crate::operations::constants::ConstantOperation<V>),
        Add(crate::operations::math::AddOperation),
        Gather(GatherOperation),
        Scatter(ScatterOperation),
    }

    impl<V: Value<Type = ArrayType>> TransposableOperation<V, TestGatherOperation<V>> for TestGatherOperation<V> {
        fn transpose<D: TranspositionDriver<V, TestGatherOperation<V>>>(
            &self,
            context: &mut TracingContext<V, TestGatherOperation<V>>,
            driver: &D,
            inputs: &[PartialValue<Tracer<TracingContext<V, TestGatherOperation<V>>>>],
            outputs: &[MaybeZero<Tracer<TracingContext<V, TestGatherOperation<V>>>>],
        ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, TestGatherOperation<V>>>>>, DifferentiationError> {
            match self {
                Self::Gather(operation) => operation.transpose(context, driver, inputs, outputs),
                _ => Err(ProgramError::UnsupportedOperation {
                    message: format!("{} is not transposed in this test enum", self.name()),
                }
                .into()),
            }
        }
    }

    #[test]
    fn test_gather_partitioned_transpose_computes_scatter_add_adjoint() {
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::programs::types::Typed;
        use crate::tests::TestArray;

        // Take rows 0 and 2 of a [3, 2] operand: the operand is linear and the [2, 1] index array is the known
        // operand. The gathered output and its cotangent have shape [2, 2].
        let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
        let operation = GatherOperation::new(dimensions, vec![1, 2]);
        let operand = TestArray::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indices = TestArray::new(indices_type(vec![2, 1]), vec![0.0, 2.0]);
        let cotangent = TestArray::matrix(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        let operand_type = operand.r#type().into_owned();
        let indices_type = indices.r#type().into_owned();

        // Build `gather(operand, indices)` over the test enum, treat only the operand as linear, and interpret the
        // pullback on `[cotangent, indices]`.
        let mut builder = ProgramBuilder::<TestArray, TestGatherOperation<TestArray>>::new();
        let operand_input = builder.add_input(operand_type.clone());
        let indices_input = builder.add_input(indices_type.clone());
        let output =
            builder.add_instruction(operation.clone(), Vec::new(), vec![operand_input, indices_input]).unwrap()[0];
        let program = builder
            .build::<(TestArray, TestArray), TestArray>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(pullback.output_ids().len(), 1, "the known index input must receive no cotangent output");
        let operand_cotangents = pullback.interpret(vec![cotangent, indices]).unwrap();
        assert_eq!(operand_cotangents.len(), 1);
        assert_eq!(*operand_cotangents[0].r#type(), operand_type);
        // The scatter-add adjoint writes the cotangent rows back into rows 0 and 2 of a zero operand.
        assert_eq!(operand_cotangents[0].values, vec![10.0, 20.0, 0.0, 0.0, 30.0, 40.0]);
    }
}
