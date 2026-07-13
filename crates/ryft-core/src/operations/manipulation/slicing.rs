use std::fmt::Display;

use crate::contexts::Context;
use crate::contexts::Domain;
use crate::contexts::StagingContext;
use crate::differentiation::DifferentiationError;
use crate::differentiation::TransposableOperation;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::constants::ZeroOperation;
use crate::operations::{Operation, OperationFormatter};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::sharding::{MeshAxisType, Sharding, ShardingDimension};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::operations::custom_derivatives::CustomVjpResidual;
use crate::tracing_v2::operations::scan::render_factor_list;
use crate::tracing_v2::operations::slicing::static_update_sizes;
use crate::types::{ArrayType, DataType, Shape, Size, TypeError, Typed};

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`SliceOperation`].
pub const SLICE_OPERATION_NAME: &'static str = "slice";

/// Canonical operation name for [`UpdateSliceOperation`].
pub const UPDATE_SLICE_OPERATION_NAME: &'static str = "update_slice";

/// Canonical operation name for [`DynamicSliceOperation`].
pub const DYNAMIC_SLICE_OPERATION_NAME: &'static str = "dynamic_slice";

/// Canonical operation name for [`DynamicUpdateSliceOperation`].
pub const DYNAMIC_UPDATE_SLICE_OPERATION_NAME: &'static str = "dynamic_update_slice";

// TODO(eaplatanios): This should be a function on `DataType` along with other helpers like
//  `is_boolean`, `is_floating_point`, etc.
/// Returns `true` when `data_type` is a signed or unsigned integer type and can therefore carry a slice start index.
pub(crate) fn is_integer(data_type: DataType) -> bool {
    matches!(
        data_type,
        DataType::I1
            | DataType::I2
            | DataType::I4
            | DataType::I8
            | DataType::I16
            | DataType::I32
            | DataType::I64
            | DataType::U1
            | DataType::U2
            | DataType::U4
            | DataType::U8
            | DataType::U16
            | DataType::U32
            | DataType::U64,
    )
}

/// Validates the scalar integer start-index operand types of a dynamic slicing operation. Each index type must be a
/// rank-0 integer type, and all indices must share one integer type. The `operation_name` parameter selects the
/// reported operation name because this helper serves both [`DynamicSliceOperation`] and
/// [`DynamicUpdateSliceOperation`].
fn validate_start_index_types(operation_name: &'static str, index_types: &[ArrayType]) -> Result<(), ProgramError> {
    for (index, index_type) in index_types.iter().enumerate() {
        if index_type.rank() != 0 || !is_integer(index_type.data_type()) {
            return Err(TypeError {
                message: format!(
                    "'{operation_name}' start index {index} must be a scalar integer but has type {index_type}",
                ),
            }
            .into());
        }
        if index_type.data_type() != index_types[0].data_type() {
            return Err(TypeError {
                message: format!(
                    "'{operation_name}' start indices must share one integer type but index {index} has type \
                    {index_type} and index 0 has type {first}",
                    first = index_types[0],
                ),
            }
            .into());
        }
    }
    Ok(())
}

/// [`Operation`] that extracts a (possibly strided) sub-array from its input using static start, limit, and stride
/// values. Refer to the documentation of [`Slice`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SliceOperation {
    /// Inclusive start index for each input axis.
    start_indices: Vec<usize>,

    /// Exclusive limit index for each input axis.
    limit_indices: Vec<usize>,

    /// Stride for each input axis (every stride is at least `1`).
    strides: Vec<usize>,
}

impl SliceOperation {
    /// Creates a new [`SliceOperation`] with the provided start and limit indices and unit strides. Use
    /// [`with_strides`](Self::with_strides) to attach non-unit strides.
    #[inline]
    pub fn new(start_indices: Vec<usize>, limit_indices: Vec<usize>) -> Self {
        let strides = vec![1; start_indices.len()];
        Self { start_indices, limit_indices, strides }
    }

    /// Replaces the strides of this [`SliceOperation`] with `strides`. There must be one stride per start index and
    /// every stride must be at least `1`.
    pub fn with_strides(mut self, strides: Vec<usize>) -> Result<Self, ProgramError> {
        if strides.len() != self.start_indices.len() {
            return Err(TypeError {
                message: format!(
                    "'slice' strides has length {} but start_indices has length {}",
                    strides.len(),
                    self.start_indices.len(),
                ),
            }
            .into());
        }
        if let Some(axis) = strides.iter().position(|stride| *stride == 0) {
            return Err(TypeError {
                message: format!("'slice' strides must be at least 1 but axis {axis} has stride 0"),
            }
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

    /// Returns the strides of this [`SliceOperation`], one per input axis.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        self.strides.as_slice()
    }
}

impl Display for SliceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for SliceOperation {
    #[inline]
    fn name(&self) -> &'static str {
        SLICE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].slice(
            self.start_indices.as_slice(),
            self.limit_indices.as_slice(),
            self.strides.as_slice(),
        ) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
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

impl<V: Value<Type = ArrayType> + Slice, C> InterpretableOperation<V, C> for SliceOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone().slice(
            self.start_indices.as_slice(),
            self.limit_indices.as_slice(),
            self.strides.as_slice(),
        )?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for SliceOperation where
    C::Operation: From<SliceOperation>
{
}

/// Trait that represents [`Operation`] types that support/include [`SliceOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`SliceOperation`]s without
/// knowing which operation type is in use.
/// Represents the ability to extract a (possibly strided) sub-array using static start, limit, and stride values.
/// This is the direct analogue of the StableHLO [`slice`](https://openxla.org/stablehlo/spec#slice) operation and
/// JAX's [`lax.slice`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.slice.html).
///
/// `t.slice(start_indices, limit_indices, strides)` returns the sub-array whose element at index `i` is the input
/// element at index `start_indices + i * strides`, with output dimension
/// `ceil((limit_indices[d] - start_indices[d]) / strides[d])` along each axis `d` (an axis with
/// `start_indices[d] == limit_indices[d]` is empty). All three slices must have length equal to the input rank, and
/// each axis must satisfy `start_indices[d] <= limit_indices[d] <= input_dimension[d]` and `strides[d] >= 1`. Slicing
/// requires static input extents: inputs with dynamic dimensions are rejected because the bounds cannot be validated
/// against an unknown extent.
///
/// # Example
///
/// The following example shows how to use [`Slice`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::Slice;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Slice the middle 1x2 block out of a 2x3 matrix. This is equivalent to
/// // `jax.lax.slice(x, start_indices=(1, 1), limit_indices=(2, 3))` in JAX.
/// let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let y = x.slice(&[1, 1], &[2, 3], &[1, 1])?;
/// // `y` has shape [1, 2] with values [[5.0, 6.0]].
/// assert_eq!(y.values, vec![5.0, 6.0]);
///
/// // A non-unit stride keeps every other element, like `x[0:6:2]` in NumPy.
/// let x = Array::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
/// let y = x.slice(&[1], &[6], &[2])?;
/// assert_eq!(y.values, vec![1.0, 3.0, 5.0]);
/// # Ok(())
/// # }
/// ```
pub trait Slice: Sized {
    /// Slices `self` between `start_indices` and `limit_indices` with `strides`. Refer to the documentation of this
    /// trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `start_indices`: Inclusive start index for each input axis.
    ///   - `limit_indices`: Exclusive limit index for each input axis.
    ///   - `strides`: Stride for each input axis (every stride must be at least `1`).
    fn slice(&self, start_indices: &[usize], limit_indices: &[usize], strides: &[usize]) -> Result<Self, ProgramError>;
}

/// Returns the output [`Sharding`] for a same-rank shape-changing operation (`slice`, `dynamic_slice`, or
/// [`pad`](super::padding)), mirroring JAX's `_get_sharding_for_varying_out_shape` (which JAX's slice, dynamic_slice,
/// and pad sharding rules all share). The operand sharding is carried through unchanged: resizing dimensions in place
/// neither changes the per-dimension placement relative to the array axes nor the pending cross-device reduction
/// state — selecting or padding elements commutes with a pending sum, so a value unreduced/reduced over a mesh axis
/// stays so (this is where ryft diverges from JAX, which leaves `dynamic_slice` on unreduced operands unimplemented).
/// The one constraint: a dimension whose size changes and is sharded over [`Explicit`](MeshAxisType::Explicit) mesh
/// axes must keep an output size divisible by the product of those axes' sizes, so the result stays evenly sharded.
/// The check is gated to explicit axes, leaving `Manual`/`Auto` shardings to `shard_map` / the compiler.
pub(crate) fn resized_output_sharding(
    operand: &ArrayType,
    output_sizes: &[Size],
    op: &'static str,
) -> Result<Option<Sharding>, TypeError> {
    let Some(sharding) = operand.sharding() else {
        return Ok(None);
    };
    for (axis, (input_size, output_size)) in operand.shape().dimensions().iter().zip(output_sizes).enumerate() {
        let (ShardingDimension::Sharded(axis_names), Size::Static(output_size)) =
            (&sharding.dimensions()[axis], output_size)
        else {
            continue;
        };
        if input_size == &Size::Static(*output_size) {
            continue;
        }
        // `product()` over an empty iterator is 1, so dimensions sharded only over Manual/Auto axes skip the check.
        let explicit_axis_product: usize = axis_names
            .iter()
            .filter(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Explicit))
            .filter_map(|name| sharding.mesh().axis_size(name))
            .product();
        if explicit_axis_product > 1 && output_size % explicit_axis_product != 0 {
            return Err(TypeError {
                message: format!(
                    "'{op}' on a dimension sharded over explicit mesh axes requires the output size ({output_size}) at \
                     axis {axis} to be divisible by the mesh-axis product ({explicit_axis_product})"
                ),
            });
        }
    }
    Ok(Some(sharding.clone()))
}

/// Returns the output [`Sharding`] for an in-place update ([`UpdateSlice`] / [`DynamicUpdateSlice`]), mirroring JAX's
/// `_dynamic_update_slice_sharding_rule` and `_dus_(un)reduced_rule`. Because the update is written into the operand
/// without resharding, the two must agree on placement and reduction state wherever an
/// [`Explicit`](MeshAxisType::Explicit) mesh axis is involved; differences confined to `Manual`/`Auto` axes are
/// tolerated (left to `shard_map` / the compiler). The output keeps the operand's sharding, except that the update's
/// [`varying_manual_axes`](Sharding::varying_manual_axes) are unioned in — the written region may vary over manual
/// axes the operand does not, so the result does too (ryft diverges from JAX here, which returns the operand sharding
/// verbatim). An operand without a sharding leaves the output unsharded.
fn update_slice_output_sharding(
    operand: &ArrayType,
    update: &ArrayType,
    op: &'static str,
) -> Result<Option<Sharding>, TypeError> {
    let Some(operand_sharding) = operand.sharding() else {
        return Ok(None);
    };
    let Some(update_sharding) = update.sharding() else {
        return Ok(Some(operand_sharding.clone()));
    };
    if operand_sharding.mesh() != update_sharding.mesh() {
        return Err(TypeError { message: format!("'{op}' operand and update must use the same mesh") });
    }
    if operand_sharding.conflicts_on_explicit_axes_with(update_sharding) {
        return Err(TypeError {
            message: format!(
                "'{op}' operand and update must be sharded identically, but got {operand_sharding} and {update_sharding}"
            ),
        });
    }
    if update_sharding.varying_manual_axes().is_subset(operand_sharding.varying_manual_axes()) {
        return Ok(Some(operand_sharding.clone()));
    }
    let varying_manual_axes = operand_sharding
        .varying_manual_axes()
        .union(update_sharding.varying_manual_axes())
        .cloned()
        .collect::<Vec<_>>();
    Sharding::with_manual_axes(
        operand_sharding.mesh().clone(),
        operand_sharding.dimensions().to_vec(),
        operand_sharding.unreduced_axes().iter().cloned().collect::<Vec<_>>(),
        operand_sharding.reduced_axes().iter().cloned().collect::<Vec<_>>(),
        varying_manual_axes,
    )
    .map(Some)
    .map_err(|error| TypeError { message: error.to_string() })
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
            return Err(TypeError {
                message: format!("'slice' start_indices has length {} but input has rank {rank}", start_indices.len(),),
            }
            .into());
        }
        if limit_indices.len() != rank {
            return Err(TypeError {
                message: format!("'slice' limit_indices has length {} but input has rank {rank}", limit_indices.len(),),
            }
            .into());
        }
        if strides.len() != rank {
            return Err(TypeError {
                message: format!("'slice' strides has length {} but input has rank {rank}", strides.len()),
            }
            .into());
        }
        let mut output_dimensions = Vec::with_capacity(rank);
        for (axis, ((&start, &limit), &stride)) in
            start_indices.iter().zip(limit_indices.iter()).zip(strides.iter()).enumerate()
        {
            let dimension = self.dimension(axis as isize);
            let Size::Static(size) = dimension else {
                return Err(TypeError {
                    message: format!(
                        "'slice' does not support dynamic input axis {axis} with size {dimension}; slice bounds \
                        cannot be validated against an unknown extent",
                    ),
                }
                .into());
            };
            if stride == 0 {
                return Err(TypeError {
                    message: format!("'slice' strides must be at least 1 but axis {axis} has stride 0"),
                }
                .into());
            }
            if start > limit {
                return Err(TypeError {
                    message: format!("'slice' start index {start} is greater than limit index {limit} at axis {axis}"),
                }
                .into());
            }
            if limit > size {
                return Err(TypeError {
                    message: format!("'slice' limit index {limit} is out of bounds for axis {axis} with size {size}"),
                }
                .into());
            }
            output_dimensions.push(Size::Static((limit - start).div_ceil(stride)));
        }
        let sharding = resized_output_sharding(self, &output_dimensions, SLICE_OPERATION_NAME)?;
        ArrayType::new(self.data_type(), Shape::new(output_dimensions))
            .with_sharding(sharding)
            .map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

/// Any context-carrying value slices by binding a [`SliceOperation`] through its own context. The
/// `From<SliceOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Slice for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<SliceOperation>,
{
    fn slice(&self, start_indices: &[usize], limit_indices: &[usize], strides: &[usize]) -> Result<Self, ProgramError> {
        let operation =
            SliceOperation::new(start_indices.to_vec(), limit_indices.to_vec()).with_strides(strides.to_vec())?;
        Ok(self.dispatch_domain().bind(operation, &[], &[], &[self.clone()])?.remove(0))
    }
}

/// [`Operation`] that overwrites a contiguous sub-array of its first operand with its second operand at static start
/// indices. Refer to the documentation of [`UpdateSlice`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UpdateSliceOperation {
    /// Inclusive start index for each input axis at which the update is written.
    start_indices: Vec<usize>,
}

impl UpdateSliceOperation {
    /// Creates a new [`UpdateSliceOperation`] with the provided start indices.
    #[inline]
    pub fn new(start_indices: Vec<usize>) -> Self {
        Self { start_indices }
    }

    /// Returns the inclusive start indices of this [`UpdateSliceOperation`], one per input axis.
    #[inline]
    pub fn start_indices(&self) -> &[usize] {
        self.start_indices.as_slice()
    }
}

impl Display for UpdateSliceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for UpdateSliceOperation {
    #[inline]
    fn name(&self) -> &'static str {
        UPDATE_SLICE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        match (&input_types[0]).update_slice(&input_types[1], self.start_indices.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("start_indices", format_args!("{:?}", self.start_indices)))
    }
}

impl<V: Value<Type = ArrayType> + UpdateSlice, C> InterpretableOperation<V, C> for UpdateSliceOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].update_slice(&inputs[1], self.start_indices.as_slice())?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for UpdateSliceOperation where
    C::Operation: From<UpdateSliceOperation>
{
}

/// Represents the ability to overwrite a contiguous sub-array with an update value at static start indices. This is
/// the statically indexed sibling of [`DynamicUpdateSlice`] and the transpose partner of [`Slice`]: writing a
/// cotangent block into a zero array at the slice offsets is exactly an update-slice of a zero input. StableHLO has
/// no statically indexed update operation, so backends lower this operation to
/// [`dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice) with constant start indices.
///
/// `t.update_slice(update, start_indices)` returns a value equal to `t` except that the block starting at
/// `start_indices` is replaced by `update`. The update must have the same data type and rank as the input, all of
/// its dimensions must be static, and each axis must satisfy
/// `start_indices[d] + update_dimension[d] <= input_dimension[d]`. Unlike [`DynamicUpdateSlice`], the start indices
/// are validated when the operation is constructed, so no index clamping occurs.
///
/// # Example
///
/// The following example shows how to use [`UpdateSlice`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::UpdateSlice;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Overwrite the last two elements of the first row of a 2x3 matrix.
/// let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let update = Array::matrix(1, 2, vec![8.0, 9.0]);
/// let y = x.update_slice(&update, &[0, 1])?;
/// assert_eq!(y.values, vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);
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
        if self.data_type() != update.data_type() {
            return Err(TypeError {
                message: format!(
                    "'update_slice' input data type {} does not match update data type {}",
                    self.data_type(),
                    update.data_type(),
                ),
            }
            .into());
        }
        let rank = self.rank();
        if update.rank() != rank {
            return Err(TypeError {
                message: format!("'update_slice' update has rank {} but input has rank {rank}", update.rank()),
            }
            .into());
        }
        if start_indices.len() != rank {
            return Err(TypeError {
                message: format!(
                    "'update_slice' start_indices has length {} but input has rank {rank}",
                    start_indices.len(),
                ),
            }
            .into());
        }
        for (axis, &start) in start_indices.iter().enumerate() {
            let update_dimension = update.dimension(axis as isize);
            let Size::Static(update_size) = update_dimension else {
                return Err(TypeError {
                    message: format!(
                        "'update_slice' does not support dynamic update axis {axis} with size {update_dimension}; \
                        update shapes must be static",
                    ),
                }
                .into());
            };
            let input_dimension = self.dimension(axis as isize);
            let Size::Static(input_size) = input_dimension else {
                return Err(TypeError {
                    message: format!(
                        "'update_slice' cannot prove that the update fits along dynamic input axis {axis} with \
                        size {input_dimension}",
                    ),
                }
                .into());
            };
            if start + update_size > input_size {
                return Err(TypeError {
                    message: format!(
                        "'update_slice' update axis {axis} with start index {start} and size {update_size} does not \
                        fit in input size {input_size}",
                    ),
                }
                .into());
            }
        }
        // The output is distributed like the input operand (the update is written in place); the operand's placement
        // and reduction state carry through, with the update's varying-manual axes folded in.
        let sharding = update_slice_output_sharding(self, update, UPDATE_SLICE_OPERATION_NAME)?;
        self.clone()
            .with_sharding(sharding)
            .map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

/// Any context-carrying value updates a slice by binding an [`UpdateSliceOperation`] through its own context. The
/// `From<UpdateSliceOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> UpdateSlice for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<UpdateSliceOperation>,
{
    fn update_slice(&self, update: &Self, start_indices: &[usize]) -> Result<Self, ProgramError> {
        let mut outputs = self.dispatch_domain().bind(
            UpdateSliceOperation::new(start_indices.to_vec()),
            &[],
            &[],
            &[self.clone(), update.clone()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// [`Operation`] that extracts a statically shaped sub-array from its input at start indices that are computed at
/// run time. Refer to the documentation of [`DynamicSlice`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicSliceOperation {
    /// Size of the extracted slice along each input axis.
    sizes: Vec<usize>,
}

impl DynamicSliceOperation {
    /// Creates a new [`DynamicSliceOperation`] with the provided slice sizes.
    #[inline]
    pub fn new(sizes: Vec<usize>) -> Self {
        Self { sizes }
    }

    /// Returns the slice sizes of this [`DynamicSliceOperation`], one per input axis.
    #[inline]
    pub fn sizes(&self) -> &[usize] {
        self.sizes.as_slice()
    }
}

impl Display for DynamicSliceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for DynamicSliceOperation {
    #[inline]
    fn name(&self) -> &'static str {
        DYNAMIC_SLICE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.is_empty() {
            return Err(TypeError {
                message: "'dynamic_slice' expects an input operand followed by its start index operands but got no \
                    inputs"
                    .to_string(),
            });
        }
        match input_types[0].dynamic_slice(&input_types[1..], self.sizes.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("sizes", format_args!("{:?}", self.sizes)))
    }
}

impl<V: Value<Type = ArrayType> + DynamicSlice, C> InterpretableOperation<V, C> for DynamicSliceOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        let [input, start_indices @ ..] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1 + self.sizes.len(), actual: 0 });
        };
        Ok(vec![input.dynamic_slice(start_indices, self.sizes.as_slice())?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DynamicSliceOperation where
    C::Operation: From<DynamicSliceOperation>
{
}

/// Represents the ability to extract a statically shaped sub-array at start indices that are computed at run time.
/// This is the direct analogue of the StableHLO
/// [`dynamic_slice`](https://openxla.org/stablehlo/spec#dynamic_slice) operation and JAX's
/// [`lax.dynamic_slice`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html).
///
/// `t.dynamic_slice(start_indices, sizes)` extracts the block of shape `sizes` whose origin is given by the scalar
/// integer values in `start_indices` (one per input axis). Start indices are clamped per StableHLO semantics so the
/// extracted block always lies in bounds: the effective start index along axis `d` is
/// `clamp(0, start_indices[d], input_dimension[d] - sizes[d])`. The output shape is exactly `sizes` and is therefore
/// fully static even though the slice origin is not. Each static input axis must satisfy
/// `sizes[d] <= input_dimension[d]`. A [`Size::Dynamic`] input axis is accepted: the clamp keeps the read in bounds
/// against any runtime extent, and the output dimension is still the static `sizes[d]`, so no bound needs to be
/// proven against the unknown extent. This is what lets a dynamically-sized stack (such as the residual stacks of an
/// unbounded-loop pullback) be read iteration by iteration.
///
/// # Example
///
/// The following example shows how to use [`DynamicSlice`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::DynamicSlice;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// # use ryft_core::types::{ArrayType, DataType};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Extract a 1x2 block starting at row 1, column 1 of a 2x3 matrix. This is equivalent to
/// // `jax.lax.dynamic_slice(x, (i, j), slice_sizes=(1, 2))` in JAX.
/// let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let i = Array::new(ArrayType::scalar(DataType::I32), vec![1.0]);
/// let j = Array::new(ArrayType::scalar(DataType::I32), vec![1.0]);
/// let y = x.dynamic_slice(&[i, j], &[1, 2])?;
/// // `y` has shape [1, 2] with values [[5.0, 6.0]].
/// assert_eq!(y.values, vec![5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait DynamicSlice: Sized {
    /// Extracts the block of shape `sizes` starting at `start_indices` from `self`. Refer to the documentation of
    /// this trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `start_indices`: Scalar integer start index values, one per input axis, clamped to keep the extracted
    ///     block in bounds.
    ///   - `sizes`: Size of the extracted slice along each input axis.
    fn dynamic_slice(&self, start_indices: &[Self], sizes: &[usize]) -> Result<Self, ProgramError>;
}

impl DynamicSlice for ArrayType {
    fn dynamic_slice(&self, start_indices: &[Self], sizes: &[usize]) -> Result<ArrayType, ProgramError> {
        let rank = self.rank();
        if start_indices.len() != rank {
            return Err(TypeError {
                message: format!(
                    "'dynamic_slice' expects one start index per input axis ({rank}) but got {}",
                    start_indices.len(),
                ),
            }
            .into());
        }
        if sizes.len() != rank {
            return Err(TypeError {
                message: format!("'dynamic_slice' sizes has length {} but input has rank {rank}", sizes.len()),
            }
            .into());
        }
        validate_start_index_types(DYNAMIC_SLICE_OPERATION_NAME, start_indices)?;
        for (axis, &size) in sizes.iter().enumerate() {
            // A dynamic input axis is accepted: StableHLO clamps the start index into
            // `[0, input_dimension - size]`, so the read always stays in bounds and the output shape is the static
            // `sizes` regardless of the unknown extent. A static input axis still validates the bound eagerly.
            if let Size::Static(input_size) = self.dimension(axis as isize) {
                if size > input_size {
                    return Err(TypeError {
                        message: format!(
                            "'dynamic_slice' size {size} is out of bounds for axis {axis} with size {input_size}",
                        ),
                    }
                    .into());
                }
            }
        }
        let output_dimensions: Vec<Size> = sizes.iter().map(|size| Size::Static(*size)).collect();
        let sharding = resized_output_sharding(self, &output_dimensions, DYNAMIC_SLICE_OPERATION_NAME)?;
        ArrayType::new(self.data_type(), Shape::new(output_dimensions))
            .with_sharding(sharding)
            .map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

/// Any context-carrying value dynamic-slices by binding a [`DynamicSliceOperation`] through its own context. The
/// `From<DynamicSliceOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> DynamicSlice for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<DynamicSliceOperation>,
{
    fn dynamic_slice(&self, start_indices: &[Self], sizes: &[usize]) -> Result<Self, ProgramError> {
        let mut inputs = Vec::with_capacity(1 + start_indices.len());
        inputs.push(self.clone());
        inputs.extend(start_indices.iter().cloned());
        Ok(self
            .dispatch_domain()
            .bind(DynamicSliceOperation::new(sizes.to_vec()), &[], &[], &inputs)?
            .remove(0))
    }
}

/// [`Operation`] that overwrites a contiguous sub-array of its first operand with its second operand at start
/// indices that are computed at run time. Refer to the documentation of [`DynamicUpdateSlice`] for more information.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicUpdateSliceOperation;

impl Display for DynamicUpdateSliceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for DynamicUpdateSliceOperation {
    #[inline]
    fn name(&self) -> &'static str {
        DYNAMIC_UPDATE_SLICE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() < 2 {
            return Err(TypeError {
                message: format!(
                    "'dynamic_update_slice' expects an input operand and an update operand followed by start index \
                    operands but got {} inputs",
                    input_types.len(),
                ),
            });
        }
        match input_types[0].dynamic_update_slice(&input_types[1], &input_types[2..]) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name()).map(|_| ())
    }
}

impl<V: Value<Type = ArrayType> + DynamicUpdateSlice, C> InterpretableOperation<V, C> for DynamicUpdateSliceOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        let [input, update, start_indices @ ..] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() });
        };
        Ok(vec![input.dynamic_update_slice(update, start_indices)?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DynamicUpdateSliceOperation where
    C::Operation: From<DynamicUpdateSliceOperation>
{
}

/// Represents the ability to overwrite a contiguous sub-array with an update value at start indices that are
/// computed at run time. This is the direct analogue of the StableHLO
/// [`dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice) operation and JAX's
/// [`lax.dynamic_update_slice`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_update_slice.html).
///
/// `t.dynamic_update_slice(update, start_indices)` returns a value equal to `t` except that the block whose origin
/// is given by the scalar integer values in `start_indices` (one per input axis) is replaced by `update`. Start
/// indices are clamped per StableHLO semantics so the updated block always lies in bounds: the effective start index
/// along axis `d` is `clamp(0, start_indices[d], input_dimension[d] - update_dimension[d])`. The update must have
/// the same data type and rank as the input, all of its dimensions must be static, and each axis must satisfy
/// `update_dimension[d] <= input_dimension[d]`; inputs with dynamic dimensions are rejected because that bound
/// cannot be proven against an unknown extent.
///
/// # Example
///
/// The following example shows how to use [`DynamicUpdateSlice`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::DynamicUpdateSlice;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// # use ryft_core::types::{ArrayType, DataType};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Overwrite the last two elements of the first row of a 2x3 matrix. This is equivalent to
/// // `jax.lax.dynamic_update_slice(x, update, (i, j))` in JAX.
/// let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let update = Array::matrix(1, 2, vec![8.0, 9.0]);
/// let i = Array::new(ArrayType::scalar(DataType::I32), vec![0.0]);
/// let j = Array::new(ArrayType::scalar(DataType::I32), vec![1.0]);
/// let y = x.dynamic_update_slice(&update, &[i, j])?;
/// assert_eq!(y.values, vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait DynamicUpdateSlice: Sized {
    /// Overwrites the block of `self` starting at `start_indices` with `update`. Refer to the documentation of this
    /// trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `update`: Value written into `self`. Must have the same data type and rank as `self`, static dimensions,
    ///     and dimensions that do not exceed those of `self`.
    ///   - `start_indices`: Scalar integer start index values, one per input axis, clamped to keep the updated block
    ///     in bounds.
    fn dynamic_update_slice(&self, update: &Self, start_indices: &[Self]) -> Result<Self, ProgramError>;
}

impl DynamicUpdateSlice for ArrayType {
    fn dynamic_update_slice(&self, update: &Self, start_indices: &[Self]) -> Result<ArrayType, ProgramError> {
        if self.data_type() != update.data_type() {
            return Err(TypeError {
                message: format!(
                    "'dynamic_update_slice' input data type {} does not match update data type {}",
                    self.data_type(),
                    update.data_type(),
                ),
            }
            .into());
        }
        let rank = self.rank();
        if update.rank() != rank {
            return Err(TypeError {
                message: format!("'dynamic_update_slice' update has rank {} but input has rank {rank}", update.rank()),
            }
            .into());
        }
        if start_indices.len() != rank {
            return Err(TypeError {
                message: format!(
                    "'dynamic_update_slice' expects one start index per input axis ({rank}) but got {}",
                    start_indices.len(),
                ),
            }
            .into());
        }
        validate_start_index_types(DYNAMIC_UPDATE_SLICE_OPERATION_NAME, start_indices)?;
        for axis in 0..rank {
            let update_dimension = update.dimension(axis as isize);
            let Size::Static(update_size) = update_dimension else {
                return Err(TypeError {
                    message: format!(
                        "'dynamic_update_slice' does not support dynamic update axis {axis} with size \
                        {update_dimension}; update shapes must be static",
                    ),
                }
                .into());
            };
            let input_dimension = self.dimension(axis as isize);
            let Size::Static(input_size) = input_dimension else {
                return Err(TypeError {
                    message: format!(
                        "'dynamic_update_slice' cannot prove that the update fits along dynamic input axis {axis} \
                        with size {input_dimension}",
                    ),
                }
                .into());
            };
            if update_size > input_size {
                return Err(TypeError {
                    message: format!(
                        "'dynamic_update_slice' update axis {axis} has size {update_size} which exceeds input size \
                        {input_size}",
                    ),
                }
                .into());
            }
        }
        // The output is distributed like the input operand (the update is written in place); the operand's placement
        // and reduction state carry through, with the update's varying-manual axes folded in.
        let sharding = update_slice_output_sharding(self, update, DYNAMIC_UPDATE_SLICE_OPERATION_NAME)?;
        self.clone()
            .with_sharding(sharding)
            .map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

/// Any context-carrying value dynamic-update-slices by binding a [`DynamicUpdateSliceOperation`] through its own
/// context. The `From<DynamicUpdateSliceOperation>` bound makes this disjoint from the eager value types (whose
/// context operation is `ConstantOperation`), so it covers the transform tracers without conflicting with the
/// concrete implementations.
impl<V: Value<Type = ArrayType>> DynamicUpdateSlice for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<DynamicUpdateSliceOperation>,
{
    fn dynamic_update_slice(&self, update: &Self, start_indices: &[Self]) -> Result<Self, ProgramError> {
        let mut inputs = vec![self.clone(), update.clone()];
        inputs.extend(start_indices.iter().cloned());
        let mut outputs = self.dispatch_domain().bind(DynamicUpdateSliceOperation, &[], &[], &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

// TODO(eaplatanios): Should this be renamed to something that's not about "linearity"? This is about captured primals.
/// Captured-index dynamic slice linear operation: the linear map `t ↦ dynamic_slice(t, start_indices, sizes)` over
/// the tangent (or cotangent) of the sliced operand.
///
/// It is the captured-index linear map emitted by the JVP of [`DynamicSliceOperation`]:
/// the scalar integer start indices are primal values captured at linearization time as residual factors (they have
/// no tangent space, so the map is linear in the single tangent operand), while its transpose scatters the output
/// cotangent back into the read window. The single operation input is the sliced operand's tangent; the captured
/// `start_indices` are appended as the dynamic slice's index operands during type inference.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearDynamicSliceOperation<F> {
    /// Captured scalar integer start index factors, one per input axis.
    start_indices: Vec<F>,

    /// Size of the extracted slice along each input axis.
    sizes: Vec<usize>,
}

impl<F> LinearDynamicSliceOperation<F> {
    /// Creates a new [`LinearDynamicSliceOperation`] capturing the provided start index factors and slice sizes.
    #[inline]
    pub fn new(start_indices: Vec<F>, sizes: Vec<usize>) -> Self {
        Self { start_indices, sizes }
    }

    /// Returns the captured scalar integer start index factors, one per input axis.
    #[inline]
    pub fn start_indices(&self) -> &[F] {
        self.start_indices.as_slice()
    }

    /// Returns the slice sizes of this [`LinearDynamicSliceOperation`], one per input axis.
    #[inline]
    pub fn sizes(&self) -> &[usize] {
        self.sizes.as_slice()
    }
}

impl<F: Value<Type = ArrayType>> Display for LinearDynamicSliceOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<F: Value<Type = ArrayType>> Operation<ArrayType> for LinearDynamicSliceOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        DYNAMIC_SLICE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let mut full_input_types = input_types.to_vec();
        full_input_types.extend(self.start_indices.iter().map(|index| index.r#type().into_owned()));
        DynamicSliceOperation::new(self.sizes.clone()).infer_output_types(full_input_types.as_slice())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("start_indices", format_args!("{}", render_factor_list(&self.start_indices)))?;
            operation.field("sizes", format_args!("{:?}", self.sizes))
        })
    }
}

impl<V, F, C> InterpretableOperation<V, C> for LinearDynamicSliceOperation<F>
where
    V: Value<Type = ArrayType> + DynamicSlice,
    F: CustomVjpResidual<V>,
{
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let start_indices =
            self.start_indices().iter().map(|index| index.residual_value()).collect::<Result<Vec<_>, _>>()?;
        Ok(vec![inputs[0].dynamic_slice(start_indices.as_slice(), self.sizes())?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate) for a [`LinearDynamicSliceOperation`].
impl<F: Value<Type = ArrayType>, C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C>
    for LinearDynamicSliceOperation<F>
where
    C::Operation: From<LinearDynamicSliceOperation<F>>,
{
}

/// Transpose rule for the captured-index dynamic slice. The forward linear map
/// `t ↦ dynamic_slice(t, start_indices, sizes)` transposes by scattering the output cotangent back into a zero array
/// of the input type at the same captured indices, i.e. a captured-index dynamic update-slice with the same start
/// indices. Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O, F: Value<Type = ArrayType>> TransposableOperation<V, O>
    for LinearDynamicSliceOperation<F>
where
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<LinearDynamicUpdateSliceOperation<F>>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().into_owned())]),
            MaybeZero::Value(cotangent) => {
                let zeros = MaybeZero::Zero(inputs[0].r#type().into_owned()).materialize(context)?;
                let outputs = context.stage_operation(
                    LinearDynamicUpdateSliceOperation::new(self.start_indices().to_vec()),
                    &[zeros, cotangent.clone()],
                )?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![MaybeZero::Value(outputs.into_iter().next().unwrap())])
            }
        }
    }
}

/// Partition-aware transpose rule for the primal [`DynamicSliceOperation`]. The scalar integer start indices
/// (operands 1 onward) have no tangent space, so in a valid pushforward they are the known operands and the sliced
/// operand (operand 0) is the linear one. The forward map `t ↦ dynamic_slice(t, start_indices, sizes)` transposes by
/// scattering the output cotangent back into a zero array of the operand type at the same start indices, i.e. a
/// dynamic update-slice at those indices. This reproduces the captured-index [`LinearDynamicSliceOperation`] transpose
/// rule, reading the start indices from the pullback through `operand_values` and staging a primal
/// [`DynamicUpdateSliceOperation`] instead of folding the indices into captured factors. The start indices receive
/// structural zeros, and a zero output cotangent stays a structural zero.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for DynamicSliceOperation
where
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<DynamicUpdateSliceOperation>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        if inputs.is_empty() {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        }
        check_count!("output", outputs, 1, ProgramError);
        // One structural zero per operand: a contribution for the linear operand and zeros for the known indices.
        let mut contributions =
            inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect::<Vec<_>>();
        if let MaybeZero::Value(cotangent) = &outputs[0] {
            let start_indices = read_known_start_indices(&inputs[1..]);
            let zeros = MaybeZero::Zero(inputs[0].r#type().into_owned()).materialize(context)?;
            let mut operands = Vec::with_capacity(2 + start_indices.len());
            operands.push(zeros);
            operands.push(cotangent.clone());
            operands.extend(start_indices);
            let outputs = context.stage_operation(DynamicUpdateSliceOperation, operands.as_slice())?;
            check_count!("output", outputs, 1, ProgramError);
            contributions[0] = MaybeZero::Value(outputs.into_iter().next().unwrap());
        }
        Ok(contributions)
    }
}

// TODO(eaplatanios): Should this be renamed to something that's not about "linearity"? This is about captured primals.
/// Captured-index dynamic update-slice linear operation: the linear map
/// `(t, u) ↦ dynamic_update_slice(t, u, start_indices)` over the tangents (or cotangents) of the input and update
/// operands.
///
/// It is the captured-index linear map emitted by the JVP of
/// [`DynamicUpdateSliceOperation`]: the scalar integer start indices are primal values captured at linearization time
/// as residual factors (they have no tangent space, so the map is jointly linear in the two tangent operands). The
/// two operation inputs are the input and update tangents; the captured `start_indices` are appended as the dynamic
/// update-slice's index operands during type inference.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearDynamicUpdateSliceOperation<F> {
    /// Captured scalar integer start index factors, one per input axis.
    start_indices: Vec<F>,
}

impl<F> LinearDynamicUpdateSliceOperation<F> {
    /// Creates a new [`LinearDynamicUpdateSliceOperation`] capturing the provided start index factors.
    #[inline]
    pub fn new(start_indices: Vec<F>) -> Self {
        Self { start_indices }
    }

    /// Returns the captured scalar integer start index factors, one per input axis.
    #[inline]
    pub fn start_indices(&self) -> &[F] {
        self.start_indices.as_slice()
    }
}

impl<F: Value<Type = ArrayType>> Display for LinearDynamicUpdateSliceOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<F: Value<Type = ArrayType>> Operation<ArrayType> for LinearDynamicUpdateSliceOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        DYNAMIC_UPDATE_SLICE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        let mut full_input_types = input_types.to_vec();
        full_input_types.extend(self.start_indices.iter().map(|index| index.r#type().into_owned()));
        DynamicUpdateSliceOperation.infer_output_types(full_input_types.as_slice())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("start_indices", format_args!("{}", render_factor_list(&self.start_indices)))
        })
    }
}

impl<V, F, C> InterpretableOperation<V, C> for LinearDynamicUpdateSliceOperation<F>
where
    V: Value<Type = ArrayType> + DynamicUpdateSlice,
    F: CustomVjpResidual<V>,
{
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let start_indices =
            self.start_indices().iter().map(|index| index.residual_value()).collect::<Result<Vec<_>, _>>()?;
        Ok(vec![inputs[0].dynamic_update_slice(&inputs[1], start_indices.as_slice())?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate) for a [`LinearDynamicUpdateSliceOperation`].
impl<F: Value<Type = ArrayType>, C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C>
    for LinearDynamicUpdateSliceOperation<F>
where
    C::Operation: From<LinearDynamicUpdateSliceOperation<F>>,
{
}

/// Transpose rule for the captured-index dynamic update-slice. The forward linear map
/// `(t, u) ↦ dynamic_update_slice(t, u, start_indices)` splits the output cotangent into two contributions at the same
/// captured indices: the input cotangent is the cotangent with the update window zeroed (a captured-index dynamic
/// update-slice with the same start indices) and the update cotangent is the dynamic slice of the cotangent at the
/// update window (a captured-index dynamic slice with the same start indices). Symbolic-zero cotangents propagate
/// unchanged.
impl<V: Value<Type = ArrayType>, O, F: Value<Type = ArrayType>> TransposableOperation<V, O>
    for LinearDynamicUpdateSliceOperation<F>
where
    O: Operation<ArrayType>
        + From<ZeroOperation<ArrayType>>
        + From<LinearDynamicUpdateSliceOperation<F>>
        + From<LinearDynamicSliceOperation<F>>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(vec![
                MaybeZero::Zero(inputs[0].r#type().into_owned()),
                MaybeZero::Zero(inputs[1].r#type().into_owned()),
            ]),
            MaybeZero::Value(cotangent) => {
                let update_sizes = static_update_sizes("'dynamic_update_slice' transpose", &inputs[1].r#type())?;
                let zeros = MaybeZero::Zero(inputs[1].r#type().into_owned()).materialize(context)?;
                let input_cotangents = context.stage_operation(
                    LinearDynamicUpdateSliceOperation::new(self.start_indices().to_vec()),
                    &[cotangent.clone(), zeros],
                )?;
                check_count!("output", input_cotangents, 1, ProgramError);
                let update_cotangents = context.stage_operation(
                    LinearDynamicSliceOperation::new(self.start_indices().to_vec(), update_sizes),
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", update_cotangents, 1, ProgramError);
                Ok(vec![
                    MaybeZero::Value(input_cotangents.into_iter().next().unwrap()),
                    MaybeZero::Value(update_cotangents.into_iter().next().unwrap()),
                ])
            }
        }
    }
}

/// Reads the known scalar integer start-index operands of a dynamic slicing operation from the pullback. Each entry of
/// `inputs` is the start index's [`PartialValue`]; the dispatch guarantees a [`Known`](PartialValue::Known) operand
/// carries its pullback value, so each tracer is read directly.
fn read_known_start_indices<V: Value<Type = ArrayType>, O: Operation<ArrayType>>(
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
) -> Vec<Tracer<TracingContext<V, O>>> {
    inputs
        .iter()
        .map(|input| input.as_known().expect("dispatch guarantees a known operand carries its pullback value").clone())
        .collect()
}

/// Partition-aware transpose rule for the primal [`DynamicUpdateSliceOperation`]. The scalar integer start indices
/// (operands 2 onward) have no tangent space, so in a valid pushforward they are the known operands and the input and
/// update (operands 0 and 1) are the linear ones. The forward map
/// `(t, u) ↦ dynamic_update_slice(t, u, start_indices)` splits the output cotangent into two contributions at the
/// same start indices: the input cotangent is the cotangent with the update window zeroed (a dynamic update-slice
/// writing zeros at the indices) and the update cotangent is the dynamic slice of the cotangent at the update window.
/// This reproduces the captured-index [`LinearDynamicUpdateSliceOperation`] transpose rule, reading the start indices
/// from the pullback through `operand_values` and staging primal slicing operations instead of folding the indices
/// into captured factors. The start indices receive structural zeros, and a zero output cotangent stays a structural
/// zero.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for DynamicUpdateSliceOperation
where
    O: Operation<ArrayType>
        + From<ZeroOperation<ArrayType>>
        + From<DynamicUpdateSliceOperation>
        + From<DynamicSliceOperation>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }
        check_count!("output", outputs, 1, ProgramError);
        // One structural zero per operand: contributions for the linear input and update, and zeros for the known
        // start indices.
        let mut contributions =
            inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect::<Vec<_>>();
        if let MaybeZero::Value(cotangent) = &outputs[0] {
            let update_sizes = static_update_sizes("'dynamic_update_slice' transpose", &inputs[1].r#type())?;
            let start_indices = read_known_start_indices(&inputs[2..]);
            let zeros = MaybeZero::Zero(inputs[1].r#type().into_owned()).materialize(context)?;
            // Input cotangent: the output cotangent with the update window overwritten by zeros.
            let mut input_operands = Vec::with_capacity(2 + start_indices.len());
            input_operands.push(cotangent.clone());
            input_operands.push(zeros);
            input_operands.extend(start_indices.iter().cloned());
            let input_cotangents = context.stage_operation(DynamicUpdateSliceOperation, input_operands.as_slice())?;
            check_count!("output", input_cotangents, 1, ProgramError);
            // Update cotangent: the dynamic slice of the output cotangent at the update window.
            let mut update_operands = Vec::with_capacity(1 + start_indices.len());
            update_operands.push(cotangent.clone());
            update_operands.extend(start_indices);
            let update_cotangents =
                context.stage_operation(DynamicSliceOperation::new(update_sizes), update_operands.as_slice())?;
            check_count!("output", update_cotangents, 1, ProgramError);
            contributions[0] = MaybeZero::Value(input_cotangents.into_iter().next().unwrap());
            contributions[1] = MaybeZero::Value(update_cotangents.into_iter().next().unwrap());
        }
        Ok(contributions)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::tests::TestArray;
    use crate::types::Typed;

    use super::*;

    /// Returns a scalar integer-typed test array carrying `value` as its in-band payload.
    fn index(value: f64) -> TestArray {
        TestArray::new(ArrayType::scalar(DataType::I32), vec![value])
    }

    #[test]
    fn test_slice() {
        let operation = SliceOperation::new(vec![1, 1], vec![2, 3]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), SLICE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "slice [start_indices=[1, 1], limit_indices=[2, 3]]");
        assert_eq!(operation.start_indices(), &[1, 1]);
        assert_eq!(operation.limit_indices(), &[2, 3]);

        // Type inference validates the slice bounds and returns the sliced type, and the type-level (abstract)
        // capability backs it without consuming the borrowed input type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2)]));
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input_type)), Ok(vec![output_type.clone()]));
        assert_eq!(input_type.slice(&[1, 1], &[2, 3], &[1, 1]), Ok(output_type.clone()));

        // Interpretation copies the selected block out of the row-major payload.
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output =
            operation.interpret(&crate::EagerContext::<TestArray>::new(), std::slice::from_ref(&input)).unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].values, vec![5.0, 6.0]);

        // Empty slices produce empty payloads and rank-0 slices pass through.
        let empty = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .slice(&[1, 1], &[1, 3], &[1, 1])
            .unwrap();
        assert_eq!(empty.values, Vec::<f64>::new());
        let scalar = TestArray::scalar(42.0).slice(&[], &[], &[]).unwrap();
        assert_eq!(scalar.values, vec![42.0]);

        // Strided operations carry their strides through the builder, accessors, rendering, and inference: the
        // output dimension per axis is `ceil((limit - start) / stride)`.
        let strided = SliceOperation::new(vec![1], vec![6]).with_strides(vec![2]).unwrap();
        assert_eq!(strided.strides(), &[2]);
        assert_eq!(format!("{strided}"), "slice [start_indices=[1], limit_indices=[6], strides=[2]]");
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(6)]));
        assert_eq!(
            strided.infer_output_types(std::slice::from_ref(&vector_type)),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]))]),
        );

        // Strided interpretation keeps the elements at `start + i * stride`.
        let vector = TestArray::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let strided_output =
            strided.interpret(&crate::EagerContext::<TestArray>::new(), std::slice::from_ref(&vector)).unwrap();
        assert_eq!(strided_output[0].values, vec![1.0, 3.0, 5.0]);

        // A stride larger than the sliced extent keeps a single element, and `start == limit` keeps none.
        let single = TestArray::vector(vec![0.0, 1.0, 2.0, 3.0]).slice(&[1], &[4], &[5]).unwrap();
        assert_eq!(single.values, vec![1.0]);
        let strided_empty = TestArray::vector(vec![0.0, 1.0, 2.0, 3.0]).slice(&[2], &[2], &[2]).unwrap();
        assert_eq!(*strided_empty.r#type(), ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0)])));
        assert_eq!(strided_empty.values, Vec::<f64>::new());

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))]),
            Err(TypeError { message: "'slice' start_indices has length 2 but input has rank 1".to_string() }),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2]).infer_output_types(std::slice::from_ref(&input_type)),
            Err(TypeError { message: "'slice' limit_indices has length 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            SliceOperation::new(vec![2, 0], vec![1, 3]).infer_output_types(std::slice::from_ref(&input_type)),
            Err(TypeError { message: "'slice' start index 2 is greater than limit index 1 at axis 0".to_string() }),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2, 4]).infer_output_types(std::slice::from_ref(&input_type)),
            Err(TypeError { message: "'slice' limit index 4 is out of bounds for axis 1 with size 3".to_string() }),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2, 3]).with_strides(vec![2]),
            Err(ProgramError::Type(TypeError {
                message: "'slice' strides has length 1 but start_indices has length 2".to_string(),
            })),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2, 3]).with_strides(vec![1, 0]),
            Err(ProgramError::Type(TypeError {
                message: "'slice' strides must be at least 1 but axis 1 has stride 0".to_string(),
            })),
        );
        assert_eq!(
            input_type.slice(&[0, 0], &[2, 3], &[1]),
            Err(ProgramError::Type(TypeError {
                message: "'slice' strides has length 1 but input has rank 2".to_string(),
            })),
        );
        assert_eq!(
            input_type.slice(&[0, 0], &[2, 3], &[1, 0]),
            Err(ProgramError::Type(TypeError {
                message: "'slice' strides must be at least 1 but axis 1 has stride 0".to_string(),
            })),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(
                DataType::F64,
                Shape::new(vec![Size::Dynamic(None), Size::Static(3)]),
            )]),
            Err(TypeError {
                message: "'slice' does not support dynamic input axis 0 with size *; slice bounds cannot be \
                    validated against an unknown extent"
                    .to_string(),
            }),
        );
        assert_eq!(
            InterpretableOperation::<TestArray, crate::EagerContext<TestArray>>::interpret(
                &operation,
                &crate::EagerContext::<TestArray>::new(),
                &[]
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured indices.
        let mut builder = ProgramBuilder::<TestArray, SliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, vec![program_input]).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![program_output], Placeholder, Placeholder).unwrap();
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
    fn test_slice_operation_rejects_malformed_strides() {
        // Building a `SliceOperation` with `with_strides` validates the strides eagerly, so malformed input (here a
        // stride of `0`) returns an error instead of panicking inside the operation.
        let result = SliceOperation::new(vec![0], vec![2]).with_strides(vec![0]);
        assert!(matches!(result, Err(ProgramError::Type(_))));
    }

    #[test]
    fn test_update_slice() {
        let operation = UpdateSliceOperation::new(vec![0, 1]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), UPDATE_SLICE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "update_slice [start_indices=[0, 1]]");
        assert_eq!(operation.start_indices(), &[0, 1]);

        // Type inference validates that the update fits and returns the input type, and the type-level (abstract)
        // capability backs it without consuming the borrowed input type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2)]));
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), update_type.clone()]),
            Ok(vec![input_type.clone()]),
        );
        assert_eq!((&input_type).update_slice(&update_type, &[0, 1]), Ok(input_type.clone()));

        // Interpretation overwrites the selected block of the row-major payload.
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let update = TestArray::matrix(1, 2, vec![8.0, 9.0]);
        let output = operation.interpret(&crate::EagerContext::<TestArray>::new(), &[input, update]).unwrap();
        assert_eq!(*output[0].r#type(), input_type);
        assert_eq!(output[0].values, vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);

        // Rank-0 updates replace the input entirely.
        let scalar = TestArray::scalar(1.0).update_slice(&TestArray::scalar(7.0), &[]).unwrap();
        assert_eq!(scalar.values, vec![7.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 2 inputs but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(2)])),
            ]),
            Err(TypeError {
                message: "'update_slice' input data type f64 does not match update data type f32".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),
            ]),
            Err(TypeError { message: "'update_slice' update has rank 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            UpdateSliceOperation::new(vec![0]).infer_output_types(&[input_type.clone(), update_type.clone()]),
            Err(TypeError { message: "'update_slice' start_indices has length 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(2)])),
            ]),
            Err(TypeError {
                message: "'update_slice' does not support dynamic update axis 0 with size *; update shapes must be \
                    static"
                    .to_string(),
            }),
        );
        assert_eq!(
            UpdateSliceOperation::new(vec![0, 2]).infer_output_types(&[input_type.clone(), update_type.clone()]),
            Err(TypeError {
                message: "'update_slice' update axis 1 with start index 2 and size 2 does not fit in input size 3"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(Some(4)), Size::Static(3)])),
                update_type.clone(),
            ]),
            Err(TypeError {
                message: "'update_slice' cannot prove that the update fits along dynamic input axis 0 with size <4"
                    .to_string(),
            }),
        );
        assert_eq!(
            InterpretableOperation::<TestArray, crate::EagerContext<TestArray>>::interpret(
                &operation,
                &crate::EagerContext::<TestArray>::new(),
                &[]
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured indices.
        let mut builder = ProgramBuilder::<TestArray, UpdateSliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_update = builder.add_input(update_type);
        let program_output = builder.add_instruction(operation, vec![program_input, program_update]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, TestArray>(vec![program_output], vec![Placeholder, Placeholder], Placeholder)
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
    fn test_dynamic_slice() {
        let operation = DynamicSliceOperation::new(vec![1, 2]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), DYNAMIC_SLICE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "dynamic_slice [sizes=[1, 2]]");
        assert_eq!(operation.sizes(), &[1, 2]);

        // Type inference validates the sizes and index operand types and returns the statically shaped output.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let index_type = ArrayType::scalar(DataType::I32);
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2)]));
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), index_type.clone(), index_type.clone()]),
            Ok(vec![output_type.clone()]),
        );
        assert_eq!(
            input_type.dynamic_slice(&[index_type.clone(), index_type.clone()], &[1, 2]),
            Ok(output_type.clone()),
        );

        // Interpretation extracts the block at the in-band start indices.
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = operation
            .interpret(&crate::EagerContext::<TestArray>::new(), &[input.clone(), index(1.0), index(1.0)])
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].values, vec![5.0, 6.0]);

        // Out-of-bounds start indices clamp per StableHLO semantics: the effective start index along axis `d` is
        // `clamp(0, start_indices[d], input_dimension[d] - sizes[d])`.
        let clamped = operation
            .interpret(&crate::EagerContext::<TestArray>::new(), &[input.clone(), index(5.0), index(-2.0)])
            .unwrap();
        assert_eq!(clamped[0].values, vec![4.0, 5.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError {
                message: "'dynamic_slice' expects an input operand followed by its start index operands but got no \
                    inputs"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), index_type.clone()]),
            Err(TypeError {
                message: "'dynamic_slice' expects one start index per input axis (2) but got 1".to_string()
            }),
        );
        assert_eq!(
            DynamicSliceOperation::new(vec![1]).infer_output_types(&[
                input_type.clone(),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError { message: "'dynamic_slice' sizes has length 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            DynamicSliceOperation::new(vec![1, 4]).infer_output_types(&[
                input_type.clone(),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError { message: "'dynamic_slice' size 4 is out of bounds for axis 1 with size 3".to_string() }),
        );
        // A dynamic input axis is accepted: StableHLO clamping keeps the read in bounds against the unknown extent
        // and the output dimension is still the static `sizes[axis]`. The static axis 1 still validates `2 <= 3`.
        assert_eq!(
            operation.infer_output_types(&[
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(3)])),
                index_type.clone(),
                index_type.clone(),
            ]),
            Ok(vec![output_type.clone()]),
        );
        // A bounded-dynamic input axis is likewise accepted (the bound does not need to cover the static size; the
        // clamp keeps the read safe at run time).
        assert_eq!(
            operation.infer_output_types(&[
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(Some(2)), Size::Static(3)])),
                index_type.clone(),
                index_type.clone(),
            ]),
            Ok(vec![output_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), ArrayType::scalar(DataType::F64), index_type.clone()]),
            Err(TypeError {
                message: "'dynamic_slice' start index 0 must be a scalar integer but has type f64[]".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(2)])),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "'dynamic_slice' start index 0 must be a scalar integer but has type i32[2]".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), index_type.clone(), ArrayType::scalar(DataType::I64)]),
            Err(TypeError {
                message: "'dynamic_slice' start indices must share one integer type but index 1 has type i64[] and \
                    index 0 has type i32[]"
                    .to_string(),
            }),
        );
        assert_eq!(
            InterpretableOperation::<TestArray, crate::EagerContext<TestArray>>::interpret(
                &operation,
                &crate::EagerContext::<TestArray>::new(),
                &[]
            ),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured sizes.
        let mut builder = ProgramBuilder::<TestArray, DynamicSliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_index_0 = builder.add_input(index_type.clone());
        let program_index_1 = builder.add_input(index_type);
        let program_output =
            builder.add_instruction(operation, vec![program_input, program_index_0, program_index_1]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, TestArray>(
                vec![program_output],
                vec![Placeholder, Placeholder, Placeholder],
                Placeholder,
            )
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
    fn test_dynamic_update_slice() {
        let operation = DynamicUpdateSliceOperation;

        // Operation identity.
        assert_eq!(operation.name(), DYNAMIC_UPDATE_SLICE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "dynamic_update_slice");

        // Type inference validates the update and index operand types and returns the input type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2)]));
        let index_type = ArrayType::scalar(DataType::I32);
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                update_type.clone(),
                index_type.clone(),
                index_type.clone(),
            ]),
            Ok(vec![input_type.clone()]),
        );
        assert_eq!(
            (&input_type).dynamic_update_slice(&update_type, &[index_type.clone(), index_type.clone()]),
            Ok(input_type.clone()),
        );

        // Interpretation overwrites the block at the in-band start indices.
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let update = TestArray::matrix(1, 2, vec![8.0, 9.0]);
        let output = operation
            .interpret(
                &crate::EagerContext::<TestArray>::new(),
                &[input.clone(), update.clone(), index(0.0), index(1.0)],
            )
            .unwrap();
        assert_eq!(*output[0].r#type(), input_type);
        assert_eq!(output[0].values, vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);

        // Out-of-bounds start indices clamp per StableHLO semantics: the effective start index along axis `d` is
        // `clamp(0, start_indices[d], input_dimension[d] - update_dimension[d])`.
        let clamped = operation
            .interpret(
                &crate::EagerContext::<TestArray>::new(),
                &[input.clone(), update.clone(), index(5.0), index(-3.0)],
            )
            .unwrap();
        assert_eq!(clamped[0].values, vec![1.0, 2.0, 3.0, 8.0, 9.0, 6.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[input_type.clone()]),
            Err(TypeError {
                message: "'dynamic_update_slice' expects an input operand and an update operand followed by start \
                    index operands but got 1 inputs"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(2)])),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "'dynamic_update_slice' input data type f64 does not match update data type f32".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError { message: "'dynamic_update_slice' update has rank 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), update_type.clone(), index_type.clone()]),
            Err(TypeError {
                message: "'dynamic_update_slice' expects one start index per input axis (2) but got 1".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(2)])),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "'dynamic_update_slice' does not support dynamic update axis 0 with size *; update shapes \
                    must be static"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(4)])),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "'dynamic_update_slice' update axis 1 has size 4 which exceeds input size 3".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(3)])),
                update_type.clone(),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "'dynamic_update_slice' cannot prove that the update fits along dynamic input axis 0 with \
                    size *"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                update_type.clone(),
                ArrayType::scalar(DataType::F64),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "'dynamic_update_slice' start index 0 must be a scalar integer but has type f64[]".to_string(),
            }),
        );
        assert_eq!(
            InterpretableOperation::<TestArray, crate::EagerContext<TestArray>>::interpret(
                &operation,
                &crate::EagerContext::<TestArray>::new(),
                &[]
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<TestArray, DynamicUpdateSliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_update = builder.add_input(update_type);
        let program_index_0 = builder.add_input(index_type.clone());
        let program_index_1 = builder.add_input(index_type);
        let program_output = builder
            .add_instruction(operation, vec![program_input, program_update, program_index_0, program_index_1])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, TestArray>(
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
    fn test_slicing_test_array_kernels() {
        // Rank-3 slice exercises the row-major odometer across non-contiguous blocks.
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)]));
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = TestArray::new(input_type.clone(), values.clone())
            .slice(&[0, 1, 2], &[2, 3, 4], &[1, 1, 1])
            .unwrap();
        assert_eq!(
            *output.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(output.values, vec![6.0, 7.0, 10.0, 11.0, 18.0, 19.0, 22.0, 23.0]);

        // The matching update-slice writes the block back into place.
        let update = TestArray::new(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)])),
            vec![-6.0, -7.0, -10.0, -11.0, -18.0, -19.0, -22.0, -23.0],
        );
        let updated = TestArray::new(input_type, values).update_slice(&update, &[0, 1, 2]).unwrap();
        assert_eq!(
            updated.values,
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -6.0, -7.0, 8.0, 9.0, -10.0, -11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                -18.0, -19.0, 20.0, 21.0, -22.0, -23.0,
            ],
        );

        // Strided slicing walks the row-major odometer with per-axis steps: rows with stride 2 and columns with
        // stride 3 keep elements at indices (0, 0), (0, 3), (1, 0), and (1, 3) of a 2x3x4 input's last two axes.
        let strided = TestArray::new(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)])),
            (0..24).map(|value| value as f64).collect(),
        )
        .slice(&[0, 0, 0], &[2, 3, 4], &[2, 2, 3])
        .unwrap();
        assert_eq!(
            *strided.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(strided.values, vec![0.0, 3.0, 8.0, 11.0]);

        // The dynamic kernels validate their index operand shapes eagerly.
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            input.dynamic_slice(&[index(0.0), TestArray::vector(vec![1.0, 2.0])], &[1, 2]),
            Err(ProgramError::Type(TypeError {
                message: "'dynamic_slice' start index 1 must be a scalar integer but has type f64[2]".to_string(),
            })),
        );
        assert_eq!(
            input.dynamic_update_slice(&TestArray::matrix(1, 2, vec![8.0, 9.0]), &[index(0.0)]),
            Err(ProgramError::Type(TypeError {
                message: "'dynamic_update_slice' expects one start index per input axis (2) but got 1".to_string(),
            })),
        );
    }

    #[test]
    fn test_slice_and_dynamic_slice_propagate_sharding() {
        use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        // [4, 4] sharded over `x` on axis 0 and unreduced over the manual axis `m`.
        let sharding = Sharding::with_manual_axes(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            ["m"],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
        )
        .unwrap();
        let input = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let start = ArrayType::scalar(DataType::I32);

        // Slicing the `x`-sharded axis to an evenly divisible size keeps the operand sharding (reduction state and
        // the unreduced manual axis included).
        assert_eq!(input.slice(&[0, 0], &[2, 4], &[1, 1]).unwrap().sharding(), Some(&sharding));
        // Slicing it to a size not divisible by the explicit mesh-axis size (2) is rejected.
        assert!(input.slice(&[0, 0], &[3, 4], &[1, 1]).is_err());

        // dynamic_slice carries the same sharding through (ryft diverges from JAX, which leaves an unreduced
        // dynamic_slice operand unimplemented) and applies the same divisibility check.
        assert_eq!(input.dynamic_slice(&[start.clone(), start.clone()], &[2, 4]).unwrap().sharding(), Some(&sharding),);
        assert!(input.dynamic_slice(&[start.clone(), start.clone()], &[3, 4]).is_err());
    }

    #[test]
    fn test_update_slice_requires_matching_sharding() {
        use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let replicated =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap();
        let operand = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(4)]))
            .with_sharding(sharded.clone())
            .unwrap();
        let matching = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(4)]))
            .with_sharding(sharded.clone())
            .unwrap();
        let conflicting = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(4)]))
            .with_sharding(replicated)
            .unwrap();
        let start = ArrayType::scalar(DataType::I32);

        // A matching update keeps the operand's sharding; an update conflicting on an Explicit axis is rejected.
        assert_eq!((&operand).update_slice(&matching, &[0, 0]).unwrap().sharding(), Some(&sharded));
        assert!((&operand).update_slice(&conflicting, &[0, 0]).is_err());

        // dynamic_update_slice applies the same operand-vs-update rule.
        assert_eq!(
            (&operand).dynamic_update_slice(&matching, &[start.clone(), start.clone()]).unwrap().sharding(),
            Some(&sharded),
        );
        assert!((&operand).dynamic_update_slice(&conflicting, &[start.clone(), start.clone()]).is_err());
    }

    #[test]
    fn test_update_slice_unions_update_varying_manual_axes() {
        use std::collections::BTreeSet;

        use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let dimensions = || vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
        // The operand does not vary over the manual axis `m`, but the update does, so the result varies over `m` too.
        let operand = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(4)]))
            .with_sharding(Sharding::new(mesh.clone(), dimensions()).unwrap())
            .unwrap();
        let update = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(4)]))
            .with_sharding(
                Sharding::with_manual_axes(mesh.clone(), dimensions(), Vec::<&str>::new(), Vec::<&str>::new(), ["m"])
                    .unwrap(),
            )
            .unwrap();
        let output = (&operand).update_slice(&update, &[0, 0]).unwrap();
        assert_eq!(output.sharding().unwrap().varying_manual_axes(), &BTreeSet::from(["m".to_string()]));
    }

    /// Minimal operation enum hosting the primal [`DynamicSliceOperation`] and [`DynamicUpdateSliceOperation`] (each is
    /// both a forward operation and the other's staged adjoint) plus the structural `zero` and `add` operations the
    /// transpose pass needs. The `Constant` variant carries the value parameter `V` so the [`Operation`] derive can
    /// infer the primary type.
    #[derive(Clone, Debug, ryft_macros::Operation, ryft_macros::TransposableOperation)]
    enum TestSlicingOperation<V: Value<Type = ArrayType>> {
        Zero(ZeroOperation<ArrayType>),
        Constant(crate::operations::constants::ConstantOperation<V>),
        Add(crate::operations::math::AddOperation),
        DynamicSlice(DynamicSliceOperation),
        DynamicUpdateSlice(DynamicUpdateSliceOperation),
    }

    #[test]
    fn test_dynamic_slice_partitioned_transpose_computes_update_slice_adjoint() {
        // Slice a [1, 2] block at start (1, 1) of a [2, 3] operand: the operand is linear and the scalar start indices
        // are the known operands. The sliced output and its cotangent have shape [1, 2].
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let cotangent = TestArray::matrix(1, 2, vec![5.0, 7.0]);
        let sizes = vec![1, 2];

        // Build `dynamic_slice(operand, start_row, start_col)` over the test enum, treat only the operand as linear,
        // and interpret the pullback on `[cotangent, start_row, start_col]`.
        let mut builder = ProgramBuilder::<TestArray, TestSlicingOperation<TestArray>>::new();
        let operand_input = builder.add_input(operand_type.clone());
        let row_input = builder.add_input(ArrayType::scalar(DataType::I32));
        let col_input = builder.add_input(ArrayType::scalar(DataType::I32));
        let output = builder
            .add_instruction(DynamicSliceOperation::new(sizes.clone()), vec![operand_input, row_input, col_input])
            .unwrap()[0];
        let program = builder
            .build::<(TestArray, TestArray, TestArray), TestArray>(
                vec![output],
                (Placeholder, Placeholder, Placeholder),
                Placeholder,
            )
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(pullback.output_ids().len(), 1, "the known start indices must receive no cotangent output");
        let operand_cotangents = pullback.interpret(vec![cotangent, index(1.0), index(1.0)]).unwrap();
        assert_eq!(operand_cotangents.len(), 1);
        assert_eq!(*operand_cotangents[0].r#type(), operand_type);
        // The dynamic-slice adjoint writes the cotangent block back at start (1, 1) of a [2, 3] zero operand.
        assert_eq!(operand_cotangents[0].values, vec![0.0, 0.0, 0.0, 0.0, 5.0, 7.0]);
    }

    #[test]
    fn test_dynamic_update_slice_partitioned_transpose_computes_operand_adjoints() {
        // Update a [1, 2] block at start (0, 1) of a [2, 3] input: the input and update are linear and the scalar
        // start indices are the known operands. The output and its cotangent have shape [2, 3].
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2)]));
        let cotangent = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Build `dynamic_update_slice(input, update, start_row, start_col)` over the test enum, treat the input and
        // update as linear, and interpret the pullback on `[cotangent, start_row, start_col]`.
        let mut builder = ProgramBuilder::<TestArray, TestSlicingOperation<TestArray>>::new();
        let input_input = builder.add_input(input_type.clone());
        let update_input = builder.add_input(update_type.clone());
        let row_input = builder.add_input(ArrayType::scalar(DataType::I32));
        let col_input = builder.add_input(ArrayType::scalar(DataType::I32));
        let output = builder
            .add_instruction(DynamicUpdateSliceOperation, vec![input_input, update_input, row_input, col_input])
            .unwrap()[0];
        let program = builder
            .build::<(TestArray, TestArray, TestArray, TestArray), TestArray>(
                vec![output],
                (Placeholder, Placeholder, Placeholder, Placeholder),
                Placeholder,
            )
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0, 1]).unwrap();
        assert_eq!(pullback.output_ids().len(), 2, "the known start indices must receive no cotangent output");
        let cotangents = pullback.interpret(vec![cotangent, index(0.0), index(1.0)]).unwrap();
        assert_eq!(cotangents.len(), 2);
        assert_eq!(*cotangents[0].r#type(), input_type);
        assert_eq!(*cotangents[1].r#type(), update_type);
        // Input cotangent: the cotangent with the update window (start (0, 1), shape [1, 2]) zeroed.
        assert_eq!(cotangents[0].values, vec![1.0, 0.0, 0.0, 4.0, 5.0, 6.0]);
        // Update cotangent: the cotangent block at the update window.
        assert_eq!(cotangents[1].values, vec![2.0, 3.0]);
    }
}
