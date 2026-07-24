use std::collections::BTreeSet;
use std::fmt::Display;

use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::reverse::{TransposableOperation, TranspositionDriver};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    ElementwiseDerivativeAlignment,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::constants::{Zero, ZeroLike, ZeroOperation};
use crate::operations::control_flow::scan::render_factor_list;
use crate::operations::manipulation::{Broadcast, PadOperation, Reshape, Transpose};
use crate::operations::sharding::Reshard;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::sharding::{MeshAxisType, Sharding, ShardingDimension};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::custom_derivatives::CustomVjpResidual;
use crate::types::{ArrayType, Memory, Shape, Size};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`SliceOperation`].
pub const SLICE_OPERATION_NAME: &str = "slice";

/// Canonical operation name for [`UpdateSliceOperation`].
pub const UPDATE_SLICE_OPERATION_NAME: &str = "update_slice";

/// Canonical operation name for [`DynamicSliceOperation`].
pub const DYNAMIC_SLICE_OPERATION_NAME: &str = "dynamic_slice";

/// Canonical operation name for [`DynamicUpdateSliceOperation`].
pub const DYNAMIC_UPDATE_SLICE_OPERATION_NAME: &str = "dynamic_update_slice";

/// Validates the scalar integer start-index operand types of a dynamic slicing operation. Each index type must be a
/// rank-0 integer type, all indices must share one integer type, and every index must reside in `operand_memory`. The
/// `operation_name` parameter selects the reported operation name because this helper serves both
/// [`DynamicSliceOperation`] and [`DynamicUpdateSliceOperation`].
fn validate_start_index_types(
    operation_name: &'static str,
    operand_memory: Memory,
    index_types: &[ArrayType],
) -> Result<(), ProgramError> {
    for (index, index_type) in index_types.iter().enumerate() {
        if index_type.rank() != 0 || !index_type.data_type().is_integer() {
            return Err(TypeError::Invalid(format!(
                "'{operation_name}' start index {index} must be a scalar integer but has type {index_type}",
            ))
            .into());
        }
        if index_type.memory() != operand_memory {
            return Err(TypeError::Invalid(format!(
                "'{operation_name}' operand and start indices must share one memory space but start index {index} \
                     resides in {} and the operand resides in {operand_memory}",
                index_type.memory(),
            ))
            .into());
        }
        if index_type.data_type() != index_types[0].data_type() {
            return Err(TypeError::Invalid(format!(
                "'{operation_name}' start indices must share one integer type but index {index} has type \
                    {index_type} and index 0 has type {first}",
                first = index_types[0],
            ))
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

    /// Axes whose slice limit is the input's runtime extent.
    full_extent_axes: BTreeSet<usize>,
}

impl SliceOperation {
    /// Creates a new [`SliceOperation`] with the provided start and limit indices and unit strides. Use
    /// [`with_strides`](Self::with_strides) to attach non-unit strides.
    #[inline]
    pub fn new(start_indices: Vec<usize>, limit_indices: Vec<usize>) -> Self {
        let strides = vec![1; start_indices.len()];
        Self { start_indices, limit_indices, strides, full_extent_axes: BTreeSet::new() }
    }

    /// Replaces the strides of this [`SliceOperation`] with `strides`. There must be one stride per start index and
    /// every stride must be at least `1`.
    pub fn with_strides(mut self, strides: Vec<usize>) -> Result<Self, ProgramError> {
        if strides.len() != self.start_indices.len() {
            return Err(TypeError::Invalid(format!(
                "'slice' strides has length {} but start_indices has length {}",
                strides.len(),
                self.start_indices.len(),
            ))
            .into());
        }
        if let Some(axis) = strides.iter().position(|stride| *stride == 0) {
            return Err(
                TypeError::Invalid(format!("'slice' strides must be at least 1 but axis {axis} has stride 0")).into()
            );
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

    /// Uses the input's runtime extent as the limit for each axis in `axes`.
    ///
    /// This internal form represents a slice from zero through a dynamic dimension's runtime extent. Every selected
    /// axis must have start index `0`; its stored static limit is ignored and its stride is applied to the runtime
    /// extent.
    pub(crate) fn with_full_extent_axes(mut self, axes: impl IntoIterator<Item = usize>) -> Result<Self, ProgramError> {
        for axis in axes {
            if axis >= self.start_indices.len() {
                return Err(TypeError::Invalid(format!(
                    "'slice' full-extent axis {axis} is out of bounds for {} slice axes",
                    self.start_indices.len(),
                ))
                .into());
            }
            if self.start_indices[axis] != 0 {
                return Err(
                    TypeError::Invalid(format!("'slice' full-extent axis {axis} must have start index 0")).into()
                );
            }
            self.full_extent_axes.insert(axis);
        }
        Ok(self)
    }

    /// Returns the axes whose slice limit is the input's runtime extent.
    #[inline]
    pub fn full_extent_axes(&self) -> &BTreeSet<usize> {
        &self.full_extent_axes
    }

    /// Resolves the effective limit indices for a concrete input value.
    fn resolved_limit_indices<T: Typed<Type = ArrayType>>(&self, input: &T) -> Result<Vec<usize>, ProgramError> {
        let input_type = input.r#type();
        let mut limit_indices = self.limit_indices.clone();
        for &axis in &self.full_extent_axes {
            let dimension = input_type.dimension(axis);
            limit_indices[axis] = dimension.value().ok_or_else(|| {
                TypeError::Invalid(format!(
                    "'slice' cannot resolve the runtime extent of dynamic input axis {axis} while interpreting a value"
                ))
            })?;
        }
        Ok(limit_indices)
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

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let result = if self.full_extent_axes.is_empty() {
            input_types[0].slice(self.start_indices.as_slice(), self.limit_indices.as_slice(), self.strides.as_slice())
        } else {
            let input_type = &input_types[0];
            if self.start_indices.len() != input_type.rank()
                || self.limit_indices.len() != input_type.rank()
                || self.strides.len() != input_type.rank()
            {
                input_type.slice(self.start_indices.as_slice(), self.limit_indices.as_slice(), self.strides.as_slice())
            } else {
                let mut output_dimensions = Vec::with_capacity(input_type.rank());
                for (axis, ((&start, &limit), &stride)) in
                    self.start_indices.iter().zip(self.limit_indices.iter()).zip(self.strides.iter()).enumerate()
                {
                    if self.full_extent_axes.contains(&axis) {
                        if start != 0 {
                            return Err(TypeError::Invalid(format!(
                                "'slice' full-extent axis {axis} must have start index 0"
                            )));
                        }
                        if stride == 0 {
                            return Err(TypeError::Invalid(format!(
                                "'slice' strides must be at least 1 but axis {axis} has stride 0"
                            )));
                        }
                        output_dimensions.push(match input_type.dimension(axis) {
                            Size::Static(size) => Size::Static(size.div_ceil(stride)),
                            Size::Dynamic(None) => Size::Dynamic(None),
                            Size::Dynamic(Some(0)) => Size::Dynamic(Some(0)),
                            Size::Dynamic(Some(upper_bound)) => {
                                Size::Dynamic((upper_bound - 1).div_ceil(stride).checked_add(1))
                            }
                        });
                        continue;
                    }
                    let dimension = input_type.dimension(axis);
                    let Size::Static(size) = dimension else {
                        return Err(TypeError::Invalid(format!(
                            "'slice' does not support dynamic input axis {axis} with size {dimension}; slice \
                                 bounds cannot be validated against an unknown extent",
                        )));
                    };
                    if stride == 0 {
                        return Err(TypeError::Invalid(format!(
                            "'slice' strides must be at least 1 but axis {axis} has stride 0"
                        )));
                    }
                    if start > limit {
                        return Err(TypeError::Invalid(format!(
                            "'slice' start index {start} is greater than limit index {limit} at axis {axis}"
                        )));
                    }
                    if limit > size {
                        return Err(TypeError::Invalid(format!(
                            "'slice' limit index {limit} is out of bounds for axis {axis} with size {size}"
                        )));
                    }
                    output_dimensions.push(Size::Static((limit - start).div_ceil(stride)));
                }
                let sharding = resized_output_sharding(input_type, &output_dimensions, SLICE_OPERATION_NAME)?;
                ArrayType::new(input_type.data_type(), Shape::new(output_dimensions))
                    .with_memory(input_type.memory())
                    .with_sharding(sharding)
                    .map_err(|error| TypeError::Invalid(error.to_string()).into())
            }
        };
        match result {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::Invalid(error.to_string())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("start_indices", format_args!("{:?}", self.start_indices))?;
            operation.field("limit_indices", format_args!("{:?}", self.limit_indices))?;
            if self.strides.iter().any(|stride| *stride != 1) {
                operation.field("strides", format_args!("{:?}", self.strides))?;
            }
            if !self.full_extent_axes.is_empty() {
                operation.field("full_extent_axes", format_args!("{:?}", self.full_extent_axes))?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Slice>> InterpretableOperation<C> for SliceOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let limit_indices = self.resolved_limit_indices(&inputs[0])?;
        Ok(vec![inputs[0].clone().slice(
            self.start_indices.as_slice(),
            limit_indices.as_slice(),
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

/// Forward-mode rule for [`SliceOperation`]: slicing is a linear map, so the primal output is the slice of the
/// operand primal and the tangent is the same slice of the operand tangent. A zero operand tangent yields a typed zero
/// output tangent.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for SliceOperation
where
    C::Operation: From<SliceOperation>,
    C::Value: Slice,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let apply = |value: &C::Value| -> Result<C::Value, ProgramError> {
            if self.full_extent_axes().is_empty() {
                return value.slice(self.start_indices(), self.limit_indices(), self.strides());
            }
            let outputs = context.bind(self.clone(), Vec::new(), std::slice::from_ref(value))?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(outputs.into_iter().next().unwrap())
        };
        let primal = apply(inputs[0].primal())?;
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
            MaybeZero::Value(tangent) => MaybeZero::Value(apply(tangent)?),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Transpose (vector-Jacobian product) for a [`SliceOperation`].
///
/// The forward map extracts a (possibly strided) block, so its pullback scatters the output cotangent back into the
/// positions the forward map read, with the strategy split on the strides:
///
///   - **Unit strides** read a contiguous block, so the pullback writes the cotangent into a zero array of the input
///     type at the same static offsets: `cotangent ↦ update_slice(zeros(input_type), cotangent, start_indices)`.
///   - **Non-unit strides** read every `strides[d]`-th element, so the pullback pads the cotangent with a zero
///     scalar at exactly the inverse geometry: `edge_padding_low[d] = start_indices[d]`,
///     `interior_padding[d] = strides[d] - 1`, and `edge_padding_high[d]` covers the rest of the input extent
///     (everything after the last element the forward slice covered). For example, slicing `[0..6)` with `start = 1`
///     and `stride = 2` reads positions `1`, `3`, and `5`, and the pullback pads the cotangent of length `3` with
///     `low = 1`, `interior = 1`, and `high = 0`, scattering its elements back to positions `1`, `3`, and `5` of a
///     zero-filled length-`6` array.
///
/// Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for SliceOperation
where
    O: Operation<ArrayType> + From<UpdateSliceOperation> + From<PadOperation> + From<ZeroOperation<ArrayType>>,
    Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
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
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]),
            MaybeZero::Value(cotangent) if self.strides().iter().all(|stride| *stride == 1) => {
                let zeros = MaybeZero::Zero(inputs[0].r#type().cotangent()).materialize(context)?;
                let outputs = context.stage_operation(
                    UpdateSliceOperation::new(self.start_indices().to_vec()),
                    Vec::new(),
                    &[zeros, cotangent.clone()],
                )?;
                check_count!("output", outputs, 1, ProgramError);
                let cotangent =
                    outputs.into_iter().next().unwrap().unalign_cotangent(&inputs[0].r#type().cotangent())?;
                Ok(vec![MaybeZero::Value(cotangent)])
            }
            MaybeZero::Value(cotangent) => {
                let input_type = inputs[0].r#type();
                let mut edge_padding_low = Vec::with_capacity(input_type.rank());
                let mut edge_padding_high = Vec::with_capacity(input_type.rank());
                let mut interior_padding = Vec::with_capacity(input_type.rank());
                for (axis, ((&start, &limit), &stride)) in
                    self.start_indices().iter().zip(self.limit_indices()).zip(self.strides()).enumerate()
                {
                    let dimension = input_type.dimension(axis);
                    let Some(input_size) = dimension.value() else {
                        return Err(TypeError::Invalid(format!(
                            "'slice' transpose requires a static input shape but axis {axis} has size {dimension}",
                        ))
                        .into());
                    };
                    let output_size = (limit - start).div_ceil(stride);
                    // The forward slice covered positions `start + i * stride` for `i < output_size`; everything
                    // after the last covered position becomes high edge padding. An empty slice covered nothing, so
                    // the pullback is pure edge padding around zero interior elements.
                    let high = match output_size {
                        0 => input_size - start,
                        size => input_size - (start + (size - 1) * stride) - 1,
                    };
                    edge_padding_low.push(i64::try_from(start).map_err(|_| {
                        TypeError::Invalid(format!("'slice' transpose start index is too large on axis {axis}"))
                    })?);
                    edge_padding_high.push(i64::try_from(high).map_err(|_| {
                        TypeError::Invalid(format!("'slice' transpose high padding is too large on axis {axis}"))
                    })?);
                    interior_padding.push(stride - 1);
                }
                let zero = MaybeZero::Zero(
                    ArrayType::scalar(input_type.data_type().cotangent()).with_memory(input_type.memory()),
                )
                .materialize(context)?;
                let outputs = context.stage_operation(
                    PadOperation::new(edge_padding_low, edge_padding_high, interior_padding)?,
                    Vec::new(),
                    &[cotangent.clone(), zero],
                )?;
                check_count!("output", outputs, 1, ProgramError);
                let cotangent =
                    outputs.into_iter().next().unwrap().unalign_cotangent(&inputs[0].r#type().cotangent())?;
                Ok(vec![MaybeZero::Value(cotangent)])
            }
        }
    }
}

/// Batching rule for [`SliceOperation`]: a batched operand keeps its batch axis by slicing it fully, so the lifted
/// operation inserts start index `0`, limit `axis_size`, and stride `1` at the batch axis position.
impl<C: Context<Type = ArrayType>> BatchableOperation<C> for SliceOperation
where
    SliceOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        match inputs[0].batch_axis_position() {
            None => self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]),
            Some(batch_axis) => {
                let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
                let mut start_indices = self.start_indices().to_vec();
                start_indices.insert(batch_axis, 0);
                let mut limit_indices = self.limit_indices().to_vec();
                limit_indices.insert(batch_axis, axis_size);
                let mut strides = self.strides().to_vec();
                strides.insert(batch_axis, 1);
                let full_extent_axes = self
                    .full_extent_axes()
                    .iter()
                    .map(|axis| if *axis >= batch_axis { *axis + 1 } else { *axis })
                    .collect::<Vec<_>>();
                let lifted = SliceOperation::new(start_indices, limit_indices)
                    .with_strides(strides)?
                    .with_full_extent_axes(full_extent_axes)?;
                lifted.interpret_with_batch_axes(context, inputs, &[BatchAxis::from_position(batch_axis)])
            }
        }
    }
}

/// Represents the ability to extract a (possibly strided) sub-array using static start, limit, and stride values.
/// Its semantics follow StableHLO's [`slice`](https://openxla.org/stablehlo/spec#slice) operation.
///
/// `t.slice(start_indices, limit_indices, strides)` returns the sub-array whose element at index `i` is the input
/// element at index `start_indices + i * strides`, with output dimension
/// `ceil((limit_indices[d] - start_indices[d]) / strides[d])` along each axis `d` (an axis with
/// `start_indices[d] == limit_indices[d]` is empty). All three slices must have length equal to the input rank, and
/// each axis must satisfy `start_indices[d] <= limit_indices[d] <= input_dimension[d]` and `strides[d] >= 1`. Slicing
/// requires static input extents: inputs with dynamic dimensions are rejected because the bounds cannot be validated
/// against an unknown extent. A slice covering the complete input with unit strides passes it through unchanged. Any
/// other output preserves the input memory space and clears explicit physical layout metadata.
///
/// # Example
///
/// The following example shows how to use [`Slice`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::Slice;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::arrays::Array;
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Slice the middle 1x2 block out of a 2x3 matrix.
/// let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let y = x.slice(&[1, 1], &[2, 3], &[1, 1])?;
/// // `y` has shape [1, 2] with values [[5.0, 6.0]].
/// assert_eq!(y.to_f64s(), vec![5.0, 6.0]);
///
/// // A non-unit stride keeps every other element, like `x[0:6:2]` in NumPy.
/// let x = Array::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
/// let y = x.slice(&[1], &[6], &[2])?;
/// assert_eq!(y.to_f64s(), vec![1.0, 3.0, 5.0]);
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
/// [`pad`](super::padding)). The operand sharding is carried through unchanged: resizing dimensions in place neither
/// changes the per-dimension placement relative to the array axes nor the pending cross-device reduction state.
/// Selecting or padding elements commutes with a pending sum, so a value unreduced/reduced over a mesh axis stays so.
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
            return Err(TypeError::Invalid(format!(
                "'{op}' on a dimension sharded over explicit mesh axes requires the output size ({output_size}) at \
                     axis {axis} to be divisible by the mesh-axis product ({explicit_axis_product})"
            )));
        }
    }
    Ok(Some(sharding.clone()))
}

/// Returns the output [`Sharding`] for an in-place update ([`UpdateSlice`] / [`DynamicUpdateSlice`]). Because the
/// update is written into the operand without resharding, the two must agree on placement and reduction state wherever an
/// [`Explicit`](MeshAxisType::Explicit) mesh axis is involved; differences confined to `Manual`/`Auto` axes are
/// tolerated (left to `shard_map` / the compiler). The output keeps the operand's sharding, except that the update's
/// [`varying_manual_axes`](Sharding::varying_manual_axes) are unioned in — the written region may vary over manual
/// axes the operand does not, so the result does too. An operand without a sharding leaves the output unsharded.
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
        return Err(TypeError::Invalid(format!("'{op}' operand and update must use the same mesh")));
    }
    if operand_sharding.conflicts_on_explicit_axes_with(update_sharding) {
        return Err(TypeError::Invalid(format!(
            "'{op}' operand and update must be sharded identically, but got {operand_sharding} and {update_sharding}"
        )));
    }
    if update_sharding.varying_manual_axes().is_subset(operand_sharding.varying_manual_axes()) {
        return Ok(Some(operand_sharding.clone()));
    }
    let varying_manual_axes = operand_sharding
        .varying_manual_axes()
        .union(update_sharding.varying_manual_axes())
        .cloned()
        .collect::<Vec<_>>();
    operand_sharding
        .clone()
        .with_varying_manual_axes(varying_manual_axes)
        .map(Some)
        .map_err(|error| TypeError::Invalid(error.to_string()))
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
            return Err(TypeError::Invalid(format!(
                "'slice' start_indices has length {} but input has rank {rank}",
                start_indices.len(),
            ))
            .into());
        }
        if limit_indices.len() != rank {
            return Err(TypeError::Invalid(format!(
                "'slice' limit_indices has length {} but input has rank {rank}",
                limit_indices.len(),
            ))
            .into());
        }
        if strides.len() != rank {
            return Err(TypeError::Invalid(format!(
                "'slice' strides has length {} but input has rank {rank}",
                strides.len()
            ))
            .into());
        }
        let mut output_dimensions = Vec::with_capacity(rank);
        for (axis, ((&start, &limit), &stride)) in
            start_indices.iter().zip(limit_indices.iter()).zip(strides.iter()).enumerate()
        {
            let dimension = self.dimension(axis);
            let Size::Static(size) = dimension else {
                return Err(TypeError::Invalid(format!(
                    "'slice' does not support dynamic input axis {axis} with size {dimension}; slice bounds \
                        cannot be validated against an unknown extent",
                ))
                .into());
            };
            if stride == 0 {
                return Err(TypeError::Invalid(format!(
                    "'slice' strides must be at least 1 but axis {axis} has stride 0"
                ))
                .into());
            }
            if start > limit {
                return Err(TypeError::Invalid(format!(
                    "'slice' start index {start} is greater than limit index {limit} at axis {axis}"
                ))
                .into());
            }
            if limit > size {
                return Err(TypeError::Invalid(format!(
                    "'slice' limit index {limit} is out of bounds for axis {axis} with size {size}"
                ))
                .into());
            }
            output_dimensions.push(Size::Static((limit - start).div_ceil(stride)));
        }
        if output_dimensions.as_slice() == self.shape().dimensions() {
            return Ok(self.clone());
        }
        let sharding = resized_output_sharding(self, &output_dimensions, SLICE_OPERATION_NAME)?;
        ArrayType::new(self.data_type(), Shape::new(output_dimensions))
            .with_memory(self.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError::Invalid(error.to_string()).into())
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
        let output_type = self.r#type().slice(start_indices, limit_indices, strides)?;
        if output_type.eq(self.r#type().as_ref()) {
            return Ok(self.clone());
        }
        let operation =
            SliceOperation::new(start_indices.to_vec(), limit_indices.to_vec()).with_strides(strides.to_vec())?;
        Ok(self.dispatch_domain().bind(operation, Vec::new(), std::slice::from_ref(self))?.remove(0))
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

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        match input_types[0].update_slice(&input_types[1], self.start_indices.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::Invalid(error.to_string())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("start_indices", format_args!("{:?}", self.start_indices)))
    }
}

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

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for UpdateSliceOperation where
    C::Operation: From<UpdateSliceOperation>
{
}

/// Forward-mode rule for [`UpdateSliceOperation`]: the operation is jointly linear in its operand and update, so
/// the tangent updates the operand tangent with the update tangent at the same static start indices. A zero operand and
/// update tangent yields a typed zero output tangent.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for UpdateSliceOperation
where
    C::Operation: From<UpdateSliceOperation>,
    C::Value: UpdateSlice,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let operand = &inputs[0];
        let update = &inputs[1];
        let primal = operand.primal().update_slice(update.primal(), self.start_indices())?;
        let tangent = if operand.tangent().is_zero() && update.tangent().is_zero() {
            MaybeZero::Zero(primal.r#type().tangent())
        } else {
            let operand_tangent = operand.tangent().clone().materialize(context)?;
            let update_tangent = update.tangent().clone().materialize(context)?;
            MaybeZero::Value(operand_tangent.update_slice(&update_tangent, self.start_indices())?)
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Transpose (vector-Jacobian product) for an [`UpdateSliceOperation`].
///
/// The forward map overwrites a block of the input with the update, so its pullback splits the output cotangent into
/// two contributions: the input cotangent is the cotangent with the update window zeroed
/// (`update_slice(cotangent, zeros(update_type), start_indices)`) and the update cotangent is the static slice of
/// the cotangent at the update window (`slice(cotangent, start_indices, start_indices + update_shape)`).
/// Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for UpdateSliceOperation
where
    O: Operation<ArrayType> + From<SliceOperation> + From<UpdateSliceOperation> + From<ZeroOperation<ArrayType>>,
    Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
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
                MaybeZero::Zero(inputs[0].r#type().cotangent()),
                MaybeZero::Zero(inputs[1].r#type().cotangent()),
            ]),
            MaybeZero::Value(cotangent) => {
                let update_type = inputs[1].r#type();
                let update_sizes = static_update_sizes(UPDATE_SLICE_TRANSPOSE_CONTEXT, &update_type)?;
                let zeros = MaybeZero::Zero(update_type.cotangent()).materialize(context)?;
                let input_cotangents = context.stage_operation(
                    UpdateSliceOperation::new(self.start_indices().to_vec()),
                    Vec::new(),
                    &[cotangent.clone(), zeros],
                )?;
                check_count!("output", input_cotangents, 1, ProgramError);
                let limit_indices: Vec<usize> =
                    self.start_indices().iter().zip(update_sizes.iter()).map(|(start, size)| start + size).collect();
                let update_cotangents = context.stage_operation(
                    SliceOperation::new(self.start_indices().to_vec(), limit_indices)
                        .with_strides(vec![1; self.start_indices().len()])?,
                    Vec::new(),
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", update_cotangents, 1, ProgramError);
                let update_cotangent =
                    update_cotangents.into_iter().next().unwrap().unalign_cotangent(&inputs[1].r#type().cotangent())?;
                Ok(vec![
                    MaybeZero::Value(input_cotangents.into_iter().next().unwrap()),
                    MaybeZero::Value(update_cotangent),
                ])
            }
        }
    }
}

/// Batching rule for [`UpdateSliceOperation`]: the input and update operands are aligned on one physical batch axis
/// (replicated operands are broadcast to gain it), and the lifted operation inserts start index `0` at that axis
/// so each batch item updates its own block.
impl<C: Context<Type = ArrayType>> BatchableOperation<C> for UpdateSliceOperation
where
    C::Value: Broadcast + Transpose,
    UpdateSliceOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let Some(batch_axis) = inputs.iter().find_map(ArrayBatch::batch_axis_position) else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
        let input = inputs[0].match_axis(batch_axis, axis_size, context.axis_sharding().clone())?;
        let update = inputs[1].match_axis(batch_axis, axis_size, context.axis_sharding().clone())?;
        let mut start_indices = self.start_indices().to_vec();
        start_indices.insert(batch_axis, 0);
        UpdateSliceOperation::new(start_indices).interpret_with_batch_axes(
            context,
            &[input, update],
            &[BatchAxis::from_position(batch_axis)],
        )
    }
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
/// are validated when the operation is constructed, so no index clamping occurs. The input and update must reside in
/// the same memory space, and the output retains the complete input type, including layout and memory placement.
///
/// # Example
///
/// The following example shows how to use [`UpdateSlice`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::UpdateSlice;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::arrays::Array;
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Overwrite the last two elements of the first row of a 2x3 matrix.
/// let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let update = Array::matrix(1, 2, vec![8.0, 9.0]);
/// let y = x.update_slice(&update, &[0, 1])?;
/// assert_eq!(y.to_f64s(), vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);
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
            return Err(TypeError::Invalid(format!(
                "'update_slice' input data type {} does not match update data type {}",
                self.data_type(),
                update.data_type(),
            ))
            .into());
        }
        if self.memory() != update.memory() {
            return Err(TypeError::Invalid(format!(
                "'update_slice' input and update must share one memory space but reside in {} and {}",
                self.memory(),
                update.memory(),
            ))
            .into());
        }
        let rank = self.rank();
        if update.rank() != rank {
            return Err(TypeError::Invalid(format!(
                "'update_slice' update has rank {} but input has rank {rank}",
                update.rank()
            ))
            .into());
        }
        if start_indices.len() != rank {
            return Err(TypeError::Invalid(format!(
                "'update_slice' start_indices has length {} but input has rank {rank}",
                start_indices.len(),
            ))
            .into());
        }
        for (axis, &start) in start_indices.iter().enumerate() {
            let update_dimension = update.dimension(axis);
            let Size::Static(update_size) = update_dimension else {
                return Err(TypeError::Invalid(format!(
                    "'update_slice' does not support dynamic update axis {axis} with size {update_dimension}; \
                        update shapes must be static",
                ))
                .into());
            };
            let input_dimension = self.dimension(axis);
            let Size::Static(input_size) = input_dimension else {
                return Err(TypeError::Invalid(format!(
                    "'update_slice' cannot prove that the update fits along dynamic input axis {axis} with \
                        size {input_dimension}",
                ))
                .into());
            };
            if start.checked_add(update_size).is_none_or(|limit| limit > input_size) {
                return Err(TypeError::Invalid(format!(
                    "'update_slice' update axis {axis} with start index {start} and size {update_size} does not \
                        fit in input size {input_size}",
                ))
                .into());
            }
        }
        // The output is distributed like the input operand (the update is written in place); the operand's placement
        // and reduction state carry through, with the update's varying-manual axes folded in.
        let sharding = update_slice_output_sharding(self, update, UPDATE_SLICE_OPERATION_NAME)?;
        self.clone().with_sharding(sharding).map_err(|error| TypeError::Invalid(error.to_string()).into())
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
            Vec::new(),
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

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.is_empty() {
            return Err(TypeError::Invalid(
                "'dynamic_slice' expects an input operand followed by its start index operands but got no \
                    inputs"
                    .to_string(),
            ));
        }
        match input_types[0].dynamic_slice(&input_types[1..], self.sizes.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::Invalid(error.to_string())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("sizes", format_args!("{:?}", self.sizes)))
    }
}

impl<C: Domain<Type = ArrayType, Value: DynamicSlice>> InterpretableOperation<C> for DynamicSliceOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
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

/// Forward-mode rule for [`DynamicSliceOperation`]: `dynamic_slice` is linear in the operand, and the scalar
/// start indices are non-differentiated primal operand edges, so the tangent slices the operand tangent at the same
/// primal start indices. A zero operand tangent yields a typed zero output tangent.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for DynamicSliceOperation
where
    C::Operation: From<DynamicSliceOperation>,
    C::Value: DynamicSlice,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let (operand, start_indices) =
            inputs.split_first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
        let primal_starts = start_indices.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        let primal = operand.primal().dynamic_slice(&primal_starts, self.sizes())?;
        let tangent = match operand.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.dynamic_slice(&primal_starts, self.sizes())?),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Batching rule for [`DynamicSliceOperation`].
///
/// Replicated start indices keep the structural fast path: a batched operand keeps its batch axis by slicing it
/// fully, so the lifted operation inserts size `axis_size` at the batch axis position and a zero start index for it,
/// derived from an existing index operand via [`ZeroLike`] so the inserted index carries the same scalar integer
/// type. Rank-0 operands have no index operands to donate a zero index, but a rank-0 dynamic slice is the identity
/// map, so the batched operand passes through unchanged.
///
/// Batch-varying (batched) start indices cannot ride along structurally — every batch item needs its own slice origin
/// while the lifted operation reads one origin for all batch items — so the rule falls back to per-item expansion via
/// `batch_by_item_expansion`: each batch item's operand (when batched; a replicated operand is used whole) and start
/// indices are extracted, sliced dynamically per item, and restacked along a fresh leading batch axis (the result's
/// batch axis is `0` even when the operand carried its batch axis elsewhere). The expansion stages `O(batch_size)`
/// operations — a gather-based rule is an explicit non-goal — and behaves identically in eager and tracing contexts
/// because it only goes through the value capability traits.
impl<C> BatchableOperation<C> for DynamicSliceOperation
where
    C: Context<Type = ArrayType> + Zero<C::Value>,
    C::Value: ZeroLike + Broadcast + Transpose + Slice + UpdateSlice + Reshape + Reshard,
    DynamicSliceOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        if inputs.is_empty() {
            return Err(ProgramError::InvalidInputCount { expected: 1 + self.sizes().len(), actual: 0 }.into());
        }
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();
        let axis_size = ArrayBatch::common_batch_size(inputs)?;
        if batch_axes[1..].iter().any(Option::is_some) {
            return batch_by_item_expansion(
                context,
                crate::operations::manipulation::DYNAMIC_SLICE_OPERATION_NAME,
                self,
                inputs,
                axis_size.expect("a mapped input pins the batch size"),
            );
        }
        let Some(batch_axis) = batch_axes[0] else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        if self.sizes().is_empty() {
            return Ok(vec![inputs[0].clone()]);
        }
        let axis_size = axis_size.expect("a mapped input pins the batch size");
        let mut sizes = self.sizes().to_vec();
        sizes.insert(batch_axis, axis_size);
        let zero_index = ArrayBatch::replicated(inputs[1].value().clone().zero_like());
        let mut lifted_inputs = inputs.to_vec();
        lifted_inputs.insert(1 + batch_axis, zero_index);
        DynamicSliceOperation::new(sizes).interpret_with_batch_axes(
            context,
            lifted_inputs.as_slice(),
            &[BatchAxis::from_position(batch_axis)],
        )
    }
}

/// Represents the ability to extract a statically shaped sub-array at start indices that are computed at run time,
/// with the semantics of StableHLO's [`dynamic_slice`](https://openxla.org/stablehlo/spec#dynamic_slice) operation.
///
/// `t.dynamic_slice(start_indices, sizes)` extracts the block of shape `sizes` whose origin is given by the scalar
/// integer values in `start_indices` (one per input axis). Start indices are clamped per StableHLO semantics so the
/// extracted block always lies in bounds: the effective start index along axis `d` is
/// `clamp(0, start_indices[d], input_dimension[d] - sizes[d])`. The output shape is exactly `sizes` and is therefore
/// fully static even though the slice origin is not. Each static input axis must satisfy
/// `sizes[d] <= input_dimension[d]`. A [`Size::Dynamic`] input axis is accepted: the clamp keeps the read in bounds
/// against any runtime extent satisfying the operation's `sizes[d] <= input_dimension[d]` precondition, and the output
/// dimension is still the static `sizes[d]`. A finite dynamic upper bound is rejected when it proves that no admissible
/// runtime extent could satisfy that precondition. This is what lets a dynamically-sized stack (such as the residual
/// stacks of an unbounded-loop pullback) be read iteration by iteration. The operand and start indices must reside in
/// the same memory space. A slice whose sizes equal the input shape passes it through unchanged because every clamped
/// origin is necessarily zero. Any other output preserves the input memory space and clears explicit physical layout
/// metadata.
///
/// # Example
///
/// The following example shows how to use [`DynamicSlice`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::DynamicSlice;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::arrays::Array;
/// # use ryft_core::types::{ArrayType, DataType};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Extract a 1x2 block starting at row 1, column 1 of a 2x3 matrix.
/// let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let i = Array::from_f64s(ArrayType::scalar(DataType::I32), vec![1.0]);
/// let j = Array::from_f64s(ArrayType::scalar(DataType::I32), vec![1.0]);
/// let y = x.dynamic_slice(&[i, j], &[1, 2])?;
/// // `y` has shape [1, 2] with values [[5.0, 6.0]].
/// assert_eq!(y.to_f64s(), vec![5.0, 6.0]);
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
            return Err(TypeError::Invalid(format!(
                "'dynamic_slice' expects one start index per input axis ({rank}) but got {}",
                start_indices.len(),
            ))
            .into());
        }
        if sizes.len() != rank {
            return Err(TypeError::Invalid(format!(
                "'dynamic_slice' sizes has length {} but input has rank {rank}",
                sizes.len()
            ))
            .into());
        }
        validate_start_index_types(DYNAMIC_SLICE_OPERATION_NAME, self.memory(), start_indices)?;
        for (axis, &size) in sizes.iter().enumerate() {
            // A dynamic input axis is accepted: StableHLO clamps the start index into
            // `[0, input_dimension - size]`, so the read always stays in bounds and the output shape is the static
            // `sizes` regardless of the unknown extent. A static input axis still validates the bound eagerly.
            match self.dimension(axis) {
                Size::Static(input_size) if size > input_size => {
                    return Err(TypeError::Invalid(format!(
                        "'dynamic_slice' size {size} is out of bounds for axis {axis} with size {input_size}",
                    ))
                    .into());
                }
                Size::Dynamic(Some(upper_bound)) if size >= upper_bound => {
                    return Err(TypeError::Invalid(format!(
                        "'dynamic_slice' size {size} cannot fit axis {axis} with exclusive upper bound \
                             {upper_bound}",
                    ))
                    .into());
                }
                _ => {}
            }
        }
        let output_dimensions: Vec<Size> = sizes.iter().map(|size| Size::Static(*size)).collect();
        if output_dimensions.as_slice() == self.shape().dimensions() {
            return Ok(self.clone());
        }
        let sharding = resized_output_sharding(self, &output_dimensions, DYNAMIC_SLICE_OPERATION_NAME)?;
        ArrayType::new(self.data_type(), Shape::new(output_dimensions))
            .with_memory(self.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError::Invalid(error.to_string()).into())
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
        let start_index_types = start_indices.iter().map(|index| index.r#type().into_owned()).collect::<Vec<_>>();
        let output_type = self.r#type().dynamic_slice(start_index_types.as_slice(), sizes)?;
        if output_type.eq(self.r#type().as_ref()) {
            return Ok(self.clone());
        }
        let mut inputs = Vec::with_capacity(1 + start_indices.len());
        inputs.push(self.clone());
        inputs.extend(start_indices.iter().cloned());
        Ok(self
            .dispatch_domain()
            .bind(DynamicSliceOperation::new(sizes.to_vec()), Vec::new(), &inputs)?
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

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() < 2 {
            return Err(TypeError::Invalid(format!(
                "'dynamic_update_slice' expects an input operand and an update operand followed by start index \
                    operands but got {} inputs",
                input_types.len(),
            )));
        }
        match input_types[0].dynamic_update_slice(&input_types[1], &input_types[2..]) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::Invalid(error.to_string())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name()).map(|_| ())
    }
}

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
        Ok(vec![input.dynamic_update_slice(update, start_indices)?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DynamicUpdateSliceOperation where
    C::Operation: From<DynamicUpdateSliceOperation>
{
}

/// Forward-mode rule for [`DynamicUpdateSliceOperation`]: `dynamic_update_slice` is jointly linear in the operand
/// and the update, while the scalar start indices are non-differentiated primal operand edges, so the tangent updates
/// the operand tangent with the update tangent at the same primal start indices. A zero operand and update tangent
/// yields a typed zero output tangent.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for DynamicUpdateSliceOperation
where
    C::Operation: From<DynamicUpdateSliceOperation>,
    C::Value: DynamicUpdateSlice,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }
        let operand = &inputs[0];
        let update = &inputs[1];
        let primal_starts = inputs[2..].iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        let primal = operand.primal().dynamic_update_slice(update.primal(), &primal_starts)?;
        let tangent = if operand.tangent().is_zero() && update.tangent().is_zero() {
            MaybeZero::Zero(primal.r#type().tangent())
        } else {
            let operand_tangent = operand.tangent().clone().materialize(context)?;
            let update_tangent = update.tangent().clone().materialize(context)?;
            MaybeZero::Value(operand_tangent.dynamic_update_slice(&update_tangent, &primal_starts)?)
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Batching rule for [`DynamicUpdateSliceOperation`].
///
/// Replicated start indices keep the structural fast path: the input and update operands are aligned on one
/// physical batch axis (replicated operands are broadcast to gain it), and the lifted operation inserts a zero
/// start index for that axis, derived from an existing index operand via [`ZeroLike`] so the inserted index carries
/// the same scalar integer type. Rank-0 operands have no index operands to donate a zero index, but a rank-0 dynamic
/// update-slice replaces the operand with the update entirely, so the update operand passes through unchanged.
///
/// Batch-varying (batched) start indices cannot ride along structurally — every batch item needs its own update origin
/// while the lifted operation reads one origin for all batch items — so the rule falls back to per-item expansion via
/// `batch_by_item_expansion`: each batch item's input, update, and start indices are extracted (replicated operands
/// are used whole), updated per item, and restacked along a fresh leading batch axis (the result's batch axis is `0`
/// even when the operands carried their batch axes elsewhere). The expansion stages `O(batch_size)` operations — a
/// scatter-based rule is an explicit non-goal — and behaves identically in eager and tracing contexts because it
/// only goes through the value capability traits.
impl<C> BatchableOperation<C> for DynamicUpdateSliceOperation
where
    C: Context<Type = ArrayType> + Zero<C::Value>,
    C::Value: ZeroLike + Broadcast + Transpose + Slice + UpdateSlice + Reshape + Reshard,
    DynamicUpdateSliceOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();
        let axis_size = ArrayBatch::common_batch_size(inputs)?;
        if batch_axes[2..].iter().any(Option::is_some) {
            return batch_by_item_expansion(
                context,
                crate::operations::manipulation::DYNAMIC_UPDATE_SLICE_OPERATION_NAME,
                self,
                inputs,
                axis_size.expect("a mapped input pins the batch size"),
            );
        }
        let Some(batch_axis) = batch_axes[..2].iter().copied().flatten().next() else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        if inputs.len() == 2 {
            return Ok(vec![inputs[1].clone()]);
        }
        let axis_size = axis_size.expect("a mapped input pins the batch size");
        let input = inputs[0].match_axis(batch_axis, axis_size, context.axis_sharding().clone())?;
        let update = inputs[1].match_axis(batch_axis, axis_size, context.axis_sharding().clone())?;
        let zero_index = ArrayBatch::replicated(inputs[2].value().clone().zero_like());
        let mut lifted_inputs = vec![input, update];
        lifted_inputs.extend(inputs[2..].iter().cloned());
        lifted_inputs.insert(2 + batch_axis, zero_index);
        self.interpret_with_batch_axes(context, lifted_inputs.as_slice(), &[BatchAxis::from_position(batch_axis)])
    }
}

/// Represents the ability to overwrite a contiguous sub-array with an update value at start indices that are
/// computed at run time, with the semantics of StableHLO's
/// [`dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice) operation.
///
/// `t.dynamic_update_slice(update, start_indices)` returns a value equal to `t` except that the block whose origin
/// is given by the scalar integer values in `start_indices` (one per input axis) is replaced by `update`. Start
/// indices are clamped per StableHLO semantics so the updated block always lies in bounds: the effective start index
/// along axis `d` is `clamp(0, start_indices[d], input_dimension[d] - update_dimension[d])`. The update must have
/// the same data type and rank as the input, all of its dimensions must be static, and each axis must satisfy
/// `update_dimension[d] <= input_dimension[d]`; inputs with dynamic dimensions are rejected because that bound
/// cannot be proven against an unknown extent. The input, update, and start indices must reside in the same memory
/// space, and the output retains the complete input type, including layout and memory placement.
///
/// # Example
///
/// The following example shows how to use [`DynamicUpdateSlice`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::DynamicUpdateSlice;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::arrays::Array;
/// # use ryft_core::types::{ArrayType, DataType};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Overwrite the last two elements of the first row of a 2x3 matrix.
/// let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let update = Array::matrix(1, 2, vec![8.0, 9.0]);
/// let i = Array::from_f64s(ArrayType::scalar(DataType::I32), vec![0.0]);
/// let j = Array::from_f64s(ArrayType::scalar(DataType::I32), vec![1.0]);
/// let y = x.dynamic_update_slice(&update, &[i, j])?;
/// assert_eq!(y.to_f64s(), vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);
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
            return Err(TypeError::Invalid(format!(
                "'dynamic_update_slice' input data type {} does not match update data type {}",
                self.data_type(),
                update.data_type(),
            ))
            .into());
        }
        if self.memory() != update.memory() {
            return Err(TypeError::Invalid(format!(
                "'dynamic_update_slice' input and update must share one memory space but reside in {} and {}",
                self.memory(),
                update.memory(),
            ))
            .into());
        }
        let rank = self.rank();
        if update.rank() != rank {
            return Err(TypeError::Invalid(format!(
                "'dynamic_update_slice' update has rank {} but input has rank {rank}",
                update.rank()
            ))
            .into());
        }
        if start_indices.len() != rank {
            return Err(TypeError::Invalid(format!(
                "'dynamic_update_slice' expects one start index per input axis ({rank}) but got {}",
                start_indices.len(),
            ))
            .into());
        }
        validate_start_index_types(DYNAMIC_UPDATE_SLICE_OPERATION_NAME, self.memory(), start_indices)?;
        for axis in 0..rank {
            let update_dimension = update.dimension(axis);
            let Size::Static(update_size) = update_dimension else {
                return Err(TypeError::Invalid(format!(
                    "'dynamic_update_slice' does not support dynamic update axis {axis} with size \
                        {update_dimension}; update shapes must be static",
                ))
                .into());
            };
            let input_dimension = self.dimension(axis);
            let Size::Static(input_size) = input_dimension else {
                return Err(TypeError::Invalid(format!(
                    "'dynamic_update_slice' cannot prove that the update fits along dynamic input axis {axis} \
                        with size {input_dimension}",
                ))
                .into());
            };
            if update_size > input_size {
                return Err(TypeError::Invalid(format!(
                    "'dynamic_update_slice' update axis {axis} has size {update_size} which exceeds input size \
                        {input_size}",
                ))
                .into());
            }
        }
        // The output is distributed like the input operand (the update is written in place); the operand's placement
        // and reduction state carry through, with the update's varying-manual axes folded in.
        let sharding = update_slice_output_sharding(self, update, DYNAMIC_UPDATE_SLICE_OPERATION_NAME)?;
        self.clone().with_sharding(sharding).map_err(|error| TypeError::Invalid(error.to_string()).into())
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
        let mut outputs = self.dispatch_domain().bind(DynamicUpdateSliceOperation, Vec::new(), &inputs)?;
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

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let mut full_input_types = input_types.to_vec();
        full_input_types.extend(self.start_indices.iter().map(|index| index.r#type().into_owned()));
        DynamicSliceOperation::new(self.sizes.clone()).infer_output_types(full_input_types.as_slice(), &[])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("start_indices", format_args!("{}", render_factor_list(&self.start_indices)))?;
            operation.field("sizes", format_args!("{:?}", self.sizes))
        })
    }
}

impl<F, C> InterpretableOperation<C> for LinearDynamicSliceOperation<F>
where
    C: Domain<Type = ArrayType, Value: DynamicSlice>,
    F: CustomVjpResidual<C::Value>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
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
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]),
            MaybeZero::Value(cotangent) => {
                let zeros = MaybeZero::Zero(inputs[0].r#type().cotangent()).materialize(context)?;
                let outputs = context.stage_operation(
                    LinearDynamicUpdateSliceOperation::new(self.start_indices().to_vec()),
                    Vec::new(),
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
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        if inputs.is_empty() {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        }
        check_count!("output", outputs, 1, ProgramError);
        // One structural zero per operand: a contribution for the linear operand and zeros for the known indices.
        let mut contributions = inputs
            .iter()
            .map(|input| {
                let input_type = input.r#type();
                MaybeZero::Zero(input_type.cotangent())
            })
            .collect::<Vec<_>>();
        if let MaybeZero::Value(cotangent) = &outputs[0] {
            let start_indices = read_known_start_indices(&inputs[1..]);
            let zeros = MaybeZero::Zero(inputs[0].r#type().cotangent()).materialize(context)?;
            let mut operands = Vec::with_capacity(2 + start_indices.len());
            operands.push(zeros);
            operands.push(cotangent.clone());
            operands.extend(start_indices);
            let outputs = context.stage_operation(DynamicUpdateSliceOperation, Vec::new(), operands.as_slice())?;
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

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        let mut full_input_types = input_types.to_vec();
        full_input_types.extend(self.start_indices.iter().map(|index| index.r#type().into_owned()));
        DynamicUpdateSliceOperation.infer_output_types(full_input_types.as_slice(), &[])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("start_indices", format_args!("{}", render_factor_list(&self.start_indices)))
        })
    }
}

impl<F, C> InterpretableOperation<C> for LinearDynamicUpdateSliceOperation<F>
where
    C: Domain<Type = ArrayType, Value: DynamicUpdateSlice>,
    F: CustomVjpResidual<C::Value>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
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
    Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
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
                MaybeZero::Zero(inputs[0].r#type().cotangent()),
                MaybeZero::Zero(inputs[1].r#type().cotangent()),
            ]),
            MaybeZero::Value(cotangent) => {
                let update_sizes = static_update_sizes("'dynamic_update_slice' transpose", &inputs[1].r#type())?;
                let zeros = MaybeZero::Zero(inputs[1].r#type().cotangent()).materialize(context)?;
                let input_cotangents = context.stage_operation(
                    LinearDynamicUpdateSliceOperation::new(self.start_indices().to_vec()),
                    Vec::new(),
                    &[cotangent.clone(), zeros],
                )?;
                check_count!("output", input_cotangents, 1, ProgramError);
                let update_cotangents = context.stage_operation(
                    LinearDynamicSliceOperation::new(self.start_indices().to_vec(), update_sizes),
                    Vec::new(),
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", update_cotangents, 1, ProgramError);
                let update_cotangent =
                    update_cotangents.into_iter().next().unwrap().unalign_cotangent(&inputs[1].r#type().cotangent())?;
                Ok(vec![
                    MaybeZero::Value(input_cotangents.into_iter().next().unwrap()),
                    MaybeZero::Value(update_cotangent),
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
    Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }
        check_count!("output", outputs, 1, ProgramError);
        // One structural zero per operand: contributions for the linear input and update, and zeros for the known
        // start indices.
        let mut contributions = inputs
            .iter()
            .map(|input| {
                let input_type = input.r#type();
                MaybeZero::Zero(input_type.cotangent())
            })
            .collect::<Vec<_>>();
        if let MaybeZero::Value(cotangent) = &outputs[0] {
            let update_sizes = static_update_sizes("'dynamic_update_slice' transpose", &inputs[1].r#type())?;
            let start_indices = read_known_start_indices(&inputs[2..]);
            let zeros = MaybeZero::Zero(inputs[1].r#type().cotangent()).materialize(context)?;
            // Input cotangent: the output cotangent with the update window overwritten by zeros.
            let mut input_operands = Vec::with_capacity(2 + start_indices.len());
            input_operands.push(cotangent.clone());
            input_operands.push(zeros);
            input_operands.extend(start_indices.iter().cloned());
            let input_cotangents =
                context.stage_operation(DynamicUpdateSliceOperation, Vec::new(), input_operands.as_slice())?;
            check_count!("output", input_cotangents, 1, ProgramError);
            // Update cotangent: the dynamic slice of the output cotangent at the update window.
            let mut update_operands = Vec::with_capacity(1 + start_indices.len());
            update_operands.push(cotangent.clone());
            update_operands.extend(start_indices);
            let update_cotangents = context.stage_operation(
                DynamicSliceOperation::new(update_sizes),
                Vec::new(),
                update_operands.as_slice(),
            )?;
            check_count!("output", update_cotangents, 1, ProgramError);
            contributions[0] = MaybeZero::Value(input_cotangents.into_iter().next().unwrap());
            contributions[1] = MaybeZero::Value(
                update_cotangents.into_iter().next().unwrap().unalign_cotangent(&inputs[1].r#type().cotangent())?,
            );
        }
        Ok(contributions)
    }
}

/// Operation-name prefix used by [`static_update_sizes`] errors raised from the update-slice transpose rule.
const UPDATE_SLICE_TRANSPOSE_CONTEXT: &str = "'update_slice' transpose";

/// Extracts the static dimensions of an update operand type, reporting a precise error when any dimension is
/// dynamic. The `context` parameter selects the reported rule name because this helper serves both the static and
/// the captured-index update-slice transpose rules.
fn static_update_sizes(context: &str, update_type: &ArrayType) -> Result<Vec<usize>, ProgramError> {
    update_type
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .map(|(axis, size)| {
            size.value().ok_or_else(|| {
                TypeError::Invalid(format!("{context} requires a static update shape but axis {axis} has size {size}"))
                    .into()
            })
        })
        .collect()
}

/// Extracts item `item` of a per-item expansion operand: batched operands (whose batch axis must already sit at the
/// leading physical axis) contribute slice `item` with the batch axis dropped, while replicated operands are used
/// whole. Batched operand types must be fully static so the item slice bounds are provable; `operation_name` selects
/// the rule named in the error reported otherwise.
fn expansion_item<V>(operation_name: &'static str, input: &ArrayBatch<V>, item: usize) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Slice + Reshape,
{
    if input.batch_axis().is_replicated() {
        return Ok(input.value().clone());
    }
    let input_type = input.r#type().into_owned();
    let dimensions = input_type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| {
            dimension.value().ok_or_else(|| {
                TypeError::Invalid(format!(
                    "'{operation_name}' per-item expansion requires static batched operand types but got \
                         {input_type}",
                ))
                .into()
            })
        })
        .collect::<Result<Vec<usize>, ProgramError>>()?;
    let mut start_indices = vec![0; dimensions.len()];
    start_indices[0] = item;
    let mut limit_indices = dimensions.clone();
    limit_indices[0] = item + 1;
    let unit_strides = vec![1; dimensions.len()];
    let item_value =
        input
            .value()
            .clone()
            .slice(start_indices.as_slice(), limit_indices.as_slice(), unit_strides.as_slice())?;
    item_value.reshape(Shape::new(dimensions[1..].iter().map(|&dimension| Size::Static(dimension)).collect()))
}

/// Returns a copy of `sharding` with the placement of array dimension `index` replaced by `dimension`, preserving the
/// mesh, reduction state, and varying manual axes.
fn replace_sharding_dimension(
    sharding: &Sharding,
    index: usize,
    dimension: ShardingDimension,
) -> Result<Sharding, BatchingError> {
    let mut dimensions = sharding.dimensions().to_vec();
    dimensions[index] = dimension;
    Sharding::new(sharding.mesh().clone(), dimensions)
        .and_then(|updated| updated.with_unreduced_axes(sharding.unreduced_axes().clone()))
        .and_then(|updated| updated.with_reduced_axes(sharding.reduced_axes().clone()))
        .and_then(|updated| updated.with_varying_manual_axes(sharding.varying_manual_axes().clone()))
        .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })
}

/// Stacks per-item expansion results along a fresh leading batch axis of size `axis_size`: item `0` seeds the stacked
/// accumulator with replicated placement and later items overwrite their slices via [`UpdateSlice`] at static item
/// offsets. The completed accumulator is then resharded once to `axis_sharding`; keeping intermediate singleton
/// updates replicated avoids assigning a nontrivial mapped-axis sharding to an extent-one update. `interpret_item`
/// produces the per-item result; an empty batch axis is rejected with a precise error naming `operation_name` because
/// no batch item can seed the accumulator.
fn stack_expansion_items<V, InterpretItemFn>(
    operation_name: &'static str,
    axis_size: usize,
    axis_sharding: ShardingDimension,
    mut interpret_item: InterpretItemFn,
) -> Result<ArrayBatch<V>, BatchingError>
where
    V: Value<Type = ArrayType> + Broadcast + UpdateSlice + Reshape + Reshard,
    InterpretItemFn: FnMut(usize) -> Result<V, ProgramError>,
{
    let mut accumulator: Option<V> = None;
    for item in 0..axis_size {
        let output_item = interpret_item(item)?;
        let output_item_type = output_item.r#type().into_owned();
        accumulator = Some(match accumulator {
            None => {
                // Item `0` seeds the stacked accumulator by replication; later items overwrite their slices.
                ArrayBatch::replicated(output_item)
                    .broadcast(0, axis_size, ShardingDimension::Replicated)?
                    .into_value()
            }
            Some(accumulator) => {
                let mut expanded_dimensions = Vec::with_capacity(output_item_type.rank() + 1);
                expanded_dimensions.push(Size::Static(1));
                expanded_dimensions.extend(output_item_type.shape().dimensions().iter().cloned());
                let expanded = output_item.reshape(Shape::new(expanded_dimensions))?;
                let mut write_indices = vec![0; output_item_type.rank() + 1];
                write_indices[0] = item;
                accumulator.update_slice(&expanded, write_indices.as_slice())?
            }
        });
    }
    let Some(accumulator) = accumulator else {
        return Err(BatchingError::UnsupportedOperation {
            message: format!("'{operation_name}' does not support per-item expansion over an empty batch axis"),
        });
    };
    let accumulator = match accumulator.r#type().sharding() {
        Some(sharding) if sharding.dimensions().first() != Some(&axis_sharding) => {
            accumulator.reshard(&replace_sharding_dimension(sharding, 0, axis_sharding)?)
        }
        _ => accumulator,
    };
    let stacked_type = accumulator.r#type().into_owned();
    ArrayBatch::new(stacked_type, accumulator, Some(0))
}

/// Applies a single-output `operation` independently per batch item and restacks the results along a fresh leading
/// batch axis: every input is realigned so any mapped batch axis sits at the leading physical axis, item `item` of each
/// batched input is extracted via [`expansion_item`] (replicated inputs are used whole), and the per-item outputs
/// are stacked via [`stack_expansion_items`]. This is the shared fallback for batched operands that cannot ride
/// along structurally — batch-varying dynamic-slice start indices and batch-varying pad padding values — and it stages
/// `O(axis_size)` operations because everything goes through the value capability traits (which also makes it work
/// identically in eager and tracing contexts, since capabilities stage on tracers). For an empty batch, it infers the
/// per-item output type and synthesizes the correctly typed empty packed result without interpreting a nonexistent
/// item. Nonempty explicitly sharded mapped inputs are resharded to replicated placement before item extraction, and
/// the completed replicated accumulator is resharded once to the context's mapped placement. This avoids assigning a
/// nontrivial sharding to the extent-one slices used internally by the expansion.
pub(crate) fn batch_by_item_expansion<C, O>(
    context: &BatchingContext<C>,
    operation_name: &'static str,
    operation: &O,
    inputs: &[ArrayBatch<C::Value>],
    axis_size: usize,
) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayType> + Zero<C::Value>,
    C::Value: Broadcast + Transpose + Slice + UpdateSlice + Reshape + Reshard,
    O: InterpretableOperation<C>,
{
    if inputs.is_empty() {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    }
    if axis_size == 0 {
        let input_types = inputs.iter().map(ArrayBatch::unbatched_type).collect::<Vec<_>>();
        let mut output_types = operation.infer_output_types(input_types.as_slice(), &[])?;
        check_count!("output", output_types, 1, ProgramError);
        let output = context.parent().zero(&output_types.remove(0))?;
        return Ok(vec![ArrayBatch::replicated(output).broadcast(0, 0, context.axis_sharding().clone())?]);
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
            let replicated = replace_sharding_dimension(sharding, 0, ShardingDimension::Replicated)?;
            let value = aligned.value().reshard(&replicated);
            ArrayBatch::new(value.r#type().into_owned(), value, BatchAxis::new(0))
        })
        .collect::<Result<Vec<_>, BatchingError>>()?;
    let stacked = stack_expansion_items(operation_name, axis_size, context.axis_sharding().clone(), |item| {
        let item_inputs = aligned
            .iter()
            .map(|input| expansion_item(operation_name, input, item))
            .collect::<Result<Vec<_>, _>>()?;
        let mut outputs =
            operation.interpret(&context.parent().clone(), &crate::EmptyRegionDriver, item_inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    })?;
    Ok(vec![stacked])
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::jacobian::jacobian_forward;
    use crate::differentiation::{LinearizationTracer, value_and_gradient};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::math::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::Typed;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::Trace;
    use crate::types::{DataType, Layout, Memory, StridedLayout};

    use super::*;

    /// Returns a scalar integer-typed test array carrying `value` as its in-band payload.
    fn index(value: f64) -> Array {
        Array::from_f64s(ArrayType::scalar(DataType::I32), vec![value])
    }

    /// Lifts a scalar `i32` index constant into the trace or differentiation context that `exemplar` belongs to.
    fn index_constant<V>(exemplar: &V, value: f64) -> V
    where
        V: crate::programs::Value<Type = ArrayType>,
        V::DispatchDomain: crate::contexts::Context<Constant = Array>,
    {
        exemplar
            .dispatch_domain()
            .lift(Array::from_f64s(ArrayType::scalar(DataType::I32), vec![value]))
            .unwrap()
    }

    /// Returns a batch-varying scalar integer index batch carrying one start index per batch item, mapped at axis `0`.
    fn batch_varying_indices(values: Vec<f64>) -> ArrayBatch<Array> {
        let length = values.len();
        let value = Array::from_f64s(ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(length)])), values);
        ArrayBatch::new(value.r#type().into_owned(), value, Some(0)).unwrap()
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
                    input_types = [ArrayType::new(
                        DataType::F64,
                        Shape::new(vec![Size::Dynamic(None), Size::Static(3)]),
                    )],
                    error = "'slice' does not support dynamic input axis 0 with size *; slice bounds cannot be \
                        validated against an unknown extent",
                },
            ],
        );
        assert_eq!(input_type.slice(&[1, 1], &[2, 3], &[1, 1]), Ok(output_type.clone()));

        // Interpretation copies the selected block out of the row-major payload.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![5.0, 6.0]);

        // Empty slices produce empty payloads and rank-0 slices pass through.
        let empty = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).slice(&[1, 1], &[1, 3], &[1, 1]).unwrap();
        assert_eq!(empty.to_f64s(), Vec::<f64>::new());
        let scalar = Array::scalar(42.0).slice(&[], &[], &[]).unwrap();
        assert_eq!(scalar.to_f64s(), vec![42.0]);

        // Strided operations carry their strides through the builder, accessors, rendering, and inference: the
        // output dimension per axis is `ceil((limit - start) / stride)`.
        let strided = SliceOperation::new(vec![1], vec![6]).with_strides(vec![2]).unwrap();
        assert_eq!(strided.strides(), &[2]);
        assert_eq!(format!("{strided}"), "slice [start_indices=[1], limit_indices=[6], strides=[2]]");
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(6)]));
        assert_eq!(
            strided.infer_output_types(std::slice::from_ref(&vector_type), &[]),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]))]),
        );

        // Strided interpretation keeps the elements at `start + i * stride`.
        let vector = Array::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let strided_output = strided
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&vector))
            .unwrap();
        assert_eq!(strided_output[0].to_f64s(), vec![1.0, 3.0, 5.0]);

        // A stride larger than the sliced extent keeps a single element, and `start == limit` keeps none.
        let single = Array::vector(vec![0.0, 1.0, 2.0, 3.0]).slice(&[1], &[4], &[5]).unwrap();
        assert_eq!(single.to_f64s(), vec![1.0]);
        let strided_empty = Array::vector(vec![0.0, 1.0, 2.0, 3.0]).slice(&[2], &[2], &[2]).unwrap();
        assert_eq!(*strided_empty.r#type(), ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0)])));
        assert_eq!(strided_empty.to_f64s(), Vec::<f64>::new());

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))], &[]),
            Err(TypeError::Invalid("'slice' start_indices has length 2 but input has rank 1".to_string())),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2]).infer_output_types(std::slice::from_ref(&input_type), &[]),
            Err(TypeError::Invalid("'slice' limit_indices has length 1 but input has rank 2".to_string())),
        );
        assert_eq!(
            SliceOperation::new(vec![2, 0], vec![1, 3]).infer_output_types(std::slice::from_ref(&input_type), &[]),
            Err(TypeError::Invalid("'slice' start index 2 is greater than limit index 1 at axis 0".to_string())),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2, 4]).infer_output_types(std::slice::from_ref(&input_type), &[]),
            Err(TypeError::Invalid("'slice' limit index 4 is out of bounds for axis 1 with size 3".to_string())),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2, 3]).with_strides(vec![2]),
            Err(ProgramError::Type(TypeError::Invalid(
                "'slice' strides has length 1 but start_indices has length 2".to_string()
            ))),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2, 3]).with_strides(vec![1, 0]),
            Err(ProgramError::Type(TypeError::Invalid(
                "'slice' strides must be at least 1 but axis 1 has stride 0".to_string()
            ))),
        );
        assert_eq!(
            input_type.slice(&[0, 0], &[2, 3], &[1]),
            Err(ProgramError::Type(TypeError::Invalid(
                "'slice' strides has length 1 but input has rank 2".to_string()
            ))),
        );
        assert_eq!(
            input_type.slice(&[0, 0], &[2, 3], &[1, 0]),
            Err(ProgramError::Type(TypeError::Invalid(
                "'slice' strides must be at least 1 but axis 1 has stride 0".to_string()
            ))),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured indices.
        let mut builder = ProgramBuilder::<Array, SliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, Vec::new(), vec![program_input]).unwrap()[0];
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

        // Check standard partial evaluation with known and residual operands.
        let input = Array::vector(vec![0.0, 1.0, 2.0, 3.0]);
        let expected = Array::vector(vec![1.0, 2.0]);
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
                    ))],
                    outputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 2.0, 5.0, 6.0]))],
                },
                {
                    inputs = [(@replicated, Array::vector(vec![0.0, 1.0, 2.0, 3.0]))],
                    outputs = [(@replicated, Array::vector(vec![1.0, 2.0]))],
                },
            ],
        );

        // Static slicing is linear; check both the JVP and the unit- and non-unit-stride pullbacks.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = SliceOperation::new(vec![1], vec![3]),
            cases = [{
                primals = [Array::vector(vec![0.0, 1.0, 2.0, 3.0])],
                tangents = [Array::vector(vec![4.0, 5.0, 6.0, 7.0])],
                primal_outputs = [Array::vector(vec![1.0, 2.0])],
                tangent_outputs = [Array::vector(vec![5.0, 6.0])],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = SliceOperation::new(vec![1], vec![3]),
            cases = [{
                inputs = [(@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![4.into()]))))],
                output_cotangents = [Array::vector(vec![5.0, 7.0])],
                input_cotangents = [Array::vector(vec![0.0, 5.0, 7.0, 0.0])],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = SliceOperation::new(vec![1], vec![6]).with_strides(vec![2]).unwrap(),
            cases = [
                {
                    inputs = [(@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![6.into()]))))],
                    output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0])],
                    input_cotangents = [Array::vector(vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0])],
                },
                {
                    inputs = [(@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![6.into()]))
                        .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
                        .with_memory(Memory::Host { pinned: true })))],
                    output_cotangents = [Array::from_f64s(
                        ArrayType::new(DataType::F64, Shape::new(vec![3.into()]))
                            .with_memory(Memory::Host { pinned: true }),
                        vec![1.0, 2.0, 3.0],
                    )],
                    input_cotangents = [Array::from_f64s(
                        ArrayType::new(DataType::F64, Shape::new(vec![6.into()]))
                            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
                            .with_memory(Memory::Host { pinned: true }),
                        vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
                    )],
                },
            ],
        );
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
                        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(2)])),
                    ],
                    error = "'update_slice' input data type f64 does not match update data type f32",
                },
            ],
        );
        assert_eq!(input_type.update_slice(&update_type, &[0, 1]), Ok(input_type.clone()));
        assert_eq!(
            UpdateSliceOperation::new(vec![0, usize::MAX])
                .infer_output_types(&[input_type.clone(), update_type.clone()], &[]),
            Err(TypeError::Invalid(format!(
                "'update_slice' update axis 1 with start index {} and size 2 does not fit in input size 3",
                usize::MAX,
            ))),
        );

        // Interpretation overwrites the selected block of the row-major payload.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let update = Array::matrix(1, 2, vec![8.0, 9.0]);
        let output = operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[input, update]).unwrap();
        assert_eq!(*output[0].r#type(), input_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);

        // Rank-0 updates replace the input entirely.
        let scalar = Array::scalar(1.0).update_slice(&Array::scalar(7.0), &[]).unwrap();
        assert_eq!(scalar.to_f64s(), vec![7.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(
                &[input_type.clone(), ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),],
                &[],
            ),
            Err(TypeError::Invalid("'update_slice' update has rank 1 but input has rank 2".to_string())),
        );
        assert_eq!(
            UpdateSliceOperation::new(vec![0]).infer_output_types(&[input_type.clone(), update_type.clone()], &[]),
            Err(TypeError::Invalid("'update_slice' start_indices has length 1 but input has rank 2".to_string())),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    input_type.clone(),
                    ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(2)])),
                ],
                &[],
            ),
            Err(TypeError::Invalid(
                "'update_slice' does not support dynamic update axis 0 with size *; update shapes must be \
                    static"
                    .to_string()
            )),
        );
        assert_eq!(
            UpdateSliceOperation::new(vec![0, 2]).infer_output_types(&[input_type.clone(), update_type.clone()], &[]),
            Err(TypeError::Invalid(
                "'update_slice' update axis 1 with start index 2 and size 2 does not fit in input size 3".to_string()
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(Some(4)), Size::Static(3)])),
                    update_type.clone(),
                ],
                &[],
            ),
            Err(TypeError::Invalid(
                "'update_slice' cannot prove that the update fits along dynamic input axis 0 with size <4".to_string()
            )),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured indices.
        let mut builder = ProgramBuilder::<Array, UpdateSliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_update = builder.add_input(update_type);
        let program_output =
            builder.add_instruction(operation, Vec::new(), vec![program_input, program_update]).unwrap()[0];
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

        // Check standard partial evaluation with known and residual operands.
        let input = Array::vector(vec![0.0, 1.0, 2.0, 3.0]);
        let update = Array::vector(vec![8.0, 9.0]);
        let expected = Array::vector(vec![0.0, 8.0, 9.0, 3.0]);
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

        // Batching aligns mapped and replicated operands before applying the update independently to each item.
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
                        )),
                        (@replicated, Array::vector(vec![9.0, 9.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        4,
                        vec![0.0, 9.0, 9.0, 3.0, 4.0, 9.0, 9.0, 7.0],
                    ))],
                },
                {
                    inputs = [
                        (@replicated, Array::vector(vec![0.0, 1.0, 2.0, 3.0])),
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![8.0, 8.0, 9.0, 9.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        4,
                        vec![0.0, 8.0, 8.0, 3.0, 0.0, 9.0, 9.0, 3.0],
                    ))],
                },
            ],
        );

        // Static update-slice is jointly linear in the operand and update.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = UpdateSliceOperation::new(vec![1]),
            cases = [{
                primals = [
                    Array::vector(vec![0.0, 1.0, 2.0, 3.0]),
                    Array::vector(vec![8.0, 9.0]),
                ],
                tangents = [
                    Array::vector(vec![1.0, 2.0, 3.0, 4.0]),
                    Array::vector(vec![5.0, 6.0]),
                ],
                primal_outputs = [Array::vector(vec![0.0, 8.0, 9.0, 3.0])],
                tangent_outputs = [Array::vector(vec![1.0, 5.0, 6.0, 4.0])],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = UpdateSliceOperation::new(vec![1]),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![4.into()])))),
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()])))),
                ],
                output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0])],
                input_cotangents = [
                    Array::vector(vec![1.0, 0.0, 0.0, 4.0]),
                    Array::vector(vec![2.0, 3.0]),
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
                output_cotangents = [Array::from_f64s(input_type.clone(), vec![1.0, 2.0, 3.0, 4.0])],
                input_cotangents = [
                    Array::from_f64s(input_type, vec![1.0, 0.0, 0.0, 4.0]),
                    Array::from_f64s(update_type, vec![2.0, 3.0]),
                ],
            }],
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
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input_type.clone(), index_type.clone(), index_type.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [],
                    error = "'dynamic_slice' expects an input operand followed by its start index operands but got \
                        no inputs",
                },
                {
                    input_types = [input_type.clone(), index_type.clone()],
                    error = "'dynamic_slice' expects one start index per input axis (2) but got 1",
                },
                {
                    input_types = [input_type.clone(), ArrayType::scalar(DataType::F64), index_type.clone()],
                    error = "'dynamic_slice' start index 0 must be a scalar integer but has type f64[]",
                },
            ],
        );
        assert_eq!(
            input_type.dynamic_slice(&[index_type.clone(), index_type.clone()], &[1, 2]),
            Ok(output_type.clone()),
        );

        // Interpretation extracts the block at the in-band start indices.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[input.clone(), index(1.0), index(1.0)])
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![5.0, 6.0]);

        // Out-of-bounds start indices clamp per StableHLO semantics: the effective start index along axis `d` is
        // `clamp(0, start_indices[d], input_dimension[d] - sizes[d])`.
        let clamped = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[input.clone(), index(5.0), index(-2.0)])
            .unwrap();
        assert_eq!(clamped[0].to_f64s(), vec![4.0, 5.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            DynamicSliceOperation::new(vec![1])
                .infer_output_types(&[input_type.clone(), index_type.clone(), index_type.clone(),], &[]),
            Err(TypeError::Invalid("'dynamic_slice' sizes has length 1 but input has rank 2".to_string())),
        );
        assert_eq!(
            DynamicSliceOperation::new(vec![1, 4])
                .infer_output_types(&[input_type.clone(), index_type.clone(), index_type.clone(),], &[]),
            Err(TypeError::Invalid("'dynamic_slice' size 4 is out of bounds for axis 1 with size 3".to_string())),
        );
        // A dynamic input axis is accepted: StableHLO clamping keeps the read in bounds against the unknown extent
        // and the output dimension is still the static `sizes[axis]`. The static axis 1 still validates `2 <= 3`.
        assert_eq!(
            operation.infer_output_types(
                &[
                    ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(3)])),
                    index_type.clone(),
                    index_type.clone(),
                ],
                &[],
            ),
            Ok(vec![output_type.clone()]),
        );
        // A bounded-dynamic input axis is accepted when its maximum possible extent can contain the static slice.
        assert_eq!(
            operation.infer_output_types(
                &[
                    ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(Some(2)), Size::Static(3)])),
                    index_type.clone(),
                    index_type.clone(),
                ],
                &[],
            ),
            Ok(vec![output_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(Some(1)), Size::Static(3)])),
                    index_type.clone(),
                    index_type.clone(),
                ],
                &[],
            ),
            Err(TypeError::Invalid(
                "'dynamic_slice' size 1 cannot fit axis 0 with exclusive upper bound 1".to_string()
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    input_type.clone(),
                    ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(2)])),
                    index_type.clone(),
                ],
                &[],
            ),
            Err(TypeError::Invalid(
                "'dynamic_slice' start index 0 must be a scalar integer but has type i32[2]".to_string()
            )),
        );
        assert_eq!(
            operation
                .infer_output_types(&[input_type.clone(), index_type.clone(), ArrayType::scalar(DataType::I64)], &[]),
            Err(TypeError::Invalid(
                "'dynamic_slice' start indices must share one integer type but index 1 has type i64[] and \
                    index 0 has type i32[]"
                    .to_string()
            )),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured sizes.
        let mut builder = ProgramBuilder::<Array, DynamicSliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_index_0 = builder.add_input(index_type.clone());
        let program_index_1 = builder.add_input(index_type);
        let program_output = builder
            .add_instruction(operation, Vec::new(), vec![program_input, program_index_0, program_index_1])
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

        // Partial evaluation folds known starts and residualizes the read when the operand remains unknown.
        let input = Array::vector(vec![0.0, 1.0, 2.0, 3.0]);
        let start = index(1.0);
        let expected = Array::vector(vec![1.0, 2.0]);
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
                    )),
                    (@replicated, start),
                ],
                outputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 2.0, 5.0, 6.0]))],
            }],
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
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    input_types = [input_type.clone(), update_type.clone(), index_type.clone(), index_type.clone()],
                    output_types = [input_type.clone()],
                },
                {
                    input_types = [input_type.clone()],
                    error = "'dynamic_update_slice' expects an input operand and an update operand followed by start \
                        index operands but got 1 inputs",
                },
                {
                    input_types = [input_type.clone(), update_type.clone(), index_type.clone()],
                    error = "'dynamic_update_slice' expects one start index per input axis (2) but got 1",
                },
            ],
        );
        assert_eq!(
            input_type.dynamic_update_slice(&update_type, &[index_type.clone(), index_type.clone()]),
            Ok(input_type.clone()),
        );

        // Interpretation overwrites the block at the in-band start indices.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let update = Array::matrix(1, 2, vec![8.0, 9.0]);
        let output = operation
            .interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[input.clone(), update.clone(), index(0.0), index(1.0)],
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
                &[input.clone(), update.clone(), index(5.0), index(-3.0)],
            )
            .unwrap();
        assert_eq!(clamped[0].to_f64s(), vec![1.0, 2.0, 3.0, 8.0, 9.0, 6.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(
                &[
                    input_type.clone(),
                    ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(2)])),
                    index_type.clone(),
                    index_type.clone(),
                ],
                &[],
            ),
            Err(TypeError::Invalid(
                "'dynamic_update_slice' input data type f64 does not match update data type f32".to_string()
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    input_type.clone(),
                    ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),
                    index_type.clone(),
                    index_type.clone(),
                ],
                &[],
            ),
            Err(TypeError::Invalid("'dynamic_update_slice' update has rank 1 but input has rank 2".to_string())),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    input_type.clone(),
                    ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(2)])),
                    index_type.clone(),
                    index_type.clone(),
                ],
                &[],
            ),
            Err(TypeError::Invalid(
                "'dynamic_update_slice' does not support dynamic update axis 0 with size *; update shapes \
                    must be static"
                    .to_string()
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    input_type.clone(),
                    ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(4)])),
                    index_type.clone(),
                    index_type.clone(),
                ],
                &[],
            ),
            Err(TypeError::Invalid(
                "'dynamic_update_slice' update axis 1 has size 4 which exceeds input size 3".to_string()
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(3)])),
                    update_type.clone(),
                    index_type.clone(),
                    index_type.clone(),
                ],
                &[],
            ),
            Err(TypeError::Invalid(
                "'dynamic_update_slice' cannot prove that the update fits along dynamic input axis 0 with \
                    size *"
                    .to_string()
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                &[input_type.clone(), update_type.clone(), ArrayType::scalar(DataType::F64), index_type.clone(),],
                &[],
            ),
            Err(TypeError::Invalid(
                "'dynamic_update_slice' start index 0 must be a scalar integer but has type f64[]".to_string()
            )),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );

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

        // Partial evaluation folds known updates and residualizes an unknown operand with captured start indices.
        let input = Array::vector(vec![0.0, 1.0, 2.0, 3.0]);
        let update = Array::vector(vec![8.0, 9.0]);
        let start = index(1.0);
        let expected = Array::vector(vec![0.0, 8.0, 9.0, 3.0]);
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = DynamicUpdateSliceOperation,
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

        // Replicated starts align the input and update on one mapped axis.
        check_operation_batching!(
            @exact,
            operation = DynamicUpdateSliceOperation,
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::matrix(
                        2,
                        4,
                        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    )),
                    (@replicated, update),
                    (@replicated, start),
                ],
                outputs = [(@mapped(axis = 0), Array::matrix(
                    2,
                    4,
                    vec![0.0, 8.0, 9.0, 3.0, 4.0, 8.0, 9.0, 7.0],
                ))],
            }],
        );
    }

    #[test]
    fn test_array_slicing() {
        // Rank-3 slice exercises the row-major odometer across non-contiguous blocks.
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)]));
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = Array::from_f64s(input_type.clone(), values.clone())
            .slice(&[0, 1, 2], &[2, 3, 4], &[1, 1, 1])
            .unwrap();
        assert_eq!(
            *output.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(output.to_f64s(), vec![6.0, 7.0, 10.0, 11.0, 18.0, 19.0, 22.0, 23.0]);

        // The matching update-slice writes the block back into place.
        let update = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)])),
            vec![-6.0, -7.0, -10.0, -11.0, -18.0, -19.0, -22.0, -23.0],
        );
        let updated = Array::from_f64s(input_type, values).update_slice(&update, &[0, 1, 2]).unwrap();
        assert_eq!(
            updated.to_f64s(),
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -6.0, -7.0, 8.0, 9.0, -10.0, -11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                -18.0, -19.0, 20.0, 21.0, -22.0, -23.0,
            ],
        );

        // Strided slicing walks the row-major odometer with per-axis steps: rows with stride 2 and columns with
        // stride 3 keep elements at indices (0, 0), (0, 3), (1, 0), and (1, 3) of a 2x3x4 input's last two axes.
        let strided = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)])),
            (0..24).map(|value| value as f64).collect(),
        )
        .slice(&[0, 0, 0], &[2, 3, 4], &[2, 2, 3])
        .unwrap();
        assert_eq!(
            *strided.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(strided.to_f64s(), vec![0.0, 3.0, 8.0, 11.0]);

        // The dynamic kernels validate their index operand shapes eagerly.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            input.dynamic_slice(&[index(0.0), Array::vector(vec![1.0, 2.0])], &[1, 2]),
            Err(ProgramError::Type(TypeError::Invalid(
                "'dynamic_slice' start index 1 must be a scalar integer but has type f64[2]".to_string()
            ))),
        );
        assert_eq!(
            input.dynamic_update_slice(&Array::matrix(1, 2, vec![8.0, 9.0]), &[index(0.0)]),
            Err(ProgramError::Type(TypeError::Invalid(
                "'dynamic_update_slice' expects one start index per input axis (2) but got 1".to_string()
            ))),
        );
    }

    #[test]
    fn test_array_type_slicing() {
        use std::collections::BTreeSet;

        use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        // Every slicing operation preserves the operand's memory placement, and operations with update or dynamic
        // index operands reject combinations that would require an implicit transfer.
        let host_operand =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)])).with_memory(Memory::Host { pinned: true });
        let host_update =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)])).with_memory(Memory::Host { pinned: true });
        let host_index = ArrayType::scalar(DataType::I32).with_memory(Memory::Host { pinned: true });
        let laid_out_host_operand = host_operand.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(laid_out_host_operand.slice(&[0], &[4], &[1]), Ok(laid_out_host_operand.clone()));
        assert_eq!(
            laid_out_host_operand.dynamic_slice(std::slice::from_ref(&host_index), &[4]),
            Ok(laid_out_host_operand),
        );
        assert_eq!(host_operand.slice(&[0], &[2], &[1]).unwrap().memory(), Memory::Host { pinned: true });
        assert_eq!(
            host_operand.dynamic_slice(std::slice::from_ref(&host_index), &[2]).unwrap().memory(),
            Memory::Host { pinned: true },
        );
        assert_eq!(host_operand.update_slice(&host_update, &[0]).unwrap().memory(), Memory::Host { pinned: true });
        assert_eq!(
            host_operand.dynamic_update_slice(&host_update, std::slice::from_ref(&host_index)).unwrap().memory(),
            Memory::Host { pinned: true },
        );
        assert_eq!(
            host_operand.update_slice(&ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)])), &[0]),
            Err(ProgramError::Type(TypeError::Invalid(
                "'update_slice' input and update must share one memory space but reside in Host[Pinned] and \
                          Device"
                    .to_string()
            ))),
        );
        assert_eq!(
            host_operand.dynamic_slice(&[ArrayType::scalar(DataType::I32)], &[2]),
            Err(ProgramError::Type(TypeError::Invalid(
                "'dynamic_slice' operand and start indices must share one memory space but start index 0 \
                          resides in Device and the operand resides in Host[Pinned]"
                    .to_string()
            ))),
        );

        {
            let mesh = LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
                MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            ])
            .unwrap();
            // [4, 4] sharded over `x` on axis 0 and unreduced over the manual axis `m`.
            let sharding =
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["m"])
                    .unwrap();
            let input = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(4)]))
                .with_sharding(sharding.clone())
                .unwrap();
            let start = ArrayType::scalar(DataType::I32);

            // Static and dynamic slicing preserve the sharding for evenly divisible output sizes and reject an
            // incompatible size on an explicitly sharded axis.
            assert_eq!(input.slice(&[0, 0], &[2, 4], &[1, 1]).unwrap().sharding(), Some(&sharding));
            assert!(input.slice(&[0, 0], &[3, 4], &[1, 1]).is_err());
            assert_eq!(
                input.dynamic_slice(&[start.clone(), start.clone()], &[2, 4]).unwrap().sharding(),
                Some(&sharding),
            );
            assert!(input.dynamic_slice(&[start.clone(), start.clone()], &[3, 4]).is_err());
        }

        {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            let sharded =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap();
            let replicated =
                Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::replicated()]).unwrap();
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

            // Static and dynamic updates keep matching operand placement and reject explicit placement conflicts.
            assert_eq!(operand.update_slice(&matching, &[0, 0]).unwrap().sharding(), Some(&sharded));
            assert!(operand.update_slice(&conflicting, &[0, 0]).is_err());
            assert_eq!(
                operand.dynamic_update_slice(&matching, &[start.clone(), start.clone()]).unwrap().sharding(),
                Some(&sharded),
            );
            assert!(operand.dynamic_update_slice(&conflicting, &[start.clone(), start.clone()]).is_err());
        }

        {
            let mesh = LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
                MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            ])
            .unwrap();
            let dimensions = || vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
            let operand = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(4)]))
                .with_sharding(Sharding::new(mesh.clone(), dimensions()).unwrap())
                .unwrap();
            let update = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(4)]))
                .with_sharding(Sharding::new(mesh, dimensions()).unwrap().with_varying_manual_axes(["m"]).unwrap())
                .unwrap();

            // An update that varies over a manual axis makes the result vary over that axis as well.
            let output = operand.update_slice(&update, &[0, 0]).unwrap();
            assert_eq!(output.sharding().unwrap().varying_manual_axes(), &BTreeSet::from(["m".to_string()]));
        }
    }

    #[test]
    fn test_dynamic_slice_differentiation() {
        // Slice a [1, 2] block at start (1, 1) of a [2, 3] operand: the operand is linear and the scalar start indices
        // are the known operands. The sliced output and its cotangent have shape [1, 2].
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let cotangent = Array::matrix(1, 2, vec![5.0, 7.0]);
        let sizes = vec![1, 2];

        check_operation_transposition!(
            @exact,
            operation = DynamicSliceOperation::new(sizes),
            cases = [{
                inputs = [
                    (@linear(type = operand_type)),
                    (@known, index(1.0)),
                    (@known, index(1.0)),
                ],
                output_cotangents = [cotangent],
                input_cotangents = [Array::matrix(2, 3, vec![0.0, 0.0, 0.0, 0.0, 5.0, 7.0])],
            }],
        );

        // Forward mode through `f(x) = dynamic_slice(x, [1], [2])` exercises the captured-index dynamic slice under
        // batched basis tangents.
        let jacobian = jacobian_forward(
            |x| {
                let start = index_constant(&x, 1.0);
                Ok(x.dynamic_slice(&[start], &[2]).unwrap())
            },
            Array::vector(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(block.value().values(), &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_dynamic_update_slice_differentiation() {
        // Update a [1, 2] block at start (0, 1) of a [2, 3] input: the input and update are linear and the scalar
        // start indices are the known operands. The output and its cotangent have shape [2, 3].
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2)]));
        let cotangent = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        check_operation_transposition!(
            @exact,
            operation = DynamicUpdateSliceOperation,
            cases = [{
                inputs = [
                    (@linear(type = input_type)),
                    (@linear(type = update_type)),
                    (@known, index(0.0)),
                    (@known, index(1.0)),
                ],
                output_cotangents = [cotangent],
                input_cotangents = [
                    Array::matrix(2, 3, vec![1.0, 0.0, 0.0, 4.0, 5.0, 6.0]),
                    Array::matrix(1, 2, vec![2.0, 3.0]),
                ],
            }],
        );

        // Dynamic update-slice restores the layout-bearing update cotangent after its dynamic slice.
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![4.into()])).with_memory(Memory::Host { pinned: true });
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        let start =
            Array::from_f64s(ArrayType::scalar(DataType::I32).with_memory(Memory::Host { pinned: true }), vec![1.0]);
        check_operation_transposition!(
            @exact,
            operation = DynamicUpdateSliceOperation,
            cases = [{
                inputs = [
                    (@linear(type = input_type.clone())),
                    (@linear(type = update_type.clone())),
                    (@known, start),
                ],
                output_cotangents = [Array::from_f64s(input_type.clone(), vec![1.0, 2.0, 3.0, 4.0])],
                input_cotangents = [
                    Array::from_f64s(input_type, vec![1.0, 0.0, 0.0, 4.0]),
                    Array::from_f64s(update_type, vec![2.0, 3.0]),
                ],
            }],
        );

        // Composing JVP and transposition must retain the captured start index: the input gradient is the output
        // cotangent with the update window zeroed, while the update gradient is that window of the cotangent.
        let (value, (input_gradient, update_gradient)) = value_and_gradient(
            |(x, update)| {
                let start = index_constant(&x, 1.0);
                x.dynamic_update_slice(&update, &[start]).unwrap().reduce(&[0], ReductionKind::Sum)
            },
            (Array::vector(vec![1.0, 2.0, 3.0, 4.0]), Array::vector(vec![7.0, 8.0])),
        )
        .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 20.0, epsilon = 1e-9);
        assert_eq!(input_gradient.to_f64s(), vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(update_gradient.to_f64s(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_slice_batching_sharding() {
        use crate::operations::manipulation::Slice;
        use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        // The full input is [2 (batch), 4]: the batch axis is replicated and the data axis is sharded over `x`.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(4)]))
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
    fn test_update_slice_batching_sharding() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let physical_sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(4)]))
                .with_sharding(physical_sharding.clone())
                .unwrap();
            let update_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))
                .with_sharding(Sharding::replicated(mesh, 1))
                .unwrap();
            let context = BatchingContext::new(EagerContext::<Array>::new(), 2)
                .with_axis_sharding(ShardingDimension::sharded(["x"]));
            let make_input = || {
                ArrayBatch::new(
                    input_type.clone(),
                    Array::from_f64s(input_type.clone(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
                    BatchAxis::new(0),
                )
                .unwrap()
            };
            let make_update = || ArrayBatch::replicated(Array::from_f64s(update_type.clone(), vec![9.0, 9.0]));

            let static_outputs = UpdateSliceOperation::new(vec![1])
                .batch(&context, &crate::EmptyRegionDriver, &[make_input(), make_update()])
                .unwrap();
            let dynamic_outputs = DynamicUpdateSliceOperation
                .batch(
                    &context,
                    &crate::EmptyRegionDriver,
                    &[make_input(), make_update(), ArrayBatch::replicated(index(1.0))],
                )
                .unwrap();

            for output in [static_outputs[0].clone(), dynamic_outputs[0].clone()] {
                assert_eq!(output.batch_axis(), BatchAxis::new(0));
                assert_eq!(output.r#type().sharding().unwrap().dimensions(), physical_sharding.dimensions());
                assert_eq!(output.value().to_f64s(), vec![0.0, 9.0, 9.0, 3.0, 4.0, 9.0, 9.0, 7.0]);
            }
        }
    }

    #[test]
    fn test_dynamic_slice_batching_expands_batch_varying_indices() {
        // Batch-varying start indices over a replicated operand expand per item: item 0 reads `x[0..2]` and item 1
        // reads `x[2..4]` of the shared operand, restacked along a fresh leading batch axis.
        let uniform = ArrayBatch::replicated(Array::vector(vec![0.0, 1.0, 2.0, 3.0]));
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(
                &BatchingContext::new(crate::EagerContext::<Array>::new(), 2),
                &crate::EmptyRegionDriver,
                &[uniform, batch_varying_indices(vec![0.0, 2.0])],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().shape().dimensions(), &[Size::Static(2), Size::Static(2)]);
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0, 1.0, 2.0, 3.0]);

        // A batched operand pairs item `i` of the operand with item `i` of the indices; item 1's start index 3 is
        // clamped to 2 so the extracted block stays in bounds.
        let input = {
            let value = Array::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(
                &BatchingContext::new(crate::EagerContext::<Array>::new(), 2),
                &crate::EmptyRegionDriver,
                &[input, batch_varying_indices(vec![1.0, 3.0])],
            )
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![1.0, 2.0, 6.0, 7.0]);

        // An operand batched on a non-leading axis is realigned to the fresh leading batch axis first: the physical
        // `[4, 2]` operand carries per-item vectors `[0, 1, 2, 3]` and `[4, 5, 6, 7]` along axis 1.
        let trailing = {
            let value = Array::matrix(4, 2, vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(1))
        }
        .unwrap();
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(
                &BatchingContext::new(crate::EagerContext::<Array>::new(), 2),
                &crate::EmptyRegionDriver,
                &[trailing, batch_varying_indices(vec![1.0, 2.0])],
            )
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![1.0, 2.0, 6.0, 7.0]);
    }

    #[test]
    fn test_dynamic_update_slice_batching_expands_batch_varying_indices() {
        // A batched update with batch-varying start indices over a replicated input expands per item: item 0
        // writes `[9, 9]` at offset 0 and item 1 writes `[8, 8]` at offset 2 of the shared input.
        let uniform_input = ArrayBatch::replicated(Array::vector(vec![0.0, 1.0, 2.0, 3.0]));
        let update = {
            let value = Array::matrix(2, 2, vec![9.0, 9.0, 8.0, 8.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = DynamicUpdateSliceOperation
            .batch(
                &BatchingContext::new(crate::EagerContext::<Array>::new(), 2),
                &crate::EmptyRegionDriver,
                &[uniform_input, update, batch_varying_indices(vec![0.0, 2.0])],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().shape().dimensions(), &[Size::Static(2), Size::Static(4)]);
        assert_eq!(outputs[0].value().to_f64s(), vec![9.0, 9.0, 2.0, 3.0, 0.0, 1.0, 8.0, 8.0]);

        // A batched input with a replicated update writes the same block at each batch item's own offset.
        let input = {
            let value = Array::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let uniform_update = ArrayBatch::replicated(Array::vector(vec![9.0, 9.0]));
        let outputs = DynamicUpdateSliceOperation
            .batch(
                &BatchingContext::new(crate::EagerContext::<Array>::new(), 2),
                &crate::EmptyRegionDriver,
                &[input, uniform_update, batch_varying_indices(vec![1.0, 0.0])],
            )
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0, 9.0, 9.0, 3.0, 9.0, 9.0, 6.0, 7.0]);
    }

    #[test]
    fn test_dynamic_slice_batching_under_tracing() {
        // vmap-under-tracing composition: each batch item extracts a window of the differentiated vector at its own
        // start index, so the batching rule must stage the per-item expansion (instead of rejecting the batch-varying
        // indices) and the staged slicing operations must transpose. With `starts = [1, 2]` over `x = [1, 2, 3, 4]`
        // the batch items read `[x1, x2]` and `[x2, x3]`, so `f(x) = sum(stack * w)` with `w = [[1, 2], [3, 4]]` is
        // `f = x1 + 2 * x2 + 3 * x2 + 4 * x3` and the gradient is `[0, 1, 5, 4]`.
        let (value, gradient) = value_and_gradient(
            |x| {
                let context = x.context().clone();
                let starts = context
                    .lift(Array::from_f64s(
                        ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(2)])),
                        vec![1.0, 2.0],
                    ))
                    .unwrap();
                let stacked: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = batch(
                    |(item, start)| item.dynamic_slice(&[start], &[2]),
                    (x, starts),
                    (BatchAxis::replicated(), BatchAxis::new(0)),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap();
                let weights = context.lift(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])).unwrap();
                (stacked * weights).reduce(&[0, 1], ReductionKind::Sum)
            },
            Array::vector(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        // f = 1 * 2 + 2 * 3 + 3 * 3 + 4 * 4 = 33.
        assert_abs_diff_eq!(value.to_f64s()[0], 33.0, epsilon = 1e-9);
        assert_eq!(gradient.to_f64s(), vec![0.0, 1.0, 5.0, 4.0]);
    }
}
