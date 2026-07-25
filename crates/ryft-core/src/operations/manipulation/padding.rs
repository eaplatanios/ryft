use std::fmt::Display;

use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    ElementwiseDerivativeAlignment, TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::constants::{One, OneOperation, Zero, ZeroOperation};
use crate::operations::control_flow::{Select, SelectOperation};
use crate::operations::manipulation::{Broadcast, SliceOperation, Transpose};
use crate::operations::math::{ReduceOperation, ReductionKind};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::programs::{MaybeZero, ProgramError};
use crate::sharding::Sharding;
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, DataType, Dimension, Shape};

// TODO(eaplatanios): Review this.

use super::slicing::resized_output_sharding;

/// Canonical operation name for [`PadOperation`].
pub const PAD_OPERATION_NAME: &str = "pad";

/// [`Operation`] that expands its first operand by adding edge and interior padding filled with its second (scalar)
/// operand. Refer to the documentation of [`Pad`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PadOperation {
    /// Padding added before the first element of each input axis.
    edge_padding_low: Vec<i64>,

    /// Padding added after the last element of each input axis.
    edge_padding_high: Vec<i64>,

    /// Padding added between any two adjacent elements of each input axis.
    interior_padding: Vec<usize>,
}

impl PadOperation {
    /// Creates a new [`PadOperation`] with the provided edge and interior padding amounts. The three vectors must
    /// share one length (one entry per input axis); whether that shared length matches the input rank is validated
    /// during type inference, once an input type is known.
    pub fn new(
        edge_padding_low: Vec<i64>,
        edge_padding_high: Vec<i64>,
        interior_padding: Vec<usize>,
    ) -> Result<Self, ProgramError> {
        if edge_padding_low.len() != edge_padding_high.len() || edge_padding_low.len() != interior_padding.len() {
            return Err(TypeError::invalid(format!(
                "'pad' expects edge_padding_low, edge_padding_high, and interior_padding to share one length but \
                    got lengths {}, {}, and {}",
                edge_padding_low.len(),
                edge_padding_high.len(),
                interior_padding.len(),
            ))
            .into());
        }
        Ok(Self { edge_padding_low, edge_padding_high, interior_padding })
    }

    /// Returns the padding added before the first element of each input axis.
    #[inline]
    pub fn edge_padding_low(&self) -> &[i64] {
        self.edge_padding_low.as_slice()
    }

    /// Returns the padding added after the last element of each input axis.
    #[inline]
    pub fn edge_padding_high(&self) -> &[i64] {
        self.edge_padding_high.as_slice()
    }

    /// Returns the padding added between any two adjacent elements of each input axis.
    #[inline]
    pub fn interior_padding(&self) -> &[usize] {
        self.interior_padding.as_slice()
    }
}

impl Display for PadOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for PadOperation {
    #[inline]
    fn name(&self) -> &'static str {
        PAD_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        match input_types[0].pad(
            &input_types[1],
            self.edge_padding_low.as_slice(),
            self.edge_padding_high.as_slice(),
            self.interior_padding.as_slice(),
        ) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("edge_padding_low", format_args!("{:?}", self.edge_padding_low))?;
            operation.field("edge_padding_high", format_args!("{:?}", self.edge_padding_high))?;
            operation.field("interior_padding", format_args!("{:?}", self.interior_padding))
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Pad>> InterpretableOperation<C> for PadOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].pad(
            &inputs[1],
            self.edge_padding_low.as_slice(),
            self.edge_padding_high.as_slice(),
            self.interior_padding.as_slice(),
        )?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for PadOperation where
    C::Operation: From<PadOperation>
{
}

/// Forward-mode rule for [`PadOperation`]: `pad` is linear in both the operand and the padding value, so the
/// tangent pads the operand tangent with the padding-value tangent using the same padding amounts.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for PadOperation
where
    C::Operation: From<PadOperation>,
    C::Value: Pad,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let primal = inputs[0].primal().pad(
            inputs[1].primal(),
            self.edge_padding_low(),
            self.edge_padding_high(),
            self.interior_padding(),
        )?;
        // The pad needs both the operand and padding-value tangents as real values, so materialize the structurally
        // zero side (the shared all-zero fast path already handled the case where both are zero).
        let operand_tangent = inputs[0].tangent().clone().materialize(context)?;
        let padding_tangent = inputs[1].tangent().clone().materialize(context)?;
        let tangent = operand_tangent.pad(
            &padding_tangent,
            self.edge_padding_low(),
            self.edge_padding_high(),
            self.interior_padding(),
        )?;
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Transpose (vector-Jacobian product) for a [`PadOperation`].
///
/// The forward map `(t, p) ↦ pad(t, p, low, high, interior)` writes input element `i` to output position
/// `low + i * (interior + 1)` along each axis and the padding value everywhere else, so its pullback splits the
/// output cotangent into two contributions:
///
///   - **Input cotangent**: edge-unpad the output cotangent and take a full-extent slice with stride
///     `interior + 1`, recovering both cropped and dilated input positions.
///   - **Padding-value cotangent**: pad an all-false input-shaped mask with `true`, select the output cotangent only
///     at those padding positions, and sum the selected tensor. Selection rather than subtraction keeps non-finite
///     cotangents at input positions from contaminating this contribution.
///
/// Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for PadOperation
where
    O: Operation<ArrayType>
        + From<OneOperation<ArrayType>>
        + From<PadOperation>
        + From<SelectOperation>
        + From<SliceOperation>
        + From<ReduceOperation>
        + From<ZeroOperation<ArrayType>>,
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
                let input_cotangent = if inputs[0].is_unknown() {
                    let inverse_edge_padding_low = self
                        .edge_padding_low()
                        .iter()
                        .enumerate()
                        .map(|(axis, padding)| {
                            padding.checked_neg().ok_or_else(|| TypeError::invalid(format!(
                                    "'pad' transpose cannot negate edge_padding_low at axis {axis} with value {padding}",
                                )))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let inverse_edge_padding_high = self
                        .edge_padding_high()
                        .iter()
                        .enumerate()
                        .map(|(axis, padding)| {
                            padding.checked_neg().ok_or_else(|| {
                                TypeError::invalid(format!(
                                    "'pad' transpose cannot negate edge_padding_high at axis {axis} with value \
                                     {padding}",
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let zero_type = dependency_scalar_type(cotangent.r#type().as_ref())?;
                    let zero = MaybeZero::Zero(zero_type).materialize(context)?;
                    let mut unpadded = context.stage_operation(
                        PadOperation::new(
                            inverse_edge_padding_low,
                            inverse_edge_padding_high,
                            vec![0; self.interior_padding().len()],
                        )?,
                        Vec::new(),
                        &[cotangent.clone(), zero],
                    )?;
                    check_count!("output", unpadded, 1, ProgramError);
                    let unpadded = unpadded.remove(0);
                    let strides = self
                        .interior_padding()
                        .iter()
                        .enumerate()
                        .map(|(axis, padding)| {
                            padding.checked_add(1).ok_or_else(|| {
                                TypeError::invalid(format!("'pad' transpose stride overflows usize on axis {axis}"))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let rank = strides.len();
                    let slice = SliceOperation::new(vec![0; rank], vec![0; rank])
                        .with_strides(strides)?
                        .with_full_extent_axes(0..rank)?;
                    let mut sliced = context.stage_operation(slice, Vec::new(), std::slice::from_ref(&unpadded))?;
                    check_count!("output", sliced, 1, ProgramError);
                    MaybeZero::Value(sliced.remove(0).unalign_cotangent(&inputs[0].r#type().cotangent())?)
                } else {
                    MaybeZero::Zero(inputs[0].r#type().cotangent())
                };
                let padding_value_cotangent = if inputs[1].is_unknown() {
                    let mask_input_type =
                        inputs[0].r#type().cotangent().with_data_type(DataType::Boolean).with_layout(None);
                    let mask_padding_type =
                        inputs[1].r#type().cotangent().with_data_type(DataType::Boolean).with_layout(None);
                    let mask_input = MaybeZero::Zero(mask_input_type).materialize(context)?;
                    let no_inputs: [Tracer<TracingContext<V, O>>; 0] = [];
                    let mut mask_padding =
                        context.stage_operation(OneOperation::new(mask_padding_type), Vec::new(), &no_inputs)?;
                    check_count!("output", mask_padding, 1, ProgramError);
                    let mut mask =
                        context.stage_operation(self.clone(), Vec::new(), &[mask_input, mask_padding.remove(0)])?;
                    check_count!("output", mask, 1, ProgramError);
                    let zero = MaybeZero::Zero(cotangent.r#type().into_owned()).materialize(context)?;
                    let mut selected = context.stage_operation(
                        SelectOperation,
                        Vec::new(),
                        &[mask.remove(0), cotangent.clone(), zero],
                    )?;
                    check_count!("output", selected, 1, ProgramError);
                    let all_axes = (0..cotangent.r#type().rank()).collect::<Vec<_>>();
                    let mut reduced = context.stage_operation(
                        ReduceOperation::new(all_axes, ReductionKind::Sum),
                        Vec::new(),
                        &[selected.remove(0)],
                    )?;
                    check_count!("output", reduced, 1, ProgramError);
                    MaybeZero::Value(reduced.remove(0).unalign_cotangent(&inputs[1].r#type().cotangent())?)
                } else {
                    MaybeZero::Zero(inputs[1].r#type().cotangent())
                };
                Ok(vec![input_cotangent, padding_value_cotangent])
            }
        }
    }
}

/// Batching rule for [`PadOperation`].
///
/// A batched input with a replicated padding value keeps its batch axis by padding it with zero amounts: the
/// lifted operation inserts `0` into all three padding vectors at the batch axis position. A batch-varying (batched)
/// padding value is vectorized with a constant-size mask construction: pad the operand with zero, pad an all-true
/// input mask with false, broadcast the per-item padding values over the padded result, and select those values at
/// padding positions.
impl<C> BatchableOperation<C> for PadOperation
where
    C: Context<Type = ArrayType> + One<C::Value> + Zero<C::Value>,
    C::Value: Broadcast + Pad + Select + Transpose,
    PadOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let axis_size = ArrayBatch::common_batch_size(inputs)?;
        if inputs[1].batch_axis_position().is_none() {
            let Some(batch_axis) = inputs[0].batch_axis_position() else {
                return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
            };
            let mut edge_padding_low = self.edge_padding_low().to_vec();
            edge_padding_low.insert(batch_axis, 0);
            let mut edge_padding_high = self.edge_padding_high().to_vec();
            edge_padding_high.insert(batch_axis, 0);
            let mut interior_padding = self.interior_padding().to_vec();
            interior_padding.insert(batch_axis, 0);
            let lifted = PadOperation::new(edge_padding_low, edge_padding_high, interior_padding)?;
            return lifted.interpret_with_batch_axes(context, inputs, &[BatchAxis::from_position(batch_axis)]);
        }
        let axis_size = axis_size.expect("a mapped input pins the batch size");
        let batch_axis = inputs[0].batch_axis_position().unwrap_or(0);
        let operand = inputs[0].match_axis(batch_axis, axis_size, context.axis_sharding().clone())?;
        let mut edge_padding_low = self.edge_padding_low().to_vec();
        edge_padding_low.insert(batch_axis, 0);
        let mut edge_padding_high = self.edge_padding_high().to_vec();
        edge_padding_high.insert(batch_axis, 0);
        let mut interior_padding = self.interior_padding().to_vec();
        interior_padding.insert(batch_axis, 0);

        let padding_type = inputs[1].unbatched_type();
        let zero_padding = context.parent().zero(&padding_type)?;
        let padded = operand.value().pad(
            &zero_padding,
            edge_padding_low.as_slice(),
            edge_padding_high.as_slice(),
            interior_padding.as_slice(),
        )?;
        let mask_input_type = operand.r#type().into_owned().with_data_type(DataType::Boolean).with_layout(None);
        let mask_input = context.parent().one(&mask_input_type)?;
        let mask_padding_type = padding_type.with_data_type(DataType::Boolean).with_layout(None);
        let mask_padding = context.parent().zero(&mask_padding_type)?;
        let mask = mask_input.pad(
            &mask_padding,
            edge_padding_low.as_slice(),
            edge_padding_high.as_slice(),
            interior_padding.as_slice(),
        )?;
        let broadcasted_padding = inputs[1].value().broadcast(padded.r#type().into_owned(), &[batch_axis])?;
        let output = C::Value::select(&mask, &padded, &broadcasted_padding)?;
        let output_type = output.r#type().into_owned();
        Ok(vec![ArrayBatch::new(output_type, output, BatchAxis::from_position(batch_axis))?])
    }
}

/// Represents the ability to resize an array by adding edge and interior padding filled with a scalar padding value,
/// with the semantics of StableHLO's [`pad`](https://openxla.org/stablehlo/spec#pad) operation. Negative edge padding
/// crops the dilated input, while interior padding remains non-negative.
///
/// `t.pad(padding_value, edge_padding_low, edge_padding_high, interior_padding)` returns an array that holds the
/// input element with index `i` at output index `edge_padding_low + i * (interior_padding + 1)` along each axis and
/// `padding_value` everywhere else. The output dimension along an axis whose input dimension is `d` is:
///
///   - `edge_padding_low + edge_padding_high` when `d == 0` (there are no elements, so no interior padding is
///     inserted and the output holds only the edge padding), and
///   - `edge_padding_low + (d - 1) * (interior_padding + 1) + 1 + edge_padding_high` otherwise (`d` elements with
///     `interior_padding` padding elements between each adjacent pair).
///
/// All three padding slices must have length equal to the input rank, and the padding value must be a rank-0 scalar
/// with the input's data type in the same memory space. Dynamic extents remain dynamic. A bounded dynamic output keeps
/// a transformed bound when the largest permitted input extent has a non-negative representable result; smaller
/// runtime extents must still satisfy the padding geometry. An effective identity passes its input through unchanged.
/// Any non-identity output keeps the input memory space, clears explicit physical layout metadata, and preserves
/// compatible sharding and distributed dependency state.
///
/// [`Pad`] is the transpose dual of strided [`Slice`]: slicing with stride
/// `s` keeps every `s`-th element, while padding with `interior_padding = s - 1` puts elements back at every `s`-th
/// position.
///
/// # Example
///
/// The following example shows how to use [`Pad`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::Pad;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::arrays::Array;
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Pad [1, 2, 3] with one leading zero, two trailing zeros, and one zero between adjacent elements. With
/// // d = 3, low = 1, high = 2, and interior = 1, the output dimension is 1 + (3 - 1) * 2 + 1 + 2 = 8 and the
/// // input elements land at output positions 1, 3, and 5.
/// let x = Array::vector(vec![1.0, 2.0, 3.0]);
/// let y = x.pad(&Array::scalar(0.0), &[1], &[2], &[1])?;
/// assert_eq!(y.to_f64s(), vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0]);
/// # Ok(())
/// # }
/// ```
pub trait Pad: Sized {
    /// Pads `self` with `padding_value` using the provided edge and interior padding amounts. Refer to the
    /// documentation of this trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `padding_value`: Rank-0 scalar with the input's data type, written into every padding position.
    ///   - `edge_padding_low`: Padding added before the first element of each input axis.
    ///   - `edge_padding_high`: Padding added after the last element of each input axis.
    ///   - `interior_padding`: Padding added between any two adjacent elements of each input axis.
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError>;
}

/// Returns whether this padding geometry leaves every possible element and its position unchanged.
fn is_effective_identity(
    input_type: &ArrayType,
    edge_padding_low: &[i64],
    edge_padding_high: &[i64],
    interior_padding: &[usize],
) -> bool {
    edge_padding_low.iter().all(|padding| *padding == 0)
        && edge_padding_high.iter().all(|padding| *padding == 0)
        && input_type.shape().dimensions().iter().zip(interior_padding).all(|(dimension, padding)| {
            *padding == 0
                || matches!(dimension, Dimension::Static(0 | 1))
                || matches!(
                    dimension,
                    Dimension::Dynamic(variable) if variable.bounds().upper().is_some_and(|upper| upper <= 2)
                )
        })
}

/// Constructs a scalar type on the same mesh with the same non-dimensional dependency metadata as `source`.
fn dependency_scalar_type(source: &ArrayType) -> Result<ArrayType, TypeError> {
    let sharding = source
        .sharding()
        .map(|sharding| {
            Sharding::replicated(sharding.mesh().clone(), 0)
                .with_unreduced_axes(sharding.unreduced_axes().clone())
                .and_then(|output| output.with_reduced_axes(sharding.reduced_axes().clone()))
                .and_then(|output| output.with_varying_manual_axes(sharding.varying_manual_axes().clone()))
                .map_err(|error| {
                    TypeError::invalid(format!("'pad' dependency scalar sharding construction failed: {error}"))
                })
        })
        .transpose()?;
    ArrayType::scalar(source.data_type())
        .with_memory(source.memory())
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))
}

/// Computes one concrete padded extent in a wide signed representation.
fn padded_extent(
    input_size: usize,
    edge_padding_low: i64,
    edge_padding_high: i64,
    interior_padding: usize,
    axis: usize,
) -> Result<i128, TypeError> {
    let gap_count = input_size.saturating_sub(1);
    let input_size = i128::try_from(input_size)
        .map_err(|_| TypeError::invalid(format!("'pad' input size is too large on axis {axis}")))?;
    let gap_count = i128::try_from(gap_count)
        .map_err(|_| TypeError::invalid(format!("'pad' input size is too large on axis {axis}")))?;
    let interior_padding = i128::try_from(interior_padding)
        .map_err(|_| TypeError::invalid(format!("'pad' interior padding is too large on axis {axis}")))?;
    let dilated_size = input_size
        .checked_add(
            gap_count
                .checked_mul(interior_padding)
                .ok_or_else(|| TypeError::invalid(format!("'pad' output size overflows usize on axis {axis}")))?,
        )
        .and_then(|size| size.checked_add(i128::from(edge_padding_low)))
        .and_then(|size| size.checked_add(i128::from(edge_padding_high)))
        .ok_or_else(|| TypeError::invalid(format!("'pad' output size overflows usize on axis {axis}")))?;
    Ok(dilated_size)
}

/// Computes one concrete padded extent and validates that it is representable by [`Dimension::Static`].
fn static_padded_extent(
    input_size: usize,
    edge_padding_low: i64,
    edge_padding_high: i64,
    interior_padding: usize,
    axis: usize,
) -> Result<usize, TypeError> {
    let output_size = padded_extent(input_size, edge_padding_low, edge_padding_high, interior_padding, axis)?;
    if output_size < 0 {
        return Err(TypeError::invalid(format!("'pad' output size is negative ({output_size}) on axis {axis}")));
    }
    usize::try_from(output_size)
        .map_err(|_| TypeError::invalid(format!("'pad' output size overflows usize on axis {axis}")))
}

impl Pad for ArrayType {
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        if self.data_type() != padding_value.data_type() {
            return Err(TypeError::invalid(format!(
                "'pad' input data type {} does not match padding value data type {}",
                self.data_type(),
                padding_value.data_type(),
            ))
            .into());
        }
        if padding_value.rank() != 0 {
            return Err(TypeError::invalid(format!(
                "'pad' padding value must be a scalar but has type {padding_value}"
            ))
            .into());
        }
        if self.memory() != padding_value.memory() {
            return Err(TypeError::invalid(format!(
                "'pad' input and padding value must share one memory space but reside in {} and {}",
                self.memory(),
                padding_value.memory(),
            ))
            .into());
        }
        let rank = self.rank();
        for (name, length) in [
            ("edge_padding_low", edge_padding_low.len()),
            ("edge_padding_high", edge_padding_high.len()),
            ("interior_padding", interior_padding.len()),
        ] {
            if length != rank {
                return Err(
                    TypeError::invalid(format!("'pad' {name} has length {length} but input has rank {rank}")).into()
                );
            }
        }
        if is_effective_identity(self, edge_padding_low, edge_padding_high, interior_padding) {
            return Ok(self.clone());
        }
        let mut output_dimensions = Vec::with_capacity(rank);
        for axis in 0..rank {
            let dimension = self.dimension(axis);
            let output_dimension = match dimension {
                Dimension::Static(size) => Dimension::Static(static_padded_extent(
                    size,
                    edge_padding_low[axis],
                    edge_padding_high[axis],
                    interior_padding[axis],
                    axis,
                )?),
                Dimension::Dynamic(variable) => {
                    if let Some(upper) = variable.bounds().upper() {
                        let maximum_input_extent = upper - 1;
                        let maximum_output_extent = padded_extent(
                            maximum_input_extent,
                            edge_padding_low[axis],
                            edge_padding_high[axis],
                            interior_padding[axis],
                            axis,
                        )?;
                        if maximum_output_extent < 0 {
                            return Err(TypeError::invalid(format!(
                                "'pad' output size is negative ({maximum_output_extent}) on dynamic axis {axis} \
                                 even at its maximum input extent {maximum_input_extent}",
                            ))
                            .into());
                        }
                    }
                    return Err(TypeError::invalid(format!(
                        "'pad' dynamic axis {axis} requires an explicit result-dimension operand",
                    ))
                    .into());
                }
            };
            output_dimensions.push(output_dimension);
        }
        let padding_positions_may_exist = self.shape().dimensions().iter().enumerate().any(|(axis, dimension)| {
            edge_padding_low[axis] > 0
                || edge_padding_high[axis] > 0
                || (interior_padding[axis] > 0
                    && !matches!(dimension, Dimension::Static(0 | 1))
                    && !matches!(
                        dimension,
                        Dimension::Dynamic(variable)
                            if variable.bounds().upper().is_some_and(|upper| upper <= 2)
                    ))
        });
        let sharding = resized_output_sharding(self, &output_dimensions, PAD_OPERATION_NAME)?;
        if padding_positions_may_exist {
            if self.unreduced_axes() != padding_value.unreduced_axes()
                || self.reduced_axes() != padding_value.reduced_axes()
            {
                return Err(TypeError::invalid(format!(
                    "'pad' input and padding value must have matching reduced and unreduced mesh axes but got \
                         input type {self} and padding value type {padding_value}",
                ))
                .into());
            }
            let input_varying_manual_axes = self.sharding().map(|sharding| sharding.varying_manual_axes());
            let padding_varying_manual_axes = padding_value.sharding().map(|sharding| sharding.varying_manual_axes());
            if input_varying_manual_axes.cloned().unwrap_or_default()
                != padding_varying_manual_axes.cloned().unwrap_or_default()
            {
                return Err(TypeError::invalid(format!(
                    "'pad' input and padding value must have matching varying manual axes but got input type \
                         {self} and padding value type {padding_value}",
                ))
                .into());
            }
            let has_distributed_dependencies = !self.unreduced_axes().is_empty()
                || !self.reduced_axes().is_empty()
                || input_varying_manual_axes.is_some_and(|axes| !axes.is_empty());
            if has_distributed_dependencies
                && self.sharding().map(|sharding| sharding.mesh())
                    != padding_value.sharding().map(|sharding| sharding.mesh())
            {
                return Err(TypeError::invalid(
                    "'pad' input and padding value with distributed dependencies must use the same mesh".to_string(),
                )
                .into());
            }
        }
        ArrayType::new(self.data_type(), Shape::new(output_dimensions))
            .with_memory(self.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError::invalid(error.to_string()).into())
    }
}

/// Any context-carrying value pads by binding a [`PadOperation`] through its own context. The `From<PadOperation>`
/// bound makes this disjoint from the eager value types (whose context operation is `ConstantOperation`), so it covers
/// the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Pad for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<PadOperation>,
{
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        self.r#type()
            .pad(padding_value.r#type().as_ref(), edge_padding_low, edge_padding_high, interior_padding)?;
        if is_effective_identity(self.r#type().as_ref(), edge_padding_low, edge_padding_high, interior_padding) {
            return Ok(self.clone());
        }
        let mut outputs = self.dispatch_domain().bind(
            PadOperation::new(edge_padding_low.to_vec(), edge_padding_high.to_vec(), interior_padding.to_vec())?,
            Vec::new(),
            &[self.clone(), padding_value.clone()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::{BatchAxis, BatchingContext};
    use crate::contexts::EagerContext;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::Typed;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::dimensions::{DimensionBounds, DimensionVariable};
    use crate::types::{DataType, Layout, Memory, StridedLayout};

    use super::*;

    #[test]
    fn test_pad() {
        let operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap();

        // Operation identity and accessors.
        assert_eq!(operation.name(), PAD_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]]");
        assert_eq!(operation.edge_padding_low(), &[1]);
        assert_eq!(operation.edge_padding_high(), &[2]);
        assert_eq!(operation.interior_padding(), &[1]);

        // Type inference validates the padding geometry and returns the padded type, and the type-level (abstract)
        // capability backs it without consuming the borrowed input type. With d = 3, low = 1, high = 2, and
        // interior = 1, the output dimension is 1 + (3 - 1) * 2 + 1 + 2 = 8.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let padding_value_type = ArrayType::scalar(DataType::F64);
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(8)]));
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input_type.clone(), padding_value_type.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [input_type.clone()],
                    error = "expected 2 inputs but got 1",
                },
                {
                    input_types = [input_type.clone(), ArrayType::scalar(DataType::F32)],
                    error = "'pad' input data type f64 does not match padding value data type f32",
                },
                {
                    input_types = [input_type.clone(), input_type.clone()],
                    error = "'pad' padding value must be a scalar but has type f64[3]",
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                            "input",
                            DimensionBounds::unbounded(),
                        ))])),
                        padding_value_type.clone(),
                    ],
                    error = "'pad' dynamic axis 0 requires an explicit result-dimension operand",
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                            "input",
                            DimensionBounds::non_negative(Some(4)).unwrap(),
                        ))])),
                        padding_value_type.clone(),
                    ],
                    error = "'pad' dynamic axis 0 requires an explicit result-dimension operand",
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(usize::MAX)])),
                        padding_value_type.clone(),
                    ],
                    error = "'pad' output size overflows usize on axis 0",
                },
            ],
        );
        assert_eq!(input_type.pad(&padding_value_type, &[1], &[2], &[1]), Ok(output_type.clone()));
        assert_eq!(
            input_type.pad(&padding_value_type, &[], &[0], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "'pad' edge_padding_low has length 0 but input has rank 1".to_string()
            ))),
        );
        assert_eq!(
            input_type.pad(&padding_value_type, &[0], &[], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "'pad' edge_padding_high has length 0 but input has rank 1".to_string()
            ))),
        );
        assert_eq!(
            input_type.pad(&padding_value_type, &[0], &[0], &[]),
            Err(ProgramError::Type(TypeError::invalid(
                "'pad' interior_padding has length 0 but input has rank 1".to_string()
            ))),
        );
        // Negative inverse edges still validate their abstract extent. A valid derived dynamic extent requires the
        // explicit result-dimension operand introduced by the mixed operation signature, while an always-negative
        // extent is rejected immediately.
        assert_eq!(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                    "input",
                    DimensionBounds::non_negative(Some(9)).unwrap(),
                ))]),
            )
            .pad(&padding_value_type, &[-1], &[-2], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "'pad' dynamic axis 0 requires an explicit result-dimension operand".to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("input", DimensionBounds::unbounded(),))]),
            )
            .pad(&padding_value_type, &[-1], &[-2], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "'pad' dynamic axis 0 requires an explicit result-dimension operand".to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(crate::types::dimensions::DimensionVariable::new(
                    "dynamic",
                    crate::types::dimensions::DimensionBounds::non_negative(Some(2)).unwrap()
                ))])
            )
            .pad(&padding_value_type, &[-5], &[0], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "'pad' output size is negative (-4) on dynamic axis 0 even at its maximum input extent 1".to_string()
            ))),
        );

        // Interpretation writes the input elements at `low + i * (interior + 1)` (positions 1, 3, and 5) and fills
        // every other position with the padding value.
        let input = Array::vector(vec![1.0, 2.0, 3.0]);
        let output = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[input, Array::scalar(9.0)])
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0]);

        // Empty input axes hold only the edge padding (the `d == 0` case skips interior padding entirely) and
        // rank-0 inputs pass through unchanged.
        let empty_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)]));
        assert_eq!(
            empty_type.pad(&padding_value_type, &[1], &[2], &[1]),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))),
        );
        let empty = Array::from_f64s(empty_type, vec![]).pad(&Array::scalar(7.0), &[1], &[2], &[1]).unwrap();
        assert_eq!(empty.to_f64s(), vec![7.0, 7.0, 7.0]);
        let scalar = Array::scalar(42.0).pad(&Array::scalar(7.0), &[], &[], &[]).unwrap();
        assert_eq!(scalar.to_f64s(), vec![42.0]);

        // Invalid construction and inputs report precise operation and interpreter errors.
        assert_eq!(
            PadOperation::new(vec![1], vec![2, 0], vec![1]),
            Err(ProgramError::Type(TypeError::invalid(
                "'pad' expects edge_padding_low, edge_padding_high, and interior_padding to share one length \
                    but got lengths 1, 2, and 1"
                    .to_string()
            ))),
        );
        assert_eq!(
            PadOperation::new(vec![1, 0], vec![2, 0], vec![1, 0])
                .unwrap()
                .infer_output_types(&[input_type.clone(), padding_value_type.clone()], &[]),
            Err(TypeError::invalid("'pad' edge_padding_low has length 2 but input has rank 1".to_string())),
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

        // Program rendering uses the canonical operation name and includes all three padding vectors.
        let mut builder = ProgramBuilder::<Array, PadOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_padding_value = builder.add_input(padding_value_type);
        let program_output =
            builder.add_instruction(operation, Vec::new(), vec![program_input, program_padding_value]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![program_output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[8] = pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Check standard partial evaluation with known and residual operands.
        let input = Array::vector(vec![1.0, 2.0, 3.0]);
        let padding_value = Array::scalar(9.0);
        let expected = Array::vector(vec![9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0]);
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, padding_value.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, padding_value.clone()),
                    ],
                    outputs = [(@residual, expected.clone())],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, input.clone()),
                        (@unknown(type = padding_value.r#type().into_owned(), replay = padding_value.clone())),
                    ],
                    outputs = [(@residual, expected.clone())],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@unknown(type = padding_value.r#type().into_owned(), replay = padding_value.clone())),
                    ],
                    outputs = [(@residual, expected.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        // Batching inserts zero padding on the mapped axis and vectorizes a mapped padding value.
        check_operation_batching!(
            @exact,
            operation = PadOperation::new(vec![1], vec![0], vec![0]).unwrap(),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])),
                        (@replicated, Array::scalar(0.0)),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        3,
                        vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0],
                    ))],
                },
                {
                    inputs = [
                        (@replicated, Array::vector(vec![1.0, 2.0])),
                        (@replicated, Array::scalar(0.0)),
                    ],
                    outputs = [(@replicated, Array::vector(vec![0.0, 1.0, 2.0]))],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])),
                        (@mapped(axis = 0), Array::vector(vec![8.0, 9.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        3,
                        vec![8.0, 1.0, 2.0, 9.0, 3.0, 4.0],
                    ))],
                },
                {
                    inputs = [
                        (@replicated, Array::vector(vec![1.0, 2.0])),
                        (@mapped(axis = 0), Array::vector(vec![8.0, 9.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        3,
                        vec![8.0, 1.0, 2.0, 9.0, 1.0, 2.0],
                    ))],
                },
                {
                    inputs = [
                        (@mapped(axis = 1), Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])),
                        (@mapped(axis = 0), Array::vector(vec![8.0, 9.0])),
                    ],
                    outputs = [(@mapped(axis = 1), Array::matrix(
                        3,
                        2,
                        vec![8.0, 9.0, 1.0, 2.0, 3.0, 4.0],
                    ))],
                },
            ],
        );

        // Pad is linear in both inputs: its JVP pads tangent values and its pullback separates written and padding
        // positions.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0, 3.0]), Array::scalar(9.0)],
                tangents = [Array::vector(vec![0.1, 0.2, 0.3]), Array::scalar(0.5)],
                primal_outputs = [Array::vector(vec![9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0])],
                tangent_outputs = [Array::vector(vec![0.5, 0.1, 0.5, 0.2, 0.5, 0.3, 0.5, 0.5])],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()])))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])],
                    input_cotangents = [Array::vector(vec![2.0, 4.0, 6.0]), Array::scalar(24.0)],
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()])))),
                        (@known, Array::scalar(9.0)),
                    ],
                    output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])],
                    input_cotangents = [Array::vector(vec![2.0, 4.0, 6.0])],
                },
                {
                    inputs = [
                        (@known, Array::vector(vec![1.0, 2.0, 3.0])),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])],
                    input_cotangents = [Array::scalar(24.0)],
                },
            ],
        );
        // Selecting padding positions before reduction avoids the `infinity - infinity` contamination that a
        // total-sum-minus-input-sum formulation would introduce.
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![1], vec![0], vec![0]).unwrap(),
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()])))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::vector(vec![5.0, f64::INFINITY, 7.0])],
                    input_cotangents = [Array::vector(vec![f64::INFINITY, 7.0]), Array::scalar(5.0)],
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()])))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::vector(vec![3.0, 1e20, -1e20])],
                    input_cotangents = [Array::vector(vec![1e20, -1e20]), Array::scalar(3.0)],
                },
            ],
        );
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![-1], vec![1], vec![0]).unwrap(),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()])))),
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                ],
                output_cotangents = [Array::vector(vec![2.0, 3.0, 5.0])],
                input_cotangents = [Array::vector(vec![0.0, 2.0, 3.0]), Array::scalar(5.0)],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![0.into()])))),
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                ],
                output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0])],
                input_cotangents = [Array::vector(Vec::<f64>::new()), Array::scalar(6.0)],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(Vec::new(), Vec::new(), Vec::new()).unwrap(),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                ],
                output_cotangents = [Array::scalar(f64::INFINITY)],
                input_cotangents = [Array::scalar(f64::INFINITY), Array::scalar(0.0)],
            }],
        );

        // Until the mixed pad signature lands, positive padding over a bounded-dynamic operand cannot manufacture a
        // stable result identity. P3 restores this pullback path by supplying its derived extents as explicit
        // dimension operands.
        let dynamic_input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "input",
                DimensionBounds::non_negative(Some(4)).unwrap(),
            ))]),
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let dynamic_input = builder.add_input(dynamic_input_type);
        let dynamic_padding = builder.add_input(ArrayType::scalar(DataType::F64));
        assert_eq!(
            builder.add_instruction(
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![dynamic_input, dynamic_padding],
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "'pad' dynamic axis 0 requires an explicit result-dimension operand".to_string(),
            ))),
        );

        // A pure crop never reads the padding scalar, so its dependency metadata may differ from the operand's. The
        // inverse pad nevertheless introduces zeros for cropped input positions and must derive that internal zero's
        // dependencies from the operand cotangent rather than from the unused primal padding scalar.
        let crop_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let crop_input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(Sharding::replicated(crop_mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let crop_padding_type =
            ArrayType::scalar(DataType::F64).with_sharding(Sharding::replicated(crop_mesh, 0)).unwrap();
        let crop_output_type = crop_input_type.pad(&crop_padding_type, &[-1], &[0], &[0]).unwrap();
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![-1], vec![0], vec![0]).unwrap(),
            cases = [{
                inputs = [
                    (@linear(type = crop_input_type.clone())),
                    (@known, Array::from_f64s(crop_padding_type, vec![9.0])),
                ],
                output_cotangents = [Array::from_f64s(crop_output_type, vec![2.0, 3.0])],
                input_cotangents = [Array::from_f64s(crop_input_type, vec![0.0, 2.0, 3.0])],
            }],
        );

        // The pullback restores the complete cotangent types of both operands after slicing and reducing the output
        // cotangent.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        let padding_type = ArrayType::scalar(DataType::F64)
            .with_layout(Layout::Strided(StridedLayout::new(Vec::new())))
            .with_memory(Memory::Host { pinned: true });
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![8.into()])).with_memory(Memory::Host { pinned: true });
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [{
                inputs = [(@linear(type = input_type.clone())), (@linear(type = padding_type.clone()))],
                output_cotangents = [Array::from_f64s(
                    output_type,
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                )],
                input_cotangents = [
                    Array::from_f64s(input_type, vec![2.0, 4.0, 6.0]),
                    Array::from_f64s(padding_type, vec![24.0]),
                ],
            }],
        );
    }

    #[test]
    fn test_array_pad() {
        // A rank-2 pad exercises the odometer across axes with different padding amounts: rows gain one interior
        // row and columns gain asymmetric edge padding.
        let input = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let output = input.pad(&Array::scalar(0.0), &[0, 1], &[1, 0], &[1, 0]).unwrap();
        assert_eq!(
            *output.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4), Dimension::Static(3)])),
        );
        assert_eq!(output.to_f64s(), vec![0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0],);

        // Signed edge padding crops the dilated operand. Cropping can be asymmetric, can combine with interior
        // dilation, and must not be elided merely because the output shape happens to equal the input shape.
        assert_eq!(
            Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0])
                .pad(&Array::scalar(0.0), &[-1], &[-2], &[0])
                .unwrap()
                .to_f64s(),
            vec![2.0, 3.0],
        );
        assert_eq!(
            Array::vector(vec![1.0, 2.0, 3.0]).pad(&Array::scalar(9.0), &[-1], &[1], &[1]).unwrap().to_f64s(),
            vec![9.0, 2.0, 9.0, 3.0, 9.0],
        );
        assert_eq!(
            Array::vector(vec![1.0, 2.0, 3.0]).pad(&Array::scalar(9.0), &[-1], &[1], &[0]).unwrap().to_f64s(),
            vec![2.0, 3.0, 9.0],
        );
        assert_eq!(
            Array::vector(vec![1.0]).pad(&Array::scalar(0.0), &[-2], &[0], &[0]),
            Err(ProgramError::Type(TypeError::invalid("'pad' output size is negative (-1) on axis 0".to_string()))),
        );

        // Interior padding is an effective identity on singleton axes. The eager and abstract fast paths preserve
        // the complete type and avoid overflowing `interior + 1` for a value that can never be used as a stride.
        let singleton_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![7])));
        let singleton = Array::from_f64s(singleton_type.clone(), vec![3.0]);
        let identity = singleton.pad(&Array::scalar(0.0), &[0], &[0], &[usize::MAX]).unwrap();
        assert_eq!(*identity.r#type(), singleton_type);
        assert_eq!(identity.to_f64s(), vec![3.0]);

        // The kernel validates the padding value shape eagerly.
        assert_eq!(
            Array::vector(vec![1.0, 2.0]).pad(&Array::vector(vec![0.0]), &[0], &[0], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "'pad' padding value must be a scalar but has type f64[1]".to_string()
            ))),
        );
    }

    #[test]
    fn test_array_type_pad() {
        use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        // [4] sharded over `x` and unreduced over the manual axis `m`.
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_unreduced_axes(["m"])
            .unwrap();
        let input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let pad_value = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh, 0).with_unreduced_axes(["m"]).unwrap())
            .unwrap();

        // Padding preserves a common memory placement and rejects a padding scalar that would require an implicit
        // transfer.
        let host_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_memory(Memory::Host { pinned: true });
        let host_padding = ArrayType::scalar(DataType::F32).with_memory(Memory::Host { pinned: true });
        assert_eq!(host_input.pad(&host_padding, &[0], &[1], &[0]).unwrap().memory(), Memory::Host { pinned: true },);
        assert_eq!(
            host_input.pad(&pad_value, &[0], &[1], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "'pad' input and padding value must share one memory space but reside in Host[Pinned] and \
                          Device"
                    .to_string()
            ))),
        );
        let laid_out_input = host_input.with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(laid_out_input.pad(&host_padding, &[0], &[0], &[0]), Ok(laid_out_input.clone()));

        // Padding to an evenly divisible size keeps the operand sharding (including the unreduced manual axis): with
        // low = 0, interior = 0, and high = 4 the output is 0 + 4 + 4 = 8, divisible by the `x` mesh-axis size (2).
        assert_eq!(input.pad(&pad_value, &[0], &[4], &[0]).unwrap().sharding(), Some(&sharding));
        // Padding to a size not divisible by the explicit mesh-axis size (output 0 + 4 + 1 = 5) is rejected.
        assert!(input.pad(&pad_value, &[0], &[1], &[0]).is_err());

        // JAX requires exact dependency metadata whenever the padding value can contribute. Neither reduced axes nor
        // varying manual axes are implicitly unioned from the scalar.
        let plain_padding = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(sharding.mesh().clone(), 0))
            .unwrap();
        assert!(input.pad(&plain_padding, &[0], &[4], &[0]).is_err());
        let reduced_padding = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(sharding.mesh().clone(), 0).with_reduced_axes(["m"]).unwrap())
            .unwrap();
        assert!(input.pad(&reduced_padding, &[0], &[4], &[0]).is_err());
        let varying_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(sharding.mesh().clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["m"])
                    .unwrap(),
            )
            .unwrap();
        assert!(varying_input.pad(&pad_value, &[0], &[4], &[0]).is_err());

        // Mesh identity is irrelevant for an ordinary scalar when neither side carries VMA or reduction metadata;
        // the result placement is derived solely from the operand. Effective identities do not consult the unused
        // padding value's dependency metadata at all.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("other", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let ordinary_input_sharding =
            Sharding::new(sharding.mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let ordinary_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(ordinary_input_sharding.clone())
            .unwrap();
        let other_mesh_padding =
            ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(other_mesh, 0)).unwrap();
        assert_eq!(
            ordinary_input.pad(&other_mesh_padding, &[0], &[4], &[0]).unwrap().sharding(),
            Some(&ordinary_input_sharding),
        );
        assert_eq!(varying_input.pad(&plain_padding, &[0], &[0], &[0]), Ok(varying_input));
    }

    #[test]
    fn test_pad_batching() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let physical_sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            let input_type =
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
                    .with_sharding(physical_sharding)
                    .unwrap();
            let input = ArrayBatch::new(
                input_type.clone(),
                Array::from_f64s(input_type, vec![1.0, 2.0, 3.0, 4.0]),
                BatchAxis::new(0),
            )
            .unwrap();
            let padding_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])
                        .unwrap()
                        .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                        .unwrap(),
                )
                .unwrap();
            let padding = ArrayBatch::new(
                padding_type.clone(),
                Array::from_f64s(padding_type, vec![8.0, 9.0]),
                BatchAxis::new(0),
            )
            .unwrap();
            let context = BatchingContext::new(EagerContext::<Array>::new(), 2)
                .with_axis_sharding(ShardingDimension::sharded(["x"]));

            let outputs = PadOperation::new(vec![1], vec![0], vec![0])
                .unwrap()
                .batch(&context, &crate::EmptyRegionDriver, &[input, padding])
                .unwrap();

            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(
                outputs[0].r#type().sharding().unwrap().dimensions(),
                &[ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            );
            assert_eq!(outputs[0].value().to_f64s(), vec![8.0, 1.0, 2.0, 9.0, 3.0, 4.0]);
        }

        // The vectorized mapped-padding rule handles an empty batch without inventing values or dropping placement.
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let physical_sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            let input_type =
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0), Dimension::Static(2)]))
                    .with_sharding(physical_sharding.clone())
                    .unwrap();
            let input =
                ArrayBatch::new(input_type.clone(), Array::from_f64s(input_type, Vec::new()), BatchAxis::new(0))
                    .unwrap();
            let padding_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)]))
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])
                        .unwrap()
                        .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                        .unwrap(),
                )
                .unwrap();
            let padding =
                ArrayBatch::new(padding_type.clone(), Array::from_f64s(padding_type, Vec::new()), BatchAxis::new(0))
                    .unwrap();
            let context = BatchingContext::new(EagerContext::<Array>::new(), 0)
                .with_axis_sharding(ShardingDimension::sharded(["x"]));

            let outputs = PadOperation::new(vec![1], vec![0], vec![0])
                .unwrap()
                .batch(&context, &crate::EmptyRegionDriver, &[input, padding])
                .unwrap();

            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type().sharding().unwrap().dimensions(), physical_sharding.dimensions(),);
            assert_eq!(outputs[0].r#type().shape().dimensions(), &[Dimension::Static(0), Dimension::Static(3)]);
            assert!(outputs[0].value().values().is_empty());
        }
    }
}
