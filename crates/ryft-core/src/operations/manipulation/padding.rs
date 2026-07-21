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
use crate::operations::constants::{Zero, ZeroOperation};
use crate::operations::manipulation::{Broadcast, Reshape, Slice, SliceOperation, Transpose, UpdateSlice};
use crate::operations::math::{ReduceOperation, ReductionKind, SubOperation};
use crate::operations::sharding::Reshard;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::programs::{MaybeZero, ProgramError};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, Shape, Size};

use super::slicing::{batch_by_item_expansion, resized_output_sharding};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`PadOperation`].
pub const PAD_OPERATION_NAME: &str = "pad";

/// [`Operation`] that expands its first operand by adding edge and interior padding filled with its second (scalar)
/// operand. Refer to the documentation of [`Pad`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PadOperation {
    /// Padding added before the first element of each input axis.
    edge_padding_low: Vec<usize>,

    /// Padding added after the last element of each input axis.
    edge_padding_high: Vec<usize>,

    /// Padding added between any two adjacent elements of each input axis.
    interior_padding: Vec<usize>,
}

impl PadOperation {
    /// Creates a new [`PadOperation`] with the provided edge and interior padding amounts. The three vectors must
    /// share one length (one entry per input axis); whether that shared length matches the input rank is validated
    /// during type inference, once an input type is known.
    pub fn new(
        edge_padding_low: Vec<usize>,
        edge_padding_high: Vec<usize>,
        interior_padding: Vec<usize>,
    ) -> Result<Self, ProgramError> {
        if edge_padding_low.len() != edge_padding_high.len() || edge_padding_low.len() != interior_padding.len() {
            return Err(TypeError {
                message: format!(
                    "'pad' expects edge_padding_low, edge_padding_high, and interior_padding to share one length but \
                    got lengths {}, {}, and {}",
                    edge_padding_low.len(),
                    edge_padding_high.len(),
                    interior_padding.len(),
                ),
            }
            .into());
        }
        Ok(Self { edge_padding_low, edge_padding_high, interior_padding })
    }

    /// Returns the padding added before the first element of each input axis.
    #[inline]
    pub fn edge_padding_low(&self) -> &[usize] {
        self.edge_padding_low.as_slice()
    }

    /// Returns the padding added after the last element of each input axis.
    #[inline]
    pub fn edge_padding_high(&self) -> &[usize] {
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
            Err(error) => Err(TypeError { message: error.to_string() }),
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
///   - **Input cotangent**: the strided slice of the cotangent at the pad geometry — `start = low`,
///     `stride = interior + 1`, and `limit = low + (d - 1) * (interior + 1) + 1` for input dimension `d > 0`
///     (`limit = low` for `d == 0`, an empty slice) — which reads back exactly the positions the forward map wrote
///     input elements to. For example, padding `d = 3` elements with `low = 1`, `high = 2`, and `interior = 1`
///     produces an output of dimension `1 + (3 - 1) * 2 + 1 + 2 = 8` whose positions `1`, `3`, and `5` hold the
///     input elements, and the pullback slices the cotangent with `start = 1`, `limit = 6`, and `stride = 2`,
///     reading positions `1`, `3`, and `5`.
///   - **Padding-value cotangent**: the sum of the cotangent over every *padding* position, computed as the full
///     sum of the cotangent minus the sum of the strided-slice region (two staged full reductions and a
///     subtraction, which avoids materializing a mask). When the input has no elements (some dimension is `0`),
///     the sliced region is empty and its sum is a staged scalar zero.
///
/// Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for PadOperation
where
    O: Operation<ArrayType>
        + From<SliceOperation>
        + From<ReduceOperation>
        + From<SubOperation>
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
                let input_type = inputs[0].r#type();
                let rank = input_type.rank();
                let mut start_indices = Vec::with_capacity(rank);
                let mut limit_indices = Vec::with_capacity(rank);
                let mut strides = Vec::with_capacity(rank);
                let mut input_is_empty = false;
                for axis in 0..rank {
                    let dimension = input_type.dimension(axis as isize);
                    let Some(input_size) = dimension.value() else {
                        return Err(TypeError {
                            message: format!(
                                "'pad' transpose requires a static input shape but axis {axis} has size {dimension}",
                            ),
                        }
                        .into());
                    };
                    let low = self.edge_padding_low()[axis];
                    let stride = self.interior_padding()[axis] + 1;
                    let limit = match input_size {
                        0 => low,
                        size => low + (size - 1) * stride + 1,
                    };
                    input_is_empty |= input_size == 0;
                    start_indices.push(low);
                    limit_indices.push(limit);
                    strides.push(stride);
                }
                let input_cotangents = context.stage_operation(
                    SliceOperation::new(start_indices, limit_indices).with_strides(strides)?,
                    Vec::new(),
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", input_cotangents, 1, ProgramError);
                let input_cotangent =
                    input_cotangents.into_iter().next().unwrap().unalign_cotangent(&inputs[0].r#type().cotangent())?;
                let all_axes: Vec<usize> = (0..cotangent.r#type().as_ref().rank()).collect();
                let total_sums = context.stage_operation(
                    ReduceOperation::new(all_axes.clone(), ReductionKind::Sum),
                    Vec::new(),
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", total_sums, 1, ProgramError);
                let sliced_sum = if input_is_empty {
                    // The strided slice covered no positions, so its sum is a scalar zero of the padding value's
                    // type.
                    MaybeZero::Zero(inputs[1].r#type().cotangent()).materialize(context)?
                } else {
                    let sliced_sums = context.stage_operation(
                        ReduceOperation::new(all_axes, ReductionKind::Sum),
                        Vec::new(),
                        std::slice::from_ref(&input_cotangent),
                    )?;
                    check_count!("output", sliced_sums, 1, ProgramError);
                    sliced_sums.into_iter().next().unwrap()
                };
                let padding_value_cotangents = context.stage_operation(
                    O::from(SubOperation),
                    Vec::new(),
                    &[total_sums.into_iter().next().unwrap(), sliced_sum],
                )?;
                check_count!("output", padding_value_cotangents, 1, ProgramError);
                let padding_value_cotangent = padding_value_cotangents
                    .into_iter()
                    .next()
                    .unwrap()
                    .unalign_cotangent(&inputs[1].r#type().cotangent())?;
                Ok(vec![MaybeZero::Value(input_cotangent), MaybeZero::Value(padding_value_cotangent)])
            }
        }
    }
}

/// Batching rule for [`PadOperation`].
///
/// A batched input with a replicated padding value keeps its batch axis by padding it with zero amounts: the
/// lifted operation inserts `0` into all three padding vectors at the batch axis position. A batch-varying (batched)
/// padding value cannot ride along structurally — the lifted operation would need a rank-1 padding operand, which
/// the operation cannot represent — so the rule falls back to per-item expansion via `batch_by_item_expansion`:
/// the batch size is static, so each batch item's input and padding value are extracted, padded independently, and
/// restacked along a fresh leading batch axis (`O(batch)` staged operations, the same trade the batch-varying
/// dynamic-slice start-index rules make). This keeps the direct batched JVP path (dense forward Jacobians and
/// batched pullbacks) total even though the padding-value tangent is represented as a per-item batch there.
impl<C> BatchableOperation<C> for PadOperation
where
    C: Context<Type = ArrayType> + Zero<C::Value>,
    C::Value: Broadcast + Transpose + Slice + UpdateSlice + Reshape + Reshard,
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
        // Batch-varying padding value: pad each batch item independently and restack along a fresh leading batch axis.
        let axis_size = axis_size.expect("a mapped input pins the batch size");
        batch_by_item_expansion(context, crate::operations::manipulation::PAD_OPERATION_NAME, self, inputs, axis_size)
    }
}

/// Represents the ability to expand an array by adding edge and interior padding filled with a scalar padding value,
/// with the semantics of StableHLO's [`pad`](https://openxla.org/stablehlo/spec#pad) operation restricted to
/// non-negative padding amounts. StableHLO also allows negative edge padding, which trims elements instead; that form
/// is not supported.
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
/// with the input's data type in the same memory space. Padding requires static input extents: inputs with dynamic
/// dimensions are rejected because the padded extent cannot be computed from an unknown extent. A zero-padding
/// operation passes its input through unchanged. Any non-identity output keeps the input memory space, clears explicit
/// physical layout metadata, and preserves compatible sharding and reduction state.
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
        edge_padding_low: &[usize],
        edge_padding_high: &[usize],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError>;
}

impl Pad for ArrayType {
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[usize],
        edge_padding_high: &[usize],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        if self.data_type() != padding_value.data_type() {
            return Err(TypeError {
                message: format!(
                    "'pad' input data type {} does not match padding value data type {}",
                    self.data_type(),
                    padding_value.data_type(),
                ),
            }
            .into());
        }
        if padding_value.rank() != 0 {
            return Err(TypeError {
                message: format!("'pad' padding value must be a scalar but has type {padding_value}"),
            }
            .into());
        }
        if self.memory() != padding_value.memory() {
            return Err(TypeError {
                message: format!(
                    "'pad' input and padding value must share one memory space but reside in {} and {}",
                    self.memory(),
                    padding_value.memory(),
                ),
            }
            .into());
        }
        let rank = self.rank();
        for (name, padding) in [
            ("edge_padding_low", edge_padding_low),
            ("edge_padding_high", edge_padding_high),
            ("interior_padding", interior_padding),
        ] {
            if padding.len() != rank {
                return Err(TypeError {
                    message: format!("'pad' {name} has length {} but input has rank {rank}", padding.len()),
                }
                .into());
            }
        }
        if edge_padding_low.iter().all(|padding| *padding == 0)
            && edge_padding_high.iter().all(|padding| *padding == 0)
            && interior_padding.iter().all(|padding| *padding == 0)
        {
            return Ok(self.clone());
        }
        let mut output_dimensions = Vec::with_capacity(rank);
        for axis in 0..rank {
            let dimension = self.dimension(axis as isize);
            let Size::Static(size) = dimension else {
                return Err(TypeError {
                    message: format!(
                        "'pad' does not support dynamic input axis {axis} with size {dimension}; the padded extent \
                        cannot be computed from an unknown extent",
                    ),
                }
                .into());
            };
            let output_size = if size == 0 {
                edge_padding_low[axis].checked_add(edge_padding_high[axis])
            } else {
                interior_padding[axis]
                    .checked_add(1)
                    .and_then(|stride| (size - 1).checked_mul(stride))
                    .and_then(|interior| interior.checked_add(1))
                    .and_then(|interior| edge_padding_low[axis].checked_add(interior))
                    .and_then(|size| size.checked_add(edge_padding_high[axis]))
            }
            .ok_or_else(|| TypeError { message: format!("'pad' output size overflows usize on axis {axis}") })?;
            output_dimensions.push(Size::Static(output_size));
        }
        // Padding resizes dimensions in place, so the operand sharding (placement and reduction state) carries
        // through, with the same divisibility check on padded sharded dimensions that `slice` applies. The scalar
        // padding value's sharding does not affect the output.
        let sharding = resized_output_sharding(self, &output_dimensions, PAD_OPERATION_NAME)?;
        ArrayType::new(self.data_type(), Shape::new(output_dimensions))
            .with_memory(self.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError { message: error.to_string() }.into())
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
        edge_padding_low: &[usize],
        edge_padding_high: &[usize],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        let output_type = self.r#type().pad(
            padding_value.r#type().as_ref(),
            edge_padding_low,
            edge_padding_high,
            interior_padding,
        )?;
        if output_type.eq(self.r#type().as_ref()) {
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
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let padding_value_type = ArrayType::scalar(DataType::F64);
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(8)]));
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
                        ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)])),
                        padding_value_type.clone(),
                    ],
                    error = "'pad' does not support dynamic input axis 0 with size *; the padded extent cannot be \
                        computed from an unknown extent",
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(usize::MAX)])),
                        padding_value_type.clone(),
                    ],
                    error = "'pad' output size overflows usize on axis 0",
                },
            ],
        );
        assert_eq!(input_type.pad(&padding_value_type, &[1], &[2], &[1]), Ok(output_type.clone()));

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
        let empty_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0)]));
        assert_eq!(
            empty_type.pad(&padding_value_type, &[1], &[2], &[1]),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]))),
        );
        let empty = Array::from_f64s(empty_type, vec![]).pad(&Array::scalar(7.0), &[1], &[2], &[1]).unwrap();
        assert_eq!(empty.to_f64s(), vec![7.0, 7.0, 7.0]);
        let scalar = Array::scalar(42.0).pad(&Array::scalar(7.0), &[], &[], &[]).unwrap();
        assert_eq!(scalar.to_f64s(), vec![42.0]);

        // Invalid construction and inputs report precise operation and interpreter errors.
        assert_eq!(
            PadOperation::new(vec![1], vec![2, 0], vec![1]),
            Err(ProgramError::Type(TypeError {
                message: "'pad' expects edge_padding_low, edge_padding_high, and interior_padding to share one length \
                    but got lengths 1, 2, and 1"
                    .to_string(),
            })),
        );
        assert_eq!(
            PadOperation::new(vec![1, 0], vec![2, 0], vec![1, 0])
                .unwrap()
                .infer_output_types(&[input_type.clone(), padding_value_type.clone()], &[]),
            Err(TypeError { message: "'pad' edge_padding_low has length 2 but input has rank 1".to_string() }),
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
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );

        // Batching inserts zero padding on the mapped axis and expands per item for a mapped padding value.
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
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()])))),
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                ],
                output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])],
                input_cotangents = [Array::vector(vec![2.0, 4.0, 6.0]), Array::scalar(24.0)],
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
        assert_eq!(*output.r#type(), ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4), Size::Static(3)])),);
        assert_eq!(output.to_f64s(), vec![0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0],);

        // The kernel validates the padding value shape eagerly.
        assert_eq!(
            Array::vector(vec![1.0, 2.0]).pad(&Array::vector(vec![0.0]), &[0], &[0], &[0]),
            Err(ProgramError::Type(TypeError {
                message: "'pad' padding value must be a scalar but has type f64[1]".to_string(),
            })),
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
        let input = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let pad_value = ArrayType::scalar(DataType::F32);

        // Padding preserves a common memory placement and rejects a padding scalar that would require an implicit
        // transfer.
        let host_input =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)])).with_memory(Memory::Host { pinned: true });
        let host_padding = ArrayType::scalar(DataType::F32).with_memory(Memory::Host { pinned: true });
        assert_eq!(host_input.pad(&host_padding, &[0], &[1], &[0]).unwrap().memory(), Memory::Host { pinned: true },);
        assert_eq!(
            host_input.pad(&pad_value, &[0], &[1], &[0]),
            Err(ProgramError::Type(TypeError {
                message: "'pad' input and padding value must share one memory space but reside in Host[Pinned] and \
                          Device"
                    .to_string(),
            })),
        );
        let laid_out_input = host_input.with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(laid_out_input.pad(&host_padding, &[0], &[0], &[0]), Ok(laid_out_input.clone()));

        // Padding to an evenly divisible size keeps the operand sharding (including the unreduced manual axis): with
        // low = 0, interior = 0, and high = 4 the output is 0 + 4 + 4 = 8, divisible by the `x` mesh-axis size (2).
        assert_eq!(input.pad(&pad_value, &[0], &[4], &[0]).unwrap().sharding(), Some(&sharding));
        // Padding to a size not divisible by the explicit mesh-axis size (output 0 + 4 + 1 = 5) is rejected.
        assert!(input.pad(&pad_value, &[0], &[1], &[0]).is_err());
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
            let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)]))
                .with_sharding(physical_sharding)
                .unwrap();
            let input = ArrayBatch::new(
                input_type.clone(),
                Array::from_f64s(input_type, vec![1.0, 2.0, 3.0, 4.0]),
                BatchAxis::new(0),
            )
            .unwrap();
            let padding_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))
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

        // Per-item expansion handles an empty batch without inventing values or dropping the mapped placement.
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let physical_sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0), Size::Static(2)]))
                .with_sharding(physical_sharding.clone())
                .unwrap();
            let input =
                ArrayBatch::new(input_type.clone(), Array::from_f64s(input_type, Vec::new()), BatchAxis::new(0))
                    .unwrap();
            let padding_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0)]))
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
            assert_eq!(outputs[0].r#type().shape().dimensions(), &[Size::Static(0), Size::Static(3)]);
            assert!(outputs[0].value().values().is_empty());
        }
    }
}
