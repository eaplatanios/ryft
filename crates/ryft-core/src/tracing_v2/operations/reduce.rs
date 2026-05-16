use std::fmt::Display;

use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::domains::{Tracer, TracingDomain};
use crate::tracing::{ProgramTracingContext, Traceable, TracingError};
use crate::tracing_v2::operations::control_flow::ControlFlowError;
use crate::types::{ArrayType, DataType, Type, TypeError};

/// Kind of reduction performed by a [`ReduceOperation`].
///
/// Reductions collapse one or more axes of an input array by combining their elements with a
/// binary associative-commutative operator that defines an identity element. Each kind corresponds
/// to one such operator/identity pair and lowers to the equivalent `stablehlo.reduce` body in the
/// XLA backend.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReductionKind {
    /// Numeric sum reduction. The identity is `0` and the combiner is addition.
    Sum,

    /// Numeric mean reduction: a [`Sum`](Self::Sum) divided by the product of reduced extents.
    /// The numeric data type must support division.
    Mean,

    /// Numeric maximum reduction. The identity is the data type's smallest representable value.
    Max,

    /// Numeric minimum reduction. The identity is the data type's largest representable value.
    Min,

    /// Boolean disjunction reduction (logical-OR). The identity is `false` and the combiner is OR.
    /// Inputs must have [`DataType::Boolean`].
    Any,

    /// Boolean conjunction reduction (logical-AND). The identity is `true` and the combiner is
    /// AND. Inputs must have [`DataType::Boolean`].
    All,
}

impl ReductionKind {
    /// Returns the canonical operation name suffix for this kind.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Max => "max",
            Self::Min => "min",
            Self::Any => "any",
            Self::All => "all",
        }
    }

    /// Returns `true` when this kind requires [`DataType::Boolean`] inputs.
    pub fn requires_boolean(self) -> bool {
        matches!(self, Self::Any | Self::All)
    }
}

impl Display for ReductionKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Trait that represents [`Operation`] carrier types that support/include [`ReduceOperation`].
/// Backend-owned closed [`Operation`] carrier types (such as
/// [`ArrayOperation`](super::ArrayOperation), for example) implement this trait so that generic
/// transform code can stage [`ReduceOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsReduce<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the reduce [`Operation`] with the
    /// provided reduced axes and reduction kind.
    fn reduce_operation(axes: Vec<usize>, kind: ReductionKind) -> Self;
}

/// Value-level reduction capability.
///
/// [`Reduce`] is the receiver-style entry point for staging or executing N-D
/// [`ReduceOperation`]. Reduces the receiver along `axes` using the operator/identity pair
/// described by `kind`, returning a value whose rank is `self.rank() - axes.len()`.
pub trait Reduce: Sized {
    /// Reduces `self` along `axes` using the operator selected by `kind`.
    fn reduce(self, axes: &[usize], kind: ReductionKind) -> Self;
}

impl<'domain, D> Reduce for Tracer<'domain, D>
where
    D: TracingDomain<Type = ArrayType>,
    D::OperationCarrier: SupportsReduce<ArrayType, D::Value>,
{
    #[inline]
    fn reduce(self, axes: &[usize], kind: ReductionKind) -> Self {
        if axes.is_empty() {
            return self;
        }
        self.unary(D::OperationCarrier::reduce_operation(axes.to_vec(), kind))
    }
}

/// Symbolic-zero-aware reduce: `Zero[type].reduce(axes, kind) -> Zero[reduced_type]`.
///
/// Sum/Max/Min/Mean of symbolic zero are zero; Any/All of symbolic zero are unsupported on the
/// tangent space and are not produced by autodiff in practice. We preserve the symbolic-zero
/// metadata uniformly here and rely on type inference for the reduced shape.
impl<V> Reduce for crate::differentiation::Tangent<ArrayType, V>
where
    V: Traceable<ArrayType> + Reduce,
{
    fn reduce(self, axes: &[usize], kind: ReductionKind) -> Self {
        match self {
            Self::Zero(r#type) => match reduce_abstract(&r#type, axes, kind, "reduce") {
                Ok(reduced_type) => Self::Zero(reduced_type),
                Err(_) => Self::Zero(r#type),
            },
            Self::Value(value) => Self::Value(value.reduce(axes, kind)),
        }
    }
}

/// Returns the output [`ArrayType`] produced by reducing `input` along `axes` with `kind`.
///
/// Validates that:
///   - `axes` are unique and within `0..rank(input)`;
///   - `kind` matches the input data type (Boolean for Any/All, non-Boolean for the others).
///
/// The reduced axes are removed from the output shape; non-reduced axes keep their order.
pub fn reduce_abstract(
    input: &ArrayType,
    axes: &[usize],
    kind: ReductionKind,
    op: &'static str,
) -> Result<ArrayType, TypeError> {
    let rank = input.rank();
    let mut seen = vec![false; rank];
    for axis in axes {
        if *axis >= rank {
            return Err(TypeError { message: (format!("{op} axis {axis} is out of bounds for rank {rank}")).into() });
        }
        if seen[*axis] {
            return Err(TypeError { message: (format!("{op} contains duplicate axis {axis}")).into() });
        }
        seen[*axis] = true;
    }

    let data_type = input.data_type();
    if kind.requires_boolean() && data_type != DataType::Boolean {
        return Err(TypeError {
            message: (format!("{op} kind {kind} requires Boolean inputs but got {data_type}")).into(),
        });
    }
    if !kind.requires_boolean() && data_type == DataType::Boolean {
        return Err(TypeError {
            message: (format!("{op} kind {kind} requires numeric inputs but got {data_type}")).into(),
        });
    }

    let mut current = input.clone();
    let mut sorted_axes: Vec<usize> = axes.to_vec();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
    for axis in sorted_axes {
        current = current.without_dimension(axis)?.0;
    }
    Ok(current)
}

/// Lifts a reduce's `axes` through one batching level inserted at `batch_axis`.
///
/// Returns the rewritten axes and the output batch axis position. Each user axis `i` shifts to
/// `i + 1` when `i >= batch_axis`. The output batch axis is `batch_axis` minus the number of
/// reduced axes that lie strictly below it (because those axes get dropped in the output).
///
/// Reducing the batch axis itself is rejected because the user's reduce describes per-lane
/// semantics; collapsing the lane axis would change the meaning of `vmap`. Callers should
/// surface this as a [`BatchingError::MissingBatchingRule`](
/// crate::tracing_v2::BatchingError::MissingBatchingRule).
pub fn lift_reduce_axes(axes: &[usize], batch_axis: usize) -> Option<(Vec<usize>, usize)> {
    let mut lifted = Vec::with_capacity(axes.len());
    let mut axes_below_batch = 0usize;
    for axis in axes {
        if *axis == batch_axis {
            return None;
        }
        if *axis < batch_axis {
            lifted.push(*axis);
            axes_below_batch += 1;
        } else {
            lifted.push(*axis + 1);
        }
    }
    let output_batch_axis = batch_axis - axes_below_batch;
    Some((lifted, output_batch_axis))
}

/// Primitive representing one N-dimensional axis-collapsing reduction.
///
/// [`ReduceOperation`] collapses the input array along `axes` using the operator/identity pair
/// described by [`kind`](Self::kind). The output rank is the input rank minus the number of
/// reduced axes; non-reduced axes keep their relative order. Lowers to StableHLO's
/// `stablehlo.reduce` op in the XLA backend.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReduceOperation {
    /// Axes to reduce.
    axes: Vec<usize>,

    /// Kind of reduction.
    kind: ReductionKind,
}

impl ReduceOperation {
    /// Creates a new [`ReduceOperation`] reducing along `axes` with the supplied `kind`.
    #[inline]
    pub fn new(axes: Vec<usize>, kind: ReductionKind) -> Self {
        Self { axes, kind }
    }

    /// Returns the axes reduced by this operation.
    #[inline]
    pub fn axes(&self) -> &[usize] {
        self.axes.as_slice()
    }

    /// Returns the kind of reduction.
    #[inline]
    pub fn kind(&self) -> ReductionKind {
        self.kind
    }
}

impl Display for ReduceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "reduce_{}({:?})", self.kind, self.axes)
    }
}

impl Operation<ArrayType> for ReduceOperation {
    #[inline]
    fn name(&self) -> &'static str {
        match self.kind {
            ReductionKind::Sum => "reduce_sum",
            ReductionKind::Mean => "reduce_mean",
            ReductionKind::Max => "reduce_max",
            ReductionKind::Min => "reduce_min",
            ReductionKind::Any => "reduce_any",
            ReductionKind::All => "reduce_all",
        }
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![reduce_abstract(&input_types[0], self.axes.as_slice(), self.kind, self.name())?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("axes", &format_args!("{:?}", self.axes)))
    }
}

impl<V: Traceable<ArrayType> + Reduce> InterpretableOperation<ArrayType, V> for ReduceOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().reduce(self.axes.as_slice(), self.kind)])
    }
}

/// Transpose (vector-Jacobian product) for a [`ReduceOperation`].
///
/// For a sum/mean reduction along the stored axes, the cotangent of the input is the output
/// cotangent broadcast back to the input shape (singleton-broadcasting over each reduced axis).
/// The current implementation stores only the reduced axes — not the original input shape — so
/// reconstructing the input shape requires extra metadata that is not yet available here. The
/// `Cotangent::Zero` branch passes through; the `Cotangent::Staged` branch surfaces a
/// [`ControlFlowError::MissingTransformRule`] documenting the follow-up.
impl<V, O> LinearOperation<ArrayType, V, O> for ReduceOperation
where
    V: Traceable<ArrayType> + Reduce,
    O: Clone + Operation<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, ArrayType, V, O>,
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match &output_cotangents[0] {
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
            Cotangent::Staged(_) => Err(ControlFlowError::MissingTransformRule {
                transform: "reduce transpose (would need broadcast-back with stored input shape)",
            }
            .into()),
        }
    }
}

impl<V> crate::tracing_v2::batching::BatchableOperation<V> for ReduceOperation
where
    V: Traceable<ArrayType>,
    ReduceOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let (_, input_axes, _) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        let Some(batch_axis) = input_axes[0] else {
            return crate::tracing_v2::batching::apply_with_axes(self, inputs, &[None]);
        };
        let Some((lifted_axes, output_axis)) = lift_reduce_axes(self.axes.as_slice(), batch_axis) else {
            return Err(crate::tracing_v2::batching::BatchingError::MissingBatchingRule {
                operation: format!(
                    "{} cannot reduce along the mapped lane axis {batch_axis}; use an explicit \
                    reduction inside the function instead of vmap-collapsing the lane",
                    self.name(),
                ),
            }
            .into());
        };
        let lifted_op = ReduceOperation::new(lifted_axes, self.kind);
        crate::tracing_v2::batching::apply_with_axes(&lifted_op, inputs, &[Some(output_axis)])
    }
}

/// N-D reduce helper that operates on a flat row-major payload and shape.
///
/// Returns `(reduced_values, reduced_shape)`. `axes` may be in any order; duplicates are not
/// permitted (callers should validate beforehand). The `combiner` function applies the reduction
/// operator and `identity` returns the initial accumulator value for each output cell.
///
/// # Parameters
///
///   - `values`: Row-major input payload.
///   - `shape`: Input shape.
///   - `axes`: Axes to reduce.
///   - `identity`: Initial accumulator value for each output element.
///   - `combiner`: Binary reduction operator.
pub fn reduce_evaluate<T: Clone>(
    values: &[T],
    shape: &[usize],
    axes: &[usize],
    identity: impl Fn() -> T,
    combiner: impl Fn(T, T) -> T,
) -> (Vec<T>, Vec<usize>) {
    let rank = shape.len();
    let mut reduce_mask = vec![false; rank];
    for axis in axes {
        reduce_mask[*axis] = true;
    }
    let output_shape: Vec<usize> =
        shape.iter().enumerate().filter_map(|(axis, size)| if reduce_mask[axis] { None } else { Some(*size) }).collect();
    let output_element_count: usize = output_shape.iter().product();
    let mut output = (0..output_element_count).map(|_| identity()).collect::<Vec<_>>();
    if output_element_count == 0 {
        return (output, output_shape);
    }

    let input_strides = row_major_strides(shape);
    let output_strides = row_major_strides(output_shape.as_slice());

    let mut input_index = vec![0usize; rank];
    let input_element_count: usize = shape.iter().product();
    if input_element_count == 0 {
        return (output, output_shape);
    }

    loop {
        let mut input_flat = 0usize;
        let mut output_flat = 0usize;
        let mut output_axis = 0usize;
        for (axis, position) in input_index.iter().enumerate() {
            input_flat += position * input_strides[axis];
            if !reduce_mask[axis] {
                output_flat += position * output_strides[output_axis];
                output_axis += 1;
            }
        }
        output[output_flat] = combiner(output[output_flat].clone(), values[input_flat].clone());

        let mut position = rank;
        let mut carry = true;
        while position > 0 && carry {
            position -= 1;
            input_index[position] += 1;
            if input_index[position] < shape[position] {
                carry = false;
            } else {
                input_index[position] = 0;
            }
        }
        if carry {
            return (output, output_shape);
        }
    }
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride *= shape[axis];
    }
    strides
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, BatchingError};
    use crate::tracing_v2::test_util::TestArray;
    use crate::types::{ArrayType, DataType, Shape, Size, Typed};

    use super::*;

    fn array_type(dimensions: &[usize], data_type: DataType) -> ArrayType {
        ArrayType::new(data_type, Shape::new(dimensions.iter().copied().map(Size::Static).collect()), None, None)
            .unwrap()
    }

    #[test]
    fn test_reduce_abstract_drops_reduced_axes_and_keeps_remaining_order() {
        let input = array_type(&[2, 3, 4], DataType::F64);
        assert_eq!(reduce_abstract(&input, &[1], ReductionKind::Sum, "reduce_sum"), Ok(array_type(&[2, 4], DataType::F64)));
        assert_eq!(
            reduce_abstract(&input, &[0, 2], ReductionKind::Max, "reduce_max"),
            Ok(array_type(&[3], DataType::F64))
        );
    }

    #[test]
    fn test_reduce_abstract_rejects_out_of_bounds_and_duplicate_axes() {
        let input = array_type(&[2, 3], DataType::F64);
        assert!(reduce_abstract(&input, &[2], ReductionKind::Sum, "reduce_sum").is_err());
        assert!(reduce_abstract(&input, &[0, 0], ReductionKind::Sum, "reduce_sum").is_err());
    }

    #[test]
    fn test_reduce_abstract_enforces_boolean_data_type_for_any_and_all() {
        let numeric = array_type(&[2, 3], DataType::F64);
        assert!(reduce_abstract(&numeric, &[1], ReductionKind::Any, "reduce_any").is_err());
        let boolean = array_type(&[2, 3], DataType::Boolean);
        assert!(reduce_abstract(&boolean, &[1], ReductionKind::Sum, "reduce_sum").is_err());
        assert_eq!(
            reduce_abstract(&boolean, &[1], ReductionKind::Any, "reduce_any"),
            Ok(array_type(&[2], DataType::Boolean))
        );
    }

    #[test]
    fn test_lift_reduce_axes_shifts_axes_above_batch_and_keeps_axes_below() {
        // Per-lane reduce over axes [0, 2] of a rank-3 input. Batching at axis 1 inserts a new
        // dimension at position 1, so per-lane axis 0 stays at 0, per-lane axis 2 shifts to 3.
        // Output batch axis is at position 1 - 1 = 0 (one reduced axis was below the batch axis).
        assert_eq!(lift_reduce_axes(&[0, 2], 1), Some((vec![0, 3], 0)));
        // Reducing only above the batch axis leaves the batch axis position unchanged.
        assert_eq!(lift_reduce_axes(&[2], 0), Some((vec![3], 0)));
        // Reducing the batch axis itself is rejected.
        assert_eq!(lift_reduce_axes(&[0, 1], 1), None);
    }

    #[test]
    fn test_reduce_operation_interprets_sum_over_axis() {
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let outputs = ReduceOperation::new(vec![1], ReductionKind::Sum).interpret(std::slice::from_ref(&input)).unwrap();
        let output = outputs.into_iter().next().unwrap();
        assert_eq!(output.array_type().shape(), &Shape::new(vec![Size::Static(2)]));
        assert_eq!(output.values(), &[6.0, 15.0]);
    }

    #[test]
    fn test_reduce_operation_batches_lane_uniform_input_as_pass_through() {
        let input = ArrayBatch::unbatched(TestArray::matrix(2, 3, vec![1.0; 6]));
        let outputs = ReduceOperation::new(vec![1], ReductionKind::Sum).batch(std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].value().values(), &[3.0, 3.0]);
    }

    #[test]
    fn test_reduce_operation_batches_along_non_lane_axis() {
        // Physical input is [3 lanes, 2 rows, 3 cols] mapped at axis 0. Per-lane reduce over
        // axis 1 (the "cols" axis from the per-lane view; physically axis 2 after batching).
        let values: Vec<f64> = (0..18).map(|index| index as f64).collect();
        let input = ArrayBatch::mapped(
            TestArray::new(array_type(&[3, 2, 3], DataType::F64), values),
            0,
        )
        .unwrap();
        let outputs = ReduceOperation::new(vec![1], ReductionKind::Sum).batch(std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].r#type().shape(), &Shape::new(vec![Size::Static(3), Size::Static(2)]));
        assert_eq!(outputs[0].value().values(), &[3.0, 12.0, 21.0, 30.0, 39.0, 48.0]);
    }

    #[test]
    fn test_reduce_operation_rejects_reducing_the_batch_axis() {
        let input = ArrayBatch::mapped(TestArray::matrix(3, 2, vec![1.0; 6]), 0).unwrap();
        // Per-lane axis 0 collides with the mapped lane axis once lifted.
        let error =
            ReduceOperation::new(vec![0], ReductionKind::Sum).batch(std::slice::from_ref(&input)).unwrap_err();
        assert!(matches!(
            error,
            TracingError::Batching(BatchingError::MissingBatchingRule { ref operation })
                if operation.contains("reduce along the mapped lane axis"),
        ));
    }

    #[test]
    fn test_reduce_evaluate_combines_along_specified_axes() {
        let values: Vec<f64> = (1..=24).map(|index| index as f64).collect();
        let (reduced, shape) = reduce_evaluate(values.as_slice(), &[2, 3, 4], &[1], || 0.0, |acc, value| acc + value);
        assert_eq!(shape, vec![2, 4]);
        // Row 0 sums across axis 1: [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]
        // Row 1 sums across axis 1: [13+17+21, 14+18+22, 15+19+23, 16+20+24] = [51, 54, 57, 60]
        assert_eq!(reduced, vec![15.0, 18.0, 21.0, 24.0, 51.0, 54.0, 57.0, 60.0]);
    }
}
