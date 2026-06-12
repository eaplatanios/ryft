use std::fmt::Display;
use std::ops::Mul;

use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::constants::SupportsFill;
use crate::operations::manipulation::{Broadcast, SupportsBroadcast};
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::{AbstractTracingContext, Tracer};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, ResidualFactor, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::{ArrayType, DataType, Shape, StaticShape, Type, TypeError, Typed};

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

/// Trait for operation types that include or can wrap [`ReduceOperation`].
/// Backend-owned closed operation enums (such as
/// [`ArrayOperation`](super::ArrayOperation), for example) implement this trait so that generic
/// transform code can stage [`ReduceOperation`] without knowing the concrete operation enum.
#[doc(hidden)]
pub trait SupportsReduce<T: Type> {
    /// Constructs the backend-specific representation of the reduce [`Operation`] with the
    /// provided input shape, reduced axes, and reduction kind.
    ///
    /// The `input_shape` is needed by the linear transpose rule to broadcast the cotangent back
    /// to the input rank; staging callers already know the input shape from the operand and
    /// supply it here.
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

impl<C> Reduce for Tracer<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: SupportsReduce<ArrayType>,
{
    #[inline]
    fn reduce(self, axes: &[usize], kind: ReductionKind) -> Self {
        if axes.is_empty() {
            return self;
        }
        self.unary(C::Operation::reduce_operation(axes.to_vec(), kind))
    }
}

/// Symbolic-zero-aware reduce: `Zero[type].reduce(axes, kind) -> Zero[reduced_type]`.
///
/// Sum/Max/Min/Mean of symbolic zero are zero; Any/All of symbolic zero are unsupported on the
/// tangent space and are not produced by autodiff in practice. We preserve the symbolic-zero
/// metadata uniformly here and rely on type inference for the reduced shape.
impl<V: Value<ArrayType> + Reduce> Reduce for crate::differentiation::Tangent<ArrayType, V> {
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
            return Err(TypeError { message: format!("{op} axis {axis} is out of bounds for rank {rank}") });
        }
        if seen[*axis] {
            return Err(TypeError { message: format!("{op} contains duplicate axis {axis}") });
        }
        seen[*axis] = true;
    }

    let data_type = input.data_type();
    if kind.requires_boolean() && data_type != DataType::Boolean {
        return Err(TypeError { message: format!("{op} kind {kind} requires Boolean inputs but got {data_type}") });
    }
    if !kind.requires_boolean() && data_type == DataType::Boolean {
        return Err(TypeError { message: format!("{op} kind {kind} requires numeric inputs but got {data_type}") });
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
/// semantics; collapsing the lane axis would change the meaning of `batch`. Callers should
/// surface this as a [`BatchingError::UnsupportedOperation`](
/// crate::batching::BatchingError::UnsupportedOperation).
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
    /// Creates a new [`ReduceOperation`] reducing along `axes` with the supplied `kind`. The input
    /// shape is not part of the operation payload: it is recoverable from the staged input types
    /// wherever a rule needs it.
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
        self.render(formatter, 0)
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
            .bracketed(|operation| operation.field("axes", format_args!("{:?}", self.axes)))
    }
}

impl<V: Value<ArrayType> + Reduce> InterpretableOperation<ArrayType, V> for ReduceOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone().reduce(self.axes.as_slice(), self.kind)])
    }
}

/// Transpose (vector-Jacobian product) for a [`ReduceOperation`].
///
/// For a `Sum` reduction, the cotangent of the input is the output cotangent broadcast back to
/// the input shape — singleton-broadcasting over each reduced axis. For a `Mean` reduction, the
/// same broadcast-back result is additionally scaled by `1 / N` where `N` is the product of the
/// reduced axis extents. `Max`/`Min` would need an argmax-style gather to route the cotangent
/// only to the lane that produced the reduction's output, and `Any`/`All` are not
/// differentiable.
impl<V: Value<ArrayType> + Broadcast<Output = V> + Mul<Output = V>, O> TransposableOperation<ArrayType, V, O>
    for ReduceOperation
where
    O: Operation<ArrayType>
        + SupportsBroadcast<ArrayType>
        + SupportsFill<ArrayType, f64>
        + crate::operations::arithmetic::SupportsMul<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_cotangents, 1, ProgramError);
        let input_shape = input_types[0].shape();
        match &output_cotangents[0] {
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
            Cotangent::Staged(cotangent) => match self.kind {
                ReductionKind::Sum | ReductionKind::Mean => {
                    let output_type = ArrayType::new(cotangent.r#type().data_type(), input_shape.clone());
                    let output_axes = output_to_input_axis_map(input_shape.rank(), &self.axes);
                    let broadcasted = cotangent.clone().broadcast(output_type, output_axes.as_slice())?;
                    let cotangent_input = match self.kind {
                        ReductionKind::Sum => broadcasted,
                        ReductionKind::Mean => {
                            let element_count: usize = self
                                .axes
                                .iter()
                                .map(|axis| {
                                    input_shape.dimension(*axis as isize).value().ok_or(TypeError {
                                        message: format!(
                                            "mean transpose requires static reduced extents but axis {axis} of \
                                            {input_shape} is dynamic",
                                        ),
                                    })
                                })
                                .product::<Result<usize, _>>()?;
                            let inverse_count = 1.0 / element_count as f64;
                            // Stage a nullary rank-0 fill holding `1 / N` and rely on implicit rank-0 broadcasting in
                            // the subsequent multiplication to scale the broadcast-back cotangent to the input shape.
                            let factor_type = ArrayType::new(cotangent.r#type().data_type(), Shape::scalar());
                            let factor = context
                                .stage_operation::<&crate::tracing::AbstractTracer<ArrayType, V, O>>(
                                    O::fill_operation(factor_type, inverse_count),
                                    &[],
                                )?
                                .into_iter()
                                .next()
                                .ok_or(ProgramError::InvalidOutputCount { expected: 1, actual: 0 })?;
                            factor * broadcasted
                        }
                        _ => unreachable!("outer match handled the only two supported kinds"),
                    };
                    Ok(vec![Cotangent::Staged(cotangent_input)])
                }
                other => Err(TypeError {
                    message: format!(
                        "reduce transpose for {other} is not yet supported; only Sum and Mean are wired \
                        (Max/Min need argmax-style gather; Any/All are not differentiable)"
                    ),
                }
                .into()),
            },
        }
    }
}

/// JVP rule for [`ReduceOperation`].
///
/// `Sum` and `Mean` linearize to themselves: the tangent of `reduce_sum(x)` is `reduce_sum(Δx)`
/// and similarly for `Mean`. `Max`/`Min` use a primal-domain argmax mask: the tangent of
/// `reduce_max(x)` along axis `a` is `reduce_sum(mask * Δx)` along the same axis, where
/// `mask[i] = 1` exactly when `x[i]` equals the per-axis maximum (ties are split evenly,
/// matching the JAX convention). `Any`/`All` are not differentiable.
impl<D> DifferentiableOperation<D> for ReduceOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Reduce + Broadcast<Output = D::Value> + crate::operations::compare::Compare<Output = D::Value>,
    D::Tangent: Reduce,
    LinearOperationOf<D>: SupportsReduce<ArrayType>
        + crate::operations::arithmetic::SupportsScale<ArrayType, ResidualFactor<ArrayType, D::Value>>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        match self.kind {
            ReductionKind::Sum | ReductionKind::Mean => {
                let primal = inputs[0].primal().clone().reduce(self.axes.as_slice(), self.kind);
                let tangent = inputs[0].tangent().clone().reduce(self.axes.as_slice(), self.kind);
                Ok(vec![JvpTracer::new(primal, tangent)])
            }
            ReductionKind::Max | ReductionKind::Min => {
                use crate::operations::arithmetic::Scale;
                use crate::operations::compare::{Compare, ComparisonDirection};
                let primal_input = inputs[0].primal().clone();
                let primal_y = primal_input.clone().reduce(self.axes.as_slice(), self.kind);
                let input_type = primal_input.r#type().into_owned();
                let output_axes = output_to_input_axis_map(input_type.rank(), &self.axes);
                let broadcast_y = primal_y.clone().broadcast(input_type, output_axes.as_slice())?;
                let mask = primal_input.compare(broadcast_y, ComparisonDirection::Equal);
                let masked_tangent = inputs[0].tangent().clone().scale(context.factor(mask));
                let tangent_y = masked_tangent.reduce(self.axes.as_slice(), ReductionKind::Sum);
                Ok(vec![JvpTracer::new(primal_y, tangent_y)])
            }
            other => Err(TypeError {
                message: format!("reduce jvp for {other} is not supported: Any and All are not differentiable"),
            }
            .into()),
        }
    }
}

/// Builds the `output_axes` vector that maps a reduced output's axes back to the
/// corresponding input axes. Output axis `j` corresponds to the `j`-th non-reduced input axis;
/// the returned vector lists those input-axis indices in order.
fn output_to_input_axis_map(input_rank: usize, reduced_axes: &[usize]) -> Vec<usize> {
    let mut reduce_mask = vec![false; input_rank];
    for axis in reduced_axes {
        reduce_mask[*axis] = true;
    }
    (0..input_rank).filter(|axis| !reduce_mask[*axis]).collect()
}

impl<V: Value<ArrayType>, C> crate::tracing_v2::batching::BatchableOperation<V, C> for ReduceOperation
where
    ReduceOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        _context: &C,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (_, input_axes, _) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        let Some(batch_axis) = input_axes[0] else {
            return crate::tracing_v2::batching::apply_with_axes(self, inputs, &[None]);
        };
        let Some((lifted_axes, output_axis)) = lift_reduce_axes(self.axes.as_slice(), batch_axis) else {
            return Err(crate::batching::BatchingError::UnsupportedOperation {
                message: format!(
                    "{} cannot reduce along the mapped lane axis {batch_axis}; use an explicit \
                    reduction inside the function instead of batch-collapsing the lane",
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
    shape: &StaticShape,
    axes: &[usize],
    identity: impl Fn() -> T,
    combiner: impl Fn(T, T) -> T,
) -> (Vec<T>, StaticShape) {
    let rank = shape.rank();
    let mut reduce_mask = vec![false; rank];
    for axis in axes {
        reduce_mask[*axis] = true;
    }
    let output_shape = StaticShape::new(
        shape
            .dimensions()
            .iter()
            .enumerate()
            .filter_map(|(axis, size)| if reduce_mask[axis] { None } else { Some(*size) })
            .collect(),
    );
    let output_element_count: usize = output_shape.dimensions().iter().product();
    let mut output = (0..output_element_count).map(|_| identity()).collect::<Vec<_>>();
    if output_element_count == 0 {
        return (output, output_shape);
    }

    let input_strides = shape.row_major_strides();
    let output_strides = output_shape.row_major_strides();

    let mut input_index = vec![0usize; rank];
    let input_element_count: usize = shape.dimensions().iter().product();
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::batching::BatchingError;
    use crate::tests::TestArray;
    use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};
    use crate::types::{ArrayType, DataType, Shape, Size, Typed};

    use super::*;

    fn array_type(dimensions: &[usize], data_type: DataType) -> ArrayType {
        ArrayType::new(data_type, Shape::new(dimensions.iter().copied().map(Size::Static).collect()))
    }

    #[test]
    fn test_reduce_abstract_drops_reduced_axes_and_keeps_remaining_order() {
        let input = array_type(&[2, 3, 4], DataType::F64);
        assert_eq!(
            reduce_abstract(&input, &[1], ReductionKind::Sum, "reduce_sum"),
            Ok(array_type(&[2, 4], DataType::F64))
        );
        assert_eq!(
            reduce_abstract(&input, &[0, 2], ReductionKind::Max, "reduce_max"),
            Ok(array_type(&[3], DataType::F64))
        );
    }

    #[test]
    fn test_reduce_abstract_propagates_dynamic_dimensions() {
        // Dynamic dimensions flow through reduce inference: reduced axes are dropped whether they are static or
        // dynamic, and the remaining dynamic dimensions are preserved in order.
        let input = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Size::Dynamic(None), Size::Static(3), Size::Dynamic(Some(4))]),
        );
        assert_eq!(
            reduce_abstract(&input, &[1], ReductionKind::Sum, "reduce_sum"),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Dynamic(Some(4))]))),
        );
        assert_eq!(
            reduce_abstract(&input, &[0, 2], ReductionKind::Sum, "reduce_sum"),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]))),
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
        let outputs =
            ReduceOperation::new(vec![1], ReductionKind::Sum).interpret(std::slice::from_ref(&input)).unwrap();
        let output = outputs.into_iter().next().unwrap();
        assert_eq!(output.array_type().shape(), &Shape::new(vec![Size::Static(2)]));
        assert_eq!(output.values(), &[6.0, 15.0]);
    }

    #[test]
    fn test_reduce_operation_batches_lane_uniform_input_as_pass_through() {
        let input = ArrayBatch::unbatched(TestArray::matrix(2, 3, vec![1.0; 6]));
        let outputs =
            ReduceOperation::new(vec![1], ReductionKind::Sum).batch(&(), std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].value().values(), &[3.0, 3.0]);
    }

    #[test]
    fn test_reduce_operation_batches_along_non_lane_axis() {
        // Physical input is [3 lanes, 2 rows, 3 cols] mapped at axis 0. Per-lane reduce over
        // axis 1 (the "cols" axis from the per-lane view; physically axis 2 after batching).
        let values: Vec<f64> = (0..18).map(|index| index as f64).collect();
        let physical_type = array_type(&[3, 2, 3], DataType::F64);
        let input = ArrayBatch::mapped(TestArray::new(physical_type, values), 0).unwrap();
        let outputs =
            ReduceOperation::new(vec![1], ReductionKind::Sum).batch(&(), std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].r#type().shape(), &Shape::new(vec![Size::Static(3), Size::Static(2)]));
        assert_eq!(outputs[0].value().values(), &[3.0, 12.0, 21.0, 30.0, 39.0, 48.0]);
    }

    #[test]
    fn test_reduce_operation_rejects_reducing_the_batch_axis() {
        let input = ArrayBatch::mapped(TestArray::matrix(3, 2, vec![1.0; 6]), 0).unwrap();
        // Per-lane axis 0 collides with the mapped lane axis once lifted.
        let error = ReduceOperation::new(vec![0], ReductionKind::Sum)
            .batch(&(), std::slice::from_ref(&input))
            .unwrap_err();
        // The `batch` rule runs at the operation level, so its `BatchingError` rides up as a `ProgramError::Custom`
        // payload; recover the concrete error with `downcast_custom`.
        assert!(matches!(
            error.downcast_custom::<BatchingError>(),
            Some(BatchingError::UnsupportedOperation { message }) if message.contains("reduce along the mapped lane axis"),
        ));
    }

    #[test]
    fn test_output_to_input_axis_map_handles_reduced_and_kept_axes() {
        // Input rank 3, reduce axis 1: output axes [0, 1] map back to input axes [0, 2].
        assert_eq!(super::output_to_input_axis_map(3, &[1]), vec![0, 2]);
        // Input rank 3, reduce axes [0, 2]: output axis [0] maps back to input axis [1].
        assert_eq!(super::output_to_input_axis_map(3, &[0, 2]), vec![1]);
        // Input rank 4, reduce axes [1, 3]: output axes [0, 1] map back to input axes [0, 2].
        assert_eq!(super::output_to_input_axis_map(4, &[1, 3]), vec![0, 2]);
        // No reduction: identity map.
        assert_eq!(super::output_to_input_axis_map(3, &[]), vec![0, 1, 2]);
    }

    #[test]
    fn test_reduce_operation_infer_output_types_follows_the_input_type() {
        // The operation carries no input shape; the output type is derived from the actual staged
        // input type, and out-of-range axes are rejected against it.
        let operation = ReduceOperation::new(vec![1], ReductionKind::Sum);
        let input = array_type(&[3, 2], DataType::F64);
        assert_eq!(operation.infer_output_types(&[input]), Ok(vec![array_type(&[3], DataType::F64)]));
        let rank_one_input = array_type(&[3], DataType::F64);
        assert!(operation.infer_output_types(&[rank_one_input]).is_err());
    }

    #[test]
    fn test_reduce_evaluate_combines_along_specified_axes() {
        let values: Vec<f64> = (1..=24).map(|index| index as f64).collect();
        let (reduced, shape) = reduce_evaluate(
            values.as_slice(),
            &StaticShape::new(vec![2, 3, 4]),
            &[1],
            || 0.0,
            |acc, value| acc + value,
        );
        assert_eq!(shape, StaticShape::new(vec![2, 4]));
        // Row 0 sums across axis 1: [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]
        // Row 1 sums across axis 1: [13+17+21, 14+18+22, 15+19+23, 16+20+24] = [51, 54, 57, 60]
        assert_eq!(reduced, vec![15.0, 18.0, 21.0, 24.0, 51.0, 54.0, 57.0, 60.0]);
    }

    #[test]
    fn test_reduce_mean_transpose_divides_by_axis_size() {
        // Mean over a length-4 axis: transpose maps a unit cotangent to a broadcast-back
        // cotangent of `1 / 4` at every input position.
        use std::cell::RefCell;
        use std::rc::Rc;

        use crate::differentiation::Cotangent;
        use crate::domains::AbstractDomain;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::tracing::AbstractTracingContext;
        use crate::tracing_v2::LinearArrayOperation;

        let input_shape = Shape::new(vec![Size::Static(4)]);
        let input_type = ArrayType::new(DataType::F64, input_shape.clone());
        let cotangent_type = ArrayType::scalar(DataType::F64);
        let transpose_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            TestArray,
            LinearArrayOperation<TestArray, TestArray, ArrayType>,
        >::new()));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(cotangent_type);
        let domain = AbstractDomain::new();
        let mut context = AbstractTracingContext::<
            ArrayType,
            TestArray,
            LinearArrayOperation<TestArray, TestArray, ArrayType>,
        >::new(&domain, transpose_builder.clone());
        let output_cotangent = context.tracer(output_cotangent_atom, None);
        let contribution = ReduceOperation::new(vec![0], ReductionKind::Mean)
            .transpose(&mut context, &[&input_type], &[Cotangent::Staged(output_cotangent)])
            .unwrap()
            .into_iter()
            .next()
            .expect("transpose should return one contribution");
        let Cotangent::Staged(contribution) = contribution else {
            panic!("transpose should produce one cotangent contribution");
        };
        let contribution_atom = contribution.atom_id().unwrap();
        drop(contribution);
        drop(context);
        let transpose_builder = Rc::try_unwrap(transpose_builder)
            .expect("transpose builder should not have outstanding linear terms")
            .into_inner();
        let transpose_program = transpose_builder
            .build::<TestArray, TestArray>(vec![contribution_atom], Placeholder, Placeholder)
            .unwrap();
        let result = transpose_program.interpret(TestArray::scalar(1.0)).unwrap();
        assert_eq!(result.array_type().shape(), &input_shape);
        for value in result.values() {
            let delta = (*value - 0.25).abs();
            assert!(delta < 1e-9, "expected ≈ 0.25, got {value}");
        }
    }

    #[test]
    fn test_collective_pmean_divides_by_lane_count() {
        use crate::tracing_v2::operations::collective::{CollectiveKind, CollectiveOperation};
        // Per-lane scalar input of shape [3] mapped at axis 0. PMean returns the mean of the
        // three lane values as a lane-uniform scalar.
        let input = ArrayBatch::mapped(TestArray::vector(vec![2.0, 4.0, 6.0]), 0).unwrap();
        let outputs = CollectiveOperation::new("data".to_string(), CollectiveKind::PMean).batch(&(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), None);
        let values = outputs[0].value().values();
        assert_eq!(values.len(), 1);
        let delta = (values[0] - 4.0).abs();
        assert!(delta < 1e-9, "expected pmean = 4.0, got {}", values[0]);
    }

    fn run_reduce_jvp(
        primal_values: Vec<f64>,
        tangent_input_values: Vec<f64>,
        axes: Vec<usize>,
        kind: ReductionKind,
        expected_primal: &[f64],
        expected_tangent: &[f64],
    ) {
        use std::cell::RefCell;
        use std::collections::HashMap;
        use std::rc::Rc;

        use crate::differentiation::Tangent;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::tests::{TestArray, TestArrayDomain};
        use crate::tracing_v2::differentiation::{JvpTracer, ResidualFactor, TangentContext};
        use crate::tracing_v2::{LinearArrayOperation, ResidualizedOperation};

        let domain = TestArrayDomain;
        let builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            TestArray,
            LinearArrayOperation<
                TestArray,
                TestArray,
                ArrayType,
                std::convert::Infallible,
                ResidualFactor<ArrayType, TestArray>,
            >,
        >::new()));
        let residuals = Rc::new(RefCell::new(Vec::new()));
        let residual_atoms = Rc::new(RefCell::new(HashMap::new()));
        let mut context =
            TangentContext::new_with_residuals(&domain, builder.clone(), residuals.clone(), residual_atoms);
        let input_array_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(primal_values.len())]));
        let tangent_input = context.input(input_array_type.clone());
        let primal_input = TestArray::vector(primal_values);
        let operation = ReduceOperation::new(axes, kind);
        let outputs = DifferentiableOperation::<TestArrayDomain>::jvp(
            &operation,
            &mut context,
            &[JvpTracer::from_value(primal_input, tangent_input)],
        )
        .unwrap();
        assert_eq!(outputs[0].primal().values(), expected_primal);
        let tangent_atom = match outputs[0].tangent() {
            Tangent::Value(tracer) => tracer.atom_id().unwrap(),
            Tangent::Zero(_) => panic!("max/min jvp should produce a concrete tangent"),
        };
        drop(outputs);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let tangent_program =
            builder.build::<TestArray, TestArray>(vec![tangent_atom], Placeholder, Placeholder).unwrap();
        let residuals = residuals.borrow();
        let tangent_program = tangent_program
            .map_operations(|operation| {
                ResidualizedOperation::<TestArrayDomain>::instantiate_residuals(operation, residuals.as_slice())
            })
            .unwrap();
        let result = tangent_program.interpret(TestArray::vector(tangent_input_values)).unwrap();
        assert_eq!(result.values(), expected_tangent);
    }

    #[test]
    fn test_reduce_max_jvp_routes_tangent_through_argmax_mask() {
        // JVP of `reduce_max([1, 5, 3])` with tangent `[10, 20, 30]` returns `(5, 20)`. The
        // argmax mask is `[0, 1, 0]`, so the masked tangent contributes only the tangent value
        // at the argmax position.
        run_reduce_jvp(vec![1.0, 5.0, 3.0], vec![10.0, 20.0, 30.0], vec![0], ReductionKind::Max, &[5.0], &[20.0]);
    }

    #[test]
    fn test_reduce_min_jvp_mirrors_max() {
        // JVP of `reduce_min([1, 5, 3])` with tangent `[10, 20, 30]` returns `(1, 10)`. The
        // argmin mask is `[1, 0, 0]`, so the masked tangent contributes only the first value.
        run_reduce_jvp(vec![1.0, 5.0, 3.0], vec![10.0, 20.0, 30.0], vec![0], ReductionKind::Min, &[1.0], &[10.0]);
    }

    #[test]
    fn test_reduce_max_jvp_splits_ties_evenly() {
        // Ties: primal `[1, 5, 5, 3]` has two argmax positions. The mask is `[0, 1, 1, 0]`, so
        // the masked-and-summed tangent for an input `[a, b, c, d]` is `b + c`.
        run_reduce_jvp(
            vec![1.0, 5.0, 5.0, 3.0],
            vec![7.0, 11.0, 13.0, 17.0],
            vec![0],
            ReductionKind::Max,
            &[5.0],
            &[24.0],
        );
    }
}
