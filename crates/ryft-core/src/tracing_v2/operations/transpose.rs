use std::fmt::Display;

use half::{bf16, f16};

use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::{Context, ProgramTracingContext, Traceable, Tracer, TracingError};
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
use crate::tracing_v2::{Differentiable, DifferentiableOperation};
use crate::types::{ArrayType, Shape, Type, TypeError};

/// Trait that represents [`Operation`] carrier types that support/include [`TransposeOperation`].
/// Backend-owned closed [`Operation`] carrier types (such as
/// [`ArrayOperation`](super::ArrayOperation), for example) implement this trait so that generic
/// transform code can stage [`TransposeOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsTranspose<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the N-D transpose [`Operation`] with the
    /// provided axis permutation.
    fn transpose_operation(permutation: Vec<usize>) -> Self;
}

/// Value-level N-dimensional transpose capability.
///
/// [`Transpose`] is the receiver-style entry point for staging or executing N-D
/// [`TransposeOperation`]. The output's `i`-th axis corresponds to the input's
/// `permutation[i]`-th axis, generalizing the rank-2 matrix transpose to arbitrary tensor ranks.
pub trait Transpose: Sized {
    /// Reorders the axes of `self` according to `permutation`.
    fn transpose(self, permutation: Vec<usize>) -> Self;
}

impl<C> Transpose for Tracer<C>
where
    C: Context<Type = ArrayType>,
    C::Operation: SupportsTranspose<ArrayType, C::Value>,
{
    #[inline]
    fn transpose(self, permutation: Vec<usize>) -> Self {
        if transpose_is_identity(&permutation) {
            return self;
        }
        self.unary(C::Operation::transpose_operation(permutation))
    }
}

macro_rules! impl_transpose_for_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Transpose for $ty {
                #[inline]
                fn transpose(self, _permutation: Vec<usize>) -> Self {
                    self
                }
            }
        )*
    };
}

impl_transpose_for_scalar!(bf16, f16, f32, f64);

/// Symbolic-zero-aware N-D transpose: `Zero[type].transpose(perm) -> Zero[permuted_type]`.
impl<V> Transpose for crate::differentiation::Tangent<ArrayType, V>
where
    V: Traceable<ArrayType> + Transpose,
{
    fn transpose(self, permutation: Vec<usize>) -> Self {
        match self {
            Self::Zero(r#type) => match permute_array_type(&r#type, permutation.as_slice()) {
                Ok(permuted_type) => Self::Zero(permuted_type),
                Err(_) => Self::Zero(r#type),
            },
            Self::Value(value) => Self::Value(value.transpose(permutation)),
        }
    }
}

/// Returns `true` when `permutation` is the identity permutation of its own length.
#[inline]
pub fn transpose_is_identity(permutation: &[usize]) -> bool {
    permutation.iter().enumerate().all(|(index, value)| index == *value)
}

/// Computes the abstract output [`ArrayType`] produced by applying `permutation` to `input`.
///
/// Validates that `permutation` is a permutation of `0..rank(input)` and returns
/// `Err(TypeError)` otherwise.
pub fn transpose_abstract_nd(
    input: &ArrayType,
    permutation: &[usize],
    op: &'static str,
) -> Result<ArrayType, TypeError> {
    let rank = input.rank();
    if permutation.len() != rank {
        return Err(TypeError {
            message: format!("{op} permutation has length {} but input has rank {rank}", permutation.len()),
        });
    }
    let mut seen = vec![false; rank];
    for axis in permutation {
        if *axis >= rank {
            return Err(TypeError { message: format!("{op} permutation axis {axis} is out of bounds") });
        }
        if seen[*axis] {
            return Err(TypeError { message: format!("{op} permutation contains duplicate axis {axis}") });
        }
        seen[*axis] = true;
    }
    permute_array_type(input, permutation)
}

fn permute_array_type(input: &ArrayType, permutation: &[usize]) -> Result<ArrayType, TypeError> {
    let permuted_dimensions: Vec<_> = permutation.iter().map(|axis| input.dimension(*axis as isize)).collect();
    ArrayType::new(input.data_type(), Shape::new(permuted_dimensions), None, None)
        .map_err(|error| TypeError { message: error.to_string() })
}

/// Lifts an axis `permutation` through one batching level inserted at `batch_axis`.
///
/// The returned permutation has length `permutation.len() + 1`, places the batch axis at the
/// same output position as it appears in the input (so the output's batch axis stays at the
/// input's `batch_axis`), and shifts every other axis index `i` to `i + 1` when `i >= batch_axis`.
pub fn lift_permutation(permutation: &[usize], batch_axis: usize) -> Vec<usize> {
    let mut lifted = Vec::with_capacity(permutation.len() + 1);
    for output_axis in 0..=permutation.len() {
        if output_axis == batch_axis {
            lifted.push(batch_axis);
        } else {
            let original_output_axis = if output_axis < batch_axis { output_axis } else { output_axis - 1 };
            let input_axis = permutation[original_output_axis];
            lifted.push(if input_axis >= batch_axis { input_axis + 1 } else { input_axis });
        }
    }
    lifted
}

/// Returns the inverse permutation of `permutation`, i.e., the permutation that undoes it.
pub fn inverse_permutation(permutation: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0usize; permutation.len()];
    for (position, axis) in permutation.iter().enumerate() {
        inverse[*axis] = position;
    }
    inverse
}

/// Primitive representing N-dimensional axis permutation.
///
/// [`TransposeOperation`] reorders the axes of its input according to `permutation`. The output
/// shape is the input shape with axes permuted: output dim `i` = input dim `permutation[i]`.
/// Lowers to StableHLO's `transpose` op in the XLA backend.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TransposeOperation {
    /// Axis permutation for the transpose.
    permutation: Vec<usize>,
}

impl TransposeOperation {
    /// Creates a new N-D [`TransposeOperation`] with the supplied permutation.
    #[inline]
    pub fn new(permutation: Vec<usize>) -> Self {
        Self { permutation }
    }

    /// Returns the axis permutation for this transpose.
    #[inline]
    pub fn permutation(&self) -> &[usize] {
        self.permutation.as_slice()
    }
}

impl Display for TransposeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}({:?})", self.name(), self.permutation)
    }
}

impl Operation<ArrayType> for TransposeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "transpose"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![transpose_abstract_nd(&input_types[0], self.permutation.as_slice(), "transpose")?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("permutation", format_args!("{:?}", self.permutation)))
    }
}

impl<V: Traceable<ArrayType> + Transpose> InterpretableOperation<ArrayType, V> for TransposeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().transpose(self.permutation.clone())])
    }
}

impl<V, O> LinearOperation<ArrayType, V, O> for TransposeOperation
where
    V: Traceable<ArrayType> + Transpose,
    O: Clone + Operation<ArrayType> + SupportsTranspose<ArrayType, V>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, ArrayType, V, O>,
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        let inverse = inverse_permutation(self.permutation.as_slice());
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(cotangent.clone().transpose(inverse))]),
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<D> DifferentiableOperation<D> for TransposeOperation
where
    D: Differentiable<Type = ArrayType>,
    D::Value: Transpose,
    D::Tangent: Transpose,
    D::LinearOperationCarrier: SupportsTranspose<ArrayType, D::Tangent>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        let primal = inputs[0].primal().clone().transpose(self.permutation.clone());
        let tangent = inputs[0].tangent().clone().transpose(self.permutation.clone());
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

/// N-D transpose helper that operates on a flat row-major payload and shape.
///
/// Returns `(permuted_values, permuted_shape)`.
pub fn transpose_evaluate<T: Clone>(values: &[T], shape: &[usize], permutation: &[usize]) -> (Vec<T>, Vec<usize>) {
    let rank = shape.len();
    let permuted_shape: Vec<usize> = permutation.iter().map(|axis| shape[*axis]).collect();
    let element_count: usize = shape.iter().product();
    let mut permuted = Vec::with_capacity(element_count);
    if element_count == 0 {
        return (permuted, permuted_shape);
    }

    let input_strides = row_major_strides(shape);
    let mut permuted_index = vec![0usize; rank];
    loop {
        let mut input_flat = 0usize;
        for (position, &input_axis) in permutation.iter().enumerate() {
            input_flat += permuted_index[position] * input_strides[input_axis];
        }
        permuted.push(values[input_flat].clone());

        let mut position = rank;
        while position > 0 {
            position -= 1;
            permuted_index[position] += 1;
            if permuted_index[position] < permuted_shape[position] {
                break;
            }
            permuted_index[position] = 0;
            if position == 0 {
                return (permuted, permuted_shape);
            }
        }
        if rank == 0 {
            return (permuted, permuted_shape);
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

impl<V> crate::tracing_v2::batching::BatchableOperation<V> for TransposeOperation
where
    V: Traceable<ArrayType>,
    TransposeOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let (_, input_axes, _) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        let (lifted_permutation, output_axis) = match input_axes[0] {
            Some(batch_axis) => (lift_permutation(self.permutation(), batch_axis), Some(batch_axis)),
            None => (self.permutation().to_vec(), None),
        };
        let lifted_op = TransposeOperation::new(lifted_permutation);
        crate::tracing_v2::batching::apply_with_axes(&lifted_op, inputs, &[output_axis])
    }
}
