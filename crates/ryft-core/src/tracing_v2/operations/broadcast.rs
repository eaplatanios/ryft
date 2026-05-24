use std::fmt::Display;

use half::{bf16, f16};

use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::{Context, ProgramTracingContext, Traceable, Tracer, TracingError};
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
use crate::tracing_v2::operations::ControlFlowError;
use crate::tracing_v2::{Differentiable, DifferentiableOperation};
use crate::types::{ArrayType, Shape, Size, Type, TypeError, Typed};

/// Trait that represents [`Operation`] carrier types that support/include
/// [`BroadcastInDimOperation`]. Backend-owned closed [`Operation`] carrier types implement this
/// trait so that generic transform code can stage [`BroadcastInDimOperation`] without knowing
/// which carrier is in use.
#[doc(hidden)]
pub trait SupportsBroadcastInDim<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of [`BroadcastInDimOperation`].
    fn broadcast_in_dim_operation(target_type: T, broadcast_dimensions: Vec<usize>) -> Self;
}

/// Value-level capability for the general N-dimensional broadcast primitive — the direct
/// analogue of JAX's
/// [`lax.broadcast_in_dim`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast_in_dim.html)
/// and the lowering target of StableHLO's
/// [`broadcast_in_dim`](https://openxla.org/stablehlo/spec#broadcast_in_dim).
///
/// `t.broadcast_in_dim(target_type, broadcast_dimensions)` expands `t` to `target_type` by
/// mapping each input axis `i` to output axis `broadcast_dimensions[i]`, replicating the value
/// along the axes of `target_type` that are not named in `broadcast_dimensions`. For each `i`,
/// the input dimension at axis `i` must either equal the corresponding output dimension or be
/// `1` (in which case it is replicated to match).
///
/// # Parameters
///
///   - `target_type`: Shape (and element type) of the output array.
///   - `broadcast_dimensions`: For each axis `i` of the input, the output axis it maps to.
///     Must have length equal to the input's rank and contain distinct values in
///     `0..target_type.rank()`.
///
/// # Examples
///
/// Manually adding axes to broadcast a length-3 vector to a `[2, 3]` matrix by mapping the
/// input's single axis to output axis `1`. Mirrors the corresponding JAX example
/// `jax.lax.broadcast_in_dim(jnp.array([1, 2, 3]), shape=(2, 3), broadcast_dimensions=(1,))`:
///
/// ```text
/// let x = TestArray::vector(vec![1.0, 2.0, 3.0]);
/// let target = ArrayType::new(
///     DataType::F64,
///     Shape::new(vec![Size::Static(2), Size::Static(3)]),
///     None,
///     None,
/// )?;
/// let y = x.broadcast_in_dim(target, vec![1]);
/// // y has shape [2, 3] with values:
/// //   [[1.0, 2.0, 3.0],
/// //    [1.0, 2.0, 3.0]]
/// ```
///
/// Broadcasting a `[2, 2]` matrix over a new middle dimension of size `3` by mapping the
/// input's axes to output axes `0` and `2`. Mirrors
/// `jax.lax.broadcast_in_dim(jnp.array([[1, 2], [3, 4]]), shape=(2, 3, 2), broadcast_dimensions=(0, 2))`:
///
/// ```text
/// let x = TestArray { /* shape [2, 2], values [1.0, 2.0, 3.0, 4.0] */ };
/// let target = ArrayType::new(
///     DataType::F64,
///     Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(2)]),
///     None,
///     None,
/// )?;
/// let y = x.broadcast_in_dim(target, vec![0, 2]);
/// // y has shape [2, 3, 2] with values:
/// //   [[[1.0, 2.0],
/// //     [1.0, 2.0],
/// //     [1.0, 2.0]],
/// //    [[3.0, 4.0],
/// //     [3.0, 4.0],
/// //     [3.0, 4.0]]]
/// ```
pub trait BroadcastInDim: Sized {
    /// Broadcasts `self` to `target_type` using `broadcast_dimensions`. See the trait-level
    /// documentation for the full semantics, parameter contract, and examples.
    fn broadcast_in_dim(self, target_type: ArrayType, broadcast_dimensions: Vec<usize>) -> Self;
}

impl<C> BroadcastInDim for Tracer<C>
where
    C: Context<Type = ArrayType>,
    C::Operation: SupportsBroadcastInDim<ArrayType, C::Value>,
{
    #[inline]
    fn broadcast_in_dim(self, target_type: ArrayType, broadcast_dimensions: Vec<usize>) -> Self {
        self.unary(C::Operation::broadcast_in_dim_operation(target_type, broadcast_dimensions))
    }
}

macro_rules! impl_broadcast_in_dim_for_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl BroadcastInDim for $ty {
                #[inline]
                fn broadcast_in_dim(self, _target_type: ArrayType, _broadcast_dimensions: Vec<usize>) -> Self {
                    self
                }
            }
        )*
    };
}

impl_broadcast_in_dim_for_scalar!(bf16, f16, f32, f64);

/// Symbolic-zero-aware broadcast: `Zero[type].broadcast_in_dim(target, dims) -> Zero[target]`.
impl<V> BroadcastInDim for crate::differentiation::Tangent<ArrayType, V>
where
    V: Traceable<ArrayType> + BroadcastInDim,
{
    fn broadcast_in_dim(self, target_type: ArrayType, broadcast_dimensions: Vec<usize>) -> Self {
        match self {
            Self::Zero(_) => Self::Zero(target_type),
            Self::Value(value) => Self::Value(value.broadcast_in_dim(target_type, broadcast_dimensions)),
        }
    }
}

/// Convenience capability for prepending leading dimensions of given sizes — the direct
/// analogue of JAX's
/// [`lax.broadcast`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast.html).
///
/// `t.broadcast([s0, s1, ...])` produces a value whose shape is `[s0, s1, ..., t.shape...]`,
/// with the original value replicated across the new leading axes. Equivalent to
/// `t.broadcast_in_dim(target_type, broadcast_dimensions)` with `target_type` of shape
/// `[s0, s1, ..., t.shape...]` and `broadcast_dimensions` mapping each input axis `i` to
/// output axis `i + sizes.len()`.
///
/// # Parameters
///
///   - `sizes`: The sizes of the new leading dimensions to prepend, in order. Each entry must
///     be a non-negative integer.
///
/// # Example
///
/// Broadcasting a length-3 vector to a `[2, 3]` matrix by prepending one leading axis of size
/// `2` — replicates the original vector across the new axis. Mirrors the corresponding JAX
/// example `jax.lax.broadcast(jnp.array([1, 2, 3]), sizes=(2,))`:
///
/// ```text
/// let x = TestArray::vector(vec![1.0, 2.0, 3.0]);
/// let y = x.broadcast(vec![2]);
/// // y has shape [2, 3] with values:
/// //   [[1.0, 2.0, 3.0],
/// //    [1.0, 2.0, 3.0]]
/// ```
pub trait Broadcast: Sized {
    /// Broadcasts `self` by prepending leading dimensions of the supplied sizes. See the
    /// trait-level documentation for the full semantics and example.
    fn broadcast(self, sizes: Vec<usize>) -> Self;
}

impl<T> Broadcast for T
where
    T: BroadcastInDim + Typed<ArrayType>,
{
    fn broadcast(self, sizes: Vec<usize>) -> Self {
        let input_type = self.r#type().into_owned();
        let input_rank = input_type.rank();
        let mut output_dimensions: Vec<Size> = sizes.iter().map(|size| Size::Static(*size)).collect();
        output_dimensions.extend(input_type.shape().dimensions().iter().copied());
        let target_type = ArrayType::new(input_type.data_type(), Shape::new(output_dimensions), None, None)
            .expect("prepended leading sizes preserve shape validity");
        let broadcast_dimensions: Vec<usize> = (0..input_rank).map(|axis| axis + sizes.len()).collect();
        self.broadcast_in_dim(target_type, broadcast_dimensions)
    }
}

/// Convenience capability for NumPy-style broadcasting to a target shape — the direct analogue
/// of JAX's
/// [`jnp.broadcast_to`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.broadcast_to.html),
/// which itself mirrors
/// [`numpy.broadcast_to`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html).
///
/// `t.broadcast_to(target_shape)` right-aligns the input shape with `target_shape`: input
/// axis `i` corresponds to output axis `target_shape.rank() - input.rank() + i`. Each
/// corresponding input dimension must equal the output dimension or be `1` (in which case it
/// is replicated). Missing leading input dimensions are treated as size `1`, so a smaller-rank
/// array can be broadcast to a larger-rank target shape. Equivalent to
/// `t.broadcast_in_dim(target_type, broadcast_dimensions)` with `broadcast_dimensions`
/// computed as the trailing range of indices.
///
/// # Parameters
///
///   - `target_shape`: Shape to which `self` is broadcast. Must have rank at least equal to
///     the input's rank, with each output dimension compatible with the right-aligned input
///     dimension under NumPy broadcasting rules.
///
/// # Example
///
/// Broadcasting a length-3 vector to a `[3, 3]` matrix replicates the input across the
/// leading axis. Mirrors the corresponding JAX example
/// `jnp.broadcast_to(jnp.array([1, 2, 3]), (3, 3))`:
///
/// ```text
/// let x = TestArray::vector(vec![1.0, 2.0, 3.0]);
/// let y = x.broadcast_to(Shape::new(vec![Size::Static(3), Size::Static(3)]));
/// // y has shape [3, 3] with values:
/// //   [[1.0, 2.0, 3.0],
/// //    [1.0, 2.0, 3.0],
/// //    [1.0, 2.0, 3.0]]
/// ```
pub trait BroadcastTo: Sized {
    /// Broadcasts `self` to `target_shape` using NumPy-style right-aligned semantics. See the
    /// trait-level documentation for the full semantics and example.
    fn broadcast_to(self, target_shape: Shape) -> Self;
}

impl<T> BroadcastTo for T
where
    T: BroadcastInDim + Typed<ArrayType>,
{
    fn broadcast_to(self, target_shape: Shape) -> Self {
        let input_type = self.r#type().into_owned();
        let input_rank = input_type.rank();
        let offset = target_shape.rank().saturating_sub(input_rank);
        let target_type = ArrayType::new(input_type.data_type(), target_shape, None, None)
            .expect("broadcast_to target shape preserves ArrayType validity");
        let broadcast_dimensions: Vec<usize> = (0..input_rank).map(|axis| axis + offset).collect();
        self.broadcast_in_dim(target_type, broadcast_dimensions)
    }
}

/// Convenience capability for broadcasting to match another typed value's shape — the direct
/// analogue of JAX's
/// [`lax.broadcast_to_rank`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast_to_rank.html)
/// when called with the rank of another array, and of the common
/// `jnp.broadcast_to(operand, like.shape)` pattern when a target array is on hand. Plays the
/// same role as
/// [`torch.Tensor.expand_as`](https://pytorch.org/docs/stable/generated/torch.Tensor.expand_as.html)
/// in PyTorch.
///
/// `t.broadcast_like(like)` broadcasts `self` to the shape of `like` using NumPy-style
/// right-aligned semantics. Useful in autodiff and deep-learning patterns where a gradient or
/// constant must be expanded to match the shape of another tensor. Equivalent to
/// `t.broadcast_to(like.r#type().shape().clone())`.
///
/// # Parameters
///
///   - `like`: Any typed value whose `ArrayType` defines the target shape. Each right-aligned
///     dimension of `self` must either equal the corresponding dimension of `like` or be `1`;
///     the input may have lower rank than `like`, in which case the missing leading dimensions
///     are treated as size `1`.
///
/// # Example
///
/// Broadcasting a length-3 vector so its shape matches a `[3, 3]` matrix:
///
/// ```text
/// let x = TestArray::vector(vec![1.0, 2.0, 3.0]);
/// let like = TestArray { /* shape [3, 3] */ };
/// let y = x.broadcast_like(&like);
/// // y has shape [3, 3] with values:
/// //   [[1.0, 2.0, 3.0],
/// //    [1.0, 2.0, 3.0],
/// //    [1.0, 2.0, 3.0]]
/// ```
pub trait BroadcastLike: Sized {
    /// Broadcasts `self` to match `like`'s shape. See the trait-level documentation for the
    /// full semantics and example.
    fn broadcast_like<L: Typed<ArrayType>>(self, like: &L) -> Self;
}

impl<T> BroadcastLike for T
where
    T: BroadcastTo,
{
    #[inline]
    fn broadcast_like<L: Typed<ArrayType>>(self, like: &L) -> Self {
        self.broadcast_to(like.r#type().shape().clone())
    }
}

/// Computes the abstract output [`ArrayType`] produced by broadcasting an input of `input_type`
/// to `target_type` via `broadcast_dimensions`.
///
/// Validates that `broadcast_dimensions` has length equal to `input_type.rank()`, each value
/// addresses a distinct output axis within `target_type.rank()`, and each mapped input
/// dimension either matches the corresponding output dimension or equals `1`.
pub fn broadcast_in_dim_abstract(
    input_type: &ArrayType,
    target_type: &ArrayType,
    broadcast_dimensions: &[usize],
    op: &'static str,
) -> Result<ArrayType, TypeError> {
    if input_type.data_type() != target_type.data_type() {
        return Err(TypeError {
            message: format!(
                "{op} input element type {} does not match target element type {}",
                input_type.data_type(),
                target_type.data_type(),
            ),
        });
    }
    let input_rank = input_type.rank();
    let target_rank = target_type.rank();
    if broadcast_dimensions.len() != input_rank {
        return Err(TypeError {
            message: format!(
                "{op} broadcast_dimensions has length {} but input has rank {input_rank}",
                broadcast_dimensions.len(),
            ),
        });
    }
    let mut seen = vec![false; target_rank];
    for (input_axis, &output_axis) in broadcast_dimensions.iter().enumerate() {
        if output_axis >= target_rank {
            return Err(TypeError {
                message: format!(
                    "{op} broadcast_dimensions[{input_axis}] = {output_axis} is out of bounds for target rank {target_rank}",
                ),
            });
        }
        if seen[output_axis] {
            return Err(TypeError {
                message: format!("{op} broadcast_dimensions maps two input axes to output axis {output_axis}"),
            });
        }
        seen[output_axis] = true;
        let input_dim = input_type.dimension(input_axis as isize);
        let target_dim = target_type.dimension(output_axis as isize);
        match (input_dim.value(), target_dim.value()) {
            (Some(input_size), Some(target_size)) if input_size != target_size && input_size != 1 => {
                return Err(TypeError {
                    message: format!(
                        "{op} input axis {input_axis} has size {input_size}, which is neither {target_size} nor 1",
                    ),
                });
            }
            _ => {}
        }
    }
    Ok(target_type.clone())
}

/// Primitive representing the general N-dimensional broadcast — the direct analogue of JAX's
/// [`lax.broadcast_in_dim`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast_in_dim.html).
///
/// Expands its input to `target_type` by mapping each input axis `i` to output axis
/// `broadcast_dimensions[i]`, replicating along the remaining axes of `target_type`. Lowers to
/// StableHLO's `broadcast_in_dim` op in the XLA backend.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BroadcastInDimOperation {
    /// Target output [`ArrayType`].
    target_type: ArrayType,

    /// For each input axis `i`, the output axis it maps to.
    broadcast_dimensions: Vec<usize>,
}

impl BroadcastInDimOperation {
    /// Creates a new [`BroadcastInDimOperation`] with the supplied target type and broadcast
    /// dimensions.
    #[inline]
    pub fn new(target_type: ArrayType, broadcast_dimensions: Vec<usize>) -> Self {
        Self { target_type, broadcast_dimensions }
    }

    /// Returns the target output [`ArrayType`].
    #[inline]
    pub fn target_type(&self) -> &ArrayType {
        &self.target_type
    }

    /// Returns the broadcast dimensions: for each input axis, the output axis it maps to.
    #[inline]
    pub fn broadcast_dimensions(&self) -> &[usize] {
        self.broadcast_dimensions.as_slice()
    }
}

impl Display for BroadcastInDimOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Operation<ArrayType> for BroadcastInDimOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "broadcast_in_dim"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![broadcast_in_dim_abstract(
            &input_types[0],
            &self.target_type,
            self.broadcast_dimensions.as_slice(),
            "broadcast_in_dim",
        )?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("target_type", &self.target_type)?;
            operation.field("broadcast_dimensions", format_args!("{:?}", self.broadcast_dimensions))
        })
    }
}

impl<V: Traceable<ArrayType> + BroadcastInDim> InterpretableOperation<ArrayType, V> for BroadcastInDimOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().broadcast_in_dim(self.target_type.clone(), self.broadcast_dimensions.clone())])
    }
}

impl<V, O> LinearOperation<ArrayType, V, O> for BroadcastInDimOperation
where
    V: Traceable<ArrayType>,
    O: Clone + Operation<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, ArrayType, V, O>,
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        // Transpose (pullback) of a broadcast is a sum-reduction over the added axes. We don't
        // currently have a `ReduceSum` primitive, so transpose is not yet implemented for
        // broadcasts. Symbolic-zero cotangents propagate through unchanged.
        match &output_cotangents[0] {
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
            Cotangent::Staged(_) => Err(ControlFlowError::MissingTransformRule {
                transform: "broadcast_in_dim transpose (would need reduce-sum)",
            }
            .into()),
        }
    }
}

impl<D> DifferentiableOperation<D> for BroadcastInDimOperation
where
    D: Differentiable<Type = ArrayType>,
    D::Value: BroadcastInDim,
    D::Tangent: BroadcastInDim,
    D::LinearOperationCarrier: SupportsBroadcastInDim<ArrayType, D::Tangent>,
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
        let primal = inputs[0]
            .primal()
            .clone()
            .broadcast_in_dim(self.target_type.clone(), self.broadcast_dimensions.clone());
        let tangent = inputs[0]
            .tangent()
            .clone()
            .broadcast_in_dim(self.target_type.clone(), self.broadcast_dimensions.clone());
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

/// N-D broadcast helper that operates on a flat row-major payload and shape.
///
/// Returns the broadcasted values in row-major order over `target_shape`. Each input axis `i`
/// maps to output axis `broadcast_dimensions[i]`; output axes not named in `broadcast_dimensions`
/// are added (the input value is replicated along them). When the input's dimension at axis `i`
/// is `1`, that axis is also replicated along the corresponding output axis.
pub fn broadcast_in_dim_evaluate<T: Clone>(
    values: &[T],
    input_shape: &[usize],
    target_shape: &[usize],
    broadcast_dimensions: &[usize],
) -> Vec<T> {
    let input_rank = input_shape.len();
    let target_rank = target_shape.len();
    let output_count: usize = target_shape.iter().product();
    if output_count == 0 {
        return Vec::new();
    }
    let input_strides = row_major_strides(input_shape);
    let mut output = Vec::with_capacity(output_count);
    let mut target_index = vec![0usize; target_rank];
    loop {
        let mut input_flat = 0usize;
        for input_axis in 0..input_rank {
            let target_axis = broadcast_dimensions[input_axis];
            let coordinate = if input_shape[input_axis] == 1 { 0 } else { target_index[target_axis] };
            input_flat += coordinate * input_strides[input_axis];
        }
        output.push(values[input_flat].clone());

        let mut position = target_rank;
        while position > 0 {
            position -= 1;
            target_index[position] += 1;
            if target_index[position] < target_shape[position] {
                break;
            }
            target_index[position] = 0;
            if position == 0 {
                return output;
            }
        }
        if target_rank == 0 {
            return output;
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

/// Lifts a broadcast's [`broadcast_dimensions`] and target shape through one batching level.
///
/// When the input is batched at axis `k` (in the input's per-lane logical shape), the lifted
/// broadcast inserts a batch dimension of size `axis_size` at position `k` in the target shape,
/// places a corresponding batch axis at output position `k_out = k`, and shifts the existing
/// `broadcast_dimensions` so each previously-mapped output axis `>= k_out` is shifted by one.
/// The new batch input axis itself maps to `k_out`.
pub fn lift_broadcast_in_dim(
    broadcast_dimensions: &[usize],
    target_type: &ArrayType,
    input_batch_axis: usize,
    axis_size: usize,
) -> Result<(Vec<usize>, ArrayType, usize), TypeError> {
    let target_batch_axis = input_batch_axis;
    let mut lifted_target_dimensions: Vec<Size> = target_type.shape().dimensions().to_vec();
    lifted_target_dimensions.insert(target_batch_axis, Size::Static(axis_size));
    let lifted_target = ArrayType::new(target_type.data_type(), Shape::new(lifted_target_dimensions), None, None)
        .map_err(|error| TypeError { message: error.to_string() })?;

    let mut lifted_dimensions = Vec::with_capacity(broadcast_dimensions.len() + 1);
    for &output_axis in broadcast_dimensions.iter() {
        let shifted_output_axis = if output_axis >= target_batch_axis { output_axis + 1 } else { output_axis };
        lifted_dimensions.push(shifted_output_axis);
    }
    lifted_dimensions.insert(input_batch_axis, target_batch_axis);

    Ok((lifted_dimensions, lifted_target, target_batch_axis))
}

impl<V> crate::tracing_v2::batching::BatchableOperation<V> for BroadcastInDimOperation
where
    V: Traceable<ArrayType> + BroadcastInDim,
{
    fn batch(
        &self,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let (_, input_axes, axis_size) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        match input_axes[0] {
            None => {
                // Lane-uniform input: the broadcast itself does not change. Pass through.
                let output_value = inputs[0]
                    .value()
                    .clone()
                    .broadcast_in_dim(self.target_type.clone(), self.broadcast_dimensions.clone());
                Ok(vec![crate::tracing_v2::batching::ArrayBatch::new(self.target_type.clone(), output_value, None)?])
            }
            Some(batch_axis) => {
                let (lifted_dimensions, lifted_target, target_batch_axis) =
                    lift_broadcast_in_dim(&self.broadcast_dimensions, &self.target_type, batch_axis, axis_size)?;
                let output_value = inputs[0].value().clone().broadcast_in_dim(lifted_target.clone(), lifted_dimensions);
                Ok(vec![crate::tracing_v2::batching::ArrayBatch::new(
                    lifted_target,
                    output_value,
                    Some(target_batch_axis),
                )?])
            }
        }
    }
}
