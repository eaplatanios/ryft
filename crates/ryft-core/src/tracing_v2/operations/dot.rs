use std::fmt::Display;

use half::{bf16, f16};

use crate::contexts::StagingContext;
use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, ResidualFactor, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::matrix::dot_abstract;

/// Specification of contracting and batching dimensions for a generalized dot product.
///
/// Mirrors StableHLO's `dot_general` operand: the contracting dimensions index axes that are
/// summed over (the "K" axes in matrix multiplication), and the batching dimensions index axes
/// that are aligned 1:1 between the two operands and preserved in the output (the leading "B"
/// axes in batched matrix multiplication).
///
/// Both `lhs_contracting_dimensions` and `rhs_contracting_dimensions` must have the same length
/// and their corresponding dimensions in the two operands must match in size. The same applies
/// to `lhs_batching_dimensions` / `rhs_batching_dimensions`.
///
/// The output shape is `[batching..., lhs_result..., rhs_result...]`, where the result
/// dimensions are the remaining (non-contracting, non-batching) dimensions of each operand, in
/// their original order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotDimensionNumbers {
    /// Axes on the LHS operand that contract with `rhs_contracting_dimensions` on the RHS.
    lhs_contracting_dimensions: Vec<usize>,

    /// Axes on the RHS operand that contract with `lhs_contracting_dimensions` on the LHS.
    rhs_contracting_dimensions: Vec<usize>,

    /// Axes on the LHS operand that are aligned 1:1 with `rhs_batching_dimensions` on the RHS
    /// and that are preserved in the output.
    lhs_batching_dimensions: Vec<usize>,

    /// Axes on the RHS operand that are aligned 1:1 with `lhs_batching_dimensions` on the LHS
    /// and that are preserved in the output.
    rhs_batching_dimensions: Vec<usize>,
}

impl DotDimensionNumbers {
    /// Creates dimension numbers from explicit contracting and batching dimensions.
    #[inline]
    pub fn new(
        lhs_contracting_dimensions: Vec<usize>,
        rhs_contracting_dimensions: Vec<usize>,
        lhs_batching_dimensions: Vec<usize>,
        rhs_batching_dimensions: Vec<usize>,
    ) -> Self {
        Self {
            lhs_contracting_dimensions,
            rhs_contracting_dimensions,
            lhs_batching_dimensions,
            rhs_batching_dimensions,
        }
    }

    /// Dimension numbers for a standard rank-2 matrix multiplication:
    /// `[M, K] @ [K, N] -> [M, N]`. Contracting dimension is the last axis of the LHS and the
    /// first axis of the RHS; there are no batching dimensions.
    #[inline]
    pub fn matmul() -> Self {
        Self::new(vec![1], vec![0], Vec::new(), Vec::new())
    }

    /// Dimension numbers for a rank-1 inner product: `[K] · [K] -> []`. The single dimension of
    /// each operand contracts.
    #[inline]
    pub fn inner_product() -> Self {
        Self::new(vec![0], vec![0], Vec::new(), Vec::new())
    }

    /// Returns the contracting dimensions of the LHS operand.
    #[inline]
    pub fn lhs_contracting_dimensions(&self) -> &[usize] {
        &self.lhs_contracting_dimensions
    }

    /// Returns the contracting dimensions of the RHS operand.
    #[inline]
    pub fn rhs_contracting_dimensions(&self) -> &[usize] {
        &self.rhs_contracting_dimensions
    }

    /// Returns the batching dimensions of the LHS operand.
    #[inline]
    pub fn lhs_batching_dimensions(&self) -> &[usize] {
        &self.lhs_batching_dimensions
    }

    /// Returns the batching dimensions of the RHS operand.
    #[inline]
    pub fn rhs_batching_dimensions(&self) -> &[usize] {
        &self.rhs_batching_dimensions
    }
}

impl Display for DotDimensionNumbers {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "(lhs_contracting={:?}, rhs_contracting={:?}, lhs_batching={:?}, rhs_batching={:?})",
            self.lhs_contracting_dimensions,
            self.rhs_contracting_dimensions,
            self.lhs_batching_dimensions,
            self.rhs_batching_dimensions,
        )
    }
}

/// Trait for operation types that include or can wrap [`DotOperation`].
/// Backend-owned closed operation enums (such as [`ArrayOperation`](super::ArrayOperation),
/// for example) implement this trait so that generic transform code can stage [`DotOperation`]
/// without knowing the concrete operation enum.
#[doc(hidden)]
pub trait SupportsDot<T: Type> {
    /// Constructs the backend-specific representation of the dot [`Operation`] with the
    /// provided dimension numbers.
    fn dot_operation(dimensions: DotDimensionNumbers) -> Self;
}

/// Value-level generalized dot capability.
///
/// [`Dot`] is the receiver-style entry point for staging or executing [`DotOperation`]. It
/// performs the contraction described by `dimensions`, supporting standard matrix
/// multiplication, batched matrix multiplication, vector inner products, and arbitrary tensor
/// contractions.
pub trait Dot<Rhs = Self>: Sized {
    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`.
    fn dot(self, rhs: Rhs, dimensions: &DotDimensionNumbers) -> Self;
}

impl<C> Dot for Tracer<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: SupportsDot<ArrayType>,
{
    #[inline]
    fn dot(self, rhs: Self, dimensions: &DotDimensionNumbers) -> Self {
        self.binary(rhs, C::Operation::dot_operation(dimensions.clone()))
    }
}

macro_rules! impl_dot_for_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Dot for $ty {
                #[inline]
                fn dot(self, rhs: Self, _dimensions: &DotDimensionNumbers) -> Self {
                    self * rhs
                }
            }
        )*
    };
}

impl_dot_for_scalar!(bf16, f16, f32, f64);

macro_rules! impl_left_right_dot_for_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl LeftDot for $ty {
                #[inline]
                fn left_dot(self, factor: Self, _dimensions: &DotDimensionNumbers) -> Self {
                    factor * self
                }
            }

            impl RightDot for $ty {
                #[inline]
                fn right_dot(self, factor: Self, _dimensions: &DotDimensionNumbers) -> Self {
                    self * factor
                }
            }
        )*
    };
}

impl_left_right_dot_for_scalar!(bf16, f16, f32, f64);

/// Primitive representing a generalized dot (tensor contraction).
///
/// [`DotOperation`] is the unified primitive for matrix multiplication, batched matrix
/// multiplication, vector inner products, and arbitrary tensor contractions. It lowers to
/// StableHLO's `dot_general` op in the XLA backend.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotOperation {
    /// Contracting and batching dimension specification.
    dimensions: DotDimensionNumbers,
}

impl DotOperation {
    /// Creates a new [`DotOperation`] with the supplied dimension numbers.
    #[inline]
    pub fn new(dimensions: DotDimensionNumbers) -> Self {
        Self { dimensions }
    }

    /// Returns a [`DotOperation`] configured for standard rank-2 matrix multiplication.
    #[inline]
    pub fn matmul() -> Self {
        Self::new(DotDimensionNumbers::matmul())
    }

    /// Returns the contracting and batching dimension specification.
    #[inline]
    pub fn dimensions(&self) -> &DotDimensionNumbers {
        &self.dimensions
    }
}

impl Display for DotOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}{}", self.name(), self.dimensions)
    }
}

impl Operation<ArrayType> for DotOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "dot"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        Ok(vec![dot_abstract(&input_types[0], &input_types[1], &self.dimensions, "dot")?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("dimensions", &self.dimensions))
    }
}

impl<V: Value<ArrayType> + Dot> InterpretableOperation<ArrayType, V> for DotOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].clone().dot(inputs[1].clone(), &self.dimensions)])
    }
}

impl<V: Value<ArrayType> + crate::tracing_v2::operations::broadcast::BroadcastInDim, RuleContext>
    crate::tracing_v2::batching::BatchableOperation<V, RuleContext> for DotOperation
where
    DotOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        _context: &RuleContext,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let (_, input_axes, axis_size) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        // Mixed batched/unbatched: broadcast the lane-uniform operand to gain a singleton batch
        // axis at position 0 (JAX's `matchaxis(0)` convention), then fall through to the
        // both-batched arm of `lift_dot_dimensions`.
        let aligned_inputs: Vec<crate::tracing_v2::batching::ArrayBatch<V>> = match (input_axes[0], input_axes[1]) {
            (Some(_), Some(_)) | (None, None) => inputs.to_vec(),
            (Some(_), None) => {
                vec![inputs[0].clone(), crate::tracing_v2::batching::broadcast_to_batched(&inputs[1], 0, axis_size)?]
            }
            (None, Some(_)) => {
                vec![crate::tracing_v2::batching::broadcast_to_batched(&inputs[0], 0, axis_size)?, inputs[1].clone()]
            }
        };
        let (_, aligned_axes, _) = crate::tracing_v2::batching::batch_input_metadata(&aligned_inputs)?;
        let (lifted_dimensions, output_axis) = lift_dot_dimensions(&self.dimensions, aligned_axes[0], aligned_axes[1])
            .expect("aligned dot inputs must produce a valid lift");
        let lifted_op = DotOperation::new(lifted_dimensions);
        crate::tracing_v2::batching::apply_with_axes(&lifted_op, &aligned_inputs, &[output_axis])
    }
}

/// JVP rule for the generalized dot product.
///
/// The pushforward of `dot(A, B; D)` is `dot(ΔA, B; D) + dot(A, ΔB; D)`: each operand's
/// contribution is itself a dot with the same dimension numbers, holding the other operand
/// constant. The two contributions are staged through [`RightDotOperation`] (holding the
/// right primal `B` constant on the right) and [`LeftDotOperation`] (holding the left primal
/// `A` constant on the left), respectively.
impl<D> DifferentiableOperation<D> for DotOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Dot,
    LinearOperationOf<D>: SupportsAdd<ArrayType>
        + SupportsLeftDot<ArrayType, ResidualFactor<ArrayType, D::Value>>
        + SupportsRightDot<ArrayType, ResidualFactor<ArrayType, D::Value>>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 2, ProgramError);
        let left = &inputs[0];
        let right = &inputs[1];
        let primal = left.primal().clone().dot(right.primal().clone(), &self.dimensions);
        let left_term = match left.tangent().clone() {
            Tangent::Zero(r#type) => Tangent::Zero(r#type),
            Tangent::Value(tangent) => Tangent::Value(tangent.right_dot(right.factor(context), &self.dimensions)),
        };
        let right_term = match right.tangent().clone() {
            Tangent::Zero(r#type) => Tangent::Zero(r#type),
            Tangent::Value(tangent) => Tangent::Value(tangent.left_dot(left.factor(context), &self.dimensions)),
        };
        let tangent = left_term + right_term;
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

/// Value-level "factor-on-the-left" dot capability.
///
/// `t.left_dot(factor, dimensions)` computes `dot(factor, t; dimensions)`. This is the linear
/// map produced by [`DotOperation`]'s JVP when the LHS primal is held constant.
pub trait LeftDot<F = Self>: Sized {
    /// Computes `dot(factor, self; dimensions)`.
    fn left_dot(self, factor: F, dimensions: &DotDimensionNumbers) -> Self;
}

/// Value-level "factor-on-the-right" dot capability.
///
/// `t.right_dot(factor, dimensions)` computes `dot(t, factor; dimensions)`. This is the linear
/// map produced by [`DotOperation`]'s JVP when the RHS primal is held constant.
pub trait RightDot<F = Self>: Sized {
    /// Computes `dot(self, factor; dimensions)`.
    fn right_dot(self, factor: F, dimensions: &DotDimensionNumbers) -> Self;
}

/// Trait for operation types that include or can wrap [`LeftDotOperation`].
#[doc(hidden)]
pub trait SupportsLeftDot<T: Type, F: Value<T>> {
    /// Constructs the backend-specific representation of the captured-factor left dot
    /// [`Operation`] with the provided factor and dimension numbers.
    fn left_dot_operation(factor: F, dimensions: DotDimensionNumbers) -> Self;
}

/// Trait for operation types that include or can wrap [`RightDotOperation`].
#[doc(hidden)]
pub trait SupportsRightDot<T: Type, F: Value<T>> {
    /// Constructs the backend-specific representation of the captured-factor right dot
    /// [`Operation`] with the provided factor and dimension numbers.
    fn right_dot_operation(factor: F, dimensions: DotDimensionNumbers) -> Self;
}

impl<C, F> LeftDot<F> for Tracer<C>
where
    C: StagingContext<Type = ArrayType>,
    F: Value<ArrayType>,
    C::Operation: SupportsLeftDot<ArrayType, F>,
{
    #[inline]
    fn left_dot(self, factor: F, dimensions: &DotDimensionNumbers) -> Self {
        self.unary(C::Operation::left_dot_operation(factor, dimensions.clone()))
    }
}

impl<C, F> RightDot<F> for Tracer<C>
where
    C: StagingContext<Type = ArrayType>,
    F: Value<ArrayType>,
    C::Operation: SupportsRightDot<ArrayType, F>,
{
    #[inline]
    fn right_dot(self, factor: F, dimensions: &DotDimensionNumbers) -> Self {
        self.unary(C::Operation::right_dot_operation(factor, dimensions.clone()))
    }
}

/// Symbolic-zero-aware tangent left dot. `Zero.left_dot(_, _) -> Zero`.
impl<T, V, F> LeftDot<F> for crate::differentiation::Tangent<T, V>
where
    T: crate::types::Type,
    V: crate::programs::Value<T> + LeftDot<F>,
{
    #[inline]
    fn left_dot(self, factor: F, dimensions: &DotDimensionNumbers) -> Self {
        match self {
            Self::Zero(r#type) => Self::Zero(r#type),
            Self::Value(value) => Self::Value(value.left_dot(factor, dimensions)),
        }
    }
}

/// Symbolic-zero-aware tangent right dot. `Zero.right_dot(_, _) -> Zero`.
impl<T, V, F> RightDot<F> for crate::differentiation::Tangent<T, V>
where
    T: crate::types::Type,
    V: crate::programs::Value<T> + RightDot<F>,
{
    #[inline]
    fn right_dot(self, factor: F, dimensions: &DotDimensionNumbers) -> Self {
        match self {
            Self::Zero(r#type) => Self::Zero(r#type),
            Self::Value(value) => Self::Value(value.right_dot(factor, dimensions)),
        }
    }
}

/// Captured-factor "left dot" linear operation.
///
/// Represents the linear map `t ↦ dot(factor, t; dimensions)`. Emitted by [`DotOperation`]'s
/// JVP rule when the LHS primal is held constant, and by the transpose of
/// [`RightDotOperation`] (the adjoint of `t ↦ dot(t, factor; dimensions)`).
#[derive(Clone, Debug, PartialEq)]
pub struct LeftDotOperation<F: Value<ArrayType>> {
    /// Captured constant factor (the LHS of the underlying dot).
    factor: F,

    /// Dimension numbers of the underlying dot.
    dimensions: DotDimensionNumbers,
}

impl<F: Value<ArrayType>> LeftDotOperation<F> {
    /// Creates a new [`LeftDotOperation`].
    #[inline]
    pub fn new(factor: F, dimensions: DotDimensionNumbers) -> Self {
        Self { factor, dimensions }
    }

    /// Returns the captured constant factor.
    #[inline]
    pub fn factor(&self) -> &F {
        &self.factor
    }

    /// Returns the dimension numbers of the underlying dot.
    #[inline]
    pub fn dimensions(&self) -> &DotDimensionNumbers {
        &self.dimensions
    }
}

impl<F: Value<ArrayType>> Display for LeftDotOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<F: Value<ArrayType>> Operation<ArrayType> for LeftDotOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        "left_dot"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![dot_abstract(self.factor.r#type().as_ref(), &input_types[0], &self.dimensions, "left_dot")?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("dimensions", &self.dimensions))
    }
}

impl<F, V> InterpretableOperation<ArrayType, V> for LeftDotOperation<F>
where
    F: Value<ArrayType>,
    V: Value<ArrayType> + LeftDot<F>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // dot(factor, input; dimensions): factor is on the left.
        Ok(vec![inputs[0].clone().left_dot(self.factor.clone(), &self.dimensions)])
    }
}

/// Captured-factor "right dot" linear operation.
///
/// Represents the linear map `t ↦ dot(t, factor; dimensions)`. Emitted by [`DotOperation`]'s
/// JVP rule when the RHS primal is held constant, and by the transpose of
/// [`LeftDotOperation`] (the adjoint of `t ↦ dot(factor, t; dimensions)`).
#[derive(Clone, Debug, PartialEq)]
pub struct RightDotOperation<F: Value<ArrayType>> {
    /// Captured constant factor (the RHS of the underlying dot).
    factor: F,

    /// Dimension numbers of the underlying dot.
    dimensions: DotDimensionNumbers,
}

impl<F: Value<ArrayType>> RightDotOperation<F> {
    /// Creates a new [`RightDotOperation`].
    #[inline]
    pub fn new(factor: F, dimensions: DotDimensionNumbers) -> Self {
        Self { factor, dimensions }
    }

    /// Returns the captured constant factor.
    #[inline]
    pub fn factor(&self) -> &F {
        &self.factor
    }

    /// Returns the dimension numbers of the underlying dot.
    #[inline]
    pub fn dimensions(&self) -> &DotDimensionNumbers {
        &self.dimensions
    }
}

impl<F: Value<ArrayType>> Display for RightDotOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<F: Value<ArrayType>> Operation<ArrayType> for RightDotOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        "right_dot"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![dot_abstract(&input_types[0], self.factor.r#type().as_ref(), &self.dimensions, "right_dot")?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("dimensions", &self.dimensions))
    }
}

impl<F, V> InterpretableOperation<ArrayType, V> for RightDotOperation<F>
where
    F: Value<ArrayType>,
    V: Value<ArrayType> + RightDot<F>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // dot(input, factor; dimensions): factor is on the right.
        Ok(vec![inputs[0].clone().right_dot(self.factor.clone(), &self.dimensions)])
    }
}

/// Returns the lhs result axes of `dimensions` for an LHS of the supplied rank.
pub fn lhs_result_axes(dimensions: &DotDimensionNumbers, lhs_rank: usize) -> Vec<usize> {
    (0..lhs_rank)
        .filter(|axis| {
            !dimensions.lhs_batching_dimensions.contains(axis) && !dimensions.lhs_contracting_dimensions.contains(axis)
        })
        .collect()
}

/// Returns the rhs result axes of `dimensions` for an RHS of the supplied rank.
pub fn rhs_result_axes(dimensions: &DotDimensionNumbers, rhs_rank: usize) -> Vec<usize> {
    (0..rhs_rank)
        .filter(|axis| {
            !dimensions.rhs_batching_dimensions.contains(axis) && !dimensions.rhs_contracting_dimensions.contains(axis)
        })
        .collect()
}

/// Lifts a [`DotDimensionNumbers`] through one batching level.
///
/// Given the per-lane dimension numbers and the batch-axis positions of the two operands (each
/// optional — `None` indicates a lane-uniform operand), returns the dimension numbers that
/// describe the same contraction over the parent-physical (batched) operands. The mapping:
///
/// - When both operands are batched at positions `(k_lhs, k_rhs)`, the lifted op gains one new
///   batching dimension pair `(k_lhs, k_rhs)` at the front of the batching lists, and every
///   existing contracting / batching index `i` is shifted to `i + 1` if `i >= k_{lhs|rhs}`. The
///   new batch axis ends up at position `0` of the output (since batching dims are output-first).
/// - When neither operand is batched, the dimension numbers are unchanged.
/// - Mixed cases (exactly one operand batched) are not yet supported and return `Ok(None)` so
///   the caller can surface `MissingBatchingRule`.
pub fn lift_dot_dimensions(
    dimensions: &DotDimensionNumbers,
    lhs_batch_axis: Option<usize>,
    rhs_batch_axis: Option<usize>,
) -> Option<(DotDimensionNumbers, Option<usize>)> {
    let shift = |axes: &[usize], k: Option<usize>| -> Vec<usize> {
        match k {
            Some(k) => axes.iter().map(|i| if *i >= k { *i + 1 } else { *i }).collect(),
            None => axes.to_vec(),
        }
    };
    match (lhs_batch_axis, rhs_batch_axis) {
        (Some(k_lhs), Some(k_rhs)) => {
            let mut lhs_batching = vec![k_lhs];
            lhs_batching.extend(shift(&dimensions.lhs_batching_dimensions, Some(k_lhs)));
            let mut rhs_batching = vec![k_rhs];
            rhs_batching.extend(shift(&dimensions.rhs_batching_dimensions, Some(k_rhs)));
            Some((
                DotDimensionNumbers {
                    lhs_contracting_dimensions: shift(&dimensions.lhs_contracting_dimensions, Some(k_lhs)),
                    rhs_contracting_dimensions: shift(&dimensions.rhs_contracting_dimensions, Some(k_rhs)),
                    lhs_batching_dimensions: lhs_batching,
                    rhs_batching_dimensions: rhs_batching,
                },
                Some(0),
            ))
        }
        (None, None) => Some((dimensions.clone(), None)),
        _ => None,
    }
}

/// Lifts the dimension numbers of a captured-factor [`LeftDotOperation`] through one batching
/// level applied to its single (non-factor) input.
///
/// The factor stays lane-uniform; only the RHS operand of the underlying dot gains a new batch
/// dimension at position `k = t_batch_axis`. Existing RHS contracting / batching indices `i`
/// are shifted to `i + 1` when `i >= k`. The new axis at `k` becomes an RHS result axis (it
/// has no counterpart on the factor, so it can't be batching, and it isn't contracting).
///
/// Returns the lifted dimension numbers plus the output axis position. The output structure is
/// `[lhs_batching..., lhs_result..., rhs_result...]`, and the new batch axis ends up at
/// `lhs_batching_count + lhs_result_count + k_position_in_rhs_result`.
pub fn lift_left_dot_dimensions(
    dimensions: &DotDimensionNumbers,
    factor_rank: usize,
    t_batch_axis: Option<usize>,
) -> (DotDimensionNumbers, Option<usize>) {
    let Some(k) = t_batch_axis else {
        return (dimensions.clone(), None);
    };
    let shift = |axes: &[usize]| -> Vec<usize> { axes.iter().map(|i| if *i >= k { i + 1 } else { *i }).collect() };
    let lifted = DotDimensionNumbers {
        lhs_contracting_dimensions: dimensions.lhs_contracting_dimensions.clone(),
        rhs_contracting_dimensions: shift(&dimensions.rhs_contracting_dimensions),
        lhs_batching_dimensions: dimensions.lhs_batching_dimensions.clone(),
        rhs_batching_dimensions: shift(&dimensions.rhs_batching_dimensions),
    };
    let lhs_batching_count = dimensions.lhs_batching_dimensions.len();
    let lhs_result_count = factor_rank - lhs_batching_count - dimensions.lhs_contracting_dimensions.len();
    let rhs_non_result: std::collections::BTreeSet<usize> = dimensions
        .rhs_contracting_dimensions
        .iter()
        .copied()
        .chain(dimensions.rhs_batching_dimensions.iter().copied())
        .collect();
    let k_position_in_rhs_result = (0..k).filter(|i| !rhs_non_result.contains(i)).count();
    let output_axis = lhs_batching_count + lhs_result_count + k_position_in_rhs_result;
    (lifted, Some(output_axis))
}

/// Lifts the dimension numbers of a captured-factor [`RightDotOperation`] through one batching
/// level applied to its single (non-factor) input.
///
/// Symmetric to [`lift_left_dot_dimensions`]: the LHS operand of the underlying dot is the
/// non-factor input, so it gains the new batch dimension. The new axis at `k` becomes an LHS
/// result axis in the output.
pub fn lift_right_dot_dimensions(
    dimensions: &DotDimensionNumbers,
    t_batch_axis: Option<usize>,
) -> (DotDimensionNumbers, Option<usize>) {
    let Some(k) = t_batch_axis else {
        return (dimensions.clone(), None);
    };
    let shift = |axes: &[usize]| -> Vec<usize> { axes.iter().map(|i| if *i >= k { i + 1 } else { *i }).collect() };
    let lifted = DotDimensionNumbers {
        lhs_contracting_dimensions: shift(&dimensions.lhs_contracting_dimensions),
        rhs_contracting_dimensions: dimensions.rhs_contracting_dimensions.clone(),
        lhs_batching_dimensions: shift(&dimensions.lhs_batching_dimensions),
        rhs_batching_dimensions: dimensions.rhs_batching_dimensions.clone(),
    };
    let lhs_batching_count = dimensions.lhs_batching_dimensions.len();
    let lhs_non_result: std::collections::BTreeSet<usize> = dimensions
        .lhs_contracting_dimensions
        .iter()
        .copied()
        .chain(dimensions.lhs_batching_dimensions.iter().copied())
        .collect();
    let k_position_in_lhs_result = (0..k).filter(|i| !lhs_non_result.contains(i)).count();
    let output_axis = lhs_batching_count + k_position_in_lhs_result;
    (lifted, Some(output_axis))
}

/// Computes the dimension numbers for the adjoint of [`LeftDotOperation`]: maps the linear map
/// `t ↦ dot(factor, t; dimensions)`'s output cotangent back to a cotangent for `t`.
pub fn adjoint_dimensions_for_left_dot(dimensions: &DotDimensionNumbers, factor_rank: usize) -> DotDimensionNumbers {
    let n_batching = dimensions.lhs_batching_dimensions.len();
    let factor_result = lhs_result_axes(dimensions, factor_rank);
    let n_factor_result = factor_result.len();
    DotDimensionNumbers {
        lhs_batching_dimensions: dimensions.lhs_batching_dimensions.clone(),
        rhs_batching_dimensions: (0..n_batching).collect(),
        lhs_contracting_dimensions: factor_result,
        rhs_contracting_dimensions: (n_batching..(n_batching + n_factor_result)).collect(),
    }
}

/// Computes the dimension numbers for the adjoint of [`RightDotOperation`]: maps the linear map
/// `t ↦ dot(t, factor; dimensions)`'s output cotangent back to a cotangent for `t`.
pub fn adjoint_dimensions_for_right_dot(
    dimensions: &DotDimensionNumbers,
    factor_rank: usize,
    t_rank: usize,
) -> DotDimensionNumbers {
    let n_batching = dimensions.rhs_batching_dimensions.len();
    let factor_result = rhs_result_axes(dimensions, factor_rank);
    let t_result_count =
        t_rank - dimensions.rhs_batching_dimensions.len() - dimensions.rhs_contracting_dimensions.len();
    DotDimensionNumbers {
        lhs_batching_dimensions: (0..n_batching).collect(),
        rhs_batching_dimensions: dimensions.rhs_batching_dimensions.clone(),
        lhs_contracting_dimensions: ((n_batching + t_result_count)
            ..(n_batching + t_result_count + factor_result.len()))
            .collect(),
        rhs_contracting_dimensions: factor_result,
    }
}

impl<V: Value<ArrayType> + Dot, O> crate::differentiation::TransposableOperation<ArrayType, V, O>
    for LeftDotOperation<V>
where
    O: Operation<ArrayType> + SupportsLeftDot<ArrayType, V>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut crate::tracing::AbstractTracingContext<'transpose, ArrayType, V, O>,
        output_cotangents: &[crate::differentiation::Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<crate::differentiation::Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        let factor_rank = self.factor.r#type().as_ref().rank();
        let adjoint_dims = adjoint_dimensions_for_left_dot(&self.dimensions, factor_rank);
        match &output_cotangents[0] {
            crate::differentiation::Cotangent::Staged(cotangent) => {
                Ok(vec![crate::differentiation::Cotangent::Staged(
                    cotangent.clone().left_dot(self.factor.clone(), &adjoint_dims),
                )])
            }
            crate::differentiation::Cotangent::Zero => Ok(vec![crate::differentiation::Cotangent::Zero]),
        }
    }
}

impl<V: Value<ArrayType> + Dot, O> crate::differentiation::TransposableOperation<ArrayType, V, O>
    for RightDotOperation<V>
where
    O: Operation<ArrayType> + SupportsRightDot<ArrayType, V>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut crate::tracing::AbstractTracingContext<'transpose, ArrayType, V, O>,
        output_cotangents: &[crate::differentiation::Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<crate::differentiation::Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        let factor_rank = self.factor.r#type().as_ref().rank();
        let cotangent_rank = match &output_cotangents[0] {
            crate::differentiation::Cotangent::Staged(value) => value.r#type().as_ref().rank(),
            crate::differentiation::Cotangent::Zero => {
                return Ok(vec![crate::differentiation::Cotangent::Zero]);
            }
        };
        // t_rank = (batching + lhs_result) + lhs_contracting
        //        = (cotangent_rank - rhs_result_count) + lhs_contracting_count
        //        = cotangent_rank + 2 * rhs_contracting_count + rhs_batching_count - factor_rank.
        let t_rank = cotangent_rank
            + 2 * self.dimensions.rhs_contracting_dimensions.len()
            + self.dimensions.rhs_batching_dimensions.len()
            - factor_rank;
        let adjoint_dims = adjoint_dimensions_for_right_dot(&self.dimensions, factor_rank, t_rank);
        match &output_cotangents[0] {
            crate::differentiation::Cotangent::Staged(cotangent) => {
                Ok(vec![crate::differentiation::Cotangent::Staged(
                    cotangent.clone().right_dot(self.factor.clone(), &adjoint_dims),
                )])
            }
            crate::differentiation::Cotangent::Zero => Ok(vec![crate::differentiation::Cotangent::Zero]),
        }
    }
}

impl<F, V, RuleContext> crate::tracing_v2::batching::BatchableOperation<V, RuleContext> for LeftDotOperation<F>
where
    F: Value<ArrayType>,
    V: Value<ArrayType>,
    LeftDotOperation<F>: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        _context: &RuleContext,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (_, input_axes, _) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        let factor_rank = self.factor.r#type().as_ref().rank();
        let (lifted_dimensions, output_axis) = lift_left_dot_dimensions(&self.dimensions, factor_rank, input_axes[0]);
        let lifted_op = LeftDotOperation::new(self.factor.clone(), lifted_dimensions);
        crate::tracing_v2::batching::apply_with_axes(&lifted_op, inputs, &[output_axis])
    }
}

impl<F, V, RuleContext> crate::tracing_v2::batching::BatchableOperation<V, RuleContext> for RightDotOperation<F>
where
    F: Value<ArrayType>,
    V: Value<ArrayType>,
    RightDotOperation<F>: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        _context: &RuleContext,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (_, input_axes, _) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        let (lifted_dimensions, output_axis) = lift_right_dot_dimensions(&self.dimensions, input_axes[0]);
        let lifted_op = RightDotOperation::new(self.factor.clone(), lifted_dimensions);
        crate::tracing_v2::batching::apply_with_axes(&lifted_op, inputs, &[output_axis])
    }
}

/// Generalized N-dimensional dot-product helper.
///
/// Implements StableHLO `dot_general` semantics over a flat row-major payload and an explicit
/// shape. Used by value-level [`Dot`] implementations for `Vec`-backed array types.
///
/// # Parameters
///
///   - `lhs`: Flat row-major payload of the left operand.
///   - `lhs_shape`: Shape of the left operand.
///   - `rhs`: Flat row-major payload of the right operand.
///   - `rhs_shape`: Shape of the right operand.
///   - `dimensions`: Contracting and batching dimension numbers.
///   - `accumulator_init`: Zero value of the accumulator type (called once per output element).
///   - `multiply_accumulate`: Accumulator update — `accumulator + lhs_value * rhs_value`.
pub fn dot_general_evaluate<T, FInit, FAcc>(
    lhs: &[T],
    lhs_shape: &[usize],
    rhs: &[T],
    rhs_shape: &[usize],
    dimensions: &DotDimensionNumbers,
    accumulator_init: FInit,
    multiply_accumulate: FAcc,
) -> (Vec<T>, Vec<usize>)
where
    T: Clone,
    FInit: Fn() -> T,
    FAcc: Fn(T, &T, &T) -> T,
{
    let lhs_batching = dimensions.lhs_batching_dimensions.as_slice();
    let rhs_batching = dimensions.rhs_batching_dimensions.as_slice();
    let lhs_contracting = dimensions.lhs_contracting_dimensions.as_slice();
    let rhs_contracting = dimensions.rhs_contracting_dimensions.as_slice();

    let lhs_result: Vec<usize> = (0..lhs_shape.len())
        .filter(|axis| !lhs_batching.contains(axis) && !lhs_contracting.contains(axis))
        .collect();
    let rhs_result: Vec<usize> = (0..rhs_shape.len())
        .filter(|axis| !rhs_batching.contains(axis) && !rhs_contracting.contains(axis))
        .collect();

    let batching_extents: Vec<usize> = lhs_batching.iter().map(|axis| lhs_shape[*axis]).collect();
    let lhs_result_extents: Vec<usize> = lhs_result.iter().map(|axis| lhs_shape[*axis]).collect();
    let rhs_result_extents: Vec<usize> = rhs_result.iter().map(|axis| rhs_shape[*axis]).collect();
    let contracting_extents: Vec<usize> = lhs_contracting.iter().map(|axis| lhs_shape[*axis]).collect();

    let output_shape: Vec<usize> = batching_extents
        .iter()
        .copied()
        .chain(lhs_result_extents.iter().copied())
        .chain(rhs_result_extents.iter().copied())
        .collect();
    let output_count: usize = output_shape.iter().product();
    let mut output = Vec::with_capacity(output_count);
    if output_count == 0 {
        return (output, output_shape);
    }

    let lhs_strides = row_major_strides(lhs_shape);
    let rhs_strides = row_major_strides(rhs_shape);
    let mut lhs_index = vec![0usize; lhs_shape.len()];
    let mut rhs_index = vec![0usize; rhs_shape.len()];

    for_each_multi_index(batching_extents.as_slice(), |batching_index| {
        for (slot, axis) in lhs_batching.iter().enumerate() {
            lhs_index[*axis] = batching_index[slot];
        }
        for (slot, axis) in rhs_batching.iter().enumerate() {
            rhs_index[*axis] = batching_index[slot];
        }
        for_each_multi_index(lhs_result_extents.as_slice(), |lhs_result_index| {
            for (slot, axis) in lhs_result.iter().enumerate() {
                lhs_index[*axis] = lhs_result_index[slot];
            }
            for_each_multi_index(rhs_result_extents.as_slice(), |rhs_result_index| {
                for (slot, axis) in rhs_result.iter().enumerate() {
                    rhs_index[*axis] = rhs_result_index[slot];
                }
                let mut accumulator = accumulator_init();
                for_each_multi_index(contracting_extents.as_slice(), |contracting_index| {
                    for (slot, axis) in lhs_contracting.iter().enumerate() {
                        lhs_index[*axis] = contracting_index[slot];
                    }
                    for (slot, axis) in rhs_contracting.iter().enumerate() {
                        rhs_index[*axis] = contracting_index[slot];
                    }
                    let lhs_flat = flat_index(&lhs_index, &lhs_strides);
                    let rhs_flat = flat_index(&rhs_index, &rhs_strides);
                    accumulator = multiply_accumulate(accumulator.clone(), &lhs[lhs_flat], &rhs[rhs_flat]);
                });
                output.push(accumulator);
            });
        });
    });

    (output, output_shape)
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

fn flat_index(multi_index: &[usize], strides: &[usize]) -> usize {
    multi_index.iter().zip(strides.iter()).map(|(index, stride)| index * stride).sum()
}

fn for_each_multi_index(extents: &[usize], mut action: impl FnMut(&[usize])) {
    if extents.is_empty() {
        action(&[]);
        return;
    }
    let mut index = vec![0usize; extents.len()];
    loop {
        action(&index);
        let mut axis = extents.len();
        while axis > 0 {
            axis -= 1;
            index[axis] += 1;
            if index[axis] < extents[axis] {
                break;
            }
            index[axis] = 0;
            if axis == 0 {
                return;
            }
        }
    }
}
