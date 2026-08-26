use std::fmt::Display;
use std::ops::{Mul as StandardMul, Sub as StandardSub};

use crate::arrays::{ArrayBatch, ArrayBatching, ArrayType, DataType, RaggedArrayBatchingPolicy, RaggedMaskIdentity};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    ElementwiseDerivativeAlignment,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_transposable_operation};
use crate::operations::manipulation::broadcasting::Broadcast;
use crate::operations::math::exp::Exp;
use crate::operations::math::log_add_exp::{is_log_add_exp_identity_data_type, log_add_exp_identity_data_type_error};
use crate::operations::math::reduce::{
    Reduce, ReductionKind, batch_reducing_operation, lift_reduce_axes, output_to_input_axis_map, reduce_shape_abstract,
};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`LogSumExpOperation`].
pub const LOG_SUM_EXP_OPERATION_NAME: &str = "log_sum_exp";

/// Primitive representing one numerically stable `log(sum(exp(x)))` over a set of array axes.
///
/// [`LogSumExpOperation`] collapses the input array along [`axes`](Self::axes), following the same axis conventions as
/// [`ReduceOperation`](crate::ReduceOperation): the reduced axes are removed from the output shape and the remaining
/// axes keep their order.
///
/// The result is computed through the guarded construction pinned to JAX's `logsumexp` (`jax/_src/ops/special.py`)
/// rather than by exponentiating directly, which would overflow for any input above about `709` in double precision:
///
/// ```text
/// m      = reduce_max(x)                      // with a -∞ initial value
/// safe_m = select(isfinite(m), m, 0)
/// result = log(reduce_sum(exp(x - safe_m))) + safe_m
/// ```
///
/// The `safe_m` substitution is what the guard buys. Shifting by a raw maximum of `-∞` (the identity of a maximum, and
/// therefore the result for an all-`-∞` slice or an empty reduction) would compute `-∞ - -∞ = NaN`; substituting zero
/// there leaves `log(0) + 0 = -∞`, which is the correct value of an empty or all-zero sum of exponentials. A maximum
/// of `+∞` is guarded the same way, and a NaN input propagates as usual.
///
/// Only real floating-point operands are supported, and among those only the formats whose lowest value acts as the
/// identity of the inner sum of exponentials. That sentinel is not merely a kernel detail: the ragged batching rule
/// below writes it over the padding of a reduced bounded axis, and the padded positions of a slice fold as many
/// copies of it as the axis is padded by, a count no type carries. [`DataType::F8E8M0FNU`] fails that outright,
/// having neither the zero the inner sum's identity needs nor a sign at all, and [`DataType::F6E2M3FN`],
/// [`DataType::F4E2M1FN`], [`DataType::F8E4M3B11FNUZ`], and [`DataType::F6E3M2FN`] fail it by drift, their lowest
/// values holding across only one, two, two, and seven folded copies. All five are rejected rather than accepted into
/// a program whose masked slices would quietly read high. This primitive is the *unweighted, unmasked* subset of
/// [`jax.nn.logsumexp`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.logsumexp.html): the `b` weights, the
/// `where` mask, the sign return value, and complex inputs are explicit non-goals.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogSumExpOperation {
    /// Axes over which the exponentials are summed.
    axes: Vec<usize>,
}

impl LogSumExpOperation {
    /// Creates a new [`LogSumExpOperation`] reducing along `axes`. The input shape is not part of the operation
    /// payload: it is recoverable from the staged input type wherever a rule needs it.
    #[inline]
    pub fn new(axes: Vec<usize>) -> Self {
        Self { axes }
    }

    /// Returns the axes reduced by this operation.
    #[inline]
    pub fn axes(&self) -> &[usize] {
        self.axes.as_slice()
    }
}

impl Display for LogSumExpOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for LogSumExpOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        LOG_SUM_EXP_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![log_sum_exp_abstract(&input_types[0], self.axes.as_slice())?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("axes", format_args!("{:?}", self.axes)))
    }
}

/// Returns the output [`ArrayType`] produced by a [`LogSumExpOperation`] over `axes`.
///
/// The axis geometry — validation, the removal of the reduced axes from the output shape, and the
/// [`Sharding`](crate::arrays::Sharding) rule — is the reduction family's, and is delegated to
/// [`reduce_shape_abstract`]. What this primitive adds is its element data-type domain: real floating-point formats
/// only (the exponential and the logarithm have no meaning for the integer, Boolean, token, or structural-zero
/// element types, and complex support is an explicit non-goal), and, among those, only the formats whose lowest value
/// is an identity of the inner sum of exponentials across every copy of itself that a padded slice folds, which the
/// ragged batching rule writes over the padding of a reduced bounded axis. [`DataType::F8E8M0FNU`],
/// [`DataType::F6E2M3FN`], [`DataType::F4E2M1FN`], [`DataType::F8E4M3B11FNUZ`], and [`DataType::F6E3M2FN`] fail that
/// second requirement and are therefore rejected rather than accepted into a program that would only read high once
/// it evaluates.
///
/// The eager kernel and the operation's type inference share this rule, so a directly invoked [`LogSumExp`]
/// capability rejects exactly what a staged program rejects.
pub(crate) fn log_sum_exp_abstract(input: &ArrayType, axes: &[usize]) -> Result<ArrayType, TypeError> {
    reduce_shape_abstract(input, axes, LOG_SUM_EXP_OPERATION_NAME, validate_log_sum_exp_data_type)
}

/// Validates the element data-type domain documented on [`log_sum_exp_abstract`], which the operation's type
/// inference and its eager entry points share.
pub(crate) fn validate_log_sum_exp_data_type(data_type: DataType) -> Result<(), TypeError> {
    // The domain is exactly the one `cumulative_log_sum_exp` accepts, and for the same reason: both operations can be
    // asked to write the format's lowest value over ragged padding, so both need that sentinel to fold as an
    // identity. The predicate and its diagnostic are therefore shared rather than restated here.
    match is_log_add_exp_identity_data_type(data_type) {
        true => Ok(()),
        false => Err(TypeError::invalid(log_add_exp_identity_data_type_error(LOG_SUM_EXP_OPERATION_NAME, data_type))),
    }
}

impl<C: Domain<Type = ArrayType, Value: LogSumExp>> InterpretableOperation<C> for LogSumExpOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].log_sum_exp(self.axes.as_slice())?])
    }
}

// Partial evaluation defers to the default fold-or-residualize behavior of `Program::partially_evaluate`.
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for LogSumExpOperation where
    C::Operation: From<LogSumExpOperation>
{
}

// The reduced axes are expressed in the per-item coordinate system, so the rule lifts them past the inserted batch
// dimension with `lift_reduce_axes` and hands the lifted operation to the shared axis-collapsing skeleton of
// `batch_reducing_operation`. The identity written over the padding of a reduced ragged axis is negative infinity,
// because `exp(-∞) = 0` is the additive identity of the inner sum.
impl<C: Context<Type = ArrayType>, P: RaggedArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for LogSumExpOperation
where
    LogSumExpOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let Some(batch_axis) = inputs[0].batch_axis_position() else {
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into());
        };
        let (lifted_axes, output_axis) = lift_reduce_axes(self.axes.as_slice(), batch_axis);
        let lifted_operation = Self::new(lifted_axes);
        batch_reducing_operation(
            context,
            &lifted_operation,
            &inputs[0],
            lifted_operation.axes(),
            output_axis,
            |input| P::mask_identity_input(context, input, lifted_operation.axes(), RaggedMaskIdentity::Lowest),
        )
    }
}

// The partial derivative of `log(sum(exp(x)))` with respect to each reduced element is that element's softmax weight,
// so the output tangent is the softmax-weighted sum of the operand tangents over the reduced axes:
// `reduce_sum(exp(x - broadcast(lse(x))) · Δx)`.
//
// The weights are staged capture-free from the operand primal and the rule's own primal output — a `broadcast` of the
// result back over the reduced axes, a `sub`, and an `exp` — so no residual factor is captured. The subtraction is
// stable for ordinary inputs because `lse(x) ≥ max(x)` makes every exponent non-positive. The one input for which it
// is not is an all-`-∞` slice, whose result is `-∞` and whose weights are therefore `exp(-∞ - -∞) = NaN`; that
// gradient is a NaN edge case here exactly as it is in JAX, and it is left as is rather than papered over.
//
// The composite array universe reaches this rule through the default projected fall-through of
// `MemberDifferentiableOperation`, which is correct for the static shapes this rule supports: it neither broadcasts
// several operands into one result (so no operand ever needs replication) nor observes a reduced extent as a value. A
// runtime-sized reduced axis is rejected by the `broadcast` staged above, which carries its complete output geometry
// as payload metadata.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for LogSumExpOperation
where
    C::Value: LogSumExp
        + Broadcast
        + Exp
        + Reduce
        + StandardSub<Output = C::Value>
        + StandardMul<Output = C::Value>
        + ElementwiseDerivativeAlignment<ArrayType>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal_input = inputs[0].primal();
        let primal = primal_input.log_sum_exp(self.axes.as_slice())?;
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
            MaybeZero::Value(input_tangent) => {
                let input_type = primal_input.r#type().into_owned();
                let output_axes = output_to_input_axis_map(input_type.rank(), self.axes.as_slice());
                let broadcast_primal = primal.broadcast(input_type, output_axes.as_slice())?;
                let weights = (primal_input.clone() - broadcast_primal).exp()?;
                let weights = weights.align_tangent(input_tangent.r#type().as_ref(), input_tangent)?;
                let weighted = weights * input_tangent.clone();
                MaybeZero::Value(weighted.reduce(self.axes.as_slice(), ReductionKind::Sum))
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

// `log_sum_exp` is not linear in its operand, so it has no primitive transposition rule. Reverse-mode differentiation
// remains available by transposing the linear operations that the forward-mode rule above stages.
impl_non_transposable_operation!(LogSumExpOperation);

/// Value-level `log(sum(exp(x)))` capability.
///
/// [`LogSumExp`] is the receiver-style entry point for staging or executing a [`LogSumExpOperation`]: it reduces the
/// receiver along `axes` with the numerically stable construction documented on that operation, returning a value
/// whose rank is `self.rank() - axes.len()`. Reducing along no axes is the identity and returns the receiver
/// unchanged, matching `log(exp(x)) = x`; the element data type is still validated, so the shortcut accepts exactly
/// the operands a staged program accepts.
pub trait LogSumExp: Sized {
    /// Computes the stable `log(sum(exp(self)))` of `self` over `axes`.
    fn log_sum_exp(&self, axes: &[usize]) -> Result<Self, ProgramError>;
}

// Any context-carrying value reduces by binding a `LogSumExpOperation` through its own context. The
// `From<LogSumExpOperation>` bound makes this disjoint from the eager value types (whose context operation is
// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> LogSumExp for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<LogSumExpOperation>,
{
    #[inline]
    fn log_sum_exp(&self, axes: &[usize]) -> Result<Self, ProgramError> {
        // Reducing along no axes is the identity, but only for the operands this primitive accepts at all, so the
        // element data type is validated before the shortcut is taken.
        if axes.is_empty() {
            validate_log_sum_exp_data_type(self.r#type().data_type())?;
            return Ok(self.clone());
        }
        Ok(self
            .dispatch_domain()
            .bind(LogSumExpOperation::new(axes.to_vec()), Vec::new(), &[self.clone()])?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayBatchingPolicy;
    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, Dimension, DimensionBounds, DimensionType, DimensionVariable,
        LogicalMesh, MeshAxis, MeshAxisType, RaggedAxis, Shape, Sharding, ShardingDimension,
    };
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ValueProjection};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_log_sum_exp_abstract() {
        // The reduced axes are dropped and the remaining axes keep their order.
        let input = ArrayType::new_static(DataType::F64, [2, 3, 4]);
        assert_eq!(log_sum_exp_abstract(&input, &[1]), Ok(ArrayType::new_static(DataType::F64, [2, 4])));
        assert_eq!(log_sum_exp_abstract(&input, &[0, 2]), Ok(ArrayType::new_static(DataType::F64, [3])));
        assert_eq!(log_sum_exp_abstract(&input, &[]), Ok(input.clone()));

        // Axis validation mirrors the reduction family's.
        assert_eq!(
            log_sum_exp_abstract(&input, &[3]),
            Err(TypeError::invalid("`log_sum_exp` axis 3 is out of bounds for rank 3".to_string())),
        );
        assert_eq!(
            log_sum_exp_abstract(&input, &[1, 1]),
            Err(TypeError::invalid("`log_sum_exp` contains duplicate axis 1".to_string())),
        );

        // Only real floating-point payloads have the exponential and logarithm this primitive is built from.
        for data_type in [DataType::I32, DataType::Boolean, DataType::C64, DataType::Token, DataType::Zero] {
            assert_eq!(
                log_sum_exp_abstract(&ArrayType::new_static(data_type, [2, 3]), &[1]),
                Err(TypeError::invalid(format!(
                    "`log_sum_exp` requires real floating-point inputs but got {data_type}"
                ))),
            );
        }

        // `f8e8m0fnu` is floating-point but encodes bare positive exponents, so it has neither the zero the inner sum
        // needs nor the negative infinity an empty reduction returns. It is rejected here rather than in the kernel.
        assert_eq!(
            log_sum_exp_abstract(&ArrayType::new_static(DataType::F8E8M0FNU, [2, 3]), &[1]),
            Err(TypeError::invalid(
                "`log_sum_exp` requires a floating-point format that represents zero and negative infinity but got \
                 f8e8m0fnu"
                    .to_string(),
            )),
        );

        // Four more formats do have a sentinel whose exponential underflows to zero, but one that holds across too
        // few copies of itself: the ragged batching rule below masks padding with the format's lowest value, and a
        // padded slice folds as many copies as the axis is padded by, which lifts `-7.5` at two copies, `-6` and
        // `-30` at three, and `-28` at eight.
        for data_type in [DataType::F6E2M3FN, DataType::F4E2M1FN, DataType::F8E4M3B11FNUZ, DataType::F6E3M2FN] {
            assert_eq!(
                log_sum_exp_abstract(&ArrayType::new_static(data_type, [2, 3]), &[1]),
                Err(TypeError::invalid(format!(
                    "`log_sum_exp` requires a floating-point format whose lowest value is a `log_add_exp` identity \
                     but got {data_type}"
                ))),
            );
        }

        // Reducing over a sharded dimension deletes its entry without error (the partitioner owns the collective),
        // and the surviving dimension keeps its sharding.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = ArrayType::new_static(DataType::F64, [2, 3])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            log_sum_exp_abstract(&sharded, &[0]),
            Ok(ArrayType::new_static(DataType::F64, [3])
                .with_sharding(Sharding::new(mesh, vec![ShardingDimension::replicated()]).unwrap())
                .unwrap()),
        );
    }

    #[test]
    fn test_log_sum_exp_operation_type_inference() {
        check_operation_type_inference!(
            operation = LogSumExpOperation::new(vec![1]),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [3, 2])],
                    output_types = [ArrayType::new_static(DataType::F32, [3])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F64, [3])],
                    error = "`log_sum_exp` axis 1 is out of bounds for rank 1",
                },
            ],
        );
    }

    #[test]
    fn test_log_sum_exp_operation_rendering() {
        assert_eq!(LogSumExpOperation::new(vec![0, 2]).to_string(), "log_sum_exp [axes=[0, 2]]");
        assert_eq!(LogSumExpOperation::new(vec![1]).axes(), &[1]);
    }

    #[test]
    fn test_log_sum_exp_over_eager_arrays() {
        // The expected values below spell out the guarded construction the primitive documents (shift by the safe
        // maximum, sum the exponentials, take the logarithm, add the shift back) so that they pin that construction
        // rather than an equivalent-in-exact-arithmetic alternative.
        let values = Array::vector(vec![1.0, 2.0, 3.0]);
        let expected = ((1.0f64 - 3.0).exp() + (2.0f64 - 3.0).exp() + 1.0).ln() + 3.0;
        assert_eq!(values.log_sum_exp(&[0]), Ok(Array::scalar(expected)));
        assert_abs_diff_eq!(expected, (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln(), epsilon = 1e-12);

        // Reducing along no axes is the identity, matching `log(exp(x)) = x`, but only for the operands the staged
        // operation accepts: the shortcut still validates the element data type.
        assert_eq!(values.log_sum_exp(&[]), Ok(values.clone()));
        assert_eq!(
            Array::vector(vec![1_i32, 2]).log_sum_exp(&[]),
            Err(ProgramError::Type(TypeError::invalid(
                "`log_sum_exp` requires real floating-point inputs but got i32".to_string(),
            ))),
        );

        // Two equal operands add exactly `log(2)`, at any magnitude: the shift keeps the exponentials at one where
        // the naive composition would already have overflowed.
        assert_eq!(Array::vector(vec![0.0, 0.0]).log_sum_exp(&[0]), Ok(Array::scalar(std::f64::consts::LN_2)));
        assert_eq!(
            Array::vector(vec![1000.0, 1000.0]).log_sum_exp(&[0]),
            Ok(Array::scalar(1000.0 + std::f64::consts::LN_2)),
        );
        assert!((1000.0f64.exp() + 1000.0f64.exp()).ln().is_infinite());

        // The guard's reason to exist: an all-`-∞` slice and an empty reduction both pin to `-∞` (`log(0) + 0`)
        // instead of the `-∞ - -∞ = NaN` that shifting by the raw maximum would produce.
        assert_eq!(
            Array::vector(vec![f64::NEG_INFINITY, f64::NEG_INFINITY]).log_sum_exp(&[0]),
            Ok(Array::scalar(f64::NEG_INFINITY)),
        );
        assert_eq!(
            Array::new(ArrayType::new_static(DataType::F64, [0]), Vec::new()).unwrap().log_sum_exp(&[0]),
            Ok(Array::scalar(f64::NEG_INFINITY)),
        );

        // A `+∞` element saturates the result, and NaN propagates.
        assert_eq!(Array::vector(vec![1.0, f64::INFINITY]).log_sum_exp(&[0]), Ok(Array::scalar(f64::INFINITY)));
        assert!(Array::vector(vec![1.0, f64::NAN]).log_sum_exp(&[0]).unwrap().to_f64s()[0].is_nan());

        // Reducing one axis of a matrix leaves the other, in order.
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let row = |maximum: f64, values: [f64; 3]| {
            (values.iter().map(|value| (value - maximum).exp()).sum::<f64>()).ln() + maximum
        };
        assert_eq!(
            matrix.log_sum_exp(&[1]),
            Ok(Array::vector(vec![row(3.0, [1.0, 2.0, 3.0]), row(6.0, [4.0, 5.0, 6.0])])),
        );

        // Validation errors are reported rather than panicking.
        assert_eq!(
            values.log_sum_exp(&[1]),
            Err(ProgramError::Type(
                TypeError::invalid("`log_sum_exp` axis 1 is out of bounds for rank 1".to_string(),)
            )),
        );
        assert_eq!(
            Array::vector(vec![1_i32, 2]).log_sum_exp(&[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "`log_sum_exp` requires real floating-point inputs but got i32".to_string(),
            ))),
        );
    }

    #[test]
    fn test_log_sum_exp_operation_batches_replicated_input_as_pass_through() {
        check_operation_batching!(
            @exact,
            operation = LogSumExpOperation::new(vec![0]),
            axis_size = 2,
            cases = [{
                inputs = [(@replicated, Array::vector(vec![0.0, 0.0]))],
                outputs = [(@replicated, Array::scalar(std::f64::consts::LN_2))],
            }],
        );
    }

    #[test]
    fn test_log_sum_exp_operation_batches_along_the_shifted_axis() {
        // Physical input is [2 batch items, 2 columns] mapped at axis 0, so the per-item axis 0 reduces physical
        // axis 1 and each batch item is reduced independently.
        check_operation_batching!(
            @approx(epsilon = 1e-12),
            operation = LogSumExpOperation::new(vec![0]),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![0.0, 0.0, 1000.0, 1000.0]))],
                outputs = [(@mapped(
                    axis = 0
                ), Array::vector(vec![std::f64::consts::LN_2, 1000.0 + std::f64::consts::LN_2]))],
            }],
        );
    }

    #[test]
    fn test_log_sum_exp_operation_consumes_a_reduced_ragged_axis() {
        // Static array batching cannot neutralize ragged padding, and says so rather than summing the padding's
        // exponentials into the live result.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![0.0_f32; 6]), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 3]), variable.clone(), vec![0])])
            .unwrap();
        assert_eq!(
            LogSumExpOperation::new(vec![0]).batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[input],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "static array batching cannot identity-mask bounded ragged dimension `length` on axis 1 \
                          with `Lowest`"
                    .to_string(),
            }),
        );

        // The composite dynamic policy can, and stages the mask ahead of the reduction: the padded positions of the
        // reduced axis are selected away in favor of negative infinity, whose exponential is the sum's zero identity.
        // The ragged axis is then genuinely consumed, so it leaves the result and is reported as evidence.
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::new(items.clone()).into());
        let packed = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(items.clone()), Dimension::Static(3)]))
                .into(),
        );
        let extents = trace.input(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(items)])).into());
        let context = BatchingContext::<_, ArrayBatching<DynamicArrayBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace.clone()),
            batch_extent,
        );
        let input = ArrayBatch::new(packed.into_projected().unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents.into_projected().unwrap(), variable.clone(), vec![0])])
            .unwrap();
        // The per-item reduced axis 0 is the packed axis 1 that carries the ragged extents.
        let (outputs, evidence) =
            LogSumExpOperation::new(vec![0]).batch(&context, &EmptyRegionDriver, &[input]).unwrap().into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert!(outputs[0].ragged_axes().is_empty());
        assert_eq!(evidence, vec![variable]);

        let output_id = outputs.into_iter().next().unwrap().into_value().into_value().atom_id().unwrap();
        drop(context);
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output_id],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items, 3], %2:i32[items] .
                let %3:dimension<items ∈ [1, 9)> = dimension_size [axis=0] %1
                    %4:dimension<3> = constant [value=3]
                    %5:i32[3] = iota [type=i32[3], dimension=0]
                    %6:i32[items, 3] = broadcast [output_axes=[1]] %5 %3 %4
                    %7:i32[items, 3] = broadcast [output_axes=[0]] %2 %3 %4
                    %8:bool[items, 3] = compare [direction=LessThan] %6 %7
                    %9:f32[] = constant [value=-inf]
                    %10:f32[items, 3] = broadcast [output_axes=[]] %9 %3 %4
                    %11:f32[items, 3] = select %8 %1 %10
                    %12:f32[items] = log_sum_exp [axes=[1]] %11
                in (%12)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_log_sum_exp_operation_differentiation() {
        // The tangent is the softmax-weighted sum of the operand tangents over the reduced axes.
        let primals = [1.0f64, 2.0, 3.0];
        let tangents = [0.5f64, -1.5, 2.0];
        let output = ((1.0f64 - 3.0).exp() + (2.0f64 - 3.0).exp() + 1.0).ln() + 3.0;
        let tangent = primals
            .iter()
            .zip(tangents.iter())
            .map(|(primal, tangent)| (primal - output).exp() * tangent)
            .sum::<f64>();
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = LogSumExpOperation::new(vec![0]),
            cases = [{
                primals = [Array::vector(primals.to_vec())],
                tangents = [Array::vector(tangents.to_vec())],
                primal_outputs = [Array::scalar(output)],
                tangent_outputs = [Array::scalar(tangent)],
                jvp = indoc! {"
                    lambda %0:f64[3], %1:f64[3] .
                    let %2:f64[] = log_sum_exp [axes=[0]] %0
                        %3:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %2
                        %4:f64[3] = sub %0 %3
                        %5:f64[3] = exp %4
                        %6:f64[3] = mul %5 %1
                        %7:f64[] = reduce_sum [axes=[0]] %6
                    in (%2, %7)
                "},
            }],
        );
    }

    #[test]
    fn test_log_sum_exp_operation_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = LogSumExpOperation::new(vec![0]),
            inputs = [Array::vector(vec![0.0, 0.0])],
            expected = Array::scalar(std::f64::consts::LN_2),
        );
    }

    #[test]
    fn test_log_sum_exp_operation_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = LogSumExpOperation::new(vec![0]),
            input_types = [ArrayType::new_static(DataType::F64, [3])],
        );
    }
}
