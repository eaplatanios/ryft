use std::collections::BTreeSet;
use std::ops::Mul as StandardMul;

use crate::arrays::ArrayType;
use crate::differentiation::{DifferentiableType, ElementwiseDerivativeAlignment};
use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};
use crate::operations::ElementwiseOperation;
use crate::programs::{ProgramError, TypeError};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`MulOperation`].
pub const MUL_OPERATION_NAME: &str = "mul";

/// Infers multiplication output array types using its bilinear reduction-state rule.
fn infer_mul_output_array_types(input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
    // Multiplication is bilinear, so its output sharding combines the operands' unreduced/reduced state by the
    // bilinear rule rather than the congruent rule used by generic elementwise broadcasting. The reduction state
    // is combined independently of per-dimension placement, so the placement is broadcast with that state stripped
    // and the recomputed state is reattached afterward.
    let stripped = [input_types[0].without_reduction_axes(), input_types[1].without_reduction_axes()];
    let output = MulOperation::<ArrayType>::new().infer_elementwise_broadcast_type(&stripped)?;
    let left_unreduced = input_types[0].unreduced_axes();
    let left_reduced = input_types[0].reduced_axes();
    let right_unreduced = input_types[1].unreduced_axes();
    let right_reduced = input_types[1].reduced_axes();

    // An operand unreduced over some axes is a partial sum still awaiting an all-reduce over them. The product of
    // two partial sums is not a partial sum, so at most one operand may be unreduced. The other must then be
    // reduced over exactly those axes, and the product stays unreduced over them (its matching reduced marker is
    // consumed when the reduced set is computed below).
    let output_unreduced = match (left_unreduced.is_empty(), right_unreduced.is_empty()) {
        (false, false) => {
            return Err(TypeError::invalid(format!(
                "'{MUL_OPERATION_NAME}' cannot multiply two operands that are both unreduced",
            )));
        }
        (false, true) => {
            if left_unreduced != right_reduced {
                return Err(TypeError::invalid(format!(
                    "'{MUL_OPERATION_NAME}' requires the second operand to be reduced over the axes \
                             the first is unreduced over",
                )));
            }
            left_unreduced.clone()
        }
        (true, false) => {
            if right_unreduced != left_reduced {
                return Err(TypeError::invalid(format!(
                    "'{MUL_OPERATION_NAME}' requires the first operand to be reduced over the axes \
                             the second is unreduced over",
                )));
            }
            right_unreduced.clone()
        }
        (true, true) => BTreeSet::new(),
    };

    // Plain reduced axes must agree. The only one-sided reduced marker that is valid is the marker consumed by the
    // partial-sum-times-reduced case above; a one-sided marker without a matching unreduced operand would
    // incorrectly propagate reduction state from only one input.
    let mut output_reduced = if left_reduced == right_reduced {
        left_reduced.clone()
    } else if left_reduced.is_empty() && right_reduced == &output_unreduced {
        right_reduced.clone()
    } else if right_reduced.is_empty() && left_reduced == &output_unreduced {
        left_reduced.clone()
    } else {
        return Err(TypeError::invalid(format!("'{MUL_OPERATION_NAME}' operands must be reduced over the same axes")));
    };
    output_reduced.retain(|axis| !output_unreduced.contains(axis));

    // A non-empty result reduction state means some operand was sharded, so the broadcast output (already stripped
    // of reduction axes) carries a sharding onto which the recomputed state is reattached; otherwise it is already
    // correct as is.
    if output_unreduced.is_empty() && output_reduced.is_empty() {
        return Ok(vec![output]);
    }
    let sharding = output.sharding().expect("bilinear reduction state implies a sharded output");
    let rebuilt = sharding
        .clone()
        .with_unreduced_axes(output_unreduced)
        .map_err(|error| TypeError::invalid(error.to_string()))?
        .with_reduced_axes(output_reduced)
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    Ok(vec![output.with_sharding(rebuilt).map_err(|error| TypeError::invalid(error.to_string()))?])
}

define_elementwise_operation!(
    @binary
    /// [`Operation`] that multiplies two numeric values elementwise, promoting their element types and broadcasting
    /// their shapes. Its bilinear reduction-state rule permits one unreduced operand only when the other operand is
    /// reduced over exactly the same mesh axes.
    MulOperation, MUL_OPERATION_NAME,
    Mul, mul,
    infer_array_types = infer_mul_output_array_types,
    check_data_types = [@numeric],
);

// Transposition accepts exactly one linear operand and scales its output cotangent by the other, known operand. The
// contribution is unbroadcast to the linear operand's exact cotangent type, while the known operand receives a
// structural zero.
impl_differentiable_elementwise_operation! {
    @binary
    MulOperation,
    jvp<C> where C::Value: StandardMul<Output = C::Value> {
        |(_, left_tangent), (right, _)| right * left_tangent;
        |(left, _), (_, right_tangent)| left * right_tangent;
    },
    transpose<V, O>
    where
        V::Type: DifferentiableType,
        O: From<MulOperation<V::Type>>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
    {
        [left = @linear, right = @known] =>
            |output_cotangent| right.binary(output_cotangent, MulOperation::new());
        [left = @known, right = @linear] =>
            |output_cotangent| left.binary(output_cotangent, MulOperation::new());
    },
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise multiplication capability. [`Mul`] is the fallible Ryft counterpart to
    /// [`std::ops::Mul`] that [`MulOperation`] interprets through, surfacing a [`ProgramError`] when something goes
    /// wrong, instead of panicking. Value types additionally provide [`std::ops::Mul`] as ergonomic (albeit panicking)
    /// sugar layered on top of this capability.
    Mul,
    /// Multiplies `self` by `right`, returning a [`ProgramError`] if something goes wrong.
    mul(right),
    MulOperation,
);

define_tracer_operator!(@binary std::ops::Mul, mul, capability = Mul, method = mul);

/// Implements [`Mul`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    // Integer primitives use checked arithmetic so that host bookkeeping (e.g., dimension-extent math) reports
    // arithmetic failures as errors instead of wrapping like the XLA-mirroring reference backends do on devices.
    (@integer $type:ty) => {
        impl Mul for $type {
            fn mul(&self, right: &Self) -> Result<Self, ProgramError> {
                self.checked_mul(*right).ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("'mul' result does not fit in {}", stringify!($type)),
                })
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 arithmetic, which cannot fail.
    (@float $type:ty) => {
        impl Mul for $type {
            fn mul(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(*self * *right)
            }
        }
    };
}

impl_capability_for_primitive!(@integer i8);
impl_capability_for_primitive!(@integer i16);
impl_capability_for_primitive!(@integer i32);
impl_capability_for_primitive!(@integer i64);
impl_capability_for_primitive!(@integer i128);
impl_capability_for_primitive!(@integer isize);
impl_capability_for_primitive!(@integer u8);
impl_capability_for_primitive!(@integer u16);
impl_capability_for_primitive!(@integer u32);
impl_capability_for_primitive!(@integer u64);
impl_capability_for_primitive!(@integer u128);
impl_capability_for_primitive!(@integer usize);
impl_capability_for_primitive!(@float f32);
impl_capability_for_primitive!(@float f64);

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayType, DataType, Dimension, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension,
    };
    use crate::backends::Array;
    use crate::contexts::EagerContext;
    use crate::differentiation::{jvp, vjp};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::programs::{EmptyRegionDriver, Operation};

    use super::*;

    #[test]
    fn test_mul() {
        let operation = MulOperation::<ArrayType>::new();

        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0f32), Array::scalar(3.5f64)],
            ),
            Ok(vec![Array::scalar(7.0f64)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &MulOperation::<ArrayType>::new(),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0), Array::scalar(3.5)],
            ),
            Ok(vec![Array::scalar(7.0)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(Complex::new(1.0f64, 2.0)), Array::scalar(Complex::new(0.5f64, -1.0))],
            ),
            Ok(vec![Array::scalar(Complex::new(1.0f64, 2.0) * Complex::new(0.5f64, -1.0))]),
        );
        assert_eq!(Mul::mul(&3_usize, &4), Ok(12));
        assert_eq!(
            Mul::mul(&usize::MAX, &2),
            Err(ProgramError::InvalidArgument { message: "'mul' result does not fit in usize".to_string() }),
        );
    }

    #[test]
    fn test_mul_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = MulOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::F8E3M4, DataType::F32],
                    error = format!("'{MUL_OPERATION_NAME}' input types are not broadcast-compatible"),
                },
            ],
        );

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let vector_type = || ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let unreduced = |axis: &str| {
            vector_type()
                .with_sharding(
                    Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                        .unwrap()
                        .with_unreduced_axes([axis])
                        .unwrap(),
                )
                .unwrap()
        };
        let reduced = |axis: &str| {
            vector_type()
                .with_sharding(
                    Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                        .unwrap()
                        .with_reduced_axes([axis])
                        .unwrap(),
                )
                .unwrap()
        };

        // Unreduced over `x` times reduced over `x` is the partial-sum-times-replicated case: the product stays
        // unreduced over `x`, and the reduced marker is cleared.
        let output = <MulOperation<ArrayType> as Operation>::infer_output_types(
            &MulOperation::new(),
            &[unreduced("x"), reduced("x")],
            &[],
        )
        .unwrap();
        assert_eq!(output[0].sharding().unwrap().unreduced_axes(), &BTreeSet::from(["x".to_string()]));
        assert_eq!(output[0].sharding().unwrap().reduced_axes(), &BTreeSet::new());

        // Two operands both unreduced cannot be multiplied (the product of two partial sums is not a partial sum).
        check_operation_type_inference!(
            operation = MulOperation::<ArrayType>::new(),
            cases = [{
                input_types = [unreduced("x"), unreduced("x")],
                error = format!("'{MUL_OPERATION_NAME}' cannot multiply two operands that are both unreduced"),
            }],
        );

        // Unreduced over `x` requires the other operand to be reduced over exactly `x`, not a different axis.
        check_operation_type_inference!(
            operation = MulOperation::<ArrayType>::new(),
            cases = [{
                input_types = [unreduced("x"), reduced("y")],
                error = format!(
                    "'{MUL_OPERATION_NAME}' requires the second operand to be reduced over the axes the first is \
                     unreduced over",
                ),
            }],
        );

        // Two operands reduced over the same axis multiply to a value reduced over that axis.
        let output = <MulOperation<ArrayType> as Operation>::infer_output_types(
            &MulOperation::new(),
            &[reduced("x"), reduced("x")],
            &[],
        )
        .unwrap();
        assert_eq!(output[0].sharding().unwrap().reduced_axes(), &BTreeSet::from(["x".to_string()]));
        assert_eq!(output[0].sharding().unwrap().unreduced_axes(), &BTreeSet::new());

        // A reduced operand cannot be multiplied by an otherwise replicated operand because the result would inherit
        // a reduction marker that does not describe both inputs.
        check_operation_type_inference!(
            operation = MulOperation::<ArrayType>::new(),
            cases = [{
                input_types = [reduced("x"), vector_type()],
                error = format!("'{MUL_OPERATION_NAME}' operands must be reduced over the same axes"),
            }],
        );
    }

    #[test]
    fn test_mul_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = MulOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                    (@replicated, Array::scalar(3.0)),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![3.0, -6.0]))],
            }],
        );
    }

    #[test]
    fn test_mul_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = MulOperation::new(),
            cases = [{
                primals = [Array::scalar(2.0), Array::scalar(5.0)],
                tangents = [Array::scalar(3.0), Array::scalar(-1.0)],
                primal_outputs = [Array::scalar(10.0)],
                tangent_outputs = [Array::scalar(13.0)],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = mul %0 %1
                        %5:f64[] = mul %1 %2
                        %6:f64[] = mul %0 %3
                        %7:f64[] = add %5 %6
                    in (%4, %7)
                "},
            }],
        );
    }

    #[test]
    fn test_mul_complex_differentiation() {
        // Complex arrays differentiate through the same rule: the eager JVP computes `l·dr + dl·r` elementwise over
        // `c128` payloads, and the reverse-mode pullback applies the bilinear (conjugation-free) transpose pairing.
        let left = Complex::new(1.0f64, 2.0);
        let right = Complex::new(0.5f64, -1.0);
        let left_tangent = Complex::new(-0.5f64, 0.25);
        let right_tangent = Complex::new(2.0f64, 1.0);
        let (primal, tangent) = jvp(
            |(left, right)| Ok(left * right),
            (Array::vector(vec![left]), Array::vector(vec![right])),
            (Array::vector(vec![left_tangent]), Array::vector(vec![right_tangent])),
        )
        .unwrap();
        assert_eq!(primal, Array::vector(vec![left * right]));
        assert_eq!(tangent, Array::vector(vec![left_tangent * right + left * right_tangent]));
        let (_, pullback) =
            vjp(|(left, right)| Ok(left * right), (Array::vector(vec![left]), Array::vector(vec![right]))).unwrap();
        let cotangent = Complex::new(0.5f64, 3.0);
        let (left_cotangent, right_cotangent) = pullback.apply(Array::vector(vec![cotangent])).unwrap();
        assert_eq!(left_cotangent, Array::vector(vec![cotangent * right]));
        assert_eq!(right_cotangent, Array::vector(vec![cotangent * left]));
    }

    #[test]
    fn test_mul_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = MulOperation::new(),
            inputs = [Array::scalar(2.0), Array::scalar(3.5)],
            expected = Array::scalar(7.0),
        );
    }

    #[test]
    fn test_mul_transposition() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        check_operation_transposition!(
            @exact,
            operation = MulOperation::new(),
            cases = [
                {
                    inputs = [
                        (@known, Array::scalar(4.0)),
                        (@linear(type = scalar_type.clone())),
                    ],
                    output_cotangents = [Array::scalar(1.0)],
                    input_cotangents = [Array::scalar(4.0)],
                    pullback = indoc! {"
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %1 %0
                        in (%2)
                    "},
                },
                {
                    inputs = [
                        (@known, Array::from_f64s(vector_type.clone(), vec![1.0, 2.0, 3.0])),
                        (@linear(type = scalar_type)),
                    ],
                    output_cotangents = [Array::from_f64s(vector_type.clone(), vec![2.0, 3.0, 4.0])],
                    input_cotangents = [Array::scalar(20.0)],
                    pullback = indoc! {"
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = mul %1 %0
                            %3:f64[] = reduce_sum [axes=[0]] %2
                        in (%3)
                    "},
                },
            ],
        );
    }

    #[test]
    fn test_mul_for_primitives() {
        assert_eq!(Mul::mul(&3_usize, &4), Ok(12));
        assert_eq!(
            Mul::mul(&i8::MAX, &2),
            Err(ProgramError::InvalidArgument { message: "'mul' result does not fit in i8".to_string() }),
        );
        assert_eq!(Mul::mul(&2.5_f64, &4.0), Ok(10.0));
    }
}
