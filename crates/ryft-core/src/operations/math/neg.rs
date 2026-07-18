use std::fmt::Display;
use std::ops::Neg as StandardNeg;

use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, define_tracer_operator};
use crate::operations::ElementwiseOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::operations::Operation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::TypeError;
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, DataType};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`NegOperation`].
pub const NEG_OPERATION_NAME: &str = "neg";

/// Infers the output data type for numeric negation.
fn infer_neg_output_data_type(input_type: DataType) -> Result<DataType, TypeError> {
    super::validate_numeric_input_types(std::slice::from_ref(&input_type), NEG_OPERATION_NAME)?;
    if input_type == DataType::F8E8M0FNU {
        return Err(TypeError { message: "'neg' does not support input data type f8e8m0fnu".to_string() });
    }
    Ok(input_type)
}

/// [`Operation`] that negates one integer, floating-point, or complex value while preserving its array metadata and
/// reduction state. Boolean, token, structural-zero, and the unsigned-only `f8e8m0fnu` data types are rejected.
#[derive(Clone, Debug, Default)]
pub struct NegOperation;

impl Display for NegOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(NEG_OPERATION_NAME)
    }
}

impl Operation<DataType> for NegOperation {
    #[inline]
    fn name(&self) -> &'static str {
        NEG_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![infer_neg_output_data_type(input_types[0])?])
    }
}

impl Operation<ArrayType> for NegOperation {
    #[inline]
    fn name(&self) -> &'static str {
        NEG_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        infer_neg_output_data_type(input_types[0].data_type())?;
        ElementwiseOperation::infer_output_types(self, input_types)
    }
}

impl ElementwiseOperation for NegOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<C: Domain<Value: Neg>> InterpretableOperation<C> for NegOperation
where
    Self: Operation<C::Type>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].neg()?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for NegOperation where C::Operation: From<NegOperation> {}

impl<C: Context> DifferentiableOperation<C> for NegOperation
where
    C::Type: DifferentiableType,
    C::Value: StandardNeg<Output = C::Value>,
    NegOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = -inputs[0].primal().clone();
        // A negated structural zero stays structural, keeping `neg(zero)` out of the tangent program.
        let tangent = inputs[0].tangent().clone().map(|tangent| -tangent);
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

impl<V: Value, O: Operation<V::Type> + From<NegOperation>> TransposableOperation<V, O> for NegOperation
where
    NegOperation: Operation<V::Type>,
{
    #[inline]
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Value(cotangent) => Ok(vec![MaybeZero::Value(-cotangent.clone())]),
            MaybeZero::Zero(r#type) => Ok(vec![MaybeZero::Zero(r#type.clone())]),
        }
    }
}

/// Value-level elementwise negation capability. [`Neg`] is the fallible Ryft counterpart to [`std::ops::Neg`]
/// that [`NegOperation`] interprets through, surfacing a [`ProgramError`] when something goes wrong, instead of
/// panicking. Value types additionally provide [`std::ops::Neg`] as ergonomic (albeit panicking) sugar layered on top
/// of this capability.
pub trait Neg: Sized {
    /// Negates `self`, returning a [`ProgramError`] if something goes wrong.
    fn neg(&self) -> Result<Self, ProgramError>;
}

impl<V: Value<DispatchDomain: Context<Operation: From<NegOperation>>>> Neg for V {
    #[inline]
    fn neg(&self) -> Result<Self, ProgramError> {
        Ok(self.dispatch_domain().bind(NegOperation, Vec::new(), std::slice::from_ref(self))?.remove(0))
    }
}

define_tracer_operator!(@unary std::ops::Neg, neg, NegOperation, "`neg` operation failed");

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::differentiation::{gradient, gradient_holomorphic};
    use crate::macros::check_gradient;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing_v2::ForwardModeDifferentiate;
    use crate::types::{ArrayType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_neg() {
        let operation = NegOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), NEG_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "NegOperation");
        assert_eq!(format!("{operation}"), NEG_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32], &[]),
            Ok(vec![DataType::F32]),
        );
        let output_types = Operation::<DataType>::infer_output_types(&operation, &[DataType::U8], &[]);
        assert_eq!(output_types, Ok(vec![DataType::U8]));
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0)],
            ),
            Ok(vec![Scalar::from(-2.0)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(1u8)],
            ),
            Ok(vec![Scalar::from(u8::MAX)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0)],
            ),
            Ok(vec![Array::scalar(-2.0)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(Complex::new(1.0f64, -2.0))],
            ),
            Ok(vec![Scalar::from(Complex::new(-1.0f64, 2.0))]),
        );

        // Array type inference preserves shape, layout, and sharding metadata for its single input.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![3, 1])))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            <NegOperation as Operation<ArrayType>>::infer_output_types(&operation, std::slice::from_ref(&input), &[]),
            Ok(vec![input]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Scalar, NegOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = neg %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_neg_type_inference() {
        for input_type in [DataType::Token, DataType::Zero, DataType::Boolean, DataType::F8E8M0FNU] {
            let expected =
                TypeError { message: format!("'{NEG_OPERATION_NAME}' does not support input data type {input_type}") };
            assert_eq!(
                Operation::<DataType>::infer_output_types(&NegOperation, &[input_type], &[]),
                Err(expected.clone()),
            );
            assert_eq!(
                Operation::<ArrayType>::infer_output_types(&NegOperation, &[ArrayType::scalar(input_type)], &[]),
                Err(expected),
            );
        }

        // Negation is linear, so partial-sum and reduced markers pass through unchanged.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&NegOperation, std::slice::from_ref(&unreduced), &[]),
            Ok(vec![unreduced]),
        );
        let reduced = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&NegOperation, std::slice::from_ref(&reduced), &[]),
            Ok(vec![reduced]),
        );
    }

    #[test]
    fn test_neg_batching() {
        crate::operations::math::tests::assert_unary_batching(NegOperation, &[1.0, -2.0], &[-1.0, 2.0]);
    }

    #[test]
    fn test_neg_differentiation() {
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = context.jvp(|x| Ok(-x), Scalar::from(2.0), Scalar::from(3.0)).unwrap();
        assert_eq!(primal, -2.0);
        assert_eq!(tangent, -3.0);
        check_gradient!(@scalar, |x| -x, at = 0.7, step = 1e-6, tolerance = 1e-6);
        assert_eq!(
            gradient_holomorphic(|input| -input, Scalar::from(Complex::new(0.7f64, -0.3))),
            Ok(Scalar::from(Complex::new(-1.0, 0.0))),
        );

        // Second-order differentiation recovers d²(-x)/dx² = 0.
        assert_abs_diff_eq!(
            gradient(|x| gradient(|x| -x, x).unwrap(), Scalar::from(0.7f64)).unwrap(),
            0.0,
            epsilon = 1e-9,
        );

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(NegOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = neg %0
                    %3:f64[] = neg %1
                in (%2, %3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_neg_partial_evaluation() {
        crate::operations::math::tests::assert_partial_evaluation(NegOperation, &[2.0], -2.0);
    }

    #[test]
    fn test_neg_transposition() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(NegOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        // The pullback negates the output cotangent.
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = neg %0
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(pullback.interpret(vec![Array::scalar(3.0)]), Ok(vec![Array::scalar(-3.0)]));
    }
}
