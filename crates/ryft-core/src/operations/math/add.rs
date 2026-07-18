use std::ops::Add as StandardAdd;

use crate::contexts::Context;
use crate::differentiation::elementwise::{ElementwiseDerivativeAlignment, binary_elementwise_jvp};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::macros::{check_count, define_elementwise_capability, define_elementwise_operation, define_tracer_operator};
use crate::partial::PartialValue;
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::operations::Operation;
use crate::programs::types::Typed;
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`AddOperation`].
pub const ADD_OPERATION_NAME: &str = "add";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that adds two numeric values elementwise, promoting their element types and
    /// broadcasting their shapes. Array operands that carry partial sums must both be unreduced over exactly the same
    /// mesh axes; mixing an unreduced operand with an already reduced operand would duplicate the reduced contribution
    /// when the result is subsequently reduced. Their reduced-axis markers must likewise agree.
    AddOperation, ADD_OPERATION_NAME,
    Add, add,
    validate_data_types = super::validate_numeric_input_types,
    validate_array_types = super::validate_linear_reduction_state,
);

impl<C: Context> DifferentiableOperation<C> for AddOperation
where
    C::Type: DifferentiableType,
    C::Value: StandardAdd<Output = C::Value> + ElementwiseDerivativeAlignment<C::Type>,
    AddOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // d(x + y) = dx + dy.
        binary_elementwise_jvp(
            self,
            inputs,
            |left, right| Ok(left.clone() + right.clone()),
            |_, tangent| Ok(tangent),
            |_, tangent| Ok(tangent),
        )
    }
}

/// Transposes addition by unbroadcasting its output cotangent to each operand's exact cotangent type. A
/// structural-zero output cotangent propagates as structural zeros — including to operands without a cotangent space
/// (e.g., integer operands), which only reject *live* cotangents.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for AddOperation
where
    V::Type: DifferentiableType,
    Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
    AddOperation: Operation<V::Type>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect()),
            MaybeZero::Value(cotangent) => inputs
                .iter()
                .map(|input| {
                    let target = input.r#type().cotangent();
                    if target.is_zero_space() {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "'add' input has no cotangent space".to_string(),
                        }
                        .into());
                    }
                    Ok(MaybeZero::Value(cotangent.unalign_cotangent(&target)?))
                })
                .collect(),
        }
    }
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise addition capability. [`Add`] is the fallible Ryft counterpart to [`std::ops::Add`] that
    /// [`AddOperation`] interprets through, surfacing a [`ProgramError`] when something goes
    /// wrong, instead of panicking. Value types additionally provide [`std::ops::Add`] as ergonomic (albeit
    /// panicking) sugar layered on top of this capability.
    Add, add, AddOperation,
);

define_tracer_operator!(@binary std::ops::Add, add, AddOperation, "`add` operation failed");

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::differentiation::{gradient, gradient_holomorphic};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_gradient;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing_v2::ForwardModeDifferentiate;
    use crate::types::{ArrayType, DataType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_add() {
        let operation = AddOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), ADD_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "AddOperation");
        assert_eq!(format!("{operation}"), ADD_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F64], &[]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0f32), Scalar::from(3.5f64)],
            ),
            Ok(vec![Scalar::from(5.5f64)])
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0), Array::scalar(3.5)],
            ),
            Ok(vec![Array::scalar(5.5)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(Complex::new(1.0f64, 2.0)), Scalar::from(Complex::new(0.5f64, -1.0))],
            ),
            Ok(vec![Scalar::from(Complex::new(1.5f64, 1.0))]),
        );

        // Array type inference broadcasts shapes and promotes data types.
        let output = <AddOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::scalar(DataType::F32),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)])),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(output, vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))]);

        // Array type inference drops layout metadata when inputs disagree.
        let output = <AddOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::new(DataType::F32, Shape::scalar()).with_layout(Layout::Strided(StridedLayout::new(vec![]))),
                ArrayType::scalar(DataType::F32),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(output, vec![ArrayType::scalar(DataType::F32)]);

        // Array type inference tolerates compatible inputs that only disagree on varying manual axes.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let left = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let right = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["y"])
                    .unwrap(),
            )
            .unwrap();
        let output =
            <AddOperation as Operation<ArrayType>>::infer_output_types(&operation, &[left, right], &[]).unwrap();
        assert_eq!(
            output[0].sharding().as_ref().unwrap().varying_manual_axes(),
            &BTreeSet::from(["x".to_string(), "y".to_string()]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64], &[]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F64)], &[]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0)],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0)]
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Scalar, AddOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_add_type_inference() {
        assert_eq!(
            Operation::<DataType>::infer_output_types(&AddOperation, &[DataType::F8E3M4, DataType::F32], &[]),
            Err(TypeError { message: format!("'{ADD_OPERATION_NAME}' input types are not broadcast-compatible") }),
        );
        for input_type in [DataType::Token, DataType::Zero, DataType::Boolean] {
            let expected =
                TypeError { message: format!("'{ADD_OPERATION_NAME}' does not support input data type {input_type}") };
            assert_eq!(
                Operation::<DataType>::infer_output_types(&AddOperation, &[input_type, input_type], &[]),
                Err(expected.clone()),
            );
            assert_eq!(
                Operation::<ArrayType>::infer_output_types(
                    &AddOperation,
                    &[ArrayType::scalar(input_type), ArrayType::scalar(input_type)],
                    &[],
                ),
                Err(expected),
            );
        }
        assert_eq!(
            <AddOperation as Operation<ArrayType>>::infer_output_types(
                &AddOperation,
                &[
                    ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)])),
                    ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])),
                ],
                &[],
            ),
            Err(TypeError { message: format!("'{ADD_OPERATION_NAME}' input types are not broadcast-compatible") }),
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let plain = || {
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
                .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap())
                .unwrap()
        };
        let unreduced = || {
            plain()
                .with_sharding(plain().sharding().unwrap().clone().with_unreduced_axes(["x"]).unwrap())
                .unwrap()
        };

        let output =
            Operation::<ArrayType>::infer_output_types(&AddOperation, &[unreduced(), unreduced()], &[]).unwrap();
        assert_eq!(output[0].unreduced_axes(), &BTreeSet::from(["x".to_string()]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&AddOperation, &[unreduced(), plain()], &[]),
            Err(TypeError { message: "'add' operands must be unreduced over the same axes".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&AddOperation, &[plain(), unreduced()], &[]),
            Err(TypeError { message: "'add' operands must be unreduced over the same axes".to_string() }),
        );
        crate::operations::math::tests::assert_rejects_mismatched_reduced(AddOperation, ADD_OPERATION_NAME);
    }

    #[test]
    fn test_add_batching() {
        crate::operations::math::tests::assert_binary_batching(AddOperation, &[1.0, -2.0], 3.0, &[4.0, 1.0]);
    }

    #[test]
    fn test_add_differentiation() {
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = context
            .jvp(
                |(left, right)| Ok(left + right),
                (Scalar::from(2.0), Scalar::from(5.0)),
                (Scalar::from(3.0), Scalar::from(-1.0)),
            )
            .unwrap();
        assert_eq!(primal, 7.0);
        assert_eq!(tangent, 2.0);
        fn double<V: Clone + std::ops::Add<Output = V>>(input: V) -> V {
            input.clone() + input
        }
        check_gradient!(@scalar, double, at = 0.7, step = 1e-6, tolerance = 1e-6);
        let input = Complex::new(0.7f64, -0.3);
        assert_eq!(gradient_holomorphic(double, Scalar::from(input)), Ok(Scalar::from(Complex::new(2.0, 0.0))));

        // Second-order differentiation recovers d²(x + x)/dx² = 0.
        assert_abs_diff_eq!(
            gradient(|x| gradient(double, x).unwrap(), Scalar::from(0.7f64)).unwrap(),
            0.0,
            epsilon = 1e-9,
        );

        // The `DataType`-typed staged tangent program adds the operand tangents.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64, %2:f64, %3:f64 .
                let %4:f64 = add %0 %1
                    %5:f64 = add %2 %3
                in (%4, %5)
            "}
            .trim_end(),
        );

        // The array-typed staged tangent program has the same shape.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[] = add %0 %1
                    %5:f64[] = add %2 %3
                in (%4, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_add_partial_evaluation() {
        crate::operations::math::tests::assert_partial_evaluation(AddOperation, &[2.0, 3.5], 5.5);
    }

    #[test]
    fn test_add_transposition() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(Array, Array), Array>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0, 1]).unwrap();
        // The pullback fans the output cotangent out to both operands.
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[] .
                in (%0, %0)
            "}
            .trim_end(),
        );
        assert_eq!(pullback.interpret(vec![Array::scalar(3.0)]), Ok(vec![Array::scalar(3.0), Array::scalar(3.0)]),);

        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(vector_type.clone());
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(Array, Array), Array>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0, 1]).unwrap();
        // The pullback sum-reduces the broadcast operand's cotangent back to its scalar type.
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[] = reduce_sum [axes=[0]] %0
                in (%1, %0)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.interpret(vec![Array::from_f64s(vector_type.clone(), vec![2.0, 3.0, 4.0])]),
            Ok(vec![Array::scalar(9.0), Array::from_f64s(vector_type, vec![2.0, 3.0, 4.0])]),
        );
    }
}
