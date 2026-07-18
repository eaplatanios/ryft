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
    check_data_types = [@numeric],
    check_array_types = [@same_unreduced_axes, @same_reduced_axes],
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

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::differentiation::{gradient, gradient_holomorphic};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
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
        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = AddOperation,
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_add_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = AddOperation,
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                    (@replicated, Array::scalar(3.0)),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![4.0, 1.0]))],
            }],
        );
    }

    #[test]
    fn test_add_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = AddOperation,
            cases = [{
                primals = [Array::scalar(2.0), Array::scalar(5.0)],
                tangents = [Array::scalar(3.0), Array::scalar(-1.0)],
                primal_outputs = [Array::scalar(7.0)],
                tangent_outputs = [Array::scalar(2.0)],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = add %0 %1
                        %5:f64[] = add %2 %3
                    in (%4, %5)
                "},
            }],
        );
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
    }

    #[test]
    fn test_add_partial_evaluation() {
        check_operation_partial_evaluation!(operation = AddOperation, inputs = [2.0, 3.5], expected = 5.5,);
    }

    #[test]
    fn test_add_transposition() {
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        check_operation_transposition!(
            @exact,
            operation = AddOperation,
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::scalar(3.0)],
                    input_cotangents = [Array::scalar(3.0), Array::scalar(3.0)],
                    pullback = indoc! {"
                        lambda %0:f64[] .
                        in (%0, %0)
                    "},
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = vector_type.clone())),
                    ],
                    output_cotangents = [Array::from_f64s(vector_type.clone(), vec![2.0, 3.0, 4.0])],
                    input_cotangents = [
                        Array::scalar(9.0),
                        Array::from_f64s(vector_type, vec![2.0, 3.0, 4.0]),
                    ],
                    pullback = indoc! {"
                        lambda %0:f64[3] .
                        let %1:f64[] = reduce_sum [axes=[0]] %0
                        in (%1, %0)
                    "},
                },
            ],
        );
    }
}
