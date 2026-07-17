use std::ops::Add as StandardAdd;

use crate::contexts::Context;
use crate::define_elementwise_capability;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::macros::{check_count, define_elementwise_operation, define_tracer_operator};
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::types::Typed;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::operations::broadcasting::ElementwiseDifferentiableValue;

/// Canonical operation name for [`AddOperation`].
pub const ADD_OPERATION_NAME: &str = "add";

define_elementwise_operation!(
    @binary_base
    /// [`Operation`] that adds two numeric values elementwise, promoting their element types and
    /// broadcasting their shapes. Array operands that carry partial sums must both be unreduced over exactly the same
    /// mesh axes; mixing an unreduced operand with an already reduced operand would duplicate the reduced contribution
    /// when the result is subsequently reduced. Their reduced-axis markers must likewise agree.
    AddOperation, ADD_OPERATION_NAME,
    Add, add,
    validate = super::validate_numeric_input_types,
    validate_array = validate_add_reduction_state,
);

/// Validates the reduction-state rule for elementwise addition.
///
/// # Parameters
///
///   - `input_types`: The two array operand types.
///   - `operation_name`: Canonical operation name used to identify invalid signatures in the returned error.
fn validate_add_reduction_state(
    input_types: &[crate::ArrayType],
    operation_name: &str,
) -> Result<(), crate::TypeError> {
    let left = input_types[0].unreduced_axes();
    let right = input_types[1].unreduced_axes();
    if left.is_empty() != right.is_empty() || (!left.is_empty() && left != right) {
        return Err(crate::TypeError {
            message: format!("'{operation_name}' operands must be unreduced over the same axes"),
        });
    }
    if input_types[0].reduced_axes() != input_types[1].reduced_axes() {
        return Err(crate::TypeError {
            message: format!("'{operation_name}' operands must be reduced over the same axes"),
        });
    }
    Ok(())
}

impl<C: Context> DifferentiableOperation<C> for AddOperation
where
    C::Type: DifferentiableType,
    C::Value: StandardAdd<Output = C::Value> + ElementwiseDifferentiableValue<C::Type>,
    AddOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let primal = inputs[0].primal().clone() + inputs[1].primal().clone();
        let left = inputs[0].tangent().as_value().cloned();
        let right = inputs[1].tangent().as_value().cloned();
        let target = primal.r#type().tangent();
        if target.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'add' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        let tangent = match (left, right) {
            (Some(left), Some(right)) => MaybeZero::Value(
                left.normalize_elementwise_tangent(&target)? + right.normalize_elementwise_tangent(&target)?,
            ),
            (Some(tangent), None) | (None, Some(tangent)) => {
                MaybeZero::Value(tangent.normalize_elementwise_tangent(&target)?)
            }
            (None, None) => MaybeZero::Zero(target),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Transposes addition by unbroadcasting its output cotangent to each operand's exact cotangent type.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for AddOperation
where
    V::Type: DifferentiableType,
    Tracer<TracingContext<V, O>>: ElementwiseDifferentiableValue<V::Type>,
    AddOperation: Operation<V::Type>,
{
    #[inline]
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
            MaybeZero::Zero(_) => inputs
                .iter()
                .map(|input| {
                    let target = input.r#type().cotangent();
                    if target.is_zero_space() {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "'add' input has no cotangent space".to_string(),
                        });
                    }
                    Ok(MaybeZero::Zero(target))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(Into::into),
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
                    Ok(MaybeZero::Value(cotangent.unbroadcast_elementwise_cotangent(&target)?))
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

    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::differentiation::gradient_holomorphic;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::{TestArray, check_gradient};
    use crate::tracing_v2::{ArrayOperation, ForwardModeDifferentiate};
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
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[TestArray::scalar(2.0), TestArray::scalar(3.5)],
            ),
            Ok(vec![TestArray::scalar(5.5)]),
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
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[TestArray::scalar(2.0)]
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F8E3M4, DataType::F32], &[]),
            Err(TypeError { message: format!("'{ADD_OPERATION_NAME}' input types are not broadcast-compatible") }),
        );
        for input_type in [DataType::Token, DataType::Zero, DataType::Boolean] {
            let expected =
                TypeError { message: format!("'{ADD_OPERATION_NAME}' does not support input data type {input_type}") };
            assert_eq!(
                Operation::<DataType>::infer_output_types(&operation, &[input_type, input_type], &[]),
                Err(expected.clone()),
            );
            assert_eq!(
                Operation::<ArrayType>::infer_output_types(
                    &operation,
                    &[ArrayType::scalar(input_type), ArrayType::scalar(input_type)],
                    &[],
                ),
                Err(expected),
            );
        }
        let error = <AddOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)])),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])),
            ],
            &[],
        )
        .unwrap_err();
        assert_eq!(
            error,
            TypeError { message: format!("'{ADD_OPERATION_NAME}' input types are not broadcast-compatible") }
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
        check_gradient!(double, 0.7, 1e-6, 1e-6);
        let input = Complex::new(0.7f64, -0.3);
        assert_eq!(gradient_holomorphic(double, Scalar::from(input)), Ok(Scalar::from(Complex::new(2.0, 0.0))));

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
    }

    #[test]
    fn test_add_transposition() {
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(TestArray, TestArray), TestArray>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0, 1]).unwrap();
        assert_eq!(
            pullback.interpret(vec![TestArray::scalar(3.0)]),
            Ok(vec![TestArray::scalar(3.0), TestArray::scalar(3.0)]),
        );

        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(vector_type.clone());
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(TestArray, TestArray), TestArray>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0, 1]).unwrap();
        assert_eq!(
            pullback.interpret(vec![TestArray::new(vector_type.clone(), vec![2.0, 3.0, 4.0])]),
            Ok(vec![TestArray::scalar(9.0), TestArray::new(vector_type, vec![2.0, 3.0, 4.0]),]),
        );
    }
}
