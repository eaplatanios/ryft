//! Array IR instantiations of the collectives operation family contracts.
//!
//! Collective operations are homogeneous over ordinary array data, so the composite array IR only needs to lift them
//! into its array member family. The mixed collective variants whose trailing operands are first-class dimensions are
//! declared directly by [`ArrayIrOperation`](crate::ArrayIrOperation) instead.

use crate::arrays::operations::{ArrayIrOperation, ArrayOperation};
use crate::arrays::types::arrays::ArrayType;
use crate::operations::collectives::PpermuteOperation;
use crate::programs::Value;

// TODO(eaplatanios): Review this.

impl<A: Value<Type = ArrayType>> From<PpermuteOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: PpermuteOperation) -> Self {
        Self::Array(ArrayOperation::Ppermute(operation))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, DimensionOperation};
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape};
    use crate::arrays::types::ir::ArrayIrType;
    use crate::axes::NamedAxis;
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::DifferentiationError;
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::collectives::{
        AllGather, AllGatherOperation, AllGatherOutputVariance, AllToAllOperation, CollectiveOptions, PSumScatter,
        PSumScatterOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError, TypeError, Typed};
    use crate::tracing::TracingContext;

    #[test]
    fn test_array_ir_explicit_collective_eager_contracts() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());

        assert_eq!(
            context.bind(
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                PSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::tiled()),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],)),
                    ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],))]),
        );

        assert_eq!(
            context
                .bind(
                    AllGatherOperation::new(
                        "x".to_string(),
                        1,
                        0,
                        CollectiveOptions::tiled(),
                        AllGatherOutputVariance::Varying
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()),],
                )
                .unwrap_err()
                .to_string(),
            "`all_gather` output axis 0 extent must equal observed result extent 3 but got 4",
        );
        assert_eq!(
            context
                .bind(
                    AllGatherOperation::new(
                        "x".to_string(),
                        2,
                        0,
                        CollectiveOptions::tiled(),
                        AllGatherOutputVariance::Varying
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(6).unwrap()),],
                )
                .unwrap_err(),
            ProgramError::UnsupportedOperation {
                message: "cannot interpret `all_gather` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            },
        );
        assert_eq!(
            context
                .bind(
                    PSumScatterOperation::new("empty".to_string(), 0, 0, CollectiveOptions::tiled()),
                    Vec::new(),
                    &[input.clone(), extent.clone()],
                )
                .unwrap_err(),
            ProgramError::Type(TypeError::invalid("`psum_scatter` axis size must be greater than zero")),
        );

        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = AllGatherOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled(), AllGatherOutputVariance::Varying),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, extent.clone())],
                    outputs = [(@known, input.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, extent.clone()),
                    ],
                    outputs = [(@residual, input.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        let variable = DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap());
        let dimension_type = DimensionType::new(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![array, result_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let primal = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let tangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let result_extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 3).unwrap());
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![primal.clone(), result_extent.clone(), tangent.clone(),]),
            Ok(vec![primal, tangent]),
        );
        assert!(
            jvp.instructions()
                .iter()
                .any(|instruction| { matches!(instruction.operation(), ArrayIrOperation::LinearCall(_)) })
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0])), result_extent])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
        let zero_extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 0).unwrap());
        let zero_array = || {
            ArrayIrValue::Array(Array::from_f64s(
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(0)])),
                Vec::new(),
            ))
        };
        let mut primal_outputs = linearization.primal().interpret(vec![zero_array(), zero_extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let zero_cotangent = zero_array();
        let mut pullback_inputs = vec![zero_cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![zero_cotangent]));
        assert!(matches!(
            program.transpose_with_respect_to(&[0]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "direct `all_gather` transposition with runtime-dependent type metadata requires \
                    linearization so that the relevant primal information can be retained as residuals",
        ));
    }

    #[test]
    fn test_array_ir_invariant_all_gather_linearization() {
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dimension_type = DimensionType::new(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Invariant,
                ),
                Vec::new(),
                vec![array, result_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let rendered_tangent = linearization.tangent().to_string();
        assert!(rendered_tangent.contains("dynamic_shape_slice"));
        assert!(rendered_tangent.contains("reshape"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let mut pullback_inputs = vec![cotangent];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]))]),
        );

        // A nondegenerate untiled invariant gather selects the current participant's size-one slice and reshapes
        // away the ranked participant axis.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let participant_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let input_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::default(),
                    AllGatherOutputVariance::Invariant,
                ),
                Vec::new(),
                vec![array, participant_extent, input_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        // The mixed boundary delegates its array contribution to the homogeneous all-gather rule, so the invariant
        // guard that rule owns is what rejects direct transposition here.
        assert!(matches!(
            program.transpose_with_respect_to(&[0]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "direct transposition of invariant `all_gather` cannot represent the participant-indexed \
                    slice; linearize so that the current participant can select its gathered chunk",
        ));
        let pullback = program.linearize().unwrap().pullback().unwrap().to_string();
        assert!(pullback.contains("axis_index [axis_name=\"x\"]"));
        assert!(pullback.contains("dimension_from_scalar"));
        assert!(pullback.contains("dimension_mul"));
        assert!(pullback.contains("dynamic_shape_slice"));
    }

    #[test]
    fn test_array_ir_shape_changing_collective_linearization() {
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dimension_type = DimensionType::new(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                PSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![array, result_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(
                AllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![array, extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert!(linearization.tangent().to_string().contains("linear_call"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
    }

    #[test]
    fn test_array_ir_explicit_collective_tracing_import_and_rendering() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let bounds = DimensionBounds::new(1, Some(5)).unwrap();
        let input_variable = DimensionVariable::new("items", bounds);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(input_variable.clone())]));
        let (_, program) = TestContext::trace_with_named_axes(
            |input| input.all_gather_tiled("devices", 0),
            ArrayIrType::Array(input_type),
            vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2 })],
        )
        .unwrap();

        let [dimension_size, multiplied_extent, all_gather] = program.instructions() else {
            panic!("expected dimension observation, multiplication, and all-gather");
        };
        assert!(matches!(dimension_size.operation(), ArrayIrOperation::DimensionSize(_)));
        assert!(matches!(multiplied_extent.operation(), ArrayIrOperation::Dimension(DimensionOperation::Mul(_)),));
        assert!(matches!(all_gather.operation(), ArrayIrOperation::AllGather(_)));
        assert_eq!(multiplied_extent.inputs()[0], dimension_size.outputs()[0]);
        assert_eq!(all_gather.inputs(), &[program.input_ids()[0], multiplied_extent.outputs()[0]]);
        let rendered = program.to_string();
        assert!(rendered.contains("dimension_size"));
        assert!(rendered.contains("dimension_mul"));
        assert!(rendered.contains("all_gather ["));
        assert!(rendered.contains("axis_name=\"devices\""));
        assert!(rendered.contains("options=Tiled"));

        let target_variable = DimensionVariable::new("target", bounds);
        let target_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target_variable)]));
        let instantiated = program
            .with_instantiated_type_identities(&[ArrayIrType::Array(target_type.clone())])
            .unwrap()
            .into_owned();
        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let imported_input = destination.add_input(target_type.into());
        let imported_outputs = destination.splice_program(&instantiated, &[imported_input]).unwrap();
        let [imported_dimension_size, imported_multiplied_extent, imported_all_gather] = destination.instructions()
        else {
            panic!("expected the imported explicit collective graph");
        };
        assert_eq!(imported_dimension_size.inputs(), &[imported_input]);
        assert_eq!(imported_all_gather.inputs(), &[imported_input, imported_multiplied_extent.outputs()[0]]);
        assert_eq!(imported_all_gather.outputs(), imported_outputs.as_slice());
    }

    #[test]
    fn test_array_ir_untiled_collective_retains_dynamic_extent_requirement() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let input_variable = DimensionVariable::new("items", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(input_variable)]));
        let (_, program) = TestContext::trace_with_named_axes(
            |input| input.psum_scatter("devices", 0),
            ArrayIrType::Array(input_type),
            vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2 })],
        )
        .unwrap();

        let [dimension_size, requirement, psum_scatter] = program.instructions() else {
            panic!("expected dimension observation, equality requirement, and sum-scatter");
        };
        assert!(matches!(dimension_size.operation(), ArrayIrOperation::DimensionSize(_)));
        assert!(matches!(requirement.operation(), ArrayIrOperation::Dimension(DimensionOperation::Requirement(_)),));
        assert!(matches!(psum_scatter.operation(), ArrayIrOperation::PSumScatter(_)));
        assert_eq!(requirement.inputs()[0], dimension_size.outputs()[0]);
        assert_eq!(psum_scatter.inputs(), &[program.input_ids()[0]]);
        assert_eq!(program.output_types(), &[ArrayIrType::Array(ArrayType::scalar(DataType::F32))],);
    }
}
