//! Array IR instantiations of the shape-manipulation operation family contracts.
//!
//! Shape manipulation is where first-class runtime dimensions do their most visible work: broadcasting an array to a
//! dynamic output shape consumes one first-class dimension operand per output axis. This module supplies the array
//! universe's answers to those contracts.

use crate::arrays::ir::ArrayIrValue;
use crate::arrays::sharding::shardings::Sharding;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{Dimension, DimensionType, Shape};
use crate::backends::arrays::BroadcastKernel;
use crate::operations::manipulation::broadcasting::infer_explicit_broadcast_output_type;
use crate::operations::{Broadcast, BroadcastOperation, DimensionSize};
use crate::programs::{ProgramError, Value, ValueProjection};

impl<A: BroadcastKernel + DimensionSize<usize> + Value<Type = ArrayType>> Broadcast for ArrayIrValue<A> {
    fn broadcast_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_axes: &[usize],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let output_shape = Shape::new(
            output_dimensions
                .iter()
                .map(<Self as ValueProjection<DimensionType>>::projected)
                .map(|result| result.map(|dimension| Dimension::Static(dimension.extent())))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let operation = BroadcastOperation::new(output_axes.to_vec()).with_output_sharding(output_sharding);
        let output_type = infer_explicit_broadcast_output_type(input.r#type().as_ref(), output_shape, &operation)?;
        Ok(Self::Array(input.broadcast_to_type(output_type, output_axes)?))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::{ArrayIrBatch, ArrayIrBatching};
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ArrayOperation, DimensionOperation};
    use crate::arrays::sharding::meshes::{LogicalMesh, MeshAxis, MeshAxisType};
    use crate::arrays::sharding::shardings::ShardingDimension;
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape};
    use crate::arrays::types::ir::ArrayIrType;
    use crate::backends::Array;
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::{DifferentiableType, DifferentiationError};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::{
        Broadcast, BroadcastOperation, CONCATENATE_OPERATION_NAME, ConcatenateOperation, DimensionAddOperation,
        DimensionMulOperation, DimensionSizeOperation, DynamicShapeSliceOperation, DynamicSliceOperation,
        DynamicUpdateSliceOperation, GatherDimensionNumbers, GatherOperation, PadOperation, ReshapeOperation,
        ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind, SliceOperation, UpdateSliceOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, ProgramError, Typed};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_array_ir_reshape_partial_evaluation() {
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let first_extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let second_extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let output = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let input_type = input.r#type().into_owned();
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = ReshapeOperation::new(),
            cases = [
                {
                    inputs = [
                        (@known, input.clone()),
                        (@known, first_extent.clone()),
                        (@known, second_extent.clone()),
                    ],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input_type, replay = input)),
                        (@known, first_extent),
                        (@known, second_extent),
                    ],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );

        let identity_input = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let identity_input_type = identity_input.r#type().into_owned();
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = ReshapeOperation::new(),
            cases = [{
                inputs = [
                    (@unknown(type = identity_input_type, replay = identity_input.clone())),
                    (@known, ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())),
                    (@known, ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())),
                ],
                outputs = [(@residual, identity_input)],
                residual_instructions = 0,
            }],
        );
    }

    #[test]
    fn test_array_ir_reshape_differentiation() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(6)])).into());
        let first_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let second_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(ReshapeOperation::new(), Vec::new(), vec![input, first_extent, second_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 2);
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])),
                ArrayIrValue::Array(Array::vector(vec![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0])),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])),
                ArrayIrValue::Array(Array::matrix(2, 3, vec![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0])),
            ]),
        );

        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.interpret(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],))]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0,]))]),
        );

        // The inverse cannot recover `n` from the `[2, 2*n]` output shape without division. The reshape JVP must
        // therefore retain the original source extent as an explicit residual while it still has the source array.
        let source = DimensionVariable::new("source", DimensionBounds::new(0, Some(9)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let source_extent = builder
            .add_instruction(DimensionSizeOperation::new(&input_type, 0).unwrap(), Vec::new(), vec![input])
            .unwrap()[0];
        let two_value = DimensionValue::constant(2).unwrap();
        let two_type = two_value.r#type().clone();
        let two = builder.add_constant(ArrayIrValue::Dimension(two_value));
        let source_type = DimensionType::new(input_type.shape().dimensions()[0].variable().unwrap().clone());
        let doubled_extent = builder
            .add_instruction(
                DimensionOperation::Mul(DimensionMulOperation::new(&source_type, &two_type).unwrap()),
                Vec::new(),
                vec![source_extent, two],
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(ReshapeOperation::new(), Vec::new(), vec![input, two, doubled_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 2);
        for size in [0, 1, 3, 8] {
            let element_count = size * 4;
            let primal_values = (0..element_count).map(|value| value as f64).collect::<Vec<_>>();
            let tangent_values = (element_count..2 * element_count).map(|value| value as f64).collect::<Vec<_>>();
            assert_eq!(
                jvp.interpret(vec![
                    ArrayIrValue::Array(Array::matrix(size, 4, primal_values.clone())),
                    ArrayIrValue::Array(Array::matrix(size, 4, tangent_values.clone())),
                ]),
                Ok(vec![
                    ArrayIrValue::Array(Array::matrix(2, 2 * size, primal_values)),
                    ArrayIrValue::Array(Array::matrix(2, 2 * size, tangent_values)),
                ]),
            );
        }
        let linearization = program.linearize().unwrap();
        let rendered_primal = linearization.primal().to_string();
        let rendered_tangent = linearization.tangent().to_string();
        assert_eq!(
            rendered_primal,
            "
lambda %0:f64[source, 4] .
let %1:dimension<2> = const
    %2:dimension<source ∈ [0, 9)> = dimension_size [axis=0] %0
    %3:dimension<source * 2 ∈ [0, 17)> = dimension_mul %2 %1
    %4:f64[2, source * 2] = reshape %0 %1 %3
    %5:dimension<source ∈ [0, 9)> = dimension_size [axis=0] %0
in (%4, %3, %5)
            "
            .trim(),
        );
        assert_eq!(
            rendered_tangent,
            "
lambda %0:f64[source, 4], %1:dimension<source * 2 ∈ [0, 17)>, %2:dimension<source ∈ [0, 9)> .
let %3:dimension<2> = const
    %4:f64[2, source * 2] = linear_call [residual_count=3] %3 %1 %2 %0 [
        forward={
            lambda %0:dimension<2>, %1:dimension<source * 2 ∈ [0, 17)>, %2:dimension<source ∈ [0, 9)>, \
%3:f64[source, 4] .
            let %4:f64[2, source * 2] = reshape %3 %0 %1
            in (%4)
        },
        transpose={
            lambda %0:dimension<2>, %1:dimension<source * 2 ∈ [0, 17)>, \
%2:dimension<source ∈ [0, 9)>, %3:f64[2, source * 2] .
            let %4:dimension<4> = constant [value=4]
                %5:f64[source, 4] = reshape %3 %2 %4
            in (%5)
        },
    ]
in (%4)
            "
            .trim(),
        );
        assert_eq!(linearization.tangent().input_types()[0], ArrayIrType::Array(input_type.tangent()));
        assert!(
            linearization
                .tangent()
                .input_types()
                .iter()
                .skip(1)
                .all(|r#type| matches!(r#type, ArrayIrType::Dimension(_)))
        );
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::matrix(3, 4, (0..12).map(|value| value as f64).collect()))])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(residuals.len(), linearization.residual_count());
        assert_eq!(residuals.len(), 2);

        let tangent_values = (12..24).map(|value| value as f64).collect::<Vec<_>>();
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::matrix(3, 4, tangent_values.clone()))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs.clone()),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 6, tangent_values))]),
        );

        // The executable linear boundary remains structural when imported, including both attached regions and every
        // residual edge. Nested forward differentiation likewise treats only the array input as differentiable.
        let mut imported_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let imported_inputs = linearization
            .tangent()
            .input_types()
            .into_iter()
            .map(|r#type| imported_builder.add_input(r#type))
            .collect::<Vec<_>>();
        let imported_outputs =
            imported_builder.splice_program(linearization.tangent(), imported_inputs.as_slice()).unwrap();
        let imported = imported_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                imported_outputs,
                vec![Placeholder; imported_inputs.len()],
                vec![Placeholder],
            )
            .unwrap();
        let [imported_call] = imported.instructions() else {
            panic!("expected one imported linear call");
        };
        assert!(matches!(imported_call.operation(), ArrayIrOperation::LinearCall(_)));
        assert_eq!(imported_call.regions().len(), 2);
        assert_eq!(imported.interpret(tangent_inputs.clone()), linearization.tangent().interpret(tangent_inputs));

        let nested_jvp = linearization.tangent().jvp().unwrap();
        let mut nested_inputs =
            vec![ArrayIrValue::Array(Array::matrix(3, 4, (12..24).map(|value| value as f64).collect()))];
        nested_inputs.extend(residuals.clone());
        nested_inputs.push(ArrayIrValue::Array(Array::matrix(3, 4, (24..36).map(|value| value as f64).collect())));
        assert_eq!(nested_jvp.input_ids().len(), 2 + residuals.len());
        assert_eq!(
            nested_jvp.interpret(nested_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(2, 6, (12..24).map(|value| value as f64).collect(),)),
                ArrayIrValue::Array(Array::matrix(2, 6, (24..36).map(|value| value as f64).collect(),)),
            ]),
        );

        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::matrix(2, 6, (24..36).map(|value| value as f64).collect()))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(3, 4, (24..36).map(|value| value as f64).collect(),))]),
        );

        // A matching explicit output-extent operand is already the authoritative SSA value for the source axis, so
        // the residual path reuses it and does not read the source array again.
        let source = DimensionVariable::new("reused_source", DimensionBounds::new(1, Some(9)).unwrap());
        let source_type = DimensionType::new(source.clone());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)])).into(),
        );
        let source_extent = builder.add_input(source_type.into());
        let four = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output = builder
            .add_instruction(ReshapeOperation::new(), Vec::new(), vec![input, source_extent, four])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert!(!linearization.primal().to_string().contains("dimension_size"));
        assert!(!linearization.tangent().to_string().contains("dimension_size"));
    }

    #[test]
    fn test_dynamic_reshape_differentiation_deduplicates_repeated_permuted_extents() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Dynamic(extent)]),
        );
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let extent = builder
            .add_instruction(DimensionSizeOperation::new(&input_type, 0).unwrap(), Vec::new(), vec![input])
            .unwrap()[0];
        let output = builder
            .add_instruction(ReshapeOperation::new().with_dimensions([1, 0]), Vec::new(), vec![input, extent, extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        // Both output axes and both inverse axes use the same SSA extent. Partial evaluation carries it once even
        // though the linear call consumes it in multiple operand positions.
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(linearization.primal().to_string().matches("dimension_size").count(), 1);
        let input = ArrayIrValue::Array(Array::matrix(3, 3, (0..9).map(|value| value as f64).collect()));
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let tangent = ArrayIrValue::Array(Array::matrix(3, 3, (9..18).map(|value| value as f64).collect()));
        let mut tangent_inputs = vec![tangent];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(
                3,
                3,
                vec![9.0, 12.0, 15.0, 10.0, 13.0, 16.0, 11.0, 14.0, 17.0],
            ))]),
        );
        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::matrix(3, 3, (18..27).map(|value| value as f64).collect()))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(
                3,
                3,
                vec![18.0, 21.0, 24.0, 19.0, 22.0, 25.0, 20.0, 23.0, 26.0],
            ))]),
        );

        // The same compiled programs accept the lower-bound zero without inventing an extent tangent input.
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::matrix(0, 0, Vec::<f64>::new()))])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::matrix(0, 0, Vec::<f64>::new()))];
        tangent_inputs.extend(residuals);
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(0, 0, Vec::<f64>::new()))]),
        );
    }

    #[test]
    fn test_dynamic_reshape_differentiation_preserves_sharding_through_the_inverse() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Static(4)]))
                .with_sharding(sharding.clone())
                .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let extent = builder.add_input(DimensionType::new(extent).into());
        let four = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output = builder
            .add_instruction(
                ReshapeOperation::new().with_output_sharding(sharding),
                Vec::new(),
                vec![input, extent, four],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.tangent().output_types(), vec![input_type.tangent().into()]);
        assert_eq!(linearization.pullback().unwrap().output_types(), vec![input_type.cotangent().into()]);
    }

    #[test]
    fn test_array_ir_pad_differentiation() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(8).unwrap()));
        let output = builder
            .add_instruction(
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![input, padding_value, output_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        assert_eq!(
            program
                .transpose_with_respect_to(&[0, 1])
                .unwrap()
                .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,]))]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 4.0, 6.0])),
                ArrayIrValue::Array(Array::scalar(24.0_f64)),
            ]),
        );

        let source = DimensionVariable::new("source", DimensionBounds::new(0, Some(5)).unwrap());
        let result = DimensionVariable::new("result", DimensionBounds::new(3, Some(11)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let result_type = DimensionType::new(result);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output_extent = builder.add_input(result_type.clone().into());
        let output = builder
            .add_instruction(
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![input, padding_value, output_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 2);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=2]"));
        assert!(linearization.pullback().unwrap().to_string().contains("dynamic_shape_slice [strides=[2]]"));

        let input = ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0]));
        let padding_value = ArrayIrValue::Array(Array::scalar(-1.0_f64));
        let output_extent = ArrayIrValue::Dimension(DimensionValue::new(result_type.clone(), 8).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, padding_value, output_extent]).unwrap();
        assert_eq!(
            primal_outputs[0],
            ArrayIrValue::Array(Array::vector(vec![-1.0_f64, 10.0, -1.0, 20.0, -1.0, 30.0, -1.0, -1.0])),
        );
        let residuals = primal_outputs.split_off(1);

        let mut tangent_inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0])),
            ArrayIrValue::Array(Array::scalar(4.0_f64)),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0_f64, 1.0, 4.0, 2.0, 4.0, 3.0, 4.0, 4.0]))]),
        );

        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 4.0, 6.0])),
                ArrayIrValue::Array(Array::scalar(24.0_f64)),
            ]),
        );

        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(Vec::<f64>::new())),
                ArrayIrValue::Array(Array::scalar(-1.0_f64)),
                ArrayIrValue::Dimension(DimensionValue::new(result_type, 3).unwrap()),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![-1.0_f64, -1.0, -1.0])),);
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(Vec::<f64>::new())),
                ArrayIrValue::Array(Array::scalar(6.0_f64)),
            ]),
        );

        // Explicit pad geometry retains one output extent per physical axis, including statically typed axes. Keep a
        // static leading axis to verify that the pullback selects dynamic constructor operands from the right axis.
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(5)).unwrap());
        let padded_columns = DimensionVariable::new("padded_columns", DimensionBounds::new(3, Some(7)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(columns)]));
        let padded_columns_type = DimensionType::new(padded_columns);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let rows = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let output_extent = builder.add_input(padded_columns_type.clone().into());
        let output = builder
            .add_instruction(
                PadOperation::new(vec![0, 1], vec![0, 1], vec![0, 0]).unwrap(),
                Vec::new(),
                vec![input, padding_value, rows, output_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0])),
                ArrayIrValue::Array(Array::scalar(-1.0_f64)),
                ArrayIrValue::Dimension(DimensionValue::new(padded_columns_type, 4).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(2, 2, vec![2.0_f64, 3.0, 6.0, 7.0])),
                ArrayIrValue::Array(Array::scalar(18.0_f64)),
            ]),
        );
    }

    #[test]
    fn test_array_ir_dynamic_slice_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let start = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::DynamicSlice(DynamicSliceOperation::new(vec![2]))),
                Vec::new(),
                vec![input, start],
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

        assert_eq!(linearization.residual_count(), 2);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=2]"));
        assert!(linearization.pullback().unwrap().to_string().contains("dynamic_update_slice"));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
                ArrayIrValue::Array(Array::scalar(1_i32)),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0])));
        let residuals = primal_outputs.split_off(1);

        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 10.0, 11.0, 12.0]))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 11.0]))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![5.0_f64, 7.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 7.0, 0.0]))]),
        );

        let extent = DimensionVariable::new("strided_extent", DimensionBounds::new(4, Some(7)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Slice(
                    SliceOperation::new(vec![0], vec![4]).with_strides(vec![2]).unwrap(),
                )),
                Vec::new(),
                vec![input],
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
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]))])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![1.0_f64, 3.0])));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 10.0, 11.0, 12.0]))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 11.0]))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![5.0_f64, 7.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![5.0_f64, 0.0, 7.0, 0.0]))]),
        );
    }

    #[test]
    fn test_array_ir_gather_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3), Dimension::Static(1)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let indices = builder.add_input(indices_type.clone().into());
        let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]);
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Gather(operation)),
                Vec::new(),
                vec![input, indices],
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

        assert_eq!(linearization.residual_count(), 2);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=2]"));
        let indices = ArrayIrValue::Array(Array::from_f64s(indices_type, vec![1.0, 1.0, 3.0]));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])), indices])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![20.0_f64, 20.0, 40.0])));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![2.0_f64, 2.0, 4.0]))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0, 5.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 0.0, 5.0]))]),
        );
    }

    #[test]
    fn test_array_ir_scatter_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(7)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let indices = builder.add_input(indices_type.clone().into());
        let updates = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Scatter(ScatterOperation::new(
                    ScatterDimensionNumbers::new(Vec::new(), vec![0], vec![0]),
                    ScatterReductionKind::Add,
                ))),
                Vec::new(),
                vec![input, indices, updates],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        assert_eq!(linearization.residual_count(), 1);
        let indices = ArrayIrValue::Array(Array::from_f64s(indices_type, vec![1.0, 3.0]));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
                indices,
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0])),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![1.0_f64, 12.0, 3.0, 24.0])));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
            ArrayIrValue::Array(Array::vector(vec![5.0_f64, 6.0])),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 7.0, 3.0, 10.0]))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])),
                ArrayIrValue::Array(Array::vector(vec![20.0_f64, 40.0])),
            ]),
        );
    }

    #[test]
    fn test_array_ir_slice_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(3, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Slice(SliceOperation::new(vec![1], vec![3]))),
                Vec::new(),
                vec![input],
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

        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]))])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0])));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 10.0, 11.0, 12.0]))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 11.0]))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![5.0_f64, 7.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 7.0, 0.0]))]),
        );
    }

    #[test]
    fn test_array_ir_update_slice_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(3, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let update = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::UpdateSlice(UpdateSliceOperation::new(vec![1]))),
                Vec::new(),
                vec![input, update],
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

        assert_eq!(linearization.residual_count(), 0);
        let primal = vec![
            ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
            ArrayIrValue::Array(Array::vector(vec![9.0_f64, 8.0])),
        ];
        assert_eq!(
            linearization.primal().interpret(primal),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 9.0, 8.0, 4.0]))]),
        );
        assert_eq!(
            linearization.tangent().interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])),
                ArrayIrValue::Array(Array::vector(vec![5.0_f64, 6.0])),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 5.0, 6.0, 40.0]))]),
        );
        assert_eq!(
            linearization
                .pullback()
                .unwrap()
                .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0,]))]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 0.0, 0.0, 4.0])),
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0])),
            ]),
        );
    }

    #[test]
    fn test_array_ir_dynamic_update_slice_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let update = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let start = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::DynamicUpdateSlice(DynamicUpdateSliceOperation)),
                Vec::new(),
                vec![input, update, start],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
                ArrayIrValue::Array(Array::vector(vec![9.0_f64, 8.0])),
                ArrayIrValue::Array(Array::scalar(1_i32)),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![1.0_f64, 9.0, 8.0, 4.0])));
        let residuals = primal_outputs.split_off(1);

        let mut tangent_inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])),
            ArrayIrValue::Array(Array::vector(vec![5.0_f64, 6.0])),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 5.0, 6.0, 40.0]))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 0.0, 0.0, 4.0])),
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0])),
            ]),
        );
    }

    #[test]
    fn test_array_ir_reshape_identity_instantiation() {
        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let source_dimension_type = DimensionType::new(source.clone());
        let source_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone()), Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(source_array_type.clone().into());
        let extent = builder.add_input(source_dimension_type.into());
        let four = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output =
            builder.add_instruction(ReshapeOperation::new(), Vec::new(), vec![array, extent, four]).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.output_types(),
            vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)]),)
                    .into()
            ],
        );

        let target = DimensionVariable::new("target", bounds);
        let target_dimension_type = DimensionType::new(target.clone());
        let target_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target.clone()), Dimension::Static(4)]));
        let instantiated = program
            .with_instantiated_type_identities(&[
                target_array_type.clone().into(),
                target_dimension_type.clone().into(),
            ])
            .unwrap()
            .into_owned();
        assert_eq!(
            instantiated.output_types(),
            vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(target.clone()), Dimension::Static(4)]),
                )
                .into()
            ],
        );

        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = destination.add_input(target_array_type.into());
        let extent = destination.add_input(target_dimension_type.into());
        let outputs = destination.splice_program(&instantiated, &[array, extent]).unwrap();
        let [instruction] = destination.instructions() else {
            panic!("expected the imported reshape instruction");
        };
        assert_eq!(instruction.inputs()[..2], [array, extent]);
        assert_eq!(instruction.outputs(), outputs.as_slice());
        assert_eq!(
            destination.atoms()[outputs[0].index()].r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(target), Dimension::Static(4)]),
            )),
        );
    }

    #[test]
    fn test_array_ir_broadcast() {
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0]));
        let first_extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let second_extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let expected_output = ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f64, 2.0, 1.0, 2.0, 1.0, 2.0]));
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            context.bind(
                BroadcastOperation::new(vec![1]),
                Vec::new(),
                &[input.clone(), first_extent.clone(), second_extent.clone()],
            ),
            Ok(vec![expected_output.clone()]),
        );
        let eager_dynamic_type =
            DimensionType::new(DimensionVariable::new("eager_extent", DimensionBounds::new(1, Some(9)).unwrap()));
        assert_eq!(
            context.bind(
                BroadcastOperation::new(vec![1]),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::vector(vec![7.0_f64])),
                    ArrayIrValue::Dimension(DimensionValue::new(eager_dynamic_type, 3).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::matrix(3, 1, vec![7.0_f64, 7.0, 7.0]))]),
        );

        let input_type = input.r#type().into_owned();
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = BroadcastOperation::new(vec![1]),
            cases = [
                {
                    inputs = [
                        (@known, input.clone()),
                        (@known, first_extent.clone()),
                        (@known, second_extent.clone()),
                    ],
                    outputs = [(@known, expected_output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input_type, replay = input.clone())),
                        (@known, first_extent.clone()),
                        (@known, second_extent.clone()),
                    ],
                    outputs = [(@residual, expected_output.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        let identity_input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0]));
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = BroadcastOperation::new(vec![0]),
            cases = [{
                inputs = [
                    (@unknown(type = identity_input.r#type().into_owned(), replay = identity_input.clone())),
                    (@known, second_extent.clone()),
                ],
                outputs = [(@residual, identity_input)],
                residual_instructions = 0,
            }],
        );

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let first_extent = builder.add_constant(first_extent);
        let second_extent = builder.add_constant(second_extent);
        let output = builder
            .add_instruction(BroadcastOperation::new(vec![1]), Vec::new(), vec![input, first_extent, second_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(program.instructions()[0].operation(), ArrayIrOperation::Broadcast(_)));
        assert_eq!(program.instructions()[0].inputs(), &[input, first_extent, second_extent]);
        assert!(program.to_string().contains("broadcast [output_axes=[1]]"));

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0])),
                ArrayIrValue::Array(Array::vector(vec![3.0_f64, 4.0])),
            ]),
            Ok(
                vec![expected_output, ArrayIrValue::Array(Array::matrix(3, 2, vec![3.0_f64, 4.0, 3.0, 4.0, 3.0, 4.0])),]
            ),
        );
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.interpret(vec![ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],))]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 12.0]))]),
        );

        let dynamic_variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dynamic_extent = DimensionType::new(dynamic_variable.clone());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(dynamic_variable)])).into());
        let extent = builder.add_input(dynamic_extent.clone().into());
        let output =
            builder.add_instruction(BroadcastOperation::new(vec![0]), Vec::new(), vec![input, extent]).unwrap()[0];
        let dynamic_program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(dynamic_program.jvp().is_ok());
        let linearization = dynamic_program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dynamic_extent, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let tangent = ArrayIrValue::Array(Array::vector(vec![4.0_f64, 5.0, 6.0]));
        let mut tangent_inputs = vec![tangent.clone()];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(linearization.tangent().interpret(tangent_inputs), Ok(vec![tangent.clone()]));
        let mut pullback_inputs = vec![tangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![tangent]));

        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let extent = builder.add_input(DimensionType::new(source.clone()).into());
        let one = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()));
        let output = builder
            .add_instruction(BroadcastOperation::new(vec![1]), Vec::new(), vec![input, extent, one])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let target = DimensionVariable::new("target", bounds);
        let instantiated = program
            .with_instantiated_type_identities(&[
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into(),
                DimensionType::new(target.clone()).into(),
            ])
            .unwrap()
            .into_owned();
        assert_eq!(
            instantiated.output_types(),
            vec![
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(target.clone()), Dimension::Static(1)]),
                )
                .into()
            ],
        );
        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = destination.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let extent = destination.add_input(DimensionType::new(target.clone()).into());
        let outputs = destination.splice_program(&instantiated, &[input, extent]).unwrap();
        let [instruction] = destination.instructions() else {
            panic!("expected the imported broadcast instruction");
        };
        assert_eq!(instruction.inputs()[..2], [input, extent]);
        assert_eq!(
            destination.atoms()[outputs[0].index()].r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(target), Dimension::Static(1)]),
            )),
        );
    }

    #[test]
    fn test_array_ir_dynamic_shape_slice() -> Result<(), ProgramError> {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = ArrayIrValue::Array(Array::matrix(3, 4, (0..12).map(|value| value as f64).collect::<Vec<_>>()));
        let dimension = |extent| Ok::<_, ProgramError>(ArrayIrValue::Dimension(DimensionValue::constant(extent)?));
        let output = context.bind(
            DynamicShapeSliceOperation::new(2),
            Vec::new(),
            &[input, dimension(1)?, dimension(1)?, dimension(2)?, dimension(2)?],
        )?;
        assert_eq!(output, vec![ArrayIrValue::Array(Array::matrix(2, 2, vec![5.0, 6.0, 9.0, 10.0]))]);

        // The slice geometry is discrete, but the array operand remains linear: JVP applies the same runtime slice to
        // the primal and tangent instead of treating the complete mixed operation as a constant.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)])).into());
        let start = builder.add_constant(dimension(1)?);
        let size = builder.add_constant(dimension(2)?);
        let output =
            builder.add_instruction(DynamicShapeSliceOperation::new(1), Vec::new(), vec![input, start, size])?[0];
        let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output],
            vec![Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.jvp().unwrap().interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![20.0_f64, 30.0])),
            ]),
        );
        assert!(matches!(
            program.transpose_with_respect_to(&[0]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation 'dynamic_shape_slice' does not yet support reverse-mode differentiation",
        ));

        Ok(())
    }

    #[test]
    fn test_array_ir_broadcast_to_first_class_dimensions() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let eager = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]));
        assert_eq!(
            eager.broadcast_leading_sizes(&[2]),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 1.0, 2.0, 3.0],))),
        );

        let context = TestContext::new();
        let value = context.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).into());
        let extent = context.constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        assert_eq!(value.broadcast(&[extent], &[0]).unwrap().atom_id(), value.atom_id());
        assert!(context.builder().borrow().instructions().is_empty());

        // A shape-preserving axis permutation is still a real broadcast. Eager execution transposes the payload and
        // tracing retains the operation even though its input and output types are equal.
        let square_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let square = ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]));
        let two = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let expected = ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f64, 3.0, 2.0, 4.0]));
        assert_eq!(square.broadcast(&[two.clone(), two.clone()], &[1, 0]), Ok(expected.clone()));

        let context = TestContext::new();
        let value = context.input(square_type.into());
        let extent = context.constant(two);
        let output = value.broadcast(&[extent.clone(), extent], &[1, 0]).unwrap();
        {
            let builder = context.builder().borrow();
            let [instruction] = builder.instructions() else {
                panic!("expected one shape-preserving broadcast instruction");
            };
            let ArrayIrOperation::Broadcast(operation) = instruction.operation() else {
                panic!("expected a broadcast instruction");
            };
            assert_eq!(operation.output_axes(), &[1, 0]);
        }
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(program.interpret(vec![square]), Ok(vec![expected]));

        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let context = TestContext::new();
        let scalar = context.input(ArrayType::scalar(DataType::F64).into());
        let extent = context.input(extent_type.clone().into());
        let output = scalar.broadcast_to(std::slice::from_ref(&extent)).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:dimension<extent \u{2208} [1, 5)> .
                let %2:f64[extent] = broadcast [output_axes=[]] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::scalar(2.5_f64)),
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![2.5_f64, 2.5, 2.5]))]),
        );
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0])),
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(6.0_f64))]),
        );

        let context = TestContext::new();
        let value = context.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let rows = context.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let columns = context.constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = value.broadcast_to(&[rows, columns]).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(Array::vector(vec![7.0_f64]))]),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![7.0_f64; 6]))]),
        );

        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let context = TestContext::new();
        let value = context.input(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(3)]))
                .into(),
        );
        let output = value.broadcast_leading_sizes(&[2]).unwrap();
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(batch), Dimension::Static(3)]),
            )),
        );
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let rendered = program.to_string();
        assert_eq!(rendered.matches("dimension_size").count(), 1);
        assert!(rendered.contains("broadcast [output_axes=[1, 2]]"));
    }

    #[test]
    fn test_array_ir_concatenate() {
        let left = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let right = ArrayIrValue::Array(Array::vector(vec![3.0_f32]));
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let output = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[left.r#type().into_owned(), right.r#type().into_owned(), extent.r#type().into_owned()],
        )
        .unwrap();

        // Eager execution consumes the explicit extent without copying either array during member projection.
        assert_eq!(
            context.bind(operation.clone(), Vec::new(), &[left.clone(), right.clone(), extent.clone()],),
            Ok(vec![output.clone()]),
        );
        assert_eq!(
            context.bind(
                operation.clone(),
                Vec::new(),
                &[left.clone(), ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),],
            ),
            Ok(vec![left.clone()]),
        );

        let observed_extent_type =
            DimensionType::new(DimensionVariable::new("observed", DimensionBounds::new(1, Some(9)).unwrap()));
        let checked_operation = ConcatenateOperation::<ArrayIrType>::from(ConcatenateOperation::new(0, 1).unwrap());
        assert_eq!(
            ArrayIrOperation::<Array>::from(checked_operation).interpret(
                &context,
                &EmptyRegionDriver,
                &[
                    left.clone(),
                    right.clone(),
                    ArrayIrValue::Dimension(DimensionValue::new(observed_extent_type, 4).unwrap()),
                ],
            ),
            Err(ProgramError::InvalidArgument {
                message: format!(
                    "'{}' result extent must equal the sum of input axis 0 extents; expected 3 but got 4",
                    CONCATENATE_OPERATION_NAME,
                ),
            }),
        );

        // Partial evaluation folds a fully known concatenate and otherwise retains exactly one operation with the
        // explicit extent edge, including when only that extent is unknown.
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = operation.clone(),
            cases = [
                {
                    inputs = [(@known, left.clone()), (@known, right.clone()), (@known, extent.clone())],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = left.r#type().into_owned(), replay = left.clone())),
                        (@known, right.clone()),
                        (@known, extent.clone()),
                    ],
                    outputs = [(@residual, output.clone())],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, left.clone()),
                        (@known, right.clone()),
                        (@unknown(type = extent.r#type().into_owned(), replay = extent.clone())),
                    ],
                    outputs = [(@residual, output.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        // A stored dynamic program computes the trailing extent through ordinary dimension SSA and records every
        // dependency explicitly on the concatenate instruction.
        let left_variable = DimensionVariable::new("left", DimensionBounds::new(1, Some(5)).unwrap());
        let right_variable = DimensionVariable::new("right", DimensionBounds::new(1, Some(6)).unwrap());
        let left_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(left_variable.clone())]));
        let right_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(right_variable.clone())]));
        let left_size_operation = DimensionSizeOperation::new(&left_type, 0).unwrap();
        let right_size_operation = DimensionSizeOperation::new(&right_type, 0).unwrap();
        let left_size_type = left_size_operation.result_type().clone();
        let right_size_type = right_size_operation.result_type().clone();
        let add_operation = DimensionAddOperation::new(&left_size_type, &right_size_type).unwrap();
        let result_extent_type =
            DimensionType::new(DimensionVariable::new(add_operation.result_name(), add_operation.result_bounds()));
        let dynamic_operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[left_type.clone().into(), right_type.clone().into(), result_extent_type.into()],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let left_input = builder.add_input(left_type.into());
        let right_input = builder.add_input(right_type.into());
        let left_size = builder.add_instruction(left_size_operation, Vec::new(), vec![left_input]).unwrap()[0];
        let right_size = builder.add_instruction(right_size_operation, Vec::new(), vec![right_input]).unwrap()[0];
        let result_extent = builder
            .add_instruction(DimensionOperation::Add(add_operation), Vec::new(), vec![left_size, right_size])
            .unwrap()[0];
        let concatenated = builder
            .add_instruction(dynamic_operation, Vec::new(), vec![left_input, right_input, result_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![concatenated],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let [left_size_instruction, right_size_instruction, add_instruction, concatenate_instruction] =
            program.instructions()
        else {
            panic!("expected two dimension reads, one dimension addition, and one concatenate");
        };
        assert_eq!(left_size_instruction.inputs(), &[left_input]);
        assert_eq!(right_size_instruction.inputs(), &[right_input]);
        assert_eq!(add_instruction.inputs(), &[left_size, right_size]);
        assert_eq!(concatenate_instruction.inputs(), &[left_input, right_input, result_extent]);
        assert!(matches!(concatenate_instruction.operation(), ArrayIrOperation::Concatenate(_),));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[left], %1:f32[right] .
                let %2:dimension<left ∈ [1, 5)> = dimension_size [axis=0] %0
                    %3:dimension<right ∈ [1, 6)> = dimension_size [axis=0] %1
                    %4:dimension<left + right ∈ [2, 10)> = dimension_add %2 %3
                    %5:f32[left + right] = concatenate [axis=0] %0 %1 %4
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(program.interpret(vec![left, right]), Ok(vec![output.clone()]));

        // The same stored dynamic program composes dimension arithmetic with both forward differentiation and
        // batching. Its tangent retains the result extent and both operand extents as ordinary residual SSA edges.
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 4);
        let transformed_result_extent = jvp
            .instructions()
            .iter()
            .find_map(|instruction| {
                matches!(instruction.operation(), ArrayIrOperation::Dimension(DimensionOperation::Add(_)),)
                    .then_some(instruction.outputs()[0])
            })
            .unwrap();
        assert_eq!(
            jvp.instructions()
                .iter()
                .filter_map(|instruction| match instruction.operation() {
                    ArrayIrOperation::Concatenate(_) => instruction.inputs().last().copied(),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![transformed_result_extent],
        );
        let tangent_call = jvp
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayIrOperation::LinearCall(_)))
            .unwrap();
        assert_eq!(tangent_call.inputs()[0], transformed_result_extent);
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])),
                ArrayIrValue::Array(Array::vector(vec![3.0_f32])),
                ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0])),
                ArrayIrValue::Array(Array::vector(vec![6.0_f32])),
            ]),
            Ok(vec![output, ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0])),]),
        );

        type Parent = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let batching_context = BatchingContext::<_, ArrayIrBatching>::new(
            Parent::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("items".to_string())
        .with_axis_sharding(ShardingDimension::Unconstrained);
        let batched_outputs = program
            .interpret_in_context(
                &batching_context,
                vec![
                    BatchingTracer::new(
                        batching_context.clone(),
                        ArrayIrBatch::new(
                            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 4.0, 5.0])),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                    BatchingTracer::new(
                        batching_context.clone(),
                        ArrayIrBatch::new(
                            ArrayIrValue::Array(Array::matrix(2, 1, vec![3.0_f32, 6.0])),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                ],
            )
            .unwrap();
        assert_eq!(batched_outputs.len(), 1);
        assert_eq!(batched_outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            batched_outputs[0].batch().value(),
            &ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],)),
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 3);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=3]"));
        assert!(linearization.pullback().unwrap().to_string().contains("dynamic_shape_slice"));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])),
                ArrayIrValue::Array(Array::vector(vec![3.0_f32])),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![7.0_f32, 8.0, 9.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![7.0_f32, 8.0])),
                ArrayIrValue::Array(Array::vector(vec![9.0_f32])),
            ]),
        );
    }

    #[test]
    fn test_array_ir_concatenate_differentiation() {
        let left_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let right_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)]));
        let extent_value = DimensionValue::constant(3).unwrap();
        let operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[left_type.clone().into(), right_type.clone().into(), extent_value.r#type().clone().into()],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let left = builder.add_input(left_type.into());
        let right = builder.add_input(right_type.into());
        let extent = builder.add_constant(ArrayIrValue::Dimension(extent_value));
        let output = builder.add_instruction(operation, Vec::new(), vec![left, right, extent]).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        assert_eq!(
            program.jvp().unwrap().interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0])),
                ArrayIrValue::Array(Array::vector(vec![3.0_f64])),
                ArrayIrValue::Array(Array::vector(vec![4.0_f64, 5.0])),
                ArrayIrValue::Array(Array::vector(vec![6.0_f64])),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![4.0_f64, 5.0, 6.0])),
            ]),
        );
        assert_eq!(
            program
                .transpose_with_respect_to(&[0, 1])
                .unwrap()
                .interpret(vec![ArrayIrValue::Array(Array::vector(vec![7.0_f64, 8.0, 9.0]))]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![7.0_f64, 8.0])),
                ArrayIrValue::Array(Array::vector(vec![9.0_f64])),
            ]),
        );
    }

    #[test]
    fn test_array_ir_concatenate_identity_instantiation() {
        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let result = DimensionVariable::new("result", DimensionBounds::new(2, Some(12)).unwrap());
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let fixed_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let result_extent_type = DimensionType::new(result.clone());
        let operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[source_array_type.clone().into(), fixed_array_type.clone().into(), result_extent_type.clone().into()],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let source_array = builder.add_input(source_array_type.into());
        let fixed_array = builder.add_input(fixed_array_type.clone().into());
        let result_extent = builder.add_input(result_extent_type.into());
        let output = builder
            .add_instruction(operation, Vec::new(), vec![source_array, fixed_array, result_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let target = DimensionVariable::new("target", bounds);
        let target_result = DimensionVariable::new("target_result", DimensionBounds::new(2, Some(12)).unwrap());
        let target_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target.clone())]));
        let target_result_type = DimensionType::new(target_result.clone());
        let instantiated = program
            .with_instantiated_type_identities(&[
                target_array_type.clone().into(),
                fixed_array_type.clone().into(),
                target_result_type.clone().into(),
            ])
            .unwrap()
            .into_owned();
        assert_eq!(
            instantiated.output_types(),
            vec![ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target_result.clone())])).into()],
        );

        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let imported_source = destination.add_input(target_array_type.into());
        let imported_fixed = destination.add_input(fixed_array_type.into());
        let imported_extent = destination.add_input(target_result_type.into());
        let imported_outputs = destination
            .splice_program(&instantiated, &[imported_source, imported_fixed, imported_extent])
            .unwrap();
        let [instruction] = destination.instructions() else {
            panic!("expected the imported concatenate instruction");
        };
        assert_eq!(instruction.inputs(), &[imported_source, imported_fixed, imported_extent]);
        assert_eq!(instruction.outputs(), imported_outputs.as_slice());
        assert_eq!(
            destination.atoms()[imported_outputs[0].index()].r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target_result)]),)),
        );
    }
}
