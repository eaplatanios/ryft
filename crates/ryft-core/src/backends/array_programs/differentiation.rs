//! Differentiation and transposition rules for composite array IR operations.
//!
//! Homogeneous and mixed-operation rules remain with their owning operation payloads. This module implements the
//! composite operation-family boundary: it delegates operation-owned rules and retains only behavior that belongs to
//! the family as a whole, such as member projection and structural-zero cotangents for non-differentiable dimensions.

use crate::differentiation::forward::{MemberDifferentiableOperation, jvp_projected_operation};
use crate::operations::control_flow::{
    TemporalResidualOperation, TemporalResidualType, WhileResidualStackOperation, WhileResidualStackType,
};
use crate::operations::dimensions::RUNTIME_DIMENSION_DATA_TYPE;
use crate::operations::logical::AndOperation;

use super::*;

impl TemporalResidualType for ArrayIrType {
    #[inline]
    fn temporal_storage_type(&self) -> Result<Self, TypeError> {
        Ok(match self {
            Self::Array(r#type) => Self::Array(r#type.clone()),
            Self::Dimension(_) => Self::Array(ArrayType::scalar(RUNTIME_DIMENSION_DATA_TYPE)),
        })
    }
}

impl<O> TemporalResidualOperation<ArrayIrType> for O
where
    O: Operation<Type = ArrayIrType> + From<DimensionFromScalarOperation> + From<DimensionToScalarOperation>,
{
    fn residual_to_storage(residual_type: &ArrayIrType) -> Result<Option<Self>, TypeError> {
        Ok(match residual_type {
            ArrayIrType::Array(_) => None,
            ArrayIrType::Dimension(_) => Some(Self::from(DimensionToScalarOperation)),
        })
    }

    fn residual_from_storage(residual_type: &ArrayIrType) -> Result<Option<Self>, TypeError> {
        Ok(match residual_type {
            ArrayIrType::Array(_) => None,
            ArrayIrType::Dimension(r#type) => {
                Some(Self::from(DimensionFromScalarOperation::new(r#type.variable().clone())))
            }
        })
    }
}

impl WhileResidualStackType for ArrayIrType {
    #[inline]
    fn from_array_type(r#type: ArrayType) -> Self {
        Self::Array(r#type)
    }

    fn array_type(&self) -> Result<&ArrayType, TypeError> {
        match self {
            Self::Array(r#type) => Ok(r#type),
            Self::Dimension(r#type) => {
                Err(TypeError::invalid(format!("expected an array-backed bounded-while state type but got {}", r#type)))
            }
        }
    }

    #[inline]
    fn maskable_array_type(&self) -> Option<&ArrayType> {
        match self {
            Self::Array(r#type) => Some(r#type),
            Self::Dimension(_) => None,
        }
    }
}

impl<A: Value<Type = ArrayType>, O> WhileResidualStackOperation<ArrayIrType, A> for O
where
    O: Operation<Type = ArrayIrType> + From<ArrayIrOperation<A>> + TemporalResidualOperation<ArrayIrType>,
{
    fn residual_stack_zero(r#type: ArrayIrType) -> Self {
        let ArrayIrType::Array(r#type) = r#type else { unreachable!("bounded-while stack zeros are always arrays") };
        Self::from(ArrayIrOperation::<A>::from(ZeroOperation::new(r#type)))
    }

    fn residual_stack_one(r#type: ArrayIrType) -> Self {
        let ArrayIrType::Array(r#type) = r#type else { unreachable!("bounded-while stack ones are always arrays") };
        Self::from(ArrayIrOperation::<A>::from(OneOperation::new(r#type)))
    }

    #[inline]
    fn residual_stack_broadcast(output_type: ArrayType, output_axes: Vec<usize>) -> Self {
        Self::from(ArrayIrOperation::<A>::Array(ArrayOperation::Broadcast(LegacyBroadcastOperation::new(
            output_type,
            output_axes,
        ))))
    }

    #[inline]
    fn residual_stack_update() -> Self {
        Self::from(ArrayIrOperation::<A>::Array(ArrayOperation::DynamicUpdateSlice(DynamicUpdateSliceOperation)))
    }

    #[inline]
    fn residual_stack_add() -> Self {
        Self::from(ArrayIrOperation::<A>::Array(ArrayOperation::Add(AddOperation::new())))
    }

    #[inline]
    fn residual_stack_select() -> Self {
        Self::from(ArrayIrOperation::<A>::Array(ArrayOperation::Select(SelectOperation::new())))
    }

    #[inline]
    fn mask_reduce_any(axes: Vec<usize>) -> Self {
        Self::from(ArrayIrOperation::<A>::Array(ArrayOperation::Reduce(ReduceOperation::new(axes, ReductionKind::Any))))
    }

    #[inline]
    fn mask_and() -> Self {
        Self::from(ArrayIrOperation::<A>::Array(ArrayOperation::And(AndOperation::new())))
    }
}

impl<A, C> MemberDifferentiableOperation<C> for ArrayOperation<A>
where
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayIrType,
            Constant: ValueProjection<ArrayType, Projected = A>,
            Operation: From<ArrayIrOperation<A>>
                           + From<BroadcastOperation>
                           + From<DimensionSizeOperation>
                           + From<DimensionToScalarOperation>
                           + From<LinearCallOperation<ArrayIrType>>
                           + From<ZeroOperation<ArrayType>>
                           + OperationProjection<ArrayType, Projected = ArrayOperation<A>>
                           + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
        > + Zero<C::Value>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    ArrayOperation<A>: Operation<Type = ArrayType> + DifferentiableOperation<ProjectedContext<C, ArrayType>>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let output_duals = match self {
            Self::Slice(operation) => operation.jvp_in_parent(context, driver, inputs)?,
            Self::DynamicSlice(operation) => operation.jvp_in_parent(context, driver, inputs)?,
            Self::DynamicUpdateSlice(operation) => operation.jvp_in_parent(context, driver, inputs)?,
            Self::Gather(operation) => operation.jvp_in_parent(context, driver, inputs)?,
            Self::Reduce(operation) => operation.jvp_in_parent(context, driver, inputs)?,
            operation => jvp_projected_operation(context, operation, inputs)?,
        };
        output_duals
            .into_iter()
            .map(|output| {
                let tangent_type = output.tangent().r#type().into_owned();
                if !output.tangent().is_zero()
                    || tangent_type.identities().all(|(position, _)| position != TypeIdentityPosition::Reference)
                {
                    return Ok(output);
                }

                // A projected array rule can return a structural zero even when its result has runtime extents. Use
                // the primal result as its geometry exemplar before lifting the dual into the composite family.
                let (primal, _) = output.into_parts();
                let tangent_array_type = <&ArrayType>::try_from(&tangent_type)?;
                let primal_type = primal.r#type();
                let primal_data_type = <&ArrayType>::try_from(primal_type.as_ref())?.data_type();
                let exemplar = if tangent_array_type.data_type() == primal_data_type {
                    primal.clone()
                } else {
                    context
                        .bind(
                            ArrayIrOperation::<A>::Array(ArrayOperation::ConvertElementType(
                                ConvertElementTypeOperation::new(tangent_array_type.data_type()),
                            )),
                            Vec::new(),
                            std::slice::from_ref(&primal),
                        )?
                        .remove(0)
                };
                let tangent = context
                    .bind(
                        ArrayIrOperation::<A>::Array(ArrayOperation::ZeroLike(ZeroLikeOperation::new())),
                        Vec::new(),
                        &[exemplar],
                    )?
                    .remove(0);
                DifferentiationDual::new(primal, MaybeZero::Value(tangent)).map_err(Into::into)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrType, ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue, DimensionVariable,
        Shape,
    };
    use crate::backends::array_programs::{ArrayIrOperation, ArrayIrValue};
    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::{DifferentiableType, ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::ZeroOperation;
    use crate::operations::control_flow::{ConditionOperation, ScanOperation, WhileOperation};
    use crate::operations::dimensions::DimensionFromScalarOperation;
    use crate::operations::manipulation::{BroadcastOperation, ReshapeOperation};
    use crate::operations::math::{AddOperation, MulOperation, ReduceOperation, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::types::Typed;
    use crate::programs::{Program, ProgramBuilder};
    use crate::tracing::TracingContext;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;

    fn array(value: Array) -> TestValue {
        TestValue::Array(value)
    }

    fn dimension(r#type: &DimensionType, extent: usize) -> TestValue {
        TestValue::Dimension(DimensionValue::new(r#type.clone(), extent).unwrap())
    }

    fn scale_branch(
        dimension_type: DimensionType,
        factor: f64,
    ) -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(dimension_type));
        let operand = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let factor = builder.add_constant(array(Array::scalar(factor)));
        let output = builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(MulOperation::new())),
                Vec::new(),
                vec![operand, factor],
            )
            .unwrap()[0];
        builder.build(vec![extent, output], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap()
    }

    #[test]
    fn test_composite_condition_jvp_preserves_dimension_outputs_without_tangent_slots() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let operand = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let true_branch = scale_branch(extent_type.clone(), 2.0);
        let false_branch = scale_branch(extent_type.clone(), 3.0);
        let regions = vec![
            builder.import_region(true_branch.entry_region_ref()),
            builder.import_region(false_branch.entry_region_ref()),
        ];
        let outputs = builder
            .add_instruction(
                TestOperation::Condition(ConditionOperation::new()),
                regions,
                vec![predicate, extent, operand],
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 3], vec![Placeholder; 2]).unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_count(), 4);
        assert_eq!(jvp.output_count(), 3);
        let outputs = jvp
            .interpret(vec![
                array(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0])),
                dimension(&extent_type, 4),
                array(Array::scalar(5.0)),
                array(Array::scalar(7.0)),
            ])
            .unwrap();
        assert!(matches!(&outputs[0], TestValue::Dimension(value) if value.extent() == 4));
        assert!(matches!(&outputs[1], TestValue::Array(value) if value.to_f64s() == vec![10.0]));
        assert!(matches!(&outputs[2], TestValue::Array(value) if value.to_f64s() == vec![14.0]));

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                array(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0])),
                dimension(&extent_type, 4),
                array(Array::scalar(5.0)),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        let mut pullback_inputs = vec![array(Array::scalar(1.0))];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![array(Array::scalar(2.0))]),);
    }

    #[test]
    fn test_composite_condition_all_zero_jvp_materializes_a_dynamic_output_tangent() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let extent = builder.add_input(extent_type.clone().into());
            let output =
                builder.add_instruction(ZeroOperation::new(output_type.clone()), Vec::new(), vec![extent]).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let extent = builder.add_input(extent_type.clone().into());
        let regions = vec![
            builder.import_region(branch().entry_region_ref()),
            builder.import_region(branch().entry_region_ref()),
        ];
        let output = builder.add_instruction(ConditionOperation::new(), regions, vec![predicate, extent]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Both inputs have zero tangent spaces, but the dynamic floating-point result does not. Its zero tangent must
        // therefore consume the selected primal result's explicit runtime extent instead of using a nullary zero.
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_count(), 2);
        assert_eq!(jvp.output_count(), 2);
        assert_eq!(
            jvp.interpret(vec![
                array(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0])),
                dimension(&extent_type, 3),
            ]),
            Ok(vec![array(Array::vector(vec![0.0_f64; 3])), array(Array::vector(vec![0.0_f64; 3])),]),
        );

        // Eager direct JVP keeps the operation's all-zero region fast path and derives the concrete output tangent
        // extent from the selected primal result at the public boundary.
        let eager = EagerContext::<TestValue, TestOperation>::new();
        let (primal, tangent) = eager
            .jvp(
                |inputs| {
                    let context = inputs[0].context().clone();
                    context.bind(ConditionOperation::new(), vec![branch(), branch()], inputs.as_slice())
                },
                vec![
                    array(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0])),
                    dimension(&extent_type, 3),
                ],
                vec![
                    array(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap()),
                    array(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap()),
                ],
            )
            .unwrap();
        assert_eq!(primal, vec![array(Array::vector(vec![0.0_f64; 3]))]);
        assert_eq!(tangent, vec![array(Array::vector(vec![0.0_f64; 3]))]);

        // Split program linearization stages the same extent read on the primal side and forces the shaped zero into
        // the tangent program, rather than folding it into an affine known tangent.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                array(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0])),
                dimension(&extent_type, 3),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(linearization.tangent().interpret(residuals), Ok(vec![array(Array::vector(vec![0.0_f64; 3]))]),);

        // A known symbolic predicate cannot select a branch during partial evaluation. Because the dynamic output
        // edge refers to the extent identity, the condition remains whole instead of fabricating an opposite-branch
        // placeholder with arbitrary geometry.
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let symbolic_predicate = outer.input(ArrayType::scalar(DataType::Boolean).into());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(symbolic_predicate), PartialValue::Unknown(extent_type.clone().into())],
            )
            .unwrap();
        assert!(matches!(evaluation.outputs.as_slice(), [PartialEvaluationOutput::Unknown(0)]));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayIrOperation::Condition(_),));

        // Direct transform dispatch must make the same decision before it has a staged instruction whose result type
        // it can inspect. The condition rule retains the selected branch's extent and constructs the tangent there.
        let context = TracingContext::<TestValue, TestOperation>::new();
        let predicate = context.input(ArrayType::scalar(DataType::Boolean).into());
        let extent = context.input(extent_type.clone().into());
        let predicate_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let extent_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let (_, tangent) = context
            .jvp(
                |inputs| {
                    let context = inputs[0].context().clone();
                    Ok(context.bind(ConditionOperation::new(), vec![branch(), branch()], inputs.as_slice())?.remove(0))
                },
                vec![predicate, extent],
                vec![predicate_tangent, extent_tangent],
            )
            .unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayIrType::Array(output_type.clone()));

        // Reusable linearization follows the same ordinary region rule and closes over the dynamic result geometry;
        // applying its null linear map therefore reconstructs the shaped tangent without a type-only zero.
        let predicate = context.input(ArrayType::scalar(DataType::Boolean).into());
        let extent = context.input(extent_type.clone().into());
        let (_, pushforward) = context
            .linearize(
                |inputs| {
                    let context = inputs[0].context().clone();
                    Ok(context.bind(ConditionOperation::new(), vec![branch(), branch()], inputs.as_slice())?.remove(0))
                },
                vec![predicate, extent],
            )
            .unwrap();
        let predicate_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let extent_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        assert_eq!(pushforward.apply(vec![predicate_tangent, extent_tangent]).unwrap().r#type(), tangent.r#type(),);
    }

    #[test]
    fn test_composite_pullback_materializes_a_dynamic_zero_space_input_cotangent() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let key_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let context = TracingContext::<TestValue, TestOperation>::new();
        let key = context.input(key_type.clone().into());
        let accumulator = context.input(ArrayType::scalar(DataType::F64).into());
        let (_, pullback) = context.vjp(|inputs: Vec<_>| Ok(inputs[1].clone()), vec![key, accumulator]).unwrap();
        let cotangent = context.input(ArrayType::scalar(DataType::F64).into());

        // The compact pullback has no result slot for the key's zero differential space. Rebuilding the public result
        // must use the key extent captured at linearization time rather than attempt a nullary dynamic zero.
        let cotangents = pullback.apply(cotangent).unwrap();
        assert_eq!(cotangents[0].r#type().as_ref(), &ArrayIrType::Array(key_type.tangent()));
        assert_eq!(cotangents[1].r#type().as_ref(), &ArrayType::scalar(DataType::F64).into());
    }

    #[test]
    fn test_composite_pushforward_materializes_a_dynamic_zero_space_output_tangent() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let key_type =
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let context = TracingContext::<TestValue, TestOperation>::new();
        let extent = context.input(extent_type.into());
        let key = context.input(key_type.clone().into());
        let (_, pushforward) = context.linearize(|inputs: Vec<_>| Ok(inputs[1].clone()), vec![extent, key]).unwrap();
        let extent_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let key_tangent = context.input(key_type.tangent().into());

        // The compact pushforward has no output slot for the key's zero differential space. Rebuilding its public
        // result must consume the key extent captured at linearization time.
        let tangent = pushforward.apply(vec![extent_tangent, key_tangent]).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayIrType::Array(key_type.tangent()));
    }

    fn product_scan_body(
        extent_type: DimensionType,
    ) -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type));
        let carry = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let item = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let product = builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(MulOperation::new())),
                Vec::new(),
                vec![carry, item],
            )
            .unwrap()[0];
        builder.build(vec![extent, product, product], vec![Placeholder; 3], vec![Placeholder; 3]).unwrap()
    }

    #[test]
    fn test_composite_scan_jvp_forwards_a_dynamic_length_and_dimension_carry() {
        let carry_extent_type =
            DimensionType::new(DimensionVariable::new("carry_extent", DimensionBounds::positive(Some(8)).unwrap()));
        let length_variable = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let length_type = DimensionType::new(length_variable.clone());
        let length = Dimension::Dynamic(length_variable);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(carry_extent_type.clone()));
        let carry = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let values =
            builder.add_input(ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![length.clone()]))));
        let runtime_length = builder.add_input(ArrayIrType::Dimension(length_type.clone()));
        let body = product_scan_body(carry_extent_type.clone());
        let region = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(
                TestOperation::Scan(ScanOperation::new(2, length)),
                vec![region],
                vec![extent, carry, values, runtime_length],
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 4], vec![Placeholder; 3]).unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_count(), 6);
        assert_eq!(jvp.output_count(), 5);
        let outputs = jvp
            .interpret(vec![
                dimension(&carry_extent_type, 4),
                array(Array::scalar(1.0)),
                array(Array::vector(vec![2.0, 3.0, 4.0])),
                dimension(&length_type, 3),
                array(Array::scalar(5.0)),
                array(Array::vector(vec![0.5, 1.0, 1.5])),
            ])
            .unwrap();
        assert!(matches!(&outputs[0], TestValue::Dimension(value) if value.extent() == 4));
        assert!(matches!(&outputs[1], TestValue::Array(value) if value.to_f64s() == vec![24.0]));
        assert!(matches!(&outputs[3], TestValue::Array(value) if value.to_f64s() == vec![143.0]));

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 3);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                dimension(&carry_extent_type, 4),
                array(Array::scalar(1.0)),
                array(Array::vector(vec![2.0, 3.0, 4.0])),
                dimension(&length_type, 3),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(3);
        let mut pullback_inputs = vec![array(Array::scalar(1.0)), array(Array::vector(vec![0.0, 0.0, 0.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![array(Array::scalar(24.0)), array(Array::vector(vec![12.0, 8.0, 6.0]))]),
        );
    }

    #[test]
    fn test_composite_scan_pullback_stacks_varying_dimension_residuals_through_scalar_gateways() {
        let iteration_variable = DimensionVariable::new("iteration", DimensionBounds::positive(Some(4)).unwrap());
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let scalar_u64 = ArrayType::scalar(DataType::U64);

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = body_builder.add_input(ArrayIrType::Array(scalar_f64.clone()));
        let counter = body_builder.add_input(ArrayIrType::Array(scalar_u64.clone()));
        let iteration = body_builder
            .add_instruction(
                TestOperation::from(DimensionFromScalarOperation::new(iteration_variable)),
                Vec::new(),
                vec![counter],
            )
            .unwrap()[0];
        let repeated = body_builder
            .add_instruction(
                TestOperation::from(BroadcastOperation::new(Vec::new())),
                Vec::new(),
                vec![state, iteration],
            )
            .unwrap()[0];
        let next_state = body_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(ReduceOperation::new(vec![0], ReductionKind::Sum))),
                Vec::new(),
                vec![repeated],
            )
            .unwrap()[0];
        let one = body_builder.add_constant(array(Array::scalar(1_u64)));
        let next_counter = body_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![counter, one],
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![next_state, next_counter, next_state],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = builder.add_input(ArrayIrType::Array(scalar_f64));
        let counter = builder.add_input(ArrayIrType::Array(scalar_u64));
        let region = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(TestOperation::Scan(ScanOperation::new(2, 2)), vec![region], vec![state, counter])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 2);
        let rendered_primal = linearization.primal().to_string();
        let rendered_tangent = linearization.tangent().to_string();
        assert!(rendered_primal.contains("dimension_to_scalar"), "{rendered_primal}");
        assert!(rendered_tangent.contains("dimension_from_scalar"), "{rendered_tangent}");
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![array(Array::scalar(2.0)), array(Array::scalar(1_u64))])
            .unwrap();
        let residuals = primal_outputs.split_off(3);
        let mut pullback_inputs = vec![array(Array::scalar(1.0)), array(Array::vector(vec![0.0, 0.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![array(Array::scalar(2.0))]));
    }

    fn doubling_while_regions(
        extent_type: DimensionType,
    ) -> Vec<Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>> {
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let state = condition_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let limit = condition_builder.add_constant(array(Array::scalar(8.0)));
        let predicate = condition_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![state, limit],
            )
            .unwrap()[0];
        let condition = condition_builder.build(vec![predicate], vec![Placeholder; 2], vec![Placeholder]).unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = body_builder.add_input(ArrayIrType::Dimension(extent_type));
        let state = body_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let doubled = body_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![state, state],
            )
            .unwrap()[0];
        let body = body_builder.build(vec![extent, doubled], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        vec![condition, body]
    }

    #[test]
    fn test_composite_while_jvp_omits_the_dimension_state_tangent() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let state = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let regions = doubling_while_regions(extent_type.clone());
        let regions = regions.iter().map(|region| builder.import_region(region.entry_region_ref())).collect();
        let outputs = builder
            .add_instruction(
                TestOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![extent, state],
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_count(), 3);
        assert_eq!(jvp.output_count(), 3);
        let outputs = jvp
            .interpret(vec![dimension(&extent_type, 4), array(Array::scalar(1.0)), array(Array::scalar(3.0))])
            .unwrap();
        assert!(matches!(&outputs[0], TestValue::Dimension(value) if value.extent() == 4));
        assert!(matches!(&outputs[1], TestValue::Array(value) if value.to_f64s() == vec![8.0]));
        assert!(matches!(&outputs[2], TestValue::Array(value) if value.to_f64s() == vec![24.0]));

        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![dimension(&extent_type, 4), array(Array::scalar(1.0))])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        let mut pullback_inputs = vec![array(Array::scalar(1.0))];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![array(Array::scalar(8.0))]),);
    }

    #[test]
    fn test_composite_bounded_while_pullback_supports_batched_predicates() {
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));

        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = condition_builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let limits = condition_builder.add_constant(array(Array::vector(vec![2.0, 4.0, 8.0])));
        let predicate = condition_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![state, limits],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = body_builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let doubled = body_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![state, state],
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = builder.add_input(ArrayIrType::Array(vector_type));
        let regions =
            vec![builder.import_region(condition.entry_region_ref()), builder.import_region(body.entry_region_ref())];
        let outputs = builder
            .add_instruction(
                TestOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![state],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let mut primal_outputs =
            linearization.primal().interpret(vec![array(Array::vector(vec![1.0, 1.0, 1.0]))]).unwrap();
        assert_eq!(primal_outputs[0], array(Array::vector(vec![2.0, 4.0, 8.0])));
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![2.0, 4.0, 8.0]))]),
        );
    }

    #[test]
    fn test_composite_bounded_while_differentiation_supports_batched_predicates_with_dimension_state() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));

        // Condition: a per-item predicate `state < [2, 4, 8]` that ignores the loop-invariant dimension carry.
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let state = condition_builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let limits = condition_builder.add_constant(array(Array::vector(vec![2.0, 4.0, 8.0])));
        let predicate = condition_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![state, limits],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Body: the dimension carry is forwarded unchanged and the array carry doubles.
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = body_builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let state = body_builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let doubled = body_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![state, state],
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![extent, doubled], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let state = builder.add_input(ArrayIrType::Array(vector_type));
        let regions =
            vec![builder.import_region(condition.entry_region_ref()), builder.import_region(body.entry_region_ref())];
        let outputs = builder
            .add_instruction(
                TestOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![extent, state],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // Starting from `[1, 1, 1]`, item `i` doubles `i + 1` times before its own predicate turns false, so the
        // primal outputs are `[2, 4, 8]` and the per-item tangent scale factors are the matching `[2, 4, 8]`. The
        // dimension carry has an empty tangent space, so it contributes neither a tangent input nor a tangent output.
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_count(), 3);
        assert_eq!(jvp.output_count(), 3);
        let outputs = jvp
            .interpret(vec![
                dimension(&extent_type, 4),
                array(Array::vector(vec![1.0, 1.0, 1.0])),
                array(Array::vector(vec![1.0, 1.0, 1.0])),
            ])
            .unwrap();
        assert_eq!(outputs[0], dimension(&extent_type, 4));
        assert_eq!(outputs[1], array(Array::vector(vec![2.0, 4.0, 8.0])));
        assert_eq!(outputs[2], array(Array::vector(vec![2.0, 4.0, 8.0])));

        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![dimension(&extent_type, 4), array(Array::vector(vec![1.0, 1.0, 1.0]))])
            .unwrap();
        assert_eq!(primal_outputs[0], dimension(&extent_type, 4));
        assert_eq!(primal_outputs[1], array(Array::vector(vec![2.0, 4.0, 8.0])));
        let residuals = primal_outputs.split_off(2);
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![2.0, 4.0, 8.0]))]),
        );
    }

    #[test]
    fn test_composite_bounded_while_pullback_threads_invariant_dimension_residuals_as_scan_carries() {
        let extent_variable = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let extent_type = DimensionType::new(extent_variable.clone());
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_variable)]));

        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        condition_builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let counter = condition_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let limit = condition_builder.add_constant(array(Array::scalar(2_i64)));
        let predicate = condition_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![counter, limit],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = body_builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let vector = body_builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let counter = body_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let reshaped = body_builder
            .add_instruction(TestOperation::from(ReshapeOperation::new()), Vec::new(), vec![vector, extent])
            .unwrap()[0];
        let one = body_builder.add_constant(array(Array::scalar(1_i64)));
        let next_counter = body_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![counter, one],
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![extent, reshaped, next_counter],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let vector = builder.add_input(ArrayIrType::Array(vector_type));
        let counter = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let regions =
            vec![builder.import_region(condition.entry_region_ref()), builder.import_region(body.entry_region_ref())];
        let outputs = builder
            .add_instruction(
                TestOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![extent, vector, counter],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let rendered_primal = linearization.primal().to_string();
        let rendered_tangent = linearization.tangent().to_string();
        assert!(rendered_primal.contains("scan [carry_count=1"), "{rendered_primal}");
        assert!(!rendered_primal.contains("dimension_to_scalar"), "{rendered_primal}");
        assert!(!rendered_tangent.contains("dimension_from_scalar"), "{rendered_tangent}");
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                dimension(&extent_type, 3),
                array(Array::vector(vec![1.0, 2.0, 3.0])),
                array(Array::scalar(0_i64)),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(3);
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![1.0, 1.0, 1.0]))]),
        );
    }

    #[test]
    fn test_composite_bounded_while_pullback_stacks_varying_dimension_residuals_through_scalar_gateways() {
        let iteration_variable = DimensionVariable::new("iteration", DimensionBounds::positive(Some(4)).unwrap());
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let scalar_u64 = ArrayType::scalar(DataType::U64);

        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(ArrayIrType::Array(scalar_f64.clone()));
        let counter = condition_builder.add_input(ArrayIrType::Array(scalar_u64.clone()));
        let limit = condition_builder.add_constant(array(Array::scalar(3_u64)));
        let predicate = condition_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![counter, limit],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = body_builder.add_input(ArrayIrType::Array(scalar_f64.clone()));
        let counter = body_builder.add_input(ArrayIrType::Array(scalar_u64.clone()));
        let iteration = body_builder
            .add_instruction(
                TestOperation::from(DimensionFromScalarOperation::new(iteration_variable)),
                Vec::new(),
                vec![counter],
            )
            .unwrap()[0];
        let repeated = body_builder
            .add_instruction(
                TestOperation::from(BroadcastOperation::new(Vec::new())),
                Vec::new(),
                vec![state, iteration],
            )
            .unwrap()[0];
        let next_state = body_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(ReduceOperation::new(vec![0], ReductionKind::Sum))),
                Vec::new(),
                vec![repeated],
            )
            .unwrap()[0];
        let one = body_builder.add_constant(array(Array::scalar(1_u64)));
        let next_counter = body_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![counter, one],
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![next_state, next_counter],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = builder.add_input(ArrayIrType::Array(scalar_f64));
        let counter = builder.add_input(ArrayIrType::Array(scalar_u64));
        let regions =
            vec![builder.import_region(condition.entry_region_ref()), builder.import_region(body.entry_region_ref())];
        let outputs = builder
            .add_instruction(
                TestOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![state, counter],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let rendered_primal = linearization.primal().to_string();
        let rendered_tangent = linearization.tangent().to_string();
        assert!(rendered_primal.contains("dimension_to_scalar"), "{rendered_primal}");
        assert!(rendered_tangent.contains("dimension_from_scalar"), "{rendered_tangent}");
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![array(Array::scalar(2.0)), array(Array::scalar(1_u64))])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        let mut pullback_inputs = vec![array(Array::scalar(1.0))];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![array(Array::scalar(2.0))]));
    }
}
