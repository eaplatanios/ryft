//! Array IR instantiations of the control-flow operation family contracts.
//!
//! Control-flow operations are universe-neutral: their differentiation and lowering rules describe how residuals are
//! stored across time and how bounded-while state is stacked, without knowing what a program value is. This module
//! supplies the array universe's answers to those questions, where a value is either ordinary array data or a
//! first-class runtime dimension.

use std::sync::Arc;

use crate::arrays::addressing::ArrayAddressing;
use crate::arrays::arrays::Array;
use crate::arrays::broadcasting::Broadcastable;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::operations::{ArrayIrOperation, ArrayOperation};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::{Dimension, DimensionType, Shape};
use crate::arrays::types::ir::ArrayIrType;
use crate::contexts::EagerContext;
use crate::interpretation::InterpretationDriver;
use crate::macros::check_count;
use crate::operations::control_flow::scan::{
    ScanInterpretation, read_scan_iteration, stacked_scan_type, write_scan_iteration,
};
use crate::operations::{
    AddOperation, AndOperation, BroadcastOperation, DimensionFromScalarOperation, DimensionToScalarOperation,
    DynamicUpdateSliceOperation, OneOperation, RUNTIME_DIMENSION_DATA_TYPE, ReduceOperation, ReductionKind, Reshape,
    Select, SelectOperation, Slice, TemporalResidualOperation, TemporalResidualType, UpdateSlice, WhilePredicate,
    WhileResidualStackOperation, WhileResidualStackType, Zero, ZeroOperation,
};
use crate::programs::{Operation, ProgramError, TypeError, Typed, Value, ValueProjection};

// TODO(eaplatanios): Review this.

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
        Self::from(ArrayIrOperation::<A>::Array(ArrayOperation::Broadcast(BroadcastOperation::new(
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

impl<A: Value<Type = ArrayType> + WhilePredicate> WhilePredicate for ArrayIrValue<A> {
    fn any_true(&self) -> Result<bool, ProgramError> {
        match self {
            Self::Array(predicate) => predicate.any_true(),
            Self::Dimension(value) => Err(ProgramError::Concretization {
                message: format!("cannot use first-class dimension `{value}` as a while predicate"),
            }),
        }
    }

    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let Self::Array(predicate) = self else {
            return Err(ProgramError::Concretization {
                message: format!("cannot use first-class dimension `{self}` as a while predicate"),
            });
        };
        match (on_true, on_false) {
            (Self::Array(on_true), Self::Array(on_false)) => Ok(Self::Array(predicate.mask_select(on_true, on_false)?)),
            (Self::Dimension(on_true), Self::Dimension(on_false)) => {
                // Selecting between equal dimension carries (e.g., the loop-invariant mapped extent that structural
                // batching threads through a while loop's state) is the identity for every item, so it needs no
                // scalar predicate. Distinct dimension carries fall back to the scalar-predicate semantics because
                // one dimension value cannot represent independently masked per-item extents.
                if on_true == on_false {
                    return Ok(Self::Dimension(on_true.clone()));
                }
                Ok(Self::Dimension(if predicate.concretize()? { on_true.clone() } else { on_false.clone() }))
            }
            _ => Err(TypeError::invalid(format!(
                "while predicate cannot select between mismatched state types {} and {}",
                on_true.r#type().as_ref(),
                on_false.r#type().as_ref(),
            ))
            .into()),
        }
    }
}

impl<A> ScanInterpretation<EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>> for ArrayIrType
where
    A: Reshape + Slice + UpdateSlice + Value<Type = ArrayType>,
    EagerContext<A, ArrayOperation<A>>: Zero<A>,
{
    fn interpret_scan<D: InterpretationDriver<EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>>>(
        carry_count: usize,
        length: &Dimension,
        reverse: bool,
        context: &EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>,
        driver: &D,
        inputs: &[ArrayIrValue<A>],
    ) -> Result<Vec<ArrayIrValue<A>>, ProgramError> {
        let (inputs, length) = match length {
            Dimension::Static(length) => (inputs, *length),
            Dimension::Dynamic(variable) => {
                let (runtime_length, inputs) =
                    inputs.split_last().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
                let runtime_length = <ArrayIrValue<A> as ValueProjection<DimensionType>>::projected(runtime_length)?;
                if runtime_length.r#type().variable() != variable {
                    return Err(TypeError::invalid(format!(
                        "'scan' runtime length operand has type {} but scan length requires {variable}",
                        runtime_length.r#type().as_ref(),
                    ))
                    .into());
                }
                (inputs, runtime_length.extent())
            }
        };
        let body = driver.region(0)?;
        let y_slice_types = body.interface().output_types()[carry_count..]
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let (initial_carries, stacks) = inputs.split_at(carry_count);
        let mut carries = initial_carries.to_vec();
        let array_context = EagerContext::<A, ArrayOperation<A>>::new();
        let mut accumulators = y_slice_types
            .iter()
            .map(|r#type| {
                let dimensions = r#type
                    .shape()
                    .dimensions()
                    .iter()
                    .map(|dimension| match dimension {
                        Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
                        Dimension::Dynamic(variable) => inputs
                            .iter()
                            .find_map(|input| match input {
                                ArrayIrValue::Dimension(value) if value.r#type().variable() == variable => {
                                    Some(Dimension::Static(value.extent()))
                                }
                                _ => None,
                            })
                            .ok_or_else(|| {
                                TypeError::invalid(format!(
                                    "cannot eagerly allocate scan output {type} because its dynamic dimension \
                                     {variable} is not supplied as a first-class scan input",
                                    r#type = r#type,
                                ))
                            }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                array_context.zero(&stacked_scan_type(&r#type.clone().with_shape(Shape::new(dimensions)), length))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let iterations: Box<dyn Iterator<Item = usize>> =
            if reverse { Box::new((0..length).rev()) } else { Box::new(0..length) };
        for iteration in iterations {
            let mut iteration_inputs = carries.clone();
            iteration_inputs.extend(
                stacks
                    .iter()
                    .map(|stack| {
                        Ok(ArrayIrValue::Array(read_scan_iteration(
                            <ArrayIrValue<A> as ValueProjection<ArrayType>>::projected(stack)?,
                            iteration,
                        )?))
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?,
            );
            let mut iteration_outputs = driver.interpret_region(context, 0, iteration_inputs)?;
            check_count!("output", iteration_outputs, carry_count + y_slice_types.len(), ProgramError);
            let iteration_outputs_to_stack = iteration_outputs.split_off(carry_count);
            carries = iteration_outputs;
            for (accumulator, value) in accumulators.iter_mut().zip(iteration_outputs_to_stack) {
                let value = <ArrayIrValue<A> as ValueProjection<ArrayType>>::into_projected(value)?;
                *accumulator = write_scan_iteration(accumulator.clone(), iteration, value)?;
            }
        }
        carries.extend(accumulators.into_iter().map(ArrayIrValue::Array));
        Ok(carries)
    }
}

impl Select for Array {
    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        // Mirrors the broadcasting `SelectOperation` type-inference contract: the condition must be Boolean-typed,
        // the three operand shapes broadcast together, and the two branch data types promote together to the output
        // data type. The condition is retyped to a branch data type before broadcasting so its Boolean data type
        // acts as a mask rather than promoting into the output.
        assert_eq!(condition.r#type().data_type(), DataType::Boolean, "select condition must have a Boolean data type");
        let output_type = ArrayType::broadcasted(&[
            condition.r#type().into_owned().with_data_type(on_true.r#type().data_type()),
            on_true.r#type().into_owned(),
            on_false.r#type().into_owned(),
        ])
        .map_err(|error| TypeError::invalid(error.to_string()))?;

        // Convert only when promotion requires it. Equal-typed branches retain their original physical storage and
        // arbitrary layouts; conversion remains responsible for the element semantics until its own typed-byte slice.
        let output_data_type = output_type.data_type();
        let on_true = on_true.promoted_to(output_data_type)?;
        let on_false = on_false.promoted_to(output_data_type)?;

        let output_shape = output_type.static_shape().unwrap();
        let condition_shape = condition.r#type().static_shape().unwrap();
        let true_shape = on_true.r#type().static_shape().unwrap();
        let false_shape = on_false.r#type().static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let condition_strides = condition_shape.row_major_strides();
        let true_strides = true_shape.row_major_strides();
        let false_strides = false_shape.row_major_strides();
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let condition_addressing = ArrayAddressing::new(condition.r#type().into_owned())?;
        let true_addressing = ArrayAddressing::new(on_true.r#type().into_owned())?;
        let false_addressing = ArrayAddressing::new(on_false.r#type().into_owned())?;
        let mut output_bytes = vec![0; output_addressing.storage_byte_len()];
        for output_index in 0..output_addressing.element_count() {
            let condition_index = Self::broadcast_index(
                output_index,
                &output_shape,
                &output_strides,
                &condition_shape,
                &condition_strides,
            );
            let condition_range = condition_addressing.byte_range_for_flat_index(condition_index);
            let (source, source_range) = if condition.storage_bytes()[condition_range.start] != 0 {
                let source_index =
                    Self::broadcast_index(output_index, &output_shape, &output_strides, &true_shape, &true_strides);
                (on_true.storage_bytes(), true_addressing.byte_range_for_flat_index(source_index))
            } else {
                let source_index =
                    Self::broadcast_index(output_index, &output_shape, &output_strides, &false_shape, &false_strides);
                (on_false.storage_bytes(), false_addressing.byte_range_for_flat_index(source_index))
            };
            let output_range = output_addressing.byte_range_for_flat_index(output_index);
            output_bytes[output_range].copy_from_slice(&source[source_range]);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(output_bytes)))
    }
}

/// Batched while-predicate semantics for [`Array`]: `any_true` reduces the whole Boolean payload with `or`, and
/// `mask_select` broadcasts the predicate against the operands along its leading (prefix) axes, so predicate item `i`
/// masks the contiguous per-item block of `on_true` / `on_false` elements it governs.
impl crate::operations::control_flow::WhilePredicate for Array {
    fn any_true(&self) -> Result<bool, ProgramError> {
        if !self.r#type().data_type().is_boolean() {
            return Err(ProgramError::Concretization {
                message: format!("cannot use a value of type {} as a Boolean while predicate", self.r#type()),
            });
        }
        let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        Ok((0..addressing.element_count())
            .any(|index| self.storage_bytes()[addressing.byte_range_for_flat_index(index).start] != 0))
    }

    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let predicate_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let true_addressing = ArrayAddressing::new(on_true.r#type().into_owned())?;
        if !self.r#type().data_type().is_boolean()
            || on_true.r#type() != on_false.r#type()
            || predicate_addressing.element_count() == 0
            || !true_addressing.element_count().is_multiple_of(predicate_addressing.element_count())
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "mask_select requires a Boolean predicate whose element count divides congruent operands, but \
                     got predicate {} with operands {} and {}",
                    self.r#type(),
                    on_true.r#type(),
                    on_false.r#type(),
                ),
            });
        }
        let block = true_addressing.element_count() / predicate_addressing.element_count();
        let mut output_bytes = vec![0; true_addressing.storage_byte_len()];
        for index in 0..true_addressing.element_count() {
            let predicate_range = predicate_addressing.byte_range_for_flat_index(index / block);
            let source = if self.storage_bytes()[predicate_range.start] != 0 {
                on_true.storage_bytes()
            } else {
                on_false.storage_bytes()
            };
            let source_range = true_addressing.byte_range_for_flat_index(index);
            output_bytes[source_range.clone()].copy_from_slice(&source[source_range]);
        }
        Ok(Self::new_unchecked(on_true.r#type().into_owned(), Arc::new(output_bytes)))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::{Array, array_type};
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ArrayOperation};
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape};
    use crate::arrays::types::ir::ArrayIrType;
    use crate::arrays::types::layouts::{Layout, StridedLayout};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::ForwardModeDifferentiate;
    use crate::operations::{
        AddOperation, CompareOperation, ComparisonDirection, ConditionOperation, DimensionFromScalarOperation,
        DynamicBroadcastOperation, DynamicReshapeOperation, MulOperation, ReduceOperation, ReductionKind,
        ScanOperation, Select, StopGradientOperation, WhileOperation, ZeroOperation,
    };
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::{Program, ProgramBuilder, Typed};
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
    fn test_composite_condition_tracing_rendering_and_eager_execution() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let operand = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let regions = vec![
            builder.import_region(scale_branch(extent_type.clone(), 2.0).entry_region_ref()),
            builder.import_region(scale_branch(extent_type.clone(), 3.0).entry_region_ref()),
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

        // A dimension carried through a condition is an ordinary structural value: it appears in both branch
        // interfaces and in the composite output signature exactly like the array beside it.
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:dimension<extent ∈ [1, 8)>, %2:f64[] .
                let %3:dimension<extent ∈ [1, 8)>, %4:f64[] = condition %0 %1 %2 [
                    true={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[] .
                        let %2:f64[] = const
                            %3:f64[] = mul %1 %2
                        in (%0, %3)
                    },
                    false={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[] .
                        let %2:f64[] = const
                            %3:f64[] = mul %1 %2
                        in (%0, %3)
                    },
                ]
                in (%3, %4)"},
        );

        // Eager interpretation selects one branch per predicate value and forwards the same dimension either way.
        let boolean = |value: f64| array(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![value]));
        assert_eq!(
            program.interpret(vec![boolean(1.0), dimension(&extent_type, 4), array(Array::scalar(5.0))]),
            Ok(vec![dimension(&extent_type, 4), array(Array::scalar(10.0))]),
        );
        assert_eq!(
            program.interpret(vec![boolean(0.0), dimension(&extent_type, 4), array(Array::scalar(5.0))]),
            Ok(vec![dimension(&extent_type, 4), array(Array::scalar(15.0))]),
        );

        // Relocating the composite program imports both branch regions unchanged, so it renders and executes exactly
        // like its source.
        let mut relocated_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let relocated_inputs = vec![
            relocated_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))),
            relocated_builder.add_input(ArrayIrType::Dimension(extent_type.clone())),
            relocated_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64))),
        ];
        let relocated_outputs = relocated_builder.splice_program(&program, &relocated_inputs).unwrap();
        let relocated = relocated_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(relocated_outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(relocated.to_string(), program.to_string());
        assert_eq!(
            relocated.interpret(vec![boolean(1.0), dimension(&extent_type, 4), array(Array::scalar(5.0))]),
            Ok(vec![dimension(&extent_type, 4), array(Array::scalar(10.0))]),
        );
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
    fn test_composite_condition_jvp_shapes_a_disconnected_dynamic_operand_tangent_from_its_primal() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let array_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
            let left = builder.add_input(ArrayIrType::Array(array_type.clone()));
            let right = builder.add_input(ArrayIrType::Array(array_type.clone()));
            let sum = builder
                .add_instruction(
                    TestOperation::Array(ArrayOperation::from(AddOperation::new())),
                    Vec::new(),
                    vec![left, right],
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![extent, sum], vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let left = builder.add_input(ArrayIrType::Array(array_type.clone()));
        let right = builder.add_input(ArrayIrType::Array(array_type.clone()));
        // Severing the second operand's tangent leaves the fused conditional with one live and one structurally zero
        // dynamic tangent operand, which is exactly the case a type-only nullary zero cannot construct.
        let severed = builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(StopGradientOperation::<ArrayType>::new())),
                Vec::new(),
                vec![right],
            )
            .unwrap()[0];
        let regions = vec![
            builder.import_region(branch().entry_region_ref()),
            builder.import_region(branch().entry_region_ref()),
        ];
        let outputs = builder
            .add_instruction(
                TestOperation::Condition(ConditionOperation::new()),
                regions,
                vec![predicate, extent, left, severed],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![outputs[1]], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();

        // The severed operand's tangent is staged as `zero_like` over its own primal, which pins the runtime extent.
        let jvp = program.jvp().unwrap();
        let rendered = jvp.to_string();
        assert!(rendered.contains("zero_like"), "{rendered}");
        assert_eq!(
            jvp.interpret(vec![
                array(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0])),
                dimension(&extent_type, 3),
                array(Array::vector(vec![1.0, 2.0, 3.0])),
                array(Array::vector(vec![10.0, 20.0, 30.0])),
                array(Array::vector(vec![1.0, 1.0, 1.0])),
                array(Array::vector(vec![5.0, 5.0, 5.0])),
            ]),
            Ok(vec![array(Array::vector(vec![11.0, 22.0, 33.0])), array(Array::vector(vec![1.0, 1.0, 1.0]))]),
        );
    }

    #[test]
    fn test_composite_condition_pullback_shapes_a_dead_dynamic_output_cotangent_from_a_live_peer() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let array_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
            let operand = builder.add_input(ArrayIrType::Array(array_type.clone()));
            let doubled = builder
                .add_instruction(
                    TestOperation::Array(ArrayOperation::from(AddOperation::new())),
                    Vec::new(),
                    vec![operand, operand],
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![extent, doubled, operand],
                    vec![Placeholder; 2],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let operand = builder.add_input(ArrayIrType::Array(array_type.clone()));
        let regions = vec![
            builder.import_region(branch().entry_region_ref()),
            builder.import_region(branch().entry_region_ref()),
        ];
        let outputs = builder
            .add_instruction(
                TestOperation::Condition(ConditionOperation::new()),
                regions,
                vec![predicate, extent, operand],
            )
            .unwrap()
            .to_vec();
        // Keeping only the doubled output leaves the third branch output dead, so its dynamic cotangent reaches the
        // transposed condition as a structural zero that no type-only constructor can build.
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![outputs[1]], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:bool[] .
                let %2:f64[extent] = zero_like %0
                    %3:f64[extent] = condition %1 %0 %2 [
                        true={
                            lambda %0:f64[extent], %1:f64[extent] .
                            let %2:f64[extent] = add %1 %0
                                %3:f64[extent] = add %2 %0
                            in (%3)
                        },
                        false={
                            lambda %0:f64[extent], %1:f64[extent] .
                            let %2:f64[extent] = add %1 %0
                                %3:f64[extent] = add %2 %0
                            in (%3)
                        },
                    ]
                in (%3)"}
            .trim_end(),
        );
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                array(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0])),
                dimension(&extent_type, 3),
                array(Array::vector(vec![1.0, 2.0, 3.0])),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![array(Array::vector(vec![2.0, 2.0, 2.0]))]));
    }

    #[test]
    fn test_composite_scan_pullback_shapes_a_dead_dynamic_carry_cotangent_from_a_live_peer() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let array_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = body_builder.add_input(ArrayIrType::Array(array_type.clone()));
        let second = body_builder.add_input(ArrayIrType::Array(array_type.clone()));
        let sum = body_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![first, second],
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum, second], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = builder.add_input(ArrayIrType::Array(array_type.clone()));
        let second = builder.add_input(ArrayIrType::Array(array_type.clone()));
        let region = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(TestOperation::Scan(ScanOperation::new(2, 2)), vec![region], vec![first, second])
            .unwrap()
            .to_vec();
        // Keeping only the accumulating carry leaves the second carry dead, so the reversed scan needs a dynamic zero
        // cotangent for it. The live first-carry cotangent has exactly that type and supplies the runtime extent.
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![outputs[0]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[extent] .
                let %1:f64[extent] = zero_like %0
                    %2:f64[extent], %3:f64[extent] = scan [carry_count=2, length=2, reverse=true] %0 %1 [
                        body={
                            lambda %0:f64[extent], %1:f64[extent] .
                            let %2:f64[extent] = add %1 %0
                            in (%0, %2)
                        },
                    ]
                in (%2, %3)"}
            .trim_end(),
        );
        assert_eq!(
            pullback.interpret(vec![array(Array::vector(vec![1.0, 1.0, 1.0]))]),
            Ok(vec![array(Array::vector(vec![1.0, 1.0, 1.0])), array(Array::vector(vec![2.0, 2.0, 2.0]))]),
        );
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
                TestOperation::from(DynamicBroadcastOperation::new(Vec::new())),
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
            .add_instruction(TestOperation::from(DynamicReshapeOperation::new()), Vec::new(), vec![vector, extent])
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
                TestOperation::from(DynamicBroadcastOperation::new(Vec::new())),
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

    #[test]
    fn test_array_select() {
        let condition = Array::vector(vec![true, false, true]);
        let on_true = Array::vector(vec![1.0, 2.0, 3.0]);
        let on_false = Array::vector(vec![-1.0, -2.0, -3.0]);
        assert_eq!(Array::select(&condition, &on_true, &on_false).unwrap(), Array::vector(vec![1.0, -2.0, 3.0]));
        // The condition broadcasts against the branches, and the branch data types promote together.
        let broadcast =
            Array::select(&Array::scalar(true), &Array::vector(vec![1.0f32, 2.0]), &Array::vector(vec![-1.0f64, -2.0]))
                .unwrap();
        assert_eq!(broadcast, Array::vector(vec![1.0f64, 2.0]));

        // General broadcasting reads every input through its physical layout and writes one dense output without
        // converting equal-typed branch elements through an intermediate representation.
        let condition_type =
            array_type(DataType::Boolean, &[2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-3, 1])));
        let condition = Array::from_elements(condition_type, &[true, false]).unwrap();
        let true_type =
            array_type(DataType::U16, &[1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![8, -2])));
        let on_true = Array::from_elements(true_type, &[0x1111u16, 0x2222, 0x3333]).unwrap();
        let false_type =
            array_type(DataType::U16, &[2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-4, 2])));
        let on_false = Array::from_elements(false_type, &[0xaaaau16, 0xbbbb]).unwrap();
        let selected = Array::select(&condition, &on_true, &on_false).unwrap();
        assert_eq!(selected.r#type().as_ref(), &array_type(DataType::U16, &[2, 3]));
        assert_eq!(selected.elements::<u16>(), Ok(vec![0x1111, 0x2222, 0x3333, 0xbbbb, 0xbbbb, 0xbbbb]),);
        assert_eq!(selected.storage_bytes(), [0x11, 0x11, 0x22, 0x22, 0x33, 0x33, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb],);
    }

    #[test]
    fn test_array_while_predicate() {
        use crate::operations::WhilePredicate;

        let predicate = Array::vector(vec![false, true]);
        assert_eq!(predicate.any_true(), Ok(true));
        assert_eq!(Array::vector(vec![false, false]).any_true(), Ok(false));
        assert!(Array::vector(vec![1.0]).any_true().is_err());
        // Predicate item `i` masks the contiguous per-item block of operand elements it governs.
        let on_true = Array::from_f64s(array_type(DataType::F64, &[2, 2]), vec![1.0, 2.0, 3.0, 4.0]);
        let on_false = Array::from_f64s(array_type(DataType::F64, &[2, 2]), vec![-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(
            predicate.mask_select(&on_true, &on_false).unwrap(),
            Array::from_f64s(array_type(DataType::F64, &[2, 2]), vec![-1.0, -2.0, 3.0, 4.0]),
        );

        // Predicate and branch layouts are independent of logical masking. The output preserves the congruent branch
        // layout, including its hole, while selecting exact element bytes in logical order.
        let predicate_type =
            array_type(DataType::Boolean, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
        let predicate = Array::from_elements(predicate_type, &[false, true]).unwrap();
        assert_eq!(predicate.any_true(), Ok(true));
        let branch_type =
            array_type(DataType::U16, &[2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![-6, 2])));
        let on_true = Array::from_elements(branch_type.clone(), &[0x1111u16, 0x2222, 0x3333, 0x4444]).unwrap();
        let on_false = Array::from_elements(branch_type.clone(), &[0xaaaau16, 0xbbbb, 0xcccc, 0xdddd]).unwrap();
        let selected = predicate.mask_select(&on_true, &on_false).unwrap();
        assert_eq!(selected.r#type().as_ref(), &branch_type);
        assert_eq!(selected.elements::<u16>(), Ok(vec![0xaaaa, 0xbbbb, 0x3333, 0x4444]));
        assert_eq!(selected.storage_bytes(), [0x33, 0x33, 0x44, 0x44, 0, 0, 0xaa, 0xaa, 0xbb, 0xbb]);
    }
}
