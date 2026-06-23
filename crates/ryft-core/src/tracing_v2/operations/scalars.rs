use std::ops::{Add, Neg, Sub};

use crate::contexts::Context;
use crate::macros::check_count;
use crate::operations::arithmetic::ScaleOperation;
use crate::operations::constants::{One, Zero};
use crate::operations::control_flow::ScanOperation;
use crate::operations::scalars::LinearScalarOperation;
use crate::operations::{InterpretableOperation, Operation, Select};
use crate::payloads::Input;
use crate::programs::{Program, ProgramError, Value};
use crate::tracing_v2::ValueOrCapture;
use crate::tracing_v2::differentiation::CaptureParameterizedOperation;
use crate::tracing_v2::operations::bounds::SupportsConstantOperations;
use crate::tracing_v2::operations::scan::{
    InterpretableNestedProgram, LinearScanBodyInstantiable, LinearScanBodyTransposable,
};
use crate::tracing_v2::operations::select::LinearSelectOperation;
use crate::types::DataType;

impl<V: Value<DataType>, C: Value<DataType>, F: Value<DataType>> CaptureParameterizedOperation<DataType, F>
    for LinearScalarOperation<V, C, F>
{
    type WithCapture<MappedFactor: Value<DataType>> = LinearScalarOperation<V, C, MappedFactor>;
    type WithLocalCapture<MappedFactor: Value<DataType>> = LinearScalarOperation<V, C, MappedFactor>;

    fn try_map_captures<MappedFactor: Value<DataType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<Self::WithCapture<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        match self {
            Self::Zero(operation) => Ok(operation.clone().into()),
            Self::ZeroLike(operation) => Ok(operation.clone().into()),
            Self::One(operation) => Ok(operation.clone().into()),
            Self::OneLike(operation) => Ok(operation.clone().into()),
            Self::Constant(operation) => Ok(operation.clone().into()),
            Self::Neg(operation) => Ok(operation.clone().into()),
            Self::Add(operation) => Ok(operation.clone().into()),
            Self::Sub(operation) => Ok(operation.clone().into()),
            Self::Scale(operation) => {
                Ok(ScaleOperation::<DataType, MappedFactor, Input>::new(map_factor(operation.factor())?).into())
            }
            Self::Select(operation) => Ok(LinearSelectOperation::new(map_factor(operation.condition())?).into()),
            Self::Scan(operation) => {
                let scan = ScanOperation::<DataType, _, _>::new(
                    operation.body().clone(),
                    operation.carry_count(),
                    operation.length(),
                )?
                .with_reverse(operation.reverse())
                .with_unroll(operation.unroll())?
                .with_captures(operation.captures().iter().map(map_factor).collect::<Result<Vec<_>, _>>()?);
                Ok(LinearScalarOperation::Scan(Box::new(scan)))
            }
            Self::CustomVjpCall(call) => {
                Ok(LinearScalarOperation::CustomVjpCall(Box::new(call.map_captures(map_factor)?)))
            }
        }
    }

    fn try_map_local_captures<MappedFactor: Value<DataType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<Self::WithLocalCapture<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        self.try_map_captures(map_factor)
    }
}

impl<V, C> LinearScanBodyInstantiable<DataType, V> for LinearScalarOperation<V, C, ValueOrCapture<DataType, V>>
where
    V: Value<DataType>,
    C: Value<DataType>,
{
    type Instantiated = LinearScalarOperation<V, C, V>;

    fn instantiate_linear_scan_body_factors(&self, residuals: &[V]) -> Result<Self::Instantiated, ProgramError> {
        self.try_map_captures(&mut |factor| factor.instantiate(residuals))
    }
}

impl<V, C> InterpretableNestedProgram<DataType, V> for LinearScalarOperation<V, C, V>
where
    V: Value<DataType>
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + SupportsConstantOperations<DataType>
        + Select<Condition = <V as crate::operations::control_flow::SelectCondition>::Condition>
        + crate::operations::control_flow::SelectCondition,
    C: Value<DataType>,
    V::InterpretationContext: Context<Type = DataType, Constant = C, Value = V> + Zero<DataType, V> + One<DataType, V>,
    ScaleOperation<DataType, V, Input>: InterpretableOperation<DataType, V>,
    crate::operations::constants::ConstantOperation<DataType, V, Input>: InterpretableOperation<DataType, V>,
    crate::operations::scalars::ScalarOperation<C>: InterpretableOperation<DataType, V>,
    Vec<V>: crate::parameters::Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret_nested_program(
        context: &V::InterpretationContext,
        program: &Program<DataType, V, Self, Vec<V>, Vec<V>>,
        input: Vec<V>,
    ) -> Result<Vec<V>, ProgramError>
    where
        Self: Sized,
    {
        program.interpret_with(
            input,
            |_, constant| Ok(constant.clone()),
            |instruction, instruction_inputs| match instruction.operation() {
                Self::CustomVjpCall(operation) => operation.interpret(context, instruction_inputs),
                Self::Zero(operation) => operation.interpret(context, instruction_inputs),
                Self::One(operation) => operation.interpret(context, instruction_inputs),
                Self::Constant(operation) => operation.interpret(context, instruction_inputs),
                Self::ZeroLike(operation) => operation.interpret(context, instruction_inputs),
                Self::OneLike(operation) => operation.interpret(context, instruction_inputs),
                Self::Add(operation) => operation.interpret(context, instruction_inputs),
                Self::Sub(operation) => operation.interpret(context, instruction_inputs),
                Self::Neg(operation) => operation.interpret(context, instruction_inputs),
                Self::Scale(operation) => operation.interpret(context, instruction_inputs),
                Self::Select(operation) => operation.interpret(context, instruction_inputs),
                Self::Scan(operation) => {
                    let input_types =
                        instruction_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                    operation.infer_output_types(input_types.as_slice())?;
                    let body = operation
                        .body()
                        .map_operations(|operation| operation.instantiate_linear_scan_body_factors(&[]))?;
                    let mut state = instruction_inputs.to_vec();
                    for _ in 0..operation.length() {
                        state = Self::interpret_nested_program(context, &body, state)?;
                        check_count!("output", state, operation.carry_count(), ProgramError);
                    }
                    Ok(state)
                }
            },
        )
    }
}

impl<V, C> LinearScanBodyTransposable<DataType, V> for LinearScalarOperation<V, C, ValueOrCapture<DataType, V>>
where
    V: Value<DataType>,
    C: Value<DataType>,
{
    fn transpose_linear_scan_body(
        body: &Program<DataType, V, Self, Vec<V>, Vec<V>>,
    ) -> Result<Program<DataType, V, Self, Vec<V>, Vec<V>>, ProgramError>
    where
        Self: Sized,
    {
        body.transpose()
    }
}
