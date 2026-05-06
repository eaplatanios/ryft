use std::fmt::Debug;

use crate::operations::InterpretableOperation;
use crate::operations::arithmetic::{AddOperation, SupportsAdd};
use crate::operations::constants::SupportsZeroLike;
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::tracing::engines::{Engine, Tracer, TracingContext, TracingEngine};
use crate::tracing::{Program, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::Differentiable;
use crate::tracing_v2::linear::linearize;
use crate::tracing_v2::operations::{SupportsNeg, SupportsScale};
use crate::tracing_v2::{
    DifferentiableEngine, DifferentiableOperation, DifferentiableOperationTracingEngine, DifferentiableTracingEngine,
    DifferentiationError,
};
use crate::types::Typed;

/// Evaluates `function` on `primals` and propagates the supplied tangent values forward.
///
/// The returned pair is `(primal_output, tangent_output)`. Architecturally, [`jvp`] is the most
/// direct forward-mode transform in the crate: it either traces the body once to build a staged
/// pushforward or stages the whole JVP into an outer trace if the inputs are already symbolic.
/// Primitive-specific local JVP rules live in [`crate::tracing_v2::operations`]; [`jvp`] is the
/// orchestration layer that selects the concrete or traced execution path.
#[allow(private_bounds, private_interfaces)]
pub fn jvp<
    'engine,
    E: Engine,
    F: FnOnce(D::FunctionInput) -> D::FunctionOutput,
    Input: Parameterized<D, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<D>,
    D: JvpDispatch<'engine, E, Input, Output, Marker>,
    Marker,
>(
    engine: &'engine E,
    function: F,
    primals: Input,
    tangents: Input::To<D::Tangent>,
) -> Result<(Output, Output::To<D::Tangent>), TracingError>
where
    Input::Family: ParameterizedFamily<D::Tangent>,
    Output::Family: ParameterizedFamily<D::Tangent>,
{
    D::invoke(engine, function, primals, tangents)
}

/// Marker selecting concrete-value [`jvp`] dispatch.
#[doc(hidden)]
pub struct JvpDispatchValueMarker;

/// Marker selecting already-traced [`jvp`] dispatch.
#[doc(hidden)]
pub struct JvpDispatchTracerMarker;

/// Dispatch trait used by [`jvp`] so it can operate both on concrete values and on already traced values.
///
/// The public transform is intentionally small; this trait is where the concrete, traced, and
/// batched execution strategies branch apart.
pub(crate) trait JvpDispatch<'engine, E: Engine, Input, Output, Marker>:
    Differentiable<E::Type> + Parameter + Sized
where
    Input: Parameterized<Self, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<Self>,
    Input::Family: ParameterizedFamily<Self::Tangent>,
    Output::Family: ParameterizedFamily<Self::Tangent>,
{
    /// Input type expected by the user-provided function.
    type FunctionInput;

    /// Output type produced by the user-provided function.
    type FunctionOutput;

    /// Invokes [`jvp`] for one leaf regime.
    fn invoke<F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput>(
        engine: &'engine E,
        function: F,
        primals: Input,
        tangents: Input::To<Self::Tangent>,
    ) -> Result<(Output, Output::To<Self::Tangent>), TracingError>;
}

/// Concrete-value dispatch for [`jvp`]: traces the user function with [`Tracer`] to build a staged
/// pushforward via [`linearize`] and evaluates it at the supplied tangents.
impl<
    'engine,
    E: DifferentiableEngine<Value = V> + 'static,
    V: Value<E::Type> + Differentiable<E::Type, Tangent = E::Tangent> + Parameterized<V, ParameterStructure: PartialEq>,
    Input: Parameterized<
            V,
            Family: for<'call> ParameterizedFamily<Tracer<'call, DifferentiableOperationTracingEngine<E>>>,
            ParameterStructure: Debug + PartialEq,
            To<V> = Input,
        >,
    Output: for<'call> Parameterized<
            V,
            Family: ParameterizedFamily<Tracer<'call, DifferentiableOperationTracingEngine<E>>>,
            To<Tracer<'call, DifferentiableOperationTracingEngine<E>>>: Parameterized<
                Tracer<'call, DifferentiableOperationTracingEngine<E>>,
                To<V> = Output,
            >,
            To<V> = Output,
        >,
> JvpDispatch<'engine, E, Input, Output, JvpDispatchValueMarker> for V
where
    E::DifferentiableOperationCarrier: DifferentiableOperation<DifferentiableOperationTracingEngine<E>>,
    <E::LinearEngine as crate::tracing_v2::LinearizableEngine>::LinearOperationCarrier: InterpretableOperation<E::Type, E::Tangent>
        + SupportsNeg<E::Type, E::Tangent>
        + SupportsAdd<E::Type, E::Tangent>
        + SupportsScale<E::Type, E::Tangent>,
    Input::Family: ParameterizedFamily<E::Tangent>,
    Output::Family: ParameterizedFamily<E::Tangent>,
{
    type FunctionInput = Input::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>;
    type FunctionOutput = Output::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>;

    fn invoke<F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput>(
        engine: &'engine E,
        function: F,
        primals: Input,
        tangents: Input::To<Self::Tangent>,
    ) -> Result<(Output, Output::To<Self::Tangent>), TracingError> {
        let primal_structure = primals.parameter_structure();
        let tangent_structure = tangents.parameter_structure();
        if primal_structure != tangent_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{primal_structure:?}"),
                right_structure: format!("{tangent_structure:?}"),
            }
            .into());
        }

        let (primal_output, tangent_program): (
            Output,
            Program<
                E::Type,
                E::Tangent,
                <E::LinearEngine as crate::tracing_v2::LinearizableEngine>::LinearOperationCarrier,
                Input::To<E::Tangent>,
                Output::To<E::Tangent>,
            >,
        ) = linearize(engine, |input| Ok(function(input)), primals)?;
        let tangent_output = tangent_program.interpret(tangents)?;
        Ok((primal_output, tangent_output))
    }
}

/// Already-traced dispatch for [`jvp`]: replays the user function symbolically inside an enclosing
/// [`Tracer`] scope, staging both the primal output and tangent propagation as part of the outer
/// compiled program.
impl<
    'engine,
    E: DifferentiableTracingEngine<Value = V> + TracingEngine + 'static,
    V: Traceable<E::Type> + Differentiable<E::Type> + Parameterized<V, ParameterStructure = Placeholder>,
    Input,
    Output,
> JvpDispatch<'engine, E, Input, Output, JvpDispatchTracerMarker> for Tracer<'engine, E>
where
    E::OperationCarrier:
        DifferentiableOperation<TracingContext<'engine, E>> + SupportsZeroLike<E::Type, V> + SupportsAdd<E::Type, V>,
    E::LinearOperationCarrier<'engine>: InterpretableOperation<E::Type, Tracer<'engine, E>>,
    Input: Parameterized<Tracer<'engine, E>, To<Tracer<'engine, E>> = Input>,
    Input::Family: ParameterizedFamily<Tracer<'engine, E>> + ParameterizedFamily<V> + ParameterizedFamily<E::Type>,
    Input::To<E::Type>: Parameterized<E::Type, To<Tracer<'engine, E>> = Input>,
    Input::ParameterStructure: Debug + PartialEq,
    Output: Parameterized<Tracer<'engine, E>, To<Tracer<'engine, E>> = Output>,
    Output::Family: ParameterizedFamily<Tracer<'engine, E>> + ParameterizedFamily<V> + ParameterizedFamily<E::Type>,
    Output::To<E::Type>: Parameterized<E::Type, To<Tracer<'engine, E>> = Output>,
    AddOperation: InterpretableOperation<E::Type, Tracer<'engine, E>>,
{
    type FunctionInput = Input;
    type FunctionOutput = Output;

    fn invoke<F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput>(
        _engine: &'engine E,
        function: F,
        primals: Input,
        tangents: Input::To<Self::Tangent>,
    ) -> Result<(Output, Output::To<Self::Tangent>), TracingError> {
        let primal_structure = primals.parameter_structure();
        let tangent_structure = tangents.parameter_structure();
        if primal_structure != tangent_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{primal_structure:?}"),
                right_structure: format!("{tangent_structure:?}"),
            }
            .into());
        }

        let traced_primals = primals.into_parameters().collect::<Vec<_>>();
        let traced_tangents = tangents.into_parameters().collect::<Vec<_>>();
        let Some(tracing_context) = traced_primals.first().map(|traced_primal| traced_primal.context.clone()) else {
            return Err(DifferentiationError::MissingTracedJvpInputLeaves.into());
        };
        let staged_input_types = Input::To::<E::Type>::from_parameters(
            primal_structure,
            traced_primals.iter().map(|traced_primal| traced_primal.r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let (primal_output_types, traced_program) =
            tracing_context.engine.trace(|staged_input| Ok(function(staged_input)), staged_input_types)?;
        let output_structure = primal_output_types.parameter_structure();
        let (traced_primal_output, pushforward) = tracing_context.linearize(&traced_program, traced_primals)?;
        let traced_tangent_output = pushforward.interpret(traced_tangents)?;
        Ok((
            Output::from_parameters(output_structure.clone(), traced_primal_output)?,
            Output::from_parameters(output_structure, traced_tangent_output)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::fmt::Display;
    use std::ops::{Add, Div, Mul, Neg, Sub};
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_macros::Parameter;

    use crate::macros::check_count;
    use crate::operations::Operation;
    use crate::operations::arithmetic::{
        AddOperation, MulOperation, SubOperation, SupportsAdd, SupportsMul, SupportsSub,
    };
    use crate::operations::constants::{
        One, OneLike, OneOperation, SupportsOne, SupportsZero, Zero, ZeroLike, ZeroOperation,
    };
    use crate::parameters::{ParameterError, Parameterized};
    use crate::tracing::TranspositionContext;
    use crate::tracing::engines::{Engine, ScalarEngine, TracingContext, TracingEngine};
    use crate::tracing::transposition::LinearOperation;
    use crate::tracing::{AtomId, Program, ProgramBuilder, Traceable, Value};
    use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
    use crate::tracing_v2::operations::{NegOperation, SupportsNeg, SupportsScale};
    use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation, LinearizableEngine};
    use crate::tracing_v2::{LinearScalarOperation, ScalarOperation, Sin};
    use crate::types::{DataType, Typed};

    use super::*;

    #[derive(Copy, Clone, Debug, PartialEq, Parameter)]
    struct DistinctPrimal(f64);

    impl Display for DistinctPrimal {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            Display::fmt(&self.0, formatter)
        }
    }

    impl Typed<DataType> for DistinctPrimal {
        fn r#type(&self) -> Cow<'_, DataType> {
            Cow::Owned(DataType::F64)
        }
    }

    impl Traceable<DataType> for DistinctPrimal {}
    impl Value<DataType> for DistinctPrimal {}
    impl Differentiable<DataType> for DistinctPrimal {
        type Tangent = DistinctTangent;

        fn tangent_type(&self) -> Result<Self::Tangent, TracingError> {
            Ok(DistinctTangent(0.0))
        }
    }

    impl Add for DistinctPrimal {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    impl Sub for DistinctPrimal {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }

    impl Mul for DistinctPrimal {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Self(self.0 * rhs.0)
        }
    }

    impl Div for DistinctPrimal {
        type Output = Self;

        fn div(self, rhs: Self) -> Self::Output {
            Self(self.0 / rhs.0)
        }
    }

    impl Neg for DistinctPrimal {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }

    impl Zero<DataType> for DistinctPrimal {
        fn zero(r#type: &DataType) -> Result<Self, TracingError> {
            assert_eq!(r#type, &DataType::F64);
            Ok(Self(0.0))
        }
    }

    impl One<DataType> for DistinctPrimal {
        fn one(r#type: &DataType) -> Result<Self, TracingError> {
            assert_eq!(r#type, &DataType::F64);
            Ok(Self(1.0))
        }
    }

    impl ZeroLike for DistinctPrimal {
        fn zero_like(&self) -> Self {
            Self(0.0)
        }
    }

    impl OneLike for DistinctPrimal {
        fn one_like(&self) -> Self {
            Self(1.0)
        }
    }

    #[derive(Copy, Clone, Debug, PartialEq, Parameter)]
    struct DistinctTangent(f64);

    impl Display for DistinctTangent {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            Display::fmt(&self.0, formatter)
        }
    }

    impl Typed<DataType> for DistinctTangent {
        fn r#type(&self) -> Cow<'_, DataType> {
            Cow::Owned(DataType::F64)
        }
    }

    impl Traceable<DataType> for DistinctTangent {}

    impl Add for DistinctTangent {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    impl Sub for DistinctTangent {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }

    impl Mul for DistinctTangent {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Self(self.0 * rhs.0)
        }
    }

    impl Neg for DistinctTangent {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }

    impl Zero<DataType> for DistinctTangent {
        fn zero(r#type: &DataType) -> Result<Self, TracingError> {
            assert_eq!(r#type, &DataType::F64);
            Ok(Self(0.0))
        }
    }

    impl One<DataType> for DistinctTangent {
        fn one(r#type: &DataType) -> Result<Self, TracingError> {
            assert_eq!(r#type, &DataType::F64);
            Ok(Self(1.0))
        }
    }

    impl ZeroLike for DistinctTangent {
        fn zero_like(&self) -> Self {
            Self(0.0)
        }
    }

    impl OneLike for DistinctTangent {
        fn one_like(&self) -> Self {
            Self(1.0)
        }
    }

    #[derive(Clone, Debug)]
    enum DistinctLinearOperation {
        Zero(ZeroOperation<DataType>),
        One(OneOperation<DataType>),
        Neg,
        Add,
        Sub,
        ScaleByTangent { factor: DistinctTangent },
        ScaleByPrimal { factor: DistinctPrimal },
    }

    impl Operation<DataType> for DistinctLinearOperation {
        fn name(&self) -> &'static str {
            match self {
                Self::Zero(operation) => operation.name(),
                Self::One(operation) => operation.name(),
                Self::Neg => Operation::<DataType>::name(&NegOperation),
                Self::Add => Operation::<DataType>::name(&AddOperation),
                Self::Sub => Operation::<DataType>::name(&SubOperation),
                Self::ScaleByTangent { .. } | Self::ScaleByPrimal { .. } => "scale",
            }
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, crate::types::TypeError> {
            match self {
                Self::Zero(operation) => operation.infer_output_types(input_types),
                Self::One(operation) => operation.infer_output_types(input_types),
                Self::Neg => NegOperation.infer_output_types(input_types),
                Self::Add => Operation::<DataType>::infer_output_types(&AddOperation, input_types),
                Self::Sub => Operation::<DataType>::infer_output_types(&SubOperation, input_types),
                Self::ScaleByTangent { .. } | Self::ScaleByPrimal { .. } => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0]])
                }
            }
        }
    }

    impl InterpretableOperation<DataType, DistinctTangent> for DistinctLinearOperation {
        fn interpret(&self, inputs: &[DistinctTangent]) -> Result<Vec<DistinctTangent>, TracingError> {
            match self {
                Self::Zero(operation) => operation.interpret(inputs),
                Self::One(operation) => operation.interpret(inputs),
                Self::Neg => NegOperation.interpret(inputs),
                Self::Add => AddOperation.interpret(inputs),
                Self::Sub => SubOperation.interpret(inputs),
                Self::ScaleByTangent { factor } => {
                    check_count!("input", inputs, 1, TracingError);
                    Ok(vec![DistinctTangent(factor.0 * inputs[0].0)])
                }
                Self::ScaleByPrimal { factor } => {
                    check_count!("input", inputs, 1, TracingError);
                    Ok(vec![DistinctTangent(factor.0 * inputs[0].0)])
                }
            }
        }
    }

    impl SupportsZero<DataType, DistinctTangent> for DistinctLinearOperation {
        fn zero_operation(r#type: DataType) -> Self {
            Self::Zero(ZeroOperation::new(r#type))
        }

        fn as_zero_operation(&self) -> Option<&ZeroOperation<DataType>> {
            match self {
                Self::Zero(operation) => Some(operation),
                _ => None,
            }
        }
    }

    impl SupportsOne<DataType, DistinctTangent> for DistinctLinearOperation {
        fn one_operation(r#type: DataType) -> Self {
            Self::One(OneOperation::new(r#type))
        }
    }

    impl SupportsNeg<DataType, DistinctTangent> for DistinctLinearOperation {
        fn neg_operation() -> Self {
            Self::Neg
        }
    }

    impl SupportsAdd<DataType, DistinctTangent> for DistinctLinearOperation {
        fn add_operation() -> Self {
            Self::Add
        }
    }

    impl SupportsSub<DataType, DistinctTangent> for DistinctLinearOperation {
        fn sub_operation() -> Self {
            Self::Sub
        }
    }

    impl SupportsScale<DataType, DistinctTangent> for DistinctLinearOperation {
        fn scale_operation(factor: DistinctTangent) -> Self {
            Self::ScaleByTangent { factor }
        }
    }

    impl SupportsScale<DataType, DistinctTangent, DistinctPrimal> for DistinctLinearOperation {
        fn scale_operation(factor: DistinctPrimal) -> Self {
            Self::ScaleByPrimal { factor }
        }
    }

    impl LinearOperation<DataType, DistinctTangent, DistinctLinearOperation> for DistinctLinearOperation {
        fn transpose(
            &self,
            context: &mut TranspositionContext<DataType, DistinctTangent, DistinctLinearOperation>,
            output_cotangents: &[Option<AtomId>],
        ) -> Result<Vec<Option<AtomId>>, TracingError> {
            check_count!("output", output_cotangents, 1, TracingError);
            let Some(output_cotangent) = output_cotangents[0] else {
                return Ok(match self {
                    Self::Zero(_) | Self::One(_) => vec![],
                    Self::Neg | Self::ScaleByTangent { .. } | Self::ScaleByPrimal { .. } => vec![None],
                    Self::Add | Self::Sub => vec![None, None],
                });
            };
            match self {
                Self::Zero(_) | Self::One(_) => Ok(vec![]),
                Self::Neg => {
                    let inputs = context.stage(Self::Neg, &[output_cotangent])?;
                    check_count!("output", inputs, 1, TracingError);
                    Ok(vec![Some(inputs[0])])
                }
                Self::Add => Ok(vec![Some(output_cotangent), Some(output_cotangent)]),
                Self::Sub => {
                    let right_inputs = context.stage(Self::Neg, &[output_cotangent])?;
                    check_count!("output", right_inputs, 1, TracingError);
                    Ok(vec![Some(output_cotangent), Some(right_inputs[0])])
                }
                Self::ScaleByTangent { factor } => {
                    let inputs = context.stage(Self::ScaleByTangent { factor: *factor }, &[output_cotangent])?;
                    check_count!("output", inputs, 1, TracingError);
                    Ok(vec![Some(inputs[0])])
                }
                Self::ScaleByPrimal { factor } => {
                    let inputs = context.stage(Self::ScaleByPrimal { factor: *factor }, &[output_cotangent])?;
                    check_count!("output", inputs, 1, TracingError);
                    Ok(vec![Some(inputs[0])])
                }
            }
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct DistinctTangentEngine;

    impl Engine for DistinctTangentEngine {
        type Type = DataType;
        type Value = DistinctTangent;

        fn zero(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
            DistinctTangent::zero(r#type)
        }

        fn one(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
            DistinctTangent::one(r#type)
        }
    }

    impl LinearizableEngine for DistinctTangentEngine {
        type LinearOperationCarrier = DistinctLinearOperation;
    }

    #[derive(Clone, Debug)]
    enum DistinctPrimalOperation {
        Add,
        Mul,
    }

    impl Operation<DataType> for DistinctPrimalOperation {
        fn name(&self) -> &'static str {
            match self {
                Self::Add => Operation::<DataType>::name(&AddOperation),
                Self::Mul => Operation::<DataType>::name(&MulOperation),
            }
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, crate::types::TypeError> {
            match self {
                Self::Add => Operation::<DataType>::infer_output_types(&AddOperation, input_types),
                Self::Mul => Operation::<DataType>::infer_output_types(&MulOperation, input_types),
            }
        }
    }

    impl InterpretableOperation<DataType, DistinctPrimal> for DistinctPrimalOperation {
        fn interpret(&self, inputs: &[DistinctPrimal]) -> Result<Vec<DistinctPrimal>, TracingError> {
            match self {
                Self::Add => AddOperation.interpret(inputs),
                Self::Mul => MulOperation.interpret(inputs),
            }
        }
    }

    impl SupportsAdd<DataType, DistinctPrimal> for DistinctPrimalOperation {
        fn add_operation() -> Self {
            Self::Add
        }
    }

    impl SupportsMul<DataType, DistinctPrimal> for DistinctPrimalOperation {
        fn mul_operation() -> Self {
            Self::Mul
        }
    }

    impl<E: DifferentiableEngine<Type = DataType, Value = DistinctPrimal>> DifferentiableOperation<E>
        for DistinctPrimalOperation
    where
        <E::LinearEngine as crate::tracing_v2::LinearizableEngine>::LinearOperationCarrier:
            SupportsAdd<DataType, E::Tangent>,
        <E::LinearEngine as crate::tracing_v2::LinearizableEngine>::LinearOperationCarrier:
            SupportsScale<DataType, E::Tangent, DistinctPrimal>,
    {
        fn jvp(
            &self,
            context: &mut JvpContext<'_, E>,
            inputs: &[JvpTracer<DistinctPrimal, AtomId>],
        ) -> Result<Vec<JvpTracer<DistinctPrimal, AtomId>>, TracingError> {
            match self {
                Self::Add => AddOperation.jvp(context, inputs),
                Self::Mul => MulOperation.jvp(context, inputs),
            }
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct DistinctPrimalEngine {
        linear_engine: DistinctTangentEngine,
    }

    impl DistinctPrimalEngine {
        fn new() -> Self {
            Self { linear_engine: DistinctTangentEngine }
        }
    }

    impl Engine for DistinctPrimalEngine {
        type Type = DataType;
        type Value = DistinctPrimal;

        fn zero(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
            DistinctPrimal::zero(r#type)
        }

        fn one(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
            DistinctPrimal::one(r#type)
        }
    }

    impl TracingEngine for DistinctPrimalEngine {
        type OperationCarrier = DistinctPrimalOperation;
    }

    impl LinearizableEngine for DistinctPrimalEngine {
        type LinearOperationCarrier = LinearScalarOperation<DistinctPrimal>;
    }

    impl DifferentiableEngine for DistinctPrimalEngine {
        type Tangent = DistinctTangent;
        type LinearEngine = DistinctTangentEngine;
        type DifferentiableOperationCarrier = DistinctPrimalOperation;

        fn linear_engine(&self) -> &Self::LinearEngine {
            &self.linear_engine
        }
    }

    /// Validates that [`TracingContext`] can host a JVP rule like [`AddOperation`] when its
    /// `Value` is `Tracer<E>`: the rule stages its primal effect through the underlying engine and
    /// its tangent effect through the context's `LinearOperation` carrier.
    #[test]
    fn tracing_context_dispatches_add_jvp_with_traced_primals() {
        let engine = ScalarEngine::<f64>::new();
        let outer_builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let outer_input_a = outer_builder.borrow_mut().add_input(crate::types::DataType::F64);
        let outer_input_b = outer_builder.borrow_mut().add_input(crate::types::DataType::F64);
        let outer_tracing_context = TracingContext::new(&engine, outer_builder.clone());
        let primal_a = outer_tracing_context.tracer(outer_input_a, None);
        let primal_b = outer_tracing_context.tracer(outer_input_b, None);

        let linear_builder = Rc::new(RefCell::new(ProgramBuilder::<
            DataType,
            Tracer<'_, ScalarEngine<f64>>,
            LinearScalarOperation<Tracer<'_, ScalarEngine<f64>>>,
        >::new()));
        let tangent_a = linear_builder.borrow_mut().add_input(crate::types::DataType::F64);
        let tangent_b = linear_builder.borrow_mut().add_input(crate::types::DataType::F64);
        let mut context = JvpContext::new(&outer_tracing_context, linear_builder.clone());

        let outputs = AddOperation
            .jvp(
                &mut context,
                &[
                    JvpTracer { primal: primal_a, tangent: tangent_a },
                    JvpTracer { primal: primal_b, tangent: tangent_b },
                ],
            )
            .expect("AddOperation::jvp should run on a TracingContext");

        assert_eq!(outputs.len(), 1);
        assert_eq!(linear_builder.borrow().instructions.len(), 1);
        assert_eq!(outer_builder.borrow().instructions.len(), 1);
    }

    #[test]
    fn concrete_jvp_supports_distinct_primal_and_tangent_types() {
        let engine = DistinctPrimalEngine::new();

        let (primal, tangent): (DistinctPrimal, DistinctTangent) = jvp(
            &engine,
            |(left, right)| left + right,
            (DistinctPrimal(2.0), DistinctPrimal(5.0)),
            (DistinctTangent(3.0), DistinctTangent(7.0)),
        )
        .unwrap();

        assert_eq!(primal, DistinctPrimal(7.0));
        assert_eq!(tangent, DistinctTangent(10.0));

        let (_, pushforward): (
            DistinctPrimal,
            Program<DataType, DistinctTangent, DistinctLinearOperation, DistinctTangent, DistinctTangent>,
        ) = linearize(&engine, |input| Ok(input.clone() + input), DistinctPrimal(2.0)).unwrap();

        assert_eq!(
            pushforward.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = add %0 %0
                in (%1)
            "}
            .trim_end(),
        );

        let (output, pullback): (
            DistinctPrimal,
            Program<DataType, DistinctTangent, DistinctLinearOperation, DistinctTangent, DistinctTangent>,
        ) = crate::tracing_v2::linear::vjp(&engine, |input| Ok(input.clone() + input), DistinctPrimal(2.0)).unwrap();
        assert_eq!(output, DistinctPrimal(4.0));
        assert_eq!(pullback.interpret(DistinctTangent(4.0)).unwrap(), DistinctTangent(8.0));

        let (product_primal, product_tangent): (DistinctPrimal, DistinctTangent) = jvp(
            &engine,
            |(left, right)| left * right,
            (DistinctPrimal(2.0), DistinctPrimal(5.0)),
            (DistinctTangent(3.0), DistinctTangent(7.0)),
        )
        .unwrap();
        assert_eq!(product_primal, DistinctPrimal(10.0));
        assert_eq!(product_tangent, DistinctTangent(29.0));

        let (reverse_primal, reverse_gradient): (DistinctPrimal, (DistinctTangent, DistinctTangent)) =
            crate::tracing_v2::value_and_grad(
                &engine,
                |(left, right)| left * right,
                (DistinctPrimal(2.0), DistinctPrimal(5.0)),
            )
            .unwrap();
        assert_eq!(reverse_primal, DistinctPrimal(10.0));
        assert_eq!(reverse_gradient, (DistinctTangent(5.0), DistinctTangent(2.0)));
    }

    #[test]
    fn jvp_rejects_mismatched_parameter_structures() {
        let engine = ScalarEngine::<f64>::new();
        let result: Result<(f64, f64), TracingError> =
            jvp(&engine, |xs| xs[0].clone(), vec![2.0f64], vec![1.0f64, 2.0f64]);
        assert!(matches!(
            result,
            Err(TracingError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure,
                right_structure,
            })) if left_structure == format!("{:?}", vec![2.0f64].parameter_structure())
                && right_structure == format!("{:?}", vec![1.0f64, 2.0f64].parameter_structure())
        ));

        let (_, pushforward): (f64, Program<DataType, f64, LinearScalarOperation<f64>, f64, f64>) =
            linearize(&engine, |x| Ok(x.clone() * x.clone() + x.sin()), 2.0f64).unwrap();

        assert_eq!(
            pushforward.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = scale [factor=2] %0
                    %2:f64 = scale [factor=2] %0
                    %3:f64 = add %1 %2
                    %4:f64 = scale [factor=-0.4161468365471424] %0
                    %5:f64 = add %3 %4
                in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn traced_jvp_requires_input_leaves() {
        let engine = ScalarEngine::<f64>::new();
        let empty_primals: Vec<Tracer<'_, ScalarEngine<f64>>> = Vec::new();
        let empty_tangents: Vec<Tracer<'_, ScalarEngine<f64>>> = Vec::new();

        let result: Result<(Vec<Tracer<'_, ScalarEngine<f64>>>, Vec<Tracer<'_, ScalarEngine<f64>>>), TracingError> =
            jvp(&engine, |inputs: Vec<Tracer<'_, ScalarEngine<f64>>>| inputs, empty_primals, empty_tangents);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::MissingTracedJvpInputLeaves))
        ));
    }
}
