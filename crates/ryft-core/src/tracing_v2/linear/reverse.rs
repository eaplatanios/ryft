use super::*;
use crate::parameters::Parameter;
use crate::tracing_v2::DifferentiationError;

/// Traces `function` once and returns both its primal output and a reusable pushforward program.
///
/// [`linearize`] is the staged counterpart to [`crate::tracing_v2::jvp`]. Instead of immediately
/// applying a tangent input, it captures the Jacobian-vector product as a staged [`Program`] over
/// linear operations that can be replayed later on any tangent with the same parameter structure.
pub fn linearize<
    'engine,
    E: DifferentiableEngine<Value = V> + 'static,
    F: FnOnce(
        Input::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
    ) -> Result<Output::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>, TracingError>,
    Input: Parameterized<
            V,
            Family: ParameterizedFamily<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
            ParameterStructure: Debug + PartialEq,
        >,
    Output: Parameterized<
            V,
            Family: ParameterizedFamily<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
            To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>: Parameterized<
                Tracer<'engine, DifferentiableOperationTracingEngine<E>>,
                To<V> = Output,
            >,
        >,
    V: Differentiable<E::Type, Tangent = V> + Zero<E::Type>,
>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<(Output, Program<E::Type, V, E::LinearOperationCarrier, Input, Output>), TracingError> {
    let input_structure = primals.parameter_structure();
    let input_primals: Vec<V> = primals.into_parameters().collect();
    let reconstructed_primals = Input::from_parameters(input_structure, input_primals.iter().cloned())?;
    let differentiable_tracing_engine = DifferentiableOperationTracingEngine::new(engine);
    let (primal_output, program) =
        differentiable_tracing_engine.interpret_and_trace(function, reconstructed_primals)?;
    Ok((primal_output, program.linearize(engine, input_primals)?))
}

/// Returns the primal output together with a pullback produced by transposing the staged pushforward.
///
/// [`vjp`] is the reusable reverse-mode primitive in the public API. It traces the primal function,
/// builds the corresponding pushforward program, and then transposes that pushforward into a staged
/// pullback that maps output cotangents back to input cotangents.
#[allow(private_bounds)]
pub fn vjp<
    'engine,
    E: DifferentiableEngine<
            Value = V,
            LinearOperationCarrier: Clone
                                        + InterpretableOperation<E::Type, V>
                                        + LinearOperation<E::Type, V, E::LinearOperationCarrier>
                                        + SupportsZero<E::Type, V>,
        > + 'static,
    F: FnOnce(
        Input::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
    ) -> Result<Output::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>, TracingError>,
    Input: Parameterized<
            V,
            Family: ParameterizedFamily<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
            ParameterStructure: Debug + PartialEq,
        >,
    Output: Parameterized<
            V,
            Family: ParameterizedFamily<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
            To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>: Parameterized<
                Tracer<'engine, DifferentiableOperationTracingEngine<E>>,
                To<V> = Output,
            >,
        >,
    V: Differentiable<E::Type, Tangent = V> + Zero<E::Type>,
>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<(Output, Program<E::Type, V, E::LinearOperationCarrier, Output, Input>), TracingError> {
    let (output, pushforward) = linearize::<E, F, Input, Output, V>(engine, function, primals)?;
    let output_examples = output.parameters().cloned().collect::<Vec<_>>();
    let pullback = pushforward.transpose(output_examples.as_slice())?;
    Ok((output, pullback))
}

impl<'engine, E: TracingEngine + ?Sized> TracingContext<'engine, E> {
    /// Linearizes one traced scalar-output program and stages its pullback with a unit cotangent seed.
    ///
    /// This is the internal core of traced reverse-mode for scalar-output functions. Given a staged
    /// primal body and symbolic primals from this enclosing trace, it builds the pushforward,
    /// transposes it into a pullback, seeds that pullback with a symbolic one, and returns both the
    /// traced scalar output and the traced gradient leaves.
    pub(super) fn value_and_grad<V, Input, Output>(
        self,
        traced_program: &Program<E::Type, V, E::OperationCarrier, Input, Output>,
        traced_primals: Vec<Tracer<'engine, E>>,
    ) -> Result<(Tracer<'engine, E>, Vec<Tracer<'engine, E>>), TracingError>
    where
        V: Traceable<E::Type> + Differentiable<E::Type, Tangent = V> + One<E::Type>,
        Input: Parameterized<V>,
        Output: Parameterized<V>,
        E: DifferentiableTracingEngine<Value = V> + 'static,
        E::OperationCarrier: DifferentiableOperation<TracingContext<'engine, E>>
            + SupportsAdd<E::Type, V>
            + SupportsZeroLike<E::Type, V>
            + 'static,
        <E as DifferentiableTracingEngine>::LinearOperationCarrier<'engine>: Clone
            + InterpretableOperation<E::Type, Tracer<'engine, E>>
            + LinearOperation<
                E::Type,
                Tracer<'engine, E>,
                <E as DifferentiableTracingEngine>::LinearOperationCarrier<'engine>,
            > + SupportsZero<E::Type, Tracer<'engine, E>>,
        AddOperation: InterpretableOperation<E::Type, Tracer<'engine, E>>,
    {
        let (outputs, pushforward) = self.linearize(traced_program, traced_primals)?;
        if outputs.len() != 1 {
            return Err(DifferentiationError::InvalidGradientOutputLeafCount { expected: 1, got: outputs.len() }.into());
        }
        let traced_output = outputs[0].clone();
        let pullback = self.transpose(&pushforward)?;
        let seed_type = traced_output.r#type().into_owned();
        let _ = <V as One<E::Type>>::one(&seed_type)?;
        let seed_value = self.engine.one(&seed_type)?;
        let seed = self.constant(seed_value);
        let traced_gradient = pullback.interpret(vec![seed])?;
        Ok((traced_output, traced_gradient))
    }
}

/// Marker selecting concrete-value [`value_and_grad`] dispatch.
#[doc(hidden)]
pub struct ConcreteValueAndGrad;

/// Marker selecting already-traced [`value_and_grad`] dispatch.
#[doc(hidden)]
pub struct TracedValueAndGrad;

/// Dispatch trait shared by [`grad`] and [`value_and_grad`] so they can operate both on concrete
/// values and on already traced values.
///
/// The trait always produces `(value, gradient)`; [`grad`] is a thin wrapper that drops the primal
/// value, while [`value_and_grad`] exposes the full pair. This keeps the public reverse-mode API
/// compact while allowing concrete replay, traced replay, and batched replay to specialize
/// independently.
#[doc(hidden)]
pub(crate) trait ValueAndGradDispatch<
    E: Engine,
    Input: Parameterized<Self, ParameterStructure: Debug + PartialEq>,
    Mode,
>: Parameter + Sized
{
    /// Primal scalar output value produced for the corresponding input regime.
    type Value;

    /// Traced input type expected by the user-provided function.
    type FunctionInput<'engine>
    where
        E: 'engine;

    /// Traced scalar output type expected from the user-provided function.
    type FunctionOutput<'engine>
    where
        E: 'engine;

    /// Invokes [`value_and_grad`] for one concrete leaf regime.
    fn invoke<'engine, F: FnOnce(Self::FunctionInput<'engine>) -> Self::FunctionOutput<'engine>>(
        engine: &'engine E,
        function: F,
        primals: Input,
    ) -> Result<(Self::Value, Input), TracingError>;
}

/// Concrete-value dispatch for [`value_and_grad`]: evaluates the user function via [`vjp`], checks
/// that the output is a scalar, and pulls back a unit seed to obtain both the primal output and gradient.
impl<
    E: DifferentiableEngine<
            Value = V,
            LinearOperationCarrier: Clone
                                        + InterpretableOperation<E::Type, V>
                                        + LinearOperation<E::Type, V, E::LinearOperationCarrier>
                                        + SupportsZero<E::Type, V>,
        > + 'static,
    V: Value<E::Type>
        + Differentiable<E::Type, Tangent = V>
        + Zero<E::Type>
        + One<E::Type>
        + for<'engine> Parameterized<
            V,
            Family: ParameterizedFamily<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
            To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>> = Tracer<
                'engine,
                DifferentiableOperationTracingEngine<E>,
            >,
            ParameterStructure: Debug + PartialEq,
        >,
    Input: Parameterized<
            V,
            Family: for<'engine> ParameterizedFamily<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
            ParameterStructure: Debug + PartialEq,
        >,
> ValueAndGradDispatch<E, Input, ConcreteValueAndGrad> for V
{
    type Value = V;

    type FunctionInput<'engine>
        = Input::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>
    where
        E: 'engine;
    type FunctionOutput<'engine>
        = Tracer<'engine, DifferentiableOperationTracingEngine<E>>
    where
        E: 'engine;

    fn invoke<'engine, F: FnOnce(Self::FunctionInput<'engine>) -> Self::FunctionOutput<'engine>>(
        engine: &'engine E,
        function: F,
        primals: Input,
    ) -> Result<(Self::Value, Input), TracingError> {
        let (output, pullback): (V, Program<E::Type, V, E::LinearOperationCarrier, V, Input>) =
            vjp(engine, |input| Ok(function(input)), primals)?;
        let gradient = pullback.interpret(<V as One<E::Type>>::one(output.r#type().as_ref())?)?;
        Ok((output, gradient))
    }
}

/// Already-traced dispatch for [`value_and_grad`]: replays the user function symbolically inside an
/// enclosing [`Tracer`] engine, linearizes, transposes, and stages the output and gradient.
impl<
    'engine,
    E: DifferentiableTracingEngine<
            Value = V,
            OperationCarrier: DifferentiableOperation<TracingContext<'engine, E>>
                                  + SupportsAdd<E::Type, V>
                                  + SupportsZeroLike<E::Type, V>,
            LinearOperationCarrier<'engine>: Clone
                                                 + InterpretableOperation<E::Type, Tracer<'engine, E>>
                                                 + LinearOperation<
                E::Type,
                Tracer<'engine, E>,
                <E as DifferentiableTracingEngine>::LinearOperationCarrier<'engine>,
            >,
        > + 'static,
    V: Traceable<E::Type>
        + Differentiable<E::Type, Tangent = V>
        + One<E::Type>
        + Parameterized<
            V,
            Family: ParameterizedFamily<E::Type> + ParameterizedFamily<Tracer<'engine, E>>,
            To<E::Type>: Parameterized<E::Type, To<Tracer<'engine, E>> = Tracer<'engine, E>>,
            ParameterStructure = Placeholder,
        >,
    Input: Parameterized<
            Tracer<'engine, E>,
            Family: ParameterizedFamily<V> + ParameterizedFamily<E::Type>,
            To<E::Type>: Parameterized<E::Type, To<Tracer<'engine, E>> = Input>,
            ParameterStructure: Debug + PartialEq,
        >,
> ValueAndGradDispatch<E, Input, TracedValueAndGrad> for Tracer<'engine, E>
where
    AddOperation: InterpretableOperation<E::Type, Tracer<'engine, E>>,
{
    type Value = Tracer<'engine, E>;

    type FunctionInput<'call>
        = Input
    where
        E: 'call;
    type FunctionOutput<'call>
        = Tracer<'engine, E>
    where
        E: 'call;

    fn invoke<'call, F: FnOnce(Self::FunctionInput<'call>) -> Self::FunctionOutput<'call>>(
        _engine: &'call E,
        function: F,
        primals: Input,
    ) -> Result<(Self::Value, Input), TracingError> {
        let input_structure = primals.parameter_structure();
        let traced_primals = primals.into_parameters().collect::<Vec<_>>();
        let Some(tracing_context) = traced_primals.first().map(|traced_primal| traced_primal.context.clone()) else {
            return Err(DifferentiationError::MissingTracedReverseModeInputLeaves.into());
        };
        let staged_input_types = Input::To::<E::Type>::from_parameters(
            input_structure.clone(),
            traced_primals.iter().map(|traced_primal| traced_primal.r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let (_, traced_program) =
            tracing_context.engine.trace(|staged_input| Ok(function(staged_input)), staged_input_types)?;
        let (traced_output, traced_gradient) = tracing_context.value_and_grad(&traced_program, traced_primals)?;
        Ok((traced_output, Input::from_parameters(input_structure, traced_gradient)?))
    }
}

/// Computes both the primal scalar output and its reverse-mode gradient.
///
/// This is the most direct reverse-mode API when the caller needs both the function value and the
/// gradient at the same primal point. The function must return exactly one rank-0 scalar array
/// leaf. Use [`vjp`] directly for vector-valued functions that need an explicit output cotangent.
#[allow(private_bounds, private_interfaces)]
pub fn value_and_grad<
    'engine,
    E: Engine,
    F: FnOnce(
        <Leaf as ValueAndGradDispatch<E, Input, Mode>>::FunctionInput<'engine>,
    ) -> <Leaf as ValueAndGradDispatch<E, Input, Mode>>::FunctionOutput<'engine>,
    Input: Parameterized<Leaf, ParameterStructure: Debug + PartialEq>,
    Leaf: ValueAndGradDispatch<E, Input, Mode>,
    Mode,
>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<(<Leaf as ValueAndGradDispatch<E, Input, Mode>>::Value, Input), TracingError> {
    Leaf::invoke(engine, function, primals)
}

/// Computes a scalar-output value, auxiliary outputs, and the reverse-mode gradient.
///
/// The differentiated value is the first element returned by `function`; it must be exactly one
/// rank-0 scalar array leaf. Auxiliary leaves are returned to the caller but seeded with zero
/// cotangents when the pullback is interpreted, so they do not contribute to the gradient.
///
/// This mirrors the semantics of a `has_aux` transform while keeping the Rust API explicit: the
/// primal value and auxiliary data are returned as `((value, aux), gradient)`.
#[allow(private_bounds)]
pub fn value_and_grad_with_aux<
    'engine,
    E: DifferentiableEngine<
            Value = V,
            LinearOperationCarrier: Clone
                                        + InterpretableOperation<E::Type, V>
                                        + LinearOperation<E::Type, V, E::LinearOperationCarrier>
                                        + SupportsZero<E::Type, V>,
        > + 'static,
    F: FnOnce(
        Input::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
    ) -> (
        Tracer<'engine, DifferentiableOperationTracingEngine<E>>,
        Aux::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
    ),
    Input: Parameterized<
            V,
            Family: ParameterizedFamily<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
            ParameterStructure: Debug + PartialEq,
        >,
    Aux: Parameterized<
            V,
            Family: ParameterizedFamily<
                Tracer<'engine, DifferentiableOperationTracingEngine<E>>,
                To = Aux::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
            >,
            ParameterStructure: Debug + PartialEq,
        >,
    V: Traceable<E::Type>
        + Differentiable<E::Type, Tangent = V>
        + Zero<E::Type>
        + One<E::Type>
        + Parameterized<
            V,
            Family: ParameterizedFamily<
                Tracer<'engine, DifferentiableOperationTracingEngine<E>>,
                To = Tracer<'engine, DifferentiableOperationTracingEngine<E>>,
            >,
            To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>> = Tracer<
                'engine,
                DifferentiableOperationTracingEngine<E>,
            >,
            ParameterStructure: Debug + PartialEq,
        >,
>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<((V, Aux), Input), TracingError> {
    let ((output, aux), pullback) = vjp(engine, |input| Ok(function(input)), primals)?;
    let aux_zeros = Aux::from_parameters(
        aux.parameter_structure(),
        aux.parameters().map(|value| V::zero(value.r#type().as_ref())).collect::<Result<Vec<_>, _>>()?,
    )?;
    let gradient = pullback.interpret((V::one(output.r#type().as_ref())?, aux_zeros))?;
    Ok(((output, aux), gradient))
}

/// Computes the reverse-mode gradient of a scalar-output function.
///
/// [`grad`] is just [`value_and_grad`] with the primal result discarded, but it is the most common
/// user-facing reverse-mode entry point and therefore gets its own dedicated wrapper. The function
/// must return exactly one rank-0 scalar array leaf. Use [`vjp`] directly for vector-valued functions
/// that need an explicit output cotangent.
#[allow(private_bounds, private_interfaces)]
pub fn grad<
    'engine,
    E: Engine,
    F: FnOnce(
        <Leaf as ValueAndGradDispatch<E, Input, Mode>>::FunctionInput<'engine>,
    ) -> <Leaf as ValueAndGradDispatch<E, Input, Mode>>::FunctionOutput<'engine>,
    Input: Parameterized<Leaf, ParameterStructure: Debug + PartialEq>,
    Leaf: ValueAndGradDispatch<E, Input, Mode>,
    Mode,
>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<Input, TracingError> {
    Leaf::invoke(engine, function, primals).map(|(_, gradient)| gradient)
}

/// Computes the reverse-mode gradient and auxiliary outputs of a scalar-output function.
///
/// This is [`value_and_grad_with_aux`] with the primal scalar value discarded. The return order is
/// `(gradient, aux)`, matching the common use case where auxiliary outputs are diagnostics or
/// cached intermediates and the gradient remains the primary result.
#[allow(private_bounds)]
pub fn grad_with_aux<
    'engine,
    E: DifferentiableEngine<
            Value = V,
            LinearOperationCarrier: Clone
                                        + InterpretableOperation<E::Type, V>
                                        + LinearOperation<E::Type, V, E::LinearOperationCarrier>
                                        + SupportsZero<E::Type, V>,
        > + 'static,
    F: FnOnce(
        Input::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
    ) -> (
        Tracer<'engine, DifferentiableOperationTracingEngine<E>>,
        Aux::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
    ),
    Input: Parameterized<
            V,
            Family: ParameterizedFamily<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
            ParameterStructure: Debug + PartialEq,
        >,
    Aux: Parameterized<
            V,
            Family: ParameterizedFamily<
                Tracer<'engine, DifferentiableOperationTracingEngine<E>>,
                To = Aux::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>,
            >,
            ParameterStructure: Debug + PartialEq,
        >,
    V: Traceable<E::Type>
        + Differentiable<E::Type, Tangent = V>
        + Zero<E::Type>
        + One<E::Type>
        + Parameterized<
            V,
            Family: ParameterizedFamily<
                Tracer<'engine, DifferentiableOperationTracingEngine<E>>,
                To = Tracer<'engine, DifferentiableOperationTracingEngine<E>>,
            >,
            To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>> = Tracer<
                'engine,
                DifferentiableOperationTracingEngine<E>,
            >,
            ParameterStructure: Debug + PartialEq,
        >,
>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<(Input, Aux), TracingError> {
    value_and_grad_with_aux(engine, function, primals).map(|((_, aux), gradient)| (gradient, aux))
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::fmt::{self, Display};
    use std::ops::{Add, Neg};

    use ryft_macros::Parameter;

    use crate::macros::check_input_count;
    use crate::operations::constants::{One, Zero};
    use crate::operations::constants::{OneLike, ZeroLike};
    use crate::operations::{InterpretableOperation, Operation};
    use crate::tracing::engines::{ScalarEngine, Tracer};
    use crate::tracing::{Traceable, TracingError, Value};
    use crate::tracing_v2::operations::add::{AddOperation, SupportsAdd};
    use crate::tracing_v2::operations::neg::SupportsNeg;
    use crate::tracing_v2::operations::scale::SupportsScale;
    use crate::tracing_v2::{Differentiable, DifferentiationError};
    use crate::types::{Type, TypeError, Typed};

    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq, Parameter)]
    struct TestType;

    impl Display for TestType {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("test")
        }
    }

    impl Type for TestType {
        fn is_compatible_with(&self, _other: &Self) -> bool {
            true
        }
    }

    #[derive(Clone, Debug, PartialEq, Parameter)]
    struct TestValue(f64);

    impl Display for TestValue {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            Display::fmt(&self.0, formatter)
        }
    }

    impl Add for TestValue {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    impl Neg for TestValue {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }

    impl Typed<TestType> for TestValue {
        fn r#type(&self) -> Cow<'_, TestType> {
            Cow::Owned(TestType)
        }
    }

    impl Traceable<TestType> for TestValue {}

    impl Value<TestType> for TestValue {}

    impl ZeroLike for TestValue {
        fn zero_like(&self) -> Self {
            Self(0.0)
        }
    }

    impl OneLike for TestValue {
        fn one_like(&self) -> Self {
            Self(1.0)
        }
    }

    impl Zero<TestType> for TestValue {
        fn zero(_type: &TestType) -> Result<Self, TracingError> {
            Ok(Self(0.0))
        }
    }

    impl One<TestType> for TestValue {
        fn one(_type: &TestType) -> Result<Self, TracingError> {
            Ok(Self(1.0))
        }
    }

    impl Differentiable<TestType> for TestValue {
        type Tangent = Self;
    }

    #[derive(Clone, Debug)]
    enum TestLinearOperation {
        Add,
        Neg,
        Scale { factor: TestValue },
    }

    impl Display for TestLinearOperation {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation<TestType> for AddOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "add"
        }

        fn infer_output_types(&self, input_types: &[TestType]) -> Result<Vec<TestType>, TypeError> {
            check_input_count!(input_types, 2, TypeError);
            Ok(vec![TestType])
        }
    }

    impl InterpretableOperation<TestType, TestValue> for AddOperation {
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, TracingError> {
            check_input_count!(inputs, 2, TracingError);
            Ok(vec![inputs[0].clone() + inputs[1].clone()])
        }
    }

    impl SupportsAdd<TestType, TestValue> for AddOperation {
        fn add_operation() -> Self {
            AddOperation
        }
    }

    impl Operation<TestType> for TestLinearOperation {
        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Add => "add",
                Self::Neg => "neg",
                Self::Scale { .. } => "scale",
            }
        }

        fn infer_output_types(&self, input_types: &[TestType]) -> Result<Vec<TestType>, TypeError> {
            let expected = match self {
                Self::Add => 2,
                Self::Neg | Self::Scale { .. } => 1,
            };
            check_input_count!(input_types, expected, TypeError);
            Ok(vec![TestType])
        }
    }

    impl InterpretableOperation<TestType, TestValue> for TestLinearOperation {
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, TracingError> {
            let expected = match self {
                Self::Add => 2,
                Self::Neg | Self::Scale { .. } => 1,
            };
            check_input_count!(inputs, expected, TracingError);
            Ok(vec![match self {
                Self::Add => inputs[0].clone() + inputs[1].clone(),
                Self::Neg => -inputs[0].clone(),
                Self::Scale { factor } => TestValue(factor.0 * inputs[0].0),
            }])
        }
    }

    impl SupportsAdd<TestType, TestValue> for TestLinearOperation {
        fn add_operation() -> Self {
            Self::Add
        }
    }

    impl SupportsNeg<TestType, TestValue> for TestLinearOperation {
        fn neg_operation() -> Self {
            Self::Neg
        }
    }

    impl SupportsScale<TestType, TestValue> for TestLinearOperation {
        fn scale_operation(factor: TestValue) -> Self {
            Self::Scale { factor }
        }
    }

    impl LinearOperation<TestType, TestValue, TestLinearOperation> for TestLinearOperation {
        fn transpose(
            &self,
            context: &mut crate::tracing::transposition::TranspositionContext<TestType, TestValue, TestLinearOperation>,
            output_cotangents: &[Option<crate::tracing::AtomId>],
        ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
            check_input_count!(output_cotangents, 1, TracingError);
            Ok(match self {
                Self::Add => vec![output_cotangents[0], output_cotangents[0]],
                Self::Neg => match output_cotangents[0] {
                    Some(cotangent) => {
                        vec![Some(context.stage(Self::Neg, &[cotangent])?[0])]
                    }
                    None => vec![None],
                },
                Self::Scale { factor } => match output_cotangents[0] {
                    Some(cotangent) => {
                        vec![Some(context.stage(Self::Scale { factor: factor.clone() }, &[cotangent])?[0])]
                    }
                    None => vec![None],
                },
            })
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct TestEngine;

    impl Engine for TestEngine {
        type Type = TestType;
        type Value = TestValue;

        fn zero(&self, _type: &TestType) -> Result<TestValue, TracingError> {
            Ok(TestValue(0.0))
        }

        fn one(&self, _type: &TestType) -> Result<TestValue, TracingError> {
            Ok(TestValue(1.0))
        }
    }

    impl TracingEngine for TestEngine {
        type OperationCarrier = AddOperation;
    }

    impl crate::tracing_v2::LinearizableEngine for TestEngine {
        type LinearOperationCarrier = TestLinearOperation;
    }

    impl DifferentiableEngine for TestEngine {
        type DifferentiableOperationCarrier = AddOperation;
    }

    #[test]
    fn test_linearize_supports_non_array_type_metadata() {
        let engine = TestEngine;
        let (output, pushforward) = linearize(
            &engine,
            |x: Tracer<'_, DifferentiableOperationTracingEngine<TestEngine>>| Ok(x.clone() + x),
            TestValue(3.0),
        )
        .unwrap();

        assert_eq!(output, TestValue(6.0));
        assert_eq!(pushforward.interpret(TestValue(5.0)), Ok(TestValue(10.0)));
    }

    #[test]
    fn test_traced_value_and_grad_requires_input_leaves() {
        let engine = ScalarEngine::<f64>::new();
        let empty_primals: Vec<Tracer<'_, ScalarEngine<f64>>> = Vec::new();

        let result = <Tracer<'_, ScalarEngine<f64>> as ValueAndGradDispatch<
            ScalarEngine<f64>,
            Vec<Tracer<'_, ScalarEngine<f64>>>,
            TracedValueAndGrad,
        >>::invoke(
            &engine, |_inputs| panic!("closure should not run without traced inputs"), empty_primals
        );

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::MissingTracedReverseModeInputLeaves))
        ));
    }

    #[test]
    fn test_value_and_grad_with_aux_ignores_aux_cotangents() {
        let engine = ScalarEngine::<f64>::new();

        let ((value, aux), gradient): ((f64, (f64, f64)), (f64, f64)) = value_and_grad_with_aux(
            &engine,
            |(x, y)| {
                let value = x.clone() * y.clone();
                let aux = (x.clone() + y, x.clone() * x);
                (value, aux)
            },
            (2.0f64, 3.0f64),
        )
        .unwrap();

        assert_eq!(value, 6.0);
        assert_eq!(aux, (5.0, 4.0));
        assert_eq!(gradient, (3.0, 2.0));
    }

    #[test]
    fn test_grad_with_aux_returns_gradient_and_aux() {
        let engine = ScalarEngine::<f64>::new();

        let (gradient, aux): ((f64, f64), f64) =
            grad_with_aux(&engine, |(x, y)| (x.clone() * y.clone(), x + y), (2.0f64, 3.0f64)).unwrap();

        assert_eq!(gradient, (3.0, 2.0));
        assert_eq!(aux, 5.0);
    }
}
