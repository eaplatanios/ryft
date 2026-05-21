use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

use crate::parameters::Parameter;
use crate::tracing_v2::{
    DifferentiableDomain, DifferentiableOperation, DifferentiableTracingDomain, LinearizationTracer,
};
use crate::{
    AddOperation, InterpretableOperation, One, Parameterized, ParameterizedFamily, Placeholder, Program,
    ProgramBuilder, RuntimeDomain, SupportsAdd, SupportsZeroLike, Traceable, Tracer, TracingContext, TracingError,
    Typed, Value,
};

impl<'domain, D> TracingContext<'domain, D>
where
    D: DifferentiableTracingDomain + RuntimeDomain + 'domain,
    D::Value: One<D::Type> + 'domain,
    D::OperationCarrier: DifferentiableOperation<TracingContext<'domain, D>>
        + SupportsZeroLike<D::Type, D::Value>
        + SupportsAdd<D::Type, D::Value>
        + 'domain,
    AddOperation: InterpretableOperation<D::Type, Tracer<'domain, D>>,
{
    /// Linearizes one traced scalar-output program and stages its pullback with a unit cotangent seed.
    ///
    /// This is the internal core of traced reverse-mode for scalar-output functions. Given a staged
    /// primal body and symbolic primals from this enclosing trace, it builds the pushforward,
    /// transposes it into a pullback, seeds that pullback with a symbolic one, and returns both the
    /// traced scalar output and the traced gradient leaves.
    pub(super) fn value_and_grad<Input, Output>(
        self,
        traced_program: &Program<D::Type, D::Value, D::OperationCarrier, Input, Output>,
        traced_primals: Vec<Tracer<'domain, D>>,
    ) -> Result<(Tracer<'domain, D>, Vec<Tracer<'domain, D>>), TracingError>
    where
        Input: Parameterized<D::Value>,
        Output: Parameterized<D::Value>,
    {
        let (outputs, pushforward) = self.linearize_program(traced_program, traced_primals)?;
        if outputs.len() != 1 {
            return Err(TracingError::InvalidOutputCount { expected: 1, got: outputs.len() });
        }
        let traced_output = outputs[0].clone();
        let pullback = self.transpose(&pushforward)?;
        let seed_type = traced_output.r#type().into_owned();
        let _ = <D::Value as One<D::Type>>::one(&seed_type)?;
        let seed = self.one(&seed_type)?;
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

/// Dispatch trait shared by [`DifferentiableDomain::value_and_gradient`] and [`value_and_grad`] so they can operate
/// both on concrete values and on already traced values.
///
/// The trait always produces `(value, gradient)`; [`DifferentiableDomain::value_and_gradient`] is a thin wrapper that
/// drops the primal value, while [`value_and_grad`] exposes the full pair. This keeps the public reverse-mode API
/// compact while allowing concrete replay, traced replay, and batched replay to specialize independently.
#[doc(hidden)]
pub trait ValueAndGradientDispatch<D: RuntimeDomain, Input, Marker>: Parameter + Sized
where
    Input: Parameterized<Self, ParameterStructure: Debug + PartialEq>,
{
    /// Primal scalar output value produced for the corresponding input regime.
    type Value;

    /// Gradient value produced for the corresponding input regime.
    type Gradient;

    /// Traced input type expected by the user-provided function.
    type FunctionInput<'domain>
    where
        D: 'domain;

    /// Traced scalar output type expected from the user-provided function.
    type FunctionOutput<'domain>
    where
        D: 'domain;

    /// Invokes [`value_and_grad`] for one concrete leaf regime.
    fn invoke<'domain, F: FnOnce(Self::FunctionInput<'domain>) -> Self::FunctionOutput<'domain>>(
        domain: &'domain D,
        function: F,
        primals: Input,
    ) -> Result<(Self::Value, Self::Gradient), TracingError>;
}

/// Concrete-value dispatch for [`value_and_grad`]: evaluates the user function via [`vjp`], checks
/// that the output is a scalar, and pulls back a unit seed to obtain both the primal output and gradient.
impl<
    D: DifferentiableDomain<Value = V> + 'static,
    V: Value<D::Type>
        + 'static
        + for<'domain> Parameterized<
            V,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>>,
            To<LinearizationTracer<'domain, D>> = LinearizationTracer<'domain, D>,
            To<V> = V,
            ParameterStructure: Debug + PartialEq,
        >,
    Input: Parameterized<
            V,
            Family: for<'domain> ParameterizedFamily<LinearizationTracer<'domain, D>>,
            To<V> = Input,
            ParameterStructure: Debug + PartialEq,
        >,
> ValueAndGradientDispatch<D, Input, ConcreteValueAndGrad> for V
where
    D::OperationCarrier: DifferentiableOperation<D>,
    D::Tangent: One<D::Type>,
    V::Family: ParameterizedFamily<D::Tangent>,
    Input::Family: ParameterizedFamily<D::Tangent>,
{
    type Value = V;
    type Gradient = Input::To<D::Tangent>;

    type FunctionInput<'domain>
        = Input::To<LinearizationTracer<'domain, D>>
    where
        D: 'domain;
    type FunctionOutput<'domain>
        = LinearizationTracer<'domain, D>
    where
        D: 'domain;

    fn invoke<'domain, F: FnOnce(Self::FunctionInput<'domain>) -> Self::FunctionOutput<'domain>>(
        domain: &'domain D,
        function: F,
        primals: Input,
    ) -> Result<(Self::Value, Self::Gradient), TracingError> {
        let (output, pullback): (
            V,
            Program<D::Type, D::Tangent, D::LinearOperationCarrier, V::To<D::Tangent>, Self::Gradient>,
        ) = domain.vjp(|input| Ok(function(input)), primals)?;
        let seed = V::To::<D::Tangent>::from_parameters(
            output.parameter_structure(),
            [<D::Tangent as One<D::Type>>::one(output.r#type().as_ref())?],
        )?;
        let gradient = pullback.interpret(seed)?;
        Ok((output, gradient))
    }
}

/// Already-traced dispatch for [`value_and_grad`]: replays the user function symbolically inside an
/// enclosing [`Tracer`] domain, linearizes, transposes, and stages the output and gradient.
impl<'domain, D: DifferentiableTracingDomain + RuntimeDomain + 'static, Input>
    ValueAndGradientDispatch<D, Input, TracedValueAndGrad> for Tracer<'domain, D>
where
    D::Value: One<D::Type> + Parameterized<D::Value, ParameterStructure = Placeholder>,
    D::OperationCarrier: DifferentiableOperation<TracingContext<'domain, D>>
        + SupportsZeroLike<D::Type, D::Value>
        + SupportsAdd<D::Type, D::Value>
        + 'domain,
    <D::Value as Parameterized<D::Value>>::Family:
        ParameterizedFamily<D::Type> + ParameterizedFamily<Tracer<'domain, D>>,
    <D::Value as Parameterized<D::Value>>::To<D::Type>:
        Parameterized<D::Type, To<Tracer<'domain, D>> = Tracer<'domain, D>>,
    Input: Parameterized<Tracer<'domain, D>>,
    Input::Family: ParameterizedFamily<D::Value> + ParameterizedFamily<D::Type>,
    Input::To<D::Value>: Parameterized<D::Value, ParameterStructure = Input::ParameterStructure>,
    Input::To<D::Type>: Parameterized<D::Type, To<Tracer<'domain, D>> = Input>,
    Input::ParameterStructure: Debug + PartialEq,
    AddOperation: InterpretableOperation<D::Type, Tracer<'domain, D>>,
{
    type Value = Tracer<'domain, D>;
    type Gradient = Input;

    type FunctionInput<'call>
        = Input
    where
        D: 'call;
    type FunctionOutput<'call>
        = Tracer<'domain, D>
    where
        D: 'call;

    fn invoke<'call, F: FnOnce(Self::FunctionInput<'call>) -> Self::FunctionOutput<'call>>(
        _domain: &'call D,
        function: F,
        primals: Input,
    ) -> Result<(Self::Value, Self::Gradient), TracingError> {
        let input_structure = primals.parameter_structure();
        let traced_primals = primals.into_parameters().collect::<Vec<_>>();
        let Some(tracing_context) = traced_primals.first().map(|traced_primal| traced_primal.context().clone()) else {
            return Err(TracingError::InvalidInputCount { expected: 1, got: 0 });
        };
        if traced_primals
            .iter()
            .any(|tracer| !Rc::ptr_eq(tracing_context.builder(), tracer.context().builder()))
        {
            return Err(tracing_context.error(TracingError::MismatchedProgramBuilders));
        }
        let staged_input_types = Input::To::<D::Type>::from_parameters(
            input_structure.clone(),
            traced_primals.iter().map(|traced_primal| traced_primal.r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let builder = Rc::new(RefCell::new(ProgramBuilder::<D::Type, D::Value, D::OperationCarrier>::new()));
        let staged_input = staged_input_types
            .map_parameters(|r#type| TracingContext::new(tracing_context.domain(), builder.clone()).input(r#type))?;
        let traced_output = function(staged_input);
        if let Some(error) = builder.borrow().error().cloned() {
            return Err(error);
        }
        let output_structure = traced_output.parameter_structure();
        let output_atoms = traced_output.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
        drop(traced_output);
        let builder = Rc::try_unwrap(builder).map_err(|_| TracingError::EscapedProgramBuilder)?.into_inner();
        let traced_program: Program<D::Type, D::Value, D::OperationCarrier, Input::To<D::Value>, D::Value> =
            builder.build(output_atoms, input_structure.clone(), output_structure)?;
        let (traced_output, traced_gradient) = tracing_context.value_and_grad(&traced_program, traced_primals)?;
        Ok((traced_output, Input::from_parameters(input_structure, traced_gradient)?))
    }
}

/// Computes both the primal scalar output and its reverse-mode gradient.
///
/// This is the most direct reverse-mode API when the caller needs both the function value and the
/// gradient at the same primal point. The function must return exactly one rank-0 scalar array
/// leaf. Use [`DifferentiableDomain::vjp`] directly for vector-valued functions that need an explicit output
/// cotangent.
#[allow(private_bounds, private_interfaces)]
pub fn value_and_grad<
    'domain,
    D: RuntimeDomain,
    F: FnOnce(
        <Leaf as ValueAndGradientDispatch<D, Input, Marker>>::FunctionInput<'domain>,
    ) -> <Leaf as ValueAndGradientDispatch<D, Input, Marker>>::FunctionOutput<'domain>,
    Input: Parameterized<Leaf, ParameterStructure: Debug + PartialEq>,
    Leaf: ValueAndGradientDispatch<D, Input, Marker>,
    Marker,
>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<
    (
        <Leaf as ValueAndGradientDispatch<D, Input, Marker>>::Value,
        <Leaf as ValueAndGradientDispatch<D, Input, Marker>>::Gradient,
    ),
    TracingError,
> {
    Leaf::invoke(domain, function, primals)
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
    'domain,
    D: DifferentiableDomain<Value = V> + 'static,
    F: FnOnce(
        Input::To<LinearizationTracer<'domain, D>>,
    ) -> (LinearizationTracer<'domain, D>, Aux::To<LinearizationTracer<'domain, D>>),
    Input: Parameterized<
            V,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>>,
            To<V> = Input,
            ParameterStructure: Debug + PartialEq,
        >,
    Aux: Parameterized<
            V,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>, To = Aux::To<LinearizationTracer<'domain, D>>>,
            ParameterStructure: Debug + PartialEq,
        >,
    V: Traceable<D::Type>
        + Parameterized<
            V,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>, To = LinearizationTracer<'domain, D>>,
            To<LinearizationTracer<'domain, D>> = LinearizationTracer<'domain, D>,
            ParameterStructure: Debug + PartialEq,
        > + 'domain,
>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<((V, Aux), Input::To<D::Tangent>), TracingError>
where
    D::OperationCarrier: DifferentiableOperation<D>,
    D::Tangent: One<D::Type>,
    V::Family: ParameterizedFamily<D::Tangent>,
    Input::Family: ParameterizedFamily<D::Tangent>,
    Aux::Family: ParameterizedFamily<D::Tangent>,
{
    let ((output, aux), pullback) = domain.vjp(|input| Ok(function(input)), primals)?;
    let output_cotangent_structure = (output.parameter_structure(), aux.parameter_structure());
    let seed = <D::Tangent as One<D::Type>>::one(output.r#type().as_ref())?;
    let aux_zeros = aux
        .parameters()
        .map(|value| domain.zero_tangent(value.r#type().as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let output_cotangent = <(V, Aux) as Parameterized<V>>::To::<D::Tangent>::from_parameters(
        output_cotangent_structure,
        std::iter::once(seed).chain(aux_zeros.into_iter()),
    )?;
    let gradient = pullback.interpret(output_cotangent)?;
    Ok(((output, aux), gradient))
}

/// Computes the reverse-mode gradient and auxiliary outputs of a scalar-output function.
///
/// This is [`value_and_grad_with_aux`] with the primal scalar value discarded. The return order is
/// `(gradient, aux)`, matching the common use case where auxiliary outputs are diagnostics or
/// cached intermediates and the gradient remains the primary result.
#[allow(private_bounds)]
pub fn grad_with_aux<
    'domain,
    D: DifferentiableDomain<Value = V> + 'static,
    F: FnOnce(
        Input::To<LinearizationTracer<'domain, D>>,
    ) -> (LinearizationTracer<'domain, D>, Aux::To<LinearizationTracer<'domain, D>>),
    Input: Parameterized<
            V,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>>,
            To<V> = Input,
            ParameterStructure: Debug + PartialEq,
        >,
    Aux: Parameterized<
            V,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>, To = Aux::To<LinearizationTracer<'domain, D>>>,
            ParameterStructure: Debug + PartialEq,
        >,
    V: Traceable<D::Type>
        + Parameterized<
            V,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>, To = LinearizationTracer<'domain, D>>,
            To<LinearizationTracer<'domain, D>> = LinearizationTracer<'domain, D>,
            ParameterStructure: Debug + PartialEq,
        > + 'domain,
>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<(Input::To<D::Tangent>, Aux), TracingError>
where
    D::OperationCarrier: DifferentiableOperation<D>,
    D::Tangent: One<D::Type>,
    V::Family: ParameterizedFamily<D::Tangent>,
    Input::Family: ParameterizedFamily<D::Tangent>,
    Aux::Family: ParameterizedFamily<D::Tangent>,
{
    value_and_grad_with_aux(domain, function, primals).map(|((_, aux), gradient)| (gradient, aux))
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::fmt::{self, Display};
    use std::ops::{Add, Neg};
    use std::rc::Rc;

    use ryft_macros::Parameter;

    use crate::differentiation::{Cotangent, LinearOperation};
    use crate::macros::check_count;
    use crate::operations::arithmetic::{
        ADD_OPERATION_NAME, AddOperation, Scale, SupportsAdd, SupportsNeg, SupportsScale,
    };
    use crate::operations::constants::{One, OneLike, SupportsZero, Zero, ZeroLike};
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::{InterpretableOperation, Operation};
    use crate::tracing::domains::{Domain, ScalarDomain, Tracer, TracingContext, TracingDomain};
    use crate::tracing::{ProgramBuilder, ProgramTracingContext, Traceable, TracingError, Value};
    use crate::tracing_v2::LinearizableDomain;
    use crate::types::{DataType, Type, TypeError, Typed};

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

    #[derive(Clone, Debug)]
    enum TestLinearOperation {
        Zero(TestType),
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
            ADD_OPERATION_NAME
        }

        fn infer_output_types(&self, input_types: &[TestType]) -> Result<Vec<TestType>, TypeError> {
            check_count!("input", input_types, 2, TypeError);
            Ok(vec![TestType])
        }
    }

    impl InterpretableOperation<TestType, TestValue> for AddOperation {
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, TracingError> {
            check_count!("input", inputs, 2, TracingError);
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
                Self::Zero(_) => "zero",
                Self::Add => ADD_OPERATION_NAME,
                Self::Neg => "neg",
                Self::Scale { .. } => "scale",
            }
        }

        fn infer_output_types(&self, input_types: &[TestType]) -> Result<Vec<TestType>, TypeError> {
            let expected = match self {
                Self::Zero(_) => 0,
                Self::Add => 2,
                Self::Neg | Self::Scale { .. } => 1,
            };
            check_count!("input", input_types, expected, TypeError);
            Ok(vec![TestType])
        }
    }

    impl InterpretableOperation<TestType, TestValue> for TestLinearOperation {
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, TracingError> {
            let expected = match self {
                Self::Zero(_) => 0,
                Self::Add => 2,
                Self::Neg | Self::Scale { .. } => 1,
            };
            check_count!("input", inputs, expected, TracingError);
            Ok(vec![match self {
                Self::Zero(_) => TestValue(0.0),
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

    impl SupportsZero<TestType, TestValue> for TestLinearOperation {
        fn zero_operation(r#type: TestType) -> Self {
            Self::Zero(r#type)
        }
    }

    impl SupportsNeg<TestType, TestValue> for TestLinearOperation {
        fn neg_operation() -> Self {
            Self::Neg
        }
    }

    impl SupportsScale<TestType, TestValue, TestValue> for TestLinearOperation {
        fn scale_operation(factor: TestValue) -> Self {
            Self::Scale { factor }
        }
    }

    impl LinearOperation<TestType, TestValue, TestLinearOperation> for TestLinearOperation {
        fn transpose<'transpose>(
            &self,
            _context: &mut ProgramTracingContext<'transpose, TestType, TestValue, TestLinearOperation>,
            output_cotangents: &[Cotangent<'transpose, TestType, TestValue, TestLinearOperation>],
        ) -> Result<Vec<Cotangent<'transpose, TestType, TestValue, TestLinearOperation>>, TracingError> {
            check_count!("output", output_cotangents, 1, TracingError);
            Ok(match self {
                Self::Zero(_) => Vec::new(),
                Self::Add => vec![output_cotangents[0].clone(), output_cotangents[0].clone()],
                Self::Neg => match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => vec![Cotangent::Staged(-cotangent.clone())],
                    Cotangent::Zero => vec![Cotangent::Zero],
                },
                Self::Scale { factor } => match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => vec![Cotangent::Staged(cotangent.clone().scale(factor.clone()))],
                    Cotangent::Zero => vec![Cotangent::Zero],
                },
            })
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct TestDomain;

    impl Domain for TestDomain {
        type Type = TestType;
        type Value = TestValue;
    }

    impl RuntimeDomain for TestDomain {
        fn zero(&self, _type: &TestType) -> Result<TestValue, TracingError> {
            Ok(TestValue(0.0))
        }

        fn one(&self, _type: &TestType) -> Result<TestValue, TracingError> {
            Ok(TestValue(1.0))
        }
    }

    impl TracingDomain for TestDomain {
        type OperationCarrier = AddOperation;
    }

    #[derive(Copy, Clone, Debug)]
    struct TestLinearDomain;

    impl Domain for TestLinearDomain {
        type Type = TestType;
        type Value = TestValue;
    }

    impl RuntimeDomain for TestLinearDomain {
        fn zero(&self, _type: &TestType) -> Result<TestValue, TracingError> {
            Ok(TestValue(0.0))
        }

        fn one(&self, _type: &TestType) -> Result<TestValue, TracingError> {
            Ok(TestValue(1.0))
        }
    }

    impl TracingDomain for TestLinearDomain {
        type OperationCarrier = TestLinearOperation;
    }

    static TEST_LINEAR_DOMAIN: TestLinearDomain = TestLinearDomain;

    impl LinearizableDomain for TestDomain {
        type LinearDomain = TestLinearDomain;

        fn linear_domain(&self) -> &Self::LinearDomain {
            &TEST_LINEAR_DOMAIN
        }
    }

    #[test]
    fn test_linearize_supports_non_array_type_metadata() {
        let domain = TestDomain;
        let (output, pushforward): (
            TestValue,
            Program<TestType, TestValue, TestLinearOperation, TestValue, TestValue>,
        ) = domain.linearize(|x| Ok(x.clone() + x), TestValue(3.0)).unwrap();

        assert_eq!(output, TestValue(6.0));
        assert_eq!(pushforward.interpret(TestValue(5.0)), Ok(TestValue(10.0)));
    }

    #[test]
    fn test_traced_value_and_grad_requires_input_leaves() {
        let domain = ScalarDomain::<f64>::new();
        let empty_primals: Vec<Tracer<'_, ScalarDomain<f64>>> = Vec::new();

        let result = <Tracer<'_, ScalarDomain<f64>> as ValueAndGradientDispatch<
            ScalarDomain<f64>,
            Vec<Tracer<'_, ScalarDomain<f64>>>,
            TracedValueAndGrad,
        >>::invoke(
            &domain, |_inputs| panic!("closure should not run without traced inputs"), empty_primals
        );

        assert!(matches!(result, Err(TracingError::InvalidInputCount { expected: 1, got: 0 })));
    }

    #[test]
    fn test_traced_value_and_grad_rejects_mismatched_program_builders() {
        let domain = ScalarDomain::<f64>::new();
        let builder_a = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let builder_b = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let context_a = TracingContext::new(&domain, builder_a);
        let context_b = TracingContext::new(&domain, builder_b);
        let primal_a = context_a.input(DataType::F64);
        let primal_b = context_b.input(DataType::F64);

        let result = value_and_grad(
            &domain,
            |inputs: Vec<Tracer<'_, ScalarDomain<f64>>>| inputs[0].clone() + inputs[1].clone(),
            vec![primal_a, primal_b],
        );

        assert!(matches!(result, Err(TracingError::MismatchedProgramBuilders)));
    }

    #[test]
    fn test_value_and_grad_with_aux_ignores_aux_cotangents() {
        let domain = ScalarDomain::<f64>::new();

        let ((value, aux), gradient): ((f64, (f64, f64)), (f64, f64)) = value_and_grad_with_aux(
            &domain,
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
        let domain = ScalarDomain::<f64>::new();

        let (gradient, aux): ((f64, f64), f64) =
            grad_with_aux(&domain, |(x, y)| (x.clone() * y.clone(), x + y), (2.0f64, 3.0f64)).unwrap();

        assert_eq!(gradient, (3.0, 2.0));
        assert_eq!(aux, 5.0);
    }
}
