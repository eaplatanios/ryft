use std::fmt::Debug;

use crate::differentiation::TransposableOperation;
use crate::operations::InterpretableOperation;
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::constants::SupportsZero;
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation, LinearOperationOf, LinearizationTracer};
use crate::{Domain, One, Parameterized, ParameterizedFamily, Program, TracingError, Typed};

/// Computes both the primal scalar output and its reverse-mode gradient.
///
/// This is the most direct reverse-mode API when the caller needs both the function value and the
/// gradient at the same primal point. The function must return exactly one rank-0 scalar array
/// leaf. Use [`DifferentiableDomain::vjp`] directly for vector-valued functions that need an explicit output
/// cotangent.
#[allow(private_bounds)]
pub fn value_and_grad<'domain, D, F, Input>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<(<D as Domain>::Value, Input::To<D::Tangent>), TracingError>
where
    D: DifferentiableDomain<Operation: DifferentiableOperation<D>>,
    F: FnOnce(Input::To<LinearizationTracer<'domain, D>>) -> LinearizationTracer<'domain, D>,
    Input: Parameterized<
            <D as Domain>::Value,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>> + ParameterizedFamily<D::Tangent>,
            To<<D as Domain>::Value> = Input,
            ParameterStructure: Debug + PartialEq,
        >,
    <D as Domain>::Value: Parameterized<
            <D as Domain>::Value,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>> + ParameterizedFamily<D::Tangent>,
            To<LinearizationTracer<'domain, D>> = LinearizationTracer<'domain, D>,
            To<<D as Domain>::Value> = <D as Domain>::Value,
            ParameterStructure: Debug + PartialEq,
        > + 'domain,
    D::Tangent: One<<D as Domain>::Type>,
    LinearOperationOf<D>: InterpretableOperation<<D as Domain>::Type, D::Tangent>
        + TransposableOperation<<D as Domain>::Type, D::Tangent, LinearOperationOf<D>>
        + SupportsZero<<D as Domain>::Type, D::Tangent>
        + SupportsAdd<<D as Domain>::Type, D::Tangent>,
{
    let (output, pullback): (
        <D as Domain>::Value,
        Program<
            <D as Domain>::Type,
            D::Tangent,
            LinearOperationOf<D>,
            <<D as Domain>::Value as Parameterized<<D as Domain>::Value>>::To<D::Tangent>,
            Input::To<D::Tangent>,
        >,
    ) = domain.vjp(|input| Ok(function(input)), primals)?;
    let seed = <<D as Domain>::Value as Parameterized<<D as Domain>::Value>>::To::<D::Tangent>::from_parameters(
        output.parameter_structure(),
        [<D::Tangent as One<<D as Domain>::Type>>::one(output.r#type().as_ref())?],
    )?;
    let gradient = pullback.interpret(seed)?;
    Ok((output, gradient))
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
pub fn value_and_grad_with_aux<'domain, D, F, Input, Aux>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<((<D as Domain>::Value, Aux), Input::To<D::Tangent>), TracingError>
where
    D: DifferentiableDomain,
    F: FnOnce(
        Input::To<LinearizationTracer<'domain, D>>,
    ) -> (LinearizationTracer<'domain, D>, Aux::To<LinearizationTracer<'domain, D>>),
    Input: Parameterized<
            <D as Domain>::Value,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>>,
            To<<D as Domain>::Value> = Input,
            ParameterStructure: Debug + PartialEq,
        >,
    Aux: Parameterized<
            <D as Domain>::Value,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>, To = Aux::To<LinearizationTracer<'domain, D>>>,
            ParameterStructure: Debug + PartialEq,
        >,
    D::Operation: DifferentiableOperation<D>,
    <D as Domain>::Value: Parameterized<
            <D as Domain>::Value,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>, To = LinearizationTracer<'domain, D>>,
            To<LinearizationTracer<'domain, D>> = LinearizationTracer<'domain, D>,
            ParameterStructure: Debug + PartialEq,
        > + 'domain,
    D::Tangent: One<<D as Domain>::Type>,
    LinearOperationOf<D>: InterpretableOperation<<D as Domain>::Type, D::Tangent>
        + TransposableOperation<<D as Domain>::Type, D::Tangent, LinearOperationOf<D>>
        + SupportsZero<<D as Domain>::Type, D::Tangent>
        + SupportsAdd<<D as Domain>::Type, D::Tangent>,
    <<D as Domain>::Value as Parameterized<<D as Domain>::Value>>::Family: ParameterizedFamily<D::Tangent>,
    Input::Family: ParameterizedFamily<D::Tangent>,
    Aux::Family: ParameterizedFamily<D::Tangent>,
{
    let ((output, aux), pullback) = domain.vjp(|input| Ok(function(input)), primals)?;
    let output_cotangent_structure = (output.parameter_structure(), aux.parameter_structure());
    let seed = <D::Tangent as One<<D as Domain>::Type>>::one(output.r#type().as_ref())?;
    let aux_zeros = aux
        .parameters()
        .map(|value| domain.zero_tangent(value.r#type().as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let output_cotangent =
        <(<D as Domain>::Value, Aux) as Parameterized<<D as Domain>::Value>>::To::<D::Tangent>::from_parameters(
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
pub fn grad_with_aux<'domain, D, F, Input, Aux>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<(Input::To<D::Tangent>, Aux), TracingError>
where
    D: DifferentiableDomain,
    F: FnOnce(
        Input::To<LinearizationTracer<'domain, D>>,
    ) -> (LinearizationTracer<'domain, D>, Aux::To<LinearizationTracer<'domain, D>>),
    Input: Parameterized<
            <D as Domain>::Value,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>>,
            To<<D as Domain>::Value> = Input,
            ParameterStructure: Debug + PartialEq,
        >,
    Aux: Parameterized<
            <D as Domain>::Value,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>, To = Aux::To<LinearizationTracer<'domain, D>>>,
            ParameterStructure: Debug + PartialEq,
        >,
    D::Operation: DifferentiableOperation<D>,
    <D as Domain>::Value: Parameterized<
            <D as Domain>::Value,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>, To = LinearizationTracer<'domain, D>>,
            To<LinearizationTracer<'domain, D>> = LinearizationTracer<'domain, D>,
            ParameterStructure: Debug + PartialEq,
        > + 'domain,
    D::Tangent: One<<D as Domain>::Type>,
    LinearOperationOf<D>: InterpretableOperation<<D as Domain>::Type, D::Tangent>
        + TransposableOperation<<D as Domain>::Type, D::Tangent, LinearOperationOf<D>>
        + SupportsZero<<D as Domain>::Type, D::Tangent>
        + SupportsAdd<<D as Domain>::Type, D::Tangent>,
    <<D as Domain>::Value as Parameterized<<D as Domain>::Value>>::Family: ParameterizedFamily<D::Tangent>,
    Input::Family: ParameterizedFamily<D::Tangent>,
    Aux::Family: ParameterizedFamily<D::Tangent>,
{
    value_and_grad_with_aux(domain, function, primals).map(|((_, aux), gradient)| (gradient, aux))
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::{Cell, RefCell};
    use std::fmt::{self, Display};
    use std::ops::{Add, Neg};
    use std::rc::Rc;

    use ryft_macros::Parameter;

    use crate::Context;
    use crate::differentiation::{Cotangent, TransposableOperation};
    use crate::macros::check_count;
    use crate::operations::arithmetic::{
        ADD_OPERATION_NAME, AddOperation, Scale, SupportsAdd, SupportsNeg, SupportsScale,
    };
    use crate::operations::constants::{One, OneLike, SupportsZero, Zero, ZeroLike};
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Parameter;
    use crate::tracing::contexts::TracingContext;
    use crate::tracing::domains::{Domain, DomainTracer, RuntimeDomain, ScalarDomain, TracingDomain};
    use crate::tracing::{ProgramBuilder, ProgramTracingContext, Traceable, TracingError, Value};
    use crate::tracing_v2::{Differentiable, DifferentiableContext};
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

    impl TransposableOperation<TestType, TestValue, TestLinearOperation> for TestLinearOperation {
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
        type Constant = TestValue;
        type Operation = AddOperation;

        fn lift_constant(&self, constant: TestValue) -> Result<TestValue, TracingError> {
            Ok(constant)
        }
    }

    impl Differentiable for TestDomain {
        type Type = TestType;
        type Value = TestValue;
        type Tangent = TestValue;
        type Constant = TestValue;
        type LinearOperation<V: Traceable<TestType>> = TestLinearOperation;

        fn zero_primal(&self, type_: &TestType) -> Result<Self::Value, TracingError> {
            self.zero(type_)
        }

        fn one_primal(&self, type_: &TestType) -> Result<Self::Value, TracingError> {
            self.one(type_)
        }

        fn constant_primal(&self, constant: Self::Constant) -> Result<Self::Value, TracingError> {
            Ok(constant)
        }

        fn zero_tangent(&self, type_: &TestType) -> Result<Self::Tangent, TracingError> {
            self.zero(type_)
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
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let context = TracingContext::new(&domain, builder);
        let empty_primals: Vec<DomainTracer<'_, ScalarDomain<f64>>> = Vec::new();

        let result = context
            .value_and_grad(|_inputs: Vec<_>| panic!("closure should not run without traced inputs"), empty_primals);

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

        let result = context_a.value_and_grad(|inputs| inputs[0].clone() + inputs[1].clone(), vec![primal_a, primal_b]);

        assert!(matches!(result, Err(TracingError::MismatchedProgramBuilders)));
    }

    #[test]
    fn test_traced_value_and_grad_invokes_function_once() {
        let domain = ScalarDomain::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let context = TracingContext::new(&domain, builder);
        let primal = context.input(DataType::F64);
        let calls = Cell::new(0);

        let (_value, gradient): (DomainTracer<'_, ScalarDomain<f64>>, Vec<DomainTracer<'_, ScalarDomain<f64>>>) =
            context
                .value_and_grad(
                    |inputs| {
                        calls.set(calls.get() + 1);
                        inputs[0].clone() * inputs[0].clone()
                    },
                    vec![primal],
                )
                .unwrap();

        assert_eq!(calls.get(), 1);
        assert_eq!(gradient.len(), 1);
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
