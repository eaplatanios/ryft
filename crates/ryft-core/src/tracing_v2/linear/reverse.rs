use std::fmt::Debug;

use crate::differentiation::{DifferentiableType, TransposableOperation};
use crate::operations::InterpretableOperation;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::ZeroOperation;
use crate::tracing_v2::{
    DifferentiableOperation, DifferentiationContext, DifferentiationError, DirectLinearOperationOf, LinearOperationOf,
    LinearizationTracer, ProgramLinearizableOperation, ResidualizedOperation,
};
use crate::{Domain, One, Parameterized, ParameterizedFamily, ProgramError, ProvidesContext, Type, Typed, Value};

/// Computes both the primal scalar output and its reverse-mode gradient.
///
/// This is the most direct reverse-mode API when the caller needs both the function value and the
/// gradient at the same primal point. The function must return exactly one rank-0 scalar array
/// leaf. Use [`DifferentiationContext::vjp`] directly for vector-valued functions that need an explicit output
/// cotangent.
#[allow(private_bounds)]
pub fn value_and_grad<'domain, D, F, Input>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<(<D as Domain>::Value, Input::To<D::Tangent>), DifferentiationError>
where
    D: DifferentiationContext + ProvidesContext<<D::Tangent as Value<<D as Domain>::Type>>::InterpretationContext>,
    <D as Domain>::Operation: Clone + DifferentiableOperation<D> + ProgramLinearizableOperation<D>,
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
    <D as Domain>::Type: DifferentiableType,
    DirectLinearOperationOf<D>: InterpretableOperation<<D as Domain>::Type, D::Tangent>
        + TransposableOperation<<D as Domain>::Type, D::Tangent, DirectLinearOperationOf<D>>
        + From<ZeroOperation<<D as Domain>::Type>>
        + From<AddOperation>,
    for<'a> &'a ZeroOperation<<D as Domain>::Type>: TryFrom<&'a DirectLinearOperationOf<D>>,
    LinearOperationOf<D>: ResidualizedOperation<D>,
{
    let (output, pullback) = domain.vjp(|input| Ok(function(input)), primals)?;
    // Reverse mode only defines a gradient for scalar-output functions; reject non-scalar outputs before seeding
    // (see `DifferentiationError::NonScalarGradientOutput`).
    if !output.r#type().is_scalar() {
        return Err(DifferentiationError::NonScalarGradientOutput { output_type: output.r#type().to_string() });
    }
    let seed = <D::Tangent as One<<D as Domain>::Type>>::one(output.r#type().as_ref())?;
    let context = domain.context();
    let gradient = pullback.interpret_in_context(&context, seed)?;
    Ok((output, gradient))
}

/// Computes the reverse-mode gradient of a scalar-output function.
///
/// This is [`value_and_grad`] with the primal value discarded — the analogue of JAX's
/// [`grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html). The function must return
/// exactly one rank-0 scalar array leaf. Use [`value_and_grad`] when the function value is also
/// needed, and [`grad_with_aux`] when the function carries auxiliary outputs.
#[allow(private_bounds)]
pub fn grad<'domain, D, F, Input>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<Input::To<D::Tangent>, DifferentiationError>
where
    D: DifferentiationContext + ProvidesContext<<D::Tangent as Value<<D as Domain>::Type>>::InterpretationContext>,
    <D as Domain>::Operation: Clone + DifferentiableOperation<D> + ProgramLinearizableOperation<D>,
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
    <D as Domain>::Type: DifferentiableType,
    DirectLinearOperationOf<D>: InterpretableOperation<<D as Domain>::Type, D::Tangent>
        + TransposableOperation<<D as Domain>::Type, D::Tangent, DirectLinearOperationOf<D>>
        + From<ZeroOperation<<D as Domain>::Type>>
        + From<AddOperation>,
    for<'a> &'a ZeroOperation<<D as Domain>::Type>: TryFrom<&'a DirectLinearOperationOf<D>>,
    LinearOperationOf<D>: ResidualizedOperation<D>,
{
    value_and_grad(domain, function, primals).map(|(_, gradient)| gradient)
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
) -> Result<((<D as Domain>::Value, Aux), Input::To<D::Tangent>), DifferentiationError>
where
    D: DifferentiationContext + ProvidesContext<<D::Tangent as Value<<D as Domain>::Type>>::InterpretationContext>,
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
    <D as Domain>::Operation: Clone + DifferentiableOperation<D> + ProgramLinearizableOperation<D>,
    <D as Domain>::Value: Parameterized<
            <D as Domain>::Value,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>, To = LinearizationTracer<'domain, D>>,
            To<LinearizationTracer<'domain, D>> = LinearizationTracer<'domain, D>,
            ParameterStructure: Debug + PartialEq,
        > + 'domain,
    Aux::To<LinearizationTracer<'domain, D>>:
        Parameterized<LinearizationTracer<'domain, D>, To<D::Tangent> = Aux::To<D::Tangent>>,
    D::Tangent: One<<D as Domain>::Type>,
    <D as Domain>::Type: DifferentiableType,
    DirectLinearOperationOf<D>: InterpretableOperation<<D as Domain>::Type, D::Tangent>
        + TransposableOperation<<D as Domain>::Type, D::Tangent, DirectLinearOperationOf<D>>
        + From<ZeroOperation<<D as Domain>::Type>>
        + From<AddOperation>,
    for<'a> &'a ZeroOperation<<D as Domain>::Type>: TryFrom<&'a DirectLinearOperationOf<D>>,
    LinearOperationOf<D>: ResidualizedOperation<D>,
    Input::Family: ParameterizedFamily<D::Tangent>,
    Aux::Family: ParameterizedFamily<D::Tangent>,
{
    let ((output, aux), pullback) = domain.vjp(|input| Ok(function(input)), primals)?;
    // Reverse mode only defines a gradient for scalar-output functions; reject non-scalar outputs before seeding
    // (see `DifferentiationError::NonScalarGradientOutput`).
    if !output.r#type().is_scalar() {
        return Err(DifferentiationError::NonScalarGradientOutput { output_type: output.r#type().to_string() });
    }
    let seed = <D::Tangent as One<<D as Domain>::Type>>::one(output.r#type().as_ref())?;
    let context = domain.context();
    let aux_zeros = aux
        .parameters()
        .map(|value| domain.zero_tangent(value.r#type().as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let aux_cotangent =
        <Aux::Family as ParameterizedFamily<D::Tangent>>::To::from_parameters(aux.parameter_structure(), aux_zeros)
            .map_err(ProgramError::from)?;
    let output_cotangent = (seed, aux_cotangent);
    let gradient = pullback.interpret_in_context(&context, output_cotangent)?;
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
) -> Result<(Input::To<D::Tangent>, Aux), DifferentiationError>
where
    D: DifferentiationContext + ProvidesContext<<D::Tangent as Value<<D as Domain>::Type>>::InterpretationContext>,
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
    <D as Domain>::Operation: Clone + DifferentiableOperation<D> + ProgramLinearizableOperation<D>,
    <D as Domain>::Value: Parameterized<
            <D as Domain>::Value,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>, To = LinearizationTracer<'domain, D>>,
            To<LinearizationTracer<'domain, D>> = LinearizationTracer<'domain, D>,
            ParameterStructure: Debug + PartialEq,
        > + 'domain,
    Aux::To<LinearizationTracer<'domain, D>>:
        Parameterized<LinearizationTracer<'domain, D>, To<D::Tangent> = Aux::To<D::Tangent>>,
    D::Tangent: One<<D as Domain>::Type>,
    <D as Domain>::Type: DifferentiableType,
    DirectLinearOperationOf<D>: InterpretableOperation<<D as Domain>::Type, D::Tangent>
        + TransposableOperation<<D as Domain>::Type, D::Tangent, DirectLinearOperationOf<D>>
        + From<ZeroOperation<<D as Domain>::Type>>
        + From<AddOperation>,
    for<'a> &'a ZeroOperation<<D as Domain>::Type>: TryFrom<&'a DirectLinearOperationOf<D>>,
    LinearOperationOf<D>: ResidualizedOperation<D>,
    Input::Family: ParameterizedFamily<D::Tangent>,
    Aux::Family: ParameterizedFamily<D::Tangent>,
{
    value_and_grad_with_aux(domain, function, primals).map(|((_, aux), gradient)| (gradient, aux))
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::{Cell, RefCell};
    use std::convert::Infallible;
    use std::fmt::{self, Display};
    use std::ops::{Add, Neg};
    use std::rc::Rc;

    use ryft_macros::Parameter;

    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{Cotangent, TransposableOperation};
    use crate::domains::Domain;
    use crate::macros::check_count;
    use crate::operations::arithmetic::{ADD_OPERATION_NAME, AddOperation, NegOperation, Scale, ScaleOperation};
    use crate::operations::constants::{One, OneLike, Zero, ZeroLike, ZeroOperation};
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Parameter;
    use crate::programs::{ProgramBuilder, ProgramError, Value};
    use crate::scalars::ScalarDomain;
    use crate::tracing::{AbstractTracingContext, DomainTracer, TracingContext};
    use crate::tracing_v2::{
        DifferentiableOperation, DifferentiationContext, FactorParameterizedOperation, JvpTracer, TangentContext,
    };
    use crate::types::{DataType, Type, TypeError, Typed};
    use crate::{Context, ProvidesContext};

    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq, Parameter)]
    struct TestType;

    impl Display for TestType {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("test")
        }
    }

    impl Type for TestType {
        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
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

    impl Value<TestType> for TestValue {
        type InterpretationContext = EagerContext<TestType, Self, Infallible>;

        #[inline]
        fn interpretation_context(&self) -> Option<Self::InterpretationContext> {
            Some(EagerContext::new())
        }
    }

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
        fn zero(_type: &TestType) -> Result<Self, ProgramError> {
            Ok(Self(0.0))
        }
    }

    impl One<TestType> for TestValue {
        fn one(_type: &TestType) -> Result<Self, ProgramError> {
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

    #[derive(Clone, Debug)]
    enum TestDomainOperation {
        Zero(TestType),
        Add,
    }

    impl Operation<TestType> for TestDomainOperation {
        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Zero(_) => "zero",
                Self::Add => ADD_OPERATION_NAME,
            }
        }

        fn infer_output_types(&self, input_types: &[TestType]) -> Result<Vec<TestType>, TypeError> {
            let expected = match self {
                Self::Zero(_) => 0,
                Self::Add => 2,
            };
            check_count!("input", input_types, expected, TypeError);
            Ok(vec![TestType])
        }
    }

    impl InterpretableOperation<TestType, TestValue> for TestDomainOperation {
        fn interpret(
            &self,
            _context: &<TestValue as Value<TestType>>::InterpretationContext,
            inputs: &[TestValue],
        ) -> Result<Vec<TestValue>, ProgramError> {
            let expected = match self {
                Self::Zero(_) => 0,
                Self::Add => 2,
            };
            check_count!("input", inputs, expected, ProgramError);
            Ok(vec![match self {
                Self::Zero(_) => TestValue(0.0),
                Self::Add => inputs[0].clone() + inputs[1].clone(),
            }])
        }
    }

    impl From<AddOperation> for TestDomainOperation {
        fn from(_operation: AddOperation) -> Self {
            Self::Add
        }
    }

    impl From<ZeroOperation<TestType>> for TestDomainOperation {
        fn from(operation: ZeroOperation<TestType>) -> Self {
            Self::Zero(operation.r#type().clone())
        }
    }

    /// Generic JVP dispatch for the test operation enum, mirroring the closed-enum dispatch shape so the operations
    /// also differentiate against derived contexts such as the nested symbolic-linearization context (whose primal
    /// values are tracers). Primal results are produced through [`TangentContext::bind_primal`] so that they are
    /// interpreted eagerly or staged depending on the context.
    impl<D> DifferentiableOperation<D> for TestDomainOperation
    where
        D: DifferentiationContext<Type = TestType, Constant = TestValue> + Domain<Operation = TestDomainOperation>,
        D::Value: Add<Output = D::Value>,
        LinearOperationOf<D>: From<AddOperation>,
    {
        fn jvp<'jvp>(
            &self,
            context: &mut TangentContext<'jvp, D>,
            inputs: &[JvpTracer<'jvp, D>],
        ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
        where
            D: 'jvp,
        {
            match self {
                Self::Zero(r#type) => {
                    check_count!("input", inputs, 0, ProgramError);
                    let mut primals = context.bind_primal(Self::Zero(r#type.clone()), &[])?;
                    check_count!("output", primals, 1, ProgramError);
                    Ok(vec![JvpTracer::from_zero_tangent(primals.pop().expect("checked above"), r#type.clone())])
                }
                Self::Add => {
                    check_count!("input", inputs, 2, ProgramError);
                    Ok(vec![JvpTracer::new(
                        inputs[0].primal().clone() + inputs[1].primal().clone(),
                        inputs[0].tangent().clone() + inputs[1].tangent().clone(),
                    )])
                }
            }
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
        fn interpret(
            &self,
            _context: &<TestValue as Value<TestType>>::InterpretationContext,
            inputs: &[TestValue],
        ) -> Result<Vec<TestValue>, ProgramError> {
            let expected = match self {
                Self::Zero(_) => 0,
                Self::Add => 2,
                Self::Neg | Self::Scale { .. } => 1,
            };
            check_count!("input", inputs, expected, ProgramError);
            Ok(vec![match self {
                Self::Zero(_) => TestValue(0.0),
                Self::Add => inputs[0].clone() + inputs[1].clone(),
                Self::Neg => -inputs[0].clone(),
                Self::Scale { factor } => TestValue(factor.0 * inputs[0].0),
            }])
        }
    }

    impl From<AddOperation> for TestLinearOperation {
        fn from(_operation: AddOperation) -> Self {
            Self::Add
        }
    }

    impl From<ZeroOperation<TestType>> for TestLinearOperation {
        fn from(operation: ZeroOperation<TestType>) -> Self {
            Self::Zero(operation.r#type().clone())
        }
    }

    impl From<NegOperation> for TestLinearOperation {
        fn from(_operation: NegOperation) -> Self {
            Self::Neg
        }
    }

    impl From<ScaleOperation<TestType, TestValue>> for TestLinearOperation {
        fn from(operation: ScaleOperation<TestType, TestValue>) -> Self {
            Self::Scale { factor: operation.factor().clone() }
        }
    }

    impl TransposableOperation<TestType, TestValue, TestLinearOperation> for TestLinearOperation {
        fn transpose<'transpose>(
            &self,
            _context: &mut AbstractTracingContext<'transpose, TestType, TestValue, TestLinearOperation>,
            _input_types: &[&TestType],
            output_cotangents: &[Cotangent<'transpose, TestType, TestValue, TestLinearOperation>],
        ) -> Result<Vec<Cotangent<'transpose, TestType, TestValue, TestLinearOperation>>, ProgramError> {
            check_count!("output", output_cotangents, 1, ProgramError);
            Ok(match self {
                Self::Zero(_) => Vec::new(),
                Self::Add => vec![output_cotangents[0].clone(), output_cotangents[0].clone()],
                Self::Neg => match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => vec![Cotangent::Staged(-cotangent.clone())],
                    Cotangent::Zero => vec![Cotangent::Zero],
                },
                Self::Scale { factor } => match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => vec![Cotangent::Staged(cotangent.scale(factor.clone()))],
                    Cotangent::Zero => vec![Cotangent::Zero],
                },
            })
        }
    }

    impl<Factor: Value<TestType>> FactorParameterizedOperation<TestType, Factor> for TestLinearOperation {
        type WithFactor<MappedFactor: Value<TestType>> = Self;

        fn try_map_factors<MappedFactor: Value<TestType>, MapFactorFn>(
            &self,
            _map_factor: &mut MapFactorFn,
        ) -> Result<Self::WithFactor<MappedFactor>, ProgramError>
        where
            MapFactorFn: FnMut(&Factor) -> Result<MappedFactor, ProgramError>,
        {
            Ok(self.clone())
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct TestDomain;

    impl crate::tracing_v2::ProgramLinearizableOperation<TestDomain> for TestDomainOperation {
        fn linearize_program(
            differentiable: &TestDomain,
            program: &crate::programs::Program<TestType, TestValue, Self, Vec<TestValue>, Vec<TestValue>>,
        ) -> Result<crate::tracing_v2::NestedLinearization<TestDomain, Self>, ProgramError> {
            crate::tracing_v2::differentiation::linearize_program(differentiable, program)
        }
    }

    impl Domain for TestDomain {
        type Type = TestType;
        type Value = TestValue;
        type Constant = TestValue;
        type Operation = TestDomainOperation;
    }

    impl Context for TestDomain {
        fn lift(&self, constant: TestValue) -> Result<TestValue, ProgramError> {
            Ok(constant)
        }

        fn bind<P: Into<Self::Operation>>(
            &self,
            operation: P,
            inputs: &[Self::Value],
        ) -> Result<Vec<Self::Value>, ProgramError> {
            let operation = operation.into();
            operation.interpret(&EagerContext::new(), inputs)
        }
    }

    impl DifferentiationContext for TestDomain {
        type Tangent = TestValue;
        type LinearOperation<V: Value<TestType>, F: Value<TestType>> = TestLinearOperation;

        fn zero_tangent(&self, type_: &Self::Type) -> Result<Self::Tangent, ProgramError> {
            let mut outputs = self.bind(ZeroOperation::new(type_.clone()), &[])?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(outputs.pop().expect("zero operation produces exactly one output"))
        }
    }

    impl ProvidesContext<<TestValue as Value<TestType>>::InterpretationContext> for TestDomain {
        fn context(&self) -> <TestValue as Value<TestType>>::InterpretationContext {
            EagerContext::new()
        }
    }

    #[test]
    fn test_linearize_supports_non_array_type_metadata() {
        let domain = TestDomain;
        let (output, pushforward) = domain.linearize(|x| Ok(x.clone() + x), TestValue(3.0)).unwrap();

        assert_eq!(output, TestValue(6.0));
        let tangent_context = domain.context();
        assert_eq!(pushforward.apply(&tangent_context, TestValue(5.0)), Ok(TestValue(10.0)));
    }

    #[test]
    fn test_traced_value_and_grad_requires_input_leaves() {
        let domain = ScalarDomain::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let context = TracingContext::new(&domain, builder);
        let empty_primals: Vec<DomainTracer<ScalarDomain<f64>>> = Vec::new();

        let result = context
            .value_and_grad(|_inputs: Vec<_>| panic!("closure should not run without traced inputs"), empty_primals);

        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 }))
        ));
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

        assert!(matches!(result, Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders))));
    }

    #[test]
    fn test_traced_value_and_grad_invokes_function_once() {
        let domain = ScalarDomain::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let context = TracingContext::new(&domain, builder);
        let primal = context.input(DataType::F64);
        let calls = Cell::new(0);

        let (_value, gradient): (DomainTracer<ScalarDomain<f64>>, Vec<DomainTracer<ScalarDomain<f64>>>) = context
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
    fn test_grad_returns_only_the_gradient() {
        let domain = ScalarDomain::<f64>::new();

        let gradient: (f64, f64) = grad(&domain, |(x, y)| x.clone() * y.clone() + x, (2.0f64, 3.0f64)).unwrap();

        assert_eq!(gradient, (4.0, 2.0));
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
