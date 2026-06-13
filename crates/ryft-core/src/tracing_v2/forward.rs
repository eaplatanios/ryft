#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::{Cell, RefCell};
    use std::fmt::Display;
    use std::ops::{Add, Div, Mul, Neg, Sub};
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_macros::Parameter;

    use crate::contexts::{Context, StagingContext};
    use crate::differentiation::{Cotangent, TransposableOperation};
    use crate::domains::Domain;
    use crate::macros::check_count;
    use crate::operations::arithmetic::{
        AddOperation, MulOperation, NegOperation, Scale, SubOperation, SupportsAdd, SupportsMul, SupportsNeg,
        SupportsScale, SupportsSub,
    };
    use crate::operations::constants::{
        One, OneLike, OneOperation, SupportsOne, SupportsZero, Zero, ZeroLike, ZeroOperation,
    };
    use crate::operations::scalars::{LinearScalarOperation, ScalarOperation};
    use crate::operations::trigonometric::Sin;
    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::{Parameter, ParameterError, Parameterized};
    use crate::programs::{Program, ProgramBuilder, ProgramError, Value};
    use crate::scalars::ScalarDomain;
    use crate::tracing::{AbstractTracingContext, DomainTracer, TracingContext};
    use crate::tracing_v2::differentiation::{
        FactorParameterizedOperation, JvpTracer, LinearOperationOf, ResidualFactor, TangentContext,
    };
    use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
    use crate::types::{DataType, Typed};

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

    impl Value<DataType> for DistinctPrimal {}

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
        fn zero(r#type: &DataType) -> Result<Self, ProgramError> {
            assert_eq!(r#type, &DataType::F64);
            Ok(Self(0.0))
        }
    }

    impl One<DataType> for DistinctPrimal {
        fn one(r#type: &DataType) -> Result<Self, ProgramError> {
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

    impl Value<DataType> for DistinctTangent {}

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
        fn zero(r#type: &DataType) -> Result<Self, ProgramError> {
            assert_eq!(r#type, &DataType::F64);
            Ok(Self(0.0))
        }
    }

    impl One<DataType> for DistinctTangent {
        fn one(r#type: &DataType) -> Result<Self, ProgramError> {
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

    impl Scale<DistinctTangent> for DistinctTangent {
        type Output = Self;

        fn scale(self, factor: DistinctTangent) -> Self::Output {
            Self(factor.0 * self.0)
        }
    }

    impl Scale<DistinctPrimal> for DistinctTangent {
        type Output = Self;

        fn scale(self, factor: DistinctPrimal) -> Self::Output {
            Self(factor.0 * self.0)
        }
    }

    #[derive(Clone, Debug)]
    enum DistinctLinearOperation<Factor = DistinctPrimal> {
        Zero(ZeroOperation<DataType>),
        One(OneOperation<DataType>),
        Neg,
        Add,
        Sub,
        ScaleByTangent { factor: DistinctTangent },
        ScaleByPrimal { factor: Factor },
    }

    impl<Factor: Value<DataType>> Operation<DataType> for DistinctLinearOperation<Factor> {
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

    impl InterpretableOperation<DataType, DistinctTangent> for DistinctLinearOperation<DistinctPrimal> {
        fn interpret(&self, inputs: &[DistinctTangent]) -> Result<Vec<DistinctTangent>, ProgramError> {
            match self {
                Self::Zero(operation) => operation.interpret(inputs),
                Self::One(operation) => operation.interpret(inputs),
                Self::Neg => NegOperation.interpret(inputs),
                Self::Add => AddOperation.interpret(inputs),
                Self::Sub => SubOperation.interpret(inputs),
                Self::ScaleByTangent { factor } => {
                    check_count!("input", inputs, 1, ProgramError);
                    Ok(vec![DistinctTangent(factor.0 * inputs[0].0)])
                }
                Self::ScaleByPrimal { factor } => {
                    check_count!("input", inputs, 1, ProgramError);
                    Ok(vec![DistinctTangent(factor.0 * inputs[0].0)])
                }
            }
        }
    }

    impl<Factor: Value<DataType>> SupportsZero<DataType> for DistinctLinearOperation<Factor> {
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

    impl<Factor: Value<DataType>> SupportsOne<DataType> for DistinctLinearOperation<Factor> {
        fn one_operation(r#type: DataType) -> Self {
            Self::One(OneOperation::new(r#type))
        }
    }

    impl<Factor: Value<DataType>> SupportsNeg<DataType> for DistinctLinearOperation<Factor> {
        fn neg_operation() -> Self {
            Self::Neg
        }
    }

    impl<Factor: Value<DataType>> SupportsAdd<DataType> for DistinctLinearOperation<Factor> {
        fn add_operation() -> Self {
            Self::Add
        }
    }

    impl<Factor: Value<DataType>> SupportsSub<DataType> for DistinctLinearOperation<Factor> {
        fn sub_operation() -> Self {
            Self::Sub
        }
    }

    impl SupportsScale<DataType, DistinctTangent> for DistinctLinearOperation<DistinctPrimal> {
        fn scale_operation(factor: DistinctTangent) -> Self {
            Self::ScaleByTangent { factor }
        }
    }

    impl<Factor: Value<DataType>> SupportsScale<DataType, Factor> for DistinctLinearOperation<Factor> {
        fn scale_operation(factor: Factor) -> Self {
            Self::ScaleByPrimal { factor }
        }
    }

    impl TransposableOperation<DataType, DistinctTangent, DistinctLinearOperation> for DistinctLinearOperation {
        fn transpose<'transpose>(
            &self,
            _context: &mut AbstractTracingContext<'transpose, DataType, DistinctTangent, DistinctLinearOperation>,
            _input_types: &[&DataType],
            output_cotangents: &[Cotangent<'transpose, DataType, DistinctTangent, DistinctLinearOperation>],
        ) -> Result<Vec<Cotangent<'transpose, DataType, DistinctTangent, DistinctLinearOperation>>, ProgramError>
        {
            check_count!("output", output_cotangents, 1, ProgramError);
            match (&output_cotangents[0], self) {
                (_, Self::Zero(_) | Self::One(_)) => Ok(vec![]),
                (Cotangent::Zero, Self::Neg | Self::ScaleByTangent { .. } | Self::ScaleByPrimal { .. }) => {
                    Ok(vec![Cotangent::Zero])
                }
                (Cotangent::Zero, Self::Add | Self::Sub) => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
                (Cotangent::Staged(output_cotangent), Self::Neg) => {
                    Ok(vec![Cotangent::Staged(-output_cotangent.clone())])
                }
                (Cotangent::Staged(output_cotangent), Self::Add) => {
                    Ok(vec![Cotangent::Staged(output_cotangent.clone()), Cotangent::Staged(output_cotangent.clone())])
                }
                (Cotangent::Staged(output_cotangent), Self::Sub) => {
                    Ok(vec![Cotangent::Staged(output_cotangent.clone()), Cotangent::Staged(-output_cotangent.clone())])
                }
                (Cotangent::Staged(output_cotangent), Self::ScaleByTangent { factor }) => {
                    Ok(vec![Cotangent::Staged(output_cotangent.clone().scale(*factor))])
                }
                (Cotangent::Staged(output_cotangent), Self::ScaleByPrimal { factor }) => {
                    Ok(vec![Cotangent::Staged(output_cotangent.clone().scale(*factor))])
                }
            }
        }
    }

    impl<F: Value<DataType>> FactorParameterizedOperation<DataType, F> for DistinctLinearOperation<F> {
        type WithFactor<MappedFactor: Value<DataType>> = DistinctLinearOperation<MappedFactor>;

        fn try_map_factors<MappedFactor: Value<DataType>, MapFactorFn>(
            &self,
            map_factor: &mut MapFactorFn,
        ) -> Result<Self::WithFactor<MappedFactor>, ProgramError>
        where
            MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
        {
            Ok(match self {
                Self::Zero(operation) => DistinctLinearOperation::Zero(operation.clone()),
                Self::One(operation) => DistinctLinearOperation::One(operation.clone()),
                Self::Neg => DistinctLinearOperation::Neg,
                Self::Add => DistinctLinearOperation::Add,
                Self::Sub => DistinctLinearOperation::Sub,
                Self::ScaleByTangent { factor } => DistinctLinearOperation::ScaleByTangent { factor: *factor },
                Self::ScaleByPrimal { factor } => {
                    DistinctLinearOperation::ScaleByPrimal { factor: map_factor(factor)? }
                }
            })
        }
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
        fn interpret(&self, inputs: &[DistinctPrimal]) -> Result<Vec<DistinctPrimal>, ProgramError> {
            match self {
                Self::Add => AddOperation.interpret(inputs),
                Self::Mul => MulOperation.interpret(inputs),
            }
        }
    }

    impl SupportsAdd<DataType> for DistinctPrimalOperation {
        fn add_operation() -> Self {
            Self::Add
        }
    }

    impl SupportsMul<DataType> for DistinctPrimalOperation {
        fn mul_operation() -> Self {
            Self::Mul
        }
    }

    impl<D> DifferentiableOperation<D> for DistinctPrimalOperation
    where
        D: DifferentiationContext<Type = DataType>,
        D::Value: Add<Output = D::Value> + Mul<Output = D::Value>,
        LinearOperationOf<D>:
            SupportsAdd<DataType> + SupportsScale<DataType, ResidualFactor<DataType, <D as Domain>::Value>>,
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
                Self::Add => AddOperation.jvp(context, inputs),
                Self::Mul => MulOperation.jvp(context, inputs),
            }
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct DistinctPrimalDomain;

    impl crate::tracing_v2::ProgramLinearizableOperation<DistinctPrimalDomain> for DistinctPrimalOperation {
        fn linearize_program(
            differentiable: &DistinctPrimalDomain,
            program: &Program<DataType, DistinctPrimal, Self, Vec<DistinctPrimal>, Vec<DistinctPrimal>>,
        ) -> Result<crate::tracing_v2::NestedLinearization<DistinctPrimalDomain, Self>, ProgramError> {
            crate::tracing_v2::differentiation::linearize_program(differentiable, program)
        }
    }

    impl DistinctPrimalDomain {
        fn new() -> Self {
            Self
        }
    }

    impl Domain for DistinctPrimalDomain {
        type Type = DataType;
        type Value = DistinctPrimal;
        type Constant = DistinctPrimal;
        type Operation = DistinctPrimalOperation;
    }

    impl Context for DistinctPrimalDomain {
        fn lift(&self, constant: DistinctPrimal) -> Result<DistinctPrimal, ProgramError> {
            Ok(constant)
        }

        fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError> {
            operation.interpret(inputs)
        }
    }

    impl DifferentiationContext for DistinctPrimalDomain {
        type Tangent = DistinctTangent;
        type LinearOperation<V: Value<DataType>, F: Value<DataType>> = DistinctLinearOperation<F>;

        fn zero_tangent(&self, type_: &DataType) -> Result<Self::Tangent, ProgramError> {
            DistinctTangent::zero(type_)
        }
    }

    /// Validates that [`TracingContext`] can host a JVP rule like [`AddOperation`] when the differentiable host uses
    /// [`DomainTracer`] values.
    #[test]
    fn tracing_context_dispatches_add_jvp_with_traced_primals() {
        let domain = ScalarDomain::<f64>::new();
        let outer_builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let outer_input_a = outer_builder.borrow_mut().add_input(crate::types::DataType::F64);
        let outer_input_b = outer_builder.borrow_mut().add_input(crate::types::DataType::F64);
        let outer_tracing_context = TracingContext::new(&domain, outer_builder.clone());
        let primal_a = outer_tracing_context.tracer(outer_input_a, None);
        let primal_b = outer_tracing_context.tracer(outer_input_b, None);

        let linear_builder = Rc::new(RefCell::new(ProgramBuilder::<
            DataType,
            DomainTracer<ScalarDomain<f64>>,
            LinearScalarOperation<f64, ResidualFactor<DataType, DomainTracer<ScalarDomain<f64>>>>,
        >::new()));
        let mut context = TangentContext::new(&outer_tracing_context, linear_builder.clone());
        let tangent_a = context.input(crate::types::DataType::F64);
        let tangent_b = context.input(crate::types::DataType::F64);

        let outputs = AddOperation
            .jvp(
                &mut context,
                &[JvpTracer::from_value(primal_a, tangent_a), JvpTracer::from_value(primal_b, tangent_b)],
            )
            .expect("AddOperation::jvp should run on a TracingContext differentiable host");

        assert_eq!(outputs.len(), 1);
        assert_eq!(linear_builder.borrow().instructions().len(), 1);
        assert_eq!(outer_builder.borrow().instructions().len(), 1);
    }

    #[test]
    fn concrete_jvp_supports_distinct_primal_and_zero_tangents() {
        let domain = DistinctPrimalDomain::new();

        let (primal, tangent): (DistinctPrimal, DistinctTangent) = domain
            .jvp(
                |(left, right)| left + right,
                (DistinctPrimal(2.0), DistinctPrimal(5.0)),
                (DistinctTangent(3.0), DistinctTangent(7.0)),
            )
            .unwrap();

        assert_eq!(primal, DistinctPrimal(7.0));
        assert_eq!(tangent, DistinctTangent(10.0));

        let linearized = domain.linearize(|input| Ok(input.clone() + input), DistinctPrimal(2.0)).unwrap();
        let (_, pushforward) = linearized.into_parts();
        let pushforward = pushforward.instantiate_program().unwrap();

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
        ) = domain.vjp(|input| Ok(input.clone() + input), DistinctPrimal(2.0)).unwrap();
        assert_eq!(output, DistinctPrimal(4.0));
        assert_eq!(pullback.interpret(DistinctTangent(4.0)).unwrap(), DistinctTangent(8.0));

        let (product_primal, product_tangent): (DistinctPrimal, DistinctTangent) = domain
            .jvp(
                |(left, right)| left * right,
                (DistinctPrimal(2.0), DistinctPrimal(5.0)),
                (DistinctTangent(3.0), DistinctTangent(7.0)),
            )
            .unwrap();
        assert_eq!(product_primal, DistinctPrimal(10.0));
        assert_eq!(product_tangent, DistinctTangent(29.0));

        let (reverse_primal, reverse_gradient): (DistinctPrimal, (DistinctTangent, DistinctTangent)) =
            crate::tracing_v2::value_and_grad(
                &domain,
                |(left, right)| left * right,
                (DistinctPrimal(2.0), DistinctPrimal(5.0)),
            )
            .unwrap();
        assert_eq!(reverse_primal, DistinctPrimal(10.0));
        assert_eq!(reverse_gradient, (DistinctTangent(5.0), DistinctTangent(2.0)));
    }

    #[test]
    fn jvp_rejects_mismatched_parameter_structures() {
        let domain = ScalarDomain::<f64>::new();
        let result: Result<(f64, f64), ProgramError> =
            domain.jvp(|xs| xs[0].clone(), vec![2.0f64], vec![1.0f64, 2.0f64]);
        assert!(matches!(
            result,
            Err(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure,
                right_structure,
            })) if left_structure == format!("{:?}", vec![2.0f64].parameter_structure())
                && right_structure == format!("{:?}", vec![1.0f64, 2.0f64].parameter_structure())
        ));

        let linearized = domain.linearize(|x| Ok(x.clone() * x.clone() + x.sin()), 2.0f64).unwrap();
        let (_, pushforward) = linearized.into_parts();
        let pushforward = pushforward.instantiate_program().unwrap();

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
    fn linearize_invokes_function_once_and_handles_existing_atoms() {
        let domain = ScalarDomain::<f64>::new();
        let call_count = Cell::new(0);

        let linearized = domain
            .linearize(
                |x| {
                    call_count.set(call_count.get() + 1);
                    let constant = x.context().constant(3.0);
                    Ok((x.clone(), x.clone(), x + constant.clone(), constant))
                },
                2.0f64,
            )
            .unwrap();
        let (output, pushforward) = linearized.into_parts();

        assert_eq!(call_count.get(), 1);
        assert_eq!(output, (2.0, 2.0, 5.0, 3.0));
        assert_eq!(pushforward.apply(4.0).unwrap(), (4.0, 4.0, 4.0, 0.0));
    }

    #[test]
    fn traced_jvp_requires_input_leaves() {
        let domain = ScalarDomain::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let context = TracingContext::new(&domain, builder);
        let empty_primals: Vec<DomainTracer<ScalarDomain<f64>>> = Vec::new();
        let empty_tangents: Vec<DomainTracer<ScalarDomain<f64>>> = Vec::new();

        let result: Result<(Vec<DomainTracer<ScalarDomain<f64>>>, Vec<DomainTracer<ScalarDomain<f64>>>), ProgramError> =
            DifferentiationContext::jvp(&context, |inputs| inputs, empty_primals, empty_tangents);

        assert!(matches!(result, Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 })));
    }

    #[test]
    fn traced_jvp_rejects_mismatched_program_builders() {
        let domain = ScalarDomain::<f64>::new();
        let builder_a = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let builder_b = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let context_a = TracingContext::new(&domain, builder_a);
        let context_b = TracingContext::new(&domain, builder_b);
        let primal_a = context_a.input(DataType::F64);
        let primal_b = context_b.input(DataType::F64);
        let tangent_a = context_a.input(DataType::F64);
        let tangent_b = context_a.input(DataType::F64);

        let result: Result<(DomainTracer<ScalarDomain<f64>>, DomainTracer<ScalarDomain<f64>>), ProgramError> =
            DifferentiationContext::jvp(
                &context_a,
                |inputs| inputs[0].clone() + inputs[1].clone(),
                vec![primal_a, primal_b],
                vec![tangent_a, tangent_b],
            );

        assert!(matches!(result, Err(ProgramError::MismatchedProgramBuilders)));
    }
}
