use std::borrow::Cow;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::broadcasting::Broadcastable;
use crate::operations::arithmetic::Scale;
use crate::operations::constants::{One, OneLike, Zero, ZeroLike};
use crate::operations::trigonometric::{Cos, Sin};
use crate::parameters::Parameter;
use crate::tracing::domains::{Domain, RuntimeDomain, Tracer, TracingDomain};
use crate::tracing::{Traceable, TracingError, Value};
use crate::tracing_v2::operations::{ControlFlowError, ControlFlowValue};
use crate::tracing_v2::{
    ArrayOperation, CoordinateValue, Differentiable, DifferentiableDomain, DifferentiableTracingDomain,
    LinearArrayOperation, MatMul, MatrixTranspose, Reshape,
};
use crate::types::{ArrayType, DataType, Shape, Size, Typed};

/// Minimal array value used by `ryft-core` unit tests.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TestArray {
    /// Staged array type of this test value.
    pub(crate) r#type: ArrayType,

    /// Row-major payload used by tests that need concrete interpretation.
    pub(crate) values: Vec<f64>,
}

impl TestArray {
    /// Creates a rank-0 scalar test array.
    pub(crate) fn scalar(value: f64) -> Self {
        Self { r#type: ArrayType::scalar(DataType::F64), values: vec![value] }
    }

    /// Creates a rank-1 test array.
    pub(crate) fn vector(values: Vec<f64>) -> Self {
        Self {
            r#type: ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(values.len())]), None, None).unwrap(),
            values,
        }
    }

    /// Creates a rank-2 test array.
    pub(crate) fn matrix(rows: usize, cols: usize, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), rows * cols);
        Self {
            r#type: ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(rows), Size::Static(cols)]), None, None)
                .unwrap(),
            values,
        }
    }

    /// Returns the number of elements represented by `type`.
    pub(crate) fn element_count(r#type: &ArrayType) -> usize {
        if r#type.rank() == 0 {
            1
        } else {
            r#type.shape.dimensions.iter().map(|dimension| dimension.value().unwrap()).product()
        }
    }

    /// Applies an elementwise binary function using scalar broadcasting.
    fn binary(self, rhs: Self, function: impl Fn(f64, f64) -> f64) -> Self {
        let output_type = self.r#type.broadcast(&rhs.r#type).unwrap();
        let output_len = Self::element_count(&output_type);
        let left = self.broadcast_values(output_len);
        let right = rhs.broadcast_values(output_len);
        Self {
            r#type: output_type,
            values: left.into_iter().zip(right).map(|(left, right)| function(left, right)).collect(),
        }
    }

    /// Broadcasts the payload to `output_len`.
    fn broadcast_values(&self, output_len: usize) -> Vec<f64> {
        if self.values.len() == output_len {
            self.values.clone()
        } else if self.values.len() == 1 {
            vec![self.values[0]; output_len]
        } else {
            panic!("cannot broadcast {} values to {output_len}", self.values.len());
        }
    }
}

impl Parameter for TestArray {}

impl Display for TestArray {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}", self.values)
    }
}

impl Typed<ArrayType> for TestArray {
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl Traceable<ArrayType> for TestArray {}

impl Value<ArrayType> for TestArray {}

impl ControlFlowValue for TestArray {
    fn control_flow_predicate(&self) -> Result<bool, TracingError> {
        Err(ControlFlowError::InvalidPredicateValue { type_: self.r#type().into_owned() }.into())
    }
}

impl Zero<ArrayType> for TestArray {
    fn zero(r#type: &ArrayType) -> Result<Self, TracingError> {
        Ok(Self { r#type: r#type.clone(), values: vec![0.0; Self::element_count(r#type)] })
    }
}

impl One<ArrayType> for TestArray {
    fn one(r#type: &ArrayType) -> Result<Self, TracingError> {
        if r#type.rank() != 0 {
            return Err(crate::tracing_v2::DifferentiationError::NonScalarGradientOutput {
                output_type: r#type.clone(),
            }
            .into());
        }
        Ok(Self { r#type: r#type.clone(), values: vec![1.0] })
    }
}

impl ZeroLike for TestArray {
    fn zero_like(&self) -> Self {
        Self { r#type: self.r#type.clone(), values: vec![0.0; self.values.len()] }
    }
}

impl OneLike for TestArray {
    fn one_like(&self) -> Self {
        Self { r#type: self.r#type.clone(), values: vec![1.0; self.values.len()] }
    }
}

impl Differentiable<ArrayType> for TestArray {
    type Tangent = Self;

    fn zero_tangent(&self) -> Result<Self::Tangent, TracingError> {
        Ok(self.zero_like())
    }
}

impl CoordinateValue for TestArray {
    type Coordinate = f64;

    fn coordinate_count(&self) -> usize {
        self.values.len()
    }

    fn coordinate_basis(&self) -> Vec<Self> {
        (0..self.values.len())
            .map(|index| {
                let mut values = vec![0.0; self.values.len()];
                values[index] = 1.0;
                Self { r#type: self.r#type.clone(), values }
            })
            .collect()
    }

    fn coordinates(&self) -> Vec<Self::Coordinate> {
        self.values.clone()
    }
}

impl Add for TestArray {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, |left, right| left + right)
    }
}

impl Sub for TestArray {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.binary(rhs, |left, right| left - right)
    }
}

impl Mul for TestArray {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, |left, right| left * right)
    }
}

impl Scale for TestArray {
    type Output = Self;

    fn scale(self, factor: Self) -> Self::Output {
        factor * self
    }
}

impl Div for TestArray {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.binary(rhs, |left, right| left / right)
    }
}

impl Neg for TestArray {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { r#type: self.r#type, values: self.values.into_iter().map(|value| -value).collect() }
    }
}

impl Sin for TestArray {
    fn sin(self) -> Self {
        Self { r#type: self.r#type, values: self.values.into_iter().map(f64::sin).collect() }
    }
}

impl Cos for TestArray {
    fn cos(self) -> Self {
        Self { r#type: self.r#type, values: self.values.into_iter().map(f64::cos).collect() }
    }
}

impl MatMul for TestArray {
    fn matmul(self, rhs: Self) -> Self {
        self * rhs
    }
}

impl MatrixTranspose for TestArray {
    fn transpose_matrix(self) -> Self {
        self
    }
}

impl Reshape for TestArray {
    fn reshape(self, target_shape: Shape) -> Result<Self, TracingError> {
        let output_type = ArrayType::new(self.r#type.data_type, target_shape, None, None).unwrap();
        assert_eq!(Self::element_count(&self.r#type), Self::element_count(&output_type));
        Ok(Self { r#type: output_type, values: self.values })
    }
}

/// Minimal array domain used by `ryft-core` unit tests.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TestArrayDomain;

impl Domain for TestArrayDomain {
    type Type = ArrayType;
    type Value = TestArray;
}

impl RuntimeDomain for TestArrayDomain {
    fn zero(&self, r#type: &ArrayType) -> Result<Self::Value, TracingError> {
        TestArray::zero(r#type)
    }

    fn one(&self, r#type: &ArrayType) -> Result<Self::Value, TracingError> {
        Ok(TestArray { r#type: r#type.clone(), values: vec![1.0; TestArray::element_count(r#type)] })
    }
}

impl TracingDomain for TestArrayDomain {
    type OperationCarrier = ArrayOperation<TestArray, ArrayType>;
}

/// Minimal linear array domain used by `ryft-core` unit tests.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TestArrayLinearDomain;

impl Domain for TestArrayLinearDomain {
    type Type = ArrayType;
    type Value = TestArray;
}

impl RuntimeDomain for TestArrayLinearDomain {
    fn zero(&self, r#type: &ArrayType) -> Result<Self::Value, TracingError> {
        TestArray::zero(r#type)
    }

    fn one(&self, r#type: &ArrayType) -> Result<Self::Value, TracingError> {
        Ok(TestArray { r#type: r#type.clone(), values: vec![1.0; TestArray::element_count(r#type)] })
    }
}

impl TracingDomain for TestArrayLinearDomain {
    type OperationCarrier = LinearArrayOperation<TestArray, ArrayType>;
}

static TEST_ARRAY_LINEAR_DOMAIN: TestArrayLinearDomain = TestArrayLinearDomain;

impl DifferentiableDomain for TestArrayDomain {
    type Tangent = TestArray;
    type LinearDomain = TestArrayLinearDomain;
    type LinearOperationCarrier = LinearArrayOperation<TestArray, ArrayType>;
    type DifferentiableOperationCarrier = ArrayOperation<TestArray, ArrayType>;

    fn linear_domain(&self) -> &Self::LinearDomain {
        &TEST_ARRAY_LINEAR_DOMAIN
    }
}

impl DifferentiableTracingDomain for TestArrayDomain {
    type LinearOperationCarrier<'domain>
        = LinearArrayOperation<Tracer<'domain, TestArrayDomain>, ArrayType>
    where
        Self: 'domain;
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::fmt::Display;
    use std::rc::Rc;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use crate::differentiation::{Cotangent, LinearOperation};
    use crate::macros::check_count;
    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::tracing::domains::{Tracer, TracingContext, TracingDomain};
    use crate::tracing::{Program, ProgramBuilder, ProgramTracingContext};
    use crate::tracing_v2::linear::{Grad, JacFwd};
    use crate::tracing_v2::operations::custom::CustomTracedLinearizationRule;
    use crate::tracing_v2::{
        ArrayBatch, BatchableOperation, BatchingError, ConditionOperation, CustomOperationError, CustomPrimitive,
        DifferentiableDomain, DifferentiableOperation, JvpContext, JvpTracer, jacrev, vmap,
    };
    use crate::types::TypeError;

    use super::*;

    type TestArrayTracer<'domain> = Tracer<'domain, TestArrayDomain>;
    type TestArrayTracingContext<'domain> = TracingContext<'domain, TestArrayDomain>;

    fn assert_close(actual: f64, expected: f64) {
        let delta = (actual - expected).abs();
        assert!(delta <= 1e-9, "expected {actual} ~= {expected}; absolute error {delta} exceeded tolerance");
    }

    #[derive(Clone, Debug)]
    struct IdentityOp;

    impl Display for IdentityOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation<ArrayType> for IdentityOp {
        #[inline]
        fn name(&self) -> &'static str {
            "custom_test"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            Ok(input_types.to_vec())
        }
    }

    impl InterpretableOperation<ArrayType, TestArray> for IdentityOp {
        fn interpret(&self, inputs: &[TestArray]) -> Result<Vec<TestArray>, TracingError> {
            Ok(inputs.to_vec())
        }
    }

    #[test]
    fn test_array_batch_derives_logical_type_from_batch_axis() {
        let batch = ArrayBatch::mapped(TestArray::vector(vec![1.0, 2.0, 3.0]), 0).unwrap();

        assert_eq!(batch.axis_size(), Ok(Some(3)));
        assert_eq!(batch.logical_type(), Ok(ArrayType::scalar(DataType::F64)));
    }

    #[test]
    fn test_vmap_uses_one_packed_array_value() {
        let output = vmap::<TestArrayDomain, _, TestArray, TestArray, TestArray>(
            &TestArrayDomain,
            |x| Ok(x.clone() * x.clone() + x.sin()),
            TestArray::vector(vec![0.0, 1.0, 2.0]),
            0,
        )
        .unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]), None, None).unwrap(),
        );
        for (actual, expected) in output.values.iter().zip([0.0, 1.0 + 1.0f64.sin(), 4.0 + 2.0f64.sin()]) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn test_vmap_broadcasts_scalar_constants_inside_packed_operations() {
        let output = vmap::<TestArrayDomain, _, TestArray, TestArray, TestArray>(
            &TestArrayDomain,
            |x| Ok(x.clone() + x.one_like()),
            TestArray::vector(vec![2.0, 4.0, 6.0]),
            0,
        )
        .unwrap();

        assert_eq!(output.values, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_vmap_maps_structured_packed_inputs_and_outputs() {
        let output = vmap::<TestArrayDomain, _, (TestArray, TestArray), (TestArray, TestArray), TestArray>(
            &TestArrayDomain,
            |(left, right)| Ok((left.clone() + right.clone(), left * right)),
            (TestArray::vector(vec![1.0, 3.0]), TestArray::vector(vec![2.0, 4.0])),
            0,
        )
        .unwrap();

        assert_eq!(output.0.values, vec![3.0, 7.0]);
        assert_eq!(output.1.values, vec![2.0, 12.0]);
    }

    #[test]
    fn test_batching_rule_rejects_unaligned_batch_axes() {
        let left = ArrayBatch::mapped(TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0).unwrap();
        let right = ArrayBatch::mapped(TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 1).unwrap();

        assert!(matches!(
            ArrayOperation::<TestArray, ArrayType>::Add.batch(&[left, right]),
            Err(TracingError::Batching(BatchingError::UnsupportedBatchAxisAlignment { .. })),
        ));
    }

    #[test]
    fn test_custom_primitive_requires_explicit_batching_rule() {
        let operation = ArrayOperation::<TestArray, ArrayType>::Custom(Arc::new(CustomPrimitive::new(IdentityOp)));
        let input = ArrayBatch::mapped(TestArray::vector(vec![1.0, 2.0]), 0).unwrap();

        assert!(matches!(
            operation.batch(&[input]),
            Err(TracingError::Batching(BatchingError::MissingBatchingRule { operation })) if operation == "custom_test",
        ));
    }

    #[test]
    fn test_jacfwd_batches_basis_tangents() {
        let jacobian = TestArrayDomain
            .jacfwd::<_, (TestArray, TestArray), (TestArray, TestArray), TestArray>(
                |(x, y)| Ok((x.clone() * y.clone() + x.clone().sin(), x + y)),
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
            )
            .unwrap();

        assert_eq!(jacobian.rows(), 2);
        assert_eq!(jacobian.cols(), 2);
        assert_close(jacobian.values()[0], 3.0 + 2.0f64.cos());
        assert_close(jacobian.values()[1], 2.0);
        assert_close(jacobian.values()[2], 1.0);
        assert_close(jacobian.values()[3], 1.0);
    }

    #[test]
    fn test_jacrev_batches_basis_cotangents() {
        let jacobian = jacrev::<TestArrayDomain, _, (TestArray, TestArray), (TestArray, TestArray), TestArray>(
            &TestArrayDomain,
            |(x, y)| Ok((x.clone() * y.clone() + x.clone().sin(), x + y)),
            (TestArray::scalar(2.0), TestArray::scalar(3.0)),
        )
        .unwrap();

        assert_eq!(jacobian.rows(), 2);
        assert_eq!(jacobian.cols(), 2);
        assert_close(jacobian.values()[0], 3.0 + 2.0f64.cos());
        assert_close(jacobian.values()[1], 2.0);
        assert_close(jacobian.values()[2], 1.0);
        assert_close(jacobian.values()[3], 1.0);
    }

    #[test]
    fn test_hessian_accepts_original_scalar_function() {
        let hessian = TestArrayDomain
            .hessian::<_, (TestArray, TestArray), TestArray>(
                |(x, y)| x.clone() * y + x.sin(),
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
            )
            .unwrap();

        assert_eq!(hessian.rows(), 2);
        assert_eq!(hessian.cols(), 2);
        assert_close(hessian.values()[0], -2.0f64.sin());
        assert_close(hessian.values()[1], 1.0);
        assert_close(hessian.values()[2], 1.0);
        assert_close(hessian.values()[3], 0.0);
    }

    fn traced_bilinear_sin<'domain>(
        (x, y): (Tracer<'domain, TestArrayDomain>, Tracer<'domain, TestArrayDomain>),
    ) -> Tracer<'domain, TestArrayDomain> {
        x.clone() * y + x.sin()
    }

    #[test]
    fn test_hessian_matches_composed_jacfwd_of_grad_transform() {
        let direct = TestArrayDomain
            .hessian::<_, (TestArray, TestArray), TestArray>(
                |(x, y)| x.clone() * y + x.sin(),
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
            )
            .unwrap();
        let composed = JacFwd::new(Grad::new(traced_bilinear_sin))
            .evaluate_gradient::<TestArrayDomain, (TestArray, TestArray), TestArray>(
                &TestArrayDomain,
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
            )
            .unwrap();

        assert_eq!(direct.values(), composed.values());
        assert_eq!(direct.input_coordinate_counts(), composed.input_coordinate_counts());
        assert_eq!(direct.output_coordinate_counts(), composed.output_coordinate_counts());
    }

    fn scalar_scale_branch(
        factor: f64,
    ) -> crate::tracing_v2::FlatProgram<TestArray, ArrayOperation<TestArray, ArrayType>> {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray, ArrayType>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(ArrayOperation::Scale { factor: TestArray::scalar(factor) }, vec![input])
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_array_carrier_condition_interprets_captured_predicate() {
        let condition =
            ConditionOperation::with_captured_predicate(false, scalar_scale_branch(2.0), scalar_scale_branch(3.0))
                .unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));

        assert_eq!(operation.interpret(&[TestArray::scalar(4.0)]).map(|outputs| outputs[0].values[0]), Ok(12.0));
    }

    #[test]
    fn test_condition_jvp_uses_selected_captured_branch() {
        let condition =
            ConditionOperation::with_captured_predicate(true, scalar_scale_branch(2.0), scalar_scale_branch(3.0))
                .unwrap();
        let builder =
            Rc::new(RefCell::new(
                ProgramBuilder::<ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>>::new(),
            ));
        let mut context = JvpContext::new(&TestArrayDomain, builder.clone());
        let tangent_input = context.linear_context.input(ArrayType::scalar(DataType::F64));
        let outputs = condition
            .jvp(&mut context, &[JvpTracer { primal: TestArray::scalar(4.0), tangent: tangent_input }])
            .unwrap();

        assert_eq!(outputs[0].primal.values[0], 8.0);
        let tangent_output = outputs[0].tangent.atom_id().unwrap();
        drop(outputs);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let tangent_program =
            builder.build::<TestArray, TestArray>(vec![tangent_output], Placeholder, Placeholder).unwrap();
        assert_eq!(tangent_program.interpret(TestArray::scalar(10.0)).map(|output| output.values[0]), Ok(20.0));
    }

    #[derive(Clone, Debug)]
    struct ShiftOp {
        amount: f64,
    }

    impl ShiftOp {
        fn new(amount: f64) -> Self {
            Self { amount }
        }
    }

    impl Display for ShiftOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation<ArrayType> for ShiftOp {
        #[inline]
        fn name(&self) -> &'static str {
            "test_shift"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            check_count!("input", input_types, 1, TypeError);
            Ok(vec![input_types[0].clone()])
        }
    }

    impl InterpretableOperation<ArrayType, TestArray> for ShiftOp {
        fn interpret(&self, inputs: &[TestArray]) -> Result<Vec<TestArray>, TracingError> {
            check_count!("input", inputs, 1, TracingError);
            Ok(vec![TestArray {
                r#type: inputs[0].r#type.clone(),
                values: inputs[0].values.iter().map(|value| value + self.amount).collect(),
            }])
        }
    }

    impl LinearOperation<ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>> for ShiftOp {
        fn transpose<'transpose>(
            &self,
            _context: &mut ProgramTracingContext<
                'transpose,
                ArrayType,
                TestArray,
                LinearArrayOperation<TestArray, ArrayType>,
            >,
            output_cotangents: &[Cotangent<
                'transpose,
                ArrayType,
                TestArray,
                LinearArrayOperation<TestArray, ArrayType>,
            >],
        ) -> Result<
            Vec<Cotangent<'transpose, ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>>>,
            TracingError,
        > {
            check_count!("output", output_cotangents, 1, TracingError);
            Ok(vec![output_cotangents[0].clone()])
        }
    }

    impl DifferentiableOperation<TestArrayDomain> for ShiftOp {
        fn jvp<'jvp>(
            &self,
            _context: &mut JvpContext<'jvp, TestArrayDomain>,
            inputs: &[JvpTracer<TestArray, Tracer<'jvp, TestArrayLinearDomain>>],
        ) -> Result<Vec<JvpTracer<TestArray, Tracer<'jvp, TestArrayLinearDomain>>>, TracingError> {
            check_count!("input", inputs, 1, TracingError);
            Ok(vec![JvpTracer {
                primal: TestArray {
                    r#type: inputs[0].primal.r#type.clone(),
                    values: inputs[0].primal.values.iter().map(|value| value + self.amount).collect(),
                },
                tangent: inputs[0].tangent.clone(),
            }])
        }
    }

    impl CustomTracedLinearizationRule<TestArrayDomain> for ShiftOp {
        fn jvp_traced_linearization<'jvp, 'domain>(
            &self,
            _context: &mut JvpContext<'jvp, TestArrayTracingContext<'domain>>,
            inputs: &[JvpTracer<TestArrayTracer<'domain>, Tracer<'jvp, TestArrayTracingContext<'domain>>>],
        ) -> Result<
            Vec<JvpTracer<TestArrayTracer<'domain>, Tracer<'jvp, TestArrayTracingContext<'domain>>>>,
            TracingError,
        > {
            check_count!("input", inputs, 1, TracingError);
            let primal = apply_custom_traced_unary(
                inputs[0].primal.clone(),
                CustomPrimitive::<ArrayType, TestArray>::new(self.clone()),
            )?;
            Ok(vec![JvpTracer { primal, tangent: inputs[0].tangent.clone() }])
        }
    }

    fn apply_custom_traced_unary<'domain, D>(
        input: Tracer<'domain, D>,
        primitive: CustomPrimitive<ArrayType, TestArray>,
    ) -> Result<Tracer<'domain, D>, TracingError>
    where
        D: TracingDomain<Type = ArrayType, Value = TestArray, OperationCarrier = ArrayOperation<TestArray, ArrayType>>,
    {
        let context = input.context.clone();
        Ok(context
            .trace(ArrayOperation::Custom(Arc::new(primitive)), &[&input])?
            .into_iter()
            .next()
            .expect("unary custom primitive should produce one output"))
    }

    fn stage_custom_traced_unary<'domain, D>(
        input: Tracer<'domain, D>,
        primitive: CustomPrimitive<ArrayType, TestArray>,
    ) -> Tracer<'domain, D>
    where
        D: TracingDomain<Type = ArrayType, Value = TestArray, OperationCarrier = ArrayOperation<TestArray, ArrayType>>,
    {
        apply_custom_traced_unary(input, primitive).expect("custom primitive staging should succeed")
    }

    #[test]
    fn test_custom_primitive_base_execution_replays_without_optional_rules() {
        let primitive = CustomPrimitive::<ArrayType, TestArray>::new(ShiftOp::new(2.0));
        let (output, compiled): (
            TestArray,
            Program<ArrayType, TestArray, ArrayOperation<TestArray, ArrayType>, TestArray, TestArray>,
        ) = TestArrayDomain
            .interpret_and_trace(
                {
                    let primitive = primitive.clone();
                    move |x| Ok(stage_custom_traced_unary(x, primitive.clone()))
                },
                TestArray::scalar(3.0),
            )
            .unwrap();

        assert_eq!(output.values[0], 5.0);
        assert_eq!(compiled.interpret(TestArray::scalar(4.0)).map(|output| output.values[0]), Ok(6.0));
    }

    #[test]
    fn test_custom_primitive_missing_jvp_rule_reports_targeted_error() {
        let primitive = CustomPrimitive::<ArrayType, TestArray>::new(ShiftOp::new(2.0));
        let result: Result<(TestArray, TestArray), TracingError> = TestArrayDomain.jvp(
            {
                let primitive = primitive.clone();
                move |x| stage_custom_traced_unary(x, primitive.clone())
            },
            TestArray::scalar(3.0),
            TestArray::scalar(1.0),
        );

        assert_eq!(
            result,
            Err(TracingError::CustomOperation(CustomOperationError::MissingRule {
                op: "test_shift",
                transform: "jvp",
            })),
        );
    }

    #[test]
    fn test_custom_primitive_jvp_rule_participates_in_grad_and_traced_linearization() {
        let primitive = CustomPrimitive::<ArrayType, TestArray>::new(ShiftOp::new(2.0))
            .with_derivative_rule::<TestArrayDomain, _>(ShiftOp::new(2.0));

        assert_eq!(
            TestArrayDomain.grad(
                {
                    let primitive = primitive.clone();
                    move |x| stage_custom_traced_unary(x, primitive.clone())
                },
                TestArray::scalar(3.0),
            ),
            Ok(TestArray::scalar(1.0)),
        );

        let (output, compiled): (
            TestArray,
            Program<ArrayType, TestArray, ArrayOperation<TestArray, ArrayType>, TestArray, TestArray>,
        ) = TestArrayDomain
            .interpret_and_trace(
                {
                    let primitive = primitive.clone();
                    move |x: Tracer<'_, TestArrayDomain>| {
                        let (primal, tangent) = TestArrayDomain.jvp(
                            {
                                let primitive = primitive.clone();
                                move |inner| stage_custom_traced_unary(inner, primitive.clone())
                            },
                            x.clone(),
                            x.one_like(),
                        )?;
                        Ok(primal + tangent)
                    }
                },
                TestArray::scalar(3.0),
            )
            .unwrap();

        assert_eq!(output.values[0], 6.0);
        assert_eq!(compiled.interpret(TestArray::scalar(4.0)).map(|output| output.values[0]), Ok(7.0));
    }
}
