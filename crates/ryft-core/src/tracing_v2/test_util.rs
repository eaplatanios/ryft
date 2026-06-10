use std::borrow::Cow;
use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::broadcasting::Broadcastable;
use crate::contexts::Context;
use crate::domains::Domain;
use crate::operations::InterpretableOperation;
use crate::operations::arithmetic::Scale;
use crate::operations::constants::{Fill, One, OneLike, Zero, ZeroLike};
use crate::operations::trigonometric::{Cos, Sin};
use crate::parameters::{Parameter, Placeholder};
use crate::programs::{ProgramBuilder, ProgramError, Value};
use crate::tracing_v2::operations::{ControlFlowError, ControlFlowValue};
use crate::tracing_v2::{
    ArrayOperation, CoordinateValue, DifferentiationContext, LinearArrayOperation, Reshape, ResidualFactor,
    ResidualizedOperation,
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
    /// Creates a test array from its staged array type and row-major payload.
    pub(crate) fn new(r#type: ArrayType, values: Vec<f64>) -> Self {
        Self { r#type, values }
    }

    /// Creates a rank-0 scalar test array.
    pub(crate) fn scalar(value: f64) -> Self {
        Self::new(ArrayType::scalar(DataType::F64), vec![value])
    }

    /// Creates a rank-1 test array.
    pub(crate) fn vector(values: Vec<f64>) -> Self {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(values.len())]), None, None).unwrap();
        Self::new(r#type, values)
    }

    /// Creates a rank-2 test array.
    pub(crate) fn matrix(rows: usize, cols: usize, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), rows * cols);
        let r#type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(rows), Size::Static(cols)]), None, None)
                .unwrap();
        Self::new(r#type, values)
    }

    /// Returns the staged array type of this test value.
    pub(crate) fn array_type(&self) -> &ArrayType {
        &self.r#type
    }

    /// Returns the row-major payload used by concrete test interpretation.
    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns the number of elements represented by `type`.
    pub(crate) fn element_count(r#type: &ArrayType) -> usize {
        r#type.element_count().unwrap().unwrap()
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

impl Value<ArrayType> for TestArray {}

impl crate::tracing_v2::rematerialization::RematerializationName for TestArray {
    #[inline]
    fn rematerialization_name(self, _name: &str) -> Self {
        self
    }
}

impl ControlFlowValue for TestArray {
    fn control_flow_predicate(&self) -> Result<bool, ProgramError> {
        // Accept scalar Boolean predicates (rank-0, one element, encoded as 0.0=false / nonzero=true)
        // so that lane-varying while can extract a final `any(mask)` result. Higher-rank predicates
        // still error because they cannot collapse to a single Boolean.
        if self.r#type.rank() == 0 && self.r#type.data_type() == DataType::Boolean && self.values.len() == 1 {
            return Ok(self.values[0] != 0.0);
        }
        Err(ControlFlowError::InvalidPredicateValue { type_: self.r#type().into_owned() }.into())
    }
}

impl Zero<ArrayType> for TestArray {
    fn zero(r#type: &ArrayType) -> Result<Self, ProgramError> {
        Ok(Self { r#type: r#type.clone(), values: vec![0.0; Self::element_count(r#type)] })
    }
}

impl One<ArrayType> for TestArray {
    fn one(r#type: &ArrayType) -> Result<Self, ProgramError> {
        Ok(Self { r#type: r#type.clone(), values: vec![1.0; Self::element_count(r#type)] })
    }
}

impl Fill<ArrayType, f64> for TestArray {
    fn fill(r#type: &ArrayType, value: f64) -> Result<Self, ProgramError> {
        Ok(Self { r#type: r#type.clone(), values: vec![value; Self::element_count(r#type)] })
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

    fn stack(values: Vec<Self>) -> Result<Self, ProgramError> {
        let lane_count = values.len();
        assert!(lane_count > 0, "cannot stack zero values");
        let first_type = &values[0].r#type;
        for value in values.iter().skip(1) {
            assert_eq!(value.r#type, *first_type, "stacked test arrays must share the same type");
        }
        let stacked_dimensions = std::iter::once(Size::Static(lane_count))
            .chain(first_type.shape().dimensions().iter().copied())
            .collect::<Vec<_>>();
        let stacked_type = ArrayType::new(first_type.data_type(), Shape::new(stacked_dimensions), None, None).unwrap();
        let mut stacked_values = Vec::with_capacity(lane_count * values[0].values.len());
        for value in values {
            stacked_values.extend(value.values);
        }
        Ok(Self { r#type: stacked_type, values: stacked_values })
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

impl crate::tracing_v2::operations::broadcast::BroadcastInDim for TestArray {
    fn broadcast_in_dim(self, target_type: ArrayType, broadcast_dimensions: Vec<usize>) -> Self {
        let input_shape: Vec<usize> =
            self.r#type.shape().dimensions().iter().map(|size| size.value().unwrap()).collect();
        let target_shape: Vec<usize> =
            target_type.shape().dimensions().iter().map(|size| size.value().unwrap()).collect();
        let values = crate::tracing_v2::operations::broadcast::broadcast_in_dim_evaluate(
            self.values.as_slice(),
            input_shape.as_slice(),
            target_shape.as_slice(),
            broadcast_dimensions.as_slice(),
        );
        Self { r#type: target_type, values }
    }
}

impl crate::tracing_v2::operations::dot::Dot for TestArray {
    fn dot(self, rhs: Self, dimensions: &crate::tracing_v2::operations::dot::DotDimensionNumbers) -> Self {
        let lhs_shape: Vec<usize> = self.r#type.shape().dimensions().iter().map(|size| size.value().unwrap()).collect();
        let rhs_shape: Vec<usize> = rhs.r#type.shape().dimensions().iter().map(|size| size.value().unwrap()).collect();
        let (values, output_shape) = crate::tracing_v2::operations::dot::dot_general_evaluate(
            self.values.as_slice(),
            lhs_shape.as_slice(),
            rhs.values.as_slice(),
            rhs_shape.as_slice(),
            dimensions,
            || 0.0f64,
            |accumulator, lhs_value, rhs_value| accumulator + lhs_value * rhs_value,
        );
        let output_dimensions: Vec<Size> = output_shape.iter().map(|size| Size::Static(*size)).collect();
        let output_type = ArrayType::new(self.r#type.data_type(), Shape::new(output_dimensions), None, None).unwrap();
        Self { r#type: output_type, values }
    }
}

impl crate::tracing_v2::operations::dot::LeftDot for TestArray {
    #[inline]
    fn left_dot(self, factor: Self, dimensions: &crate::tracing_v2::operations::dot::DotDimensionNumbers) -> Self {
        use crate::tracing_v2::operations::dot::Dot;
        factor.dot(self, dimensions)
    }
}

impl crate::tracing_v2::operations::dot::RightDot for TestArray {
    #[inline]
    fn right_dot(self, factor: Self, dimensions: &crate::tracing_v2::operations::dot::DotDimensionNumbers) -> Self {
        use crate::tracing_v2::operations::dot::Dot;
        self.dot(factor, dimensions)
    }
}

impl crate::tracing_v2::operations::transpose::Transpose for TestArray {
    fn transpose(self, permutation: Vec<usize>) -> Self {
        if crate::tracing_v2::operations::transpose::transpose_is_identity(&permutation) {
            return self;
        }
        let shape: Vec<usize> = self.r#type.shape().dimensions().iter().map(|size| size.value().unwrap()).collect();
        let (values, output_shape) = crate::tracing_v2::operations::transpose::transpose_evaluate(
            self.values.as_slice(),
            shape.as_slice(),
            permutation.as_slice(),
        );
        let output_dimensions: Vec<Size> = output_shape.iter().map(|size| Size::Static(*size)).collect();
        let output_type = ArrayType::new(self.r#type.data_type(), Shape::new(output_dimensions), None, None).unwrap();
        Self { r#type: output_type, values }
    }
}

impl Reshape for TestArray {
    fn reshape(self, target_shape: Shape) -> Result<Self, ProgramError> {
        let output_type = ArrayType::new(self.r#type.data_type(), target_shape, None, None).unwrap();
        assert_eq!(Self::element_count(&self.r#type), Self::element_count(&output_type));
        Ok(Self { r#type: output_type, values: self.values })
    }
}

impl crate::tracing_v2::operations::select::Select for TestArray {
    fn select(predicate: Self, on_true: Self, on_false: Self) -> Result<Self, ProgramError> {
        assert_eq!(predicate.r#type, on_true.r#type, "select predicate and on_true must share the same type");
        assert_eq!(on_true.r#type, on_false.r#type, "select on_true and on_false must share the same type");
        let values: Vec<f64> = predicate
            .values
            .iter()
            .zip(on_true.values.iter())
            .zip(on_false.values.iter())
            .map(|((pred, t), f)| if *pred != 0.0 { *t } else { *f })
            .collect();
        Ok(Self { r#type: on_true.r#type, values })
    }
}

impl crate::tracing_v2::operations::compare::Compare for TestArray {
    type Output = Self;

    fn compare(self, rhs: Self, kind: crate::tracing_v2::operations::compare::CompareKind) -> Self {
        use crate::tracing_v2::operations::compare::CompareKind;
        let output_shape = self.r#type.shape().clone();
        let output_len = Self::element_count(&self.r#type);
        let left = self.broadcast_values(output_len);
        let right = rhs.broadcast_values(output_len);
        let values: Vec<f64> = left
            .into_iter()
            .zip(right)
            .map(|(left, right)| {
                let predicate = match kind {
                    CompareKind::Eq => left == right,
                    CompareKind::Ne => left != right,
                    CompareKind::Lt => left < right,
                    CompareKind::Le => left <= right,
                    CompareKind::Gt => left > right,
                    CompareKind::Ge => left >= right,
                };
                if predicate { 1.0 } else { 0.0 }
            })
            .collect();
        let output_type = ArrayType::new(DataType::Boolean, output_shape, None, None).unwrap();
        Self { r#type: output_type, values }
    }
}

impl crate::tracing_v2::operations::logical::LogicalBinary for TestArray {
    fn logical_binary(self, rhs: Self, kind: crate::tracing_v2::operations::logical::LogicalKind) -> Self {
        use crate::tracing_v2::operations::logical::LogicalKind;
        let output_len = Self::element_count(&self.r#type);
        let left = self.broadcast_values(output_len);
        let right = rhs.broadcast_values(output_len);
        let values: Vec<f64> = left
            .into_iter()
            .zip(right)
            .map(|(left, right)| {
                let left_bool = left != 0.0;
                let right_bool = right != 0.0;
                let result = match kind {
                    LogicalKind::And => left_bool && right_bool,
                    LogicalKind::Or => left_bool || right_bool,
                    LogicalKind::Xor => left_bool ^ right_bool,
                    LogicalKind::Not => unreachable!("LogicalKind::Not is unary"),
                };
                if result { 1.0 } else { 0.0 }
            })
            .collect();
        Self { r#type: self.r#type, values }
    }
}

impl crate::tracing_v2::operations::logical::LogicalNot for TestArray {
    fn logical_not(self) -> Self {
        let values: Vec<f64> = self.values.into_iter().map(|value| if value != 0.0 { 0.0 } else { 1.0 }).collect();
        Self { r#type: self.r#type, values }
    }
}

impl crate::tracing_v2::operations::reduce::Reduce for TestArray {
    fn reduce(self, axes: &[usize], kind: crate::tracing_v2::operations::reduce::ReductionKind) -> Self {
        use crate::tracing_v2::operations::reduce::{ReductionKind, reduce_evaluate};
        if axes.is_empty() {
            return self;
        }
        let shape: Vec<usize> = self.r#type.shape().dimensions().iter().map(|size| size.value().unwrap()).collect();
        let (reduced_values, reduced_shape) = match kind {
            ReductionKind::Sum | ReductionKind::Mean => {
                reduce_evaluate(self.values.as_slice(), shape.as_slice(), axes, || 0.0, |acc, value| acc + value)
            }
            ReductionKind::Max => reduce_evaluate(
                self.values.as_slice(),
                shape.as_slice(),
                axes,
                || f64::NEG_INFINITY,
                |acc, value| acc.max(value),
            ),
            ReductionKind::Min => reduce_evaluate(
                self.values.as_slice(),
                shape.as_slice(),
                axes,
                || f64::INFINITY,
                |acc, value| acc.min(value),
            ),
            ReductionKind::Any => reduce_evaluate(
                self.values.as_slice(),
                shape.as_slice(),
                axes,
                || 0.0,
                |acc, value| if acc != 0.0 || value != 0.0 { 1.0 } else { 0.0 },
            ),
            ReductionKind::All => reduce_evaluate(
                self.values.as_slice(),
                shape.as_slice(),
                axes,
                || 1.0,
                |acc, value| if acc != 0.0 && value != 0.0 { 1.0 } else { 0.0 },
            ),
        };
        let mut values = reduced_values;
        if matches!(kind, ReductionKind::Mean) {
            let reduced_count: usize = axes.iter().map(|axis| shape[*axis]).product();
            let divisor = reduced_count.max(1) as f64;
            for value in values.iter_mut() {
                *value /= divisor;
            }
        }
        let output_dimensions: Vec<Size> = reduced_shape.iter().map(|size| Size::Static(*size)).collect();
        let data_type = self.r#type.data_type();
        let output_type = ArrayType::new(data_type, Shape::new(output_dimensions), None, None).unwrap();
        Self { r#type: output_type, values }
    }
}

/// Minimal array domain used by `ryft-core` unit tests.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TestArrayDomain;

impl Domain for TestArrayDomain {
    type Type = ArrayType;
    type Value = TestArray;
    type Constant = TestArray;
    type Operation = ArrayOperation<TestArray, ArrayType>;
}

impl Context for TestArrayDomain {
    fn lift(&self, constant: TestArray) -> Result<TestArray, ProgramError> {
        Ok(constant)
    }

    fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError> {
        operation.interpret(inputs)
    }
}

impl DifferentiationContext for TestArrayDomain {
    type Tangent = TestArray;
    type LinearOperation<V: Value<ArrayType>, F: Value<ArrayType>> =
        LinearArrayOperation<V, TestArray, ArrayType, Infallible, F>;

    fn zero_tangent(&self, type_: &ArrayType) -> Result<Self::Tangent, ProgramError> {
        TestArray::zero(type_)
    }
}

/// Asserts that `actual` is within absolute tolerance `1e-9` of `expected`.
pub(crate) fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(delta <= 1e-9, "expected {actual} ~= {expected}; absolute error {delta} exceeded tolerance");
}

/// Builds a single-input flat program that scales its scalar input by `factor`.
pub(crate) fn scalar_scale_branch(
    factor: f64,
) -> crate::tracing_v2::FlatProgram<TestArray, ArrayOperation<TestArray, ArrayType>> {
    let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray, ArrayType>>::new();
    let input = builder.add_input(ArrayType::scalar(DataType::F64));
    let output = builder
        .add_instruction(ArrayOperation::Scale { factor: TestArray::scalar(factor) }, vec![input])
        .unwrap()[0];
    builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::contexts::StagingContext;
    use crate::operations::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tracing_v2::{
        ArrayBatch, BatchableOperation, ConditionOperation, DifferentiableDomainExtension, DifferentiableOperation,
        JvpTracer, TangentContext, jacrev,
    };

    use super::*;

    #[test]
    fn test_dot_batches_mixed_lhs_batched_rhs_lane_uniform() {
        // LHS is mapped at axis 0 with per-lane shape [3]; RHS is lane-uniform with shape [3].
        // Per-lane semantics: dot(lhs_row, rhs) over the shared K=3 dimension. The batching rule
        // should broadcast the RHS to gain a singleton batch axis at position 0, then thread the
        // batch axis through `lift_dot_dimensions`.
        use crate::tracing_v2::operations::dot::{DotDimensionNumbers, DotOperation};
        let lhs = ArrayBatch::mapped(TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0).unwrap();
        let rhs = ArrayBatch::unbatched(TestArray::vector(vec![10.0, 100.0, 1000.0]));
        let dimensions = DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]);
        let outputs = DotOperation::new(dimensions).batch(&(), &[lhs, rhs]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        // Lane 0: 1*10 + 2*100 + 3*1000 = 3210; lane 1: 4*10 + 5*100 + 6*1000 = 6540.
        assert_eq!(outputs[0].value().values(), &[3210.0, 6540.0]);
    }

    #[test]
    fn test_reduce_sum_jvp_linearizes_to_itself() {
        // Verify the JVP rule for `ReduceOperation::Sum`: the tangent of `sum(x)` is `sum(Δx)`.
        // We exercise the rule directly on a `Tangent::Value` over a TestArray vector. Result
        // should match summing the values directly.
        use crate::differentiation::Tangent;
        use crate::tracing_v2::operations::reduce::{ReduceOperation, ReductionKind};
        let primal = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let tangent_value = TestArray::vector(vec![0.5, 0.5, 0.5, 0.5]);
        let tangent: Tangent<ArrayType, TestArray> = Tangent::Value(tangent_value);

        let operation = ReduceOperation::new(vec![0], ReductionKind::Sum);

        // Primal: reduce(x, [0], Sum) on `TestArray` directly.
        let primal_output = operation.interpret(std::slice::from_ref(&primal)).unwrap().into_iter().next().unwrap();
        assert_eq!(primal_output.values(), &[10.0]);

        // Tangent: linearizes to itself (Sum is linear), so the tangent of the reduce is the
        // reduce of the tangent.
        let tangent_outputs = operation.interpret(std::slice::from_ref(&tangent)).unwrap();
        let tangent_output = tangent_outputs.into_iter().next().unwrap();
        match tangent_output {
            Tangent::Value(value) => assert_eq!(value.values(), &[2.0]),
            Tangent::Zero(_) => panic!("expected non-zero tangent output"),
        }
    }

    #[test]
    fn test_lane_varying_while_terminates_lanes_independently() {
        // Build a batched while loop with a per-lane termination predicate. Each lane starts at a
        // different value and decrements by 1 until it reaches 0. Lane 0 (initial 3.0) iterates
        // three times, lane 1 (initial 1.0) iterates once, lane 2 (initial 2.0) iterates twice;
        // inactive lanes retain their final state via per-lane `Select` masking.
        use crate::tracing_v2::operations::compare::CompareKind;
        use crate::tracing_v2::operations::{FlatProgram, WhileOperation};
        type TestOp = ArrayOperation<TestArray, ArrayType>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);

        // Condition program: state -> (state > 0). Returns a scalar Boolean.
        let mut condition_builder = ProgramBuilder::<ArrayType, TestArray, TestOp>::new();
        let cond_input = condition_builder.add_input(scalar_f64.clone());
        let cond_zero = condition_builder.add_instruction(TestOp::ZeroLike, vec![cond_input]).unwrap()[0];
        let cond_output = condition_builder
            .add_instruction(TestOp::Compare { kind: CompareKind::Gt }, vec![cond_input, cond_zero])
            .unwrap()[0];
        let condition: FlatProgram<TestArray, TestOp> = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![cond_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Body program: state -> state - 1.
        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, TestOp>::new();
        let body_input = body_builder.add_input(scalar_f64);
        let body_one = body_builder.add_instruction(TestOp::OneLike, vec![body_input]).unwrap()[0];
        let body_output = body_builder.add_instruction(TestOp::Sub, vec![body_input, body_one]).unwrap()[0];
        let body: FlatProgram<TestArray, TestOp> = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![body_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let while_op = WhileOperation::<TestArray, TestOp, ArrayType>::new(condition, body).unwrap();

        let initial_state = ArrayBatch::mapped(TestArray::vector(vec![3.0, 1.0, 2.0]), 0).unwrap();
        let outputs = while_op.batch(&(), &[initial_state]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        // Each lane terminates when its value reaches 0; inactive lanes retain their last value.
        assert_eq!(outputs[0].value().values(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_jacfwd_batches_basis_tangents() {
        let jacobian = TestArrayDomain
            .jacfwd(
                |(x, y)| Ok((x.clone() * y.clone() + x.clone().sin(), x + y)),
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
            )
            .unwrap();

        let (row_0, row_1) = jacobian.rows();
        let (block_00, block_01) = row_0.partials();
        let (block_10, block_11) = row_1.partials();

        assert_eq!(block_00.output_shape(), &[] as &[usize]);
        assert_eq!(block_00.input_shape(), &[] as &[usize]);

        assert_close(block_00.values()[0], 3.0 + 2.0f64.cos());
        assert_close(block_01.values()[0], 2.0);
        assert_close(block_10.values()[0], 1.0);
        assert_close(block_11.values()[0], 1.0);
    }

    #[test]
    fn test_jacrev_batches_basis_cotangents() {
        let jacobian = jacrev(
            &TestArrayDomain,
            |(x, y)| Ok((x.clone() * y.clone() + x.clone().sin(), x + y)),
            (TestArray::scalar(2.0), TestArray::scalar(3.0)),
        )
        .unwrap();

        let (row_0, row_1) = jacobian.rows();
        let (block_00, block_01) = row_0.partials();
        let (block_10, block_11) = row_1.partials();

        assert_close(block_00.values()[0], 3.0 + 2.0f64.cos());
        assert_close(block_01.values()[0], 2.0);
        assert_close(block_10.values()[0], 1.0);
        assert_close(block_11.values()[0], 1.0);
    }

    #[test]
    fn test_jacfwd_iter_blocks_yields_each_output_input_pair() {
        let jacobian = TestArrayDomain
            .jacfwd(
                |(x, y)| Ok((x.clone() * y.clone() + x.clone().sin(), x + y)),
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
            )
            .unwrap();

        let triples = jacobian
            .iter_blocks()
            .map(|(output_path, input_path, block)| {
                (output_path.to_string(), input_path.to_string(), block.values()[0])
            })
            .collect::<Vec<_>>();

        assert_eq!(triples.len(), 4);
        assert_eq!(triples[0].0, "$.0");
        assert_eq!(triples[0].1, "$.0");
        assert_close(triples[0].2, 3.0 + 2.0f64.cos());
        assert_eq!(triples[1].0, "$.0");
        assert_eq!(triples[1].1, "$.1");
        assert_close(triples[1].2, 2.0);
        assert_eq!(triples[2].0, "$.1");
        assert_eq!(triples[2].1, "$.0");
        assert_close(triples[2].2, 1.0);
        assert_eq!(triples[3].0, "$.1");
        assert_eq!(triples[3].1, "$.1");
        assert_close(triples[3].2, 1.0);
    }

    #[test]
    fn test_hessian_accepts_original_scalar_function() {
        let hessian = TestArrayDomain
            .hessian(|(x, y)| x.clone() * y + x.sin(), (TestArray::scalar(2.0), TestArray::scalar(3.0)))
            .unwrap();

        let (row_0, row_1) = hessian.rows();
        let (block_00, block_01) = row_0.partials();
        let (block_10, block_11) = row_1.partials();

        assert_close(block_00.values()[0], -2.0f64.sin());
        assert_close(block_01.values()[0], 1.0);
        assert_close(block_10.values()[0], 1.0);
        assert_close(block_11.values()[0], 0.0);
    }

    #[test]
    fn test_jacfwd_handles_function_with_independent_outputs() {
        // f(x, y) = (x*y + sin(x), y, x + y) — output[1] is independent of x.
        let jacobian = TestArrayDomain
            .jacfwd(
                |(x, y)| Ok((x.clone() * y.clone() + x.clone().sin(), y.clone(), x + y)),
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
            )
            .unwrap();

        let triples = jacobian
            .iter_blocks()
            .map(|(output_path, input_path, block)| {
                (output_path.to_string(), input_path.to_string(), block.values()[0])
            })
            .collect::<Vec<_>>();

        // 3 outputs * 2 inputs = 6 blocks
        assert_eq!(triples.len(), 6);
        // d(x*y + sin(x))/dx = y + cos(x) = 3 + cos(2)
        assert_close(triples[0].2, 3.0 + 2.0f64.cos());
        // d(x*y + sin(x))/dy = x = 2
        assert_close(triples[1].2, 2.0);
        // dy/dx = 0  (independent of x — exercise the all-zero short-circuit downstream)
        assert_close(triples[2].2, 0.0);
        // dy/dy = 1
        assert_close(triples[3].2, 1.0);
        // d(x + y)/dx = 1
        assert_close(triples[4].2, 1.0);
        // d(x + y)/dy = 1
        assert_close(triples[5].2, 1.0);
    }

    #[test]
    fn test_condition_batches_captured_branch_over_array_batches() {
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        // Captured `true` selects scalar_scale_branch(2.0). Pass a 3-lane batched input and
        // verify each lane is independently scaled by 2.
        let condition =
            ConditionOperation::with_captured_predicate(true, scalar_scale_branch(2.0), scalar_scale_branch(3.0))
                .unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));

        let batched_input = ArrayBatch::mapped(TestArray::vector(vec![1.0, 4.0, 9.0]), 0).unwrap();
        let outputs = operation.batch(&(), &[batched_input]).unwrap();
        assert_eq!(outputs.len(), 1);
        let output_batch = &outputs[0];
        assert_eq!(output_batch.batch_axis(), Some(0));
        assert_eq!(output_batch.value().values, vec![2.0, 8.0, 18.0]);
    }

    #[test]
    fn test_condition_batches_false_branch_when_captured_predicate_is_false() {
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        let condition =
            ConditionOperation::with_captured_predicate(false, scalar_scale_branch(2.0), scalar_scale_branch(3.0))
                .unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));

        let batched_input = ArrayBatch::mapped(TestArray::vector(vec![1.0, 4.0, 9.0]), 0).unwrap();
        let outputs = operation.batch(&(), &[batched_input]).unwrap();
        assert_eq!(outputs.len(), 1);
        let output_batch = &outputs[0];
        assert_eq!(output_batch.batch_axis(), Some(0));
        assert_eq!(output_batch.value().values, vec![3.0, 12.0, 27.0]);
    }

    #[test]
    fn test_linear_condition_batches_through_symbolic_zero_path() {
        use crate::differentiation::Tangent;
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        // Build a LinearArrayOperation::Condition with captured `true` predicate and a linear
        // scale branch. Pass an all-`Tangent::Zero` batched input and verify the symbolic-zero
        // short-circuit fires (no concrete arithmetic, output is Tangent::Zero).
        let mut builder =
            ProgramBuilder::<ArrayType, TestArray, LinearArrayOperation<TestArray, TestArray, ArrayType>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(LinearArrayOperation::Scale { factor: TestArray::scalar(5.0) }, vec![input])
            .unwrap()[0];
        let linear_branch = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let condition =
            ConditionOperation::with_captured_predicate(true, linear_branch.clone(), linear_branch).unwrap();
        let operation: LinearArrayOperation<TestArray, TestArray, ArrayType> =
            LinearArrayOperation::Condition(Box::new(condition));

        let batched_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]), None, None).unwrap();
        let zero_input =
            ArrayBatch::new(batched_type.clone(), Tangent::<ArrayType, TestArray>::zero(batched_type), Some(0))
                .unwrap();
        let outputs = <LinearArrayOperation<TestArray, TestArray, ArrayType> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
        >>::batch(&operation, &(), &[zero_input])
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].value().is_zero(), "expected symbolic-zero output from all-zero linear condition inputs");
    }

    fn linear_scalar_scale_branch(
        factor: f64,
    ) -> crate::tracing_v2::FlatProgram<TestArray, LinearArrayOperation<TestArray, TestArray, ArrayType>> {
        let mut builder =
            ProgramBuilder::<ArrayType, TestArray, LinearArrayOperation<TestArray, TestArray, ArrayType>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(LinearArrayOperation::Scale { factor: TestArray::scalar(factor) }, vec![input])
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_dot_general_evaluates_batched_matmul() {
        use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};

        // Batched matmul: [2, 2, 3] @ [2, 3, 2] -> [2, 2, 2] with axis 0 batched.
        let lhs_values: Vec<f64> = (1..=12).map(|value| value as f64).collect();
        let rhs_values: Vec<f64> = (1..=12).map(|value| value as f64).collect();
        let lhs = TestArray {
            r#type: ArrayType::new(
                DataType::F64,
                Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(3)]),
                None,
                None,
            )
            .unwrap(),
            values: lhs_values,
        };
        let rhs = TestArray {
            r#type: ArrayType::new(
                DataType::F64,
                Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(2)]),
                None,
                None,
            )
            .unwrap(),
            values: rhs_values,
        };

        let dimensions = DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]);
        let output = lhs.dot(rhs, &dimensions);

        assert_eq!(
            output.r#type,
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)]),
                None,
                None,
            )
            .unwrap(),
        );
        // Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
        // Batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]] = [[220,244],[301,334]]
        assert_eq!(output.values, vec![22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0]);
    }

    #[test]
    fn test_transpose_evaluates_general_permutation() {
        use crate::tracing_v2::operations::transpose::Transpose;

        // Rank-3 transpose with permutation [2, 0, 1]: [2, 3, 4] -> [4, 2, 3].
        let values: Vec<f64> = (0..24).map(|value| value as f64).collect();
        let input = TestArray {
            r#type: ArrayType::new(
                DataType::F64,
                Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)]),
                None,
                None,
            )
            .unwrap(),
            values,
        };

        let output = input.transpose(vec![2, 0, 1]);

        assert_eq!(
            output.r#type,
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Size::Static(4), Size::Static(2), Size::Static(3)]),
                None,
                None,
            )
            .unwrap(),
        );
        // Spot-check: input[0, 0, 0] (= 0) goes to output[0, 0, 0]; input[0, 0, 1] (= 1) -> output[1, 0, 0];
        // input[1, 2, 3] (= 23) -> output[3, 1, 2].
        assert_eq!(output.values[0], 0.0);
        assert_eq!(output.values[1 * 6], 1.0);
        let output_flat_for_23 = 3 * 6 + 1 * 3 + 2;
        assert_eq!(output.values[output_flat_for_23], 23.0);
    }

    #[test]
    fn test_jacrev_over_dot_uses_left_right_dot_batching() {
        use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};

        // jacrev internally batches cotangents of the form LeftDot/RightDot through
        // BatchableOperation::batch — exercise that path explicitly via a dot-based scalar
        // function. f(x, y) = x · y (inner product) so ∂f/∂x = y and ∂f/∂y = x.
        let jacobian = jacrev(
            &TestArrayDomain,
            |(x, y)| Ok(x.dot(y, &DotDimensionNumbers::inner_product())),
            (TestArray::vector(vec![2.0, 3.0, 5.0]), TestArray::vector(vec![7.0, 11.0, 13.0])),
        )
        .unwrap();

        let row = jacobian.rows();
        let (block_x, block_y) = row.partials();
        assert_eq!(block_x.values(), &[7.0, 11.0, 13.0]);
        assert_eq!(block_y.values(), &[2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_jacfwd_over_dot_uses_direct_batched_jvp() {
        use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};

        // jacfwd feeds all input-coordinate basis tangents through one direct JVP. A dot-product
        // scalar output exercises captured-factor linear maps instead of only elementwise tangent
        // arithmetic.
        let jacobian = TestArrayDomain
            .jacfwd(
                |(x, y)| Ok(x.dot(y, &DotDimensionNumbers::inner_product())),
                (TestArray::vector(vec![2.0, 3.0, 5.0]), TestArray::vector(vec![7.0, 11.0, 13.0])),
            )
            .unwrap();

        let row = jacobian.rows();
        let (block_x, block_y) = row.partials();
        assert_eq!(block_x.values(), &[7.0, 11.0, 13.0]);
        assert_eq!(block_y.values(), &[2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_batching_lane_varying_condition_selects_per_lane() {
        // Per-lane scalar branches: on_true scales by 2.0, on_false scales by 3.0. Operand is a
        // [4]-vector; predicate is a [4]-vector with values [1.0, 0.0, 1.0, 0.0]. Expected per-lane
        // output: [1*2, 2*3, 3*2, 4*3] = [2, 6, 6, 12].
        let condition = ConditionOperation::new(
            ArrayType::scalar(DataType::Boolean),
            scalar_scale_branch(2.0),
            scalar_scale_branch(3.0),
        )
        .unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));

        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(4)]), None, None).unwrap();
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]), None, None).unwrap();
        let predicate_batch =
            ArrayBatch::new(predicate_type, TestArray::vector(vec![1.0, 0.0, 1.0, 0.0]), Some(0)).unwrap();
        let operand_batch =
            ArrayBatch::new(operand_type, TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]), Some(0)).unwrap();

        let outputs = operation.batch(&(), &[predicate_batch, operand_batch]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![2.0, 6.0, 6.0, 12.0]);
    }

    #[test]
    fn test_broadcast_in_dim_replicates_across_added_axes() {
        use crate::tracing_v2::operations::broadcast::BroadcastInDim;

        // A length-3 vector broadcast to shape [2, 3] with broadcast_dimensions=[1]: the input
        // axis maps to output axis 1, so the value replicates across output axis 0.
        let input = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let target =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap();
        let output = input.broadcast_in_dim(target, vec![1]);
        assert_eq!(output.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_prepends_leading_axes() {
        use crate::tracing_v2::operations::broadcast::Broadcast;

        // `t.broadcast([2])` prepends a leading axis of size 2 and replicates the original
        // values across it. Matches `jax.lax.broadcast(t, [2])`.
        let input = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let output = input.broadcast(vec![2]);
        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap(),
        );
        assert_eq!(output.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_to_uses_numpy_right_alignment() {
        use crate::tracing_v2::operations::broadcast::BroadcastTo;

        // A scalar (rank-0) broadcasts to shape [2, 3] by replicating across both axes.
        let scalar = TestArray::scalar(7.0);
        let output = scalar.broadcast_to(Shape::new(vec![Size::Static(2), Size::Static(3)]));
        assert_eq!(output.values, vec![7.0; 6]);

        // A rank-1 `[3]` vector broadcasts to `[2, 3]` by right-aligning: input axis 0 maps
        // to output axis 1, replicating across output axis 0 — matches NumPy's
        // `np.broadcast_to(x, (2, 3))`.
        let vector = TestArray::vector(vec![10.0, 20.0, 30.0]);
        let output = vector.broadcast_to(Shape::new(vec![Size::Static(2), Size::Static(3)]));
        assert_eq!(output.values, vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_broadcast_like_matches_another_value_shape() {
        use crate::tracing_v2::operations::broadcast::BroadcastLike;

        // `x.broadcast_like(&like)` expands `x` to match `like`'s shape via NumPy
        // right-alignment. A length-3 vector broadcast to match a [3, 3] reference replicates
        // across the leading axis.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let like = TestArray {
            r#type: ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(3)]), None, None)
                .unwrap(),
            values: vec![0.0; 9],
        };
        let output = x.broadcast_like(&like);
        assert_eq!(output.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_select_batches_with_lane_uniform_predicate_via_broadcast() {
        // Predicate is a rank-0 lane-uniform scalar; on_true / on_false are mapped vectors of
        // size 3. With the JAX-style broadcasting elementwise rule, `apply_elementwise_batch`
        // promotes the lane-uniform predicate to the batched physical shape before invoking
        // `Select::select`, so the mixed-batching case succeeds with the expected per-lane
        // pick.
        use crate::tracing_v2::operations::select::SelectOperation;

        let pred_type = ArrayType::scalar(DataType::F64);
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]), None, None).unwrap();
        let pred_batch = ArrayBatch::new(pred_type, TestArray::scalar(1.0), None).unwrap();
        let on_true_batch =
            ArrayBatch::new(operand_type.clone(), TestArray::vector(vec![1.0, 2.0, 3.0]), Some(0)).unwrap();
        let on_false_batch = ArrayBatch::new(operand_type, TestArray::vector(vec![4.0, 5.0, 6.0]), Some(0)).unwrap();

        let outputs = SelectOperation.batch(&(), &[pred_batch, on_true_batch, on_false_batch]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batching_rule_zero_operation_is_lane_uniform() {
        // `ZeroOperation` takes no inputs and produces a constant of its captured type. The same
        // constant is the right value for every lane, so the per-op rule wraps the output as
        // lane-uniform (`batch_axis = None`) with no inserted axis.
        let scalar = ArrayType::scalar(DataType::F64);
        let operation = crate::operations::constants::ZeroOperation::new(scalar.clone());

        let outputs: Vec<ArrayBatch<TestArray>> = operation.batch(&(), &[]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].r#type().into_owned(), scalar);
        assert_eq!(outputs[0].value().values, vec![0.0]);
    }

    #[test]
    fn test_batching_rule_one_operation_is_lane_uniform() {
        // Symmetric to `ZeroOperation`: `OneOperation` is lane-uniform by construction.
        let scalar = ArrayType::scalar(DataType::F64);
        let operation = crate::operations::constants::OneOperation::new(scalar.clone());

        let outputs: Vec<ArrayBatch<TestArray>> = operation.batch(&(), &[]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].r#type().into_owned(), scalar);
        assert_eq!(outputs[0].value().values, vec![1.0]);
    }

    #[test]
    fn test_tangent_condition_runtime_predicate_lane_varying_uses_select() {
        use crate::differentiation::Tangent;
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        // Tangent + Condition with a lane-varying runtime predicate: materialize-then-dispatch
        // evaluates both branches and combines per lane via the V-level Condition rule's Select.
        // Predicate is `[1, 0, 1, 0]`; per-lane output is `[1*2, 2*3, 3*2, 4*3] = [2, 6, 6, 12]`.
        let condition = ConditionOperation::new(
            ArrayType::scalar(DataType::Boolean),
            linear_scalar_scale_branch(2.0),
            linear_scalar_scale_branch(3.0),
        )
        .unwrap();
        let operation: LinearArrayOperation<TestArray, TestArray, ArrayType> =
            LinearArrayOperation::Condition(Box::new(condition));

        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(4)]), None, None).unwrap();
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]), None, None).unwrap();
        let predicate_batch = ArrayBatch::new(
            predicate_type,
            Tangent::<ArrayType, TestArray>::Value(TestArray::vector(vec![1.0, 0.0, 1.0, 0.0])),
            Some(0),
        )
        .unwrap();
        let operand_batch = ArrayBatch::new(
            operand_type,
            Tangent::<ArrayType, TestArray>::Value(TestArray::vector(vec![1.0, 2.0, 3.0, 4.0])),
            Some(0),
        )
        .unwrap();

        let outputs = <LinearArrayOperation<TestArray, TestArray, ArrayType> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
        >>::batch(&operation, &(), &[predicate_batch, operand_batch])
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        match outputs[0].value() {
            Tangent::Value(v) => assert_eq!(v.values, vec![2.0, 6.0, 6.0, 12.0]),
            Tangent::Zero(_) => panic!("expected a Tangent::Value output from a lane-varying predicate"),
        }
    }

    #[test]
    fn test_tangent_condition_with_all_zero_tangents_materializes_correctly() {
        use crate::differentiation::Tangent;
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        // Tangent + Condition with a runtime predicate and an all-zero tangent operand:
        // materializing `Tangent::Zero` to `V::zero` and dispatching to the V-level rule produces
        // a `Tangent::Value(zero)` output rather than an `UnsupportedOperation` error. The linear
        // scale branches multiply zero by their factor, so the per-lane output is still zero.
        let condition = ConditionOperation::new(
            ArrayType::scalar(DataType::Boolean),
            linear_scalar_scale_branch(2.0),
            linear_scalar_scale_branch(3.0),
        )
        .unwrap();
        let operation: LinearArrayOperation<TestArray, TestArray, ArrayType> =
            LinearArrayOperation::Condition(Box::new(condition));

        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(4)]), None, None).unwrap();
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]), None, None).unwrap();
        let predicate_batch = ArrayBatch::new(
            predicate_type,
            Tangent::<ArrayType, TestArray>::Value(TestArray::vector(vec![1.0, 0.0, 1.0, 0.0])),
            Some(0),
        )
        .unwrap();
        let zero_operand_batch =
            ArrayBatch::new(operand_type.clone(), Tangent::<ArrayType, TestArray>::zero(operand_type), Some(0))
                .unwrap();

        let outputs = <LinearArrayOperation<TestArray, TestArray, ArrayType> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
        >>::batch(&operation, &(), &[predicate_batch, zero_operand_batch])
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        match outputs[0].value() {
            Tangent::Value(v) => assert_eq!(v.values, vec![0.0, 0.0, 0.0, 0.0]),
            Tangent::Zero(_) => {
                // The materialize-then-dispatch path wraps the V-level output as Tangent::Value,
                // even when that value happens to be all zeros. Either representation is correct
                // for downstream consumers — accept both.
            }
        }
    }

    #[test]
    fn test_jacrev_through_function_using_zero_like() {
        // `f(x) = x + zero_like(x)` is functionally the identity, but exercises the
        // `ZeroLikeOperation` rule through `jacrev`'s internal Jacobian batching path. Verifies
        // that the constant-op rule composes cleanly with reverse-mode autodiff.
        let jacobian = jacrev(&TestArrayDomain, |x| Ok(x.clone() + x.zero_like()), TestArray::scalar(2.0)).unwrap();
        let row = jacobian.rows();
        let block = row.partials();
        // d(x + 0) / dx = 1 at the scalar point.
        assert_close(block.values()[0], 1.0);
    }

    #[test]
    fn test_jacfwd_through_function_using_one_like() {
        // `f(x) = x + one_like(x)` shifts x by a constant; the Jacobian is still 1. Exercises
        // `OneLikeOperation` through jacfwd's internal batching.
        let jacobian = TestArrayDomain.jacfwd(|x| Ok(x.clone() + x.one_like()), TestArray::scalar(2.0)).unwrap();
        let row = jacobian.rows();
        let block = row.partials();
        // d(x + 1) / dx = 1.
        assert_close(block.values()[0], 1.0);
    }

    // Note on `Condition` × autodiff composition: tests that stage a `Condition` operation
    // through `ArrayOperation::Condition` inside a `jacrev` / `jacfwd` body currently fail
    // because the generic-array JVP dispatch in `ArrayOperation::jvp` ([primitive.rs:2600])
    // explicitly errors for `Condition`/`While`. The `ConditionOperation` JVP exists via the
    // separate context-aware traced-JVP impl, but
    // autodiff transforms invoke the context-less path. Closing that compose-with-autodiff
    // gap is a follow-up beyond this plan's three originally-scoped fixes; the Tangent +
    // lane-varying Condition batching rule itself (Step 2) is covered by the direct
    // `BatchableOperation::batch` tests above.

    #[test]
    fn test_array_operation_condition_interprets_captured_predicate() {
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
        let builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            TestArray,
            LinearArrayOperation<TestArray, TestArray, ArrayType, Infallible, ResidualFactor<ArrayType, TestArray>>,
        >::new()));
        let residuals = Rc::new(RefCell::new(Vec::new()));
        let residual_atoms = Rc::new(RefCell::new(HashMap::new()));
        let mut context =
            TangentContext::new_with_residuals(&TestArrayDomain, builder.clone(), residuals.clone(), residual_atoms);
        let tangent_input = context.input(ArrayType::scalar(DataType::F64));
        let outputs = condition
            .jvp(&mut context, &[JvpTracer::from_value(TestArray::scalar(4.0), tangent_input)])
            .unwrap();

        assert_eq!(outputs[0].primal().values[0], 8.0);
        let tangent_output = match outputs[0].tangent().clone() {
            crate::differentiation::Tangent::Value(tracer) => tracer.atom_id().unwrap(),
            crate::differentiation::Tangent::Zero(_) => {
                panic!("expected a concrete tangent output for the captured branch")
            }
        };
        drop(outputs);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let tangent_program =
            builder.build::<TestArray, TestArray>(vec![tangent_output], Placeholder, Placeholder).unwrap();
        let residuals = residuals.borrow();
        let tangent_program = tangent_program
            .map_operations(|operation| {
                ResidualizedOperation::<TestArrayDomain>::instantiate_residuals(operation, residuals.as_slice())
            })
            .unwrap();
        assert_eq!(tangent_program.interpret(TestArray::scalar(10.0)).map(|output| output.values[0]), Ok(20.0));
    }
}
