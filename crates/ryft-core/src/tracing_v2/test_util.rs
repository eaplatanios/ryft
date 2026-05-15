use std::borrow::Cow;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::broadcasting::Broadcastable;
use crate::operations::arithmetic::Scale;
use crate::operations::constants::{One, OneLike, Zero, ZeroLike};
use crate::operations::trigonometric::{Cos, Sin};
use crate::parameters::Parameter;
use crate::tracing::domains::{Domain, RuntimeDomain, TracingDomain};
use crate::tracing::{Traceable, TracingError, Value};
use crate::tracing_v2::operations::{ControlFlowError, ControlFlowValue};
use crate::tracing_v2::{ArrayOperation, CoordinateValue, LinearArrayOperation, LinearizableDomain, Reshape};
use crate::types::{ArrayType, DataType, Shape, Size, Typed};

/// Minimal array value used by `ryft-core` unit tests.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TestArray {
    /// Staged array type of this test value.
    r#type: ArrayType,

    /// Row-major payload used by tests that need concrete interpretation.
    values: Vec<f64>,
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
        if r#type.rank() == 0 {
            1
        } else {
            r#type.shape().dimensions().iter().map(|dimension| dimension.value().unwrap()).product()
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

impl crate::tracing_v2::batching::Batchable for TestArray {
    type CarrierValue = TestArray;

    fn batch(
        _template: &crate::tracing_v2::batching::ArrayBatch<Self>,
        value: TestArray,
    ) -> Result<crate::tracing_v2::batching::ArrayBatch<Self>, TracingError> {
        Ok(crate::tracing_v2::batching::ArrayBatch::unbatched(value))
    }
}

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

    fn stack(values: Vec<Self>) -> Result<Self, TracingError> {
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
    fn reshape(self, target_shape: Shape) -> Result<Self, TracingError> {
        let output_type = ArrayType::new(self.r#type.data_type(), target_shape, None, None).unwrap();
        assert_eq!(Self::element_count(&self.r#type), Self::element_count(&output_type));
        Ok(Self { r#type: output_type, values: self.values })
    }
}

impl crate::tracing_v2::operations::select::Select for TestArray {
    fn select(predicate: Self, on_true: Self, on_false: Self) -> Result<Self, TracingError> {
        assert_eq!(predicate.r#type, on_true.r#type, "select predicate and on_true must share the same type");
        assert_eq!(on_true.r#type, on_false.r#type, "select on_true and on_false must share the same type");
        let values: Vec<f64> = predicate
            .values
            .iter()
            .zip(on_true.values.iter())
            .zip(on_false.values.iter())
            .map(|((pred, t), f)| if *pred != 0.0 { *t } else { *f })
            .collect();
        Ok(Self { r#type: on_true.r#type.clone(), values })
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

impl LinearizableDomain for TestArrayDomain {
    type LinearDomain = TestArrayLinearDomain;

    fn linear_domain(&self) -> &Self::LinearDomain {
        &TEST_ARRAY_LINEAR_DOMAIN
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::operations::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::tracing::ProgramBuilder;
    use crate::tracing_v2::{
        ArrayBatch, BatchableOperation, BatchingError, ConditionOperation, DifferentiableDomain,
        DifferentiableOperation, JvpContext, JvpTracer, Vmap, jacrev,
    };

    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        let delta = (actual - expected).abs();
        assert!(delta <= 1e-9, "expected {actual} ~= {expected}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_array_batch_derives_logical_type_from_batch_axis() {
        let batch = ArrayBatch::mapped(TestArray::vector(vec![1.0, 2.0, 3.0]), 0).unwrap();

        assert_eq!(batch.axis_size(), Ok(Some(3)));
        assert_eq!(batch.logical_type(), Ok(ArrayType::scalar(DataType::F64)));
    }

    #[test]
    fn test_vmap_uses_one_packed_array_value() {
        let output: TestArray = TestArrayDomain
            .vmap(
                |x| Ok(x.clone() * x.clone() + x.sin()),
                TestArray::vector(vec![0.0, 1.0, 2.0]),
                Some(0),
                Some(0),
                None,
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
        let output: TestArray = TestArrayDomain
            .vmap(|x| Ok(x.clone() + x.one_like()), TestArray::vector(vec![2.0, 4.0, 6.0]), Some(0), Some(0), None)
            .unwrap();

        assert_eq!(output.values, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_vmap_maps_structured_packed_inputs_and_outputs() {
        let output: (TestArray, TestArray) = TestArrayDomain
            .vmap(
                |(left, right)| Ok((left.clone() + right.clone(), left * right)),
                (TestArray::vector(vec![1.0, 3.0]), TestArray::vector(vec![2.0, 4.0])),
                (Some(0), Some(0)),
                (Some(0), Some(0)),
                None,
            )
            .unwrap();

        assert_eq!(output.0.values, vec![3.0, 7.0]);
        assert_eq!(output.1.values, vec![2.0, 12.0]);
    }

    #[test]
    fn test_batching_rule_rejects_unaligned_batch_axes() {
        // Both square so the lane sizes agree (4), but they sit on different batch axes.
        // The per-op elementwise lift catches the axis misalignment.
        let left = ArrayBatch::mapped(TestArray::matrix(4, 4, vec![1.0; 16]), 0).unwrap();
        let right = ArrayBatch::mapped(TestArray::matrix(4, 4, vec![1.0; 16]), 1).unwrap();

        assert!(matches!(
            ArrayOperation::<TestArray, ArrayType>::Add.batch(&[left, right]),
            Err(TracingError::Batching(BatchingError::UnsupportedBatchAxisAlignment { .. })),
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
        let jacobian = jacrev::<TestArrayDomain, _, (TestArray, TestArray), (TestArray, TestArray), TestArray>(
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
            .jacfwd::<_, (TestArray, TestArray), (TestArray, TestArray), TestArray>(
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
            .hessian::<_, (TestArray, TestArray), TestArray>(
                |(x, y)| x.clone() * y + x.sin(),
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
            )
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
            .jacfwd::<_, (TestArray, TestArray), (TestArray, TestArray, TestArray), TestArray>(
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
        let outputs = operation.batch(&[batched_input]).unwrap();
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
        let outputs = operation.batch(&[batched_input]).unwrap();
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
        let mut builder = ProgramBuilder::<ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(LinearArrayOperation::Scale { factor: TestArray::scalar(5.0) }, vec![input])
            .unwrap()[0];
        let linear_branch = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let condition =
            ConditionOperation::with_captured_predicate(true, linear_branch.clone(), linear_branch).unwrap();
        let operation: LinearArrayOperation<TestArray, ArrayType> =
            LinearArrayOperation::Condition(Box::new(condition));

        let batched_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]), None, None).unwrap();
        let zero_input =
            ArrayBatch::new(batched_type.clone(), Tangent::<ArrayType, TestArray>::zero(batched_type.clone()), Some(0))
                .unwrap();
        let outputs = <LinearArrayOperation<TestArray, ArrayType> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
        >>::batch(&operation, &[zero_input])
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].value().is_zero(), "expected symbolic-zero output from all-zero linear condition inputs");
    }

    #[test]
    fn test_batched_linear_operation_short_circuits_all_zero_inputs() {
        use crate::differentiation::Tangent;
        use crate::operations::Operation;
        use crate::tracing_v2::LinearArrayOperation;
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        // Build an Add over two all-zero batched Tangent inputs and confirm the result is also
        // structurally zero — i.e., Tangent::Zero — without going through the underlying V::add.
        let batched_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]), None, None).unwrap();
        let zero_input =
            ArrayBatch::new(batched_type.clone(), Tangent::<ArrayType, TestArray>::zero(batched_type.clone()), Some(0))
                .unwrap();

        let op: LinearArrayOperation<TestArray, ArrayType> = LinearArrayOperation::Add;
        let outputs = <LinearArrayOperation<TestArray, ArrayType> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
        >>::batch(&op, &[zero_input.clone(), zero_input])
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].value().is_zero(), "expected symbolic-zero output from all-zero Add inputs");

        // Sanity-check that the same input type used through op.infer_output_types matches the
        // type reported on the symbolic-zero output.
        let expected_output_type =
            op.infer_output_types(&[batched_type.clone(), batched_type.clone()]).unwrap()[0].clone();
        assert_eq!(outputs[0].r#type().into_owned(), expected_output_type);
    }

    #[test]
    fn test_batched_linear_operation_short_circuit_uses_later_batched_input_axis() {
        use crate::differentiation::Tangent;
        use crate::tracing_v2::LinearArrayOperation;
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        let unbatched_type = ArrayType::scalar(DataType::F64);
        let unbatched_zero = ArrayBatch::unbatched(Tangent::<ArrayType, TestArray>::zero(unbatched_type));
        let batched_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]), None, None).unwrap();
        let batched_zero =
            ArrayBatch::new(batched_type.clone(), Tangent::<ArrayType, TestArray>::zero(batched_type.clone()), Some(0))
                .unwrap();

        let operation: LinearArrayOperation<TestArray, ArrayType> = LinearArrayOperation::Add;
        let outputs = <LinearArrayOperation<TestArray, ArrayType> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
        >>::batch(&operation, &[unbatched_zero, batched_zero])
        .unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].r#type().into_owned(), batched_type);
        assert!(outputs[0].value().is_zero(), "expected symbolic-zero output from all-zero Add inputs");
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

    fn linear_scalar_scale_branch(
        factor: f64,
    ) -> crate::tracing_v2::FlatProgram<TestArray, LinearArrayOperation<TestArray, ArrayType>> {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(LinearArrayOperation::Scale { factor: TestArray::scalar(factor) }, vec![input])
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_lift_elementwise_binary_op() {
        let scalar = ArrayType::scalar(DataType::F64);
        let op = ArrayOperation::<TestArray, ArrayType>::Add;
        let (lifted_op, output_axes) = crate::tracing_v2::batching::lift_elementwise(
            &op,
            &[scalar.clone(), scalar.clone()],
            &[Some(0), Some(0)],
            5,
        )
        .unwrap();

        assert!(matches!(lifted_op, ArrayOperation::Add));
        assert_eq!(output_axes, vec![Some(0)]);
    }

    #[test]
    fn test_lift_elementwise_unary_op() {
        let scalar = ArrayType::scalar(DataType::F64);
        let op = ArrayOperation::<TestArray, ArrayType>::Sin;
        let (lifted_op, output_axes) =
            crate::tracing_v2::batching::lift_elementwise(&op, &[scalar.clone()], &[Some(0)], 7).unwrap();

        assert!(matches!(lifted_op, ArrayOperation::Sin));
        assert_eq!(output_axes, vec![Some(0)]);
    }

    #[test]
    fn test_lift_elementwise_rejects_misaligned_input_axes() {
        let scalar = ArrayType::scalar(DataType::F64);
        let op = ArrayOperation::<TestArray, ArrayType>::Add;
        let err = crate::tracing_v2::batching::lift_elementwise(
            &op,
            &[scalar.clone(), scalar.clone()],
            &[Some(0), Some(1)],
            5,
        )
        .unwrap_err();
        assert!(matches!(err, TracingError::Batching(BatchingError::UnsupportedBatchAxisAlignment { .. }),));
    }

    #[test]
    fn test_lift_elementwise_passes_through_lane_uniform_inputs() {
        let scalar = ArrayType::scalar(DataType::F64);
        let op = ArrayOperation::<TestArray, ArrayType>::Add;
        let (lifted_op, output_axes) =
            crate::tracing_v2::batching::lift_elementwise(&op, &[scalar.clone(), scalar.clone()], &[Some(0), None], 5)
                .unwrap();

        assert!(matches!(lifted_op, ArrayOperation::Add));
        assert_eq!(output_axes, vec![Some(0)]);
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
    fn test_nested_vmap_squares_every_element() {
        // x has shape [3, 4]; outer vmap maps axis 0 (size 3), inner vmap maps axis 0 of the
        // per-outer-lane shape [4]. Each element should be squared.
        let x_data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let x = TestArray::matrix(3, 4, x_data.clone());

        let output: TestArray = TestArrayDomain
            .vmap(
                |row| row.context().domain().vmap(|scalar| Ok(scalar.clone() * scalar), row, Some(0), Some(0), None),
                x,
                Some(0),
                Some(0),
                None,
            )
            .unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4)]), None, None,).unwrap(),
        );
        let expected: Vec<f64> = x_data.iter().map(|value| value * value).collect();
        for (actual, expected) in output.values.iter().zip(expected.iter()) {
            assert_close(*actual, *expected);
        }
    }

    #[test]
    fn test_nested_vmap_over_dot_lifts_dimension_numbers() {
        use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};

        // x has shape [3, 4]; outer vmap over axis 0 produces per-lane rank-1 vectors. Inside,
        // we want every per-lane vector dotted with itself, giving a per-lane scalar; vmap
        // over the leading axis then yields a length-3 vector of dot products.
        let x_data: Vec<f64> = (1..=12).map(|value| value as f64).collect();
        let x = TestArray::matrix(3, 4, x_data.clone());

        let output: TestArray = TestArrayDomain
            .vmap(|row| Ok(row.clone().dot(row, &DotDimensionNumbers::inner_product())), x, Some(0), Some(0), None)
            .unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]), None, None).unwrap(),
        );
        // Lane 0: [1,2,3,4]·[1,2,3,4] = 30. Lane 1: [5,6,7,8]·[5,6,7,8] = 174. Lane 2: 446.
        for (actual, expected) in output.values.iter().zip([30.0_f64, 174.0, 446.0].iter()) {
            assert_close(*actual, *expected);
        }
    }

    #[test]
    fn test_nested_vmap_over_transpose_lifts_permutation() {
        use crate::tracing_v2::operations::transpose::Transpose;

        // x has shape [2, 3, 4]; outer vmap over axis 0 yields per-lane rank-2 matrices,
        // which we transpose. The combined effect is to permute axes 1 and 2 of the original
        // tensor, leaving the batch axis (originally axis 0) in place.
        let x_data: Vec<f64> = (0..24).map(|value| value as f64).collect();
        let x = TestArray {
            r#type: ArrayType::new(
                DataType::F64,
                Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)]),
                None,
                None,
            )
            .unwrap(),
            values: x_data,
        };

        let output: TestArray =
            TestArrayDomain.vmap(|row| Ok(row.transpose(vec![1, 0])), x, Some(0), Some(0), None).unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Size::Static(2), Size::Static(4), Size::Static(3)]),
                None,
                None,
            )
            .unwrap(),
        );
        // Spot-check: original [0, 0, 0] = 0 → output[0, 0, 0] = 0. Original [0, 0, 1] = 1 → output[0, 1, 0] = 1.
        assert_eq!(output.values[0], 0.0);
        assert_eq!(output.values[1 * 3], 1.0);
    }

    #[test]
    fn test_vmap_broadcasts_lane_uniform_input_with_in_axes_none() {
        // x is a [4]-vector mapped on axis 0 (lanes), y is a lane-uniform scalar that should be
        // added to every lane. The output should be element-wise `x + y` over the 4 lanes.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let y = TestArray::scalar(10.0);
        let output: TestArray = TestArrayDomain
            .vmap(|(left, right)| Ok(left + right), (x, y), (Some(0), None), Some(0), None)
            .unwrap();
        assert_eq!(output.values, vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_vmap_with_axis_size_validates_mapped_lane_count() {
        // With explicit axis_size = Some(4), the lane count is pinned. A mapped input of size 4
        // must agree, and the lane count flows through to subsequent operations.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let output: TestArray = TestArrayDomain.vmap(|x| Ok(x.clone() + x), x, Some(0), Some(0), Some(4)).unwrap();
        assert_eq!(output.values, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_vmap_with_out_axes_none_rejects_mapped_output() {
        // Function produces a per-lane output (mapped on axis 0), but out_axes=None requests a
        // lane-collapsed output. This is rejected because lane-collapsing reductions are not yet
        // supported.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let result: Result<TestArray, TracingError> =
            TestArrayDomain.vmap(|x| Ok(x.clone() + x), x, Some(0), None, None);
        assert!(matches!(
            result,
            Err(TracingError::Batching(BatchingError::UnbatchedOutput { message }))
                if message.contains("lane-collapsing"),
        ));
    }

    #[test]
    fn test_vmap_rejects_dynamic_batch_axis() {
        // A mapped input whose batch dimension is `Size::Dynamic` cannot be batched: vmap has no
        // way to determine the lane count.
        let dynamic_input = TestArray {
            r#type: ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)]), None, None).unwrap(),
            values: vec![1.0, 2.0, 3.0],
        };
        let result: Result<TestArray, TracingError> =
            TestArrayDomain.vmap(|x| Ok(x.clone() + x), dynamic_input, Some(0), Some(0), None);
        assert!(matches!(result, Err(TracingError::Batching(BatchingError::DynamicBatchAxis { axis: 0, .. }))));
    }

    #[test]
    fn test_vmap_with_mismatched_axis_size_rejects_mapped_input() {
        // axis_size=Some(5) conflicts with the mapped input of length 4; this should be detected.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let result: Result<TestArray, TracingError> =
            TestArrayDomain.vmap(|x| Ok(x.clone() + x), x, Some(0), Some(0), Some(5));
        assert!(matches!(result, Err(TracingError::Batching(BatchingError::MismatchedBatchSize))));
    }

    #[test]
    fn test_vmap_repositions_output_with_out_axes() {
        // Outer vmap over axis 0 of a [3, 4] matrix: each lane returns its row unchanged.
        // out_axes=Some(1) requests that the batch axis end up at position 1 of the rank-2
        // output, which forces a transpose to swap the axes.
        let x_data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let x = TestArray::matrix(3, 4, x_data.clone());
        let output: TestArray = TestArrayDomain.vmap(|row| Ok(row), x, Some(0), Some(1), None).unwrap();
        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4), Size::Static(3)]), None, None).unwrap(),
        );
        // Transpose of [3, 4]: output[i, j] = x[j, i]. Row-major flat indexing:
        // x[j, i] = x_data[j*4 + i]; output[i, j] = output_values[i*3 + j].
        for j in 0..3 {
            for i in 0..4 {
                assert_eq!(output.values[i * 3 + j], x_data[j * 4 + i]);
            }
        }
    }

    #[test]
    fn test_nested_vmap_with_mixed_in_axes_propagates_broadcast() {
        // Outer vmap over axis 0 of `x: [3, 4]` exposes a rank-1 row to the closure; inside, a
        // second inner vmap maps that row's lane axis 0 while broadcasting a captured `bias`
        // scalar to every inner lane. The combined output is x + bias broadcasted.
        let x_data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let x = TestArray::matrix(3, 4, x_data.clone());
        let bias = TestArray::scalar(0.5);

        let output: TestArray = TestArrayDomain
            .vmap(
                |(row, bias_inner)| {
                    row.context().domain().vmap(
                        |(scalar, bias_inner)| Ok(scalar + bias_inner),
                        (row, bias_inner),
                        (Some(0), None),
                        Some(0),
                        None,
                    )
                },
                (x, bias),
                (Some(0), None),
                Some(0),
                None,
            )
            .unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4)]), None, None).unwrap(),
        );
        let expected: Vec<f64> = x_data.iter().map(|value| value + 0.5).collect();
        for (actual, expected) in output.values.iter().zip(expected.iter()) {
            assert_close(*actual, *expected);
        }
    }

    #[test]
    fn test_nested_vmap_over_reshape_lifts_input_and_output_shapes() {
        use crate::tracing_v2::operations::reshape::Reshape;

        // x has shape [2, 6]; outer vmap over axis 0 yields per-lane rank-1 vectors of size 6,
        // which we reshape to per-lane [2, 3]. The combined effect should be a [2, 2, 3] tensor
        // whose leading axis is the original batch dimension.
        let x_data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let x = TestArray::matrix(2, 6, x_data.clone());

        let output: TestArray = TestArrayDomain
            .vmap(|row| row.reshape(Shape::new(vec![Size::Static(2), Size::Static(3)])), x, Some(0), Some(0), None)
            .unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(3)]),
                None,
                None,
            )
            .unwrap(),
        );
        // Row-major reshape preserves payload ordering; the lifted op only repositions strides.
        assert_eq!(output.values, x_data);
    }

    #[test]
    fn test_jacrev_over_dot_uses_left_right_dot_batching() {
        use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};

        // jacrev internally batches cotangents of the form LeftDot/RightDot through
        // BatchableOperation::batch — exercise that path explicitly via a dot-based scalar
        // function. f(x, y) = x · y (inner product) so ∂f/∂x = y and ∂f/∂y = x.
        let jacobian = jacrev::<TestArrayDomain, _, (TestArray, TestArray), TestArray, TestArray>(
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

        let outputs = operation.batch(&[predicate_batch, operand_batch]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![2.0, 6.0, 6.0, 12.0]);
    }

    #[test]
    fn test_vmap_lifts_captured_condition_at_trace_time() {
        // A captured-true Condition inside vmap: each lane scaled by 2.0. The
        // `ConditionOperation::lift` trace-time path re-traces the picked branch through a
        // fresh BatchingDomain and stages the lifted ConditionOperation directly into the
        // outer trace.
        let output: TestArray = TestArrayDomain
            .vmap(
                |x| {
                    let condition = ConditionOperation::with_captured_predicate(
                        true,
                        scalar_scale_branch(2.0),
                        scalar_scale_branch(3.0),
                    )
                    .unwrap();
                    let op = ArrayOperation::Condition(Box::new(condition));
                    let outputs = x.context().stage(op, &[&x])?;
                    Ok(outputs.into_iter().next().unwrap())
                },
                TestArray::vector(vec![1.0, 4.0, 9.0]),
                Some(0),
                Some(0),
                None,
            )
            .unwrap();
        assert_eq!(output.values, vec![2.0, 8.0, 18.0]);
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

        let outputs = SelectOperation.batch(&[pred_batch, on_true_batch, on_false_batch]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vmap_lifts_lane_varying_condition_via_select() {
        // A runtime-predicate Condition inside vmap with a lane-varying predicate: each lane
        // independently chooses between `on_true` (scale by 2.0) and `on_false` (scale by 3.0).
        // The trace-time `BatchingDomain::stage` dispatches the rule's `batch`, whose lane-varying
        // branch evaluates both branches over the operand axes and combines per lane via
        // `Select`. Multi-op staging emerges automatically through `Tracer`'s value-level traits.
        let predicate = TestArray::vector(vec![1.0, 0.0, 1.0, 0.0]);
        let operand = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);

        let output: TestArray = TestArrayDomain
            .vmap(
                |(pred, operand)| {
                    let condition = ConditionOperation::new(
                        ArrayType::scalar(DataType::Boolean),
                        scalar_scale_branch(2.0),
                        scalar_scale_branch(3.0),
                    )
                    .unwrap();
                    let op = ArrayOperation::Condition(Box::new(condition));
                    let outputs = pred.context().stage(op, &[&pred, &operand])?;
                    Ok(outputs.into_iter().next().unwrap())
                },
                (predicate, operand),
                (Some(0), Some(0)),
                Some(0),
                None,
            )
            .unwrap();
        // Expected per-lane: [1*2, 2*3, 3*2, 4*3] = [2, 6, 6, 12].
        assert_eq!(output.values, vec![2.0, 6.0, 6.0, 12.0]);
    }

    #[test]
    fn test_batching_rule_zero_operation_is_lane_uniform() {
        // `ZeroOperation` takes no inputs and produces a constant of its captured type. The same
        // constant is the right value for every lane, so the per-op rule wraps the output as
        // lane-uniform (`batch_axis = None`) with no inserted axis.
        let scalar = ArrayType::scalar(DataType::F64);
        let operation = crate::operations::constants::ZeroOperation::new(scalar.clone());

        let outputs: Vec<ArrayBatch<TestArray>> = operation.batch(&[]).unwrap();
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

        let outputs: Vec<ArrayBatch<TestArray>> = operation.batch(&[]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].r#type().into_owned(), scalar);
        assert_eq!(outputs[0].value().values, vec![1.0]);
    }

    #[test]
    fn test_vmap_over_zero_operation_yields_lane_uniform_output() {
        // End-to-end: a vmap'd function that stages `ZeroOperation` produces a lane-uniform zero
        // value at the per-lane scalar type. Verifies that the trace-time stage hook accepts a
        // zero-input operation and that the post-trace replay materializes the same zero for
        // every lane through the lane-uniform broadcast path.
        let output: TestArray = TestArrayDomain
            .vmap(
                |x| {
                    let zero_op = ArrayOperation::<TestArray, ArrayType>::Zero(
                        crate::operations::constants::ZeroOperation::new(ArrayType::scalar(DataType::F64)),
                    );
                    let zero = x.context().stage(zero_op, &[])?.into_iter().next().unwrap();
                    Ok(x + zero)
                },
                TestArray::vector(vec![1.0, 2.0, 3.0]),
                Some(0),
                Some(0),
                None,
            )
            .unwrap();

        assert_eq!(output.values, vec![1.0, 2.0, 3.0]);
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
        let operation: LinearArrayOperation<TestArray, ArrayType> =
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

        let outputs = <LinearArrayOperation<TestArray, ArrayType> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
        >>::batch(&operation, &[predicate_batch, operand_batch])
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
        // a `Tangent::Value(zero)` output rather than a `MissingBatchingRule` error. The linear
        // scale branches multiply zero by their factor, so the per-lane output is still zero.
        let condition = ConditionOperation::new(
            ArrayType::scalar(DataType::Boolean),
            linear_scalar_scale_branch(2.0),
            linear_scalar_scale_branch(3.0),
        )
        .unwrap();
        let operation: LinearArrayOperation<TestArray, ArrayType> =
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

        let outputs = <LinearArrayOperation<TestArray, ArrayType> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
        >>::batch(&operation, &[predicate_batch, zero_operand_batch])
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
        let jacobian = jacrev::<TestArrayDomain, _, TestArray, TestArray, TestArray>(
            &TestArrayDomain,
            |x| Ok(x.clone() + x.zero_like()),
            TestArray::scalar(2.0),
        )
        .unwrap();
        let row = jacobian.rows();
        let block = row.partials();
        // d(x + 0) / dx = 1 at the scalar point.
        assert_close(block.values()[0], 1.0);
    }

    #[test]
    fn test_jacfwd_through_function_using_one_like() {
        // `f(x) = x + one_like(x)` shifts x by a constant; the Jacobian is still 1. Exercises
        // `OneLikeOperation` through jacfwd's internal batching.
        let jacobian = TestArrayDomain
            .jacfwd::<_, TestArray, TestArray, TestArray>(|x| Ok(x.clone() + x.one_like()), TestArray::scalar(2.0))
            .unwrap();
        let row = jacobian.rows();
        let block = row.partials();
        // d(x + 1) / dx = 1.
        assert_close(block.values()[0], 1.0);
    }

    // Note on `Condition` × autodiff composition: tests that stage a `Condition` operation
    // through `ArrayOperation::Condition` inside a `jacrev` / `jacfwd` body currently fail
    // because the generic-array JVP dispatch in `ArrayOperation::jvp` ([primitive.rs:2600])
    // explicitly errors for `Condition`/`While`. The `ConditionOperation` JVP exists via the
    // separate context-aware `DifferentiableOperation<TracingContext<...>>` impl, but
    // autodiff transforms invoke the context-less path. Closing that compose-with-autodiff
    // gap is a follow-up beyond this plan's three originally-scoped fixes; the Tangent +
    // lane-varying Condition batching rule itself (Step 2) is covered by the direct
    // `BatchableOperation::batch` tests above.

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
        let tangent_input = context.linear_context().input(ArrayType::scalar(DataType::F64));
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
        assert_eq!(tangent_program.interpret(TestArray::scalar(10.0)).map(|output| output.values[0]), Ok(20.0));
    }
}
