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
use crate::tracing_v2::{
    ArrayOperation, CoordinateValue, Differentiable, LinearArrayOperation, LinearizableDomain, MatMul, MatrixTranspose,
    Reshape,
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
    use crate::tracing::domains::Tracer;
    use crate::tracing_v2::{
        ArrayBatch, BatchableOperation, BatchingError, ConditionOperation, DifferentiableDomain,
        DifferentiableOperation, JvpContext, JvpTracer, jacrev, vmap,
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
}
