// TODO(eaplatanios): This module needs careful review.

//! Reference test value types for exercising `ryft` programs without a real backend.
//!
//! This module provides [`TestArray`], a deliberately simple dense array value backed by a flat row-major `Vec<f64>`
//! payload, together with [`TestArrayDomain`], a minimal interpreting [`Domain`] whose operation set is
//! [`ArrayOperation`]. They implement every value-level capability that [`ArrayOperation`] interpretation requires,
//! so unit tests, doctests, and downstream crates can stage, transform, and interpret programs end-to-end without
//! depending on an optimized array backend such as `ryft-ndarray`.
//!
//! These types prioritize transparency over performance: payloads are plain `f64` vectors with public fields, and
//! every operation is implemented with straightforward index arithmetic. Do not use them outside of tests and
//! documentation examples.
//!
//! This module is compiled only for `ryft-core`'s own tests and behind the `test-utilities` feature. Downstream
//! crates should enable that feature from their dev-dependencies (e.g.,
//! `ryft-core = { workspace = true, features = ["test-utilities"] }`) so that the module is available to their tests
//! without entering production builds.

use std::borrow::Cow;
use std::convert::Infallible;
use std::fmt::Display;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Sub};

use crate::broadcasting::Broadcastable;
use crate::contexts::Context;
use crate::domains::Domain;
use crate::operations::arithmetic::Scale;
use crate::operations::constants::{Fill, One, OneLike, Zero, ZeroLike};
use crate::operations::manipulation::Reshape;
use crate::operations::trigonometric::{Cos, Sin};
use crate::operations::{BooleanLike, InterpretableOperation};
use crate::parameters::Parameter;
use crate::programs::{ProgramError, Value};
use crate::tracing_v2::operations::TransferToMemory;
use crate::tracing_v2::{
    ArrayOperation, CoordinateValue, DifferentiationContext, LinearArrayOperation, RematerializationName,
};
use crate::types::{ArrayType, DataType, Shape, Size, StaticShape, Typed};
use crate::{Compare, ComparisonDirection, Select};

/// Minimal dense array value used by `ryft` tests and documentation examples. Refer to the [module
/// documentation](crate::tests) for more information.
#[derive(Clone, Debug, PartialEq)]
pub struct TestArray {
    /// Staged array type of this test value.
    pub r#type: ArrayType,

    /// Row-major payload used by tests that need concrete interpretation.
    pub values: Vec<f64>,
}

impl TestArray {
    /// Creates a test array from its staged array type and row-major payload.
    pub fn new(r#type: ArrayType, values: Vec<f64>) -> Self {
        Self { r#type, values }
    }

    /// Creates a rank-0 scalar test array.
    pub fn scalar(value: f64) -> Self {
        Self::new(ArrayType::scalar(DataType::F64), vec![value])
    }

    /// Creates a rank-1 test array.
    pub fn vector(values: Vec<f64>) -> Self {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(values.len())]));
        Self::new(r#type, values)
    }

    /// Creates a rank-2 test array.
    pub fn matrix(rows: usize, cols: usize, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), rows * cols);
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(rows), Size::Static(cols)]));
        Self::new(r#type, values)
    }

    /// Returns the staged array type of this test value.
    pub fn array_type(&self) -> &ArrayType {
        &self.r#type
    }

    /// Returns the row-major payload used by concrete test interpretation.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns the number of elements represented by `type`.
    pub fn element_count(r#type: &ArrayType) -> usize {
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

impl RematerializationName for TestArray {
    #[inline]
    fn rematerialization_name(self, _name: &str) -> Self {
        self
    }
}

impl TransferToMemory for TestArray {
    #[inline]
    fn transfer_to_memory(self, _destination: crate::types::Memory) -> Self {
        self
    }
}

impl BooleanLike for TestArray {
    /// Returns a [`TestArray`] with a Boolean-typed counterpart of this array's type and with every in-band `f64`
    /// element reinterpreted as Boolean (i.e., `0.0` maps to `0.0`/false and any nonzero element maps to `1.0`/true).
    fn as_boolean(&self) -> Self {
        Self {
            r#type: self.r#type.as_boolean(),
            values: self.values.iter().map(|value| if *value != 0.0 { 1.0 } else { 0.0 }).collect(),
        }
    }

    fn boolean(&self) -> Result<bool, ProgramError> {
        // Accept scalar Boolean predicates (rank-0, one element, encoded as 0.0=false / nonzero=true)
        // so that lane-varying while can extract a final `any(mask)` result. Higher-rank predicates
        // still error because they cannot collapse to a single Boolean.
        if self.r#type.rank() == 0 && self.r#type.data_type() == DataType::Boolean && self.values.len() == 1 {
            return Ok(self.values[0] != 0.0);
        }
        Err(ProgramError::Concretization {
            message: format!(
                "cannot extract a concrete boolean from a value of type {}; expected bool[]",
                self.r#type()
            ),
        })
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
        let stacked_type = ArrayType::new(first_type.data_type(), Shape::new(stacked_dimensions));
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

impl Scale<f64> for TestArray {
    type Output = Self;

    fn scale(self, factor: f64) -> Self::Output {
        Self { r#type: self.r#type, values: self.values.into_iter().map(|value| value * factor).collect() }
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

impl crate::operations::manipulation::Broadcast for TestArray {
    type Output = Self;

    fn broadcast(self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        let r#type = crate::operations::manipulation::Broadcast::broadcast(&self.r#type, output_type, output_axes)?;
        let input_shape = self.r#type.static_shape().unwrap();
        let target_shape = r#type.static_shape().unwrap();
        let input_rank = input_shape.rank();
        let target_rank = target_shape.rank();
        let input_strides = input_shape.row_major_strides();
        let output_count: usize = target_shape.dimensions().iter().product();
        let mut values = Vec::with_capacity(output_count);
        let mut target_index = vec![0usize; target_rank];
        while values.len() < output_count {
            let mut input_flat = 0usize;
            for input_axis in 0..input_rank {
                let target_axis = output_axes[input_axis];
                let coordinate = if input_shape[input_axis] == 1 { 0 } else { target_index[target_axis] };
                input_flat += coordinate * input_strides[input_axis];
            }
            values.push(self.values[input_flat]);
            for position in (0..target_rank).rev() {
                target_index[position] += 1;
                if target_index[position] < target_shape[position] {
                    break;
                }
                target_index[position] = 0;
            }
        }
        Ok(Self { r#type, values })
    }
}

impl crate::tracing_v2::operations::dot::Dot for TestArray {
    fn dot(self, rhs: Self, dimensions: &crate::tracing_v2::operations::dot::DotDimensionNumbers) -> Self {
        let lhs_shape = self.r#type.static_shape().unwrap();
        let rhs_shape = rhs.r#type.static_shape().unwrap();
        let (values, output_shape) = crate::tracing_v2::operations::dot::dot_general_evaluate(
            self.values.as_slice(),
            &lhs_shape,
            rhs.values.as_slice(),
            &rhs_shape,
            dimensions,
            || 0.0f64,
            |accumulator, lhs_value, rhs_value| accumulator + lhs_value * rhs_value,
        );
        let output_type = ArrayType::new(self.r#type.data_type(), Shape::from(&output_shape));
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

impl crate::operations::manipulation::Transpose for TestArray {
    fn transpose(self, permutation: Vec<usize>) -> Self {
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return self;
        }
        let shape = self.r#type.static_shape().unwrap();
        let rank = shape.rank();
        let permuted_shape = StaticShape::new(permutation.iter().map(|axis| shape[*axis]).collect());
        let input_strides = shape.row_major_strides();
        let element_count: usize = shape.dimensions().iter().product();
        let mut values = Vec::with_capacity(element_count);
        let mut permuted_index = vec![0usize; rank];
        while values.len() < element_count {
            let mut input_flat = 0usize;
            for (position, &input_axis) in permutation.iter().enumerate() {
                input_flat += permuted_index[position] * input_strides[input_axis];
            }
            values.push(self.values[input_flat]);
            for position in (0..rank).rev() {
                permuted_index[position] += 1;
                if permuted_index[position] < permuted_shape[position] {
                    break;
                }
                permuted_index[position] = 0;
            }
        }
        let output_type = ArrayType::new(self.r#type.data_type(), Shape::from(&permuted_shape));
        Self { r#type: output_type, values }
    }
}

impl Reshape for TestArray {
    type Output = Self;

    fn reshape(self, target_shape: Shape) -> Result<Self, ProgramError> {
        let output_type = ArrayType::new(self.r#type.data_type(), target_shape);
        assert_eq!(Self::element_count(&self.r#type), Self::element_count(&output_type));
        Ok(Self { r#type: output_type, values: self.values })
    }
}

impl Select for TestArray {
    type Condition = Self;

    fn select(condition: Self, on_true: Self, on_false: Self) -> Result<Self, ProgramError> {
        // Mirrors the `SelectOperation` type-inference contract: the condition must be Boolean-typed and match the
        // branch shapes.
        assert_eq!(condition.r#type.data_type(), DataType::Boolean, "select condition must have a Boolean data type",);
        assert_eq!(
            condition.r#type.shape(),
            on_true.r#type.shape(),
            "select condition and on_true must share the same shape",
        );
        assert_eq!(on_true.r#type, on_false.r#type, "select on_true and on_false must share the same type");
        let values: Vec<f64> = condition
            .values
            .iter()
            .zip(on_true.values.iter())
            .zip(on_false.values.iter())
            .map(|((condition, t), f)| if *condition != 0.0 { *t } else { *f })
            .collect();
        Ok(Self { r#type: on_true.r#type, values })
    }
}

impl Compare for TestArray {
    type Output = Self;

    fn compare(self, rhs: Self, direction: ComparisonDirection) -> Self {
        let output_shape = self.r#type.shape().clone();
        let output_len = Self::element_count(&self.r#type);
        let left = self.broadcast_values(output_len);
        let right = rhs.broadcast_values(output_len);
        let values: Vec<f64> = left
            .into_iter()
            .zip(right)
            .map(|(left, right)| {
                let predicate = match direction {
                    ComparisonDirection::Equal => left == right,
                    ComparisonDirection::NotEqual => left != right,
                    ComparisonDirection::LessThan => left < right,
                    ComparisonDirection::LessThanOrEqual => left <= right,
                    ComparisonDirection::GreaterThan => left > right,
                    ComparisonDirection::GreaterThanOrEqual => left >= right,
                };
                if predicate { 1.0 } else { 0.0 }
            })
            .collect();
        let output_type = ArrayType::new(DataType::Boolean, output_shape);
        Self { r#type: output_type, values }
    }
}

impl TestArray {
    /// Applies one elementwise binary logical operator, treating nonzero elements as logically true.
    fn binary_logical(self, rhs: Self, operator: impl Fn(bool, bool) -> bool) -> Self {
        let output_len = Self::element_count(&self.r#type);
        let left = self.broadcast_values(output_len);
        let right = rhs.broadcast_values(output_len);
        let values: Vec<f64> = left
            .into_iter()
            .zip(right)
            .map(|(left, right)| if operator(left != 0.0, right != 0.0) { 1.0 } else { 0.0 })
            .collect();
        Self { r#type: self.r#type, values }
    }
}

impl BitAnd for TestArray {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.binary_logical(rhs, |left, right| left && right)
    }
}

impl BitOr for TestArray {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.binary_logical(rhs, |left, right| left || right)
    }
}

impl BitXor for TestArray {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        self.binary_logical(rhs, |left, right| left ^ right)
    }
}

impl Not for TestArray {
    type Output = Self;

    fn not(self) -> Self::Output {
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
        let shape = self.r#type.static_shape().unwrap();
        let (reduced_values, reduced_shape) = match kind {
            ReductionKind::Sum | ReductionKind::Mean => {
                reduce_evaluate(self.values.as_slice(), &shape, axes, || 0.0, |acc, value| acc + value)
            }
            ReductionKind::Max => {
                reduce_evaluate(self.values.as_slice(), &shape, axes, || f64::NEG_INFINITY, |acc, value| acc.max(value))
            }
            ReductionKind::Min => {
                reduce_evaluate(self.values.as_slice(), &shape, axes, || f64::INFINITY, |acc, value| acc.min(value))
            }
            ReductionKind::Any => reduce_evaluate(
                self.values.as_slice(),
                &shape,
                axes,
                || 0.0,
                |acc, value| if acc != 0.0 || value != 0.0 { 1.0 } else { 0.0 },
            ),
            ReductionKind::All => reduce_evaluate(
                self.values.as_slice(),
                &shape,
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
        let data_type = self.r#type.data_type();
        let output_type = ArrayType::new(data_type, Shape::from(&reduced_shape));
        Self { r#type: output_type, values }
    }
}

/// Minimal interpreting array [`Domain`] over [`TestArray`] values and [`ArrayOperation`]s. Refer to the [module
/// documentation](crate::tests) for more information.
#[derive(Copy, Clone, Debug)]
pub struct TestArrayDomain;

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
