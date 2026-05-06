use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

use ndarray::{Array2, ArrayD, Ix2, IxDyn, Zip};
use thiserror::Error;

use ryft_core::operations::constants::{One, Zero};
use ryft_core::operations::constants::{OneLike, ZeroLike};
use ryft_core::parameters::Parameter;
use ryft_core::tracing::{Traceable, TracingError, Value};
use ryft_core::tracing_v2::operations::{ControlFlowError, ControlFlowValue};
use ryft_core::tracing_v2::{CoordinateValue, Cos, Differentiable, DifferentiationError, MatrixOps, ReshapeOps, Sin};
use ryft_core::types::{ArrayType, DataType, Shape, Size, TypeError, Typed};

/// Element type supported by the `ryft-ndarray` backend.
pub trait NdArrayElement: Copy + Clone + Debug + Display + PartialEq + 'static {
    /// `ryft-core` data type corresponding to this element.
    const DATA_TYPE: DataType;

    /// Additive identity.
    fn zero() -> Self;

    /// Multiplicative identity.
    fn one() -> Self;

    /// Adds two element values.
    fn add(left: Self, right: Self) -> Self;

    /// Subtracts two element values.
    fn subtract(left: Self, right: Self) -> Self;

    /// Multiplies two element values.
    fn multiply(left: Self, right: Self) -> Self;

    /// Divides two element values.
    fn divide(left: Self, right: Self) -> Self;

    /// Negates one element value.
    fn negate(value: Self) -> Self;

    /// Computes the sine of one element value.
    fn sin(value: Self) -> Self;

    /// Computes the cosine of one element value.
    fn cos(value: Self) -> Self;
}

impl NdArrayElement for f32 {
    const DATA_TYPE: DataType = DataType::F32;

    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn add(left: Self, right: Self) -> Self {
        left + right
    }

    #[inline]
    fn subtract(left: Self, right: Self) -> Self {
        left - right
    }

    #[inline]
    fn multiply(left: Self, right: Self) -> Self {
        left * right
    }

    #[inline]
    fn divide(left: Self, right: Self) -> Self {
        left / right
    }

    #[inline]
    fn negate(value: Self) -> Self {
        -value
    }

    #[inline]
    fn sin(value: Self) -> Self {
        f32::sin(value)
    }

    #[inline]
    fn cos(value: Self) -> Self {
        f32::cos(value)
    }
}

impl NdArrayElement for f64 {
    const DATA_TYPE: DataType = DataType::F64;

    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn add(left: Self, right: Self) -> Self {
        left + right
    }

    #[inline]
    fn subtract(left: Self, right: Self) -> Self {
        left - right
    }

    #[inline]
    fn multiply(left: Self, right: Self) -> Self {
        left * right
    }

    #[inline]
    fn divide(left: Self, right: Self) -> Self {
        left / right
    }

    #[inline]
    fn negate(value: Self) -> Self {
        -value
    }

    #[inline]
    fn sin(value: Self) -> Self {
        f64::sin(value)
    }

    #[inline]
    fn cos(value: Self) -> Self {
        f64::cos(value)
    }
}

/// Error returned while constructing or reshaping [`Array`] values.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ArrayError {
    /// The requested [`ArrayType`] uses an element type that does not match this Rust element type.
    #[error("ndarray backend expected element type {expected} but got {actual}")]
    ElementTypeMismatch { expected: DataType, actual: DataType },

    /// The requested [`ArrayType`] carries sharding metadata, which this CPU backend does not support.
    #[error("ndarray backend does not support sharded array types")]
    ShardedArrayType,

    /// The requested [`ArrayType`] carries physical layout metadata, which this backend does not model.
    #[error("ndarray backend does not support array types with explicit layouts")]
    ExplicitLayout,

    /// The requested [`ArrayType`] contains a dynamic dimension.
    #[error("ndarray backend requires static shape dimensions, but dimension #{dimension} is {size}")]
    DynamicShapeDimension { dimension: usize, size: Size },

    /// The provided dense values do not match the requested shape.
    #[error("{message}")]
    Shape { message: String },

    /// The provided operands cannot be broadcast together.
    #[error("array shapes {left:?} and {right:?} are not broadcast-compatible")]
    IncompatibleBroadcast { left: Vec<usize>, right: Vec<usize> },
}

/// Owned CPU array value backed by `ndarray::ArrayD`.
#[derive(Clone, Debug, PartialEq)]
pub struct Array<T = f64> {
    /// Dense row-major array storage.
    values: ArrayD<T>,
}

impl<T> Array<T> {
    /// Wraps an owned `ndarray::ArrayD`.
    #[inline]
    pub fn new(values: ArrayD<T>) -> Self {
        Self { values }
    }

    /// Returns the underlying `ndarray` value by reference.
    #[inline]
    pub fn as_ndarray(&self) -> &ArrayD<T> {
        &self.values
    }

    /// Consumes this wrapper and returns the owned `ndarray` value.
    #[inline]
    pub fn into_ndarray(self) -> ArrayD<T> {
        self.values
    }
}

impl<T: NdArrayElement> Array<T> {
    /// Constructs an array from one static shape and row-major values.
    pub fn from_shape_vec<S: Into<Vec<usize>>>(shape: S, values: Vec<T>) -> Result<Self, ArrayError> {
        let shape = shape.into();
        ArrayD::from_shape_vec(IxDyn(shape.as_slice()), values)
            .map(Self::new)
            .map_err(|_| ArrayError::Shape { message: "values do not match the requested array shape".to_string() })
    }

    /// Constructs a scalar rank-0 array.
    #[inline]
    pub fn scalar(value: T) -> Self {
        Self::new(ArrayD::from_elem(IxDyn(&[]), value))
    }

    /// Constructs an array with every element set to `value`.
    pub fn full(array_type: &ArrayType, value: T) -> Result<Self, ArrayError> {
        let shape = validate_array_type::<T>(array_type)?;
        Ok(Self::new(ArrayD::from_elem(IxDyn(shape.as_slice()), value)))
    }

    /// Constructs a zero-filled array matching `array_type`.
    #[inline]
    pub fn zeros(array_type: &ArrayType) -> Result<Self, ArrayError> {
        Self::full(array_type, T::zero())
    }

    /// Constructs a one-filled array matching `array_type`.
    #[inline]
    pub fn ones(array_type: &ArrayType) -> Result<Self, ArrayError> {
        Self::full(array_type, T::one())
    }

    /// Returns the dense shape of this array.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.values.shape()
    }
}

impl<T: NdArrayElement> Parameter for Array<T> {}

impl<T: NdArrayElement> Display for Array<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.values, formatter)
    }
}

impl<T: NdArrayElement> Typed<ArrayType> for Array<T> {
    fn r#type(&self) -> Cow<'_, ArrayType> {
        let shape = Shape::new(self.values.shape().iter().copied().map(Size::Static).collect::<Vec<_>>());
        Cow::Owned(
            ArrayType::new(T::DATA_TYPE, shape, None, None)
                .expect("unsharded ndarray values should always produce valid array types"),
        )
    }
}

impl<T: NdArrayElement> Traceable<ArrayType> for Array<T> {}

impl<T: NdArrayElement> Value<ArrayType> for Array<T> {}

impl<T: NdArrayElement> ControlFlowValue for Array<T> {
    #[inline]
    fn control_flow_predicate(&self) -> Result<bool, TracingError> {
        Err(ControlFlowError::InvalidPredicateValue { type_: self.r#type().into_owned() }.into())
    }
}

impl<T: NdArrayElement> ZeroLike for Array<T> {
    #[inline]
    fn zero_like(&self) -> Self {
        Self::new(ArrayD::from_elem(self.values.raw_dim(), T::zero()))
    }
}

impl<T: NdArrayElement> OneLike for Array<T> {
    #[inline]
    fn one_like(&self) -> Self {
        Self::new(ArrayD::from_elem(self.values.raw_dim(), T::one()))
    }
}

impl<T: NdArrayElement> Zero<ArrayType> for Array<T> {
    #[inline]
    fn zero(array_type: &ArrayType) -> Result<Self, TracingError> {
        Array::zeros(array_type).map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

impl<T: NdArrayElement> One<ArrayType> for Array<T> {
    #[inline]
    fn one(array_type: &ArrayType) -> Result<Self, TracingError> {
        if array_type.rank() != 0 {
            return Err(DifferentiationError::NonScalarGradientOutput { output_type: array_type.clone() }.into());
        }
        Array::ones(array_type).map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

impl<T: NdArrayElement> Differentiable<ArrayType> for Array<T> {
    type Tangent = Self;

    #[inline]
    fn tangent_type(&self) -> Result<Self::Tangent, TracingError> {
        Self::zero(self.r#type().as_ref())
    }
}

impl<T: NdArrayElement> CoordinateValue for Array<T> {
    type Coordinate = T;

    #[inline]
    fn coordinate_count(&self) -> usize {
        self.values.len()
    }

    fn coordinate_basis(&self) -> Vec<Self> {
        let shape = self.values.shape().to_vec();
        let mut basis = Vec::with_capacity(self.values.len());
        for basis_index in 0..self.values.len() {
            let mut values = vec![T::zero(); self.values.len()];
            values[basis_index] = T::one();
            basis.push(
                ArrayD::from_shape_vec(IxDyn(shape.as_slice()), values)
                    .map(Self::new)
                    .expect("coordinate basis shape should match value count"),
            );
        }
        basis
    }

    #[inline]
    fn coordinates(&self) -> Vec<Self::Coordinate> {
        self.values.iter().copied().collect::<Vec<_>>()
    }
}

impl<T: NdArrayElement> Add for Array<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        binary_elementwise(self, rhs, T::add)
    }
}

impl<T: NdArrayElement> Sub for Array<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        binary_elementwise(self, rhs, T::subtract)
    }
}

impl<T: NdArrayElement> Mul for Array<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        binary_elementwise(self, rhs, T::multiply)
    }
}

impl<T: NdArrayElement> Div for Array<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        binary_elementwise(self, rhs, T::divide)
    }
}

impl<T: NdArrayElement> Neg for Array<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(self.values.mapv(T::negate))
    }
}

impl<T: NdArrayElement> Sin for Array<T> {
    #[inline]
    fn sin(self) -> Self {
        Self::new(self.values.mapv(T::sin))
    }
}

impl<T: NdArrayElement> Cos for Array<T> {
    #[inline]
    fn cos(self) -> Self {
        Self::new(self.values.mapv(T::cos))
    }
}

impl<T: NdArrayElement> MatrixOps for Array<T> {
    fn matmul(self, rhs: Self) -> Self {
        let lhs_shape = self.values.shape().to_vec();
        let rhs_shape = rhs.values.shape().to_vec();
        let lhs = self
            .values
            .into_dimensionality::<Ix2>()
            .unwrap_or_else(|_| panic!("matmul expected a rank-2 left operand but got shape {lhs_shape:?}"));
        let rhs = rhs
            .values
            .into_dimensionality::<Ix2>()
            .unwrap_or_else(|_| panic!("matmul expected a rank-2 right operand but got shape {rhs_shape:?}"));
        let (rows, inner) = lhs.dim();
        let (rhs_inner, cols) = rhs.dim();
        assert!(inner == rhs_inner, "matmul expected compatible inner dimensions but got {inner} and {rhs_inner}");
        let mut result = Array2::from_elem((rows, cols), T::zero());
        for row in 0..rows {
            for col in 0..cols {
                let mut value = T::zero();
                for index in 0..inner {
                    value = T::add(value, T::multiply(lhs[(row, index)], rhs[(index, col)]));
                }
                result[(row, col)] = value;
            }
        }
        Self::new(result.into_dyn())
    }

    fn transpose_matrix(self) -> Self {
        let shape = self.values.shape().to_vec();
        let matrix = self
            .values
            .into_dimensionality::<Ix2>()
            .unwrap_or_else(|_| panic!("matrix transpose expected a rank-2 operand but got shape {shape:?}"));
        if matrix.nrows() == 1 && matrix.ncols() == 1 {
            return Self::new(matrix.into_dyn());
        }
        Self::new(matrix.reversed_axes().into_dyn())
    }
}

impl<T: NdArrayElement> ReshapeOps for Array<T> {
    fn reshape(self, target_shape: Shape) -> Result<Self, TracingError> {
        let input_type = self.r#type().into_owned();
        let output_type =
            ryft_core::tracing_v2::operations::reshape::reshape_abstract(&input_type, &target_shape, "reshape")?;
        if input_type == output_type {
            return Ok(self);
        }
        let output_shape = static_shape(&output_type.shape).map_err(array_error_to_tracing_error)?;
        let values = self.values.into_iter().collect::<Vec<_>>();
        ArrayD::from_shape_vec(IxDyn(output_shape.as_slice()), values).map(Self::new).map_err(|_| {
            array_error_to_tracing_error(ArrayError::Shape {
                message: "reshape could not realize the requested array shape".to_string(),
            })
        })
    }
}

fn validate_array_type<T: NdArrayElement>(array_type: &ArrayType) -> Result<Vec<usize>, ArrayError> {
    if array_type.data_type != T::DATA_TYPE {
        return Err(ArrayError::ElementTypeMismatch { expected: T::DATA_TYPE, actual: array_type.data_type });
    }
    if array_type.layout.is_some() {
        return Err(ArrayError::ExplicitLayout);
    }
    if array_type.sharding.is_some() {
        return Err(ArrayError::ShardedArrayType);
    }
    static_shape(&array_type.shape)
}

fn static_shape(shape: &Shape) -> Result<Vec<usize>, ArrayError> {
    shape
        .dimensions
        .iter()
        .enumerate()
        .map(|(dimension, size)| match size {
            Size::Static(size) => Ok(*size),
            Size::Dynamic(_) => Err(ArrayError::DynamicShapeDimension { dimension, size: *size }),
        })
        .collect::<Result<Vec<_>, _>>()
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, ArrayError> {
    let rank = left.len().max(right.len());
    let mut shape = Vec::with_capacity(rank);
    for axis_from_end in 0..rank {
        let left_dimension = if axis_from_end < left.len() { left[left.len() - 1 - axis_from_end] } else { 1 };
        let right_dimension = if axis_from_end < right.len() { right[right.len() - 1 - axis_from_end] } else { 1 };
        if left_dimension == right_dimension || left_dimension == 1 || right_dimension == 1 {
            shape.push(left_dimension.max(right_dimension));
        } else {
            return Err(ArrayError::IncompatibleBroadcast { left: left.to_vec(), right: right.to_vec() });
        }
    }
    shape.reverse();
    Ok(shape)
}

fn binary_elementwise<T: NdArrayElement>(left: Array<T>, right: Array<T>, operation: fn(T, T) -> T) -> Array<T> {
    let output_shape = broadcast_shape(left.values.shape(), right.values.shape())
        .expect("array operands should be broadcast-compatible");
    let left_values = left
        .values
        .broadcast(IxDyn(output_shape.as_slice()))
        .expect("validated broadcast shape should be valid for the left operand");
    let right_values = right
        .values
        .broadcast(IxDyn(output_shape.as_slice()))
        .expect("validated broadcast shape should be valid for the right operand");
    Array::new(Zip::from(left_values).and(right_values).map_collect(|&left, &right| operation(left, right)))
}

fn array_error_to_tracing_error(error: ArrayError) -> TracingError {
    TypeError { message: error.to_string() }.into()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use ndarray::{arr0, arr1, arr2};
    use pretty_assertions::assert_eq;
    use ryft_core::parameters::Placeholder;
    use ryft_core::tracing::ProgramBuilder;
    use ryft_core::tracing::engines::TracingEngine;
    use ryft_core::tracing::transposition::{LinearOperation, TranspositionContext};
    use ryft_core::tracing_v2::operations::{ControlFlowValue, ReshapeOperation};
    use ryft_core::tracing_v2::{MatrixOps, ReshapeOps};
    use ryft_core::types::{ArrayType, DataType, Shape, Size, Typed};

    use crate::{LinearNdarrayOperation, NdArrayEngine};

    use super::Array;

    #[test]
    fn test_array_reports_static_unsharded_type() {
        let array = Array::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let expected_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap();

        assert_eq!(array.r#type().into_owned(), expected_type);
        assert_eq!(Array::scalar(2.0).as_ndarray(), &arr0(2.0).into_dyn());
    }

    #[test]
    fn test_array_display_delegates_to_ndarray_values() {
        assert_eq!(Array::scalar(2.0).to_string(), "2");
        assert_eq!(Array::from_shape_vec([2], vec![1.0, 2.0]).unwrap().to_string(), "[1, 2]");
    }

    #[test]
    fn test_array_control_flow_predicate_reports_invalid_type() {
        let array = Array::from_shape_vec([2], vec![1.0, 2.0]).unwrap();

        assert_eq!(
            array.control_flow_predicate().unwrap_err().to_string(),
            "control-flow predicate value has type f64[2], but expected bool[]"
        );
    }

    #[test]
    fn test_elementwise_operations_broadcast() {
        let left = Array::from_shape_vec([2, 1], vec![1.0, 2.0]).unwrap();
        let right = Array::from_shape_vec([1, 3], vec![10.0, 20.0, 30.0]).unwrap();

        let sum = left.clone() + right.clone();
        let difference = left.clone() - right.clone();
        let product = left.clone() * right.clone();
        let quotient = right / left;

        assert_eq!(sum.as_ndarray(), &arr2(&[[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]).into_dyn());
        assert_eq!(difference.as_ndarray(), &arr2(&[[-9.0, -19.0, -29.0], [-8.0, -18.0, -28.0]]).into_dyn());
        assert_eq!(product.as_ndarray(), &arr2(&[[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]]).into_dyn());
        assert_eq!(quotient.as_ndarray(), &arr2(&[[10.0, 20.0, 30.0], [5.0, 10.0, 15.0]]).into_dyn());
    }

    #[test]
    fn test_reshape_preserves_row_major_order() {
        let array = Array::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let reshaped = array.reshape(Shape::new(vec![Size::Static(3), Size::Static(2)])).unwrap();

        assert_eq!(reshaped.as_ndarray(), &arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).into_dyn());
    }

    #[test]
    fn test_reshape_jit_rendering_includes_target_shape() {
        let input = Array::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let engine = NdArrayEngine::<f64>::new();
        let (_, compiled): (Array<f64>, _) = engine
            .interpret_and_trace(|x| x.reshape(Shape::new(vec![Size::Static(1), Size::Static(4)])), input)
            .unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc::indoc! {"
                lambda %0:f64[2, 2] .
                let %1:f64[1, 4] = reshape [input_shape=[2, 2], output_shape=[1, 4]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reshape_transpose_restores_the_input_shape() {
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)]), None, None).unwrap();
        let output_value = Array::from_shape_vec([1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let transpose_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, Array<f64>, LinearNdarrayOperation<Array<f64>>>::new()));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(output_value.r#type().into_owned());
        let mut context = TranspositionContext::new(transpose_builder.clone());
        let contribution_atom =
            ReshapeOperation::new(input_type.shape.clone(), Shape::new(vec![Size::Static(1), Size::Static(4)]))
                .transpose(&mut context, &[Some(output_cotangent_atom)])
                .unwrap()
                .into_iter()
                .next()
                .expect("transpose should return one contribution")
                .expect("transpose should produce one cotangent contribution");
        drop(context);

        let transpose_builder = Rc::try_unwrap(transpose_builder)
            .expect("transpose builder should not have outstanding linear terms")
            .into_inner();
        let transpose_program = transpose_builder
            .build::<Array<f64>, Array<f64>>(vec![contribution_atom], Placeholder, Placeholder)
            .unwrap();
        assert_eq!(
            transpose_program
                .interpret(Array::from_shape_vec([1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap())
                .unwrap()
                .as_ndarray(),
            &arr2(&[[1.0f64, 2.0], [3.0, 4.0]]).into_dyn()
        );
    }

    #[test]
    fn test_matrix_operations() {
        let left = Array::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let right = Array::from_shape_vec([3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let transposed = right.clone().transpose_matrix();

        assert_eq!(left.matmul(right).as_ndarray(), &arr2(&[[58.0, 64.0], [139.0, 154.0]]).into_dyn());
        assert_eq!(transposed.as_ndarray(), &arr2(&[[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]).into_dyn());
    }

    #[test]
    fn test_coordinate_basis_uses_flat_row_major_order() {
        let array = Array::from_shape_vec([3], vec![2.0, 4.0, 6.0]).unwrap();
        let basis = ryft_core::tracing_v2::CoordinateValue::coordinate_basis(&array);

        assert_eq!(ryft_core::tracing_v2::CoordinateValue::coordinates(&array), vec![2.0, 4.0, 6.0]);
        assert_eq!(basis[0].as_ndarray(), &arr1(&[1.0, 0.0, 0.0]).into_dyn());
        assert_eq!(basis[1].as_ndarray(), &arr1(&[0.0, 1.0, 0.0]).into_dyn());
        assert_eq!(basis[2].as_ndarray(), &arr1(&[0.0, 0.0, 1.0]).into_dyn());
    }
}
