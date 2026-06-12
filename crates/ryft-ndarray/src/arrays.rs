use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Sub};

use ndarray::{ArrayD, IxDyn, Zip};
use thiserror::Error;

use ryft_core::operations::BooleanLike;
use ryft_core::operations::arithmetic::Scale;
use ryft_core::operations::constants::{Fill, One, OneLike, Zero, ZeroLike};
use ryft_core::operations::control_flow::Select;
use ryft_core::operations::manipulation::{Broadcast, Reshape, Transpose};
use ryft_core::parameters::Parameter;
use ryft_core::programs::{ProgramError, Value};
use ryft_core::tracing_v2::operations::dot::{Dot, DotDimensionNumbers, LeftDot, RightDot, dot_general_evaluate};
use ryft_core::tracing_v2::{CoordinateValue, Cos, Sin};
use ryft_core::types::{ArrayType, DataType, Shape, Size, StaticShape, TypeError, Typed};

/// Element type supported by the `ryft-ndarray` backend.
pub trait NdArrayElement: Copy + Clone + Debug + Display + PartialEq + PartialOrd + 'static {
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

    /// Returns the smallest representable value of this element type. Used as the identity for
    /// max reductions.
    fn min_value() -> Self;

    /// Returns the largest representable value of this element type. Used as the identity for
    /// min reductions.
    fn max_value() -> Self;

    /// Returns the maximum of two element values. Used by max reductions.
    fn max(left: Self, right: Self) -> Self;

    /// Returns the minimum of two element values. Used by min reductions.
    fn min(left: Self, right: Self) -> Self;

    /// Multiplies this element by a runtime `f64` constant. Used by transform rules that need to
    /// materialize a numeric factor without being parameterized over the element type
    /// (`Mean` transpose, `PMean` batching).
    fn scale_by_constant(self, factor: f64) -> Self;
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

    #[inline]
    fn min_value() -> Self {
        f32::NEG_INFINITY
    }

    #[inline]
    fn max_value() -> Self {
        f32::INFINITY
    }

    #[inline]
    fn max(left: Self, right: Self) -> Self {
        f32::max(left, right)
    }

    #[inline]
    fn min(left: Self, right: Self) -> Self {
        f32::min(left, right)
    }

    #[inline]
    fn scale_by_constant(self, factor: f64) -> Self {
        ((self as f64) * factor) as f32
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

    #[inline]
    fn min_value() -> Self {
        f64::NEG_INFINITY
    }

    #[inline]
    fn max_value() -> Self {
        f64::INFINITY
    }

    #[inline]
    fn max(left: Self, right: Self) -> Self {
        f64::max(left, right)
    }

    #[inline]
    fn min(left: Self, right: Self) -> Self {
        f64::min(left, right)
    }

    #[inline]
    fn scale_by_constant(self, factor: f64) -> Self {
        self * factor
    }
}

impl NdArrayElement for bool {
    const DATA_TYPE: DataType = DataType::Boolean;

    #[inline]
    fn zero() -> Self {
        false
    }

    #[inline]
    fn one() -> Self {
        true
    }

    #[inline]
    fn add(left: Self, right: Self) -> Self {
        // Boolean addition has no canonical numeric meaning; treat as logical OR (matches the
        // identity used by `Any` reductions and most usages in mask combinators).
        left || right
    }

    #[inline]
    fn subtract(left: Self, right: Self) -> Self {
        // Boolean subtraction has no canonical numeric meaning; treat as `left AND NOT right`
        // (set difference), consistent with treating `add` as logical OR.
        left && !right
    }

    #[inline]
    fn multiply(left: Self, right: Self) -> Self {
        // Boolean multiplication is logical AND.
        left && right
    }

    #[inline]
    fn divide(_left: Self, _right: Self) -> Self {
        panic!("bool division is not defined")
    }

    #[inline]
    fn negate(value: Self) -> Self {
        !value
    }

    #[inline]
    fn sin(_value: Self) -> Self {
        panic!("bool sin is not defined")
    }

    #[inline]
    fn cos(_value: Self) -> Self {
        panic!("bool cos is not defined")
    }

    #[inline]
    fn min_value() -> Self {
        true
    }

    #[inline]
    fn max_value() -> Self {
        false
    }

    #[inline]
    fn max(left: Self, right: Self) -> Self {
        // Max under the natural false<true ordering is logical OR.
        left || right
    }

    #[inline]
    fn min(left: Self, right: Self) -> Self {
        // Min under the natural false<true ordering is logical AND.
        left && right
    }

    #[inline]
    fn scale_by_constant(self, factor: f64) -> Self {
        // Treat bool as 0/1 under f64 multiplication and reinterpret the result as nonzero/zero.
        ((self as u8 as f64) * factor) != 0.0
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
        Cow::Owned(ArrayType::new(T::DATA_TYPE, shape))
    }
}

impl<T: NdArrayElement> Value<ArrayType> for Array<T> {}

impl<T: NdArrayElement> ryft_core::tracing_v2::RematerializationName for Array<T> {
    #[inline]
    fn rematerialization_name(self, _name: &str) -> Self {
        self
    }
}

impl<T: NdArrayElement> ryft_core::tracing_v2::operations::TransferToMemory for Array<T> {
    #[inline]
    fn transfer_to_memory(self, _destination: ryft_core::types::Memory) -> Self {
        self
    }
}

impl<T: NdArrayElement> BooleanLike for Array<T> {
    /// Returns an [`Array`] with every element reinterpreted as an in-band Boolean (i.e., `T::zero()` maps to
    /// `T::zero()`/false and any nonzero element maps to `T::one()`/true). The element type `T` is fixed by the
    /// array's Rust type, so the reinterpretation keeps the in-band encoding instead of switching to a dedicated
    /// Boolean element type.
    fn as_boolean(&self) -> Self {
        Self::new(self.values.mapv(|value| if value != T::zero() { T::one() } else { T::zero() }))
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        // `Array` reports its element data type from `T`, which is never `DataType::Boolean`, so it can never carry
        // the scalar Boolean predicate expected by control-flow operations.
        Err(ProgramError::Concretization {
            message: format!(
                "cannot extract a concrete boolean from a value of type {}; expected bool[]",
                self.r#type()
            ),
        })
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
    fn zero(array_type: &ArrayType) -> Result<Self, ProgramError> {
        Array::zeros(array_type).map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

impl<T: NdArrayElement> One<ArrayType> for Array<T> {
    #[inline]
    fn one(array_type: &ArrayType) -> Result<Self, ProgramError> {
        Array::ones(array_type).map_err(|error| TypeError { message: error.to_string() }.into())
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

    fn stack(values: Vec<Self>) -> Result<Self, ProgramError> {
        let lane_count = values.len();
        if lane_count == 0 {
            return Err(TypeError { message: ("cannot stack zero values").into() }.into());
        }
        let first_shape = values[0].values.shape().to_vec();
        for value in values.iter().skip(1) {
            if value.values.shape() != first_shape.as_slice() {
                return Err(TypeError {
                    message: format!(
                        "cannot stack arrays with mismatched shapes: expected {:?}, got {:?}",
                        first_shape,
                        value.values.shape(),
                    ),
                }
                .into());
            }
        }
        let lane_views = values.iter().map(|value| value.values.view()).collect::<Vec<_>>();
        let stacked = ndarray::stack(ndarray::Axis(0), lane_views.as_slice())
            .map_err(|error| TypeError { message: error.to_string() })?;
        Ok(Self::new(stacked))
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

impl<T: NdArrayElement> Scale for Array<T> {
    type Output = Self;

    #[inline]
    fn scale(self, factor: Self) -> Self::Output {
        factor * self
    }
}

impl<T: NdArrayElement> Fill<ArrayType, f64> for Array<T> {
    #[inline]
    fn fill(array_type: &ArrayType, value: f64) -> Result<Self, ProgramError> {
        // Convert the `f64` value to `T` through the element type's `scale_by_constant` helper —
        // multiplying `T::one()` by `value` lifts the constant via the standard cast chain used
        // elsewhere in the backend.
        let element = T::scale_by_constant(T::one(), value);
        Self::full(array_type, element).map_err(|error| TypeError { message: error.to_string() }.into())
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

impl<T: NdArrayElement> Broadcast for Array<T> {
    type Output = Self;

    fn broadcast(self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        let output_type = self.r#type().broadcast(output_type, output_axes)?;
        let input_shape = StaticShape::new(self.values.shape().to_vec());
        let target_shape = StaticShape::new(static_shape(output_type.shape()).map_err(array_error_to_tracing_error)?);
        let standard = self.values.as_standard_layout().to_owned();
        let input_values = standard.as_slice().expect("standard-layout ndarray should produce a flat slice");
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
            values.push(input_values[input_flat].clone());
            for position in (0..target_rank).rev() {
                target_index[position] += 1;
                if target_index[position] < target_shape[position] {
                    break;
                }
                target_index[position] = 0;
            }
        }
        let result = ArrayD::from_shape_vec(IxDyn(target_shape.as_slice()), values)
            .expect("broadcast result shape and value count agree by construction");
        Ok(Self::new(result))
    }
}

impl<T: NdArrayElement> Dot for Array<T> {
    fn dot(self, rhs: Self, dimensions: &DotDimensionNumbers) -> Self {
        let lhs_shape = StaticShape::new(self.values.shape().to_vec());
        let rhs_shape = StaticShape::new(rhs.values.shape().to_vec());
        let lhs_standard = self.values.as_standard_layout().to_owned();
        let rhs_standard = rhs.values.as_standard_layout().to_owned();
        let (values, output_shape) = dot_general_evaluate(
            lhs_standard.as_slice().expect("standard-layout ndarray should produce a flat slice"),
            &lhs_shape,
            rhs_standard.as_slice().expect("standard-layout ndarray should produce a flat slice"),
            &rhs_shape,
            dimensions,
            T::zero,
            |accumulator, lhs_value, rhs_value| T::add(accumulator, T::multiply(*lhs_value, *rhs_value)),
        );
        let result = ArrayD::from_shape_vec(IxDyn(output_shape.as_slice()), values)
            .expect("dot result shape and value count agree by construction");
        Self::new(result)
    }
}

impl<T: NdArrayElement> LeftDot for Array<T> {
    #[inline]
    fn left_dot(self, factor: Self, dimensions: &DotDimensionNumbers) -> Self {
        factor.dot(self, dimensions)
    }
}

impl<T: NdArrayElement> RightDot for Array<T> {
    #[inline]
    fn right_dot(self, factor: Self, dimensions: &DotDimensionNumbers) -> Self {
        self.dot(factor, dimensions)
    }
}

impl<T: NdArrayElement> Select for Array<T> {
    // Staged operation sets are homogeneous over one array type, so the condition is a same-typed array whose
    // elements are interpreted as Booleans (zero is false and any non-zero element is true).
    type Condition = Self;

    fn select(condition: Self, on_true: Self, on_false: Self) -> Result<Self, ProgramError> {
        let predicate_standard = condition.values.as_standard_layout().to_owned();
        let on_true_standard = on_true.values.as_standard_layout().to_owned();
        let on_false_standard = on_false.values.as_standard_layout().to_owned();
        let predicate_slice =
            predicate_standard.as_slice().expect("standard-layout ndarray should produce a flat slice");
        let on_true_slice = on_true_standard.as_slice().expect("standard-layout ndarray should produce a flat slice");
        let on_false_slice = on_false_standard.as_slice().expect("standard-layout ndarray should produce a flat slice");
        let values: Vec<T> = predicate_slice
            .iter()
            .zip(on_true_slice.iter())
            .zip(on_false_slice.iter())
            .map(|((pred, t), f)| if *pred != T::zero() { *t } else { *f })
            .collect();
        let shape = on_true.values.shape().to_vec();
        let result = ArrayD::from_shape_vec(IxDyn(shape.as_slice()), values)
            .expect("select result shape and value count agree by construction");
        Ok(Self::new(result))
    }
}

impl<T: NdArrayElement> Transpose for Array<T> {
    fn transpose(self, permutation: Vec<usize>) -> Self {
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return self;
        }
        let shape = StaticShape::new(self.values.shape().to_vec());
        let standard = self.values.as_standard_layout().to_owned();
        let input_values = standard.as_slice().expect("standard-layout ndarray should produce a flat slice");
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
            values.push(input_values[input_flat].clone());
            for position in (0..rank).rev() {
                permuted_index[position] += 1;
                if permuted_index[position] < permuted_shape[position] {
                    break;
                }
                permuted_index[position] = 0;
            }
        }
        let result = ArrayD::from_shape_vec(IxDyn(permuted_shape.as_slice()), values)
            .expect("transpose result shape and value count agree by construction");
        Self::new(result)
    }
}

impl<T: NdArrayElement> Reshape for Array<T> {
    type Output = Self;

    fn reshape(self, target_shape: Shape) -> Result<Self, ProgramError> {
        let input_type = self.r#type().into_owned();
        let output_type = input_type.reshape(target_shape)?;
        if input_type == output_type {
            return Ok(self);
        }
        let output_shape = static_shape(output_type.shape()).map_err(array_error_to_tracing_error)?;
        let values = self.values.into_iter().collect::<Vec<_>>();
        ArrayD::from_shape_vec(IxDyn(output_shape.as_slice()), values).map(Self::new).map_err(|_| {
            array_error_to_tracing_error(ArrayError::Shape {
                message: "reshape could not realize the requested array shape".to_string(),
            })
        })
    }
}

impl<T: NdArrayElement> ryft_core::operations::compare::Compare for Array<T> {
    type Output = Self;

    fn compare(self, rhs: Self, direction: ryft_core::operations::compare::ComparisonDirection) -> Self {
        use ryft_core::operations::compare::ComparisonDirection;
        // Numeric encoding (matching TestArray and ShardMapTensor): produce Array<T> whose
        // values are `T::zero()` / `T::one()` representing false / true. The `r#type` on
        // Array<T> is derived from T::DATA_TYPE rather than being remapped to Boolean, so
        // downstream consumers check values rather than the type declaration.
        let standard_self = self.values.as_standard_layout().to_owned();
        let standard_rhs = rhs.values.as_standard_layout().to_owned();
        let left = standard_self.as_slice().expect("standard-layout ndarray should produce a flat slice");
        let right = standard_rhs.as_slice().expect("standard-layout ndarray should produce a flat slice");
        let values: Vec<T> = left
            .iter()
            .zip(right.iter())
            .map(|(left, right)| {
                let predicate = match direction {
                    ComparisonDirection::Equal => left == right,
                    ComparisonDirection::NotEqual => left != right,
                    ComparisonDirection::LessThan => left < right,
                    ComparisonDirection::LessThanOrEqual => left <= right,
                    ComparisonDirection::GreaterThan => left > right,
                    ComparisonDirection::GreaterThanOrEqual => left >= right,
                };
                if predicate { T::one() } else { T::zero() }
            })
            .collect();
        let shape = self.values.shape().to_vec();
        let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .expect("compare result shape and value count agree by construction");
        Self::new(result)
    }
}

impl<T: NdArrayElement> Array<T> {
    /// Applies one elementwise binary logical operator, treating any non-zero element as `true` and zero as `false`
    /// (matches the `TestArray` encoding).
    fn binary_logical(self, rhs: Self, operator: impl Fn(bool, bool) -> bool) -> Self {
        let standard_self = self.values.as_standard_layout().to_owned();
        let standard_rhs = rhs.values.as_standard_layout().to_owned();
        let left = standard_self.as_slice().expect("standard-layout ndarray should produce a flat slice");
        let right = standard_rhs.as_slice().expect("standard-layout ndarray should produce a flat slice");
        let values: Vec<T> = left
            .iter()
            .zip(right.iter())
            .map(|(left, right)| if operator(*left != T::zero(), *right != T::zero()) { T::one() } else { T::zero() })
            .collect();
        let shape = self.values.shape().to_vec();
        let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .expect("logical result shape and value count agree by construction");
        Self::new(result)
    }
}

impl<T: NdArrayElement> BitAnd for Array<T> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.binary_logical(rhs, |left, right| left && right)
    }
}

impl<T: NdArrayElement> BitOr for Array<T> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.binary_logical(rhs, |left, right| left || right)
    }
}

impl<T: NdArrayElement> BitXor for Array<T> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        self.binary_logical(rhs, |left, right| left ^ right)
    }
}

impl<T: NdArrayElement> Not for Array<T> {
    type Output = Self;

    fn not(self) -> Self::Output {
        let standard = self.values.as_standard_layout().to_owned();
        let flat = standard.as_slice().expect("standard-layout ndarray should produce a flat slice");
        let values: Vec<T> = flat.iter().map(|value| if *value == T::zero() { T::one() } else { T::zero() }).collect();
        let shape = self.values.shape().to_vec();
        let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .expect("logical negation result shape and value count agree by construction");
        Self::new(result)
    }
}

impl<T: NdArrayElement> ryft_core::tracing_v2::operations::reduce::Reduce for Array<T> {
    fn reduce(self, axes: &[usize], kind: ryft_core::tracing_v2::operations::reduce::ReductionKind) -> Self {
        use ryft_core::tracing_v2::operations::reduce::{ReductionKind, reduce_evaluate};
        if axes.is_empty() {
            return self;
        }
        let shape = StaticShape::new(self.values.shape().to_vec());
        let standard = self.values.as_standard_layout().to_owned();
        let flat = standard.as_slice().expect("standard-layout ndarray should produce a flat slice");
        let (reduced_values, reduced_shape) = match kind {
            ReductionKind::Sum | ReductionKind::Mean => {
                reduce_evaluate(flat, &shape, axes, T::zero, |left, right| T::add(left, right))
            }
            ReductionKind::Max => reduce_evaluate(flat, &shape, axes, T::min_value, |left, right| T::max(left, right)),
            ReductionKind::Min => reduce_evaluate(flat, &shape, axes, T::max_value, |left, right| T::min(left, right)),
            // Any/All on bool-encoded inputs (post-Compare/Logical) use `T::add` (OR for bool,
            // sum for numeric) and `T::multiply` (AND for bool, product for numeric). For
            // numeric T whose values are constrained to `{T::zero(), T::one()}` by upstream
            // Compare/Logical, sum-then-nonzero == any and product-then-nonzero == all. For
            // `T = bool`, the identity element and combiner are exactly the Boolean ones via
            // the recently added `NdArrayElement for bool` impl (`zero = false`, `add = OR`,
            // `one = true`, `multiply = AND`).
            ReductionKind::Any => reduce_evaluate(flat, &shape, axes, T::zero, |left, right| T::add(left, right)),
            ReductionKind::All => reduce_evaluate(flat, &shape, axes, T::one, |left, right| T::multiply(left, right)),
        };
        let mut values = reduced_values;
        if matches!(kind, ReductionKind::Mean) {
            let reduced_count: usize = axes.iter().map(|axis| shape[*axis]).product();
            // For integer element types this is integer division; JAX upcasts to float for
            // `pmean`. The current impl matches JAX only for floating point elements.
            let divisor = if reduced_count == 0 {
                T::one()
            } else {
                let mut acc = T::zero();
                for _ in 0..reduced_count {
                    acc = T::add(acc, T::one());
                }
                acc
            };
            for value in values.iter_mut() {
                *value = T::divide(*value, divisor);
            }
        }
        let result = ArrayD::from_shape_vec(IxDyn(reduced_shape.as_slice()), values)
            .expect("reduce result shape and value count agree by construction");
        Self::new(result)
    }
}

fn validate_array_type<T: NdArrayElement>(array_type: &ArrayType) -> Result<Vec<usize>, ArrayError> {
    if array_type.data_type() != T::DATA_TYPE {
        return Err(ArrayError::ElementTypeMismatch { expected: T::DATA_TYPE, actual: array_type.data_type() });
    }
    if array_type.layout().is_some() {
        return Err(ArrayError::ExplicitLayout);
    }
    if array_type.sharding().is_some() {
        return Err(ArrayError::ShardedArrayType);
    }
    static_shape(array_type.shape())
}

fn static_shape(shape: &Shape) -> Result<Vec<usize>, ArrayError> {
    shape
        .dimensions()
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

fn array_error_to_tracing_error(error: ArrayError) -> ProgramError {
    TypeError { message: error.to_string() }.into()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use ndarray::{arr0, arr1, arr2};
    use pretty_assertions::assert_eq;
    use ryft_core::contexts::StagingContext;
    use ryft_core::differentiation::{Cotangent, TransposableOperation};
    use ryft_core::domains::AbstractDomain;
    use ryft_core::operations::BooleanLike;
    use ryft_core::operations::manipulation::Reshape;
    use ryft_core::operations::manipulation::ReshapeOperation;
    use ryft_core::operations::manipulation::Transpose;
    use ryft_core::parameters::Placeholder;
    use ryft_core::programs::ProgramBuilder;
    use ryft_core::tracing::{AbstractTracingContext, TracingContext};
    use ryft_core::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};
    use ryft_core::types::{ArrayType, DataType, Shape, Size, Typed};

    use crate::{LinearNdarrayOperation, NdArrayDomain};

    use super::Array;

    #[test]
    fn test_array_reports_static_unsharded_type() {
        let array = Array::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let expected_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));

        assert_eq!(array.r#type().into_owned(), expected_type);
        assert_eq!(Array::scalar(2.0).as_ndarray(), &arr0(2.0).into_dyn());
    }

    #[test]
    fn test_array_display_delegates_to_ndarray_values() {
        assert_eq!(Array::scalar(2.0).to_string(), "2");
        assert_eq!(Array::from_shape_vec([2], vec![1.0, 2.0]).unwrap().to_string(), "[1, 2]");
    }

    #[test]
    fn test_array_constant_kernels_reject_dynamically_sized_types() {
        use ryft_core::operations::constants::{Fill, One, Zero};

        // Kernels that materialize a payload from a type cannot do so for dynamically sized types and must error
        // instead of panicking.
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(3)]));
        let expected_message = "ndarray backend requires static shape dimensions, but dimension #0 is *";
        assert_eq!(<Array>::zero(&dynamic_type).unwrap_err().to_string(), expected_message);
        assert_eq!(<Array>::one(&dynamic_type).unwrap_err().to_string(), expected_message);
        assert_eq!(<Array>::fill(&dynamic_type, 42.0).unwrap_err().to_string(), expected_message);
    }

    #[test]
    fn test_array_boolean_reports_invalid_type() {
        let array = Array::from_shape_vec([2], vec![1.0, 2.0]).unwrap();

        assert_eq!(
            array.boolean().unwrap_err().to_string(),
            "cannot extract a concrete boolean from a value of type f64[2]; expected bool[]"
        );
        assert_eq!(array.as_boolean().as_ndarray(), &arr1(&[1.0, 1.0]).into_dyn());
        assert_eq!(
            Array::from_shape_vec([2], vec![0.0, 2.0]).unwrap().as_boolean().as_ndarray(),
            &arr1(&[0.0, 1.0]).into_dyn(),
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
        let domain = NdArrayDomain::<f64>::new();
        let (_, compiled): (Array<f64>, _) = TracingContext::interpret_and_trace(
            &domain,
            |x| x.reshape(Shape::new(vec![Size::Static(1), Size::Static(4)])),
            input,
        )
        .unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc::indoc! {"
                lambda %0:f64[2, 2] .
                let %1:f64[1, 4] = reshape [shape=[1, 4]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reshape_transpose_restores_the_input_shape() {
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)]));
        let output_value = Array::from_shape_vec([1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let transpose_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, Array<f64>, LinearNdarrayOperation<Array<f64>>>::new()));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(output_value.r#type().into_owned());
        let domain = AbstractDomain::new();
        let mut context = AbstractTracingContext::new(&domain, transpose_builder.clone());
        let output_cotangent = context.tracer(output_cotangent_atom, None);
        let contribution = ReshapeOperation::new(Shape::new(vec![Size::Static(1), Size::Static(4)]))
            .transpose(&mut context, &[&input_type], &[Cotangent::Staged(output_cotangent)])
            .unwrap()
            .into_iter()
            .next()
            .expect("transpose should return one contribution");
        let Cotangent::Staged(contribution) = contribution else {
            panic!("transpose should produce one cotangent contribution");
        };
        let contribution_atom = contribution.atom_id().unwrap();
        drop(contribution);
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
        let transposed = right.clone().transpose(vec![1, 0]);

        assert_eq!(
            left.dot(right, &DotDimensionNumbers::matmul()).as_ndarray(),
            &arr2(&[[58.0, 64.0], [139.0, 154.0]]).into_dyn(),
        );
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
