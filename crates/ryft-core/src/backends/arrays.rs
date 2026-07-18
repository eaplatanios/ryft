//! Contains the reference array backend that supports concrete [`Array`] values over the [`ArrayOperation`] family,
//! together with the [`ArrayTracingContext`] used to stage array [`Program`](crate::Program)s. This backend serves
//! programs whose values are dense multidimensional arrays typed by [`ArrayType`]. It is meant primarily for exercising
//! the Ryft tracing, transformation, and interpretation machinery without depending on an optimized backend such as
//! `ryft-xla`: unit tests, documentation tests, and downstream crates can stage, transform, and interpret complete
//! array programs eagerly. [`Array`] stores a flat row-major [`Scalar`] payload, so that every per-element concern
//! (e.g., the exact `f4`/`f8` bit encodings, complex arithmetic, integer wrapping semantics, and fallible arithmetic)
//! delegates to the scalar reference backend and [`Array`] adds only the shape logic.
//!
//! # Warning
//!
//! This backend prioritizes transparency over performance: payloads are contiguous [`Scalar`] vectors with no strides,
//! views, or vectorization, and every operation is implemented with straightforward index arithmetic. Do not use it
//! outside of tests, documentation examples, and reference-semantics checks.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::Display;

use approx::AbsDiffEq;

use ryft_macros::Operation;

// TODO(eaplatanios): Review from here onwards.

use crate::backends::scalars::Scalar;
use crate::broadcasting::Broadcastable;
use crate::contexts::EagerContext;
use crate::operations::BooleanLike;
use crate::operations::compare::{Compare, CompareOperation, ComparisonDirection};
use crate::operations::complex::{
    ComplexOperation, Conjugate, ConjugateOperation, Imaginary, ImaginaryOperation, Real, RealOperation,
};
use crate::operations::constants::{
    ConstantOperation, Fill, FillOperation, IotaOperation, One, OneLike, OneLikeOperation, OneOperation, Zero,
    ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::{ConditionOperation, ScanOperation, SelectOperation, WhileOperation};
use crate::operations::debugging::PrintOperation;
use crate::operations::differentiation::{CoordinateBasisOperation, StopGradientOperation};
use crate::operations::logical::{And, AndOperation, Not, NotOperation, Or, OrOperation, Xor, XorOperation};
use crate::operations::manipulation::{
    Broadcast, BroadcastOperation, Concatenate, ConcatenateOperation, ConvertElementType, ConvertElementTypeOperation,
    DynamicSlice, DynamicSliceOperation, DynamicUpdateSlice, DynamicUpdateSliceOperation, Gather, GatherOperation,
    GatherScatterMode, Pad, PadOperation, Reshape, ReshapeOperation, Scatter, ScatterOperation, ScatterReductionKind,
    Slice, SliceOperation, Transpose, TransposeOperation, UpdateSlice, UpdateSliceOperation,
};
use crate::operations::math::{
    Abs, AbsOperation, Add, AddOperation, Atan2, Atan2Operation, Cos, CosOperation, Div, DivOperation, Exp,
    ExpOperation, Log, LogOperation, Mul, MulOperation, Neg, NegOperation, Sin, SinOperation, Sqrt, SqrtOperation, Sub,
    SubOperation,
};
use crate::operations::sharding::{ReshardOperation, ShardingConstraintOperation};
use crate::operations::tag::{Tag, TagOperation};
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::TracingContext;
use crate::tracing_v2::operations::collective::{AxisIndexOperation, CollectiveOperation};
use crate::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpOperation, CustomVjpTangentOperation,
};
use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers, DotOperation, dot_general_evaluate};
use crate::tracing_v2::operations::memory::{TransferToMemory, TransferToMemoryOperation};
use crate::tracing_v2::operations::reduce::{Reduce, ReduceOperation, ReductionKind, reduce_evaluate};
use crate::tracing_v2::rematerialization::RematerializeOperation;
use crate::types::{ArrayType, DataType, Shape, Size, StaticShape};
use crate::{Select, SelectCondition};

/// Reusable [`Operation`] enum for ordinary staged array programs.
///
/// [`ArrayOperation`] is the ordinary operation enum for core tests and backend crates, pairing with [`Array`] the
/// same way [`ScalarOperation`](crate::backends::scalars::ScalarOperation) pairs with [`Scalar`]. Most variants are
/// thin tags around one semantic primitive defined in [`crate::operations`] or [`crate::tracing_v2::operations`].
///
/// Each variant wraps exactly the backing operation struct that owns the variant's semantics (type inference,
/// rendering, and interpretation): for example [`Zero`](Self::Zero) wraps a [`ZeroOperation`] and
/// [`Dot`](Self::Dot) a [`DotOperation`].
#[derive(Clone, Debug, Operation)]
#[ryft(dispatch(batching, differentiation, transposition))]
pub enum ArrayOperation<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<ArrayType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<V>),
    Fill(FillOperation<ArrayType, Scalar>),
    Iota(IotaOperation<ArrayType>),
    CoordinateBasis(CoordinateBasisOperation<ArrayType>),
    Abs(AbsOperation),
    Neg(NegOperation),
    Add(AddOperation),
    Sub(SubOperation),
    Mul(MulOperation),
    Div(DivOperation),
    Sin(SinOperation),
    Cos(CosOperation),
    Atan2(Atan2Operation),
    Exp(ExpOperation),
    Log(LogOperation),
    Sqrt(SqrtOperation),
    Not(NotOperation),
    And(AndOperation),
    Or(OrOperation),
    Xor(XorOperation),
    Complex(ComplexOperation),
    Conjugate(ConjugateOperation),
    Real(RealOperation),
    Imaginary(ImaginaryOperation),
    Dot(DotOperation),
    Reduce(ReduceOperation),
    Collective(CollectiveOperation),
    AxisIndex(AxisIndexOperation),
    Transpose(TransposeOperation),
    Reshape(ReshapeOperation),
    Broadcast(BroadcastOperation),
    Pad(PadOperation),
    Concatenate(ConcatenateOperation),
    Gather(GatherOperation),
    Scatter(ScatterOperation),
    Slice(SliceOperation),
    UpdateSlice(UpdateSliceOperation),
    DynamicSlice(DynamicSliceOperation),
    DynamicUpdateSlice(DynamicUpdateSliceOperation),
    Compare(CompareOperation),
    Select(SelectOperation),
    Condition(ConditionOperation<V>),
    While(WhileOperation),
    Scan(ScanOperation<V>),
    ConvertElementType(ConvertElementTypeOperation),
    TransferToMemory(TransferToMemoryOperation),
    Reshard(ReshardOperation),
    ShardingConstraint(ShardingConstraintOperation),
    StopGradient(StopGradientOperation),
    Tag(TagOperation),
    Rematerialize(RematerializeOperation),
    Print(PrintOperation),
    CustomJvp(CustomJvpOperation),
    CustomVjp(CustomVjpOperation),
    CustomVjpTangent(CustomVjpTangentOperation<ArrayType>),
}

/// [`TracingContext`] over the array universe, pairing [`ArrayType`] types and [`Array`] staged constants with the
/// [`ArrayOperation`] family.
pub type ArrayTracingContext = TracingContext<Array, ArrayOperation<Array>>;

/// Dense multidimensional [`Value`] whose [`Type`] is an [`ArrayType`] and which is meant to be used
/// primarily for testing the Ryft infrastructure and machinery with programs that involve multidimensional arrays,
/// without depending on an optimized backend such as `ryft-xla`.
///
/// The payload is a flat row-major [`Scalar`] vector whose elements all share the array's element [`DataType`], so
/// per-element semantics (including complex arithmetic, integer wrapping, and the exact low-precision floating-point
/// encodings) match the scalar reference backend exactly. [`Array::new`] enforces the payload invariants: every
/// element's data type matches the array type's element data type, and the payload length matches the array type's
/// static element count.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::backends::arrays::Array;
/// # use ryft_core::operations::math::Add;
/// let left = Array::vector(vec![1.0, 2.0]);
/// let right = Array::vector(vec![3.0, 4.0]);
/// assert_eq!(left.add(&right).unwrap(), Array::vector(vec![4.0, 6.0]));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Array {
    /// Staged array type of this array value.
    r#type: ArrayType,

    /// Row-major payload whose elements all have this array's element data type.
    values: Vec<Scalar>,
}

impl Array {
    /// Creates an array from its staged array type and row-major payload, enforcing that every element of `values`
    /// has the element data type declared by `type` and that `values` has exactly as many elements as `type`'s
    /// static shape requires. Dynamically shaped types are rejected because they cannot describe a materialized
    /// payload.
    ///
    /// # Parameters
    ///
    ///   - `type`: Staged array type of the array.
    ///   - `values`: Row-major payload of the array.
    pub fn new(r#type: ArrayType, values: Vec<Scalar>) -> Result<Self, ProgramError> {
        let element_count = Self::materialized_element_count(&r#type)?;
        if values.len() != element_count {
            return Err(TypeError {
                message: format!(
                    "array type {type} requires {element_count} elements but got {count}",
                    count = values.len(),
                ),
            }
            .into());
        }
        let data_type = r#type.data_type();
        for value in &values {
            let value_type = value.r#type().into_owned();
            if value_type != data_type {
                return Err(TypeError {
                    message: format!(
                        "array of element data type {data_type} cannot store an element of data type {value_type}"
                    ),
                }
                .into());
            }
        }
        Ok(Self { r#type, values })
    }

    /// Creates a rank-0 scalar array whose element data type is that of the provided scalar. Panics if the value
    /// cannot form a valid array, which cannot happen for any [`Scalar`], making this constructor effectively
    /// infallible test-writing sugar.
    pub fn scalar(value: impl Into<Scalar>) -> Self {
        let value = value.into();
        let r#type = ArrayType::scalar(value.r#type().into_owned());
        Self::new(r#type, vec![value]).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Creates a rank-1 array whose element data type is inferred from the first element (defaulting to
    /// [`DataType::F64`] for an empty payload). Panics if the elements do not all share one data type.
    pub fn vector<S: Into<Scalar>>(values: Vec<S>) -> Self {
        let values: Vec<Scalar> = values.into_iter().map(Into::into).collect();
        let data_type = values.first().map_or(DataType::F64, |value| value.r#type().into_owned());
        let r#type = ArrayType::new(data_type, Shape::new(vec![Size::Static(values.len())]));
        Self::new(r#type, values).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Creates a rank-2 array whose element data type is inferred from the first element (defaulting to
    /// [`DataType::F64`] for an empty payload). Panics if the elements do not all share one data type or if the
    /// payload length does not match `rows * columns`.
    pub fn matrix<S: Into<Scalar>>(rows: usize, columns: usize, values: Vec<S>) -> Self {
        let values: Vec<Scalar> = values.into_iter().map(Into::into).collect();
        let data_type = values.first().map_or(DataType::F64, |value| value.r#type().into_owned());
        let r#type = ArrayType::new(data_type, Shape::new(vec![Size::Static(rows), Size::Static(columns)]));
        Self::new(r#type, values).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Creates an array from its staged array type and a row-major `f64` payload, converting each element into the
    /// element data type declared by `type` (e.g., rounding into a low-precision floating-point encoding). This is
    /// test-writing sugar for payloads written as plain floating-point literals, and it panics when an element
    /// cannot be converted or when the payload does not match `type`, because in tests such a failure is the
    /// assertion failing.
    ///
    /// # Parameters
    ///
    ///   - `type`: Staged array type of the array.
    ///   - `values`: Row-major payload of the array, converted elementwise into `type`'s element data type.
    pub fn from_f64s(r#type: ArrayType, values: Vec<f64>) -> Self {
        let data_type = r#type.data_type();
        let values = values
            .into_iter()
            .map(|value| Scalar::from(value).convert_element_type(data_type).unwrap_or_else(|error| panic!("{error}")))
            .collect();
        Self::new(r#type, values).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Returns the row-major payload of this array.
    pub fn values(&self) -> &[Scalar] {
        &self.values
    }

    /// Returns the row-major payload of this array converted elementwise to `f64`. This is a test-assertion view for
    /// real-valued arrays (Booleans convert to `0.0`/`1.0` and integers to their exact values where representable),
    /// and it panics for arrays whose elements cannot be viewed as real numbers (complex, token, and structural-zero
    /// element data types), because in tests such a failure is the assertion failing.
    pub fn to_f64s(&self) -> Vec<f64> {
        let data_type = self.r#type.data_type();
        if data_type.is_complex() {
            panic!("cannot view an array of complex element data type {data_type} as f64 values");
        }
        self.values
            .iter()
            .map(|value| match value.convert_element_type(DataType::F64) {
                Ok(Scalar::F64(converted)) => converted,
                _ => panic!("cannot view an array of element data type {data_type} as f64 values"),
            })
            .collect()
    }

    /// Returns the number of elements represented by `type`. Panics if the type has dynamic dimensions, so this
    /// helper is reserved for types of already-materialized values (which are always fully static); kernels that
    /// materialize values from payload types use [`Array::materialized_element_count`] instead.
    pub fn element_count(r#type: &ArrayType) -> usize {
        r#type.element_count().unwrap().unwrap()
    }

    /// Returns the number of elements represented by `type`, or an error when `type` has dynamic dimensions and
    /// therefore cannot be materialized into a concrete payload.
    pub fn materialized_element_count(r#type: &ArrayType) -> Result<usize, ProgramError> {
        r#type.element_count().map_err(|error| TypeError { message: error.to_string() })?.ok_or_else(|| {
            TypeError { message: format!("cannot materialize a value of dynamically sized type {}", r#type) }.into()
        })
    }

    /// Returns the zero [`Scalar`] of the provided element data type, used by kernels that need a payload identity
    /// element (e.g., dot-product accumulators and dropped gather results).
    fn zero_element(data_type: DataType) -> Result<Scalar, ProgramError> {
        EagerContext::<Scalar>::new().zero(&data_type)
    }

    /// Applies an elementwise unary function to the payload, preserving this array's type.
    fn unary(&self, function: impl Fn(&Scalar) -> Result<Scalar, ProgramError>) -> Result<Self, ProgramError> {
        let values = self.values.iter().map(function).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { r#type: self.r#type.clone(), values })
    }

    /// Applies an elementwise binary function using scalar broadcasting. The output type is the broadcast of the two
    /// operand types (including element-type promotion), and the provided function is expected to promote its scalar
    /// operands congruently (as all the [`Scalar`] arithmetic capabilities do).
    fn binary(
        &self,
        rhs: &Self,
        function: impl Fn(&Scalar, &Scalar) -> Result<Scalar, ProgramError>,
    ) -> Result<Self, ProgramError> {
        let output_type = Broadcastable::broadcast(&self.r#type, &rhs.r#type)
            .map_err(|error| TypeError { message: error.to_string() })?;
        let output_len = Self::element_count(&output_type);
        let left = self.broadcast_values(output_len);
        let right = rhs.broadcast_values(output_len);
        let values = left
            .iter()
            .zip(right.iter())
            .map(|(left, right)| function(left, right))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { r#type: output_type, values })
    }

    /// Broadcasts the payload to `output_len`.
    fn broadcast_values(&self, output_len: usize) -> Vec<Scalar> {
        if self.values.len() == output_len {
            self.values.clone()
        } else if self.values.len() == 1 {
            vec![self.values[0]; output_len]
        } else {
            panic!("cannot broadcast {} values to {output_len}", self.values.len());
        }
    }

    /// Extracts the in-band integer payload of a scalar index element. Panics for non-integer elements, which the
    /// type-level validation performed by every indexing kernel rules out before payload access.
    fn index_value(index: Scalar) -> i64 {
        match index.convert_element_type(DataType::I64) {
            Ok(Scalar::I64(value)) => value,
            _ => panic!("cannot use a scalar of data type {} as an index", index.r#type()),
        }
    }
}

#[cfg(test)]
impl Array {
    /// Creates an array without enforcing the payload invariants, so that `ryft-core`'s own transform-validation
    /// tests can materialize values whose declared types are deliberately not materializable (e.g., dynamically
    /// shaped types) and exercise the type-level rejection paths. Never use this outside of such validation tests.
    pub(crate) fn with_unchecked_type(r#type: ArrayType, values: Vec<Scalar>) -> Self {
        Self { r#type, values }
    }
}

impl Parameter for Array {}

// The rendering intentionally matches how `Vec<f64>` debug-formats: a bracketed, comma-separated element list in
// which real floating-point payloads keep a decimal point (e.g., `[1.0, 2.0]`), so program and interpreter
// diagnostics involving constant arrays stay readable and stable.
impl Display for Array {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("[")?;
        for (index, value) in self.values.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            match value {
                Scalar::F32(value) => write!(formatter, "{value:?}")?,
                Scalar::F64(value) => write!(formatter, "{value:?}")?,
                other => Display::fmt(other, formatter)?,
            }
        }
        formatter.write_str("]")
    }
}

impl Typed for Array {
    type Type = ArrayType;

    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl Value for Array {
    type DispatchDomain = EagerContext<Self>;
    // A concrete `Array`'s active context is the reference backend's rich eager domain (unlike the constant-only
    // `EagerContext<Array>` it declares as its `Value::DispatchDomain`, which cannot bind operations), so free
    // transform entry points such as `crate::batching::batch` serve top-level concrete values.
    type ExecutionDomain = EagerContext<Self, ArrayOperation<Self>>;

    fn dispatch_domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }

    fn execution_domain(&self) -> EagerContext<Self, ArrayOperation<Self>> {
        EagerContext::new()
    }
}

// Approximate equality requires identical array types and delegates to the elementwise `Scalar` approximation, which
// compares real floating-point payloads through their exactly widened `f64` values and complex payloads through both
// parts.
impl AbsDiffEq for Array {
    type Epsilon = f64;

    fn default_epsilon() -> f64 {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f64) -> bool {
        self.r#type == other.r#type
            && self.values.len() == other.values.len()
            && self.values.iter().zip(other.values.iter()).all(|(left, right)| left.abs_diff_eq(right, epsilon))
    }
}

impl<O: Operation<ArrayType>> Zero<Array> for EagerContext<Array, O> {
    fn zero(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
        let element = Array::zero_element(r#type.data_type())?;
        Ok(Array { r#type: r#type.clone(), values: vec![element; Array::materialized_element_count(r#type)?] })
    }
}

impl ZeroLike for Array {
    fn zero_like(&self) -> Self {
        Self { r#type: self.r#type.clone(), values: self.values.iter().map(|value| value.zero_like()).collect() }
    }
}

impl<O: Operation<ArrayType>> One<Array> for EagerContext<Array, O> {
    fn one(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
        let element = EagerContext::<Scalar>::new().one(&r#type.data_type())?;
        Ok(Array { r#type: r#type.clone(), values: vec![element; Array::materialized_element_count(r#type)?] })
    }
}

impl OneLike for Array {
    fn one_like(&self) -> Self {
        Self { r#type: self.r#type.clone(), values: self.values.iter().map(|value| value.one_like()).collect() }
    }
}

impl<O: Operation<ArrayType>> Fill<Scalar, Array> for EagerContext<Array, O> {
    fn fill(&self, r#type: &ArrayType, value: Scalar) -> Result<Array, ProgramError> {
        // The fill element must be losslessly representable in the array's element data type, so this uses the
        // promotion-checked conversion rather than an unchecked cast: filling a real array with a complex scalar (or
        // any other narrowing fill) is rejected instead of silently discarding payload.
        let element = value.promote_element_type(r#type.data_type())?;
        Ok(Array { r#type: r#type.clone(), values: vec![element; Array::materialized_element_count(r#type)?] })
    }
}

impl<O: Operation<ArrayType>> crate::operations::constants::Iota<Array> for EagerContext<Array, O> {
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<Array, ProgramError> {
        let sizes = r#type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| {
                dimension.value().ok_or_else(|| TypeError {
                    message: format!("cannot materialize an iota of dynamically sized type {type}"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if dimension >= sizes.len() {
            return Err(TypeError {
                message: format!("iota dimension {dimension} is out of bounds for array type {type}"),
            }
            .into());
        }
        // In row-major order, the index along `dimension` at flat position `flat` is `(flat / stride) % size`, where
        // `stride` is the product of the sizes of the dimensions after `dimension`.
        let size = sizes[dimension];
        let stride: usize = sizes[dimension + 1..].iter().product();
        let data_type = r#type.data_type();
        let element_count = Array::materialized_element_count(r#type)?;
        let values = (0..element_count)
            .map(|flat| Scalar::from(((flat / stride) % size) as u64).convert_element_type(data_type))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Array { r#type: r#type.clone(), values })
    }
}

impl Abs for Array {
    fn abs(&self) -> Result<Self, ProgramError> {
        // The absolute value of a complex array is its elementwise magnitude, so the element data type maps to its
        // real part data type, mirroring the `AbsOperation` type-inference contract.
        let data_type = match self.r#type.data_type() {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            other => other,
        };
        let values = self.values.iter().map(|value| value.abs()).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { r#type: self.r#type.clone().with_data_type(data_type), values })
    }
}

impl Neg for Array {
    fn neg(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.neg())
    }
}

impl std::ops::Neg for Array {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Neg::neg(&self).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl Add for Array {
    fn add(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary(rhs, |left, right| left.add(right))
    }
}

impl Sub for Array {
    fn sub(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary(rhs, |left, right| left.sub(right))
    }
}

impl Mul for Array {
    fn mul(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary(rhs, |left, right| left.mul(right))
    }
}

impl Div for Array {
    fn div(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary(rhs, |left, right| left.div(right))
    }
}

impl std::ops::Add for Array {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Add::add(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Sub for Array {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Sub::sub(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Mul for Array {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Mul::mul(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Mul<f64> for Array {
    type Output = Self;

    /// Scales every element by `rhs`, converting `rhs` into this array's element data type first so that scaling
    /// preserves the array's type (e.g., scaling an `f32` array does not promote it to `f64`).
    fn mul(self, rhs: f64) -> Self::Output {
        let factor = Scalar::from(rhs)
            .convert_element_type(self.r#type.data_type())
            .unwrap_or_else(|error| panic!("{error}"));
        let values = self.values.into_iter().map(|value| value * factor).collect();
        Self { r#type: self.r#type, values }
    }
}

impl std::ops::Div for Array {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Div::div(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl Sin for Array {
    fn sin(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.sin())
    }
}

impl Cos for Array {
    fn cos(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.cos())
    }
}

impl Atan2 for Array {
    fn atan2(&self, x: &Self) -> Result<Self, ProgramError> {
        self.binary(x, |y, x| y.atan2(x))
    }
}

impl Exp for Array {
    fn exp(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.exp())
    }
}

impl Log for Array {
    fn log(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.log())
    }
}

impl Sqrt for Array {
    fn sqrt(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.sqrt())
    }
}

impl Not for Array {
    fn not(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.not())
    }
}

impl And for Array {
    fn and(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary(rhs, |left, right| left.and(right))
    }
}

impl Or for Array {
    fn or(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary(rhs, |left, right| left.or(right))
    }
}

impl Xor for Array {
    fn xor(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary(rhs, |left, right| left.xor(right))
    }
}

impl std::ops::Not for Array {
    type Output = Self;

    fn not(self) -> Self::Output {
        Not::not(&self).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::BitAnd for Array {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        And::and(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::BitOr for Array {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Or::or(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::BitXor for Array {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Xor::xor(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl crate::operations::complex::Complex for Array {
    fn complex(&self, imaginary: &Self) -> Result<Self, ProgramError> {
        // Mirrors the `ComplexOperation` type-inference contract: the two part arrays must have identical types, and
        // the element data type maps to the complex data type with the parts' precision.
        if self.r#type != imaginary.r#type {
            return Err(TypeError {
                message: format!(
                    "'complex' requires identical part types but got {} and {}",
                    self.r#type, imaginary.r#type,
                ),
            }
            .into());
        }
        let data_type = match self.r#type.data_type() {
            DataType::F32 => DataType::C64,
            DataType::F64 => DataType::C128,
            other => {
                return Err(TypeError {
                    message: format!("cannot construct a complex value from parts of data type {other}"),
                }
                .into());
            }
        };
        let values = self
            .values
            .iter()
            .zip(imaginary.values.iter())
            .map(|(real, imaginary)| real.complex(imaginary))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { r#type: self.r#type.clone().with_data_type(data_type), values })
    }
}

impl Conjugate for Array {
    fn conjugate(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.conjugate())
    }
}

impl Real for Array {
    fn real(&self) -> Result<Self, ProgramError> {
        // The real part of a complex array has the parts' real data type, mirroring the `RealOperation`
        // type-inference contract; real arrays keep their element data type.
        let data_type = match self.r#type.data_type() {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            other => other,
        };
        let values = self.values.iter().map(|value| value.real()).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { r#type: self.r#type.clone().with_data_type(data_type), values })
    }
}

impl Imaginary for Array {
    fn imaginary(&self) -> Result<Self, ProgramError> {
        // The imaginary part of a complex array has the parts' real data type, mirroring the `ImaginaryOperation`
        // type-inference contract; real arrays keep their element data type.
        let data_type = match self.r#type.data_type() {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            other => other,
        };
        let values = self.values.iter().map(|value| value.imaginary()).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { r#type: self.r#type.clone().with_data_type(data_type), values })
    }
}

impl Dot for Array {
    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Self {
        let lhs_shape = self.r#type.static_shape().unwrap();
        let rhs_shape = rhs.r#type.static_shape().unwrap();
        let zero = Self::zero_element(self.r#type.data_type()).unwrap_or_else(|error| panic!("{error}"));
        let (values, output_shape) = dot_general_evaluate(
            self.values.as_slice(),
            &lhs_shape,
            rhs.values.as_slice(),
            &rhs_shape,
            dimensions,
            || zero,
            |accumulator, lhs_value, rhs_value| accumulator + *lhs_value * *rhs_value,
        );
        let output_type = ArrayType::new(self.r#type.data_type(), Shape::from(&output_shape));
        Self { r#type: output_type, values }
    }
}

impl Reduce for Array {
    fn reduce(&self, axes: &[usize], kind: ReductionKind) -> Self {
        if axes.is_empty() {
            return self.clone();
        }
        let data_type = self.r#type.data_type();
        let shape = self.r#type.static_shape().unwrap();
        let (mut values, reduced_shape) = match kind {
            ReductionKind::Sum | ReductionKind::Mean => {
                let zero = Self::zero_element(data_type).unwrap_or_else(|error| panic!("{error}"));
                reduce_evaluate(self.values.as_slice(), &shape, axes, || zero, |accumulator, value| accumulator + value)
            }
            ReductionKind::Max => reduce_extremum(&self.values, &shape, axes, ComparisonDirection::GreaterThan),
            ReductionKind::Min => reduce_extremum(&self.values, &shape, axes, ComparisonDirection::LessThan),
            ReductionKind::Any => reduce_evaluate(
                self.values.as_slice(),
                &shape,
                axes,
                || Scalar::Bool(false),
                |accumulator, value| accumulator | value,
            ),
            ReductionKind::All => reduce_evaluate(
                self.values.as_slice(),
                &shape,
                axes,
                || Scalar::Bool(true),
                |accumulator, value| accumulator & value,
            ),
        };
        if matches!(kind, ReductionKind::Mean) {
            let reduced_count: usize = axes.iter().map(|axis| shape[*axis]).product();
            let divisor = Scalar::from(reduced_count.max(1) as f64)
                .convert_element_type(data_type)
                .unwrap_or_else(|error| panic!("{error}"));
            for value in values.iter_mut() {
                *value = *value / divisor;
            }
        }
        let output_type = ArrayType::new(data_type, Shape::from(&reduced_shape));
        Self { r#type: output_type, values }
    }
}

/// Reduces `values` along `axes` keeping the extremum in the provided `direction` (the maximum for
/// [`ComparisonDirection::GreaterThan`] and the minimum for [`ComparisonDirection::LessThan`]). The accumulator is an
/// `Option` because max/min have no identity element that is representable for every element data type (e.g.,
/// integers have no infinities), so reducing an empty axis panics instead of materializing a synthetic identity.
fn reduce_extremum(
    values: &[Scalar],
    shape: &StaticShape,
    axes: &[usize],
    direction: ComparisonDirection,
) -> (Vec<Scalar>, StaticShape) {
    let wrapped: Vec<Option<Scalar>> = values.iter().map(|value| Some(*value)).collect();
    let (reduced, reduced_shape) = reduce_evaluate(
        wrapped.as_slice(),
        shape,
        axes,
        || None,
        |accumulator, value| match (accumulator, value) {
            (None, value) => value,
            (accumulator, None) => accumulator,
            (Some(accumulator), Some(value)) => Some(extremum(accumulator, value, direction)),
        },
    );
    let values = reduced
        .into_iter()
        .map(|value| value.expect("cannot reduce an empty axis with a max or min reduction"))
        .collect();
    (values, reduced_shape)
}

/// Returns the extremum of two same-data-type scalars in the provided `direction` (the maximum for
/// [`ComparisonDirection::GreaterThan`] and the minimum for [`ComparisonDirection::LessThan`]), panicking for
/// unordered element data types such as the complex ones.
fn extremum(left: Scalar, right: Scalar, direction: ComparisonDirection) -> Scalar {
    let keep_left = left.compare(&right, direction).unwrap_or_else(|error| panic!("{error}"));
    if matches!(keep_left, Scalar::Bool(true)) { left } else { right }
}

impl Transpose for Array {
    fn transpose<P: AsRef<[usize]>>(&self, permutation: P) -> Result<Self, ProgramError> {
        // Validate the permutation and compute the output type (including sharding) via the type-level rule, so an
        // out-of-range or duplicated axis is a clean error rather than an out-of-bounds panic.
        let permutation = permutation.as_ref();
        let output_type = self.r#type.transpose(permutation)?;
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return Ok(self.clone());
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
        Ok(Self { r#type: output_type, values })
    }
}

impl Reshape for Array {
    fn reshape(&self, target_shape: Shape) -> Result<Self, ProgramError> {
        // Delegate to the type-level reshape so that element-count mismatches and dynamic target shapes surface the
        // canonical reshape errors instead of panicking, and reinterpret the row-major payload under the result.
        let output_type = self.r#type.reshape(target_shape)?;
        Ok(Self { r#type: output_type, values: self.values.clone() })
    }
}

impl Broadcast for Array {
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        let r#type = Broadcast::broadcast(&self.r#type, output_type, output_axes)?;
        let input_shape = self.r#type.static_shape().unwrap();
        let Some(target_shape) = r#type.static_shape() else {
            return Err(TypeError {
                message: format!("cannot materialize a value of dynamically sized type {}", r#type),
            }
            .into());
        };
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

impl Array {
    /// Copies the row-major block of shape `sizes` out of this array's payload, reading the element at index
    /// `start_indices + block_index * strides` along each axis. The caller guarantees that the block lies in bounds.
    fn copy_block(&self, start_indices: &[usize], strides: &[usize], sizes: &[usize]) -> Vec<Scalar> {
        let input_shape = self.r#type.static_shape().unwrap();
        let input_strides = input_shape.row_major_strides();
        let rank = input_shape.rank();
        let output_count: usize = sizes.iter().product();
        let mut values = Vec::with_capacity(output_count);
        let mut block_index = vec![0usize; rank];
        while values.len() < output_count {
            let mut input_flat = 0usize;
            for axis in 0..rank {
                input_flat += (start_indices[axis] + block_index[axis] * strides[axis]) * input_strides[axis];
            }
            values.push(self.values[input_flat]);
            for position in (0..rank).rev() {
                block_index[position] += 1;
                if block_index[position] < sizes[position] {
                    break;
                }
                block_index[position] = 0;
            }
        }
        values
    }

    /// Overwrites the row-major block of `update`'s shape starting at `start_indices` in this array's payload with
    /// `update`'s payload. The caller guarantees that the block lies in bounds.
    fn replace_block(mut self, update: &Array, start_indices: &[usize]) -> Self {
        let input_shape = self.r#type.static_shape().unwrap();
        let update_shape = update.r#type.static_shape().unwrap();
        let input_strides = input_shape.row_major_strides();
        let rank = input_shape.rank();
        let update_count: usize = update_shape.dimensions().iter().product();
        let mut block_index = vec![0usize; rank];
        let mut written = 0usize;
        while written < update_count {
            let mut input_flat = 0usize;
            for axis in 0..rank {
                input_flat += (start_indices[axis] + block_index[axis]) * input_strides[axis];
            }
            self.values[input_flat] = update.values[written];
            written += 1;
            for position in (0..rank).rev() {
                block_index[position] += 1;
                if block_index[position] < update_shape[position] {
                    break;
                }
                block_index[position] = 0;
            }
        }
        self
    }

    /// Extracts the in-band scalar start indices of a dynamic slicing operation and clamps them per StableHLO
    /// semantics: the effective start index along axis `d` is
    /// `clamp(0, start_indices[d], input_dimension[d] - block_sizes[d])`.
    fn clamped_start_indices(start_indices: &[Array], input_shape: &StaticShape, block_sizes: &[usize]) -> Vec<usize> {
        start_indices
            .iter()
            .enumerate()
            .map(|(axis, index)| {
                let raw = Self::index_value(index.values[0]);
                let maximum = (input_shape[axis] - block_sizes[axis]) as i64;
                raw.clamp(0, maximum) as usize
            })
            .collect()
    }
}

impl Pad for Array {
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[usize],
        edge_padding_high: &[usize],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        let output_type =
            self.r#type.pad(&padding_value.r#type, edge_padding_low, edge_padding_high, interior_padding)?;
        let input_shape = self.r#type.static_shape().unwrap();
        let output_shape = output_type.static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let rank = input_shape.rank();
        let mut values = vec![padding_value.values[0]; Self::element_count(&output_type)];
        let mut input_index = vec![0usize; rank];
        let mut written = 0usize;
        while written < self.values.len() {
            let mut output_flat = 0usize;
            for axis in 0..rank {
                output_flat +=
                    (edge_padding_low[axis] + input_index[axis] * (interior_padding[axis] + 1)) * output_strides[axis];
            }
            values[output_flat] = self.values[written];
            written += 1;
            for position in (0..rank).rev() {
                input_index[position] += 1;
                if input_index[position] < input_shape[position] {
                    break;
                }
                input_index[position] = 0;
            }
        }
        Ok(Self { r#type: output_type, values })
    }
}

impl Concatenate for Array {
    fn concatenate(operands: &[Self], axis: usize) -> Result<Self, ProgramError> {
        let operand_types: Vec<ArrayType> = operands.iter().map(|operand| operand.r#type.clone()).collect();
        let output_type = ArrayType::concatenate(&operand_types, axis)?;
        // Each operand owns a contiguous run of `axis` coordinates; writing its block at the running offset along
        // `axis` (and offset zero on every other axis) into a zero-initialized output reuses the row-major
        // odometer in `replace_block`.
        let zero = Self::zero_element(output_type.data_type())?;
        let mut output = Self { r#type: output_type.clone(), values: vec![zero; Self::element_count(&output_type)] };
        let mut offset = 0usize;
        for operand in operands {
            let operand_axis_size = operand.r#type.static_shape().unwrap()[axis];
            let mut start_indices = vec![0usize; output_type.rank()];
            start_indices[axis] = offset;
            output = output.replace_block(operand, start_indices.as_slice());
            offset += operand_axis_size;
        }
        Ok(output)
    }
}

impl Gather for Array {
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError> {
        let output_type = self.r#type.gather(&indices.r#type, operation)?;
        let dimensions = operation.dimensions();
        let slice_sizes = operation.slice_sizes();
        let operand_shape = self.r#type.static_shape().unwrap();
        let operand_strides = operand_shape.row_major_strides();
        let indices_shape = indices.r#type.static_shape().unwrap();
        let indices_strides = indices_shape.row_major_strides();
        let output_shape = output_type.static_shape().unwrap();
        let operand_rank = operand_shape.rank();
        let indices_rank = indices_shape.rank();
        let output_rank = output_shape.rank();
        let index_vector_dimension = indices_rank - 1;
        let index_vector_extent = indices_shape[index_vector_dimension];

        // Classify operand axes (window axes carry the slice; collapsed/batching do not) and output axes (offset
        // positions carry the window, the rest carry the indices' batch coordinates).
        let collapsed: BTreeSet<usize> = dimensions.collapsed_slice_dimensions().iter().copied().collect();
        let batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        let operand_window_axes: Vec<usize> =
            (0..operand_rank).filter(|axis| !collapsed.contains(axis) && !batching.contains(axis)).collect();
        let offset_positions: BTreeSet<usize> = dimensions.offset_dimensions().iter().copied().collect();
        let batch_output_positions: Vec<usize> =
            (0..output_rank).filter(|position| !offset_positions.contains(position)).collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();

        let dropped_fill = Self::zero_element(output_type.data_type())?;
        let extents = output_shape.dimensions();
        let output_count: usize = extents.iter().product();
        let mut values = Vec::with_capacity(output_count);
        let mut output_index = vec![0usize; output_rank];
        for _ in 0..output_count {
            // Place the output's batch coordinates into the indices multi-index and read this query's start vector.
            let mut indices_index = vec![0usize; indices_rank];
            for (position, &output_position) in batch_output_positions.iter().enumerate() {
                indices_index[indices_batch_axes[position]] = output_index[output_position];
            }
            let mut starts = vec![0i64; index_vector_extent];
            for (component, start) in starts.iter_mut().enumerate() {
                indices_index[index_vector_dimension] = component;
                let flat: usize = (0..indices_rank).map(|axis| indices_index[axis] * indices_strides[axis]).sum();
                *start = Self::index_value(indices.values[flat]);
            }
            // Assemble the operand multi-index: window offsets, then batching coordinates, then start offsets.
            let mut operand_index = vec![0i64; operand_rank];
            for (window, &operand_axis) in operand_window_axes.iter().enumerate() {
                operand_index[operand_axis] = output_index[dimensions.offset_dimensions()[window]] as i64;
            }
            for (batch, &operand_axis) in dimensions.operand_batching_dimensions().iter().enumerate() {
                operand_index[operand_axis] =
                    indices_index[dimensions.start_indices_batching_dimensions()[batch]] as i64;
            }
            let mut dropped = false;
            for (component, &operand_axis) in dimensions.start_index_map().iter().enumerate() {
                let raw = starts[component];
                let maximum = (operand_shape[operand_axis] - slice_sizes[operand_axis]) as i64;
                match operation.mode() {
                    GatherScatterMode::FillOrDrop => {
                        if raw < 0 || raw > maximum {
                            dropped = true;
                        }
                        operand_index[operand_axis] += raw;
                    }
                    GatherScatterMode::PromiseInBounds | GatherScatterMode::Clip => {
                        operand_index[operand_axis] += raw.clamp(0, maximum)
                    }
                }
            }
            let value = if dropped {
                dropped_fill
            } else {
                let flat: usize =
                    (0..operand_rank).map(|axis| operand_index[axis] as usize * operand_strides[axis]).sum();
                self.values[flat]
            };
            values.push(value);
            for position in (0..output_rank).rev() {
                output_index[position] += 1;
                if output_index[position] < extents[position] {
                    break;
                }
                output_index[position] = 0;
            }
        }
        Ok(Self { r#type: output_type, values })
    }
}

impl Scatter for Array {
    fn scatter(&self, indices: &Self, updates: &Self, operation: &ScatterOperation) -> Result<Self, ProgramError> {
        let output_type = self.r#type.scatter(&indices.r#type, &updates.r#type, operation)?;
        let dimensions = operation.dimensions();
        let operand_shape = self.r#type.static_shape().unwrap();
        let operand_strides = operand_shape.row_major_strides();
        let indices_shape = indices.r#type.static_shape().unwrap();
        let indices_strides = indices_shape.row_major_strides();
        let updates_shape = updates.r#type.static_shape().unwrap();
        let operand_rank = operand_shape.rank();
        let indices_rank = indices_shape.rank();
        let updates_rank = updates_shape.rank();
        let index_vector_dimension = indices_rank - 1;
        let index_vector_extent = indices_shape[index_vector_dimension];

        let inserted: BTreeSet<usize> = dimensions.inserted_window_dimensions().iter().copied().collect();
        let batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        let operand_window_axes: Vec<usize> =
            (0..operand_rank).filter(|axis| !inserted.contains(axis) && !batching.contains(axis)).collect();
        let update_window: BTreeSet<usize> = dimensions.update_window_dimensions().iter().copied().collect();
        let update_scatter_axes: Vec<usize> = (0..updates_rank).filter(|axis| !update_window.contains(axis)).collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();
        // Window size per operand axis (the update extent on window axes, 1 elsewhere), used to clamp the start so the
        // whole window stays in bounds.
        let mut operand_window_size = vec![1usize; operand_rank];
        for (window, &operand_axis) in operand_window_axes.iter().enumerate() {
            operand_window_size[operand_axis] = updates_shape[dimensions.update_window_dimensions()[window]];
        }

        let mut values = self.values.clone();
        let extents = updates_shape.dimensions();
        let update_count: usize = extents.iter().product();
        let mut update_index = vec![0usize; updates_rank];
        for written in 0..update_count {
            let mut indices_index = vec![0usize; indices_rank];
            for (position, &update_axis) in update_scatter_axes.iter().enumerate() {
                indices_index[indices_batch_axes[position]] = update_index[update_axis];
            }
            let mut starts = vec![0i64; index_vector_extent];
            for (component, start) in starts.iter_mut().enumerate() {
                indices_index[index_vector_dimension] = component;
                let flat: usize = (0..indices_rank).map(|axis| indices_index[axis] * indices_strides[axis]).sum();
                *start = Self::index_value(indices.values[flat]);
            }
            let mut operand_index = vec![0i64; operand_rank];
            for (window, &operand_axis) in operand_window_axes.iter().enumerate() {
                operand_index[operand_axis] = update_index[dimensions.update_window_dimensions()[window]] as i64;
            }
            for (batch, &operand_axis) in dimensions.operand_batching_dimensions().iter().enumerate() {
                operand_index[operand_axis] =
                    indices_index[dimensions.scatter_indices_batching_dimensions()[batch]] as i64;
            }
            let mut dropped = false;
            for (component, &operand_axis) in dimensions.scatter_dimensions_to_operand_dimensions().iter().enumerate() {
                let raw = starts[component];
                let maximum = (operand_shape[operand_axis] - operand_window_size[operand_axis]) as i64;
                match operation.mode() {
                    GatherScatterMode::FillOrDrop => {
                        if raw < 0 || raw > maximum {
                            dropped = true;
                        }
                        operand_index[operand_axis] += raw;
                    }
                    GatherScatterMode::PromiseInBounds | GatherScatterMode::Clip => {
                        operand_index[operand_axis] += raw.clamp(0, maximum)
                    }
                }
            }
            if !dropped {
                let flat: usize =
                    (0..operand_rank).map(|axis| operand_index[axis] as usize * operand_strides[axis]).sum();
                values[flat] = combine_scatter(operation.kind(), values[flat], updates.values[written]);
            }
            for position in (0..updates_rank).rev() {
                update_index[position] += 1;
                if update_index[position] < extents[position] {
                    break;
                }
                update_index[position] = 0;
            }
        }
        Ok(Self { r#type: output_type, values })
    }
}

/// Combines an existing operand element with a scattered update element under the given [`ScatterReductionKind`],
/// panicking (via the [`Scalar`] arithmetic sugar) for element data types that do not support the requested
/// combination, which the type-level scatter validation rules out before payload access.
fn combine_scatter(kind: ScatterReductionKind, current: Scalar, update: Scalar) -> Scalar {
    match kind {
        ScatterReductionKind::Overwrite => update,
        ScatterReductionKind::Add => current + update,
        ScatterReductionKind::Mul => current * update,
        ScatterReductionKind::Min => extremum(current, update, ComparisonDirection::LessThan),
        ScatterReductionKind::Max => extremum(current, update, ComparisonDirection::GreaterThan),
    }
}

impl Slice for Array {
    fn slice(&self, start_indices: &[usize], limit_indices: &[usize], strides: &[usize]) -> Result<Self, ProgramError> {
        let output_type = self.r#type.slice(start_indices, limit_indices, strides)?;
        let sizes: Vec<usize> = start_indices
            .iter()
            .zip(limit_indices.iter())
            .zip(strides.iter())
            .map(|((start, limit), stride)| (limit - start).div_ceil(*stride))
            .collect();
        let values = self.copy_block(start_indices, strides, sizes.as_slice());
        Ok(Self { r#type: output_type, values })
    }
}

impl UpdateSlice for Array {
    fn update_slice(&self, update: &Self, start_indices: &[usize]) -> Result<Self, ProgramError> {
        self.r#type.update_slice(&update.r#type, start_indices)?;
        Ok(self.clone().replace_block(update, start_indices))
    }
}

impl DynamicSlice for Array {
    fn dynamic_slice(&self, start_indices: &[Self], sizes: &[usize]) -> Result<Self, ProgramError> {
        let index_types: Vec<ArrayType> = start_indices.iter().map(|index| index.r#type.clone()).collect();
        let output_type = self.r#type.dynamic_slice(&index_types, sizes)?;
        let input_shape = self.r#type.static_shape().unwrap();
        let starts = Self::clamped_start_indices(start_indices, &input_shape, sizes);
        let unit_strides = vec![1; sizes.len()];
        let values = self.copy_block(starts.as_slice(), unit_strides.as_slice(), sizes);
        Ok(Self { r#type: output_type, values })
    }
}

impl DynamicUpdateSlice for Array {
    fn dynamic_update_slice(&self, update: &Self, start_indices: &[Self]) -> Result<Self, ProgramError> {
        let index_types: Vec<ArrayType> = start_indices.iter().map(|index| index.r#type.clone()).collect();
        self.r#type.dynamic_update_slice(&update.r#type, &index_types)?;
        let input_shape = self.r#type.static_shape().unwrap();
        let update_shape = update.r#type.static_shape().unwrap();
        let starts = Self::clamped_start_indices(start_indices, &input_shape, update_shape.dimensions());
        Ok(self.clone().replace_block(update, starts.as_slice()))
    }
}

impl Compare for Array {
    type Output = Self;

    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self::Output, ProgramError> {
        // Broadcast the operand types together (including element-type promotion) so mixed-precision comparisons
        // mirror the `CompareOperation` type-inference contract, then compare the promoted elements pairwise. The
        // output type is the Boolean-typed counterpart of the broadcast type.
        let broadcast_type = Broadcastable::broadcast(&self.r#type, &rhs.r#type)
            .map_err(|error| TypeError { message: error.to_string() })?;
        let target = broadcast_type.data_type();
        let output_len = Self::element_count(&broadcast_type);
        let left = self.broadcast_values(output_len);
        let right = rhs.broadcast_values(output_len);
        let values = left
            .iter()
            .zip(right.iter())
            .map(|(left, right)| {
                left.convert_element_type(target)?.compare(&right.convert_element_type(target)?, direction)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { r#type: broadcast_type.as_boolean(), values })
    }
}

impl Select for Array {
    type Condition = Self;

    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        // Mirrors the broadcasting `SelectOperation` type-inference contract: the condition must be Boolean-typed,
        // the three operand shapes broadcast together, and the two branch data types promote together to the output
        // data type. The condition is retyped to a branch data type before broadcasting so its Boolean data type
        // acts as a mask rather than promoting into the output.
        assert_eq!(condition.r#type.data_type(), DataType::Boolean, "select condition must have a Boolean data type",);
        let output_type = Broadcastable::broadcast(
            &Broadcastable::broadcast(
                &condition.r#type.clone().with_data_type(on_true.r#type.data_type()),
                &on_true.r#type,
            )
            .map_err(|error| TypeError { message: error.to_string() })?,
            &on_false.r#type,
        )
        .map_err(|error| TypeError { message: error.to_string() })?;
        let output_len = Self::element_count(&output_type);
        let condition = condition.broadcast_values(output_len);
        let on_true = on_true.broadcast_values(output_len);
        let on_false = on_false.broadcast_values(output_len);
        let values = condition
            .iter()
            .zip(on_true.iter())
            .zip(on_false.iter())
            .map(|((condition, on_true), on_false)| Scalar::select(&condition.boolean()?, on_true, on_false))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { r#type: output_type, values })
    }
}

impl SelectCondition for Array {
    type Condition = Self;

    fn select_condition(&self) -> Result<Self, ProgramError> {
        Ok(self.clone())
    }
}

impl BooleanLike for Array {
    /// Returns an [`Array`] with a Boolean-typed counterpart of this array's type and with every element
    /// reinterpreted as Boolean through the elementwise [`Scalar`] conversion (i.e., zero maps to `false` and any
    /// nonzero element maps to `true`).
    fn as_boolean(&self) -> Self {
        Self { r#type: self.r#type.as_boolean(), values: self.values.iter().map(|value| value.as_boolean()).collect() }
    }

    fn boolean(&self) -> Result<bool, ProgramError> {
        // Accept scalar Boolean predicates (rank-0, one element) so that batch-varying while can extract a final
        // `any(mask)` result. Higher-rank predicates still error because they cannot collapse to a single Boolean.
        if self.r#type.rank() == 0 && self.r#type.data_type() == DataType::Boolean && self.values.len() == 1 {
            return self.values[0].boolean();
        }
        Err(ProgramError::Concretization {
            message: format!(
                "cannot extract a concrete boolean from a value of type {}; expected bool[]",
                self.r#type()
            ),
        })
    }
}

/// Batched while-predicate semantics for [`Array`]: `any_true` reduces the whole Boolean payload with `or`, and
/// `mask_select` broadcasts the predicate against the operands along its leading (prefix) axes, so predicate item `i`
/// masks the contiguous per-item block of `on_true` / `on_false` elements it governs.
impl crate::operations::control_flow::WhilePredicate for Array {
    fn any_true(&self) -> Result<bool, ProgramError> {
        if self.r#type.data_type() != DataType::Boolean {
            return Err(ProgramError::Concretization {
                message: format!("cannot use a value of type {} as a Boolean while predicate", self.r#type),
            });
        }
        for value in &self.values {
            if value.boolean()? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        if self.r#type.data_type() != DataType::Boolean
            || on_true.r#type != on_false.r#type
            || self.values.is_empty()
            || !on_true.values.len().is_multiple_of(self.values.len())
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "mask_select requires a Boolean predicate whose element count divides congruent operands, but \
                     got predicate {} with operands {} and {}",
                    self.r#type, on_true.r#type, on_false.r#type,
                ),
            });
        }
        let block = on_true.values.len() / self.values.len();
        let values = on_true
            .values
            .iter()
            .zip(on_false.values.iter())
            .enumerate()
            .map(|(index, (on_true, on_false))| {
                Ok(if self.values[index / block].boolean()? { *on_true } else { *on_false })
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        Ok(Self { r#type: on_true.r#type.clone(), values })
    }
}

impl ConvertElementType for Array {
    fn convert_element_type(&self, data_type: DataType) -> Result<Self, ProgramError> {
        if self.r#type.data_type() == DataType::Token || data_type == DataType::Token {
            return Err(
                TypeError { message: "cannot convert values to or from the token data type".to_string() }.into()
            );
        }
        let values = self
            .values
            .iter()
            .map(|value| value.convert_element_type(data_type))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { r#type: self.r#type.clone().with_data_type(data_type), values })
    }
}

impl TransferToMemory for Array {
    /// Re-places this [`Array`] in `destination` by updating the [`Memory`](crate::types::Memory) carried by its
    /// type. The payload is host-resident either way, but the carried type must reflect the transfer so that staged
    /// programs whose declared types park values in other memories (e.g., offloaded residuals) accept the
    /// interpreted value.
    #[inline]
    fn transfer_to_memory(&self, destination: crate::types::Memory) -> Self {
        Self { r#type: self.r#type.clone().with_memory(destination), values: self.values.clone() }
    }
}

// An `Array` is a concrete single-device value, so resharding is a no-op on its payload. Its type still records the
// requested distribution metadata — mirroring the `ReshardOperation` type-inference rule, which carries the input's
// varying manual axes over to the target sharding — so interpreted programs preserve their declared boundaries
// exactly. The infallible capability signature makes an invalid target sharding a panic rather than an error, which
// the type-level validation performed before interpretation rules out for staged programs.
impl crate::operations::sharding::Reshard for Array {
    fn reshard(&self, sharding: &crate::Sharding) -> Self {
        let varying_manual_axes =
            self.r#type.sharding().map(|sharding| sharding.varying_manual_axes().clone()).unwrap_or_default();
        let sharding = sharding
            .clone()
            .with_varying_manual_axes(varying_manual_axes)
            .unwrap_or_else(|error| panic!("{error}"));
        let r#type = self.r#type.clone().with_sharding(sharding).unwrap_or_else(|error| panic!("{error}"));
        Self { r#type, values: self.values.clone() }
    }
}

// The sharding-constraint hint is untracked: the output type (sharding included) is identical to the input, so the
// identity default is exactly the `ShardingConstraintOperation` interpretation contract for a concrete value.
impl crate::operations::sharding::ConstrainSharding for Array {}

impl Tag for Array {
    #[inline]
    fn tag(self, _key: &str) -> Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::operations::complex::Complex;
    use crate::operations::constants::Iota;
    use crate::operations::manipulation::{GatherDimensionNumbers, ScatterDimensionNumbers};
    use crate::operations::sharding::{ConstrainSharding, Reshard};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::Memory;

    use super::*;

    /// Creates a static [`ArrayType`] with the provided element data type and dimension sizes.
    fn array_type(data_type: DataType, dimensions: &[usize]) -> ArrayType {
        ArrayType::new(data_type, Shape::new(dimensions.iter().map(|size| Size::Static(*size)).collect()))
    }

    #[test]
    fn test_array_new_enforces_payload_invariants() {
        // Element data types must match the declared element data type.
        assert!(matches!(
            Array::new(array_type(DataType::F64, &[2]), vec![Scalar::F64(1.0), Scalar::F32(2.0)]),
            Err(ProgramError::Type(TypeError { message }))
                if message == "array of element data type f64 cannot store an element of data type f32",
        ));
        // The payload length must match the static element count.
        assert!(matches!(
            Array::new(array_type(DataType::F64, &[3]), vec![Scalar::F64(1.0)]),
            Err(ProgramError::Type(TypeError { message }))
                if message == "array type f64[3] requires 3 elements but got 1",
        ));
        // Dynamically shaped types cannot describe a materialized payload.
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)]));
        assert!(matches!(
            Array::new(dynamic_type, vec![Scalar::F64(1.0)]),
            Err(ProgramError::Type(TypeError { message }))
                if message == "cannot materialize a value of dynamically sized type f64[*]",
        ));
        // A well-formed payload constructs successfully and round-trips through the accessors.
        let array = Array::new(array_type(DataType::F64, &[2]), vec![Scalar::F64(1.0), Scalar::F64(2.0)]).unwrap();
        assert_eq!(array.r#type().into_owned(), array_type(DataType::F64, &[2]));
        assert_eq!(array.values(), &[Scalar::F64(1.0), Scalar::F64(2.0)]);
    }

    #[test]
    fn test_array_convenience_constructors() {
        // The scalar, vector, and matrix constructors infer the element data type from their payloads.
        assert_eq!(Array::scalar(2.5).r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(Array::vector(vec![1.0f32, 2.0]).r#type().into_owned(), array_type(DataType::F32, &[2]));
        assert_eq!(Array::vector(vec![true, false]).r#type().into_owned(), array_type(DataType::Boolean, &[2]));
        assert_eq!(Array::matrix(2, 2, vec![1, 2, 3, 4]).r#type().into_owned(), array_type(DataType::I32, &[2, 2]));
        // An empty vector defaults to `f64`.
        assert_eq!(Array::vector(Vec::<f64>::new()).r#type().into_owned(), array_type(DataType::F64, &[0]));
        // `from_f64s` converts the payload into the declared element data type, including exact low-precision
        // floating-point encodings (1.5 is representable in `f8e4m3fn` as `0x3c`).
        let array = Array::from_f64s(array_type(DataType::F8E4M3FN, &[2]), vec![1.5, -1.5]);
        assert_eq!(array.values()[0].low_precision_float_bits(), Some(0x3c));
        assert_eq!(array.values()[1].low_precision_float_bits(), Some(0xbc));
        let array = Array::from_f64s(array_type(DataType::I32, &[2]), vec![1.0, -2.0]);
        assert_eq!(array.values(), &[Scalar::I32(1), Scalar::I32(-2)]);
    }

    #[test]
    fn test_array_to_f64s() {
        assert_eq!(Array::vector(vec![1.5, 2.5]).to_f64s(), vec![1.5, 2.5]);
        assert_eq!(Array::vector(vec![true, false]).to_f64s(), vec![1.0, 0.0]);
        assert_eq!(Array::vector(vec![1i32, -2]).to_f64s(), vec![1.0, -2.0]);
        // Low-precision floating-point elements decode to the exact values they denote.
        assert_eq!(Array::from_f64s(array_type(DataType::F8E4M3FN, &[1]), vec![1.5]).to_f64s(), vec![1.5]);
    }

    #[test]
    #[should_panic(expected = "cannot view an array of complex element data type c128 as f64 values")]
    fn test_array_to_f64s_rejects_complex_arrays() {
        let real = Array::vector(vec![1.0, 2.0]);
        let imaginary = Array::vector(vec![3.0, 4.0]);
        let _ = real.complex(&imaginary).unwrap().to_f64s();
    }

    #[test]
    fn test_array_display() {
        // Real floating-point payloads keep a decimal point (matching how `Vec<f64>` debug-formats), while other
        // payloads use the scalar rendering.
        assert_eq!(Array::vector(vec![1.0, 2.5]).to_string(), "[1.0, 2.5]");
        assert_eq!(Array::vector(vec![1i32, 2]).to_string(), "[1, 2]");
        assert_eq!(Array::vector(vec![true, false]).to_string(), "[true, false]");
        let complex = Array::vector(vec![1.0]).complex(&Array::vector(vec![2.0])).unwrap();
        assert_eq!(complex.to_string(), "[1+2i]");
        assert_eq!(Array::vector(Vec::<f64>::new()).to_string(), "[]");
    }

    #[test]
    fn test_array_equality() {
        // Exact equality requires identical types and elementwise-equal payloads.
        assert_eq!(Array::vector(vec![1.0, 2.0]), Array::vector(vec![1.0, 2.0]));
        assert_ne!(Array::vector(vec![1.0, 2.0]), Array::vector(vec![1.0, 2.5]));
        assert_ne!(Array::vector(vec![1.0f32]), Array::vector(vec![1.0f64]));
        // Low-precision floating-point elements compare through their decoded values, so signed zeros compare equal.
        let positive_zero = Array::from_f64s(array_type(DataType::F8E4M3FN, &[1]), vec![0.0]);
        let negative_zero = Array::from_f64s(array_type(DataType::F8E4M3FN, &[1]), vec![-0.0]);
        assert_eq!(positive_zero, negative_zero);
        // Approximate equality delegates to the elementwise scalar approximation.
        assert_abs_diff_eq!(Array::vector(vec![1.0, 2.0]), Array::vector(vec![1.0 + 1e-10, 2.0]), epsilon = 1e-9);
        let left = Array::vector(vec![1.0]).complex(&Array::vector(vec![2.0])).unwrap();
        let right = Array::vector(vec![1.0 + 1e-10]).complex(&Array::vector(vec![2.0])).unwrap();
        assert_abs_diff_eq!(left, right, epsilon = 1e-9);
    }

    #[test]
    fn test_array_constants() {
        let context = EagerContext::<Array>::new();
        let r#type = array_type(DataType::F32, &[2, 2]);
        assert_eq!(
            context.zero(&r#type),
            Array::new(r#type.clone(), vec![Scalar::F32(0.0); 4]).map_err(|_| unreachable!())
        );
        assert_eq!(
            context.one(&r#type),
            Array::new(r#type.clone(), vec![Scalar::F32(1.0); 4]).map_err(|_| unreachable!())
        );
        assert_eq!(
            context.fill(&r#type, Scalar::F32(2.5)),
            Array::new(r#type.clone(), vec![Scalar::F32(2.5); 4]).map_err(|_| unreachable!()),
        );
        // The fill element must promote into the array's element data type, so narrowing and complex-to-real fills
        // are rejected instead of silently discarding payload.
        assert!(context.fill(&r#type, Scalar::F64(2.5)).is_err());
        assert!(context.fill(&r#type, Scalar::C64(ComplexNumber::new(1.0, 2.0))).is_err());
        // Iota materializes coordinates along the requested dimension in the declared element data type.
        assert_eq!(
            context.iota(&array_type(DataType::I32, &[2, 3]), 1).unwrap().values(),
            &[Scalar::I32(0), Scalar::I32(1), Scalar::I32(2), Scalar::I32(0), Scalar::I32(1), Scalar::I32(2)],
        );
        assert_eq!(context.iota(&array_type(DataType::F64, &[3]), 0).unwrap().to_f64s(), vec![0.0, 1.0, 2.0]);
        // Kernels that materialize a payload from a type reject dynamically sized types.
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(3)]));
        let expected_message = "cannot materialize a value of dynamically sized type f64[*, 3]";
        assert!(matches!(
            context.zero(&dynamic_type),
            Err(ProgramError::Type(TypeError { message })) if message == expected_message,
        ));
        assert!(matches!(
            context.one(&dynamic_type),
            Err(ProgramError::Type(TypeError { message })) if message == expected_message,
        ));
        assert!(matches!(
            context.fill(&dynamic_type, Scalar::from(42.0)),
            Err(ProgramError::Type(TypeError { message })) if message == expected_message,
        ));
    }

    #[test]
    fn test_array_zero_like_and_one_like() {
        let array = Array::vector(vec![1.5f32, -2.5]);
        assert_eq!(array.zero_like().values(), &[Scalar::F32(0.0), Scalar::F32(0.0)]);
        assert_eq!(array.one_like().values(), &[Scalar::F32(1.0), Scalar::F32(1.0)]);
        assert_eq!(array.zero_like().r#type().into_owned(), array_type(DataType::F32, &[2]));
    }

    #[test]
    fn test_array_arithmetic() {
        // Elementwise arithmetic with scalar broadcasting.
        let vector = Array::vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(vector.add(&Array::scalar(1.0)).unwrap(), Array::vector(vec![2.0, 3.0, 4.0]));
        assert_eq!(vector.sub(&Array::vector(vec![0.5, 1.0, 1.5])).unwrap(), Array::vector(vec![0.5, 1.0, 1.5]));
        assert_eq!(vector.mul(&vector).unwrap(), Array::vector(vec![1.0, 4.0, 9.0]));
        assert_eq!(vector.div(&Array::scalar(2.0)).unwrap(), Array::vector(vec![0.5, 1.0, 1.5]));
        assert_eq!(vector.neg().unwrap(), Array::vector(vec![-1.0, -2.0, -3.0]));
        // Mixed-precision operands promote to the common element data type.
        let promoted = Array::vector(vec![1.0f32, 2.0]).add(&Array::vector(vec![0.5f64, 0.5])).unwrap();
        assert_eq!(promoted, Array::vector(vec![1.5f64, 2.5]));
        // The `std::ops` sugar delegates to the fallible capabilities.
        assert_eq!(vector.clone() + Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0]));
        assert_eq!(-vector.clone(), Array::vector(vec![-1.0, -2.0, -3.0]));
        // Scaling by an `f64` preserves the array's element data type.
        let scaled = Array::vector(vec![1.0f32, 2.0]) * 2.0;
        assert_eq!(scaled, Array::vector(vec![2.0f32, 4.0]));
        // Integer arithmetic wraps deterministically, matching the scalar reference backend.
        let wrapped = Array::vector(vec![Scalar::U8(255)]).add(&Array::vector(vec![Scalar::U8(1)])).unwrap();
        assert_eq!(wrapped.values(), &[Scalar::U8(0)]);
    }

    #[test]
    fn test_array_low_precision_float_arithmetic() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_f64s(array_type(DataType::F8E4M3FN, &[2]), vec![1.0, 2.0]);
        let right = Array::from_f64s(array_type(DataType::F8E4M3FN, &[2]), vec![0.5, 0.25]);
        let sum = left.add(&right).unwrap();
        assert_eq!(sum.r#type().into_owned(), array_type(DataType::F8E4M3FN, &[2]));
        assert_eq!(sum.to_f64s(), vec![1.5, 2.25]);
    }

    #[test]
    fn test_array_math() {
        let vector = Array::vector(vec![0.0, 1.0]);
        assert_abs_diff_eq!(vector.sin().unwrap(), Array::vector(vec![0.0, 1.0f64.sin()]), epsilon = 1e-12);
        assert_abs_diff_eq!(vector.cos().unwrap(), Array::vector(vec![1.0, 1.0f64.cos()]), epsilon = 1e-12);
        assert_abs_diff_eq!(vector.exp().unwrap(), Array::vector(vec![1.0, 1.0f64.exp()]), epsilon = 1e-12);
        assert_abs_diff_eq!(
            Array::vector(vec![1.0, 4.0]).sqrt().unwrap(),
            Array::vector(vec![1.0, 2.0]),
            epsilon = 1e-12,
        );
        assert_abs_diff_eq!(
            Array::vector(vec![1.0, std::f64::consts::E]).log().unwrap(),
            Array::vector(vec![0.0, 1.0]),
            epsilon = 1e-12,
        );
        assert_abs_diff_eq!(
            Array::vector(vec![1.0]).atan2(&Array::vector(vec![1.0])).unwrap(),
            Array::vector(vec![std::f64::consts::FRAC_PI_4]),
            epsilon = 1e-12,
        );
        assert_eq!(Array::vector(vec![-1.5, 2.5]).abs().unwrap(), Array::vector(vec![1.5, 2.5]));
        // The absolute value of a complex array is its elementwise magnitude with a real element data type.
        let complex = Array::vector(vec![3.0]).complex(&Array::vector(vec![4.0])).unwrap();
        let magnitude = complex.abs().unwrap();
        assert_eq!(magnitude.r#type().into_owned(), array_type(DataType::F64, &[1]));
        assert_abs_diff_eq!(magnitude, Array::vector(vec![5.0]), epsilon = 1e-12);
    }

    #[test]
    fn test_array_complex_parts() {
        let real = Array::vector(vec![1.0, 2.0]);
        let imaginary = Array::vector(vec![3.0, -4.0]);
        let complex = real.complex(&imaginary).unwrap();
        assert_eq!(complex.r#type().into_owned(), array_type(DataType::C128, &[2]));
        assert_eq!(
            complex.values(),
            &[Scalar::C128(ComplexNumber::new(1.0, 3.0)), Scalar::C128(ComplexNumber::new(2.0, -4.0))],
        );
        assert_eq!(complex.real().unwrap(), real);
        assert_eq!(complex.imaginary().unwrap(), imaginary);
        let conjugate = complex.conjugate().unwrap();
        assert_eq!(conjugate.imaginary().unwrap(), imaginary.neg().unwrap());
        // Complex construction requires identical part types.
        assert!(real.complex(&Array::vector(vec![1.0f32])).is_err());
        assert!(Array::vector(vec![1i32]).complex(&Array::vector(vec![2i32])).is_err());
    }

    #[test]
    fn test_array_logical_operations() {
        let left = Array::vector(vec![true, true, false, false]);
        let right = Array::vector(vec![true, false, true, false]);
        assert_eq!(left.and(&right).unwrap(), Array::vector(vec![true, false, false, false]));
        assert_eq!(left.or(&right).unwrap(), Array::vector(vec![true, true, true, false]));
        assert_eq!(left.xor(&right).unwrap(), Array::vector(vec![false, true, true, false]));
        assert_eq!(left.not().unwrap(), Array::vector(vec![false, false, true, true]));
        // Same-data-type integer operands combine bitwise.
        let bits = Array::vector(vec![0b1100u8]).and(&Array::vector(vec![0b1010u8])).unwrap();
        assert_eq!(bits.values(), &[Scalar::U8(0b1000)]);
        // Real floating-point operands are rejected, matching the scalar reference backend.
        assert!(Array::vector(vec![1.0]).and(&Array::vector(vec![0.0])).is_err());
        // The `std::ops` sugar delegates to the fallible capabilities.
        assert_eq!(left.clone() & right.clone(), Array::vector(vec![true, false, false, false]));
        assert_eq!(!left.clone(), Array::vector(vec![false, false, true, true]));
    }

    #[test]
    fn test_array_compare() {
        let left = Array::vector(vec![1.0, 2.0, 3.0]);
        let right = Array::vector(vec![2.0, 2.0, 2.0]);
        let less_than = left.compare(&right, ComparisonDirection::LessThan).unwrap();
        assert_eq!(less_than.r#type().into_owned(), array_type(DataType::Boolean, &[3]));
        assert_eq!(less_than, Array::vector(vec![true, false, false]));
        // Operands broadcast and promote before comparing.
        let mixed = Array::vector(vec![1.0f32, 3.0]).compare(&Array::scalar(2.0f64), ComparisonDirection::GreaterThan);
        assert_eq!(mixed.unwrap(), Array::vector(vec![false, true]));
    }

    #[test]
    fn test_array_convert_element_type() {
        let vector = Array::vector(vec![0.0, 1.5]);
        assert_eq!(vector.convert_element_type(DataType::Boolean).unwrap(), Array::vector(vec![false, true]));
        assert_eq!(vector.convert_element_type(DataType::I32).unwrap(), Array::vector(vec![0i32, 1]));
        // Conversions into low-precision floating-point element types produce exact encodings.
        let low_precision = vector.convert_element_type(DataType::F8E5M2).unwrap();
        assert_eq!(low_precision.values()[1].low_precision_float_bits(), Some(0x3e));
        // Conversions to or from the token data type are rejected.
        assert!(matches!(
            vector.convert_element_type(DataType::Token),
            Err(ProgramError::Type(TypeError { message }))
                if message == "cannot convert values to or from the token data type",
        ));
    }

    #[test]
    fn test_array_select() {
        let condition = Array::vector(vec![true, false, true]);
        let on_true = Array::vector(vec![1.0, 2.0, 3.0]);
        let on_false = Array::vector(vec![-1.0, -2.0, -3.0]);
        assert_eq!(Array::select(&condition, &on_true, &on_false).unwrap(), Array::vector(vec![1.0, -2.0, 3.0]));
        // The condition broadcasts against the branches, and the branch data types promote together.
        let broadcast =
            Array::select(&Array::scalar(true), &Array::vector(vec![1.0f32, 2.0]), &Array::vector(vec![-1.0f64, -2.0]))
                .unwrap();
        assert_eq!(broadcast, Array::vector(vec![1.0f64, 2.0]));
        assert_eq!(condition.select_condition().unwrap(), condition);
    }

    #[test]
    fn test_array_boolean_like() {
        let vector = Array::vector(vec![0.0, 2.5]);
        let boolean = vector.as_boolean();
        assert_eq!(boolean.r#type().into_owned(), array_type(DataType::Boolean, &[2]));
        assert_eq!(boolean, Array::vector(vec![false, true]));
        assert_eq!(Array::scalar(true).boolean(), Ok(true));
        assert!(Array::vector(vec![true, false]).boolean().is_err());
        assert!(Array::scalar(1.0).boolean().is_err());
    }

    #[test]
    fn test_array_while_predicate() {
        use crate::operations::control_flow::WhilePredicate;

        let predicate = Array::vector(vec![false, true]);
        assert_eq!(predicate.any_true(), Ok(true));
        assert_eq!(Array::vector(vec![false, false]).any_true(), Ok(false));
        assert!(Array::vector(vec![1.0]).any_true().is_err());
        // Predicate item `i` masks the contiguous per-item block of operand elements it governs.
        let on_true = Array::from_f64s(array_type(DataType::F64, &[2, 2]), vec![1.0, 2.0, 3.0, 4.0]);
        let on_false = Array::from_f64s(array_type(DataType::F64, &[2, 2]), vec![-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(
            predicate.mask_select(&on_true, &on_false).unwrap(),
            Array::from_f64s(array_type(DataType::F64, &[2, 2]), vec![-1.0, -2.0, 3.0, 4.0]),
        );
    }

    #[test]
    fn test_array_broadcast() {
        let vector = Array::vector(vec![1.0, 2.0]);
        let output_type = array_type(DataType::F64, &[3, 2]);
        let broadcast = Broadcast::broadcast(&vector, output_type.clone(), &[1]).unwrap();
        assert_eq!(broadcast.r#type().into_owned(), output_type);
        assert_eq!(broadcast.to_f64s(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_array_transpose() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = matrix.transpose([1, 0]).unwrap();
        assert_eq!(transposed.r#type().into_owned(), array_type(DataType::F64, &[3, 2]));
        assert_eq!(transposed.to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert!(matrix.transpose([0, 0]).is_err());
    }

    #[test]
    fn test_array_reshape() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = matrix.reshape(Shape::new(vec![Size::Static(3), Size::Static(2)])).unwrap();
        assert_eq!(reshaped.r#type().into_owned(), array_type(DataType::F64, &[3, 2]));
        assert_eq!(reshaped.to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(matrix.reshape(Shape::new(vec![Size::Static(4)])).is_err());
    }

    #[test]
    fn test_array_slicing() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(vector.slice(&[1], &[5], &[2]).unwrap(), Array::vector(vec![2.0, 4.0]));
        assert_eq!(
            vector.update_slice(&Array::vector(vec![10.0, 20.0]), &[1]).unwrap(),
            Array::vector(vec![1.0, 10.0, 20.0, 4.0, 5.0]),
        );
        // Dynamic start indices clamp so the block stays in bounds.
        let start = [Array::scalar(4i64)];
        assert_eq!(vector.dynamic_slice(&start, &[2]).unwrap(), Array::vector(vec![4.0, 5.0]));
        assert_eq!(
            vector.dynamic_update_slice(&Array::vector(vec![10.0, 20.0]), &start).unwrap(),
            Array::vector(vec![1.0, 2.0, 3.0, 10.0, 20.0]),
        );
    }

    #[test]
    fn test_array_pad() {
        let vector = Array::vector(vec![1.0, 2.0]);
        let padded = vector.pad(&Array::scalar(0.5), &[1], &[2], &[1]).unwrap();
        assert_eq!(padded, Array::vector(vec![0.5, 1.0, 0.5, 2.0, 0.5, 0.5]));
    }

    #[test]
    fn test_array_concatenate() {
        let concatenated = Array::concatenate(&[Array::vector(vec![1.0, 2.0]), Array::vector(vec![3.0])], 0).unwrap();
        assert_eq!(concatenated, Array::vector(vec![1.0, 2.0, 3.0]));
        let matrices = [Array::matrix(1, 2, vec![1.0, 2.0]), Array::matrix(2, 2, vec![3.0, 4.0, 5.0, 6.0])];
        let concatenated = Array::concatenate(&matrices, 0).unwrap();
        assert_eq!(concatenated.r#type().into_owned(), array_type(DataType::F64, &[3, 2]));
        assert_eq!(concatenated.to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_array_gather() {
        // Gather rows 2 and 0 of a 3x2 matrix.
        let operand = Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indices = Array::from_f64s(array_type(DataType::I64, &[2, 1]), vec![2.0, 0.0]);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let gathered = operand.gather(&indices, &operation).unwrap();
        assert_eq!(gathered.r#type().into_owned(), array_type(DataType::F64, &[2, 2]));
        assert_eq!(gathered.to_f64s(), vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn test_array_scatter() {
        // Scatter-add updates 10 and 20 into elements 3 and 0 of a vector.
        let operand = Array::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let indices = Array::from_f64s(array_type(DataType::I64, &[2, 1]), vec![3.0, 0.0]);
        let updates = Array::vector(vec![10.0, 20.0]);
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        let scattered = operand.scatter(&indices, &updates, &operation).unwrap();
        assert_eq!(scattered, Array::vector(vec![21.0, 2.0, 3.0, 14.0]));
    }

    #[test]
    fn test_array_reduce() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(matrix.reduce(&[1], ReductionKind::Sum), Array::vector(vec![6.0, 15.0]));
        assert_eq!(matrix.reduce(&[1], ReductionKind::Mean), Array::vector(vec![2.0, 5.0]));
        assert_eq!(
            matrix.reduce(&[0, 1], ReductionKind::Sum),
            Array::from_f64s(array_type(DataType::F64, &[]), vec![21.0])
        );
        assert_eq!(matrix.reduce(&[], ReductionKind::Sum), matrix);
        // Max and min work for element data types without infinities because they do not materialize an identity.
        let integers = Array::vector(vec![3i32, -1, 2]);
        assert_eq!(integers.reduce(&[0], ReductionKind::Max).values(), &[Scalar::I32(3)]);
        assert_eq!(integers.reduce(&[0], ReductionKind::Min).values(), &[Scalar::I32(-1)]);
        // Boolean reductions.
        let booleans = Array::vector(vec![true, false, true]);
        assert_eq!(booleans.reduce(&[0], ReductionKind::Any).values(), &[Scalar::Bool(true)]);
        assert_eq!(booleans.reduce(&[0], ReductionKind::All).values(), &[Scalar::Bool(false)]);
    }

    #[test]
    fn test_array_dot() {
        let lhs = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = Array::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let dimensions = DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]);
        let product = lhs.dot(&rhs, &dimensions);
        assert_eq!(product.r#type().into_owned(), array_type(DataType::F64, &[2, 2]));
        assert_eq!(product.to_f64s(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_array_complex_math() {
        // Elementwise math over complex arrays delegates to the scalar reference backend, so complex semantics come
        // out of the same kernels as `Scalar` and only the shape logic is array-specific.
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]);
        let right = Array::vector(vec![ComplexNumber::new(0.5f64, -1.0), ComplexNumber::new(2.0f64, 0.5)]);
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        let right_values = [ComplexNumber::new(0.5f64, -1.0), ComplexNumber::new(2.0f64, 0.5)];
        let expect = |values: [ComplexNumber<f64>; 2]| Array::vector(values.to_vec());
        assert_eq!(
            left.add(&right).unwrap(),
            expect([left_values[0] + right_values[0], left_values[1] + right_values[1]]),
        );
        assert_eq!(
            left.sub(&right).unwrap(),
            expect([left_values[0] - right_values[0], left_values[1] - right_values[1]]),
        );
        assert_eq!(
            left.mul(&right).unwrap(),
            expect([left_values[0] * right_values[0], left_values[1] * right_values[1]]),
        );
        assert_abs_diff_eq!(
            left.div(&right).unwrap(),
            expect([left_values[0] / right_values[0], left_values[1] / right_values[1]]),
            epsilon = 1e-12,
        );
        assert_eq!(left.neg().unwrap(), expect([-left_values[0], -left_values[1]]));
        assert_abs_diff_eq!(left.exp().unwrap(), expect([left_values[0].exp(), left_values[1].exp()]), epsilon = 1e-12);
        assert_abs_diff_eq!(left.log().unwrap(), expect([left_values[0].ln(), left_values[1].ln()]), epsilon = 1e-12);
        assert_abs_diff_eq!(
            left.sqrt().unwrap(),
            expect([left_values[0].sqrt(), left_values[1].sqrt()]),
            epsilon = 1e-12,
        );
        assert_abs_diff_eq!(left.sin().unwrap(), expect([left_values[0].sin(), left_values[1].sin()]), epsilon = 1e-12);
        assert_abs_diff_eq!(left.cos().unwrap(), expect([left_values[0].cos(), left_values[1].cos()]), epsilon = 1e-12);
        // The absolute value is the elementwise magnitude with a real element data type.
        let magnitude = left.abs().unwrap();
        assert_eq!(magnitude.r#type().into_owned(), array_type(DataType::F64, &[2]));
        assert_abs_diff_eq!(
            magnitude,
            Array::vector(vec![left_values[0].norm(), left_values[1].norm()]),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_array_integer_semantics() {
        // Negation wraps deterministically for unsigned and two's-complement signed elements, matching the scalar
        // reference backend (and StableHLO's integer semantics), rather than panicking or saturating.
        let unsigned = Array::vector(vec![Scalar::U8(0), Scalar::U8(1), Scalar::U8(255)]);
        assert_eq!(unsigned.neg().unwrap().values(), &[Scalar::U8(0), Scalar::U8(255), Scalar::U8(1)]);
        let minimum = Array::vector(vec![Scalar::I8(i8::MIN), Scalar::I8(-5)]);
        assert_eq!(minimum.neg().unwrap().values(), &[Scalar::I8(i8::MIN), Scalar::I8(5)]);
        // Integer division by zero is a clean error rather than a panic.
        assert!(Array::vector(vec![1i32]).div(&Array::vector(vec![0i32])).is_err());
    }

    #[test]
    fn test_array_encoding_fidelity() {
        // Conversions and arithmetic on low-precision floating-point arrays operate on genuine encodings: the payload
        // round-trips through the exact bit patterns rather than an `f64` pun.
        let array = Array::from_f64s(array_type(DataType::F8E8M0FNU, &[2]), vec![2.0, 0.5]);
        assert_eq!(array.values()[0], Scalar::from_low_precision_float_bits(DataType::F8E8M0FNU, 0x80).unwrap());
        assert_eq!(array.values()[1], Scalar::from_low_precision_float_bits(DataType::F8E8M0FNU, 0x7e).unwrap());
        let converted = array.convert_element_type(DataType::BF16).unwrap();
        assert_eq!(converted.to_f64s(), vec![2.0, 0.5]);
        let round_trip = converted.convert_element_type(DataType::F8E8M0FNU).unwrap();
        assert_eq!(round_trip, array);
    }

    #[test]
    fn test_array_type_metadata_operations() {
        // The sharding, memory, and tagging operations alter only the carried type (or nothing at all): the payload
        // of a concrete single-device array is host-resident metadata-free storage either way.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();

        // Memory transfers re-place the array by updating the memory carried by its type.
        let array = Array::vector(vec![1.0, 2.0]);
        let transferred = array.transfer_to_memory(Memory::Host { pinned: true });
        assert_eq!(transferred.r#type().memory(), Memory::Host { pinned: true });
        assert_eq!(transferred.r#type().into_owned().with_memory(Memory::Device), array.r#type().into_owned());
        assert_eq!(transferred.values(), array.values());

        // Resharding records the requested distribution metadata on the type, carrying the input's varying manual
        // axes over to the target sharding exactly like the `ReshardOperation` type-inference rule.
        let input_sharding = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap();
        let input =
            Array::from_f64s(array_type(DataType::F64, &[2]).with_sharding(input_sharding).unwrap(), vec![1.0, 2.0]);
        let target = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = input.reshard(&target);
        assert_eq!(resharded.r#type().sharding(), Some(&target.clone().with_varying_manual_axes(["m"]).unwrap()),);
        assert_eq!(resharded.values(), input.values());

        // The sharding-constraint hint is untracked, so constraining leaves the value (type included) unchanged.
        assert_eq!(input.constrain_sharding(&target), input);
    }
}
