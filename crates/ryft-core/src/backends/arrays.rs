//! Contains the reference array backend that supports concrete [`Array`] values over the [`ArrayOperation`] family,
//! together with the [`ArrayTracingContext`] used to stage array [`Program`](crate::Program)s. This backend serves
//! programs whose values are dense multidimensional arrays typed by [`ArrayType`]. It is meant primarily for exercising
//! the Ryft tracing, transformation, and interpretation machinery without depending on an optimized backend such as
//! `ryft-xla`: unit tests, documentation tests, and downstream crates can stage, transform, and interpret complete
//! programs over arrays eagerly. [`Array`] stores one shared immutable byte buffer whose physical placement follows its
//! [`ArrayType`]. Checked typed codecs preserve exact element encodings, while the reference kernels implement the
//! corresponding arithmetic and shape semantics.
//!
//! # Warning
//!
//! This backend prioritizes transparency over performance. It supports the physical strided and tiled layouts carried
//! by [`ArrayType`], but operations materialize owned outputs rather than views and use straightforward reference
//! implementations rather than vectorized kernels. Do not use it outside tests, documentation examples, and
//! reference-semantics checks.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use approx::AbsDiffEq;
use half::{bf16, f16};
use num_complex::Complex;

use ryft_macros::{Operation, Parameter};

// TODO(eaplatanios): Review from here onwards.

use crate::arrays::encoding::{
    decode_elements, decode_logical_bytes, encode_elements, encode_logical_bytes, validate_storage_bytes,
};
use crate::arrays::macros::dispatch_on_array_element_type;
use crate::arrays::{ArrayAddressing, ArrayElement, ArraySliceAxis};
use crate::axes::{Axis, AxisIndexOperation};
use crate::backends::scalars::Scalar;
use crate::broadcasting::Broadcastable;
use crate::contexts::EagerContext;
use crate::differentiation::LinearCallOperation;
use crate::macros::check_count;
use crate::operations::attention::{
    AttentionMask, DotProductAttention, DotProductAttentionBackward, DotProductAttentionBackwardOperation,
    DotProductAttentionOperation, dot_product_attention_backward_composition, dot_product_attention_composition,
};
use crate::operations::collectives::{
    AllGatherOperation, AllToAllOperation, CollectiveOperation, PSumScatterOperation, PpermuteOperation,
};
use crate::operations::compare::{Compare, CompareOperation, ComparisonDirection};
use crate::operations::complex::{
    ComplexOperation, Conjugate, ConjugateOperation, Imaginary, ImaginaryOperation, Real, RealOperation,
};
use crate::operations::constants::{
    ConstantOperation, Fill, IOTA_OPERATION_NAME, IotaOperation, One, OneLike, OneLikeOperation, OneOperation, Zero,
    ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::{ConditionOperation, ScanOperation, Select, SelectOperation, WhileOperation};
use crate::operations::custom_call::{CustomCall, CustomCallOperation};
use crate::operations::debugging::PrintOperation;
use crate::operations::differentiation::{CoordinateBasisOperation, StopGradient, StopGradientOperation};
use crate::operations::dimensions::{DIMENSION_SIZE_OPERATION_NAME, DimensionSize};
use crate::operations::logical::{And, AndOperation, Not, NotOperation, Or, OrOperation, Xor, XorOperation};
use crate::operations::manipulation::conversion::ElementType;
use crate::operations::manipulation::{
    Concatenate, ConcatenateOperation, ConvertElementType, ConvertElementTypeOperation, DynamicSlice,
    DynamicSliceOperation, DynamicUpdateSlice, DynamicUpdateSliceOperation, Gather, GatherOperation, GatherScatterMode,
    LegacyBroadcast, LegacyBroadcastOperation, LegacyReshapeOperation, Pad, PadOperation, Permutation, Reshape,
    ReshapeParameters, Scatter, ScatterOperation, ScatterReductionKind, Slice, SliceOperation, Transpose,
    TransposeOperation, UpdateSlice, UpdateSliceOperation,
};
use crate::operations::math::dot::dot_general_evaluate;
use crate::operations::math::reduce::{reduce_abstract, reduce_evaluate};
use crate::operations::math::{
    Abs, AbsOperation, Add, AddOperation, Atan2, Atan2Operation, Ceil, CeilOperation, Cos, CosOperation, Div,
    DivOperation, Dot, DotDimensionNumbers, DotOperation, Erf, ErfOperation, Exp, ExpOperation, Floor, FloorOperation,
    Log, LogOperation, Logistic, LogisticOperation, Max, MaxOperation, Min, MinOperation, Mul, MulOperation, Neg,
    NegOperation, Pow, PowOperation, Reduce, ReduceOperation, ReductionKind, Rem, RemOperation, Round, RoundOperation,
    Rsqrt, RsqrtOperation, ScaledDot, ScaledDotOperation, Sign, SignOperation, Sin, SinOperation, Sqrt, SqrtOperation,
    Sub, SubOperation, Tanh, TanhOperation, scaled_dot_composition,
};
use crate::operations::memory::{TransferToMemory, TransferToMemoryOperation};
use crate::operations::random::{
    RandomAlgorithm, RngBitGenerator, RngBitGeneratorOperation, philox_u32_words, philox_u64_words, threefry_u32_words,
    threefry_u64_words,
};
use crate::operations::sharding::{ReshardOperation, ShardingConstraintOperation};
use crate::operations::sort::{
    ArgMax, ArgMin, Sort, SortDirection, SortOperation, TopK, extremal_index_from_index_passenger, sort_permutation,
    top_k_from_index_passenger, top_k_via_squeezed_view,
};
use crate::operations::tag::{Tag, TagOperation};
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::{Concretizable, Value};
use crate::tracing::TracingContext;
use crate::tracing_v2::custom_derivatives::{CustomJvpOperation, CustomVjpOperation};
use crate::tracing_v2::rematerialization::RematerializeOperation;
use crate::types::{ArrayType, DataType, Dimension, Shape, StaticShape};

/// Backend execution contract for broadcasting to an already-concrete [`ArrayType`].
///
/// This kernel does not stage a program operation or create first-class dimension values. Composite eager
/// interpretation resolves first-class dimension operands and validates the result type before invoking it. Program
/// construction uses [`Broadcast`](crate::operations::manipulation::Broadcast) instead.
pub trait BroadcastKernel: Sized {
    /// Broadcasts `self` to `output_type` using `output_axes`.
    fn broadcast_to_type(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError>;
}

/// Reusable [`Operation`] enum for ordinary staged programs over arrays.
///
/// [`ArrayOperation`] is the ordinary operation enum for core tests and backend crates, pairing with [`Array`] the
/// same way [`ScalarOperation`](crate::backends::scalars::ScalarOperation) pairs with [`Scalar`]. Most variants are
/// thin tags around one semantic primitive defined in [`crate::operations`] or
/// [`crate::tracing_v2::custom_derivatives`].
///
/// Each variant wraps exactly the backing operation struct that owns the variant's semantics (type inference,
/// rendering, and interpretation): for example [`Zero`](Self::Zero) wraps a [`ZeroOperation`] and
/// [`Dot`](Self::Dot) a [`DotOperation`].
#[derive(Clone, Debug, Operation)]
#[ryft(dispatch(batching, differentiation, transposition))]
pub enum ArrayOperation<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    ZeroLike(ZeroLikeOperation<ArrayType>),
    One(OneOperation<ArrayType>),
    OneLike(OneLikeOperation<ArrayType>),
    Constant(ConstantOperation<Array>),
    Iota(IotaOperation<ArrayType>),
    CoordinateBasis(CoordinateBasisOperation<ArrayType>),
    Abs(AbsOperation<ArrayType>),
    Neg(NegOperation<ArrayType>),
    Add(AddOperation<ArrayType>),
    Sub(SubOperation<ArrayType>),
    Mul(MulOperation<ArrayType>),
    Div(DivOperation<ArrayType>),
    Sin(SinOperation<ArrayType>),
    Cos(CosOperation<ArrayType>),
    Atan2(Atan2Operation<ArrayType>),
    Exp(ExpOperation<ArrayType>),
    Log(LogOperation<ArrayType>),
    Sqrt(SqrtOperation<ArrayType>),
    Rsqrt(RsqrtOperation<ArrayType>),
    Tanh(TanhOperation<ArrayType>),
    Logistic(LogisticOperation<ArrayType>),
    Erf(ErfOperation<ArrayType>),
    Pow(PowOperation<ArrayType>),
    Sign(SignOperation<ArrayType>),
    Floor(FloorOperation<ArrayType>),
    Ceil(CeilOperation<ArrayType>),
    Round(RoundOperation<ArrayType>),
    Max(MaxOperation<ArrayType>),
    Min(MinOperation<ArrayType>),
    Rem(RemOperation<ArrayType>),
    Not(NotOperation<ArrayType>),
    And(AndOperation<ArrayType>),
    Or(OrOperation<ArrayType>),
    Xor(XorOperation<ArrayType>),
    Complex(ComplexOperation<ArrayType>),
    Conjugate(ConjugateOperation<ArrayType>),
    Real(RealOperation<ArrayType>),
    Imaginary(ImaginaryOperation<ArrayType>),
    Dot(DotOperation),
    ScaledDot(ScaledDotOperation),
    DotProductAttention(DotProductAttentionOperation),
    DotProductAttentionBackward(DotProductAttentionBackwardOperation),
    Reduce(ReduceOperation),
    Sort(SortOperation),
    RngBitGenerator(RngBitGeneratorOperation<ArrayType>),
    Collective(CollectiveOperation),
    AllGather(AllGatherOperation),
    PSumScatter(PSumScatterOperation),
    Ppermute(PpermuteOperation),
    AllToAll(AllToAllOperation),
    AxisIndex(AxisIndexOperation),
    Transpose(TransposeOperation),
    Reshape(LegacyReshapeOperation),
    Broadcast(LegacyBroadcastOperation),
    Pad(PadOperation<ArrayType>),
    Concatenate(ConcatenateOperation<ArrayType>),
    Gather(GatherOperation),
    Scatter(ScatterOperation),
    Slice(SliceOperation),
    UpdateSlice(UpdateSliceOperation),
    DynamicSlice(DynamicSliceOperation),
    DynamicUpdateSlice(DynamicUpdateSliceOperation),
    Compare(CompareOperation<ArrayType>),
    Select(SelectOperation<ArrayType>),
    Condition(ConditionOperation<V>),
    While(WhileOperation<ArrayType>),
    Scan(ScanOperation<V>),
    ConvertElementType(ConvertElementTypeOperation<ArrayType>),
    TransferToMemory(TransferToMemoryOperation),
    Reshard(ReshardOperation),
    ShardingConstraint(ShardingConstraintOperation),
    StopGradient(StopGradientOperation<ArrayType>),
    Tag(TagOperation<ArrayType>),
    Rematerialize(RematerializeOperation<ArrayType>),
    Print(PrintOperation<ArrayType>),
    CustomCall(CustomCallOperation<ArrayType>),
    CustomJvp(CustomJvpOperation<ArrayType>),
    CustomVjp(CustomVjpOperation<ArrayType>),
    LinearCall(LinearCallOperation<ArrayType>),
}

/// [`TracingContext`] over the array universe, pairing [`ArrayType`] types and [`Array`] staged constants with the
/// [`ArrayOperation`] family.
pub type ArrayTracingContext = TracingContext<Array, ArrayOperation<Array>>;

/// Dense multidimensional [`Value`] whose [`Type`] is an [`ArrayType`] and which is meant to be used
/// primarily for testing the Ryft infrastructure and machinery with programs that involve multidimensional arrays,
/// without depending on an optimized backend such as `ryft-xla`.
///
/// The payload is immutable physical storage whose byte placement is determined by the array's [`ArrayType`]. Missing
/// layout metadata means dense row-major storage; explicit strided and tiled layouts determine the physical ordering,
/// holes, and padding. [`Array::new`] validates the complete physical representation, while
/// [`Array::from_elements`] and [`Array::from_logical_bytes`] construct it from logical row-major values. Cloning an
/// array shares its payload without copying it.
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
#[derive(Clone, Parameter)]
pub struct Array {
    /// Staged array type of this array value.
    r#type: ArrayType,

    /// Shared immutable physical storage, including any layout holes or tile padding.
    bytes: Arc<Vec<u8>>,
}

impl Array {
    /// Creates an array from its staged array type and complete physical storage. The storage must have the exact
    /// layout-derived byte count, contain a valid encoding for every logical element, and contain zero in every layout
    /// hole or tile-padding byte. Dynamically shaped types are rejected because they cannot describe materialized
    /// storage.
    ///
    /// # Parameters
    ///
    ///   - `type`: Staged array type of the array.
    ///   - `bytes`: Complete physical storage, including any layout holes or tile padding.
    pub fn new(r#type: ArrayType, bytes: Vec<u8>) -> Result<Self, ProgramError> {
        validate_storage_bytes(&r#type, &bytes)?;
        Ok(Self { r#type, bytes: Arc::new(bytes) })
    }

    /// Creates an array from typed elements provided in logical row-major order.
    ///
    /// # Parameters
    ///
    ///   - `type`: Staged array type of the array.
    ///   - `elements`: Logical row-major elements. Their Rust type and count must match `type`.
    pub fn from_elements<T: ArrayElement>(r#type: ArrayType, elements: &[T]) -> Result<Self, ProgramError> {
        let bytes = encode_elements(&r#type, elements)?;
        Ok(Self { r#type, bytes: Arc::new(bytes) })
    }

    /// Creates an array from concatenated logical element encodings in row-major order.
    ///
    /// # Parameters
    ///
    ///   - `type`: Staged array type of the array.
    ///   - `bytes`: Concatenated logical element bytes, without layout holes or tile padding.
    pub fn from_logical_bytes(r#type: ArrayType, bytes: &[u8]) -> Result<Self, ProgramError> {
        let bytes = encode_logical_bytes(&r#type, bytes)?;
        Ok(Self { r#type, bytes: Arc::new(bytes) })
    }

    /// Creates a rank-0 array containing `value`.
    pub fn scalar<T: ArrayElement>(value: T) -> Self {
        Self::from_elements(ArrayType::scalar(T::data_type()), &[value]).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Creates a rank-1 array containing `elements` in logical order.
    pub fn vector<T: ArrayElement>(elements: Vec<T>) -> Self {
        let r#type = ArrayType::new(T::data_type(), Shape::new(vec![Dimension::Static(elements.len())]));
        Self::from_elements(r#type, &elements).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Creates a rank-2 array containing `elements` in logical row-major order. Panics if the element count does not
    /// equal `rows * columns`.
    pub fn matrix<T: ArrayElement>(rows: usize, columns: usize, elements: Vec<T>) -> Self {
        let r#type =
            ArrayType::new(T::data_type(), Shape::new(vec![Dimension::Static(rows), Dimension::Static(columns)]));
        Self::from_elements(r#type, &elements).unwrap_or_else(|error| panic!("{error}"))
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
        let values: Vec<_> = values
            .into_iter()
            .map(|value| Scalar::from(value).convert_element_type(data_type).unwrap_or_else(|error| panic!("{error}")))
            .collect();
        Self::from_scalar_values(r#type, values).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Returns the complete immutable physical storage, including layout holes and tile padding.
    pub fn storage_bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    /// Decodes this array as typed elements in logical row-major order.
    pub fn elements<T: ArrayElement>(&self) -> Result<Vec<T>, ProgramError> {
        decode_elements(&self.r#type, self.bytes.as_slice())
    }

    /// Returns the concatenated logical element encodings in row-major order, omitting layout holes and tile padding.
    pub fn logical_bytes(&self) -> Vec<u8> {
        decode_logical_bytes(&self.r#type, self.bytes.as_slice()).unwrap()
    }

    /// Applies a typed elementwise function to this array in logical row-major order, producing a new array of
    /// `output_type`. Both arrays use their sealed codecs one element at a time, so the only payload allocation is the
    /// result buffer, and the output layout may differ from the input layout.
    ///
    /// # Parameters
    ///
    ///   - `output_type`: static array type of the result. Its [`DataType`] must be represented by `Output` and its
    ///     logical element count must equal this array's (elementwise kernels typically preserve the shape).
    ///   - `function`: elementwise function applied to each decoded `Input` element.
    ///
    /// # Errors
    ///
    /// Returns an error if either array type cannot describe materialized storage, if `Input` or `Output` represents
    /// a different [`DataType`] than the corresponding array type, if the logical element counts differ, or if
    /// `function` fails.
    pub fn map_elements<Input: ArrayElement, Output: ArrayElement>(
        &self,
        output_type: ArrayType,
        function: impl Fn(Input) -> Result<Output, ProgramError>,
    ) -> Result<Self, ProgramError> {
        if self.r#type.data_type() != Input::DATA_TYPE {
            return Err(TypeError::invalid(format!(
                "cannot map elements of data type {} as {} values",
                self.r#type.data_type(),
                Input::DATA_TYPE,
            ))
            .into());
        }
        if output_type.data_type() != Output::DATA_TYPE {
            return Err(TypeError::invalid(format!(
                "cannot store mapped {} values in an array of element data type {}",
                Output::DATA_TYPE,
                output_type.data_type(),
            ))
            .into());
        }
        let input_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        if input_addressing.element_count() != output_addressing.element_count() {
            return Err(TypeError::invalid(format!(
                "cannot map {} logical elements onto array type {} with {} logical elements",
                input_addressing.element_count(),
                output_type,
                output_addressing.element_count(),
            ))
            .into());
        }
        let mut output_bytes = vec![0; output_addressing.storage_byte_len()];
        for element in 0..output_addressing.element_count() {
            let input = Input::decode(&self.bytes[input_addressing.byte_range_for_flat_index(element)]);
            let output = function(input)?;
            output.encode(&mut output_bytes[output_addressing.byte_range_for_flat_index(element)]);
        }
        Ok(Self { r#type: output_type, bytes: Arc::new(output_bytes) })
    }

    /// Converts this array to the provided element data type, borrowing it unchanged when it already has that data
    /// type so that already-promoted operands keep their exact physical storage and layout. Kernels that promote
    /// mixed-type operands to a common element data type (which each kernel computes from its own type-inference
    /// contract) use this to convert only the mismatched operands.
    pub fn promoted_to(&self, data_type: DataType) -> Result<Cow<'_, Self>, ProgramError> {
        if self.r#type.data_type() == data_type {
            Ok(Cow::Borrowed(self))
        } else {
            Ok(Cow::Owned(self.convert_element_type(data_type)?))
        }
    }

    /// Broadcasts the types of the provided arrays together (including element data type promotion) and promotes
    /// every array to the broadcast element data type, borrowing the ones that already have it. This is the shared
    /// entry step of broadcasting elementwise kernels: the returned operands all have the broadcast element data
    /// type, while their shapes may still differ from the returned broadcast type, which the shared elementwise
    /// loops bridge by indexing the operands with NumPy-style broadcasting.
    pub fn broadcast_promoted<'a>(arrays: &[&'a Self]) -> Result<(ArrayType, Vec<Cow<'a, Self>>), ProgramError> {
        let types = arrays.iter().map(|array| &array.r#type).collect::<Vec<_>>();
        let output_type = ArrayType::broadcasted(&types).map_err(|error| TypeError::invalid(error.to_string()))?;
        let operands = arrays
            .iter()
            .map(|array| array.promoted_to(output_type.data_type()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((output_type, operands))
    }

    /// Creates an array of `type` by evaluating a typed function at every flat logical row-major element index. This
    /// is the constructor form of [`Array::map_elements`], serving iota-style and coordinate-dependent kernels.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: static array type of the result, whose [`DataType`] must be represented by `T`.
    ///   - `function`: function producing the element at each flat logical row-major index.
    ///
    /// # Errors
    ///
    /// Returns an error if `r#type` cannot describe materialized storage, if `T` represents a different [`DataType`],
    /// or if `function` fails.
    pub fn from_fn_elements<T: ArrayElement>(
        r#type: ArrayType,
        function: impl Fn(usize) -> Result<T, ProgramError>,
    ) -> Result<Self, ProgramError> {
        if r#type.data_type() != T::DATA_TYPE {
            return Err(TypeError::invalid(format!(
                "cannot store {} values in an array of element data type {}",
                T::DATA_TYPE,
                r#type.data_type(),
            ))
            .into());
        }
        let addressing = ArrayAddressing::new(r#type.clone())?;
        let mut bytes = vec![0; addressing.storage_byte_len()];
        for element in 0..addressing.element_count() {
            function(element)?.encode(&mut bytes[addressing.byte_range_for_flat_index(element)]);
        }
        Ok(Self { r#type, bytes: Arc::new(bytes) })
    }

    /// Folds this array's typed elements in logical row-major order into one accumulated value, serving full
    /// reductions and accumulation-style kernels such as dot products.
    ///
    /// # Parameters
    ///
    ///   - `initial`: initial accumulator value.
    ///   - `function`: fold step combining the accumulator with each decoded element.
    ///
    /// # Errors
    ///
    /// Returns an error if this array's type cannot describe materialized storage, if `T` represents a different
    /// [`DataType`], or if `function` fails.
    pub fn fold_elements<T: ArrayElement, Accumulator>(
        &self,
        initial: Accumulator,
        function: impl Fn(Accumulator, T) -> Result<Accumulator, ProgramError>,
    ) -> Result<Accumulator, ProgramError> {
        if self.r#type.data_type() != T::DATA_TYPE {
            return Err(TypeError::invalid(format!(
                "cannot fold elements of data type {} as {} values",
                self.r#type.data_type(),
                T::DATA_TYPE,
            ))
            .into());
        }
        let addressing = ArrayAddressing::new(self.r#type.clone())?;
        let mut accumulator = initial;
        for element in 0..addressing.element_count() {
            accumulator = function(accumulator, T::decode(&self.bytes[addressing.byte_range_for_flat_index(element)]))?;
        }
        Ok(accumulator)
    }

    /// Creates an array of `output_type` whose every element is copied from this array through an
    /// output-index-to-input-index mapping over flat logical row-major indices. The copy moves whole element
    /// encodings without decoding them, so this is the element-data-type-agnostic workhorse behind structural kernels
    /// such as transpose, broadcast, slice, reverse, and gather, which never need element-type dispatch.
    ///
    /// # Parameters
    ///
    ///   - `output_type`: static array type of the result, which must have the same [`DataType`] as this array.
    ///   - `index`: mapping from each flat logical output element index to the flat logical input element index whose
    ///     element it copies. Input indices may repeat or be skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if either array type cannot describe materialized storage, if the element data types differ,
    /// or if `index` produces an out-of-bounds input index.
    pub fn gather_elements(
        &self,
        output_type: ArrayType,
        index: impl Fn(usize) -> usize,
    ) -> Result<Self, ProgramError> {
        if output_type.data_type() != self.r#type.data_type() {
            return Err(TypeError::invalid(format!(
                "cannot gather elements of data type {} into an array of element data type {}",
                self.r#type.data_type(),
                output_type.data_type(),
            ))
            .into());
        }
        let input_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut output_bytes = vec![0; output_addressing.storage_byte_len()];
        for output_element in 0..output_addressing.element_count() {
            let input_element = index(output_element);
            if input_element >= input_addressing.element_count() {
                return Err(TypeError::invalid(format!(
                    "gather index {input_element} is out of bounds for {} elements",
                    input_addressing.element_count(),
                ))
                .into());
            }
            let input_range = input_addressing.byte_range_for_flat_index(input_element);
            output_bytes[output_addressing.byte_range_for_flat_index(output_element)]
                .copy_from_slice(&self.bytes[input_range]);
        }
        Ok(Self { r#type: output_type, bytes: Arc::new(output_bytes) })
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
        self.scalar_values()
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
        r#type.element_count().map_err(|error| TypeError::invalid(error.to_string()))?.ok_or_else(|| {
            TypeError::invalid(format!("cannot materialize a value of dynamically sized type {}", r#type)).into()
        })
    }

    /// Returns the zero [`Scalar`] of the provided element data type, used by kernels that need a payload identity
    /// element (e.g., dot-product accumulators and dropped gather results).
    fn zero_element(data_type: DataType) -> Result<Scalar, ProgramError> {
        EagerContext::<Scalar>::new().zero(&data_type)
    }

    /// Decodes the physical payload into the temporary scalar representation used by kernels that have not yet moved
    /// to direct typed-byte dispatch. Phase 9a2 removes this bridge family by family; it is deliberately private and
    /// never becomes stored state or a public compatibility accessor.
    fn scalar_values(&self) -> Vec<Scalar> {
        let addressing = ArrayAddressing::new(self.r#type.clone()).unwrap();
        if self.r#type.data_type() == DataType::Token {
            return vec![Scalar::Token; addressing.element_count()];
        }
        if self.r#type.data_type() == DataType::Zero {
            return vec![Scalar::Zero; addressing.element_count()];
        }
        let logical_bytes = self.logical_bytes();
        logical_bytes
            .chunks_exact(addressing.element_byte_width())
            .map(|bytes| match self.r#type.data_type() {
                DataType::Boolean => Scalar::Bool(bytes[0] != 0),
                DataType::I8 => Scalar::I8(bytes[0] as i8),
                DataType::I16 => Scalar::I16(i16::from_le_bytes(bytes.try_into().unwrap())),
                DataType::I32 => Scalar::I32(i32::from_le_bytes(bytes.try_into().unwrap())),
                DataType::I64 => Scalar::I64(i64::from_le_bytes(bytes.try_into().unwrap())),
                DataType::U8 => Scalar::U8(bytes[0]),
                DataType::U16 => Scalar::U16(u16::from_le_bytes(bytes.try_into().unwrap())),
                DataType::U32 => Scalar::U32(u32::from_le_bytes(bytes.try_into().unwrap())),
                DataType::U64 => Scalar::U64(u64::from_le_bytes(bytes.try_into().unwrap())),
                DataType::F4E2M1FN
                | DataType::F6E2M3FN
                | DataType::F6E3M2FN
                | DataType::F8E3M4
                | DataType::F8E4M3
                | DataType::F8E4M3FN
                | DataType::F8E4M3FNUZ
                | DataType::F8E4M3B11FNUZ
                | DataType::F8E5M2
                | DataType::F8E5M2FNUZ
                | DataType::F8E8M0FNU => {
                    Scalar::from_low_precision_float_bits(self.r#type.data_type(), bytes[0]).unwrap()
                }
                DataType::BF16 => Scalar::BF16(bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap()))),
                DataType::F16 => Scalar::F16(f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap()))),
                DataType::F32 => Scalar::F32(f32::from_le_bytes(bytes.try_into().unwrap())),
                DataType::F64 => Scalar::F64(f64::from_le_bytes(bytes.try_into().unwrap())),
                DataType::C64 => Scalar::C64(Complex::new(
                    f32::from_le_bytes(bytes[..4].try_into().unwrap()),
                    f32::from_le_bytes(bytes[4..].try_into().unwrap()),
                )),
                DataType::C128 => Scalar::C128(Complex::new(
                    f64::from_le_bytes(bytes[..8].try_into().unwrap()),
                    f64::from_le_bytes(bytes[8..].try_into().unwrap()),
                )),
                DataType::Token | DataType::Zero => unreachable!(),
                DataType::I1 | DataType::I2 | DataType::I4 | DataType::U1 | DataType::U2 | DataType::U4 => {
                    panic!("scalar-backed reference kernels do not yet support {} elements", self.r#type.data_type())
                }
            })
            .collect()
    }

    /// Encodes a temporary scalar sequence directly into the final physical storage selected by `type`.
    pub(crate) fn from_scalar_values(
        r#type: ArrayType,
        values: impl IntoIterator<Item = Scalar>,
    ) -> Result<Self, ProgramError> {
        let addressing = ArrayAddressing::new(r#type.clone())?;
        let mut bytes = vec![0; addressing.storage_byte_len()];
        let mut count = 0;
        for (index, value) in values.into_iter().enumerate() {
            if index >= addressing.element_count() {
                return Err(TypeError::invalid(format!(
                    "array type {} requires {} logical elements but got more",
                    r#type,
                    addressing.element_count(),
                ))
                .into());
            }
            let value_type = value.r#type().into_owned();
            if value_type != r#type.data_type() {
                return Err(TypeError::invalid(format!(
                    "array of element data type {} cannot store an element of data type {}",
                    r#type.data_type(),
                    value_type,
                ))
                .into());
            }
            let element_bytes = &mut bytes[addressing.byte_range_for_flat_index(index)];
            match value {
                Scalar::Token | Scalar::Zero => {}
                Scalar::Bool(value) => element_bytes[0] = u8::from(value),
                Scalar::I8(value) => element_bytes[0] = value as u8,
                Scalar::I16(value) => element_bytes.copy_from_slice(&value.to_le_bytes()),
                Scalar::I32(value) => element_bytes.copy_from_slice(&value.to_le_bytes()),
                Scalar::I64(value) => element_bytes.copy_from_slice(&value.to_le_bytes()),
                Scalar::U8(value) => element_bytes[0] = value,
                Scalar::U16(value) => element_bytes.copy_from_slice(&value.to_le_bytes()),
                Scalar::U32(value) => element_bytes.copy_from_slice(&value.to_le_bytes()),
                Scalar::U64(value) => element_bytes.copy_from_slice(&value.to_le_bytes()),
                Scalar::F4E2M1FN(value)
                | Scalar::F6E2M3FN(value)
                | Scalar::F6E3M2FN(value)
                | Scalar::F8E3M4(value)
                | Scalar::F8E4M3(value)
                | Scalar::F8E4M3FN(value)
                | Scalar::F8E4M3FNUZ(value)
                | Scalar::F8E4M3B11FNUZ(value)
                | Scalar::F8E5M2(value)
                | Scalar::F8E5M2FNUZ(value)
                | Scalar::F8E8M0FNU(value) => element_bytes[0] = value,
                Scalar::BF16(value) => element_bytes.copy_from_slice(&value.to_bits().to_le_bytes()),
                Scalar::F16(value) => element_bytes.copy_from_slice(&value.to_bits().to_le_bytes()),
                Scalar::F32(value) => element_bytes.copy_from_slice(&value.to_le_bytes()),
                Scalar::F64(value) => element_bytes.copy_from_slice(&value.to_le_bytes()),
                Scalar::C64(value) => {
                    element_bytes[..4].copy_from_slice(&value.re.to_le_bytes());
                    element_bytes[4..].copy_from_slice(&value.im.to_le_bytes());
                }
                Scalar::C128(value) => {
                    element_bytes[..8].copy_from_slice(&value.re.to_le_bytes());
                    element_bytes[8..].copy_from_slice(&value.im.to_le_bytes());
                }
            }
            count = index + 1;
        }
        if count != addressing.element_count() {
            return Err(TypeError::invalid(format!(
                "array type {} requires {} logical elements but got {}",
                r#type,
                addressing.element_count(),
                count,
            ))
            .into());
        }
        Ok(Self { r#type, bytes: Arc::new(bytes) })
    }

    /// Creates an array of `type` by repeating one scalar element, which must already have `type`'s element data
    /// type, across every logical element.
    fn from_scalar_element(r#type: &ArrayType, element: Scalar) -> Result<Self, ProgramError> {
        Self::from_scalar_values(r#type.clone(), vec![element; Self::materialized_element_count(r#type)?])
    }

    /// Applies an elementwise unary function to the payload, preserving this array's type.
    fn unary(&self, function: impl Fn(&Scalar) -> Result<Scalar, ProgramError>) -> Result<Self, ProgramError> {
        let values = self.scalar_values();
        Self::from_scalar_values(self.r#type.clone(), values.iter().map(function).collect::<Result<Vec<_>, _>>()?)
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
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let output_len = Self::element_count(&output_type);
        let left = self.broadcast_values(output_len);
        let right = rhs.broadcast_values(output_len);
        let values = left
            .iter()
            .zip(right.iter())
            .map(|(left, right)| function(left, right))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_scalar_values(output_type, values)
    }

    /// Applies a typed binary element function with NumPy-style broadcasting directly over addressed storage. Inputs
    /// and outputs use their sealed codecs one element at a time, so the only payload allocation is the result buffer.
    fn binary_elements<Input: ArrayElement, Output: ArrayElement>(
        &self,
        rhs: &Self,
        output_type: ArrayType,
        function: impl Fn(Input, Input) -> Result<Output, ProgramError>,
    ) -> Result<Self, ProgramError> {
        debug_assert_eq!(self.r#type.data_type(), Input::data_type());
        debug_assert_eq!(rhs.r#type.data_type(), Input::data_type());
        debug_assert_eq!(output_type.data_type(), Output::data_type());
        let output_shape = output_type.static_shape().unwrap();
        let left_shape = self.r#type.static_shape().unwrap();
        let right_shape = rhs.r#type.static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let left_strides = left_shape.row_major_strides();
        let right_strides = right_shape.row_major_strides();
        let left_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let right_addressing = ArrayAddressing::new(rhs.r#type.clone())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut output_bytes = vec![0; output_addressing.storage_byte_len()];
        for output_index in 0..output_addressing.element_count() {
            let left_index =
                Self::broadcast_index(output_index, &output_shape, &output_strides, &left_shape, &left_strides);
            let right_index =
                Self::broadcast_index(output_index, &output_shape, &output_strides, &right_shape, &right_strides);
            let left = Input::decode(&self.bytes[left_addressing.byte_range_for_flat_index(left_index)]);
            let right = Input::decode(&rhs.bytes[right_addressing.byte_range_for_flat_index(right_index)]);
            let output = function(left, right)?;
            output.encode(&mut output_bytes[output_addressing.byte_range_for_flat_index(output_index)]);
        }
        Ok(Self { r#type: output_type, bytes: Arc::new(output_bytes) })
    }

    /// Compares two arrays of the same type elementwise using their typed value semantics rather than their physical
    /// byte patterns. In particular, signed floating-point zeros compare equal and NaNs compare unequal.
    fn elements_equal<T: ArrayElement + PartialEq>(&self, other: &Self) -> bool {
        let addressing = ArrayAddressing::new(self.r#type.clone()).unwrap();
        (0..addressing.element_count()).all(|index| {
            let range = addressing.byte_range_for_flat_index(index);
            T::decode(&self.bytes[range.clone()]) == T::decode(&other.bytes[range])
        })
    }

    /// Compares two same-type arrays elementwise using their typed value semantics rather than their physical byte
    /// patterns. The equality directions apply to every element type, while the ordered directions apply to the
    /// partially ordered element types and are rejected with an error for the unordered complex ones.
    fn compare_elements(
        &self,
        rhs: &Self,
        output_type: ArrayType,
        direction: ComparisonDirection,
    ) -> Result<Self, ProgramError> {
        let data_type = self.r#type.data_type();
        if data_type.is_complex() {
            // The unordered complex element types define only the equality comparison directions. The compare
            // operation's type inference already rejects ordered complex comparisons, but the direct `Array`
            // comparison API reaches this kernel without it.
            if !matches!(direction, ComparisonDirection::Equal | ComparisonDirection::NotEqual) {
                return Err(TypeError::invalid(format!(
                    "cannot apply an ordered comparison to unordered complex scalars of data type {data_type}",
                ))
                .into());
            }
            let equal = matches!(direction, ComparisonDirection::Equal);
            return dispatch_on_array_element_type!(@complex data_type, |Element| {
                self.binary_elements::<Element, bool>(rhs, output_type, |left, right| {
                    Ok(if equal { left == right } else { left != right })
                })
            });
        }
        dispatch_on_array_element_type!(@ordered data_type, |Element| {
            self.binary_elements::<Element, bool>(rhs, output_type, |left, right| {
                // An unordered pair (a comparison involving a floating-point NaN) satisfies only `NotEqual`.
                let ordering = left.partial_cmp(&right);
                Ok(match direction {
                    ComparisonDirection::Equal => ordering == Some(Ordering::Equal),
                    ComparisonDirection::NotEqual => ordering != Some(Ordering::Equal),
                    ComparisonDirection::LessThan => ordering == Some(Ordering::Less),
                    ComparisonDirection::LessThanOrEqual => matches!(ordering, Some(Ordering::Less | Ordering::Equal)),
                    ComparisonDirection::GreaterThan => ordering == Some(Ordering::Greater),
                    ComparisonDirection::GreaterThanOrEqual => {
                        matches!(ordering, Some(Ordering::Greater | Ordering::Equal))
                    }
                })
            })
        })
    }

    /// Applies a binary logical or bitwise operation directly to validated Boolean or integer element bytes. Since
    /// bitwise operations act independently on every bit, their result is independent of integer signedness and host
    /// endianness. Logical Boolean encodings use the same `0` and `1` bitwise truth tables.
    fn binary_logical(
        &self,
        rhs: &Self,
        operation: &str,
        function: impl Fn(u8, u8) -> u8,
    ) -> Result<Self, ProgramError> {
        let left_data_type = self.r#type.data_type();
        let right_data_type = rhs.r#type.data_type();
        let output_type = Broadcastable::broadcast(&self.r#type, &rhs.r#type)
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        if left_data_type != right_data_type || !(left_data_type.is_boolean() || left_data_type.is_integer()) {
            return Err(TypeError::invalid(format!(
                "cannot apply `{operation}` to arrays of element data types {left_data_type} and {right_data_type}",
            ))
            .into());
        }

        let output_shape = output_type.static_shape().unwrap();
        let left_shape = self.r#type.static_shape().unwrap();
        let right_shape = rhs.r#type.static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let left_strides = left_shape.row_major_strides();
        let right_strides = right_shape.row_major_strides();
        let left_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let right_addressing = ArrayAddressing::new(rhs.r#type.clone())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let left_bytes = self.storage_bytes();
        let right_bytes = rhs.storage_bytes();
        let element_byte_width = output_addressing.element_byte_width();
        let mut output_bytes = vec![0; output_addressing.storage_byte_len()];
        for output_index in 0..output_addressing.element_count() {
            let left_range = left_addressing.byte_range_for_flat_index(Self::broadcast_index(
                output_index,
                &output_shape,
                &output_strides,
                &left_shape,
                &left_strides,
            ));
            let right_range = right_addressing.byte_range_for_flat_index(Self::broadcast_index(
                output_index,
                &output_shape,
                &output_strides,
                &right_shape,
                &right_strides,
            ));
            let output_range = output_addressing.byte_range_for_flat_index(output_index);
            for byte in 0..element_byte_width {
                output_bytes[output_range.start + byte] =
                    function(left_bytes[left_range.start + byte], right_bytes[right_range.start + byte]);
            }
        }
        // Valid inputs, bitwise closure outputs, and zero-initialized unoccupied storage preserve every `Array`
        // encoding invariant without a second validation traversal.
        Ok(Self { r#type: output_type, bytes: Arc::new(output_bytes) })
    }

    /// Maps one flat row-major output index to the corresponding flat input index under NumPy-style broadcasting.
    /// Input axes are right-aligned with output axes, and an input extent of one always selects coordinate zero.
    fn broadcast_index(
        output_index: usize,
        output_shape: &StaticShape,
        output_strides: &[usize],
        input_shape: &StaticShape,
        input_strides: &[usize],
    ) -> usize {
        let output_axis_offset = output_shape.rank() - input_shape.rank();
        (0..input_shape.rank()).fold(0, |index, input_axis| {
            let output_axis = output_axis_offset + input_axis;
            let coordinate = if input_shape[input_axis] == 1 {
                0
            } else {
                (output_index / output_strides[output_axis]) % output_shape[output_axis]
            };
            index + coordinate * input_strides[input_axis]
        })
    }

    /// Broadcasts the payload to `output_len`.
    fn broadcast_values(&self, output_len: usize) -> Vec<Scalar> {
        let values = self.scalar_values();
        if values.len() == output_len {
            values
        } else if values.len() == 1 {
            vec![values[0]; output_len]
        } else {
            panic!("cannot broadcast {} values to {output_len}", values.len());
        }
    }

    /// Decodes one logical integer element as the signed index representation used by reference indexing kernels.
    /// Unsigned `u64` values narrow with Rust's two's-complement `as i64` semantics, matching the former scalar
    /// conversion path. The type-level validation performed by every caller rules out non-integer element types.
    fn index_value(&self, addressing: &ArrayAddressing, index: usize) -> i64 {
        let bytes = &self.bytes[addressing.byte_range_for_flat_index(index)];
        match self.r#type.data_type() {
            DataType::I1 => i64::from(crate::arrays::i1::decode(bytes).value()),
            DataType::I2 => i64::from(crate::arrays::i2::decode(bytes).value()),
            DataType::I4 => i64::from(crate::arrays::i4::decode(bytes).value()),
            DataType::I8 => i64::from(i8::decode(bytes)),
            DataType::I16 => i64::from(i16::decode(bytes)),
            DataType::I32 => i64::from(i32::decode(bytes)),
            DataType::I64 => i64::decode(bytes),
            DataType::U1 => i64::from(crate::arrays::u1::decode(bytes).value()),
            DataType::U2 => i64::from(crate::arrays::u2::decode(bytes).value()),
            DataType::U4 => i64::from(crate::arrays::u4::decode(bytes).value()),
            DataType::U8 => i64::from(u8::decode(bytes)),
            DataType::U16 => i64::from(u16::decode(bytes)),
            DataType::U32 => i64::from(u32::decode(bytes)),
            DataType::U64 => u64::decode(bytes) as i64,
            data_type => unreachable!("cannot use an array of element data type {data_type} as indices"),
        }
    }
}

#[cfg(test)]
impl Array {
    /// Creates an array without enforcing the storage invariants, so that `ryft-core`'s own transform-validation tests
    /// can materialize values whose declared types are deliberately not materializable (e.g., dynamically shaped
    /// types) and exercise the type-level rejection paths. Never use this outside of such validation tests.
    pub(crate) fn with_unchecked_type(r#type: ArrayType, bytes: Vec<u8>) -> Self {
        Self { r#type, bytes: Arc::new(bytes) }
    }
}

impl Debug for Array {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The payload renders through `Display`, which supports every element data type, including the sub-byte ones
        // that have no `Scalar` representation.
        struct Values<'a>(&'a Array);
        impl Debug for Values<'_> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                Display::fmt(self.0, formatter)
            }
        }
        formatter.debug_struct("Array").field("type", &self.r#type).field("values", &Values(self)).finish()
    }
}

impl PartialEq for Array {
    fn eq(&self, other: &Self) -> bool {
        if self.r#type != other.r#type {
            return false;
        }
        let data_type = self.r#type.data_type();
        if matches!(data_type, DataType::Token | DataType::Zero) {
            return true;
        }
        dispatch_on_array_element_type!(data_type, |Element| Self::elements_equal::<Element>(self, other))
    }
}

// The rendering intentionally matches how `Vec<f64>` debug-formats: a bracketed, comma-separated element list in
// which real floating-point payloads keep a decimal point (e.g., `[1.0, 2.0]`), so program and interpreter
// diagnostics involving constant arrays stay readable and stable.
impl Display for Array {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Writes one bracketed, comma-separated element list through the provided per-element renderer.
        fn write_elements<T>(
            formatter: &mut std::fmt::Formatter<'_>,
            elements: impl IntoIterator<Item = T>,
            mut write_element: impl FnMut(&mut std::fmt::Formatter<'_>, T) -> std::fmt::Result,
        ) -> std::fmt::Result {
            formatter.write_str("[")?;
            for (index, element) in elements.into_iter().enumerate() {
                if index > 0 {
                    formatter.write_str(", ")?;
                }
                write_element(formatter, element)?;
            }
            formatter.write_str("]")
        }
        let data_type = self.r#type.data_type();
        match data_type {
            // The payload-free element data types have no element encoding to decode.
            DataType::Token | DataType::Zero => {
                write_elements(formatter, 0..Self::element_count(&self.r#type), |formatter, _| {
                    formatter.write_str(if data_type == DataType::Token { "token" } else { "zero" })
                })
            }
            // `f32` and `f64` payloads keep a decimal point through debug formatting, per the rendering contract
            // stated above this implementation.
            DataType::F32 => write_elements(formatter, self.elements::<f32>().unwrap(), |formatter, value| {
                write!(formatter, "{value:?}")
            }),
            DataType::F64 => write_elements(formatter, self.elements::<f64>().unwrap(), |formatter, value| {
                write!(formatter, "{value:?}")
            }),
            _ => dispatch_on_array_element_type!(data_type, |Element| {
                write_elements(formatter, self.elements::<Element>().unwrap(), |formatter, value| {
                    Display::fmt(&value, formatter)
                })
            }),
        }
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

impl DimensionSize<usize> for Array {
    fn dimension_size<AxisValue: Into<Axis>>(&self, axis: AxisValue) -> Result<usize, ProgramError> {
        let axis = axis.into();
        let position = axis.normalize(self.r#type.rank()).map_err(|_| {
            TypeError::invalid(format!(
                "'{DIMENSION_SIZE_OPERATION_NAME}' axis {axis} is out of bounds for rank {}",
                self.r#type.rank(),
            ))
        })?;
        let dimension = &self.r#type.shape().dimensions()[position];
        dimension.value().ok_or_else(|| {
            TypeError::invalid(format!("materialized reference array has a dynamic dimension at axis {position}",))
                .into()
        })
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
        let left = self.scalar_values();
        let right = other.scalar_values();
        self.r#type == other.r#type
            && left.len() == right.len()
            && left.iter().zip(right.iter()).all(|(left, right)| left.abs_diff_eq(right, epsilon))
    }
}

impl<O: Operation<Type = ArrayType>> Zero<Array> for EagerContext<Array, O> {
    fn zero(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
        let element = Array::zero_element(r#type.data_type())?;
        Array::from_scalar_element(r#type, element)
    }
}

impl ZeroLike for Array {
    fn zero_like(&self) -> Self {
        Self::from_scalar_values(self.r#type.clone(), self.scalar_values().iter().map(|value| value.zero_like()))
            .unwrap()
    }
}

impl<O: Operation<Type = ArrayType>> One<Array> for EagerContext<Array, O> {
    fn one(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
        let element = EagerContext::<Scalar>::new().one(&r#type.data_type())?;
        Array::from_scalar_element(r#type, element)
    }
}

impl OneLike for Array {
    fn one_like(&self) -> Self {
        Self::from_scalar_values(self.r#type.clone(), self.scalar_values().iter().map(|value| value.one_like()))
            .unwrap()
    }
}

impl StopGradient for Array {
    #[inline]
    fn stop_gradient(&self) -> Self {
        self.clone()
    }
}

impl<O: Operation<Type = ArrayType>> Fill<Scalar, Array> for EagerContext<Array, O> {
    fn fill(&self, r#type: &ArrayType, value: Scalar) -> Result<Array, ProgramError> {
        let element = value.convert_element_type(r#type.data_type())?;
        Array::from_scalar_element(r#type, element)
    }
}

impl<O: Operation<Type = ArrayType>> crate::operations::constants::Iota<Array> for EagerContext<Array, O> {
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<Array, ProgramError> {
        if !r#type.data_type().is_numeric() {
            return Err(TypeError::invalid(format!(
                "'{}' requires a numeric element type but has {}",
                IOTA_OPERATION_NAME,
                r#type.data_type(),
            ))
            .into());
        }
        let sizes = r#type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| {
                dimension.value().ok_or_else(|| {
                    TypeError::invalid(format!("cannot materialize an iota of dynamically sized type {type}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if dimension >= sizes.len() {
            return Err(TypeError::invalid(format!(
                "iota dimension {dimension} is out of bounds for array type {type}",
            ))
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
        Array::from_scalar_values(r#type.clone(), values)
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
        let values = self.scalar_values().iter().map(|value| value.abs()).collect::<Result<Vec<_>, _>>()?;
        Self::from_scalar_values(self.r#type.clone().with_data_type(data_type), values)
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
        let r#type = self.r#type.clone();
        Self::from_scalar_values(r#type, self.scalar_values().into_iter().map(|value| value * factor)).unwrap()
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

impl Rsqrt for Array {
    fn rsqrt(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.rsqrt())
    }
}

impl Tanh for Array {
    fn tanh(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.tanh())
    }
}

impl Logistic for Array {
    fn logistic(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.logistic())
    }
}

impl Erf for Array {
    fn erf(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.erf())
    }
}

impl Pow for Array {
    fn pow(&self, exponent: &Self) -> Result<Self, ProgramError> {
        self.binary(exponent, |base, exponent| base.pow(exponent))
    }
}

impl Sign for Array {
    fn sign(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.sign())
    }
}

impl Floor for Array {
    fn floor(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.floor())
    }
}

impl Ceil for Array {
    fn ceil(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.ceil())
    }
}

impl Round for Array {
    fn round(&self) -> Result<Self, ProgramError> {
        self.unary(|value| value.round())
    }
}

impl Max for Array {
    fn max(&self, right: &Self) -> Result<Self, ProgramError> {
        self.binary(right, |left, right| left.max(right))
    }
}

impl Min for Array {
    fn min(&self, right: &Self) -> Result<Self, ProgramError> {
        self.binary(right, |left, right| left.min(right))
    }
}

impl Rem for Array {
    fn rem(&self, right: &Self) -> Result<Self, ProgramError> {
        self.binary(right, |left, right| left.rem(right))
    }
}

impl Sort for Array {
    fn sort_with_key_count(
        operands: &[Self],
        axis: usize,
        direction: SortDirection,
        key_count: usize,
    ) -> Result<Vec<Self>, ProgramError> {
        let Some(key) = operands.first() else {
            return Err(ProgramError::UnsupportedOperation { message: "'sort' needs at least one input".to_string() });
        };
        if key_count == 0 {
            return Err(ProgramError::UnsupportedOperation {
                message: "'sort' key_count must be at least 1".to_string(),
            });
        }
        if key_count > operands.len() {
            return Err(TypeError::invalid(format!(
                "'sort' key_count {} exceeds operand count {}",
                key_count,
                operands.len(),
            ))
            .into());
        }
        let shape = key.r#type.static_shape().unwrap();
        if axis >= shape.rank() {
            return Err(
                TypeError::invalid(format!("'sort' axis {axis} is out of bounds for rank {}", shape.rank())).into()
            );
        }
        for operand in operands {
            if operand.r#type.shape() != key.r#type.shape() {
                return Err(TypeError::invalid(format!(
                    "'sort' operands must agree on shape but got {} and {}",
                    key.r#type.shape(),
                    operand.r#type.shape(),
                ))
                .into());
            }
        }
        let key_ranks = operands[..key_count]
            .iter()
            .map(|key| {
                let data_type = key.r#type.data_type();
                let unsupported = || {
                    ProgramError::from(TypeError::invalid(format!("'sort' does not support key data type {data_type}")))
                };
                // The rank computation still rides the temporary scalar bridge, which has no sub-byte representation.
                if matches!(
                    data_type,
                    DataType::I1 | DataType::I2 | DataType::I4 | DataType::U1 | DataType::U2 | DataType::U4,
                ) {
                    return Err(unsupported());
                }
                key.scalar_values()
                    .iter()
                    .map(|value| value.total_order_rank().ok_or_else(unsupported))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let key_rank_slices = key_ranks.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let gather = sort_permutation(key_rank_slices.as_slice(), shape.dimensions(), axis, direction);
        // Applying the gather map moves whole element encodings, so non-key operands of any element data type
        // (including the sub-byte ones without a scalar representation) sort without being decoded.
        operands
            .iter()
            .map(|operand| operand.gather_elements(operand.r#type.clone(), |index| gather[index]))
            .collect()
    }
}

/// Materializes the `i32` index passenger that rides a ranking sort for the concrete eager [`Array`] backend
/// (the transform tracers stage it as an [`IotaOperation`](crate::operations::constants::IotaOperation) instead),
/// returning it together with the operand's static dimensions.
fn eager_index_passenger(value: &Array, axis: usize) -> Result<(Array, Vec<usize>), ProgramError> {
    let shape = value.r#type.static_shape().unwrap();
    let dimensions = shape.dimensions().to_vec();
    if axis >= dimensions.len() {
        return Err(
            TypeError::invalid(format!("'sort' axis {} is out of bounds for rank {}", axis, dimensions.len())).into()
        );
    }
    let inner_stride: usize = dimensions[axis + 1..].iter().product();
    let axis_size = dimensions[axis];
    let indices = Array::from_fn_elements(ArrayType::new(DataType::I32, value.r#type.shape().clone()), |index| {
        Ok(((index / inner_stride) % axis_size) as i32)
    })?;
    Ok((indices, dimensions))
}

impl TopK for Array {
    fn top_k(&self, k: usize, axis: usize) -> Result<(Self, Self), ProgramError> {
        if let Some(outputs) = top_k_via_squeezed_view(self, k, axis)? {
            return Ok(outputs);
        }
        let (indices, dimensions) = eager_index_passenger(self, axis)?;
        top_k_from_index_passenger(self, indices, dimensions.as_slice(), k, axis)
    }
}

impl ArgMax for Array {
    fn argmax(&self, axis: usize) -> Result<Self, ProgramError> {
        let (indices, dimensions) = eager_index_passenger(self, axis)?;
        extremal_index_from_index_passenger(self, indices, dimensions.as_slice(), axis, SortDirection::Descending)
    }
}

impl ArgMin for Array {
    fn argmin(&self, axis: usize) -> Result<Self, ProgramError> {
        let (indices, dimensions) = eager_index_passenger(self, axis)?;
        extremal_index_from_index_passenger(self, indices, dimensions.as_slice(), axis, SortDirection::Ascending)
    }
}

impl ScaledDot for Array {
    fn scaled_dot(
        &self,
        lhs_scales: &Self,
        rhs: &Self,
        rhs_scales: &Self,
        block_size: usize,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError> {
        scaled_dot_composition(self, lhs_scales, rhs, rhs_scales, None, block_size, accumulation_type)
    }

    fn scaled_dot_with_global_scale(
        &self,
        lhs_scales: &Self,
        rhs: &Self,
        rhs_scales: &Self,
        global_scale: &Self,
        block_size: usize,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError> {
        scaled_dot_composition(self, lhs_scales, rhs, rhs_scales, Some(global_scale), block_size, accumulation_type)
    }
}

impl DotProductAttention for Array {
    fn dot_product_attention_with_options(
        &self,
        key: &Self,
        value: &Self,
        bias: Option<&Self>,
        sequence_lengths: Option<(&Self, &Self)>,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
        dropout: Option<(f64, u64)>,
    ) -> Result<Self, ProgramError> {
        let (output, _) = dot_product_attention_composition(
            &self.dispatch_domain(),
            self,
            key,
            value,
            bias,
            sequence_lengths,
            scale,
            mask,
            sliding_window,
            dropout,
            false,
        )?;
        Ok(output)
    }

    fn dot_product_attention_with_activation(
        &self,
        key: &Self,
        value: &Self,
        bias: Option<&Self>,
        sequence_lengths: Option<(&Self, &Self)>,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
        dropout: Option<(f64, u64)>,
    ) -> Result<(Self, Self), ProgramError> {
        let (output, activation) = dot_product_attention_composition(
            &self.dispatch_domain(),
            self,
            key,
            value,
            bias,
            sequence_lengths,
            scale,
            mask,
            sliding_window,
            dropout,
            true,
        )?;
        // The composition returns the activation statistic whenever it is requested.
        Ok((output, activation.unwrap()))
    }
}

impl DotProductAttentionBackward for Array {
    fn dot_product_attention_backward_with_options(
        &self,
        key: &Self,
        value: &Self,
        bias: Option<&Self>,
        sequence_lengths: Option<(&Self, &Self)>,
        output: &Self,
        activation: &Self,
        output_cotangent: &Self,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
        dropout: Option<(f64, u64)>,
    ) -> Result<(Self, Self, Self, Option<Self>), ProgramError> {
        let mut cotangents = dot_product_attention_backward_composition(
            &self.dispatch_domain(),
            self,
            key,
            value,
            bias,
            sequence_lengths,
            output,
            activation,
            output_cotangent,
            scale,
            mask,
            sliding_window,
            dropout,
        )?;
        let bias_cotangent = bias.is_some().then(|| cotangents.remove(3));
        check_count!("output", cotangents, 3, ProgramError);
        let value_cotangent = cotangents.remove(2);
        let key_cotangent = cotangents.remove(1);
        Ok((cotangents.remove(0), key_cotangent, value_cotangent, bias_cotangent))
    }
}

impl RngBitGenerator for Array {
    fn rng_bit_generator(
        &self,
        algorithm: RandomAlgorithm,
        output_type: &ArrayType,
    ) -> Result<(Self, Self), ProgramError> {
        let Some(output_shape) = output_type.static_shape() else {
            return Err(TypeError::invalid(
                "'rng_bit_generator' does not support dynamically shaped outputs".to_string(),
            )
            .into());
        };
        let count = output_shape.dimensions().iter().product::<usize>();
        let data_type = output_type.data_type();
        if !matches!(data_type, DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64) {
            return Err(TypeError::invalid(format!(
                "'rng_bit_generator' does not support output data type {data_type}",
            ))
            .into());
        }
        let expected_state_type = algorithm.state_type();
        if self.r#type.data_type() != expected_state_type.data_type()
            || self.r#type.shape() != expected_state_type.shape()
        {
            return Err(TypeError::invalid(format!(
                "'rng_bit_generator' with the {algorithm} algorithm needs a {expected_state_type} state but got {}",
                self.r#type,
            ))
            .into());
        }
        // Narrower-than-32-bit outputs retain the low bits of each generated `u32` word.
        let bits_from_u32_words = |words: Vec<u32>| match data_type {
            DataType::U32 => Array::from_elements(output_type.clone(), &words),
            DataType::U16 => Array::from_elements(
                output_type.clone(),
                &words.into_iter().map(|word| word as u16).collect::<Vec<_>>(),
            ),
            DataType::U8 => {
                Array::from_elements(output_type.clone(), &words.into_iter().map(|word| word as u8).collect::<Vec<_>>())
            }
            _ => unreachable!(),
        };
        match algorithm {
            RandomAlgorithm::ThreeFry => {
                // The state-type check above guarantees exactly two decoded `u64` elements.
                let [key, counter]: [u64; 2] = self.elements::<u64>()?.try_into().unwrap();
                let (new_counter, bits) = if data_type == DataType::U64 {
                    let (words, new_counter) = threefry_u64_words(key, counter, count);
                    (new_counter, Array::from_elements(output_type.clone(), &words)?)
                } else {
                    let (words, new_counter) = threefry_u32_words(key, counter, count);
                    (new_counter, bits_from_u32_words(words)?)
                };
                Ok((Array::from_elements(self.r#type.clone(), &[key, new_counter])?, bits))
            }
            RandomAlgorithm::Philox => {
                // The state-type check above guarantees exactly three decoded `u64` elements.
                let [key, counter_low, counter_high]: [u64; 3] = self.elements::<u64>()?.try_into().unwrap();
                let counter = u128::from(counter_low) | (u128::from(counter_high) << 64);
                let (new_counter, bits) = if data_type == DataType::U64 {
                    let (words, new_counter) = philox_u64_words(key, counter, count);
                    (new_counter, Array::from_elements(output_type.clone(), &words)?)
                } else {
                    let (words, new_counter) = philox_u32_words(key, counter, count);
                    (new_counter, bits_from_u32_words(words)?)
                };
                let advanced_state = [key, new_counter as u64, (new_counter >> 64) as u64];
                Ok((Array::from_elements(self.r#type.clone(), &advanced_state)?, bits))
            }
        }
    }
}

impl CustomCall for Array {
    /// The reference array backend has no foreign-kernel registry, so custom calls always report an
    /// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
    fn custom_call<'a, I: IntoIterator<Item = &'a Self>>(
        operation: &CustomCallOperation<ArrayType>,
        _inputs: I,
    ) -> Result<Vec<Self>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "the reference array backend cannot execute the foreign kernel '{}'",
                operation.target_name(),
            ),
        })
    }
}

impl Not for Array {
    fn not(&self) -> Result<Self, ProgramError> {
        let mask = match self.r#type.data_type() {
            DataType::Boolean | DataType::I1 | DataType::U1 => 0b1,
            DataType::I2 | DataType::U2 => 0b11,
            DataType::I4 | DataType::U4 => 0b1111,
            data_type if data_type.is_integer() => u8::MAX,
            data_type => {
                return Err(TypeError::invalid(format!(
                    "cannot apply `not` to an array of element data type {data_type}"
                ))
                .into());
            }
        };
        let addressing = ArrayAddressing::new(self.r#type.clone())?;
        let input_bytes = self.storage_bytes();
        let mut bytes = vec![0; addressing.storage_byte_len()];
        for element in 0..addressing.element_count() {
            for byte in addressing.byte_range_for_flat_index(element) {
                bytes[byte] = !input_bytes[byte] & mask;
            }
        }
        // Masking retains valid Boolean and sub-byte encodings; full-width integers admit every bit pattern, and
        // zero-initialization preserves all layout holes and padding.
        Ok(Self { r#type: self.r#type.clone(), bytes: Arc::new(bytes) })
    }
}

impl And for Array {
    fn and(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary_logical(rhs, "and", |left, right| left & right)
    }
}

impl Or for Array {
    fn or(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary_logical(rhs, "or", |left, right| left | right)
    }
}

impl Xor for Array {
    fn xor(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary_logical(rhs, "xor", |left, right| left ^ right)
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
            return Err(TypeError::invalid(format!(
                "'complex' requires identical part types but got {} and {}",
                self.r#type, imaginary.r#type,
            ))
            .into());
        }
        let data_type = match self.r#type.data_type() {
            DataType::F32 => DataType::C64,
            DataType::F64 => DataType::C128,
            other => {
                return Err(TypeError::invalid(format!(
                    "cannot construct a complex value from parts of data type {other}",
                ))
                .into());
            }
        };
        let output_type = self.r#type.clone().with_data_type(data_type);
        if data_type == DataType::C64 {
            self.binary_elements::<f32, Complex<f32>>(imaginary, output_type, |real, imaginary| {
                Ok(Complex::new(real, imaginary))
            })
        } else {
            self.binary_elements::<f64, Complex<f64>>(imaginary, output_type, |real, imaginary| {
                Ok(Complex::new(real, imaginary))
            })
        }
    }
}

impl Conjugate for Array {
    fn conjugate(&self) -> Result<Self, ProgramError> {
        match self.r#type.data_type() {
            DataType::C64 => {
                self.map_elements::<Complex<f32>, Complex<f32>>(self.r#type.clone(), |value| Ok(value.conj()))
            }
            DataType::C128 => {
                self.map_elements::<Complex<f64>, Complex<f64>>(self.r#type.clone(), |value| Ok(value.conj()))
            }
            other => Err(TypeError::invalid(format!("cannot conjugate a scalar of data type {other}")).into()),
        }
    }
}

impl Real for Array {
    fn real(&self) -> Result<Self, ProgramError> {
        // The real part of a complex array has the parts' real data type, mirroring the `RealOperation`
        // type-inference contract, which requires a complex operand.
        match self.r#type.data_type() {
            DataType::C64 => self
                .map_elements::<Complex<f32>, f32>(self.r#type.clone().with_data_type(DataType::F32), |value| {
                    Ok(value.re)
                }),
            DataType::C128 => self
                .map_elements::<Complex<f64>, f64>(self.r#type.clone().with_data_type(DataType::F64), |value| {
                    Ok(value.re)
                }),
            other => {
                Err(TypeError::invalid(format!("cannot extract the real part of a scalar of data type {other}")).into())
            }
        }
    }
}

impl Imaginary for Array {
    fn imaginary(&self) -> Result<Self, ProgramError> {
        // The imaginary part of a complex array has the parts' real data type, mirroring the `ImaginaryOperation`
        // type-inference contract, which requires a complex operand.
        match self.r#type.data_type() {
            DataType::C64 => self
                .map_elements::<Complex<f32>, f32>(self.r#type.clone().with_data_type(DataType::F32), |value| {
                    Ok(value.im)
                }),
            DataType::C128 => self
                .map_elements::<Complex<f64>, f64>(self.r#type.clone().with_data_type(DataType::F64), |value| {
                    Ok(value.im)
                }),
            other => {
                Err(TypeError::invalid(format!("cannot extract the imaginary part of a scalar of data type {other}",))
                    .into())
            }
        }
    }
}

impl Dot for Array {
    /// Computes an accumulation-typed dot by upcasting both operands to `accumulation_type` and delegating to the
    /// ordinary evaluator, which is exactly the upcast-then-accumulate contract of
    /// [`DotOperation::with_accumulation_type`].
    fn dot_with_accumulation_type(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Self {
        let lhs = self.convert_element_type(accumulation_type).unwrap_or_else(|error| panic!("{error}"));
        let rhs = rhs.convert_element_type(accumulation_type).unwrap_or_else(|error| panic!("{error}"));
        lhs.dot(&rhs, dimensions)
    }

    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Self {
        let lhs_shape = self.r#type.static_shape().unwrap();
        let rhs_shape = rhs.r#type.static_shape().unwrap();
        let lhs_values = self.scalar_values();
        let rhs_values = rhs.scalar_values();
        let zero = Self::zero_element(self.r#type.data_type()).unwrap_or_else(|error| panic!("{error}"));
        let (values, output_shape) = dot_general_evaluate(
            lhs_values.as_slice(),
            &lhs_shape,
            rhs_values.as_slice(),
            &rhs_shape,
            dimensions,
            || zero,
            |accumulator, lhs_value, rhs_value| accumulator + *lhs_value * *rhs_value,
        );
        let output_type = ArrayType::new(self.r#type.data_type(), Shape::from(&output_shape));
        Self::from_scalar_values(output_type, values).unwrap()
    }
}

impl Reduce for Array {
    fn reduce(&self, axes: &[usize], kind: ReductionKind) -> Self {
        if axes.is_empty() {
            return self.clone();
        }
        let data_type = self.r#type.data_type();
        let shape = self.r#type.static_shape().unwrap();
        let input_values = self.scalar_values();
        let (mut values, reduced_shape) = match kind {
            ReductionKind::Sum | ReductionKind::Mean => {
                let zero = Self::zero_element(data_type).unwrap_or_else(|error| panic!("{error}"));
                reduce_evaluate(
                    input_values.as_slice(),
                    &shape,
                    axes,
                    || zero,
                    |accumulator, value| accumulator + value,
                )
            }
            ReductionKind::Max => reduce_extremum(&input_values, &shape, axes, ComparisonDirection::GreaterThan),
            ReductionKind::Min => reduce_extremum(&input_values, &shape, axes, ComparisonDirection::LessThan),
            ReductionKind::Any => reduce_evaluate(
                input_values.as_slice(),
                &shape,
                axes,
                || Scalar::Bool(false),
                |accumulator, value| accumulator | value,
            ),
            ReductionKind::All => reduce_evaluate(
                input_values.as_slice(),
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
        // The eager payload kernel computes only the reduced shape. Reuse the operation's abstract rule for the
        // complete result type so interpretation preserves memory placement, projects sharding, and clears the
        // rank-specific layout exactly like tracing and compiled backends do.
        let output_type = reduce_abstract(&self.r#type, axes, kind, "reduce").unwrap_or_else(|error| panic!("{error}"));
        debug_assert_eq!(output_type.shape(), &Shape::from(&reduced_shape));
        Self::from_scalar_values(output_type, values).unwrap()
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
    fn transpose<P: Into<Permutation>>(&self, permutation: P) -> Result<Self, ProgramError> {
        // Validate the permutation and compute the output type (including sharding) via the type-level rule, so an
        // out-of-range or duplicated axis is a clean error rather than an out-of-bounds panic.
        let permutation = permutation.into();
        let output_type = self.r#type.transpose(permutation.clone())?;
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return Ok(self.clone());
        }
        let shape = self.r#type.static_shape().unwrap();
        let rank = shape.rank();
        let permuted_shape = StaticShape::new(permutation.iter().map(|axis| shape[*axis]).collect());
        let input_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        if output_addressing.element_count() == 0 {
            return Ok(Self { r#type: output_type, bytes: Arc::new(bytes) });
        }
        let input_strides = shape.row_major_strides();
        let mut permuted_index = vec![0usize; rank];
        for output_flat in 0..output_addressing.element_count() {
            let mut input_flat = 0usize;
            for (position, &input_axis) in permutation.iter().enumerate() {
                input_flat += permuted_index[position] * input_strides[input_axis];
            }
            bytes[output_addressing.byte_range_for_flat_index(output_flat)]
                .copy_from_slice(&self.bytes[input_addressing.byte_range_for_flat_index(input_flat)]);
            for position in (0..rank).rev() {
                permuted_index[position] += 1;
                if permuted_index[position] < permuted_shape[position] {
                    break;
                }
                permuted_index[position] = 0;
            }
        }
        Ok(Self { r#type: output_type, bytes: Arc::new(bytes) })
    }
}

impl Reshape for Array {
    fn reshape<P: Into<ReshapeParameters>>(&self, parameters: P) -> Result<Self, ProgramError> {
        // Resolve runtime dimension expressions from the concrete eager input shape, then delegate to the type-level
        // reshape so all element-count and sharding validation remains shared with staged execution.
        let parameters = parameters.into().resolve_target(self.r#type.shape())?;
        let output_type = self.r#type.reshape(parameters.clone())?;
        let transposed = parameters.dimensions().map(|dimensions| self.transpose(dimensions)).transpose()?;
        let input = transposed.as_ref().unwrap_or(self);
        let input_addressing = ArrayAddressing::new(input.r#type.clone())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        if input_addressing.is_dense_row_major() && output_addressing.is_dense_row_major() {
            bytes.copy_from_slice(&input.bytes);
        } else {
            for index in 0..input_addressing.element_count() {
                bytes[output_addressing.byte_range_for_flat_index(index)]
                    .copy_from_slice(&input.bytes[input_addressing.byte_range_for_flat_index(index)]);
            }
        }
        Ok(Self { r#type: output_type, bytes: Arc::new(bytes) })
    }
}

impl BroadcastKernel for Array {
    fn broadcast_to_type(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        let r#type = self.r#type.legacy_broadcast(output_type, output_axes)?;
        let Some(target_shape) = r#type.static_shape() else {
            return Err(
                TypeError::invalid(format!("cannot materialize a value of dynamically sized type {}", r#type)).into()
            );
        };
        if r#type == self.r#type && output_axes.iter().copied().eq(0..r#type.rank()) {
            return Ok(self.clone());
        }
        let input_shape = self.r#type.static_shape().unwrap();
        let input_rank = input_shape.rank();
        let target_rank = target_shape.rank();
        let output_count = Self::materialized_element_count(&r#type)?;
        let input_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let output_addressing = ArrayAddressing::new(r#type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        if output_count == 0 {
            return Ok(Self { r#type, bytes: Arc::new(bytes) });
        }
        let input_strides = input_shape.row_major_strides();
        let mut target_index = vec![0usize; target_rank];
        for output_flat in 0..output_count {
            let mut input_flat = 0usize;
            for input_axis in 0..input_rank {
                let target_axis = output_axes[input_axis];
                let coordinate = if input_shape[input_axis] == 1 { 0 } else { target_index[target_axis] };
                input_flat += coordinate * input_strides[input_axis];
            }
            bytes[output_addressing.byte_range_for_flat_index(output_flat)]
                .copy_from_slice(&self.bytes[input_addressing.byte_range_for_flat_index(input_flat)]);
            for position in (0..target_rank).rev() {
                target_index[position] += 1;
                if target_index[position] < target_shape[position] {
                    break;
                }
                target_index[position] = 0;
            }
        }
        Ok(Self { r#type, bytes: Arc::new(bytes) })
    }
}

impl LegacyBroadcast for Array {
    #[inline]
    fn legacy_broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        self.broadcast_to_type(output_type, output_axes)
    }
}

impl Array {
    /// Copies the logical block selected by `axes` into a new array of `output_type`. The caller guarantees that the
    /// selection lies in bounds and contains exactly the output's logical element count.
    fn copy_block(&self, output_type: ArrayType, axes: &[ArraySliceAxis]) -> Result<Self, ProgramError> {
        let input_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let ranges = input_addressing.ranges(axes)?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        debug_assert_eq!(ranges.element_count(), output_addressing.element_count());
        let element_byte_width = input_addressing.element_byte_width();
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        let output_is_dense = output_addressing.is_dense_row_major();
        let mut output_index = 0usize;
        for range in ranges {
            let input_bytes = range.bytes();
            let element_count = range.elements().len();
            if output_is_dense {
                let output_start = output_index * element_byte_width;
                bytes[output_start..output_start + input_bytes.len()].copy_from_slice(&self.bytes[input_bytes]);
                output_index += element_count;
                continue;
            }
            for offset in 0..element_count {
                let input_start = input_bytes.start + offset * element_byte_width;
                bytes[output_addressing.byte_range_for_flat_index(output_index)]
                    .copy_from_slice(&self.bytes[input_start..input_start + element_byte_width]);
                output_index += 1;
            }
        }
        debug_assert_eq!(output_index, output_addressing.element_count());
        Ok(Self { r#type: output_type, bytes: Arc::new(bytes) })
    }

    /// Overwrites the logical block of `update`'s shape starting at `start_indices` in this array with `update`. The
    /// caller guarantees that the block lies in bounds.
    fn replace_block(self, update: &Array, start_indices: &[usize]) -> Self {
        let update_shape = update.r#type.static_shape().unwrap();
        let addressing = ArrayAddressing::new(self.r#type.clone()).unwrap();
        let update_addressing = ArrayAddressing::new(update.r#type.clone()).unwrap();
        let axes = start_indices
            .iter()
            .zip(update_shape.dimensions())
            .map(|(start, size)| ArraySliceAxis::new(*start, *size, 1))
            .collect::<Vec<_>>();
        let ranges = addressing.ranges(&axes).unwrap();
        let element_byte_width = addressing.element_byte_width();
        let mut output = self;
        let bytes = Arc::make_mut(&mut output.bytes);
        let update_is_dense = update_addressing.is_dense_row_major();
        let mut written = 0usize;
        for range in ranges {
            let output_bytes = range.bytes();
            let element_count = range.elements().len();
            if update_is_dense {
                let update_start = written * element_byte_width;
                bytes[output_bytes]
                    .copy_from_slice(&update.bytes[update_start..update_start + element_count * element_byte_width]);
                written += element_count;
                continue;
            }
            for offset in 0..element_count {
                let output_start = output_bytes.start + offset * element_byte_width;
                bytes[output_start..output_start + element_byte_width]
                    .copy_from_slice(&update.bytes[update_addressing.byte_range_for_flat_index(written)]);
                written += 1;
            }
        }
        debug_assert_eq!(written, update_addressing.element_count());
        output
    }

    /// Extracts the in-band scalar start indices of a dynamic slicing operation and clamps them per StableHLO
    /// semantics: the effective start index along axis `d` is
    /// `clamp(0, start_indices[d], input_dimension[d] - block_sizes[d])`.
    fn clamped_start_indices(start_indices: &[Array], input_shape: &StaticShape, block_sizes: &[usize]) -> Vec<usize> {
        start_indices
            .iter()
            .enumerate()
            .map(|(axis, index)| {
                let addressing = ArrayAddressing::new(index.r#type.clone()).unwrap();
                let raw = index.index_value(&addressing, 0);
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
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        let output_type =
            self.r#type.pad(&padding_value.r#type, edge_padding_low, edge_padding_high, interior_padding)?;
        let input_shape = self.r#type.static_shape().unwrap();
        if edge_padding_low.iter().all(|padding| *padding == 0)
            && edge_padding_high.iter().all(|padding| *padding == 0)
            && input_shape
                .dimensions()
                .iter()
                .zip(interior_padding)
                .all(|(size, padding)| *padding == 0 || *size <= 1)
        {
            return Ok(self.clone());
        }
        let output_shape = output_type.static_shape().unwrap();
        let rank = input_shape.rank();
        let input_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let padding_addressing = ArrayAddressing::new(padding_value.r#type.clone())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let padding_bytes = &padding_value.bytes[padding_addressing.byte_range_for_flat_index(0)];
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        if output_addressing.is_dense_row_major() && output_addressing.element_byte_width() != 0 {
            for output_bytes in bytes.chunks_exact_mut(output_addressing.element_byte_width()) {
                output_bytes.copy_from_slice(padding_bytes);
            }
        } else {
            for output_index in 0..output_addressing.element_count() {
                bytes[output_addressing.byte_range_for_flat_index(output_index)].copy_from_slice(padding_bytes);
            }
        }
        if input_addressing.element_count() == 0 {
            return Ok(Self { r#type: output_type, bytes: Arc::new(bytes) });
        }
        let output_strides = output_shape.row_major_strides();
        let mut input_index = vec![0usize; rank];
        let mut written = 0usize;
        'elements: while written < input_addressing.element_count() {
            let mut output_flat = 0usize;
            for axis in 0..rank {
                let input_coordinate = i128::try_from(input_index[axis])
                    .map_err(|_| TypeError::invalid(format!("'pad' input index is too large on axis {axis}")))?;
                let stride = i128::try_from(interior_padding[axis])
                    .ok()
                    .and_then(|padding| padding.checked_add(1))
                    .ok_or_else(|| TypeError::invalid(format!("'pad' stride is too large on axis {axis}")))?;
                let output_index =
                    i128::from(edge_padding_low[axis])
                        .checked_add(input_coordinate.checked_mul(stride).ok_or_else(|| {
                            TypeError::invalid(format!("'pad' output index overflows on axis {axis}"))
                        })?)
                        .ok_or_else(|| TypeError::invalid(format!("'pad' output index overflows on axis {axis}")))?;
                let output_extent = i128::try_from(output_shape[axis])
                    .map_err(|_| TypeError::invalid(format!("'pad' output extent is too large on axis {axis}")))?;
                if output_index < 0 || output_index >= output_extent {
                    written += 1;
                    for position in (0..rank).rev() {
                        input_index[position] += 1;
                        if input_index[position] < input_shape[position] {
                            break;
                        }
                        input_index[position] = 0;
                    }
                    continue 'elements;
                }
                let output_index = usize::try_from(output_index)
                    .map_err(|_| TypeError::invalid(format!("'pad' output index is too large on axis {axis}")))?;
                output_flat =
                    output_flat
                        .checked_add(output_index.checked_mul(output_strides[axis]).ok_or_else(|| {
                            TypeError::invalid(format!("'pad' output index overflows on axis {axis}"))
                        })?)
                        .ok_or_else(|| TypeError::invalid(format!("'pad' output index overflows on axis {axis}")))?;
            }
            bytes[output_addressing.byte_range_for_flat_index(output_flat)]
                .copy_from_slice(&self.bytes[input_addressing.byte_range_for_flat_index(written)]);
            written += 1;
            for position in (0..rank).rev() {
                input_index[position] += 1;
                if input_index[position] < input_shape[position] {
                    break;
                }
                input_index[position] = 0;
            }
        }
        Ok(Self { r#type: output_type, bytes: Arc::new(bytes) })
    }
}

impl Concatenate for Array {
    fn concatenate<'i, I: IntoIterator<Item = &'i Self>, A: Into<Axis>>(
        inputs: I,
        axis: A,
    ) -> Result<Self, ProgramError> {
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        let Some(first) = inputs.first() else {
            return Err(
                TypeError::invalid("'concatenate' expects at least one operand but got none".to_string()).into()
            );
        };
        if inputs.len() == 1 {
            return Ok((*first).clone());
        }
        let operation = ConcatenateOperation::new(axis, first.r#type.rank())?;
        let axis = operation.axis();
        let output_type = ArrayType::concatenate(inputs.iter().map(|input| &input.r#type), axis)?;
        // Each operand owns a contiguous run of `axis` coordinates. Write its logical block at the running offset
        // along `axis` and offset zero on every other axis.
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        // Zero-initialization establishes every layout hole and tile-padding byte. Every logical element is replaced
        // below, so this does not require the element data type itself to represent zero.
        let mut output =
            Self { r#type: output_type.clone(), bytes: Arc::new(vec![0; output_addressing.storage_byte_len()]) };
        let mut offset = 0usize;
        for input in inputs {
            let input_axis_size = input.r#type.static_shape().unwrap()[axis];
            let mut start_indices = vec![0usize; output_type.rank()];
            start_indices[axis] = offset;
            output = output.replace_block(input, start_indices.as_slice());
            offset += input_axis_size;
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

        // Only `FillOrDrop` needs an out-of-bounds fill element. Construct it through the ordinary array capability so
        // this kernel does not assume an all-zero encoding; the other modes therefore also support element formats
        // such as F8E8M0FNU that cannot represent zero at all.
        let dropped_fill = if operation.mode() == GatherScatterMode::FillOrDrop {
            let value = EagerContext::<Array>::new().zero(&ArrayType::scalar(output_type.data_type()))?;
            let addressing = ArrayAddressing::new(value.r#type.clone())?;
            Some((value, addressing))
        } else {
            None
        };
        let input_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let indices_addressing = ArrayAddressing::new(indices.r#type.clone())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let extents = output_shape.dimensions();
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        let mut output_index = vec![0usize; output_rank];
        let mut indices_index = vec![0usize; indices_rank];
        let mut starts = vec![0i64; index_vector_extent];
        let mut operand_index = vec![0i64; operand_rank];
        for output_element in 0..output_addressing.element_count() {
            // Place the output's batch coordinates into the indices multi-index and read this query's start vector.
            indices_index.fill(0);
            for (position, &output_position) in batch_output_positions.iter().enumerate() {
                indices_index[indices_batch_axes[position]] = output_index[output_position];
            }
            for (component, start) in starts.iter_mut().enumerate() {
                indices_index[index_vector_dimension] = component;
                let flat: usize = (0..indices_rank).map(|axis| indices_index[axis] * indices_strides[axis]).sum();
                *start = indices.index_value(&indices_addressing, flat);
            }
            // Assemble the operand multi-index: window offsets, then batching coordinates, then start offsets.
            operand_index.fill(0);
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
            let source = if dropped {
                let (value, addressing) = dropped_fill.as_ref().unwrap();
                &value.bytes[addressing.byte_range_for_flat_index(0)]
            } else {
                let flat: usize =
                    (0..operand_rank).map(|axis| operand_index[axis] as usize * operand_strides[axis]).sum();
                &self.bytes[input_addressing.byte_range_for_flat_index(flat)]
            };
            bytes[output_addressing.byte_range_for_flat_index(output_element)].copy_from_slice(source);
            for position in (0..output_rank).rev() {
                output_index[position] += 1;
                if output_index[position] < extents[position] {
                    break;
                }
                output_index[position] = 0;
            }
        }
        Ok(Self { r#type: output_type, bytes: Arc::new(bytes) })
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
        let indices_addressing = ArrayAddressing::new(indices.r#type.clone())?;
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

        let mut values = self.scalar_values();
        let update_values = updates.scalar_values();
        let extents = updates_shape.dimensions();
        let update_count: usize = extents.iter().product();
        let mut update_index = vec![0usize; updates_rank];
        let mut indices_index = vec![0usize; indices_rank];
        let mut starts = vec![0i64; index_vector_extent];
        let mut operand_index = vec![0i64; operand_rank];
        for written in 0..update_count {
            indices_index.fill(0);
            for (position, &update_axis) in update_scatter_axes.iter().enumerate() {
                indices_index[indices_batch_axes[position]] = update_index[update_axis];
            }
            for (component, start) in starts.iter_mut().enumerate() {
                indices_index[index_vector_dimension] = component;
                let flat: usize = (0..indices_rank).map(|axis| indices_index[axis] * indices_strides[axis]).sum();
                *start = indices.index_value(&indices_addressing, flat);
            }
            operand_index.fill(0);
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
                values[flat] = combine_scatter(operation.kind(), values[flat], update_values[written]);
            }
            for position in (0..updates_rank).rev() {
                update_index[position] += 1;
                if update_index[position] < extents[position] {
                    break;
                }
                update_index[position] = 0;
            }
        }
        Self::from_scalar_values(output_type, values)
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
        let axes = start_indices
            .iter()
            .zip(limit_indices.iter())
            .zip(strides.iter())
            .map(|((start, limit), stride)| ArraySliceAxis::new(*start, (limit - start).div_ceil(*stride), *stride))
            .collect::<Vec<_>>();
        self.copy_block(output_type, &axes)
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
        let axes = starts
            .iter()
            .zip(sizes)
            .map(|(start, size)| ArraySliceAxis::new(*start, *size, 1))
            .collect::<Vec<_>>();
        self.copy_block(output_type, &axes)
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
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self, ProgramError> {
        // Broadcast the operand types together (including element-type promotion) so mixed-precision comparisons
        // mirror the `CompareOperation` type-inference contract, then compare the promoted elements pairwise. The
        // output type is the Boolean-typed counterpart of the broadcast type.
        let (broadcast_type, operands) = Self::broadcast_promoted(&[self, rhs])?;
        let target = broadcast_type.data_type();
        let output_type = broadcast_type.with_element_type(DataType::Boolean);
        // Empty comparisons perform no element operation, so even payload-free data types retain the vacuous success
        // behavior of the former scalar loop.
        if Self::element_count(&output_type) == 0 {
            let addressing = ArrayAddressing::new(output_type.clone())?;
            return Ok(Self { r#type: output_type, bytes: Arc::new(vec![0; addressing.storage_byte_len()]) });
        }
        if target == DataType::Token {
            return Err(TypeError::invalid("cannot compare token scalars".to_string()).into());
        }
        if target == DataType::Zero {
            return Err(TypeError::invalid("cannot compare scalars of data types zero and zero".to_string()).into());
        }

        // `broadcast_promoted` converts only mismatched inputs, so equal-typed inputs retain their exact physical
        // storage and are decoded one addressed element at a time by the shared binary loop.
        let [left, right] = <[_; 2]>::try_from(operands).unwrap();
        left.compare_elements(&right, output_type, direction)
    }
}

impl Select for Array {
    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        // Mirrors the broadcasting `SelectOperation` type-inference contract: the condition must be Boolean-typed,
        // the three operand shapes broadcast together, and the two branch data types promote together to the output
        // data type. The condition is retyped to a branch data type before broadcasting so its Boolean data type
        // acts as a mask rather than promoting into the output.
        assert_eq!(condition.r#type.data_type(), DataType::Boolean, "select condition must have a Boolean data type",);
        let output_type = ArrayType::broadcasted(&[
            condition.r#type.clone().with_data_type(on_true.r#type.data_type()),
            on_true.r#type.clone(),
            on_false.r#type.clone(),
        ])
        .map_err(|error| TypeError::invalid(error.to_string()))?;

        // Convert only when promotion requires it. Equal-typed branches retain their original physical storage and
        // arbitrary layouts; conversion remains responsible for the element semantics until its own typed-byte slice.
        let output_data_type = output_type.data_type();
        let on_true = on_true.promoted_to(output_data_type)?;
        let on_false = on_false.promoted_to(output_data_type)?;

        let output_shape = output_type.static_shape().unwrap();
        let condition_shape = condition.r#type.static_shape().unwrap();
        let true_shape = on_true.r#type.static_shape().unwrap();
        let false_shape = on_false.r#type.static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let condition_strides = condition_shape.row_major_strides();
        let true_strides = true_shape.row_major_strides();
        let false_strides = false_shape.row_major_strides();
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let condition_addressing = ArrayAddressing::new(condition.r#type.clone())?;
        let true_addressing = ArrayAddressing::new(on_true.r#type.clone())?;
        let false_addressing = ArrayAddressing::new(on_false.r#type.clone())?;
        let mut output_bytes = vec![0; output_addressing.storage_byte_len()];
        for output_index in 0..output_addressing.element_count() {
            let condition_index = Self::broadcast_index(
                output_index,
                &output_shape,
                &output_strides,
                &condition_shape,
                &condition_strides,
            );
            let condition_range = condition_addressing.byte_range_for_flat_index(condition_index);
            let (source, source_range) = if condition.bytes[condition_range.start] != 0 {
                let source_index =
                    Self::broadcast_index(output_index, &output_shape, &output_strides, &true_shape, &true_strides);
                (&on_true.bytes, true_addressing.byte_range_for_flat_index(source_index))
            } else {
                let source_index =
                    Self::broadcast_index(output_index, &output_shape, &output_strides, &false_shape, &false_strides);
                (&on_false.bytes, false_addressing.byte_range_for_flat_index(source_index))
            };
            let output_range = output_addressing.byte_range_for_flat_index(output_index);
            output_bytes[output_range].copy_from_slice(&source[source_range]);
        }
        Ok(Self { r#type: output_type, bytes: Arc::new(output_bytes) })
    }
}

impl Concretizable<bool> for Array {
    fn concretize(&self) -> Result<bool, ProgramError> {
        // Accept scalar Boolean predicates (rank-0, one element) so that batch-varying while can extract a final
        // `any(mask)` result. Higher-rank predicates still error because they cannot collapse to a single Boolean.
        if self.r#type.rank() == 0 && self.r#type.data_type().is_boolean() {
            let range = ArrayAddressing::new(self.r#type.clone())?.byte_range_for_flat_index(0);
            return Ok(self.bytes[range.start] != 0);
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
        if !self.r#type.data_type().is_boolean() {
            return Err(ProgramError::Concretization {
                message: format!("cannot use a value of type {} as a Boolean while predicate", self.r#type),
            });
        }
        let addressing = ArrayAddressing::new(self.r#type.clone())?;
        Ok((0..addressing.element_count())
            .any(|index| self.bytes[addressing.byte_range_for_flat_index(index).start] != 0))
    }

    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let predicate_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let true_addressing = ArrayAddressing::new(on_true.r#type.clone())?;
        if !self.r#type.data_type().is_boolean()
            || on_true.r#type != on_false.r#type
            || predicate_addressing.element_count() == 0
            || !true_addressing.element_count().is_multiple_of(predicate_addressing.element_count())
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "mask_select requires a Boolean predicate whose element count divides congruent operands, but \
                     got predicate {} with operands {} and {}",
                    self.r#type, on_true.r#type, on_false.r#type,
                ),
            });
        }
        let block = true_addressing.element_count() / predicate_addressing.element_count();
        let mut output_bytes = vec![0; true_addressing.storage_byte_len()];
        for index in 0..true_addressing.element_count() {
            let predicate_range = predicate_addressing.byte_range_for_flat_index(index / block);
            let source = if self.bytes[predicate_range.start] != 0 { &on_true.bytes } else { &on_false.bytes };
            let source_range = true_addressing.byte_range_for_flat_index(index);
            output_bytes[source_range.clone()].copy_from_slice(&source[source_range]);
        }
        Ok(Self { r#type: on_true.r#type.clone(), bytes: Arc::new(output_bytes) })
    }
}

impl ConvertElementType for Array {
    fn convert_element_type(&self, data_type: DataType) -> Result<Self, ProgramError> {
        if self.r#type.data_type().is_token() || data_type.is_token() {
            return Err(TypeError::invalid("cannot convert values to or from the token data type".to_string()).into());
        }
        let values = self
            .scalar_values()
            .iter()
            .map(|value| value.convert_element_type(data_type))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_scalar_values(self.r#type.clone().with_data_type(data_type), values)
    }
}

impl TransferToMemory for Array {
    /// Re-places this [`Array`] in `destination` by updating the [`Memory`](crate::types::Memory) carried by its
    /// type. The payload is host-resident either way, but the carried type must reflect the transfer so that staged
    /// programs whose declared types park values in other memories (e.g., offloaded residuals) accept the
    /// interpreted value.
    #[inline]
    fn transfer_to_memory(&self, destination: crate::types::Memory) -> Self {
        Self { r#type: self.r#type.clone().with_memory(destination), bytes: self.bytes.clone() }
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
        Self { r#type, bytes: self.bytes.clone() }
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
    use half::{bf16, f16};
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        f4e2m1fn, f6e2m3fn, f6e3m2fn, f8e3m4, f8e4m3, f8e4m3b11fnuz, f8e4m3fn, f8e4m3fnuz, f8e5m2, f8e5m2fnuz,
        f8e8m0fnu, i1, i2, i4, u1, u2, u4,
    };
    use crate::operations::complex::Complex;
    use crate::operations::constants::Iota;
    use crate::operations::manipulation::{GatherDimensionNumbers, ScatterDimensionNumbers};
    use crate::operations::sharding::{ConstrainSharding, Reshard};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{DimensionBounds, DimensionVariable, Layout, Memory, StridedLayout};

    use super::*;

    /// Creates a static [`ArrayType`] with the provided element data type and dimension sizes.
    fn array_type(data_type: DataType, dimensions: &[usize]) -> ArrayType {
        ArrayType::new(data_type, Shape::new(dimensions.iter().map(|size| Dimension::Static(*size)).collect()))
    }

    #[test]
    fn test_array_construction_enforces_storage_invariants() {
        // Typed logical elements must match the declared element data type.
        assert!(matches!(
            Array::from_elements(array_type(DataType::F64, &[2]), &[1.0f32, 2.0]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot encode f32 values as array elements of data type f64",
        ));
        // The logical element count must match the static shape.
        assert!(matches!(
            Array::from_elements(array_type(DataType::F64, &[3]), &[1.0f64]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array type f64[3] requires 3 logical elements but got 1",
        ));
        // Dynamically shaped types cannot describe materialized storage.
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        );
        assert!(matches!(
            Array::from_elements(dynamic_type, &[1.0f64]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot materialize a value of dynamically sized type f64[dynamic]",
        ));
        // Well-formed logical elements construct successfully and round-trip through typed and byte accessors.
        let array = Array::from_elements(array_type(DataType::F64, &[2]), &[1.0f64, 2.0]).unwrap();
        assert_eq!(array.r#type().into_owned(), array_type(DataType::F64, &[2]));
        assert_eq!(array.elements::<f64>(), Ok(vec![1.0, 2.0]));
        assert_eq!(array.storage_bytes(), array.logical_bytes());
    }

    #[test]
    fn test_array_boolean_and_integer_encoding_round_trips() {
        macro_rules! check_integer_round_trip {
            ($data_type:expr, $element_type:ty, $values:expr $(,)?) => {{
                let values: &[$element_type] = &$values;
                let r#type = array_type($data_type, &[values.len()]);
                let expected_bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
                let array = Array::from_elements(r#type.clone(), values).unwrap();
                assert_eq!(array.storage_bytes(), expected_bytes);
                assert_eq!(array.logical_bytes(), expected_bytes);
                assert_eq!(array.elements::<$element_type>(), Ok(values.to_vec()));
                assert_eq!(Array::new(r#type.clone(), expected_bytes.clone()).unwrap().elements(), Ok(values.to_vec()));
                assert_eq!(Array::from_logical_bytes(r#type, &expected_bytes).unwrap().elements(), Ok(values.to_vec()));
            }};
        }

        let booleans = Array::from_elements(array_type(DataType::Boolean, &[2]), &[false, true]).unwrap();
        assert_eq!(booleans.storage_bytes(), [0, 1]);
        assert_eq!(booleans.logical_bytes(), [0, 1]);
        assert_eq!(booleans.elements::<bool>(), Ok(vec![false, true]));

        check_integer_round_trip!(DataType::I8, i8, [i8::MIN, -1, 0, i8::MAX]);
        check_integer_round_trip!(DataType::I16, i16, [i16::MIN, -0x1234, 0x2345, i16::MAX]);
        check_integer_round_trip!(DataType::I32, i32, [i32::MIN, -0x1234_567, 0x2345_678, i32::MAX]);
        check_integer_round_trip!(DataType::I64, i64, [i64::MIN, -0x1234_5678_9abc_def, i64::MAX]);
        check_integer_round_trip!(DataType::U8, u8, [0, 0x12, 0xfe, u8::MAX]);
        check_integer_round_trip!(DataType::U16, u16, [0, 0x1234, 0xfedc, u16::MAX]);
        check_integer_round_trip!(DataType::U32, u32, [0, 0x1234_5678, 0xfedc_ba98, u32::MAX]);
        check_integer_round_trip!(DataType::U64, u64, [0, (1_u64 << 53) + 1, u64::MAX - 1, u64::MAX]);
    }

    #[test]
    fn test_array_sub_byte_integer_construction_and_validation() {
        macro_rules! check_sub_byte_integer {
            ($data_type:expr, $element_type:ty, $values:expr, $expected_bytes:expr, $invalid_byte:expr $(,)?) => {{
                let values: &[$element_type] = &$values;
                let expected_bytes: &[u8] = &$expected_bytes;
                let r#type = array_type($data_type, &[values.len()]);

                // Typed construction must preserve native signedness while storing only each element's low bits.
                let array = Array::from_elements(r#type.clone(), values).unwrap();
                assert_eq!(array.storage_bytes(), expected_bytes);
                assert_eq!(array.logical_bytes(), expected_bytes);
                assert_eq!(array.elements::<$element_type>(), Ok(values.to_vec()));

                // Both raw-byte construction paths accept the same valid encoding.
                assert_eq!(
                    Array::new(r#type.clone(), expected_bytes.to_vec()).unwrap().elements(),
                    Ok(values.to_vec()),
                );
                assert_eq!(Array::from_logical_bytes(r#type, expected_bytes).unwrap().elements(), Ok(values.to_vec()));

                // A set bit above the data type's width must be rejected at the array ownership boundary.
                assert!(matches!(
                    Array::new(array_type($data_type, &[1]), vec![$invalid_byte]),
                    Err(ProgramError::Type(TypeError::Invalid { message }))
                        if message == format!(
                            "array element 0 has invalid {} byte encoding [{}]",
                            $data_type,
                            $invalid_byte,
                        ),
                ));
            }};
        }

        check_sub_byte_integer!(DataType::I1, i1, [i1::MIN, i1::MAX], [0x01, 0x00], 0x02);
        check_sub_byte_integer!(
            DataType::I2,
            i2,
            [i2::MIN, i2::new(-1).unwrap(), i2::new(0).unwrap(), i2::MAX],
            [0x02, 0x03, 0x00, 0x01],
            0x04,
        );
        check_sub_byte_integer!(
            DataType::I4,
            i4,
            [i4::MIN, i4::new(-1).unwrap(), i4::new(0).unwrap(), i4::MAX],
            [0x08, 0x0f, 0x00, 0x07],
            0x10,
        );
        check_sub_byte_integer!(DataType::U1, u1, [u1::MIN, u1::MAX], [0x00, 0x01], 0x02);
        check_sub_byte_integer!(
            DataType::U2,
            u2,
            [u2::MIN, u2::new(1).unwrap(), u2::new(2).unwrap(), u2::MAX],
            [0x00, 0x01, 0x02, 0x03],
            0x04,
        );
        check_sub_byte_integer!(
            DataType::U4,
            u4,
            [u4::MIN, u4::new(1).unwrap(), u4::new(14).unwrap(), u4::MAX],
            [0x00, 0x01, 0x0e, 0x0f],
            0x10,
        );
    }

    #[test]
    fn test_array_floating_point_encoding_round_trips() {
        macro_rules! check_float_round_trip {
            ($data_type:expr, $element_type:ty, $bit_type:ty, $bits:expr $(,)?) => {{
                let bits: &[$bit_type] = &$bits;
                let values = bits.iter().copied().map(<$element_type>::from_bits).collect::<Vec<_>>();
                let r#type = array_type($data_type, &[values.len()]);
                let expected_bytes = bits.iter().flat_map(|bits| bits.to_le_bytes()).collect::<Vec<_>>();
                let array = Array::from_elements(r#type.clone(), &values).unwrap();
                assert_eq!(array.storage_bytes(), expected_bytes);
                assert_eq!(array.logical_bytes(), expected_bytes);
                assert_eq!(
                    array
                        .elements::<$element_type>()
                        .unwrap()
                        .into_iter()
                        .map(<$element_type>::to_bits)
                        .collect::<Vec<_>>(),
                    bits,
                );
                assert_eq!(Array::new(r#type.clone(), expected_bytes.clone()).unwrap().logical_bytes(), expected_bytes);
                assert_eq!(Array::from_logical_bytes(r#type, &expected_bytes).unwrap().logical_bytes(), expected_bytes);
            }};
        }

        check_float_round_trip!(DataType::BF16, bf16, u16, [0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc1]);
        check_float_round_trip!(DataType::F16, f16, u16, [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e01]);
        check_float_round_trip!(
            DataType::F32,
            f32,
            u32,
            [0x0000_0000, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_1234]
        );
        check_float_round_trip!(
            DataType::F64,
            f64,
            u64,
            [
                0x0000_0000_0000_0000,
                0x8000_0000_0000_0000,
                0x7ff0_0000_0000_0000,
                0xfff0_0000_0000_0000,
                0x7ff8_0000_0000_1234
            ],
        );

        macro_rules! check_low_precision_round_trip {
            ($data_type:expr, $element_type:ty, $bits:expr $(,)?) => {{
                let bits: &[u8] = &$bits;
                let r#type = array_type($data_type, &[bits.len()]);
                let array = Array::from_logical_bytes(r#type.clone(), bits).unwrap();
                let elements = array.elements::<$element_type>().unwrap();
                assert_eq!(array.storage_bytes(), bits);
                assert_eq!(array.logical_bytes(), bits);
                assert_eq!(elements.iter().copied().map(<$element_type>::to_bits).collect::<Vec<_>>(), bits);
                assert_eq!(Array::from_elements(r#type.clone(), &elements).unwrap().storage_bytes(), bits);
                assert_eq!(Array::new(r#type, bits.to_vec()).unwrap().logical_bytes(), bits);
            }};
        }

        check_low_precision_round_trip!(DataType::F4E2M1FN, f4e2m1fn, [0x00, 0x08, 0x07, 0x0f]);
        check_low_precision_round_trip!(DataType::F6E2M3FN, f6e2m3fn, [0x00, 0x20, 0x1f, 0x3f]);
        check_low_precision_round_trip!(DataType::F6E3M2FN, f6e3m2fn, [0x00, 0x20, 0x1f, 0x3f]);
        check_low_precision_round_trip!(DataType::F8E3M4, f8e3m4, [0x00, 0x80, 0x70, 0xf0, 0x79]);
        check_low_precision_round_trip!(DataType::F8E4M3, f8e4m3, [0x00, 0x80, 0x78, 0xf8, 0x7d]);
        check_low_precision_round_trip!(DataType::F8E4M3FN, f8e4m3fn, [0x00, 0x80, 0x7e, 0xfe, 0x7f]);
        check_low_precision_round_trip!(DataType::F8E4M3FNUZ, f8e4m3fnuz, [0x00, 0x01, 0x7f, 0x80]);
        check_low_precision_round_trip!(DataType::F8E4M3B11FNUZ, f8e4m3b11fnuz, [0x00, 0x01, 0x7f, 0x80]);
        check_low_precision_round_trip!(DataType::F8E5M2, f8e5m2, [0x00, 0x80, 0x7c, 0xfc, 0x7e]);
        check_low_precision_round_trip!(DataType::F8E5M2FNUZ, f8e5m2fnuz, [0x00, 0x01, 0x7f, 0x80]);
        check_low_precision_round_trip!(DataType::F8E8M0FNU, f8e8m0fnu, [0x00, 0x7f, 0xfe, 0xff]);
    }

    #[test]
    fn test_array_complex_encoding_round_trips() {
        let c64_components = [0x8000_0000_u32, 0x7fc0_1234, 0x7f80_0000, 0xff80_0000];
        let c64_values = [
            ComplexNumber::new(f32::from_bits(c64_components[0]), f32::from_bits(c64_components[1])),
            ComplexNumber::new(f32::from_bits(c64_components[2]), f32::from_bits(c64_components[3])),
        ];
        let c64 = Array::from_elements(array_type(DataType::C64, &[2]), &c64_values).unwrap();
        let expected_c64_bytes = c64_components.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        assert_eq!(c64.storage_bytes(), expected_c64_bytes);
        let decoded_c64 = c64.elements::<ComplexNumber<f32>>().unwrap();
        assert_eq!(
            decoded_c64.iter().flat_map(|value| [value.re.to_bits(), value.im.to_bits()]).collect::<Vec<_>>(),
            c64_components,
        );

        let c128_components =
            [0x8000_0000_0000_0000_u64, 0x7ff8_0000_0000_1234, 0x7ff0_0000_0000_0000, 0xfff0_0000_0000_0000];
        let c128_values = [
            ComplexNumber::new(f64::from_bits(c128_components[0]), f64::from_bits(c128_components[1])),
            ComplexNumber::new(f64::from_bits(c128_components[2]), f64::from_bits(c128_components[3])),
        ];
        let c128_type = array_type(DataType::C128, &[2]);
        let c128 = Array::from_elements(c128_type.clone(), &c128_values).unwrap();
        let expected_c128_bytes = c128_components.into_iter().flat_map(u64::to_le_bytes).collect::<Vec<_>>();
        assert_eq!(c128.storage_bytes(), expected_c128_bytes);
        assert_eq!(
            Array::new(c128_type.clone(), expected_c128_bytes.clone()).unwrap().logical_bytes(),
            expected_c128_bytes
        );
        assert_eq!(
            Array::from_logical_bytes(c128_type, &expected_c128_bytes).unwrap().storage_bytes(),
            expected_c128_bytes
        );
        let decoded_c128 = c128.elements::<ComplexNumber<f64>>().unwrap();
        assert_eq!(
            decoded_c128.iter().flat_map(|value| [value.re.to_bits(), value.im.to_bits()]).collect::<Vec<_>>(),
            c128_components,
        );
    }

    #[test]
    fn test_array_empty_and_payload_free_encoding_round_trips() {
        let empty_type = array_type(DataType::F32, &[0, 3]);
        let empty = Array::from_elements(empty_type.clone(), &[] as &[f32]).unwrap();
        assert!(empty.storage_bytes().is_empty());
        assert!(empty.logical_bytes().is_empty());
        assert_eq!(empty.elements::<f32>(), Ok(Vec::new()));
        assert_eq!(Array::new(empty_type.clone(), Vec::new()).unwrap().elements::<f32>(), Ok(Vec::new()));
        assert_eq!(Array::from_logical_bytes(empty_type, &[]).unwrap().elements::<f32>(), Ok(Vec::new()));

        for data_type in [DataType::Token, DataType::Zero] {
            let r#type = array_type(data_type, &[3]);
            let array = Array::new(r#type.clone(), Vec::new()).unwrap();
            assert_eq!(array.r#type().as_ref(), &r#type);
            assert_eq!(Array::element_count(&r#type), 3);
            assert!(array.storage_bytes().is_empty());
            assert!(array.logical_bytes().is_empty());
            assert!(Array::from_logical_bytes(r#type, &[]).unwrap().storage_bytes().is_empty());
        }
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
        assert_eq!(array.elements::<f8e4m3fn>().unwrap()[0].to_bits(), 0x3c);
        assert_eq!(array.elements::<f8e4m3fn>().unwrap()[1].to_bits(), 0xbc);
        let array = Array::from_f64s(array_type(DataType::I32, &[2]), vec![1.0, -2.0]);
        assert_eq!(array.elements::<i32>(), Ok(vec![1, -2]));
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
    fn test_array_element_combinators() {
        // `map_elements` applies a typed elementwise function, allowing input and output element types to differ.
        let integers = Array::vector(vec![1i32, -2, 3]);
        let doubled = integers.map_elements::<i32, i32>(integers.r#type().into_owned(), |value| Ok(value * 2)).unwrap();
        assert_eq!(doubled.elements::<i32>(), Ok(vec![2, -4, 6]));
        let negative = integers
            .map_elements::<i32, bool>(array_type(DataType::Boolean, &[3]), |value| Ok(value < 0))
            .unwrap();
        assert_eq!(negative.storage_bytes(), [0, 1, 0]);
        assert!(matches!(
            integers.map_elements::<i64, i64>(array_type(DataType::I64, &[3]), Ok),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot map elements of data type i32 as i64 values",
        ));
        assert!(matches!(
            integers.map_elements::<i32, i32>(array_type(DataType::I32, &[2]), Ok),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot map 3 logical elements onto array type i32[2] with 2 logical elements",
        ));

        // `from_fn_elements` constructs an array from its flat logical row-major element indices.
        let iota = Array::from_fn_elements(array_type(DataType::U16, &[2, 2]), |index| Ok(index as u16)).unwrap();
        assert_eq!(iota.elements::<u16>(), Ok(vec![0, 1, 2, 3]));
        assert!(matches!(
            Array::from_fn_elements(array_type(DataType::U16, &[1]), |_| Ok(0u32)),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot store u32 values in an array of element data type u16",
        ));

        // `fold_elements` accumulates typed elements in logical row-major order.
        assert_eq!(integers.fold_elements(0i64, |sum, value: i32| Ok(sum + i64::from(value))), Ok(2));
        assert!(matches!(
            integers.fold_elements(0i64, |sum, _: i64| Ok(sum)),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot fold elements of data type i32 as i64 values",
        ));

        // `gather_elements` copies whole element encodings through a flat index mapping without decoding them, so it
        // serves reversal, repetition, and selection over any element data type, including sub-byte ones.
        let reversed =
            integers.gather_elements(integers.r#type().into_owned(), |output_index| 2 - output_index).unwrap();
        assert_eq!(reversed.elements::<i32>(), Ok(vec![3, -2, 1]));
        let repeated = integers.gather_elements(array_type(DataType::I32, &[4]), |_| 1).unwrap();
        assert_eq!(repeated.elements::<i32>(), Ok(vec![-2, -2, -2, -2]));
        let narrow =
            Array::from_elements(array_type(DataType::I4, &[2]), &[i4::new(-8).unwrap(), i4::new(7).unwrap()]).unwrap();
        let swapped = narrow.gather_elements(narrow.r#type().into_owned(), |output_index| 1 - output_index).unwrap();
        assert_eq!(swapped.storage_bytes(), [0x07, 0x08]);
        assert!(matches!(
            integers.gather_elements(array_type(DataType::I64, &[3]), |output_index| output_index),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot gather elements of data type i32 into an array of element data type i64",
        ));
        assert!(matches!(
            integers.gather_elements(array_type(DataType::I32, &[3]), |_| 3),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "gather index 3 is out of bounds for 3 elements",
        ));
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

        // Sub-byte payloads render through their typed elements, which have no scalar representation, and the debug
        // rendering shares the same element list.
        let narrow =
            Array::from_elements(array_type(DataType::I4, &[2]), &[i4::new(-8).unwrap(), i4::new(7).unwrap()]).unwrap();
        assert_eq!(narrow.to_string(), "[-8, 7]");
        assert!(format!("{narrow:?}").ends_with("values: [-8, 7] }"));
    }

    #[test]
    fn test_array_sort_gathers_sub_byte_operands() {
        // Non-key operands sort by moving whole element encodings, so sub-byte operands (which have no scalar
        // representation) ride an f32 key without being decoded.
        let key = Array::vector(vec![3.0f32, 1.0, 2.0]);
        let passenger = Array::from_elements(
            array_type(DataType::I4, &[3]),
            &[i4::new(-8).unwrap(), i4::new(0).unwrap(), i4::new(7).unwrap()],
        )
        .unwrap();
        let outputs = Array::sort_with_key_count(&[key, passenger.clone()], 0, SortDirection::Ascending, 1).unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![1.0, 2.0, 3.0]);
        assert_eq!(outputs[1].storage_bytes(), [0x00, 0x07, 0x08]);

        // Sub-byte keys stay rejected with an error rather than a panic while key ranks ride the scalar bridge.
        assert!(matches!(
            Array::sort_with_key_count(&[passenger], 0, SortDirection::Ascending, 1),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "'sort' does not support key data type i4",
        ));
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
        // Equality decodes typed values directly, retaining IEEE NaN and signed-zero semantics without depending on
        // either physical byte equality or the transitional scalar bridge.
        let positive_zero = Array::vector(vec![0.0f32]);
        let negative_zero = Array::vector(vec![-0.0f32]);
        assert_eq!(positive_zero, negative_zero);
        let nan = Array::vector(vec![f32::from_bits(0x7fc0_1234)]);
        assert_ne!(nan, nan.clone());
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
            Array::from_elements(r#type.clone(), &[0.0f32; 4]).map_err(|_| unreachable!())
        );
        assert_eq!(
            context.one(&r#type),
            Array::from_elements(r#type.clone(), &[1.0f32; 4]).map_err(|_| unreachable!())
        );
        assert_eq!(
            context.fill(&r#type, Scalar::F32(2.5)),
            Array::from_elements(r#type.clone(), &[2.5f32; 4]).map_err(|_| unreachable!()),
        );
        // Explicit output types use ordinary element conversion, including narrowing.
        assert_eq!(
            context.fill(&r#type, Scalar::F64(2.5)),
            Array::from_elements(r#type.clone(), &[2.5f32; 4]).map_err(|_| unreachable!()),
        );
        assert_eq!(
            context.fill(&r#type, Scalar::C64(ComplexNumber::new(1.0, 2.0))),
            Array::from_elements(r#type.clone(), &[1.0f32; 4]).map_err(|_| unreachable!()),
        );
        let integer_type = array_type(DataType::I32, &[2]);
        assert_eq!(
            context.fill(&integer_type, Scalar::F64(2.5)),
            Array::from_elements(integer_type, &[2i32; 2]).map_err(|_| unreachable!()),
        );
        let boolean_type = array_type(DataType::Boolean, &[2]);
        assert_eq!(
            context.fill(&boolean_type, Scalar::C64(ComplexNumber::new(0.0, 2.0))),
            Array::from_elements(boolean_type, &[true; 2]).map_err(|_| unreachable!()),
        );
        // Iota materializes coordinates along the requested dimension in the declared element data type.
        assert_eq!(
            context.iota(&array_type(DataType::I32, &[2, 3]), 1).unwrap().elements::<i32>(),
            Ok(vec![0, 1, 2, 0, 1, 2]),
        );
        assert_eq!(context.iota(&array_type(DataType::F64, &[3]), 0).unwrap().to_f64s(), vec![0.0, 1.0, 2.0]);
        assert_eq!(
            context.iota(&array_type(DataType::C64, &[3]), 0).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(0.0, 0.0), ComplexNumber::new(1.0, 0.0), ComplexNumber::new(2.0, 0.0),]),
        );
        // Kernels that materialize a payload from a type reject dynamically sized types.
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(3),
            ]),
        );
        let expected_message = "cannot materialize a value of dynamically sized type f64[dynamic, 3]";
        assert!(matches!(
            context.zero(&dynamic_type),
            Err(ProgramError::Type(TypeError::Invalid { message })) if message == expected_message,
        ));
        assert!(matches!(
            context.one(&dynamic_type),
            Err(ProgramError::Type(TypeError::Invalid { message })) if message == expected_message,
        ));
        assert!(matches!(
            context.fill(&dynamic_type, Scalar::from(42.0)),
            Err(ProgramError::Type(TypeError::Invalid { message })) if message == expected_message,
        ));
    }

    #[test]
    fn test_array_zero_like_and_one_like() {
        let array = Array::vector(vec![1.5f32, -2.5]);
        assert_eq!(array.zero_like().elements::<f32>(), Ok(vec![0.0, 0.0]));
        assert_eq!(array.one_like().elements::<f32>(), Ok(vec![1.0, 1.0]));
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
        let wrapped = Array::vector(vec![255u8]).add(&Array::vector(vec![1u8])).unwrap();
        assert_eq!(wrapped.elements::<u8>(), Ok(vec![0]));
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
            complex.elements::<ComplexNumber<f64>>(),
            Ok(vec![ComplexNumber::new(1.0, 3.0), ComplexNumber::new(2.0, -4.0)]),
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
        // General NumPy-style broadcasting maps each input coordinate into the common output shape.
        assert_eq!(
            Array::matrix(2, 1, vec![true, false]).and(&Array::matrix(1, 3, vec![true, false, true])).unwrap(),
            Array::matrix(2, 3, vec![true, false, true, false, false, false]),
        );
        // Same-data-type integers combine bitwise directly over all bytes of each encoding.
        let bits = Array::vector(vec![0b1100u8]).and(&Array::vector(vec![0b1010u8])).unwrap();
        assert_eq!(bits.elements::<u8>(), Ok(vec![0b1000]));
        assert_eq!(Array::vector(vec![0x00ff_i16, -1]).not().unwrap().elements::<i16>(), Ok(vec![-256, 0]));
        // Sub-byte negation complements only the declared low bits, retaining a valid sign-extended encoding.
        let signed_sub_byte = Array::vector(vec![i2::MIN, i2::new(-1).unwrap(), i2::new(0).unwrap(), i2::MAX]);
        assert_eq!(
            signed_sub_byte.not().unwrap().elements::<i2>(),
            Ok(vec![i2::MAX, i2::new(0).unwrap(), i2::new(-1).unwrap(), i2::MIN]),
        );
        assert_eq!(
            Array::scalar(u4::new(0b1100).unwrap())
                .xor(&Array::scalar(u4::new(0b1010).unwrap()))
                .unwrap()
                .elements::<u4>(),
            Ok(vec![u4::new(0b0110).unwrap()]),
        );
        // Physical layouts are traversed through addressing, so holes stay zero rather than being complemented.
        let strided_type =
            array_type(DataType::Boolean, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![2])));
        let strided = Array::new(strided_type.clone(), vec![1, 0, 0]).unwrap().not().unwrap();
        assert_eq!(strided.r#type().as_ref(), &strided_type);
        assert_eq!(strided.storage_bytes(), [0, 0, 1]);
        assert_eq!(strided.elements::<bool>(), Ok(vec![false, true]));
        // Real floating-point operands are rejected, matching the scalar reference backend.
        assert!(matches!(
            Array::vector(vec![1.0]).and(&Array::vector(vec![0.0])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot apply `and` to arrays of element data types f64 and f64",
        ));
        // The `std::ops` sugar delegates to the fallible capabilities.
        assert_eq!(left.clone() & right.clone(), Array::vector(vec![true, false, false, false]));
        assert_eq!(!left.clone(), Array::vector(vec![false, false, true, true]));
    }

    #[test]
    fn test_array_random_bit_generation() {
        // State and result layouts remain physical storage contracts while the generator consumes and produces values
        // in logical order.
        let state_type =
            RandomAlgorithm::ThreeFry.state_type().with_layout(Layout::Strided(StridedLayout::new(vec![-8])));
        let state = Array::from_elements(state_type.clone(), &[42u64, 7]).unwrap();
        let output_type = array_type(DataType::U16, &[5]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let (advanced_state, bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type).unwrap();
        let (expected_words, expected_counter) = threefry_u32_words(42, 7, 5);
        let expected_words = expected_words.into_iter().map(|word| word as u16).collect::<Vec<_>>();
        assert_eq!(advanced_state.r#type().as_ref(), &state_type);
        assert_eq!(advanced_state.elements::<u64>(), Ok(vec![42, expected_counter]));
        assert_eq!(bits.r#type().as_ref(), &output_type);
        assert_eq!(bits.elements::<u16>(), Ok(expected_words.clone()));
        let mut expected_storage = vec![0; 18];
        for (index, word) in expected_words.into_iter().enumerate() {
            expected_storage[index * 4..index * 4 + 2].copy_from_slice(&word.to_le_bytes());
        }
        assert_eq!(bits.storage_bytes(), expected_storage);

        // Eight-bit outputs retain the low byte of each generated `u32` word.
        let output_type = array_type(DataType::U8, &[5]);
        let (_, bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type).unwrap();
        let (expected_words, _) = threefry_u32_words(42, 7, 5);
        assert_eq!(bits.elements::<u8>(), Ok(expected_words.into_iter().map(|word| word as u8).collect()));

        // Direct backend calls enforce the same state contract as operation type inference.
        let invalid_state = Array::vector(vec![42u64, 7, 9]);
        assert!(matches!(
            invalid_state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "'rng_bit_generator' with the three_fry algorithm needs a u64[2] state but got u64[3]",
        ));
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

        // Sealed sub-byte elements use their signed value ordering and participate in full NumPy-style broadcasting.
        let left = Array::matrix(2, 1, vec![i2::new(-1).unwrap(), i2::new(1).unwrap()]);
        let right = Array::matrix(1, 3, vec![i2::new(-2).unwrap(), i2::new(0).unwrap(), i2::new(1).unwrap()]);
        assert_eq!(
            left.compare(&right, ComparisonDirection::LessThan).unwrap().elements::<bool>(),
            Ok(vec![false, true, true, false, false, false]),
        );

        // Addressed input and output layouts remain physical contracts; comparison writes only the Boolean element
        // ranges and leaves output holes zero.
        let strided_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let left = Array::from_elements(strided_type.clone(), &[1u16, 3]).unwrap();
        let right = Array::from_elements(strided_type, &[2u16, 2]).unwrap();
        let compared = left.compare(&right, ComparisonDirection::LessThan).unwrap();
        assert_eq!(compared.elements::<bool>(), Ok(vec![true, false]));
        assert_eq!(compared.storage_bytes(), [1, 0, 0, 0, 0]);

        // Floating-point NaNs are unordered, while complex arrays expose only equality comparisons.
        let nan = Array::vector(vec![f8e5m2::NAN]);
        assert_eq!(nan.compare(&nan, ComparisonDirection::NotEqual).unwrap().elements::<bool>(), Ok(vec![true]));
        let complex = Array::vector(vec![ComplexNumber::new(1.0f32, 2.0)]);
        assert_eq!(complex.compare(&complex, ComparisonDirection::Equal).unwrap().elements::<bool>(), Ok(vec![true]),);
        assert!(matches!(
            complex.compare(&complex, ComparisonDirection::LessThan),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot apply an ordered comparison to unordered complex scalars of data type c64",
        ));

        // Empty payload-free comparisons are vacuous because they evaluate no unsupported element comparison; a
        // nonempty token array retains the established scalar-backend error.
        let empty_token = Array::from_logical_bytes(array_type(DataType::Token, &[0]), &[]).unwrap();
        assert_eq!(
            empty_token.compare(&empty_token, ComparisonDirection::Equal).unwrap().elements::<bool>(),
            Ok(vec![]),
        );
        let token = Array::from_logical_bytes(array_type(DataType::Token, &[1]), &[]).unwrap();
        assert!(matches!(
            token.compare(&token, ComparisonDirection::Equal),
            Err(ProgramError::Type(TypeError::Invalid { message })) if message == "cannot compare token scalars",
        ));
    }

    #[test]
    fn test_array_convert_element_type() {
        let vector = Array::vector(vec![0.0, 1.5]);
        assert_eq!(vector.convert_element_type(DataType::Boolean).unwrap(), Array::vector(vec![false, true]));
        assert_eq!(vector.convert_element_type(DataType::I32).unwrap(), Array::vector(vec![0i32, 1]));
        // Conversions into low-precision floating-point element types produce exact encodings.
        let low_precision = vector.convert_element_type(DataType::F8E5M2).unwrap();
        assert_eq!(low_precision.elements::<f8e5m2>().unwrap()[1].to_bits(), 0x3e);
        // Conversions to or from the token data type are rejected.
        assert!(matches!(
            vector.convert_element_type(DataType::Token),
            Err(ProgramError::Type(TypeError::Invalid { message }))
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

        // General broadcasting reads every input through its physical layout and writes one dense output without
        // converting equal-typed branch elements through an intermediate representation.
        let condition_type =
            array_type(DataType::Boolean, &[2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-3, 1])));
        let condition = Array::from_elements(condition_type, &[true, false]).unwrap();
        let true_type =
            array_type(DataType::U16, &[1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![8, -2])));
        let on_true = Array::from_elements(true_type, &[0x1111u16, 0x2222, 0x3333]).unwrap();
        let false_type =
            array_type(DataType::U16, &[2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-4, 2])));
        let on_false = Array::from_elements(false_type, &[0xaaaau16, 0xbbbb]).unwrap();
        let selected = Array::select(&condition, &on_true, &on_false).unwrap();
        assert_eq!(selected.r#type().as_ref(), &array_type(DataType::U16, &[2, 3]));
        assert_eq!(selected.elements::<u16>(), Ok(vec![0x1111, 0x2222, 0x3333, 0xbbbb, 0xbbbb, 0xbbbb]),);
        assert_eq!(selected.storage_bytes(), [0x11, 0x11, 0x22, 0x22, 0x33, 0x33, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb],);
    }

    #[test]
    fn test_array_boolean_concretization() {
        let vector = Array::vector(vec![0.0, 2.5]);
        let boolean = vector.compare(&Array::vector(vec![0.0, 0.0]), ComparisonDirection::NotEqual).unwrap();
        assert_eq!(boolean.r#type().into_owned(), array_type(DataType::Boolean, &[2]));
        assert_eq!(boolean, Array::vector(vec![false, true]));
        assert_eq!(Array::scalar(true).concretize(), Ok(true));
        assert!(Array::vector(vec![true, false]).concretize().is_err());
        assert!(Array::scalar(1.0).concretize().is_err());
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

        // Predicate and branch layouts are independent of logical masking. The output preserves the congruent branch
        // layout, including its hole, while selecting exact element bytes in logical order.
        let predicate_type =
            array_type(DataType::Boolean, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
        let predicate = Array::from_elements(predicate_type, &[false, true]).unwrap();
        assert_eq!(predicate.any_true(), Ok(true));
        let branch_type =
            array_type(DataType::U16, &[2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![-6, 2])));
        let on_true = Array::from_elements(branch_type.clone(), &[0x1111u16, 0x2222, 0x3333, 0x4444]).unwrap();
        let on_false = Array::from_elements(branch_type.clone(), &[0xaaaau16, 0xbbbb, 0xcccc, 0xdddd]).unwrap();
        let selected = predicate.mask_select(&on_true, &on_false).unwrap();
        assert_eq!(selected.r#type().as_ref(), &branch_type);
        assert_eq!(selected.elements::<u16>(), Ok(vec![0xaaaa, 0xbbbb, 0x3333, 0x4444]));
        assert_eq!(selected.storage_bytes(), [0x33, 0x33, 0x44, 0x44, 0, 0, 0xaa, 0xaa, 0xbb, 0xbb]);
    }

    #[test]
    fn test_array_broadcast() {
        let vector = Array::vector(vec![1.0, 2.0]);
        let output_type = array_type(DataType::F64, &[3, 2]);
        let broadcast = LegacyBroadcast::legacy_broadcast(&vector, output_type.clone(), &[1]).unwrap();
        assert_eq!(broadcast.r#type().into_owned(), output_type);
        assert_eq!(broadcast.to_f64s(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        // Broadcasting reads a reversed input layout and writes the output's requested physical layout, retaining
        // zero in its holes.
        let input_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let input = Array::from_elements(input_type, &[0x1122u16, 0x3344]).unwrap();
        let output_type =
            array_type(DataType::U16, &[2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![6, 2])));
        let broadcast = input.broadcast_to_type(output_type.clone(), &[1]).unwrap();
        assert_eq!(broadcast.r#type().as_ref(), &output_type);
        assert_eq!(broadcast.elements::<u16>(), Ok(vec![0x1122, 0x3344, 0x1122, 0x3344]));
        assert_eq!(broadcast.storage_bytes(), [0x22, 0x11, 0x44, 0x33, 0, 0, 0x22, 0x11, 0x44, 0x33]);

        // Deliberately malformed concrete values with dynamic types fail through the structured materialization
        // diagnostic before either the identity fast path or static-shape payload logic can accept or panic on them.
        let dynamic = DimensionVariable::new("dynamic", DimensionBounds::unbounded());
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(dynamic.clone()), Dimension::Dynamic(dynamic)]),
        );
        let dynamic = Array::with_unchecked_type(
            dynamic_type.clone(),
            [1.0f64, 2.0, 3.0, 4.0].into_iter().flat_map(f64::to_le_bytes).collect(),
        );
        for output_axes in [vec![0, 1], vec![1, 0]] {
            assert!(matches!(
                dynamic.broadcast_to_type(dynamic_type.clone(), output_axes.as_slice()),
                Err(ProgramError::Type(TypeError::Invalid { message }))
                    if message == "cannot materialize a value of dynamically sized type f64[dynamic, dynamic]",
            ));
        }
    }

    #[test]
    fn test_array_transpose() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = matrix.transpose([1, 0]).unwrap();
        assert_eq!(transposed.r#type().into_owned(), array_type(DataType::F64, &[3, 2]));
        assert_eq!(transposed.to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert!(matrix.transpose([0, 0]).is_err());

        // Transposition traverses the input's physical layout while producing the canonical layout-free output type.
        let input_type =
            array_type(DataType::U16, &[2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![8, 2])));
        let matrix = Array::from_elements(input_type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        let transposed = matrix.transpose([1, 0]).unwrap();
        assert_eq!(transposed.r#type().into_owned(), array_type(DataType::U16, &[3, 2]));
        assert_eq!(transposed.elements::<u16>(), Ok(vec![1, 4, 2, 5, 3, 6]));
        assert_eq!(transposed.storage_bytes(), [1, 0, 4, 0, 2, 0, 5, 0, 3, 0, 6, 0]);

        // Empty arrays transpose without calculating strides that may overflow for otherwise irrelevant dimensions.
        let empty = Array::new(array_type(DataType::F64, &[0, usize::MAX, usize::MAX]), Vec::new()).unwrap();
        assert_eq!(
            empty.transpose([1, 2, 0]).unwrap(),
            Array::new(array_type(DataType::F64, &[usize::MAX, usize::MAX, 0]), Vec::new()).unwrap(),
        );
    }

    #[test]
    fn test_array_reshape() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = matrix.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])).unwrap();
        assert_eq!(reshaped.r#type().into_owned(), array_type(DataType::F64, &[3, 2]));
        assert_eq!(reshaped.to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(matrix.reshape(Shape::new(vec![Dimension::Static(4)])).is_err());

        // Reshaping preserves logical order independently of the input's physical placement.
        let input_type =
            array_type(DataType::U16, &[2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![8, 2])));
        let matrix = Array::from_elements(input_type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        let reshaped = matrix.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])).unwrap();
        assert_eq!(reshaped.elements::<u16>(), Ok(vec![1, 2, 3, 4, 5, 6]));
        assert_eq!(reshaped.storage_bytes(), [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]);
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
        // Index decoding is typed and supports sub-byte integers directly; a negative start still clamps to zero.
        let start = [Array::scalar(crate::arrays::i4::new(-1).unwrap())];
        assert_eq!(vector.dynamic_slice(&start, &[2]).unwrap(), Array::vector(vec![1.0, 2.0]));

        // Static slicing and updating traverse arbitrary source and update layouts while preserving the destination
        // layout for updates.
        let input_type = array_type(DataType::U16, &[5]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let vector = Array::from_elements(input_type.clone(), &[1u16, 2, 3, 4, 5]).unwrap();
        assert_eq!(vector.slice(&[1], &[5], &[2]).unwrap().elements::<u16>(), Ok(vec![2, 4]));
        let update_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let update = Array::from_elements(update_type, &[10u16, 20]).unwrap();
        let updated = vector.update_slice(&update, &[1]).unwrap();
        assert_eq!(updated.r#type().as_ref(), &input_type);
        assert_eq!(updated.elements::<u16>(), Ok(vec![1, 10, 20, 4, 5]));
        assert_eq!(updated.storage_bytes(), [5, 0, 4, 0, 20, 0, 10, 0, 1, 0]);
    }

    #[test]
    fn test_array_pad() {
        let vector = Array::vector(vec![1.0, 2.0]);
        let padded = vector.pad(&Array::scalar(0.5), &[1], &[2], &[1]).unwrap();
        assert_eq!(padded, Array::vector(vec![0.5, 1.0, 0.5, 2.0, 0.5, 0.5]));

        // Padding copies both the reversed input layout and the rank-zero padding element by their exact bytes.
        let input_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let vector = Array::from_elements(input_type, &[1u16, 2]).unwrap();
        let padded = vector.pad(&Array::scalar(9u16), &[1], &[1], &[1]).unwrap();
        assert_eq!(padded.r#type().into_owned(), array_type(DataType::U16, &[5]));
        assert_eq!(padded.elements::<u16>(), Ok(vec![9, 1, 9, 2, 9]));
        assert_eq!(padded.storage_bytes(), [9, 0, 1, 0, 9, 0, 2, 0, 9, 0]);
    }

    #[test]
    fn test_array_concatenate() {
        // Three operands joined along axis 0 preserve their order.
        let concatenated = Array::concatenate(
            [&Array::vector(vec![1.0]), &Array::vector(vec![2.0, 3.0]), &Array::vector(vec![4.0])],
            0,
        )
        .unwrap();
        assert_eq!(concatenated, Array::vector(vec![1.0, 2.0, 3.0, 4.0]));

        // A rank-3 middle-axis concatenation exercises the row-major block odometer.
        let first = Array::from_f64s(array_type(DataType::F64, &[2, 1, 2]), vec![1.0, 2.0, 3.0, 4.0]);
        let second =
            Array::from_f64s(array_type(DataType::F64, &[2, 2, 2]), vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let concatenated = Array::concatenate([&first, &second], 1).unwrap();
        assert_eq!(concatenated.r#type().into_owned(), array_type(DataType::F64, &[2, 3, 2]));
        assert_eq!(concatenated.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0],);

        // Concatenation traverses each input's physical layout and emits the canonical layout-free result.
        let first_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let second_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let first = Array::from_elements(first_type, &[1u16, 2]).unwrap();
        let second = Array::from_elements(second_type, &[3u16, 4]).unwrap();
        let concatenated = Array::concatenate([&first, &second], 0).unwrap();
        assert_eq!(concatenated.r#type().into_owned(), array_type(DataType::U16, &[4]));
        assert_eq!(concatenated.elements::<u16>(), Ok(vec![1, 2, 3, 4]));
        assert_eq!(concatenated.storage_bytes(), [1, 0, 2, 0, 3, 0, 4, 0]);

        // Concatenation does not require an artificial additive zero, including when the output itself is empty.
        let element_type = array_type(DataType::F8E8M0FNU, &[1]);
        let first = Array::new(element_type.clone(), vec![1]).unwrap();
        let second = Array::new(element_type, vec![2]).unwrap();
        assert_eq!(
            Array::concatenate([&first, &second], 0),
            Array::new(array_type(DataType::F8E8M0FNU, &[2]), vec![1, 2]),
        );
        let empty_type = array_type(DataType::F8E8M0FNU, &[0]);
        let empty = Array::new(empty_type.clone(), Vec::new()).unwrap();
        assert_eq!(Array::concatenate([&empty, &empty], 0), Array::new(empty_type, Vec::new()));
    }

    #[test]
    fn test_array_gather() {
        // Gather rows 2 and 0 of a 3x2 matrix.
        let operand = Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indices = Array::matrix(2, 1, vec![2i64, 0]);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let gathered = operand.gather(&indices, &operation).unwrap();
        assert_eq!(gathered.r#type().into_owned(), array_type(DataType::F64, &[2, 2]));
        assert_eq!(gathered.to_f64s(), vec![5.0, 6.0, 1.0, 2.0]);

        // In-bounds and clipping modes do not materialize an unused zero fill, so they work for formats that cannot
        // represent zero.
        let operand = Array::new(array_type(DataType::F8E8M0FNU, &[2]), vec![0x7f, 0x80]).unwrap();
        let indices = Array::matrix(1, 1, vec![1i64]);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1]);
        assert_eq!(operand.gather(&indices, &operation), Array::new(array_type(DataType::F8E8M0FNU, &[1]), vec![0x80]));

        // Gather reads both a reversed operand and reversed sub-byte indices through their physical addressing. An
        // out-of-bounds query in fill-or-drop mode writes the element type's zero encoding into the dense result.
        let operand_type = array_type(DataType::U16, &[3]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let operand = Array::from_elements(operand_type, &[10u16, 20, 30]).unwrap();
        let indices_type =
            array_type(DataType::I4, &[3, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-1, 1])));
        let indices = Array::from_elements(
            indices_type,
            &[
                crate::arrays::i4::new(2).unwrap(),
                crate::arrays::i4::new(-1).unwrap(),
                crate::arrays::i4::new(1).unwrap(),
            ],
        )
        .unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1])
            .with_mode(GatherScatterMode::FillOrDrop);
        let gathered = operand.gather(&indices, &operation).unwrap();
        assert_eq!(gathered.elements::<u16>(), Ok(vec![30, 0, 20]));
        assert_eq!(gathered.storage_bytes(), [30, 0, 0, 0, 20, 0]);
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

        // Scatter decodes sub-byte indices through their physical layout without materializing a scalar index vector.
        let indices_type =
            array_type(DataType::I4, &[2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-1, 1])));
        let indices = Array::from_elements(
            indices_type,
            &[crate::arrays::i4::new(3).unwrap(), crate::arrays::i4::new(0).unwrap()],
        )
        .unwrap();
        assert_eq!(operand.scatter(&indices, &updates, &operation).unwrap(), Array::vector(vec![21.0, 2.0, 3.0, 14.0]),);
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
        assert_eq!(integers.reduce(&[0], ReductionKind::Max).elements::<i32>(), Ok(vec![3]));
        assert_eq!(integers.reduce(&[0], ReductionKind::Min).elements::<i32>(), Ok(vec![-1]));
        // Boolean reductions.
        let booleans = Array::vector(vec![true, false, true]);
        assert_eq!(booleans.reduce(&[0], ReductionKind::Any).elements::<bool>(), Ok(vec![true]));
        assert_eq!(booleans.reduce(&[0], ReductionKind::All).elements::<bool>(), Ok(vec![false]));
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
        let unsigned = Array::vector(vec![0u8, 1, 255]);
        assert_eq!(unsigned.neg().unwrap().elements::<u8>(), Ok(vec![0, 255, 1]));
        let minimum = Array::vector(vec![i8::MIN, -5]);
        assert_eq!(minimum.neg().unwrap().elements::<i8>(), Ok(vec![i8::MIN, 5]));
        // Integer division by zero is a clean error rather than a panic.
        assert!(Array::vector(vec![1i32]).div(&Array::vector(vec![0i32])).is_err());
    }

    #[test]
    fn test_array_encoding_fidelity() {
        // Conversions and arithmetic on low-precision floating-point arrays operate on genuine encodings: the payload
        // round-trips through the exact bit patterns rather than an `f64` pun.
        let array = Array::from_f64s(array_type(DataType::F8E8M0FNU, &[2]), vec![2.0, 0.5]);
        assert_eq!(array.elements::<f8e8m0fnu>().unwrap()[0].to_bits(), 0x80);
        assert_eq!(array.elements::<f8e8m0fnu>().unwrap()[1].to_bits(), 0x7e);
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
        assert_eq!(transferred.storage_bytes(), array.storage_bytes());

        // Resharding records the requested distribution metadata on the type, carrying the input's varying manual
        // axes over to the target sharding exactly like the `ReshardOperation` type-inference rule.
        let input_sharding = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap();
        let input =
            Array::from_f64s(array_type(DataType::F64, &[2]).with_sharding(input_sharding).unwrap(), vec![1.0, 2.0]);
        let target = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = input.reshard(&target);
        assert_eq!(resharded.r#type().sharding(), Some(&target.clone().with_varying_manual_axes(["m"]).unwrap()),);
        assert_eq!(resharded.storage_bytes(), input.storage_bytes());

        // The sharding-constraint hint is untracked, so constraining leaves the value (type included) unchanged.
        assert_eq!(input.constrain_sharding(&target), input);
    }
}
