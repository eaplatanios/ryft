use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use approx::AbsDiffEq;
use half::{bf16, f16};
use num_complex::Complex;

use ryft_macros::Parameter;

// TODO(eaplatanios): Review from here onwards.

use crate::arrays::addressing::ArrayAddressing;
use crate::arrays::broadcasting::Broadcastable;
use crate::arrays::encoding::{
    ArrayElement, decode_elements, decode_logical_bytes, encode_elements, encode_logical_bytes, f4e2m1fn, f6e2m3fn,
    f6e3m2fn, f8e3m4, f8e4m3, f8e4m3b11fnuz, f8e4m3fn, f8e4m3fnuz, f8e5m2, f8e5m2fnuz, f8e8m0fnu, i1, i2, i4, u1, u2,
    u4, validate_storage_bytes,
};
use crate::arrays::macros::dispatch_on_array_element_type;
use crate::arrays::operations::ArrayOperation;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::{Dimension, Shape, StaticShape};
use crate::contexts::EagerContext;
use crate::parameters::Parameter;
use crate::programs::{Concretizable, ProgramError, TypeError, Typed, Value};

/// Dense multidimensional [`Value`] whose [`Type`](crate::programs::Type) is an [`ArrayType`]. It is the reference
/// array value of Ryft: it exists primarily to exercise the tracing, transformation, and interpretation machinery
/// with programs over multidimensional arrays without depending on an optimized backend such as `ryft-xla`. Unit
/// tests, documentation tests, and downstream crates can therefore interpret complete array programs eagerly and
/// stage them through [`ArrayTracingContext`](crate::arrays::ArrayTracingContext).
///
/// The payload is one shared immutable byte buffer whose physical placement is determined by the array's
/// [`ArrayType`]. Missing layout metadata means dense row-major storage; explicit strided and tiled layouts determine
/// the physical ordering, holes, and padding. [`Array::new`] validates the complete physical representation, while
/// [`Array::from_elements`] and [`Array::from_logical_bytes`] construct it from logical row-major values through
/// checked typed codecs that preserve exact element encodings. Cloning an array shares its payload without copying
/// it. The per-family kernels in [`crate::arrays::operations`] own the element-level arithmetic, conversion, and
/// shape semantics computed over these values.
///
/// A production [`Array`] always carries a fully static [`ArrayType`]: every constructor that sizes or addresses a
/// payload funnels through [`ArrayAddressing::new`], which rejects any type with a [`Dimension::Dynamic`] axis, so a
/// dynamically shaped array value cannot be built. Reference kernels may therefore assume static geometry and read
/// extents directly off the stored type instead of resolving first-class dimension extents; programs that genuinely
/// need dynamic shapes stage over [`ArrayIrOperation`](crate::ArrayIrOperation) instead, where
/// each dynamic axis is carried by an explicit dimension operand. The single bypass is the `#[cfg(test)]`-gated
/// `Array::with_unchecked_type`, which exists so that transform-validation tests can pin how the dynamic-shape
/// rejections behave when a deliberately malformed value reaches them.
///
/// # Warning
///
/// This backend prioritizes transparency over performance. It supports the physical strided and tiled layouts carried
/// by [`ArrayType`], but operations materialize owned outputs rather than views and use straightforward reference
/// implementations rather than vectorized kernels. Do not use it outside tests, documentation examples, and
/// reference-semantics checks.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::arrays::Array;
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

    /// Creates an array from its staged array type and shared physical storage without revalidating either. The
    /// caller guarantees what [`Array::new`] would otherwise check: `bytes` has the exact layout-derived byte count
    /// for `type`, holds a valid encoding for every logical element, and is zero in every layout hole and
    /// tile-padding byte. This exists because the reference kernels in [`crate::arrays::operations`] build their
    /// results by writing into an addressed buffer they sized from the output type, so they already uphold the
    /// storage invariants by construction and would otherwise pay for a second full traversal per operation. Taking
    /// the payload as an [`Arc`] also lets kernels that only retype a value (such as a memory transfer or a reshard)
    /// share the original payload instead of copying it.
    ///
    /// # Parameters
    ///
    ///   - `type`: Staged array type of the array.
    ///   - `bytes`: Complete physical storage, including any layout holes or tile padding.
    pub(crate) fn new_unchecked(r#type: ArrayType, bytes: Arc<Vec<u8>>) -> Self {
        Self { r#type, bytes }
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
        if data_type.is_token() || data_type.is_zero() {
            panic!("cannot convert f64 values to the {data_type} data type");
        }
        dispatch_on_array_element_type!(data_type, |Element| {
            let elements = values
                .into_iter()
                .map(Element::from_real)
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|error| panic!("{error}"));
            Self::from_elements(r#type, &elements).unwrap_or_else(|error| panic!("{error}"))
        })
    }

    /// Returns the complete immutable physical storage, including layout holes and tile padding.
    pub fn storage_bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    /// Returns the shared handle to this array's physical storage, so that a kernel which only retypes a value can
    /// hand the same payload to [`Array::new_unchecked`] instead of copying it.
    pub(crate) fn shared_storage(&self) -> &Arc<Vec<u8>> {
        &self.bytes
    }

    /// Returns the complete physical storage for in-place mutation, copying the payload first when it is shared with
    /// another array. Kernels that build a result by mutating a buffer they own (or one they just cloned from an
    /// operand) use this to avoid a second allocation.
    pub(crate) fn storage_bytes_mut(&mut self) -> &mut [u8] {
        Arc::make_mut(&mut self.bytes).as_mut_slice()
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
        if self.r#type.data_type() != Input::data_type() {
            return Err(TypeError::invalid(format!(
                "cannot map elements of data type {} as {} values",
                self.r#type.data_type(),
                Input::data_type(),
            ))
            .into());
        }
        if output_type.data_type() != Output::data_type() {
            return Err(TypeError::invalid(format!(
                "cannot store mapped {} values in an array of element data type {}",
                Output::data_type(),
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

    /// Creates a new array holding this array's elements converted into `data_type`, preserving the shape, the
    /// physical layout, and every other component of the array's type. This is the foundational cast of the reference
    /// backend: the [`ConvertElementType`](crate::operations::ConvertElementType) capability delegates to it, and so
    /// does every kernel that promotes mixed-type operands through [`Array::promoted_to`].
    ///
    /// Conversion of an individual element is exactly [`ArrayElement::convert_to`], so the per-element semantics
    /// (which category carries each source, which destination performs the single rounding, truncation, or
    /// saturation step, and which formats reject a value outright) are the ones documented on that trait. Converting
    /// an array to its own element data type shares the existing payload instead of copying it, and the token and
    /// structural-zero data types have no elements to convert, so only that same-type no-op is accepted for them.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Element [`DataType`] of the result.
    ///
    /// # Errors
    ///
    /// Returns an error if either data type is [`DataType::Token`], if exactly one of them is [`DataType::Zero`], or
    /// if any element has no representation in `data_type` (for example, converting a zero into `f8e8m0fnu`, which
    /// cannot represent it).
    pub fn converted_to(&self, data_type: DataType) -> Result<Self, ProgramError> {
        let source_data_type = self.r#type.data_type();
        if source_data_type.is_token() || data_type.is_token() {
            return Err(TypeError::invalid("cannot convert values to or from the token data type".to_string()).into());
        }
        if source_data_type == data_type {
            return Ok(self.clone());
        }
        if source_data_type.is_zero() || data_type.is_zero() {
            return Err(TypeError::invalid("cannot convert values to or from the zero data type".to_string()).into());
        }
        let output_type = self.r#type.clone().with_data_type(data_type);
        // The nested dispatch selects the concrete source and destination element types, which monomorphizes
        // `convert_to` into the pair's direct conversion (refer to the documentation of
        // [`ArrayElement::convert_to`]). Should a measured hot pair ever justify a bespoke kernel, it can be matched
        // here ahead of the generic path without changing the element interchange contract.
        dispatch_on_array_element_type!(source_data_type, |Input| {
            dispatch_on_array_element_type!(data_type, |Output| {
                self.map_elements::<Input, Output>(output_type, Input::convert_to::<Output>)
            })
        })
    }

    /// Converts this array to the provided element data type, borrowing it unchanged when it already has that data
    /// type so that already-promoted operands keep their exact physical storage and layout. Kernels that promote
    /// mixed-type operands to a common element data type (which each kernel computes from its own type-inference
    /// contract) use this to convert only the mismatched operands.
    pub fn promoted_to(&self, data_type: DataType) -> Result<Cow<'_, Self>, ProgramError> {
        if self.r#type.data_type() == data_type {
            Ok(Cow::Borrowed(self))
        } else {
            Ok(Cow::Owned(self.converted_to(data_type)?))
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
        if r#type.data_type() != T::data_type() {
            return Err(TypeError::invalid(format!(
                "cannot store {} values in an array of element data type {}",
                T::data_type(),
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
        let addressing = ArrayAddressing::new(self.r#type.clone()).unwrap();
        (0..addressing.element_count())
            .map(|index| {
                Self::element_as_f64(data_type, &self.bytes[addressing.byte_range_for_flat_index(index)])
                    .unwrap_or_else(|| panic!("cannot view an array of element data type {data_type} as f64 values"))
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

    /// Applies a typed binary element function with NumPy-style broadcasting directly over addressed storage. Inputs
    /// and outputs use their sealed codecs one element at a time, so the only payload allocation is the result buffer.
    pub(crate) fn binary_elements<Input: ArrayElement, Output: ArrayElement>(
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

    /// Maps one flat row-major output index to the corresponding flat input index under NumPy-style broadcasting.
    /// Input axes are right-aligned with output axes, and an input extent of one always selects coordinate zero.
    pub(crate) fn broadcast_index(
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

    /// Decodes one real-valued element as `f64`, returning `None` for complex and payload-free element data types.
    /// Integer conversions use Rust's ordinary `as f64` semantics.
    pub(crate) fn element_as_f64(data_type: DataType, bytes: &[u8]) -> Option<f64> {
        Some(match data_type {
            DataType::Boolean => f64::from(u8::from(bool::decode(bytes))),
            DataType::I1 => f64::from(i1::decode(bytes).value()),
            DataType::I2 => f64::from(i2::decode(bytes).value()),
            DataType::I4 => f64::from(i4::decode(bytes).value()),
            DataType::I8 => f64::from(i8::decode(bytes)),
            DataType::I16 => f64::from(i16::decode(bytes)),
            DataType::I32 => f64::from(i32::decode(bytes)),
            DataType::I64 => i64::decode(bytes) as f64,
            DataType::U1 => f64::from(u1::decode(bytes).value()),
            DataType::U2 => f64::from(u2::decode(bytes).value()),
            DataType::U4 => f64::from(u4::decode(bytes).value()),
            DataType::U8 => f64::from(u8::decode(bytes)),
            DataType::U16 => f64::from(u16::decode(bytes)),
            DataType::U32 => f64::from(u32::decode(bytes)),
            DataType::U64 => u64::decode(bytes) as f64,
            DataType::F4E2M1FN => f4e2m1fn::decode(bytes).to_f64(),
            DataType::F6E2M3FN => f6e2m3fn::decode(bytes).to_f64(),
            DataType::F6E3M2FN => f6e3m2fn::decode(bytes).to_f64(),
            DataType::F8E3M4 => f8e3m4::decode(bytes).to_f64(),
            DataType::F8E4M3 => f8e4m3::decode(bytes).to_f64(),
            DataType::F8E4M3FN => f8e4m3fn::decode(bytes).to_f64(),
            DataType::F8E4M3FNUZ => f8e4m3fnuz::decode(bytes).to_f64(),
            DataType::F8E4M3B11FNUZ => f8e4m3b11fnuz::decode(bytes).to_f64(),
            DataType::F8E5M2 => f8e5m2::decode(bytes).to_f64(),
            DataType::F8E5M2FNUZ => f8e5m2fnuz::decode(bytes).to_f64(),
            DataType::F8E8M0FNU => f8e8m0fnu::decode(bytes).to_f64(),
            DataType::BF16 => bf16::decode(bytes).to_f64(),
            DataType::F16 => f16::decode(bytes).to_f64(),
            DataType::F32 => f64::from(f32::decode(bytes)),
            DataType::F64 => f64::decode(bytes),
            DataType::C64 | DataType::C128 | DataType::Token | DataType::Zero => return None,
        })
    }
}

#[cfg(test)]
impl Array {
    /// Creates an array without enforcing the storage invariants, so that `ryft-core`'s own transform-validation tests
    /// can materialize values whose declared types are deliberately not materializable (e.g., dynamically shaped
    /// types) and exercise the type-level rejection paths. Never use this outside of such validation tests.
    pub(crate) fn with_unchecked_type(r#type: ArrayType, bytes: Vec<u8>) -> Self {
        Self::new_unchecked(r#type, Arc::new(bytes))
    }
}

/// Creates a static [`ArrayType`] with the provided element data type and dimension sizes. This is the shared
/// type-construction fixture for the reference backend's own unit tests and for the per-family kernel tests in
/// [`crate::arrays::operations`].
#[cfg(test)]
pub(crate) fn array_type(data_type: DataType, dimensions: &[usize]) -> ArrayType {
    ArrayType::new(data_type, Shape::new(dimensions.iter().map(|size| Dimension::Static(*size)).collect()))
}

impl Debug for Array {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The payload renders through `Display`, which supports every element data type, including sub-byte types.
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

// Arrays render in logical shape order: a scalar renders as one element, and every array dimension contributes one
// bracketed nesting level. Real floating-point payloads use debug formatting so integral values retain a decimal point
// (e.g., `1.0` rather than `1`), keeping the element type visually apparent in diagnostics.
impl Display for Array {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Renders elements in logical row-major order, adding one bracketed level per static array dimension.
        fn write_elements(
            formatter: &mut std::fmt::Formatter<'_>,
            dimensions: &[Dimension],
            mut write_element: impl FnMut(&mut std::fmt::Formatter<'_>, usize) -> std::fmt::Result,
        ) -> std::fmt::Result {
            // Renders the suffix of dimensions rooted at `dimensions`, consuming leaf elements through `flat_index`.
            fn write_dimensions(
                formatter: &mut std::fmt::Formatter<'_>,
                dimensions: &[Dimension],
                flat_index: &mut usize,
                write_element: &mut impl FnMut(&mut std::fmt::Formatter<'_>, usize) -> std::fmt::Result,
            ) -> std::fmt::Result {
                let Some((dimension, nested_dimensions)) = dimensions.split_first() else {
                    let index = *flat_index;
                    *flat_index += 1;
                    return write_element(formatter, index);
                };
                let Dimension::Static(extent) = dimension else {
                    unreachable!("materialized arrays always have static shapes")
                };
                formatter.write_str("[")?;
                for index in 0..*extent {
                    if index > 0 {
                        formatter.write_str(", ")?;
                    }
                    write_dimensions(formatter, nested_dimensions, flat_index, write_element)?;
                }
                formatter.write_str("]")
            }

            let mut flat_index = 0;
            write_dimensions(formatter, dimensions, &mut flat_index, &mut write_element)
        }
        let dimensions = self.r#type.shape().dimensions();
        let data_type = self.r#type.data_type();
        if matches!(data_type, DataType::Token | DataType::Zero) {
            return write_elements(formatter, dimensions, |formatter, _| {
                formatter.write_str(if data_type == DataType::Token { "token" } else { "zero" })
            });
        }
        let addressing = ArrayAddressing::new(self.r#type.clone()).unwrap();
        match data_type {
            // `f32` and `f64` payloads keep a decimal point through debug formatting, per the rendering contract
            // stated above this implementation.
            DataType::F32 => write_elements(formatter, dimensions, |formatter, element| {
                let value = f32::decode(&self.bytes[addressing.byte_range_for_flat_index(element)]);
                write!(formatter, "{value:?}")
            }),
            DataType::F64 => write_elements(formatter, dimensions, |formatter, element| {
                let value = f64::decode(&self.bytes[addressing.byte_range_for_flat_index(element)]);
                write!(formatter, "{value:?}")
            }),
            _ => dispatch_on_array_element_type!(data_type, |Element| {
                write_elements(formatter, dimensions, |formatter, element| {
                    let value = Element::decode(&self.bytes[addressing.byte_range_for_flat_index(element)]);
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

// Approximate equality requires identical array types. Floating-point payloads compare through their exactly widened
// `f64` values, complex payloads compare both components, and all other element types use exact equality.
impl AbsDiffEq for Array {
    type Epsilon = f64;

    fn default_epsilon() -> f64 {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f64) -> bool {
        if self.r#type != other.r#type {
            return false;
        }
        let data_type = self.r#type.data_type();
        let addressing = ArrayAddressing::new(self.r#type.clone()).unwrap();
        if data_type.is_floating_point() {
            return (0..addressing.element_count()).all(|index| {
                let range = addressing.byte_range_for_flat_index(index);
                let left = Self::element_as_f64(data_type, &self.bytes[range.clone()]).unwrap();
                let right = Self::element_as_f64(data_type, &other.bytes[range]).unwrap();
                (left - right).abs() <= epsilon
            });
        }
        match data_type {
            DataType::C64 => (0..addressing.element_count()).all(|index| {
                let range = addressing.byte_range_for_flat_index(index);
                let left = Complex::<f32>::decode(&self.bytes[range.clone()]);
                let right = Complex::<f32>::decode(&other.bytes[range]);
                (f64::from(left.re) - f64::from(right.re)).abs() <= epsilon
                    && (f64::from(left.im) - f64::from(right.im)).abs() <= epsilon
            }),
            DataType::C128 => (0..addressing.element_count()).all(|index| {
                let range = addressing.byte_range_for_flat_index(index);
                let left = Complex::<f64>::decode(&self.bytes[range.clone()]);
                let right = Complex::<f64>::decode(&other.bytes[range]);
                (left.re - right.re).abs() <= epsilon && (left.im - right.im).abs() <= epsilon
            }),
            _ => self == other,
        }
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

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionVariable, Layout, StridedLayout};
    use crate::operations::complex::Complex;
    use crate::operations::{Compare, ComparisonDirection};

    use super::*;

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
                if message == "cannot materialize a value of dynamically sized type f64[dynamic]; dynamically \
                               shaped values exist only in array programs over 'ArrayIrOperation'",
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
        check_integer_round_trip!(DataType::I32, i32, [i32::MIN, -0x0123_4567, 0x0234_5678, i32::MAX]);
        check_integer_round_trip!(DataType::I64, i64, [i64::MIN, -0x0123_4567_89ab_cdef, i64::MAX]);
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
        assert_eq!(
            Array::from_elements(array_type(DataType::I4, &[2]), &[i4::new(-8).unwrap(), i4::new(7).unwrap()],)
                .unwrap()
                .to_f64s(),
            vec![-8.0, 7.0],
        );
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
        // Rank zero renders as a scalar and each higher rank contributes one bracketed level in logical row-major
        // order. Real floating-point payloads keep a decimal point, while other payloads use scalar rendering.
        assert_eq!(Array::scalar(1.0).to_string(), "1.0");
        assert_eq!(Array::vector(vec![1.0, 2.5]).to_string(), "[1.0, 2.5]");
        assert_eq!(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).to_string(), "[[1.0, 2.0], [3.0, 4.0]]");
        assert_eq!(
            Array::from_elements(array_type(DataType::F64, &[2, 1, 2]), &[1.0, 2.0, 3.0, 4.0])
                .unwrap()
                .to_string(),
            "[[[1.0, 2.0]], [[3.0, 4.0]]]",
        );
        assert_eq!(Array::vector(vec![1i32, 2]).to_string(), "[1, 2]");
        assert_eq!(Array::vector(vec![true, false]).to_string(), "[true, false]");
        let complex = Array::vector(vec![1.0]).complex(&Array::vector(vec![2.0])).unwrap();
        assert_eq!(complex.to_string(), "[1+2i]");
        assert_eq!(Array::vector(Vec::<f64>::new()).to_string(), "[]");
        assert_eq!(
            Array::from_elements(array_type(DataType::F64, &[2, 0]), &[] as &[f64]).unwrap().to_string(),
            "[[], []]",
        );
        assert_eq!(Array::new(array_type(DataType::Token, &[]), Vec::new()).unwrap().to_string(), "token");
        assert_eq!(
            Array::new(array_type(DataType::Zero, &[2, 1]), Vec::new()).unwrap().to_string(),
            "[[zero], [zero]]",
        );

        // Rendering follows logical coordinates rather than physical storage order.
        let column_major = array_type(DataType::F64, &[2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![
            size_of::<f64>() as isize,
            2 * size_of::<f64>() as isize,
        ])));
        assert_eq!(
            Array::from_elements(column_major, &[1.0, 2.0, 3.0, 4.0]).unwrap().to_string(),
            "[[1.0, 2.0], [3.0, 4.0]]",
        );

        // Sub-byte payloads render through their typed elements, which have no scalar representation, and the debug
        // rendering shares the same element list.
        let narrow =
            Array::from_elements(array_type(DataType::I4, &[2]), &[i4::new(-8).unwrap(), i4::new(7).unwrap()]).unwrap();
        assert_eq!(narrow.to_string(), "[-8, 7]");
        assert!(format!("{narrow:?}").ends_with("values: [-8, 7] }"));
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
        // Equality decodes typed values directly, retaining IEEE NaN and signed-zero semantics rather than relying on
        // physical byte equality.
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
        // Approximate equality reads low-precision, arbitrarily laid-out values directly from physical storage.
        let r#type = array_type(DataType::F8E4M3FN, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
        let left =
            Array::from_elements(r#type.clone(), &[f8e4m3fn::from_f64(1.0).unwrap(), f8e4m3fn::from_f64(2.0).unwrap()])
                .unwrap();
        let right =
            Array::from_elements(r#type, &[f8e4m3fn::from_f64(1.125).unwrap(), f8e4m3fn::from_f64(2.0).unwrap()])
                .unwrap();
        assert_abs_diff_eq!(left, right, epsilon = 0.2);
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
    fn test_array_convert_element_type() {
        // Every materialized element data type converts to every other one without falling back to a dynamic scalar
        // representation. The common value one is exactly representable in every supported format.
        let data_types = [
            DataType::Boolean,
            DataType::I1,
            DataType::I2,
            DataType::I4,
            DataType::I8,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::U1,
            DataType::U2,
            DataType::U4,
            DataType::U8,
            DataType::U16,
            DataType::U32,
            DataType::U64,
            DataType::F4E2M1FN,
            DataType::F6E2M3FN,
            DataType::F6E3M2FN,
            DataType::F8E3M4,
            DataType::F8E4M3,
            DataType::F8E4M3FN,
            DataType::F8E4M3FNUZ,
            DataType::F8E4M3B11FNUZ,
            DataType::F8E5M2,
            DataType::F8E5M2FNUZ,
            DataType::F8E8M0FNU,
            DataType::BF16,
            DataType::F16,
            DataType::F32,
            DataType::F64,
            DataType::C64,
            DataType::C128,
        ];
        for source_data_type in data_types {
            let source = Array::from_f64s(array_type(source_data_type, &[1]), vec![1.0]);
            for target_data_type in data_types {
                let converted = source.converted_to(target_data_type).unwrap();
                assert_eq!(converted.r#type().into_owned(), array_type(target_data_type, &[1]));
            }
        }

        // Representative values pin Boolean truth, integer truncation and sub-byte modular narrowing.
        let vector = Array::vector(vec![0.0, 1.5]);
        assert_eq!(vector.converted_to(DataType::Boolean).unwrap(), Array::vector(vec![false, true]));
        assert_eq!(vector.converted_to(DataType::I32).unwrap(), Array::vector(vec![0i32, 1]));
        let signed = Array::from_elements(
            array_type(DataType::I4, &[3]),
            &[i4::new(-8).unwrap(), i4::new(-1).unwrap(), i4::new(7).unwrap()],
        )
        .unwrap();
        assert_eq!(
            signed.converted_to(DataType::U2).unwrap().elements::<u2>(),
            Ok(vec![u2::new(0).unwrap(), u2::new(3).unwrap(), u2::new(3).unwrap()]),
        );
        assert_eq!(
            Array::from_elements(array_type(DataType::U4, &[1]), &[u4::new(15).unwrap()])
                .unwrap()
                .converted_to(DataType::I4)
                .unwrap()
                .elements::<i4>(),
            Ok(vec![i4::new(-1).unwrap()]),
        );

        // Complex conversion preserves both components only for complex destinations and otherwise converts the real
        // component, except that Boolean conversion observes whether either component is nonzero.
        let complex = Array::vector(vec![ComplexNumber::new(0.0f32, 2.0), ComplexNumber::new(-1.5, 0.0)]);
        assert_eq!(complex.converted_to(DataType::Boolean).unwrap().elements::<bool>(), Ok(vec![true, true]),);
        assert_eq!(complex.converted_to(DataType::I32).unwrap().elements::<i32>(), Ok(vec![0, -1]),);
        assert_eq!(
            complex.converted_to(DataType::C128).unwrap().elements::<ComplexNumber<f64>>(),
            Ok(vec![ComplexNumber::new(0.0, 2.0), ComplexNumber::new(-1.5, 0.0)]),
        );

        // Conversions into low-precision floating-point element types produce exact encodings, including their
        // format-specific fallible cases.
        let low_precision = vector.converted_to(DataType::F8E5M2).unwrap();
        assert_eq!(low_precision.elements::<f8e5m2>().unwrap()[1].to_bits(), 0x3e);
        assert_eq!(
            Array::scalar(1e9f64).converted_to(DataType::F8E4M3FN).unwrap().elements::<f8e4m3fn>().unwrap()[0]
                .to_bits(),
            0x7f,
        );
        assert!(matches!(
            Array::scalar(0.0f64).converted_to(DataType::F8E8M0FNU),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "data type f8e8m0fnu cannot represent zero",
        ));

        // Cross-type conversion traverses the logical order selected by the input layout and preserves the same
        // physical-layout descriptor on its output type.
        let input_type = array_type(DataType::F64, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-16])));
        let converted = Array::from_elements(input_type, &[1.9f64, -2.9]).unwrap().converted_to(DataType::I32).unwrap();
        assert_eq!(
            converted.r#type().into_owned(),
            array_type(DataType::I32, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-16]))),
        );
        assert_eq!(converted.elements::<i32>(), Ok(vec![1, -2]));
        assert_eq!(converted.storage_bytes().len(), 20);

        // Same-type conversion shares the original bytes, preserving NaN payloads and every unoccupied layout byte.
        let nan = Array::vector(vec![f32::from_bits(0x7fc0_1234)]);
        let unchanged = nan.converted_to(DataType::F32).unwrap();
        assert!(Arc::ptr_eq(nan.shared_storage(), unchanged.shared_storage()));
        assert_eq!(unchanged.storage_bytes(), nan.storage_bytes());

        // Token conversion is always rejected. Structural-zero conversion is valid only when it is a same-type no-op.
        assert!(matches!(
            vector.converted_to(DataType::Token),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the token data type",
        ));
        let token = Array::from_logical_bytes(array_type(DataType::Token, &[1]), &[]).unwrap();
        assert!(matches!(
            token.converted_to(DataType::Token),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the token data type",
        ));
        let zero = Array::from_logical_bytes(array_type(DataType::Zero, &[2]), &[]).unwrap();
        assert_eq!(zero.converted_to(DataType::Zero), Ok(zero.clone()));
        assert!(matches!(
            vector.converted_to(DataType::Zero),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the zero data type",
        ));
        assert!(matches!(
            zero.converted_to(DataType::F32),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the zero data type",
        ));
    }

    #[test]
    fn test_array_encoding_fidelity() {
        // Conversions and arithmetic on low-precision floating-point arrays operate on genuine encodings: the payload
        // round-trips through the exact bit patterns rather than an `f64` pun.
        let array = Array::from_f64s(array_type(DataType::F8E8M0FNU, &[2]), vec![2.0, 0.5]);
        assert_eq!(array.elements::<f8e8m0fnu>().unwrap()[0].to_bits(), 0x80);
        assert_eq!(array.elements::<f8e8m0fnu>().unwrap()[1].to_bits(), 0x7e);
        let converted = array.converted_to(DataType::BF16).unwrap();
        assert_eq!(converted.to_f64s(), vec![2.0, 0.5]);
        let round_trip = converted.converted_to(DataType::F8E8M0FNU).unwrap();
        assert_eq!(round_trip, array);
    }
}
