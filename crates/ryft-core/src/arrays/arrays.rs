use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use approx::AbsDiffEq;
use half::{bf16, f16};
use num_complex::Complex;

use ryft_macros::Parameter;

use crate::arrays::addressing::{ArrayAddressing, ArraySliceAxis};
use crate::arrays::broadcasting::Broadcastable;
use crate::arrays::elements::{
    ArrayElement, decode_elements, decode_logical_bytes, encode_elements, encode_logical_bytes, f4e2m1fn, f6e2m3fn,
    f6e3m2fn, f8e3m4, f8e4m3, f8e4m3b11fnuz, f8e4m3fn, f8e4m3fnuz, f8e5m2, f8e5m2fnuz, f8e8m0fnu, i1, i2, i4, u1, u2,
    u4,
};
use crate::arrays::macros::dispatch_on_array_element_type;
use crate::arrays::operations::ArrayOperation;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::{Dimension, Shape, StaticShape};
use crate::contexts::EagerContext;
use crate::operations::ElementType;
use crate::parameters::Parameter;
use crate::programs::{Concretizable, LiteralIdentity, ProgramError, TypeError, Typed, Value, ValueDirectDispatch};

/// Dense multidimensional [`Value`] whose [`Type`](crate::Type) is an [`ArrayType`]. It is the reference array value
/// of Ryft, and it exists primarily to exercise the tracing, transformation, and interpretation machinery with programs
/// over multidimensional arrays without depending on an optimized backend such as the Ryft XLA backend. Unit tests,
/// documentation tests, and downstream crates can therefore interpret complete array programs eagerly and stage them
/// through [`TracingContext`](crate::TracingContext).
///
/// The payload is a shared immutable byte buffer whose physical placement is determined by the array's [`ArrayType`].
/// Missing [`Layout`](crate::Layout) metadata implies a dense row-major storage while explicit strided and tiled
/// layouts determine the physical ordering, as well as any potential "holes" and padding. [`Array::new`] validates the
/// complete physical representation, while [`Array::from_elements`] and [`Array::from_logical_bytes`] construct it from
/// logical row-major values through checked typed codecs that preserve exact element encodings. Cloning an array shares
/// its payload without copying it.
///
/// A production [`Array`] always carries a fully static [`ArrayType`]. Every constructor that sizes or addresses a
/// payload funnels through [`ArrayAddressing::new`], which rejects any type with a [`Dimension::Dynamic`] axis, so a
/// dynamically shaped array value cannot be built. Reference kernels may therefore assume static geometry and read
/// extents directly off the stored type instead of resolving first-class dimension extent. [`Program`](crate::Program)s
/// that genuinely need dynamic shapes stage over [`ArrayIrOperation`](crate::ArrayIrOperation) instead, where each
/// dynamic axis is carried by an explicit dimension input.
///
/// Host concretization through [`Concretizable`] requires rank zero. Integer scalars can be extracted into any Rust
/// integer or Ryft sub-byte integer type when the value fits; [`Concretizable<i128>`] preserves every supported integer
/// value, including [`u64::MAX`]. Boolean, floating-point, and complex extraction requires the matching element type
/// and preserves its exact encoding, including signed zeros and NaN payloads. Incompatible shapes, element types, and
/// out-of-range integers return [`ProgramError::Concretization`]. Use [`Array::converted_to`] for numerical
/// conversions.
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
/// # use ryft_core::operations::arithmetic::Add;
/// let left = Array::vector(vec![1.0, 2.0]).unwrap();
/// let right = Array::vector(vec![3.0, 4.0]).unwrap();
/// assert_eq!(left.add(&right).unwrap(), Array::vector(vec![4.0, 6.0]).unwrap());
/// ```
#[derive(Clone, Parameter)]
pub struct Array {
    /// [`ArrayType`] of this [`Array`].
    r#type: ArrayType,

    /// Shared immutable physical storage, accounting for any layout "holes" or tile padding.
    bytes: Arc<Vec<u8>>,
}

impl Array {
    /// Creates a new [`Array`] with the provided [`ArrayType`] and backed by the provided complete physical storage.
    /// The provided storage (i.e., `bytes`) must have the exact [`Layout`](crate::Layout)-derived byte count, must
    /// contain a valid encoding for every logical element of the array, and must contain zero in every layout "hole"
    /// or padding byte. Dynamically shaped types are rejected because they cannot describe materialized storage.
    #[inline]
    pub fn new(r#type: ArrayType, bytes: Vec<u8>) -> Result<Self, ProgramError> {
        ArrayAddressing::new(r#type.clone())?.validate_storage_bytes(&bytes)?;
        Ok(Self { r#type, bytes: Arc::new(bytes) })
    }

    /// Creates a new [`Array`] from `elements` assuming they are provided densely packed in logical row-major order.
    #[inline]
    pub fn from_elements<T: ArrayElement>(r#type: ArrayType, elements: &[T]) -> Result<Self, ProgramError> {
        let bytes = encode_elements(&r#type, elements)?;
        Ok(Self { r#type, bytes: Arc::new(bytes) })
    }

    /// Creates a new [`Array`] from concatenated logical element encodings in row-major order. Places each element
    /// according to the provided type's layout and fills any storage holes or tile padding with zeros. In contrast,
    /// [`Array::new`] expects bytes that already represent the complete physical storage, including holes and padding.
    /// The two byte representations coincide for a dense row-major layout.
    ///
    /// # Example
    ///
    /// A two-element [`DataType::U8`] array with a byte stride of two requires a zero-filled hole between its elements:
    ///
    /// ```
    /// # use ryft_core::{Array, ArrayType, DataType, Layout, StridedLayout};
    /// #
    /// # fn example() -> Result<(), ryft_core::ProgramError> {
    /// let array_type = ArrayType::new_static(DataType::U8, [2])
    ///     .with_layout(Layout::Strided(StridedLayout::new(vec![2])));
    /// let array = Array::from_logical_bytes(array_type.clone(), &[10, 20])?;
    /// assert_eq!(array.logical_bytes(), vec![10, 20]);
    /// assert_eq!(array.storage_bytes(), &[10, 0, 20]);
    /// assert_eq!(array, Array::new(array_type, vec![10, 0, 20])?);
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    #[inline]
    pub fn from_logical_bytes(r#type: ArrayType, bytes: &[u8]) -> Result<Self, ProgramError> {
        let bytes = encode_logical_bytes(&r#type, bytes)?;
        Ok(Self { r#type, bytes: Arc::new(bytes) })
    }

    /// Creates a new rank-0 [`Array`] containing `value`.
    #[inline]
    pub fn scalar<T: ArrayElement>(value: T) -> Result<Self, ProgramError> {
        Self::from_elements(ArrayType::scalar(T::data_type()), &[value])
    }

    /// Creates a new rank-1 [`Array`] containing `elements` in logical order.
    #[inline]
    pub fn vector<T: ArrayElement>(elements: Vec<T>) -> Result<Self, ProgramError> {
        Self::from_elements(
            ArrayType::new(T::data_type(), Shape::new(vec![Dimension::Static(elements.len())])),
            &elements,
        )
    }

    /// Creates a new rank-2 [`Array`] containing `elements` in logical row-major order. Returns an error if the element
    /// count does not equal `rows * columns` or the shape cannot be represented.
    #[inline]
    pub fn matrix<T: ArrayElement>(rows: usize, columns: usize, elements: Vec<T>) -> Result<Self, ProgramError> {
        Self::from_elements(
            ArrayType::new(T::data_type(), Shape::new(vec![Dimension::Static(rows), Dimension::Static(columns)])),
            &elements,
        )
    }

    /// Creates a new [`Array`] with the provided [`ArrayType`] and backed by the provided shared physical storage
    /// without performing any validation for either. The caller guarantees what [`Array::new`] would otherwise check,
    /// namely that `bytes` has the exact layout-derived byte count for `type`, holds a valid encoding for every logical
    /// element, and is zero in every layout "hole" and padding byte. This exists because the reference Ryft kernels for
    /// [`Array`]s build their results by writing into an addressed buffer they sized from the output type, and so they
    /// already uphold the storage invariants by construction and would otherwise pay for a second full traversal per
    /// operation. Taking the payload as an [`Arc`] also lets kernels that only retype a value (such as a memory
    /// transfer or a reshard kernel) share the original payload instead of copying it.
    #[inline]
    pub(crate) fn new_unchecked(r#type: ArrayType, bytes: Arc<Vec<u8>>) -> Self {
        Self { r#type, bytes }
    }

    /// Returns the number of elements represented by `type`, or an error if its element count is not statically
    /// known or does not fit in [`usize`]. A statically zero dimension makes the count zero even when another
    /// dimension is dynamic.
    #[inline]
    pub fn element_count(r#type: &ArrayType) -> Result<usize, ProgramError> {
        r#type.element_count()?.ok_or_else(|| {
            TypeError::invalid(format!("cannot materialize a value of dynamically sized type `{type}`")).into()
        })
    }

    /// Decodes this array as typed elements in logical row-major order.
    #[inline]
    pub fn elements<T: ArrayElement>(&self) -> Result<Vec<T>, ProgramError> {
        decode_elements(&self.r#type, self.bytes.as_slice())
    }

    /// Decodes this integer array as host indices or sizes in logical row-major order, rejecting negative entries and
    /// entries that do not fit in `usize`. Every signed and unsigned integer element type, including the sub-byte ones,
    /// is widened losslessly before the range check, so large unsigned values are never misreported as negative. This
    /// is used by reference kernels that consume integer metadata inputs (e.g., offsets and group sizes).
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of this array (e.g., the metadata input it is passed as), used in error messages.
    pub(crate) fn non_negative_integer_elements(&self, name: &str) -> Result<Vec<usize>, ProgramError> {
        let data_type = self.r#type.data_type();
        dispatch_on_array_element_type!(@integer data_type, |Element| {
            self.elements::<Element>()?
                .into_iter()
                .enumerate()
                .map(|(index, value)| {
                    let value = if data_type.is_signed() {
                        value.convert_to::<i64>().map(i128::from)?
                    } else {
                        value.convert_to::<u64>().map(i128::from)?
                    };
                    if value < 0 {
                        return Err(ProgramError::InvalidArgument {
                            message: format!("`{name}[{index}]` must be non-negative but got {value}"),
                        });
                    }
                    usize::try_from(value).map_err(|_| ProgramError::InvalidArgument {
                        message: format!("`{name}[{index}]` value {value} does not fit in `usize`"),
                    })
                })
                .collect()
        })
    }

    /// Returns the row-major payload of this array converted elementwise to `f64`. This is a "test assertion view" for
    /// real-valued arrays (Booleans convert to `0.0`/`1.0` and integers to their exact values where representable),
    /// and it panics for arrays whose elements cannot be viewed as real numbers (i.e., complex, token, and
    /// structural-zero element data types), because in tests such a failure corresponds to the assertion failing.
    pub fn to_f64s(&self) -> Vec<f64> {
        let data_type = self.r#type.data_type();
        if data_type.is_complex() {
            panic!("cannot view an array of complex element data type `{data_type}` as `f64` values");
        }
        let addressing = ArrayAddressing::new(self.r#type.clone()).unwrap();
        (0..addressing.element_count())
            .map(|index| {
                data_type.element_as_f64(&self.bytes[addressing.byte_range_for_flat_index(index)]).unwrap_or_else(
                    || panic!("cannot view an array of element data type `{data_type}` as `f64` values"),
                )
            })
            .collect()
    }

    /// Returns the concatenated logical element encodings in row-major order, omitting layout holes and tile padding.
    #[inline]
    pub fn logical_bytes(&self) -> Vec<u8> {
        decode_logical_bytes(&self.r#type, self.bytes.as_slice()).unwrap()
    }

    /// Returns the complete immutable physical storage, including layout holes and tile padding.
    #[inline]
    pub fn storage_bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    /// Returns the complete physical storage for in-place mutation, copying the payload first when it is shared with
    /// another array. Kernels that build a result by mutating a buffer they own (or one they just cloned from an
    /// input) use this to avoid a second allocation.
    #[inline]
    pub fn storage_bytes_mut(&mut self) -> &mut [u8] {
        Arc::make_mut(&mut self.bytes).as_mut_slice()
    }

    /// Returns the shared handle to this array's physical storage, so that a kernel which only retypes a value can
    /// hand the same payload to [`Array::new_unchecked`] instead of copying it.
    #[inline]
    pub fn shared_storage_bytes(&self) -> &Arc<Vec<u8>> {
        &self.bytes
    }

    /// Creates a new array holding this array's elements converted into `data_type`, preserving shape, sharding, and
    /// memory space. Tiled layouts are preserved while byte-stride layouts are cleared when element storage width
    /// changes. This is the foundational cast of the reference backend: the
    /// [`ConvertElementType`](crate::ConvertElementType) capability delegates to it, and
    /// so does every kernel that promotes mixed-type inputs through [`Array::promoted_to`].
    ///
    /// Conversion of an individual element is exactly [`ArrayElement::convert_to`], so the per-element semantics
    /// (including rounding, truncation, saturation, and exceptional-value handling) are documented on that trait.
    /// Converting an array to its own element data type shares the existing payload instead of copying it. Token
    /// conversions are always rejected, and structural-zero conversion is accepted only as a same-type no-op.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Element [`DataType`] of the result.
    ///
    /// # Errors
    ///
    /// Returns an error if either data type is [`DataType::Token`] or exactly one is [`DataType::Zero`]. Numerical
    /// conversion maps zero to NaN for `f8e8m0fnu`. Finite-only microscaling formats map NaN to their positive maximum
    /// and saturate infinities to their signed finite limits. Explicit checked element constructors retain their own
    /// representability checks.
    pub fn converted_to(&self, data_type: DataType) -> Result<Self, ProgramError> {
        let source_data_type = self.r#type.data_type();
        if source_data_type.is_token() || data_type.is_token() {
            return Err(TypeError::invalid("cannot convert values to or from the `token` data type").into());
        }
        if source_data_type == data_type {
            return Ok(self.clone());
        }
        if source_data_type.is_zero() || data_type.is_zero() {
            return Err(TypeError::invalid("cannot convert values to or from the `zero` data type").into());
        }
        let output_type = self.r#type.with_element_type(data_type);

        // The nested dispatch selects the concrete source and destination element types, which monomorphizes
        // `convert_to` into the pair's direct conversion (refer to the documentation of `ArrayElement::convert_to`).
        // Should a measured hot pair ever justify a bespoke kernel, it can be matched here ahead of the generic path
        // without changing the element interchange contract.
        dispatch_on_array_element_type!(source_data_type, |Input| {
            dispatch_on_array_element_type!(data_type, |Output| {
                self.map_elements::<Input, Output>(output_type, Input::convert_to::<Output>)
            })
        })
    }

    /// Converts this array to the provided element data type, borrowing it unchanged when it already has that data type
    /// so that already-promoted inputs keep their exact physical storage and layout. Kernels that promote mixed-type
    /// inputs to a common element data type (which each kernel computes from its own type-inference contract) use this
    /// to convert only the mismatched inputs.
    #[inline]
    pub fn promoted_to(&self, data_type: DataType) -> Result<Cow<'_, Self>, ProgramError> {
        if self.r#type.data_type() == data_type {
            Ok(Cow::Borrowed(self))
        } else {
            Ok(Cow::Owned(self.converted_to(data_type)?))
        }
    }

    /// Applies a typed elementwise function to this array in logical row-major order, producing a new array of
    /// `output_type`. Both arrays use their sealed codecs one element at a time, so the only payload allocation
    /// is the result buffer, and the output layout may differ from the input layout.
    ///
    /// # Parameters
    ///
    ///   - `output_type`: Static array type of the result. Its [`DataType`] must be represented by `Output` and
    ///     its logical element count must equal this array's (elementwise kernels typically preserve the shape).
    ///   - `function`: Elementwise function applied to each decoded `Input` element.
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
                "cannot map elements of data type `{}` as `{}` values",
                self.r#type.data_type(),
                Input::data_type(),
            ))
            .into());
        }
        if output_type.data_type() != Output::data_type() {
            return Err(TypeError::invalid(format!(
                "cannot store mapped `{}` values in an array of element data type `{}`",
                Output::data_type(),
                output_type.data_type(),
            ))
            .into());
        }
        let input_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        if input_addressing.element_count() != output_addressing.element_count() {
            return Err(TypeError::invalid(format!(
                "cannot map {} logical elements onto array type `{}` with {} logical elements",
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

    /// Applies a typed binary element function with NumPy-style broadcasting directly over addressed storage. Inputs
    /// and outputs use their sealed codecs one element at a time, so the only payload allocation is the result buffer.
    /// Both inputs must already have `Input`'s element type as this function does not promote or convert inputs.
    /// The output must have `Output`'s element type and the fully static broadcast shape of the inputs. Invalid
    /// element types, shapes, or layouts return an error. Empty outputs do not invoke `function`.
    ///
    /// # Parameters
    ///
    ///   - `rhs`: Right input, whose shape must broadcast with this array's shape.
    ///   - `output_type`: Result type, including its element type, broadcast shape, and storage layout.
    ///     Operation-specific sharding and reduction metadata constraints must be checked by the caller.
    ///   - `function`: Function applied to each decoded pair of `Input` elements.
    pub fn map_element_pairs<Input: ArrayElement, Output: ArrayElement>(
        &self,
        rhs: &Self,
        output_type: ArrayType,
        function: impl Fn(Input, Input) -> Result<Output, ProgramError>,
    ) -> Result<Self, ProgramError> {
        if self.r#type.data_type() != Input::data_type() || rhs.r#type.data_type() != Input::data_type() {
            return Err(TypeError::invalid(format!(
                "binary element inputs must both have data type `{}`, got `{}` and `{}`",
                Input::data_type(),
                self.r#type.data_type(),
                rhs.r#type.data_type(),
            ))
            .into());
        }

        if output_type.data_type() != Output::data_type() {
            return Err(TypeError::invalid(format!(
                "binary element output must have data type `{}`, got `{}`",
                Output::data_type(),
                output_type.data_type(),
            ))
            .into());
        }

        Self::element_count(&output_type)?;

        let broadcast_shape = self
            .r#type
            .shape()
            .broadcast(rhs.r#type.shape())
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        if output_type.shape() != &broadcast_shape {
            return Err(TypeError::invalid(format!(
                "binary element output shape must be {}, got {}",
                broadcast_shape,
                output_type.shape(),
            ))
            .into());
        }

        let output_shape = output_type.static_shape().unwrap();
        let lhs_shape = self.r#type.static_shape().unwrap();
        let rhs_shape = rhs.r#type.static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let lhs_strides = lhs_shape.row_major_strides();
        let rhs_strides = rhs_shape.row_major_strides();
        let lhs_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let rhs_addressing = ArrayAddressing::new(rhs.r#type.clone())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut output_bytes = vec![0; output_addressing.storage_byte_len()];
        for output_index in 0..output_addressing.element_count() {
            let lhs_index =
                Self::broadcast_index(output_index, &output_shape, &output_strides, &lhs_shape, &lhs_strides);
            let rhs_index =
                Self::broadcast_index(output_index, &output_shape, &output_strides, &rhs_shape, &rhs_strides);
            let left = Input::decode(&self.bytes[lhs_addressing.byte_range_for_flat_index(lhs_index)]);
            let right = Input::decode(&rhs.bytes[rhs_addressing.byte_range_for_flat_index(rhs_index)]);
            let output = function(left, right)?;
            output.encode(&mut output_bytes[output_addressing.byte_range_for_flat_index(output_index)]);
        }

        Ok(Self { r#type: output_type, bytes: Arc::new(output_bytes) })
    }

    /// Creates an array of `type` by evaluating a typed function at every flat logical row-major element index. This
    /// is the constructor form of [`Array::map_elements`], serving iota-style and coordinate-dependent kernels.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Static array type of the result, whose [`DataType`] must be represented by `T`.
    ///   - `function`: Function producing the element at each flat logical row-major index.
    ///
    /// # Errors
    ///
    /// Returns an error if `r#type` cannot describe materialized storage, if `T` represents a different [`DataType`],
    /// or if `function` fails.
    pub fn from_fn_elements<T: ArrayElement, F: Fn(usize) -> Result<T, ProgramError>>(
        r#type: ArrayType,
        function: F,
    ) -> Result<Self, ProgramError> {
        if r#type.data_type() != T::data_type() {
            return Err(TypeError::invalid(format!(
                "cannot store `{}` values in an array of element data type `{}`",
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

    /// Creates an array of `output_type` whose every element is copied from this array through an output-to-input index
    /// mapping over flat logical row-major indices. The copy moves whole element encodings without decoding them, so
    /// this is the element-data-type-agnostic workhorse behind structural kernels such as transpose, broadcast, slice,
    /// reverse, and gather, which never need element-type dispatch.
    ///
    /// # Parameters
    ///
    ///   - `output_type`: Static array type of the result, which must have the same [`DataType`] as this array.
    ///   - `index`: Mapping from each flat logical output element index to the flat logical input element index whose
    ///     element it copies. Input indices may repeat or be skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if either array type cannot describe materialized storage, if the element data types differ,
    /// or if `index` produces an out-of-bounds input index.
    pub fn gather_elements<F: Fn(usize) -> usize>(
        &self,
        output_type: ArrayType,
        index: F,
    ) -> Result<Self, ProgramError> {
        if output_type.data_type() != self.r#type.data_type() {
            return Err(TypeError::invalid(format!(
                "cannot gather elements of data type `{}` into an array of element data type `{}`",
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
                    "gather index {} is out of bounds for {} elements",
                    input_element,
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

    /// Maps one flat row-major output index to the corresponding flat input index under NumPy-style broadcasting.
    /// Input axes are right-aligned with output axes, and an input extent of one always selects coordinate zero.
    /// Both stride slices must be logical row-major element strides, independent of physical storage layouts.
    /// The caller uses [`ArrayAddressing::byte_range_for_flat_index`] to map the returned logical input index
    /// and the output index to their respective physical byte ranges, including strided and tiled layouts.
    pub(crate) fn broadcast_index(
        output_index: usize,
        output_shape: &StaticShape,
        output_row_major_strides: &[usize],
        input_shape: &StaticShape,
        input_row_major_strides: &[usize],
    ) -> usize {
        let output_axis_offset = output_shape.rank() - input_shape.rank();
        (0..input_shape.rank()).fold(0, |index, input_axis| {
            let output_axis = output_axis_offset + input_axis;
            let coordinate = if input_shape[input_axis] == 1 {
                0
            } else {
                (output_index / output_row_major_strides[output_axis]) % output_shape[output_axis]
            };
            index + coordinate * input_row_major_strides[input_axis]
        })
    }

    /// Copies the logical block selected by `axes` into a new array of `output_type`. The caller guarantees that the
    /// selection lies in bounds and contains _exactly_ the output's logical element count.
    pub(crate) fn copy_block(&self, output_type: ArrayType, axes: &[ArraySliceAxis]) -> Result<Self, ProgramError> {
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
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
                bytes[output_start..output_start + input_bytes.len()]
                    .copy_from_slice(&self.storage_bytes()[input_bytes]);
                output_index += element_count;
                continue;
            }
            for offset in 0..element_count {
                let input_start = input_bytes.start + offset * element_byte_width;
                bytes[output_addressing.byte_range_for_flat_index(output_index)]
                    .copy_from_slice(&self.storage_bytes()[input_start..input_start + element_byte_width]);
                output_index += 1;
            }
        }
        debug_assert_eq!(output_index, output_addressing.element_count());
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }

    /// Overwrites the logical block of `update`'s shape starting at `start_indices` in this array with `update`.
    /// The caller guarantees that the block lies in bounds.
    pub(crate) fn replace_block(self, update: &Array, start_indices: &[usize]) -> Self {
        let update_shape = update.r#type().static_shape().unwrap();
        let addressing = ArrayAddressing::new(self.r#type().into_owned()).unwrap();
        let update_addressing = ArrayAddressing::new(update.r#type().into_owned()).unwrap();
        let axes = start_indices
            .iter()
            .zip(update_shape.dimensions())
            .map(|(start, size)| ArraySliceAxis::new(*start, *size, 1))
            .collect::<Vec<_>>();
        let ranges = addressing.ranges(&axes).unwrap();
        let element_byte_width = addressing.element_byte_width();
        let mut output = self;
        let bytes = output.storage_bytes_mut();
        let update_is_dense = update_addressing.is_dense_row_major();
        let mut written = 0usize;
        for range in ranges {
            let output_bytes = range.bytes();
            let element_count = range.elements().len();
            if update_is_dense {
                let update_start = written * element_byte_width;
                bytes[output_bytes].copy_from_slice(
                    &update.storage_bytes()[update_start..update_start + element_count * element_byte_width],
                );
                written += element_count;
                continue;
            }
            for offset in 0..element_count {
                let output_start = output_bytes.start + offset * element_byte_width;
                bytes[output_start..output_start + element_byte_width]
                    .copy_from_slice(&update.storage_bytes()[update_addressing.byte_range_for_flat_index(written)]);
                written += 1;
            }
        }
        debug_assert_eq!(written, update_addressing.element_count());
        output
    }
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

impl Display for Array {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Arrays render in logical shape order. Specifically, a scalar renders as one element, and every array
        // dimension contributes one bracketed nesting level. Real floating-point payloads use debug formatting so
        // integral values retain a decimal point (e.g., `1.0` rather than `1`), keeping the element type visually
        // apparent in diagnostics.

        /// Renders elements in logical row-major order, adding one bracketed level per static array dimension.
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
            // `f32` and `f64` payloads keep a decimal point through debug formatting,
            // per the rendering contract stated above this implementation.
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

impl PartialEq for Array {
    fn eq(&self, other: &Self) -> bool {
        if self.r#type != other.r#type {
            return false;
        }

        let data_type = self.r#type.data_type();
        if matches!(data_type, DataType::Token | DataType::Zero) {
            return true;
        }

        // Compare typed values rather than physical byte patterns: signed floating-point zeros compare equal,
        // while NaNs compare unequal.
        let addressing = ArrayAddressing::new(self.r#type.clone()).unwrap();
        dispatch_on_array_element_type!(data_type, |Element| {
            (0..addressing.element_count()).all(|index| {
                let range = addressing.byte_range_for_flat_index(index);
                Element::decode(&self.bytes[range.clone()]) == Element::decode(&other.bytes[range])
            })
        })
    }
}

impl Typed for Array {
    type Type = ArrayType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl Value for Array {
    type Dispatch = ValueDirectDispatch;

    // A concrete `Array`'s active context is the reference backend's rich eager domain (unlike the constant-only
    // `EagerContext<Array>` it declares as its `Value::Domain`, which cannot bind operations), so free transform
    // entry points such as `crate::batching::batch` serve top-level concrete values.
    type Domain = EagerContext<Self, ArrayOperation<Self>>;

    #[inline]
    fn domain(&self) -> EagerContext<Self, ArrayOperation<Self>> {
        EagerContext::new()
    }

    fn is_zero(&self) -> bool {
        match self.r#type.data_type() {
            DataType::Token => false,
            DataType::Zero => true,
            data_type => dispatch_on_array_element_type!(data_type, |Element| {
                // Decode logical elements rather than inspecting physical bytes: signed zeros have non-zero bytes,
                // and strided or tiled layouts can contain padding that is not part of the array's value.
                let addressing = ArrayAddressing::new(self.r#type.clone()).unwrap();
                (0..addressing.element_count()).all(|index| {
                    Element::decode(&self.bytes[addressing.byte_range_for_flat_index(index)])
                        .convert_to::<Complex<f64>>()
                        .is_ok_and(|value| value == Complex::new(0.0, 0.0))
                })
            }),
        }
    }
}

impl LiteralIdentity for Array {
    // An array's physical storage is canonical for its type (layout holes and tile padding are always zero), so
    // comparing the type and the storage bytes compares the literal exactly. Shared storage short-circuits the
    // byte comparison.

    #[inline]
    fn literal_eq(&self, other: &Self) -> bool {
        self.r#type() == other.r#type()
            && (std::ptr::eq(self.storage_bytes(), other.storage_bytes())
                || self.storage_bytes() == other.storage_bytes())
    }

    #[inline]
    fn literal_hash<H: Hasher>(&self, state: &mut H) {
        self.r#type().hash(state);
        self.storage_bytes().hash(state);
    }
}

impl TryFrom<bool> for Array {
    type Error = ProgramError;

    #[inline]
    fn try_from(value: bool) -> Result<Self, Self::Error> {
        Self::scalar(value)
    }
}

impl AbsDiffEq for Array {
    // Approximate equality requires identical array types. Floating-point payloads compare through their exactly
    // widened `f64` values, complex payloads compare both components, and all other element types use exact equality.

    type Epsilon = f64;

    #[inline]
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
                let left = data_type.element_as_f64(&self.bytes[range.clone()]).unwrap();
                let right = data_type.element_as_f64(&other.bytes[range]).unwrap();
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

// The lossless integer decoder is the foundation for checked concretization into other host integer types.
impl Concretizable<i128> for Array {
    fn concretize(&self) -> Result<i128, ProgramError> {
        let data_type = self.r#type.data_type();
        if self.r#type.rank() != 0 || !data_type.is_integer() {
            return Err(ProgramError::Concretization {
                message: format!("cannot extract a concrete integer from `{}`; expected a scalar integer", self.r#type),
            });
        }

        // Widen through the source's signed or unsigned carrier before converting to `i128`, preserving every
        // supported integer exactly. Checked target narrowing is handled by the generated implementations below.
        let range = ArrayAddressing::new(self.r#type.clone())?.byte_range_for_flat_index(0);
        let bytes = &self.bytes[range];
        dispatch_on_array_element_type!(@integer data_type, |Element| {
            let element = Element::decode(bytes);
            if data_type.is_signed() {
                element.convert_to::<i64>().map(i128::from)
            } else {
                element.convert_to::<u64>().map(i128::from)
            }
        })
    }
}

/// Implements checked integer extraction or exact element decoding for supported host scalar types.
macro_rules! impl_array_scalar_concretization {
    // Boolean extraction shares exact decoding while preserving its established diagnostic.
    (@exact bool) => {
        impl_array_scalar_concretization!(@decode bool, |array| format!(
            "cannot extract a concrete boolean from a value of type `{}`; expected `bool[]`",
            array.r#type(),
        ));
    };

    // Exact floating-point and complex extraction requires the matching element type.
    (@exact $scalar:ty) => {
        impl_array_scalar_concretization!(@decode $scalar, |array| format!(
            "cannot extract a concrete `{}` from `{}`; expected `{}[]`",
            stringify!($scalar),
            array.r#type,
            <$scalar>::data_type(),
        ));
    };

    // Shared exact extraction preserves the stored representation without numerical conversion.
    (@decode $scalar:ty, |$array:ident| $message:expr) => {
        impl Concretizable<$scalar> for Array {
            fn concretize(&self) -> Result<$scalar, ProgramError> {
                if self.r#type.rank() != 0 || self.r#type.data_type() != <$scalar>::data_type() {
                    let $array = self;
                    return Err(ProgramError::Concretization { message: $message });
                }
                let range = ArrayAddressing::new(self.r#type.clone())?.byte_range_for_flat_index(0);
                Ok(<$scalar>::decode(&self.bytes[range]))
            }
        }
    };

    // Integer targets use checked conversion from the lossless common integer representation.
    (@integer $scalar:ty) => {
        impl_array_scalar_concretization!(@checked $scalar, |value| <$scalar>::try_from(value).ok());
    };

    // Sub-byte constructors infer the storage integer type and validate the narrower range.
    (@sub_byte $scalar:ty) => {
        impl_array_scalar_concretization!(@checked $scalar, |value| {
            value.try_into().ok().and_then(|value| <$scalar>::new(value).ok())
        });
    };

    // Shared integer extraction rejects incompatible arrays before checking the target's range.
    (@checked $scalar:ty, |$value:ident| $convert:expr) => {
        impl Concretizable<$scalar> for Array {
            fn concretize(&self) -> Result<$scalar, ProgramError> {
                let $value: i128 = self.concretize()?;
                ($convert).ok_or_else(|| ProgramError::Concretization {
                    message: format!(
                        "cannot extract a concrete `{}` from `{}`; value `{}` is out of range",
                        stringify!($scalar),
                        self.r#type,
                        $value,
                    ),
                })
            }
        }
    };
}

impl_array_scalar_concretization!(@exact bool);
impl_array_scalar_concretization!(@integer i8);
impl_array_scalar_concretization!(@integer i16);
impl_array_scalar_concretization!(@integer i32);
impl_array_scalar_concretization!(@integer i64);
impl_array_scalar_concretization!(@integer isize);
impl_array_scalar_concretization!(@integer u8);
impl_array_scalar_concretization!(@integer u16);
impl_array_scalar_concretization!(@integer u32);
impl_array_scalar_concretization!(@integer u64);
impl_array_scalar_concretization!(@integer u128);
impl_array_scalar_concretization!(@integer usize);
impl_array_scalar_concretization!(@sub_byte i1);
impl_array_scalar_concretization!(@sub_byte i2);
impl_array_scalar_concretization!(@sub_byte i4);
impl_array_scalar_concretization!(@sub_byte u1);
impl_array_scalar_concretization!(@sub_byte u2);
impl_array_scalar_concretization!(@sub_byte u4);
impl_array_scalar_concretization!(@exact f4e2m1fn);
impl_array_scalar_concretization!(@exact f6e2m3fn);
impl_array_scalar_concretization!(@exact f6e3m2fn);
impl_array_scalar_concretization!(@exact f8e3m4);
impl_array_scalar_concretization!(@exact f8e4m3);
impl_array_scalar_concretization!(@exact f8e4m3b11fnuz);
impl_array_scalar_concretization!(@exact f8e4m3fn);
impl_array_scalar_concretization!(@exact f8e4m3fnuz);
impl_array_scalar_concretization!(@exact f8e5m2);
impl_array_scalar_concretization!(@exact f8e5m2fnuz);
impl_array_scalar_concretization!(@exact f8e8m0fnu);
impl_array_scalar_concretization!(@exact bf16);
impl_array_scalar_concretization!(@exact f16);
impl_array_scalar_concretization!(@exact f32);
impl_array_scalar_concretization!(@exact f64);
impl_array_scalar_concretization!(@exact Complex<f32>);
impl_array_scalar_concretization!(@exact Complex<f64>);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::sharding::meshes::{LogicalMesh, MeshAxis, MeshAxisType};
    use crate::arrays::sharding::shardings::Sharding;
    use crate::arrays::types::dimensions::{DimensionBounds, DimensionVariable};
    use crate::arrays::types::layouts::{Layout, StridedLayout, Tile, TileDimension, TiledLayout};
    use crate::arrays::types::memories::Memory;
    use crate::contexts::Context;
    use crate::tests::literal_hash_of;

    use super::*;

    /// Checks that integer values round-trip through typed and byte-based construction with exact little-endian
    /// encodings.
    macro_rules! check_integer_round_trip {
        ($data_type:expr, $element_type:ty, $values:expr $(,)?) => {{
            let values: &[$element_type] = &$values;
            let r#type = ArrayType::new_static($data_type, [values.len()]);
            let expected_bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
            let array = Array::from_elements(r#type.clone(), values).unwrap();
            assert_eq!(array.storage_bytes(), expected_bytes);
            assert_eq!(array.logical_bytes(), expected_bytes);
            assert_eq!(array.elements::<$element_type>(), Ok(values.to_vec()));
            assert_eq!(Array::new(r#type.clone(), expected_bytes.clone()).unwrap().elements(), Ok(values.to_vec()));
            assert_eq!(Array::from_logical_bytes(r#type, &expected_bytes).unwrap().elements(), Ok(values.to_vec()));
        }};
    }

    /// Checks sub-byte integer construction, exact encodings, and rejection of nonzero bits above the element width.
    macro_rules! check_sub_byte_integer {
        ($data_type:expr, $element_type:ty, $values:expr, $expected_bytes:expr, $invalid_byte:expr $(,)?) => {{
            let values: &[$element_type] = &$values;
            let expected_bytes: &[u8] = &$expected_bytes;
            let r#type = ArrayType::new_static($data_type, [values.len()]);

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
                Array::new(ArrayType::new_static($data_type, [1]), vec![$invalid_byte]),
                Err(ProgramError::Type(TypeError::Invalid { message }))
                    if message == format!(
                        "array element 0 has invalid `{}` byte encoding [{}]",
                        $data_type,
                        $invalid_byte,
                    ),
            ));
        }};
    }

    /// Checks that floating-point construction and decoding preserve the supplied bit patterns exactly.
    macro_rules! check_floating_point_round_trip {
        ($data_type:expr, $element_type:ty, $bit_type:ty, $bits:expr $(,)?) => {{
            let bits: &[$bit_type] = &$bits;
            let values = bits.iter().copied().map(<$element_type>::from_bits).collect::<Vec<_>>();
            let r#type = ArrayType::new_static($data_type, [values.len()]);
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

    /// Checks that low-precision floating-point encodings round-trip unchanged through typed and byte-based
    /// construction.
    macro_rules! check_low_precision_round_trip {
        ($data_type:expr, $element_type:ty, $bits:expr $(,)?) => {{
            let bits: &[u8] = &$bits;
            let r#type = ArrayType::new_static($data_type, [bits.len()]);
            let array = Array::from_logical_bytes(r#type.clone(), bits).unwrap();
            let elements = array.elements::<$element_type>().unwrap();
            assert_eq!(array.storage_bytes(), bits);
            assert_eq!(array.logical_bytes(), bits);
            assert_eq!(elements.iter().copied().map(<$element_type>::to_bits).collect::<Vec<_>>(), bits);
            assert_eq!(Array::from_elements(r#type.clone(), &elements).unwrap().storage_bytes(), bits);
            assert_eq!(Array::new(r#type, bits.to_vec()).unwrap().logical_bytes(), bits);
        }};
    }

    /// Checks that an integer source type's minimum and maximum concretize losslessly to `i128`.
    macro_rules! check_integer_concretization_source {
        ($source:ty, $minimum:expr, $maximum:expr $(,)?) => {{
            let minimum: Result<i128, ProgramError> = Array::scalar(<$source>::MIN).unwrap().concretize();
            let maximum: Result<i128, ProgramError> = Array::scalar(<$source>::MAX).unwrap().concretize();
            assert_eq!(minimum, Ok($minimum));
            assert_eq!(maximum, Ok($maximum));
        }};
    }

    /// Checks that the supplied integer values concretize to the target type's minimum and maximum.
    macro_rules! check_integer_concretization {
        ($target:ty, $minimum:expr, $maximum:expr $(,)?) => {{
            let minimum: Result<$target, _> = Array::scalar($minimum).unwrap().concretize();
            let maximum: Result<$target, _> = Array::scalar($maximum).unwrap().concretize();
            assert_eq!(minimum, Ok(<$target>::MIN));
            assert_eq!(maximum, Ok(<$target>::MAX));
        }};
    }

    /// Checks that out-of-range integer concretization returns the expected error and diagnostic.
    macro_rules! check_integer_concretization_out_of_range {
        ($target:ty, $value:expr, $message:literal $(,)?) => {
            assert!(matches!(
                Concretizable::<$target>::concretize(&Array::scalar($value).unwrap()),
                Err(ProgramError::Concretization { message }) if message == $message,
            ));
        };
    }

    /// Checks that floating-point scalar concretization preserves the supplied bit patterns exactly.
    macro_rules! check_floating_point_concretization {
        ($target:ty, $bits:expr $(,)?) => {
            for bits in $bits {
                let array = Array::new(ArrayType::scalar(<$target>::data_type()), bits.to_le_bytes().to_vec()).unwrap();
                let scalar: $target = array.concretize().unwrap();
                assert_eq!(scalar.to_bits(), bits);
            }
        };
    }

    #[test]
    fn test_array_new() {
        let array_type =
            ArrayType::new_static(DataType::U8, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![2])));
        let array = Array::new(array_type.clone(), vec![10, 0, 20]).unwrap();
        assert_eq!(array.r#type().as_ref(), &array_type);
        assert_eq!(array.elements::<u8>(), Ok(vec![10, 20]));
        assert_eq!(array.storage_bytes(), [10, 0, 20]);

        // Physical construction validates the full storage span, including layout holes, and every element encoding.
        assert!(matches!(
            Array::new(array_type.clone(), vec![10, 20]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array type `u8[2][layout=strided{2}]` requires 3 physical storage bytes but got 2",
        ));
        assert!(matches!(
            Array::new(array_type, vec![10, 1, 20]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array layout holes and tile padding must contain zero bytes",
        ));
        assert!(matches!(
            Array::new(ArrayType::scalar(DataType::Boolean), vec![2]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array element 0 has invalid `bool` byte encoding [2]",
        ));

        // Dynamically shaped types cannot describe materialized storage.
        let dynamic_type = ArrayType::new(
            DataType::U8,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        );
        assert!(matches!(
            Array::new(dynamic_type, Vec::new()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot materialize a value of dynamically sized type `u8[dynamic]`; dynamically \
                               shaped values exist only in array programs over `ArrayIrOperation`",
        ));
    }

    #[test]
    fn test_array_from_elements() {
        // Logical row-major elements are placed according to the layout, including reversed axes and holes.
        let array_type =
            ArrayType::new_static(DataType::U16, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-4])));
        let array = Array::from_elements(array_type.clone(), &[1u16, 256]).unwrap();
        assert_eq!(array.r#type().as_ref(), &array_type);
        assert_eq!(array.elements::<u16>(), Ok(vec![1, 256]));
        assert_eq!(array.storage_bytes(), [0, 1, 0, 0, 1, 0]);

        // Typed logical elements must match the declared element data type.
        assert!(matches!(
            Array::from_elements(ArrayType::new_static(DataType::F64, [2]), &[1.0f32, 2.0]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot encode `f32` values as array elements of data type `f64`",
        ));

        // The logical element count must match the static shape.
        assert!(matches!(
            Array::from_elements(ArrayType::new_static(DataType::F64, [3]), &[1.0f64]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array type `f64[3]` requires 3 logical elements but got 1",
        ));

        // Dynamically shaped types cannot describe materialized storage.
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        );
        assert!(matches!(
            Array::from_elements(dynamic_type, &[1.0f64]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot materialize a value of dynamically sized type `f64[dynamic]`; dynamically \
                               shaped values exist only in array programs over `ArrayIrOperation`",
        ));
    }

    #[test]
    fn test_array_from_elements_boolean_and_integer_encoding_round_trips() {
        // Booleans occupy one byte each, and integers use their exact little-endian encodings.
        let booleans = Array::from_elements(ArrayType::new_static(DataType::Boolean, [2]), &[false, true]).unwrap();
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
        check_integer_round_trip!(DataType::U64, u64, [0, (1u64 << 53) + 1, u64::MAX - 1, u64::MAX]);
    }

    #[test]
    fn test_array_from_elements_sub_byte_integer_encoding_round_trips() {
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
    fn test_array_from_elements_floating_point_encoding_round_trips() {
        // Signed zeros, infinities, and NaN payloads round-trip bit-exactly rather than being canonicalized.
        check_floating_point_round_trip!(DataType::BF16, bf16, u16, [0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc1]);
        check_floating_point_round_trip!(DataType::F16, f16, u16, [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e01]);
        check_floating_point_round_trip!(
            DataType::F32,
            f32,
            u32,
            [0x0000_0000, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_1234],
        );
        check_floating_point_round_trip!(
            DataType::F64,
            f64,
            u64,
            [
                0x0000_0000_0000_0000,
                0x8000_0000_0000_0000,
                0x7ff0_0000_0000_0000,
                0xfff0_0000_0000_0000,
                0x7ff8_0000_0000_1234,
            ],
        );

        // Low-precision encodings likewise preserve signed zeros, finite extremes, and any infinity or NaN encodings.
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
    fn test_array_from_elements_complex_encoding_round_trips() {
        // Complex elements store their real and imaginary components consecutively, and both components preserve
        // signed zeros, infinities, and NaN payloads bit-exactly.
        let complex64_components = [0x8000_0000u32, 0x7fc0_1234, 0x7f80_0000, 0xff80_0000];
        let complex64_values = [
            ComplexNumber::new(f32::from_bits(complex64_components[0]), f32::from_bits(complex64_components[1])),
            ComplexNumber::new(f32::from_bits(complex64_components[2]), f32::from_bits(complex64_components[3])),
        ];
        let complex64_type = ArrayType::new_static(DataType::C64, [2]);
        let complex64 = Array::from_elements(complex64_type.clone(), &complex64_values).unwrap();
        let expected_complex64_bytes = complex64_components.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        assert_eq!(complex64.storage_bytes(), expected_complex64_bytes);
        assert_eq!(
            Array::new(complex64_type.clone(), expected_complex64_bytes.clone()).unwrap().logical_bytes(),
            expected_complex64_bytes,
        );
        assert_eq!(
            Array::from_logical_bytes(complex64_type, &expected_complex64_bytes).unwrap().storage_bytes(),
            expected_complex64_bytes,
        );
        let decoded_complex64 = complex64.elements::<ComplexNumber<f32>>().unwrap();
        assert_eq!(
            decoded_complex64
                .iter()
                .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
                .collect::<Vec<_>>(),
            complex64_components,
        );

        let complex128_components =
            [0x8000_0000_0000_0000u64, 0x7ff8_0000_0000_1234, 0x7ff0_0000_0000_0000, 0xfff0_0000_0000_0000];
        let complex128_values = [
            ComplexNumber::new(f64::from_bits(complex128_components[0]), f64::from_bits(complex128_components[1])),
            ComplexNumber::new(f64::from_bits(complex128_components[2]), f64::from_bits(complex128_components[3])),
        ];
        let complex128_type = ArrayType::new_static(DataType::C128, [2]);
        let complex128 = Array::from_elements(complex128_type.clone(), &complex128_values).unwrap();
        let expected_complex128_bytes =
            complex128_components.into_iter().flat_map(u64::to_le_bytes).collect::<Vec<_>>();
        assert_eq!(complex128.storage_bytes(), expected_complex128_bytes);
        assert_eq!(
            Array::new(complex128_type.clone(), expected_complex128_bytes.clone()).unwrap().logical_bytes(),
            expected_complex128_bytes,
        );
        assert_eq!(
            Array::from_logical_bytes(complex128_type, &expected_complex128_bytes).unwrap().storage_bytes(),
            expected_complex128_bytes,
        );
        let decoded_complex128 = complex128.elements::<ComplexNumber<f64>>().unwrap();
        assert_eq!(
            decoded_complex128
                .iter()
                .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
                .collect::<Vec<_>>(),
            complex128_components,
        );
    }

    #[test]
    fn test_array_from_elements_empty_and_payload_free_encoding_round_trips() {
        // Empty arrays have no storage bytes and decode to no elements.
        let empty_type = ArrayType::new_static(DataType::F32, [0, 3]);
        let empty = Array::from_elements(empty_type.clone(), &[] as &[f32]).unwrap();
        assert!(empty.storage_bytes().is_empty());
        assert!(empty.logical_bytes().is_empty());
        assert_eq!(empty.elements::<f32>(), Ok(Vec::new()));
        assert_eq!(Array::new(empty_type.clone(), Vec::new()).unwrap().elements::<f32>(), Ok(Vec::new()));
        assert_eq!(Array::from_logical_bytes(empty_type, &[]).unwrap().elements::<f32>(), Ok(Vec::new()));

        // Payload-free element data types have logical elements but no storage bytes.
        for data_type in [DataType::Token, DataType::Zero] {
            let r#type = ArrayType::new_static(data_type, [3]);
            let array = Array::new(r#type.clone(), Vec::new()).unwrap();
            assert_eq!(array.r#type().as_ref(), &r#type);
            assert_eq!(Array::element_count(&r#type), Ok(3));
            assert!(array.storage_bytes().is_empty());
            assert!(array.logical_bytes().is_empty());
            assert!(Array::from_logical_bytes(r#type, &[]).unwrap().storage_bytes().is_empty());
        }
    }

    #[test]
    fn test_array_from_logical_bytes() {
        // Logical element encodings are placed according to the layout, with zero-filled holes and tile padding.
        let array_type =
            ArrayType::new_static(DataType::U8, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let array = Array::from_logical_bytes(array_type.clone(), &[10, 20]).unwrap();
        assert_eq!(array.r#type().as_ref(), &array_type);
        assert_eq!(array.storage_bytes(), [20, 0, 10]);
        assert_eq!(array.logical_bytes(), [10, 20]);
        let tiled_type = ArrayType::new_static(DataType::U8, [3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        let tiled = Array::from_logical_bytes(tiled_type, &[10, 20, 30]).unwrap();
        assert_eq!(tiled.storage_bytes(), [10, 20, 30, 0]);
        assert_eq!(tiled.logical_bytes(), [10, 20, 30]);

        // The byte count and every element encoding are validated.
        assert!(matches!(
            Array::from_logical_bytes(array_type, &[10]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array type `u8[2][layout=strided{-2}]` requires 2 logical element bytes but got 1",
        ));
        assert!(matches!(
            Array::from_logical_bytes(ArrayType::scalar(DataType::Boolean), &[2]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array element 0 has invalid `bool` byte encoding [2]",
        ));
    }

    #[test]
    fn test_array_scalar() {
        for value in [false, true] {
            let array = Array::scalar(value).unwrap();
            assert_eq!(array.r#type().as_ref(), &ArrayType::scalar(DataType::Boolean));
            assert_eq!(array.elements::<bool>(), Ok(vec![value]));
        }
        assert_eq!(Array::scalar(2.5).unwrap().r#type().as_ref(), &ArrayType::scalar(DataType::F64));
        assert_eq!(Array::scalar(2.5), Array::from_elements(ArrayType::scalar(DataType::F64), &[2.5]));
    }

    #[test]
    fn test_array_vector() {
        assert_eq!(
            Array::vector(vec![1.0f32, 2.0]),
            Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[1.0f32, 2.0]),
        );
        assert_eq!(
            Array::vector(vec![true, false]).unwrap().r#type().as_ref(),
            &ArrayType::new_static(DataType::Boolean, [2]),
        );
        assert_eq!(Array::vector(Vec::<f64>::new()).unwrap().elements::<f64>(), Ok(Vec::new()));
    }

    #[test]
    fn test_array_matrix() {
        assert_eq!(
            Array::matrix(2, 2, vec![1, 2, 3, 4]),
            Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2]), &[1, 2, 3, 4]),
        );
        assert_eq!(Array::matrix(0, 2, Vec::<i32>::new()).unwrap().elements::<i32>(), Ok(Vec::new()));

        // Invalid sizes are returned as errors before attempting to allocate output storage.
        let array_type = ArrayType::new_static(DataType::I32, [2, 2]);
        assert!(matches!(
            Array::matrix(2, 2, vec![1, 2, 3]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("array type `{array_type}` requires 4 logical elements but got 3"),
        ));
        let array_type = ArrayType::new_static(DataType::U8, [usize::MAX, 2]);
        assert!(matches!(
            Array::matrix(usize::MAX, 2, Vec::<u8>::new()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("array type `{array_type}` requires more bytes than can be represented"),
        ));
    }

    #[test]
    fn test_array_new_unchecked() {
        // Kernels hand off already validated storage without reallocating the shared payload.
        let array_type = ArrayType::new_static(DataType::U8, [2]);
        let storage = Arc::new(vec![10, 20]);
        let array = Array::new_unchecked(array_type.clone(), storage.clone());
        assert_eq!(array.r#type().as_ref(), &array_type);
        assert_eq!(array.elements::<u8>(), Ok(vec![10, 20]));
        assert!(Arc::ptr_eq(array.shared_storage_bytes(), &storage));
    }

    #[test]
    fn test_array_element_count() {
        assert_eq!(Array::element_count(&ArrayType::scalar(DataType::F32)), Ok(1));
        assert_eq!(Array::element_count(&ArrayType::new_static(DataType::F32, [2, 3])), Ok(6));

        // A statically zero dimension makes the count zero even when another dimension is dynamic.
        let variable = DimensionVariable::new("dynamic", DimensionBounds::unbounded());
        let empty_type = ArrayType::new(DataType::F32, Shape::new(vec![variable.clone().into(), Dimension::Static(0)]));
        assert_eq!(Array::element_count(&empty_type), Ok(0));

        // Counts that are not statically known or that do not fit in `usize` are rejected.
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![variable.into()]));
        assert!(matches!(
            Array::element_count(&dynamic_type),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot materialize a value of dynamically sized type `f32[dynamic]`",
        ));
        assert!(matches!(
            Array::element_count(&ArrayType::new_static(DataType::F32, [usize::MAX, 2])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("shape [{}, 2] element count does not fit in `usize`", usize::MAX),
        ));
    }

    #[test]
    fn test_array_elements() {
        // Decoding follows logical order through a reversed layout with holes and requires the matching element type.
        let array = Array::from_elements(
            ArrayType::new_static(DataType::U16, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-4]))),
            &[1u16, 256],
        )
        .unwrap();
        assert_eq!(array.elements::<u16>(), Ok(vec![1, 256]));
        assert!(matches!(
            array.elements::<i16>(),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot decode array elements of data type `u16` as `i16` values",
        ));
    }

    #[test]
    fn test_array_non_negative_integer_elements() {
        assert_eq!(
            Array::vector(vec![0i32, 2, 3]).unwrap().non_negative_integer_elements("indices"),
            Ok(vec![0, 2, 3]),
        );
        assert_eq!(
            Array::vector(vec![u4::MIN, u4::MAX]).unwrap().non_negative_integer_elements("indices"),
            Ok(vec![0, 15]),
        );

        // Negative entries are rejected after signed widening, including for sub-byte element types.
        assert!(matches!(
            Array::vector(vec![0i32, -1]).unwrap().non_negative_integer_elements("indices"),
            Err(ProgramError::InvalidArgument { message }) if message == "`indices[1]` must be non-negative but got -1",
        ));
        assert!(matches!(
            Array::vector(vec![i4::MIN]).unwrap().non_negative_integer_elements("indices"),
            Err(ProgramError::InvalidArgument { message }) if message == "`indices[0]` must be non-negative but got -8",
        ));

        // Unsigned values are widened before the host-size check, so their sign is never misreported.
        let actual = Array::vector(vec![u64::MAX]).unwrap().non_negative_integer_elements("indices");
        if usize::BITS == 64 {
            assert_eq!(actual, Ok(vec![usize::MAX]));
        } else {
            assert!(matches!(
                actual,
                Err(ProgramError::InvalidArgument { message })
                    if message == "`indices[0]` value 18446744073709551615 does not fit in `usize`",
            ));
        }
    }

    #[test]
    fn test_array_to_f64s() {
        // Floating-point values, Booleans, and integers (including sub-byte integers) convert to their exact values.
        assert_eq!(Array::vector(vec![1.5, 2.5]).unwrap().to_f64s(), vec![1.5, 2.5]);
        assert_eq!(Array::vector(vec![true, false]).unwrap().to_f64s(), vec![1.0, 0.0]);
        assert_eq!(Array::vector(vec![1i32, -2]).unwrap().to_f64s(), vec![1.0, -2.0]);
        assert_eq!(
            Array::from_elements(
                ArrayType::new_static(DataType::I4, [2]),
                &[i4::new(-8).unwrap(), i4::new(7).unwrap()],
            )
            .unwrap()
            .to_f64s(),
            vec![-8.0, 7.0],
        );

        // Low-precision floating-point elements decode to the exact values they denote.
        assert_eq!(Array::vector(vec![f8e4m3fn::from_f64(1.5).unwrap()]).unwrap().to_f64s(), vec![1.5]);
    }

    #[test]
    #[should_panic(expected = "cannot view an array of complex element data type `c128` as `f64` values")]
    fn test_array_to_f64s_rejects_complex_arrays() {
        Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap().to_f64s();
    }

    #[test]
    #[should_panic(expected = "cannot view an array of element data type `token` as `f64` values")]
    fn test_array_to_f64s_rejects_payload_free_arrays() {
        Array::new(ArrayType::scalar(DataType::Token), Vec::new()).unwrap().to_f64s();
    }

    #[test]
    fn test_array_logical_bytes() {
        let array = Array::from_elements(
            ArrayType::new_static(DataType::U8, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2]))),
            &[10u8, 20],
        )
        .unwrap();
        assert_eq!(array.logical_bytes(), [10, 20]);
    }

    #[test]
    fn test_array_storage_bytes() {
        let array = Array::from_elements(
            ArrayType::new_static(DataType::U8, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2]))),
            &[10u8, 20],
        )
        .unwrap();
        assert_eq!(array.storage_bytes(), [20, 0, 10]);
    }

    #[test]
    fn test_array_storage_bytes_mut() {
        // Mutating shared storage copies it first, leaving the other array unchanged.
        let original = Array::vector(vec![10u8, 20]).unwrap();
        let mut changed = original.clone();
        changed.storage_bytes_mut()[0] = 30;
        assert_eq!(changed.elements::<u8>(), Ok(vec![30, 20]));
        assert_eq!(original.elements::<u8>(), Ok(vec![10, 20]));
        assert!(!Arc::ptr_eq(changed.shared_storage_bytes(), original.shared_storage_bytes()));

        // Uniquely owned storage is mutated in place.
        let storage = Arc::as_ptr(changed.shared_storage_bytes());
        changed.storage_bytes_mut()[1] = 40;
        assert_eq!(changed.elements::<u8>(), Ok(vec![30, 40]));
        assert!(std::ptr::eq(Arc::as_ptr(changed.shared_storage_bytes()), storage));
    }

    #[test]
    fn test_array_shared_storage_bytes() {
        let original = Array::vector(vec![10u8, 20]).unwrap();
        let cloned = original.clone();
        assert_eq!(original.shared_storage_bytes().as_slice(), &[10, 20]);
        assert!(Arc::ptr_eq(original.shared_storage_bytes(), cloned.shared_storage_bytes()));
    }

    #[test]
    fn test_array_converted_to() {
        // Converting to the same element data type shares the existing payload.
        let array_type =
            ArrayType::new_static(DataType::I32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-8])));
        let input = Array::from_elements(array_type, &[1i32, -2]).unwrap();
        let unchanged = input.converted_to(DataType::I32).unwrap();
        assert_eq!(unchanged, input);
        assert!(Arc::ptr_eq(unchanged.shared_storage_bytes(), input.shared_storage_bytes()));

        // Byte-stride layouts are cleared when the element storage width changes and are retained otherwise.
        let widened = input.converted_to(DataType::I64).unwrap();
        assert_eq!(widened.r#type().as_ref(), &ArrayType::new_static(DataType::I64, [2]));
        assert_eq!(widened.elements::<i64>(), Ok(vec![1, -2]));
        let same_width = input.converted_to(DataType::F32).unwrap();
        assert_eq!(same_width.r#type().layout(), input.r#type().layout());
        assert_eq!(same_width.elements::<f32>(), Ok(vec![1.0, -2.0]));

        // Token conversions are always rejected, while structural-zero conversions are accepted only as same-type
        // no-ops.
        assert!(matches!(
            input.converted_to(DataType::Token),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `token` data type",
        ));
        assert!(matches!(
            input.converted_to(DataType::Zero),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `zero` data type",
        ));
        let zero = Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap();
        assert_eq!(zero.converted_to(DataType::Zero), Ok(zero.clone()));
        assert!(matches!(
            zero.converted_to(DataType::F32),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `zero` data type",
        ));
        let token = Array::new(ArrayType::scalar(DataType::Token), Vec::new()).unwrap();
        assert!(matches!(
            token.converted_to(DataType::Token),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `token` data type",
        ));

        // Conversion retains tiled placement, sharding, and memory even when the element storage width changes.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1);
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])]));
        let input_type = ArrayType::new_static(DataType::I32, [3])
            .with_layout(layout.clone())
            .with_sharding(sharding.clone())
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        let input = Array::from_elements(input_type, &[1i32, -2, 3]).unwrap();
        let expected_type = ArrayType::new_static(DataType::I64, [3])
            .with_layout(layout)
            .with_sharding(sharding)
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(input.converted_to(DataType::I64), Array::from_elements(expected_type, &[1i64, -2, 3]));
    }

    #[test]
    fn test_array_promoted_to() {
        // Arrays that already have the requested element data type are borrowed unchanged, while others are converted.
        let array = Array::vector(vec![1i32, -2]).unwrap();
        assert!(matches!(array.promoted_to(DataType::I32), Ok(Cow::Borrowed(value)) if std::ptr::eq(value, &array)));
        let promoted = array.promoted_to(DataType::I64).unwrap();
        assert!(matches!(promoted, Cow::Owned(_)));
        assert_eq!(promoted.elements::<i64>(), Ok(vec![1, -2]));

        // Conversion errors propagate.
        assert!(matches!(
            array.promoted_to(DataType::Token),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `token` data type",
        ));
    }

    #[test]
    fn test_array_map_elements() {
        // Mapping traverses logical elements while retaining a caller-selected physical layout.
        let input_type =
            ArrayType::new_static(DataType::F64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-16])));
        let input = Array::from_elements(input_type.clone(), &[0.0f64, 1.0]).unwrap();
        let output = input.map_elements::<f64, f64>(input_type.clone(), |value| Ok(value + 1.0)).unwrap();
        assert_eq!(output.r#type().as_ref(), &input_type);
        assert_eq!(output.elements::<f64>(), Ok(vec![1.0, 2.0]));

        // Input and output element types may differ.
        let integers = Array::vector(vec![1i32, -2, 3]).unwrap();
        let doubled = integers.map_elements::<i32, i32>(integers.r#type().into_owned(), |value| Ok(value * 2)).unwrap();
        assert_eq!(doubled.elements::<i32>(), Ok(vec![2, -4, 6]));
        let negative = integers
            .map_elements::<i32, bool>(ArrayType::new_static(DataType::Boolean, [3]), |value| Ok(value < 0))
            .unwrap();
        assert_eq!(negative.storage_bytes(), [0, 1, 0]);

        // Callers must supply the actual input codec, the actual output codec, and a matching element count.
        assert!(matches!(
            integers.map_elements::<i64, i64>(ArrayType::new_static(DataType::I64, [3]), Ok),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot map elements of data type `i32` as `i64` values",
        ));
        assert!(matches!(
            integers.map_elements::<i32, bool>(ArrayType::new_static(DataType::I32, [3]), |_| Ok(true)),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot store mapped `bool` values in an array of element data type `i32`",
        ));
        assert!(matches!(
            integers.map_elements::<i32, i32>(ArrayType::new_static(DataType::I32, [2]), Ok),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot map 3 logical elements onto array type `i32[2]` with 2 logical elements",
        ));

        // Element function errors propagate unchanged.
        assert!(matches!(
            integers.map_elements::<i32, i32>(integers.r#type().into_owned(), |_| {
                Err(ProgramError::InvalidArgument { message: "scalar mapping failed".into() })
            }),
            Err(ProgramError::InvalidArgument { message }) if message == "scalar mapping failed",
        ));

        // Empty traversal returns an empty output without evaluating the element function.
        let empty = Array::vector(Vec::<i32>::new()).unwrap();
        assert_eq!(
            empty.map_elements::<i32, i32>(empty.r#type().into_owned(), |_| panic!("empty traversal")),
            Ok(empty),
        );
    }

    #[test]
    fn test_array_map_element_pairs() {
        let column = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 1]), &[1i32, 3]).unwrap();
        let row = Array::vector(vec![0i32, 2, 4]).unwrap();

        // Broadcasting preserves the input element type while allowing a different output element type.
        let comparisons = column
            .map_element_pairs::<i32, bool>(&row, ArrayType::new_static(DataType::Boolean, [2, 3]), |left, right| {
                Ok(left < right)
            })
            .unwrap();
        assert_eq!(comparisons.r#type().as_ref(), &ArrayType::new_static(DataType::Boolean, [2, 3]));
        assert_eq!(comparisons.elements::<bool>(), Ok(vec![false, true, true, false, false, true]));

        // Broadcasting reads negative-stride inputs with holes in logical order.
        let strided_column = Array::from_elements(
            ArrayType::new_static(DataType::F64, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-16, 8]))),
            &[0.0f64, 1.0],
        )
        .unwrap();
        let strided_row = Array::from_elements(
            ArrayType::new_static(DataType::F64, [1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![24, -8]))),
            &[1.0f64, 1.0, -1.0],
        )
        .unwrap();
        assert_eq!(
            strided_column.map_element_pairs::<f64, f64>(
                &strided_row,
                ArrayType::new_static(DataType::F64, [2, 3]),
                |left, right| Ok(left + right),
            ),
            Ok(Array::matrix(2, 3, vec![1.0f64, 1.0, -1.0, 2.0, 2.0, 0.0]).unwrap()),
        );

        // Logical broadcast indexing also supports an independently laid-out output, including reversed axes and
        // holes between rows. Addressing maps each logical input and output index to its own storage.
        let output_type = ArrayType::new_static(DataType::F64, [2, 3])
            .with_layout(Layout::Strided(StridedLayout::new(vec![-40, -8])));
        let output = strided_column
            .map_element_pairs::<f64, f64>(&strided_row, output_type.clone(), |left, right| Ok(left + right))
            .unwrap();
        let expected = Array::from_elements(output_type, &[1.0f64, 1.0, -1.0, 2.0, 2.0, 0.0]).unwrap();
        assert_eq!(output, expected);
        assert_eq!(output.storage_bytes(), expected.storage_bytes());

        // Callers must supply the actual input codec, the actual output codec, and the broadcast shape.
        assert!(matches!(
            column.map_element_pairs::<i64, bool>(
                &row,
                ArrayType::new_static(DataType::Boolean, [2, 3]),
                |left, right| Ok(left < right),
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "binary element inputs must both have data type `i64`, got `i32` and `i32`",
        ));
        assert!(matches!(
            column.map_element_pairs::<i32, bool>(
                &row,
                ArrayType::new_static(DataType::I32, [2, 3]),
                |left, right| Ok(left < right),
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "binary element output must have data type `bool`, got `i32`",
        ));
        assert!(matches!(
            column.map_element_pairs::<i32, bool>(
                &row,
                ArrayType::new_static(DataType::Boolean, [3, 2]),
                |left, right| Ok(left < right),
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "binary element output shape must be [2, 3], got [3, 2]",
        ));

        // Input shapes that do not broadcast are rejected.
        assert!(matches!(
            Array::vector(vec![1i32, 2]).unwrap().map_element_pairs::<i32, i32>(
                &row,
                ArrayType::new_static(DataType::I32, [3]),
                |left, right| Ok(left + right),
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "failed to broadcast shape `[2]` to shape `[3]`",
        ));

        // Element function errors propagate unchanged.
        assert!(matches!(
            column.map_element_pairs::<i32, i32>(&row, ArrayType::new_static(DataType::I32, [2, 3]), |_, _| {
                Err(ProgramError::InvalidArgument { message: "scalar pair mapping failed".into() })
            }),
            Err(ProgramError::InvalidArgument { message }) if message == "scalar pair mapping failed",
        ));

        // Empty outputs do not invoke the element function.
        let empty = Array::vector(Vec::<i32>::new()).unwrap();
        assert_eq!(
            empty.map_element_pairs::<i32, i32>(
                &Array::scalar(1i32).unwrap(),
                empty.r#type().into_owned(),
                |_, _| panic!("empty traversal"),
            ),
            Ok(empty),
        );
    }

    #[test]
    fn test_array_from_fn_elements() {
        // `from_fn_elements` constructs an array from its flat logical row-major element indices.
        let iota =
            Array::from_fn_elements(ArrayType::new_static(DataType::U16, [2, 2]), |index| Ok(index as u16)).unwrap();
        assert_eq!(iota.elements::<u16>(), Ok(vec![0, 1, 2, 3]));

        // The element type must match the array type, and element function errors propagate unchanged.
        assert!(matches!(
            Array::from_fn_elements(ArrayType::new_static(DataType::U16, [1]), |_| Ok(0u32)),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot store `u32` values in an array of element data type `u16`",
        ));
        assert!(matches!(
            Array::from_fn_elements(ArrayType::new_static(DataType::U16, [1]), |_| {
                Err::<u16, _>(ProgramError::InvalidArgument { message: "element construction failed".into() })
            }),
            Err(ProgramError::InvalidArgument { message }) if message == "element construction failed",
        ));

        // Empty arrays do not invoke the element function.
        let empty = Array::vector(Vec::<u16>::new()).unwrap();
        assert_eq!(
            Array::from_fn_elements::<u16, _>(empty.r#type().into_owned(), |_| panic!("empty construction")),
            Ok(empty),
        );
    }

    #[test]
    fn test_array_gather_elements() {
        // `gather_elements` copies whole element encodings through a flat index mapping without decoding them, so it
        // serves reversal, repetition, and selection over any element data type, including sub-byte ones.
        let integers = Array::vector(vec![1i32, -2, 3]).unwrap();
        let reversed =
            integers.gather_elements(integers.r#type().into_owned(), |output_index| 2 - output_index).unwrap();
        assert_eq!(reversed.elements::<i32>(), Ok(vec![3, -2, 1]));
        let repeated = integers.gather_elements(ArrayType::new_static(DataType::I32, [4]), |_| 1).unwrap();
        assert_eq!(repeated.elements::<i32>(), Ok(vec![-2, -2, -2, -2]));
        let narrow = Array::from_elements(
            ArrayType::new_static(DataType::I4, [2]),
            &[i4::new(-8).unwrap(), i4::new(7).unwrap()],
        )
        .unwrap();
        let swapped = narrow.gather_elements(narrow.r#type().into_owned(), |output_index| 1 - output_index).unwrap();
        assert_eq!(swapped.storage_bytes(), [0x07, 0x08]);

        // Input and output elements are addressed through their own layouts.
        let reversed_input = Array::from_elements(
            ArrayType::new_static(DataType::U16, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![-2]))),
            &[1u16, 2, 3],
        )
        .unwrap();
        let output_type =
            ArrayType::new_static(DataType::U16, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let selected = reversed_input.gather_elements(output_type, |output_index| 2 * output_index).unwrap();
        assert_eq!(selected.elements::<u16>(), Ok(vec![1, 3]));
        assert_eq!(selected.storage_bytes(), [1, 0, 0, 0, 3, 0]);

        // Element data types must match, and every gathered index must lie in bounds.
        assert!(matches!(
            integers.gather_elements(ArrayType::new_static(DataType::I64, [3]), |output_index| output_index),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot gather elements of data type `i32` into an array of element data type `i64`",
        ));
        assert!(matches!(
            integers.gather_elements(ArrayType::new_static(DataType::I32, [3]), |_| 3),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "gather index 3 is out of bounds for 3 elements",
        ));

        // Empty outputs do not invoke the index mapping.
        let empty = Array::vector(Vec::<i32>::new()).unwrap();
        assert_eq!(empty.gather_elements(empty.r#type().into_owned(), |_| panic!("empty gather")), Ok(empty));
    }

    #[test]
    fn test_array_broadcast_index() {
        // Input axes with extent one always select coordinate zero.
        let output = StaticShape::new(vec![2, 3]);
        let input = StaticShape::new(vec![2, 1]);
        assert_eq!(
            (0..6)
                .map(|index| Array::broadcast_index(index, &output, &[3, 1], &input, &[1, 1]))
                .collect::<Vec<_>>(),
            vec![0, 0, 0, 1, 1, 1],
        );

        // Right alignment reuses a vector across rows; rank-zero inputs always select their sole element.
        let vector = StaticShape::new(vec![3]);
        assert_eq!(
            (0..6)
                .map(|index| Array::broadcast_index(index, &output, &[3, 1], &vector, &[1]))
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 0, 1, 2],
        );
        assert_eq!(Array::broadcast_index(5, &output, &[3, 1], &StaticShape::new(vec![]), &[]), 0);
    }

    #[test]
    fn test_array_copy_block() {
        // A reversed source and an output with holes exercise both sides of logical byte traversal.
        let input_type =
            ArrayType::new_static(DataType::U16, [4]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let input = Array::from_elements(input_type, &[1u16, 2, 3, 4]).unwrap();
        let output_type =
            ArrayType::new_static(DataType::U16, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let output = input.copy_block(output_type.clone(), &[ArraySliceAxis::new(1, 2, 1)]).unwrap();
        assert_eq!(output.r#type().as_ref(), &output_type);
        assert_eq!(output.elements::<u16>(), Ok(vec![2, 3]));
        assert_eq!(output.storage_bytes(), [2, 0, 0, 0, 3, 0]);

        // Dense outputs take the contiguous-copy path while respecting the source's logical order.
        assert_eq!(
            input.copy_block(ArrayType::new_static(DataType::U16, [2]), &[ArraySliceAxis::new(1, 2, 1)]),
            Ok(Array::vector(vec![2u16, 3]).unwrap()),
        );

        // Dense sources copy whole contiguous runs per selected row, and strided selections skip coordinates.
        assert_eq!(
            Array::matrix(2, 3, vec![1u16, 2, 3, 4, 5, 6]).unwrap().copy_block(
                ArrayType::new_static(DataType::U16, [2, 2]),
                &[ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(1, 2, 1)],
            ),
            Ok(Array::matrix(2, 2, vec![2u16, 3, 5, 6]).unwrap()),
        );
        assert_eq!(
            Array::vector(vec![1u16, 2, 3, 4])
                .unwrap()
                .copy_block(ArrayType::new_static(DataType::U16, [2]), &[ArraySliceAxis::new(0, 2, 2)]),
            Ok(Array::vector(vec![1u16, 3]).unwrap()),
        );
    }

    #[test]
    fn test_array_replace_block() {
        // Copying a block preserves the destination layout and leaves the shared original storage untouched.
        let input_type =
            ArrayType::new_static(DataType::U16, [4]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let input = Array::from_elements(input_type.clone(), &[1u16, 2, 3, 4]).unwrap();
        let update_type =
            ArrayType::new_static(DataType::U16, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let update = Array::from_elements(update_type, &[8u16, 9]).unwrap();
        let output = input.clone().replace_block(&update, &[1]);
        assert_eq!(output.r#type().as_ref(), &input_type);
        assert_eq!(output.elements::<u16>(), Ok(vec![1, 8, 9, 4]));
        assert_eq!(output.storage_bytes(), [4, 0, 9, 0, 8, 0, 1, 0]);
        assert_eq!(input.elements::<u16>(), Ok(vec![1, 2, 3, 4]));

        // Dense updates take the contiguous-copy path even when the destination has reversed storage.
        assert_eq!(
            input.clone().replace_block(&Array::vector(vec![8u16, 9]).unwrap(), &[1]).elements::<u16>(),
            Ok(vec![1, 8, 9, 4]),
        );
        assert_eq!(input.elements::<u16>(), Ok(vec![1, 2, 3, 4]));

        // Multi-dimensional blocks overwrite one contiguous run per selected row of a dense destination.
        assert_eq!(
            Array::matrix(2, 3, vec![0u16; 6])
                .unwrap()
                .replace_block(&Array::matrix(2, 2, vec![1u16, 2, 3, 4]).unwrap(), &[0, 1]),
            Array::matrix(2, 3, vec![0u16, 1, 2, 0, 3, 4]).unwrap(),
        );
    }

    #[test]
    fn test_array_debug() {
        // Debug exposes the complete type and the logical payload, including sub-byte element values.
        let array = Array::vector(vec![i4::MIN, i4::MAX]).unwrap();
        assert_eq!(
            format!("{array:?}"),
            concat!(
                "Array { type: ArrayType { data_type: I4, shape: Shape { dimensions: [Static(2)] }, ",
                "layout: None, sharding: None, memory: Device }, values: [-8, 7] }",
            ),
        );
    }

    #[test]
    fn test_array_display() {
        // Rank zero renders as a scalar and each higher rank contributes one bracketed level in logical row-major
        // order. Real floating-point payloads keep a decimal point, while other payloads use scalar rendering.
        assert_eq!(Array::scalar(1.0).unwrap().to_string(), "1.0");
        assert_eq!(Array::vector(vec![1.0, 2.5]).unwrap().to_string(), "[1.0, 2.5]");
        assert_eq!(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap().to_string(), "[[1.0, 2.0], [3.0, 4.0]]");
        assert_eq!(
            Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 2]), &[1.0, 2.0, 3.0, 4.0])
                .unwrap()
                .to_string(),
            "[[[1.0, 2.0]], [[3.0, 4.0]]]",
        );
        assert_eq!(Array::vector(vec![1i32, 2]).unwrap().to_string(), "[1, 2]");
        assert_eq!(Array::vector(vec![true, false]).unwrap().to_string(), "[true, false]");
        assert_eq!(Array::vector(vec![ComplexNumber::new(1.0f64, 2.0)]).unwrap().to_string(), "[1+2i]");

        // Empty dimensions render as empty brackets, and payload-free arrays render their data type at every position.
        assert_eq!(Array::vector(Vec::<f64>::new()).unwrap().to_string(), "[]");
        assert_eq!(
            Array::from_elements(ArrayType::new_static(DataType::F64, [2, 0]), &[] as &[f64])
                .unwrap()
                .to_string(),
            "[[], []]",
        );
        assert_eq!(Array::new(ArrayType::scalar(DataType::Token), Vec::new()).unwrap().to_string(), "token");
        assert_eq!(
            Array::new(ArrayType::new_static(DataType::Zero, [2, 1]), Vec::new()).unwrap().to_string(),
            "[[zero], [zero]]",
        );

        // Rendering follows logical coordinates rather than physical storage order.
        let column_major =
            ArrayType::new_static(DataType::F64, [2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![8, 16])));
        assert_eq!(
            Array::from_elements(column_major, &[1.0, 2.0, 3.0, 4.0]).unwrap().to_string(),
            "[[1.0, 2.0], [3.0, 4.0]]",
        );

        // Sub-byte integer elements render their numeric values rather than their storage bytes.
        let narrow = Array::from_elements(
            ArrayType::new_static(DataType::I4, [2]),
            &[i4::new(-8).unwrap(), i4::new(7).unwrap()],
        )
        .unwrap();
        assert_eq!(narrow.to_string(), "[-8, 7]");
    }

    #[test]
    fn test_array_eq() {
        // Exact equality requires identical types and elementwise-equal payloads.
        assert_eq!(Array::vector(vec![1.0, 2.0]).unwrap(), Array::vector(vec![1.0, 2.0]).unwrap());
        assert_ne!(Array::vector(vec![1.0, 2.0]).unwrap(), Array::vector(vec![1.0, 2.5]).unwrap());
        assert_ne!(Array::vector(vec![1.0f32]).unwrap(), Array::vector(vec![1.0f64]).unwrap());

        // Equality decodes typed values rather than comparing physical bytes, so signed zeros compare equal (including
        // for low-precision floating-point elements) while NaNs compare unequal even to themselves.
        let positive_zero = Array::vector(vec![0.0f32]).unwrap();
        let negative_zero = Array::vector(vec![-0.0f32]).unwrap();
        assert_ne!(positive_zero.storage_bytes(), negative_zero.storage_bytes());
        assert_eq!(positive_zero, negative_zero);
        let positive_zero = Array::vector(vec![f8e4m3fn::from_f64(0.0).unwrap()]).unwrap();
        let negative_zero = Array::vector(vec![f8e4m3fn::from_f64(-0.0).unwrap()]).unwrap();
        assert_ne!(positive_zero.storage_bytes(), negative_zero.storage_bytes());
        assert_eq!(positive_zero, negative_zero);
        let nan = Array::vector(vec![f32::from_bits(0x7fc0_1234)]).unwrap();
        assert_ne!(nan, nan.clone());

        // Shape and physical-layout metadata are part of the array type, even when values agree.
        assert_ne!(Array::scalar(1i32).unwrap(), Array::vector(vec![1i32]).unwrap());
        let dense = Array::vector(vec![1i32, 2]).unwrap();
        let reversed_type =
            ArrayType::new_static(DataType::I32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-4])));
        assert_ne!(dense, Array::from_elements(reversed_type, &[1i32, 2]).unwrap());

        // Payload-free values compare by type alone.
        assert_eq!(
            Array::new(ArrayType::scalar(DataType::Token), Vec::new()).unwrap(),
            Array::new(ArrayType::scalar(DataType::Token), Vec::new()).unwrap(),
        );
        assert_eq!(
            Array::new(ArrayType::new_static(DataType::Zero, [2]), Vec::new()).unwrap(),
            Array::new(ArrayType::new_static(DataType::Zero, [2]), Vec::new()).unwrap(),
        );
        assert_ne!(
            Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap(),
            Array::new(ArrayType::new_static(DataType::Zero, [2]), Vec::new()).unwrap(),
        );

        // Integer equality preserves values beyond `f64` precision.
        assert_ne!(Array::scalar(u64::MAX).unwrap(), Array::scalar(u64::MAX - 1).unwrap());
    }

    #[test]
    fn test_array_type() {
        let array = Array::vector(vec![1i32, 2]).unwrap();
        let r#type = array.r#type();
        assert!(matches!(r#type, Cow::Borrowed(_)));
        assert_eq!(r#type.as_ref(), &ArrayType::new_static(DataType::I32, [2]));
    }

    #[test]
    fn test_array_domain() {
        // An array's domain is the reference backend's rich eager domain, which can bind array operations.
        let context: EagerContext<Array, ArrayOperation<Array>> = Array::scalar(1i32).unwrap().domain();
        assert!(context.is_eager());
    }

    #[test]
    fn test_array_is_zero() {
        // Both signs of floating-point zero represent the zero vector; NaNs and non-zero elements do not.
        assert!(Array::vector(vec![0.0f32, -0.0]).unwrap().is_zero());
        assert!(!Array::vector(vec![0.0f32, 1.0]).unwrap().is_zero());
        assert!(!Array::scalar(f32::NAN).unwrap().is_zero());

        // Empty arrays are vacuously zero, and integer and complex elements are zero only when they equal zero.
        assert!(Array::vector(Vec::<f32>::new()).unwrap().is_zero());
        assert!(Array::scalar(0i64).unwrap().is_zero());
        assert!(!Array::scalar(1u64).unwrap().is_zero());
        assert!(Array::scalar(ComplexNumber::new(-0.0f32, 0.0)).unwrap().is_zero());
        assert!(!Array::scalar(ComplexNumber::new(0.0f32, 1.0)).unwrap().is_zero());

        // The zero differential space has a unique zero value, while an effect token is not a numerical zero.
        assert!(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap().is_zero());
        assert!(!Array::new(ArrayType::scalar(DataType::Token), Vec::new()).unwrap().is_zero());
    }

    #[test]
    fn test_array_literal_eq() {
        // An array is literally identical to its clone and to an independently constructed array with the same type
        // and storage.
        let array = Array::vector(vec![1.0f32, -2.5]).unwrap();
        let same_array = Array::vector(vec![1.0f32, -2.5]).unwrap();
        assert!(array.literal_eq(&array.clone()));
        assert!(array.literal_eq(&same_array));
        assert!(!array.literal_eq(&Array::vector(vec![1.0f32, 2.5]).unwrap()));

        // Identical storage bytes under different types are distinct literals.
        let float = Array::scalar(1.0f32).unwrap();
        let integer = Array::new(ArrayType::scalar(DataType::I32), 1.0f32.to_le_bytes().to_vec()).unwrap();
        assert_eq!(float.storage_bytes(), integer.storage_bytes());
        assert!(!float.literal_eq(&integer));

        // Literal identity is bitwise, unlike the IEEE value equality of `Array::eq`: signed zeros are numerically
        // equal but distinct literals.
        let negative_zero = Array::scalar(-0.0f64).unwrap();
        let positive_zero = Array::scalar(0.0f64).unwrap();
        assert_eq!(negative_zero, positive_zero);
        assert!(!negative_zero.literal_eq(&positive_zero));

        // A NaN is numerically unequal to itself but is the same literal as any NaN with identical bits.
        let nan = Array::scalar(f64::NAN).unwrap();
        let same_nan = Array::scalar(f64::NAN).unwrap();
        assert_ne!(nan, nan.clone());
        assert!(nan.literal_eq(&nan));
        assert!(nan.literal_eq(&same_nan));
        assert!(!nan.literal_eq(&Array::scalar(-f64::NAN).unwrap()));
        let payload_nan = Array::scalar(f64::from_bits(0x7ff8_0000_0000_1234)).unwrap();
        let different_payload_nan = Array::scalar(f64::from_bits(0x7ff8_0000_0000_5678)).unwrap();
        assert!(!payload_nan.literal_eq(&different_payload_nan));
    }

    #[test]
    fn test_array_literal_hash() {
        // Literal identity implies equal hashes for both shared and independently constructed storage.
        let array = Array::vector(vec![1.0f32, -2.5]).unwrap();
        assert_eq!(literal_hash_of(&array), literal_hash_of(&array.clone()));
        assert_eq!(literal_hash_of(&array), literal_hash_of(&Array::vector(vec![1.0f32, -2.5]).unwrap()));
        let nan = Array::scalar(f64::from_bits(0x7ff8_0000_0000_1234)).unwrap();
        let same_nan = Array::scalar(f64::from_bits(0x7ff8_0000_0000_1234)).unwrap();
        assert_eq!(literal_hash_of(&nan), literal_hash_of(&same_nan));
    }

    #[test]
    fn test_array_try_from() {
        assert_eq!(Array::try_from(false), Array::scalar(false));
        assert_eq!(Array::try_from(true), Array::scalar(true));
    }

    #[test]
    fn test_array_abs_diff_eq() {
        // Approximate equality compares elementwise within the absolute tolerance.
        assert_eq!(Array::default_epsilon(), f64::EPSILON);
        assert_abs_diff_eq!(
            Array::vector(vec![1.0, 2.0]).unwrap(),
            Array::vector(vec![1.0 + 1e-10, 2.0]).unwrap(),
            epsilon = 1e-9,
        );
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0)]).unwrap();
        let right = Array::vector(vec![ComplexNumber::new(1.0f64 + 1e-10, 2.0)]).unwrap();
        assert_abs_diff_eq!(left, right, epsilon = 1e-9);

        // Approximate equality reads low-precision, arbitrarily laid-out values directly from physical storage.
        let r#type =
            ArrayType::new_static(DataType::F8E4M3FN, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
        let left =
            Array::from_elements(r#type.clone(), &[f8e4m3fn::from_f64(1.0).unwrap(), f8e4m3fn::from_f64(2.0).unwrap()])
                .unwrap();
        let right =
            Array::from_elements(r#type, &[f8e4m3fn::from_f64(1.125).unwrap(), f8e4m3fn::from_f64(2.0).unwrap()])
                .unwrap();
        assert_abs_diff_eq!(left, right, epsilon = 0.2);

        // Real values and both complex components must be within tolerance, and NaNs are never approximately equal.
        assert!(!Array::scalar(1.0f64).unwrap().abs_diff_eq(&Array::scalar(1.5f64).unwrap(), 0.1));
        assert!(
            !Array::scalar(ComplexNumber::new(1.0f64, 2.0))
                .unwrap()
                .abs_diff_eq(&Array::scalar(ComplexNumber::new(1.0f64, 2.5)).unwrap(), 0.1)
        );
        assert!(!Array::scalar(f64::NAN).unwrap().abs_diff_eq(&Array::scalar(f64::NAN).unwrap(), 1.0));

        // Array types must match exactly, and the exact fallback never rounds integer payloads.
        assert!(!Array::scalar(1.0f32).unwrap().abs_diff_eq(&Array::scalar(1.0f64).unwrap(), 1.0));
        assert!(Array::scalar(u64::MAX).unwrap().abs_diff_eq(&Array::scalar(u64::MAX).unwrap(), 1.0));
        assert!(!Array::scalar(u64::MAX).unwrap().abs_diff_eq(&Array::scalar(u64::MAX - 1).unwrap(), 1.0));
    }

    #[test]
    fn test_array_concretize() {
        // Every source representation widens losslessly, including its signed or unsigned boundary values.
        check_integer_concretization_source!(i1, -1, 0);
        check_integer_concretization_source!(i2, -2, 1);
        check_integer_concretization_source!(i4, -8, 7);
        check_integer_concretization_source!(i8, i128::from(i8::MIN), i128::from(i8::MAX));
        check_integer_concretization_source!(i16, i128::from(i16::MIN), i128::from(i16::MAX));
        check_integer_concretization_source!(i32, i128::from(i32::MIN), i128::from(i32::MAX));
        check_integer_concretization_source!(i64, i128::from(i64::MIN), i128::from(i64::MAX));
        check_integer_concretization_source!(u1, 0, 1);
        check_integer_concretization_source!(u2, 0, 3);
        check_integer_concretization_source!(u4, 0, 15);
        check_integer_concretization_source!(u8, 0, i128::from(u8::MAX));
        check_integer_concretization_source!(u16, 0, i128::from(u16::MAX));
        check_integer_concretization_source!(u32, 0, i128::from(u32::MAX));
        check_integer_concretization_source!(u64, 0, i128::from(u64::MAX));
    }

    #[test]
    fn test_array_concretize_bool() {
        assert_eq!(Array::scalar(false).unwrap().concretize(), Ok(false));
        assert_eq!(Array::scalar(true).unwrap().concretize(), Ok(true));
    }

    #[test]
    fn test_array_concretize_integers() {
        // Each target accepts values from another integer representation without changing their value.
        check_integer_concretization!(i1, -1i64, 0u64);
        check_integer_concretization!(i2, -2i64, 1u64);
        check_integer_concretization!(i4, -8i64, 7u64);
        check_integer_concretization!(i8, -128i64, 127u64);
        check_integer_concretization!(i16, -32768i64, 32767u64);
        check_integer_concretization!(i32, i64::from(i32::MIN), u64::from(i32::MAX as u32));
        check_integer_concretization!(i64, i64::MIN, i64::MAX as u64);
        check_integer_concretization!(isize, isize::MIN as i64, isize::MAX as u64);
        check_integer_concretization!(u1, 0i64, 1u64);
        check_integer_concretization!(u2, 0i64, 3u64);
        check_integer_concretization!(u4, 0i64, 15u64);
        check_integer_concretization!(u8, 0i64, 255u64);
        check_integer_concretization!(u16, 0i64, 65535u64);
        check_integer_concretization!(u32, 0i64, u64::from(u32::MAX));
        check_integer_concretization!(u64, 0i64, u64::MAX);
        check_integer_concretization!(usize, 0i64, usize::MAX as u64);

        // `u128` targets accept the widest unsigned source value.
        let wide: Result<u128, _> = Array::scalar(u64::MAX).unwrap().concretize();
        assert_eq!(wide, Ok(u128::from(u64::MAX)));

        // Sub-byte integer sources retain their signedness and numeric value when widened.
        assert_eq!(Concretizable::<i8>::concretize(&Array::scalar(i1::MIN).unwrap()), Ok(-1));
        assert_eq!(Concretizable::<i16>::concretize(&Array::scalar(i2::MIN).unwrap()), Ok(-2));
        assert_eq!(Concretizable::<i32>::concretize(&Array::scalar(i4::MIN).unwrap()), Ok(-8));
        assert_eq!(Concretizable::<u8>::concretize(&Array::scalar(u1::MAX).unwrap()), Ok(1));
        assert_eq!(Concretizable::<u16>::concretize(&Array::scalar(u2::MAX).unwrap()), Ok(3));
        assert_eq!(Concretizable::<u32>::concretize(&Array::scalar(u4::MAX).unwrap()), Ok(15));
    }

    #[test]
    fn test_array_concretize_integers_out_of_range() {
        check_integer_concretization_out_of_range!(
            i1,
            1i8,
            "cannot extract a concrete `i1` from `i8[]`; value `1` is out of range",
        );
        check_integer_concretization_out_of_range!(
            i2,
            -3i8,
            "cannot extract a concrete `i2` from `i8[]`; value `-3` is out of range",
        );
        check_integer_concretization_out_of_range!(
            i4,
            8i8,
            "cannot extract a concrete `i4` from `i8[]`; value `8` is out of range",
        );
        check_integer_concretization_out_of_range!(
            u1,
            2u8,
            "cannot extract a concrete `u1` from `u8[]`; value `2` is out of range",
        );
        check_integer_concretization_out_of_range!(
            u2,
            -1i8,
            "cannot extract a concrete `u2` from `i8[]`; value `-1` is out of range",
        );
        check_integer_concretization_out_of_range!(
            u4,
            16u8,
            "cannot extract a concrete `u4` from `u8[]`; value `16` is out of range",
        );
        check_integer_concretization_out_of_range!(
            i8,
            128i16,
            "cannot extract a concrete `i8` from `i16[]`; value `128` is out of range",
        );
        check_integer_concretization_out_of_range!(
            i16,
            32768i32,
            "cannot extract a concrete `i16` from `i32[]`; value `32768` is out of range",
        );
        check_integer_concretization_out_of_range!(
            i32,
            2147483648i64,
            "cannot extract a concrete `i32` from `i64[]`; value `2147483648` is out of range",
        );
        check_integer_concretization_out_of_range!(
            i64,
            u64::MAX,
            "cannot extract a concrete `i64` from `u64[]`; value `18446744073709551615` is out of range",
        );
        check_integer_concretization_out_of_range!(
            u8,
            -1i8,
            "cannot extract a concrete `u8` from `i8[]`; value `-1` is out of range",
        );
        check_integer_concretization_out_of_range!(
            u16,
            65536u32,
            "cannot extract a concrete `u16` from `u32[]`; value `65536` is out of range",
        );
        check_integer_concretization_out_of_range!(
            u32,
            4294967296u64,
            "cannot extract a concrete `u32` from `u64[]`; value `4294967296` is out of range",
        );
        check_integer_concretization_out_of_range!(
            u64,
            -1i8,
            "cannot extract a concrete `u64` from `i8[]`; value `-1` is out of range",
        );
        check_integer_concretization_out_of_range!(
            u128,
            -1i8,
            "cannot extract a concrete `u128` from `i8[]`; value `-1` is out of range",
        );
        check_integer_concretization_out_of_range!(
            usize,
            -1i8,
            "cannot extract a concrete `usize` from `i8[]`; value `-1` is out of range",
        );
        check_integer_concretization_out_of_range!(
            isize,
            u64::MAX,
            "cannot extract a concrete `isize` from `u64[]`; value `18446744073709551615` is out of range",
        );
    }

    #[test]
    fn test_array_concretize_floating_point() {
        check_floating_point_concretization!(f4e2m1fn, 0u8..16);
        check_floating_point_concretization!(f6e2m3fn, 0u8..64);
        check_floating_point_concretization!(f6e3m2fn, 0u8..64);
        check_floating_point_concretization!(f8e3m4, 0u8..=255);
        check_floating_point_concretization!(f8e4m3, 0u8..=255);
        check_floating_point_concretization!(f8e4m3b11fnuz, 0u8..=255);
        check_floating_point_concretization!(f8e4m3fn, 0u8..=255);
        check_floating_point_concretization!(f8e4m3fnuz, 0u8..=255);
        check_floating_point_concretization!(f8e5m2, 0u8..=255);
        check_floating_point_concretization!(f8e5m2fnuz, 0u8..=255);
        check_floating_point_concretization!(f8e8m0fnu, 0u8..=255);
        check_floating_point_concretization!(bf16, [0x0000u16, 0x8000, 0x0001, 0x3f80, 0x7f80, 0xff80, 0x7fc1]);
        check_floating_point_concretization!(f16, [0x0000u16, 0x8000, 0x0001, 0x3c00, 0x7c00, 0xfc00, 0x7e01]);
        check_floating_point_concretization!(
            f32,
            [0x0000_0000u32, 0x8000_0000, 0x0000_0001, 0x3f80_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_1234],
        );
        check_floating_point_concretization!(
            f64,
            [
                0x0000_0000_0000_0000u64,
                0x8000_0000_0000_0000,
                0x0000_0000_0000_0001,
                0x3ff0_0000_0000_0000,
                0x7ff0_0000_0000_0000,
                0xfff0_0000_0000_0000,
                0x7ff8_0000_0000_1234,
            ],
        );
    }

    #[test]
    fn test_array_concretize_complex() {
        let narrow = ComplexNumber::new(f32::from_bits(0x7fc0_1234), -0.0f32);
        let narrow_output: ComplexNumber<f32> = Array::scalar(narrow).unwrap().concretize().unwrap();
        assert_eq!(narrow_output.re.to_bits(), narrow.re.to_bits());
        assert_eq!(narrow_output.im.to_bits(), narrow.im.to_bits());
        let wide = ComplexNumber::new(-0.0f64, f64::from_bits(0x7ff8_0000_0000_1234));
        let wide_output: ComplexNumber<f64> = Array::scalar(wide).unwrap().concretize().unwrap();
        assert_eq!(wide_output.re.to_bits(), wide.re.to_bits());
        assert_eq!(wide_output.im.to_bits(), wide.im.to_bits());
    }

    #[test]
    fn test_array_concretize_incompatible() {
        // Every target requires a rank-zero array of a compatible element data type.
        assert!(matches!(
            Concretizable::<bool>::concretize(&Array::vector(vec![true]).unwrap()),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete boolean from a value of type `bool[1]`; expected `bool[]`",
        ));
        assert!(matches!(
            Concretizable::<bool>::concretize(&Array::scalar(1.0f64).unwrap()),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete boolean from a value of type `f64[]`; expected `bool[]`",
        ));
        assert!(matches!(
            Concretizable::<i128>::concretize(&Array::scalar(1.0f32).unwrap()),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete integer from `f32[]`; expected a scalar integer",
        ));
        assert!(matches!(
            Concretizable::<usize>::concretize(&Array::vector(vec![1i32]).unwrap()),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete integer from `i32[1]`; expected a scalar integer",
        ));
        assert!(matches!(
            Concretizable::<i8>::concretize(&Array::scalar(true).unwrap()),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete integer from `bool[]`; expected a scalar integer",
        ));
        assert!(matches!(
            Concretizable::<f32>::concretize(&Array::vector(vec![1.0f32]).unwrap()),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete `f32` from `f32[1]`; expected `f32[]`",
        ));
        assert!(matches!(
            Concretizable::<f64>::concretize(&Array::scalar(1.0f32).unwrap()),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete `f64` from `f32[]`; expected `f64[]`",
        ));
        assert!(matches!(
            Concretizable::<ComplexNumber<f32>>::concretize(&Array::scalar(1.0f32).unwrap()),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete `Complex<f32>` from `f32[]`; expected `c64[]`",
        ));
        assert!(matches!(
            Concretizable::<ComplexNumber<f32>>::concretize(
                &Array::vector(vec![ComplexNumber::new(1.0f32, 2.0)]).unwrap(),
            ),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete `Complex<f32>` from `c64[1]`; expected `c64[]`",
        ));
        assert!(matches!(
            Concretizable::<ComplexNumber<f64>>::concretize(&Array::scalar(ComplexNumber::new(1.0f32, 2.0)).unwrap()),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete `Complex<f64>` from `c64[]`; expected `c128[]`",
        ));
    }
}
