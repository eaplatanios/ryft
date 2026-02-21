use std::cmp::Ordering;
use std::marker::PhantomData;

use half::{bf16, f16};

use crate::extensions::ffi::buffers::ffi::{
    XLA_FFI_DataType, XLA_FFI_DataType_BF16, XLA_FFI_DataType_C64, XLA_FFI_DataType_C128, XLA_FFI_DataType_F4E2M1FN,
    XLA_FFI_DataType_F8E3M4, XLA_FFI_DataType_F8E4M3, XLA_FFI_DataType_F8E4M3B11FNUZ, XLA_FFI_DataType_F8E4M3FN,
    XLA_FFI_DataType_F8E4M3FNUZ, XLA_FFI_DataType_F8E5M2, XLA_FFI_DataType_F8E5M2FNUZ, XLA_FFI_DataType_F8E8M0FNU,
    XLA_FFI_DataType_F16, XLA_FFI_DataType_F32, XLA_FFI_DataType_F64, XLA_FFI_DataType_PRED, XLA_FFI_DataType_S1,
    XLA_FFI_DataType_S2, XLA_FFI_DataType_S4, XLA_FFI_DataType_S8, XLA_FFI_DataType_S16, XLA_FFI_DataType_S32,
    XLA_FFI_DataType_S64, XLA_FFI_DataType_TOKEN, XLA_FFI_DataType_U1, XLA_FFI_DataType_U2, XLA_FFI_DataType_U4,
    XLA_FFI_DataType_U8, XLA_FFI_DataType_U16, XLA_FFI_DataType_U32, XLA_FFI_DataType_U64,
};
use crate::extensions::ffi::errors::FfiError;
use crate::extensions::ffi::handlers::FfiCallFrame;

/// XLA [`FfiAttribute::Scalar`] value. Refer to [`FfiBufferType`](crate::extensions::ffi::FfiBufferType)
/// for information about how each scalar type is represented in the XLA FFI API.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FfiScalar {
    Token,
    Predicate(bool),
    I1(i8),
    I2(i8),
    I4(i8),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U1(u8),
    U2(u8),
    U4(u8),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F4E2M1FN(u8),
    F8E3M4(u8),
    F8E4M3(u8),
    F8E4M3FN(u8),
    F8E4M3FNUZ(u8),
    F8E4M3B11FNUZ(u8),
    F8E5M2(u8),
    F8E5M2FNUZ(u8),
    F8E8M0FNU(u8),
    BF16(bf16),
    F16(f16),
    F32(f32),
    F64(f64),
    C64 { real: f32, imaginary: f32 },
    C128 { real: f64, imaginary: f64 },
}

impl FfiScalar {
    /// Constructs a new [`FfiScalar`] from the provided [`XLA_FFI_Scalar`](ffi::XLA_FFI_Scalar)
    /// that came from a function in the XLA FFI API.
    #[allow(non_upper_case_globals)]
    pub unsafe fn from_c_api(handle: *const ffi::XLA_FFI_Scalar) -> Result<FfiScalar, FfiError> {
        unsafe {
            if handle.is_null() {
                return Err(FfiError::invalid_argument("the provided XLA FFI scalar handle is a null pointer"));
            }
            let r#type = (*handle).data_type;
            let value = (*handle).value;
            match r#type {
                XLA_FFI_DataType_TOKEN => Ok(Self::Token),
                XLA_FFI_DataType_PRED => Ok(Self::Predicate(Self::parse_value::<u8>(value, r#type)? != 0)),
                XLA_FFI_DataType_S1 => Ok(Self::I1(Self::parse_value::<i8>(value, r#type)?)),
                XLA_FFI_DataType_S2 => Ok(Self::I2(Self::parse_value::<i8>(value, r#type)?)),
                XLA_FFI_DataType_S4 => Ok(Self::I4(Self::parse_value::<i8>(value, r#type)?)),
                XLA_FFI_DataType_S8 => Ok(Self::I8(Self::parse_value::<i8>(value, r#type)?)),
                XLA_FFI_DataType_S16 => Ok(Self::I16(Self::parse_value::<i16>(value, r#type)?)),
                XLA_FFI_DataType_S32 => Ok(Self::I32(Self::parse_value::<i32>(value, r#type)?)),
                XLA_FFI_DataType_S64 => Ok(Self::I64(Self::parse_value::<i64>(value, r#type)?)),
                XLA_FFI_DataType_U1 => Ok(Self::U1(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_U2 => Ok(Self::U2(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_U4 => Ok(Self::U4(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_U8 => Ok(Self::U8(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_U16 => Ok(Self::U16(Self::parse_value::<u16>(value, r#type)?)),
                XLA_FFI_DataType_U32 => Ok(Self::U32(Self::parse_value::<u32>(value, r#type)?)),
                XLA_FFI_DataType_U64 => Ok(Self::U64(Self::parse_value::<u64>(value, r#type)?)),
                XLA_FFI_DataType_F4E2M1FN => Ok(Self::F4E2M1FN(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_F8E3M4 => Ok(Self::F8E3M4(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_F8E4M3 => Ok(Self::F8E4M3(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_F8E4M3FN => Ok(Self::F8E4M3FN(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_F8E4M3FNUZ => Ok(Self::F8E4M3FNUZ(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_F8E4M3B11FNUZ => Ok(Self::F8E4M3B11FNUZ(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_F8E5M2 => Ok(Self::F8E5M2(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_F8E5M2FNUZ => Ok(Self::F8E5M2FNUZ(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_F8E8M0FNU => Ok(Self::F8E8M0FNU(Self::parse_value::<u8>(value, r#type)?)),
                XLA_FFI_DataType_BF16 => Ok(Self::BF16(bf16::from_bits(Self::parse_value::<u16>(value, r#type)?))),
                XLA_FFI_DataType_F16 => Ok(Self::F16(f16::from_bits(Self::parse_value::<u16>(value, r#type)?))),
                XLA_FFI_DataType_F32 => Ok(Self::F32(Self::parse_value::<f32>(value, r#type)?)),
                XLA_FFI_DataType_F64 => Ok(Self::F64(Self::parse_value::<f64>(value, r#type)?)),
                XLA_FFI_DataType_C64 => {
                    let parts = Self::parse_value::<[f32; 2]>(value, r#type)?;
                    Ok(Self::C64 { real: parts[0], imaginary: parts[1] })
                }
                XLA_FFI_DataType_C128 => {
                    let parts = Self::parse_value::<[f64; 2]>(value, r#type)?;
                    Ok(Self::C128 { real: parts[0], imaginary: parts[1] })
                }
                _ => {
                    Err(FfiError::invalid_argument(format!("invalid XLA FFI scalar attribute data type '{}'", r#type)))
                }
            }
        }
    }

    /// Internal helper for [`Self::from_c_api`].
    unsafe fn parse_value<T: Copy>(value: *mut std::ffi::c_void, data_type: XLA_FFI_DataType) -> Result<T, FfiError> {
        if value.is_null() {
            Err(FfiError::invalid_argument(format!(
                "encountered null scalar value pointer for data type '{data_type}'"
            )))
        } else {
            Ok(unsafe { *(value as *const T) })
        }
    }
}

/// XLA [`FfiAttribute::Array`] value that contains [`FfiScalar`]s.
#[derive(Copy, Clone)]
pub struct FfiArray<'o> {
    /// Handle that represents this [`FfiArray`] in the XLA FFI API.
    handle: *const ffi::XLA_FFI_Array,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`FfiArray`].
    owner: PhantomData<&'o ()>,
}

impl<'o> FfiArray<'o> {
    /// Constructs a new [`FfiArray`] from the provided [`XLA_FFI_Array`](ffi::XLA_FFI_Array)
    /// that came from a function in the XLA FFI API.
    pub unsafe fn from_c_api(handle: *const ffi::XLA_FFI_Array) -> Result<Self, FfiError> {
        unsafe {
            if handle.is_null() {
                return Err(FfiError::invalid_argument("the provided XLA FFI array handle is a null pointer"));
            }
            let data_type = (*handle).data_type;
            Self::element_size_in_bytes(data_type)?;
            if (*handle).size > 0 && data_type != XLA_FFI_DataType_TOKEN {
                if (*handle).data.is_null() {
                    return Err(FfiError::invalid_argument("encountered null XLA FFI attribute values pointer"));
                }
            }
            Ok(Self { handle, owner: PhantomData })
        }
    }

    /// Returns length of this [`FfiArray`] (i.e., the number of [`FfiScalar`] that it contains).
    pub fn len(&self) -> usize {
        unsafe { (*self.handle).size }
    }

    /// Returns `true` if and only if this [`FfiArray`] has no elements (i.e., it has [`FfiArray::len`] equal to `0`).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the [`FfiScalar`] at index `index` of this [`FfiArray`].
    pub fn get(&self, index: usize) -> Result<FfiScalar, FfiError> {
        unsafe {
            let length = self.len();
            if index >= length {
                return Err(FfiError::invalid_argument(format!(
                    "XLA FFI array index {index} is out of bounds for an array with {length} elements"
                )));
            }
            let data_type = (*self.handle).data_type;
            if data_type == XLA_FFI_DataType_TOKEN {
                return Ok(FfiScalar::Token);
            }
            let data = (*self.handle).data;
            if data.is_null() {
                return Err(FfiError::invalid_argument(format!(
                    "encountered null XLA FFI array data pointer for array with data type '{data_type}'"
                )));
            }
            let offset = index.checked_mul(Self::element_size_in_bytes(data_type)?).ok_or_else(|| {
                FfiError::invalid_argument(format!("XLA FFI array byte offset overflow for element at index {index}"))
            })?;
            let value = (data as *const u8).add(offset) as *mut std::ffi::c_void;
            let scalar = ffi::XLA_FFI_Scalar { data_type, value };
            FfiScalar::from_c_api(&scalar as *const _)
        }
    }

    /// Returns an [`Iterator`] over the [`FfiScalar`]s in this [`FfiArray`].
    pub fn iter(&self) -> impl Iterator<Item = Result<FfiScalar, FfiError>> + '_ {
        (0..self.len()).map(|index| self.get(index))
    }

    /// Returns the number of bytes used to store each [`FfiScalar`] in this [`FfiArray`].
    #[allow(non_upper_case_globals)]
    fn element_size_in_bytes(data_type: XLA_FFI_DataType) -> Result<usize, FfiError> {
        let element_size = match data_type {
            XLA_FFI_DataType_TOKEN => 0,
            XLA_FFI_DataType_PRED => size_of::<u8>(),
            XLA_FFI_DataType_S1 | XLA_FFI_DataType_S2 | XLA_FFI_DataType_S4 | XLA_FFI_DataType_S8 => size_of::<i8>(),
            XLA_FFI_DataType_S16 => size_of::<i16>(),
            XLA_FFI_DataType_S32 => size_of::<i32>(),
            XLA_FFI_DataType_S64 => size_of::<i64>(),
            XLA_FFI_DataType_U1 | XLA_FFI_DataType_U2 | XLA_FFI_DataType_U4 | XLA_FFI_DataType_U8 => size_of::<u8>(),
            XLA_FFI_DataType_U16 => size_of::<u16>(),
            XLA_FFI_DataType_U32 => size_of::<u32>(),
            XLA_FFI_DataType_U64 => size_of::<u64>(),
            XLA_FFI_DataType_F4E2M1FN
            | XLA_FFI_DataType_F8E3M4
            | XLA_FFI_DataType_F8E4M3
            | XLA_FFI_DataType_F8E4M3FN
            | XLA_FFI_DataType_F8E4M3FNUZ
            | XLA_FFI_DataType_F8E4M3B11FNUZ
            | XLA_FFI_DataType_F8E5M2
            | XLA_FFI_DataType_F8E5M2FNUZ
            | XLA_FFI_DataType_F8E8M0FNU => size_of::<u8>(),
            XLA_FFI_DataType_BF16 | XLA_FFI_DataType_F16 => size_of::<u16>(),
            XLA_FFI_DataType_F32 => size_of::<f32>(),
            XLA_FFI_DataType_F64 => size_of::<f64>(),
            XLA_FFI_DataType_C64 => size_of::<[f32; 2]>(),
            XLA_FFI_DataType_C128 => size_of::<[f64; 2]>(),
            _ => {
                return Err(FfiError::invalid_argument(format!("invalid XLA FFI array data type '{data_type}'")));
            }
        };
        Ok(element_size)
    }
}

/// XLA [`FfiAttribute::Dictionary`] value that contains a mapping from string names to [`FfiAttribute`]s.
#[derive(Copy, Clone)]
pub struct FfiAttributes<'o> {
    /// Handle that represents this [`FfiAttributes`] in the XLA FFI API.
    handle: *const ffi::XLA_FFI_Attrs,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`FfiAttributes`].
    owner: PhantomData<&'o ()>,
}

impl<'o> FfiAttributes<'o> {
    /// Constructs a new [`FfiAttributes`] from the provided [`XLA_FFI_Attrs`](ffi::XLA_FFI_Attrs)
    /// that came from a function in the XLA FFI API.
    pub unsafe fn from_c_api(handle: *const ffi::XLA_FFI_Attrs) -> Result<Self, FfiError> {
        unsafe {
            if handle.is_null() {
                return Err(FfiError::invalid_argument("the provided XLA FFI attributes handle is a null pointer"));
            }
            let size = (*handle).size as usize;
            if size > 0 {
                if (*handle).types.is_null() {
                    return Err(FfiError::invalid_argument("encountered null XLA FFI attribute types pointer"));
                }
                if (*handle).names.is_null() {
                    return Err(FfiError::invalid_argument("encountered null XLA FFI attribute names pointer"));
                }
                if (*handle).attributes.is_null() {
                    return Err(FfiError::invalid_argument("encountered null XLA FFI attribute values pointer"));
                }
            }
            Ok(Self { handle, owner: PhantomData })
        }
    }

    /// Returns length of this [`FfiAttributes`] dictionary (i.e., the number of attribute values that it contains).
    pub fn len(&self) -> usize {
        unsafe { (*self.handle).size as usize }
    }

    /// Returns `true` if and only if this [`FfiAttributes`] dictionary has no elements
    /// (i.e., it has [`FfiAttributes::len`] equal to `0`).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if and only if this [`FfiAttributes`] dictionary contains the provided key.
    pub fn contains_key<S: AsRef<str>>(&self, key: S) -> Result<bool, FfiError> {
        match self.find_index(key.as_ref()) {
            Ok(_) => Ok(true),
            Err(FfiError::NotFound { .. }) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Returns the [`FfiAttribute`] that corresponds to the provided `key`, and an [`FfiError`] if the `key` is not
    /// valid UTF-8, if there is no matching entry, or if the matching entry is invalid.
    pub fn get<S: AsRef<str>>(&self, key: S) -> Result<FfiAttribute<'o>, FfiError> {
        self.value_at(self.find_index(key.as_ref())?)
    }

    /// Returns an [`Iterator`] over `(name, attribute)` pairs in this [`FfiAttributes`] dictionary.
    pub fn iter(&self) -> FfiAttributesIterator<'_, 'o> {
        FfiAttributesIterator { attributes: self, index: 0 }
    }

    /// Returns an [`Iterator`] over the keys/names of all attributes in this [`FfiAttributes`] dictionary.
    pub fn keys(&self) -> impl Iterator<Item = Result<&'o str, FfiError>> + '_ {
        (0..self.len()).map(|index| self.key_at(index))
    }

    /// Returns an [`Iterator`] over the values/[`FfiAttribute`]s in this [`FfiAttributes`] dictionary.
    pub fn values(&self) -> impl Iterator<Item = Result<FfiAttribute<'o>, FfiError>> + '_ {
        (0..self.len()).map(|index| self.value_at(index))
    }

    /// Internal helper that returns the index of the provided `key` in this [`FfiAttributes`] dictionary.
    fn find_index(&self, key: &str) -> Result<usize, FfiError> {
        // XLA FFI attribute dictionaries are sorted by key name, and so we can use binary search.
        let mut low = 0usize;
        let mut high = self.len();
        while low < high {
            let middle = low + (high - low) / 2;
            match self.key_at(middle)?.cmp(key) {
                Ordering::Less => low = middle + 1,
                _ => high = middle,
            }
        }
        if low < self.len() && self.key_at(low)? == key {
            Ok(low)
        } else {
            Err(FfiError::not_found(format!("dictionary key '{key}' not found in XLA FFI attributes")))
        }
    }

    /// Internal helper that returns the key/name at the specified index of this [`FfiAttributes`] dictionary.
    fn key_at(&self, index: usize) -> Result<&'o str, FfiError> {
        unsafe {
            let len = self.len();
            if index >= len {
                return Err(FfiError::invalid_argument(format!(
                    "XLA FFI attribute dictionary index {index} is out of bounds for a dictionary with {len} entries",
                )));
            }
            if (*self.handle).names.is_null() {
                return Err(FfiError::invalid_argument(
                    "encountered null attribute names pointer in XLA FFI attributes",
                ));
            }
            let name_span = *(*self.handle).names.add(index);
            if name_span.is_null() {
                return Err(FfiError::invalid_argument(format!(
                    "encountered null XLA FFI attribute name span pointer for dictionary entry at index {index}",
                )));
            }
            if (*name_span).ptr.is_null() {
                return Err(FfiError::invalid_argument(format!(
                    "encountered null XLA FFI attribute name pointer for dictionary entry at index {index}",
                )));
            }
            let bytes = std::slice::from_raw_parts((*name_span).ptr as *const u8, (*name_span).len);
            std::str::from_utf8(bytes).map_err(|error| {
                FfiError::invalid_argument(format!(
                    "XLA FFI attribute dictionary entry name at index {index} is not valid UTF-8: {error}",
                ))
            })
        }
    }

    /// Internal helper that returns the value/[`FfiAttribute`] at the specified index of this [`FfiAttributes`]
    /// dictionary.
    fn value_at(&self, index: usize) -> Result<FfiAttribute<'o>, FfiError> {
        unsafe {
            let len = self.len();
            if index >= len {
                return Err(FfiError::invalid_argument(format!(
                    "XLA FFI attribute dictionary index {index} is out of bounds for a dictionary with {len} entries",
                )));
            }
            if (*self.handle).types.is_null() {
                return Err(FfiError::invalid_argument(
                    "encountered null attribute types pointer in XLA FFI attributes",
                ));
            }
            if (*self.handle).attributes.is_null() {
                return Err(FfiError::invalid_argument(
                    "encountered null attribute values pointer in XLA FFI attributes",
                ));
            }
            let r#type = *(*self.handle).types.add(index);
            let value = *(*self.handle).attributes.add(index);
            FfiAttribute::from_c_api(r#type, value)
        }
    }
}

/// Iterator over entries in [`FfiAttributes`].
pub struct FfiAttributesIterator<'a, 'o> {
    /// [`FfiAttributes`] that this [`FfiAttributesIterator`] is iterating over.
    attributes: &'a FfiAttributes<'o>,

    /// Current entry index.
    index: usize,
}

impl<'a, 'o> Iterator for FfiAttributesIterator<'a, 'o> {
    type Item = Result<(&'o str, FfiAttribute<'o>), FfiError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.attributes.len() {
            None
        } else {
            let index = self.index;
            self.index += 1;
            Some(
                self.attributes
                    .key_at(index)
                    .and_then(|key| self.attributes.value_at(index).map(|value| (key, value))),
            )
        }
    }
}

/// Attribute value in an XLA [`FfiCallFrame`].
#[derive(Copy, Clone)]
pub enum FfiAttribute<'o> {
    Scalar { scalar: FfiScalar },
    String { string: &'o str },
    Array { array: FfiArray<'o> },
    Dictionary { dictionary: FfiAttributes<'o> },
}

impl FfiAttribute<'_> {
    /// Constructs a new [`FfiAttribute`] from the provided [`XLA_FFI_AttrType`](ffi::XLA_FFI_AttrType)
    /// and raw attribute value pointer that came from a function in the XLA FFI API.
    pub unsafe fn from_c_api(r#type: ffi::XLA_FFI_AttrType, value: *mut std::ffi::c_void) -> Result<Self, FfiError> {
        unsafe {
            match r#type {
                ffi::XLA_FFI_AttrType_SCALAR => {
                    let scalar = value as *const ffi::XLA_FFI_Scalar;
                    if scalar.is_null() {
                        return Err(FfiError::invalid_argument(
                            "encountered null scalar pointer for XLA FFI attribute",
                        ));
                    }
                    Ok(Self::Scalar { scalar: FfiScalar::from_c_api(scalar)? })
                }
                ffi::XLA_FFI_AttrType_STRING => {
                    let span = value as *const ffi::XLA_FFI_ByteSpan;
                    if span.is_null() {
                        return Err(FfiError::invalid_argument(
                            "encountered null string span pointer for XLA FFI attribute",
                        ));
                    }
                    if (*span).ptr.is_null() {
                        return Err(FfiError::invalid_argument("encountered null string pointer for attribute"));
                    }
                    let bytes = std::slice::from_raw_parts((*span).ptr as *const u8, (*span).len);
                    let string = std::str::from_utf8(bytes).map_err(|error| {
                        FfiError::invalid_argument(format!("XLA FFI attribute is not a valid UTF-8 string: {error}"))
                    })?;
                    Ok(Self::String { string })
                }
                ffi::XLA_FFI_AttrType_ARRAY => {
                    Ok(Self::Array { array: FfiArray::from_c_api(value as *const ffi::XLA_FFI_Array)? })
                }
                ffi::XLA_FFI_AttrType_DICTIONARY => {
                    Ok(Self::Dictionary { dictionary: FfiAttributes::from_c_api(value as *const ffi::XLA_FFI_Attrs)? })
                }
                _ => Err(FfiError::invalid_argument(format!("invalid XLA FFI attribute type '{}'", r#type))),
            }
        }
    }
}

impl<'o> FfiCallFrame<'o> {
    /// Returns the number of [`FfiAttribute`]s of this [`FfiCallFrame`].
    pub fn attribute_count(&self) -> usize {
        unsafe { (*self.to_c_api()).attributes.size as usize }
    }

    /// Returns the name of the `index`-th [`FfiAttribute`] of this [`FfiCallFrame`], and an [`FfiError`] if the
    /// provided index is out of bounds or if the runtime reports invalid attribute name metadata.
    pub fn attribute_name(&self, index: usize) -> Result<&'o str, FfiError> {
        unsafe {
            let attributes = &(*self.to_c_api()).attributes;
            let count = usize::try_from(attributes.size).unwrap_or(0);
            if index >= count {
                return Err(FfiError::invalid_argument(format!(
                    "attribute index {index} is out of bounds for an XLA FFI call frame with {count} attributes",
                )));
            }
            if attributes.names.is_null() {
                return Err(FfiError::invalid_argument(
                    "encountered null attribute names pointer in XLA FFI call frame",
                ));
            }
            let span = *attributes.names.add(index);
            if span.is_null() {
                return Err(FfiError::invalid_argument(format!(
                    "encountered null attribute name span pointer for XLA FFI attribute at index {index}",
                )));
            }
            if (*span).ptr.is_null() {
                return Err(FfiError::invalid_argument(format!(
                    "encountered null attribute name pointer for XLA FFI attribute at index {index}",
                )));
            }
            let bytes = std::slice::from_raw_parts((*span).ptr as *const u8, (*span).len);
            std::str::from_utf8(bytes).map_err(|error| {
                FfiError::invalid_argument(format!(
                    "XLA FFI attribute name at index {index} is not valid UTF-8: {error}",
                ))
            })
        }
    }

    /// Returns the `index`-th [`FfiAttribute`] of this [`FfiCallFrame`], and an [`FfiError`] if the provided index is
    /// out of bounds, if the XLA runtime reports an unknown attribute type value, or if the runtime reports invalid
    /// metadata for that attribute.
    pub fn attribute(&self, index: usize) -> Result<FfiAttribute<'o>, FfiError> {
        unsafe {
            let attributes = &(*self.to_c_api()).attributes;
            let count = attributes.size as usize;
            if index >= count {
                return Err(FfiError::invalid_argument(format!(
                    "attribute index {index} is out of bounds for an XLA FFI call frame with {count} attributes",
                )));
            }
            if attributes.types.is_null() {
                return Err(FfiError::invalid_argument(
                    "encountered null attribute types pointer in XLA FFI call frame",
                ));
            }
            if attributes.attributes.is_null() {
                return Err(FfiError::invalid_argument(
                    "encountered null attribute values pointer in XLA FFI call frame",
                ));
            }
            FfiAttribute::from_c_api(*attributes.types.add(index), *attributes.attributes.add(index))
        }
    }

    /// Returns an [`Iterator`] over the [`FfiAttribute`]s of this [`FfiCallFrame`].
    pub fn attributes(&self) -> impl Iterator<Item = Result<(&'o str, FfiAttribute<'o>), FfiError>> {
        (0..self.attribute_count())
            .map(|index| self.attribute_name(index).and_then(|name| self.attribute(index).map(|value| (name, value))))
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::extensions::ffi::buffers::ffi::XLA_FFI_DataType;
    use crate::extensions::ffi::handlers::ffi::XLA_FFI_Extension_Base;

    // XLA FFI uses byte spans to pass strings to handlers because strings might not be null terminated,
    // and even if they are, looking for a null terminator can become very expensive in tight loops.
    #[repr(C)]
    pub struct XLA_FFI_ByteSpan {
        pub ptr: *const std::ffi::c_char,
        pub len: usize,
    }

    impl<S: AsRef<str>> From<S> for XLA_FFI_ByteSpan {
        fn from(value: S) -> Self {
            let value = value.as_ref();
            Self { ptr: value.as_ptr() as *const std::ffi::c_char, len: value.len() }
        }
    }

    pub type XLA_FFI_AttrType = std::ffi::c_uint;
    pub const XLA_FFI_AttrType_ARRAY: XLA_FFI_AttrType = 1;
    pub const XLA_FFI_AttrType_DICTIONARY: XLA_FFI_AttrType = 2;
    pub const XLA_FFI_AttrType_SCALAR: XLA_FFI_AttrType = 3;
    pub const XLA_FFI_AttrType_STRING: XLA_FFI_AttrType = 4;

    #[repr(C)]
    pub struct XLA_FFI_Scalar {
        pub data_type: XLA_FFI_DataType,
        pub value: *mut std::ffi::c_void,
    }

    #[repr(C)]
    pub struct XLA_FFI_Array {
        pub data_type: XLA_FFI_DataType,
        pub size: usize,
        pub data: *mut std::ffi::c_void,
    }

    #[repr(C)]
    pub struct XLA_FFI_Attrs {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub size: i64,
        pub types: *mut XLA_FFI_AttrType,
        pub names: *mut *mut XLA_FFI_ByteSpan,
        pub attributes: *mut *mut std::ffi::c_void,
    }

    impl XLA_FFI_Attrs {
        pub fn new(
            size: i64,
            types: *mut XLA_FFI_AttrType,
            names: *mut *mut XLA_FFI_ByteSpan,
            attributes: *mut *mut std::ffi::c_void,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                size,
                types,
                names,
                attributes,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::extensions::ffi::errors::FfiError;
    use crate::extensions::ffi::tests::with_test_ffi_call_frame;

    use super::{FfiAttribute, FfiScalar};

    #[test]
    fn test_ffi_attribute() {
        with_test_ffi_call_frame(|call_frame| {
            assert_eq!(call_frame.attribute_count(), 4);
            for name_and_attribute in call_frame.attributes() {
                assert!(name_and_attribute.is_ok());
                let (name, attribute) = name_and_attribute.unwrap();
                match (name, attribute) {
                    ("array_attr", FfiAttribute::Array { array }) => {
                        assert_eq!(array.len(), 3);
                        assert!(!array.is_empty());
                        assert_eq!(array.get(0), Ok(FfiScalar::I32(1)));
                        assert_eq!(array.get(1), Ok(FfiScalar::I32(2)));
                        assert_eq!(array.get(2), Ok(FfiScalar::I32(3)));
                        assert!(matches!(array.get(3), Err(FfiError::InvalidArgument { .. })));
                        assert_eq!(
                            array.iter().collect::<Result<Vec<_>, _>>(),
                            Ok(vec![FfiScalar::I32(1), FfiScalar::I32(2), FfiScalar::I32(3)]),
                        );
                    }
                    ("dictionary_attr", FfiAttribute::Dictionary { dictionary }) => {
                        assert_eq!(dictionary.len(), 4);
                        assert!(!dictionary.is_empty());
                        assert_eq!(dictionary.contains_key("nested_array_attr"), Ok(true));
                        assert_eq!(dictionary.contains_key("nested_dictionary_attr"), Ok(true));
                        assert_eq!(dictionary.contains_key("nested_scalar_attr"), Ok(true));
                        assert_eq!(dictionary.contains_key("nested_string_attr"), Ok(true));
                        assert_eq!(dictionary.contains_key("missing"), Ok(false));
                        let nested_array = dictionary.get("nested_array_attr");
                        let Ok(FfiAttribute::Array { array }) = nested_array else {
                            panic!("failed to get the 'nested_array_attr' XLA FFI attribute");
                        };
                        assert_eq!(array.len(), 2);
                        assert_eq!(array.get(0), Ok(FfiScalar::I32(4)));
                        assert_eq!(array.get(1), Ok(FfiScalar::I32(5)));
                        let nested_dictionary = dictionary.get("nested_dictionary_attr");
                        let Ok(FfiAttribute::Dictionary { dictionary: nested_dictionary }) = nested_dictionary else {
                            panic!("failed to get the 'nested_dictionary_attr' XLA FFI attribute");
                        };
                        assert_eq!(nested_dictionary.len(), 2);
                        assert!(matches!(
                            nested_dictionary.get("nested_inner_scalar_attr"),
                            Ok(FfiAttribute::Scalar { scalar: FfiScalar::I32(11) }),
                        ));
                        assert!(matches!(
                            nested_dictionary.get("nested_inner_string_attr"),
                            Ok(FfiAttribute::String { string }) if string == "inner",
                        ));
                        assert!(matches!(
                            dictionary.get("nested_scalar_attr"),
                            Ok(FfiAttribute::Scalar { scalar: FfiScalar::I32(9) }),
                        ));
                        assert!(matches!(
                            dictionary.get("nested_string_attr"),
                            Ok(FfiAttribute::String { string }) if string == "nested_value",
                        ));
                        assert!(matches!(dictionary.get("missing"), Err(FfiError::NotFound { .. })));
                        let keys = dictionary.keys().collect::<Result<Vec<_>, _>>().unwrap();
                        assert_eq!(keys.len(), 4);
                        assert!(keys.contains(&"nested_array_attr"));
                        assert!(keys.contains(&"nested_dictionary_attr"));
                        assert!(keys.contains(&"nested_scalar_attr"));
                        assert!(keys.contains(&"nested_string_attr"));
                        let values = dictionary.values().collect::<Result<Vec<_>, _>>().unwrap();
                        assert_eq!(values.len(), 4);
                        assert!(values.iter().any(|value| matches!(value, FfiAttribute::Array { .. })));
                        assert!(values.iter().any(|value| matches!(value, FfiAttribute::Dictionary { .. })));
                        assert!(
                            values
                                .iter()
                                .any(|value| matches!(value, FfiAttribute::Scalar { scalar: FfiScalar::I32(9) }))
                        );
                        assert!(values.iter().any(|value| {
                            matches!(value, FfiAttribute::String { string } if *string == "nested_value")
                        }));
                        let entries = dictionary.iter().collect::<Result<Vec<_>, _>>().unwrap();
                        assert_eq!(entries.len(), 4);
                        assert!(entries.iter().any(|(key, value)| {
                            *key == "nested_array_attr" && matches!(value, FfiAttribute::Array { .. })
                        }));
                        assert!(entries.iter().any(|(key, value)| {
                            *key == "nested_dictionary_attr" && matches!(value, FfiAttribute::Dictionary { .. })
                        }));
                        assert!(entries.iter().any(|(key, value)| {
                            *key == "nested_scalar_attr"
                                && matches!(value, FfiAttribute::Scalar { scalar: FfiScalar::I32(9) })
                        }));
                        assert!(entries.iter().any(|(key, value)| {
                            *key == "nested_string_attr"
                                && matches!(value, FfiAttribute::String { string } if *string == "nested_value")
                        }));
                    }
                    ("scalar_attr", FfiAttribute::Scalar { scalar }) => assert_eq!(scalar, FfiScalar::I32(7)),
                    ("string_attr", FfiAttribute::String { string }) => assert_eq!(string, "value"),
                    ("api_version", FfiAttribute::Scalar { scalar }) => assert_eq!(scalar, FfiScalar::I32(4)),
                    _ => panic!("encountered unexpected XLA FFI attribute"),
                }
            }
        });
    }
}
