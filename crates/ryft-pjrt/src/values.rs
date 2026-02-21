use std::fmt::Display;

use crate::{slice_from_c_api, str_from_c_api};

/// Represents a constant value that can be used when interfacing with the PJRT C API.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Value {
    Bool(bool),
    I64(i64),
    I64List(Vec<i64>),
    F32(f32),
    String(String),
}

impl Value {
    /// Creates a new [`Value::Bool`].
    pub fn r#bool<V: Into<bool>>(value: V) -> Self {
        Self::Bool(value.into())
    }

    /// Creates a new [`Value::I64`].
    pub fn i64<V: Into<i64>>(value: V) -> Self {
        Self::I64(value.into())
    }

    /// Creates a new [`Value::I64List`].
    pub fn i64_list<V: Into<Vec<i64>>>(value: V) -> Self {
        Self::I64List(value.into())
    }

    /// Creates a new [`Value::F32`].
    pub fn f32<V: Into<f32>>(value: V) -> Self {
        Self::F32(value.into())
    }

    /// Creates a new [`Value::String`].
    pub fn string<V: Into<String>>(value: V) -> Self {
        Self::String(value.into())
    }
}

impl Display for Value {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(value) => write!(formatter, "{value}"),
            Self::I64(value) => write!(formatter, "{value}"),
            Self::I64List(value) => write!(formatter, "{value:?}"),
            Self::F32(value) => write!(formatter, "{value}"),
            Self::String(value) => write!(formatter, "\"{value}\""),
        }
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<Vec<i64>> for Value {
    fn from(value: Vec<i64>) -> Self {
        Self::I64List(value)
    }
}

impl<const N: usize> From<[i64; N]> for Value {
    fn from(value: [i64; N]) -> Self {
        Self::I64List(value.into())
    }
}

impl From<&[i64]> for Value {
    fn from(value: &[i64]) -> Self {
        Self::I64List(value.into())
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::String(value.into())
    }
}

/// Represents a named [`Value`] that can be used when interfacing with PJRT.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedValue {
    /// Name of the value.
    pub name: String,

    /// Underlying value.
    pub value: Value,
}

impl NamedValue {
    /// Constructs a new [`NamedValue`] from the provided [`PJRT_NamedValue`](ffi::PJRT_NamedValue) handle that came
    /// from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(handle: &ffi::PJRT_NamedValue) -> Self {
        Self {
            name: str_from_c_api(handle.name, handle.name_size).into_owned(),
            value: match handle.value_type {
                ffi::PJRT_NamedValue_Type_kBool => unsafe { Value::Bool(handle.value.bool_value) },
                ffi::PJRT_NamedValue_Type_kInt64 => unsafe { Value::I64(handle.value.int64_value) },
                ffi::PJRT_NamedValue_Type_kInt64List => unsafe {
                    let value_ptr = handle.value.int64_array_value;
                    let value_size = handle.value_size;
                    let value = slice_from_c_api(value_ptr, value_size);
                    Value::I64List(value.to_vec())
                },
                ffi::PJRT_NamedValue_Type_kFloat => unsafe { Value::F32(handle.value.float_value) },
                ffi::PJRT_NamedValue_Type_kString => unsafe {
                    let value_ptr = handle.value.string_value;
                    let value_size = handle.value_size;
                    let value = slice_from_c_api(value_ptr as *const u8, value_size);
                    Value::String(String::from_utf8_lossy(value).into_owned())
                },
                value_type => panic!("unsupported PJRT value type: {value_type}"),
            },
        }
    }

    /// Returns the [`PJRT_NamedValue`](ffi::PJRT_NamedValue) that corresponds to this [`NamedValue`] and which can
    /// be passed to functions in the PJRT C API.
    ///
    /// # Safety
    ///
    /// This function is marked as unsafe because the resulting [`PJRT_NamedValue`](ffi::PJRT_NamedValue) can become
    /// invalid after this [`NamedValue`] is dropped.
    pub(crate) unsafe fn to_c_api(&self) -> ffi::PJRT_NamedValue {
        let name = self.name.as_ptr() as *const i8;
        let name_size = self.name.as_bytes().len();
        match &self.value {
            Value::Bool(value) => ffi::PJRT_NamedValue {
                struct_size: size_of::<ffi::PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name,
                name_size,
                value_type: ffi::PJRT_NamedValue_Type_kBool,
                value: ffi::PJRT_Value { bool_value: *value },
                value_size: 1,
            },
            Value::I64(value) => ffi::PJRT_NamedValue {
                struct_size: size_of::<ffi::PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name,
                name_size,
                value_type: ffi::PJRT_NamedValue_Type_kInt64,
                value: ffi::PJRT_Value { int64_value: *value },
                value_size: 1,
            },
            Value::I64List(value) => ffi::PJRT_NamedValue {
                struct_size: size_of::<ffi::PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name,
                name_size,
                value_type: ffi::PJRT_NamedValue_Type_kInt64List,
                value: ffi::PJRT_Value { int64_array_value: value.as_ptr() },
                value_size: value.len(),
            },
            Value::F32(value) => ffi::PJRT_NamedValue {
                struct_size: size_of::<ffi::PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name,
                name_size,
                value_type: ffi::PJRT_NamedValue_Type_kFloat,
                value: ffi::PJRT_Value { float_value: *value },
                value_size: 1,
            },
            Value::String(value) => ffi::PJRT_NamedValue {
                struct_size: size_of::<ffi::PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name,
                name_size,
                value_type: ffi::PJRT_NamedValue_Type_kString,
                value: ffi::PJRT_Value { string_value: value.as_ptr() as *const i8 },
                value_size: value.as_bytes().len(),
            },
        }
    }

    /// Creates a new [`NamedValue`] with the provided name and underlying value.
    pub fn new<S: AsRef<str>, V: Into<Value>>(name: S, value: V) -> Self {
        Self { name: name.as_ref().to_string(), value: value.into() }
    }
}

impl Display for NamedValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{name}: {value}", name = self.name, value = self.value)
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::ffi::PJRT_Extension_Base;

    pub type PJRT_NamedValue_Type = std::ffi::c_uint;
    pub const PJRT_NamedValue_Type_kString: PJRT_NamedValue_Type = 0;
    pub const PJRT_NamedValue_Type_kInt64: PJRT_NamedValue_Type = 1;
    pub const PJRT_NamedValue_Type_kInt64List: PJRT_NamedValue_Type = 2;
    pub const PJRT_NamedValue_Type_kFloat: PJRT_NamedValue_Type = 3;
    pub const PJRT_NamedValue_Type_kBool: PJRT_NamedValue_Type = 4;

    #[repr(C)]
    pub union PJRT_Value {
        pub string_value: *const std::ffi::c_char,
        pub int64_value: i64,
        pub int64_array_value: *const i64,
        pub float_value: f32,
        pub bool_value: bool,
    }

    #[repr(C)]
    pub struct PJRT_NamedValue {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub name: *const std::ffi::c_char,
        pub name_size: usize,
        pub value_type: PJRT_NamedValue_Type,
        pub value: PJRT_Value,
        pub value_size: usize,
    }

    impl PJRT_NamedValue {
        pub fn new(
            name: *const std::ffi::c_char,
            name_size: usize,
            value_type: PJRT_NamedValue_Type,
            value: PJRT_Value,
            value_size: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                name,
                name_size,
                value_type,
                value,
                value_size,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value() {
        let value_true = Value::r#bool(true);
        let value_false: Value = false.into();
        let value_0i64 = Value::i64(0);
        let value_42i64 = Value::i64(42);
        let value_m100i64: Value = (-100).into();
        let value_i64_list: Value = [3, 2, 2].into();
        let value_3p14f32: Value = 3.14f32.into();
        let value_3p15f32 = Value::f32(3.15);
        let value_0f32 = Value::f32(0.0);
        let value_empty_string = Value::string("");
        let value_hello_string: Value = "hello".into();
        assert_eq!(value_true, Value::r#bool(true));
        assert_ne!(value_false, value_true);
        assert_eq!(value_false, value_false);
        assert_eq!(value_42i64, Value::i64(42));
        assert_eq!(value_42i64, Value::i64(42));
        assert_ne!(value_0i64, value_m100i64);
        assert_ne!(value_42i64, value_m100i64);
        assert_ne!(value_42i64, value_true);
        assert_eq!(Value::from(i64::MAX), Value::i64(i64::MAX));
        assert_eq!(Value::i64_list([3, 2, 2]), value_i64_list);
        assert_ne!(Value::i64_list([3, 2, 1]), value_i64_list);
        assert_ne!(value_i64_list, value_0i64);
        assert_eq!(value_3p14f32, Value::f32(3.14));
        assert_ne!(value_3p14f32, value_3p15f32);
        assert_ne!(value_0f32, value_0i64);
        assert_ne!(value_0f32, value_empty_string);
        assert_eq!(Value::string(""), value_empty_string);
        assert_ne!(value_hello_string, value_empty_string);
        assert_eq!(value_hello_string, value_hello_string);
        assert_ne!(value_0i64, value_empty_string);
        assert_eq!(Value::from("こんにちは"), Value::string("こんにちは"));
        assert!(Value::i64(1) < Value::i64(2));
        assert!(Value::i64(2) > Value::i64(1));
        assert!(Value::i64(1) <= Value::i64(1));
        assert!(Value::i64(1) >= Value::i64(1));
        assert!(Value::f32(1.0) < Value::f32(2.0));
        assert!(Value::string("a") < Value::string("b"));
        assert!(Value::f32(1.0) < Value::string("b"));

        let value_0 = Value::string("");
        let value_1 = Value::string("hello");
        assert_eq!(value_0, value_0);
        assert_eq!(value_0, value_0.clone());
        assert_eq!(value_0.clone(), value_0);
        assert_ne!(value_0, value_1);
        assert_ne!(value_0, value_1.clone());
        assert_eq!(value_1.clone(), value_1.clone());
    }

    #[test]
    fn test_value_display_and_debug() {
        assert_eq!(format!("{}", Value::r#bool(true)), "true");
        assert_eq!(format!("{}", Value::i64(42)), "42");
        assert_eq!(format!("{}", Value::i64_list([1, 2, 3])), "[1, 2, 3]");
        assert_eq!(format!("{}", Value::f32(3.14)), "3.14");
        assert_eq!(format!("{}", Value::string("hello")), "\"hello\"");
        assert_eq!(format!("{:?}", Value::r#bool(true)), "Bool(true)");
        assert_eq!(format!("{:?}", Value::i64(42)), "I64(42)");
        assert_eq!(format!("{:?}", Value::i64_list([1, 2, 3])), "I64List([1, 2, 3])");
        assert_eq!(format!("{:?}", Value::f32(3.14)), "F32(3.14)");
        assert_eq!(format!("{:?}", Value::string("hello")), "String(\"hello\")");
    }

    #[test]
    fn test_named_value() {
        let value_0 = NamedValue::new("bool", true);
        let value_1 = NamedValue::new("list", [3, 2, 2]);
        let value_0_roundtripped = unsafe { NamedValue::from_c_api(&value_0.to_c_api()) };
        let value_1_roundtripped = unsafe { NamedValue::from_c_api(&value_1.to_c_api()) };
        assert_eq!(value_0, value_0);
        assert_eq!(value_0, value_0_roundtripped);
        assert_eq!(value_0_roundtripped, value_0_roundtripped);
        assert_ne!(value_0, value_1);
        assert_ne!(value_0, value_1_roundtripped);
        assert_eq!(value_1, value_1_roundtripped);
        assert_ne!(value_1_roundtripped, value_0);

        let value_0 = NamedValue::new("value_0", vec![1, 2, 3]);
        let value_1 = NamedValue::new("value_1", Value::string("hello"));
        assert_eq!(value_0, value_0);
        assert_eq!(value_0, value_0.clone());
        assert_eq!(value_0.clone(), value_0);
        assert_ne!(value_0, value_1);
        assert_ne!(value_0, value_1.clone());
        assert_eq!(value_1.clone(), value_1.clone());
    }

    #[test]
    fn test_named_value_display_and_debug() {
        let value_0 = NamedValue::new("value_0", vec![1, 2, 3]);
        let value_1 = NamedValue::new("value_1", Value::string("hello"));
        assert_eq!(format!("{value_0}"), "value_0: [1, 2, 3]");
        assert_eq!(format!("{value_1}"), "value_1: \"hello\"");
        assert_eq!(format!("{value_0:?}"), "NamedValue { name: \"value_0\", value: I64List([1, 2, 3]) }");
        assert_eq!(format!("{value_1:?}"), "NamedValue { name: \"value_1\", value: String(\"hello\") }");
    }
}
