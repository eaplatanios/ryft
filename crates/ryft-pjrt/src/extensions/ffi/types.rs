use crate::extensions::ffi::errors::FfiError;
use crate::extensions::ffi::handlers::FfiApi;
use crate::invoke_xla_ffi_api_error_fn;

/// [`FfiTypeId`] uniquely identifies a user-defined type in a given XLA FFI instance.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FfiTypeId(i64);

impl FfiTypeId {
    /// Returns the [`XLA_FFI_TypeId`](ffi::XLA_FFI_TypeId) that corresponds to this [`FfiTypeId`].
    pub unsafe fn to_c_api(self) -> ffi::XLA_FFI_TypeId {
        ffi::XLA_FFI_TypeId::new(self.0)
    }

    /// [`FfiTypeId`] that represents an _unknown_ type. This value can be used to request automatic type ID allocation
    /// by the XLA runtime when invoking [`Client::register_ffi_type`](crate::Client::register_ffi_type).
    pub const UNKNOWN: Self = Self(0);

    /// Creates a new [`FfiTypeId`].
    pub const fn new(type_id: i64) -> Self {
        Self(type_id)
    }

    /// Returns `true` if and only if this [`FfiTypeId`] is equal to [`FfiTypeId::UNKNOWN`].
    pub const fn is_unknown(self) -> bool {
        self.0 == 0
    }
}

impl Default for FfiTypeId {
    fn default() -> Self {
        Self::UNKNOWN
    }
}

impl From<i64> for FfiTypeId {
    fn from(value: i64) -> Self {
        Self::new(value)
    }
}

impl From<FfiTypeId> for i64 {
    fn from(value: FfiTypeId) -> Self {
        value.0
    }
}

/// [`FfiTypeInformation`] contains function pointers that are required by the XLA runtime for manipulating values
/// that correspond to user-defined types. For example, stateful handlers must tell the XLA runtime how to destroy
/// their state when an executable is being destroyed.
pub struct FfiTypeInformation {
    /// Optional callback used by the runtime to destroy an instance of this type.
    pub(crate) deleter: Option<unsafe extern "C" fn(object: *mut std::ffi::c_void)>,
}

impl FfiTypeInformation {
    /// Returns the [`XLA_FFI_TypeInfo`](ffi::XLA_FFI_TypeInfo) that corresponds to this [`FfiTypeInformation`].
    pub unsafe fn to_c_api(self) -> ffi::XLA_FFI_TypeInfo {
        ffi::XLA_FFI_TypeInfo::new(self.deleter)
    }

    /// Creates a new [`FfiTypeInformation`].
    pub const fn new(deleter: Option<unsafe extern "C" fn(object: *mut std::ffi::c_void)>) -> Self {
        Self { deleter }
    }

    /// Returns the optional deleter callback of this [`FfiTypeInformation`].
    pub const fn deleter(self) -> Option<unsafe extern "C" fn(object: *mut std::ffi::c_void)> {
        self.deleter
    }
}

impl FfiApi {
    /// Registers a user-defined type with the provided name and ID in the XLA FFI runtime type registry. If the
    /// provided ID is [`FfiTypeId::UNKNOWN`] then XLA will assign a unique type ID which will be returned by this
    /// function. Otherwise, XLA will verify that the provided type ID is unique and matches the type ID of the type
    /// registered under the same name, if a type with that name has already been registered.
    pub fn register_type<T: AsRef<str>>(
        &self,
        name: T,
        id: FfiTypeId,
        information: FfiTypeInformation,
    ) -> Result<FfiTypeId, FfiError> {
        use ffi::XLA_FFI_Type_Register_Args;
        let type_name = name.as_ref();
        let mut type_id = unsafe { id.to_c_api() };
        let type_information = unsafe { information.to_c_api() };
        invoke_xla_ffi_api_error_fn!(
            *self,
            XLA_FFI_Type_Register,
            {
                name = type_name.into(),
                type_id = &mut type_id as *mut _,
                type_info = &type_information as *const _,
            },
        )?;
        Ok(FfiTypeId::new(type_id.type_id))
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::extensions::ffi::attributes::ffi::XLA_FFI_ByteSpan;
    use crate::extensions::ffi::errors::ffi::XLA_FFI_Error;
    use crate::extensions::ffi::handlers::ffi::XLA_FFI_Extension_Base;

    #[repr(C)]
    pub struct XLA_FFI_TypeId {
        pub type_id: i64,
    }

    impl XLA_FFI_TypeId {
        pub const fn new(type_id: i64) -> Self {
            Self { type_id }
        }
    }

    #[repr(C)]
    pub struct XLA_FFI_TypeInfo {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub deleter: Option<unsafe extern "C" fn(object: *mut std::ffi::c_void)>,
    }

    impl XLA_FFI_TypeInfo {
        pub fn new(deleter: Option<unsafe extern "C" fn(object: *mut std::ffi::c_void)>) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), deleter }
        }
    }

    #[repr(C)]
    pub struct XLA_FFI_Type_Register_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub name: XLA_FFI_ByteSpan,
        pub type_id: *mut XLA_FFI_TypeId,
        pub type_info: *const XLA_FFI_TypeInfo,
    }

    impl XLA_FFI_Type_Register_Args {
        pub fn new(name: XLA_FFI_ByteSpan, type_id: *mut XLA_FFI_TypeId, type_info: *const XLA_FFI_TypeInfo) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), name, type_id, type_info }
        }
    }

    pub type XLA_FFI_Type_Register = unsafe extern "C" fn(args: *mut XLA_FFI_Type_Register_Args) -> *mut XLA_FFI_Error;
}

#[cfg(test)]
mod tests {
    use crate::extensions::ffi::tests::test_ffi_api;

    use super::{FfiTypeId, FfiTypeInformation, ffi};

    #[test]
    fn test_type_id() {
        let type_id = FfiTypeId::new(42);
        assert_eq!(i64::from(type_id), 42);
        assert!(!type_id.is_unknown());
        assert_eq!(unsafe { type_id.to_c_api() }.type_id, 42);
        assert!(FfiTypeId::UNKNOWN.is_unknown());
    }

    #[test]
    fn test_type_information() {
        let type_information = FfiTypeInformation::new(None);
        assert!(type_information.deleter().is_none());

        unsafe extern "C" fn dummy_deleter(_object: *mut std::ffi::c_void) {}

        let type_information = FfiTypeInformation::new(Some(dummy_deleter));
        assert!(type_information.deleter().is_some());

        let type_information = FfiTypeInformation::new(Some(dummy_deleter));
        let pjrt_ffi_type_information = unsafe { type_information.to_c_api() };
        assert!(pjrt_ffi_type_information.deleter.is_some());

        let type_information = FfiTypeInformation::new(Some(dummy_deleter));
        let xla_ffi_type_information = unsafe { type_information.to_c_api() };
        assert_eq!(xla_ffi_type_information.struct_size, size_of::<ffi::XLA_FFI_TypeInfo>());
        assert!(xla_ffi_type_information.extension_start.is_null());
        assert!(xla_ffi_type_information.deleter.is_some());
    }

    #[test]
    fn test_register_type() {
        let api = test_ffi_api();
        let type_name = "ryft.test.ffi.type";
        let type_id = api
            .register_type(type_name, FfiTypeId::UNKNOWN, FfiTypeInformation::new(None))
            .expect("failed to register XLA FFI type");
        assert!(!type_id.is_unknown());
        assert_eq!(api.register_type(type_name, type_id, FfiTypeInformation::new(None)), Ok(type_id));
    }
}
