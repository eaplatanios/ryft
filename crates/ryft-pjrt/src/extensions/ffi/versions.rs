use std::fmt::Display;

/// XLA FFI API [`Version`] that this library has been built for.
pub static VERSION: Version = Version { major: ffi::XLA_FFI_API_VERSION_MAJOR, minor: ffi::XLA_FFI_API_VERSION_MINOR };

/// Represents the version of the XLA FFI API. XLA FFI provides a stable binary API for registering custom calls
/// with the XLA runtime. The XLA runtime guarantees that old API versions are supported for at least 12 months, after
/// which point the FFI library has to be recompiled with the latest XLA FFI headers to support new features.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    /// Major version number. This number is incremented when an ABI-incompatible change is made to the XLA FFI
    /// interface. Such changes include deleting a method or an argument, changing the type of an argument, re-arranging
    /// fields in any of the XLA FFI C API structs, etc.
    pub major: usize,

    /// Minor version number. This number is incremented when the XLA FFI interface is updated in a way that is
    /// potentially ABI-compatible with older versions, if supported by the caller and/or the implementation. Such
    /// changes include adding a new field to any of the PJRT C API structs, renaming a method or argument, etc.
    pub minor: usize,
}

impl Display for Version {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}", self.major, self.minor)
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::extensions::ffi::handlers::ffi::XLA_FFI_Extension_Base;

    pub const XLA_FFI_API_VERSION_MAJOR: usize = 0;
    pub const XLA_FFI_API_VERSION_MINOR: usize = 3;

    #[repr(C)]
    pub struct XLA_FFI_Api_Version {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub major_version: std::ffi::c_int,
        pub minor_version: std::ffi::c_int,
    }

    impl XLA_FFI_Api_Version {
        pub fn new(major_version: std::ffi::c_int, minor_version: std::ffi::c_int) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), major_version, minor_version }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::extensions::ffi::tests::test_ffi_api;

    use super::VERSION;

    #[test]
    fn test_version_display() {
        assert_eq!(format!("{VERSION}"), "0.3");
    }

    #[test]
    fn test_ffi_api_version() {
        assert_eq!(test_ffi_api().version(), VERSION);
    }
}
