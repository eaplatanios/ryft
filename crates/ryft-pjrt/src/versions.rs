use std::fmt::Display;

/// PJRT API [`Version`] that this library has been built for.
pub static VERSION: Version = Version { major: ffi::PJRT_API_MAJOR as usize, minor: ffi::PJRT_API_MINOR as usize };

/// Represents the version of a PJRT API. Specifically, callers can check for forward compatibility of a PJRT plugin
/// by using [`Api::version`](crate::api::Api::version) to check if the implementation is aware of newer interface
/// additions. Refer to the [official documentation on PJRT API ABI versioning and compatibility](
/// https://docs.google.com/document/d/1TKB5NyGtdzrpgw5mpyFjVAhJjpSNdF31T6pjPl_UT2o) for more information.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    /// Major version number. This number is incremented when an ABI-incompatible change is made to the PJRT interface.
    /// Such changes include deleting a method or an argument, changing the type of an argument, re-arranging fields
    /// in any of the PJRT C API structs, etc.
    pub major: usize,

    /// Minor version number. This number is incremented when the PJRT interface is updated in a way that is potentially
    /// ABI-compatible with older versions, if supported by the caller and/or the implementation. Such changes include
    /// adding a new field to any of the PJRT C API structs, renaming a method or argument, etc.
    pub minor: usize,
}

impl Display for Version {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}", self.major, self.minor)
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::ffi::PJRT_Extension_Base;

    pub const PJRT_API_MAJOR: u32 = 0;
    pub const PJRT_API_MINOR: u32 = 91;

    #[repr(C)]
    pub struct PJRT_Api_Version {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub major_version: std::ffi::c_int,
        pub minor_version: std::ffi::c_int,
    }

    impl PJRT_Api_Version {
        pub fn new(major_version: std::ffi::c_int, minor_version: std::ffi::c_int) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), major_version, minor_version }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::test_cpu_client;

    use super::VERSION;

    #[test]
    fn test_client_version() {
        assert_eq!(test_cpu_client().version(), VERSION);
    }

    #[test]
    fn test_version_display() {
        assert_eq!(format!("{VERSION}"), "0.91");
    }
}
