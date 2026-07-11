#![allow(non_camel_case_types, non_snake_case)]

#[repr(C)]
pub struct RYFT_XLA_Profiler_XSpace_To_Profiled_Instructions_Args {
    pub xspace: *const u8,
    pub xspace_size: usize,
    pub profile: *mut u8,
    pub profile_size: usize,
    pub error: *mut u8,
    pub error_size: usize,
}

impl RYFT_XLA_Profiler_XSpace_To_Profiled_Instructions_Args {
    pub fn new(xspace: &[u8]) -> Self {
        Self {
            xspace: xspace.as_ptr(),
            xspace_size: xspace.len(),
            profile: std::ptr::null_mut(),
            profile_size: 0,
            error: std::ptr::null_mut(),
            error_size: 0,
        }
    }
}

unsafe extern "C" {
    pub fn RYFT_XLA_Profiler_XSpace_To_Profiled_Instructions(
        args: *mut RYFT_XLA_Profiler_XSpace_To_Profiled_Instructions_Args,
    );
}

#[repr(C)]
pub struct RYFT_XLA_Profiler_Aggregate_Profiled_Instructions_Args {
    pub profiles: *const *const u8,
    pub profile_sizes: *const usize,
    pub profile_count: usize,
    pub percentile: i32,
    pub profile: *mut u8,
    pub profile_size: usize,
    pub error: *mut u8,
    pub error_size: usize,
}

impl RYFT_XLA_Profiler_Aggregate_Profiled_Instructions_Args {
    pub fn new(profiles: &[*const u8], profile_sizes: &[usize], percentile: i32) -> Self {
        Self {
            profiles: profiles.as_ptr(),
            profile_sizes: profile_sizes.as_ptr(),
            profile_count: profiles.len(),
            percentile,
            profile: std::ptr::null_mut(),
            profile_size: 0,
            error: std::ptr::null_mut(),
            error_size: 0,
        }
    }
}

unsafe extern "C" {
    pub fn RYFT_XLA_Profiler_Aggregate_Profiled_Instructions(
        args: *mut RYFT_XLA_Profiler_Aggregate_Profiled_Instructions_Args,
    );

    pub fn RYFT_XLA_Profiler_Byte_Buffer_Destroy(buffer: *mut u8);
}
