#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::marker::{PhantomData, PhantomPinned};

// We represent opaque C types as structs with a particular structure that is following the convention
// suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
#[repr(C)]
pub struct PJRT_Error {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

// We represent opaque C types as structs with a particular structure that is following the convention
// suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
#[repr(C)]
pub struct PJRT_Distributed_Runtime_Service {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

#[repr(C)]
pub struct PJRT_Distributed_Runtime_Service_New_Args {
    pub address: *const std::ffi::c_char,
    pub num_nodes: u32,
    pub heartbeat_timeout: u32,
    pub cluster_register_timeout: u32,
    pub shutdown_timeout: u32,
    pub service: *mut PJRT_Distributed_Runtime_Service,
}

impl PJRT_Distributed_Runtime_Service_New_Args {
    pub fn new(
        address: *const std::ffi::c_char,
        num_nodes: u32,
        heartbeat_timeout: u32,
        cluster_register_timeout: u32,
        shutdown_timeout: u32,
    ) -> Self {
        Self {
            address,
            num_nodes,
            heartbeat_timeout,
            cluster_register_timeout,
            shutdown_timeout,
            service: std::ptr::null_mut(),
        }
    }
}

unsafe extern "C" {
    pub fn PJRT_Distributed_Runtime_Service_New(
        args: *mut PJRT_Distributed_Runtime_Service_New_Args,
    ) -> *mut PJRT_Error;
}

#[repr(C)]
pub struct PJRT_Distributed_Runtime_Service_Shutdown_Args {
    pub service: *mut PJRT_Distributed_Runtime_Service,
}

impl PJRT_Distributed_Runtime_Service_Shutdown_Args {
    pub fn new(service: *mut PJRT_Distributed_Runtime_Service) -> Self {
        Self { service }
    }
}

unsafe extern "C" {
    pub fn PJRT_Distributed_Runtime_Service_Shutdown(
        args: *mut PJRT_Distributed_Runtime_Service_Shutdown_Args,
    ) -> *mut PJRT_Error;
}

#[repr(C)]
pub struct PJRT_Distributed_Runtime_Service_Destroy_Args {
    pub service: *mut PJRT_Distributed_Runtime_Service,
}

impl PJRT_Distributed_Runtime_Service_Destroy_Args {
    pub fn new(service: *mut PJRT_Distributed_Runtime_Service) -> Self {
        Self { service }
    }
}

unsafe extern "C" {
    pub fn PJRT_Distributed_Runtime_Service_Destroy(
        args: *mut PJRT_Distributed_Runtime_Service_Destroy_Args,
    ) -> *mut PJRT_Error;
}

// We represent opaque C types as structs with a particular structure that is following the convention
// suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
#[repr(C)]
pub struct PJRT_Distributed_Runtime_Client {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

pub type PJRT_Distributed_Missed_Heartbeat_Callback =
    unsafe extern "C" fn(error: *const PJRT_Error, user_arg: *mut std::ffi::c_void);

#[repr(C)]
pub struct PJRT_Distributed_Runtime_Client_New_Args {
    pub address: *const std::ffi::c_char,
    pub node_id: u32,
    pub rpc_timeout: u32,
    pub init_timeout: u32,
    pub shutdown_timeout: u32,
    pub heartbeat_timeout: u32,
    pub missed_heartbeat_callback: PJRT_Distributed_Missed_Heartbeat_Callback,
    pub missed_heartbeat_callback_user_arg: *mut std::ffi::c_void,
    pub shutdown_on_destruction: bool,
    pub recoverable: bool,
    pub use_compression: bool,
    pub client: *mut PJRT_Distributed_Runtime_Client,
}

impl PJRT_Distributed_Runtime_Client_New_Args {
    pub fn new(
        address: *const std::ffi::c_char,
        node_id: u32,
        rpc_timeout: u32,
        init_timeout: u32,
        shutdown_timeout: u32,
        heartbeat_timeout: u32,
        missed_heartbeat_callback: PJRT_Distributed_Missed_Heartbeat_Callback,
        missed_heartbeat_callback_user_arg: *mut std::ffi::c_void,
        shutdown_on_destruction: bool,
        recoverable: bool,
        use_compression: bool,
    ) -> Self {
        Self {
            address,
            node_id,
            rpc_timeout,
            init_timeout,
            shutdown_timeout,
            heartbeat_timeout,
            missed_heartbeat_callback,
            missed_heartbeat_callback_user_arg,
            shutdown_on_destruction,
            recoverable,
            use_compression,
            client: std::ptr::null_mut(),
        }
    }
}

unsafe extern "C" {
    pub fn PJRT_Distributed_Runtime_Client_New(args: *mut PJRT_Distributed_Runtime_Client_New_Args) -> *mut PJRT_Error;
}

#[repr(C)]
pub struct PJRT_Distributed_Runtime_Client_Connect_Args {
    pub client: *mut PJRT_Distributed_Runtime_Client,
}

impl PJRT_Distributed_Runtime_Client_Connect_Args {
    pub fn new(client: *mut PJRT_Distributed_Runtime_Client) -> Self {
        Self { client }
    }
}

unsafe extern "C" {
    pub fn PJRT_Distributed_Runtime_Client_Connect(
        args: *mut PJRT_Distributed_Runtime_Client_Connect_Args,
    ) -> *mut PJRT_Error;
}

#[repr(C)]
pub struct PJRT_Distributed_Runtime_Client_Blocking_Key_Value_Get_Args {
    pub client: *mut PJRT_Distributed_Runtime_Client,
    pub key: *const std::ffi::c_char,
    pub timeout: u32,
    // The caller takes ownership of the returned pointer.
    pub value: *mut std::ffi::c_char,
}

impl PJRT_Distributed_Runtime_Client_Blocking_Key_Value_Get_Args {
    pub fn new(client: *mut PJRT_Distributed_Runtime_Client, key: *const std::ffi::c_char, timeout: u32) -> Self {
        Self { client, key, timeout, value: std::ptr::null_mut() }
    }
}

unsafe extern "C" {
    pub fn PJRT_Distributed_Runtime_Client_Blocking_Key_Value_Get(
        args: *mut PJRT_Distributed_Runtime_Client_Blocking_Key_Value_Get_Args,
    ) -> *mut PJRT_Error;
}

#[repr(C)]
pub struct PJRT_Distributed_Runtime_Client_Key_Value_Try_Get_Args {
    pub client: *mut PJRT_Distributed_Runtime_Client,
    pub key: *const std::ffi::c_char,
    // The caller takes ownership of the returned pointer.
    pub value: *mut std::ffi::c_char,
}

impl PJRT_Distributed_Runtime_Client_Key_Value_Try_Get_Args {
    pub fn new(client: *mut PJRT_Distributed_Runtime_Client, key: *const std::ffi::c_char) -> Self {
        Self { client, key, value: std::ptr::null_mut() }
    }
}

unsafe extern "C" {
    pub fn PJRT_Distributed_Runtime_Client_Key_Value_Try_Get(
        args: *mut PJRT_Distributed_Runtime_Client_Key_Value_Try_Get_Args,
    ) -> *mut PJRT_Error;
}

#[repr(C)]
pub struct PJRT_Distributed_Runtime_Client_Key_Value_Set_Args {
    pub client: *mut PJRT_Distributed_Runtime_Client,
    pub key: *const std::ffi::c_char,
    pub value: *const std::ffi::c_char,
}

impl PJRT_Distributed_Runtime_Client_Key_Value_Set_Args {
    pub fn new(
        client: *mut PJRT_Distributed_Runtime_Client,
        key: *const std::ffi::c_char,
        value: *const std::ffi::c_char,
    ) -> Self {
        Self { client, key, value }
    }
}

unsafe extern "C" {
    pub fn PJRT_Distributed_Runtime_Client_Key_Value_Set(
        args: *mut PJRT_Distributed_Runtime_Client_Key_Value_Set_Args,
    ) -> *mut PJRT_Error;
}

#[repr(C)]
pub struct PJRT_Distributed_Runtime_Client_Shutdown_Args {
    pub client: *mut PJRT_Distributed_Runtime_Client,
}

impl PJRT_Distributed_Runtime_Client_Shutdown_Args {
    pub fn new(client: *mut PJRT_Distributed_Runtime_Client) -> Self {
        Self { client }
    }
}

unsafe extern "C" {
    pub fn PJRT_Distributed_Runtime_Client_Shutdown(
        args: *mut PJRT_Distributed_Runtime_Client_Shutdown_Args,
    ) -> *mut PJRT_Error;
}

#[repr(C)]
pub struct PJRT_Distributed_Runtime_Client_Destroy_Args {
    pub client: *mut PJRT_Distributed_Runtime_Client,
}

impl PJRT_Distributed_Runtime_Client_Destroy_Args {
    pub fn new(client: *mut PJRT_Distributed_Runtime_Client) -> Self {
        Self { client }
    }
}

unsafe extern "C" {
    pub fn PJRT_Distributed_Runtime_Client_Destroy(
        args: *mut PJRT_Distributed_Runtime_Client_Destroy_Args,
    ) -> *mut PJRT_Error;
}
