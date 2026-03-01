use std::marker::PhantomData;

use crate::extensions::ffi::FfiExecutionStage;
use crate::extensions::ffi::errors::FfiError;
use crate::extensions::ffi::handlers::FfiApi;
use crate::extensions::ffi::types::FfiTypeId;
use crate::invoke_xla_ffi_api_error_fn;

/// Function pointer for a task that is to be scheduled on a thread pool. The XLA runtime will call this function with
/// a user-defined data pointer on one of the runtime-managed threads. For CPU backends, the task will be invoked on a
/// thread pool that runs all compute tasks (i.e., an [Eigen](https://libeigen.gitlab.io/) thread pool).
///
/// Users must not rely on any particular execution order or on the number of available threads in the thread pool.
/// Tasks can be executed in the caller thread or in a thread pool with size `1` and it is unsafe to assume that all
/// scheduled tasks can be executed in parallel.
pub type FfiTask = ffi::XLA_FFI_Task;

/// Unique identifier of a particular _logical execution_ of an XLA module/program. A _logical execution_ might
/// encompass multiple executions of one or more HLO modules. Runs that are part of the same logical execution can
/// communicate via collective operations, whereas runs that are part of different logical executions are isolated.
pub type FfiRunId = i64;

/// Unique identifier of a device that XLA uses to run computations. This ID is an _ordinal_ ID whose value ranges from
/// `0` to `device_count - 1`. It represents a _logical_ device ordinal ID since multiple logical devices could reside
/// on the same _physical_ device (e.g., this is the case with virtual GPUs). A value of `-1` indicates that the target
/// device has not been set.
pub type FfiDeviceId = i32;

/// Represents a platform-specific stream (e.g., for CUDA this would be a `CUstream`).
pub type FfiStream = *mut std::ffi::c_void;

/// Represents the execution state associated with a specific type for a stateful
/// XLA [`FfiHandler`](crate::extensions::ffi::FfiHandler).
pub type FfiExecutionState = *mut std::ffi::c_void;

/// User data that can be attached to an [`ExecutionContext`](crate::ExecutionContext) and consumed
/// by [`FfiHandler`](crate::extensions::ffi::FfiHandler)s.
pub struct FfiUserData {
    /// [`FfiTypeId`] that corresponds to the type of this [`FfiUserData`].
    pub type_id: FfiTypeId,

    /// Pointer to the underlying data.
    pub data: *mut std::ffi::c_void,
}

impl FfiUserData {
    /// Returns the [`PJRT_FFI_UserData`](crate::extensions::ffi::ffi::PJRT_FFI_UserData) that corresponds to this
    /// [`FfiUserData`] and which can be passed to functions in the XLA FFI API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_c_api(self) -> crate::extensions::ffi::ffi::PJRT_FFI_UserData {
        crate::extensions::ffi::ffi::PJRT_FFI_UserData { type_id: self.type_id.into(), data: self.data }
    }

    /// Creates a new [`FfiUserData`] instance.
    pub fn new(type_id: FfiTypeId, data: *mut std::ffi::c_void) -> Self {
        Self { type_id, data }
    }
}

/// XLA execution context that provides access to per-invocation state
/// for [`FfiHandler`](crate::extensions::ffi::FfiHandler)s.
pub struct FfiExecutionContext<'o> {
    /// Handle that represents this [`FfiExecutionContext`] in the XLA FFI API.
    handle: *mut ffi::XLA_FFI_ExecutionContext,

    /// Underlying XLA [`FfiApi`].
    api: FfiApi,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`FfiExecutionContext`].
    owner: PhantomData<&'o mut ffi::XLA_FFI_ExecutionContext>,
}

impl<'o> FfiExecutionContext<'o> {
    /// Constructs a new [`FfiExecutionContext`] from the provided
    /// [`XLA_FFI_ExecutionContext`](ffi::XLA_FFI_ExecutionContext) handle that came
    /// from a function in the XLA FFI API.
    pub unsafe fn from_c_api(handle: *mut ffi::XLA_FFI_ExecutionContext, api: FfiApi) -> Result<Self, FfiError> {
        if handle.is_null() {
            Err(FfiError::invalid_argument("the provided XLA FFI execution context handle is a null pointer"))
        } else {
            Ok(Self { handle, api, owner: PhantomData })
        }
    }

    /// Returns the [`XLA_FFI_ExecutionContext`](ffi::XLA_FFI_ExecutionContext) that corresponds to this
    /// [`FfiExecutionContext`] and which can be passed to functions in the XLA FFI API.
    pub unsafe fn to_c_api(&self) -> *mut ffi::XLA_FFI_ExecutionContext {
        self.handle
    }

    /// Returns the underlying XLA [`FfiApi`].
    pub fn api(&self) -> FfiApi {
        self.api
    }

    /// Returns opaque user data from this [`FfiExecutionContext`] for the provided `type_id`.
    pub fn user_data(&self, type_id: FfiTypeId) -> Result<FfiUserData, FfiError> {
        use ffi::XLA_FFI_ExecutionContext_Get_Args;
        let mut type_id_handle = unsafe { type_id.to_c_api() };
        invoke_xla_ffi_api_error_fn!(
            self.api,
            XLA_FFI_ExecutionContext_Get,
            { context = self.handle, type_id = &mut type_id_handle as *mut _ },
            { data },
        )
        .map(|data| FfiUserData { type_id, data })
    }

    /// Sets [`FfiExecutionState`] for the provided [`FfiExecutionStage`] associated with the provided [`FfiTypeId`],
    /// for this [`FfiExecutionContext`]. Note that this function will return an [`FfiError`] if the state for the
    /// specified type has already been set, or if the provided `stage` is the [`FfiExecutionStage::Execution`]
    /// [stage](https://github.com/openxla/xla/blob/964a0a45a0c3090cd484a3c51e8f9d05ed10b968/xla/ffi/ffi_api.cc#L327).
    ///
    /// # Safety
    ///
    /// The caller must ensure that `state` remains valid for the full duration for which the XLA runtime might access
    /// it through this execution context.
    pub unsafe fn set_state(
        &self,
        stage: FfiExecutionStage,
        type_id: FfiTypeId,
        state: FfiExecutionState,
    ) -> Result<(), FfiError> {
        use ffi::XLA_FFI_State_Set_Args;
        if stage == FfiExecutionStage::Execution {
            return Err(FfiError::invalid_argument("execution state cannot be set for the execution stage"));
        };
        let mut type_id_handle = unsafe { type_id.to_c_api() };
        invoke_xla_ffi_api_error_fn!(
            self.api,
            XLA_FFI_State_Set,
            {
                context = self.handle,
                stage = stage.to_c_api(),
                type_id = &mut type_id_handle as *mut _,
                state = state,
            },
        )
    }

    /// Returns the [`FfiExecutionState`] for the provided [`FfiExecutionStage`] associated with the provided
    /// [`FfiTypeId`] for this [`FfiExecutionContext`]. Note that this function will return an [`FfiError`] if
    /// the state for the specified type has not been set, if it has been set to a value of a different type,
    /// or if the provided `stage` is the [`FfiExecutionStage::Execution`]
    /// [stage](https://github.com/openxla/xla/blob/964a0a45a0c3090cd484a3c51e8f9d05ed10b968/xla/ffi/ffi_api.cc#L327).
    pub fn state(&self, stage: FfiExecutionStage, type_id: FfiTypeId) -> Result<FfiExecutionState, FfiError> {
        use ffi::XLA_FFI_State_Get_Args;
        if stage == FfiExecutionStage::Execution {
            return Err(FfiError::invalid_argument("execution state cannot be obtained for the execution stage"));
        };
        let mut type_id_handle = unsafe { type_id.to_c_api() };
        invoke_xla_ffi_api_error_fn!(
            self.api,
            XLA_FFI_State_Get,
            { context = self.handle, stage = stage.to_c_api(), type_id = &mut type_id_handle as *mut _ },
            { state },
        )
    }

    /// Returns the underlying [`FfiStream`] associated with this [`FfiExecutionContext`].
    pub fn stream(&self) -> Result<FfiStream, FfiError> {
        use ffi::XLA_FFI_Stream_Get_Args;
        invoke_xla_ffi_api_error_fn!(self.api, XLA_FFI_Stream_Get, { context = self.handle }, { stream })
    }

    /// Allocates a block of memory on the device that is bound to this [`FfiExecutionContext`].
    pub fn allocate_device_memory(&self, size: usize, alignment: usize) -> Result<*mut std::ffi::c_void, FfiError> {
        use ffi::XLA_FFI_DeviceMemory_Allocate_Args;
        invoke_xla_ffi_api_error_fn!(
            self.api,
            XLA_FFI_DeviceMemory_Allocate,
            { context = self.handle, size = size, alignment = alignment },
            { data },
        )
    }

    /// Frees a previously-allocated block of memory on the device that is bound to this [`FfiExecutionContext`].
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data` points to memory that was allocated by [`Self::allocate_device_memory`]
    /// in this execution context and that `size` matches the allocation size.
    pub unsafe fn free_device_memory(&self, size: usize, data: *mut std::ffi::c_void) -> Result<(), FfiError> {
        use ffi::XLA_FFI_DeviceMemory_Free_Args;
        invoke_xla_ffi_api_error_fn!(
            self.api,
            XLA_FFI_DeviceMemory_Free,
            { context = self.handle, size = size, data = data },
        )
    }

    /// Schedules the provided [`FfiTask`] for execution on the XLA runtime-managed thread pool. The task will be
    /// provided `data` when it is executed. Note that this function will return an [`FfiError`] if the XLA
    /// thread pool is not available.
    pub fn schedule_thread_pool_task(&self, task: FfiTask, data: *mut std::ffi::c_void) -> Result<(), FfiError> {
        use ffi::XLA_FFI_ThreadPool_Schedule_Args;
        invoke_xla_ffi_api_error_fn!(
            self.api,
            XLA_FFI_ThreadPool_Schedule,
            { context = self.handle, task = Some(task), data = data },
        )
    }

    /// Returns the number of threads in the XLA runtime-managed thread pool.
    pub fn thread_count(&self) -> Result<i64, FfiError> {
        use ffi::XLA_FFI_ThreadPool_NumThreads_Args;
        let mut thread_count = 0i64;
        invoke_xla_ffi_api_error_fn!(
            self.api,
            XLA_FFI_ThreadPool_NumThreads,
            { context = self.handle, num_threads = &mut thread_count as *mut _ },
        )?;
        Ok(thread_count)
    }

    // // Returns a unique identifier for the current logical execution.
    /// Returns the [`FfiRunId`] for the current _logical execution_.
    pub fn run_id(&self) -> Result<FfiRunId, FfiError> {
        use ffi::XLA_FFI_RunId_Get_Args;
        invoke_xla_ffi_api_error_fn!(self.api, XLA_FFI_RunId_Get, { context = self.handle }, { run_id })
    }

    /// Returns the [`FfiDeviceId`] for the current _logical execution_.
    pub fn device_ordinal(&self) -> Result<FfiDeviceId, FfiError> {
        use ffi::XLA_FFI_DeviceOrdinal_Get_Args;
        invoke_xla_ffi_api_error_fn!(self.api, XLA_FFI_DeviceOrdinal_Get, { context = self.handle }, { device_ordinal })
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::extensions::ffi::errors::ffi::XLA_FFI_Error;
    use crate::extensions::ffi::handlers::ffi::{XLA_FFI_ExecutionStage, XLA_FFI_Extension_Base};
    use crate::extensions::ffi::types::ffi::XLA_FFI_TypeId;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct XLA_FFI_ExecutionContext {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct XLA_FFI_ExecutionContext_Get_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub context: *mut XLA_FFI_ExecutionContext,
        pub type_id: *mut XLA_FFI_TypeId,
        pub data: *mut std::ffi::c_void,
    }

    impl XLA_FFI_ExecutionContext_Get_Args {
        pub fn new(context: *mut XLA_FFI_ExecutionContext, type_id: *mut XLA_FFI_TypeId) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                context,
                type_id,
                data: std::ptr::null_mut(),
            }
        }
    }

    pub type XLA_FFI_ExecutionContext_Get =
        unsafe extern "C" fn(args: *mut XLA_FFI_ExecutionContext_Get_Args) -> *mut XLA_FFI_Error;

    #[repr(C)]
    pub struct XLA_FFI_State_Set_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub context: *mut XLA_FFI_ExecutionContext,
        pub stage: XLA_FFI_ExecutionStage,
        pub type_id: *mut XLA_FFI_TypeId,
        pub state: *mut std::ffi::c_void,
    }

    impl XLA_FFI_State_Set_Args {
        pub fn new(
            context: *mut XLA_FFI_ExecutionContext,
            stage: XLA_FFI_ExecutionStage,
            type_id: *mut XLA_FFI_TypeId,
            state: *mut std::ffi::c_void,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                context,
                stage,
                type_id,
                state,
            }
        }
    }

    pub type XLA_FFI_State_Set = unsafe extern "C" fn(args: *mut XLA_FFI_State_Set_Args) -> *mut XLA_FFI_Error;

    #[repr(C)]
    pub struct XLA_FFI_State_Get_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub context: *mut XLA_FFI_ExecutionContext,
        pub stage: XLA_FFI_ExecutionStage,
        pub type_id: *mut XLA_FFI_TypeId,
        pub state: *mut std::ffi::c_void,
    }

    impl XLA_FFI_State_Get_Args {
        pub fn new(
            context: *mut XLA_FFI_ExecutionContext,
            stage: XLA_FFI_ExecutionStage,
            type_id: *mut XLA_FFI_TypeId,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                context,
                stage,
                type_id,
                state: std::ptr::null_mut(),
            }
        }
    }

    pub type XLA_FFI_State_Get = unsafe extern "C" fn(args: *mut XLA_FFI_State_Get_Args) -> *mut XLA_FFI_Error;

    #[repr(C)]
    pub struct XLA_FFI_Stream_Get_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub context: *mut XLA_FFI_ExecutionContext,
        pub stream: *mut std::ffi::c_void,
    }

    impl XLA_FFI_Stream_Get_Args {
        pub fn new(context: *mut XLA_FFI_ExecutionContext) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                context,
                stream: std::ptr::null_mut(),
            }
        }
    }

    pub type XLA_FFI_Stream_Get = unsafe extern "C" fn(args: *mut XLA_FFI_Stream_Get_Args) -> *mut XLA_FFI_Error;

    #[repr(C)]
    pub struct XLA_FFI_DeviceMemory_Allocate_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub context: *mut XLA_FFI_ExecutionContext,
        pub size: usize,
        pub alignment: usize,
        pub data: *mut std::ffi::c_void,
    }

    impl XLA_FFI_DeviceMemory_Allocate_Args {
        pub fn new(context: *mut XLA_FFI_ExecutionContext, size: usize, alignment: usize) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                context,
                size,
                alignment,
                data: std::ptr::null_mut(),
            }
        }
    }

    pub type XLA_FFI_DeviceMemory_Allocate =
        unsafe extern "C" fn(args: *mut XLA_FFI_DeviceMemory_Allocate_Args) -> *mut XLA_FFI_Error;

    #[repr(C)]
    pub struct XLA_FFI_DeviceMemory_Free_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub context: *mut XLA_FFI_ExecutionContext,
        pub size: usize,
        pub data: *mut std::ffi::c_void,
    }

    impl XLA_FFI_DeviceMemory_Free_Args {
        pub fn new(context: *mut XLA_FFI_ExecutionContext, size: usize, data: *mut std::ffi::c_void) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), context, size, data }
        }
    }

    pub type XLA_FFI_DeviceMemory_Free =
        unsafe extern "C" fn(args: *mut XLA_FFI_DeviceMemory_Free_Args) -> *mut XLA_FFI_Error;

    pub type XLA_FFI_Task = unsafe extern "C" fn(data: *mut std::ffi::c_void);

    #[repr(C)]
    pub struct XLA_FFI_ThreadPool_Schedule_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub context: *mut XLA_FFI_ExecutionContext,
        pub task: Option<XLA_FFI_Task>,
        pub data: *mut std::ffi::c_void,
    }

    impl XLA_FFI_ThreadPool_Schedule_Args {
        pub fn new(
            context: *mut XLA_FFI_ExecutionContext,
            task: Option<XLA_FFI_Task>,
            data: *mut std::ffi::c_void,
        ) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), context, task, data }
        }
    }

    pub type XLA_FFI_ThreadPool_Schedule =
        unsafe extern "C" fn(args: *mut XLA_FFI_ThreadPool_Schedule_Args) -> *mut XLA_FFI_Error;

    #[repr(C)]
    pub struct XLA_FFI_ThreadPool_NumThreads_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub context: *mut XLA_FFI_ExecutionContext,
        pub num_threads: *mut i64,
    }

    impl XLA_FFI_ThreadPool_NumThreads_Args {
        pub fn new(context: *mut XLA_FFI_ExecutionContext, num_threads: *mut i64) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), context, num_threads }
        }
    }

    pub type XLA_FFI_ThreadPool_NumThreads =
        unsafe extern "C" fn(args: *mut XLA_FFI_ThreadPool_NumThreads_Args) -> *mut XLA_FFI_Error;

    #[repr(C)]
    pub struct XLA_FFI_RunId_Get_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub context: *mut XLA_FFI_ExecutionContext,
        pub run_id: i64,
    }

    impl XLA_FFI_RunId_Get_Args {
        pub fn new(context: *mut XLA_FFI_ExecutionContext) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), context, run_id: 0 }
        }
    }

    pub type XLA_FFI_RunId_Get = unsafe extern "C" fn(args: *mut XLA_FFI_RunId_Get_Args) -> *mut XLA_FFI_Error;

    #[repr(C)]
    pub struct XLA_FFI_DeviceOrdinal_Get_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub context: *mut XLA_FFI_ExecutionContext,
        pub device_ordinal: i32,
    }

    impl XLA_FFI_DeviceOrdinal_Get_Args {
        pub fn new(context: *mut XLA_FFI_ExecutionContext) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), context, device_ordinal: 0 }
        }
    }

    pub type XLA_FFI_DeviceOrdinal_Get =
        unsafe extern "C" fn(args: *mut XLA_FFI_DeviceOrdinal_Get_Args) -> *mut XLA_FFI_Error;
}

#[cfg(test)]
mod tests {
    use crate::extensions::ffi::FfiExecutionStage;
    use crate::extensions::ffi::errors::FfiError;
    use crate::extensions::ffi::tests::with_test_ffi_call_frame;
    use crate::extensions::ffi::types::FfiTypeId;

    #[test]
    fn test_ffi_execution_context() {
        with_test_ffi_call_frame(|call_frame| unsafe {
            let context = call_frame.context().unwrap();
            assert!(matches!(
                context.user_data(FfiTypeId::new(42)),
                Err(FfiError::Unknown { message, .. })
                  if message.contains("User data with type id 42 not found in execution context"),
            ));
            assert!(matches!(
                context.set_state(FfiExecutionStage::Execution, FfiTypeId::new(42), std::ptr::null_mut()),
                Err(FfiError::InvalidArgument { message, .. })
                  if message.contains("execution stage does not support execution-state get/set operations"),
            ));
            assert!(matches!(
                context.state(FfiExecutionStage::Execution, FfiTypeId::new(42)),
                Err(FfiError::InvalidArgument { message, .. })
                  if message.contains("execution stage does not support execution-state get/set operations"),
            ));
            assert!(matches!(
                context.set_state(FfiExecutionStage::Instantiation, FfiTypeId::new(42), std::ptr::null_mut()),
                Err(FfiError::Unknown { message, .. })
                  if message.contains("Type id 42 is not registered with a static registry"),
            ));
            assert!(matches!(
                context.state(FfiExecutionStage::Instantiation, FfiTypeId::new(42)),
                Err(FfiError::Unknown { message, .. }) if message.contains("State is not set"),
            ));
            assert!(matches!(
                context.stream(),
                Err(FfiError::Unknown { message, .. }) if message.contains("XLA FFI GPU context is not available"),
            ));
            assert!(matches!(
                context.allocate_device_memory(256, 16),
                Err(FfiError::Unknown { message, .. }) if message.contains("XLA FFI GPU context is not available"),
            ));
            assert!(matches!(
                context.free_device_memory(256, std::ptr::null_mut()),
                Err(FfiError::Unknown { message, .. }) if message.contains("XLA FFI GPU context is not available"),
            ));

            unsafe extern "C" fn task_callback(_data: *mut std::ffi::c_void) {}

            assert!(context.schedule_thread_pool_task(task_callback, std::ptr::null_mut()).is_ok());
            assert!(matches!(context.thread_count(), Ok(thread_count) if thread_count >= 1));
            assert!(matches!(context.run_id(), Ok(0)));
            assert!(matches!(context.device_ordinal(), Ok(0)));
        });
    }
}
