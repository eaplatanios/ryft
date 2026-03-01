use std::marker::PhantomData;
use std::ops::{BitOr, BitOrAssign};

use crate::extensions::ffi::buffers::{self, FfiBuffer};
use crate::extensions::ffi::context::FfiExecutionContext;
use crate::extensions::ffi::errors::FfiError;
use crate::extensions::ffi::futures::FfiFuture;
use crate::extensions::ffi::versions::Version;
use crate::extensions::ffi::{FfiTypeId, VERSION};

/// Wrapper of a [`XLA_FFI_Api`](ffi::XLA_FFI_Api) handle that can be used to interact with the XLA FFI API.
#[derive(Copy, Clone)]
pub struct FfiApi {
    /// Handle that represents this [`FfiApi`] in the XLA FFI API.
    handle: *const ffi::XLA_FFI_Api,
}

impl FfiApi {
    /// Constructs a new [`FfiApi`] from the provided [`XLA_FFI_Api`](ffi::XLA_FFI_Api) handle
    /// that came from a function in the XLA FFI API.
    pub(crate) unsafe fn from_c_api(handle: *const ffi::XLA_FFI_Api) -> Result<Self, FfiError> {
        if handle.is_null() {
            Err(FfiError::invalid_argument("the provided XLA FFI API handle is a null pointer"))
        } else {
            Ok(Self { handle })
        }
    }

    /// Returns the [`XLA_FFI_Api`](ffi::XLA_FFI_Api) that corresponds to this [`FfiApi`] and
    /// which can be passed to functions in the XLA FFI API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::XLA_FFI_Api {
        self.handle
    }

    /// Returns the XLA FFI API [`Version`] that this [`FfiApi`] supports.
    pub fn version(&self) -> Version {
        let handle = unsafe { &(*self.to_c_api()).api_version };
        Version { major: handle.major_version as usize, minor: handle.minor_version as usize }
    }
}

/// Represents the _execution stage_ of an XLA [`FfiHandler`] invocation. The XLA FFI runtime has multiple execution
/// stages, and it supports using different [`FfiHandler`]s for each stage by using [`FfiHandlerBundle`]s.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FfiExecutionStage {
    /// [`FfiHandler`]s registered for this [`FfiExecutionStage`] are invoked when an XLA executable is instantiated.
    /// Every call site will have its own "instance" of the [`FfiHandler`] and it is possible to attach an arbitrary
    /// user-defined state to the [`FfiHandler`] instance, and be able to access it during other [`FfiExecutionStage`]s.
    /// Any such state that is constructed during this stage is owned by the XLA runtime and will eventually be
    /// destroyed together with the parent executable.
    Instantiation,

    /// [`FfiHandler`]s registered for this [`FfiExecutionStage`] are invoked before the execution of an executable in
    /// order to enable any necessary preparation, including requesting resources from the XLA runtime (e.g., in the
    /// XLA GPU backend the preparation stage might be used to request collective cliques).
    Preparation,

    /// [`FfiHandler`]s registered for this [`FfiExecutionStage`] are invoked before the execution of an executable and
    /// after all resources from the preparation stage have been acquired.
    Initialization,

    /// [`FfiHandler`]s registered for this [`FfiExecutionStage`] are invoked when the corresponding custom call
    /// operation in the StableHLO program being executed, needs to be executed (i.e., when its inputs become ready).
    /// For GPU backends, [`FfiHandler`]s typically run on the host CPU and enqueue device work using the stream that
    /// they can obtain from the [`FfiExecutionContext`]. Note that, [`FfiHandler`]s may run during command-buffer
    /// capture (or CUDA graph capture), in which case the argument/input buffers may contain uninitialized values.
    /// This means that handlers should use the arguments as _device addresses_ to wire into enqueue GPU operations,
    /// and not as host-readable values. Specifically, handlers can obtain input shapes, data types, and other
    /// attributes from the XLA [`FfiCallFrame`], and then enqueue operations to the stream they obtain from the
    /// [`FfiExecutionContext`], but they *must not* attempt to dereference any input device buffers. This also means
    /// that they cannot have host-side control flow depend on the runtime values of those buffers.
    Execution,
}

impl FfiExecutionStage {
    /// Constructs a new [`FfiExecutionStage`] from the provided [`XLA_FFI_ExecutionStage`](ffi::XLA_FFI_ExecutionStage)
    /// that came from a function in the XLA FFI API.
    pub fn from_c_api(stage: ffi::XLA_FFI_ExecutionStage) -> Self {
        match stage {
            ffi::XLA_FFI_ExecutionStage::XLA_FFI_ExecutionStage_INSTANTIATE => Self::Instantiation,
            ffi::XLA_FFI_ExecutionStage::XLA_FFI_ExecutionStage_PREPARE => Self::Preparation,
            ffi::XLA_FFI_ExecutionStage::XLA_FFI_ExecutionStage_INITIALIZE => Self::Initialization,
            ffi::XLA_FFI_ExecutionStage::XLA_FFI_ExecutionStage_EXECUTE => Self::Execution,
        }
    }

    /// Returns the [`XLA_FFI_ExecutionStage`](ffi::XLA_FFI_ExecutionStage) that corresponds to this
    /// [`FfiExecutionStage`] and which can be passed to functions in the XLA FFI API.
    pub fn to_c_api(&self) -> ffi::XLA_FFI_ExecutionStage {
        match self {
            Self::Instantiation => ffi::XLA_FFI_ExecutionStage::XLA_FFI_ExecutionStage_INSTANTIATE,
            Self::Preparation => ffi::XLA_FFI_ExecutionStage::XLA_FFI_ExecutionStage_PREPARE,
            Self::Initialization => ffi::XLA_FFI_ExecutionStage::XLA_FFI_ExecutionStage_INITIALIZE,
            Self::Execution => ffi::XLA_FFI_ExecutionStage::XLA_FFI_ExecutionStage_EXECUTE,
        }
    }
}

/// Input to an [`FfiHandler`] that appears as part of the [`FfiCallFrame`] it is invoked with.
pub enum FfiInput<'o> {
    Buffer { buffer: FfiBuffer<'o> },
}

impl FfiInput<'_> {
    /// Constructs a new [`FfiInput`] from the provided [`XLA_FFI_ArgType`](ffi::XLA_FFI_ArgType)
    /// and raw data pointer that came from a function in the XLA FFI API.
    pub unsafe fn from_c_api(r#type: ffi::XLA_FFI_ArgType, value: *mut std::ffi::c_void) -> Result<Self, FfiError> {
        unsafe {
            match r#type {
                ffi::XLA_FFI_ArgType_BUFFER => {
                    let buffer = value as *const buffers::ffi::XLA_FFI_Buffer;
                    if buffer.is_null() {
                        return Err(FfiError::invalid_argument("encountered null buffer pointer for XLA FFI input"));
                    }
                    Ok(Self::Buffer { buffer: FfiBuffer::from_c_api(buffer)? })
                }
                _ => Err(FfiError::invalid_argument(format!("invalid XLA FFI input type '{}'", r#type))),
            }
        }
    }
}

/// Pre-allocated output of an [`FfiHandler`] that appears as part of the [`FfiCallFrame`] it is invoked with.
pub enum FfiOutput<'o> {
    Buffer { buffer: FfiBuffer<'o> },
}

impl FfiOutput<'_> {
    /// Constructs a new [`FfiOutput`] from the provided [`XLA_FFI_RetType`](ffi::XLA_FFI_RetType)
    /// and raw data pointer that came from a function in the XLA FFI API.
    pub unsafe fn from_c_api(r#type: ffi::XLA_FFI_RetType, value: *mut std::ffi::c_void) -> Result<Self, FfiError> {
        unsafe {
            match r#type {
                ffi::XLA_FFI_RetType_BUFFER => {
                    let buffer = value as *const buffers::ffi::XLA_FFI_Buffer;
                    if buffer.is_null() {
                        return Err(FfiError::invalid_argument("encountered null buffer pointer for XLA FFI output"));
                    }
                    Ok(Self::Buffer { buffer: FfiBuffer::from_c_api(buffer)? })
                }
                _ => Err(FfiError::invalid_argument(format!("invalid XLA FFI output type '{}'", r#type))),
            }
        }
    }
}

/// Represents an XLA FFI call frame for a single [`FfiHandler`] invocation (i.e., the full context in which the handler
/// is invoked along with its input buffers and pre-allocated output buffers).
pub struct FfiCallFrame<'o> {
    /// Handle that represents this [`FfiCallFrame`] in the XLA FFI API.
    handle: *mut ffi::XLA_FFI_CallFrame,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`FfiCallFrame`].
    owner: PhantomData<&'o ()>,
}

impl<'o> FfiCallFrame<'o> {
    /// Constructs a new [`FfiCallFrame`] from the provided [`XLA_FFI_CallFrame`](ffi::XLA_FFI_CallFrame)
    /// handle that came from a function in the XLA FFI API.
    pub unsafe fn from_c_api(handle: *mut ffi::XLA_FFI_CallFrame) -> Result<Self, FfiError> {
        if handle.is_null() {
            Err(FfiError::invalid_argument("the provided XLA FFI call frame handle is a null pointer"))
        } else {
            Ok(Self { handle, owner: PhantomData })
        }
    }

    /// Returns the [`XLA_FFI_CallFrame`](ffi::XLA_FFI_CallFrame) that corresponds to this [`FfiCallFrame`]
    /// and which can be passed to functions in the XLA FFI API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::XLA_FFI_CallFrame {
        self.handle
    }

    /// Returns the underlying XLA [`FfiApi`].
    pub fn api(&self) -> Result<FfiApi, FfiError> {
        unsafe { FfiApi::from_c_api((*self.handle).api) }
    }

    /// Returns the current [`FfiExecutionContext`] of this [`FfiCallFrame`].
    pub fn context(&self) -> Result<FfiExecutionContext<'o>, FfiError> {
        unsafe { FfiExecutionContext::from_c_api((*self.handle).context, self.api()?) }
    }

    /// Returns the current [`FfiExecutionStage`] of this [`FfiCallFrame`].
    pub fn stage(&self) -> FfiExecutionStage {
        FfiExecutionStage::from_c_api(unsafe { (*self.handle).stage })
    }

    /// Returns the number of inputs/arguments of this [`FfiCallFrame`].
    pub fn input_count(&self) -> usize {
        unsafe { (*self.handle).args.size as usize }
    }

    /// Returns the `index`-th input of this [`FfiCallFrame`], and an [`FfiError`] if the provided index is out of
    /// bounds, if the XLA runtime reports an unknown input type value, or if the runtime reports invalid metadata
    /// for that input.
    pub fn input(&self, index: usize) -> Result<FfiInput<'o>, FfiError> {
        unsafe {
            let inputs = &(*self.handle).args;
            let count = self.input_count();
            if index >= count {
                return Err(FfiError::invalid_argument(format!(
                    "input index {index} is out of bounds for an XLA FFI call frame with {count} inputs",
                )));
            }
            if inputs.types.is_null() {
                return Err(FfiError::invalid_argument("encountered null input types pointer in XLA FFI call frame"));
            }
            if inputs.args.is_null() {
                return Err(FfiError::invalid_argument("encountered null input values pointer in XLA FFI call frame"));
            }
            FfiInput::from_c_api(*inputs.types.add(index), *inputs.args.add(index))
        }
    }

    /// Returns an [`Iterator`] over the inputs of this [`FfiCallFrame`].
    pub fn inputs(&self) -> impl Iterator<Item = Result<FfiInput<'o>, FfiError>> {
        (0..self.input_count()).map(|index| self.input(index))
    }

    /// Returns the number of outputs/results of this [`FfiCallFrame`].
    pub fn output_count(&self) -> usize {
        unsafe { (*self.handle).rets.size as usize }
    }

    /// Returns the `index`-th output of this [`FfiCallFrame`], and an [`FfiError`] if the provided index is out of
    /// bounds, if the XLA runtime reports an unknown output type value, or if the runtime reports invalid metadata
    /// for that output.
    pub fn output(&self, index: usize) -> Result<FfiOutput<'o>, FfiError> {
        unsafe {
            let outputs = &(*self.handle).rets;
            let count = self.output_count();
            if index >= count {
                return Err(FfiError::invalid_argument(format!(
                    "output index {index} is out of bounds for an XLA FFI call frame with {count} outputs"
                )));
            }
            if outputs.types.is_null() {
                return Err(FfiError::invalid_argument("encountered null output types pointer in XLA FFI call frame"));
            }
            if outputs.rets.is_null() {
                return Err(FfiError::invalid_argument("encountered null output values pointer in XLA FFI call frame"));
            }
            FfiOutput::from_c_api(*outputs.types.add(index), *outputs.rets.add(index))
        }
    }

    /// Returns an [`Iterator`] over the outputs of this [`FfiCallFrame`].
    pub fn outputs(&self) -> impl Iterator<Item = Result<FfiOutput<'o>, FfiError>> {
        (0..self.output_count()).map(|index| self.output(index))
    }

    /// Returns the [`FfiFuture`] associated with this [`FfiCallFrame`], if present. [`FfiHandler`] implementations can
    /// use the returned [`FfiFuture`] to signal the result of an asynchronous computation to the XLA runtime. Note that
    /// the XLA runtime will keep all inputs/arguments, outputs/results, and attributes alive until the [`FfiFuture`] is
    /// completed.
    pub fn future(&self) -> Result<Option<FfiFuture>, FfiError> {
        unsafe {
            let handle = (*self.handle).future;
            if handle.is_null() { Ok(None) } else { FfiFuture::from_c_api(handle, self.api()?).map(Some) }
        }
    }

    /// Attempts to register metadata about the current [`FfiHandler`] with the XLA runtime. This is accomplished by
    /// checking whether this [`FfiCallFrame`] carries with it the XLA FFI metadata extension. If it does, then the
    /// current XLA FFI API [`VERSION`] will be registered with it along with the [`FfiTypeId`] for its state (if
    /// stateful; [`FfiTypeId::UNKNOWN`] or [`FfiTypeId::default()`] must be used for stateless handlers). If the
    /// registration is successful, then this function will return `true` and the caller must immediately return a null
    /// pointer from the FFI handler since that invocation is not expected to be an actual runtime invocation of that
    /// handler. FFI handler call frames only carry with them the XLA FFI metadata extension during handler registration
    /// with the XLA runtime.
    ///
    /// Refer to the documentation of [`FfiHandler`] for how to properly use this function.
    pub fn register_metadata(&self, handler_state_type_id: FfiTypeId) -> bool {
        unsafe {
            let mut extension = (*self.handle).extension_start;
            while !extension.is_null() {
                if (*extension).r#type == ffi::XLA_FFI_Extension_Type_Metadata {
                    let metadata = (*(extension as *mut ffi::XLA_FFI_Metadata_Extension)).metadata;
                    if !metadata.is_null() {
                        (*metadata).api_version.major_version = VERSION.major as std::ffi::c_int;
                        (*metadata).api_version.minor_version = VERSION.minor as std::ffi::c_int;
                        (*metadata).state_type_id = handler_state_type_id.to_c_api();
                        return true;
                    }
                }
                extension = (*extension).next;
            }
            false
        }
    }
}

/// [`FfiHandler`]s are functions that can be invoked by the XLA runtime at the various [`FfiExecutionStage`]s,
/// and are typically associated with custom call operations in StableHLO programs.
///
/// In order to implement [`FfiHandler`]s correctly, the handler implementation must always look as follows (with
/// appropriate trait and state type ID) in order to handle the registration of the handler with the runtime correctly:
///
/// ```rust
/// use ryft_pjrt::extensions::ffi::*;
///
/// unsafe extern "C" fn custom_call(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
///     unsafe {
///         match FfiCallFrame::from_c_api(call_frame) {
///             Err(_) => std::ptr::null_mut(),
///             Ok(call_frame) if call_frame.register_metadata(FfiTypeId::default()) => std::ptr::null_mut(),
///             Ok(call_frame) => { panic!("implement body of XLA FFI handler") },
///         }
///     }
/// }
/// ```
#[derive(Copy, Clone)]
pub struct FfiHandler {
    /// Underlying C function that will be invoked when this [`FfiHandler`] is invoked by the XLA runtime.
    callback: ffi::XLA_FFI_Handler,
}

impl FfiHandler {
    /// Constructs a new [`FfiHandler`] from the provided [`XLA_FFI_Handler`](ffi::XLA_FFI_Handler).
    pub(crate) fn from_c_api(callback: ffi::XLA_FFI_Handler) -> Self {
        Self { callback }
    }

    /// Returns the [`XLA_FFI_Handler`](ffi::XLA_FFI_Handler) that corresponds to this
    /// [`FfiHandler`] and which can be passed to functions in the XLA FFI API.
    pub(crate) fn to_c_api(self) -> ffi::XLA_FFI_Handler {
        self.callback
    }
}

impl From<ffi::XLA_FFI_Handler> for FfiHandler {
    fn from(value: ffi::XLA_FFI_Handler) -> Self {
        Self::from_c_api(value)
    }
}

/// Bundle of [`FfiHandler`]s that associates a potentially different [`FfiHandler`] for each [`FfiExecutionStage`].
#[derive(Copy, Clone)]
pub struct FfiHandlerBundle {
    /// Optional [`FfiHandler`] to use for the [`FfiExecutionStage::Instantiation`] stage.
    instantiate: Option<FfiHandler>,

    /// Optional [`FfiHandler`] to use for the [`FfiExecutionStage::Preparation`] stage.
    prepare: Option<FfiHandler>,

    /// Optional [`FfiHandler`] to use for the [`FfiExecutionStage::Initialization`] stage.
    initialize: Option<FfiHandler>,

    /// [`FfiHandler`] to use for the [`FfiExecutionStage::Execution`] stage.
    execute: FfiHandler,
}

impl FfiHandlerBundle {
    /// Constructs a new [`FfiHandlerBundle`].
    pub fn new(
        instantiate: Option<FfiHandler>,
        prepare: Option<FfiHandler>,
        initialize: Option<FfiHandler>,
        execute: FfiHandler,
    ) -> Self {
        Self { instantiate, prepare, initialize, execute }
    }

    /// Returns the [`XLA_FFI_Handler_Bundle`](ffi::XLA_FFI_Handler_Bundle) that corresponds to this
    /// [`FfiHandlerBundle`] and which can be passed to functions in the XLA FFI API.
    pub fn to_c_api(self) -> ffi::XLA_FFI_Handler_Bundle {
        ffi::XLA_FFI_Handler_Bundle {
            instantiate: self.instantiate.map(|handler| handler.to_c_api()),
            prepare: self.prepare.map(|handler| handler.to_c_api()),
            initialize: self.initialize.map(|handler| handler.to_c_api()),
            execute: self.execute.to_c_api(),
        }
    }
}

/// [`FfiHandlerTraits`] describe optional behavioral traits of [`FfiHandler`]s using a bit flag representation.
/// [`FfiHandlerTraits`] instances can be combined using [`BitOr`] in order to obtain an [`FfiHandlerTraits`] instance
/// that has the union of the traits of the other instances.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct FfiHandlerTraits {
    /// Underlying [`XLA_FFI_Handler_TraitsBits`](ffi::XLA_FFI_Handler_TraitsBits) backing this [`FfiHandlerTraits`].
    bits: ffi::XLA_FFI_Handler_TraitsBits,
}

impl FfiHandlerTraits {
    /// Returns the [`XLA_FFI_Handler_TraitsBits`](ffi::XLA_FFI_Handler_TraitsBits) that corresponds to this
    /// [`FfiHandlerTraits`] instance and which can be passed to functions in the XLA FFI API.
    pub(crate) const fn to_c_api(self) -> ffi::XLA_FFI_Handler_TraitsBits {
        self.bits
    }

    /// Default trait that assigns no behavioral traits to [`FfiHandler`]s.
    pub const NONE: Self = Self { bits: 0 };

    /// Trait which indicates that calls to this handler are safe to trace into a command buffer. It means that calls to
    /// the corresponding [`FfiHandler`] always launch exactly the same device operations (which may depend on attribute
    /// values) that can be captured and then replayed.
    pub const COMMAND_BUFFER_COMPATIBLE: Self =
        Self { bits: ffi::XLA_FFI_Handler_TraitsBits_COMMAND_BUFFER_COMPATIBLE };

    /// Constructs an [`FfiHandlerTraits`] instance from the bitwise representation of the provided `u32` value.
    pub const fn from_bits(bits: u32) -> Self {
        Self { bits: bits as ffi::XLA_FFI_Handler_TraitsBits }
    }

    /// Returns the underlying bitwise representation of this [`FfiHandlerTraits`] instance as a `u32` value.
    pub const fn bits(self) -> u32 {
        self.bits
    }

    /// Returns `true` if and only if all traits that are present in the provided [`FfiHandlerTraits`] instance are also
    /// present in this [`FfiHandlerTraits`] instance.
    pub const fn contains(self, other: Self) -> bool {
        (self.bits & other.bits) == other.bits
    }
}

impl BitOr for FfiHandlerTraits {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self { bits: self.bits | rhs.bits }
    }
}

impl BitOrAssign for FfiHandlerTraits {
    fn bitor_assign(&mut self, rhs: Self) {
        self.bits |= rhs.bits;
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::extensions::ffi::attributes::ffi::{XLA_FFI_Attrs, XLA_FFI_ByteSpan};
    use crate::extensions::ffi::context::ffi::{
        XLA_FFI_DeviceMemory_Allocate, XLA_FFI_DeviceMemory_Free, XLA_FFI_DeviceOrdinal_Get, XLA_FFI_ExecutionContext,
        XLA_FFI_ExecutionContext_Get, XLA_FFI_RunId_Get, XLA_FFI_State_Get, XLA_FFI_State_Set, XLA_FFI_Stream_Get,
        XLA_FFI_ThreadPool_NumThreads, XLA_FFI_ThreadPool_Schedule,
    };
    use crate::extensions::ffi::errors::ffi::{
        XLA_FFI_Error, XLA_FFI_Error_Create, XLA_FFI_Error_Destroy, XLA_FFI_Error_GetMessage,
    };
    use crate::extensions::ffi::futures::ffi::{
        XLA_FFI_Future, XLA_FFI_Future_Create, XLA_FFI_Future_SetAvailable, XLA_FFI_Future_SetError,
    };
    use crate::extensions::ffi::types::ffi::{XLA_FFI_Type_Register, XLA_FFI_TypeId};
    use crate::extensions::ffi::versions::ffi::XLA_FFI_Api_Version;

    pub type XLA_FFI_Extension_Type = std::ffi::c_uint;
    pub const XLA_FFI_Extension_Type_Metadata: XLA_FFI_Extension_Type = 1;

    #[repr(C)]
    pub struct XLA_FFI_Metadata_Extension {
        pub extension_base: XLA_FFI_Extension_Base,
        pub metadata: *mut XLA_FFI_Metadata,
    }

    #[repr(C)]
    pub struct XLA_FFI_Metadata {
        pub struct_size: usize,
        pub api_version: XLA_FFI_Api_Version,
        pub traits: XLA_FFI_Handler_Traits,
        pub state_type_id: XLA_FFI_TypeId,
    }

    #[repr(C)]
    pub struct XLA_FFI_Extension_Base {
        pub struct_size: usize,
        pub r#type: XLA_FFI_Extension_Type,
        pub next: *mut XLA_FFI_Extension_Base,
    }

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct XLA_FFI_InternalApi {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct XLA_FFI_Api {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub api_version: XLA_FFI_Api_Version,
        pub internal_api: *const XLA_FFI_InternalApi,
        pub XLA_FFI_Error_Create: Option<XLA_FFI_Error_Create>,
        pub XLA_FFI_Error_GetMessage: Option<XLA_FFI_Error_GetMessage>,
        pub XLA_FFI_Error_Destroy: Option<XLA_FFI_Error_Destroy>,
        pub XLA_FFI_Handler_Register: Option<XLA_FFI_Handler_Register>,
        pub XLA_FFI_Stream_Get: Option<XLA_FFI_Stream_Get>,
        pub XLA_FFI_Type_Register: Option<XLA_FFI_Type_Register>,
        pub XLA_FFI_ExecutionContext_Get: Option<XLA_FFI_ExecutionContext_Get>,
        pub XLA_FFI_State_Set: Option<XLA_FFI_State_Set>,
        pub XLA_FFI_State_Get: Option<XLA_FFI_State_Get>,
        pub XLA_FFI_DeviceMemory_Allocate: Option<XLA_FFI_DeviceMemory_Allocate>,
        pub XLA_FFI_DeviceMemory_Free: Option<XLA_FFI_DeviceMemory_Free>,
        pub XLA_FFI_ThreadPool_Schedule: Option<XLA_FFI_ThreadPool_Schedule>,
        pub XLA_FFI_ThreadPool_NumThreads: Option<XLA_FFI_ThreadPool_NumThreads>,
        pub XLA_FFI_Future_Create: Option<XLA_FFI_Future_Create>,
        pub XLA_FFI_Future_SetAvailable: Option<XLA_FFI_Future_SetAvailable>,
        pub XLA_FFI_Future_SetError: Option<XLA_FFI_Future_SetError>,
        pub XLA_FFI_RunId_Get: Option<XLA_FFI_RunId_Get>,
        pub XLA_FFI_DeviceOrdinal_Get: Option<XLA_FFI_DeviceOrdinal_Get>,
    }

    #[repr(C)]
    #[derive(Copy, Clone, PartialEq, Eq, Hash)]
    pub enum XLA_FFI_ExecutionStage {
        XLA_FFI_ExecutionStage_INSTANTIATE = 0,
        XLA_FFI_ExecutionStage_PREPARE = 1,
        XLA_FFI_ExecutionStage_INITIALIZE = 2,
        XLA_FFI_ExecutionStage_EXECUTE = 3,
    }

    pub type XLA_FFI_ArgType = std::ffi::c_uint;
    pub const XLA_FFI_ArgType_BUFFER: XLA_FFI_ArgType = 1;

    #[repr(C)]
    pub struct XLA_FFI_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub size: i64,
        pub types: *mut XLA_FFI_ArgType,
        pub args: *mut *mut std::ffi::c_void,
    }

    pub type XLA_FFI_RetType = std::ffi::c_uint;
    pub const XLA_FFI_RetType_BUFFER: XLA_FFI_RetType = 1;

    #[repr(C)]
    pub struct XLA_FFI_Rets {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub size: i64,
        pub types: *mut XLA_FFI_RetType,
        pub rets: *mut *mut std::ffi::c_void,
    }

    #[repr(C)]
    pub struct XLA_FFI_CallFrame {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub api: *const XLA_FFI_Api,
        pub context: *mut XLA_FFI_ExecutionContext,
        pub stage: XLA_FFI_ExecutionStage,
        pub args: XLA_FFI_Args,
        pub rets: XLA_FFI_Rets,
        pub attributes: XLA_FFI_Attrs,
        pub future: *mut XLA_FFI_Future,
    }

    pub type XLA_FFI_Handler = unsafe extern "C" fn(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error;

    #[repr(C)]
    pub struct XLA_FFI_Handler_Bundle {
        pub instantiate: Option<XLA_FFI_Handler>,
        pub prepare: Option<XLA_FFI_Handler>,
        pub initialize: Option<XLA_FFI_Handler>,
        pub execute: XLA_FFI_Handler,
    }

    pub type XLA_FFI_Handler_TraitsBits = std::ffi::c_uint;
    pub const XLA_FFI_Handler_TraitsBits_COMMAND_BUFFER_COMPATIBLE: XLA_FFI_Handler_TraitsBits = 1;

    pub type XLA_FFI_Handler_Traits = u32;

    #[repr(C)]
    pub struct XLA_FFI_Handler_Register_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub name: XLA_FFI_ByteSpan,
        pub platform: XLA_FFI_ByteSpan,
        pub bundle: XLA_FFI_Handler_Bundle,
        pub traits: XLA_FFI_Handler_Traits,
    }

    pub type XLA_FFI_Handler_Register =
        unsafe extern "C" fn(args: *mut XLA_FFI_Handler_Register_Args) -> *mut XLA_FFI_Error;
}

#[cfg(test)]
mod tests {
    use crate::extensions::ffi::errors::FfiError;
    use crate::extensions::ffi::tests::with_test_ffi_call_frame;
    use crate::extensions::ffi::types::FfiTypeId;

    use super::{FfiExecutionStage, FfiInput, FfiOutput};

    #[test]
    fn test_ffi_call_frame() {
        with_test_ffi_call_frame(|call_frame| {
            assert!(call_frame.api().is_ok());
            assert!(call_frame.context().is_ok());
            assert_eq!(call_frame.stage(), FfiExecutionStage::Execution);
            assert_eq!(call_frame.input_count(), 1);
            assert!(matches!(call_frame.input(0), Ok(FfiInput::Buffer { .. })));
            assert!(matches!(call_frame.input(1), Err(FfiError::InvalidArgument { .. })));
            assert!(matches!(
                call_frame.inputs().collect::<Result<Vec<_>, _>>(),
                Ok(inputs) if inputs.len() == 1));
            assert_eq!(call_frame.output_count(), 1);
            assert!(matches!(call_frame.output(0), Ok(FfiOutput::Buffer { .. })));
            assert!(matches!(call_frame.output(1), Err(FfiError::InvalidArgument { .. })));
            assert!(matches!(call_frame.outputs().collect::<Result<Vec<_>, _>>(), Ok(outputs) if outputs.len() == 1));
            assert!(call_frame.future().is_ok());
            assert!(!call_frame.register_metadata(FfiTypeId::default()));
        });
    }
}
