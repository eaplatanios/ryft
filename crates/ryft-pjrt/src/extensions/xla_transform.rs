use std::marker::{PhantomData, PhantomPinned};

use crate::{Api, Client, Error, Plugin, invoke_pjrt_api_error_fn, slice_from_c_api};

/// The PJRT XLA transform extension provides hooks for registering HLO module transformations that are applied during
/// backend compilation. The extension is optional for PJRT [`Plugin`]s and _experimental_, meaning that incompatible
/// changes may be introduced at any time, including changes that break _Application Binary Interface (ABI)_
/// compatibility.
#[derive(Copy, Clone)]
pub struct XlaTransformExtension {
    /// Handle that represents this [`XlaTransformExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_Xla_Transform_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl XlaTransformExtension {
    /// Constructs a new [`XlaTransformExtension`] from the provided
    /// [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base) handle if the type of that PJRT extension matches the
    /// PJRT XLA transform extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_XlaTransform {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_Xla_Transform_Extension`](ffi::PJRT_Xla_Transform_Extension) that corresponds
    /// to this [`XlaTransformExtension`] and which can be passed to functions in the PJRT C API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_Xla_Transform_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for XlaTransformExtension {}
unsafe impl Sync for XlaTransformExtension {}

impl XlaTransformExtension {
    /// Registers the provided XLA transform callbacks under `name` for the specified compilation pipeline `stage`.
    pub fn register_transform<N: AsRef<str>>(
        &self,
        name: N,
        stage: XlaTransformPipelineStage,
        callbacks: &mut XlaTransformCallbacks,
    ) -> Result<(), Error> {
        use ffi::PJRT_Register_Xla_Transform_Args;
        let name = name.as_ref();
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_Xla_Transform_Extension => self,
            PJRT_Register_Xla_Transform,
            {
                name = name.as_ptr() as *const _,
                name_size = name.len(),
                stage = stage.to_c_api(),
                callbacks = callbacks.to_c_api(),
            },
        )
    }
}

impl Client<'_> {
    /// Attempts to load the [`XlaTransformExtension`] from this [`Client`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub fn xla_transform_extension(&self) -> Result<XlaTransformExtension, Error> {
        self.api().xla_transform_extension()
    }

    /// Registers the provided XLA transform callbacks. Refer to the documentation of
    /// [`XlaTransformExtension::register_transform`] for more information.
    pub fn register_xla_transform<N: AsRef<str>>(
        &self,
        name: N,
        stage: XlaTransformPipelineStage,
        callbacks: &mut XlaTransformCallbacks,
    ) -> Result<(), Error> {
        self.xla_transform_extension()?.register_transform(name, stage, callbacks)
    }
}

impl Plugin {
    /// Attempts to load the [`XlaTransformExtension`] from this [`Plugin`] and returns [`Error::Unimplemented`]
    /// if it is not provided by this [`Plugin`].
    pub fn xla_transform_extension(&self) -> Result<XlaTransformExtension, Error> {
        self.api().xla_transform_extension()
    }

    /// Registers the provided XLA transform callbacks. Refer to the documentation of
    /// [`XlaTransformExtension::register_transform`] for more information.
    pub fn register_xla_transform<N: AsRef<str>>(
        &self,
        name: N,
        stage: XlaTransformPipelineStage,
        callbacks: &mut XlaTransformCallbacks,
    ) -> Result<(), Error> {
        self.xla_transform_extension()?.register_transform(name, stage, callbacks)
    }
}

impl Api {
    /// Attempts to load the [`XlaTransformExtension`] from this [`Api`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub(crate) fn xla_transform_extension(&self) -> Result<XlaTransformExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let xla_transform_extension = XlaTransformExtension::from_c_api(extension, *self);
                if let Some(xla_transform_extension) = xla_transform_extension {
                    return Ok(xla_transform_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the XLA transform extension is not provided by the PJRT plugin"))
        }
    }
}

/// Pipeline stage at which an [`XlaTransformCallbacks`] instance should be applied.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum XlaTransformPipelineStage {
    /// Applies the transform before scheduling.
    PreScheduler,

    /// Applies the transform after scheduling.
    PostScheduler,
}

impl XlaTransformPipelineStage {
    /// Returns the [`PJRT_XlaTransform_PipelineStage`](ffi::PJRT_XlaTransform_PipelineStage) that corresponds
    /// to this [`XlaTransformPipelineStage`] and which can be passed to functions in the PJRT C API.
    pub(crate) fn to_c_api(self) -> ffi::PJRT_XlaTransform_PipelineStage {
        match self {
            Self::PreScheduler => ffi::PJRT_XlaTransform_PipelineStage_kPreScheduler,
            Self::PostScheduler => ffi::PJRT_XlaTransform_PipelineStage_kPostScheduler,
        }
    }
}

/// Serialized HLO module transform callback.
pub type XlaTransformCallback =
    unsafe extern "C" fn(callbacks: *mut XlaTransformCallbacks, args: *mut XlaTransformRawArguments);

/// Optional destructor callback for an [`XlaTransformCallbacks`] instance.
pub type XlaTransformDestructor = unsafe extern "C" fn(callbacks: *mut XlaTransformCallbacks);

/// Opaque raw argument pointer passed to [`XlaTransformCallback`] implementations.
#[repr(C)]
pub struct XlaTransformRawArguments {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Callback table used by the PJRT XLA transform extension.
#[repr(C)]
pub struct XlaTransformCallbacks {
    /// Extension callback version.
    version: i64,

    /// Optional callback invoked when the transform is no longer needed by the backend.
    destructor: Option<XlaTransformDestructor>,

    /// Callback invoked to transform a serialized HLO module.
    callback: Option<XlaTransformCallback>,
}

impl XlaTransformCallbacks {
    /// Constructs a new [`XlaTransformCallbacks`] table.
    pub fn new(callback: XlaTransformCallback, destructor: Option<XlaTransformDestructor>) -> Self {
        Self {
            version: ffi::PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION as i64,
            destructor,
            callback: Some(callback),
        }
    }

    /// Returns the [`PJRT_XlaTransform_Callbacks`](ffi::PJRT_XlaTransform_Callbacks) pointer that corresponds
    /// to this [`XlaTransformCallbacks`] and which can be passed to functions in the PJRT C API.
    pub(crate) fn to_c_api(&mut self) -> *mut ffi::PJRT_XlaTransform_Callbacks {
        self as *mut _ as *mut _
    }
}

/// Wrapper around the XLA transform callback arguments.
pub struct XlaTransformArguments {
    /// Handle that represents this [`XlaTransformArguments`] in the PJRT C API.
    handle: *mut ffi::PJRT_XlaTransform_Args,
}

impl XlaTransformArguments {
    /// Constructs a new [`XlaTransformArguments`] from the provided
    /// [`XlaTransformRawArguments`] handle that came from a function in the PJRT C API.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `handle` is the live argument pointer passed by a PJRT XLA transform callback and
    /// that it remains valid for the returned wrapper's lifetime. The wrapper is exposed so callback implementations
    /// can inspect and populate the C API callback arguments without depending on private FFI structs directly.
    pub unsafe fn from_c_api(handle: *mut XlaTransformRawArguments) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided XLA transform arguments handle is a null pointer"))
        } else {
            Ok(Self { handle: handle as *mut _ })
        }
    }

    /// Returns the serialized input HLO module.
    pub fn hlo_module(&self) -> &[u8] {
        unsafe {
            let string = (*self.handle).hlo_module;
            slice_from_c_api(string.data as *const u8, string.size)
        }
    }

    /// Marks the transform as unchanged.
    pub fn set_unchanged(&mut self) {
        unsafe {
            (*self.handle).changed = false;
        }
    }

    /// Sets the transformed serialized HLO module output.
    ///
    /// The caller must ensure that `data` remains valid until PJRT has consumed the callback result. Implementations
    /// that allocate a transformed module specifically for the callback should generally keep that storage owned by the
    /// callback state and release it through the callback destructor.
    pub fn set_transformed_hlo_module(&mut self, data: &[u8]) {
        unsafe {
            (*self.handle).transformed_hlo_module =
                ffi::PJRT_XlaTransform_string { data: data.as_ptr() as *const _, size: data.len() };
            (*self.handle).changed = true;
        }
    }

    /// Sets the callback error state using the provided PJRT error `code` and message bytes.
    ///
    /// The caller must ensure that `message` remains valid until PJRT has consumed the callback result.
    pub fn set_error(&mut self, code: u32, message: &[u8]) {
        unsafe {
            (*self.handle).header.has_error = true;
            (*self.handle).header.code = code;
            (*self.handle).header.error_msg =
                ffi::PJRT_XlaTransform_string { data: message.as_ptr() as *const _, size: message.len() };
        }
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::errors::ffi::{PJRT_Error, PJRT_Error_Code};
    use crate::ffi::PJRT_Extension_Base;

    pub const PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION: usize = 1;

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct PJRT_XlaTransform_string {
        pub data: *const std::ffi::c_char,
        pub size: usize,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct PJRT_XlaTransform_version_and_error {
        pub api_version: i64,
        pub data: *mut std::ffi::c_void,
        pub cleanup_fn: Option<unsafe extern "C" fn(data: *mut std::ffi::c_void)>,
        pub has_error: bool,
        pub code: PJRT_Error_Code,
        pub error_msg: PJRT_XlaTransform_string,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct PJRT_XlaTransform_Args {
        pub struct_size: usize,
        pub header: PJRT_XlaTransform_version_and_error,
        pub hlo_module: PJRT_XlaTransform_string,
        pub transformed_hlo_module: PJRT_XlaTransform_string,
        pub changed: bool,
    }

    #[repr(C)]
    pub struct PJRT_XlaTransform_Callbacks {
        pub version: i64,
        pub dtor: Option<unsafe extern "C" fn(callbacks: *mut PJRT_XlaTransform_Callbacks)>,
        pub callback: Option<
            unsafe extern "C" fn(callbacks: *mut PJRT_XlaTransform_Callbacks, args: *mut PJRT_XlaTransform_Args),
        >,
    }

    pub type PJRT_XlaTransform_PipelineStage = std::ffi::c_uint;
    pub const PJRT_XlaTransform_PipelineStage_kPreScheduler: PJRT_XlaTransform_PipelineStage = 0;
    pub const PJRT_XlaTransform_PipelineStage_kPostScheduler: PJRT_XlaTransform_PipelineStage = 1;

    #[repr(C)]
    pub struct PJRT_Register_Xla_Transform_Args {
        pub struct_size: usize,
        pub name: *const std::ffi::c_char,
        pub name_size: usize,
        pub stage: PJRT_XlaTransform_PipelineStage,
        pub callbacks: *mut PJRT_XlaTransform_Callbacks,
    }

    impl PJRT_Register_Xla_Transform_Args {
        pub fn new(
            name: *const std::ffi::c_char,
            name_size: usize,
            stage: PJRT_XlaTransform_PipelineStage,
            callbacks: *mut PJRT_XlaTransform_Callbacks,
        ) -> Self {
            Self { struct_size: size_of::<Self>(), name, name_size, stage, callbacks }
        }
    }

    pub type PJRT_Register_Xla_Transform =
        unsafe extern "C" fn(args: *mut PJRT_Register_Xla_Transform_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Xla_Transform_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_Register_Xla_Transform: Option<PJRT_Register_Xla_Transform>,
    }
}

#[cfg(test)]
mod tests {
    use crate::Error;
    use crate::extensions::xla_transform::{
        XlaTransformArguments, XlaTransformCallbacks, XlaTransformPipelineStage, XlaTransformRawArguments,
    };
    use crate::tests::{TestPlatform, test_for_each_platform};

    unsafe extern "C" fn no_op_transform(_callbacks: *mut XlaTransformCallbacks, args: *mut XlaTransformRawArguments) {
        let mut args = unsafe { XlaTransformArguments::from_c_api(args).unwrap() };
        args.set_unchanged();
    }

    #[test]
    fn test_xla_transform_extension() {
        test_for_each_platform!(|plugin, client, platform| {
            let mut callbacks = XlaTransformCallbacks::new(no_op_transform, None);
            match platform {
                TestPlatform::Cpu | TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    if plugin.xla_transform_extension().is_ok() {
                        assert!(
                            plugin
                                .register_xla_transform(
                                    "ryft_test_no_op",
                                    XlaTransformPipelineStage::PreScheduler,
                                    &mut callbacks,
                                )
                                .is_ok()
                        );
                        assert!(client.xla_transform_extension().is_ok());
                    } else {
                        assert!(matches!(plugin.xla_transform_extension(), Err(Error::Unimplemented { .. })));
                        assert!(matches!(client.xla_transform_extension(), Err(Error::Unimplemented { .. })));
                    }
                }
                _ => {
                    assert!(matches!(plugin.xla_transform_extension(), Err(Error::Unimplemented { .. })));
                    assert!(matches!(client.xla_transform_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }
}
