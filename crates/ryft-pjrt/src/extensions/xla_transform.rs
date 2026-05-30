use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

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
    /// Registers the provided XLA transform under `name` for the specified compilation pipeline `stage`.
    pub fn register_transform<N: AsRef<str>, T: XlaTransform + 'static>(
        &self,
        name: N,
        stage: XlaTransformPipelineStage,
        transform: T,
    ) -> Result<(), Error> {
        use ffi::PJRT_Register_Xla_Transform_Args;
        let name = name.as_ref();
        let mut callback_registration = XlaTransformCallbackRegistration::new(Box::new(transform));
        let result = invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_Xla_Transform_Extension => self,
            PJRT_Register_Xla_Transform,
            {
                name = name.as_ptr() as *const _,
                name_size = name.len(),
                stage = stage.to_c_api(),
                callbacks = callback_registration.to_c_api(),
            },
        );
        if result.is_ok() {
            callback_registration.commit();
        }
        result
    }
}

impl Client<'_> {
    /// Attempts to load the [`XlaTransformExtension`] from this [`Client`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub fn xla_transform_extension(&self) -> Result<XlaTransformExtension, Error> {
        self.api().xla_transform_extension()
    }

    /// Registers the provided XLA transform. Refer to the documentation of
    /// [`XlaTransformExtension::register_transform`] for more information.
    pub fn register_xla_transform<N: AsRef<str>, T: XlaTransform + 'static>(
        &self,
        name: N,
        stage: XlaTransformPipelineStage,
        transform: T,
    ) -> Result<(), Error> {
        self.xla_transform_extension()?.register_transform(name, stage, transform)
    }
}

impl Plugin {
    /// Attempts to load the [`XlaTransformExtension`] from this [`Plugin`] and returns [`Error::Unimplemented`]
    /// if it is not provided by this [`Plugin`].
    pub fn xla_transform_extension(&self) -> Result<XlaTransformExtension, Error> {
        self.api().xla_transform_extension()
    }

    /// Registers the provided XLA transform. Refer to the documentation of
    /// [`XlaTransformExtension::register_transform`] for more information.
    pub fn register_xla_transform<N: AsRef<str>, T: XlaTransform + 'static>(
        &self,
        name: N,
        stage: XlaTransformPipelineStage,
        transform: T,
    ) -> Result<(), Error> {
        self.xla_transform_extension()?.register_transform(name, stage, transform)
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

/// Pipeline stage at which an [`XlaTransform`] should be applied.
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

/// Result produced by an [`XlaTransform`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum XlaTransformResult {
    /// The transform did not modify the serialized HLO module.
    Unchanged,

    /// The transform produced a replacement serialized HLO module.
    Changed(Vec<u8>),
}

impl XlaTransformResult {
    /// Returns an unchanged transform result.
    pub fn unchanged() -> Self {
        Self::Unchanged
    }

    /// Returns a changed transform result with the provided serialized HLO module.
    pub fn changed<T: Into<Vec<u8>>>(module: T) -> Self {
        Self::Changed(module.into())
    }
}

/// Safe Rust interface for XLA HLO module transforms registered through the PJRT [`XlaTransformExtension`].
/// Implementations receive the serialized `HloModuleProto` bytes and either return [`XlaTransformResult::Unchanged`]
/// or replacement serialized `HloModuleProto` bytes. The implementation may be invoked by backend-owned compilation
/// threads, so the transform is stored behind a mutex and must be [`Send`].
pub trait XlaTransform: Send {
    /// Transforms the provided serialized HLO module.
    ///
    /// # Parameters
    ///
    ///   - `module`: Serialized `HloModuleProto` bytes supplied by XLA.
    fn transform_module(&mut self, module: &[u8]) -> Result<XlaTransformResult, Error>;
}

impl<F: FnMut(&[u8]) -> Result<XlaTransformResult, Error> + Send> XlaTransform for F {
    fn transform_module(&mut self, module: &[u8]) -> Result<XlaTransformResult, Error> {
        self(module)
    }
}

type XlaTransformCallback =
    unsafe extern "C" fn(callbacks: *mut ffi::PJRT_XlaTransform_Callbacks, args: *mut ffi::PJRT_XlaTransform_Args);

/// Private callback table layout used to carry Rust callback state through XLA's copied callback table. The upstream
/// adapter copies [`PJRT_XlaTransform_Callbacks`](ffi::PJRT_XlaTransform_Callbacks) by value and later passes that copy
/// to [`XlaTransform::transform_module`]. The ABI currently has no user-data field and does not call `dtor`, and so
/// this layout-compatible table carries the Rust state pointer in the copied `dtor` slot. Successful registrations
/// leak the table and state for process lifetime because the C API has no unregister or destructor path. If upstream
/// starts invoking `dtor`, this layout must be replaced with an upstream-supported user-data mechanism.
#[repr(C)]
struct XlaTransformCallbackTable {
    /// XLA transform extension callback ABI version.
    version: i64,

    /// Rust callback state pointer carried in the upstream `dtor` slot.
    state: *mut XlaTransformCallbackState,

    /// Common trampoline for all registered transforms.
    transform_hlo_module: Option<XlaTransformCallback>,
}

const _: () = assert!(size_of::<XlaTransformCallbackTable>() == size_of::<ffi::PJRT_XlaTransform_Callbacks>());
const _: () = assert!(align_of::<XlaTransformCallbackTable>() == align_of::<ffi::PJRT_XlaTransform_Callbacks>());

/// Owns a callback table while XLA transform registration is in progress.
struct XlaTransformCallbackRegistration {
    /// [`XlaTransformCallbackTable`] passed to the PJRT [`XlaTransformExtension`].
    table: Box<XlaTransformCallbackTable>,
}

impl XlaTransformCallbackRegistration {
    /// Creates a new [`XlaTransformCallbackRegistration`] for the provided [`XlaTransform`].
    fn new(transform: Box<dyn XlaTransform>) -> Self {
        let state = Box::into_raw(Box::new(XlaTransformCallbackState::new(transform)));
        Self {
            table: Box::new(XlaTransformCallbackTable {
                version: ffi::PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION as i64,
                state,
                transform_hlo_module: Some(xla_transform_callback),
            }),
        }
    }

    /// Returns the [`PJRT_XlaTransform_Callbacks`](ffi::PJRT_XlaTransform_Callbacks) pointer that corresponds to this
    /// [`XlaTransformCallbackRegistration`] and that is expected by the PJRT C API.
    fn to_c_api(&mut self) -> *mut ffi::PJRT_XlaTransform_Callbacks {
        (self.table.as_mut() as *mut XlaTransformCallbackTable).cast()
    }

    /// Marks registration as successful by keeping the callback table and state alive for the process lifetime.
    fn commit(self) {
        std::mem::forget(self);
    }
}

impl Drop for XlaTransformCallbackRegistration {
    fn drop(&mut self) {
        if !self.table.state.is_null() {
            unsafe { drop(Box::from_raw(self.table.state)) };
            self.table.state = std::ptr::null_mut();
        }
    }
}

/// Mutable state associated with a registered [`XlaTransform`].
struct XlaTransformCallbackState {
    /// [`XlaTransformAndBuffers`] guarded for backend compilation threads.
    transform_and_buffers: Mutex<XlaTransformAndBuffers>,
}

impl XlaTransformCallbackState {
    /// Creates new [`XlaTransformCallbackState`] for the provided [`XlaTransform`].
    fn new(transform: Box<dyn XlaTransform>) -> Self {
        Self { transform_and_buffers: Mutex::new(XlaTransformAndBuffers { transform, buffers: Vec::new() }) }
    }
}

/// User-provided [`XlaTransform`] and retained buffers guarded by [`XlaTransformCallbackState::transform_and_buffers`].
struct XlaTransformAndBuffers {
    /// User-provided [`XlaTransform`].
    transform: Box<dyn XlaTransform>,

    /// Output and error buffers retained for the lifetime of the current process because the upstream callback ABI
    /// does not provide a callback-completion cleanup hook.
    buffers: Vec<Box<[u8]>>,
}

unsafe extern "C" fn xla_transform_callback(
    callbacks: *mut ffi::PJRT_XlaTransform_Callbacks,
    args: *mut ffi::PJRT_XlaTransform_Args,
) {
    if args.is_null() {
        return;
    }

    if callbacks.is_null() {
        unsafe {
            set_xla_transform_error(
                args,
                crate::errors::ffi::PJRT_Error_Code_INTERNAL,
                b"missing XLA transform callback table",
            )
        };
        return;
    }

    let callback_table = callbacks.cast::<XlaTransformCallbackTable>();
    let callback_state = unsafe { (*callback_table).state };
    if callback_state.is_null() {
        unsafe {
            set_xla_transform_error(
                args,
                crate::errors::ffi::PJRT_Error_Code_INTERNAL,
                b"missing XLA transform callback state",
            )
        };
        return;
    }

    let callback_state = unsafe { &*callback_state };
    let mut state = callback_state.transform_and_buffers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

    let module = unsafe {
        let module = (*args).hlo_module;
        slice_from_c_api(module.data as *const u8, module.size)
    };
    let result = catch_unwind(AssertUnwindSafe(|| state.transform.transform_module(module)))
        .unwrap_or_else(|_| Err(Error::internal("XLA transform callback panicked")));
    match result {
        Ok(XlaTransformResult::Unchanged) => unsafe {
            (*args).changed = false;
            (*args).transformed_hlo_module = ffi::PJRT_XlaTransform_string { data: std::ptr::null(), size: 0 };
        },
        Ok(XlaTransformResult::Changed(hlo_module)) => {
            let hlo_module = hlo_module.into_boxed_slice();
            let data = hlo_module.as_ptr();
            let size = hlo_module.len();
            state.buffers.push(hlo_module);
            unsafe {
                (*args).changed = true;
                (*args).transformed_hlo_module = ffi::PJRT_XlaTransform_string { data: data as *const _, size };
            }
        }
        Err(error) => {
            let code = error.code();
            let message = error.to_string().into_bytes().into_boxed_slice();
            let data = message.as_ptr();
            let size = message.len();
            state.buffers.push(message);
            unsafe { set_xla_transform_error(args, code, slice_from_c_api(data, size)) };
        }
    }
}

unsafe fn set_xla_transform_error(
    args: *mut ffi::PJRT_XlaTransform_Args,
    code: crate::errors::ffi::PJRT_Error_Code,
    message: &[u8],
) {
    unsafe {
        (*args).header.has_error = true;
        (*args).header.code = code;
        (*args).header.error_msg =
            ffi::PJRT_XlaTransform_string { data: message.as_ptr() as *const _, size: message.len() };
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
        pub transform_hlo_module: Option<
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
    use crate::extensions::xla_transform::{XlaTransformPipelineStage, XlaTransformResult};
    use crate::tests::{TestPlatform, test_for_each_platform};
    use crate::{Error, errors, slice_from_c_api};

    struct NoOpTransform;

    impl super::XlaTransform for NoOpTransform {
        fn transform_module(&mut self, _module: &[u8]) -> Result<XlaTransformResult, Error> {
            Ok(XlaTransformResult::Unchanged)
        }
    }

    #[test]
    fn test_xla_transform_callback() {
        let mut registration = super::XlaTransformCallbackRegistration::new(Box::new(
            |module: &[u8]| -> Result<XlaTransformResult, Error> {
                assert_eq!(module, b"input");
                Ok(XlaTransformResult::changed(b"output".to_vec()))
            },
        ));
        let input = b"input";
        let mut args = super::ffi::PJRT_XlaTransform_Args {
            struct_size: size_of::<super::ffi::PJRT_XlaTransform_Args>(),
            header: super::ffi::PJRT_XlaTransform_version_and_error {
                api_version: super::ffi::PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION as i64,
                data: std::ptr::null_mut(),
                cleanup_fn: None,
                has_error: false,
                code: errors::ffi::PJRT_Error_Code_OK,
                error_msg: super::ffi::PJRT_XlaTransform_string { data: std::ptr::null(), size: 0 },
            },
            hlo_module: super::ffi::PJRT_XlaTransform_string { data: input.as_ptr() as *const _, size: input.len() },
            transformed_hlo_module: super::ffi::PJRT_XlaTransform_string { data: std::ptr::null(), size: 0 },
            changed: false,
        };
        unsafe { super::xla_transform_callback(registration.to_c_api(), &mut args as *mut _) };
        assert!(!args.header.has_error);
        assert!(args.changed);
        assert_eq!(
            unsafe {
                slice_from_c_api(args.transformed_hlo_module.data as *const u8, args.transformed_hlo_module.size)
            },
            b"output",
        );
    }

    #[test]
    fn test_xla_transform_callback_registration_is_unbounded() {
        let mut registrations = Vec::new();
        for index in 0..128 {
            let expected = index as u8;
            registrations.push(super::XlaTransformCallbackRegistration::new(Box::new(
                move |module: &[u8]| -> Result<XlaTransformResult, Error> {
                    assert_eq!(module, &[expected]);
                    Ok(XlaTransformResult::changed(vec![expected, expected]))
                },
            )));
        }

        for (index, registration) in registrations.iter_mut().enumerate() {
            let input = [index as u8];
            let mut args = super::ffi::PJRT_XlaTransform_Args {
                struct_size: size_of::<super::ffi::PJRT_XlaTransform_Args>(),
                header: super::ffi::PJRT_XlaTransform_version_and_error {
                    api_version: super::ffi::PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION as i64,
                    data: std::ptr::null_mut(),
                    cleanup_fn: None,
                    has_error: false,
                    code: errors::ffi::PJRT_Error_Code_OK,
                    error_msg: super::ffi::PJRT_XlaTransform_string { data: std::ptr::null(), size: 0 },
                },
                hlo_module: super::ffi::PJRT_XlaTransform_string {
                    data: input.as_ptr() as *const _,
                    size: input.len(),
                },
                transformed_hlo_module: super::ffi::PJRT_XlaTransform_string { data: std::ptr::null(), size: 0 },
                changed: false,
            };
            unsafe { super::xla_transform_callback(registration.to_c_api(), &mut args as *mut _) };
            assert!(!args.header.has_error);
            assert!(args.changed);
            assert_eq!(
                unsafe {
                    slice_from_c_api(args.transformed_hlo_module.data as *const u8, args.transformed_hlo_module.size)
                },
                &[index as u8, index as u8],
            );
        }
    }

    #[test]
    fn test_xla_transform_extension() {
        test_for_each_platform!(|plugin, client, platform| {
            match platform {
                TestPlatform::Cpu | TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    if plugin.xla_transform_extension().is_ok() {
                        assert!(
                            plugin
                                .register_xla_transform(
                                    "ryft_test_no_op",
                                    XlaTransformPipelineStage::PreScheduler,
                                    NoOpTransform,
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
