use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::{
    Api, Client, DeviceDescription, Error, MemoryKindId, Plugin, invoke_pjrt_api_error_fn, slice_from_c_api,
    str_from_c_api,
};

/// The PJRT memory descriptions extension provides capabilities around querying [`MemoryDescription`]s
/// for [`DeviceDescription`]s. This extension is optional and experimental for PJRT [`Plugin`]s.
/// If present, it enables the discovery of all supported types of memory for a given [`DeviceDescription`].
/// This is useful for specifying non-default memories in Ahead-Of-Time (AOT) compilations of PJRT programs
/// (as opposed to the physically present memories associated with a [`Client`]).
#[derive(Copy, Clone)]
pub struct MemoryDescriptionsExtension {
    /// Handle that represents this [`MemoryDescriptionsExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_MemoryDescriptions_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl MemoryDescriptionsExtension {
    /// Constructs a new [`MemoryDescriptionsExtension`] from the provided
    /// [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base) handle if the type of that
    /// PJRT extension matches the PJRT memory descriptions extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_MemoryDescriptions {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_MemoryDescriptions_Extension`](ffi::PJRT_MemoryDescriptions_Extension)
    /// that corresponds to this [`MemoryDescriptionsExtension`] and which can be passed to
    /// functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_MemoryDescriptions_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for MemoryDescriptionsExtension {}
unsafe impl Sync for MemoryDescriptionsExtension {}

impl Client<'_> {
    /// Attempts to load the [`MemoryDescriptionsExtension`] from this [`Client`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub fn memory_descriptions_extension(&self) -> Result<MemoryDescriptionsExtension, Error> {
        self.api().memory_descriptions_extension()
    }
}

impl Plugin {
    /// Attempts to load the [`MemoryDescriptionsExtension`] from this [`Plugin`] and returns [`Error::Unimplemented`]
    /// if it is not provided by this [`Plugin`].
    pub fn memory_descriptions_extension(&self) -> Result<MemoryDescriptionsExtension, Error> {
        self.api().memory_descriptions_extension()
    }
}

impl Api {
    /// Attempts to load the [`MemoryDescriptionsExtension`] from this [`Api`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub(crate) fn memory_descriptions_extension(&self) -> Result<MemoryDescriptionsExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let memory_descriptions_extension = MemoryDescriptionsExtension::from_c_api(extension, *self);
                if let Some(memory_descriptions_extension) = memory_descriptions_extension {
                    return Ok(memory_descriptions_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the memory descriptions extension is not provided by the PJRT plugin"))
        }
    }
}

/// Description of a [`Memory`](crate::Memory) that is associated with a [`DeviceDescription`].
#[derive(Copy, Clone)]
pub struct MemoryDescription<'o> {
    /// Handle that represents this [`MemoryDescription`] in the PJRT C API.
    handle: *const ffi::PJRT_MemoryDescription,

    /// [`MemoryDescriptionsExtension`] that was used to create this [`MemoryDescription`].
    extension: MemoryDescriptionsExtension,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`MemoryDescription`].
    owner: PhantomData<&'o ()>,
}

impl MemoryDescription<'_> {
    /// Constructs a new [`MemoryDescription`] from the provided
    /// [`PJRT_MemoryDescription`](ffi::PJRT_MemoryDescription) handle.
    pub(crate) unsafe fn from_c_api(
        handle: *const ffi::PJRT_MemoryDescription,
        extension: MemoryDescriptionsExtension,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT memory description handle is a null pointer"))
        } else {
            Ok(Self { handle, extension, owner: PhantomData })
        }
    }

    /// Returns the [`PJRT_MemoryDescription`](ffi::PJRT_MemoryDescription) that corresponds to
    /// this [`MemoryDescription`] and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_MemoryDescription {
        self.handle
    }

    /// Platform-dependent ID that uniquely identifies the kind of the underlying [`Memory`](crate::Memory).
    pub fn kind_id(&self) -> Result<MemoryKindId, Error> {
        self.kind_id_and_kind().map(|(kind_id, _)| kind_id)
    }

    /// Platform-dependent string that uniquely identifies the kind of the underlying [`Memory`](crate::Memory).
    pub fn kind(&'_ self) -> Result<Cow<'_, str>, Error> {
        self.kind_id_and_kind().map(|(_, kind)| kind)
    }

    /// Internal helper that is used to implement [`MemoryDescription::kind_id`] and [`MemoryDescription::kind`].
    fn kind_id_and_kind(&'_ self) -> Result<(MemoryKindId, Cow<'_, str>), Error> {
        use ffi::PJRT_MemoryDescription_Kind_Args;
        invoke_pjrt_api_error_fn!(
            @unchecked self.extension,
            PJRT_MemoryDescription_Kind,
            { memory_description = self.to_c_api() },
            { kind, kind_size, kind_id },
        )
        .map(|(kind, kind_size, kind_id)| (kind_id as usize, str_from_c_api(kind, kind_size)))
    }
}

impl Display for MemoryDescription<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.kind().unwrap())
    }
}

impl Debug for MemoryDescription<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MemoryDescription[{self}]")
    }
}

impl PartialEq for MemoryDescription<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.kind_id().is_ok() && other.kind_id().is_ok() && self.kind_id() == other.kind_id()
    }
}

impl Eq for MemoryDescription<'_> {}

impl DeviceDescription<'_> {
    /// Returns all [`MemoryDescription`]s associated with this [`DeviceDescription`] (i.e., the [`MemoryDescription`]s
    /// that correspond to the [`Memory`](crate::Memory)s attached to the [`Device`](crate::Device) that corresponds to
    /// this [`DeviceDescription`]). The resulting [`MemoryDescription`]s are returned in no particular order.
    pub fn memory_descriptions(&'_ self) -> Result<Vec<MemoryDescription<'_>>, Error> {
        self.memory_descriptions_with_default_index().map(|(memory_descriptions, _)| memory_descriptions)
    }

    /// Returns the default [`MemoryDescription`] associated with this [`DeviceDescription`], if one exists.
    pub fn default_memory_description(&'_ self) -> Result<Option<MemoryDescription<'_>>, Error> {
        let (memory_descriptions, default_memory_index) = self.memory_descriptions_with_default_index()?;
        Ok(default_memory_index.and_then(|index| memory_descriptions.get(index).copied()))
    }

    /// Internal helper that is used to implement [`DeviceDescription::memory_descriptions`]
    /// and [`DeviceDescription::default_memory_description`].
    fn memory_descriptions_with_default_index(&'_ self) -> Result<(Vec<MemoryDescription<'_>>, Option<usize>), Error> {
        use ffi::PJRT_DeviceDescription_MemoryDescriptions_Args;
        let extension = self.api().memory_descriptions_extension()?;
        invoke_pjrt_api_error_fn!(
            @unchecked extension,
            PJRT_DeviceDescription_MemoryDescriptions,
            { device_description = self.to_c_api() },
            { memory_descriptions, num_memory_descriptions, default_memory_index },
        )
        .and_then(|(memory_descriptions, memory_descriptions_count, default_memory_index)| {
            unsafe { slice_from_c_api(memory_descriptions, memory_descriptions_count) }
                .iter()
                .map(|handle| unsafe { MemoryDescription::from_c_api(*handle, extension) })
                .collect::<Result<Vec<_>, _>>()
                .map(|memory_descriptions| {
                    let default_memory_index =
                        if default_memory_index >= 0 && default_memory_index < memory_descriptions_count as isize {
                            Some(default_memory_index as usize)
                        } else {
                            None
                        };
                    (memory_descriptions, default_memory_index)
                })
        })
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::devices::ffi::PJRT_DeviceDescription;
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;

    pub const PJRT_API_MEMORY_DESCRIPTIONS_EXTENSION_VERSION: usize = 1;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_MemoryDescription {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_DeviceDescription_MemoryDescriptions_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device_description: *mut PJRT_DeviceDescription,
        pub memory_descriptions: *const *const PJRT_MemoryDescription,
        pub num_memory_descriptions: usize,
        pub default_memory_index: isize,
    }

    impl PJRT_DeviceDescription_MemoryDescriptions_Args {
        pub fn new(device_description: *mut PJRT_DeviceDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device_description,
                memory_descriptions: std::ptr::null(),
                num_memory_descriptions: 0,
                default_memory_index: 0,
            }
        }
    }

    pub type PJRT_DeviceDescription_MemoryDescriptions =
        unsafe extern "C" fn(args: *mut PJRT_DeviceDescription_MemoryDescriptions_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_MemoryDescription_Kind_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub memory_description: *const PJRT_MemoryDescription,
        pub kind: *const std::ffi::c_char,
        pub kind_size: usize,
        pub kind_id: std::ffi::c_int,
    }

    impl PJRT_MemoryDescription_Kind_Args {
        pub fn new(memory_description: *const PJRT_MemoryDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                memory_description,
                kind: std::ptr::null(),
                kind_size: 0,
                kind_id: 0,
            }
        }
    }

    pub type PJRT_MemoryDescription_Kind =
        unsafe extern "C" fn(args: *mut PJRT_MemoryDescription_Kind_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_MemoryDescriptions_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_DeviceDescription_MemoryDescriptions: Option<PJRT_DeviceDescription_MemoryDescriptions>,
        pub PJRT_MemoryDescription_Kind: Option<PJRT_MemoryDescription_Kind>,
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{test_cpu_client, test_cpu_plugin, test_for_each_platform};

    #[test]
    fn test_memory_descriptions_extension() {
        assert!(test_cpu_plugin().memory_descriptions_extension().is_ok());
        assert!(test_cpu_client().memory_descriptions_extension().is_ok());
    }

    #[test]
    fn test_device_memory_descriptions() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let device = client.addressable_devices().unwrap().remove(0);
            let device_description = device.description().unwrap();
            let memory_descriptions = device_description.memory_descriptions();
            if let Ok(memory_descriptions) = memory_descriptions {
                for memory_description in memory_descriptions {
                    assert_ne!(memory_description.kind_id().unwrap(), 0);
                    assert_ne!(memory_description.kind().unwrap().len(), 0);
                }
                let default_memory_description = device_description.default_memory_description().unwrap();
                if let Some(default_memory_description) = default_memory_description {
                    assert_ne!(default_memory_description.kind_id().unwrap(), 0);
                    assert_ne!(default_memory_description.kind().unwrap().len(), 0);
                }
            }
        });
    }
}
