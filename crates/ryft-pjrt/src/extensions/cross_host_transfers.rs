use crate::{
    Api, Buffer, BufferType, Client, Device, Error, Event, Plugin, invoke_pjrt_api_error_fn, invoke_pjrt_api_void_fn,
    slice_from_c_api, str_from_c_api,
};

/// The PJRT cross-host transfers extension provides capabilities for initiating and receiving cross-host device
/// transfers. The extension is both optional for PJRT [`Plugin`]s and _experimental_, meaning that incompatible
/// changes may be introduced at any time, including changes that break _Application Binary Interface (ABI)_
/// compatibility.
#[derive(Copy, Clone)]
pub struct CrossHostTransfersExtension {
    /// Handle that represents this [`CrossHostTransfersExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_CrossHostTransfers_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl CrossHostTransfersExtension {
    /// Constructs a new [`CrossHostTransfersExtension`] from the provided
    /// [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base) handle if the type of that PJRT
    /// extension matches the PJRT cross-host transfers extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_CrossHostTransfers {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_CrossHostTransfers_Extension`](ffi::PJRT_CrossHostTransfers_Extension)
    /// that corresponds to this [`CrossHostTransfersExtension`] and which can be passed to
    /// functions in the PJRT C API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_CrossHostTransfers_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for CrossHostTransfersExtension {}
unsafe impl Sync for CrossHostTransfersExtension {}

/// Unique global identifier of a device across all hosts that are participating in the current job.
pub type GlobalDeviceId = i32;

/// Unique identifier of a cross-host transfer. Each cross-host transfer is assigned a unique ID by PJRT.
pub type CrossHostTransferKey = i64;

impl Buffer<'_> {
    #[allow(deprecated)]
    /// Sends this buffer to a remote host using the deprecated PJRT transfer API. This function must be called on the
    /// source/sender host after the destination/receiver host has called [`Client::make_cross_host_receive_buffers`]
    /// and communicated one serialized descriptor for the target receive slot. This function is the counterpart to
    /// [`Client::make_cross_host_receive_buffers`].
    ///
    /// # Parameters
    ///
    ///   - `serialized_descriptor`: Descriptor bytes for one remote receive slot, produced by the destination host's
    ///     receive notifier. This must be treated as an opaque binary token and transmitted verbatim, without parsing
    ///     or text conversion, from receiver to sender. It must then be passed back here to perform the transfer for
    ///     the matching buffer.
    ///   - `on_done`: Rust callback that will be invoked once the transfer completes and receives as input an optional
    ///     [`Error`], if something went wrong during the transfer, and a boolean flag indicating whether any send
    ///     operations were enqueued by PJRT during the transfer.
    #[deprecated(note = "The descriptor-based PJRT cross-host transfer API wrapper has been deprecated in favor of \
        a new API exposed via [`Client::cross_host_send_buffers`] and [`Client::cross_host_receive_buffers`].")]
    pub fn copy_to_remote_host<S: AsRef<[u8]>, F: FnOnce(Option<Error>, bool) + 'static>(
        &self,
        serialized_descriptor: S,
        on_done: F,
    ) -> Result<(), Error> {
        use ffi::PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args;
        let extension = self.api().cross_host_transfers_extension()?;

        extern "C" fn on_done_callback<F: FnOnce(Option<Error>, bool)>(
            error: *mut crate::errors::ffi::PJRT_Error,
            sends_were_enqueued: bool,
            user_arg: *mut std::ffi::c_void,
        ) {
            let mut callback_state = unsafe { Box::from_raw(user_arg as *mut CopyToRemoteHostCallbackState<F>) };
            let error = if error.is_null() {
                None
            } else {
                unsafe {
                    match Error::from_c_api(error, callback_state.api) {
                        Ok(Some(error)) => Some(error),
                        Ok(None) => None,
                        Err(error) => Some(error),
                    }
                }
            };
            if let Some(on_done) = callback_state.on_done.take() {
                on_done(error, sends_were_enqueued);
            }
        }

        let serialized_descriptor = serialized_descriptor.as_ref();
        if serialized_descriptor.is_empty() {
            return Err(Error::invalid_argument("serialized descriptors must not be empty"));
        }

        let mut callback_state = Box::new(CopyToRemoteHostCallbackState {
            api: (*self).api(),
            serialized_descriptor: serialized_descriptor.to_vec(),
            on_done: Some(on_done),
        });

        let serialized_descriptor = if callback_state.serialized_descriptor.is_empty() {
            std::ptr::null_mut()
        } else {
            callback_state.serialized_descriptor.as_mut_ptr() as *mut std::ffi::c_char
        };

        let serialized_descriptor = Box::into_raw(Box::new(serialized_descriptor));
        let serialized_descriptor_size = Box::into_raw(Box::new(callback_state.serialized_descriptor.len()));
        let callback_state = Box::into_raw(callback_state);

        let result = invoke_pjrt_api_void_fn!(
            @extension ffi::PJRT_CrossHostTransfers_Extension => extension,
            PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice,
            {
                buffer = self.to_c_api(),
                event = std::ptr::null_mut(),
                serialized_descriptor = serialized_descriptor,
                serialized_descriptor_size = serialized_descriptor_size,
                on_done = ffi::PJRT_Transfers_CrossHostRemoteSendCallbackInfo {
                    user_arg: callback_state as *mut std::ffi::c_void,
                    on_done: Some(on_done_callback::<F>),
                },
            },
        );

        if result.is_err() {
            unsafe {
                drop(Box::from_raw(serialized_descriptor));
                drop(Box::from_raw(serialized_descriptor_size));
                drop(Box::from_raw(callback_state));
            }
        }

        result
    }
}

impl Client<'_> {
    /// Attempts to load the [`CrossHostTransfersExtension`] from this [`Client`] and returns
    /// [`Error::Unimplemented`] if it is not provided by the underlying [`Plugin`].
    pub fn cross_host_transfers_extension(&self) -> Result<CrossHostTransfersExtension, Error> {
        self.api().cross_host_transfers_extension()
    }

    /// Initiates a cross-host transfer for the provided [`Buffer`]s.
    /// This function is the counterpart to [`Client::cross_host_receive_buffers`].
    ///
    /// # Parameters
    ///
    ///   - `buffers`: [`Buffer`]s to send. For each index `i`, `buffers[i]` is paired with
    ///     `destination_devices[i]` and `transfer_keys[i]`.
    ///   - `destination_devices`: IDs of the destination [`Device`]s for each [`Buffer`].
    ///   - `transfer_keys`: Transfer rendezvous keys that correspond to each [`Buffer`].
    ///     Each key must match the receiver-side key for the same transfer.
    ///
    /// # Returns
    ///
    /// [`Vec`] that contains one [`Event`] per [`Buffer`] that can be used to determine when the cross-host transfer
    /// for that [`Buffer`] completes, or an [`Error`] if something went wrong.
    pub fn cross_host_send_buffers(
        &self,
        buffers: &[&Buffer<'_>],
        destination_devices: &[GlobalDeviceId],
        transfer_keys: &[CrossHostTransferKey],
    ) -> Result<Vec<Event<()>>, Error> {
        use ffi::PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args;

        if buffers.is_empty() {
            return Ok(Vec::new());
        }

        if buffers.len() != destination_devices.len() || buffers.len() != transfer_keys.len() {
            return Err(Error::invalid_argument(format!(
                "the numbers of buffers ({}), destination devices ({}), and transfer keys ({}) must all match",
                buffers.len(),
                destination_devices.len(),
                transfer_keys.len(),
            )));
        }

        let mut buffers = buffers.iter().map(|buffer| unsafe { buffer.to_c_api() }).collect::<Vec<_>>();
        let destination_devices = destination_devices.as_ptr() as *const ffi::PJRT_Global_Device_Id;
        let transfer_keys = transfer_keys.as_ptr() as *const ffi::PJRT_Cross_Host_Transfer_Key;
        let mut send_events = vec![std::ptr::null_mut(); buffers.len()];

        let extension = self.api().cross_host_transfers_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_CrossHostTransfers_Extension => extension,
            PJRT_Transfers_PJRT_Client_CrossHostSendBuffers,
            {
                client = self.to_c_api(),
                num_buffers = buffers.len(),
                buffers = buffers.as_mut_ptr(),
                dst_global_device_ids = destination_devices,
                transfer_keys = transfer_keys,
                send_events = send_events.as_mut_ptr(),
            },
        )?;

        send_events
            .into_iter()
            .map(|handle| unsafe { Event::from_c_api(handle, (*self).api(), ()) })
            .collect()
    }

    /// Receives a cross-host transfer for the provided [`Buffer`]s.
    /// This function is the counterpart to [`Client::cross_host_send_buffers`].
    ///
    /// # Parameters
    ///
    ///   - `element_types`: Element [`BufferType`] for each of the [`Buffer`]s that are to be received.
    ///     Refer to [`Buffer::element_type`] for information on what the buffer element types represent.
    ///   - `dimensions`: Dimension arrays for each of the [`Buffer`]s that are to be received.
    ///     Refer to [`Buffer::dimensions`] for information on what the buffer dimensions represent.
    ///   - `device`: Destination [`Device`] on the current host, on which the received [`Buffer`]s
    ///     will be materialized.
    ///   - `source_devices`: IDs of the source [`Device`]s for each [`Buffer`].
    ///   - `transfer_keys`: Transfer rendezvous keys that correspond to each [`Buffer`].
    ///     Each key must match the sender-side key for the same transfer.
    ///
    /// # Returns
    ///
    /// [`Vec`] that contains the received [`Buffer`]s, or an [`Error`] if something went wrong.
    pub fn cross_host_receive_buffers<'c>(
        &'c self,
        element_types: &[BufferType],
        dimensions: &[&[i64]],
        device: &Device<'_>,
        source_devices: &[GlobalDeviceId],
        transfer_keys: &[CrossHostTransferKey],
    ) -> Result<Vec<Buffer<'c>>, Error> {
        use ffi::PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args;

        if element_types.is_empty() {
            return Ok(Vec::new());
        }

        if element_types.len() != dimensions.len()
            || element_types.len() != source_devices.len()
            || element_types.len() != transfer_keys.len()
        {
            return Err(Error::invalid_argument(format!(
                "the numbers of element types ({}), dimension arrays ({}), source devices ({}), \
                and transfer keys ({}) must all match",
                element_types.len(),
                dimensions.len(),
                source_devices.len(),
                transfer_keys.len(),
            )));
        }

        let source_devices = source_devices.as_ptr() as *const ffi::PJRT_Global_Device_Id;
        let transfer_keys = transfer_keys.as_ptr() as *const ffi::PJRT_Cross_Host_Transfer_Key;

        let ranks = dimensions.iter().map(|dimensions| dimensions.as_ptr()).collect::<Vec<_>>();
        let mut dimensions = dimensions.iter().map(|dimensions| dimensions.len()).collect::<Vec<_>>();
        let mut element_types = element_types.iter().map(|r#type| unsafe { r#type.to_c_api() }).collect::<Vec<_>>();
        let mut layouts = vec![std::ptr::null_mut(); dimensions.len()];
        let mut buffers = vec![std::ptr::null_mut(); dimensions.len()];

        let extension = self.api().cross_host_transfers_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_CrossHostTransfers_Extension => extension,
            PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers,
            {
                client = self.to_c_api(),
                num_shapes = dimensions.len(),
                shape_num_dims = dimensions.as_mut_ptr(),
                num_dims = ranks.as_ptr(),
                element_types = element_types.as_mut_ptr(),
                layouts = layouts.as_mut_ptr(),
                device = device.to_c_api(),
                src_global_device_ids = source_devices,
                transfer_keys = transfer_keys,
                buffers = buffers.as_mut_ptr(),
            },
        )?;

        buffers
            .into_iter()
            .map(|handle| unsafe { Buffer::from_c_api(handle, (*self).api(), self.to_c_api()) })
            .collect()
    }

    #[allow(deprecated)]
    /// Receives [`Buffer`]s from remote hosts using the deprecated PJRT transfer API. This function communicates
    /// descriptors using the provided `notifier`. The callback receives serialized descriptors that must be transmitted
    /// _out-of-band_ to the sender and later supplied to [`Buffer::copy_to_remote_host`] on the source host. This
    /// function is the counterpart to [`Buffer::copy_to_remote_host`].
    ///
    /// # Parameters
    ///
    ///   - `element_types`: Element [`BufferType`] for each of the [`Buffer`]s that are to be received.
    ///     Refer to [`Buffer::element_type`] for information on what the buffer element types represent.
    ///   - `dimensions`: Dimension arrays for each of the [`Buffer`]s that are to be received.
    ///     Refer to [`Buffer::dimensions`] for information on what the buffer dimensions represent.
    ///   - `device`: Destination [`Device`] on which the received [`Buffer`]s will be materialized.
    ///   - `notifier`: Callback function that will be invoked by PJRT with an optional [`Error`], serialized
    ///     descriptors, and an optional Rust cancel notifier closure so that the caller can coordinate with
    ///     the sender _out-of-band_.
    #[deprecated(note = "The descriptor-based PJRT cross-host transfer API wrapper has been deprecated in favor of \
        a new API exposed via [`Client::cross_host_send_buffers`] and [`Client::cross_host_receive_buffers`].")]
    pub fn make_cross_host_receive_buffers<
        'c,
        F: FnOnce(Option<Error>, &[&str], Option<CrossHostSendCancellationNotifierCallback>) + 'static,
    >(
        &'c self,
        element_types: &[BufferType],
        dimensions: &[&[i64]],
        device: &Device<'_>,
        notifier: F,
    ) -> Result<Vec<Buffer<'c>>, Error> {
        use ffi::PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args;

        if element_types.is_empty() {
            return Ok(Vec::new());
        }

        if element_types.len() != dimensions.len() {
            return Err(Error::invalid_argument(format!(
                "the numbers of element types ({}) and dimension arrays ({}) must match",
                element_types.len(),
                dimensions.len(),
            )));
        }

        unsafe extern "C" fn on_canceled_callback(
            error: *mut crate::errors::ffi::PJRT_Error,
            user_arg: *mut std::ffi::c_void,
        ) {
            if user_arg.is_null() {
                return;
            }

            let (on_canceled, api) = *unsafe { Box::from_raw(user_arg as *mut (CrossHostOnCanceledCallback, Api)) };
            let error = if error.is_null() {
                None
            } else {
                unsafe {
                    match Error::from_c_api(error, api) {
                        Ok(Some(error)) => Some(error),
                        Ok(None) => None,
                        Err(error) => Some(error),
                    }
                }
            };

            on_canceled(error);
        }

        unsafe extern "C" fn notifier_callback<
            F: FnOnce(Option<Error>, &[&str], Option<CrossHostSendCancellationNotifierCallback>) + 'static,
        >(
            error: *mut crate::errors::ffi::PJRT_Error,
            serialized_descriptors: *const *const std::ffi::c_char,
            descriptors_sizes: *mut usize,
            num_descriptors: usize,
            user_arg: *mut std::ffi::c_void,
            cancel_notifier: Option<ffi::PJRT_Transfers_CrossHostSendCancelNotifier>,
            cancel_notifier_user_arg: *mut std::ffi::c_void,
        ) {
            if user_arg.is_null() {
                return;
            }

            let mut notifier_state = unsafe { Box::from_raw(user_arg as *mut CrossHostReceiveBuffersNotifierState<F>) };
            let api = notifier_state.api;
            let error = if error.is_null() {
                None
            } else {
                unsafe {
                    match Error::from_c_api(error, api) {
                        Ok(Some(error)) => Some(error),
                        Ok(None) => None,
                        Err(error) => Some(error),
                    }
                }
            };

            let descriptors = unsafe { slice_from_c_api(serialized_descriptors, num_descriptors) };
            let descriptor_sizes = unsafe { slice_from_c_api(descriptors_sizes as *const usize, num_descriptors) };
            let descriptors = descriptors
                .iter()
                .zip(descriptor_sizes.iter())
                .map(|(descriptor, descriptor_size)| str_from_c_api(*descriptor, *descriptor_size).into_owned())
                .collect::<Vec<_>>();
            let descriptors = descriptors.iter().map(String::as_str).collect::<Vec<_>>();

            let cancel_notifier = cancel_notifier.map(|cancel_notifier| {
                Box::new(
                    move |serialized_descriptor: &str,
                          reason: Error,
                          on_canceled: Option<CrossHostOnCanceledCallback>| {
                        let (on_canceled, on_canceled_user_arg) = if let Some(on_canceled) = on_canceled {
                            (Some(on_canceled_callback as _), Box::into_raw(Box::new((on_canceled, api))) as *mut _)
                        } else {
                            (None, std::ptr::null_mut())
                        };
                        let reason_message = reason.message();
                        unsafe {
                            cancel_notifier(
                                serialized_descriptor.as_ptr() as *const std::ffi::c_char,
                                serialized_descriptor.len(),
                                reason.code(),
                                reason_message.as_ptr(),
                                reason_message.count_bytes(),
                                on_canceled,
                                on_canceled_user_arg,
                                cancel_notifier_user_arg,
                            );
                        }
                    },
                ) as CrossHostSendCancellationNotifierCallback
            });

            if let Some(notifier) = notifier_state.notifier.take() {
                notifier(error, descriptors.as_slice(), cancel_notifier);
            }
        }

        let ranks = dimensions.iter().map(|dimensions| dimensions.as_ptr()).collect::<Vec<_>>();
        let mut dimensions = dimensions.iter().map(|dimensions| dimensions.len()).collect::<Vec<_>>();
        let mut element_types = element_types.iter().map(|r#type| unsafe { r#type.to_c_api() }).collect::<Vec<_>>();
        let mut layouts = vec![std::ptr::null_mut(); dimensions.len()];
        let notifier_state = CrossHostReceiveBuffersNotifierState { notifier: Some(notifier), api: self.api() };
        let notifier_state = Box::into_raw(Box::new(notifier_state));
        let notifier = ffi::PJRT_Transfers_CrossHostRecvNotifierInfo {
            user_arg: notifier_state as *mut std::ffi::c_void,
            notifier: Some(notifier_callback::<F>),
        };

        let extension = self.api().cross_host_transfers_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_CrossHostTransfers_Extension => extension,
            PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers,
            {
                client = self.to_c_api(),
                num_shapes = dimensions.len(),
                shape_num_dims = dimensions.as_mut_ptr(),
                num_dims = ranks.as_ptr(),
                element_types = element_types.as_mut_ptr(),
                layouts = layouts.as_mut_ptr(),
                device = device.to_c_api(),
                notifier = notifier,
            },
            { buffers, num_buffers },
        )
        .inspect_err(|_| unsafe { drop(Box::from_raw(notifier_state)) })
        .and_then(|(buffers, buffer_count)| {
            if buffer_count == 0 {
                Ok(Vec::new())
            } else if buffers.is_null() {
                Err(Error::internal(format!(
                    "failed to make cross-host receive buffers; PJRT plugin returned a null output pointer \
                    for {buffer_count} buffers",
                )))
            } else {
                unsafe { slice_from_c_api(buffers, buffer_count) }
                    .iter()
                    .enumerate()
                    .map(|(index, buffer)| {
                        if buffer.is_null() {
                            Err(Error::internal(format!(
                                "failed to make cross-host receive buffers; plugin returned a null buffer \
                                at index {index}",
                            )))
                        } else {
                            unsafe { Buffer::from_c_api(*buffer, (*self).api(), self.to_c_api()) }
                        }
                    })
                    .collect()
            }
        })
    }
}

impl Plugin {
    /// Attempts to load the [`CrossHostTransfersExtension`] from this [`Plugin`] and returns
    /// [`Error::Unimplemented`] if it is not provided by this [`Plugin`].
    pub fn cross_host_transfers_extension(&self) -> Result<CrossHostTransfersExtension, Error> {
        self.api().cross_host_transfers_extension()
    }
}

impl Api {
    /// Attempts to load the [`CrossHostTransfersExtension`] from this [`Api`] and returns
    /// [`Error::Unimplemented`] if it is not provided by the underlying [`Plugin`].
    pub(crate) fn cross_host_transfers_extension(&self) -> Result<CrossHostTransfersExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let cross_host_transfers_extension = CrossHostTransfersExtension::from_c_api(extension, *self);
                if let Some(cross_host_transfers_extension) = cross_host_transfers_extension {
                    return Ok(cross_host_transfers_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the cross-host transfers extension is not provided by the PJRT plugin"))
        }
    }
}

#[allow(deprecated)]
/// Internal state used by the deprecated [`Buffer::copy_to_remote_host`] function.
struct CopyToRemoteHostCallbackState<F: FnOnce(Option<Error>, bool)> {
    serialized_descriptor: Vec<u8>,
    on_done: Option<F>,
    api: Api,
}

#[allow(deprecated)]
/// Callback that can be passed to a cross-host send cancellation notifier and which
/// is invoked once cancel propagation has been processed by PJRT.
pub type CrossHostOnCanceledCallback = Box<dyn FnOnce(Option<Error>) + 'static>;

#[allow(deprecated)]
/// Rust closure wrapper for the deprecated PJRT cross-host send cancel notifier.
/// The closure receives the following arguments:
///
///   - `serialized_descriptor`: Descriptor that identifies the receive slot to cancel.
///   - `reason`: Cancellation reason represented as a PJRT [`Error`].
///   - `on_canceled`: Optional callback that will be invoked once PJRT acknowledges the cancellation propagation.
pub type CrossHostSendCancellationNotifierCallback =
    Box<dyn FnOnce(&str, Error, Option<CrossHostOnCanceledCallback>) + 'static>;

#[allow(deprecated)]
/// Internal state used by the deprecated [`Client::make_cross_host_receive_buffers`] function.
struct CrossHostReceiveBuffersNotifierState<
    F: FnOnce(Option<Error>, &[&str], Option<CrossHostSendCancellationNotifierCallback>),
> {
    notifier: Option<F>,
    api: Api,
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::buffers::ffi::{PJRT_Buffer, PJRT_Buffer_MemoryLayout, PJRT_Buffer_Type};
    use crate::clients::ffi::PJRT_Client;
    use crate::devices::ffi::PJRT_Device;
    use crate::errors::ffi::{PJRT_Error, PJRT_Error_Code};
    use crate::events::ffi::PJRT_Event;
    use crate::ffi::PJRT_Extension_Base;

    pub const PJRT_API_CROSS_HOST_TRANSFERS_EXTENSION_VERSION: usize = 5;

    pub type PJRT_Global_Device_Id = i32;
    pub type PJRT_Cross_Host_Transfer_Key = i64;

    #[repr(C)]
    pub struct PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub num_buffers: usize,
        pub buffers: *mut *mut PJRT_Buffer,
        pub dst_global_device_ids: *const PJRT_Global_Device_Id,
        pub transfer_keys: *const PJRT_Cross_Host_Transfer_Key,
        pub send_events: *mut *mut PJRT_Event,
    }

    impl PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args {
        pub fn new(
            client: *mut PJRT_Client,
            num_buffers: usize,
            buffers: *mut *mut PJRT_Buffer,
            dst_global_device_ids: *const PJRT_Global_Device_Id,
            transfer_keys: *const PJRT_Cross_Host_Transfer_Key,
            send_events: *mut *mut PJRT_Event,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                num_buffers,
                buffers,
                dst_global_device_ids,
                transfer_keys,
                send_events,
            }
        }
    }

    pub type PJRT_Transfers_PJRT_Client_CrossHostSendBuffers =
        unsafe extern "C" fn(args: *mut PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub num_shapes: usize,
        pub shape_num_dims: *mut usize,
        pub num_dims: *const *const i64,
        pub element_types: *mut PJRT_Buffer_Type,
        pub layouts: *mut *mut PJRT_Buffer_MemoryLayout,
        pub device: *mut PJRT_Device,
        pub src_global_device_ids: *const PJRT_Global_Device_Id,
        pub transfer_keys: *const PJRT_Cross_Host_Transfer_Key,
        pub buffers: *mut *mut PJRT_Buffer,
    }

    impl PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            client: *mut PJRT_Client,
            num_shapes: usize,
            shape_num_dims: *mut usize,
            num_dims: *const *const i64,
            element_types: *mut PJRT_Buffer_Type,
            layouts: *mut *mut PJRT_Buffer_MemoryLayout,
            device: *mut PJRT_Device,
            src_global_device_ids: *const PJRT_Global_Device_Id,
            transfer_keys: *const PJRT_Cross_Host_Transfer_Key,
            buffers: *mut *mut PJRT_Buffer,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                num_shapes,
                shape_num_dims,
                num_dims,
                element_types,
                layouts,
                device,
                src_global_device_ids,
                transfer_keys,
                buffers,
            }
        }
    }

    pub type PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers =
        unsafe extern "C" fn(args: *mut PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args) -> *mut PJRT_Error;

    pub type PJRT_Transfers_CrossHostOnCanceledCallback =
        Option<unsafe extern "C" fn(error: *mut PJRT_Error, user_arg: *mut std::ffi::c_void)>;

    pub type PJRT_Transfers_CrossHostSendCancelNotifier = unsafe extern "C" fn(
        serialized_descriptor: *const std::ffi::c_char,
        serialized_descriptor_size: usize,
        reason: PJRT_Error_Code,
        error_message: *const std::ffi::c_char,
        error_message_size: usize,
        on_canceled: PJRT_Transfers_CrossHostOnCanceledCallback,
        on_canceled_user_arg: *mut std::ffi::c_void,
        user_arg: *mut std::ffi::c_void,
    );

    pub type PJRT_Transfers_CrossHostRecvNotifier = unsafe extern "C" fn(
        error: *mut PJRT_Error,
        serialized_descriptors: *const *const std::ffi::c_char,
        descriptors_sizes: *mut usize,
        num_descriptors: usize,
        user_arg: *mut std::ffi::c_void,
        cancel_notifier: Option<PJRT_Transfers_CrossHostSendCancelNotifier>,
        cancel_notifier_user_arg: *mut std::ffi::c_void,
    );

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct PJRT_Transfers_CrossHostRecvNotifierInfo {
        pub user_arg: *mut std::ffi::c_void,
        pub notifier: Option<PJRT_Transfers_CrossHostRecvNotifier>,
    }

    #[repr(C)]
    pub struct PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub num_shapes: usize,
        pub shape_num_dims: *mut usize,
        pub num_dims: *const *const i64,
        pub element_types: *mut PJRT_Buffer_Type,
        pub layouts: *mut *mut PJRT_Buffer_MemoryLayout,
        pub device: *mut PJRT_Device,
        pub notifier: PJRT_Transfers_CrossHostRecvNotifierInfo,
        pub buffers: *mut *mut PJRT_Buffer,
        pub num_buffers: usize,
    }

    impl PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            client: *mut PJRT_Client,
            num_shapes: usize,
            shape_num_dims: *mut usize,
            num_dims: *const *const i64,
            element_types: *mut PJRT_Buffer_Type,
            layouts: *mut *mut PJRT_Buffer_MemoryLayout,
            device: *mut PJRT_Device,
            notifier: PJRT_Transfers_CrossHostRecvNotifierInfo,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                num_shapes,
                shape_num_dims,
                num_dims,
                element_types,
                layouts,
                device,
                notifier,
                buffers: std::ptr::null_mut(),
                num_buffers: 0,
            }
        }
    }

    pub type PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers =
        unsafe extern "C" fn(args: *mut PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args) -> *mut PJRT_Error;

    pub type PJRT_Transfers_CrossHostRemoteSendCallback =
        unsafe extern "C" fn(error: *mut PJRT_Error, sends_were_enqueued: bool, user_arg: *mut std::ffi::c_void);

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct PJRT_Transfers_CrossHostRemoteSendCallbackInfo {
        pub user_arg: *mut std::ffi::c_void,
        pub on_done: Option<PJRT_Transfers_CrossHostRemoteSendCallback>,
    }

    #[repr(C)]
    pub struct PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub event: *mut PJRT_Event,
        pub serialized_descriptor: *mut *mut std::ffi::c_char,
        pub serialized_descriptor_size: *mut usize,
        pub on_done: PJRT_Transfers_CrossHostRemoteSendCallbackInfo,
    }

    impl PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args {
        pub fn new(
            buffer: *mut PJRT_Buffer,
            event: *mut PJRT_Event,
            serialized_descriptor: *mut *mut std::ffi::c_char,
            serialized_descriptor_size: *mut usize,
            on_done: PJRT_Transfers_CrossHostRemoteSendCallbackInfo,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                event,
                serialized_descriptor,
                serialized_descriptor_size,
                on_done,
            }
        }
    }

    pub type PJRT_Buffer_CopyToRemoteDevice =
        unsafe extern "C" fn(args: *mut PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args);

    #[repr(C)]
    pub struct PJRT_CrossHostTransfers_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers:
            Option<PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers>,
        pub PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice: Option<PJRT_Buffer_CopyToRemoteDevice>,
        pub PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers:
            Option<PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers>,
        pub PJRT_Transfers_PJRT_Client_CrossHostSendBuffers: Option<PJRT_Transfers_PJRT_Client_CrossHostSendBuffers>,
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{TestPlatform, test_for_each_platform};
    use crate::{BufferType, Error};

    #[test]
    fn test_cross_host_transfers_extension() {
        test_for_each_platform!(|plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    assert!(plugin.cross_host_transfers_extension().is_ok());
                    assert!(client.cross_host_transfers_extension().is_ok());
                }
                _ => {
                    assert!(matches!(plugin.cross_host_transfers_extension(), Err(Error::Unimplemented { .. })));
                    assert!(matches!(client.cross_host_transfers_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }

    // TODO(eaplatanios): The following unit tests are not great, but I am not sure how to test this effectively.

    #[test]
    fn test_cross_host_transfers() {
        test_for_each_platform!(|_plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    let device = client.addressable_devices().unwrap().remove(0);
                    let buffer = client.buffer(&[1u8, 2u8], BufferType::U8, [4], None, device.clone(), None).unwrap();
                    let buffers = [&buffer];
                    assert!(matches!(
                        client.cross_host_send_buffers(&buffers, &[], &[]),
                        Err(Error::InvalidArgument { .. })
                    ));
                    assert!(matches!(
                        client.cross_host_send_buffers(&buffers, &[0], &[]),
                        Err(Error::InvalidArgument { .. })
                    ));
                    assert!(matches!(
                        client.cross_host_send_buffers(&buffers, &[], &[42i64]),
                        Err(Error::InvalidArgument { .. })
                    ));
                    assert!(matches!(
                        client.cross_host_send_buffers(&buffers, &[0], &[42i64]),
                        Err(Error::InvalidArgument { .. })
                    ));
                    assert!(matches!(
                        client.cross_host_receive_buffers(&[BufferType::U8], &[&[2i64]], &device, &[], &[]),
                        Err(Error::InvalidArgument { .. }),
                    ));
                    assert!(matches!(
                        client.cross_host_receive_buffers(&[BufferType::U8], &[&[2i64]], &device, &[0], &[]),
                        Err(Error::InvalidArgument { .. }),
                    ));
                    assert!(matches!(
                        client.cross_host_receive_buffers(&[BufferType::U8], &[&[2i64]], &device, &[], &[42i64]),
                        Err(Error::InvalidArgument { .. }),
                    ));
                    assert!(matches!(
                        client.cross_host_receive_buffers(&[BufferType::U8], &[&[2i64]], &device, &[0], &[42i64]),
                        Err(Error::InvalidArgument { .. }),
                    ));
                }
                _ => {
                    assert!(matches!(client.cross_host_transfers_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }

    #[test]
    #[allow(deprecated)]
    fn test_cross_host_transfers_deprecated() {
        test_for_each_platform!(|_plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    let device = client.addressable_devices().unwrap().remove(0);
                    assert!(matches!(
                        client.make_cross_host_receive_buffers(&[], &[], &device, |_error, _descriptors, _cancel_notifier| {}),
                        Ok(received_buffers) if received_buffers.is_empty(),
                    ));
                    assert!(matches!(
                        client.make_cross_host_receive_buffers(
                            &[BufferType::U8],
                            &[],
                            &device,
                            |_error, _descriptors, _cancel_notifier| {},
                        ),
                        Err(Error::InvalidArgument { .. }),
                    ));

                    let buffer = client.buffer(&[1u8, 2u8], BufferType::U8, [4], None, device.clone(), None).unwrap();
                    assert!(matches!(
                        buffer.copy_to_remote_host([], |_error, _sends_were_enqueued| {}),
                        Err(Error::InvalidArgument { .. }),
                    ));
                }
                _ => {
                    assert!(matches!(client.cross_host_transfers_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }
}
