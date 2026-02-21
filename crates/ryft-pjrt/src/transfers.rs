use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::{
    Api, Buffer, BufferSpecification, Client, Device, Error, Event, HostBufferData, Memory, NamedValue,
    invoke_pjrt_api_error_fn, slice_from_c_api,
};

/// Manager for transferring data from the host (i.e., the CPU) to a PJRT [`Device`]. This manager is essentially
/// a high-performance "conveyor belt" for data whose primary goal is **pipelining**. While [`Client::buffer`],
/// [`Client::borrowed_buffer`], and [`Client::borrowed_mut_buffer`] are great for one-off transfers,
/// [`HostToDeviceTransferManager`]s are designed for the high-frequency ingestion seen in training loops,
/// where you need to feed thousands of mini-batches without the CPU becoming a bottleneck.
///
/// When you transfer data using [`Client::buffer`], [`Client::borrowed_buffer`], or [`Client::borrowed_mut_buffer`]
/// the runtime often has to perform "bookkeeping" for every single [`Buffer`] like allocating memory for it, creating
/// an [`Event`] for the transfer, tracking dependencies, etc. [`HostToDeviceTransferManager`]s allow you to group these
/// operations. More importantly, they provide a mechanism for **pre-allocating** the necessary device memory so that
/// the actual transfers can happen with zero "negotiation" overhead between the host and the target device.
///
/// # Lifecycle
///
/// [`HostToDeviceTransferManager`]s can be thought of as stateful sessions for transferring a batch of [`Buffer`]s
/// from the host to a specific [`Device`] (or rather more precisely a specific [`Memory`]), and the following steps
/// roughly comprise their typical lifecycle:
///
///   1. Use [`Client::host_to_device_transfer_manager`] to construct a [`HostToDeviceTransferManager`] and allocate
///      "placeholder" [`Buffer`]s for the desired transfers.
///   2. Attach optional transfer metadata using [`HostToDeviceTransferManager::add_metadata`].
///   3. Initiate / queue up data transfers using (potentially multiple calls to) either
///      [`HostToDeviceTransferManager::transfer_data`] or [`HostToDeviceTransferManager::set_error`].
///   4. Retrieve the new [`Buffer`]s using [`HostToDeviceTransferManager::retrieve_buffer`]. Note that the resulting
///      [`Buffer`]s may not be immediately ready since the whole point of this transfer manager is that it handles
///      data transfers completely asynchronously.
///
/// The lifetime parameter `'c` captures the lifetime of the [`Client`] that owns this [`HostToDeviceTransferManager`],
/// ensuring that the client outlives the transfer manager.
pub struct HostToDeviceTransferManager<'c> {
    /// Handle that represents this [`HostToDeviceTransferManager`] in the PJRT C API.
    handle: *mut ffi::PJRT_AsyncHostToDeviceTransferManager,

    /// Underlying PJRT [`Api`].
    api: Api,

    /// Handle of the [`Client`] that owns this [`HostToDeviceTransferManager`]. Note that it is safe to hold a raw
    /// pointer here because the corresponding [`Client`] is guaranteed to outlive this [`HostToDeviceTransferManager`]
    /// by design. The reason we do not hold a reference to the [`Client`] itself is to avoid having to carry around an
    /// additional lifetime for the [`KeyValueStore`](crate::KeyValueStore) that is associated with that [`Client`].
    client: *mut crate::clients::ffi::PJRT_Client,

    /// [`PhantomData`] used to track the lifetime of the [`Client`] that owns this [`HostToDeviceTransferManager`].
    owner: PhantomData<&'c ()>,
}

impl<'c> HostToDeviceTransferManager<'c> {
    /// Constructs a new [`HostToDeviceTransferManager`] from the provided
    /// [`PJRT_AsyncHostToDeviceTransferManager`](ffi::PJRT_AsyncHostToDeviceTransferManager)
    /// handle that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(
        handle: *mut ffi::PJRT_AsyncHostToDeviceTransferManager,
        api: Api,
        client: *mut crate::clients::ffi::PJRT_Client,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument(
                "the provided PJRT async host-to-device transfer manager handle is a null pointer",
            ))
        } else if client.is_null() {
            Err(Error::invalid_argument("the provided PJRT client handle is a null pointer"))
        } else {
            Ok(Self { handle, api, client, owner: PhantomData })
        }
    }

    /// Returns the [`PJRT_AsyncHostToDeviceTransferManager`](ffi::PJRT_AsyncHostToDeviceTransferManager) that
    /// corresponds to this [`HostToDeviceTransferManager`] and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_AsyncHostToDeviceTransferManager {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }

    /// Returns the [`Device`] of this [`HostToDeviceTransferManager`] (i.e., the device that it targets).
    pub fn device(&self) -> Result<Device<'c>, Error> {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_Device_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_Device,
            { transfer_manager = self.to_c_api() },
            { device_out }
        )
        .and_then(|handle| unsafe { Device::from_c_api(handle, self.api()) })
    }

    /// Returns the number of [`Buffer`]s that are managed by this [`HostToDeviceTransferManager`].
    pub fn buffer_count(&self) -> Result<usize, Error> {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_BufferCount,
            { transfer_manager = self.to_c_api() },
            { buffer_count }
        )
    }

    /// Returns the number of bytes that the `index`-th [`Buffer`] managed by this [`HostToDeviceTransferManager`]
    /// occupies in the underlying [`Memory`]. The provided `index` must be smaller than the result of
    /// [`HostToDeviceTransferManager::buffer_count`] for this transfer manager.
    pub fn buffer_on_device_size_in_bytes(&self, index: usize) -> Result<usize, Error> {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_BufferSize,
            { transfer_manager = self.to_c_api(), buffer_index = index as std::ffi::c_int },
            { buffer_size }
        )
    }

    /// Transfers the provided data to the `index`-th [`Buffer`] managed by this [`HostToDeviceTransferManager`].
    ///
    /// # Parameters
    ///
    ///   - `index`: Index of the [`Buffer`] managed by this [`HostToDeviceTransferManager`] in which to transfer
    ///     the provided data.
    ///   - `data`: Data to transfer. Note that the data will be copied "as-is" (i.e., without performing any layout
    ///     transformations. This means that the data must already have the expected memory layout.
    ///   - `offset`: Offset in the storage representation of the target [`Buffer`] on which to transfer the provided
    ///     data. This enables transferring different chunks of a [`Buffer`] potentially from different buffers in the
    ///     host memory.
    ///   - `is_last_transfer`: Boolean indicating whether this transfer should be treated as the last transfer for the
    ///     `index`-th [`Buffer`] managed by this [`HostToDeviceTransferManager`]. If `true`, then the [`Buffer`] will
    ///     be marked as "ready" after this transfer completes. Otherwise, it will not be marked as ready and will thus
    ///     also be unavailable to consumers after this transfer completes. Note that if `true`, then no further
    ///     transfer calls (via [`HostToDeviceTransferManager::transfer_data`] or
    ///     [`HostToDeviceTransferManager::set_error`]) will be allowed for the same buffer `index` in this
    ///     [`HostToDeviceTransferManager`].
    pub fn transfer_data<B: AsRef<[u8]>>(
        &self,
        index: usize,
        data: Rc<RefCell<B>>,
        offset: usize,
        is_last_transfer: bool,
    ) -> Result<Event<()>, Error> {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args;
        let transfer_size = data.borrow().as_ref().len();
        let data = HostBufferData::from_host_buffer_rc_refcell(&data, false);
        let handle = invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_TransferData,
            {
                transfer_manager = self.to_c_api(),
                buffer_index = index as std::ffi::c_int,
                data = data.ptr,
                offset = offset as i64,
                transfer_size = transfer_size as i64,
                is_last_transfer = is_last_transfer,
            },
            { done_with_h2d_transfer },
        )?;
        let event = unsafe { Event::from_c_api(handle, self.api(), ()) }?;

        // Register a callback to decrease the reference count of `data` once the transfer is completed.
        if let Some(drop_fn) = data.drop_fn {
            // Register the callback that will be invoked once the host buffer data has been copied.
            event.on_ready(|_| {
                // We ignore the error because there is nothing we can do with it here,
                // and if something goes wrong, it should be reflected in [`Buffer::ready`].
                drop_fn();
            })?;
        }

        Ok(event)
    }

    /// Transfers the provided host buffer (interpreted as an XLA literal) to the `index`-th [`Buffer`] managed
    /// by this [`HostToDeviceTransferManager`].
    ///
    /// # XLA Literal Memory Layout
    ///
    /// The bytes in `data` must match the in-memory representation of an XLA array literal for the shape
    /// described by `specification`:
    ///
    ///   1. `data` is wrapped directly in an [`xla::BorrowingLiteral`](
    ///      https://github.com/openxla/xla/blob/main/xla/literal.cc#L2920-L2928)). This means `data` is interpreted as
    ///      a literal buffer directly (i.e., it is **not** parsed as a serialized [`LiteralProto`](
    ///      https://github.com/openxla/xla/blob/main/xla/xla_data.proto#L669-L705)).
    ///   2. `data` must point to a contiguous buffer large enough for the literal's dense array storage, including
    ///      layout padding. In XLA terms this corresponds to [`ShapeUtil::ByteSizeOf`](
    ///      https://github.com/openxla/xla/blob/main/xla/shape_util.h#L177-L183).
    ///   3. Element ordering must follow the provided `specification.layout` (or XLA's default dense layout if
    ///      `specification.layout` is [`None`]). For untiled layouts, linearization follows minor-to-major dimension
    ///      ordering as documented by [`IndexUtil::MultidimensionalIndexToLinearIndex`](
    ///      https://github.com/openxla/xla/blob/main/xla/index_util.h#L41-L114). For tiled layouts, bytes must follow
    ///      XLA tiled-layout rules (i.e., tile-major ordering, within-tile ordering, and edge padding) as described in
    ///      the [official documentation](
    ///      https://openxla.org/xla/tiled_layout#linear_index_formulas_for_tiling_given_a_shape_and_a_tile).
    ///   4. Note that, in the reference OpenXLA PJRT wrapper, [`StridedLayout`](crate::StridedLayout)s are rejected
    ///      for this code path.
    ///
    /// The returned [`Event`] becomes ready once PJRT is done consuming the host literal bytes.
    ///
    /// # Parameters
    ///
    ///   - `index`: Index of the [`Buffer`] managed by this [`HostToDeviceTransferManager`] in which to transfer
    ///     the provided literal.
    ///   - `data`: Host buffer containing the literal bytes to transfer. Refer to the memory layout contract above
    ///     for information on the memory layout of this buffer.
    ///   - `specification`: [`BufferSpecification`] describing how the provided `data` should be interpreted
    ///     as an XLA literal.
    pub fn transfer_literal<B: AsRef<[u8]>, D: AsRef<[u64]>>(
        &self,
        index: usize,
        data: Rc<RefCell<B>>,
        specification: BufferSpecification<D>,
    ) -> Result<Event<()>, Error> {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args;
        let dimensions = specification.dimensions.as_ref().iter().map(|&d| d as i64).collect::<Vec<_>>();
        let data = HostBufferData::from_host_buffer_rc_refcell(&data, false);
        let handle = invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_TransferLiteral,
            {
                transfer_manager = self.to_c_api(),
                buffer_index = index as std::ffi::c_int,
                data = data.ptr,
                shape_dims = dimensions.as_ptr(),
                shape_num_dims = dimensions.len(),
                shape_element_type = specification.element_type.to_c_api(),
                shape_layout = specification.layout.map(|layout| &layout.to_c_api() as *const _ as *mut _)
                    .unwrap_or(std::ptr::null_mut()),
            },
            { done_with_h2d_transfer },
        )?;
        let event = unsafe { Event::from_c_api(handle, self.api(), ()) }?;

        // Register a callback to decrease the reference count of `data` once the transfer is completed.
        if let Some(drop_fn) = data.drop_fn {
            // Register the callback that will be invoked once the host buffer data has been copied.
            event.on_ready(|_| {
                // We ignore the error because there is nothing we can do with it here,
                // and if something goes wrong, it should be reflected in [`Buffer::ready`].
                drop_fn();
            })?;
        }

        Ok(event)
    }

    /// Sets the `index`-th [`Buffer`] managed by this [`HostToDeviceTransferManager`] to an "error" buffer that
    /// contains the provided [`Error`]. Refer to [`Client::error_buffer`] for information on "error" buffers.
    /// The provided `index` must be smaller than the result of [`HostToDeviceTransferManager::buffer_count`]
    /// for this transfer manager.
    ///
    /// # Safety
    ///
    /// You must not call [`HostToDeviceTransferManager::transfer_data`] using the same buffer `index` after calling
    /// this function. Doing so may result in undefined behavior since not all PJRT [`Plugin`](crate::Plugin)s appear
    /// to handle this scenario consistently.
    pub fn set_error(&self, index: usize, error: Error) -> Result<(), Error> {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args;
        let error_message = error.message();
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_SetBufferError,
            {
                transfer_manager = self.to_c_api(),
                buffer_index = index as std::ffi::c_int,
                error_code = error.code(),
                error_message = error_message.as_ptr(),
                error_message_size = error_message.count_bytes(),
            },
        )
    }

    /// Attaches the provided metadata to this [`HostToDeviceTransferManager`]. This function allows frameworks to
    /// attach arbitrary, plugin-specific attributes to transfers using [`Vec`]s containing [`NamedValue`]s. It is
    /// an extensibility mechanism, allowing frameworks to pass hints or extra information to PJRT plugins without
    /// changing the stable PJRT C API.
    ///
    /// Typically, the attached metadata are visited by the underlying PJRT plugin and are only used to modify how the
    /// plugin handles the upcoming data transfer if the plugin recognizes them. Otherwise, they are typically ignored
    /// allowing for better forward and backward compatibility.
    ///
    /// # Examples
    ///
    /// The following are some examples of how one could leverage transfer metadata:
    ///
    ///   - **Debugging & Profiling Names:** Using keys like `"debug_name"` and `"variable_name"` and values like
    ///     `"ResNet/conv/weights"` to label memory allocations in hardware profilers or debug logs, making it easier
    ///     to trace where memory is going.
    ///   - **Memory Layout Hints:** Using keys like `"layout_id"` and `"memory_space"` and values like
    ///     `"SRAM-friendly"` to hint where a target buffer for a transfer should be placed.
    ///   - **Source Information:** Using keys like `"source_location"` paired with values that are source descriptions
    ///     to provide information about where the data that is being transferred is coming from (e.g., a specific host
    ///     thread or NUMA node) in order to optimize the Direct Memory Access (DMA) engine settings.
    pub fn add_metadata(&self, metadata: Vec<NamedValue>) -> Result<(), Error> {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args;
        let metadata = metadata.iter().map(|value| unsafe { value.to_c_api() }).collect::<Vec<_>>();
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_AddMetadata,
            {
                transfer_manager = self.to_c_api(),
                transfer_metadata = metadata.as_slice().as_ptr(),
                num_metadata = metadata.len(),
            },
        )
    }

    /// Retrieves the `index`-th [`Buffer`] managed by this [`HostToDeviceTransferManager`], taking ownership of that
    /// buffer. If this function is called multiple times for the same `index`, then only the first call will be
    /// successful because afterward the manager will not have ownership of that buffer anymore. The provided `index`
    /// must be smaller than the result of [`HostToDeviceTransferManager::buffer_count`] for this transfer manager.
    ///
    /// Note that even if the returned [`Buffer`] is dropped before the underlying transfers have been completed,
    /// nothing unexpected will happen since PJRT buffers act as reference-counted objects. In that case, the PJRT
    /// runtime (and the underlying device driver) will keep the actual physical memory and the transfer operation
    /// alive until the hardware transfer completes. Once the transfer finishes and the runtime sees the reference
    /// count is zero, it will then automatically free the corresponding device memory.
    pub fn retrieve_buffer(&self, index: usize) -> Result<Buffer<'c>, Error> {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer,
            { transfer_manager = self.to_c_api(), buffer_index = index as std::ffi::c_int },
            { buffer_out }
        )
        .and_then(|handle| unsafe { Buffer::from_c_api(handle, self.api(), self.client) })
    }
}

impl Drop for HostToDeviceTransferManager<'_> {
    fn drop(&mut self) {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_AsyncHostToDeviceTransferManager_Destroy, {
            buffer = self.to_c_api(),
        })
        .expect("failed to destroy PJRT host-to-device transfer manager");
    }
}

impl<'s> Client<'s> {
    /// Creates a new [`HostToDeviceTransferManager`], allocating space for the [`Buffer`]s specified in the provided
    /// [`BufferSpecification`]s in the provided [`Memory`].
    pub fn host_to_device_transfer_manager<D: AsRef<[u64]>>(
        &'_ self,
        buffer_specifications: Vec<BufferSpecification<D>>,
        memory: Memory,
    ) -> Result<HostToDeviceTransferManager<'_>, Error> {
        use ffi::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args;
        let shapes = buffer_specifications
            .iter()
            .map(|specification| {
                ffi::PJRT_ShapeSpec::new(
                    specification.dimensions.as_ref().as_ptr() as *const i64,
                    specification.dimensions.as_ref().len(),
                    unsafe { specification.element_type.to_c_api() },
                )
            })
            .collect::<Vec<_>>();
        let layouts = buffer_specifications
            .iter()
            .map(|specification| specification.layout.as_ref().map(|layout| unsafe { layout.to_c_api() }))
            .collect::<Vec<_>>();
        let layouts = layouts
            .iter()
            .map(|layout| layout.as_ref().map(|layout| layout as *const _ as *mut _).unwrap_or(std::ptr::null_mut()))
            .collect::<Vec<*mut crate::buffers::ffi::PJRT_Buffer_MemoryLayout>>();
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_CreateBuffersForAsyncHostToDevice,
            {
                client = self.to_c_api(),
                shape_specs = shapes.as_ptr() as *mut _,
                num_shape_specs = shapes.len(),
                device_layouts = layouts.as_ptr() as *mut _,
                num_device_layouts = layouts.len(),
                memory = memory.to_c_api(),
            },
            { transfer_manager },
        )
        .and_then(|handle| unsafe { HostToDeviceTransferManager::from_c_api(handle, self.api(), self.to_c_api()) })
    }
}

/// Sized chunk of host data that can be either in host layout or in device layout, and it can be one part of the
/// entire buffer. Different PJRT implementations can customize how the memory is allocated and deallocated.
pub struct Chunk {
    /// PJRT C API representation of this [`Chunk`].
    handle: *mut ffi::PJRT_Chunk,
}

impl Chunk {
    /// Constructs a new [`Chunk`] from the provided [`PJRT_Chunk`](ffi::PJRT_Chunk) handle that came
    /// from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(handle: *mut ffi::PJRT_Chunk) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT chunk handle is a null pointer"))
        } else {
            Ok(Self { handle })
        }
    }

    /// Returns a pointer to the underlying data (i.e., bytes) contained in this [`Chunk`].
    pub fn data(&self) -> &[u8] {
        unsafe { slice_from_c_api((*self.handle).data as *const u8, (*self.handle).size) }
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        unsafe {
            if let Some(deleter) = (*self.handle).deleter {
                deleter((*self.handle).data, (*self.handle).deleter_arg);
            }
        }
    }
}

/// Stream that copies [`Chunk`]s of data from the host to a [`Device`].
///
/// The lifetime parameter `'c` captures the lifetime of the [`Client`] that owns this [`CopyToDeviceStream`],
/// ensuring that the client outlives the stream.
pub struct CopyToDeviceStream<'c> {
    /// Handle that represents this [`CopyToDeviceStream`] in the PJRT C API.
    handle: *mut ffi::PJRT_CopyToDeviceStream,

    /// Underlying PJRT [`Api`].
    api: Api,

    /// [`PhantomData`] used to track the lifetime of the [`Client`] that owns this [`CopyToDeviceStream`].
    owner: PhantomData<&'c ()>,
}

impl CopyToDeviceStream<'_> {
    /// Constructs a new [`CopyToDeviceStream`] from the provided
    /// [`PJRT_CopyToDeviceStream`](ffi::PJRT_CopyToDeviceStream) handle that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(handle: *mut ffi::PJRT_CopyToDeviceStream, api: Api) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT copy-to-device stream handle is a null pointer"))
        } else {
            Ok(Self { handle, api, owner: PhantomData })
        }
    }

    /// Returns the [`PJRT_CopyToDeviceStream`](ffi::PJRT_CopyToDeviceStream) that corresponds to this
    /// [`CopyToDeviceStream`] and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_CopyToDeviceStream {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }

    /// Returns the amount of data (as a number of bytes) that this [`CopyToDeviceStream`] has either already
    /// transferred or has buffered to transfer.
    pub fn current_byte_count(&self) -> Result<usize, Error> {
        use ffi::PJRT_CopyToDeviceStream_CurrentBytes_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_CopyToDeviceStream_CurrentBytes, { stream = self.to_c_api() }, {
            current_bytes
        })
        .map(|count| count as usize)
    }

    /// Returns the total amount of data (as a number of bytes) that this [`CopyToDeviceStream`]
    /// expects to be transferred.
    pub fn total_byte_count(&self) -> Result<usize, Error> {
        use ffi::PJRT_CopyToDeviceStream_TotalBytes_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_CopyToDeviceStream_TotalBytes, { stream = self.to_c_api() }, {
            total_bytes
        })
        .map(|count| count as usize)
    }

    /// Returns the granule size (as a number of bytes) of this [`CopyToDeviceStream`]. The size of each [`Chunk`]
    /// added to this stream must be a multiple of this number.
    pub fn granule_byte_count(&self) -> Result<usize, Error> {
        use ffi::PJRT_CopyToDeviceStream_GranuleSize_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_CopyToDeviceStream_GranuleSize, { stream = self.to_c_api() }, {
            granule_size_in_bytes
        })
        .map(|count| count as usize)
    }

    /// Enqueues a new [`Chunk`] of data to the copy to the target [`Device`]. The transfer starts immediately, and
    /// this function returns an [`Event`] that will be triggered when the transfer completes or fails. Note that the
    /// transfer will fail if the provided [`Chunk`]'s size causes the amount of transferred data to exceed the total
    /// number of bytes expected by this stream, if this stream is already complete, or if the [`Chunk`] size is not a
    /// multiple of this stream's [`CopyToDeviceStream::granule_byte_count`].
    pub fn add_chunk(&self, chunk: Chunk) -> Result<Event<()>, Error> {
        use ffi::PJRT_CopyToDeviceStream_AddChunk_Args;
        let result = invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_CopyToDeviceStream_AddChunk,
            { stream = self.handle, chunk = chunk.handle },
            { transfer_complete },
        )
        .and_then(|handle| unsafe { Event::from_c_api(handle, self.api(), ()) });
        std::mem::forget(chunk);
        result
    }
}

unsafe impl Send for CopyToDeviceStream<'_> {}
unsafe impl Sync for CopyToDeviceStream<'_> {}

impl Drop for CopyToDeviceStream<'_> {
    fn drop(&mut self) {
        use ffi::PJRT_CopyToDeviceStream_Destroy_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_CopyToDeviceStream_Destroy, { stream = self.to_c_api() })
            .expect("failed to destroy PJRT copy-to-device stream");
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::buffers::ffi::{PJRT_Buffer, PJRT_Buffer_MemoryLayout, PJRT_Buffer_Type};
    use crate::clients::ffi::PJRT_Client;
    use crate::devices::ffi::PJRT_Device;
    use crate::errors::ffi::{PJRT_Error, PJRT_Error_Code};
    use crate::events::ffi::PJRT_Event;
    use crate::ffi::PJRT_Extension_Base;
    use crate::memories::ffi::PJRT_Memory;
    use crate::values::ffi::PJRT_NamedValue;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_AsyncHostToDeviceTransferManager {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_AsyncHostToDeviceTransferManager_Device_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
        pub device_out: *mut PJRT_Device,
    }

    impl PJRT_AsyncHostToDeviceTransferManager_Device_Args {
        pub fn new(transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                transfer_manager,
                device_out: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_AsyncHostToDeviceTransferManager_Device =
        unsafe extern "C" fn(args: *mut PJRT_AsyncHostToDeviceTransferManager_Device_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
        pub buffer_count: usize,
    }

    impl PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args {
        pub fn new(transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                transfer_manager,
                buffer_count: 0,
            }
        }
    }

    pub type PJRT_AsyncHostToDeviceTransferManager_BufferCount =
        unsafe extern "C" fn(args: *mut PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
        pub buffer_index: std::ffi::c_int,
        pub buffer_size: usize,
    }

    impl PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args {
        pub fn new(
            transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
            buffer_index: std::ffi::c_int,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                transfer_manager,
                buffer_index,
                buffer_size: 0,
            }
        }
    }

    pub type PJRT_AsyncHostToDeviceTransferManager_BufferSize =
        unsafe extern "C" fn(args: *mut PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
        pub buffer_index: std::ffi::c_int,
        pub buffer_out: *mut PJRT_Buffer,
    }

    impl PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args {
        pub fn new(
            transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
            buffer_index: std::ffi::c_int,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                transfer_manager,
                buffer_index,
                buffer_out: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer =
        unsafe extern "C" fn(args: *mut PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_AsyncHostToDeviceTransferManager_TransferData_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
        pub buffer_index: std::ffi::c_int,
        pub data: *const std::ffi::c_void,
        pub offset: i64,
        pub transfer_size: i64,
        pub is_last_transfer: bool,
        pub done_with_h2d_transfer: *mut PJRT_Event,
    }

    impl PJRT_AsyncHostToDeviceTransferManager_TransferData_Args {
        pub fn new(
            transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
            buffer_index: std::ffi::c_int,
            data: *const std::ffi::c_void,
            offset: i64,
            transfer_size: i64,
            is_last_transfer: bool,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                transfer_manager,
                buffer_index,
                data,
                offset,
                transfer_size,
                is_last_transfer,
                done_with_h2d_transfer: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_AsyncHostToDeviceTransferManager_TransferData =
        unsafe extern "C" fn(args: *mut PJRT_AsyncHostToDeviceTransferManager_TransferData_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
        pub buffer_index: std::ffi::c_int,
        pub data: *const std::ffi::c_void,
        pub shape_dims: *const i64,
        pub shape_num_dims: usize,
        pub shape_element_type: PJRT_Buffer_Type,
        pub shape_layout: *mut PJRT_Buffer_MemoryLayout,
        pub done_with_h2d_transfer: *mut PJRT_Event,
    }

    impl PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args {
        pub fn new(
            transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
            buffer_index: std::ffi::c_int,
            data: *const std::ffi::c_void,
            shape_dims: *const i64,
            shape_num_dims: usize,
            shape_element_type: PJRT_Buffer_Type,
            shape_layout: *mut PJRT_Buffer_MemoryLayout,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                transfer_manager,
                buffer_index,
                data,
                shape_dims,
                shape_num_dims,
                shape_element_type,
                shape_layout,
                done_with_h2d_transfer: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_AsyncHostToDeviceTransferManager_TransferLiteral =
        unsafe extern "C" fn(args: *mut PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
        pub buffer_index: std::ffi::c_int,
        pub error_code: PJRT_Error_Code,
        pub error_message: *const std::ffi::c_char,
        pub error_message_size: usize,
    }

    impl PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args {
        pub fn new(
            transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
            buffer_index: std::ffi::c_int,
            error_code: PJRT_Error_Code,
            error_message: *const std::ffi::c_char,
            error_message_size: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                transfer_manager,
                buffer_index,
                error_code,
                error_message,
                error_message_size,
            }
        }
    }

    pub type PJRT_AsyncHostToDeviceTransferManager_SetBufferError =
        unsafe extern "C" fn(args: *mut PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
        pub transfer_metadata: *const PJRT_NamedValue,
        pub num_metadata: usize,
    }

    impl PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args {
        pub fn new(
            transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
            transfer_metadata: *const PJRT_NamedValue,
            num_metadata: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                transfer_manager,
                transfer_metadata,
                num_metadata,
            }
        }
    }

    pub type PJRT_AsyncHostToDeviceTransferManager_AddMetadata =
        unsafe extern "C" fn(args: *mut PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_AsyncHostToDeviceTransferManager_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
    }

    impl PJRT_AsyncHostToDeviceTransferManager_Destroy_Args {
        pub fn new(transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), transfer_manager }
        }
    }

    pub type PJRT_AsyncHostToDeviceTransferManager_Destroy =
        unsafe extern "C" fn(args: *mut PJRT_AsyncHostToDeviceTransferManager_Destroy_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_ShapeSpec {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub dims: *const i64,
        pub num_dims: usize,
        pub element_type: PJRT_Buffer_Type,
    }

    impl PJRT_ShapeSpec {
        pub fn new(dims: *const i64, num_dims: usize, element_type: PJRT_Buffer_Type) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), dims, num_dims, element_type }
        }
    }

    #[repr(C)]
    pub struct PJRT_Client_CreateBuffersForAsyncHostToDevice_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub shape_specs: *mut PJRT_ShapeSpec,
        pub num_shape_specs: usize,
        pub device_layouts: *mut *mut PJRT_Buffer_MemoryLayout,
        pub num_device_layouts: usize,
        pub memory: *mut PJRT_Memory,
        pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
    }

    impl PJRT_Client_CreateBuffersForAsyncHostToDevice_Args {
        pub fn new(
            client: *mut PJRT_Client,
            shape_specs: *mut PJRT_ShapeSpec,
            num_shape_specs: usize,
            device_layouts: *mut *mut PJRT_Buffer_MemoryLayout,
            num_device_layouts: usize,
            memory: *mut PJRT_Memory,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                shape_specs,
                num_shape_specs,
                device_layouts,
                num_device_layouts,
                memory,
                transfer_manager: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Client_CreateBuffersForAsyncHostToDevice =
        unsafe extern "C" fn(args: *mut PJRT_Client_CreateBuffersForAsyncHostToDevice_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Chunk {
        pub data: *mut std::ffi::c_void,
        pub size: usize,
        pub deleter: Option<unsafe extern "C" fn(data: *mut std::ffi::c_void, deleter_arg: *mut std::ffi::c_void)>,
        pub deleter_arg: *mut std::ffi::c_void,
    }

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_CopyToDeviceStream {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_CopyToDeviceStream_CurrentBytes_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub stream: *mut PJRT_CopyToDeviceStream,
        pub current_bytes: i64,
    }

    impl PJRT_CopyToDeviceStream_CurrentBytes_Args {
        pub fn new(stream: *mut PJRT_CopyToDeviceStream) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), stream, current_bytes: 0 }
        }
    }

    pub type PJRT_CopyToDeviceStream_CurrentBytes =
        unsafe extern "C" fn(args: *mut PJRT_CopyToDeviceStream_CurrentBytes_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_CopyToDeviceStream_TotalBytes_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub stream: *mut PJRT_CopyToDeviceStream,
        pub total_bytes: i64,
    }

    impl PJRT_CopyToDeviceStream_TotalBytes_Args {
        pub fn new(stream: *mut PJRT_CopyToDeviceStream) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), stream, total_bytes: 0 }
        }
    }

    pub type PJRT_CopyToDeviceStream_TotalBytes =
        unsafe extern "C" fn(args: *mut PJRT_CopyToDeviceStream_TotalBytes_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_CopyToDeviceStream_GranuleSize_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub stream: *mut PJRT_CopyToDeviceStream,
        pub granule_size_in_bytes: i64,
    }

    impl PJRT_CopyToDeviceStream_GranuleSize_Args {
        pub fn new(stream: *mut PJRT_CopyToDeviceStream) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                stream,
                granule_size_in_bytes: 0,
            }
        }
    }

    pub type PJRT_CopyToDeviceStream_GranuleSize =
        unsafe extern "C" fn(args: *mut PJRT_CopyToDeviceStream_GranuleSize_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_CopyToDeviceStream_AddChunk_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub stream: *mut PJRT_CopyToDeviceStream,
        pub chunk: *mut PJRT_Chunk,
        pub transfer_complete: *mut PJRT_Event,
    }

    impl PJRT_CopyToDeviceStream_AddChunk_Args {
        pub fn new(stream: *mut PJRT_CopyToDeviceStream, chunk: *mut PJRT_Chunk) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                stream,
                chunk,
                transfer_complete: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_CopyToDeviceStream_AddChunk =
        unsafe extern "C" fn(args: *mut PJRT_CopyToDeviceStream_AddChunk_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_CopyToDeviceStream_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub stream: *mut PJRT_CopyToDeviceStream,
    }

    impl PJRT_CopyToDeviceStream_Destroy_Args {
        pub fn new(stream: *mut PJRT_CopyToDeviceStream) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), stream }
        }
    }

    pub type PJRT_CopyToDeviceStream_Destroy =
        unsafe extern "C" fn(args: *mut PJRT_CopyToDeviceStream_Destroy_Args) -> *mut PJRT_Error;
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::ffi::c_void;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::tests::{test_cpu_plugin, test_for_each_platform};
    use crate::{
        BufferSpecification, BufferType, Chunk, CopyToDeviceStream, Error, HostToDeviceTransferManager, NamedValue,
    };

    use super::ffi;

    unsafe extern "C" fn chunk_deleter(_data: *mut c_void, deleter_arg: *mut c_void) {
        let counter = unsafe { &*(deleter_arg as *const AtomicUsize) };
        counter.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn test_null_pointer_handling() {
        let api = test_cpu_plugin().api();
        let client = std::ptr::NonNull::<crate::clients::ffi::PJRT_Client>::dangling().as_ptr();
        let manager = std::ptr::NonNull::<ffi::PJRT_AsyncHostToDeviceTransferManager>::dangling().as_ptr();
        assert!(matches!(
            unsafe { HostToDeviceTransferManager::from_c_api(std::ptr::null_mut(), api, client) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT async host-to-device transfer manager handle is a null pointer",
        ));
        assert!(matches!(
            unsafe { HostToDeviceTransferManager::from_c_api(manager, api, std::ptr::null_mut()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT client handle is a null pointer",
        ));
        assert!(matches!(
            unsafe { CopyToDeviceStream::from_c_api(std::ptr::null_mut(), api) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT copy-to-device stream handle is a null pointer",
        ));
    }

    #[test]
    fn test_host_to_device_transfer_manager() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let device = client.addressable_devices().unwrap().remove(0);
            let memory = device.default_memory().unwrap();
            let specification = BufferSpecification { element_type: BufferType::U8, dimensions: [8u64], layout: None };

            // Test a successful transfer.
            let manager = client.host_to_device_transfer_manager(vec![specification.clone()], memory).unwrap();
            assert_eq!(manager.device().unwrap().id(), device.id());
            assert_eq!(manager.buffer_count(), Ok(1));
            assert_eq!(manager.buffer_on_device_size_in_bytes(0), Ok(8));
            let rc = Rc::new(RefCell::new(&[1u8, 3u8, 5u8, 7u8]));
            assert!(manager.transfer_data(0, rc.clone(), 0, false).is_ok());
            let rc = Rc::new(RefCell::new(&[9u8, 11u8, 13u8, 15u8]));
            assert!(manager.transfer_data(0, rc.clone(), 4, true).unwrap().r#await().is_ok());
            assert!(manager.add_metadata(vec![NamedValue::new("4", "2")]).is_ok());
            let buffer = manager.retrieve_buffer(0).unwrap();
            assert_eq!(buffer.ready().unwrap().r#await(), Ok(()));
            assert_eq!(
                buffer.copy_to_host(None).unwrap().r#await().unwrap(),
                vec![1u8, 3u8, 5u8, 7u8, 9u8, 11u8, 13u8, 15u8],
            );

            // Test a failed transfer.
            let manager = client.host_to_device_transfer_manager(vec![specification.clone()], memory).unwrap();
            assert_eq!(manager.device().unwrap().id(), device.id());
            assert_eq!(manager.buffer_count(), Ok(1));
            assert_eq!(manager.buffer_on_device_size_in_bytes(0), Ok(8));
            let rc = Rc::new(RefCell::new(&[1u8, 3u8, 5u8, 7u8]));
            assert!(manager.transfer_data(0, rc.clone(), 0, false).is_ok());
            assert!(manager.set_error(0, Error::aborted("test error")).is_ok());
            assert!(manager.add_metadata(vec![NamedValue::new("4", "2")]).is_ok());
            let buffer = manager.retrieve_buffer(0).unwrap();
            assert!(matches!(
                buffer.ready().unwrap().r#await(),
                Err(Error::Aborted { message, .. }) if message.contains("test error"),
            ));

            // Test mixed transfers using both `transfer_data` and `transfer_literal`.
            let manager = client
                .host_to_device_transfer_manager(vec![specification.clone(), specification.clone()], memory)
                .unwrap();
            assert_eq!(manager.device().unwrap().id(), device.id());
            assert_eq!(manager.buffer_count(), Ok(2));
            assert_eq!(manager.buffer_on_device_size_in_bytes(0), Ok(8));
            assert_eq!(manager.buffer_on_device_size_in_bytes(1), Ok(8));
            let data_0 = Rc::new(RefCell::new(&[1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8]));
            assert!(manager.transfer_data(0, data_0.clone(), 0, true).unwrap().r#await().is_ok());
            let data_1 = Rc::new(RefCell::new(&[20u8, 21u8, 22u8, 23u8, 24u8, 25u8, 26u8, 27u8]));
            assert!(manager.transfer_literal(1, data_1.clone(), specification.clone()).unwrap().r#await().is_ok());
            let data_2 = Rc::new(RefCell::new(&[30u8, 31u8, 32u8, 33u8, 34u8, 35u8, 36u8, 37u8]));
            assert!(manager.transfer_literal(1, data_2.clone(), specification.clone()).is_err());
            let buffer_0 = manager.retrieve_buffer(0).unwrap();
            let buffer_1 = manager.retrieve_buffer(1).unwrap();
            assert_eq!(buffer_0.ready().unwrap().r#await(), Ok(()));
            assert_eq!(buffer_1.ready().unwrap().r#await(), Ok(()));
            assert_eq!(
                buffer_0.copy_to_host(None).unwrap().r#await().unwrap(),
                vec![1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8],
            );
            assert_eq!(
                buffer_1.copy_to_host(None).unwrap().r#await().unwrap(),
                vec![20u8, 21u8, 22u8, 23u8, 24u8, 25u8, 26u8, 27u8],
            );
        });
    }

    #[test]
    fn test_chunk() {
        let mut bytes = vec![201u8, 202u8, 203u8, 204u8];
        let deleted_counter = AtomicUsize::new(0);
        let mut ffi_chunk = ffi::PJRT_Chunk {
            data: bytes.as_mut_ptr() as *mut c_void,
            size: bytes.len(),
            deleter: Some(chunk_deleter),
            deleter_arg: &deleted_counter as *const AtomicUsize as *mut c_void,
        };

        // Create a [`Chunk`] with a nested scope so that we can verify that its `deleter` was called on `drop`.
        {
            let chunk = unsafe { Chunk::from_c_api(&mut ffi_chunk as *mut _) }.unwrap();
            assert_eq!(chunk.data(), bytes.as_slice());
            assert_eq!(deleted_counter.load(Ordering::SeqCst), 0);
        }

        assert_eq!(deleted_counter.load(Ordering::SeqCst), 1);

        // Test with an empty [`Chunk`].
        let mut value = 1u8;
        assert!(
            unsafe {
                Chunk::from_c_api(&mut ffi::PJRT_Chunk {
                    data: (&mut value as *mut u8) as *mut c_void,
                    size: 0,
                    deleter: None,
                    deleter_arg: std::ptr::null_mut(),
                } as *mut _)
            }
            .unwrap()
            .data()
            .is_empty()
        );
        assert!(
            unsafe {
                Chunk::from_c_api(&mut ffi::PJRT_Chunk {
                    data: std::ptr::null_mut(),
                    size: 8,
                    deleter: None,
                    deleter_arg: std::ptr::null_mut(),
                } as *mut _)
            }
            .unwrap()
            .data()
            .is_empty()
        );

        // Test creating a [`Chunk`] from a null pointer.
        assert!(matches!(
            unsafe { Chunk::from_c_api(std::ptr::null_mut()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT chunk handle is a null pointer",
        ));
    }
}
