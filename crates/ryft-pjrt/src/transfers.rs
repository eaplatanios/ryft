use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use crate::{
    Api, Buffer, BufferSpecification, BufferType, Client, Device, Error, Event, HostBufferData, Layout, Memory,
    NamedValue, invoke_pjrt_api_error_fn, slice_from_c_api,
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
///   3. Initiate / queue up data transfers using (potentially multiple calls to)
///      [`HostToDeviceTransferManager::transfer_data`], or a single call to
///      [`HostToDeviceTransferManager::transfer_literal`] or [`HostToDeviceTransferManager::set_error`],
///      for each [`Buffer`]. [`HostToDeviceTransferManager::supports_transfer_data`] and
///      [`HostToDeviceTransferManager::supports_transfer_literal`] report whether the PJRT plugin provides the
///      corresponding transfer functions, which lets callers choose a transfer function before they hand out the
///      [`Buffer`]s.
///   4. Retrieve the new [`Buffer`]s using [`HostToDeviceTransferManager::retrieve_buffer`]. Note that the resulting
///      [`Buffer`]s may not be immediately ready since the whole point of this transfer manager is that it handles
///      data transfers completely asynchronously.
///
/// # Transfer Validation
///
/// The [PJRT C API](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) forwards buffer indices to
/// [`AsyncHostToDeviceTransferManager`](https://github.com/openxla/xla/blob/main/xla/pjrt/pjrt_client.h) without
/// validating them, and the [reference implementation](
/// https://github.com/openxla/xla/blob/main/xla/pjrt/host_to_device_transfer_manager.cc) only checks them in debug
/// builds. That implementation also aborts the process when an error is set on a buffer that already received its last
/// transfer or an error, when data is transferred to a buffer after an error was set on it, and when a literal is
/// transferred to a buffer while raw data transfers to it are in flight. This wrapper therefore tracks the progress of
/// each managed [`Buffer`] and rejects such calls with an [`Error`] before they reach the PJRT plugin. Specifically,
/// a [`Buffer`] accepts either any number of [`HostToDeviceTransferManager::transfer_data`] calls or a single
/// [`HostToDeviceTransferManager::transfer_literal`] call, and nothing after its last transfer or an error.
/// It also remembers the element type and dimensions of each managed [`Buffer`], which lets
/// [`HostToDeviceTransferManager::transfer_literal`] verify that the host data covers the whole literal.
///
/// # Thread Safety
///
/// [`HostToDeviceTransferManager`]s are [`Send`] but not [`Sync`]. A typical usage pattern is to create the transfer
/// manager and retrieve its [`Buffer`]s on one thread, and then move the transfer manager to a background thread that
/// transfers the data and awaits the resulting [`Event`]s.
///
/// The lifetime parameter `'c` captures the lifetime of the [`Client`] that owns this [`HostToDeviceTransferManager`],
/// ensuring that the client outlives the transfer manager.
pub struct HostToDeviceTransferManager<'c> {
    /// Handle that represents this [`HostToDeviceTransferManager`] in the PJRT C API.
    handle: *mut ffi::PJRT_AsyncHostToDeviceTransferManager,

    /// Shape and transfer progress of each [`Buffer`] managed by this [`HostToDeviceTransferManager`], indexed by
    /// buffer index. The length of this slice is the [`HostToDeviceTransferManager::buffer_count`] of this transfer
    /// manager.
    buffers: Box<[HostToDeviceTransferBuffer]>,

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
    /// [`PJRT_AsyncHostToDeviceTransferManager`](ffi::PJRT_AsyncHostToDeviceTransferManager) handle that came from a
    /// function in the PJRT C API. The new transfer manager takes ownership of that handle even if this function fails
    /// after validating it.
    ///
    /// # Parameters
    ///
    ///   - `handle`: Handle of the PJRT transfer manager.
    ///   - `api`: PJRT [`Api`] that created `handle`.
    ///   - `client`: Handle of the PJRT client that owns `handle`.
    ///   - `buffers`: Element type and dimensions of each [`Buffer`] that `handle` was created for, in buffer index
    ///     order. Their number must match the buffer count that PJRT reports for `handle`.
    pub(crate) unsafe fn from_c_api(
        handle: *mut ffi::PJRT_AsyncHostToDeviceTransferManager,
        api: Api,
        client: *mut crate::clients::ffi::PJRT_Client,
        buffers: Vec<(BufferType, Vec<u64>)>,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument(
                "the provided PJRT async host-to-device transfer manager handle is a null pointer",
            ))
        } else if client.is_null() {
            Err(Error::invalid_argument("the provided PJRT client handle is a null pointer"))
        } else {
            let mut manager = Self { handle, buffers: Box::default(), api, client, owner: PhantomData };
            let buffer_count = manager.buffer_count()?;
            if buffer_count != buffers.len() {
                return Err(Error::internal(format!(
                    "the PJRT host-to-device transfer manager reports buffer count \
                     {} instead of the {} requested buffers",
                    buffer_count,
                    buffers.len(),
                )));
            }
            manager.buffers = buffers
                .into_iter()
                .map(|(element_type, dimensions)| HostToDeviceTransferBuffer {
                    element_type,
                    dimensions: dimensions.into_boxed_slice(),
                    retrieved: Cell::new(false),
                    progress: Cell::new(HostToDeviceTransferProgress::Pending),
                })
                .collect();
            Ok(manager)
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
        let (_, index) = self.buffer(index)?;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_BufferSize,
            { transfer_manager = self.to_c_api(), buffer_index = index },
            { buffer_size }
        )
    }

    /// Returns `true` if the PJRT plugin of this [`HostToDeviceTransferManager`] provides
    /// `PJRT_AsyncHostToDeviceTransferManager_TransferData`, which [`HostToDeviceTransferManager::transfer_data`]
    /// requires. If this function returns `false`, then [`HostToDeviceTransferManager::transfer_data`] fails with
    /// [`Error::MissingFunction`] without changing the state of this transfer manager.
    pub fn supports_transfer_data(&self) -> bool {
        unsafe {
            let api = self.api().to_c_api();
            let offset = std::mem::offset_of!(crate::ffi::PJRT_Api, PJRT_AsyncHostToDeviceTransferManager_TransferData);
            (*api).struct_size >= offset + size_of::<Option<ffi::PJRT_AsyncHostToDeviceTransferManager_TransferData>>()
                && (*api).PJRT_AsyncHostToDeviceTransferManager_TransferData.is_some()
        }
    }

    /// Returns `true` if the PJRT plugin of this [`HostToDeviceTransferManager`] provides
    /// `PJRT_AsyncHostToDeviceTransferManager_TransferLiteral`, which [`HostToDeviceTransferManager::transfer_literal`]
    /// requires. That function was added to the PJRT C API after the other transfer manager functions, and so a plugin
    /// can support transfer managers without supporting literal transfers. Callers that hand out the managed
    /// [`Buffer`]s before transferring data into them can use this function to choose a transfer function
    /// (or not to use a transfer manager at all) beforehand. If this function returns `false`, then
    /// [`HostToDeviceTransferManager::transfer_literal`] fails with [`Error::MissingFunction`]
    /// without changing the state of this transfer manager.
    pub fn supports_transfer_literal(&self) -> bool {
        unsafe {
            let api = self.api().to_c_api();
            let offset =
                std::mem::offset_of!(crate::ffi::PJRT_Api, PJRT_AsyncHostToDeviceTransferManager_TransferLiteral);
            (*api).struct_size
                >= offset + size_of::<Option<ffi::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral>>()
                && (*api).PJRT_AsyncHostToDeviceTransferManager_TransferLiteral.is_some()
        }
    }

    /// Transfers the provided data to the `index`-th [`Buffer`] managed by this [`HostToDeviceTransferManager`].
    ///
    /// # Parameters
    ///
    ///   - `index`: Index of the [`Buffer`] managed by this [`HostToDeviceTransferManager`] in which to transfer
    ///     the provided data.
    ///   - `data`: Data to transfer. Note that the data will be copied "as-is" (i.e., without performing any layout
    ///     transformations). This means that the data must already have the expected memory layout. PJRT may keep
    ///     reading `data` after this function returns and until the returned [`Event`] is ready, and so this function
    ///     keeps a clone of the provided [`Arc`] alive until then.
    ///   - `offset`: Offset in the storage representation of the target [`Buffer`] on which to transfer the provided
    ///     data. This enables transferring different chunks of a [`Buffer`] potentially from different buffers in the
    ///     host memory. `offset` plus the length of `data` must not exceed
    ///     [`HostToDeviceTransferManager::buffer_on_device_size_in_bytes`] for the target [`Buffer`].
    ///   - `is_last_transfer`: Boolean indicating whether this transfer should be treated as the last transfer for the
    ///     `index`-th [`Buffer`] managed by this [`HostToDeviceTransferManager`]. If `true`, then the [`Buffer`] will
    ///     be marked as "ready" after this transfer completes. Otherwise, it will not be marked as ready and will thus
    ///     also be unavailable to consumers after this transfer completes. Note that if `true`, then no further
    ///     transfer calls (via [`HostToDeviceTransferManager::transfer_data`] or
    ///     [`HostToDeviceTransferManager::set_error`]) will be allowed for the same buffer `index` in this
    ///     [`HostToDeviceTransferManager`]. Note that after any successful call to this function, the same buffer
    ///     no longer accepts [`HostToDeviceTransferManager::transfer_literal`] calls. Also note that the reference
    ///     PJRT implementation aborts the process when a transfer or an error follows a partial transfer to the same
    ///     buffer that has already failed asynchronously, which this wrapper cannot detect before calling PJRT.
    pub fn transfer_data<B: AsRef<[u8]> + Send + Sync + 'static>(
        &self,
        index: usize,
        data: Arc<B>,
        offset: usize,
        is_last_transfer: bool,
    ) -> Result<Event<()>, Error> {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args;
        let (buffer, c_index) = self.incomplete_buffer(index)?;
        let transfer_size = data.as_ref().as_ref().len();
        let buffer_size = self.buffer_on_device_size_in_bytes(index)?;
        if offset.checked_add(transfer_size).is_none_or(|end| end > buffer_size) {
            return Err(Error::invalid_argument(format!(
                "transfer size {transfer_size} at offset {offset} exceeds the size {buffer_size} of buffer {index} of \
                this PJRT host-to-device transfer manager",
            )));
        }
        let data = HostBufferData::from_host_buffer_arc(&data);
        let handle = invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_TransferData,
            {
                transfer_manager = self.to_c_api(),
                buffer_index = c_index,
                data = data.ptr,
                offset = offset as i64,
                transfer_size = transfer_size as i64,
                is_last_transfer = is_last_transfer,
            },
            { done_with_h2d_transfer },
        )?;
        buffer.progress.set(if is_last_transfer {
            HostToDeviceTransferProgress::Completed
        } else {
            HostToDeviceTransferProgress::PartiallyTransferred
        });

        // After a successful enqueue, PJRT may keep reading the host allocation until the transfer completes, so every
        // failure path below must leak the keep-alive `drop_fn` closure instead of dropping it. Dropping the closure
        // without invoking it would release our clone of `data` while the transfer may still be reading it. The
        // `ManuallyDrop` wrapper suppresses that release whenever the closure is dropped without being invoked.
        let drop_fn = data.drop_fn.map(ManuallyDrop::new);
        let event = unsafe { Event::from_c_api(handle, self.api(), ()) }?;

        // Register a callback to decrease the reference count of `data` once the transfer is completed.
        if let Some(drop_fn) = drop_fn {
            event.on_ready(move |_| {
                // We ignore the error because there is nothing we can do with it here,
                // and if something goes wrong, it should be reflected in `Buffer::ready`.
                ManuallyDrop::into_inner(drop_fn)();
            })?;
        }

        Ok(event)
    }

    /// Transfers the provided host buffer (interpreted as an XLA literal) to the `index`-th [`Buffer`] managed
    /// by this [`HostToDeviceTransferManager`], converting it to the layout of that [`Buffer`]. A literal transfer
    /// always transfers the whole [`Buffer`], and so it is the only transfer allowed for that [`Buffer`]. This function
    /// fails if [`HostToDeviceTransferManager::transfer_data`], [`HostToDeviceTransferManager::transfer_literal`], or
    /// [`HostToDeviceTransferManager::set_error`] has already succeeded for the same buffer `index`, and it fails with
    /// [`Error::MissingFunction`] if [`HostToDeviceTransferManager::supports_transfer_literal`] returns `false`.
    ///
    /// # XLA Literal Memory Layout
    ///
    /// The [PJRT C API wrapper](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_wrapper_impl.cc) wraps
    /// `data` directly in an [`xla::BorrowingLiteral`](https://github.com/openxla/xla/blob/main/xla/literal.cc) that
    /// carries no byte count. This means `data` is interpreted as a dense literal buffer directly (i.e., it is **not**
    /// parsed as a serialized [`LiteralProto`](https://github.com/openxla/xla/blob/main/xla/xla_data.proto)), and so
    /// this function validates the following before handing `data` to PJRT:
    ///
    ///   1. `specification.element_type` and `specification.dimensions` must be equal to the element type and
    ///      dimensions of the managed [`Buffer`]. PJRT linearizes the literal using the element type and dimensions of
    ///      the managed [`Buffer`] together with the byte strides of the literal, and so any mismatch would make it
    ///      read the wrong bytes.
    ///   2. `specification.layout` must be a [`TiledLayout`](crate::TiledLayout) without tiles, whose minor-to-major
    ///      dimension ordering is a permutation of the literal dimensions. Element ordering follows that minor-to-major
    ///      ordering as documented by [`IndexUtil::MultidimensionalIndexToLinearIndex`](
    ///      https://github.com/openxla/xla/blob/main/xla/index_util.h). PJRT rejects
    ///      [`StridedLayout`](crate::StridedLayout)s for literals and ignores tiles when computing the strides of a
    ///      literal, so neither is accepted here. The layout of [`BufferType::Token`] literals is ignored.
    ///   3. `data` must contain exactly as many bytes as the literal, which is the number of elements times the dense
    ///      storage size of one element (i.e., [`BufferType::element_size_in_bytes`], which stores sub-byte types in
    ///      one byte per element). In XLA terms, this corresponds to [`ShapeUtil::ByteSizeOf`](
    ///      https://github.com/openxla/xla/blob/main/xla/shape_util.h). [`BufferType::Token`] literals have no bytes.
    ///
    /// The returned [`Event`] becomes ready once PJRT is done consuming the host literal bytes.
    ///
    /// # Parameters
    ///
    ///   - `index`: Index of the [`Buffer`] managed by this [`HostToDeviceTransferManager`] in which to transfer
    ///     the provided literal.
    ///   - `data`: Host buffer containing the literal bytes to transfer. Refer to the memory layout contract above
    ///     for information on the memory layout of this buffer. PJRT may keep reading `data` after this function
    ///     returns and until the returned [`Event`] is ready, and so this function keeps a clone of the provided
    ///     [`Arc`] alive until then.
    ///   - `specification`: [`BufferSpecification`] describing how the provided `data` should be interpreted
    ///     as an XLA literal.
    pub fn transfer_literal<B: AsRef<[u8]> + Send + Sync + 'static, D: AsRef<[u64]>>(
        &self,
        index: usize,
        data: Arc<B>,
        specification: BufferSpecification<D>,
    ) -> Result<Event<()>, Error> {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args;
        let (buffer, c_index) = self.incomplete_buffer(index)?;
        if buffer.progress.get() == HostToDeviceTransferProgress::PartiallyTransferred {
            return Err(Error::failed_precondition(format!(
                "buffer {index} of this PJRT host-to-device transfer manager already received raw data transfers and \
                cannot also receive a literal transfer",
            )));
        }
        self.validate_literal(index, buffer, data.as_ref().as_ref().len(), &specification)?;
        let dimensions = specification.dimensions.as_ref().iter().map(|&d| d as i64).collect::<Vec<_>>();
        let data = HostBufferData::from_host_buffer_arc(&data);
        let handle = invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_TransferLiteral,
            {
                transfer_manager = self.to_c_api(),
                buffer_index = c_index,
                data = data.ptr,
                shape_dims = dimensions.as_ptr(),
                shape_num_dims = dimensions.len(),
                shape_element_type = specification.element_type.to_c_api(),
                shape_layout = &specification.layout.to_c_api() as *const _ as *mut _,
            },
            { done_with_h2d_transfer },
        )?;
        buffer.progress.set(HostToDeviceTransferProgress::Completed);

        // After a successful enqueue, PJRT may keep reading the host allocation until the transfer completes, so every
        // failure path below must leak the keep-alive `drop_fn` closure instead of dropping it. Dropping the closure
        // without invoking it would release our clone of `data` while the transfer may still be reading it. The
        // `ManuallyDrop` wrapper suppresses that release whenever the closure is dropped without being invoked.
        let drop_fn = data.drop_fn.map(ManuallyDrop::new);
        let event = unsafe { Event::from_c_api(handle, self.api(), ()) }?;

        // Register a callback to decrease the reference count of `data` once the transfer is completed.
        if let Some(drop_fn) = drop_fn {
            event.on_ready(move |_| {
                // We ignore the error because there is nothing we can do with it here,
                // and if something goes wrong, it should be reflected in `Buffer::ready`.
                ManuallyDrop::into_inner(drop_fn)();
            })?;
        }

        Ok(event)
    }

    /// Sets the `index`-th [`Buffer`] managed by this [`HostToDeviceTransferManager`] to an "error" buffer that
    /// contains the provided [`Error`]. Refer to [`Client::error_buffer`] for information on "error" buffers.
    /// The provided `index` must be smaller than the result of [`HostToDeviceTransferManager::buffer_count`]
    /// for this transfer manager.
    ///
    /// No further transfer or error calls are allowed for the same buffer `index` after this function succeeds, and
    /// this function fails if the last transfer or an error has already been submitted for that buffer. It may follow
    /// [`HostToDeviceTransferManager::transfer_data`] calls that were not the last transfer for that buffer. Note that
    /// the reference PJRT implementation also aborts the process if one of those partial transfers has already failed
    /// asynchronously, which this wrapper cannot detect before calling PJRT, and so callers that set an error after
    /// partial transfers should await their [`Event`]s first and skip this function if any of them failed.
    pub fn set_error(&self, index: usize, error: Error) -> Result<(), Error> {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args;
        let (buffer, c_index) = self.incomplete_buffer(index)?;
        let error_message = error.message();
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_SetBufferError,
            {
                transfer_manager = self.to_c_api(),
                buffer_index = c_index,
                error_code = error.code(),
                error_message = error_message.as_ptr(),
                error_message_size = error_message.count_bytes(),
            },
        )?;
        buffer.progress.set(HostToDeviceTransferProgress::Completed);
        Ok(())
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
    /// The returned [`Buffer`] is tied to the [`Client`] and not to this transfer manager, and so it can outlive it.
    ///
    /// Note that even if the returned [`Buffer`] is dropped before the underlying transfers have been completed,
    /// nothing unexpected will happen since PJRT buffers act as reference-counted objects. In that case, the PJRT
    /// runtime (and the underlying device driver) will keep the actual physical memory and the transfer operation
    /// alive until the hardware transfer completes. Once the transfer finishes and the runtime sees the reference
    /// count is zero, it will then automatically free the corresponding device memory.
    pub fn retrieve_buffer(&self, index: usize) -> Result<Buffer<'c>, Error> {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args;
        let (buffer, c_index) = self.buffer(index)?;
        if buffer.retrieved.get() {
            return Err(Error::failed_precondition(format!(
                "buffer {index} has already been retrieved from this PJRT host-to-device transfer manager",
            )));
        }
        let handle = invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer,
            { transfer_manager = self.to_c_api(), buffer_index = c_index },
            { buffer_out }
        )?;
        buffer.retrieved.set(true);
        unsafe { Buffer::from_c_api(handle, self.api(), self.client) }
    }

    /// Returns the `index`-th [`HostToDeviceTransferBuffer`] of this [`HostToDeviceTransferManager`], along with that
    /// index converted to the integer type used by the PJRT C API, or an [`Error`] if `index` is out of range.
    fn buffer(&self, index: usize) -> Result<(&HostToDeviceTransferBuffer, std::ffi::c_int), Error> {
        match (self.buffers.get(index), std::ffi::c_int::try_from(index)) {
            (Some(buffer), Ok(c_index)) => Ok((buffer, c_index)),
            _ => Err(Error::invalid_argument(format!(
                "buffer index {} is out of range for a PJRT host-to-device transfer manager with buffer count {}",
                index,
                self.buffers.len(),
            ))),
        }
    }

    /// Returns the `index`-th [`HostToDeviceTransferBuffer`] of this [`HostToDeviceTransferManager`], along with that
    /// index converted to the integer type used by the PJRT C API, or an [`Error`] if `index` is out of range or if
    /// the last transfer or an error has already been submitted for that buffer.
    fn incomplete_buffer(&self, index: usize) -> Result<(&HostToDeviceTransferBuffer, std::ffi::c_int), Error> {
        let (buffer, c_index) = self.buffer(index)?;
        if buffer.progress.get() == HostToDeviceTransferProgress::Completed {
            Err(Error::failed_precondition(format!(
                "buffer {index} of this PJRT host-to-device transfer manager already received its last transfer \
                 or an error",
            )))
        } else {
            Ok((buffer, c_index))
        }
    }

    /// Validates that `data_size` bytes interpreted as a literal described by `specification` can be transferred
    /// into `buffer`, which is the `index`-th buffer of this [`HostToDeviceTransferManager`]. Refer to
    /// [`HostToDeviceTransferManager::transfer_literal`] for the rules that this function enforces.
    fn validate_literal<D: AsRef<[u64]>>(
        &self,
        index: usize,
        buffer: &HostToDeviceTransferBuffer,
        data_size: usize,
        specification: &BufferSpecification<D>,
    ) -> Result<(), Error> {
        let element_type = specification.element_type;
        let dimensions = specification.dimensions.as_ref();
        if element_type != buffer.element_type {
            return Err(Error::invalid_argument(format!(
                "literal element type `{}` does not match the element type `{}` of buffer {} of this \
                PJRT host-to-device transfer manager",
                element_type, buffer.element_type, index,
            )));
        }

        if element_type == BufferType::Token {
            return if data_size == 0 {
                Ok(())
            } else {
                Err(Error::invalid_argument(format!(
                    "literal data has {data_size} bytes but a `{element_type}` literal has no bytes",
                )))
            };
        }

        if dimensions != buffer.dimensions.as_ref() {
            return Err(Error::invalid_argument(format!(
                "literal dimensions {:?} do not match the dimensions {:?} of buffer {} of this PJRT \
                host-to-device transfer manager",
                dimensions, buffer.dimensions, index,
            )));
        }

        match &specification.layout {
            Layout::Tiled(layout) if !layout.tiles().is_empty() => {
                return Err(Error::invalid_argument(format!(
                    "literal layout `{}` has tiles, which PJRT literal transfers do not support",
                    specification.layout,
                )));
            }
            Layout::Tiled(layout) => {
                let mut minor_to_major = layout.minor_to_major().to_vec();
                minor_to_major.sort_unstable();
                if !minor_to_major.iter().copied().eq(0..dimensions.len() as u64) {
                    return Err(Error::invalid_argument(format!(
                        "literal layout `{}` is not a minor-to-major ordering of the dimensions of a rank-{} literal",
                        specification.layout,
                        dimensions.len(),
                    )));
                }
            }
            Layout::Strided(_) => {
                return Err(Error::invalid_argument(format!(
                    "literal layout `{}` is strided, which PJRT literal transfers do not support",
                    specification.layout,
                )));
            }
        }

        let literal_size = dense_size_in_bytes(element_type, dimensions)?;
        if data_size as u64 != literal_size {
            return Err(Error::invalid_argument(format!(
                "literal data has {data_size} bytes but a `{element_type}` literal with dimensions {dimensions:?} has \
                {literal_size} bytes",
            )));
        }

        Ok(())
    }
}

impl Drop for HostToDeviceTransferManager<'_> {
    fn drop(&mut self) {
        use ffi::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_AsyncHostToDeviceTransferManager_Destroy, {
            transfer_manager = self.to_c_api(),
        })
        .expect("failed to destroy PJRT host-to-device transfer manager");
    }
}

// PJRT transfer managers have no thread affinity, so moving one to another thread and using or destroying it there
// is sound. The PJRT C API does not state that a transfer manager supports concurrent calls (e.g., the reference
// implementation reads its buffer list without synchronization in `RetrieveBuffer`), and so this type is deliberately
// not `Sync`. This is enforced structurally by the `Cell`-based progress of its buffers.
unsafe impl Send for HostToDeviceTransferManager<'_> {}

impl<'s> Client<'s> {
    /// Creates a new [`HostToDeviceTransferManager`], allocating space for the [`Buffer`]s specified in the
    /// provided [`BufferSpecification`]s in the provided [`Memory`]. Each [`Buffer`] uses the `layout` of its
    /// [`BufferSpecification`] as its device layout, except for [`BufferType::Token`] specifications,
    /// which do not support array layouts and whose `layout` field is therefore ignored. Use
    /// [`Client::host_to_device_transfer_manager_with_default_layouts`] to let the PJRT
    /// plugin choose the device layouts instead.
    pub fn host_to_device_transfer_manager<D: AsRef<[u64]>>(
        &'_ self,
        buffer_specifications: Vec<BufferSpecification<D>>,
        memory: Memory,
    ) -> Result<HostToDeviceTransferManager<'_>, Error> {
        let buffers = buffer_specifications
            .iter()
            .map(|specification| (specification.element_type, specification.dimensions.as_ref().to_vec()))
            .collect();
        let layouts = buffer_specifications.iter().map(|specification| &specification.layout).collect();
        self.create_host_to_device_transfer_manager(buffers, Some(layouts), memory)
    }

    /// Creates a new [`HostToDeviceTransferManager`], allocating space in the provided [`Memory`] for [`Buffer`]s with
    /// the provided element types and dimensions, in the default device layouts that the PJRT plugin chooses for that
    /// [`Memory`]. This is equivalent to omitting the device layouts of
    /// `PJRT_Client_CreateBuffersForAsyncHostToDevice`, and it does not require the PJRT plugin to report its default
    /// layouts (e.g., through [`Client::default_layout`]). Use [`Client::host_to_device_transfer_manager`] to specify
    /// the device layouts explicitly.
    ///
    /// # Parameters
    ///
    ///   - `buffers`: Element type and dimensions of each [`Buffer`] to allocate, in buffer index order.
    ///   - `memory`: [`Memory`] in which to allocate the [`Buffer`]s.
    pub fn host_to_device_transfer_manager_with_default_layouts<D: AsRef<[u64]>>(
        &'_ self,
        buffers: Vec<(BufferType, D)>,
        memory: Memory,
    ) -> Result<HostToDeviceTransferManager<'_>, Error> {
        let buffers = buffers
            .into_iter()
            .map(|(element_type, dimensions)| (element_type, dimensions.as_ref().to_vec()))
            .collect();
        self.create_host_to_device_transfer_manager(buffers, None, memory)
    }

    /// Creates a new [`HostToDeviceTransferManager`] for [`Buffer`]s with the provided element types and dimensions
    /// in the provided [`Memory`]. If `layouts` is [`Some`], it contains the device layout of each [`Buffer`], and
    /// otherwise the PJRT plugin chooses the device layouts.
    fn create_host_to_device_transfer_manager(
        &'_ self,
        buffers: Vec<(BufferType, Vec<u64>)>,
        layouts: Option<Vec<&Layout>>,
        memory: Memory,
    ) -> Result<HostToDeviceTransferManager<'_>, Error> {
        use ffi::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args;

        // PJRT aborts the process when it cannot construct the shape of a buffer (e.g., because its dimensions are
        // negative when reinterpreted as `i64` values or because its size overflows `i64`), and so we validate the
        // shapes of non-token buffers here. PJRT ignores the dimensions of token buffers.
        for (element_type, dimensions) in &buffers {
            if *element_type != BufferType::Token {
                dense_size_in_bytes(*element_type, dimensions)?;
            }
        }

        let shapes = buffers
            .iter()
            .map(|(element_type, dimensions)| {
                ffi::PJRT_ShapeSpec::new(dimensions.as_ptr() as *const i64, dimensions.len(), unsafe {
                    element_type.to_c_api()
                })
            })
            .collect::<Vec<_>>();
        let layouts = layouts
            .unwrap_or_default()
            .into_iter()
            .map(|layout| unsafe { layout.to_c_api() })
            .collect::<Vec<_>>();
        let layouts = layouts
            .iter()
            .zip(&buffers)
            .map(|(layout, (element_type, _))| {
                // PJRT token shapes cannot carry a layout. Passing even an empty array layout makes
                // PJRT plugins attempt to mutate a token's nonexistent layout and abort.
                if *element_type == BufferType::Token { std::ptr::null_mut() } else { layout as *const _ as *mut _ }
            })
            .collect::<Vec<*mut crate::buffers::ffi::PJRT_Buffer_MemoryLayout>>();

        // An empty `device_layouts` array makes PJRT use the default layout of every buffer.
        let handle = invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_CreateBuffersForAsyncHostToDevice,
            {
                client = self.to_c_api(),
                shape_specs = shapes.as_ptr() as *mut _,
                num_shape_specs = shapes.len(),
                device_layouts = if layouts.is_empty() { std::ptr::null_mut() } else { layouts.as_ptr() as *mut _ },
                num_device_layouts = layouts.len(),
                memory = memory.to_c_api(),
            },
            { transfer_manager },
        )?;

        unsafe { HostToDeviceTransferManager::from_c_api(handle, self.api(), self.to_c_api(), buffers) }
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

/// [`Buffer`] managed by a [`HostToDeviceTransferManager`], described by the shape that it was created with and
/// the progress of the transfers into it.
struct HostToDeviceTransferBuffer {
    /// [`BufferType`] of the elements of the [`Buffer`].
    element_type: BufferType,

    /// Dimensions of the [`Buffer`].
    dimensions: Box<[u64]>,

    /// Boolean indicating whether the [`Buffer`] has already been retrieved
    /// using [`HostToDeviceTransferManager::retrieve_buffer`].
    retrieved: Cell<bool>,

    /// Progress of the transfers into the [`Buffer`].
    progress: Cell<HostToDeviceTransferProgress>,
}

/// Progress of the transfers into a [`Buffer`] managed by a [`HostToDeviceTransferManager`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum HostToDeviceTransferProgress {
    /// No transfer or error has been submitted for the [`Buffer`].
    Pending,

    /// One or more [`HostToDeviceTransferManager::transfer_data`] calls that were not the last transfer have been
    /// submitted for the [`Buffer`]. The [`Buffer`] accepts further raw data transfers and an error, but no literal.
    PartiallyTransferred,

    /// The last transfer (i.e., a raw data transfer marked as the last transfer or a literal transfer) or an error has
    /// been submitted for the [`Buffer`], which therefore accepts no further transfers or errors.
    Completed,
}

/// Returns the number of bytes that a dense array with the provided element type and dimensions occupies,
/// which is the number of elements times [`BufferType::element_size_in_bytes`]. This matches
/// [`ShapeUtil::ByteSizeOf`](https://github.com/openxla/xla/blob/main/xla/shape_util.h) for untiled shapes and it fails
/// when XLA cannot represent such an array (i.e., when the element type is [`BufferType::Invalid`] or when the size
/// overflows `i64`, which also covers dimensions that do not fit in `i64`).
fn dense_size_in_bytes(element_type: BufferType, dimensions: &[u64]) -> Result<u64, Error> {
    dimensions
        .iter()
        .try_fold(element_type.element_size_in_bytes()? as u64, |size, &dimension| {
            size.checked_mul(dimension).filter(|&size| size <= i64::MAX as u64 && dimension <= i64::MAX as u64)
        })
        .ok_or_else(|| {
            Error::invalid_argument(format!(
                "the size of a `{element_type}` array with dimensions {dimensions:?} overflows `i64`",
            ))
        })
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
    use std::ffi::c_void;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use crate::tests::{TestPlatform, test_cpu_client, test_cpu_plugin, test_for_each_platform};
    use crate::{
        Api, BufferSpecification, BufferType, Chunk, Client, CopyToDeviceStream, Error, HostToDeviceTransferManager,
        Layout, NamedValue, StridedLayout, Tile, TileDimension, TiledLayout,
    };

    use super::ffi;

    unsafe extern "C" fn chunk_deleter(_data: *mut c_void, deleter_arg: *mut c_void) {
        let counter = unsafe { &*(deleter_arg as *const AtomicUsize) };
        counter.fetch_add(1, Ordering::SeqCst);
    }

    /// Creates a [`HostToDeviceTransferManager`] for `specifications` in the default memory of the first addressable
    /// device of `client`. Returns [`None`] after asserting that `platform` does not support transfer managers.
    fn test_transfer_manager<'c>(
        client: &'c Client<'_>,
        platform: TestPlatform,
        specifications: Vec<BufferSpecification<Vec<u64>>>,
    ) -> Option<HostToDeviceTransferManager<'c>> {
        let device = client.addressable_devices().unwrap().remove(0);
        let manager = client.host_to_device_transfer_manager(specifications, device.default_memory().unwrap());
        match platform {
            TestPlatform::Mps => {
                assert!(matches!(
                    manager,
                    Err(Error::Unimplemented { message, .. })
                        if message == "`PJRT_Client_CreateBuffersForAsyncHostToDevice` is not implemented in the \
                        loaded PJRT plugin (version 0.104)",
                ));
                None
            }
            _ => Some(manager.unwrap()),
        }
    }

    #[test]
    fn test_null_pointer_handling() {
        let api = test_cpu_plugin().api();
        let client = std::ptr::NonNull::<crate::clients::ffi::PJRT_Client>::dangling().as_ptr();
        let manager = std::ptr::NonNull::<ffi::PJRT_AsyncHostToDeviceTransferManager>::dangling().as_ptr();
        assert!(matches!(
            unsafe { HostToDeviceTransferManager::from_c_api(std::ptr::null_mut(), api, client, Vec::new()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT async host-to-device transfer manager handle is a null pointer",
        ));
        assert!(matches!(
            unsafe { HostToDeviceTransferManager::from_c_api(manager, api, std::ptr::null_mut(), Vec::new()) },
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
        test_for_each_platform!(|_plugin, client, platform| {
            let device = client.addressable_devices().unwrap().remove(0);
            let specification = BufferSpecification::new(BufferType::U8, vec![8u64]);
            if let Some(manager) = test_transfer_manager(&client, platform, vec![specification.clone()]) {
                // Test a successful transfer.
                assert_eq!(manager.device().unwrap().id(), device.id());
                assert_eq!(manager.buffer_count(), Ok(1));
                assert_eq!(manager.buffer_on_device_size_in_bytes(0), Ok(8));
                let data = Arc::new([1u8, 3u8, 5u8, 7u8]);
                assert!(manager.transfer_data(0, data, 0, false).is_ok());
                let data = Arc::new([9u8, 11u8, 13u8, 15u8]);
                assert!(manager.transfer_data(0, data, 4, true).unwrap().r#await().is_ok());
                assert!(manager.add_metadata(vec![NamedValue::new("4", "2")]).is_ok());
                let buffer = manager.retrieve_buffer(0).unwrap();
                assert_eq!(buffer.ready().unwrap().r#await(), Ok(()));
                assert_eq!(
                    buffer.copy_to_host(None).unwrap().r#await().unwrap(),
                    vec![1u8, 3u8, 5u8, 7u8, 9u8, 11u8, 13u8, 15u8],
                );

                // Test a failed transfer.
                let manager = test_transfer_manager(&client, platform, vec![specification.clone()]).unwrap();
                assert_eq!(manager.device().unwrap().id(), device.id());
                assert_eq!(manager.buffer_count(), Ok(1));
                assert_eq!(manager.buffer_on_device_size_in_bytes(0), Ok(8));
                let data = Arc::new([1u8, 3u8, 5u8, 7u8]);
                assert!(manager.transfer_data(0, data, 0, false).is_ok());
                assert!(manager.set_error(0, Error::aborted("test error")).is_ok());
                assert!(manager.add_metadata(vec![NamedValue::new("4", "2")]).is_ok());
                let buffer = manager.retrieve_buffer(0).unwrap();
                assert!(matches!(
                    buffer.ready().unwrap().r#await(),
                    Err(Error::Aborted { message, .. }) if message.contains("test error"),
                ));

                // Test mixed transfers using both `transfer_data` and `transfer_literal`.
                let specifications = vec![specification.clone(), specification.clone()];
                let manager = test_transfer_manager(&client, platform, specifications).unwrap();
                assert_eq!(manager.device().unwrap().id(), device.id());
                assert_eq!(manager.buffer_count(), Ok(2));
                assert_eq!(manager.buffer_on_device_size_in_bytes(0), Ok(8));
                assert_eq!(manager.buffer_on_device_size_in_bytes(1), Ok(8));
                let data_0 = Arc::new([1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8]);
                assert!(manager.transfer_data(0, data_0, 0, true).unwrap().r#await().is_ok());
                let data_1 = Arc::new([20u8, 21u8, 22u8, 23u8, 24u8, 25u8, 26u8, 27u8]);
                assert!(manager.transfer_literal(1, data_1.clone(), specification.clone()).unwrap().r#await().is_ok(),);
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
            }
        });
    }

    #[test]
    fn test_host_to_device_transfer_manager_token() {
        let client = test_cpu_client();
        let devices = client.addressable_devices().unwrap();
        let memory = devices[0].default_memory().unwrap();
        let manager = client
            .host_to_device_transfer_manager(vec![BufferSpecification::new(BufferType::Token, [])], memory)
            .unwrap();
        let buffer = manager.retrieve_buffer(0).unwrap();
        assert_eq!(buffer.element_type(), Ok(BufferType::Token));
        assert_eq!(manager.buffer_on_device_size_in_bytes(0), Ok(0));
        assert_eq!(buffer.ready().unwrap().ready(), Ok(false));
        assert_eq!(manager.transfer_data(0, Arc::new([] as [u8; 0]), 0, true).unwrap().r#await(), Ok(()));
        assert_eq!(buffer.ready().unwrap().r#await(), Ok(()));
    }

    #[test]
    fn test_host_to_device_transfer_manager_token_error() {
        let client = test_cpu_client();
        let devices = client.addressable_devices().unwrap();
        let memory = devices[0].default_memory().unwrap();
        let manager = client
            .host_to_device_transfer_manager(vec![BufferSpecification::new(BufferType::Token, [])], memory)
            .unwrap();
        let buffer = manager.retrieve_buffer(0).unwrap();
        assert_eq!(buffer.ready().unwrap().ready(), Ok(false));
        assert_eq!(manager.set_error(0, Error::aborted("native token transfer failure")), Ok(()));
        assert!(matches!(
            buffer.ready().unwrap().r#await(),
            Err(Error::Aborted { message, .. }) if message.contains("native token transfer failure"),
        ));
    }

    #[test]
    fn test_host_to_device_transfer_manager_buffer_on_device_size_in_bytes() {
        test_for_each_platform!(|_plugin, client, platform| {
            let specifications = vec![
                BufferSpecification::new(BufferType::F32, vec![4u64]),
                BufferSpecification::new(BufferType::I32, vec![2u64, 2u64]),
            ];
            if let Some(manager) = test_transfer_manager(&client, platform, specifications) {
                assert_eq!(manager.buffer_count(), Ok(2));
                assert_eq!(manager.buffer_on_device_size_in_bytes(0), Ok(16));
                assert_eq!(manager.buffer_on_device_size_in_bytes(1), Ok(16));
                assert!(matches!(
                    manager.buffer_on_device_size_in_bytes(2),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "buffer index 2 is out of range for a PJRT host-to-device transfer manager \
                        with buffer count 2",
                ));
            }
        });
    }

    #[test]
    fn test_host_to_device_transfer_manager_supports_transfer_data() {
        test_for_each_platform!(|_plugin, client, platform| {
            let specifications = vec![BufferSpecification::new(BufferType::U8, vec![4u64])];
            if let Some(mut manager) = test_transfer_manager(&client, platform, specifications) {
                assert!(manager.supports_transfer_data());

                // Emulate a plugin without raw data transfers. The emulated API must outlive the transfer manager,
                // which uses it to destroy the native transfer manager, unless the original API is restored.
                let api = manager.api;
                let mut native_api = unsafe { std::ptr::read(api.to_c_api()) };
                native_api.PJRT_AsyncHostToDeviceTransferManager_TransferData = None;
                manager.api = unsafe { Api::from_c_api(&native_api) }.unwrap();
                assert!(!manager.supports_transfer_data());
                assert!(matches!(
                    manager.transfer_data(0, Arc::new([1u8, 2u8, 3u8, 4u8]), 0, true),
                    Err(Error::MissingFunction { function_name, .. })
                        if function_name == "PJRT_AsyncHostToDeviceTransferManager_TransferData",
                ));

                // The rejected transfer leaves the buffer available for a valid transfer.
                manager.api = api;
                let buffer = manager.retrieve_buffer(0).unwrap();
                assert_eq!(
                    manager.transfer_data(0, Arc::new([1u8, 2u8, 3u8, 4u8]), 0, true).unwrap().r#await(),
                    Ok(()),
                );
                assert_eq!(buffer.copy_to_host(None).unwrap().r#await(), Ok(vec![1u8, 2u8, 3u8, 4u8]));
            }
        });
    }

    #[test]
    fn test_host_to_device_transfer_manager_supports_transfer_literal() {
        test_for_each_platform!(|_plugin, client, platform| {
            let specification = BufferSpecification::new(BufferType::U8, vec![4u64]);
            if let Some(mut manager) = test_transfer_manager(&client, platform, vec![specification.clone()]) {
                assert!(manager.supports_transfer_literal());

                // Emulate a plugin that predates literal transfers by truncating its API struct. The emulated API must
                // outlive the transfer manager, which uses it to destroy the native transfer manager, unless the
                // original API is restored.
                let api = manager.api;
                let mut native_api = unsafe { std::ptr::read(api.to_c_api()) };
                native_api.struct_size =
                    std::mem::offset_of!(crate::ffi::PJRT_Api, PJRT_AsyncHostToDeviceTransferManager_TransferLiteral);
                manager.api = unsafe { Api::from_c_api(&native_api) }.unwrap();
                assert!(!manager.supports_transfer_literal());
                assert!(matches!(
                    manager.transfer_literal(0, Arc::new([1u8, 2u8, 3u8, 4u8]), specification.clone()),
                    Err(Error::MissingFunction { function_name, .. })
                        if function_name == "PJRT_AsyncHostToDeviceTransferManager_TransferLiteral",
                ));

                // The rejected transfer leaves the buffer available for a valid transfer.
                manager.api = api;
                let buffer = manager.retrieve_buffer(0).unwrap();
                assert_eq!(
                    manager.transfer_literal(0, Arc::new([1u8, 2u8, 3u8, 4u8]), specification).unwrap().r#await(),
                    Ok(()),
                );
                assert_eq!(buffer.copy_to_host(None).unwrap().r#await(), Ok(vec![1u8, 2u8, 3u8, 4u8]));
            }
        });
    }

    #[test]
    fn test_host_to_device_transfer_manager_transfer_data() {
        test_for_each_platform!(|_plugin, client, platform| {
            let specifications = vec![
                BufferSpecification::new(BufferType::F32, vec![4u64]),
                BufferSpecification::new(BufferType::I32, vec![2u64, 2u64]),
            ];
            if let Some(manager) = test_transfer_manager(&client, platform, specifications) {
                let f32_bytes =
                    [1.0f32, 2.5f32, -3.0f32, 4.25f32].iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>();
                let i32_bytes = [1i32, -2i32, 3i32, 4i32].iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>();

                // Retrieve the buffers on this thread and then transfer their data from a background thread.
                let buffer_0 = manager.retrieve_buffer(0).unwrap();
                let buffer_1 = manager.retrieve_buffer(1).unwrap();
                thread::scope(|scope| {
                    let f32_bytes = f32_bytes.clone();
                    let i32_bytes = i32_bytes.clone();
                    scope.spawn(move || {
                        assert_eq!(manager.transfer_data(0, Arc::new(f32_bytes), 0, true).unwrap().r#await(), Ok(()));
                        let first_chunk =
                            manager.transfer_data(1, Arc::new(i32_bytes[..8].to_vec()), 0, false).unwrap();
                        let second_chunk =
                            manager.transfer_data(1, Arc::new(i32_bytes[8..].to_vec()), 8, true).unwrap();
                        assert_eq!(first_chunk.r#await(), Ok(()));
                        assert_eq!(second_chunk.r#await(), Ok(()));

                        // Transfers after the last transfer for a buffer are rejected.
                        assert!(matches!(
                            manager.transfer_data(0, Arc::new([0u8; 4]), 0, true),
                            Err(Error::FailedPrecondition { message, .. })
                                if message == "buffer 0 of this PJRT host-to-device transfer manager already \
                                received its last transfer or an error",
                        ));
                    });
                });
                assert_eq!(buffer_0.ready().unwrap().r#await(), Ok(()));
                assert_eq!(buffer_1.ready().unwrap().r#await(), Ok(()));
                assert_eq!(buffer_0.copy_to_host(None).unwrap().r#await(), Ok(f32_bytes));
                assert_eq!(buffer_1.copy_to_host(None).unwrap().r#await(), Ok(i32_bytes));
            }
        });
    }

    #[test]
    fn test_host_to_device_transfer_manager_transfer_data_validation() {
        test_for_each_platform!(|_plugin, client, platform| {
            let specifications = vec![BufferSpecification::new(BufferType::U8, vec![4u64])];
            if let Some(manager) = test_transfer_manager(&client, platform, specifications) {
                assert!(matches!(
                    manager.transfer_data(1, Arc::new([0u8; 4]), 0, true),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "buffer index 1 is out of range for a PJRT host-to-device transfer manager \
                        with buffer count 1",
                ));
                assert!(matches!(
                    manager.transfer_data(0, Arc::new([0u8; 4]), 1, true),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "transfer size 4 at offset 1 exceeds the size 4 of buffer 0 of this PJRT \
                        host-to-device transfer manager",
                ));
                assert!(matches!(
                    manager.transfer_data(0, Arc::new([0u8; 1]), usize::MAX, true),
                    Err(Error::InvalidArgument { message, .. })
                        if message == format!(
                            "transfer size 1 at offset {} exceeds the size 4 of buffer 0 of this PJRT \
                            host-to-device transfer manager",
                            usize::MAX,
                        ),
                ));

                // Rejected transfers leave the buffer available for a valid transfer.
                let buffer = manager.retrieve_buffer(0).unwrap();
                assert_eq!(
                    manager.transfer_data(0, Arc::new([1u8, 2u8, 3u8, 4u8]), 0, true).unwrap().r#await(),
                    Ok(()),
                );
                assert_eq!(buffer.copy_to_host(None).unwrap().r#await(), Ok(vec![1u8, 2u8, 3u8, 4u8]));
            }
        });
    }

    #[test]
    fn test_host_to_device_transfer_manager_transfer_literal() {
        test_for_each_platform!(|_plugin, client, platform| {
            let specification = BufferSpecification::new(BufferType::U8, vec![2u64, 3u64]);
            if let Some(manager) = test_transfer_manager(&client, platform, vec![specification.clone()]) {
                assert!(matches!(
                    manager.transfer_literal(1, Arc::new([0u8; 6]), specification.clone()),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "buffer index 1 is out of range for a PJRT host-to-device transfer manager \
                        with buffer count 1",
                ));

                // PJRT converts a column-major literal to the row-major layout of the managed buffer.
                let column_major = Layout::Tiled(TiledLayout::new(vec![0u64, 1u64], Vec::new()));
                let literal = BufferSpecification::with_layout(BufferType::U8, vec![2u64, 3u64], column_major);
                let buffer = manager.retrieve_buffer(0).unwrap();
                assert_eq!(
                    manager.transfer_literal(0, Arc::new([1u8, 4u8, 2u8, 5u8, 3u8, 6u8]), literal).unwrap().r#await(),
                    Ok(()),
                );
                assert_eq!(buffer.copy_to_host(None).unwrap().r#await(), Ok(vec![1u8, 2u8, 3u8, 4u8, 5u8, 6u8]));

                // A literal transfer is the last transfer for its buffer.
                assert!(matches!(
                    manager.transfer_literal(0, Arc::new([0u8; 6]), specification.clone()),
                    Err(Error::FailedPrecondition { message, .. })
                        if message == "buffer 0 of this PJRT host-to-device transfer manager already received its \
                        last transfer or an error",
                ));
                assert!(matches!(
                    manager.transfer_data(0, Arc::new([0u8; 6]), 0, true),
                    Err(Error::FailedPrecondition { message, .. })
                        if message == "buffer 0 of this PJRT host-to-device transfer manager already received its \
                        last transfer or an error",
                ));
                assert!(matches!(
                    manager.set_error(0, Error::aborted("late error")),
                    Err(Error::FailedPrecondition { message, .. })
                        if message == "buffer 0 of this PJRT host-to-device transfer manager already received its \
                        last transfer or an error",
                ));
            }
        });
    }

    #[test]
    fn test_host_to_device_transfer_manager_transfer_literal_validation() {
        test_for_each_platform!(|_plugin, client, platform| {
            let specification = BufferSpecification::new(BufferType::F32, vec![2u64, 3u64]);
            if let Some(manager) = test_transfer_manager(&client, platform, vec![specification.clone()]) {
                let data = Arc::new([0u8; 24]);

                // The literal must have the element type and dimensions of the managed buffer.
                assert!(matches!(
                    manager.transfer_literal(0, data.clone(), BufferSpecification::new(BufferType::I32, [2u64, 3u64])),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "literal element type `i32` does not match the element type `f32` of buffer 0 \
                        of this PJRT host-to-device transfer manager",
                ));
                assert!(matches!(
                    manager.transfer_literal(0, data.clone(), BufferSpecification::new(BufferType::F32, [3u64, 2u64])),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "literal dimensions [3, 2] do not match the dimensions [2, 3] of buffer 0 of \
                        this PJRT host-to-device transfer manager",
                ));

                // The literal layout must be an untiled permutation of the literal dimensions.
                let tiled = TiledLayout::new(vec![1u64, 0u64], vec![Tile::new(vec![TileDimension::sized(2)])]);
                assert!(matches!(
                    manager.transfer_literal(
                        0,
                        data.clone(),
                        BufferSpecification::with_layout(BufferType::F32, [2u64, 3u64], Layout::Tiled(tiled)),
                    ),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "literal layout `{1,0:T(2)}` has tiles, which PJRT literal transfers do not \
                        support",
                ));
                let repeated = TiledLayout::new(vec![1u64, 1u64], Vec::new());
                assert!(matches!(
                    manager.transfer_literal(
                        0,
                        data.clone(),
                        BufferSpecification::with_layout(BufferType::F32, [2u64, 3u64], Layout::Tiled(repeated)),
                    ),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "literal layout `{1,1}` is not a minor-to-major ordering of the dimensions of \
                        a rank-2 literal",
                ));
                let truncated = TiledLayout::new(vec![0u64], Vec::new());
                assert!(matches!(
                    manager.transfer_literal(
                        0,
                        data.clone(),
                        BufferSpecification::with_layout(BufferType::F32, [2u64, 3u64], Layout::Tiled(truncated)),
                    ),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "literal layout `{0}` is not a minor-to-major ordering of the dimensions of a \
                        rank-2 literal",
                ));
                let strided = StridedLayout::new(vec![12i64, 4i64]);
                assert!(matches!(
                    manager.transfer_literal(
                        0,
                        data.clone(),
                        BufferSpecification::with_layout(BufferType::F32, [2u64, 3u64], Layout::Strided(strided)),
                    ),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "literal layout `strides(12,4)` is strided, which PJRT literal transfers do \
                        not support",
                ));

                // The host data must contain exactly the bytes of the literal.
                assert!(matches!(
                    manager.transfer_literal(0, Arc::new([0u8; 23]), specification.clone()),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "literal data has 23 bytes but a `f32` literal with dimensions [2, 3] has 24 \
                        bytes",
                ));
                assert!(matches!(
                    manager.transfer_literal(0, Arc::new([0u8; 25]), specification.clone()),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "literal data has 25 bytes but a `f32` literal with dimensions [2, 3] has 24 \
                        bytes",
                ));

                // Rejected transfers leave the buffer available for a valid transfer.
                let values = [1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
                let bytes = values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
                let buffer = manager.retrieve_buffer(0).unwrap();
                assert_eq!(
                    manager.transfer_literal(0, Arc::new(bytes.clone()), specification).unwrap().r#await(),
                    Ok(())
                );
                assert_eq!(buffer.copy_to_host(None).unwrap().r#await(), Ok(bytes));
            }
        });
    }

    #[test]
    fn test_host_to_device_transfer_manager_transfer_literal_after_transfer_data() {
        test_for_each_platform!(|_plugin, client, platform| {
            let specification = BufferSpecification::new(BufferType::U8, vec![4u64]);
            let specifications = vec![specification.clone(), specification.clone()];
            if let Some(manager) = test_transfer_manager(&client, platform, specifications) {
                // A literal cannot follow a raw data transfer that was not the last transfer for its buffer, but more
                // raw data transfers can.
                let buffer_0 = manager.retrieve_buffer(0).unwrap();
                let first_chunk = manager.transfer_data(0, Arc::new([1u8, 2u8]), 0, false).unwrap();
                assert_eq!(first_chunk.r#await(), Ok(()));
                assert!(matches!(
                    manager.transfer_literal(0, Arc::new([0u8; 4]), specification.clone()),
                    Err(Error::FailedPrecondition { message, .. })
                        if message == "buffer 0 of this PJRT host-to-device transfer manager already received raw \
                        data transfers and cannot also receive a literal transfer",
                ));
                assert_eq!(manager.transfer_data(0, Arc::new([3u8, 4u8]), 2, true).unwrap().r#await(), Ok(()));
                assert_eq!(buffer_0.copy_to_host(None).unwrap().r#await(), Ok(vec![1u8, 2u8, 3u8, 4u8]));

                // A literal cannot follow an error either.
                let buffer_1 = manager.retrieve_buffer(1).unwrap();
                assert_eq!(manager.set_error(1, Error::aborted("poisoned buffer")), Ok(()));
                assert!(matches!(
                    manager.transfer_literal(1, Arc::new([0u8; 4]), specification),
                    Err(Error::FailedPrecondition { message, .. })
                        if message == "buffer 1 of this PJRT host-to-device transfer manager already received its \
                        last transfer or an error",
                ));
                assert!(matches!(
                    buffer_1.ready().unwrap().r#await(),
                    Err(Error::Aborted { message, .. }) if message.contains("poisoned buffer"),
                ));
            }
        });
    }

    #[test]
    fn test_host_to_device_transfer_manager_transfer_literal_token() {
        let client = test_cpu_client();
        let devices = client.addressable_devices().unwrap();
        let memory = devices[0].default_memory().unwrap();
        let specification = BufferSpecification::new(BufferType::Token, Vec::new());
        let manager = client.host_to_device_transfer_manager(vec![specification.clone()], memory).unwrap();
        let buffer = manager.retrieve_buffer(0).unwrap();
        assert!(matches!(
            manager.transfer_literal(0, Arc::new([0u8; 1]), specification.clone()),
            Err(Error::InvalidArgument { message, .. })
                if message == "literal data has 1 bytes but a `token` literal has no bytes",
        ));
        assert_eq!(manager.transfer_literal(0, Arc::new([] as [u8; 0]), specification).unwrap().r#await(), Ok(()));
        assert_eq!(buffer.ready().unwrap().r#await(), Ok(()));
    }

    #[test]
    fn test_host_to_device_transfer_manager_set_error() {
        test_for_each_platform!(|_plugin, client, platform| {
            let specifications = vec![
                BufferSpecification::new(BufferType::F32, vec![4u64]),
                BufferSpecification::new(BufferType::I32, vec![2u64, 2u64]),
            ];
            if let Some(manager) = test_transfer_manager(&client, platform, specifications) {
                let buffer_0 = manager.retrieve_buffer(0).unwrap();
                let buffer_1 = manager.retrieve_buffer(1).unwrap();
                assert_eq!(manager.transfer_data(0, Arc::new([0u8; 16]), 0, true).unwrap().r#await(), Ok(()));
                assert_eq!(manager.set_error(1, Error::aborted("poisoned buffer")), Ok(()));

                // The poisoned buffer reports the error while the other buffer is unaffected.
                assert_eq!(buffer_0.ready().unwrap().r#await(), Ok(()));
                assert_eq!(buffer_0.copy_to_host(None).unwrap().r#await(), Ok(vec![0u8; 16]));
                assert!(matches!(
                    buffer_1.ready().unwrap().r#await(),
                    Err(Error::Aborted { message, .. }) if message.contains("poisoned buffer"),
                ));
                assert!(matches!(
                    buffer_1.copy_to_host(None).unwrap().r#await(),
                    Err(Error::Aborted { message, .. }) if message.contains("poisoned buffer"),
                ));

                // Errors and transfers after a buffer received its last transfer or an error are rejected.
                assert!(matches!(
                    manager.set_error(0, Error::aborted("late error")),
                    Err(Error::FailedPrecondition { message, .. })
                        if message == "buffer 0 of this PJRT host-to-device transfer manager already received its \
                        last transfer or an error",
                ));
                assert!(matches!(
                    manager.set_error(1, Error::aborted("repeated error")),
                    Err(Error::FailedPrecondition { message, .. })
                        if message == "buffer 1 of this PJRT host-to-device transfer manager already received its \
                        last transfer or an error",
                ));
                assert!(matches!(
                    manager.transfer_data(1, Arc::new([0u8; 16]), 0, true),
                    Err(Error::FailedPrecondition { message, .. })
                        if message == "buffer 1 of this PJRT host-to-device transfer manager already received its \
                        last transfer or an error",
                ));
                assert!(matches!(
                    manager.set_error(2, Error::aborted("out of range")),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "buffer index 2 is out of range for a PJRT host-to-device transfer manager \
                        with buffer count 2",
                ));
            }
        });
    }

    #[test]
    fn test_host_to_device_transfer_manager_retrieve_buffer() {
        test_for_each_platform!(|_plugin, client, platform| {
            let specifications = vec![BufferSpecification::new(BufferType::U8, vec![4u64])];
            if let Some(manager) = test_transfer_manager(&client, platform, specifications) {
                let buffer = manager.retrieve_buffer(0).unwrap();
                assert_eq!(buffer.element_type(), Ok(BufferType::U8));
                assert!(matches!(
                    manager.retrieve_buffer(0),
                    Err(Error::FailedPrecondition { message, .. })
                        if message == "buffer 0 has already been retrieved from this PJRT host-to-device transfer \
                        manager",
                ));
                assert!(matches!(
                    manager.retrieve_buffer(1),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "buffer index 1 is out of range for a PJRT host-to-device transfer manager \
                        with buffer count 1",
                ));

                // Retrieved buffers outlive their transfer manager.
                assert_eq!(
                    manager.transfer_data(0, Arc::new([9u8, 8u8, 7u8, 6u8]), 0, true).unwrap().r#await(),
                    Ok(()),
                );
                drop(manager);
                assert_eq!(buffer.copy_to_host(None).unwrap().r#await(), Ok(vec![9u8, 8u8, 7u8, 6u8]));
            }
        });
    }

    #[test]
    fn test_client_host_to_device_transfer_manager_validation() {
        let client = test_cpu_client();
        let devices = client.addressable_devices().unwrap();
        assert!(matches!(
            client.host_to_device_transfer_manager(
                vec![BufferSpecification::new(BufferType::F32, vec![u64::MAX])],
                devices[0].default_memory().unwrap(),
            ),
            Err(Error::InvalidArgument { message, .. })
                if message == "the size of a `f32` array with dimensions [18446744073709551615] overflows `i64`",
        ));
        assert!(matches!(
            client.host_to_device_transfer_manager(
                vec![BufferSpecification::new(BufferType::U8, vec![1u64 << 32, 1u64 << 31])],
                devices[0].default_memory().unwrap(),
            ),
            Err(Error::InvalidArgument { message, .. })
                if message == "the size of a `u8` array with dimensions [4294967296, 2147483648] overflows `i64`",
        ));
        assert!(matches!(
            client.host_to_device_transfer_manager(
                vec![BufferSpecification::new(BufferType::Invalid, vec![1u64])],
                devices[0].default_memory().unwrap(),
            ),
            Err(Error::InvalidArgument { message, .. }) if message == "invalid buffer type has no element size",
        ));
    }

    #[test]
    fn test_client_host_to_device_transfer_manager_with_default_layouts() {
        let client = test_cpu_client();
        let devices = client.addressable_devices().unwrap();
        let memory = devices[0].default_memory().unwrap();
        let manager = client
            .host_to_device_transfer_manager_with_default_layouts(
                vec![(BufferType::F32, vec![2u64, 3u64]), (BufferType::Token, Vec::new())],
                memory,
            )
            .unwrap();
        assert_eq!(manager.buffer_count(), Ok(2));
        assert_eq!(manager.buffer_on_device_size_in_bytes(0), Ok(24));

        // The buffer uses the default layout that the CPU plugin reports for its shape.
        let default_layout = client.default_layout(BufferType::F32, [2u64, 3u64]).unwrap().proto().unwrap();
        let default_layout = Layout::from_proto(default_layout).unwrap();
        assert_eq!(default_layout, Layout::dense_major_to_minor(2));
        let buffer = manager.retrieve_buffer(0).unwrap();
        assert_eq!(
            buffer.specification(),
            Ok(BufferSpecification::with_layout(BufferType::F32, [2u64, 3u64].as_slice(), default_layout)),
        );
        let token = manager.retrieve_buffer(1).unwrap();
        assert_eq!(token.element_type(), Ok(BufferType::Token));

        // Literal transfers are converted to the default layout.
        let values = [1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
        let bytes = values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
        let specification = BufferSpecification::new(BufferType::F32, vec![2u64, 3u64]);
        assert_eq!(manager.transfer_literal(0, Arc::new(bytes.clone()), specification).unwrap().r#await(), Ok(()));
        assert_eq!(buffer.copy_to_host(None).unwrap().r#await(), Ok(bytes));
        assert_eq!(manager.transfer_data(1, Arc::new([] as [u8; 0]), 0, true).unwrap().r#await(), Ok(()));
        assert_eq!(token.ready().unwrap().r#await(), Ok(()));
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

        // Create a `Chunk` with a nested scope so that we can verify that its `deleter` was called on `drop`.
        {
            let chunk = unsafe { Chunk::from_c_api(&mut ffi_chunk as *mut _) }.unwrap();
            assert_eq!(chunk.data(), bytes.as_slice());
            assert_eq!(deleted_counter.load(Ordering::SeqCst), 0);
        }

        assert_eq!(deleted_counter.load(Ordering::SeqCst), 1);

        // Test with an empty `Chunk`.
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

        // Test creating a `Chunk` from a null pointer.
        assert!(matches!(
            unsafe { Chunk::from_c_api(std::ptr::null_mut()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT chunk handle is a null pointer",
        ));
    }
}
