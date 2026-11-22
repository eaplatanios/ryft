//! Transfers between host memory and the device buffers of an [`Array`].
//!
//! Host values enter through two constructors. [`Array::from_host_buffer`] takes the dense row-major bytes of the
//! whole logical value, derives the shard of every device from the type, mesh, and sharding, and uploads only the
//! shards of the devices of this process, so every process of a distributed array passes the same value.
//! [`Array::from_host_shards`] instead takes the bytes of each local shard, which may differ between processes and
//! between the devices along unreduced axes. [`Session::place`] also accepts host values and uploads them through
//! [`Array::from_host_buffer`]. Along unreduced axes of kind [`UnreducedKind::Sum`], a full host value is split into
//! the value itself at coordinate zero and additive identities elsewhere, so the summands add up to the value. Each
//! process uploads only the shards of the devices that the mesh assigns to it, so both constructors reject a device of
//! this process that the mesh assigns to another process with [`ArrayError::DeviceProcessIndexMismatch`] and a device
//! that the mesh assigns to this process but that its PJRT client does not address with
//! [`ArrayError::NonAddressableDevice`], because no process would upload its shard, as imports of buffers and
//! placements onto a mesh do. A device that the PJRT client does not list keeps the process that the mesh assigns to
//! it, because it can belong to a process whose devices the client cannot see. Both constructors also reject a layout
//! that does not address the shape of every shard and host bytes that do not encode valid elements of the data type
//! (e.g., a [`DataType::Boolean`] byte other than `0` or `1`) before they upload anything, because a typed readback
//! of the array would fail otherwise. They validate the bytes of every upload, including the storage of a
//! [`ryft_core::Array`], which its safe mutable view can make invalid.
//!
//! Values leave through [`Array::copy_to_host`] and [`Array::copy_to_host_bytes`], which assemble the dense row-major
//! value from one local replica of each distinct contribution (i.e., of each tile and, along unreduced axes, of each
//! per-device value of a tile), combine the contributions along unreduced axes, and skip tiles without elements. They
//! fail with [`ArrayError::MissingShardBuffer`] before issuing any copy when this process does not hold the whole
//! value, and they reject an array whose devices can hold different values of one tile along a varying manual mesh
//! axis, which one host value cannot represent. [`ArrayShard::copy_to_host_bytes`] reads the value of one device.
//!
//! # Completion
//!
//! Uploads return as soon as PJRT has accepted them, so the returned arrays can hold pending buffers. Readbacks are
//! synchronization boundaries, and the observations of an array report the following failures:
//!
//!   - [`Array::ready`] and [`Array::block_until_ready`] report the failures of every addressable buffer, of the
//!     execution fence of the array (i.e., of the work that produced it), and of the cross-host sends of the placement
//!     that produced it, if any.
//!   - [`Array::copy_to_host`] and [`Array::copy_to_host_bytes`] block until their copies and the execution fence
//!     complete and report the errors of both. They never wait for the replicas or empty tiles that they do not read,
//!     nor for cross-host sends. An array without elements copies nothing and waits for its execution fence, or for
//!     every addressable buffer when it has none.
//!   - [`Session::synchronize`] reports the failures of every upload, copy, transfer, and execution that the session
//!     registered, including the uploads of replicas that no readback reads.
//!
//! [`Array::from_host_buffer`] and [`Array::from_host_shards`] give the array no execution fence, so a readback does
//! not report the failed upload of a replica that it skips. Every replica of an upload receives the same host bytes,
//! so such a failure never changes the value that the readback returns, and the other observations report it. The
//! uploads that a placement makes join their readiness into the execution fence of its result instead (e.g., the
//! uploads of its host fallback, in the background or on the calling thread), because the replicas of a placement
//! can come from different sources, so its readbacks also report a failed replica that they skip.
//!
//! Placements that read an array back and upload it again (e.g., the host fallback and the host relayouts of
//! [`Array::to_placement`]) hand out their destination buffers first and finish the work on the background upload
//! workers of the session, which a failure of that work poisons. Such a placement blocks on the readback only when the
//! PJRT plugin lacks host to device transfer managers or their literal transfers. A process that holds no destination
//! buffer of such a placement has no upload to report the work that produces the value, so the execution fence of its
//! result is the fence of that work instead.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::num::NonZeroUsize;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use ryft_core::{
    Add, Array as CoreArray, ArrayAddressing, ArrayType, DataType, DeviceMesh, Max, Min, Or, ProgramError, Sharding,
    Typed, UnreducedKind,
};
use ryft_pjrt::{
    Buffer, BufferSpecification, BufferType, Client, Device as PjrtDevice, DeviceId, Error as PjrtError,
    ExecutionFence, HostBuffer, HostDestination, HostToDeviceTransferManager, Memory as PjrtMemory,
};

use crate::arrays::metadata::resolve_placement;
use crate::arrays::shards::{ArrayShard, ShardDescriptor, ShardLayout};
use crate::arrays::{Array, ArrayError};
use crate::devices::PjrtDeviceExtension;
use crate::types::{ArrayTypeExtension, DataTypeExtension};
use crate::{Error, Session, ToPjrt};

impl Array {
    /// Creates an [`Array`] by transferring a dense row-major host buffer to the local shards implied by
    /// `r#type` and `mesh`, while avoiding redundant allocations external to the PJRT backend of the session's
    /// [`Client`]. This function derives the per-device shard slices from the provided type and mesh, transfers only
    /// the shards addressable by that client, and returns an [`Array`] that belongs to `session` and whose global shard
    /// metadata covers the full mesh.
    ///
    /// With [`None`] for `sharding`, the function uses the sharding of `r#type` if present and replicates the array
    /// across the selected mesh otherwise. A supplied sharding can place the array over auto mesh axes. After removing
    /// its auto axes with [`Sharding::without_auto_axes`], it must match any sharding in the type, or the function
    /// returns [`ArrayError::PlacementTypeMismatch`]. The type of the resulting array records that projection, and
    /// [`Self::sharding`] returns the full runtime placement.
    ///
    /// Passing [`None`] for `mesh` uses the mesh configured on `session` if present. Otherwise, the logical mesh
    /// from the runtime or type sharding is laid out over the client's addressable devices in enumeration order.
    /// Without sharding, the first addressable device forms the canonical mesh of one device, whose only axis is named
    /// `device` (refer to [`LogicalMesh::single_device`](ryft_core::LogicalMesh::single_device)). The result is
    /// [committed](Self::is_committed) to its devices when `mesh` or the mesh of `session` selected them, or when it
    /// lives on more than one device, like every eager result. It is uncommitted only when its devices are this default
    /// and it lives on a single device, so an uncommitted array always lives on a single device. Whether committed or
    /// not, devices that neither `mesh` nor the mesh of `session` selected are ones that this process chose on its own,
    /// which a later placement onto devices of several processes takes into account (refer to
    /// [`crate::arrays::resharding`]).
    ///
    /// When the destination mesh spans several processes, each process uploads only the shards of its own devices,
    /// so every process must call this function with the same host value. Nothing compares the values across processes,
    /// so differing values leave the shards and the replicas of the array inconsistent. Use [`Self::from_host_shards`]
    /// for values that are meant to differ between processes.
    ///
    /// Each addressable shard is uploaded to the memory requested by `r#type`.
    /// [`Memory::Device`](ryft_core::Memory::Device) uses the device's default memory. Host placements select
    /// its `pinned_host` or `unpinned_host` memory. A device without the requested host space produces
    /// [`ArrayError::UnsupportedMemory`]. Bufferless zero-space arrays require no allocation.
    ///
    /// Along an unreduced axis of a sharding of kind [`UnreducedKind::Sum`], the devices hold summands of the logical
    /// value, so the devices at coordinate zero along every unreduced axis receive their shards of the host value and
    /// the others receive the additive identity of its data type. That identity is negative zero for the floating
    /// point and complex types that have signed zeros, because adding positive zero would turn a negative zero into a
    /// positive one, and plain zero for the integer types and for the floating point types with a single zero. The
    /// summands of [`DataType::Boolean`] values combine with logical or (refer to [`Self::copy_to_host_bytes`]), so
    /// their identity is `false`. A data type without an additive identity (i.e., [`DataType::F8E8M0FNU`], whose every
    /// encoding is a nonzero number) cannot be split into summands, so it is rejected before anything is uploaded
    /// whenever a device away from coordinate zero would receive a non-empty shard. It is accepted when every
    /// unreduced axis has size one or when those shards are empty (e.g., for an array without elements), because no
    /// identity is written then. With kind [`UnreducedKind::Max`] or [`UnreducedKind::Min`], every device receives
    /// its shard of the host value, which is its own extremum for every data type.
    ///
    /// The bytes must encode valid elements of the data type, as for a [`ryft_core::Array`], whose typed readback
    /// would otherwise fail. Only the data types whose valid encodings are a strict subset of their storage bytes can
    /// be invalid (i.e., [`DataType::Boolean`], whose bytes must be `0` or `1`, and the data types narrower than a
    /// byte, whose bits above their width must be zero), so only their bytes are validated, and every process validates
    /// the whole host value before it uploads anything.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] the constructed array belongs to. Its client transfers the local addressable shard
    ///     buffers.
    ///   - `r#type`: Global [`ArrayType`] of the constructed array.
    ///   - `mesh`: Destination [`DeviceMesh`], or [`None`] to use eager placement.
    ///   - `sharding`: Optional runtime [`Sharding`] over the destination mesh.
    ///   - `buffer`: Dense row-major host bytes for the full logical array.
    ///
    /// # Errors
    ///
    /// Returns an error if placement cannot be resolved, the layout of `r#type` does not address its shape, the bytes
    /// do not match the type, or a device upload fails, an [`Error::Core`] that names the data type and the first
    /// element whose bytes do not encode a value of that data type, [`ArrayError::MissingUnreducedSumIdentity`] if the
    /// sharding sums along unreduced axes a data type without an additive identity and some device would have to hold
    /// that identity, and [`ArrayError::DeviceProcessIndexMismatch`] if the PJRT client reports another process for a
    /// device than the one that the mesh assigns to it.
    pub fn from_host_buffer<M: Into<Option<DeviceMesh>>, B: AsRef<[u8]>>(
        session: &Session,
        r#type: ArrayType,
        mesh: M,
        sharding: Option<Sharding>,
        buffer: B,
    ) -> Result<Self, Error> {
        let (mesh, buffer) = (mesh.into(), buffer.as_ref());

        // Without a mesh of the caller or of `session`, this process chooses the devices on its own.
        let committed = mesh.is_some() || session.commits_placement(&[], &[]);
        let uploads = HostBufferUploads::new(session, r#type, mesh, sharding, Some(buffer.len()))?;
        let HostBufferUploads { r#type, sharding, mesh, device_layout, shards, .. } = uploads;
        validate_host_element_encodings(r#type.data_type(), buffer)?;
        if r#type.data_type().is_zero() {
            let array = Self::zero(session, r#type, Some(sharding), mesh)?;
            return Ok(array.commit_if(committed).with_locally_assigned(!committed));
        }

        // Transfer only the shards owned by this process. Remote shards remain represented by metadata when
        // `Self::from_canonical_addressable_buffers` materializes the final array.
        let client = session.client();
        let buffer_type = r#type.data_type().executable_buffer_type();
        let mut work = session.reserve_work();
        let mut addressable_buffers = Vec::with_capacity(shards.len());
        let mut identities = Vec::<(usize, Arc<[u8]>)>::new();
        for shard in shards {
            // Resolve the requested space explicitly, because passing the device itself would always upload to its
            // default memory and could silently mislabel a host-resident array as pinned or unpinned memory.
            let memory = shard.device.memory(r#type.memory())?;
            let dimensions = shard.dimensions.as_slice();
            let uploaded = match shard.source {
                HostShardSource::AdditiveIdentity { element, count } => {
                    // The shards that hold an identity of the same size share one host allocation, which PJRT retains
                    // until it has read it, instead of each holding a copy of their own.
                    let identity = match identities.iter().find(|(identity_count, _)| *identity_count == count) {
                        Some((_, identity)) => Arc::clone(identity),
                        None => {
                            let identity = Arc::<[u8]>::from(element.repeat(count));
                            identities.push((count, Arc::clone(&identity)));
                            identity
                        }
                    };
                    client.buffer(identity, buffer_type, dimensions, None, memory, device_layout.clone())?
                }
                HostShardSource::Window { window, byte_strides } => client.buffer(
                    &buffer[window],
                    buffer_type,
                    dimensions,
                    byte_strides.as_deref(),
                    memory,
                    device_layout.clone(),
                )?,
            };
            work.track_buffer(&uploaded);
            addressable_buffers.push(uploaded);
        }

        // Reuse the buffer constructor for final buffer validation and global sharding metadata assembly.
        let array =
            Self::from_canonical_addressable_buffers(session, r#type, Some(sharding), mesh, addressable_buffers)?;
        Ok(array.commit_if(committed).with_locally_assigned(!committed))
    }

    /// Creates an [`Array`] like [`Self::from_host_buffer`] does, but from host bytes that `readback` produces later.
    /// The destination buffers come from PJRT host to device transfer managers, which hand them out before their
    /// contents exist, so this function returns without waiting for `readback`. Once `sources` is ready, one job on the
    /// [`HostUploadWorkers`] of `session` runs `readback` and then uploads the bytes of each addressable shard. The job
    /// runs even when `sources` reports an error, which `readback` then returns. The shards receive the same slices,
    /// additive identities along unreduced axes, memory, and layout as with [`Self::from_host_buffer`], and the same
    /// data types are rejected before anything is uploaded. The shards that are contiguous in the host bytes are
    /// uploaded from the bytes that `readback` returned without copying them, and the shards that hold the same bytes
    /// otherwise (e.g., the replicas of a strided tile, or the additive identities along an unreduced axis) are
    /// uploaded from one shared copy (refer to [`PreparedHostShards`]).
    ///
    /// The destination buffers are registered with the work tracking of `session` before this function returns, so
    /// [`Session::synchronize`] waits for the uploads. When `readback` fails, returns bytes of the wrong size or bytes
    /// that do not encode valid elements of the data type (refer to [`Self::from_host_buffer`]), or an upload cannot be
    /// issued, the background job sets the error that `poison` derives from that failure on every destination buffer
    /// that has not received its bytes. The execution fence of the returned array joins the readiness of every
    /// destination buffer, so its readiness functions and readbacks all report that error, even a readback that reads
    /// only one replica of each tile or that reads no bytes because the array has no elements. When no device of `mesh`
    /// belongs to this process, there is nothing to upload, so this function submits no job and never calls `readback`.
    /// No destination buffer could report the work that produces the value either, so the execution fence of the
    /// returned array is `sources` instead, and the array still waits for and reports that work, as the uploads
    /// of the processes that hold its shards do.
    ///
    /// Returns [`None`] without calling `readback` when the PJRT plugin lacks host to device transfer managers or their
    /// literal transfers (refer to [`PendingHostUpload::new`]), or when `r#type` has [`DataType::Zero`] (i.e., when
    /// there are no bytes to upload). The caller can then fall back to [`Self::from_host_buffer`].
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] the constructed array belongs to.
    ///   - `r#type`: Global [`ArrayType`] of the constructed array.
    ///   - `mesh`: Destination [`DeviceMesh`].
    ///   - `sharding`: Runtime [`Sharding`] over `mesh`.
    ///   - `sources`: [`ExecutionFence`] of everything that `readback` waits for (e.g., the fence that
    ///     [`Self::host_readback_fence`] returns for the array that `readback` reads back), so that the job does not
    ///     occupy a worker while those dependencies are pending (refer to [`HostUploadWorkers`]).
    ///   - `readback`: Function that returns the dense row-major host bytes of the full logical array. It runs in the
    ///     background job at most once, and it may block only on the work that `sources` covers.
    ///   - `poison`: Function that converts a failure of the background work into the error that the destination
    ///     buffers report. It runs in the background job at most once.
    pub(super) fn from_host_buffer_in_background<R, P>(
        session: &Session,
        r#type: ArrayType,
        mesh: DeviceMesh,
        sharding: Sharding,
        sources: ExecutionFence,
        readback: R,
        poison: P,
    ) -> Result<Option<Self>, Error>
    where
        R: 'static + Send + FnOnce() -> Result<Vec<u8>, Error>,
        P: 'static + Send + FnOnce(Error) -> PjrtError,
    {
        let uploads = HostBufferUploads::new(session, r#type, Some(mesh), Some(sharding), None)?;
        if uploads.r#type.data_type().is_zero() {
            return Ok(None);
        }

        // Register the submission before any destination buffer exists or the background job is submitted, so that a
        // concurrent `Session::synchronize` takes a snapshot that includes this upload whenever the upload may already
        // be running (and may already have failed) by the time that snapshot waits. If this function returns before
        // attaching the destination buffers (e.g., to fall back or because the job could not be submitted), dropping
        // the reservation completes it right away without work, so it never blocks a synchronization.
        let mut work = session.reserve_work();

        // Hand out every destination buffer before starting the background work, so that a plugin without transfer
        // managers or literal transfers falls back before anything has been read back.
        let client = session.client();
        let buffer_type = uploads.r#type.data_type().executable_buffer_type();
        let element_size_in_bytes = buffer_type.element_size_in_bytes()?;
        let mut pending_uploads = Vec::with_capacity(uploads.shards.len());
        let mut buffers = Vec::with_capacity(uploads.shards.len());
        let mut shards = Vec::with_capacity(uploads.shards.len());
        for shard in uploads.shards {
            let memory = shard.device.memory(uploads.r#type.memory())?;
            let destination = PendingHostUpload::new(
                client,
                buffer_type,
                shard.dimensions.clone(),
                memory,
                uploads.device_layout.clone(),
            )?;
            let Some((upload, buffer)) = destination else {
                return Ok(None);
            };
            pending_uploads.push(upload);
            buffers.push(buffer);
            shards.push((shard.dimensions, shard.source));
        }
        let (r#type, sharding, mesh, byte_count) = (uploads.r#type, uploads.sharding, uploads.mesh, uploads.byte_count);

        // A process without destination buffers has nothing to upload, so it submits no job, which would only retain
        // `readback` (and whatever it reads) until `sources` completes. No destination buffer could report the work
        // that produces the value either, so the result waits for and reports `sources` instead.
        if buffers.is_empty() {
            let output = Self::from_canonical_addressable_buffers(session, r#type, Some(sharding), mesh, buffers)?;
            return Ok(Some(output.with_execution_fence(sources)));
        }

        // The background job reads the host value back once, before it prepares the bytes of the first shard, so that
        // a failed readback or a readback with invalid element encodings poisons every destination buffer. The readback
        // moves into a shared allocation without a copy, and the shards that are windows of it transfer straight from
        // that allocation.
        let data_type = r#type.data_type();
        let mut readback = Some(readback);
        let mut host_bytes = Arc::new(Vec::new());
        let mut shards = PreparedHostShards::new(shards, element_size_in_bytes);
        let prepare = move |index: usize| {
            if let Some(readback) = readback.take() {
                let bytes = readback()?;
                if bytes.len() != byte_count {
                    return Err(Error::from(ArrayError::HostBufferByteCountMismatch {
                        expected: byte_count,
                        actual: bytes.len(),
                    }));
                }
                validate_host_element_encodings(data_type, &bytes)?;
                host_bytes = Arc::new(bytes);
            }
            Ok(shards.bytes(&host_bytes, index))
        };

        // Hand the job that fills the destination buffers to the workers of the session before anything observes those
        // buffers. Until it runs, nothing can complete them, and observing them can wait for them (e.g., building a
        // fence from their events awaits an event whose callback PJRT fails to register), which would then never
        // return. If the job cannot be handed over, dropping its transfer managers has already poisoned the buffers
        // that are about to be dropped here, and the reservation completes without work.
        session
            .host_upload_workers()
            .submit(sources, move || run_host_uploads(pending_uploads, prepare, poison))?;

        // Attach the destination buffers to the reservation only once the job that completes them has been handed over,
        // and only then join their readiness into the fence of the result, so that every readback of the result reports
        // a failed upload, even a readback that reads only one replica of each tile or no bytes at all. A failure to
        // build the result leaves the buffers tracked by the session until the job has filled or poisoned them.
        for buffer in &buffers {
            work.track_buffer(buffer);
        }

        let ready_events = buffers.iter().map(Buffer::ready).collect::<Result<Vec<_>, _>>()?;
        let output = Self::from_canonical_addressable_buffers(session, r#type, Some(sharding), mesh, buffers)?;
        Ok(Some(output.with_execution_fence(ExecutionFence::new(ready_events))))
    }

    /// Creates an [`Array`] by uploading the dense row-major bytes for each shard owned by this process. The type
    /// describes the global array. Every process supplies bytes only for its own devices and uses the same global
    /// type, mesh, and placement. Each replica needs its own entry, and along unreduced axes the bytes represent
    /// the partial value of that device. The replicas of one contribution (i.e., the devices that hold the same slice
    /// and, along unreduced axes, the same coordinates) must receive equal bytes, on this process and across processes,
    /// as for [`Self::from_addressable_buffers`]. Nothing compares them, and a readback reads only one of them, so
    /// differing bytes make the value that the array reads back depend on which replica a readback or a later
    /// placement selects. Shards of [`DataType::Zero`] take empty buffers and require no upload. The bytes of every
    /// shard must encode valid elements of the data type, as for [`Self::from_host_buffer`], and are validated before
    /// anything is uploaded.
    ///
    /// With [`None`] for `sharding`, the function uses the sharding of `r#type` if present and replicates the
    /// array across `mesh` otherwise. A supplied sharding can place the array over auto mesh axes. After
    /// removing its auto axes with [`Sharding::without_auto_axes`], it must match any sharding in the type.
    ///
    /// The result may contain pending device buffers. Use [`Self::block_until_ready`] to wait synchronously or
    /// [`Self::ready`] to wait asynchronously for the local uploads. A borrowed byte slice can be reused once
    /// this function returns. Pass an [`Arc<[u8]>`] to let PJRT retain the host allocation until it finishes
    /// reading it. Processes can enqueue their uploads independently.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] whose PJRT client uploads the host shards.
    ///   - `r#type`: Global static [`ArrayType`] of the new array.
    ///   - `mesh`: Global [`DeviceMesh`] over which the array is placed.
    ///   - `sharding`: Optional runtime [`Sharding`] over `mesh`.
    ///   - `shards`: One pair of device ID and host bytes per local device in `mesh`, in any order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Array`] if the type has a dynamic shape, the type and runtime sharding disagree, the PJRT
    /// client reports another process for a device than the one that the mesh assigns to it, or the host shards have
    /// missing or duplicate entries, unexpected devices, or incorrect byte counts. Returns an [`Error::Core`] if the
    /// layout of `r#type` does not address its shape, or one that names the data type and the first element within a
    /// shard whose bytes do not encode a value of that data type. Other placement and PJRT errors propagate from the
    /// layout and upload.
    pub fn from_host_shards<B: AsRef<[u8]> + HostBuffer, S: IntoIterator<Item = (DeviceId, B)>>(
        session: &Session,
        r#type: ArrayType,
        mesh: DeviceMesh,
        sharding: Option<Sharding>,
        shards: S,
    ) -> Result<Self, Error> {
        let (r#type, sharding) = resolve_placement(r#type, sharding, &mesh)?;
        let shape = r#type
            .static_shape()
            .ok_or_else(|| Error::from(ArrayError::DynamicShape { shape: r#type.shape().clone() }))?;
        let layout = ShardLayout::new(&shape, &mesh, &sharding)?;
        let client = session.client();
        validate_mesh_device_processes(client, &mesh)?;
        let process_index = client.process_index()?;
        let mut devices = HashMap::new();
        for device in client.addressable_devices()? {
            devices.insert(device.id()?, device);
        }

        let mut host_shards = HashMap::new();
        for (device_id, bytes) in shards {
            let index =
                layout.shard_index_by_device().get(&device_id).ok_or(ArrayError::DeviceNotInMesh { device_id })?;
            let descriptor = &layout.descriptors()[*index];
            if descriptor.device().process_index() != process_index || !devices.contains_key(&device_id) {
                return Err(ArrayError::NonAddressableDevice { device_id, process_index }.into());
            }
            if host_shards.insert(device_id, bytes).is_some() {
                return Err(ArrayError::MultipleHostShardsOnDevice { device_id }.into());
            }
        }

        let data_type = r#type.data_type();
        let mut uploads = Vec::new();
        for descriptor in layout.descriptors() {
            if descriptor.device().process_index() != process_index {
                continue;
            }

            let device_id = descriptor.device().id();
            let device =
                devices.get(&device_id).ok_or(ArrayError::NonAddressableDevice { device_id, process_index })?;
            let bytes = host_shards.remove(&device_id).ok_or(ArrayError::MissingHostShard { device_id })?;
            let shape = descriptor.shape();

            let expected = ArrayType::new(data_type, shape.clone().into()).dense_size_in_bytes()?;
            let actual = bytes.as_ref().len();
            if actual != expected {
                return Err(ArrayError::HostShardByteCountMismatch { device_id, expected, actual }.into());
            }
            validate_host_element_encodings(data_type, bytes.as_ref())?;

            if !data_type.is_zero() {
                let memory = device.memory(r#type.memory())?;
                uploads.push((bytes, shape, memory));
            }
        }

        if data_type.is_zero() {
            return Self::zero(session, r#type, Some(sharding), mesh);
        }

        let buffer_type = data_type.executable_buffer_type();
        let device_layout = r#type.layout().map(ToPjrt::to_pjrt).transpose()?;
        let mut work = session.reserve_work();
        let mut buffers = Vec::with_capacity(uploads.len());
        for (bytes, shape, memory) in uploads {
            let dimensions = shape.as_slice().iter().map(|&extent| extent as u64).collect::<Vec<_>>();
            let buffer = client.buffer(bytes, buffer_type, dimensions, None, memory, device_layout.clone())?;
            work.track_buffer(&buffer);
            buffers.push(buffer);
        }

        Self::from_canonical_addressable_buffers(session, r#type, Some(sharding), mesh, buffers)
    }

    /// Copies the value of this [`Array`] to the host and returns it as a host array of the reference backend
    /// (i.e., a [`ryft_core::Array`]) with the data type and shape of this array. The type of the result has no
    /// sharding and no physical layout, because the host value is one dense row-major array. Read its elements with
    /// [`ryft_core::Array::elements`], which checks the requested element type against the data type of the array.
    ///
    /// This function reads the value as [`Self::copy_to_host_bytes`] does, with the same synchronization, shard
    /// selection, and combination of unreduced values, and it then validates the bytes as the storage of the host
    /// array, which visits every element once. Use [`Self::copy_to_host_bytes`] to skip that validation.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::copy_to_host_bytes`], and an [`Error::Core`] if the bytes that the devices
    /// returned are not a valid encoding of the data type of this array.
    #[inline]
    pub fn copy_to_host(&self) -> Result<CoreArray, Error> {
        let bytes = self.copy_to_host_bytes()?;
        Ok(CoreArray::new(ArrayType::new(self.data_type(), self.shape().into()), bytes)?)
    }

    /// Copies the value of this [`Array`] to the host and returns its dense row-major bytes, irrespective of the
    /// physical [`Layout`](ryft_core::Layout) of its shard buffers. This is an explicit synchronization boundary.
    /// It blocks the current thread until the copies and the whole execution that produced this array complete, so
    /// it also surfaces the errors of that execution.
    ///
    /// Only one replica of each distinct contribution to the value is copied (i.e., the first local replica of each
    /// tile and, along unreduced axes, of each per-device value of a tile). The other replicas are neither read nor
    /// compared with it, and the readback does not wait for them. When each tile occupies a contiguous byte range of
    /// the dense array, each copy goes directly into the destination. Otherwise, the copies are assembled on the host.
    /// When the [`Sharding`] has unreduced axes, the separate per-device values (e.g., summands) are combined with the
    /// [`UnreducedKind`] of the sharding. The values of each tile are combined in ascending order of their coordinates
    /// along the unreduced axes, irrespective of which replicas this process holds, so every process that holds the
    /// whole value reads back the same bytes. Sums of [`DataType::Boolean`] values combine with logical or. Tiles
    /// without elements contribute nothing, so they need no replica in this process. Arrays with no elements and
    /// [`DataType::Zero`] arrays produce no bytes and issue no copies. Use [`ArrayShard::copy_to_host_bytes`] to
    /// read the bytes that a single device holds.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::MissingShardBuffer`] when the current process does not hold the whole value (i.e., when
    /// some tile, or along unreduced axes some per-device value of a tile, has no replica in this process), and
    /// [`ProgramError::InvalidArgument`] when two devices that differ along a varying manual mesh axis hold one tile
    /// (refer to [`Sharding::varying_manual_axes`]), because they can hold different values that one host value cannot
    /// represent. Both are checked before issuing any copies. PJRT transfer failures and the errors of the producing
    /// execution are returned unchanged.
    pub fn copy_to_host_bytes(&self) -> Result<Vec<u8>, Error> {
        let completion = self.execution_fence();
        let global_shape = self.shape();
        let element_type = self.data_type();

        // A PJRT size query for an empty dynamic buffer can expose its bound allocation and trailer. Skip transfers
        // for arrays with no payload, and wait for the producer or local buffers so their errors still surface.
        if element_type.is_zero() || global_shape.as_slice().contains(&0) {
            if let Some(completion) = completion {
                completion.block_until_ready()?;
            } else {
                let events = self
                    .addressable_shards()
                    .filter_map(|shard| shard.buffer())
                    .map(|buffer| buffer.ready())
                    .collect::<Result<Vec<_>, _>>()?;
                ExecutionFence::new(events).block_until_ready()?;
            }
            return Ok(Vec::new());
        }

        // Look up the element size once for the array and all its shards. The checked global byte count bounds the
        // byte offsets of tiles that copy directly into the destination.
        let element_size_in_bytes = element_type.size_in_bytes()?;
        let global_element_count = global_shape.element_count()?;
        let total_byte_count =
            global_element_count.checked_mul(element_size_in_bytes).ok_or_else(|| Error::SizeLimitExceeded {
                message: format!(
                    "dense byte size for array type `{}` exceeds the maximum allowed size of {}",
                    self.r#type(),
                    usize::MAX,
                ),
            })?;
        let unreduced_axis_indices = self.unreduced_axis_indices();

        // Select the shards to copy before issuing any copy, so that no copy is in flight when the selection fails.
        // The checked global byte count bounds the global strides.
        let (shards, contiguous_tiles) = self.host_readback_shards(unreduced_axis_indices.as_slice())?;
        let global_strides = row_major_element_strides(global_shape.as_slice(), element_type)?;
        let mut work = self.session().reserve_work();

        // When the selected shards are one replica of each tile and every tile occupies one contiguous byte range of
        // the dense representation, each is copied directly into that range of one destination.
        if contiguous_tiles {
            // The checked global byte count bounds the size and the offset of every tile.
            let tiles = shards
                .iter()
                .map(|shard| {
                    let slices = shard.slice();
                    let byte_count = slices.iter().map(Range::len).product::<usize>() * element_size_in_bytes;
                    let start =
                        slices.iter().zip(&global_strides).map(|(slice, stride)| slice.start * stride).sum::<usize>()
                            * element_size_in_bytes;
                    (*shard, start..start + byte_count)
                })
                .collect::<Vec<_>>();
            let mut destination = HostDestination::new(total_byte_count);
            let ranges = tiles.iter().map(|(_, range)| range.clone()).collect::<Vec<_>>();
            let slices = destination.split(ranges.as_slice())?;
            let copies = tiles
                .iter()
                .zip(slices)
                .map(|((shard, _), slice)| {
                    let layout = ryft_pjrt::Layout::dense_major_to_minor(shard.shape().rank());
                    let event = shard.buffer().unwrap().copy_to_host_buffer(Some(layout), slice)?;
                    Ok(work.track_host_copy(event))
                })
                .collect::<Result<Vec<_>, Error>>()?;

            // Publish the readback to the session after issuing all copies, then check the producer before consuming
            // the transfer results so any execution error takes precedence.
            drop(work);

            if let Some(completion) = completion {
                completion.block_until_ready()?;
            }

            // Drop each returned slice so the destination can yield its full byte buffer.
            for copy in copies {
                drop(copy.wait()?);
            }

            return Ok(destination.into_bytes().ok().unwrap());
        }

        // Non-contiguous tiles and per-device values along unreduced axes need separate host buffers. Request
        // row-major bytes for the merge regardless of the source layouts, and issue all copies before waiting.
        let copies = shards
            .into_iter()
            .map(|shard| {
                let buffer = shard.buffer().unwrap();
                let shard_shape = shard.shape();
                let event = buffer.copy_to_host(Some(ryft_pjrt::Layout::dense_major_to_minor(shard_shape.rank())))?;
                let copy = work.track_host_copy(event);
                let element_count = shard_shape.element_count()?;
                let expected_byte_count =
                    element_count.checked_mul(element_size_in_bytes).ok_or_else(|| Error::SizeLimitExceeded {
                        message: format!(
                            "dense byte size for array type `{}{}` exceeds the maximum allowed size of {}",
                            element_type,
                            shard_shape,
                            usize::MAX,
                        ),
                    })?;
                Ok((shard, copy, expected_byte_count))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        // Publish the readback before waiting, and check the producer before consuming the transfer results.
        drop(work);

        if let Some(completion) = completion {
            completion.block_until_ready()?;
        }

        // Wait for the copies, check their byte counts, and record which coordinates along unreduced axes each holds.
        let mut contributions = copies
            .into_iter()
            .map(|(shard, copy, expected_byte_count)| {
                let bytes = copy.wait()?;
                if bytes.len() != expected_byte_count {
                    return Err(Error::from(ArrayError::ShardByteCountMismatch {
                        shard_index: shard.index(),
                        device_id: shard.device().id(),
                        expected: expected_byte_count,
                        actual: bytes.len(),
                    }));
                }
                let unreduced_coordinates = self.unreduced_coordinates(shard, unreduced_axis_indices.as_slice());
                Ok((shard.slice().to_vec(), unreduced_coordinates, bytes))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let mut tiles = if contributions.iter().all(|(_, unreduced_coordinates, _)| unreduced_coordinates.is_empty()) {
            contributions.into_iter().map(|(slices, _, bytes)| (slices, bytes)).collect::<Vec<_>>()
        } else {
            // Combine the contributions to each tile elementwise using the sharding's unreduced kind, in ascending
            // order of their unreduced coordinates. That order does not depend on which replica of each contribution
            // this process holds, so every process computes the same bytes even for floating point sums, which do not
            // associate. Boolean values do not support addition, so their sums combine with logical or instead, whose
            // identity is `false`.
            contributions.sort_by(|(_, left, _), (_, right, _)| left.cmp(right));
            let unreduced_kind = self.sharding().unreduced_kind();
            let mut tiles = Vec::<(Vec<Range<usize>>, CoreArray)>::new();
            for (slices, _, bytes) in contributions {
                let shape = slices.iter().map(|slice| slice.len()).collect::<Vec<_>>();
                let contribution = CoreArray::new(ArrayType::new_static(element_type, shape), bytes)?;
                match tiles.iter_mut().find(|(tile_slices, _)| *tile_slices == slices) {
                    Some((_, tile)) => {
                        *tile = match unreduced_kind {
                            UnreducedKind::Sum if element_type == DataType::Boolean => Or::or(&*tile, &contribution)?,
                            UnreducedKind::Sum => Add::add(&*tile, &contribution)?,
                            UnreducedKind::Max => Max::max(&*tile, &contribution)?,
                            UnreducedKind::Min => Min::min(&*tile, &contribution)?,
                        }
                    }
                    None => tiles.push((slices, contribution)),
                }
            }

            tiles.into_iter().map(|(slices, tile)| (slices, tile.logical_bytes())).collect()
        };

        // Reuse a single full tile's bytes when the copies and any reduction already produced the complete value.
        // A scalar array has a single tile, so it always returns here.
        if tiles.len() == 1 {
            let (slices, bytes) = tiles.pop().unwrap();
            let covers_global_shape = slices.len() == global_shape.rank()
                && slices
                    .iter()
                    .zip(global_shape.as_slice())
                    .all(|(slice, extent)| slice.start == 0 && slice.end == *extent);
            if covers_global_shape && bytes.len() == total_byte_count {
                return Ok(bytes);
            }
            tiles.push((slices, bytes));
        }

        // Copy the tiles into the global buffer. They are disjoint, because the tiles of a sharding are either
        // disjoint or identical and the selection holds one replica of each.
        let mut global_bytes = vec![0u8; total_byte_count];
        for (slices, bytes) in tiles {
            let tile_shape = slices.iter().map(|slice| slice.len()).collect::<Vec<_>>();
            let tile_strides = row_major_element_strides(tile_shape.as_slice(), element_type)?;
            copy_dense_shard_bytes(
                bytes.as_slice(),
                slices.as_slice(),
                global_strides.as_slice(),
                tile_strides.as_slice(),
                0,
                0,
                0,
                element_size_in_bytes,
                &mut global_bytes,
            );
        }

        Ok(global_bytes)
    }

    /// Returns the error that [`Self::copy_to_host_bytes`] reports before it issues any copy when the current process
    /// does not hold the whole value of this array (i.e., [`ArrayError::MissingShardBuffer`]) or when the devices of
    /// this array can hold different values of one tile along a varying manual mesh axis, so that a caller which reads
    /// the value back later can report that error right away.
    pub(super) fn validate_host_readback(&self) -> Result<(), Error> {
        let global_shape = self.shape();
        if self.data_type().is_zero() || global_shape.as_slice().contains(&0) {
            return Ok(());
        }
        self.readback_shards(self.unreduced_axis_indices().as_slice())?;
        Ok(())
    }

    /// Returns an [`ExecutionFence`] that completes once [`Self::copy_to_host_bytes`] can read this array back without
    /// waiting for anything but its own copies (i.e., once the execution that produced this array and the buffers that
    /// the readback copies are ready), and that reports the first error among them. The fence waits for exactly the
    /// buffers that the readback waits for, which [`Self::host_readback_shards`] selects for both (i.e., one local
    /// replica of each distinct contribution to the value). It therefore does not wait for any other replica (e.g., a
    /// replica that a producer fills only once the result of the upload that reads this array back is ready). A host
    /// upload that reads this array back waits for this fence before it occupies a worker (refer to
    /// [`HostUploadWorkers`]).
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::MissingShardBuffer`] as [`Self::validate_host_readback`] does, and the PJRT errors
    /// of querying the readiness of the shard buffers.
    pub(super) fn host_readback_fence(&self) -> Result<ExecutionFence, Error> {
        let global_shape = self.shape();
        let shards = if self.data_type().is_zero() || global_shape.as_slice().contains(&0) {
            // A readback without elements copies nothing and waits for the producing execution or, without one,
            // for every addressable buffer.
            match self.execution_fence() {
                Some(_) => Vec::new(),
                None => self.addressable_shards().collect(),
            }
        } else {
            self.host_readback_shards(self.unreduced_axis_indices().as_slice())?.0
        };
        let events = shards
            .into_iter()
            .filter_map(ArrayShard::buffer)
            .map(|buffer| buffer.ready())
            .collect::<Result<Vec<_>, _>>()?;
        let buffers = ExecutionFence::new(events);
        Ok(ExecutionFence::join(self.execution_fence().cloned().into_iter().chain([buffers])))
    }

    /// Selects the shards that [`Self::copy_to_host_bytes`] of an array with elements copies, given the indices of the
    /// unreduced axes of the sharding within its mesh, and returns them together with whether they are tiles that the
    /// readback copies directly into the dense destination. [`Self::host_readback_fence`] waits for the same shards.
    /// The selection is that of [`Self::readback_shards`] (i.e., one local replica of each distinct contribution to
    /// the value). The readback copies the selected tiles directly when there are no unreduced axes and every tile
    /// occupies one contiguous byte range of the dense representation (e.g., when only the leading dimension is
    /// sharded, or for a replicated array). Otherwise, it assembles their copies on the host.
    fn host_readback_shards(&self, unreduced_axis_indices: &[usize]) -> Result<(Vec<&ArrayShard>, bool), Error> {
        let shards = self.readback_shards(unreduced_axis_indices)?;
        let global_shape = self.shape();

        // A tile occupies one contiguous span if every dimension after the first non-unit slice is whole.
        let contiguous_tiles = unreduced_axis_indices.is_empty()
            && shards.iter().all(|shard| {
                let slices = shard.slice();
                let first_non_unit_dimension = slices.iter().position(|slice| slice.len() != 1).unwrap_or(slices.len());
                slices
                    .iter()
                    .zip(global_shape.as_slice())
                    .skip(first_non_unit_dimension + 1)
                    .all(|(slice, extent)| slice.start == 0 && slice.end == *extent)
            });
        Ok((shards, contiguous_tiles))
    }

    /// Selects one local replica of each distinct contribution to the value of this array, which is what
    /// [`Self::copy_to_host_bytes`] reads, given the indices of the unreduced axes of the sharding within its mesh.
    /// A contribution is identified by the tile of a shard and its coordinates along the unreduced axes (refer to
    /// [`Self::unreduced_coordinates`]), and its selected replica is the first addressable one in shard order, so the
    /// selection is deterministic. The other replicas are not read, as in JAX. The tiles of a sharding are either
    /// disjoint or identical, so the selection holds the whole value exactly when the current process can assemble
    /// it (e.g., for a replicated array, or for an unreduced array whose summands each have a local replica). A tile
    /// without elements (e.g., the last tile of an uneven split) contributes nothing, so it needs no local replica and
    /// is never selected, even when this process holds one. The readback and its fence therefore never wait for the
    /// buffer of such a tile, which the separate execution fence of the producer still covers. Returns
    /// [`ArrayError::MissingShardBuffer`] for the first shard with elements whose contribution has no local replica.
    ///
    /// Devices whose coordinates differ along a varying manual mesh axis (refer to [`Sharding::varying_manual_axes`])
    /// can hold different values for one contribution, which one host value cannot represent. The selection rejects
    /// such an array with [`ProgramError::InvalidArgument`] when two shards with elements share a contribution but
    /// not their coordinates along those axes, which is the case for a varying axis that does not split the array.
    /// The check reads only the metadata of every shard, so it rejects the same arrays in every process.
    fn readback_shards(&self, unreduced_axis_indices: &[usize]) -> Result<Vec<&ArrayShard>, Error> {
        let all_shards = self.shards();
        let contribution =
            |shard: &ArrayShard| (shard.slice().to_vec(), self.unreduced_coordinates(shard, unreduced_axis_indices));
        let required = |shard: &&ArrayShard| !shard.slice().iter().any(Range::is_empty);

        let varying_manual_axes = self.sharding().varying_manual_axes();
        if !varying_manual_axes.is_empty() {
            let varying_axis_indices = self
                .mesh()
                .logical_mesh()
                .axes()
                .iter()
                .enumerate()
                .filter(|(_, axis)| varying_manual_axes.contains(axis.name()))
                .map(|(axis_index, _)| axis_index)
                .collect::<Vec<_>>();
            let mut first_shards = HashMap::new();
            for shard in all_shards.iter().filter(required) {
                let coordinates = self.unreduced_coordinates(shard, varying_axis_indices.as_slice());
                let (first_index, first_coordinates) =
                    first_shards.entry(contribution(shard)).or_insert((shard.index(), coordinates.clone()));
                if *first_coordinates != coordinates {
                    let axes = varying_manual_axes.iter().map(|axis| format!("`{axis}`")).collect::<Vec<_>>();
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "cannot read back an array that varies along the manual mesh axes {} as one host value, \
                             because shards #{first_index} and #{} can hold different values of the same slice",
                            axes.join(", "),
                            shard.index(),
                        ),
                    }
                    .into());
                }
            }
        }

        let mut covered = HashSet::new();
        let selected = all_shards
            .iter()
            .filter(required)
            .filter(|shard| shard.buffer().is_some() && covered.insert(contribution(shard)))
            .collect::<Vec<_>>();

        let missing = all_shards.iter().filter(required).find(|shard| !covered.contains(&contribution(shard)));
        if let Some(missing) = missing {
            return Err(Error::from(ArrayError::MissingShardBuffer {
                shard_index: missing.index(),
                device_id: missing.device().id(),
            }));
        }

        Ok(selected)
    }

    /// Returns the indices of the unreduced axes of the sharding of this array within its mesh, in mesh order.
    fn unreduced_axis_indices(&self) -> Vec<usize> {
        self.mesh()
            .logical_mesh()
            .axes()
            .iter()
            .enumerate()
            .filter(|(_, axis)| self.sharding().unreduced_axes().contains(axis.name()))
            .map(|(axis_index, _)| axis_index)
            .collect()
    }

    /// Returns the coordinates of the device of `shard` along the mesh axes at `axis_indices` (e.g., the unreduced
    /// axes of [`Self::unreduced_axis_indices`]). Shards with the same tile and the same coordinates along unreduced
    /// axes hold replicas of one contribution to the value (e.g., one summand), which a readback needs only once.
    fn unreduced_coordinates(&self, shard: &ArrayShard, axis_indices: &[usize]) -> Vec<usize> {
        match self.mesh().device_coordinates(shard.index()) {
            Some(coordinates) => axis_indices.iter().map(|&axis| coordinates[axis]).collect(),
            None => Vec::new(),
        }
    }
}

/// Uploads of a dense row-major host buffer to the addressable shards of an [`Array`]. [`Array::from_host_buffer`]
/// issues them right away and [`Array::from_host_buffer_in_background`] issues them once the bytes are available.
struct HostBufferUploads {
    /// Global [`ArrayType`] of the array, with its placement resolved.
    r#type: ArrayType,

    /// Runtime [`Sharding`] of the array over [`Self::mesh`].
    sharding: Sharding,

    /// [`DeviceMesh`] of the array.
    mesh: DeviceMesh,

    /// Number of bytes of the dense row-major host buffer of the full logical array.
    byte_count: usize,

    /// Physical layout of every shard buffer, or [`None`] for the default layout of the PJRT plugin.
    device_layout: Option<ryft_pjrt::Layout>,

    /// Uploads of the shards owned by this process, in shard order. Arrays of [`DataType::Zero`] have none.
    shards: Vec<HostShardUpload>,
}

impl HostBufferUploads {
    /// Resolves the placement of a dense row-major host buffer and derives the bytes that each addressable shard
    /// receives. Refer to [`Array::from_host_buffer`] for the meaning of the parameters. `byte_count` is the size
    /// of the host buffer, which is validated against the type when it is already known.
    fn new(
        session: &Session,
        r#type: ArrayType,
        mesh: Option<DeviceMesh>,
        sharding: Option<Sharding>,
        byte_count: Option<usize>,
    ) -> Result<Self, Error> {
        let logical_mesh = sharding.as_ref().or(r#type.sharding()).map(Sharding::mesh);
        let mesh = match mesh {
            Some(mesh) => mesh,
            None => session.device_mesh_for(&[], logical_mesh)?,
        };

        // Normalize missing sharding metadata before any placement-derived work. A dense host buffer with no explicit
        // sharding is interpreted as one replicated logical array over the provided mesh.
        let (r#type, sharding) = resolve_placement(r#type, sharding, &mesh)?;

        // Validate that the host buffer contains exactly the dense row-major bytes for the full logical array.
        let expected_byte_count = r#type.dense_size_in_bytes()?;
        if let Some(byte_count) = byte_count
            && byte_count != expected_byte_count
        {
            return Err(Error::from(ArrayError::HostBufferByteCountMismatch {
                expected: expected_byte_count,
                actual: byte_count,
            }));
        }

        // Each process uploads only the shards of the devices that the mesh assigns to it, so a device of this process
        // that the mesh assigns to another process would receive no buffer at all.
        let client = session.client();
        validate_mesh_device_processes(client, &mesh)?;

        if r#type.data_type().is_zero() {
            return Ok(Self {
                r#type,
                sharding,
                mesh,
                byte_count: expected_byte_count,
                device_layout: None,
                shards: Vec::new(),
            });
        }

        // Build a lookup table for the PJRT devices that this process can upload to directly.
        let client_process_index = client.process_index()?;
        let addressable_devices = client.addressable_devices()?;
        let mut addressable_device_by_id = HashMap::with_capacity(addressable_devices.len());
        for device in addressable_devices {
            addressable_device_by_id.insert(device.id()?, device);
        }

        // Derive the global shard layout and plan uploads only for the shards owned by this process.
        let data_type = r#type.data_type();
        let device_layout = r#type.layout().map(ToPjrt::to_pjrt).transpose()?;
        let shape = r#type
            .static_shape()
            .ok_or_else(|| Error::from(ArrayError::DynamicShape { shape: r#type.shape().clone() }))?;
        let layout = ShardLayout::new(&shape, &mesh, &sharding)?;
        let no_axes = BTreeSet::new();

        // Along an unreduced axis of kind sum, the devices hold summands of the logical value. The devices at
        // coordinate zero along every unreduced axis receive the host value and the others receive the additive
        // identity, so the summands add up to it. Copies of the host value are their own maximum and minimum, so the
        // other kinds need no identity.
        let summed_axes =
            if sharding.unreduced_kind() == UnreducedKind::Sum { sharding.unreduced_axes() } else { &no_axes };
        let unreduced_axis_indices = mesh
            .logical_mesh()
            .axes()
            .iter()
            .enumerate()
            .filter(|(_, axis)| summed_axes.contains(axis.name()))
            .map(|(axis_index, _)| axis_index)
            .collect::<Vec<_>>();
        let holds_identity = |shard: &ShardDescriptor| {
            mesh.device_coordinates(shard.index())
                .is_some_and(|coordinates| unreduced_axis_indices.iter().any(|&axis| coordinates[axis] != 0))
        };

        // The identity is needed only when some device, on this or another process, receives a non-empty shard that
        // must hold it. An unreduced axis of size one or an array without elements writes none, so those placements
        // also accept data types without an identity. Deciding this from the global layout makes every process agree.
        let needs_identity = layout
            .descriptors()
            .iter()
            .any(|shard| holds_identity(shard) && !shard.shape().as_slice().contains(&0));
        let additive_identity = if needs_identity {
            // Use negative zero for each signed floating-point component so adding the identity preserves negative
            // zero. FNUZ types have only unsigned zero. Sub-byte elements occupy the low bits of one host byte.
            // Boolean sums combine with logical or, whose identity is `false`.
            match data_type {
                DataType::Token | DataType::Zero | DataType::F8E8M0FNU => {
                    return Err(ArrayError::MissingUnreducedSumIdentity { data_type }.into());
                }
                DataType::Boolean => vec![0x00],
                DataType::I1
                | DataType::I2
                | DataType::I4
                | DataType::I8
                | DataType::I16
                | DataType::I32
                | DataType::I64
                | DataType::U1
                | DataType::U2
                | DataType::U4
                | DataType::U8
                | DataType::U16
                | DataType::U32
                | DataType::U64 => vec![0u8; data_type.size_in_bytes()?],
                DataType::F4E2M1FN => vec![0x08],
                DataType::F6E2M3FN | DataType::F6E3M2FN => vec![0x20],
                DataType::F8E3M4 | DataType::F8E4M3 | DataType::F8E4M3FN | DataType::F8E5M2 => vec![0x80],
                DataType::F8E4M3FNUZ | DataType::F8E4M3B11FNUZ | DataType::F8E5M2FNUZ => vec![0x00],
                DataType::BF16 | DataType::F16 => 0x8000u16.to_ne_bytes().to_vec(),
                DataType::F32 => (-0f32).to_ne_bytes().to_vec(),
                DataType::F64 => (-0f64).to_ne_bytes().to_vec(),
                DataType::C64 => [(-0f32).to_ne_bytes(); 2].concat(),
                DataType::C128 => [(-0f64).to_ne_bytes(); 2].concat(),
            }
        } else {
            Vec::new()
        };

        // Row-major strides in units of elements translate each shard slice into flat element offsets in the full dense
        // host buffer. Only the shards of an array with elements read bytes at such offsets.
        let strides = if shape.as_slice().contains(&0) {
            Vec::new()
        } else {
            row_major_element_strides(shape.as_slice(), data_type)?
        };

        let mut shards = Vec::new();
        for shard in layout.descriptors() {
            let shard_device = shard.device();
            if shard_device.process_index() != client_process_index {
                continue;
            }

            let shard_device_id = shard_device.id();
            let device = addressable_device_by_id.get(&shard_device_id).ok_or_else(|| {
                Error::from(ArrayError::NonAddressableDevice {
                    device_id: shard_device_id,
                    process_index: client_process_index,
                })
            })?;

            // Upload the local shard bytes using the static shape of the shard. For non-scalar shards, the host
            // window starts at the shard's first element in the dense source buffer, and host byte strides let
            // `Array::from_host_buffer` read the shard without materializing a packed temporary vector.
            let shard_shape = shard.shape();
            let shard_dimensions = shard_shape.as_slice().iter().map(|&dimension| dimension as u64).collect::<Vec<_>>();
            let shard_slice = shard.slice();
            let buffer_type = data_type.executable_buffer_type();

            // An empty shard that holds the identity receives no bytes, so its empty identity element is not read.
            let source = if holds_identity(shard) {
                HostShardSource::AdditiveIdentity {
                    element: additive_identity.clone(),
                    count: shard_shape.element_count()?,
                }
            } else if shard_slice.is_empty() {
                HostShardSource::Window { window: 0..expected_byte_count, byte_strides: None }
            } else if shard_shape.as_slice().contains(&0) {
                // An empty shard reads no bytes, but the offset of its first element can lie past the end of the host
                // buffer (e.g., the second column shard of an `f32[0, 4]` array would start at byte 8 of an empty
                // buffer). Its window is therefore the empty range at the start of the buffer.
                HostShardSource::Window { window: 0..0, byte_strides: None }
            } else {
                // Convert the shard's starting coordinates into a flat row-major element offset in the full host
                // buffer, and then into the corresponding byte offset.
                let element_size_in_bytes = buffer_type.element_size_in_bytes()?;
                let start_element_offset = shard_slice
                    .iter()
                    .enumerate()
                    .map(|(dimension, slice)| slice.start * strides[dimension])
                    .sum::<usize>();
                let start_byte_offset = start_element_offset * element_size_in_bytes;

                // Cover every byte that may be read for this shard, which ends with its last element.
                let last_element_offset = start_element_offset
                    + shard_shape
                        .as_slice()
                        .iter()
                        .enumerate()
                        .map(|(dimension, size)| (size - 1) * strides[dimension])
                        .sum::<usize>();
                let end_byte_offset = last_element_offset * element_size_in_bytes + element_size_in_bytes;

                // The elements of the shard are distinct positions of its window, in row-major order, so the shard is
                // contiguous in the dense row-major host buffer exactly when they fill the window (e.g., a column tile
                // of a single row, whose trailing dimension is partial). Such a shard reads the window as a packed
                // dense shard without explicit strides. Otherwise, describe the global row-major strides in bytes so
                // that the strided shard can be gathered directly from the original host buffer.
                let dense_byte_count = shard_shape.as_slice().iter().product::<usize>() * element_size_in_bytes;
                let byte_strides = if end_byte_offset - start_byte_offset == dense_byte_count {
                    None
                } else {
                    Some(
                        strides
                            .iter()
                            .map(|stride| {
                                let byte_stride = stride * element_size_in_bytes;
                                i64::try_from(byte_stride).map_err(|_| Error::SizeLimitExceeded {
                                    message: format!(
                                        "row-major byte stride for array with shape `{}` and element type \
                                         `{}` exceeds the maximum allowed size of {}",
                                        shape,
                                        buffer_type,
                                        i64::MAX,
                                    ),
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                };

                HostShardSource::Window { window: start_byte_offset..end_byte_offset, byte_strides }
            };

            shards.push(HostShardUpload { device: device.clone(), dimensions: shard_dimensions, source });
        }

        Ok(Self { r#type, sharding, mesh, byte_count: expected_byte_count, device_layout, shards })
    }
}

/// Upload of the host bytes of one addressable shard.
struct HostShardUpload {
    /// PJRT device that holds the shard. Its memory that the array type requests receives the bytes.
    device: PjrtDevice<'static>,

    /// Dimensions of the shard.
    dimensions: Vec<u64>,

    /// Source of the bytes to upload to this shard.
    source: HostShardSource,
}

/// Describes how to obtain the bytes for one addressable shard from a host buffer or a repeated additive identity.
#[derive(PartialEq, Eq, Hash)]
enum HostShardSource {
    /// `count` copies of `element`, which encodes the additive identity of one element, for a summand along an
    /// unreduced axis that does not hold the host value.
    AdditiveIdentity { element: Vec<u8>, count: usize },

    /// Bytes of the host buffer within `window`, which starts at the first element of the shard. With `byte_strides`,
    /// the shard is not contiguous in the host buffer, and these global row-major strides locate its elements within
    /// the window. Otherwise, the window holds the dense row-major bytes of the shard.
    Window { window: Range<usize>, byte_strides: Option<Vec<i64>> },
}

impl HostShardSource {
    /// Returns the dense row-major bytes for a shard with `dimensions`. Window sources read from the host buffer
    /// `bytes`, whose elements occupy `element_size_in_bytes` bytes each.
    fn to_dense_bytes(&self, bytes: &[u8], dimensions: &[u64], element_size_in_bytes: usize) -> Vec<u8> {
        match self {
            Self::AdditiveIdentity { element, count } => element.repeat(*count),
            Self::Window { window, byte_strides: None } => bytes[window.clone()].to_vec(),
            Self::Window { window, byte_strides: Some(byte_strides) } => {
                if dimensions.contains(&0) {
                    return Vec::new();
                }

                // The last dimension is contiguous in the row-major host buffer, so copy one row of it at a time and
                // visit the row indices of the leading dimensions in row-major order.
                let window = &bytes[window.clone()];
                let (row_dimensions, row_extent) = dimensions.split_at(dimensions.len() - 1);
                let row_byte_count = row_extent[0] as usize * element_size_in_bytes;
                let row_count = row_dimensions.iter().product::<u64>() as usize;
                let mut dense_bytes = Vec::with_capacity(row_count * row_byte_count);
                let mut row_index = vec![0usize; row_dimensions.len()];
                for _ in 0..row_count {
                    let offset = row_index
                        .iter()
                        .zip(byte_strides)
                        .map(|(&index, &stride)| index * stride as usize)
                        .sum::<usize>();
                    dense_bytes.extend_from_slice(&window[offset..offset + row_byte_count]);
                    for dimension in (0..row_dimensions.len()).rev() {
                        row_index[dimension] += 1;
                        if row_index[dimension] < row_dimensions[dimension] as usize {
                            break;
                        }
                        row_index[dimension] = 0;
                    }
                }
                dense_bytes
            }
        }
    }
}

/// Dense row-major bytes of the addressable shards of a host upload, which [`Array::from_host_buffer_in_background`]
/// prepares from one host buffer once that buffer is available. A shard that is contiguous in the host buffer is a
/// window of it, so it shares the allocation of the host buffer without a copy (e.g., every replica of a fully
/// replicated array, which would otherwise hold a copy of the whole array each). The other shards need bytes of their
/// own (i.e., the strided shards, whose elements must be gathered, and the additive identities along unreduced axes),
/// and those are prepared once for all the shards that hold the same bytes (e.g., the replicas of one strided tile).
struct PreparedHostShards {
    /// Dimensions and source of each shard, in shard order.
    shards: Vec<(Vec<u64>, HostShardSource)>,

    /// Index of the first shard with the same dimensions and source as each shard, in shard order. The shards with
    /// equal dimensions and sources share the slice prepared for that first shard.
    representatives: Vec<usize>,

    /// Slices prepared for strided windows and additive identities, indexed by the shards of
    /// [`Self::representatives`].
    prepared: Vec<Option<HostBufferSlice>>,

    /// Size of one element in bytes.
    element_size_in_bytes: usize,
}

impl PreparedHostShards {
    /// Creates a cache for `shards` from their dimensions, sources, and size of each element in bytes.
    fn new(shards: Vec<(Vec<u64>, HostShardSource)>, element_size_in_bytes: usize) -> Self {
        // Index the shards by their dimensions and sources, so that finding the shards that share bytes does not scan
        // the earlier shards for every distinct strided tile.
        let mut first_shards = HashMap::with_capacity(shards.len());
        let representatives = shards
            .iter()
            .enumerate()
            .map(|(index, shard)| *first_shards.entry(shard).or_insert(index))
            .collect();
        let prepared = vec![None; shards.len()];
        Self { shards, representatives, prepared, element_size_in_bytes }
    }

    /// Returns a slice containing the dense row-major bytes of the shard at `index`. A contiguous window shares the
    /// allocation of `host_bytes`, which must be the same buffer in every call. Other sources share the allocation
    /// prepared for the first shard with the same dimensions and source.
    fn bytes(&mut self, host_bytes: &Arc<Vec<u8>>, index: usize) -> HostBufferSlice {
        let (dimensions, source) = &self.shards[index];
        if let HostShardSource::Window { window, byte_strides: None } = source {
            return HostBufferSlice::new(Arc::clone(host_bytes), window.clone());
        }

        self.prepared[self.representatives[index]]
            .get_or_insert_with(|| {
                HostBufferSlice::from(source.to_dense_bytes(host_bytes, dimensions, self.element_size_in_bytes))
            })
            .clone()
    }
}

/// A contiguous byte slice that keeps its host buffer alive. Clones share the backing allocation, so each call to
/// [`HostToDeviceTransferManager::transfer_literal`] can retain a slice until PJRT has read it.
#[derive(Clone)]
pub(super) struct HostBufferSlice {
    /// Shared host allocation.
    bytes: Arc<Vec<u8>>,

    /// Range of [`Self::bytes`] that this slice covers.
    range: Range<usize>,
}

impl HostBufferSlice {
    /// Creates a slice covering `range`, which must lie within `bytes`.
    fn new(bytes: Arc<Vec<u8>>, range: Range<usize>) -> Self {
        Self { bytes, range }
    }
}

impl From<Vec<u8>> for HostBufferSlice {
    fn from(bytes: Vec<u8>) -> Self {
        let range = 0..bytes.len();
        Self::new(Arc::new(bytes), range)
    }
}

impl AsRef<[u8]> for HostBufferSlice {
    fn as_ref(&self) -> &[u8] {
        &self.bytes[self.range.clone()]
    }
}

/// Destination buffer of an upload from host memory whose bytes become available only after the buffer has been
/// handed out, together with the PJRT [`HostToDeviceTransferManager`] that fills it. The manager is created for the
/// memory of one device, so an upload to several devices needs several managers. It borrows the `'static` client of
/// a [`Session`], so the upload job can move the manager to a background worker.
pub(super) struct PendingHostUpload {
    /// Transfer manager that created the destination buffer, at index zero, and fills it.
    manager: HostToDeviceTransferManager<'static>,

    /// [`BufferType`] of the elements of the destination buffer.
    buffer_type: BufferType,

    /// Dimensions of the destination buffer.
    dimensions: Vec<u64>,
}

impl PendingHostUpload {
    /// Creates the transfer manager for one destination buffer and returns it with that buffer, whose contents are
    /// not yet defined. With [`None`] for `layout`, the PJRT plugin chooses the default device layout of the buffer
    /// when it creates the manager, so this function does not need the plugin to report that layout.
    ///
    /// Returns [`None`] when the PJRT plugin does not implement transfer managers, or when its transfer managers do
    /// not support [literal transfers](HostToDeviceTransferManager::transfer_literal), which [`run_host_uploads`] uses
    /// to convert the dense row-major host bytes to the layout of the buffer. Both are checked before the buffer is
    /// retrieved from the manager, so that no caller can hand out a buffer that could never receive its bytes. The
    /// caller can then upload the bytes once they are available instead (e.g., with [`Client::buffer`]).
    ///
    /// # Parameters
    ///
    ///   - `client`: PJRT client of the [`Session`] that the destination belongs to.
    ///   - `buffer_type`: [`BufferType`] of the elements of the destination buffer.
    ///   - `dimensions`: Dimensions of the destination buffer.
    ///   - `memory`: PJRT memory that holds the destination buffer.
    ///   - `layout`: Physical layout of the destination buffer, or [`None`] for the default layout of the plugin.
    pub(super) fn new(
        client: &'static Client<'static>,
        buffer_type: BufferType,
        dimensions: Vec<u64>,
        memory: PjrtMemory<'_>,
        layout: Option<ryft_pjrt::Layout>,
    ) -> Result<Option<(Self, Buffer<'static>)>, Error> {
        let manager = match layout {
            Some(layout) => {
                let specification = BufferSpecification::with_layout(buffer_type, dimensions.as_slice(), layout);
                client.host_to_device_transfer_manager(vec![specification], memory)
            }
            None => client.host_to_device_transfer_manager_with_default_layouts(
                vec![(buffer_type, dimensions.as_slice())],
                memory,
            ),
        };

        let manager = match manager {
            Ok(manager) => manager,
            Err(PjrtError::Unimplemented { .. } | PjrtError::MissingFunction { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };

        // Dropping a manager whose buffer was never retrieved releases that buffer,
        // so declining here leaves nothing behind.
        if !manager.supports_transfer_literal() {
            return Ok(None);
        }

        let buffer = manager.retrieve_buffer(0)?;

        Ok(Some((Self { manager, buffer_type, dimensions }, buffer)))
    }
}

/// Bounded set of named background threads (i.e., workers) that run the jobs of host uploads whose destination buffers
/// have already been handed out (refer to [`run_host_uploads`]). Every [`Session`] owns one, which all handles of that
/// session share like its other resources, so the threads that pending host uploads occupy are bounded per session
/// instead of growing with the number of uploads whose producers have stalled.
///
/// # Jobs
///
/// Each job comes with an [`ExecutionFence`] of its sources (i.e., of everything that it waits for, such as the arrays
/// that it reads back). [`Self::submit`] accepts the job right away, but the job is parked outside the workers until
/// that fence is ready, and only then does it join the queue that the workers take jobs from, even when the fence
/// reports an error, which the job itself then reports (e.g., as the error of its readback). The readiness callback
/// that queues the job may run on a PJRT thread, so it only moves the job into the queue and wakes a worker. When PJRT
/// cannot register a callback for the events of a fence, [`ExecutionFence`] waits for those events on the thread that
/// builds the fence instead, which still queues the job once they complete.
///
/// # Workers
///
/// Workers start lazily, so a session that never uploads in the background starts no thread. [`Self::submit`] starts
/// one, on the submitting thread, whenever fewer workers run than there are accepted jobs (i.e., parked, queued, or
/// running ones) and fewer than the limit of [`Self::new`], so a job that its sources release always finds a worker.
/// A worker takes queued jobs until its owner is dropped and no accepted job remains. Jobs run without any lock held,
/// and a job that panics is dropped while it unwinds, which releases what it owns (e.g., its transfer managers, so
/// that PJRT poisons the buffers that have not received their bytes), and its worker goes on with the next job.
///
/// # Ordering and Progress
///
/// A queued job may block only on the work that its fence covers, which is complete by the time the job is queued, and
/// on PJRT transfers, which complete on their own. In particular, a job never waits for another job or for an external
/// producer while it holds a worker, so every queued job completes once a worker takes it, and some worker is always
/// running while a job is accepted. A job that depends on another one (e.g., because it reads back an array whose
/// buffers that job fills) is ordered after it by its fence, which completes only once that job has filled those
/// buffers, so dependent chains complete in dependency order with any number of workers. A job whose sources stay
/// pending (e.g., on a buffer that a transfer manager fills only later) stays parked without holding a worker, so
/// it does not delay independent jobs. Workers take queued jobs in the order in which their sources became ready.
///
/// # Shutdown
///
/// Dropping the owner (i.e., when the last handle of its session goes away) stops each worker once no accepted job
/// remains. The parked and queued jobs still run rather than being dropped (which would poison their destination
/// buffers), because those buffers can outlive the session (e.g., in arrays that were moved to another session of the
/// same client, or in PJRT buffers retained elsewhere) and must then hold the right values. The owner does not join
/// its workers, because a worker can drop the owner itself (e.g., when a job holds the last handle of the session) and
/// because a job can stay parked for as long as its sources stay pending. The workers themselves hold only the shared
/// queue, so they do not keep the owner alive. A job, however, retains whatever it captures, which can include a handle
/// of the session (e.g., the source [`Array`] that the host fallback of [`Array::to_placement`] reads back, which also
/// retains the caches of its session). Such a job keeps the owner alive until it has run, so the owner shuts down only
/// after the sources of that job have completed and the job has released its captures.
pub(crate) struct HostUploadWorkers {
    /// State that the workers share with this owner.
    state: Arc<HostUploadWorkersState>,
}

impl HostUploadWorkers {
    /// Largest number of workers that [`Self::default`] allows. Workers mostly wait for PJRT transfers, and their own
    /// work is copying memory, so a few of them already overlap the independent uploads of a session. A fixed cap keeps
    /// the number of threads of a session small irrespective of the size of the machine.
    const MAX_DEFAULT_WORKER_COUNT: usize = 4;

    /// Creates a set of workers that runs at most `worker_limit` threads, none of which start before the first job
    /// is submitted.
    fn new(worker_limit: NonZeroUsize) -> Self {
        Self {
            state: Arc::new(HostUploadWorkersState {
                queue: Mutex::new(HostUploadQueue::default()),
                available: Condvar::new(),
                worker_limit: worker_limit.get(),
            }),
        }
    }

    /// Accepts `job`, which runs on one of the workers once `sources` is ready, and starts a new worker when fewer run
    /// than there are accepted jobs and the limit allows one. Returns without waiting for `sources` or for `job`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if no worker is running and none can be started, in which case `job` is dropped without
    /// running before this function returns. When some worker is running, a failure to start another one leaves `job`
    /// to the running workers, which run it eventually (refer to the progress argument of [`HostUploadWorkers`]).
    pub(crate) fn submit<J: 'static + Send + FnOnce()>(&self, sources: ExecutionFence, job: J) -> Result<(), Error> {
        // Account for the job and make sure that a worker exists for it before its sources can release it, because the
        // readiness callback that queues it must not start threads.
        let mut queue = self.state.lock();
        queue.accepted_job_count += 1;
        if queue.worker_count < queue.accepted_job_count.min(self.state.worker_limit) {
            let state = Arc::clone(&self.state);
            match std::thread::Builder::new().name("ryft-host-upload".to_string()).spawn(move || state.run()) {
                Ok(_) => {
                    // Dropping the handle detaches the worker. It exits once the owner is gone and all jobs finish.
                    queue.worker_count += 1
                }
                Err(error) if queue.worker_count == 0 => {
                    // Returning drops the local queue guard before the job parameter, so the job releases its
                    // captured resources outside the lock.
                    queue.accepted_job_count -= 1;
                    return Err(error.into());
                }
                Err(_) => {
                    // A running worker takes the job once it is queued and that worker is free, so the job still runs.
                }
            }
        }

        // Release the lock before registering the callback. If the fence is already ready, `on_ready` calls
        // `enqueue` inline, which would deadlock trying to lock the same queue.
        drop(queue);

        let state = Arc::clone(&self.state);
        sources.on_ready(move |_| state.enqueue(Box::new(job)));
        Ok(())
    }
}

impl Default for HostUploadWorkers {
    fn default() -> Self {
        // A machine that cannot report its parallelism still gets one worker.
        let parallelism = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        Self::new(NonZeroUsize::new(parallelism.min(Self::MAX_DEFAULT_WORKER_COUNT)).unwrap())
    }
}

impl Drop for HostUploadWorkers {
    fn drop(&mut self) {
        // Idle workers wake up and exit unless jobs remain, and the others exit once the last accepted job completes.
        self.state.lock().shut_down = true;
        self.state.available.notify_all();
    }
}

/// State that a [`HostUploadWorkers`] owner shares with its workers.
struct HostUploadWorkersState {
    /// Queued jobs and the bookkeeping of the workers.
    queue: Mutex<HostUploadQueue>,

    /// Wakes idle workers when a job is queued, when the owner is dropped, or when the last accepted job completes
    /// after the owner was dropped.
    available: Condvar,

    /// Largest number of workers.
    worker_limit: usize,
}

impl HostUploadWorkersState {
    /// Runs the loop of one worker, which takes the queued jobs in queue order and runs each of them without holding
    /// the lock, and returns once the owner has been dropped and no accepted job remains.
    fn run(&self) {
        let mut queue = self.lock();
        loop {
            if let Some(job) = queue.jobs.pop_front() {
                // Unlock before running the job so other workers can take jobs and callbacks can enqueue them.
                // The job can also drop the last session handle, which locks this queue during shutdown.
                drop(queue);

                // Unwinding drops what the job owns, and the panic itself has already been reported by the panic hook.
                if let Err(payload) = catch_unwind(AssertUnwindSafe(job))
                    // Drop the payload inside a second `catch_unwind` because its destructor can panic. The worker
                    // must still decrement the accepted job count and continue serving the queue.
                    && let Err(drop_panic) = catch_unwind(AssertUnwindSafe(|| drop(payload)))
                    && !drop_panic.is::<&'static str>()
                    && !drop_panic.is::<String>()
                {
                    // String payloads drop safely. A custom payload from the second panic could itself panic on
                    // drop, so leak that payload to keep the worker running.
                    std::mem::forget(drop_panic);
                }

                queue = self.lock();
                queue.accepted_job_count -= 1;
                if queue.shut_down && queue.accepted_job_count == 0 {
                    self.available.notify_all();
                }
            } else if queue.shut_down && queue.accepted_job_count == 0 {
                queue.worker_count -= 1;
                return;
            } else {
                queue = self.available.wait(queue).unwrap_or_else(PoisonError::into_inner);
            }
        }
    }

    /// Queues `job`, whose sources are ready, and wakes a worker for it. This function only takes the lock and
    /// notifies, so it can run inside a PJRT callback.
    fn enqueue(&self, job: Box<dyn FnOnce() + Send>) {
        self.lock().jobs.push_back(job);
        self.available.notify_one();
    }

    /// Locks the queue. No code panics while holding the lock, and jobs run without it, so a poisoned lock still holds
    /// a consistent state, which is used as it is.
    fn lock(&self) -> MutexGuard<'_, HostUploadQueue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Queued jobs and worker bookkeeping of [`HostUploadWorkersState`].
#[derive(Default)]
struct HostUploadQueue {
    /// Jobs whose sources are ready and that no worker has taken yet, in the order in which they were queued.
    jobs: VecDeque<Box<dyn FnOnce() + Send>>,

    /// Number of jobs that were accepted and have not completed (i.e., parked, queued, and running jobs).
    accepted_job_count: usize,

    /// Number of running workers.
    worker_count: usize,

    /// Whether the owner has been dropped, after which workers exit once no accepted job remains.
    shut_down: bool,
}

/// Fills the destination buffers of `uploads` on the current worker. For each upload in order, calls `prepare` with
/// the index of the upload to obtain its dense row-major host bytes and hands them to PJRT, which converts them to the
/// layout of the destination buffer. After the first failure, the job stops preparing bytes and sets the error that
/// `poison` derives from that failure on every destination buffer that has not received its bytes yet. Awaiting or
/// reading such a buffer then reports that error, so no reader of a destination buffer can wait forever. The source
/// fence passed to [`HostUploadWorkers::submit`] must cover everything that `prepare` waits for, apart from PJRT
/// transfers that complete on their own, so that the job never holds a worker while it waits for another job or for
/// an external producer (refer to [`HostUploadWorkers`] for why this guarantees progress). The job runs even when the
/// fence reports an error, and `prepare` then reports that error itself (e.g., from the readback of a failed buffer).
/// The job does not call back into PJRT from a PJRT callback. Transfer managers are [`Send`] but not [`Sync`], so the
/// job takes ownership of them, uses them only from its worker, and drops them once every issued transfer has
/// completed. Each transfer keeps its host bytes alive until PJRT has consumed them, and PJRT keeps each destination
/// buffer alive independently of its manager, so callers may drop the destination buffers at any time. If submission
/// fails or `prepare` panics, dropping the job's transfer managers makes PJRT poison the unfinished buffers.
pub(super) fn run_host_uploads<P, E>(uploads: Vec<PendingHostUpload>, mut prepare: P, poison: E)
where
    P: FnMut(usize) -> Result<HostBufferSlice, Error>,
    E: FnOnce(Error) -> PjrtError,
{
    // Issue every transfer before awaiting any of them, so that the uploads to different devices overlap.
    let mut events = Vec::with_capacity(uploads.len());
    let mut result = Ok(());
    for (index, upload) in uploads.iter().enumerate() {
        // The literal has the element type and dimensions of the managed buffer and a dense row-major layout, which
        // `transfer_literal` requires together with exactly the dense byte count of those dimensions. `prepare`
        // returns those bytes, which are empty for an empty shard and hold one byte per element for sub-byte types.
        let specification = BufferSpecification::new(upload.buffer_type, upload.dimensions.as_slice());
        let transfer =
            prepare(index).and_then(|bytes| Ok(upload.manager.transfer_literal(0, Arc::new(bytes), specification)?));
        match transfer {
            Ok(event) => events.push(event),
            Err(error) => {
                result = Err(error);
                break;
            }
        }
    }

    // Wait for the issued transfers before the transfer managers are dropped. A transfer that fails sets its error
    // on its own destination buffer, so its error needs no further handling here.
    let issued_count = events.len();
    for event in events {
        if let Err(error) = event.r#await() {
            drop(error);
        }
    }

    if let Err(error) = result {
        let error = poison(error);
        for upload in uploads.into_iter().skip(issued_count) {
            // The transfer manager rejects the error if the failed transfer of this upload was handed to PJRT before
            // it failed, in which case that transfer reports its own outcome. If PJRT rejects it for another reason,
            // dropping the manager still poisons the buffer, because PJRT poisons every buffer whose transfers have
            // not completed when their transfer manager is destroyed.
            if let Err(rejection) = upload.manager.set_error(0, error.clone()) {
                drop((rejection, upload));
            }
        }
    }
}

/// Validates that every device of `mesh` belongs to the process that the mesh assigns to it, as far as `client` can
/// tell. Each process uploads, copies, or receives only the shards of the devices that the mesh assigns to it, so
/// every array constructor (i.e., imports, uploads, and the placements onto a mesh) checks the devices of its mesh
/// here, whether or not it holds a buffer for them, and they all reject the same meshes:
///
///   - **Devices of This Process:** Every device that the mesh assigns to this process must be an addressable device
///     of `client`, because its shard would otherwise have no buffer in any process, also when this process supplies
///     no buffer for it (e.g., for a partial import of buffers or an array of the zero data type).
///   - **Devices of Other Processes:** A device that the mesh assigns to another process must not be a device that
///     `client` lists for a different process (e.g., device 0 of a single process client as `Device::new(0, 1)`),
///     which would otherwise silently receive no buffer in any process. A device that `client` does not list keeps the
///     process that the mesh assigns to it, because it can belong to a process whose devices `client` cannot see.
///
/// # Errors
///
/// Returns [`ArrayError::NonAddressableDevice`] for the first device that the mesh assigns to this process and that
/// `client` cannot address, [`ArrayError::DeviceProcessIndexMismatch`] for the first device that the mesh assigns to
/// another process and whose process index differs from the one that `client` reports, and the PJRT errors of
/// querying the devices of `client`.
pub(super) fn validate_mesh_device_processes(client: &Client<'_>, mesh: &DeviceMesh) -> Result<(), Error> {
    let process_index = client.process_index()?;
    if mesh.devices().iter().any(|device| device.process_index() == process_index) {
        let addressable_device_ids =
            client.addressable_devices()?.iter().map(|device| device.id()).collect::<Result<HashSet<_>, _>>()?;
        for device in mesh.devices().iter().filter(|device| device.process_index() == process_index) {
            if !addressable_device_ids.contains(&device.id()) {
                return Err(ArrayError::NonAddressableDevice { device_id: device.id(), process_index }.into());
            }
        }
    }

    let mut process_indices = None;
    for device in mesh.devices().iter().filter(|device| device.process_index() != process_index) {
        let process_indices = match &mut process_indices {
            Some(process_indices) => process_indices,
            None => process_indices.insert(
                client
                    .devices()?
                    .iter()
                    .map(|device| Ok((device.id()?, device.process_index()?)))
                    .collect::<Result<HashMap<_, _>, PjrtError>>()?,
            ),
        };

        if let Some(&actual) = process_indices.get(&device.id())
            && actual != device.process_index()
        {
            return Err(ArrayError::DeviceProcessIndexMismatch {
                device_id: device.id(),
                expected: device.process_index(),
                actual,
            }
            .into());
        }
    }

    Ok(())
}

/// Validates that `bytes`, which hold dense row-major elements of `data_type`, encode valid values of that data type,
/// as the constructors of a [`ryft_core::Array`] do with [`ArrayAddressing::validate_storage_bytes`]. Only the data
/// types whose valid encodings are a strict subset of their storage bytes can be invalid (i.e., [`DataType::Boolean`],
/// whose bytes must be `0` or `1`, and the data types narrower than a byte, each of whose elements occupies one byte
/// whose bits above its width must be zero), so the bytes of every other data type are not scanned.
///
/// # Errors
///
/// Returns the [`ProgramError`] of [`ArrayAddressing::validate_storage_bytes`], which names `data_type` and the
/// row-major index of the first invalid element within `bytes`.
fn validate_host_element_encodings(data_type: DataType, bytes: &[u8]) -> Result<(), Error> {
    if data_type == DataType::Boolean || (1..8).contains(&data_type.bit_width()) {
        ArrayAddressing::new(ArrayType::new_static(data_type, [bytes.len()]))?.validate_storage_bytes(bytes)?;
    }
    Ok(())
}

/// Returns row-major element strides for `shape`. `strides[i]` is the number of logical elements skipped when
/// incrementing dimension `i` by one in a dense major-to-minor layout. `element_type` is only used in the error
/// that reports a stride that overflows [`usize`].
fn row_major_element_strides(shape: &[usize], element_type: DataType) -> Result<Vec<usize>, Error> {
    let mut strides = vec![1usize; shape.len()];
    let mut stride = 1usize;
    for dimension in (0..shape.len()).rev() {
        strides[dimension] = stride;
        stride = stride.checked_mul(shape[dimension]).ok_or_else(|| Error::SizeLimitExceeded {
            message: format!(
                "row-major stride for array with shape `{:?}` and element type `{}` exceeds \
                 the maximum allowed size of {}",
                shape,
                element_type,
                usize::MAX,
            ),
        })?;
    }
    Ok(strides)
}

/// Recursively copies the row-major bytes of one tile into `global_bytes`, from dimension `dimension` onward and
/// starting at the element offsets `base_global_element_offset` and `base_shard_element_offset`. The strides are
/// row-major element strides. The tile must have at least one dimension.
#[allow(clippy::too_many_arguments)]
fn copy_dense_shard_bytes(
    shard_bytes: &[u8],
    shard_slices: &[Range<usize>],
    global_strides: &[usize],
    shard_strides: &[usize],
    dimension: usize,
    base_global_element_offset: usize,
    base_shard_element_offset: usize,
    element_size_in_bytes: usize,
    global_bytes: &mut [u8],
) {
    let slice = &shard_slices[dimension];
    if dimension + 1 == shard_slices.len() {
        let global_element_offset =
            base_global_element_offset + slice.start.checked_mul(global_strides[dimension]).unwrap();
        let global_byte_offset = global_element_offset.checked_mul(element_size_in_bytes).unwrap();
        let shard_byte_offset = base_shard_element_offset.checked_mul(element_size_in_bytes).unwrap();
        let byte_count = slice.len().checked_mul(element_size_in_bytes).unwrap();
        global_bytes[global_byte_offset..global_byte_offset + byte_count]
            .copy_from_slice(&shard_bytes[shard_byte_offset..shard_byte_offset + byte_count]);
        return;
    }

    for (local_index, global_index) in (slice.start..slice.end).enumerate() {
        let next_global_element_offset =
            base_global_element_offset + global_index.checked_mul(global_strides[dimension]).unwrap();
        let next_shard_element_offset =
            base_shard_element_offset + local_index.checked_mul(shard_strides[dimension]).unwrap();
        copy_dense_shard_bytes(
            shard_bytes,
            shard_slices,
            global_strides,
            shard_strides,
            dimension + 1,
            next_global_element_offset,
            next_shard_element_offset,
            element_size_in_bytes,
            global_bytes,
        );
    }
}

// Per-axis tile ranges are lists that often hold a single range, which this lint mistakes for a range to collect.
#[allow(clippy::single_range_in_vec_init)]
#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::thread;
    use std::time::{Duration, Instant};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_core::{
        Device, DeviceMesh, Dimension, Error as CoreError, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType,
        ProgramError, Shape, Sharding, ShardingDimension, StaticShape, StridedLayout, TiledLayout, TypeError,
    };
    use ryft_pjrt::{BufferSpecification, BufferType, Error as PjrtError, ExecutionFence};

    use crate::sessions::tests::{domain_mesh, replicated_vector_type};
    use crate::tests::{cpu_client, execution_client, logical_mesh_2x2, values_from_bytes, values_to_bytes};
    use crate::{ExecutionError, FromPjrt};

    use super::*;

    /// Deadline of the bounded waits for the jobs of background upload workers.
    const WAIT_TIMEOUT: Duration = Duration::from_secs(60);

    #[test]
    fn test_array_from_host_buffer() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let bytes = values_to_bytes::<f32>(&[1.0, 2.0]);

        // Without a configured mesh or sharding, the first addressable device forms the default mesh, whose logical
        // mesh is the single device mesh.
        let default_mesh = domain_mesh(client, "device", 1);
        let array = Array::from_host_buffer(&session, r#type.clone(), None, None, bytes.as_slice()).unwrap();
        assert_eq!(array.r#type().into_owned(), replicated_vector_type(&default_mesh, 2));
        assert_eq!(array.mesh(), &default_mesh);
        assert!(array.session().is_same_session(&session));
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0]);

        // Without a configured mesh, the type's sharding supplies the logical mesh.
        let inferred_mesh = domain_mesh(client, "y", 1);
        let sharded_type = replicated_vector_type(&inferred_mesh, 2);
        let array = Array::from_host_buffer(&session, sharded_type.clone(), None, None, bytes.as_slice()).unwrap();
        assert_eq!(array.r#type().into_owned(), sharded_type);
        assert_eq!(array.mesh(), &inferred_mesh);
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0]);

        // An explicit mesh takes precedence over the session's configured mesh.
        let configured_mesh = domain_mesh(client, "z", 1);
        let configured_session = session.with_device_mesh(configured_mesh.clone());
        let array = Array::from_host_buffer(&configured_session, r#type.clone(), None, None, bytes.as_slice()).unwrap();
        assert_eq!(array.r#type().into_owned(), replicated_vector_type(&configured_mesh, 2));
        assert_eq!(array.mesh(), &configured_mesh);
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0]);
        let array =
            Array::from_host_buffer(&configured_session, r#type.clone(), default_mesh.clone(), None, bytes.as_slice())
                .unwrap();
        assert_eq!(array.mesh(), &default_mesh);

        // The host buffer must hold exactly the dense bytes of the array.
        let error = Array::from_host_buffer(&session, r#type.clone(), None, None, &bytes[..4]).unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error)
                if matches!(error.as_ref(), ArrayError::HostBufferByteCountMismatch { expected: 8, actual: 4 }),
        ));
        assert_eq!(error.to_string(), "host buffer needs 8 byte(s) but got 4");

        // A sharding over more devices than the client has cannot be inferred.
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let r#type = r#type.with_sharding(Sharding::replicated(logical_mesh.clone(), 1)).unwrap();
        assert!(matches!(
            Array::from_host_buffer(&session, r#type, None, None, bytes.as_slice()),
            Err(Error::Core(error))
                if *error == CoreError::Program(ProgramError::InvalidArgument {
                    message: format!(
                        "eager XLA placement of a value without array inputs over mesh {logical_mesh:?} requires 2 \
                         addressable device(s) but the client only has 1",
                    ),
                }),
        ));
    }

    #[allow(deprecated)] // PJRT currently exposes actual buffer layout only through this query.
    #[test]
    fn test_array_from_host_buffer_layout() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F64, [2, 3]).with_layout(layout.clone()),
            mesh,
            None,
            values_to_bytes(&[1f64, 2., 3., 4., 5., 6.]),
        )
        .unwrap();

        // The device buffer is stored in the layout of the type, and reading it back restores the row-major order.
        assert_eq!(array.layout(), Some(&layout));
        assert_eq!(
            array.addressable_shards().next().unwrap().buffer().unwrap().layout(),
            Ok(layout.to_pjrt().unwrap()),
        );
        assert_eq!(array.copy_to_host().unwrap().elements::<f64>().unwrap(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn test_array_from_host_buffer_memory() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap()[0].clone();
        let mesh = domain_mesh(client, "x", 1);
        let values = values_to_bytes::<i64>(&[i64::MIN, i64::MAX]);

        // Verify physical placement and exact contents for every public memory variant. Exercise both scalar and
        // nonscalar uploads because their host buffer paths use different stride handling.
        for memory in [Memory::Device, Memory::Host { pinned: true }, Memory::Host { pinned: false }] {
            for shape in [Shape::new(vec![]), Shape::new(vec![Dimension::Static(2)])] {
                let count = if shape.rank() == 0 { 1 } else { 2 };
                let r#type = ArrayType::new(DataType::I64, shape).with_memory(memory);
                let array =
                    Array::from_host_buffer(&session, r#type, mesh.clone(), None, &values[..count * 8]).unwrap();
                assert_eq!(array.r#type().memory(), memory);
                let buffer = array.addressable_shards().next().unwrap().buffer().unwrap();
                let actual = buffer.memory().unwrap();
                match memory {
                    Memory::Device => assert_eq!(actual, device.default_memory().unwrap()),
                    Memory::Host { pinned: true } => assert_eq!(actual.kind().unwrap(), "pinned_host"),
                    Memory::Host { pinned: false } => assert_eq!(actual.kind().unwrap(), "unpinned_host"),
                }
                assert_eq!(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice(), &values[..count * 8]);
            }
        }
    }

    #[test]
    fn test_array_from_host_buffer_unreduced() {
        // Along the unreduced axis `x`, the devices hold summands of the logical value, and `y` tiles its only
        // dimension. A host upload places the whole value on the devices at coordinate zero along `x` and negative
        // zeros on the others, and host materialization sums the summands of each tile.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let sharding = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["y"])])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let device_values = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| {
                    let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                    values_from_bytes::<f32>(bytes.as_slice())
                })
                .collect::<Vec<_>>()
        };
        let materialize = |array: &Array| array.copy_to_host().unwrap().elements::<f32>().unwrap();

        let uploaded =
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap();
        assert_eq!(device_values(&uploaded), vec![vec![1., 2.], vec![3., 4.], vec![0., 0.], vec![0., 0.]]);
        assert_eq!(materialize(&uploaded), vec![1., 2., 3., 4.]);

        let buffers = [[1f32, 2.], [3., 4.], [10., 20.], [30., 40.]]
            .iter()
            .zip(&devices)
            .map(|(values, device)| {
                client
                    .buffer(values_to_bytes(values).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let summands = Array::from_addressable_buffers(&session, r#type, mesh.clone(), None, buffers).unwrap();
        assert_eq!(materialize(&summands), vec![11., 22., 33., 44.]);

        // With kind `max` or `min`, the devices along `x` hold candidates whose maximum or minimum, respectively, is
        // the logical value. A host upload places the whole value on every device, which is its own maximum, and host
        // materialization takes the elementwise maximum of the candidates of each tile.
        for (kind, expected) in [(UnreducedKind::Max, [10., 2., 30., 40.]), (UnreducedKind::Min, [1., -20., 3., 4.])] {
            let r#type = ArrayType::new_static(DataType::F32, [4])
                .with_sharding(sharding.clone().with_unreduced_kind(kind).unwrap())
                .unwrap();
            let uploaded = Array::from_host_buffer(
                &session,
                r#type.clone(),
                mesh.clone(),
                None,
                values_to_bytes(&[-1f32, -2., 3., 4.]),
            )
            .unwrap();
            assert_eq!(device_values(&uploaded), vec![vec![-1., -2.], vec![3., 4.], vec![-1., -2.], vec![3., 4.]]);
            assert_eq!(materialize(&uploaded), vec![-1., -2., 3., 4.]);
            let buffers = [[1f32, 2.], [3., 4.], [10., -20.], [30., 40.]]
                .iter()
                .zip(&devices)
                .map(|(values, device)| {
                    client
                        .buffer(values_to_bytes(values).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                        .unwrap()
                })
                .collect();
            let candidates = Array::from_addressable_buffers(&session, r#type, mesh.clone(), None, buffers).unwrap();
            assert_eq!(materialize(&candidates), expected.to_vec());
        }
    }

    #[test]
    fn test_array_from_host_buffer_unreduced_additive_identity() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0).with_unreduced_axes(["x"]).unwrap();
        let upload = |data_type: DataType, bytes: Vec<u8>| {
            Array::from_host_buffer(&session, ArrayType::scalar(data_type), mesh.clone(), Some(sharding.clone()), bytes)
        };
        let upload_in_background = |data_type: DataType, bytes: Vec<u8>| {
            Array::from_host_buffer_in_background(
                &session,
                ArrayType::scalar(data_type),
                mesh.clone(),
                sharding.clone(),
                ExecutionFence::new(Vec::new()),
                move || Ok(bytes),
                |error| PjrtError::internal(error.to_string()),
            )
        };
        let device_bits =
            |array: &Array| array.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();

        // The summands that do not hold the host value hold negative zero, which is the additive identity that also
        // preserves the sign of a zero.
        let negative_zero = (-0f32).to_bits().to_ne_bytes().to_vec();
        for array in [
            upload(DataType::F32, negative_zero.clone()).unwrap(),
            upload_in_background(DataType::F32, negative_zero.clone()).unwrap().unwrap(),
        ] {
            assert_eq!(device_bits(&array), vec![negative_zero.clone(); 2]);
            assert_eq!(array.copy_to_host_bytes().unwrap(), negative_zero);
        }

        // Boolean summands combine with logical or, so the summands that do not hold the host value hold `false`.
        for array in [
            upload(DataType::Boolean, vec![1]).unwrap(),
            upload_in_background(DataType::Boolean, vec![1]).unwrap().unwrap(),
        ] {
            assert_eq!(device_bits(&array), vec![vec![1], vec![0]]);
            assert_eq!(array.copy_to_host_bytes().unwrap(), vec![1]);
        }

        // Every encoding of `F8E8M0FNU` is a nonzero number, so it cannot be split into summands.
        let data_type = DataType::F8E8M0FNU;
        let message = format!(
            "type `{data_type}` has no additive identity to split a host value into summands along unreduced axes"
        );
        let blocking = upload(data_type, vec![0]).map(|_| ());
        let background = upload_in_background(data_type, vec![0]).map(|_| ());
        for result in [blocking, background] {
            assert!(matches!(
                &result,
                Err(Error::Array(error))
                    if matches!(error.as_ref(), ArrayError::MissingUnreducedSumIdentity { data_type: actual }
                        if *actual == data_type),
            ));
            assert_eq!(result.unwrap_err().to_string(), message);
        }

        // No identity is written when the unreduced axis has size one or when every shard is empty, so such placements
        // of that type succeed.
        let singleton_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![mesh.devices()[0]],
        )
        .unwrap();
        let singleton =
            Sharding::replicated(singleton_mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let empty = Sharding::replicated(mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let cases = [
            (ArrayType::new_static(data_type, [2]), singleton_mesh.clone(), singleton, vec![0x7f, 0x80]),
            (ArrayType::new_static(data_type, [0]), mesh.clone(), empty, Vec::new()),
        ];
        for (r#type, mesh, sharding, bytes) in cases {
            let expected_shards = vec![bytes.clone(); mesh.devices().len()];
            let blocking =
                Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), Some(sharding.clone()), &bytes);
            let readback = bytes.clone();
            let background = Array::from_host_buffer_in_background(
                &session,
                r#type,
                mesh,
                sharding,
                ExecutionFence::new(Vec::new()),
                move || Ok(readback),
                |error| PjrtError::internal(error.to_string()),
            );
            for array in [blocking.unwrap(), background.unwrap().unwrap()] {
                assert_eq!(device_bits(&array), expected_shards);
                assert_eq!(array.copy_to_host_bytes().unwrap(), bytes);
            }
        }
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_from_host_buffer_zero_space() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let r#type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));

        // Zero-space arrays are created without device buffers and read back as no bytes.
        let array = Array::from_host_buffer(&session, r#type, mesh, None, []).unwrap();
        assert_eq!(array.data_type(), DataType::Zero);
        assert_eq!(array.size_in_bytes().unwrap(), 0);
        let shard = array.addressable_shards().next().unwrap();
        assert!(shard.buffer().is_none());
        assert_eq!(array, array.clone());
        assert_eq!(array.copy_to_host_bytes().unwrap(), Vec::<u8>::new());

        // So are zero-space arrays that have no elements.
        let zero_sized_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(0)]));
        let zero_sized = Array::from_host_buffer(&session, zero_sized_type, array.mesh().clone(), None, []).unwrap();
        assert_eq!(zero_sized.shape(), StaticShape::new(vec![0]));
        assert!(zero_sized.addressable_shards().next().unwrap().buffer().is_none());
    }

    #[test]
    fn test_array_from_host_buffer_zero_space_remote_shards() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote_device = Device::new(local_device.id() + 1, client.process_index().unwrap() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh, vec![local_device, remote_device]).unwrap();
        let r#type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));

        // Shards on devices of other processes are described without being addressable.
        let array = Array::from_host_buffer(&session, r#type, mesh, None, []).unwrap();
        assert_eq!(array.shards().len(), 2);
        assert!(array.shards()[0].is_addressable());
        assert!(array.shards()[0].buffer().is_none());
        assert!(!array.shards()[1].is_addressable());
        assert!(array.shards()[1].buffer().is_none());
    }

    #[test]
    fn test_array_from_host_buffer_rejects_misattributed_devices() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let (device_id, process_index) = (local_device.id(), local_device.process_index());
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = |device: Device| DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let r#type = ArrayType::scalar(DataType::F32);
        let bytes = values_to_bytes(&[7f32]);

        // The mesh assigns the device of this process to another process, so no process would upload its shard. Both
        // the blocking and the background upload reject it before they upload anything.
        let misattributed = mesh(Device::new(device_id, process_index + 1));
        let message =
            format!("device {device_id} reports process index {process_index}, but expected {}", process_index + 1);
        let errors = [
            Array::from_host_buffer(&session, r#type.clone(), misattributed.clone(), None, bytes.as_slice())
                .unwrap_err(),
            Array::from_host_buffer_in_background(
                &session,
                r#type.clone(),
                misattributed.clone(),
                Sharding::replicated(logical_mesh.clone(), 0),
                ExecutionFence::new(Vec::new()),
                move || Ok(values_to_bytes(&[7f32])),
                |error| PjrtError::internal(error.to_string()),
            )
            .err()
            .unwrap(),
        ];
        for error in errors {
            assert!(matches!(
                &error,
                Error::Array(error) if matches!(
                    error.as_ref(),
                    ArrayError::DeviceProcessIndexMismatch { device_id: id, expected, actual }
                        if *id == device_id && *expected == process_index + 1 && *actual == process_index,
                ),
            ));
            assert_eq!(error.to_string(), message);
        }
        session.synchronize().unwrap();

        // A device that the mesh assigns to this process must be one that the client addresses, because no process
        // would upload its shard otherwise. Every upload rejects it, including one of the zero data type, which
        // uploads no buffer at all.
        let missing = mesh(Device::new(device_id + 999, process_index));
        let message =
            format!("device {} is not addressable by the PJRT client in process {process_index}", device_id + 999);
        let errors = [
            Array::from_host_buffer(&session, r#type.clone(), missing.clone(), None, bytes.as_slice()).unwrap_err(),
            Array::from_host_buffer(&session, ArrayType::scalar(DataType::Zero), missing.clone(), None, [])
                .unwrap_err(),
            Array::from_host_shards(&session, r#type.clone(), missing.clone(), None, Vec::<(DeviceId, &[u8])>::new())
                .unwrap_err(),
            Array::from_host_buffer_in_background(
                &session,
                r#type.clone(),
                missing,
                Sharding::replicated(logical_mesh.clone(), 0),
                ExecutionFence::new(Vec::new()),
                move || Ok(values_to_bytes(&[7f32])),
                |error| PjrtError::internal(error.to_string()),
            )
            .err()
            .unwrap(),
        ];
        for error in errors {
            assert!(matches!(
                &error,
                Error::Array(error) if matches!(
                    error.as_ref(),
                    ArrayError::NonAddressableDevice { device_id: id, process_index: index }
                        if *id == device_id + 999 && *index == process_index,
                ),
            ));
            assert_eq!(error.to_string(), message);
        }
        session.synchronize().unwrap();

        // A device that the client does not know can belong to a process whose devices it cannot see, so its shard is
        // described without a buffer.
        let remote_device = Device::new(device_id + 1, process_index + 1);
        let remote = Array::from_host_buffer(&session, r#type, mesh(remote_device), None, bytes.as_slice()).unwrap();
        assert_eq!(remote.mesh().devices(), [remote_device]);
        assert_eq!(remote.addressable_shards().count(), 0);
        assert!(remote.shards()[0].buffer().is_none());
        assert!(remote.is_committed());
    }

    #[test]
    fn test_array_from_host_buffer_rejects_invalid_layout() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote_device = Device::new(local_device.id() + 1, local_device.process_index() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let local_mesh = DeviceMesh::new(logical_mesh.clone(), vec![local_device]).unwrap();
        let remote_mesh = DeviceMesh::new(logical_mesh.clone(), vec![remote_device]).unwrap();
        let rank_two = Layout::Strided(StridedLayout::new(vec![4, 4]));
        let message = "strided layout rank 2 does not match array rank 1";

        // A rank-2 layout cannot address a rank-1 shape. The blocking and the background upload reject it alike for
        // the zero data type, whose arrays have no buffer that would reject it, and on a process that holds no shard of
        // the array, so every process gives the same answer as the one that holds the shards.
        let cases = [
            (DataType::Zero, 2, local_mesh.clone(), Vec::new()),
            (DataType::F32, 2, local_mesh, values_to_bytes(&[1f32, 2.])),
            (DataType::F32, 2, remote_mesh.clone(), values_to_bytes(&[1f32, 2.])),
            (DataType::F32, 0, remote_mesh, Vec::new()),
        ];
        for (data_type, extent, mesh, bytes) in cases {
            let r#type = ArrayType::new_static(data_type, [extent]).with_layout(rank_two.clone());
            let blocking = Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, bytes.as_slice());
            let background = Array::from_host_buffer_in_background(
                &session,
                r#type,
                mesh,
                Sharding::replicated(logical_mesh.clone(), 1),
                ExecutionFence::new(Vec::new()),
                move || Ok(bytes),
                |error| PjrtError::internal(error.to_string()),
            );
            for error in [blocking.err().unwrap(), background.err().unwrap()] {
                assert!(matches!(
                    &error,
                    Error::Core(error) if **error == CoreError::Type(TypeError::invalid(message)),
                ));
                assert_eq!(error.to_string(), message);
            }
        }
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_from_host_buffer_rejects_invalid_element_encodings() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let invalid = |data_type: DataType, element: usize, byte: u8| {
            CoreError::Program(ProgramError::Type(TypeError::invalid(format!(
                "array element {element} has invalid `{data_type}` byte encoding [{byte}]",
            ))))
        };

        // A Boolean byte must be `0` or `1`, and the bits of a sub-byte element above its width must be zero, as for a
        // host array of the reference backend, because a typed readback of the array would fail otherwise. Every
        // process validates the whole host value, so all of them reject it alike.
        let cases = [
            (DataType::Boolean, vec![1, 2], 1, 2),
            (DataType::I1, vec![2, 0], 0, 2),
            (DataType::U1, vec![0, 2], 1, 2),
            (DataType::U4, vec![0x0f, 0xf7], 1, 0xf7),
            (DataType::F4E2M1FN, vec![0x10, 0x0f], 0, 0x10),
        ];
        for (data_type, bytes, element, byte) in cases {
            let r#type = ArrayType::new_static(data_type, [2]);
            assert!(matches!(
                Array::from_host_buffer(&session, r#type, mesh.clone(), None, bytes),
                Err(Error::Core(error)) if *error == invalid(data_type, element, byte),
            ));
        }

        // The background upload reads its bytes only once it runs, so its validation failure poisons the buffers of
        // the array that it already returned.
        let background = Array::from_host_buffer_in_background(
            &session,
            ArrayType::new_static(DataType::Boolean, [2]),
            mesh.clone(),
            Sharding::replicated(mesh.logical_mesh().clone(), 1),
            ExecutionFence::new(Vec::new()),
            || Ok(vec![1, 2]),
            |error| PjrtError::internal(error.to_string()),
        )
        .unwrap()
        .unwrap();
        let message = invalid(DataType::Boolean, 1, 2).to_string();
        assert!(matches!(
            background.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Internal { message: actual, .. })) if actual == message,
        ));

        // Valid encodings upload and read back unchanged, and the data types whose every byte pattern is valid accept
        // any byte.
        let cases = [
            (DataType::Boolean, vec![1, 0]),
            (DataType::I1, vec![1, 0]),
            (DataType::U4, vec![0x0f, 0x07]),
            (DataType::U8, vec![0xff, 0xf7]),
            (DataType::F8E4M3FN, vec![0xff, 0x80]),
        ];
        for (data_type, bytes) in cases {
            let r#type = ArrayType::new_static(data_type, [2]);
            let array = Array::from_host_buffer(&session, r#type, mesh.clone(), None, bytes.clone()).unwrap();
            assert_eq!(array.copy_to_host_bytes().unwrap(), bytes);
            assert_eq!(array.copy_to_host().unwrap().storage_bytes(), bytes.as_slice());
        }
        assert!(matches!(
            session.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if failures.iter().map(|(_, error)| error.to_string()).collect::<Vec<_>>() == [message.as_str()],
        ));
    }

    #[test]
    fn test_array_from_host_buffer_uneven_1d_partitioning() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let values = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(5)]));

        let array = Array::from_host_buffer(
            &session,
            r#type,
            None,
            Some(sharding),
            values_to_bytes::<f32>(values.as_slice()).as_slice(),
        )
        .unwrap();
        assert_eq!(array.mesh(), &mesh);

        // The first partition receives the extra element of the uneven dimension.
        assert_eq!(array.addressable_shards().count(), 2);
        assert!(array.shards().iter().all(|shard| shard.buffer().is_some()));
        assert_eq!(
            array
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>(),
            vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0]],
        );
        assert_eq!(
            array.sharding().visualize().unwrap().render(false),
            indoc! {"
                    ┌─────┬─────┐
                    │  0  │  1  │
                    └─────┴─────┘
                "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_from_host_buffer_2d_partitioning() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect();
        let mesh = DeviceMesh::new(logical_mesh_2x2(), devices).unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let values = (0..48).map(|value| value as f32).collect::<Vec<_>>();
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(6)]));

        let array = Array::from_host_buffer(
            &session,
            r#type,
            mesh,
            Some(sharding),
            values_to_bytes::<f32>(values.as_slice()).as_slice(),
        )
        .unwrap();
        assert_eq!(array.addressable_shards().count(), 4);
        assert!(array.shards().iter().all(|shard| shard.buffer().is_some()));
        assert_eq!(
            array.sharding().visualize().unwrap().render(false),
            indoc! {"
                    ┌─────┬─────┐
                    │     │     │
                    │  0  │  1  │
                    │     │     │
                    ├─────┼─────┤
                    │     │     │
                    │  2  │  3  │
                    │     │     │
                    └─────┴─────┘
                "}
            .trim_end(),
        );

        // Shards are ordered like the mesh devices, and each one holds its row-major `4x3` tile of the host value.
        assert_eq!(
            array
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>(),
            vec![
                vec![0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 12.0, 13.0, 14.0, 18.0, 19.0, 20.0],
                vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0, 15.0, 16.0, 17.0, 21.0, 22.0, 23.0],
                vec![24.0, 25.0, 26.0, 30.0, 31.0, 32.0, 36.0, 37.0, 38.0, 42.0, 43.0, 44.0],
                vec![27.0, 28.0, 29.0, 33.0, 34.0, 35.0, 39.0, 40.0, 41.0, 45.0, 46.0, 47.0],
            ],
        );
    }

    #[test]
    fn test_array_from_host_buffer_empty_shards() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();

        // The second column shard of an empty array starts past the end of its empty host buffer.
        let r#type = ArrayType::new_static(DataType::F32, [0, 4]);
        let array = Array::from_host_buffer(&session, r#type, mesh, Some(sharding.clone()), [0u8; 0]).unwrap();
        array.block_until_ready().unwrap();
        assert_eq!(array.sharding(), &sharding);
        assert_eq!(
            array.shards().iter().map(ArrayShard::shape).collect::<Vec<_>>(),
            vec![StaticShape::new(vec![0, 2]); 2],
        );
        assert_eq!(array.copy_to_host_bytes().unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn test_array_from_host_buffer_in_background() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let values = [0f32, 1., 2., 3., 4., 5., 6., 7.];
        let r#type = ArrayType::new_static(DataType::F32, [2, 4]);
        let upload = |sharding: &Sharding| {
            let bytes = values_to_bytes(&values);
            Array::from_host_buffer_in_background(
                &session,
                r#type.clone(),
                mesh.clone(),
                sharding.clone(),
                ExecutionFence::new(Vec::new()),
                move || Ok(bytes),
                |error| PjrtError::internal(error.to_string()),
            )
            .unwrap()
            .unwrap()
        };

        // Shards along the minor dimension are strided in the host buffer and are gathered before their upload.
        let sharded = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        let array = upload(&sharded);
        array.block_until_ready().unwrap();
        assert_eq!(array.sharding(), &sharded);
        assert_eq!(
            array
                .addressable_shards()
                .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>(),
            vec![vec![0., 1., 4., 5.], vec![2., 3., 6., 7.]],
        );
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), values);

        // Along an unreduced axis of kind sum, the device at coordinate zero receives the value and the other negative
        // zeros.
        let unreduced = Sharding::replicated(mesh.logical_mesh().clone(), 2).with_unreduced_axes(["x"]).unwrap();
        let array = upload(&unreduced);
        array.block_until_ready().unwrap();
        assert_eq!(
            array
                .addressable_shards()
                .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>(),
            vec![values.to_vec(), vec![0.; 8]],
        );
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), values);
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_from_host_buffer_in_background_empty_shards() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();

        // Each empty shard receives an empty literal, even though the second one starts past the end of the buffer.
        let array = Array::from_host_buffer_in_background(
            &session,
            ArrayType::new_static(DataType::F32, [0, 4]),
            mesh,
            sharding.clone(),
            ExecutionFence::new(Vec::new()),
            || Ok(Vec::new()),
            |error| PjrtError::internal(error.to_string()),
        )
        .unwrap()
        .unwrap();
        array.block_until_ready().unwrap();
        assert_eq!(array.sharding(), &sharding);
        assert_eq!(
            array.shards().iter().map(ArrayShard::shape).collect::<Vec<_>>(),
            vec![StaticShape::new(vec![0, 2]); 2],
        );
        assert_eq!(array.copy_to_host_bytes().unwrap(), Vec::<u8>::new());
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_from_host_buffer_in_background_poisons_shards_on_readback_failure() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let upload = |readback: Result<Vec<u8>, Error>| {
            Array::from_host_buffer_in_background(
                &session,
                ArrayType::new_static(DataType::F32, [4]),
                mesh.clone(),
                sharding.clone(),
                ExecutionFence::new(Vec::new()),
                move || readback,
                |error| PjrtError::aborted(format!("upload failed: {error}")),
            )
            .unwrap()
            .unwrap()
        };
        let shard_errors = |array: &Array| {
            array
                .addressable_shards()
                .map(|shard| shard.buffer().unwrap().ready().unwrap().r#await().unwrap_err())
                .collect::<Vec<_>>()
        };

        // A failed readback poisons every shard with the error that `poison` derives from it.
        let array = upload(Err(PjrtError::cancelled("readback failed").into()));
        for error in shard_errors(&array) {
            assert!(matches!(error, PjrtError::Aborted { message, .. } if message == "upload failed: readback failed"));
        }

        // So does a readback with the wrong number of bytes.
        let array = upload(Ok(vec![0u8; 3]));
        let message = "upload failed: host buffer needs 16 byte(s) but got 3";
        for error in shard_errors(&array) {
            assert!(matches!(error, PjrtError::Aborted { message: shard_message, .. } if shard_message == message));
        }
        assert!(matches!(
            array.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message: array_message, .. })) if array_message == message,
        ));

        // The session tracks the destination buffers of both arrays and reports their failures in order.
        assert!(matches!(
            session.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if failures.iter().map(|(_, error)| error.to_string()).collect::<Vec<_>>()
                    == ["upload failed: readback failed", message],
        ));
    }

    #[test]
    fn test_array_from_host_buffer_in_background_replicated() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 4);
        let values = [1f32, 2., 3.];
        let bytes = values_to_bytes(&values);

        // Every replica receives the whole host value from the allocation of the readback, which no replica copies
        // (refer to `test_prepared_host_shards_bytes` for the sharing itself).
        let array = Array::from_host_buffer_in_background(
            &session,
            ArrayType::new_static(DataType::F32, [3]),
            mesh.clone(),
            Sharding::replicated(mesh.logical_mesh().clone(), 1),
            ExecutionFence::new(Vec::new()),
            move || Ok(bytes),
            |error| PjrtError::internal(error.to_string()),
        )
        .unwrap()
        .unwrap();
        array.block_until_ready().unwrap();
        assert_eq!(
            array
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>(),
            vec![values.to_vec(); 4],
        );
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), values);
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_from_host_buffer_in_background_dependent_chain() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let upload = |previous: Array| {
            Array::from_host_buffer_in_background(
                &session,
                r#type.clone(),
                mesh.clone(),
                sharding.clone(),
                previous.host_readback_fence().unwrap(),
                move || {
                    let values = values_from_bytes::<f32>(&previous.copy_to_host_bytes()?);
                    Ok(values_to_bytes(&values.iter().map(|value| value + 1.).collect::<Vec<_>>()))
                },
                |error| PjrtError::internal(error.to_string()),
            )
            .unwrap()
            .unwrap()
        };

        // The chain starts from an array whose buffers transfer managers fill only once the whole chain has been
        // requested. Each upload reads back the array of the previous one and adds one to it, so it can start only once
        // the previous upload has completed. The chain is longer than the number of workers, so it completes only if
        // the uploads that wait for their predecessors do not hold the workers that those predecessors need.
        let devices = client.addressable_devices().unwrap();
        let managers = devices
            .iter()
            .map(|device| {
                let specification = BufferSpecification::new(BufferType::F32, [2u64]);
                client
                    .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let buffers = managers.iter().map(|manager| manager.retrieve_buffer(0).unwrap()).collect();
        let source =
            Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), Some(sharding.clone()), buffers);
        let mut array = source.unwrap();
        let chain_length = 3 * HostUploadWorkers::MAX_DEFAULT_WORKER_COUNT;
        for _ in 0..chain_length {
            array = upload(array);
        }
        for (manager, values) in managers.iter().zip([[0f32, 1.], [2., 3.]]) {
            manager.transfer_data(0, Arc::new(values_to_bytes(&values)), 0, true).unwrap().r#await().unwrap();
        }

        // Wait for the end of the chain with a deadline, so that a chain that stalls fails instead of hanging.
        let (sender, receiver) = mpsc::channel();
        let chain_session = session.clone();
        thread::spawn(move || {
            let values = array.copy_to_host().unwrap().elements::<f32>().unwrap();
            chain_session.synchronize().unwrap();
            sender.send(values).unwrap();
        });
        let offset = chain_length as f32;
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Ok(vec![offset, 1. + offset, 2. + offset, 3. + offset]));
    }

    #[test]
    fn test_array_from_host_buffer_in_background_without_local_devices() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote_device = Device::new(local_device.id() + 1, local_device.process_index() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![remote_device]).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 1);

        // No device of the mesh belongs to this process, so there is nothing to upload. No job retains the readback
        // (and whatever it reads) until the sources complete, and the result waits for and reports those sources.
        let (event, producer) = client.event(()).unwrap();
        let retained = Arc::new(());
        let readback_retained = Arc::clone(&retained);
        let placed = Array::from_host_buffer_in_background(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            mesh.clone(),
            sharding.clone(),
            ExecutionFence::new(vec![event]),
            move || -> Result<Vec<u8>, Error> {
                let _retained = readback_retained;
                panic!("a process without destination devices must not read the value back")
            },
            |error| PjrtError::aborted(format!("upload failed: {error}")),
        )
        .unwrap()
        .unwrap();
        assert_eq!(Arc::strong_count(&retained), 1);
        assert_eq!(placed.mesh(), &mesh);
        assert_eq!(placed.sharding(), &sharding);
        assert_eq!(placed.addressable_shards().count(), 0);
        assert_eq!(placed.execution_fence().unwrap().is_ready(), Ok(false));
        producer.set(Some(PjrtError::aborted("producer failed"))).unwrap();
        assert!(matches!(
            placed.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "producer failed",
        ));
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_from_host_buffer_in_background_pending_sources() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let devices = client.addressable_devices().unwrap();
        let memory = devices[0].default_memory().unwrap();
        let upload = |source: Array| {
            Array::from_host_buffer_in_background(
                &session,
                r#type.clone(),
                mesh.clone(),
                sharding.clone(),
                source.host_readback_fence().unwrap(),
                move || source.copy_to_host_bytes(),
                |error| PjrtError::aborted(format!("upload failed: {error}")),
            )
            .unwrap()
            .unwrap()
        };
        let pending_source = || {
            let specification = BufferSpecification::new(BufferType::F32, [2u64]);
            let manager = client.host_to_device_transfer_manager(vec![specification], memory.clone()).unwrap();
            let buffers = vec![manager.retrieve_buffer(0).unwrap()];
            let source = Array::from_addressable_buffers(
                &session,
                r#type.clone(),
                mesh.clone(),
                Some(sharding.clone()),
                buffers,
            );
            (manager, source.unwrap())
        };

        // Each of the first uploads reads back a source whose buffer a transfer manager fills only later, and there are
        // as many of them as the workers of the session can run at once.
        let mut managers = Vec::new();
        let mut pending = Vec::new();
        for _ in 0..HostUploadWorkers::MAX_DEFAULT_WORKER_COUNT {
            let (manager, source) = pending_source();
            pending.push(upload(source));
            managers.push(manager);
        }

        // An upload whose source is ready completes while the earlier sources are still pending, so that a producer
        // of those sources may depend on it.
        let bytes = values_to_bytes(&[1f32, 2.]);
        let ready = Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), Some(sharding.clone()), &bytes);
        let independent = upload(ready.unwrap());
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || sender.send(independent.copy_to_host_bytes().unwrap()).unwrap());
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Ok(bytes));
        for array in &pending {
            assert_eq!(array.execution_fence().unwrap().is_ready(), Ok(false));
        }

        // Supplying the pending sources completes the uploads that read them back.
        for (index, manager) in managers.iter().enumerate() {
            let bytes = values_to_bytes(&[index as f32, index as f32 + 1.]);
            manager.transfer_data(0, Arc::new(bytes), 0, true).unwrap().r#await().unwrap();
        }
        assert_eq!(
            pending
                .iter()
                .map(|array| array.copy_to_host().unwrap().elements::<f32>().unwrap())
                .collect::<Vec<_>>(),
            (0..HostUploadWorkers::MAX_DEFAULT_WORKER_COUNT)
                .map(|index| vec![index as f32, index as f32 + 1.])
                .collect::<Vec<_>>(),
        );

        // A source that fails releases its upload, which reports the error of the readback.
        let (manager, source) = pending_source();
        let failed = upload(source);
        manager.set_error(0, PjrtError::cancelled("source failed")).unwrap();
        assert!(matches!(
            failed.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "upload failed: source failed",
        ));

        // The session reports the failed upload and then the readback of the failed source, in the order in which it
        // started tracking them.
        let Err(Error::Execution(ExecutionError::SynchronizationFailures { failures })) = session.synchronize() else {
            panic!("expected synchronization failures");
        };
        assert_eq!(
            failures.iter().map(|(_, error)| error.to_string()).collect::<Vec<_>>(),
            vec!["upload failed: source failed", "source failed"],
        );
    }

    #[test]
    fn test_array_from_host_buffer_in_background_without_local_shards() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote_device = Device::new(local_device.id() + 1, local_device.process_index() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![remote_device]).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 1);
        let upload = |sources: ExecutionFence| {
            Array::from_host_buffer_in_background(
                &session,
                ArrayType::new_static(DataType::F32, [2]),
                mesh.clone(),
                sharding.clone(),
                sources,
                || Ok(values_to_bytes(&[1f32, 2.])),
                |error| PjrtError::internal(error.to_string()),
            )
            .unwrap()
            .unwrap()
        };

        // This process holds no destination shard, so no upload of its own can report the producer of the value. The
        // result waits for the sources instead, as the uploads of the processes that hold the shards do.
        let (event, producer) = client.event(()).unwrap();
        let array = upload(ExecutionFence::new(vec![event]));
        assert_eq!(array.mesh(), &mesh);
        assert_eq!(array.sharding(), &sharding);
        assert_eq!(array.addressable_shards().count(), 0);
        assert!(array.shards()[0].buffer().is_none());
        assert!(matches!(array.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))));
        producer.set(None).unwrap();
        assert_eq!(array.block_until_ready().map_err(|error| error.to_string()), Ok(()));
        assert!(matches!(
            array.copy_to_host_bytes(),
            Err(Error::Array(error)) if matches!(
                error.as_ref(),
                ArrayError::MissingShardBuffer { shard_index: 0, device_id } if *device_id == remote_device.id(),
            ),
        ));

        // A failed producer fails the readiness of the result.
        let (event, producer) = client.event(()).unwrap();
        let array = upload(ExecutionFence::new(vec![event]));
        producer.set(Some(PjrtError::aborted("producer failed"))).unwrap();
        assert!(matches!(
            array.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "producer failed",
        ));
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_from_host_shards() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mut devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let first = devices[0].id();
        let second = devices[1].id();
        let remote = Device::new(first.max(second) + 1, devices[0].process_index() + 1);
        devices.push(remote);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh, devices).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [5]).with_sharding(sharding).unwrap();
        let first_bytes = [1, 2].as_slice();
        let second_bytes = [3, 4].as_slice();

        // Host shards are matched to their devices by identifier, in any order, and shards on devices of other
        // processes need no bytes.
        let array = Array::from_host_shards(
            &session,
            r#type.clone(),
            mesh.clone(),
            None,
            [(second, second_bytes), (first, first_bytes)],
        )
        .unwrap();
        array.block_until_ready().unwrap();
        assert_eq!(array.shape(), StaticShape::new(vec![5]));
        assert_eq!(array.addressable_shards().count(), 2);
        assert_eq!(array.shards()[0].copy_to_host_bytes().unwrap(), first_bytes);
        assert_eq!(array.shards()[1].copy_to_host_bytes().unwrap(), second_bytes);
        assert!(!array.shards()[2].is_addressable());

        // Every addressable device needs exactly one host shard of the size of its tile.
        let missing =
            Array::from_host_shards(&session, r#type.clone(), mesh.clone(), None, [(first, first_bytes)]).unwrap_err();
        assert!(matches!(
            &missing,
            Error::Array(error)
                if matches!(error.as_ref(), ArrayError::MissingHostShard { device_id } if *device_id == second),
        ));
        assert_eq!(missing.to_string(), format!("missing host shard for device {second}"));

        let duplicate = Array::from_host_shards(
            &session,
            r#type.clone(),
            mesh.clone(),
            None,
            [(first, first_bytes), (first, first_bytes), (second, second_bytes)],
        )
        .unwrap_err();
        assert!(matches!(
            &duplicate,
            Error::Array(error) if matches!(
                error.as_ref(),
                ArrayError::MultipleHostShardsOnDevice { device_id } if *device_id == first,
            ),
        ));
        assert_eq!(duplicate.to_string(), format!("got multiple host shards for device {first}"));

        let wrong_size = Array::from_host_shards(
            &session,
            r#type.clone(),
            mesh.clone(),
            None,
            [(first, first_bytes), (second, &second_bytes[..1])],
        )
        .unwrap_err();
        assert!(matches!(
            &wrong_size,
            Error::Array(error) if matches!(
                error.as_ref(),
                ArrayError::HostShardByteCountMismatch { device_id, expected: 2, actual: 1 } if *device_id == second,
            ),
        ));
        assert_eq!(wrong_size.to_string(), format!("host shard on device {second} needs 2 byte(s) but got 1"));

        // Host shards can only be uploaded to devices of the mesh that this process can address.
        assert!(matches!(
            Array::from_host_shards(
                &session,
                r#type.clone(),
                mesh.clone(),
                None,
                [(first, first_bytes), (second, second_bytes), (remote.id(), [5].as_slice())],
            ),
            Err(Error::Array(error)) if matches!(error.as_ref(), ArrayError::NonAddressableDevice { device_id, .. }
                if *device_id == remote.id()),
        ));
        let absent = remote.id() + 1;
        let error = Array::from_host_shards(&session, r#type, mesh, None, [(absent, first_bytes)]).unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error)
                if matches!(error.as_ref(), ArrayError::DeviceNotInMesh { device_id } if *device_id == absent),
        ));
        assert_eq!(error.to_string(), format!("device {absent} is not present in the device mesh"));
    }

    #[test]
    fn test_array_from_host_shards_with_auto_axis() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [3]);
        let first = mesh.devices()[0].id();
        let second = mesh.devices()[1].id();

        // The type of the array records none of the placement over the auto axis, while its runtime sharding does.
        let array = Array::from_host_shards(
            &session,
            r#type,
            mesh,
            Some(sharding.clone()),
            [(first, Arc::<[u8]>::from([1, 2])), (second, Arc::<[u8]>::from([3]))],
        )
        .unwrap();
        array.block_until_ready().unwrap();
        assert_eq!(array.sharding(), &sharding);
        assert_eq!(array.r#type().sharding(), Some(&sharding.without_auto_axes()));
        assert_eq!(array.copy_to_host_bytes().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_array_from_host_shards_rejects_misattributed_devices() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let misattributed = Device::new(devices[1].id(), devices[1].process_index() + 1);
        let mesh = DeviceMesh::new(logical_mesh, vec![devices[0], misattributed]).unwrap();

        // The mesh assigns the second device of this process to another process, so this process would upload only
        // the shard of the first one, and no process would upload the shard of the second.
        let error = Array::from_host_shards(
            &session,
            ArrayType::new_static(DataType::U8, [1]),
            mesh,
            None,
            [(devices[0].id(), [7u8].as_slice())],
        )
        .unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(
                error.as_ref(),
                ArrayError::DeviceProcessIndexMismatch { device_id, expected, actual }
                    if *device_id == devices[1].id()
                        && *expected == devices[1].process_index() + 1
                        && *actual == devices[1].process_index(),
            ),
        ));
        assert_eq!(
            error.to_string(),
            format!(
                "device {} reports process index {}, but expected {}",
                devices[1].id(),
                devices[1].process_index(),
                devices[1].process_index() + 1,
            ),
        );
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_from_host_shards_rejects_invalid_layout() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote_device = Device::new(local_device.id() + 1, local_device.process_index() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let local_mesh = DeviceMesh::new(logical_mesh.clone(), vec![local_device]).unwrap();
        let remote_mesh = DeviceMesh::new(logical_mesh, vec![remote_device]).unwrap();
        let rank_two = Layout::Strided(StridedLayout::new(vec![4, 4]));
        let message = "strided layout rank 2 does not match array rank 1";

        // A rank-2 layout cannot address a rank-1 shape, which is rejected for the zero data type, whose shards take no
        // buffer, and on a process that holds no shard of the array, as on the process that holds them.
        let cases = [
            (DataType::Zero, local_mesh.clone(), vec![(local_device.id(), Arc::<[u8]>::from([]))]),
            (DataType::F32, local_mesh, vec![(local_device.id(), Arc::from(values_to_bytes(&[1f32, 2.])))]),
            (DataType::F32, remote_mesh, Vec::new()),
        ];
        for (data_type, mesh, shards) in cases {
            let r#type = ArrayType::new_static(data_type, [2]).with_layout(rank_two.clone());
            let error = Array::from_host_shards(&session, r#type, mesh, None, shards).unwrap_err();
            assert!(matches!(
                &error,
                Error::Core(error) if **error == CoreError::Type(TypeError::invalid(message)),
            ));
            assert_eq!(error.to_string(), message);
        }
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_from_host_shards_rejects_invalid_element_encodings() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices =
            client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // The element of a shard that does not encode a value of the data type is rejected before anything is uploaded,
        // and the error names its position within that shard.
        let cases = [
            (DataType::Boolean, [[1u8, 0], [0, 2]], 1, 2),
            (DataType::I1, [[3, 0], [0, 1]], 0, 3),
            (DataType::U4, [[0x0f, 0x07], [0xf7, 0x00]], 0, 0xf7),
        ];
        for (data_type, [first, second], element, byte) in cases {
            let r#type = ArrayType::new_static(data_type, [4]);
            let shards = [(devices[0], Arc::<[u8]>::from(first)), (devices[1], Arc::from(second))];
            assert!(matches!(
                Array::from_host_shards(&session, r#type, mesh.clone(), Some(sharding.clone()), shards),
                Err(Error::Core(error)) if *error == CoreError::Program(ProgramError::Type(TypeError::invalid(
                    format!("array element {element} has invalid `{data_type}` byte encoding [{byte}]"),
                ))),
            ));
        }
        session.synchronize().unwrap();

        // Valid encodings upload and read back unchanged.
        let r#type = ArrayType::new_static(DataType::U4, [4]);
        let shards = [(devices[0], Arc::<[u8]>::from([0x0f, 0x07])), (devices[1], Arc::from([0x00, 0x01]))];
        let array = Array::from_host_shards(&session, r#type, mesh, Some(sharding), shards).unwrap();
        assert_eq!(array.copy_to_host_bytes().unwrap(), [0x0f, 0x07, 0x00, 0x01]);
    }

    #[test]
    fn test_array_copy_to_host() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let bytes = values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]);
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [3, 2]),
            mesh,
            Some(sharding),
            bytes.as_slice(),
        )
        .unwrap();

        // The host array holds the whole value with the element type and the shape of the array but not its sharding.
        let host_array = array.copy_to_host().unwrap();
        assert_eq!(host_array, CoreArray::new(ArrayType::new_static(DataType::F32, [3, 2]), bytes).unwrap());
        assert_eq!(host_array.elements::<f32>().unwrap(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn test_array_copy_to_host_bytes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let values = values_to_bytes(&[1f32, 2., 3., 4.]);

        // Sharded and replicated arrays read back as the dense row-major bytes of their whole value.
        for sharding in [sharded.clone(), replicated.clone()] {
            let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
            let array = Array::from_host_buffer(&session, r#type, mesh.clone(), None, values.as_slice()).unwrap();
            assert_eq!(array.copy_to_host_bytes().unwrap(), values);
        }

        // An array with no elements produces no bytes.
        let r#type = ArrayType::new_static(DataType::F32, [0]).with_sharding(sharded.clone()).unwrap();
        let empty = Array::from_host_buffer(&session, r#type, mesh, None, []).unwrap();
        assert_eq!(empty.copy_to_host_bytes().unwrap(), Vec::<u8>::new());

        // When the second mesh device lives in another process, the current process holds a replicated value in full,
        // but only half of a sharded value and only one of the summands of an unreduced value.
        let remote_device =
            Device::new(devices.iter().map(Device::id).max().unwrap() + 1, client.process_index().unwrap() + 1);
        let remote_mesh = DeviceMesh::new(logical_mesh, vec![devices[0], remote_device]).unwrap();
        let with_remote_shard = |sharding: Sharding, local_values: &[f32]| {
            let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
            let dimensions = [local_values.len() as u64];
            let bytes = values_to_bytes(local_values);
            let buffer =
                client.buffer(bytes.as_slice(), BufferType::F32, dimensions, None, client_devices[0].clone(), None);
            Array::from_addressable_buffers(&session, r#type, remote_mesh.clone(), None, vec![buffer.unwrap()]).unwrap()
        };
        assert_eq!(with_remote_shard(replicated, &[1., 2., 3., 4.]).copy_to_host_bytes().unwrap(), values);
        for (sharding, local_values) in [(sharded, [1f32, 2.].as_slice()), (unreduced, [1f32, 2., 3., 4.].as_slice())] {
            let error = with_remote_shard(sharding, local_values).copy_to_host_bytes().unwrap_err();
            assert!(matches!(
                &error,
                Error::Array(error) if matches!(error.as_ref(),
                    ArrayError::MissingShardBuffer { shard_index: 1, device_id } if *device_id == remote_device.id(),
                ),
            ));
            assert_eq!(
                error.to_string(),
                format!("array shard #1 on device {} has no addressable buffer", remote_device.id()),
            );
        }
    }

    #[test]
    fn test_array_copy_to_host_bytes_empty_remote_shard() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote_device = Device::new(local_device.id() + 1, client.process_index().unwrap() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![local_device, remote_device]).unwrap();
        let sharding = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

        // A single element split over two devices leaves the remote shard without elements, so the local shard holds
        // the whole value.
        let r#type = ArrayType::new_static(DataType::U8, [1]);
        let array =
            Array::from_host_shards(&session, r#type, mesh, Some(sharding), [(local_device.id(), [7u8].as_slice())])
                .unwrap();
        assert_eq!(array.shards()[1].slice(), [1..1]);
        array.validate_host_readback().unwrap();
        assert_eq!(array.copy_to_host_bytes().unwrap(), vec![7]);
    }

    #[test]
    fn test_array_copy_to_host_bytes_empty_local_shard() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // A single element split over two devices leaves the second shard without elements. Its buffer stays pending
        // and then fails, but the first shard holds the whole value, so the readback neither waits for nor reports it.
        let first = client.buffer([7u8].as_slice(), BufferType::U8, [1u64], None, devices[0].clone(), None).unwrap();
        let specification = BufferSpecification::new(BufferType::U8, [0u64]);
        let manager = client
            .host_to_device_transfer_manager(vec![specification], devices[1].default_memory().unwrap())
            .unwrap();
        let empty = manager.retrieve_buffer(0).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [1]);
        let array =
            Array::from_addressable_buffers(&session, r#type, mesh, Some(sharding), vec![first, empty]).unwrap();
        assert_eq!(array.shards()[1].slice(), [1..1]);
        assert!(!array.shards()[1].buffer().unwrap().ready().unwrap().ready().unwrap());
        let (sender, receiver) = mpsc::channel();
        let readback = array.clone();
        let reader = thread::spawn(move || sender.send(readback.copy_to_host_bytes().unwrap()).unwrap());
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT).unwrap(), vec![7]);
        reader.join().unwrap();
        array.host_readback_fence().unwrap().block_until_ready().unwrap();

        // The failure of the empty buffer still surfaces through the readiness of the array, which waits for every
        // buffer, but not through the readback.
        manager.set_error(0, PjrtError::aborted("empty shard failed")).unwrap();
        assert_eq!(array.copy_to_host_bytes().unwrap(), vec![7]);
        assert!(matches!(
            array.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "empty shard failed",
        ));
    }

    #[test]
    fn test_array_copy_to_host_bytes_reads_one_replica_per_tile() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let shape = [4, 2];
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let rows = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
        let replicated = vec![ShardingDimension::replicated(), ShardingDimension::replicated()];

        // Leave one readable shard per tile. Any extra replica copy would fail the readback.
        for (dimensions, tile_count) in [(rows, 2), (replicated, 1)] {
            let sharding = Sharding::new(logical_mesh.clone(), dimensions).unwrap();
            let r#type = ArrayType::new_static(DataType::U8, shape).with_sharding(sharding).unwrap();
            let tile_shape = [(shape[0] / tile_count) as u64, shape[1] as u64];
            let replicas_per_tile = devices.len() / tile_count;
            let tiles = bytes.chunks_exact(bytes.len() / tile_count).collect::<Vec<_>>();
            let buffers = devices
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, device)| {
                    if index.is_multiple_of(replicas_per_tile) {
                        client.buffer(tiles[index / replicas_per_tile], BufferType::U8, tile_shape, None, device, None)
                    } else {
                        client.error_buffer(
                            PjrtError::aborted("unexpected replica copy"),
                            BufferSpecification::new(BufferType::U8, tile_shape),
                            device,
                        )
                    }
                    .unwrap()
                })
                .collect();
            let array = Array::from_addressable_buffers(&session, r#type, mesh.clone(), None, buffers).unwrap();
            assert_eq!(array.copy_to_host_bytes().unwrap(), bytes);
        }
    }

    #[test]
    fn test_array_copy_to_host_bytes_unreduced_replicas() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let sharding = Sharding::replicated(logical_mesh, 1).with_unreduced_axes(["x"]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding).unwrap();
        let readback = |values: [[f32; 2]; 4]| {
            let buffers = values
                .iter()
                .zip(&devices)
                .map(|(values, device)| {
                    client
                        .buffer(values_to_bytes(values).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                        .unwrap()
                })
                .collect();
            Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), None, buffers)
                .unwrap()
                .copy_to_host_bytes()
        };

        // The two shards at each coordinate along `x` are replicas, so each summand contributes once.
        let first = [1f32, 2.];
        let second = [10f32, 20.];
        assert_eq!(readback([first, first, second, second]).unwrap(), values_to_bytes(&[11f32, 22.]));

        // Only the first replica of each summand is read, so the other replicas are not compared with it.
        assert_eq!(readback([first, [1., 3.], second, [0., 0.]]).unwrap(), values_to_bytes(&[11f32, 22.]));
    }

    #[test]
    fn test_array_copy_to_host_bytes_unreduced_remote_replicas() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let remote_process_index = client.process_index().unwrap() + 1;
        let first_remote_id = devices.iter().map(Device::id).max().unwrap() + 1;
        let remote_devices = [
            Device::new(first_remote_id, remote_process_index),
            Device::new(first_remote_id + 1, remote_process_index),
        ];
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::replicated(logical_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding).unwrap();
        let array = |mesh_devices: Vec<Device>, local_values: [[f32; 2]; 2]| {
            let mesh = DeviceMesh::new(logical_mesh.clone(), mesh_devices).unwrap();
            let buffers = local_values
                .iter()
                .zip(&client_devices)
                .map(|(values, device)| {
                    client
                        .buffer(values_to_bytes(values).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                        .unwrap()
                })
                .collect();
            Array::from_addressable_buffers(&session, r#type.clone(), mesh, None, buffers).unwrap()
        };

        // The local devices hold one replica of each summand along `x`, and the other replicas live in another process,
        // so the current process still holds the whole value.
        let covered = array(vec![devices[0], remote_devices[0], devices[1], remote_devices[1]], [[1., 2.], [10., 20.]]);
        covered.validate_host_readback().unwrap();
        assert_eq!(covered.copy_to_host_bytes().unwrap(), values_to_bytes(&[11f32, 22.]));

        // When both local devices hold replicas of the first summand, the second one is missing.
        let uncovered = array(vec![devices[0], devices[1], remote_devices[0], remote_devices[1]], [[1., 2.], [1., 2.]]);
        for error in [uncovered.validate_host_readback().unwrap_err(), uncovered.copy_to_host_bytes().unwrap_err()] {
            assert!(matches!(
                &error,
                Error::Array(error) if matches!(error.as_ref(),
                    ArrayError::MissingShardBuffer { shard_index: 2, device_id } if *device_id == first_remote_id,
                ),
            ));
            assert_eq!(
                error.to_string(),
                format!("array shard #2 on device {first_remote_id} has no addressable buffer"),
            );
        }
    }

    #[test]
    fn test_array_copy_to_host_bytes_unreduced_canonical_order() {
        let client = cpu_client(3);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let local_devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let remote_process_index = client.process_index().unwrap() + 1;
        let first_remote_id = local_devices.iter().map(Device::id).max().unwrap() + 1;
        let remote_devices =
            (0..3).map(|index| Device::new(first_remote_id + index, remote_process_index)).collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("r", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("x", 3, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::replicated(logical_mesh.clone(), 0).with_unreduced_axes(["x"]).unwrap();
        let summands = [1e20f32, -1e20, 1.];

        // Every process holds one replica of each summand along `x`, but the processes hold replicas at different
        // coordinates along `r`. Both orders of the local replicas sum the summands in the order of their coordinates
        // along `x`, so every process reads back the same value, even though floating point addition does not
        // associate (i.e., `(1e20 + 1) - 1e20` is zero).
        let readback = |local_replica: usize| {
            let mut local = local_devices.iter();
            let mut remote = remote_devices.iter();
            let mut shards = Vec::new();
            let mesh_devices = (0..6)
                .map(|index| {
                    if index % 2 == local_replica {
                        let device = *local.next().unwrap();
                        shards.push((device.id(), summands[index % 3].to_ne_bytes().to_vec()));
                        device
                    } else {
                        *remote.next().unwrap()
                    }
                })
                .collect();
            let mesh = DeviceMesh::new(logical_mesh.clone(), mesh_devices).unwrap();
            let r#type = ArrayType::scalar(DataType::F32);
            let shards = shards.iter().map(|(device_id, bytes)| (*device_id, bytes.as_slice()));
            let array = Array::from_host_shards(&session, r#type, mesh, Some(sharding.clone()), shards).unwrap();
            array.copy_to_host_bytes().unwrap()
        };
        assert_eq!(readback(0), 1f32.to_ne_bytes());
        assert_eq!(readback(1), 1f32.to_ne_bytes());
    }

    #[test]
    fn test_array_copy_to_host_bytes_unreduced_boolean() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let readback = |first: [u8; 2], second: [u8; 2]| {
            let shards = [(mesh.devices()[0].id(), first.as_slice()), (mesh.devices()[1].id(), second.as_slice())];
            let r#type = ArrayType::new_static(DataType::Boolean, [2]);
            Array::from_host_shards(&session, r#type, mesh.clone(), Some(sharding.clone()), shards)
                .unwrap()
                .copy_to_host_bytes()
                .unwrap()
        };

        // Boolean summands combine with logical or, whose identity is `false`.
        assert_eq!(readback([1, 0], [0, 0]), vec![1, 0]);
        assert_eq!(readback([1, 0], [1, 1]), vec![1, 1]);
    }

    #[test]
    fn test_array_copy_to_host_bytes_rejects_varying_manual_axes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        let upload = |sharding: Sharding, shape: [usize; 1]| {
            let shards = [(devices[0].id(), [1u8].as_slice()), (devices[1].id(), [2u8].as_slice())];
            let r#type = ArrayType::new_static(DataType::U8, shape);
            Array::from_host_shards(&session, r#type, mesh.clone(), Some(sharding), shards).unwrap()
        };

        // The two devices hold different values of the same slice, which one dense host value cannot represent, so
        // the readback is rejected before any copy, while each shard stays readable on its own.
        let varying = Sharding::replicated(logical_mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap();
        let array = upload(varying.clone(), [1]);
        assert_eq!(array.sharding(), &varying);
        assert_eq!(
            array.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>(),
            [[1], [2]],
        );
        let message = "cannot read back an array that varies along the manual mesh axes `m` as one host value, because \
                       shards #0 and #1 can hold different values of the same slice";
        for error in [array.copy_to_host_bytes().unwrap_err(), array.validate_host_readback().unwrap_err()] {
            assert!(matches!(
                &error,
                Error::Core(error) if **error == CoreError::Program(ProgramError::InvalidArgument {
                    message: message.to_string(),
                }),
            ));
            assert_eq!(error.to_string(), message);
        }
        assert_eq!(array.host_readback_fence().err().map(|error| error.to_string()), Some(message.to_string()));

        // Shards that vary along a manual axis that splits the array hold distinct slices, so they read back densely.
        let split = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["m"])])
            .unwrap()
            .with_varying_manual_axes(["m"])
            .unwrap();
        assert_eq!(upload(split, [2]).copy_to_host_bytes().unwrap(), [1, 2]);
    }

    #[test]
    fn test_array_copy_to_host_bytes_preserves_producer_errors_without_payload() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);

        // Arrays without elements and zero-space arrays issue no copies, and the failure of the execution that
        // produced them is still returned.
        for (data_type, shape) in [(DataType::F32, [0]), (DataType::Zero, [2])] {
            let (event, promise) = client.event(()).unwrap();
            promise.set(Some(PjrtError::invalid_argument("empty producer failed"))).unwrap();
            let array =
                Array::from_host_buffer(&session, ArrayType::new_static(data_type, shape), mesh.clone(), None, [])
                    .unwrap()
                    .with_execution_fence(ExecutionFence::new(vec![event]));
            assert_eq!(array.copy_to_host_bytes().unwrap_err().to_string(), "empty producer failed");
        }
    }

    #[test]
    fn test_array_validate_host_readback() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let client_device = client.addressable_devices().unwrap().remove(0);
        let local_device = Device::from_pjrt(&client_device).unwrap();
        let remote_device = Device::new(local_device.id() + 1, client.process_index().unwrap() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![local_device, remote_device]).unwrap();
        let array = |sharding: Sharding, local_values: &[f32]| {
            let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
            let dimensions = [local_values.len() as u64];
            let bytes = values_to_bytes(local_values);
            let buffer =
                client.buffer(bytes.as_slice(), BufferType::F32, dimensions, None, client_device.clone(), None);
            Array::from_addressable_buffers(&session, r#type, mesh.clone(), None, vec![buffer.unwrap()]).unwrap()
        };

        // The local replica of a replicated array holds the whole value.
        let replicated = array(Sharding::replicated(logical_mesh.clone(), 1), &[1., 2., 3., 4.]);
        assert!(matches!(replicated.validate_host_readback(), Ok(())));

        // The local shard of a sharded array holds only its own tile, so the readback is rejected before any copy.
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let error = array(sharded, &[1., 2.]).validate_host_readback().unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::MissingShardBuffer { shard_index: 1, device_id } if *device_id == remote_device.id(),
            ),
        ));
        assert_eq!(
            error.to_string(),
            format!("array shard #1 on device {} has no addressable buffer", remote_device.id()),
        );

        // A readback without elements copies nothing, so it needs no shard.
        let zero = Array::zero(&session, ArrayType::new_static(DataType::Zero, [4]), None, mesh).unwrap();
        assert!(matches!(zero.validate_host_readback(), Ok(())));
    }

    #[test]
    fn test_array_host_readback_fence() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let bytes = values_to_bytes(&[1f32, 2.]);
        let pending_array = |sharding: Sharding, dimensions: u64| {
            let managers = client
                .addressable_devices()
                .unwrap()
                .iter()
                .map(|device| {
                    let specification = BufferSpecification::new(BufferType::F32, [dimensions]);
                    client
                        .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let buffers = managers.iter().map(|manager| manager.retrieve_buffer(0).unwrap()).collect();
            let array =
                Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), Some(sharding), buffers);
            (array.unwrap(), managers)
        };
        let wait = |fence: ExecutionFence| {
            let (sender, receiver) = mpsc::channel();
            thread::spawn(move || sender.send(fence.block_until_ready()).unwrap());
            receiver.recv_timeout(WAIT_TIMEOUT)
        };

        // A readback of a replicated array reads only its first replica, so the fence does not wait for the second.
        let (replicated, managers) = pending_array(Sharding::replicated(mesh.logical_mesh().clone(), 1), 2);
        assert_eq!(replicated.host_readback_fence().unwrap().is_ready(), Ok(false));
        managers[0].transfer_data(0, Arc::new(bytes.clone()), 0, true).unwrap().r#await().unwrap();
        assert_eq!(wait(replicated.host_readback_fence().unwrap()), Ok(Ok(())));
        assert_eq!(replicated.copy_to_host_bytes().unwrap(), bytes);

        // The fence still waits for the execution that produced the array.
        let (event, promise) = client.event(()).unwrap();
        let produced = replicated.with_execution_fence(ExecutionFence::new(vec![event]));
        assert_eq!(produced.host_readback_fence().unwrap().is_ready(), Ok(false));
        promise.set(None).unwrap();
        assert_eq!(wait(produced.host_readback_fence().unwrap()), Ok(Ok(())));
        managers[1].set_error(0, PjrtError::aborted("unused replica failed")).unwrap();

        // A readback of a sharded array reads every tile, so the fence waits for both shards.
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (sharded, managers) = pending_array(sharded, 1);
        managers[0].transfer_data(0, Arc::new(bytes[..4].to_vec()), 0, true).unwrap().r#await().unwrap();
        assert_eq!(sharded.host_readback_fence().unwrap().is_ready(), Ok(false));
        managers[1].transfer_data(0, Arc::new(bytes[4..].to_vec()), 0, true).unwrap().r#await().unwrap();
        assert_eq!(wait(sharded.host_readback_fence().unwrap()), Ok(Ok(())));
        assert_eq!(sharded.copy_to_host_bytes().unwrap(), bytes);
    }

    #[test]
    fn test_array_host_readback_fence_column_replicas() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let managers = devices
            .iter()
            .map(|device| {
                let specification = BufferSpecification::new(BufferType::U8, [2, 1]);
                client
                    .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let buffers = managers.iter().map(|manager| manager.retrieve_buffer(0).unwrap()).collect();
        let sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [2, 2]);
        let array = Array::from_addressable_buffers(&session, r#type, mesh, Some(sharding), buffers).unwrap();

        // The columns are not contiguous in the dense representation, and the devices along `y` hold replicas of
        // each column. The readback reads only the first replica of each column, so the pending second replicas
        // block neither the fence nor the readback.
        managers[0].transfer_data(0, Arc::new(vec![1, 3]), 0, true).unwrap().r#await().unwrap();
        managers[2].transfer_data(0, Arc::new(vec![2, 4]), 0, true).unwrap().r#await().unwrap();
        let fence = array.host_readback_fence().unwrap();
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || sender.send(fence.block_until_ready()).unwrap());
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Ok(Ok(())));
        assert_eq!(array.copy_to_host_bytes().unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_array_host_readback_fence_unreduced_replicas() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let managers = devices
            .iter()
            .map(|device| {
                let specification = BufferSpecification::new(BufferType::F32, [2]);
                client
                    .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let buffers = managers.iter().map(|manager| manager.retrieve_buffer(0).unwrap()).collect();
        let sharding = Sharding::replicated(logical_mesh, 1).with_unreduced_axes(["x"]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let array = Array::from_addressable_buffers(&session, r#type, mesh, Some(sharding), buffers).unwrap();

        // The devices along `y` hold replicas of the summand at their coordinate along `x`. The readback reads only
        // the first replica of each summand, so the pending second replicas block neither the fence nor the readback.
        let first = values_to_bytes(&[1f32, 2.]);
        let second = values_to_bytes(&[10f32, 20.]);
        managers[0].transfer_data(0, Arc::new(first), 0, true).unwrap().r#await().unwrap();
        managers[2].transfer_data(0, Arc::new(second), 0, true).unwrap().r#await().unwrap();
        let fence = array.host_readback_fence().unwrap();
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || sender.send(fence.block_until_ready()).unwrap());
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Ok(Ok(())));
        assert_eq!(array.copy_to_host_bytes().unwrap(), values_to_bytes(&[11f32, 22.]));
    }

    #[test]
    fn test_array_host_readback_shards() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let selection = |dimensions: Vec<ShardingDimension>, unreduced: bool| {
            let mut sharding = Sharding::new(logical_mesh.clone(), dimensions).unwrap();
            if unreduced {
                sharding = sharding.with_unreduced_axes(["x"]).unwrap();
            }
            let r#type = ArrayType::new_static(DataType::U8, [2, 2]);
            let array =
                Array::from_host_buffer(&session, r#type, mesh.clone(), Some(sharding), [1u8, 2, 3, 4]).unwrap();
            let (shards, direct) = array.host_readback_shards(array.unreduced_axis_indices().as_slice()).unwrap();
            (shards.into_iter().map(ArrayShard::index).collect::<Vec<_>>(), direct)
        };
        let replicated = vec![ShardingDimension::replicated(), ShardingDimension::replicated()];
        let rows = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
        let columns = vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])];

        // The first replica of each tile is selected, and tiles that are contiguous in the dense representation are
        // copied directly into it.
        assert_eq!(selection(replicated.clone(), false), (vec![0], true));
        assert_eq!(selection(rows, false), (vec![0, 2], true));

        // Column tiles are not contiguous, so their copies are assembled on the host.
        assert_eq!(selection(columns, false), (vec![0, 2], false));

        // Along the unreduced axis `x`, the devices at each coordinate hold replicas of one contribution, and the
        // selected contributions are combined on the host.
        assert_eq!(selection(replicated, true), (vec![0, 2], false));
    }

    #[test]
    fn test_array_unreduced_axis_indices() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let indices = |unreduced_axes: &[&str]| {
            let sharding = Sharding::replicated(logical_mesh.clone(), 1)
                .with_unreduced_axes(unreduced_axes.iter().copied())
                .unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [1]);
            let array = Array::from_host_buffer(&session, r#type, mesh.clone(), Some(sharding), 1f32.to_ne_bytes());
            array.unwrap().unreduced_axis_indices()
        };

        // The indices are those of the unreduced axes within the mesh, in mesh order.
        assert_eq!(indices(&[]), Vec::<usize>::new());
        assert_eq!(indices(&["y"]), vec![1]);
        assert_eq!(indices(&["y", "x"]), vec![0, 1]);
    }

    #[test]
    fn test_array_unreduced_coordinates() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [1]);
        let sharding = Sharding::replicated(logical_mesh, 1);
        let array = Array::from_host_buffer(&session, r#type, mesh, Some(sharding), 1f32.to_ne_bytes()).unwrap();
        let coordinates = |unreduced_axis_indices: &[usize]| {
            array
                .shards()
                .iter()
                .map(|shard| array.unreduced_coordinates(shard, unreduced_axis_indices))
                .collect::<Vec<_>>()
        };

        // The coordinates of the device of each shard are restricted to the requested mesh axes, in their order.
        assert_eq!(coordinates(&[0, 1]), vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
        assert_eq!(coordinates(&[1]), vec![vec![0], vec![1], vec![0], vec![1]]);
        assert_eq!(coordinates(&[]), vec![Vec::<usize>::new(); 4]);
    }

    #[test]
    fn test_host_shard_source_to_dense_bytes() {
        let host_bytes = [0u8, 1, 2, 3, 4, 5, 6, 7];

        // An additive identity repeats the encoding of one element and reads nothing from the host buffer.
        let identity = HostShardSource::AdditiveIdentity { element: vec![0, 128], count: 3 };
        assert_eq!(identity.to_dense_bytes(&host_bytes, &[3], 2), vec![0, 128, 0, 128, 0, 128]);

        // A contiguous window holds the dense bytes of its shard.
        let contiguous = HostShardSource::Window { window: 4..8, byte_strides: None };
        assert_eq!(contiguous.to_dense_bytes(&host_bytes, &[1, 4], 1), vec![4, 5, 6, 7]);

        // The last two columns of a `[2, 4]` array start at its third element, and their rows are four bytes apart.
        let strided = HostShardSource::Window { window: 2..8, byte_strides: Some(vec![4, 1]) };
        assert_eq!(strided.to_dense_bytes(&host_bytes, &[2, 2], 1), vec![2, 3, 6, 7]);

        // A strided shard without elements has no bytes, wherever its window starts.
        let empty = HostShardSource::Window { window: 8..8, byte_strides: Some(vec![4, 1]) };
        assert_eq!(empty.to_dense_bytes(&host_bytes, &[0, 2], 1), Vec::<u8>::new());
    }

    #[test]
    fn test_prepared_host_shards_bytes() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect();
        let mesh = DeviceMesh::new(logical_mesh_2x2(), devices).unwrap();
        let host_bytes = Arc::new(values_to_bytes(&[0f32, 1., 2., 3., 4., 5., 6., 7.]));
        let prepare = |r#type: ArrayType, sharding: Sharding| {
            let uploads = HostBufferUploads::new(&session, r#type, Some(mesh.clone()), Some(sharding), None).unwrap();
            let shards = uploads.shards.into_iter().map(|shard| (shard.dimensions, shard.source)).collect();
            let mut prepared = PreparedHostShards::new(shards, 4);
            (0..4).map(|index| prepared.bytes(&host_bytes, index)).collect::<Vec<_>>()
        };
        let values = |bytes: &HostBufferSlice| values_from_bytes::<f32>(bytes.as_ref());

        // Every replica of a fully replicated array is a window of the host bytes, so none of them copies them.
        let replicated =
            prepare(ArrayType::new_static(DataType::F32, [2, 4]), Sharding::replicated(mesh.logical_mesh().clone(), 2));
        for bytes in &replicated {
            assert!(Arc::ptr_eq(&bytes.bytes, &host_bytes));
            assert_eq!(bytes.range, 0..32);
        }

        // A column tile of a single row is contiguous in the host bytes even though it does not span its trailing
        // dimension, so it is a window of them as well.
        let row_columns = prepare(
            ArrayType::new_static(DataType::F32, [1, 8]),
            Sharding::new(
                mesh.logical_mesh().clone(),
                vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
            )
            .unwrap(),
        );
        for bytes in &row_columns {
            assert!(Arc::ptr_eq(&bytes.bytes, &host_bytes));
        }
        assert_eq!(
            row_columns.iter().map(|bytes| bytes.range.clone()).collect::<Vec<_>>(),
            [0..16, 16..32, 0..16, 16..32],
        );

        // The replicas of a strided column tile share one gathered copy, and the tiles that differ have their own.
        let columns = prepare(
            ArrayType::new_static(DataType::F32, [2, 4]),
            Sharding::new(
                mesh.logical_mesh().clone(),
                vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
            )
            .unwrap(),
        );
        assert_eq!(
            columns.iter().map(values).collect::<Vec<_>>(),
            vec![vec![0., 1., 4., 5.], vec![2., 3., 6., 7.], vec![0., 1., 4., 5.], vec![2., 3., 6., 7.]],
        );
        assert!(Arc::ptr_eq(&columns[0].bytes, &columns[2].bytes));
        assert!(Arc::ptr_eq(&columns[1].bytes, &columns[3].bytes));
        assert!(!Arc::ptr_eq(&columns[0].bytes, &columns[1].bytes));
        assert!(!Arc::ptr_eq(&columns[0].bytes, &host_bytes));

        // Along the unreduced axis `x`, the devices at coordinate zero hold windows of the host bytes and the others
        // share one copy of the additive identity.
        let unreduced = prepare(
            ArrayType::new_static(DataType::F32, [8]),
            Sharding::replicated(mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap(),
        );
        assert!(Arc::ptr_eq(&unreduced[0].bytes, &host_bytes));
        assert!(Arc::ptr_eq(&unreduced[1].bytes, &host_bytes));
        assert_eq!(unreduced[2].as_ref(), values_to_bytes(&[-0f32; 8]).as_slice());
        assert!(Arc::ptr_eq(&unreduced[2].bytes, &unreduced[3].bytes));
        assert!(!Arc::ptr_eq(&unreduced[2].bytes, &host_bytes));
    }

    #[test]
    fn test_host_buffer_slice_new() {
        // A slice covers its range of the host bytes, and its clones share the allocation of those bytes.
        let bytes = Arc::new(vec![1u8, 2, 3, 4]);
        let slice = HostBufferSlice::new(Arc::clone(&bytes), 1..3);
        assert_eq!(slice.as_ref(), [2, 3]);
        assert!(Arc::ptr_eq(&slice.clone().bytes, &bytes));
    }

    #[test]
    fn test_host_buffer_slice_from() {
        // A slice that is created from owned bytes covers all of them.
        let slice = HostBufferSlice::from(vec![1u8, 2, 3]);
        assert_eq!(slice.range, 0..3);
        assert_eq!(slice.as_ref(), [1, 2, 3]);
    }

    #[test]
    fn test_pending_host_upload_new() {
        let client = cpu_client(1);
        let devices = client.addressable_devices().unwrap();
        let memory = devices[0].default_memory().unwrap();
        let (upload, buffer) = PendingHostUpload::new(client, BufferType::F32, vec![2], memory, None).unwrap().unwrap();
        assert_eq!(upload.buffer_type, BufferType::F32);
        assert_eq!(upload.dimensions, vec![2]);

        // The destination buffer is pending until its transfer manager fills it, and dropping the manager without
        // filling the buffer poisons it, so that no reader of the buffer can wait forever.
        assert_eq!(ExecutionFence::new(vec![buffer.ready().unwrap()]).is_ready(), Ok(false));
        drop(upload);
        assert!(buffer.ready().unwrap().r#await().is_err());
    }

    #[test]
    fn test_host_upload_workers_new() {
        // Workers start lazily, so a new set of workers only records its limit.
        let workers = HostUploadWorkers::new(NonZeroUsize::new(3).unwrap());
        assert_eq!(workers.state.worker_limit, 3);
        let queue = workers.state.lock();
        assert_eq!((queue.jobs.len(), queue.accepted_job_count, queue.worker_count, queue.shut_down), (0, 0, 0, false));
    }

    #[test]
    fn test_host_upload_workers_submit() {
        let workers = HostUploadWorkers::new(NonZeroUsize::new(2).unwrap());
        let (sender, receiver) = mpsc::channel();

        // Every job whose sources are ready runs once, on a named worker thread.
        for index in 0..3 {
            let sender = sender.clone();
            workers
                .submit(ExecutionFence::new(Vec::new()), move || {
                    sender.send((index, thread::current().name().map(str::to_string))).unwrap()
                })
                .unwrap();
        }
        let mut results = (0..3).map(|_| receiver.recv_timeout(WAIT_TIMEOUT).unwrap()).collect::<Vec<_>>();
        results.sort();
        let name = Some("ryft-host-upload".to_string());
        assert_eq!(results, vec![(0, name.clone()), (1, name.clone()), (2, name)]);
    }

    #[test]
    fn test_host_upload_workers_submit_worker_limit() {
        let workers = HostUploadWorkers::new(NonZeroUsize::new(2).unwrap());
        let (started_sender, started) = mpsc::channel();
        let (finished_sender, finished) = mpsc::channel();
        let mut gates = Vec::new();
        for index in 0..5 {
            let (gate_sender, gate) = mpsc::channel::<()>();
            gates.push(gate_sender);
            let started_sender = started_sender.clone();
            let finished_sender = finished_sender.clone();
            workers
                .submit(ExecutionFence::new(Vec::new()), move || {
                    if index < 2 {
                        started_sender.send(()).unwrap();
                        gate.recv().unwrap();
                    }
                    finished_sender.send(thread::current().id()).unwrap();
                })
                .unwrap();
        }

        // The first two jobs block two different workers while the other three are queued. If the limit did not hold,
        // the queued jobs would start further workers, and they would then run on more than two threads.
        started.recv_timeout(WAIT_TIMEOUT).unwrap();
        started.recv_timeout(WAIT_TIMEOUT).unwrap();
        for gate in &gates[..2] {
            gate.send(()).unwrap();
        }
        let threads = (0..5).map(|_| finished.recv_timeout(WAIT_TIMEOUT).unwrap()).collect::<HashSet<_>>();
        assert_eq!(threads.len(), 2);
    }

    #[test]
    fn test_host_upload_workers_submit_dependent_jobs() {
        let workers = HostUploadWorkers::new(NonZeroUsize::new(2).unwrap());

        // Each job waits for the value of the job submitted before it and passes it on incremented. The first job waits
        // until the whole chain, which is longer than the number of workers, has been submitted.
        let (first, mut previous) = mpsc::channel::<usize>();
        for _ in 0..6 {
            let (sender, next) = mpsc::channel();
            let receiver = std::mem::replace(&mut previous, next);
            workers
                .submit(ExecutionFence::new(Vec::new()), move || sender.send(receiver.recv().unwrap() + 1).unwrap())
                .unwrap();
        }
        first.send(0).unwrap();
        assert_eq!(previous.recv_timeout(WAIT_TIMEOUT), Ok(6));
    }

    #[test]
    fn test_host_upload_workers_submit_pending_sources() {
        let client = cpu_client(1);
        let workers = HostUploadWorkers::new(NonZeroUsize::new(1).unwrap());
        let (sender, receiver) = mpsc::channel();

        // A job whose sources are pending does not occupy the only worker, so a later job with ready sources runs
        // first. The parked jobs run once their sources complete, including when the sources fail.
        let (succeeding, succeeding_promise) = client.event(()).unwrap();
        let (failing, failing_promise) = client.event(()).unwrap();
        let sources = [
            ExecutionFence::new(vec![succeeding]),
            ExecutionFence::new(vec![failing]),
            ExecutionFence::new(Vec::new()),
        ];
        for (index, sources) in sources.into_iter().enumerate() {
            let sender = sender.clone();
            workers.submit(sources, move || sender.send(index).unwrap()).unwrap();
        }
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Ok(2));
        failing_promise.set(Some(PjrtError::cancelled("source failed"))).unwrap();
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Ok(1));
        succeeding_promise.set(None).unwrap();
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Ok(0));
    }

    #[test]
    fn test_host_upload_workers_submit_dependent_sources() {
        let client = cpu_client(1);
        let workers = HostUploadWorkers::new(NonZeroUsize::new(2).unwrap());
        let (sender, receiver) = mpsc::channel();

        // Each job completes the sources of the next one once it has run. The chain is longer than the number of
        // workers and is submitted in reverse order, so it runs in dependency order only because the jobs that wait
        // for their sources do not occupy workers.
        let chain_length = 6;
        let (events, promises): (Vec<_>, Vec<_>) = (0..chain_length).map(|_| client.event(()).unwrap()).unzip();
        let mut promises = promises.into_iter().map(Some).chain([None]).collect::<Vec<_>>();
        let first_promise = promises.remove(0).unwrap();
        for (index, event) in events.into_iter().enumerate().rev() {
            let sender = sender.clone();
            let next_promise = promises.pop().unwrap();
            let job = move || {
                sender.send(index).unwrap();
                if let Some(promise) = next_promise {
                    promise.set(None).unwrap();
                }
            };
            workers.submit(ExecutionFence::new(vec![event]), job).unwrap();
        }
        drop(sender);
        first_promise.set(None).unwrap();
        assert_eq!(
            (0..chain_length).map(|_| receiver.recv_timeout(WAIT_TIMEOUT).unwrap()).collect::<Vec<_>>(),
            (0..chain_length).collect::<Vec<_>>(),
        );
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Err(RecvTimeoutError::Disconnected));
    }

    #[test]
    fn test_host_upload_workers_submit_panicking_job() {
        let workers = HostUploadWorkers::new(NonZeroUsize::new(1).unwrap());
        let (threads_sender, threads) = mpsc::channel();
        let (owned, owned_receiver) = mpsc::channel::<()>();
        let panicking_sender = threads_sender.clone();
        workers
            .submit(ExecutionFence::new(Vec::new()), move || {
                panicking_sender.send(thread::current().id()).unwrap();
                owned.send(()).unwrap();
                panic!("host upload job failed");
            })
            .unwrap();
        workers
            .submit(ExecutionFence::new(Vec::new()), move || threads_sender.send(thread::current().id()).unwrap())
            .unwrap();

        // The panicking job is dropped while it unwinds, which drops what it owns, and its worker runs the next job.
        let worker = threads.recv_timeout(WAIT_TIMEOUT).unwrap();
        assert_eq!(owned_receiver.recv_timeout(WAIT_TIMEOUT), Ok(()));
        assert_eq!(owned_receiver.recv_timeout(WAIT_TIMEOUT), Err(RecvTimeoutError::Disconnected));
        assert_eq!(threads.recv_timeout(WAIT_TIMEOUT), Ok(worker));
    }

    #[test]
    fn test_host_upload_workers_submit_panicking_job_with_panicking_payload() {
        /// Panic payload whose destructor panics as well.
        struct PanickingPayload;

        impl Drop for PanickingPayload {
            fn drop(&mut self) {
                panic!("host upload panic payload failed");
            }
        }

        let workers = HostUploadWorkers::new(NonZeroUsize::new(1).unwrap());
        let (sender, receiver) = mpsc::channel();
        workers.submit(ExecutionFence::new(Vec::new()), || std::panic::panic_any(PanickingPayload)).unwrap();
        workers.submit(ExecutionFence::new(Vec::new()), move || sender.send(()).unwrap()).unwrap();

        // The panic of the payload destructor neither stops the only worker nor skips the bookkeeping of the job.
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Ok(()));
        let deadline = Instant::now() + WAIT_TIMEOUT;
        while workers.state.lock().accepted_job_count > 0 {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(workers.state.lock().worker_count, 1);
    }

    #[test]
    fn test_host_upload_workers_default() {
        // The default limit follows the parallelism of the machine, with at least one worker and a fixed maximum.
        let worker_limit = HostUploadWorkers::default().state.worker_limit;
        assert!((1..=HostUploadWorkers::MAX_DEFAULT_WORKER_COUNT).contains(&worker_limit));
    }

    #[test]
    fn test_host_upload_workers_drop() {
        let workers = HostUploadWorkers::new(NonZeroUsize::new(1).unwrap());
        let state = Arc::downgrade(&workers.state);
        let (release, gate) = mpsc::channel::<()>();
        let (sender, receiver) = mpsc::channel();
        workers.submit(ExecutionFence::new(Vec::new()), move || gate.recv().unwrap()).unwrap();
        for index in 1..3 {
            let sender = sender.clone();
            workers.submit(ExecutionFence::new(Vec::new()), move || sender.send(index).unwrap()).unwrap();
        }
        let (event, promise) = cpu_client(1).event(()).unwrap();
        workers.submit(ExecutionFence::new(vec![event]), move || sender.send(3).unwrap()).unwrap();

        // Dropping the owner while its only worker is blocked, two jobs are queued, and one waits for its sources still
        // runs the queued jobs in submission order and the waiting one once its sources complete, after which the
        // worker exits and releases the shared state.
        drop(workers);
        release.send(()).unwrap();
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Ok(1));
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Ok(2));
        promise.set(None).unwrap();
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Ok(3));
        assert_eq!(receiver.recv_timeout(WAIT_TIMEOUT), Err(RecvTimeoutError::Disconnected));
        let deadline = Instant::now() + WAIT_TIMEOUT;
        while state.strong_count() > 0 {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn test_run_host_uploads() {
        let client = cpu_client(2);
        let devices = client.addressable_devices().unwrap();
        let destinations = || {
            devices
                .iter()
                .map(|device| {
                    let memory = device.default_memory().unwrap();
                    PendingHostUpload::new(client, BufferType::F32, vec![2], memory, None).unwrap().unwrap()
                })
                .unzip::<_, _, Vec<_>, Vec<_>>()
        };
        let read = |buffer: &Buffer<'static>| {
            values_from_bytes::<f32>(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice())
        };

        // Each destination buffer receives the bytes that are prepared for the index of its upload.
        let (uploads, buffers) = destinations();
        run_host_uploads(
            uploads,
            |index| Ok(HostBufferSlice::from(values_to_bytes(&[index as f32, 1.]))),
            |error| PjrtError::internal(error.to_string()),
        );
        assert_eq!(buffers.iter().map(read).collect::<Vec<_>>(), vec![vec![0., 1.], vec![1., 1.]]);

        // After the first failure no further bytes are prepared, the buffers that already received their bytes keep
        // them, and the others report the error that is derived from the failure.
        let (mut uploads, mut buffers) = destinations();
        let (more_uploads, more_buffers) = destinations();
        uploads.extend(more_uploads);
        buffers.extend(more_buffers);
        let mut prepared = Vec::new();
        run_host_uploads(
            uploads,
            |index| {
                prepared.push(index);
                match index {
                    0 => Ok(HostBufferSlice::from(values_to_bytes(&[1f32, 2.]))),
                    _ => Err(PjrtError::cancelled("readback failed").into()),
                }
            },
            |error| PjrtError::aborted(format!("upload failed: {error}")),
        );
        assert_eq!(prepared, vec![0, 1]);
        assert_eq!(read(&buffers[0]), vec![1., 2.]);
        for buffer in &buffers[1..] {
            assert!(matches!(
                buffer.ready().unwrap().r#await(),
                Err(PjrtError::Aborted { message, .. }) if message == "upload failed: readback failed",
            ));
        }
    }

    #[test]
    fn test_row_major_element_strides() {
        assert_eq!(row_major_element_strides(&[2, 3, 4], DataType::F32).unwrap(), vec![12, 4, 1]);
        assert_eq!(row_major_element_strides(&[], DataType::F32).unwrap(), Vec::<usize>::new());

        // A stride that overflows `usize` is an error.
        let error = row_major_element_strides(&[2, usize::MAX, 2], DataType::F32).unwrap_err();
        let message = format!(
            "row-major stride for array with shape `[2, {}, 2]` and element type `f32` exceeds the maximum allowed \
             size of {}",
            usize::MAX,
            usize::MAX,
        );
        assert!(matches!(&error, Error::SizeLimitExceeded { message: actual } if actual == &message));
        assert_eq!(error.to_string(), message);
    }

    #[test]
    fn test_copy_dense_shard_bytes() {
        // The second column of a `[2, 2]` array occupies two separate byte ranges of its dense representation.
        let mut global_bytes = vec![0u8; 16];
        copy_dense_shard_bytes(
            values_to_bytes(&[1f32, 2.]).as_slice(),
            &[0..2, 1..2],
            &[2, 1],
            &[1, 1],
            0,
            0,
            0,
            size_of::<f32>(),
            &mut global_bytes,
        );
        assert_eq!(values_from_bytes::<f32>(global_bytes.as_slice()), vec![0., 1., 0., 2.]);
    }
}
