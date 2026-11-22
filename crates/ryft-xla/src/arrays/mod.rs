//! Distributed device arrays backed by PJRT [`Buffer`]s, which are the concrete eager values of the XLA backend.
//! An [`Array`] is one logical value of an [`ArrayType`] with a static [`Shape`], placed on the devices of a
//! [`DeviceMesh`] according to a [`Sharding`]. [`ShardLayout`] expands that placement into one [`ShardDescriptor`] per
//! device, which records the slice of the global array that the device holds. Every [`Array`] keeps one [`ArrayShard`]
//! per device, including the shards that live in other processes, so that every process can reason about the complete
//! global placement while only transferring, executing with, or reading the buffers that it can address. Clones of an
//! array share its type and shards, so cloning never copies device data. The [`shards`](mod@shards) module defines the
//! shard types.
//!
//! Every array belongs to a [`Session`], which determines the PJRT [`Client`], session, effect scope, and compilation
//! options that operations on the array use. Arrays implement `ryft-core` value capabilities by binding each operation
//! through that session, which executes it eagerly, operation-by-operation, as a sequence of cached single-operation
//! programs.
//!
//! [`Array::from_host_buffer`] uploads a dense host value, [`Array::from_host_shards`] uploads the host shards owned by
//! this process, and [`Array::from_addressable_buffers`] wraps existing PJRT [`Buffer`]s. These constructors support
//! placement over auto mesh axes when the caller supplies runtime sharding. [`Array::copy_to_host`] and
//! [`ArrayShard::copy_to_host_bytes`] read values back to the host. The [`host`](mod@host) module implements the
//! uploads and the readback of whole arrays, including the combination of the separate values that devices hold along
//! unreduced mesh axes. [`Array::to_placement`], [`Array::into_placement`], and [`Session::place`] move arrays and host
//! values to other devices or shardings. The [`placement`](mod@placement) module defines what a placement is and how to
//! request one, and the [`resharding`](mod@resharding) module implements the moves. As in JAX, an array is committed to
//! its devices when they were chosen on purpose (e.g., by a placement with a destination) and uncommitted when they are
//! only a default (e.g., for a host value that was uploaded to the first device), and eager operations move uncommitted
//! inputs to the devices of committed ones (refer to [`Array::is_committed`]). Array failures are reported through
//! [`Error::Array`] with an [`ArrayError`].
//!
//! Every session counts the arrays that are created through it, which [`ArrayStatistics`] reports, and can describe
//! the ones that are still alive as [`LiveArray`]s. The unit that it counts is the storage that an array shares with
//! its clones, so cloning and dropping array handles involves no bookkeeping.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::sync::Arc;

use ryft_core::{
    Add, And, Array as CoreArray, ArrayType, AssertionValue, Axis, Broadcast, Concretizable, DataType, DeviceId,
    DeviceMesh, DimensionError, DimensionFromScalar, DimensionFromScalarOperation, DimensionSize,
    DimensionSizeOperation, DimensionType, DimensionValue, DimensionVariable, Div, ElementType, Layout, Memory, Mul,
    Neg, Not, Operation, Or, Parameter, ParameterPath, Parameterized, Placed, Placement, ProgramError,
    ProjectedContext, Select, Shape, ShapeChangingCollectiveValue, Sharding, StaticShape, Sub, Typed, UnreducedKind,
    Value, ValueDomainDispatch, WhilePredicate, Xor,
};
use ryft_macros::Parameter;
use ryft_pjrt::{Buffer, Client, Error as PjrtError, ExecutionFence};

use crate::arrays::host::validate_mesh_device_processes;
use crate::arrays::metadata::{ArrayMetadata, ArrayOutputPlan};
use crate::arrays::shards::validate_addressable_buffer;
use crate::arrays::statistics::ArrayRegistration;
use crate::types::ArrayTypeExtension;
use crate::{Error, Session, Type};

pub mod host;
pub mod metadata;
pub mod placement;
pub mod resharding;
pub mod retained;
pub mod shards;
pub mod statistics;

pub use placement::{HostOrArray, Placeable, PlacementOptions};
pub use retained::RetainedArray;
pub use shards::{ArrayShard, ArrayShardStorage, ShardDescriptor, ShardIndex, ShardLayout};
pub use statistics::{ArrayStatistics, LiveArray};

/// Represents errors produced while validating distributed [`Array`] metadata and buffers, placing arrays on devices,
/// or materializing arrays on the host. Array APIs return these errors through [`Error::Array`]. Core and PJRT failures
/// generally retain their original error types. [`Self::InvalidShardDimension`] also records the shard and dimension
/// that failed transport encoding alongside the original core error.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ArrayError {
    #[error("expected static shape but got `{shape}`")]
    DynamicShape { shape: Shape },

    #[error("no default local device is available, because the PJRT client has no addressable devices")]
    MissingDefaultDevice,

    #[error(
        "array type sharding `{type_sharding}` does not match the placement `{sharding}` over the non-auto mesh axes"
    )]
    PlacementTypeMismatch { type_sharding: Sharding, sharding: Sharding },

    #[error("array `{path}` cannot be donated while `may_alias` is `Some(true)`")]
    DonatedArrayMayAlias { path: ParameterPath },

    #[error("compiled reshard does not support `Manual` mesh axis {axis_name}")]
    ManualReshardingAxis { axis_name: String },

    #[error(
        "compiled reshard cannot produce the shards of different sizes of shape `{shape}` with sharding `{sharding}`"
    )]
    UnevenReshardingShards { shape: StaticShape, sharding: Sharding },

    #[error("type `{data_type}` has no additive identity to split a host value into summands along unreduced axes")]
    MissingUnreducedSumIdentity { data_type: DataType },

    #[error("device {device_id} is not present in the device mesh")]
    DeviceNotInMesh { device_id: DeviceId },

    #[error("device {device_id} reports process index {actual}, but expected {expected}")]
    DeviceProcessIndexMismatch { device_id: DeviceId, expected: usize, actual: usize },

    #[error("device {device_id} is not addressable by the PJRT client in process {process_index}")]
    NonAddressableDevice { device_id: DeviceId, process_index: usize },

    #[error("device {device_id} has no addressable memory for `{memory}`")]
    UnsupportedMemory { device_id: DeviceId, memory: Memory },

    #[error("got multiple buffers for device {device_id}")]
    MultipleBuffersOnDevice { device_id: DeviceId },

    #[error("buffer has type `{actual}`, but expected `{expected}`")]
    BufferTypeMismatch { expected: ArrayType, actual: ArrayType },

    #[error("buffer on device {device_id} uses memory kind `{actual}`, but expected `{expected}`")]
    BufferMemoryMismatch { device_id: DeviceId, expected: Memory, actual: String },

    #[error("zero-space buffer on device {device_id} contains a nonzero physical carrier value")]
    NonCanonicalZeroBuffer { device_id: DeviceId },

    #[error("host buffer needs {expected} byte(s) but got {actual}")]
    HostBufferByteCountMismatch { expected: usize, actual: usize },

    #[error("missing host shard for device {device_id}")]
    MissingHostShard { device_id: DeviceId },

    #[error("got multiple host shards for device {device_id}")]
    MultipleHostShardsOnDevice { device_id: DeviceId },

    #[error("host shard on device {device_id} needs {expected} byte(s) but got {actual}")]
    HostShardByteCountMismatch { device_id: DeviceId, expected: usize, actual: usize },

    #[error("shape rank {shape_rank} does not match shard slice rank {slice_rank}")]
    ShardSliceRankMismatch { shape_rank: usize, slice_rank: usize },

    #[error("array shard #{shard_index} on device {device_id} has no addressable buffer")]
    MissingShardBuffer { shard_index: ShardIndex, device_id: DeviceId },

    #[error(
        "copied shard #{shard_index} from device {device_id} to host and got {actual} byte(s), but expected {expected}"
    )]
    ShardByteCountMismatch { shard_index: ShardIndex, device_id: DeviceId, expected: usize, actual: usize },

    #[error("cross-host transfer for shard #{shard_index}, dimension #{dimension}: {source}")]
    InvalidShardDimension {
        shard_index: ShardIndex,
        dimension: usize,
        #[source]
        source: DimensionError,
    },
}

/// Distributed array with a global [`StaticShape`] and element [`DataType`] as well as [`Sharding`] information.
/// An [`Array`] represents one logical [`ArrayType`] whose elements may be split or replicated across the multiple
/// devices in a [`DeviceMesh`], potentially spanning multiple nodes or processes. The global array is described by
/// its type and sharding metadata, while each physical piece of that global array is represented by an [`ArrayShard`].
/// [`Array::shards`] is a global list, which contains one [`ArrayShard`] for each device that participates in the array
/// placement, and not only for the devices that are visible to the current process. In a single-process setup, every
/// shard is normally _addressable_ because the local [`Client`] can directly access its storage. In a multi-device or
/// multi-node setup, the same logical array can span devices owned by other processes. Shards on the current process's
/// devices are addressable and ordinarily carry local [`Buffer`]s. Bufferless [`DataType::Zero`] shards are also
/// addressable because their unique value requires no storage. Shards on remote devices are non-addressable and carry
/// only metadata such as their global [`ShardIndex`], [`DeviceId`], and [`ArrayType`]. Keeping both addressable and
/// non-addressable shards in the same [`Array`] lets local code reason about the complete global placement while only
/// transferring, executing with, or materializing buffers that this process can access directly. This distinction is
/// what allows array movement, execution argument assembly, and cross-host transfers to preserve the full global
/// sharding contract without requiring every process to own every shard buffer.
///
/// Note that the current StableHLO execution path represents logical one-bit integers with predicate buffers. Arrays
/// retain their logical `I1` or `U1` type while using one byte per element in this physical representation. This
/// executable convention is separate from [`ToPjrt`](crate::ToPjrt) for [`DataType`], which preserves native PJRT
/// one-bit integer types.
#[derive(Clone, Parameter)]
pub struct Array {
    /// Type and shards of this [`Array`]. They are immutable once constructed and shared by every clone,
    /// so cloning an array costs a few reference-count increments rather than a deep copy.
    parts: Arc<ArrayParts>,

    /// Whole-execution completion fence for arrays produced by an asynchronous PJRT launch. This is separate from
    /// each shard buffer's ready event because it also retains errors from zero-output effects and other device work
    /// associated with the launch. Cloned output arrays share the same fence. The result of a placement joins the
    /// fence of its source with the readiness of every buffer that the placement created and with the fences of the
    /// programs that it ran, because readbacks wait for this fence but read only one replica of each tile. Plain host
    /// uploads leave it empty, because every replica receives the same host bytes. The uploads of placements set it
    /// whether they run in the background or, when the PJRT plugin lacks host to device transfer managers, on the
    /// calling thread (refer to [`Self::with_upload_completion`]).
    execution_fence: Option<ExecutionFence>,

    /// Completion fence of the PJRT cross-host sends that the placement which produced this array issued from the
    /// current process, if it issued any. The data of this array does not depend on those sends, because they deliver
    /// shards to the devices of other processes, but a send that fails after it was issued is otherwise only noticed
    /// by its receiver. Retaining the fence here lets [`Self::ready`] and [`Self::block_until_ready`] surface that
    /// failure on the sending process. A send completes only once its receiver has issued the matching receive, so
    /// the placement must not wait for it, and only those explicit readiness functions do. Clones of this array share
    /// the same fence. Dropping every handle of the fence neither cancels nor waits for the sends, and their errors
    /// are then no longer observed by this process. Derived arrays (e.g., the results of operations on this array)
    /// do not inherit the fence.
    send_fence: Option<ExecutionFence>,

    /// [`Session`] this array belongs to, retained across clones and derived values. It determines the PJRT client,
    /// compilation cache, kernel runtimes, and default compilation options, as well as the effect scope that
    /// receiver-based operations on this array execute with. Explicit sessions select their own scope, while
    /// receiver-based operations use this value's session, so `x + y` and `y + x` may select different scopes.
    session: Session,

    /// Whether this array is committed to its devices. Refer to [`Self::is_committed`] for more information.
    committed: bool,

    /// Whether this process chose the devices of this array independently of the other processes.
    /// Refer to [`Self::is_locally_assigned`] for more information.
    locally_assigned: bool,
}

impl Array {
    /// Creates an [`Array`] of type `r#type` from the provided _addressable_ [`Buffer`]s using the provided concrete
    /// [`DeviceMesh`] to determine what [`ArrayShard`]s make up the array and which ones correspond to the addressable
    /// buffers. The provided [`ArrayType`] describes the global logical array and must have a static [`Shape`].
    ///
    /// With [`None`] for `sharding`, the function uses the sharding of `r#type` if present and defaults to replication
    /// over `mesh` otherwise. Each buffer in a replicated placement must have the full logical shape, and callers are
    /// responsible for supplying the same value in each replica.
    ///
    /// A supplied sharding can place the array over auto mesh axes. After removing its auto axes with
    /// [`Sharding::without_auto_axes`], it must match any sharding in the type, or the function returns
    /// [`ArrayError::PlacementTypeMismatch`]. The type of the resulting array records that projection, and
    /// [`Self::sharding`] returns the full runtime placement.
    ///
    /// Each [`Buffer`] is assigned to a global shard based on the [`DeviceId`] of its owning device. Failures use
    /// [`Error::Array`] with [`ArrayError::MultipleBuffersOnDevice`] when multiple buffers belong to the same device,
    /// [`ArrayError::DeviceNotInMesh`] when a buffer's device is absent from `mesh`,
    /// [`ArrayError::DeviceProcessIndexMismatch`] when a buffer's process differs from the corresponding mesh device or
    /// when the PJRT client reports another process for a device of `mesh` than the one that `mesh` assigns to it, and
    /// [`ArrayError::NonAddressableDevice`] when `mesh` assigns a device to this process that the PJRT client does not
    /// address. Both device checks apply whether or not a buffer is supplied for that device, because no process would
    /// then hold the buffer of its shard, so a partial import accepts the same meshes as a host upload. A device that
    /// the client does not list keeps the process that `mesh` assigns to it, because it can belong to a process whose
    /// devices the client cannot see. A layout of `r#type` that does not address the shape of each shard is rejected
    /// with the [`TypeError`](ryft_core::TypeError) of [`ArrayType::validate_layout`], even without buffers.
    /// [`ArrayError::BufferTypeMismatch`] reports a data type or static shape that differs from the shard type derived
    /// from `r#type`, `mesh`, and the effective sharding. A buffer whose physical memory differs from `r#type.memory()`
    /// produces [`ArrayError::BufferMemoryMismatch`]. [`Memory::Device`] denotes the owning device's default memory.
    /// Host placements require the corresponding pinned or unpinned memory kind. Shards without a local, addressable
    /// buffer are retained as non-addressable [`ArrayShard`]s. For a logical [`DataType::Zero`] array, every private
    /// predicate carrier must contain only false bits. A noncanonical buffer is rejected so that its physical payload
    /// cannot become observable through the logical zero-space value. Valid carriers are discarded after validation
    /// and the local shards become addressable bufferless zero-space shards.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] the new [`Array`] belongs to. Its client must own every provided buffer, and it is
    ///     what eager execution and free transforms recover through [`Value::domain`].
    ///   - `r#type`: Global [`ArrayType`] for the new [`Array`].
    ///   - `mesh`: [`DeviceMesh`] that is used to determine the [`ArrayShard`] placement.
    ///   - `sharding`: Optional runtime [`Sharding`] over `mesh`.
    ///   - `buffers`: [`Buffer`]s for the [`ArrayShard`]s that are addressable from the current process. An empty
    ///     list creates an array with only shard metadata.
    #[inline]
    pub fn from_addressable_buffers(
        session: &Session,
        r#type: ArrayType,
        mesh: DeviceMesh,
        sharding: Option<Sharding>,
        buffers: Vec<Buffer<'static>>,
    ) -> Result<Self, Error> {
        Self::from_addressable_buffers_internal(
            session,
            r#type,
            sharding,
            mesh,
            buffers.into_iter().map(Arc::new).collect(),
            true,
        )
    }

    /// Creates an [`Array`] from buffers produced by the backend whose zero-space carriers are known to be canonical.
    /// This is the trusted internal counterpart of [`Self::from_addressable_buffers`]. It avoids synchronizing device
    /// buffers back to the host solely to revalidate an invariant already enforced by Ryft's lowering and input
    /// boundaries, and discards canonical zero-space carriers after extracting their placement metadata.
    pub(crate) fn from_canonical_addressable_buffers(
        session: &Session,
        r#type: ArrayType,
        sharding: Option<Sharding>,
        mesh: DeviceMesh,
        buffers: Vec<Buffer<'static>>,
    ) -> Result<Self, Error> {
        Self::from_addressable_buffers_internal(
            session,
            r#type,
            sharding,
            mesh,
            buffers.into_iter().map(Arc::new).collect(),
            false,
        )
    }

    /// Implements addressable buffer construction. The buffers may be ones that other arrays also hold. Sharing is safe
    /// because arrays never mutate their buffers and because donation skips arrays whose buffers are shared (refer to
    /// [`Self::has_unique_shard_buffers`]). When `imported` is set, the buffers and the mesh come from outside of Ryft
    /// (i.e., through [`Self::from_addressable_buffers`]), so this function also validates the private zero-space
    /// carrier payloads of the buffers and the processes that the mesh assigns to its devices, as a host upload does.
    /// The mesh describes every device, whether or not a buffer is supplied for it, so a device of this process that
    /// the mesh assigns to another process, or a device that the mesh assigns to this process but that this process
    /// does not address, would otherwise leave its shard without a buffer on any process. The other constructors build
    /// their meshes and buffers from ones that were validated already.
    fn from_addressable_buffers_internal(
        session: &Session,
        r#type: ArrayType,
        sharding: Option<Sharding>,
        mesh: DeviceMesh,
        buffers: Vec<Arc<Buffer<'static>>>,
        imported: bool,
    ) -> Result<Self, Error> {
        // Derive the metadata and the global shard descriptors implied by the runtime placement and the device mesh.
        let (metadata, descriptors) = ArrayMetadata::new(r#type, sharding, mesh)?;
        if imported {
            validate_mesh_device_processes(session.client(), metadata.mesh())?;
        }

        // Index the provided `Buffer`s by device while rejecting duplicate local buffers for the same device and
        // buffers that the session's client does not own.
        let mut buffers_by_device = HashMap::with_capacity(buffers.len());
        for buffer in buffers {
            let device_id = validate_addressable_buffer(
                session.client(),
                metadata.r#type(),
                &descriptors,
                metadata.shard_index_by_device(),
                &buffer,
                imported,
            )?;
            if buffers_by_device.contains_key(&device_id) {
                return Err(Error::from(ArrayError::MultipleBuffersOnDevice { device_id }));
            }
            buffers_by_device.insert(device_id, buffer);
        }

        // Materialize the global shard list with local storage attached to the shards addressable from this process.
        // A validated zero-space carrier is discarded because its false bits carry no logical information.
        let zero_space = metadata.r#type().data_type().is_zero();
        let shards = descriptors
            .into_iter()
            .map(|descriptor| {
                let buffer = buffers_by_device.remove(&descriptor.device().id());
                if zero_space && buffer.is_some() {
                    ArrayShard::zero(descriptor)
                } else {
                    ArrayShard::new(descriptor, buffer)
                }
            })
            .collect::<Vec<_>>();

        Ok(Self {
            parts: Arc::new(ArrayParts::new(session, Arc::new(metadata), shards)),
            execution_fence: None,
            send_fence: None,
            session: session.clone(),
            committed: true,
            locally_assigned: false,
        })
    }

    /// Constructs a bufferless array of the unique value represented by [`DataType::Zero`]. Local shards are marked
    /// addressable without allocating PJRT buffers, while remote shards retain metadata only.
    pub(crate) fn zero(
        session: &Session,
        r#type: ArrayType,
        sharding: Option<Sharding>,
        mesh: DeviceMesh,
    ) -> Result<Self, Error> {
        if !r#type.data_type().is_zero() {
            return Err(PjrtError::invalid_argument(format!(
                "bufferless zero-space construction requires zero element type but got `{}`",
                r#type.data_type(),
            ))
            .into());
        }

        let (metadata, descriptors) = ArrayMetadata::new(r#type, sharding, mesh)?;
        let client = session.client();
        let process_index = client.process_index()?;
        let addressable_device_ids =
            client.addressable_devices()?.iter().map(|device| device.id()).collect::<Result<HashSet<_>, _>>()?;

        let shards = descriptors
            .into_iter()
            .map(|descriptor| -> Result<_, Error> {
                if descriptor.device().process_index() == process_index {
                    if !addressable_device_ids.contains(&descriptor.device().id()) {
                        return Err(Error::from(ArrayError::NonAddressableDevice {
                            device_id: descriptor.device().id(),
                            process_index,
                        }));
                    }
                    Ok(ArrayShard::zero(descriptor))
                } else {
                    Ok(ArrayShard::new(descriptor, None))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            parts: Arc::new(ArrayParts::new(session, Arc::new(metadata), shards)),
            execution_fence: None,
            send_fence: None,
            session: session.clone(),
            committed: true,
            locally_assigned: false,
        })
    }

    /// Creates an [`Array`] that belongs to `session` from the per-device `buffers` that one execution of a compiled
    /// program returned for one of its outputs, placed according to that output's precomputed `plan`. The buffers
    /// must be ordered like the addressable devices that `plan` was created for, and `fence` is the whole-execution
    /// completion fence of the launch. Unlike [`Self::from_addressable_buffers`], this constructor does not query the
    /// buffers' devices, memories, element types, or dimensions, because XLA guarantees those properties for the
    /// buffers that it returns from a program whose signature Ryft lowered, and skipping the queries keeps each
    /// execution free of per-output PJRT calls. Low-level compile options can override the layouts and the memories of
    /// that signature, so an execution plan provides `plan` only for an executable whose reported layouts and memory
    /// kinds match the declared output types. In builds with debug assertions, every buffer is still validated with
    /// the same checks.
    pub(crate) fn from_execution_output(
        session: &Session,
        plan: &ArrayOutputPlan,
        buffers: Vec<Buffer<'static>>,
        fence: ExecutionFence,
    ) -> Self {
        let metadata = plan.metadata();
        let descriptors = plan.descriptors();
        let buffer_positions = plan.buffer_positions();

        #[cfg(debug_assertions)]
        for (position, buffer) in buffers.iter().enumerate() {
            let device_id = validate_addressable_buffer(
                session.client(),
                metadata.r#type(),
                descriptors,
                metadata.shard_index_by_device(),
                buffer,
                false,
            );
            debug_assert!(
                device_id.as_ref().is_ok_and(|device_id| {
                    let shard_index = metadata.shard_index_by_device()[device_id];
                    descriptors[shard_index].device().id() == *device_id
                        && buffer_positions[shard_index] == Some(position)
                }),
                "execution output buffer {position} does not match its planned placement: {device_id:?}",
            );
        }

        let is_zero = metadata.r#type().data_type().is_zero();
        let mut buffers = buffers.into_iter().map(Some).collect::<Vec<_>>();
        let shards = descriptors
            .iter()
            .zip(buffer_positions)
            .map(|(descriptor, position)| match position {
                // Zero-space outputs need no storage. A returned carrier holds only false bits and is discarded like
                // in the validating constructors, and a program may also return no carrier at all.
                Some(_) if is_zero => ArrayShard::zero(descriptor.clone()),
                Some(position) => {
                    let buffer = buffers.get_mut(*position).and_then(Option::take).map(Arc::new);
                    ArrayShard::new(descriptor.clone(), buffer)
                }
                None => ArrayShard::new(descriptor.clone(), None),
            })
            .collect::<Vec<_>>();

        Self {
            parts: Arc::new(ArrayParts::new(session, Arc::clone(metadata), shards)),
            execution_fence: Some(fence),
            send_fence: None,
            session: session.clone(),
            committed: true,
            locally_assigned: false,
        }
    }

    /// Attaches the whole-execution completion fence that produced this array. This never waits and is used by the
    /// XLA executor when reassembling pending PJRT output buffers.
    #[inline]
    pub(crate) fn with_execution_fence(mut self, fence: ExecutionFence) -> Self {
        self.execution_fence = Some(fence);
        self
    }

    /// Attaches the completion fence of the PJRT cross-host sends that the placement which produced this array
    /// issued from the current process. This never waits, and [`Self::ready`] and [`Self::block_until_ready`] later
    /// wait for the fence and return the first error of those sends.
    #[inline]
    pub(crate) fn with_send_fence(mut self, fence: ExecutionFence) -> Self {
        self.send_fence = Some(fence);
        self
    }

    /// Returns this [`Array`] with its commitment set to `committed`, keeping its storage, placement, session, and
    /// completion fences. Refer to [`Self::is_committed`] for what commitment means and which arrays are committed.
    #[inline]
    pub(crate) fn with_committed(mut self, committed: bool) -> Self {
        self.committed = committed;
        self
    }

    /// Returns this [`Array`], which an eager operation, an eager reshard, or a host upload produced, committed to its
    /// devices when `committed` is `true` or when it lives on more than one device, and uncommitted otherwise (refer to
    /// [`Self::is_committed`]). `committed` is the commitment that its operands give it (i.e., the result of
    /// [`Session::commits_placement`] and the commitment of the captured arrays for an eager operation, the commitment
    /// of the source for a reshard, and whether a mesh was chosen for an upload through [`Self::from_host_buffer`]).
    /// This is the one rule that decides the commitment of eager results and uploads, and it keeps every uncommitted
    /// one on a single device, as in JAX, where an uncommitted array always lives on one device. The outputs of
    /// compiled functions that run on more than one device are committed for the same reason.
    #[inline]
    pub(crate) fn commit_if(self, committed: bool) -> Self {
        let committed = committed || self.mesh().devices().len() > 1;
        self.with_committed(committed)
    }

    /// Sets whether this array is [locally assigned](Self::is_locally_assigned). An array with a shard on another
    /// process always clears the flag, because this process could not have chosen all its devices independently.
    #[inline]
    pub(crate) fn with_locally_assigned(mut self, locally_assigned: bool) -> Self {
        self.locally_assigned = locally_assigned && self.shards().iter().all(ArrayShard::is_addressable);
        self
    }

    /// Returns this [`Array`] with `session` as the session for future receiver-based dispatch. This lets the handle
    /// use another session, effect scope, or session mesh while keeping the array's storage, device placement, and
    /// completion fences. Existing clones keep their session, and previously submitted work keeps its original scope
    /// and ordering.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::Pjrt`] that wraps an [`InvalidArgument`](ryft_pjrt::Error::InvalidArgument) error if
    /// `session` belongs to a different PJRT client. The array's addressable shard buffers belong to its original
    /// client.
    pub(crate) fn with_session(mut self, session: &Session) -> Result<Self, Error> {
        if !std::ptr::eq(self.client(), session.client()) {
            return Err(PjrtError::invalid_argument(
                "cannot associate an array with a session of a different PJRT client",
            )
            .into());
        }
        self.session = session.clone();
        Ok(self)
    }

    /// Returns the [`DataType`] of the elements stored in this [`Array`].
    #[inline]
    pub fn data_type(&self) -> DataType {
        self.parts.metadata.r#type().data_type()
    }

    /// Returns the global [`StaticShape`] of this [`Array`].
    #[inline]
    pub fn shape(&self) -> StaticShape {
        self.parts.metadata.r#type().static_shape().unwrap()
    }

    /// Returns the physical storage [`Layout`] of this [`Array`] if it is known.
    #[inline]
    pub fn layout(&self) -> Option<&Layout> {
        self.parts.metadata.r#type().layout()
    }

    /// Returns the runtime placement of this [`Array`] over all mesh axes. Unlike the sharding of its [`Typed`] type,
    /// which records only placement over explicit and manual mesh axes, it also records placement over auto mesh axes.
    #[inline]
    pub fn sharding(&self) -> &Sharding {
        self.parts.metadata.sharding()
    }

    /// Returns the concrete [`DeviceMesh`] implied by this [`Array`]'s global shard placement metadata (i.e., the
    /// mesh that the array was constructed over).
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        self.parts.metadata.mesh()
    }

    /// Returns whether this [`Array`] is _committed_ to its devices. This is the Ryft analogue of the commitment of
    /// JAX arrays (refer to the section on [placement of data on devices][jax-placement] in the JAX documentation).
    /// A committed array was placed on its devices on purpose, while an uncommitted array lives on its devices only
    /// because nothing chose any (e.g., a host value that was uploaded to the first device by default). Commitment
    /// is runtime state of the array, like its placement, and it is not part of its [`ArrayType`].
    ///
    /// An array is committed when one of the following holds, and it is uncommitted otherwise:
    ///
    ///   - **Explicit Devices:** It was constructed on a [`DeviceMesh`] that its caller provided (e.g., through
    ///     [`Self::from_addressable_buffers`], [`Self::from_host_shards`], or [`Self::from_host_buffer`] with a mesh).
    ///   - **Session Mesh:** It was created without a mesh and without array inputs by a [`Session`] that has a
    ///     [default mesh](Session::device_mesh), which then decides its placement. This covers
    ///     [`Self::from_host_buffer`] without a mesh, literal constants, and the results of eager operations without
    ///     array inputs (e.g., zeros, ones, fills, and `iota`). Without a default mesh, such arrays are uncommitted,
    ///     unless they live on more than one device (e.g., when [`Self::from_host_buffer`] without a mesh lays the
    ///     logical mesh of a sharding out over the first addressable devices).
    ///   - **Explicit Placement:** It was placed with a destination by [`Self::to_placement`],
    ///     [`Self::into_placement`], [`Session::place`], or a [`PlaceOperation`](ryft_core::PlaceOperation) with a
    ///     target. A host value that [`Session::place`] uploads without a destination is uncommitted.
    ///   - **Committed Inputs:** It is the result of an eager operation with a committed array input or a committed
    ///     array that the operation captures, or an output of a compiled function with a committed public argument.
    ///     The arrays that a compiled function captures never commit its outputs, whether it is executed directly or
    ///     called through [`CompiledFunction::call`](crate::CompiledFunction::call).
    ///   - **Eager Results on Several Devices:** It is the result of an eager operation, of a host upload through
    ///     [`Self::from_host_buffer`], or of a reshard that moves an array to the devices that an eager operation runs
    ///     on, and it lives on more than one device. An uncommitted eager result or upload therefore always lives on a
    ///     single device.
    ///   - **Compiled Functions:** It is an output of a compiled function that was compiled for more than one device,
    ///     or a public output of a compiled function that requests a placement explicitly anywhere (i.e., through a
    ///     sharding override, a [`PlaceOperation`](ryft_core::PlaceOperation) with a target, or a sharding constraint),
    ///     whether or not that request produces the output. The outputs of a function that was compiled for a single
    ///     device without such requests are committed only when the function has a committed public argument. The
    ///     same holds for the results of an eager operation with regions (e.g., a condition, a scan, or a loop), which
    ///     are committed when a region requests a placement explicitly, as the results of a compiled function with that
    ///     region would be. An eager operation without regions that is itself such a request (e.g., a sharding
    ///     constraint) leaves the commitment of its results to its inputs.
    ///
    /// Clones keep the commitment of their array, and so do the arrays that keep the storage of an array or
    /// copy it (e.g., a [`Session::place`] without a destination, also when it must not alias its input, and
    /// a [`RetainedArray`] turned back into an array).
    ///
    /// Commitment decides where an eager operation runs when its array inputs live on different devices. Uncommitted
    /// inputs follow committed ones, so when all committed inputs live on the same devices, the operation moves its
    /// uncommitted inputs to the mesh of the committed inputs and runs there. When inputs on different devices are
    /// all uncommitted, the operation moves them to the [default mesh](Session::device_mesh) of the session (or,
    /// without one, to the first addressable device), as JAX does, and its result stays uncommitted when it lives on a
    /// single device. A moved input keeps its sharding when that sharding is over the logical mesh of its destination,
    /// and it is replicated over that mesh when it is replicated. A move shares the buffers of the shards that stay on
    /// their devices. The operation still fails when two committed inputs live on different devices and when an
    /// uncommitted input cannot be moved (i.e., when it is sharded over another logical mesh, or when it or its
    /// destination is not fully addressable from this process). Compiled functions always run on the devices that
    /// they were compiled for. They move their captures and their uncommitted public arguments there, and they reject a
    /// public argument that is committed to other devices (i.e., to other physical devices or to the same ones in
    /// another order) with
    /// [`ExecutionError::CommittedInputDeviceMismatch`](crate::ExecutionError::CommittedInputDeviceMismatch).
    ///
    /// [jax-placement]: https://docs.jax.dev/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
    #[inline]
    pub fn is_committed(&self) -> bool {
        self.committed
    }

    /// Returns whether this process chose the devices of this [`Array`] independently of the other processes. Each
    /// process may hold the same value on devices of its own choice. A destination that names only a memory or a layout
    /// sets [commitment](Self::is_committed) while preserving the original device assignment.
    ///
    /// An array is locally assigned when no destination that names devices was ever given for it or for the
    /// arrays that it derives from. That is the case for the following arrays, and for no others:
    ///
    ///   - **Default Uploads:** It is a host value that was uploaded without a mesh by a [`Session`] without a
    ///     [default mesh](Session::device_mesh) (e.g., through [`Self::from_host_buffer`]), or a host value that
    ///     [`Session::place`] uploaded without a destination or with a destination that names only a memory or a
    ///     layout, which lands on the first addressable device of the client whether or not the session has a default
    ///     mesh.
    ///   - **Storage Placements:** It is the result of a placement of such an array that names only a memory or a
    ///     layout, or a copy of such an array on its own devices.
    ///   - **Eager Results:** It is the result of an eager operation whose array inputs are all locally assigned,
    ///     or of one without array inputs that runs where such an upload would (refer to
    ///     [`Session::assigns_placement_locally`]).
    ///
    /// A placement onto a device, a mesh, or a sharding, a construction on a mesh that the caller provides (e.g.,
    /// through [`Self::from_addressable_buffers`]), and direct execution of a compiled function all produce arrays
    /// whose devices every process knows. A call with no placement request follows its public arguments. Clones and
    /// [`RetainedArray`]s preserve local assignment. It is excluded from retained-array identity.
    ///
    /// Every process of a distributed program that applies the same functions to the same kinds of inputs therefore
    /// computes the same fact for its own array, which is what lets every process choose the same strategy when it
    /// places such an array on devices of several processes (refer to [`resharding`]).
    ///
    /// An array without the fact is treated as one whose devices every process knows, which holds only when every
    /// process of a distributed run names the same devices wherever a program names devices (e.g., the default mesh
    /// of a [`Session`], the mesh of a construction, or a [`Device`](ryft_core::Device) target). No process can check
    /// that, because each one sees only its own request, so it is a precondition. A choice of devices that differs
    /// between processes is expressible only by letting each process choose its own devices. For example, an array that
    /// each process placed on its own first addressable device with an explicit device target does not have the fact.
    /// A later placement over every process's devices then makes each process send its own value without any receiver.
    #[inline]
    pub(crate) fn is_locally_assigned(&self) -> bool {
        self.locally_assigned
    }

    /// Returns the [`ArrayShard`]s that make up this [`Array`].
    #[inline]
    pub fn shards(&self) -> &[ArrayShard] {
        self.parts.shards.as_slice()
    }

    /// Returns an [`Iterator`] over the _addressable_ [`ArrayShard`]s of this [`Array`].
    #[inline]
    pub fn addressable_shards(&self) -> impl Iterator<Item = &ArrayShard> {
        self.parts.shards.iter().filter(|shard| shard.is_addressable())
    }

    /// Returns the [`ArrayShard`] of this [`Array`] that is placed on the device with the provided
    /// [`DeviceId`], if such a shard exists.
    #[inline]
    pub fn device_shard(&self, device_id: DeviceId) -> Option<&ArrayShard> {
        let index = self.parts.metadata.shard_index_by_device().get(&device_id)?;
        self.parts.shards.get(*index)
    }

    /// Returns the _addressable_ [`ArrayShard`] of this [`Array`] that is placed on the device with the provided
    /// [`DeviceId`], if such a shard exists.
    #[inline]
    pub fn addressable_device_shard(&self, device_id: DeviceId) -> Option<&ArrayShard> {
        self.device_shard(device_id).filter(|shard| shard.is_addressable())
    }

    /// Returns the number of logical payload bytes that this [`Array`] occupies across all devices and processes.
    /// Bufferless [`DataType::Zero`] arrays occupy zero bytes.
    #[inline]
    pub fn size_in_bytes(&self) -> Result<usize, Error> {
        self.parts.metadata.r#type().dense_size_in_bytes()
    }

    /// Returns the whole-execution completion fence shared by the outputs of this array's producer, when it has one.
    #[inline]
    pub fn execution_fence(&self) -> Option<&ExecutionFence> {
        self.execution_fence.as_ref()
    }

    /// Returns the [`Session`] this [`Array`] belongs to.
    #[inline]
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Returns the PJRT [`Client`] that this [`Array`] lives on, which is the client of its session. Note that the
    /// client cannot be recovered from the addressable shard buffers themselves. [`Buffer`]s only store a raw handle to
    /// their owning PJRT client, by design, to avoid carrying the [`KeyValueStore`](ryft_pjrt::KeyValueStore) lifetime.
    /// [`Client`] is an owning wrapper whose [`Drop`] implementation destroys the underlying PJRT client, so no
    /// `&Client` can be materialized from that raw handle. The session is therefore threaded through explicitly at
    /// array construction time instead, and [`Buffer::is_owned_by`] validates that its client owns every addressable
    /// shard buffer.
    #[inline]
    pub fn client(&self) -> &'static Client<'static> {
        self.session.client()
    }

    /// Returns whether this array has no clones and every addressable shard buffer is uniquely owned by it. An array
    /// without any shard buffers (e.g., a bufferless zero-space array) is never reported as uniquely owned, so it is
    /// never donated. Donation is all or nothing across shards, so any shared buffer disables donation for the complete
    /// logical input. The answer can change after this function returns, because a [`Weak`](std::sync::Weak) handle
    /// of a shard buffer can still be upgraded, so it only selects the arrays that
    /// [`Self::into_exclusive_shard_buffers`] then tries to take exclusively.
    pub(crate) fn has_unique_shard_buffers(&self) -> bool {
        let mut buffer_count = 0usize;

        // Clones share their parts, and with them the shard buffers, so the array itself must also be unique.
        if Arc::strong_count(&self.parts) != 1 {
            return false;
        }

        for buffer in self.parts.shards.iter().filter_map(ArrayShard::buffer) {
            if Arc::strong_count(buffer) != 1 {
                return false;
            }
            buffer_count += 1;
        }

        buffer_count > 0
    }

    /// Consumes this array and returns the buffers of its shards in shard order, with [`None`] for a shard without
    /// one, together with whether the caller now holds every buffer exclusively, which is what donating them requires.
    /// That is the case only when this array has no clones and no other handle shares any of its buffers (refer to
    /// [`Self::has_unique_shard_buffers`]). Each buffer is then moved out of its shared allocation into a new one,
    /// which fails when another handle holds the buffer, and which a [`Weak`](std::sync::Weak) handle of the old
    /// allocation (e.g., one that a caller derived from [`ArrayShard::buffer`]) can no longer reach, so that no other
    /// handle can observe a buffer that an execution invalidates by donation. The decision and the hand-over are
    /// therefore one step. Otherwise, the buffers stay in their allocations, and the result is not exclusive. That is
    /// also the case when another handle upgrades a weak handle of a buffer while this function takes them, in which
    /// case the buffers that it took before that one have already left their allocations.
    pub(crate) fn into_exclusive_shard_buffers(self) -> (Vec<Option<Arc<Buffer<'static>>>>, bool) {
        if !self.has_unique_shard_buffers() {
            return (self.parts.shards.iter().map(|shard| shard.buffer().cloned()).collect(), false);
        }

        let parts = match Arc::try_unwrap(self.parts) {
            Ok(parts) => parts,
            Err(parts) => return (parts.shards.iter().map(|shard| shard.buffer().cloned()).collect(), false),
        };

        let mut exclusive = true;
        let buffers = parts
            .shards
            .into_iter()
            .map(|shard| {
                let buffer = shard.into_buffer()?;
                if !exclusive {
                    return Some(buffer);
                }
                match Arc::try_unwrap(buffer) {
                    Ok(buffer) => Some(Arc::new(buffer)),
                    Err(buffer) => {
                        exclusive = false;
                        Some(buffer)
                    }
                }
            })
            .collect();

        (buffers, exclusive)
    }

    /// Waits asynchronously for every addressable shard buffer to become ready, then for the whole-execution fence
    /// so errors not tied to an individual output buffer are surfaced as well. When the placement that produced this
    /// array sent shards to other processes, this function finally waits for those sends and returns the first error
    /// among them. Those sends complete only once every receiving process has issued its matching receive. None of
    /// these waits blocks the calling thread, so a task that awaits this function can share its thread with the task
    /// that issues such a receive. Calling this function does not copy data to the host.
    pub async fn ready(&self) -> Result<(), Error> {
        for shard in self.addressable_shards() {
            if let Some(buffer) = shard.buffer() {
                buffer.ready()?.await?;
            }
        }

        if let Some(fence) = &self.execution_fence {
            // Once all output buffers are ready, the launch completion event is normally complete as well,
            // so this preserves whole-execution errors without introducing a wait into device pipelines.
            fence.ready().await?;
        }

        if let Some(fence) = &self.send_fence {
            fence.ready().await?;
        }

        Ok(())
    }

    /// Blocks the current thread until this array and the whole execution that produced it are ready, and until the
    /// sends to other processes that the placement which produced this array issued have completed. It returns the
    /// first error of that work. This is an explicit synchronization boundary and does not copy data to the host.
    pub fn block_until_ready(&self) -> Result<(), Error> {
        if let Some(fence) = &self.execution_fence {
            fence.block_until_ready()?;
        }

        for shard in self.addressable_shards() {
            if let Some(buffer) = shard.buffer() {
                buffer.ready()?.r#await()?;
            }
        }

        if let Some(fence) = &self.send_fence {
            fence.block_until_ready()?;
        }

        Ok(())
    }
}

impl Debug for Array {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Array")
            .field("type", self.parts.metadata.r#type())
            .field("sharding", self.parts.metadata.sharding())
            .field("shards", &self.shards())
            .finish()
    }
}

impl Display for Array {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "array(type={})", self.parts.metadata.r#type())
    }
}

impl PartialEq for Array {
    fn eq(&self, other: &Self) -> bool {
        // `Array` equality is _storage identity_, not element-wise value equality. Arrays that share their storage are
        // equal. Otherwise, ordinary arrays are equal only when they have the same type and sharding and every shard
        // references the same underlying `Buffer`, as determined by `Arc` pointer identity, while bufferless zero-space
        // shards compare equal because their type admits exactly one value, and shards on other processes never compare
        // equal because they have no local storage to compare. Arrays holding equal ordinary data in distinct buffers
        // deliberately compare unequal, which is the conservative answer that analyses comparing flowing values need
        // and mirrors the `PartialEq` implementation of `Tracer`, which compares staging identity. Element-wise
        // equality would require device-to-host transfers and is deliberately not what this implements.
        let (left, right) = (&self.parts, &other.parts);
        Arc::ptr_eq(left, right)
            || (left.metadata.r#type() == right.metadata.r#type()
                && left.metadata.sharding() == right.metadata.sharding()
                && left.shards.len() == right.shards.len()
                && left.shards.iter().zip(right.shards.iter()).all(|(left, right)| {
                    match (left.storage(), right.storage()) {
                        (ArrayShardStorage::Zero, ArrayShardStorage::Zero) => true,
                        (ArrayShardStorage::Buffer(left), ArrayShardStorage::Buffer(right)) => Arc::ptr_eq(left, right),
                        _ => {
                            // Two shards on other processes have no local storage to compare, so they are only known
                            // to be the same storage when their arrays share their parts, which is checked first.
                            false
                        }
                    }
                }))
    }
}

impl Typed for Array {
    type Type = ArrayType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(self.parts.metadata.r#type())
    }
}

impl Placed for Array {
    fn placement(&self) -> Option<Placement> {
        // An array lives on the devices of its mesh, sharded by its runtime sharding (which may name auto mesh axes),
        // in the memory that its type records and with its layout when it has one. The sharding of an array is
        // always over the topology of its mesh, so the placement is always valid.
        let placement = Placement::new(self.mesh().clone(), self.sharding().clone())
            .unwrap()
            .with_memory(self.r#type().memory());
        Some(match self.layout() {
            Some(layout) => placement.with_layout(layout.clone()),
            None => placement,
        })
    }
}

impl Value for Array {
    // A concrete `Array` binds and executes every operation through its PJRT-backed `Session`. It uses
    // `ValueDomainDispatch`, so the blanket value capabilities in `ryft-core` (e.g., arithmetic, comparison, selection,
    // manipulation, and reductions) bind their operations through `domain()` and execute them eagerly, operation by
    // operation, through the session recovered below, and free transform entry points (e.g., `ryft_core::batch`)
    // recover the same session. Only capabilities without blanket implementations that bind operations (e.g.,
    // `Concretizable<bool>`, `WhilePredicate`, and the foreign `std::ops` sugar) get the direct implementations
    // that follow.

    type Dispatch = ValueDomainDispatch;
    type Domain = ProjectedContext<Session, ArrayType>;

    fn domain(&self) -> Self::Domain {
        // Recovers the eager `Session` this `Array` belongs to, which `Array::session` also returns, and with it
        // the session, compilation cache, effect scope, and compilation options that receiver-based operations execute
        // with. Association is per value, so receiver-based `x + y` and `y + x` may choose different scopes even when
        // their numerical values match. Start new scopes at task boundaries rather than inside expressions.
        ProjectedContext::new(self.session.clone())
    }
}

// Every eager value capability of `Array` follows the JAX-style operation-by-operation dispatch shape. The blanket
// capability implementations in `ryft-core` (e.g., arithmetic, logical, comparison, selection, manipulation, and
// reductions) bind each operation through `ryft_core::Value::domain`, which for a concrete `Array` recovers the
// PJRT-backed `Session` that the array belongs to. That session executes the operation through a single-operation
// program that its session caches per operation application, so a repeated eager operation skips program construction
// and lowering. The implementations that follow therefore only cover the capabilities that those blankets cannot cover,
// which are the capabilities that read values back to the host, such as `Concretizable` and `WhilePredicate`, and the
// foreign `std::ops` operator sugar, which the orphan rule requires to be implemented per type. Backend arrays opt into
// the shared shape-changing collective capabilities through `ShapeChangingCollectiveValue`.

impl Concretizable<bool> for Array {
    fn concretize(&self) -> Result<bool, ProgramError> {
        // Extracts a concrete scalar Rust Boolean by copying the value of a rank-0 Boolean `Array` to the host with
        // `Array::copy_to_host`. Arrays of higher rank or of non-Boolean types error because they cannot collapse to
        // a single Boolean, and arrays whose value the current process does not hold error because it cannot read
        // their payload.
        if self.r#type().rank() == 0 && self.data_type().is_boolean() {
            let bytes = self
                .copy_to_host_bytes()
                .map_err(|error| ProgramError::Concretization { message: error.to_string() })?;
            return Ok(bytes.iter().any(|byte| *byte != 0));
        }
        Err(ProgramError::Concretization {
            message: format!(
                "cannot extract a concrete boolean from a value of type `{}`; expected `bool[]`",
                self.r#type().as_ref(),
            ),
        })
    }
}

impl Concretizable<i128> for Array {
    fn concretize(&self) -> Result<i128, ProgramError> {
        // Decode through the CPU capability to preserve every signed and unsigned integer value. The host array keeps
        // the full type of this array, including its sharding, so that errors name the type that the caller sees.
        let bytes = self
            .copy_to_host_bytes()
            .map_err(|error| ProgramError::Concretization { message: error.to_string() })?;
        CoreArray::new(self.r#type().into_owned(), bytes)?.concretize()
    }
}

impl AssertionValue for Array {
    fn assertion_observation(&self) -> Result<String, ProgramError> {
        // The host array keeps the full type of this array, including its sharding, so that errors name the type
        // that the caller sees.
        let bytes = self
            .copy_to_host_bytes()
            .map_err(|error| ProgramError::Concretization { message: error.to_string() })?;
        CoreArray::new(self.r#type().into_owned(), bytes)?.assertion_observation()
    }

    fn assertion_array(&self) -> Result<Option<CoreArray>, ProgramError> {
        Ok(Some(self.copy_to_host().map_err(|error| ProgramError::Concretization { message: error.to_string() })?))
    }
}

impl DimensionSize<ArrayType> for Array {
    type Extent = usize;

    fn dimension_size<A: Into<Axis>>(&self, axis: A) -> Result<usize, ProgramError> {
        // Returns the concrete global extent of `axis` from this array's complete shard metadata. The descriptor
        // list includes remote shards, so taking the greatest exclusive slice end recovers the global extent without
        // synchronizing a device buffer. Constructing a `DimensionValue` validates that extent against the identity
        // and bounds of the selected declared axis before the host integer is returned.
        let operation = DimensionSizeOperation::new(self.r#type().as_ref(), axis)?;
        let extent = self.shards().iter().map(|shard| shard.slice()[operation.axis()].end).max().ok_or_else(|| {
            ProgramError::Concretization {
                message: format!(
                    "cannot read the extent of an array of type `{}` with no shard metadata",
                    self.r#type()
                ),
            }
        })?;
        DimensionValue::new(operation.output_type().clone(), extent)?;
        Ok(extent)
    }
}

impl DimensionFromScalar<DimensionValue, ArrayType> for Array {
    fn to_dimension(&self, output: DimensionVariable) -> Result<DimensionValue, ProgramError> {
        // Copies this rank-0 integer array to the host and grants its checked value first-class dimension authority.
        let operation = DimensionFromScalarOperation::new(output.into());
        let mut output_types = operation.infer_output_types(&[Type::Array(self.r#type().into_owned())], &[])?;
        let output_type = <&DimensionType>::try_from(&output_types.remove(0))?.clone();
        let bytes = self
            .copy_to_host_bytes()
            .map_err(|error| ProgramError::Concretization { message: error.to_string() })?;

        let invalid_extent = |value: String| ProgramError::InvalidArgument {
            message: format!(
                "`{}` scalar input must be a nonnegative host-representable extent but is {value}",
                operation.name(),
            ),
        };

        let extent = match self.data_type() {
            DataType::I8 => {
                let value = i8::from_ne_bytes(bytes.as_slice().try_into().unwrap());
                usize::try_from(value).map_err(|_| invalid_extent(value.to_string()))?
            }
            DataType::I16 => {
                let value = i16::from_ne_bytes(bytes.as_slice().try_into().unwrap());
                usize::try_from(value).map_err(|_| invalid_extent(value.to_string()))?
            }
            DataType::I32 => {
                let value = i32::from_ne_bytes(bytes.as_slice().try_into().unwrap());
                usize::try_from(value).map_err(|_| invalid_extent(value.to_string()))?
            }
            DataType::I64 => {
                let value = i64::from_ne_bytes(bytes.as_slice().try_into().unwrap());
                usize::try_from(value).map_err(|_| invalid_extent(value.to_string()))?
            }
            DataType::U8 => usize::from(u8::from_ne_bytes(bytes.as_slice().try_into().unwrap())),
            DataType::U16 => usize::from(u16::from_ne_bytes(bytes.as_slice().try_into().unwrap())),
            DataType::U32 => {
                let value = u32::from_ne_bytes(bytes.as_slice().try_into().unwrap());
                usize::try_from(value).map_err(|_| invalid_extent(value.to_string()))?
            }
            DataType::U64 => {
                let value = u64::from_ne_bytes(bytes.as_slice().try_into().unwrap());
                usize::try_from(value).map_err(|_| invalid_extent(value.to_string()))?
            }
            _ => unreachable!("`dimension_from_scalar` input type is validated before reading its payload"),
        };

        Ok(DimensionValue::new(output_type, extent)?)
    }
}

impl ShapeChangingCollectiveValue for Array {
    // Backend arrays opt into the shared homogeneous collective staging rules of `ryft-core`.
}

impl WhilePredicate for Array {
    // The batched while predicate semantics for `Array`, mirror the reference semantics of `ryft_core::Array`.
    // `WhilePredicate::any_true` reduces the whole Boolean payload with `or` by reading every shard back to the host,
    // and `WhilePredicate::select` broadcasts the predicate against the operands along its leading prefix axes on
    // device before selecting.

    fn any_true(&self) -> Result<bool, ProgramError> {
        if !self.data_type().is_boolean() {
            return Err(ProgramError::Concretization {
                message: format!("cannot use a value of type `{}` as a Boolean while predicate", self.r#type()),
            });
        }

        // The shards are read one at a time, so the whole execution that produced this array is observed first
        // to surface its errors, including those that are not tied to an individual shard buffer.
        if let Some(fence) = self.execution_fence() {
            fence
                .block_until_ready()
                .map_err(|error| ProgramError::Concretization { message: error.to_string() })?;
        }

        // The devices along unreduced axes hold candidates that the unreduced kind combines. For the kinds `sum` and
        // `max`, an element of the logical value is true when any candidate is, which the scan of every shard below
        // decides. For the kind `min`, all of its candidates have to be true, so the logical value comes from the host
        // copy, which combines them.
        if !self.sharding().unreduced_axes().is_empty() && self.sharding().unreduced_kind() == UnreducedKind::Min {
            let bytes = self
                .copy_to_host_bytes()
                .map_err(|error| ProgramError::Concretization { message: error.to_string() })?;
            return Ok(bytes.iter().any(|byte| *byte != 0));
        }

        // Reading every addressable shard covers all elements for sharded placements, because the shards partition the
        // payload, and for replicated placements, because `or` is idempotent over duplicates. Non-addressable shards
        // would leave remote elements unobserved, so they surface as concretization errors instead of a silently
        // partial answer.
        let mut addressable_count = 0usize;
        let mut any = false;
        for shard in self.addressable_shards() {
            addressable_count += 1;
            if !any {
                let bytes = shard
                    .copy_to_host_bytes()
                    .map_err(|error| ProgramError::Concretization { message: error.to_string() })?;
                any = bytes.iter().any(|byte| *byte != 0);
            }
        }

        if addressable_count != self.shards().len() {
            return Err(ProgramError::Concretization {
                message: format!(
                    "cannot decide a while predicate for an array of type `{}` whose shards are not all addressable \
                     from the current process",
                    self.r#type().as_ref(),
                ),
            });
        }

        Ok(any)
    }

    fn select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        if !self.data_type().is_boolean() || on_true.r#type() != on_false.r#type() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`select` requires a Boolean prefix-shaped predicate and congruent inputs, but got \
                     predicate `{}` with inputs `{}` and `{}`",
                    self.r#type().as_ref(),
                    on_true.r#type().as_ref(),
                    on_false.r#type().as_ref(),
                ),
            });
        }

        // Broadcast a predicate whose shape is a prefix of the operand shape up to that full shape, mapping its axes
        // to the leading output axes, so that the staged select receives operands with congruent shapes.
        let condition = if self.r#type().shape() == on_true.r#type().shape() {
            self.clone()
        } else {
            let output_type = on_true.r#type().with_element_type(DataType::Boolean);
            let output_axes = (0..self.r#type().rank()).collect::<Vec<_>>();
            self.broadcast(output_type, output_axes.as_slice())?
        };

        Select::select(&condition, on_true, on_false)
    }
}

// The `std::ops` operator traits are foreign, so the blanket tracer implementations in `ryft-core` cannot cover
// concrete backend values. These per-type implementations provide the ergonomic panicking sugar by delegating to
// the fallible `ryft` capabilities.

impl std::ops::Neg for Array {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Neg::neg(&self).expect("`neg` operation failed")
    }
}

impl std::ops::Add for Array {
    type Output = Self;

    #[inline]
    fn add(self, right: Self) -> Self {
        Add::add(&self, &right).expect("`add` operation failed")
    }
}

impl std::ops::Sub for Array {
    type Output = Self;

    #[inline]
    fn sub(self, right: Self) -> Self {
        Sub::sub(&self, &right).expect("`sub` operation failed")
    }
}

impl std::ops::Mul for Array {
    type Output = Self;

    #[inline]
    fn mul(self, right: Self) -> Self {
        Mul::mul(&self, &right).expect("`mul` operation failed")
    }
}

impl std::ops::Div for Array {
    type Output = Self;

    #[inline]
    fn div(self, right: Self) -> Self {
        Div::div(&self, &right).expect("`div` operation failed")
    }
}

impl std::ops::Not for Array {
    type Output = Self;

    #[inline]
    fn not(self) -> Self {
        Not::not(&self).expect("`not` operation failed")
    }
}

impl std::ops::BitAnd for Array {
    type Output = Self;

    #[inline]
    fn bitand(self, right: Self) -> Self {
        And::and(&self, &right).expect("`and` operation failed")
    }
}

impl std::ops::BitOr for Array {
    type Output = Self;

    #[inline]
    fn bitor(self, right: Self) -> Self {
        Or::or(&self, &right).expect("`or` operation failed")
    }
}

impl std::ops::BitXor for Array {
    type Output = Self;

    #[inline]
    fn bitxor(self, right: Self) -> Self {
        Xor::xor(&self, &right).expect("`xor` operation failed")
    }
}

/// Waits asynchronously for every [`Array`] leaf in `values` to become ready without copying data to the host.
#[inline]
pub async fn ready<Values: Parameterized<Array>>(values: &Values) -> Result<(), Error> {
    for array in values.parameters() {
        array.ready().await?;
    }
    Ok(())
}

/// Blocks until every [`Array`] leaf in `values` is ready without copying data to the host.
#[inline]
pub fn block_until_ready<Values: Parameterized<Array>>(values: &Values) -> Result<(), Error> {
    for array in values.parameters() {
        array.block_until_ready()?;
    }
    Ok(())
}

/// Immutable storage of an [`Array`] that its clones share.
struct ArrayParts {
    /// Type and runtime placement of the [`Array`]. It is shared because it does not depend on the device buffers, so
    /// the arrays that one compiled program produces for the same output share it, and so do the retained arrays and
    /// the live array descriptions that are created from the array.
    metadata: Arc<ArrayMetadata>,

    /// [`ArrayShard`]s that make up the [`Array`], in the order of the shard indices of [`Self::metadata`].
    shards: Vec<ArrayShard>,

    /// Registration of this storage as one logical array of the session that created it, which keeps the array counted
    /// as live in [`Session::array_statistics`] until the storage is dropped. It is the last field so that the shards
    /// and caches are released before the array stops counting as live.
    _registration: ArrayRegistration,
}

impl ArrayParts {
    /// Creates new [`ArrayParts`] and registers them as one logical array of `session`.
    /// Storage can only be created through this function, so every logical array is counted exactly once.
    fn new(session: &Session, metadata: Arc<ArrayMetadata>, shards: Vec<ArrayShard>) -> Self {
        let registration = session.array_tracker().register(&metadata, &shards);
        Self { metadata, shards, _registration: registration }
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    use pretty_assertions::assert_eq;

    use ryft_core::{
        Array as CoreArray, ArrayType, AssertionValue, Concretizable, DataType, Device, DeviceMesh, Dimension,
        DimensionBounds, DimensionError, DimensionFromScalar, DimensionSize, DimensionVariable, Error as CoreError,
        Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, Neg, Placed, Placement, ProgramError, Reference, Shape,
        Sharding, ShardingDimension, StaticShape, StridedLayout, TiledLayout, TypeError, Typed, UnreducedKind, Value,
        WhilePredicate,
    };
    use ryft_pjrt::{BufferType, Error as PjrtError, ExecutionFence};

    use crate::arrays::metadata::ArrayOutputPlan;
    use crate::arrays::shards::ArrayShard;
    use crate::arrays::statistics::ArrayStatistics;
    use crate::sessions::tests::{domain_mesh, shares_effect_scope};
    use crate::tests::{
        assert_send_sync, cpu_client, execution_client, logical_mesh_2x2, other_cpu_client, replicated_array,
        values_from_bytes, values_to_bytes,
    };
    use crate::types::DataTypeExtension;
    use crate::{Error, FromPjrt, RetainedArray, Session, ToPjrt};

    use super::{Array, ArrayError, block_until_ready, ready};

    /// Builds a rank-0 array whose sharding is unreduced along a two-device mesh axis `x`, with `kind` combining the
    /// separate values `device_bytes` that the two devices of the client of `session` hold.
    fn unreduced_scalar(
        session: &Session,
        data_type: DataType,
        kind: UnreducedKind,
        device_bytes: [&[u8]; 2],
    ) -> Array {
        let client = session.client();
        let devices = client.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let sharding = Sharding::replicated(logical_mesh, 0)
            .with_unreduced_axes(["x"])
            .unwrap()
            .with_unreduced_kind(kind)
            .unwrap();
        let r#type = ArrayType::new_static(data_type, []).with_sharding(sharding).unwrap();
        let buffers = device_bytes
            .iter()
            .zip(&devices)
            .map(|(bytes, device)| {
                client.buffer(*bytes, data_type.executable_buffer_type(), [], None, device.clone(), None).unwrap()
            })
            .collect();
        Array::from_addressable_buffers(session, r#type, mesh, None, buffers).unwrap()
    }

    /// Builds a rank-1 `f32` array whose only shard is on a device of another process, so that it has no storage in
    /// this process. The device is one that the client of `session` does not list, as a device of another process is.
    fn remote_array(session: &Session) -> Array {
        let devices = session.client().addressable_devices().unwrap();
        let local = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).max_by_key(Device::id).unwrap();
        let remote = Device::new(local.id() + 1, local.process_index() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![remote]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(logical_mesh, 1))
            .unwrap();
        Array::from_addressable_buffers(session, r#type, mesh, None, Vec::new()).unwrap()
    }

    /// Uploads the rank-1 `f32` array `[0, 1, 2, 3]` sharded along the axis `x` of a mesh over the two devices of the
    /// client of `session`, and returns it together with that mesh and its runtime sharding.
    fn sharded_vector(session: &Session) -> (Array, DeviceMesh, Sharding) {
        let mesh = domain_mesh(session.client(), "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array = Array::from_host_buffer(
            session,
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[0f32, 1., 2., 3.]),
        )
        .unwrap();
        (array, mesh, sharding)
    }

    #[test]
    fn test_array_is_send_and_sync() {
        assert_send_sync::<Array>();
    }

    #[test]
    fn test_array_reference_is_send_and_sync() {
        // Exact reference clones share immutable handle-local type and identity metadata, so the XLA reference family
        // relies on the array value and that metadata being safe for concurrent access.
        assert_send_sync::<Reference<Array>>();
    }

    #[test]
    fn test_array_clone() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));

        // Clones share the storage, the session, and the client of the array that they were created from.
        let clone = array.clone();
        assert!(Arc::ptr_eq(&array.parts, &clone.parts));
        assert_eq!(clone, array);
        assert!(clone.session().is_same_session(&session));
        assert!(std::ptr::eq(clone.client(), client));
        drop(clone);

        // Clones are handles of one logical array, so creating and dropping them leaves the session's counts unchanged.
        assert_eq!(session.array_statistics(), ArrayStatistics { live_arrays: 1, created_arrays: 1 });
    }

    #[test]
    fn test_array_from_addressable_buffers() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = domain_mesh(client, "x", devices.len());
        let values = CoreArray::vector(vec![1.0f32, 2.0]).unwrap();
        let array_type = values.r#type().into_owned();
        let shape = array_type.static_shape().unwrap();
        let length = shape.as_slice()[0];
        let data_type = array_type.data_type();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), shape.rank());
        let bytes = values.storage_bytes();
        let buffer_type = data_type.to_pjrt();
        let upload = |device| client.buffer(bytes, buffer_type, [length as u64], None, device, None).unwrap();

        // The full replicated layout is present even when some or all buffers are absent.
        for buffer_count in 0..=devices.len() {
            let buffers = devices.iter().take(buffer_count).cloned().map(&upload).collect();
            let array =
                Array::from_addressable_buffers(&session, array_type.clone(), mesh.clone(), None, buffers).unwrap();
            assert_eq!(array.shape(), shape);
            assert_eq!(array.data_type(), data_type);
            assert_eq!(array.sharding(), &sharding);
            assert_eq!(array.shards().len(), devices.len());
            assert!(array.shards().iter().all(|shard| shard.shape() == shape));
            assert_eq!(
                array.addressable_shards().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>(),
                vec![bytes; buffer_count],
            );
        }

        // Replication requires each buffer to hold the full logical value.
        let global_type = array_type.clone().with_shape(StaticShape::new(vec![length * devices.len()]));
        let buffers = devices.into_iter().map(upload).collect();
        let error = Array::from_addressable_buffers(&session, global_type.clone(), mesh, None, buffers).unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(
                error.as_ref(), ArrayError::BufferTypeMismatch { expected, actual }
                    if expected == &global_type && actual == &array_type
            ),
        ));
        assert_eq!(error.to_string(), format!("buffer has type `{array_type}`, but expected `{global_type}`"));
    }

    #[test]
    fn test_array_from_addressable_buffers_rejects_dynamic_shape() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 1)],
        )
        .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let dynamic = DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(10)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(dynamic.clone())]))
            .with_sharding(sharding)
            .unwrap();

        assert!(matches!(
            Array::from_addressable_buffers(&session, array_type, mesh, None, Vec::new()),
            Err(Error::Array(error)) if matches!(error.as_ref(),
                ArrayError::DynamicShape { shape } if shape == &Shape::new(vec![Dimension::Dynamic(dynamic)]),
            ),
        ));
    }

    #[test]
    fn test_array_from_addressable_buffers_rejects_mismatched_memory() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap()[0].clone();
        let mesh = domain_mesh(client, "x", 1);
        let r#type = ArrayType::scalar(DataType::I32);
        let values = values_to_bytes::<i32>(&[7]);

        // Reject a valid device buffer whose logical type falsely claims placement in pinned host memory.
        let buffer = client.buffer(values.as_slice(), BufferType::I32, [], None, device.clone(), None).unwrap();
        let actual = buffer.memory().unwrap().kind().unwrap().to_string();
        assert_ne!(actual, "pinned_host");
        assert!(matches!(
            Array::from_addressable_buffers(
                &session,
                r#type.with_memory(Memory::Host { pinned: true }),
                mesh,
                None,
                vec![buffer],
            ),
            Err(Error::Array(error)) if matches!(error.as_ref(),
                ArrayError::BufferMemoryMismatch {
                    device_id,
                    expected: Memory::Host { pinned: true },
                    actual: reported,
                } if *device_id == device.id().unwrap() && reported == &actual,
            ),
        ));
    }

    #[test]
    fn test_array_from_addressable_buffers_rejects_mismatched_layout() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap().remove(0);
        let mesh = domain_mesh(client, "x", 1);
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));

        // Reject a row-major buffer whose logical type declares the column-major layout.
        let bytes = values_to_bytes(&[1f64, 2., 3., 4., 5., 6.]);
        let buffer = client.buffer(bytes.as_slice(), BufferType::F64, [2, 3], None, device.clone(), None).unwrap();
        let error = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::F64, [2, 3]).with_layout(layout),
            mesh,
            None,
            vec![buffer],
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "invalid layout: buffer on device {} uses layout `tiled{{1,0}}` but its type declares `tiled{{0,1}}`",
                device.id().unwrap(),
            ),
        );
    }

    #[test]
    fn test_array_from_addressable_buffers_rejects_invalid_layout() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let rank_two = Layout::Strided(StridedLayout::new(vec![4, 4]));
        let message = "strided layout rank 2 does not match array rank 1";

        // An import without buffers meets no buffer that would reject a layout which cannot address the shape of its
        // type, so the layout is validated with the type, for the zero data type and for ordinary data types alike.
        for data_type in [DataType::Zero, DataType::F32] {
            let r#type = ArrayType::new_static(data_type, [4]).with_layout(rank_two.clone());
            let error = Array::from_addressable_buffers(&session, r#type, mesh.clone(), None, Vec::new()).unwrap_err();
            assert!(matches!(
                &error,
                Error::Core(error) if **error == CoreError::Type(TypeError::invalid(message)),
            ));
            assert_eq!(error.to_string(), message);
        }
    }

    #[test]
    fn test_array_from_addressable_buffers_rejects_misattributed_devices() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let misattributed = Device::new(devices[1].id(), devices[1].process_index() + 1);
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0], misattributed]).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [1]);
        let upload = || client.buffer([7u8].as_slice(), BufferType::U8, [1u64], None, client_devices[0].clone(), None);

        // The mesh assigns the second device of this process to another process. No process would hold its replica,
        // whether or not a buffer is supplied for it, so the import is rejected as a host upload to that mesh is.
        let message = format!(
            "device {} reports process index {}, but expected {}",
            devices[1].id(),
            devices[1].process_index(),
            devices[1].process_index() + 1,
        );
        for buffers in [vec![upload().unwrap()], Vec::new()] {
            let error =
                Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), None, buffers).unwrap_err();
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
            assert_eq!(error.to_string(), message);
        }

        // A device that the mesh assigns to this process must be one that the client addresses, whether or not a buffer
        // is supplied for it, because its shard would otherwise have no buffer in any process, as for a host upload.
        let missing = Device::new(devices[1].id() + 999, devices[1].process_index());
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0], missing]).unwrap();
        let (missing_id, missing_process) = (missing.id(), missing.process_index());
        let message = format!("device {missing_id} is not addressable by the PJRT client in process {missing_process}");
        for buffers in [vec![upload().unwrap()], Vec::new()] {
            let error =
                Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), None, buffers).unwrap_err();
            assert!(matches!(
                &error,
                Error::Array(error) if matches!(
                    error.as_ref(),
                    ArrayError::NonAddressableDevice { device_id, process_index }
                        if *device_id == missing.id() && *process_index == missing.process_index(),
                ),
            ));
            assert_eq!(error.to_string(), message);
        }

        // A device that the client does not know can belong to a process whose devices it cannot see, so its shard is
        // described without a buffer.
        let remote = Device::new(devices[1].id() + 1, devices[1].process_index() + 1);
        let mesh = DeviceMesh::new(logical_mesh, vec![devices[0], remote]).unwrap();
        let array = Array::from_addressable_buffers(&session, r#type, mesh, None, vec![upload().unwrap()]).unwrap();
        assert_eq!(array.addressable_shards().count(), 1);
        assert!(!array.shards()[1].is_addressable());
        assert!(array.shards()[1].buffer().is_none());
        assert_eq!(array.copy_to_host_bytes().unwrap(), [7]);
    }

    #[test]
    fn test_array_from_addressable_buffers_rejects_noncanonical_zero_carrier() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap()[0].clone();
        let device_id = device.id().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let r#type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let buffer = client.buffer(&[0u8, 1u8, 0u8], BufferType::Predicate, [3u64], None, device, None).unwrap();

        let error = Array::from_addressable_buffers(&session, r#type, mesh, None, vec![buffer]).unwrap_err();

        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::NonCanonicalZeroBuffer { device_id: actual } if *actual == device_id,
            ),
        ));
        assert_eq!(
            error.to_string(),
            format!("zero-space buffer on device {device_id} contains a nonzero physical carrier value"),
        );
    }

    #[test]
    fn test_array_from_addressable_buffers_discards_canonical_zero_carrier() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap()[0].clone();
        let mesh = domain_mesh(client, "x", 1);
        let r#type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let buffer = client.buffer(&[0u8; 3], BufferType::Predicate, [3u64], None, device, None).unwrap();

        let array = Array::from_addressable_buffers(&session, r#type, mesh, None, vec![buffer]).unwrap();

        assert_eq!(array.addressable_shards().count(), 1);
        assert!(array.addressable_shards().next().unwrap().buffer().is_none());
        assert_eq!(array.size_in_bytes().unwrap(), 0);
    }

    #[test]
    fn test_array_from_addressable_buffers_rejects_native_one_bit_carrier() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap().remove(0);
        let mesh = domain_mesh(client, "x", 1);
        for data_type in [DataType::I1, DataType::U1] {
            let array_type = ArrayType::scalar(data_type)
                .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
                .unwrap();
            let buffer = client.buffer(&[1], data_type.to_pjrt(), [], None, device.clone(), None).unwrap();
            assert!(matches!(
                Array::from_addressable_buffers(&session, array_type, mesh.clone(), None, vec![buffer]),
                Err(Error::Array(error)) if matches!(error.as_ref(),
                    ArrayError::BufferTypeMismatch { expected, actual }
                        if expected == &ArrayType::scalar(DataType::Boolean) && actual == &ArrayType::scalar(data_type),
                ),
            ));
        }
    }

    #[test]
    fn test_array_from_addressable_buffers_rejects_buffers_of_another_client() {
        let client = cpu_client(1);
        let other_client = other_cpu_client(1);
        let other_session = Session::builder().client(other_client).build().unwrap();
        let device = client.addressable_devices().unwrap().remove(0);
        let mesh = domain_mesh(client, "x", 1);
        let r#type = ArrayType::scalar(DataType::F32);

        // A session whose client does not own a buffer is rejected, even when that client was created from the same
        // plugin with identical options.
        let buffer = client.buffer(&1f32.to_ne_bytes(), BufferType::F32, [], None, device.clone(), None).unwrap();
        let error = Array::from_addressable_buffers(&other_session, r#type.clone(), mesh.clone(), None, vec![buffer])
            .unwrap_err();
        let message = format!(
            "the session's client does not own the addressable shard buffer for device {}",
            device.id().unwrap(),
        );
        assert!(matches!(
            &error,
            Error::Pjrt(PjrtError::InvalidArgument { message: actual, .. }) if actual == &message,
        ));
        assert_eq!(error.to_string(), message);

        // An array without buffers has no storage to validate, so it belongs to the session that it was created in.
        let array = Array::from_addressable_buffers(&other_session, r#type, mesh, None, Vec::new()).unwrap();
        assert_eq!(array.addressable_shards().count(), 0);
        assert!(std::ptr::eq(array.client(), other_client));
    }

    #[test]
    fn test_array_from_addressable_buffers_with_runtime_sharding() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh_2x2(), devices.clone()).unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(4)]));
        let shard_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(shard_index, device)| {
                let base = (shard_index * 10) as f32;
                let bytes = values_to_bytes::<f32>(&[base, base + 1.0, base + 2.0, base + 3.0]);
                client.buffer(bytes.as_slice(), BufferType::F32, [2u64, 2u64], None, device.clone(), None).unwrap()
            })
            .collect::<Vec<_>>();

        // Placement over auto mesh axes is runtime state. The array type records the projection onto explicit and
        // manual axes.
        let array = Array::from_addressable_buffers(
            &session,
            array_type.clone(),
            mesh.clone(),
            Some(sharding.clone()),
            shard_buffers,
        )
        .unwrap();

        assert_eq!(
            array.r#type().as_ref(),
            &array_type.clone().with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2)).unwrap(),
        );
        assert_eq!(array.sharding(), &sharding);
        assert_eq!(
            array.shards().iter().map(|shard| (shard.device(), shard.shape())).collect::<Vec<_>>(),
            devices.iter().map(|device| (*device, StaticShape::new(vec![2, 2]))).collect::<Vec<_>>(),
        );
        assert_eq!(array.addressable_shards().count(), 4);

        // A type sharding that disagrees with the placement over the non-auto mesh axes is rejected.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let explicit_sharding = Sharding::new(
            explicit_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let replicated_type = array_type.with_sharding(Sharding::replicated(explicit_mesh.clone(), 2)).unwrap();
        let explicit_device_mesh = DeviceMesh::new(explicit_mesh, devices).unwrap();
        let error = Array::from_addressable_buffers(
            &session,
            replicated_type.clone(),
            explicit_device_mesh,
            Some(explicit_sharding.clone()),
            Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::PlacementTypeMismatch { type_sharding, sharding }
                    if Some(type_sharding) == replicated_type.sharding() && sharding == &explicit_sharding,
            ),
        ));
        assert_eq!(
            error.to_string(),
            "array type sharding `{mesh<['x'=4:explicit]>, [{}, {}]}` does not match the placement \
             `{mesh<['x'=4:explicit]>, [{'x'}, {}]}` over the non-auto mesh axes",
        );
    }

    #[test]
    fn test_array_from_canonical_addressable_buffers() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap().remove(0);
        let mesh = domain_mesh(client, "x", 1);

        // Ordinary buffers are validated and kept, as they are by `Array::from_addressable_buffers`.
        let bytes = values_to_bytes(&[1f32, 2.]);
        let buffer = client.buffer(bytes.as_slice(), BufferType::F32, [2u64], None, device.clone(), None).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let array =
            Array::from_canonical_addressable_buffers(&session, r#type, None, mesh.clone(), vec![buffer]).unwrap();
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1f32, 2.]);
        let buffer = client.buffer(bytes.as_slice(), BufferType::F32, [2u64], None, device.clone(), None).unwrap();
        assert!(matches!(
            Array::from_canonical_addressable_buffers(
                &session,
                ArrayType::new_static(DataType::F32, [3]),
                None,
                mesh.clone(),
                vec![buffer],
            ),
            Err(Error::Array(error)) if matches!(error.as_ref(),
                ArrayError::BufferTypeMismatch { expected, actual }
                    if expected == &ArrayType::new_static(DataType::F32, [3])
                        && actual == &ArrayType::new_static(DataType::F32, [2]),
            ),
        ));

        // The carrier of a zero-space array is trusted to be canonical, so it is discarded without being read,
        // even when its contents are not.
        let buffer = client.buffer(&[0u8, 1u8, 0u8], BufferType::Predicate, [3u64], None, device, None).unwrap();
        let r#type = ArrayType::new_static(DataType::Zero, [3]);
        let array = Array::from_canonical_addressable_buffers(&session, r#type, None, mesh, vec![buffer]).unwrap();
        assert_eq!(array.addressable_shards().count(), 1);
        assert!(array.addressable_shards().next().unwrap().buffer().is_none());
    }

    #[test]
    fn test_array_zero() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);

        // A zero-space array is created without any buffers, and its shards are still addressable.
        let array = Array::zero(&session, ArrayType::new_static(DataType::Zero, [3]), None, mesh.clone()).unwrap();
        assert_eq!(array.data_type(), DataType::Zero);
        assert_eq!(array.shape(), StaticShape::new(vec![3]));
        assert_eq!(array.shards().len(), 1);
        assert!(array.shards().iter().all(|shard| shard.is_addressable() && shard.buffer().is_none()));

        // Every other element type needs buffers.
        assert!(matches!(
            Array::zero(&session, ArrayType::new_static(DataType::F32, [3]), None, mesh),
            Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))
                if message == "bufferless zero-space construction requires zero element type but got `f32`",
        ));
    }

    #[test]
    fn test_array_from_execution_output() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_ids = devices.iter().map(Device::id).collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh_2x2(), devices).unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(4)]));
        let shard_buffers = || {
            client_devices
                .iter()
                .enumerate()
                .map(|(shard_index, device)| {
                    let base = (shard_index * 10) as f32;
                    let bytes = values_to_bytes::<f32>(&[base, base + 1.0, base + 2.0, base + 3.0]);
                    client.buffer(bytes.as_slice(), BufferType::F32, [2u64, 2u64], None, device.clone(), None).unwrap()
                })
                .collect::<Vec<_>>()
        };
        let plan =
            ArrayOutputPlan::new(array_type.clone(), Some(sharding.clone()), &mesh, device_ids.as_slice()).unwrap();

        // Planned outputs describe the same placement as the validating constructor, and every output built from one
        // plan shares its metadata. Their types record none of the placement over the auto mesh axes.
        let expected = Array::from_addressable_buffers(
            &session,
            array_type.clone(),
            mesh.clone(),
            Some(sharding.clone()),
            shard_buffers(),
        )
        .unwrap();
        let output = Array::from_execution_output(&session, &plan, shard_buffers(), ExecutionFence::new(Vec::new()));
        let other_output =
            Array::from_execution_output(&session, &plan, shard_buffers(), ExecutionFence::new(Vec::new()));
        assert_eq!(output.r#type(), expected.r#type());
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(
            output.shards().iter().map(ArrayShard::descriptor).collect::<Vec<_>>(),
            expected.shards().iter().map(ArrayShard::descriptor).collect::<Vec<_>>(),
        );
        assert!(output.shards().iter().all(ArrayShard::is_addressable));
        assert!(Arc::ptr_eq(&output.parts.metadata, &other_output.parts.metadata));
        output.block_until_ready().unwrap();
        let shard_bytes = output.device_shard(device_ids[3]).unwrap().copy_to_host_bytes().unwrap();
        assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), vec![30.0, 31.0, 32.0, 33.0]);

        // Zero-space outputs need no returned carriers.
        let zero_type = array_type.with_data_type(DataType::Zero);
        let zero_plan = ArrayOutputPlan::new(zero_type, Some(sharding), &mesh, device_ids.as_slice()).unwrap();
        let zero_output =
            Array::from_execution_output(&session, &zero_plan, Vec::new(), ExecutionFence::new(Vec::new()));
        assert_eq!(zero_output.r#type().data_type(), DataType::Zero);
        assert_eq!(zero_output.r#type().sharding(), output.r#type().sharding());
        assert!(zero_output.shards().iter().all(|shard| shard.is_addressable() && shard.buffer().is_none()));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "does not match its planned placement")]
    fn test_array_from_execution_output_rejects_misplaced_buffers_in_debug_builds() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let device_ids = client_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let plan = ArrayOutputPlan::new(array_type, Some(sharding), &mesh, device_ids.as_slice()).unwrap();

        // Release builds trust XLA to return buffers in addressable device order, while debug builds still validate
        // that every returned buffer matches its planned shard.
        let buffers = client_devices
            .iter()
            .rev()
            .map(|device| {
                let bytes = values_to_bytes::<f32>(&[0.0, 1.0]);
                client.buffer(bytes.as_slice(), BufferType::F32, [2u64], None, device.clone(), None).unwrap()
            })
            .collect::<Vec<_>>();
        drop(Array::from_execution_output(&session, &plan, buffers, ExecutionFence::new(Vec::new())));
    }

    #[test]
    fn test_array_with_execution_fence() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);

        // Attaching the fence of a pending execution never waits, and clones share that fence.
        let (event, promise) = client.event(()).unwrap();
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]))
            .with_execution_fence(ExecutionFence::new(vec![event]));
        let clone = array.clone();
        assert!(matches!(array.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))));
        assert!(matches!(clone.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))));
        assert!(array.send_fence.is_none());
        promise.set(None).unwrap();
        array.block_until_ready().unwrap();
        assert!(matches!(array.execution_fence().map(ExecutionFence::is_ready), Some(Ok(true))));
        assert!(matches!(clone.execution_fence().map(ExecutionFence::is_ready), Some(Ok(true))));
    }

    #[test]
    fn test_array_with_send_fence() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let mut context = Context::from_waker(Waker::noop());

        // Attaching the fence of pending sends never waits, and clones and retained copies share that fence.
        let (event, promise) = client.event(()).unwrap();
        let array = Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(&[1f32, 2.]))
            .unwrap()
            .with_send_fence(ExecutionFence::new(vec![event]));
        let clone = array.clone();
        let retained = RetainedArray::new(&array).to_array(&session).unwrap();
        let pending = [&array, &clone, &retained]
            .map(|array| matches!(array.send_fence.as_ref().map(ExecutionFence::is_ready), Some(Ok(false))));
        promise.set(Some(PjrtError::invalid_argument("cross-host send failed"))).unwrap();
        assert_eq!(pending, [true; 3]);
        assert!(array.execution_fence().is_none());

        // Every readiness function of every array that shares the fence surfaces the failure of the sends. The
        // asynchronous functions complete within one poll because the buffers and the fence are already complete.
        for array in [&array, &clone, &retained] {
            assert!(matches!(
                array.block_until_ready(),
                Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. })) if message == "cross-host send failed",
            ));
            assert!(matches!(
                pin!(array.ready()).poll(&mut context),
                Poll::Ready(Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. })))
                    if message == "cross-host send failed",
            ));
        }
        assert!(matches!(
            block_until_ready(&vec![array.clone()]),
            Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. })) if message == "cross-host send failed",
        ));
        assert!(matches!(
            pin!(ready(&vec![array.clone()])).poll(&mut context),
            Poll::Ready(Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. })))
                if message == "cross-host send failed",
        ));

        // Pending sends leave the asynchronous readiness functions pending without blocking the thread,
        // and successful sends leave every readiness function successful.
        let (event, promise) = client.event(()).unwrap();
        let array = Array::from_host_buffer(&session, r#type, mesh, None, values_to_bytes(&[1f32, 2.]))
            .unwrap()
            .with_send_fence(ExecutionFence::new(vec![event]));
        assert!(matches!(pin!(array.ready()).poll(&mut context), Poll::Pending));
        assert!(matches!(pin!(ready(&vec![array.clone()])).poll(&mut context), Poll::Pending));
        promise.set(None).unwrap();
        assert!(matches!(array.block_until_ready(), Ok(())));
        assert!(matches!(block_until_ready(&vec![array.clone()]), Ok(())));
        assert!(matches!(pin!(array.ready()).poll(&mut context), Poll::Ready(Ok(()))));
        assert!(matches!(pin!(ready(&vec![array.clone()])).poll(&mut context), Poll::Ready(Ok(()))));
    }

    #[test]
    fn test_array_with_locally_assigned() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let uploaded = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        assert!(uploaded.is_locally_assigned());

        // The fact can be cleared and set again on an array whose shards this process addresses, and the commitment
        // of the array stays as it is.
        let cleared = uploaded.with_locally_assigned(false);
        assert!(!cleared.is_locally_assigned());
        assert!(!cleared.is_committed());
        assert!(cleared.with_locally_assigned(true).is_locally_assigned());

        // The devices of another process are never ones that this process chose on its own.
        assert!(!remote_array(&session).with_locally_assigned(true).is_locally_assigned());
    }

    #[test]
    fn test_array_with_session() {
        let client = cpu_client(1);
        let other_client = other_cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        let original = array.clone();

        // A handle of the same session with a new effect scope switches the scope while sharing storage.
        let fork = session.with_new_effect_scope();
        let scoped = array.clone().with_session(&fork).unwrap();
        assert!(scoped.session().is_same_session(&session));
        assert!(shares_effect_scope(scoped.session(), &fork));
        assert!(!shares_effect_scope(scoped.session(), original.session()));
        assert_eq!(scoped, original);

        // A different session on the same client uses its own compilation cache and effect scope.
        let other_session = Session::builder().client(client).build().unwrap();
        let switched = array.with_session(&other_session.with_new_effect_scope()).unwrap();
        assert!(switched.session().is_same_session(&other_session));
        assert!(std::ptr::eq(switched.client(), client));
        assert_eq!(switched, original);

        // Receiver-based operations use the other session for compilation and execution.
        let negated = Neg::neg(&switched).unwrap();
        assert_eq!(negated.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![-1f32]);
        assert_eq!(other_session.cached_program_count(), 1);
        assert_eq!(session.cached_program_count(), 0);

        // Existing clones keep their original session and scope.
        assert!(original.session().is_same_session(&session));
        assert!(shares_effect_scope(original.session(), &session));

        // `with_session` rejects a session of a different client, even when no numerical buffer could establish
        // ownership (e.g., for bufferless zero-space arrays).
        let other_client_session = Session::builder().client(other_client).build().unwrap();
        let zero = Array::from_host_buffer(&session, ArrayType::scalar(DataType::Zero), None, None, []).unwrap();
        for array in [original, zero] {
            assert!(matches!(
                array.with_session(&other_client_session),
                Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))
                    if message == "cannot associate an array with a session of a different PJRT client",
            ));
        }
    }

    #[test]
    fn test_array_data_type() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        assert_eq!(array.data_type(), DataType::F32);
    }

    #[test]
    fn test_array_shape() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        assert_eq!(array.shape(), StaticShape::new(vec![2]));
    }

    #[test]
    fn test_array_layout() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        assert_eq!(array.layout(), None);

        // The layout of the type of an array is returned as it is.
        let layout = Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()));
        let r#type = ArrayType::new_static(DataType::F32, [2, 2]).with_layout(layout.clone());
        let array = Array::from_addressable_buffers(&session, r#type, mesh, None, Vec::new()).unwrap();
        assert_eq!(array.layout(), Some(&layout));
    }

    #[test]
    fn test_array_sharding() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();

        // The runtime sharding includes the placement over auto mesh axes, which the type of the array omits.
        let (array, mesh, sharding) = sharded_vector(&session);
        assert_eq!(array.sharding(), &sharding);
        assert_ne!(Some(array.sharding()), array.r#type().sharding());

        // Arrays are replicated unless a sharding is requested.
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        assert_eq!(array.sharding(), &Sharding::replicated(mesh.logical_mesh().clone(), 1));
    }

    #[test]
    fn test_array_mesh() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (array, mesh, _) = sharded_vector(&session);
        assert_eq!(array.mesh(), &mesh);
    }

    #[test]
    fn test_array_is_committed() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let bytes = values_to_bytes(&[1f32, 2.]);

        // Arrays constructed on a mesh that the caller provides are committed to its devices.
        let uploaded = Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, bytes.as_slice()).unwrap();
        assert!(uploaded.is_committed());
        let shards = mesh.devices().iter().map(|device| (device.id(), bytes.as_slice()));
        assert!(
            Array::from_host_shards(&session, r#type.clone(), mesh.clone(), None, shards)
                .unwrap()
                .is_committed()
        );
        let wrapped =
            Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), None, Vec::new()).unwrap();
        assert!(wrapped.is_committed());

        // A host upload without a mesh is committed only when the mesh of its session decides its placement.
        let uploaded = Array::from_host_buffer(&session, r#type.clone(), None, None, bytes.as_slice()).unwrap();
        assert!(!uploaded.is_committed());
        let meshed_session = session.with_device_mesh(mesh.clone());
        assert!(
            Array::from_host_buffer(&meshed_session, r#type.clone(), None, None, bytes.as_slice())
                .unwrap()
                .is_committed()
        );
        let zero_type = ArrayType::new_static(DataType::Zero, [2]);
        assert!(!Array::from_host_buffer(&session, zero_type.clone(), None, None, []).unwrap().is_committed());
        assert!(Array::from_host_buffer(&session, zero_type, mesh.clone(), None, []).unwrap().is_committed());

        // Clones, handles of other sessions, and retained arrays keep the commitment of their array.
        assert!(!uploaded.clone().is_committed());
        assert!(!uploaded.clone().with_session(&meshed_session).unwrap().is_committed());
        assert!(!RetainedArray::new(&uploaded).to_array(&meshed_session).unwrap().is_committed());
        assert!(!uploaded.clone().with_committed(true).with_committed(false).is_committed());
        assert!(uploaded.with_committed(true).is_committed());
    }

    #[test]
    fn test_array_is_locally_assigned() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let meshed_session = session.with_device_mesh(domain_mesh(client, "x", 2));
        let mesh = domain_mesh(client, "x", 2);
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let bytes = values_to_bytes(&[1f32, 2.]);

        // A host upload without a mesh of the caller or of the session lives on devices that this process chose on its
        // own, also when its sharding lays it out over two devices, which commits it to them.
        let uploaded = Array::from_host_buffer(&session, r#type.clone(), None, None, bytes.as_slice()).unwrap();
        assert!(uploaded.is_locally_assigned());
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let split = Array::from_host_buffer(&session, r#type.clone(), None, Some(sharding), bytes.as_slice()).unwrap();
        assert!(split.is_committed());
        assert!(split.is_locally_assigned());
        let zero_type = ArrayType::new_static(DataType::Zero, [2]);
        let zero = Array::from_host_buffer(&session, zero_type.clone(), None, None, []).unwrap();
        assert!(zero.is_locally_assigned());

        // A mesh of the caller or of the session names the devices, which every process then knows.
        let on_mesh = Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, bytes.as_slice()).unwrap();
        assert!(!on_mesh.is_locally_assigned());
        let zero = Array::from_host_buffer(&session, zero_type, mesh.clone(), None, []).unwrap();
        assert!(!zero.is_locally_assigned());
        let shards = mesh.devices().iter().map(|device| (device.id(), bytes.as_slice()));
        let from_shards = Array::from_host_shards(&session, r#type.clone(), mesh.clone(), None, shards).unwrap();
        assert!(!from_shards.is_locally_assigned());
        let wrapped = Array::from_addressable_buffers(&session, r#type.clone(), mesh, None, Vec::new()).unwrap();
        assert!(!wrapped.is_locally_assigned());
        let meshed = Array::from_host_buffer(&meshed_session, r#type, None, None, bytes.as_slice()).unwrap();
        assert!(!meshed.is_locally_assigned());

        // Clones, handles of other sessions, and retained arrays keep the fact of their array.
        assert!(uploaded.clone().is_locally_assigned());
        assert!(uploaded.clone().with_session(&meshed_session).unwrap().is_locally_assigned());
        assert!(RetainedArray::new(&uploaded).to_array(&meshed_session).unwrap().is_locally_assigned());
        assert!(!RetainedArray::new(&on_mesh).to_array(&session).unwrap().is_locally_assigned());
    }

    #[test]
    fn test_array_shards() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();

        // There is one shard per mesh device, in the order of the mesh devices, whether or not it is addressable.
        let (array, mesh, _) = sharded_vector(&session);
        assert_eq!(
            array.shards().iter().map(|shard| (shard.device(), shard.shape())).collect::<Vec<_>>(),
            mesh.devices().iter().map(|device| (*device, StaticShape::new(vec![2]))).collect::<Vec<_>>(),
        );
        assert_eq!(
            array.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>(),
            vec![values_to_bytes(&[0f32, 1.]), values_to_bytes(&[2f32, 3.])],
        );
        let remote = remote_array(&session);
        assert_eq!(remote.shards().len(), 1);
        assert!(!remote.shards()[0].is_addressable());
    }

    #[test]
    fn test_array_addressable_shards() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (array, mesh, _) = sharded_vector(&session);
        assert_eq!(array.addressable_shards().map(ArrayShard::device).collect::<Vec<_>>(), mesh.devices().to_vec(),);

        // Shards on devices of other processes are skipped.
        assert_eq!(remote_array(&session).addressable_shards().count(), 0);
    }

    #[test]
    fn test_array_device_shard() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (array, mesh, _) = sharded_vector(&session);
        for device in mesh.devices() {
            assert_eq!(array.device_shard(device.id()).map(ArrayShard::device), Some(*device));
        }

        // Devices of other processes have shards as well, while devices that are not in the mesh have none.
        let remote = remote_array(&session);
        let remote_device = remote.mesh().devices()[0];
        assert_eq!(remote.device_shard(remote_device.id()).map(ArrayShard::is_addressable), Some(false));
        assert!(array.device_shard(mesh.devices().iter().map(Device::id).max().unwrap() + 1).is_none());
    }

    #[test]
    fn test_array_addressable_device_shard() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (array, mesh, _) = sharded_vector(&session);
        for device in mesh.devices() {
            assert_eq!(array.addressable_device_shard(device.id()).map(ArrayShard::device), Some(*device));
        }

        // Neither devices of other processes nor devices that are not in the mesh have addressable shards.
        let remote = remote_array(&session);
        assert!(remote.addressable_device_shard(remote.mesh().devices()[0].id()).is_none());
        assert!(array.addressable_device_shard(mesh.devices().iter().map(Device::id).max().unwrap() + 1).is_none());
    }

    #[test]
    fn test_array_size_in_bytes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();

        // The size covers the whole logical array rather than the shards of one device.
        let (array, mesh, _) = sharded_vector(&session);
        assert_eq!(array.size_in_bytes().unwrap(), 16);

        // Zero-space arrays have no payload.
        let zero = Array::zero(&session, ArrayType::new_static(DataType::Zero, [4]), None, mesh).unwrap();
        assert_eq!(zero.size_in_bytes().unwrap(), 0);
    }

    #[test]
    fn test_array_execution_fence() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        assert!(array.execution_fence().is_none());
        let array = array.with_execution_fence(ExecutionFence::new(Vec::new()));
        assert!(matches!(array.execution_fence().map(ExecutionFence::is_ready), Some(Ok(true))));
    }

    #[test]
    fn test_array_session() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        assert!(array.session().is_same_session(&session));
        assert!(shares_effect_scope(array.session(), &session));

        // An array belongs to the session handle that it was created with, including the effect scope of that handle.
        let fork = session.with_new_effect_scope();
        let array = replicated_array(&fork, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        assert!(array.session().is_same_session(&session));
        assert!(shares_effect_scope(array.session(), &fork));
        assert!(!shares_effect_scope(array.session(), &session));
    }

    #[test]
    fn test_array_client() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        assert!(std::ptr::eq(array.client(), client));
        assert!(std::ptr::eq(array.client(), session.client()));
    }

    #[test]
    fn test_array_has_unique_shard_buffers() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));

        // Clones share their storage, so an array is only unique while it has no clones.
        assert!(array.has_unique_shard_buffers());
        let clone = array.clone();
        assert!(!array.has_unique_shard_buffers());
        assert!(!clone.has_unique_shard_buffers());
        drop(clone);
        assert!(array.has_unique_shard_buffers());

        // An array without any buffers is never unique, so that it is never donated.
        let zero = Array::zero(&session, ArrayType::new_static(DataType::Zero, [2]), None, mesh).unwrap();
        assert!(!zero.has_unique_shard_buffers());
    }

    #[test]
    fn test_array_ready() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        let mut context = Context::from_waker(Waker::noop());

        // Once the buffers of an array are ready, waiting for it completes within one poll.
        array.block_until_ready().unwrap();
        assert!(matches!(pin!(array.ready()).poll(&mut context), Poll::Ready(Ok(()))));

        // A pending execution leaves the wait pending without blocking the thread, and its failure is returned.
        let (event, promise) = client.event(()).unwrap();
        let array = array.with_execution_fence(ExecutionFence::new(vec![event]));
        assert!(matches!(pin!(array.ready()).poll(&mut context), Poll::Pending));
        promise.set(Some(PjrtError::invalid_argument("execution failed"))).unwrap();
        assert!(matches!(
            pin!(array.ready()).poll(&mut context),
            Poll::Ready(Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))) if message == "execution failed",
        ));
    }

    #[test]
    fn test_array_block_until_ready() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        assert!(matches!(array.block_until_ready(), Ok(())));

        // The failure of the execution that produced an array is returned even though its buffers are ready.
        let (event, promise) = client.event(()).unwrap();
        let array = array.with_execution_fence(ExecutionFence::new(vec![event]));
        promise.set(Some(PjrtError::invalid_argument("execution failed"))).unwrap();
        assert!(matches!(
            array.block_until_ready(),
            Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. })) if message == "execution failed",
        ));
    }

    #[test]
    fn test_array_debug() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let shape = Shape::new(vec![Dimension::Static(2)]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        // The device belongs to another process, whose devices the client does not list.
        let device_mesh = DeviceMesh::new(logical_mesh, vec![Device::new(1, 1)]).unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array_type = ArrayType::new(DataType::F32, shape);
        let array =
            Array::from_addressable_buffers(&session, array_type, device_mesh, Some(sharding), Vec::new()).unwrap();

        // The type records no placement over the auto mesh axis, while the runtime placement does.
        assert_eq!(
            format!("{array:?}"),
            concat!(
                "Array { type: ArrayType { data_type: F32, shape: Shape { dimensions: [Static(2)] }, layout: None, ",
                "sharding: Some(Sharding { mesh: LogicalMesh { axes: [MeshAxis { name: \"x\", size: 1, type: Auto }] }, ",
                "dimensions: [Replicated], unreduced_axes: {}, unreduced_kind: Sum, reduced_axes: {}, ",
                "varying_manual_axes: {} }), memory: Device }, ",
                "sharding: Sharding { mesh: LogicalMesh { axes: [MeshAxis { name: \"x\", size: 1, type: Auto }] }, ",
                "dimensions: [Sharded([\"x\"])], unreduced_axes: {}, unreduced_kind: Sum, reduced_axes: {}, ",
                "varying_manual_axes: {} }, shards: [ArrayShard { index: 0, device_id: 1, process_index: 1, ",
                "shape: StaticShape { dimensions: [2] }, is_addressable: false }] }",
            ),
        );
    }

    #[test]
    fn test_array_display() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        assert_eq!(array.to_string(), "array(type=f32[2][sharding={mesh<['x'=1:auto]>, [{}]}])");
    }

    #[test]
    fn test_array_eq() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);

        // Equality is the identity of the storage rather than of the elements, so arrays that hold equal elements in
        // different buffers are not equal.
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        let other = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        assert_eq!(array, array.clone());
        assert_ne!(array, other);

        // Zero-space arrays of one type have the same single value, so they are equal.
        let zero_type = ArrayType::new_static(DataType::Zero, [2]);
        assert_eq!(
            Array::zero(&session, zero_type.clone(), None, mesh.clone()).unwrap(),
            Array::zero(&session, zero_type, None, mesh).unwrap(),
        );

        // Arrays whose shards all live on another process have no local storage to compare, so two of them are equal
        // only when they share their storage.
        let (left, right) = (remote_array(&session), remote_array(&session));
        assert_eq!(left, left.clone());
        assert_ne!(left, right);
    }

    #[test]
    fn test_array_type() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();

        // The type of an array records none of its placement over auto mesh axes.
        let (array, mesh, _) = sharded_vector(&session);
        assert_eq!(
            array.r#type().into_owned(),
            ArrayType::new_static(DataType::F32, [4])
                .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
                .unwrap(),
        );
    }

    #[test]
    fn test_array_domain() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));

        // Arrays recover the session they were constructed in, together with its session and client.
        let recovered = array.domain();
        assert!(std::ptr::eq(recovered.parent().client(), client));
        assert!(recovered.parent().is_same_session(&session));
    }

    #[test]
    fn test_array_concretize_bool() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let truth = replicated_array(&session, &mesh, DataType::Boolean, &[], &[1]);
        let falsity = replicated_array(&session, &mesh, DataType::Boolean, &[], &[0]);
        assert_eq!(Concretizable::<bool>::concretize(&truth), Ok(true));
        assert_eq!(Concretizable::<bool>::concretize(&falsity), Ok(false));

        // Rank-1 arrays cannot collapse to a single Boolean.
        let vector = replicated_array(&session, &mesh, DataType::Boolean, &[2], &[1, 0]);
        assert_eq!(
            Concretizable::<bool>::concretize(&vector),
            Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from a value of type \
                          `bool[2][sharding={mesh<['x'=1:auto]>, [{}]}]`; expected `bool[]`"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_array_concretize_bool_unreduced() {
        // The two devices hold the candidates `false` and `true` of a Boolean scalar whose unreduced kind is `max`, so
        // its value is `true` even though the first addressable shard holds `false`.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let array = unreduced_scalar(&session, DataType::Boolean, UnreducedKind::Max, [&[0], &[1]]);
        assert_eq!(Concretizable::<bool>::concretize(&array), Ok(true));
    }

    #[test]
    fn test_array_concretize_i128() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let signed = replicated_array(&session, &mesh, DataType::I64, &[], &i64::MIN.to_ne_bytes());
        let unsigned = replicated_array(&session, &mesh, DataType::U64, &[], &u64::MAX.to_ne_bytes());

        // Keep the complete integer range before the caller applies bounds or converts to a host index.
        assert_eq!(Concretizable::<i128>::concretize(&signed), Ok(i128::from(i64::MIN)));
        assert_eq!(Concretizable::<i128>::concretize(&unsigned), Ok(i128::from(u64::MAX)));

        // Only rank-0 integer arrays concretize to integers.
        let float = replicated_array(&session, &mesh, DataType::F32, &[], &1f32.to_ne_bytes());
        assert_eq!(
            Concretizable::<i128>::concretize(&float),
            Err(ProgramError::Concretization {
                message: "cannot extract a concrete integer from `f32[][sharding={mesh<['x'=1:auto]>, []}]`; expected \
                          a scalar integer"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_array_concretize_i128_unreduced() {
        // The two devices hold the summands `3` and `4` of a scalar that is unreduced along `x`, so it reads as their
        // sum rather than as the summand of the first addressable shard.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let array =
            unreduced_scalar(&session, DataType::I32, UnreducedKind::Sum, [&3i32.to_ne_bytes(), &4i32.to_ne_bytes()]);
        assert_eq!(Concretizable::<i128>::concretize(&array), Ok(7));
    }

    #[test]
    fn test_array_placement() {
        // The placement records the runtime sharding over the auto axis `x`, which the type of the array omits, and
        // the memory of the array.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let memory = Memory::Host { pinned: true };
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]).with_memory(memory),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        assert_eq!(array.placement(), Some(Placement::new(mesh.clone(), sharding).unwrap().with_memory(memory)));

        // An array with a known layout also records that layout.
        let layout = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 2]).with_layout(layout.clone()),
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        assert_eq!(
            array.placement(),
            Some(
                Placement::new(mesh, array.sharding().clone())
                    .unwrap()
                    .with_memory(Memory::Device)
                    .with_layout(layout)
            ),
        );
    }

    #[test]
    fn test_array_assertion_observation() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let scalar = replicated_array(&session, &mesh, DataType::F32, &[], &2.5f32.to_ne_bytes());
        assert_eq!(scalar.assertion_observation(), Ok("2.5".to_string()));

        // Complex observations are not supported.
        let complex = replicated_array(&session, &mesh, DataType::C64, &[], &values_to_bytes(&[1f32, 2.]));
        assert_eq!(
            complex.assertion_observation(),
            Err(ProgramError::Concretization {
                message: "unsupported assertion observation type `c64[][sharding={mesh<['x'=1:auto]>, []}]`"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_array_assertion_observation_unreduced() {
        // The observation of a scalar that is unreduced along `x` is the sum of the summands `3` and `4`.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let array =
            unreduced_scalar(&session, DataType::I32, UnreducedKind::Sum, [&3i32.to_ne_bytes(), &4i32.to_ne_bytes()]);
        assert_eq!(array.assertion_observation(), Ok("7".to_string()));
    }

    #[test]
    fn test_array_assertion_array() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let bytes = values_to_bytes(&[1f32, 2.]);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &bytes);

        // The observed reference array carries the element type and shape of the array but not its sharding.
        assert_eq!(
            array.assertion_array(),
            Ok(Some(CoreArray::new(ArrayType::new_static(DataType::F32, [2]), bytes).unwrap())),
        );
    }

    #[test]
    fn test_array_dimension_size() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let matrix = replicated_array(&session, &mesh, DataType::F32, &[2, 3], &values_to_bytes(&[0f32; 6]));
        assert_eq!(matrix.dimension_size(0), Ok(2));
        assert_eq!(matrix.dimension_size(-1), Ok(3));
        assert!(matches!(
            matrix.dimension_size(2),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`dimension_size` axis 2 is out of bounds for rank 2",
        ));

        // The extent of an unevenly sharded axis is the greatest end of the slices of its shards.
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let sharded = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [5]),
            mesh,
            Some(sharding),
            values_to_bytes(&[0f32; 5]),
        )
        .unwrap();
        assert_eq!(sharded.dimension_size(0), Ok(5));
    }

    #[test]
    fn test_array_to_dimension() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let scalar = replicated_array(&session, &mesh, DataType::I32, &[], &5i32.to_ne_bytes());
        assert_eq!(scalar.to_dimension(variable.clone()).unwrap().extent(), 5);

        let negative = replicated_array(&session, &mesh, DataType::I32, &[], &(-1i32).to_ne_bytes());
        assert!(matches!(
            negative.to_dimension(variable),
            Err(ProgramError::InvalidArgument { message })
                if message
                    == "`dimension_from_scalar` scalar input must be a nonnegative host-representable extent but is -1",
        ));
    }

    #[test]
    fn test_array_to_dimension_unreduced() {
        // The devices of a scalar that is unreduced along `x` hold the summands `3` and `4`, which are not the same on
        // every device, so the scalar must be combined before it can become a dimension.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let array =
            unreduced_scalar(&session, DataType::I32, UnreducedKind::Sum, [&3i32.to_ne_bytes(), &4i32.to_ne_bytes()]);
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        assert_eq!(
            array.to_dimension(variable.clone()),
            Err(ProgramError::Type(
                DimensionError::UnreducedScalar { r#type: Box::new(variable.into()), axis_name: "x".to_string() }
                    .into()
            )),
        );
    }

    #[test]
    fn test_array_any_true() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let mixed = replicated_array(&session, &mesh, DataType::Boolean, &[2], &[0, 1]);
        let none = replicated_array(&session, &mesh, DataType::Boolean, &[2], &[0, 0]);
        assert_eq!(mixed.any_true(), Ok(true));
        assert_eq!(none.any_true(), Ok(false));

        // The devices of an unreduced predicate hold candidates, which its unreduced kind combines.
        let client = cpu_client(2);
        let unreduced_session = Session::builder().client(client).build().unwrap();
        for (kind, expected) in [(UnreducedKind::Min, false), (UnreducedKind::Max, true), (UnreducedKind::Sum, true)] {
            let candidates = unreduced_scalar(&unreduced_session, DataType::Boolean, kind, [&[1], &[0]]);
            assert_eq!(candidates.any_true(), Ok(expected));
        }

        // Only Boolean arrays are while predicates.
        let float = replicated_array(&session, &mesh, DataType::F32, &[1], &1f32.to_ne_bytes());
        assert_eq!(
            float.any_true(),
            Err(ProgramError::Concretization {
                message: "cannot use a value of type `f32[1][sharding={mesh<['x'=1:auto]>, [{}]}]` as a Boolean \
                          while predicate"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_array_select() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let mixed = replicated_array(&session, &mesh, DataType::Boolean, &[2], &[0, 1]);
        let on_true = replicated_array(&session, &mesh, DataType::F32, &[2, 2], &values_to_bytes(&[1f32, 2., 3., 4.]));
        let on_false = replicated_array(&session, &mesh, DataType::F32, &[2, 2], &values_to_bytes(&[5f32, 6., 7., 8.]));

        // The predicate, whose shape is a prefix of the operand shape, broadcasts along its leading axes. Item 0 keeps
        // the first row of `on_false` and item 1 takes the second row of `on_true`.
        let selected = WhilePredicate::select(&mixed, &on_true, &on_false).unwrap();
        assert_eq!(selected.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![5f32, 6., 3., 4.]);

        // Non-Boolean predicates are rejected.
        assert_eq!(
            WhilePredicate::select(&on_true, &on_true, &on_false).map(|_| ()),
            Err(ProgramError::UnsupportedOperation {
                message: "`select` requires a Boolean prefix-shaped predicate and congruent inputs, but got \
                          predicate `f32[2, 2][sharding={mesh<['x'=1:auto]>, [{}, {}]}]` with inputs \
                          `f32[2, 2][sharding={mesh<['x'=1:auto]>, [{}, {}]}]` and \
                          `f32[2, 2][sharding={mesh<['x'=1:auto]>, [{}, {}]}]`"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_array_operators() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let read = |array: Array| array.copy_to_host().unwrap().elements::<f32>().unwrap();
        let a = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        let b = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[3f32, 5.]));
        assert_eq!(read(a.clone() + b.clone()), vec![4f32, 7.]);
        assert_eq!(read(b.clone() - a.clone()), vec![2f32, 3.]);
        assert_eq!(read(a.clone() * b.clone()), vec![3f32, 10.]);
        assert_eq!(read(b / a.clone()), vec![3f32, 2.5]);
        assert_eq!(read(-a), vec![-1f32, -2.]);

        let truths = replicated_array(&session, &mesh, DataType::Boolean, &[4], &[1, 1, 0, 0]);
        let mixed = replicated_array(&session, &mesh, DataType::Boolean, &[4], &[1, 0, 1, 0]);
        assert_eq!((truths.clone() & mixed.clone()).copy_to_host_bytes().unwrap(), vec![1u8, 0, 0, 0]);
        assert_eq!((truths.clone() | mixed.clone()).copy_to_host_bytes().unwrap(), vec![1u8, 1, 1, 0]);
        assert_eq!((truths ^ mixed.clone()).copy_to_host_bytes().unwrap(), vec![0u8, 1, 1, 0]);
        assert_eq!((!mixed).copy_to_host_bytes().unwrap(), vec![0u8, 1, 0, 1]);
    }

    #[test]
    fn test_ready() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        let (event, promise) = client.event(()).unwrap();
        let failed = array.clone().with_execution_fence(ExecutionFence::new(vec![event]));
        promise.set(Some(PjrtError::invalid_argument("execution failed"))).unwrap();
        array.block_until_ready().unwrap();
        let mut context = Context::from_waker(Waker::noop());

        // Every array of the structure is waited for, and the first failure among them is returned.
        assert!(matches!(pin!(ready(&vec![array.clone(), array.clone()])).poll(&mut context), Poll::Ready(Ok(()))));
        assert!(matches!(
            pin!(ready(&vec![array, failed])).poll(&mut context),
            Poll::Ready(Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))) if message == "execution failed",
        ));
    }

    #[test]
    fn test_block_until_ready() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        let (event, promise) = client.event(()).unwrap();
        let failed = array.clone().with_execution_fence(ExecutionFence::new(vec![event]));
        promise.set(Some(PjrtError::invalid_argument("execution failed"))).unwrap();
        array.block_until_ready().unwrap();

        // Every array of the structure is waited for, and the first failure among them is returned.
        assert!(matches!(block_until_ready(&vec![array.clone(), array.clone()]), Ok(())));
        assert!(matches!(
            block_until_ready(&vec![array, failed]),
            Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. })) if message == "execution failed",
        ));
    }
}
