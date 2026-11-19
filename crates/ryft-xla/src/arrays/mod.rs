//! Distributed device arrays backed by PJRT buffers, which are the concrete eager values of the XLA backend.
//!
//! An [`Array`] is one logical value of an [`ArrayType`] with a static [`Shape`], placed on the devices of a
//! [`DeviceMesh`] according to a [`Sharding`]. [`ShardLayout`] expands that placement into one [`ShardDescriptor`] per
//! device, which records the slice of the global array that the device holds. Every [`Array`] keeps one
//! [`ArrayShard`] per device, including the shards that live in other processes, so that every process can reason
//! about the complete global placement while only transferring, executing with, or reading the buffers that it can
//! address. Clones of an array share its type, shards, and value-local caches, so cloning never copies device data.
//! The [`mod@shards`] module defines the shard types.
//!
//! Every array belongs to a [`Session`], which determines the PJRT client, session, effect scope, and compilation
//! options that operations on the array use. Arrays implement the `ryft-core` value capabilities by binding each
//! operation through that session, which executes it eagerly, op by op, as a cached single-operation program.
//!
//! [`Array::from_host_buffer`] uploads a dense host value, [`Array::from_host_shards`] uploads the host shards
//! owned by this process, and [`Array::from_addressable_buffers`] wraps existing PJRT [`Buffer`]s. These
//! constructors support placement over auto mesh axes when the caller supplies runtime sharding.
//! [`Array::copy_to_host`] and [`ArrayShard::copy_to_host`] read values back to the host. The `host` module implements
//! the uploads and the readback of whole arrays, including the combination of the separate values that devices hold
//! along unreduced mesh axes.
//! [`Array::to_placement`], [`Array::into_placement`], and [`Session::place`] move arrays and host values to other
//! devices or shardings. The [`mod@placement`] module defines what a placement is and how to request one, and the
//! `resharding` module implements the moves. Finally, the
//! `materialization` module pads arrays to the static bounds of executables compiled for bounded dynamic shapes and
//! caches the padded copies per array. Array failures are reported through [`Error::Array`] with an [`ArrayError`].
//!
//! Every session counts the arrays that are created through it, which [`ArrayStatistics`] reports, and can describe
//! the ones that are still alive as [`LiveArray`]s. The unit that it counts is the storage that an array shares with
//! its clones, so cloning and dropping array handles involves no bookkeeping.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Display};
use std::sync::{Arc, Mutex, OnceLock};

use ryft_core::{
    Add, And, ArrayType, AssertionValue, Broadcast, Concretizable, DataType, Device, DeviceId, DeviceMesh,
    DimensionError, DimensionFromScalar, DimensionFromScalarOperation, DimensionSize, DimensionSizeOperation,
    DimensionType, DimensionValue, DimensionVariable, Div, ElementType, Layout, LayoutError, Memory, Mul, Neg, Not,
    Operation as CoreOperation, Or, Parameter, Parameterized, ProgramError, ProjectedContext, Select, Shape,
    ShapeChangingCollectiveValue, Sharding, StaticShape, Sub, Typed, UnreducedKind, Value as CoreValue,
    ValueDomainDispatch, WhilePredicate, Xor,
};
use ryft_macros::Parameter;
use ryft_pjrt::{Buffer, BufferType, Client, Error as PjrtError, ExecutionFence};

use crate::arrays::materialization::{
    BoundedMaterializationCache, BoundedMaterializationKey, BoundedMaterializationProbe,
    LOGICAL_EXTENT_SCALAR_CACHE_CAPACITY,
};
use crate::arrays::statistics::ArrayRegistration;
use crate::{Error, FromPjrt, Session, ToMlir, ToPjrt, Type};

// TODO(eaplatanios): Review this module.

pub(crate) mod host;
pub(crate) mod materialization;
pub mod placement;
pub(crate) mod resharding;
pub mod retained;
pub mod shards;
pub mod statistics;

pub use placement::{ArrayPlacement, Placeable, PlacementOptions};
pub use retained::RetainedArray;
pub use shards::{ArrayShard, ShardDescriptor, ShardIndex, ShardLayout};
pub use statistics::{ArrayStatistics, LiveArray};

/// Represents errors produced while validating distributed [`Array`] metadata and buffers, placing arrays on devices,
/// or materializing arrays on the host.
///
/// Array APIs return these errors through [`Error::Array`]. Core and PJRT failures generally retain their original
/// error types. [`Self::InvalidShardDimension`] also records the shard and dimension that failed transport encoding
/// alongside the original core error.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ArrayError {
    /// An array type is missing required sharding metadata.
    #[error("missing required sharding metadata")]
    MissingSharding,

    /// The sharding of an array type disagrees with the placement that an array was constructed with.
    #[error(
        "array type sharding `{type_sharding}` does not match the placement `{sharding}` over the non-auto mesh axes"
    )]
    PlacementTypeMismatch { type_sharding: Sharding, sharding: Sharding },

    /// An operation requires a session with a device mesh.
    #[error("session has no device mesh")]
    MissingMesh,

    /// An operation requires a static array shape.
    #[error("expected static shape but got `{shape}`")]
    DynamicShape { shape: Shape },

    /// Default placement could not select a local device.
    #[error("`Session::place` needs a default local device, but the PJRT client has no addressable devices")]
    MissingDefaultDevice,

    /// An explicit source placement differs from the array's current placement.
    #[error(
        "`Session::place` source mesh `{expected_mesh:?}` with sharding `{expected_sharding:?}` does not match the \
         array's current mesh `{actual_mesh:?}` with sharding `{actual_sharding:?}`"
    )]
    SourcePlacementMismatch {
        expected_mesh: DeviceMesh,
        expected_sharding: Sharding,
        actual_mesh: DeviceMesh,
        actual_sharding: Sharding,
    },

    /// The current PJRT client cannot address a required device.
    #[error("device {device_id} is not addressable by the PJRT client in process {process_index}")]
    NonAddressableDevice { device_id: DeviceId, process_index: usize },

    /// A buffer's owning device is absent from the array mesh.
    #[error("device {device_id} is not present in the device mesh")]
    DeviceNotInMesh { device_id: DeviceId },

    /// A buffer's owning process differs from the corresponding mesh device.
    #[error("device {device_id} reports process index {actual_process_index}, but expected {expected_process_index}")]
    DeviceProcessIndexMismatch { device_id: DeviceId, expected_process_index: usize, actual_process_index: usize },

    /// A device has no memory matching the requested placement.
    #[error("device {device_id} has no addressable memory for `{memory}`")]
    UnsupportedMemory { device_id: DeviceId, memory: Memory },

    /// Compiled resharding cannot handle an axis already owned by a manual computation.
    #[error("compiled reshard does not support `Manual` mesh axis {axis_name}")]
    ManualReshardingAxis { axis_name: String },

    /// Multiple buffers were supplied for one device.
    #[error("got multiple buffers for device {device_id}")]
    MultipleBuffersOnDevice { device_id: DeviceId },

    /// A physical buffer has a different array type than its declared shard.
    #[error("buffer has type `{actual}`, but expected `{expected}`")]
    BufferTypeMismatch { expected: ArrayType, actual: ArrayType },

    /// A physical buffer occupies a different memory tier than its declared shard.
    #[error("buffer on device {device_id} uses memory kind `{actual}`, but expected `{expected}`")]
    BufferMemoryMismatch { device_id: DeviceId, expected: Memory, actual: String },

    /// A zero-space buffer contains an observable nonzero physical carrier value.
    #[error("zero-space buffer on device {device_id} contains a nonzero physical carrier value")]
    NonCanonicalZeroBuffer { device_id: DeviceId },

    /// Supplied host bytes do not match the declared array type.
    #[error("expected {expected} byte(s) but got {actual}")]
    ByteCountMismatch { expected: usize, actual: usize },

    /// Multiple host shards were supplied for one device.
    #[error("got multiple host shards for device {device_id}")]
    MultipleHostShardsOnDevice { device_id: DeviceId },

    /// No host shard was supplied for a required local device.
    #[error("missing host shard for device {device_id}")]
    MissingHostShard { device_id: DeviceId },

    /// The bytes supplied for a host shard do not match the size implied by its shape and element type.
    #[error("host shard on device {device_id} needs {expected} byte(s) but got {actual}")]
    HostShardByteCountMismatch { device_id: DeviceId, expected: usize, actual: usize },

    /// A shard slice has a different rank than its array shape.
    #[error("shape rank {shape_rank} does not match shard slice rank {slice_rank}")]
    ShardSliceRankMismatch { shape_rank: usize, slice_rank: usize },

    /// Array placement or host materialization requires a shard that has no local buffer.
    #[error("array shard #{shard_index} on device {device_id} has no addressable buffer")]
    MissingShardBuffer { shard_index: ShardIndex, device_id: DeviceId },

    /// A shard dimension cannot be encoded by the transfer ABI, retaining its canonical core cause.
    #[error("cross-host transfer for shard #{shard_index}, dimension #{dimension}: {source}")]
    InvalidShardDimension {
        shard_index: ShardIndex,
        dimension: usize,
        #[source]
        source: DimensionError,
    },

    /// Downloaded shard bytes do not match the expected logical shard size.
    #[error(
        "copied shard #{shard_index} from device {device_id} to host and got {actual_byte_count} byte(s), but expected \
         {expected_byte_count}"
    )]
    ShardByteCountMismatch {
        shard_index: ShardIndex,
        device_id: DeviceId,
        expected_byte_count: usize,
        actual_byte_count: usize,
    },

    /// Overlapping physical shard slices contain different values.
    #[error("array move found inconsistent overlapping data while materializing shard #{shard_index}")]
    InconsistentOverlappingShardData { shard_index: ShardIndex },
}

// TODO(eaplatanios): Is `ArrayType::memory` being set, handled, and propagated correctly throughout this crate?

/// Distributed array with a global [`StaticShape`] and element [`DataType`] as well as [`Sharding`] information. An
/// [`Array`] represents one logical [`ArrayType`] whose elements may be split or replicated across the multiple
/// devices in a [`DeviceMesh`], potentially spanning multiple nodes or processes. The global array is described by its
/// type and sharding metadata, while each physical piece of that global array is represented by an [`ArrayShard`].
/// [`Array::shards`] is a global list, which contains one [`ArrayShard`] for each device that participates in the
/// array placement, and not only for the devices that are visible to the current process. In a single-process setup,
/// every shard is normally _addressable_ because the local [`Client`] can directly access its storage. In a
/// multi-device or multi-node setup, the same logical array can span devices owned by other processes. Shards on the
/// current process's devices are addressable and ordinarily carry local [`Buffer`]s. Bufferless [`DataType::Zero`]
/// shards are also addressable because their unique value requires no storage. Shards on remote devices are
/// non-addressable and carry only metadata such as their global [`ShardIndex`], [`DeviceId`], and [`ArrayType`].
/// Keeping both addressable and non-addressable shards in the same [`Array`] lets local code reason about the
/// complete global placement while only transferring, executing with, or materializing buffers that this process can
/// access directly. This distinction is what allows array movement, execution argument assembly, and cross-host
/// transfers to preserve the full global sharding contract without requiring every process to own every shard buffer.
///
/// The current StableHLO execution path represents logical one-bit integers with predicate buffers. Arrays retain
/// their logical `I1` or `U1` type while using one byte per element in this physical representation. This executable
/// convention is separate from [`ToPjrt`] for [`DataType`], which preserves native PJRT one-bit integer types.
#[derive(Clone, Parameter)]
pub struct Array {
    /// Type, shards, and value-local caches of this [`Array`]. They are immutable once constructed and shared by
    /// every clone, so cloning an array costs a few reference-count increments rather than a deep copy.
    parts: Arc<ArrayParts>,

    /// Whole-execution completion fence for arrays produced by an asynchronous PJRT launch. This is separate from
    /// each shard buffer's ready event because it also retains errors from zero-output effects and other device work
    /// associated with the launch. Cloned output arrays share the same fence.
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
}

impl Array {
    /// Creates an [`Array`] of type `r#type` from the provided _addressable_ [`Buffer`]s using the provided concrete
    /// [`DeviceMesh`] to determine what [`ArrayShard`]s make up the array and which ones correspond to the
    /// addressable buffers. The provided [`ArrayType`] describes the global logical array and must have a static
    /// [`Shape`].
    ///
    /// With [`None`] for `sharding`, the function uses the sharding of `r#type` if present and defaults to
    /// replication over `mesh` otherwise. Each buffer in a replicated placement must have the full logical shape,
    /// and callers are responsible for supplying the same value in each replica.
    ///
    /// A supplied sharding can place the array over auto mesh axes. After removing its auto axes with
    /// [`Sharding::without_auto_axes`], it must match any sharding in the type, or the function returns
    /// [`ArrayError::PlacementTypeMismatch`]. The type of the resulting array records that projection, and
    /// [`Self::sharding`] returns the full runtime placement.
    ///
    /// Each [`Buffer`] is assigned to a global shard based on the [`DeviceId`] of its owning device. Failures use
    /// [`Error::Array`] with [`ArrayError::MultipleBuffersOnDevice`] when multiple buffers belong to the same device,
    /// [`ArrayError::DeviceNotInMesh`] when a buffer's device is absent from `mesh`, and
    /// [`ArrayError::DeviceProcessIndexMismatch`] when a buffer's process differs from the corresponding mesh device.
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
    ///     what eager execution and free transforms recover through [`CoreValue::domain`](ryft_core::Value::domain).
    ///   - `r#type`: Global [`ArrayType`] for the new [`Array`].
    ///   - `mesh`: [`DeviceMesh`] that is used to determine the [`ArrayShard`] placement.
    ///   - `sharding`: Optional runtime [`Sharding`] over `mesh`.
    ///   - `buffers`: [`Buffer`]s for the [`ArrayShard`]s that are addressable from the current process. An empty
    ///     list creates an array with only shard metadata.
    pub fn from_addressable_buffers(
        session: &Session,
        r#type: ArrayType,
        mesh: DeviceMesh,
        sharding: Option<Sharding>,
        buffers: Vec<Buffer<'static>>,
    ) -> Result<Self, Error> {
        let buffers = buffers.into_iter().map(Arc::new).collect();
        Self::from_addressable_buffers_internal(session, r#type, sharding, mesh, buffers, true)
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
        let buffers = buffers.into_iter().map(Arc::new).collect();
        Self::from_addressable_buffers_internal(session, r#type, sharding, mesh, buffers, false)
    }

    /// Creates an [`Array`] like [`Self::from_canonical_addressable_buffers`] from buffers that other arrays may also
    /// hold. Sharing is safe because arrays never mutate their buffers and because donation skips arrays whose
    /// buffers are shared (refer to [`Self::has_unique_shard_buffers`]).
    pub(crate) fn from_shared_addressable_buffers(
        session: &Session,
        r#type: ArrayType,
        sharding: Sharding,
        mesh: DeviceMesh,
        buffers: Vec<Arc<Buffer<'static>>>,
    ) -> Result<Self, Error> {
        Self::from_addressable_buffers_internal(session, r#type, Some(sharding), mesh, buffers, false)
    }

    /// Implements addressable buffer construction with optional validation of private zero-space carrier payloads.
    fn from_addressable_buffers_internal(
        session: &Session,
        r#type: ArrayType,
        sharding: Option<Sharding>,
        mesh: DeviceMesh,
        buffers: Vec<Arc<Buffer<'static>>>,
        validate_zero_carriers: bool,
    ) -> Result<Self, Error> {
        let (r#type, sharding) = Self::resolve_placement(r#type, sharding, &mesh)?;

        // Compute the global `ShardLayout` implied by the runtime placement and concrete device mesh.
        let shape = r#type
            .static_shape()
            .ok_or_else(|| Error::from(ArrayError::DynamicShape { shape: r#type.shape().clone() }))?;
        let layout = ShardLayout::new(&shape, &mesh, &sharding)?;
        let (descriptors, shard_index_by_device) = layout.into_parts();

        // Index the provided `Buffer`s by device while rejecting duplicate local buffers for the same device and
        // buffers that the session's client does not own.
        let mut buffers_by_device = HashMap::with_capacity(buffers.len());
        for buffer in buffers {
            let device_id = Self::validate_addressable_buffer(
                session.client(),
                &r#type,
                &descriptors,
                &shard_index_by_device,
                &buffer,
                validate_zero_carriers,
            )?;
            if buffers_by_device.contains_key(&device_id) {
                return Err(Error::from(ArrayError::MultipleBuffersOnDevice { device_id }));
            }
            buffers_by_device.insert(device_id, buffer);
        }

        // Materialize the global shard list with local storage attached to the shards addressable from this process.
        // A validated zero-space carrier is discarded because its false bits carry no logical information.
        let zero_space = r#type.data_type().is_zero();
        let shards = descriptors
            .into_iter()
            .map(|descriptor| {
                let buffer = buffers_by_device.remove(&descriptor.device().id());
                if zero_space && buffer.is_some() {
                    ArrayShard::new_zero(descriptor)
                } else {
                    ArrayShard::new(descriptor, buffer)
                }
            })
            .collect::<Vec<_>>();

        Ok(Self {
            parts: Arc::new(ArrayParts::new(
                session,
                Arc::new(r#type),
                Arc::new(sharding),
                shards,
                Arc::new(shard_index_by_device),
                Arc::new(mesh),
            )),
            execution_fence: None,
            send_fence: None,
            session: session.clone(),
        })
    }

    /// Constructs a bufferless array of the unique value represented by [`DataType::Zero`]. Local shards are marked
    /// addressable without allocating PJRT buffers, while remote shards retain metadata only.
    pub(crate) fn from_zero_space(
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
        let (r#type, sharding) = Self::resolve_placement(r#type, sharding, &mesh)?;
        let shape = r#type
            .static_shape()
            .ok_or_else(|| Error::from(ArrayError::DynamicShape { shape: r#type.shape().clone() }))?;
        let layout = ShardLayout::new(&shape, &mesh, &sharding)?;
        let (descriptors, shard_index_by_device) = layout.into_parts();
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
                    Ok(ArrayShard::new_zero(descriptor))
                } else {
                    Ok(ArrayShard::new(descriptor, None))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            parts: Arc::new(ArrayParts::new(
                session,
                Arc::new(r#type),
                Arc::new(sharding),
                shards,
                Arc::new(shard_index_by_device),
                Arc::new(mesh),
            )),
            execution_fence: None,
            send_fence: None,
            session: session.clone(),
        })
    }

    /// Creates an [`Array`] that belongs to `session` from the per-device `buffers` that one execution of a compiled
    /// program returned for one of its outputs, placed according to that output's precomputed `layout`. The buffers
    /// must be ordered like the addressable devices that `layout` was created for, and `fence` is the whole-execution
    /// completion fence of the launch.
    ///
    /// Unlike [`Self::from_addressable_buffers`], this constructor does not query the buffers' devices, memories,
    /// element types, or dimensions, because XLA guarantees those properties for the buffers that it returns from a
    /// program whose signature Ryft lowered, and skipping the queries keeps each execution free of per-output PJRT
    /// calls. In builds with debug assertions, every buffer is still validated with the same checks.
    pub(crate) fn from_execution_output(
        session: &Session,
        layout: &ArrayOutputLayout,
        buffers: Vec<Buffer<'static>>,
        fence: ExecutionFence,
    ) -> Self {
        #[cfg(debug_assertions)]
        for (position, buffer) in buffers.iter().enumerate() {
            let device_id = Self::validate_addressable_buffer(
                session.client(),
                &layout.r#type,
                &layout.descriptors,
                &layout.shard_index_by_device,
                buffer,
                false,
            );
            debug_assert!(
                device_id.as_ref().is_ok_and(|device_id| {
                    layout.descriptors[layout.shard_index_by_device[device_id]].device().id() == *device_id
                        && layout.buffer_positions[layout.shard_index_by_device[device_id]] == Some(position)
                }),
                "execution output buffer {position} does not match its planned placement: {device_id:?}",
            );
        }
        let zero_space = layout.r#type.data_type().is_zero();
        let mut buffers = buffers.into_iter().map(Some).collect::<Vec<_>>();
        let shards = layout
            .descriptors
            .iter()
            .zip(layout.buffer_positions.iter())
            .map(|(descriptor, position)| match position {
                // Zero-space outputs need no storage. A returned carrier holds only false bits and is discarded like in
                // the validating constructors, and a program may also return no carrier at all.
                Some(_) if zero_space => ArrayShard::new_zero(descriptor.clone()),
                Some(position) => {
                    let buffer = buffers.get_mut(*position).and_then(Option::take).map(Arc::new);
                    ArrayShard::new(descriptor.clone(), buffer)
                }
                None => ArrayShard::new(descriptor.clone(), None),
            })
            .collect::<Vec<_>>();
        Self {
            parts: Arc::new(ArrayParts::new(
                session,
                Arc::clone(&layout.r#type),
                Arc::clone(&layout.sharding),
                shards,
                Arc::clone(&layout.shard_index_by_device),
                Arc::clone(&layout.mesh),
            )),
            execution_fence: Some(fence),
            send_fence: None,
            session: session.clone(),
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

    /// Associates this [`Array`] with `session` for future receiver-based dispatch, keeping its storage. This selects
    /// a different effect scope or mesh of the same session, or a different session on the same PJRT client. It does
    /// not merge or retroactively order work that was previously submitted through the array's former scope.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::Pjrt`] that wraps an [`InvalidArgument`](ryft_pjrt::Error::InvalidArgument) error if
    /// `session` belongs to a different PJRT client than this array. Arrays cannot move between clients because their
    /// addressable shard buffers are owned by the client they were created on.
    pub fn associate(mut self, session: &Session) -> Result<Self, Error> {
        if !std::ptr::eq(self.client(), session.client()) {
            return Err(PjrtError::invalid_argument(
                "cannot associate an array with a session of a different PJRT client",
            )
            .into());
        }
        self.session = session.clone();
        Ok(self)
    }

    /// Returns the PJRT [`Client`] that this [`Array`] lives on, which is the client of its session.
    ///
    /// Note that the client cannot be recovered from the addressable shard buffers themselves. [`Buffer`]s only store
    /// a raw handle to their owning PJRT client, by design, to avoid carrying the
    /// [`KeyValueStore`](ryft_pjrt::KeyValueStore) lifetime. [`Client`] is an owning wrapper whose [`Drop`]
    /// implementation destroys the underlying PJRT client, so no `&Client` can be materialized from that raw handle.
    /// The session is therefore threaded through explicitly at array construction time instead, and
    /// [`Buffer::is_owned_by`] validates that its client owns every addressable shard buffer.
    #[inline]
    pub fn client(&self) -> &'static Client<'static> {
        self.session.client()
    }

    /// Returns the [`Session`] this [`Array`] belongs to.
    #[inline]
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Returns the whole-execution completion fence shared by the outputs of this array's producer, when it has one.
    #[inline]
    pub(crate) fn execution_fence(&self) -> Option<&ExecutionFence> {
        self.execution_fence.as_ref()
    }

    /// Returns the [`DataType`] of the elements stored in this [`Array`].
    #[inline]
    pub fn data_type(&self) -> DataType {
        self.parts.r#type.data_type()
    }

    /// Returns the global [`StaticShape`] of this [`Array`].
    #[inline]
    pub fn shape(&self) -> StaticShape {
        self.parts.r#type.static_shape().unwrap()
    }

    /// Returns the physical storage [`Layout`] of this [`Array`] if it is known.
    #[inline]
    pub fn layout(&self) -> Option<&Layout> {
        self.parts.r#type.layout()
    }

    /// Returns the runtime placement of this [`Array`] over all mesh axes, which corresponds to `jax.Array.sharding`.
    /// Unlike the sharding of its [`Typed`] type, which records only placement over explicit and manual mesh axes and
    /// corresponds to `jax.typeof(x).sharding`, it also records placement over auto mesh axes.
    #[inline]
    pub fn sharding(&self) -> &Sharding {
        &self.parts.sharding
    }

    /// Returns the concrete [`DeviceMesh`] implied by this [`Array`]'s global shard placement metadata (i.e., the
    /// mesh that the array was constructed over).
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.parts.mesh
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
        self.parts.shard_index_by_device.get(&device_id).and_then(|index| self.parts.shards.get(*index))
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
        self.parts.r#type.size_in_bytes()
    }

    /// Returns whether every global shard has process-local storage and can therefore retain a bounded materialization.
    pub(crate) fn supports_bounded_materialization_cache(&self) -> bool {
        self.parts.shards.iter().all(ArrayShard::is_addressable)
    }

    /// Probes the bounded materialization cache that the clones of this array share for one structural `key`.
    pub(crate) fn probe_bounded_materialization(&self, key: BoundedMaterializationKey) -> BoundedMaterializationProbe {
        self.parts.bounded_materializations.get_or_init(Arc::default).probe(key)
    }

    /// Returns the replicated device-resident `i32` scalar for `extent` and whether this call uploaded it.
    pub(crate) fn logical_extent_scalar(&self, session: &Session, extent: i32) -> Result<(Array, bool), Error> {
        let mut scalars = self.parts.logical_extent_scalars.lock().expect("logical extent scalar cache mutex poisoned");
        if let Some(index) = scalars.iter().position(|(candidate, _)| *candidate == extent) {
            let entry = scalars.remove(index).unwrap();
            let scalar = entry.1.clone();
            scalars.push_back(entry);
            return Ok((scalar, false));
        }

        // Release the cache lock before the upload, because `Array::from_host_buffer` issues a PJRT host transfer and
        // other users of this array should not wait behind it. Another thread may therefore insert the same extent
        // while the lock is released, so the cache is checked again after relocking. A cached scalar wins over the
        // fresh upload, which is dropped, so that every caller observes one shared scalar per extent. The second
        // element of the result still reports `true` because this call performed an upload.
        drop(scalars);

        let scalar_type =
            ArrayType::scalar(DataType::I32).with_sharding(Sharding::replicated(self.sharding().mesh().clone(), 0))?;
        let scalar = Array::from_host_buffer(session, scalar_type, self.mesh().clone(), None, extent.to_ne_bytes())?;
        let mut scalars = self.parts.logical_extent_scalars.lock().expect("logical extent scalar cache mutex poisoned");
        if let Some(index) = scalars.iter().position(|(candidate, _)| *candidate == extent) {
            let entry = scalars.remove(index).unwrap();
            let cached = entry.1.clone();
            scalars.push_back(entry);
            return Ok((cached, true));
        }
        scalars.push_back((extent, scalar.clone()));
        while scalars.len() > LOGICAL_EXTENT_SCALAR_CACHE_CAPACITY {
            scalars.pop_front();
        }
        Ok((scalar, true))
    }

    /// Returns whether this array has no clones and every addressable shard buffer is uniquely owned by it. An array
    /// without any shard buffers (e.g., a bufferless zero-space array) is never reported as uniquely owned, so it is
    /// never donated.
    ///
    /// Donation is all or nothing across shards, so any shared buffer disables donation for the complete logical
    /// input. The check is sound, rather than a heuristic, when the caller owns `self` by value, because a strong count
    /// of one means that this array holds the only reference to the buffer, so no concurrent clone can appear between
    /// this check and the buffer's consumption. Note that caches shared by clones (e.g., retained bounded
    /// materializations) keep their own references, so cached arrays are never donated.
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

    /// Renders the Shardy tensor sharding attribute (i.e., `#sdy.sharding<...>`) implied by this array, using the
    /// canonical `@mesh` symbol name.
    pub fn to_shardy_tensor_sharding_attribute(&self) -> Result<String, ryft_mlir::Error> {
        let context = ryft_mlir::Context::new();
        self.sharding().to_mlir(context.unknown_location()).map(|attribute| attribute.to_string())
    }

    /// Selects the buffer representation required by the current StableHLO executable boundary. General PJRT type
    /// conversion remains faithful to native one-bit types, and only this array execution path uses predicate carriers.
    pub(crate) fn physical_buffer_type(data_type: DataType) -> BufferType {
        match data_type {
            DataType::I1 | DataType::U1 => BufferType::Predicate,
            _ => data_type.to_pjrt(),
        }
    }

    /// Returns the [`ArrayType`] and runtime placement of an array of type `r#type` that is constructed on `mesh` with
    /// the optional placement `sharding`. Types never record placement over auto mesh axes, so the returned type
    /// carries the placement without its auto mesh axes. Without `sharding`, the placement is the sharding of
    /// `r#type` if present and replication over `mesh` otherwise.
    fn resolve_placement(
        r#type: ArrayType,
        sharding: Option<Sharding>,
        mesh: &DeviceMesh,
    ) -> Result<(ArrayType, Sharding), Error> {
        let sharding = match (sharding, r#type.sharding()) {
            (Some(sharding), Some(type_sharding)) if *type_sharding != sharding.without_auto_axes() => {
                return Err(Error::from(ArrayError::PlacementTypeMismatch {
                    type_sharding: type_sharding.clone(),
                    sharding,
                }));
            }
            (Some(sharding), _) => sharding,
            (None, Some(type_sharding)) => type_sharding.clone(),
            (None, None) => Sharding::replicated(mesh.logical_mesh().clone(), r#type.rank()),
        };
        let r#type = r#type.with_sharding(sharding.without_auto_axes())?;
        Ok((r#type, sharding))
    }

    /// Validates one addressable `buffer` of an array of type `r#type` whose shards are described by `descriptors` and
    /// `shard_index_by_device`, returning the identifier of the device that owns it. The buffer must be owned by
    /// `client`, live on a device of the mesh in the process that the mesh expects, live in the memory that `r#type`
    /// requests, and have the physical element type and the shard shape implied by `r#type`. When
    /// `validate_zero_carriers` is set, the buffer of a zero-space array must also contain only false bits.
    fn validate_addressable_buffer(
        client: &Client<'_>,
        r#type: &ArrayType,
        descriptors: &[ShardDescriptor],
        shard_index_by_device: &HashMap<DeviceId, ShardIndex>,
        buffer: &Buffer<'_>,
        validate_zero_carriers: bool,
    ) -> Result<DeviceId, Error> {
        let device = Device::from_pjrt(buffer.device()?)?;
        if !buffer.is_owned_by(client) {
            return Err(PjrtError::invalid_argument(format!(
                "the session's client does not own the addressable shard buffer for device {}",
                device.id(),
            ))
            .into());
        }
        let device_id = device.id();

        let shard_index = shard_index_by_device
            .get(&device_id)
            .ok_or_else(|| Error::from(ArrayError::DeviceNotInMesh { device_id }))?;
        let descriptor = descriptors.get(*shard_index).unwrap();

        // Validate that each buffer is owned by the process expected for the corresponding mesh device.
        let process_index = device.process_index();
        if process_index != descriptor.device().process_index() {
            return Err(Error::from(ArrayError::DeviceProcessIndexMismatch {
                device_id,
                expected_process_index: descriptor.device().process_index(),
                actual_process_index: process_index,
            }));
        }

        // A type's placement must describe the actual buffer, not merely relabel its storage. Device placement
        // means the plugin's default memory, whose kind is platform-dependent (e.g., CPU uses host memory).
        let buffer_memory = buffer.memory()?;
        let buffer_device = buffer.device()?;
        let memory_matches = match r#type.memory() {
            Memory::Device => buffer_memory == buffer_device.default_memory()?,
            Memory::Host { pinned: true } => buffer_memory.kind()? == "pinned_host",
            Memory::Host { pinned: false } => buffer_memory.kind()? == "unpinned_host",
        };
        if !memory_matches {
            return Err(Error::from(ArrayError::BufferMemoryMismatch {
                device_id,
                expected: r#type.memory(),
                actual: buffer_memory.kind()?.to_string(),
            }));
        }

        if let Some(expected) = r#type.layout() {
            let actual = Layout::from_pjrt(buffer.specification()?.layout)?;
            if &actual != expected {
                return Err(LayoutError::InvalidLayout {
                    message: format!(
                        "buffer on device {device_id} uses layout `{actual}` but its type declares `{expected}`",
                    ),
                }
                .into());
            }
        }

        // Validate the concrete buffer type against the physical representation of the logical shard type.
        // Zero-space values and one-bit integers use predicate carriers in this execution path. Converting a
        // carrier back to a `DataType` would incorrectly relabel those logical arrays as Boolean.
        let buffer_type = buffer.element_type()?;
        let expected_buffer_type = Self::physical_buffer_type(r#type.data_type());
        let data_type =
            if buffer_type == expected_buffer_type { r#type.data_type() } else { DataType::from_pjrt(buffer_type)? };
        let shape = StaticShape::new(
            buffer
                .unpadded_dimensions()?
                .iter()
                .map(|size| {
                    usize::try_from(*size).map_err(|_| Error::SizeLimitExceeded {
                        message: format!(
                            "buffer dimension size {size} exceeds the maximum allowed size of {}",
                            usize::MAX,
                        ),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        if buffer_type != expected_buffer_type && matches!(r#type.data_type(), DataType::I1 | DataType::U1) {
            return Err(Error::from(ArrayError::BufferTypeMismatch {
                expected: ArrayType::new(DataType::Boolean, shape.clone().into()),
                actual: ArrayType::new(data_type, shape.into()),
            }));
        }
        let array_type = ArrayType::new(data_type, shape.into());
        let expected_array_type = ArrayType::new(r#type.data_type(), descriptor.shape().into());
        if array_type != expected_array_type {
            return Err(Error::from(ArrayError::BufferTypeMismatch {
                expected: expected_array_type,
                actual: array_type,
            }));
        }
        if validate_zero_carriers
            && r#type.data_type().is_zero()
            && buffer.copy_to_host(None)?.r#await()?.iter().any(|value| *value != 0)
        {
            return Err(Error::from(ArrayError::NonCanonicalZeroBuffer { device_id }));
        }
        Ok(device_id)
    }
}

/// Waits asynchronously for every [`Array`] leaf in `values` to become ready without copying data to the host.
pub async fn ready<Values: Parameterized<Array>>(values: &Values) -> Result<(), Error> {
    for array in values.parameters() {
        array.ready().await?;
    }
    Ok(())
}

/// Blocks until every [`Array`] leaf in `values` is ready without copying data to the host.
pub fn block_until_ready<Values: Parameterized<Array>>(values: &Values) -> Result<(), Error> {
    for array in values.parameters() {
        array.block_until_ready()?;
    }
    Ok(())
}

impl Debug for Array {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Array")
            .field("type", &self.parts.r#type)
            .field("sharding", &self.parts.sharding)
            .field("shards", &self.shards())
            .finish()
    }
}

impl Display for Array {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "array(type={})", self.parts.r#type)
    }
}

// `Array` equality is *storage identity*, not element-wise value equality. Arrays that share their storage are equal.
// Otherwise, ordinary arrays are equal only when they have the same type and sharding and every shard references the
// same underlying `Buffer`, as determined by `Arc` pointer identity, while bufferless zero-space shards compare equal
// because their type admits exactly one value, and shards on other processes never compare equal because they have no
// local storage to compare. Arrays holding equal ordinary data in distinct buffers deliberately compare unequal, which
// is the conservative answer that analyses comparing flowing values need and mirrors the `PartialEq` implementation of
// `Tracer`, which compares staging identity. Element-wise equality would require device-to-host transfers and is
// deliberately not what this implements.
impl PartialEq for Array {
    fn eq(&self, other: &Self) -> bool {
        let (left, right) = (&self.parts, &other.parts);
        Arc::ptr_eq(left, right)
            || (left.r#type == right.r#type
                && left.sharding == right.sharding
                && left.shards.len() == right.shards.len()
                && left.shards.iter().zip(right.shards.iter()).all(|(left, right)| left.storage_eq(right)))
    }
}

impl Typed for Array {
    type Type = ArrayType;

    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.parts.r#type)
    }
}

impl CoreValue for Array {
    // A concrete `Array` binds and executes every operation through its PJRT-backed `Session`. It uses
    // `ValueDomainDispatch`, so the blanket value capabilities in `ryft-core` (e.g., arithmetic, comparison,
    // selection, manipulation, and reductions) bind their operations through `domain()` and execute them eagerly, op by
    // op, through the session recovered below, and free transform entry points (e.g., `ryft_core::batching::batch`)
    // recover the same session. Only capabilities without blanket implementations that bind operations (e.g.,
    // `Concretizable<bool>`, `WhilePredicate`, and the foreign `std::ops` sugar) get the direct implementations that
    // follow.
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

// TODO(eaplatanios): Review from here onwards.

// Every eager value capability of `Array` follows the JAX-style op-by-op dispatch shape. The blanket capability
// implementations in `ryft-core` (e.g., arithmetic, logical, comparison, selection, manipulation, and reductions)
// bind each operation through `ryft_core::Value::domain`, which for a concrete `Array` recovers the PJRT-backed
// `Session` that the array belongs to. That session executes the operation through a single-operation program that its
// session caches per operation application, so a repeated eager operation skips program construction and lowering.
// The implementations that follow therefore only cover the capabilities that those blankets cannot cover, which are
// the capabilities that read values back to the host, such as `Concretizable` and `WhilePredicate`, and the foreign
// `std::ops` operator sugar, which the orphan rule requires to be implemented per type. Backend arrays opt into the
// shared shape-changing collective capabilities through `ShapeChangingCollectiveValue`.

impl Concretizable<bool> for Array {
    fn concretize(&self) -> Result<bool, ProgramError> {
        // Extracts a concrete scalar Rust Boolean by copying the value of a rank-0 Boolean `Array` to the host with
        // `Array::copy_to_host`. Arrays of higher rank or of non-Boolean types error because they cannot collapse to
        // a single Boolean, and arrays whose value the current process does not hold error because it cannot read
        // their payload.
        if self.r#type().rank() == 0 && self.data_type().is_boolean() {
            let bytes =
                self.copy_to_host().map_err(|error| ProgramError::Concretization { message: error.to_string() })?;
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
        // Decode through the CPU capability to preserve every signed and unsigned integer value.
        let bytes = self.copy_to_host().map_err(|error| ProgramError::Concretization { message: error.to_string() })?;
        ryft_core::Array::new(self.r#type().into_owned(), bytes)?.concretize()
    }
}

// Batched while predicate semantics for `Array`, mirroring the reference semantics of `ryft_core::arrays::Array`.
// `WhilePredicate::any_true` reduces the whole Boolean payload with `or` by reading every shard back to the host, and
// `WhilePredicate::select` broadcasts the predicate against the operands along its leading prefix axes on device
// before selecting.
impl WhilePredicate for Array {
    fn any_true(&self) -> Result<bool, ProgramError> {
        if !self.data_type().is_boolean() {
            return Err(ProgramError::Concretization {
                message: format!("cannot use a value of type `{}` as a Boolean while predicate", self.r#type()),
            });
        }

        // The shards are read one at a time, so the whole execution that produced this array is observed first to
        // surface its errors, including those that are not tied to an individual shard buffer.
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
            let bytes =
                self.copy_to_host().map_err(|error| ProgramError::Concretization { message: error.to_string() })?;
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
                    .copy_to_host_internal(Some(self.session()))
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

        // Broadcast a predicate whose shape is a prefix of the operand shape up to that full shape, mapping its axes to
        // the leading output axes, so that the staged select receives operands with congruent shapes.
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

impl AssertionValue for Array {
    fn assertion_observation(&self) -> Result<String, ProgramError> {
        let bytes = self.copy_to_host().map_err(|error| ProgramError::Concretization { message: error.to_string() })?;
        ryft_core::Array::new(self.r#type().into_owned(), bytes)?.assertion_observation()
    }

    fn assertion_array(&self) -> Result<Option<ryft_core::Array>, ProgramError> {
        let bytes = self.copy_to_host().map_err(|error| ProgramError::Concretization { message: error.to_string() })?;
        Ok(Some(ryft_core::Array::new(ArrayType::new(self.data_type(), self.shape().into()), bytes)?))
    }
}

impl DimensionSize<ArrayType> for Array {
    type Extent = usize;

    fn dimension_size<AxisValue: Into<ryft_core::Axis>>(&self, axis: AxisValue) -> Result<usize, ProgramError> {
        // Returns the concrete global extent of `axis` from this array's complete shard metadata. The descriptor list
        // includes remote shards, so taking the greatest exclusive slice end recovers the global extent without
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
        let bytes = self.copy_to_host().map_err(|error| ProgramError::Concretization { message: error.to_string() })?;
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

// Backend arrays opt into the shared homogeneous collective staging rules of `ryft-core`.
impl ShapeChangingCollectiveValue for Array {}

// The `std::ops` operator traits are foreign, so the blanket tracer implementations in `ryft-core` cannot cover
// concrete backend values. These per-type implementations provide the ergonomic panicking sugar by delegating to
// the fallible `ryft` capabilities.

impl std::ops::Neg for Array {
    type Output = Self;

    fn neg(self) -> Self {
        Neg::neg(&self).expect("`neg` operation failed")
    }
}

impl std::ops::Add for Array {
    type Output = Self;

    fn add(self, right: Self) -> Self {
        Add::add(&self, &right).expect("`add` operation failed")
    }
}

impl std::ops::Sub for Array {
    type Output = Self;

    fn sub(self, right: Self) -> Self {
        Sub::sub(&self, &right).expect("`sub` operation failed")
    }
}

impl std::ops::Mul for Array {
    type Output = Self;

    fn mul(self, right: Self) -> Self {
        Mul::mul(&self, &right).expect("`mul` operation failed")
    }
}

impl std::ops::Div for Array {
    type Output = Self;

    fn div(self, right: Self) -> Self {
        Div::div(&self, &right).expect("`div` operation failed")
    }
}

impl std::ops::Not for Array {
    type Output = Self;

    fn not(self) -> Self {
        Not::not(&self).expect("`not` operation failed")
    }
}

impl std::ops::BitAnd for Array {
    type Output = Self;

    fn bitand(self, right: Self) -> Self {
        And::and(&self, &right).expect("`and` operation failed")
    }
}

impl std::ops::BitOr for Array {
    type Output = Self;

    fn bitor(self, right: Self) -> Self {
        Or::or(&self, &right).expect("`or` operation failed")
    }
}

impl std::ops::BitXor for Array {
    type Output = Self;

    fn bitxor(self, right: Self) -> Self {
        Xor::xor(&self, &right).expect("`xor` operation failed")
    }
}

/// Placement of one output of a compiled program, computed once per program so that every execution can build that
/// output's [`Array`] from the returned buffers (refer to [`Array::from_execution_output`]) without deriving or
/// validating its placement again. Every array built from one layout shares its type, device index, and mesh.
pub(crate) struct ArrayOutputLayout {
    /// [`ArrayType`] of the output, which has a static shape and sharding metadata.
    r#type: Arc<ArrayType>,

    /// Runtime placement of the output over all mesh axes.
    sharding: Arc<Sharding>,

    /// Shard descriptors of the output, in mesh order.
    descriptors: Vec<ShardDescriptor>,

    /// For each shard, the position of its buffer among the executable's addressable devices, or [`None`] for a shard
    /// that is not addressable from the current process.
    buffer_positions: Vec<Option<usize>>,

    /// Lookup table mapping [`DeviceId`]s to their corresponding [`ShardIndex`]es.
    shard_index_by_device: Arc<HashMap<DeviceId, ShardIndex>>,

    /// Concrete [`DeviceMesh`] that the output is placed on.
    mesh: Arc<DeviceMesh>,
}

impl ArrayOutputLayout {
    /// Creates the layout of an output of type `r#type` placed on `mesh` according to `sharding`, whose buffers an
    /// execution returns in the order of `addressable_device_ids`. When `sharding` is [`None`], the output is placed
    /// according to the sharding of `r#type`, or replicated when `r#type` has no sharding.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if `r#type` does not have a static shape, if `sharding` disagrees with the sharding of
    /// `r#type`, or if the placement is not valid on `mesh`.
    pub(crate) fn new(
        r#type: ArrayType,
        sharding: Option<Sharding>,
        mesh: Arc<DeviceMesh>,
        addressable_device_ids: &[DeviceId],
    ) -> Result<Self, Error> {
        let (r#type, sharding) = Array::resolve_placement(r#type, sharding, &mesh)?;
        let shape = r#type
            .static_shape()
            .ok_or_else(|| Error::from(ArrayError::DynamicShape { shape: r#type.shape().clone() }))?;
        let (descriptors, shard_index_by_device) = ShardLayout::new(&shape, &mesh, &sharding)?.into_parts();
        let buffer_positions = descriptors
            .iter()
            .map(|descriptor| {
                addressable_device_ids.iter().position(|device_id| *device_id == descriptor.device().id())
            })
            .collect();
        Ok(Self {
            r#type: Arc::new(r#type),
            sharding: Arc::new(sharding),
            descriptors,
            buffer_positions,
            shard_index_by_device: Arc::new(shard_index_by_device),
            mesh,
        })
    }
}

/// Provides XLA-specific helpers for [`ArrayType`]s.
pub(crate) trait ArrayTypeExtension {
    /// Returns the number of logical payload bytes required by a dense [`Array`] of this [`ArrayType`].
    /// [`DataType::Zero`] requires no payload bytes even though a backend may use a temporary private carrier.
    fn size_in_bytes(&self) -> Result<usize, Error>;
}

impl ArrayTypeExtension for ArrayType {
    #[inline]
    fn size_in_bytes(&self) -> Result<usize, Error> {
        if self.data_type().is_zero() {
            return Ok(0);
        }
        let element_count = self
            .element_count()?
            .ok_or_else(|| Error::from(ArrayError::DynamicShape { shape: self.shape().clone() }))?;
        let element_size_in_bytes = self.data_type().to_pjrt().element_size_in_bytes()?;
        element_count.checked_mul(element_size_in_bytes).ok_or_else(|| Error::SizeLimitExceeded {
            message: format!(
                "dense byte size for array type `{self}` exceeds the maximum allowed size of {}",
                usize::MAX,
            ),
        })
    }
}

/// Immutable storage of an [`Array`] that its clones share.
struct ArrayParts {
    /// [`ArrayType`] of the [`Array`], shared with the other arrays produced for the same compiled program output. Its
    /// sharding is [`Self::sharding`] without auto mesh axes, because types never record placement over those axes.
    r#type: Arc<ArrayType>,

    /// Runtime placement of the [`Array`] over all mesh axes, including the auto mesh axes that its type omits.
    sharding: Arc<Sharding>,

    /// [`ArrayShard`]s that make up the [`Array`].
    shards: Vec<ArrayShard>,

    /// Lookup table that maps each [`DeviceId`] to the [`ShardIndex`] of its shard in [`Self::shards`]. It is shared
    /// because it depends only on the placement, so arrays produced by one compiled program share it.
    shard_index_by_device: Arc<HashMap<DeviceId, ShardIndex>>,

    /// Concrete [`DeviceMesh`] implied by the global shard placement, shared for the same reason.
    mesh: Arc<DeviceMesh>,

    /// Value-local cache of lazily padded device materializations with bound shapes, allocated on first use.
    bounded_materializations: OnceLock<Arc<BoundedMaterializationCache>>,

    /// Value-local LRU of device-resident scalar arguments for bounded logical extents.
    logical_extent_scalars: Mutex<VecDeque<(i32, Array)>>,

    /// Registration of this storage as one logical array of the session that created it, which keeps the array counted
    /// as live in [`Session::array_statistics`] until the storage is dropped. It is the last field so that the shards
    /// and caches are released before the array stops counting as live.
    _registration: ArrayRegistration,
}

impl ArrayParts {
    /// Creates new [`ArrayParts`] with empty value-local caches and registers them as one logical array of `session`.
    /// Storage can only be created through this function, so every logical array is counted exactly once.
    fn new(
        session: &Session,
        r#type: Arc<ArrayType>,
        sharding: Arc<Sharding>,
        shards: Vec<ArrayShard>,
        shard_index_by_device: Arc<HashMap<DeviceId, ShardIndex>>,
        mesh: Arc<DeviceMesh>,
    ) -> Self {
        let registration = session.array_tracker().register(&r#type, &sharding, &mesh, &shards);
        Self {
            r#type,
            sharding,
            shards,
            shard_index_by_device,
            mesh,
            bounded_materializations: OnceLock::new(),
            logical_extent_scalars: Mutex::new(VecDeque::new()),
            _registration: registration,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array as CoreArray, ArrayType, AssertionValue, Concretizable, DataType, Device, DeviceMesh,
        Dimension as CoreDimension, DimensionBounds, DimensionError, DimensionFromScalar, DimensionSize,
        DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, Neg, ProgramError,
        Reference as CoreReference, Shape, Sharding, ShardingDimension, StaticShape, TiledLayout, TypeError, Typed,
        UnreducedKind, Value as CoreValue, WhilePredicate,
    };
    use ryft_pjrt::{BufferType, Error as PjrtError, ExecutionFence};

    use crate::arrays::shards::ArrayShard;
    use crate::arrays::statistics::ArrayStatistics;
    use crate::sessions::tests::{domain_mesh, shares_effect_scope};
    use crate::tests::{
        assert_send_sync, cpu_client, execution_client, logical_mesh_2x2, other_cpu_client, replicated_array,
        values_from_bytes, values_to_bytes,
    };
    use crate::{Error, FromPjrt, RetainedArray, Session, ToPjrt};

    use super::{Array, ArrayError, ArrayOutputLayout, ArrayTypeExtension, block_until_ready, ready};

    // Builds a rank-0 array whose sharding is unreduced along a two-device mesh axis `x`, with `kind` combining the
    // separate values `device_bytes` that the two devices of the client of `session` hold.
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
                client
                    .buffer(*bytes, Array::physical_buffer_type(data_type), [], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        Array::from_addressable_buffers(session, r#type, mesh, None, buffers).unwrap()
    }

    #[test]
    fn test_array_is_send_and_sync() {
        assert_send_sync::<Array>();
    }

    #[test]
    fn test_array_reference_is_send_and_sync() {
        // Exact reference clones share immutable handle-local type and identity metadata, so the XLA reference family
        // relies on the array value and that metadata being safe for concurrent access.
        assert_send_sync::<CoreReference<Array>>();
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
                array.addressable_shards().map(|shard| shard.copy_to_host().unwrap()).collect::<Vec<_>>(),
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
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(dynamic.clone())]))
            .with_sharding(sharding)
            .unwrap();

        assert!(matches!(
            Array::from_addressable_buffers(&session, array_type, mesh, None, Vec::new()),
            Err(Error::Array(error)) if matches!(error.as_ref(),
                ArrayError::DynamicShape { shape } if shape == &Shape::new(vec![CoreDimension::Dynamic(dynamic)]),
            ),
        ));
    }

    #[test]
    fn test_array_from_addressable_buffers_memory() {
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
    fn test_array_from_addressable_buffers_rejects_noncanonical_zero_carrier() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap()[0].clone();
        let device_id = device.id().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let r#type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]));
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
        let r#type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]));
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

    // TODO(eaplatanios): Review this test.
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
        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4), CoreDimension::Static(4)]));
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
        assert_eq!(array.data_type(), DataType::F32);
        assert_eq!(array.shape(), StaticShape::new(vec![4, 4]));
        assert_eq!(array.layout(), None);
        assert_eq!(array.sharding(), &sharding);
        assert_eq!(array.mesh(), &mesh);
        assert_eq!(array.shards().len(), 4);
        assert_eq!(array.addressable_shards().count(), 4);
        assert!(array.shards().iter().all(|shard| shard.shape() == StaticShape::new(vec![2, 2])));
        assert_eq!(array.size_in_bytes().unwrap(), 64);
        array.block_until_ready().unwrap();
        block_until_ready(&vec![array.clone()]).unwrap();

        // Every device in the mesh has a corresponding addressable shard, while device ids absent from the mesh
        // resolve to no shard at all.
        for device in &devices {
            let shard = array.device_shard(device.id()).unwrap();
            assert_eq!(shard.device(), *device);
            assert!(shard.is_addressable());
            assert_eq!(array.addressable_device_shard(device.id()).map(|shard| shard.device()), Some(*device));
        }
        let absent_device_id = devices.iter().map(Device::id).max().unwrap() + 1;
        assert!(array.device_shard(absent_device_id).is_none());
        assert!(array.addressable_device_shard(absent_device_id).is_none());

        // Layout metadata stored on the underlying array type is exposed verbatim by `Array::layout`.
        let layout = Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()));
        let layout_type = array_type.clone().with_layout(layout.clone());
        let layout_array =
            Array::from_addressable_buffers(&session, layout_type, mesh.clone(), Some(sharding.clone()), Vec::new())
                .unwrap();
        assert_eq!(layout_array.layout(), Some(&layout));
        assert_eq!(layout_array.addressable_shards().count(), 0);
        assert!(layout_array.shards().iter().all(|shard| shard.buffer().is_none()));

        // A type sharding that disagrees with the placement over the non-auto mesh axes is rejected.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let explicit_sharding = Sharding::new(
            explicit_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let replicated_type = array_type.clone().with_sharding(Sharding::replicated(explicit_mesh.clone(), 2)).unwrap();
        let explicit_device_mesh = DeviceMesh::new(explicit_mesh, devices.clone()).unwrap();
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

        // A dense row-major upload uses the same shard layout. With a `[4, 4]` `F32` array sharded over both mesh
        // axes, the first mesh device owns the `2x2` block over rows `0..2` and columns `0..2`.
        let values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let host_array = Array::from_host_buffer(
            &session,
            array_type,
            mesh,
            Some(sharding.clone()),
            values_to_bytes::<f32>(values.as_slice()).as_slice(),
        )
        .unwrap();

        assert_eq!(host_array.sharding(), &sharding);
        assert_eq!(host_array.shape(), StaticShape::new(vec![4, 4]));
        assert_eq!(host_array.addressable_shards().count(), 4);
        assert!(host_array.shards().iter().all(|shard| shard.is_addressable()));
        let shard_bytes = host_array
            .device_shard(devices[0].id())
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), vec![0.0, 1.0, 4.0, 5.0]);
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
        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4), CoreDimension::Static(4)]));
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
        let layout = ArrayOutputLayout::new(
            array_type.clone(),
            Some(sharding.clone()),
            Arc::new(mesh.clone()),
            device_ids.as_slice(),
        )
        .unwrap();

        // Planned outputs describe the same placement as the validating constructor, and every output built from one
        // layout shares its metadata. Their types record none of the placement over the auto mesh axes.
        let expected = Array::from_addressable_buffers(
            &session,
            array_type.clone(),
            mesh.clone(),
            Some(sharding.clone()),
            shard_buffers(),
        )
        .unwrap();
        let output = Array::from_execution_output(&session, &layout, shard_buffers(), ExecutionFence::new(Vec::new()));
        let other_output =
            Array::from_execution_output(&session, &layout, shard_buffers(), ExecutionFence::new(Vec::new()));
        assert_eq!(output.r#type(), expected.r#type());
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(
            output.shards().iter().map(ArrayShard::descriptor).collect::<Vec<_>>(),
            expected.shards().iter().map(ArrayShard::descriptor).collect::<Vec<_>>(),
        );
        assert!(output.shards().iter().all(ArrayShard::is_addressable));
        assert!(Arc::ptr_eq(&output.parts.r#type, &other_output.parts.r#type));
        assert!(Arc::ptr_eq(&output.parts.mesh, &other_output.parts.mesh));
        output.block_until_ready().unwrap();
        let shard_bytes = output.device_shard(device_ids[3]).unwrap().copy_to_host().unwrap();
        assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), vec![30.0, 31.0, 32.0, 33.0]);

        // Zero-space outputs need no returned carriers.
        let zero_type = array_type.with_data_type(DataType::Zero);
        let zero_layout =
            ArrayOutputLayout::new(zero_type, Some(sharding), Arc::new(mesh), device_ids.as_slice()).unwrap();
        let zero_output =
            Array::from_execution_output(&session, &zero_layout, Vec::new(), ExecutionFence::new(Vec::new()));
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
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let layout = ArrayOutputLayout::new(array_type, Some(sharding), Arc::new(mesh), device_ids.as_slice()).unwrap();

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
        drop(Array::from_execution_output(&session, &layout, buffers, ExecutionFence::new(Vec::new())));
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
    fn test_array_associate() {
        let client = cpu_client(1);
        let other_client = other_cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        let original = array.clone();

        // Associating with a handle of the same session that has a new effect scope switches the effect scope while
        // keeping the array's storage.
        let fork = session.with_new_effect_scope();
        let reassociated = array.clone().associate(&fork).unwrap();
        assert!(reassociated.session().is_same_session(&session));
        assert!(shares_effect_scope(reassociated.session(), &fork));
        assert!(!shares_effect_scope(reassociated.session(), original.session()));
        assert_eq!(reassociated, original);

        // Associating with a handle of another session on the same client switches the session, and therefore its
        // compilation cache and effect scope, while keeping the array's storage.
        let other_session = Session::builder().client(client).build().unwrap();
        let reassociated = array.associate(&other_session.with_new_effect_scope()).unwrap();
        assert!(reassociated.session().is_same_session(&other_session));
        assert!(std::ptr::eq(reassociated.client(), client));
        assert_eq!(reassociated, original);

        // Receiver-based operations on the reassociated array compile and execute in the other session.
        let negated = Neg::neg(&reassociated).unwrap();
        assert_eq!(values_from_bytes::<f32>(&negated.copy_to_host().unwrap()), vec![-1f32]);
        assert_eq!(other_session.cached_program_count(), 1);
        assert_eq!(session.cached_program_count(), 0);

        // Re-association never affects the clones that existed beforehand, which keep their original session.
        assert!(original.session().is_same_session(&session));
        assert!(shares_effect_scope(original.session(), &session));

        // Association rejects a session of a different client, even when no numerical buffer could establish
        // ownership (e.g., for bufferless zero-space arrays).
        let other_client_session = Session::builder().client(other_client).build().unwrap();
        let zero = Array::from_host_buffer(&session, ArrayType::scalar(DataType::Zero), None, None, []).unwrap();
        for array in [original, zero] {
            assert!(matches!(
                array.associate(&other_client_session),
                Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))
                    if message == "cannot associate an array with a session of a different PJRT client",
            ));
        }
    }

    #[test]
    fn test_array_client() {
        let client = cpu_client(4);
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh_2x2(), devices).unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4), CoreDimension::Static(4)]));

        // Arrays constructed from host buffers belong to the provided session and live on its client, which transferred
        // their shard buffers. Cloning an array preserves that association.
        let session = Session::builder().client(client).build().unwrap();
        let values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let bytes = values_to_bytes::<f32>(values.as_slice());
        let array = Array::from_host_buffer(
            &session,
            array_type.clone(),
            mesh.clone(),
            Some(sharding.clone()),
            bytes.as_slice(),
        )
        .unwrap();
        let recovered_client = array.client();
        assert!(std::ptr::eq(recovered_client, client));
        assert_eq!(recovered_client.process_index().unwrap(), client.process_index().unwrap());
        assert_eq!(recovered_client.addressable_devices().unwrap().len(), client_devices.len());
        assert!(std::ptr::eq(array.clone().client(), client));
        assert!(array.clone().session().is_same_session(&session));

        // Arrays constructed from addressable buffers belong to the provided session, whose client is validated to own
        // every addressable shard buffer.
        let shard_buffers = || {
            client_devices
                .iter()
                .map(|device| {
                    let bytes = values_to_bytes::<f32>(&[0.0, 1.0, 2.0, 3.0]);
                    client.buffer(bytes.as_slice(), BufferType::F32, [2u64, 2u64], None, device.clone(), None).unwrap()
                })
                .collect::<Vec<_>>()
        };
        let array = Array::from_addressable_buffers(
            &session,
            array_type.clone(),
            mesh.clone(),
            Some(sharding.clone()),
            shard_buffers(),
        )
        .unwrap();
        assert!(std::ptr::eq(array.client(), client));
        assert!(array.session().is_same_session(&session));

        // A session whose client does not own the addressable shard buffers is rejected, even when that client was
        // created from the same plugin with identical options.
        let other_client = other_cpu_client(4);
        let other_session = Session::builder().client(other_client).build().unwrap();
        let error = Array::from_addressable_buffers(
            &other_session,
            array_type.clone(),
            mesh.clone(),
            Some(sharding.clone()),
            shard_buffers(),
        )
        .unwrap_err();
        let message = format!(
            "the session's client does not own the addressable shard buffer for device {}",
            client_devices[0].id().unwrap(),
        );
        assert!(matches!(
            &error,
            Error::Pjrt(PjrtError::InvalidArgument { message: actual, .. }) if actual == &message,
        ));
        assert_eq!(error.to_string(), message);

        // Metadata-only arrays with no addressable buffers carry no storage to validate, and still belong to the
        // provided session and live on its client.
        let array =
            Array::from_addressable_buffers(&other_session, array_type, mesh, Some(sharding), Vec::new()).unwrap();
        assert_eq!(array.addressable_shards().count(), 0);
        assert!(std::ptr::eq(array.client(), other_client));
    }

    #[test]
    fn test_array_to_shardy_tensor_sharding_attribute() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            mesh,
            Some(sharding),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        assert_eq!(array.to_shardy_tensor_sharding_attribute(), Ok("#sdy.sharding<@mesh, [{\"x\"}]>".to_string()));
    }

    #[test]
    fn test_array_physical_buffer_type() {
        assert_eq!(Array::physical_buffer_type(DataType::I1), BufferType::Predicate);
        assert_eq!(Array::physical_buffer_type(DataType::U1), BufferType::Predicate);
        assert_eq!(Array::physical_buffer_type(DataType::Boolean), BufferType::Predicate);
        assert_eq!(Array::physical_buffer_type(DataType::I2), BufferType::I2);
        assert_eq!(DataType::I1.to_pjrt(), BufferType::I1);
        assert_eq!(DataType::U1.to_pjrt(), BufferType::U1);
    }

    #[test]
    fn test_array_clone() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]));
        let array =
            Array::from_host_buffer(&session, array_type, mesh, None, values_to_bytes::<f32>(&[1.0, 2.0]).as_slice())
                .unwrap();

        // Clones share their storage, so a shared array is never donatable while a uniquely held one is.
        assert!(array.has_unique_shard_buffers());
        let clone = array.clone();
        assert!(Arc::ptr_eq(&array.parts, &clone.parts));
        assert_eq!(clone, array);
        assert!(!array.has_unique_shard_buffers());
        drop(clone);
        assert!(array.has_unique_shard_buffers());

        // Clones are handles of one logical array, so creating and dropping them leaves the session's counts unchanged.
        assert_eq!(session.array_statistics(), ArrayStatistics { live_arrays: 1, created_arrays: 1 });
    }

    #[test]
    fn test_array_debug() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let shape = Shape::new(vec![CoreDimension::Static(2)]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(logical_mesh, vec![Device::new(0, 1)]).unwrap();
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
                "varying_manual_axes: {} }, shards: [ArrayShard { index: 0, device_id: 0, process_index: 1, ",
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
        // Arrays whose shards all live on another process have no local storage to compare, so two of them are equal
        // only when they share their parts.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let local = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote = Device::new(local.id() + 1, local.process_index() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![remote]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(logical_mesh, 1))
            .unwrap();
        let remote_array =
            || Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), None, Vec::new()).unwrap();
        let (left, right) = (remote_array(), remote_array());
        assert_eq!(left, left.clone());
        assert_ne!(left, right);
    }

    #[test]
    fn test_array_domain() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));

        // Arrays recover the session they were constructed in, together with its session and client.
        let recovered = CoreValue::domain(&array);
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
        assert_eq!(values_from_bytes::<f32>(&selected.copy_to_host().unwrap()), vec![5f32, 6., 3., 4.]);

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
    fn test_array_operators() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let read = |array: Array| values_from_bytes::<f32>(&array.copy_to_host().unwrap());
        let a = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        let b = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[3f32, 5.]));
        assert_eq!(read(a.clone() + b.clone()), vec![4f32, 7.]);
        assert_eq!(read(b.clone() - a.clone()), vec![2f32, 3.]);
        assert_eq!(read(a.clone() * b.clone()), vec![3f32, 10.]);
        assert_eq!(read(b / a.clone()), vec![3f32, 2.5]);
        assert_eq!(read(-a), vec![-1f32, -2.]);

        let truths = replicated_array(&session, &mesh, DataType::Boolean, &[4], &[1, 1, 0, 0]);
        let mixed = replicated_array(&session, &mesh, DataType::Boolean, &[4], &[1, 0, 1, 0]);
        assert_eq!((truths.clone() & mixed.clone()).copy_to_host().unwrap(), vec![1u8, 0, 0, 0]);
        assert_eq!((truths.clone() | mixed.clone()).copy_to_host().unwrap(), vec![1u8, 1, 1, 0]);
        assert_eq!((truths ^ mixed.clone()).copy_to_host().unwrap(), vec![0u8, 1, 1, 0]);
        assert_eq!((!mixed).copy_to_host().unwrap(), vec![0u8, 1, 0, 1]);
    }

    #[test]
    fn test_array_type_size_in_bytes() {
        let matrix_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2), CoreDimension::Static(3)]));
        let scalar_type = ArrayType::scalar(DataType::C128);
        let token_type = ArrayType::new(DataType::Token, Shape::new(vec![CoreDimension::Static(4)]));

        assert_eq!(matrix_type.size_in_bytes().unwrap(), 24);
        assert_eq!(scalar_type.size_in_bytes().unwrap(), 16);
        assert_eq!(token_type.size_in_bytes().unwrap(), 0);

        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("dynamic", DimensionBounds::unbounded()).into()]),
        );
        let error = dynamic_type.size_in_bytes().unwrap_err();
        assert!(matches!(&error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::DynamicShape { shape } if shape == dynamic_type.shape(),
            ),
        ));
        assert_eq!(error.to_string(), "expected static shape but got `[dynamic]`");

        let oversized_type = ArrayType::new(DataType::U16, Shape::new(vec![CoreDimension::Static(usize::MAX)]));
        let error = oversized_type.size_in_bytes().unwrap_err();
        let message = format!(
            "dense byte size for array type `{oversized_type}` exceeds the maximum allowed size of {}",
            usize::MAX,
        );
        assert!(matches!(&error, Error::SizeLimitExceeded { message: actual } if actual == &message));
        assert_eq!(error.to_string(), message);
    }
}
