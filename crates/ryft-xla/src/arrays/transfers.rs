//! Placement of runtime [`Array`]s on other devices, shardings, memories, and layouts. [`Array::to_placement`] copies
//! an array to a target and leaves the source usable, and [`Array::into_placement`] consumes the source so that a
//! compiled reshard may reuse its buffers. Host values go through the uploads of [`crate::arrays::host`] instead. The
//! same device work also moves the inputs of compiled programs and eager operations whose placement differs from the
//! one that the program needs.
//!
//! # Strategies
//!
//! A placement that keeps the devices and sharding of its source only changes the storage of each shard, copying it
//! to the requested memory and converting it to the requested layout on its device, or through host memory when that
//! conversion cannot be compiled. Any other placement chooses one device strategy, which is the first of the following
//! that applies:
//!
//!   1. **Exact Shards:** Every target shard with elements has the slices of one source shard, and both shardings
//!      hold one value per slice, or the placement only moves the contributions of a pending reduction or of manual
//!      variation to other devices (e.g., onto another order of the same devices) and each target shard holds the
//!      contribution of one source shard, which keeps every contribution bit for bit and introduces or completes
//!      nothing. Each process creates its own empty target shards. Each target shard with elements shares the buffer
//!      of its source shard on the same device, copies it from another device of its process, or receives it through
//!      a PJRT cross-host transfer. Moving the shards compiles nothing, but a sender compiles a conversion on its
//!      device for a shard that it cannot send as it is (i.e., in another layout than the one that the receiver
//!      allocates, or with a dynamic allocation).
//!   2. **Shard Regions:** The source devices slice out the regions that each target needs from device memory.
//!      The target devices join them and move them to the requested memory and layout. When both shardings hold
//!      one value per slice, regions apply across meshes for dense row-major sources or when uneven shards or manual
//!      axes rule out a compiled reshard. Regions also move contributions of a pending reduction or of manual variation
//!      within a tile without combining them. The source devices convert other declared layouts to dense row-major
//!      before slicing.
//!   3. **Compiled Reshard:** An identity program with input and output sharding annotations runs on the mesh of the
//!      source. Across meshes, a replicated copy of the source (i.e., the source itself or its all-gather on its own
//!      mesh) is first copied or transferred to every target device as whole shards, and the program runs on the
//!      target mesh. Such a program keeps the reduction state that the target keeps on the same mesh (i.e.,
//!      it moves the contributions of a pending sum, maximum, or minimum as they are), and it completes a pending
//!      reduction that the target drops and introduces a pending sum or a completed reduction that the target
//!      adds, with the arithmetic of XLA. A process that owns no device of the mesh of such a program compiles and runs
//!      nothing for it, and its result has the type and the placement of the results of the processes that run it,
//!      without buffers.
//!   4. **Host Round Trip:** The collectives of XLA compute what the operation of a data type computes in a compiled
//!      program, which is the value that the host readback of a pending reduction defines up to the sign of a zero
//!      result (refer to [`crate::arrays::host`]), only for some data types, kinds of reduction, and changes of
//!      reduction state (e.g., not for a sum of one-bit integers, which XLA adds as predicates, nor for floating point
//!      maxima and minima, from which XLA drops NaN values), and they cannot introduce a pending maximum or minimum.
//!      Such a change, and a placement that no other strategy makes for a reason that every process predicts (i.e.,
//!      a manual target axis or uneven shards that shard regions cannot place), takes the host fallback below, which
//!      reads the value back and uploads it with the host functions that define it, when one process owns every device
//!      of the target and one replica of every source contribution with elements (replicas of those contributions
//!      on devices of other processes take no part). That process does the round trip, and every other process returns
//!      the placement without buffers. Otherwise no process can read the value back alone, so every process returns the
//!      same error before it does anything. A pending sum that the target introduces for a data type without an
//!      additive identity (i.e., [`DataType::F8E8M0FNU`]) is rejected before anything else, with the error that a host
//!      upload of the same value reports, because no strategy can produce it.
//!
//! The choice reads only information that every process shares (i.e., the type and shard metadata of the source,
//! whether the source has locally assigned devices, the target, and the capabilities of the PJRT plugin of the
//! session, which do not depend on whether the session shares compilation artifacts with other processes), so all the
//! processes of a distributed placement choose the same strategy and issue matching transfers and collectives. What
//! only one process knows (e.g., the physical layout of a buffer when the type declares none, or which buffers it
//! holds) never changes the choice. A source with locally assigned devices takes none of these strategies, as the
//! next section describes.
//!
//! # Locally Assigned Devices
//!
//! The shard metadata of a source is information that every process shares only when every process placed the source
//! on the same devices. A source whose devices this process chose on its own (e.g., a host value that each process
//! uploads without a mesh, which lands on the first device of each process) has a mesh that differs on every process,
//! so a strategy planned from it would make each process send its shards to the others, and no process would plan the
//! matching receives. Every array therefore records whether it has such locally assigned devices (refer to
//! [`Array::is_locally_assigned`]), and the following invariant holds:
//!
//!   - **Meaning:** An array is locally assigned when this process chose its devices without any information that the
//!     other processes share, so each process may hold the same value on devices of its own choice. Such an array lives
//!     on devices of this process only.
//!   - **Sources:** Host uploads without a mesh of the caller or of the session set the fact, and so do uploads onto
//!     the default device of the session (refer to [`Session::with_default_device`]), which only this process holds,
//!     and [`Session::place`] uploads of host values without a target or with a target that names only a
//!     memory or a layout.
//!   - **Propagation:** Placements that name only a memory or a layout, copies of an array on its own devices, moves of
//!     an array between devices of this process for an operation, and eager operations whose array inputs all have the
//!     fact keep it.
//!   - **Clearing:** A placement onto a device, a mesh, or a sharding clears it, even when that target holds the
//!     devices of its source. Construction on a caller-supplied mesh and direct execution on a compiled mesh also clear
//!     it. A call without a placement request inherits local assignment from its public arguments.
//!
//! The fact is separate from [commitment](Array::is_committed), which records whether the devices were chosen on
//! purpose and which a target that names only a memory or a layout also sets, as in JAX, so commitment cannot tell
//! whether the other processes know the devices. Every process of a distributed program applies the same functions to
//! the same kinds of inputs, so every process computes the same fact for its own source. Every process therefore
//! chooses the same strategy for a locally assigned source. Each process places its full value on its own devices of
//! the target with local copies, slices, or a host upload, so no process transfers anything or runs a collective
//! to place it. A process whose devices hold the whole target may do that local work with the strategies above,
//! because they then move nothing between processes. The values of the processes must therefore be equal. When the
//! devices of the target belong to exactly the processes of a run of more than one process, the processes compare
//! the SHA-256 digests of their values before they place them, and the result fails on every process when they differ
//! (refer to [`Array::to_placement`]), unless the session disables the check.
//!
//! Clearing the fact for devices that the caller or the session names is sound only because every process of a
//! distributed run names the same devices there, which is a precondition that no process can check, because each
//! one sees only its own request. Every device, mesh, and sharding that a program names (e.g., the default mesh of a
//! [`Session`], the mesh of a construction, or the device of an [`PlacementRequest::Device`] request) must therefore be
//! the same on every process of the run, and a choice that differs between processes is expressible only by naming no
//! device, which leaves the choice to each process and records the fact. For example, when each process places a value
//! on its own first addressable device with [`PlacementRequest::Device`], or builds its session with a default mesh of
//! its own addressable devices and uploads a value without a mesh, and then places the result replicated over the
//! devices of every process, every process plans whole shards from its own device as the source of the value, so each
//! one sends its shard to the others and none of them receives, and the placement never completes. Every device that
//! a placement names must also be a device that the session knows (refer to [`Session::devices`]), which includes the
//! devices of other processes only when the client lists them or the session learned them from its distributed runtime
//! (refer to [`SessionBuilder::distributed_runtime`](crate::SessionBuilder::distributed_runtime)), so that a mistype
//! device is rejected before any work is submitted.
//!
//! A cross-host transfer moves the whole allocation of a buffer in device memory into a static buffer in the default
//! layout and the default device memory of the receiver, so the sender first copies a shard that its type declares in
//! host memory to device memory and converts any buffer whose allocation differs from that of the receive (i.e., one in
//! another layout or one whose type has dynamic dimensions) to a static buffer in that layout, on its device or through
//! host memory, once for all the receivers of that shard. Every process issues its cross-host transfers in one global
//! order of device pairs, and prepares the buffers that it sends and looks up the devices that it receives on before
//! the first one, because a transfer that one process issues without its peer stays paired with the next transfer
//! between the same devices. A receiver copies its buffers to the requested memory only after it issued all of its
//! transfers, so that a memory that only it lacks never leaves a sender waiting.
//!
//! When the device work that this process runs fails (e.g., a failed compilation or execution on its own devices),
//! [`Array::to_placement`] and [`Array::into_placement`] fall back to a host round trip that reads the whole value
//! back and uploads it again. This needs a target that this process fully addresses and a source that this process
//! can read back, which holds a local replica of every contribution to its value with elements (replicas on devices of
//! other processes are fine) and whose devices that hold one contribution do not vary along a manual mesh axis (refer
//! to [`Array::to_placement`]). Otherwise, they report the error of the device work. An error of the planning, which
//! every process meets alike, is never recovered by one process alone (refer to the host round trip above).
//!
//! # Completion and Failures
//!
//! Placements return without waiting for the device work that they start. The execution fence of a result joins the
//! execution fence of its source with the readiness of every buffer that the placement copies, receives, or uploads
//! and with the fences of the programs that produce its buffers (e.g., compiled reshards, slices, and conversions),
//! because a readback reads only one replica of each tile and would otherwise miss a failed replica or a failure of
//! the work that produced the source. A result also retains the completion of the cross-host sends of its process,
//! which no placement waits for, because a send completes only once its receiver issues the matching receive. The
//! observations of a result report the following failures:
//!
//!   - [`Array::ready`] and [`Array::block_until_ready`] report the failures of every addressable buffer,
//!     of the execution fence, and of those sends.
//!   - [`Array::copy_to_host`] and [`Array::copy_to_host_bytes`] report the failures of the execution fence and of
//!     the buffers that they read. They never wait for the sends, because a send may wait for a receive that another
//!     process issues only after its own readback, so a send that fails only on its receiver is not reported there.
//!     A result without elements copies nothing and waits for its execution fence and every addressable buffer.
//!   - [`Session::synchronize`] reports the failures of every copy, transfer, upload, and program that a placement
//!     registered with the session, including its sends.
//!
//! The CUDA plugin performs the first cross-host rendezvous between two devices on the calling thread. Host readbacks
//! can block placement when the PJRT plugin lacks host to device transfer managers or their literal transfers. A
//! completion fence waits synchronously when the PJRT plugin rejects its callback. Value comparison reads back a
//! locally assigned source on the calling thread when its target names devices of every process of a run. Whole-call
//! admission waits for the collective verdict before placing any leaf that compares the inputs. A failure after the
//! transfers (e.g., a target memory that a device of the receiver lacks) is returned only once every transfer is
//! issued, so the peers of the failing process complete their transfers.
//!
//! A process that fails before it issues its transfers returns its error, and its peers that already issued theirs
//! wait for transfers that never come. The same holds for the later parameters of a [`Session::place`] of a tree, whose
//! transfers the failing process never issues while its peers wait for them. This is an accepted limitation, because
//! the PJRT cross-host transfers offer no way to fail a pending transfer of another process. Every process therefore
//! decides from shared information alone everything that could stop all of them, and before its first transfer it does
//! only what its own transfers need, which is preparing the buffers that it sends. Everything else that only it can
//! check (e.g., which replica serves a local copy, or a memory of its devices) it checks once its transfers are issued,
//! so that a failure there stops only this process. Which buffers a process holds is such information, and a missing
//! buffer of a replica is served by another replica of the same slice on this process. What remains before the
//! transfers are the following failures, which only the failing process can see:
//!
//!   - **A Sent Slice Without Any Local Buffer:** A process that imported an array with only metadata for a tile
//!     that the plan has it send (i.e., without a buffer for that tile or for any local replica of it) returns
//!     [`ArrayError::MissingShardBuffer`] before it issues anything, while the receiver of that tile issues its
//!     receive and waits.
//!   - **A Program That Reads The Source:** A compiled program whose input is the source (i.e., a compiled reshard
//!     on the mesh of the source, or the gather of a reshard across meshes) reads every shard of this process, so
//!     a failure to start that program on this process stops it before that program, while its peers wait in its
//!     collectives and in the transfers that follow it. Such failures are a missing buffer of any of these shards,
//!     a failed conversion of a buffer whose physical layout only this process knows, and a failed launch (e.g., a
//!     failed allocation).
//!   - **A Failed Preparation Of A Sent Buffer:** A copy, a staging in device memory, or a conversion to the layout
//!     of the receiver that fails to start stops this process before its transfers.

use std::cell::OnceCell;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::ops::Range;
use std::sync::Arc;

use ryft_core::{
    ArrayType, CastOperation, Concatenate, DataType, Device, DeviceMesh, DimensionError, FlatProgram, Layout,
    LayoutError, LogicalMesh, Memory, MeshAxis, MeshAxisType, PLACE_OPERATION_NAME, Placeholder, Placement,
    PlacementRequest, ProcessIndex, ProgramBuilder, ProgramError, Sharding, ShardingDimension, ShardingError, Slice,
    StaticShape, TiledLayout, TypeError, Typed, UnreducedKind, check_sharding,
};
use ryft_pjrt::extensions::cross_host_transfers::{CrossHostTransferKey, GlobalDeviceId};
use ryft_pjrt::{
    Buffer, BufferSpecification, BufferType, Client, Device as PjrtDevice, DeviceId, Error as PjrtError, Event,
    ExecutionFence, Memory as PjrtMemory,
};

use crate::arrays::host::{
    CollectiveReduction, HostBufferSlice, PendingHostUpload, ReductionChange, collective_reduction,
    holds_unreduced_sum_identity, partial_completion_rejection, requires_unreduced_sum_identity, run_host_uploads,
    unreduced_sum_identity, validate_device_layout, validate_mesh_devices, validate_one_value_per_contribution,
};
use crate::arrays::metadata::ArrayMetadata;
use crate::arrays::shards::{
    ArrayShard, ShardContribution, ShardContributionMember, ShardDescriptor, ShardIndex, ShardLayout,
    read_shard_to_host,
};
use crate::arrays::{Array, ArrayError};
use crate::devices::{PjrtDeviceExtension, devices_span_processes};
use crate::sessions::synchronization::WorkReservation;
use crate::sharding::DeviceMeshExtension;
use crate::types::DataTypeExtension;
use crate::{
    CompiledFunction, Constant, Error, FromPjrt, LoweringError, Operation, Options, Session, ToPjrt, Type, stage,
};

/// Diagnostic message for a placement that can only combine contributions it must keep separate.
const COMBINED_CONTRIBUTIONS_ERROR_MESSAGE: &str = "cannot place an array with a pending reduction or with manual \
     variation within a tile when neither whole shards nor regions of shards can move its contributions as they are \
     (e.g., when the PJRT plugin does not report the default layout that a cross-host receive allocates), because the \
     placement would have to combine the contributions of the pending reduction (complete the reduction first)";

impl Array {
    /// Wraps one buffer with its physical layout so a compiled operation can consume it directly on its device.
    fn from_device_buffer(
        session: &Session,
        r#type: ArrayType,
        device: Device,
        buffer: Arc<Buffer<'static>>,
    ) -> Result<Self, Error> {
        let placement = Placement::single_device(device, r#type.rank());
        let (mesh, sharding) = (placement.mesh().clone(), placement.sharding().clone());
        let r#type = if r#type.layout().is_some() {
            r#type
        } else {
            r#type.with_layout(Layout::from_pjrt(buffer.specification()?.layout)?)
        };
        Self::from_addressable_buffers_internal(session, r#type, Some(sharding), mesh, None, vec![buffer], false)
    }

    /// Returns this array, which a host upload of a placement has just created on the calling thread (e.g., with
    /// [`Self::from_host_buffer`] when the PJRT plugin lacks host to device transfer managers), with an execution fence
    /// that joins its own fence, if any, and `fence` (i.e., the execution fence of the source of the placement) with
    /// the readiness of every addressable buffer of this array, so that its readbacks report a failed upload of any
    /// replica, as for background uploads (refer to [`crate::arrays::transfers`]). Plain host uploads keep no such
    /// fence, because every one of their replicas receives the same host bytes (refer to [`crate::arrays::host`]).
    fn with_upload_completion(mut self, fence: Option<ExecutionFence>) -> Result<Self, Error> {
        let mut completion = PlacementCompletion::default();
        completion.track_fence(fence.as_ref());
        for buffer in self.addressable_shards().filter_map(ArrayShard::buffer) {
            completion.track_buffer(buffer)?;
        }
        self.execution_fence = completion.join(self.execution_fence.take());
        Ok(self)
    }

    /// Returns an array with the type, the placement, the commitment, and the
    /// [locally assigned devices](Self::is_locally_assigned) of this array whose addressable shard buffers are copies
    /// of the buffers of this array for which `copy` returns `true`, on the same devices and in the same memories, and
    /// the buffers of this array otherwise. With a `copy` that selects every buffer, the result shares no buffer with
    /// any other array. `copy` is called once for each buffer, in shard order, before anything is copied. The execution
    /// fence of the result joins the readiness of every copy with the fence of this array. The array's
    /// [session](Self::session) tracks the copy work and counts the result as a new array. A buffer that the PJRT
    /// plugin cannot copy on its device (e.g., a host memory buffer that a compiled program produced in a layout that
    /// the CPU plugin cannot copy) is copied through host memory with [`Self::copy_shard_through_host`] instead, into
    /// an independent buffer in the same memory and in the declared layout of this array.
    pub(crate) fn with_copied_shard_buffers<F: FnMut(&Arc<Buffer<'static>>) -> bool>(
        &self,
        copy: F,
    ) -> Result<Self, Error> {
        let shards = self.addressable_shards().filter(|shard| shard.buffer().is_some()).collect::<Vec<_>>();
        let copies = shards.iter().map(|shard| shard.buffer().unwrap()).map(copy).collect::<Vec<_>>();
        let session = self.session();
        let mut work = session.reserve_work();
        let mut completion = PlacementCompletion::default();
        let layout = self.layout().map(ToPjrt::to_pjrt).transpose()?;
        let buffers = shards
            .into_iter()
            .zip(copies)
            .map(|(shard, copy)| {
                let buffer = shard.buffer().unwrap();
                if !copy {
                    return Ok(Arc::clone(buffer));
                }
                let (memory, layout) = (buffer.memory()?, || Ok(layout.clone()));
                let (work, completion) = (&mut work, &mut completion);
                self.copy_shard_to_memory(Some(buffer), buffer, shard.descriptor(), memory, layout, work, completion)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let r#type = self.r#type().into_owned().with_sharding(None)?;

        let mut output = if self.data_type().is_zero() {
            Self::zero(session, r#type, Some(self.sharding().clone()), self.mesh().clone())?
        } else {
            let (sharding, mesh) = (self.sharding().clone(), self.mesh().clone());
            Self::from_addressable_buffers_internal(session, r#type, Some(sharding), mesh, None, buffers, false)?
        };

        output.execution_fence = completion.join(self.execution_fence.clone());
        output.send_fence = self.send_fence.clone();
        Ok(output.with_committed(self.is_committed()).with_locally_assigned(self.is_locally_assigned()))
    }

    /// Returns this array with a buffer on every device of this process, or `None` when it already has one there (or
    /// holds the [`DataType::Zero`] data type, whose shards have no buffers). A device without its own buffer receives
    /// a copy of the local replica of its contribution that [`Self::local_replica_shard`] selects, or a new empty
    /// buffer for a shard without elements (refer to [`Self::buffer_for_local_contribution`]). This is how a compiled
    /// program, which reads the buffer of every device of this process, takes an array imported with only some of its
    /// local replicas (e.g., a compiled reshard, or an argument of a compiled function). Nothing is started for the
    /// shards of other processes. The result keeps the placement, the fences, the commitment, and the locally assigned
    /// devices of this array, and its execution fence joins the readiness of every copy.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::MissingShardBuffer`] for the first shard of this process with elements whose contribution
    /// has no local replica with a buffer.
    pub(crate) fn with_filled_local_buffers(&self) -> Result<Option<Self>, Error> {
        let process_index = self.client().process_index()?;

        if self.data_type().is_zero()
            || !self
                .shards()
                .iter()
                .any(|shard: &ArrayShard| shard.device().process_index() == process_index && shard.buffer().is_none())
        {
            return Ok(None);
        }

        self.restricted_to(self.mesh()).map(Some)
    }

    /// Returns whether placing this array on `mesh` with `sharding` can move bytes or run collectives between
    /// processes. Refer to [`PlacementPlanningView::communicates`] for more information.
    #[inline]
    pub(crate) fn placement_communicates(&self, mesh: &DeviceMesh, sharding: &Sharding) -> bool {
        self.placement_planning_view().communicates(self.is_locally_assigned(), mesh, sharding)
    }

    /// Returns the layout that each shard of a placement of this array on `mesh` according to `sharding` has, which
    /// is `requested` when the placement names a layout, and the layout of this array otherwise, but only while that
    /// layout still describes the target shards, as the type rule of the placement keeps it (e.g., the strides of a
    /// column shard do not describe the value that a gather produces). It reads only the type of this array and the
    /// target, which every process shares, so every process and every strategy of a placement record the same layout.
    /// A requested layout is validated against the target shards here, because a placement without buffers never
    /// meets a buffer that would reject it.
    fn placement_layout(
        &self,
        requested: Option<&Layout>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<Option<Layout>, Error> {
        if requested.is_none() && self.layout().is_none() {
            return Ok(None);
        }

        check_sharding!(mesh, sharding);

        let r#type = self.r#type();
        if sharding.rank() != r#type.rank() {
            return Err(ShardingError::ShardingRankMismatch {
                sharding_rank: sharding.rank(),
                array_rank: r#type.rank(),
            }
            .into());
        }

        let sharding_is_complete = true;
        let validate = |layout: &Layout| r#type.validate_shard_layout(layout, Some(sharding), sharding_is_complete);
        match requested {
            Some(layout) => {
                validate(layout).map_err(|error| {
                    TypeError::invalid(format!("`{PLACE_OPERATION_NAME}` target layout is invalid: {error}"))
                })?;
                Ok(Some(layout.clone()))
            }
            None => Ok(self.layout().filter(|layout| validate(layout).is_ok()).cloned()),
        }
    }

    /// Returns whether this array lacks the buffer of a shard on a device of this process, whose process index is
    /// `process_index`, and a local replica of the contribution of every such shard with elements has a buffer (refer
    /// to [`Self::local_replica_shard`]), so that [`Self::with_filled_local_buffers`] can supply every missing buffer. The
    /// replicas are the shards with the same contribution (refer to [`ShardContribution`]), so this holds for an array
    /// with a pending reduction or with manual variation as well. An array with every buffer of this process, which is
    /// the common case, costs one scan of its shards.
    pub(crate) fn can_fill_missing_local_buffers(&self, process_index: usize) -> bool {
        let mut missing = self
            .shards()
            .iter()
            .filter(|shard: &&ArrayShard| shard.device().process_index() == process_index)
            .filter(|shard| shard.buffer().is_none())
            .peekable();
        if missing.peek().is_none() || self.data_type().is_zero() {
            return false;
        }
        let replicas = LocalReplicas::new(process_index);
        missing.all(|shard| {
            shard.slice().iter().any(Range::is_empty)
                || self.local_replica_shard(&replicas, shard.index(), shard.device()).is_ok()
        })
    }

    /// Returns the addressable devices of the client of this array, which runs in the process with index
    /// `process_index`, by their identifiers. A failed query of an identifier is returned as the error of PJRT that it
    /// is, rather than leaving that device out, which a later lookup would report as a device that this process does
    /// not address.
    fn addressable_devices_by_id(&self, process_index: usize) -> Result<AddressableDevices, Error> {
        let devices = self.client().addressable_devices()?;
        let devices = devices.into_iter().map(|device| Ok((device.id()?, device))).collect::<Result<_, PjrtError>>()?;
        Ok(AddressableDevices { process_index, devices })
    }

    /// Places this array on the devices, with the sharding, and in the storage that `target` names, with the memory,
    /// layout, and host fallback rules of [`Self::to_placement`]. A request resolves its default memory into `target`
    /// (refer to [`PlacementRequest::target_memory`]), so a memory that `target` omits keeps the memory of this array.
    /// The result always belongs to `session`, so this array is first associated with `session`, which rejects an
    /// array of another PJRT client, even when the placement leaves the array where it is. The result is
    /// [committed](Self::is_committed) to the devices that `target` names, so it never has
    /// [locally assigned devices](Self::is_locally_assigned), even when they are the devices of this array, and a
    /// caller whose request named only a memory or a layout restores that fact (refer to [`Self::place_with_request`]).
    ///
    /// The requested layout (refer to [`Self::validate_placement_layout`]) and the target devices (refer to
    /// [`validate_mesh_devices`]) are checked before any device work, because a placement without buffers (e.g.,
    /// of a [`DataType::Zero`] array) would otherwise never check the layout, and a device that the target
    /// misattributes would receive no buffer in any process. When the device work fails and returns this array, the
    /// host fallback runs [`Self::upload_host_readback`]. Whatever this function can reject before reading back (e.g.,
    /// a target mesh that this process does not fully address or a source that this process cannot read back) is
    /// reported with the error of the device work.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `target`: Resolved [`Placement`] (refer to [`Self::resolve_placement_request`]).
    ///   - `donate`: Whether a compiled reshard on the mesh of this array may reuse the buffers of this array for
    ///     its output. The donation only takes effect when this array is the only handle of its buffers.
    pub(crate) fn place_within_client(
        self,
        session: &Session,
        target: &Placement,
        donate: bool,
    ) -> Result<Self, Error> {
        // A placement may return the input or share its buffers. Associate it with the target session before
        // choosing a strategy. Consuming `self` keeps the buffers uniquely owned for a donation.
        let array = self.with_session(session)?;
        let layout = array.validate_placement_layout(session, target)?;
        let (mesh, sharding) = (target.mesh().clone(), target.sharding().clone());
        let memory = target.memory().unwrap_or_else(|| array.r#type().memory());
        let placed = |array: Self| array.with_committed(true).with_locally_assigned(false);

        if array.mesh() == &mesh && array.sharding() == &sharding {
            // A placement that only changes the storage of this array, or nothing, does not inherit the sends of the
            // placement that produced it (refer to `Self::without_send_fence`).
            return Ok(placed(array.without_send_fence().place_storage(memory, layout)?));
        }

        validate_mesh_devices(session, &mesh)?;

        // Keep the error of the device work, which is the most informative one if the host fallback fails too.
        let (compiled_error, source) =
            match array.try_reshard_on_devices(&mesh, &sharding, memory, layout.as_ref(), donate, true) {
                Ok(array) => return Ok(placed(array.place_storage(memory, layout)?)),
                Err((error, Some(source))) => (error, source),
                Err((error, None)) => return Err(error),
            };

        // The fallback uploads the dense readback directly into the requested storage. If it cannot complete from this
        // process, surface the original device error. That includes a source whose devices hold one slice but vary
        // along a manual mesh axis, so that they can hold different values for it, which the readback rejects rather
        // than silently replacing the values of the other devices with one of them.
        let source_session = source.session();
        let local_process = source_session.client().process_index()?;

        // A placement that keeps some contributions of a pending reduction or of manual variation within a tile never
        // reads back their combined value (refer to `keeps_some_contributions`), so the failure of its device work is
        // final.
        let target_fully_addressable = mesh.devices().iter().all(|device| device.process_index() == local_process);
        if keeps_some_contributions(source.sharding(), &sharding)
            || !target_fully_addressable
            || source.validate_host_readback().is_err()
        {
            return Err(compiled_error);
        }

        // The round trip reads back and uploads in the background. Its failure poisons the result with an error that
        // also describes the failure of the device work, because that failure is what made the host fallback necessary.
        let device_error = compiled_error.to_string();
        match source.upload_host_readback(source_session, mesh, sharding, memory, layout, move |error| {
            PjrtError::internal(format!(
                "the host fallback of an array placement failed with: {error} (the device work that it replaced \
                failed with: {device_error})",
            ))
        }) {
            Ok(array) => Ok(placed(array)),
            Err(_) => Err(compiled_error),
        }
    }

    /// Places this array, whose PJRT client is not that of `session`, onto `target` as an array of `session`.
    /// PJRT copies buffers only within one client, so this process reads the value back and uploads it with the host
    /// fallback of [`Self::place_within_client`] (refer to [`Self::upload_host_readback`]), whose upload waits for the
    /// work that produces this array and reports its failures. The result is [committed](Self::is_committed), not
    /// [locally assigned](Self::is_locally_assigned), and in the memory that `target` names, or in the memory of this
    /// array otherwise, as within one client (refer to [`PlacementRequest::target_memory`], which resolves the default
    /// memory of a request for a device into `target`). A target device without that memory is an
    /// [`ArrayError::UnsupportedMemory`] error of the process that owns it. Donation never applies, because no buffer
    /// of this array can become a buffer of the result. A process that owns no device of `target` reads nothing back
    /// and returns the placement without buffers. The caller checks the placement first with
    /// [`Self::validate_placement_from_other_client`].
    pub(super) fn place_across_clients(self, session: &Session, target: &Placement) -> Result<Self, Error> {
        let (mesh, sharding) = (target.mesh().clone(), target.sharding().clone());
        let memory = target.memory().unwrap_or_else(|| self.r#type().memory());
        let layout = self.placement_layout(target.layout(), &mesh, &sharding)?;
        let uploaded = self.upload_host_readback(session, mesh, sharding, memory, layout, |error: Error| {
            PjrtError::internal(format!("the transfer of an array from another PJRT client failed with: {error}"))
        })?;
        Ok(uploaded.with_committed(true).with_locally_assigned(false))
    }

    /// Applies storage requirements to the shards already on their target devices. A layout change first tries a
    /// compiled conversion on the device of each shard, and uses a host round trip for that shard when the plugin
    /// cannot lower or compile that conversion (refer to [`Self::try_relayout_shard_on_device`]). Once a buffer has the
    /// requested layout, it can be reused or copied directly to the requested memory. The result keeps the devices of
    /// this array, and with them its [locally assigned devices](Self::is_locally_assigned). It declares exactly
    /// `layout`, which [`Self::placement_layout`] resolved from information that every process shares, whatever layout
    /// the strategy that produced this array recorded, so that every process records the same layout. Without `layout`,
    /// it declares no layout and keeps the buffers as they are. The result keeps the send fence of this array, which
    /// callers pass in only when its sends belong to the same public placement (i.e., when this array is an
    /// intermediate step of that placement, refer to [`Self::place_within_client`]).
    ///
    /// The host round trip of a shard runs in the background as [`Self::copy_shard_through_host`] describes. The
    /// execution fence of the result joins the fences of this array with the readiness of every buffer that this
    /// function creates (i.e., the copies to another memory and the buffers of host round trips) and with the
    /// fences of the compiled conversions.
    pub(super) fn place_storage(self, memory: Memory, layout: Option<Layout>) -> Result<Self, Error> {
        if self.r#type().memory() == memory && self.layout() == layout.as_ref() {
            return Ok(self);
        }

        validate_device_layout(self.client(), self.data_type(), layout.as_ref(), false)?;

        let session = self.session();
        let r#type = self.r#type().into_owned().with_memory(memory).with_layout(layout);
        let mesh = self.mesh().clone();
        let sharding = Some(self.sharding().clone());
        let mut completion = PlacementCompletion::default();

        let mut output = if r#type.data_type().is_zero() {
            Self::zero(session, r#type, sharding, mesh)?
        } else {
            // Register the submission before the first copy, so that a concurrent `Session::synchronize` snapshot
            // includes every copy that may already be running. The host round trips register their own work.
            let mut work = session.reserve_work();
            let mut buffers = Vec::new();
            for shard in self.addressable_shards() {
                let (source_buffer, layout) = (shard.buffer().unwrap(), r#type.layout());
                let completion = &mut completion;
                buffers.push(self.place_shard_storage(shard, source_buffer, memory, layout, &mut work, completion)?);
            }
            Self::from_addressable_buffers_internal(session, r#type, sharding, mesh, None, buffers, false)?
        };

        output.execution_fence = completion.join(self.execution_fence);
        output.send_fence = self.send_fence;

        Ok(output.with_locally_assigned(self.locally_assigned))
    }

    /// Returns `source_buffer`, which holds the contribution of `shard`, a shard of this array, on the device of
    /// `shard`, in `memory` and in `layout` (or in its own layout for [`None`]), as [`Self::place_storage`] places
    /// each shard. A layout change first tries a compiled conversion on that device, and a buffer in the requested
    /// layout is reused or copied to `memory`. When the plugin cannot lower or compile the conversion, or reports that
    /// it cannot copy the buffer to `memory` on the device (e.g., a host memory buffer that a compiled program produced
    /// in a layout that the CPU plugin cannot copy), the shard is copied through host memory with
    /// [`Self::copy_shard_through_host`] instead, from its own buffer, so the contributions of different
    /// devices stay separate. `work` and `completion` record the copies and the fences of the conversions.
    fn place_shard_storage(
        &self,
        shard: &ArrayShard,
        source_buffer: &Arc<Buffer<'static>>,
        memory: Memory,
        layout: Option<&Layout>,
        work: &mut WorkReservation,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        let session = self.session();
        let device = source_buffer.device()?;
        let target_memory = device.memory(memory)?;

        let target_layout = layout.map(ToPjrt::to_pjrt).transpose()?;
        let layout_change = match layout.zip(target_layout.as_ref()) {
            Some((layout, target_layout)) => {
                let current_layout = source_buffer.specification()?.layout;
                (current_layout != *target_layout).then_some((current_layout, layout))
            }
            None => None,
        };

        let buffer = match layout_change {
            Some((current_layout, layout)) => {
                let input_type = ArrayType::new(self.data_type(), shard.shape().into())
                    .with_memory(self.r#type().memory())
                    .with_layout(Layout::from_pjrt(current_layout)?);
                let (device, source_buffer) = (shard.device(), Arc::clone(source_buffer));
                Self::try_relayout_shard_on_device(session, device, source_buffer, input_type, layout, completion)?
            }
            None => Some(Arc::clone(source_buffer)),
        };

        if let Some(buffer) = &buffer
            && buffer.memory()? == target_memory
        {
            return Ok(Arc::clone(buffer));
        }

        self.copy_shard_to_memory(
            buffer.as_ref(),
            source_buffer,
            shard.descriptor(),
            target_memory,
            || Ok(target_layout),
            work,
            completion,
        )
    }

    /// Performs the part of this process in `strategy`, which [`Self::device_placement_strategy`] chose
    /// for placing this array on `mesh` according to `sharding`, with the parameters and the errors of
    /// [`Self::try_reshard_on_devices`]. `process_index` is the index of this process. This function checks that this
    /// array has a buffer on every device of this process only where this array itself is the input of a compiled
    /// program, which reads every such buffer (refer to the documentation of this module). Any other strategy reads
    /// only the shards that its moves name and serves a missing one from another replica of this process (refer to
    /// [`Self::local_replica_shard`]).
    // The error carries the unconsumed array back to the caller, which falls back to a host transfer with it.
    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    fn place_with_strategy(
        self,
        strategy: DevicePlacementStrategy,
        process_index: usize,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        memory: Memory,
        layout: Option<&Layout>,
        donate: bool,
    ) -> Result<Self, (Error, Option<Self>)> {
        match strategy {
            DevicePlacementStrategy::Bufferless => {
                // Only the placement metadata changes, and the result keeps the fence of the execution that produced
                // this array, so that its errors still surface through the result.
                let r#type = self
                    .r#type()
                    .into_owned()
                    .with_memory(memory)
                    .with_sharding(None)
                    .map_err(|error| (error.into(), None))?;
                let mut placed = Self::zero(self.session(), r#type, Some(sharding.clone()), mesh.clone())
                    .map_err(|error| (error, None))?;
                placed.execution_fence = self.execution_fence().cloned();
                Ok(placed)
            }
            DevicePlacementStrategy::ProcessLocalValue => {
                match self.place_process_local_value(mesh, sharding, memory, layout) {
                    Ok(output) => Ok(output),
                    Err(error) => Err((error, Some(self))),
                }
            }
            DevicePlacementStrategy::ExactShards { plan, cross_host_layouts } => self
                .place_exact_shards(mesh, sharding, memory, &plan, &cross_host_layouts)
                .map_err(|error| (error, None)),
            DevicePlacementStrategy::ShardRegions { plan, cross_host_layouts } => {
                // The regions land in default device memory without a declared layout, and each process then moves its
                // own shards to the requested storage, which needs no other process. A source shard in another declared
                // layout is converted to the dense row-major layout only when a move reads it, and before the sends of
                // this process only when a send reads it (refer to `Self::buffer_for_shard_slicing`).
                let placed = self
                    .place_shard_regions(mesh, sharding, &plan, &cross_host_layouts)
                    .and_then(|output| output.place_storage(memory, layout.cloned()));
                match placed {
                    Ok(output) => Ok(output),
                    Err(error) => Err((error, Some(self))),
                }
            }
            DevicePlacementStrategy::CompiledReshard { program } => {
                // The program reads the buffer of every device of this process, so a device without its own buffer
                // first receives a copy of a local replica of its contribution, and a contribution without any local
                // replica stops this process here (refer to `Self::with_filled_local_buffers`).
                let array = match self.with_filled_local_buffers() {
                    Ok(Some(served)) => {
                        drop(self);
                        served
                    }
                    Ok(None) => self,
                    Err(error) => return Err((error, Some(self))),
                };
                match program.compile(&array, donate) {
                    Ok(reshard) => {
                        // Keep a source for host fallback only if execution cannot take its buffers. A clone here
                        // would prevent a donation when this handle owns the buffers exclusively.
                        let source = (!donate || !array.has_unique_shard_buffers()).then(|| array.clone());
                        program.execute(array, reshard.as_ref()).map_err(|error| (error, source))
                    }
                    Err(error) => Err((error, Some(array))),
                }
            }
            DevicePlacementStrategy::CompiledReshardAcrossMeshes { gather, replicas, cross_host_layouts, reshard } => {
                // A change of memory can use device storage for the intermediate compiled reshard. The target
                // then moves the completed shards to the requested tier, without requiring SPMD execution there.
                let source_memory = self.r#type().memory();
                let intermediate_memory = if memory == source_memory { memory } else { Memory::Device };
                let replicated = match &gather {
                    // The caller can still use the source for host fallback. The all-gather therefore borrows
                    // its buffers and fills any missing local ones that the executable needs.
                    Some(gather) => self.with_filled_local_buffers().and_then(|source| {
                        let source = source.unwrap_or_else(|| self.clone());
                        let compiled = gather.compile(&source, false)?;
                        gather.execute(source, compiled.as_ref())
                    }),
                    None => Ok(self.clone()),
                };
                let placed = replicated
                    .and_then(|replicated| {
                        // The whole-shard moves can use another local replica when the planned one has no buffer.
                        replicated.place_exact_shards(
                            &reshard.mesh,
                            &reshard.input,
                            intermediate_memory,
                            &replicas,
                            &cross_host_layouts,
                        )
                    })
                    .and_then(|copies| {
                        // The replicated handle has been dropped, so PJRT can reuse the copies' buffers when
                        // they share no storage with the original. Carry the cross-host sends into the result.
                        let send_fence = copies.send_fence.clone();
                        let compiled = reshard.compile(&copies, true)?;
                        let mut output = reshard.execute(copies, compiled.as_ref())?;
                        output.send_fence = send_fence;
                        Ok(output)
                    });
                placed.map_err(|error| (error, Some(self)))
            }
            DevicePlacementStrategy::IntroducedSums {
                mesh: first_mesh,
                sharding: first_sharding,
                restrict_source,
                first,
                axis_indices,
            } => {
                // The first step places the value on the devices at coordinate zero along the introduced axes,
                // from this array or from its restriction to those devices (which holds the same value, and which the
                // host fallback can read back as well), and this process then gives its other devices the additive
                // identity. An error of the first step returns its input, and a later one this array unless the first
                // step may have donated its buffers, which only a compiled reshard does.
                let donates =
                    donate && matches!(first.as_deref(), Some(DevicePlacementStrategy::CompiledReshard { .. }));
                let source = (!donates || !self.has_unique_shard_buffers()).then(|| self.clone());

                // The restricted view can read only the devices at coordinate zero. Fill their missing buffers from
                // local replicas of the original array before passing the view to the first step.
                let input = if restrict_source {
                    match self.restricted_to(&first_mesh) {
                        Ok(view) => {
                            drop(self);
                            view
                        }
                        Err(error) => return Err((error, Some(self))),
                    }
                } else {
                    self
                };

                let first_is_none = first.is_none();
                let placed = match first.map(|first| *first) {
                    Some(DevicePlacementStrategy::HostRoundTrip { process_index: Some(owner), .. })
                        if owner == process_index =>
                    {
                        // The host round trip of the first step goes to the devices at coordinate zero, which one
                        // process owns together with the source, so the placement of this process onto them takes its
                        // host fallback there.
                        let target = Placement::new(first_mesh.clone(), first_sharding.clone())
                            .map_err(|error| (Error::from(error), source.clone()))?
                            .with_memory(memory);
                        let target = match layout {
                            Some(layout) => target.with_layout(layout.clone()),
                            None => target,
                        };
                        let session = input.session().clone();
                        input.place_within_client(&session, &target, donate).map_err(|error| (error, source.clone()))?
                    }
                    Some(first) => input.place_with_strategy(
                        first,
                        process_index,
                        &first_mesh,
                        &first_sharding,
                        memory,
                        layout,
                        donate,
                    )?,
                    None if input.r#type().memory() == memory && input.layout() == layout => input,
                    None => match input.restricted_to(&first_mesh) {
                        Ok(view) => view,
                        Err(error) => return Err((error, source)),
                    },
                };

                // Without a first step, `placed` is the source or its restriction, whose sends belong to the placement
                // that produced the source (refer to `Self::without_send_fence`).
                let placed = if first_is_none { placed.without_send_fence() } else { placed };
                placed
                    .place_storage(memory, layout.cloned())
                    .and_then(|placed| {
                        // Only devices at coordinate zero keep the first step's buffers. Other devices upload
                        // additive identities locally with the target shard shape and layout, as a host upload does.
                        let session = placed.session().clone();
                        let client = session.client();
                        let process_index = client.process_index()?;
                        let r#type = placed.r#type().into_owned().with_sharding(None)?;
                        let data_type = r#type.data_type();
                        let buffer_type = data_type.executable_buffer_type();
                        let device_layout = r#type.layout().map(ToPjrt::to_pjrt).transpose()?;
                        let layout = ShardLayout::new(&placed.shape(), mesh, sharding)?;
                        let local_shards =
                            layout.descriptors().iter().filter(|shard| shard.device().process_index() == process_index);

                        // A shard without elements holds no contribution, so it gets a new empty buffer as an identity
                        // does, rather than a buffer of this array, which a sparse import need not hold.
                        let (kept, identity_shards): (Vec<_>, Vec<_>) = local_shards.partition(|shard| {
                            !holds_unreduced_sum_identity(mesh, shard, &axis_indices)
                                && !shard.slice().iter().any(Range::is_empty)
                        });
                        let mut work = session.reserve_work();
                        let mut completion = PlacementCompletion::default();
                        completion.track_empty_shard_buffers(&placed)?;

                        // Take the buffers at coordinate zero or copy their local replicas first. If a contribution
                        // has no local buffer, this process fails before it submits any identity uploads.
                        let devices = placed.addressable_devices_by_id(process_index)?;
                        let replicas = LocalReplicas::new(process_index);
                        let mut buffers = Vec::new();
                        for shard in kept {
                            let index =
                                placed.device_shard(shard.device().id()).map(ArrayShard::index).ok_or_else(|| {
                                    Error::from(ArrayError::MissingShardBuffer {
                                        shard_index: shard.index(),
                                        device_id: shard.device().id(),
                                    })
                                })?;
                            let completion = &mut completion;
                            buffers
                                .push(placed.buffer_for_local_contribution(
                                    &replicas, index, &devices, &mut work, completion,
                                )?);
                        }

                        let mut identities = Vec::<(usize, Arc<[u8]>)>::new();
                        for shard in identity_shards {
                            let device_id = shard.device().id();

                            // Shards of one size share a host allocation until PJRT reads it. An empty shard reads
                            // no bytes, so it needs no identity.
                            let shape = shard.shape();
                            let count = shape.element_count()?;
                            let identity = match identities.iter().find(|(identity_count, _)| *identity_count == count)
                            {
                                Some((_, identity)) => Arc::clone(identity),
                                None => {
                                    let element = match count {
                                        0 => Vec::new(),
                                        _ => unreduced_sum_identity(data_type)?.ok_or_else(|| {
                                            Error::from(ArrayError::MissingUnreducedSumIdentity { data_type })
                                        })?,
                                    };
                                    let identity = Arc::<[u8]>::from(element.repeat(count));
                                    identities.push((count, Arc::clone(&identity)));
                                    identity
                                }
                            };
                            let dimensions =
                                shape.as_slice().iter().map(|&dimension| dimension as u64).collect::<Vec<_>>();
                            let memory = devices.get(device_id)?.memory(r#type.memory())?;
                            let uploaded = client.buffer(
                                identity,
                                buffer_type,
                                dimensions,
                                None,
                                memory,
                                device_layout.clone(),
                            )?;
                            buffers.push(completion.track_new(&mut work, uploaded)?);
                        }
                        let output = placed.rebuilt(r#type, mesh, sharding.clone(), buffers, completion)?;
                        Ok(output.with_locally_assigned(placed.is_locally_assigned()))
                    })
                    .map_err(|error| (error, source))
            }
            DevicePlacementStrategy::HostRoundTrip { cause, process_index: owner } => match owner {
                Some(owner) if owner == process_index => {
                    // The caller reads this array back and uploads it with the target sharding.
                    Err((cause.across_processes(), Some(self)))
                }
                _ => {
                    // No target device belongs to this process, which only derives the metadata that the owning
                    // process returns, as for a compiled reshard that this process does not run.
                    let round_trip = Some((&cause, owner, mesh, sharding));
                    self.place_elsewhere(round_trip, mesh, sharding, memory, layout).map_err(|error| (error, None))
                }
            },
            DevicePlacementStrategy::OnOwner { strategy } => {
                // This process takes no part in what the owner runs. It checks what the owner checks from metadata
                // before it submits anything, which is only the upload of a host round trip, of the placement or of
                // the first step of an introduced sum, and returns the metadata of the result.
                let round_trip = match &*strategy {
                    DevicePlacementStrategy::HostRoundTrip { cause, process_index } => {
                        Some((cause, *process_index, mesh, sharding))
                    }
                    DevicePlacementStrategy::IntroducedSums {
                        mesh: first_mesh,
                        sharding: first_sharding,
                        first: Some(first),
                        ..
                    } => match &**first {
                        DevicePlacementStrategy::HostRoundTrip { cause, process_index } => {
                            Some((cause, *process_index, first_mesh, first_sharding))
                        }
                        _ => None,
                    },
                    _ => None,
                };
                self.place_elsewhere(round_trip, mesh, sharding, memory, layout).map_err(|error| (error, None))
            }
        }
    }

    /// Places this array, which has [locally assigned devices](Self::is_locally_assigned), on the devices of this
    /// process in `mesh` according to `sharding` and in `memory`, with local work only. Every other process of the
    /// placement does the same with its own such array, so the result holds the shards of this process, and the other
    /// processes hold the rest. This is how JAX places an uncommitted array on devices of several processes, except
    /// that JAX also checks that the processes hold equal values when the target includes every process. Nothing
    /// compares the values here, so every process must pass an equal value, as for [`Array::from_host_buffer`],
    /// and differing values leave the replicas inconsistent.
    ///
    /// A target shard that covers the whole array shares the buffer of this array when it is on the same device and
    /// in the same memory, and is a copy otherwise. When a target shard covers only a part of the array, that part is
    /// sliced on the device of this array, as for shard regions, and copied to its target. Slicing needs an array in
    /// the default device memory without a declared layout, and every target device must hold one value per slice.
    /// Otherwise (e.g., along an unreduced target axis, whose devices away from coordinate zero hold additive
    /// identities), this process reads its value back and uploads the shards of its devices as
    /// [`Array::from_host_buffer`] does, in the background when the PJRT plugin supports it. The execution fence of
    /// the result joins the fence of this array with the readiness of the buffer of this array that it reads, of every
    /// buffer that this function creates, and the fences of the slices (or, for the upload, the readiness that its
    /// readback waits for). When no target device belongs to this process, it reads nothing back and its result joins
    /// the fence of this array with the readiness of every buffer of this array, so the result still waits for and
    /// reports the producer of this array.
    ///
    /// `layout` is the layout that the placement records for each target shard (refer to
    /// [`Self::placement_layout`]). The upload uses it directly, and shared or sliced buffers record it only when they
    /// have it, so this strategy records the layout that the other strategies of the placement record, whichever of
    /// them another process takes.
    fn place_process_local_value(
        &self,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        memory: Memory,
        layout: Option<&Layout>,
    ) -> Result<Self, Error> {
        let session = self.session();
        let client = session.client();
        let process_index = client.process_index()?;
        let shape = self.shape();
        let target_layout = ShardLayout::new(&shape, mesh, sharding)?;
        let targets = target_layout
            .descriptors()
            .iter()
            .filter(|target| target.device().process_index() == process_index)
            .collect::<Vec<_>>();
        let host_type = ArrayType::new(self.data_type(), shape.clone().into())
            .with_memory(memory)
            .with_layout(layout.cloned());

        let on_devices_with = |slices: bool| {
            self.shards().len() == 1
                && sharding.holds_one_value_per_slice()
                && (!slices || (self.r#type().memory() == Memory::Device && self.layout().is_none()))
        };

        let slices = targets
            .iter()
            .any(|target| target.slice().iter().zip(shape.as_slice()).any(|(range, &extent)| range.len() != extent));

        // Whether some process reads its value back and uploads it reads only metadata that every process shares (a
        // process slices whenever some target shard covers a part of the value), so a process that does not upload
        // itself (e.g., one without target devices) checks, in the same order, what the readback and the upload of
        // another process check from metadata before they submit anything, and every process returns the same error.
        // What depends on the buffers or the devices of one process (e.g., a missing local buffer) remains an error
        // of that process.
        let uploads_here = !targets.is_empty() && !on_devices_with(slices);
        if !uploads_here
            && !on_devices_with(
                target_layout.descriptors().iter().any(|shard| {
                    shard.slice().iter().zip(shape.as_slice()).any(|(range, &extent)| range.len() != extent)
                }),
            )
        {
            if !shape.as_slice().contains(&0) {
                validate_one_value_per_contribution(
                    self.mesh(),
                    self.sharding(),
                    self.shards().iter().map(ArrayShard::descriptor),
                )?;
            }
            Self::validate_host_upload(session, host_type.clone(), mesh, sharding)?;
        }

        // A process without target devices takes no part, so it reads nothing back and returns the metadata of the
        // result, which waits for and reports the buffers of this array and its producer. A process that is not the
        // owner of a target on which the owner runs a shared strategy takes `DevicePlacementStrategy::OnOwner`
        // instead, and never gets here (refer to `Self::device_placement_strategy`).
        if targets.is_empty() {
            return self.place_without_local_buffers(session, host_type, mesh, sharding);
        }

        let source = match self.shards() {
            [source] if source.buffer().is_some() => Some(source),
            _ => None,
        };

        if let Some(source) = source.filter(|_| on_devices_with(slices)) {
            let devices = self.addressable_devices_by_id(process_index)?;

            // The source buffer can be the only completion of the work that produced it (e.g., of a pending host
            // upload, which has no execution fence), and no target buffer reads it when no target device belongs to
            // this process, so the result waits for it directly.
            let mut completion = PlacementCompletion::default();
            completion.track_buffer(source.buffer().unwrap())?;
            let mut work = session.reserve_work();
            let mut buffers = Vec::with_capacity(targets.len());
            for target in targets {
                let device = devices.get(target.device().id())?;
                let source_buffer = Arc::clone(source.buffer().unwrap());
                let buffer = Self::buffer_for_shard_region(
                    session,
                    self.data_type(),
                    source,
                    source_buffer,
                    target.slice(),
                    None,
                    &mut completion,
                )?;

                let target_memory = device.memory(memory)?;
                if buffer.memory()? == target_memory {
                    buffers.push(buffer);
                    continue;
                }

                buffers.push(self.copy_shard_to_memory(
                    Some(&buffer),
                    &buffer,
                    target,
                    target_memory,
                    || self.layout().filter(|_| source.slice() == target.slice()).map(ToPjrt::to_pjrt).transpose(),
                    &mut work,
                    &mut completion,
                )?);
            }

            // Whole shards keep the physical layout of this array and slices have the default layout, so the result
            // declares the layout of this array only when that is also `layout`, and declares none otherwise, which
            // leaves the conversion to `layout` to the caller (e.g., `Self::place_storage`).
            let recorded_layout = self.layout().filter(|current| Some(*current) == layout).cloned();
            let r#type = self.r#type().into_owned().with_memory(memory).with_layout(recorded_layout);
            let mut output = Self::from_addressable_buffers_internal(
                session,
                r#type.with_sharding(None)?,
                Some(sharding.clone()),
                mesh.clone(),
                None,
                buffers,
                false,
            )?;
            output.execution_fence = completion.join(self.execution_fence().cloned());
            return Ok(output);
        }

        // Read the value back and upload the shards of the devices of this process. The early return of
        // `Self::upload_host_readback` is never taken from here, because a `DataType::Zero` source is placed as
        // `DevicePlacementStrategy::Bufferless` before `ProcessLocalValue` can be chosen, and `targets` holds a
        // target device of this process.
        let (mesh, sharding) = (mesh.clone(), sharding.clone());
        self.upload_host_readback(session, mesh, sharding, memory, layout.cloned(), Error::into_pjrt_error)
    }

    /// Places this array on `mesh` according to `sharding` and in `memory` by sharing, copying, and transferring the
    /// whole shards that `plan` describes, which [`Self::device_placement_strategy`] chose with `cross_host_layouts`.
    /// `cross_host_layouts` holds the default receive layout for every shard shape that crosses processes. The result
    /// keeps the source layout when every cross-host receive uses it. Otherwise, it records no layout and the caller
    /// can apply the requested layout with [`Self::place_storage`]. Its execution fence joins the readiness of every
    /// copy, receive, and conversion with the source's execution fence. It also keeps the send fence without waiting
    /// for it, so readiness checks can report failures from those sends.
    fn place_exact_shards(
        &self,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        memory: Memory,
        plan: &ShardMovePlan,
        cross_host_layouts: &HashMap<StaticShape, ryft_pjrt::Layout>,
    ) -> Result<Self, Error> {
        let session = self.session();
        let mut completion = PlacementCompletion::default();
        let (mut buffers, layout, send_fence) = {
            let process_index = session.client().process_index()?;
            let devices = self.addressable_devices_by_id(process_index)?;

            // A replica that this process sends to several processes is prepared once. A missing replica is copied
            // to the planned sender first, because that is the device the receivers pair with their receives.
            let replicas = LocalReplicas::new(process_index);
            let mut transfers = Vec::new();
            let mut send_buffers = HashMap::new();
            for movement in plan.moves_with_role(process_index, MoveRole::Send) {
                let buffer = match send_buffers.entry(movement.source_shard_index) {
                    Entry::Occupied(entry) => Arc::clone(entry.get()),
                    Entry::Vacant(entry) => {
                        let shard = &self.shards()[movement.source_shard_index];
                        let device = devices.get(shard.device().id())?;
                        let buffer = self.buffer_on_send_device(&replicas, shard.index(), device, &mut completion)?;
                        let layout = &cross_host_layouts[&movement.shape()];

                        // The send moves the whole device allocation. Stage host buffers in device memory,
                        // then match the static shape and default layout that the receiver allocates.
                        let buffer =
                            Self::buffer_in_device_memory(session, self.r#type().memory(), buffer, &mut completion)?;
                        let buffer = Self::buffer_for_cross_host_send(
                            session,
                            self.data_type(),
                            shard.device(),
                            buffer,
                            shard.shape(),
                            layout,
                            &mut completion,
                        )?;

                        Arc::clone(entry.insert(buffer))
                    }
                };

                transfers.push(CrossHostTransfer::Send {
                    source: movement.source_device.id(),
                    target: movement.target_device.id(),
                    transfer_key: movement.transfer_key,
                    buffer,
                });
            }

            for movement in plan.moves_with_role(process_index, MoveRole::Receive) {
                transfers.push(CrossHostTransfer::Receive {
                    source: movement.source_device.id(),
                    target: movement.target_device.id(),
                    transfer_key: movement.transfer_key,
                    device: devices.get(movement.target_device.id())?,
                    element_type: plan.element_type,
                    dimensions: cross_host_shape(movement.target_shard_index, &movement.shape())?,
                });
            }

            // Each process derives the same transfer order from the plan. Checks for local copies and target memory
            // follow the transfers, so a local failure does not leave peers waiting on work we never issued.
            let (send_fence, received_buffers) = issue_cross_host_transfers(session, transfers, memory)?;
            let mut addressable_buffers = Vec::new();
            for received in received_buffers {
                completion.track_buffer(&received.buffer)?;
                addressable_buffers.push(Arc::new(received.buffer));
            }

            // Arrays never mutate their buffers and donation skips buffers that are shared, so a target shard on the
            // device of its source shard shares the buffer of that shard instead of copying it. Copying to the target
            // device itself would always allocate in its default memory, so other copies go to the requested memory.
            let source_memory = self.r#type().memory();
            let mut work = session.reserve_work();
            for movement in plan.moves_with_role(process_index, MoveRole::Copy) {
                let (shard, buffer) =
                    self.local_replica_shard(&replicas, movement.source_shard_index, movement.target_device)?;
                let same_device = shard.device() == movement.target_device;
                if same_device && memory == source_memory {
                    addressable_buffers.push(Arc::clone(buffer));
                    continue;
                }
                let target_memory = devices.get(movement.target_device.id())?.memory(memory)?;
                if same_device && buffer.memory()? == target_memory {
                    addressable_buffers.push(Arc::clone(buffer));
                    continue;
                }
                addressable_buffers.push(self.copy_shard_to_memory(
                    Some(buffer),
                    buffer,
                    shard.descriptor(),
                    target_memory,
                    || self.layout().map(ToPjrt::to_pjrt).transpose(),
                    &mut work,
                    &mut completion,
                )?);
            }

            // Local copies keep the layout of their source shards, while received buffers have the default layout. The
            // recorded layout reads the cross-host moves of every process, not only the receives of this one, so that
            // every process records the same layout, and the steps that later prepare the result (e.g., the inputs of
            // a compiled program) take the same path on every process.
            let layout = match self.layout() {
                Some(layout) => {
                    let expected = layout.to_pjrt()?;
                    plan.cross_host_moves()
                        .all(|movement| cross_host_layouts[&movement.shape()] == expected)
                        .then(|| layout.clone())
                }
                None => None,
            };

            (addressable_buffers, layout, send_fence)
        };

        // A shard without elements holds no contribution, so the plan moves nothing to it (refer to
        // `plan_exact_shard_moves`), and this process creates its own empty target shards after its transfers, as
        // shard regions do. The buffers of its empty source shards, which no move reads, still report their failures.
        let client = session.client();
        let process_index = client.process_index()?;
        completion.track_empty_shard_buffers(self)?;
        let target_layout = ShardLayout::new(&self.shape(), mesh, sharding)?;
        let empty_targets = target_layout
            .descriptors()
            .iter()
            .filter(|shard| {
                shard.device().process_index() == process_index && shard.slice().iter().any(Range::is_empty)
            })
            .collect::<Vec<_>>();
        if !empty_targets.is_empty() {
            let devices = self.addressable_devices_by_id(process_index)?;
            let device_layout = layout.as_ref().map(ToPjrt::to_pjrt).transpose()?;
            let buffer_type = self.data_type().executable_buffer_type();
            let mut work = session.reserve_work();
            for shard in empty_targets {
                let device = devices.get(shard.device().id())?;
                let dimensions = shard.slice().iter().map(|range| range.len() as u64).collect::<Vec<_>>();
                let empty = Arc::<[u8]>::from([]);
                let buffer = client.buffer(
                    empty,
                    buffer_type,
                    dimensions,
                    None,
                    device.memory(memory)?,
                    device_layout.clone(),
                )?;
                buffers.push(completion.track_new(&mut work, buffer)?);
            }
        }

        let r#type = self.r#type().into_owned().with_shape(self.shape()).with_memory(memory).with_layout(layout);
        let output = Self::from_addressable_buffers_internal(
            session,
            r#type.with_sharding(None)?,
            Some(sharding.clone()),
            mesh.clone(),
            None,
            buffers,
            false,
        )?;

        let mut output = match send_fence {
            Some(fence) => output.with_send_fence(fence),
            None => output,
        };

        output.execution_fence = completion.join(self.execution_fence().cloned());

        Ok(output)
    }

    /// Places this array on `mesh` according to `sharding` by slicing on the source devices and joining the received
    /// regions on the target devices, which [`Self::device_placement_strategy`] chose with `plan` (i.e., the moves of
    /// the regions of every target shard) and `remote_layouts` (i.e., the default layout of the shape of every region
    /// that crosses processes), both of which [`PlacementPlanningView::shard_region_plan`] derived. Every process
    /// derives the same plan, and each one performs only the part of each move that [`ShardMove::role`] assigns to it.
    ///
    /// Slices and cross-host transfers read device buffers, so a source shard that the type of this array declares in
    /// another memory, or in a layout other than the dense row-major one, is staged in device memory and in that layout
    /// first (refer to [`Self::buffer_for_shard_slicing`]), once for all of its regions and only when a move reads it.
    /// The result is in default device memory and declares no layout, whatever this array declares, because its buffers
    /// are slices, received buffers, and concatenations in the default layout, as well as whole source shards in their
    /// own layout. The caller moves it to the requested memory and layout (e.g., with [`Self::place_storage`]).
    fn place_shard_regions(
        &self,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        plan: &ShardMovePlan,
        remote_layouts: &HashMap<StaticShape, ryft_pjrt::Layout>,
    ) -> Result<Self, Error> {
        let layout = ShardLayout::new(&self.shape(), mesh, sharding)?;
        let targets = layout.descriptors();
        let session = self.session();
        let client = session.client();
        let process_index = client.process_index()?;

        let devices = self.addressable_devices_by_id(process_index)?;
        let data_type = self.data_type();
        let mut completion = PlacementCompletion::default();

        // A region that this process sends to another one leaves from the planned device, with which its receiver pairs
        // it, so a local replica that serves a planned one without a buffer is copied to that device first, once for
        // all the regions of that shard, as for sends of whole shards. This process stages a source shard (refer to
        // `Self::buffer_for_shard_slicing`) once, before its transfers when a send reads it and after them otherwise.
        let replicas = LocalReplicas::new(process_index);
        let mut work = session.reserve_work();
        let mut transfers = Vec::new();
        let mut staged_buffers = HashMap::new();
        for movement in &plan.moves {
            let device = movement.target_device;
            match movement.role(process_index) {
                Some(MoveRole::Send) => {
                    let source = &self.shards()[movement.source_shard_index];
                    let buffer = match staged_buffers.entry(source.index()) {
                        Entry::Occupied(entry) => Arc::clone(entry.get()),
                        Entry::Vacant(entry) => {
                            let source_device = devices.get(source.device().id())?;
                            let buffer =
                                self.buffer_on_send_device(&replicas, source.index(), source_device, &mut completion)?;
                            let buffer = self.buffer_for_shard_slicing(source, buffer, &mut work, &mut completion)?;
                            Arc::clone(entry.insert(buffer))
                        }
                    };
                    let buffer = Self::buffer_for_shard_region(
                        session,
                        data_type,
                        source,
                        buffer,
                        &movement.slice,
                        Some(&remote_layouts[&movement.shape()]),
                        &mut completion,
                    )?;
                    transfers.push(CrossHostTransfer::Send {
                        source: movement.source_device.id(),
                        target: device.id(),
                        transfer_key: movement.transfer_key,
                        buffer,
                    });
                }
                Some(MoveRole::Receive) => transfers.push(CrossHostTransfer::Receive {
                    source: movement.source_device.id(),
                    target: device.id(),
                    transfer_key: movement.transfer_key,
                    device: devices.get(device.id())?,
                    element_type: plan.element_type,
                    dimensions: cross_host_shape(movement.target_shard_index, &movement.shape())?,
                }),
                Some(MoveRole::Copy) | None => {}
            }
        }

        let (send_fence, received_buffers) = issue_cross_host_transfers(session, transfers, Memory::Device)?;
        let mut target_buffers = HashMap::new();
        for received in received_buffers {
            completion.track_buffer(&received.buffer)?;
            target_buffers.insert(received.transfer_key, Arc::new(received.buffer));
        }

        completion.track_empty_shard_buffers(self)?;

        // A region between two devices of this process involves no other process, so another local replica of its
        // source slice can serve it when the planned one has no buffer, as for local copies of whole shards. A staged
        // shard is the shard itself or a copy of a replica of its contribution on its device, so the regions of both
        // phases share one staged buffer for each shard.
        for movement in plan.moves_with_role(process_index, MoveRole::Copy) {
            let device = movement.target_device;
            let (source, buffer) = self.local_replica_shard(&replicas, movement.source_shard_index, device)?;
            let buffer = match staged_buffers.entry(source.index()) {
                Entry::Occupied(entry) => Arc::clone(entry.get()),
                Entry::Vacant(entry) => {
                    let buffer =
                        self.buffer_for_shard_slicing(source, Arc::clone(buffer), &mut work, &mut completion)?;
                    Arc::clone(entry.insert(buffer))
                }
            };
            let slice = &movement.slice;
            let buffer =
                Self::buffer_for_shard_region(session, data_type, source, buffer, slice, None, &mut completion)?;
            let buffer = if source.device() == device {
                buffer
            } else {
                completion.track_new(&mut work, buffer.copy_to_device(devices.get(device.id())?.clone())?)?
            };
            target_buffers.insert(movement.transfer_key, buffer);
        }

        let mut regions = vec![Vec::new(); targets.len()];
        for movement in &plan.moves {
            regions[movement.target_shard_index].push(movement);
        }

        let mut buffers = Vec::new();
        for (target, regions) in targets.iter().zip(regions) {
            let device = target.device();
            if device.process_index() != process_index {
                continue;
            }

            // An empty shard has no intersections to transfer or concatenate. Its exact shape still belongs
            // to the result, even when another dimension of that shape is nonzero.
            if target.slice().iter().any(Range::is_empty) {
                let dimensions = target.slice().iter().map(|range| range.len() as u64).collect::<Vec<_>>();
                let specification = BufferSpecification::new(data_type.executable_buffer_type(), dimensions);
                let buffer = client.uninitialized_buffer(specification, devices.get(device.id())?.clone())?;
                buffers.push(completion.track_new(&mut work, buffer)?);
                continue;
            }

            // Every region of a local target has exactly one local copy or cross-host receive.
            let missing = ArrayError::MissingShardBuffer { shard_index: target.index(), device_id: device.id() };
            let mut pieces = regions
                .into_iter()
                .map(|movement| {
                    let buffer = target_buffers.remove(&movement.transfer_key).ok_or_else(|| missing.clone())?;
                    Ok((movement.slice.as_slice(), buffer))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            if pieces.len() == 1 {
                buffers.push(pieces.pop().unwrap().1);
                continue;
            }

            let mut values = pieces
                .into_iter()
                .map(|(slice, buffer)| {
                    let shape = StaticShape::new(slice.iter().map(Range::len).collect::<Vec<_>>());
                    let r#type = ArrayType::new(data_type, shape.into());
                    let value = Self::from_device_buffer(session, r#type, target.device(), buffer)?;
                    Ok((slice, value))
                })
                .collect::<Result<Vec<_>, Error>>()?;

            // The intersections form a Cartesian grid, sorted by their starting coordinates. Join the innermost
            // dimension first so that the pieces agree along every other dimension for each concatenation.
            for axis in (0..target.slice().len()).rev() {
                let mut joined = Vec::new();
                for group in values.chunk_by(|(left, _), (right, _)| left[..axis] == right[..axis]) {
                    let value = if group.len() == 1 {
                        group[0].1.clone()
                    } else {
                        let value = Self::concatenate(group.iter().map(|(_, value)| value), axis)?;
                        completion.track_fence(value.execution_fence());
                        value
                    };
                    joined.push((group[0].0, value));
                }
                values = joined;
            }

            buffers.push(Arc::clone(values[0].1.shards()[0].buffer().unwrap()));
        }

        let r#type = self.r#type().into_owned().with_memory(Memory::Device).with_layout(None).with_sharding(None)?;
        let mut output = Self::from_addressable_buffers_internal(
            session,
            r#type,
            Some(sharding.clone()),
            mesh.clone(),
            None,
            buffers,
            false,
        )?;

        // A readback reads one replica of each tile, so the fence of the result joins every copy, receive, slice,
        // and concatenation that produced a buffer of the result with the fence of this array.
        output.execution_fence = completion.join(self.execution_fence().cloned());
        output.send_fence = send_fence;

        Ok(output)
    }

    /// Returns the result of placing this array on `mesh` according to `sharding` in `memory` in a process that takes
    /// no part in that placement (refer to [`Self::place_without_local_buffers`]). When `round_trip` is present, this
    /// process checks from metadata what the owner checks before uploading. If no process owns the round trip or the
    /// upload cannot use the requested memory and layout, every process returns its cause.
    fn place_elsewhere(
        &self,
        round_trip: Option<(&HostRoundTripCause, Option<usize>, &DeviceMesh, &Sharding)>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        memory: Memory,
        layout: Option<&Layout>,
    ) -> Result<Self, Error> {
        if let Some((cause, owner, round_trip_mesh, round_trip_sharding)) = round_trip {
            let host_type = ArrayType::new(self.data_type(), self.shape().into())
                .with_memory(memory)
                .with_layout(layout.cloned());
            if owner.is_none()
                || Self::validate_host_upload(self.session(), host_type, round_trip_mesh, round_trip_sharding).is_err()
            {
                return Err(cause.across_processes());
            }
        }
        let r#type = ArrayType::new(self.data_type(), self.shape().into()).with_memory(memory);
        self.place_without_local_buffers(self.session(), r#type, mesh, sharding)
    }

    /// Returns the result of placing this array on `mesh` with `sharding` and `r#type` in `session` without creating
    /// local PJRT buffers. This covers processes that own no target device and placements of [`DataType::Zero`] arrays.
    /// The result's execution fence joins the source's fence with the readiness of every local source buffer. A buffer
    /// can carry the only completion of the work that produced it (e.g., a pending host upload), so the result still
    /// waits for and reports that work even when this process takes no part in the reshard or host round trip.
    fn place_without_local_buffers(
        &self,
        session: &Session,
        r#type: ArrayType,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<Self, Error> {
        let mut completion = PlacementCompletion::default();

        for buffer in self.addressable_shards().filter_map(ArrayShard::buffer) {
            completion.track_buffer(buffer)?;
        }

        let (sharding, mesh) = (Some(sharding.clone()), mesh.clone());

        let mut output = if r#type.data_type().is_zero() {
            Self::zero(session, r#type, sharding, mesh)?
        } else {
            Self::from_addressable_buffers_internal(session, r#type, sharding, mesh, None, Vec::new(), false)?
        };

        output.execution_fence = completion.join(self.execution_fence().cloned());

        Ok(output)
    }

    /// Copies `buffer` to `memory`, recording the new buffer in `work` and `completion`. If no buffer has the
    /// required layout or the plugin cannot copy it, reads `source_buffer` through host memory and uploads the
    /// shard into `memory` with `fallback_layout`. An individual shard readback preserves its contribution.
    fn copy_shard_to_memory<L: FnOnce() -> Result<Option<ryft_pjrt::Layout>, LayoutError>>(
        &self,
        buffer: Option<&Arc<Buffer<'static>>>,
        source_buffer: &Arc<Buffer<'static>>,
        shard: &ShardDescriptor,
        memory: PjrtMemory<'_>,
        fallback_layout: L,
        work: &mut WorkReservation,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        if let Some(buffer) = buffer {
            match buffer.copy_to_memory(memory) {
                Ok(copied) => return completion.track_new(work, copied),
                Err(PjrtError::Unimplemented { .. } | PjrtError::MissingFunction { .. }) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Self::copy_shard_through_host(
            self.session(),
            source_buffer,
            shard.shape(),
            self.data_type().executable_buffer_type(),
            memory,
            fallback_layout()?,
            completion,
        )
    }

    /// Returns a new buffer in `memory` and in `layout` (or the default layout of the PJRT plugin for [`None`])
    /// that holds the shard with `shape` that `buffer` holds, copied through host memory in dense row-major order.
    /// `completion` records the readiness of the new buffer, and the work tracking of `session` includes it, so
    /// [`Session::synchronize`] waits for it.
    ///
    /// The round trip is its own job on the background upload workers of `session`, which reads the shard back once
    /// `buffer` is ready and uploads it into the new buffer, which this function creates up front, so this function
    /// returns without waiting for the job. The job waits only for `buffer` before it occupies a worker, so the round
    /// trips of different shards are independent, and one never waits for the source of another (e.g., for a source
    /// that a producer fills only once the round trip of another shard is ready). Each shard is read back from its own
    /// device, because devices can hold different values for the same slice (e.g., the contributions along unreduced
    /// axes). The job owns its own handle of `buffer`, so that buffer stays alive until the job has read it back. The
    /// job is handed over before anything observes the new buffer, because nothing else can complete that buffer, and
    /// observing it can wait for it (e.g., a fence built from its event awaits an event whose callback PJRT fails to
    /// register). A failed readback or upload poisons the new buffer, and if the job cannot be handed over, dropping
    /// its transfer manager poisons it as well. This function blocks on the readback only when the PJRT plugin lacks
    /// host to device transfer managers or their literal transfers.
    fn copy_shard_through_host(
        session: &Session,
        buffer: &Arc<Buffer<'static>>,
        shape: StaticShape,
        buffer_type: BufferType,
        memory: PjrtMemory<'_>,
        layout: Option<ryft_pjrt::Layout>,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        let client = session.client();
        let dimensions = shape.as_slice().iter().map(|&size| size as u64).collect::<Vec<_>>();
        let mut work = session.reserve_work();
        let placed = match PendingHostUpload::new(client, buffer_type, dimensions.clone(), memory, layout.clone())? {
            Some((upload, placed)) => {
                let source = Arc::clone(buffer);
                let sources = ExecutionFence::new(vec![buffer.ready()?]);
                session.host_upload_workers().submit(sources, move || {
                    run_host_uploads(
                        vec![upload],
                        |_| read_shard_to_host(&source, &shape, Event::r#await).map(HostBufferSlice::from),
                        Error::into_pjrt_error,
                    )
                })?;
                placed
            }
            None => {
                let bytes = read_shard_to_host(buffer, &shape, |copy| work.track_host_copy(copy).wait())?;
                client.buffer(Arc::<[u8]>::from(bytes), buffer_type, dimensions, None, memory, layout)?
            }
        };
        completion.track_new(&mut work, placed)
    }

    /// Uploads the value of this array, which this process reads back as [`Self::copy_to_host_bytes`] does, onto `mesh`
    /// according to `sharding`, in `memory` and `layout`, as an array of `session`. This is the host round trip that
    /// the host fallback of [`Self::place_within_client`], [`Self::place_across_clients`], and the upload of
    /// [`Self::place_process_local_value`] share. The target buffers are created up front and the readback and the
    /// upload run in the background (refer to [`Self::from_host_buffer_in_background`]), whose failures, including
    /// those of the work that produces this array, `poison` converts into the error that the target buffers report.
    /// Only when the PJRT plugin lacks host to device transfer managers or their literal transfers does this function
    /// read back and upload on the calling thread, and the execution fence of the result then joins the readiness of
    /// its uploads and the execution fence of this array. A process that owns no device of `mesh` reads nothing back,
    /// and a [`DataType::Zero`] value has no bytes to read back or upload, so both return without waiting the result
    /// of [`Self::place_without_local_buffers`], which waits for and reports the work that produces this array,
    /// after the checks of the upload that read only metadata (refer to [`Self::validate_host_upload`]).
    fn upload_host_readback<P: 'static + Send + FnOnce(Error) -> PjrtError>(
        &self,
        session: &Session,
        mesh: DeviceMesh,
        sharding: Sharding,
        memory: Memory,
        layout: Option<Layout>,
        poison: P,
    ) -> Result<Self, Error> {
        let host_type = ArrayType::new(self.data_type(), self.shape().into()).with_memory(memory).with_layout(layout);
        let process_index = session.client().process_index()?;
        if self.data_type().is_zero() || mesh.devices().iter().all(|device| device.process_index() != process_index) {
            Self::validate_host_upload(session, host_type.clone(), &mesh, &sharding)?;
            return self.place_without_local_buffers(session, host_type, &mesh, &sharding);
        }

        let sources = self.host_readback_fence()?;
        let readback_source = self.clone();
        let uploaded = Self::from_host_buffer_in_background(
            session,
            host_type.clone(),
            mesh.clone(),
            sharding.clone(),
            sources,
            move || readback_source.copy_to_host_bytes(),
            poison,
        )?;

        if let Some(array) = uploaded {
            return Ok(array);
        }

        // Without host to device transfer managers, read back and upload on this thread.
        let host_bytes = self.copy_to_host_bytes()?;
        Self::from_host_buffer(session, host_type, mesh, Some(sharding), host_bytes.as_slice())
            .and_then(|array| array.with_upload_completion(self.execution_fence().cloned()))
    }

    /// Places this array on `mesh` according to `sharding` with device work only (i.e., without copying its contents
    /// through host memory), except that a source whose devices this process chose on its own can be read back and
    /// uploaded (refer to [`Self::place_process_local_value`]) and that a shard whose storage the PJRT plugin cannot
    /// change on its device is copied through host memory on its own (refer to [`Self::copy_shard_through_host`]).
    /// [`Self::to_placement`] and [`Self::into_placement`] start with this work, and the execution of a compiled
    /// program uses it for inputs whose sharding differs from the one that the program was compiled for. Refer to
    /// [`Self::try_reshard_on_devices`] for the strategies that it tries.
    ///
    /// The result has the memory of this array and the layout that [`Self::placement_layout`] keeps for the target
    /// shards, which a device reshard changes neither, as [`Self::place_within_client`] gives its result the storage
    /// that it promises (e.g., an executable that declares the layout of its input accepts the result of a reshard of
    /// that input).
    ///
    /// A result on a single device keeps the [commitment](Self::is_committed) of this array, because this is how arrays
    /// move to the devices that an operation runs on rather than to devices that a caller chose. A result on more than
    /// one device is committed, like every eager result on more than one device (refer to [`Self::commit_if`]). For the
    /// same reason, a result on devices of this process only keeps the locally assigned devices of this array, while a
    /// result on devices of other processes never has them.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that executes the reshard and owns the resulting array. It must use the same PJRT
    ///     client as this array.
    ///   - `mesh`: Target [`DeviceMesh`].
    ///   - `sharding`: Target [`Sharding`].
    ///   - `donate`: Whether a compiled reshard on the mesh of this array may reuse the buffers of this array for
    ///     its output. The donation only takes effect when this array is the only handle of its buffers.
    pub(crate) fn reshard_on_devices(
        self,
        session: &Session,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        donate: bool,
    ) -> Result<Self, Error> {
        let memory = self.r#type().memory();
        let layout = self.placement_layout(None, mesh, sharding)?;
        let (committed, locally_assigned) = (self.is_committed(), self.is_locally_assigned());
        let array = self.with_session(session)?;

        // Some strategies (e.g., a compiled reshard) record no layout, so the result takes the memory and the layout
        // of this array as the public placement path gives a result its storage, whatever strategy placed it.
        array
            .try_reshard_on_devices(mesh, sharding, memory, layout.as_ref(), donate, false)
            .map_err(|(error, _)| error)
            .and_then(|output| output.place_storage(memory, layout))
            .map(|output| output.commit_if(committed).with_locally_assigned(locally_assigned))
    }

    /// Places this array on `mesh` according to `sharding` with device work only, and on failure also returns this
    /// array when a host fallback may still place it. [`Self::device_placement_strategy`] chooses the strategy (refer
    /// to the documentation of this module), and [`Self::place_with_strategy`] performs the part of this process in it.
    /// The errors of a planning and of moves of exact shards are final, so they come without this array. A donated
    /// array is still returned when its compiled reshard fails to stage or compile. After a failed execution, it is
    /// returned only when the execution could not take its buffers.
    ///
    /// Unsupported requests return typed [`ArrayError`] variants through [`Error::Array`] (refer to
    /// [`PlacementPlanningView::strategy`]). Cross-host extension, tracing, compilation, and execution failures
    /// propagate through [`Error`] without losing their typed payloads.
    ///
    /// `layout` is the layout of each target shard that the placement records (refer to [`Self::placement_layout`]),
    /// which [`Self::place_process_local_value`] uploads into directly. `host_fallback` is whether the caller falls
    /// back to a host round trip. Without one (e.g., for [`Self::reshard_on_devices`]), a change of reduction state
    /// that only a host round trip makes exactly is an error right after the planning, whichever process could make
    /// it (refer to [`HostRoundTripCause::without_host_route`]).
    // The error carries the unconsumed array back to the caller, which falls back to a host transfer with it.
    #[allow(clippy::result_large_err)]
    fn try_reshard_on_devices(
        self,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        memory: Memory,
        layout: Option<&Layout>,
        donate: bool,
        host_fallback: bool,
    ) -> Result<Self, (Error, Option<Self>)> {
        let process_index = match self.session().client().process_index() {
            Ok(process_index) => process_index,
            Err(error) => return Err((error.into(), None)),
        };

        // A planning error reads only shared metadata, so every process meets it alike, and no process recovers from
        // it with a host fallback of its own (a decline that one process can make with a host round trip is planned
        // as one). Only failures that this process observes while carrying out a planned strategy (e.g., a failed
        // compilation or execution on its devices) can return this array for that fallback.
        let layout_resolver = cross_host_layout_resolver(self.client());
        let strategy = match self.device_placement_strategy(&layout_resolver, mesh, sharding, process_index) {
            Ok(strategy) => strategy,
            Err(error) => return Err((error, None)),
        };

        if let Some(cause) = strategy.host_round_trip_cause().filter(|_| !host_fallback) {
            return Err((cause.without_host_route(), None));
        }

        let array = if !self.data_type().is_zero()
            && self.is_locally_assigned()
            && self.mesh() != mesh
            && self.mesh().logical_mesh() == mesh.logical_mesh()
            && mesh.devices().iter().all(|device| device.process_index() == process_index)
            && keeps_some_contributions(self.sharding(), sharding)
        {
            // The owner can run the shared plan after a local whole-shard placement that preserves each
            // contribution. If that move fails, the original array remains available for host fallback.
            let (shape, element_type) = (self.shape(), self.data_type().executable_buffer_type());
            let sources = self.shards().iter().map(ArrayShard::descriptor);
            let source_sharding = self.sharding();
            let moved = plan_exact_shard_moves(
                element_type,
                sources,
                self.mesh(),
                source_sharding,
                &shape,
                mesh,
                source_sharding,
            )
            .and_then(|plan| {
                let plan = plan.ok_or_else(|| {
                    PjrtError::internal("the shards of a locally assigned array cannot move as they are")
                })?;
                self.place_exact_shards(mesh, source_sharding, self.r#type().memory(), &plan, &HashMap::new())
            });
            match moved {
                Ok(moved) => moved.with_committed(self.is_committed()).with_locally_assigned(true),
                Err(error) => return Err((error, Some(self))),
            }
        } else {
            self
        };

        array.place_with_strategy(strategy, process_index, mesh, sharding, memory, layout, donate)
    }

    /// Tries a layout conversion on one device, keeping the shard's contents and memory unchanged, and returns the
    /// converted buffer, whose execution fence `completion` records. Returns `None` when lowering or compilation does
    /// not support the conversion, so the caller can upload the shard through host memory instead. Errors from
    /// executing a compiled conversion are propagated.
    fn try_relayout_shard_on_device(
        session: &Session,
        device: Device,
        buffer: Arc<Buffer<'static>>,
        input_type: ArrayType,
        layout: &Layout,
        completion: &mut PlacementCompletion,
    ) -> Result<Option<Arc<Buffer<'static>>>, Error> {
        // A single-device program treats this buffer as an independent value, even when it holds only one
        // contribution to an unreduced array. Its input uses the buffer's existing layout.
        let input = Self::from_device_buffer(session, input_type.clone(), device, buffer)?;
        let output_type = input_type.clone().with_layout(layout.clone());
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let argument = builder.add_input(input_type.clone().into());
        let cast = CastOperation::<Type>::checked(output_type.into(), vec![input_type.into()])?;
        let output = builder.add_instruction(cast, Vec::new(), vec![argument], None)?[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder])?;
        let compiled = session
            .lower_program(&program, 0, &Options::new(input.mesh().clone()))
            .and_then(|lowered| session.compilation_cache().compile(session, &lowered));
        match compiled {
            Ok(program) => {
                let converted = session.execute_program(&program, vec![input])?.pop().unwrap();
                completion.track_fence(converted.execution_fence());
                Ok(Some(Arc::clone(converted.shards()[0].buffer().unwrap())))
            }
            Err(
                Error::Lowering(
                    LoweringError::UnsupportedOperation { .. }
                    | LoweringError::UnsupportedElementType { .. }
                    | LoweringError::UnsupportedBoundaryMemory { .. },
                )
                | Error::Pjrt(PjrtError::Unimplemented { .. } | PjrtError::MissingFunction { .. }),
            ) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Validates the target and checks that required whole-shard moves from a shared array have local sources.
    pub(super) fn validate_array_target(
        &self,
        session: &Session,
        target: Option<PlacementRequest>,
    ) -> Result<(), Error> {
        let Some(target) = target else { return Ok(()) };
        let placement = self.validate_placement_target(session, target, true)?;
        if !std::ptr::eq(self.client(), session.client()) {
            return Ok(());
        }

        let (mesh, sharding) = (placement.mesh(), placement.sharding());
        if self.is_locally_assigned()
            || self.data_type().is_zero()
            || (self.mesh() == mesh && self.sharding() == sharding)
        {
            return Ok(());
        }

        let process_index = self.client().process_index()?;
        let layout_resolver = cross_host_layout_resolver(self.client());
        if self
            .shards()
            .iter()
            .any(|shard| shard.device().process_index() == process_index && shard.buffer().is_none())
            && let Ok(DevicePlacementStrategy::ExactShards { plan, .. }) =
                self.device_placement_strategy(&layout_resolver, mesh, sharding, process_index)
        {
            let replicas = LocalReplicas::new(process_index);
            for movement in plan.moves.iter().filter(|movement| movement.source_device.process_index() == process_index)
            {
                self.local_replica_shard(&replicas, movement.source_shard_index, movement.target_device)?;
            }
        }

        Ok(())
    }

    /// Resolves `request` into a target placement and validates it from metadata. For a target of another PJRT
    /// client, validation uses [`Self::validate_placement_from_other_client`]. Otherwise it checks the devices
    /// with [`validate_mesh_devices`] if `check_devices` is set and checks the layout with
    /// [`Self::validate_placement_layout`]. The admission of [`Session::place`] and a placement that compares values,
    /// which submits its comparison before [`Self::place_within_client`] checks the devices, set `check_devices`.
    /// Any other placement leaves that check, and the order of its errors, to [`Self::place_within_client`].
    pub(super) fn validate_placement_target(
        &self,
        session: &Session,
        request: PlacementRequest,
        check_devices: bool,
    ) -> Result<Placement, Error> {
        let target = self.resolve_placement_request(request)?;
        if std::ptr::eq(self.client(), session.client()) {
            if check_devices {
                validate_mesh_devices(session, target.mesh())?;
            }
            self.validate_placement_layout(session, &target)?;
        } else {
            self.validate_placement_from_other_client(session, &target)?;
        }
        Ok(target)
    }

    /// Resolves the target layout and checks that the plugin can create it before this placement submits any
    /// device work. A placement that keeps the devices, sharding, and storage can reuse its existing buffers.
    fn validate_placement_layout(&self, session: &Session, target: &Placement) -> Result<Option<Layout>, Error> {
        let layout = self.placement_layout(target.layout(), target.mesh(), target.sharding())?;
        if self.mesh() != target.mesh()
            || self.sharding() != target.sharding()
            || self.layout() != layout.as_ref()
            || target.memory().is_some_and(|memory| memory != self.r#type().memory())
        {
            validate_device_layout(session.client(), self.data_type(), layout.as_ref(), false)?;
        }
        Ok(layout)
    }

    /// Checks, from metadata alone, a placement of this array, whose PJRT client is not that of `session`, onto
    /// `target`, which [`Self::place_across_clients`] and the admission of an [`Array`] leaf of [`Session::place`] both
    /// call, so that they admit the same placements. PJRT copies buffers only within one client, so such a placement
    /// reads the value back and uploads it. The target session must recognize the mesh. For a value with bytes, the
    /// transfer must keep no distinct contributions apart. Each process that owns a target device must be able to read
    /// the full value back. A [locally assigned](Self::is_locally_assigned) source gives each process its own value.
    /// A standalone bridge between two fully local meshes permits different client process indices. Shared distributed
    /// placement requires a common process numbering scheme and a single target owner, which every participant
    /// identifies and checks against the source shard metadata.
    fn validate_placement_from_other_client(&self, session: &Session, target: &Placement) -> Result<(), Error> {
        validate_mesh_devices(session, target.mesh())?;

        let layout = self.placement_layout(target.layout(), target.mesh(), target.sharding())?;
        validate_device_layout(session.client(), self.data_type(), layout.as_ref(), false)?;

        if self.data_type().is_zero() || self.shape().as_slice().contains(&0) {
            return Ok(());
        }

        if keeps_some_contributions(self.sharding(), target.sharding()) {
            return Err(PjrtError::invalid_argument(
                "cannot place an array of another PJRT client onto a target that keeps the contributions \
                 of its pending reduction or manual variation apart, because the transfer between the clients reads \
                 back their combined value",
            )
            .into());
        }

        let process_index = session.client().process_index()?;
        let devices = target.mesh().devices();
        let processes = devices.iter().map(Device::process_index).collect::<HashSet<_>>();

        if self.is_locally_assigned() {
            if processes.contains(&process_index) {
                self.validate_host_readback()?;
            }
            return Ok(());
        }

        if processes.len() > 1 {
            return Err(PjrtError::invalid_argument(
                "cannot place an array of another PJRT client onto the devices of more than one process \
                 unless every process holds its own value of the array",
            )
            .into());
        }

        let owner = devices[0].process_index();
        if owner == process_index {
            let source_process = self.client().process_index()?;
            if source_process != owner
                && self.mesh().devices().iter().all(|device| device.process_index() == source_process)
                && self.session().is_standalone(source_process)?
                && session.is_standalone(owner)?
            {
                self.validate_host_readback()?;
                return Ok(());
            }
        }

        // Only the owner of the target reads this array back. All participants check the source shards
        // using the common process numbering before the owner checks its buffers.
        validate_one_value_per_contribution(
            self.mesh(),
            self.sharding(),
            self.shards().iter().map(ArrayShard::descriptor),
        )?;

        if let Some(shard) = self.placement_planning_view().contribution_missing_from(owner) {
            return Err(PjrtError::invalid_argument(format!(
                "cannot place an array of another PJRT client onto the devices of process {}, which holds no replica \
                 of shard #{} of the array, because the transfer between the clients reads back the value of the array \
                 on that process",
                owner,
                shard.index(),
            ))
            .into());
        }

        if owner == process_index {
            self.validate_host_readback()?;
        }

        Ok(())
    }

    /// Returns the shard of this array that serves a move between two devices of this process, whose process index is
    /// `process_index`, to `target`, together with its buffer. The move names the shard at `preferred`, which a plan
    /// chose from metadata that every process shares (e.g., the source of a local copy of whole shards or of a shard
    /// region), and that shard serves whenever it has a buffer. Otherwise, another replica of its slice on this process
    /// with a buffer serves instead (e.g., for an array that holds only some of its replicas), the one on `target`
    /// first and then the one with the lowest device identifier, so that the choice stays deterministic. The replicas
    /// are the shards that hold the same contribution (i.e., of the same slice and with the same coordinates along
    /// unreduced and varying manual axes, refer to [`ShardContribution`]), so a replica holds the same bytes also for
    /// an array with a pending reduction (e.g., the source of an introduced sum, refer to
    /// [`Self::buffer_for_local_contribution`]) or with variation along a manual axis. Only a move without another
    /// process may read another replica directly, because the peer of a cross-host transfer pairs it with the device
    /// of the preferred shard, so a cross-host send first copies that replica to that device (refer to
    /// [`Self::buffer_on_send_device`]). `replicas` is the index of the local replicas of this array that the caller
    /// keeps for the whole placement, so that many targets of one resident replica cost linear work in all.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::MissingShardBuffer`] for the preferred shard when no replica of its contribution on this
    /// process has a buffer.
    fn local_replica_shard(
        &self,
        replicas: &LocalReplicas,
        preferred: ShardIndex,
        target: Device,
    ) -> Result<(&ArrayShard, &Arc<Buffer<'static>>), Error> {
        let preferred_shard = &self.shards()[preferred];
        if let Some(buffer) = preferred_shard.buffer() {
            return Ok((preferred_shard, buffer));
        }

        let process_index = replicas.process_index;
        let holds_buffer =
            |shard: &ArrayShard| shard.device().process_index() == process_index && shard.buffer().is_some();

        let index = replicas.shards.get_or_init(|| {
            let mut index = HashMap::<ShardContribution, Vec<ShardIndex>>::new();
            for shard in self.shards().iter().filter(|shard| holds_buffer(shard)) {
                index.entry(self.shard_contribution(shard.descriptor())).or_default().push(shard.index());
            }

            for shards in index.values_mut() {
                shards.sort_unstable_by_key(|&shard| (self.shards()[shard].device().id(), shard));
            }

            index
        });

        let contribution = self.shard_contribution(preferred_shard.descriptor());

        // The replica on `target` serves first, and otherwise the one with the lowest device identifier.
        self.device_shard(target.id())
            .filter(|shard| {
                holds_buffer(shard)
                    && shard.device() == target
                    && self.shard_contribution(shard.descriptor()) == contribution
            })
            .or_else(|| index.get(&contribution).and_then(|shards| shards.first()).map(|&shard| &self.shards()[shard]))
            .map(|shard| (shard, shard.buffer().unwrap()))
            .ok_or_else(|| {
                Error::from(ArrayError::MissingShardBuffer {
                    shard_index: preferred,
                    device_id: preferred_shard.device().id(),
                })
            })
    }

    /// Returns a buffer on the device of the shard of this array at `index`, a device of this process, whose local
    /// replicas `replicas` indexes, that holds the contribution of that shard: its own buffer, or otherwise a copy,
    /// in the memory that the type of this array declares, of the replica of the same contribution on this process that
    /// [`Self::local_replica_shard`] selects. A shard without elements and without its own buffer needs no replica and
    /// gets a new empty buffer in that memory and in the declared layout of this array. `devices` holds the addressable
    /// devices of the client of this array, which the caller looks up once for all its shards. `work` and `completion`
    /// record the copy or the new buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::MissingShardBuffer`] for that shard when it has elements and no replica of its
    /// contribution on this process has a buffer.
    fn buffer_for_local_contribution(
        &self,
        replicas: &LocalReplicas,
        index: ShardIndex,
        devices: &AddressableDevices,
        work: &mut WorkReservation,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        let shard = &self.shards()[index];
        let device = shard.device();
        if shard.buffer().is_none() && shard.slice().iter().any(Range::is_empty) {
            // A shard without elements holds no contribution, so it gets a new empty buffer rather than a replica.
            let dimensions = shard.shape().as_slice().iter().map(|&dimension| dimension as u64).collect::<Vec<_>>();
            let layout = self.layout().map(ToPjrt::to_pjrt).transpose()?;
            let memory = devices.get(device.id())?.memory(self.r#type().memory())?;
            let buffer_type = self.data_type().executable_buffer_type();
            let empty = self.client().buffer(Arc::<[u8]>::from([]), buffer_type, dimensions, None, memory, layout)?;
            return completion.track_new(work, empty);
        }

        let (replica, buffer) = self.local_replica_shard(replicas, index, device)?;
        if replica.device() == device {
            return Ok(Arc::clone(buffer));
        }

        let copied = buffer.copy_to_memory(devices.get(device.id())?.memory(self.r#type().memory())?)?;
        completion.track_new(work, copied)
    }

    /// Returns a buffer for the shard at `index` on the local `device` named by the cross-host transfer plan.
    /// Receivers pair their transfers with that device, which the plan chose from metadata shared by every process.
    /// This function reuses the shard's own buffer when present. Otherwise, [`Self::local_replica_shard`] uses
    /// `replicas` to find a local copy of the same contribution. Its buffer is copied to `device` in the memory
    /// that the array's type declares. `completion` records the readiness of the returned buffer in both cases,
    /// so that the result of a placement that only sends from this process (e.g., to a target without devices of this
    /// process) still reports a failure of the work that produced the sent buffer, which may have no other fence (e.g.,
    /// a pending host upload, whose readiness only its buffer carries) and whose failure a plugin need not report
    /// through the send.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::MissingShardBuffer`] for the shard at `index` when no replica of its slice on this process
    /// has a buffer.
    fn buffer_on_send_device(
        &self,
        replicas: &LocalReplicas,
        index: ShardIndex,
        device: &PjrtDevice<'static>,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        let shard = &self.shards()[index];
        let (replica, buffer) = self.local_replica_shard(replicas, index, shard.device())?;
        if replica.index() == index {
            completion.track_buffer(buffer)?;
            return Ok(Arc::clone(buffer));
        }
        let mut work = self.session().reserve_work();
        let copied = buffer.copy_to_memory(device.memory(self.r#type().memory())?)?;
        completion.track_new(&mut work, copied)
    }

    /// Returns `buffer`, which holds a shard of an array whose type declares `memory`, in the default memory of its
    /// device, copying it there unless `memory` is [`Memory::Device`]. `completion` records the readiness of the copy.
    /// Cross-host transfers and the programs that slice shard regions read device buffers (e.g., a cross-host transfer
    /// of the CUDA plugin waits for the GPU completion events of its buffer, which a buffer in host memory does not
    /// have, as an unpinned host buffer is a plain CPU allocation). The declared memory is information that every
    /// process shares, so every process stages the same shards, and the copy is local to the device of `buffer`.
    fn buffer_in_device_memory(
        session: &Session,
        memory: Memory,
        buffer: Arc<Buffer<'static>>,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        if memory == Memory::Device {
            return Ok(buffer);
        }
        let mut work = session.reserve_work();
        let staged = buffer.copy_to_memory(buffer.device()?.default_memory()?)?;
        completion.track_new(&mut work, staged)
    }

    /// Returns `buffer`, which holds a shard or a region with `shape` of an array of `data_type` on `device`, in the
    /// default memory of that device, as a buffer that a cross-host send can move to a receiver that allocates a static
    /// buffer of `shape` in `layout`. A static `buffer` of that shape and layout is reused. Otherwise, this function
    /// prepares a static buffer with the same elements, as every sender of a cross-host transfer does. The conversion
    /// runs on `device` with [`Self::try_relayout_shard_on_device`], whose program has the static type of `shape` and
    /// so also drops the trailer of a buffer whose type has dynamic dimensions, and `completion` then records its
    /// fence. When the plugin cannot lower or compile that conversion, the shard is copied through host memory into
    /// a new buffer in `layout` with [`Self::copy_shard_through_host`] instead, and `completion` records the readiness
    /// of that buffer. That is the only entry of a host round trip for a cross-host send.
    fn buffer_for_cross_host_send(
        session: &Session,
        data_type: DataType,
        device: Device,
        buffer: Arc<Buffer<'static>>,
        shape: StaticShape,
        layout: &ryft_pjrt::Layout,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        // A send moves the whole allocation. Bounded-dynamic buffers can carry an extent trailer even when their
        // physical dimensions equal the shard shape, while the receiver allocates a static buffer with no trailer.
        if buffer.dynamic_dimensions()?.is_empty()
            && buffer.dimensions()?.iter().map(|&size| size as usize).eq(shape.as_slice().iter().copied())
            && buffer.specification()?.layout == *layout
        {
            return Ok(buffer);
        }

        let current_layout = Layout::from_pjrt(buffer.specification()?.layout)?;
        let input_type = ArrayType::new(data_type, shape.clone().into()).with_layout(current_layout);
        let target_layout = Layout::from_pjrt(layout.clone())?;
        let source = Arc::clone(&buffer);

        if let Some(converted) =
            Self::try_relayout_shard_on_device(session, device, source, input_type, &target_layout, completion)?
        {
            return Ok(converted);
        }

        let buffer_type = data_type.executable_buffer_type();
        let memory = buffer.memory()?;

        Self::copy_shard_through_host(session, &buffer, shape, buffer_type, memory, Some(layout.clone()), completion)
    }

    /// Returns `buffer`, which holds the contribution of `shard`, a shard of this array on a device of
    /// this process, on the device of `shard`, staged for slicing and cross-host sends of its regions (refer to
    /// [`Self::place_shard_regions`]). That is the default memory of that device and, when this array declares a
    /// layout other than the dense row-major one (refer to [`PlacementPlanningView::shard_region_plan`]), a conversion
    /// of this shard alone to the dense row-major layout on its device (refer to [`Self::place_shard_storage`]), which
    /// moves bytes within the shard and keeps its contribution. The conversion lands in default device memory, where
    /// every plugin can hold the dense layout (e.g., the CPU plugin cannot in pinned host memory). `work` and
    /// `completion` record the copies and conversions.
    fn buffer_for_shard_slicing(
        &self,
        shard: &ArrayShard,
        buffer: Arc<Buffer<'static>>,
        work: &mut WorkReservation,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        let dense = dense_row_major_layout(self.shape().rank());
        if self.layout().is_some_and(|layout| *layout != dense) {
            return self.place_shard_storage(shard, &buffer, Memory::Device, Some(&dense), work, completion);
        }
        Self::buffer_in_device_memory(self.session(), self.r#type().memory(), buffer, completion)
    }

    /// Produces the region with `slice`, in global array coordinates, from `source_buffer`, which holds the shard
    /// `source` on its device, on that device. `source_buffer` must be in the default memory of that device. A region
    /// that a remote transfer sends (i.e., with `remote_layout`, which is the default layout of the plugin for its
    /// shape) is a static buffer of its shape in that layout, because a cross-host send moves the whole allocation
    /// of its buffer, so [`Self::buffer_for_cross_host_send`] converts a region whose allocation differs from the
    /// receiver's (e.g., a whole source shard of a bounded-dynamic result). `completion` records the fences of the
    /// programs and the readiness of the buffers that produce it.
    fn buffer_for_shard_region(
        session: &Session,
        data_type: DataType,
        source: &ArrayShard,
        source_buffer: Arc<Buffer<'static>>,
        slice: &[Range<usize>],
        remote_layout: Option<&ryft_pjrt::Layout>,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        let mut buffer = if source.slice() == slice {
            source_buffer
        } else {
            let input = Self::from_device_buffer(
                session,
                ArrayType::new(data_type, source.shape().into()),
                source.device(),
                source_buffer,
            )?;
            let starts = source
                .slice()
                .iter()
                .zip(slice)
                .map(|(shard, region)| region.start - shard.start)
                .collect::<Vec<_>>();
            let limits = starts.iter().zip(slice).map(|(start, range)| start + range.len()).collect::<Vec<_>>();
            let output = input.slice(&starts, &limits, &vec![1; starts.len()])?;
            completion.track_fence(output.execution_fence());
            Arc::clone(output.shards()[0].buffer().unwrap())
        };

        // A receive declares the shape of the region and uses the default device layout of the plugin.
        if let Some(layout) = remote_layout {
            let shape = StaticShape::new(slice.iter().map(Range::len).collect::<Vec<_>>());
            buffer = Self::buffer_for_cross_host_send(
                session,
                data_type,
                source.device(),
                buffer,
                shape,
                layout,
                completion,
            )?;
        }

        Ok(buffer)
    }

    /// Borrows the source metadata and global shard descriptors without exposing its buffers or session
    /// and returns them as a [`PlacementPlanningView`].
    #[inline]
    fn placement_planning_view(&self) -> PlacementPlanningView<'_, impl Iterator<Item = &ShardDescriptor> + Clone> {
        PlacementPlanningView {
            metadata: &self.parts.metadata,
            shards: self.shards().iter().map(ArrayShard::descriptor),
        }
    }

    /// Chooses the device work that places this array on `mesh` according to `sharding` in the process with index
    /// `process_index`, or returns the error that rules out all of it. `layout_resolver` supplies the layouts needed
    /// for cross-host transfers. [`Self::try_reshard_on_devices`] executes the chosen strategy, and the agreement test
    /// of this module plans every process of a distributed placement with this same function.
    ///
    /// Every process of a distributed placement calls this function for the same placement and then performs only its
    /// own part of the chosen strategy, so every process must derive the same transfers and programs. This function
    /// therefore reads only the information that every process shares. It never reads a buffer or a physical layout,
    /// and it reads `process_index` only for a locally assigned array.
    ///
    /// An array with [locally assigned devices](Self::is_locally_assigned) is placed with
    /// [`DevicePlacementStrategy::ProcessLocalValue`], except that when one process owns every target device, that
    /// owner takes the strategy that [`PlacementPlanningView::strategy`] chooses for the view of its value on its
    /// devices, if one applies, and every other process takes [`DevicePlacementStrategy::OnOwner`] with that plan,
    /// so that it admits exactly what the owner runs. Any other array is placed with the strategies of
    /// [`PlacementPlanningView::strategy`].
    fn device_placement_strategy(
        &self,
        layout_resolver: &CrossHostLayoutResolver<'_>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        process_index: usize,
    ) -> Result<DevicePlacementStrategy, Error> {
        if self.data_type().is_zero() {
            return Ok(DevicePlacementStrategy::Bufferless);
        }

        if !self.is_locally_assigned() {
            return self.placement_planning_view().strategy(layout_resolver, mesh, sharding);
        }

        // Placing the value with local work only also handles every placement that no shared strategy supports (e.g.,
        // an uneven split whose regions need a cross-host layout that the plugin does not report), so it remains the
        // choice when none applies. Its readback and upload would combine contributions that the target keeps (refer
        // to `keeps_some_contributions`), so such a placement is made by the one process that owns every target device,
        // and a target over several processes is the same error on every process. Whether one process owns the target
        // reads only the target, so every process decides alike.
        //
        // When one process owns the target, every process decides which strategy that owner runs from the plan of the
        // view of the value on the owner's devices, which reads only metadata that all of them hold. The owner runs a
        // shared strategy whenever one applies, which for a placement that keeps contributions is the only choice (so
        // its rejection is the same error on every process), and otherwise only when it is not a host round trip for
        // a cause on the devices, which the upload of the owner replaces. Every other process then takes no part and
        // checks only what that strategy checks before it submits anything, or, when the owner uploads its own value,
        // what that upload checks (refer to `Self::place_process_local_value`). The owner runs the plan of the view
        // over the logical mesh of the target, after moving its value there when the placement keeps contributions
        // and the logical meshes match (refer to `Self::try_reshard_on_devices`). Otherwise, it plans again from
        // its own value.
        let owner = mesh.devices()[0].process_index();
        let one_owner = mesh.devices().iter().all(|device| device.process_index() == owner);
        let keeps = keeps_some_contributions(self.sharding(), sharding);
        if !one_owner && keeps {
            return Err(ProgramError::UnsupportedOperation {
                message: COMBINED_CONTRIBUTIONS_ERROR_MESSAGE.to_string(),
            }
            .into());
        }

        if !one_owner {
            return Ok(DevicePlacementStrategy::ProcessLocalValue);
        }

        // Give the owner's view the source topology on the target devices. If the source needs more devices.
        // every process fills out the view with the same synthetic IDs above the largest target ID.
        let source_device_count = self.mesh().devices().len();
        let target_devices = mesh.devices();
        let first_free_id = target_devices.iter().map(Device::id).max().map_or(Some(0), |id| id.checked_add(1));
        let further = (0..source_device_count.saturating_sub(target_devices.len()))
            .map(|offset| {
                let id =
                    first_free_id.and_then(|id| id.checked_add(offset)).ok_or_else(|| Error::SizeLimitExceeded {
                        message: format!(
                            "planning a placement of a locally assigned array onto devices of process {} needs a \
                             device identifier above {}",
                            owner,
                            DeviceId::MAX,
                        ),
                    })?;
                Ok(Device::new(id, owner))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let devices = target_devices.iter().copied().chain(further).take(source_device_count).collect();
        let view_mesh = DeviceMesh::new(self.mesh().logical_mesh().clone(), devices)?;
        let view = (view_mesh != *self.mesh())
            .then(|| self.placement_planning_view().on_mesh(view_mesh, self.sharding().clone()))
            .transpose()?;
        let planned = match &view {
            Some((metadata, shards)) => {
                PlacementPlanningView { metadata, shards: shards.iter() }.strategy(layout_resolver, mesh, sharding)
            }
            None => self.placement_planning_view().strategy(layout_resolver, mesh, sharding),
        };

        let shared = |planned: Result<DevicePlacementStrategy, Error>| match planned {
            Ok(strategy)
                if keeps || !matches!(strategy.host_round_trip_cause(), Some(HostRoundTripCause::Device(_))) =>
            {
                Ok(Some(strategy))
            }
            Err(error) if keeps => Err(error),
            _ => Ok(None),
        };

        let Some(planned) = shared(planned)? else {
            return Ok(DevicePlacementStrategy::ProcessLocalValue);
        };

        if owner != process_index {
            return Ok(DevicePlacementStrategy::OnOwner { strategy: Box::new(planned) });
        }

        // A distinct view with matching logical meshes already has the target devices. When the placement keeps
        // contributions, the owner will move its value onto that view before executing this plan.
        if view.is_none() || (self.mesh().logical_mesh() == mesh.logical_mesh() && keeps) {
            return Ok(planned);
        }

        let planned = shared(self.placement_planning_view().strategy(layout_resolver, mesh, sharding))?;
        Ok(planned.unwrap_or(DevicePlacementStrategy::ProcessLocalValue))
    }

    /// Returns this array restricted to `mesh`, whose devices are devices of this array and whose logical mesh is that
    /// of this array with some axes of size one (refer to [`DeviceMeshExtension::at_coordinate_zero`]), along which
    /// this array must be replicated, so that the shards of those devices hold its whole value. The result has the
    /// sharding of this array over `mesh` and keeps the fences, the commitment, and the locally assigned devices of
    /// this array. It shares the buffers of those devices, and a device of this process without its own buffer receives
    /// a copy of another local replica of its contribution, or a new empty buffer for a shard without elements (refer
    /// to [`Self::buffer_for_local_contribution`]), whose readiness the execution fence of the result joins, and the
    /// restriction fails when no such replica exists. The execution fence of the result also joins the readiness of
    /// every buffer of this array whose shard has no elements (refer to
    /// [`PlacementCompletion::track_empty_shard_buffers`]).
    fn restricted_to(&self, mesh: &DeviceMesh) -> Result<Self, Error> {
        let sharding = sharding_on_mesh(self.sharding(), mesh.logical_mesh())?;
        let r#type = self.r#type().into_owned().with_sharding(None)?;
        let mut completion = PlacementCompletion::default();
        completion.track_empty_shard_buffers(self)?;
        let process_index = self.client().process_index()?;
        let devices = self.addressable_devices_by_id(process_index)?;
        let replicas = LocalReplicas::new(process_index);
        let mut work = self.session().reserve_work();

        // The shards of the devices of `mesh` are looked up by device, so a restriction to `n` devices costs `O(n)`.
        let shards = mesh.devices().iter().filter_map(|device| self.device_shard(device.id()));
        let mut buffers = Vec::new();
        for shard in shards.filter(|shard| shard.device().process_index() == process_index) {
            let index = shard.index();
            let buffer = self.buffer_for_local_contribution(&replicas, index, &devices, &mut work, &mut completion);
            buffers.push(buffer?);
        }

        self.rebuilt(r#type, mesh, sharding, buffers, completion)
    }

    /// Returns an array of the session of this array with `r#type`, `buffers`, and `sharding` over `mesh`, which keeps
    /// the send fence, the commitment, and the locally assigned devices of this array, and whose execution fence joins
    /// the execution fence of this array with `completion`.
    fn rebuilt(
        &self,
        r#type: ArrayType,
        mesh: &DeviceMesh,
        sharding: Sharding,
        buffers: Vec<Arc<Buffer<'static>>>,
        completion: PlacementCompletion,
    ) -> Result<Self, Error> {
        let session = self.session();
        let array = Self::from_addressable_buffers_internal(
            session,
            r#type,
            Some(sharding),
            mesh.clone(),
            None,
            buffers,
            false,
        )?;
        Ok(Self {
            execution_fence: completion.join(self.execution_fence.clone()),
            send_fence: self.send_fence.clone(),
            committed: self.committed,
            locally_assigned: self.locally_assigned,
            ..array
        })
    }
}

/// Buffer- and session-independent source of a placement plan.
pub(super) struct PlacementPlanningView<'a, Shards> {
    /// Runtime type, mesh, and sharding of the source.
    metadata: &'a ArrayMetadata,

    /// Global descriptors of the source, in mesh order.
    shards: Shards,
}

impl<'a, Shards: Iterator<Item = &'a ShardDescriptor> + Clone> PlacementPlanningView<'a, Shards> {
    /// Creates the planning view of a source with the provided `metadata` and `shards`, in mesh order.
    #[inline]
    pub(super) fn new(metadata: &'a ArrayMetadata, shards: Shards) -> Self {
        Self { metadata, shards }
    }

    /// Returns whether placing the source on `mesh` with `sharding` admits a strategy that could move bytes or run
    /// collectives between processes. `locally_assigned` states whether this process chose the devices of the source
    /// on its own, in which case every process places its own value with local work only.
    pub(super) fn communicates(&self, locally_assigned: bool, mesh: &DeviceMesh, sharding: &Sharding) -> bool {
        if locally_assigned
            || self.data_type().is_zero()
            || (self.mesh() == mesh && self.sharding() == sharding)
            || !devices_span_processes(self.mesh().devices().iter().chain(mesh.devices()))
        {
            return false;
        }

        // A remote move is possible whenever the planner can use a receive layout. The planner only carries that
        // layout into execution, so admission can assume one exists without asking a PJRT client. If execution
        // declines the assumed strategy, admission has already accounted for any communication its replacement might
        // require.
        self.strategy(&|_, _, shape| Ok(Some(dense_row_major_layout(shape.rank()).to_pjrt()?)), mesh, sharding)
            .map_or(true, |strategy| strategy.communicates())
    }

    /// Returns the element data type of the source.
    fn data_type(&self) -> DataType {
        self.metadata.r#type().data_type()
    }

    /// Returns the static shape of the source.
    fn shape(&self) -> StaticShape {
        self.metadata.r#type().static_shape().unwrap()
    }

    /// Returns the sharding of the source.
    fn sharding(&self) -> &Sharding {
        self.metadata.sharding()
    }

    /// Returns the device mesh of the source.
    fn mesh(&self) -> &DeviceMesh {
        self.metadata.mesh()
    }

    /// Derives a metadata-only view of the source on a synthetic mesh or a restriction of its existing mesh.
    fn on_mesh(&self, mesh: DeviceMesh, sharding: Sharding) -> Result<(ArrayMetadata, Vec<ShardDescriptor>), Error> {
        let r#type = self.metadata.r#type().clone().with_sharding(None)?;
        ArrayMetadata::new(r#type, Some(sharding), mesh, None)
    }

    /// Chooses the device work that places this array, whose devices every process knows, on `mesh` according to
    /// `sharding`, which is the first of the following strategies of [`Array::try_reshard_on_devices`] that applies, or
    /// returns the error that rules out all of them. Each strategy carries the transfers and programs of every process:
    ///
    ///   1. [`DevicePlacementStrategy::ExactShards`], when every target shard with elements holds what one source shard
    ///      holds and every shard that crosses processes has a default layout for its receive.
    ///   2. [`DevicePlacementStrategy::ShardRegions`], when the placement keeps the contributions of this array.
    ///   3. [`DevicePlacementStrategy::IntroducedSums`], when the target introduces a pending sum along a non-manual
    ///      axis.
    ///   4. [`DevicePlacementStrategy::ShardRegions`], when the meshes differ and this array declares no layout other
    ///      than the dense row-major one, or uneven shards or manual target axes rule out a compiled reshard (in any
    ///      layout), and [`Self::shard_region_plan`] accepts the placement.
    ///   5. [`DevicePlacementStrategy::HostRoundTrip`], when a program of the compiled reshard below would change the
    ///      value of this array or item 4 declines a placement that uneven shards or manual axes rule out, and only
    ///      when the placement keeps no contribution of this array, which is an error otherwise.
    ///   6. [`DevicePlacementStrategy::CompiledReshard`] on the mesh of this array, or
    ///      [`DevicePlacementStrategy::CompiledReshardAcrossMeshes`] to another mesh, whose replicas move as whole
    ///      shards of the replicated value.
    fn strategy(
        &self,
        layout_resolver: &CrossHostLayoutResolver<'_>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<DevicePlacementStrategy, Error> {
        // Exact shards need the default layout of every shard that crosses processes, between whichever processes it
        // crosses, so that a process with only local copies declines whenever the processes that transfer decline.
        // They come first, because a placement that only moves the contributions of a pending reduction (e.g., onto
        // another order of the same devices) keeps every contribution as it is, and introduces or completes nothing.
        let shape = self.shape();
        let element_type = self.data_type().executable_buffer_type();
        let exact = plan_exact_shard_moves(
            element_type,
            self.shards.clone(),
            self.mesh(),
            self.sharding(),
            &shape,
            mesh,
            sharding,
        )?;
        if let Some(plan) = exact {
            let cross_host_shards =
                plan.cross_host_moves().map(|movement| (movement.target_shard_index, movement.shape()));
            if let Some(cross_host_layouts) = cross_host_layouts(layout_resolver, element_type, cross_host_shards)? {
                return Ok(DevicePlacementStrategy::ExactShards { plan, cross_host_layouts });
            }
        }

        // A placement that keeps the contributions of a pending reduction or of manual variation within a tile only
        // moves bytes, so it moves each contribution as regions of its shards, and never completes the reduction.
        // Regions come first on the same mesh too, so that sparse and dense sources behave alike, and in any declared
        // layout. When regions decline on the same mesh with the same reduction state and variation by name, the
        // compiled reshard below re-splits each contribution without completing anything, so such a placement falls
        // back to it, but never to the host round trip, which would combine its contributions. Any other such placement
        // fails here (e.g., without a default layout for a cross-host receive).
        let keeps = keeps_contributions(self.sharding(), sharding);
        if keeps {
            if let Some((plan, cross_host_layouts)) = self.shard_region_plan(layout_resolver, mesh, sharding)? {
                return Ok(DevicePlacementStrategy::ShardRegions { plan, cross_host_layouts });
            }
            let keeps_state_by_name = self.mesh() == mesh
                && self.sharding().unreduced_axes() == sharding.unreduced_axes()
                && self.sharding().unreduced_kind() == sharding.unreduced_kind()
                && self.sharding().varying_manual_axes() == sharding.varying_manual_axes();
            if !keeps_state_by_name {
                return Err(ProgramError::UnsupportedOperation {
                    message: COMBINED_CONTRIBUTIONS_ERROR_MESSAGE.to_string(),
                }
                .into());
            }
        }

        // A pending sum that the target introduces is written as additive identities after a first step that places
        // the value without it on the devices at coordinate zero along the introduced axes only.
        let sum = ReductionState::Unreduced(UnreducedKind::Sum);
        let kept = kept_unreduced_axes(self.sharding(), sharding);
        let (mut axis_indices, mut introduced) = (Vec::new(), Vec::new());
        for (index, axis) in mesh.logical_mesh().axes().iter().enumerate() {
            if axis.r#type() != MeshAxisType::Manual
                && ReductionState::of(sharding, axis.name()) == sum
                && !kept.contains(&axis.name())
            {
                axis_indices.push(index);
                introduced.push(axis.name());
            }
        }
        if !axis_indices.is_empty() {
            // Empty shards and axes of size one need no identity elements. Check for an identity only when the target
            // layout requires one, then leave the sums along the kept axes pending in the first step.
            if requires_unreduced_sum_identity(mesh, &ShardLayout::new(&shape, mesh, sharding)?, &axis_indices) {
                let data_type = self.data_type();
                unreduced_sum_identity(data_type)?
                    .ok_or_else(|| Error::from(ArrayError::MissingUnreducedSumIdentity { data_type }))?;
            }

            let remaining = sharding.unreduced_axes().iter().filter(|axis| !introduced.contains(&axis.as_str()));
            let first_sharding = sharding.clone().with_unreduced_axes(remaining.cloned().collect::<Vec<_>>())?;
            let coordinate_zero = || -> Result<(DeviceMesh, Sharding), Error> {
                let first_mesh = mesh.at_coordinate_zero(&axis_indices)?;
                let first_sharding = sharding_on_mesh(&first_sharding, first_mesh.logical_mesh())?;
                Ok((first_mesh, first_sharding))
            };

            let replicated_along_introduced_axes =
                self.mesh() == mesh && introduced.iter().all(|axis| !self.sharding().distinguishes_shards_along(axis));
            let (first_mesh, first_sharding, restrict_source, first) = if replicated_along_introduced_axes {
                // The existing shards at coordinate zero already hold the whole value, including any contributions
                // the target keeps. Plan from that restriction only if its sharding still needs to change.
                let (first_mesh, first_sharding) = coordinate_zero()?;

                // Runtime arrays store unconstrained dimensions replicated, so treat those dimensions alike.
                let closed = |sharding: &Sharding| -> Result<Sharding, Error> {
                    let dimensions = sharding
                        .dimensions()
                        .iter()
                        .map(|dimension| match dimension {
                            ShardingDimension::Unconstrained => ShardingDimension::Replicated,
                            dimension => dimension.clone(),
                        })
                        .collect();
                    Ok(sharding.with_dimensions(dimensions)?)
                };

                let restricted_sharding = sharding_on_mesh(self.sharding(), first_mesh.logical_mesh())?;

                if closed(&restricted_sharding)? == closed(&first_sharding)? {
                    (first_mesh, first_sharding, false, None)
                } else {
                    let (metadata, shards) = self.on_mesh(first_mesh.clone(), restricted_sharding)?;
                    let view = PlacementPlanningView { metadata: &metadata, shards: shards.iter() };
                    let first = view.strategy(layout_resolver, &first_mesh, &first_sharding)?;
                    (first_mesh, first_sharding, true, Some(first))
                }
            } else {
                // When reduction state must change, plan on the full target mesh so the first step can use the usual
                // collective path. Devices away from coordinate zero then discard what they received.
                let on_target_mesh = !self.sharding().holds_one_value_per_slice()
                    && !keeps_contributions(self.sharding(), &first_sharding);
                let direct =
                    on_target_mesh.then(|| self.strategy(layout_resolver, mesh, &first_sharding)).transpose()?;
                match direct {
                    Some(first) if !matches!(first, DevicePlacementStrategy::HostRoundTrip { .. }) => {
                        (mesh.clone(), first_sharding, false, Some(first))
                    }
                    _ => {
                        // A pure move can target coordinate zero directly. A host round trip also goes there,
                        // so its owner only needs the source contributions and the devices that receive the value.
                        let (first_mesh, first_sharding) = coordinate_zero()?;
                        let first = self.strategy(layout_resolver, &first_mesh, &first_sharding)?;
                        (first_mesh, first_sharding, false, Some(first))
                    }
                }
            };

            return Ok(DevicePlacementStrategy::IntroducedSums {
                mesh: first_mesh,
                sharding: first_sharding,
                restrict_source,
                first: first.map(Box::new),
                axis_indices,
            });
        }

        // Validate the device programs from shared metadata before any process posts transfers or compiles.
        // The immediately invoked closure keeps `?` and `return Err(...)` inside validation, so the strategy
        // can inspect `compiled` and choose shard regions or a host round trip for a partitioner limitation.
        let compiled = (|| -> Result<Option<HostRoundTripCause>, Error> {
            let data_type = self.data_type();
            let cause = if self.mesh() == mesh {
                validate_reduction_transition(data_type, mesh, self.sharding(), sharding)
            } else {
                // Check both stages across meshes. The gather carries contributions for any reduction the target
                // keeps, and the second program reshards those replicas on the target mesh.
                let input = self.sharding();
                let gathered = gathered_sharding(input, sharding, self.mesh().logical_mesh(), shape.rank())?;
                let gather = validate_reduction_transition(data_type, self.mesh(), input, &gathered);
                let gathered = gathered_sharding(input, sharding, mesh.logical_mesh(), shape.rank())?;
                let reshard = validate_reduction_transition(data_type, mesh, &gathered, sharding);
                gather.or(reshard)
            };

            // The SPMD partitioner handles auto and explicit axes. A manual axis belongs to user-managed
            // computations such as `shard_map`, so check for it in both the reshard and any source gather.
            let gathers = self.mesh() != mesh && !self.sharding().is_replicated();
            let program_meshes = [mesh].into_iter().chain(gathers.then(|| self.mesh()));
            for program_mesh in program_meshes {
                if let Some(axis) =
                    program_mesh.logical_mesh().axes().iter().find(|axis| axis.r#type() == MeshAxisType::Manual)
                {
                    return Err(ArrayError::ManualReshardingAxis { axis_name: axis.name().to_string() }.into());
                }
            }

            // XLA pads uneven splits to equal buffer shapes. Shard regions can materialize exact shapes, so this
            // error lets planning fall back to regions when a compiled reshard would pad them.
            let target_layout = ShardLayout::new(&shape, mesh, sharding)?;
            let has_uneven_shards = |shapes: Vec<StaticShape>| shapes.iter().any(|shape| shape != &shapes[0]);
            if has_uneven_shards(self.shards.clone().map(ShardDescriptor::shape).collect()) {
                return Err(ArrayError::UnevenReshardingShards {
                    shape: shape.clone(),
                    sharding: self.sharding().clone(),
                }
                .into());
            }

            if has_uneven_shards(target_layout.descriptors().iter().map(ShardDescriptor::shape).collect()) {
                return Err(
                    ArrayError::UnevenReshardingShards { shape: shape.clone(), sharding: sharding.clone() }.into()
                );
            }

            Ok(cause)
        })();

        // A compiled reshard across meshes moves whole shards in a declared layout other than the dense row-major one,
        // which regions convert on every source device, so regions replace it only where it declines.
        let dense =
            self.metadata.r#type().layout().is_none_or(|layout| *layout == dense_row_major_layout(shape.rank()));
        let try_regions = match &compiled {
            Ok(None) => self.mesh() != mesh && dense,
            Ok(Some(_)) => false,
            Err(Error::Array(error)) => {
                matches!(
                    error.as_ref(),
                    ArrayError::UnevenReshardingShards { .. } | ArrayError::ManualReshardingAxis { .. }
                )
            }
            Err(_) => false,
        };

        // A placement that keeps contributions has asked `Self::shard_region_plan` above with the same arguments.
        if try_regions
            && !keeps
            && let Some((plan, cross_host_layouts)) = self.shard_region_plan(layout_resolver, mesh, sharding)?
        {
            return Ok(DevicePlacementStrategy::ShardRegions { plan, cross_host_layouts });
        }

        // A decline of the compiled reshard that every process predicts from shared metadata (i.e., a manual axis or
        // uneven shards where shard regions do not apply) takes the host round trip of the owner that every process
        // names alike, rather than leaving the recovery to a fallback that only the owner would take.
        let cause = match compiled {
            Ok(cause) => cause,
            Err(error) if try_regions => Some(HostRoundTripCause::Device(error)),
            Err(error) => return Err(error),
        };

        // That round trip reads back the combined value, so a placement that keeps some contributions of this array
        // (refer to `keeps_some_contributions`) fails here, alike on every process, rather than replacing them.
        if let Some(cause) = cause {
            if keeps {
                return Err(ProgramError::UnsupportedOperation {
                    message: COMBINED_CONTRIBUTIONS_ERROR_MESSAGE.to_string(),
                }
                .into());
            }
            if keeps_some_contributions(self.sharding(), sharding) {
                let kept = kept_unreduced_axes(self.sharding(), sharding)
                    .iter()
                    .map(|axis| format!("`{axis}`"))
                    .collect::<Vec<_>>();
                let (reason, remedy) = match cause {
                    HostRoundTripCause::Reduction { transition, .. } => (transition, "complete the reduction first"),
                    HostRoundTripCause::Device(error) => {
                        (error.to_string(), "complete the reduction first, or place the array onto an even split")
                    }
                };
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "cannot place an array whose pending reduction the placement keeps along the mesh axes {} and \
                         changes along others when no device strategy applies ({}), because the host round trip that \
                         remains would combine the contributions that the placement keeps; {}",
                        kept.join(", "),
                        reason,
                        remedy,
                    ),
                }
                .into());
            }

            // A host round trip runs on one process. It must own every target device and a replica of every non-empty
            // source contribution, and the source must have a well-defined host value. Shared metadata makes every
            // process select the same owner, or agree that no owner can do it.
            let owner = mesh.devices()[0].process_index();
            let process_index = (mesh.devices().iter().all(|device| device.process_index() == owner)
                && self.contribution_missing_from(owner).is_none()
                && validate_one_value_per_contribution(self.mesh(), self.sharding(), self.shards.clone()).is_ok())
            .then_some(owner);

            return Ok(DevicePlacementStrategy::HostRoundTrip { cause, process_index });
        }

        if self.mesh() == mesh {
            let (input, output) = (self.sharding().clone(), sharding.clone());
            let program = ReshardProgram { mesh: mesh.clone(), input, output };
            return Ok(DevicePlacementStrategy::CompiledReshard { program });
        }

        // A replicated copy of this array (i.e., this array itself or its all-gather on its own mesh) moves to every
        // target device as whole shards. Every process that owns a device of the mesh of this array holds such a
        // replica, so a replica crosses processes exactly when a target device belongs to a process that owns none, and
        // all replicas have the shape of this array. The gather completes only the pending reductions that the target
        // does not keep (refer to `gathered_sharding`), so a kept one moves as its contributions, which correspond by
        // their ordinals at both ends (refer to `moves_contributions`), and the final reshard keeps it.
        let rank = shape.rank();
        let replicated = gathered_sharding(self.sharding(), sharding, self.mesh().logical_mesh(), rank)?;
        let gather = (!self.sharding().is_replicated() && self.sharding() != &replicated).then(|| ReshardProgram {
            mesh: self.mesh().clone(),
            input: self.sharding().clone(),
            output: replicated.clone(),
        });
        let replicated_on_target = gathered_sharding(self.sharding(), sharding, mesh.logical_mesh(), rank)?;
        let replica_layout = ShardLayout::new(&shape, self.mesh(), &replicated)?;

        // Both shardings replicate the whole value and keep the same contributions, so every target shard has the
        // slices of every source shard of the same contribution.
        let replicas = plan_exact_shard_moves(
            element_type,
            replica_layout.descriptors(),
            self.mesh(),
            &replicated,
            &shape,
            mesh,
            &replicated_on_target,
        )?
        .unwrap();
        let mut cross_host_layouts = HashMap::new();
        if let Some(movement) = replicas.cross_host_moves().next() {
            let shape = movement.shape();
            let layout = layout_resolver(element_type, movement.target_shard_index, &shape)?.ok_or_else(|| {
                Error::Pjrt(PjrtError::unimplemented(
                    "the PJRT plugin does not report the default layout that a cross-host receive allocates",
                ))
            })?;
            cross_host_layouts.insert(shape, layout);
        }

        let reshard = ReshardProgram { mesh: mesh.clone(), input: replicated_on_target, output: sharding.clone() };
        Ok(DevicePlacementStrategy::CompiledReshardAcrossMeshes { gather, replicas, cross_host_layouts, reshard })
    }

    /// Returns the first shard with elements of this array whose contribution has no replica on a device of process
    /// `process_index`, or [`None`] when that process holds a replica of every such contribution, which is what a
    /// readback of this array on that process reads.
    fn contribution_missing_from(&self, process_index: ProcessIndex) -> Option<&'a ShardDescriptor> {
        let owned = self
            .shards
            .clone()
            .filter(|shard| !shard.slice().iter().any(Range::is_empty))
            .filter(|shard| shard.device().process_index() == process_index)
            .map(|shard| ShardContribution::new(self.mesh(), self.sharding(), shard))
            .collect::<HashSet<_>>();
        self.shards
            .clone()
            .filter(|shard| !shard.slice().iter().any(Range::is_empty))
            .find(|shard| !owned.contains(&ShardContribution::new(self.mesh(), self.sharding(), shard)))
    }

    /// Plans the shard regions that place the source on `mesh` with `sharding` and resolves their default receive
    /// layouts. Returns `None` if the target cannot be covered by corresponding source contributions or the plugin
    /// cannot provide a receive layout. Each process plans every move from the same global metadata.
    fn shard_region_plan(
        &self,
        layout_resolver: &CrossHostLayoutResolver<'_>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<Option<(ShardMovePlan, HashMap<StaticShape, ryft_pjrt::Layout>)>, Error> {
        let Some(axes) = corresponding_shards(self.sharding(), sharding) else {
            return Ok(None);
        };

        let source_axes = axes.as_ref().map(|axes| axes.source.as_slice());
        let target_axes = axes.as_ref().map(|axes| axes.target.as_slice());

        let elements = |slice: &[Range<usize>]| -> Result<usize, Error> {
            Ok(StaticShape::new(slice.iter().map(Range::len).collect::<Vec<_>>()).element_count()?)
        };

        let total = |counts: &mut dyn Iterator<Item = Result<usize, Error>>| -> Result<usize, Error> {
            let mut total = 0usize;
            for count in counts {
                total = total.checked_add(count?).ok_or_else(|| Error::SizeLimitExceeded {
                    message: format!("the number of elements of a shard region plan exceeds {}", usize::MAX),
                })?;
            }
            Ok(total)
        };

        // Group replicas by slice and contribution. For a pending reduction or manual variation within a tile,
        // each target takes regions only from the corresponding contribution.
        let replicas = SourceReplicas::by_contribution(self.mesh(), self.sharding(), self.shards.clone(), source_axes);

        // Index the distinct ranges per dimension so each target can look up its intersections by binary search.
        let mut source_ranges = vec![Vec::new(); self.shape().rank()];
        for (slice, _) in replicas.keys() {
            for (ranges, range) in source_ranges.iter_mut().zip(slice.iter()) {
                ranges.push(range.clone());
            }
        }

        for ranges in &mut source_ranges {
            ranges.sort_unstable_by_key(|range| (range.start, range.end));
            ranges.dedup();
        }

        let layout = ShardLayout::new(&self.shape(), mesh, sharding)?;
        let targets = layout.descriptors();
        let mut moves = Vec::new();
        let mut received = HashSet::new();
        for target in targets {
            let (target_slice, member) = source_replica_key(mesh, sharding, target, target_axes);

            // Combine intersecting ranges across dimensions to find candidate source slices.
            let mut candidates = vec![Vec::new()];
            for (ranges, target_range) in source_ranges.iter().zip(target_slice) {
                // The ranges that intersect `target_range` end after it starts and start before it ends.
                let first = ranges.partition_point(|range| range.end <= target_range.start);
                let last = ranges.partition_point(|range| range.start < target_range.end).max(first);
                let intersecting = ranges[first..last]
                    .iter()
                    .filter(|range| range.start.max(target_range.start) < range.end.min(target_range.end));
                let intersecting = intersecting.collect::<Vec<_>>();
                candidates = candidates
                    .into_iter()
                    .flat_map(|prefix: Vec<Range<usize>>| {
                        intersecting.iter().map(move |&range| [prefix.as_slice(), std::slice::from_ref(range)].concat())
                    })
                    .collect();
            }

            let mut regions = Vec::new();
            for candidate in candidates {
                let key = (candidate.as_slice(), member.clone());
                let Some(((source_slice, _), source_shards)) = replicas.get_key_value(&key) else {
                    continue;
                };

                let slice = source_slice
                    .iter()
                    .zip(target_slice)
                    .map(|(source, target)| source.start.max(target.start)..source.end.min(target.end))
                    .collect::<Vec<_>>();

                regions.push((source_shards.preferred(target), slice));
            }

            // The regions of one member are disjoint, so they cover the target shard exactly when they hold as many
            // elements.
            let target_elements = elements(target_slice)?;
            if total(&mut regions.iter().map(|(_, slice)| elements(slice)))? != target_elements {
                return Ok(None);
            }

            if target_elements > 0 {
                received.insert((target_slice, member));
            }

            // Emit each target's moves in order of their starts, which the executor uses to assemble the shard.
            regions.sort_unstable_by(|(_, left), (_, right)| {
                left.iter().map(|range| range.start).cmp(right.iter().map(|range| range.start))
            });

            for (source, slice) in regions {
                let transfer_key = exact_shard_transfer_key(source.index(), target.index(), targets.len())?;
                moves.push(ShardMove {
                    source_shard_index: source.index(),
                    source_device: source.device(),
                    target_shard_index: target.index(),
                    target_device: target.device(),
                    slice,
                    transfer_key,
                });
            }
        }

        // Every element of a target member comes from the same element of the corresponding source member, so the
        // target keeps every source contribution exactly when it receives as many distinct elements.
        let source_elements = total(&mut replicas.keys().map(|(slice, _)| elements(slice)))?;
        let received_elements = total(&mut received.iter().map(|(slice, _)| elements(slice)))?;
        validate_kept_contributions(self.sharding(), source_elements == received_elements)?;

        // Check all remote device IDs before resolving layouts so a missing layout cannot hide an invalid ID.
        let plan = ShardMovePlan::new(self.data_type().executable_buffer_type(), moves)?;
        let remote_regions = plan.cross_host_moves().map(|movement| (movement.target_shard_index, movement.shape()));
        let layouts = cross_host_layouts(layout_resolver, plan.element_type, remote_regions)?;
        Ok(layouts.map(|layouts| (plan, layouts)))
    }
}

/// Resolves the default layout that a cross-host receive allocates for a buffer of the given element type and shape.
/// The target shard index identifies the shard in diagnostics. Returns `None` when the plugin cannot report the layout
/// and [`PjrtError::Unimplemented`] when it lacks the cross-host transfers extension. All processes of a placement use
/// the same plugin, so they resolve the same layouts.
type CrossHostLayoutResolver<'o> =
    dyn Fn(BufferType, ShardIndex, &StaticShape) -> Result<Option<ryft_pjrt::Layout>, Error> + 'o;

/// Device work that places an [`Array`] on a mesh with a sharding, which [`Array::device_placement_strategy`] chooses
/// from information that every process shares, so that the processes of a placement agree on it.
#[allow(
    clippy::large_enum_variant,
    reason = "placement strategies are short-lived and keeping the gather plan inline avoids an extra allocation"
)]
enum DevicePlacementStrategy {
    /// Changes only the placement metadata of an array of [`DataType::Zero`], which has no buffers.
    Bufferless,

    /// Places the full value that an array with locally assigned devices holds on the devices of this process alone
    /// Refer to [`Array::is_locally_assigned`] and [`Array::place_process_local_value`] for more information.
    ProcessLocalValue,

    /// Takes no part in the placement of an array with locally assigned devices onto devices that one other process
    /// owns, which places its own value there with `strategy`, and returns the metadata of the result after the checks
    /// that `strategy` makes from metadata before it submits anything.
    OnOwner {
        /// Strategy that the owner runs, as every process plans it on the view of the value on the devices of the
        /// owner. Refer to [`Array::device_placement_strategy`] for more information.
        strategy: Box<DevicePlacementStrategy>,
    },

    /// Shares, copies, and transfers the whole shards that `plan` describes. Refer to [`Array::place_exact_shards`]
    /// for more information.
    ExactShards {
        /// Plan of the moves of whole shards of every process.
        plan: ShardMovePlan,

        /// Default layout of the PJRT plugin for the shape of every shard that crosses processes.
        cross_host_layouts: HashMap<StaticShape, ryft_pjrt::Layout>,
    },

    /// Slices the source shards and joins the regions on the target devices. Refer to [`Array::place_shard_regions`]
    /// for more information.
    ShardRegions {
        /// Plan of the moves of the regions of every target shard of every process.
        plan: ShardMovePlan,

        /// Default layout of the PJRT plugin for the shape of every region that crosses processes.
        cross_host_layouts: HashMap<StaticShape, ryft_pjrt::Layout>,
    },

    /// Reshards an array on its own mesh with a compiled program (refer to [`ReshardProgram::compile`]).
    CompiledReshard {
        /// Program that reshards the array on its mesh.
        program: ReshardProgram,
    },

    /// Moves a replicated copy of an array to another mesh, gathering a sharded array on its own mesh first, and then
    /// reshards it there with a compiled program. Refer to [`Array::place_with_strategy`] for more information.
    CompiledReshardAcrossMeshes {
        /// All-gather of a sharded array on its own mesh, or `None` for a replicated array.
        gather: Option<ReshardProgram>,

        /// Plan of the moves of the replicas of the replicated array to every device of the target mesh.
        replicas: ShardMovePlan,

        /// Default layout of the PJRT plugin for the shape of the array when a replica crosses processes.
        cross_host_layouts: HashMap<StaticShape, ryft_pjrt::Layout>,

        /// Program that reshards the replicated copy on the target mesh.
        reshard: ReshardProgram,
    },

    /// Places an array on the devices at coordinate zero along the introduced mesh axes (at `axis_indices` within the
    /// target mesh) according to the target sharding without the pending sums that the target introduces, and then
    /// gives the other devices of the target the additive identity of the data type. Refer to
    /// [`PlacementPlanningView::strategy`] and [`Array::place_with_strategy`] for more information.
    IntroducedSums {
        /// Mesh of the first step, which is the mesh of the devices at coordinate zero along the introduced axes (refer
        /// to [`DeviceMeshExtension::at_coordinate_zero`]), or the target mesh when the array cannot be placed
        /// there directly.
        mesh: DeviceMesh,

        /// Target sharding without the pending sums that the target introduces, over `mesh`.
        sharding: Sharding,

        /// Whether `first` starts from a restriction of the array to `mesh`. Refer to [`Array::restricted_to`] for more
        /// information.
        restrict_source: bool,

        /// Strategy of the first step, or [`None`] when the restriction of the array to `mesh` already has `sharding`.
        /// A host round trip in this step targets only the devices at coordinate zero.
        first: Option<Box<DevicePlacementStrategy>>,

        /// Indices within the target mesh of the axes along which the target introduces a pending sum.
        axis_indices: Vec<usize>,
    },

    /// Reads an array back and uploads it again with the host functions that define what its reduction state means, for
    /// a reason that every process predicts (i.e., [`HostRoundTripCause`]). Process `process_index` does the round trip
    /// (refer to the host fallback of [`Array::to_placement`]), and every other process returns the placement without
    /// buffers. Without such a process, every process returns the error of `cause` before starting the placement
    /// (i.e., [`HostRoundTripCause::across_processes`]).
    HostRoundTrip {
        /// Why no device strategy places the array.
        cause: HostRoundTripCause,

        /// Index of the process that owns every target device and can read the full source value, if any.
        process_index: Option<usize>,
    },
}

impl DevicePlacementStrategy {
    /// Returns whether the selected moves or compiled programs involve communication between processes.
    fn communicates(&self) -> bool {
        let moves_between_processes = |plan: &ShardMovePlan| plan.cross_host_moves().next().is_some();
        let runs_on_several_processes = |program: &ReshardProgram| devices_span_processes(program.mesh.devices());
        match self {
            Self::ExactShards { plan, .. } | Self::ShardRegions { plan, .. } => moves_between_processes(plan),
            Self::CompiledReshard { program } => runs_on_several_processes(program),
            Self::CompiledReshardAcrossMeshes { gather, replicas, reshard, .. } => {
                gather.as_ref().is_some_and(runs_on_several_processes)
                    || moves_between_processes(replicas)
                    || runs_on_several_processes(reshard)
            }
            Self::IntroducedSums { first, .. } => first.as_deref().is_some_and(Self::communicates),
            Self::Bufferless | Self::ProcessLocalValue | Self::OnOwner { .. } | Self::HostRoundTrip { .. } => false,
        }
    }

    /// Returns why this strategy takes a host round trip, in itself, in the first step of an introduced sum, or on the
    /// owner of the placement, if any.
    fn host_round_trip_cause(&self) -> Option<&HostRoundTripCause> {
        match self {
            Self::HostRoundTrip { cause, .. } => Some(cause),
            Self::IntroducedSums { first: Some(first), .. } | Self::OnOwner { strategy: first } => {
                first.host_round_trip_cause()
            }
            _ => None,
        }
    }
}

/// Deterministic plan of the moves of whole shards or of shard regions that place an [`Array`] (e.g., by
/// [`Array::to_placement`]), which holds the moves of every process, so that every process derives the same plan
/// from global metadata. Each process performs the part of every move that [`ShardMove::role`] assigns to it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ShardMovePlan {
    /// Element type of the buffers that the plan moves, which is the executable buffer type of the array (e.g., a
    /// predicate for one-bit integers), because that is the buffer that a send transfers and a receive allocates.
    element_type: BufferType,

    /// Moves of the plan, in target shard order, and for each target shard in the order of the starting coordinates
    /// of the moved slices.
    moves: Vec<ShardMove>,
}

impl ShardMovePlan {
    /// Creates a plan of the moves of whole shards or of shard regions.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SizeLimitExceeded`] when a move that crosses processes names a device whose identifier PJRT
    /// cross-host transfers cannot represent. The plan covers the moves of every process, so every process returns this
    /// error before any of them submits a transfer.
    fn new(element_type: BufferType, moves: Vec<ShardMove>) -> Result<Self, Error> {
        let plan = Self { element_type, moves };
        for movement in plan.cross_host_moves() {
            cross_host_global_device_id(movement.source_device.id())?;
            cross_host_global_device_id(movement.target_device.id())?;
        }
        Ok(plan)
    }

    /// Returns the moves in which the process with index `process_index` takes the part `role`.
    fn moves_with_role(&self, process_index: usize, role: MoveRole) -> impl Iterator<Item = &ShardMove> {
        self.moves.iter().filter(move |movement| movement.role(process_index) == Some(role))
    }

    /// Returns the moves that cross processes, whatever processes they cross between.
    fn cross_host_moves(&self) -> impl Iterator<Item = &ShardMove> {
        self.moves
            .iter()
            .filter(|movement| movement.source_device.process_index() != movement.target_device.process_index())
    }
}

/// One move in a [`ShardMovePlan`] of the elements of a slice, which is a whole source shard or the intersection
/// of a source shard with a target shard, from the device of that source shard to the device of the target shard.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ShardMove {
    /// Index of the source shard in the source array.
    source_shard_index: ShardIndex,

    /// Device of the source shard.
    source_device: Device,

    /// Index of the target shard in the target array.
    target_shard_index: ShardIndex,

    /// Device of the target shard.
    target_device: Device,

    /// Moved slice in global array coordinates, which is the slice of the target shard for a move of a whole shard.
    slice: Vec<Range<usize>>,

    /// Deterministic transfer rendezvous key that the send and the receive of a cross-host move share.
    transfer_key: CrossHostTransferKey,
}

impl ShardMove {
    /// Returns the static shape of the moved slice.
    fn shape(&self) -> StaticShape {
        StaticShape::new(self.slice.iter().map(Range::len).collect::<Vec<_>>())
    }

    /// Returns the part that the process with index `process_index` takes in this move, or `None` when it owns neither
    /// of its devices.
    fn role(&self, process_index: usize) -> Option<MoveRole> {
        match (self.source_device.process_index() == process_index, self.target_device.process_index() == process_index)
        {
            (true, true) => Some(MoveRole::Copy),
            (true, false) => Some(MoveRole::Send),
            (false, true) => Some(MoveRole::Receive),
            (false, false) => None,
        }
    }
}

/// Program that a placement compiles and runs on the devices of one mesh to reshard an array there, which is the
/// identity with sharding annotations on its input and its output. The strategy of a placement fixes its programs
/// from information that every process shares, so every process that runs one of them runs the same program.
#[derive(Clone, Debug, PartialEq)]
struct ReshardProgram {
    /// Mesh whose devices run the program, which is the mesh of its input and of its output.
    mesh: DeviceMesh,

    /// Sharding of the input of the program.
    input: Sharding,

    /// Sharding of the output of the program.
    output: Sharding,
}

impl ReshardProgram {
    /// Returns whether the process with index `process_index` runs this program, which is the case exactly when it owns
    /// a device of its mesh. Any other process holds no buffer that the program reads or writes and takes part in none
    /// of its collectives, and its PJRT client cannot compile a program for devices that it does not address, so it
    /// compiles and runs nothing and only derives the metadata of the output.
    fn runs_on(&self, process_index: usize) -> bool {
        self.mesh.devices().iter().any(|device| device.process_index() == process_index)
    }

    /// Returns the mesh over which this program is compiled, which is [`Self::mesh`] with the explicit axes along which
    /// the program changes the [`ReductionState`] typed as auto, and the auto axes along which it keeps a pending
    /// reduction typed as explicit. The program only moves data, so the types of the axes of its shardings change
    /// nothing that it computes, but over an explicit axis a type records reduction state, which no staged operation
    /// may change (i.e., a [`reshard`](ryft_core::ReshardOperation) keeps it). Over an auto axis the type omits it,
    /// so the lowering makes the change instead (i.e., it completes a pending reduction where the input enters the
    /// program, and the output sharding introduces a pending sum or a completed reduction), with the arithmetic of XLA,
    /// which [`validate_reduction_transition`] compares with the meaning of the reduction before a program is planned.
    /// A pending reduction that the program keeps therefore moves over an explicit axis, as its contributions, without
    /// being completed and introduced again, which preserves the value of every data type and needs no collective.
    /// Every other axis keeps its type.
    fn compilation_mesh(&self) -> Result<DeviceMesh, Error> {
        let logical_mesh = self.mesh.logical_mesh();
        let (mut changed_axes, mut kept_axes) = (Vec::new(), Vec::new());
        for axis in logical_mesh.axes() {
            let (from, to) =
                (ReductionState::of(&self.input, axis.name()), ReductionState::of(&self.output, axis.name()));
            match axis.r#type() {
                MeshAxisType::Explicit if from != to => changed_axes.push(axis.name()),
                MeshAxisType::Auto if from == to && matches!(from, ReductionState::Unreduced(_)) => {
                    kept_axes.push(axis.name())
                }
                _ => {}
            }
        }

        if changed_axes.is_empty() && kept_axes.is_empty() {
            return Ok(self.mesh.clone());
        }

        let logical_mesh = logical_mesh
            .with_axis_types(changed_axes, MeshAxisType::Auto)?
            .with_axis_types(kept_axes, MeshAxisType::Explicit)?;

        Ok(DeviceMesh::new(logical_mesh, self.mesh.devices().to_vec())?)
    }

    /// Compiles the reshard as an identity program with sharding annotations on its input and output. The input
    /// `array` must already be on this program's mesh with the declared input sharding. Returns [`None`] on a process
    /// that owns none of the mesh's devices. Compilation uses [`Self::compilation_mesh`], which types the explicit mesh
    /// axes along which the program changes the reduction state as auto. [`Self::execute`] moves `array` onto that mesh
    /// and moves the result back. The input and output are declared in the memory of `array`, so the program accepts
    /// `array` and returns its result in the same memory. If `donate` is `true`, the input of the program is marked as
    /// donated so that PJRT can reuse the memory of its buffers for the output buffers. The compilation cache is keyed
    /// by the complete lowered computation, including its input and output sharding annotations and PJRT options.
    fn compile(&self, array: &Array, donate: bool) -> Result<Option<CompiledFunction<ArrayType, ArrayType>>, Error> {
        let session = array.session();
        if !self.runs_on(session.client().process_index()?) {
            return Ok(None);
        }

        // Build the bare input type, which has no sharding, for tracing. The input and output shardings are attached
        // as `sdy.sharding` attributes through `Options::with_input_shardings` and `Options::with_output_shardings`.
        // The type keeps the memory of the array, which the identity carries over to the output.
        let input_type = ArrayType::new(array.data_type(), array.shape().into()).with_memory(array.r#type().memory());
        let mesh = self.compilation_mesh()?;
        let (input, output) = if mesh == self.mesh {
            (self.input.clone(), self.output.clone())
        } else {
            let logical_mesh = mesh.logical_mesh();
            (self.input.with_mesh(logical_mesh.clone())?, self.output.with_mesh(logical_mesh.clone())?)
        };

        let options = Options::new(mesh)
            .with_input_shardings(vec![input])
            .with_output_shardings(vec![output])
            .with_donate(donate);
        let staged = stage(|input| input, input_type, options)?;

        Ok(Some(session.compile_staged_function(staged)?))
    }

    /// Executes `compiled`, which [`Self::compile`] compiled from this program for `array`, on `array`. Without
    /// `compiled` (i.e., in a process that does not run this program), this function runs nothing and returns the
    /// metadata of the output of this program. The execution fence of the result joins the execution fence of `array`
    /// with the fence of the launch. This function takes `array` by value so that a donation takes effect. The
    /// executable runs on [`Self::compilation_mesh`], so when that mesh types an axis differently, `array` enters it
    /// as the same shards on that mesh, and its result leaves it as the same shards on the mesh of this program. Both
    /// conversions preserve pending reductions, storage, and completion fences.
    fn execute(&self, array: Array, compiled: Option<&CompiledFunction<ArrayType, ArrayType>>) -> Result<Array, Error> {
        let Some(compiled) = compiled else {
            let r#type = ArrayType::new(array.data_type(), array.shape().into()).with_memory(array.r#type().memory());
            return array.place_without_local_buffers(array.session(), r#type, &self.mesh, &self.output);
        };
        let mut completion = PlacementCompletion::default();
        completion.track_fence(array.execution_fence());
        let session = array.session().clone();
        let mesh = self.compilation_mesh()?;
        let mut output = if mesh == self.mesh {
            compiled.executable().execute(&session, array)?
        } else {
            // Consume each handle as it is retyped so the input's buffers can still be donated to the executable.
            let retype_mesh = |array: Array, mesh: &DeviceMesh| -> Result<Array, Error> {
                debug_assert_eq!(array.mesh().devices(), mesh.devices());
                let sharding = array.sharding().with_mesh(mesh.logical_mesh().clone())?;
                let r#type = array.r#type().into_owned().with_sharding(None)?;
                let buffers = array.shards().iter().filter_map(|shard| shard.buffer().cloned()).collect();
                array.rebuilt(r#type, mesh, sharding, buffers, PlacementCompletion::default())
            };
            let input = retype_mesh(array, &mesh)?;
            retype_mesh(compiled.executable().execute(&session, input)?, &self.mesh)?
        };
        output.execution_fence = completion.join(output.execution_fence.take());
        Ok(output)
    }
}

/// Reduction state of a value along one mesh axis, which a [`Sharding`] records per axis (with one [`UnreducedKind`]
/// for all of its unreduced axes), and which a [`ReshardProgram`] keeps or changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReductionState {
    /// The devices along the axis hold contributions that combine with a pending reduction of this kind.
    Unreduced(UnreducedKind),

    /// The devices along the axis hold the result of a completed reduction.
    Reduced,

    /// The axis carries no reduction state.
    Free,
}

impl ReductionState {
    /// Returns the reduction state of a value placed as `sharding` along the mesh axis named `axis`.
    fn of(sharding: &Sharding, axis: &str) -> Self {
        if sharding.unreduced_axes().contains(axis) {
            Self::Unreduced(sharding.unreduced_kind())
        } else if sharding.reduced_axes().contains(axis) {
            Self::Reduced
        } else {
            Self::Free
        }
    }
}

impl Display for ReductionState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreduced(kind) => write!(formatter, "a pending `{kind}` reduction"),
            Self::Reduced => write!(formatter, "a completed reduction"),
            Self::Free => write!(formatter, "no reduction state"),
        }
    }
}

/// Why a placement takes [`DevicePlacementStrategy::HostRoundTrip`], which every process tells from the metadata that
/// they share, so that every process takes the same decision (i.e., the owner does the round trip, every other process
/// returns the metadata of the placement, and without an owner every process returns the same error).
#[derive(Clone, Debug)]
enum HostRoundTripCause {
    /// Change of reduction state that no program of a placement makes exactly, which [`validate_reduction_transition`]
    /// found, and which only a host round trip makes exactly. What the user can do depends on whether the work that
    /// meets it can take that round trip at all, so the error is rendered where that is known.
    Reduction {
        /// Change, data type, mesh axis, and reason (e.g., "cannot take `u1` values from … on the devices, because …").
        transition: String,

        /// What the user can do where no single process owns the source and the target.
        remedy: &'static str,
    },

    /// Error of the compiled reshard where shard regions do not apply either (i.e.,
    /// [`ArrayError::ManualReshardingAxis`] or [`ArrayError::UnevenReshardingShards`]),
    /// which a process without the round trip returns as it is.
    Device(Error),
}

impl HostRoundTripCause {
    /// Returns the error of a placement (e.g., [`Array::to_placement`]) that no process can make with a host round
    /// trip, which needs one process that owns every device of the target and can read the source back from its
    /// own devices.
    fn across_processes(&self) -> Error {
        match self {
            Self::Reduction { transition, remedy } => Error::from(ProgramError::UnsupportedOperation {
                message: format!(
                    "{}, and the host round trip that makes it exactly needs one process that owns every device of the \
                     target and can read the source back from its own devices; {}",
                    transition, remedy,
                ),
            }),
            Self::Device(error) => error.clone(),
        }
    }

    /// Returns the error of work that has no host round trip (i.e., the preparation of the arguments of a program or
    /// an eager operation, refer to [`Array::reshard_on_devices`]).
    fn without_host_route(&self) -> Error {
        match self {
            Self::Reduction { transition, .. } => Error::from(ProgramError::UnsupportedOperation {
                message: format!(
                    "{}, and the preparation of the arguments of a program has no host round trip; place the argument \
                     with `Array::to_placement` first, which makes the change with a host round trip where one process \
                     owns every device of the target and a device of every source contribution with elements",
                    transition,
                ),
            }),
            Self::Device(error) => error.clone(),
        }
    }
}

/// Completion of the device work that one placement starts to produce the buffers of its result (e.g., copies between
/// memories or devices, cross-host receives, host uploads, and compiled conversions), which the execution fence of the
/// result joins with the fence of the source. A readback of an array reads only one replica of each tile, and the
/// readiness functions of an array wait for its execution fence. The session tracks these buffers as well, but a buffer
/// whose producing work failed would otherwise go unnoticed by a readback that happens to read another replica of the
/// same tile.
#[derive(Default)]
struct PlacementCompletion {
    /// Ready events of the buffers that the placement created.
    events: Vec<Event<()>>,

    /// Execution fences of the programs that produced buffers of the result.
    fences: Vec<ExecutionFence>,
}

impl PlacementCompletion {
    /// Records the readiness of `buffer`, which the placement created.
    fn track_buffer(&mut self, buffer: &Buffer<'_>) -> Result<(), Error> {
        self.events.push(buffer.ready()?);
        Ok(())
    }

    /// Records the readiness of every buffer of `array` on this process whose shard has no elements. Nothing reads
    /// such a buffer, but the work that produced it can still fail, so the result of the placement reports it.
    fn track_empty_shard_buffers(&mut self, array: &Array) -> Result<(), Error> {
        for shard in array.addressable_shards().filter(|shard| shard.slice().iter().any(Range::is_empty)) {
            if let Some(buffer) = shard.buffer() {
                self.track_buffer(buffer)?;
            }
        }
        Ok(())
    }

    /// Records `buffer`, which the placement has just created, in `work` and then with [`Self::track_buffer`],
    /// and returns it as a shared buffer.
    fn track_new(
        &mut self,
        work: &mut WorkReservation,
        buffer: Buffer<'static>,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        work.track_buffer(&buffer);
        self.track_buffer(&buffer)?;
        Ok(Arc::new(buffer))
    }

    /// Records `fence`, which is the execution fence of a program that produced a buffer of the result, if any.
    fn track_fence(&mut self, fence: Option<&ExecutionFence>) {
        self.fences.extend(fence.cloned());
    }

    /// Returns the fence that joins `fence` (i.e., the execution fence of the source or of the final program) with the
    /// recorded work, or `fence` itself when nothing was recorded. Building a fence from the recorded events waits for
    /// an event whose callback PJRT fails to register, so the caller must have started whatever completes the recorded
    /// buffers (e.g., the thread of a background host upload) before it calls this function.
    fn join(self, fence: Option<ExecutionFence>) -> Option<ExecutionFence> {
        let mut fences = fence.into_iter().chain(self.fences).collect::<Vec<_>>();
        if !self.events.is_empty() {
            fences.push(ExecutionFence::new(self.events));
        }
        (!fences.is_empty()).then(|| ExecutionFence::join(fences))
    }
}

/// Part that one process takes in a [`ShardMove`], which follows from the processes of its two devices alone,
/// so that every process derives the parts of all processes alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MoveRole {
    /// The process owns both devices and shares the moved data on one device or copies it between its devices.
    Copy,

    /// The process owns only the source device and sends the moved data with a PJRT cross-host transfer.
    Send,

    /// The process owns only the target device and receives the moved data with a PJRT cross-host transfer.
    Receive,
}

/// Borrowed slice and optional contribution member that identify a group of source replicas. When there is one value
/// per slice, the slice alone identifies the group.
type SourceReplicaKey<'o> = (&'o [Range<usize>], Option<ShardContributionMember>);

/// Returns the key for finding source replicas that correspond to `shard`. The key borrows the shard's slice and
/// includes the contribution member with its manual coordinates ordered by `axes` when supplied.
fn source_replica_key<'o>(
    mesh: &DeviceMesh,
    sharding: &Sharding,
    shard: &'o ShardDescriptor,
    axes: Option<&[String]>,
) -> SourceReplicaKey<'o> {
    (shard.slice(), axes.map(|axes| ShardContributionMember::ordered(mesh, sharding, shard.index(), axes)))
}

/// Replicas of one source slice (i.e., the source shards that cover it), indexed so that a plan finds the preferred
/// replica for each target shard without scanning all of them again (e.g., when a replicated array moves to another
/// mesh, where every target shard has the slice of every source shard). The preference is deterministic and depends
/// only on global shard metadata, so that every process chooses the same source shard for a given target shard. It
/// prefers a source shard on the device of the target shard, then a source shard on the process of the target shard,
/// and finally breaks ties by device ID and shard index.
struct SourceReplicas<'o> {
    /// Replica with the lowest shard index on each device.
    by_device: HashMap<Device, &'o ShardDescriptor>,

    /// Replica with the lowest device ID, and then the lowest shard index, of each process.
    by_process: HashMap<usize, &'o ShardDescriptor>,

    /// Replica with the lowest device ID, and then the lowest shard index, of all processes.
    lowest: &'o ShardDescriptor,
}

impl<'o> SourceReplicas<'o> {
    /// Indexes `replicas`, which must not be empty, creating a new [`SourceReplicas`] instance.
    fn new(replicas: &[&'o ShardDescriptor]) -> Self {
        let order = |shard: &&ShardDescriptor| (shard.device().id(), shard.index());
        let mut by_device = HashMap::<Device, &ShardDescriptor>::new();
        let mut by_process = HashMap::<usize, &ShardDescriptor>::new();
        for &replica in replicas {
            let device = replica.device();
            let preferred = by_device.entry(device).or_insert(replica);
            if replica.index() < preferred.index() {
                *preferred = replica;
            }
            let preferred = by_process.entry(device.process_index()).or_insert(replica);
            if order(&replica) < order(preferred) {
                *preferred = replica;
            }
        }
        let lowest = replicas.iter().copied().min_by_key(order).unwrap();
        Self { by_device, by_process, lowest }
    }

    /// Groups `shards`, which are placed on `mesh` according to `sharding`, into the replicas of each slice and
    /// contribution member.
    fn by_contribution(
        mesh: &DeviceMesh,
        sharding: &Sharding,
        shards: impl IntoIterator<Item = &'o ShardDescriptor>,
        axes: Option<&[String]>,
    ) -> HashMap<SourceReplicaKey<'o>, Self> {
        let mut shards_by_contributions = HashMap::<SourceReplicaKey<'o>, Vec<&ShardDescriptor>>::new();
        for shard in shards {
            let key = source_replica_key(mesh, sharding, shard, axes);
            shards_by_contributions.entry(key).or_default().push(shard);
        }
        shards_by_contributions
            .into_iter()
            .map(|(key, shards)| (key, Self::new(shards.as_slice())))
            .collect()
    }

    /// Returns the preferred replica for `target_shard`.
    fn preferred(&self, target_shard: &ShardDescriptor) -> &'o ShardDescriptor {
        let device = target_shard.device();
        self.by_device
            .get(&device)
            .or_else(|| self.by_process.get(&device.process_index()))
            .copied()
            .unwrap_or(self.lowest)
    }
}

/// Names of the varying manual axes that are not unreduced of the two ends of a placement whose contributions
/// correspond, in the order in which their coordinates are compared. The unreduced axes need no names, because the
/// contributions to a pending reduction correspond by their ordinals.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ContributionAxes {
    /// Varying manual axes of the source in mesh order.
    source: Vec<String>,

    /// Varying manual axes of the target that correspond to those of [`Self::source`], in that order.
    target: Vec<String>,
}

/// One PJRT cross-host transfer that the current process takes part in, which [`issue_cross_host_transfers`] issues.
enum CrossHostTransfer<'b> {
    /// Send of `buffer` from its device `source` of the current process to the device `target` of another process.
    Send {
        /// ID of the source device.
        source: DeviceId,

        /// ID of the target device.
        target: DeviceId,

        /// Transfer rendezvous key shared with the matching receive.
        transfer_key: CrossHostTransferKey,

        /// Buffer to send.
        buffer: Arc<Buffer<'static>>,
    },

    /// Receive of a buffer of `element_type` and `dimensions` from the device `source` of another process onto
    /// `device`, which is the device `target` of the current process.
    Receive {
        /// ID of the source device.
        source: DeviceId,

        /// ID of the target device.
        target: DeviceId,

        /// Transfer rendezvous key shared with the matching send.
        transfer_key: CrossHostTransferKey,

        /// Target device.
        device: &'b PjrtDevice<'static>,

        /// Element type of the received buffer.
        element_type: BufferType,

        /// Dimensions of the received buffer.
        dimensions: Vec<i64>,
    },
}

impl CrossHostTransfer<'_> {
    /// Returns the ID of the source device.
    fn source(&self) -> DeviceId {
        match self {
            Self::Send { source, .. } | Self::Receive { source, .. } => *source,
        }
    }

    /// Returns the ID of the target device.
    fn target(&self) -> DeviceId {
        match self {
            Self::Send { target, .. } | Self::Receive { target, .. } => *target,
        }
    }
}

/// A buffer returned by one cross-host receive and the transfer key of that receive.
struct ReceivedShardBuffer {
    /// Key of the matching transfer.
    transfer_key: CrossHostTransferKey,

    /// Pending buffer returned by PJRT.
    buffer: Buffer<'static>,
}

/// Addressable devices of the client of an array by their identifiers.
struct AddressableDevices {
    /// Index of the process of the client.
    process_index: usize,

    /// Addressable devices of the client by their identifiers.
    devices: HashMap<DeviceId, PjrtDevice<'static>>,
}

impl AddressableDevices {
    /// Returns the addressable device with identifier `device_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::NonAddressableDevice`] when the client does not address that device.
    fn get(&self, device_id: DeviceId) -> Result<&PjrtDevice<'static>, Error> {
        let process_index = self.process_index;
        Ok(self.devices.get(&device_id).ok_or(ArrayError::NonAddressableDevice { device_id, process_index })?)
    }
}

/// Index of the shards of an array that hold a buffer on the process with index [`Self::process_index`] by the
/// contribution that they hold, which a placement keeps for all of its lookups of [`Array::local_replica_shard`],
/// so that many targets of one resident replica cost linear rather than quadratic work. The index is built on the
/// first lookup that needs another replica, so a placement whose planned shards all have buffers never builds it.
struct LocalReplicas {
    /// Index of the process whose shards the index holds.
    process_index: usize,

    /// Shards with a buffer of each contribution, in the order of their device identifiers and then their indices.
    shards: OnceCell<HashMap<ShardContribution, Vec<ShardIndex>>>,
}

impl LocalReplicas {
    /// Creates an empty index of the local replicas of the process with index `process_index`.
    fn new(process_index: usize) -> Self {
        Self { process_index, shards: OnceCell::new() }
    }
}

/// Plans the moves of whole shards that place an array of `shape` whose shards are `source_shards`, placed on
/// `source_mesh` according to `source_sharding`, on `mesh` according to `sharding`, for every process, with buffers of
/// `element_type`. The plan depends only on global metadata (i.e., on the slices and devices of the source shards, on
/// the meshes, and on the shardings).
///
/// Each target shard with elements receives a source shard that holds the same contribution and the plan may move any
/// replica of it, and a missing one is served from another replica of the same contribution. When both shardings hold
/// one value per slice, that is any source shard with the slices of the target shard. Otherwise, shards of one slice
/// can hold different contributions (e.g., the summands of an unreduced value or the values along a varying manual axis
/// that does not split the array). The placement must move each contribution unchanged. The target shard receives the
/// source shard that has the same slices and belongs to the corresponding member of the sequence of arrays formed by
/// the contributions. No arithmetic changes the contributions, and the host readback combines them in the same order.
/// Each process creates its own empty shards, which hold no contribution.
///
/// Returns `None` when a target shard with elements has no such source shard, which needs slicing or concatenating
/// shards, or when the shards of one slice can hold different contributions and the placement changes them.
///
/// # Errors
///
/// Returns the error of [`validate_kept_contributions`] when the target would leave a distinct source contribution
/// with elements out, and the error of [`ShardMovePlan::new`] for a device identifier that cross-host transfers cannot
/// represent. The plan covers the moves of every process, so every process returns these errors before any of them
/// submits a transfer.
fn plan_exact_shard_moves<'o>(
    element_type: BufferType,
    source_shards: impl IntoIterator<Item = &'o ShardDescriptor>,
    source_mesh: &DeviceMesh,
    source_sharding: &Sharding,
    shape: &StaticShape,
    mesh: &DeviceMesh,
    sharding: &Sharding,
) -> Result<Option<ShardMovePlan>, Error> {
    let Some(axes) = corresponding_shards(source_sharding, sharding) else {
        return Ok(None);
    };

    let source_axes = axes.as_ref().map(|axes| axes.source.as_slice());
    let target_axes = axes.as_ref().map(|axes| axes.target.as_slice());
    let replicas_by_contributions =
        SourceReplicas::by_contribution(source_mesh, source_sharding, source_shards, source_axes);

    let target_layout = ShardLayout::new(shape, mesh, sharding)?;
    let target_shards = target_layout.descriptors();
    let target_shard_count = target_shards.len();
    let mut moves = Vec::with_capacity(target_shard_count);
    let mut received = HashSet::new();
    for target_shard in target_shards {
        if target_shard.slice().iter().any(Range::is_empty) {
            continue;
        }

        let key = source_replica_key(mesh, sharding, target_shard, target_axes);
        let Some(replicas) = replicas_by_contributions.get(&key) else {
            return Ok(None);
        };

        received.insert(key);
        let source_shard = replicas.preferred(target_shard);
        moves.push(ShardMove {
            source_shard_index: source_shard.index(),
            source_device: source_shard.device(),
            target_shard_index: target_shard.index(),
            target_device: target_shard.device(),
            slice: target_shard.slice().to_vec(),
            transfer_key: exact_shard_transfer_key(source_shard.index(), target_shard.index(), target_shard_count)?,
        });
    }

    // Every target key names a distinct source key, so the target keeps every source contribution exactly when it
    // receives as many distinct ones.
    let sources = replicas_by_contributions.keys().filter(|(slice, _)| !slice.iter().any(Range::is_empty)).count();

    validate_kept_contributions(source_sharding, sources == received.len())?;

    Ok(Some(ShardMovePlan::new(element_type, moves)?))
}

/// Returns how the shards of a placement of a value placed as `input` with the sharding `output` correspond, which
/// [`plan_exact_shard_moves`] and [`PlacementPlanningView::shard_region_plan`] read: `Some(None)` when both hold one
/// value per slice, so that any shard with the same elements corresponds, `Some(Some(axes))` when the placement only
/// moves contributions, so a shard corresponds to the shards with the same elements and contribution member, and
/// `None` otherwise.
fn corresponding_shards(input: &Sharding, output: &Sharding) -> Option<Option<ContributionAxes>> {
    if input.holds_one_value_per_slice() && output.holds_one_value_per_slice() {
        return Some(None);
    }
    moves_contributions(input, output).map(Some)
}

/// Returns the unreduced axes of `sharding` and its varying manual axes that are not unreduced, in mesh order.
fn contribution_axes(sharding: &Sharding) -> (Vec<&MeshAxis>, Vec<&MeshAxis>) {
    let axes = sharding.mesh().axes().iter();
    let unreduced = axes.clone().filter(|axis| sharding.unreduced_axes().contains(axis.name())).collect();
    let varying_manual = axes
        .filter(|axis| {
            sharding.varying_manual_axes().contains(axis.name()) && !sharding.unreduced_axes().contains(axis.name())
        })
        .collect();
    (unreduced, varying_manual)
}

/// Returns whether a placement of a value placed as `input` with the sharding `output` only moves its contributions to
/// other devices. An array with a pending reduction or with manual variation within a tile is a sequence of ordinary
/// arrays, one for each contribution, and such a placement keeps that sequence and places each member like an ordinary
/// array. Only the axes that tell the contributions of one slice apart matter, which are the unreduced axes and the
/// varying manual axes (a split axis shows in the slice, and every other axis holds replicas, whatever its size, name,
/// or type, and whether the other mesh has it):
///
///   - **Unreduced Axes:** Both ends have none, or both have the same unreduced kind and the same number of
///     contributions (i.e., the product of the sizes of their unreduced axes). The contributions then correspond by
///     their ordinals, which is the order in which the host readback combines them, whatever the names, sizes, and
///     order of the unreduced axes, as JAX keeps the flattened order of the shards (e.g., `(x=2, y=3)` onto
///     `(y=3, x=2)`, onto `(z=6)`, or onto `(x=2, y=3, u=1)` with `u` pending too). Adding a pending reduction to a
///     value without one, even along axes of size one, is not a move.
///   - **Reduced Axes:** They hold replicas of a completed reduction, so they do not take part. Whether a placement
///     may change them is decided where the reduction state of a placement is validated.
///   - **Varying Manual Axes:** They correspond by name when both ends have the same names with the same sizes, and
///     otherwise in mesh order with the same sizes (e.g., along a renamed axis). A varying axis of the source that
///     splits a dimension of the source and along which the target does not vary takes no part, because its devices
///     hold disjoint slices, so no two values compete for one slice (e.g., the summands of a pending sum split along
///     a varying `m` gather into replicas of each summand).
///
/// The contributions of both ends then correspond by their slices and their members, so the placement changes
/// no value (e.g., onto another order of the same devices or onto other devices, in any memory and layout), and
/// [`plan_exact_shard_moves`] or [`PlacementPlanningView::shard_region_plan`] moves each contribution as it is.
///
/// Returns the corresponding varying manual axes of both ends, or `None` when the placement does not only move
/// contributions.
fn moves_contributions(input: &Sharding, output: &Sharding) -> Option<ContributionAxes> {
    let names = |axes: &[&MeshAxis]| axes.iter().map(|axis| axis.name().to_string()).collect::<Vec<_>>();
    let count = |axes: &[&MeshAxis]| axes.iter().map(|axis| axis.size()).product::<usize>();

    let (source_unreduced, source_varying) = contribution_axes(input);
    let (target_unreduced, target_varying) = contribution_axes(output);

    let unreduced_correspond = match (source_unreduced.is_empty(), target_unreduced.is_empty()) {
        (true, true) => true,
        (false, false) => {
            input.unreduced_kind() == output.unreduced_kind() && count(&source_unreduced) == count(&target_unreduced)
        }
        _ => false,
    };

    if !unreduced_correspond {
        return None;
    }

    let paired = |source: &[&MeshAxis]| {
        if source.len() != target_varying.len() {
            return None;
        }

        let matches_by_name = source
            .iter()
            .all(|axis| target_varying.iter().any(|other| other.name() == axis.name() && other.size() == axis.size()));

        let target = if matches_by_name {
            names(source)
        } else {
            let same_sizes = source.iter().zip(&target_varying).all(|(axis, other)| axis.size() == other.size());
            same_sizes.then(|| names(&target_varying))?
        };

        Some(ContributionAxes { source: names(source), target })
    };

    paired(&source_varying).or_else(|| {
        // A source axis that splits a dimension holds disjoint slices, so its variation gives no slice two values,
        // and a target that does not vary along it takes the member of each slice from the one device that holds it.
        // Such an axis takes no part then, which is tried only when the varying axes do not pair as they are, so that
        // they still pair by name and in mesh order (e.g., along a renamed axis).
        let gathered = source_varying
            .iter()
            .copied()
            .filter(|axis| !input.splits_along(axis.name()) || output.varying_manual_axes().contains(axis.name()))
            .collect::<Vec<_>>();
        (gathered.len() != source_varying.len()).then(|| paired(&gathered)).flatten()
    })
}

// TODO(eaplatanios): Review from here onwards.

/// Returns whether a value placed as `input`, which has a pending reduction or manual variation within a tile (i.e.,
/// does not hold one value per slice, refer to [`Sharding::holds_one_value_per_slice`]), keeps its contributions
/// when placed with the sharding `output` (refer to [`moves_contributions`]). Such a placement only moves bytes, so no
/// strategy of it may read back the combined value of its contributions and upload it again (i.e., neither the host
/// round trip nor any other host fallback, refer to [`keeps_some_contributions`]).
fn keeps_contributions(input: &Sharding, output: &Sharding) -> bool {
    !input.holds_one_value_per_slice() && moves_contributions(input, output).is_some()
}

/// Returns whether a placement of a value placed as `input` with the sharding `output` keeps some of its contributions
/// apart, which is the one rule that admits a host fallback. Every host fallback (i.e., the planned
/// [`DevicePlacementStrategy::HostRoundTrip`], the host fallback of [`Array::place_within_client`], and the readback
/// of [`Array::place_process_local_value`]) reads back the combined value of all contributions and uploads it again,
/// which is valid only when the placement keeps none of them. It keeps all of them when it only moves them (refer to
/// [`keeps_contributions`]), and some of them when it keeps a pending reduction along a mesh axis of size greater than
/// one while it changes the reduction along others (refer to [`kept_unreduced_axes`]), whose partial results the
/// combined value would replace (e.g., completing `y` of a sum along `x` and `y` of the `f32` contributions `MAX`, `0`,
/// `MAX`, and `-MAX` leaves `MAX` and `0` along `x`, while the combined value is infinite). An axis of size one holds
/// one contribution, which combining leaves as it is.
fn keeps_some_contributions(input: &Sharding, output: &Sharding) -> bool {
    let has_distinct_contributions = input.mesh().axes().iter().any(|axis| {
        axis.size() > 1
            && (input.unreduced_axes().contains(axis.name())
                || (input.varying_manual_axes().contains(axis.name()) && !input.splits_along(axis.name())))
    });
    let kept = kept_unreduced_axes(input, output);
    (has_distinct_contributions && keeps_contributions(input, output))
        || output.mesh().axes().iter().any(|axis| axis.size() > 1 && kept.contains(&axis.name()))
}

/// Returns the names of the mesh axes of `output` along which a placement of a value placed as `input` with the
/// sharding `output` keeps a pending reduction. Such an axis is unreduced in both with the same unreduced kind, and
/// both meshes have an axis of that name with the same size. Every other axis of either mesh (e.g., a replicated or
/// unused one of any size or type, or one that only one mesh has) does not matter, so the decision never depends on
/// the equality of the meshes, and the remaining unreduced axes of `input` are completed and those of `output`
/// introduced. An unreduced axis that the target renames has no common name and is not kept, so a placement that
/// keeps it while it changes another part of the reduction state completes it and introduces it again. A placement
/// whose whole reduction state corresponds under other names is a move instead (refer to [`moves_contributions`]),
/// which never gets here.
fn kept_unreduced_axes<'o>(input: &Sharding, output: &'o Sharding) -> Vec<&'o str> {
    if input.unreduced_kind() != output.unreduced_kind() {
        return Vec::new();
    }
    let axes = output.mesh().axes().iter();
    axes.filter(|axis| {
        let name = axis.name();
        output.unreduced_axes().contains(name)
            && input.unreduced_axes().contains(name)
            && input.mesh().axes().iter().any(|other| other.name() == name && other.size() == axis.size())
    })
    .map(MeshAxis::name)
    .collect()
}

/// Checks that a placement of a value placed as `input` keeps every distinct contribution with elements of its source
/// (refer to [`ShardContribution`]), which is the case when `keeps_all` holds. A placement whose contributions
/// correspond (refer to [`corresponding_shards`]) takes each target contribution from the source contribution
/// with the same elements and member, so it can leave a source contribution out only along a varying manual axis that
/// splits a dimension at one end and not at the other (e.g., the values `10`, `11`, `20`, and `21` of a vector of two
/// elements split along `x` and varying along `m`, placed so that `m` splits it, which keeps only `10` and `21`). The
/// target cannot represent that value, so the placement is rejected rather than planned another way.
///
/// # Errors
///
/// Returns [`ProgramError::InvalidArgument`] naming the varying manual axes of `input` when `keeps_all` is false.
fn validate_kept_contributions(input: &Sharding, keeps_all: bool) -> Result<(), Error> {
    if keeps_all {
        return Ok(());
    }
    let axes = input.varying_manual_axes().iter().map(|axis| format!("`{axis}`")).collect::<Vec<_>>();
    Err(ProgramError::InvalidArgument {
        message: format!(
            "cannot place an array that varies along the manual mesh axes {} onto this sharding, because the placement \
             would discard values that differ along those axes",
            axes.join(", "),
        ),
    }
    .into())
}

/// Returns the cross-host shape, attaching the target shard and dimension to any encoding failure.
fn cross_host_shape(shard_index: ShardIndex, shape: &StaticShape) -> Result<Vec<i64>, Error> {
    shape
        .as_slice()
        .iter()
        .enumerate()
        .map(|(dimension, &size)| {
            i64::try_from(size).map_err(|_| {
                Error::from(ArrayError::InvalidShardDimension {
                    shard_index,
                    dimension,
                    source: DimensionError::ExtentExceedsBackendWidth { value: size, maximum: i64::MAX as usize },
                })
            })
        })
        .collect()
}

/// Returns a lazy resolver for the default layout that a cross-host receive allocates. Each lookup checks for the
/// PJRT cross-host transfers extension, then asks the plugin for its default layout. A missing default layout
/// query returns [`None`], and an unrepresentable shape reports the target shard and dimension.
fn cross_host_layout_resolver<'o>(
    client: &'o Client<'_>,
) -> impl Fn(BufferType, ShardIndex, &StaticShape) -> Result<Option<ryft_pjrt::Layout>, Error> + 'o {
    move |element_type, shard_index, shape| {
        client.cross_host_transfers_extension()?;
        let dimensions = cross_host_shape(shard_index, shape)?
            .into_iter()
            .map(|dimension| dimension as u64)
            .collect::<Vec<_>>();
        match client.default_layout(element_type, dimensions).and_then(|layout| layout.proto()) {
            Ok(layout) => Ok(Some(ryft_pjrt::Layout::from_proto(layout)?)),
            Err(PjrtError::Unimplemented { .. } | PjrtError::MissingFunction { .. }) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }
}

/// Uses `layout_resolver` to obtain the default layout for buffers of `element_type` with each shape of
/// `shards` (i.e., of pairs of the index of a target shard and a shape that crosses processes for it), or `None`
/// when `shards` is not empty and the plugin lacks the cross-host transfers extension or cannot report one of those
/// layouts. The first shard with each shape supplies the index for a single lookup of that shape. Every process derives
/// the same `shards` for a placement and queries the same plugin, so all of them get the same result.
fn cross_host_layouts(
    layout_resolver: &CrossHostLayoutResolver<'_>,
    element_type: BufferType,
    shards: impl Iterator<Item = (ShardIndex, StaticShape)>,
) -> Result<Option<HashMap<StaticShape, ryft_pjrt::Layout>>, Error> {
    let mut layouts = HashMap::new();
    for (shard_index, shape) in shards {
        if let Entry::Vacant(entry) = layouts.entry(shape) {
            match layout_resolver(element_type, shard_index, entry.key()) {
                Ok(Some(layout)) => {
                    entry.insert(layout);
                }
                Ok(None) | Err(Error::Pjrt(PjrtError::Unimplemented { .. })) => return Ok(None),
                Err(error) => return Err(error),
            }
        }
    }
    Ok(Some(layouts))
}

/// Returns the deterministic cross-host transfer key for moving data from `source_shard_index` to the
/// target shard `target_shard_index`, which is `source_shard_index * target_shard_count +
/// target_shard_index`, so that every pair of source and target shards gets a unique key.
///
/// # Errors
///
/// Returns [`Error::SizeLimitExceeded`] when the key overflows [`usize`] or [`i64`].
fn exact_shard_transfer_key(
    source_shard_index: ShardIndex,
    target_shard_index: ShardIndex,
    target_shard_count: usize,
) -> Result<CrossHostTransferKey, Error> {
    let transfer_key = source_shard_index
        .checked_mul(target_shard_count)
        .and_then(|key| key.checked_add(target_shard_index))
        .ok_or_else(|| Error::SizeLimitExceeded {
            message: format!(
                "cross-host transfer key for source shard #{source_shard_index} and target shard \
                 #{target_shard_index} exceeds the maximum allowed size of {}",
                usize::MAX,
            ),
        })?;
    i64::try_from(transfer_key).map_err(|_| Error::SizeLimitExceeded {
        message: format!(
            "cross-host transfer key {transfer_key} for source shard #{source_shard_index} and target shard \
             #{target_shard_index} exceeds the maximum allowed size of {}",
            i64::MAX,
        ),
    })
}

/// Returns the PJRT cross-host global device ID for `device_id`.
fn cross_host_global_device_id(device_id: DeviceId) -> Result<GlobalDeviceId, Error> {
    i32::try_from(device_id).map_err(|_| Error::SizeLimitExceeded {
        message: format!("cross-host transfer device ID {device_id} exceeds the maximum allowed value of {}", i32::MAX),
    })
}

/// Issues `transfers`, which are the cross-host transfers that the current process takes part in for one placement,
/// and returns a fence for the sends and the received buffers with their transfer keys and device IDs.
///
/// PJRT delivers a received buffer to the default memory of its target device, which every device has. When
/// `memory` is another memory, each received buffer is therefore copied on its device to `memory` once every transfer
/// is issued, and the copy is returned in its place. That copy is local to the device and does not wait for the
/// transfer, so it does not affect the order described below.
///
/// The transfers are issued one at a time in the order of their source and then target device IDs, keeping the
/// given order of the transfers between the same two devices. The CUDA plugin carries the transfers between two
/// devices over a communicator of that pair and pairs them by the order in which both sides issue them, ignoring their
/// transfer keys, and the first transfer between two devices blocks the calling thread until the peer process issues
/// its matching transfer. Two processes that each send before they receive would therefore deadlock when they swap
/// shards. Every process derives the same transfers from global metadata and issues its own in this one global order,
/// so the first pair of devices with a pending transfer always has both of its processes waiting for it.
///
/// A process that stops after issuing some of its transfers leaves its peers waiting for the rest, and the transfers
/// that it did issue are paired with the next transfers between the same devices. This function therefore validates
/// the device IDs of every transfer, which every process checks alike for the transfers that it shares with a peer,
/// before it issues the first one. It issues every transfer before it looks up `memory` or copies a received buffer
/// to it, because only the receiving process can look up the memories of its devices. A memory that a device does not
/// have or a failed copy is therefore returned only after every transfer of this process has been issued, so the peers
/// of this process complete their transfers and only this process reports the failure.
fn issue_cross_host_transfers(
    session: &Session,
    mut transfers: Vec<CrossHostTransfer<'_>>,
    memory: Memory,
) -> Result<(Option<ExecutionFence>, Vec<ReceivedShardBuffer>), Error> {
    if transfers.is_empty() {
        return Ok((None, Vec::new()));
    }
    transfers.sort_by_key(|transfer| (transfer.source(), transfer.target()));
    let transfers = transfers
        .into_iter()
        .map(|transfer| {
            let source = cross_host_global_device_id(transfer.source())?;
            let target = cross_host_global_device_id(transfer.target())?;
            Ok((source, target, transfer))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let client = session.client();
    let mut sends = session.reserve_work();
    let mut receives = session.reserve_work();
    let mut received_buffers = Vec::new();
    for (source, target, transfer) in transfers {
        match transfer {
            CrossHostTransfer::Send { transfer_key, buffer, .. } => {
                sends.track_events(client.cross_host_send_buffers(&[&buffer], &[target], &[transfer_key], &[])?);
            }
            CrossHostTransfer::Receive { transfer_key, device, element_type, dimensions, .. } => {
                let buffers = client.cross_host_receive_buffers(
                    &[element_type],
                    &[dimensions.as_slice()],
                    device,
                    &[source],
                    &[transfer_key],
                    &[],
                )?;
                for buffer in buffers {
                    receives.track_buffer(&buffer);
                    received_buffers.push((device, ReceivedShardBuffer { transfer_key, buffer }));
                }
            }
        }
    }
    let send_fence = sends.finish();
    let received_buffers = received_buffers
        .into_iter()
        .map(|(device, mut received)| {
            if memory != Memory::Device {
                received.buffer = received.buffer.copy_to_memory(device.memory(memory)?)?;
                receives.track_buffer(&received.buffer);
            }
            Ok(received)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok((send_fence, received_buffers))
}

/// Returns the dense row-major [`Layout`] of rank `rank` (i.e., the tiled layout without tiles whose minor-to-major
/// order is the reverse of the order of the dimensions), which is the default layout of dense buffers of the CPU and
/// CUDA plugins and the layout of the slices and of the received regions of [`Array::place_shard_regions`].
fn dense_row_major_layout(rank: usize) -> Layout {
    Layout::Tiled(TiledLayout::new((0..rank).rev().collect(), Vec::new()))
}

/// Returns `input` without the pending reductions that a placement with the sharding `output` does not keep (refer to
/// [`kept_unreduced_axes`]) and with every dimension replicated, over `mesh`, which has the axes that `output` keeps
/// (i.e., the mesh of `input` or of `output`). This is the sharding of the value that a compiled reshard across meshes
/// moves as whole shards (refer to [`DevicePlacementStrategy::CompiledReshardAcrossMeshes`]), so the gather on the mesh
/// of the source completes only the reductions that the target does not keep, and the kept ones move as their
/// contributions.
fn gathered_sharding(input: &Sharding, output: &Sharding, mesh: &LogicalMesh, rank: usize) -> Result<Sharding, Error> {
    let replicated = Sharding::replicated(mesh.clone(), rank);
    let kept = kept_unreduced_axes(input, output);
    if kept.is_empty() {
        return Ok(replicated);
    }
    Ok(replicated.with_unreduced_axes(kept)?.with_unreduced_kind(input.unreduced_kind())?)
}

/// Returns `sharding` over `mesh`, a logical mesh with the axes of the mesh of `sharding` in the same order and with
/// the same names and types, some of which may have another size (e.g., the logical mesh of
/// [`DeviceMeshExtension::at_coordinate_zero`]), with the same placement, reduction state, and manual variation. Unlike
/// [`Sharding::with_mesh`], which requires the same topology, this lets a sharding that splits no dimension along a
/// resized axis describe the same shards on the devices that remain.
fn sharding_on_mesh(sharding: &Sharding, mesh: &LogicalMesh) -> Result<Sharding, Error> {
    Ok(Sharding::new(mesh.clone(), sharding.dimensions().to_vec())?
        .with_unreduced_axes(sharding.unreduced_axes().clone())?
        .with_unreduced_kind(sharding.unreduced_kind())?
        .with_reduced_axes(sharding.reduced_axes().clone())?
        .with_varying_manual_axes(sharding.varying_manual_axes().clone())?)
}

/// Validates that a compiled reshard on `mesh` (refer to [`ReshardProgram`]) takes a value of `data_type` placed as
/// `input` to `output` with the value that the meaning of its reduction state gives, as far as the reduction state
/// (refer to [`ReductionState`]) along each non-manual mesh axis goes. That meaning is owned by
/// [`crate::arrays::host`], whose section on pending reductions also states the contract of floating point sums. The
/// program keeps the state of an axis as its contributions, which preserves every value, and it makes every other
/// change with a collective of XLA (refer to [`ReshardProgram::compilation_mesh`]). The device path is taken exactly when
/// that collective computes that meaning up to that contract, which is the case as follows.
///
///   - **Completing a pending reduction:** Exact only where [`collective_reduction`] classifies the completion as
///     [`CollectiveReduction::Exact`] (a maximum or minimum that drops NaN values is not).
///   - **Introducing a pending sum:** No program of a placement does, because the planner introduces pending sums with
///     additive identities after a placement without them (refer to [`PlacementPlanningView::strategy`]). This
///     check validates only the completion of a pending reduction of `input` along such an axis.
///   - **Introducing a pending maximum or minimum:** XLA cannot (refer to `validate_declared_pending_reduction` in
///     `crate::lowering`, whose check would reject the program).
///   - **Completing or introducing a completed reduction:** This changes no value.
///   - **Partial completions of wider sums:** A completion of a pending sum along some axes that keeps it along another
///     is declined as a whole where [`partial_completion_rejection`] rejects it (i.e., where the partial sums would be
///     rounded to a data type narrower than the one in which the sum accumulates), which the lowering of compiled
///     programs reads as well.
///   - **Axes of size one:** Completing a reduction of a single contribution changes no value, so such a completion is
///     always made on the devices, as the lowering of compiled programs exempts it (refer to
///     `validate_pending_reduction_change` in `crate::lowering`). XLA cannot introduce a pending maximum or minimum
///     along such an axis either, so that is declined as along any other axis.
///
/// Returns the reason why the program does not give that value, if any, which makes the placement take a host round
/// trip with the readback and upload of [`crate::arrays::host`].
fn validate_reduction_transition(
    data_type: DataType,
    mesh: &DeviceMesh,
    input: &Sharding,
    output: &Sharding,
) -> Option<HostRoundTripCause> {
    let sum = ReductionState::Unreduced(UnreducedKind::Sum);
    let sum_axes = |completed: bool| {
        mesh.logical_mesh()
            .axes()
            .iter()
            .filter(move |axis| {
                axis.r#type() != MeshAxisType::Manual
                    && ReductionState::of(input, axis.name()) == sum
                    && (ReductionState::of(output, axis.name()) == sum) != completed
            })
            .map(|axis| (axis.name(), axis.size()))
    };
    if let Some(transition) =
        partial_completion_rejection(data_type, UnreducedKind::Sum, sum_axes(true), sum_axes(false))
    {
        let remedy = "complete the sum along every mesh axis while the devices of one process hold the value";
        return Some(HostRoundTripCause::Reduction { transition, remedy });
    }
    for axis in mesh.logical_mesh().axes() {
        let (from, to) = (ReductionState::of(input, axis.name()), ReductionState::of(output, axis.name()));
        if axis.r#type() == MeshAxisType::Manual || from == to {
            continue;
        }
        let completion = match from {
            ReductionState::Unreduced(kind) if axis.size() > 1 => {
                let classification = collective_reduction(data_type, ReductionChange::Complete(kind));
                let remedy = if classification == CollectiveReduction::DropsNan {
                    "complete the reduction while the devices of one process hold the value, or in a compiled \
                     program, which drops NaN values as the collective does"
                } else {
                    "complete the reduction while the devices of one process hold the value, or in another data type"
                };
                classification.reason().map(|reason| (reason, remedy))
            }
            _ => None,
        };
        let rejection = match (completion, to) {
            (Some(rejection), _) => Some(rejection),
            (None, ReductionState::Unreduced(kind @ (UnreducedKind::Max | UnreducedKind::Min))) => {
                collective_reduction(data_type, ReductionChange::Introduce(kind)).reason().map(|reason| {
                    let remedy = "make this placement where one process owns every device of the target and a \
                                  device of every source contribution with elements";
                    (reason, remedy)
                })
            }
            _ => None,
        };
        if let Some((reason, remedy)) = rejection {
            let transition = format!(
                "cannot take `{data_type}` values from {from} to {to} along mesh axis `{}` on the devices, because \
                 {reason}",
                axis.name(),
            );
            return Some(HostRoundTripCause::Reduction { transition, remedy });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use crate::arrays::tests::{
        explicit_mesh, first_device, pending_replicated_vector, place_through_entry_points, process_one_cpu_client,
        process_outcomes, shard_memory_kinds, transfer_managers, two_process_mesh_2x2,
    };
    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array as CoreArray, ConvertElementType, DataType, Dimension, DimensionBounds, DimensionError,
        DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, PlaceDispatch, PlaceOperation, Shape,
        ShardingDimension, StridedLayout, TiledLayout, UnreducedKind, ValueDomainDispatch,
    };
    use ryft_pjrt::{BufferSpecification, BufferType, ExecutionFence};

    #[cfg(feature = "cuda-13")]
    use crate::arrays::tests::cuda_rank_session;
    #[cfg(feature = "cuda-13")]
    use crate::distributed::tests::{read_addressable_f32s, run_as_cuda_rank, run_cuda_rank_pair};
    use crate::sessions::tests::{DeclareRemoteDevices, domain_mesh, two_device_mesh};
    use crate::tests::{
        cpu_client, four_device_mesh_x, logical_mesh_2x2, peer_devices, session_with_remote_devices,
        sessions_of_processes, shard_values, values_from_bytes, values_to_bytes,
    };
    use crate::{ExecutionError, FromPjrt, PlacementOptions};

    use super::*;

    /// Cross-host transfer that one process plans for a placement, which its send and its receive must agree on.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct PlannedTransfer {
        source: Device,
        target: Device,
        key: CrossHostTransferKey,
        shape: Vec<usize>,
    }

    /// Work that crosses processes, which one process plans for its part of a placement. Its programs are the
    /// compiled reshards that it runs, which every process that owns a device of their meshes must run alike.
    struct PlannedWork {
        strategy: String,
        sends: Vec<PlannedTransfer>,
        receives: Vec<PlannedTransfer>,
        programs: Vec<ReshardProgram>,
    }

    /// Simulates a PJRT plugin that supports cross-host transfers and reports row-major layouts, as CUDA does.
    fn simulated_cross_host_layout(
        _: BufferType,
        _: ShardIndex,
        shape: &StaticShape,
    ) -> Result<Option<ryft_pjrt::Layout>, Error> {
        let minor_to_major = (0..shape.rank()).rev().collect();
        Ok(Some(Layout::Tiled(TiledLayout::new(minor_to_major, vec![])).to_pjrt()?))
    }

    /// Plans a reduction transition from shard metadata, without constructing buffers or executing a transfer.
    fn plan_reduction_transition(
        data_type: DataType,
        shape: &StaticShape,
        mesh: &DeviceMesh,
        input: &Sharding,
        output: &Sharding,
    ) -> Result<DevicePlacementStrategy, Error> {
        let r#type = ArrayType::new(data_type, shape.clone().into());
        let (metadata, shards) = ArrayMetadata::new(r#type, Some(input.clone()), mesh.clone(), None)?;
        PlacementPlanningView::new(&metadata, shards.iter()).strategy(&simulated_cross_host_layout, mesh, output)
    }

    /// Returns the work that crosses processes, which the process with index `process_index` plans when it places
    /// `source`, its own view of the source of a placement, on `mesh` according to `sharding`. The strategy, with the
    /// moves and programs of every process, comes from `Array::device_placement_strategy`, which the placement
    /// executes, and the part of this process in them from `ShardMove::role` and `ReshardProgram::runs_on`, which the
    /// placement follows as well, so this is the work that the placement performs.
    fn planned_work(
        source: &Array,
        layout_resolver: &CrossHostLayoutResolver<'_>,
        process_index: usize,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<PlannedWork, String> {
        let strategy = source
            .device_placement_strategy(layout_resolver, mesh, sharding, process_index)
            .map_err(|error| error.to_string())?;
        // The additive identities of introduced sums are uploads of each process, so only their first step crosses
        // processes.
        fn work(strategy: &DevicePlacementStrategy) -> (String, Option<&ShardMovePlan>, Vec<&ReshardProgram>) {
            match strategy {
                DevicePlacementStrategy::Bufferless => ("bufferless".to_string(), None, Vec::new()),
                DevicePlacementStrategy::ProcessLocalValue => ("process-local value".to_string(), None, Vec::new()),
                // The owner runs every part of its strategy on its own devices, so this process takes no part.
                DevicePlacementStrategy::OnOwner { strategy } => {
                    (format!("{} on its owner", work(strategy).0), None, Vec::new())
                }
                DevicePlacementStrategy::ExactShards { plan, .. } => {
                    ("exact shards".to_string(), Some(plan), Vec::new())
                }
                DevicePlacementStrategy::ShardRegions { plan, .. } => {
                    ("shard regions".to_string(), Some(plan), Vec::new())
                }
                DevicePlacementStrategy::CompiledReshard { program } => {
                    ("compiled reshard".to_string(), None, vec![program])
                }
                DevicePlacementStrategy::CompiledReshardAcrossMeshes { gather, replicas, reshard, .. } => (
                    "compiled reshard across meshes".to_string(),
                    Some(replicas),
                    gather.iter().chain([reshard]).collect(),
                ),
                DevicePlacementStrategy::IntroducedSums { first: Some(first), .. } => {
                    let (name, plan, programs) = work(first);
                    (format!("{name} with additive identities"), plan, programs)
                }
                DevicePlacementStrategy::IntroducedSums { first: None, .. } => {
                    ("additive identities".to_string(), None, Vec::new())
                }
                DevicePlacementStrategy::HostRoundTrip { .. } => ("host round trip".to_string(), None, Vec::new()),
            }
        }
        let (name, plan, programs) = work(&strategy);
        let transfers = |moves: Vec<&ShardMove>| {
            moves
                .into_iter()
                .map(|movement| PlannedTransfer {
                    source: movement.source_device,
                    target: movement.target_device,
                    key: movement.transfer_key,
                    shape: movement.shape().as_slice().to_vec(),
                })
                .collect()
        };
        Ok(PlannedWork {
            strategy: name,
            sends: transfers(
                plan.into_iter().flat_map(|plan| plan.moves_with_role(process_index, MoveRole::Send)).collect(),
            ),
            receives: transfers(
                plan.into_iter().flat_map(|plan| plan.moves_with_role(process_index, MoveRole::Receive)).collect(),
            ),
            programs: programs.into_iter().filter(|program| program.runs_on(process_index)).cloned().collect(),
        })
    }

    /// Plans the placement of the source of `views` (i.e., the source as each process sees it, in process order) on
    /// `mesh` according to `sharding` on every process, and returns the strategy that all of them choose (or the
    /// strategies of all of them in process order, separated by ` | `, when they differ) when their plans agree, or
    /// how they disagree otherwise, preceded by those strategies when every process plans without failing. The plans
    /// agree when every process fails alike before it posts anything or none fails, every send has exactly one
    /// matching receive on its peer, every receive has a matching send, and every process that owns a device of a
    /// program that one process runs runs the same program. The processes may choose different strategies only when
    /// that holds (e.g., a process whose own value lies on its devices and whose devices hold the whole target).
    ///
    /// Agreeing plans are not enough when a process holds some buffers of the source, because what only it knows
    /// (i.e., which of its shards have buffers) could still stop it before it posts the transfers that its peers
    /// expect. The view for process 0 runs its part of the strategy with the production path when it holds buffers and
    /// plans transfers. The CPU plugin lacks the cross-host transfers extension, so that part must fail with exactly
    /// the error of its first transfer, and any other outcome is a disagreement.
    fn placement_agreement(views: &[Array], mesh: &DeviceMesh, sharding: &Sharding) -> Result<String, String> {
        let works = views
            .iter()
            .enumerate()
            .map(|(process_index, view)| {
                planned_work(view, &simulated_cross_host_layout, process_index, mesh, sharding)
            })
            .collect::<Vec<_>>();
        let works = match works.iter().map(Result::as_ref).collect::<Result<Vec<_>, _>>() {
            Ok(works) => works,
            Err(error) if works.iter().all(|work| work.as_ref().err() == Some(error)) => {
                return Ok(format!("error: {error}"));
            }
            Err(_) => {
                let outcomes = works.iter().map(|work| work.as_ref().map(|work| &work.strategy)).collect::<Vec<_>>();
                return Err(format!("processes disagree on failing: {outcomes:?}"));
            }
        };
        let strategies = works.iter().map(|work| work.strategy.as_str()).collect::<Vec<_>>();
        let planned = if strategies.iter().all(|strategy| *strategy == strategies[0]) {
            strategies[0].to_string()
        } else {
            strategies.join(" | ")
        };
        for (process_index, work) in works.iter().enumerate() {
            for send in &work.sends {
                let peer = send.target.process_index();
                if works[peer].receives.iter().filter(|receive| *receive == send).count() != 1 {
                    return Err(format!(
                        "{planned}: process {process_index} sends {send:?} without one receive on process {peer}"
                    ));
                }
            }
            for receive in &work.receives {
                let peer = receive.source.process_index();
                if !works[peer].sends.contains(receive) {
                    return Err(format!(
                        "{planned}: process {process_index} receives {receive:?} that process {peer} never sends"
                    ));
                }
            }
            for program in &work.programs {
                for peer in program.mesh.devices().iter().map(Device::process_index) {
                    if !works[peer].programs.contains(program) {
                        return Err(format!(
                            "{planned}: process {process_index} runs {program:?} without process {peer}"
                        ));
                    }
                }
            }
        }
        let holds_buffers = views[0].addressable_shards().any(|shard| shard.buffer().is_some());
        if holds_buffers && !(works[0].sends.is_empty() && works[0].receives.is_empty()) {
            let source = &views[0];
            let strategy = source.device_placement_strategy(&simulated_cross_host_layout, mesh, sharding, 0);
            let layout = source.placement_layout(None, mesh, sharding).unwrap();
            let memory = source.r#type().memory();
            let placed = strategy.and_then(|strategy| {
                let placed =
                    source.clone().place_with_strategy(strategy, 0, mesh, sharding, memory, layout.as_ref(), false);
                placed.map_err(|(error, _)| error)
            });
            match placed {
                Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                    if message == "the cross-host transfers extension is not provided by the PJRT plugin" => {}
                Ok(_) => return Err(format!("{planned}: process 0 completes without posting its transfers")),
                Err(error) => return Err(format!("{planned}: process 0 fails before it posts its transfers: {error}")),
            }
        }
        Ok(planned)
    }

    /// Returns the views that two processes have of a source of type `r#type` on the [`explicit_mesh`] of `devices`,
    /// with the sharding that `sharding` returns for that mesh, whose devices every process knows. Every process sees
    /// the same metadata of such a source, whichever buffers it holds, so each view imports that metadata without
    /// buffers.
    fn shared_views(
        session: &Session,
        r#type: &ArrayType,
        devices: &[Device],
        sharding: &dyn Fn(&DeviceMesh) -> Sharding,
    ) -> Vec<Array> {
        let mesh = explicit_mesh(devices);
        let view = || {
            let sharding = Some(sharding(&mesh));
            Array::from_addressable_buffers(session, r#type.clone(), mesh.clone(), sharding, Vec::new()).unwrap()
        };
        vec![view(), view()]
    }

    /// Places `view`, the view that process 0 has of the source of a placement, on `mesh` according to `sharding`
    /// with the strategy that process 0 plans and the production path that runs it. Returns the number of the
    /// addressable shards of the result that hold buffers, as `<count> buffers`, or the error of the placement, after
    /// it asserts that the result has the requested mesh and sharding, or that a failed placement returns no source to
    /// fall back on.
    fn place_on_process_zero(view: &Array, mesh: &DeviceMesh, sharding: &Sharding) -> Result<String, String> {
        let strategy = view
            .device_placement_strategy(&simulated_cross_host_layout, mesh, sharding, 0)
            .map_err(|error| error.to_string())?;
        match view.clone().place_with_strategy(strategy, 0, mesh, sharding, Memory::Device, None, false) {
            Ok(placed) => {
                assert_eq!((placed.mesh(), placed.sharding()), (mesh, sharding));
                Ok(format!("{} buffers", placed.addressable_shards().filter(|shard| shard.buffer().is_some()).count()))
            }
            Err((error, source)) => {
                assert!(source.is_none(), "a process 0 that cannot place the source returns no source to fall back on");
                Err(error.to_string())
            }
        }
    }

    #[test]
    fn test_array_with_copied_shard_buffers() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let source = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]).with_memory(Memory::Host { pinned: true }),
            mesh,
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();

        // Every shard is copied into the same memory of its device, so the result shares no buffer with the source.
        let copied = source.with_copied_shard_buffers(|_| true).unwrap();
        let shares_buffers = |copied: &Array| {
            copied
                .shards()
                .iter()
                .zip(source.shards())
                .map(|(copied, source)| Arc::ptr_eq(copied.buffer().unwrap(), source.buffer().unwrap()))
                .collect::<Vec<_>>()
        };
        assert_eq!(copied.r#type(), source.r#type());
        assert_eq!(copied.mesh(), source.mesh());
        assert_eq!(copied.sharding(), source.sharding());
        assert_eq!(shares_buffers(&copied), vec![false; 2]);
        assert_eq!(shard_memory_kinds(&copied), ["pinned_host"; 2]);
        assert_eq!(shard_values(&copied), vec![vec![1., 2., 3., 4.]; 2]);
        assert!(copied.is_committed());

        // Only the selected buffers are copied, the others are shared, and the result keeps the commitment of the
        // array.
        let second_buffer = Arc::clone(source.shards()[1].buffer().unwrap());
        let copied = source
            .clone()
            .with_committed(false)
            .with_copied_shard_buffers(|buffer| Arc::ptr_eq(buffer, &second_buffer))
            .unwrap();
        assert_eq!(shares_buffers(&copied), vec![true, false]);
        assert_eq!(shard_memory_kinds(&copied), ["pinned_host"; 2]);
        assert_eq!(shard_values(&copied), vec![vec![1., 2., 3., 4.]; 2]);
        assert!(!copied.is_committed());

        // A copy stays on the devices of its array, so it keeps whether this process chose them on its own.
        assert!(!copied.is_locally_assigned());
        let locally_assigned = source.with_locally_assigned(true);
        assert!(locally_assigned.is_locally_assigned());
        let copied = locally_assigned.with_copied_shard_buffers(|_| true).unwrap();
        assert!(copied.is_locally_assigned());
    }

    #[test]
    fn test_array_with_copied_shard_buffers_fails_readback_of_failed_replica() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);

        // Only the copy of the second replica fails, and a readback reads one replica of each tile, so the result must
        // report the failure through its fence.
        let (source, managers) = pending_replicated_vector(&session, &mesh);
        let copied = source.with_copied_shard_buffers(|_| true).unwrap();

        // A copy of only the second replica joins its failure into the fence of the result in the same way.
        let second_buffer = Arc::clone(source.shards()[1].buffer().unwrap());
        let partially_copied = source.with_copied_shard_buffers(|buffer| Arc::ptr_eq(buffer, &second_buffer)).unwrap();
        managers[0]
            .transfer_data(0, Arc::new(values_to_bytes(&[1f32, 2., 3., 4.])), 0, true)
            .unwrap()
            .r#await()
            .unwrap();
        managers[1].set_error(0, PjrtError::aborted("second replica failed")).unwrap();
        assert!(matches!(
            copied.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second replica failed",
        ));
        assert!(matches!(
            copied.copy_to_host_bytes(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second replica failed",
        ));
        assert!(matches!(
            partially_copied.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second replica failed",
        ));
        assert!(matches!(
            partially_copied.copy_to_host_bytes(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second replica failed",
        ));

        // Each of the two copies is one submission of the session, and each reports the failure once.
        assert!(matches!(
            session.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [
                    (_, Error::Pjrt(PjrtError::Aborted { message: first, .. })),
                    (_, Error::Pjrt(PjrtError::Aborted { message: second, .. })),
                ] if first == "second replica failed" && second == "second replica failed"),
        ));
    }

    #[test]
    fn test_array_placement_communicates_excludes_grouped_local_shard_moves() {
        let clients = [cpu_client(1), process_one_cpu_client(1)];
        let devices = clients.map(first_device);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.to_vec()).unwrap();
        let sharding = Sharding::replicated(logical_mesh.clone(), 1);
        let short_target = Placement::new(mesh.clone(), Sharding::replicated(logical_mesh, 0)).unwrap();
        for (rank, client) in clients.into_iter().enumerate() {
            // Explicit peers exercise admission without a native comparison transport. Both ranks expose the same
            // shard metadata and hold only their own replica.
            let session = Session::builder().client(client).remote_devices([devices[1 - rank]]).build().unwrap();
            let source = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::U8, [1]),
                mesh.clone(),
                Some(sharding.clone()),
                [7u8],
            )
            .unwrap();
            // Each process serves both targets from the replica on its own device, so neither placement
            // communicates, alone or grouped with others.
            let one_device = DeviceMesh::new(LogicalMesh::single_device(), vec![devices[0]]).unwrap();
            let one_device_sharding = Sharding::replicated(LogicalMesh::single_device(), 1);
            assert!(!source.placement_communicates(&one_device, &one_device_sharding));
            assert!(!source.placement_communicates(&mesh, &sharding));
            let assert_output = |output: &Array, expected_devices: &[Device], owns_shard: bool| {
                assert_eq!(output.mesh().devices(), expected_devices);
                output.block_until_ready().unwrap();
                assert_eq!(output.addressable_shards().count(), usize::from(owns_shard));
                if owns_shard {
                    assert_eq!(output.copy_to_host_bytes().unwrap(), [7]);
                }
            };
            let one = session.place(source.clone(), PlacementOptions::to(devices[0])).unwrap();
            assert_output(&one, &devices[..1], rank == 0);

            for (target, expected_devices, owns_shard) in [
                (PlacementRequest::from(devices[0]), &devices[..1], rank == 0),
                (PlacementRequest::from(short_target.clone()), &devices[..], true),
            ] {
                let pair = (source.clone(), source.clone());
                let (left, right) = session.place(pair, PlacementOptions::to(target.clone())).unwrap();
                assert_output(&left, expected_devices, owns_shard);
                assert_output(&right, expected_devices, owns_shard);

                let operation = PlaceOperation::new(Some(target), None);
                let outputs = ValueDomainDispatch::place_operations(vec![
                    (source.clone(), operation.clone()),
                    (source.clone(), operation),
                ])
                .unwrap();
                let [left, right] = outputs.as_slice() else { panic!("expected one output for each input") };
                assert_output(left, expected_devices, owns_shard);
                assert_output(right, expected_devices, owns_shard);
            }
        }
    }

    #[test]
    fn test_array_place_within_client_moves_pending_contributions_to_other_devices() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let on = |axis: &str, devices: Vec<Device>| {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(axis, 2, MeshAxisType::Explicit).unwrap()]);
            DeviceMesh::new(logical_mesh.unwrap(), devices).unwrap()
        };
        let original = on("x", devices.clone());
        let reversed = on("x", vec![devices[1], devices[0]]);
        let renamed = on("y", devices.clone());
        let auto_reversed = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_reversed = DeviceMesh::new(auto_reversed, vec![devices[1], devices[0]]).unwrap();
        let unit = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("u", 1, MeshAxisType::Explicit).unwrap(),
        ]);
        let unit = DeviceMesh::new(unit.unwrap(), devices.clone()).unwrap();
        let pending = |mesh: &DeviceMesh, kind: UnreducedKind| {
            let axis = mesh.logical_mesh().axes()[0].name().to_string();
            let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0).with_unreduced_axes([axis]).unwrap();
            sharding.with_unreduced_kind(kind).unwrap()
        };
        // Returns the bytes that each device holds, in the order of the device identifiers.
        let device_bytes = |array: &Array| {
            let mut bytes = array
                .shards()
                .iter()
                .map(|shard| (shard.device().id(), shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>();
            bytes.sort();
            bytes
        };
        // Returns the bytes that each device of `mesh` holds after a placement that moves the contribution at each
        // coordinate to the device at the same coordinate of `mesh`, in the order of the device identifiers.
        let moved_bytes = |mesh: &DeviceMesh, contributions: &[Vec<u8>; 2]| {
            let bytes = mesh.devices().iter().zip(contributions).map(|(device, bytes)| (device.id(), bytes.clone()));
            let mut bytes = bytes.collect::<Vec<_>>();
            bytes.sort();
            bytes
        };
        let import = |data_type: DataType, kind: UnreducedKind, contributions: &[Vec<u8>; 2]| {
            Array::from_host_shards(
                &session,
                ArrayType::scalar(data_type),
                original.clone(),
                Some(pending(&original, kind)),
                devices
                    .iter()
                    .zip(contributions)
                    .map(|(device, bytes)| (device.id(), Arc::<[u8]>::from(&bytes[..]))),
            )
            .unwrap()
        };

        // A placement that keeps the reduction state and only changes where each contribution lives moves every
        // contribution as it is, so the contribution at coordinate zero moves to the first device of the target
        // mesh, which keeps the sum of two smallest subnormal `f32` values, the sum of two negative zeros, the `u8`
        // summands `3` and `5`, and the `f8e8m0fnu` summands of a type without an additive identity. Adding an unused
        // axis of size one changes no contribution either.
        let contributions = [
            (DataType::F32, UnreducedKind::Sum, [vec![1u8, 0, 0, 0], vec![1, 0, 0, 0]]),
            (DataType::F32, UnreducedKind::Sum, [vec![0, 0, 0, 0x80], vec![0, 0, 0, 0x80]]),
            (DataType::U8, UnreducedKind::Sum, [vec![3], vec![5]]),
            (DataType::F8E8M0FNU, UnreducedKind::Sum, [vec![0x7f], vec![0x7f]]),
            (DataType::F32, UnreducedKind::Max, [values_to_bytes(&[1f32]), values_to_bytes(&[f32::NAN])]),
        ];
        for target_mesh in [&reversed, &unit] {
            for (data_type, kind, contributions) in &contributions {
                let source = import(*data_type, *kind, contributions);
                let value = source.copy_to_host_bytes().unwrap();
                let target = Placement::new(target_mesh.clone(), pending(target_mesh, *kind)).unwrap();
                for placed in place_through_entry_points(&session, &source, target) {
                    let placed = placed.unwrap();
                    let context = format!("{data_type} {kind} onto {target_mesh:?}");
                    assert_eq!(device_bytes(&placed), moved_bytes(target_mesh, contributions), "{context}");
                    assert_eq!(placed.copy_to_host_bytes().unwrap(), value, "{context}");
                }
            }
        }

        // An explicit axis onto an auto one changes no contribution either.
        let source = import(DataType::F32, UnreducedKind::Sum, &[vec![1, 0, 0, 0], vec![1, 0, 0, 0]]);
        let target = Placement::new(auto_reversed.clone(), pending(&auto_reversed, UnreducedKind::Sum)).unwrap();
        let placed = source.to_placement(&session, target).unwrap();
        assert_eq!(device_bytes(&placed), [(devices[0].id(), vec![1, 0, 0, 0]), (devices[1].id(), vec![1, 0, 0, 0])]);
        assert_eq!(placed.copy_to_host_bytes().unwrap(), vec![2, 0, 0, 0]);

        // Renaming the axis of a pending sum is a pure move too, which the device planner serves with whole shards,
        // although the type rule of `place` rejects it before any planning because it redefines an axis that carries
        // reduction state.
        let source = import(DataType::U8, UnreducedKind::Sum, &[vec![3], vec![5]]);
        let sharding = pending(&renamed, UnreducedKind::Sum);
        let layout_resolver = cross_host_layout_resolver(client);
        let strategy = source.device_placement_strategy(&layout_resolver, &renamed, &sharding, 0).unwrap();
        assert!(matches!(strategy, DevicePlacementStrategy::ExactShards { .. }));
        let placed = source.place_with_strategy(strategy, 0, &renamed, &sharding, Memory::Device, None, false);
        let placed = placed.map_err(|(error, _)| error).unwrap();
        assert_eq!(device_bytes(&placed), [(devices[0].id(), vec![3]), (devices[1].id(), vec![5])]);

        // Adding a pending axis of size one keeps the number of contributions, so it is a pure move as well, which
        // keeps the summands of every data type as they are. Only the physical placements reach it, because the type
        // rule of `place` rejects a change of reduction state.
        let pending_unit = Sharding::replicated(unit.logical_mesh().clone(), 0).with_unreduced_axes(["x", "u"]);
        let target = Placement::new(unit.clone(), pending_unit.unwrap()).unwrap();
        for (data_type, contributions) in [
            (DataType::U8, [vec![3u8], vec![5]]),
            (DataType::F32, [vec![1, 0, 0, 0], vec![1, 0, 0, 0]]),
            (DataType::F8E8M0FNU, [vec![0x7f], vec![0x7f]]),
        ] {
            let source = import(data_type, UnreducedKind::Sum, &contributions);
            let value = source.copy_to_host_bytes().unwrap();
            for placed in
                [source.to_placement(&session, target.clone()), source.clone().into_placement(&session, target.clone())]
            {
                let placed = placed.unwrap();
                assert_eq!(device_bytes(&placed), moved_bytes(&unit, &contributions), "{data_type}");
                assert_eq!(placed.copy_to_host_bytes().unwrap(), value, "{data_type}");
            }
        }
    }

    #[test]
    fn test_array_place_within_client_moves_pending_contributions_as_shard_regions() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let mesh = |axes: &[(&str, usize, MeshAxisType)]| {
            let axes = axes.iter().map(|(name, size, r#type)| MeshAxis::new(*name, *size, *r#type).unwrap());
            DeviceMesh::new(LogicalMesh::new(axes.collect()).unwrap(), devices.clone()).unwrap()
        };
        let explicit = MeshAxisType::Explicit;
        let source_mesh = mesh(&[("x", 2, explicit), ("y", 2, explicit)]);
        let sum = Sharding::replicated(source_mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        // Returns the bytes that each device holds, in the order of `devices`.
        let device_bytes = |array: &Array| {
            devices
                .iter()
                .map(|device| array.device_shard(device.id()).unwrap().copy_to_host_bytes().unwrap())
                .collect::<Vec<_>>()
        };

        // A vector of four elements with a pending sum along `x` is split along a second axis, which only moves each
        // of its two contributions as regions of their shards, also onto a mesh with an unused axis of size one or with
        // the partitioning axis renamed. It completes no sum, so each sum of two smallest subnormal `f32` values (i.e.,
        // the bits `0x00000002`) is kept, and `f8e8m0fnu`, which has no additive identity, moves as well. On the same
        // mesh, the placement moves regions as well, so that sparse and dense sources behave alike.
        let targets = [
            (mesh(&[("x", 2, explicit), ("y", 2, explicit)]), "y"),
            (mesh(&[("x", 2, explicit), ("y", 2, explicit), ("unit", 1, explicit)]), "y"),
            (mesh(&[("x", 2, explicit), ("z", 2, explicit)]), "z"),
        ];
        let cases = [
            (
                DataType::F32,
                [values_to_bytes(&[1u32, 2, 3, 4]), values_to_bytes(&[1u32; 4])],
                values_to_bytes(&[2u32, 3, 4, 5]),
            ),
            (DataType::F8E8M0FNU, [vec![0x7f; 4], vec![0x7f; 4]], vec![0x80; 4]),
            (DataType::U8, [vec![1, 2, 3, 4], vec![10, 20, 30, 40]], vec![11, 22, 33, 44]),
        ];
        for (target_mesh, axis) in &targets {
            let split = vec![ShardingDimension::sharded([*axis])];
            let split = Sharding::new(target_mesh.logical_mesh().clone(), split).unwrap();
            let split = split.with_unreduced_axes(["x"]).unwrap();
            for (data_type, contributions, value) in &cases {
                let source = Array::from_host_shards(
                    &session,
                    ArrayType::new_static(*data_type, [4]),
                    source_mesh.clone(),
                    Some(sum.clone()),
                    devices
                        .iter()
                        .enumerate()
                        .map(|(index, device)| (device.id(), Arc::<[u8]>::from(contributions[index / 2].as_slice()))),
                )
                .unwrap();
                assert_eq!(&source.copy_to_host_bytes().unwrap(), value, "{data_type}");
                let strategy = source
                    .device_placement_strategy(&cross_host_layout_resolver(client), target_mesh, &split, 0)
                    .unwrap();
                assert!(matches!(strategy, DevicePlacementStrategy::ShardRegions { .. }), "{data_type} {axis}");
                // The device at `(x, s)` holds the half `s` of the contribution at `x`.
                let halves = |x: usize| contributions[x].chunks(contributions[x].len() / 2).map(<[u8]>::to_vec);
                let expected = halves(0).chain(halves(1)).collect::<Vec<_>>();
                let target = Placement::new(target_mesh.clone(), split.clone()).unwrap();
                for placed in place_through_entry_points(&session, &source, target) {
                    let placed = placed.unwrap();
                    assert_eq!(device_bytes(&placed), expected, "{data_type} along {axis}");
                    assert_eq!(&placed.copy_to_host_bytes().unwrap(), value, "{data_type} along {axis}");
                }
            }
        }
        // A sparse source that holds only the replicas at `y = 1` of each contribution moves on the same mesh as on
        // any other, by regions, which keep each contribution as it is.
        let pjrt_devices = client.addressable_devices().unwrap();
        let split_y = Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["y"])]);
        let split_y = split_y.unwrap().with_unreduced_axes(["x"]).unwrap();
        for (data_type, contribution, value) in [(DataType::U8, 3u8, 6u8), (DataType::F8E8M0FNU, 0x7f, 0x80)] {
            let buffers = [1, 3]
                .into_iter()
                .map(|index| {
                    let buffer_type = data_type.executable_buffer_type();
                    let device = pjrt_devices[index].clone();
                    client.buffer(&[contribution; 2], buffer_type, [2], None, device, None).unwrap()
                })
                .collect::<Vec<_>>();
            let r#type = ArrayType::new_static(data_type, [2]);
            let source =
                Array::from_addressable_buffers(&session, r#type, source_mesh.clone(), Some(sum.clone()), buffers);
            let source = source.unwrap();
            let target = Placement::new(source_mesh.clone(), split_y.clone()).unwrap();
            let borrowed = source.to_placement(&session, target.clone());
            for placed in [borrowed, source.clone().into_placement(&session, target)] {
                let placed = placed.unwrap();
                assert_eq!(device_bytes(&placed), vec![vec![contribution]; 4], "{data_type}");
                assert_eq!(placed.copy_to_host_bytes().unwrap(), vec![value; 2], "{data_type}");
            }
        }

        // A vector of two elements split along `x` and varying along the manual axis `m` holds four distinct values,
        // which a placement that splits it along `m` instead cannot represent, so the placement is rejected rather than
        // keep only some of those values.
        let manual = MeshAxisType::Manual;
        let manual_mesh = mesh(&[("x", 2, explicit), ("m", 2, manual)]);
        let along = |axis: &str| {
            let sharding = Sharding::along(manual_mesh.logical_mesh().clone(), axis, Some(0), 1).unwrap();
            sharding.with_varying_manual_axes(["m"]).unwrap()
        };
        let source = Array::from_host_shards(
            &session,
            ArrayType::new_static(DataType::U8, [2]),
            manual_mesh.clone(),
            Some(along("x")),
            devices
                .iter()
                .zip([10u8, 11, 20, 21])
                .map(|(device, value)| (device.id(), Arc::<[u8]>::from([value]))),
        )
        .unwrap();
        let target = Placement::new(manual_mesh.clone(), along("m")).unwrap();
        for placed in [source.to_placement(&session, target.clone()), source.clone().into_placement(&session, target)] {
            assert_eq!(
                placed.unwrap_err().to_string(),
                "cannot place an array that varies along the manual mesh axes `m` onto this sharding, because the \
                 placement would discard values that differ along those axes",
            );
        }
    }

    #[test]
    fn test_array_place_within_client_moves_pending_contributions_in_declared_layouts() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let pjrt_devices = client.addressable_devices().unwrap();
        let devices = session.addressable_devices().unwrap();
        let mesh = |axes: &[(&str, usize)]| {
            let axes = axes.iter().map(|(name, size)| MeshAxis::new(*name, *size, MeshAxisType::Explicit).unwrap());
            DeviceMesh::new(LogicalMesh::new(axes.collect()).unwrap(), devices.clone()).unwrap()
        };
        let source_mesh = mesh(&[("x", 2), ("y", 2)]);
        let sum = Sharding::replicated(source_mesh.logical_mesh().clone(), 2).with_unreduced_axes(["x"]).unwrap();
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let device_bytes = |array: &Array| {
            devices
                .iter()
                .map(|device| array.device_shard(device.id()).unwrap().copy_to_host_bytes().unwrap())
                .collect::<Vec<_>>()
        };

        // A column-major `[rows, 2]` matrix with a pending sum along `x`, whose devices at `x = 0` hold the first
        // contribution and those at `x = 1` the second one, keeps that sum and splits its rows along a second axis, on
        // the same mesh, onto a mesh with an unused axis of size one, and onto a mesh with that axis renamed. Each
        // process converts the shards that its moves read to the dense row-major layout on their devices and moves the
        // regions of each contribution, from a dense source and from a sparse one that holds only the replicas at
        // `y = 1`, for an uneven split of three rows and an even one of four, which a compiled reshard cannot make.
        // Every contribution keeps its value (e.g., the `u8` contributions `3` and `5` of three rows stay apart rather
        // than read back as `8` and `0`), and `f8e8m0fnu`, which has no additive identity, moves as well.
        let targets = [
            (mesh(&[("x", 2), ("y", 2)]), "y"),
            (mesh(&[("x", 2), ("y", 2), ("unit", 1)]), "y"),
            (mesh(&[("x", 2), ("z", 2)]), "z"),
        ];
        // The elements of the contribution at `x` of `rows` rows, in row-major order, and the value of both.
        let u8_contribution = |x: usize, rows: usize| {
            (0..rows * 2)
                .map(|index| [3u8, 5][x] + 10 * (index / 2) as u8 + (index % 2) as u8)
                .collect::<Vec<_>>()
        };
        let u8_value = |rows: usize| {
            u8_contribution(0, rows)
                .iter()
                .zip(u8_contribution(1, rows))
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>()
        };
        // The device at `(x, s)` holds the rows of part `s` of the contribution at `x`.
        let parts = |contribution: &dyn Fn(usize, usize) -> Vec<u8>, rows: usize| {
            let upper = rows.div_ceil(2);
            (0..4)
                .map(|index| {
                    let range = if index % 2 == 0 { 0..upper * 2 } else { upper * 2..rows * 2 };
                    contribution(index / 2, rows)[range].to_vec()
                })
                .collect::<Vec<_>>()
        };
        type Elements<'a> = &'a dyn Fn(usize, usize) -> Vec<u8>;
        let cases: [(DataType, Elements<'_>, &dyn Fn(usize) -> Vec<u8>); 2] = [
            (DataType::U8, &u8_contribution, &u8_value),
            (DataType::F8E8M0FNU, &|_, rows| vec![0x7fu8; rows * 2], &|rows| vec![0x80u8; rows * 2]),
        ];
        for rows in [3, 4] {
            for &(data_type, contribution, value) in &cases {
                let value = value(rows);
                for sparse in [false, true] {
                    let device_layout = column_major.to_pjrt().unwrap();
                    let buffers = pjrt_devices
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| !sparse || index % 2 == 1)
                        .map(|(index, device)| {
                            let bytes = contribution(index / 2, rows);
                            let buffer_type = data_type.executable_buffer_type();
                            let (device, layout) = (device.clone(), Some(device_layout.clone()));
                            let dimensions = [rows as u64, 2];
                            client.buffer(bytes.as_slice(), buffer_type, dimensions, None, device, layout).unwrap()
                        })
                        .collect::<Vec<_>>();
                    let r#type = ArrayType::new_static(data_type, [rows, 2]).with_layout(Some(column_major.clone()));
                    let (mesh, sharding) = (source_mesh.clone(), Some(sum.clone()));
                    let source = Array::from_addressable_buffers(&session, r#type, mesh, sharding, buffers).unwrap();
                    assert_eq!(source.copy_to_host_bytes().unwrap(), value, "{data_type} {rows} {sparse}");
                    for (target_mesh, axis) in &targets {
                        let split = vec![ShardingDimension::sharded([*axis]), ShardingDimension::replicated()];
                        let split = Sharding::new(target_mesh.logical_mesh().clone(), split).unwrap();
                        let split = split.with_unreduced_axes(["x"]).unwrap();
                        let context = format!("{data_type} with {rows} rows along {axis} (sparse: {sparse})");
                        let strategy = source
                            .device_placement_strategy(&cross_host_layout_resolver(client), target_mesh, &split, 0)
                            .unwrap();
                        assert!(matches!(strategy, DevicePlacementStrategy::ShardRegions { .. }), "{context}");
                        let target = Placement::new(target_mesh.clone(), split.clone()).unwrap();
                        for placed in place_through_entry_points(&session, &source, target) {
                            let placed = placed.unwrap();
                            assert_eq!(placed.layout(), Some(&column_major), "{context}");
                            assert_eq!(device_bytes(&placed), parts(contribution, rows), "{context}");
                            assert_eq!(placed.copy_to_host_bytes().unwrap(), value, "{context}");
                        }
                    }
                }
            }
        }

        // The same `u8` source in pinned host memory converts its shards into device memory, where the CPU plugin can
        // hold the dense row-major layout, and the result takes the requested memory and the column-major layout.
        let pinned = Memory::Host { pinned: true };
        let split = vec![ShardingDimension::sharded(["y"]), ShardingDimension::replicated()];
        let split = Sharding::new(source_mesh.logical_mesh().clone(), split).unwrap().with_unreduced_axes(["x"]);
        let split = split.unwrap();
        for rows in [3, 4] {
            let source = Array::from_host_shards(
                &session,
                ArrayType::new_static(DataType::U8, [rows, 2]),
                source_mesh.clone(),
                Some(sum.clone()),
                devices.iter().enumerate().map(|(index, device)| {
                    (device.id(), Arc::<[u8]>::from(u8_contribution(index / 2, rows).as_slice()))
                }),
            )
            .unwrap();
            let target = Placement::new(source_mesh.clone(), sum.clone()).unwrap();
            let target = target.with_memory(pinned).with_layout(column_major.clone());
            let source = source.to_placement(&session, target).unwrap();
            assert_eq!((source.r#type().memory(), source.layout()), (pinned, Some(&column_major)));
            let strategy = source
                .device_placement_strategy(&cross_host_layout_resolver(client), &source_mesh, &split, 0)
                .unwrap();
            assert!(matches!(strategy, DevicePlacementStrategy::ShardRegions { .. }), "{rows}");
            for memory in [pinned, Memory::Device] {
                let target = Placement::new(source_mesh.clone(), split.clone()).unwrap().with_memory(memory);
                for placed in place_through_entry_points(&session, &source, target) {
                    let placed = placed.unwrap();
                    let context = format!("{rows} rows into {memory:?}");
                    assert_eq!((placed.r#type().memory(), placed.layout()), (memory, Some(&column_major)), "{context}");
                    assert_eq!(device_bytes(&placed), parts(&u8_contribution, rows), "{context}");
                    assert_eq!(placed.copy_to_host_bytes().unwrap(), u8_value(rows), "{context}");
                }
            }
        }
    }

    #[test]
    fn test_array_place_within_client_moves_column_major_contributions_across_processes_as_shard_regions() {
        // A column-major `u8[2, 2]` with a pending sum along `x` moves its contributions as regions onto a mesh that
        // pairs each local device with a remote one, so process 0 sends regions of its contribution, receives regions
        // of the other one, and copies regions between its own devices. Both processes plan the same transfers, and
        // process 0, which converts the shards that its sends read to the dense row-major layout first, reaches its
        // transfers, which fail here with the missing cross-host transfers extension of the CPU plugin.
        let (session, target_mesh) = two_process_mesh_2x2();
        let client = session.client();
        let pjrt_devices = client.addressable_devices().unwrap();
        let logical_mesh = target_mesh.logical_mesh().clone();
        let devices = target_mesh.devices();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0], devices[2], devices[1], devices[3]]);
        let source_mesh = source_mesh.unwrap();
        let sum = Sharding::replicated(logical_mesh.clone(), 2).with_unreduced_axes(["x"]).unwrap();
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let r#type = ArrayType::new_static(DataType::U8, [2, 2]).with_layout(Some(column_major.clone()));
        let buffers = pjrt_devices
            .iter()
            .map(|device| {
                let layout = Some(column_major.to_pjrt().unwrap());
                client.buffer([1u8, 2, 3, 4].as_slice(), BufferType::U8, [2u64, 2], None, device.clone(), layout)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let (mesh, sharding) = (source_mesh.clone(), Some(sum.clone()));
        let own = Array::from_addressable_buffers(&session, r#type.clone(), mesh, sharding.clone(), buffers).unwrap();
        let peer = Array::from_addressable_buffers(&session, r#type, source_mesh, sharding, Vec::new()).unwrap();
        let split = vec![ShardingDimension::sharded(["y"]), ShardingDimension::replicated()];
        let split = Sharding::new(logical_mesh, split).unwrap().with_unreduced_axes(["x"]).unwrap();
        assert_eq!(placement_agreement(&[own, peer], &target_mesh, &split), Ok("shard regions".to_string()));
    }

    #[test]
    fn test_array_place_within_client_creates_empty_exact_shards_on_every_process() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let on = |axis: &str, devices: Vec<Device>| {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(axis, 2, MeshAxisType::Explicit).unwrap()]);
            DeviceMesh::new(logical_mesh.unwrap(), devices).unwrap()
        };
        let split = |mesh: &DeviceMesh| {
            let axis = mesh.logical_mesh().axes()[0].name().to_string();
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded([axis])]).unwrap()
        };

        // The `u8[1]` split over two devices has an empty slice on the second device, which this import holds no
        // buffer for. A renamed axis keeps every slice where it is, and each process creates its own empty shards
        // rather than requiring a buffer to move, through every entry point.
        let original = on("x", devices.clone());
        let renamed = on("y", devices.clone());
        let client_devices = client.addressable_devices().unwrap();
        let buffer = client.buffer([42u8].as_slice(), BufferType::U8, [1u64], None, client_devices[0].clone(), None);
        let r#type = ArrayType::new_static(DataType::U8, [1]);
        let sharding = Some(split(&original));
        let source =
            Array::from_addressable_buffers(&session, r#type.clone(), original, sharding, vec![buffer.unwrap()]);
        let source = source.unwrap();
        assert!(source.device_shard(devices[1].id()).unwrap().buffer().is_none());
        let target = Placement::new(renamed.clone(), split(&renamed)).unwrap();
        for placed in place_through_entry_points(&session, &source, target) {
            let placed = placed.unwrap();
            let bytes = placed.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();
            assert_eq!(bytes, [vec![42], vec![]]);
            assert_eq!(placed.block_until_ready().map_err(|error| error.to_string()), Ok(()));
            assert_eq!(placed.copy_to_host_bytes().unwrap(), [42]);
        }

        // Across processes, the empty slice moves nowhere, so neither process waits for a transfer of it, and only the
        // slice with elements crosses from process 0 to process 1.
        let remote = Device::new(devices.iter().map(Device::id).max().unwrap() + 1, 1);
        let session = session_with_remote_devices(client, [remote]);
        let both = on("x", vec![devices[0], remote]);
        let reversed = on("x", vec![remote, devices[0]]);
        let view = Array::from_addressable_buffers(&session, r#type, both.clone(), Some(split(&both)), Vec::new());
        let strategy =
            view.unwrap()
                .device_placement_strategy(&simulated_cross_host_layout, &reversed, &split(&reversed), 0);
        let Ok(DevicePlacementStrategy::ExactShards { plan, .. }) = strategy else {
            panic!("the slices of the split must move as whole shards");
        };
        let moves = plan
            .moves
            .iter()
            .map(|movement| (movement.source_device, movement.target_device, movement.slice.clone()))
            .collect::<Vec<_>>();
        assert_eq!(moves, [(devices[0], remote, vec![0..1])]);
    }

    #[test]
    fn test_array_place_within_client_preserves_value_across_reduction_state_changes() {
        // Placements that complete, introduce, keep, or change the kind of a pending reduction along the axis `x` of a
        // 2x2 mesh, for a data type of every class of `validate_reduction_transition`, compared with the host readback
        // of their source, which defines the value (refer to `crate::arrays::host`). The contributions along `x` are
        // every pair of some elements of each data type (e.g., negative zeros, NaN values, infinities, and extreme
        // integers, but no subnormal `f32` values, which the CPU plugin flushes in any addition). Each row lists, per
        // placement, the programs that `to_placement` and `into_placement` compile in sessions of their own, which is
        // `+0` for a host round trip, and asserts that both read back the value of their source bit for bit, where
        // every NaN encoding of a format counts as one value, or lists their common error. A sum of floating point or
        // complex values that the devices complete is compared as the contract of such sums states (refer to
        // `crate::arrays::host`), so a negative zero also matches a positive one there, and the end of this test pins
        // that difference. A sum that a placement introduces is compared bit for bit for every data type with an
        // additive identity (e.g., `f32` and `c64` negative zeros and `f6e2m3fn` values), which it writes without a
        // compiled program. The axis type does not change the outcome, so both axis types must give the same row.
        let client = cpu_client(4);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let f32_elements = [0f32, -0., 1., -1., f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-3, 3., f32::MAX];
        let c64_elements = [(0f32, 0f32), (-0., -0.), (1., -1.), (f32::NAN, 0.), (f32::INFINITY, 1.), (-1., 2.)];
        let elements = |data_type: DataType| -> Vec<Vec<u8>> {
            let bytes = |values: &[u8]| values.iter().map(|value| vec![*value]).collect();
            match data_type {
                DataType::Boolean | DataType::U1 | DataType::I1 => bytes(&[0, 1]),
                DataType::U2 => bytes(&[0, 1, 2, 3]),
                DataType::I4 => bytes(&(0..16).collect::<Vec<_>>()),
                DataType::U8 => bytes(&[0, 1, 127, 128, 254, 255]),
                DataType::I32 => {
                    [0i32, 1, -1, i32::MAX, i32::MIN, 2].map(|value| value.to_ne_bytes().to_vec()).to_vec()
                }
                DataType::F32 => f32_elements.map(|value| value.to_ne_bytes().to_vec()).to_vec(),
                DataType::F8E5M2 => bytes(&[0x00, 0x80, 0x3c, 0xbc, 0x7c, 0xfc, 0x7e, 0x01, 0x7b, 0xfb]),
                DataType::F8E4M3FNUZ => bytes(&[0x00, 0x80, 0x01, 0x40, 0xc0, 0x7f, 0xff, 0x38]),
                DataType::F6E2M3FN => bytes(&[0x00, 0x20, 0x01, 0x08, 0x2a, 0x1f, 0x3f, 0x10]),
                DataType::F8E8M0FNU => bytes(&[0, 1, 42, 127, 128, 255]),
                DataType::C64 => c64_elements
                    .map(|(real, imaginary)| [real.to_ne_bytes(), imaginary.to_ne_bytes()].concat())
                    .to_vec(),
                _ => unreachable!(),
            }
        };
        // Replaces every NaN encoding of a format with one marker, so that NaN payloads do not count, and with
        // `signless_zero`, also a negative zero with a positive one.
        let canonical = |data_type: DataType, element: &[u8], signless_zero: bool| -> Vec<u8> {
            let f32_value = |value: f32| match value {
                value if value.is_nan() => vec![0xff; 5],
                value if signless_zero && value == 0. => 0f32.to_ne_bytes().to_vec(),
                value => value.to_ne_bytes().to_vec(),
            };
            match data_type {
                DataType::F32 => f32_value(f32::from_ne_bytes(element.try_into().unwrap())),
                DataType::C64 => element
                    .chunks(4)
                    .flat_map(|bytes| f32_value(f32::from_ne_bytes(bytes.try_into().unwrap())))
                    .collect(),
                DataType::F8E5M2 if element[0] & 0x7c == 0x7c && element[0] & 0x03 != 0 => vec![0xff; 2],
                DataType::F8E5M2 if signless_zero && element[0] == 0x80 => vec![0x00],
                DataType::F8E4M3FNUZ if element[0] == 0x80 => vec![0xff; 2],
                DataType::F8E8M0FNU if element[0] == 0xff => vec![0xff; 2],
                _ => element.to_vec(),
            }
        };
        let data_types = [
            DataType::Boolean,
            DataType::U1,
            DataType::I1,
            DataType::U2,
            DataType::I4,
            DataType::U8,
            DataType::I32,
            DataType::F32,
            DataType::F8E5M2,
            DataType::F8E4M3FNUZ,
            DataType::F6E2M3FN,
            DataType::F8E8M0FNU,
            DataType::C64,
        ];
        let mut rows = Vec::new();
        for data_type in data_types {
            let elements = elements(data_type);
            let count = elements.len() * elements.len();
            let element_size = elements[0].len();
            // The devices at `x = 0` hold the first element of each pair and the devices at `x = 1` the second one.
            let first = (0..count).flat_map(|index| elements[index / elements.len()].clone()).collect::<Vec<_>>();
            let second = (0..count).flat_map(|index| elements[index % elements.len()].clone()).collect::<Vec<_>>();
            let mut outcomes_by_axis_type = Vec::new();
            for axis_type in [MeshAxisType::Auto, MeshAxisType::Explicit] {
                let logical_mesh = LogicalMesh::new(vec![
                    MeshAxis::new("x", 2, axis_type).unwrap(),
                    MeshAxis::new("y", 2, axis_type).unwrap(),
                ])
                .unwrap();
                let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
                let replicated = Sharding::replicated(logical_mesh.clone(), 1);
                let split_y = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["y"])]).unwrap();
                let pending = |sharding: &Sharding, kind: UnreducedKind| {
                    sharding.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap()
                };
                let placements = [
                    ("sum -> none", Some(UnreducedKind::Sum), replicated.clone()),
                    ("max -> none", Some(UnreducedKind::Max), replicated.clone()),
                    ("min -> none", Some(UnreducedKind::Min), replicated.clone()),
                    ("none -> sum", None, pending(&replicated, UnreducedKind::Sum)),
                    ("none -> max", None, pending(&replicated, UnreducedKind::Max)),
                    ("max -> sum", Some(UnreducedKind::Max), pending(&replicated, UnreducedKind::Sum)),
                    ("sum -> sum split", Some(UnreducedKind::Sum), pending(&split_y, UnreducedKind::Sum)),
                    ("max -> max split", Some(UnreducedKind::Max), pending(&split_y, UnreducedKind::Max)),
                ];
                let mut outcomes = Vec::new();
                for (name, source_kind, target_sharding) in placements {
                    let mut results = Vec::new();
                    for consume in [false, true] {
                        let session = Session::builder().client(client).build().unwrap();
                        let r#type = ArrayType::new_static(data_type, [count]);
                        let source = match source_kind {
                            Some(kind) => {
                                let contributions = devices
                                    .iter()
                                    .zip([&first, &first, &second, &second])
                                    .map(|(device, bytes)| (device.id(), Arc::<[u8]>::from(bytes.as_slice())));
                                let sharding = Some(pending(&replicated, kind));
                                Array::from_host_shards(&session, r#type, mesh.clone(), sharding, contributions)
                            }
                            None => Array::from_host_buffer(&session, r#type, mesh.clone(), None, &first),
                        }
                        .unwrap();
                        let expected = source.copy_to_host_bytes().unwrap();
                        let target = Placement::new(mesh.clone(), target_sharding.clone()).unwrap();
                        let placed = if consume {
                            source.into_placement(&session, target)
                        } else {
                            source.to_placement(&session, target)
                        };
                        let programs = session.cached_program_count();
                        // A sum that the devices complete follows the contract of floating point sums, and one that
                        // they introduce is exact.
                        let sum_contract = programs > 0 && name == "sum -> none";
                        results.push(placed.and_then(|placed| {
                            assert_eq!(placed.sharding(), &target_sharding, "{data_type} {name}");
                            let actual = placed.copy_to_host_bytes()?;
                            let differing = expected
                                .chunks(element_size)
                                .zip(actual.chunks(element_size))
                                .filter(|(expected, actual)| {
                                    canonical(data_type, expected, sum_contract)
                                        != canonical(data_type, actual, sum_contract)
                                })
                                .count();
                            assert_eq!(differing, 0, "{axis_type:?} {data_type} {name}: {expected:?} != {actual:?}");
                            Ok(programs)
                        }));
                    }
                    let outcome = match results.as_slice() {
                        [Ok(borrowed), Ok(consumed)] => format!("+{borrowed} +{consumed}"),
                        [Err(borrowed), Err(consumed)] if borrowed.to_string() == consumed.to_string() => {
                            format!("error: {borrowed}")
                        }
                        results => panic!("{data_type} {name}: the entry points disagree: {results:?}"),
                    };
                    outcomes.push(format!("{data_type} {name}: {outcome}"));
                }
                outcomes_by_axis_type.push(outcomes);
            }
            assert_eq!(outcomes_by_axis_type[0], outcomes_by_axis_type[1], "{data_type}");
            rows.extend(outcomes_by_axis_type.swap_remove(0));
        }
        let missing_identity = "type `f8e8m0fnu` has no additive identity to split a value into summands along \
                                unreduced axes";
        assert_eq!(
            rows.join("\n"),
            formatdoc! {"
                bool sum -> none: +1 +1
                bool max -> none: +1 +1
                bool min -> none: +1 +1
                bool none -> sum: +0 +0
                bool none -> max: +0 +0
                bool max -> sum: +1 +1
                bool sum -> sum split: +4 +4
                bool max -> max split: +4 +4
                u1 sum -> none: +0 +0
                u1 max -> none: +1 +1
                u1 min -> none: +1 +1
                u1 none -> sum: +0 +0
                u1 none -> max: +0 +0
                u1 max -> sum: +1 +1
                u1 sum -> sum split: +4 +4
                u1 max -> max split: +4 +4
                i1 sum -> none: +0 +0
                i1 max -> none: +0 +0
                i1 min -> none: +0 +0
                i1 none -> sum: +0 +0
                i1 none -> max: +0 +0
                i1 max -> sum: +0 +0
                i1 sum -> sum split: +4 +4
                i1 max -> max split: +4 +4
                u2 sum -> none: +0 +0
                u2 max -> none: +0 +0
                u2 min -> none: +0 +0
                u2 none -> sum: +0 +0
                u2 none -> max: +0 +0
                u2 max -> sum: +0 +0
                u2 sum -> sum split: +4 +4
                u2 max -> max split: +4 +4
                i4 sum -> none: +1 +1
                i4 max -> none: +1 +1
                i4 min -> none: +1 +1
                i4 none -> sum: +0 +0
                i4 none -> max: +0 +0
                i4 max -> sum: +1 +1
                i4 sum -> sum split: +4 +4
                i4 max -> max split: +4 +4
                u8 sum -> none: +1 +1
                u8 max -> none: +1 +1
                u8 min -> none: +1 +1
                u8 none -> sum: +0 +0
                u8 none -> max: +0 +0
                u8 max -> sum: +1 +1
                u8 sum -> sum split: +4 +4
                u8 max -> max split: +4 +4
                i32 sum -> none: +1 +1
                i32 max -> none: +1 +1
                i32 min -> none: +1 +1
                i32 none -> sum: +0 +0
                i32 none -> max: +0 +0
                i32 max -> sum: +1 +1
                i32 sum -> sum split: +4 +4
                i32 max -> max split: +4 +4
                f32 sum -> none: +1 +1
                f32 max -> none: +0 +0
                f32 min -> none: +0 +0
                f32 none -> sum: +0 +0
                f32 none -> max: +0 +0
                f32 max -> sum: +0 +0
                f32 sum -> sum split: +4 +4
                f32 max -> max split: +4 +4
                f8e5m2 sum -> none: +1 +1
                f8e5m2 max -> none: +0 +0
                f8e5m2 min -> none: +0 +0
                f8e5m2 none -> sum: +0 +0
                f8e5m2 none -> max: +0 +0
                f8e5m2 max -> sum: +0 +0
                f8e5m2 sum -> sum split: +4 +4
                f8e5m2 max -> max split: +4 +4
                f8e4m3fnuz sum -> none: +1 +1
                f8e4m3fnuz max -> none: +0 +0
                f8e4m3fnuz min -> none: +0 +0
                f8e4m3fnuz none -> sum: +0 +0
                f8e4m3fnuz none -> max: +0 +0
                f8e4m3fnuz max -> sum: +0 +0
                f8e4m3fnuz sum -> sum split: +4 +4
                f8e4m3fnuz max -> max split: +4 +4
                f6e2m3fn sum -> none: +0 +0
                f6e2m3fn max -> none: +0 +0
                f6e2m3fn min -> none: +0 +0
                f6e2m3fn none -> sum: +0 +0
                f6e2m3fn none -> max: +0 +0
                f6e2m3fn max -> sum: +0 +0
                f6e2m3fn sum -> sum split: +4 +4
                f6e2m3fn max -> max split: +4 +4
                f8e8m0fnu sum -> none: +0 +0
                f8e8m0fnu max -> none: +0 +0
                f8e8m0fnu min -> none: +0 +0
                f8e8m0fnu none -> sum: error: {missing_identity}
                f8e8m0fnu none -> max: +0 +0
                f8e8m0fnu max -> sum: error: {missing_identity}
                f8e8m0fnu sum -> sum split: +4 +4
                f8e8m0fnu max -> max split: +4 +4
                c64 sum -> none: +1 +1
                c64 max -> none: +0 +0
                c64 min -> none: +0 +0
                c64 none -> sum: +0 +0
                c64 none -> max: +0 +0
                c64 max -> sum: +0 +0
                c64 sum -> sum split: +4 +4
                c64 max -> max split: +4 +4"},
        );

        // The two differences that the contract of floating point sums allows, pinned on the CPU plugin with the bit
        // patterns of `f32` values. Completing a sum of two negative zeros on the devices gives positive zero, because
        // the collective starts from positive zero, and the plugin flushes a subnormal sum to zero, as it does in every
        // addition. Introducing a sum compiles nothing and writes negative zero, the additive identity, to the other
        // device, so a negative zero reads back as it was.
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices[..2].to_vec()).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [3]);
        let bits = |array: &Array| {
            let values = values_from_bytes::<f32>(&array.copy_to_host_bytes().unwrap());
            values.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        };
        let contributions = [[-0f32, 1e-45, 1.], [-0., 0., 2.]]
            .iter()
            .zip(&devices)
            .map(|(values, device)| (device.id(), Arc::<[u8]>::from(values_to_bytes(values))))
            .collect::<Vec<_>>();
        let sharding = Some(unreduced.clone());
        let summands =
            Array::from_host_shards(&session, r#type.clone(), mesh.clone(), sharding, contributions).unwrap();
        let target = Placement::new(mesh.clone(), replicated.clone()).unwrap();
        let completed = summands.to_placement(&session, target).unwrap();
        let bytes = values_to_bytes(&[-0f32, 1e-45, 1.]);
        let value = Array::from_host_buffer(&session, r#type, mesh.clone(), Some(replicated), bytes).unwrap();
        let introduced = value.to_placement(&session, Placement::new(mesh, unreduced).unwrap()).unwrap();
        assert_eq!(session.cached_program_count(), 1);
        assert_eq!(
            [bits(&summands), bits(&completed), bits(&value), bits(&introduced)],
            [
                vec![0x8000_0000, 1, 0x4040_0000],
                vec![0, 0, 0x4040_0000],
                vec![0x8000_0000, 1, 0x3f80_0000],
                vec![0x8000_0000, 1, 0x3f80_0000],
            ],
        );
    }

    #[test]
    fn test_array_place_within_client_keeps_pending_axes_that_both_meshes_share() {
        // A placement that changes part of the reduction state keeps every pending axis that both ends have with the
        // same name, size, and kind, whatever the other axes of either mesh, and completes or introduces only the
        // others (refer to `kept_unreduced_axes`). Each placement goes from its source mesh to the same mesh, to the
        // same devices with an unused axis `u` of size one, and to the same devices in reverse order, through both
        // entry points, which must agree, and the three targets must give the same row, which lists the elements
        // of every target shard in mesh order and the readback, which must equal that of the source. Minimum
        // subnormal `f32` contributions along `x` with a sum introduced along `y` stay as they are (a completion would
        // flush them on the CPU plugin). Integer contributions along `x` and `y` of which only `y` completes keep `7`
        // and `11` along `x`. The contributions of `f8e8m0fnu`, which has no additive identity, take a sum introduced
        // along an axis `y` of size one, which writes no identity, but not along one of size two.
        let client = cpu_client(4);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = |axes: &[(&str, usize)], devices: Vec<Device>| {
            let axes = axes.iter().map(|&(name, size)| MeshAxis::new(name, size, MeshAxisType::Auto).unwrap());
            DeviceMesh::new(LogicalMesh::new(axes.collect()).unwrap(), devices).unwrap()
        };
        let pending = |mesh: &DeviceMesh, axes: &[&str]| {
            Sharding::replicated(mesh.logical_mesh().clone(), 1)
                .with_unreduced_axes(axes.iter().copied())
                .unwrap()
        };
        let render = |data_type: DataType, bytes: &[u8]| match data_type {
            DataType::F32 => format!("{:#010x}", u32::from_ne_bytes(bytes.try_into().unwrap())),
            DataType::I32 => i32::from_ne_bytes(bytes.try_into().unwrap()).to_string(),
            _ => format!("{:#04x}", bytes[0]),
        };
        let (size_one, size_two) = (vec![("x", 2), ("y", 1)], vec![("x", 2), ("y", 2)]);
        // Each case is its name, data type, source mesh axes, source and target unreduced axes, and the bytes of
        // the contributions of the devices of the source mesh in mesh order.
        let (x, x_y) = (vec!["x"], vec!["x", "y"]);
        let cases = [
            ("f32 sum(x) -> sum(x, y)", DataType::F32, &size_two, &x, &x_y, vec![1u32.to_ne_bytes().to_vec(); 4]),
            (
                "i32 sum(x, y) -> sum(x)",
                DataType::I32,
                &size_two,
                &x_y,
                &x,
                [3i32, 4, 5, 6].map(|value| value.to_ne_bytes().to_vec()).to_vec(),
            ),
            ("f8e8m0fnu sum(x) -> sum(x, y=1)", DataType::F8E8M0FNU, &size_one, &x, &x_y, vec![vec![0x7f]; 2]),
            ("f8e8m0fnu sum(x) -> sum(x, y=2)", DataType::F8E8M0FNU, &size_two, &x, &x_y, vec![vec![0x7f]; 4]),
        ];
        let mut rows = Vec::new();
        for (name, data_type, axes, source_axes, target_axes, contributions) in cases {
            let devices = devices[..contributions.len()].to_vec();
            let source_mesh = mesh(axes, devices.clone());
            let with_unused = axes.iter().copied().chain([("u", 1)]).collect::<Vec<_>>();
            let targets = [
                ("the same mesh", source_mesh.clone()),
                ("an unused axis", mesh(&with_unused, devices.clone())),
                ("reversed devices", mesh(axes, devices.iter().rev().copied().collect())),
            ];
            let mut rows_by_target = Vec::new();
            for (target_name, target_mesh) in targets {
                let target_sharding = pending(&target_mesh, target_axes);
                let mut results = Vec::new();
                for consume in [false, true] {
                    let session = Session::builder().client(client).build().unwrap();
                    let shards = devices
                        .iter()
                        .zip(&contributions)
                        .map(|(device, bytes)| (device.id(), Arc::<[u8]>::from(bytes.as_slice())));
                    let r#type = ArrayType::new_static(data_type, [1]);
                    let sharding = Some(pending(&source_mesh, source_axes));
                    let source =
                        Array::from_host_shards(&session, r#type, source_mesh.clone(), sharding, shards).unwrap();
                    let expected = source.copy_to_host_bytes().unwrap();
                    let target = Placement::new(target_mesh.clone(), target_sharding.clone()).unwrap();
                    let placed = if consume {
                        source.into_placement(&session, target)
                    } else {
                        source.to_placement(&session, target)
                    };
                    results.push(match placed {
                        Ok(placed) => {
                            assert_eq!(placed.sharding(), &target_sharding, "{name} onto {target_name}");
                            let readback = placed.copy_to_host_bytes().unwrap();
                            assert_eq!(readback, expected, "{name} onto {target_name}");
                            let shards = placed
                                .shards()
                                .iter()
                                .map(|shard| render(data_type, &shard.copy_to_host_bytes().unwrap()))
                                .collect::<Vec<_>>();
                            format!("{} = {}", shards.join(" "), render(data_type, &readback))
                        }
                        Err(error) => format!("error: {error}"),
                    });
                }
                assert_eq!(results[0], results[1], "{name} onto {target_name}");
                rows_by_target.push(results.swap_remove(0));
            }
            assert_eq!(rows_by_target[1..], [rows_by_target[0].clone(), rows_by_target[0].clone()]);
            rows.push(format!("{name}: {}", rows_by_target[0]));
        }
        let missing_identity = "type `f8e8m0fnu` has no additive identity to split a value into summands along \
                                unreduced axes";
        assert_eq!(
            rows.join("\n"),
            formatdoc! {"
                f32 sum(x) -> sum(x, y): 0x00000001 0x80000000 0x00000001 0x80000000 = 0x00000002
                i32 sum(x, y) -> sum(x): 7 7 11 11 = 18
                f8e8m0fnu sum(x) -> sum(x, y=1): 0x7f 0x7f = 0x80
                f8e8m0fnu sum(x) -> sum(x, y=2): error: {missing_identity}"},
        );
    }

    #[test]
    fn test_array_place_within_client_serves_missing_replicas_from_the_same_contribution() {
        // Sources on a 2x2 mesh of four CPU devices imported with buffers for only some of their replicas (`held` lists
        // the devices in mesh order), placed with `to_placement` and `into_placement`, which must agree, with the bytes
        // that each device holds afterwards. A placement serves a missing replica only from a replica of the same
        // contribution (`ShardContribution`), so each sparse import gives the bytes of the full one or an error:
        //   - A scalar that varies along the manual axis `m` (`11` at `m = 0` and `22` at `m = 1`) and introduces a
        //     pending sum along `x` from devices 0 and 3 keeps both values (`[11, 22, 0, 0]`), because each is served
        //     only from a replica of its own contribution.
        //   - A replicated `[7, 8]` held only on devices 1 and 3 that a placement splits along `x` while introducing
        //     a pending sum along `y` serves the devices at `y = 0` from those replicas, also when they are missing.
        //   - A vector split along `m` that varies along it varies only between tiles, so moving it to a mesh that
        //     renames `x` moves whole shards from any replica of each tile.
        //   - A scalar that varies within a tile keeps its variation on the renamed mesh, so that move only moves its
        //     contributions, also as whole shards.
        //   - A completion on a mesh with a manual axis has no device strategy and plans a host round trip that no
        //     process owns (its value has no one host readback), so it is rejected alike for every import.
        let client = cpu_client(4);
        let devices = client.addressable_devices().unwrap();
        let mesh_devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let mesh = |names: [(&str, MeshAxisType); 2]| {
            let axes = names.iter().map(|(name, r#type)| MeshAxis::new(*name, 2, *r#type).unwrap()).collect();
            DeviceMesh::new(LogicalMesh::new(axes).unwrap(), mesh_devices.clone()).unwrap()
        };
        let manual = mesh([("x", MeshAxisType::Explicit), ("m", MeshAxisType::Manual)]);
        let renamed = mesh([("z", MeshAxisType::Explicit), ("m", MeshAxisType::Manual)]);
        let explicit = mesh([("x", MeshAxisType::Explicit), ("y", MeshAxisType::Explicit)]);
        let scalar = ArrayType::scalar(DataType::U8);
        let vector = ArrayType::new_static(DataType::U8, [2]);
        let varying = |mesh: &DeviceMesh, rank| {
            Sharding::replicated(mesh.logical_mesh().clone(), rank).with_varying_manual_axes(["m"]).unwrap()
        };
        let split = |mesh: &DeviceMesh, axis: &str| {
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded([axis])]).unwrap()
        };
        let varying_tiles = |mesh: &DeviceMesh| split(mesh, "m").with_varying_manual_axes(["m"]).unwrap();
        // Each row has a source type, mesh, sharding, and the bytes of the device at each index, a target, and
        // the sets of devices that hold buffers, the first of which holds all of them.
        let rows = [
            (
                "varying along m -> sum along x",
                (&scalar, &manual, varying(&manual, 0), [vec![11], vec![22], vec![11], vec![22]]),
                (&manual, varying(&manual, 0).with_unreduced_axes(["x"]).unwrap()),
                vec![vec![0, 1, 2, 3], vec![0, 3], vec![1, 2], vec![0]],
            ),
            (
                "varying along m -> renamed mesh",
                (&scalar, &manual, varying(&manual, 0), [vec![11], vec![22], vec![11], vec![22]]),
                (&renamed, varying(&renamed, 0)),
                vec![vec![0, 1, 2, 3], vec![0, 3]],
            ),
            (
                "sum along x, varying along m -> completed",
                (
                    &scalar,
                    &manual,
                    varying(&manual, 0).with_unreduced_axes(["x"]).unwrap(),
                    [vec![1], vec![2], vec![10], vec![20]],
                ),
                (&manual, varying(&manual, 0)),
                vec![vec![0, 1, 2, 3], vec![0, 3]],
            ),
            (
                "tiles varying along m -> renamed mesh",
                (&vector, &manual, varying_tiles(&manual), [vec![7], vec![9], vec![7], vec![9]]),
                (&renamed, varying_tiles(&renamed)),
                vec![vec![0, 1, 2, 3], vec![2, 3], vec![0, 3]],
            ),
            (
                "replicated -> split along x, sum along y",
                (
                    &vector,
                    &explicit,
                    Sharding::replicated(explicit.logical_mesh().clone(), 1),
                    [vec![7, 8], vec![7, 8], vec![7, 8], vec![7, 8]],
                ),
                (&explicit, split(&explicit, "x").with_unreduced_axes(["y"]).unwrap()),
                vec![vec![0, 1, 2, 3], vec![1, 3]],
            ),
        ];
        let mut outcomes = Vec::new();
        for (name, (r#type, source_mesh, source_sharding, bytes), (mesh, sharding), helds) in rows {
            for held in helds {
                let (mut results, mut strategy) = (Vec::new(), String::new());
                for consume in [false, true] {
                    let session = Session::builder().client(client).build().unwrap();
                    let buffers = held
                        .iter()
                        .map(|&index| {
                            let rank = r#type.static_shape().unwrap().rank();
                            let dimensions = if rank == 0 { Vec::new() } else { vec![bytes[index].len() as u64] };
                            let device = devices[index].clone();
                            client.buffer(bytes[index].as_slice(), BufferType::U8, dimensions, None, device, None)
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                    let source = Array::from_addressable_buffers(
                        &session,
                        r#type.clone(),
                        source_mesh.clone(),
                        Some(source_sharding.clone()),
                        buffers,
                    )
                    .unwrap();
                    strategy = planned_work(&source, &simulated_cross_host_layout, 0, mesh, &sharding)
                        .map_or_else(|_| "no device strategy".to_string(), |work| work.strategy);
                    let target = Placement::new(mesh.clone(), sharding.clone()).unwrap();
                    let placed = if consume {
                        source.into_placement(&session, target)
                    } else {
                        source.to_placement(&session, target)
                    };
                    results.push(
                        placed
                            .and_then(|placed| {
                                assert_eq!(placed.sharding(), &sharding, "{name}");
                                placed
                                    .shards()
                                    .iter()
                                    .map(ArrayShard::copy_to_host_bytes)
                                    .collect::<Result<Vec<_>, _>>()
                            })
                            .map_err(|error| error.to_string()),
                    );
                }
                assert_eq!(results[0], results[1], "{name} held {held:?}: the entry points disagree");
                let outcome = match &results[0] {
                    Ok(bytes) => format!("{bytes:?}"),
                    Err(error) => format!("error: {error}"),
                };
                outcomes.push(format!("{name} held {held:?}: {strategy}: {outcome}"));
            }
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                varying along m -> sum along x held [0, 1, 2, 3]: additive identities: [[11], [22], [0], [0]]
                varying along m -> sum along x held [0, 3]: additive identities: [[11], [22], [0], [0]]
                varying along m -> sum along x held [1, 2]: additive identities: [[11], [22], [0], [0]]
                varying along m -> sum along x held [0]: additive identities: error: array shard #1 on device 1 has no addressable buffer
                varying along m -> renamed mesh held [0, 1, 2, 3]: exact shards: [[11], [22], [11], [22]]
                varying along m -> renamed mesh held [0, 3]: exact shards: [[11], [22], [11], [22]]
                sum along x, varying along m -> completed held [0, 1, 2, 3]: host round trip: error: compiled reshard does not support `Manual` mesh axis m
                sum along x, varying along m -> completed held [0, 3]: host round trip: error: compiled reshard does not support `Manual` mesh axis m
                tiles varying along m -> renamed mesh held [0, 1, 2, 3]: exact shards: [[7], [9], [7], [9]]
                tiles varying along m -> renamed mesh held [2, 3]: exact shards: [[7], [9], [7], [9]]
                tiles varying along m -> renamed mesh held [0, 3]: exact shards: [[7], [9], [7], [9]]
                replicated -> split along x, sum along y held [0, 1, 2, 3]: compiled reshard with additive identities: [[7], [0], [8], [0]]
                replicated -> split along x, sum along y held [1, 3]: compiled reshard with additive identities: [[7], [0], [8], [0]]"},
        );
    }

    #[test]
    fn test_array_place_within_client_preserves_value_across_partial_completions() {
        // Transitions of pending sums along the axes `x` and `y` of a 2x2 mesh that complete some of them and keep or
        // introduce others, with contributions whose partial sums overflow or round in the element type (element 0 sums
        // along `y` first and element 1 along `x`), each placed with `to_placement` and `into_placement`, which must
        // agree. Each row lists the strategy, the programs that each entry point compiles, and the host readbacks of
        // the source and of the result, which are equal bit for bit for every data type but `f32`. A partial completion
        // of a data type that accumulates its sums in a wider type cannot run on the devices, because partial sums in
        // the element type would change the value (e.g., the `f8e4m3fn` contributions `448`, `448`, `-448`, and
        // `-448`, which sum to zero, have the partial sums `896` and `-896`, which are NaN in `f8e4m3fn`, and the
        // `bf16` contributions `256`, `1`, `-256`, and `1` have the partial sum `257`, which rounds to `256`). The
        // host round trip that remains would combine the partial sums that the target keeps, so both entry points
        // return the partial-contribution error, which the rows list as `error` and `errors` in
        // full. A data type that accumulates its sums in itself keeps them on the devices: integers are exact, and an
        // `f32` partial completion fixes the order of accumulation to its partial sums first, as the contract of
        // floating point sums allows, so the `f32` contributions `2^24`, `1`, `-2^24`, and `1`, which read back as
        // `1`, read back as `2` after completing `x`. Moving a pending sum from `x` to `y` only moves its contributions
        // (refer to `moves_contributions`), so it takes exact shards without arithmetic.
        let client = cpu_client(4);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let rows = [
            (DataType::F8E4M3FN, [448f64, 448., -448., -448.]),
            (DataType::BF16, [256., 1., -256., 1.]),
            (DataType::F32, [16777216., 1., -16777216., 1.]),
            (DataType::I32, [f64::from(i32::MAX), 1., f64::from(i32::MIN), -1.]),
        ];
        let (mut outcomes, mut errors) = (Vec::new(), Vec::new());
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Auto] {
            let logical_mesh = LogicalMesh::new(vec![
                MeshAxis::new("x", 2, axis_type).unwrap(),
                MeshAxis::new("y", 2, axis_type).unwrap(),
            ])
            .unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
            let replicated = Sharding::replicated(logical_mesh.clone(), 1);
            let split_x = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
            let pending = |sharding: &Sharding, axes: &[&str]| {
                sharding.clone().with_unreduced_axes(axes.iter().copied()).unwrap()
            };
            let transitions = [
                ("sum(x, y) -> sum(x)", &["x", "y"][..], pending(&replicated, &["x"])),
                ("sum(x, y) -> sum(y)", &["x", "y"][..], pending(&replicated, &["y"])),
                ("sum(x, y) -> none", &["x", "y"][..], replicated.clone()),
                ("sum(x, y) -> sum(y) split along x", &["x", "y"][..], pending(&split_x, &["y"])),
                ("sum(x) -> none", &["x"][..], replicated.clone()),
                ("sum(x) -> sum(y)", &["x"][..], pending(&replicated, &["y"])),
                ("sum(x) -> sum(x, y)", &["x"][..], pending(&replicated, &["x", "y"])),
            ];
            for (data_type, values) in rows {
                let r#type = ArrayType::new_static(data_type, [2]);
                let decode = |bytes: &[u8]| {
                    let array = CoreArray::new(r#type.clone(), bytes.to_vec()).unwrap();
                    array.convert_element_type(DataType::F64).unwrap().elements::<f64>().unwrap()
                };
                for (name, source_axes, target_sharding) in &transitions {
                    let mut results = Vec::new();
                    for consume in [false, true] {
                        let session = Session::builder().client(client).build().unwrap();
                        // The device at `(x, y)` holds `values[2 * x + y]` in element 0 and `values[x + 2 * y]` in
                        // element 1, or, for a source with a pending sum along `x` only, `values[2 * x]` and
                        // `values[2 * x + 1]`, so that its replicas along `y` hold equal contributions.
                        let shards = devices.iter().enumerate().map(|(index, device)| {
                            let (x, y) = (index / 2, index % 2);
                            let elements = match source_axes.len() {
                                2 => [values[2 * x + y], values[x + 2 * y]],
                                _ => [values[2 * x], values[2 * x + 1]],
                            };
                            let bytes = elements
                                .map(|value: f64| {
                                    CoreArray::scalar(value)
                                        .unwrap()
                                        .convert_element_type(data_type)
                                        .unwrap()
                                        .logical_bytes()
                                })
                                .concat();
                            (device.id(), Arc::<[u8]>::from(bytes))
                        });
                        let sharding = Some(pending(&replicated, source_axes));
                        let source = Array::from_host_shards(&session, r#type.clone(), mesh.clone(), sharding, shards);
                        let source = source.unwrap();
                        let expected = source.copy_to_host_bytes().unwrap();
                        let strategy = planned_work(&source, &simulated_cross_host_layout, 0, &mesh, target_sharding)
                            .map(|work| work.strategy);
                        let target = Placement::new(mesh.clone(), target_sharding.clone()).unwrap();
                        let placed = if consume {
                            source.into_placement(&session, target)
                        } else {
                            source.to_placement(&session, target)
                        };
                        let actual = placed.and_then(|placed| {
                            assert_eq!(placed.sharding(), target_sharding, "{data_type} {name}");
                            placed.copy_to_host_bytes()
                        });
                        let actual = actual.map_err(|error| error.to_string());
                        results.push((strategy, session.cached_program_count(), expected, actual));
                    }
                    let [(strategy, programs, expected, actual), (_, consumed_programs, _, consumed)] =
                        results.try_into().unwrap();
                    assert_eq!(actual, consumed, "{axis_type:?} {data_type} {name}: the entry points disagree");
                    let programs = format!("+{programs} +{consumed_programs}");
                    let outcome = match (strategy, actual) {
                        (Ok(strategy), Ok(actual)) => {
                            if data_type != DataType::F32 {
                                assert_eq!(expected, actual, "{axis_type:?} {data_type} {name}");
                            }
                            format!("{strategy} {programs}: {:?} -> {:?}", decode(&expected), decode(&actual))
                        }
                        (Err(planned), Err(error)) => {
                            assert_eq!(planned, error, "{axis_type:?} {data_type} {name}");
                            errors.push(format!("{axis_type:?} {data_type} {name}: {error}"));
                            format!("error {programs}")
                        }
                        (strategy, actual) => panic!("{axis_type:?} {data_type} {name}: {strategy:?} {actual:?}"),
                    };
                    outcomes.push(format!("{axis_type:?} {data_type} {name}: {outcome}"));
                }
            }
        }
        let mut expected_errors = Vec::new();
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Auto] {
            for (data_type, accumulation_type) in [(DataType::F8E4M3FN, DataType::F16), (DataType::BF16, DataType::F32)]
            {
                for (name, completed, kept) in [
                    ("sum(x, y) -> sum(x)", "y", "x"),
                    ("sum(x, y) -> sum(y)", "x", "y"),
                    ("sum(x, y) -> sum(y) split along x", "x", "y"),
                ] {
                    expected_errors.push(format!(
                        "{axis_type:?} {data_type} {name}: cannot place an array whose pending reduction the placement \
                         keeps along the mesh axes `{kept}` and changes along others when no device strategy applies \
                         (cannot complete a pending `sum` of `{data_type}` values along mesh axis `{completed}` and \
                         keep it along mesh axis `{kept}` on the devices, because the partial sums would be rounded \
                         to `{data_type}`, while the sum accumulates in `{accumulation_type}`), because the host \
                         round trip that remains would combine the contributions that the placement keeps; complete \
                         the reduction first",
                    ));
                }
            }
        }
        assert_eq!(errors, expected_errors);
        let outcomes = outcomes.into_iter().map(|outcome| outcome.replacen("Auto ", "", 1)).collect::<Vec<_>>();
        let (explicit, auto) = outcomes.split_at(outcomes.len() / 2);
        let explicit = explicit.iter().map(|outcome| outcome.replacen("Explicit ", "", 1)).collect::<Vec<_>>();
        assert_eq!(explicit, auto, "both axis types must give the same rows");
        assert_eq!(
            auto.join("\n"),
            indoc! {"
                f8e4m3fn sum(x, y) -> sum(x): error +0 +0
                f8e4m3fn sum(x, y) -> sum(y): error +0 +0
                f8e4m3fn sum(x, y) -> none: compiled reshard +1 +1: [0.0, 0.0] -> [0.0, 0.0]
                f8e4m3fn sum(x, y) -> sum(y) split along x: error +0 +0
                f8e4m3fn sum(x) -> none: compiled reshard +1 +1: [0.0, 0.0] -> [0.0, 0.0]
                f8e4m3fn sum(x) -> sum(y): exact shards +0 +0: [0.0, 0.0] -> [0.0, 0.0]
                f8e4m3fn sum(x) -> sum(x, y): additive identities +0 +0: [0.0, 0.0] -> [0.0, 0.0]
                bf16 sum(x, y) -> sum(x): error +0 +0
                bf16 sum(x, y) -> sum(y): error +0 +0
                bf16 sum(x, y) -> none: compiled reshard +1 +1: [2.0, 2.0] -> [2.0, 2.0]
                bf16 sum(x, y) -> sum(y) split along x: error +0 +0
                bf16 sum(x) -> none: compiled reshard +1 +1: [0.0, 2.0] -> [0.0, 2.0]
                bf16 sum(x) -> sum(y): exact shards +0 +0: [0.0, 2.0] -> [0.0, 2.0]
                bf16 sum(x) -> sum(x, y): additive identities +0 +0: [0.0, 2.0] -> [0.0, 2.0]
                f32 sum(x, y) -> sum(x): compiled reshard +1 +1: [1.0, 2.0] -> [1.0, 2.0]
                f32 sum(x, y) -> sum(y): compiled reshard +1 +1: [1.0, 2.0] -> [2.0, 1.0]
                f32 sum(x, y) -> none: compiled reshard +1 +1: [1.0, 2.0] -> [1.0, 2.0]
                f32 sum(x, y) -> sum(y) split along x: compiled reshard +1 +1: [1.0, 2.0] -> [2.0, 1.0]
                f32 sum(x) -> none: compiled reshard +1 +1: [0.0, 2.0] -> [0.0, 2.0]
                f32 sum(x) -> sum(y): exact shards +0 +0: [0.0, 2.0] -> [0.0, 2.0]
                f32 sum(x) -> sum(x, y): additive identities +0 +0: [0.0, 2.0] -> [0.0, 2.0]
                i32 sum(x, y) -> sum(x): compiled reshard +1 +1: [-1.0, -1.0] -> [-1.0, -1.0]
                i32 sum(x, y) -> sum(y): compiled reshard +1 +1: [-1.0, -1.0] -> [-1.0, -1.0]
                i32 sum(x, y) -> none: compiled reshard +1 +1: [-1.0, -1.0] -> [-1.0, -1.0]
                i32 sum(x, y) -> sum(y) split along x: compiled reshard +1 +1: [-1.0, -1.0] -> [-1.0, -1.0]
                i32 sum(x) -> none: compiled reshard +1 +1: [-1.0, 0.0] -> [-1.0, 0.0]
                i32 sum(x) -> sum(y): exact shards +0 +0: [-1.0, 0.0] -> [-1.0, 0.0]
                i32 sum(x) -> sum(x, y): additive identities +0 +0: [-1.0, 0.0] -> [-1.0, 0.0]"},
        );
    }

    #[test]
    fn test_array_place_within_client_keeps_partial_sums_or_rejects_the_placement() {
        // The devices at `(0, 0)`, `(0, 1)`, `(1, 0)`, and `(1, 1)` of a 2x2 mesh hold the `f32` contributions `MAX`,
        // `0`, `MAX`, and `-MAX`, whose combined value is infinite, to a value of five elements with a pending sum
        // along `x` and `y`. Completing `y` keeps the partial sums `MAX` and `0` along `x`, which an untiled
        // target does with a compiled reshard. A target split unevenly along `y` rules out every device
        // strategy, and the host round trip that remains would combine the partial sums, so both entry points return
        // the same error, which every process derives from the shardings. Each row lists the shards in mesh order and
        // the readback of the result, with the equal elements of each as `value xcount`.
        let client = cpu_client(4);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        let source_sharding = Sharding::replicated(logical_mesh.clone(), 1).with_unreduced_axes(["x", "y"]).unwrap();
        let untiled = Sharding::replicated(logical_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let uneven = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["y"])])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let uneven_shards =
            ArrayError::UnevenReshardingShards { shape: StaticShape::new(vec![5]), sharding: uneven.clone() };
        let values = [f32::MAX, 0., f32::MAX, -f32::MAX];
        let r#type = ArrayType::new_static(DataType::F32, [5]);
        let render = |bytes: &[u8]| {
            let elements = values_from_bytes::<f32>(bytes);
            assert_eq!(elements, vec![elements[0]; elements.len()]);
            let value = match elements[0] {
                value if value == f32::MAX => "MAX".to_string(),
                value => value.to_string(),
            };
            format!("{value} x{}", elements.len())
        };
        let mut outcomes = Vec::new();
        for (name, target_sharding) in [("untiled", &untiled), ("split unevenly along y", &uneven)] {
            let mut results = Vec::new();
            for consume in [false, true] {
                let session = Session::builder().client(client).build().unwrap();
                let shards = devices
                    .iter()
                    .zip(values)
                    .map(|(device, value)| (device.id(), Arc::<[u8]>::from(values_to_bytes(&[value; 5]))));
                let sharding = Some(source_sharding.clone());
                let source = Array::from_host_shards(&session, r#type.clone(), mesh.clone(), sharding, shards);
                let source = source.unwrap();
                let target = Placement::new(mesh.clone(), target_sharding.clone()).unwrap();
                let placed = if consume {
                    source.into_placement(&session, target)
                } else {
                    source.to_placement(&session, target)
                };
                let outcome = match placed {
                    Ok(placed) => {
                        assert_eq!(placed.sharding(), target_sharding);
                        let shards = placed.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap());
                        let shards = shards.map(|bytes| render(&bytes)).collect::<Vec<_>>().join(", ");
                        format!("[{shards}] -> {}", render(&placed.copy_to_host_bytes().unwrap()))
                    }
                    Err(error) => {
                        assert_eq!(
                            error.to_string(),
                            format!(
                                "cannot place an array whose pending reduction the placement keeps along the mesh \
                                 axes `x` and changes along others when no device strategy applies ({uneven_shards}), \
                                 because the host round trip that remains would combine the contributions that the \
                                 placement keeps; complete the reduction first, or place the array onto an even split",
                            ),
                        );
                        "error".to_string()
                    }
                };
                results.push(outcome);
            }
            assert_eq!(results[0], results[1], "{name}: the entry points disagree");
            outcomes.push(format!("{name}: {}", results[0]));
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                untiled: [MAX x5, MAX x5, 0 x5, 0 x5] -> MAX x5
                split unevenly along y: error"},
        );
    }

    #[test]
    fn test_array_place_within_client_compiled_reshard_on_mesh_of_other_processes() {
        let client = cpu_client(2);
        let local = Session::builder().client(client).build().unwrap().addressable_devices().unwrap();
        let next_id = local.iter().map(Device::id).max().unwrap() + 1;
        let remote = [Device::new(next_id, 1), Device::new(next_id + 1, 1)];
        let session = session_with_remote_devices(client, remote);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = |devices: [Device; 2]| DeviceMesh::new(logical_mesh.clone(), devices.to_vec()).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 2);
        let rows = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let values = values_to_bytes(&[1f32, 2., 3., 4.]);

        // Each case places a source with the given type and sharding on its own mesh onto the given sharding over the
        // given devices with a compiled strategy, once on the devices of this process, which runs every program, and
        // once on the devices of another process, of which this process holds only metadata. The first case reshards
        // on the mesh of the source, and the second one gathers the source on its mesh, copies it to the reversed mesh,
        // and reshards it there, because the declared layout rules out shard regions.
        let r#type = ArrayType::new_static(DataType::F32, [2, 2]);
        let cases = [
            ("same mesh", r#type.clone(), &replicated, &rows, false),
            ("across meshes", r#type.with_layout(column_major.clone()), &rows, &replicated, true),
        ];
        for (case, r#type, source_sharding, sharding, reversed) in cases {
            let target_mesh =
                |[first, second]: [Device; 2]| if reversed { mesh([second, first]) } else { mesh([first, second]) };
            let process_index = local[0].process_index();
            let source = Array::from_host_buffer(
                &session,
                r#type.clone(),
                mesh([local[0], local[1]]),
                Some(source_sharding.clone()),
                &values,
            )
            .unwrap();
            let strategy = source.device_placement_strategy(
                &cross_host_layout_resolver(session.client()),
                &target_mesh([local[0], local[1]]),
                sharding,
                process_index,
            );
            if reversed {
                assert!(matches!(strategy, Ok(DevicePlacementStrategy::CompiledReshardAcrossMeshes { .. })), "{case}");
            } else {
                assert!(matches!(strategy, Ok(DevicePlacementStrategy::CompiledReshard { .. })), "{case}");
            }
            let target = Placement::new(target_mesh([local[0], local[1]]), sharding.clone()).unwrap();
            let participant = source.to_placement(&session, target).unwrap();
            let participant_moved = source
                .clone()
                .reshard_on_devices(&session, &target_mesh([local[0], local[1]]), sharding, false)
                .unwrap();
            assert_eq!(participant.addressable_shards().count(), 2, "{case}");
            assert_eq!(participant.copy_to_host().unwrap().elements::<f32>().unwrap(), [1., 2., 3., 4.], "{case}");

            // This process owns no device of either mesh, so it compiles and runs nothing, and it returns the type, the
            // placement, and the commitment that the processes which own those devices compute for their results.
            let peer_source = Array::from_addressable_buffers(
                &session,
                r#type,
                mesh(remote),
                Some(source_sharding.clone()),
                Vec::new(),
            )
            .unwrap();
            let cached_program_count = session.cached_program_count();
            let target = Placement::new(target_mesh(remote), sharding.clone()).unwrap();
            let placed = peer_source.to_placement(&session, target).unwrap();
            let moved =
                peer_source.clone().reshard_on_devices(&session, &target_mesh(remote), sharding, false).unwrap();
            for output in [&placed, &moved] {
                assert_eq!(output.mesh(), &target_mesh(remote), "{case}");
                assert_eq!(output.sharding(), participant.sharding(), "{case}");
                assert_eq!(output.addressable_shards().count(), 0, "{case}");
                assert!(output.is_committed(), "{case}");
                assert_eq!(output.block_until_ready().map_err(|error| error.to_string()), Ok(()), "{case}");
            }
            assert_eq!(placed.r#type(), participant.r#type(), "{case}");
            assert_eq!(moved.r#type(), participant_moved.r#type(), "{case}");
            assert_eq!(session.cached_program_count(), cached_program_count, "{case}");
        }
    }

    #[test]
    fn test_array_place_within_client_moves_only_required_cross_mesh_regions() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let source_mesh = two_device_mesh(client, MeshAxisType::Auto);
        let source_sharding = Sharding::new(
            source_mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let source = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4, 4]),
            source_mesh,
            Some(source_sharding),
            values_to_bytes(&values),
        )
        .unwrap();
        let target_mesh = four_device_mesh_x(client);
        let target_sharding = Sharding::new(
            target_mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        // The strategy keeps the regions that it derived to choose itself, which the placement then executes.
        let process_index = client.process_index().unwrap();
        let Ok(DevicePlacementStrategy::ShardRegions { plan, cross_host_layouts }) = source.device_placement_strategy(
            &cross_host_layout_resolver(session.client()),
            &target_mesh,
            &target_sharding,
            process_index,
        ) else {
            panic!("a placement on another mesh of this process must use shard regions");
        };
        assert!(cross_host_layouts.is_empty());

        // Each column needs two regions. Two of the eight regions are already on their target devices.
        let copied_bytes = plan
            .moves
            .iter()
            .filter(|movement| movement.source_device != movement.target_device)
            .map(|movement| movement.shape().as_slice().iter().product::<usize>() * size_of::<f32>())
            .sum::<usize>();
        assert_eq!(plan.moves.len(), 8);
        assert_eq!(copied_bytes, 48);

        let output = source.to_placement(&session, Placement::new(target_mesh, target_sharding).unwrap()).unwrap();
        output.block_until_ready().unwrap();
        assert_ne!(session.eager_dispatch_statistics().misses, 0);
        assert!(output.execution_fence().is_some());
        assert_eq!(
            shard_values(&output),
            vec![vec![0., 4., 8., 12.], vec![1., 5., 9., 13.], vec![2., 6., 10., 14.], vec![3., 7., 11., 15.]],
        );
        assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), values);
    }

    #[test]
    fn test_array_place_within_client_reuses_donated_buffers_when_every_device_keeps_its_shard_type() {
        // A donated array whose compiled reshard leaves every device with a shard of the type that it held there (e.g.,
        // one that completes a pending sum) gives its buffers to the result, because the lowering pairs the input and
        // the output of that reshard (refer to `has_same_shard_types`), as JAX pairs a donated argument of `jit` with a
        // result of the same shard shape. Each row lists, per device in the order of the mesh, whether the result holds
        // the buffer that the source held there, with donation and without it (i.e., with `to_placement`, which keeps
        // the source), and the values of every shard. A row marked as forced runs the compiled reshard although
        // another strategy (moves of whole shards) places it otherwise.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let square = DeviceMesh::new(logical_mesh_2x2(), devices.clone()).unwrap();
        let auto = two_device_mesh(client, MeshAxisType::Auto);
        let explicit = two_device_mesh(client, MeshAxisType::Explicit);
        let renamed_axes = LogicalMesh::new(vec![MeshAxis::new("a", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let renamed = DeviceMesh::new(renamed_axes, devices[..2].to_vec()).unwrap();
        let split = |mesh: &DeviceMesh, axes: &[&str]| {
            let dimension = match axes {
                [] => ShardingDimension::replicated(),
                axes => ShardingDimension::sharded(axes.iter().copied()),
            };
            Sharding::new(mesh.logical_mesh().clone(), vec![dimension]).unwrap()
        };
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let values = values_to_bytes(&[1f32, 2., 3., 4.]);
        let dense = |mesh: &DeviceMesh, sharding: Sharding| {
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), Some(sharding), values.as_slice()).unwrap()
        };
        let pending = |mesh: &DeviceMesh, sharding: Sharding, contributions: &[&[f32]]| {
            let sharding = sharding.with_unreduced_axes(["x"]).unwrap();
            let shards = contributions
                .iter()
                .zip(mesh.devices())
                .map(|(values, device)| (device.id(), Arc::<[u8]>::from(values_to_bytes(values))))
                .collect::<Vec<_>>();
            Array::from_host_shards(&session, r#type.clone(), mesh.clone(), Some(sharding), shards).unwrap()
        };
        // Native addresses of the buffers of `array`, which are ready and alive while they are read, and are only
        // compared and never dereferenced.
        let addresses = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| {
                    let buffer = shard.buffer().unwrap();
                    buffer.ready().unwrap().r#await().unwrap();
                    unsafe { buffer.unsafe_pointer().unwrap() }
                })
                .collect::<Vec<_>>()
        };
        let place = |source: Array, target: &Placement, forced: bool, donate: bool| {
            let before = addresses(&source);
            let placed = match (forced, donate) {
                (true, _) => {
                    let (mesh, sharding) = (target.mesh(), target.sharding());
                    let (input, output) = (source.sharding().clone(), sharding.clone());
                    let program = ReshardProgram { mesh: mesh.clone(), input, output };
                    let strategy = DevicePlacementStrategy::CompiledReshard { program };
                    let placed = source.place_with_strategy(strategy, 0, mesh, sharding, Memory::Device, None, donate);
                    placed.map_err(|(error, _)| error).unwrap()
                }
                (false, true) => source.into_placement(&session, target.clone()).unwrap(),
                (false, false) => source.to_placement(&session, target.clone()).unwrap(),
            };
            let reused = before.iter().zip(addresses(&placed)).map(|(before, after)| *before == after);
            let reused = reused.collect::<Vec<_>>();
            (reused, shard_values(&placed))
        };

        let sums: &[&[f32]] = &[&[1., 2., 3., 4.], &[10., 20., 30., 40.]];
        let split_sums: &[&[f32]] = &[&[1., 2.], &[3., 4.], &[10., 20.], &[30., 40.]];
        type Source<'a> = Box<dyn Fn() -> Array + 'a>;
        let cases: Vec<(&str, Source<'_>, Placement, bool)> = vec![
            (
                "completed sum on an auto axis",
                Box::new(|| pending(&auto, split(&auto, &[]), sums)),
                Placement::new(auto.clone(), split(&auto, &[])).unwrap(),
                false,
            ),
            (
                "completed sum on an explicit axis",
                Box::new(|| pending(&explicit, split(&explicit, &[]), sums)),
                Placement::new(explicit.clone(), split(&explicit, &[])).unwrap(),
                false,
            ),
            (
                "completed sum of a split",
                Box::new(|| pending(&square, split(&square, &["y"]), split_sums)),
                Placement::new(square.clone(), split(&square, &["y"])).unwrap(),
                false,
            ),
            (
                "split to replicated",
                Box::new(|| dense(&explicit, split(&explicit, &["x"]))),
                Placement::new(explicit.clone(), split(&explicit, &[])).unwrap(),
                false,
            ),
            (
                "split along another axis, forced",
                Box::new(|| dense(&square, split(&square, &["x"]))),
                Placement::new(square.clone(), split(&square, &["y"])).unwrap(),
                true,
            ),
            (
                "renamed mesh",
                Box::new(|| dense(&explicit, split(&explicit, &["x"]))),
                Placement::new(renamed.clone(), split(&renamed, &["a"])).unwrap(),
                false,
            ),
        ];
        let outcomes = cases
            .into_iter()
            .map(|(name, source, target, forced)| {
                let (donated, values) = place(source(), &target, forced, true);
                let (borrowed, borrowed_values) = place(source(), &target, forced, false);
                assert_eq!(values, borrowed_values, "{name}");
                format!("{name}: donated {donated:?}, borrowed {borrowed:?}, values {values:?}")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                completed sum on an auto axis: donated [true, true], borrowed [false, false], values [[11.0, 22.0, 33.0, 44.0], [11.0, 22.0, 33.0, 44.0]]
                completed sum on an explicit axis: donated [true, true], borrowed [false, false], values [[11.0, 22.0, 33.0, 44.0], [11.0, 22.0, 33.0, 44.0]]
                completed sum of a split: donated [true, true, true, true], borrowed [false, false, false, false], values [[11.0, 22.0], [33.0, 44.0], [11.0, 22.0], [33.0, 44.0]]
                split to replicated: donated [false, false], borrowed [false, false], values [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
                split along another axis, forced: donated [true, true, true, true], borrowed [false, false, false, false], values [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]]
                renamed mesh: donated [true, true], borrowed [true, true], values [[1.0, 2.0], [3.0, 4.0]]"},
        );
    }

    #[test]
    fn test_array_reshard_on_devices() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        let tiled = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        // A sharding with the same shards on the same devices shares every buffer, compiles nothing, and keeps the
        // completion of the execution that produced the array.
        let (event, promise) = client.event(()).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(tiled.clone()).unwrap();
        let source =
            Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap()
                .with_execution_fence(ExecutionFence::new(vec![event]));
        let reduced = tiled.clone().with_reduced_axes(["x"]).unwrap();
        let output = source.clone().reshard_on_devices(&session, &mesh, &reduced, false).unwrap();
        let pending = matches!(output.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false)));
        promise.set(None).unwrap();
        assert!(pending);
        assert_eq!(output.sharding(), &reduced);
        assert_eq!(output.mesh(), &mesh);
        let shared_buffers = source
            .shards()
            .iter()
            .zip(output.shards())
            .map(|(left, right)| Arc::ptr_eq(left.buffer().unwrap(), right.buffer().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(shared_buffers, vec![true; 4]);
        assert_eq!(session.cached_program_count(), 0);

        // On another device mesh, the shard on a device of the source shares its buffer and the others are copies.
        let single_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[1]],
        )
        .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let source =
            Array::from_host_buffer(&session, r#type, single_device_mesh, None, values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let output = source.clone().reshard_on_devices(&session, &mesh, &replicated, false).unwrap();
        assert_eq!(output.sharding(), &replicated);
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(
            output
                .shards()
                .iter()
                .map(|shard| Arc::ptr_eq(shard.buffer().unwrap(), source.shards()[0].buffer().unwrap()))
                .collect::<Vec<_>>(),
            vec![false, true, false, false],
        );
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4.]; 4]);
        assert_eq!(session.cached_program_count(), 0);

        // A sharding with other shards needs a compiled program.
        let sharded = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let resharded = output.clone().reshard_on_devices(&session, &mesh, &sharded, false).unwrap();
        assert_eq!(resharded.sharding(), &sharded);
        assert_eq!(shard_values(&resharded), vec![vec![1.], vec![2.], vec![3.], vec![4.]]);
        assert_eq!(session.cached_program_count(), 1);

        // A result on more than one device is committed, whichever strategy moves it and whatever the commitment of
        // the array.
        assert!(source.clone().reshard_on_devices(&session, &mesh, &replicated, false).unwrap().is_committed());
        let uncommitted = source.with_committed(false).reshard_on_devices(&session, &mesh, &replicated, false);
        assert!(uncommitted.unwrap().is_committed());
        let uncommitted = output.clone().with_committed(false).reshard_on_devices(&session, &mesh, &sharded, false);
        assert!(uncommitted.unwrap().is_committed());

        // Region placement also reaches a different mesh with a manual axis.
        let manual_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let manual =
            Sharding::new(manual_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = output.reshard_on_devices(&session, &manual_mesh, &manual, false).unwrap();
        assert_eq!(shard_values(&resharded), vec![vec![1.], vec![2.], vec![3.], vec![4.]]);
    }

    #[test]
    fn test_array_reshard_on_devices_empty_regions() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = four_device_mesh_x(client);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let columns = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        let upload = |shape, values: &[f32]| {
            Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, shape),
                mesh.clone(),
                Some(replicated.clone()),
                values_to_bytes(values),
            )
            .unwrap()
        };
        // Two devices receive a column each. The other two own real buffers with zero columns.
        let split = upload([2, 2], &[1., 2., 3., 4.]).reshard_on_devices(&session, &mesh, &columns, false).unwrap();
        let dimensions = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| shard.buffer().unwrap().dimensions().unwrap().to_vec())
                .collect::<Vec<_>>()
        };
        assert_eq!(dimensions(&split), [[2, 1], [2, 1], [2, 0], [2, 0]]);
        assert_eq!(shard_values(&split), [vec![1., 3.], vec![2., 4.], vec![], vec![]]);
        let gathered = split.reshard_on_devices(&session, &mesh, &replicated, false).unwrap();
        assert_eq!(shard_values(&gathered), vec![vec![1., 2., 3., 4.]; 4]);

        // With zero rows, even the one-column shards have no intersections. The newly allocated empty
        // buffers must retain the producer's completion through the result.
        let (event, producer) = client.event(()).unwrap();
        let empty = upload([0, 2], &[]).with_execution_fence(ExecutionFence::new(vec![event]));
        let split = empty.reshard_on_devices(&session, &mesh, &columns, false).unwrap();
        assert_eq!(dimensions(&split), [[0, 1], [0, 1], [0, 0], [0, 0]]);
        assert!(matches!(split.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))));
        producer.set(Some(PjrtError::aborted("empty producer failed"))).unwrap();
        assert!(matches!(
            split.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "empty producer failed",
        ));

        // Buffers supplied by a caller can carry an error even when there is no execution fence.
        let buffers = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| {
                client
                    .error_buffer(
                        PjrtError::aborted("empty buffer failed"),
                        BufferSpecification::new(BufferType::F32, [0, 2]),
                        device,
                    )
                    .unwrap()
            })
            .collect();
        let empty = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::F32, [0, 2]),
            mesh.clone(),
            Some(replicated),
            buffers,
        )
        .unwrap();
        let split = empty.reshard_on_devices(&session, &mesh, &columns, false).unwrap();
        assert!(matches!(
            split.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "empty buffer failed",
        ));
    }

    #[test]
    fn test_array_reshard_on_devices_declines_changes_that_need_a_host_round_trip() {
        // The preparation of the arguments of a program has no host fallback, so a change of reduction state that only
        // a host round trip makes exactly (here, introducing a pending maximum) is an error that points to a placement
        // of the concrete array, also on a single process, which could make that round trip.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let maximum = replicated.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(UnreducedKind::Max);
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let source =
            Array::from_host_buffer(&session, r#type, mesh.clone(), Some(replicated), values_to_bytes(&[1f32; 4]));
        let placed = source.unwrap().reshard_on_devices(&session, &mesh, &maximum.unwrap(), false);
        let message = "cannot take `f32` values from no reduction state to a pending `max` reduction along mesh axis \
                       `x` on the devices, because XLA can only introduce a pending `sum` reduction, and the \
                       preparation of the arguments of a program has no host round trip; place the argument with \
                       `Array::to_placement` first, which makes the change with a host round trip where one process \
                       owns every device of the target and a device of every source contribution with elements";
        assert_eq!(placed.map(|_| ()).map_err(|error| error.to_string()), Err(message.to_string()));
    }

    #[test]
    fn test_array_reshard_on_devices_serves_missing_local_replicas_of_compiled_reshard() {
        // On a 2x2 mesh, an `f32[4]` imported with the buffers of the devices at `y = 1` only holds one local replica
        // of every tile. A compiled reshard from replicated to split along `x`, and from split to replicated, reads the
        // buffer of every device, so it first copies the local replica of the same tile to each device without one.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let pjrt_devices = client.addressable_devices().unwrap();
        let axes = vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ];
        let logical_mesh = LogicalMesh::new(axes).unwrap();
        let devices = pjrt_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let split = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let halves = vec![vec![1f32, 2.], vec![1., 2.], vec![3., 4.], vec![3., 4.]];
        let cases = [
            (&replicated, &split, [vec![1f32, 2., 3., 4.], vec![1., 2., 3., 4.]], halves),
            (&split, &replicated, [vec![1., 2.], vec![3., 4.]], vec![vec![1., 2., 3., 4.]; 4]),
        ];
        for (source, target_sharding, tiles, expected) in cases {
            let buffers = [1, 3]
                .into_iter()
                .map(|index| {
                    let (bytes, dimensions) = (values_to_bytes(&tiles[index / 2]), [tiles[index / 2].len() as u64]);
                    let device = pjrt_devices[index].clone();
                    client.buffer(bytes.as_slice(), BufferType::F32, dimensions, None, device, None).unwrap()
                })
                .collect();
            let r#type = ArrayType::new_static(DataType::F32, [4]);
            let (mesh, sharding) = (mesh.clone(), Some(source.clone()));
            let input = Array::from_addressable_buffers(&session, r#type, mesh.clone(), sharding, buffers).unwrap();
            let layout_resolver = cross_host_layout_resolver(client);
            let strategy = input.device_placement_strategy(&layout_resolver, &mesh, target_sharding, 0).unwrap();
            assert!(matches!(strategy, DevicePlacementStrategy::CompiledReshard { .. }), "{source:?}");
            let target = Placement::new(mesh.clone(), target_sharding.clone()).unwrap();
            let placed = input.clone().reshard_on_devices(&session, &mesh, target_sharding, false);
            for placed in [placed, input.to_placement(&session, target)] {
                assert_eq!(shard_values(&placed.unwrap()), expected, "{source:?}");
            }
        }

        // A column-major `f32[4, 2]` split along `x` and imported with the buffers of the devices at `y = 1` holds one
        // local replica of every tile. Replicating it on a differently named mesh over the same devices gathers it on
        // its own mesh first, which reads the buffer of every device, so the gather first copies the local replica of
        // the same tile to each device without one, through a placement, an executable, and an eager call. An input
        // without any local buffer of the second tile cannot be gathered.
        let auto_mesh = |first: &str, second: &str| {
            let axes = vec![
                MeshAxis::new(first, 2, MeshAxisType::Auto).unwrap(),
                MeshAxis::new(second, 2, MeshAxisType::Auto).unwrap(),
            ];
            DeviceMesh::new(LogicalMesh::new(axes).unwrap(), mesh.devices().to_vec()).unwrap()
        };
        let (source_mesh, target_mesh) = (auto_mesh("x", "y"), auto_mesh("q", "r"));
        let split = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
        let split = Sharding::new(source_mesh.logical_mesh().clone(), split).unwrap();
        let replicated = Sharding::replicated(target_mesh.logical_mesh().clone(), 2);
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()));
        let r#type = ArrayType::new_static(DataType::F32, [4, 2]).with_layout(layout);
        let values = (1..=8).map(|value| value as f32).collect::<Vec<_>>();
        let bytes = values_to_bytes(&values);
        let sharding = Some(split.clone());
        let original = Array::from_host_buffer(&session, r#type.clone(), source_mesh.clone(), sharding, bytes).unwrap();
        let import = |indices: &[usize]| {
            let shards = original.shards();
            let buffers = indices.iter().map(|&index| {
                let buffer = shards[index].buffer().unwrap();
                buffer.copy_to_memory(buffer.memory().unwrap()).unwrap()
            });
            let (sharding, buffers) = (Some(split.clone()), buffers.collect());
            Array::from_addressable_buffers(&session, r#type.clone(), source_mesh.clone(), sharding, buffers).unwrap()
        };
        let input = import(&[1, 3]);
        let layout_resolver = cross_host_layout_resolver(client);
        let strategy = input.device_placement_strategy(&layout_resolver, &target_mesh, &replicated, 0).unwrap();
        assert!(matches!(strategy, DevicePlacementStrategy::CompiledReshardAcrossMeshes { gather: Some(_), .. }));
        let options = Options::new(target_mesh.clone()).with_input_shardings(vec![replicated.clone()]);
        let identity: CompiledFunction<ArrayType, ArrayType> =
            crate::compile_with_options(|input| input, r#type.clone(), &session, options).unwrap();
        let target = Placement::new(target_mesh.clone(), replicated.clone()).unwrap();
        let placed = [
            input.clone().reshard_on_devices(&session, &target_mesh, &replicated, false),
            input.to_placement(&session, target),
            identity.executable().execute(&session, input.clone()),
        ];
        for placed in placed {
            assert_eq!(shard_values(&placed.unwrap()), vec![values.clone(); 4]);
        }
        assert_eq!(shard_values(&identity.call(input).unwrap()), vec![values.clone(); 4]);
        let missing = import(&[1]).reshard_on_devices(&session, &target_mesh, &replicated, false);
        assert_eq!(missing.unwrap_err().to_string(), "array shard #2 on device 2 has no addressable buffer");
    }

    #[test]
    fn test_array_reshard_on_devices_keeps_declared_layouts() {
        // A column split `f32[4, 4]` in a declared layout and memory that is resharded to replicated over two devices
        // keeps that layout and memory, so a two-device identity that requires a replicated `f32[4, 4]` in them takes
        // that input through its executable and through an eager call, which reshards the input with
        // `Array::reshard_on_devices`.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let logical_mesh = mesh.logical_mesh().clone();
        let replicated = Sharding::replicated(logical_mesh.clone(), 2);
        let columns = vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])];
        let columns = Sharding::new(logical_mesh, columns).unwrap();
        let values = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
        let read = |output: Array| values_from_bytes::<f32>(&output.copy_to_host_bytes().unwrap());
        let mut outcomes = Vec::new();
        for minor_to_major in [vec![0, 1], vec![1, 0]] {
            for memory in [Memory::Device, Memory::Host { pinned: true }] {
                let layout = Layout::Tiled(TiledLayout::new(minor_to_major.clone(), Vec::new()));
                let r#type =
                    ArrayType::new_static(DataType::F32, [4, 4]).with_layout(layout.clone()).with_memory(memory);
                let options = Options::new(mesh.clone()).with_input_shardings(vec![replicated.clone()]);
                let identity: CompiledFunction<ArrayType, ArrayType> =
                    crate::compile_with_options(|input| input, r#type.clone(), &session, options).unwrap();
                let bytes = values_to_bytes(&values);
                let input = Array::from_host_buffer(&session, r#type, mesh.clone(), Some(columns.clone()), bytes);
                let input = input.unwrap();
                let resharded = input.clone().reshard_on_devices(&session, &mesh, &replicated, false).unwrap();
                assert_eq!(resharded.layout(), Some(&layout), "{minor_to_major:?} {memory:?}");
                assert_eq!(resharded.r#type().memory(), memory, "{minor_to_major:?} {memory:?}");
                let direct = identity.executable().execute(&session, input.clone());
                let direct = direct.map(read).map_err(|error| error.to_string());
                let called = identity.call(input).map(read).map_err(|error| error.to_string());
                outcomes.push((minor_to_major.clone(), memory, direct, called));
            }
        }
        let (pinned, expected) = (Memory::Host { pinned: true }, Ok(values.clone()));
        assert_eq!(
            outcomes,
            [
                (vec![0, 1], Memory::Device, expected.clone(), expected.clone()),
                (vec![0, 1], pinned, expected.clone(), expected.clone()),
                (vec![1, 0], Memory::Device, expected.clone(), expected.clone()),
                (vec![1, 0], pinned, expected.clone(), expected),
            ],
        );
    }

    // Worker of `test_array_reshard_on_devices_keeps_declared_layout_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_array_reshard_on_devices_keeps_declared_layout_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            let (_, devices, session) = cuda_rank_session(client);
            let source_mesh = explicit_mesh(&devices[..1]);
            let mesh = explicit_mesh(&devices);
            let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
            let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
            let r#type = ArrayType::new_static(DataType::F32, [2, 3])
                .with_layout(column_major.clone())
                .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 2))
                .unwrap();
            let values = values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]);
            let source = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();

            // Resharding the source onto both devices (i.e., `reshard_on_devices`) copies the column-major replica of
            // rank 0 to its own device and sends it to rank 1, which receives it in the default row-major layout. The
            // result has the declared layout of its input on both ranks, as a placement that changes only the devices
            // promises, so rank 1 converts the buffer that it received, and both ranks hold column-major buffers and
            // prepare the result for the compiled function below in the same way. Each shard is described by whether
            // its buffer is column-major and by its dense row-major values.
            let moved = source.clone().reshard_on_devices(&session, &mesh, &replicated, false).unwrap();
            moved.block_until_ready().unwrap();
            let shards = moved
                .addressable_shards()
                .map(|shard| {
                    let layout = shard.buffer().unwrap().specification().unwrap().layout;
                    (
                        layout == column_major.to_pjrt().unwrap(),
                        values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(moved.layout(), Some(&column_major));
            assert_eq!(shards, [(true, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);

            // A compiled function on both devices takes the moved array, whose buffers differ in layout between the
            // ranks, and both ranks prepare it in the same way. It rejects the source on both ranks before either moves
            // anything, because the source is committed to the device of rank 0 alone and a fixed device assignment
            // constrains committed public arguments.
            let input_type = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(replicated).unwrap();
            let compiled =
                crate::compile::<_, ArrayType, ArrayType>(|x| x.clone() + x, input_type, &session, mesh).unwrap();
            let rejected = compiled.executable().execute(&session, source).err().map(|error| error.to_string());
            let expected = Error::Execution(ExecutionError::CommittedInputDeviceMismatch {
                index: 0,
                actual: vec![devices[0].id()],
                expected: devices.iter().map(Device::id).collect(),
            });
            assert_eq!(rejected, Some(expected.to_string()));
            let output = compiled.executable().execute(&session, moved).unwrap();
            output.block_until_ready().unwrap();
            assert_eq!(read_addressable_f32s(&output), [vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]]);
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_reshard_on_devices_keeps_declared_layout_on_two_cuda_processes() {
        run_cuda_rank_pair(
            "arrays::transfers::tests::test_array_reshard_on_devices_keeps_declared_layout_cuda_process_helper",
        );
    }

    #[test]
    fn test_array_reshard_on_devices_commits_results_on_multiple_devices() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let second_mesh = DeviceMesh::new(logical_mesh.clone(), vec![mesh.devices()[1]]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let source = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        assert_eq!(source.mesh().devices(), [mesh.devices()[0]]);
        assert!(!source.is_committed());

        // An uncommitted array that is resharded over both devices is committed to them.
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let output = source.clone().reshard_on_devices(&session, &mesh, &sharded, false).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1.], vec![2.]]);

        // An uncommitted array that is moved to a single device stays uncommitted.
        let replicated = Sharding::replicated(logical_mesh, 1);
        let output = source.clone().reshard_on_devices(&session, &second_mesh, &replicated, false).unwrap();
        assert_eq!(output.mesh(), &second_mesh);
        assert!(!output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);

        // Both results stay on devices of this process, so they keep the locally assigned devices of the source, and a
        // source whose devices every process knows gives results whose devices every process knows.
        assert!(source.is_locally_assigned());
        assert!(output.is_locally_assigned());
        let shared = source.with_locally_assigned(false);
        let output = shared.reshard_on_devices(&session, &mesh, &sharded, false).unwrap();
        assert!(!output.is_locally_assigned());
    }

    #[test]
    fn test_array_with_upload_completion() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);

        // The CPU plugin has transfer managers, so the blocking host fallback that this function completes cannot be
        // forced there. The replicas of this array stand in for its uploads, and only the second one fails.
        let (uploaded, managers) = pending_replicated_vector(&session, &mesh);
        let (producer_event, producer) = client.event(()).unwrap();
        let completed =
            uploaded.clone().with_upload_completion(Some(ExecutionFence::new(vec![producer_event]))).unwrap();
        assert_eq!(completed.mesh(), &mesh);
        assert_eq!(completed.sharding(), uploaded.sharding());
        assert_eq!(completed.r#type(), uploaded.r#type());
        assert!(
            completed
                .shards()
                .iter()
                .zip(uploaded.shards())
                .all(|(completed, uploaded)| Arc::ptr_eq(completed.buffer().unwrap(), uploaded.buffer().unwrap())),
        );
        assert!(matches!(completed.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))));
        managers[0]
            .transfer_data(0, Arc::new(values_to_bytes(&[1f32, 2., 3., 4.])), 0, true)
            .unwrap()
            .r#await()
            .unwrap();
        managers[1].set_error(0, PjrtError::aborted("second upload failed")).unwrap();
        producer.set(None).unwrap();

        // A readback of the plain upload reads only the first replica and succeeds, while every observation of the
        // completed array reports the failed upload of the second one.
        assert_eq!(values_from_bytes::<f32>(&uploaded.copy_to_host_bytes().unwrap()), [1., 2., 3., 4.]);
        assert!(matches!(
            completed.copy_to_host_bytes(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second upload failed",
        ));
        assert!(matches!(
            completed.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second upload failed",
        ));

        // A failed source of the placement fails the completed array too, even when every upload succeeded.
        let (uploaded, managers) = pending_replicated_vector(&session, &mesh);
        let (producer_event, producer) = client.event(()).unwrap();
        let completed = uploaded.with_upload_completion(Some(ExecutionFence::new(vec![producer_event]))).unwrap();
        for manager in &managers {
            manager
                .transfer_data(0, Arc::new(values_to_bytes(&[1f32, 2., 3., 4.])), 0, true)
                .unwrap()
                .r#await()
                .unwrap();
        }
        producer.set(Some(PjrtError::aborted("source failed"))).unwrap();
        assert!(matches!(
            completed.copy_to_host_bytes(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "source failed",
        ));
    }

    #[test]
    fn test_array_copy_shard_through_host() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let values = [1f32, 2., 3., 4., 5., 6.];
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![])).to_pjrt().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]).with_layout(column_major);
        let source = Array::from_host_buffer(&session, r#type, mesh, None, values_to_bytes(&values)).unwrap();
        let shard = source.addressable_shards().next().unwrap();
        let buffer = shard.buffer().unwrap();

        // A conversion that the plugin cannot compile copies the shard through host memory into a new buffer in the
        // requested layout and memory, which compiles nothing, and the completion records the readiness of that buffer.
        let mut completion = PlacementCompletion::default();
        let memory = buffer.memory().unwrap();
        let copied = Array::copy_shard_through_host(
            &session,
            buffer,
            shard.shape(),
            BufferType::F32,
            memory,
            Some(row_major.clone()),
            &mut completion,
        )
        .unwrap();
        completion.join(None).unwrap().block_until_ready().unwrap();
        assert_eq!(session.cached_program_count(), 0);
        assert!(!Arc::ptr_eq(&copied, buffer));
        assert_eq!(copied.memory().unwrap().kind().unwrap(), buffer.memory().unwrap().kind().unwrap());
        assert_eq!(copied.specification().unwrap().layout, row_major);
        assert_eq!(copied.copy_to_host(None).unwrap().r#await().unwrap(), values_to_bytes(&values));
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_copy_shard_through_host_completes_each_shard_independently() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let managers = transfer_managers(client, BufferSpecification::new(BufferType::F32, [2u64, 2]));
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])).to_pjrt().unwrap();
        let mut completion = PlacementCompletion::default();
        let copies = managers
            .iter()
            .map(|manager| {
                let buffer = Arc::new(manager.retrieve_buffer(0).unwrap());
                let memory = buffer.memory().unwrap();
                let shape = StaticShape::new(vec![2, 2]);
                let layout = Some(layout.clone());
                Array::copy_shard_through_host(
                    &session,
                    &buffer,
                    shape,
                    BufferType::F32,
                    memory,
                    layout,
                    &mut completion,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        // The copies are returned before their sources are defined, in the requested layout and memory, and without
        // compiling any program.
        assert_eq!(session.cached_program_count(), 0);
        for copy in &copies {
            assert_eq!(copy.specification().unwrap().layout, layout);
            assert_eq!(copy.memory().unwrap(), copy.device().unwrap().default_memory().unwrap());
            assert_eq!(copy.ready().unwrap().ready(), Ok(false));
        }

        // The background worker reads the first source back once it is defined and uploads it in the new layout, while
        // the source of the second shard is still pending, so that a producer of that source may depend on the first
        // copy.
        let (sender, receiver) = mpsc::channel();
        let first_copy_ready = copies[0].ready().unwrap();
        thread::spawn(move || sender.send(first_copy_ready.r#await()).unwrap());
        managers[0]
            .transfer_data(0, Arc::new(values_to_bytes(&[1f32, 2., 3., 4.])), 0, true)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(receiver.recv_timeout(Duration::from_secs(60)), Ok(Ok(())));
        let fence = completion.join(None).unwrap();
        assert_eq!(fence.is_ready(), Ok(false));
        let row_major = ryft_pjrt::Layout::dense_major_to_minor(2);
        let first_copy = copies[0].copy_to_host(Some(row_major)).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f32>(&first_copy), [1., 2., 3., 4.]);

        // The failed readback of the second source poisons its copy with its error, and the completion still joins
        // that copy, so it reports the failure, as the session does once.
        managers[1].set_error(0, PjrtError::aborted("second source failed")).unwrap();
        assert!(matches!(
            copies[1].ready().unwrap().r#await(),
            Err(PjrtError::Aborted { message, .. }) if message == "second source failed",
        ));
        assert!(matches!(
            fence.block_until_ready(),
            Err(PjrtError::Aborted { message, .. }) if message == "second source failed",
        ));
        assert!(matches!(
            session.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(_, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "second source failed"),
        ));
    }

    #[test]
    fn test_array_copy_shard_through_host_reports_failed_source_without_elements() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();

        // A shard without elements issues no copy, but its round trip still waits for its source and fails with it.
        let manager = transfer_managers(client, BufferSpecification::new(BufferType::F32, [0u64, 2])).remove(0);
        let buffer = Arc::new(manager.retrieve_buffer(0).unwrap());
        let mut completion = PlacementCompletion::default();
        let shape = StaticShape::new(vec![0, 2]);
        let memory = buffer.memory().unwrap();
        let copy =
            Array::copy_shard_through_host(&session, &buffer, shape, BufferType::F32, memory, None, &mut completion)
                .unwrap();
        manager.set_error(0, PjrtError::aborted("source failed")).unwrap();
        assert!(matches!(
            copy.ready().unwrap().r#await(),
            Err(PjrtError::Aborted { message, .. }) if message == "source failed",
        ));
        assert!(matches!(
            completion.join(None).unwrap().block_until_ready(),
            Err(PjrtError::Aborted { message, .. }) if message == "source failed",
        ));
        assert!(matches!(
            session.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(_, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "source failed"),
        ));
    }

    #[test]
    fn test_array_place_exact_shards_prepares_cross_host_sends() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote = [Device::new(local.id() + 1, 1), Device::new(local.id() + 2, 1)];
        let logical_mesh =
            |size| LogicalMesh::new(vec![MeshAxis::new("x", size, MeshAxisType::Explicit).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh(1), vec![local]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh(3), vec![local, remote[0], remote[1]]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let memory = Memory::Host { pinned: true };
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]).with_layout(column_major).with_memory(memory);
        let values = values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]);
        let source = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();
        let sources = source.shards().iter().map(ArrayShard::descriptor);
        let (source_mesh, source_sharding, shape) = (source.mesh(), source.sharding(), source.shape());
        let plan =
            plan_exact_shard_moves(BufferType::F32, sources, source_mesh, source_sharding, &shape, &mesh, &replicated);
        let plan = plan.unwrap().unwrap();
        assert_eq!(
            plan.moves_with_role(0, MoveRole::Send).map(|movement| movement.target_device).collect::<Vec<_>>(),
            remote,
        );
        let cross_host_layouts = HashMap::from([(StaticShape::new(vec![2, 3]), row_major.to_pjrt().unwrap())]);

        // The only shard of the source is sent to both remote devices. It is staged in device memory and converted to
        // the layout of the receives once, and both sends read the same buffer. The CPU plugin lacks the cross-host
        // transfers extension, so the first send fails once every buffer that the sends read is prepared. The result
        // would wait for the readiness of the sent shard and of its staged copy and for the fence of the conversion.
        let error = source.place_exact_shards(&mesh, &replicated, memory, &plan, &cross_host_layouts).err();
        assert!(matches!(
            error,
            Some(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));
        assert_eq!(session.cached_program_count(), 1);
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_place_exact_shards_records_layout_across_processes() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(20, 2), Device::new(21, 2)],
        )
        .unwrap();
        let sharding = Sharding::new(
            target_mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
        )
        .unwrap();
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let received_row_major = row_major.to_pjrt().unwrap();
        let received_column_major = column_major.to_pjrt().unwrap();
        let recorded_layout = |layout: Option<&Layout>, receives: &[&ryft_pjrt::Layout]| {
            let r#type = ArrayType::new_static(DataType::F32, [3, 3]).with_layout(layout.cloned());
            let source = Array::from_host_buffer(&session, r#type, mesh.clone(), None, [0u8; 36]).unwrap();
            // These moves involve other processes only. This process still records their layouts, without posting
            // any PJRT transfers of its own.
            let moves = receives.iter().enumerate().map(|(index, _)| ShardMove {
                source_shard_index: index,
                source_device: Device::new(index + 10, 1),
                target_shard_index: index,
                target_device: Device::new(index + 20, 2),
                slice: vec![[0..2, 2..3][index].clone(), 0..3],
                transfer_key: index as CrossHostTransferKey,
            });
            let plan = ShardMovePlan::new(BufferType::F32, moves.collect()).unwrap();
            let layouts = plan.moves.iter().zip(receives).map(|(movement, &layout)| (movement.shape(), layout.clone()));
            let placed = source
                .place_exact_shards(&target_mesh, &sharding, Memory::Device, &plan, &layouts.collect())
                .unwrap();
            placed.layout().cloned()
        };

        assert_eq!(recorded_layout(None, &[&received_row_major]), None);
        assert_eq!(recorded_layout(Some(&column_major), &[]), Some(column_major.clone()));
        assert_eq!(recorded_layout(Some(&row_major), &[&received_row_major]), Some(row_major));
        assert_eq!(recorded_layout(Some(&column_major), &[&received_column_major, &received_row_major]), None);
    }

    #[test]
    fn test_array_buffer_on_send_device() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let process_index = client.process_index().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let devices = mesh.devices().to_vec();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let values = values_to_bytes(&[1f32, 2., 3.]);

        // The array holds only its replica on the second device in each memory. A send that the plan assigns to the
        // first device copies that replica to the first device, in the memory that the type declares, because the
        // receiver pairs the transfer with the first device.
        for (memory, kind) in [(Memory::Device, "device"), (Memory::Host { pinned: true }, "pinned_host")] {
            let target = client_devices[1].memory(memory).unwrap();
            let buffer = client.buffer(values.as_slice(), BufferType::F32, [3u64], None, target, None).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [3]).with_memory(memory);
            let source =
                Array::from_addressable_buffers(&session, r#type, mesh.clone(), Some(replicated.clone()), vec![buffer])
                    .unwrap();
            assert!(source.shards()[0].buffer().is_none());
            let held = Arc::clone(source.shards()[1].buffer().unwrap());

            let mut completion = PlacementCompletion::default();
            let replicas = LocalReplicas::new(process_index);
            let sent = source.buffer_on_send_device(&replicas, 0, &client_devices[0], &mut completion).unwrap();
            completion.join(None).unwrap().block_until_ready().unwrap();
            assert!(!Arc::ptr_eq(&sent, &held));
            assert_eq!(sent.device().unwrap().id().unwrap(), devices[0].id());
            assert_eq!(sent.memory().unwrap().kind().unwrap(), kind);
            assert_eq!(sent.dimensions().unwrap(), [3]);
            assert_eq!(sent.copy_to_host(None).unwrap().r#await().unwrap(), values);

            // The shard that holds a buffer sends it without a copy, and the result of the placement then waits for
            // and reports its readiness, which may be the only completion of the work that produced it.
            let mut completion = PlacementCompletion::default();
            let sent = source.buffer_on_send_device(&replicas, 1, &client_devices[1], &mut completion).unwrap();
            assert!(Arc::ptr_eq(&sent, &held));
            let fence = completion.join(None).unwrap();
            fence.block_until_ready().unwrap();
        }

        // A shard whose slice has no buffer on this process cannot be sent.
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let buffer = client.buffer([7u8].as_slice(), BufferType::U8, [1u64], None, client_devices[0].clone(), None);
        let source = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::U8, [2]),
            mesh,
            Some(sharded),
            vec![buffer.unwrap()],
        )
        .unwrap();
        let mut completion = PlacementCompletion::default();
        let error = source
            .buffer_on_send_device(&LocalReplicas::new(process_index), 1, &client_devices[1], &mut completion)
            .err()
            .unwrap();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(
                error.as_ref(),
                ArrayError::MissingShardBuffer { shard_index: 1, device_id } if *device_id == devices[1].id(),
            ),
        ));
        assert!(completion.join(None).is_none());
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_buffer_for_cross_host_send() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let values = [1f32, 2., 3., 4., 5., 6.];
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let default_layout = row_major.to_pjrt().unwrap();

        // A shard that is already in the receiver's layout and in device memory needs no work. Host buffers
        // must first be staged in device memory, and a shard in a different layout must be converted on its device.
        for (memory, layout, programs) in [
            (Memory::Device, &row_major, Some(0)),
            (Memory::Device, &column_major, Some(1)),
            (Memory::Host { pinned: true }, &row_major, None),
            (Memory::Host { pinned: false }, &column_major, None),
        ] {
            let r#type = ArrayType::new_static(DataType::F32, [2, 3]).with_layout(layout.clone()).with_memory(memory);
            let source =
                Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&values)).unwrap();
            let shard = source.addressable_shards().next().unwrap();
            let original = shard.buffer().unwrap();
            if memory != Memory::Device {
                assert_ne!(original.memory().unwrap().kind().unwrap(), "device");
            }
            let mut completion = PlacementCompletion::default();
            let buffer =
                Array::buffer_in_device_memory(&session, memory, Arc::clone(original), &mut completion).unwrap();
            let sent = Array::buffer_for_cross_host_send(
                &session,
                DataType::F32,
                shard.device(),
                buffer,
                shard.shape(),
                &default_layout,
                &mut completion,
            )
            .unwrap();
            let reused = memory == Memory::Device && layout == &row_major;
            assert_eq!(Arc::ptr_eq(&sent, original), reused);
            let fence = completion.join(None);
            assert_eq!(fence.is_none(), reused);
            if let Some(fence) = fence {
                fence.block_until_ready().unwrap();
            }
            if let Some(programs) = programs {
                assert_eq!(session.cached_program_count(), programs);
            }
            assert_eq!(sent.memory().unwrap().kind().unwrap(), "device");
            assert_eq!(sent.specification().unwrap().layout, default_layout);
            assert_eq!(sent.dimensions().unwrap(), [2, 3]);
            assert_eq!(sent.copy_to_host(None).unwrap().r#await().unwrap(), values_to_bytes(&values));
        }

        // A bounded-dynamic execution returns a buffer that is allocated for the bound of its dynamic dimension with a
        // trailer that records its extent, in the default layout. A plugin can take the size of a send from the
        // allocation of the sender while the receiver allocates a static buffer of the shape of the shard, so the shard
        // is sent as such a static buffer, and so is a whole shard that a region sends. At the maximum extent, the
        // physical dimensions of the dynamic buffer equal the shape of the shard, but its allocation still differs.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(1024)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let identity: CompiledFunction<ArrayType, ArrayType> =
            crate::compile_with_options(|input| input, dynamic_type, &session, Options::new(mesh.clone())).unwrap();
        for (size, padded_size) in [(2usize, 4096usize), (1023, 4096)] {
            let values = (0..size).map(|value| value as f32).collect::<Vec<_>>();
            let r#type = ArrayType::new_static(DataType::F32, [size]);
            let input = Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&values));
            let padded = identity.executable().execute(&session, input.unwrap()).unwrap();
            let shard = padded.addressable_shards().next().unwrap();
            let padded_buffer = shard.buffer().unwrap();
            assert_eq!(padded_buffer.dimensions().unwrap(), [1023]);
            assert_eq!(padded_buffer.unpadded_dimensions().unwrap(), [size as u64]);
            assert_eq!(padded_buffer.dynamic_dimensions().unwrap(), [0]);
            assert_eq!(padded_buffer.on_device_size_in_bytes().unwrap(), padded_size);
            let default_layout = dense_row_major_layout(1).to_pjrt().unwrap();
            assert_eq!(padded_buffer.specification().unwrap().layout, default_layout);
            let mut completion = PlacementCompletion::default();
            let buffer = Arc::clone(padded_buffer);
            let sent = Array::buffer_for_cross_host_send(
                &session,
                DataType::F32,
                shard.device(),
                buffer,
                shard.shape(),
                &default_layout,
                &mut completion,
            )
            .unwrap();
            let mut region_completion = PlacementCompletion::default();
            let region = Array::buffer_for_shard_region(
                &session,
                DataType::F32,
                shard,
                Arc::clone(padded_buffer),
                shard.slice(),
                Some(&default_layout),
                &mut region_completion,
            )
            .unwrap();
            completion.join(None).unwrap().block_until_ready().unwrap();
            region_completion.join(None).unwrap().block_until_ready().unwrap();
            for sent in [sent, region] {
                assert!(!Arc::ptr_eq(padded_buffer, &sent));
                assert_eq!(sent.dimensions().unwrap(), [size as u64]);
                assert_eq!(sent.dynamic_dimensions().unwrap(), Vec::<u64>::new());
                assert_eq!(sent.on_device_size_in_bytes().unwrap(), 4 * size, "extent {size}");
                assert_eq!(sent.specification().unwrap().layout, default_layout);
                assert_eq!(sent.copy_to_host(None).unwrap().r#await().unwrap(), values_to_bytes(&values));
            }
        }
    }

    #[test]
    fn test_array_place_with_strategy_returns_sources_that_a_failed_execution_cannot_take() {
        // A compiled reshard compiled for an input without a pending reduction over the auto axis `x` rejects this
        // source, which has one, when it prepares the arguments of its execution, before the launch. A source that the
        // execution cannot take (i.e., one that is not donated, or one whose buffers a clone shares) is intact then,
        // so the placement returns it for the host fallback. Only a donated source that holds its buffers exclusively
        // is not returned, because the failure of an execution does not report whether it took them.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let pending = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let program = ReshardProgram { mesh: mesh.clone(), input: replicated.clone(), output: replicated.clone() };
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let mut outcomes = Vec::new();
        for (name, donate, keep_clone) in
            [("borrowed", false, false), ("donated with a clone", true, true), ("donated exclusively", true, false)]
        {
            let contributions = [[1f32, 2.], [10., 20.]]
                .iter()
                .zip(mesh.devices())
                .map(|(values, device)| (device.id(), Arc::<[u8]>::from(values_to_bytes(values))))
                .collect::<Vec<_>>();
            let source =
                Array::from_host_shards(&session, r#type.clone(), mesh.clone(), Some(pending.clone()), contributions);
            let source = source.unwrap();
            let clone = keep_clone.then(|| source.clone());
            let strategy = DevicePlacementStrategy::CompiledReshard { program: program.clone() };
            let placed = source.place_with_strategy(strategy, 0, &mesh, &replicated, Memory::Device, None, donate);
            let Err((error, returned)) = placed else {
                panic!("{name}: the execution must reject the source");
            };
            let returned = returned.map(|source| values_from_bytes::<f32>(&source.copy_to_host_bytes().unwrap()));
            outcomes.push(format!("{name}: {returned:?}"));
            drop(clone);
            let message = "public argument #0 is unreduced along the mesh axes [\"x\"], which the executable cannot \
                           receive because a staged type does not record a pending reduction over auto mesh axes; \
                           reduce the argument (e.g., with `Array::to_placement` onto a sharding without unreduced \
                           axes), or place it over explicit mesh axes, before passing it to a compiled function or an \
                           eager operation";
            assert_eq!(error.to_string(), message, "{name}");
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                borrowed: Some([11.0, 22.0])
                donated with a clone: Some([11.0, 22.0])
                donated exclusively: None"},
        );
    }

    #[test]
    fn test_array_device_placement_strategy_rejects_owner_view_device_identifiers_that_overflow() {
        // A locally assigned value on two devices that a target with one device of the largest identifier
        // receives is planned on a view with one more device, whose identifier would exceed that one, which is an
        // error rather than an overflow.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let pair = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [2]);
        let source = Array::from_host_buffer(&session, r#type, None, Some(Sharding::replicated(pair, 1)), [1u8, 2]);
        let source = source.unwrap();
        assert_eq!((source.is_locally_assigned(), source.mesh().devices().len()), (true, 2));
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![Device::new(DeviceId::MAX, 0)]).unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        assert_eq!(
            source
                .device_placement_strategy(&simulated_cross_host_layout, &mesh, &sharding, 0)
                .err()
                .unwrap()
                .to_string(),
            format!(
                "planning a placement of a locally assigned array onto devices of process 0 needs a device identifier \
                 above {}",
                DeviceId::MAX,
            ),
        );
    }

    #[test]
    fn test_array_device_placement_strategy() {
        let client = cpu_client(1);
        assert!(matches!(client.cross_host_transfers_extension(), Err(PjrtError::Unimplemented { .. })));
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local = Device::from_pjrt(&local_device).unwrap();
        let first_remote = Device::new(local.id() + 1, local.process_index() + 1);
        let second_remote = Device::new(local.id() + 2, local.process_index() + 2);
        let session = session_with_remote_devices(client, [first_remote, second_remote]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharding = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![local, first_remote]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let local_buffer = client
            .buffer(values_to_bytes(&[1f32, 2.]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
            .unwrap();
        let held = Array::from_addressable_buffers(
            &session,
            r#type.clone(),
            source_mesh.clone(),
            Some(sharding.clone()),
            vec![local_buffer],
        )
        .unwrap();
        let metadata_only =
            Array::from_addressable_buffers(&session, r#type, source_mesh.clone(), Some(sharding.clone()), Vec::new())
                .unwrap();
        let layout_resolver = cross_host_layout_resolver(session.client());
        let process_index = local.process_index();

        // The choice reads only metadata that every process shares, so whether this process holds its buffers does not
        // change it. Keeping each shard on its device moves no shard across processes.
        for source in [&held, &metadata_only] {
            assert!(matches!(
                source.device_placement_strategy(&layout_resolver, &source_mesh, &sharding, process_index),
                Ok(DevicePlacementStrategy::ExactShards { plan, cross_host_layouts })
                    if plan.cross_host_moves().count() == 0 && cross_host_layouts.is_empty(),
            ));
        }

        // This process only copies its own shard, but the shard of the first remote process moves to the second one,
        // which needs the missing cross-host transfers extension. Every process therefore declines whole shards and
        // shard regions alike, and all of them report the missing extension, instead of this process returning after
        // its local copy while the others switch to another strategy.
        let target_mesh = DeviceMesh::new(logical_mesh, vec![local, second_remote]).unwrap();
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let sources = held.shards().iter().map(ArrayShard::descriptor);
        let plan = plan_exact_shard_moves(
            BufferType::F32,
            sources,
            held.mesh(),
            &sharding,
            &held.shape(),
            &target_mesh,
            &target_sharding,
        )
        .unwrap()
        .unwrap();
        assert_eq!(plan.moves_with_role(local.process_index(), MoveRole::Copy).count(), 1);
        assert_eq!(plan.moves_with_role(local.process_index(), MoveRole::Send).count(), 0);
        assert_eq!(plan.moves_with_role(local.process_index(), MoveRole::Receive).count(), 0);
        assert_eq!(plan.cross_host_moves().count(), 1);
        for source in [&held, &metadata_only] {
            assert!(matches!(
                source.device_placement_strategy(&layout_resolver, &target_mesh, &target_sharding, process_index),
                Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                    if message == "the cross-host transfers extension is not provided by the PJRT plugin",
            ));
        }
        assert!(matches!(
            held.to_placement(
                &session,
                Placement::new(target_mesh.clone(), target_sharding.clone()).unwrap(),
            ),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));

        // An array on devices that this process chose on its own holds a full value that every process holds on
        // devices of its own choice, so it is placed with local work only, whether or not it is committed, by every
        // process when the target spans several processes. A process that owns no device of a target on
        // devices of one other process (e.g., the second remote process for a target on this process) takes no
        // part in the strategy of that owner. The same array on devices that every process knows plans its transfers
        // as usual.
        let uploaded = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            None,
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let local_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![local],
        )
        .unwrap();
        let local_sharding = Sharding::replicated(local_mesh.logical_mesh().clone(), 1);
        for source in [uploaded.clone(), uploaded.clone().with_committed(true)] {
            assert!(source.is_locally_assigned());
            let (mesh, sharding) = (&target_mesh, &target_sharding);
            let strategy = source.device_placement_strategy(&layout_resolver, mesh, sharding, process_index);
            assert!(matches!(strategy, Ok(DevicePlacementStrategy::ProcessLocalValue)));
            assert!(matches!(
                source.device_placement_strategy(
                    &layout_resolver,
                    &local_mesh,
                    &local_sharding,
                    second_remote.process_index(),
                ),
                Ok(DevicePlacementStrategy::OnOwner { strategy })
                    if matches!(*strategy, DevicePlacementStrategy::ExactShards { .. }),
            ));
        }
        assert!(matches!(
            uploaded.clone().with_locally_assigned(false).device_placement_strategy(
                &layout_resolver,
                &target_mesh,
                &target_sharding,
                process_index,
            ),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));

        // A target on devices of this process alone moves nothing between processes, so this process places such
        // an array there with the strategy of an array on devices that every process knows.
        assert!(matches!(
            uploaded.device_placement_strategy(&layout_resolver, &local_mesh, &local_sharding, process_index),
            Ok(DevicePlacementStrategy::ExactShards { plan, cross_host_layouts })
                if plan.cross_host_moves().count() == 0 && cross_host_layouts.is_empty(),
        ));
    }

    #[test]
    fn test_array_device_placement_strategy_receives_executable_buffer_type() {
        let client = cpu_client(1);
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local = Device::from_pjrt(&local_device).unwrap();
        let remote = Device::new(local.id() + 1, 1);
        let session = session_with_remote_devices(client, [remote]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![local, remote]).unwrap();
        let sharding = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_source_buffer =
            client.buffer([1u8, 0].as_slice(), BufferType::Predicate, [2u64], None, local_device, None).unwrap();
        let source = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::I1, [4]),
            source_mesh,
            Some(sharding.clone()),
            vec![local_source_buffer],
        )
        .unwrap();

        // The sender transfers the predicate buffer that holds one-bit integers, so the receiver requests the same
        // buffer type rather than the logical one-bit integer type.
        let target_mesh = DeviceMesh::new(logical_mesh, vec![remote, local]).unwrap();
        let strategy = source.device_placement_strategy(
            &simulated_cross_host_layout,
            &target_mesh,
            &sharding,
            local.process_index(),
        );
        let Ok(DevicePlacementStrategy::ExactShards { plan, .. }) = strategy else {
            panic!("swapping the shards of two processes must move whole shards");
        };
        assert_eq!(plan.element_type, BufferType::Predicate);
        assert_eq!(
            plan.moves_with_role(local.process_index(), MoveRole::Receive).collect::<Vec<_>>(),
            [&ShardMove {
                source_shard_index: 1,
                source_device: remote,
                target_shard_index: 1,
                target_device: local,
                slice: vec![2..4],
                transfer_key: 3
            }],
        );
    }

    #[test]
    fn test_array_device_placement_strategy_agrees_across_processes() {
        let client = cpu_client(2);
        let local = Session::builder().client(client).build().unwrap().addressable_devices().unwrap();
        assert_eq!(local.iter().map(Device::process_index).collect::<Vec<_>>(), [0, 0]);
        let next_id = local.iter().map(Device::id).max().unwrap() + 1;
        let remote = [Device::new(next_id, 1), Device::new(next_id + 1, 1)];
        let session = session_with_remote_devices(client, remote);
        let replicated = |mesh: &DeviceMesh| Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let sharded = |mesh: &DeviceMesh| {
            let dimensions = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
            Sharding::new(mesh.logical_mesh().clone(), dimensions).unwrap()
        };
        let r#type = ArrayType::new_static(DataType::F32, [2, 2]);
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let values = values_to_bytes(&[1f32, 2., 3., 4.]);

        // A process that holds a source with locally assigned devices sees its own devices, which are the remote
        // devices for the second process, while every process sees the same metadata of a shared source.
        let locally_assigned = |committed: bool, sharding: Option<Sharding>, devices: &[Device]| {
            let own = Array::from_host_buffer(&session, r#type.clone(), None, sharding.clone(), &values).unwrap();
            let own = own.with_committed(committed);
            let mesh = explicit_mesh(devices);
            let sharding = sharding.map(|sharding| sharding.with_mesh(mesh.logical_mesh().clone()).unwrap());
            let mut peer =
                Array::from_addressable_buffers(&session, r#type.clone(), mesh, sharding, Vec::new()).unwrap();
            peer.committed = committed;
            peer.locally_assigned = true;
            vec![own, peer]
        };
        // A process that imported only some of its shards holds a buffer on the second local device alone, in the
        // layout and the memory of the type, with the elements of its shard there, which `values` lists in row-major
        // order.
        let partially_addressable = |r#type: &ArrayType,
                                     devices: &[Device],
                                     sharding: &dyn Fn(&DeviceMesh) -> Sharding,
                                     dimensions: [u64; 2],
                                     values: &[f32]| {
            let mesh = explicit_mesh(devices);
            let device = client.addressable_devices().unwrap()[1].clone();
            let layout = r#type.layout().map(|layout| layout.to_pjrt().unwrap());
            let bytes = values_to_bytes(values);
            let buffer = client.buffer(bytes.as_slice(), BufferType::F32, dimensions, None, device.clone(), layout);
            let buffer = buffer.unwrap().copy_to_memory(device.memory(r#type.memory()).unwrap()).unwrap();
            let sharding = Some(sharding(&mesh));
            let own =
                Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), sharding.clone(), vec![buffer]);
            let peer = Array::from_addressable_buffers(&session, r#type.clone(), mesh, sharding, Vec::new());
            vec![own.unwrap(), peer.unwrap()]
        };
        let multi_device_sharding = sharded(&explicit_mesh(&local));
        let sources = [
            (
                "shared replicated over both processes",
                shared_views(&session, &r#type, &[local[0], remote[0]], &replicated),
            ),
            ("shared sharded over both processes", shared_views(&session, &r#type, &[local[0], remote[0]], &sharded)),
            ("shared on process 0", shared_views(&session, &r#type, &[local[0]], &replicated)),
            (
                "shared with a declared layout",
                shared_views(&session, &r#type.clone().with_layout(column_major.clone()), &[local[0]], &replicated),
            ),
            (
                "shared sharded with a declared layout",
                shared_views(&session, &r#type.clone().with_layout(column_major.clone()), &[local[0]], &sharded),
            ),
            (
                "shared in pinned host memory",
                shared_views(
                    &session,
                    &r#type.clone().with_memory(Memory::Host { pinned: true }),
                    &[local[0]],
                    &replicated,
                ),
            ),
            (
                "partially addressable import",
                partially_addressable(
                    &r#type,
                    &[local[0], local[1], remote[0]],
                    &replicated,
                    [2, 2],
                    &[1., 2., 3., 4.],
                ),
            ),
            (
                "partially addressable import with a declared layout",
                partially_addressable(
                    &r#type.clone().with_layout(column_major.clone()),
                    &[local[0], local[1]],
                    &replicated,
                    [2, 2],
                    &[1., 2., 3., 4.],
                ),
            ),
            (
                "partially addressable import in pinned host memory",
                partially_addressable(
                    &r#type.clone().with_memory(Memory::Host { pinned: true }),
                    &[local[0], local[1]],
                    &replicated,
                    [2, 2],
                    &[1., 2., 3., 4.],
                ),
            ),
            (
                "partially addressable sharded import",
                partially_addressable(&r#type, &[local[0], local[1]], &sharded, [1, 2], &[3., 4.]),
            ),
            ("process-local committed", locally_assigned(true, None, &remote[..1])),
            ("process-local uncommitted", locally_assigned(false, None, &remote[..1])),
            ("process-local on two devices", locally_assigned(true, Some(multi_device_sharding), &remote)),
        ];
        let targets = [
            (
                "replicated over both processes",
                explicit_mesh(&[local[0], remote[0]]),
                replicated(&explicit_mesh(&[local[0], remote[0]])),
            ),
            (
                "sharded over both processes",
                explicit_mesh(&[local[0], remote[0]]),
                sharded(&explicit_mesh(&[local[0], remote[0]])),
            ),
            (
                "unreduced over both processes",
                explicit_mesh(&[local[0], remote[0]]),
                replicated(&explicit_mesh(&[local[0], remote[0]])).with_unreduced_axes(["x"]).unwrap(),
            ),
            ("replicated on process 0", explicit_mesh(&local), replicated(&explicit_mesh(&local))),
            ("sharded on process 1", explicit_mesh(&remote), sharded(&explicit_mesh(&remote))),
        ];
        let outcomes = sources
            .iter()
            .flat_map(|(source, views)| {
                targets.iter().map(move |(target, mesh, sharding)| {
                    let outcome = match placement_agreement(views, mesh, sharding) {
                        Ok(strategy) => strategy,
                        Err(disagreement) => format!("DISAGREE: {disagreement}"),
                    };
                    format!("{source} -> {target}: {outcome}")
                })
            })
            .collect::<Vec<_>>();
        // Every transfer and program that one process plans has its counterpart on the other process. The processes
        // choose the same strategy, except that a process whose own value lies on its devices and whose devices hold
        // the whole target places it with the strategy of a shared source, which moves nothing between processes.
        // A partial import reaches its first transfer in production whenever another local replica serves a missing one
        // or a missing tile is only copied locally, which in pinned host memory includes the staging of each sent
        // replica in device memory. A disagreement names the strategy of each process. The partially addressable
        // sharded import has no local buffer at all for its first tile, so process 0 stops before its transfers
        // wherever that tile must leave the process or feed a gather, while process 1 waits for its receive, which is
        // the accepted limit of the PJRT cross-host transfers that `.tasks/plan_placement_decisions.md` records. The
        // target with a pending sum is the exception, because a placement that introduces one puts the value
        // only on the devices at coordinate zero along the summed axis, which are those of process 0 here, so process
        // 1 only writes its additive identity, receives nothing, and waits for nothing.
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                shared replicated over both processes -> replicated over both processes: exact shards
                shared replicated over both processes -> sharded over both processes: compiled reshard
                shared replicated over both processes -> unreduced over both processes: additive identities
                shared replicated over both processes -> replicated on process 0: exact shards
                shared replicated over both processes -> sharded on process 1: shard regions
                shared sharded over both processes -> replicated over both processes: compiled reshard
                shared sharded over both processes -> sharded over both processes: exact shards
                shared sharded over both processes -> unreduced over both processes: shard regions with additive identities
                shared sharded over both processes -> replicated on process 0: shard regions
                shared sharded over both processes -> sharded on process 1: exact shards
                shared on process 0 -> replicated over both processes: exact shards
                shared on process 0 -> sharded over both processes: shard regions
                shared on process 0 -> unreduced over both processes: exact shards with additive identities
                shared on process 0 -> replicated on process 0: exact shards
                shared on process 0 -> sharded on process 1: shard regions
                shared with a declared layout -> replicated over both processes: exact shards
                shared with a declared layout -> sharded over both processes: compiled reshard across meshes
                shared with a declared layout -> unreduced over both processes: exact shards with additive identities
                shared with a declared layout -> replicated on process 0: exact shards
                shared with a declared layout -> sharded on process 1: compiled reshard across meshes
                shared sharded with a declared layout -> replicated over both processes: exact shards
                shared sharded with a declared layout -> sharded over both processes: compiled reshard across meshes
                shared sharded with a declared layout -> unreduced over both processes: exact shards with additive identities
                shared sharded with a declared layout -> replicated on process 0: exact shards
                shared sharded with a declared layout -> sharded on process 1: compiled reshard across meshes
                shared in pinned host memory -> replicated over both processes: exact shards
                shared in pinned host memory -> sharded over both processes: shard regions
                shared in pinned host memory -> unreduced over both processes: exact shards with additive identities
                shared in pinned host memory -> replicated on process 0: exact shards
                shared in pinned host memory -> sharded on process 1: shard regions
                partially addressable import -> replicated over both processes: exact shards
                partially addressable import -> sharded over both processes: shard regions
                partially addressable import -> unreduced over both processes: exact shards with additive identities
                partially addressable import -> replicated on process 0: exact shards
                partially addressable import -> sharded on process 1: shard regions
                partially addressable import with a declared layout -> replicated over both processes: exact shards
                partially addressable import with a declared layout -> sharded over both processes: compiled reshard across meshes
                partially addressable import with a declared layout -> unreduced over both processes: exact shards with additive identities
                partially addressable import with a declared layout -> replicated on process 0: exact shards
                partially addressable import with a declared layout -> sharded on process 1: compiled reshard across meshes
                partially addressable import in pinned host memory -> replicated over both processes: exact shards
                partially addressable import in pinned host memory -> sharded over both processes: shard regions
                partially addressable import in pinned host memory -> unreduced over both processes: exact shards with additive identities
                partially addressable import in pinned host memory -> replicated on process 0: exact shards
                partially addressable import in pinned host memory -> sharded on process 1: shard regions
                partially addressable sharded import -> replicated over both processes: DISAGREE: shard regions: process 0 fails before it posts its transfers: array shard #0 on device 0 has no addressable buffer
                partially addressable sharded import -> sharded over both processes: exact shards
                partially addressable sharded import -> unreduced over both processes: shard regions with additive identities
                partially addressable sharded import -> replicated on process 0: compiled reshard
                partially addressable sharded import -> sharded on process 1: DISAGREE: exact shards: process 0 fails before it posts its transfers: array shard #0 on device 0 has no addressable buffer
                process-local committed -> replicated over both processes: process-local value
                process-local committed -> sharded over both processes: process-local value
                process-local committed -> unreduced over both processes: process-local value
                process-local committed -> replicated on process 0: exact shards | exact shards on its owner
                process-local committed -> sharded on process 1: shard regions on its owner | shard regions
                process-local uncommitted -> replicated over both processes: process-local value
                process-local uncommitted -> sharded over both processes: process-local value
                process-local uncommitted -> unreduced over both processes: process-local value
                process-local uncommitted -> replicated on process 0: exact shards | exact shards on its owner
                process-local uncommitted -> sharded on process 1: shard regions on its owner | shard regions
                process-local on two devices -> replicated over both processes: process-local value
                process-local on two devices -> sharded over both processes: process-local value
                process-local on two devices -> unreduced over both processes: process-local value
                process-local on two devices -> replicated on process 0: compiled reshard | compiled reshard on its owner
                process-local on two devices -> sharded on process 1: exact shards on its owner | exact shards"},
        );

        // A pending sum over both processes that moves onto the reversed devices only moves its contributions, so both
        // processes plan whole shards, and each one sends its contribution to the other instead of completing the sum.
        let unreduced = |mesh: &DeviceMesh| replicated(mesh).with_unreduced_axes(["x"]).unwrap();
        let views = shared_views(&session, &r#type, &[local[0], remote[0]], &unreduced);
        let reversed = explicit_mesh(&[remote[0], local[0]]);
        assert_eq!(placement_agreement(&views, &reversed, &unreduced(&reversed)), Ok("exact shards".to_string()));

        // A replicated `u8[5]` split unevenly over both processes needs only a slice of the replica that each process
        // already holds, so a declared layout and a memory that each process stages its replica out of on its own
        // devices leave every process with shard regions. That includes a strided layout, which each process converts
        // on its own devices.
        let vector = ArrayType::new_static(DataType::U8, [5]);
        let split_mesh = explicit_mesh(&[local[0], remote[0]]);
        let split = Sharding::new(split_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let strided = Layout::Strided(StridedLayout::new(vec![1]));
        let outcomes = [
            ("implicit layout", vector.clone()),
            ("declared default layout", vector.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![])))),
            ("pinned host memory", vector.clone().with_memory(Memory::Host { pinned: true })),
            ("unpinned host memory", vector.clone().with_memory(Memory::Host { pinned: false })),
            ("strided layout in host memory", vector.with_layout(strided).with_memory(Memory::Host { pinned: true })),
        ]
        .iter()
        .map(|(source, r#type)| {
            let views = shared_views(&session, r#type, &[local[0], remote[0]], &|mesh| {
                Sharding::replicated(mesh.logical_mesh().clone(), 1)
            });
            let outcome = match placement_agreement(&views, &split_mesh, &split) {
                Ok(strategy) => strategy,
                Err(disagreement) => format!("DISAGREE: {disagreement}"),
            };
            format!("{source} -> uneven split: {outcome}")
        })
        .collect::<Vec<_>>();
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                implicit layout -> uneven split: shard regions
                declared default layout -> uneven split: shard regions
                pinned host memory -> uneven split: shard regions
                unpinned host memory -> uneven split: shard regions
                strided layout in host memory -> uneven split: shard regions"},
        );

        // A replicated column-major `u8[4,2]` split evenly over the manual axis `m`, and a replicated column-major
        // `u8[5,2]` split unevenly over the explicit axis `x`, on devices that process 1 owns alone, take shard
        // regions, which convert each source shard that they read to the dense row-major layout on its own device,
        // because a compiled reshard supports neither manual axes nor uneven shards. Every process tells so from
        // shared metadata, and process 0, which holds no target device, returns the placement without buffers.
        // A pending sum that a target over both processes introduces along `s`, whose devices at coordinate
        // zero are those of process 1, first places the uneven split on those devices with regions, and process 0
        // only writes its additive identities.
        let manual_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            remote.to_vec(),
        )
        .unwrap();
        let summed_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![
                MeshAxis::new("s", 2, MeshAxisType::Explicit).unwrap(),
                MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            ])
            .unwrap(),
            vec![remote[0], remote[1], local[0], local[1]],
        )
        .unwrap();
        let split_rows = |mesh: &DeviceMesh, axis: &str| {
            let dimensions = vec![ShardingDimension::sharded([axis]), ShardingDimension::replicated()];
            Sharding::new(mesh.logical_mesh().clone(), dimensions).unwrap()
        };
        let summed = split_rows(&summed_mesh, "x").with_unreduced_axes(["s"]).unwrap();
        let cases = [
            ("manual even split", [4, 2], manual_mesh.clone(), split_rows(&manual_mesh, "m")),
            ("explicit uneven split", [5, 2], explicit_mesh(&remote), split_rows(&explicit_mesh(&remote), "x")),
            ("uneven split with an introduced sum", [5, 2], summed_mesh, summed),
        ];
        let outcomes = cases
            .iter()
            .map(|(name, dimensions, target_mesh, target)| {
                let r#type = ArrayType::new_static(DataType::U8, *dimensions).with_layout(column_major.clone());
                let views = shared_views(&session, &r#type, &remote, &replicated);
                let planned = placement_agreement(&views, target_mesh, target);
                let placed = place_on_process_zero(&views[0], target_mesh, target);
                format!("{name}: {planned:?}, process 0 {placed:?}")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            outcomes,
            [
                "manual even split: Ok(\"shard regions\"), process 0 Ok(\"0 buffers\")",
                "explicit uneven split: Ok(\"shard regions\"), process 0 Ok(\"0 buffers\")",
                "uneven split with an introduced sum: Ok(\"shard regions with additive identities\"), process 0 \
                 Ok(\"2 buffers\")",
            ],
        );

        // A process that imported only the metadata of a tile that the plan has it send (i.e., without a buffer for it
        // or for any local replica) agrees on the plan, and then fails before it issues any transfer, while its peer
        // issues its receive and waits. This is the accepted limit of the PJRT cross-host transfers that
        // `.tasks/plan_placement_decisions.md` records, because a process cannot fail the pending receive of another.
        let views = shared_views(&session, &r#type, &[local[0]], &replicated);
        let mesh = &explicit_mesh(&[local[0], remote[0]]);
        let sharding = &replicated(mesh);
        assert_eq!(placement_agreement(&views, mesh, sharding), Ok("exact shards".to_string()));
        let Ok(DevicePlacementStrategy::ExactShards { plan, cross_host_layouts }) =
            views[0].device_placement_strategy(&simulated_cross_host_layout, mesh, sharding, 0)
        else {
            panic!("a replica that crosses processes must move as a whole shard");
        };
        let sends = plan.moves_with_role(0, MoveRole::Send).map(|movement| movement.target_device).collect::<Vec<_>>();
        assert_eq!(sends, [remote[0]]);
        assert_eq!(
            plan.moves_with_role(1, MoveRole::Receive)
                .map(|movement| movement.source_device)
                .collect::<Vec<_>>(),
            [local[0]],
        );
        let error = views[0]
            .place_exact_shards(mesh, sharding, Memory::Device, &plan, &cross_host_layouts)
            .err()
            .unwrap();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(
                error.as_ref(),
                ArrayError::MissingShardBuffer { shard_index: 0, device_id } if *device_id == local[0].id(),
            ),
        ));
    }

    #[test]
    fn test_array_device_placement_strategy_assigns_host_round_trip_owner_across_processes() {
        // Returns the process that each view plans as the owner of the host round trip of a placement on `mesh`
        // according to `sharding`, which is the strategy of that placement or of its first step.
        fn owners(views: &[Array], mesh: &DeviceMesh, sharding: &Sharding) -> Vec<Option<usize>> {
            views
                .iter()
                .enumerate()
                .map(|(process_index, view)| {
                    let strategy =
                        view.device_placement_strategy(&simulated_cross_host_layout, mesh, sharding, process_index);
                    let strategy = match strategy.unwrap() {
                        DevicePlacementStrategy::IntroducedSums { first: Some(first), .. } => *first,
                        strategy => strategy,
                    };
                    match strategy {
                        DevicePlacementStrategy::HostRoundTrip { process_index, .. } => process_index,
                        _ => panic!("the target needs a host round trip"),
                    }
                })
                .collect()
        }

        let client = cpu_client(2);
        let local = Session::builder().client(client).build().unwrap().addressable_devices().unwrap();
        let next_id = local.iter().map(Device::id).max().unwrap() + 1;
        let remote = [Device::new(next_id, 1), Device::new(next_id + 1, 1)];
        let session = session_with_remote_devices(client, remote);
        let replicated = |mesh: &DeviceMesh| Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let maximum = |mesh: &DeviceMesh| {
            replicated(mesh)
                .with_unreduced_axes(["x"])
                .unwrap()
                .with_unreduced_kind(UnreducedKind::Max)
                .unwrap()
        };
        let r#type = ArrayType::new_static(DataType::F32, [2, 2]);

        // XLA drops NaN values from a maximum, so completing a pending maximum of `f32` values needs a host round
        // trip. Over the devices of both processes, no process can read the value back alone, so both plan the round
        // trip without an owner and return the same error before they do anything. When every device belongs to
        // process 1, process 1 reads the value back and uploads it, and process 0 returns the placement without
        // buffers, as it does for a compiled reshard that it does not run.
        let mut outcomes = Vec::new();
        for (name, devices) in [("over both processes", [local[0], remote[0]]), ("on process 1", remote)] {
            let views = shared_views(&session, &r#type, &devices, &maximum);
            let target_mesh = explicit_mesh(&devices);
            let target = replicated(&target_mesh);
            let planned = placement_agreement(&views, &target_mesh, &target);
            let owners = owners(&views, &target_mesh, &target);
            let placed = place_on_process_zero(&views[0], &target_mesh, &target);
            outcomes.push(format!("{name}: {planned:?}, owners {owners:?}, process 0 {placed:?}"));
        }
        assert_eq!(
            outcomes,
            [
                "over both processes: Ok(\"host round trip\"), owners [None, None], process 0 Err(\"cannot take `f32` \
                 values from a pending `max` reduction to no reduction state along mesh axis `x` on the devices, \
                 because XLA's maximum and minimum reductions drop NaN values, which the maximum and minimum of a \
                 compiled program propagate, and the host round trip that makes it exactly needs one process that \
                 owns every device of the target and can read the source back from its own devices; complete \
                 the reduction while the devices of one process hold the value, or in a compiled program, which drops \
                 NaN values as the collective does\")",
                "on process 1: Ok(\"host round trip\"), owners [Some(1), Some(1)], process 0 Ok(\"0 buffers\")",
            ],
        );

        // A pending maximum of `f32` values on the devices of process 0 that a target over both processes turns
        // into a pending sum needs a host round trip only onto the device at coordinate zero, which process 0 owns
        // with the source, so both processes plan process 0 as its owner, and process 1 only writes its identity.
        let views = shared_views(&session, &r#type, &local, &maximum);
        let target_mesh = explicit_mesh(&[local[0], remote[0]]);
        let target = replicated(&target_mesh).with_unreduced_axes(["x"]).unwrap();
        assert_eq!(
            (placement_agreement(&views, &target_mesh, &target), owners(&views, &target_mesh, &target),),
            (Ok("host round trip with additive identities".to_string()), vec![Some(0), Some(0)]),
        );

        // A scalar that varies along the manual axis `m` of process 0 (`11` and `22`) cannot be gathered into one
        // replicated value, nor read back as one host value, which every process can tell from the metadata that
        // they share, including the manual axes of the source mesh. Both processes therefore plan a host round trip
        // without an owner and return the same error before either posts or compiles anything, whether the
        // target is on process 1 or over both processes.
        let manual_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            local.clone(),
        )
        .unwrap();
        let varying = Sharding::replicated(manual_mesh.logical_mesh().clone(), 0).with_varying_manual_axes(["m"]);
        let varying = varying.unwrap();
        let scalar = ArrayType::scalar(DataType::U8);
        let own = Array::from_host_shards(
            &session,
            scalar.clone(),
            manual_mesh.clone(),
            Some(varying.clone()),
            [(local[0].id(), &[11u8]), (local[1].id(), &[22u8])],
        );
        let peer = Array::from_addressable_buffers(&session, scalar, manual_mesh, Some(varying), Vec::new());
        let views = [own.unwrap(), peer.unwrap()];
        let outcomes = [explicit_mesh(&remote[..1]), explicit_mesh(&[local[0], remote[0]])]
            .iter()
            .map(|target_mesh| {
                let target = Sharding::replicated(target_mesh.logical_mesh().clone(), 0);
                let planned = placement_agreement(&views, target_mesh, &target);
                (planned, owners(&views, target_mesh, &target))
            })
            .collect::<Vec<_>>();
        let planned = (Ok("host round trip".to_string()), vec![None, None]);
        assert_eq!(outcomes, [planned.clone(), planned]);
        let target_mesh = explicit_mesh(&[local[0], remote[0]]);
        let target = Sharding::replicated(target_mesh.logical_mesh().clone(), 0);
        assert_eq!(
            place_on_process_zero(&views[0], &target_mesh, &target),
            Err("compiled reshard does not support `Manual` mesh axis m".to_string()),
        );

        // With remote devices that hold only replicas of what process 0 holds, process 0 reads the value back alone,
        // so it owns the host round trip of a completion that the collective does not make (here a pending maximum of
        // the `f32` values `1` and NaN along `x`), and process 1 takes no part.
        let mixed_mesh = DeviceMesh::new(logical_mesh_2x2(), vec![local[0], remote[0], local[1], remote[1]]).unwrap();
        let maximum = Sharding::replicated(mixed_mesh.logical_mesh().clone(), 0)
            .with_unreduced_axes(["x"])
            .unwrap()
            .with_unreduced_kind(UnreducedKind::Max)
            .unwrap();
        let contributions = [1f32, f32::NAN].map(|value| Arc::<[u8]>::from(value.to_ne_bytes().as_slice()));
        let scalar = ArrayType::scalar(DataType::F32);
        let own = Array::from_host_shards(
            &session,
            scalar.clone(),
            mixed_mesh.clone(),
            Some(maximum.clone()),
            local.iter().zip(contributions).map(|(device, bytes)| (device.id(), bytes)),
        );
        let peer = Array::from_addressable_buffers(&session, scalar, mixed_mesh, Some(maximum), Vec::new());
        let views = [own.unwrap(), peer.unwrap()];
        let target_mesh = explicit_mesh(&local[..1]);
        let target = Sharding::replicated(target_mesh.logical_mesh().clone(), 0);
        let owners = owners(&views, &target_mesh, &target);
        let planned = placement_agreement(&views, &target_mesh, &target);
        let placed = views[0].to_placement(&session, Placement::new(target_mesh, target).unwrap());
        let placed = f32::from_ne_bytes(placed.unwrap().copy_to_host_bytes().unwrap().try_into().unwrap());
        assert_eq!(
            (planned, owners, placed.is_nan()),
            (Ok("host round trip".to_string()), vec![Some(0), Some(0)], true),
        );
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_device_placement_strategy_agrees_on_kept_contributions_of_process_local_values() {
        // Two processes, each with two devices, upload `u8[2] = [7, 11]` with a pending sum or maximum along `x`
        // without naming a mesh, so that each places its value on its own devices. A target on the devices of
        // process 0 that keeps those contributions is placed by process 0 alone, while process 1 takes no part and
        // returns the metadata of the result without buffers, as for a target without a pending reduction,
        // because one process owns every target device. A target over the devices of both processes is the
        // same error on both processes. Each row lists the agreed plan and the shard bytes that each process holds
        // after `Array::to_placement` and `Session::place`, which agree.
        let clients = [cpu_client(2), process_one_cpu_client(2)];
        let sessions = sessions_of_processes(clients);
        let (local, remote) = (sessions[0].addressable_devices().unwrap(), sessions[1].addressable_devices().unwrap());
        assert_eq!(local.iter().chain(&remote).map(Device::process_index).collect::<Vec<_>>(), [0, 0, 1, 1]);
        let combined = "cannot place an array with a pending reduction or with manual variation within a tile when \
                        neither whole shards nor regions of shards can move its contributions as they are (e.g., when \
                        the PJRT plugin does not report the default layout that a cross-host receive allocates), \
                        because the placement would have to combine the contributions of the pending reduction \
                        (complete the reduction first)";
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [2]);
        let targets = [
            ("on process 0", vec![local[0], local[1]]),
            ("reversed on process 0", vec![local[1], local[0]]),
            ("over both processes", vec![local[0], remote[0]]),
        ];
        let mut outcomes = Vec::new();
        for kind in [UnreducedKind::Sum, UnreducedKind::Max] {
            let pending = Sharding::replicated(logical_mesh.clone(), 1)
                .with_unreduced_axes(["x"])
                .unwrap()
                .with_unreduced_kind(kind)
                .unwrap();
            let sources = sessions
                .iter()
                .map(|session| {
                    Array::from_host_buffer(session, r#type.clone(), None, Some(pending.clone()), [7u8, 11]).unwrap()
                })
                .collect::<Vec<_>>();
            for (name, target_devices) in &targets {
                let mesh = DeviceMesh::new(logical_mesh.clone(), target_devices.clone()).unwrap();
                let planned = placement_agreement(&sources, &mesh, &pending)
                    .unwrap_or_else(|disagreement| format!("DISAGREE: {disagreement}"));
                let target = Placement::new(mesh, pending.clone()).unwrap();
                let mut placements = Vec::new();
                for (source, session) in sources.iter().zip(&sessions) {
                    let direct = source.to_placement(session, target.clone());
                    let placed = session.place(source.clone(), PlacementOptions::to(target.clone()));
                    placements.extend([direct, placed].map(|placed| placed.map_err(|error| error.to_string())));
                }
                let (owner, other) = process_outcomes(&placements, &format!("{kind:?} {name}"));
                let outcome = format!("{kind:?} {name}: {planned}; process 0 {owner}; process 1 {other}");
                outcomes.push(outcome.replace(&format!("error: {combined}"), "error"));
            }
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                Sum on process 0: exact shards | exact shards on its owner; process 0 [[7, 11], [0, 0]]; process 1 []
                Sum reversed on process 0: exact shards | exact shards on its owner; process 0 [[7, 11], [0, 0]]; process 1 []
                Sum over both processes: error; process 0 error; process 1 error
                Max on process 0: exact shards | exact shards on its owner; process 0 [[7, 11], [7, 11]]; process 1 []
                Max reversed on process 0: exact shards | exact shards on its owner; process 0 [[7, 11], [7, 11]]; process 1 []
                Max over both processes: error; process 0 error; process 1 error"},
        );
    }

    #[test]
    fn test_array_device_placement_strategy_leaves_array_statistics_unchanged() {
        // Two processes, each with two devices, place `u8[1] = [3]`, which each uploads to its own devices, varying
        // along the manual axis `m`, onto the devices of process 0 in either order. Planning registers no array on
        // either process, so the array statistics and the live arrays of each session are unchanged by it, and a
        // placement changes them only by the arrays that it creates: process 1 creates only its result without
        // buffers, and process 0 its committed result and, in reversed order, also the copy of its value onto the
        // target devices. Each row also names the strategy that the process plans.
        let clients = [cpu_client(2), process_one_cpu_client(2)];
        let sessions = clients.map(|client| {
            let builder = Session::builder().client(client).live_array_tracking(true);
            builder.remote_devices(peer_devices(client, &clients)).build().unwrap()
        });
        let devices = sessions[0].addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = Sharding::replicated(logical_mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap();
        let mut outcomes = Vec::new();
        let orders = [("in order", devices.clone()), ("reversed", vec![devices[1], devices[0]])];
        for (order, target_devices) in orders {
            let mesh = DeviceMesh::new(logical_mesh.clone(), target_devices).unwrap();
            let target = Placement::new(mesh.clone(), varying.clone()).unwrap();
            for (process_index, session) in sessions.iter().enumerate() {
                let r#type = ArrayType::new_static(DataType::U8, [1]);
                let source = Array::from_host_buffer(session, r#type, None, Some(varying.clone()), [3u8]).unwrap();
                let counts = || {
                    let statistics = session.array_statistics();
                    (statistics.created_arrays, statistics.live_arrays, session.live_arrays().unwrap().len())
                };
                let before = counts();
                let layout_resolver = cross_host_layout_resolver(session.client());
                let strategy =
                    planned_work(&source, &layout_resolver, process_index, &mesh, &varying).unwrap().strategy;
                assert_eq!(counts(), before, "{order} {process_index}: planning");
                let placed = source.to_placement(session, target.clone()).unwrap();
                let during = counts();
                drop(placed);
                let after = counts();
                let delta = |counts: (u64, u64, usize)| {
                    (counts.0 - before.0, counts.1 as i64 - before.1 as i64, counts.2 as i64 - before.2 as i64)
                };
                outcomes.push(format!(
                    "process {process_index} {order}: {strategy}, placed {:?}, dropped {:?}",
                    delta(during),
                    delta(after),
                ));
            }
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                process 0 in order: exact shards, placed (1, 1, 1), dropped (1, 0, 0)
                process 1 in order: exact shards on its owner, placed (1, 1, 1), dropped (1, 0, 0)
                process 0 reversed: exact shards, placed (2, 1, 1), dropped (2, 0, 0)
                process 1 reversed: exact shards on its owner, placed (1, 1, 1), dropped (1, 0, 0)"},
        );
    }

    #[test]
    fn test_reshard_program_compile_keeps_reduction_state() {
        // A reshard that keeps a pending reduction along the axis `x` of a 2x2 mesh whose axes are explicit or auto
        // moves the contributions as they are (over an auto axis, because the program types that axis as explicit), so
        // its optimized program has no all-reduce (only the all-to-all that a change of split needs), while one that
        // completes the reduction reduces it. A kept pending sum of complex values, whose introduction crashes XLA,
        // therefore never reaches a collective.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let collectives = ["all-gather", "all-reduce", "all-to-all", "collective-permute", "reduce-scatter"];
        let mut outcomes = Vec::new();
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Auto] {
            let logical_mesh = LogicalMesh::new(vec![
                MeshAxis::new("x", 2, axis_type).unwrap(),
                MeshAxis::new("y", 2, axis_type).unwrap(),
            ])
            .unwrap();
            let mesh = DeviceMesh::new(logical_mesh, devices.clone()).unwrap();
            let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
            let split_y = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
            let columns = Sharding::new(
                mesh.logical_mesh().clone(),
                vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
            )
            .unwrap();
            let rows = Sharding::new(
                mesh.logical_mesh().clone(),
                vec![ShardingDimension::sharded(["y"]), ShardingDimension::replicated()],
            )
            .unwrap();
            let unconstrained =
                Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::unconstrained()]).unwrap();
            let unreduced = |sharding: &Sharding, kind: UnreducedKind| {
                sharding.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap()
            };
            let (sum, max) = (UnreducedKind::Sum, UnreducedKind::Max);
            let (pending_sum, split_sum) = (unreduced(&replicated, sum), unreduced(&split_y, sum));
            let unconstrained_sum = unreduced(&unconstrained, sum);
            let (columns_sum, rows_sum) = (unreduced(&columns, sum), unreduced(&rows, sum));
            for (name, data_type, shape, input, output) in [
                ("keep sum", DataType::F32, vec![4], pending_sum.clone(), split_sum.clone()),
                ("keep sum, unconstrained", DataType::F32, vec![4], pending_sum.clone(), unconstrained_sum),
                ("keep sum, c64", DataType::C64, vec![4], pending_sum.clone(), split_sum),
                ("keep max", DataType::F32, vec![4], unreduced(&replicated, max), unreduced(&split_y, max)),
                ("keep sum, columns to rows", DataType::F32, vec![4, 4], columns_sum, rows_sum),
                ("complete sum", DataType::F32, vec![4], pending_sum, split_y.clone()),
            ] {
                let shard_size =
                    ShardLayout::new(&StaticShape::new(shape.clone()), &mesh, &input).unwrap().descriptors()[0]
                        .shape()
                        .as_slice()
                        .iter()
                        .product::<usize>();
                // A `c64` element holds two `f32` components.
                let components = if data_type == DataType::C64 { 2 } else { 1 };
                let zeros = values_to_bytes(&vec![0f32; shard_size * components]);
                let contributions =
                    devices.iter().map(|device| (device.id(), Arc::<[u8]>::from(zeros.as_slice()))).collect::<Vec<_>>();
                let r#type = ArrayType::new_static(data_type, shape);
                let source =
                    Array::from_host_shards(&session, r#type, mesh.clone(), Some(input.clone()), contributions)
                        .unwrap();
                let program = ReshardProgram { mesh: mesh.clone(), input, output };
                let compiled = program.compile(&source, false).unwrap().unwrap();
                // The cache returns the program that the reshard compiled, because its staged function lowers alike.
                let lowered = compiled.staged().into_inner().lower(&session).unwrap();
                let optimized = session.compilation_cache().compile(&session, lowered.lowered_program()).unwrap();
                let optimized = optimized.optimized_program().unwrap();
                let present = collectives
                    .iter()
                    .filter(|opcode| optimized.bytes().windows(opcode.len()).any(|window| window == opcode.as_bytes()))
                    .copied()
                    .collect::<Vec<_>>();
                outcomes.push(format!("{axis_type:?} {name}: {present:?}"));
            }
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {r#"
                Explicit keep sum: []
                Explicit keep sum, unconstrained: []
                Explicit keep sum, c64: []
                Explicit keep max: []
                Explicit keep sum, columns to rows: ["all-to-all"]
                Explicit complete sum: ["all-reduce"]
                Auto keep sum: []
                Auto keep sum, unconstrained: []
                Auto keep sum, c64: []
                Auto keep max: []
                Auto keep sum, columns to rows: ["all-to-all"]
                Auto complete sum: ["all-reduce"]"#},
        );
    }

    #[test]
    fn test_placement_planning_view_shard_region_plan() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();

        // A `[4, 6]` grid of four tiles of `[2, 3]` is placed in rows of `[1, 6]` on another mesh of the same devices.
        // Each row intersects the two tiles of its half of the rows, which it joins in the order of their columns.
        let source_mesh = DeviceMesh::new(logical_mesh_2x2(), devices.clone()).unwrap();
        let tiles = Sharding::new(
            source_mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4, 6]);
        let source = Array::from_addressable_buffers(&session, r#type, source_mesh, Some(tiles), Vec::new()).unwrap();
        let rows_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("z", 4, MeshAxisType::Auto).unwrap()]).unwrap(),
            devices.clone(),
        )
        .unwrap();
        let rows = Sharding::new(
            rows_mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["z"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let (plan, _) = source
            .placement_planning_view()
            .shard_region_plan(&simulated_cross_host_layout, &rows_mesh, &rows)
            .unwrap()
            .unwrap();
        let moves = plan
            .moves
            .iter()
            .map(|movement| (movement.source_shard_index, movement.target_shard_index, movement.slice.clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            moves,
            vec![
                (0, 0, vec![0..1, 0..3]),
                (1, 0, vec![0..1, 3..6]),
                (0, 1, vec![1..2, 0..3]),
                (1, 1, vec![1..2, 3..6]),
                (2, 2, vec![2..3, 0..3]),
                (3, 2, vec![2..3, 3..6]),
                (2, 3, vec![3..4, 0..3]),
                (3, 3, vec![3..4, 3..6]),
            ],
        );

        // A replicated source has one slice, which every target shard intersects, and each region moves from the
        // replica on the device of its target shard, also for the shards of different sizes of a split of five
        // elements over four devices.
        let replicated_mesh = DeviceMesh::new(logical_mesh_2x2(), devices.clone()).unwrap();
        let replicated = Sharding::replicated(replicated_mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [5]);
        let source =
            Array::from_addressable_buffers(&session, r#type, replicated_mesh, Some(replicated), Vec::new()).unwrap();
        let split = Sharding::new(rows_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["z"])]).unwrap();
        let (plan, _) = source
            .placement_planning_view()
            .shard_region_plan(&simulated_cross_host_layout, &rows_mesh, &split)
            .unwrap()
            .unwrap();
        let moves = plan
            .moves
            .iter()
            .map(|movement| (movement.source_shard_index, movement.target_shard_index, movement.slice.clone()))
            .collect::<Vec<_>>();
        assert_eq!(moves, vec![(0, 0, vec![0..2]), (1, 1, vec![2..3]), (2, 2, vec![3..4]), (3, 3, vec![4..5])]);

        // The boundaries of a split of six elements over four devices (i.e., `[0, 2)`, `[2, 4)`, `[4, 5)`, and
        // `[5, 6)`) are offset from those of a split over two devices (i.e., `[0, 3)` and `[3, 6)`), so each
        // target shard joins the parts of the source shards that it intersects, in the order of their starts.
        let r#type = ArrayType::new_static(DataType::F32, [6]);
        let source = Array::from_addressable_buffers(&session, r#type, rows_mesh, Some(split), Vec::new()).unwrap();
        let halves_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("w", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            devices[..2].to_vec(),
        )
        .unwrap();
        let halves =
            Sharding::new(halves_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["w"])]).unwrap();
        let (plan, _) = source
            .placement_planning_view()
            .shard_region_plan(&simulated_cross_host_layout, &halves_mesh, &halves)
            .unwrap()
            .unwrap();
        let moves = plan
            .moves
            .iter()
            .map(|movement| (movement.source_shard_index, movement.target_shard_index, movement.slice.clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            moves,
            vec![(0, 0, vec![0..2]), (1, 0, vec![2..3]), (1, 1, vec![3..4]), (2, 1, vec![4..5]), (3, 1, vec![5..6])],
        );
    }

    #[test]
    fn test_source_replicas_by_contribution() {
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), (0..8).map(|index| Device::new(index, 0)).collect()).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 1);
        let layout = ShardLayout::new(&StaticShape::new(vec![4]), &mesh, &sharding).unwrap();
        let shards = layout.descriptors();

        let groups = SourceReplicas::by_contribution(&mesh, &sharding, shards, None);
        assert_eq!(groups.len(), 1);
        let (key, replicas) = groups.get_key_value(&(shards[7].slice(), None)).unwrap();
        assert!(std::ptr::eq(key.0, shards[0].slice()));
        assert_eq!(replicas.preferred(&shards[7]).index(), 7);

        // Every shard still covers the same slice, and only the two shards along `r` in each member are replicas.
        let sharding = sharding.with_unreduced_axes(["x"]).unwrap().with_varying_manual_axes(["m"]).unwrap();
        let axes = ["m".to_string()];
        let groups = SourceReplicas::by_contribution(&mesh, &sharding, shards, Some(&axes));
        assert_eq!(groups.len(), 4);
        let key = source_replica_key(&mesh, &sharding, &shards[7], Some(&axes));
        let (key, replicas) = groups.get_key_value(&key).unwrap();
        assert!(std::ptr::eq(key.0, shards[6].slice()));
        assert_eq!(replicas.preferred(&shards[0]).index(), 6);
        assert_eq!(replicas.preferred(&shards[7]).index(), 7);
    }

    #[test]
    fn test_source_replicas_preferred() {
        let replicated_shards = |devices: Vec<Device>| {
            let logical_mesh =
                LogicalMesh::new(vec![MeshAxis::new("x", devices.len(), MeshAxisType::Auto).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(logical_mesh, devices).unwrap();
            ShardLayout::new(&StaticShape::new(vec![2]), &mesh, &Sharding::replicated(mesh.logical_mesh().clone(), 1))
                .unwrap()
        };

        // Two replicas on each of two processes, listed out of device order.
        let sources =
            replicated_shards(vec![Device::new(1, 0), Device::new(0, 0), Device::new(3, 1), Device::new(2, 1)]);
        let sources = sources.descriptors().iter().collect::<Vec<_>>();
        let replicas = SourceReplicas::new(sources.as_slice());

        // A target prefers the replica on its own device, then the one with the lowest device ID on its process,
        // and then the one with the lowest device ID of all processes.
        let targets = replicated_shards(vec![
            Device::new(1, 0),
            Device::new(2, 1),
            Device::new(5, 0),
            Device::new(7, 1),
            Device::new(9, 2),
        ]);
        let preferred = targets
            .descriptors()
            .iter()
            .map(|target| (replicas.preferred(target).device(), replicas.preferred(target).index()))
            .collect::<Vec<_>>();
        assert_eq!(
            preferred,
            [
                (Device::new(1, 0), 0),
                (Device::new(2, 1), 3),
                (Device::new(0, 0), 1),
                (Device::new(2, 1), 3),
                (Device::new(0, 0), 1),
            ],
        );
    }

    #[test]
    fn test_plan_exact_shard_moves() {
        // Process 0 holds the first half of an `f32[4]` and process 1 the second half, and the target swaps them.
        let (local, remote) = (Device::new(0, 0), Device::new(1, 1));
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![local, remote]).unwrap();
        let target_mesh = DeviceMesh::new(logical_mesh.clone(), vec![remote, local]).unwrap();
        let sharding = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let shape = StaticShape::new(vec![4]);
        let sources = ShardLayout::new(&shape, &source_mesh, &sharding).unwrap();
        let sources = sources.descriptors().iter();
        let plan =
            plan_exact_shard_moves(BufferType::F32, sources, &source_mesh, &sharding, &shape, &target_mesh, &sharding);
        let send = ShardMove {
            source_shard_index: 0,
            source_device: local,
            target_shard_index: 0,
            target_device: remote,
            slice: vec![0..2],
            transfer_key: 0,
        };
        let receive = ShardMove {
            source_shard_index: 1,
            source_device: remote,
            target_shard_index: 1,
            target_device: local,
            slice: vec![2..4],
            transfer_key: 3,
        };
        let plan = plan.unwrap().unwrap();
        assert_eq!(plan, ShardMovePlan::new(BufferType::F32, vec![send.clone(), receive.clone()]).unwrap());

        // Every process derives the same plan and performs the moves of its own devices, so the send of process 0 is
        // the receive of process 1, and the other way around.
        assert_eq!(plan.cross_host_moves().collect::<Vec<_>>(), [&send, &receive]);
        assert_eq!(plan.moves_with_role(0, MoveRole::Copy).count(), 0);
        assert_eq!(plan.moves_with_role(0, MoveRole::Send).collect::<Vec<_>>(), [&send]);
        assert_eq!(plan.moves_with_role(0, MoveRole::Receive).collect::<Vec<_>>(), [&receive]);
        assert_eq!(plan.moves_with_role(1, MoveRole::Send).collect::<Vec<_>>(), [&receive]);
        assert_eq!(plan.moves_with_role(1, MoveRole::Receive).collect::<Vec<_>>(), [&send]);
    }

    #[test]
    fn test_plan_exact_shard_moves_rejects_invalid_cross_host_device_ids_on_every_process() {
        // Process 0 sends the halves of an `f32[4]` to a valid device of process 2 and to a device of process 1 whose
        // identifier PJRT cross-host transfers cannot represent. The plan covers the moves of every process, so every
        // process rejects it before any transfer, including process 2, which would otherwise wait for a send that
        // never comes.
        let invalid_id = i32::MAX as usize + 1;
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![Device::new(2, 2), Device::new(invalid_id, 1)]).unwrap();
        let sharding = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let shape = StaticShape::new(vec![4]);
        let sources = ShardLayout::new(&shape, &source_mesh, &sharding).unwrap();
        let sources = sources.descriptors().iter();
        let plan = plan_exact_shard_moves(BufferType::F32, sources, &source_mesh, &sharding, &shape, &mesh, &sharding);
        assert_eq!(
            plan.map_err(|error| error.to_string()),
            Err(format!(
                "cross-host transfer device ID {invalid_id} exceeds the maximum allowed value of {}",
                i32::MAX,
            )),
        );
    }

    #[test]
    fn test_moves_contributions_pairs_the_axes_that_distinguish_contributions() {
        let mesh = |axes: &[(&str, usize, MeshAxisType)]| {
            let axes = axes.iter().map(|(name, size, r#type)| MeshAxis::new(*name, *size, *r#type).unwrap());
            LogicalMesh::new(axes.collect()).unwrap()
        };
        let pending = |mesh: LogicalMesh, axes: &[&str]| {
            Sharding::replicated(mesh, 0).with_unreduced_axes(axes.iter().map(|axis| axis.to_string())).unwrap()
        };
        let varying = |mesh: LogicalMesh, axes: &[&str]| {
            Sharding::replicated(mesh, 0)
                .with_varying_manual_axes(axes.iter().map(|axis| axis.to_string()))
                .unwrap()
        };
        let axes = |source: &[&str], target: &[&str]| {
            let names = |names: &[&str]| names.iter().map(|name| name.to_string()).collect();
            Some(ContributionAxes { source: names(source), target: names(target) })
        };
        let (auto, explicit, manual) = (MeshAxisType::Auto, MeshAxisType::Explicit, MeshAxisType::Manual);
        let xy = mesh(&[("x", 2, auto), ("y", 2, auto)]);
        let yx = mesh(&[("y", 2, auto), ("x", 2, auto)]);
        let x2y3 = mesh(&[("x", 2, auto), ("y", 3, auto)]);
        let y3x2 = mesh(&[("y", 3, auto), ("x", 2, auto)]);
        let xm = mesh(&[("x", 2, auto), ("m", 2, manual)]);
        let cases = [
            // Unreduced axes correspond by the ordinals of their contributions, whatever their names, which keeps the
            // order of the host readback.
            (pending(xy.clone(), &["x"]), pending(yx.clone(), &["x"]), axes(&[], &[])),
            (pending(xy.clone(), &["x", "y"]), pending(yx.clone(), &["y", "x"]), axes(&[], &[])),
            (pending(xy.clone(), &["x"]), pending(xy.clone(), &["y"]), axes(&[], &[])),
            (pending(mesh(&[("x", 2, explicit)]), &["x"]), pending(mesh(&[("y", 2, auto)]), &["y"]), axes(&[], &[])),
            // The same number of contributions corresponds whatever the sizes and the number of the unreduced axes.
            (pending(x2y3.clone(), &["x", "y"]), pending(y3x2.clone(), &["y", "x"]), axes(&[], &[])),
            (
                pending(x2y3.clone(), &["x", "y"]),
                pending(mesh(&[("y", 3, auto), ("z", 2, auto)]), &["y", "z"]),
                axes(&[], &[]),
            ),
            (pending(x2y3.clone(), &["x", "y"]), pending(mesh(&[("z", 6, auto)]), &["z"]), axes(&[], &[])),
            (
                pending(x2y3.clone(), &["x", "y"]),
                pending(mesh(&[("x", 2, auto), ("y", 3, auto), ("u", 1, auto)]), &["x", "y", "u"]),
                axes(&[], &[]),
            ),
            // An added or removed axis that holds replicas changes no contribution.
            (
                pending(mesh(&[("x", 2, auto)]), &["x"]),
                pending(mesh(&[("x", 2, auto), ("u", 1, auto)]), &["x"]),
                axes(&[], &[]),
            ),
            (pending(xy.clone(), &["y"]), pending(mesh(&[("y", 2, auto), ("z", 2, auto)]), &["y"]), axes(&[], &[])),
            // Reduced axes hold replicas, so their order and sizes change no contribution.
            (
                pending(mesh(&[("r", 1, auto), ("s", 2, auto), ("x", 2, auto)]), &["x"])
                    .with_reduced_axes(["r", "s"])
                    .unwrap(),
                pending(mesh(&[("s", 2, auto), ("r", 1, auto), ("x", 2, auto)]), &["x"])
                    .with_reduced_axes(["r", "s"])
                    .unwrap(),
                axes(&[], &[]),
            ),
            // Another unreduced kind, another number of contributions, or a pending reduction at one end only is not a
            // pure move.
            (
                pending(xy.clone(), &["x"]),
                pending(xy.clone(), &["x"]).with_unreduced_kind(UnreducedKind::Max).unwrap(),
                None,
            ),
            (pending(xy.clone(), &["x"]), pending(xy.clone(), &["x", "y"]), None),
            (pending(mesh(&[("x", 2, auto)]), &["x"]), pending(mesh(&[("x", 4, auto)]), &["x"]), None),
            (pending(x2y3.clone(), &["x", "y"]), pending(mesh(&[("z", 4, auto)]), &["z"]), None),
            (pending(xy.clone(), &[]), pending(mesh(&[("x", 2, auto), ("u", 1, auto)]), &["u"]), None),
            // Varying manual axes pair by name, and otherwise in mesh order.
            (
                varying(mesh(&[("a", 2, manual), ("b", 3, manual)]), &["a", "b"]),
                varying(mesh(&[("b", 3, manual), ("a", 2, manual)]), &["a", "b"]),
                axes(&["a", "b"], &["a", "b"]),
            ),
            (
                varying(mesh(&[("a", 2, manual)]), &["a"]),
                varying(mesh(&[("c", 2, manual)]), &["c"]),
                axes(&["a"], &["c"]),
            ),
            (varying(mesh(&[("a", 2, manual)]), &["a"]), varying(mesh(&[("c", 3, manual)]), &["c"]), None),
            // A varying axis of the source that splits a dimension holds disjoint slices, so it takes no part unless
            // the target varies along it as well.
            (
                Sharding::new(xm.clone(), vec![ShardingDimension::sharded(["m"])])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap()
                    .with_varying_manual_axes(["m"])
                    .unwrap(),
                Sharding::replicated(xm.clone(), 1).with_unreduced_axes(["x"]).unwrap(),
                axes(&[], &[]),
            ),
            (
                Sharding::new(xm.clone(), vec![ShardingDimension::sharded(["m"])])
                    .unwrap()
                    .with_varying_manual_axes(["m"])
                    .unwrap(),
                Sharding::replicated(xm.clone(), 1).with_varying_manual_axes(["m"]).unwrap(),
                axes(&["m"], &["m"]),
            ),
            // A varying axis that splits a dimension pairs in mesh order with a renamed axis of the target that
            // splits the same dimension.
            (
                Sharding::new(xm, vec![ShardingDimension::sharded(["m"])])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap()
                    .with_varying_manual_axes(["m"])
                    .unwrap(),
                Sharding::new(mesh(&[("x", 2, auto), ("n", 2, manual)]), vec![ShardingDimension::sharded(["n"])])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap()
                    .with_varying_manual_axes(["n"])
                    .unwrap(),
                axes(&["m"], &["n"]),
            ),
        ];
        for (input, output, expected) in cases {
            assert_eq!(moves_contributions(&input, &output), expected, "{input:?} onto {output:?}");
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_cross_host_shape() {
        assert_eq!(cross_host_shape(0, &StaticShape::new(vec![2, 3])).unwrap(), vec![2, 3]);

        // An extent that the backend cannot represent is reported with the shard and the dimension that hold it.
        let error = cross_host_shape(2, &StaticShape::new(vec![usize::MAX])).unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::InvalidShardDimension {
                    shard_index: 2,
                    dimension: 0,
                    source: DimensionError::ExtentExceedsBackendWidth { value, maximum },
                } if *value == usize::MAX && *maximum == i64::MAX as usize,
            ),
        ));
        assert_eq!(
            error.to_string(),
            format!(
                "cross-host transfer for shard #2, dimension #0: dimension extent {} exceeds backend maximum {}",
                usize::MAX,
                i64::MAX,
            ),
        );
    }

    #[test]
    fn test_cross_host_layout_resolver_requires_transfer_support() {
        // The CPU plugin reports row-major layouts, but they are usable for a remote receive only when the plugin
        // also supports cross-host transfers.
        let client = cpu_client(1);
        let layout_resolver = cross_host_layout_resolver(client);
        for (element_type, shape) in [(BufferType::F32, vec![2, 3]), (BufferType::Predicate, vec![4])] {
            let dimensions = shape.iter().map(|&size| size as u64).collect::<Vec<_>>();
            let layout = client.default_layout(element_type, dimensions).unwrap().proto().unwrap();
            assert_eq!(
                ryft_pjrt::Layout::from_proto(layout).unwrap(),
                dense_row_major_layout(shape.len()).to_pjrt().unwrap(),
            );
            assert!(matches!(
                layout_resolver(element_type, 0, &StaticShape::new(shape)),
                Err(Error::Pjrt(PjrtError::Unimplemented { .. })),
            ));
        }
    }

    #[test]
    fn test_exact_shard_transfer_key() {
        assert_eq!(exact_shard_transfer_key(2, 1, 3).unwrap(), 7);
        let error = exact_shard_transfer_key(usize::MAX, 0, 2).unwrap_err();
        assert!(matches!(&error, Error::SizeLimitExceeded { .. }));
        assert_eq!(
            error.to_string(),
            format!(
                "cross-host transfer key for source shard #{} and target shard #0 exceeds the maximum allowed \
                 size of {}",
                usize::MAX,
                usize::MAX,
            ),
        );
    }

    #[test]
    fn test_cross_host_global_device_id() {
        assert_eq!(cross_host_global_device_id(1).unwrap(), 1);
        let device_id = i32::MAX as usize + 1;
        let error = cross_host_global_device_id(device_id).unwrap_err();
        assert!(matches!(&error, Error::SizeLimitExceeded { .. }));
        assert_eq!(
            error.to_string(),
            format!("cross-host transfer device ID {device_id} exceeds the maximum allowed value of {}", i32::MAX),
        );
    }

    #[test]
    fn test_validate_reduction_transition() {
        // Every change of reduction state along the axis `x` of size two between no state, a pending reduction of each
        // kind, and a completed reduction, for a data type of every class of the rule, as the planner decides it: `.`
        // for a compiled reshard, `h` for a host round trip, `i` for the additive identities that introduce a pending
        // sum after a placement that a compiled reshard (or a move) makes, and `E` for an error. An introduced sum
        // takes the host round trip only when the placement without it does (e.g., to complete a pending maximum of
        // `f32` values first). The rule reads neither the type of the axis nor anything that only one process knows.
        // Along an axis of size one, a completion changes no value, so it is always a compiled reshard, while XLA
        // still cannot introduce a pending maximum or minimum.
        let client = cpu_client(2);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let data_types = [
            DataType::Boolean,
            DataType::U1,
            DataType::I1,
            DataType::U2,
            DataType::I32,
            DataType::F6E2M3FN,
            DataType::F8E4M3FNUZ,
            DataType::F32,
            DataType::C64,
            DataType::F8E8M0FNU,
        ];
        let shape = StaticShape::new(vec![4]);
        let decide = |data_type: DataType, mesh: &DeviceMesh, input: &Sharding, output: &Sharding| {
            match plan_reduction_transition(data_type, &shape, mesh, input, output) {
                Err(error) => format!("error: {error}"),
                Ok(strategy) => {
                    if let Some(cause) = strategy.host_round_trip_cause() {
                        format!("host round trip: {}", cause.across_processes())
                    } else if matches!(strategy, DevicePlacementStrategy::IntroducedSums { .. }) {
                        "additive identities".to_string()
                    } else {
                        "compiled reshard".to_string()
                    }
                }
            }
        };
        let mut outcomes = Vec::new();
        for (size, axis_type) in [(2, MeshAxisType::Auto), (2, MeshAxisType::Explicit), (1, MeshAxisType::Explicit)] {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", size, axis_type).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices[..size].to_vec()).unwrap();
            let replicated = Sharding::replicated(logical_mesh, 1);
            let pending = |kind: UnreducedKind| {
                replicated.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap()
            };
            let states = [
                ("none", replicated.clone()),
                ("sum", pending(UnreducedKind::Sum)),
                ("max", pending(UnreducedKind::Max)),
                ("min", pending(UnreducedKind::Min)),
                ("reduced", replicated.clone().with_reduced_axes(["x"]).unwrap()),
            ];
            for data_type in data_types {
                for (name, input) in &states {
                    let codes = states
                        .iter()
                        .map(|(_, output)| match decide(data_type, &mesh, input, output).split(':').next().unwrap() {
                            "compiled reshard" => ".",
                            "host round trip" => "h",
                            "additive identities" => "i",
                            _ => "E",
                        })
                        .collect::<Vec<_>>();
                    outcomes.push(format!("{axis_type:?} {size} {data_type} {name}: {}", codes.join(" ")));
                }
            }
        }
        let rows = indoc! {"
            bool none: . i h h .
            bool sum: . . h h .
            bool max: . i . h .
            bool min: . i h . .
            bool reduced: . i h h .
            u1 none: . i h h .
            u1 sum: h . h h h
            u1 max: . i . h .
            u1 min: . i h . .
            u1 reduced: . i h h .
            i1 none: . i h h .
            i1 sum: h . h h h
            i1 max: h h . h h
            i1 min: h h h . h
            i1 reduced: . i h h .
            u2 none: . i h h .
            u2 sum: h . h h h
            u2 max: h h . h h
            u2 min: h h h . h
            u2 reduced: . i h h .
            i32 none: . i h h .
            i32 sum: . . h h .
            i32 max: . i . h .
            i32 min: . i h . .
            i32 reduced: . i h h .
            f6e2m3fn none: . i h h .
            f6e2m3fn sum: h . h h h
            f6e2m3fn max: h h . h h
            f6e2m3fn min: h h h . h
            f6e2m3fn reduced: . i h h .
            f8e4m3fnuz none: . i h h .
            f8e4m3fnuz sum: . . h h .
            f8e4m3fnuz max: h h . h h
            f8e4m3fnuz min: h h h . h
            f8e4m3fnuz reduced: . i h h .
            f32 none: . i h h .
            f32 sum: . . h h .
            f32 max: h h . h h
            f32 min: h h h . h
            f32 reduced: . i h h .
            c64 none: . i h h .
            c64 sum: . . h h .
            c64 max: h h . h h
            c64 min: h h h . h
            c64 reduced: . i h h .
            f8e8m0fnu none: . E h h .
            f8e8m0fnu sum: h . h h h
            f8e8m0fnu max: h E . h h
            f8e8m0fnu min: h E h . h
            f8e8m0fnu reduced: . E h h ."};
        let size_one_rows = indoc! {"
            none: . i h h .
            sum: . . h h .
            max: . i . h .
            min: . i h . .
            reduced: . i h h ."};
        let mut expected = ["Auto 2", "Explicit 2"]
            .iter()
            .flat_map(|prefix| rows.lines().map(move |row| format!("{prefix} {row}")))
            .collect::<Vec<_>>();
        expected.extend(
            data_types
                .iter()
                .flat_map(|data_type| size_one_rows.lines().map(move |row| format!("Explicit 1 {data_type} {row}"))),
        );
        assert_eq!(outcomes, expected);

        // The diagnostics name the data type, both states, the axis, the reason, and what the user can do. A value
        // without elements needs no additive identity, so even a data type without one introduces a pending sum.
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
        let pending = |kind: UnreducedKind| {
            replicated.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap()
        };
        let empty = plan_reduction_transition(
            DataType::F8E8M0FNU,
            &StaticShape::new(vec![0]),
            &mesh,
            &replicated,
            &pending(UnreducedKind::Sum),
        )
        .unwrap();
        let DevicePlacementStrategy::IntroducedSums { sharding, axis_indices, .. } = empty else {
            panic!("an empty placement must still introduce the pending sum");
        };
        assert_eq!(axis_indices, vec![0]);
        assert_eq!(sharding_on_mesh(&sharding, mesh.logical_mesh()).unwrap(), replicated);
        let host = "and the host round trip that makes it exactly needs one process that owns every device of the \
                    target and can read the source back from its own devices";
        let complete = "complete the reduction while the devices of one process hold the value, or in another data \
                        type";
        let compiled = "complete the reduction while the devices of one process hold the value, or in a compiled \
                        program, which drops NaN values as the collective does";
        let anywhere = "make this placement where one process owns every device of the target and a device of \
                        every source contribution with elements";
        // Returns the diagnostic of a completion of a pending reduction of `kind` of `data_type` values.
        let completion = |data_type: &str, kind: &str, reason: &str, remedy: &str| {
            format!(
                "host round trip: cannot take `{data_type}` values from a pending `{kind}` reduction to no reduction \
                 state along mesh axis `x` on the devices, because {reason}, {host}; {remedy}"
            )
        };
        assert_eq!(
            [
                decide(DataType::U1, &mesh, &pending(UnreducedKind::Sum), &replicated),
                decide(DataType::I1, &mesh, &pending(UnreducedKind::Max), &replicated),
                decide(DataType::U2, &mesh, &pending(UnreducedKind::Sum), &replicated),
                decide(DataType::F6E2M3FN, &mesh, &pending(UnreducedKind::Sum), &replicated),
                decide(DataType::F8E8M0FNU, &mesh, &pending(UnreducedKind::Sum), &replicated),
                decide(DataType::C64, &mesh, &pending(UnreducedKind::Min), &replicated),
                decide(DataType::F32, &mesh, &pending(UnreducedKind::Max), &replicated),
                decide(DataType::I32, &mesh, &replicated, &pending(UnreducedKind::Max)),
                decide(DataType::F8E8M0FNU, &mesh, &replicated, &pending(UnreducedKind::Sum)),
            ]
            .join("\n"),
            [
                completion(
                    "u1",
                    "sum",
                    "XLA adds one-bit integers as predicates (i.e., with a logical or), while their sum wraps around",
                    complete,
                ),
                completion(
                    "i1",
                    "max",
                    "XLA orders one-bit integers as predicates, which puts `-1` above `0`",
                    complete
                ),
                completion("u2", "sum", "XLA does not compile reductions of two-bit integers", complete),
                completion(
                    "f6e2m3fn",
                    "sum",
                    "XLA does not compile reductions of six-bit floating point values",
                    complete
                ),
                completion(
                    "f8e8m0fnu",
                    "sum",
                    "the arithmetic of XLA for `f8e8m0fnu` values, in any compiled program, is not that of the host \
                     readback that defines the value (e.g., `2^-127 + 2^-127` gives NaN in XLA)",
                    complete,
                ),
                completion("c64", "min", "XLA has no maximum or minimum reduction of complex values", complete),
                completion(
                    "f32",
                    "max",
                    "XLA's maximum and minimum reductions drop NaN values, which the maximum and minimum of a compiled \
                     program propagate",
                    compiled,
                ),
                format!(
                    "host round trip: cannot take `i32` values from no reduction state to a pending `max` reduction \
                     along mesh axis `x` on the devices, because XLA can only introduce a pending `sum` reduction, \
                     {host}; {anywhere}"
                ),
                "error: type `f8e8m0fnu` has no additive identity to split a value into summands along unreduced axes"
                    .to_string(),
            ]
            .join("\n"),
        );
    }

    #[test]
    fn test_validate_reduction_transition_plans_only_programs_that_lowering_accepts() {
        // Every change of reduction state that the planner leaves to a compiled reshard lowers, because the lowering
        // rejects the changes that the same classification (`collective_reduction`) does not let the planner make on
        // the devices. Each such cell lowers the program that `ReshardProgram::compile` stages, over the
        // mesh of `ReshardProgram::compilation_mesh`, for every data type with values, both axis types, and every pair of
        // states (none, a pending sum, maximum, or minimum, and a completed reduction). A cell that introduces a
        // pending sum plans the program of its placement without that sum, whose identities are written without a
        // program. Only lowering runs, so that a wrong plan cannot abort the test process.
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let data_types = [
            DataType::Boolean,
            DataType::I1,
            DataType::I2,
            DataType::I4,
            DataType::I8,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::U1,
            DataType::U2,
            DataType::U4,
            DataType::U8,
            DataType::U16,
            DataType::U32,
            DataType::U64,
            DataType::F4E2M1FN,
            DataType::F6E2M3FN,
            DataType::F6E3M2FN,
            DataType::F8E3M4,
            DataType::F8E4M3,
            DataType::F8E4M3FN,
            DataType::F8E4M3FNUZ,
            DataType::F8E4M3B11FNUZ,
            DataType::F8E5M2,
            DataType::F8E5M2FNUZ,
            DataType::F8E8M0FNU,
            DataType::BF16,
            DataType::F16,
            DataType::F32,
            DataType::F64,
            DataType::C64,
            DataType::C128,
        ];
        let shape = StaticShape::new(vec![4]);
        let (mut planned, mut declined) = (0, 0);
        for axis_type in [MeshAxisType::Auto, MeshAxisType::Explicit] {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
            let replicated = Sharding::replicated(logical_mesh, 1);
            let pending = |kind: UnreducedKind| {
                replicated.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap()
            };
            let states = [
                replicated.clone(),
                pending(UnreducedKind::Sum),
                pending(UnreducedKind::Max),
                pending(UnreducedKind::Min),
                replicated.clone().with_reduced_axes(["x"]).unwrap(),
            ];
            for data_type in data_types {
                for input in &states {
                    for output in &states {
                        let first;
                        let output = match plan_reduction_transition(data_type, &shape, &mesh, input, output) {
                            Ok(DevicePlacementStrategy::IntroducedSums { sharding, .. }) => {
                                // Check the reduction rules on the full mesh even if the plan puts the first step
                                // on its coordinate-zero submesh.
                                first = sharding_on_mesh(&sharding, mesh.logical_mesh()).unwrap();
                                &first
                            }
                            Ok(_) => output,
                            Err(_) => {
                                declined += 1;
                                continue;
                            }
                        };
                        match validate_reduction_transition(data_type, &mesh, input, output) {
                            None => planned += 1,
                            Some(_) => {
                                declined += 1;
                                continue;
                            }
                        }
                        let program =
                            ReshardProgram { mesh: mesh.clone(), input: input.clone(), output: output.clone() };
                        let compilation_mesh = program.compilation_mesh().unwrap();
                        let logical_mesh = compilation_mesh.logical_mesh().clone();
                        let options = Options::new(compilation_mesh)
                            .with_input_shardings(vec![input.with_mesh(logical_mesh.clone()).unwrap()])
                            .with_output_shardings(vec![output.with_mesh(logical_mesh).unwrap()]);
                        let r#type = ArrayType::new_static(data_type, [4]);
                        let lowered = crate::stage(|input| input, r#type, options).and_then(
                            |staged: crate::StagedFunction<ArrayType, ArrayType>| {
                                staged.into_inner().lower(&session).map(|_| ())
                            },
                        );
                        let lowered = lowered.map_err(|error| error.to_string());
                        assert_eq!(lowered, Ok(()), "{axis_type:?} {data_type} {input} -> {output}");
                    }
                }
            }
        }
        assert_eq!((planned, declined), (828, 772));
    }
}
