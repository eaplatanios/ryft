//! Placement of runtime [`Array`]s on other devices, shardings, memories, and layouts.
//!
//! [`Array::to_placement`] copies an array to a destination and leaves the source usable, and
//! [`Array::into_placement`] consumes the source so that a compiled reshard may reuse its buffers. Both are the
//! analogues of `jax.device_put` for arrays that already live on devices, and [`Session::place`] and the placement
//! operation of `ryft_core` use them for runtime arrays. Host values go through the uploads of [`crate::arrays::host`]
//! instead. The same device work also moves the inputs of compiled programs and eager operations whose placement
//! differs from the one that the program needs.
//!
//! # Strategies
//!
//! A placement that keeps the devices and sharding of its source only changes the storage of each shard, copying it
//! to the requested memory and converting it to the requested layout on its device, or through host memory when that
//! conversion cannot be compiled. Any other placement chooses one device strategy, which is the first of the
//! following that applies.
//!
//!   1. **Exact Shards:** Every destination shard has the slices of one source shard, and both shardings hold one
//!      value per slice. Each destination shard shares the buffer of its source shard on the same device, copies it
//!      from another device of its process, or receives it through a PJRT cross-host transfer. Moving the shards
//!      compiles nothing, but a sender compiles a conversion on its device for a shard that it cannot send as it is
//!      (i.e., in another layout than the one that the receiver allocates, or with a dynamic allocation).
//!   2. **Shard Regions:** The source declares no layout other than the dense row-major one, both shardings hold one
//!      value per slice, and either the meshes differ or uneven shards or manual destination axes rule out a compiled
//!      reshard. The source devices slice out the regions that each destination needs, from device memory, and the
//!      destination devices join them and move them to the requested memory and layout.
//!   3. **Compiled Reshard:** An identity program with input and output sharding annotations runs on the mesh of the
//!      source. Across meshes, a replicated copy of the source (i.e., the source itself or its all-gather on its own
//!      mesh) is first copied or transferred to every destination device as whole shards, and the program runs on the
//!      destination mesh. Such a program keeps the reduction state that the destination keeps on the same mesh (i.e.,
//!      it moves the contributions of a pending sum, maximum, or minimum as they are), and it completes a pending
//!      reduction that the destination drops and introduces a pending sum or a completed reduction that the destination
//!      adds, with the arithmetic of XLA. A process that owns no device of the mesh of such a program compiles and runs
//!      nothing for it, and its result has the type and the placement of the results of the processes that run it,
//!      without buffers.
//!   4. **Host Round Trip:** The collectives of XLA compute what the operation of a data type computes in a compiled
//!      program, which is the value that the host readback of a pending reduction defines up to the sign of a zero
//!      result (refer to [`crate::arrays::host`]), only for some data types, kinds of reduction, and changes of
//!      reduction state (e.g., not for a sum of one-bit integers, which XLA adds as predicates, nor for floating point
//!      maxima and minima, from which XLA drops NaN values), and they cannot introduce a pending maximum or minimum
//!      (refer to `validate_reduction_transition`). Such a change takes the host fallback below, which reads the value
//!      back and uploads it with the host functions that define it, when one process owns every device of the source
//!      and of the destination. That process does the round trip, and every other process returns the placement without
//!      buffers. Otherwise no process can read the value back alone, so every process returns the same error before it
//!      does anything. A pending sum that the destination introduces for a data type without an additive identity
//!      (i.e., [`DataType::F8E8M0FNU`]) is rejected before anything else, with the error that a host upload of the same
//!      value reports, because no strategy can produce it.
//!
//! The choice reads only information that every process shares (i.e., the type and shard metadata of the source,
//! whether the source has locally assigned devices, the destination, and the capabilities of the PJRT plugin of the
//! session, which do not depend on whether the session shares compilation artifacts with other processes), so all the
//! processes of a distributed placement choose the same strategy and issue matching transfers and collectives. What
//! only one process knows (e.g., the physical layout of a buffer when the type declares none, or which buffers it
//! holds) never changes the choice. A source with locally assigned devices takes none of these strategies, as the next
//! section describes.
//!
//! # Locally Assigned Devices
//!
//! The shard metadata of a source is information that every process shares only when every process placed the source
//! on the same devices. A source whose devices this process chose on its own (e.g., a host value that each process
//! uploads without a mesh, which lands on the first device of each process) has a mesh that differs on every process,
//! so a strategy planned from it would make each process send its shards to the others, and no process would plan the
//! matching receives. Every array therefore records whether it has such locally assigned devices (refer to
//! `Array::is_locally_assigned`), and the following invariant holds.
//!
//!   - **Meaning:** An array is locally assigned when this process chose its devices without any information
//!     that the other processes share, so each process may hold the same value on devices of its own choice. Such an
//!     array lives on devices of this process only.
//!   - **Sources:** Host uploads without a mesh of the caller or of the session set the fact, and so do
//!     [`Session::place`] uploads of host values without a destination or with a destination that names only a
//!     memory or a layout.
//!   - **Propagation:** Placements that name only a memory or a layout, copies of an array on its own devices, moves
//!     of an array between devices of this process for an operation, and eager operations whose array inputs all have
//!     the fact keep it.
//!   - **Clearing:** A placement onto a device, a mesh, or a sharding clears it, even when that destination holds the
//!     devices of its source. Construction on a caller-supplied mesh and direct execution on a compiled mesh also
//!     clear it. A call without a placement request inherits local assignment from its public arguments.
//!
//! The fact is separate from [commitment](Array::is_committed), which records whether the devices were chosen on
//! purpose and which a destination that names only a memory or a layout also sets, as in JAX, so commitment cannot
//! tell whether the other processes know the devices. Every process of a distributed program applies the same
//! functions to the same kinds of inputs, so every process computes the same fact for its own source. Every process
//! therefore chooses the same strategy for a locally assigned source. Each process places its full value on its own
//! devices of the destination with local copies, slices, or a host upload, so no process transfers anything
//! or runs a collective, as JAX does for uncommitted arrays. A process whose devices hold the whole destination may do
//! that local work with the strategies above, because they then move nothing between processes.
//!
//! Clearing the fact for devices that the caller or the session names is sound only because every process of a
//! distributed run names the same devices there, which is a precondition that no process can check, because each one
//! sees only its own request. Every device, mesh, and sharding that a program names (e.g., the default mesh of a
//! [`Session`], the mesh of a construction, or the device of an [`PlacementRequest::Device`] request) must therefore be
//! the same on every process of the run, and a choice that differs between processes is expressible only by naming no
//! device, which leaves the choice to each process and records the fact. For example, when each process places a value
//! on its own first addressable device with [`PlacementRequest::Device`], or builds its session with a default mesh of
//! its own addressable devices and uploads a value without a mesh, and then places the result replicated over the
//! devices of every process, every process plans whole shards from its own device as the source of the value, so each
//! one sends its shard to the others and none of them receives, and the placement never completes.
//!
//! A cross-host transfer moves the whole allocation of a buffer in device memory into a static buffer in the default
//! layout and the default device memory of the receiver, so the sender first copies a shard that its type declares in
//! host memory to device memory and converts any buffer whose allocation differs from that of the receive (i.e., one in
//! another layout or one whose type has dynamic dimensions) to a static buffer in that layout, on its device or through
//! host memory, once for all the receivers of that shard. Every process issues its cross-host transfers in one global
//! order of device pairs, and prepares the buffers that it sends and looks up the devices that it receives on before
//! the first one, because a transfer that one process issues without its peer stays paired with the next transfer
//! between the same devices. A receiver
//! copies its buffers to the requested memory only after it issued all of its transfers, so that a memory that only it
//! lacks never leaves a sender waiting.
//!
//! When the device work fails, [`Array::to_placement`] and [`Array::into_placement`] fall back to a host round trip
//! that reads the whole value back and uploads it again. This needs a destination that this process fully addresses
//! and a source that this process can read back, which holds a local replica of every contribution to its value and
//! whose devices that hold one contribution do not vary along a manual mesh axis (refer to [`Array::to_placement`]).
//! Otherwise, they report the error of the device work.
//!
//! # Completion and Failures
//!
//! Placements return without waiting for the device work that they start. The execution fence of a result joins the
//! execution fence of its source with the readiness of every buffer that the placement copies, receives, or uploads
//! and with the fences of the programs that produce its buffers (e.g., compiled reshards, slices, and conversions),
//! because a readback reads only one replica of each tile and would otherwise miss a failed replica or a failure of
//! the work that produced the source. A result also retains the completion fence of the cross-host sends of its
//! process (i.e., its send fence), which no placement waits for, because a send completes only once its receiver
//! issues the matching receive. The observations of a result report the following failures.
//!
//!   - [`Array::ready`] and [`Array::block_until_ready`] report the failures of every addressable buffer, of the
//!     execution fence, and of the send fence.
//!   - [`Array::copy_to_host`] and [`Array::copy_to_host_bytes`] report the failures of the execution fence and of
//!     the buffers that they read. They never wait for the send fence, because a send may wait for a receive that
//!     another process issues only after its own readback, so a send that fails only on its receiver is not reported
//!     there. A result without elements copies nothing and waits for its execution fence, or for every addressable
//!     buffer when it has none.
//!   - [`Session::synchronize`] reports the failures of every copy, transfer, upload, and program that a placement
//!     registered with the session, including its sends.
//!
//! A placement blocks only on the first cross-host rendezvous between two devices, which the CUDA plugin performs on
//! the calling thread, and on host readbacks when the PJRT plugin lacks host to device transfer managers or their
//! literal transfers, whereas `jax.device_put` never blocks its caller. A failure after the transfers (e.g., a
//! destination memory that a device of the receiver lacks) is returned only once every transfer is issued, so the
//! peers of the failing process complete their transfers.
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
//! transfers are the following failures, which only the failing process can see.
//!
//!   - **A Sent Slice Without Any Local Buffer:** A process that imported an array with only metadata for a tile that
//!     the plan has it send (i.e., without a buffer for that tile or for any local replica of it) returns
//!     [`ArrayError::MissingShardBuffer`] before it issues anything, while the receiver of that tile issues its receive
//!     and waits.
//!   - **A Program That Reads The Source:** A compiled program whose input is the source (i.e., a compiled reshard on
//!     the mesh of the source, or the gather of a reshard across meshes) reads every shard of this process, so a
//!     failure to start that program on this process stops it before that program, while its peers wait in its
//!     collectives and in the transfers that follow it. Such failures are a missing buffer of any of these shards, a
//!     failed conversion of a buffer whose physical layout only this process knows, and a failed launch (e.g., a
//!     failed allocation).
//!   - **A Failed Preparation Of A Sent Buffer:** A copy, a staging in device memory, or a conversion to the layout of
//!     the receiver that fails to start stops this process before its transfers.

use std::collections::HashMap;
use std::fmt::Display;
use std::ops::Range;
use std::sync::Arc;

use ryft_core::{
    ArrayType, CastOperation, Concatenate, DataType, Device, DeviceMesh, DimensionError, Layout, Memory, MeshAxisType,
    PLACE_OPERATION_NAME, Placeholder, Placement, PlacementRequest, ProgramError, Sharding, Slice, StaticShape,
    TiledLayout, TypeError, Typed, UnreducedKind,
};
use ryft_pjrt::extensions::cross_host_transfers::{CrossHostTransferKey, GlobalDeviceId};
use ryft_pjrt::{
    Buffer, BufferSpecification, BufferType, Client, Device as PjrtDevice, DeviceId, Error as PjrtError, Event,
    ExecutionFence, Memory as PjrtMemory,
};

use crate::arrays::host::{
    CollectiveReduction, HostBufferSlice, PendingHostUpload, ReductionChange, collective_reduction,
    requires_unreduced_sum_identity, run_host_uploads, unreduced_sum_identity, validate_mesh_device_processes,
};
use crate::arrays::metadata::validate_shard_layout;
use crate::arrays::placement::{resolve_array_placement, single_device_placement};
use crate::arrays::shards::{ArrayShard, ShardDescriptor, ShardIndex, ShardLayout, read_shard_to_host};
use crate::arrays::{Array, ArrayError};
use crate::devices::PjrtDeviceExtension;
use crate::types::DataTypeExtension;
use crate::{
    CompiledFunction, Error, FlatProgram, FromPjrt, LoweringError, Options, ProgramBuilder, Session, ToPjrt, Type,
};

// TODO(eaplatanios): Review this module.

impl Array {
    /// Moves or copies this array to the provided placement without donating the source buffers, so the original
    /// [`Array`] remains fully usable after the call returns. This is the `ryft` analogue of JAX's
    /// `jax.device_put(arr, sharding, donate=False)`.
    ///
    /// When the placement and storage already match, the result is this array (i.e., the same logical array, refer to
    /// the identity rules of [`RetainedArray`](crate::RetainedArray)). If each process chose the devices of this array
    /// on its own, a destination that names devices gives a new array on every process instead, which shares the
    /// buffers of this array where they already are, so that independent placements remain distinct when captured.
    /// Every other placement gives a new array. A change of devices or sharding tries the following strategies in
    /// order.
    ///
    ///   1. **Exact Shards:** When every destination shard has the slices of one source shard, each destination shard
    ///      shares the buffer of its source shard on the same device, or is filled by a copy between devices or by a
    ///      PJRT cross-host transfer for a source shard on a remote host.
    ///   2. **Shard Regions:** Across meshes, or on the same mesh for uneven shards or manual destination axes, when
    ///      both shardings hold one value per slice and this array declares no layout other than the dense row-major
    ///      one, the source devices slice
    ///      out the regions each destination needs. Each destination joins its regions locally or allocates an empty
    ///      buffer for an empty shard. The regions move through default device memory, and each process then moves its
    ///      own destination shards to the requested memory and layout. This path needs the PJRT cross-host transfers
    ///      extension and default layouts if a region crosses processes. Each participating process issues the same
    ///      placements in the same order.
    ///   3. **Compiled Reshard:** An `identity(x) = x` program with input and output sharding constraints is lowered
    ///      to StableHLO and Shardy, compiled through the compilation cache of `session`, and executed. This needs
    ///      shards of one size on both sides, so a split whose parts do not divide the extent of its dimension (e.g.,
    ///      five elements over two devices, which [`ShardLayout`] splits into shards of three and two elements) skips
    ///      this strategy. Only the processes that own a device of the mesh of such a program compile and run it, and
    ///      every other process returns the same type and placement without buffers.
    ///   4. **Host Fallback:** The global array is copied to the host as [`Array::copy_to_host`] does and uploaded
    ///      again as [`Array::from_host_buffer`] does, so the result has the shards that a host upload of the same
    ///      values with the destination sharding has. This is used only when the device work fails or is skipped (e.g.,
    ///      when no strategy supports a manual destination, or when a compiled reshard would change the value of an
    ///      array whose reduction state the placement changes, refer to [`crate::arrays::resharding`]), when the
    ///      destination mesh is fully addressable from this process, and when this process can read the source back.
    ///      That is the case when this process holds a buffer for one replica of every distinct contribution to the
    ///      value with elements (i.e., of every tile and, along unreduced axes, of every per-device value of a tile),
    ///      whatever the other processes hold, and when no two devices that hold one such contribution differ along a
    ///      varying manual mesh axis, because those devices can hold different values for the same slice that a dense
    ///      host copy cannot represent. Devices that differ along a manual axis which splits the array hold distinct
    ///      tiles, so they read back like any other tiles. The destination buffers are created up front and the
    ///      readback and the upload continue on a background thread, so this call returns without waiting for them. A
    ///      failure of that background work, whose error also describes the failure of the device work, surfaces
    ///      through the readiness functions and the readbacks of the result and through [`Session::synchronize`]. Only
    ///      when the PJRT plugin lacks host to device transfer managers or their literal transfers does this call block
    ///      until the readback completes, and the readiness functions and readbacks of the result then report a failed
    ///      upload of any of its replicas.
    ///
    /// A split whose parts do not divide the extent of its dimension (i.e., uneven shards) is therefore supported by
    /// whole shards, by shard regions, and by the host fallback only. Shard regions handle a source in any memory and
    /// with no declared layout or a declared dense row-major one, as long as both shardings hold one value per slice.
    /// An uneven split of a source that declares another layout (e.g., a column-major one) or whose sharding or
    /// destination has unreduced or varying manual mesh axes needs the host fallback, and so fails when the
    /// destination includes devices of other processes.
    ///
    /// A source whose devices this process chose on its own (e.g., the result of [`Array::from_host_buffer`] without
    /// a mesh, also after a placement that names only a memory or a layout) is a full value that every process holds
    /// on devices of its own choice. Every process then places its own value on its own destination devices instead of
    /// using these strategies, by sharing, copying, or slicing it on its device or by uploading its readback, and no
    /// process transfers anything (refer to [`crate::arrays::resharding`]). Every process of such a placement must
    /// pass its own such array.
    ///
    /// The host fallback reads back the combined value of an array with unreduced axes and uploads it again as
    /// [`Array::from_host_buffer`] does, so when the destination keeps an unreduced axis, it redistributes the
    /// contributions along that axis. With [`UnreducedKind::Sum`], the devices at coordinate zero along it hold the
    /// whole value and the others hold the additive identity. With [`UnreducedKind::Max`] or [`UnreducedKind::Min`],
    /// every device along it holds the whole value, which is its own extremum. The logical value is the same, but the
    /// shards of the result can then differ from those that a device strategy would produce.
    ///
    /// The result always belongs to `session`, even when it is this array or shares its buffers, and an array of
    /// another PJRT client is rejected. The result is [committed](Self::is_committed) to its devices, as in JAX.
    ///
    /// When the host fallback also cannot satisfy the request, typically because some source shards that the value
    /// needs live on a remote process or because the source varies along manual mesh axes, the original error of the
    /// device work is propagated as the most informative diagnostic.
    ///
    /// The destination uses the requested [`Memory`]. When the memory is omitted, a destination that names a single
    /// device (refer to [`PlacementRequest::Device`]) uses the default memory of that device, as
    /// `jax.device_put(x, device)` does, and a mesh or sharding destination keeps the memory of the source. PJRT
    /// cross-host receives land in device memory and can then be copied to the requested memory on that device. An
    /// explicit [`Layout`] applies to each destination shard. When the layout is omitted, the result keeps the layout
    /// of the source only while that layout still describes each destination shard (e.g., the strides of a column
    /// shard do not describe the whole value that a gather produces), and declares no layout otherwise. Layout changes
    /// use a compiled conversion on each shard's device when supported. Otherwise, each affected local shard is read
    /// back and uploaded in the requested layout on a background thread, as for the host fallback, and this call only
    /// blocks on the readback when the PJRT plugin lacks host to device transfer managers or their literal transfers.
    /// Both paths preserve the separate contributions along unreduced axes.
    ///
    /// Use [`Self::into_placement`] when the source array is no longer needed after the move. It consumes `self` and
    /// lets PJRT reuse the source buffers for the destination where it can, which may avoid an extra allocation.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `target`: Resolved [`Placement`] or a [`PlacementRequest`] specifying the destination.
    pub fn to_placement<P: Into<PlacementRequest>>(&self, session: &Session, target: P) -> Result<Self, Error> {
        self.clone().run_placement_request(session, target.into(), false)
    }

    /// Moves this array to the provided placement, consuming `self` and donating its source buffers to the
    /// destination. This is the `ryft` analogue of JAX's `jax.device_put(arr, sharding, donate=True)`.
    ///
    /// Donation is best effort, as in JAX. Only the compiled reshard on the mesh of this array donates its input, and
    /// PJRT may then reuse a source buffer for a destination shard, which can save the memory of the source array for
    /// large training tensors. Nothing guarantees that saving. A buffer can be reused only for a destination shard that
    /// has the shape of the source shard on its device, and the lowering of the reshard pairs its output with its
    /// donated input only when their shardings are equal, which they never are for a compiled reshard (a placement
    /// that keeps the sharding only changes the storage of each shard), so the compiled reshard currently reuses no
    /// buffer (e.g., moving a uniquely owned `64x64` array from row shards to column shards on four devices allocates
    /// four new buffers). The other strategies (e.g., moves of whole shards, shard regions, a reshard across meshes, or
    /// the host fallback) do not donate, a buffer that another handle shares is never donated, and PJRT may still
    /// allocate new output buffers. Because `self` is consumed at the language boundary, the donated buffers are not
    /// observable to the caller after this function returns. Rust ownership therefore enforces the same invariant that
    /// the `donate=True` flag carries in JAX, without exposing a runtime hazard.
    ///
    /// The strategies follow [`Self::to_placement`], and donation never changes which of them is chosen, so the
    /// compiled reshard of an array with a pending reduction (i.e., unreduced axes in its runtime sharding) completes
    /// or keeps the reduction on the devices with or without donation, and a change of reduction state that the
    /// compiled reshard cannot make (e.g., introducing a pending maximum, or completing a pending maximum of floating
    /// point values) takes the host fallback from both entry points. Donation changes only what follows a
    /// compiled reshard that fails after its launch, which may have handed the buffers of this array to PJRT.
    /// [`Self::to_placement`] then still falls back to a host transfer, whereas this function returns the error.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `target`: Resolved [`Placement`] or a [`PlacementRequest`] specifying the destination.
    pub fn into_placement<P: Into<PlacementRequest>>(self, session: &Session, target: P) -> Result<Self, Error> {
        self.run_placement_request(session, target.into(), true)
    }

    /// Places this array as `request` asks, which [`Self::to_placement`] and [`Self::into_placement`] implement. The
    /// request is resolved against this array (refer to [`resolve_array_placement`]) and placed with
    /// [`Self::run_placement_dispatch`]. Only a request that names neither devices nor a sharding (i.e., an
    /// [`PlacementRequest::InPlace`] request, which names only a memory or a layout) leaves the result on the
    /// devices that this process chose on its own when this array has such devices, because every other request names
    /// the devices of the result (refer to [`Self::is_locally_assigned`]).
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `request`: [`PlacementRequest`] that specifies the destination.
    ///   - `donate`: Whether a compiled reshard on the mesh of this array may reuse the buffers of this array for
    ///     its output (refer to [`Self::run_placement_dispatch`]).
    pub(crate) fn run_placement_request(
        self,
        session: &Session,
        request: PlacementRequest,
        donate: bool,
    ) -> Result<Self, Error> {
        let in_place = matches!(request, PlacementRequest::InPlace { .. });
        let locally_assigned = self.is_locally_assigned();
        let target = resolve_array_placement(request, &self)?;
        let output = self.run_placement_dispatch(session, &target, donate)?;
        // A destination that names devices can already hold the input that this process chose on its own, while
        // another process must move its input there, so such a placement is a new array on every process (refer to
        // the identity rules of `RetainedArray`).
        let output = if locally_assigned && !in_place { output.with_new_identity() } else { output };
        Ok(output.with_locally_assigned(locally_assigned && in_place))
    }

    /// Places this array on `mesh` according to `sharding` with device work only (i.e., without copying its contents
    /// through host memory). [`Self::to_placement`] and [`Self::into_placement`] start with this work, and the
    /// execution of a compiled program uses it for inputs whose sharding differs from the one that the program was
    /// compiled for. Refer to [`Self::reshard_on_devices_or_return`] for the strategies that it tries.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that executes the reshard and owns the resulting array. It must use the same PJRT
    ///     client as this array.
    ///   - `mesh`: Destination [`DeviceMesh`].
    ///   - `sharding`: Destination [`Sharding`].
    ///   - `donate`: Whether a compiled reshard on the mesh of this array may reuse the buffers of this array for
    ///     its output. The donation only takes effect when this array is the only handle of its buffers.
    ///
    /// A result on a single device keeps the [commitment](Self::is_committed) of this array, because this is how arrays
    /// move to the devices that an operation runs on rather than to devices that a caller chose. A result on more than
    /// one device is committed, like every eager result on more than one device (refer to [`Self::commit_if`]). For the
    /// same reason, a result on devices of this process only keeps the [locally assigned
    /// devices](Self::is_locally_assigned) of this array, while a result on devices of other processes never has
    /// them.
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
        let resolve_layout = cross_host_layout_resolver(session.client());
        array
            .reshard_on_devices_or_return(&resolve_layout, mesh, sharding, memory, layout.as_ref(), donate)
            .map(|output| output.commit_if(committed).with_locally_assigned(locally_assigned))
            .map_err(|(error, _)| error)
    }

    /// Returns an array with the type, the placement, the commitment, and the [locally assigned
    /// devices](Self::is_locally_assigned) of this array whose addressable shard buffers are copies of the
    /// buffers of this array for which `copy` returns `true`, on the same devices and in the same memories, and the
    /// buffers of this array otherwise. With a `copy` that selects every buffer, the result shares no buffer with any
    /// other array. `copy` is called once for each buffer, in shard order, before anything is copied. The execution
    /// fence of the result joins the readiness of every copy with the fence of this array, because a readback reads
    /// only one replica of each tile and would otherwise succeed with a replica whose copy did not fail.
    /// The array's [session](Self::session) tracks the copy work and counts the result as a new array.
    pub(crate) fn with_copied_shard_buffers<F: FnMut(&Arc<Buffer<'static>>) -> bool>(
        &self,
        copy: F,
    ) -> Result<Self, Error> {
        let copies = self.addressable_shards().filter_map(ArrayShard::buffer).map(copy).collect::<Vec<_>>();
        let session = self.session();
        let mut work = session.reserve_work();
        let mut completion = PlacementCompletion::default();
        let buffers = self
            .addressable_shards()
            .filter_map(ArrayShard::buffer)
            .zip(copies)
            .map(|(buffer, copy)| {
                if !copy {
                    return Ok(Arc::clone(buffer));
                }
                let copied = buffer.copy_to_memory(buffer.memory()?)?;
                work.track_buffer(&copied);
                completion.track_buffer(&copied)?;
                Ok(Arc::new(copied))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let r#type = self.r#type().into_owned().with_sharding(None)?;
        let mut output = if self.data_type().is_zero() {
            Self::zero(session, r#type, Some(self.sharding().clone()), self.mesh().clone())?
        } else {
            let (sharding, mesh) = (self.sharding().clone(), self.mesh().clone());
            Self::from_addressable_buffers_internal(session, r#type, Some(sharding), mesh, buffers, false)?
        };
        output.execution_fence = completion.join(self.execution_fence.clone());
        output.send_fence = self.send_fence.clone();
        Ok(output.with_committed(self.is_committed()).with_locally_assigned(self.is_locally_assigned()))
    }

    /// Places this array on the devices, with the sharding, and in the storage that `target` names. A memory that
    /// `target` omits keeps the memory of this array. That is the memory that the request chose, because a request
    /// resolves its default memory into `target` (e.g., the default memory of the device that a request for a single
    /// device names, refer to [`PlacementRequest::target_memory`](ryft_core::PlacementRequest::target_memory)). A
    /// layout that `target` omits keeps the layout of this array only while that layout still describes the destination
    /// shards (refer to [`Self::placement_layout`]). The result always belongs to `session`, so this array is first
    /// associated with `session`, which rejects an array of another PJRT client, even when the placement leaves the
    /// array where it is. The result is [committed](Self::is_committed) to its devices, because they were chosen on
    /// purpose. `target` names those devices, so the result never has [locally assigned
    /// devices](Self::is_locally_assigned), even when they are the devices of this array, and a caller whose
    /// request named only a memory or a layout restores that fact (refer to [`Self::run_placement_request`]).
    ///
    /// A requested layout is the physical layout of each destination shard, so it must address the shape of every
    /// destination shard (refer to [`validate_shard_layout`]), and every device of the destination must belong to the
    /// process that the destination assigns to it (refer to [`validate_mesh_device_processes`]). Both are checked
    /// before any device work, because a placement without buffers (e.g., of a [`DataType::Zero`] array) would
    /// otherwise never check the layout, and a device that the destination misattributes would receive no buffer in
    /// any process. Every strategy records the same layout, the one that [`Self::placement_layout`] resolves from
    /// information that every process shares, so processes that take different strategies record the same layout.
    ///
    /// An array that this process can read back (refer to the host fallback of [`Self::to_placement`]) can also use a
    /// host round trip when the device reshard fails and returns the source array. That round trip returns the result
    /// before the readback completes and continues the readback and the upload on a background thread, whose failures
    /// surface through the readiness functions and the readbacks of the result and through [`Session::synchronize`].
    /// It blocks until the readback completes only when the PJRT plugin lacks host to device transfer managers or their
    /// literal transfers, and the execution fence of the result then joins the readiness of its uploads. Whatever this
    /// function can reject before reading back (e.g., a destination mesh that this process does not fully address or
    /// a source that this process cannot read back) is still reported by this function with the error of the device
    /// work. That includes a source whose devices hold one slice but vary along a manual mesh axis, whose per-device
    /// values a dense host copy cannot preserve.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `target`: Resolved destination [`Placement`] (refer to [`resolve_array_placement`]).
    ///   - `donate`: Whether a compiled reshard on the mesh of this array may reuse the buffers of this array for
    ///     its output. The donation only takes effect when this array is the only handle of its buffers.
    pub(crate) fn run_placement_dispatch(
        self,
        session: &Session,
        target: &Placement,
        donate: bool,
    ) -> Result<Self, Error> {
        let resolve_layout = cross_host_layout_resolver(session.client());
        // A placement may return the input or share its buffers. Associate it with the destination session before
        // choosing a strategy. Consuming `self` keeps the buffers uniquely owned for a donation.
        let array = self.with_session(session)?;
        let (mesh, sharding) = (target.mesh().clone(), target.sharding().clone());
        let memory = target.memory().unwrap_or_else(|| array.r#type().memory());
        let layout = array.placement_layout(target.layout(), &mesh, &sharding)?;
        let placed = |array: Self| array.with_committed(true).with_locally_assigned(false);
        if array.mesh() == &mesh && array.sharding() == &sharding {
            return Ok(placed(array.place_storage(memory, layout)?));
        }
        validate_mesh_device_processes(array.client(), &mesh)?;

        // Keep the error of the device work, which is the most informative one if the host fallback fails too.
        let (compiled_error, source) = match array.reshard_on_devices_or_return(
            &resolve_layout,
            &mesh,
            &sharding,
            memory,
            layout.as_ref(),
            donate,
        ) {
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
        let destination_fully_addressable = mesh.devices().iter().all(|device| device.process_index() == local_process);
        if !destination_fully_addressable || source.validate_host_readback().is_err() {
            return Err(compiled_error);
        }
        let host_type =
            ArrayType::new(source.data_type(), source.shape().into()).with_memory(memory).with_layout(layout);

        // Prefer to read back and upload in the background. Its failure poisons the result with an error that also
        // describes the failure of the device work, because that failure is what made the host fallback necessary.
        let readback_source = source.clone();
        let device_error = compiled_error.to_string();
        let Ok(readback_fence) = source.host_readback_fence() else {
            return Err(compiled_error);
        };
        let uploaded = Self::from_host_buffer_in_background(
            source_session,
            host_type.clone(),
            mesh.clone(),
            sharding.clone(),
            readback_fence,
            move || readback_source.copy_to_host_bytes(),
            move |error| {
                PjrtError::internal(format!(
                    "the host fallback of an array placement failed with: {error} (the device work that it replaced \
                    failed with: {device_error})",
                ))
            },
        );
        match uploaded {
            Ok(Some(array)) => return Ok(placed(array)),
            Ok(None) => {}
            Err(_) => return Err(compiled_error),
        }

        // Without host to device transfer managers, read back and upload on this thread.
        let host_bytes = match source.copy_to_host_bytes() {
            Ok(bytes) => bytes,
            Err(_) => return Err(compiled_error),
        };
        let uploaded = Self::from_host_buffer(source_session, host_type, mesh, Some(sharding), host_bytes.as_slice())
            .and_then(|array| array.with_upload_completion(source.execution_fence().cloned()));
        match uploaded {
            Ok(array) => Ok(placed(array)),
            Err(_) => Err(compiled_error),
        }
    }

    /// Returns the layout that each shard of a placement of this array on `mesh` according to `sharding` has, which is
    /// `requested` when the placement names a layout, and the layout of this array otherwise, but only while that
    /// layout still describes the destination shards, as the type rule of the placement keeps it (e.g., the strides of
    /// a column shard do not describe the value that a gather produces). It reads only the type of this array and the
    /// destination, which every process shares, so every process and every strategy of a placement record the same
    /// layout. A requested layout is validated against the destination shards here, because a placement without
    /// buffers never meets a buffer that would reject it.
    fn placement_layout(
        &self,
        requested: Option<&Layout>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<Option<Layout>, Error> {
        if requested.is_none() && self.layout().is_none() {
            return Ok(None);
        }
        let destination_layout = ShardLayout::new(&self.shape(), mesh, sharding)?;
        let validate = |layout: &Layout| {
            let r#type = self.r#type().into_owned().with_layout(layout.clone());
            let shard_shapes = destination_layout.descriptors().iter().map(ShardDescriptor::shape);
            validate_shard_layout(&r#type, shard_shapes)
        };
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

    /// Returns this array, which a host upload of a placement has just created on the calling thread (e.g., with
    /// [`Self::from_host_buffer`] when the PJRT plugin lacks host to device transfer managers), with an execution fence
    /// that joins its own fence, if any, and `fence` (i.e., the execution fence of the source of the placement) with
    /// the readiness of every addressable buffer of this array. A readback reads only one replica of each tile, so
    /// without that readiness it could succeed although the upload of a replica that the placement created failed,
    /// which the readbacks of a placement report as its background uploads do. Plain host uploads keep no such fence,
    /// because every one of their replicas receives the same host bytes (refer to [`crate::arrays::host`]).
    fn with_upload_completion(mut self, fence: Option<ExecutionFence>) -> Result<Self, Error> {
        let mut completion = PlacementCompletion::default();
        completion.track_fence(fence.as_ref());
        for buffer in self.addressable_shards().filter_map(ArrayShard::buffer) {
            completion.track_buffer(buffer)?;
        }
        self.execution_fence = completion.join(self.execution_fence.take());
        Ok(self)
    }

    /// Applies storage requirements to the shards already on their destination devices. A layout change first tries a
    /// compiled conversion on the device of each shard, and uses a host round trip for that shard when the plugin
    /// cannot lower or compile that conversion (refer to [`Self::try_relayout_shard_on_device`]). Once a buffer
    /// has the requested layout, it can be reused or copied directly to the requested memory. The result keeps the
    /// devices of this array, and with them its [locally assigned devices](Self::is_locally_assigned). It declares
    /// exactly `layout`, which [`Self::placement_layout`] resolved from information that every process shares, whatever
    /// layout the strategy that produced this array recorded, so that every process records the same layout. Without
    /// `layout`, it declares no layout and keeps the buffers as they are.
    ///
    /// The host round trip of each shard is its own job on the background upload workers of the array's
    /// [session](Self::session). The job reads the shard back once its source buffer is ready and uploads it into a
    /// destination buffer that this function creates up front. This function returns without waiting for these jobs,
    /// and each job waits only for its own shard. A failed readback or upload poisons the destination buffer, so the
    /// readiness functions and readbacks of the result and [`Session::synchronize`] report it. The execution fence of
    /// the result joins the fences of this array with the readiness of every buffer that this function creates (i.e.,
    /// the copies to another memory and the buffers of host round trips) and with the fences of the compiled
    /// conversions, because a readback reads only one replica of each tile and would otherwise succeed with a replica
    /// that did not fail. This function blocks on the readback of a shard only when the PJRT plugin lacks host to
    /// device transfer managers or their literal transfers.
    fn place_storage(self, memory: Memory, layout: Option<Layout>) -> Result<Self, Error> {
        if self.r#type().memory() == memory && self.layout() == layout.as_ref() {
            return Ok(self);
        }

        let session = self.session();
        let r#type = self.r#type().into_owned().with_memory(memory).with_layout(layout);
        let mesh = self.mesh().clone();
        let sharding = Some(self.sharding().clone());
        let mut completion = PlacementCompletion::default();
        let mut output = if r#type.data_type().is_zero() {
            Self::zero(session, r#type, sharding, mesh)?
        } else {
            let target_layout = r#type.layout().map(ToPjrt::to_pjrt).transpose()?;
            let buffer_type = r#type.data_type().executable_buffer_type();

            // Register the submission before the first copy, so that a concurrent `Session::synchronize` snapshot
            // includes every copy that may already be running. The host round trips register their own work.
            let mut work = session.reserve_work();
            let mut buffers = Vec::new();
            for shard in self.addressable_shards() {
                let source_buffer = shard.buffer().unwrap();
                let device = source_buffer.device()?;
                let destination_memory = device.memory(memory)?;
                let layout_change = if let Some(target_layout) = &target_layout {
                    let current_layout = source_buffer.specification()?.layout;
                    (current_layout != *target_layout).then_some(current_layout)
                } else {
                    None
                };
                let changes_layout = layout_change.is_some();
                let buffer = match layout_change {
                    Some(current_layout) => {
                        let input_type = ArrayType::new(self.data_type(), shard.shape().into())
                            .with_memory(self.r#type().memory())
                            .with_layout(Layout::from_pjrt(current_layout)?);
                        Self::try_relayout_shard_on_device(
                            session,
                            shard.device(),
                            Arc::clone(source_buffer),
                            input_type,
                            r#type.layout().unwrap(),
                        )?
                        .map(|converted| {
                            completion.track_fence(converted.execution_fence());
                            Arc::clone(converted.shards()[0].buffer().unwrap())
                        })
                    }
                    None => Some(Arc::clone(source_buffer)),
                };
                if let Some(buffer) = buffer {
                    if buffer.memory()? == destination_memory {
                        buffers.push(buffer);
                        continue;
                    }
                    match buffer.copy_to_memory(destination_memory) {
                        Ok(copied) => {
                            work.track_buffer(&copied);
                            completion.track_buffer(&copied)?;
                            buffers.push(Arc::new(copied));
                            continue;
                        }
                        // Some plugins cannot copy the layout produced by the executable to the requested memory.
                        Err(PjrtError::Unimplemented { .. } | PjrtError::MissingFunction { .. }) if changes_layout => {}
                        Err(error) => return Err(error.into()),
                    }
                }

                // Read this shard in row-major order and upload it into the requested layout and memory.
                // A whole-array readback would combine unreduced contributions and may need remote shards.
                buffers.push(Self::copy_shard_through_host(
                    session,
                    source_buffer,
                    shard.shape(),
                    buffer_type,
                    destination_memory,
                    target_layout.clone(),
                    &mut completion,
                )?);
            }
            Self::from_addressable_buffers_internal(session, r#type, sharding, mesh, buffers, false)?
        };
        output.execution_fence = completion.join(self.execution_fence);
        output.send_fence = self.send_fence;
        Ok(output.with_locally_assigned(self.locally_assigned))
    }

    /// Tries a layout conversion on one device, keeping the shard's contents and memory unchanged. Returns `None`
    /// when lowering or compilation does not support the conversion, so the caller can upload the shard through
    /// host memory instead. Errors from executing a compiled conversion are propagated.
    fn try_relayout_shard_on_device(
        session: &Session,
        device: Device,
        buffer: Arc<Buffer<'static>>,
        input_type: ArrayType,
        layout: &Layout,
    ) -> Result<Option<Self>, Error> {
        // A single-device program treats this buffer as an independent value, even when it holds only one
        // contribution to an unreduced array. Its input uses the buffer's existing layout.
        let input = Self::wrap_device_buffer(session, input_type.clone(), device, buffer)?;
        let output_type = input_type.clone().with_layout(layout.clone());
        let mut builder = ProgramBuilder::new();
        let argument = builder.add_input(input_type.clone().into());
        let cast = CastOperation::<Type>::checked(output_type.into(), vec![input_type.into()])?;
        let output = builder.add_instruction(cast, Vec::new(), vec![argument], None)?[0];
        let program: FlatProgram = builder.build(vec![output], vec![Placeholder], vec![Placeholder])?;
        let compiled = session
            .lower_program(&program, 0, &Options::new(input.mesh().clone()))
            .and_then(|lowered| session.compilation_cache().compile(session, &lowered));
        match compiled {
            Ok(program) => Ok(Some(session.execute_program(&program, vec![input])?.pop().unwrap())),
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

    /// Returns a new buffer on the device of `buffer`, in `memory`, and in `layout` (or the default layout of the PJRT
    /// plugin for [`None`]) that holds the shard with `shape` that `buffer` holds, copied through host memory in dense
    /// row-major order. `completion` records the readiness of the new buffer, and the work tracking of `session`
    /// includes it, so [`Session::synchronize`] waits for it.
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
                        |error| match error {
                            Error::Pjrt(error) => error,
                            error => PjrtError::internal(error.to_string()),
                        },
                    )
                })?;
                placed
            }
            None => {
                let bytes = read_shard_to_host(buffer, &shape, |copy| work.track_host_copy(copy).wait())?;
                client.buffer(Arc::<[u8]>::from(bytes), buffer_type, dimensions, None, memory, layout)?
            }
        };
        work.track_buffer(&placed);
        completion.track_buffer(&placed)?;
        Ok(Arc::new(placed))
    }

    /// Places this array, which has [locally assigned devices](Self::is_locally_assigned) (refer to
    /// [`Self::device_placement_strategy`]), on the devices of this process in `mesh` according to `sharding` and in
    /// `memory`, with local work only. Every other process of the placement does the same with its own such array, so
    /// the result holds the shards of this process, and the other processes hold the rest. This is how JAX places an
    /// uncommitted array on devices of several processes, except that JAX also checks that the processes hold equal
    /// values when the destination includes every process. Nothing compares the values here, so every process must
    /// pass an equal value, as for [`Array::from_host_buffer`], and differing values leave the replicas inconsistent.
    ///
    /// A destination shard that covers the whole array shares the buffer of this array when it is on the same device
    /// and in the same memory, and is a copy otherwise. When a destination shard covers only a part of the array, that
    /// part is sliced on the device of this array, as for shard regions, and copied to its destination. Slicing needs an array in the default device memory without
    /// a declared layout, and every destination device must hold one value per slice. Otherwise (e.g., along an
    /// unreduced destination axis, whose devices away from coordinate zero hold additive identities), this process
    /// reads its value back and uploads the shards of its devices as [`Array::from_host_buffer`] does, in the
    /// background when the PJRT plugin supports it. The execution fence of the result joins the fence of this array
    /// with the readiness of the buffer of this array that it reads, of every buffer that this function creates, and
    /// the fences of the slices (or, for the upload, the readiness that its readback waits for), so the result waits
    /// for and reports the producer of this array even when no destination device belongs to this process.
    ///
    /// `layout` is the layout that the placement records for each destination shard (refer to
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
        let destination_layout = ShardLayout::new(&shape, mesh, sharding)?;
        let destinations = destination_layout
            .descriptors()
            .iter()
            .filter(|destination| destination.device().process_index() == process_index)
            .collect::<Vec<_>>();
        let whole =
            |slice: &[Range<usize>]| slice.iter().zip(shape.as_slice()).all(|(range, &extent)| range.len() == extent);
        let source = match self.shards() {
            [source] if source.buffer().is_some() => Some(source),
            _ => None,
        };
        let slices = destinations.iter().any(|destination| !whole(destination.slice()));
        let on_devices = source.is_some()
            && holds_one_value_per_slice(sharding)
            && (!slices || (self.r#type().memory() == Memory::Device && self.layout().is_none()));
        if let (true, Some(source)) = (on_devices, source) {
            let devices = client
                .addressable_devices()?
                .into_iter()
                .map(|device| Ok((device.id()?, device)))
                .collect::<Result<HashMap<_, _>, PjrtError>>()?;
            // The source buffer can be the only completion of the work that produced it (e.g., of a pending host
            // upload, which has no execution fence), and no destination buffer reads it when no destination device
            // belongs to this process, so the result waits for it directly.
            let mut completion = PlacementCompletion::default();
            completion.track_buffer(source.buffer().unwrap())?;
            let mut work = session.reserve_work();
            let mut buffers = Vec::with_capacity(destinations.len());
            for destination in destinations {
                let device_id = destination.device().id();
                let device =
                    devices.get(&device_id).ok_or(ArrayError::NonAddressableDevice { device_id, process_index })?;
                let source_buffer = Arc::clone(source.buffer().unwrap());
                let buffer = Self::prepare_shard_region(
                    session,
                    self.data_type(),
                    source,
                    source_buffer,
                    destination.slice(),
                    None,
                    &mut completion,
                )?;
                let destination_memory = device.memory(memory)?;
                if buffer.memory()? == destination_memory {
                    buffers.push(buffer);
                    continue;
                }
                let copied = buffer.copy_to_memory(destination_memory)?;
                work.track_buffer(&copied);
                completion.track_buffer(&copied)?;
                buffers.push(Arc::new(copied));
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
                buffers,
                false,
            )?;
            output.execution_fence = completion.join(self.execution_fence().cloned());
            return Ok(output);
        }

        // Read the value back and upload the shards of the devices of this process, as the host fallback of
        // `Self::run_placement_dispatch` does, but for a destination that other processes share.
        let host_type = ArrayType::new(self.data_type(), shape.into()).with_memory(memory).with_layout(layout.cloned());
        let readback_source = self.clone();
        let uploaded = Self::from_host_buffer_in_background(
            session,
            host_type.clone(),
            mesh.clone(),
            sharding.clone(),
            self.host_readback_fence()?,
            move || readback_source.copy_to_host_bytes(),
            |error| match error {
                Error::Pjrt(error) => error,
                error => PjrtError::internal(error.to_string()),
            },
        )?;
        match uploaded {
            Some(output) => Ok(output),
            None => {
                let host_bytes = self.copy_to_host_bytes()?;
                let sharding = Some(sharding.clone());
                Self::from_host_buffer(session, host_type, mesh.clone(), sharding, host_bytes.as_slice())?
                    .with_upload_completion(self.execution_fence().cloned())
            }
        }
    }

    /// Places this array on `mesh` according to `sharding` and in `memory` by sharing, copying, and transferring the
    /// whole shards that `plan` describes, which [`Self::device_placement_strategy`] chose with `cross_host_layouts`.
    /// [`Self::exact_shard_buffers`] provides the destination shards and the layout that the result records. The
    /// execution fence of the result joins the execution fence of this array, so that errors of the execution that
    /// produced this array still surface through the result, with the readiness of every buffer that this placement
    /// copies or receives and the fences of the conversions that it runs, because a readback reads only one replica of
    /// each tile and would otherwise succeed with a replica whose copy failed. The result also retains the completion
    /// fence of the cross-host sends that this process issued, so that the readiness functions of the result surface
    /// the errors of those sends.
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
        let (buffers, layout, send_fence) =
            self.exact_shard_buffers(plan, cross_host_layouts, memory, &mut completion)?;
        let r#type = self.r#type().into_owned().with_shape(self.shape()).with_memory(memory).with_layout(layout);
        let output = Self::from_addressable_buffers_internal(
            session,
            r#type.with_sharding(None)?,
            Some(sharding.clone()),
            mesh.clone(),
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

    /// Returns the buffers of the destination shards that `session` can address for the moves of whole shards that
    /// `plan` describes (refer to [`plan_exact_shard_put`]). A shard can share its source buffer when the device
    /// and memory match. Other local source shards are copied to the destination memory, and remote source shards
    /// arrive through the PJRT cross-host transfers extension. `cross_host_layouts` holds the default layout of the
    /// shape of every shard that crosses processes. A local copy may read another replica of this process than the one
    /// that `plan` names, when that one has no buffer, because no other process takes part in it (refer to
    /// [`Self::local_replica`]). A cross-host send may read another replica in that case too, but it sends from the
    /// device that `plan` names, with which its receiver pairs it, so that replica is first copied to that device
    /// (refer to [`Self::cross_host_source_buffer`]).
    ///
    /// A cross-host transfer moves the raw bytes of a buffer, and its receiver allocates the buffer in the default
    /// layout of the plugin for the shape of the shard. A source shard in another layout is therefore converted to
    /// that default layout before it is sent (refer to [`Self::cross_host_send_buffer`]), and the layout that the
    /// result records is the layout of this array only when the default layout of every shard that crosses processes,
    /// on any process, is that layout too. Every process therefore records the same layout. Otherwise, the result
    /// records no layout, and the caller applies the requested layout afterwards (e.g., [`Self::place_storage`]).
    ///
    /// The buffers come with the recorded layout and with the completion fence of the cross-host sends that this
    /// process issues for the destination shards of other processes, or `None` when it issues none. This function
    /// never waits for those sends, because a send completes only once its receiver issues the matching receive, and
    /// the receiver may be this same process further below or a process that is itself waiting for a send of its own.
    /// The caller retains the fence in the placed array so that [`Self::ready`] and [`Self::block_until_ready`] surface
    /// send failures. Every process derives the same plan from global metadata, so each send has exactly one matching
    /// receive, and [`issue_cross_host_transfers`] issues the transfers of all processes in one global order.
    ///
    /// A process that stops before it issued all of its transfers leaves its peers waiting for the rest, so this
    /// function does before its first transfer only what its transfers need, which is preparing every buffer that it
    /// sends (i.e., a copy of a replica that serves a missing one and a conversion to the layout of the receiver, each
    /// once per source shard, however many processes receive that shard) and looking up the devices of its receives.
    /// What only this process knows can therefore stop it before its first transfer only when a shard that it sends
    /// has no buffer for any replica of its slice on this process, or when preparing such a buffer fails. Everything
    /// else that only this process can check (i.e., the replicas that serve its local copies and the destination
    /// memory of its devices) it checks only after it issued every transfer, so that a failure there stops only this
    /// process, while its peers complete their transfers.
    ///
    /// `completion` records the readiness of every buffer that this function copies or receives and the fences of
    /// the conversions that it runs, which the caller joins into the execution fence of the result.
    fn exact_shard_buffers(
        &self,
        plan: &ShardMovePlan,
        cross_host_layouts: &HashMap<StaticShape, ryft_pjrt::Layout>,
        memory: Memory,
        completion: &mut PlacementCompletion,
    ) -> Result<(Vec<Arc<Buffer<'static>>>, Option<Layout>, Option<ExecutionFence>), Error> {
        let session = self.session();
        let client = session.client();
        let process_index = client.process_index()?;
        let addressable_devices = client.addressable_devices()?;
        let mut addressable_device_by_id = HashMap::with_capacity(addressable_devices.len());
        for device in addressable_devices {
            addressable_device_by_id.insert(device.id()?, device);
        }
        let addressable_device = |device_id: DeviceId| {
            addressable_device_by_id
                .get(&device_id)
                .ok_or_else(|| Error::from(ArrayError::NonAddressableDevice { device_id, process_index }))
        };

        // A replica that this process sends to several processes is prepared once and sent to each of them, in the
        // order of the plan.
        let mut transfers = Vec::new();
        let mut send_buffers = HashMap::new();
        for movement in plan.cross_host_sends(process_index) {
            let buffer = match send_buffers.entry(movement.source_shard_index()) {
                std::collections::hash_map::Entry::Occupied(entry) => Arc::clone(entry.get()),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let shard = &self.shards()[movement.source_shard_index()];
                    let device = addressable_device(shard.device().id())?;
                    let buffer = self.cross_host_source_buffer(shard.index(), device, process_index, completion)?;
                    let layout = &cross_host_layouts[&movement.shape()];
                    let buffer = self.cross_host_send_buffer(shard, buffer, layout, completion)?;
                    Arc::clone(entry.insert(buffer))
                }
            };
            transfers.push(CrossHostTransfer::Send {
                source: movement.source_device().id(),
                destination: movement.destination_device().id(),
                transfer_key: movement.transfer_key(),
                buffer,
            });
        }
        for movement in plan.cross_host_receives(process_index) {
            transfers.push(CrossHostTransfer::Receive {
                source: movement.source_device().id(),
                destination: movement.destination_device().id(),
                transfer_key: movement.transfer_key(),
                device: addressable_device(movement.destination_device().id())?,
                element_type: plan.element_type(),
                dimensions: cross_host_shape(movement.destination_shard_index(), &movement.shape())?,
            });
        }
        let (send_fence, received_buffers) = issue_cross_host_transfers(session, transfers, memory)?;
        let mut addressable_buffers = Vec::new();
        for received in received_buffers {
            completion.track_buffer(&received.buffer)?;
            addressable_buffers.push(Arc::new(received.buffer));
        }

        // Arrays never mutate their buffers and donation skips buffers that are shared, so a destination shard on the
        // device of its source shard shares the buffer of that shard instead of copying it. Copying to the destination
        // device itself would always allocate in its default memory, so other copies go to the requested memory.
        let source_memory = self.r#type().memory();
        let mut work = session.reserve_work();
        for movement in plan.local_copies(process_index) {
            let (shard, buffer) =
                self.local_replica(movement.source_shard_index(), movement.destination_device(), process_index)?;
            let same_device = shard.device() == movement.destination_device();
            if same_device && memory == source_memory {
                addressable_buffers.push(Arc::clone(buffer));
                continue;
            }
            let destination_memory = addressable_device(movement.destination_device().id())?.memory(memory)?;
            if same_device && buffer.memory()? == destination_memory {
                addressable_buffers.push(Arc::clone(buffer));
                continue;
            }
            let copied = buffer.copy_to_memory(destination_memory)?;
            work.track_buffer(&copied);
            completion.track_buffer(&copied)?;
            addressable_buffers.push(Arc::new(copied));
        }

        // Local copies keep the layout of their source shards, while received buffers have the default layout. The
        // recorded layout reads the cross-host moves of every process, not only the receives of this one, so that every
        // process records the same layout, and the steps that later prepare the result (e.g., the inputs of a compiled
        // program) take the same path on every process.
        let received_layouts = plan.cross_host_moves().map(|movement| &cross_host_layouts[&movement.shape()]);
        let layout = exact_shards_layout(self.layout(), received_layouts)?;
        Ok((addressable_buffers, layout, send_fence))
    }

    /// Returns the shard of this array that serves a move between two devices of this process, whose process index is
    /// `process_index`, to `destination`, together with its buffer. The move names the shard at `preferred`, which a
    /// plan chose from metadata that every process shares (e.g., the source of a local copy of whole shards or of a
    /// shard region), and that shard serves whenever it has a buffer. Otherwise, another replica of its slice on this
    /// process with a buffer serves instead (e.g., for an array that holds only some of its replicas), the one on
    /// `destination` first and then the one with the lowest device identifier, so that the choice stays deterministic.
    /// Any replica holds the same value only because the shardings of the moves that call this function hold one
    /// value per slice. Only a move without another process may read another replica directly, because the peer of a
    /// cross-host transfer pairs it with the device of the preferred shard, so a cross-host send first copies that
    /// replica to that device (refer to [`Self::cross_host_source_buffer`]).
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::MissingShardBuffer`] for the preferred shard when no replica of its slice on this process
    /// has a buffer.
    fn local_replica(
        &self,
        preferred: ShardIndex,
        destination: Device,
        process_index: usize,
    ) -> Result<(&ArrayShard, &Arc<Buffer<'static>>), Error> {
        let preferred_shard = &self.shards()[preferred];
        if let Some(buffer) = preferred_shard.buffer() {
            return Ok((preferred_shard, buffer));
        }
        self.shards()
            .iter()
            .filter(|shard| shard.slice() == preferred_shard.slice() && shard.device().process_index() == process_index)
            .filter_map(|shard| shard.buffer().map(|buffer| (shard, buffer)))
            .min_by_key(|(shard, _)| (shard.device() != destination, shard.device().id(), shard.index()))
            .ok_or_else(|| {
                Error::from(ArrayError::MissingShardBuffer {
                    shard_index: preferred,
                    device_id: preferred_shard.device().id(),
                })
            })
    }

    /// Returns the buffer that a cross-host transfer sends for the shard at `index`, which is a shard of this array on
    /// `device` of this process, whose process index is `process_index`. The receivers of the transfer pair it with the
    /// device of that shard, which a plan chose from metadata that every process shares, so the buffer must live on
    /// that device, whatever replicas this process holds. The shard sends its own buffer when it has one. Otherwise,
    /// the replica of its slice that [`Self::local_replica`] selects (e.g., of an array imported with a buffer for
    /// only one of its local replicas) is copied to the memory of `device` that the type of this array declares.
    /// `completion` records the readiness of the returned buffer in both cases, so that the result of a placement that
    /// only sends from this process (e.g., to a destination without devices of this process) still reports a failure
    /// of the work that produced the sent buffer, which may have no other fence (e.g., a pending host upload, whose
    /// readiness only its buffer carries) and whose failure a plugin need not report through the send. The callers
    /// prepare every buffer that they send before they
    /// issue their first transfer, so a slice without any local buffer is reported before this process issues anything,
    /// while its receiver still waits for it, which is the accepted limitation that [`crate::arrays::resharding`]
    /// describes.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::MissingShardBuffer`] for the shard at `index` when no replica of its slice on this process
    /// has a buffer.
    fn cross_host_source_buffer(
        &self,
        index: ShardIndex,
        device: &PjrtDevice<'static>,
        process_index: usize,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        let shard = &self.shards()[index];
        let (replica, buffer) = self.local_replica(index, shard.device(), process_index)?;
        if replica.index() == index {
            completion.track_buffer(buffer)?;
            return Ok(Arc::clone(buffer));
        }
        let mut work = self.session().reserve_work();
        let copied = buffer.copy_to_memory(device.memory(self.r#type().memory())?)?;
        work.track_buffer(&copied);
        completion.track_buffer(&copied)?;
        Ok(Arc::new(copied))
    }

    /// Returns `buffer`, which is the buffer that this process sends to another process for `shard`, an addressable
    /// shard of this array (refer to [`Self::cross_host_source_buffer`]), as a buffer that the cross-host send can
    /// move as it is (refer to [`is_cross_host_sendable`]), in the default memory of the device of that shard and in
    /// `layout`, which is the default layout that the receiver allocates for its shape. The buffer is first staged in
    /// device memory with [`Self::device_memory_buffer`] and then converted with [`Self::cross_host_sendable_buffer`],
    /// whose conversion may compile a program on the device of the shard.
    ///
    /// The physical layout and the allocation of a buffer are known only to the process that holds it, so this
    /// function never declines. The receivers of the shard chose the same strategy without knowing them, and they wait
    /// for this send whatever they are.
    fn cross_host_send_buffer(
        &self,
        shard: &ArrayShard,
        buffer: Arc<Buffer<'static>>,
        layout: &ryft_pjrt::Layout,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        let session = self.session();
        let buffer = Self::device_memory_buffer(session, self.r#type().memory(), buffer, completion)?;
        Self::cross_host_sendable_buffer(
            session,
            self.data_type(),
            shard.device(),
            buffer,
            shard.shape(),
            layout,
            completion,
        )
    }

    /// Returns `buffer`, which holds a shard of an array whose type declares `memory`, in the default memory of its
    /// device, copying it there unless `memory` is [`Memory::Device`]. `completion` records the readiness of the copy.
    ///
    /// Cross-host transfers and the programs that slice shard regions read device buffers (e.g., a cross-host transfer
    /// of the CUDA plugin waits for the GPU completion events of its buffer, which a buffer in host memory does not
    /// have, as an unpinned host buffer is a plain CPU allocation). The declared memory is information that every
    /// process shares, so every process stages the same shards, and the copy is local to the device of `buffer`.
    fn device_memory_buffer(
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
        work.track_buffer(&staged);
        completion.track_buffer(&staged)?;
        Ok(Arc::new(staged))
    }

    /// Returns `buffer`, which holds a shard or a region with `shape` of an array of `data_type` on `device`, in the
    /// default memory of that device, as a buffer that a cross-host send can move to a receiver that allocates a static
    /// buffer of `shape` in `layout`. This is `buffer` itself when [`is_cross_host_sendable`] accepts it, and otherwise
    /// a static buffer of `shape` in `layout` with the same elements, which every sender of a cross-host transfer uses.
    ///
    /// The conversion runs on `device` with [`Self::try_relayout_shard_on_device`], whose program has the static type
    /// of `shape` and so also drops the trailer of a buffer whose type has dynamic dimensions, and `completion` then
    /// records its fence. When the plugin cannot lower or compile that conversion, the shard is copied through host
    /// memory into a new buffer in `layout` with [`Self::copy_shard_through_host`] instead, and `completion` records the
    /// readiness of that buffer. That is the only entry of a host round trip for a cross-host send.
    fn cross_host_sendable_buffer(
        session: &Session,
        data_type: DataType,
        device: Device,
        buffer: Arc<Buffer<'static>>,
        shape: StaticShape,
        layout: &ryft_pjrt::Layout,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        if is_cross_host_sendable(&buffer, &shape, layout)? {
            return Ok(buffer);
        }
        let current_layout = Layout::from_pjrt(buffer.specification()?.layout)?;
        let input_type = ArrayType::new(data_type, shape.clone().into()).with_layout(current_layout);
        let target_layout = Layout::from_pjrt(layout.clone())?;
        let converted =
            Self::try_relayout_shard_on_device(session, device, Arc::clone(&buffer), input_type, &target_layout)?;
        if let Some(converted) = converted {
            completion.track_fence(converted.execution_fence());
            return Ok(Arc::clone(converted.shards()[0].buffer().unwrap()));
        }
        let buffer_type = data_type.executable_buffer_type();
        let memory = buffer.memory()?;
        Self::copy_shard_through_host(session, &buffer, shape, buffer_type, memory, Some(layout.clone()), completion)
    }

    /// Places this array on `mesh` according to `sharding` by slicing on the source devices and joining the received
    /// regions on the destination devices, which [`Self::device_placement_strategy`] chose with `plan` (i.e., the moves
    /// of the regions of every destination shard) and `remote_layouts` (i.e., the default layout of the shape of every
    /// region that crosses processes), both of which [`Self::shard_region_plan`] derived. Every process derives the
    /// same plan, and each one performs only the part of each move that [`ShardMove::role`] assigns to it.
    ///
    /// Slices and cross-host transfers read device buffers, so a source shard that the type of this array declares in
    /// another memory is staged in device memory first (refer to [`Self::device_memory_buffer`]), once for all of its
    /// regions. The result is in default device memory and declares no layout, whatever this array declares, because
    /// its buffers are slices, received buffers, and concatenations in the default layout, as well as whole source
    /// shards in their own layout. The caller moves it to the requested memory and layout (e.g., with
    /// [`Self::place_storage`]).
    fn reshard_shard_regions(
        &self,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        plan: &ShardMovePlan,
        remote_layouts: &HashMap<StaticShape, ryft_pjrt::Layout>,
    ) -> Result<Self, Error> {
        let layout = ShardLayout::new(&self.shape(), mesh, sharding)?;
        let destinations = layout.descriptors();
        let session = self.session();
        let client = session.client();
        let process_index = client.process_index()?;

        let devices = client
            .addressable_devices()?
            .into_iter()
            .map(|device| Ok((device.id()?, device)))
            .collect::<Result<HashMap<_, _>, PjrtError>>()?;
        let (data_type, memory) = (self.data_type(), self.r#type().memory());
        let mut completion = PlacementCompletion::default();
        let destination_device = |device: Device| {
            devices
                .get(&device.id())
                .ok_or(ArrayError::NonAddressableDevice { device_id: device.id(), process_index })
        };

        // A region that this process sends to another one leaves from the planned device, with which its receiver pairs
        // it, so a local replica that serves a planned one without a buffer is copied to that device first, once for
        // all the regions of that shard, as for sends of whole shards. This process prepares only what its transfers
        // need before it issues them, so that what only it can check stops it after its peers got their transfers.
        let mut transfers = Vec::new();
        let mut source_buffers = HashMap::new();
        for movement in plan.moves() {
            let device = movement.destination_device();
            match movement.role(process_index) {
                Some(MoveRole::Send) => {
                    let source = &self.shards()[movement.source_shard_index()];
                    let buffer = match source_buffers.entry(source.index()) {
                        std::collections::hash_map::Entry::Occupied(entry) => Arc::clone(entry.get()),
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            let source_device = destination_device(source.device())?;
                            let buffer = self.cross_host_source_buffer(
                                source.index(),
                                source_device,
                                process_index,
                                &mut completion,
                            )?;
                            let buffer = Self::device_memory_buffer(session, memory, buffer, &mut completion)?;
                            Arc::clone(entry.insert(buffer))
                        }
                    };
                    let buffer = Self::prepare_shard_region(
                        session,
                        data_type,
                        source,
                        buffer,
                        movement.slice(),
                        Some(&remote_layouts[&movement.shape()]),
                        &mut completion,
                    )?;
                    transfers.push(CrossHostTransfer::Send {
                        source: movement.source_device().id(),
                        destination: device.id(),
                        transfer_key: movement.transfer_key(),
                        buffer,
                    });
                }
                Some(MoveRole::Receive) => transfers.push(CrossHostTransfer::Receive {
                    source: movement.source_device().id(),
                    destination: device.id(),
                    transfer_key: movement.transfer_key(),
                    device: destination_device(device)?,
                    element_type: plan.element_type(),
                    dimensions: cross_host_shape(movement.destination_shard_index(), &movement.shape())?,
                }),
                Some(MoveRole::Copy) | None => {}
            }
        }
        let (send_fence, received_buffers) = issue_cross_host_transfers(session, transfers, Memory::Device)?;
        let mut destination_buffers = HashMap::new();
        for received in received_buffers {
            completion.track_buffer(&received.buffer)?;
            destination_buffers.insert(received.transfer_key, Arc::new(received.buffer));
        }

        // Empty source shards contribute no regions, so their buffer errors have no data dependency through
        // which to reach the result (e.g., when gathering a split with some empty shards).
        for shard in self.addressable_shards().filter(|shard| shard.slice().iter().any(Range::is_empty)) {
            if let Some(buffer) = shard.buffer() {
                completion.track_buffer(buffer)?;
            }
        }

        // A region between two devices of this process involves no other process, so another local replica of its
        // source slice can serve it when the planned one has no buffer, as for local copies of whole shards.
        let mut work = session.reserve_work();
        let mut staged_buffers = HashMap::new();
        for movement in plan.moves_with_role(process_index, MoveRole::Copy) {
            let device = movement.destination_device();
            let (source, buffer) = self.local_replica(movement.source_shard_index(), device, process_index)?;
            let buffer = match staged_buffers.entry(source.index()) {
                std::collections::hash_map::Entry::Occupied(entry) => Arc::clone(entry.get()),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let buffer = Self::device_memory_buffer(session, memory, Arc::clone(buffer), &mut completion)?;
                    Arc::clone(entry.insert(buffer))
                }
            };
            let slice = movement.slice();
            let buffer = Self::prepare_shard_region(session, data_type, source, buffer, slice, None, &mut completion)?;
            let buffer = if source.device() == device {
                buffer
            } else {
                let copied = buffer.copy_to_device(destination_device(device)?.clone())?;
                work.track_buffer(&copied);
                completion.track_buffer(&copied)?;
                Arc::new(copied)
            };
            destination_buffers.insert(movement.transfer_key(), buffer);
        }

        let mut regions = vec![Vec::new(); destinations.len()];
        for movement in plan.moves() {
            regions[movement.destination_shard_index()].push(movement);
        }
        let mut buffers = Vec::new();
        for (destination, regions) in destinations.iter().zip(regions) {
            let device = destination.device();
            if device.process_index() != process_index {
                continue;
            }
            // An empty shard has no intersections to transfer or concatenate. Its exact shape still belongs
            // to the result, even when another dimension of that shape is nonzero.
            if destination.slice().iter().any(Range::is_empty) {
                let dimensions = destination.slice().iter().map(|range| range.len() as u64).collect::<Vec<_>>();
                let specification = BufferSpecification::new(data_type.executable_buffer_type(), dimensions);
                let buffer = client.uninitialized_buffer(specification, destination_device(device)?.clone())?;
                work.track_buffer(&buffer);
                completion.track_buffer(&buffer)?;
                buffers.push(Arc::new(buffer));
                continue;
            }
            // Every region of a local destination has exactly one local copy or cross-host receive.
            let missing = ArrayError::MissingShardBuffer { shard_index: destination.index(), device_id: device.id() };
            let pieces = regions
                .into_iter()
                .map(|movement| {
                    let buffer = destination_buffers.remove(&movement.transfer_key()).ok_or_else(|| missing.clone())?;
                    Ok((movement.slice(), buffer))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            buffers.push(Self::assemble_shard_regions(session, data_type, destination, pieces, &mut completion)?);
        }
        let r#type = self.r#type().into_owned().with_memory(Memory::Device).with_layout(None).with_sharding(None)?;
        let mut output = Self::from_addressable_buffers_internal(
            session,
            r#type,
            Some(sharding.clone()),
            mesh.clone(),
            buffers,
            false,
        )?;
        // A readback reads one replica of each tile, so the fence of the result joins every copy, receive, slice, and
        // concatenation that produced a buffer of the result with the fence of this array.
        output.execution_fence = completion.join(self.execution_fence().cloned());
        if let Some(send_fence) = send_fence {
            output = output.with_send_fence(send_fence);
        }
        Ok(output)
    }

    /// Plans the moves of the regions that place this array according to `layout`, which are the nonempty
    /// intersections of its shards with the destination shards, for every process. Identical source slices are
    /// replicas, and each region moves from the replica that [`SourceReplicas::preferred`] chooses. The moves are in
    /// destination shard order, and the regions of each destination shard are in the order of their starting
    /// coordinates, which [`Self::assemble_shard_regions`] needs.
    ///
    /// A source slice intersects a destination shard exactly when its range along every dimension intersects the range
    /// of that shard along that dimension. Each destination shard therefore looks up the source slices that it
    /// intersects among the combinations of the distinct source ranges that it intersects along each dimension (e.g., a
    /// row of a grid of tiles combines the ranges of its rows and of its columns, not every tile of the grid). The
    /// tiles of one sharding are disjoint or identical, so the distinct ranges along one dimension are disjoint, and
    /// sorted by their starts they are sorted by their ends as well. Each destination shard finds the ranges that it
    /// intersects with two binary searches, so planning a one-dimensional placement of `n` shards costs `O(n log n)`
    /// rather than the `O(n^2)` of a scan of every range for every destination shard (e.g., 2 ms rather than 102 ms
    /// for 16,384 shards, measured on the planning loop alone).
    fn plan_shard_regions(&self, layout: &ShardLayout) -> Result<ShardMovePlan, Error> {
        let mut shards_by_slices = HashMap::<&[Range<usize>], Vec<&ShardDescriptor>>::new();
        for shard in self.shards() {
            shards_by_slices.entry(shard.slice()).or_default().push(shard.descriptor());
        }
        let replicas = shards_by_slices
            .into_iter()
            .map(|(slice, shards)| (slice, SourceReplicas::new(shards.as_slice())))
            .collect::<HashMap<_, _>>();
        let intersects = |source: &Range<usize>, destination: &Range<usize>| {
            source.start.max(destination.start) < source.end.min(destination.end)
        };
        let mut source_ranges = vec![Vec::new(); self.shape().rank()];
        for slice in replicas.keys() {
            for (ranges, range) in source_ranges.iter_mut().zip(slice.iter()) {
                ranges.push(range.clone());
            }
        }
        for ranges in &mut source_ranges {
            ranges.sort_unstable_by_key(|range| (range.start, range.end));
            ranges.dedup();
        }
        let destinations = layout.descriptors();
        let mut moves = Vec::new();
        for destination in destinations {
            let mut candidates = vec![Vec::new()];
            for (ranges, destination_range) in source_ranges.iter().zip(destination.slice()) {
                // The ranges that intersect `destination_range` end after it starts and start before it ends.
                let first = ranges.partition_point(|range| range.end <= destination_range.start);
                let last = ranges.partition_point(|range| range.start < destination_range.end).max(first);
                let intersecting = ranges[first..last].iter().filter(|range| intersects(range, destination_range));
                let intersecting = intersecting.collect::<Vec<_>>();
                candidates = candidates
                    .into_iter()
                    .flat_map(|prefix: Vec<Range<usize>>| {
                        intersecting.iter().map(move |&range| [prefix.as_slice(), &[range.clone()]].concat())
                    })
                    .collect();
            }
            let mut regions = Vec::new();
            for candidate in candidates {
                let Some((source_slice, source_shards)) = replicas.get_key_value(candidate.as_slice()) else {
                    continue;
                };
                let slice = source_slice
                    .iter()
                    .zip(destination.slice())
                    .map(|(source, destination)| source.start.max(destination.start)..source.end.min(destination.end))
                    .collect::<Vec<_>>();
                regions.push((source_shards.preferred(destination), slice));
            }
            regions.sort_unstable_by(|(_, left), (_, right)| {
                left.iter().map(|range| range.start).cmp(right.iter().map(|range| range.start))
            });
            for (source, slice) in regions {
                let transfer_key = exact_shard_transfer_key(source.index(), destination.index(), destinations.len())?;
                moves.push(ShardMove::new(
                    source.index(),
                    source.device(),
                    destination.index(),
                    destination.device(),
                    slice,
                    transfer_key,
                ));
            }
        }
        Ok(ShardMovePlan::new(self.data_type().executable_buffer_type(), moves))
    }

    /// Returns the moves of the regions of every destination shard (refer to [`Self::plan_shard_regions`]) and the
    /// default layout of the shape of every region that crosses processes when [`Self::reshard_shard_regions`] can
    /// place this array on `mesh` according to `sharding`, or `None` when it cannot. That is the case when either
    /// sharding can assign different values to the same slice, this array declares a layout other than the dense
    /// row-major one (refer to [`dense_row_major_layout`]), or `resolve_layout` cannot obtain a default layout for a
    /// remote region because the plugin lacks the cross-host transfers extension or cannot report the layout. Every
    /// process derives the same moves and checks all of them, not only its own, so all processes decline alike, and
    /// the placement then executes these moves rather than deriving them again.
    ///
    /// The memory of this array does not matter, and neither does a declared dense row-major layout, which is the
    /// layout that the buffers of an array without a declared layout, the slices, and the received regions have
    /// anyway. Each process makes the changes that they need on its own devices. It stages a source shard in another
    /// memory in device memory before it slices or sends it, and it moves its destination shards to the requested
    /// memory and layout once its transfers are issued (refer to [`Self::reshard_shard_regions`]). An array that
    /// declares another layout (e.g., a column-major one) keeps the other strategies, which move whole shards in
    /// that layout and convert each of them only where it crosses processes.
    fn shard_region_plan(
        &self,
        resolve_layout: &CrossHostLayoutResolver<'_>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<Option<(ShardMovePlan, HashMap<StaticShape, ryft_pjrt::Layout>)>, Error> {
        if !holds_one_value_per_slice(self.sharding())
            || !holds_one_value_per_slice(sharding)
            || self.layout().is_some_and(|layout| *layout != dense_row_major_layout(self.shape().rank()))
        {
            return Ok(None);
        }
        let plan = self.plan_shard_regions(&ShardLayout::new(&self.shape(), mesh, sharding)?)?;
        let mut remote_regions = Vec::new();
        for movement in plan.cross_host_moves() {
            cross_host_global_device_id(movement.source_device().id())?;
            cross_host_global_device_id(movement.destination_device().id())?;
            remote_regions.push((movement.destination_shard_index(), movement.shape()));
        }
        let layouts = cross_host_layouts(resolve_layout, plan.element_type(), remote_regions)?;
        Ok(layouts.map(|layouts| (plan, layouts)))
    }

    /// Produces the region with `slice`, in global array coordinates, from `source_buffer`, which holds the shard
    /// `source` on its device, on that device. `source_buffer` must be in the default memory of that device. A region
    /// that a remote transfer sends (i.e., with `remote_layout`, which is the default layout of the plugin for its
    /// shape) is a static buffer of its shape in that layout, because a cross-host send moves the whole allocation of its
    /// buffer, so a region that [`is_cross_host_sendable`] rejects (e.g., a whole source shard of a bounded-dynamic
    /// result) is converted with [`Self::cross_host_sendable_buffer`]. `completion` records the fences of the programs
    /// and the readiness of the buffers that produce it.
    fn prepare_shard_region(
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
            let input = Self::wrap_device_buffer(
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
            buffer = Self::cross_host_sendable_buffer(
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

    /// Concatenates the transferred regions of a shard on its destination device, each given with its slice in global
    /// array coordinates. `completion` records the fences of the concatenations.
    fn assemble_shard_regions(
        session: &Session,
        data_type: DataType,
        destination: &ShardDescriptor,
        mut pieces: Vec<(&[Range<usize>], Arc<Buffer<'static>>)>,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        if pieces.len() == 1 {
            return Ok(pieces.pop().unwrap().1);
        }
        let mut values = pieces
            .into_iter()
            .map(|(slice, buffer)| {
                let shape = StaticShape::new(slice.iter().map(Range::len).collect::<Vec<_>>());
                let r#type = ArrayType::new(data_type, shape.into());
                let value = Self::wrap_device_buffer(session, r#type, destination.device(), buffer)?;
                Ok((slice, value))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        // The intersections form a Cartesian grid, sorted by their starting coordinates. Join the innermost
        // dimension first so that the pieces agree along every other dimension for each concatenation.
        for axis in (0..destination.slice().len()).rev() {
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
        Ok(Arc::clone(values[0].1.shards()[0].buffer().unwrap()))
    }

    /// Wraps one buffer with its physical layout so a compiled operation can consume it directly on its device.
    fn wrap_device_buffer(
        session: &Session,
        r#type: ArrayType,
        device: Device,
        buffer: Arc<Buffer<'static>>,
    ) -> Result<Self, Error> {
        let placement = single_device_placement(device, r#type.rank())?;
        let (mesh, sharding) = (placement.mesh().clone(), placement.sharding().clone());
        let r#type = if r#type.layout().is_some() {
            r#type
        } else {
            r#type.with_layout(Layout::from_pjrt(buffer.specification()?.layout)?)
        };
        Self::from_addressable_buffers_internal(session, r#type, Some(sharding), mesh, vec![buffer], false)
    }

    /// Places this array on `mesh` according to `sharding` with device work only, and on failure also returns this
    /// array when a host fallback may still place it. [`Self::device_placement_strategy`] chooses the first of the
    /// following strategies that applies, and the placement then uses that strategy alone.
    ///
    ///   1. **Exact Shards:** [`Self::place_exact_shards`] shares, copies, or transfers whole shards, and compiles only
    ///      the conversions of sent shards that [`Self::cross_host_send_buffer`] needs. Its errors are final, so they
    ///      come without this array.
    ///   2. **Shard Regions:** [`Self::reshard_shard_regions`] transfers the parts that each destination needs when
    ///      switching meshes or when uneven shards or manual destination axes prevent a same-mesh compiled reshard.
    ///   3. **Compiled Reshard:** The `ryft` analogue of how JAX's `jax.device_put(arr, new_sharding)` lowers a
    ///      reshard of a committed `jax.Array`, which is an `identity(x) = x` program with input and output sharding
    ///      constraints that is lowered to StableHLO and Shardy, compiled, and executed on the devices. When this
    ///      array already is on `mesh`, one such program reshards it. Otherwise, [`Self::reshard_across_meshes`] first
    ///      moves a replicated copy to `mesh`.
    ///
    /// The strategy fixes the transfers and the compiled programs of every process, and this process performs exactly
    /// its own part of them. A process that owns no device of the mesh of a compiled program runs nothing for it and
    /// only derives the metadata of its output (refer to [`ReshardProgram::runs_on`]), so its result has the placement
    /// and the type of the results of the processes that run it, without buffers.
    ///
    /// An array with [locally assigned devices](Self::is_locally_assigned) uses none of these strategies, and
    /// [`Self::place_process_local_value`] places it with local work only instead. When the destination lies on devices
    /// of this process alone, the first of these strategies that applies places it, because that strategy then does
    /// local work only as well.
    ///
    /// Unsupported requests return typed [`ArrayError`] variants through [`Error::Array`] (refer to
    /// [`Self::validate_compiled_reshard`]), and cross-host extension, tracing, compilation, and execution failures
    /// propagate through [`Error`] without losing their typed payloads.
    ///
    /// A donated array must be the only handle of its buffers, so the compiled reshard on the mesh of this array
    /// consumes this array when `donate` is `true`, and this array is then not returned if that execution fails. It is
    /// still returned when the program fails to stage or to compile, which happens before the execution consumes it.
    /// An array with a pending reduction (i.e., unreduced axes in its runtime sharding) is no exception, because the
    /// program receives it with that reduction over explicit and auto mesh axes alike (refer to
    /// [`ReshardProgram::compiled_mesh`]), so execution admission accepts it, and a change of reduction state that the
    /// program cannot make is declined before anything is compiled (refer to [`validate_reduction_transition`]).
    ///
    /// `layout` is the layout of each destination shard that the placement records (refer to
    /// [`Self::placement_layout`]), which [`Self::place_process_local_value`] uploads into directly.
    // The error carries the unconsumed array back to the caller, which falls back to a host transfer with it.
    #[allow(clippy::result_large_err)]
    fn reshard_on_devices_or_return(
        self,
        resolve_layout: &CrossHostLayoutResolver<'_>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        memory: Memory,
        layout: Option<&Layout>,
        donate: bool,
    ) -> Result<Self, (Error, Option<Self>)> {
        let strategy =
            self.session().client().process_index().map_err(Error::from).and_then(|process_index| {
                self.device_placement_strategy(resolve_layout, mesh, sharding, process_index)
            });
        match strategy {
            Ok(strategy) => self.place_with_strategy(strategy, mesh, sharding, memory, layout, donate),
            Err(error) => Err((error, Some(self))),
        }
    }

    /// Performs the part of this process in `strategy`, which [`Self::device_placement_strategy`] chose for placing
    /// this array on `mesh` according to `sharding`, with the parameters and the errors of
    /// [`Self::reshard_on_devices_or_return`].
    ///
    /// Only this process knows which buffers it holds, so a missing buffer can stop only this process, and its peers
    /// then wait for whatever it has not issued yet. This function therefore checks that this array has a buffer on
    /// every device of this process only where this array itself is the input of a compiled program (i.e., of a
    /// compiled reshard on its own mesh or of the gather of a reshard across meshes), which reads every such buffer.
    /// Such a program stops this process before its collectives and before every transfer after it whenever it fails
    /// to start here, which besides a missing buffer covers a failed conversion of a buffer whose physical layout
    /// only this process knows and a failed launch (e.g., a failed allocation).
    /// Any other strategy reads only the shards that its moves name, serves a missing one from another replica of this
    /// process (refer to [`Self::local_replica`]), and fails only for a slice without any local buffer, and it looks up
    /// everything that its transfers do not need only once they are issued.
    // The error carries the unconsumed array back to the caller, which falls back to a host transfer with it.
    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    fn place_with_strategy(
        self,
        strategy: DevicePlacementStrategy,
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
                let place = || {
                    let r#type = self.r#type().into_owned().with_memory(memory).with_sharding(None)?;
                    let mut placed = Self::zero(self.session(), r#type, Some(sharding.clone()), mesh.clone())?;
                    placed.execution_fence = self.execution_fence().cloned();
                    Ok::<_, Error>(placed)
                };
                place().map_err(|error| (error, None))
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
                // own shards to the requested storage, which needs no other process.
                let placed = self
                    .reshard_shard_regions(mesh, sharding, &plan, &cross_host_layouts)
                    .and_then(|output| output.place_storage(memory, layout.cloned()));
                match placed {
                    Ok(output) => Ok(output),
                    Err(error) => Err((error, Some(self))),
                }
            }
            DevicePlacementStrategy::CompiledReshard { program } => {
                let compiled = self
                    .validate_addressable_shard_buffers()
                    .and_then(|()| self.compile_reshard_on_same_mesh(&program, donate));
                match compiled {
                    Ok(reshard) => {
                        // Staging and compilation, which can reject a placement, are behind us, and the program
                        // receives the runtime sharding of this array, pending reduction included, so execution admits
                        // it (refer to `validate_input_reduction`). What can still fail (e.g., the launch) may follow
                        // the surrender of donated buffers to PJRT, so a donated source is not returned.
                        let source = (!donate).then(|| self.clone());
                        self.execute_reshard(&program, reshard.as_ref()).map_err(|error| (error, source))
                    }
                    Err(error) => Err((error, Some(self))),
                }
            }
            DevicePlacementStrategy::CompiledReshardAcrossMeshes { gather, replicas, cross_host_layouts, reshard } => {
                // A change of memory can use device storage for the intermediate compiled reshard. The destination
                // then moves the completed shards to the requested tier, without requiring SPMD execution there.
                let source_memory = self.r#type().memory();
                let intermediate_memory = if memory == source_memory { memory } else { Memory::Device };
                self.reshard_across_meshes(
                    gather.as_ref(),
                    &replicas,
                    &cross_host_layouts,
                    &reshard,
                    intermediate_memory,
                )
                .map_err(|error| (error, Some(self)))
            }
            DevicePlacementStrategy::HostRoundTrip { reason, process_index } => {
                let this_process = match self.client().process_index() {
                    Ok(process_index) => process_index,
                    Err(error) => return Err((error.into(), Some(self))),
                };
                match process_index {
                    // The caller reads this array back and uploads it with the destination sharding.
                    Some(process_index) if process_index == this_process => Err((reason, Some(self))),
                    Some(_) => {
                        // No device of either end belongs to this process, which only derives the metadata that the
                        // owning process returns, as for a compiled reshard that this process does not run.
                        let place = || {
                            let r#type = ArrayType::new(self.data_type(), self.shape().into()).with_memory(memory);
                            let (mesh, sharding) = (mesh.clone(), Some(sharding.clone()));
                            let mut placed = Self::from_addressable_buffers_internal(
                                self.session(),
                                r#type,
                                sharding,
                                mesh,
                                Vec::new(),
                                false,
                            )?;
                            placed.execution_fence = self.execution_fence().cloned();
                            Ok::<_, Error>(placed)
                        };
                        place().map_err(|error| (error, None))
                    }
                    None => Err((reason, None)),
                }
            }
        }
    }

    /// Chooses the device work that places this array on `mesh` according to `sharding` in the process with index
    /// `process_index`, or returns the error that rules out all of it. `resolve_layout` supplies the layouts needed
    /// for cross-host transfers. [`Self::reshard_on_devices_or_return`] executes the chosen strategy, and the
    /// agreement test of this module plans every process of a distributed placement with this same function.
    ///
    /// When the source or the destination spans several processes, every process calls this function for the same
    /// placement and then performs only its own part of the chosen strategy (e.g., its sends, receives, and local
    /// copies, or its share of a compiled reshard). Every process must therefore derive the same transfers and
    /// programs, because a process that performed the part of another strategy would leave its peers waiting for
    /// transfers or collectives that never come. Every process derives them alike, because this function reads only
    /// information that all of them share, which is the following.
    ///
    ///   - **Source:** The type of this array (i.e., its data type, shape, declared layout, and memory), the slices and
    ///     devices of its shards (but not whether this process holds their buffers), and whether it has
    ///     [locally assigned devices](Self::is_locally_assigned).
    ///   - **Destination:** `mesh` and `sharding`.
    ///   - **Cross-host layouts:** The layouts and transfer support reported by the session's PJRT plugin, which
    ///     every process loads alike (refer to [`CrossHostLayoutResolver`]). Each process leads compilation of
    ///     programs over its own shards, regardless of whether the session shares compilation artifacts.
    ///
    /// It never reads a buffer or a physical layout. It reads `process_index` only for a locally assigned array.
    ///
    /// An array with locally assigned devices holds a full value that every process has, each on devices of its own
    /// choice, as JAX assumes for uncommitted arguments of a computation over several processes. Its devices are the
    /// one part of its metadata that the processes do not share, so it is placed with
    /// [`DevicePlacementStrategy::ProcessLocalValue`], in which every process places its own value on its own
    /// destination devices with local work only, and no process plans a transfer or a collective. Every process knows
    /// the fact for its own array and computes the same one (refer to [`Self::is_locally_assigned`]). There is
    /// one refinement. When every destination device belongs to this process, such an array takes the strategy that [`Self::shared_device_placement_strategy`] chooses for it, if one
    /// applies, because that strategy then moves nothing between processes and needs no other process (e.g., a compiled
    /// reshard on the devices of this process rather than a host round trip). Every other process places nothing on
    /// those devices, whatever it chooses for its own value, so the transfers and programs of all processes still
    /// match. Any other array is placed with the strategies of [`Self::shared_device_placement_strategy`].
    ///
    /// What only one process knows does not change the strategy. The chosen strategy handles it instead (e.g., a
    /// buffer whose physical layout differs from the default layout of a cross-host receive is converted on its device
    /// or through host memory, which leaves the transfers unchanged, and a missing buffer of a replica is served by
    /// another replica of this process), and anything else (e.g., a slice without any local buffer, an unsupported
    /// memory, or a failed local conversion) is an error of that process. It reports such an error after it issued its
    /// transfers whenever its transfers do not need what failed (refer to [`Self::place_with_strategy`],
    /// [`Self::exact_shard_buffers`], and [`issue_cross_host_transfers`]).
    fn device_placement_strategy(
        &self,
        resolve_layout: &CrossHostLayoutResolver<'_>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        process_index: usize,
    ) -> Result<DevicePlacementStrategy, Error> {
        if self.data_type().is_zero() {
            return Ok(DevicePlacementStrategy::Bufferless);
        }
        if !self.is_locally_assigned() {
            return self.shared_device_placement_strategy(resolve_layout, mesh, sharding);
        }

        // Placing the value with local work only also handles every placement that no shared strategy supports (e.g.,
        // an uneven split of a value with a column-major layout), so it remains the choice when none applies.
        let local_destination = mesh.devices().iter().all(|device| device.process_index() == process_index);
        if local_destination && let Ok(strategy) = self.shared_device_placement_strategy(resolve_layout, mesh, sharding)
        {
            return Ok(strategy);
        }
        Ok(DevicePlacementStrategy::ProcessLocalValue)
    }

    /// Chooses the device work that places this array, whose devices every process knows, on `mesh` according to
    /// `sharding`, which is the first of the following strategies of [`Self::reshard_on_devices_or_return`] that
    /// applies, or returns the error that rules out all of them. It reads only information that every process shares
    /// (refer to [`Self::device_placement_strategy`]), and each strategy carries the transfers and programs of every
    /// process.
    ///
    ///   1. [`DevicePlacementStrategy::ExactShards`], when every destination shard has the slices of one source shard
    ///      and every shard that crosses processes has a default layout for its receive.
    ///   2. [`DevicePlacementStrategy::ShardRegions`], when the meshes differ or uneven shards or manual destination
    ///      axes rule out a compiled reshard, and [`Self::shard_region_plan`] accepts the placement.
    ///   3. [`DevicePlacementStrategy::HostRoundTrip`], when a program of the compiled reshard below would change the
    ///      value of this array (refer to [`Self::validate_compiled_reshard`]).
    ///   4. [`DevicePlacementStrategy::CompiledReshard`] on the mesh of this array, or
    ///      [`DevicePlacementStrategy::CompiledReshardAcrossMeshes`] to another mesh, whose replicas move as whole
    ///      shards of the replicated value (refer to [`plan_exact_shard_put`]).
    fn shared_device_placement_strategy(
        &self,
        resolve_layout: &CrossHostLayoutResolver<'_>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<DevicePlacementStrategy, Error> {
        // Exact shards need the default layout of every shard that crosses processes, between whichever processes it
        // crosses, so that a process with only local copies declines whenever the processes that transfer decline.
        let shape = self.shape();
        let element_type = self.data_type().executable_buffer_type();
        let sources = self.shards().iter().map(ArrayShard::descriptor);
        if let Some(plan) = plan_exact_shard_put(element_type, sources, self.sharding(), &shape, mesh, sharding)? {
            let cross_host_shards = plan
                .cross_host_moves()
                .map(|movement| (movement.destination_shard_index(), movement.shape()))
                .collect::<Vec<_>>();
            if let Some(cross_host_layouts) = cross_host_layouts(resolve_layout, element_type, cross_host_shards)? {
                return Ok(DevicePlacementStrategy::ExactShards { plan, cross_host_layouts });
            }
        }

        let compiled = self.validate_compiled_reshard(mesh, sharding);
        let try_regions = match &compiled {
            Ok(None) => self.mesh() != mesh,
            Ok(Some(_)) => false,
            Err(Error::Array(error)) => {
                matches!(
                    error.as_ref(),
                    ArrayError::UnevenReshardingShards { .. } | ArrayError::ManualReshardingAxis { .. }
                )
            }
            Err(_) => false,
        };
        if try_regions
            && let Some((plan, cross_host_layouts)) = self.shard_region_plan(resolve_layout, mesh, sharding)?
        {
            return Ok(DevicePlacementStrategy::ShardRegions { plan, cross_host_layouts });
        }
        if let Some(reason) = compiled? {
            // Only one process can read the value back and upload it again, so the host round trip is available only
            // when every device of both ends belongs to one process.
            let first_process = self.mesh().devices()[0].process_index();
            let process_index = self
                .mesh()
                .devices()
                .iter()
                .chain(mesh.devices())
                .all(|device| device.process_index() == first_process)
                .then_some(first_process);
            return Ok(DevicePlacementStrategy::HostRoundTrip { reason, process_index });
        }
        if self.mesh() == mesh {
            let (input, output) = (self.sharding().clone(), sharding.clone());
            let program = ReshardProgram { mesh: mesh.clone(), input, output };
            return Ok(DevicePlacementStrategy::CompiledReshard { program });
        }

        // A replicated copy of this array (i.e., this array itself or its all-gather on its own mesh) moves to every
        // destination device as whole shards. Every process that owns a device of the mesh of this array holds such a
        // replica, so a replica crosses processes exactly when a destination device belongs to a process that owns
        // none, and all replicas have the shape of this array.
        let replicated = Sharding::replicated(self.mesh().logical_mesh().clone(), shape.rank());
        let gather = (!self.sharding().is_replicated()).then(|| ReshardProgram {
            mesh: self.mesh().clone(),
            input: self.sharding().clone(),
            output: replicated.clone(),
        });
        let replicated_on_destination = Sharding::replicated(mesh.logical_mesh().clone(), shape.rank());
        let replica_layout = ShardLayout::new(&shape, self.mesh(), &replicated)?;
        // Both shardings replicate the whole value, so every destination shard has the slices of every source shard.
        let replicas = plan_exact_shard_put(
            element_type,
            replica_layout.descriptors(),
            &replicated,
            &shape,
            mesh,
            &replicated_on_destination,
        )?
        .unwrap();
        let mut cross_host_layouts = HashMap::new();
        if let Some(movement) = replicas.cross_host_moves().next() {
            let shape = movement.shape();
            let layout =
                resolve_layout(element_type, movement.destination_shard_index(), &shape)?.ok_or_else(|| {
                    Error::Pjrt(PjrtError::unimplemented(
                        "the PJRT plugin does not report the default layout that a cross-host receive allocates",
                    ))
                })?;
            cross_host_layouts.insert(shape, layout);
        }
        let reshard = ReshardProgram { mesh: mesh.clone(), input: replicated_on_destination, output: sharding.clone() };
        Ok(DevicePlacementStrategy::CompiledReshardAcrossMeshes { gather, replicas, cross_host_layouts, reshard })
    }

    /// Checks that a compiled program can place this array on `destination_mesh` with `destination_sharding`, and
    /// returns the reason why it would not preserve the value of this array, if any (refer to
    /// [`validate_reduction_transition`]), in which case the placement takes a host round trip instead. The errors are
    /// [`ArrayError::MissingUnreducedSumIdentity`] when the destination introduces a pending sum that the data type of
    /// this array cannot represent, which no strategy can do and which is checked first, so that the placement reports
    /// what a host upload of the same value reports, [`ArrayError::ManualReshardingAxis`] when the destination mesh has
    /// a `Manual` axis, while `Auto` and `Explicit` axes are supported, and [`ArrayError::UnevenReshardingShards`] when
    /// the shards of this array or of the destination differ in size. These checks read only metadata that every
    /// process shares (refer to [`Self::device_placement_strategy`]).
    ///
    /// The SPMD partitioner splits a dimension into parts of one size and pads the last parts, so a dimension whose
    /// extent the number of its parts does not divide (e.g., five elements over two devices) yields buffers whose
    /// shapes differ from the exact shards that [`ShardLayout`] describes (i.e., `[3]` and `[2]`). Such a placement
    /// can use whole shards or shard regions, both of which materialize those exact shapes.
    fn validate_compiled_reshard(
        &self,
        destination_mesh: &DeviceMesh,
        destination_sharding: &Sharding,
    ) -> Result<Option<Error>, Error> {
        // The programs are those of `Self::shared_device_placement_strategy`, which on another mesh gathers this array
        // into a replicated value and reshards a replicated value on the destination mesh.
        let (data_type, shape) = (self.data_type(), self.shape());
        let decline = if self.mesh() == destination_mesh {
            validate_reduction_transition(data_type, &shape, destination_mesh, self.sharding(), destination_sharding)?
        } else {
            let rank = shape.rank();
            let replicated = Sharding::replicated(self.mesh().logical_mesh().clone(), rank);
            let gather = validate_reduction_transition(data_type, &shape, self.mesh(), self.sharding(), &replicated)?;
            let replicated = Sharding::replicated(destination_mesh.logical_mesh().clone(), rank);
            let reshard =
                validate_reduction_transition(data_type, &shape, destination_mesh, &replicated, destination_sharding)?;
            gather.or(reshard)
        };

        // `Manual` axes are managed explicitly by the user (e.g., inside `shard_map`) and cannot be planned by the SPMD
        // partitioner from the top level. `Auto` and `Explicit` axes both go through SPMD partitioning, so both are
        // allowed.
        if let Some(axis) =
            destination_mesh.logical_mesh().axes().iter().find(|axis| axis.r#type() == MeshAxisType::Manual)
        {
            return Err(ArrayError::ManualReshardingAxis { axis_name: axis.name().to_string() }.into());
        }

        // The shards of one sharding have equal shapes exactly when every split divides its dimension evenly.
        let destination_layout = ShardLayout::new(&shape, destination_mesh, destination_sharding)?;
        let has_uneven_shards = |shapes: Vec<StaticShape>| shapes.iter().any(|shape| shape != &shapes[0]);
        if has_uneven_shards(self.shards().iter().map(ArrayShard::shape).collect()) {
            return Err(ArrayError::UnevenReshardingShards { shape, sharding: self.sharding().clone() }.into());
        }
        if has_uneven_shards(destination_layout.descriptors().iter().map(ShardDescriptor::shape).collect()) {
            return Err(ArrayError::UnevenReshardingShards { shape, sharding: destination_sharding.clone() }.into());
        }
        Ok(decline)
    }

    /// Checks that every shard of this array on this process has a buffer for the compiled program that reads it
    /// (i.e., a compiled reshard on the mesh of this array or the gather of a reshard across meshes). Returns
    /// [`ArrayError::MissingShardBuffer`] for the first missing buffer. The program reads remote shards on their own
    /// processes. Only this process knows which of its buffers exist, so it checks them just before executing the
    /// program, after [`Self::device_placement_strategy`] has chosen it. A failure stops this process before the
    /// program, and its peers then wait in its collectives and in the transfers that follow it.
    fn validate_addressable_shard_buffers(&self) -> Result<(), Error> {
        let client_process_index = self.client().process_index()?;
        for shard in self.shards() {
            if shard.device().process_index() == client_process_index && shard.buffer().is_none() {
                return Err(Error::from(ArrayError::MissingShardBuffer {
                    shard_index: shard.index(),
                    device_id: shard.device().id(),
                }));
            }
        }
        Ok(())
    }

    /// Compiles `program`, the identity that reshards this array, which must be on the mesh of `program` with its
    /// input sharding, to its output sharding, or returns `None` when this process does not run `program` (refer to
    /// [`ReshardProgram::runs_on`]). The program is compiled over [`ReshardProgram::compiled_mesh`], which types the
    /// explicit mesh axes along which it changes the reduction state as auto, and [`Self::execute_reshard`] then moves
    /// this array onto that mesh and its result back.
    /// The input and the output of the program are declared in the memory of this array, so the program accepts this
    /// array and returns its result in the same memory. If `donate` is `true`, the input of the program is marked as
    /// donated so that PJRT can reuse the memory of its buffers for the output buffers.
    ///
    /// The compilation cache is keyed by the complete lowered computation, including its input and output sharding
    /// annotations and PJRT options.
    fn compile_reshard_on_same_mesh(
        &self,
        program: &ReshardProgram,
        donate: bool,
    ) -> Result<Option<CompiledFunction<ArrayType, ArrayType>>, Error> {
        let session = self.session();
        if !program.runs_on(session.client().process_index()?) {
            return Ok(None);
        }

        // Build the bare input type, which has no sharding, for tracing. The input and output shardings are attached as
        // `sdy.sharding` attributes through `Options::with_input_shardings` and `Options::with_output_shardings`. The
        // type keeps the memory of this array, which the identity carries over to the output.
        let bare_input_type = ArrayType::new(self.data_type(), self.shape().into()).with_memory(self.r#type().memory());
        let mesh = program.compiled_mesh()?;
        let (input, output) = if mesh == program.mesh {
            (program.input.clone(), program.output.clone())
        } else {
            let logical_mesh = mesh.logical_mesh();
            (program.input.with_mesh(logical_mesh.clone())?, program.output.with_mesh(logical_mesh.clone())?)
        };
        let options = Options::new(mesh)
            .with_input_shardings(vec![input])
            .with_output_shardings(vec![output])
            .with_donate(donate);
        let staged: crate::StagedFunction<ArrayType, ArrayType> =
            crate::stage(|input| input, bare_input_type, options)?;
        Ok(Some(session.compile_staged_function(staged)?))
    }

    /// Executes `reshard`, which [`Self::compile_reshard_on_same_mesh`] compiled from `program` for this array, on this
    /// array. Without `reshard` (i.e., in a process that does not run `program`), this function runs nothing and
    /// returns an array with the type and the placement of the output of `program` and without buffers, which are the
    /// type and the placement that the outputs of the processes that run it have, so every process of a placement
    /// returns the same metadata. The execution fence of the result joins the execution fence of this array with the
    /// fence of the launch, so that a failure of the work that produced this array still surfaces through the result.
    /// The launch only reads the buffers of this array, and those can be readable while that work fails (e.g., when the
    /// fence of this array joins a failed copy of a replica that the launch does not read). This function consumes the
    /// array so that a donation takes effect.
    ///
    /// The executable runs on [`ReshardProgram::compiled_mesh`], so when that mesh types an axis differently, this array
    /// enters it as the same shards on that mesh, and its result leaves it as the same shards on the mesh of `program`
    /// (refer to [`Self::with_mesh`]).
    fn execute_reshard(
        self,
        program: &ReshardProgram,
        reshard: Option<&CompiledFunction<ArrayType, ArrayType>>,
    ) -> Result<Self, Error> {
        let mut completion = PlacementCompletion::default();
        completion.track_fence(self.execution_fence());
        let mut output = match reshard {
            Some(reshard) => {
                let session = self.session().clone();
                let mesh = program.compiled_mesh()?;
                if mesh == program.mesh {
                    reshard.executable().execute(&session, self)?
                } else {
                    reshard.executable().execute(&session, self.with_mesh(&mesh)?)?.with_mesh(&program.mesh)?
                }
            }
            None => {
                let r#type = ArrayType::new(self.data_type(), self.shape().into()).with_memory(self.r#type().memory());
                let (mesh, sharding) = (program.mesh.clone(), Some(program.output.clone()));
                Self::from_addressable_buffers_internal(self.session(), r#type, sharding, mesh, Vec::new(), false)?
            }
        };
        output.execution_fence = completion.join(output.execution_fence.take());
        Ok(output)
    }

    /// Returns this array on `mesh`, which has the devices and the topology of the mesh of this array and may type its
    /// axes differently, with the same shards, buffers, completion fences, commitment, and locally assigned devices.
    /// The axis types decide only which part of the placement a type records, so the sharding of this array places
    /// every shard on `mesh` exactly as it does now (refer to [`Sharding::with_mesh`]), pending reduction included,
    /// and the type of the result records that placement over the axes that `mesh` does not type as auto.
    fn with_mesh(self, mesh: &DeviceMesh) -> Result<Self, Error> {
        debug_assert_eq!(self.mesh().devices(), mesh.devices());
        let sharding = self.sharding().with_mesh(mesh.logical_mesh().clone())?;
        let r#type = self.r#type().into_owned().with_sharding(None)?;
        let buffers = self.shards().iter().filter_map(|shard| shard.buffer().cloned()).collect();
        let array = Self::from_addressable_buffers_internal(
            self.session(),
            r#type,
            Some(sharding),
            mesh.clone(),
            buffers,
            false,
        )?;
        Ok(Self {
            execution_fence: self.execution_fence.clone(),
            send_fence: self.send_fence.clone(),
            committed: self.committed,
            locally_assigned: self.locally_assigned,
            ..array
        })
    }

    /// Places this array on the mesh of `reshard` according to its output sharding in three steps, which
    /// [`Self::shared_device_placement_strategy`] planned for every process.
    ///
    ///   1. `gather`, if any, is an all-gather on the mesh of this array, so that every device of that mesh ends up
    ///      with the full value. A replicated array needs none.
    ///   2. The replicated value then moves to every device of the destination mesh as the whole shards that `replicas`
    ///      describes (refer to [`Self::place_exact_shards`]), with copies between the devices of this process and
    ///      PJRT cross-host transfers between processes, into `memory`. `cross_host_layouts` holds the default layout
    ///      that a cross-host receive allocates for a replica, which the sender converts its replica to first (refer
    ///      to [`Self::cross_host_send_buffer`]).
    ///   3. `reshard` reshards the replicated copy on the destination mesh, consuming it so that PJRT can reuse its
    ///      buffers for the output.
    ///
    /// Both compiled programs go through the compilation cache of `session`, so repeated calls with the same input and
    /// output shardings pay the compilation cost once, and this process runs each of them only when it owns a device of
    /// its mesh (refer to [`ReshardProgram::runs_on`]). The execution fence of the result joins the execution fence of
    /// this array, the readiness of every copied or received replica, and the fences of both programs, because the
    /// final reshard reads only the replicas that this process copied or received, and a failure of another replica of
    /// this array would otherwise not surface through the result. The result retains the completion fence of the
    /// cross-host sends that this process issued, so that [`Self::ready`] and [`Self::block_until_ready`] surface their
    /// errors without this function waiting for them.
    ///
    /// A failed replica of this array that this process copies locally therefore always fails the readiness functions
    /// and the readbacks of the result, which is never ready and never reads back a value. A failed replica that this
    /// process sends to another process (e.g., the source buffer itself when there is no gather, refer to
    /// [`Self::cross_host_source_buffer`]) instead strands its receiver, whose readiness and readbacks block, because a
    /// plugin may never launch a send of a failed buffer. This is the accepted limit of a process that fails before
    /// it posts its part of a transfer (refer to the documentation of this module). Which error a locally copied
    /// failed replica reports is not fixed, though. A failed
    /// replica that a device of the destination mesh copies (e.g., the replica on its own device) fails that copy, and
    /// the final reshard that reads the copy fails as well, and the fence of the result reports whichever of these
    /// failures it observes first (refer to [`ExecutionFence::join`]). Both name the cause of the failed replica, the
    /// first as the copy reports it and the second as the failed execution of the final reshard reports it.
    fn reshard_across_meshes(
        &self,
        gather: Option<&ReshardProgram>,
        replicas: &ShardMovePlan,
        cross_host_layouts: &HashMap<StaticShape, ryft_pjrt::Layout>,
        reshard: &ReshardProgram,
        memory: Memory,
    ) -> Result<Array, Error> {
        // The source array is owned externally, so it is not donated to the all-gather, which reads every shard of this
        // array on this process. The moves of whole replicas read only the replicas that they name, and serve a missing
        // one from another replica of this process, so a replicated array needs a buffer only where it is read.
        let replicated = match gather {
            Some(gather) => {
                self.validate_addressable_shard_buffers()?;
                let compiled = self.compile_reshard_on_same_mesh(gather, false)?;
                self.clone().execute_reshard(gather, compiled.as_ref())?
            }
            None => self.clone(),
        };
        let copies =
            replicated.place_exact_shards(&reshard.mesh, &reshard.input, memory, replicas, cross_host_layouts)?;
        drop(replicated);

        // The copies are owned exclusively by this function and are never observed by callers, so donating them lets
        // PJRT reuse their memory for the output of the final reshard whenever they share no buffer with this array.
        let send_fence = copies.send_fence.clone();
        let compiled = copies.compile_reshard_on_same_mesh(reshard, true)?;
        let output = copies.execute_reshard(reshard, compiled.as_ref())?;
        Ok(match send_fence {
            Some(fence) => output.with_send_fence(fence),
            None => output,
        })
    }
}

/// Device work that places an [`Array`] on a mesh with a sharding, which [`Array::device_placement_strategy`] chooses
/// from information that every process shares, so that the processes of a placement agree on it.
enum DevicePlacementStrategy {
    /// Changes only the placement metadata of an array of [`DataType::Zero`], which has no buffers.
    Bufferless,

    /// Places the full value that an array with locally assigned devices holds on the devices of this process alone
    /// (refer to [`Array::is_locally_assigned`] and [`Array::place_process_local_value`]).
    ProcessLocalValue,

    /// Shares, copies, and transfers the whole shards that `plan` describes (refer to [`Array::place_exact_shards`]).
    ExactShards {
        /// Plan of the moves of whole shards of every process.
        plan: ShardMovePlan,

        /// Default layout of the PJRT plugin for the shape of every shard that crosses processes.
        cross_host_layouts: HashMap<StaticShape, ryft_pjrt::Layout>,
    },

    /// Slices the source shards and joins the regions on the destination devices (refer to
    /// [`Array::reshard_shard_regions`]).
    ShardRegions {
        /// Plan of the moves of the regions of every destination shard of every process.
        plan: ShardMovePlan,

        /// Default layout of the PJRT plugin for the shape of every region that crosses processes.
        cross_host_layouts: HashMap<StaticShape, ryft_pjrt::Layout>,
    },

    /// Reshards an array on its own mesh with a compiled program (refer to [`Array::compile_reshard_on_same_mesh`]).
    CompiledReshard {
        /// Program that reshards the array on its mesh.
        program: ReshardProgram,
    },

    /// Moves a replicated copy of an array to another mesh, gathering a sharded array on its own mesh first, and then
    /// reshards it there with a compiled program (refer to [`Array::reshard_across_meshes`]).
    CompiledReshardAcrossMeshes {
        /// All-gather of a sharded array on its own mesh, or `None` for a replicated array.
        gather: Option<ReshardProgram>,

        /// Plan of the moves of the replicas of the replicated array to every device of the destination mesh.
        replicas: ShardMovePlan,

        /// Default layout of the PJRT plugin for the shape of the array when a replica crosses processes.
        cross_host_layouts: HashMap<StaticShape, ryft_pjrt::Layout>,

        /// Program that reshards the replicated copy on the destination mesh.
        reshard: ReshardProgram,
    },

    /// Reads an array back and uploads it again with the host functions that define what its reduction state means,
    /// because the compiled reshard would change its value (refer to [`validate_reduction_transition`]). The process
    /// that owns every device of the array and of the destination does the round trip (refer to the host fallback of
    /// [`Array::to_placement`]), and every other process returns the placement without buffers. When no process owns
    /// all of them, no process can read the value back and upload it, so every process returns `reason` before it does
    /// anything.
    HostRoundTrip {
        /// Error that names the change of reduction state that the compiled reshard would not make exactly.
        reason: Error,

        /// Index of the process that owns every device of the array and of the destination, if one does.
        process_index: Option<usize>,
    },
}

/// Resolves the default layout that a cross-host receive allocates for a buffer of the given element type and shape.
/// The destination shard index identifies the shard in diagnostics. Returns `None` when the plugin cannot report the
/// layout and [`PjrtError::Unimplemented`] when it lacks the cross-host transfers extension. All processes of a
/// placement use the same plugin, and the agreement test supplies a simulated resolver.
///
/// Each process leads compilation of programs over its own shards (refer to
/// [`artifact_exchange_leader`](ryft_core::CompilationCacheDomain::artifact_exchange_leader)). It discovers whether
/// the plugin can lower and compile a layout conversion when it tries the conversion, and falls back to a host copy
/// when necessary (refer to [`Array::try_relayout_shard_on_device`]). These local decisions leave the shared strategy
/// intact.
type CrossHostLayoutResolver<'o> =
    dyn Fn(BufferType, ShardIndex, &StaticShape) -> Result<Option<ryft_pjrt::Layout>, Error> + 'o;

/// Completion of the device work that one placement starts to produce the buffers of its result (e.g., copies between
/// memories or devices, cross-host receives, host uploads, and compiled conversions), which the execution fence of the
/// result joins with the fence of the source.
///
/// A readback of an array reads only one replica of each tile, and the readiness functions of an array wait for its
/// execution fence. The session tracks these buffers as well, but a buffer whose producing work failed would otherwise
/// go unnoticed by a readback that happens to read another replica of the same tile.
#[derive(Default)]
struct PlacementCompletion {
    /// Ready events of the buffers that the placement created.
    events: Vec<Event<()>>,

    /// Execution fences of the programs that produced buffers of the result.
    fences: Vec<ExecutionFence>,
}

impl PlacementCompletion {
    /// Records the readiness of `buffer`, which the placement created. This only takes its ready event and never waits.
    fn track_buffer(&mut self, buffer: &Buffer<'_>) -> Result<(), Error> {
        self.events.push(buffer.ready()?);
        Ok(())
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

/// Compiled program that a placement runs on the devices of one mesh to reshard an array there, which is the identity
/// with sharding annotations on its input and its output (refer to [`Array::compile_reshard_on_same_mesh`]). The
/// strategy of a placement fixes its programs from information that every process shares, so every process that runs
/// one of them runs the same program.
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
    /// Returns whether the process with index `process_index` runs this program, which is the case exactly when it
    /// owns a device of its mesh. Any other process holds no buffer that the program reads or writes and takes part in
    /// none of its collectives, and its PJRT client cannot compile a program for devices that it does not address, so
    /// it compiles and runs nothing and only derives the metadata of the output (refer to [`Array::execute_reshard`]).
    fn runs_on(&self, process_index: usize) -> bool {
        self.mesh.devices().iter().any(|device| device.process_index() == process_index)
    }

    /// Returns the mesh over which this program is compiled, which is [`Self::mesh`] with the explicit axes along which
    /// the program changes the reduction state (refer to [`ReductionState`]) typed as auto, and the auto axes along
    /// which it keeps a pending reduction typed as explicit. The program only moves data, so the types of the axes of
    /// its shardings change nothing that it computes, but over an explicit axis a type records reduction state, which
    /// no staged operation may change (i.e., a `reshard` keeps it, refer to
    /// [`ReshardOperation`](ryft_core::ReshardOperation)). Over an auto axis the type omits it, so the lowering makes
    /// the change instead (i.e., it completes a pending reduction where the input enters the program, and the output
    /// sharding introduces a pending sum or a completed reduction), with the arithmetic of XLA, which
    /// [`validate_reduction_transition`] compares with the meaning of the reduction before a program is planned. A
    /// pending reduction that the program keeps therefore moves over an explicit axis, as its contributions, without
    /// being completed and introduced again, which preserves the value of every data type and needs no collective.
    /// Every other axis keeps its type. The rule reads only the program, so every process compiles the same program.
    fn compiled_mesh(&self) -> Result<DeviceMesh, Error> {
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
}

/// Reduction state of a value along one mesh axis, which a [`Sharding`] records per axis (with one
/// [`UnreducedKind`] for all of its unreduced axes), and which a [`ReshardProgram`] keeps or changes.
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

/// Part that one process takes in a [`ShardMove`], which follows from the processes of its two devices alone, so that
/// every process derives the parts of all processes alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MoveRole {
    /// The process owns both devices and shares the moved data on one device or copies it between its devices.
    Copy,

    /// The process owns only the source device and sends the moved data with a PJRT cross-host transfer.
    Send,

    /// The process owns only the destination device and receives the moved data with a PJRT cross-host transfer.
    Receive,
}

/// Deterministic plan of the moves of whole shards or of shard regions that place an [`Array`] (e.g., by
/// [`Array::to_placement`]), which holds the moves of every process, so that every process derives the same plan from
/// global metadata (refer to [`plan_exact_shard_put`] and [`Array::plan_shard_regions`]). Each process performs the
/// part of every move that [`ShardMove::role`] assigns to it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ShardMovePlan {
    /// Element type of the buffers that the plan moves, which is the executable buffer type of the array (e.g., a
    /// predicate for one-bit integers), because that is the buffer that a send transfers and a receive allocates.
    element_type: BufferType,

    /// Moves of the plan, in destination shard order, and for each destination shard in the order of the starting
    /// coordinates of the moved slices.
    moves: Vec<ShardMove>,
}

impl ShardMovePlan {
    /// Creates a plan of the moves of whole shards or of shard regions.
    fn new(element_type: BufferType, moves: Vec<ShardMove>) -> Self {
        Self { element_type, moves }
    }

    /// Returns the element type of the buffers that this plan moves.
    fn element_type(&self) -> BufferType {
        self.element_type
    }

    /// Returns all the moves of this plan, in the order that [`Self::moves`] describes.
    fn moves(&self) -> &[ShardMove] {
        &self.moves
    }

    /// Returns the moves in which the process with index `process_index` takes the part `role`.
    fn moves_with_role(&self, process_index: usize, role: MoveRole) -> impl Iterator<Item = &ShardMove> {
        self.moves.iter().filter(move |movement| movement.role(process_index) == Some(role))
    }

    /// Returns the moves whose source and destination are both on the process with index `process_index`, each of
    /// which shares the buffer of its source shard on the same device or copies it from another device.
    fn local_copies(&self, process_index: usize) -> impl Iterator<Item = &ShardMove> {
        self.moves_with_role(process_index, MoveRole::Copy)
    }

    /// Returns the moves that cross processes, whatever processes they cross between.
    fn cross_host_moves(&self) -> impl Iterator<Item = &ShardMove> {
        self.moves.iter().filter(|movement| movement.is_cross_host())
    }

    /// Returns the cross-host sends of the process with index `process_index`.
    fn cross_host_sends(&self, process_index: usize) -> impl Iterator<Item = &ShardMove> {
        self.moves_with_role(process_index, MoveRole::Send)
    }

    /// Returns the cross-host receives of the process with index `process_index`.
    fn cross_host_receives(&self, process_index: usize) -> impl Iterator<Item = &ShardMove> {
        self.moves_with_role(process_index, MoveRole::Receive)
    }
}

/// One move in a [`ShardMovePlan`] of the elements of a slice, which is a whole source shard or the intersection of a
/// source shard with a destination shard, from the device of that source shard to the device of the destination shard.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ShardMove {
    /// Index of the source shard in the source array.
    source_shard_index: ShardIndex,

    /// Device of the source shard.
    source_device: Device,

    /// Index of the destination shard in the destination array.
    destination_shard_index: ShardIndex,

    /// Device of the destination shard.
    destination_device: Device,

    /// Moved slice in global array coordinates, which is the slice of the destination shard for a move of a whole
    /// shard.
    slice: Vec<Range<usize>>,

    /// Deterministic transfer rendezvous key that the send and the receive of a cross-host move share.
    transfer_key: CrossHostTransferKey,
}

impl ShardMove {
    /// Creates one move of the elements of `slice` from a source shard to a destination shard.
    fn new(
        source_shard_index: ShardIndex,
        source_device: Device,
        destination_shard_index: ShardIndex,
        destination_device: Device,
        slice: Vec<Range<usize>>,
        transfer_key: CrossHostTransferKey,
    ) -> Self {
        Self { source_shard_index, source_device, destination_shard_index, destination_device, slice, transfer_key }
    }

    /// Returns the index of the source shard in the source array.
    fn source_shard_index(&self) -> ShardIndex {
        self.source_shard_index
    }

    /// Returns the device of the source shard.
    fn source_device(&self) -> Device {
        self.source_device
    }

    /// Returns the index of the destination shard in the destination array.
    fn destination_shard_index(&self) -> ShardIndex {
        self.destination_shard_index
    }

    /// Returns the device of the destination shard.
    fn destination_device(&self) -> Device {
        self.destination_device
    }

    /// Returns the moved slice in global array coordinates.
    fn slice(&self) -> &[Range<usize>] {
        &self.slice
    }

    /// Returns the static shape of the moved slice.
    fn shape(&self) -> StaticShape {
        StaticShape::new(self.slice.iter().map(Range::len).collect::<Vec<_>>())
    }

    /// Returns the transfer rendezvous key that the send and the receive of a cross-host move share.
    fn transfer_key(&self) -> CrossHostTransferKey {
        self.transfer_key
    }

    /// Returns whether this move crosses processes, which needs a PJRT cross-host transfer.
    fn is_cross_host(&self) -> bool {
        self.source_device.process_index() != self.destination_device.process_index()
    }

    /// Returns the part that the process with index `process_index` takes in this move, or `None` when it owns neither
    /// of its devices.
    fn role(&self, process_index: usize) -> Option<MoveRole> {
        match (
            self.source_device.process_index() == process_index,
            self.destination_device.process_index() == process_index,
        ) {
            (true, true) => Some(MoveRole::Copy),
            (true, false) => Some(MoveRole::Send),
            (false, true) => Some(MoveRole::Receive),
            (false, false) => None,
        }
    }
}

/// One PJRT cross-host transfer that the current process takes part in, which [`issue_cross_host_transfers`] issues.
enum CrossHostTransfer<'b> {
    /// Send of `buffer` from its device `source` of the current process to the device `destination` of another
    /// process.
    Send {
        /// ID of the source device.
        source: DeviceId,

        /// ID of the destination device.
        destination: DeviceId,

        /// Transfer rendezvous key shared with the matching receive.
        transfer_key: CrossHostTransferKey,

        /// Buffer to send.
        buffer: Arc<Buffer<'static>>,
    },

    /// Receive of a buffer of `element_type` and `dimensions` from the device `source` of another process onto
    /// `device`, which is the device `destination` of the current process.
    Receive {
        /// ID of the source device.
        source: DeviceId,

        /// ID of the destination device.
        destination: DeviceId,

        /// Transfer rendezvous key shared with the matching send.
        transfer_key: CrossHostTransferKey,

        /// Destination device.
        device: &'b PjrtDevice<'static>,

        /// Element type of the received buffer.
        element_type: BufferType,

        /// Dimensions of the received buffer.
        dimensions: Vec<i64>,
    },
}

impl CrossHostTransfer<'_> {
    /// Returns the IDs of the source and destination devices of this transfer.
    fn devices(&self) -> (DeviceId, DeviceId) {
        match self {
            Self::Send { source, destination, .. } | Self::Receive { source, destination, .. } => {
                (*source, *destination)
            }
        }
    }
}

/// A buffer returned by one cross-host receive and the placement that requested it.
struct ReceivedShardBuffer {
    /// Key of the matching transfer.
    transfer_key: CrossHostTransferKey,

    /// Pending buffer returned by PJRT.
    buffer: Buffer<'static>,
}

/// Returns the cross-host shape, attaching the destination shard and dimension to any encoding failure.
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

/// Returns a resolver that checks the cross-host transfers extension and queries the default layout only when called.
fn cross_host_layout_resolver<'o>(
    client: &'o Client<'_>,
) -> impl Fn(BufferType, ShardIndex, &StaticShape) -> Result<Option<ryft_pjrt::Layout>, Error> + 'o {
    move |element_type, shard_index, shape| {
        client.cross_host_transfers_extension()?;
        cross_host_default_layout(client, element_type, shard_index, shape)
    }
}

/// Returns the default layout that the PJRT plugin of `client` uses for a buffer of `element_type` with `shape`, which
/// is the layout of the buffer that a cross-host receive of that shape allocates, or `None` when the plugin cannot
/// report it. `shard_index` identifies the shard in the error for a shape that PJRT cannot represent.
fn cross_host_default_layout(
    client: &Client<'_>,
    element_type: BufferType,
    shard_index: ShardIndex,
    shape: &StaticShape,
) -> Result<Option<ryft_pjrt::Layout>, Error> {
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

/// Uses `resolve_layout` to obtain the default layout for buffers of `element_type` with each shape of
/// `shards` (i.e., of pairs of the index of a destination shard and a shape that crosses processes for it), or `None`
/// when `shards` is not empty and the plugin lacks the cross-host transfers extension or cannot report one of those
/// layouts. Every process derives the same `shards` for a placement and queries the same plugin, so all of them get the
/// same result.
fn cross_host_layouts(
    resolve_layout: &CrossHostLayoutResolver<'_>,
    element_type: BufferType,
    shards: Vec<(ShardIndex, StaticShape)>,
) -> Result<Option<HashMap<StaticShape, ryft_pjrt::Layout>>, Error> {
    let mut layouts = HashMap::new();
    for (shard_index, shape) in shards {
        if let std::collections::hash_map::Entry::Vacant(entry) = layouts.entry(shape) {
            match resolve_layout(element_type, shard_index, entry.key()) {
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

/// Plans the moves of whole shards that place an array of `shape` whose shards are `source_shards`, placed according
/// to `source_sharding`, on `mesh` according to `sharding`, for every process, with buffers of `element_type`. The plan
/// depends only on global metadata (i.e., on the slices and devices of the source shards, on `mesh`, and on
/// `sharding`), so every process derives the same plan, and each process then performs the part of each move that
/// [`ShardMove::role`] assigns to it.
///
/// Returns `None` when a destination shard does not have the slices of one source shard, which needs slicing or
/// concatenating shards, and when either sharding has unreduced or varying manual mesh axes. The devices along such
/// axes hold different values for the same slices (e.g., the summands of an unreduced value), so a destination shard is
/// not a copy of the source shard with its slices.
fn plan_exact_shard_put<'o>(
    element_type: BufferType,
    source_shards: impl IntoIterator<Item = &'o ShardDescriptor>,
    source_sharding: &Sharding,
    shape: &StaticShape,
    mesh: &DeviceMesh,
    sharding: &Sharding,
) -> Result<Option<ShardMovePlan>, Error> {
    if !holds_one_value_per_slice(source_sharding) || !holds_one_value_per_slice(sharding) {
        return Ok(None);
    }

    let mut source_shards_by_slices = HashMap::<&[Range<usize>], Vec<&ShardDescriptor>>::new();
    for shard in source_shards {
        source_shards_by_slices.entry(shard.slice()).or_default().push(shard);
    }
    let replicas_by_slices = source_shards_by_slices
        .into_iter()
        .map(|(slice, shards)| (slice, SourceReplicas::new(shards.as_slice())))
        .collect::<HashMap<_, _>>();

    let destination_layout = ShardLayout::new(shape, mesh, sharding)?;
    let destination_shards = destination_layout.descriptors();
    let destination_shard_count = destination_shards.len();
    let mut moves = Vec::with_capacity(destination_shard_count);
    for destination_shard in destination_shards {
        let Some(replicas) = replicas_by_slices.get(destination_shard.slice()) else {
            return Ok(None);
        };
        let source_shard = replicas.preferred(destination_shard);
        moves.push(ShardMove::new(
            source_shard.index(),
            source_shard.device(),
            destination_shard.index(),
            destination_shard.device(),
            destination_shard.slice().to_vec(),
            exact_shard_transfer_key(source_shard.index(), destination_shard.index(), destination_shard_count)?,
        ));
    }
    Ok(Some(ShardMovePlan::new(element_type, moves)))
}

/// Returns the layout that an array placed as whole shards records, given the declared `layout` of its source and the
/// layouts of the buffers that the cross-host moves of every process receive for it. Local copies keep the layout of
/// their source shards, so the declared layout holds for them, and it is kept only when every received buffer has it
/// too. Otherwise, the buffers of the array can have different layouts, and the array records no layout rather than a
/// wrong one. Reading the moves of every process rather than only the receives of one makes every process record the
/// same layout, even one whose buffers all keep the declared layout.
fn exact_shards_layout<'o>(
    layout: Option<&Layout>,
    mut received_layouts: impl Iterator<Item = &'o ryft_pjrt::Layout>,
) -> Result<Option<Layout>, Error> {
    let Some(layout) = layout else {
        return Ok(None);
    };
    let expected = layout.to_pjrt()?;
    Ok(received_layouts.all(|received| *received == expected).then(|| layout.clone()))
}

/// Returns the deterministic cross-host transfer key for moving data from `source_shard_index` to the
/// destination shard `destination_shard_index`, which is `source_shard_index * destination_shard_count +
/// destination_shard_index`, so that every pair of source and destination shards gets a unique key.
///
/// # Errors
///
/// Returns [`Error::SizeLimitExceeded`] when the key overflows [`usize`] or [`i64`].
fn exact_shard_transfer_key(
    source_shard_index: ShardIndex,
    destination_shard_index: ShardIndex,
    destination_shard_count: usize,
) -> Result<CrossHostTransferKey, Error> {
    let transfer_key = source_shard_index
        .checked_mul(destination_shard_count)
        .and_then(|key| key.checked_add(destination_shard_index))
        .ok_or_else(|| Error::SizeLimitExceeded {
            message: format!(
                "cross-host transfer key for source shard #{source_shard_index} and destination shard \
                 #{destination_shard_index} exceeds the maximum allowed size of {}",
                usize::MAX,
            ),
        })?;
    i64::try_from(transfer_key).map_err(|_| Error::SizeLimitExceeded {
        message: format!(
            "cross-host transfer key {transfer_key} for source shard #{source_shard_index} and destination shard \
             #{destination_shard_index} exceeds the maximum allowed size of {}",
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
/// PJRT delivers a received buffer to the default memory of its destination device, which every device has. When
/// `memory` is another memory, each received buffer is therefore copied on its device to `memory` once every transfer
/// is issued, and the copy is returned in its place. That copy is local to the device and does not wait for the
/// transfer, so it does not affect the order described below.
///
/// The transfers are issued one at a time in the order of their source and then destination device IDs, keeping the
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
    transfers.sort_by_key(CrossHostTransfer::devices);
    for transfer in &transfers {
        let (source, destination) = transfer.devices();
        cross_host_global_device_id(source)?;
        cross_host_global_device_id(destination)?;
    }
    let client = session.client();
    let mut sends = session.reserve_work();
    let mut receives = session.reserve_work();
    let mut received_buffers = Vec::new();
    for transfer in transfers {
        match transfer {
            CrossHostTransfer::Send { destination, transfer_key, buffer, .. } => {
                let destination = cross_host_global_device_id(destination)?;
                sends.track_events(client.cross_host_send_buffers(&[&buffer], &[destination], &[transfer_key], &[])?);
            }
            CrossHostTransfer::Receive { source, transfer_key, device, element_type, dimensions, .. } => {
                let source = cross_host_global_device_id(source)?;
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

/// Returns whether a cross-host transfer can send `buffer` as it is to a receiver that allocates a static buffer of
/// `shape` (i.e., the shape of the shard or the region that `buffer` holds) in `layout`. A send moves the whole
/// allocation of its buffer and some plugins (e.g., CUDA) take its size from the allocation of the sender, so the
/// sender must send exactly the allocation that the receiver makes, which is the case exactly when `buffer` is a static
/// buffer of `shape` in `layout` on the same plugin. Every sender of a cross-host transfer decides with this predicate
/// whether it converts its buffer first (refer to [`Array::cross_host_send_buffer`] and
/// [`Array::prepare_shard_region`]).
///
/// Dimensions alone do not establish that. A buffer whose type has dynamic dimensions (e.g., the result of a
/// bounded-dynamic execution) is allocated for the bound of each such dimension with a trailer that records its
/// runtime extents, so it is larger than a static buffer of `shape` even at its maximum extent, where its physical
/// dimensions equal `shape` (e.g., `4,096` bytes rather than `4,092` for `f32[1023]` on the CPU plugin). Such a buffer
/// is therefore always converted. Only the process that holds `buffer` knows these facts, so they change which buffer
/// this process sends but never the strategy or the transfers that the processes agree on.
fn is_cross_host_sendable(buffer: &Buffer<'_>, shape: &StaticShape, layout: &ryft_pjrt::Layout) -> Result<bool, Error> {
    Ok(buffer.dynamic_dimensions()?.is_empty()
        && buffer.dimensions()?.iter().map(|&size| size as usize).eq(shape.as_slice().iter().copied())
        && buffer.specification()?.layout == *layout)
}

/// Returns the dense row-major [`Layout`] of rank `rank` (i.e., the tiled layout without tiles whose minor-to-major
/// order is the reverse of the order of the dimensions), which is the default layout of dense buffers of the CPU and
/// CUDA plugins and the layout of the slices and of the received regions of [`Array::reshard_shard_regions`].
fn dense_row_major_layout(rank: usize) -> Layout {
    Layout::Tiled(TiledLayout::new((0..rank).rev().collect(), Vec::new()))
}

/// Returns whether replicas of each shard slice hold one value across the mesh.
fn holds_one_value_per_slice(sharding: &Sharding) -> bool {
    sharding.unreduced_axes().is_empty() && sharding.varying_manual_axes().is_empty()
}

/// Validates that a compiled reshard on `mesh` (refer to [`ReshardProgram`]) takes a value of `data_type` and `shape`
/// placed as `input` to `output` with the value that the meaning of its reduction state gives, as far as the reduction
/// state (refer to [`ReductionState`]) along each non-manual mesh axis goes. That meaning is owned by
/// [`crate::arrays::host`], whose section on pending reductions also states the contract of floating point sums. The
/// program keeps the state of an axis as its contributions, which preserves every value, and it makes every other
/// change with a collective of XLA (refer to [`ReshardProgram::compiled_mesh`]). The device path is taken exactly when
/// that collective computes what the operation of the data type computes in a compiled program, up to the sign of a
/// zero result, which is the case as follows.
///
///   - **Completing a pending reduction:** Exact only where [`collective_reduction`] classifies the completion as
///     [`CollectiveReduction::Exact`] (a maximum or minimum that drops NaN values is not).
///   - **Introducing a pending sum:** XLA writes zero to the devices that hold the additive identity (refer to
///     [`holds_unreduced_sum_identity`](crate::arrays::host::holds_unreduced_sum_identity)), which is the identity up
///     to the sign of zero, so it is exact unless [`collective_reduction`] classifies the introduction as
///     [`CollectiveReduction::Unsupported`].
///   - **Introducing a pending maximum or minimum:** XLA cannot (refer to `validate_declared_pending_reduction` in
///     `crate::lowering`, whose check would reject the program).
///   - **Completing or introducing a completed reduction:** This changes no value.
///
/// The rule reads only the data type, the shape, the mesh, and the two shardings, so every process decides alike.
///
/// Returns the reason why the program does not give that value, if any, which makes the placement take a host round
/// trip with the readback and upload of [`crate::arrays::host`].
///
/// # Errors
///
/// Returns [`ArrayError::MissingUnreducedSumIdentity`] when `output` introduces a pending sum whose additive identity
/// some device would have to hold (refer to [`requires_unreduced_sum_identity`]) and the data type has none, which
/// a host upload of the same value rejects as well, so that no strategy can produce it.
fn validate_reduction_transition(
    data_type: DataType,
    shape: &StaticShape,
    mesh: &DeviceMesh,
    input: &Sharding,
    output: &Sharding,
) -> Result<Option<Error>, Error> {
    let decline = |axis: &str, from: ReductionState, to: ReductionState, reason: &str| {
        Error::from(ProgramError::UnsupportedOperation {
            message: format!(
                "a compiled reshard cannot take `{data_type}` values from {from} to {to} along mesh axis `{axis}`, \
                 because {reason}",
            ),
        })
    };
    let mut first_decline = None;
    let mut introduced_sums = Vec::new();
    for (axis_index, axis) in mesh.logical_mesh().axes().iter().enumerate() {
        let (from, to) = (ReductionState::of(input, axis.name()), ReductionState::of(output, axis.name()));
        if axis.r#type() == MeshAxisType::Manual || from == to {
            continue;
        }
        if to == ReductionState::Unreduced(UnreducedKind::Sum) {
            introduced_sums.push((axis_index, axis.name(), from));
        }
        let completion = match from {
            ReductionState::Unreduced(kind) => {
                collective_reduction(data_type, ReductionChange::Complete(kind)).reason()
            }
            _ => None,
        };
        let reason = match (completion, to) {
            (Some(reason), _) => Some(reason),
            (None, ReductionState::Unreduced(kind @ (UnreducedKind::Max | UnreducedKind::Min))) => {
                collective_reduction(data_type, ReductionChange::Introduce(kind)).reason()
            }
            _ => None,
        };
        if first_decline.is_none()
            && let Some(reason) = reason
        {
            first_decline = Some(decline(axis.name(), from, to, reason));
        }
    }

    // A split into summands writes the additive identity only to non-empty shards away from coordinate zero, so a data
    // type without one is rejected exactly when a host upload of the same value with `output` rejects it.
    let axis_indices = introduced_sums.iter().map(|(axis_index, _, _)| *axis_index).collect::<Vec<_>>();
    if !axis_indices.is_empty()
        && requires_unreduced_sum_identity(mesh, &ShardLayout::new(shape, mesh, output)?, &axis_indices)
    {
        unreduced_sum_identity(data_type)?
            .ok_or_else(|| Error::from(ArrayError::MissingUnreducedSumIdentity { data_type }))?;
        let introduction = collective_reduction(data_type, ReductionChange::Introduce(UnreducedKind::Sum));
        if first_decline.is_none()
            && let CollectiveReduction::Unsupported(reason) | CollectiveReduction::DifferentNumber(reason) =
                introduction
        {
            let (_, axis, from) = introduced_sums[0];
            first_decline = Some(decline(axis, from, ReductionState::Unreduced(UnreducedKind::Sum), reason));
        }
    }
    Ok(first_decline)
}

/// Replicas of one source slice (i.e., the source shards that cover it), indexed so that a plan finds the preferred
/// replica for each destination shard without scanning all of them again (e.g., when a replicated array moves to
/// another mesh, where every destination shard has the slice of every source shard).
///
/// The preference is deterministic and depends only on global shard metadata, so that every process chooses the same
/// source shard for a given destination shard. It prefers a source shard on the device of the destination shard, then
/// a source shard on the process of the destination shard, and finally breaks ties by device ID and shard index.
struct SourceReplicas<'o> {
    /// Replica with the lowest shard index on each device.
    by_device: HashMap<Device, &'o ShardDescriptor>,

    /// Replica with the lowest device ID, and then the lowest shard index, of each process.
    by_process: HashMap<usize, &'o ShardDescriptor>,

    /// Replica with the lowest device ID, and then the lowest shard index, of all processes.
    lowest: &'o ShardDescriptor,
}

impl<'o> SourceReplicas<'o> {
    /// Indexes `replicas`, which must not be empty.
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

    /// Returns the preferred replica for `destination_shard`.
    fn preferred(&self, destination_shard: &ShardDescriptor) -> &'o ShardDescriptor {
        let device = destination_shard.device();
        self.by_device
            .get(&device)
            .or_else(|| self.by_process.get(&device.process_index()))
            .copied()
            .unwrap_or(self.lowest)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, mpsc};
    use std::thread;
    use std::time::Duration;

    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array as CoreArray, CompilationArtifactExchange, CompilationArtifactExchangeError,
        CompilationArtifactExchangePolicy, CompilationCache, ConvertElementType, DataType, Dimension, DimensionBounds,
        DimensionError, DimensionVariable, Error as CoreError, Layout, LogicalMesh, Memory, MeshAxis, Shape,
        ShardingDimension, StridedLayout, TiledLayout, UnreducedKind,
    };
    use ryft_pjrt::{
        BufferSpecification, BufferType, ClientOptions, CpuClientOptions, ExecutionFence, HostToDeviceTransferManager,
        load_cpu_plugin,
    };

    use crate::sessions::tests::{domain_mesh, two_device_mesh};
    use crate::tests::{
        cpu_client, four_device_mesh_x, logical_mesh_2x2, other_cpu_client, values_from_bytes, values_to_bytes,
    };
    use crate::{ExecutionError, FromPjrt};

    use super::*;

    // Returns the `f32` values that the devices of `array` hold, in the device order of its mesh.
    fn shard_values(array: &Array) -> Vec<Vec<f32>> {
        array
            .shards()
            .iter()
            .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
            .collect()
    }

    // Returns the PJRT memory kinds of the buffers that the addressable shards of `array` hold.
    fn shard_memory_kinds(array: &Array) -> Vec<String> {
        array
            .addressable_shards()
            .map(|shard| shard.buffer().unwrap().memory().unwrap().kind().unwrap().into_owned())
            .collect()
    }

    // Compilation artifact exchange of process 0 with one fictitious peer process. Process 0 leads every compilation of
    // a program over its own devices, so it publishes such programs and never waits for its peer.
    struct PeerCompilationExchange;

    impl CompilationArtifactExchange for PeerCompilationExchange {
        fn process_index(&self) -> usize {
            0
        }

        fn process_count(&self) -> usize {
            2
        }

        fn publish(&self, _: &[u8], _: &[u8]) -> Result<(), CompilationArtifactExchangeError> {
            Ok(())
        }

        fn receive(&self, _: &[u8], _: Duration) -> Result<Option<Vec<u8>>, CompilationArtifactExchangeError> {
            panic!("process 0 leads the compilations of programs over its own devices and never waits for a peer");
        }
    }

    // Returns a session of `client` whose compilation cache must share every artifact with a peer process.
    fn peer_exchange_session(client: &'static ryft_pjrt::Client<'static>) -> Session {
        let cache = CompilationCache::new().with_artifact_exchange(
            Arc::new(PeerCompilationExchange),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(1) },
        );
        Session::builder().client(client).compilation_cache(cache).build().unwrap()
    }

    // Returns a mesh with one auto axis `x` over the first two addressable devices of `client`.
    fn two_device_sub_mesh_x(client: &ryft_pjrt::Client<'_>) -> DeviceMesh {
        let client_devices = client.addressable_devices().unwrap();
        let devices =
            client_devices.iter().take(2).map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
            .unwrap()
    }

    // Returns an `f32[4]` array that is replicated over `mesh`, which must hold every addressable device of the client
    // of `session`, together with the transfer managers whose buffers are its replicas. Their contents stay undefined
    // until the test defines them or makes them fail through those managers, which it can do after a placement has
    // started the work that reads them.
    fn pending_replicated_vector(
        session: &Session,
        mesh: &DeviceMesh,
    ) -> (Array, Vec<HostToDeviceTransferManager<'static>>) {
        let client = session.client();
        let managers = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| {
                let specification = BufferSpecification::new(BufferType::F32, [4u64]);
                client
                    .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let buffers = managers.iter().map(|manager| manager.retrieve_buffer(0).unwrap()).collect();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let array = Array::from_addressable_buffers(session, r#type, mesh.clone(), Some(replicated), buffers).unwrap();
        (array, managers)
    }

    // Cross-host transfer that one process plans for a placement, which its send and its receive must agree on.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct PlannedTransfer {
        source: Device,
        destination: Device,
        key: CrossHostTransferKey,
        shape: Vec<usize>,
    }

    // Work that crosses processes, which one process plans for its part of a placement. Its programs are the compiled
    // reshards that it runs, which every process that owns a device of their meshes must run alike.
    struct PlannedWork {
        strategy: &'static str,
        sends: Vec<PlannedTransfer>,
        receives: Vec<PlannedTransfer>,
        programs: Vec<ReshardProgram>,
    }

    // Simulates a PJRT plugin that supports cross-host transfers and reports row-major layouts, as CUDA does.
    fn simulated_cross_host_layout(
        _: BufferType,
        _: ShardIndex,
        shape: &StaticShape,
    ) -> Result<Option<ryft_pjrt::Layout>, Error> {
        let minor_to_major = (0..shape.rank()).rev().collect();
        Ok(Some(Layout::Tiled(TiledLayout::new(minor_to_major, vec![])).to_pjrt()?))
    }

    // Returns the work that crosses processes, which the process with index `process_index` plans when it places
    // `source`, its own view of the source of a placement, on `mesh` according to `sharding`. The strategy, with the
    // moves and programs of every process, comes from `Array::device_placement_strategy`, which the placement executes,
    // and the part of this process in them from `ShardMove::role` and `ReshardProgram::runs_on`, which the placement
    // follows as well, so this is the work that the placement performs.
    fn planned_work(
        source: &Array,
        resolve_layout: &CrossHostLayoutResolver<'_>,
        process_index: usize,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<PlannedWork, String> {
        let strategy = source
            .device_placement_strategy(resolve_layout, mesh, sharding, process_index)
            .map_err(|error| error.to_string())?;
        let (name, plan, programs) = match &strategy {
            DevicePlacementStrategy::Bufferless => ("bufferless", None, Vec::new()),
            DevicePlacementStrategy::ProcessLocalValue => ("process-local value", None, Vec::new()),
            DevicePlacementStrategy::ExactShards { plan, .. } => ("exact shards", Some(plan), Vec::new()),
            DevicePlacementStrategy::ShardRegions { plan, .. } => ("shard regions", Some(plan), Vec::new()),
            DevicePlacementStrategy::CompiledReshard { program } => ("compiled reshard", None, vec![program]),
            DevicePlacementStrategy::CompiledReshardAcrossMeshes { gather, replicas, reshard, .. } => {
                ("compiled reshard across meshes", Some(replicas), gather.iter().chain([reshard]).collect())
            }
            DevicePlacementStrategy::HostRoundTrip { .. } => ("host round trip", None, Vec::new()),
        };
        let transfers = |moves: Vec<&ShardMove>| {
            moves
                .into_iter()
                .map(|movement| PlannedTransfer {
                    source: movement.source_device(),
                    destination: movement.destination_device(),
                    key: movement.transfer_key(),
                    shape: movement.shape().as_slice().to_vec(),
                })
                .collect()
        };
        Ok(PlannedWork {
            strategy: name,
            sends: transfers(plan.into_iter().flat_map(|plan| plan.cross_host_sends(process_index)).collect()),
            receives: transfers(plan.into_iter().flat_map(|plan| plan.cross_host_receives(process_index)).collect()),
            programs: programs.into_iter().filter(|program| program.runs_on(process_index)).cloned().collect(),
        })
    }

    // Plans the placement of the source of `views` (i.e., the source as each process sees it, in process order) on
    // `mesh` according to `sharding` on every process, and returns the strategy that all of them choose (or the
    // strategies of all of them in process order, separated by ` | `, when they differ) when their plans agree, or how
    // they disagree otherwise, preceded by those strategies when every process plans without failing. The plans agree
    // when every process fails alike before it posts anything or none fails, every send has exactly one matching
    // receive on its peer, every receive has a matching send, and every process that owns a device of a program that
    // one process runs runs the same program. The processes may choose different strategies only when that holds
    // (e.g., a process whose own value lies on its devices and whose devices hold the whole destination).
    //
    // Agreeing plans are not enough when a process holds some buffers of the source, because what only it knows (i.e.,
    // which of its shards have buffers) could still stop it before it posts the transfers that its peers expect. The
    // view for process 0 runs its part of the strategy with the production path when it holds buffers and plans
    // transfers. The CPU plugin lacks the cross-host transfers extension, so that part must fail with exactly the
    // error of its first transfer, and any other outcome is a disagreement.
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
                let outcomes = works.iter().map(|work| work.as_ref().map(|work| work.strategy)).collect::<Vec<_>>();
                return Err(format!("processes disagree on failing: {outcomes:?}"));
            }
        };
        let strategies = works.iter().map(|work| work.strategy).collect::<Vec<_>>();
        let planned = if strategies.iter().all(|strategy| *strategy == strategies[0]) {
            strategies[0].to_string()
        } else {
            strategies.join(" | ")
        };
        for (process_index, work) in works.iter().enumerate() {
            for send in &work.sends {
                let peer = send.destination.process_index();
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
                    source.clone().place_with_strategy(strategy, mesh, sharding, memory, layout.as_ref(), false);
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

    #[test]
    fn test_array_to_placement_reshards_fully_addressable_array() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(client_devices[0].id().unwrap(), client_devices[0].process_index().unwrap())],
        )
        .unwrap();
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        // Use an even split of 4 elements over 2 devices. XLA's CPU SPMD partitioner does not currently support uneven
        // splits (e.g., 5 elements over 2 devices), for which the output buffers of the devices would need different
        // shapes (i.e., `[3]` and `[2]`). The partitioner returns the full unsliced buffer instead, while ryft's shard
        // descriptors expect the exact logical shape of every device shard. Revisit this if a future XLA release or
        // backend handles uneven splits.
        let source_values = [0.0f32, 1.0, 2.0, 3.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(source_sharding)
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            source_mesh,
            None,
            values_to_bytes::<f32>(source_values.as_slice()).as_slice(),
        )
        .unwrap();

        let target_mesh = domain_mesh(client, "x", 2);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let moved_array = source_array
            .to_placement(&session, Placement::new(target_mesh.clone(), target_sharding).unwrap())
            .unwrap();

        assert_eq!(moved_array.addressable_shards().count(), 2);
        assert_eq!(
            moved_array.sharding().visualize().unwrap().render(false),
            indoc! {"
                    ┌─────┬─────┐
                    │  0  │  1  │
                    └─────┴─────┘
                "}
            .trim_end()
            .to_string()
        );
        assert_eq!(shard_values(&moved_array), vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
    }

    #[test]
    fn test_array_to_placement_splits_unevenly() {
        // Exact local shapes require differently sized slices for the split and a concatenation for the gather.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let shard_values = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<i32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>()
        };
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Auto] {
            let mesh = two_device_mesh(client, axis_type);
            let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
            let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let source = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::I32, [5]),
                mesh.clone(),
                Some(replicated.clone()),
                values_to_bytes(&[0i32, 1, 2, 3, 4]),
            )
            .unwrap();
            let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();
            let split = source.to_placement(&session, target.clone()).unwrap();
            let on_devices = source.clone().reshard_on_devices(&session, &mesh, &sharded, false).unwrap();
            let donated = source.into_placement(&session, target).unwrap();
            for placed in [&split, &donated, &on_devices] {
                assert_eq!(placed.sharding(), &sharded);
                assert_eq!(shard_values(placed), vec![vec![0, 1, 2], vec![3, 4]]);
            }
            let gathered = split.reshard_on_devices(&session, &mesh, &replicated, false).unwrap();
            assert_eq!(gathered.sharding(), &replicated);
            assert_eq!(shard_values(&gathered), vec![vec![0, 1, 2, 3, 4]; 2]);
        }
    }

    #[test]
    fn test_array_to_placement_splits_unevenly_with_and_without_artifact_sharing() {
        // A replicated `u8[5]` on a device of this process and a device of another process is split into shards of
        // three and two elements. Each process slices its own replica on its own device and leads the compilation of
        // that slice, so sharing compilation artifacts with the other process changes nothing. Neither does a declared
        // layout that the buffers already have, nor a memory other than device memory, which each process stages its
        // replica out of and moves its shard back into on its own devices.
        let client = cpu_client(1);
        let local = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote = Device::new(local.id() + 1, local.process_index() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![local, remote]).unwrap();
        let sharded = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let timeout = Duration::from_secs(1);
        for policy in [
            CompilationArtifactExchangePolicy::Disabled,
            CompilationArtifactExchangePolicy::PreferSharing { timeout },
            CompilationArtifactExchangePolicy::RequireSharing { timeout },
        ] {
            let cache = CompilationCache::new().with_artifact_exchange(Arc::new(PeerCompilationExchange), policy);
            let session = Session::builder().client(client).compilation_cache(cache).build().unwrap();
            let vector = ArrayType::new_static(DataType::U8, [5]);
            let default_layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
            for (r#type, kind) in [
                (vector.clone(), "device"),
                (vector.clone().with_layout(default_layout), "device"),
                (vector.with_memory(Memory::Host { pinned: true }), "pinned_host"),
            ] {
                let source =
                    Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, vec![1u8, 2, 3, 4, 5])
                        .unwrap();
                let placed = source.to_placement(&session, Placement::new(mesh.clone(), sharded.clone()).unwrap());
                let placed = placed.map_err(|error| error.to_string());
                let bytes = placed.clone().and_then(|placed| {
                    let shard = placed.addressable_shards().next().unwrap().clone();
                    shard.copy_to_host_bytes().map_err(|error| error.to_string())
                });
                assert_eq!(bytes, Ok(vec![1, 2, 3]), "{policy:?} {}", r#type);
                let placed = placed.unwrap();
                let shard = placed.addressable_shards().next().unwrap();
                assert_eq!(placed.r#type().memory(), r#type.memory(), "{policy:?} {}", r#type);
                assert_eq!(placed.layout(), r#type.layout(), "{policy:?} {}", r#type);
                assert_eq!(shard.buffer().unwrap().memory().unwrap().kind().unwrap(), kind);
            }
        }
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
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message.contains("empty buffer failed"),
        ));
    }

    #[test]
    fn test_array_to_placement_returns_array_of_session() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let meshed_session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let first_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![mesh.devices()[0]],
        )
        .unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            first_device_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();

        // The identity placement and the placement that copies whole shards both return arrays of the receiving
        // session, which has a default mesh that the session of the input lacks.
        let unchanged = Placement::new(
            first_device_mesh.clone(),
            Sharding::replicated(first_device_mesh.logical_mesh().clone(), 1),
        )
        .unwrap();
        let copied = Placement::new(mesh.clone(), replicated).unwrap();
        for target in [unchanged, copied] {
            let placed = array.to_placement(&meshed_session, target.clone()).unwrap();
            let donated = array.clone().into_placement(&meshed_session, target).unwrap();
            for placed in [placed, donated] {
                assert!(placed.session().is_same_session(&meshed_session));
                assert_eq!(placed.session().device_mesh().unwrap(), &mesh);
            }
        }

        // An array of another client is rejected even when its devices have the same IDs as those of this client.
        let other_session = Session::builder().client(other_cpu_client(2)).build().unwrap();
        let other_array = Array::from_host_buffer(
            &other_session,
            ArrayType::new_static(DataType::F32, [2]),
            first_device_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert_eq!(other_array.mesh(), &first_device_mesh);
        let target = Placement::new(first_device_mesh.clone(), other_array.sharding().clone()).unwrap();
        for result in
            [other_array.to_placement(&session, target.clone()), other_array.clone().into_placement(&session, target)]
        {
            assert!(matches!(
                result,
                Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))
                    if message == "cannot associate an array with a session of a different PJRT client",
            ));
        }
    }

    #[test]
    fn test_array_to_placement_commits_result() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert!(!array.is_committed());

        // A destination commits the result, even one that leaves the array where it is, and the source is unchanged.
        let devices = client.addressable_devices().unwrap();
        for device in [Device::from_pjrt(&devices[0]).unwrap(), Device::from_pjrt(&devices[1]).unwrap()] {
            let placed = array.to_placement(&session, PlacementRequest::from(device)).unwrap();
            assert_eq!(placed.mesh().devices(), [device]);
            assert!(placed.is_committed());
            let donated = array.clone().into_placement(&session, PlacementRequest::from(device)).unwrap();
            assert_eq!(donated.mesh().devices(), [device]);
            assert!(donated.is_committed());
        }
        assert!(!array.is_committed());
    }

    #[test]
    fn test_array_to_placement_keeps_local_assignment_only_in_place() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let source = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        assert!(source.is_locally_assigned());
        let own_mesh = source.mesh().clone();
        let pinned = Memory::Host { pinned: true };

        // A destination that names only a memory or a layout commits the result, as in JAX, but leaves it on the
        // devices that this process chose on its own.
        for placed in [
            source.to_placement(&session, pinned).unwrap(),
            source.clone().into_placement(&session, pinned).unwrap(),
            source
                .to_placement(
                    &session,
                    PlacementRequest::InPlace {
                        memory: None,
                        layout: Some(Layout::Tiled(TiledLayout::new(vec![0], vec![]))),
                    },
                )
                .unwrap(),
        ] {
            assert_eq!(placed.mesh(), &own_mesh);
            assert!(placed.is_committed());
            assert!(placed.is_locally_assigned());
        }

        // A destination that names devices or a sharding names the devices of the result, even when they are the
        // devices of the source.
        let same_devices = Placement::new(own_mesh.clone(), source.sharding().clone()).unwrap();
        let other_devices = Placement::new(
            two_device_sub_mesh_x(client),
            Sharding::replicated(two_device_sub_mesh_x(client).logical_mesh().clone(), 1),
        )
        .unwrap();
        for target in [same_devices.into(), other_devices.into(), PlacementRequest::from(own_mesh.devices()[0])] {
            let placed = source.to_placement(&session, target.clone()).unwrap();
            assert!(!placed.is_locally_assigned());
            assert!(!source.clone().into_placement(&session, target).unwrap().is_locally_assigned());
        }
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_to_placement_copies_matching_local_shards_without_full_source_addressability() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_source_buffer = client
            .buffer(values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
            .unwrap();
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let source_array = Array::from_addressable_buffers(
            &session,
            source_array_type,
            mesh.clone(),
            Some(sharding.clone()),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array =
            source_array.to_placement(&session, Placement::new(mesh.clone(), sharding).unwrap()).unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied_array.device_shard(local_device_id).unwrap().copy_to_host_bytes().unwrap();
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_array_to_placement_copies_available_local_replica() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let devices = mesh.devices().to_vec();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let buffer = client.buffer([7u8].as_slice(), BufferType::U8, [1u64], None, client_devices[0].clone(), None);
        let source = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::U8, [1]),
            mesh.clone(),
            Some(replicated),
            vec![buffer.unwrap()],
        )
        .unwrap();
        assert!(source.device_shard(devices[1].id()).unwrap().buffer().is_none());
        assert_eq!(source.copy_to_host_bytes().unwrap(), [7]);

        // The plan names the replica on the destination device, which this array lacks. Only this process takes part
        // in the move, so the replica on the other device of this process serves it instead.
        let source_buffer = Arc::clone(source.device_shard(devices[0].id()).unwrap().buffer().unwrap());
        let target = PlacementRequest::from(devices[1]);
        for placed in [source.to_placement(&session, target.clone()), source.clone().into_placement(&session, target)] {
            let placed = placed.unwrap();
            assert_eq!(placed.mesh().devices(), [devices[1]]);
            let buffer = placed.shards()[0].buffer().unwrap();
            assert!(!Arc::ptr_eq(buffer, &source_buffer));
            assert_eq!(buffer.device().unwrap().id().unwrap(), devices[1].id());
            assert_eq!(buffer.dimensions().unwrap(), [1]);
            assert_eq!(placed.block_until_ready().map_err(|error| error.to_string()), Ok(()));
            assert_eq!(placed.copy_to_host_bytes().unwrap(), [7]);
        }

        // A compiled reshard to another mesh moves the replicas of a replicated source as whole shards, and no program
        // reads the source itself, so a missing replica is served by the other one here too, rather than ruling out
        // the device work. The declared layout rules out shard regions, and the reversed mesh names the replica on
        // the first device for its second shard, which this array lacks in the second case.
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let logical_mesh = mesh.logical_mesh().clone();
        let reversed = DeviceMesh::new(logical_mesh.clone(), vec![devices[1], devices[0]]).unwrap();
        let rows = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        for held in [vec![0, 1], vec![1]] {
            let session = Session::builder().client(client).build().unwrap();
            let buffers = held
                .iter()
                .map(|&index: &usize| {
                    let memory = client_devices[index].default_memory().unwrap();
                    let values = values_to_bytes(&[1f32, 2., 3., 4.]);
                    let layout = Some(column_major.to_pjrt().unwrap());
                    client.buffer(values.as_slice(), BufferType::F32, [2u64, 2], None, memory, layout).unwrap()
                })
                .collect();
            let r#type = ArrayType::new_static(DataType::F32, [2, 2]).with_layout(column_major.clone());
            let replicated = Some(Sharding::replicated(logical_mesh.clone(), 2));
            let source = Array::from_addressable_buffers(&session, r#type, mesh.clone(), replicated, buffers).unwrap();
            let placed = source.to_placement(&session, Placement::new(reversed.clone(), rows.clone()).unwrap());
            let placed = placed.unwrap();
            assert_eq!(placed.block_until_ready().map_err(|error| error.to_string()), Ok(()));
            assert_eq!(placed.layout(), Some(&column_major));
            assert_eq!(shard_values(&placed), [vec![1., 2.], vec![3., 4.]]);
            assert_eq!(session.cached_program_count(), 4, "the device work must place a source with replicas {held:?}");
        }

        // A missing shard whose slice has no other replica on this process is still an error.
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let buffer = client.buffer([7u8].as_slice(), BufferType::U8, [1u64], None, client_devices[0].clone(), None);
        let source = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::U8, [2]),
            mesh.clone(),
            Some(sharded.clone()),
            vec![buffer.unwrap()],
        )
        .unwrap();
        let reversed = DeviceMesh::new(mesh.logical_mesh().clone(), vec![devices[1], devices[0]]).unwrap();
        let error = source.to_placement(&session, Placement::new(reversed, sharded).unwrap()).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("array shard #1 on device {} has no addressable buffer", devices[1].id()),
        );
    }

    #[test]
    fn test_array_to_placement_slices_available_local_replica() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let remote = Device::new(devices[1].id() + 1, devices[1].process_index() + 1);
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 3, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0], devices[1], remote],
        )
        .unwrap();
        let destination_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("destination", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0], remote],
        )
        .unwrap();
        let replicated = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let split =
            Sharding::new(destination_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["destination"])])
                .unwrap();
        let target = Placement::new(destination_mesh.clone(), split.clone()).unwrap();

        // The source holds only one of its local replicas, and the plan names the replica on the destination device.
        // Only this process takes part in the region of that device, so whichever local replica holds a buffer serves
        // it, sliced on its own device and copied to the destination device when that differs.
        for (holder, holder_device) in client_devices.iter().zip(&devices) {
            let buffer =
                client.buffer([10u8, 20, 30, 40].as_slice(), BufferType::U8, [4u64], None, holder.clone(), None);
            let source = Array::from_addressable_buffers(
                &session,
                ArrayType::new_static(DataType::U8, [4]),
                source_mesh.clone(),
                Some(replicated.clone()),
                vec![buffer.unwrap()],
            )
            .unwrap();
            assert_eq!(source.copy_to_host_bytes().unwrap(), [10, 20, 30, 40]);
            let placed = source.to_placement(&session, target.clone()).unwrap();
            assert_eq!(placed.mesh(), &destination_mesh);
            assert_eq!(placed.sharding(), &split);
            assert!(placed.is_committed());
            assert_eq!(placed.addressable_shards().count(), 1);
            assert!(placed.device_shard(remote.id()).unwrap().buffer().is_none());
            let buffer = placed.device_shard(devices[0].id()).unwrap().buffer().unwrap();
            assert_eq!(buffer.device().unwrap().id().unwrap(), devices[0].id());
            assert_eq!(buffer.dimensions().unwrap(), [2]);
            assert_eq!(placed.block_until_ready().map_err(|error| error.to_string()), Ok(()));
            assert_eq!(placed.device_shard(devices[0].id()).unwrap().copy_to_host_bytes().unwrap(), [10, 20]);
            assert_eq!(
                source.device_shard(holder_device.id()).unwrap().copy_to_host_bytes().unwrap(),
                [10, 20, 30, 40],
            );
        }
    }

    #[test]
    fn test_array_to_placement_rejects_non_addressable_source_shards() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 1)],
        )
        .unwrap();
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let source_array = Array::from_addressable_buffers(
            &session,
            source_array_type,
            source_mesh,
            Some(source_sharding),
            Vec::new(),
        )
        .unwrap();
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("y", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0)],
        )
        .unwrap();
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);

        let error = source_array
            .to_placement(&session, Placement::new(target_mesh, target_sharding).unwrap())
            .unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::MissingShardBuffer { shard_index: 0, device_id: 0 },
            ),
        ));
        assert_eq!(error.to_string(), "array shard #0 on device 0 has no addressable buffer");
    }

    #[test]
    fn test_array_to_placement_combines_unreduced_device_values() {
        // The devices along the unreduced axis `x` hold summands of the logical value, and `y` tiles its only
        // dimension. Every destination shard has the slices of a source shard, but a copy of that shard would hold
        // one summand instead of the logical value.
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
        let tiled = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["y"])]).unwrap();
        let unreduced = tiled.clone().with_unreduced_axes(["x"]).unwrap();

        let buffers = [[1f32, 2.], [3., 4.], [10., 20.], [30., 40.]]
            .iter()
            .zip(&devices)
            .map(|(values, device)| {
                client
                    .buffer(values_to_bytes(values).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(unreduced.clone()).unwrap();
        let summands = Array::from_addressable_buffers(&session, r#type, mesh.clone(), None, buffers).unwrap();
        let reduced = summands.to_placement(&session, Placement::new(mesh.clone(), tiled.clone()).unwrap()).unwrap();
        assert_eq!(reduced.sharding(), &tiled);
        assert_eq!(shard_values(&reduced), vec![vec![11., 22.], vec![33., 44.], vec![11., 22.], vec![33., 44.]]);

        // In the other direction, a copy of every shard would hold the logical value once per device along `x`.
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(tiled).unwrap();
        let value = Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
            .unwrap();
        let summands = value.to_placement(&session, Placement::new(mesh, unreduced.clone()).unwrap()).unwrap();
        assert_eq!(summands.sharding(), &unreduced);
        assert_eq!(summands.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2., 3., 4.]);
    }

    #[test]
    fn test_array_to_placement_completes_pending_reduction_with_and_without_donation() {
        // Two devices hold the contributions `[1, 5, 3, 0]` and `[2, 4, 3, 7]` to an `f32[4]` value with a pending
        // reduction of each kind, over an auto and over an explicit mesh axis, which a placement splits along that
        // axis, on the same mesh and onto two other devices. Completing a pending sum (e.g., of the partial sums of a
        // gradient) is a capability of the placement of an array, whether or not it donates its source, and it
        // completes on the devices for either axis type, because the host round trip needs a destination that this
        // process fully addresses. On the same mesh, one compiled reshard completes it (i.e., one program with and one
        // without a donated input). Across meshes, the gather of the source completes it before its replicas move.
        // XLA's maximum and minimum reductions drop NaN values, so a pending maximum or minimum of floating point
        // values takes the host round trip from both entry points and compiles nothing (refer to
        // `test_array_to_placement_preserves_value_across_reduction_state_changes`). Each axis type has a session of
        // its own, so that every cell that compiles a program adds it to the count.
        let client = cpu_client(4);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mut outcomes = Vec::new();
        for axis_type in [MeshAxisType::Auto, MeshAxisType::Explicit] {
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices[..2].to_vec()).unwrap();
            let other_mesh = DeviceMesh::new(logical_mesh.clone(), devices[2..].to_vec()).unwrap();
            let split = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            for kind in [UnreducedKind::Sum, UnreducedKind::Max, UnreducedKind::Min] {
                let unreduced = Sharding::replicated(logical_mesh.clone(), 1)
                    .with_unreduced_axes(["x"])
                    .unwrap()
                    .with_unreduced_kind(kind)
                    .unwrap();
                let source = || {
                    let contributions = devices
                        .iter()
                        .zip([[1f32, 5., 3., 0.], [2., 4., 3., 7.]])
                        .map(|(device, values)| (device.id(), Arc::<[u8]>::from(values_to_bytes(&values))))
                        .collect::<Vec<_>>();
                    let r#type = ArrayType::new_static(DataType::F32, [4]);
                    let sharding = Some(unreduced.clone());
                    Array::from_host_shards(&session, r#type, mesh.clone(), sharding, contributions).unwrap()
                };
                for (entry_point, destination, consume) in
                    [("borrowed", &mesh, false), ("consumed", &mesh, true), ("across meshes", &other_mesh, false)]
                {
                    let programs = session.cached_program_count();
                    let target = Placement::new(destination.clone(), split.clone()).unwrap();
                    let placed = if consume {
                        source().into_placement(&session, target)
                    } else {
                        source().to_placement(&session, target)
                    };
                    let placed = placed.map(|placed| {
                        assert_eq!((placed.mesh(), placed.sharding()), (destination, &split));
                        shard_values(&placed)
                    });
                    let programs = session.cached_program_count() - programs;
                    outcomes.push(format!("{axis_type:?} {kind:?} {entry_point}: +{programs} {placed:?}"));
                }
            }
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                Auto Sum borrowed: +1 Ok([[3.0, 9.0], [6.0, 7.0]])
                Auto Sum consumed: +1 Ok([[3.0, 9.0], [6.0, 7.0]])
                Auto Sum across meshes: +2 Ok([[3.0, 9.0], [6.0, 7.0]])
                Auto Max borrowed: +0 Ok([[2.0, 5.0], [3.0, 7.0]])
                Auto Max consumed: +0 Ok([[2.0, 5.0], [3.0, 7.0]])
                Auto Max across meshes: +0 Ok([[2.0, 5.0], [3.0, 7.0]])
                Auto Min borrowed: +0 Ok([[1.0, 4.0], [3.0, 0.0]])
                Auto Min consumed: +0 Ok([[1.0, 4.0], [3.0, 0.0]])
                Auto Min across meshes: +0 Ok([[1.0, 4.0], [3.0, 0.0]])
                Explicit Sum borrowed: +1 Ok([[3.0, 9.0], [6.0, 7.0]])
                Explicit Sum consumed: +1 Ok([[3.0, 9.0], [6.0, 7.0]])
                Explicit Sum across meshes: +2 Ok([[3.0, 9.0], [6.0, 7.0]])
                Explicit Max borrowed: +0 Ok([[2.0, 5.0], [3.0, 7.0]])
                Explicit Max consumed: +0 Ok([[2.0, 5.0], [3.0, 7.0]])
                Explicit Max across meshes: +0 Ok([[2.0, 5.0], [3.0, 7.0]])
                Explicit Min borrowed: +0 Ok([[1.0, 4.0], [3.0, 0.0]])
                Explicit Min consumed: +0 Ok([[1.0, 4.0], [3.0, 0.0]])
                Explicit Min across meshes: +0 Ok([[1.0, 4.0], [3.0, 0.0]])"},
        );
    }

    #[test]
    fn test_validate_reduction_transition() {
        // Every change of reduction state along the axis `x` of size two between no state, a pending reduction of each
        // kind, and a completed reduction, for a data type of every class of the rule, as `.` for a compiled reshard,
        // `h` for a host round trip, and `E` for an error. The rule reads neither the type of the axis nor anything
        // that only one process knows.
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
        let mut outcomes = Vec::new();
        for axis_type in [MeshAxisType::Auto, MeshAxisType::Explicit] {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
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
                        .map(|(_, output)| {
                            match validate_reduction_transition(
                                data_type,
                                &StaticShape::new(vec![4]),
                                &mesh,
                                input,
                                output,
                            ) {
                                Ok(None) => ".",
                                Ok(Some(_)) => "h",
                                Err(_) => "E",
                            }
                        })
                        .collect::<Vec<_>>();
                    outcomes.push(format!("{axis_type:?} {data_type} {name}: {}", codes.join(" ")));
                }
            }
        }
        let rows = indoc! {"
            bool none: . . h h .
            bool sum: . . h h .
            bool max: . . . h .
            bool min: . . h . .
            bool reduced: . . h h .
            u1 none: . . h h .
            u1 sum: h . h h h
            u1 max: . . . h .
            u1 min: . . h . .
            u1 reduced: . . h h .
            i1 none: . . h h .
            i1 sum: h . h h h
            i1 max: h h . h h
            i1 min: h h h . h
            i1 reduced: . . h h .
            u2 none: . . h h .
            u2 sum: h . h h h
            u2 max: h h . h h
            u2 min: h h h . h
            u2 reduced: . . h h .
            i32 none: . . h h .
            i32 sum: . . h h .
            i32 max: . . . h .
            i32 min: . . h . .
            i32 reduced: . . h h .
            f6e2m3fn none: . h h h .
            f6e2m3fn sum: h . h h h
            f6e2m3fn max: h h . h h
            f6e2m3fn min: h h h . h
            f6e2m3fn reduced: . h h h .
            f8e4m3fnuz none: . . h h .
            f8e4m3fnuz sum: . . h h .
            f8e4m3fnuz max: h h . h h
            f8e4m3fnuz min: h h h . h
            f8e4m3fnuz reduced: . . h h .
            f32 none: . . h h .
            f32 sum: . . h h .
            f32 max: h h . h h
            f32 min: h h h . h
            f32 reduced: . . h h .
            c64 none: . h h h .
            c64 sum: . . h h .
            c64 max: h h . h h
            c64 min: h h h . h
            c64 reduced: . h h h .
            f8e8m0fnu none: . E h h .
            f8e8m0fnu sum: h . h h h
            f8e8m0fnu max: h E . h h
            f8e8m0fnu min: h E h . h
            f8e8m0fnu reduced: . E h h ."};
        let expected = ["Auto", "Explicit"]
            .iter()
            .flat_map(|axis_type| rows.lines().map(move |row| format!("{axis_type} {row}")))
            .collect::<Vec<_>>();
        assert_eq!(outcomes, expected);

        // The diagnostics name the data type, both states, the axis, and the reason. A value without elements needs no
        // additive identity, so even a data type without one introduces a pending sum.
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
        let pending = |kind: UnreducedKind| {
            replicated.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap()
        };
        let shape = StaticShape::new(vec![4]);
        let outcome = |data_type: DataType, shape: &StaticShape, input: &Sharding, output: &Sharding| {
            match validate_reduction_transition(data_type, shape, &mesh, input, output) {
                Ok(None) => "compiled reshard".to_string(),
                Ok(Some(reason)) => format!("host round trip: {reason}"),
                Err(error) => format!("error: {error}"),
            }
        };
        assert_eq!(
            [
                outcome(DataType::U1, &shape, &pending(UnreducedKind::Sum), &replicated),
                outcome(DataType::I1, &shape, &pending(UnreducedKind::Max), &replicated),
                outcome(DataType::U2, &shape, &pending(UnreducedKind::Sum), &replicated),
                outcome(DataType::F6E2M3FN, &shape, &pending(UnreducedKind::Sum), &replicated),
                outcome(DataType::F8E8M0FNU, &shape, &pending(UnreducedKind::Sum), &replicated),
                outcome(DataType::C64, &shape, &pending(UnreducedKind::Min), &replicated),
                outcome(DataType::F32, &shape, &pending(UnreducedKind::Max), &replicated),
                outcome(DataType::F32, &shape, &pending(UnreducedKind::Sum), &replicated),
                outcome(DataType::F32, &shape, &replicated, &pending(UnreducedKind::Sum)),
                outcome(DataType::F6E2M3FN, &shape, &replicated, &pending(UnreducedKind::Sum)),
                outcome(DataType::C64, &shape, &replicated, &pending(UnreducedKind::Sum)),
                outcome(DataType::I32, &shape, &replicated, &pending(UnreducedKind::Max)),
                outcome(DataType::F8E8M0FNU, &shape, &replicated, &pending(UnreducedKind::Sum)),
                outcome(DataType::F8E8M0FNU, &StaticShape::new(vec![0]), &replicated, &pending(UnreducedKind::Sum)),
            ]
            .join("\n"),
            [
                "host round trip: a compiled reshard cannot take `u1` values from a pending `sum` reduction to no \
                 reduction state along mesh axis `x`, because XLA adds one-bit integers as predicates (i.e., with a \
                 logical or), while their sum wraps around",
                "host round trip: a compiled reshard cannot take `i1` values from a pending `max` reduction to no \
                 reduction state along mesh axis `x`, because XLA orders one-bit integers as predicates, which puts \
                 `-1` above `0`",
                "host round trip: a compiled reshard cannot take `u2` values from a pending `sum` reduction to no \
                 reduction state along mesh axis `x`, because XLA does not compile reductions of two-bit integers",
                "host round trip: a compiled reshard cannot take `f6e2m3fn` values from a pending `sum` reduction to \
                 no reduction state along mesh axis `x`, because XLA does not compile reductions of six-bit floating \
                 point values",
                "host round trip: a compiled reshard cannot take `f8e8m0fnu` values from a pending `sum` reduction to \
                 no reduction state along mesh axis `x`, because XLA does not add `f8e8m0fnu` values as the format \
                 does (e.g., `2^-127 + 2^-127` gives NaN)",
                "host round trip: a compiled reshard cannot take `c64` values from a pending `min` reduction to no \
                 reduction state along mesh axis `x`, because XLA has no maximum or minimum reduction of complex \
                 values",
                "host round trip: a compiled reshard cannot take `f32` values from a pending `max` reduction to no \
                 reduction state along mesh axis `x`, because XLA's maximum and minimum reductions drop NaN values, \
                 which the maximum and minimum of a compiled program propagate",
                "compiled reshard",
                "compiled reshard",
                "host round trip: a compiled reshard cannot take `f6e2m3fn` values from no reduction state to a \
                 pending `sum` reduction along mesh axis `x`, because XLA aborts the process when it introduces a \
                 pending sum of six-bit floating point values",
                "host round trip: a compiled reshard cannot take `c64` values from no reduction state to a pending \
                 `sum` reduction along mesh axis `x`, because XLA crashes the process when it introduces a pending sum \
                 of complex values",
                "host round trip: a compiled reshard cannot take `i32` values from no reduction state to a pending \
                 `max` reduction along mesh axis `x`, because XLA can only introduce a pending `sum` reduction",
                "error: type `f8e8m0fnu` has no additive identity to split a host value into summands along unreduced \
                 axes",
                "compiled reshard",
            ]
            .join("\n"),
        );
    }

    #[test]
    fn test_validate_reduction_transition_plans_only_programs_that_lowering_accepts() {
        // Every change of reduction state that the planner leaves to a compiled reshard lowers, because the lowering
        // rejects the changes that the same classification (`collective_reduction`) does not let the planner make on
        // the devices. Each such cell lowers the program that `Array::compile_reshard_on_same_mesh` stages, over the
        // mesh of `ReshardProgram::compiled_mesh`, for every data type with values, both axis types, and every pair of
        // states (none, a pending sum, maximum, or minimum, and a completed reduction). Only lowering runs, so that a
        // wrong plan cannot abort the test process.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
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
                        match validate_reduction_transition(data_type, &shape, &mesh, input, output) {
                            Ok(None) => planned += 1,
                            Ok(Some(_)) | Err(_) => {
                                declined += 1;
                                continue;
                            }
                        }
                        let program =
                            ReshardProgram { mesh: mesh.clone(), input: input.clone(), output: output.clone() };
                        let compiled_mesh = program.compiled_mesh().unwrap();
                        let logical_mesh = compiled_mesh.logical_mesh().clone();
                        let options = Options::new(compiled_mesh)
                            .with_input_shardings(vec![input.with_mesh(logical_mesh.clone()).unwrap()])
                            .with_output_shardings(vec![output.with_mesh(logical_mesh).unwrap()]);
                        let r#type = ArrayType::new_static(data_type, [4]);
                        let lowered = crate::stage(|input| input, r#type, options).and_then(
                            |staged: crate::StagedFunction<ArrayType, ArrayType>| {
                                staged.into_inner().lower(&session).map(|_| ())
                            },
                        );
                        assert!(lowered.is_ok(), "{axis_type:?} {data_type} {input} -> {output}: {lowered:?}");
                    }
                }
            }
        }
        assert_eq!((planned, declined), (812, 788));
    }

    #[test]
    fn test_array_to_placement_keeps_or_declines_reduction_state_changes() {
        // Placements of an `f32[4]` value on a 2x2 mesh whose axes `x` and `y` are explicit or auto, from and to
        // shardings with reduction state, with the number of programs that each entry point compiles in a session of
        // its own (i.e., on the same mesh with and without donation, and onto the same devices in reverse order). A
        // compiled reshard keeps the reduction state of an axis as a pure move over an explicit axis, so a pending
        // reduction that a destination keeps on the same mesh moves without being completed, whatever the type of its
        // axes, and it makes every other change over auto axes (i.e., it completes a pending reduction and introduces a
        // pending sum or a completed reduction). XLA cannot introduce a pending maximum or minimum, so a destination
        // with one that the source does not keep compiles nothing and takes the host round trip (e.g., across meshes,
        // where the gather completes the reduction that the final reshard would have to introduce again). Every cell
        // reads back the logical value of its source with the destination sharding.
        let client = cpu_client(4);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let (a, b, c, d) = ([1f32, 5., 3., 0.], [2f32, 4., 3., 7.], [10f32, 20., 30., 40.], [100f32, 200., 300., 400.]);
        let combine = |kind: UnreducedKind, values: &[[f32; 4]]| -> Vec<f32> {
            (0..4)
                .map(|index| {
                    let column = values.iter().map(|value| value[index]);
                    match kind {
                        UnreducedKind::Sum => column.sum(),
                        UnreducedKind::Max => column.fold(f32::MIN, f32::max),
                        UnreducedKind::Min => column.fold(f32::MAX, f32::min),
                    }
                })
                .collect()
        };
        let mut outcomes = Vec::new();
        for (x_type, y_type) in [
            (MeshAxisType::Explicit, MeshAxisType::Explicit),
            (MeshAxisType::Auto, MeshAxisType::Auto),
            (MeshAxisType::Explicit, MeshAxisType::Auto),
            (MeshAxisType::Auto, MeshAxisType::Explicit),
        ] {
            let logical_mesh =
                LogicalMesh::new(vec![MeshAxis::new("x", 2, x_type).unwrap(), MeshAxis::new("y", 2, y_type).unwrap()])
                    .unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
            let reversed_mesh = DeviceMesh::new(logical_mesh.clone(), devices.iter().rev().copied().collect()).unwrap();
            let replicated = Sharding::replicated(logical_mesh.clone(), 1);
            let split_y = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
            let unconstrained = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::unconstrained()]).unwrap();
            let unreduced = |sharding: &Sharding, axes: &[&str], kind: UnreducedKind| {
                sharding
                    .clone()
                    .with_unreduced_axes(axes.iter().copied())
                    .unwrap()
                    .with_unreduced_kind(kind)
                    .unwrap()
            };
            let reduced = |sharding: &Sharding| sharding.clone().with_reduced_axes(["x"]).unwrap();
            let maximum = unreduced(&replicated, &["x"], UnreducedKind::Max);
            // Each source is its sharding, the contributions of the devices in mesh order, and its logical value.
            let sum_x = (
                unreduced(&replicated, &["x"], UnreducedKind::Sum),
                vec![a.to_vec(), a.to_vec(), b.to_vec(), b.to_vec()],
                combine(UnreducedKind::Sum, &[a, b]),
            );
            let sum_x_split_y = (
                unreduced(&split_y, &["x"], UnreducedKind::Sum),
                vec![a[..2].to_vec(), a[2..].to_vec(), b[..2].to_vec(), b[2..].to_vec()],
                combine(UnreducedKind::Sum, &[a, b]),
            );
            let max_x = (
                maximum.clone(),
                vec![a.to_vec(), a.to_vec(), b.to_vec(), b.to_vec()],
                combine(UnreducedKind::Max, &[a, b]),
            );
            let min_x = (
                unreduced(&replicated, &["x"], UnreducedKind::Min),
                vec![a.to_vec(), a.to_vec(), b.to_vec(), b.to_vec()],
                combine(UnreducedKind::Min, &[a, b]),
            );
            let sum_y = (
                unreduced(&replicated, &["y"], UnreducedKind::Sum),
                vec![a.to_vec(), b.to_vec(), a.to_vec(), b.to_vec()],
                combine(UnreducedKind::Sum, &[a, b]),
            );
            let sum_x_y = (
                unreduced(&replicated, &["x", "y"], UnreducedKind::Sum),
                vec![a.to_vec(), b.to_vec(), c.to_vec(), d.to_vec()],
                combine(UnreducedKind::Sum, &[a, b, c, d]),
            );
            let plain = (replicated.clone(), vec![c.to_vec(); 4], c.to_vec());
            let plain_split_y =
                (split_y.clone(), vec![c[..2].to_vec(), c[2..].to_vec(), c[..2].to_vec(), c[2..].to_vec()], c.to_vec());
            let reduced_x = (reduced(&replicated), vec![c.to_vec(); 4], c.to_vec());
            let cells = [
                // Pure moves that keep reduction state.
                ("sum(x) -> sum(x) split(y)", &sum_x, unreduced(&split_y, &["x"], UnreducedKind::Sum)),
                ("sum(x) split(y) -> sum(x)", &sum_x_split_y, unreduced(&replicated, &["x"], UnreducedKind::Sum)),
                ("sum(x) -> sum(x) {?}", &sum_x, unreduced(&unconstrained, &["x"], UnreducedKind::Sum)),
                ("max(x) -> max(x) split(y)", &max_x, unreduced(&split_y, &["x"], UnreducedKind::Max)),
                ("min(x) -> min(x) split(y)", &min_x, unreduced(&split_y, &["x"], UnreducedKind::Min)),
                ("reduced(x) -> reduced(x) split(y)", &reduced_x, reduced(&split_y)),
                // A completed reduction along `x` and a kept one along `y`.
                ("sum(x, y) -> sum(y)", &sum_x_y, unreduced(&replicated, &["y"], UnreducedKind::Sum)),
                // Destinations with a pending maximum.
                ("sum(x) -> max(x)", &sum_x, maximum.clone()),
                ("max(x) -> max(x)", &max_x, maximum.clone()),
                ("min(x) -> max(x)", &min_x, maximum.clone()),
                ("sum(y) -> max(x)", &sum_y, maximum.clone()),
                ("sum(x, y) -> max(x)", &sum_x_y, maximum.clone()),
                ("replicated -> max(x)", &plain, maximum.clone()),
                ("split(y) -> max(x)", &plain_split_y, maximum.clone()),
                ("reduced(x) -> max(x)", &reduced_x, maximum.clone()),
            ];
            for (name, (source_sharding, contributions, logical_value), destination) in cells {
                let mut counts = Vec::new();
                for (destination_mesh, consume) in [(&mesh, false), (&mesh, true), (&reversed_mesh, false)] {
                    let session = Session::builder().client(client).build().unwrap();
                    let contributions = devices
                        .iter()
                        .zip(contributions)
                        .map(|(device, values)| (device.id(), Arc::<[u8]>::from(values_to_bytes(values))))
                        .collect::<Vec<_>>();
                    let r#type = ArrayType::new_static(DataType::F32, [4]);
                    let sharding = Some(source_sharding.clone());
                    let source =
                        Array::from_host_shards(&session, r#type, mesh.clone(), sharding, contributions).unwrap();
                    let target = Placement::new(destination_mesh.clone(), destination.clone()).unwrap();
                    let placed = if consume {
                        source.into_placement(&session, target)
                    } else {
                        source.to_placement(&session, target)
                    };
                    let placed = placed.unwrap();
                    assert_eq!((placed.mesh(), placed.sharding()), (destination_mesh, &destination), "{name}");
                    assert_eq!(&placed.copy_to_host().unwrap().elements::<f32>().unwrap(), logical_value, "{name}");
                    counts.push(format!("+{}", session.cached_program_count()));
                }
                outcomes.push(format!("{x_type:?}/{y_type:?} {name}: {}", counts.join(" ")));
            }
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                Explicit/Explicit sum(x) -> sum(x) split(y): +1 +1 +2
                Explicit/Explicit sum(x) split(y) -> sum(x): +1 +1 +2
                Explicit/Explicit sum(x) -> sum(x) {?}: +1 +1 +2
                Explicit/Explicit max(x) -> max(x) split(y): +1 +1 +0
                Explicit/Explicit min(x) -> min(x) split(y): +1 +1 +0
                Explicit/Explicit reduced(x) -> reduced(x) split(y): +1 +1 +4
                Explicit/Explicit sum(x, y) -> sum(y): +1 +1 +2
                Explicit/Explicit sum(x) -> max(x): +0 +0 +0
                Explicit/Explicit max(x) -> max(x): +0 +0 +0
                Explicit/Explicit min(x) -> max(x): +0 +0 +0
                Explicit/Explicit sum(y) -> max(x): +0 +0 +0
                Explicit/Explicit sum(x, y) -> max(x): +0 +0 +0
                Explicit/Explicit replicated -> max(x): +0 +0 +0
                Explicit/Explicit split(y) -> max(x): +0 +0 +0
                Explicit/Explicit reduced(x) -> max(x): +0 +0 +0
                Auto/Auto sum(x) -> sum(x) split(y): +1 +1 +2
                Auto/Auto sum(x) split(y) -> sum(x): +1 +1 +2
                Auto/Auto sum(x) -> sum(x) {?}: +1 +1 +2
                Auto/Auto max(x) -> max(x) split(y): +1 +1 +0
                Auto/Auto min(x) -> min(x) split(y): +1 +1 +0
                Auto/Auto reduced(x) -> reduced(x) split(y): +1 +1 +4
                Auto/Auto sum(x, y) -> sum(y): +1 +1 +2
                Auto/Auto sum(x) -> max(x): +0 +0 +0
                Auto/Auto max(x) -> max(x): +0 +0 +0
                Auto/Auto min(x) -> max(x): +0 +0 +0
                Auto/Auto sum(y) -> max(x): +0 +0 +0
                Auto/Auto sum(x, y) -> max(x): +0 +0 +0
                Auto/Auto replicated -> max(x): +0 +0 +0
                Auto/Auto split(y) -> max(x): +0 +0 +0
                Auto/Auto reduced(x) -> max(x): +0 +0 +0
                Explicit/Auto sum(x) -> sum(x) split(y): +1 +1 +2
                Explicit/Auto sum(x) split(y) -> sum(x): +1 +1 +2
                Explicit/Auto sum(x) -> sum(x) {?}: +1 +1 +2
                Explicit/Auto max(x) -> max(x) split(y): +1 +1 +0
                Explicit/Auto min(x) -> min(x) split(y): +1 +1 +0
                Explicit/Auto reduced(x) -> reduced(x) split(y): +1 +1 +4
                Explicit/Auto sum(x, y) -> sum(y): +1 +1 +2
                Explicit/Auto sum(x) -> max(x): +0 +0 +0
                Explicit/Auto max(x) -> max(x): +0 +0 +0
                Explicit/Auto min(x) -> max(x): +0 +0 +0
                Explicit/Auto sum(y) -> max(x): +0 +0 +0
                Explicit/Auto sum(x, y) -> max(x): +0 +0 +0
                Explicit/Auto replicated -> max(x): +0 +0 +0
                Explicit/Auto split(y) -> max(x): +0 +0 +0
                Explicit/Auto reduced(x) -> max(x): +0 +0 +0
                Auto/Explicit sum(x) -> sum(x) split(y): +1 +1 +2
                Auto/Explicit sum(x) split(y) -> sum(x): +1 +1 +2
                Auto/Explicit sum(x) -> sum(x) {?}: +1 +1 +2
                Auto/Explicit max(x) -> max(x) split(y): +1 +1 +0
                Auto/Explicit min(x) -> min(x) split(y): +1 +1 +0
                Auto/Explicit reduced(x) -> reduced(x) split(y): +1 +1 +4
                Auto/Explicit sum(x, y) -> sum(y): +1 +1 +2
                Auto/Explicit sum(x) -> max(x): +0 +0 +0
                Auto/Explicit max(x) -> max(x): +0 +0 +0
                Auto/Explicit min(x) -> max(x): +0 +0 +0
                Auto/Explicit sum(y) -> max(x): +0 +0 +0
                Auto/Explicit sum(x, y) -> max(x): +0 +0 +0
                Auto/Explicit replicated -> max(x): +0 +0 +0
                Auto/Explicit split(y) -> max(x): +0 +0 +0
                Auto/Explicit reduced(x) -> max(x): +0 +0 +0"},
        );
    }

    #[test]
    fn test_array_to_placement_preserves_value_across_reduction_state_changes() {
        // Placements that complete, introduce, keep, or change the kind of a pending reduction along the axis `x` of a
        // 2x2 mesh, for a data type of every class of `validate_reduction_transition`, compared with the host readback
        // of their source, which defines the value (refer to `crate::arrays::host`). The contributions along `x` are
        // every pair of some elements of each data type (e.g., negative zeros, NaN values, infinities, and extreme
        // integers, but no subnormal `f32` values, which the CPU plugin flushes in any addition). Each row lists, per
        // placement, the programs that `to_placement` and `into_placement` compile in sessions of their own, which is
        // `+0` for a host round trip, and asserts that both read back the value of their source bit for bit, where
        // every NaN encoding of a format counts as one value, or lists their common error. A sum of floating point or
        // complex values that the devices complete or introduce is compared as the contract of such sums states (refer
        // to `crate::arrays::host`), so a negative zero also matches a positive one there, and the end of this test
        // pins that difference. The axis type does not change the outcome, so both axis types must give the same row.
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
                for (name, source_kind, destination) in placements {
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
                        let target = Placement::new(mesh.clone(), destination.clone()).unwrap();
                        let placed = if consume {
                            source.into_placement(&session, target)
                        } else {
                            source.to_placement(&session, target)
                        };
                        let programs = session.cached_program_count();
                        // A sum that the devices complete or introduce follows the contract of floating point sums.
                        let sum_contract = programs > 0 && matches!(name, "sum -> none" | "none -> sum" | "max -> sum");
                        results.push(placed.and_then(|placed| {
                            assert_eq!(placed.sharding(), &destination, "{data_type} {name}");
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
        let missing_identity = "type `f8e8m0fnu` has no additive identity to split a host value into summands along \
                                unreduced axes";
        assert_eq!(
            rows.join("\n"),
            formatdoc! {"
                bool sum -> none: +1 +1
                bool max -> none: +1 +1
                bool min -> none: +1 +1
                bool none -> sum: +1 +1
                bool none -> max: +0 +0
                bool max -> sum: +1 +1
                bool sum -> sum split: +1 +1
                bool max -> max split: +1 +1
                u1 sum -> none: +0 +0
                u1 max -> none: +1 +1
                u1 min -> none: +1 +1
                u1 none -> sum: +1 +1
                u1 none -> max: +0 +0
                u1 max -> sum: +1 +1
                u1 sum -> sum split: +1 +1
                u1 max -> max split: +1 +1
                i1 sum -> none: +0 +0
                i1 max -> none: +0 +0
                i1 min -> none: +0 +0
                i1 none -> sum: +1 +1
                i1 none -> max: +0 +0
                i1 max -> sum: +0 +0
                i1 sum -> sum split: +1 +1
                i1 max -> max split: +1 +1
                u2 sum -> none: +0 +0
                u2 max -> none: +0 +0
                u2 min -> none: +0 +0
                u2 none -> sum: +1 +1
                u2 none -> max: +0 +0
                u2 max -> sum: +0 +0
                u2 sum -> sum split: +1 +1
                u2 max -> max split: +1 +1
                i4 sum -> none: +1 +1
                i4 max -> none: +1 +1
                i4 min -> none: +1 +1
                i4 none -> sum: +1 +1
                i4 none -> max: +0 +0
                i4 max -> sum: +1 +1
                i4 sum -> sum split: +1 +1
                i4 max -> max split: +1 +1
                u8 sum -> none: +1 +1
                u8 max -> none: +1 +1
                u8 min -> none: +1 +1
                u8 none -> sum: +1 +1
                u8 none -> max: +0 +0
                u8 max -> sum: +1 +1
                u8 sum -> sum split: +1 +1
                u8 max -> max split: +1 +1
                i32 sum -> none: +1 +1
                i32 max -> none: +1 +1
                i32 min -> none: +1 +1
                i32 none -> sum: +1 +1
                i32 none -> max: +0 +0
                i32 max -> sum: +1 +1
                i32 sum -> sum split: +1 +1
                i32 max -> max split: +1 +1
                f32 sum -> none: +1 +1
                f32 max -> none: +0 +0
                f32 min -> none: +0 +0
                f32 none -> sum: +1 +1
                f32 none -> max: +0 +0
                f32 max -> sum: +0 +0
                f32 sum -> sum split: +1 +1
                f32 max -> max split: +1 +1
                f8e5m2 sum -> none: +1 +1
                f8e5m2 max -> none: +0 +0
                f8e5m2 min -> none: +0 +0
                f8e5m2 none -> sum: +1 +1
                f8e5m2 none -> max: +0 +0
                f8e5m2 max -> sum: +0 +0
                f8e5m2 sum -> sum split: +1 +1
                f8e5m2 max -> max split: +1 +1
                f8e4m3fnuz sum -> none: +1 +1
                f8e4m3fnuz max -> none: +0 +0
                f8e4m3fnuz min -> none: +0 +0
                f8e4m3fnuz none -> sum: +1 +1
                f8e4m3fnuz none -> max: +0 +0
                f8e4m3fnuz max -> sum: +0 +0
                f8e4m3fnuz sum -> sum split: +1 +1
                f8e4m3fnuz max -> max split: +1 +1
                f8e8m0fnu sum -> none: +0 +0
                f8e8m0fnu max -> none: +0 +0
                f8e8m0fnu min -> none: +0 +0
                f8e8m0fnu none -> sum: error: {missing_identity}
                f8e8m0fnu none -> max: +0 +0
                f8e8m0fnu max -> sum: error: {missing_identity}
                f8e8m0fnu sum -> sum split: +1 +1
                f8e8m0fnu max -> max split: +1 +1
                c64 sum -> none: +1 +1
                c64 max -> none: +0 +0
                c64 min -> none: +0 +0
                c64 none -> sum: +0 +0
                c64 none -> max: +0 +0
                c64 max -> sum: +0 +0
                c64 sum -> sum split: +1 +1
                c64 max -> max split: +1 +1"},
        );

        // The two differences that the contract of floating point sums allows, pinned on the CPU plugin with the bit
        // patterns of `f32` values. Completing a sum of two negative zeros on the devices gives positive zero, because
        // the collective starts from positive zero, and the plugin flushes a subnormal sum to zero, as it does in every
        // addition. Introducing a sum writes positive zero to the other device, so a negative zero reads back as a
        // positive one. Both placements are compiled reshards.
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
        assert_eq!(session.cached_program_count(), 2);
        assert_eq!(
            [bits(&summands), bits(&completed), bits(&value), bits(&introduced)],
            [
                vec![0x8000_0000, 1, 0x4040_0000],
                vec![0, 0, 0x4040_0000],
                vec![0x8000_0000, 1, 0x3f80_0000],
                vec![0, 1, 0x3f80_0000],
            ],
        );
    }

    #[test]
    fn test_array_to_placement_completes_pending_sums_in_the_order_and_precision_of_readback() {
        // Four contributions whose sum depends on the order and the precision of accumulation, rotated so that every
        // element sees another order (element `e` of device `d` holds `contributions[(d + e) % 4]`). The collective of
        // the CPU plugin accumulates in ascending order of the coordinates, `bf16` sums in `f32`, and the sums of the
        // eight-bit formats in `f16`, which the readback reproduces, so a readback and a placement that completes the
        // sum agree on both axis types. Before, the readback added in the element type, which gave `[1, 2, 0, 0]` for
        // the `bf16` and `f8e4m3fn` rows whose placement gives `2` everywhere.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let rows = [
            (DataType::BF16, 256.0, [2.0, 2.0, 2.0, 2.0]),
            (DataType::BF16, 16777216.0, [1.0, 2.0, 2.0, 2.0]),
            (DataType::F32, 16777216.0, [1.0, 2.0, 2.0, 2.0]),
            (DataType::F16, 2048.0, [1.0, 2.0, 2.0, 2.0]),
            (DataType::F8E4M3FN, 256.0, [2.0, 2.0, 2.0, 2.0]),
            (DataType::F8E5M2, 57344.0, [1.0, 2.0, 0.0, 0.0]),
        ];
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Auto] {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, axis_type).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
            let pending = Sharding::replicated(logical_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
            let replicated = Placement::new(mesh.clone(), Sharding::replicated(logical_mesh, 1)).unwrap();
            for (data_type, power, expected) in rows {
                let r#type = ArrayType::new_static(data_type, [4]);
                let contributions = [1.0f64, power, -power, 1.0];
                let encode = |value: f64| {
                    CoreArray::scalar(value).unwrap().convert_element_type(data_type).unwrap().logical_bytes()
                };
                let shards = (0..4).map(|device| {
                    let bytes = (0..4).flat_map(|element| encode(contributions[(device + element) % 4]));
                    (devices[device].id(), Arc::<[u8]>::from(bytes.collect::<Vec<_>>()))
                });
                let decode = |bytes: Vec<u8>| {
                    let array = CoreArray::new(r#type.clone(), bytes).unwrap();
                    array.convert_element_type(DataType::F64).unwrap().elements::<f64>().unwrap()
                };
                let source =
                    Array::from_host_shards(&session, r#type.clone(), mesh.clone(), Some(pending.clone()), shards)
                        .unwrap();
                let readback = decode(source.copy_to_host_bytes().unwrap());
                let placed = decode(source.to_placement(&session, replicated.clone()).unwrap().copy_to_host_bytes().unwrap());
                assert_eq!((data_type, axis_type, readback, placed), (data_type, axis_type, expected.to_vec(), expected.to_vec()));
            }
        }
    }

    #[test]
    fn test_array_to_placement_splits_boolean_values_into_unreduced_summands() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let r#type = ArrayType::new_static(DataType::Boolean, [2]);
        let value = Array::from_host_buffer(&session, r#type, mesh.clone(), Some(replicated), [1u8, 0]).unwrap();

        // Boolean summands combine with logical or, so the device away from coordinate zero along `x` holds `false`.
        let summands = value.to_placement(&session, Placement::new(mesh, unreduced.clone()).unwrap()).unwrap();
        assert_eq!(summands.sharding(), &unreduced);
        assert_eq!(
            summands.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>(),
            vec![vec![1, 0], vec![0, 0]],
        );
        assert_eq!(summands.copy_to_host_bytes().unwrap(), vec![1, 0]);
    }

    #[test]
    fn test_array_to_placement_preserves_host_memory() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = four_device_mesh_x(client);
        let single_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![mesh.devices()[0]],
        )
        .unwrap();
        let sub_mesh = two_device_sub_mesh_x(client);
        let manual_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            mesh.devices().to_vec(),
        )
        .unwrap();
        let replicated = |mesh: &DeviceMesh| Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = |mesh: &DeviceMesh| {
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()
        };
        let values = [1f32, 2., 3., 4.];
        let tiles = vec![vec![1f32], vec![2.], vec![3.], vec![4.]];
        for (memory, kind) in
            [(Memory::Host { pinned: true }, "pinned_host"), (Memory::Host { pinned: false }, "unpinned_host")]
        {
            let upload = |mesh: &DeviceMesh, sharding: Sharding| {
                let r#type = ArrayType::new_static(DataType::F32, [4]).with_memory(memory);
                Array::from_host_buffer(&session, r#type, mesh.clone(), Some(sharding), values_to_bytes(&values))
                    .unwrap()
            };
            let place = |source: &Array, mesh: &DeviceMesh, sharding: Sharding| {
                source.to_placement(&session, Placement::new(mesh.clone(), sharding).unwrap()).unwrap()
            };

            // Whole shards that move to other devices are copied to the same memory of those devices, while the
            // shard that stays on its device shares its buffer. No program is compiled for them.
            let programs = session.cached_program_count();
            let source = upload(&single_device_mesh, replicated(&single_device_mesh));
            let placed = place(&source, &mesh, replicated(&mesh));
            assert_eq!(session.cached_program_count(), programs);
            assert_eq!(placed.r#type().memory(), memory);
            assert_eq!(shard_memory_kinds(&placed), vec![kind; 4]);
            assert_eq!(shard_values(&placed), vec![values.to_vec(); 4]);

            // The compiled reshard on one mesh declares its input and its output in the memory of the source, with
            // and without donating the source.
            let source = upload(&mesh, replicated(&mesh));
            let placed = place(&source, &mesh, sharded(&mesh));
            assert_eq!(placed.r#type().memory(), memory);
            assert_eq!(shard_memory_kinds(&placed), vec![kind; 4]);
            assert_eq!(shard_values(&placed), tiles);
            let placed =
                source.into_placement(&session, Placement::new(mesh.clone(), sharded(&mesh)).unwrap()).unwrap();
            assert_eq!(placed.r#type().memory(), memory);
            assert_eq!(shard_memory_kinds(&placed), vec![kind; 4]);
            assert_eq!(shard_values(&placed), tiles);

            // A replicated source on another mesh is copied to the memory of the source on the destination devices
            // before it is resharded there.
            let source = upload(&single_device_mesh, replicated(&single_device_mesh));
            let placed = place(&source, &mesh, sharded(&mesh));
            assert_eq!(placed.r#type().memory(), memory);
            assert_eq!(shard_memory_kinds(&placed), vec![kind; 4]);
            assert_eq!(shard_values(&placed), tiles);

            // A sharded source on another mesh is first gathered on its own mesh, which also keeps its memory.
            let source = upload(&sub_mesh, sharded(&sub_mesh));
            let placed = place(&source, &mesh, sharded(&mesh));
            assert_eq!(placed.r#type().memory(), memory);
            assert_eq!(shard_memory_kinds(&placed), vec![kind; 4]);
            assert_eq!(shard_values(&placed), tiles);

            // A manual placement of a source in host memory uploads to its original memory tier.
            let source = upload(&manual_mesh, replicated(&manual_mesh));
            let placed = place(&source, &manual_mesh, sharded(&manual_mesh));
            assert_eq!(placed.r#type().memory(), memory);
            assert_eq!(shard_memory_kinds(&placed), vec![kind; 4]);
            assert_eq!(shard_values(&placed), tiles);
        }
    }

    #[test]
    fn test_array_to_placement_changes_memory_and_layout() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let sharded = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let memory = Memory::Host { pinned: true };
        let values = [1f32, 2., 3., 4., 5., 6., 7., 8.];
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4, 2]).with_layout(row_major.clone()),
            mesh.clone(),
            Some(replicated.clone()),
            values_to_bytes(&values),
        )
        .unwrap();

        // Changing just the memory on the same mesh keeps the source layout and needs no reshard program.
        let programs = session.cached_program_count();
        let copied = input
            .to_placement(&session, Placement::new(mesh.clone(), replicated).unwrap().with_memory(memory))
            .unwrap();
        assert_eq!(session.cached_program_count(), programs);
        assert_eq!(copied.r#type().memory(), memory);
        assert_eq!(copied.layout(), Some(&row_major));
        assert_eq!(shard_memory_kinds(&copied), ["pinned_host"; 2]);
        assert_eq!(copied.copy_to_host().unwrap().elements::<f32>().unwrap(), values);

        // The storage request also applies after a compiled reshard changes the slices.
        let placed = input
            .into_placement(
                &session,
                Placement::new(mesh.clone(), sharded.clone())
                    .unwrap()
                    .with_layout(column_major.clone())
                    .with_memory(memory),
            )
            .unwrap();
        assert_eq!(placed.sharding(), &sharded);
        assert_eq!(placed.r#type().memory(), memory);
        assert_eq!(placed.layout(), Some(&column_major));
        assert_eq!(shard_memory_kinds(&placed), ["pinned_host"; 2]);
        for shard in placed.addressable_shards() {
            assert_eq!(shard.buffer().unwrap().specification().unwrap().layout, column_major.to_pjrt().unwrap());
        }
        assert_eq!(placed.copy_to_host().unwrap().elements::<f32>().unwrap(), values);

        // Whole shards can move directly to the requested memory on other devices.
        let reversed =
            DeviceMesh::new(mesh.logical_mesh().clone(), mesh.devices().iter().rev().copied().collect()).unwrap();
        let moved = placed
            .to_placement(
                &session,
                Placement::new(reversed.clone(), sharded.clone()).unwrap().with_memory(Memory::Device),
            )
            .unwrap();
        assert_eq!(moved.mesh(), &reversed);
        assert_eq!(moved.r#type().memory(), Memory::Device);
        for shard in moved.addressable_shards() {
            let buffer = shard.buffer().unwrap();
            assert_eq!(buffer.memory().unwrap(), buffer.device().unwrap().default_memory().unwrap());
        }
        assert_eq!(moved.copy_to_host().unwrap().elements::<f32>().unwrap(), values);

        // When crossing meshes also changes the slices, the compiled reshard can use device intermediates.
        let unpinned = Memory::Host { pinned: false };
        let moved = copied
            .into_placement(&session, Placement::new(reversed.clone(), sharded).unwrap().with_memory(unpinned))
            .unwrap();
        assert_eq!(moved.mesh(), &reversed);
        assert_eq!(moved.r#type().memory(), unpinned);
        assert_eq!(shard_memory_kinds(&moved), ["unpinned_host"; 2]);
        assert_eq!(moved.copy_to_host().unwrap().elements::<f32>().unwrap(), values);
    }

    #[test]
    fn test_array_to_placement_changes_memory_fails_readback_of_failed_replica() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let other_mesh = domain_mesh(client, "y", 2);

        // A move to pinned host memory copies each replica on its device, as a change of storage on the same mesh and
        // as a copy of whole shards to another mesh of the same devices. Only the copy of the second replica fails,
        // and a readback reads one replica of each tile, so the result must report the failure through its fence.
        for target_mesh in [mesh.clone(), other_mesh] {
            let (source, managers) = pending_replicated_vector(&session, &mesh);
            let replicated = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
            let target = Placement::new(target_mesh, replicated).unwrap().with_memory(Memory::Host { pinned: true });
            let placed = source.to_placement(&session, target).unwrap();
            assert_eq!(shard_memory_kinds(&placed), ["pinned_host"; 2]);
            managers[0]
                .transfer_data(0, Arc::new(values_to_bytes(&[1f32, 2., 3., 4.])), 0, true)
                .unwrap()
                .r#await()
                .unwrap();
            managers[1].set_error(0, PjrtError::aborted("second replica failed")).unwrap();
            assert!(matches!(
                placed.block_until_ready(),
                Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second replica failed",
            ));
            assert!(matches!(
                placed.copy_to_host_bytes(),
                Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second replica failed",
            ));
            assert!(matches!(
                session.synchronize(),
                Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                    if matches!(&failures[..], [(_, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                        if message == "second replica failed"),
            ));
        }
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_across_meshes_fails_readback_of_failed_replica() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let other_mesh = domain_mesh(client, "y", 2);

        // The source is in pinned host memory in a declared column-major layout, which rules out shard regions, and its
        // second replica comes from a copy that fails. A compiled reshard to a split over another mesh of the
        // same devices copies each replica to the device memory of its own device and reads each copy on its device,
        // so the failed replica fails both its copy and the program that reads that copy. The fence of the result reports the first failure that it observes,
        // which is either of the two, so the readiness and the readback of the result both fail with the cause of the
        // failed replica, as the copy reports it or as the program reports it. A failed replica never yields a result
        // that is ready or a readback that succeeds.
        let failure = |result: Result<(), Error>| match result {
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) => Some(("aborted", message)),
            Err(Error::Pjrt(PjrtError::Internal { message, .. })) => Some(("internal", message)),
            _ => None,
        };
        let replica_failures = [
            Some(("aborted", "second replica failed".to_string())),
            Some(("internal", "Error dispatching computation: second replica failed".to_string())),
        ];
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let managers = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| {
                let layout = column_major.to_pjrt().unwrap();
                let specification = BufferSpecification::with_layout(BufferType::F32, [2u64, 2], layout);
                client
                    .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let buffers = managers.iter().map(|manager| manager.retrieve_buffer(0).unwrap()).collect();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let r#type = ArrayType::new_static(DataType::F32, [2, 2]).with_layout(column_major);
        let source = Array::from_addressable_buffers(&session, r#type, mesh.clone(), Some(replicated.clone()), buffers);
        let pinned = Placement::new(mesh, replicated).unwrap().with_memory(Memory::Host { pinned: true });
        let source = source.unwrap().to_placement(&session, pinned).unwrap();
        let sharded = Sharding::new(
            other_mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["y"]), ShardingDimension::replicated()],
        )
        .unwrap();
        // The compiled reshard keeps the column-major layout of its input, and the result is requested in the row-major
        // layout, so a conversion on each device reads the result of the reshard.
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let target = Placement::new(other_mesh.clone(), sharded.clone())
            .unwrap()
            .with_memory(Memory::Device)
            .with_layout(row_major);
        let placed = source.to_placement(&session, target).unwrap();
        assert_eq!(placed.mesh(), &other_mesh);
        assert_eq!(placed.sharding(), &sharded);
        assert_eq!(shard_memory_kinds(&placed), ["device"; 2]);
        assert_eq!(session.cached_program_count(), 2);
        managers[0]
            .transfer_data(0, Arc::new(values_to_bytes(&[1f32, 2., 3., 4.])), 0, true)
            .unwrap()
            .r#await()
            .unwrap();
        managers[1].set_error(0, PjrtError::aborted("second replica failed")).unwrap();
        assert!(matches!(
            source.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second replica failed",
        ));
        let ready = failure(placed.block_until_ready());
        assert!(replica_failures.contains(&ready), "unexpected readiness of the result: {ready:?}");
        let readback = failure(placed.copy_to_host_bytes().map(|_| ()));
        assert!(replica_failures.contains(&readback), "unexpected readback of the result: {readback:?}");

        // The session also reports the failed replica, once for each failed submission in registration order. The
        // copies that read the failed replica report its cause as the copy reports it, the execution of the reshard
        // that reads those copies reports it as the failed execution reports it, and the conversions that read the
        // failed result of the reshard wrap that report once more.
        let Err(Error::Execution(ExecutionError::SynchronizationFailures { failures })) = session.synchronize() else {
            panic!("expected the session to report the failed replica");
        };
        let failures = failures
            .into_iter()
            .map(|(identity, error)| (identity, failure(Err(error)).unwrap()))
            .collect::<Vec<_>>();
        let message = "Error dispatching computation: Error dispatching computation: second replica failed";
        let converted_failure = ("internal", message.to_string());
        assert_eq!(
            failures,
            [
                (0, replica_failures[0].clone().unwrap()),
                (1, replica_failures[0].clone().unwrap()),
                (2, replica_failures[1].clone().unwrap()),
                (3, converted_failure.clone()),
                (5, converted_failure),
            ],
        );
    }

    #[test]
    fn test_array_to_placement_relayouts_addressable_unreduced_shards() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let shared_session = peer_exchange_session(client);
        let devices = client.addressable_devices().unwrap();
        let mut mesh_devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let remote_device =
            Device::new(mesh_devices.iter().map(Device::id).max().unwrap() + 1, client.process_index().unwrap() + 1);
        mesh_devices.push(remote_device);
        let logical_mesh =
            LogicalMesh::new(vec![MeshAxis::new("x", mesh_devices.len(), MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), mesh_devices).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 2).with_unreduced_axes(["x"]).unwrap();
        let summands = [[1f32, 2., 3., 4.], [10f32, 20., 30., 40.]];
        let buffers = summands
            .iter()
            .zip(devices)
            .map(|(values, device)| {
                client
                    .buffer(values_to_bytes(values).as_slice(), BufferType::F32, [2, 2], None, device, None)
                    .unwrap()
            })
            .collect();
        let (producer_event, producer) = client.event(()).unwrap();
        let (send_event, send) = client.event(()).unwrap();
        let source = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::F32, [2, 2]),
            mesh.clone(),
            Some(sharding.clone()),
            buffers,
        )
        .unwrap()
        .with_execution_fence(ExecutionFence::new(vec![producer_event]))
        .with_send_fence(ExecutionFence::new(vec![send_event]));
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let target = Placement::new(mesh, sharding.clone()).unwrap().with_layout(layout.clone());
        let placed = source.to_placement(&session, target.clone()).unwrap();
        // A session that shares compilation artifacts leads the conversions of the shards of its own process, so it
        // converts them on their devices as well.
        let shared = source.to_placement(&shared_session, target).unwrap();
        assert!(session.cached_program_count() > 0);
        assert!(shared_session.cached_program_count() > 0);

        for output in [&placed, &shared] {
            assert_eq!(output.sharding(), &sharding);
            assert_eq!(output.layout(), Some(&layout));
            assert!(output.device_shard(remote_device.id()).unwrap().buffer().is_none());
            assert!(!output.execution_fence().unwrap().is_ready().unwrap());
            for (shard, summand) in output.addressable_shards().zip(summands) {
                assert_eq!(shard.buffer().unwrap().specification().unwrap().layout, layout.to_pjrt().unwrap());
                assert_eq!(values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()), summand);
            }
        }

        // Every placement path retains the pending producer and the source's asynchronous sends.
        producer.set(None).unwrap();
        send.set(Some(PjrtError::aborted("source transfer failed"))).unwrap();
        for output in [placed, shared] {
            assert_eq!(output.block_until_ready().unwrap_err().to_string(), "source transfer failed");
        }
    }

    #[test]
    fn test_array_to_placement_zero_space_is_bufferless_and_preserves_type_metadata() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = four_device_mesh_x(client);
        let source_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let layout = Layout::Tiled(TiledLayout::new(vec![0], Vec::new()));
        let source_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(4)]))
            .with_layout(layout.clone())
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(source_sharding)
            .unwrap();
        let source = Array::from_host_buffer(&session, source_type, mesh.clone(), None, []).unwrap();
        let target_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let resharded = source
            .to_placement(&session, Placement::new(mesh.clone(), target_sharding.clone()).unwrap())
            .unwrap();

        assert_eq!(resharded.sharding(), &target_sharding);
        assert_eq!(resharded.layout(), Some(&layout));
        assert_eq!(resharded.r#type().memory(), Memory::Host { pinned: true });
        assert_eq!(resharded.addressable_shards().count(), mesh.devices().len());
        assert!(resharded.addressable_shards().all(|shard| shard.buffer().is_none()));
    }

    #[test]
    fn test_array_to_placement_zero_space_rejects_invalid_layout() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let source =
            Array::from_host_buffer(&session, ArrayType::new_static(DataType::Zero, [4]), mesh.clone(), None, [])
                .unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()));
        let message = "`place` target layout is invalid: tiled layout rank 2 does not match array rank 1";

        // A rank-1 array has no buffers whose layout could be checked, so the layout is validated against its shape,
        // both when the placement keeps its devices and sharding and when it changes them.
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        for sharding in [source.sharding().clone(), sharded] {
            let target = Placement::new(mesh.clone(), sharding).unwrap().with_layout(layout.clone());
            let error = source.to_placement(&session, target).unwrap_err();
            assert!(matches!(
                &error,
                Error::Core(error) if **error == CoreError::Type(TypeError::Invalid { message: message.to_string() }),
            ));
            assert_eq!(error.to_string(), message);
        }

        // A layout of the right rank is recorded on the bufferless result.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], Vec::new()));
        let target = Placement::new(mesh.clone(), source.sharding().clone()).unwrap().with_layout(layout.clone());
        let placed = source.to_placement(&session, target).unwrap();
        assert_eq!(placed.layout(), Some(&layout));
        assert!(placed.addressable_shards().all(|shard| shard.buffer().is_none()));
    }

    #[test]
    fn test_array_to_placement_validates_layout_of_each_shard() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_sub_mesh_x(client);
        let columns = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4, 4]);
        let values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let source = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&values)).unwrap();
        let strided = |strides: Vec<isize>| Layout::Strided(StridedLayout::new(strides));

        // A target layout describes each destination shard, so the strides `[8, 4]` of a dense `f32[4, 2]` column
        // shard pass the validation for the column shards of a global `[4, 4]` value, which they would overlap. The CPU
        // plugin cannot hold buffers in strided layouts, so only its device work then fails.
        let target = Placement::new(mesh.clone(), columns.clone()).unwrap().with_layout(strided(vec![8, 4]));
        assert_eq!(
            source.to_placement(&session, target).unwrap_err().to_string(),
            "PJRT_Buffer_MemoryLayout_Type_Strides is not supported to be converted to a xla::Layout.",
        );

        // The same strides overlap the rows of a replicated shard, and strides that overlap a column shard are
        // rejected.
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        for (sharding, strides, axis) in [(replicated, vec![8, 4], 0), (columns, vec![4, 8], 1)] {
            let target = Placement::new(mesh.clone(), sharding).unwrap().with_layout(strided(strides));
            let error = source.to_placement(&session, target).unwrap_err();
            assert_eq!(
                error.to_string(),
                format!(
                    "`{PLACE_OPERATION_NAME}` target layout is invalid: strided layout stride 8 on axis {axis} is \
                     smaller than the 16-byte span occupied by more minor axes and may alias array elements",
                ),
            );
        }
    }

    #[test]
    fn test_array_to_placement_zero_space_preserves_producer_completion() {
        let client = crate::tests::execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap().remove(0);
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&device).unwrap()],
        )
        .unwrap();
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("target", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            source_mesh.devices().to_vec(),
        )
        .unwrap();
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
        let target = Placement::new(target_mesh.clone(), target_sharding.clone()).unwrap();

        for fails in [false, true] {
            let (event, promise) = client.event(()).unwrap();
            let source = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::Zero, [2]),
                source_mesh.clone(),
                None,
                [],
            )
            .unwrap()
            .with_execution_fence(ExecutionFence::new(vec![event]));
            let outputs = [
                source.to_placement(&session, target.clone()).unwrap(),
                source.clone().into_placement(&session, target.clone()).unwrap(),
                source.clone().reshard_on_devices(&session, &target_mesh, &target_sharding, false).unwrap(),
            ];
            // A move that only changes the representation must neither wait for nor discard the producer's completion.
            // Complete the event before asserting so that even a failing regression releases its pending native
            // callbacks.
            let pending = outputs
                .iter()
                .map(|output| matches!(output.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))))
                .collect::<Vec<_>>();
            promise.set(fails.then(|| PjrtError::invalid_argument("zero producer failed"))).unwrap();
            assert_eq!(pending, vec![true; 3]);
            for output in outputs {
                let result = output.block_until_ready();
                if fails {
                    assert!(matches!(
                        result,
                        Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))
                            if message == "zero producer failed",
                    ));
                } else {
                    assert_eq!(result.map_err(|error| error.to_string()), Ok(()));
                }
            }
        }
    }

    #[test]
    fn test_array_to_placement_exact_shards_replicated_cross_mesh_to_replicated_destination() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();

        // The source is replicated on a 1-device submesh.
        let source_device_id = client_devices[0].id().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(source_device_id, client_devices[0].process_index().unwrap())],
        )
        .unwrap();
        let values = [60.0f32, 61.0, 62.0, 63.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            source_mesh,
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // The target is replicated on the full 4-device mesh. The exact shard path matches every destination shard to
        // the single source shard, which covers the full array. The destination shard on the source device shares the
        // source buffer, `copy_to_memory` produces independent buffers on the other devices, and `copy_to_host`
        // works on every one of them.
        let target_mesh = four_device_mesh_x(client);
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
        let resharded = source_array
            .to_placement(&session, Placement::new(target_mesh.clone(), target_sharding).unwrap())
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        assert_eq!(shard_values(&resharded), vec![values.to_vec(); 4]);
        // The exact shard path satisfies this request without compiling any SPMD program.
        assert_eq!(session.cached_program_count(), 0);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_replicated_to_sharded_on_same_mesh() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = four_device_mesh_x(client);

        let values = [10.0f32, 11.0, 12.0, 13.0];
        let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        let sharded_target =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded =
            source_array.to_placement(&session, Placement::new(mesh.clone(), sharded_target).unwrap()).unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        assert_eq!(shard_values(&resharded), vec![vec![10.0], vec![11.0], vec![12.0], vec![13.0]]);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_on_same_mesh_preserves_producer_failure() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();

        // Both replicas of the source are readable, but the execution that produced it fails afterwards (e.g., a copy
        // of a replica that an earlier placement made and joined into its fence). The compiled reshard reads only the
        // readable buffers, so its result must carry the fence of the source across the launch, with and without
        // donating the source.
        for donate in [false, true] {
            let (producer_event, producer) = client.event(()).unwrap();
            let source = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, [4]),
                mesh.clone(),
                Some(replicated.clone()),
                values_to_bytes(&[1f32, 2., 3., 4.]),
            )
            .unwrap()
            .with_execution_fence(ExecutionFence::new(vec![producer_event]));
            source
                .addressable_shards()
                .for_each(|shard| shard.buffer().unwrap().ready().unwrap().r#await().unwrap());
            let programs = session.cached_program_count();
            let placed = if donate {
                source.into_placement(&session, target.clone()).unwrap()
            } else {
                source.to_placement(&session, target.clone()).unwrap()
            };
            assert_eq!(session.cached_program_count(), programs + 1);
            assert_eq!(placed.mesh(), &mesh);
            assert_eq!(placed.sharding(), &sharded);
            assert_eq!(
                placed
                    .addressable_shards()
                    .map(|shard| shard.buffer().unwrap().dimensions().unwrap().to_vec())
                    .collect::<Vec<_>>(),
                [[2], [2]],
            );
            assert_eq!(shard_values(&placed), vec![vec![1., 2.], vec![3., 4.]]);
            assert!(matches!(placed.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))));
            producer.set(Some(PjrtError::aborted("replica copy failed"))).unwrap();
            assert!(matches!(
                placed.block_until_ready(),
                Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "replica copy failed",
            ));
            assert!(matches!(
                placed.copy_to_host_bytes(),
                Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "replica copy failed",
            ));
        }
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_sharded_to_replicated_on_same_mesh() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = four_device_mesh_x(client);

        let values = [20.0f32, 21.0, 22.0, 23.0];
        let sharded_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]));
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            mesh.clone(),
            Some(sharded_sharding),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        let replicated_target = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let resharded = source_array
            .to_placement(&session, Placement::new(mesh.clone(), replicated_target).unwrap())
            .unwrap();

        assert_eq!(shard_values(&resharded), vec![values.to_vec(); 4]);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_sharded_to_differently_sharded_on_same_mesh() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh_2x2(), devices).unwrap();

        let row_values = [
            100.0f32, 101.0, 102.0, 103.0, 110.0, 111.0, 112.0, 113.0, 120.0, 121.0, 122.0, 123.0, 130.0, 131.0, 132.0,
            133.0,
        ];

        let sharded_along_x = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(4)]));
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            mesh.clone(),
            Some(sharded_along_x),
            values_to_bytes::<f32>(&row_values).as_slice(),
        )
        .unwrap();

        let sharded_along_y = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let resharded =
            source_array.to_placement(&session, Placement::new(mesh.clone(), sharded_along_y).unwrap()).unwrap();
        assert_eq!(resharded.addressable_shards().count(), 4);

        // Every device holds two whole columns, and the devices along `x` hold copies of the same columns.
        assert_eq!(
            shard_values(&resharded),
            vec![
                vec![100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 130.0, 131.0],
                vec![102.0, 103.0, 112.0, 113.0, 122.0, 123.0, 132.0, 133.0],
                vec![100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 130.0, 131.0],
                vec![102.0, 103.0, 112.0, 113.0, 122.0, 123.0, 132.0, 133.0],
            ],
        );
    }

    #[test]
    fn test_array_compile_reshard_on_same_mesh_keeps_reduction_state() {
        // A reshard that keeps a pending reduction along the axis `x` of a 2x2 mesh whose axes are explicit or auto
        // moves the contributions as they are (over an auto axis, because the program types that axis as explicit), so
        // its optimized program has no all-reduce (only the all-to-all that a change of split needs), while one that
        // completes the reduction reduces it. A kept pending sum of complex values, whose introduction crashes XLA,
        // therefore never reaches a collective.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
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
            for (name, shape, input, output) in [
                (
                    "keep sum",
                    vec![4],
                    unreduced(&replicated, UnreducedKind::Sum),
                    unreduced(&split_y, UnreducedKind::Sum),
                ),
                (
                    "keep sum, unconstrained",
                    vec![4],
                    unreduced(&replicated, UnreducedKind::Sum),
                    unreduced(&unconstrained, UnreducedKind::Sum),
                ),
                (
                    "keep sum, c64",
                    vec![4],
                    unreduced(&replicated, UnreducedKind::Sum),
                    unreduced(&split_y, UnreducedKind::Sum),
                ),
                (
                    "keep max",
                    vec![4],
                    unreduced(&replicated, UnreducedKind::Max),
                    unreduced(&split_y, UnreducedKind::Max),
                ),
                (
                    "keep sum, columns to rows",
                    vec![4, 4],
                    unreduced(&columns, UnreducedKind::Sum),
                    unreduced(&rows, UnreducedKind::Sum),
                ),
                ("complete sum", vec![4], unreduced(&replicated, UnreducedKind::Sum), split_y.clone()),
            ] {
                let shard_size =
                    ShardLayout::new(&StaticShape::new(shape.clone()), &mesh, &input).unwrap().descriptors()[0]
                        .shape()
                        .as_slice()
                        .iter()
                        .product::<usize>();
                // A `c64` element holds two `f32` components.
                let (data_type, components) =
                    if name.ends_with("c64") { (DataType::C64, 2) } else { (DataType::F32, 1) };
                let zeros = values_to_bytes(&vec![0f32; shard_size * components]);
                let contributions =
                    devices.iter().map(|device| (device.id(), Arc::<[u8]>::from(zeros.as_slice()))).collect::<Vec<_>>();
                let r#type = ArrayType::new_static(data_type, shape);
                let source =
                    Array::from_host_shards(&session, r#type, mesh.clone(), Some(input.clone()), contributions)
                        .unwrap();
                let program = ReshardProgram { mesh: mesh.clone(), input, output };
                let compiled = source.compile_reshard_on_same_mesh(&program, false).unwrap().unwrap();
                // The cache returns the program that the reshard compiled, because its staged function lowers alike.
                let lowered = compiled.staged().into_inner().lower(&session).unwrap();
                let optimized = session.compilation_cache().compile(&session, lowered.lowered_program()).unwrap();
                let optimized = optimized.optimized_program().unwrap();
                let names =
                    |opcode: &str| optimized.bytes().windows(opcode.len()).any(|window| window == opcode.as_bytes());
                let present = collectives.iter().filter(|opcode| names(opcode)).copied().collect::<Vec<_>>();
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
    fn test_array_to_placement_compiled_reshard_with_explicit_mesh_axes() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        // Build a 4-device mesh whose axis type is `Explicit`, for which the user declares sharding propagation rather
        // than having it inferred. The compiled reshard should accept this axis type just like `Auto`.
        let explicit_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();

        let values = [400.0f32, 401.0, 402.0, 403.0];
        let replicated_sharding = Sharding::replicated(explicit_mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            explicit_mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        let sharded_target =
            Sharding::new(explicit_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded =
            source_array.to_placement(&session, Placement::new(explicit_mesh, sharded_target).unwrap()).unwrap();
        assert_eq!(resharded.addressable_shards().count(), 4);
        assert_eq!(shard_values(&resharded), vec![vec![400.0], vec![401.0], vec![402.0], vec![403.0]]);
    }

    #[test]
    fn test_array_to_placement_cross_mesh_replicated_source_to_sharded_destination() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();

        // The source lives on a 1-device submesh.
        let source_device_id = client_devices[0].id().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(source_device_id, client_devices[0].process_index().unwrap())],
        )
        .unwrap();
        let values = [10.0f32, 11.0, 12.0, 13.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            source_mesh,
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // Reshard onto the full 4-device mesh, sharded along "x".
        let target_mesh = four_device_mesh_x(client);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&session, Placement::new(target_mesh.clone(), target_sharding).unwrap())
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        assert_eq!(shard_values(&resharded), vec![vec![10.0], vec![11.0], vec![12.0], vec![13.0]]);
    }

    #[test]
    fn test_array_to_placement_cross_mesh_sharded_source_to_replicated_destination() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();

        // The source is sharded along "x" on a 2-device submesh, so devices 0 and 1 each hold half the data.
        let source_mesh = two_device_sub_mesh_x(client);
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let values = [70.0f32, 71.0, 72.0, 73.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]));
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            source_mesh,
            Some(source_sharding),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // The destination is replicated on the full 4-device mesh. The sharded source is first all-gathered on the
        // source mesh and then copied onto the destination mesh, and no further reshard is needed because the
        // intermediate sharding matches the requested replicated destination sharding.
        let target_mesh = four_device_mesh_x(client);
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
        let resharded = source_array
            .to_placement(&session, Placement::new(target_mesh.clone(), target_sharding).unwrap())
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        assert_eq!(shard_values(&resharded), vec![values.to_vec(); 4]);
    }

    #[test]
    fn test_array_to_placement_cross_mesh_sharded_source_to_sharded_destination() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();

        // The source is sharded along "x" on a 2-device submesh.
        let source_mesh = two_device_sub_mesh_x(client);
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let values = [80.0f32, 81.0, 82.0, 83.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]));
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            source_mesh,
            Some(source_sharding),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // The destination is sharded along "x" on the full 4-device mesh, so each destination shard holds exactly one
        // element of the global array.
        let target_mesh = four_device_mesh_x(client);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&session, Placement::new(target_mesh.clone(), target_sharding).unwrap())
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        assert_eq!(shard_values(&resharded), vec![vec![80.0], vec![81.0], vec![82.0], vec![83.0]]);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_on_mesh_of_other_processes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let local = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let next_id = local.iter().map(Device::id).max().unwrap() + 1;
        let remote = [Device::new(next_id, 1), Device::new(next_id + 1, 1)];
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
            let destination =
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
                &destination([local[0], local[1]]),
                sharding,
                process_index,
            );
            let strategy = match strategy {
                Ok(DevicePlacementStrategy::CompiledReshard { .. }) => "compiled reshard",
                Ok(DevicePlacementStrategy::CompiledReshardAcrossMeshes { .. }) => "compiled reshard across meshes",
                _ => "another strategy",
            };
            assert_eq!(
                strategy,
                if reversed { "compiled reshard across meshes" } else { "compiled reshard" },
                "{case}"
            );
            let target = Placement::new(destination([local[0], local[1]]), sharding.clone()).unwrap();
            let participant = source.to_placement(&session, target).unwrap();
            let participant_moved = source
                .clone()
                .reshard_on_devices(&session, &destination([local[0], local[1]]), sharding, false)
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
            let target = Placement::new(destination(remote), sharding.clone()).unwrap();
            let placed = peer_source.to_placement(&session, target).unwrap();
            let moved =
                peer_source.clone().reshard_on_devices(&session, &destination(remote), sharding, false).unwrap();
            for output in [&placed, &moved] {
                assert_eq!(output.mesh(), &destination(remote), "{case}");
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
    fn test_array_to_placement_compiled_reshard_shares_artifact_without_process_zero() {
        // Compilation artifact exchange of process 1 of three that records its calls and never receives an artifact.
        struct RecordingExchange(Mutex<Vec<String>>);

        impl CompilationArtifactExchange for RecordingExchange {
            fn process_index(&self) -> usize {
                1
            }

            fn process_count(&self) -> usize {
                3
            }

            fn validate_compatibility(
                &self,
                _key: &[u8],
                leader: usize,
                _timeout: Duration,
            ) -> Result<(), CompilationArtifactExchangeError> {
                self.0.lock().unwrap().push(format!("validate with leader {leader}"));
                Ok(())
            }

            fn publish(&self, _key: &[u8], _artifact: &[u8]) -> Result<(), CompilationArtifactExchangeError> {
                self.0.lock().unwrap().push("publish".to_string());
                Ok(())
            }

            fn receive(&self, _: &[u8], _: Duration) -> Result<Option<Vec<u8>>, CompilationArtifactExchangeError> {
                self.0.lock().unwrap().push("receive".to_string());
                Ok(None)
            }
        }

        // This client is process 1 of a job of three processes, and both of its devices form the mesh of the source
        // and of the destination, so it runs the compiled reshard, as process 2 would on a mesh that spans the devices
        // of both, while process 0 owns no device of that mesh and compiles nothing. Process 1 is therefore the leader
        // of the compilation, which compiles and publishes the program rather than waiting for process 0.
        let options = CpuClientOptions { device_count: Some(2), process_id: Some(1) };
        let client = load_cpu_plugin().unwrap().client(ClientOptions::CPU(options)).unwrap();
        let client: &'static ryft_pjrt::Client<'static> = Box::leak(Box::new(client));
        let exchange = Arc::new(RecordingExchange(Mutex::new(Vec::new())));
        let cache = CompilationCache::new().with_artifact_exchange(
            Arc::clone(&exchange) as Arc<dyn CompilationArtifactExchange>,
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_millis(10) },
        );
        let session = Session::builder().client(client).compilation_cache(cache).build().unwrap();
        let mesh = two_device_sub_mesh_x(client);
        assert_eq!(mesh.devices().iter().map(Device::process_index).collect::<Vec<_>>(), [1, 1]);
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let source =
            Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]));
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let placed = source.unwrap().to_placement(&session, Placement::new(mesh, sharded).unwrap()).unwrap();
        assert_eq!(shard_values(&placed), [vec![1., 2.], vec![3., 4.]]);
        assert_eq!(*exchange.0.lock().unwrap(), ["validate with leader 1", "publish"]);
    }

    #[test]
    fn test_array_to_placement_moves_only_required_cross_mesh_regions() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let source_mesh = two_device_sub_mesh_x(client);
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
        let layout = ShardLayout::new(&source.shape(), &target_mesh, &target_sharding).unwrap();
        let regions = source.plan_shard_regions(&layout).unwrap();
        // Each column needs two regions. Two of the eight regions are already on their destination devices.
        let copied_bytes = regions
            .moves()
            .iter()
            .filter(|movement| movement.source_device() != movement.destination_device())
            .map(|movement| movement.shape().as_slice().iter().product::<usize>() * size_of::<f32>())
            .sum::<usize>();
        assert_eq!(regions.moves().len(), 8);
        assert_eq!(copied_bytes, 48);

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
        assert_eq!(plan, regions);

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
    fn test_array_to_placement_compiled_reshard_caches_executable_across_calls() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = four_device_mesh_x(client);

        let values = [30.0f32, 31.0, 32.0, 33.0];
        let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        let sharded_target =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(session.cached_program_count(), 0);
        let first = source_array
            .clone()
            .to_placement(&session, Placement::new(mesh.clone(), sharded_target.clone()).unwrap())
            .unwrap();
        let after_first = session.cached_program_count();
        let second =
            source_array.to_placement(&session, Placement::new(mesh.clone(), sharded_target).unwrap()).unwrap();
        let after_second = session.cached_program_count();

        assert_eq!(first.addressable_shards().count(), 4);
        assert_eq!(second.addressable_shards().count(), 4);
        assert_eq!(after_first, 1, "first reshard should populate the executable cache");
        assert_eq!(after_second, 1, "second reshard with identical MLIR should reuse the cached executable");
    }

    // Validates that the second reshard with the same structural signature hits the cache and skips tracing and
    // lowering, because the structural cache key (i.e., the input and output shardings, mesh, shape, and data type)
    // matches without ever materializing the MLIR text. This mirrors how JAX caches `jit` invocations on abstract
    // value signatures.
    #[test]
    #[ignore = "timing-sensitive; runs locally to validate the structural cache hit"]
    fn test_array_to_placement_compiled_reshard_cache_hit_avoids_tracing_and_lowering() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = four_device_mesh_x(client);
        let values = (0..4096).map(|index| index as f32).collect::<Vec<_>>();
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let target_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let cold_start = std::time::Instant::now();
        let _ = source_array
            .clone()
            .to_placement(&session, Placement::new(mesh.clone(), target_sharding.clone()).unwrap())
            .unwrap();
        let cold_elapsed = cold_start.elapsed();

        let warm_start = std::time::Instant::now();
        let _ = source_array.to_placement(&session, Placement::new(mesh.clone(), target_sharding).unwrap()).unwrap();
        let warm_elapsed = warm_start.elapsed();

        eprintln!("cold reshard (trace+lower+compile+execute): {cold_elapsed:?}");
        eprintln!("warm reshard (structural cache hit + execute, no trace/lower): {warm_elapsed:?}");
        assert_eq!(session.cached_program_count(), 1);
    }

    #[test]
    fn test_array_to_placement_preserves_missing_cross_host_extension() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;

        // The source array is on the local device, replicated on a 1-device submesh.
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device_id, local_device.process_index().unwrap())],
        )
        .unwrap();
        let values = [100.0f32, 101.0, 102.0, 103.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            source_mesh,
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // Sending to the remote destination requires the extension even though local copies are possible.
        assert!(matches!(client.cross_host_transfers_extension(), Err(PjrtError::Unimplemented { .. })));
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
        )
        .unwrap();
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);

        let result = source_array.to_placement(&session, Placement::new(target_mesh.clone(), target_sharding).unwrap());
        assert!(matches!(
            result,
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));

        // Moving a proper region also requires the extension, even when the source device can slice it locally.
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let result = source_array.to_placement(&session, Placement::new(target_mesh, target_sharding).unwrap());
        assert!(matches!(
            result,
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));
    }

    #[test]
    fn test_array_to_placement_preserves_missing_cross_host_extension_for_receiver() {
        let client = cpu_client(1);
        assert!(matches!(client.cross_host_transfers_extension(), Err(PjrtError::Unimplemented { .. })));
        let session = Session::builder().client(client).build().unwrap();
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device.id() + 1, local_device.process_index() + 1)],
        )
        .unwrap();
        let source_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source = Array::from_addressable_buffers(&session, source_type, source_mesh, None, Vec::new()).unwrap();
        assert_eq!(source.addressable_shards().count(), 0);
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("destination", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![local_device],
        )
        .unwrap();
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);

        // This process only receives. Every destination is local, so there is no remote destination to blame for the
        // missing extension. Preserve that error instead of searching for a nonexistent remote destination and
        // panicking.
        assert!(matches!(
            source.to_placement(&session, Placement::new(target_mesh, target_sharding).unwrap()),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));
    }

    #[test]
    fn test_array_to_placement_places_uncommitted_value_on_devices_of_this_process() {
        let client = cpu_client(2);
        assert!(matches!(client.cross_host_transfers_extension(), Err(PjrtError::Unimplemented { .. })));
        let session = Session::builder().client(client).build().unwrap();
        let local_devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let remote_device =
            Device::new(local_devices.iter().map(Device::id).max().unwrap() + 1, client.process_index().unwrap() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh =
            DeviceMesh::new(logical_mesh.clone(), vec![local_devices[0], local_devices[1], remote_device]).unwrap();
        let values = [1f32, 2., 3., 4., 5., 6.];

        // Each process uploads the same value without a destination, so its array lives uncommitted on its own first
        // device. A remote process would have chosen its own device, so the source mesh is not shared, and each
        // process places its own value on its own destination devices without planning any cross-host transfer. Both
        // the borrowing and the consuming placement do so, as does the implicit reshard of a compiled input.
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        for (sharding, expected) in [
            (&replicated, vec![values.to_vec(), values.to_vec()]),
            (&sharded, vec![vec![1., 2.], vec![3., 4.]]),
            (&unreduced, vec![values.to_vec(), vec![-0.; 6]]),
        ] {
            let source = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, [6]),
                None,
                None,
                values_to_bytes(&values),
            )
            .unwrap();
            assert!(!source.is_committed());
            assert_eq!(source.mesh().devices(), [local_devices[0]]);
            let target = Placement::new(mesh.clone(), sharding.clone()).unwrap();
            let outputs = [
                source.to_placement(&session, target.clone()).unwrap(),
                source.clone().into_placement(&session, target).unwrap(),
                source.clone().reshard_on_devices(&session, &mesh, sharding, false).unwrap(),
            ];
            for output in outputs {
                assert_eq!(output.mesh(), &mesh);
                assert_eq!(output.sharding(), sharding);
                assert!(output.is_committed());
                assert_eq!(output.r#type().memory(), Memory::Device);
                assert!(output.device_shard(remote_device.id()).unwrap().buffer().is_none());
                assert_eq!(output.block_until_ready().map_err(|error| error.to_string()), Ok(()));
                assert_eq!(
                    output
                        .addressable_shards()
                        .map(|shard| shard.buffer().unwrap().dimensions().unwrap().to_vec())
                        .collect::<Vec<_>>(),
                    expected.iter().map(|values| vec![values.len() as u64]).collect::<Vec<_>>(),
                );
                let shards = output
                    .addressable_shards()
                    .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(
                    shards.iter().flatten().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    expected.iter().flatten().map(|value| value.to_bits()).collect::<Vec<_>>(),
                );
            }
        }

        // The value placed into a memory of its device is committed to it, but its device is still one that this
        // process chose on its own, so it is placed with local work only as well. The same array on a device that every
        // process knows plans a transfer to the remote device, which needs the cross-host transfers extension.
        let uploaded = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [6]),
            None,
            None,
            values_to_bytes(&values),
        )
        .unwrap();
        let target = Placement::new(mesh.clone(), replicated.clone()).unwrap();
        let in_memory = uploaded.to_placement(&session, Memory::Device).unwrap();
        assert!(in_memory.is_committed());
        let placed = in_memory.to_placement(&session, target.clone()).unwrap();
        let placed_values = placed
            .addressable_shards()
            .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(placed_values, [values.to_vec(), values.to_vec()]);
        let shared = uploaded.with_locally_assigned(false);
        assert!(matches!(
            shared.to_placement(&session, target),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));

        // The upload along an unreduced destination axis uses the layout that the placement records, which is the
        // requested one here, so it lands in that layout directly, rather than in the default layout followed by a
        // compiled conversion of every shard, and every process records that layout whichever strategy it takes.
        let session = Session::builder().client(client).build().unwrap();
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]);
        let uploaded = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&values)).unwrap();
        let unreduced = Sharding::replicated(logical_mesh, 2).with_unreduced_axes(["x"]).unwrap();
        let target = Placement::new(mesh.clone(), unreduced).unwrap().with_layout(column_major.clone());
        let placed = uploaded.to_placement(&session, target).unwrap();
        assert_eq!(placed.layout(), Some(&column_major));
        assert_eq!(placed.block_until_ready().map_err(|error| error.to_string()), Ok(()));
        assert_eq!(
            placed
                .addressable_shards()
                .map(|shard| shard.buffer().unwrap().specification().unwrap().layout)
                .collect::<Vec<_>>(),
            [column_major.to_pjrt().unwrap(), column_major.to_pjrt().unwrap()],
        );
        let shards = placed
            .addressable_shards()
            .flat_map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(shards, values.iter().chain(&[-0f32; 6]).map(|value| value.to_bits()).collect::<Vec<_>>());
        assert_eq!(session.cached_program_count(), 0);
    }

    #[test]
    fn test_array_to_placement_places_pending_uncommitted_value_on_remote_devices() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote_device = Device::new(local_device.id() + 1, local_device.process_index() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![remote_device]).unwrap();
        let unreduced = Sharding::replicated(logical_mesh, 1).with_unreduced_axes(["x"]).unwrap();
        let target = Placement::new(mesh.clone(), unreduced.clone()).unwrap();
        let pending_source = || {
            let (event, producer) = client.event(()).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            let source = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.]))
                .unwrap()
                .with_execution_fence(ExecutionFence::new(vec![event]));
            assert!(!source.is_committed());
            (source, producer)
        };

        // The value of this process goes through a host upload along the unreduced destination axis, and no
        // destination device belongs to this process. The result still waits for and reports the producer of the
        // source, which the uploads of the processes that hold the destination devices wait for.
        let (source, producer) = pending_source();
        let placed = source.to_placement(&session, target.clone()).unwrap();
        assert_eq!(placed.mesh(), &mesh);
        assert_eq!(placed.sharding(), &unreduced);
        assert!(placed.is_committed());
        assert_eq!(placed.addressable_shards().count(), 0);
        assert!(placed.shards()[0].buffer().is_none());
        assert!(matches!(placed.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))));
        producer.set(None).unwrap();
        assert_eq!(placed.block_until_ready().map_err(|error| error.to_string()), Ok(()));

        let (source, producer) = pending_source();
        let placed = source.to_placement(&session, target).unwrap();
        producer.set(Some(PjrtError::aborted("producer failed"))).unwrap();
        assert!(matches!(
            placed.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "producer failed",
        ));

        // Along a replicated destination, the value of this process is shared or copied on devices, and no destination
        // device belongs to this process. A source whose only completion is the readiness of its pending buffer (e.g.,
        // a host upload) has no execution fence, so the result waits for and reports that buffer.
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let target = Placement::new(mesh.clone(), replicated.clone()).unwrap();
        let specification = BufferSpecification::new(BufferType::F32, [2u64]);
        let devices = client.addressable_devices().unwrap();
        let manager = client.host_to_device_transfer_manager(vec![specification], devices[0].default_memory().unwrap());
        let manager = manager.unwrap();
        let source_mesh = domain_mesh(client, "x", 1);
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let buffers = vec![manager.retrieve_buffer(0).unwrap()];
        let source = Array::from_addressable_buffers(&session, r#type, source_mesh, None, buffers).unwrap();
        let source = source.with_locally_assigned(true);
        assert!(source.execution_fence().is_none());
        let placed = source.to_placement(&session, target).unwrap();
        assert_eq!(placed.sharding(), &replicated);
        assert_eq!(placed.addressable_shards().count(), 0);
        manager.set_error(0, PjrtError::aborted("upload failed")).unwrap();
        assert!(matches!(
            placed.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "upload failed",
        ));
    }

    #[test]
    fn test_array_to_placement_rejects_misattributed_destination_devices() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let (device_id, process_index) = (local_device.id(), local_device.process_index());
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let misattributed = DeviceMesh::new(logical_mesh.clone(), vec![Device::new(device_id, process_index + 1)]);
        let target = Placement::new(misattributed.unwrap(), Sharding::replicated(logical_mesh.clone(), 0)).unwrap();
        let bytes = values_to_bytes(&[7f32]);
        let source = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, bytes).unwrap();

        // The destination assigns the device of this process to another process, so every process would leave its
        // shard to another one. Both placements reject it before any device work, and the source stays readable.
        let message =
            format!("device {device_id} reports process index {process_index}, but expected {}", process_index + 1);
        for error in [
            source.to_placement(&session, target.clone()).unwrap_err(),
            source.clone().into_placement(&session, target).unwrap_err(),
        ] {
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
        assert_eq!(values_from_bytes::<f32>(&source.copy_to_host_bytes().unwrap()), [7.]);

        // A destination device of this process that the client does not address would receive no buffer either, and
        // both placements reject it as an import or an upload onto that mesh does.
        let missing = DeviceMesh::new(logical_mesh.clone(), vec![Device::new(device_id + 999, process_index)]);
        let target = Placement::new(missing.unwrap(), Sharding::replicated(logical_mesh, 0)).unwrap();
        let message =
            format!("device {} is not addressable by the PJRT client in process {process_index}", device_id + 999);
        for error in [
            source.to_placement(&session, target.clone()).unwrap_err(),
            source.clone().into_placement(&session, target).unwrap_err(),
        ] {
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
    }

    #[test]
    fn test_array_to_placement_with_manual_mesh_axes_uses_shard_regions() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let manual_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();

        let values = [40.0f32, 41.0, 42.0, 43.0];
        let replicated_sharding = Sharding::replicated(manual_mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            manual_mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // Each device can slice its own replica even though the SPMD partitioner cannot plan manual mesh axes.
        let sharded_target =
            Sharding::new(manual_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&session, Placement::new(manual_mesh.clone(), sharded_target.clone()).unwrap())
            .unwrap();
        assert_eq!(resharded.sharding(), &sharded_target);
        assert_eq!(resharded.addressable_shards().count(), 4);

        // Each device-local shard now owns one element of the input vector.
        assert_eq!(shard_values(&resharded), vec![vec![40.0], vec![41.0], vec![42.0], vec![43.0]]);

        assert_ne!(session.eager_dispatch_statistics().misses, 0);
        let on_devices = source_array.reshard_on_devices(&session, &manual_mesh, &sharded_target, false).unwrap();
        assert_eq!(shard_values(&on_devices), shard_values(&resharded));
    }

    #[test]
    fn test_array_to_placement_host_fallback_returns_before_readback() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let values = [0f32, 1., 2., 3., 4., 5., 6., 7.];
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let r#type = ArrayType::new_static(DataType::F32, [2, 4]);
        let (producer_event, producer) = client.event(()).unwrap();
        let source =
            Array::from_host_buffer(&session, r#type, mesh.clone(), Some(replicated), values_to_bytes(&values))
                .unwrap()
                .with_execution_fence(ExecutionFence::new(vec![producer_event]));

        // A destination with unreduced axes on a manual mesh has no device strategy, because shard regions need one
        // value per slice and a compiled reshard needs a mesh without manual axes, so the host fallback places it. The
        // readback waits for the pending producer of the source, so the placement can only return because the readback
        // and the upload continue in the background.
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2).with_unreduced_axes(["x"]).unwrap();
        let memory = Memory::Host { pinned: true };
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let target = Placement::new(mesh.clone(), sharding.clone())
            .unwrap()
            .with_memory(memory)
            .with_layout(layout.clone());
        let placed = source.to_placement(&session, target).unwrap();
        assert_eq!(placed.data_type(), DataType::F32);
        assert_eq!(placed.shape(), StaticShape::new(vec![2, 4]));
        assert_eq!(placed.mesh(), &mesh);
        assert_eq!(placed.sharding(), &sharding);
        assert_eq!(placed.r#type().memory(), memory);
        assert_eq!(placed.layout(), Some(&layout));
        assert!(matches!(placed.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))));
        assert_eq!(shard_memory_kinds(&placed), ["pinned_host"; 2]);
        for shard in placed.addressable_shards() {
            let buffer = shard.buffer().unwrap();
            assert_eq!(buffer.specification().unwrap().layout, layout.to_pjrt().unwrap());
            assert_eq!(buffer.ready().unwrap().ready(), Ok(false));
        }

        // The device at coordinate zero along the unreduced axis holds the whole value, and the other one holds the
        // additive identity.
        producer.set(None).unwrap();
        placed.block_until_ready().unwrap();
        assert_eq!(shard_values(&placed), vec![values.to_vec(), vec![0.; 8]]);
        assert_eq!(placed.copy_to_host().unwrap().elements::<f32>().unwrap(), values);
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_to_placement_host_fallback_poisons_result_on_readback_failure() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let (producer_event, producer) = client.event(()).unwrap();
        let source = Array::from_host_buffer(
            &session,
            r#type,
            mesh.clone(),
            Some(replicated),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap()
        .with_execution_fence(ExecutionFence::new(vec![producer_event]));
        // A destination with unreduced axes on a manual mesh has no device strategy (refer to
        // `test_array_to_placement_host_fallback_returns_before_readback`), so the host fallback places it.
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let placed = source.to_placement(&session, Placement::new(mesh, sharding).unwrap()).unwrap();

        // The failure of the producer fails the readback, which poisons every destination buffer with an error that
        // also carries the failure of the device work.
        producer.set(Some(PjrtError::aborted("producer failed"))).unwrap();
        let Err(Error::Pjrt(PjrtError::Internal { message, .. })) = placed.block_until_ready() else {
            panic!("the result of a failed host fallback must be poisoned");
        };
        assert!(message.starts_with(
            "the host fallback of an array placement failed with: producer failed (the device work that it replaced \
            failed with: ",
        ));
        for shard in placed.addressable_shards() {
            assert!(matches!(
                shard.buffer().unwrap().ready().unwrap().r#await(),
                Err(PjrtError::Internal { message: shard_message, .. }) if shard_message == message,
            ));
        }
        assert!(matches!(
            session.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(_, Error::Pjrt(PjrtError::Internal { message: failure, .. }))]
                    if *failure == message),
        ));
    }

    #[test]
    fn test_array_to_placement_host_fallback_rejects_varying_manual_axes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        let sharding = Sharding::replicated(logical_mesh.clone(), 0).with_varying_manual_axes(["x"]).unwrap();
        let first_bytes = values_to_bytes(&[11f32]);
        let second_bytes = values_to_bytes(&[22f32]);
        let source = Array::from_host_shards(
            &session,
            ArrayType::scalar(DataType::F32),
            mesh,
            Some(sharding.clone()),
            [(devices[0].id(), first_bytes.as_slice()), (devices[1].id(), second_bytes.as_slice())],
        )
        .unwrap();

        // The devices hold different values for the same slice, which a dense host copy cannot represent, so the
        // host fallback reports the error of the device work instead of uploading one of those values everywhere.
        let reversed_mesh = DeviceMesh::new(logical_mesh, vec![devices[1], devices[0]]).unwrap();
        let target = Placement::new(reversed_mesh, sharding).unwrap();
        assert!(matches!(
            source.to_placement(&session, target),
            Err(Error::Array(error))
                if matches!(error.as_ref(), ArrayError::ManualReshardingAxis { axis_name } if axis_name == "x"),
        ));
        assert_eq!(shard_values(&source), vec![vec![11.], vec![22.]]);
    }

    #[test]
    fn test_array_into_placement_preserves_same_partially_addressable_placement() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_source_buffer = client
            .buffer(values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
            .unwrap();
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let source_array = Array::from_addressable_buffers(
            &session,
            source_array_type,
            mesh.clone(),
            Some(sharding.clone()),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array = source_array
            .into_placement(&session, Placement::new(mesh.clone(), sharding.clone()).unwrap())
            .unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied_array.device_shard(local_device_id).unwrap().copy_to_host_bytes().unwrap();
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_array_into_placement_donates_source_and_returns_independently_readable_output() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = four_device_mesh_x(client);

        let values = [300.0f32, 301.0, 302.0, 303.0];
        let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // `Array::into_placement` consumes the source array and donates its buffers to the compiled SPMD reshard. PJRT
        // may reuse the memory of those buffers for the output. The donated source is not readable after this call
        // because it has been consumed, but the output buffers of the returned array must be independently
        // addressable on each device.
        let target_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let moved = source_array
            .into_placement(&session, Placement::new(mesh.clone(), target_sharding).unwrap())
            .unwrap();

        assert_eq!(shard_values(&moved), vec![vec![300.0], vec![301.0], vec![302.0], vec![303.0]]);
    }

    #[test]
    fn test_array_reshard_on_devices() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
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
        let tiled = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let shared_buffers = |left: &Array, right: &Array| {
            left.shards()
                .iter()
                .zip(right.shards())
                .map(|(left, right)| Arc::ptr_eq(left.buffer().unwrap(), right.buffer().unwrap()))
                .collect::<Vec<_>>()
        };

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
        assert_eq!(shared_buffers(&source, &output), vec![true; 4]);
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

        // Only the selected buffers are copied, the others are shared, and the result keeps the commitment of the array.
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
    fn test_array_exact_shard_buffers() {
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
        let plan =
            plan_exact_shard_put(BufferType::F32, sources, source.sharding(), &source.shape(), &mesh, &replicated);
        let plan = plan.unwrap().unwrap();
        assert_eq!(plan.cross_host_sends(0).map(ShardMove::destination_device).collect::<Vec<_>>(), remote);
        let cross_host_layouts = HashMap::from([(StaticShape::new(vec![2, 3]), row_major.to_pjrt().unwrap())]);

        // The only shard of the source is sent to both remote devices. It is staged in device memory and converted to
        // the layout of the receives once, and both sends read the same buffer. The CPU plugin lacks the cross-host
        // transfers extension, so the first send fails once every buffer that the sends read is prepared. The result
        // would wait for the readiness of the sent shard and of its staged copy and for the fence of the conversion.
        let mut completion = PlacementCompletion::default();
        let error = source.exact_shard_buffers(&plan, &cross_host_layouts, memory, &mut completion).err();
        assert!(matches!(
            error,
            Some(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));
        assert_eq!((completion.events.len(), completion.fences.len()), (2, 1));
        assert_eq!(session.cached_program_count(), 1);
        completion.join(None).unwrap().block_until_ready().unwrap();
    }

    #[test]
    fn test_array_cross_host_source_buffer() {
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
            let destination = client_devices[1].memory(memory).unwrap();
            let buffer = client.buffer(values.as_slice(), BufferType::F32, [3u64], None, destination, None).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [3]).with_memory(memory);
            let source =
                Array::from_addressable_buffers(&session, r#type, mesh.clone(), Some(replicated.clone()), vec![buffer])
                    .unwrap();
            assert!(source.shards()[0].buffer().is_none());
            let held = Arc::clone(source.shards()[1].buffer().unwrap());

            let mut completion = PlacementCompletion::default();
            let sent = source.cross_host_source_buffer(0, &client_devices[0], process_index, &mut completion).unwrap();
            completion.join(None).unwrap().block_until_ready().unwrap();
            assert!(!Arc::ptr_eq(&sent, &held));
            assert_eq!(sent.device().unwrap().id().unwrap(), devices[0].id());
            assert_eq!(sent.memory().unwrap().kind().unwrap(), kind);
            assert_eq!(sent.dimensions().unwrap(), [3]);
            assert_eq!(sent.copy_to_host(None).unwrap().r#await().unwrap(), values);

            // The shard that holds a buffer sends it without a copy, and the result of the placement then waits for
            // and reports its readiness, which may be the only completion of the work that produced it.
            let mut completion = PlacementCompletion::default();
            let sent = source.cross_host_source_buffer(1, &client_devices[1], process_index, &mut completion).unwrap();
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
            .cross_host_source_buffer(1, &client_devices[1], process_index, &mut completion)
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
    fn test_array_cross_host_send_buffer() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let values = [1f32, 2., 3., 4., 5., 6.];
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let default_layout = cross_host_default_layout(client, BufferType::F32, 0, &StaticShape::new(vec![2, 3]))
            .unwrap()
            .unwrap();
        assert_eq!(default_layout, row_major.to_pjrt().unwrap());
        let upload = |session: &Session, layout: &Layout| {
            let r#type = ArrayType::new_static(DataType::F32, [2, 3]).with_layout(layout.clone());
            Array::from_host_buffer(session, r#type, mesh.clone(), None, values_to_bytes(&values)).unwrap()
        };

        // A shard that already has the default layout of the receiver is sent as it is.
        let source = upload(&session, &row_major);
        let shard = source.addressable_shards().next().unwrap();
        let mut completion = PlacementCompletion::default();
        let buffer = Arc::clone(shard.buffer().unwrap());
        let sent = source.cross_host_send_buffer(shard, buffer, &default_layout, &mut completion).unwrap();
        assert!(Arc::ptr_eq(&sent, shard.buffer().unwrap()));
        assert!(completion.join(None).is_none());
        assert_eq!(session.cached_program_count(), 0);

        // A transfer moves the raw bytes of the buffer, so a shard in another layout is converted on its device first.
        let source = upload(&session, &column_major);
        let shard = source.addressable_shards().next().unwrap();
        let mut completion = PlacementCompletion::default();
        let buffer = Arc::clone(shard.buffer().unwrap());
        let sent = source.cross_host_send_buffer(shard, buffer, &default_layout, &mut completion).unwrap();
        completion.join(None).unwrap().block_until_ready().unwrap();
        assert_eq!(session.cached_program_count(), 1);
        assert_eq!(sent.specification().unwrap().layout, default_layout);
        assert_eq!(sent.copy_to_host(None).unwrap().r#await().unwrap(), values_to_bytes(&values));

        // A shard that the type declares in host memory is staged in the default memory of its device first, because
        // the cross-host transfers of the CUDA plugin read device buffers, also when it needs no conversion.
        for (memory, layout) in
            [(Memory::Host { pinned: true }, &row_major), (Memory::Host { pinned: false }, &column_major)]
        {
            let r#type = ArrayType::new_static(DataType::F32, [2, 3]).with_layout(layout.clone()).with_memory(memory);
            let source =
                Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&values)).unwrap();
            let shard = source.addressable_shards().next().unwrap();
            assert_ne!(shard.buffer().unwrap().memory().unwrap().kind().unwrap(), "device");
            let mut completion = PlacementCompletion::default();
            let buffer = Arc::clone(shard.buffer().unwrap());
            let sent = source.cross_host_send_buffer(shard, buffer, &default_layout, &mut completion).unwrap();
            completion.join(None).unwrap().block_until_ready().unwrap();
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
            let default_layout = cross_host_default_layout(client, BufferType::F32, 0, &StaticShape::new(vec![size]))
                .unwrap()
                .unwrap();
            assert_eq!(padded_buffer.specification().unwrap().layout, default_layout);
            assert!(!is_cross_host_sendable(padded_buffer, &StaticShape::new(vec![size]), &default_layout).unwrap());
            let mut completion = PlacementCompletion::default();
            let buffer = Arc::clone(padded_buffer);
            let sent = padded.cross_host_send_buffer(shard, buffer, &default_layout, &mut completion).unwrap();
            let mut region_completion = PlacementCompletion::default();
            let region = Array::prepare_shard_region(
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
                assert_eq!(sent.dimensions().unwrap(), [size as u64]);
                assert_eq!(sent.dynamic_dimensions().unwrap(), Vec::<u64>::new());
                assert_eq!(sent.on_device_size_in_bytes().unwrap(), 4 * size, "extent {size}");
                assert_eq!(sent.specification().unwrap().layout, default_layout);
                assert!(is_cross_host_sendable(&sent, &StaticShape::new(vec![size]), &default_layout).unwrap());
                assert_eq!(sent.copy_to_host(None).unwrap().r#await().unwrap(), values_to_bytes(&values));
            }
        }
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
    fn test_array_copy_shard_through_host_in_background() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();

        // Transfer managers provide source buffers whose contents stay undefined until the test transfers them, so
        // the shard readbacks cannot complete before then.
        let managers = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| {
                let specification = BufferSpecification::new(BufferType::F32, [2u64, 2]);
                client
                    .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])).to_pjrt().unwrap();
        let mut completion = PlacementCompletion::default();
        let copies = managers
            .iter()
            .map(|manager| {
                let buffer = Arc::new(manager.retrieve_buffer(0).unwrap());
                let memory = buffer.memory().unwrap();
                let shape = StaticShape::new(vec![2, 2]);
                Array::copy_shard_through_host(
                    &session,
                    &buffer,
                    shape,
                    BufferType::F32,
                    memory,
                    Some(layout.clone()),
                    &mut completion,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        // The copies are returned before their uploads complete, in the requested layout and memory.
        assert_eq!(session.cached_program_count(), 0);
        for copy in &copies {
            assert_eq!(copy.specification().unwrap().layout, layout);
            assert_eq!(copy.memory().unwrap(), copy.device().unwrap().default_memory().unwrap());
            assert_eq!(copy.ready().unwrap().ready(), Ok(false));
        }

        // Once the sources are defined, the background workers read them back and upload them in the new layout.
        for (manager, values) in managers.iter().zip([[1f32, 2., 3., 4.], [5., 6., 7., 8.]]) {
            manager.transfer_data(0, Arc::new(values_to_bytes(&values)), 0, true).unwrap().r#await().unwrap();
        }
        completion.join(None).unwrap().block_until_ready().unwrap();
        let values = copies
            .iter()
            .map(|copy| {
                let row_major = ryft_pjrt::Layout::dense_major_to_minor(2);
                values_from_bytes::<f32>(&copy.copy_to_host(Some(row_major)).unwrap().r#await().unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(values, vec![vec![1., 2., 3., 4.], vec![5., 6., 7., 8.]]);
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_copy_shard_through_host_in_background_poisons_copy_on_readback_failure() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let managers = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| {
                let specification = BufferSpecification::new(BufferType::F32, [2u64, 2]);
                client
                    .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut completion = PlacementCompletion::default();
        let copies = managers
            .iter()
            .map(|manager| {
                let buffer = Arc::new(manager.retrieve_buffer(0).unwrap());
                let memory = buffer.memory().unwrap();
                let shape = StaticShape::new(vec![2, 2]);
                Array::copy_shard_through_host(&session, &buffer, shape, BufferType::F32, memory, None, &mut completion)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        // The readback of the first shard fails, which poisons its copy with its error, while the independent copy of
        // the second shard completes. The completion of both reports the failure.
        managers[0].set_error(0, PjrtError::aborted("source upload failed")).unwrap();
        managers[1]
            .transfer_data(0, Arc::new(values_to_bytes(&[5f32, 6., 7., 8.])), 0, true)
            .unwrap()
            .r#await()
            .unwrap();
        assert!(matches!(
            copies[0].ready().unwrap().r#await(),
            Err(PjrtError::Aborted { message, .. }) if message == "source upload failed",
        ));
        assert_eq!(copies[1].copy_to_host(None).unwrap().r#await().unwrap(), values_to_bytes(&[5f32, 6., 7., 8.]));
        assert!(matches!(
            completion.join(None).unwrap().block_until_ready(),
            Err(PjrtError::Aborted { message, .. }) if message == "source upload failed",
        ));
        assert!(matches!(
            session.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(_, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "source upload failed"),
        ));
    }

    #[test]
    fn test_array_copy_shard_through_host_in_background_reports_failed_source_without_elements() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap().remove(0);

        // A shard without elements issues no copy, but its round trip still waits for its source and fails with it.
        let specification = BufferSpecification::new(BufferType::F32, [0u64, 2]);
        let manager = client
            .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
            .unwrap();
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
        assert!(session.synchronize().is_err());
    }

    #[test]
    fn test_array_copy_shard_through_host_in_background_independently() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let managers = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| {
                let specification = BufferSpecification::new(BufferType::F32, [2u64, 2]);
                client
                    .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut completion = PlacementCompletion::default();
        let copies = managers
            .iter()
            .map(|manager| {
                let buffer = Arc::new(manager.retrieve_buffer(0).unwrap());
                let memory = buffer.memory().unwrap();
                let shape = StaticShape::new(vec![2, 2]);
                Array::copy_shard_through_host(&session, &buffer, shape, BufferType::F32, memory, None, &mut completion)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        // The copy of the first shard completes while the source of the second one is still pending, so that a
        // producer of that source may depend on the first copy.
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

        // The completion still joins the copy of the second shard, and with it the failure of its source.
        managers[1].set_error(0, PjrtError::aborted("second source failed")).unwrap();
        assert_eq!(copies[0].copy_to_host(None).unwrap().r#await().unwrap(), values_to_bytes(&[1f32, 2., 3., 4.]));
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
    fn test_array_plan_shard_regions() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();

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
        let plan = source.plan_shard_regions(&ShardLayout::new(&source.shape(), &rows_mesh, &rows).unwrap()).unwrap();
        let moves = plan
            .moves()
            .iter()
            .map(|movement| {
                (movement.source_shard_index(), movement.destination_shard_index(), movement.slice().to_vec())
            })
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

        // A replicated source has one slice, which every destination shard intersects, and each region moves from the
        // replica on the device of its destination shard, also for the shards of different sizes of a split of five
        // elements over four devices.
        let replicated_mesh = DeviceMesh::new(logical_mesh_2x2(), devices.clone()).unwrap();
        let replicated = Sharding::replicated(replicated_mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [5]);
        let source =
            Array::from_addressable_buffers(&session, r#type, replicated_mesh, Some(replicated), Vec::new()).unwrap();
        let split = Sharding::new(rows_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["z"])]).unwrap();
        let plan = source.plan_shard_regions(&ShardLayout::new(&source.shape(), &rows_mesh, &split).unwrap()).unwrap();
        let moves = plan
            .moves()
            .iter()
            .map(|movement| {
                (movement.source_shard_index(), movement.destination_shard_index(), movement.slice().to_vec())
            })
            .collect::<Vec<_>>();
        assert_eq!(moves, vec![(0, 0, vec![0..2]), (1, 1, vec![2..3]), (2, 2, vec![3..4]), (3, 3, vec![4..5])]);

        // The boundaries of a split of six elements over four devices (i.e., `[0, 2)`, `[2, 4)`, `[4, 5)`, and
        // `[5, 6)`) are offset from those of a split over two devices (i.e., `[0, 3)` and `[3, 6)`), so each destination
        // shard joins the parts of the source shards that it intersects, in the order of their starts.
        let r#type = ArrayType::new_static(DataType::F32, [6]);
        let source = Array::from_addressable_buffers(&session, r#type, rows_mesh, Some(split), Vec::new()).unwrap();
        let halves_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("w", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            devices[..2].to_vec(),
        )
        .unwrap();
        let halves =
            Sharding::new(halves_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["w"])]).unwrap();
        let plan = source
            .plan_shard_regions(&ShardLayout::new(&source.shape(), &halves_mesh, &halves).unwrap())
            .unwrap();
        let moves = plan
            .moves()
            .iter()
            .map(|movement| {
                (movement.source_shard_index(), movement.destination_shard_index(), movement.slice().to_vec())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            moves,
            vec![(0, 0, vec![0..2]), (1, 0, vec![2..3]), (1, 1, vec![3..4]), (2, 1, vec![4..5]), (3, 1, vec![5..6])],
        );
    }

    #[test]
    fn test_array_device_placement_strategy() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        assert!(matches!(client.cross_host_transfers_extension(), Err(PjrtError::Unimplemented { .. })));
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local = Device::from_pjrt(&local_device).unwrap();
        let first_remote = Device::new(local.id() + 1, local.process_index() + 1);
        let second_remote = Device::new(local.id() + 2, local.process_index() + 2);
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
        let resolve_layout = cross_host_layout_resolver(session.client());
        let process_index = local.process_index();

        // The choice reads only metadata that every process shares, so whether this process holds its buffers does not
        // change it. Keeping each shard on its device moves no shard across processes.
        for source in [&held, &metadata_only] {
            assert!(matches!(
                source.device_placement_strategy(&resolve_layout, &source_mesh, &sharding, process_index),
                Ok(DevicePlacementStrategy::ExactShards { plan, cross_host_layouts })
                    if plan.cross_host_moves().count() == 0 && cross_host_layouts.is_empty(),
            ));
        }

        // This process only copies its own shard, but the shard of the first remote process moves to the second one,
        // which needs the missing cross-host transfers extension. Every process therefore declines whole shards and
        // shard regions alike, and all of them report the missing extension, instead of this process returning after
        // its local copy while the others switch to another strategy.
        let destination_mesh = DeviceMesh::new(logical_mesh, vec![local, second_remote]).unwrap();
        let destination_sharding =
            Sharding::new(destination_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let sources = held.shards().iter().map(ArrayShard::descriptor);
        let plan = plan_exact_shard_put(
            BufferType::F32,
            sources,
            &sharding,
            &held.shape(),
            &destination_mesh,
            &destination_sharding,
        )
        .unwrap()
        .unwrap();
        assert_eq!(plan.local_copies(local.process_index()).count(), 1);
        assert_eq!(plan.cross_host_sends(local.process_index()).count(), 0);
        assert_eq!(plan.cross_host_receives(local.process_index()).count(), 0);
        assert_eq!(plan.cross_host_moves().count(), 1);
        for source in [&held, &metadata_only] {
            assert!(matches!(
                source.device_placement_strategy(&resolve_layout, &destination_mesh, &destination_sharding, process_index),
                Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                    if message == "the cross-host transfers extension is not provided by the PJRT plugin",
            ));
        }
        assert!(matches!(
            held.to_placement(
                &session,
                Placement::new(destination_mesh.clone(), destination_sharding.clone()).unwrap(),
            ),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));

        // An array on devices that this process chose on its own holds a full value that every process holds on
        // devices of its own choice, so it is placed with local work only, whether or not it is committed, by every
        // process whose devices do not hold the whole destination (e.g., the second remote process for a destination on
        // this process). The same array on devices that every process knows plans its transfers as usual.
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
            for (mesh, sharding, process_index) in [
                (&destination_mesh, &destination_sharding, process_index),
                (&local_mesh, &local_sharding, second_remote.process_index()),
            ] {
                assert!(matches!(
                    source.device_placement_strategy(&resolve_layout, mesh, sharding, process_index),
                    Ok(DevicePlacementStrategy::ProcessLocalValue),
                ));
            }
        }
        assert!(matches!(
            uploaded.clone().with_locally_assigned(false).device_placement_strategy(
                &resolve_layout,
                &destination_mesh,
                &destination_sharding,
                process_index,
            ),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));

        // A destination on devices of this process alone moves nothing between processes, so this process places such
        // an array there with the strategy of an array on devices that every process knows.
        assert!(matches!(
            uploaded.device_placement_strategy(&resolve_layout, &local_mesh, &local_sharding, process_index),
            Ok(DevicePlacementStrategy::ExactShards { plan, cross_host_layouts })
                if plan.cross_host_moves().count() == 0 && cross_host_layouts.is_empty(),
        ));
    }

    #[test]
    fn test_array_device_placement_strategy_receives_executable_buffer_type() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let local = Device::new(local_device_id, local_device.process_index().unwrap());
        let remote = Device::new(remote_device_id, 1);
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
        assert_eq!(plan.element_type(), BufferType::Predicate);
        assert_eq!(
            plan.cross_host_receives(local.process_index()).collect::<Vec<_>>(),
            [&ShardMove::new(1, remote, 1, local, vec![2..4], 3)],
        );
    }

    #[test]
    fn test_array_device_placement_strategy_agrees_across_processes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let local = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(local.iter().map(Device::process_index).collect::<Vec<_>>(), [0, 0]);
        let next_id = local.iter().map(Device::id).max().unwrap() + 1;
        let remote = [Device::new(next_id, 1), Device::new(next_id + 1, 1)];
        let axis = |size| LogicalMesh::new(vec![MeshAxis::new("x", size, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = |devices: &[Device]| DeviceMesh::new(axis(devices.len()), devices.to_vec()).unwrap();
        let replicated = |mesh: &DeviceMesh| Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let sharded = |mesh: &DeviceMesh| {
            let dimensions = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
            Sharding::new(mesh.logical_mesh().clone(), dimensions).unwrap()
        };
        let r#type = ArrayType::new_static(DataType::F32, [2, 2]);
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let values = values_to_bytes(&[1f32, 2., 3., 4.]);

        // Every process sees the same metadata of a source whose devices every process knows, whichever buffers it
        // holds, so its view is the import of that metadata without buffers. A process that holds a source with
        // locally assigned devices sees its own devices instead, which are the remote devices for the second process.
        let shared = |r#type: &ArrayType, devices: &[Device], sharding: &dyn Fn(&DeviceMesh) -> Sharding| {
            let mesh = mesh(devices);
            let view = || {
                let sharding = Some(sharding(&mesh));
                Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), sharding, Vec::new()).unwrap()
            };
            vec![view(), view()]
        };
        let locally_assigned = |committed: bool, sharding: Option<Sharding>, devices: &[Device]| {
            let own = Array::from_host_buffer(&session, r#type.clone(), None, sharding.clone(), &values).unwrap();
            let own = own.with_committed(committed);
            let mesh = mesh(devices);
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
            let mesh = mesh(devices);
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
        let multi_device_sharding = sharded(&mesh(&local));
        let sources = [
            ("shared replicated over both processes", shared(&r#type, &[local[0], remote[0]], &replicated)),
            ("shared sharded over both processes", shared(&r#type, &[local[0], remote[0]], &sharded)),
            ("shared on process 0", shared(&r#type, &[local[0]], &replicated)),
            (
                "shared with a declared layout",
                shared(&r#type.clone().with_layout(column_major.clone()), &[local[0]], &replicated),
            ),
            (
                "shared sharded with a declared layout",
                shared(&r#type.clone().with_layout(column_major.clone()), &[local[0]], &sharded),
            ),
            (
                "shared in pinned host memory",
                shared(&r#type.clone().with_memory(Memory::Host { pinned: true }), &[local[0]], &replicated),
            ),
            (
                "shared in unpinned host memory",
                shared(&r#type.clone().with_memory(Memory::Host { pinned: false }), &[local[0]], &replicated),
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
        let destinations = [
            ("replicated over both processes", mesh(&[local[0], remote[0]]), replicated(&mesh(&[local[0], remote[0]]))),
            ("sharded over both processes", mesh(&[local[0], remote[0]]), sharded(&mesh(&[local[0], remote[0]]))),
            (
                "unreduced over both processes",
                mesh(&[local[0], remote[0]]),
                replicated(&mesh(&[local[0], remote[0]])).with_unreduced_axes(["x"]).unwrap(),
            ),
            ("replicated on process 0", mesh(&local), replicated(&mesh(&local))),
            ("sharded on process 1", mesh(&remote), sharded(&mesh(&remote))),
        ];
        let outcomes = sources
            .iter()
            .flat_map(|(source, views)| {
                destinations.iter().map(move |(destination, mesh, sharding)| {
                    let outcome = match placement_agreement(views, mesh, sharding) {
                        Ok(strategy) => strategy,
                        Err(disagreement) => format!("DISAGREE: {disagreement}"),
                    };
                    format!("{source} -> {destination}: {outcome}")
                })
            })
            .collect::<Vec<_>>();
        // Every transfer and program that one process plans has its counterpart on the other process. The processes
        // choose the same strategy, except that a process whose own value lies on its devices and whose devices hold
        // the whole destination places it with the strategy of a shared source, which moves nothing between processes.
        // A partial import reaches its first transfer in production whenever another local replica serves a missing
        // one or a missing tile is only copied locally, which in pinned host memory includes the staging of each sent
        // replica in device memory. A disagreement names the strategy of each process. The partially addressable sharded import has no local buffer
        // at all for its first tile, so process 0 stops before its transfers wherever that tile must leave the
        // process or feed a gather, while process 1 waits for its receive, which is the accepted limit of the PJRT
        // cross-host transfers that `.tasks/plan_placement_decisions.md` records.
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                shared replicated over both processes -> replicated over both processes: exact shards
                shared replicated over both processes -> sharded over both processes: compiled reshard
                shared replicated over both processes -> unreduced over both processes: compiled reshard
                shared replicated over both processes -> replicated on process 0: exact shards
                shared replicated over both processes -> sharded on process 1: shard regions
                shared sharded over both processes -> replicated over both processes: compiled reshard
                shared sharded over both processes -> sharded over both processes: exact shards
                shared sharded over both processes -> unreduced over both processes: compiled reshard
                shared sharded over both processes -> replicated on process 0: shard regions
                shared sharded over both processes -> sharded on process 1: exact shards
                shared on process 0 -> replicated over both processes: exact shards
                shared on process 0 -> sharded over both processes: shard regions
                shared on process 0 -> unreduced over both processes: compiled reshard across meshes
                shared on process 0 -> replicated on process 0: exact shards
                shared on process 0 -> sharded on process 1: shard regions
                shared with a declared layout -> replicated over both processes: exact shards
                shared with a declared layout -> sharded over both processes: compiled reshard across meshes
                shared with a declared layout -> unreduced over both processes: compiled reshard across meshes
                shared with a declared layout -> replicated on process 0: exact shards
                shared with a declared layout -> sharded on process 1: compiled reshard across meshes
                shared sharded with a declared layout -> replicated over both processes: exact shards
                shared sharded with a declared layout -> sharded over both processes: compiled reshard across meshes
                shared sharded with a declared layout -> unreduced over both processes: compiled reshard across meshes
                shared sharded with a declared layout -> replicated on process 0: exact shards
                shared sharded with a declared layout -> sharded on process 1: compiled reshard across meshes
                shared in pinned host memory -> replicated over both processes: exact shards
                shared in pinned host memory -> sharded over both processes: shard regions
                shared in pinned host memory -> unreduced over both processes: compiled reshard across meshes
                shared in pinned host memory -> replicated on process 0: exact shards
                shared in pinned host memory -> sharded on process 1: shard regions
                shared in unpinned host memory -> replicated over both processes: exact shards
                shared in unpinned host memory -> sharded over both processes: shard regions
                shared in unpinned host memory -> unreduced over both processes: compiled reshard across meshes
                shared in unpinned host memory -> replicated on process 0: exact shards
                shared in unpinned host memory -> sharded on process 1: shard regions
                partially addressable import -> replicated over both processes: exact shards
                partially addressable import -> sharded over both processes: shard regions
                partially addressable import -> unreduced over both processes: compiled reshard across meshes
                partially addressable import -> replicated on process 0: exact shards
                partially addressable import -> sharded on process 1: shard regions
                partially addressable import with a declared layout -> replicated over both processes: exact shards
                partially addressable import with a declared layout -> sharded over both processes: compiled reshard across meshes
                partially addressable import with a declared layout -> unreduced over both processes: compiled reshard across meshes
                partially addressable import with a declared layout -> replicated on process 0: exact shards
                partially addressable import with a declared layout -> sharded on process 1: compiled reshard across meshes
                partially addressable import in pinned host memory -> replicated over both processes: exact shards
                partially addressable import in pinned host memory -> sharded over both processes: shard regions
                partially addressable import in pinned host memory -> unreduced over both processes: compiled reshard across meshes
                partially addressable import in pinned host memory -> replicated on process 0: exact shards
                partially addressable import in pinned host memory -> sharded on process 1: shard regions
                partially addressable sharded import -> replicated over both processes: DISAGREE: shard regions: process 0 fails before it posts its transfers: array shard #0 on device 0 has no addressable buffer
                partially addressable sharded import -> sharded over both processes: exact shards
                partially addressable sharded import -> unreduced over both processes: DISAGREE: compiled reshard across meshes: process 0 fails before it posts its transfers: array shard #0 on device 0 has no addressable buffer
                partially addressable sharded import -> replicated on process 0: compiled reshard
                partially addressable sharded import -> sharded on process 1: DISAGREE: exact shards: process 0 fails before it posts its transfers: array shard #0 on device 0 has no addressable buffer
                process-local committed -> replicated over both processes: process-local value
                process-local committed -> sharded over both processes: process-local value
                process-local committed -> unreduced over both processes: process-local value
                process-local committed -> replicated on process 0: exact shards | process-local value
                process-local committed -> sharded on process 1: process-local value | shard regions
                process-local uncommitted -> replicated over both processes: process-local value
                process-local uncommitted -> sharded over both processes: process-local value
                process-local uncommitted -> unreduced over both processes: process-local value
                process-local uncommitted -> replicated on process 0: exact shards | process-local value
                process-local uncommitted -> sharded on process 1: process-local value | shard regions
                process-local on two devices -> replicated over both processes: process-local value
                process-local on two devices -> sharded over both processes: process-local value
                process-local on two devices -> unreduced over both processes: process-local value
                process-local on two devices -> replicated on process 0: compiled reshard | process-local value
                process-local on two devices -> sharded on process 1: process-local value | exact shards"},
        );

        // A replicated `u8[5]` split unevenly over both processes needs only a slice of the replica that each process
        // already holds, so a declared layout that its buffers already have and a memory that each process stages
        // its replica out of on its own devices leave every process with shard regions. A strided layout, which no
        // device program addresses, rules them out, and no other device strategy produces shards of two sizes.
        let vector = ArrayType::new_static(DataType::U8, [5]);
        let split_mesh = mesh(&[local[0], remote[0]]);
        let split = Sharding::new(split_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated_vector = |mesh: &DeviceMesh| Sharding::replicated(mesh.logical_mesh().clone(), 1);
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
            let views = shared(r#type, &[local[0], remote[0]], &replicated_vector);
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
                strided layout in host memory -> uneven split: error: compiled reshard cannot produce the shards of different sizes of shape `[5]` with sharding `{mesh<['x'=2:explicit]>, [{'x'}]}`"},
        );

        // A process that imported only the metadata of a tile that the plan has it send (i.e., without a buffer for it
        // or for any local replica) agrees on the plan, and then fails before it issues any transfer, while its peer
        // issues its receive and waits. This is the accepted limit of the PJRT cross-host transfers that
        // `.tasks/plan_placement_decisions.md` records, because a process cannot fail the pending receive of another.
        let (views, (_, mesh, sharding)) = (&sources[2].1, &destinations[0]);
        assert_eq!(placement_agreement(views, mesh, sharding), Ok("exact shards".to_string()));
        let Ok(DevicePlacementStrategy::ExactShards { plan, cross_host_layouts }) =
            views[0].device_placement_strategy(&simulated_cross_host_layout, mesh, sharding, 0)
        else {
            panic!("a replica that crosses processes must move as a whole shard");
        };
        let sends = plan.cross_host_sends(0).map(ShardMove::destination_device).collect::<Vec<_>>();
        assert_eq!(sends, [remote[0]]);
        assert_eq!(plan.cross_host_receives(1).map(ShardMove::source_device).collect::<Vec<_>>(), [local[0]]);
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

        // XLA drops NaN values from a maximum, so completing a pending maximum of `f32` values needs a host round
        // trip. Over the devices of both processes, no process can read the value back alone, so both plan the round
        // trip without an owner and return the same error before they do anything. When every device belongs to
        // process 1, process 1 reads the value back and uploads it, and process 0 returns the placement without
        // buffers, as it does for a compiled reshard that it does not run.
        let maximum = |mesh: &DeviceMesh| {
            replicated(mesh)
                .with_unreduced_axes(["x"])
                .unwrap()
                .with_unreduced_kind(UnreducedKind::Max)
                .unwrap()
        };
        let mut outcomes = Vec::new();
        for (name, devices) in [("over both processes", [local[0], remote[0]]), ("on process 1", remote)] {
            let views = shared(&r#type, &devices, &maximum);
            let destination_mesh = DeviceMesh::new(axis(devices.len()), devices.to_vec()).unwrap();
            let destination = replicated(&destination_mesh);
            let planned = placement_agreement(&views, &destination_mesh, &destination);
            let owners = views
                .iter()
                .enumerate()
                .map(|(process_index, view)| {
                    match view.device_placement_strategy(
                        &simulated_cross_host_layout,
                        &destination_mesh,
                        &destination,
                        process_index,
                    ) {
                        Ok(DevicePlacementStrategy::HostRoundTrip { process_index, .. }) => process_index,
                        _ => panic!("the destination needs a host round trip"),
                    }
                })
                .collect::<Vec<_>>();
            let strategy =
                views[0].device_placement_strategy(&simulated_cross_host_layout, &destination_mesh, &destination, 0);
            let placed = strategy.and_then(|strategy| {
                let placed = views[0].clone().place_with_strategy(
                    strategy,
                    &destination_mesh,
                    &destination,
                    Memory::Device,
                    None,
                    false,
                );
                placed.map_err(|(error, source)| {
                    assert!(
                        source.is_none(),
                        "a process that cannot do the round trip returns no source to fall back on"
                    );
                    error
                })
            });
            let placed = placed.map(|placed| {
                assert_eq!((placed.mesh(), placed.sharding()), (&destination_mesh, &destination));
                format!("{} buffers", placed.addressable_shards().filter(|shard| shard.buffer().is_some()).count())
            });
            outcomes.push(format!(
                "{name}: {planned:?}, owners {owners:?}, process 0 {:?}",
                placed.map_err(|error| error.to_string())
            ));
        }
        assert_eq!(
            outcomes,
            [
                "over both processes: Ok(\"host round trip\"), owners [None, None], process 0 Err(\"a compiled reshard \
                 cannot take `f32` values from a pending `max` reduction to no reduction state along mesh axis `x`, \
                 because XLA's maximum and minimum reductions drop NaN values, which the maximum and minimum of a \
                 compiled program propagate\")",
                "on process 1: Ok(\"host round trip\"), owners [Some(1), Some(1)], process 0 Ok(\"0 buffers\")",
            ],
        );
        session.synchronize().unwrap();
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_cross_host_shape_retains_dimension_error_and_shard_context() {
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
    fn test_cross_host_default_layout() {
        // A receive of a shard allocates the default layout of the plugin for its shape, which is row-major on CPU.
        let client = cpu_client(1);
        assert_eq!(
            cross_host_default_layout(client, BufferType::F32, 0, &StaticShape::new(vec![2, 3])).unwrap(),
            Some(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![])).to_pjrt().unwrap()),
        );
        assert_eq!(
            cross_host_default_layout(client, BufferType::Predicate, 0, &StaticShape::new(vec![4])).unwrap(),
            Some(Layout::Tiled(TiledLayout::new(vec![0], vec![])).to_pjrt().unwrap()),
        );
    }

    #[test]
    fn test_source_replicas_preferred() {
        let logical_mesh =
            |size| LogicalMesh::new(vec![MeshAxis::new("x", size, MeshAxisType::Auto).unwrap()]).unwrap();
        let replicated_shards = |devices: Vec<Device>| {
            let mesh = DeviceMesh::new(logical_mesh(devices.len()), devices).unwrap();
            ShardLayout::new(&StaticShape::new(vec![2]), &mesh, &Sharding::replicated(mesh.logical_mesh().clone(), 1))
                .unwrap()
        };

        // Two replicas on each of two processes, listed out of device order.
        let sources =
            replicated_shards(vec![Device::new(1, 0), Device::new(0, 0), Device::new(3, 1), Device::new(2, 1)]);
        let sources = sources.descriptors().iter().collect::<Vec<_>>();
        let replicas = SourceReplicas::new(sources.as_slice());

        // A destination prefers the replica on its own device, then the one with the lowest device ID on its process,
        // and then the one with the lowest device ID of all processes.
        let destinations = replicated_shards(vec![
            Device::new(1, 0),
            Device::new(2, 1),
            Device::new(5, 0),
            Device::new(7, 1),
            Device::new(9, 2),
        ]);
        let preferred = destinations
            .descriptors()
            .iter()
            .map(|destination| (replicas.preferred(destination).device(), replicas.preferred(destination).index()))
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
    fn test_plan_exact_shard_put_uses_cross_host_send_and_receive_for_remote_moves() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
        )
        .unwrap();
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_source_buffer = client
            .buffer(values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
            .unwrap();
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let source_array = Array::from_addressable_buffers(
            &session,
            source_array_type,
            source_mesh,
            Some(source_sharding),
            vec![local_source_buffer],
        )
        .unwrap();
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(remote_device_id, 1), Device::new(local_device_id, client.process_index().unwrap())],
        )
        .unwrap();
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let sources = source_array.shards().iter().map(ArrayShard::descriptor);
        let (shape, source_sharding) = (source_array.shape(), source_array.sharding());
        let plan =
            plan_exact_shard_put(BufferType::F32, sources, source_sharding, &shape, &target_mesh, &target_sharding)
                .unwrap();
        let local = Device::new(local_device_id, client.process_index().unwrap());
        let remote = Device::new(remote_device_id, 1);
        let send = ShardMove::new(0, local, 0, remote, vec![0..2], 0);
        let receive = ShardMove::new(1, remote, 1, local, vec![2..4], 3);
        assert_eq!(plan, Some(ShardMovePlan::new(BufferType::F32, vec![send.clone(), receive.clone()])));

        // Every process derives the same plan and performs the moves of its own devices, so the send of this process
        // is the receive of the other one, and the other way around.
        let plan = plan.unwrap();
        assert_eq!(plan.cross_host_moves().collect::<Vec<_>>(), [&send, &receive]);
        assert_eq!(plan.local_copies(0).count(), 0);
        assert_eq!(plan.cross_host_sends(0).collect::<Vec<_>>(), [&send]);
        assert_eq!(plan.cross_host_receives(0).collect::<Vec<_>>(), [&receive]);
        assert_eq!(plan.cross_host_sends(1).collect::<Vec<_>>(), [&receive]);
        assert_eq!(plan.cross_host_receives(1).collect::<Vec<_>>(), [&send]);
    }

    #[test]
    fn test_exact_shards_layout() {
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let received_row_major = row_major.to_pjrt().unwrap();
        let received_column_major = column_major.to_pjrt().unwrap();

        // A source without a declared layout places an array without one, whatever this process receives.
        assert_eq!(exact_shards_layout(None, [&received_row_major].into_iter()).unwrap(), None);

        // The declared layout holds for local copies, and for received buffers only when they have it as well.
        assert_eq!(exact_shards_layout(Some(&column_major), std::iter::empty()).unwrap(), Some(column_major.clone()));
        assert_eq!(exact_shards_layout(Some(&row_major), [&received_row_major].into_iter()).unwrap(), Some(row_major));
        assert_eq!(
            exact_shards_layout(Some(&column_major), [&received_column_major, &received_row_major].into_iter())
                .unwrap(),
            None,
        );
    }

    #[test]
    fn test_exact_shard_transfer_key() {
        assert_eq!(exact_shard_transfer_key(2, 1, 3).unwrap(), 7);
        let error = exact_shard_transfer_key(usize::MAX, 0, 2).unwrap_err();
        assert!(matches!(&error, Error::SizeLimitExceeded { .. }));
        assert_eq!(
            error.to_string(),
            format!(
                "cross-host transfer key for source shard #{} and destination shard #0 exceeds the maximum allowed \
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
}
