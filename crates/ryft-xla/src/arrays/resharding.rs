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
//!      from another device of its process, or receives it through a PJRT cross-host transfer. Nothing is compiled.
//!   2. **Shard Regions:** The source is in default device memory without a declared layout, both shardings hold one
//!      value per slice, and either the meshes differ or uneven shards or manual destination axes rule out a compiled
//!      reshard. The source devices slice out the regions that each destination needs, and the destination devices
//!      join them.
//!   3. **Compiled Reshard:** An identity program with input and output sharding annotations runs on the mesh of the
//!      source. Across meshes, a replicated copy of the source (i.e., the source itself or its all-gather on its own
//!      mesh) is first copied or transferred to every destination device as whole shards, and the program runs on the
//!      destination mesh. A process that owns no device of the mesh of such a program compiles and runs nothing for
//!      it, and its result has the type and the placement of the results of the processes that run it, without
//!      buffers.
//!
//! The choice reads only information that every process shares (i.e., the type and shard metadata of the source,
//! whether the source has locally assigned devices, the destination, the compilation configuration of the session, and
//! the capabilities of its PJRT plugin), so all the processes of a distributed placement choose the same strategy and
//! issue matching transfers and collectives. What only one process knows (e.g., the physical layout of a buffer when
//! the type declares none, or which buffers it holds) never changes the choice. A source with locally assigned devices
//! takes none of these strategies, as the next section describes.
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
//! [`Session`], the mesh of a construction, or the device of an [`ArrayPlacement::device`] request) must therefore be
//! the same on every process of the run, and a choice that differs between processes is expressible only by naming no
//! device, which leaves the choice to each process and records the fact. For example, when each process places a value
//! on its own first addressable device with [`ArrayPlacement::device`], or builds its session with a default mesh of
//! its own addressable devices and uploads a value without a mesh, and then places the result replicated over the
//! devices of every process, every process plans whole shards from its own device as the source of the value, so each
//! one sends its shard to the others and none of them receives, and the placement never completes.
//!
//! A cross-host transfer moves raw bytes from a buffer in device memory into a buffer in the default layout and the
//! default device memory of the receiver, so the sender first copies a shard that its type declares in host memory to
//! device memory and converts it to that layout, on its device or through host memory, once for all the receivers of
//! that shard. Every process issues its cross-host transfers in one global order of device pairs, and prepares the
//! buffers that it sends and looks up the devices that it receives on before the first one, because a transfer that
//! one process issues without its peer stays paired with the next transfer between the same devices. A receiver
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
//! literal transfers. A failure after the transfers (e.g., a destination memory that a device of the receiver lacks)
//! is returned only once every transfer is issued, so the peers of the failing process complete their transfers.
//!
//! A process that fails before it issues its transfers returns its error, and its peers that already issued theirs
//! wait for transfers that never come. This is an accepted limitation, because the PJRT cross-host transfers offer no
//! way to fail a pending transfer of another process. Every process therefore decides from shared information alone
//! everything that could stop all of them, and before its first transfer it does only what its own transfers need,
//! which is preparing the buffers that it sends. Everything else that only it can check (e.g., which replica serves a
//! local copy, or a memory of its devices) it checks once its transfers are issued, so that a failure there stops only
//! this process. Which buffers a process holds is such information, and a missing buffer of a replica is served by
//! another replica of the same slice on this process. What remains before the transfers are the following failures,
//! which only the failing process can see.
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
use std::ops::Range;
use std::sync::Arc;

use ryft_core::{
    ArrayType, CastOperation, Concatenate, DataType, Device, DeviceMesh, DimensionError, Layout, Memory, MeshAxisType,
    PLACE_OPERATION_NAME, Placeholder, Placement, Sharding, Slice, StaticShape, TypeError, Typed,
};
use ryft_pjrt::extensions::cross_host_transfers::{CrossHostTransferKey, GlobalDeviceId};
use ryft_pjrt::{
    Buffer, BufferSpecification, BufferType, Client, Device as PjrtDevice, DeviceId, Error as PjrtError, Event,
    ExecutionFence, Memory as PjrtMemory,
};

use crate::arrays::host::{HostBufferSlice, PendingHostUpload, run_host_uploads, validate_mesh_device_processes};
use crate::arrays::metadata::validate_shard_layout;
use crate::arrays::placement::{ArrayPlacement, ArrayPlacementTarget};
use crate::arrays::shards::{ArrayShard, ShardDescriptor, ShardIndex, ShardLayout};
use crate::arrays::{Array, ArrayError, ArrayParts};
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
    /// When the placement and storage already match, the result shares this array's device buffers. If each process
    /// chose its input devices independently, placing that input on named devices creates a distinct logical array on
    /// every process, so independent placements remain distinct when captured. A change of devices or sharding tries
    /// the following strategies in order.
    ///
    ///   1. **Exact Shards:** When every destination shard has the slices of one source shard, each destination shard
    ///      shares the buffer of its source shard on the same device, or is filled by a copy between devices or by a
    ///      PJRT cross-host transfer for a source shard on a remote host.
    ///   2. **Shard Regions:** Across meshes, or on the same mesh for uneven shards or manual destination axes, when
    ///      both shardings hold one value per slice, the source devices slice out the regions each destination needs.
    ///      Each destination joins its regions locally or allocates an empty buffer for an empty shard.
    ///      This path uses default device memory and needs the PJRT cross-host transfers extension and default
    ///      layouts if a region crosses processes. Each participating process issues the same placements in the same
    ///      order.
    ///   3. **Compiled Reshard:** An `identity(x) = x` program with input and output sharding constraints is lowered
    ///      to StableHLO and Shardy, compiled through the compilation cache of `session`, and executed. This needs
    ///      shards of one size on both sides, so a split whose parts do not divide the extent of its dimension (e.g.,
    ///      five elements over two devices, which [`ShardLayout`] splits into shards of three and two elements) skips
    ///      this strategy. Only the processes that own a device of the mesh of such a program compile and run it, and
    ///      every other process returns the same type and placement without buffers.
    ///   4. **Host Fallback:** The global array is copied to the host as [`Array::copy_to_host`] does and uploaded
    ///      again as [`Array::from_host_buffer`] does, so the result has the shards that a host upload of the same
    ///      values with the destination sharding has. This is used only when the device work fails or is skipped
    ///      (e.g., when a manual placement cannot compile shard-local operations), when the destination mesh is fully
    ///      addressable from this process, and when this process can read the source back. That is the case when this
    ///      process holds a buffer for one replica of every distinct contribution to the value with elements (i.e., of
    ///      every tile and, along unreduced axes, of every per-device value of a tile), whatever the other processes
    ///      hold, and when no two devices that hold one such contribution differ along a varying manual mesh axis,
    ///      because those devices can hold different values for the same slice that a dense host copy cannot
    ///      represent. Devices that differ along a manual axis which splits the array hold distinct tiles, so they read
    ///      back like any other tiles. The destination buffers are created up front and the readback and the upload
    ///      continue on a background thread, so this call returns without waiting for them. A failure of that
    ///      background work, whose error also describes the failure of the device work, surfaces through the readiness
    ///      functions and the readbacks of the result and through [`Session::synchronize`]. Only when the PJRT plugin
    ///      lacks host to device transfer managers or their literal transfers does this call block until the readback
    ///      completes, and the readiness functions and readbacks of the result then report a failed upload of any of
    ///      its replicas.
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
    /// contributions along that axis. With [`UnreducedKind::Sum`](ryft_core::UnreducedKind::Sum), the devices at
    /// coordinate zero along it hold the whole value and the others hold the additive identity. With
    /// [`UnreducedKind::Max`](ryft_core::UnreducedKind::Max) or [`UnreducedKind::Min`](ryft_core::UnreducedKind::Min),
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
    /// device (i.e., [`ArrayPlacementTarget::Device`]) uses the default memory of that device, as
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
    ///   - `target`: Resolved [`Placement`] or an [`ArrayPlacement`] request specifying the destination.
    pub fn to_placement<P: Into<ArrayPlacement>>(&self, session: &Session, target: P) -> Result<Self, Error> {
        self.clone().run_placement_request(session, target.into(), false)
    }

    /// Moves this array to the provided placement, consuming `self` and donating its source buffers to the
    /// destination. This is the `ryft` analogue of JAX's `jax.device_put(arr, sharding, donate=True)`.
    ///
    /// Donation is best effort, as in JAX. Only the compiled reshard on the mesh of this array donates its input, and
    /// PJRT then may reuse the source buffers for the output, which can save the memory of the source array for large
    /// training tensors. Nothing guarantees that saving. The other strategies (e.g., moves of whole shards, shard
    /// regions, a reshard across meshes, or the host fallback) do not donate, a buffer that another handle shares is
    /// never donated, and PJRT may still allocate new output buffers. Because `self` is consumed at the language
    /// boundary, the donated buffers are not observable to the caller after this function returns. Rust ownership
    /// therefore enforces the same invariant that the `donate=True` flag carries in JAX, without exposing a runtime
    /// hazard.
    ///
    /// The strategies follow [`Self::to_placement`].
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `target`: Resolved [`Placement`] or an [`ArrayPlacement`] request specifying the destination.
    pub fn into_placement<P: Into<ArrayPlacement>>(self, session: &Session, target: P) -> Result<Self, Error> {
        self.run_placement_request(session, target.into(), true)
    }

    /// Places this array as `request` asks, which [`Self::to_placement`] and [`Self::into_placement`] implement. The
    /// request is resolved against this array (refer to [`ArrayPlacement::resolve_for`]) and placed with
    /// [`Self::run_placement_dispatch`]. Only a request that names neither devices nor a sharding (i.e., an
    /// [`ArrayPlacementTarget::InPlace`] request, which names only a memory or a layout) leaves the result on the
    /// devices that this process chose on its own when this array has such devices, because every other request names
    /// the devices of the result (refer to [`Self::is_locally_assigned`]).
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `request`: [`ArrayPlacement`] that specifies the destination.
    ///   - `donate`: Whether a compiled reshard on the mesh of this array may reuse the buffers of this array for
    ///     its output (refer to [`Self::run_placement_dispatch`]).
    pub(crate) fn run_placement_request(
        self,
        session: &Session,
        request: ArrayPlacement,
        donate: bool,
    ) -> Result<Self, Error> {
        let in_place = matches!(request.target(), ArrayPlacementTarget::InPlace);
        let locally_assigned = in_place && self.is_locally_assigned();
        let target = request.resolve_for(session, &self)?;
        // A named destination can already hold this process's local input while another process must move its input.
        // Give each such placement its own logical array on every process so that retaining two independent results
        // gives them distinct executable arguments everywhere. A weak handle keeps donation possible.
        let local_source = (!in_place && self.is_locally_assigned()).then(|| Arc::downgrade(&self.parts));
        let mut output = self.run_placement_dispatch(session, &target, donate)?;
        if local_source.is_some_and(|source| source.as_ptr() == Arc::as_ptr(&output.parts)) {
            output.parts =
                Arc::new(ArrayParts::new(session, Arc::clone(&output.parts.metadata), output.parts.shards.clone()));
        }
        Ok(output.with_locally_assigned(locally_assigned))
    }

    /// Places this array on `mesh` according to `sharding` with device work only (i.e., without copying its contents
    /// through host memory). [`Self::to_placement`] and [`Self::into_placement`] start with this work, and the
    /// execution of a compiled program uses it for inputs whose sharding differs from the one that the program was
    /// compiled for. Refer to [`Self::reshard_on_devices_or_return`] for the strategies that it tries.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
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
        self.reshard_on_devices_or_return(session, mesh, sharding, memory, layout.as_ref(), donate)
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
    pub(crate) fn with_copied_shard_buffers<F: FnMut(&Arc<Buffer<'static>>) -> bool>(
        &self,
        session: &Session,
        copy: F,
    ) -> Result<Self, Error> {
        let copies = self.addressable_shards().filter_map(ArrayShard::buffer).map(copy).collect::<Vec<_>>();
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
    ///   - `target`: Resolved destination [`Placement`] (refer to [`ArrayPlacement::resolve_for`]).
    ///   - `donate`: Whether a compiled reshard on the mesh of this array may reuse the buffers of this array for
    ///     its output. The donation only takes effect when this array is the only handle of its buffers.
    pub(crate) fn run_placement_dispatch(
        self,
        session: &Session,
        target: &Placement,
        donate: bool,
    ) -> Result<Self, Error> {
        // Every path below either returns this array or reuses some of its buffers, so associate it with `session`
        // before choosing one. Consuming `self` keeps the buffers uniquely owned for a donation.
        let array = self.with_session(session)?;
        let (mesh, sharding) = (target.mesh().clone(), target.sharding().clone());
        let memory = target.memory().unwrap_or_else(|| array.r#type().memory());
        let layout = array.placement_layout(target.layout(), &mesh, &sharding)?;
        let placed = |array: Self| array.with_committed(true).with_locally_assigned(false);
        if array.mesh() == &mesh && array.sharding() == &sharding {
            return Ok(placed(array.place_storage(session, memory, layout)?));
        }
        validate_mesh_device_processes(session.client(), &mesh)?;

        // Keep the error of the device work, which is the most informative one if the host fallback fails too.
        let (compiled_error, source) =
            match array.reshard_on_devices_or_return(session, &mesh, &sharding, memory, layout.as_ref(), donate) {
                Ok(array) => return Ok(placed(array.place_storage(session, memory, layout)?)),
                Err((error, Some(source))) => (error, source),
                Err((error, None)) => return Err(error),
            };

        // The fallback uploads the dense readback directly into the requested storage. If it cannot complete from this
        // process, surface the original device error. That includes a source whose devices hold one slice but vary
        // along a manual mesh axis, so that they can hold different values for it, which the readback rejects rather
        // than silently replacing the values of the other devices with one of them.
        let local_process = session.client().process_index()?;
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
            session,
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
        let uploaded = Self::from_host_buffer(session, host_type, mesh, Some(sharding), host_bytes.as_slice())
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
    /// compiled conversion on the device and can fall back to a host round trip for that shard. Once a buffer has the
    /// requested layout, it can be reused or copied directly to the requested memory. The result keeps the devices of
    /// this array, and with them its [locally assigned devices](Self::is_locally_assigned). It declares exactly
    /// `layout`, which [`Self::placement_layout`] resolved from information that every process shares, whatever layout
    /// the strategy that produced this array recorded, so that every process records the same layout. Without
    /// `layout`, it declares no layout and keeps the buffers as they are.
    ///
    /// The host round trip of each shard is its own job on the background upload workers of `session`, which reads the
    /// shard back once its source buffer is ready and uploads it into a destination buffer that this function creates
    /// up front, so this function returns without waiting for them, and the round trip of one shard never waits for
    /// the source of another. A failed readback or upload poisons the destination buffer of its shard, so the
    /// readiness functions and readbacks of the result and [`Session::synchronize`] report it. The execution fence of
    /// the result joins the fences of this array with the readiness of every buffer that this function creates (i.e.,
    /// the copies to another memory and the buffers of host round trips) and with the fences of the compiled
    /// conversions, because a readback reads only one replica of each tile and would otherwise succeed with a replica
    /// that did not fail. This function blocks on the readback of a shard only when the PJRT plugin lacks host to
    /// device transfer managers or their literal transfers.
    fn place_storage(self, session: &Session, memory: Memory, layout: Option<Layout>) -> Result<Self, Error> {
        if self.r#type().memory() == memory && self.layout() == layout.as_ref() {
            return Ok(self);
        }

        let source = self.with_session(session)?;
        let r#type = source.r#type().into_owned().with_memory(memory).with_layout(layout);
        let mesh = source.mesh().clone();
        let sharding = Some(source.sharding().clone());
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
            for shard in source.addressable_shards() {
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
                    Some(current_layout) if can_compile_local_operations(session) => {
                        let input_type = ArrayType::new(source.data_type(), shard.shape().into())
                            .with_memory(source.r#type().memory())
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
                    Some(_) => None,
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
        output.execution_fence = completion.join(source.execution_fence);
        output.send_fence = source.send_fence;
        Ok(output.with_locally_assigned(source.locally_assigned))
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
    /// and in the same memory, and is a copy otherwise. When a destination shard covers only a part of the array and
    /// this process can compile operations for its own shards, that part is sliced on the device of this array, as
    /// for shard regions, and copied to its destination. Slicing needs an array in the default device memory without
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
        session: &Session,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        memory: Memory,
        layout: Option<&Layout>,
    ) -> Result<Self, Error> {
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
            && (!slices
                || (can_compile_local_operations(session)
                    && self.r#type().memory() == Memory::Device
                    && self.layout().is_none()));
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
        session: &Session,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        memory: Memory,
        plan: &ShardMovePlan,
        cross_host_layouts: &HashMap<StaticShape, ryft_pjrt::Layout>,
    ) -> Result<Self, Error> {
        let mut completion = PlacementCompletion::default();
        let (buffers, layout, send_fence) =
            self.exact_shard_buffers(session, plan, cross_host_layouts, memory, &mut completion)?;
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
        session: &Session,
        plan: &ShardMovePlan,
        cross_host_layouts: &HashMap<StaticShape, ryft_pjrt::Layout>,
        memory: Memory,
        completion: &mut PlacementCompletion,
    ) -> Result<(Vec<Arc<Buffer<'static>>>, Option<Layout>, Option<ExecutionFence>), Error> {
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
                    let buffer =
                        self.cross_host_source_buffer(session, shard.index(), device, process_index, completion)?;
                    let layout = &cross_host_layouts[&movement.shape()];
                    Arc::clone(entry.insert(self.cross_host_send_buffer(session, shard, buffer, layout, completion)?))
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
        session: &Session,
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
        let mut work = session.reserve_work();
        let copied = buffer.copy_to_memory(device.memory(self.r#type().memory())?)?;
        work.track_buffer(&copied);
        completion.track_buffer(&copied)?;
        Ok(Arc::new(copied))
    }

    /// Returns `buffer`, which is the buffer that this process sends to another process for `shard`, an addressable
    /// shard of this array (refer to [`Self::cross_host_source_buffer`]), in the default memory of the device of that
    /// shard and in `layout`, which is the default layout that the receiver allocates for its shape.
    ///
    /// A cross-host transfer of the CUDA plugin reads a device buffer and waits for its GPU completion events, which a
    /// buffer in host memory does not have (e.g., an unpinned host buffer is a plain CPU allocation). A shard of an
    /// array whose type declares another memory than [`Memory::Device`] is therefore first copied to the default
    /// memory of its device, and `completion` records the readiness of that copy. The declared memory is information
    /// that every process shares, so every process stages the same shards.
    ///
    /// A cross-host transfer also moves the raw bytes of the buffer, and some plugins (e.g., CUDA) take the number of
    /// bytes from the allocation of the sender, while the receiver allocates the exact shape of the shard. A buffer in
    /// another layout or whose physical dimensions differ from the shape of the shard (e.g., the padded buffer of a
    /// bounded-dynamic result, whose allocation covers the bound of a dimension) is therefore converted next into an
    /// exactly sized buffer in `layout`. The conversion runs on its device with [`Self::try_relayout_shard_on_device`]
    /// when this process can compile operations for its own shards and the conversion compiles, and `completion` then
    /// records its fence. Otherwise, the shard is copied through host memory into a buffer in `layout` in the default
    /// memory of its device with [`Self::copy_shard_through_host`], and `completion` records the readiness of that
    /// buffer.
    ///
    /// The physical layout and dimensions of a buffer are known only to the process that holds it when this array
    /// declares no layout, so this function never declines. The receivers of the shard chose the same strategy without
    /// knowing them, and they wait for this send whatever they are.
    fn cross_host_send_buffer(
        &self,
        session: &Session,
        shard: &ArrayShard,
        buffer: Arc<Buffer<'static>>,
        layout: &ryft_pjrt::Layout,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        let buffer = if self.r#type().memory() == Memory::Device {
            buffer
        } else {
            let mut work = session.reserve_work();
            let staged = buffer.copy_to_memory(buffer.device()?.default_memory()?)?;
            work.track_buffer(&staged);
            completion.track_buffer(&staged)?;
            Arc::new(staged)
        };
        let current_layout = buffer.specification()?.layout;
        let exactly_sized = holds_exact_shape(&buffer, shard.shape().as_slice())?;
        if exactly_sized && current_layout == *layout {
            return Ok(buffer);
        }
        if can_compile_local_operations(session) {
            let input_type =
                ArrayType::new(self.data_type(), shard.shape().into()).with_layout(Layout::from_pjrt(current_layout)?);
            let target_layout = Layout::from_pjrt(layout.clone())?;
            let converted = Self::try_relayout_shard_on_device(
                session,
                shard.device(),
                Arc::clone(&buffer),
                input_type,
                &target_layout,
            )?;
            if let Some(converted) = converted {
                completion.track_fence(converted.execution_fence());
                return Ok(Arc::clone(converted.shards()[0].buffer().unwrap()));
            }
        }
        let buffer_type = self.data_type().executable_buffer_type();
        let memory = buffer.memory()?;
        Self::copy_shard_through_host(
            session,
            &buffer,
            shard.shape(),
            buffer_type,
            memory,
            Some(layout.clone()),
            completion,
        )
    }

    /// Places this array on `mesh` according to `sharding` by slicing on the source devices and joining the received
    /// regions on the destination devices, which [`Self::device_placement_strategy`] chose with `plan` (i.e., the moves
    /// of the regions of every destination shard) and `remote_layouts` (i.e., the default layout of the shape of every
    /// region that crosses processes), both of which [`Self::shard_region_plan`] derived. Every process derives the
    /// same plan, and each one performs only the part of each move that [`ShardMove::role`] assigns to it.
    fn reshard_shard_regions(
        &self,
        session: &Session,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        plan: &ShardMovePlan,
        remote_layouts: &HashMap<StaticShape, ryft_pjrt::Layout>,
    ) -> Result<Self, Error> {
        let layout = ShardLayout::new(&self.shape(), mesh, sharding)?;
        let destinations = layout.descriptors();
        let client = session.client();
        let process_index = client.process_index()?;

        let devices = client
            .addressable_devices()?
            .into_iter()
            .map(|device| Ok((device.id()?, device)))
            .collect::<Result<HashMap<_, _>, PjrtError>>()?;
        let data_type = self.data_type();
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
                                session,
                                source.index(),
                                source_device,
                                process_index,
                                &mut completion,
                            )?;
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
        for movement in plan.moves_with_role(process_index, MoveRole::Copy) {
            let device = movement.destination_device();
            let (source, buffer) = self.local_replica(movement.source_shard_index(), device, process_index)?;
            let buffer = Arc::clone(buffer);
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
        let r#type = self.r#type().into_owned().with_sharding(None)?;
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
    fn plan_shard_regions(&self, layout: &ShardLayout) -> Result<ShardMovePlan, Error> {
        let mut shards_by_slices = HashMap::<&[Range<usize>], Vec<&ShardDescriptor>>::new();
        for shard in self.shards() {
            shards_by_slices.entry(shard.slice()).or_default().push(shard.descriptor());
        }
        let replicas = shards_by_slices
            .into_iter()
            .map(|(slice, shards)| (slice, SourceReplicas::new(shards.as_slice())))
            .collect::<HashMap<_, _>>();
        let destinations = layout.descriptors();
        let mut moves = Vec::new();
        for destination in destinations {
            let mut regions = Vec::new();
            for (source_slice, source_shards) in &replicas {
                if !source_slice
                    .iter()
                    .zip(destination.slice())
                    .all(|(source, destination)| source.start.max(destination.start) < source.end.min(destination.end))
                {
                    continue;
                }
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
    /// sharding can assign different values to the same slice, this array declares a layout or lives in a memory other
    /// than the default device memory, this process cannot compile operations for the shards of this array on its own
    /// (refer to [`PlacementEnvironment::compiles_local_operations`]), or the plugin of `environment` lacks the
    /// cross-host transfers extension or cannot report a default layout that a region which crosses processes needs.
    /// Every process derives the same moves and checks all of them, not only its own, so all processes decline alike,
    /// and the placement then executes these moves rather than deriving them again.
    fn shard_region_plan(
        &self,
        environment: &PlacementEnvironment,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<Option<(ShardMovePlan, HashMap<StaticShape, ryft_pjrt::Layout>)>, Error> {
        if !holds_one_value_per_slice(self.sharding())
            || !holds_one_value_per_slice(sharding)
            || self.r#type().memory() != Memory::Device
            || self.r#type().layout().is_some()
            || !environment.compiles_local_operations
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
        let layouts = cross_host_layouts(environment, plan.element_type(), remote_regions)?;
        Ok(layouts.map(|layouts| (plan, layouts)))
    }

    /// Produces the region with `slice`, in global array coordinates, from `source_buffer`, which holds the shard
    /// `source` on its device, on that device, using the default PJRT layout for a remote transfer. A region that a
    /// remote transfer sends is also exactly sized, because a cross-host send moves the whole allocation of its buffer
    /// (refer to [`Self::cross_host_send_buffer`]), so a whole source shard whose buffer is padded (e.g., a
    /// bounded-dynamic result) is converted as a buffer in another layout is, with a compiled conversion on its device
    /// or, when that conversion does not compile, through host memory. `completion` records the fences of the programs
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
        // A receive declares the region's shape and uses the plugin's default device layout.
        // An eager cast to the type of its input returns that input, so a padded buffer in the default layout is
        // converted by a compiled conversion, as `Self::cross_host_send_buffer` converts it, or through host memory.
        if let Some(layout) = remote_layout {
            let shape = StaticShape::new(slice.iter().map(Range::len).collect::<Vec<_>>());
            let current_layout = buffer.specification()?.layout;
            if current_layout != *layout || !holds_exact_shape(&buffer, shape.as_slice())? {
                let input_type =
                    ArrayType::new(data_type, shape.clone().into()).with_layout(Layout::from_pjrt(current_layout)?);
                let target_layout = Layout::from_pjrt(layout.clone())?;
                let converted = Self::try_relayout_shard_on_device(
                    session,
                    source.device(),
                    Arc::clone(&buffer),
                    input_type,
                    &target_layout,
                )?;
                buffer = match converted {
                    Some(converted) => {
                        completion.track_fence(converted.execution_fence());
                        Arc::clone(converted.shards()[0].buffer().unwrap())
                    }
                    None => {
                        let buffer_type = data_type.executable_buffer_type();
                        let memory = buffer.memory()?;
                        let layout = Some(layout.clone());
                        Self::copy_shard_through_host(session, &buffer, shape, buffer_type, memory, layout, completion)?
                    }
                };
            }
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
        let placement = ArrayPlacement::device(device).resolve(session, r#type.rank())?;
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
    ///   1. **Exact Shards:** [`Self::place_exact_shards`] shares, copies, or transfers whole shards and compiles
    ///      nothing. Its errors are final, so they come without this array.
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
    /// of this process alone and this process can compile operations of its own, the first of these strategies that
    /// applies places it, because that strategy then does local work only as well.
    ///
    /// Unsupported requests return typed [`ArrayError`] variants through [`Error::Array`] (refer to
    /// [`Self::validate_compiled_reshard`]), and cross-host extension, tracing, compilation, and execution failures
    /// propagate through [`Error`] without losing their typed payloads.
    ///
    /// A donated array must be the only handle of its buffers, so the compiled reshard on the mesh of this array
    /// consumes this array when `donate` is `true`. This array is then not returned if that execution fails.
    ///
    /// `layout` is the layout of each destination shard that the placement records (refer to
    /// [`Self::placement_layout`]), which [`Self::place_process_local_value`] uploads into directly.
    // The error carries the unconsumed array back to the caller, which falls back to a host transfer with it.
    #[allow(clippy::result_large_err)]
    fn reshard_on_devices_or_return(
        self,
        session: &Session,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        memory: Memory,
        layout: Option<&Layout>,
        donate: bool,
    ) -> Result<Self, (Error, Option<Self>)> {
        let environment = PlacementEnvironment::new(session, &self);
        let strategy = session
            .client()
            .process_index()
            .map_err(Error::from)
            .and_then(|process_index| self.device_placement_strategy(&environment, mesh, sharding, process_index));
        match strategy {
            Ok(strategy) => self.place_with_strategy(session, strategy, mesh, sharding, memory, layout, donate),
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
        session: &Session,
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
                    let mut placed = Self::zero(session, r#type, Some(sharding.clone()), mesh.clone())?;
                    placed.execution_fence = self.execution_fence().cloned();
                    Ok::<_, Error>(placed)
                };
                place().map_err(|error| (error, None))
            }
            DevicePlacementStrategy::ProcessLocalValue => {
                match self.place_process_local_value(session, mesh, sharding, memory, layout) {
                    Ok(output) => Ok(output),
                    Err(error) => Err((error, Some(self))),
                }
            }
            DevicePlacementStrategy::ExactShards { plan, cross_host_layouts } => self
                .place_exact_shards(session, mesh, sharding, memory, &plan, &cross_host_layouts)
                .map_err(|error| (error, None)),
            DevicePlacementStrategy::ShardRegions { plan, cross_host_layouts } => {
                match self.reshard_shard_regions(session, mesh, sharding, &plan, &cross_host_layouts) {
                    Ok(output) => Ok(output),
                    Err(error) => Err((error, Some(self))),
                }
            }
            DevicePlacementStrategy::CompiledReshard { program } => {
                let compiled = self
                    .validate_addressable_shard_buffers(session)
                    .and_then(|()| self.compile_reshard_on_same_mesh(session, &program, donate));
                match compiled {
                    Ok(reshard) => {
                        let source = (!donate).then(|| self.clone());
                        self.execute_reshard(session, &program, reshard.as_ref()).map_err(|error| (error, source))
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
                    session,
                    gather.as_ref(),
                    &replicas,
                    &cross_host_layouts,
                    &reshard,
                    intermediate_memory,
                )
                .map_err(|error| (error, Some(self)))
            }
        }
    }

    /// Chooses the device work that places this array on `mesh` according to `sharding` in the process with index
    /// `process_index`, or returns the error that rules out all of it. `environment` holds everything else that the
    /// choice reads. [`Self::reshard_on_devices_or_return`] executes the chosen strategy, and the agreement test of
    /// this module plans every process of a distributed placement with this same function.
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
    ///   - **Environment:** The compilation configuration of the session and the capabilities of its PJRT plugin, which
    ///     every process loads alike (refer to [`PlacementEnvironment`]).
    ///
    /// It never reads a buffer or a physical layout. It reads `process_index` only for a locally assigned array.
    ///
    /// An array with locally assigned devices holds a full value that every process has, each on devices of its own
    /// choice, as JAX assumes for uncommitted arguments of a computation over several processes. Its devices are the
    /// one part of its metadata that the processes do not share, so it is placed with
    /// [`DevicePlacementStrategy::ProcessLocalValue`], in which every process places its own value on its own
    /// destination devices with local work only, and no process plans a transfer or a collective. Every process knows
    /// the fact for its own array and computes the same one (refer to [`Self::is_locally_assigned`]). There is
    /// one refinement. When every destination device belongs to this process and this process can compile operations of
    /// its own, such an array takes the strategy that [`Self::shared_device_placement_strategy`] chooses for it, if one
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
        environment: &PlacementEnvironment,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        process_index: usize,
    ) -> Result<DevicePlacementStrategy, Error> {
        if self.data_type().is_zero() {
            return Ok(DevicePlacementStrategy::Bufferless);
        }
        if !self.is_locally_assigned() {
            return self.shared_device_placement_strategy(environment, mesh, sharding);
        }

        // Placing the value with local work only also handles every placement that no shared strategy supports (e.g.,
        // an uneven split of a value in host memory), so it remains the choice when none applies.
        let local_destination = mesh.devices().iter().all(|device| device.process_index() == process_index);
        if environment.compiles_local_operations
            && local_destination
            && let Ok(strategy) = self.shared_device_placement_strategy(environment, mesh, sharding)
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
    ///   3. [`DevicePlacementStrategy::CompiledReshard`] on the mesh of this array, or
    ///      [`DevicePlacementStrategy::CompiledReshardAcrossMeshes`] to another mesh, whose replicas move as whole
    ///      shards of the replicated value (refer to [`plan_exact_shard_put`]).
    fn shared_device_placement_strategy(
        &self,
        environment: &PlacementEnvironment,
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
            if let Some(cross_host_layouts) = cross_host_layouts(environment, element_type, cross_host_shards)? {
                return Ok(DevicePlacementStrategy::ExactShards { plan, cross_host_layouts });
            }
        }

        let compiled = self.validate_compiled_reshard(mesh, sharding);
        let try_regions = match &compiled {
            Ok(()) => self.mesh() != mesh,
            Err(Error::Array(error)) => {
                matches!(
                    error.as_ref(),
                    ArrayError::UnevenReshardingShards { .. } | ArrayError::ManualReshardingAxis { .. }
                )
            }
            Err(_) => false,
        };
        if try_regions && let Some((plan, cross_host_layouts)) = self.shard_region_plan(environment, mesh, sharding)? {
            return Ok(DevicePlacementStrategy::ShardRegions { plan, cross_host_layouts });
        }
        compiled?;
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
            let layout = (environment.cross_host_layout)(element_type, movement.destination_shard_index(), &shape)?
                .ok_or_else(|| {
                    Error::Pjrt(PjrtError::unimplemented(
                        "the PJRT plugin does not report the default layout that a cross-host receive allocates",
                    ))
                })?;
            cross_host_layouts.insert(shape, layout);
        }
        let reshard = ReshardProgram { mesh: mesh.clone(), input: replicated_on_destination, output: sharding.clone() };
        Ok(DevicePlacementStrategy::CompiledReshardAcrossMeshes { gather, replicas, cross_host_layouts, reshard })
    }

    /// Checks that a compiled program can place this array on `destination_mesh` with `destination_sharding`. The
    /// errors are [`ArrayError::ManualReshardingAxis`] when the destination mesh has a `Manual` axis, while `Auto` and
    /// `Explicit` axes are supported, and [`ArrayError::UnevenReshardingShards`] when the shards of this array or of
    /// the destination differ in size. These checks read only metadata that every process shares (refer to
    /// [`Self::device_placement_strategy`]).
    ///
    /// The SPMD partitioner splits a dimension into parts of one size and pads the last parts, so a dimension whose
    /// extent the number of its parts does not divide (e.g., five elements over two devices) yields buffers whose
    /// shapes differ from the exact shards that [`ShardLayout`] describes (i.e., `[3]` and `[2]`). Such a placement
    /// can use whole shards or shard regions, both of which materialize those exact shapes.
    fn validate_compiled_reshard(
        &self,
        destination_mesh: &DeviceMesh,
        destination_sharding: &Sharding,
    ) -> Result<(), Error> {
        // `Manual` axes are managed explicitly by the user (e.g., inside `shard_map`) and cannot be planned by the SPMD
        // partitioner from the top level. `Auto` and `Explicit` axes both go through SPMD partitioning, so both are
        // allowed.
        if let Some(axis) =
            destination_mesh.logical_mesh().axes().iter().find(|axis| axis.r#type() == MeshAxisType::Manual)
        {
            return Err(ArrayError::ManualReshardingAxis { axis_name: axis.name().to_string() }.into());
        }

        // The shards of one sharding have equal shapes exactly when every split divides its dimension evenly.
        let shape = self.shape();
        let destination_layout = ShardLayout::new(&shape, destination_mesh, destination_sharding)?;
        let has_uneven_shards = |shapes: Vec<StaticShape>| shapes.iter().any(|shape| shape != &shapes[0]);
        if has_uneven_shards(self.shards().iter().map(ArrayShard::shape).collect()) {
            return Err(ArrayError::UnevenReshardingShards { shape, sharding: self.sharding().clone() }.into());
        }
        if has_uneven_shards(destination_layout.descriptors().iter().map(ShardDescriptor::shape).collect()) {
            return Err(ArrayError::UnevenReshardingShards { shape, sharding: destination_sharding.clone() }.into());
        }
        Ok(())
    }

    /// Checks that every shard of this array on the process of `session` has a buffer, which a compiled program whose
    /// input is this array reads (i.e., a compiled reshard on the mesh of this array or the gather of a reshard across
    /// meshes), and returns [`ArrayError::MissingShardBuffer`] for the first one that does not. Shards on other
    /// processes are normal, and the program reads them on their own processes. Only this process knows which of its
    /// buffers exist, so this check runs only where such a program reads this array, right before it, rather than as
    /// part of the choice of [`Self::device_placement_strategy`]. A failure stops this process before the program, and
    /// its peers then wait in the collectives of that program and in the transfers that follow it.
    fn validate_addressable_shard_buffers(&self, session: &Session) -> Result<(), Error> {
        let client_process_index = session.client().process_index()?;
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
    /// [`ReshardProgram::runs_on`]). The input and the output of the program are declared in the memory of this array,
    /// so the program accepts this array and returns its result in the same memory. If `donate` is `true`, the input of
    /// the program is marked as donated so that PJRT can reuse the memory of its buffers for the output buffers.
    ///
    /// The compilation cache is keyed by the complete lowered computation, including its input and output sharding
    /// annotations and PJRT options.
    fn compile_reshard_on_same_mesh(
        &self,
        session: &Session,
        program: &ReshardProgram,
        donate: bool,
    ) -> Result<Option<CompiledFunction<ArrayType, ArrayType>>, Error> {
        if !program.runs_on(session.client().process_index()?) {
            return Ok(None);
        }

        // Build the bare input type, which has no sharding, for tracing. The input and output shardings are attached as
        // `sdy.sharding` attributes through `Options::with_input_shardings` and `Options::with_output_shardings`. The
        // type keeps the memory of this array, which the identity carries over to the output.
        let bare_input_type = ArrayType::new(self.data_type(), self.shape().into()).with_memory(self.r#type().memory());
        let options = Options::new(program.mesh.clone())
            .with_input_shardings(vec![program.input.clone()])
            .with_output_shardings(vec![program.output.clone()])
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
    fn execute_reshard(
        self,
        session: &Session,
        program: &ReshardProgram,
        reshard: Option<&CompiledFunction<ArrayType, ArrayType>>,
    ) -> Result<Self, Error> {
        let mut completion = PlacementCompletion::default();
        completion.track_fence(self.execution_fence());
        let mut output = match reshard {
            Some(reshard) => reshard.executable().execute(session, self)?,
            None => {
                let r#type = ArrayType::new(self.data_type(), self.shape().into()).with_memory(self.r#type().memory());
                let (mesh, sharding) = (program.mesh.clone(), Some(program.output.clone()));
                Self::from_addressable_buffers_internal(session, r#type, sharding, mesh, Vec::new(), false)?
            }
        };
        output.execution_fence = completion.join(output.execution_fence.take());
        Ok(output)
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
        session: &Session,
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
                self.validate_addressable_shard_buffers(session)?;
                let compiled = self.compile_reshard_on_same_mesh(session, gather, false)?;
                self.clone().execute_reshard(session, gather, compiled.as_ref())?
            }
            None => self.clone(),
        };
        let copies = replicated.place_exact_shards(
            session,
            &reshard.mesh,
            &reshard.input,
            memory,
            replicas,
            cross_host_layouts,
        )?;
        drop(replicated);

        // The copies are owned exclusively by this function and are never observed by callers, so donating them lets
        // PJRT reuse their memory for the output of the final reshard whenever they share no buffer with this array.
        let send_fence = copies.send_fence.clone();
        let compiled = copies.compile_reshard_on_same_mesh(session, reshard, true)?;
        let output = copies.execute_reshard(session, reshard, compiled.as_ref())?;
        Ok(match send_fence {
            Some(fence) => output.with_send_fence(fence),
            None => output,
        })
    }
}

/// Device work that places an [`Array`] on a mesh with a sharding, which [`Array::device_placement_strategy`] chooses
/// from information that every process shares, so that the processes of a placement agree on it.
enum DevicePlacementStrategy {
    /// Changes only the placement metadata of an array of [`DataType::Zero`](ryft_core::DataType::Zero), which has
    /// no buffers.
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
}

/// Everything besides the source and the destination of a placement that [`Array::device_placement_strategy`] reads,
/// which is information that every process of the placement shares (i.e., the compilation configuration of the
/// session and the capabilities of its PJRT plugin, which every process loads alike).
struct PlacementEnvironment {
    /// Whether this process can compile operations for the shards of the source on its own devices without the other
    /// processes, which needs a source of the client of the session and a session that does not exchange compilation
    /// artifacts with other processes (refer to [`can_compile_local_operations`]).
    compiles_local_operations: bool,

    /// Function that returns the default layout of the PJRT plugin for a buffer of an element type with a shape, which
    /// is the layout of the buffer that a cross-host receive of that shape allocates (refer to
    /// [`cross_host_default_layout`], which also takes the index of the destination shard for its errors). It returns
    /// `None` when the plugin cannot report that layout, and the [`PjrtError::Unimplemented`] error of a plugin without
    /// the cross-host transfers extension.
    cross_host_layout: Box<dyn Fn(BufferType, ShardIndex, &StaticShape) -> Result<Option<ryft_pjrt::Layout>, Error>>,
}

impl PlacementEnvironment {
    /// Creates the [`PlacementEnvironment`] of a placement of `source` in `session`.
    fn new(session: &Session, source: &Array) -> Self {
        let client = session.client();
        Self {
            compiles_local_operations: std::ptr::eq(source.client(), client) && can_compile_local_operations(session),
            cross_host_layout: Box::new(move |element_type, shard_index, shape| {
                client.cross_host_transfers_extension()?;
                cross_host_default_layout(client, element_type, shard_index, shape)
            }),
        }
    }
}

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

/// Returns the default layout of the PJRT plugin of `environment` for buffers of `element_type` with each shape of
/// `shards` (i.e., of pairs of the index of a destination shard and a shape that crosses processes for it), or `None`
/// when `shards` is not empty and the plugin lacks the cross-host transfers extension or cannot report one of those
/// layouts. Every process derives the same `shards` for a placement and queries the same plugin, so all of them get the
/// same result.
fn cross_host_layouts(
    environment: &PlacementEnvironment,
    element_type: BufferType,
    shards: Vec<(ShardIndex, StaticShape)>,
) -> Result<Option<HashMap<StaticShape, ryft_pjrt::Layout>>, Error> {
    let mut layouts = HashMap::new();
    for (shard_index, shape) in shards {
        if let std::collections::hash_map::Entry::Vacant(entry) = layouts.entry(shape) {
            match (environment.cross_host_layout)(element_type, shard_index, entry.key()) {
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

/// Reads the shard of `buffer` with `shape` back in dense row-major order. `wait` waits for the copy to the host and
/// returns its bytes. An empty shard issues no copy, and this function only waits for its buffer so that the error of
/// that buffer still surfaces.
fn read_shard_to_host(
    buffer: &Buffer<'static>,
    shape: &StaticShape,
    wait: impl FnOnce(Event<Vec<u8>>) -> Result<Vec<u8>, PjrtError>,
) -> Result<Vec<u8>, Error> {
    if shape.as_slice().contains(&0) {
        buffer.ready()?.r#await()?;
        return Ok(Vec::new());
    }
    let host_layout = ryft_pjrt::Layout::dense_major_to_minor(shape.rank());
    Ok(wait(buffer.copy_to_host(Some(host_layout))?)?)
}

/// Returns whether this process can compile operations for its own shards independently. With an artifact exchange,
/// all processes must request the same compilations, while local shard operations can differ on each process.
fn can_compile_local_operations(session: &Session) -> bool {
    let cache = session.compilation_cache();
    cache.artifact_exchange_policy().timeout().is_none()
        || cache.artifact_exchange().is_none_or(|exchange| exchange.process_count() <= 1)
}

/// Returns whether the physical dimensions of `buffer` are `shape`, which is the shape of the shard or the region that
/// it holds. A buffer that a bounded-dynamic execution produced is padded to the bound of each dynamic dimension, so
/// its allocation is larger than that shape, although it holds only that shape and copies and readbacks preserve that.
fn holds_exact_shape(buffer: &Buffer<'_>, shape: &[usize]) -> Result<bool, Error> {
    Ok(buffer.dimensions()?.iter().map(|&size| size as usize).eq(shape.iter().copied()))
}

/// Returns whether replicas of each shard slice hold one value across the mesh.
fn holds_one_value_per_slice(sharding: &Sharding) -> bool {
    sharding.unreduced_axes().is_empty() && sharding.varying_manual_axes().is_empty()
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

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        CompilationArtifactExchange, CompilationArtifactExchangeError, CompilationArtifactExchangePolicy,
        CompilationCache, DataType, Dimension, DimensionBounds, DimensionError, DimensionVariable, Error as CoreError,
        Layout, LogicalMesh, Memory, MeshAxis, Shape, ShardingDimension, StridedLayout, TiledLayout,
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

    // Compilation artifact exchange with one fictitious peer process, which shard-local conversions must not use.
    struct PeerCompilationExchange;

    impl CompilationArtifactExchange for PeerCompilationExchange {
        fn process_index(&self) -> usize {
            0
        }

        fn process_count(&self) -> usize {
            2
        }

        fn publish(&self, _: &[u8], _: &[u8]) -> Result<(), CompilationArtifactExchangeError> {
            panic!("a shard-local conversion must not publish an artifact");
        }

        fn receive(&self, _: &[u8], _: Duration) -> Result<Option<Vec<u8>>, CompilationArtifactExchangeError> {
            panic!("a shard-local conversion must not wait for a peer");
        }
    }

    // Returns a session of `client` whose compilation cache must share every artifact with a peer process. Such a
    // session cannot compile operations for the shards of one process, so its layout changes go through host memory.
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

    // Returns the placement environment of a process whose PJRT plugin has the cross-host transfers extension and
    // reports row-major default layouts, as the CUDA plugin does, and which compiles operations for its own shards.
    fn simulated_placement_environment() -> PlacementEnvironment {
        PlacementEnvironment {
            compiles_local_operations: true,
            cross_host_layout: Box::new(|_, _, shape| {
                let minor_to_major = (0..shape.rank()).rev().collect();
                Ok(Some(Layout::Tiled(TiledLayout::new(minor_to_major, vec![])).to_pjrt()?))
            }),
        }
    }

    // Returns the work that crosses processes, which the process with index `process_index` plans when it places
    // `source`, its own view of the source of a placement, on `mesh` according to `sharding`. The strategy, with the
    // moves and programs of every process, comes from `Array::device_placement_strategy`, which the placement executes,
    // and the part of this process in them from `ShardMove::role` and `ReshardProgram::runs_on`, which the placement
    // follows as well, so this is the work that the placement performs.
    fn planned_work(
        source: &Array,
        process_index: usize,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<PlannedWork, String> {
        let environment = simulated_placement_environment();
        let strategy = source
            .device_placement_strategy(&environment, mesh, sharding, process_index)
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
    // process of `session` is process 0, so when its view holds buffers and it plans transfers, it also runs its part
    // of the planned strategy with the production path. The CPU plugin lacks the cross-host transfers extension, so
    // that part must fail with exactly the error of its first transfer, and any other outcome is a disagreement.
    fn placement_agreement(
        session: &Session,
        views: &[Array],
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<String, String> {
        let works = views
            .iter()
            .enumerate()
            .map(|(process_index, view)| planned_work(view, process_index, mesh, sharding))
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
            let strategy = source.device_placement_strategy(&simulated_placement_environment(), mesh, sharding, 0);
            let layout = source.placement_layout(None, mesh, sharding).unwrap();
            let memory = source.r#type().memory();
            let placed = strategy.and_then(|strategy| {
                let placed = source.clone().place_with_strategy(
                    session,
                    strategy,
                    mesh,
                    sharding,
                    memory,
                    layout.as_ref(),
                    false,
                );
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
            .to_placement(&session, ArrayPlacement::mesh(target_mesh.clone(), target_sharding).unwrap())
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
            let target = ArrayPlacement::mesh(mesh.clone(), sharded.clone()).unwrap();
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
        let unchanged = ArrayPlacement::mesh(
            first_device_mesh.clone(),
            Sharding::replicated(first_device_mesh.logical_mesh().clone(), 1),
        )
        .unwrap();
        let copied = ArrayPlacement::mesh(mesh.clone(), replicated).unwrap();
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
        let target = ArrayPlacement::mesh(first_device_mesh.clone(), other_array.sharding().clone()).unwrap();
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
            let placed = array.to_placement(&session, ArrayPlacement::device(device)).unwrap();
            assert_eq!(placed.mesh().devices(), [device]);
            assert!(placed.is_committed());
            let donated = array.clone().into_placement(&session, ArrayPlacement::device(device)).unwrap();
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
                    ArrayPlacement::in_place().with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![]))),
                )
                .unwrap(),
        ] {
            assert_eq!(placed.mesh(), &own_mesh);
            assert!(placed.is_committed());
            assert!(placed.is_locally_assigned());
        }

        // A destination that names devices or a sharding names the devices of the result, even when they are the
        // devices of the source.
        let same_devices = ArrayPlacement::mesh(own_mesh.clone(), source.sharding().clone()).unwrap();
        let other_devices = ArrayPlacement::mesh(
            two_device_sub_mesh_x(client),
            Sharding::replicated(two_device_sub_mesh_x(client).logical_mesh().clone(), 1),
        )
        .unwrap();
        for target in [same_devices, other_devices, ArrayPlacement::device(own_mesh.devices()[0])] {
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
            source_array.to_placement(&session, ArrayPlacement::mesh(mesh.clone(), sharding).unwrap()).unwrap();
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
        let target = ArrayPlacement::device(devices[1]);
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
            let placed = source.to_placement(&session, ArrayPlacement::mesh(reversed.clone(), rows.clone()).unwrap());
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
        let error = source.to_placement(&session, ArrayPlacement::mesh(reversed, sharded).unwrap()).unwrap_err();
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
        let target = ArrayPlacement::mesh(destination_mesh.clone(), split.clone()).unwrap();

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
            .to_placement(&session, ArrayPlacement::mesh(target_mesh, target_sharding).unwrap())
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
        let reduced =
            summands.to_placement(&session, ArrayPlacement::mesh(mesh.clone(), tiled.clone()).unwrap()).unwrap();
        assert_eq!(reduced.sharding(), &tiled);
        assert_eq!(shard_values(&reduced), vec![vec![11., 22.], vec![33., 44.], vec![11., 22.], vec![33., 44.]]);

        // In the other direction, a copy of every shard would hold the logical value once per device along `x`.
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(tiled).unwrap();
        let value = Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
            .unwrap();
        let summands = value.to_placement(&session, ArrayPlacement::mesh(mesh, unreduced.clone()).unwrap()).unwrap();
        assert_eq!(summands.sharding(), &unreduced);
        assert_eq!(summands.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2., 3., 4.]);
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
        let summands = value.to_placement(&session, ArrayPlacement::mesh(mesh, unreduced.clone()).unwrap()).unwrap();
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
                source.to_placement(&session, ArrayPlacement::mesh(mesh.clone(), sharding).unwrap()).unwrap()
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
            let placed = source
                .into_placement(&session, ArrayPlacement::mesh(mesh.clone(), sharded(&mesh)).unwrap())
                .unwrap();
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
                ArrayPlacement::mesh(reversed.clone(), sharded.clone()).unwrap().with_memory(Memory::Device),
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
            .into_placement(&session, ArrayPlacement::mesh(reversed.clone(), sharded).unwrap().with_memory(unpinned))
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

        // The source is in pinned host memory, which rules out shard regions, and its second replica comes from a copy
        // that fails. A compiled reshard to a split over another mesh of the same devices copies each replica to the
        // device memory of its own device and reads each copy on its device, so the failed replica fails both its copy
        // and the program that reads that copy. The fence of the result reports the first failure that it observes,
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
        let (source, managers) = pending_replicated_vector(&session, &mesh);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let pinned = Placement::new(mesh, replicated).unwrap().with_memory(Memory::Host { pinned: true });
        let source = source.to_placement(&session, pinned).unwrap();
        let sharded =
            Sharding::new(other_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let target = Placement::new(other_mesh.clone(), sharded.clone()).unwrap().with_memory(Memory::Device);
        let placed = source.to_placement(&session, target).unwrap();
        assert_eq!(placed.mesh(), &other_mesh);
        assert_eq!(placed.sharding(), &sharded);
        assert_eq!(shard_memory_kinds(&placed), ["device"; 2]);
        assert_eq!(session.cached_program_count(), 1);
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
        // copies that read the failed replica report its cause as the copy reports it, and the executions that read
        // those copies report it as the failed execution reports it.
        let Err(Error::Execution(ExecutionError::SynchronizationFailures { failures })) = session.synchronize() else {
            panic!("expected the session to report the failed replica");
        };
        let failures = failures
            .into_iter()
            .map(|(identity, error)| (identity, failure(Err(error)).unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            failures,
            [
                (0, replica_failures[0].clone().unwrap()),
                (1, replica_failures[0].clone().unwrap()),
                (2, replica_failures[1].clone().unwrap()),
                (3, replica_failures[1].clone().unwrap()),
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
        let target = ArrayPlacement::mesh(mesh, sharding.clone()).unwrap().with_layout(layout.clone());
        let placed = source.to_placement(&session, target.clone()).unwrap();
        // The fallback cannot use the peer exchange because each process only relayouts its own shards.
        let fallback = source.to_placement(&shared_session, target).unwrap();
        assert!(session.cached_program_count() > 0);
        assert_eq!(shared_session.cached_program_count(), 0);

        for output in [&placed, &fallback] {
            assert_eq!(output.sharding(), &sharding);
            assert_eq!(output.layout(), Some(&layout));
            assert!(output.device_shard(remote_device.id()).unwrap().buffer().is_none());
            assert!(!output.execution_fence().unwrap().is_ready().unwrap());
            for (shard, summand) in output.addressable_shards().zip(summands) {
                assert_eq!(shard.buffer().unwrap().specification().unwrap().layout, layout.to_pjrt().unwrap());
                assert_eq!(values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()), summand);
            }
        }

        // Both placement paths retain the pending producer and the source's asynchronous sends.
        producer.set(None).unwrap();
        send.set(Some(PjrtError::aborted("source transfer failed"))).unwrap();
        for output in [placed, fallback] {
            assert_eq!(output.block_until_ready().unwrap_err().to_string(), "source transfer failed");
        }
    }

    #[test]
    fn test_array_to_placement_relayouts_shards_through_host_in_background() {
        let client = cpu_client(2);
        let session = peer_exchange_session(client);
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();

        // Transfer managers provide source buffers whose contents stay undefined until the test transfers them, so
        // the shard readbacks of the placement cannot complete before then.
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
        let buffers = managers.iter().map(|manager| manager.retrieve_buffer(0).unwrap()).collect();
        let r#type = ArrayType::new_static(DataType::F32, [4, 2]);
        let source =
            Array::from_addressable_buffers(&session, r#type, mesh.clone(), Some(sharding.clone()), buffers).unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let placed = source
            .to_placement(
                &session,
                ArrayPlacement::mesh(mesh.clone(), sharding.clone()).unwrap().with_layout(layout.clone()),
            )
            .unwrap();

        // The placement returns the destination buffers before their uploads complete.
        assert_eq!(session.cached_program_count(), 0);
        assert_eq!(placed.data_type(), DataType::F32);
        assert_eq!(placed.shape(), StaticShape::new(vec![4, 2]));
        assert_eq!(placed.mesh(), &mesh);
        assert_eq!(placed.sharding(), &sharding);
        assert_eq!(placed.r#type().memory(), Memory::Device);
        assert_eq!(placed.layout(), Some(&layout));
        for shard in placed.addressable_shards() {
            let buffer = shard.buffer().unwrap();
            assert_eq!(buffer.specification().unwrap().layout, layout.to_pjrt().unwrap());
            assert_eq!(buffer.memory().unwrap(), buffer.device().unwrap().default_memory().unwrap());
            assert_eq!(buffer.ready().unwrap().ready(), Ok(false));
        }

        // Once the sources are defined, the background thread reads them back and uploads them in the new layout.
        for (manager, values) in managers.iter().zip([[1f32, 2., 3., 4.], [5., 6., 7., 8.]]) {
            manager.transfer_data(0, Arc::new(values_to_bytes(&values)), 0, true).unwrap().r#await().unwrap();
        }
        placed.block_until_ready().unwrap();
        assert_eq!(shard_values(&placed), vec![vec![1., 2., 3., 4.], vec![5., 6., 7., 8.]]);
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_to_placement_relayouts_shards_through_host_in_background_poisons_result_on_readback_failure() {
        let client = cpu_client(2);
        let session = peer_exchange_session(client);
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
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
        let buffers = managers.iter().map(|manager| manager.retrieve_buffer(0).unwrap()).collect();
        let r#type = ArrayType::new_static(DataType::F32, [4, 2]);
        let source =
            Array::from_addressable_buffers(&session, r#type, mesh.clone(), Some(sharding.clone()), buffers).unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let placed = source
            .to_placement(&session, ArrayPlacement::mesh(mesh, sharding).unwrap().with_layout(layout))
            .unwrap();

        // The readback of the first shard fails, which poisons its destination buffer with its error, while the
        // independent relayout of the second shard completes. The result reports the failure through its fence.
        managers[0].set_error(0, PjrtError::aborted("source upload failed")).unwrap();
        managers[1]
            .transfer_data(0, Arc::new(values_to_bytes(&[5f32, 6., 7., 8.])), 0, true)
            .unwrap()
            .r#await()
            .unwrap();
        assert!(matches!(
            placed.shards()[0].buffer().unwrap().ready().unwrap().r#await(),
            Err(PjrtError::Aborted { message, .. }) if message == "source upload failed",
        ));
        assert_eq!(values_from_bytes::<f32>(&placed.shards()[1].copy_to_host_bytes().unwrap()), [5., 6., 7., 8.]);
        assert!(matches!(
            placed.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "source upload failed",
        ));
        assert!(matches!(
            session.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(_, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "source upload failed"),
        ));
    }

    #[test]
    fn test_array_to_placement_relayouts_shards_through_host_in_background_fails_readback_of_failed_replica() {
        let client = cpu_client(2);
        let session = peer_exchange_session(client);
        let mesh = domain_mesh(client, "x", 2);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let target = ArrayPlacement::mesh(mesh.clone(), replicated.clone()).unwrap().with_layout(layout);

        // Only the relayout of the second replica fails, and readback reads one replica of each tile. The result must
        // still report the failure, also when it has no elements and carries the fence of an earlier producer.
        for (shape, values) in [([2u64, 2], vec![1f32, 2., 3., 4.]), ([0, 2], Vec::new())] {
            let managers = client
                .addressable_devices()
                .unwrap()
                .iter()
                .map(|device| {
                    let specification = BufferSpecification::new(BufferType::F32, shape);
                    client
                        .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let buffers = managers.iter().map(|manager| manager.retrieve_buffer(0).unwrap()).collect();
            let r#type = ArrayType::new_static(DataType::F32, shape.map(|extent| extent as usize));
            let source =
                Array::from_addressable_buffers(&session, r#type, mesh.clone(), Some(replicated.clone()), buffers)
                    .unwrap()
                    .with_execution_fence(ExecutionFence::new(Vec::new()));
            let placed = source.to_placement(&session, target.clone()).unwrap();
            managers[0]
                .transfer_data(0, Arc::new(values_to_bytes(&values)), 0, true)
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
    fn test_array_to_placement_relayouts_shards_through_host_in_background_independently() {
        let client = cpu_client(2);
        let session = peer_exchange_session(client);
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
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
        let buffers = managers.iter().map(|manager| manager.retrieve_buffer(0).unwrap()).collect();
        let r#type = ArrayType::new_static(DataType::F32, [4, 2]);
        let source =
            Array::from_addressable_buffers(&session, r#type, mesh.clone(), Some(sharding.clone()), buffers).unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let placed = source
            .to_placement(&session, ArrayPlacement::mesh(mesh, sharding).unwrap().with_layout(layout))
            .unwrap();

        // The relayout of the first shard completes while the source of the second one is still pending, so that a
        // producer of that source may depend on the first destination shard.
        let (sender, receiver) = mpsc::channel();
        let first_shard_ready = placed.shards()[0].buffer().unwrap().ready().unwrap();
        thread::spawn(move || sender.send(first_shard_ready.r#await()).unwrap());
        managers[0]
            .transfer_data(0, Arc::new(values_to_bytes(&[1f32, 2., 3., 4.])), 0, true)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(receiver.recv_timeout(Duration::from_secs(60)), Ok(Ok(())));
        assert_eq!(placed.execution_fence().unwrap().is_ready(), Ok(false));

        // The result still joins the relayout of the second shard, and with it the failure of its source.
        managers[1].set_error(0, PjrtError::aborted("second source failed")).unwrap();
        assert_eq!(values_from_bytes::<f32>(&placed.shards()[0].copy_to_host_bytes().unwrap()), [1., 2., 3., 4.]);
        assert!(matches!(
            placed.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second source failed",
        ));
        assert!(matches!(
            session.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(_, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "second source failed"),
        ));
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
            .to_placement(&session, ArrayPlacement::mesh(mesh.clone(), target_sharding.clone()).unwrap())
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
            let target = ArrayPlacement::mesh(mesh.clone(), sharding).unwrap().with_layout(layout.clone());
            let error = source.to_placement(&session, target).unwrap_err();
            assert!(matches!(
                &error,
                Error::Core(error) if **error == CoreError::Type(TypeError::Invalid { message: message.to_string() }),
            ));
            assert_eq!(error.to_string(), message);
        }

        // A layout of the right rank is recorded on the bufferless result.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], Vec::new()));
        let target = ArrayPlacement::mesh(mesh.clone(), source.sharding().clone()).unwrap().with_layout(layout.clone());
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
        let target = ArrayPlacement::mesh(mesh.clone(), columns.clone()).unwrap().with_layout(strided(vec![8, 4]));
        assert_eq!(
            source.to_placement(&session, target).unwrap_err().to_string(),
            "PJRT_Buffer_MemoryLayout_Type_Strides is not supported to be converted to a xla::Layout.",
        );

        // The same strides overlap the rows of a replicated shard, and strides that overlap a column shard are
        // rejected.
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        for (sharding, strides, axis) in [(replicated, vec![8, 4], 0), (columns, vec![4, 8], 1)] {
            let target = ArrayPlacement::mesh(mesh.clone(), sharding).unwrap().with_layout(strided(strides));
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
        let target = ArrayPlacement::mesh(target_mesh.clone(), target_sharding.clone()).unwrap();

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
            .to_placement(&session, ArrayPlacement::mesh(target_mesh.clone(), target_sharding).unwrap())
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
        let resharded = source_array
            .to_placement(&session, ArrayPlacement::mesh(mesh.clone(), sharded_target).unwrap())
            .unwrap();

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
        let target = ArrayPlacement::mesh(mesh.clone(), sharded.clone()).unwrap();

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
            .to_placement(&session, ArrayPlacement::mesh(mesh.clone(), replicated_target).unwrap())
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
        let resharded = source_array
            .to_placement(&session, ArrayPlacement::mesh(mesh.clone(), sharded_along_y).unwrap())
            .unwrap();
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
        let resharded = source_array
            .to_placement(&session, ArrayPlacement::mesh(explicit_mesh, sharded_target).unwrap())
            .unwrap();
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
            .to_placement(&session, ArrayPlacement::mesh(target_mesh.clone(), target_sharding).unwrap())
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
            .to_placement(&session, ArrayPlacement::mesh(target_mesh.clone(), target_sharding).unwrap())
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
            .to_placement(&session, ArrayPlacement::mesh(target_mesh.clone(), target_sharding).unwrap())
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
                &PlacementEnvironment::new(&session, &source),
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
            let target = ArrayPlacement::mesh(destination([local[0], local[1]]), sharding.clone()).unwrap();
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
            let target = ArrayPlacement::mesh(destination(remote), sharding.clone()).unwrap();
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
            &PlacementEnvironment::new(&session, &source),
            &target_mesh,
            &target_sharding,
            process_index,
        ) else {
            panic!("a placement on another mesh of this process must use shard regions");
        };
        assert!(cross_host_layouts.is_empty());
        assert_eq!(plan, regions);

        let output =
            source.to_placement(&session, ArrayPlacement::mesh(target_mesh, target_sharding).unwrap()).unwrap();
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
            .to_placement(&session, ArrayPlacement::mesh(mesh.clone(), sharded_target.clone()).unwrap())
            .unwrap();
        let after_first = session.cached_program_count();
        let second = source_array
            .to_placement(&session, ArrayPlacement::mesh(mesh.clone(), sharded_target).unwrap())
            .unwrap();
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
            .to_placement(&session, ArrayPlacement::mesh(mesh.clone(), target_sharding.clone()).unwrap())
            .unwrap();
        let cold_elapsed = cold_start.elapsed();

        let warm_start = std::time::Instant::now();
        let _ = source_array
            .to_placement(&session, ArrayPlacement::mesh(mesh.clone(), target_sharding).unwrap())
            .unwrap();
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

        let result =
            source_array.to_placement(&session, ArrayPlacement::mesh(target_mesh.clone(), target_sharding).unwrap());
        assert!(matches!(
            result,
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));

        // Moving a proper region also requires the extension, even when the source device can slice it locally.
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let result = source_array.to_placement(&session, ArrayPlacement::mesh(target_mesh, target_sharding).unwrap());
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
            source.to_placement(&session, ArrayPlacement::mesh(target_mesh, target_sharding).unwrap()),
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
            let target = ArrayPlacement::mesh(mesh.clone(), sharding.clone()).unwrap();
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
        let target = ArrayPlacement::mesh(mesh.clone(), replicated.clone()).unwrap();
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
        let target = ArrayPlacement::mesh(mesh.clone(), unreduced).unwrap().with_layout(column_major.clone());
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
        let target = ArrayPlacement::mesh(mesh.clone(), unreduced.clone()).unwrap();
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
        let target = ArrayPlacement::mesh(mesh.clone(), replicated.clone()).unwrap();
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
        let target =
            ArrayPlacement::mesh(misattributed.unwrap(), Sharding::replicated(logical_mesh.clone(), 0)).unwrap();
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
        let target = ArrayPlacement::mesh(missing.unwrap(), Sharding::replicated(logical_mesh, 0)).unwrap();
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
            .to_placement(&session, ArrayPlacement::mesh(manual_mesh.clone(), sharded_target.clone()).unwrap())
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
        let session = peer_exchange_session(client);
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

        // The readback waits for the pending producer of the source, so the placement can only return because the
        // readback and the upload of the manual mesh reshard continue in the background.
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        let memory = Memory::Host { pinned: true };
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let target = ArrayPlacement::mesh(mesh.clone(), sharding.clone())
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

        // Each shard holds strided columns of the host value.
        producer.set(None).unwrap();
        placed.block_until_ready().unwrap();
        assert_eq!(shard_values(&placed), vec![vec![0., 1., 4., 5.], vec![2., 3., 6., 7.]]);
        assert_eq!(placed.copy_to_host().unwrap().elements::<f32>().unwrap(), values);
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_to_placement_host_fallback_poisons_result_on_readback_failure() {
        let client = cpu_client(2);
        let session = peer_exchange_session(client);
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
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let placed = source.to_placement(&session, ArrayPlacement::mesh(mesh, sharding).unwrap()).unwrap();

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
        let target = ArrayPlacement::mesh(reversed_mesh, sharding).unwrap();
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
            .into_placement(&session, ArrayPlacement::mesh(mesh.clone(), sharding.clone()).unwrap())
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
            .into_placement(&session, ArrayPlacement::mesh(mesh.clone(), target_sharding).unwrap())
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
        let copied = source.with_copied_shard_buffers(&session, |_| true).unwrap();
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
            .with_copied_shard_buffers(&session, |buffer| Arc::ptr_eq(buffer, &second_buffer))
            .unwrap();
        assert_eq!(shares_buffers(&copied), vec![true, false]);
        assert_eq!(shard_memory_kinds(&copied), ["pinned_host"; 2]);
        assert_eq!(shard_values(&copied), vec![vec![1., 2., 3., 4.]; 2]);
        assert!(!copied.is_committed());

        // A copy stays on the devices of its array, so it keeps whether this process chose them on its own.
        assert!(!copied.is_locally_assigned());
        let locally_assigned = source.with_locally_assigned(true);
        assert!(locally_assigned.is_locally_assigned());
        let copied = locally_assigned.with_copied_shard_buffers(&session, |_| true).unwrap();
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
        let copied = source.with_copied_shard_buffers(&session, |_| true).unwrap();

        // A copy of only the second replica joins its failure into the fence of the result in the same way.
        let second_buffer = Arc::clone(source.shards()[1].buffer().unwrap());
        let partially_copied =
            source.with_copied_shard_buffers(&session, |buffer| Arc::ptr_eq(buffer, &second_buffer)).unwrap();
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
        let error = source.exact_shard_buffers(&session, &plan, &cross_host_layouts, memory, &mut completion).err();
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
            let sent = source
                .cross_host_source_buffer(&session, 0, &client_devices[0], process_index, &mut completion)
                .unwrap();
            completion.join(None).unwrap().block_until_ready().unwrap();
            assert!(!Arc::ptr_eq(&sent, &held));
            assert_eq!(sent.device().unwrap().id().unwrap(), devices[0].id());
            assert_eq!(sent.memory().unwrap().kind().unwrap(), kind);
            assert_eq!(sent.dimensions().unwrap(), [3]);
            assert_eq!(sent.copy_to_host(None).unwrap().r#await().unwrap(), values);

            // The shard that holds a buffer sends it without a copy, and the result of the placement then waits for
            // and reports its readiness, which may be the only completion of the work that produced it.
            let mut completion = PlacementCompletion::default();
            let sent = source
                .cross_host_source_buffer(&session, 1, &client_devices[1], process_index, &mut completion)
                .unwrap();
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
            .cross_host_source_buffer(&session, 1, &client_devices[1], process_index, &mut completion)
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
        let sent = source.cross_host_send_buffer(&session, shard, buffer, &default_layout, &mut completion).unwrap();
        assert!(Arc::ptr_eq(&sent, shard.buffer().unwrap()));
        assert!(completion.join(None).is_none());
        assert_eq!(session.cached_program_count(), 0);

        // A transfer moves the raw bytes of the buffer, so a shard in another layout is converted on its device first.
        let source = upload(&session, &column_major);
        let shard = source.addressable_shards().next().unwrap();
        let mut completion = PlacementCompletion::default();
        let buffer = Arc::clone(shard.buffer().unwrap());
        let sent = source.cross_host_send_buffer(&session, shard, buffer, &default_layout, &mut completion).unwrap();
        completion.join(None).unwrap().block_until_ready().unwrap();
        assert_eq!(session.cached_program_count(), 1);
        assert_eq!(sent.specification().unwrap().layout, default_layout);
        assert_eq!(sent.copy_to_host(None).unwrap().r#await().unwrap(), values_to_bytes(&values));

        // A process that cannot compile operations for its own shards converts the shard through host memory instead
        // of declining, because its receivers wait for the send whatever the layout of its buffer is.
        let shared_session = peer_exchange_session(client);
        let source = upload(&shared_session, &column_major);
        let shard = source.addressable_shards().next().unwrap();
        let mut completion = PlacementCompletion::default();
        let buffer = Arc::clone(shard.buffer().unwrap());
        let sent = source
            .cross_host_send_buffer(&shared_session, shard, buffer, &default_layout, &mut completion)
            .unwrap();
        completion.join(None).unwrap().block_until_ready().unwrap();
        assert_eq!(shared_session.cached_program_count(), 0);
        assert!(!Arc::ptr_eq(&sent, shard.buffer().unwrap()));
        assert_eq!(sent.memory().unwrap().kind().unwrap(), shard.buffer().unwrap().memory().unwrap().kind().unwrap());
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
            let sent =
                source.cross_host_send_buffer(&session, shard, buffer, &default_layout, &mut completion).unwrap();
            completion.join(None).unwrap().block_until_ready().unwrap();
            assert_eq!(sent.memory().unwrap().kind().unwrap(), "device");
            assert_eq!(sent.specification().unwrap().layout, default_layout);
            assert_eq!(sent.dimensions().unwrap(), [2, 3]);
            assert_eq!(sent.copy_to_host(None).unwrap().r#await().unwrap(), values_to_bytes(&values));
        }

        // A bounded-dynamic execution returns a buffer that is padded to the bound of its dynamic dimension, in the
        // default layout. A plugin can take the size of a send from the allocation of the sender while the receiver
        // allocates the shape of the shard, so the shard is sent as an exactly sized buffer, converted on its device or
        // through host memory, and so is a whole shard that a region sends.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(1024)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let identity: CompiledFunction<ArrayType, ArrayType> =
            crate::compile_with_options(|input| input, dynamic_type, &session, Options::new(mesh.clone())).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let input = Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&[11f32, 22.]));
        let padded = identity.executable().execute(&session, input.unwrap()).unwrap();
        let shard = padded.addressable_shards().next().unwrap();
        let padded_buffer = shard.buffer().unwrap();
        assert_ne!(padded_buffer.dimensions().unwrap(), [2]);
        assert_eq!(padded_buffer.unpadded_dimensions().unwrap(), [2]);
        let default_layout =
            cross_host_default_layout(client, BufferType::F32, 0, &StaticShape::new(vec![2])).unwrap().unwrap();
        assert_eq!(padded_buffer.specification().unwrap().layout, default_layout);
        for session in [&session, &shared_session] {
            let mut completion = PlacementCompletion::default();
            let buffer = Arc::clone(padded_buffer);
            let sent = padded.cross_host_send_buffer(session, shard, buffer, &default_layout, &mut completion).unwrap();
            completion.join(None).unwrap().block_until_ready().unwrap();
            assert_eq!(sent.dimensions().unwrap(), [2]);
            assert_eq!(sent.on_device_size_in_bytes().unwrap(), 8);
            assert_eq!(sent.specification().unwrap().layout, default_layout);
            assert_eq!(sent.copy_to_host(None).unwrap().r#await().unwrap(), values_to_bytes(&[11f32, 22.]));
        }
        let mut completion = PlacementCompletion::default();
        let buffer = Arc::clone(padded_buffer);
        let region = Array::prepare_shard_region(
            &session,
            DataType::F32,
            shard,
            buffer,
            shard.slice(),
            Some(&default_layout),
            &mut completion,
        )
        .unwrap();
        completion.join(None).unwrap().block_until_ready().unwrap();
        assert_eq!(region.dimensions().unwrap(), [2]);
        assert_eq!(region.on_device_size_in_bytes().unwrap(), 8);
        assert_eq!(region.copy_to_host(None).unwrap().r#await().unwrap(), values_to_bytes(&[11f32, 22.]));
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
        let environment = PlacementEnvironment::new(&session, &held);
        let process_index = local.process_index();

        // The choice reads only metadata that every process shares, so whether this process holds its buffers does not
        // change it. Keeping each shard on its device moves no shard across processes.
        for source in [&held, &metadata_only] {
            assert!(matches!(
                source.device_placement_strategy(&environment, &source_mesh, &sharding, process_index),
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
                source.device_placement_strategy(&environment, &destination_mesh, &destination_sharding, process_index),
                Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                    if message == "the cross-host transfers extension is not provided by the PJRT plugin",
            ));
        }
        assert!(matches!(
            held.to_placement(
                &session,
                ArrayPlacement::mesh(destination_mesh.clone(), destination_sharding.clone()).unwrap(),
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
                    source.device_placement_strategy(&environment, mesh, sharding, process_index),
                    Ok(DevicePlacementStrategy::ProcessLocalValue),
                ));
            }
        }
        assert!(matches!(
            uploaded.clone().with_locally_assigned(false).device_placement_strategy(
                &environment,
                &destination_mesh,
                &destination_sharding,
                process_index,
            ),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));

        // A destination on devices of this process alone moves nothing between processes, so this process places such
        // an array there with the strategy of an array on devices that every process knows, but only when it can
        // compile operations of its own.
        assert!(matches!(
            uploaded.device_placement_strategy(&environment, &local_mesh, &local_sharding, process_index),
            Ok(DevicePlacementStrategy::ExactShards { plan, cross_host_layouts })
                if plan.cross_host_moves().count() == 0 && cross_host_layouts.is_empty(),
        ));
        let peer_exchange = peer_exchange_session(client);
        assert!(matches!(
            uploaded.device_placement_strategy(
                &PlacementEnvironment::new(&peer_exchange, &uploaded),
                &local_mesh,
                &local_sharding,
                process_index,
            ),
            Ok(DevicePlacementStrategy::ProcessLocalValue),
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
            &simulated_placement_environment(),
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
                let session = &session;
                destinations.iter().map(move |(destination, mesh, sharding)| {
                    let outcome = match placement_agreement(session, views, mesh, sharding) {
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
                shared in pinned host memory -> sharded over both processes: compiled reshard across meshes
                shared in pinned host memory -> unreduced over both processes: compiled reshard across meshes
                shared in pinned host memory -> replicated on process 0: exact shards
                shared in pinned host memory -> sharded on process 1: compiled reshard across meshes
                shared in unpinned host memory -> replicated over both processes: exact shards
                shared in unpinned host memory -> sharded over both processes: compiled reshard across meshes
                shared in unpinned host memory -> unreduced over both processes: compiled reshard across meshes
                shared in unpinned host memory -> replicated on process 0: exact shards
                shared in unpinned host memory -> sharded on process 1: compiled reshard across meshes
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
                partially addressable import in pinned host memory -> sharded over both processes: compiled reshard across meshes
                partially addressable import in pinned host memory -> unreduced over both processes: compiled reshard across meshes
                partially addressable import in pinned host memory -> replicated on process 0: exact shards
                partially addressable import in pinned host memory -> sharded on process 1: compiled reshard across meshes
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

        // A process that imported only the metadata of a tile that the plan has it send (i.e., without a buffer for it
        // or for any local replica) agrees on the plan, and then fails before it issues any transfer, while its peer
        // issues its receive and waits. This is the accepted limit of the PJRT cross-host transfers that
        // `.tasks/plan_placement_decisions.md` records, because a process cannot fail the pending receive of another.
        let (views, (_, mesh, sharding)) = (&sources[2].1, &destinations[0]);
        assert_eq!(placement_agreement(&session, views, mesh, sharding), Ok("exact shards".to_string()));
        let Ok(DevicePlacementStrategy::ExactShards { plan, cross_host_layouts }) =
            views[0].device_placement_strategy(&simulated_placement_environment(), mesh, sharding, 0)
        else {
            panic!("a replica that crosses processes must move as a whole shard");
        };
        let sends = plan.cross_host_sends(0).map(ShardMove::destination_device).collect::<Vec<_>>();
        assert_eq!(sends, [remote[0]]);
        assert_eq!(plan.cross_host_receives(1).map(ShardMove::source_device).collect::<Vec<_>>(), [local[0]]);
        let error = views[0]
            .place_exact_shards(&session, mesh, sharding, Memory::Device, &plan, &cross_host_layouts)
            .err()
            .unwrap();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(
                error.as_ref(),
                ArrayError::MissingShardBuffer { shard_index: 0, device_id } if *device_id == local[0].id(),
            ),
        ));
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
