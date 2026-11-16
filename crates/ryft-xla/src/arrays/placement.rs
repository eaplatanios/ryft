//! Placement of an [`Array`] on another sharding or device mesh, by copying exact shards, by a compiled reshard, or
//! through host memory.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use ryft_core::{
    ArrayType, Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Parameter, Sharding, StaticShape, Typed,
    check_sharding,
};
use ryft_macros::Parameter;
use ryft_pjrt::extensions::cross_host_transfers::{CrossHostTransferKey, GlobalDeviceId};
use ryft_pjrt::{
    Buffer, BufferType, Client, Device as PjrtDevice, DeviceId, Error as PjrtError, Event, ExecutionFence,
};

use crate::arrays::{Array, ArrayError, ArrayShard, ShardDescriptor, ShardIndex, ShardLayout};
use crate::{CompiledFunction, Error, Options, Session, ToPjrt};

// TODO(eaplatanios): Review this module.

/// Target leaf accepted by [`Session::device_put`], which models the subset of the `device` and `src` arguments of
/// JAX's `jax.device_put` that `ryft` currently supports. [`Self::Device`] commits one leaf to a single concrete
/// device, which is represented internally as a mesh of size 1 with fully replicated sharding, and
/// [`Self::Placement`] commits one leaf to an explicit mesh and sharding.
#[derive(Clone, Debug, PartialEq, Eq, Parameter)]
pub enum DevicePutTarget {
    /// Commit the value to one concrete device.
    Device(Device),

    /// Commit the value to the provided mesh and sharding.
    Placement {
        /// Concrete destination mesh describing the device topology.
        mesh: DeviceMesh,

        /// Sharding to apply over `mesh`.
        sharding: Sharding,
    },
}

impl DevicePutTarget {
    /// Creates a single-device placement.
    #[inline]
    pub fn device(device: Device) -> Self {
        Self::Device(device)
    }

    /// Creates a placement on an explicit mesh and sharding.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if `sharding` refers to a different logical mesh than `mesh`.
    #[inline]
    pub fn placement(mesh: DeviceMesh, sharding: Sharding) -> Result<Self, Error> {
        check_sharding!(&mesh, &sharding);
        Ok(Self::Placement { mesh, sharding })
    }

    /// Resolves this target into the destination [`DeviceMesh`] and [`Sharding`] of a value of rank `rank`. A
    /// [`Self::Device`] target resolves to a mesh with one auto axis named `device` of size 1 and to the replicated
    /// sharding over that mesh.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the sharding of a [`Self::Placement`] target refers to a different logical mesh than
    /// its mesh.
    pub(crate) fn resolve(self, rank: usize) -> Result<(DeviceMesh, Sharding), Error> {
        match self {
            Self::Device(device) => {
                let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto)?])?;
                let mesh = DeviceMesh::new(logical_mesh, vec![device])?;
                let sharding = Sharding::replicated(mesh.logical_mesh().clone(), rank);
                Ok((mesh, sharding))
            }
            Self::Placement { mesh, sharding } => {
                check_sharding!(&mesh, &sharding);
                Ok((mesh, sharding))
            }
        }
    }
}

impl From<Device> for DevicePutTarget {
    fn from(value: Device) -> Self {
        Self::Device(value)
    }
}

impl<'o> Array<'o> {
    /// Moves or copies this array to the provided placement without donating the source buffers, so the original
    /// [`Array`] remains fully usable after the call returns. This is the `ryft` analogue of JAX's
    /// `jax.device_put(arr, sharding, donate=False)`.
    ///
    /// `target` is resolved to a concrete [`DeviceMesh`] and [`Sharding`] by `DevicePutTarget::resolve`. When the
    /// resolved placement matches the array's current placement, this function returns a cheap clone of `self`.
    /// Otherwise, it tries three strategies in order.
    ///
    ///   1. **Exact Shards:** When every destination shard has the slices of one source shard, each destination shard
    ///      shares the buffer of its source shard on the same device, or is filled by a copy between devices or by a
    ///      PJRT cross-host transfer for a source shard on a remote host.
    ///   2. **Compiled Reshard:** An `identity(x) = x` program with input and output sharding constraints is lowered
    ///      to StableHLO and Shardy, compiled through the compilation cache of `session`, and executed.
    ///   3. **Host Fallback:** The global array is copied to the host with [`Array::copy_to_host`] and uploaded again
    ///      with [`Array::from_host_buffer_with_sharding`]. This is used only when the device work fails (e.g., for
    ///      `Manual` mesh axes) and both the source array and the destination mesh are fully addressable from this
    ///      process, which is the same restriction that the host round trip of JAX has.
    ///
    /// When the host fallback also cannot satisfy the request, typically because some source shards live on a remote
    /// process, the original error of the device work is propagated as the most informative diagnostic.
    ///
    /// Use [`Self::into_placement`] when the source array is no longer needed after the move. It consumes `self` and
    /// lets PJRT donate the source buffers to the destination, which avoids the extra allocation.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `target`: [`DevicePutTarget`] that resolves to the destination [`DeviceMesh`] and [`Sharding`].
    pub fn to_placement(&self, session: &Session<'o>, target: DevicePutTarget) -> Result<Self, Error> {
        let (target_mesh, target_sharding) = target.resolve(self.sharding().rank())?;
        if self.mesh() == &target_mesh && self.sharding() == &target_sharding {
            return Ok(self.clone());
        }
        self.clone().run_placement_dispatch(session, target_mesh, target_sharding, false)
    }

    /// Moves this array to the provided placement, consuming `self` and donating its source buffers to the
    /// destination. This is the `ryft` analogue of JAX's `jax.device_put(arr, sharding, donate=True)`.
    ///
    /// Donation lets PJRT reuse the source buffers for the output, which saves the memory of the source array for
    /// large training tensors. Because `self` is consumed at the language boundary, the donated buffers are not
    /// observable to the caller after this function returns. Rust ownership therefore enforces the same invariant that
    /// the `donate=True` flag carries in JAX, without exposing a runtime hazard.
    ///
    /// The strategies are identical to those of [`Self::to_placement`] (i.e., exact shards, a compiled reshard, and a
    /// host fallback), and only the donation flag differs.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `target`: [`DevicePutTarget`] that resolves to the destination [`DeviceMesh`] and [`Sharding`].
    pub fn into_placement(self, session: &Session<'o>, target: DevicePutTarget) -> Result<Self, Error> {
        let (target_mesh, target_sharding) = target.resolve(self.sharding().rank())?;
        if self.mesh() == &target_mesh && self.sharding() == &target_sharding {
            return Ok(self);
        }
        self.run_placement_dispatch(session, target_mesh, target_sharding, true)
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
    pub(crate) fn reshard_on_devices(
        self,
        session: &Session<'o>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        donate: bool,
    ) -> Result<Self, Error> {
        self.reshard_on_devices_or_return(session, mesh, sharding, donate).map_err(|(error, _)| error)
    }

    /// Returns an array with the type and the placement of this array whose addressable shards are copies of the
    /// shards of this array on the same devices, so that the result shares no buffer with any other array.
    pub(crate) fn with_copied_shard_buffers(&self, session: &Session<'o>) -> Result<Self, Error> {
        let buffers = self
            .addressable_shards()
            .filter_map(ArrayShard::buffer)
            .map(|buffer| Ok(Arc::new(buffer.copy_to_device(buffer.device()?)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        let r#type = self.r#type().into_owned().with_sharding(None)?;
        let output = if self.data_type().is_zero() {
            Self::from_zero_space(session, r#type, Some(self.sharding().clone()), self.mesh().clone())?
        } else {
            let (sharding, mesh) = (self.sharding().clone(), self.mesh().clone());
            Self::from_shared_addressable_buffers(session, r#type, sharding, mesh, buffers)?
        };
        Ok(match self.execution_fence() {
            Some(fence) => output.with_execution_fence(fence.clone()),
            None => output,
        })
    }

    /// Placement dispatch shared by [`Self::to_placement`] and [`Self::into_placement`], which adds a host fallback
    /// to the device work of [`Self::reshard_on_devices_or_return`]. The host fallback needs no `donate` argument,
    /// because a host round trip cannot reuse device buffers.
    fn run_placement_dispatch(
        self,
        session: &Session<'o>,
        target_mesh: DeviceMesh,
        target_sharding: Sharding,
        donate: bool,
    ) -> Result<Self, Error> {
        check_sharding!(&target_mesh, &target_sharding);

        // Keep the error of the device work, which is the most informative one if the host fallback fails too.
        let (compiled_error, source) =
            match self.reshard_on_devices_or_return(session, &target_mesh, &target_sharding, donate) {
                Ok(array) => return Ok(array),
                Err((error, Some(source))) => (error, source),
                Err((error, None)) => return Err(error),
            };

        // Host fallback, which copies the value of the source array to the host and uploads it again with
        // `from_host_buffer_with_sharding`. It only applies when both the source array and the destination mesh are
        // fully addressable from this process, which is the same restriction that the host round trip of JAX has. If
        // any condition fails, the error of the device work is surfaced as the more informative diagnostic.
        let local_process = session.client().process_index()?;
        let destination_fully_addressable =
            target_mesh.devices().iter().all(|device| device.process_index() == local_process);
        if !destination_fully_addressable {
            return Err(compiled_error);
        }
        let host_bytes = match source.copy_to_host() {
            Ok(bytes) => bytes,
            Err(_) => return Err(compiled_error),
        };
        let host_type = ArrayType::new(source.data_type(), source.shape().into());
        match Self::from_host_buffer_with_sharding(
            session,
            host_type,
            target_sharding.clone(),
            target_mesh,
            host_bytes.as_slice(),
        ) {
            Ok(array) => Ok(array),
            Err(_) => Err(compiled_error),
        }
    }

    /// Places this array on `mesh` according to `sharding` without compiling a program, or returns `None` when that
    /// placement needs one. An array of [`DataType::Zero`](ryft_core::DataType::Zero) has no buffers, so only its
    /// placement metadata changes. For any other array, [`Self::exact_shard_buffers`] provides the destination
    /// shards. The result carries the execution fence of this array, so that errors of the execution that produced
    /// this array still surface through the result. It also retains the completion fence of the cross-host sends that
    /// this process issued, so that the readiness functions of the result surface the errors of those sends.
    fn place_exact_shards(
        &self,
        session: &Session<'o>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<Option<Self>, Error> {
        let output = if self.data_type().is_zero() {
            let r#type = self.r#type().into_owned().with_sharding(None)?;
            Self::from_zero_space(session, r#type, Some(sharding.clone()), mesh.clone())?
        } else {
            let shape = self.shape();
            let Some((buffers, send_fence)) = self.exact_shard_buffers(session.client(), &shape, mesh, sharding)?
            else {
                return Ok(None);
            };
            let r#type = self.r#type().into_owned().with_shape(shape).with_sharding(None)?;
            let output =
                Self::from_shared_addressable_buffers(session, r#type, sharding.clone(), mesh.clone(), buffers)?;
            match send_fence {
                Some(fence) => output.with_send_fence(fence),
                None => output,
            }
        };
        Ok(Some(match self.execution_fence() {
            Some(fence) => output.with_execution_fence(fence.clone()),
            None => output,
        }))
    }

    /// Returns the buffers of the destination shards that `client` can address when every destination shard has the
    /// slices of one source shard and both shardings hold one value per slice (refer to
    /// [`Self::plan_exact_shard_put`]). A destination shard on the device of its source shard shares the buffer of
    /// that shard. Other local source shards are copied between devices, and remote source shards arrive through
    /// the PJRT cross-host transfers extension. The result is `None` when the destination needs other shards or
    /// when a remote move needs the cross-host transfers extension and the plugin does not provide it.
    ///
    /// The buffers come with the completion fence of the cross-host sends that this process issues for the
    /// destination shards of other processes, or `None` when it issues none. This function never waits for those
    /// sends, because a send completes only once its receiver issues the matching receive, and the receiver may be
    /// this same process further below or a process that is itself waiting for a send of its own. The caller retains
    /// the fence in the placed array so that [`Self::ready`] and [`Self::block_until_ready`] surface send failures.
    /// Every process derives the same plan from global metadata, so each send has exactly one matching receive, and
    /// [`issue_cross_host_transfers`] issues the transfers of all processes in one global order.
    fn exact_shard_buffers(
        &self,
        client: &'o Client<'_>,
        global_shape: &StaticShape,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<Option<(Vec<Arc<Buffer<'o>>>, Option<ExecutionFence>)>, Error> {
        let client_process_index = client.process_index()?;
        let plan = match self.plan_exact_shard_put(client_process_index, global_shape, mesh, sharding)? {
            Some(plan) => plan,
            None => return Ok(None),
        };
        let needs_cross_host_transfers = !plan.cross_host_sends().is_empty() || !plan.cross_host_receives().is_empty();
        if needs_cross_host_transfers {
            match client.cross_host_transfers_extension() {
                Ok(_) => {}
                Err(PjrtError::Unimplemented { .. }) => return Ok(None),
                Err(error) => return Err(error.into()),
            }
        }

        let addressable_devices = client.addressable_devices()?;
        let mut addressable_device_by_id = HashMap::with_capacity(addressable_devices.len());
        for device in addressable_devices {
            addressable_device_by_id.insert(device.id()?, device);
        }

        let mut addressable_buffers = Vec::new();
        for local_copy_plan in plan.local_copies() {
            let source_buffer = self
                .addressable_device_shard(local_copy_plan.source_device_id())
                .ok_or_else(|| {
                    Error::from(ArrayError::MissingShardBuffer {
                        shard_index: local_copy_plan.source_shard_index(),
                        device_id: local_copy_plan.source_device_id(),
                    })
                })?
                .buffer()
                .unwrap();
            // Arrays never mutate their buffers and donation skips buffers that are shared, so a destination shard on
            // the device of its source shard shares the buffer of that shard instead of copying it.
            if local_copy_plan.source_device_id() == local_copy_plan.destination_device_id() {
                addressable_buffers.push(Arc::clone(source_buffer));
                continue;
            }
            let destination_device =
                addressable_device_by_id.get(&local_copy_plan.destination_device_id()).ok_or_else(|| {
                    Error::from(ArrayError::NonAddressableDevice {
                        device_id: local_copy_plan.destination_device_id(),
                        process_index: client_process_index,
                    })
                })?;
            addressable_buffers.push(Arc::new(source_buffer.copy_to_device(destination_device.clone())?));
        }

        let mut transfers = Vec::with_capacity(plan.cross_host_sends().len() + plan.cross_host_receives().len());
        for send_plan in plan.cross_host_sends() {
            let buffer = self
                .addressable_device_shard(send_plan.source_device_id())
                .ok_or_else(|| {
                    Error::from(ArrayError::MissingShardBuffer {
                        shard_index: send_plan.source_shard_index(),
                        device_id: send_plan.source_device_id(),
                    })
                })?
                .buffer()
                .map(|buffer| buffer.as_ref())
                .unwrap();
            transfers.push(CrossHostTransfer::Send {
                source: send_plan.source_device_id(),
                destination: send_plan.destination_device_id(),
                transfer_key: send_plan.transfer_key(),
                buffer,
            });
        }
        for receive_plan in plan.cross_host_receives() {
            let device = addressable_device_by_id.get(&receive_plan.destination_device_id()).ok_or_else(|| {
                Error::from(ArrayError::NonAddressableDevice {
                    device_id: receive_plan.destination_device_id(),
                    process_index: client_process_index,
                })
            })?;
            transfers.push(CrossHostTransfer::Receive {
                source: receive_plan.source_device_id(),
                destination: receive_plan.destination_device_id(),
                transfer_key: receive_plan.transfer_key(),
                device,
                element_type: self.data_type().to_pjrt(),
                dimensions: receive_plan.cross_host_shape()?,
            });
        }
        let (send_events, received_buffers) = issue_cross_host_transfers(client, transfers)?;
        addressable_buffers.extend(received_buffers.into_iter().map(|(_, buffer)| Arc::new(buffer)));
        let send_fence = (!send_events.is_empty()).then(|| ExecutionFence::new(send_events));

        Ok(Some((addressable_buffers, send_fence)))
    }

    /// Plans the moves of whole shards that place this array on `mesh` according to `sharding`, as seen from the
    /// process with index `client_process_index`.
    ///
    /// Returns `None` when a destination shard does not have the slices of one source shard, which needs slicing or
    /// concatenating shards, and when either sharding has unreduced or varying manual mesh axes. The devices along
    /// such axes hold different values for the same slices (e.g., the summands of an unreduced value), so a
    /// destination shard is not a copy of the source shard with its slices.
    fn plan_exact_shard_put(
        &self,
        client_process_index: usize,
        global_shape: &StaticShape,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<Option<ExactShardPutPlan>, Error> {
        let holds_one_value_per_slice =
            |sharding: &Sharding| sharding.unreduced_axes().is_empty() && sharding.varying_manual_axes().is_empty();
        if !holds_one_value_per_slice(self.sharding()) || !holds_one_value_per_slice(sharding) {
            return Ok(None);
        }

        let mut source_shards_by_slices = HashMap::<Vec<Range<usize>>, Vec<&ArrayShard<'o>>>::new();
        for shard in self.shards() {
            source_shards_by_slices.entry(shard.slice().to_vec()).or_default().push(shard);
        }

        let destination_layout = ShardLayout::new(global_shape, mesh, sharding)?;
        let destination_shards = destination_layout.descriptors();
        let destination_shard_count = destination_shards.len();
        let mut local_copies = Vec::new();
        let mut cross_host_sends = Vec::new();
        let mut cross_host_receives = Vec::new();
        for destination_shard in destination_shards {
            let source_shards = match source_shards_by_slices.get(destination_shard.slice()) {
                Some(source_shards) => source_shards,
                None => return Ok(None),
            };
            let source_shard = preferred_exact_source_shard(source_shards.as_slice(), destination_shard);
            let source_device = source_shard.device();
            let source_shard_index = source_shard.index();
            let source_process_index = source_device.process_index();
            let destination_device = destination_shard.device();
            let destination_process_index = destination_device.process_index();

            if destination_process_index == client_process_index {
                if source_process_index == client_process_index {
                    if !source_shard.is_addressable() {
                        return Ok(None);
                    }
                    local_copies.push(LocalShardCopyPlan::new(
                        source_shard_index,
                        source_device.id(),
                        destination_shard.index(),
                        destination_device.id(),
                    ));
                } else {
                    cross_host_receives.push(CrossHostShardReceivePlan::new(
                        source_shard_index,
                        source_device.id(),
                        destination_shard.index(),
                        destination_device.id(),
                        destination_shard.shape(),
                        exact_shard_transfer_key(
                            source_shard_index,
                            destination_shard.index(),
                            destination_shard_count,
                        )?,
                    ));
                }
            } else if source_process_index == client_process_index {
                if !source_shard.is_addressable() {
                    return Ok(None);
                }
                cross_host_sends.push(CrossHostShardSendPlan::new(
                    source_shard_index,
                    source_device.id(),
                    destination_shard.index(),
                    destination_device.id(),
                    exact_shard_transfer_key(source_shard_index, destination_shard.index(), destination_shard_count)?,
                ));
            }
        }

        Ok(Some(ExactShardPutPlan::new(local_copies, cross_host_sends, cross_host_receives)))
    }

    /// Places this array on `mesh` according to `sharding` with device work only, and on failure also returns this
    /// array when a host fallback may still place it. It tries the following strategies in order.
    ///
    ///   1. **Exact Shards:** [`Self::place_exact_shards`] shares or copies whole shards and compiles nothing. Its
    ///      errors are final, so they come without this array.
    ///   2. **Compiled Reshard:** The `ryft` analogue of how JAX's `jax.device_put(arr, new_sharding)` lowers a
    ///      reshard of a committed `jax.Array`, which is an `identity(x) = x` program with input and output sharding
    ///      constraints that is lowered to StableHLO and Shardy, compiled, and executed on the devices. When this
    ///      array already is on `mesh`, one such program reshards it. Otherwise,
    ///      [`Self::reshard_replicated_across_meshes`] or [`Self::reshard_sharded_across_meshes`] first moves a
    ///      replicated copy to `mesh`.
    ///
    /// Unsupported requests return typed [`ArrayError`] variants through [`Error::Array`] (refer to
    /// [`Self::validate_compiled_reshard`]), and cross-host extension, tracing, compilation, and execution failures
    /// propagate through [`Error`] without losing their typed payloads.
    ///
    /// A donated array must be the only handle of its buffers, so the compiled reshard on the mesh of this array
    /// consumes this array when `donate` is `true`. This array is then not returned if that execution fails.
    // The error carries the unconsumed array back to the caller, which falls back to a host transfer with it.
    #[allow(clippy::result_large_err)]
    fn reshard_on_devices_or_return(
        self,
        session: &Session<'o>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        donate: bool,
    ) -> Result<Self, (Error, Option<Self>)> {
        match self.place_exact_shards(session, mesh, sharding) {
            Ok(Some(output)) => return Ok(output),
            Ok(None) => {}
            Err(error) => return Err((error, None)),
        }
        if let Err(error) = self.validate_compiled_reshard(session, mesh) {
            return Err((error, Some(self)));
        }
        if self.mesh() == mesh {
            match self.compile_reshard_on_same_mesh(session, mesh, sharding, donate) {
                Ok(reshard) => {
                    let source = (!donate).then(|| self.clone());
                    reshard.executable().execute(session, self).map_err(|error| (error, source))
                }
                Err(error) => Err((error, Some(self))),
            }
        } else {
            let output = if self.sharding().is_replicated() {
                self.reshard_replicated_across_meshes(session, mesh, sharding)
            } else {
                self.reshard_sharded_across_meshes(session, mesh, sharding)
            };
            output.map_err(|error| (error, Some(self)))
        }
    }

    /// Checks that a compiled program can place this array on `destination_mesh`. The errors are
    /// [`ArrayError::ManualReshardingAxis`] when the destination mesh has a `Manual` axis, while `Auto` and `Explicit`
    /// axes are supported, and [`ArrayError::MissingShardBuffer`] when a source shard on the current process has no
    /// buffer.
    fn validate_compiled_reshard(&self, session: &Session<'o>, destination_mesh: &DeviceMesh) -> Result<(), Error> {
        // `Manual` axes are managed explicitly by the user (e.g., inside `shard_map`) and cannot be planned by the SPMD
        // partitioner from the top level. `Auto` and `Explicit` axes both go through SPMD partitioning, so both are
        // allowed.
        if let Some(axis) =
            destination_mesh.logical_mesh().axes().iter().find(|axis| axis.r#type() == MeshAxisType::Manual)
        {
            return Err(ArrayError::ManualReshardingAxis { axis_name: axis.name().to_string() }.into());
        }

        // Source shards on the current process must carry a local buffer, because a missing one indicates a real
        // misconfiguration. Shards on other processes are normal, and the cross-host transfers extension serves them
        // when the compiled reshard needs them.
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

    /// Reshards this array, which must already be on `mesh`, by executing the program of
    /// [`Self::compile_reshard_on_same_mesh`]. This function consumes the array so that a donation takes effect,
    /// which requires the executed array to be the only handle of its buffers.
    fn reshard_on_same_mesh(
        self,
        session: &Session<'o>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        donate: bool,
    ) -> Result<Self, Error> {
        let reshard = self.compile_reshard_on_same_mesh(session, mesh, sharding, donate)?;
        reshard.executable().execute(session, self)
    }

    /// Compiles the identity program that reshards this array, which must already be on `destination_mesh`, to
    /// `destination_sharding`. If `donate` is `true`, the input of the program is marked as donated so that PJRT can
    /// reuse the memory of its buffers for the output buffers.
    ///
    /// The compilation cache is keyed by the complete lowered computation, including its input and output sharding
    /// annotations and PJRT options.
    fn compile_reshard_on_same_mesh(
        &self,
        session: &Session<'o>,
        destination_mesh: &DeviceMesh,
        destination_sharding: &Sharding,
        donate: bool,
    ) -> Result<CompiledFunction<'o, ArrayType, ArrayType>, Error> {
        let element_type = self.data_type();
        let shape = self.shape();
        let source_sharding = self.sharding().clone();
        let destination_sharding = destination_sharding.clone();

        // Build the bare input type, which has no sharding, for tracing. The input and output shardings are attached as
        // `sdy.sharding` attributes through `Options::with_input_shardings` and `Options::with_output_shardings`.
        let bare_input_type = ArrayType::new(element_type, shape.clone().into());

        let options = Options::new(destination_mesh.clone())
            .with_input_shardings(vec![source_sharding.clone()])
            .with_output_shardings(vec![destination_sharding.clone()])
            .with_donate(donate);

        let staged: crate::StagedFunction<'_, ArrayType, ArrayType> =
            crate::stage(|input| input, bare_input_type, options)?;
        session.compile_staged_function(staged)
    }

    /// Copies this fully replicated array onto every device of `destination_mesh`, using copies between the devices of
    /// this process and PJRT cross-host transfers between processes, and then reshards the copy on
    /// `destination_mesh` to `destination_sharding` with [`Self::reshard_on_same_mesh`]. The result retains the
    /// completion fence of the cross-host sends that this process issued, so that [`Self::ready`] and
    /// [`Self::block_until_ready`] surface their errors without this function waiting for them.
    fn reshard_replicated_across_meshes(
        &self,
        session: &Session<'o>,
        destination_mesh: &DeviceMesh,
        destination_sharding: &Sharding,
    ) -> Result<Array<'o>, Error> {
        let client = session.client();
        let client_process_index = client.process_index()?;
        let addressable_devices = client.addressable_devices()?;
        let mut destination_device_by_id = HashMap::with_capacity(addressable_devices.len());
        for device in addressable_devices {
            let id = device.id()?;
            destination_device_by_id.insert(id, device);
        }

        // A fully replicated source array holds its data on every device of its mesh. The first device of that mesh is
        // the canonical sender that drives all cross-host sends, and receiving processes use the same device ID as
        // the source of their cross-host receives.
        let canonical_source_device =
            self.mesh().devices().first().copied().ok_or_else(|| ryft_core::ProgramError::InvalidArgument {
                message: "cannot reshard an array on an empty self device mesh".to_string(),
            })?;
        let this_process_is_canonical_sender = canonical_source_device.process_index() == client_process_index;
        let local_source_buffer = self.addressable_shards().find_map(|shard| shard.buffer().cloned());

        // Every process must derive the same transfers from global metadata, because a send that no receive matches
        // is still delivered over the communicator of its pair of devices (e.g., by NCCL on CUDA) and then corrupts
        // the next transfer between those devices. A process that owns a device of the source mesh holds a replica of
        // this array, which `Self::validate_compiled_reshard` guarantees, so it copies that replica to its destination
        // devices, and only the other processes receive from the canonical sender.
        let source_process_indices =
            self.mesh().devices().iter().map(|device| device.process_index()).collect::<HashSet<_>>();

        // Decide the transfer plan of each destination device. Local destinations are served by `copy_to_device` when
        // this process owns a device of the source mesh, or by a cross-host receive otherwise. Remote destinations of
        // processes that own no device of the source mesh are served by cross-host sends from the process of the
        // canonical sender, and every other remote destination is ignored.
        let mut local_buffers: HashMap<DeviceId, Buffer<'o>> = HashMap::new();
        let mut transfers = Vec::new();
        let element_type = self.data_type().to_pjrt();
        let dimensions = self.shape().as_slice().iter().map(|&size| size as i64).collect::<Vec<_>>();
        let destination_count = destination_mesh.devices().len();
        for (destination_index, destination_device) in destination_mesh.devices().iter().enumerate() {
            let destination_device_id = destination_device.id();
            let destination_local = destination_device.process_index() == client_process_index;
            let transfer_key = exact_shard_transfer_key(0, destination_index, destination_count)?;

            if destination_local {
                let pjrt_device =
                    destination_device_by_id.get(&destination_device_id).ok_or(ArrayError::NonAddressableDevice {
                        device_id: destination_device_id,
                        process_index: client_process_index,
                    })?;
                if source_process_indices.contains(&client_process_index) {
                    let local_source = local_source_buffer.as_ref().ok_or_else(|| {
                        Error::from(ArrayError::MissingShardBuffer {
                            shard_index: 0,
                            device_id: canonical_source_device.id(),
                        })
                    })?;
                    let buffer = local_source.copy_to_device(pjrt_device.clone())?;
                    local_buffers.insert(destination_device_id, buffer);
                } else {
                    transfers.push(CrossHostTransfer::Receive {
                        source: canonical_source_device.id(),
                        destination: destination_device_id,
                        transfer_key,
                        device: pjrt_device,
                        element_type,
                        dimensions: dimensions.clone(),
                    });
                }
            } else if this_process_is_canonical_sender
                && !source_process_indices.contains(&destination_device.process_index())
            {
                let local_source = local_source_buffer.as_ref().ok_or_else(|| {
                    Error::from(ArrayError::MissingShardBuffer {
                        shard_index: 0,
                        device_id: canonical_source_device.id(),
                    })
                })?;
                transfers.push(CrossHostTransfer::Send {
                    source: canonical_source_device.id(),
                    destination: destination_device_id,
                    transfer_key,
                    buffer: local_source.as_ref(),
                });
            }
            // Otherwise, another process serves this remote destination, and there is nothing to do here.
        }

        // A receiver may have only local destinations, so an unavailable extension cannot be attributed to a remote
        // destination device. Preserve the plugin's actual extension error before issuing any transfers.
        if !transfers.is_empty() {
            client.cross_host_transfers_extension()?;
        }

        // Waiting for the sends here could deadlock, because a send completes only once its receiver issues the
        // matching receive, so the returned array retains their completion fence instead.
        let (send_events, received_buffers) = issue_cross_host_transfers(client, transfers)?;
        local_buffers.extend(received_buffers);
        let send_fence = (!send_events.is_empty()).then(|| ExecutionFence::new(send_events));

        // Assemble the buffers of the devices in the device order of `destination_mesh`. Only locally addressable
        // destinations contribute to the intermediate `Array`, and the owning processes handle the remote
        // destinations through their own parallel calls to this function.
        let mut buffers = Vec::with_capacity(local_buffers.len());
        for destination_device in destination_mesh.devices() {
            if let Some(buffer) = local_buffers.remove(&destination_device.id()) {
                buffers.push(buffer);
            }
        }

        let shape = self.shape();
        let element_type = self.data_type();
        let replicated_on_destination = Sharding::replicated(destination_mesh.logical_mesh().clone(), shape.rank());
        let intermediate_type = ArrayType::new(element_type, shape.into()).with_sharding(replicated_on_destination)?;
        let intermediate =
            Array::from_addressable_buffers(session, intermediate_type, destination_mesh.clone(), buffers)?;

        // The intermediate is owned exclusively by this function and is never observed by callers.
        // Donating its buffers lets PJRT reuse their memory for the output of the final SPMD reshard.
        let output = intermediate.reshard_on_same_mesh(session, destination_mesh, destination_sharding, true)?;
        Ok(match send_fence {
            Some(fence) => output.with_send_fence(fence),
            None => output,
        })
    }

    /// Reshards this sharded array onto a different destination mesh by composing two compiled SPMD programs.
    ///
    ///   1. An all-gather on the mesh of this array produces a fully replicated intermediate, so that every device of
    ///      that mesh ends up with the full logical array.
    ///   2. [`Self::reshard_replicated_across_meshes`] then copies that intermediate to `destination_mesh` and
    ///      compiles the final reshard from the replicated sharding on `destination_mesh` to `destination_sharding`.
    ///
    /// Both compiled programs go through the compilation cache of `session`, so repeated calls with the same input and
    /// output shardings pay the compilation cost once.
    fn reshard_sharded_across_meshes(
        &self,
        session: &Session<'o>,
        destination_mesh: &DeviceMesh,
        destination_sharding: &Sharding,
    ) -> Result<Array<'o>, Error> {
        let source_mesh = self.mesh();
        let replicated_on_source = Sharding::replicated(source_mesh.logical_mesh().clone(), self.sharding().rank());
        // The source array is owned externally, so it is not donated to the all-gather. The gathered intermediate is
        // owned by this function and is donated to the subsequent reshard inside
        // `Self::reshard_replicated_across_meshes`.
        let gathered = self.clone().reshard_on_same_mesh(session, source_mesh, &replicated_on_source, false)?;
        gathered.reshard_replicated_across_meshes(session, destination_mesh, destination_sharding)
    }
}

/// Deterministic transfer plan of exact shards for one placement of an [`Array`] (e.g., by [`Array::to_placement`]),
/// as seen from the current process.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ExactShardPutPlan {
    /// Local destination shards, each of which shares the buffer of its source shard on the same device or is copied
    /// from another device of this process.
    local_copies: Vec<LocalShardCopyPlan>,

    /// Cross-host sends that this process must initiate for remote destination shards.
    cross_host_sends: Vec<CrossHostShardSendPlan>,

    /// Cross-host receives that this process must initiate for local destination shards.
    cross_host_receives: Vec<CrossHostShardReceivePlan>,
}

impl ExactShardPutPlan {
    /// Creates a deterministic transfer plan of exact shards.
    fn new(
        local_copies: Vec<LocalShardCopyPlan>,
        cross_host_sends: Vec<CrossHostShardSendPlan>,
        cross_host_receives: Vec<CrossHostShardReceivePlan>,
    ) -> Self {
        Self { local_copies, cross_host_sends, cross_host_receives }
    }

    /// Returns the local destination shards, which share or copy the buffers of source shards of this process.
    fn local_copies(&self) -> &[LocalShardCopyPlan] {
        &self.local_copies
    }

    /// Returns the cross-host sends initiated by this process.
    fn cross_host_sends(&self) -> &[CrossHostShardSendPlan] {
        &self.cross_host_sends
    }

    /// Returns the cross-host receives initiated by this process.
    fn cross_host_receives(&self) -> &[CrossHostShardReceivePlan] {
        &self.cross_host_receives
    }
}

/// One local placement of an exact shard in an [`ExactShardPutPlan`], which shares the source buffer when both shards
/// are on the same device and otherwise copies it between devices.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalShardCopyPlan {
    /// Source shard index in the source array.
    source_shard_index: ShardIndex,

    /// Source device ID.
    source_device_id: DeviceId,

    /// Destination shard index in the destination array.
    destination_shard_index: ShardIndex,

    /// Destination device ID.
    destination_device_id: DeviceId,
}

impl LocalShardCopyPlan {
    /// Creates a plan for one local placement of an exact shard.
    fn new(
        source_shard_index: ShardIndex,
        source_device_id: DeviceId,
        destination_shard_index: ShardIndex,
        destination_device_id: DeviceId,
    ) -> Self {
        Self { source_shard_index, source_device_id, destination_shard_index, destination_device_id }
    }

    /// Returns the source shard index in the source array.
    fn source_shard_index(&self) -> ShardIndex {
        self.source_shard_index
    }

    /// Returns the source device ID.
    fn source_device_id(&self) -> DeviceId {
        self.source_device_id
    }

    /// Returns the destination device ID.
    fn destination_device_id(&self) -> DeviceId {
        self.destination_device_id
    }
}

/// One cross-host send of an exact shard in an [`ExactShardPutPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
struct CrossHostShardSendPlan {
    /// Source shard index in the source array.
    source_shard_index: ShardIndex,

    /// Source device ID.
    source_device_id: DeviceId,

    /// Destination shard index in the destination array.
    destination_shard_index: ShardIndex,

    /// Destination device ID.
    destination_device_id: DeviceId,

    /// Deterministic transfer rendezvous key shared with the matching receive.
    transfer_key: CrossHostTransferKey,
}

impl CrossHostShardSendPlan {
    /// Creates a plan for one cross-host send of an exact shard.
    fn new(
        source_shard_index: ShardIndex,
        source_device_id: DeviceId,
        destination_shard_index: ShardIndex,
        destination_device_id: DeviceId,
        transfer_key: CrossHostTransferKey,
    ) -> Self {
        Self { source_shard_index, source_device_id, destination_shard_index, destination_device_id, transfer_key }
    }

    /// Returns the source shard index in the source array.
    fn source_shard_index(&self) -> ShardIndex {
        self.source_shard_index
    }

    /// Returns the source device ID.
    fn source_device_id(&self) -> DeviceId {
        self.source_device_id
    }

    /// Returns the destination device ID.
    fn destination_device_id(&self) -> DeviceId {
        self.destination_device_id
    }

    /// Returns the deterministic transfer rendezvous key shared with the matching receive.
    fn transfer_key(&self) -> CrossHostTransferKey {
        self.transfer_key
    }
}

/// One cross-host receive of an exact shard in an [`ExactShardPutPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
struct CrossHostShardReceivePlan {
    /// Source shard index in the source array.
    source_shard_index: ShardIndex,

    /// Source device ID.
    source_device_id: DeviceId,

    /// Destination shard index in the destination array.
    destination_shard_index: ShardIndex,

    /// Destination device ID.
    destination_device_id: DeviceId,

    /// Logical destination shard shape.
    destination_shape: StaticShape,

    /// Deterministic transfer rendezvous key shared with the matching send.
    transfer_key: CrossHostTransferKey,
}

impl CrossHostShardReceivePlan {
    /// Creates a plan for one cross-host receive of an exact shard.
    fn new(
        source_shard_index: ShardIndex,
        source_device_id: DeviceId,
        destination_shard_index: ShardIndex,
        destination_device_id: DeviceId,
        destination_shape: StaticShape,
        transfer_key: CrossHostTransferKey,
    ) -> Self {
        Self {
            source_shard_index,
            source_device_id,
            destination_shard_index,
            destination_device_id,
            destination_shape,
            transfer_key,
        }
    }

    /// Returns the source device ID.
    fn source_device_id(&self) -> DeviceId {
        self.source_device_id
    }

    /// Returns the destination shard index in the destination array.
    fn destination_shard_index(&self) -> ShardIndex {
        self.destination_shard_index
    }

    /// Returns the destination device ID.
    fn destination_device_id(&self) -> DeviceId {
        self.destination_device_id
    }

    /// Returns the logical destination shard shape.
    fn destination_shape(&self) -> &StaticShape {
        &self.destination_shape
    }

    /// Returns the deterministic transfer rendezvous key shared with the matching send.
    fn transfer_key(&self) -> CrossHostTransferKey {
        self.transfer_key
    }

    /// Returns the PJRT cross-host shape of the destination shard.
    fn cross_host_shape(&self) -> Result<Vec<i64>, Error> {
        self.destination_shape()
            .as_slice()
            .iter()
            .enumerate()
            .map(|(dimension, &size)| {
                i64::try_from(size).map_err(|_| {
                    Error::from(ArrayError::InvalidShardDimension {
                        shard_index: self.destination_shard_index(),
                        dimension,
                        source: ryft_core::DimensionError::ExtentExceedsBackendWidth {
                            value: size,
                            maximum: i64::MAX as usize,
                        },
                    })
                })
            })
            .collect()
    }
}

/// One PJRT cross-host transfer that the current process takes part in, which [`issue_cross_host_transfers`] issues.
enum CrossHostTransfer<'b, 'o> {
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
        buffer: &'b Buffer<'o>,
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
        device: &'b PjrtDevice<'o>,

        /// Element type of the received buffer.
        element_type: BufferType,

        /// Dimensions of the received buffer.
        dimensions: Vec<i64>,
    },
}

impl CrossHostTransfer<'_, '_> {
    /// Returns the IDs of the source and destination devices of this transfer.
    fn devices(&self) -> (DeviceId, DeviceId) {
        match self {
            Self::Send { source, destination, .. } | Self::Receive { source, destination, .. } => {
                (*source, *destination)
            }
        }
    }
}

/// Returns the deterministic cross-host transfer key for moving the source shard `source_shard_index` to the
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
/// and returns the completion events of the sends and the received buffers with the IDs of their devices.
///
/// The transfers are issued one at a time in the order of their source and then destination device IDs, keeping the
/// given order of the transfers between the same two devices. The CUDA plugin carries the transfers between two
/// devices over a communicator of that pair and pairs them by the order in which both sides issue them, ignoring their
/// transfer keys, and the first transfer between two devices blocks the calling thread until the peer process issues
/// its matching transfer. Two processes that each send before they receive would therefore deadlock when they swap
/// shards. Every process derives the same transfers from global metadata and issues its own in this one global order,
/// so the first pair of devices with a pending transfer always has both of its processes waiting for it.
fn issue_cross_host_transfers<'o>(
    client: &'o Client<'_>,
    mut transfers: Vec<CrossHostTransfer<'_, 'o>>,
) -> Result<(Vec<Event<()>>, Vec<(DeviceId, Buffer<'o>)>), Error> {
    transfers.sort_by_key(CrossHostTransfer::devices);
    let mut send_events = Vec::new();
    let mut received_buffers = Vec::new();
    for transfer in transfers {
        match transfer {
            CrossHostTransfer::Send { destination, transfer_key, buffer, .. } => {
                let destination = cross_host_global_device_id(destination)?;
                send_events.extend(client.cross_host_send_buffers(&[buffer], &[destination], &[transfer_key], &[])?);
            }
            CrossHostTransfer::Receive { source, destination, transfer_key, device, element_type, dimensions } => {
                let source = cross_host_global_device_id(source)?;
                let buffers = client.cross_host_receive_buffers(
                    &[element_type],
                    &[dimensions.as_slice()],
                    device,
                    &[source],
                    &[transfer_key],
                    &[],
                )?;
                received_buffers.extend(buffers.into_iter().map(|buffer| (destination, buffer)));
            }
        }
    }
    Ok((send_events, received_buffers))
}

/// Returns the preferred source shard among the `source_shards` whose slices exactly match `destination_shard`.
///
/// The preference is deterministic and depends only on global shard metadata, so that every process chooses the same
/// source shard for a given destination shard. It prefers a source shard on the process of the destination shard,
/// then a source shard on the device of the destination shard, and finally breaks ties by device ID and shard index.
fn preferred_exact_source_shard<'o, 'c>(
    source_shards: &[&'o ArrayShard<'c>],
    destination_shard: &ShardDescriptor,
) -> &'o ArrayShard<'c> {
    source_shards
        .iter()
        .min_by_key(|source_shard| {
            let source_device = source_shard.device();
            let destination_device = destination_shard.device();
            (
                source_device.process_index() != destination_device.process_index(),
                source_device.id() != destination_device.id(),
                source_device.id(),
                source_shard.index(),
            )
        })
        .copied()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        DataType, Dimension as CoreDimension, DimensionError, Layout, Memory, Shape, ShardingDimension, TiledLayout,
    };
    use ryft_pjrt::{BufferType, ExecutionFence};

    use crate::domains::tests::domain_mesh;
    use crate::tests::{
        cpu_client, device_mesh_2x2, four_device_mesh_x, logical_mesh_2x2, values_from_bytes, values_to_bytes,
    };
    use crate::{FromPjrt, Session};

    use super::*;

    // Returns the `f32` values that the devices of `array` hold, in the device order of its mesh.
    fn shard_values(array: &Array<'_>) -> Vec<Vec<f32>> {
        array
            .shards()
            .iter()
            .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host().unwrap()))
            .collect()
    }

    // Returns a mesh with one auto axis `x` over the first two addressable devices of `client`.
    fn two_device_sub_mesh_x(client: &ryft_pjrt::Client<'_>) -> DeviceMesh {
        let client_devices = client.addressable_devices().unwrap();
        let devices =
            client_devices.iter().take(2).map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
            .unwrap()
    }

    #[test]
    fn test_device_put_target_device() {
        let device = Device::new(3, 0);
        assert_eq!(DevicePutTarget::device(device), DevicePutTarget::Device(device));
        assert_eq!(DevicePutTarget::from(device), DevicePutTarget::Device(device));
    }

    #[test]
    fn test_device_put_target_placement() {
        let mesh = device_mesh_2x2();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        assert_eq!(
            DevicePutTarget::placement(mesh.clone(), sharding.clone()).unwrap(),
            DevicePutTarget::Placement { mesh: mesh.clone(), sharding },
        );

        // A sharding over another logical mesh is rejected.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let error = DevicePutTarget::placement(mesh.clone(), Sharding::replicated(other_mesh.clone(), 1)).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("mesh mismatch; expected `{:?}` but got `{other_mesh:?}`", mesh.logical_mesh()),
        );
    }

    #[test]
    fn test_device_put_target_resolve() {
        // A device target resolves to a replicated placement on a mesh with one auto axis of size 1.
        let device = Device::new(3, 0);
        let device_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        assert_eq!(
            DevicePutTarget::Device(device).resolve(2).unwrap(),
            (DeviceMesh::new(device_mesh.clone(), vec![device]).unwrap(), Sharding::replicated(device_mesh, 2)),
        );

        // A placement target resolves to its own mesh and sharding.
        let mesh = device_mesh_2x2();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharding.clone() }.resolve(1).unwrap(),
            (mesh, sharding),
        );
    }

    #[test]
    fn test_array_to_placement_reshards_fully_addressable_array() {
        let client = cpu_client(2);
        let session = Session::with_client(&client);
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
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(source_sharding)
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            source_mesh,
            values_to_bytes::<f32>(source_values.as_slice()).as_slice(),
        )
        .unwrap();

        let target_mesh = domain_mesh(&client, "x", 2);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let moved_array = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding })
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
    fn test_array_to_placement_copies_matching_local_shards_without_full_source_addressability() {
        let client = cpu_client(1);
        let session = Session::with_client(&client);
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
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let source_array = Array::from_addressable_buffers_with_sharding(
            &session,
            source_array_type,
            sharding.clone(),
            mesh.clone(),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: mesh.clone(), sharding })
            .unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied_array.device_shard(local_device_id).unwrap().copy_to_host().unwrap();
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_array_to_placement_rejects_non_addressable_source_shards() {
        let client = cpu_client(1);
        let session = Session::with_client(&client);
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 1)],
        )
        .unwrap();
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let source_array = Array::from_addressable_buffers_with_sharding(
            &session,
            source_array_type,
            source_sharding,
            source_mesh,
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
            .to_placement(&session, DevicePutTarget::Placement { mesh: target_mesh, sharding: target_sharding })
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
        let session = Session::with_client(&client);
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
        let summands = Array::from_addressable_buffers(&session, r#type, mesh.clone(), buffers).unwrap();
        let reduced = summands
            .to_placement(&session, DevicePutTarget::placement(mesh.clone(), tiled.clone()).unwrap())
            .unwrap();
        assert_eq!(reduced.sharding(), &tiled);
        assert_eq!(shard_values(&reduced), vec![vec![11., 22.], vec![33., 44.], vec![11., 22.], vec![33., 44.]]);

        // In the other direction, a copy of every shard would hold the logical value once per device along `x`.
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(tiled).unwrap();
        let value =
            Array::from_host_buffer(&session, r#type, mesh.clone(), values_to_bytes(&[1f32, 2., 3., 4.])).unwrap();
        let summands =
            value.to_placement(&session, DevicePutTarget::placement(mesh, unreduced.clone()).unwrap()).unwrap();
        assert_eq!(summands.sharding(), &unreduced);
        assert_eq!(values_from_bytes::<f32>(&summands.copy_to_host().unwrap()), vec![1., 2., 3., 4.]);
    }

    #[test]
    fn test_array_to_placement_zero_space_is_bufferless_and_preserves_type_metadata() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);
        let mesh = four_device_mesh_x(&client);
        let source_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let layout = Layout::Tiled(TiledLayout::new(vec![0], Vec::new()));
        let source_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(4)]))
            .with_layout(layout.clone())
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(source_sharding)
            .unwrap();
        let source = Array::from_host_buffer(&session, source_type, mesh.clone(), []).unwrap();
        let target_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let resharded = source
            .to_placement(
                &session,
                DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding.clone() },
            )
            .unwrap();

        assert_eq!(resharded.sharding(), &target_sharding);
        assert_eq!(resharded.layout(), Some(&layout));
        assert_eq!(resharded.r#type().memory(), Memory::Host { pinned: true });
        assert_eq!(resharded.addressable_shards().count(), mesh.devices().len());
        assert!(resharded.addressable_shards().all(|shard| shard.buffer().is_none()));
    }

    #[test]
    fn test_array_to_placement_zero_space_preserves_producer_completion() {
        let client = crate::tests::execution_client();
        let session = Session::with_client(&client);
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
        let target = DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding.clone() };

        for fails in [false, true] {
            let (event, promise) = client.event(()).unwrap();
            let source =
                Array::from_host_buffer(&session, ArrayType::new_static(DataType::Zero, [2]), source_mesh.clone(), [])
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
        let session = Session::with_client(&client);
        let client_devices = client.addressable_devices().unwrap();

        // The source is replicated on a 1-device submesh.
        let source_device_id = client_devices[0].id().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(source_device_id, client_devices[0].process_index().unwrap())],
        )
        .unwrap();
        let values = [60.0f32, 61.0, 62.0, 63.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source_array =
            Array::from_host_buffer(&session, source_type, source_mesh, values_to_bytes::<f32>(&values).as_slice())
                .unwrap();

        // The target is replicated on the full 4-device mesh. The exact shard path matches every destination shard to
        // the single source shard, which covers the full array. The destination shard on the source device shares the
        // source buffer, `copy_to_device` produces independent buffers on the other devices, and `copy_to_host`
        // works on every one of them.
        let target_mesh = four_device_mesh_x(&client);
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
        let resharded = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding })
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        assert_eq!(shard_values(&resharded), vec![values.to_vec(); 4]);
        // The exact shard path satisfies this request without compiling any SPMD program.
        assert_eq!(session.program_count(), 0);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_replicated_to_sharded_on_same_mesh() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);
        let mesh = four_device_mesh_x(&client);

        let values = [10.0f32, 11.0, 12.0, 13.0];
        let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array =
            Array::from_host_buffer(&session, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();

        let sharded_target =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_target })
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        assert_eq!(shard_values(&resharded), vec![vec![10.0], vec![11.0], vec![12.0], vec![13.0]]);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_sharded_to_replicated_on_same_mesh() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);
        let mesh = four_device_mesh_x(&client);

        let values = [20.0f32, 21.0, 22.0, 23.0];
        let sharded_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]));
        let source_array = Array::from_host_buffer_with_sharding(
            &session,
            source_type,
            sharded_sharding,
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        let replicated_target = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let resharded = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: replicated_target })
            .unwrap();

        assert_eq!(shard_values(&resharded), vec![values.to_vec(); 4]);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_sharded_to_differently_sharded_on_same_mesh() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);
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
        let source_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4), CoreDimension::Static(4)]));
        let source_array = Array::from_host_buffer_with_sharding(
            &session,
            source_type,
            sharded_along_x,
            mesh.clone(),
            values_to_bytes::<f32>(&row_values).as_slice(),
        )
        .unwrap();

        let sharded_along_y = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let resharded = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_along_y })
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
        let session = Session::with_client(&client);
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
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            explicit_mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        let sharded_target =
            Sharding::new(explicit_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: explicit_mesh, sharding: sharded_target })
            .unwrap();
        assert_eq!(resharded.addressable_shards().count(), 4);
        assert_eq!(shard_values(&resharded), vec![vec![400.0], vec![401.0], vec![402.0], vec![403.0]]);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_cross_mesh_replicated_source_to_sharded_destination() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);
        let client_devices = client.addressable_devices().unwrap();

        // The source lives on a 1-device submesh.
        let source_device_id = client_devices[0].id().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(source_device_id, client_devices[0].process_index().unwrap())],
        )
        .unwrap();
        let values = [10.0f32, 11.0, 12.0, 13.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source_array =
            Array::from_host_buffer(&session, source_type, source_mesh, values_to_bytes::<f32>(&values).as_slice())
                .unwrap();

        // Reshard onto the full 4-device mesh, sharded along "x".
        let target_mesh = four_device_mesh_x(&client);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding })
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        assert_eq!(shard_values(&resharded), vec![vec![10.0], vec![11.0], vec![12.0], vec![13.0]]);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_cross_mesh_sharded_source_to_replicated_destination() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);

        // The source is sharded along "x" on a 2-device submesh, so devices 0 and 1 each hold half the data.
        let source_mesh = two_device_sub_mesh_x(&client);
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let values = [70.0f32, 71.0, 72.0, 73.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]));
        let source_array = Array::from_host_buffer_with_sharding(
            &session,
            source_type,
            source_sharding,
            source_mesh,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // The destination is replicated on the full 4-device mesh. The sharded source is first all-gathered on the
        // source mesh and then copied onto the destination mesh, and no further reshard is needed because the
        // intermediate sharding matches the requested replicated destination sharding.
        let target_mesh = four_device_mesh_x(&client);
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
        let resharded = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding })
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        assert_eq!(shard_values(&resharded), vec![values.to_vec(); 4]);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_cross_mesh_sharded_source_to_sharded_destination() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);

        // The source is sharded along "x" on a 2-device submesh.
        let source_mesh = two_device_sub_mesh_x(&client);
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let values = [80.0f32, 81.0, 82.0, 83.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]));
        let source_array = Array::from_host_buffer_with_sharding(
            &session,
            source_type,
            source_sharding,
            source_mesh,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // The destination is sharded along "x" on the full 4-device mesh, so each destination shard holds exactly one
        // element of the global array.
        let target_mesh = four_device_mesh_x(&client);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding })
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        assert_eq!(shard_values(&resharded), vec![vec![80.0], vec![81.0], vec![82.0], vec![83.0]]);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_cross_mesh_sharded_source_compiles_two_executables() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);

        let source_mesh = two_device_sub_mesh_x(&client);
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let values = [90.0f32, 91.0, 92.0, 93.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]));
        let source_array = Array::from_host_buffer_with_sharding(
            &session,
            source_type,
            source_sharding,
            source_mesh,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        let target_mesh = four_device_mesh_x(&client);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(session.program_count(), 0);
        let _ = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: target_mesh, sharding: target_sharding })
            .unwrap();

        // A sharded source on another mesh compiles two executables, which are an all-gather on the source mesh and a
        // reshard from the replicated sharding to the requested sharding on the destination mesh.
        assert_eq!(session.program_count(), 2);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_caches_executable_across_calls() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);
        let mesh = four_device_mesh_x(&client);

        let values = [30.0f32, 31.0, 32.0, 33.0];
        let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array =
            Array::from_host_buffer(&session, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();

        let sharded_target =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(session.program_count(), 0);
        let first = source_array
            .clone()
            .to_placement(&session, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_target.clone() })
            .unwrap();
        let after_first = session.program_count();
        let second = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_target })
            .unwrap();
        let after_second = session.program_count();

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
        let session = Session::with_client(&client);
        let mesh = four_device_mesh_x(&client);
        let values = (0..4096).map(|index| index as f32).collect::<Vec<_>>();
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source_array =
            Array::from_host_buffer(&session, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let target_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let cold_start = std::time::Instant::now();
        let _ = source_array
            .clone()
            .to_placement(
                &session,
                DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding.clone() },
            )
            .unwrap();
        let cold_elapsed = cold_start.elapsed();

        let warm_start = std::time::Instant::now();
        let _ = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding })
            .unwrap();
        let warm_elapsed = warm_start.elapsed();

        eprintln!("cold reshard (trace+lower+compile+execute): {cold_elapsed:?}");
        eprintln!("warm reshard (structural cache hit + execute, no trace/lower): {warm_elapsed:?}");
        assert_eq!(session.program_count(), 1);
    }

    #[test]
    fn test_array_to_placement_preserves_missing_cross_host_extension() {
        let client = cpu_client(1);
        let session = Session::with_client(&client);
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
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source_array =
            Array::from_host_buffer(&session, source_type, source_mesh, values_to_bytes::<f32>(&values).as_slice())
                .unwrap();

        // Sending to the remote destination requires the extension even though local copies are possible.
        assert!(matches!(client.cross_host_transfers_extension(), Err(PjrtError::Unimplemented { .. })));
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
        )
        .unwrap();
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);

        let result = source_array
            .to_placement(&session, DevicePutTarget::Placement { mesh: target_mesh, sharding: target_sharding });
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
        let session = Session::with_client(&client);
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device.id() + 1, local_device.process_index() + 1)],
        )
        .unwrap();
        let source_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source = Array::from_addressable_buffers(&session, source_type, source_mesh, Vec::new()).unwrap();
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
            source.to_placement(&session, DevicePutTarget::Placement { mesh: target_mesh, sharding: target_sharding }),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));
    }

    #[test]
    fn test_array_to_placement_with_manual_mesh_axes_uses_host_fallback() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);
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
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            manual_mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // The SPMD partitioner cannot plan manual mesh axes, so the compiled reshard declines. `Array::to_placement`
        // falls through to the host fallback, because the full source is addressable and the destination is on the
        // same single process, and successfully places the data according to the requested sharding.
        let sharded_target =
            Sharding::new(manual_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(
                &session,
                DevicePutTarget::Placement { mesh: manual_mesh.clone(), sharding: sharded_target.clone() },
            )
            .unwrap();
        assert_eq!(resharded.sharding(), &sharded_target);
        assert_eq!(resharded.addressable_shards().count(), 4);

        // Each device-local shard now owns one element of the input vector.
        assert_eq!(shard_values(&resharded), vec![vec![40.0], vec![41.0], vec![42.0], vec![43.0]]);

        // No executable was compiled, because the host fallback bypasses the compilation cache.
        assert_eq!(session.program_count(), 0, "host fallback should not populate the executable cache");
    }

    #[test]
    fn test_array_into_placement_preserves_same_partially_addressable_placement() {
        let client = cpu_client(1);
        let session = Session::with_client(&client);
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
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let source_array = Array::from_addressable_buffers_with_sharding(
            &session,
            source_array_type,
            sharding.clone(),
            mesh.clone(),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array = source_array
            .into_placement(&session, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharding.clone() })
            .unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied_array.device_shard(local_device_id).unwrap().copy_to_host().unwrap();
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_array_into_placement_donates_source_and_returns_independently_readable_output() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);
        let mesh = four_device_mesh_x(&client);

        let values = [300.0f32, 301.0, 302.0, 303.0];
        let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array =
            Array::from_host_buffer(&session, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();

        // `Array::into_placement` consumes the source array and donates its buffers to the compiled SPMD reshard. PJRT
        // may reuse the memory of those buffers for the output. The donated source is not readable after this call
        // because it has been consumed, but the output buffers of the returned array must be independently
        // addressable on each device.
        let target_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let moved = source_array
            .into_placement(&session, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding })
            .unwrap();

        assert_eq!(shard_values(&moved), vec![vec![300.0], vec![301.0], vec![302.0], vec![303.0]]);
    }

    #[test]
    fn test_array_reshard_on_devices() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);
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
        let shared_buffers = |left: &Array<'_>, right: &Array<'_>| {
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
        let source = Array::from_host_buffer(&session, r#type, mesh.clone(), values_to_bytes(&[1f32, 2., 3., 4.]))
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
        assert_eq!(session.program_count(), 0);

        // On another device mesh, the shard on a device of the source shares its buffer and the others are copies.
        let single_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[1]],
        )
        .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let source =
            Array::from_host_buffer(&session, r#type, single_device_mesh, values_to_bytes(&[1f32, 2., 3., 4.]))
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
        assert_eq!(session.program_count(), 0);

        // A sharding with other shards needs a compiled program.
        let sharded = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let resharded = output.clone().reshard_on_devices(&session, &mesh, &sharded, false).unwrap();
        assert_eq!(resharded.sharding(), &sharded);
        assert_eq!(shard_values(&resharded), vec![vec![1.], vec![2.], vec![3.], vec![4.]]);
        assert_eq!(session.program_count(), 1);

        // There is no host fallback, so a destination that a compiled program cannot serve is an error.
        let manual_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let manual =
            Sharding::new(manual_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            output.reshard_on_devices(&session, &manual_mesh, &manual, false).unwrap_err().to_string(),
            "compiled reshard does not support `Manual` mesh axis x",
        );
    }

    #[test]
    fn test_array_plan_exact_shard_put_uses_cross_host_send_and_receive_for_remote_moves() {
        let client = cpu_client(1);
        let session = Session::with_client(&client);
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
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let source_array = Array::from_addressable_buffers_with_sharding(
            &session,
            source_array_type,
            source_sharding,
            source_mesh,
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

        let plan = source_array
            .plan_exact_shard_put(
                client.process_index().unwrap(),
                &source_array.shape(),
                &target_mesh,
                &target_sharding,
            )
            .unwrap();

        assert_eq!(
            plan,
            Some(ExactShardPutPlan::new(
                Vec::new(),
                vec![CrossHostShardSendPlan::new(0, local_device_id, 0, remote_device_id, 0)],
                vec![CrossHostShardReceivePlan::new(
                    1,
                    remote_device_id,
                    1,
                    local_device_id,
                    StaticShape::new(vec![2]),
                    3,
                )],
            ))
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_cross_host_shard_receive_plan_cross_host_shape_retains_dimension_error_and_shard_context() {
        let plan = CrossHostShardReceivePlan::new(0, 0, 2, 1, StaticShape::new(vec![usize::MAX]), 0);
        let error = plan.cross_host_shape().unwrap_err();
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
