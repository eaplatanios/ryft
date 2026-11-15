//! Placement of an [`Array`] on another sharding or device mesh, by copying exact shards, by a compiled reshard, or
//! through host memory.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use ryft_core::{
    ArrayType, Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Parameter, Sharding, StaticShape, Typed,
    check_sharding,
};
use ryft_macros::Parameter;
use ryft_pjrt::extensions::cross_host_transfers::{CrossHostTransferKey, GlobalDeviceId};
use ryft_pjrt::{Buffer, Client, DeviceId, Error as PjrtError};

use crate::arrays::{Array, ArrayError, ArrayShard, ShardDescriptor, ShardIndex, ShardLayout};
use crate::{CompiledFunction, Domain, Error, Options, ToPjrt};

// TODO(eaplatanios): Review this module.

/// Target leaf accepted by the higher-level [`device_put()`](crate::arrays::device_put()) API.
///
/// This models the current `ryft` subset of JAX's `device` / `src` arguments:
/// - [`Self::Device`] commits one leaf to a single concrete device, represented internally as a
///   size-1 mesh with fully replicated sharding, and
/// - [`Self::Placement`] commits one leaf to an explicit mesh/sharding pair.
#[derive(Clone, Debug, PartialEq, Eq, Parameter)]
pub enum DevicePutTarget {
    /// Commit the value to one concrete device.
    Device(Device),

    /// Commit the value to the provided mesh/sharding pair.
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

    /// Creates an explicit mesh/sharding placement.
    ///
    /// Returns an error if `sharding` refers to a different logical mesh than `mesh`.
    #[inline]
    pub fn placement(mesh: DeviceMesh, sharding: Sharding) -> Result<Self, Error> {
        check_sharding!(&mesh, &sharding);
        Ok(Self::Placement { mesh, sharding })
    }

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
    /// Moves or copies this array to the provided placement, **without** donating the source
    /// buffers. The original [`Array`] remains fully usable after the call returns. `ryft`'s
    /// analogue of JAX's `jax.device_put(arr, sharding, donate=False)`.
    ///
    /// `target` is resolved to a concrete [`DeviceMesh`] and [`Sharding`] via
    /// `DevicePutTarget::resolve`. When the resolved placement matches the array's current
    /// placement, the method returns a cheap clone of `self`. Otherwise it tries three
    /// strategies in order:
    ///
    /// 1. **Exact-shard fast path** — when every destination shard is exactly one source shard,
    ///    satisfy each with a direct device-to-device copy or a PJRT cross-host transfer for
    ///    remote-host destinations.
    /// 2. **Compiled-XLA path** — trace `identity(x) = x` with input and output
    ///    `with_sharding_constraint` ops, lower to StableHLO + Shardy, compile via the cache on
    ///    `cache`, and execute.
    /// 3. **Host fallback** — materialize the global array on host via per-shard `copy_to_host`,
    ///    merge shard bytes into a row-major buffer, and re-upload via
    ///    [`Array::from_host_buffer`]. Used only when the compiled path declines (Manual mesh
    ///    axes, etc.) and both endpoints are fully addressable from this process. JAX's
    ///    host-roundtrip path has the same restriction.
    ///
    /// When the host fallback also can't satisfy the request — typically because some source
    /// shards live on a remote process — the original compiled-path error is propagated as the
    /// most informative diagnostic.
    ///
    /// Use [`Self::into_placement`] when the source array is no longer needed after the move:
    /// it consumes `self` and lets PJRT donate the source buffers to the destination, avoiding
    /// the extra allocation.
    ///
    /// # Parameters
    ///
    ///   - `domain`: [`Domain`] wrapping the PJRT client and the compile-program cache.
    ///   - `target`: Resolved into a destination [`DeviceMesh`] + [`Sharding`].
    pub fn to_placement(&self, domain: &Domain<'o>, target: DevicePutTarget) -> Result<Self, Error> {
        let (target_mesh, target_sharding) = target.resolve(self.sharding().rank())?;
        if self.mesh() == &target_mesh && self.sharding() == &target_sharding {
            return Ok(self.clone());
        }
        self.clone().run_placement_dispatch(domain, target_mesh, target_sharding, false)
    }

    /// Moves this array to the provided placement, donating its source buffers to the
    /// destination. Consumes `self`. `ryft`'s analogue of JAX's
    /// `jax.device_put(arr, sharding, donate=True)`.
    ///
    /// Donation lets PJRT reuse the source buffers for the output, saving the source-array
    /// footprint for large training tensors. Because `self` is consumed at the language
    /// boundary, the donated buffers are not observable to the caller after this method
    /// returns — Rust ownership enforces the same invariant the `donate=True` flag carries in
    /// JAX without exposing a runtime footgun.
    ///
    /// The dispatch tiers are identical to [`Self::to_placement`] — fast path, compiled SPMD
    /// path, host fallback — only the donation flag differs.
    ///
    /// # Parameters
    ///
    ///   - `domain`: [`Domain`] wrapping the PJRT client and the compile-program cache.
    ///   - `target`: Resolved into a destination [`DeviceMesh`] + [`Sharding`].
    pub fn into_placement(self, domain: &Domain<'o>, target: DevicePutTarget) -> Result<Self, Error> {
        let (target_mesh, target_sharding) = target.resolve(self.sharding().rank())?;
        if self.mesh() == &target_mesh && self.sharding() == &target_sharding {
            return Ok(self);
        }
        self.run_placement_dispatch(domain, target_mesh, target_sharding, true)
    }

    /// Places this array on `mesh` according to `sharding` with device work only (i.e., without copying its contents
    /// through host memory). [`Self::to_placement`] and [`Self::into_placement`] start with this work, and the
    /// execution of a compiled program uses it for inputs whose sharding differs from the one that the program was
    /// compiled for. Refer to [`Self::reshard_on_devices_or_return`] for the strategies that it tries.
    ///
    /// # Parameters
    ///
    ///   - `domain`: [`Domain`] that provides the PJRT client and the compilation cache.
    ///   - `mesh`: Destination [`DeviceMesh`].
    ///   - `sharding`: Destination [`Sharding`].
    ///   - `donate`: Whether a compiled reshard on the mesh of this array may reuse the buffers of this array for
    ///     its output. The donation only takes effect when this array is the only handle of its buffers.
    pub(crate) fn reshard_on_devices(
        self,
        domain: &Domain<'o>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        donate: bool,
    ) -> Result<Self, Error> {
        self.reshard_on_devices_or_return(domain, mesh, sharding, donate).map_err(|(error, _)| error)
    }

    /// Placement dispatch shared by [`Self::to_placement`] and [`Self::into_placement`], which adds a host fallback
    /// to the device work of [`Self::reshard_on_devices_or_return`]. The host fallback needs no `donate` argument,
    /// because a host round trip cannot reuse device buffers.
    fn run_placement_dispatch(
        self,
        domain: &Domain<'o>,
        target_mesh: DeviceMesh,
        target_sharding: Sharding,
        donate: bool,
    ) -> Result<Self, Error> {
        check_sharding!(&target_mesh, &target_sharding);

        // Keep the error of the device work, which is the most informative one if the host fallback fails too.
        let (compiled_error, source) =
            match self.reshard_on_devices_or_return(domain, &target_mesh, &target_sharding, donate) {
                Ok(array) => return Ok(array),
                Err((error, Some(source))) => (error, source),
                Err((error, None)) => return Err(error),
            };

        // Tier 3: host fallback. Materializes every shard on host and reuploads via
        // `from_host_buffer`. Only engages when both endpoints are fully addressable from this
        // process — JAX's host-roundtrip path has the same restriction. If any condition fails,
        // surface the compiled-path error as the more informative diagnostic.
        let local_process = domain.client().process_index().map_err(crate::Error::from)?;
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
            domain,
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
    /// this array still surface through the result.
    fn place_exact_shards(
        &self,
        domain: &Domain<'o>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<Option<Self>, Error> {
        let output = if self.data_type().is_zero() {
            let r#type = self.r#type().into_owned().with_sharding(None)?;
            Self::from_zero_space(domain, r#type, Some(sharding.clone()), mesh.clone())?
        } else {
            let shape = self.shape();
            let Some(buffers) = self.exact_shard_buffers(domain.client(), &shape, mesh, sharding)? else {
                return Ok(None);
            };
            let r#type = self.r#type().into_owned().with_shape(shape).with_sharding(None)?;
            Self::from_shared_addressable_buffers(domain, r#type, sharding.clone(), mesh.clone(), buffers)?
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
    fn exact_shard_buffers(
        &self,
        client: &'o Client<'_>,
        global_shape: &StaticShape,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<Option<Vec<Arc<Buffer<'o>>>>, Error> {
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

        if !plan.cross_host_sends().is_empty() {
            let mut send_buffers = Vec::with_capacity(plan.cross_host_sends().len());
            let mut destination_devices = Vec::with_capacity(plan.cross_host_sends().len());
            let mut transfer_keys = Vec::with_capacity(plan.cross_host_sends().len());
            for send_plan in plan.cross_host_sends() {
                let source_buffer = self
                    .addressable_device_shard(send_plan.source_device_id())
                    .ok_or_else(|| {
                        Error::from(ArrayError::MissingShardBuffer {
                            shard_index: send_plan.source_shard_index(),
                            device_id: send_plan.source_device_id(),
                        })
                    })?
                    .buffer()
                    .map(|buffer| buffer.as_ref())
                    .expect("addressable shard lookups should always return a local buffer");
                send_buffers.push(source_buffer);
                destination_devices.push(cross_host_global_device_id(send_plan.destination_device_id())?);
                transfer_keys.push(send_plan.transfer_key());
            }
            let _send_events = client.cross_host_send_buffers(
                send_buffers.as_slice(),
                destination_devices.as_slice(),
                transfer_keys.as_slice(),
                &[],
            )?;
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
                .expect("addressable shard lookups should always return a local buffer");
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

        let mut receive_plans_by_device = HashMap::<DeviceId, Vec<&CrossHostShardReceivePlan>>::new();
        for receive_plan in plan.cross_host_receives() {
            receive_plans_by_device.entry(receive_plan.destination_device_id()).or_default().push(receive_plan);
        }
        let mut receive_device_ids = receive_plans_by_device.keys().copied().collect::<Vec<_>>();
        receive_device_ids.sort_unstable();
        for receive_device_id in receive_device_ids {
            let receive_plans = receive_plans_by_device
                .get(&receive_device_id)
                .expect("grouped receive plans should exist for every grouped destination device");
            let destination_device = addressable_device_by_id.get(&receive_device_id).ok_or_else(|| {
                Error::from(ArrayError::NonAddressableDevice {
                    device_id: receive_device_id,
                    process_index: client_process_index,
                })
            })?;
            let element_types = receive_plans.iter().map(|_| self.data_type().to_pjrt()).collect::<Vec<_>>();
            let dimensions = receive_plans
                .iter()
                .map(|receive_plan| cross_host_shape(receive_plan))
                .collect::<Result<Vec<_>, _>>()?;
            let dimension_slices = dimensions.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let source_devices = receive_plans
                .iter()
                .map(|receive_plan| cross_host_global_device_id(receive_plan.source_device_id()))
                .collect::<Result<Vec<_>, _>>()?;
            let transfer_keys =
                receive_plans.iter().map(|receive_plan| receive_plan.transfer_key()).collect::<Vec<_>>();
            let received_buffers = client.cross_host_receive_buffers(
                element_types.as_slice(),
                dimension_slices.as_slice(),
                destination_device,
                source_devices.as_slice(),
                transfer_keys.as_slice(),
                &[],
            )?;
            addressable_buffers.extend(received_buffers.into_iter().map(Arc::new));
        }

        Ok(Some(addressable_buffers))
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
    fn reshard_on_devices_or_return(
        self,
        domain: &Domain<'o>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        donate: bool,
    ) -> Result<Self, (Error, Option<Self>)> {
        match self.place_exact_shards(domain, mesh, sharding) {
            Ok(Some(output)) => return Ok(output),
            Ok(None) => {}
            Err(error) => return Err((error, None)),
        }
        if let Err(error) = self.validate_compiled_reshard(domain, mesh) {
            return Err((error, Some(self)));
        }
        if self.mesh() == mesh {
            match self.compile_reshard_on_same_mesh(domain, mesh, sharding, donate) {
                Ok(reshard) => {
                    let source = (!donate).then(|| self.clone());
                    reshard.executable().execute(&domain, self).map_err(|error| (error, source))
                }
                Err(error) => Err((error, Some(self))),
            }
        } else {
            let output = if self.sharding().is_replicated() {
                self.reshard_replicated_across_meshes(domain, mesh, sharding)
            } else {
                self.reshard_sharded_across_meshes(domain, mesh, sharding)
            };
            output.map_err(|error| (error, Some(self)))
        }
    }

    /// Checks that a compiled program can place this array on `destination_mesh`. The errors are
    /// [`ArrayError::ManualReshardingAxis`] when the destination mesh has a `Manual` axis (`Auto` and `Explicit` axes
    /// are supported), and [`ArrayError::MissingShardBuffer`] when a source shard on the current process has no
    /// buffer.
    fn validate_compiled_reshard(&self, domain: &Domain<'o>, destination_mesh: &DeviceMesh) -> Result<(), Error> {
        // `Manual` axes are managed explicitly by the user (e.g. inside `shard_map`) and cannot be
        // planned by the SPMD partitioner from the top level. `Auto` and `Explicit` axes both go
        // through SPMD partitioning, so allow both.
        if let Some(axis) =
            destination_mesh.logical_mesh().axes().iter().find(|axis| axis.r#type() == MeshAxisType::Manual)
        {
            return Err(ArrayError::ManualReshardingAxis { axis_name: axis.name().to_string() }.into());
        }

        // Source shards on the current process must carry a local buffer (a missing one indicates a
        // real misconfiguration). Shards on other processes are normal — they're served by the
        // cross-host transfers extension when the compiled path needs them.
        let client_process_index = domain.client().process_index()?;
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
        domain: &Domain<'o>,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        donate: bool,
    ) -> Result<Self, Error> {
        let reshard = self.compile_reshard_on_same_mesh(domain, mesh, sharding, donate)?;
        reshard.executable().execute(&domain, self)
    }

    /// Compiles the identity program that reshards this array, which must already be on `destination_mesh`, to
    /// `destination_sharding`. If `donate` is `true`, the input of the program is marked as donated so that PJRT can
    /// reuse the memory of its buffers for the output buffers.
    ///
    /// The compilation cache is keyed by the complete lowered computation, including its input/output sharding
    /// annotations and PJRT options.
    fn compile_reshard_on_same_mesh(
        &self,
        domain: &Domain<'o>,
        destination_mesh: &DeviceMesh,
        destination_sharding: &Sharding,
        donate: bool,
    ) -> Result<CompiledFunction<'o, ArrayType, ArrayType>, Error> {
        let element_type = self.data_type();
        let shape = self.shape();
        let source_sharding = self.sharding().clone();
        let destination_sharding = destination_sharding.clone();

        // Build the bare input type for tracing (no sharding); the in/out shardings are attached as
        // `sdy.sharding` attributes via `Options::{input_shardings, output_shardings}`.
        let bare_input_type = ArrayType::new(element_type, shape.clone().into());

        let options = Options::new(destination_mesh.clone())
            .with_input_shardings(vec![source_sharding.clone()])
            .with_output_shardings(vec![destination_sharding.clone()])
            .with_donate(donate);

        let staged: crate::StagedFunction<'_, ArrayType, ArrayType> =
            crate::stage(|input| input, bare_input_type, options)?;
        domain.compile_staged_function(staged)
    }

    /// Broadcasts this fully replicated array onto every device in `destination_mesh` via intra-host D2D
    /// copies, then defers to [`Self::reshard_on_same_mesh`] to perform the on-`destination_mesh` reshard.
    fn reshard_replicated_across_meshes(
        &self,
        domain: &Domain<'o>,
        destination_mesh: &DeviceMesh,
        destination_sharding: &Sharding,
    ) -> Result<Array<'o>, Error> {
        let client = domain.client();
        let client_process_index = client.process_index()?;
        let addressable_devices = client.addressable_devices()?;
        let mut destination_device_by_id = HashMap::with_capacity(addressable_devices.len());
        for device in addressable_devices {
            let id = device.id()?;
            destination_device_by_id.insert(id, device);
        }

        // For a fully-replicated self, the data is present on every source_mesh device. Pick a
        // canonical sender — the first source_mesh device — to drive all cross-host sends from this
        // process. Receiving processes use the same device id as their cross-host self.
        let canonical_source_device =
            self.mesh().devices().first().copied().ok_or_else(|| ryft_core::ProgramError::InvalidArgument {
                message: "cannot reshard an array on an empty self device mesh".to_string(),
            })?;
        let canonical_source_global_id = cross_host_global_device_id(canonical_source_device.id())?;
        let this_process_is_canonical_sender = canonical_source_device.process_index() == client_process_index;
        let local_source_buffer = self.addressable_shards().find_map(|shard| shard.buffer().cloned());

        // Decide each destination device's transfer plan. Local destinations are served by
        // `copy_to_device` when this process owns a self buffer, or by `cross_host_receive_buffers`
        // when the self is on another process. Remote destinations are served by
        // `cross_host_send_buffers` from the canonical-sender process and ignored elsewhere.
        let mut local_buffers: HashMap<DeviceId, Buffer<'o>> = HashMap::new();
        let mut send_buffers: Vec<&Buffer<'o>> = Vec::new();
        let mut send_destination_devices: Vec<GlobalDeviceId> = Vec::new();
        let mut send_transfer_keys: Vec<CrossHostTransferKey> = Vec::new();
        // Receives must be grouped by destination device for the cross-host API.
        let mut receive_plans_by_device: HashMap<DeviceId, Vec<(GlobalDeviceId, CrossHostTransferKey)>> =
            HashMap::new();

        let destination_count = destination_mesh.devices().len();
        for (destination_index, destination_device) in destination_mesh.devices().iter().enumerate() {
            let destination_device_id = destination_device.id();
            let destination_local = destination_device.process_index() == client_process_index;
            let transfer_key = exact_shard_transfer_key(0, destination_index, destination_count)?;
            let destination_global_id = cross_host_global_device_id(destination_device_id)?;

            if destination_local {
                let pjrt_device =
                    destination_device_by_id.get(&destination_device_id).ok_or(ArrayError::NonAddressableDevice {
                        device_id: destination_device_id,
                        process_index: client_process_index,
                    })?;
                if let Some(local_source) = local_source_buffer.as_ref() {
                    let buffer = local_source.copy_to_device(pjrt_device.clone())?;
                    local_buffers.insert(destination_device_id, buffer);
                } else {
                    receive_plans_by_device
                        .entry(destination_device_id)
                        .or_default()
                        .push((canonical_source_global_id, transfer_key));
                }
            } else if this_process_is_canonical_sender {
                let local_source = local_source_buffer.as_ref().ok_or_else(|| {
                    Error::from(ArrayError::MissingShardBuffer {
                        shard_index: 0,
                        device_id: canonical_source_device.id(),
                    })
                })?;
                send_buffers.push(local_source.as_ref());
                send_destination_devices.push(destination_global_id);
                send_transfer_keys.push(transfer_key);
            }
            // Otherwise: remote destination served by some other process; nothing to do here.
        }

        // A receiver may have only local destinations, so an unavailable extension cannot be attributed to a remote
        // destination device. Preserve the plugin's actual extension error before issuing any transfers.
        if !send_buffers.is_empty() || !receive_plans_by_device.is_empty() {
            client.cross_host_transfers_extension()?;
        }

        // Issue cross-host sends, if any.
        if !send_buffers.is_empty() {
            client.cross_host_send_buffers(
                send_buffers.as_slice(),
                send_destination_devices.as_slice(),
                send_transfer_keys.as_slice(),
                &[],
            )?;
        }

        // Issue cross-host receives, if any. Each call covers one destination device, matching the
        // existing fast-path pattern in `transfers.rs`.
        let element_type_pjrt = self.data_type().to_pjrt();
        let shape_i64 = self.shape().as_slice().iter().map(|&size| size as i64).collect::<Vec<_>>();
        let mut receive_device_ids = receive_plans_by_device.keys().copied().collect::<Vec<_>>();
        receive_device_ids.sort_unstable();
        for destination_device_id in receive_device_ids {
            let plans = receive_plans_by_device.get(&destination_device_id).expect("receive plans present after sort");
            let pjrt_device =
                destination_device_by_id.get(&destination_device_id).ok_or(ArrayError::NonAddressableDevice {
                    device_id: destination_device_id,
                    process_index: client_process_index,
                })?;
            let element_types = plans.iter().map(|_| element_type_pjrt).collect::<Vec<_>>();
            let dimensions = plans.iter().map(|_| shape_i64.as_slice()).collect::<Vec<_>>();
            let source_devices = plans.iter().map(|(source_global_id, _)| *source_global_id).collect::<Vec<_>>();
            let transfer_keys = plans.iter().map(|(_, key)| *key).collect::<Vec<_>>();
            let received = client.cross_host_receive_buffers(
                element_types.as_slice(),
                dimensions.as_slice(),
                pjrt_device,
                source_devices.as_slice(),
                transfer_keys.as_slice(),
                &[],
            )?;
            for buffer in received {
                local_buffers.insert(destination_device_id, buffer);
            }
        }

        // Assemble per-device buffers in `destination_mesh` device order. Only locally addressable destinations
        // contribute to the intermediate Array; remote destinations are handled by their owning
        // processes via parallel calls to this function.
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
            Array::from_addressable_buffers(domain, intermediate_type, destination_mesh.clone(), buffers)?;

        // The intermediate is owned exclusively by this function and is never observed by callers.
        // Donating its buffers lets PJRT reuse their memory for the output of the final SPMD reshard.
        intermediate.reshard_on_same_mesh(domain, destination_mesh, destination_sharding, true)
    }

    /// Reshards a sharded source array onto a different destination mesh.
    ///
    /// Composes two compiled SPMD programs:
    ///
    ///   1. An all-gather on the mesh of this array that produces a fully-replicated intermediate (every device
    ///      in that mesh ends up with the full logical array).
    ///   2. [`Self::reshard_replicated_across_meshes`] then broadcasts that intermediate to `destination_mesh` and
    ///      compiles the final reshard from "replicated on `destination_mesh`" to `destination_sharding`.
    ///
    /// Both compiled programs go through `cache`'s LRU, so repeated calls with the same input/output
    /// shardings pay the compile cost once.
    fn reshard_sharded_across_meshes(
        &self,
        domain: &Domain<'o>,
        destination_mesh: &DeviceMesh,
        destination_sharding: &Sharding,
    ) -> Result<Array<'o>, Error> {
        let source_mesh = self.mesh();
        let replicated_on_source = Sharding::replicated(source_mesh.logical_mesh().clone(), self.sharding().rank());
        // Source is owned externally, so we do not donate it during the all-gather. The gathered
        // intermediate is owned by this function and is donated to the subsequent broadcast/reshard
        // step inside [`Self::reshard_replicated_across_meshes`].
        let gathered = self.clone().reshard_on_same_mesh(domain, source_mesh, &replicated_on_source, false)?;
        gathered.reshard_replicated_across_meshes(domain, destination_mesh, destination_sharding)
    }
}

/// Deterministic exact-shard transfer plan for one [`Array::to_placement`] call on the current process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExactShardPutPlan {
    /// Local destination shards that can be satisfied via intra-host device-to-device copies.
    local_copies: Vec<LocalShardCopyPlan>,

    /// Cross-host sends that this process must initiate for remote destination shards.
    cross_host_sends: Vec<CrossHostShardSendPlan>,

    /// Cross-host receives that this process must initiate for local destination shards.
    cross_host_receives: Vec<CrossHostShardReceivePlan>,
}

impl ExactShardPutPlan {
    /// Creates a deterministic exact-shard transfer plan.
    pub(crate) fn new(
        local_copies: Vec<LocalShardCopyPlan>,
        cross_host_sends: Vec<CrossHostShardSendPlan>,
        cross_host_receives: Vec<CrossHostShardReceivePlan>,
    ) -> Self {
        Self { local_copies, cross_host_sends, cross_host_receives }
    }

    /// Returns the local destination shards satisfied by intra-host device-to-device copies.
    pub(crate) fn local_copies(&self) -> &[LocalShardCopyPlan] {
        &self.local_copies
    }

    /// Returns the cross-host sends initiated by this process.
    pub(crate) fn cross_host_sends(&self) -> &[CrossHostShardSendPlan] {
        &self.cross_host_sends
    }

    /// Returns the cross-host receives initiated by this process.
    pub(crate) fn cross_host_receives(&self) -> &[CrossHostShardReceivePlan] {
        &self.cross_host_receives
    }
}

/// One exact-shard local device-to-device copy in an [`ExactShardPutPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LocalShardCopyPlan {
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
    /// Creates one exact-shard local device-to-device copy plan.
    pub(crate) fn new(
        source_shard_index: ShardIndex,
        source_device_id: DeviceId,
        destination_shard_index: ShardIndex,
        destination_device_id: DeviceId,
    ) -> Self {
        Self { source_shard_index, source_device_id, destination_shard_index, destination_device_id }
    }

    /// Returns the source shard index in the source array.
    pub(crate) fn source_shard_index(&self) -> ShardIndex {
        self.source_shard_index
    }

    /// Returns the source device ID.
    pub(crate) fn source_device_id(&self) -> DeviceId {
        self.source_device_id
    }

    /// Returns the destination device ID.
    pub(crate) fn destination_device_id(&self) -> DeviceId {
        self.destination_device_id
    }
}

/// One exact-shard cross-host send in an [`ExactShardPutPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CrossHostShardSendPlan {
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
    /// Creates one exact-shard cross-host send plan.
    pub(crate) fn new(
        source_shard_index: ShardIndex,
        source_device_id: DeviceId,
        destination_shard_index: ShardIndex,
        destination_device_id: DeviceId,
        transfer_key: CrossHostTransferKey,
    ) -> Self {
        Self { source_shard_index, source_device_id, destination_shard_index, destination_device_id, transfer_key }
    }

    /// Returns the source shard index in the source array.
    pub(crate) fn source_shard_index(&self) -> ShardIndex {
        self.source_shard_index
    }

    /// Returns the source device ID.
    pub(crate) fn source_device_id(&self) -> DeviceId {
        self.source_device_id
    }

    /// Returns the destination device ID.
    pub(crate) fn destination_device_id(&self) -> DeviceId {
        self.destination_device_id
    }

    /// Returns the deterministic transfer rendezvous key shared with the matching receive.
    pub(crate) fn transfer_key(&self) -> CrossHostTransferKey {
        self.transfer_key
    }
}

/// One exact-shard cross-host receive in an [`ExactShardPutPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CrossHostShardReceivePlan {
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
    /// Creates one exact-shard cross-host receive plan.
    pub(crate) fn new(
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
    pub(crate) fn source_device_id(&self) -> DeviceId {
        self.source_device_id
    }

    /// Returns the destination shard index in the destination array.
    pub(crate) fn destination_shard_index(&self) -> ShardIndex {
        self.destination_shard_index
    }

    /// Returns the destination device ID.
    pub(crate) fn destination_device_id(&self) -> DeviceId {
        self.destination_device_id
    }

    /// Returns the logical destination shard shape.
    pub(crate) fn destination_shape(&self) -> &StaticShape {
        &self.destination_shape
    }

    /// Returns the deterministic transfer rendezvous key shared with the matching send.
    pub(crate) fn transfer_key(&self) -> CrossHostTransferKey {
        self.transfer_key
    }
}

/// Returns the deterministic exact-shard cross-host transfer key for one source/destination pair.
pub(crate) fn exact_shard_transfer_key(
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
pub(crate) fn cross_host_global_device_id(device_id: DeviceId) -> Result<GlobalDeviceId, Error> {
    i32::try_from(device_id).map_err(|_| Error::SizeLimitExceeded {
        message: format!("cross-host transfer device ID {device_id} exceeds the maximum allowed value of {}", i32::MAX),
    })
}

/// Returns the PJRT cross-host shape for one destination shard.
fn cross_host_shape(plan: &CrossHostShardReceivePlan) -> Result<Vec<i64>, Error> {
    plan.destination_shape()
        .as_slice()
        .iter()
        .enumerate()
        .map(|(dimension, &size)| {
            i64::try_from(size).map_err(|_| {
                Error::from(ArrayError::InvalidShardDimension {
                    shard_index: plan.destination_shard_index(),
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

/// Returns the preferred exact matching source shard for `destination_shard`.
///
/// Preference is deterministic and depends only on global shard metadata so that every process
/// chooses the same source shard for a given destination shard:
/// 1. prefer a source shard on the destination process,
/// 2. then prefer the same device ID as the destination shard, and
/// 3. finally break ties by device ID and shard index.
fn preferred_exact_source_shard<'a, 'o>(
    source_shards: &[&'a ArrayShard<'o>],
    destination_shard: &ShardDescriptor,
) -> &'a ArrayShard<'o> {
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
        .expect("preferred exact source shard selection requires at least one candidate")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        DataType, Dimension as CoreDimension, DimensionError, Layout, Memory, Shape, ShardingDimension, TiledLayout,
    };
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, ExecutionFence, load_cpu_plugin};

    use crate::tests::{four_device_mesh_x, logical_mesh_2x2, values_from_bytes, values_to_bytes};
    use crate::{FromPjrt, Session};

    use super::*;

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

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_cross_host_shape_retains_dimension_error_and_shard_context() {
        let plan = CrossHostShardReceivePlan::new(0, 0, 2, 1, StaticShape::new(vec![usize::MAX]), 0);
        let error = cross_host_shape(&plan).unwrap_err();
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
    fn test_array_put_reshards_fully_addressable_array() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(client_devices[0].id().unwrap(), client_devices[0].process_index().unwrap())],
        )
        .unwrap();
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        // Use an evenly-divisible (4 elements / 2 devices) split. Uneven splits (e.g. 5/2) are not
        // currently supported by XLA's CPU SPMD partitioner — the per-device output buffer would
        // need different shapes ([3] + [2]), but the partitioner returns the full unsliced buffer
        // and ryft's descriptor model expects exact logical per-device shapes. See M9 in the plan
        // for the limitation; revisit if a future XLA release or backend handles this.
        let source_values = [0.0f32, 1.0, 2.0, 3.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(source_sharding)
            .unwrap();
        let source_array = Array::from_host_buffer(
            &domain,
            source_type,
            source_mesh,
            values_to_bytes::<f32>(source_values.as_slice()).as_slice(),
        )
        .unwrap();

        let target_devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            target_devices,
        )
        .unwrap();
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let moved_array = source_array
            .to_placement(&domain, DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding })
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
        let first_shard_bytes = moved_array
            .device_shard(client_devices[0].id().unwrap())
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let second_shard_bytes = moved_array
            .device_shard(client_devices[1].id().unwrap())
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(first_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
        assert_eq!(values_from_bytes::<f32>(first_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert_eq!(second_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
        assert_eq!(values_from_bytes::<f32>(second_shard_bytes.as_slice()), vec![2.0, 3.0]);
    }

    #[test]
    fn test_array_put_copies_matching_local_shards_without_full_source_addressability() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
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
            &domain,
            source_array_type,
            sharding.clone(),
            mesh.clone(),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array = source_array
            .to_placement(&domain, DevicePutTarget::Placement { mesh: mesh.clone(), sharding })
            .unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied_array
            .device_shard(local_device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(copied_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_plan_exact_shard_put_uses_cross_host_send_and_receive_for_remote_exact_moves() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
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
            &domain,
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

    #[test]
    fn test_array_put_rejects_non_addressable_source_shards() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 1)],
        )
        .unwrap();
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let source_array = Array::from_addressable_buffers_with_sharding(
            &domain,
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

        assert!(matches!(
            source_array.to_placement(
                &domain,
                DevicePutTarget::Placement { mesh: target_mesh, sharding: target_sharding }
            ),
            Err(Error::Array(error)) if matches!(error.as_ref(),
                ArrayError::MissingShardBuffer { shard_index: 0, device_id: 0 },
            ),
        ));
    }

    #[test]
    fn test_array_to_device_preserves_same_partially_addressable_placement() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
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
            &domain,
            source_array_type,
            sharding.clone(),
            mesh.clone(),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array = source_array
            .into_placement(&domain, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharding.clone() })
            .unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied_array
            .device_shard(local_device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(copied_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_array_reshard_on_devices() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
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
        let device_values = |array: &Array<'_>| {
            array
                .shards()
                .iter()
                .map(|shard| {
                    let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                    values_from_bytes::<f32>(bytes.as_slice())
                })
                .collect::<Vec<_>>()
        };
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
        let source = Array::from_host_buffer(&domain, r#type, mesh.clone(), values_to_bytes(&[1f32, 2., 3., 4.]))
            .unwrap()
            .with_execution_fence(ExecutionFence::new(vec![event]));
        let reduced = tiled.clone().with_reduced_axes(["x"]).unwrap();
        let output = source.clone().reshard_on_devices(&domain, &mesh, &reduced, false).unwrap();
        let pending = matches!(output.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false)));
        promise.set(None).unwrap();
        assert!(pending);
        assert_eq!(output.sharding(), &reduced);
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(shared_buffers(&source, &output), vec![true; 4]);
        assert_eq!(domain.program_count(), 0);

        // On another device mesh, the shard on a device of the source shares its buffer and the others are copies.
        let single_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[1]],
        )
        .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let source =
            Array::from_host_buffer(&domain, r#type, single_device_mesh, values_to_bytes(&[1f32, 2., 3., 4.])).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let output = source.clone().reshard_on_devices(&domain, &mesh, &replicated, false).unwrap();
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
        assert_eq!(device_values(&output), vec![vec![1., 2., 3., 4.]; 4]);
        assert_eq!(domain.program_count(), 0);

        // A sharding with other shards needs a compiled program.
        let sharded = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let resharded = output.clone().reshard_on_devices(&domain, &mesh, &sharded, false).unwrap();
        assert_eq!(resharded.sharding(), &sharded);
        assert_eq!(device_values(&resharded), vec![vec![1.], vec![2.], vec![3.], vec![4.]]);
        assert_eq!(domain.program_count(), 1);

        // There is no host fallback, so a destination that a compiled program cannot serve is an error.
        let manual_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let manual =
            Sharding::new(manual_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            output.reshard_on_devices(&domain, &manual_mesh, &manual, false).unwrap_err().to_string(),
            "compiled reshard does not support `Manual` mesh axis x",
        );
    }

    #[test]
    fn test_probe_donation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
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
        let by_x = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let by_y = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let pointers = |array: &Array<'_>| {
            array.shards().iter().map(|shard| unsafe { shard.buffer().unwrap().unsafe_pointer() }.unwrap()).collect::<Vec<_>>()
        };
        for donate in [false, true] {
            let r#type = ArrayType::new_static(DataType::F32, [4096]).with_sharding(by_x.clone()).unwrap();
            let values = (0..4096).map(|value| value as f32).collect::<Vec<_>>();
            let source = Array::from_host_buffer(&domain, r#type, mesh.clone(), values_to_bytes(&values)).unwrap();
            source.block_until_ready().unwrap();
            let before = pointers(&source);
            let target = DevicePutTarget::placement(mesh.clone(), by_y.clone()).unwrap();
            let output = if donate { source.into_placement(&domain, target).unwrap() } else { source.to_placement(&domain, target).unwrap() };
            output.block_until_ready().unwrap();
            let after = pointers(&output);
            eprintln!("donate={donate} before={before:?} after={after:?}");
        }
    }

    #[test]
    fn test_array_to_placement_combines_unreduced_device_values() {
        // The devices along the unreduced axis `x` hold summands of the logical value, and `y` tiles its only
        // dimension. Every destination shard has the slices of a source shard, but a copy of that shard would hold
        // one summand instead of the logical value.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
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
        let device_values = |array: &Array<'_>| {
            array
                .shards()
                .iter()
                .map(|shard| {
                    let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                    values_from_bytes::<f32>(bytes.as_slice())
                })
                .collect::<Vec<_>>()
        };

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
        let summands = Array::from_addressable_buffers(&domain, r#type, mesh.clone(), buffers).unwrap();
        let reduced = summands
            .to_placement(&domain, DevicePutTarget::placement(mesh.clone(), tiled.clone()).unwrap())
            .unwrap();
        assert_eq!(reduced.sharding(), &tiled);
        assert_eq!(device_values(&reduced), vec![vec![11., 22.], vec![33., 44.], vec![11., 22.], vec![33., 44.]]);

        // In the other direction, a copy of every shard would hold the logical value once per device along `x`.
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(tiled).unwrap();
        let value =
            Array::from_host_buffer(&domain, r#type, mesh.clone(), values_to_bytes(&[1f32, 2., 3., 4.])).unwrap();
        let summands =
            value.to_placement(&domain, DevicePutTarget::placement(mesh, unreduced.clone()).unwrap()).unwrap();
        assert_eq!(summands.sharding(), &unreduced);
        assert_eq!(values_from_bytes::<f32>(&summands.copy_to_host().unwrap()), vec![1., 2., 3., 4.]);
    }

    #[test]
    fn test_compiled_reshard_replicated_to_sharded_on_same_mesh() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let mesh = four_device_mesh_x(&client);

        let values = [10.0f32, 11.0, 12.0, 13.0];
        let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array =
            Array::from_host_buffer(&domain, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();

        let sharded_target =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&domain, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_target })
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        let device_ids =
            client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        for (shard_index, device_id) in device_ids.iter().copied().enumerate() {
            let shard_bytes = resharded
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            let shard_values = values_from_bytes::<f32>(shard_bytes.as_slice());
            assert_eq!(shard_values, vec![values[shard_index]]);
        }
    }

    #[test]
    fn test_zero_space_reshard_is_bufferless_and_preserves_type_metadata() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let mesh = four_device_mesh_x(&client);
        let source_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let layout = Layout::Tiled(TiledLayout::new(vec![0], Vec::new()));
        let source_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(4)]))
            .with_layout(layout.clone())
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(source_sharding)
            .unwrap();
        let source = Array::from_host_buffer(&domain, source_type, mesh.clone(), []).unwrap();
        let target_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let resharded = source
            .to_placement(&domain, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding.clone() })
            .unwrap();

        assert_eq!(resharded.sharding(), &target_sharding);
        assert_eq!(resharded.layout(), Some(&layout));
        assert_eq!(resharded.r#type().memory(), Memory::Host { pinned: true });
        assert_eq!(resharded.addressable_shards().count(), mesh.devices().len());
        assert!(resharded.addressable_shards().all(|shard| shard.buffer().is_none()));
    }

    #[test]
    fn test_zero_space_reshard_preserves_producer_completion() {
        let client = crate::tests::execution_client();
        let domain = Session::new(&client).domain();
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
                Array::from_host_buffer(&domain, ArrayType::new_static(DataType::Zero, [2]), source_mesh.clone(), [])
                    .unwrap()
                    .with_execution_fence(ExecutionFence::new(vec![event]));
            let outputs = [
                source.to_placement(&domain, target.clone()).unwrap(),
                source.clone().into_placement(&domain, target.clone()).unwrap(),
                source.clone().reshard_on_devices(&domain, &target_mesh, &target_sharding, false).unwrap(),
            ];
            // A representation-only move must neither wait for nor discard the producer's completion. Complete the event
            // before asserting so even a failing regression releases its pending native callbacks.
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
    fn test_compiled_reshard_sharded_to_replicated_on_same_mesh() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let mesh = four_device_mesh_x(&client);

        let values = [20.0f32, 21.0, 22.0, 23.0];
        let sharded_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]));
        let source_array = Array::from_host_buffer_with_sharding(
            &domain,
            source_type,
            sharded_sharding,
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        let replicated_target = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let resharded = source_array
            .to_placement(&domain, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: replicated_target })
            .unwrap();

        let device_ids =
            client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        for device_id in device_ids {
            let shard_bytes = resharded
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            let shard_values = values_from_bytes::<f32>(shard_bytes.as_slice());
            assert_eq!(shard_values, values.to_vec());
        }
    }

    #[test]
    fn test_compiled_reshard_sharded_to_differently_sharded_on_same_mesh() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
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
            &domain,
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
            .to_placement(&domain, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_along_y })
            .unwrap();
        assert_eq!(resharded.addressable_shards().count(), 4);

        for shard in resharded.addressable_shards() {
            let shard_buffer = shard.buffer().unwrap();
            let shard_bytes = shard_buffer.copy_to_host(None).unwrap().r#await().unwrap();
            let shard_values = values_from_bytes::<f32>(shard_bytes.as_slice());
            let slice = shard.descriptor().slice();
            let row_range = slice[0].clone();
            let col_range = slice[1].clone();
            let mut expected = Vec::new();
            for row in row_range.clone() {
                for column in col_range.clone() {
                    expected.push(row_values[row * 4 + column]);
                }
            }
            assert_eq!(shard_values, expected, "shard slice {:?}", slice);
        }
    }

    #[test]
    fn test_compiled_reshard_cross_mesh_replicated_source_to_sharded_destination() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();

        // Source lives on a 1-device sub-mesh.
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
            Array::from_host_buffer(&domain, source_type, source_mesh, values_to_bytes::<f32>(&values).as_slice())
                .unwrap();

        // Reshard onto the full 4-device mesh, sharded along "x".
        let target_mesh = four_device_mesh_x(&client);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&domain, DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding })
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        let device_ids = client_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        for (shard_index, device_id) in device_ids.iter().copied().enumerate() {
            let shard_bytes = resharded
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            let shard_values = values_from_bytes::<f32>(shard_bytes.as_slice());
            assert_eq!(shard_values, vec![values[shard_index]]);
        }
    }

    #[test]
    fn test_to_device_donates_source_and_returns_independently_readable_output() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let mesh = four_device_mesh_x(&client);

        let values = [300.0f32, 301.0, 302.0, 303.0];
        let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array =
            Array::from_host_buffer(&domain, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();

        // to_device consumes self and donates the source's input buffers to the compiled SPMD
        // reshard. PJRT may reuse those buffers' memory for the output; the donated source must not
        // be readable after this call (it has been consumed), but the returned array's output
        // buffers must be independently addressable on each device.
        let target_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let moved = source_array
            .into_placement(&domain, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding })
            .unwrap();

        let device_ids =
            client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        for (shard_index, device_id) in device_ids.iter().copied().enumerate() {
            let shard_bytes = moved
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), vec![values[shard_index]]);
        }
    }

    /// Validates that the second reshard with the same structural signature hits the cache and
    /// **skips** the trace + lower work — the structural cache key (input/output shardings, mesh,
    /// shape, dtype) matches without ever materializing the MLIR text. Mirrors how JAX caches
    /// `jit` invocations on abstract value signatures.
    #[test]
    #[ignore = "timing-sensitive; runs locally to validate the structural cache hit"]
    fn bench_compiled_reshard_cache_hit_avoids_trace_and_lower() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let mesh = four_device_mesh_x(&client);
        let values = (0..4096).map(|index| index as f32).collect::<Vec<_>>();
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source_array =
            Array::from_host_buffer(&domain, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let target_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let cold_start = std::time::Instant::now();
        let _ = source_array
            .clone()
            .to_placement(&domain, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding.clone() })
            .unwrap();
        let cold_elapsed = cold_start.elapsed();

        let warm_start = std::time::Instant::now();
        let _ = source_array
            .to_placement(&domain, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding })
            .unwrap();
        let warm_elapsed = warm_start.elapsed();

        eprintln!("cold reshard (trace+lower+compile+execute): {cold_elapsed:?}");
        eprintln!("warm reshard (structural cache hit + execute, no trace/lower): {warm_elapsed:?}");
        assert_eq!(domain.program_count(), 1);
    }

    #[test]
    fn test_to_placement_preserves_missing_cross_host_extension() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;

        // Source array on the local device, replicated on a 1-device sub-mesh.
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
            Array::from_host_buffer(&domain, source_type, source_mesh, values_to_bytes::<f32>(&values).as_slice())
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
            .to_placement(&domain, DevicePutTarget::Placement { mesh: target_mesh, sharding: target_sharding });
        assert!(matches!(
            result,
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));
    }

    #[test]
    fn test_to_placement_preserves_missing_cross_host_extension_for_receiver() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        assert!(matches!(client.cross_host_transfers_extension(), Err(PjrtError::Unimplemented { .. })));
        let domain = Session::new(&client).domain();
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device.id() + 1, local_device.process_index() + 1)],
        )
        .unwrap();
        let source_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source = Array::from_addressable_buffers(&domain, source_type, source_mesh, Vec::new()).unwrap();
        assert_eq!(source.addressable_shards().count(), 0);
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("destination", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![local_device],
        )
        .unwrap();
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);

        // This process only receives. Every destination is local, so there is no remote destination to blame for the
        // missing extension. Preserve that error instead of searching for a nonexistent remote destination and panicking.
        assert!(matches!(
            source.to_placement(&domain, DevicePutTarget::Placement { mesh: target_mesh, sharding: target_sharding }),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));
    }

    #[test]
    fn test_compiled_reshard_with_explicit_mesh_axes() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        // Build a 4-device mesh whose axis type is `Explicit` (sharding propagation is user-declared
        // rather than inferred). The compiled reshard path should accept this just like `Auto`.
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
            &domain,
            source_type,
            explicit_mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        let sharded_target =
            Sharding::new(explicit_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&domain, DevicePutTarget::Placement { mesh: explicit_mesh, sharding: sharded_target })
            .unwrap();
        assert_eq!(resharded.addressable_shards().count(), 4);

        let device_ids =
            client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        for (shard_index, device_id) in device_ids.iter().copied().enumerate() {
            let shard_bytes = resharded
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), vec![values[shard_index]]);
        }
    }

    #[test]
    fn test_to_with_manual_mesh_axes_uses_host_fallback() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
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
            &domain,
            source_type,
            manual_mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // Manual mesh axes cannot be planned by the SPMD partitioner — the compiled path declines.
        // `Array::to` falls through to the host materialization fallback (full source is addressable,
        // destination is on the same single process) and successfully reshapes the data to the
        // requested sharding.
        let sharded_target =
            Sharding::new(manual_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(
                &domain,
                DevicePutTarget::Placement { mesh: manual_mesh.clone(), sharding: sharded_target.clone() },
            )
            .expect("host fallback should satisfy the reshard when both endpoints are local");
        assert_eq!(resharded.sharding(), &sharded_target);
        assert_eq!(resharded.addressable_shards().count(), 4);

        // Each device-local shard now owns one element of the input vector.
        for (index, device) in client.addressable_devices().unwrap().iter().enumerate() {
            let shard_bytes = resharded
                .device_shard(device.id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), vec![values[index]]);
        }

        // No executable was compiled — the host fallback bypasses the compile cache.
        assert_eq!(domain.program_count(), 0, "host fallback should not populate the executable cache");
    }

    fn two_device_sub_mesh_x(client: &ryft_pjrt::Client<'_>) -> DeviceMesh {
        let client_devices = client.addressable_devices().unwrap();
        let devices =
            client_devices.iter().take(2).map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
            .unwrap()
    }

    #[test]
    fn test_compiled_reshard_cross_mesh_sharded_source_to_replicated_destination() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();

        // Source: sharded along "x" on a 2-device sub-mesh (devices 0 and 1 each hold half the data).
        let source_mesh = two_device_sub_mesh_x(&client);
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let values = [70.0f32, 71.0, 72.0, 73.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]));
        let source_array = Array::from_host_buffer_with_sharding(
            &domain,
            source_type,
            source_sharding,
            source_mesh,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // Destination: replicated on the full 4-device mesh. The sharded source first all-gathers on
        // source_mesh, broadcasts onto destination_mesh, and (since the intermediate sharding matches the
        // requested replicated destination_sharding) no further reshard is needed.
        let target_mesh = four_device_mesh_x(&client);
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
        let resharded = source_array
            .to_placement(&domain, DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding })
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        let device_ids =
            client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        for device_id in device_ids {
            let shard_bytes = resharded
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), values.to_vec());
        }
    }

    #[test]
    fn test_compiled_reshard_cross_mesh_sharded_source_to_sharded_destination() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();

        // Source: sharded along "x" on a 2-device sub-mesh.
        let source_mesh = two_device_sub_mesh_x(&client);
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let values = [80.0f32, 81.0, 82.0, 83.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]));
        let source_array = Array::from_host_buffer_with_sharding(
            &domain,
            source_type,
            source_sharding,
            source_mesh,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // Destination: sharded along "x" on the full 4-device mesh. Each destination shard holds
        // exactly one element of the global array.
        let target_mesh = four_device_mesh_x(&client);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&domain, DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding })
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        let device_ids =
            client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        for (shard_index, device_id) in device_ids.iter().copied().enumerate() {
            let shard_bytes = resharded
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), vec![values[shard_index]]);
        }
    }

    #[test]
    fn test_compiled_reshard_cross_mesh_sharded_source_compiles_two_executables() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();

        let source_mesh = two_device_sub_mesh_x(&client);
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let values = [90.0f32, 91.0, 92.0, 93.0];
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]));
        let source_array = Array::from_host_buffer_with_sharding(
            &domain,
            source_type,
            source_sharding,
            source_mesh,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        let target_mesh = four_device_mesh_x(&client);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(domain.program_count(), 0);
        let _ = source_array
            .to_placement(&domain, DevicePutTarget::Placement { mesh: target_mesh, sharding: target_sharding })
            .unwrap();

        // Sub-case B compiles two executables: an all-gather on source_mesh and a replicated->sharded
        // reshard on destination_mesh.
        assert_eq!(domain.program_count(), 2);
    }

    #[test]
    fn test_fast_path_replicated_cross_mesh_to_replicated_destination() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();

        // Source: replicated on a 1-device sub-mesh.
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
            Array::from_host_buffer(&domain, source_type, source_mesh, values_to_bytes::<f32>(&values).as_slice())
                .unwrap();

        // Target: replicated on the full 4-device mesh. The fast path matches every destination shard
        // to the source's single full-array shard. With the bitcast branch removed, copy_to_device
        // produces independent buffers on each device and copy_to_host works on every one of them.
        let target_mesh = four_device_mesh_x(&client);
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
        let resharded = source_array
            .to_placement(&domain, DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding })
            .unwrap();

        assert_eq!(resharded.addressable_shards().count(), 4);
        for device in client_devices.iter() {
            let shard_bytes = resharded
                .device_shard(device.id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), values.to_vec());
        }
        // The fast path satisfies this request without compiling any SPMD program.
        assert_eq!(domain.program_count(), 0);
    }

    #[test]
    fn test_compiled_reshard_caches_executable_across_calls() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let mesh = four_device_mesh_x(&client);

        let values = [30.0f32, 31.0, 32.0, 33.0];
        let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array =
            Array::from_host_buffer(&domain, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();

        let sharded_target =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(domain.program_count(), 0);
        let first = source_array
            .clone()
            .to_placement(&domain, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_target.clone() })
            .unwrap();
        let after_first = domain.program_count();
        let second = source_array
            .to_placement(&domain, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_target })
            .unwrap();
        let after_second = domain.program_count();

        assert_eq!(first.addressable_shards().count(), 4);
        assert_eq!(second.addressable_shards().count(), 4);
        assert_eq!(after_first, 1, "first reshard should populate the executable cache");
        assert_eq!(after_second, 1, "second reshard with identical MLIR should reuse the cached executable");
    }
}
