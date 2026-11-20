use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use ryft_core::{
    ArrayType, Cast, CastOperation, Concatenate, DataType, Device, DeviceMesh, DimensionError, Layout, Memory,
    MeshAxisType, Placeholder, Placement, ProgramError, Sharding, Slice, StaticShape, Typed,
};
use ryft_pjrt::extensions::cross_host_transfers::{CrossHostTransferKey, GlobalDeviceId};
use ryft_pjrt::{
    Buffer, BufferType, Client, Device as PjrtDevice, DeviceId, Error as PjrtError, Event, ExecutionFence,
};

use crate::arrays::host::{PendingHostUpload, spawn_host_uploads};
use crate::arrays::placement::ArrayPlacement;
use crate::arrays::shards::{ArrayShard, ShardDescriptor, ShardIndex, ShardLayout};
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
    /// When the resolved placement and requested storage match this array, returns a cheap clone of `self`.
    /// A change of devices or sharding tries the following strategies in order.
    ///
    ///   1. **Exact Shards:** When every destination shard has the slices of one source shard, each destination shard
    ///      shares the buffer of its source shard on the same device, or is filled by a copy between devices or by a
    ///      PJRT cross-host transfer for a source shard on a remote host.
    ///   2. **Shard Regions:** On different meshes, when both shardings hold one value per slice, the source devices
    ///      slice out regions needed by each destination shard. The destination devices receive those regions
    ///      and join them locally. This path uses default device memory and needs the PJRT cross-host transfers
    ///      extension and default layouts if a region crosses processes. Each participating process issues the same
    ///      placements in the same order.
    ///   3. **Compiled Reshard:** An `identity(x) = x` program with input and output sharding constraints is lowered
    ///      to StableHLO and Shardy, compiled through the compilation cache of `session`, and executed. This needs
    ///      shards of one size on both sides, so a split whose parts do not divide the extent of its dimension (e.g.,
    ///      five elements over two devices, which [`ShardLayout`] splits into shards of three and two elements) skips
    ///      this strategy as well as the shard regions strategy.
    ///   4. **Host Fallback:** The global array is copied to the host as [`Array::copy_to_host`] does and uploaded
    ///      again as [`Array::from_host_buffer`] does, so the result has the shards that a host upload of the same
    ///      values with the destination sharding has. This is used only when the device work fails or is skipped
    ///      (e.g., for `Manual` mesh axes or uneven shards), when the destination mesh is fully addressable from this
    ///      process and this process holds the whole value of the source, and when the array does not vary along
    ///      manual mesh axes, whose devices can hold different values for the same slice that a dense host copy cannot
    ///      represent. This process holds the whole value when it addresses every source shard, or when the source has
    ///      no unreduced axes and its addressable shards cover the array (e.g., a replicated source with some replicas
    ///      on other processes). The destination buffers are created up front and the readback and the upload continue
    ///      on a background thread, so this call returns without waiting for them. A failure of that background work,
    ///      whose error also describes the failure of the device work, surfaces through the readiness functions and
    ///      the readbacks of the result and through [`Session::synchronize`]. Only when the PJRT plugin lacks host to
    ///      device transfer managers or their literal transfers does this call block until the readback completes.
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
    /// another PJRT client is rejected.
    ///
    /// When the host fallback also cannot satisfy the request, typically because some source shards that the value
    /// needs live on a remote process or because the source varies along manual mesh axes, the original error of the
    /// device work is propagated as the most informative diagnostic.
    ///
    /// The destination uses the requested [`Memory`], or the source memory if omitted. PJRT cross-host receives land
    /// in device memory and can then be copied to the requested memory on that device. An explicit [`Layout`] applies
    /// to each destination shard, and a source with a known layout keeps it when the layout is omitted. Layout changes
    /// use a compiled conversion on each shard's device when supported. Otherwise, each affected local shard is read
    /// back and uploaded in the requested layout on a background thread, as for the host fallback, and this call only
    /// blocks on the readback when the PJRT plugin lacks host to device transfer managers or their literal transfers.
    /// Both paths preserve the separate contributions along unreduced axes.
    ///
    /// Use [`Self::into_placement`] when the source array is no longer needed after the move. It consumes `self` and
    /// lets PJRT donate the source buffers to the destination, which avoids the extra allocation.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `target`: Resolved [`Placement`] or an [`ArrayPlacement`] request specifying the destination.
    pub fn to_placement<P: Into<ArrayPlacement>>(&self, session: &Session, target: P) -> Result<Self, Error> {
        let target = target.into().resolve_for(session, self)?;
        self.clone().run_placement_dispatch(session, &target, false)
    }

    /// Moves this array to the provided placement, consuming `self` and donating its source buffers to the
    /// destination. This is the `ryft` analogue of JAX's `jax.device_put(arr, sharding, donate=True)`.
    ///
    /// Donation lets PJRT reuse the source buffers for the output, which saves the memory of the source array for
    /// large training tensors. Because `self` is consumed at the language boundary, the donated buffers are not
    /// observable to the caller after this function returns. Rust ownership therefore enforces the same invariant that
    /// the `donate=True` flag carries in JAX, without exposing a runtime hazard.
    ///
    /// The strategies follow [`Self::to_placement`], enabling donation for the compiled reshard on the source mesh.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `target`: Resolved [`Placement`] or an [`ArrayPlacement`] request specifying the destination.
    pub fn into_placement<P: Into<ArrayPlacement>>(self, session: &Session, target: P) -> Result<Self, Error> {
        let target = target.into().resolve_for(session, &self)?;
        self.run_placement_dispatch(session, &target, true)
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
        session: &Session,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        donate: bool,
    ) -> Result<Self, Error> {
        let memory = self.r#type().memory();
        self.reshard_on_devices_or_return(session, mesh, sharding, memory, donate)
            .map_err(|(error, _)| error)
    }

    /// Returns an array with the type and the placement of this array whose addressable shards are copies of the
    /// shards of this array on the same devices and in the same memories, so that the result shares no buffer with
    /// any other array. The execution fence of the result joins the readiness of every copy with the fence of this
    /// array, because a readback reads only one replica of each tile and would otherwise succeed with a replica whose
    /// copy did not fail.
    pub(crate) fn with_copied_shard_buffers(&self, session: &Session) -> Result<Self, Error> {
        let mut work = session.reserve_work();
        let mut completion = PlacementCompletion::default();
        let buffers = self
            .addressable_shards()
            .filter_map(ArrayShard::buffer)
            .map(|buffer| {
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
        Ok(output)
    }

    /// Places this array on the devices, with the sharding, and in the storage that `target` names. A memory or layout
    /// that `target` omits keeps the memory or layout of this array. The result always belongs to `session`, so this
    /// array is first associated with `session`, which rejects an array of another PJRT client, even when the
    /// placement leaves the array where it is.
    ///
    /// An array whose whole value this process holds can also use a host round trip when the device reshard fails and
    /// returns the source array. That round trip returns the result before the readback completes and continues the
    /// readback and the upload on a background thread, whose failures surface through the readiness functions and the
    /// readbacks of the result and through [`Session::synchronize`]. It blocks until the readback completes only when
    /// the PJRT plugin lacks host to device transfer managers or their literal transfers. Whatever this function can
    /// reject before reading back (e.g., a destination mesh that this process does not fully address or a source whose
    /// value this process does not hold) is still reported by this function with the error of the device work. That
    /// includes a source that varies along manual mesh axes, whose per-device values a dense host copy cannot preserve.
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
        let layout = target.layout().or(array.layout()).cloned();
        if array.mesh() == &mesh && array.sharding() == &sharding {
            return array.place_storage(session, memory, layout);
        }

        // Keep the error of the device work, which is the most informative one if the host fallback fails too.
        let (compiled_error, source) =
            match array.reshard_on_devices_or_return(session, &mesh, &sharding, memory, donate) {
                Ok(array) => return array.place_storage(session, memory, layout),
                Err((error, Some(source))) => (error, source),
                Err((error, None)) => return Err(error),
            };

        // The fallback uploads the dense readback directly into the requested storage. If it cannot complete
        // from this process, surface the original device error. That includes a source that varies along manual mesh
        // axes, whose devices can hold different values for the same slice. The dense readback keeps only one value
        // per slice, so uploading it would silently replace the values of the other devices.
        let local_process = session.client().process_index()?;
        let destination_fully_addressable = mesh.devices().iter().all(|device| device.process_index() == local_process);
        if !destination_fully_addressable
            || !source.sharding().varying_manual_axes().is_empty()
            || source.validate_host_readback().is_err()
        {
            return Err(compiled_error);
        }
        let host_type =
            ArrayType::new(source.data_type(), source.shape().into()).with_memory(memory).with_layout(layout);

        // Prefer to read back and upload in the background. Its failure poisons the result with an error that also
        // describes the failure of the device work, because that failure is what made the host fallback necessary.
        let readback_source = source.clone();
        let device_error = compiled_error.to_string();
        let uploaded = Self::from_host_buffer_in_background(
            session,
            host_type.clone(),
            mesh.clone(),
            sharding.clone(),
            move || readback_source.copy_to_host_bytes(),
            move |error| {
                PjrtError::internal(format!(
                    "the host fallback of an array placement failed with: {error} (the device work that it replaced \
                    failed with: {device_error})",
                ))
            },
        );
        match uploaded {
            Ok(Some(array)) => return Ok(array),
            Ok(None) => {}
            Err(_) => return Err(compiled_error),
        }

        // Without host to device transfer managers, read back and upload on this thread.
        let host_bytes = match source.copy_to_host_bytes() {
            Ok(bytes) => bytes,
            Err(_) => return Err(compiled_error),
        };
        match Self::from_host_buffer(session, host_type, mesh, Some(sharding), host_bytes.as_slice()) {
            Ok(array) => Ok(array),
            Err(_) => Err(compiled_error),
        }
    }

    /// Applies storage requirements to the shards already on their destination devices. A layout change first tries a
    /// compiled conversion on the device and can fall back to a host round trip for that shard. Once a buffer has the
    /// requested layout, it can be reused or copied directly to the requested memory.
    ///
    /// The host round trips of all shards share one background thread, which reads each shard back and uploads it
    /// into a destination buffer that this function creates up front, so this function returns without waiting for
    /// them. A failed readback or upload poisons the destination buffers that have not received their bytes, so the
    /// readiness functions and readbacks of the result and [`Session::synchronize`] report it. The execution fence of
    /// the result joins the fences of this array with the readiness of every buffer that this function creates (i.e.,
    /// the copies to another memory and the buffers of host round trips) and with the fences of the compiled
    /// conversions, because a readback reads only one replica of each tile and would otherwise succeed with a replica
    /// that did not fail. This function blocks on the readback of a shard only when the PJRT plugin lacks host to
    /// device transfer managers or their literal transfers.
    fn place_storage(self, session: &Session, memory: Memory, layout: Option<Layout>) -> Result<Self, Error> {
        let layout = layout.or_else(|| self.layout().cloned());
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

            // Register the submission before the first copy, transfer manager, or background thread, so that a
            // concurrent `Session::synchronize` snapshot includes every upload that may already be running. The
            // destination buffers of the background uploads are attached only after their thread has started, and a
            // failure to start it drops this reservation with the work that was attached up to then.
            let mut work = session.reserve_work();
            let mut buffers = Vec::new();
            let mut uploads = Vec::new();
            let mut upload_sources = Vec::new();
            let mut upload_buffers = Vec::new();
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
                // The upload continues in the background when the plugin can hand out its destination buffer first.
                let shape = shard.shape();
                let dimensions = shape.as_slice().iter().map(|&size| size as u64).collect::<Vec<_>>();
                let client = session.client();
                let destination = PendingHostUpload::new(
                    client,
                    buffer_type,
                    dimensions.clone(),
                    destination_memory,
                    target_layout.clone(),
                )?;
                if let Some((upload, placed)) = destination {
                    let placed = Arc::new(placed);
                    uploads.push(upload);
                    upload_sources.push((Arc::clone(source_buffer), shape));
                    upload_buffers.push(Arc::clone(&placed));
                    buffers.push(placed);
                    continue;
                }
                let bytes = read_shard_to_host(source_buffer, &shape, |copy| work.track_host_copy(copy).wait())?;
                let placed = client.buffer(
                    Arc::<[u8]>::from(bytes),
                    buffer_type,
                    dimensions,
                    None,
                    destination_memory,
                    target_layout.clone(),
                )?;
                work.track_buffer(&placed);
                completion.track_buffer(&placed)?;
                buffers.push(Arc::new(placed));
            }
            if !uploads.is_empty() {
                // The background thread owns its own handles of the source buffers, so they stay alive until it has
                // read them back even if the source array is dropped first. It starts before anything observes the
                // destination buffers that it fills, because nothing else can complete them, and observing them can
                // wait for them (e.g., a fence built from their events awaits an event whose callback PJRT fails to
                // register). Their readiness joins the reservation and the result only once the thread is running.
                spawn_host_uploads(
                    uploads,
                    move |index| {
                        let (source_buffer, shape) = &upload_sources[index];
                        read_shard_to_host(source_buffer, shape, Event::r#await)
                    },
                    |error| match error {
                        Error::Pjrt(error) => error,
                        error => PjrtError::internal(error.to_string()),
                    },
                )?;
                for buffer in &upload_buffers {
                    work.track_buffer(buffer);
                    completion.track_buffer(buffer)?;
                }
            }
            Self::from_addressable_buffers_internal(session, r#type, sharding, mesh, buffers, false)?
        };
        output.execution_fence = completion.join(source.execution_fence);
        output.send_fence = source.send_fence;
        Ok(output)
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

    /// Places this array on `mesh` according to `sharding` without compiling a program, or returns `None` when that
    /// placement needs one. An array of [`DataType::Zero`](ryft_core::DataType::Zero) has no buffers, so only its
    /// placement metadata changes. For any other array, [`Self::exact_shard_buffers`] provides the destination
    /// shards and the layout that the result records. The execution fence of the result joins the execution fence of
    /// this array, so that errors of the execution that produced this array still surface through the result, with
    /// the readiness of every buffer that this placement copies or receives and the fences of the conversions that it
    /// runs, because a readback reads only one replica of each tile and would otherwise succeed with a replica whose
    /// copy failed. The result also retains the completion fence of the cross-host sends that this process issued, so
    /// that the readiness functions of the result surface the errors of those sends.
    fn place_exact_shards(
        &self,
        session: &Session,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        memory: Memory,
    ) -> Result<Option<Self>, Error> {
        let mut completion = PlacementCompletion::default();
        let mut output = if self.data_type().is_zero() {
            let r#type = self.r#type().into_owned().with_memory(memory).with_sharding(None)?;
            Self::zero(session, r#type, Some(sharding.clone()), mesh.clone())?
        } else {
            let shape = self.shape();
            let buffers = self.exact_shard_buffers(session, &shape, mesh, sharding, memory, &mut completion)?;
            let Some((buffers, layout, send_fence)) = buffers else {
                return Ok(None);
            };
            let r#type = self.r#type().into_owned().with_shape(shape).with_memory(memory).with_layout(layout);
            let output = Self::from_addressable_buffers_internal(
                session,
                r#type.with_sharding(None)?,
                Some(sharding.clone()),
                mesh.clone(),
                buffers,
                false,
            )?;
            match send_fence {
                Some(fence) => output.with_send_fence(fence),
                None => output,
            }
        };
        output.execution_fence = completion.join(self.execution_fence().cloned());
        Ok(Some(output))
    }

    /// Returns the buffers of the destination shards that `session` can address when every destination shard has the
    /// slices of one source shard and both shardings hold one value per slice (refer to
    /// [`Self::plan_exact_shard_put`]). A shard can share its source buffer when the device and memory match.
    /// Other local source shards are copied to the destination memory, and remote source shards arrive through
    /// the PJRT cross-host transfers extension. The result is `None` when the destination needs other shards, the
    /// plugin lacks the required extension or cannot report the default layout of a shard that crosses processes, or a
    /// source shard that this process sends cannot be converted to that layout on its device.
    ///
    /// A cross-host transfer moves the raw bytes of a buffer, and its receiver allocates the buffer in the default
    /// layout of the plugin for the shape of the shard. A source shard in another layout is therefore converted to
    /// that default layout on its device before it is sent, and the layout that the result records is the layout of
    /// this array only when every received buffer has it too. Otherwise, the result records no layout, and the caller
    /// applies the requested layout afterwards (e.g., [`Self::place_storage`]).
    ///
    /// The buffers come with the recorded layout and with the completion fence of the cross-host sends that this
    /// process issues for the destination shards of other processes, or `None` when it issues none. This function
    /// never waits for those sends, because a send completes only once its receiver issues the matching receive, and
    /// the receiver may be this same process further below or a process that is itself waiting for a send of its own.
    /// The caller retains the fence in the placed array so that [`Self::ready`] and [`Self::block_until_ready`] surface
    /// send failures. Every process derives the same plan from global metadata, so each send has exactly one matching
    /// receive, and [`issue_cross_host_transfers`] issues the transfers of all processes in one global order.
    ///
    /// `completion` records the readiness of every buffer that this function copies or receives and the fences of
    /// the conversions that it runs, which the caller joins into the execution fence of the result.
    fn exact_shard_buffers(
        &self,
        session: &Session,
        global_shape: &StaticShape,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        memory: Memory,
        completion: &mut PlacementCompletion,
    ) -> Result<Option<(Vec<Arc<Buffer<'static>>>, Option<Layout>, Option<ExecutionFence>)>, Error> {
        let client = session.client();
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

        // Resolve the default layout of every shard that crosses processes before the first copy or rendezvous. The
        // sender and the receiver of a shard query the same plugin for the same shape, so both decline alike when the
        // plugin cannot report it.
        let element_type = self.data_type().executable_buffer_type();
        let cross_host_shards = plan
            .cross_host_sends()
            .iter()
            .map(|send_plan| (send_plan.source_shard_index(), self.shards()[send_plan.source_shard_index()].shape()))
            .chain(plan.cross_host_receives().iter().map(|receive_plan| {
                (receive_plan.destination_shard_index(), receive_plan.destination_shape().clone())
            }));
        let mut default_layouts = HashMap::new();
        for (shard_index, shape) in cross_host_shards {
            if let std::collections::hash_map::Entry::Vacant(entry) = default_layouts.entry(shape) {
                let Some(layout) = cross_host_default_layout(client, element_type, shard_index, entry.key())? else {
                    return Ok(None);
                };
                entry.insert(layout);
            }
        }

        // Convert the source shards that this process sends before it copies or transfers anything, so that a
        // conversion that cannot be compiled can still decline this strategy.
        let mut transfers = Vec::with_capacity(plan.cross_host_sends().len() + plan.cross_host_receives().len());
        for send_plan in plan.cross_host_sends() {
            let shard = self.addressable_device_shard(send_plan.source_device_id()).ok_or_else(|| {
                Error::from(ArrayError::MissingShardBuffer {
                    shard_index: send_plan.source_shard_index(),
                    device_id: send_plan.source_device_id(),
                })
            })?;
            let layout = &default_layouts[&shard.shape()];
            let Some(buffer) = self.cross_host_send_buffer(session, shard, layout, completion)? else {
                return Ok(None);
            };
            transfers.push(CrossHostTransfer::Send {
                source: send_plan.source_device_id(),
                destination: send_plan.destination_device_id(),
                transfer_key: send_plan.transfer_key(),
                buffer,
            });
        }

        let addressable_devices = client.addressable_devices()?;
        let mut addressable_device_by_id = HashMap::with_capacity(addressable_devices.len());
        for device in addressable_devices {
            addressable_device_by_id.insert(device.id()?, device);
        }

        let source_memory = self.r#type().memory();
        let mut work = session.reserve_work();
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
            let same_device = local_copy_plan.source_device_id() == local_copy_plan.destination_device_id();
            if same_device && memory == source_memory {
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
            // Copying to the destination device itself would always allocate in its default memory.
            let destination_memory = destination_device.memory(memory)?;
            if same_device && source_buffer.memory()? == destination_memory {
                addressable_buffers.push(Arc::clone(source_buffer));
                continue;
            }
            let copied = source_buffer.copy_to_memory(destination_memory)?;
            work.track_buffer(&copied);
            completion.track_buffer(&copied)?;
            addressable_buffers.push(Arc::new(copied));
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
                element_type: receive_plan.element_type(),
                dimensions: receive_plan.cross_host_shape()?,
            });
        }
        let (send_fence, received_buffers) = issue_cross_host_transfers(session, transfers, memory)?;
        for received in received_buffers {
            completion.track_buffer(&received.buffer)?;
            addressable_buffers.push(Arc::new(received.buffer));
        }

        // Local copies keep the layout of their source shards, while received buffers have the default layout.
        let received_layouts = plan
            .cross_host_receives()
            .iter()
            .map(|receive_plan| &default_layouts[receive_plan.destination_shape()]);
        let layout = exact_shards_layout(self.layout(), received_layouts)?;
        Ok(Some((addressable_buffers, layout, send_fence)))
    }

    /// Returns the buffer of `shard`, which is an addressable shard of this array that this process sends to another
    /// process, in `layout`, which is the default layout that the receiver allocates for its shape. A cross-host
    /// transfer moves the raw bytes of the buffer, so a buffer in another layout is converted on its device with
    /// [`Self::try_relayout_shard_on_device`], and `completion` records the fence of that conversion. Returns `None`
    /// when the conversion cannot be compiled (e.g., when this process cannot compile operations for its own shards
    /// independently), so that a later strategy can place the array instead.
    fn cross_host_send_buffer(
        &self,
        session: &Session,
        shard: &ArrayShard,
        layout: &ryft_pjrt::Layout,
        completion: &mut PlacementCompletion,
    ) -> Result<Option<Arc<Buffer<'static>>>, Error> {
        let buffer = shard
            .buffer()
            .ok_or(ArrayError::MissingShardBuffer { shard_index: shard.index(), device_id: shard.device().id() })?;
        let current_layout = buffer.specification()?.layout;
        if current_layout == *layout {
            return Ok(Some(Arc::clone(buffer)));
        }
        if !can_compile_local_operations(session) {
            return Ok(None);
        }
        let input_type = ArrayType::new(self.data_type(), shard.shape().into())
            .with_memory(self.r#type().memory())
            .with_layout(Layout::from_pjrt(current_layout)?);
        let layout = Layout::from_pjrt(layout.clone())?;
        let converted =
            Self::try_relayout_shard_on_device(session, shard.device(), Arc::clone(buffer), input_type, &layout)?;
        Ok(converted.map(|converted| {
            completion.track_fence(converted.execution_fence());
            Arc::clone(converted.shards()[0].buffer().unwrap())
        }))
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
        if !holds_one_value_per_slice(self.sharding()) || !holds_one_value_per_slice(sharding) {
            return Ok(None);
        }

        let mut source_shards_by_slices = HashMap::<Vec<Range<usize>>, Vec<&ArrayShard>>::new();
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
            let source_shard = preferred_source_shard(source_shards.as_slice(), destination_shard);
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
                        self.data_type().executable_buffer_type(),
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

    /// Places an array on a different mesh by slicing on the source devices and joining the received regions on
    /// the destination devices. Each process derives the same region plan before filtering it to its own devices.
    ///
    /// Returns `None` when the shardings can assign different values to the same slice, the type requires a
    /// nondefault layout or memory, a destination is empty, compilation artifacts are exchanged across processes,
    /// or the plugin lacks the cross-host APIs required by the plan.
    fn reshard_shard_regions(
        &self,
        session: &Session,
        mesh: &DeviceMesh,
        sharding: &Sharding,
    ) -> Result<Option<Self>, Error> {
        if !holds_one_value_per_slice(self.sharding())
            || !holds_one_value_per_slice(sharding)
            || self.r#type().memory() != Memory::Device
            || self.r#type().layout().is_some()
            || !std::ptr::eq(self.client(), session.client())
        {
            return Ok(None);
        }

        if !can_compile_local_operations(session) {
            return Ok(None);
        }

        let layout = ShardLayout::new(&self.shape(), mesh, sharding)?;
        let regions = self.plan_shard_regions(&layout);
        if regions.iter().any(Vec::is_empty) {
            return Ok(None);
        }
        let destinations = layout.descriptors();
        let client = session.client();
        let process_index = client.process_index()?;
        let Some(remote_layouts) = self.cross_host_region_layouts(client, destinations, &regions)? else {
            return Ok(None);
        };

        let devices = client
            .addressable_devices()?
            .into_iter()
            .map(|device| Ok((device.id()?, device)))
            .collect::<Result<HashMap<_, _>, PjrtError>>()?;
        let data_type = self.data_type();
        let mut completion = PlacementCompletion::default();
        let mut work = session.reserve_work();
        let mut transfers = Vec::new();
        let mut destination_buffers = HashMap::new();
        for (destination, pieces) in destinations.iter().zip(&regions) {
            let device = destination.device();
            let destination_local = device.process_index() == process_index;
            for piece in pieces {
                let source = &self.shards()[piece.source];
                let source_local = source.device().process_index() == process_index;
                let transfer_key = exact_shard_transfer_key(piece.source, destination.index(), destinations.len())?;
                let buffer = source_local
                    .then(|| {
                        let remote_layout = (!destination_local).then(|| &remote_layouts[&piece.shape()]);
                        Self::prepare_shard_region(session, data_type, source, piece, remote_layout, &mut completion)
                    })
                    .transpose()?;
                match (buffer, destination_local) {
                    (Some(buffer), true) => {
                        let buffer = if source.device() == device {
                            buffer
                        } else {
                            let destination_device = devices
                                .get(&device.id())
                                .ok_or(ArrayError::NonAddressableDevice { device_id: device.id(), process_index })?;
                            let copied = buffer.copy_to_device(destination_device.clone())?;
                            work.track_buffer(&copied);
                            completion.track_buffer(&copied)?;
                            Arc::new(copied)
                        };
                        destination_buffers.insert(transfer_key, buffer);
                    }
                    (Some(buffer), false) => transfers.push(CrossHostTransfer::Send {
                        source: source.device().id(),
                        destination: device.id(),
                        transfer_key,
                        buffer,
                    }),
                    (None, true) => {
                        let destination_device = devices
                            .get(&device.id())
                            .ok_or(ArrayError::NonAddressableDevice { device_id: device.id(), process_index })?;
                        transfers.push(CrossHostTransfer::Receive {
                            source: source.device().id(),
                            destination: device.id(),
                            transfer_key,
                            device: destination_device,
                            element_type: data_type.executable_buffer_type(),
                            dimensions: cross_host_shape(destination.index(), &piece.shape())?,
                        });
                    }
                    (None, false) => {}
                }
            }
        }
        let (send_fence, received_buffers) = issue_cross_host_transfers(session, transfers, Memory::Device)?;
        for received in received_buffers {
            completion.track_buffer(&received.buffer)?;
            destination_buffers.insert(received.transfer_key, Arc::new(received.buffer));
        }

        let mut buffers = Vec::new();
        for (destination, regions) in destinations.iter().zip(&regions) {
            let device = destination.device();
            if device.process_index() != process_index {
                continue;
            }
            let pieces = regions
                .iter()
                .map(|region| {
                    let key = exact_shard_transfer_key(region.source, destination.index(), destinations.len())?;
                    // Every region of a local destination has exactly one local copy or cross-host receive.
                    let buffer = destination_buffers.remove(&key).ok_or(ArrayError::MissingShardBuffer {
                        shard_index: destination.index(),
                        device_id: device.id(),
                    })?;
                    Ok((region, buffer))
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
        Ok(Some(output))
    }

    /// Finds a source for each nonempty intersection with a destination shard. Identical source slices are replicas.
    /// The chosen replica prefers the destination device, then another device on the same process.
    fn plan_shard_regions(&self, layout: &ShardLayout) -> Vec<Vec<ShardRegion>> {
        let mut replicas = HashMap::<&[Range<usize>], Vec<&ArrayShard>>::new();
        for shard in self.shards() {
            replicas.entry(shard.slice()).or_default().push(shard);
        }
        layout
            .descriptors()
            .iter()
            .map(|destination| {
                let mut regions = Vec::new();
                for (source_slice, source_shards) in &replicas {
                    if !source_slice.iter().zip(destination.slice()).all(|(source, destination)| {
                        source.start.max(destination.start) < source.end.min(destination.end)
                    }) {
                        continue;
                    }
                    let slice = source_slice
                        .iter()
                        .zip(destination.slice())
                        .map(|(source, destination)| {
                            source.start.max(destination.start)..source.end.min(destination.end)
                        })
                        .collect::<Vec<_>>();
                    let source = preferred_source_shard(source_shards, destination);
                    regions.push(ShardRegion { source: source.index(), slice });
                }
                regions.sort_unstable_by(|left, right| {
                    left.slice.iter().map(|range| range.start).cmp(right.slice.iter().map(|range| range.start))
                });
                regions
            })
            .collect()
    }

    /// Resolves each remote region's default layout on every rank before the first cross-host rendezvous.
    /// Returns `None` when the client lacks the cross-host transfer or default layout API.
    fn cross_host_region_layouts(
        &self,
        client: &Client<'_>,
        destinations: &[ShardDescriptor],
        regions: &[Vec<ShardRegion>],
    ) -> Result<Option<HashMap<StaticShape, ryft_pjrt::Layout>>, Error> {
        let mut layouts = HashMap::new();
        let element_type = self.data_type().executable_buffer_type();
        for (destination, regions) in destinations.iter().zip(regions) {
            for region in regions {
                let source = self.shards()[region.source].device();
                if source.process_index() == destination.device().process_index() {
                    continue;
                }
                if layouts.is_empty() {
                    match client.cross_host_transfers_extension() {
                        Ok(_) => {}
                        Err(PjrtError::Unimplemented { .. }) => return Ok(None),
                        Err(error) => return Err(error.into()),
                    }
                }
                cross_host_global_device_id(source.id())?;
                cross_host_global_device_id(destination.device().id())?;
                if let std::collections::hash_map::Entry::Vacant(entry) = layouts.entry(region.shape()) {
                    let Some(layout) =
                        cross_host_default_layout(client, element_type, destination.index(), entry.key())?
                    else {
                        return Ok(None);
                    };
                    entry.insert(layout);
                }
            }
        }
        Ok(Some(layouts))
    }

    /// Produces the requested region on the source device, using the default PJRT layout for a remote transfer.
    /// `completion` records the fences of the programs that produce it.
    fn prepare_shard_region(
        session: &Session,
        data_type: DataType,
        source: &ArrayShard,
        region: &ShardRegion,
        remote_layout: Option<&ryft_pjrt::Layout>,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        let source_buffer = source
            .buffer()
            .ok_or(ArrayError::MissingShardBuffer { shard_index: source.index(), device_id: source.device().id() })?;
        let mut buffer = if source.slice() == region.slice {
            Arc::clone(source_buffer)
        } else {
            let input = Self::wrap_device_buffer(
                session,
                ArrayType::new(data_type, source.shape().into()),
                source.device(),
                Arc::clone(source_buffer),
            )?;
            let starts = source
                .slice()
                .iter()
                .zip(&region.slice)
                .map(|(shard, region)| region.start - shard.start)
                .collect::<Vec<_>>();
            let limits = starts.iter().zip(&region.slice).map(|(start, range)| start + range.len()).collect::<Vec<_>>();
            let output = input.slice(&starts, &limits, &vec![1; starts.len()])?;
            completion.track_fence(output.execution_fence());
            Arc::clone(output.shards()[0].buffer().unwrap())
        };
        // A receive declares the region's shape and uses the plugin's default device layout.
        if let Some(layout) = remote_layout
            && buffer.specification()?.layout != *layout
        {
            let r#type = ArrayType::new(data_type, region.shape().into());
            let input = Self::wrap_device_buffer(session, r#type, source.device(), Arc::clone(&buffer))?;
            let r#type = input.r#type().into_owned().with_layout(Layout::from_pjrt(layout.clone())?);
            let output = input.cast(r#type, &[])?;
            completion.track_fence(output.execution_fence());
            buffer = Arc::clone(output.shards()[0].buffer().unwrap());
        }
        Ok(buffer)
    }

    /// Concatenates the transferred regions of a shard on its destination device. `completion` records the fences of
    /// the concatenations.
    fn assemble_shard_regions(
        session: &Session,
        data_type: DataType,
        destination: &ShardDescriptor,
        mut pieces: Vec<(&ShardRegion, Arc<Buffer<'static>>)>,
        completion: &mut PlacementCompletion,
    ) -> Result<Arc<Buffer<'static>>, Error> {
        if pieces.len() == 1 {
            return Ok(pieces.pop().unwrap().1);
        }
        let mut values = pieces
            .into_iter()
            .map(|(region, buffer)| {
                let r#type = ArrayType::new(data_type, region.shape().into());
                let value = Self::wrap_device_buffer(session, r#type, destination.device(), buffer)?;
                Ok((region, value))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        // The intersections form a Cartesian grid, sorted by their starting coordinates. Join the innermost
        // dimension first so that the pieces agree along every other dimension for each concatenation.
        for axis in (0..destination.slice().len()).rev() {
            let mut joined = Vec::new();
            for group in values.chunk_by(|(left, _), (right, _)| left.slice[..axis] == right.slice[..axis]) {
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
    /// array when a host fallback may still place it. It tries the following strategies in order.
    ///
    ///   1. **Exact Shards:** [`Self::place_exact_shards`] shares or copies whole shards and compiles nothing. Its
    ///      errors are final, so they come without this array.
    ///   2. **Shard Regions:** [`Self::reshard_shard_regions`] transfers the parts that each destination needs across
    ///      different meshes.
    ///   3. **Compiled Reshard:** The `ryft` analogue of how JAX's `jax.device_put(arr, new_sharding)` lowers a
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
        session: &Session,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        memory: Memory,
        donate: bool,
    ) -> Result<Self, (Error, Option<Self>)> {
        match self.place_exact_shards(session, mesh, sharding, memory) {
            Ok(Some(output)) => return Ok(output),
            Ok(None) => {}
            Err(error) => return Err((error, None)),
        }
        if let Err(error) = self.validate_compiled_reshard(session, mesh, sharding) {
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
            match self.reshard_shard_regions(session, mesh, sharding) {
                Ok(Some(output)) => return Ok(output),
                Ok(None) => {}
                Err(error) => return Err((error, Some(self))),
            }
            // A change of memory can use device storage for the intermediate compiled reshard. The destination
            // then moves the completed shards to the requested tier, without requiring SPMD execution there.
            let source_memory = self.r#type().memory();
            let intermediate_memory = if memory == source_memory { memory } else { Memory::Device };
            let output = if self.sharding().is_replicated() {
                self.reshard_replicated_across_meshes(session, mesh, sharding, intermediate_memory)
            } else {
                self.reshard_sharded_across_meshes(session, mesh, sharding, intermediate_memory)
            };
            output.map_err(|error| (error, Some(self)))
        }
    }

    /// Checks that a compiled program can place this array on `destination_mesh` with `destination_sharding`. The
    /// errors are [`ArrayError::ManualReshardingAxis`] when the destination mesh has a `Manual` axis, while `Auto` and
    /// `Explicit` axes are supported, [`ArrayError::UnevenReshardingShards`] when the shards of this array or of the
    /// destination differ in size, and [`ArrayError::MissingShardBuffer`] when a source shard on the current process
    /// has no buffer.
    ///
    /// The SPMD partitioner splits a dimension into parts of one size and pads the last parts, so a dimension whose
    /// extent the number of its parts does not divide (e.g., five elements over two devices) yields buffers whose
    /// shapes differ from the exact shards that [`ShardLayout`] describes (i.e., `[3]` and `[2]`). Such a placement
    /// therefore uses whole shards or the host fallback instead, which materialize exactly those shards.
    fn validate_compiled_reshard(
        &self,
        session: &Session,
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
        session: &Session,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        donate: bool,
    ) -> Result<Self, Error> {
        let reshard = self.compile_reshard_on_same_mesh(session, mesh, sharding, donate)?;
        reshard.executable().execute(session, self)
    }

    /// Compiles the identity program that reshards this array, which must already be on `destination_mesh`, to
    /// `destination_sharding`. The input and the output of the program are declared in the memory of this array, so
    /// the program accepts this array and returns its result in the same memory. If `donate` is `true`, the input of
    /// the program is marked as donated so that PJRT can reuse the memory of its buffers for the output buffers.
    ///
    /// The compilation cache is keyed by the complete lowered computation, including its input and output sharding
    /// annotations and PJRT options.
    fn compile_reshard_on_same_mesh(
        &self,
        session: &Session,
        destination_mesh: &DeviceMesh,
        destination_sharding: &Sharding,
        donate: bool,
    ) -> Result<CompiledFunction<ArrayType, ArrayType>, Error> {
        let element_type = self.data_type();
        let shape = self.shape();
        let source_sharding = self.sharding().clone();
        let destination_sharding = destination_sharding.clone();

        // Build the bare input type, which has no sharding, for tracing. The input and output shardings are attached as
        // `sdy.sharding` attributes through `Options::with_input_shardings` and `Options::with_output_shardings`. The
        // type keeps the memory of this array, which the identity carries over to the output.
        let bare_input_type = ArrayType::new(element_type, shape.clone().into()).with_memory(self.r#type().memory());

        let options = Options::new(destination_mesh.clone())
            .with_input_shardings(vec![source_sharding.clone()])
            .with_output_shardings(vec![destination_sharding.clone()])
            .with_donate(donate);

        let staged: crate::StagedFunction<ArrayType, ArrayType> =
            crate::stage(|input| input, bare_input_type, options)?;
        session.compile_staged_function(staged)
    }

    /// Copies this fully replicated array onto every device of `destination_mesh`, using copies between the devices of
    /// this process and PJRT cross-host transfers between processes, and then reshards the copy on
    /// `destination_mesh` to `destination_sharding` with [`Self::reshard_on_same_mesh`]. The copy and the result use
    /// the destination `memory`. The execution fence of the result also joins the readiness of every copied or
    /// received replica. The result retains the completion fence of the cross-host sends that this process issued, so
    /// that [`Self::ready`] and [`Self::block_until_ready`] surface their errors without this function waiting for them.
    fn reshard_replicated_across_meshes(
        &self,
        session: &Session,
        destination_mesh: &DeviceMesh,
        destination_sharding: &Sharding,
        memory: Memory,
    ) -> Result<Array, Error> {
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
            self.mesh().devices().first().copied().ok_or_else(|| ProgramError::InvalidArgument {
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

        // Decide the transfer plan of each destination device. Local destinations are served by `copy_to_memory` when
        // this process owns a device of the source mesh, or by a cross-host receive otherwise. Remote destinations of
        // processes that own no device of the source mesh are served by cross-host sends from the process of the
        // canonical sender, and every other remote destination is ignored.
        let mut work = session.reserve_work();
        let mut completion = PlacementCompletion::default();
        let mut local_buffers: HashMap<DeviceId, Buffer<'static>> = HashMap::new();
        let mut transfers = Vec::new();
        // Receive the executable buffer type that the canonical sender sends (e.g., a predicate for one-bit integers).
        let element_type = self.data_type().executable_buffer_type();
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
                    let buffer = local_source.copy_to_memory(pjrt_device.memory(memory)?)?;
                    work.track_buffer(&buffer);
                    completion.track_buffer(&buffer)?;
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
                    buffer: Arc::clone(local_source),
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
        let (send_fence, received_buffers) = issue_cross_host_transfers(session, transfers, memory)?;
        for received in received_buffers {
            completion.track_buffer(&received.buffer)?;
            local_buffers.insert(received.destination, received.buffer);
        }

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
        let intermediate_type = ArrayType::new(element_type, shape.into())
            .with_memory(memory)
            .with_sharding(replicated_on_destination)?;
        let intermediate =
            Array::from_addressable_buffers(session, intermediate_type, destination_mesh.clone(), None, buffers)?;

        // The intermediate is owned exclusively by this function and is never observed by callers.
        // Donating its buffers lets PJRT reuse their memory for the output of the final SPMD reshard.
        let mut output = intermediate.reshard_on_same_mesh(session, destination_mesh, destination_sharding, true)?;
        let reshard_fence = output.execution_fence.take();
        output.execution_fence = completion.join(reshard_fence);
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
        session: &Session,
        destination_mesh: &DeviceMesh,
        destination_sharding: &Sharding,
        memory: Memory,
    ) -> Result<Array, Error> {
        let source_mesh = self.mesh();
        let replicated_on_source = Sharding::replicated(source_mesh.logical_mesh().clone(), self.sharding().rank());
        // The source array is owned externally, so it is not donated to the all-gather. The gathered intermediate is
        // owned by this function and is donated to the subsequent reshard inside
        // `Self::reshard_replicated_across_meshes`.
        let gathered = self.clone().reshard_on_same_mesh(session, source_mesh, &replicated_on_source, false)?;
        gathered.reshard_replicated_across_meshes(session, destination_mesh, destination_sharding, memory)
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

/// Intersection of one source shard and a destination shard, in global array coordinates.
struct ShardRegion {
    /// Index of the selected source shard.
    source: ShardIndex,

    /// Slice covered by the region.
    slice: Vec<Range<usize>>,
}

impl ShardRegion {
    /// Returns the static shape of this region.
    fn shape(&self) -> StaticShape {
        StaticShape::new(self.slice.iter().map(Range::len).collect::<Vec<_>>())
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

    /// Element type of the received buffer, which is the executable buffer type of the array (e.g., a predicate for
    /// one-bit integers), because that is the buffer that the matching send transfers.
    element_type: BufferType,

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
        element_type: BufferType,
        transfer_key: CrossHostTransferKey,
    ) -> Self {
        Self {
            source_shard_index,
            source_device_id,
            destination_shard_index,
            destination_device_id,
            destination_shape,
            element_type,
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

    /// Returns the element type of the received buffer.
    fn element_type(&self) -> BufferType {
        self.element_type
    }

    /// Returns the deterministic transfer rendezvous key shared with the matching send.
    fn transfer_key(&self) -> CrossHostTransferKey {
        self.transfer_key
    }

    /// Returns the PJRT cross-host shape of the destination shard.
    fn cross_host_shape(&self) -> Result<Vec<i64>, Error> {
        cross_host_shape(self.destination_shard_index(), self.destination_shape())
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

    /// Device that owns the buffer.
    destination: DeviceId,

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

/// Returns the layout that an array placed as whole shards records, given the declared `layout` of its source and the
/// layouts of the buffers that this process received for it. Local copies keep the layout of their source shards, so
/// the declared layout holds for them, and it is kept only when every received buffer has it too. Otherwise, the
/// buffers of the array have different layouts, and the array records no layout rather than a wrong one.
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
/// PJRT delivers a received buffer to the default memory of its destination device. When `memory` is another memory,
/// each received buffer is therefore copied on its device to `memory`, and the copy is returned in its place. That
/// copy is local to the device and does not wait for the transfer, so it does not affect the order described below.
///
/// The transfers are issued one at a time in the order of their source and then destination device IDs, keeping the
/// given order of the transfers between the same two devices. The CUDA plugin carries the transfers between two
/// devices over a communicator of that pair and pairs them by the order in which both sides issue them, ignoring their
/// transfer keys, and the first transfer between two devices blocks the calling thread until the peer process issues
/// its matching transfer. Two processes that each send before they receive would therefore deadlock when they swap
/// shards. Every process derives the same transfers from global metadata and issues its own in this one global order,
/// so the first pair of devices with a pending transfer always has both of its processes waiting for it.
fn issue_cross_host_transfers(
    session: &Session,
    mut transfers: Vec<CrossHostTransfer<'_>>,
    memory: Memory,
) -> Result<(Option<ExecutionFence>, Vec<ReceivedShardBuffer>), Error> {
    if transfers.is_empty() {
        return Ok((None, Vec::new()));
    }
    transfers.sort_by_key(CrossHostTransfer::devices);
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
                for buffer in buffers {
                    receives.track_buffer(&buffer);
                    let buffer = if memory == Memory::Device {
                        buffer
                    } else {
                        let placed = buffer.copy_to_memory(device.memory(memory)?)?;
                        receives.track_buffer(&placed);
                        placed
                    };
                    received_buffers.push(ReceivedShardBuffer { transfer_key, destination, buffer });
                }
            }
        }
    }
    Ok((sends.finish(), received_buffers))
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

/// Returns whether replicas of each shard slice hold one value across the mesh.
fn holds_one_value_per_slice(sharding: &Sharding) -> bool {
    sharding.unreduced_axes().is_empty() && sharding.varying_manual_axes().is_empty()
}

/// Returns the preferred replica among `source_shards` that cover the same slice.
///
/// The preference is deterministic and depends only on global shard metadata, so that every process chooses the same
/// source shard for a given destination shard. It prefers a source shard on the process of the destination shard,
/// then a source shard on the device of the destination shard, and finally breaks ties by device ID and shard index.
fn preferred_source_shard<'o>(source_shards: &[&'o ArrayShard], destination_shard: &ShardDescriptor) -> &'o ArrayShard {
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
    use std::time::Duration;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        CompilationArtifactExchange, CompilationArtifactExchangeError, CompilationArtifactExchangePolicy,
        CompilationCache, DataType, Dimension, DimensionError, Layout, LogicalMesh, Memory, MeshAxis, Shape,
        ShardingDimension, TiledLayout,
    };
    use ryft_pjrt::{BufferSpecification, BufferType, ExecutionFence, HostToDeviceTransferManager};

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
        // A compiled reshard cannot produce shards of different sizes, so an uneven split along an explicit or an auto
        // axis has the shards that a host upload of the same values with the same sharding has, and so does the
        // replicated array that gathers those shards again.
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
            let donated = source.into_placement(&session, target).unwrap();
            for placed in [&split, &donated] {
                assert_eq!(placed.sharding(), &sharded);
                assert_eq!(shard_values(placed), vec![vec![0, 1, 2], vec![3, 4]]);
            }
            let gathered =
                split.to_placement(&session, ArrayPlacement::mesh(mesh, replicated.clone()).unwrap()).unwrap();
            assert_eq!(gathered.sharding(), &replicated);
            assert_eq!(shard_values(&gathered), vec![vec![0, 1, 2, 3, 4]; 2]);
        }
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

            // The host fallback, which manual mesh axes require, uploads to the memory of the source.
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

        // The readback of the first shard fails, which poisons both destination buffers with its error.
        managers[0].set_error(0, PjrtError::aborted("source upload failed")).unwrap();
        managers[1]
            .transfer_data(0, Arc::new(values_to_bytes(&[5f32, 6., 7., 8.])), 0, true)
            .unwrap()
            .r#await()
            .unwrap();
        for shard in placed.addressable_shards() {
            assert!(matches!(
                shard.buffer().unwrap().ready().unwrap().r#await(),
                Err(PjrtError::Aborted { message, .. }) if message == "source upload failed",
            ));
        }
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
        let regions = source.plan_shard_regions(&layout);
        // Each column needs two regions. Two of the eight regions are already on their destination devices.
        let copied_bytes = layout
            .descriptors()
            .iter()
            .zip(&regions)
            .flat_map(|(destination, regions)| {
                regions
                    .iter()
                    .filter(|region| source.shards()[region.source].device() != destination.device())
                    .map(|region| region.shape().as_slice().iter().product::<usize>() * size_of::<f32>())
            })
            .sum::<usize>();
        assert_eq!(copied_bytes, 48);

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
    fn test_array_to_placement_with_manual_mesh_axes_uses_host_fallback() {
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

        // The SPMD partitioner cannot plan manual mesh axes, so the compiled reshard declines. `Array::to_placement`
        // falls through to the host fallback, because the full source is addressable and the destination is on the
        // same single process, and successfully places the data according to the requested sharding.
        let sharded_target =
            Sharding::new(manual_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&session, ArrayPlacement::mesh(manual_mesh.clone(), sharded_target.clone()).unwrap())
            .unwrap();
        assert_eq!(resharded.sharding(), &sharded_target);
        assert_eq!(resharded.addressable_shards().count(), 4);

        // Each device-local shard now owns one element of the input vector.
        assert_eq!(shard_values(&resharded), vec![vec![40.0], vec![41.0], vec![42.0], vec![43.0]]);

        // No executable was compiled, because the host fallback bypasses the compilation cache.
        assert_eq!(session.cached_program_count(), 0, "host fallback should not populate the executable cache");
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
        let copied = source.with_copied_shard_buffers(&session).unwrap();
        assert_eq!(copied.r#type(), source.r#type());
        assert_eq!(copied.mesh(), source.mesh());
        assert_eq!(copied.sharding(), source.sharding());
        assert_eq!(
            copied
                .shards()
                .iter()
                .zip(source.shards())
                .map(|(copied, source)| Arc::ptr_eq(copied.buffer().unwrap(), source.buffer().unwrap()))
                .collect::<Vec<_>>(),
            vec![false; 2],
        );
        assert_eq!(shard_memory_kinds(&copied), ["pinned_host"; 2]);
        assert_eq!(shard_values(&copied), vec![vec![1., 2., 3., 4.]; 2]);
    }

    #[test]
    fn test_array_with_copied_shard_buffers_fails_readback_of_failed_replica() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);

        // Only the copy of the second replica fails, and a readback reads one replica of each tile, so the result must
        // report the failure through its fence.
        let (source, managers) = pending_replicated_vector(&session, &mesh);
        let copied = source.with_copied_shard_buffers(&session).unwrap();
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
            session.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(_, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "second replica failed"),
        ));
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
        let sent = source.cross_host_send_buffer(&session, shard, &default_layout, &mut completion).unwrap().unwrap();
        assert!(Arc::ptr_eq(&sent, shard.buffer().unwrap()));
        assert!(completion.join(None).is_none());
        assert_eq!(session.cached_program_count(), 0);

        // A transfer moves the raw bytes of the buffer, so a shard in another layout is converted on its device first.
        let source = upload(&session, &column_major);
        let shard = source.addressable_shards().next().unwrap();
        let mut completion = PlacementCompletion::default();
        let sent = source.cross_host_send_buffer(&session, shard, &default_layout, &mut completion).unwrap().unwrap();
        completion.join(None).unwrap().block_until_ready().unwrap();
        assert_eq!(session.cached_program_count(), 1);
        assert_eq!(sent.specification().unwrap().layout, default_layout);
        assert_eq!(sent.copy_to_host(None).unwrap().r#await().unwrap(), values_to_bytes(&values));

        // A process that cannot compile operations for its own shards declines, so that a later strategy can place the
        // array instead.
        let shared_session = peer_exchange_session(client);
        let source = upload(&shared_session, &column_major);
        let shard = source.addressable_shards().next().unwrap();
        let mut completion = PlacementCompletion::default();
        assert!(
            source
                .cross_host_send_buffer(&shared_session, shard, &default_layout, &mut completion)
                .unwrap()
                .is_none()
        );
        assert_eq!(shared_session.cached_program_count(), 0);
    }

    #[test]
    fn test_array_plan_exact_shard_put_uses_cross_host_send_and_receive_for_remote_moves() {
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
                    BufferType::F32,
                    3,
                )],
            ))
        );
    }

    #[test]
    fn test_array_plan_exact_shard_put_receives_executable_buffer_type() {
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
        let plan = source
            .plan_exact_shard_put(client.process_index().unwrap(), &source.shape(), &target_mesh, &sharding)
            .unwrap()
            .unwrap();
        assert_eq!(
            plan.cross_host_receives(),
            [CrossHostShardReceivePlan::new(
                1,
                remote_device_id,
                1,
                local_device_id,
                StaticShape::new(vec![2]),
                BufferType::Predicate,
                3,
            )],
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_cross_host_shard_receive_plan_cross_host_shape_retains_dimension_error_and_shard_context() {
        let plan = CrossHostShardReceivePlan::new(0, 0, 2, 1, StaticShape::new(vec![usize::MAX]), BufferType::F32, 0);
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
