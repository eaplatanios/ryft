use std::collections::HashMap;
use std::sync::Arc;

use ryft_core::{ArrayType, DataType, DeviceMesh, Shape, Sharding, Size, Typed, check_sharding};
use ryft_pjrt::{Buffer, DeviceId};

use crate::arrays_v0::host::materialize_dense_array_bytes;
use crate::arrays_v0::{
    DevicePutTarget, ExecuteArguments, compiled_reshard, copy_addressable_destination_shards_from_exact_source_shards,
};
use crate::experimental::domains::XlaDomain;
use crate::{Array, ArrayError, ToMlir};

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
    ///   - `engine`: [`XlaDomain`] wrapping the PJRT client and the compile-program cache.
    ///   - `target`: Resolved into a destination [`DeviceMesh`] + [`Sharding`].
    pub fn to_placement(&self, engine: &XlaDomain<'o>, target: DevicePutTarget) -> Result<Self, ArrayError> {
        let (target_mesh, target_sharding) = target.resolve(self.sharding().rank())?;
        if self.mesh() == target_mesh && self.sharding() == &target_sharding {
            return Ok(self.clone());
        }
        self.run_placement_dispatch(engine, target_mesh, target_sharding, false)
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
    ///   - `engine`: [`XlaDomain`] wrapping the PJRT client and the compile-program cache.
    ///   - `target`: Resolved into a destination [`DeviceMesh`] + [`Sharding`].
    pub fn into_placement(self, engine: &XlaDomain<'o>, target: DevicePutTarget) -> Result<Self, ArrayError> {
        let (target_mesh, target_sharding) = target.resolve(self.sharding().rank())?;
        if self.mesh() == target_mesh && self.sharding() == &target_sharding {
            return Ok(self);
        }
        self.run_placement_dispatch(engine, target_mesh, target_sharding, true)
    }

    /// Three-tier placement dispatch shared by [`Self::to_placement`] and [`Self::into_placement`].
    /// Borrows `self` so the consuming caller's `self` can stay alive until this returns. The
    /// `donate` flag is forwarded to the compiled SPMD path; the host fallback ignores it (the
    /// host roundtrip cannot reuse device buffers anyway).
    fn run_placement_dispatch(
        &self,
        engine: &XlaDomain<'o>,
        target_mesh: DeviceMesh,
        target_sharding: Sharding,
        donate: bool,
    ) -> Result<Self, ArrayError> {
        let client = engine.client()?;
        check_sharding!(&target_mesh, &target_sharding);

        let global_shape = self.shape();
        let global_dimensions = global_shape.as_slice();
        if self.data_type() == DataType::Zero {
            let r#type = self.r#type().into_owned().with_sharding(target_sharding)?;
            return Ok(Self::from_zero_space(client, r#type, target_mesh)?);
        }

        // Tier 1: exact-shard fast path. Doesn't materialize anything new on host.
        if let Some(addressable_buffers) = copy_addressable_destination_shards_from_exact_source_shards(
            self,
            client,
            &global_shape,
            &target_mesh,
            &target_sharding,
        )? {
            let shape = Shape::new(global_dimensions.iter().copied().map(Size::Static).collect());
            let array_type = ArrayType::new(self.data_type(), shape).with_sharding(target_sharding)?;
            return Ok(Self::from_addressable_buffers(client, array_type, target_mesh, addressable_buffers)?);
        }

        // Tier 2: compiled-XLA SPMD path. Captures whatever error the path produces so we can
        // surface the most informative diagnostic if both tier 2 and tier 3 fail.
        let compiled_error =
            match compiled_reshard::reshard_with_donation(self, engine, &target_mesh, &target_sharding, donate) {
                Ok(array) => return Ok(array),
                Err(error) => error,
            };

        // Tier 3: host fallback. Materializes every shard on host and reuploads via
        // `from_host_buffer`. Only engages when both endpoints are fully addressable from this
        // process — JAX's host-roundtrip path has the same restriction. If any condition fails,
        // surface the compiled-path error as the more informative diagnostic.
        let local_process = client.process_index().map_err(crate::Error::from)?;
        let destination_fully_addressable =
            target_mesh.devices().iter().all(|device| device.process_index() == local_process);
        if !destination_fully_addressable {
            return Err(compiled_error);
        }
        let host_bytes = match materialize_dense_array_bytes(self) {
            Ok(bytes) => bytes,
            Err(_) => return Err(compiled_error),
        };
        let shape = Shape::new(global_dimensions.iter().copied().map(Size::Static).collect());
        let host_type = ArrayType::new(self.data_type(), shape).with_sharding(target_sharding.clone())?;
        match Self::from_host_buffer(client, host_type, target_mesh, host_bytes.as_slice()) {
            Ok(array) => Ok(array),
            Err(_) => Err(compiled_error),
        }
    }

    /// Renders the Shardy tensor sharding attribute (`#sdy.sharding<...>`) implied by this array.
    ///
    /// Uses the canonical `@mesh` symbol name.
    pub fn to_shardy_tensor_sharding_attribute(&self) -> Result<String, ryft_mlir::Error> {
        let context = ryft_mlir::Context::new();
        self.sharding().to_mlir(context.unknown_location()).map(|attribute| attribute.to_string())
    }

    /// Converts distributed arrays to per-device execution arguments for [`ryft_pjrt::LoadedExecutable::execute`].
    ///
    /// Inputs are generated in `addressable_device_ids` order. The resulting [`ExecuteArguments`] can be converted
    /// to `Vec<ExecutionDeviceInputs>` via [`ExecuteArguments::as_execution_device_inputs`].
    pub fn into_execute_arguments(
        arrays: Vec<Self>,
        addressable_device_ids: &[DeviceId],
    ) -> Result<ExecuteArguments<'o>, ArrayError> {
        let donation_flags = vec![false; arrays.len()];
        ExecuteArguments::from_arrays_with_donation(arrays, addressable_device_ids, donation_flags.as_slice())
    }

    /// Same as [`Array::into_execute_arguments`] but with explicit per-input donation flags.
    pub fn into_execute_arguments_with_donation(
        arrays: Vec<Self>,
        addressable_device_ids: &[DeviceId],
        donation_flags: &[bool],
    ) -> Result<ExecuteArguments<'o>, ArrayError> {
        ExecuteArguments::from_arrays_with_donation(arrays, addressable_device_ids, donation_flags)
    }

    pub(crate) fn into_addressable_buffers_by_device(self) -> HashMap<DeviceId, Arc<Buffer<'o>>> {
        self.shards()
            .iter()
            .filter_map(|shard| {
                let (descriptor, buffer) = shard.clone().into_parts();
                let device_id = descriptor.device().id();
                buffer.map(|buffer| (device_id, buffer))
            })
            .collect()
    }
}
