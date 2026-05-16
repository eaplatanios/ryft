use std::collections::HashMap;
use std::sync::Arc;

use ryft_core::{ArrayType, DeviceMesh, Shape, Sharding, Size, check_sharding};
use ryft_pjrt::{Buffer, DeviceId};

use crate::arrays_v0::{
    DevicePutTarget, ExecuteArguments, compiled_reshard, copy_addressable_destination_shards_from_exact_source_shards,
};
use crate::{Array, ArrayError, CompilationContext, ToMlir};

impl<'o> Array<'o> {
    /// Moves or copies this array to the provided placement.
    ///
    /// This is the `ryft` analogue of applying JAX's `device_put(array, sharding)` or
    /// `Array.to_device(sharding)` to an existing array. The method tries two strategies in
    /// order, both fully on device:
    ///
    /// 1. **Exact-shard fast path** — when every destination shard is exactly one source shard,
    ///    satisfy each with a direct device-to-device copy or, for shards on remote hosts, a PJRT
    ///    cross-host transfer.
    /// 2. **Compiled-XLA path** — trace `identity(x) = x` with explicit input and output
    ///    `with_sharding_constraint` ops, lower to StableHLO + Shardy, compile via the cache on
    ///    `context`, and execute. Matches the behavior of JAX's `jax.device_put` for committed
    ///    arrays, including the all-gather + broadcast composition for sharded cross-mesh
    ///    reshards.
    ///
    /// Unsupported requests (Manual mesh axes, cross-host destination devices, etc.) surface as
    /// typed [`ArrayError`] variants rather than degrading silently to a host round-trip.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`CompilationContext`] wrapping the PJRT client. Caches the compiled
    ///     resharding executable across repeated calls.
    ///   - `mesh`: Concrete destination mesh describing the device topology.
    ///   - `sharding`: Sharding to apply over `mesh`.
    pub fn to_placement(
        &self,
        context: &CompilationContext<'o>,
        mesh: DeviceMesh,
        sharding: Sharding,
    ) -> Result<Self, ArrayError> {
        let client = context.client();
        check_sharding!(&mesh, &sharding);
        let global_shape = self.shape();
        let global_dimensions = global_shape.as_slice();
        if let Some(addressable_buffers) =
            copy_addressable_destination_shards_from_exact_source_shards(self, client, &global_shape, &mesh, &sharding)?
        {
            let shape = Shape::new(global_dimensions.iter().copied().map(Size::Static).collect());
            let array_type = ArrayType::new(self.data_type(), shape, None, Some(sharding))?;
            return Ok(Self::from_addressable_buffers(array_type, mesh, addressable_buffers)?);
        }

        compiled_reshard::reshard(self, context, &mesh, &sharding)
    }

    /// Moves or copies this array to the provided placement, consuming `self`.
    ///
    /// This is the closest `ryft` analogue to JAX's
    /// [`jax.Array.to_device`](https://docs.jax.dev/en/latest/_autosummary/jax.Array.to_device.html).
    /// When the resolved placement matches the current placement, the method returns `self`
    /// unchanged. Otherwise it runs the same dispatch as [`Array::to_placement`] but with input
    /// buffer **donation** enabled: the compiled SPMD reshard may reuse `self`'s device-side
    /// memory for the output buffers, saving the source-array footprint for large training
    /// arrays. Because `self` is consumed at the language boundary, the donated buffers are no
    /// longer observable to the caller after this method returns.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`CompilationContext`] wrapping the PJRT client used to materialize any new
    ///     destination buffers.
    ///   - `device`: Destination placement for this array.
    pub fn to_device(self, context: &CompilationContext<'o>, device: DevicePutTarget) -> Result<Self, ArrayError> {
        let current_mesh = self.mesh();
        let current_sharding = self.sharding().clone();
        let (target_mesh, target_sharding) = device.resolve(current_sharding.rank())?;
        if current_mesh == target_mesh && current_sharding == target_sharding {
            return Ok(self);
        }
        check_sharding!(&target_mesh, &target_sharding);
        let global_shape = self.shape();
        let global_dimensions = global_shape.as_slice();
        let client = context.client();
        if let Some(addressable_buffers) = copy_addressable_destination_shards_from_exact_source_shards(
            &self,
            client,
            &global_shape,
            &target_mesh,
            &target_sharding,
        )? {
            let shape = Shape::new(global_dimensions.iter().copied().map(Size::Static).collect());
            let array_type = ArrayType::new(self.data_type(), shape, None, Some(target_sharding))?;
            return Ok(Self::from_addressable_buffers(array_type, target_mesh, addressable_buffers)?);
        }
        compiled_reshard::reshard_with_donation(&self, context, &target_mesh, &target_sharding, true)
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
