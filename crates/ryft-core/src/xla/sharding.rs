//! This module provides the core data structures for representing how arrays are _sharded_ (or _partitioned_) across
//! devices in a multi-device or multi-host environment. The design mirrors [JAX's sharding model][jax-sharding] and
//! supports conversion to [Shardy][shardy] MLIR dialect attributes for annotating StableHLO programs.
//!
//! [jax-sharding]: https://docs.jax.dev/en/latest/jax.sharding.html
//! [shardy]: https://openxla.org/shardy/overview
//!
//! # Relationship to JAX and Shardy
//!
//! The types in this module correspond directly to their JAX and Shardy (OpenXLA) counterparts:
//!
//! | Ryft type              | JAX equivalent                         | Shardy MLIR representation                 |
//! | ---------------------- | -------------------------------------- | ------------------------------------------ |
//! | [`LogicalMesh`]        | `Mesh` axes + shape only               | `sdy.mesh @name = <["axis"=size, ...]>`    |
//! | [`DeviceMesh`]         | [`jax.sharding.Mesh`][jax-mesh]        | `sdy.mesh @name = <["axis"=size, ...]>`    |
//! | [`MeshAxis`]           | One entry in `Mesh.shape`              | `MeshAxisAttr` (name + size pair)          |
//! | [`MeshDevice`]         | One element in `Mesh.devices`          | Device ID in `MeshAttr.device_ids`         |
//! | [`Sharding`] | [`jax.sharding.NamedSharding`][jax-ns] | `#sdy.sharding<@mesh, [dim_shardings...]>` |
//! | [`ShardingDimension`]     | One element of a `PartitionSpec`-like payload | `DimensionShardingAttr`                    |
//! | [`Shard`]   | `jax.Shard` from `array.global_shards` | runtime metadata only                      |
//!
//! [jax-mesh]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Mesh
//! [jax-ns]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.NamedSharding
//!
//! # Logical mesh vs concrete mesh
//!
//! [`LogicalMesh`] captures only the logical topology (axis names and sizes) and is used
//! wherever device identity is irrelevant - principally in [`Sharding`] and for
//! rendering Shardy MLIR attributes at compilation time.
//!
//! [`DeviceMesh`] wraps a [`LogicalMesh`] and adds a concrete device list, which is needed at
//! runtime for computing per-device [`Shard`] metadata.
//!
//! # Generic vs specialized Shardy lowering
//!
//! The generic [`Sharding`] renderer emits fully explicit Shardy shardings, where
//! replicated dimensions are closed (`{}`). Specialized lowering sites such as `shard_map` can
//! still compute open dimensions (`{?}`) when that is required by Shardy's manual-computation
//! semantics.
//!
//! # Practical usage
//!
//! The typical workflow for sharded computation mirrors JAX's model:
//!
//! 1. **Create a device mesh** that organizes available devices into a named logical grid:
//!
//!    ```ignore
//!    // 1D mesh for data parallelism across 8 devices.
//!    // JAX equivalent: Mesh(devices, ('batch',))
//!    let mesh = DeviceMesh::new(
//!        LogicalMesh::new(vec![MeshAxis::new("batch", 8, MeshAxisType::Auto)?])?,
//!        mesh_devices,
//!    )?;
//!
//!    // 2D mesh for data + model parallelism.
//!    // JAX equivalent: Mesh(np.array(devices).reshape(4, 2), ('data', 'model'))
//!    let mesh = DeviceMesh::new(
//!        LogicalMesh::new(vec![
//!            MeshAxis::new("data", 4, MeshAxisType::Auto)?,
//!            MeshAxis::new("model", 2, MeshAxisType::Auto)?,
//!        ])?,
//!        mesh_devices,
//!    )?;
//!    ```
//!
//! 2. **Create shardings** that describe how each array dimension maps to mesh axes:
//!
//!    ```ignore
//!    // Shard dim 0 along "data", replicate dim 1.
//!    // JAX equivalent: NamedSharding(mesh, PartitionSpec('data', None))
//!    let sharding = Sharding::new(
//!        mesh.logical_mesh.clone(),
//!        vec![
//!            ShardingDimension::sharded(["data"]),
//!            ShardingDimension::replicated(),
//!        ],
//!    )?;
//!
//!    // Shard dim 0 along both "data" and "model" axes.
//!    // JAX equivalent: NamedSharding(mesh, PartitionSpec(('data', 'model'),))
//!    let sharding = Sharding::new(
//!        mesh.logical_mesh.clone(),
//!        vec![ShardingDimension::sharded(["data", "model"])],
//!    )?;
//!
//!    // Fully replicated across all devices.
//!    // JAX equivalent: NamedSharding(mesh, PartitionSpec())
//!    let sharding = Sharding::replicated(mesh.logical_mesh.clone(), 2);
//!
//!    // Unconstrained dimension (let the propagator decide).
//!    // JAX equivalent: NamedSharding(mesh, PartitionSpec(UNCONSTRAINED))
//!    let sharding = Sharding::new(
//!        mesh.logical_mesh.clone(),
//!        vec![ShardingDimension::unconstrained()],
//!    )?;
//!    ```
//!
//! 3. **Use the sharding metadata at runtime** when pairing a logical array type with a concrete
//!    [`DeviceMesh`]. Runtime XLA arrays cache one [`Shard`] per mesh device so they can
//!    validate local PJRT buffers, identify addressable shards, and determine the per-device slices
//!    implied by the global sharding.
//!
//! 4. **Convert to Shardy MLIR attributes** for StableHLO program annotation:
//!
//!    ```ignore
//!    // Generates: sdy.mesh @mesh = <["data"=4, "model"=2]>
//!    let context = MlirContext::new();
//!    let mesh_module = context.module(context.unknown_location());
//!    let mesh_op = mesh.logical_mesh.to_shardy(context.unknown_location());
//!    let mesh_op = mesh_module.body().append_operation(mesh_op);
//!
//!    // Generates: #sdy.sharding<@mesh, [{"data"}, {}]>
//!    let attr = sharding.to_shardy(context.unknown_location());
//!    ```
//!
//! # Multi-host and addressability
//!
//! In multi-host (multi-process) execution, each host owns a disjoint subset of devices.
//! [`MeshDevice`] records both the global device ID and the owning process index, mirroring
//! JAX's distinction between [`jax.devices()`][jax-devices] (all global devices) and
//! [`jax.local_devices()`][jax-local-devices] (devices addressable by the current process).
//!
//! [jax-devices]: https://docs.jax.dev/en/latest/jax.html#jax.devices
//! [jax-local-devices]: https://docs.jax.dev/en/latest/jax.html#jax.local_devices
//!
//! Runtime shard-metadata computation covers *all* global shards - including those on remote
//! hosts - so that the full sharding picture is available for compilation and execution. Only
//! addressable shards can be backed by actual PJRT buffers; non-addressable shard descriptors
//! exist only as metadata describing remote device placements.
//!
//! This mirrors JAX's `array.addressable_shards` (local) vs `array.global_shards` (all),
//! where accessing `.data` on a non-addressable shard raises an error.

use std::collections::HashMap;
use std::ops::Range;

#[cfg(test)]
use ryft_mlir::Context as MlirContext;

use crate::sharding::{DeviceMesh, MeshDevice, MeshDeviceId, Sharding, ShardingDimension, ShardingError};

#[cfg(test)]
use crate::sharding::{LogicalMesh, MeshAxisType};

/// Information about a single shard of a distributed (i.e., sharded) array. Each shard corresponds to one device in a
/// [`DeviceMesh`] and describes the portion of the global array that that device holds. This is pure metadata; it does
/// not contain any actual array data.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shard {
    /// Global (ordinal) index of this [`Shard`] in a row-major device mesh ordering.
    pub index: usize,

    /// [`MeshDevice`] that owns this [`Shard`].
    pub device: MeshDevice,

    /// Row-major device mesh coordinates of the device that owns this [`Shard`].
    pub device_coordinates: Vec<usize>,

    /// Per-dimension ranges describing the portion of the corresponding global array that this [`Shard`] corresponds
    /// to. Each range describes which contiguous range of elements along a single array dimension this [`Shard`] owns.
    /// For a replicated dimension, the slice spans the full extent of that dimension, `[0, dimension_size)`. For a
    /// sharded dimension, the slice covers the partition assigned to a specific device based on its mesh coordinates.
    pub slice: Vec<Range<usize>>,
}

impl Shard {
    /// Logical shape of this shard, derived from the per-dimension [`slices`](Self::slices).
    #[inline]
    pub fn shape(&self) -> Vec<usize> {
        self.slice.iter().map(|slice| slice.len()).collect()
    }
}

// ---------------------------------------------------------------------------
// Sharding-metadata helpers
// ---------------------------------------------------------------------------

/// Computes one [`Shard`] per mesh device for the provided global shape and [`Sharding`].
pub(crate) fn compute_shard_descriptors(
    global_shape: &[usize],
    mesh: &DeviceMesh,
    sharding: &Sharding,
) -> Result<(Vec<Shard>, HashMap<MeshDeviceId, usize>), ShardingError> {
    if mesh.logical_mesh != sharding.mesh {
        return Err(ShardingError::MeshMismatch { expected: mesh.logical_mesh.clone(), actual: sharding.mesh.clone() });
    }

    let partition_rank = sharding.rank();
    let array_rank = global_shape.len();
    if partition_rank != array_rank {
        return Err(ShardingError::ShardingRankMismatch { sharding_rank: partition_rank, array_rank });
    }

    let mut shards = Vec::with_capacity(mesh.device_count());
    let mut shard_index_by_device = HashMap::with_capacity(mesh.device_count());
    for (index, mesh_device) in mesh.devices.iter().copied().enumerate() {
        let device_coordinates =
            mesh.device_coordinates(index).expect("mesh coordinate should exist for valid mesh device index");

        let mut slices = Vec::with_capacity(global_shape.len());
        for (dimension, dimension_size) in global_shape.iter().copied().enumerate() {
            let slice = match &sharding.dimensions[dimension] {
                ShardingDimension::Replicated => 0..dimension_size,
                ShardingDimension::Sharded(axis_names) => {
                    let mut partition_index = 0usize;
                    let mut partition_count = 1usize;
                    for axis_name in axis_names {
                        let axis_index = mesh
                            .logical_mesh
                            .axis_indices
                            .get(axis_name.as_str())
                            .copied()
                            .expect("sharding mesh axes should be validated before building shard slices");
                        let axis_size = mesh.logical_mesh.axes[axis_index].size;
                        let axis_coordinate = device_coordinates[axis_index];

                        partition_index = partition_index * axis_size + axis_coordinate;
                        partition_count *= axis_size;
                    }

                    let base_size = dimension_size / partition_count;
                    let remainder = dimension_size % partition_count;
                    let extra_before = partition_index.min(remainder);

                    let start = partition_index * base_size + extra_before;
                    let size = base_size + usize::from(partition_index < remainder);
                    start..start + size
                }
                ShardingDimension::Unconstrained => 0..dimension_size,
            };
            slices.push(slice);
        }

        shard_index_by_device.insert(mesh_device.id, index);
        shards.push(Shard { index, device: mesh_device, device_coordinates, slice: slices });
    }

    Ok((shards, shard_index_by_device))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::sharding::MeshAxis;

    use super::*;

    fn test_logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    fn test_device_mesh_2x2() -> DeviceMesh {
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 1), MeshDevice::new(3, 1)];
        DeviceMesh::new(test_logical_mesh_2x2(), devices).unwrap()
    }

    #[cfg(feature = "xla")]
    fn to_shardy_string(sharding: &Sharding) -> String {
        let context = MlirContext::new();
        sharding.to_shardy(context.unknown_location()).to_string()
    }

    // -----------------------------------------------------------------------
    // Sharding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sharding_validation() {
        let mesh = test_logical_mesh_2x2();

        assert!(matches!(
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["z"])]),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "z",
        ));

        assert!(matches!(
            Sharding::new(
                mesh.clone(),
                vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["x"])],
            ),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));

        assert!(matches!(
            Sharding::new(mesh, vec![ShardingDimension::Sharded(Vec::new())]),
            Err(ShardingError::EmptySharding { dimension }) if dimension == 0,
        ));
    }

    #[test]
    fn test_sharding_shardy_rendering() {
        let mesh = test_logical_mesh_2x2();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap();
        assert_eq!(to_shardy_string(&sharding), "#sdy.sharding<@mesh, [{\"x\"}, {}]>");
    }

    #[test]
    fn test_sharding_replicated_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap();

        assert_eq!(sharding.replicated_axes(), vec!["y"]);
        assert_eq!(to_shardy_string(&sharding), "#sdy.sharding<@mesh, [{\"x\"}, {}], replicated={\"y\"}>");
    }

    #[test]
    fn test_sharding_unreduced_axes() {
        let mesh = test_logical_mesh_2x2();
        let sharding = Sharding::with_unreduced_axes(
            mesh,
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            ["y"],
        )
        .unwrap();
        assert_eq!(to_shardy_string(&sharding), "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced={\"y\"}>");
    }

    #[test]
    fn test_sharding_replicated_and_unreduced_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::with_unreduced_axes(mesh, vec![ShardingDimension::sharded(["x"])], ["z"]).unwrap();

        assert_eq!(sharding.replicated_axes(), vec!["y"]);
        assert_eq!(
            to_shardy_string(&sharding),
            "#sdy.sharding<@mesh, [{\"x\"}], replicated={\"y\"}, unreduced={\"z\"}>"
        );
    }

    #[test]
    fn test_sharding_shardy_rendering_escapes_axis_names() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x\"y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new(r"path\to", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("z\"w", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::with_unreduced_axes(mesh, vec![ShardingDimension::sharded(["x\"y"])], ["z\"w"]).unwrap();

        assert_eq!(sharding.replicated_axes(), vec![r"path\to"]);
        assert_eq!(
            to_shardy_string(&sharding),
            r#"#sdy.sharding<@mesh, [{"x\22y"}], replicated={"path\\to"}, unreduced={"z\22w"}>"#
        );
    }

    #[test]
    fn test_sharding_unreduced_axis_validation() {
        let mesh = test_logical_mesh_2x2();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        assert!(matches!(
            Sharding::with_unreduced_axes(mesh.clone(), sharding.dimensions.clone(), ["z"]),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "z",
        ));

        assert!(matches!(
            Sharding::with_unreduced_axes(mesh, sharding.dimensions, ["x"]),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));
    }

    #[test]
    fn test_sharding_reduced_axis_validation() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();

        assert!(matches!(
            Sharding::with_manual_axes(
                mesh.clone(),
                vec![ShardingDimension::replicated()],
                Vec::<&str>::new(),
                ["y"],
                Vec::<&str>::new(),
            ),
            Err(ShardingError::ExpectedManualMeshAxis { name }) if name == "y",
        ));

        assert!(matches!(
            Sharding::with_manual_axes(
                mesh,
                vec![ShardingDimension::replicated()],
                ["z"],
                ["z"],
                Vec::<&str>::new(),
            ),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "z",
        ));
    }

    #[test]
    fn test_sharding_display() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("manual", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("varying", 8, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("carry", 8, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::with_manual_axes(
            mesh,
            vec![
                ShardingDimension::sharded(["data"]),
                ShardingDimension::replicated(),
                ShardingDimension::unconstrained(),
            ],
            ["carry"],
            ["manual"],
            ["varying"],
        )
        .unwrap();

        assert_eq!(
            sharding.to_string(),
            "{mesh<['data'=4, 'manual'=2, 'varying'=8, 'carry'=8]>, [{'data'}, {}, {?}], unreduced={'carry'}, \
            reduced_manual={'manual'}, varying_manual={'varying'}}",
        );
    }

    #[test]
    fn test_sharding_replicated_axes_ignore_reduced_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::with_manual_axes(
            mesh,
            vec![ShardingDimension::replicated()],
            Vec::<&str>::new(),
            ["x"],
            Vec::<&str>::new(),
        )
        .unwrap();

        assert_eq!(sharding.replicated_axes(), vec!["y"]);
    }

    fn shard_for_device<'a>(
        shards: &'a [Shard],
        shard_index_by_device: &HashMap<MeshDeviceId, usize>,
        device_id: MeshDeviceId,
    ) -> &'a Shard {
        let shard_index =
            shard_index_by_device.get(&device_id).copied().expect("device should have a shard descriptor");
        &shards[shard_index]
    }

    fn shard_indices_for_process(shards: &[Shard], process_index: usize) -> Vec<usize> {
        shards
            .iter()
            .filter_map(|descriptor| (descriptor.device.process_index == process_index).then_some(descriptor.index))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Sharding metadata tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shard_metadata_rank_mismatch() {
        let logical_mesh = test_logical_mesh_2x2();
        let mesh = test_device_mesh_2x2();
        let sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
                .unwrap();
        assert!(matches!(
            compute_shard_descriptors(&[8usize], &mesh, &sharding),
            Err(ShardingError::ShardingRankMismatch { sharding_rank: 2, array_rank: 1 }),
        ));
    }

    #[test]
    fn test_shard_metadata_unconstrained_is_ignored() {
        let logical_mesh = test_logical_mesh_2x2();
        let mesh = test_device_mesh_2x2();
        let sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()])
                .unwrap();
        let (shards, shard_index_by_device) = compute_shard_descriptors(&[8, 6], &mesh, &sharding).unwrap();

        let shard0 = shard_for_device(&shards, &shard_index_by_device, 0);
        let shard3 = shard_for_device(&shards, &shard_index_by_device, 3);
        assert_eq!(shard0.slice[0], 0..4);
        assert_eq!(shard0.slice[1], 0..6);
        assert_eq!(shard3.slice[0], 4..8);
        assert_eq!(shard3.slice[1], 0..6);
        assert_eq!(shard0.shape(), &[4, 6]);
        assert_eq!(shard3.shape(), &[4, 6]);
    }

    #[test]
    fn test_shard_metadata_even_2d_partitioning() {
        let logical_mesh = test_logical_mesh_2x2();
        let mesh = test_device_mesh_2x2();
        let sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
                .unwrap();
        let (shards, shard_index_by_device) = compute_shard_descriptors(&[8, 6], &mesh, &sharding).unwrap();

        let shard0 = shard_for_device(&shards, &shard_index_by_device, 0);
        assert_eq!(shard0.shape(), &[4, 3]);
        assert_eq!(shard0.slice[0], 0..4);
        assert_eq!(shard0.slice[1], 0..3);

        let shard3 = shard_for_device(&shards, &shard_index_by_device, 3);
        assert_eq!(shard3.shape(), &[4, 3]);
        assert_eq!(shard3.slice[0], 4..8);
        assert_eq!(shard3.slice[1], 3..6);
    }

    #[test]
    fn test_shard_metadata_uneven_partitioning() {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0)];
        let mesh = DeviceMesh::new(logical_mesh, devices).unwrap();
        let sharding = Sharding::new(mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (shards, shard_index_by_device) = compute_shard_descriptors(&[5], &mesh, &sharding).unwrap();

        let shard0 = shard_for_device(&shards, &shard_index_by_device, 0);
        assert_eq!(shard0.shape(), &[3]);
        assert_eq!(shard0.slice[0], 0..3);

        let shard1 = shard_for_device(&shards, &shard_index_by_device, 1);
        assert_eq!(shard1.shape(), &[2]);
        assert_eq!(shard1.slice[0], 3..5);
    }

    #[test]
    fn test_shard_metadata_multi_axis_single_dimension_partitioning() {
        let logical_mesh = test_logical_mesh_2x2();
        let mesh = test_device_mesh_2x2();
        let sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x".to_string(), "y".to_string()])]).unwrap();
        let (shards, shard_index_by_device) = compute_shard_descriptors(&[10], &mesh, &sharding).unwrap();

        assert_eq!(shard_for_device(&shards, &shard_index_by_device, 0).slice[0], 0..3);
        assert_eq!(shard_for_device(&shards, &shard_index_by_device, 1).slice[0], 3..6);
        assert_eq!(shard_for_device(&shards, &shard_index_by_device, 2).slice[0], 6..8);
        assert_eq!(shard_for_device(&shards, &shard_index_by_device, 3).slice[0], 8..10);
    }

    #[test]
    fn test_shard_metadata_process_filtering() {
        let logical_mesh = test_logical_mesh_2x2();
        let mesh = test_device_mesh_2x2();
        let sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
                .unwrap();
        let (shards, _) = compute_shard_descriptors(&[8, 6], &mesh, &sharding).unwrap();

        assert_eq!(shard_indices_for_process(&shards, 0), vec![0, 1]);
        assert_eq!(shard_indices_for_process(&shards, 1), vec![2, 3]);
        assert_eq!(shard_indices_for_process(&shards, 42), Vec::<usize>::new());
    }

    #[test]
    fn test_shard_metadata_mesh_mismatch_reports_expected_and_actual_meshes() {
        let logical_mesh = test_logical_mesh_2x2();
        let mesh = test_device_mesh_2x2();
        let actual = LogicalMesh::new(vec![MeshAxis::new("z", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharding = Sharding::new(actual.clone(), vec![ShardingDimension::sharded(["z"])]).unwrap();

        assert_eq!(
            compute_shard_descriptors(&[8], &mesh, &sharding),
            Err(ShardingError::MeshMismatch { expected: logical_mesh, actual })
        );
    }
}
