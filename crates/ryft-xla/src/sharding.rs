use ryft_core::{Device, DeviceMesh, LogicalMesh, MeshAxisType, Sharding, ShardingDimension, UnreducedKind};
use ryft_mlir::Location;
use ryft_mlir::dialects::shardy;
use ryft_pjrt::Device as PjrtDevice;
use ryft_pjrt::protos::{OpSharding, OpShardingType};

use crate::{FromPjrt, ToMlir};

impl FromPjrt<PjrtDevice<'_>> for Device {
    type Output = Result<Self, ryft_pjrt::Error>;

    #[inline]
    fn from_pjrt(value: PjrtDevice<'_>) -> Self::Output {
        Self::from_pjrt(&value)
    }
}

impl FromPjrt<&PjrtDevice<'_>> for Device {
    type Output = Result<Self, ryft_pjrt::Error>;

    #[inline]
    fn from_pjrt(value: &PjrtDevice<'_>) -> Self::Output {
        Ok(Self::new(value.id()?, value.process_index()?))
    }
}

/// Canonical symbol name used for emitted Shardy [`LogicalMesh`] declarations and references.
pub(crate) const SHARDY_MESH_SYMBOL_NAME: &str = "mesh";

impl ToMlir for LogicalMesh {
    type Output<'c, 't: 'c> = shardy::DetachedMeshOperation<'c, 't>;

    #[inline]
    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> Result<shardy::DetachedMeshOperation<'c, 't>, ryft_mlir::Error> {
        let context = location.context();
        let axes = self
            .axes()
            .iter()
            .map(|axis| context.shardy_mesh_axis(axis.name(), axis.size()))
            .collect::<Result<Vec<_>, _>>()?;
        let attribute = context.shardy_mesh(axes, &[])?;
        shardy::mesh(SHARDY_MESH_SYMBOL_NAME, attribute, location)
    }
}

impl ToMlir for DeviceMesh {
    type Output<'c, 't: 'c> = shardy::DetachedMeshOperation<'c, 't>;

    #[inline]
    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> Result<shardy::DetachedMeshOperation<'c, 't>, ryft_mlir::Error> {
        self.logical_mesh().to_mlir(location)
    }
}

impl ToMlir for Sharding {
    type Output<'c, 't: 'c> = shardy::TensorShardingAttributeRef<'c, 't>;

    #[inline]
    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> Result<shardy::TensorShardingAttributeRef<'c, 't>, ryft_mlir::Error> {
        ShardyTensorSharding::new(self, false).to_mlir(location)
    }
}

impl ToMlir for UnreducedKind {
    type Output<'c, 't: 'c> = shardy::ReductionOperation;

    #[inline]
    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        _location: L,
    ) -> Result<shardy::ReductionOperation, ryft_mlir::Error> {
        Ok(match self {
            Self::Sum => shardy::ReductionOperation::Sum,
            Self::Max => shardy::ReductionOperation::Max,
            Self::Min => shardy::ReductionOperation::Min,
        })
    }
}

// TODO(eaplatanios): Review this.
/// Recovers the [`Sharding`] over `mesh` of a rank-`rank` array that the partitioner placed as `op_sharding` describes
/// (e.g., an executable output whose placement over auto mesh axes the compiler chose), or returns [`None`] when no
/// such sharding exists. The device identifiers of `op_sharding` are the logical identifiers of an executable compiled
/// for `mesh`, whose `i`-th device is the `i`-th device of `mesh` in row-major mesh order. A tiled dimension is sharded
/// over the mesh axes whose coordinates, in major-to-minor order, determine the tile index of every device along that
/// dimension, and the mesh axes that tile no dimension are replicated. This mirrors how JAX recovers a `NamedSharding`
/// from the `HloSharding` of a compiled executable (`parse_flatten_op_sharding`).
pub(crate) fn sharding_from_op_sharding(op_sharding: &OpSharding, mesh: &LogicalMesh, rank: usize) -> Option<Sharding> {
    /// Extends `prefix` to an ordered list of unused mesh axes whose sizes multiply to `size` and whose coordinates,
    /// read from major to minor, give the tile index of every device along `dimension`, and returns whether such a
    /// list exists. The axes of a matching list stay marked in `used`, so that no other dimension can use them. This
    /// is a backtracking search over the ordered subsets of the unused axes, which is cheap because meshes have few
    /// axes.
    fn match_axes(
        dimension: usize,
        size: usize,
        prefix: &mut Vec<usize>,
        used: &mut [bool],
        axis_sizes: &[usize],
        tiles: &[(Vec<usize>, Vec<usize>)],
    ) -> bool {
        // Once the axes of the prefix span all the tiles of the dimension, the prefix matches if the index that the
        // mesh coordinates of every device form along those axes is the tile index of that device.
        let product = prefix.iter().map(|axis| axis_sizes[*axis]).product::<usize>();
        if product == size {
            return tiles.iter().all(|(tile, coordinates)| {
                prefix.iter().fold(0, |index, axis| index * axis_sizes[*axis] + coordinates[*axis]) == tile[dimension]
            });
        }

        // Try every axis that can still complete the prefix. An axis of size one cannot change the tile index, and
        // an axis whose size does not divide the remaining tile count would overshoot it.
        for axis in 0..axis_sizes.len() {
            if used[axis] || axis_sizes[axis] == 1 || size % (product * axis_sizes[axis]) != 0 {
                continue;
            }
            used[axis] = true;
            prefix.push(axis);
            if match_axes(dimension, size, prefix, used, axis_sizes, tiles) {
                return true;
            }
            prefix.pop();
            used[axis] = false;
        }
        false
    }

    // Placements without tiles are either replicated or not representable. A maximal placement puts the whole array
    // on one device, which is a replicated placement only over a mesh with a single device, and a tuple with a single
    // element describes the placement of that element. Only tiled placements (i.e., `Other`) need the steps below.
    let replicated = || Some(Sharding::replicated(mesh.clone(), rank));
    let device_count = mesh.axes().iter().map(|axis| axis.size()).product::<usize>();
    match OpShardingType::try_from(op_sharding.r#type).ok()? {
        OpShardingType::Replicated => return replicated(),
        OpShardingType::Maximal if device_count == 1 => return replicated(),
        OpShardingType::Tuple if op_sharding.tuple_shardings.len() == 1 => {
            return sharding_from_op_sharding(&op_sharding.tuple_shardings[0], mesh, rank);
        }
        OpShardingType::Other => {}
        _ => return None,
    }

    // The tile assignment is an array of device identifiers with one dimension for each array dimension, followed
    // by the subgroup dimensions, and the devices that differ only in their subgroup coordinates hold the same tile.
    // Only replicated subgroups (i.e., trailing tile dimensions over which devices hold copies) are representable.
    let replicated_subgroup = op_sharding
        .last_tile_dims
        .iter()
        .all(|kind| OpShardingType::try_from(*kind) == Ok(OpShardingType::Replicated));
    if !replicated_subgroup {
        return None;
    }

    // The tile assignment must have one dimension for each array dimension and each subgroup, and it must hold every
    // device of the mesh.
    let tile_dimensions = op_sharding
        .tile_assignment_dimensions
        .iter()
        .map(|size| usize::try_from(*size).ok())
        .collect::<Option<Vec<_>>>()?;
    let subgroup_count = op_sharding.last_tile_dims.len() + usize::from(op_sharding.replicate_on_last_tile_dim);
    if tile_dimensions.len() != rank + subgroup_count || tile_dimensions.iter().product::<usize>() != device_count {
        return None;
    }

    // The device identifiers of the tile assignment are either listed in its row-major order or given in iota form.
    let devices = if op_sharding.iota_reshape_dims.is_empty() {
        op_sharding
            .tile_assignment_devices
            .iter()
            .map(|device| usize::try_from(*device).ok())
            .collect::<Option<Vec<_>>>()?
    } else {
        // The iota form stands for the identifiers `0..n` reshaped to `iota_reshape_dims`, transposed by
        // `iota_transpose_perm` (i.e., by the identity permutation when it is empty), and flattened in row-major
        // order.
        let sizes = op_sharding
            .iota_reshape_dims
            .iter()
            .map(|size| usize::try_from(*size).ok())
            .collect::<Option<Vec<_>>>()?;
        let permutation = if op_sharding.iota_transpose_perm.is_empty() {
            (0..sizes.len()).collect::<Vec<_>>()
        } else {
            op_sharding
                .iota_transpose_perm
                .iter()
                .map(|axis| usize::try_from(*axis).ok())
                .collect::<Option<Vec<_>>>()?
        };

        if permutation.len() != sizes.len() {
            return None;
        }

        // Dimension `index` of the transposed array is dimension `permutation[index]` of the reshaped array. The
        // identifier at a position of the transposed array is therefore the row-major index of the same position in
        // the reshaped array, which the row-major strides of the reshaped array compute.
        let strides = sizes.iter().rev().scan(1, |stride, size| {
            let current = *stride;
            *stride *= size;
            Some(current)
        });
        let mut strides = strides.collect::<Vec<_>>();
        strides.reverse();
        let transposed_sizes = permutation.iter().map(|axis| sizes.get(*axis).copied()).collect::<Option<Vec<_>>>()?;
        let count = sizes.iter().product::<usize>();
        let mut devices = Vec::with_capacity(count);
        for position in 0..count {
            let mut remainder = position;
            let mut device = 0;
            for (index, size) in transposed_sizes.iter().enumerate().rev() {
                device += (remainder % size) * strides[permutation[index]];
                remainder /= size;
            }
            devices.push(device);
        }
        devices
    };

    if devices.len() != device_count || devices.iter().any(|device| *device >= device_count) {
        return None;
    }

    // Pair the tile coordinates of every position of the tile assignment with the mesh coordinates of the device at
    // that position. Both are row-major coordinates, because device `i` is the `i`-th device of the mesh in row-major
    // order.
    let axis_sizes = mesh.axes().iter().map(|axis| axis.size()).collect::<Vec<_>>();
    let unravel = |mut index: usize, sizes: &[usize]| {
        let mut coordinates = vec![0; sizes.len()];
        for (coordinate, size) in coordinates.iter_mut().zip(sizes).rev() {
            *coordinate = index % size;
            index /= size;
        }
        coordinates
    };
    let tiles = devices
        .iter()
        .enumerate()
        .map(|(position, device)| (unravel(position, &tile_dimensions), unravel(*device, &axis_sizes)))
        .collect::<Vec<_>>();

    // A dimension with a single tile is replicated, and every other dimension is sharded over the ordered mesh axes
    // that determine its tile index. The mesh axes that no dimension uses, including the ones that the subgroup
    // dimensions span, are the replicated axes of the resulting sharding.
    let mut used = vec![false; axis_sizes.len()];
    let mut dimensions = Vec::with_capacity(rank);
    for (dimension, size) in tile_dimensions[..rank].iter().copied().enumerate() {
        if size == 1 {
            dimensions.push(ShardingDimension::Replicated);
            continue;
        }
        let mut axes = Vec::new();
        if !match_axes(dimension, size, &mut axes, &mut used, &axis_sizes, &tiles) {
            return None;
        }
        dimensions
            .push(ShardingDimension::Sharded(axes.iter().map(|axis| mesh.axes()[*axis].name().to_string()).collect()));
    }

    Sharding::new(mesh.clone(), dimensions).ok()
}

/// Lowering of a [`Sharding`] to its Shardy tensor sharding attribute. The replicated and sharded dimensions of the
/// attribute are closed (i.e., Shardy cannot shard them along further axes) unless the lowering is open over auto axes
/// and the mesh of the sharding has an [`Auto`](MeshAxisType::Auto) axis, in which case every dimension is open. The
/// axes that the sharding governs but does not use are listed as replicated either way, so open dimensions can only
/// gain auto axes. This is how JAX lowers the shardings of types over meshes with auto axes (i.e.,
/// `NamedSharding._to_sdy_sharding` with `modify_wrt_axis_types=True`), whose placement along those
/// axes the type does not describe and the compiler chooses.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ShardyTensorSharding<'s> {
    /// [`Sharding`] that is being lowered.
    sharding: &'s Sharding,

    /// Whether the dimensions of the lowered attribute are open over the auto axes of the mesh of the sharding.
    open_over_auto_axes: bool,
}

impl<'s> ShardyTensorSharding<'s> {
    /// Creates a new [`ShardyTensorSharding`] that lowers `sharding`, with dimensions that are open over the auto
    /// axes of its mesh if `open_over_auto_axes` is set.
    #[inline]
    pub(crate) fn new(sharding: &'s Sharding, open_over_auto_axes: bool) -> Self {
        Self { sharding, open_over_auto_axes }
    }
}

impl ToMlir for ShardyTensorSharding<'_> {
    type Output<'c, 't: 'c> = shardy::TensorShardingAttributeRef<'c, 't>;

    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> Result<shardy::TensorShardingAttributeRef<'c, 't>, ryft_mlir::Error> {
        let Self { sharding, open_over_auto_axes } = *self;
        let context = location.context();
        let closed =
            !open_over_auto_axes || sharding.mesh().axes().iter().all(|axis| axis.r#type() != MeshAxisType::Auto);
        let mesh_symbol_ref = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
        let dimensions = sharding
            .dimensions()
            .iter()
            .map(|dimension| match dimension {
                ShardingDimension::Replicated => context.shardy_dimension_sharding([], closed, None),
                ShardingDimension::Sharded(axis_names) => context.shardy_dimension_sharding(
                    axis_names
                        .iter()
                        .map(|axis_name| context.shardy_axis_ref(axis_name.as_str(), None))
                        .collect::<Result<Vec<_>, _>>()?,
                    closed,
                    None,
                ),
                ShardingDimension::Unconstrained => context.shardy_dimension_sharding([], false, None),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let replicated_axes = sharding
            .replicated_axes()
            .iter()
            .map(|axis_name| context.shardy_axis_ref(*axis_name, None))
            .collect::<Result<Vec<_>, _>>()?;
        let unreduced_axes = sharding
            .unreduced_axes()
            .iter()
            .map(|axis_name| context.shardy_axis_ref(axis_name.as_str(), None))
            .collect::<Result<Vec<_>, _>>()?;
        context.shardy_tensor_sharding(
            mesh_symbol_ref,
            dimensions.as_slice(),
            replicated_axes.as_slice(),
            unreduced_axes.as_slice(),
            sharding.unreduced_kind().to_mlir(location)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::MeshAxis;
    use ryft_mlir::{Block, Context as MlirContext};

    use crate::tests::{device_mesh_2x2, execution_client, logical_mesh_2x2, logical_mesh_3x2x1};

    use super::*;

    /// Returns a tiled [`OpSharding`] whose tile assignment has the provided `dimensions` and lists `devices` in
    /// row-major order. Its last tile dimension is a replicated subgroup if `replicate_on_last_tile_dim` is set.
    fn tiled_op_sharding(dimensions: Vec<i64>, devices: Vec<i64>, replicate_on_last_tile_dim: bool) -> OpSharding {
        OpSharding {
            r#type: OpShardingType::Other as i32,
            tile_assignment_dimensions: dimensions,
            tile_assignment_devices: devices,
            replicate_on_last_tile_dim,
            ..Default::default()
        }
    }

    #[test]
    fn test_device_from_pjrt() {
        let client = execution_client();
        let devices = client.addressable_devices().unwrap();
        let expected = Device::new(devices[0].id().unwrap(), devices[0].process_index().unwrap());
        assert_eq!(Device::from_pjrt(&devices[0]), Ok(expected));
        assert_eq!(Device::from_pjrt(devices[0].clone()), Ok(expected));
    }

    #[test]
    fn test_logical_mesh_to_mlir() {
        let mesh = logical_mesh_3x2x1();
        let context = MlirContext::new();
        let module = context.module(context.unknown_location()).unwrap();
        assert_eq!(
            module
                .body()
                .unwrap()
                .append_operation(mesh.to_mlir(context.unknown_location()).unwrap())
                .unwrap()
                .to_string(),
            format!("sdy.mesh @{SHARDY_MESH_SYMBOL_NAME} = <[\"x\"=3, \"y\"=2, \"z\"=1]>"),
        );
    }

    #[test]
    fn test_device_mesh_to_mlir() {
        let mesh = device_mesh_2x2();
        let context = MlirContext::new();
        let module = context.module(context.unknown_location()).unwrap();
        assert_eq!(
            module
                .body()
                .unwrap()
                .append_operation(mesh.to_mlir(context.unknown_location()).unwrap())
                .unwrap()
                .to_string(),
            format!("sdy.mesh @{SHARDY_MESH_SYMBOL_NAME} = <[\"x\"=2, \"y\"=2]>"),
        );
    }

    #[test]
    fn test_sharding_to_mlir() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 6, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["y"])
            .unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();
        assert_eq!(
            sharding.to_mlir(location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced={\"y\"}>",
        );

        // Kinds other than a sum lower to the reduction operation of the unreduced axes.
        let sharding = sharding.with_unreduced_kind(UnreducedKind::Max).unwrap();
        assert_eq!(
            sharding.to_mlir(location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced=max{\"y\"}>",
        );
    }

    #[test]
    fn test_unreduced_kind_to_mlir() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        assert_eq!(UnreducedKind::Sum.to_mlir(location), Ok(shardy::ReductionOperation::Sum));
        assert_eq!(UnreducedKind::Max.to_mlir(location), Ok(shardy::ReductionOperation::Max));
        assert_eq!(UnreducedKind::Min.to_mlir(location), Ok(shardy::ReductionOperation::Min));
    }

    #[test]
    fn test_sharding_from_op_sharding() {
        let mesh = logical_mesh_2x2();
        let replicated = Some(Sharding::replicated(mesh.clone(), 2));
        let dimensions = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
        let along_x = Some(Sharding::new(mesh.clone(), dimensions).unwrap());
        let dimensions = vec![ShardingDimension::sharded(["y"]), ShardingDimension::replicated()];
        let along_y = Some(Sharding::new(mesh.clone(), dimensions).unwrap());

        // A replicated placement has no tiles, a maximal placement is replicated over a mesh with a single device,
        // and a tuple with a single element has the placement of that element.
        let op_sharding = OpSharding { r#type: OpShardingType::Replicated as i32, ..Default::default() };
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 2), replicated);
        let single_device_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let op_sharding = OpSharding { r#type: OpShardingType::Maximal as i32, ..Default::default() };
        assert_eq!(
            sharding_from_op_sharding(&op_sharding, &single_device_mesh, 2),
            Some(Sharding::replicated(single_device_mesh.clone(), 2)),
        );
        let op_sharding = OpSharding {
            r#type: OpShardingType::Tuple as i32,
            tuple_shardings: vec![tiled_op_sharding(vec![2, 1, 2], vec![0, 1, 2, 3], true)],
            ..Default::default()
        };
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 2), along_x);

        // Tiling the first dimension along one axis replicates over the other axis on the last tile dimension. Device
        // `i` has the mesh coordinates `(i / 2, i % 2)`.
        let op_sharding = tiled_op_sharding(vec![2, 1, 2], vec![0, 1, 2, 3], true);
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 2), along_x);
        let op_sharding = tiled_op_sharding(vec![2, 1, 2], vec![0, 2, 1, 3], true);
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 2), along_y);

        // The iota form lists the same devices, in the order of the identity permutation unless it is transposed.
        let op_sharding =
            OpSharding { iota_reshape_dims: vec![2, 2], ..tiled_op_sharding(vec![2, 1, 2], vec![], true) };
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 2), along_x);
        let op_sharding = OpSharding { iota_transpose_perm: vec![1, 0], ..op_sharding };
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 2), along_y);

        // A dimension that is tiled over both axes lists them in major-to-minor order.
        let op_sharding = tiled_op_sharding(vec![4], vec![0, 1, 2, 3], false);
        assert_eq!(
            sharding_from_op_sharding(&op_sharding, &mesh, 1),
            Some(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap()),
        );
        let op_sharding = tiled_op_sharding(vec![4], vec![0, 2, 1, 3], false);
        assert_eq!(
            sharding_from_op_sharding(&op_sharding, &mesh, 1),
            Some(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y", "x"])]).unwrap()),
        );
    }

    #[test]
    fn test_sharding_from_op_sharding_with_unrepresentable_placements() {
        let mesh = logical_mesh_2x2();

        // Placements without tiles that are not replicated have no sharding.
        let op_sharding = OpSharding { r#type: OpShardingType::Maximal as i32, ..Default::default() };
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 2), None);
        let op_sharding = OpSharding { r#type: OpShardingType::Manual as i32, ..Default::default() };
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 2), None);
        let op_sharding = OpSharding { r#type: i32::MAX, ..Default::default() };
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 2), None);
        let element = tiled_op_sharding(vec![2, 1, 2], vec![0, 1, 2, 3], true);
        let op_sharding = OpSharding {
            r#type: OpShardingType::Tuple as i32,
            tuple_shardings: vec![element.clone(), element],
            ..Default::default()
        };
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 2), None);

        // Only replicated subgroups are representable.
        let op_sharding = OpSharding {
            last_tile_dims: vec![OpShardingType::Manual as i32],
            ..tiled_op_sharding(vec![2, 2], vec![0, 1, 2, 3], false)
        };
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 1), None);

        // The tile assignment must have one dimension for each array dimension and subgroup, and it must hold every
        // device of the mesh exactly as the mesh identifies it.
        let op_sharding = tiled_op_sharding(vec![2, 2], vec![0, 1, 2, 3], false);
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 1), None);
        let op_sharding = tiled_op_sharding(vec![2], vec![0, 1], false);
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 1), None);
        let op_sharding = tiled_op_sharding(vec![4], vec![0, 1, 2, 4], false);
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 1), None);
        let op_sharding = tiled_op_sharding(vec![4], vec![0, 1, 2, -1], false);
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 1), None);

        // An iota form whose permutation does not have one entry for each reshaped dimension is malformed.
        let op_sharding = OpSharding {
            iota_reshape_dims: vec![2, 2],
            iota_transpose_perm: vec![0],
            ..tiled_op_sharding(vec![2, 1, 2], vec![], true)
        };
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 2), None);

        // An assignment that no order of the mesh axes explains has no sharding.
        let op_sharding = tiled_op_sharding(vec![4], vec![0, 3, 1, 2], false);
        assert_eq!(sharding_from_op_sharding(&op_sharding, &mesh, 1), None);
    }

    #[test]
    fn test_shardy_tensor_sharding_to_mlir() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])]).unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();

        // Over a mesh with an auto axis, an open lowering leaves every dimension open, so the compiler can only add
        // auto axes, because the unused explicit axes stay replicated. A closed lowering closes every dimension.
        assert_eq!(
            ShardyTensorSharding::new(&sharding, true).to_mlir(location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{?}, {\"y\", ?}], replicated={\"z\"}>",
        );
        assert_eq!(
            ShardyTensorSharding::new(&sharding, false).to_mlir(location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{}, {\"y\"}], replicated={\"z\"}>",
        );

        // An open lowering over a mesh without auto axes closes every dimension too.
        let explicit_mesh = sharding.mesh().with_axis_types(["x"], MeshAxisType::Explicit).unwrap();
        let explicit_sharding = Sharding::new(explicit_mesh.clone(), sharding.dimensions().to_vec()).unwrap();
        assert_eq!(
            ShardyTensorSharding::new(&explicit_sharding, true).to_mlir(location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{}, {\"y\"}], replicated={\"x\", \"z\"}>",
        );

        // An unconstrained dimension stays open in a closed lowering.
        let unconstrained_sharding =
            Sharding::new(explicit_mesh, vec![ShardingDimension::unconstrained(), ShardingDimension::sharded(["y"])])
                .unwrap();
        assert_eq!(
            ShardyTensorSharding::new(&unconstrained_sharding, false).to_mlir(location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{?}, {\"y\"}], replicated={\"x\", \"z\"}>",
        );
    }
}
