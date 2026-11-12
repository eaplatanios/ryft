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
        shardy_tensor_sharding_attribute(self, location, false)
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

/// Returns the Shardy attribute of `sharding`. Its replicated and sharded dimensions are closed (i.e., Shardy
/// cannot shard them along further axes) unless `open_over_auto_axes` is set and the mesh of `sharding` has an
/// [`Auto`](MeshAxisType::Auto) axis, in which case every dimension is open. The axes that `sharding` governs but does
/// not use are listed as replicated either way (refer to [`Sharding::replicated_axes`]), so open dimensions can only
/// gain auto axes. This is how JAX lowers the shardings of types over meshes with auto axes (i.e.,
/// `NamedSharding._to_sdy_sharding` with `modify_wrt_axis_types=True`), whose placement along
/// those axes the type does not describe and the compiler chooses.
pub(crate) fn shardy_tensor_sharding_attribute<'c, 't: 'c, L: Location<'c, 't>>(
    sharding: &Sharding,
    location: L,
    open_over_auto_axes: bool,
) -> Result<shardy::TensorShardingAttributeRef<'c, 't>, ryft_mlir::Error> {
    let context = location.context();
    let closed = !open_over_auto_axes || sharding.mesh().axes().iter().all(|axis| axis.r#type() != MeshAxisType::Auto);
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

// TODO(eaplatanios): Review this.
/// Recovers the [`Sharding`] over `mesh` of a rank-`rank` array that the partitioner placed as `op_sharding` describes
/// (e.g., an executable output whose placement over auto mesh axes the compiler chose), or returns [`None`] when no
/// such sharding exists. The device identifiers of `op_sharding` are the logical identifiers of an executable compiled
/// for `mesh`, whose `i`-th device is the `i`-th device of `mesh` in row-major mesh order. A tiled dimension is sharded
/// over the mesh axes whose coordinates, in major-to-minor order, determine the tile index of every device along that
/// dimension, and the mesh axes that tile no dimension are replicated. This mirrors how JAX recovers a `NamedSharding`
/// from the `HloSharding` of a compiled executable (`parse_flatten_op_sharding`).
pub(crate) fn sharding_from_op_sharding(op_sharding: &OpSharding, mesh: &LogicalMesh, rank: usize) -> Option<Sharding> {
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

    // Only replicated subgroups (i.e., trailing tile dimensions over which devices hold copies) are representable.
    let replicated_subgroup = op_sharding
        .last_tile_dims
        .iter()
        .all(|kind| OpShardingType::try_from(*kind) == Ok(OpShardingType::Replicated));
    if !replicated_subgroup {
        return None;
    }
    let tile_dimensions = op_sharding
        .tile_assignment_dimensions
        .iter()
        .map(|size| usize::try_from(*size).ok())
        .collect::<Option<Vec<_>>>()?;
    let subgroup_count = op_sharding.last_tile_dims.len() + usize::from(op_sharding.replicate_on_last_tile_dim);
    if tile_dimensions.len() != rank + subgroup_count || tile_dimensions.iter().product::<usize>() != device_count {
        return None;
    }
    let devices = if op_sharding.iota_reshape_dims.is_empty() {
        op_sharding
            .tile_assignment_devices
            .iter()
            .map(|device| usize::try_from(*device).ok())
            .collect::<Option<Vec<_>>>()?
    } else {
        iota_tile_assignment(&op_sharding.iota_reshape_dims, &op_sharding.iota_transpose_perm)?
    };
    if devices.len() != device_count || devices.iter().any(|device| *device >= device_count) {
        return None;
    }

    // Row-major coordinates of every device along every mesh axis and along every tile dimension.
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

    // Each tiled dimension is matched to the ordered mesh axes that determine its tile index, by a search over the
    // ordered subsets of the axes that no other dimension uses (meshes have few axes, so this search is cheap).
    fn match_axes(
        dimension: usize,
        size: usize,
        prefix: &mut Vec<usize>,
        used: &mut [bool],
        axis_sizes: &[usize],
        tiles: &[(Vec<usize>, Vec<usize>)],
    ) -> bool {
        let product = prefix.iter().map(|axis| axis_sizes[*axis]).product::<usize>();
        if product == size {
            return tiles.iter().all(|(tile, coordinates)| {
                prefix.iter().fold(0, |index, axis| index * axis_sizes[*axis] + coordinates[*axis]) == tile[dimension]
            });
        }
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

// TODO(eaplatanios): Review this.
/// Returns the device list of an iota tile assignment: `0..n` reshaped to `reshape_dimensions`, transposed by
/// `transpose_permutation`, and flattened in row-major order.
fn iota_tile_assignment(reshape_dimensions: &[i64], transpose_permutation: &[i32]) -> Option<Vec<usize>> {
    let sizes = reshape_dimensions.iter().map(|size| usize::try_from(*size).ok()).collect::<Option<Vec<_>>>()?;
    let permutation = if transpose_permutation.is_empty() {
        (0..sizes.len()).collect::<Vec<_>>()
    } else {
        transpose_permutation.iter().map(|axis| usize::try_from(*axis).ok()).collect::<Option<Vec<_>>>()?
    };
    if permutation.len() != sizes.len() {
        return None;
    }
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
    Some(devices)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{LogicalMesh, MeshAxis, MeshAxisType, UnreducedKind};
    use ryft_mlir::{Block, Context as MlirContext};

    use crate::tests::{device_mesh_2x2, logical_mesh_3x2x1};

    use super::*;

    #[test]
    fn test_logical_mesh_to_shardy() {
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
    fn test_device_mesh_to_shardy() {
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
    fn test_sharding_to_shardy() {
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
        assert_eq!(
            sharding.to_mlir(context.unknown_location()).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced={\"y\"}>",
        );

        // Kinds other than a sum lower to the reduction operation of the unreduced axes.
        let attribute = sharding.clone().with_unreduced_kind(UnreducedKind::Max).unwrap();
        let attribute = attribute.to_mlir(context.unknown_location()).unwrap();
        assert_eq!(attribute.reduction_operation().unwrap(), shardy::ReductionOperation::Max);
        assert_eq!(attribute.to_string(), "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced=max{\"y\"}>");
        let attribute = sharding.with_unreduced_kind(UnreducedKind::Min).unwrap();
        let attribute = attribute.to_mlir(context.unknown_location()).unwrap();
        assert_eq!(attribute.reduction_operation().unwrap(), shardy::ReductionOperation::Min);
    }

    #[test]
    fn test_unreduced_kind_to_shardy() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        assert_eq!(UnreducedKind::Sum.to_mlir(location), Ok(shardy::ReductionOperation::Sum));
        assert_eq!(UnreducedKind::Max.to_mlir(location), Ok(shardy::ReductionOperation::Max));
        assert_eq!(UnreducedKind::Min.to_mlir(location), Ok(shardy::ReductionOperation::Min));
    }

    #[test]
    fn test_shardy_tensor_sharding() {
        // Over a mesh with an auto axis, an open lowering leaves every dimension open, so the compiler can only add
        // auto axes, since the unused explicit axes stay replicated. A closed lowering, and an open lowering over a
        // mesh without auto axes, close every dimension.
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
        assert_eq!(
            shardy_tensor_sharding_attribute(&sharding, location, true).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{?}, {\"y\", ?}], replicated={\"z\"}>",
        );
        assert_eq!(
            shardy_tensor_sharding_attribute(&sharding, location, false).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{}, {\"y\"}], replicated={\"z\"}>",
        );
        let explicit_sharding = Sharding::new(
            sharding.mesh().with_axis_types(["x"], MeshAxisType::Explicit).unwrap(),
            sharding.dimensions().to_vec(),
        )
        .unwrap();
        assert_eq!(
            shardy_tensor_sharding_attribute(&explicit_sharding, location, true).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{}, {\"y\"}], replicated={\"x\", \"z\"}>",
        );
    }

    #[test]
    fn test_sharding_from_op_sharding() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let tiled = |dimensions: Vec<i64>, devices: Vec<i64>, replicate_on_last_tile_dim: bool| OpSharding {
            r#type: OpShardingType::Other as i32,
            tile_assignment_dimensions: dimensions,
            tile_assignment_devices: devices,
            replicate_on_last_tile_dim,
            ..Default::default()
        };
        let sharding = |dimensions: Vec<ShardingDimension>| Some(Sharding::new(mesh.clone(), dimensions).unwrap());
        let replicated = OpSharding { r#type: OpShardingType::Replicated as i32, ..Default::default() };
        assert_eq!(sharding_from_op_sharding(&replicated, &mesh, 2), Some(Sharding::replicated(mesh.clone(), 2)));

        // Tiling the first dimension along `x` (or `y`) replicates over the other axis on the last tile dimension, in
        // either listed or iota form. Device `i` has mesh coordinates `(i / 2, i % 2)`.
        let along_x = sharding(vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]);
        let along_y = sharding(vec![ShardingDimension::sharded(["y"]), ShardingDimension::Replicated]);
        assert_eq!(sharding_from_op_sharding(&tiled(vec![2, 1, 2], vec![0, 1, 2, 3], true), &mesh, 2), along_x);
        assert_eq!(sharding_from_op_sharding(&tiled(vec![2, 1, 2], vec![0, 2, 1, 3], true), &mesh, 2), along_y);
        let iota = OpSharding {
            iota_reshape_dims: vec![2, 2],
            iota_transpose_perm: vec![1, 0],
            ..tiled(vec![2, 1, 2], vec![], true)
        };
        assert_eq!(sharding_from_op_sharding(&iota, &mesh, 2), along_y);

        // A dimension tiled over both axes lists them in major-to-minor order, and an assignment that no axis order
        // explains has no sharding.
        assert_eq!(
            sharding_from_op_sharding(&tiled(vec![4], vec![0, 1, 2, 3], false), &mesh, 1),
            sharding(vec![ShardingDimension::sharded(["x", "y"])]),
        );
        assert_eq!(
            sharding_from_op_sharding(&tiled(vec![4], vec![0, 2, 1, 3], false), &mesh, 1),
            sharding(vec![ShardingDimension::sharded(["y", "x"])]),
        );
        assert_eq!(sharding_from_op_sharding(&tiled(vec![4], vec![0, 3, 1, 2], false), &mesh, 1), None);
        assert_eq!(sharding_from_op_sharding(&tiled(vec![2, 2], vec![0, 1, 2, 3], false), &mesh, 1), None);
    }
}
