//! Conversions between `ryft` sharding types and their XLA representations. This module lowers [`LogicalMesh`],
//! [`DeviceMesh`], [`Sharding`], and [`UnreducedKind`] to Shardy MLIR through [`ToMlir`], with the crate-internal
//! `ShardySharding` controlling which dimensions of a lowered tensor sharding the compiler may still shard along auto
//! mesh axes. In the opposite direction, [`OpShardingExtension`] recovers the [`Sharding`] that an XLA
//! [`OpSharding`] describes, e.g., for executable outputs whose placement the compiler chose.

use ryft_core::{
    DeviceMesh, LogicalMesh, MeshAxisType, ProgramError, Sharding, ShardingDimension, ShardingError, UnreducedKind,
};
use ryft_mlir::dialects::shardy;
use ryft_mlir::{Block, BlockRef, Location, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef};
use ryft_pjrt::protos::{OpSharding, OpShardingType};

use crate::{Error, ToMlir};

/// Canonical symbol name used for emitted Shardy [`LogicalMesh`] declarations and references.
pub const SHARDY_MESH_SYMBOL_NAME: &str = "mesh";

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
        ShardySharding::new(self, OpenDimensions::None).to_mlir(location)
    }
}

impl ToMlir for UnreducedKind {
    type Output<'c, 't: 'c> = shardy::ReductionOperation;

    #[inline]
    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> Result<shardy::ReductionOperation, ryft_mlir::Error> {
        let _ = location;
        Ok(match self {
            Self::Sum => shardy::ReductionOperation::Sum,
            Self::Max => shardy::ReductionOperation::Max,
            Self::Min => shardy::ReductionOperation::Min,
        })
    }
}

/// Extension of [`OpSharding`], which is how XLA describes the placement of a value over the devices of an executable,
/// with conversions to the corresponding `ryft` types. [`OpSharding`] is a Protobuf message that `ryft` does not own,
/// so its `ryft` specific functions are provided by this trait.
pub trait OpShardingExtension {
    /// Recovers the [`Sharding`] over `mesh` of a rank-`rank` array that the partitioner placed as this [`OpSharding`]
    /// describes (e.g., an executable output whose placement over auto mesh axes the compiler chose), or returns
    /// [`None`] when no such sharding exists. The device identifiers of this [`OpSharding`] are the logical identifiers
    /// of an executable compiled for `mesh`, whose `i`-th device is the `i`-th device of `mesh` in row-major mesh
    /// order. A tiled dimension is sharded over the mesh axes whose coordinates, in major-to-minor order, determine
    /// the tile index of every device along that dimension, and the mesh axes that tile no dimension are replicated.
    fn to_sharding(&self, mesh: &LogicalMesh, rank: usize) -> Option<Sharding>;
}

impl OpShardingExtension for OpSharding {
    fn to_sharding(&self, mesh: &LogicalMesh, rank: usize) -> Option<Sharding> {
        /// Extends `prefix` to an ordered list of unused mesh axes whose sizes multiply to `size` and whose
        /// coordinates, read from major to minor, give the tile index of every device along `dimension`, and returns
        /// whether such a list exists. The axes of a matching list stay marked in `used`, so that no other dimension
        /// can use them. This is a backtracking search over the ordered subsets of the unused axes, which is cheap
        /// because meshes have few axes.
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
                    let index = prefix.iter().fold(0, |index, axis| index * axis_sizes[*axis] + coordinates[*axis]);
                    index == tile[dimension]
                });
            }

            // Try every axis that can still complete the prefix. An axis of size one cannot change the tile index, and
            // an axis whose size does not divide the remaining tile count would overshoot it.
            for axis in 0..axis_sizes.len() {
                if used[axis] || axis_sizes[axis] == 1 || !size.is_multiple_of(product * axis_sizes[axis]) {
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
        // on one device, which is a replicated placement only over a mesh with a single device, and a tuple with a
        // single element describes the placement of that element. Only tiled placements (i.e., `Other`) need the steps
        // below.
        let replicated = || Some(Sharding::replicated(mesh.clone(), rank));
        let device_count = mesh.device_count();
        match OpShardingType::try_from(self.r#type).ok()? {
            OpShardingType::Replicated => return replicated(),
            OpShardingType::Maximal if device_count == 1 => return replicated(),
            OpShardingType::Tuple if self.tuple_shardings.len() == 1 => {
                return self.tuple_shardings[0].to_sharding(mesh, rank);
            }
            OpShardingType::Other => {}
            _ => return None,
        }

        // The tile assignment is an array of device identifiers with one dimension for each array dimension, followed
        // by the subgroup dimensions, and the devices that differ only in their subgroup coordinates hold the same
        // tile. Only replicated subgroups (i.e., trailing tile dimensions over which devices hold copies) are
        // representable.
        let replicated_subgroup = self
            .last_tile_dims
            .iter()
            .all(|kind| OpShardingType::try_from(*kind) == Ok(OpShardingType::Replicated));
        if !replicated_subgroup {
            return None;
        }

        // The tile assignment must have one dimension for each array dimension and each subgroup, and it must hold
        // every device of the mesh.
        let tile_dimensions = self
            .tile_assignment_dimensions
            .iter()
            .map(|size| usize::try_from(*size).ok())
            .collect::<Option<Vec<_>>>()?;
        let subgroup_count = self.last_tile_dims.len() + usize::from(self.replicate_on_last_tile_dim);
        if tile_dimensions.len().checked_sub(rank) != Some(subgroup_count)
            || tile_dimensions.iter().try_fold(1usize, |count, size| count.checked_mul(*size))? != device_count
        {
            return None;
        }

        // The device identifiers of the tile assignment are either listed in its row-major order or given in iota form.
        let devices = if self.iota_reshape_dims.is_empty() {
            if self.tile_assignment_devices.len() != device_count {
                return None;
            }

            self.tile_assignment_devices
                .iter()
                .map(|device| usize::try_from(*device).ok())
                .collect::<Option<Vec<_>>>()?
        } else {
            // The iota form stands for the identifiers `0..n` reshaped to `iota_reshape_dims`, transposed by
            // `iota_transpose_perm` (i.e., by the identity permutation when it is empty), and flattened in row-major
            // order.
            let sizes =
                self.iota_reshape_dims.iter().map(|size| usize::try_from(*size).ok()).collect::<Option<Vec<_>>>()?;
            if sizes.iter().try_fold(1usize, |count, size| count.checked_mul(*size))? != device_count {
                return None;
            }

            let permutation = if self.iota_transpose_perm.is_empty() {
                (0..sizes.len()).collect::<Vec<_>>()
            } else {
                self.iota_transpose_perm
                    .iter()
                    .map(|axis| usize::try_from(*axis).ok())
                    .collect::<Option<Vec<_>>>()?
            };

            if permutation.len() != sizes.len() {
                return None;
            }

            let mut transposed = vec![false; sizes.len()];
            for &axis in &permutation {
                if std::mem::replace(transposed.get_mut(axis)?, true) {
                    return None;
                }
            }

            // Dimension `index` of the transposed array is dimension `permutation[index]` of the reshaped array. The
            // identifier at a position of the transposed array is therefore the row-major index of the same position in
            // the reshaped array, which the row-major strides of the reshaped array compute. Each stride fits because
            // the dimensions are positive and their checked product fits.
            let strides = sizes.iter().rev().scan(1, |stride, size| {
                let current = *stride;
                *stride *= size;
                Some(current)
            });
            let mut strides = strides.collect::<Vec<_>>();
            strides.reverse();

            let mut devices = Vec::with_capacity(device_count);
            for position in 0..device_count {
                let mut remainder = position;
                let mut device = 0;
                for &axis in permutation.iter().rev() {
                    let size = sizes[axis];
                    device += (remainder % size) * strides[axis];
                    remainder /= size;
                }
                devices.push(device);
            }

            devices
        };

        // The tile assignment must name every device of the mesh exactly once, because a repeated identifier would
        // otherwise leave a device of the mesh out of the assignment while the sharding that it maps to places a copy
        // on that device (e.g., `[0, 0]` in a replicated subgroup of two devices).
        let mut assigned = vec![false; device_count];
        for &device in &devices {
            if std::mem::replace(assigned.get_mut(device)?, true) {
                return None;
            }
        }

        // Pair the tile coordinates of every position of the tile assignment with the mesh coordinates of the device
        // at that position. Both are row-major coordinates, because device `i` is the `i`-th device of the mesh in
        // row-major order.
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
            let axis_names = axes.iter().map(|axis| mesh.axes()[*axis].name().to_string()).collect();
            dimensions.push(ShardingDimension::Sharded(axis_names));
        }

        Sharding::new(mesh.clone(), dimensions).ok()
    }
}

/// Dimensions of a lowered Shardy tensor sharding that the compiler may still shard along the
/// [`Auto`](MeshAxisType::Auto) axes of its mesh. Refer to [`ShardySharding`] for more information.
#[derive(Copy, Clone, Debug)]
pub(crate) enum OpenDimensions<'o> {
    /// Every dimension is open.
    All,

    /// Every dimension is open except the listed ones, which are closed.
    AllExcept(&'o [usize]),

    /// Every replicated or sharded dimension is closed. An unconstrained dimension stays open.
    None,
}

/// Lowering of a [`Sharding`] to its Shardy tensor sharding attribute. The replicated and sharded dimensions of the
/// attribute are closed (i.e., Shardy cannot shard them along further axes) unless the lowering leaves them open (refer
/// to [`OpenDimensions`]) and the mesh of the sharding has an [`Auto`](MeshAxisType::Auto) axis. The axes that the
/// sharding governs but does not use are listed as replicated either way, so open dimensions can only gain auto axes.
/// This is how JAX lowers the shardings of types over meshes with auto axes (i.e., `NamedSharding._to_sdy_sharding`
/// with `modify_wrt_axis_types=True`), whose placement along those axes the type does not describe and the compiler
/// chooses.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ShardySharding<'o> {
    /// [`Sharding`] that is being lowered.
    sharding: &'o Sharding,

    /// Dimensions of the lowered attribute that are open over the auto axes of the mesh of the sharding.
    open_dimensions: OpenDimensions<'o>,
}

impl<'o> ShardySharding<'o> {
    /// Creates a new [`ShardySharding`] that lowers `sharding` with `open_dimensions` open over the auto axes of
    /// its mesh.
    #[inline]
    pub(crate) fn new(sharding: &'o Sharding, open_dimensions: OpenDimensions<'o>) -> Self {
        Self { sharding, open_dimensions }
    }
}

impl ToMlir for ShardySharding<'_> {
    type Output<'c, 't: 'c> = shardy::TensorShardingAttributeRef<'c, 't>;

    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> Result<shardy::TensorShardingAttributeRef<'c, 't>, ryft_mlir::Error> {
        let Self { sharding, open_dimensions } = *self;
        let (open_over_auto_axes, closed_dimensions): (bool, &[usize]) = match open_dimensions {
            OpenDimensions::All => (true, &[]),
            OpenDimensions::AllExcept(dimensions) => (true, dimensions),
            OpenDimensions::None => (false, &[]),
        };
        let context = location.context();
        let closed =
            !open_over_auto_axes || sharding.mesh().axes().iter().all(|axis| axis.r#type() != MeshAxisType::Auto);
        let mesh_symbol_reference = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
        let dimensions = sharding
            .dimensions()
            .iter()
            .enumerate()
            .map(|(index, dimension)| {
                let closed_dimension = closed_dimensions.contains(&index);
                match dimension {
                    ShardingDimension::Replicated => {
                        context.shardy_dimension_sharding([], closed || closed_dimension, None)
                    }
                    ShardingDimension::Sharded(axis_names) => context.shardy_dimension_sharding(
                        axis_names
                            .iter()
                            .map(|axis_name| context.shardy_axis_ref(axis_name.as_str(), None))
                            .collect::<Result<Vec<_>, _>>()?,
                        closed || closed_dimension,
                        None,
                    ),
                    ShardingDimension::Unconstrained => context.shardy_dimension_sharding([], closed_dimension, None),
                }
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
            mesh_symbol_reference,
            dimensions.as_slice(),
            replicated_axes.as_slice(),
            unreduced_axes.as_slice(),
            sharding.unreduced_kind().to_mlir(location)?,
        )
    }
}

/// Lowers a tensor placement constraint of `input_value` to `sharding`, after excluding the mesh axes bound by
/// enclosing manual regions, and returns the constrained value. `open_dimensions` selects the dimensions that the
/// compiler may still shard along auto axes (refer to [`ShardySharding`]):
///
///   - The tracked placement of a type is lowered with [`OpenDimensions::All`], because a type does not fully describe
///     the placement along [`Auto`](MeshAxisType::Auto) axes, so the compiler stays free to place the value along them.
///   - A constraint that states the whole placement of a value, including along auto axes, is lowered with
///     [`OpenDimensions::None`]. A constraint that declares a pending reduction must use it, because Shardy drops the
///     pending reduction of a value whose constraint leaves a dimension open.
pub(crate) fn lower_sharding_constraint<'b, 'c: 'b, 't: 'c>(
    input_value: ValueRef<'b, 'c, 't>,
    sharding: &Sharding,
    open_dimensions: OpenDimensions<'_>,
    bound_manual_axes: &[String],
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    // Bound manual axes describe ownership outside this local tensor and cannot appear on constraints inside the
    // manual region. Restrict only the attribute's mesh inventory, because its symbol still names the module's full
    // mesh.
    let local_sharding;
    let sharding = if bound_manual_axes.is_empty() {
        sharding
    } else {
        let retain = |axis: &String| !bound_manual_axes.contains(axis);
        let projected = || -> Result<Sharding, ShardingError> {
            let mesh = LogicalMesh::new(
                sharding
                    .mesh()
                    .axes()
                    .iter()
                    .filter(|axis| !bound_manual_axes.iter().any(|bound| bound == axis.name()))
                    .cloned()
                    .collect(),
            )?;
            let dimensions = sharding
                .dimensions()
                .iter()
                .map(|dimension| match dimension {
                    ShardingDimension::Sharded(axes) => {
                        let axes = axes.iter().filter(|axis| retain(axis)).cloned().collect::<Vec<_>>();
                        if axes.is_empty() { ShardingDimension::Replicated } else { ShardingDimension::Sharded(axes) }
                    }
                    dimension => dimension.clone(),
                })
                .collect();
            let projected_sharding = Sharding::new(mesh, dimensions)?
                .with_unreduced_axes(sharding.unreduced_axes().iter().filter(|axis| retain(axis)).cloned())?;
            // The pending reduction keeps its kind unless the projection drops all of its axes.
            let unreduced_kind = if projected_sharding.unreduced_axes().is_empty() {
                UnreducedKind::Sum
            } else {
                sharding.unreduced_kind()
            };
            projected_sharding
                .with_unreduced_kind(unreduced_kind)?
                .with_reduced_axes(sharding.reduced_axes().iter().filter(|axis| retain(axis)).cloned())?
                .with_varying_manual_axes(sharding.varying_manual_axes().iter().filter(|axis| retain(axis)).cloned())
        };
        local_sharding = projected().map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        &local_sharding
    };
    let sharding_attribute = ShardySharding::new(sharding, open_dimensions).to_mlir(location)?;
    let operation = block.append_operation(shardy::sharding_constraint(input_value, sharding_attribute, location)?)?;
    Ok(operation.result(0).unwrap().as_ref())
}

/// Lowers a tensor placement constraint that replicates a value with the tracked placement `sharding` along `axes`
/// and returns the constrained value. The constraint is that of [`lower_sharding_constraint`] for `sharding` with
/// those dimensions replicated, except that the dimensions are closed, so that the compiler cannot shard them along
/// auto axes either. Operations whose lowering XLA cannot distribute along a dimension (e.g., a reduction whose
/// combiner is not a single operation) use it to have every device hold all of the elements along that dimension.
pub(crate) fn lower_replication_constraint<'b, 'c: 'b, 't: 'c>(
    input_value: ValueRef<'b, 'c, 't>,
    sharding: &Sharding,
    axes: &[usize],
    bound_manual_axes: &[String],
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let mut dimensions = sharding.dimensions().to_vec();
    for axis in axes {
        dimensions[*axis] = ShardingDimension::Replicated;
    }
    let replicated = sharding
        .with_dimensions(dimensions)
        .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
    lower_sharding_constraint(
        input_value,
        &replicated,
        OpenDimensions::AllExcept(axes),
        bound_manual_axes,
        block,
        location,
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{ArrayType, DataType, MeshAxis};
    use ryft_mlir::Context as MlirContext;

    use crate::lowering::types::lower_tensor_type;
    use crate::tests::{device_mesh_2x2, logical_mesh_2x2, logical_mesh_3x2x1};

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

    /// Returns the mesh with the manual axis `m`, the explicit axis `x`, and the auto axis `y`, each of size two.
    pub(crate) fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    /// Returns the `f32[4, 2]` type placed along `x` on its first dimension.
    pub(crate) fn test_input_type() -> ArrayType {
        let sharding =
            Sharding::new(test_mesh(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        ArrayType::new_static(DataType::F32, [4, 2]).with_sharding(sharding).unwrap()
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
    fn test_op_sharding_to_sharding() {
        let mesh = logical_mesh_2x2();
        let replicated = Some(Sharding::replicated(mesh.clone(), 2));
        let dimensions = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
        let along_x = Some(Sharding::new(mesh.clone(), dimensions).unwrap());
        let dimensions = vec![ShardingDimension::sharded(["y"]), ShardingDimension::replicated()];
        let along_y = Some(Sharding::new(mesh.clone(), dimensions).unwrap());

        // A replicated placement has no tiles, a maximal placement is replicated over a mesh with a single device,
        // and a tuple with a single element has the placement of that element.
        let op_sharding = OpSharding { r#type: OpShardingType::Replicated as i32, ..Default::default() };
        assert_eq!(op_sharding.to_sharding(&mesh, 2), replicated);
        let single_device_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let op_sharding = OpSharding { r#type: OpShardingType::Maximal as i32, ..Default::default() };
        assert_eq!(
            op_sharding.to_sharding(&single_device_mesh, 2),
            Some(Sharding::replicated(single_device_mesh.clone(), 2)),
        );
        let op_sharding = OpSharding {
            r#type: OpShardingType::Tuple as i32,
            tuple_shardings: vec![tiled_op_sharding(vec![2, 1, 2], vec![0, 1, 2, 3], true)],
            ..Default::default()
        };
        assert_eq!(op_sharding.to_sharding(&mesh, 2), along_x);

        // Tiling the first dimension along one axis replicates over the other axis on the last tile dimension. Device
        // `i` has the mesh coordinates `(i / 2, i % 2)`.
        let op_sharding = tiled_op_sharding(vec![2, 1, 2], vec![0, 1, 2, 3], true);
        assert_eq!(op_sharding.to_sharding(&mesh, 2), along_x);
        let op_sharding = tiled_op_sharding(vec![2, 1, 2], vec![0, 2, 1, 3], true);
        assert_eq!(op_sharding.to_sharding(&mesh, 2), along_y);

        // The iota form lists the same devices, in the order of the identity permutation unless it is transposed.
        let op_sharding =
            OpSharding { iota_reshape_dims: vec![2, 2], ..tiled_op_sharding(vec![2, 1, 2], vec![], true) };
        assert_eq!(op_sharding.to_sharding(&mesh, 2), along_x);
        let op_sharding = OpSharding { iota_transpose_perm: vec![1, 0], ..op_sharding };
        assert_eq!(op_sharding.to_sharding(&mesh, 2), along_y);

        // A dimension that is tiled over both axes lists them in major-to-minor order.
        let op_sharding = tiled_op_sharding(vec![4], vec![0, 1, 2, 3], false);
        assert_eq!(
            op_sharding.to_sharding(&mesh, 1),
            Some(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap()),
        );
        let op_sharding = tiled_op_sharding(vec![4], vec![0, 2, 1, 3], false);
        assert_eq!(
            op_sharding.to_sharding(&mesh, 1),
            Some(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y", "x"])]).unwrap()),
        );
    }

    #[test]
    fn test_op_sharding_to_sharding_with_unrepresentable_placements() {
        let mesh = logical_mesh_2x2();

        // Placements without tiles that are not replicated have no sharding.
        let op_sharding = OpSharding { r#type: OpShardingType::Maximal as i32, ..Default::default() };
        assert_eq!(op_sharding.to_sharding(&mesh, 2), None);
        let op_sharding = OpSharding { r#type: OpShardingType::Manual as i32, ..Default::default() };
        assert_eq!(op_sharding.to_sharding(&mesh, 2), None);
        let op_sharding = OpSharding { r#type: i32::MAX, ..Default::default() };
        assert_eq!(op_sharding.to_sharding(&mesh, 2), None);
        let element = tiled_op_sharding(vec![2, 1, 2], vec![0, 1, 2, 3], true);
        let op_sharding = OpSharding {
            r#type: OpShardingType::Tuple as i32,
            tuple_shardings: vec![element.clone(), element],
            ..Default::default()
        };
        assert_eq!(op_sharding.to_sharding(&mesh, 2), None);

        // Only replicated subgroups are representable.
        let op_sharding = OpSharding {
            last_tile_dims: vec![OpShardingType::Manual as i32],
            ..tiled_op_sharding(vec![2, 2], vec![0, 1, 2, 3], false)
        };
        assert_eq!(op_sharding.to_sharding(&mesh, 1), None);

        // The tile assignment must have one dimension for each array dimension and subgroup, and it must hold every
        // device of the mesh exactly as the mesh identifies it.
        let op_sharding = tiled_op_sharding(vec![2, 2], vec![0, 1, 2, 3], false);
        assert_eq!(op_sharding.to_sharding(&mesh, 1), None);
        let op_sharding = tiled_op_sharding(vec![2], vec![0, 1], false);
        assert_eq!(op_sharding.to_sharding(&mesh, 1), None);
        let op_sharding = tiled_op_sharding(vec![4], vec![0, 1, 2, 4], false);
        assert_eq!(op_sharding.to_sharding(&mesh, 1), None);
        let op_sharding = tiled_op_sharding(vec![4], vec![0, 1, 2, -1], false);
        assert_eq!(op_sharding.to_sharding(&mesh, 1), None);
        let two_device_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let op_sharding = tiled_op_sharding(vec![1, 2], vec![0, 0], true);
        assert_eq!(op_sharding.to_sharding(&two_device_mesh, 1), None);

        // An iota form whose permutation does not have one entry for each reshaped dimension is malformed.
        let op_sharding = OpSharding {
            iota_reshape_dims: vec![2, 2],
            iota_transpose_perm: vec![0],
            ..tiled_op_sharding(vec![2, 1, 2], vec![], true)
        };
        assert_eq!(op_sharding.to_sharding(&mesh, 2), None);

        // An assignment that no order of the mesh axes explains has no sharding.
        let op_sharding = tiled_op_sharding(vec![4], vec![0, 3, 1, 2], false);
        assert_eq!(op_sharding.to_sharding(&mesh, 1), None);
    }

    #[test]
    fn test_op_sharding_to_sharding_with_invalid_assignment_dimensions() {
        let mesh = logical_mesh_2x2();
        let overflowing_extent = 1_i64 << (usize::BITS / 2);
        let op_sharding = tiled_op_sharding(vec![overflowing_extent; 2], vec![], false);
        assert_eq!(op_sharding.to_sharding(&mesh, 2), None);

        // The iota reshape must hold exactly as many device identifiers as the assignment.
        for iota_reshape_dims in [vec![2, 3], vec![0, 4], vec![overflowing_extent; 2]] {
            let op_sharding = OpSharding { iota_reshape_dims, ..tiled_op_sharding(vec![4], vec![], false) };
            assert_eq!(op_sharding.to_sharding(&mesh, 1), None);
        }

        // Even if two unit extents are interchangeable, the transpose must reference each axis exactly once.
        let op_sharding = OpSharding {
            iota_reshape_dims: vec![1, 1, 4],
            iota_transpose_perm: vec![0, 0, 2],
            ..tiled_op_sharding(vec![4], vec![], false)
        };
        assert_eq!(op_sharding.to_sharding(&mesh, 1), None);
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
            ShardySharding::new(&sharding, OpenDimensions::All).to_mlir(location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{?}, {\"y\", ?}], replicated={\"z\"}>",
        );
        assert_eq!(
            ShardySharding::new(&sharding, OpenDimensions::None).to_mlir(location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{}, {\"y\"}], replicated={\"z\"}>",
        );

        // An open lowering keeps the requested dimensions closed, which the compiler cannot shard along auto axes.
        assert_eq!(
            ShardySharding::new(&sharding, OpenDimensions::AllExcept(&[0]))
                .to_mlir(location)
                .unwrap()
                .to_string(),
            "#sdy.sharding<@mesh, [{}, {\"y\", ?}], replicated={\"z\"}>",
        );

        // An open lowering over a mesh without auto axes closes every dimension too.
        let explicit_mesh = sharding.mesh().with_axis_types(["x"], MeshAxisType::Explicit).unwrap();
        let explicit_sharding = Sharding::new(explicit_mesh.clone(), sharding.dimensions().to_vec()).unwrap();
        assert_eq!(
            ShardySharding::new(&explicit_sharding, OpenDimensions::All).to_mlir(location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{}, {\"y\"}], replicated={\"x\", \"z\"}>",
        );

        // An unconstrained dimension stays open in a closed lowering.
        let unconstrained_sharding =
            Sharding::new(explicit_mesh, vec![ShardingDimension::unconstrained(), ShardingDimension::sharded(["y"])])
                .unwrap();
        assert_eq!(
            ShardySharding::new(&unconstrained_sharding, OpenDimensions::None)
                .to_mlir(location)
                .unwrap()
                .to_string(),
            "#sdy.sharding<@mesh, [{?}, {\"y\"}], replicated={\"x\", \"z\"}>",
        );
    }

    #[test]
    fn test_lower_sharding_constraint() {
        // The constraints are rendered outside any function, so their detached operands print as unknown SSA values.
        let input_type = test_input_type();
        let sharding = input_type.sharding().unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(&input_type, &context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let mut block_ref = block.as_ref();
        let location = location.as_ref();

        // The dimensions are open, open except for the listed ones, or closed.
        let open = lower_sharding_constraint(input, sharding, OpenDimensions::All, &[], &mut block_ref, location);
        let dimensions = OpenDimensions::AllExcept(&[1]);
        lower_sharding_constraint(input, sharding, dimensions, &[], &mut block_ref, location).unwrap();
        lower_sharding_constraint(input, sharding, OpenDimensions::None, &[], &mut block_ref, location).unwrap();

        // An axis that an enclosing manual region binds is left out of the constraint, and with it the dimensions
        // that are sharded along it alone.
        let bound = ["m".to_string()];
        lower_sharding_constraint(input, sharding, OpenDimensions::All, &bound, &mut block_ref, location).unwrap();
        let bound = ["x".to_string()];
        lower_sharding_constraint(input, sharding, OpenDimensions::None, &bound, &mut block_ref, location).unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(open.unwrap(), operations[0].result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{"x", ?}, {?}], replicated={"m"}> : tensor<4x2xf32>"#,
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{"x", ?}, {}], replicated={"m"}> : tensor<4x2xf32>"#,
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{"x"}, {}], replicated={"m"}> : tensor<4x2xf32>"#,
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{"x", ?}, {?}]> : tensor<4x2xf32>"#,
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{}, {}], replicated={"m"}> : tensor<4x2xf32>"#,
            ],
        );
    }

    #[test]
    fn test_lower_replication_constraint() {
        let input_type = test_input_type();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(&input_type, &context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let output = lower_replication_constraint(
            input,
            input_type.sharding().unwrap(),
            &[0],
            &[],
            &mut block.as_ref(),
            location.as_ref(),
        )
        .unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations[0].result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{}, {?}], replicated={"m", "x"}> : tensor<4x2xf32>"#,
            ],
        );
    }
}
