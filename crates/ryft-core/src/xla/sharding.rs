use ryft_mlir::{Location, dialects::shardy};

use crate::sharding::{DeviceMesh, LogicalMesh, Sharding, ShardingDimension};

/// Canonical symbol name used for emitted Shardy [`LogicalMesh`] declarations and references.
pub(crate) const SHARDY_MESH_SYMBOL_NAME: &str = "mesh";

impl LogicalMesh {
    /// Creates a new [`shardy::DetachedMeshOperation`] that corresponds to this [`LogicalMesh`].
    /// The mesh in the returned operation will be named `"mesh"`.
    #[inline]
    pub fn to_shardy<'c, 't: 'c, L: Location<'c, 't>>(&self, location: L) -> shardy::DetachedMeshOperation<'c, 't> {
        let context = location.context();
        let attribute = context
            .shardy_mesh(self.axes.iter().map(|axis| context.shardy_mesh_axis(axis.name.as_str(), axis.size)), &[]);
        shardy::mesh(SHARDY_MESH_SYMBOL_NAME, attribute, location)
    }
}

impl DeviceMesh {
    /// Creates a new [`shardy::DetachedMeshOperation`] that corresponds to this [`DeviceMesh`].
    /// The mesh in the returned operation will be named `"mesh"`.
    #[inline]
    pub fn to_shardy<'c, 't: 'c, L: Location<'c, 't>>(&self, location: L) -> shardy::DetachedMeshOperation<'c, 't> {
        self.logical_mesh.to_shardy(location)
    }
}

impl Sharding {
    /// Creates a new [`shardy::TensorShardingAttributeRef`] that corresponds to this [`Sharding`].
    /// The returned attribute uses the canonical `@mesh` symbol name in the MLIR context associated with `location`.
    pub fn to_shardy<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> shardy::TensorShardingAttributeRef<'c, 't> {
        let context = location.context();
        let mesh_symbol_ref = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
        let dimensions = self
            .dimensions
            .iter()
            .map(|dimension| match dimension {
                ShardingDimension::Replicated => context.shardy_dimension_sharding([], true, None),
                ShardingDimension::Sharded(axis_names) => context.shardy_dimension_sharding(
                    axis_names.iter().map(|axis_name| context.shardy_axis_ref(axis_name, None)),
                    true,
                    None,
                ),
                ShardingDimension::Unconstrained => context.shardy_dimension_sharding([], false, None),
            })
            .collect::<Vec<_>>();
        let replicated_axes = self
            .replicated_axes()
            .iter()
            .map(|axis_name| context.shardy_axis_ref(axis_name, None))
            .collect::<Vec<_>>();
        let unreduced_axes = self
            .unreduced_axes
            .iter()
            .map(|axis_name| context.shardy_axis_ref(axis_name, None))
            .collect::<Vec<_>>();
        context.shardy_tensor_sharding(
            mesh_symbol_ref,
            dimensions.as_slice(),
            replicated_axes.as_slice(),
            unreduced_axes.as_slice(),
        )
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_mlir::{Block, Context as MlirContext};

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, MeshDevice};

    use super::*;

    #[test]
    fn test_logical_mesh_to_shardy() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 1, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let context = MlirContext::new();
        let module = context.module(context.unknown_location());
        assert_eq!(
            module.body().append_operation(mesh.to_shardy(context.unknown_location())).to_string(),
            format!("sdy.mesh @{SHARDY_MESH_SYMBOL_NAME} = <[\"x\"=2, \"y\"=3, \"z\"=1]>"),
        );
    }

    #[test]
    fn test_device_mesh_to_shardy() {
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 1), MeshDevice::new(3, 1)];
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        let context = MlirContext::new();
        let module = context.module(context.unknown_location());
        assert_eq!(
            module.body().append_operation(mesh.to_shardy(context.unknown_location())).to_string(),
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
        let sharding = Sharding::with_unreduced_axes(
            mesh,
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            ["y"],
        )
        .unwrap();
        let context = MlirContext::new();
        assert_eq!(
            sharding.to_shardy(context.unknown_location()).to_string(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced={\"y\"}>",
        );
    }
}
