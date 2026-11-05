use ryft_core::{Device, DeviceMesh, LogicalMesh, Sharding, ShardingDimension, UnreducedKind};
use ryft_mlir::Location;
use ryft_mlir::dialects::shardy;
use ryft_pjrt::Device as PjrtDevice;

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

    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> Result<shardy::TensorShardingAttributeRef<'c, 't>, ryft_mlir::Error> {
        let context = location.context();
        let mesh_symbol_ref = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
        let dimensions = self
            .dimensions()
            .iter()
            .map(|dimension| match dimension {
                ShardingDimension::Replicated => context.shardy_dimension_sharding([], true, None),
                ShardingDimension::Sharded(axis_names) => context.shardy_dimension_sharding(
                    axis_names
                        .iter()
                        .map(|axis_name| context.shardy_axis_ref(axis_name.as_str(), None))
                        .collect::<Result<Vec<_>, _>>()?,
                    true,
                    None,
                ),
                ShardingDimension::Unconstrained => context.shardy_dimension_sharding([], false, None),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let replicated_axes = self
            .replicated_axes()
            .iter()
            .map(|axis_name| context.shardy_axis_ref(*axis_name, None))
            .collect::<Result<Vec<_>, _>>()?;
        let unreduced_axes = self
            .unreduced_axes()
            .iter()
            .map(|axis_name| context.shardy_axis_ref(axis_name.as_str(), None))
            .collect::<Result<Vec<_>, _>>()?;
        context.shardy_tensor_sharding(
            mesh_symbol_ref,
            dimensions.as_slice(),
            replicated_axes.as_slice(),
            unreduced_axes.as_slice(),
            shardy_reduction_operation(self.unreduced_kind()),
        )
    }
}

/// Returns the Shardy reduction operation of the unreduced axes of a sharding whose [`UnreducedKind`] is `kind`.
pub(crate) fn shardy_reduction_operation(kind: UnreducedKind) -> shardy::ReductionOperation {
    match kind {
        UnreducedKind::Sum => shardy::ReductionOperation::Sum,
        UnreducedKind::Max => shardy::ReductionOperation::Max,
        UnreducedKind::Min => shardy::ReductionOperation::Min,
    }
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
}
