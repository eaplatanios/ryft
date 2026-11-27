use ryft_core::{LogicalMesh, MeshAxisType, ProgramError, Sharding, ShardingDimension, UnreducedKind};
use ryft_mlir::dialects::shardy;
use ryft_mlir::{Block, BlockRef, Location, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::Error;

/// Canonical symbol name used for emitted Shardy [`LogicalMesh`] declarations and references.
pub const SHARDY_MESH_SYMBOL_NAME: &str = "mesh";

/// Provides lowering support for [`LogicalMesh`]s to Shardy.
pub trait LogicalMeshExtension {
    /// Lowers this mesh to the Shardy mesh operation that declares it under [`SHARDY_MESH_SYMBOL_NAME`],
    /// in the [`Context`](ryft_mlir::Context) of `location`.
    fn lower<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> Result<shardy::DetachedMeshOperation<'c, 't>, ryft_mlir::Error>;
}

impl LogicalMeshExtension for LogicalMesh {
    #[inline]
    fn lower<'c, 't: 'c, L: Location<'c, 't>>(
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

/// Provides lowering support for [`UnreducedKind`]s to Shardy.
pub trait UnreducedKindExtension {
    /// Lowers this kind to the Shardy reduction operation that combines the contributions of an unreduced value.
    fn lower(self) -> shardy::ReductionOperation;
}

impl UnreducedKindExtension for UnreducedKind {
    #[inline]
    fn lower(self) -> shardy::ReductionOperation {
        match self {
            Self::Sum => shardy::ReductionOperation::Sum,
            Self::Max => shardy::ReductionOperation::Max,
            Self::Min => shardy::ReductionOperation::Min,
        }
    }
}

/// Dimensions of a lowered Shardy tensor sharding that the compiler may still shard along the
/// [`Auto`](MeshAxisType::Auto) axes of its mesh. Refer to [`ShardingExtension::lower`] for more information.
#[derive(Copy, Clone, Debug)]
pub enum OpenDimensions<'o> {
    /// Every dimension is open.
    All,

    /// Every dimension is open except the listed ones, which are closed.
    AllExcept(&'o [usize]),

    /// Every replicated or sharded dimension is closed. An unconstrained dimension stays open.
    None,
}

/// Provides lowering support for [`Sharding`]s to Shardy.
pub trait ShardingExtension {
    /// Lowers this sharding to its Shardy tensor sharding attribute, in the [`Context`](ryft_mlir::Context) of
    /// `location`. The replicated and sharded dimensions of the attribute are closed (i.e., Shardy cannot shard them
    /// along further axes) unless `open_dimensions` leaves them open (refer to [`OpenDimensions`]) and the mesh of
    /// the sharding has an [`Auto`](MeshAxisType::Auto) axis. The axes that the sharding governs but does not use
    /// are listed as replicated either way, so open dimensions can only gain auto axes.
    fn lower<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        open_dimensions: OpenDimensions<'_>,
        location: L,
    ) -> Result<shardy::TensorShardingAttributeRef<'c, 't>, ryft_mlir::Error>;

    /// Lowers a tensor placement constraint of `input_value` to this sharding, after excluding the mesh axes bound by
    /// enclosing manual regions, and returns the constrained value. `open_dimensions` selects the dimensions that the
    /// compiler may still shard along auto axes:
    ///
    ///   - The tracked placement of a type is lowered with [`OpenDimensions::All`], because a type does not fully
    ///     describe the placement along [`Auto`](MeshAxisType::Auto) axes, so the compiler stays free to place the
    ///     value along them.
    ///   - A constraint that states the whole placement of a value, including along auto axes, is lowered with
    ///     [`OpenDimensions::None`]. A constraint that declares a pending reduction must use it, because Shardy
    ///     drops the pending reduction of a value whose constraint leaves a dimension open.
    fn lower_sharding_constraint<'b, 'c: 'b, 't: 'c>(
        &self,
        input_value: ValueRef<'b, 'c, 't>,
        open_dimensions: OpenDimensions<'_>,
        bound_manual_axes: &[String],
        block: &mut BlockRef<'b, 'c, 't>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;

    /// Lowers a tensor placement constraint that replicates a value with this tracked placement along `axes` and
    /// returns the constrained value. The constraint is that of [`ShardingExtension::lower_sharding_constraint`] for
    /// this sharding with those dimensions replicated, except that the dimensions are closed, so that the compiler
    /// cannot shard them along auto axes either. Operations whose lowering XLA cannot distribute along a dimension
    /// (e.g., a eduction whose combiner is not a single operation) use it to have every device hold all of the
    /// elements along that dimension.
    fn lower_replication_constraint<'b, 'c: 'b, 't: 'c>(
        &self,
        input_value: ValueRef<'b, 'c, 't>,
        axes: &[usize],
        bound_manual_axes: &[String],
        block: &mut BlockRef<'b, 'c, 't>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;
}

impl ShardingExtension for Sharding {
    fn lower<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        open_dimensions: OpenDimensions<'_>,
        location: L,
    ) -> Result<shardy::TensorShardingAttributeRef<'c, 't>, ryft_mlir::Error> {
        let (open_over_auto_axes, closed_dimensions): (bool, &[usize]) = match open_dimensions {
            OpenDimensions::All => (true, &[]),
            OpenDimensions::AllExcept(dimensions) => (true, dimensions),
            OpenDimensions::None => (false, &[]),
        };
        let context = location.context();
        let closed = !open_over_auto_axes || self.mesh().axes().iter().all(|axis| axis.r#type() != MeshAxisType::Auto);
        let mesh_symbol_reference = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
        let dimensions = self
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
            mesh_symbol_reference,
            dimensions.as_slice(),
            replicated_axes.as_slice(),
            unreduced_axes.as_slice(),
            self.unreduced_kind().lower(),
        )
    }

    fn lower_sharding_constraint<'b, 'c: 'b, 't: 'c>(
        &self,
        input_value: ValueRef<'b, 'c, 't>,
        open_dimensions: OpenDimensions<'_>,
        bound_manual_axes: &[String],
        block: &mut BlockRef<'b, 'c, 't>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        // Bound manual axes are owned outside the local value and cannot appear on a constraint inside the manual
        // region, so the constraint states this sharding without them. Its symbol still names the full mesh of the
        // module.
        let local_sharding;
        let sharding = if bound_manual_axes.is_empty() {
            self
        } else {
            local_sharding = self.without_mesh_axes(bound_manual_axes)?;
            &local_sharding
        };
        let sharding_attribute = sharding.lower(open_dimensions, location)?;
        let operation =
            block.append_operation(shardy::sharding_constraint(input_value, sharding_attribute, location)?)?;
        Ok(operation.result(0)?.as_ref())
    }

    fn lower_replication_constraint<'b, 'c: 'b, 't: 'c>(
        &self,
        input_value: ValueRef<'b, 'c, 't>,
        axes: &[usize],
        bound_manual_axes: &[String],
        block: &mut BlockRef<'b, 'c, 't>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        let mut dimensions = self.dimensions().to_vec();
        for axis in axes {
            dimensions[*axis] = ShardingDimension::Replicated;
        }
        let replicated = self
            .with_dimensions(dimensions)
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        replicated.lower_sharding_constraint(
            input_value,
            OpenDimensions::AllExcept(axes),
            bound_manual_axes,
            block,
            location,
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{ArrayType, DataType, MeshAxis};
    use ryft_mlir::Context as MlirContext;

    use crate::lowering::types::ArrayTypeLoweringExtension;
    use crate::tests::logical_mesh_3x2x1;

    use super::*;

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
    fn test_logical_mesh_lower() {
        let mesh = logical_mesh_3x2x1();
        let context = MlirContext::new();
        let module = context.module(context.unknown_location()).unwrap();
        assert_eq!(
            module
                .body()
                .unwrap()
                .append_operation(mesh.lower(context.unknown_location()).unwrap())
                .unwrap()
                .to_string(),
            format!("sdy.mesh @{SHARDY_MESH_SYMBOL_NAME} = <[\"x\"=3, \"y\"=2, \"z\"=1]>"),
        );
    }

    #[test]
    fn test_unreduced_kind_lower() {
        assert_eq!(UnreducedKind::Sum.lower(), shardy::ReductionOperation::Sum);
        assert_eq!(UnreducedKind::Max.lower(), shardy::ReductionOperation::Max);
        assert_eq!(UnreducedKind::Min.lower(), shardy::ReductionOperation::Min);
    }

    #[test]
    fn test_sharding_lower() {
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
            sharding.lower(OpenDimensions::None, location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced={\"y\"}>",
        );

        // Kinds other than a sum lower to the reduction operation of the unreduced axes.
        let sharding = sharding.with_unreduced_kind(UnreducedKind::Max).unwrap();
        assert_eq!(
            sharding.lower(OpenDimensions::None, location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced=max{\"y\"}>",
        );
    }

    #[test]
    fn test_sharding_lower_leaves_dimensions_open_over_auto_axes() {
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
            sharding.lower(OpenDimensions::All, location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{?}, {\"y\", ?}], replicated={\"z\"}>",
        );
        assert_eq!(
            sharding.lower(OpenDimensions::None, location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{}, {\"y\"}], replicated={\"z\"}>",
        );

        // An open lowering keeps the requested dimensions closed, which the compiler cannot shard along auto axes.
        assert_eq!(
            sharding.lower(OpenDimensions::AllExcept(&[0]), location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{}, {\"y\", ?}], replicated={\"z\"}>",
        );

        // An open lowering over a mesh without auto axes closes every dimension too.
        let explicit_mesh = sharding.mesh().with_axis_types(["x"], MeshAxisType::Explicit).unwrap();
        let explicit_sharding = Sharding::new(explicit_mesh.clone(), sharding.dimensions().to_vec()).unwrap();
        assert_eq!(
            explicit_sharding.lower(OpenDimensions::All, location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{}, {\"y\"}], replicated={\"x\", \"z\"}>",
        );

        // An unconstrained dimension stays open in a closed lowering.
        let unconstrained_sharding =
            Sharding::new(explicit_mesh, vec![ShardingDimension::unconstrained(), ShardingDimension::sharded(["y"])])
                .unwrap();
        assert_eq!(
            unconstrained_sharding.lower(OpenDimensions::None, location).unwrap().to_string(),
            "#sdy.sharding<@mesh, [{?}, {\"y\"}], replicated={\"x\", \"z\"}>",
        );
    }

    #[test]
    fn test_sharding_lower_constraint() {
        // The constraints are rendered outside any function, so their detached inputs print as unknown SSA values.
        let input_type = test_input_type();
        let sharding = input_type.sharding().unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = input_type.lower(&context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let mut block_ref = block.as_ref();
        let location = location.as_ref();

        // The dimensions are open, open except for the listed ones, or closed.
        let open = sharding.lower_sharding_constraint(input, OpenDimensions::All, &[], &mut block_ref, location);
        let dimensions = OpenDimensions::AllExcept(&[1]);
        sharding.lower_sharding_constraint(input, dimensions, &[], &mut block_ref, location).unwrap();
        sharding
            .lower_sharding_constraint(input, OpenDimensions::None, &[], &mut block_ref, location)
            .unwrap();

        // An axis that an enclosing manual region binds is left out of the constraint, and with it the dimensions
        // that are sharded along it alone.
        let bound = ["m".to_string()];
        sharding
            .lower_sharding_constraint(input, OpenDimensions::All, &bound, &mut block_ref, location)
            .unwrap();
        let bound = ["x".to_string()];
        sharding
            .lower_sharding_constraint(input, OpenDimensions::None, &bound, &mut block_ref, location)
            .unwrap();
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
    fn test_sharding_lower_replication_constraint() {
        let input_type = test_input_type();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = input_type.lower(&context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let output = input_type
            .sharding()
            .unwrap()
            .lower_replication_constraint(input, &[0], &[], &mut block.as_ref(), location.as_ref())
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
