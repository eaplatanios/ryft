use std::collections::BTreeSet;
use std::rc::Rc;

use ryft_core::{DataType, LogicalMesh, MeshAxisType, ProgramError, Sharding, ShardingDimension, UnreducedKind};
use ryft_mlir::dialects::shardy;
use ryft_mlir::{
    Block, BlockRef, Location, LocationRef, OpRef, Operation as MlirOperation, OperationRef, OperationResultRef,
    TensorTypeRef, Type as MlirType, Value as MlirValue, ValueRef,
};

use crate::arrays::host::{
    CollectiveReduction, ReductionChange, collective_reduction, compiled_reduction_rejection,
    partial_completion_rejection,
};
use crate::lowering::LoweringError;
use crate::lowering::contexts::ProgramLoweringContext;
use crate::{Error, Type};

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
///
/// What these functions emit refers to the mesh of the module by [`SHARDY_MESH_SYMBOL_NAME`], and a module declares
/// that mesh only when the lowering of an operation states that it refers to it. The lowering of an operation that
/// calls these functions therefore declares the mesh of the sharding with
/// [`Lowerer::require_mesh`](crate::Lowerer::require_mesh).
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

/// A reduction that still needs to combine a value's per-device contributions.
#[derive(Clone)]
struct PendingReduction {
    /// Non-empty set of mesh axes along which devices hold separate contributions. Axes bound by enclosing manual
    /// regions are excluded.
    axes: BTreeSet<String>,

    /// Operation that combines the contributions across all of these axes, such as a sum or maximum.
    kind: UnreducedKind,
}

impl PendingReduction {
    /// Returns the pending reduction of a value of type `r#type`, if it has one over an axis that is not one of the
    /// `bound_manual_axes`.
    fn of_type(r#type: &Type, bound_manual_axes: &[String]) -> Option<Self> {
        let Type::Array(r#type) = r#type else {
            return None;
        };
        let axes = r#type
            .unreduced_axes()
            .iter()
            .filter(|axis| !bound_manual_axes.contains(axis))
            .cloned()
            .collect::<BTreeSet<_>>();
        (!axes.is_empty()).then(|| Self { axes, kind: r#type.unreduced_kind() })
    }

    /// Returns the pending reduction that a `sdy.sharding_constraint` operation declares for its result, if any.
    fn of_constraint<'o, 'c: 'o, 't: 'c>(operation: OperationRef<'o, 'c, 't>) -> Result<Option<Self>, Error> {
        let Some(constraint) = (unsafe { operation.cast::<shardy::ShardingConstraintOperationRef>() }) else {
            return Ok(None);
        };
        let sharding = shardy::ShardingConstraintOperation::sharding(&constraint)?;
        let axes = sharding
            .unreduced_axes()
            .iter()
            .map(|axis| axis.name().as_str().map(str::to_string))
            .collect::<Result<BTreeSet<_>, _>>()
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        let kind = match sharding.reduction_operation()? {
            shardy::ReductionOperation::Sum => UnreducedKind::Sum,
            shardy::ReductionOperation::Max => UnreducedKind::Max,
            shardy::ReductionOperation::Min => UnreducedKind::Min,
        };
        Ok((!axes.is_empty()).then_some(Self { axes, kind }))
    }

    /// Returns the sharding of rank `rank` over `mesh` that declares this pending reduction for a value whose
    /// dimensions are partitioned as `dimensions`, if known, and are otherwise replicated. Replication is also
    /// the fallback when `dimensions` partitions a dimension along an axis of the pending reduction.
    fn sharding(
        &self,
        mesh: &LogicalMesh,
        rank: usize,
        dimensions: Option<&[ShardingDimension]>,
    ) -> Result<Sharding, Error> {
        let declare = |sharding: Sharding| {
            sharding
                .with_unreduced_axes(self.axes.iter().cloned())
                .and_then(|sharding| sharding.with_unreduced_kind(self.kind))
        };
        let partitioned = dimensions
            .filter(|dimensions| dimensions.len() == rank)
            .and_then(|dimensions| Sharding::new(mesh.clone(), dimensions.to_vec()).and_then(declare).ok());
        match partitioned {
            Some(sharding) => Ok(sharding),
            None => declare(Sharding::replicated(mesh.clone(), rank))
                .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() }.into()),
        }
    }
}

/// StableHLO operations through which [`constrain_unreduced_intermediates`] can copy dimension shardings between
/// an operand and a result of the same shape. Shape equality alone is insufficient because transposing a square matrix
/// preserves its shape while swapping its dimensions. This list covers a subset of the operations that Shardy's
/// [`sdy::isElementwise`](https://github.com/openxla/shardy/blob/57fffc23dfdf139b80013ad681701cee59c22f96/shardy/dialect/sdy/transforms/common/op_properties.cc#L30-L45)
/// recognizes. The Rust bindings do not expose that predicate.
const ELEMENTWISE_OPERATIONS: &[&str] = &[
    "stablehlo.abs",
    "stablehlo.add",
    "stablehlo.and",
    "stablehlo.bitcast_convert",
    "stablehlo.clamp",
    "stablehlo.compare",
    "stablehlo.convert",
    "stablehlo.divide",
    "stablehlo.maximum",
    "stablehlo.minimum",
    "stablehlo.multiply",
    "stablehlo.negate",
    "stablehlo.not",
    "stablehlo.or",
    "stablehlo.select",
    "stablehlo.subtract",
    "stablehlo.xor",
];

/// Returns the dimension shardings that the `sdy.sharding_constraint` that produces `value` declares, including
/// along auto mesh axes, which no type records, or [`None`] when `value` is not the result of such a constraint or the
/// constraint partitions a dimension along a part of an axis. An empty dimension is replicated when it is closed and
/// unconstrained when it is open. This is what the program states about the placement of a value along auto mesh axes
/// (e.g., the constraint that [`ProgramLoweringContext::lower_pending_reduction_constraint`] emits for an input that
/// enters with a pending reduction, or for the result of an earlier instruction), so it is read from the program rather
/// than reconstructed.
fn constrained_dimensions(value: ValueRef<'_, '_, '_>) -> Result<Option<Rc<[ShardingDimension]>>, Error> {
    let Some(result) = value.cast::<OperationResultRef>() else {
        return Ok(None);
    };

    let operation = result.operation()?;
    if operation.name().as_str() != Ok("sdy.sharding_constraint") {
        return Ok(None);
    }

    let Some(constraint) = (unsafe { operation.cast::<shardy::ShardingConstraintOperationRef>() }) else {
        return Ok(None);
    };

    let sharding = shardy::ShardingConstraintOperation::sharding(&constraint)?;
    let mut dimensions = Vec::new();
    for dimension in sharding.dim_shardings() {
        let mut axes = Vec::new();
        for axis in dimension.axes() {
            if axis.sub_axis_info()?.is_some() {
                return Ok(None);
            }

            let name = axis.name();
            let name = name.as_str().map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
            axes.push(name.to_string());
        }

        dimensions.push(if !axes.is_empty() {
            ShardingDimension::Sharded(axes)
        } else if dimension.is_closed() {
            ShardingDimension::Replicated
        } else {
            ShardingDimension::Unconstrained
        });
    }

    Ok(Some(dimensions.into()))
}

/// Returns `dimensions`, dimension shardings over `mesh` that the program states for a value (e.g., with
/// [`constrained_dimensions`]), if they agree with `stated`, the closed dimension shardings of the type of that value
/// (refer to [`Sharding::dimensions_with_unconstrained_replicated`]), along every mesh axis that is not an auto axis,
/// which is all that a type records, and `stated` otherwise. A value whose type tracks a placement is placed as its
/// type says along those axes, so the program can only add the placement along auto mesh axes, which its type cannot
/// record.
fn dimensions_along_auto_axes(
    dimensions: Option<Rc<[ShardingDimension]>>,
    stated: Rc<[ShardingDimension]>,
    mesh: &LogicalMesh,
) -> Rc<[ShardingDimension]> {
    let without_auto_axes = |dimension: &ShardingDimension| match dimension {
        ShardingDimension::Sharded(axes) => {
            let axes = axes
                .iter()
                .filter(|axis| mesh.axis_type(axis) != Some(MeshAxisType::Auto))
                .cloned()
                .collect::<Vec<_>>();
            if axes.is_empty() { ShardingDimension::Replicated } else { ShardingDimension::Sharded(axes) }
        }
        ShardingDimension::Replicated | ShardingDimension::Unconstrained => ShardingDimension::Replicated,
    };

    match dimensions {
        Some(dimensions)
            if dimensions.len() == stated.len()
                && dimensions.iter().map(without_auto_axes).eq(stated.iter().cloned()) =>
        {
            dimensions
        }
        _ => stated,
    }
}

// TODO(eaplatanios): Review this.
/// Returns the dimension shardings of the result of an elementwise operation whose operands of the shape of that
/// result are partitioned as `operands` (refer to [`constrain_unreduced_intermediates`]), whatever the order of the
/// operands: each dimension takes the split that the operands that split it state, the operands that leave it unsplit
/// being sliced to it locally, and when two operands split it differently, which they can only do along auto mesh axes
/// (their placements agree with their types along every other axis, refer to [`dimensions_along_auto_axes`]), it
/// takes the split along the other axes alone, which their types record, and the operands are resharded to it. An
/// axis that the dimensions so chosen use more than once (i.e., that operands use to split different dimensions) splits
/// none of them. Returns [`None`] when no operand states dimension shardings.
fn merged_dimensions(operands: &[Rc<[ShardingDimension]>], mesh: &LogicalMesh) -> Option<Rc<[ShardingDimension]>> {
    let (first, rest) = operands.split_first()?;
    if rest.is_empty() {
        return Some(Rc::clone(first));
    }
    let dimensions = (0..first.len())
        .map(|index| {
            let mut splits = operands.iter().filter_map(|dimensions| match dimensions.get(index) {
                Some(ShardingDimension::Sharded(axes)) => Some(axes),
                _ => None,
            });
            let Some(split) = splits.next() else { return first[index].clone() };
            if splits.all(|other| other == split) {
                return ShardingDimension::Sharded(split.clone());
            }
            let axes = split
                .iter()
                .filter(|axis| mesh.axis_type(axis) != Some(MeshAxisType::Auto))
                .cloned()
                .collect::<Vec<_>>();
            if axes.is_empty() { ShardingDimension::Replicated } else { ShardingDimension::Sharded(axes) }
        })
        .collect::<Vec<_>>();
    // An axis splits at most one dimension of a value, so an axis that operands use to split different dimensions
    // (e.g., the rows of one operand and the columns of the other along `d`) splits none of them, whatever the order of
    // the operands, and the operands are resharded to the result.
    let uses = |axis: &String| {
        dimensions
            .iter()
            .filter(|dimension| matches!(dimension, ShardingDimension::Sharded(axes) if axes.contains(axis)))
            .count()
    };
    let crossed = dimensions
        .iter()
        .flat_map(|dimension| match dimension {
            ShardingDimension::Sharded(axes) => axes.iter().filter(|axis| uses(axis) > 1).cloned().collect(),
            _ => Vec::new(),
        })
        .collect::<Vec<_>>();
    if crossed.is_empty() {
        return Some(dimensions.into());
    }
    let dimensions = dimensions
        .into_iter()
        .map(|dimension| match dimension {
            ShardingDimension::Sharded(axes) => {
                let axes = axes.into_iter().filter(|axis| !crossed.contains(axis)).collect::<Vec<_>>();
                if axes.is_empty() { ShardingDimension::Replicated } else { ShardingDimension::Sharded(axes) }
            }
            dimension => dimension,
        })
        .collect();
    Some(dimensions)
}

// TODO(eaplatanios): Review this.
/// Validates `declared`, a sharding that the lowering declares on a value of `data_type` (i.e., an output sharding of
/// the entry function, the sharding of a `constrain_sharding`, or the boundary sharding of an input or an output of a
/// `call`), against `carried`, the sharding that the type of that value records, if any. XLA exports a sharding
/// that adds unreduced axes to a value as the introduction of a pending reduction, which it implements (by filling the
/// other devices with an identity) only for [`UnreducedKind::Sum`] and otherwise aborts the process. A declared pending
/// maximum or minimum must therefore be one that the value already carries, which is the case exactly when its type
/// records it over the same axes with the same kind. A type never records reduction state over auto mesh axes (and the
/// lowering completes such a state where an input enters), so a pending maximum or minimum over an auto axis is
/// always rejected. A pending sum that the sharding introduces, and a pending reduction that the value carries and the
/// sharding drops, are changes that a collective makes, which must be one that the collective makes for `data_type`
/// (refer to [`validate_pending_reduction_change`]). `operation` names the construct in the error, and the axes that an
/// enclosing manual region binds are left to the collectives of that region.
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedOperation`] when `declared` introduces a pending maximum or minimum, or makes a
/// change of reduction state that [`validate_pending_reduction_change`] rejects.
pub(crate) fn validate_declared_pending_reduction(
    declared: &Sharding,
    carried: Option<&Sharding>,
    data_type: DataType,
    bound_manual_axes: &[String],
    operation: &str,
) -> Result<(), Error> {
    let kind = declared.unreduced_kind();
    let carries = |axis: &String| {
        carried.is_some_and(|carried| carried.unreduced_kind() == kind && carried.unreduced_axes().contains(axis))
    };
    let introduced = declared
        .unreduced_axes()
        .iter()
        .filter(|axis| !bound_manual_axes.contains(axis) && !carries(axis))
        .collect::<Vec<_>>();
    if kind != UnreducedKind::Sum && !introduced.is_empty() {
        let introduced = introduced.iter().map(|axis| format!("`{axis}`")).collect::<Vec<_>>();
        return Err(Error::Lowering(LoweringError::UnsupportedOperation {
            operation: operation.to_string(),
            reason: format!(
                "the sharding `{declared}` declares a pending `{kind}` reduction over mesh axes (i.e., {}) along \
                 which the value does not carry one, and XLA can only introduce a pending `sum` reduction (types never \
                 record reduction state over auto mesh axes, so a pending `{kind}` reduction over them is never \
                 carried)",
                introduced.join(", "),
            ),
        }));
    }
    let change = ReductionChange::Introduce(UnreducedKind::Sum);
    validate_pending_reduction_change(data_type, change, declared, introduced, operation)?;
    if let Some(carried) = carried {
        let completed = carried.unreduced_axes().iter().filter(|axis| {
            !bound_manual_axes.contains(axis)
                && (declared.unreduced_kind() != carried.unreduced_kind() || !declared.unreduced_axes().contains(*axis))
        });
        let change = ReductionChange::Complete(carried.unreduced_kind());
        validate_pending_reduction_change(data_type, change, declared, completed.clone(), operation)?;
        let kept = carried.unreduced_axes().iter().filter(|axis| {
            !bound_manual_axes.contains(axis)
                && declared.unreduced_kind() == carried.unreduced_kind()
                && declared.unreduced_axes().contains(*axis)
        });
        validate_partial_completion(data_type, carried.unreduced_kind(), declared, completed, kept, operation)?;
    }
    Ok(())
}

// TODO(eaplatanios): Review this.
/// Validates the whole transition of a pending reduction of `kind` to a value of `data_type` that the lowering of
/// `sharding` completes along the mesh axes `completed` and keeps along the mesh axes `pending`, with the rule of
/// [`partial_completion_rejection`], which the device placements of [`crate::arrays::transfers`] read as well. Those
/// placements take a host round trip where the rule rejects the transition, while a compiled program has no exact
/// alternative, so it is rejected before anything is compiled. Every site where the lowering completes a pending
/// reduction (i.e., an entry input with an input sharding override, and the user-stated shardings of
/// [`validate_declared_pending_reduction`]) checks its transition here, and not only each completed axis in isolation.
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedOperation`] for `operation` when the transition is rejected.
pub(super) fn validate_partial_completion<'a>(
    data_type: DataType,
    kind: UnreducedKind,
    sharding: &Sharding,
    completed: impl IntoIterator<Item = &'a String>,
    pending: impl IntoIterator<Item = &'a String>,
    operation: &str,
) -> Result<(), Error> {
    let sized = |axis: &'a String| (axis.as_str(), sharding.mesh().axis_size(axis).unwrap_or(1));
    let completed = completed.into_iter().map(sized);
    let Some(rejection) = partial_completion_rejection(data_type, kind, completed, pending.into_iter().map(sized))
    else {
        return Ok(());
    };
    Err(Error::Lowering(LoweringError::UnsupportedOperation {
        operation: operation.to_string(),
        reason: format!(
            "the sharding `{sharding}` {rejection}; complete the reduction along all of these mesh axes at once, or \
             use a data type whose sums accumulate in itself (e.g., `f32`)",
        ),
    }))
}

// TODO(eaplatanios): Review this.
/// Validates that a collective of XLA makes `change` (i.e., completes or introduces a pending reduction) along the mesh
/// axes `axes` for values of `data_type` as that data type means it, where `sharding` is the sharding whose lowering
/// makes the change. The rule is [`compiled_reduction_rejection`], which the lowering of every collective that reduces
/// shares: a change that the collective makes exactly up to the contract of floating point sums is accepted, and so is
/// a maximum or minimum of floating point values that the collective completes and that can drop NaN values, because a
/// compiled program has no exact alternative (refer to the section on pending reductions of [`crate::arrays::host`]).
/// Every other change computes another number, fails to compile, aborts or crashes the process, or has no meaning, and
/// is rejected before anything is compiled. A reduction over an axis of size one changes no value, so such axes are
/// exempt, as they are for the device placements of [`crate::arrays::transfers`].
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedOperation`] for `operation` when the change is rejected.
pub(super) fn validate_pending_reduction_change<'a, I: IntoIterator<Item = &'a String>>(
    data_type: DataType,
    change: ReductionChange,
    sharding: &Sharding,
    axes: I,
    operation: &str,
) -> Result<(), Error> {
    let axes = axes
        .into_iter()
        .filter(|axis| sharding.mesh().axis_size(axis) != Some(1))
        .map(|axis| format!("`{axis}`"))
        .collect::<Vec<_>>();
    if axes.is_empty() {
        return Ok(());
    }
    let Some(reason) = compiled_reduction_rejection(data_type, change) else {
        return Ok(());
    };
    // A placement of a concrete array introduces a pending sum exactly on every mesh, except for a data type without an
    // additive identity, which nothing can split into summands, while it completes a reduction exactly with a host
    // round trip only where one process owns every device of the destination and a device of every contribution of the
    // array that has elements.
    let (verb, kind, remedy) = match change {
        ReductionChange::Introduce(kind)
            if collective_reduction(data_type, change) == CollectiveReduction::MissingIdentity =>
        {
            ("introduce", kind, "use a data type that has an additive identity")
        }
        ReductionChange::Complete(kind) => (
            "complete",
            kind,
            "complete the reduction with a placement of the concrete array instead (e.g., `Array::to_placement`), \
             which is exact where one process owns every device of the destination and a device of every contribution \
             of the array that has elements, or use another data type",
        ),
        ReductionChange::Introduce(kind) => (
            "introduce",
            kind,
            "introduce the reduction with a placement of the concrete array instead (e.g., `Array::to_placement`), \
             which writes additive identities exactly on every mesh, or use another data type",
        ),
    };
    Err(Error::Lowering(LoweringError::UnsupportedOperation {
        operation: operation.to_string(),
        reason: format!(
            "the sharding `{sharding}` would {verb} a pending `{kind}` reduction of `{data_type}` values over mesh \
             axes (i.e., {}), which XLA does not do as the data type means it, because {reason}; {remedy}",
            axes.join(", "),
        ),
    }))
}

// TODO(eaplatanios): Review this.
/// Constrains the intermediate values that the lowering of one instruction emitted into `block` after `anchor` and that
/// carry a pending reduction, for `lower_instruction_operation`. The `outputs` of the instruction, of types
/// `output_types`, are left to [`constrain_unreduced_outputs`].
///
/// The pending reductions follow the dataflow of the emitted operations, in their order in `block`:
///
///   - The `input_values` of the instruction carry the pending reductions of their `input_types`.
///   - A `sdy.sharding_constraint` result carries the pending reduction that the constraint declares (e.g., none for a
///     constraint that completes the reduction). The results of the other Shardy operations, of `func.call`, and of
///     `stablehlo.case` and `stablehlo.if` carry none here, because their shardings are declared by the operation or
///     by the callee, and the values computed in their regions or callees are constrained by the lowering of the
///     instructions there.
///   - Result `i` of `stablehlo.while` or `stablehlo.optimization_barrier` carries the pending reduction of operand
///     `i`. The loop body's instructions constrain their own results. A barrier can forward pending and replicated
///     tensors alongside effect tokens. Its pending results may have different axes, but Shardy requires them to
///     have the same reduction kind.
///   - Every tensor result of any other operation (e.g., `stablehlo.compare`, `stablehlo.convert`, `stablehlo.reduce`,
///     or a `stablehlo.broadcast_in_dim` of a scalar) that has an operand with a pending reduction carries the pending
///     reduction of its operands, because each device computes the operation on its own contribution. Shardy requires
///     such a result to keep the unreduced axes of its operands with the same kind, whatever its element type (e.g.,
///     the Boolean result of a comparison of maximum-unreduced values is maximum-unreduced, and the `f32` copy of a
///     `bf16` sum-unreduced value is sum-unreduced), and all results of an operation with several results to have the
///     same unreduced axes. The declaration does not move any data, so it does not change the values. The reduction
///     is only completed where an output type of the instruction or a placement says so, and the core type rules make
///     each such output type the reduction state of the values that the operation computes. Operands without a
///     pending reduction (e.g., a constant, or an operand that the core rules require to be reduced, like the
///     denominator of a `div`) do not contribute to it. The core rules reject operands that carry pending
///     reductions of different kinds, which no single Shardy sharding can declare, so such an operation is rejected
///     with [`LoweringError::UnsupportedOperation`]. Operands that are unreduced over different axes (e.g., the
///     factors of a `mul`) give a result that is unreduced over the union of those axes, like the core rules.
///
/// Each constrained value is a `sdy.sharding_constraint` placed right after the operation that produces it, which
/// declares its pending reduction, and it replaces every later use of that value. A value that a
/// `sdy.sharding_constraint` already uses (e.g., one that the lowering of an instruction of a nested region inlined
/// into the same block constrained) is left as it is. Operations nested in the regions of emitted operations are not
/// visited, because their region arguments carry no sharding of their own (e.g., the scalar arguments of a
/// `stablehlo.reduce` body).
///
/// The constraint partitions the dimensions of the value as an input of the instruction with a pending reduction or an
/// output of the instruction partitions its own, where the value is known to be partitioned alike, so that it does not
/// gather a value that the instruction leaves split along another axis. The dimension shardings of an input are those
/// that the program states for it, including along auto mesh axes (refer to [`constrained_dimensions`]), which its type
/// does not record, and otherwise those of its type. These dimension shardings follow the [`ELEMENTWISE_OPERATIONS`]
/// whose operands and results have the same shape: forward from the inputs to the results, and backward from the
/// outputs to the operands, which covers the intermediates of a reduction that an elementwise operation turns into an
/// output (e.g., the sum of a mean before its division). Dimensions also flow in both directions between each
/// barrier operand and its corresponding result, and forward along each loop carry. Other operations may permute or
/// drop dimensions, so their results take only the dimension shardings that reach them backward from an output. An
/// intermediate that neither reaches is replicated in every dimension, as its type records. Its placement along auto
/// mesh axes is then unknown, and a dimension left open would let Shardy drop the pending reduction (refer to
/// [`ProgramLoweringContext::lower_pending_reduction_constraint`]), so the dimensions are always closed. The
/// dimension shardings that reach each output forward are returned, in the order of `outputs`, for
/// [`constrain_unreduced_outputs`].
#[allow(clippy::too_many_arguments)]
pub(super) fn constrain_unreduced_intermediates<'b, 'c: 'b, 't: 'c>(
    anchor: OperationRef<'b, 'c, 't>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[Type],
    outputs: &[ValueRef<'b, 'c, 't>],
    output_types: &[Type],
    lowering_context: &ProgramLoweringContext,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<Option<Rc<[ShardingDimension]>>>, Error> {
    /// How the emitted operation relates its operand and result shardings.
    #[derive(Copy, Clone, PartialEq, Eq)]
    enum Propagation {
        ShardingConstraint,
        DeclaredElsewhere,
        While,
        OptimizationBarrier,
        Elementwise,
        Other,
    }

    let bound_manual_axes = lowering_context.bound_manual_axes();
    let mut pending_reductions = input_values
        .iter()
        .zip(input_types)
        .filter_map(|(value, r#type)| Some((*value, PendingReduction::of_type(r#type, bound_manual_axes)?)))
        .collect::<Vec<_>>();
    // Every value of a lowered program is placed over the same mesh, which is the one of any input with a pending
    // reduction.
    let Some(mesh) = input_types.iter().zip(input_values).find_map(|(input_type, value)| match input_type {
        Type::Array(input_type) if pending_reductions.iter().any(|(candidate, _)| candidate == value) => {
            input_type.sharding().map(Sharding::mesh)
        }
        _ => None,
    }) else {
        return Ok(Vec::new());
    };
    // The emitted operations are collected before any constraint is inserted, so that the constraints are not visited.
    let operations = block
        .operations_after(&anchor)?
        .map(|operation| {
            let operation = operation?;
            let name = operation.name();
            let propagation =
                match name.as_str().map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })? {
                    "sdy.sharding_constraint" => Propagation::ShardingConstraint,
                    "stablehlo.while" => Propagation::While,
                    "stablehlo.optimization_barrier" => Propagation::OptimizationBarrier,
                    "func.call" | "stablehlo.case" | "stablehlo.if" => Propagation::DeclaredElsewhere,
                    name if name.starts_with("sdy.") => Propagation::DeclaredElsewhere,
                    name if ELEMENTWISE_OPERATIONS.contains(&name) => Propagation::Elementwise,
                    _ => Propagation::Other,
                };
            Ok((operation, propagation))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let find = |value: ValueRef<'b, 'c, 't>, dimensions: &[(ValueRef<'b, 'c, 't>, Rc<[ShardingDimension]>)]| {
        dimensions
            .iter()
            .find(|(candidate, _)| *candidate == value)
            .map(|(_, dimensions)| Rc::clone(dimensions))
    };

    // The dimension shardings of the inputs with a pending reduction, which flow forward, and those of the outputs,
    // which flow backward through the elementwise operations, before any constraint replaces a use. An input keeps the
    // placement along auto mesh axes that the program states for it (refer to `constrained_dimensions`), which its type
    // cannot record, so that a value that is split along an auto axis stays split.
    let mut forward_dimensions = Vec::new();
    for (value, r#type) in input_values.iter().zip(input_types) {
        if let Type::Array(r#type) = r#type
            && let Some(sharding) = r#type.sharding()
            && pending_reductions.iter().any(|(candidate, _)| candidate == value)
        {
            let dimensions = dimensions_along_auto_axes(
                constrained_dimensions(*value)?,
                sharding.dimensions_with_unconstrained_replicated().into(),
                mesh,
            );
            forward_dimensions.push((*value, dimensions));
        }
    }
    let mut backward_dimensions = outputs
        .iter()
        .zip(output_types)
        .filter_map(|(value, r#type)| match r#type {
            Type::Array(r#type) => r#type
                .sharding()
                .map(|sharding| (*value, sharding.dimensions_with_unconstrained_replicated().into())),
            _ => None,
        })
        .collect::<Vec<_>>();
    for &(operation, propagation) in operations.iter().rev() {
        if propagation == Propagation::OptimizationBarrier {
            for (index, operand) in operation.operand_values().enumerate() {
                let operand = operand?;
                if let Some(dimensions) = find(operation.result(index)?.as_ref(), &backward_dimensions)
                    && find(operand, &backward_dimensions).is_none()
                {
                    backward_dimensions.push((operand, dimensions));
                }
            }
            continue;
        }
        if propagation != Propagation::Elementwise || operation.result_count() != 1 {
            continue;
        }
        let result = operation.result(0)?.as_ref();
        let Some(dimensions) = find(result, &backward_dimensions) else { continue };
        let shape = result.tensor_shape()?;
        for operand in operation.operand_values() {
            let operand = operand?;
            if find(operand, &backward_dimensions).is_none() && operand.tensor_shape()? == shape {
                backward_dimensions.push((operand, Rc::clone(&dimensions)));
            }
        }
    }

    // The values of a nested region that the lowering inlines into the same block are constrained by the lowering of
    // the instructions of that region, which this walk visits again.
    let is_constrained = |value: ValueRef<'b, 'c, 't>| -> Result<bool, Error> {
        for operand in value.uses()? {
            if operand?.operation()?.name().as_str() == Ok("sdy.sharding_constraint") {
                return Ok(true);
            }
        }
        Ok(false)
    };
    let pending_reduction_of =
        |value: ValueRef<'b, 'c, 't>, pending_reductions: &[(ValueRef<'b, 'c, 't>, PendingReduction)]| {
            pending_reductions
                .iter()
                .find(|(candidate, _)| *candidate == value)
                .map(|(_, reduction)| reduction.clone())
        };
    for (operation, propagation) in operations {
        let result_reductions = match propagation {
            Propagation::ShardingConstraint => vec![PendingReduction::of_constraint(operation)?],
            Propagation::DeclaredElsewhere => continue,
            Propagation::While | Propagation::OptimizationBarrier => {
                let reductions = operation
                    .operand_values()
                    .map(|operand| Ok(pending_reduction_of(operand?, &pending_reductions)))
                    .collect::<Result<Vec<_>, Error>>()?;
                if propagation == Propagation::OptimizationBarrier {
                    let mut kinds = reductions.iter().flatten().map(|reduction| reduction.kind);
                    if let Some(first) = kinds.next()
                        && let Some(other) = kinds.find(|kind| *kind != first)
                    {
                        return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                            operation: operation.name().to_string(),
                            reason: format!(
                                "Shardy cannot represent pending reductions of different kinds `{first}` and `{other}` \
                                 on the results of one optimization barrier",
                            ),
                        }));
                    }
                }
                reductions
            }
            Propagation::Elementwise | Propagation::Other => {
                let mut merged = None::<PendingReduction>;
                for operand in operation.operand_values() {
                    let Some(reduction) = pending_reduction_of(operand?, &pending_reductions) else {
                        continue;
                    };
                    match &mut merged {
                        None => merged = Some(reduction),
                        Some(merged) if merged.kind == reduction.kind => merged.axes.extend(reduction.axes),
                        Some(merged) => {
                            return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                                operation: operation.name().to_string(),
                                reason: format!(
                                    "operands with pending reductions of different kinds `{}` and `{}`",
                                    merged.kind, reduction.kind,
                                ),
                            }));
                        }
                    }
                }
                if merged.is_none() {
                    continue;
                }
                vec![merged; operation.result_count()]
            }
        };
        let constrains = propagation != Propagation::ShardingConstraint;
        let mut anchor = operation;
        for (index, reduction) in result_reductions.into_iter().enumerate() {
            let Some(reduction) = reduction else {
                continue;
            };
            let result = operation.result(index)?.as_ref();
            // An effect token orders uses of tensor values and carries no sharding of its own.
            let Some(tensor_type) = result.r#type()?.cast::<TensorTypeRef>() else { continue };
            let mut dimensions = None;
            if propagation == Propagation::Elementwise {
                let shape = result.tensor_shape()?;
                let mut operand_dimensions = Vec::new();
                for operand in operation.operand_values() {
                    let operand = operand?;
                    if let Some(dimensions) = find(operand, &forward_dimensions)
                        && operand.tensor_shape()? == shape
                    {
                        operand_dimensions.push(dimensions);
                    }
                }
                dimensions = merged_dimensions(&operand_dimensions, mesh);
            } else if matches!(propagation, Propagation::While | Propagation::OptimizationBarrier) {
                dimensions = find(operation.operand_value(index)?, &forward_dimensions);
            } else if propagation == Propagation::ShardingConstraint {
                dimensions = constrained_dimensions(result)?;
            }
            let dimensions = dimensions.or_else(|| find(result, &backward_dimensions));
            if !constrains || outputs.contains(&result) || is_constrained(result)? {
                if let Some(dimensions) = dimensions {
                    forward_dimensions.push((result, dimensions));
                }
                pending_reductions.push((result, reduction));
                continue;
            }
            let sharding = reduction.sharding(mesh, tensor_type.rank(), dimensions.as_deref())?;
            let constrained =
                lowering_context.lower_pending_reduction_constraint(&sharding, result, block, location)?;
            let constraint = constrained.cast::<OperationResultRef>().unwrap().operation()?;
            // The constraint is appended to the block, and moved right after the operation that produces its operand
            // (or after the constraint of the previous result), so that it dominates every later use of that operand.
            anchor = unsafe { constraint.move_after(&anchor) };
            result.replace_uses_except(constrained, &[&anchor]);
            if let Some(dimensions) = dimensions {
                forward_dimensions.push((constrained, dimensions));
            }
            pending_reductions.push((constrained, reduction));
        }
    }
    // The outputs keep the dimension shardings that reach them forward, which include the placement along auto mesh
    // axes of the inputs that they are computed from elementwise (e.g., `x + x` of an input split along an auto axis).
    Ok(outputs.iter().map(|output| find(*output, &forward_dimensions)).collect())
}

// TODO(eaplatanios): Review this.
/// Constrains every lowered instruction output in `outputs` whose type in `output_types` is unreduced over a mesh axis
/// that no enclosing manual region binds to its tracked sharding, for `lower_instruction_operation`. An output that
/// a Shardy operation produces (e.g., the `sdy.sharding_constraint` of a placement transition or the
/// `sdy.manual_computation` of a `shard_map`) already declares its sharding and is left as it is. An output whose
/// dimension shardings `output_dimensions` gives (i.e., those that reach it forward through elementwise operations
/// from the inputs of the instruction, refer to [`constrain_unreduced_intermediates`]) keeps their placement along
/// auto mesh axes, which its type does not record (refer to [`dimensions_along_auto_axes`]).
pub(super) fn constrain_unreduced_outputs<'b, 'c: 'b, 't: 'c>(
    outputs: &mut [ValueRef<'b, 'c, 't>],
    output_types: &[Type],
    output_dimensions: &[Option<Rc<[ShardingDimension]>>],
    lowering_context: &ProgramLoweringContext,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<(), Error> {
    let bound_manual_axes = lowering_context.bound_manual_axes();
    let is_declared = |output: &ValueRef<'b, 'c, 't>| {
        output
            .cast::<OperationResultRef>()
            .and_then(|result| result.operation().ok())
            .is_some_and(|operation| operation.name().as_str().is_ok_and(|name| name.starts_with("sdy.")))
    };
    for (index, (output, output_type)) in outputs.iter_mut().zip(output_types).enumerate() {
        if let Type::Array(output_type) = output_type
            && let Some(sharding) = output_type.sharding()
            && sharding.unreduced_axes().iter().any(|axis| !bound_manual_axes.contains(axis))
            && !is_declared(output)
        {
            let dimensions = output_dimensions.get(index).cloned().flatten();
            let placed;
            let sharding = match dimensions_along_auto_axes(
                dimensions,
                sharding.dimensions_with_unconstrained_replicated().into(),
                sharding.mesh(),
            ) {
                dimensions if *dimensions != *sharding.dimensions_with_unconstrained_replicated() => {
                    placed = sharding
                        .with_dimensions(dimensions.to_vec())
                        .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                    &placed
                }
                _ => sharding,
            };
            *output = lowering_context.lower_pending_reduction_constraint(sharding, *output, block, location)?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_core::{
        AddOperation, ArrayOperations, ArrayType, BroadcastOperation, Compare, Condition, ConstrainSharding, Device,
        DeviceMesh, Dimension, Max, MeshAxis, Placeholder, ProgramBuilder, Reduce, ScanOptions, Shape, ValueProjection,
        WhileOptions, scan, r#while,
    };
    use ryft_mlir::Context as MlirContext;
    use ryft_pjrt::BufferType;

    use crate::lowering::tests::{
        UnreducedApplication, UnreducedInput, lower_unreduced_application, test_matrix_type, test_vector_type,
    };
    use crate::lowering::types::ArrayTypeLoweringExtension;
    use crate::tests::{cpu_client, logical_mesh_3x2x1, values_to_bytes};
    use crate::{
        ArrayTracer, CompileTracer, CompiledFunction, Constant, FromPjrt, Operation, Options, ProgramLoweringExtension,
        ProgramLoweringOptions, Session, TracedProgram, compile, compile_with_options, trace,
    };

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

    #[test]
    fn test_program_to_rendered_mlir_constrains_unreduced_outputs() {
        // Shardy rejects an operation whose result does not declare the unreduced axes of its inputs, so every output
        // that is unreduced over an axis that no manual region binds is constrained, also where its placement does not
        // change (e.g., a sum, and a broadcast that gives it a batch dimension).
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(4)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let output_type = ArrayType::new_static(DataType::F32, [2, 4])
            .with_sharding(Sharding::replicated(mesh, 2).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let left = builder.add_input(input_type.clone().into());
        let right = builder.add_input(input_type.clone().into());
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type.clone(), vec![1]), Vec::new(), vec![sum], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program
                .to_rendered_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_types(&vec![input_type.clone(), input_type])
                        .with_output_types(&vec![output_type])
                        .without_boundary_shardings()
                )
                .unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<2x4xf32> {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}], unreduced={"x"}> : tensor<4xf32>
                    %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [{}, {}], unreduced={"x"}> : tensor<2x4xf32>
                    return %3 : tensor<2x4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_instruction_constrains_unreduced_values_over_meshes_with_auto_axes() {
        // A value with a pending reduction over an explicit axis of a mesh that also has an auto axis is constrained
        // with closed dimensions, because Shardy otherwise reduces it along the open dimensions while it keeps the
        // pending reduction, which the readback then counts again (JAX 0.12.0.1 returns 12 for this `x + x`). The sum
        // of the contribution 1 on `e = 0` and 2 on `e = 1` doubled is 6 whatever the type of `a`.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        for axis_type in [MeshAxisType::Auto, MeshAxisType::Explicit] {
            let logical_mesh = LogicalMesh::new(vec![
                MeshAxis::new("e", 2, MeshAxisType::Explicit).unwrap(),
                MeshAxis::new("a", 2, axis_type).unwrap(),
            ])
            .unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
            let unreduced = Sharding::replicated(logical_mesh, 1).with_unreduced_axes(["e"]).unwrap();
            let input_type = test_vector_type(4).with_sharding(unreduced.without_auto_axes()).unwrap();
            let buffers = client
                .addressable_devices()
                .unwrap()
                .into_iter()
                .zip([1.0f32, 1.0, 2.0, 2.0])
                .map(|(device, value)| {
                    let bytes = values_to_bytes(&[value; 4]);
                    client.buffer(bytes.as_slice(), BufferType::F32, [4], None, device, None).unwrap()
                })
                .collect();
            let input = crate::Array::from_addressable_buffers(
                &session,
                input_type.clone(),
                mesh.clone(),
                Some(unreduced.clone()),
                buffers,
            )
            .unwrap();
            let function: CompiledFunction<ArrayType, ArrayType> =
                compile(|x: CompileTracer| x.clone() + x, input_type, &session, mesh).unwrap();
            let output = function.executable().execute(&session, input).unwrap();
            assert_eq!(output.sharding(), &unreduced);
            assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![6.0; 4], "{axis_type:?}");
        }
    }

    #[test]
    fn test_lower_instruction_keeps_auto_axis_splits_of_unreduced_values() {
        // An input that is split along the auto axis `d` and unreduced along the explicit axis `m` keeps its split
        // through `y = x + x; y + y`, which needs no communication. Its type cannot record the split, so the program
        // states it where the input enters, and the constraints that carry the pending reduction read it from there
        // rather than from the type, which would gather the value along `d`. The contributions 1 on `m = 0` and 2 on
        // `m = 1` sum to 3, and four times that is 12.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("d", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let unreduced = Sharding::replicated(logical_mesh.clone(), 1).with_unreduced_axes(["m"]).unwrap();
        let split = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["d"])])
            .unwrap()
            .with_unreduced_axes(["m"])
            .unwrap();
        let input_type = test_vector_type(4).with_sharding(unreduced).unwrap();
        let options = Options::new(mesh.clone())
            .with_input_shardings(vec![split.clone()])
            .with_output_shardings(vec![split.clone()]);
        let body = |x: CompileTracer| {
            let y = x.clone() + x;
            y.clone() + y
        };
        let lowered = crate::stage::<_, ArrayType, ArrayType>(body, input_type.clone(), options.clone())
            .unwrap()
            .into_inner()
            .lower(&session)
            .unwrap();
        assert_eq!(
            lowered.lowered_program().stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["d"=2, "m"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"d"}], unreduced={"m"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"d"}], unreduced={"m"}>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"d"}], unreduced={"m"}> : tensor<4xf32>
                    %1 = stablehlo.add %0, %0 : tensor<4xf32>
                    %2 = sdy.sharding_constraint %1 <@mesh, [{"d"}], unreduced={"m"}> : tensor<4xf32>
                    %3 = stablehlo.add %2, %2 : tensor<4xf32>
                    %4 = sdy.sharding_constraint %3 <@mesh, [{"d"}], unreduced={"m"}> : tensor<4xf32>
                    %5 = sdy.sharding_constraint %4 <@mesh, [{"d"}], unreduced={"m"}> : tensor<4xf32>
                    return %5 : tensor<4xf32>
                  }
                }
            "#},
        );

        // The serialized optimized HLO module names the opcode of every instruction, so it names no collective opcode.
        let optimized = lowered.compile(&session).unwrap().compiled_program().optimized_program().unwrap();
        let names_opcode =
            |opcode: &str| optimized.bytes().windows(opcode.len()).any(|window| window == opcode.as_bytes());
        assert!(names_opcode("parameter"));
        for opcode in ["all-gather", "all-reduce", "all-to-all", "collective-permute", "reduce-scatter"] {
            assert!(!names_opcode(opcode), "the optimized program contains a `{opcode}`");
        }

        // Devices are ordered as `(d, m)`, so each device holds the half of the input that `d` selects, and the
        // contribution that `m` selects.
        let buffers = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .zip([(1.0f32, 0), (2.0, 0), (1.0, 2), (2.0, 2)])
            .map(|(device, (contribution, offset))| {
                let values = [contribution * (offset + 1) as f32, contribution * (offset + 2) as f32];
                client
                    .buffer(values_to_bytes(&values).as_slice(), BufferType::F32, [2], None, device, None)
                    .unwrap()
            })
            .collect();
        let input = crate::Array::from_addressable_buffers(
            &session,
            input_type.clone(),
            mesh.clone(),
            Some(split.clone()),
            buffers,
        )
        .unwrap();
        let function: CompiledFunction<ArrayType, ArrayType> =
            compile_with_options(body, input_type.clone(), &session, options).unwrap();
        let output = function.executable().execute(&session, input.clone()).unwrap();
        assert_eq!(output.sharding(), &split);
        assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![12.0, 24.0, 36.0, 48.0]);

        // The split of the sum of an input split along `d` and one replicated along it is the split of the first,
        // whichever of the two is the left input, so neither order gathers anything.
        let replicated_buffers = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .zip([1.0f32, 2.0, 1.0, 2.0])
            .map(|(device, contribution)| {
                let values = [contribution, 2.0 * contribution, 3.0 * contribution, 4.0 * contribution];
                client
                    .buffer(values_to_bytes(&values).as_slice(), BufferType::F32, [4], None, device, None)
                    .unwrap()
            })
            .collect();
        let replicated = crate::Array::from_addressable_buffers(
            &session,
            input_type.clone(),
            mesh.clone(),
            Some(input_type.sharding().unwrap().clone()),
            replicated_buffers,
        )
        .unwrap();
        for split_first in [true, false] {
            let options = Options::new(mesh.clone())
                .with_input_shardings(vec![split.clone(), input_type.sharding().unwrap().clone()])
                .with_output_shardings(vec![split.clone()]);
            let body = move |(a, b): (CompileTracer, CompileTracer)| if split_first { a + b } else { b + a };
            let input_types = (input_type.clone(), input_type.clone());
            let lowered =
                crate::stage::<_, (ArrayType, ArrayType), ArrayType>(body, input_types.clone(), options.clone())
                    .unwrap()
                    .into_inner()
                    .lower(&session)
                    .unwrap();
            let optimized = lowered.compile(&session).unwrap().compiled_program().optimized_program().unwrap();
            let function: CompiledFunction<(ArrayType, ArrayType), ArrayType> =
                compile_with_options(body, input_types, &session, options).unwrap();
            let names_opcode =
                |opcode: &str| optimized.bytes().windows(opcode.len()).any(|window| window == opcode.as_bytes());
            for opcode in ["all-gather", "all-reduce", "all-to-all", "collective-permute", "reduce-scatter"] {
                assert!(!names_opcode(opcode), "the optimized program contains a `{opcode}` ({split_first})");
            }
            let output = function.executable().execute(&session, (input.clone(), replicated.clone())).unwrap();
            assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![6.0, 12.0, 18.0, 24.0]);
        }

        // An axis splits at most one dimension of a value, so the sum of a matrix split into rows along `d` and one
        // split into columns along `d` keeps the split along `d` of neither, whichever is the left input, where both
        // orders failed with "mesh axis name `d` appears more than once".
        let matrix_type = test_matrix_type(4, 4)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2).with_unreduced_axes(["m"]).unwrap())
            .unwrap();
        let split_matrix = |dimensions: Vec<ShardingDimension>| {
            Sharding::new(mesh.logical_mesh().clone(), dimensions).unwrap().with_unreduced_axes(["m"]).unwrap()
        };
        let rows = split_matrix(vec![ShardingDimension::sharded(["d"]), ShardingDimension::replicated()]);
        let columns = split_matrix(vec![ShardingDimension::replicated(), ShardingDimension::sharded(["d"])]);
        let matrix = |sharding: &Sharding| {
            let shards = client
                .addressable_devices()
                .unwrap()
                .into_iter()
                .enumerate()
                .map(|(index, device)| {
                    let (d, m) = (index / 2, index % 2);
                    let values = (0..16)
                        .filter(|element| match sharding == &rows {
                            true => element / 4 / 2 == d,
                            false => element % 4 / 2 == d,
                        })
                        .map(|element| element as f32 * (m + 1) as f32)
                        .collect::<Vec<_>>();
                    (Device::from_pjrt(device).unwrap().id(), Arc::<[u8]>::from(values_to_bytes(&values)))
                })
                .collect::<Vec<_>>();
            crate::Array::from_host_shards(&session, matrix_type.clone(), mesh.clone(), Some(sharding.clone()), shards)
                .unwrap()
        };
        for (left, right) in [(rows.clone(), columns.clone()), (columns.clone(), rows.clone())] {
            let options = Options::new(mesh.clone())
                .with_input_shardings(vec![left.clone(), right.clone()])
                .with_output_shardings(vec![rows.clone()]);
            let body = |(a, b): (CompileTracer, CompileTracer)| a + b;
            let input_types = (matrix_type.clone(), matrix_type.clone());
            let lowered =
                crate::stage::<_, (ArrayType, ArrayType), ArrayType>(body, input_types.clone(), options.clone())
                    .unwrap()
                    .into_inner()
                    .lower(&session)
                    .unwrap();
            let optimized = lowered.compile(&session).unwrap().compiled_program().optimized_program().unwrap();
            let collectives = ["all-gather", "all-reduce", "all-to-all", "collective-permute", "reduce-scatter"]
                .into_iter()
                .filter(|opcode| optimized.bytes().windows(opcode.len()).any(|window| window == opcode.as_bytes()))
                .collect::<Vec<_>>();
            // Both inputs are gathered along `d` before the sum, and the output is then sliced to its rows.
            assert_eq!(collectives, ["all-gather"]);
            let function: CompiledFunction<(ArrayType, ArrayType), ArrayType> =
                compile_with_options(body, input_types, &session, options).unwrap();
            let output = function.executable().execute(&session, (matrix(&left), matrix(&right))).unwrap();
            assert_eq!(
                output.copy_to_host().unwrap().elements::<f32>().unwrap(),
                (0..16).map(|element| 6.0 * element as f32).collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn test_lower_instruction_carries_pending_reductions_through_regions() {
        // A value with a pending sum along the explicit axis `m`, split along `d` or replicated, flows through every
        // construct that can carry it, on a mesh whose axis `d` is explicit, on one whose axis `d` is auto, and on one
        // whose axes are both auto (where the input override states the pending sum, which lowering completes where
        // the input enters). Device `(d, m)` holds the contribution `(m + 1) * [1, 2, 3, 4]` (of the half that `d`
        // selects, for a split value), so the value is `3 * [1, 2, 3, 4]`, and each construct multiplies it by a known
        // factor. Each cell is the right value or an error of the lowering that names the construct, never a wrong
        // value or a module that Shardy rejects.
        type Construct = Rc<dyn Fn(CompileTracer, CompileTracer, CompileTracer) -> CompileTracer>;
        fn project(value: crate::StatefulCompileTracer) -> CompileTracer {
            ValueProjection::<ArrayType>::into_projected(value).unwrap()
        }
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mut outcomes = Vec::new();
        for (mesh_label, d_type, m_type) in [
            ("explicit", MeshAxisType::Explicit, MeshAxisType::Explicit),
            ("mixed", MeshAxisType::Auto, MeshAxisType::Explicit),
            ("auto", MeshAxisType::Auto, MeshAxisType::Auto),
        ] {
            let logical_mesh =
                LogicalMesh::new(vec![MeshAxis::new("d", 2, d_type).unwrap(), MeshAxis::new("m", 2, m_type).unwrap()])
                    .unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
            let scalar_sharding = Sharding::replicated(logical_mesh.clone(), 0);
            let unreduced = Sharding::replicated(logical_mesh.clone(), 1).with_unreduced_axes(["m"]).unwrap();
            let split = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["d"])])
                .unwrap()
                .with_unreduced_axes(["m"])
                .unwrap();
            let scalar = |r#type: DataType, bytes: Vec<u8>| {
                let r#type = ArrayType::scalar(r#type).with_sharding(scalar_sharding.clone()).unwrap();
                crate::Array::from_host_buffer(&session, r#type, mesh.clone(), None, bytes).unwrap()
            };
            let predicate = scalar(DataType::Boolean, vec![1]);
            let count = scalar(DataType::F32, 3f32.to_ne_bytes().to_vec());
            for (value_label, runtime) in [("split", split.clone()), ("replicated", unreduced.clone())] {
                let input_type = test_vector_type(4).with_sharding(runtime.without_auto_axes()).unwrap();
                let shards = devices
                    .iter()
                    .enumerate()
                    .map(|(index, device)| {
                        let (d, m) = (index / 2, index % 2);
                        let values = [1f32, 2.0, 3.0, 4.0].map(|value| value * (m + 1) as f32);
                        let values = if runtime == split { values[2 * d..2 * d + 2].to_vec() } else { values.to_vec() };
                        (device.id(), Arc::<[u8]>::from(values_to_bytes(&values)))
                    })
                    .collect::<Vec<_>>();
                let input = crate::Array::from_host_shards(
                    &session,
                    input_type.clone(),
                    mesh.clone(),
                    Some(runtime.clone()),
                    shards,
                )
                .unwrap();
                let options = Options::new(mesh.clone()).with_input_shardings(vec![
                    runtime.clone(),
                    scalar_sharding.clone(),
                    scalar_sharding.clone(),
                ]);
                let doubled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
                    |x: CompileTracer| x.clone() + x,
                    test_vector_type(4),
                    &session,
                    Options::new(mesh.clone()).with_input_shardings(vec![runtime.clone()]),
                )
                .unwrap();
                let constructs: Vec<(&str, Construct, f32)> = vec![
                    ("straight-line", Rc::new(|x, _, _| x.clone() + x), 2.0),
                    (
                        "while once",
                        Rc::new(|x, p, _| {
                            let (_, output) = r#while(
                                |(p, _)| Ok(p.clone()),
                                |(p, x)| {
                                    let x = ValueProjection::<ArrayType>::into_projected(x)?;
                                    Ok((p.not_equal(&p)?, (x.clone() + x).into_value()))
                                },
                                (p.into_value(), x.into_value()),
                                WhileOptions::new(),
                            )
                            .unwrap();
                            project(output)
                        }),
                        2.0,
                    ),
                    (
                        "while three times",
                        Rc::new(|x, _, n| {
                            let (_, output) = r#while(
                                |(n, _)| {
                                    let n = ValueProjection::<ArrayType>::into_projected(n.clone())?;
                                    Ok((n.clone() - n.clone()).less_than(&n)?.into_value())
                                },
                                |(n, x)| {
                                    let n = ValueProjection::<ArrayType>::into_projected(n)?;
                                    let x = ValueProjection::<ArrayType>::into_projected(x)?;
                                    Ok(((n.clone() - n.clone() / n).into_value(), (x.clone() + x).into_value()))
                                },
                                (n.into_value(), x.into_value()),
                                WhileOptions::new(),
                            )
                            .unwrap();
                            project(output)
                        }),
                        8.0,
                    ),
                    (
                        "while with an identity body",
                        Rc::new(|x, p, _| {
                            let (_, output) = r#while(
                                |(p, _)| Ok(p.clone()),
                                |(p, x)| Ok((p.not_equal(&p)?, x)),
                                (p.into_value(), x.into_value()),
                                WhileOptions::new(),
                            )
                            .unwrap();
                            project(output)
                        }),
                        1.0,
                    ),
                    (
                        "nested while then an addition",
                        Rc::new(|x, p, _| {
                            let (_, output) = r#while(
                                |(p, _)| Ok(p.clone()),
                                |(p, x)| {
                                    let (_, inner) = r#while(
                                        |(p, _)| Ok(p.clone()),
                                        |(p, x)| {
                                            let x = ValueProjection::<ArrayType>::into_projected(x)?;
                                            Ok((p.not_equal(&p)?, (x.clone() + x).into_value()))
                                        },
                                        (p.clone(), x),
                                        WhileOptions::new(),
                                    )?;
                                    let inner = ValueProjection::<ArrayType>::into_projected(inner)?;
                                    Ok((p.not_equal(&p)?, (inner.clone() + inner).into_value()))
                                },
                                (p.into_value(), x.clone().into_value()),
                                WhileOptions::new(),
                            )
                            .unwrap();
                            project(output) + x
                        }),
                        5.0,
                    ),
                    (
                        "condition with computing branches",
                        Rc::new(|x, p, _| {
                            let output = p
                                .into_value()
                                .condition(
                                    |x| {
                                        let x = ValueProjection::<ArrayType>::into_projected(x)?;
                                        Ok((x.clone() + x).into_value())
                                    },
                                    |x| {
                                        let x = ValueProjection::<ArrayType>::into_projected(x)?;
                                        Ok((x.clone() + x.clone() + x).into_value())
                                    },
                                    x.into_value(),
                                )
                                .unwrap();
                            project(output)
                        }),
                        2.0,
                    ),
                    (
                        "condition with an identity branch",
                        Rc::new(|x, p, _| {
                            let output = p
                                .into_value()
                                .condition(
                                    |x| {
                                        let x = ValueProjection::<ArrayType>::into_projected(x)?;
                                        Ok((x.clone() + x).into_value())
                                    },
                                    |x| Ok(x),
                                    x.into_value(),
                                )
                                .unwrap();
                            project(output)
                        }),
                        2.0,
                    ),
                    ("call", Rc::new(move |x, _, _| doubled.call(x).unwrap()), 2.0),
                    (
                        "scan",
                        Rc::new(|x, _, _| {
                            let (output, ()) = scan(
                                |_, carry, ()| {
                                    let carry = ValueProjection::<ArrayType>::into_projected(carry)?;
                                    Ok(((carry.clone() + carry).into_value(), ()))
                                },
                                x.into_value(),
                                (),
                                ScanOptions::new().with_iteration_count(2usize),
                            )
                            .unwrap();
                            project(output)
                        }),
                        4.0,
                    ),
                ];
                for (construct_label, construct, factor) in constructs {
                    let input_types =
                        (test_vector_type(4), ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F32));
                    let body = {
                        let construct = Rc::clone(&construct);
                        move |(x, p, n): (CompileTracer, CompileTracer, CompileTracer)| construct(x, p, n)
                    };
                    let compiled = crate::stage::<_, (ArrayType, ArrayType, ArrayType), ArrayType>(
                        body.clone(),
                        input_types.clone(),
                        options.clone(),
                    )
                    .map_err(|error| error.to_string())
                    .and_then(|staged| staged.into_inner().lower(&session).map_err(|error| error.to_string()))
                    .and_then(|lowered| lowered.compile(&session).map_err(|error| error.to_string()));
                    let outcome = compiled.and_then(|compiled| {
                        let optimized = compiled.compiled_program().optimized_program().unwrap();
                        let collectives =
                            ["all-gather", "all-reduce", "all-to-all", "collective-permute", "reduce-scatter"]
                                .into_iter()
                                .filter(|opcode| {
                                    optimized.bytes().windows(opcode.len()).any(|window| window == opcode.as_bytes())
                                })
                                .collect::<Vec<_>>();
                        let function: CompiledFunction<(ArrayType, ArrayType, ArrayType), ArrayType> =
                            compile_with_options(body, input_types, &session, options.clone())
                                .map_err(|error| error.to_string())?;
                        let output = function
                            .executable()
                            .execute(&session, (input.clone(), predicate.clone(), count.clone()))
                            .map_err(|error| error.to_string())?;
                        let values = output.copy_to_host().unwrap().elements::<f32>().unwrap();
                        let expected = [3f32, 6.0, 9.0, 12.0].map(|value| value * factor).to_vec();
                        Ok(if values == expected {
                            format!("right, collectives {collectives:?}")
                        } else {
                            format!("WRONG {values:?} where {expected:?} is right")
                        })
                    });
                    outcomes.push(format!(
                        "{mesh_label}, {value_label}, {construct_label}: {}",
                        outcome.unwrap_or_else(|error| format!("error `{}`", error.lines().next().unwrap_or("")))
                    ));
                }
            }
        }
        // A `while`, a `condition`, and a `scan` cannot carry such a value on the mixed mesh, whose auto axis no type
        // records (refer to `validate_pending_reductions_across_regions`), so they are rejected with the ways forward,
        // where they returned double the value of a split value and Shardy rejected a replicated one. The gathers on the
        // mixed mesh replicate the split output that leaves the function, and the all-reduces on the auto mesh complete
        // the pending sum where the input enters.
        let mut expected = Vec::new();
        let constructs = [
            ("straight-line", None),
            ("while once", Some("while")),
            ("while three times", Some("while")),
            ("while with an identity body", Some("while")),
            ("nested while then an addition", Some("while")),
            ("condition with computing branches", Some("condition")),
            ("condition with an identity branch", Some("condition")),
            ("call", None),
            ("scan", Some("scan")),
        ];
        for mesh_label in ["explicit", "mixed", "auto"] {
            for value_label in ["split", "replicated"] {
                for (construct_label, region_operation) in constructs {
                    let outcome = match (mesh_label, value_label, region_operation) {
                        ("explicit", _, _) => "right, collectives []".to_string(),
                        ("auto", _, _) => "right, collectives [\"all-reduce\"]".to_string(),
                        (_, _, Some(operation)) => format!(
                            "error `cannot lower `{operation}`: a value of type `f32[4][sharding={{mesh<['d'=2:auto, \
                             'm'=2:explicit]>, [{{}}], unreduced={{'m'}}}}]` with a pending reduction cannot cross its \
                             regions on the mesh `['d'=2:auto, 'm'=2:explicit]`, which also has auto mesh axes, along \
                             which no type records where the value is; complete the reduction before the operation \
                             (e.g., with `reshard`), or use a mesh whose axes are all explicit`",
                        ),
                        (_, "split", None) => "right, collectives [\"all-gather\"]".to_string(),
                        (_, _, None) => "right, collectives []".to_string(),
                    };
                    expected.push(format!("{mesh_label}, {value_label}, {construct_label}: {outcome}"));
                }
            }
        }
        assert_eq!(outcomes, expected);
    }

    #[test]
    fn test_lower_instruction_constrains_unreduced_intermediates() {
        // The mean of a `bf16` vector converts it to `f32`, reduces it, divides it by the element count, and converts it
        // back.
        struct Mean;

        impl UnreducedApplication for Mean {
            fn apply<V: ArrayOperations>(&self, inputs: &[V]) -> Result<V, ProgramError> {
                inputs[0].reduce_mean(&[0])
            }
        }

        // The `bf16` bits of 1, 2, 3, and 4, and of -1, 0.5, 4, and 2.
        let input = UnreducedInput::new(
            DataType::BF16,
            &[4],
            Some(UnreducedKind::Sum),
            [&[0x3f80u16, 0x4000, 0x4040, 0x4080], &[0xbf80u16, 0x3f00, 0x4080, 0x4000]],
        );
        let lowering = lower_unreduced_application(&Mean, &[input]).unwrap();
        assert_eq!(lowering.outcomes, lowering.expected_outcomes);
        assert_eq!(
            lowering.module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={"x"}>}) -> (tensor<bf16> {sdy.sharding = #sdy.sharding<@mesh, [], unreduced={"x"}>}) {
                    %0 = stablehlo.convert %arg0 : (tensor<4xbf16>) -> tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}], unreduced={"x"}> : tensor<4xf32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [], unreduced={"x"}> : tensor<f32>
                    %cst_0 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
                    %4 = stablehlo.divide %3, %cst_0 : tensor<f32>
                    %5 = sdy.sharding_constraint %4 <@mesh, [], unreduced={"x"}> : tensor<f32>
                    %6 = stablehlo.convert %5 : (tensor<f32>) -> tensor<bf16>
                    %7 = sdy.sharding_constraint %6 <@mesh, [], unreduced={"x"}> : tensor<bf16>
                    return %7 : tensor<bf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_instruction_constrains_unreduced_intermediates_with_dimension_shardings() {
        // The intermediates of an instruction whose inputs carry a pending reduction and are split along another axis
        // keep that split, which flows forward through the elementwise comparisons and selections of a maximum, and
        // backward from the output through the division of a mean to the sum that it divides, so that none of them is
        // gathered.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let split = Sharding::new(mesh, vec![ShardingDimension::sharded(["y"]), ShardingDimension::Replicated])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let lower = |kind: UnreducedKind| {
            let sharding = split.clone().with_unreduced_kind(kind).unwrap();
            let input_type = ArrayType::new_static(DataType::F32, [8, 4]).with_sharding(sharding).unwrap();
            let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
                |(left, right): (ArrayTracer, ArrayTracer)| match kind {
                    UnreducedKind::Max => left.max(&right).unwrap(),
                    _ => left.reduce_mean(&[1]).unwrap() - right.reduce_mean(&[1]).unwrap(),
                },
                (input_type.clone(), input_type),
            )
            .unwrap();
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap()
        };
        assert_eq!(
            lower(UnreducedKind::Max),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=max{"x"}>}, %arg1: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=max{"x"}>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=max{"x"}>}) {
                    %0 = stablehlo.compare GT, %arg0, %arg1, FLOAT : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xi1>
                    %2 = stablehlo.select %1, %arg0, %arg1 : tensor<8x4xi1>, tensor<8x4xf32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xf32>
                    %4 = stablehlo.negate %arg0 : tensor<8x4xf32>
                    %5 = sdy.sharding_constraint %4 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xf32>
                    %6 = stablehlo.compare EQ, %arg0, %arg1, FLOAT : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
                    %7 = sdy.sharding_constraint %6 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xi1>
                    %8 = stablehlo.compare EQ, %arg0, %5, FLOAT : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
                    %9 = sdy.sharding_constraint %8 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xi1>
                    %10 = stablehlo.and %7, %9 : tensor<8x4xi1>
                    %11 = sdy.sharding_constraint %10 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xi1>
                    %12 = stablehlo.add %arg0, %arg1 : tensor<8x4xf32>
                    %13 = sdy.sharding_constraint %12 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xf32>
                    %14 = stablehlo.select %11, %13, %3 : tensor<8x4xi1>, tensor<8x4xf32>
                    %15 = sdy.sharding_constraint %14 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xf32>
                    %16 = stablehlo.compare NE, %arg0, %arg0, FLOAT : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
                    %17 = sdy.sharding_constraint %16 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xi1>
                    %18 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
                    %19 = sdy.sharding_constraint %18 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xi1>
                    %20 = stablehlo.select %19, %arg1, %15 : tensor<8x4xi1>, tensor<8x4xf32>
                    %21 = sdy.sharding_constraint %20 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xf32>
                    %22 = stablehlo.select %17, %arg0, %21 : tensor<8x4xi1>, tensor<8x4xf32>
                    %23 = sdy.sharding_constraint %22 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xf32>
                    return %23 : tensor<8x4xf32>
                  }
                }
            "#},
        );
        assert_eq!(
            lower(UnreducedKind::Sum),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>}, %arg1: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}], unreduced={"x"}>}) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"y"}], unreduced={"x"}> : tensor<8xf32>
                    %cst_0 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
                    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
                    %3 = stablehlo.divide %1, %2 : tensor<8xf32>
                    %4 = sdy.sharding_constraint %3 <@mesh, [{"y"}], unreduced={"x"}> : tensor<8xf32>
                    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %5 = stablehlo.reduce(%arg1 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
                    %6 = sdy.sharding_constraint %5 <@mesh, [{"y"}], unreduced={"x"}> : tensor<8xf32>
                    %cst_2 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
                    %7 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8xf32>
                    %8 = stablehlo.divide %6, %7 : tensor<8xf32>
                    %9 = sdy.sharding_constraint %8 <@mesh, [{"y"}], unreduced={"x"}> : tensor<8xf32>
                    %10 = stablehlo.subtract %4, %9 : tensor<8xf32>
                    %11 = sdy.sharding_constraint %10 <@mesh, [{"y"}], unreduced={"x"}> : tensor<8xf32>
                    return %11 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_validate_declared_pending_reduction() {
        // XLA aborts the process when it exports a sharding that introduces a pending maximum or minimum (it can only
        // introduce a pending sum), so lowering rejects such an output sharding or `constrain_sharding` before anything
        // is compiled. A value carries a pending reduction only where its type records it, which it never does over an
        // auto axis (an input that enters with one is reduced where it enters). Lowering alone is checked here, so
        // that a missing rejection cannot abort the test process.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mut outcomes = Vec::new();
        for site in ["output", "constraint"] {
            for axis_type in [MeshAxisType::Auto, MeshAxisType::Explicit] {
                let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
                let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
                let replicated = Sharding::replicated(logical_mesh, 1);
                let state = |name: &str| match name {
                    "none" => replicated.clone(),
                    "reduced" => replicated.clone().with_reduced_axes(["x"]).unwrap(),
                    _ => {
                        let kind = match name {
                            "sum" => UnreducedKind::Sum,
                            "max" => UnreducedKind::Max,
                            _ => UnreducedKind::Min,
                        };
                        replicated.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap()
                    }
                };
                for carried in ["none", "sum", "max", "min", "reduced"] {
                    let mut cells = Vec::new();
                    for declared in ["sum", "max", "min"] {
                        let input_type = test_vector_type(4);
                        let options = Options::new(mesh.clone()).with_input_shardings(vec![state(carried)]);
                        let declared_sharding = state(declared);
                        let staged: Result<crate::StagedFunction<ArrayType, ArrayType>, Error> = if site == "output" {
                            crate::stage(|x| x, input_type, options.with_output_shardings(vec![declared_sharding]))
                        } else {
                            crate::stage(
                                move |x: CompileTracer| x.constrain_sharding(&declared_sharding).unwrap(),
                                input_type,
                                options,
                            )
                        };
                        let lowered = staged.and_then(|staged| staged.into_inner().lower(&session).map(|_| ()));
                        cells.push(match lowered {
                            Ok(()) => format!("{declared}: ok"),
                            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, .. })) => {
                                let expected = if site == "output" { "output_shardings" } else { "constrain_sharding" };
                                assert_eq!(operation, expected);
                                format!("{declared}: rejected")
                            }
                            Err(_) => format!("{declared}: other error"),
                        });
                    }
                    outcomes.push(format!("{site} {axis_type:?} {carried}: {}", cells.join(", ")));
                }
            }
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                output Auto none: sum: ok, max: rejected, min: rejected
                output Auto sum: sum: ok, max: rejected, min: rejected
                output Auto max: sum: ok, max: rejected, min: rejected
                output Auto min: sum: ok, max: rejected, min: rejected
                output Auto reduced: sum: ok, max: rejected, min: rejected
                output Explicit none: sum: other error, max: other error, min: other error
                output Explicit sum: sum: ok, max: other error, min: other error
                output Explicit max: sum: other error, max: ok, min: other error
                output Explicit min: sum: other error, max: other error, min: ok
                output Explicit reduced: sum: other error, max: other error, min: other error
                constraint Auto none: sum: ok, max: rejected, min: rejected
                constraint Auto sum: sum: ok, max: rejected, min: rejected
                constraint Auto max: sum: ok, max: rejected, min: rejected
                constraint Auto min: sum: ok, max: rejected, min: rejected
                constraint Auto reduced: sum: ok, max: rejected, min: rejected
                constraint Explicit none: sum: ok, max: rejected, min: rejected
                constraint Explicit sum: sum: ok, max: other error, min: other error
                constraint Explicit max: sum: other error, max: ok, min: other error
                constraint Explicit min: sum: other error, max: other error, min: ok
                constraint Explicit reduced: sum: other error, max: other error, min: other error"},
        );

        // The diagnostic names the introduced state and its axes.
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let maximum = Sharding::replicated(logical_mesh, 1)
            .with_unreduced_axes(["x"])
            .unwrap()
            .with_unreduced_kind(UnreducedKind::Max)
            .unwrap();
        assert_eq!(
            validate_declared_pending_reduction(
                &maximum,
                Some(&maximum.without_auto_axes()),
                DataType::F32,
                &[],
                "output_shardings",
            )
            .unwrap_err()
            .to_string(),
            "cannot lower `output_shardings`: the sharding `{mesh<['x'=2:auto]>, [{}], unreduced={'x'}, \
             unreduced_kind=max}` declares a pending `max` reduction over mesh axes (i.e., `x`) along which the value \
             does not carry one, and XLA can only introduce a pending `sum` reduction (types never record reduction \
             state over auto mesh axes, so a pending `max` reduction over them is never carried)",
        );

        // A pending sum that a sharding introduces, and a pending reduction that an input whose type does not record
        // it brings into the program and that the program completes where it enters, are changes that a collective
        // makes, which must be one that the collective makes for the data type of the value (refer to
        // `collective_reduction`). One data type of every class of that classification, over the axis `x` of size two:
        // introducing a sum through an output override (auto) and a `constrain_sharding` (auto and explicit), and
        // completing each kind at the entry (auto). Before this check, the rejected introductions of `f6e2m3fn` and
        // `c64` values aborted the process (SIGABRT and SIGSEGV), the rejected completions of `u1`, `i1`, and
        // `f8e8m0fnu` values computed another number, those of `u2` and `f6e2m3fn` values failed to compile, and those
        // of maxima and minima of `c64` values failed when they ran. Maxima and minima of floating point values, whose
        // collective drops NaN values, are lowered.
        let mut outcomes = Vec::new();
        let lowered = |staged: Result<crate::StagedFunction<ArrayType, ArrayType>, Error>| match staged
            .and_then(|staged| staged.into_inner().lower(&session).map(|_| ()))
        {
            Ok(()) => "ok".to_string(),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, .. })) => {
                format!("rejected ({operation})")
            }
            Err(error) => format!("other error: {error}"),
        };
        for data_type in [
            DataType::Boolean,
            DataType::U1,
            DataType::I1,
            DataType::U2,
            DataType::I32,
            DataType::F6E2M3FN,
            DataType::F8E5M2,
            DataType::F8E8M0FNU,
            DataType::F32,
            DataType::C64,
        ] {
            let mut cells = Vec::new();
            let input_type = ArrayType::new(data_type, Shape::new(vec![Dimension::Static(4)]));
            for axis_type in [MeshAxisType::Auto, MeshAxisType::Explicit] {
                let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
                let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
                let replicated = Sharding::replicated(logical_mesh, 1);
                let pending = |kind: UnreducedKind| {
                    replicated.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap()
                };
                let options = Options::new(mesh.clone()).with_input_shardings(vec![replicated.clone()]);
                if axis_type == MeshAxisType::Auto {
                    let output = options.clone().with_output_shardings(vec![pending(UnreducedKind::Sum)]);
                    cells.push(format!("output {}", lowered(crate::stage(|x| x, input_type.clone(), output))));
                }
                let declared = pending(UnreducedKind::Sum);
                let constrained = crate::stage(
                    move |x: CompileTracer| x.constrain_sharding(&declared).unwrap(),
                    input_type.clone(),
                    options,
                );
                cells.push(format!("constraint {axis_type:?} {}", lowered(constrained)));
                if axis_type == MeshAxisType::Auto {
                    for kind in [UnreducedKind::Sum, UnreducedKind::Max, UnreducedKind::Min] {
                        let options = Options::new(mesh.clone())
                            .with_input_shardings(vec![pending(kind)])
                            .with_output_shardings(vec![replicated.clone()]);
                        cells.push(format!(
                            "entry {kind} {}",
                            lowered(crate::stage(|x| x, input_type.clone(), options))
                        ));
                    }
                }
            }
            outcomes.extend(cells.into_iter().map(|cell| format!("{data_type} {cell}")));
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                bool output ok
                bool constraint Auto ok
                bool entry sum ok
                bool entry max ok
                bool entry min ok
                bool constraint Explicit ok
                u1 output ok
                u1 constraint Auto ok
                u1 entry sum rejected (input_shardings)
                u1 entry max ok
                u1 entry min ok
                u1 constraint Explicit ok
                i1 output ok
                i1 constraint Auto ok
                i1 entry sum rejected (input_shardings)
                i1 entry max rejected (input_shardings)
                i1 entry min rejected (input_shardings)
                i1 constraint Explicit ok
                u2 output ok
                u2 constraint Auto ok
                u2 entry sum rejected (input_shardings)
                u2 entry max rejected (input_shardings)
                u2 entry min rejected (input_shardings)
                u2 constraint Explicit ok
                i32 output ok
                i32 constraint Auto ok
                i32 entry sum ok
                i32 entry max ok
                i32 entry min ok
                i32 constraint Explicit ok
                f6e2m3fn output rejected (output_shardings)
                f6e2m3fn constraint Auto rejected (constrain_sharding)
                f6e2m3fn entry sum rejected (input_shardings)
                f6e2m3fn entry max rejected (input_shardings)
                f6e2m3fn entry min rejected (input_shardings)
                f6e2m3fn constraint Explicit rejected (constrain_sharding)
                f8e5m2 output ok
                f8e5m2 constraint Auto ok
                f8e5m2 entry sum ok
                f8e5m2 entry max ok
                f8e5m2 entry min ok
                f8e5m2 constraint Explicit ok
                f8e8m0fnu output rejected (output_shardings)
                f8e8m0fnu constraint Auto rejected (constrain_sharding)
                f8e8m0fnu entry sum rejected (input_shardings)
                f8e8m0fnu entry max ok
                f8e8m0fnu entry min ok
                f8e8m0fnu constraint Explicit rejected (constrain_sharding)
                f32 output ok
                f32 constraint Auto ok
                f32 entry sum ok
                f32 entry max ok
                f32 entry min ok
                f32 constraint Explicit ok
                c64 output rejected (output_shardings)
                c64 constraint Auto rejected (constrain_sharding)
                c64 entry sum ok
                c64 entry max rejected (input_shardings)
                c64 entry min rejected (input_shardings)
                c64 constraint Explicit rejected (constrain_sharding)"},
        );

        // The diagnostics name the construct, the sharding, the change, the data type, the axes, the reason, and what
        // to do instead, once for each class of rejection.
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
        let pending = |kind: UnreducedKind| {
            replicated.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap()
        };
        let axes = [String::from("x")];
        let rejection = |data_type: DataType, change: ReductionChange| {
            let sharding = match change {
                ReductionChange::Complete(kind) | ReductionChange::Introduce(kind) => pending(kind),
            };
            validate_pending_reduction_change(data_type, change, &sharding, &axes, "input_shardings")
                .unwrap_err()
                .to_string()
        };
        let message = |sharding: &str, verb: &str, kind: &str, data_type: &str, reason: &str| {
            // Nothing can split a data type without an additive identity into summands, so only another data type
            // helps there.
            let remedy = match (verb, data_type) {
                ("introduce", "f8e8m0fnu") => "use a data type that has an additive identity",
                ("complete", _) => {
                    "complete the reduction with a placement of the concrete array instead (e.g., \
                     `Array::to_placement`), which is exact where one process owns every device of the destination \
                     and a device of every contribution of the array that has elements, or use another data type"
                }
                _ => {
                    "introduce the reduction with a placement of the concrete array instead (e.g., \
                     `Array::to_placement`), which writes additive identities exactly on every mesh, or use another \
                     data type"
                }
            };
            format!(
                "cannot lower `input_shardings`: the sharding `{sharding}` would {verb} a pending `{kind}` reduction \
                 of `{data_type}` values over mesh axes (i.e., `x`), which XLA does not do as the data type means it, \
                 because {reason}; {remedy}",
            )
        };
        let sum = "{mesh<['x'=2:auto]>, [{}], unreduced={'x'}}";
        let max = "{mesh<['x'=2:auto]>, [{}], unreduced={'x'}, unreduced_kind=max}";
        assert_eq!(
            [
                rejection(DataType::U1, ReductionChange::Complete(UnreducedKind::Sum)),
                rejection(DataType::U2, ReductionChange::Complete(UnreducedKind::Sum)),
                rejection(DataType::C64, ReductionChange::Complete(UnreducedKind::Max)),
                rejection(DataType::F6E2M3FN, ReductionChange::Introduce(UnreducedKind::Sum)),
                rejection(DataType::F8E8M0FNU, ReductionChange::Introduce(UnreducedKind::Sum)),
            ],
            [
                message(
                    sum,
                    "complete",
                    "sum",
                    "u1",
                    "XLA adds one-bit integers as predicates (i.e., with a logical or), while their sum wraps around",
                ),
                message(sum, "complete", "sum", "u2", "XLA does not compile reductions of two-bit integers"),
                message(max, "complete", "max", "c64", "XLA has no maximum or minimum reduction of complex values"),
                message(
                    sum,
                    "introduce",
                    "sum",
                    "f6e2m3fn",
                    "XLA aborts the process when it introduces a pending sum of six-bit floating point values",
                ),
                message(
                    sum,
                    "introduce",
                    "sum",
                    "f8e8m0fnu",
                    "the data type has no additive identity to split a value into summands",
                ),
            ],
        );

        // A completion of a pending sum along some mesh axes that keeps it along another is checked as a whole (refer
        // to `partial_completion_rejection`, which the placement planner reads as well), because the partial sums are
        // rounded to the data type before the remaining reduction. On a 2x2 mesh, an identity whose input override is
        // pending along the explicit axis `x` and the auto axis `y` and whose output override is pending along `x`
        // completes the sum along `y` where the input enters and keeps it along `x`. For the `f8e4m3fn`
        // contributions `448`, `448`, `-448`, and `-448`, which sum to zero, that program returned NaN before this
        // check. A completion along both axes at once (both auto) is lowered.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let lowered = |staged: Result<crate::StagedFunction<ArrayType, ArrayType>, Error>| match staged
            .and_then(|staged| staged.into_inner().lower(&session).map(|_| ()))
        {
            Ok(()) => "ok".to_string(),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, .. })) => {
                format!("rejected ({operation})")
            }
            Err(error) => format!("other error: {error}"),
        };
        let mesh = |x_type: MeshAxisType| {
            let x = MeshAxis::new("x", 2, x_type).unwrap();
            let logical_mesh = LogicalMesh::new(vec![x, MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap()]).unwrap();
            (DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap(), Sharding::replicated(logical_mesh, 1))
        };
        let mut outcomes = Vec::new();
        for data_type in [DataType::F8E4M3FN, DataType::BF16, DataType::F32, DataType::I32] {
            let input_type = ArrayType::new(data_type, Shape::new(vec![Dimension::Static(4)]));
            let (partial_mesh, replicated) = mesh(MeshAxisType::Explicit);
            let options = Options::new(partial_mesh)
                .with_input_shardings(vec![replicated.clone().with_unreduced_axes(["x", "y"]).unwrap()])
                .with_output_shardings(vec![replicated.with_unreduced_axes(["x"]).unwrap()]);
            let partial = lowered(crate::stage(|x| x, input_type.clone(), options));
            let (full_mesh, replicated) = mesh(MeshAxisType::Auto);
            let options = Options::new(full_mesh)
                .with_input_shardings(vec![replicated.clone().with_unreduced_axes(["x", "y"]).unwrap()])
                .with_output_shardings(vec![replicated]);
            let full = lowered(crate::stage(|x| x, input_type, options));
            outcomes.push(format!("{data_type} partial {partial}, full {full}"));
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                f8e4m3fn partial rejected (input_shardings), full ok
                bf16 partial rejected (input_shardings), full ok
                f32 partial ok, full ok
                i32 partial ok, full ok"},
        );

        // The same transition of a user-stated sharding (e.g., a `constrain_sharding`) is rejected as well, with a
        // diagnostic that names the sharding, the axes, and the accumulation type.
        let (_, replicated) = mesh(MeshAxisType::Explicit);
        let carried = replicated.clone().with_unreduced_axes(["x", "y"]).unwrap();
        let declared = replicated.with_unreduced_axes(["x"]).unwrap();
        let validated = |data_type: DataType| {
            validate_declared_pending_reduction(&declared, Some(&carried), data_type, &[], "constrain_sharding")
                .map_err(|error| error.to_string())
        };
        assert_eq!(validated(DataType::F32), Ok(()));
        assert_eq!(
            validated(DataType::F8E4M3FN),
            Err(format!(
                "cannot lower `constrain_sharding`: the sharding `{declared}` cannot complete a pending `sum` of \
                 `f8e4m3fn` values along mesh axis `y` and keep it along mesh axis `x` on the devices, because the \
                 partial sums would be rounded to `f8e4m3fn`, while the sum accumulates in `f16`; complete the \
                 reduction along all of these mesh axes at once, or use a data type whose sums accumulate in itself \
                 (e.g., `f32`)",
            )),
        );
    }
}
