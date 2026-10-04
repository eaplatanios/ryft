use std::fmt::Display;

use crate::arrays::{
    ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayType, DataType, Dimension, LogicalMesh, Shape,
    Sharding,
};
use crate::axes::{AxisError, NamedAxes, NamedAxis};
use crate::batching::{BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation, impl_nullary_transposable_operation};
use crate::operations::collectives::check_manual_mesh_axis;
use crate::operations::constants::iota::IotaOperation;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartiallyEvaluatableOperation,
};
use crate::programs::{Operation, OperationFormatter, ProgramError, RegionInterface, TypeError};

/// Canonical operation name for [`AxisIndexOperation`].
pub const AXIS_INDEX_OPERATION_NAME: &str = "axis_index";

/// Nullary primitive [`Operation`] that produces the current batch item's or device shard's index
/// along a [`NamedAxis`] as a scalar [`DataType::U64`] value. This is the Ryft analogue of JAX's
/// [`jax.lax.axis_index`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.axis_index.html).
/// [`AxisIndex::axis_index`] stages this operation uniformly for every axis kind and resolution depends on the
/// enclosing binder. A _batched_ axis is consumed by this operation's staged batching rule at the `batch` level that
/// binds it, which materializes the per-item index as an [`iota`](crate::Iota) over the known batch size. This means
/// that an [`AxisIndexOperation`] for a batched axis never survives into a staged body. A _device mesh_ axis has no
/// such trace-time binder. Its per-device coordinate is known only at execution time, and so the operation stays in the
/// staged body and lowers, for example, inside a `shard_map` operation manual region to `partition_id`-based coordinate
/// arithmetic. Only mesh uses therefore reach interpretation, which is why this operation is _not_ eagerly
/// interpretable and, having no inputs, is [partially evaluated](PartiallyEvaluatableOperation) by residualizing
/// rather than folding (that is because folding a nullary operation would result in trying to interpret it).
///
/// # Examples
///
/// The following batches a function over a named `items` axis and returns each item's index along that axis:
///
/// ```rust
/// # use ryft_core::{Array, AxisIndex, BatchAxis, BatchAxisSpecification, batch};
/// #
/// let indices: Array = batch(
///     |item| item.context().axis_index("items"),
///     Array::vector(vec![10.0, 20.0, 30.0]).unwrap(),
///     BatchAxis::new(0),
///     BatchAxis::new(0),
///     BatchAxisSpecification::named("items"),
/// )
/// .unwrap();
/// assert_eq!(indices.to_f64s(), vec![0.0, 1.0, 2.0]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AxisIndexOperation {
    /// Name of the device mesh axis whose per-shard index this [`AxisIndexOperation`] produces.
    axis_name: String,

    /// [`LogicalMesh`] for a checked device coordinate, or `None` when manual variation is not tracked.
    mesh: Option<LogicalMesh>,
}

impl AxisIndexOperation {
    /// Creates a new [`AxisIndexOperation`] without tracked mesh variation. Use [`Self::with_mesh`] when constructing
    /// a checked device coordinate directly; [`AxisIndex::axis_index`] supplies that metadata automatically.
    #[inline]
    pub fn new(axis_name: String) -> Self {
        Self { axis_name, mesh: None }
    }

    /// Returns this [`AxisIndexOperation`] with the provided logical mesh. Its output varies over [`Self::axis_name`]
    /// on that mesh, which must name a manual axis. Validation occurs during type inference.
    #[inline]
    pub fn with_mesh(mut self, mesh: LogicalMesh) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Returns the mesh axis name referenced by this [`AxisIndexOperation`].
    #[inline]
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }

    /// Returns the logical mesh for a checked device coordinate, or `None` when manual variation is not tracked.
    #[inline]
    pub fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }
}

impl Display for AxisIndexOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for AxisIndexOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        AXIS_INDEX_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        let mut output = ArrayType::scalar(DataType::U64);
        if let Some(mesh) = &self.mesh {
            check_manual_mesh_axis(AXIS_INDEX_OPERATION_NAME, &self.axis_name, None, mesh)?;
            output = output
                .with_sharding(
                    Sharding::replicated(mesh.clone(), 0)
                        .with_varying_manual_axes([self.axis_name.clone()])
                        .map_err(|error| TypeError::invalid(error.to_string()))?,
                )
                .map_err(|error| TypeError::invalid(error.to_string()))?;
        }
        Ok(vec![output])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, AXIS_INDEX_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
            if let Some(mesh) = &self.mesh {
                operation.field("mesh", mesh)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for AxisIndexOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // A mesh axis index is a per-device coordinate that only exists during sharded execution. There is no eager
        // value to produce. It is lowered inside a `shard_map` manual region and never interpreted directly.
        check_count!("input", inputs, 0, ProgramError);
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{}` for the device mesh axis `{}` has no eager value; \
                it is only defined inside a `shard_map` manual region",
                AXIS_INDEX_OPERATION_NAME, self.axis_name,
            ),
        })
    }
}

impl<C: Context<Type = ArrayType, Operation: From<AxisIndexOperation>>> PartiallyEvaluatableOperation<C>
    for AxisIndexOperation
{
    #[inline]
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        _driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Partial evaluation always residualizes an `AxisIndexOperation`. Its value depends on the executing device,
        // and so it is never a foldable constant even though it has no (known) inputs.
        context.residualize(self.clone(), Vec::new(), inputs)
    }
}

impl<
    C: Context<Type = ArrayType, Operation: From<IotaOperation<ArrayType>> + From<AxisIndexOperation>>,
    P: ArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for AxisIndexOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        _inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        if self.mesh.is_none() && context.axis_name() == Some(self.axis_name.as_str()) {
            // This level binds the axis. The per-item index is the length-`size` `iota(0)`, bound into the parent and
            // mapped on this level's batch axis (position 0). The mapped packed `[size]` dimension is then stripped
            // back to the per-item scalar `u64`.
            let size = P::axis_size(context)?;
            let r#type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(size)]));
            let operation = IotaOperation::new(r#type.clone(), 0)?;
            let mut index = context.parent().bind(operation, Vec::new(), &[])?;
            check_count!("output", index, 1, ProgramError);
            Ok(vec![ArrayBatch::new(index.remove(0), Some(0))?].into())
        } else {
            // The axis is bound by an outer `batch` level or a device mesh. Re-bind into the parent, which repeats the
            // resolution and present the forwarded index as replicated across this level.
            let operation = self.clone();
            let mut index = context.parent().bind(operation, Vec::new(), &[])?;
            check_count!("output", index, 1, ProgramError);
            Ok(vec![ArrayBatch::replicated(index.remove(0))].into())
        }
    }
}

impl_non_differentiable_operation!(AxisIndexOperation);
impl_nullary_transposable_operation!(AxisIndexOperation);

/// Capability to read the index of the current element along a named axis. This is the value-producing counterpart of
/// [`NamedAxes`]. `NamedAxes` answers whether a name is in scope, while [`AxisIndex`] reads out the position along it.
/// Resolution is validated against the active [`NamedAxes`] environment.
pub trait AxisIndex: Context {
    /// Returns a [`DataType::U64`] scalar giving the current element's position along `name`. What that position counts
    /// follows the kind of binder that introduced the axis (refer to the documentation of [`NamedAxis`] for more
    /// information): a batching axis of size `N` yields the current element's position in `0..N`, and a device mesh
    /// axis yields the current shard's coordinate along that mesh axis. `U64` matches the `usize` axis sizes the
    /// indices are drawn from and cannot be negative. A name that no enclosing binder binds will result in
    /// [`AxisError::UnboundAxisName`].
    fn axis_index(&self, name: &str) -> Result<Self::Value, ProgramError>;
}

impl<C: Context<Operation: From<AxisIndexOperation>> + NamedAxes> AxisIndex for C {
    fn axis_index(&self, name: &str) -> Result<Self::Value, ProgramError> {
        // Every context reads an axis index the same way. It validates `name` against the active `NamedAxes`
        // environment and then binds a nullary `AxisIndexOperation`, so the caller needs no knowledge of whether `name`
        // is a batching or mesh axis. That operation carries the per-axis-kind resolution as it flows outward: the
        // batching level that bound `name` consumes it (its batching rule materializes the per-element index), an inner
        // batching level re-binds it into its parent, and a mesh axis survives into the base program to lower during
        // sharded execution (refer to the documentation of `AxisIndexOperation`). Because resolution happens as the
        // operation is consumed, a batched axis reached across a non-batching wrapper that *interprets* a nested
        // program (e.g., an outer batch addressed from inside a `jvp` trace, whose primal program is spliced by
        // interpretation) is not supported. The operation is interpreted before any batching rule can consume it
        // and reports `ProgramError::UnsupportedOperation`. Mesh axes are unaffected, as they are meant to survive
        // interpretation.
        let operation = match self.named_axis(name) {
            Some(NamedAxis::Mesh { mesh, .. }) => AxisIndexOperation::new(name.to_string()).with_mesh(mesh),
            Some(NamedAxis::Batched { .. }) => AxisIndexOperation::new(name.to_string()),
            None => return Err(AxisError::UnboundAxisName { name: name.to_string() }.into()),
        };
        let mut outputs = self.bind(operation, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType, Dimension, MeshAxis, MeshAxisType, Shape};
    use crate::batching::{Batch, BatchAxis, BatchAxisSpecification, BatchingError, batch};
    use crate::contexts::EagerContext;
    use crate::programs::Typed;
    use crate::tracing::DomainTracingContext;

    use super::*;

    #[test]
    fn test_axis_index_operation() {
        let operation = AxisIndexOperation::new("devices".to_string());
        assert_eq!(operation.axis_name(), "devices");
        assert_eq!(operation.mesh(), None);
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![ArrayType::scalar(DataType::U64)]));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let operation = operation.with_mesh(mesh.clone());
        assert_eq!(operation.mesh(), Some(&mesh));
        assert_eq!(
            operation.infer_output_types(&[], &[]),
            Ok(vec![
                ArrayType::scalar(DataType::U64)
                    .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["devices"]).unwrap())
                    .unwrap()
            ]),
        );
        let explicit = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        assert_eq!(
            operation.with_mesh(explicit).infer_output_types(&[], &[]),
            Err(TypeError::invalid("`axis_index` mesh axis `devices` must be manual")),
        );
    }

    #[test]
    fn test_axis_index_stages_a_nullary_operation_for_a_bound_axis() {
        // Validate `name` against the seeded `NamedAxes` environment and stage a nullary `AxisIndexOperation`
        // producing a scalar `u64`, regardless of whether the axis is batch- or mesh-bound.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("device", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let (output_type, program) =
            DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::trace_with_named_axes(
                |input| input.context().axis_index("device"),
                ArrayType::scalar(DataType::F64),
                vec![("device".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 4 })],
            )
            .unwrap();
        assert_eq!(
            output_type,
            ArrayType::scalar(DataType::U64)
                .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["device"]).unwrap())
                .unwrap()
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:u64[][sharding={mesh<['device'=4:manual]>, [], varying_manual={'device'}}] = \
                    axis_index [axis_name=\"device\", mesh=['device'=4:manual]]
                in (%1)"},
        );
    }

    #[test]
    fn test_axis_index_rejects_an_unbound_axis() {
        // A name that no enclosing binder binds fails fast at the reader, before any operation is staged, surfacing
        // `AxisError::UnboundAxisName` as a `ProgramError::Axis` error.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("device", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let error = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::trace_with_named_axes(
            |input| input.context().axis_index("missing"),
            ArrayType::scalar(DataType::F64),
            vec![("device".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 4 })],
        )
        .unwrap_err();
        assert_eq!(error, ProgramError::Axis(AxisError::UnboundAxisName { name: "missing".to_string() }));
    }

    #[test]
    fn test_batch_axis_index_produces_per_item_indices() {
        // `axis_index("i")` gives each batch item its own position along the mapped axis `"i"` (size 3), so the
        // batched result is the `u64` index vector `[0, 1, 2]` regardless of the input values.
        let output: Array = batch(
            |item| item.context().axis_index("i"),
            Array::vector(vec![10.0, 20.0, 30.0]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("i"),
        )
        .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(3)])));
        assert_eq!(output.to_f64s(), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_nested_batch_axis_index_forwards_outer_axis_through_inner_level() {
        // Outer `batch` over axis 0 (size 2, named "o") of a [2, 3] matrix; inner `batch` over axis 0 (size 3, named
        // "i") of each row. The inner body asks for `axis_index("o")`, which the inner level does not bind, so it is
        // forwarded to the outer level and re-wrapped as replicated across the inner axis (the outer index does not
        // vary over inner items). The inner output is therefore declared replicated, and the outer level stacks the
        // per-row outer index, giving the `u64` vector `[0, 1]`.
        let x = Array::matrix(2, 3, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |row| {
                    let context = row.context().clone();
                    Ok(Batch::batch(
                        &context,
                        |scalar| scalar.context().axis_index("o"),
                        row,
                        BatchAxis::new(0),
                        BatchAxis::replicated(),
                        BatchAxisSpecification::named("i"),
                    )?)
                },
                x,
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("o"),
            )
            .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2)])));
        assert_eq!(output.to_f64s(), vec![0.0, 1.0]);
    }

    #[test]
    fn test_batch_axis_index_rejects_unbound_axis() {
        // `axis_index` over a name no enclosing batch binds fails fast, mirroring the collective readers.
        let result: Result<Array, BatchingError> = EagerContext::<Array, ArrayOperation<Array>>::new().batch(
            |item| item.context().axis_index("j"),
            Array::vector(vec![10.0, 20.0, 30.0]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("i"),
        );
        assert_eq!(result.unwrap_err(), BatchingError::Axis(AxisError::UnboundAxisName { name: "j".to_string() }));
    }
}
