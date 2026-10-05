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
use crate::operations::collectives::validate_manual_mesh_axis;
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
            validate_manual_mesh_axis(AXIS_INDEX_OPERATION_NAME, &self.axis_name, None, mesh)?;
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

    use crate::arrays::{Array, ArrayOperation, MeshAxis, MeshAxisType};
    use crate::batching::{BatchAxis, BatchAxisSpecification, batch};
    use crate::contexts::EagerContext;
    use crate::macros::check_operation_type_inference;
    use crate::parameters::Placeholder;
    use crate::partial::PartialEvaluationOutput;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, Typed};
    use crate::tracing::{DomainTracingContext, TracingContext};

    use super::*;

    #[test]
    fn test_axis_index() {
        let operation = AxisIndexOperation::new("devices".to_string());
        assert_eq!(operation.name(), AXIS_INDEX_OPERATION_NAME);
        assert_eq!(operation.axis_name(), "devices");
        assert_eq!(operation.mesh(), None);
        assert_eq!(operation.to_string(), "axis_index [axis_name=\"devices\"]");
    }

    #[test]
    fn test_axis_index_with_mesh() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let operation = AxisIndexOperation::new("devices".to_string()).with_mesh(mesh.clone());
        assert_eq!(operation.axis_name(), "devices");
        assert_eq!(operation.mesh(), Some(&mesh));
        assert_eq!(operation.to_string(), "axis_index [axis_name=\"devices\", mesh=['devices'=2:manual]]");
    }

    #[test]
    fn test_axis_index_type_inference() {
        // Without a mesh, the index is a scalar `u64` with no tracked manual variation.
        check_operation_type_inference!(
            operation = AxisIndexOperation::new("devices".to_string()),
            cases = [{
                input_types = [],
                output_types = [ArrayType::scalar(DataType::U64)],
            }, {
                input_types = [ArrayType::scalar(DataType::U64)],
                error = "expected 0 inputs but got 1",
            }],
        );

        // With a mesh, the index is a device coordinate that varies over its axis, which must be manual.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = AxisIndexOperation::new("devices".to_string()).with_mesh(mesh.clone()),
            cases = [{
                input_types = [],
                output_types = [
                    ArrayType::scalar(DataType::U64)
                        .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["devices"]).unwrap())
                        .unwrap(),
                ],
            }],
        );
        let explicit_mesh =
            LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = AxisIndexOperation::new("devices".to_string()).with_mesh(explicit_mesh),
            cases = [{
                input_types = [],
                error = "`axis_index` mesh axis `devices` must be manual",
            }],
        );
    }

    #[test]
    fn test_axis_index_interpretation() {
        // A device mesh coordinate exists only during sharded execution, so there is no eager value to produce.
        assert_eq!(
            AxisIndexOperation::new("devices".to_string()).interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "`axis_index` for the device mesh axis `devices` has no eager value; it is only defined \
                    inside a `shard_map` manual region"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_axis_index_partial_evaluation() {
        // The operation has no inputs, so all of its inputs are trivially known, but its value depends on the executing
        // device. Partial evaluation therefore residualizes it rather than folding it through eager interpretation.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let output = builder
            .add_instruction(
                AxisIndexOperation::new("devices".to_string()).with_mesh(mesh),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap()[0];
        let program = builder.build::<(), Array>(vec![output], (), Placeholder).unwrap();
        let evaluation = program.to_flat_program().partially_evaluate(&[]).unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda  .
                let %0:u64[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}] = \
                        axis_index [axis_name=\"devices\", mesh=['devices'=2:manual]]
                in (%0)"},
        );
    }

    #[test]
    fn test_axis_index_batching() {
        // `axis_index("i")` gives each batch item its own position along the mapped axis `i` (size 3), so the
        // batched result is the `u64` index vector `[0, 1, 2]` regardless of the input values.
        let output: Array = batch(
            |item| item.context().axis_index("i"),
            Array::vector(vec![10.0, 20.0, 30.0]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("i"),
        )
        .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new_static(DataType::U64, [3]));
        assert_eq!(output, Array::vector(vec![0u64, 1, 2]).unwrap());
    }

    #[test]
    fn test_axis_index_batching_nested_axes() {
        // The outer `batch` maps axis 0 (size 2, named `o`) of a `[2, 3]` matrix and the inner `batch` maps axis 0
        // (size 3, named `i`) of each row. The inner body asks for `axis_index("o")`, which the inner level does not
        // bind, so that level forwards the operation to the outer level and presents the result as replicated across
        // the inner batch items (the outer index does not vary over them). The inner output is therefore declared
        // replicated, and the outer level stacks the per-row outer index into the `u64` vector `[0, 1]`.
        let input = Array::matrix(2, 3, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
        let output: Array = batch(
            |row| {
                batch(
                    |scalar| scalar.context().axis_index("o"),
                    row,
                    BatchAxis::new(0),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("i"),
                )
                .map_err(Into::into)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("o"),
        )
        .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new_static(DataType::U64, [2]));
        assert_eq!(output, Array::vector(vec![0u64, 1]).unwrap());
    }

    #[test]
    fn test_axis_index_batching_mesh_axis() {
        // An index that carries a mesh is a device coordinate, so no `batch` level consumes it, not even one that binds
        // the same name. The level forwards it to its parent, where it survives into the staged body, and presents the
        // forwarded coordinate as replicated across its batch items.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 3).with_axis_name("devices".to_string());
        let outputs = AxisIndexOperation::new("devices".to_string())
            .with_mesh(mesh)
            .batch(&context, &EmptyRegionDriver, &[])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        let builder = trace.builder().borrow().clone();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![outputs[0].value().atom_id().unwrap()], Vec::new(), vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:u64[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}] = \
                        axis_index [axis_name=\"devices\", mesh=['devices'=2:manual]]
                in (%0)"},
        );
    }

    #[test]
    fn test_axis_index_axis_index() {
        // A mesh-bound name stages an `AxisIndexOperation` that carries the binding's mesh, so the scalar `u64` output
        // varies over the manual axis.
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
                .unwrap(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:u64[][sharding={mesh<['device'=4:manual]>, [], varying_manual={'device'}}] = \
                    axis_index [axis_name=\"device\", mesh=['device'=4:manual]]
                in (%1)"},
        );

        // A batch-bound name stages an `AxisIndexOperation` without a mesh, which the `batch` level that binds the name
        // later consumes through its batching rule.
        let (output_type, program) =
            DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::trace_with_named_axes(
                |input| input.context().axis_index("items"),
                ArrayType::scalar(DataType::F64),
                vec![("items".to_string(), NamedAxis::Batched { size: Some(3) })],
            )
            .unwrap();
        assert_eq!(output_type, ArrayType::scalar(DataType::U64));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:u64[] = axis_index [axis_name=\"items\"]
                in (%1)"},
        );
    }

    #[test]
    fn test_axis_index_axis_index_unbound_axis() {
        // A name that no enclosing binder binds fails fast at the reader, before any operation is staged, surfacing
        // `AxisError::UnboundAxisName` as a `ProgramError::Axis` error.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("device", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let error = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::trace_with_named_axes(
            |input| input.context().axis_index("missing"),
            ArrayType::scalar(DataType::F64),
            vec![("device".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 4 })],
        )
        .unwrap_err();
        assert_eq!(error, ProgramError::Axis(AxisError::UnboundAxisName { name: "missing".to_string() }));
    }

    #[test]
    fn test_axis_index_axis_index_unbound_batch_axis() {
        // Inside a `batch` level, a name that neither that level nor any enclosing binder binds fails the same way,
        // and `batch` surfaces the error as a `BatchingError::Axis` error.
        let result: Result<Array, BatchingError> = batch(
            |item| item.context().axis_index("j"),
            Array::vector(vec![10.0, 20.0, 30.0]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("i"),
        );
        assert_eq!(result, Err(BatchingError::Axis(AxisError::UnboundAxisName { name: "j".to_string() })));
    }
}
