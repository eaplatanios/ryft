use std::collections::BTreeSet;
use std::fmt::Display;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayOperation,
    ArrayType, DataType, DimensionType, MeshAxisType, Sharding,
};
use crate::axes::{NamedAxes, NamedAxis};
use crate::batching::{BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiationDual;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::collectives::parallel_reduce::{ParallelReduceOperation, ParallelReductionKind};
use crate::operations::manipulation::broadcasting::BroadcastOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProvider, ProgramError, RegionInterface, Type, TypeError, Typed,
    Value, ValueProjection,
};

/// Name of [`ParallelVaryOperation`].
pub const PARALLEL_VARY_OPERATION_NAME: &str = "parallel_vary";

/// Adds variation over one active manual mesh axis. The local value is unchanged. Transposition sums cotangents over
/// the axis through the mesh-form [`ParallelReduceOperation`]. The selected axis must be invariant and cannot carry
/// reduced or unreduced state. Applying this transition twice to the same axis is an error. Multiple axes use
/// successive operations.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelVaryOperation {
    /// Refer to the documentation of [`axis_name`](Self::axis_name) for more information.
    axis_name: String,
}

impl ParallelVaryOperation {
    /// Creates a new [`ParallelVaryOperation`] over `axis_name`.
    pub fn new(axis_name: String) -> Self {
        Self { axis_name }
    }

    /// Returns the manual mesh axis selected by this [`ParallelVaryOperation`].
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }
}

impl Display for ParallelVaryOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ParallelVaryOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        PARALLEL_VARY_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);

        let input = &input_types[0];
        let axis_name = &self.axis_name;
        let Some(sharding) = input.sharding() else {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_VARY_OPERATION_NAME}` input must carry a mesh containing manual axis `{axis_name}`",
            )));
        };

        if sharding.mesh().axis_type(axis_name) != Some(MeshAxisType::Manual) {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_VARY_OPERATION_NAME}` axis `{axis_name}` must be manual",
            )));
        }

        if sharding.unreduced_axes().contains(axis_name) || sharding.reduced_axes().contains(axis_name) {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_VARY_OPERATION_NAME}` axis `{axis_name}` must not carry reduction state",
            )));
        }

        let mut axes = sharding.varying_manual_axes().clone();
        if !axes.insert(axis_name.clone()) {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_VARY_OPERATION_NAME}` input is already varying over manual axis `{axis_name}`",
            )));
        }

        // Only the variation changes. Placement, local shape, and reduction state are preserved.
        let output_sharding = sharding
            .clone()
            .with_varying_manual_axes(axes)
            .map_err(|error| TypeError::invalid(error.to_string()))?;

        Ok(vec![
            input
                .clone()
                .with_sharding(output_sharding)
                .map_err(|error| TypeError::invalid(error.to_string()))?,
        ])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("axis_name", format_args!("{:?}", self.axis_name)))
    }
}

// Any operation family that contains `ParallelVaryOperation` can provide it on request. Families that cannot
// stage variation transitions (e.g., test-only families) override this with an implementation that returns
// `ProgramError::UnsupportedOperation`, which keeps the `ParallelVary` blanket implementation, and every
// capability that aligns manual variation through it, available to their values.
impl<O: Operation<Type = ArrayType> + From<ParallelVaryOperation>> OperationProvider<ArrayType, ParallelVaryOperation>
    for O
{
    type Operation = Self;

    #[inline]
    fn provide(request: ParallelVaryOperation, input_types: &[&ArrayType]) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        Ok(Self::from(request))
    }
}

impl<C: Domain<Type = ArrayType, Value: ParallelVary>> InterpretableOperation<C> for ParallelVaryOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].parallel_vary(&self.axis_name)?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ParallelVaryOperation where
    C::Operation: From<ParallelVaryOperation>
{
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelVaryOperation>>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for ParallelVaryOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        if context.axis_name() == Some(self.axis_name.as_str()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` requires a manual mesh axis, not a named batch axis", self.name()),
            }
            .into());
        }

        let mut outputs = context.parent().bind(self.clone(), Vec::new(), std::slice::from_ref(inputs[0].value()))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![
            ArrayBatch::new(outputs.remove(0), inputs[0].batch_axis())?
                .with_ragged_axes(inputs[0].ragged_axes().to_vec())?,
        ]
        .into())
    }
}

impl_differentiable_operation! {
    ParallelVaryOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<ParallelVaryOperation>,
    {
        |operation, context, _driver, inputs| {
            check_count!("input", inputs, 1, ProgramError);
            let mut primals = context.primal().bind(
                operation.clone(),
                Vec::new(),
                std::slice::from_ref(inputs[0].primal()),
            )?;
            check_count!("output", primals, 1, ProgramError);
            let primal = primals.remove(0);
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(r#type) => {
                    MaybeZero::Zero(operation.infer_output_types(std::slice::from_ref(r#type), &[])?.remove(0))
                }
                MaybeZero::Value(tangent) => {
                    let mut tangents = context.tangent().bind(
                        operation.clone(),
                        Vec::new(),
                        std::slice::from_ref(tangent),
                    )?;
                    check_count!("output", tangents, 1, ProgramError);
                    MaybeZero::Value(tangents.remove(0))
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<ParallelReduceOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // Treating one shared value as `n` per-device copies is the linear map `x ↦ (x, …, x)`, whose adjoint sums
            // the `n` cotangent contributions back into one: a `parallel_sum` over the same axis of the cotangent's
            // mesh, which type inference validated to contain the axis when the forward operation was staged.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);

            if inputs[0].is_known() {
                return Ok(());
            }

            if let MaybeZero::Value(cotangent) = &outputs[0] {
                let Some(mesh) = cotangent.r#type().sharding().map(|sharding| sharding.mesh().clone()) else {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{}` cotangent must carry a mesh containing manual axis `{}`",
                        operation.name(),
                        operation.axis_name,
                    ))
                    .into());
                };

                let mut contributions = context.bind(
                    ParallelReduceOperation::new(operation.axis_name.clone(), ParallelReductionKind::Sum)
                        .with_mesh(mesh),
                    Vec::new(),
                    std::slice::from_ref(cotangent),
                )?;

                check_count!("output", contributions, 1, ProgramError);
                accumulators[0].accumulate(context, MaybeZero::Value(contributions.remove(0)))?;
            }

            Ok(())
        }
    },
}

impl<A: Value<Type = ArrayType>> From<ParallelVaryOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: ParallelVaryOperation) -> Self {
        Self::Array(ArrayOperation::ParallelVary(operation))
    }
}

/// Represents the ability to declare that a value varies across a manual mesh axis. Inside a manual region (e.g., the
/// body of a `shard_map` operation in the XLA backend), every value is a per-device local shard, and its type records
/// over which manual axes the shards may differ. A value that came from a replicated input, or from a constant, is
/// _invariant_ meaning that every device holds the same shard. A value that came from a tiled input is _varying_
/// meaning that each device holds its own piece. Combining the two in one operation is only sound when the invariant
/// value is first weakened to a varying one, which is what [`parallel_vary`](Self::parallel_vary) does. It stages a
/// [`ParallelVaryOperation`], which changes no bytes on any device and only records the weaker fact, so it lowers to
/// nothing. Placement, local shape, and reduction state are left as they are; only the variation changes. It is the
/// analogue of `jax.lax.pcast(x, axis_name, to='varying')` in JAX, which appears in jaxprs as the `pvary` primitive.
/// Refer to the [JAX `shard_map` guide](https://docs.jax.dev/en/latest/201/shard-map.html) for more information on how
/// JAX handles varying manual axes, which serves as the basis for how Ryft handles them.
///
/// The reason this transition is an explicit operation rather than a silent widening of the type is differentiation.
/// Treating one shared value as `n` per-device copies is the linear map `x ↦ (x, …, x)`, whose adjoint sums the `n`
/// cotangent contributions back into one. Transposing a [`ParallelVaryOperation`] therefore stages a `parallel_sum`
/// over the axis (a mesh-form [`ParallelReduceOperation`]), and that sum is exactly the cross-device reduction that
/// the gradient of a replicated parameter needs. Without the explicit transition, each device would report only its
/// own contribution. The other transforms keep the transition intact: forward-mode differentiation repeats it on the
/// tangent (a structural-zero tangent still receives the varying output type), partial evaluation retains it for the
/// backend that owns the mesh, and batching over an unrelated named axis preserves the mapped and ragged axes.
///
/// Users rarely call this directly. Multi-input capabilities (e.g., arithmetic, comparison, selection, slicing,
/// gathering, dots, sorts, etc.) insert it on whichever inputs lack the variation of their peers, and the XLA
/// `shard_map` operation inserts it on an invariant output that is returned along a tiled axis. Users must call it when
/// an operation family or a hand-built program does not. [`parallel_reduce`](crate::ParallelReduce::parallel_reduce)
/// calls it on an invariant input so that every copy is counted by the reduction. Type inference rejects the mismatch
/// that a missing call leaves behind as every operation with several array inputs enforces matching variation through
/// [`ArrayType::check_matching_manual_variation`], and a downstream operation should do the same.
///
/// The axis must be bound by an enclosing manual region (i.e., a [`NamedAxis::Mesh`] binding). Named batch axes and
/// participant subgroups do not establish the full-axis invariance that the transition weakens, so a `batch` level that
/// binds the same name is rejected, and concrete single-device values have no manual region and always fail. A value
/// without a sharding is first given a replicated placement on the axis's mesh, and a value that already varies over
/// the axis is rejected by type inference, so the operation is never a silent no-op.
///
/// The blanket implementation covers every context-carrying value whose operation family provides
/// [`ParallelVaryOperation`] and [`BroadcastOperation`] through [`OperationProvider`], which any family that contains
/// the two operations does automatically. A downstream family that contains neither loses this capability, and with it
/// every capability that aligns manual variation before binding (e.g., arithmetic, comparison, selection, etc.): the
/// compiler reports that the family's value does not implement [`ParallelVary`] at the first such call. The remedy is
/// either to add the two variants to the family or, for a family that never enters a manual region, to implement the
/// two providers with an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error, which restores the
/// capabilities and turns a variation transition into a runtime error instead of a compile error.
///
/// # Example
///
/// A shared scalar scaled by a per-device weight varies once it is combined with that weight. The capability inserts
/// the transition on the shared scalar, and the same transition appears when the program is written out by hand:
///
/// ```rust
/// # use indoc::indoc;
/// # use ryft_core::{
/// #     Array, ArrayOperation, ArrayType, DataType, LogicalMesh, MeshAxis, MeshAxisType, Mul, NamedAxis, ParallelVary,
/// #     Sharding, TracingContext,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual)?])?;
/// let invariant = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh.clone(), 0))?;
/// let varying = ArrayType::scalar(DataType::F32)
///     .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["x"])?)?;
/// let axes = vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2 })];
/// let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
///     |(shared, weight)| shared.parallel_vary("x")?.mul(&weight),
///     (invariant, varying.clone()),
///     axes,
/// )?;
/// assert_eq!(output, varying);
/// assert_eq!(
///     program.to_string(),
///     indoc! {"
///         lambda %0:f32[][sharding={mesh<['x'=2:manual]>, []}], \
///         %1:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
///         let %2:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_vary [axis_name=\"x\"] %0
///             %3:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = mul %2 %1
///         in (%3)"},
/// );
/// # Ok(())
/// # }
/// ```
pub trait ParallelVary: Sized {
    /// Returns this value marked as varying across the manual mesh axis `axis_name`, staging a
    /// [`ParallelVaryOperation`] and, for a value without a sharding, a broadcast that first places it on the axis's
    /// mesh. The local shard is unchanged; only its type records that shards may now differ across the axis.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of a manual mesh axis bound by an enclosing manual region.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `axis_name` is not bound by an enclosing manual region, if the axis is empty,
    /// or if this value already varies over it or carries reduction state along it.
    fn parallel_vary(&self, axis_name: &str) -> Result<Self, ProgramError>;
}

// The derived `ArrayOperation` interpreter requires every variant's value capability, even when a program does not use
// that variant. A concrete `Array` has no active manual mesh, so this implementation reports unsupported execution.
// Partial evaluation can then retain the operation for a backend that owns the mesh.
impl ParallelVary for Array {
    #[inline]
    fn parallel_vary(&self, axis_name: &str) -> Result<Self, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{PARALLEL_VARY_OPERATION_NAME}` requires an active non-empty manual mesh axis `{axis_name}`",
            ),
        })
    }
}

// Any context-carrying value varies by binding a `ParallelVaryOperation` through its own context, attaching a
// replicated placement on the axis's mesh first when its type carries no sharding. Both operations are requested
// through `OperationProvider` rather than converted through `From`, so that operation families without them can
// still provide the request with an error instead of losing every capability that aligns manual variation.
impl<
    V: Value<Type = ArrayType, DispatchDomain = C>,
    C: Context<
            Value = V,
            Operation: OperationProvider<ArrayType, ParallelVaryOperation, Operation = C::Operation>
                           + OperationProvider<ArrayType, BroadcastOperation, Operation = C::Operation>,
        > + NamedAxes,
> ParallelVary for V
{
    fn parallel_vary(&self, axis_name: &str) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let named_axis = context.named_axis(axis_name);
        let Some(NamedAxis::Mesh { mesh, size: 1.., .. }) = named_axis else {
            let message =
                format!("`{PARALLEL_VARY_OPERATION_NAME}` requires an active non-empty manual mesh axis `{axis_name}`");
            return Err(if context.is_eager() && named_axis.is_none() {
                ProgramError::UnsupportedOperation { message }
            } else {
                TypeError::invalid(message).into()
            });
        };

        let mut input = self.clone();
        let input_type = input.r#type().into_owned();
        if input_type.sharding().is_none() {
            let rank = input_type.rank();
            let mesh_type = input_type
                .clone()
                .with_sharding(Sharding::replicated(mesh, rank))
                .map_err(|error| TypeError::invalid(error.to_string()))?;
            let operation = BroadcastOperation::new(mesh_type, (0..rank).collect());
            let operation = C::Operation::provide(operation, &[&input_type])?;
            let mut mesh_inputs = context.bind(operation, Vec::new(), std::slice::from_ref(&input))?;
            check_count!("output", mesh_inputs, 1, ProgramError);
            input = mesh_inputs.remove(0);
        }

        let operation = ParallelVaryOperation::new(axis_name.to_string());
        let operation = C::Operation::provide(operation, &[input.r#type().as_ref()])?;
        let mut outputs = context.bind(operation, Vec::new(), &[input])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Represents the ability to make the manual variation of an operation's inputs agree before binding it. Inside a
/// manual region (e.g., the body of a `shard_map` operation in the XLA backend), an ordinary operation with several
/// inputs computes a function of them on every device independently, so its inputs must vary over the same manual
/// mesh axes, and its output varies over exactly those axes. This is the standard variation rule that
/// [`ArrayType::check_matching_manual_variation`] enforces in type inference. A capability for such an operation calls
/// [`align_manual_variation`](Self::align_manual_variation) on its inputs before binding, which weakens every input to
/// the union of the manual axes that any input varies over by staging a [`ParallelVaryOperation`] on each axis that it
/// lacks. The inserted transition owns the collective adjoint, so gradients through the operation stay correct. Local
/// shapes and values are unchanged, and ordinary shape broadcasting never manufactures or erases variation.
///
/// Only axes that an enclosing manual region binds take part. Scalar data-type and dimension values have no manual
/// variation and pass through unchanged, and a composite [`ArrayIrType`] value aligns only its array members.
pub trait ManualVariationAlignment<T: Type>: Value<Type = T> {
    /// Returns `inputs` with their manual variation aligned, in the same order.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if staging a variation transition on one of the inputs fails.
    fn align_manual_variation(inputs: &[Self]) -> Result<Vec<Self>, ProgramError>;
}

impl<V: Value<Type = DataType>> ManualVariationAlignment<DataType> for V {
    #[inline]
    fn align_manual_variation(inputs: &[Self]) -> Result<Vec<Self>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<V: Value<Type = ArrayType, DispatchDomain: Context + NamedAxes> + ParallelVary> ManualVariationAlignment<ArrayType>
    for V
{
    fn align_manual_variation(inputs: &[Self]) -> Result<Vec<Self>, ProgramError> {
        // Every input is weakened to the union of the axes that any input varies over, computed once from the unaligned
        // inputs. Only axes that an enclosing manual region binds take part: a name that no enclosing binder owns, or
        // that a `batch` level binds, is not a manual variation axis of this computation, so no transition is inserted
        // for it. Each input resolves names against its own context.
        let axes = inputs
            .iter()
            .filter_map(|input| input.r#type().sharding().map(|sharding| sharding.varying_manual_axes().clone()))
            .flatten()
            .collect::<BTreeSet<_>>();
        inputs
            .iter()
            .map(|input| {
                let context = input.dispatch_domain();
                let input_type = input.r#type();
                axes.iter()
                    .filter(|axis| {
                        matches!(context.named_axis(axis), Some(NamedAxis::Mesh { .. }))
                            && !input_type
                                .sharding()
                                .is_some_and(|sharding| sharding.varying_manual_axes().contains(*axis))
                    })
                    .try_fold(input.clone(), |input, axis| input.parallel_vary(axis))
            })
            .collect()
    }
}

impl<V: Value<Type = DimensionType>> ManualVariationAlignment<DimensionType> for V {
    #[inline]
    fn align_manual_variation(inputs: &[Self]) -> Result<Vec<Self>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: ManualVariationAlignment<ArrayType>>>
    ManualVariationAlignment<ArrayIrType> for V
{
    fn align_manual_variation(inputs: &[Self]) -> Result<Vec<Self>, ProgramError> {
        // Only array members carry manual variation. They are aligned among themselves and put back at their
        // original positions, while dimension and reference members pass through unchanged.
        let arrays = inputs
            .iter()
            .filter(|input| matches!(input.r#type().as_ref(), ArrayIrType::Array(_)))
            .cloned()
            .map(ValueProjection::<ArrayType>::into_projected)
            .collect::<Result<Vec<_>, _>>()?;
        let mut arrays = V::Projected::align_manual_variation(&arrays)?.into_iter();
        Ok(inputs
            .iter()
            .map(|input| match input.r#type().as_ref() {
                ArrayIrType::Array(_) => Self::from_projected(arrays.next().unwrap()),
                _ => input.clone(),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        ArrayIrValue, DataType, Dimension, DimensionBounds, DimensionType, DimensionVariable, LogicalMesh, MeshAxis,
        RaggedAxis, Shape, Sharding,
    };
    use crate::batching::BatchAxis;
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::macros::check_operation_type_inference;
    use crate::operations::math::add::Add;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, PartialValue};
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, Typed, ValueProjection};
    use crate::tracing::{DomainTracingContext, TracingContext};

    use super::*;

    /// Creates the invariant and varying types of one local scalar on a two-device manual mesh.
    fn scalar_types() -> (ArrayType, ArrayType) {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 0);
        let invariant = ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap();
        let varying = ArrayType::scalar(DataType::F32)
            .with_sharding(sharding.with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        (invariant, varying)
    }

    #[test]
    fn test_parallel_vary() {
        let operation = ParallelVaryOperation::new("m".to_string());
        assert_eq!(operation.axis_name(), "m");
        assert_eq!(operation.name(), PARALLEL_VARY_OPERATION_NAME);
        assert_eq!(operation.to_string(), "parallel_vary [axis_name=\"m\"]");
    }

    #[test]
    fn test_parallel_vary_type_inference() {
        let (invariant, varying) = scalar_types();
        check_operation_type_inference!(
            operation = ParallelVaryOperation::new("m".to_string()),
            cases = [{
                input_types = [invariant.clone()],
                output_types = [varying.clone()],
            }, {
                input_types = [varying],
                error = "`parallel_vary` input is already varying over manual axis `m`",
            }, {
                input_types = [ArrayType::scalar(DataType::F32)],
                error = "`parallel_vary` input must carry a mesh containing manual axis `m`",
            }],
        );
        assert_eq!(ParallelVaryOperation::new("m".to_string()).fold(&[invariant], &[]), Ok(None));
    }

    #[test]
    fn test_parallel_vary_type_inference_preserves_unrelated_state() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("outer", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("explicit", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::replicated(mesh, 0)
            .with_varying_manual_axes(["outer"])
            .unwrap()
            .with_unreduced_axes(["explicit"])
            .unwrap();
        let input = ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap();
        let output = ParallelVaryOperation::new("m".to_string())
            .infer_output_types(std::slice::from_ref(&input), &[])
            .unwrap();
        assert_eq!(
            output,
            vec![
                input
                    .clone()
                    .with_sharding(sharding.clone().with_varying_manual_axes(["m", "outer"]).unwrap())
                    .unwrap(),
            ]
        );
        assert!(matches!(
            ParallelVaryOperation::new("explicit".to_string()).infer_output_types(&[input.clone()], &[]),
            Err(TypeError::Invalid { message }) if message == "`parallel_vary` axis `explicit` must be manual",
        ));
        let reduced = input.with_sharding(sharding.with_reduced_axes(["m"]).unwrap()).unwrap();
        assert!(matches!(
            ParallelVaryOperation::new("m".to_string()).infer_output_types(&[reduced], &[]),
            Err(TypeError::Invalid { message }) if message == "`parallel_vary` axis `m` must not carry reduction state",
        ));
    }

    #[test]
    fn test_parallel_vary_interpretation() {
        let (invariant, _) = scalar_types();
        assert!(matches!(
            ParallelVaryOperation::new("m".to_string()).interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::from_elements(invariant, &[2.0_f32]).unwrap()],
            ),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`parallel_vary` requires an active non-empty manual mesh axis `m`",
        ));
    }

    #[test]
    fn test_parallel_vary_partial_evaluation() {
        let (invariant, varying) = scalar_types();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(invariant.clone());
        let output = builder
            .add_instruction(ParallelVaryOperation::new("m".to_string()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let constant = Array::from_elements(invariant, &[2.0_f32]).unwrap();
        let evaluation = program.to_flat_program().partially_evaluate(&[PartialValue::Known(constant)]).unwrap();
        assert_eq!(evaluation.program().output_types(), vec![varying]);
        assert_eq!(
            evaluation
                .program()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec![PARALLEL_VARY_OPERATION_NAME],
        );
        assert!(evaluation.outputs()[0].is_unknown());
    }

    #[test]
    fn test_parallel_vary_partial_evaluation_known_staging() {
        let (invariant, varying) = scalar_types();
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = PartialEvaluationValue::known(trace.input(invariant));
        let context = PartialEvaluationContext::new(trace);
        let outputs = ParallelVaryOperation::new("m".to_string())
            .partially_evaluate(&context, &EmptyRegionDriver, &[input])
            .unwrap();
        assert!(outputs[0].is_known());
        assert_eq!(outputs[0].r#type().as_ref(), &varying);
    }

    #[test]
    fn test_parallel_vary_partial_evaluation_composite() {
        let (invariant, varying) = scalar_types();
        let operation = ArrayIrOperation::<Array>::from(ParallelVaryOperation::new("m".to_string()));
        let eager = PartialEvaluationContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let input = PartialEvaluationValue::known(ArrayIrValue::Array(
            Array::from_elements(invariant.clone(), &[2.0_f32]).unwrap(),
        ));
        let outputs = operation.partially_evaluate(&eager, &EmptyRegionDriver, &[input]).unwrap();
        assert!(outputs[0].is_unknown());
        assert_eq!(outputs[0].r#type().as_ref(), &varying.clone().into());

        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = PartialEvaluationValue::known(trace.input(invariant.into()));
        let context = PartialEvaluationContext::new(trace);
        let outputs = operation.partially_evaluate(&context, &EmptyRegionDriver, &[input]).unwrap();
        assert!(outputs[0].is_known());
        assert_eq!(outputs[0].r#type().as_ref(), &varying.into());
    }

    #[test]
    fn test_parallel_vary_batching() {
        let (invariant, _) = scalar_types();
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(invariant.sharding().unwrap().mesh().clone(), 2))
            .unwrap();
        let packed = trace.input(packed_type);
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let input = ArrayBatch::new(packed, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents, length, vec![0])])
            .unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace, 2);
        let operation = ParallelVaryOperation::new("m".to_string());
        let outputs =
            operation.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)).unwrap().into_parts().0;
        assert_eq!(outputs[0].batch_axis(), input.batch_axis());
        assert_eq!(outputs[0].ragged_axes(), input.ragged_axes());
        assert_eq!(
            outputs[0].value().r#type().sharding().unwrap().varying_manual_axes(),
            &BTreeSet::from(["m".to_string()]),
        );

        let named_batch =
            BatchingContext::<_, ArrayBatchingPolicy>::new(context.parent().clone(), 2).with_axis_name("m".to_string());
        assert!(matches!(
            operation.batch(&named_batch, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Err(BatchingError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`parallel_vary` requires a manual mesh axis, not a named batch axis",
        ));
    }

    #[test]
    fn test_parallel_vary_batching_dynamic_extent() {
        let (invariant, _) = scalar_types();
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let length = DimensionVariable::new("items", DimensionBounds::new(0, Some(4)).unwrap());
        let extent = trace.input(DimensionType::from(length.clone()).into());
        let packed_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length)]))
            .with_sharding(Sharding::replicated(invariant.sharding().unwrap().mesh().clone(), 1))
            .unwrap();
        let packed = trace.input(packed_type.into());
        let input = ArrayBatch::new(packed.into_projected().unwrap(), BatchAxis::new(0)).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace),
            extent,
        );
        let output = ParallelVaryOperation::new("m".to_string())
            .batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(output[0].batch_axis(), input.batch_axis());
        assert_eq!(
            output[0].value().r#type().sharding().unwrap().varying_manual_axes(),
            &BTreeSet::from(["m".to_string()]),
        );
    }

    #[test]
    fn test_parallel_vary_differentiation() {
        let (invariant, varying) = scalar_types();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_vary("m"),
            invariant,
            vec![(
                "m".to_string(),
                NamedAxis::Mesh { axis: 0, size: 2, mesh: scalar_types().0.sharding().unwrap().mesh().clone() },
            )],
        )
        .unwrap();
        let differentiated = program.to_flat_program().jvp().unwrap();
        assert_eq!(differentiated.output_types(), vec![varying.clone(), varying.clone()]);
        let zero_tangent = program.to_flat_program().jvp_with_respect_to(&[]).unwrap();
        assert_eq!(zero_tangent.output_types(), vec![varying.clone(), varying]);
        assert_eq!(
            zero_tangent
                .instructions()
                .iter()
                .filter(|instruction| instruction.operation().name() == PARALLEL_VARY_OPERATION_NAME)
                .count(),
            1
        );
        assert_eq!(
            zero_tangent
                .instructions()
                .iter()
                .filter(|instruction| instruction.operation().name() == "parallel_sum")
                .count(),
            0
        );
        assert_eq!(
            differentiated
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec![PARALLEL_VARY_OPERATION_NAME, PARALLEL_VARY_OPERATION_NAME]
        );
    }

    #[test]
    fn test_parallel_vary_transposition() {
        let (invariant, varying) = scalar_types();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(invariant.clone());
        let output = builder
            .add_instruction(ParallelVaryOperation::new("m".to_string()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(transposed.input_types(), vec![varying.clone()]);
        assert_eq!(transposed.output_types(), vec![invariant]);
        assert_eq!(
            transposed
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["parallel_sum"]
        );
        assert!(matches!(transposed.instructions()[0].operation(), ArrayOperation::ParallelReduce(operation)
            if operation.mesh() == Some(varying.sharding().unwrap().mesh())));
        let twice = transposed.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(twice.to_string(), program.to_string());
    }

    #[test]
    fn test_parallel_vary_capability() {
        let (invariant, varying) = scalar_types();
        let mesh = invariant.sharding().unwrap().mesh().clone();
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_vary("m"),
            invariant,
            vec![("m".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(output, varying);
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec![PARALLEL_VARY_OPERATION_NAME],
        );

        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_vary("m"),
            ArrayType::scalar(DataType::F32),
            vec![("m".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        assert_eq!(output, varying);
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["broadcast", PARALLEL_VARY_OPERATION_NAME],
        );
    }

    #[test]
    fn test_manual_variation_alignment() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let varying = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        let (output, program) =
            DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::trace_with_named_axes(
                |(left, right)| Add::add(&left, &right),
                (invariant, varying.clone()),
                vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap();
        assert_eq!(output, varying);
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["parallel_vary", "add"],
        );
    }

    #[test]
    fn test_manual_variation_alignment_mixed_ir() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        let (output, program) =
            DomainTracingContext::<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::trace_with_named_axes(
                |(left, right)| Ok(ManualVariationAlignment::align_manual_variation(&[left, right])?.remove(0)),
                (ArrayIrType::Array(ArrayType::scalar(DataType::F32)), ArrayIrType::Array(varying.clone())),
                vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap();
        assert_eq!(output, ArrayIrType::Array(varying));
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["broadcast", "parallel_vary"],
        );
    }

    #[test]
    fn test_manual_variation_alignment_mixed_ir_preserves_positions() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        let extent = DimensionType::new("extent", DimensionBounds::new(1, Some(4)).unwrap());

        // A dimension member passes through at its position, while the array members around it are aligned.
        let (outputs, program) =
            DomainTracingContext::<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::trace_with_named_axes(
                |(left, extent, right)| {
                    let mut outputs = ManualVariationAlignment::align_manual_variation(&[left, extent, right])?;
                    let right = outputs.pop().unwrap();
                    let extent = outputs.pop().unwrap();
                    Ok((outputs.pop().unwrap(), extent, right))
                },
                (
                    ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                    ArrayIrType::Dimension(extent.clone()),
                    ArrayIrType::Array(varying.clone()),
                ),
                vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
            )
            .unwrap();
        assert_eq!(
            outputs,
            (ArrayIrType::Array(varying.clone()), ArrayIrType::Dimension(extent), ArrayIrType::Array(varying)),
        );
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["broadcast", "parallel_vary"],
        );
    }
}
