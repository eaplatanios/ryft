use std::fmt::Display;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayOperation,
    ArrayType, MeshAxisType, Sharding,
};
use crate::axes::{NamedAxes, NamedAxis};
use crate::batching::{BatchableOperation, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiationDual;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::Capability;
use crate::operations::collectives::parallel_reduce::ParallelReduceOperation;
use crate::operations::collectives::parallel_vary::ParallelVary;
use crate::operations::manipulation::broadcasting::BroadcastOperation;
use crate::operations::reductions::ReductionKind;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
    ValueDomainDispatch,
};

/// Name of [`ParallelCastOperation`]. The operation's [`ParallelCastTarget`] is rendered as its `target` attribute.
pub const PARALLEL_CAST_OPERATION_NAME: &str = "parallel_cast";

/// State along one manual mesh axis that a [`ParallelCastOperation`] moves a value into. Each target has exactly one
/// source state from which the operation reaches it, so the source is implied:
///
///   - [`Reduced`](Self::Reduced) is reached from an invariant value. It is JAX's `preduced`, and its adjoint is the
///     sum of an unreduced value (a mesh-form [`ParallelReduceOperation`] of kind [`ReductionKind::Sum`]).
///   - [`Varying`](Self::Varying) is reached from a reduced value. It is JAX's `reduced_vary_cast`, and its adjoint is
///     the cast to [`Unreduced`](Self::Unreduced).
///   - [`Unreduced`](Self::Unreduced) is reached from a varying value. It is JAX's `vary_unreduced_cast`, and its
///     adjoint is the cast to [`Varying`](Self::Varying).
///
/// The fourth transition, from invariant to varying, is [`ParallelVaryOperation`](crate::ParallelVaryOperation),
/// whose adjoint is a sum of a varying value. [`ParallelCast::parallel_cast`] dispatches to it for invariant inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParallelCastTarget {
    /// Marks an invariant value as reduced, meaning that its reverse mode cotangent holds per-device summands
    /// (i.e., is unreduced) along the axis.
    Reduced,

    /// Marks a reduced value as varying, meaning that its devices may now hold different values along the axis.
    Varying,

    /// Marks a varying value as unreduced, meaning that the logical value is the sum of the per-device values along
    /// the axis, which a later sum completes.
    Unreduced,
}

impl Display for ParallelCastTarget {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reduced => write!(formatter, "reduced"),
            Self::Varying => write!(formatter, "varying"),
            Self::Unreduced => write!(formatter, "unreduced"),
        }
    }
}

/// [`Operation`] that moves a value between the reduction and variation states of one active manual mesh axis
/// without changing any local value, so that it lowers to nothing. Each [`ParallelCastTarget`] has one source
/// state, which type inference validates, and the transposes pair the casts as JAX does. Specifically, a cast to
/// [`Reduced`](ParallelCastTarget::Reduced) transposes to a sum of its unreduced cotangent (i.e., a mesh-form
/// [`ParallelReduceOperation`]), and the casts to [`Varying`](ParallelCastTarget::Varying) and
/// [`Unreduced`](ParallelCastTarget::Unreduced) transpose to each other. Placement, local shape, and the state
/// of every other axis are preserved. Multiple axes use successive operations. Refer to the documentation of
/// [`ParallelCast`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelCastOperation {
    /// Refer to the documentation of [`axis_name`](Self::axis_name) for more information.
    axis_name: String,

    /// Refer to the documentation of [`target`](Self::target) for more information.
    target: ParallelCastTarget,
}

impl ParallelCastOperation {
    /// Creates a new [`ParallelCastOperation`] that moves its input into `target` along the manual axis `axis_name`.
    #[inline]
    pub fn new(axis_name: String, target: ParallelCastTarget) -> Self {
        Self { axis_name, target }
    }

    /// Returns the manual mesh axis along which this [`ParallelCastOperation`] changes the state of its input.
    #[inline]
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }

    /// Returns the [`ParallelCastTarget`] that this [`ParallelCastOperation`] moves its input into.
    #[inline]
    pub fn target(&self) -> ParallelCastTarget {
        self.target
    }
}

impl Display for ParallelCastOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ParallelCastOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        PARALLEL_CAST_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);

        let input = &input_types[0];
        let axis_name = &self.axis_name;
        let target = self.target;
        let Some(sharding) = input.sharding() else {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_CAST_OPERATION_NAME}` input must carry a mesh containing manual axis `{axis_name}`",
            )));
        };

        if sharding.mesh().axis_type(axis_name) != Some(MeshAxisType::Manual) {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_CAST_OPERATION_NAME}` axis `{axis_name}` must be manual",
            )));
        }

        let source = if sharding.varying_manual_axes().contains(axis_name) {
            "varying"
        } else if sharding.unreduced_axes().contains(axis_name) {
            "unreduced"
        } else if sharding.reduced_axes().contains(axis_name) {
            "reduced"
        } else {
            "invariant"
        };
        let expected_source = match target {
            ParallelCastTarget::Reduced => "invariant",
            ParallelCastTarget::Varying => "reduced",
            ParallelCastTarget::Unreduced => "varying",
        };
        if source != expected_source {
            let remedy = if source == "invariant" && target == ParallelCastTarget::Varying {
                "; use `parallel_vary` to make an invariant value varying"
            } else {
                ""
            };
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_CAST_OPERATION_NAME}` to `{target}` requires an input that is {expected_source} along \
                 manual axis `{axis_name}` but got one that is {source}{remedy}",
            )));
        }

        // Only the state of the selected axis changes. Each step removes the axis from its source set before adding
        // it to its target set, because a sharding never holds one axis in two of these sets.
        let without = |axes: &std::collections::BTreeSet<String>| {
            axes.iter().filter(|axis| *axis != axis_name).cloned().collect::<Vec<_>>()
        };
        let with = |axes: &std::collections::BTreeSet<String>| {
            axes.iter().cloned().chain(std::iter::once(axis_name.clone())).collect::<Vec<_>>()
        };
        let sharding = sharding.clone();
        let output_sharding = match target {
            ParallelCastTarget::Reduced => {
                let reduced = with(sharding.reduced_axes());
                sharding.with_reduced_axes(reduced)
            }
            ParallelCastTarget::Varying => {
                let (reduced, varying) = (without(sharding.reduced_axes()), with(sharding.varying_manual_axes()));
                sharding.with_reduced_axes(reduced).and_then(|sharding| sharding.with_varying_manual_axes(varying))
            }
            ParallelCastTarget::Unreduced => {
                let (varying, unreduced) = (without(sharding.varying_manual_axes()), with(sharding.unreduced_axes()));
                sharding
                    .with_varying_manual_axes(varying)
                    .and_then(|sharding| sharding.with_unreduced_axes(unreduced))
            }
        }
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
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
            operation.field("target", self.target)
        })
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for ParallelCastOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // A reduction state exists only inside a manual region, whose backend lowers the cast to the identity, so there
        // is no per-value result to produce here. Partial evaluation residualizes the operation through the deferral of
        // unsupported operations in its default rule, as it does for the mesh-form `parallel_reduce`.
        check_count!("input", inputs, 1, ProgramError);
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{}` along manual mesh axis `{}` requires an execution backend owning the mesh",
                self.name(),
                self.axis_name,
            ),
        })
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelCastOperation>>> PartiallyEvaluatableOperation<C>
    for ParallelCastOperation
{
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelCastOperation>>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for ParallelCastOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // A cast changes only the state of a manual mesh axis, so unrelated mapped axes pass through untouched,
        // exactly as they do through `parallel_vary`. A named batch axis has no reduction state to change.
        check_count!("input", inputs, 1, ProgramError);
        if context.axis_name() == Some(self.axis_name.as_str()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` requires a manual mesh axis, not a named batch axis", self.name()),
            }
            .into());
        }
        Ok(context.forward_to_parent(self.clone().into(), inputs)?)
    }
}

impl_differentiable_operation! {
    ParallelCastOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<ParallelCastOperation>,
    {
        |operation, context, _driver, inputs| {
            // Every cast is linear (the identity on local values), so the tangent rides the same cast.
            check_count!("input", inputs, 1, ProgramError);
            let mut primals =
                context.primal().bind(operation.clone(), Vec::new(), std::slice::from_ref(inputs[0].primal()))?;
            check_count!("output", primals, 1, ProgramError);
            let primal = primals.remove(0);
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(r#type) => {
                    MaybeZero::Zero(operation.infer_output_types(std::slice::from_ref(r#type), &[])?.remove(0))
                }
                MaybeZero::Value(tangent) => {
                    let mut tangents =
                        context.tangent().bind(operation.clone(), Vec::new(), std::slice::from_ref(tangent))?;
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
        O: Operation<Type = ArrayType> + From<ParallelCastOperation> + From<ParallelReduceOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // A cast to `Reduced` marks an invariant value whose cotangent is unreduced, and the adjoint completes that
            // pending sum, yielding the invariant cotangent of the input. The casts between `Varying` and `Unreduced`
            // are adjoint to each other: the cotangent of a reduced input is unreduced, and that of an unreduced input
            // is reduced. Every cast is the identity on local values, so no adjoint changes any value either.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            let MaybeZero::Value(cotangent) = &outputs[0] else {
                return Ok(());
            };
            if !accumulators[0].is_needed() {
                return Ok(());
            }
            let axis_name = operation.axis_name.clone();
            let mut contributions = match operation.target {
                ParallelCastTarget::Reduced => {
                    let Some(mesh) = cotangent.r#type().sharding().map(|sharding| sharding.mesh().clone()) else {
                        return Err(ProgramError::MalformedProgram(format!(
                            "`{}` cotangent must carry a mesh containing manual axis `{}`",
                            operation.name(),
                            axis_name,
                        ))
                        .into());
                    };
                    let operation = ParallelReduceOperation::new(ReductionKind::Sum, axis_name).with_mesh(mesh);
                    context.bind(operation, Vec::new(), std::slice::from_ref(cotangent))?
                }
                ParallelCastTarget::Varying => context.bind(
                    ParallelCastOperation::new(axis_name, ParallelCastTarget::Unreduced),
                    Vec::new(),
                    std::slice::from_ref(cotangent),
                )?,
                ParallelCastTarget::Unreduced => context.bind(
                    ParallelCastOperation::new(axis_name, ParallelCastTarget::Varying),
                    Vec::new(),
                    std::slice::from_ref(cotangent),
                )?,
            };
            check_count!("output", contributions, 1, ProgramError);
            accumulators[0].accumulate(context, MaybeZero::Value(contributions.remove(0)))?;
            Ok(())
        }
    },
}

impl<A: Value<Type = ArrayType>> From<ParallelCastOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: ParallelCastOperation) -> Self {
        Self::Array(ArrayOperation::ParallelCast(operation))
    }
}

/// Represents the ability to move a value between the variation and reduction states of a manual mesh axis. Inside
/// a manual region (e.g., the body of a [`ShardMapOperation`](crate::ShardMapOperation)), the type of every per-device
/// local value records, for each manual axis, one of four states: _invariant_ (every device along the axis holds the
/// same value), _varying_ (the devices may hold different values), _unreduced_ (the logical value is the sum of the
/// per-device values, which a later sum completes), or _reduced_ (an invariant value whose reverse mode cotangent is
/// unreduced). [`parallel_cast`](Self::parallel_cast) changes that state without changing any local value, so it lowers
/// to nothing. It is the Ryft analogue of JAX's
/// [`jax.lax.pcast`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.pcast.html),
/// and it supports the same four transitions:
///
///   - Invariant to [`Varying`](ParallelCastTarget::Varying) stages a
///     [`ParallelVaryOperation`](crate::ParallelVaryOperation), whose transpose sums the cotangents of the devices.
///   - Invariant to [`Reduced`](ParallelCastTarget::Reduced), reduced to [`Varying`](ParallelCastTarget::Varying),
///     and varying to [`Unreduced`](ParallelCastTarget::Unreduced) stage a [`ParallelCastOperation`].
///
/// The remaining transition, from unreduced to invariant, communicates: it is the sum
/// [`parallel_reduce`](crate::ParallelReduce::parallel_reduce) with [`ReductionKind::Sum`] of an unreduced value, and
/// it is the transpose of the cast to [`Reduced`](ParallelCastTarget::Reduced). Every other combination of source state
/// and target is an error, same as in JAX.
///
/// The axis must be bound by an enclosing manual region (i.e., a [`NamedAxis::Mesh`] binding). A binding that disables
/// manual-variation checks returns the value unchanged without staging anything, as JAX's casts do without `check_vma`,
/// because the bodies of such regions carry no variation or reduction state along their active axes. Named batch axes
/// have no such state, so a `batch` level that binds the same name rejects the cast.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// Each device holds a summand of a shared total. Casting it to unreduced and summing yields the total, and the
/// transpose of that sum is a cast to reduced:
///
/// ```rust
/// # use indoc::indoc;
/// # use ryft_core::{
/// #     Array, ArrayOperation, ArrayType, DataType, LogicalMesh, MeshAxis, MeshAxisType, NamedAxis, ParallelCast,
/// #     ParallelCastTarget, ParallelReduce, ReductionKind, Sharding, TracingContext,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual)?])?;
/// let varying = ArrayType::scalar(DataType::F32)
///     .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["x"])?)?;
/// let axes = vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2, checks_manual_variation: true })];
/// let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
///     |summand| summand.parallel_cast("x", ParallelCastTarget::Unreduced)?.parallel_reduce(ReductionKind::Sum, "x"),
///     varying,
///     axes,
/// )?;
/// assert_eq!(
///     program.to_string(),
///     indoc! {"
///         lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
///         let %1:f32[][sharding={mesh<['x'=2:manual]>, [], unreduced={'x'}}] = \
///             parallel_cast [axis_name=\"x\", target=unreduced] %0
///             %2:f32[][sharding={mesh<['x'=2:manual]>, []}] = \
///                 parallel_reduce [kind=sum, axis_name=\"x\", mesh=['x'=2:manual]] %1
///         in (%2)"},
/// );
/// assert_eq!(
///     program.transpose_with_respect_to(&[0], &[])?.to_string(),
///     indoc! {"
///         lambda %0:f32[][sharding={mesh<['x'=2:manual]>, []}] .
///         let %1:f32[][sharding={mesh<['x'=2:manual]>, [], reduced={'x'}}] = \
///             parallel_cast [axis_name=\"x\", target=reduced] %0
///             %2:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
///                 parallel_cast [axis_name=\"x\", target=varying] %1
///         in (%2)"},
/// );
/// # Ok(())
/// # }
/// ```
#[capability(projection(ArrayIrType => ArrayType))]
pub trait ParallelCast<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns this value moved into `target` along the manual mesh axis `axis_name`, staging a
    /// [`ParallelCastOperation`] or, for an invariant value cast to [`Varying`](ParallelCastTarget::Varying), a
    /// [`ParallelVaryOperation`](crate::ParallelVaryOperation). A value without a sharding is first given a replicated
    /// placement on the axis's mesh. When the enclosing manual region disables variation tracking for this axis, the
    /// value is returned unchanged.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of a manual mesh axis bound by an enclosing manual region.
    ///   - `target`: [`ParallelCastTarget`] that this value is moved into.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `axis_name` is not bound by an enclosing manual region, if this value carries
    /// a sharding on a mesh other than that of the region, or if its state along the axis is not the source state of
    /// `target`.
    fn parallel_cast(&self, axis_name: &str, target: ParallelCastTarget) -> Result<Self, ProgramError>;
}

impl ParallelCast for Array {
    #[inline]
    fn parallel_cast(&self, axis_name: &str, _target: ParallelCastTarget) -> Result<Self, ProgramError> {
        // The derived `ArrayOperation` interpreter requires every variant's value capability, even when a program does
        // not use that variant. A concrete `Array` has no active manual mesh, so this implementation reports
        // unsupported execution.
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{PARALLEL_CAST_OPERATION_NAME}` requires an active non-empty manual mesh axis `{axis_name}`",
            ),
        })
    }
}

impl<
    V: Value<
            Type = ArrayType,
            Dispatch = ValueDomainDispatch,
            Domain: Context<Operation: From<ParallelCastOperation> + From<BroadcastOperation>> + NamedAxes,
        > + ParallelVary,
> ParallelCast<ArrayType> for V
{
    fn parallel_cast(&self, axis_name: &str, target: ParallelCastTarget) -> Result<Self, ProgramError> {
        // Any context-carrying value casts by binding a `ParallelCastOperation` through its own context, attaching a
        // replicated placement on the axis's mesh first when its type carries no sharding, exactly as `parallel_vary`
        // does.
        let context = self.domain();
        let Some(NamedAxis::Mesh { mesh, size: 1.., checks_manual_variation, .. }) = context.named_axis(axis_name)
        else {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_CAST_OPERATION_NAME}` requires an active non-empty manual mesh axis `{axis_name}`",
            ))
            .into());
        };
        if !checks_manual_variation {
            return Ok(self.clone());
        }

        let input_type = self.r#type().into_owned();
        let mut input = self.clone();
        match input_type.sharding() {
            Some(sharding) if sharding.mesh() != &mesh => {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_CAST_OPERATION_NAME}` input mesh does not match the mesh of manual axis `{axis_name}`",
                ))
                .into());
            }
            Some(_) => {}
            None => {
                let rank = input_type.rank();
                let mesh_type = input_type
                    .clone()
                    .with_sharding(Sharding::replicated(mesh.clone(), rank))
                    .map_err(|error| TypeError::invalid(error.to_string()))?;
                let operation = BroadcastOperation::new(mesh_type, (0..rank).collect());
                let mut mesh_inputs = context.bind(operation, Vec::new(), std::slice::from_ref(&input))?;
                check_count!("output", mesh_inputs, 1, ProgramError);
                input = mesh_inputs.remove(0);
            }
        }

        let invariant = input.r#type().sharding().is_some_and(|sharding| {
            !sharding.varying_manual_axes().contains(axis_name)
                && !sharding.unreduced_axes().contains(axis_name)
                && !sharding.reduced_axes().contains(axis_name)
        });
        if target == ParallelCastTarget::Varying && invariant {
            return input.parallel_vary(axis_name);
        }

        let operation = ParallelCastOperation::new(axis_name.to_string(), target);
        let mut outputs = context.bind(operation, Vec::new(), &[input])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayIrValue, DataType, LogicalMesh, MeshAxis};
    use crate::batching::BatchAxis;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::DifferentiableType;
    use crate::macros::check_operation_type_inference;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, TypedProgram};
    use crate::tracing::TracingContext;

    use super::*;

    /// Creates the invariant, varying, reduced, and unreduced types of one local scalar along the manual axis `m` of
    /// a mesh that also has a manual axis `outer`.
    fn scalar_types() -> (ArrayType, ArrayType, ArrayType, ArrayType) {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("outer", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::replicated(mesh, 0);
        let scalar = |sharding: Sharding| ArrayType::scalar(DataType::F32).with_sharding(sharding).unwrap();
        (
            scalar(sharding.clone()),
            scalar(sharding.clone().with_varying_manual_axes(["m"]).unwrap()),
            scalar(sharding.clone().with_reduced_axes(["m"]).unwrap()),
            scalar(sharding.with_unreduced_axes(["m"]).unwrap()),
        )
    }

    /// Binds `"m"` and `"outer"` as manual axes of `mesh`, optionally without manual-variation checks.
    fn manual_mesh_axes(mesh: &LogicalMesh, checks_manual_variation: bool) -> Vec<(String, NamedAxis)> {
        ["m", "outer"]
            .into_iter()
            .enumerate()
            .map(|(axis, name)| {
                (name.to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis, size: 2, checks_manual_variation })
            })
            .collect()
    }

    /// Builds a program that applies `operation` to one input of type `input_type`.
    fn cast_program(
        operation: ParallelCastOperation,
        input_type: ArrayType,
    ) -> crate::Program<Array, ArrayOperation<Array>, Array, Array> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap()
    }

    #[test]
    fn test_parallel_cast() {
        for (target, rendered) in [
            (ParallelCastTarget::Reduced, "reduced"),
            (ParallelCastTarget::Varying, "varying"),
            (ParallelCastTarget::Unreduced, "unreduced"),
        ] {
            let operation = ParallelCastOperation::new("m".to_string(), target);
            assert_eq!(operation.name(), PARALLEL_CAST_OPERATION_NAME);
            assert_eq!(operation.axis_name(), "m");
            assert_eq!(operation.target(), target);
            assert_eq!(target.to_string(), rendered);
            assert_eq!(operation.to_string(), format!("parallel_cast [axis_name=\"m\", target={rendered}]"));
        }
    }

    #[test]
    fn test_parallel_cast_type_inference() {
        let (invariant, varying, reduced, unreduced) = scalar_types();
        check_operation_type_inference!(
            operation = ParallelCastOperation::new("m".to_string(), ParallelCastTarget::Reduced),
            cases = [{
                input_types = [invariant.clone()],
                output_types = [reduced.clone()],
            }, {
                input_types = [varying.clone()],
                error = "`parallel_cast` to `reduced` requires an input that is invariant along manual axis `m` but got \
                         one that is varying",
            }, {
                input_types = [reduced.clone()],
                error = "`parallel_cast` to `reduced` requires an input that is invariant along manual axis `m` but got \
                         one that is reduced",
            }, {
                input_types = [ArrayType::scalar(DataType::F32)],
                error = "`parallel_cast` input must carry a mesh containing manual axis `m`",
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelCastOperation::new("m".to_string(), ParallelCastTarget::Varying),
            cases = [{
                input_types = [reduced.clone()],
                output_types = [varying.clone()],
            }, {
                input_types = [invariant.clone()],
                error = "`parallel_cast` to `varying` requires an input that is reduced along manual axis `m` but got \
                         one that is invariant; use `parallel_vary` to make an invariant value varying",
            }, {
                input_types = [unreduced.clone()],
                error = "`parallel_cast` to `varying` requires an input that is reduced along manual axis `m` but got \
                         one that is unreduced",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelCastOperation::new("m".to_string(), ParallelCastTarget::Unreduced),
            cases = [{
                input_types = [varying.clone()],
                output_types = [unreduced.clone()],
            }, {
                input_types = [invariant.clone()],
                error = "`parallel_cast` to `unreduced` requires an input that is varying along manual axis `m` but got \
                         one that is invariant",
            }],
        );

        // The axis must be a manual axis of the input's mesh.
        check_operation_type_inference!(
            operation = ParallelCastOperation::new("missing".to_string(), ParallelCastTarget::Reduced),
            cases = [{ input_types = [invariant], error = "`parallel_cast` axis `missing` must be manual" }],
        );
    }

    #[test]
    fn test_parallel_cast_type_inference_preserves_unrelated_state() {
        // Only the state of the selected axis changes. Placement, local shape, and the state of other axes are kept.
        let (invariant, _, _, _) = scalar_types();
        let sharding = invariant.sharding().unwrap().clone();
        let with = |sharding: Sharding| ArrayType::new_static(DataType::F32, [3]).with_sharding(sharding).unwrap();
        let base = Sharding::replicated(sharding.mesh().clone(), 1).with_unreduced_axes(["outer"]).unwrap();
        check_operation_type_inference!(
            operation = ParallelCastOperation::new("m".to_string(), ParallelCastTarget::Reduced),
            cases = [{
                input_types = [with(base.clone())],
                output_types = [with(base.clone().with_reduced_axes(["m"]).unwrap())],
            }],
        );
        let base = Sharding::replicated(sharding.mesh().clone(), 1).with_varying_manual_axes(["outer"]).unwrap();
        check_operation_type_inference!(
            operation = ParallelCastOperation::new("m".to_string(), ParallelCastTarget::Unreduced),
            cases = [{
                input_types = [with(base.clone().with_varying_manual_axes(["m", "outer"]).unwrap())],
                output_types = [with(base.with_unreduced_axes(["m"]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_parallel_cast_interpretation() {
        // A concrete `Array` never executes inside a manual region, so eager interpretation reports unsupported
        // execution.
        let (invariant, _, _, _) = scalar_types();
        assert_eq!(
            ParallelCastOperation::new("m".to_string(), ParallelCastTarget::Reduced).interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::from_elements(invariant, &[2.0f32]).unwrap()],
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "`parallel_cast` along manual mesh axis `m` requires an execution backend owning the mesh"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_cast_partial_evaluation() {
        // Folding a known input would interpret the cast eagerly, which a concrete `Array` does not support, so partial
        // evaluation retains the cast for the backend that owns the mesh.
        let (invariant, _, reduced, _) = scalar_types();
        let program =
            cast_program(ParallelCastOperation::new("m".to_string(), ParallelCastTarget::Reduced), invariant.clone());
        let constant = Array::from_elements(invariant, &[2.0f32]).unwrap();
        let evaluation = program.to_flat_program().partially_evaluate(&[PartialValue::Known(constant)]).unwrap();
        assert_eq!(evaluation.program().output_types(), vec![reduced]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, []}] .
                let %1:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [], reduced={'m'}}] = \
                        parallel_cast [axis_name=\"m\", target=reduced] %0
                in (%1)"
            },
        );
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
    }

    #[test]
    fn test_parallel_cast_batching() {
        // A `batch` level that does not bind the axis name forwards the cast to its parent, preserving the mapped axis.
        let (invariant, _, _, _) = scalar_types();
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(invariant.sharding().unwrap().mesh().clone(), 2))
            .unwrap();
        let input = ArrayBatch::new(trace.input(packed_type), BatchAxis::new(0)).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 2);
        let operation = ParallelCastOperation::new("m".to_string(), ParallelCastTarget::Reduced);
        let outputs = operation.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs[0].batch_axis(), input.batch_axis());
        let builder = trace.builder().borrow().clone();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].value().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [{}, {}]}] .
                let %1:f32[2, 3][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [{}, {}], reduced={'m'}}] = \
                        parallel_cast [axis_name=\"m\", target=reduced] %0
                in (%1)"
            },
        );

        // A `batch` level that binds the same name has no reduction state to change, so it rejects the cast.
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace, 2).with_axis_name("m".to_string());
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[input]).map(|_| ()),
            Err(BatchingError::Program(ProgramError::UnsupportedOperation {
                message: "`parallel_cast` requires a manual mesh axis, not a named batch axis".to_string(),
            })),
        );
    }

    #[test]
    fn test_parallel_cast_differentiation() {
        // Forward mode differentiation repeats the cast on the tangent, and a structural zero tangent receives the
        // cast output type.
        let (invariant, _, reduced, _) = scalar_types();
        let mesh = invariant.sharding().unwrap().mesh().clone();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_cast("m", ParallelCastTarget::Reduced),
            invariant,
            manual_mesh_axes(&mesh, true),
        )
        .unwrap();
        let differentiated = program.to_flat_program().jvp().unwrap();
        assert_eq!(differentiated.output_types(), vec![reduced.clone(), reduced.clone()]);
        assert_eq!(
            differentiated.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, []}], \
                %1:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, []}] .
                let %2:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [], reduced={'m'}}] = \
                        parallel_cast [axis_name=\"m\", target=reduced] %0
                    %3:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [], reduced={'m'}}] = \
                        parallel_cast [axis_name=\"m\", target=reduced] %1
                in (%2, %3)"
            },
        );
        let zero_tangent = program.to_flat_program().jvp_with_respect_to(&[]).unwrap();
        assert_eq!(zero_tangent.output_types(), vec![reduced.clone(), reduced]);
    }

    #[test]
    fn test_parallel_cast_transposition() {
        // The casts transpose as JAX's do: `Reduced` to a sum of the unreduced cotangent, and `Varying` and `Unreduced`
        // to each other. Transposing twice recovers each original program.
        let (invariant, varying, reduced, unreduced) = scalar_types();
        for (target, input_type, cotangent_type, expected) in [
            (
                ParallelCastTarget::Reduced,
                invariant.clone(),
                unreduced.clone(),
                indoc! {"
                    lambda %0:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [], unreduced={'m'}}] .
                    let %1:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, []}] = \
                            parallel_reduce [kind=sum, axis_name=\"m\", mesh=['m'=2:manual, 'outer'=2:manual]] %0
                    in (%1)"
                },
            ),
            (
                ParallelCastTarget::Varying,
                reduced.clone(),
                varying.clone(),
                indoc! {"
                    lambda %0:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}] .
                    let %1:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [], unreduced={'m'}}] = \
                            parallel_cast [axis_name=\"m\", target=unreduced] %0
                    in (%1)"
                },
            ),
            (
                ParallelCastTarget::Unreduced,
                varying.clone(),
                reduced.clone(),
                indoc! {"
                    lambda %0:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [], reduced={'m'}}] .
                    let %1:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}] = \
                            parallel_cast [axis_name=\"m\", target=varying] %0
                    in (%1)"
                },
            ),
        ] {
            let program = cast_program(ParallelCastOperation::new("m".to_string(), target), input_type.clone());
            let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
            assert_eq!(transposed.input_types(), vec![cotangent_type]);
            assert_eq!(transposed.output_types(), vec![input_type.cotangent().unwrap()]);
            assert_eq!(transposed.to_string(), expected);
            assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());
        }
    }

    #[test]
    fn test_parallel_cast_parallel_cast() {
        let (invariant, varying, reduced, unreduced) = scalar_types();
        let mesh = invariant.sharding().unwrap().mesh().clone();
        let trace = |input_type: ArrayType, target: ParallelCastTarget, checks_manual_variation: bool| {
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_cast("m", target),
                input_type,
                manual_mesh_axes(&mesh, checks_manual_variation),
            )
        };

        // Each target is reached from its source state through a cast, except that an invariant value becomes varying
        // through `parallel_vary`, as JAX's `pcast` dispatches to `pvary`.
        for (input_type, target, output_type) in [
            (invariant.clone(), ParallelCastTarget::Reduced, reduced.clone()),
            (reduced.clone(), ParallelCastTarget::Varying, varying.clone()),
            (varying.clone(), ParallelCastTarget::Unreduced, unreduced.clone()),
            (invariant.clone(), ParallelCastTarget::Varying, varying.clone()),
        ] {
            let (output, _) = trace(input_type, target, true).unwrap();
            assert_eq!(output, output_type);
        }
        let (_, program) = trace(invariant.clone(), ParallelCastTarget::Varying, true).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, []}] .
                let %1:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}] = \
                        parallel_vary [axis_name=\"m\"] %0
                in (%1)"
            },
        );

        // A value without a sharding is first placed on the axis's mesh.
        let (output, program) = trace(ArrayType::scalar(DataType::F32), ParallelCastTarget::Reduced, true).unwrap();
        assert_eq!(output, reduced);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, []}] = broadcast [
                    output_type=f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, []}],
                    output_axes=[],
                ] %0
                    %2:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [], reduced={'m'}}] = \
                            parallel_cast [axis_name=\"m\", target=reduced] %1
                in (%2)"
            },
        );

        // An unchecked binding carries no reduction state, so the value is returned unchanged.
        let (output, program) = trace(invariant.clone(), ParallelCastTarget::Reduced, false).unwrap();
        assert_eq!(output, invariant);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, []}] .
                in (%0)"
            },
        );

        // A value in the wrong source state, an unbound axis, and a value on another mesh are rejected.
        assert_eq!(
            trace(unreduced, ParallelCastTarget::Reduced, true).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_cast` to `reduced` requires an input that is invariant along manual axis `m` but got one \
                 that is unreduced",
            ))),
        );
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_cast("m", ParallelCastTarget::Reduced),
                invariant,
                Vec::new(),
            )
            .map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_cast` requires an active non-empty manual mesh axis `m`",
            ))),
        );
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        assert_eq!(
            trace(
                ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(other_mesh, 0)).unwrap(),
                ParallelCastTarget::Reduced,
                true,
            )
            .map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_cast` input mesh does not match the mesh of manual axis `m`",
            ))),
        );
    }

    #[test]
    fn test_parallel_cast_parallel_cast_array_ir() {
        // Array IR values cast through the projection of the capability and stage the same operation.
        let (invariant, _, reduced, _) = scalar_types();
        let mesh = invariant.sharding().unwrap().mesh().clone();
        let (output, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_cast("m", ParallelCastTarget::Reduced),
            ArrayIrType::from(invariant),
            manual_mesh_axes(&mesh, true),
        )
        .unwrap();
        assert_eq!(output, ArrayIrType::from(reduced));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, []}] .
                let %1:f32[][sharding={mesh<['m'=2:manual, 'outer'=2:manual]>, [], reduced={'m'}}] = \
                        parallel_cast [axis_name=\"m\", target=reduced] %0
                in (%1)"
            },
        );
    }
}
