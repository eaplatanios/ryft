//! Contains the named-axis [`ParallelReduceOperation`], which reduces a value across a named axis (`parallel_sum` /
//! `parallel_mean` / `parallel_max`), together with its interpretation, partial-evaluation, batching, forward-mode
//! differentiation, and transposition rules. These are the analogues of
//! [JAX's parallel operators](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators) `jax.lax.psum`,
//! `jax.lax.pmean`, and `jax.lax.pmax`.

// TODO(eaplatanios): Review this module.

use std::fmt::Display;
use std::ops::Mul as StdMul;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayType, DataType, LogicalMesh, MeshAxisType,
    RaggedArrayExtentBatchingPolicy, RaggedAxis, RaggedMaskIdentity, Shape,
};
use crate::axes::{AxisError, NamedAxes, NamedAxis};
use crate::batching::{BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiationDual;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::arithmetic::DivOperation;
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::fill::Fill;
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::memory::TransferToMemory;
use crate::operations::reductions::{Reduce, ReductionKind};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
};

use super::parallel_vary::{ManualVariationAlignment, ParallelVary, ParallelVaryOperation};
use super::{effective_collective_axis_size, forward_collective_to_parent, resolve_named_axis_size};

/// Kind of collective performed by a [`ParallelReduceOperation`].
///
/// Collectives operate on a named axis, resolved against the active [`NamedAxes`] environment. When an enclosing
/// `batch` level binds the name (a [`NamedAxis::Batched`](crate::axes::NamedAxis) axis), the matching
/// [`BatchingContext`] consumes the mapped batch axis at trace time. When a `shard_map` manual region binds the name
/// to a device mesh axis (a [`NamedAxis::Mesh`](crate::axes::NamedAxis) axis), the collective stays in the staged body
/// and lowers to a cross-device `all_reduce` over that mesh axis. The operations described here mirror JAX's
/// `jax.lax.{psum, pmean, pmax}` family.
///
/// `Sum`/`Mean`/`Max` reduce the named axis away, producing a result that is identical across all batch items or
/// device shards (replicated). Shape-changing collectives use their dedicated operation payloads below because their
/// result types also depend on a ranked array axis and on whether the named axis is materialized or tiled.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParallelReductionKind {
    /// Sum reduction across the named axis (`jax.lax.psum`).
    Sum,

    /// Mean reduction across the named axis (`jax.lax.pmean`).
    Mean,

    /// Maximum reduction across the named axis (`jax.lax.pmax`).
    Max,
}

impl ParallelReductionKind {
    /// Returns the canonical operation name suffix for this kind.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "parallel_sum",
            Self::Mean => "parallel_mean",
            Self::Max => "parallel_max",
        }
    }

    /// Returns the [`ReductionKind`] used to collapse the named axis.
    pub fn reduction_kind(self) -> ReductionKind {
        match self {
            Self::Sum | Self::Mean => ReductionKind::Sum,
            Self::Max => ReductionKind::Max,
        }
    }
}

impl Display for ParallelReductionKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Primitive representing one named-axis collective operation.
///
/// Ordinary reductions created by [`new`](Self::new) or [`grouped`](Self::grouped) are identity at the per-item level
/// (the named axis does not exist in per-item semantics) and collapse the mapped axis inside a [`BatchingContext`] whose
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches this collective's axis name. Under nested
/// `batch` levels, the batching rule below owns that decision: a matching level consumes the mapped batch axis, while
/// a non-matching level forwards the collective untouched to its parent context via
/// [`forward_collective_to_parent`], where the next level repeats the same name resolution.
///
/// A matching `parallel_sum` or `parallel_max` accepts bounded ragged operands. Padding is replaced with the
/// reduction identity before the participant reduction, and each surviving ragged extent is the elementwise maximum
/// across the participating mapped items. Participants whose local extents exclude a position contribute the identity;
/// with multiple ragged axes, a position inside the coordinatewise-maximum output bounds that no participant covers
/// is therefore the identity. For `parallel_sum`, that hole is zero. For `parallel_max`, it is the element type's
/// maximum identity: the lowest value, which is zero or false for unsigned integers and Booleans rather than an
/// unconditionally chosen numeric zero. Signed or floating-point negative live values therefore still win correctly.
/// `parallel_mean` rejects ragged operands because its denominator has no single implied meaning: participant count,
/// present-value count, and logical-element count define different operations.
///
/// A reduction created with [`with_mesh`](Self::with_mesh) reduces over a manual axis of that mesh inside a manual
/// region (e.g., the body of a `shard_map` operation in the XLA backend). Its input must vary over the axis, and its
/// output no longer does: every device holds the full reduction. This is the analogue of JAX's `psum_invariant`
/// primitive, and of `pmax` inside a `shard_map`. Its generic per-item interpreter cannot execute that collective
/// and reports an error; partial evaluation preserves it for a backend owning the mesh, and lowering emits an
/// `all_reduce` over the mesh axis. Ordinary reductions retain their per-item interpretation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelReduceOperation {
    /// Refer to the documentation of [`axis_name`](Self::axis_name) for more information.
    axis_name: String,

    /// Refer to the documentation of [`kind`](Self::kind) for more information.
    kind: ParallelReductionKind,

    /// Refer to the documentation of [`axis_size`](Self::axis_size) for more information.
    axis_size: Option<usize>,

    /// Refer to the documentation of [`axis_index_groups`](Self::axis_index_groups) for more information.
    axis_index_groups: Option<Vec<Vec<usize>>>,

    /// Refer to the documentation of [`mesh`](Self::mesh) for more information.
    mesh: Option<LogicalMesh>,
}

impl ParallelReduceOperation {
    /// Creates a new [`ParallelReduceOperation`] with the supplied axis name and kind.
    #[inline]
    pub fn new(axis_name: String, kind: ParallelReductionKind) -> Self {
        Self { axis_name, kind, axis_size: None, axis_index_groups: None, mesh: None }
    }

    /// Creates a grouped collective after validating that `axis_index_groups` is an equal-sized exact partition of
    /// `0..axis_size`.
    pub fn grouped(
        axis_name: String,
        kind: ParallelReductionKind,
        axis_size: usize,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, TypeError> {
        effective_collective_axis_size(kind.name(), axis_size, Some(axis_index_groups.as_slice()))?;
        Ok(Self { axis_name, kind, axis_size: Some(axis_size), axis_index_groups: Some(axis_index_groups), mesh: None })
    }

    /// Returns a copy of this [`ParallelReduceOperation`] that reduces over a manual axis of `mesh`. The input must
    /// vary over [`axis_name`](Self::axis_name) on that mesh and the output is invariant over it. Validation occurs
    /// during type inference. [`ParallelReduce::parallel_reduce`] supplies the mesh automatically from the enclosing
    /// manual region.
    #[inline]
    pub fn with_mesh(mut self, mesh: LogicalMesh) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Returns the axis name referenced by this collective.
    #[inline]
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }

    /// Returns the kind of collective.
    #[inline]
    pub fn kind(&self) -> ParallelReductionKind {
        self.kind
    }

    /// Returns the full named-axis size recorded for a grouped collective.
    #[inline]
    pub fn axis_size(&self) -> Option<usize> {
        self.axis_size
    }

    /// Returns the ordered participant groups, if any.
    #[inline]
    pub fn axis_index_groups(&self) -> Option<&[Vec<usize>]> {
        self.axis_index_groups.as_deref()
    }

    /// Returns the logical mesh whose manual axis this operation reduces over, or `None` for an ordinary reduction
    /// whose per-item semantics are identity. The mesh form removes the axis from its input's varying manual axes;
    /// ordinary and grouped reductions preserve the input variation.
    #[inline]
    pub fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

    /// Returns the number of participants in each group, or `None` for an ungrouped collective whose size is supplied
    /// by the enclosing binder.
    pub fn group_size(&self) -> Option<usize> {
        self.axis_index_groups.as_ref().and_then(|groups| groups.first()).map(Vec::len)
    }
}

impl Display for ParallelReduceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ParallelReduceOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        match self.kind {
            ParallelReductionKind::Sum => "parallel_sum",
            ParallelReductionKind::Mean => "parallel_mean",
            ParallelReductionKind::Max => "parallel_max",
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match (&self.axis_index_groups, self.axis_size) {
            (Some(axis_index_groups), Some(axis_size)) => {
                effective_collective_axis_size(self.name(), axis_size, Some(axis_index_groups.as_slice()))?;
            }
            (None, None) => {}
            _ => {
                return Err(TypeError::invalid(format!(
                    "`{}` must store both the full axis size and axis index groups, or neither",
                    self.name(),
                )));
            }
        }
        if let Some(mesh) = &self.mesh {
            let name = self.name();
            if self.axis_index_groups.is_some() {
                return Err(TypeError::invalid(format!(
                    "`{name}` over a manual mesh axis must not use axis index groups"
                )));
            }
            if self.kind == ParallelReductionKind::Mean {
                return Err(TypeError::invalid(
                    "`parallel_mean` over a manual mesh axis is staged as a `parallel_sum` divided by the axis size",
                ));
            }
            if mesh.axis_type(&self.axis_name) != Some(MeshAxisType::Manual) {
                return Err(TypeError::invalid(format!("`{name}` mesh axis `{}` must be manual", self.axis_name)));
            }
            let input = &input_types[0];
            let Some(sharding) = input.sharding() else {
                return Err(TypeError::invalid(format!(
                    "`{name}` input must carry a mesh containing manual axis `{}`",
                    self.axis_name,
                )));
            };
            if sharding.mesh() != mesh {
                return Err(TypeError::invalid(format!("`{name}` input mesh does not match the operation mesh")));
            }
            if sharding.unreduced_axes().contains(&self.axis_name) || sharding.reduced_axes().contains(&self.axis_name)
            {
                return Err(TypeError::invalid(format!(
                    "`{name}` axis `{}` must not carry reduction state",
                    self.axis_name,
                )));
            }
            if self.kind == ParallelReductionKind::Max && !sharding.unreduced_axes().is_empty() {
                // Maximum does not commute with summing partial contributions along other unreduced axes.
                return Err(TypeError::invalid("`parallel_max` does not support unreduced inputs"));
            }
            if !sharding.varying_manual_axes().contains(&self.axis_name) {
                return Err(TypeError::invalid(format!(
                    "`{name}` input must vary over manual axis `{}`; pass an invariant value through `parallel_vary` \
                     first so that every copy is counted",
                    self.axis_name,
                )));
            }
            let mut axes = sharding.varying_manual_axes().clone();
            axes.remove(&self.axis_name);
            let sharding = sharding
                .clone()
                .with_varying_manual_axes(axes)
                .map_err(|error| TypeError::invalid(error.to_string()))?;
            return Ok(vec![
                input.clone().with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()))?,
            ]);
        }
        // The per-item operation is identity; the named axis only exists physically inside an
        // enclosing `BatchingContext` where the batching rule will collapse it.
        Ok(vec![input_types[0].clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
            if let Some(mesh) = &self.mesh {
                operation.field("mesh", mesh)?;
            }
            if let Some(axis_size) = self.axis_size {
                operation.field("axis_size", axis_size)?;
            }
            if let Some(axis_index_groups) = &self.axis_index_groups {
                operation.field("axis_index_groups", format_args!("{axis_index_groups:?}"))?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for ParallelReduceOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        if self.mesh.is_some() {
            // A reduction over a manual mesh axis combines shards held by different devices. There is no per-item
            // value to produce; partial evaluation retains the operation for the backend that owns the mesh.
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{}` over manual mesh axis `{}` requires an execution backend owning the mesh",
                    self.name(),
                    self.axis_name,
                ),
            });
        }
        // Per-item interpretation is identity: the named axis does not exist in per-item semantics, so reducing across
        // it is a no-op. Staging a collective over an unbound axis is now rejected up front (see `ParallelReduce`), so
        // an enclosing binder always collapses the mapped axis through the batching rule before this per-item fallback
        // matters; it remains defined for a program interpreted directly, where a collective simply passes its operand
        // through per item.
        Ok(vec![inputs[0].clone()])
    }
}

// Ordinary reductions retain per-item partial evaluation; mesh-form reductions residualize through the
// unsupported-operation deferral of the default rule because their interpreter has no per-item value.
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ParallelReduceOperation where
    C::Operation: From<ParallelReduceOperation>
{
}

// Batching rule for [`ParallelReduceOperation`]. This rule owns named-axis resolution: when the active context's
// [`axis_name`](crate::batching::BatchingContext::axis_name) matches this collective's axis name, the mapped batch
// axis is consumed; otherwise the collective targets an outer `batch` level (or a device mesh axis) and is forwarded
// untouched to the parent context via [`forward_collective_to_parent`].
//
// The consuming arm collapses the mapped axis through `collective_reduce_batch` and binds a `Mean`'s `1 / N`
// rank-0 fill into the parent context — interpreted eagerly under an eager parent and staged into the enclosing
// trace under a staging parent — so one rule serves eager and staged batching alike.
impl<C, P: RaggedArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>> for ParallelReduceOperation
where
    C: Context<Type = ArrayType> + Fill<f64, C::Value>,
    C::Operation: From<ParallelReduceOperation>,
    <C as Domain>::Value: Reduce + StdMul<Output = <C as Domain>::Value>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        if self.mesh.is_some() {
            if context.axis_name() == Some(self.axis_name.as_str()) {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!("`{}` over a manual mesh axis cannot bind a named batch axis", self.name()),
                });
            }
            // A reduction over a manual mesh axis combines the local shards elementwise, so unrelated mapped and
            // ragged axes pass through untouched, exactly as they do through its adjoint `parallel_vary`.
            let mut outputs =
                context.parent().bind(self.clone(), Vec::new(), std::slice::from_ref(inputs[0].value()))?;
            check_count!("output", outputs, 1, ProgramError);
            return Ok(vec![
                ArrayBatch::new(outputs.remove(0), inputs[0].batch_axis())?
                    .with_ragged_axes(inputs[0].ragged_axes().to_vec())?,
            ]
            .into());
        }
        if context.axis_name() != Some(self.axis_name.as_str()) {
            let parent_operation = C::Operation::from(self.clone());
            return Ok(forward_collective_to_parent(context, parent_operation, inputs)?.into());
        }
        if self.axis_index_groups.is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{}` axis index groups are not supported when a batch transform binds the collective axis",
                    self.name(),
                ),
            });
        }
        Ok(collective_reduce_batch(context, self.kind, inputs, |factor_type, inverse_axis_size| {
            // The `1 / N` rank-0 factor binds into the batching context's parent — interpreted eagerly under an eager
            // parent, staged into the enclosing trace under a staging parent.
            context.parent().fill(&factor_type, inverse_axis_size)
        })?
        .into())
    }
}

impl_differentiable_operation! {
    ParallelReduceOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<ParallelReduceOperation>,
    {
        |operation, context, _driver, inputs| {
            // Forward-mode (JVP) rule for [`ParallelReduceOperation`]. `Sum`/`Mean` are linear, so the tangent is the
            // same collective applied to the operand tangent: `tangent_out = collective(input.tangent())`. A
            // structural-zero operand tangent keeps its type through the operation's inference (the mesh form changes
            // the variation) rather than staging a collective on a zero, keeping `collective(zero)` out of the tangent
            // program. `Max` is non-linear and reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation)
            // error.
            check_count!("input", inputs, 1, ProgramError);
            if matches!(operation.kind, ParallelReductionKind::Max) {
                return Err(ProgramError::UnsupportedOperation {
                    message: "`parallel_max` differentiation is not yet supported".to_string(),
                }
                .into());
            }
            let primal = stage_collective(context.primal(), operation, inputs[0].primal())?;
            // A collective of a structural zero stays a structural zero, keeping `collective(zero)` out of the
            // tangent program.
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(r#type) => {
                    MaybeZero::Zero(operation.infer_output_types(std::slice::from_ref(r#type), &[])?.remove(0))
                }
                MaybeZero::Value(tangent) => MaybeZero::Value(stage_collective(context.tangent(), operation, tangent)?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<ParallelReduceOperation> + From<ParallelVaryOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // Transpose rule for [`ParallelReduceOperation`]. An ordinary `parallel_sum`/`parallel_mean` is
            // self-adjoint, so the operand cotangent is the same collective applied to the output cotangent. A sum
            // over a manual mesh axis is the linear map `(x_1, …, x_n) ↦ Σ x_i` whose adjoint hands the shared
            // cotangent to every device, i.e., a `parallel_vary` over the same axis. The single operand is linear (its
            // [`PartialValue`] is [`Unknown`](PartialValue::Unknown)); a known operand contributes no cotangent and so
            // receives a structural zero. `Max` reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation)
            // error.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            if matches!(operation.kind, ParallelReductionKind::Max) {
                return Err(ProgramError::UnsupportedOperation {
                    message: "`parallel_max` transpose is not yet supported".to_string(),
                }
                .into());
            }
            // A known (non-linear) operand contributes no cotangent.
            if inputs[0].is_known() {
                return Ok(());
            }
            match &outputs[0] {
                MaybeZero::Value(cotangent) if operation.mesh.is_some() => {
                    let mut contributions = context.bind(
                        ParallelVaryOperation::new(operation.axis_name.clone()),
                        Vec::new(),
                        std::slice::from_ref(cotangent),
                    )?;
                    check_count!("output", contributions, 1, ProgramError);
                    accumulators[0].accumulate(context, MaybeZero::Value(contributions.remove(0)))?;
                    Ok(())
                }
                MaybeZero::Value(cotangent) => {
                    let contribution = stage_collective(&**context, operation, cotangent)?;
                    accumulators[0].accumulate(context, MaybeZero::Value(contribution))?;
                    Ok(())
                }
                MaybeZero::Zero(_) => Ok(()),
            }
        }
    },
}

/// Shared reduce-and-optionally-mean skeleton for [`ParallelReduceOperation`] batching. It collapses the mapped batch
/// axis with the kind's [`ReductionKind`] and, for `Mean`, scales the replicated result by `1 / N` using a
/// `make_parallel_mean_factor`-produced rank-0 factor (relying on implicit rank-0 broadcasting in the multiplication).
/// Outside a matching batching context (no mapped axis), it is an identity pass-through.
fn collective_reduce_batch<C, P, MakeParallelMeanFactor>(
    context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
    kind: ParallelReductionKind,
    inputs: &[ArrayBatch<C::Value>],
    make_parallel_mean_factor: MakeParallelMeanFactor,
) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    C::Value: Reduce + StdMul<Output = C::Value>,
    P: RaggedArrayExtentBatchingPolicy<C>,
    MakeParallelMeanFactor: FnOnce(ArrayType, f64) -> Result<C::Value, ProgramError>,
{
    check_count!("input", inputs, 1, ProgramError);
    let input = &inputs[0];
    if matches!(kind, ParallelReductionKind::Mean) && !input.ragged_axes().is_empty() {
        return Err(BatchingError::UnsupportedOperation {
            message: "`parallel_mean` does not define a denominator for bounded ragged inputs".to_string(),
        });
    }
    let Some(batch_axis) = input.batch_axis_position() else {
        // Outside any matching batching context: identity pass-through.
        return Ok(vec![input.clone()]);
    };
    let masked_axes = input.ragged_axes().iter().map(RaggedAxis::axis).collect::<Vec<_>>();
    let input = match kind {
        ParallelReductionKind::Sum => {
            P::mask_identity_input(context, input, masked_axes.as_slice(), RaggedMaskIdentity::Zero)?
        }
        ParallelReductionKind::Max => {
            P::mask_identity_input(context, input, masked_axes.as_slice(), RaggedMaskIdentity::Lowest)?
        }
        ParallelReductionKind::Mean => input.clone(),
    };
    // Reduce along the mapped batch axis with the corresponding reduction kind. The output is replicated: every
    // batch item sees the same reduced value, matching JAX's `psum`/`pmean`/`pmax` broadcast semantics.
    let mut output_value = input.value().clone().reduce(&[batch_axis], kind.reduction_kind())?;
    if matches!(kind, ParallelReductionKind::Mean) {
        // Mean divides the summed value by the batch size, which must be statically known to scale by `1 / N`.
        let inverse_axis_size = 1.0 / parallel_mean_batch_size(&input)? as f64;
        let factor_type = parallel_mean_factor_type(output_value.r#type().data_type());
        output_value = make_parallel_mean_factor(factor_type, inverse_axis_size)? * output_value;
    }
    let ragged_axes = input
        .ragged_axes()
        .iter()
        .map(|ragged_axis| {
            let extent_batch_axis = ragged_axis.extent_axes().iter().position(|axis| *axis == batch_axis);
            let extents = match extent_batch_axis {
                None => ragged_axis.extents().clone(),
                Some(extent_batch_axis) => {
                    ragged_axis.extents().clone().reduce(&[extent_batch_axis], ReductionKind::Max)?
                }
            };
            Ok(RaggedAxis::new(
                ragged_axis.axis() - usize::from(batch_axis < ragged_axis.axis()),
                extents,
                ragged_axis.dimension().clone(),
                ragged_axis
                    .extent_axes()
                    .iter()
                    .filter_map(|axis| (*axis != batch_axis).then_some(*axis - usize::from(batch_axis < *axis)))
                    .collect(),
            ))
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    Ok(vec![ArrayBatch::new(output_value, BatchAxis::replicated())?.with_ragged_axes(ragged_axes)?])
}

/// Returns the static batch size for a `Mean` over the mapped batch axis of `input`, erroring when
/// the batch size is dynamic (a mean cannot be scaled by `1 / N` without a static `N`).
fn parallel_mean_batch_size<V: Value<Type = ArrayType>>(input: &ArrayBatch<V>) -> Result<usize, BatchingError> {
    input.batch_size()?.ok_or_else(|| BatchingError::UnsupportedOperation {
        message: "`parallel_mean` requires a static batch size; the staged batch axis is dynamic".to_string(),
    })
}

/// Builds the rank-0 [`ArrayType`] of `data_type` used to hold a `Mean`'s `1 / N` factor.
fn parallel_mean_factor_type(data_type: DataType) -> ArrayType {
    ArrayType::new(data_type, Shape::scalar())
}

/// Re-stages this collective of the same axis name and kind on a single tracer operand, returning its single output.
///
/// Both the forward-mode (`jvp`) and the transpose rules below re-stage the collective on a tracer (the primal, the
/// tangent, or the output cotangent), which is exactly one operation with one input and one output.
fn stage_collective<C>(
    context: &C,
    operation: &ParallelReduceOperation,
    operand: &C::Value,
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayType>,
    C::Operation: From<ParallelReduceOperation>,
{
    let mut outputs = context.bind(operation.clone(), Vec::new(), std::slice::from_ref(operand))?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

/// Represents the ability to reduce a value across a named axis, producing one result that every batch item or device
/// shard holds in full. The axis is resolved by name against the active [`NamedAxes`] environment at staging time, so
/// an unbound name fails fast rather than silently acting as identity. A name bound by an enclosing `batch` level is
/// collapsed at trace time by [`BatchableOperation::batch`], which reduces the mapped batch axis. A name bound to a
/// device mesh axis by a manual region (e.g., the body of a `shard_map` operation in the XLA backend) stays in the
/// staged body program and lowers to a cross-device `all_reduce` over that mesh axis. This is the analogue of JAX's
/// `jax.lax.psum`, `jax.lax.pmean`, and `jax.lax.pmax`.
///
/// Inside a manual region, every value is a per-device local shard and its type records over which manual axes the
/// shards may differ. A value that varies over the axis holds a different shard on every device along it, and reducing
/// it stages a [`ParallelReduceOperation`] carrying the region's mesh, which combines those shards over the whole axis
/// and hands every device the same result, so the output is *invariant* over the axis. This is what JAX stages as its
/// `psum_invariant` primitive. A value that is invariant over the axis holds the same shard on every device. Reducing
/// it counts that shard once per device, which is why the operation only accepts varying inputs: an invariant value is
/// first passed through [`parallel_vary`](ParallelVary::parallel_vary), which makes the multiplication by the device
/// count an explicit, differentiable step rather than an accident. [`parallel_reduce`](Self::parallel_reduce) inserts
/// that step itself, so summing an invariant constant `c` over an axis of size `n` yields `n · c`, exactly as
/// `jax.lax.psum` does. The sum and the variation transition are adjoints of each other, which is what makes gradients
/// through a manual region correct without any bookkeeping at the region boundary: the transpose of the mesh-form sum
/// is a [`ParallelVaryOperation`], because the cotangent of a shared total is the same value handed to every device,
/// and the transpose of a [`ParallelVaryOperation`] is the mesh-form sum. A mean over a manual axis is the mesh-form
/// sum divided by the axis size, as `jax.lax.pmean` is `psum(x) / n`, and a maximum is not differentiable.
///
/// Named batch axes and participant subgroups retain the ordinary collective contract: the staged operation carries no
/// mesh and preserves the input variation, because distinct groups can produce distinct results.
///
/// # Example
///
/// Each device squares its own weight, and the sum of the squares is shared by all of them. Transposing the program
/// turns the sum back into a variation, so the gradient of the shared total reaches every device's weight:
///
/// ```rust
/// # use indoc::indoc;
/// # use ryft_core::{
/// #     Array, ArrayOperation, ArrayType, DataType, LogicalMesh, MeshAxis, MeshAxisType, NamedAxis, Operation,
/// #     ParallelReduce, ParallelReductionKind, Sharding, TracingContext,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual)?])?;
/// let invariant = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh.clone(), 0))?;
/// let varying = ArrayType::scalar(DataType::F32)
///     .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["x"])?)?;
/// let axes = vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2 })];
/// let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
///     |weight| weight.parallel_reduce("x", ParallelReductionKind::Sum),
///     varying.clone(),
///     axes,
/// )?;
/// assert_eq!(output, invariant.clone());
/// assert_eq!(
///     program.to_string(),
///     indoc! {"
///         lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
///         let %1:f32[][sharding={mesh<['x'=2:manual]>, []}] = parallel_sum [axis_name=\"x\", mesh=['x'=2:manual]] %0
///         in (%1)"},
/// );
/// let transposed = program.transpose_with_respect_to(&[0], &[])?;
/// assert_eq!(transposed.input_types(), vec![invariant]);
/// assert_eq!(transposed.output_types(), vec![varying]);
/// let operations = transposed.instructions().iter().map(|instruction| instruction.operation().name());
/// assert_eq!(operations.collect::<Vec<_>>(), ["parallel_vary"]);
/// # Ok(())
/// # }
/// ```
pub trait ParallelReduce: Sized {
    /// Returns the reduction of this value across the named axis `axis_name`, staging a [`ParallelReduceOperation`]
    /// of the given `kind`. Over a manual mesh axis, an input that is invariant over the axis is first passed
    /// through [`parallel_vary`](ParallelVary::parallel_vary), the staged operation records
    /// the axis's mesh, and a [`Mean`](ParallelReductionKind::Mean) is staged as the sum divided by the axis size.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `kind`: Kind of reduction to perform.
    ///
    /// # Errors
    ///
    /// Returns [`AxisError::UnboundAxisName`] (surfaced as [`BatchingError::Axis`] riding a [`ProgramError::Custom`]
    /// payload) when no enclosing binder binds `axis_name`, and a [`ProgramError`] if this value carries reduction
    /// state along a manual `axis_name` or its sharding is on a different mesh than the region's.
    fn parallel_reduce(&self, axis_name: &str, kind: ParallelReductionKind) -> Result<Self, ProgramError>;

    /// Returns the reduction of this value across each participant group of the named axis `axis_name`, staging a
    /// grouped [`ParallelReduceOperation`] after validating that the groups cover the axis exactly once. Grouped
    /// reductions preserve manual variation, because distinct groups can produce distinct results.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `kind`: Kind of reduction to perform.
    ///   - `axis_index_groups`: Equal-sized exact partition of `0..axis_size` into participant groups.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `axis_name` is not bound by an enclosing binder or if `axis_index_groups` is not
    /// an equal-sized exact partition of the axis.
    fn parallel_reduce_with_axis_index_groups(
        &self,
        axis_name: &str,
        kind: ParallelReductionKind,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError>;
}

// Any context-carrying value applies a collective by validating the axis name against the active [`NamedAxes`]
// environment and binding a [`ParallelReduceOperation`] through its own context: a staged tracer records the
// operation, a batching tracer resolves the named axis against the batching context stack, and a JVP dual forwards to
// the primal-side resolution. An unbound name fails fast with [`AxisError::UnboundAxisName`] rather than silently
// acting as identity.
impl<V: Value<Type = ArrayType> + ManualVariationAlignment<ArrayType> + ParallelVary> ParallelReduce for V
where
    V::DispatchDomain: Context + NamedAxes,
    <V::DispatchDomain as Domain>::Operation:
        From<ParallelReduceOperation> + From<ConstantOperation<Array>> + From<DivOperation<ArrayType>>,
{
    fn parallel_reduce(&self, axis_name: &str, kind: ParallelReductionKind) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let Some(named_axis) = context.named_axis(axis_name) else {
            return Err(BatchingError::Axis(AxisError::UnboundAxisName { name: axis_name.to_string() }).into());
        };
        let NamedAxis::Mesh { mesh, size, .. } = named_axis else {
            let operation = ParallelReduceOperation::new(axis_name.to_string(), kind);
            let mut outputs = context.bind(operation, Vec::new(), std::slice::from_ref(self))?;
            check_count!("output", outputs, 1, ProgramError);
            return Ok(outputs.remove(0));
        };
        // Reducing an invariant value counts its shard once per device, so the value is first made varying, which
        // records that multiplication as an explicit, differentiable step.
        let mut input = self.clone();
        if !input.r#type().sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis_name)) {
            input = input.parallel_vary(axis_name)?;
        }
        // A mean over a manual mesh axis is the mesh-form sum divided by the axis size, exactly as JAX's `pmean` is
        // `psum(x) / n` at the wrapper level with no `pmean` primitive.
        let operation_kind = if kind == ParallelReductionKind::Mean { ParallelReductionKind::Sum } else { kind };
        let operation = ParallelReduceOperation::new(axis_name.to_string(), operation_kind).with_mesh(mesh);
        let mut outputs = context.bind(operation, Vec::new(), &[input])?;
        check_count!("output", outputs, 1, ProgramError);
        let output = outputs.remove(0);
        if kind != ParallelReductionKind::Mean {
            return Ok(output);
        }
        let output_type = output.r#type().into_owned();
        let divisor = Array::scalar(size as f64)?
            .convert_element_type(output_type.data_type())?
            .transfer_to_memory(output_type.memory())?;
        let mut divisors = context.bind(ConstantOperation::new(divisor), Vec::new(), &[])?;
        check_count!("output", divisors, 1, ProgramError);
        let inputs = ManualVariationAlignment::align_manual_variation(&[output, divisors.remove(0)])?;
        let mut quotients = context.bind(DivOperation::<ArrayType>::new(), Vec::new(), &inputs)?;
        check_count!("output", quotients, 1, ProgramError);
        Ok(quotients.remove(0))
    }

    fn parallel_reduce_with_axis_index_groups(
        &self,
        axis_name: &str,
        kind: ParallelReductionKind,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let operation = ParallelReduceOperation::grouped(axis_name.to_string(), kind, axis_size, axis_index_groups)?;
        let mut outputs = context.bind(operation, Vec::new(), std::slice::from_ref(self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, Dimension, DimensionBounds, DimensionType,
        DimensionValue, DimensionVariable, LogicalMesh, MeshAxis, MeshAxisType, RaggedAxis, Shape, Sharding,
    };
    use crate::batching::{BatchAxisSpecification, batch};
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::differentiation::differentiate_at;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, PartialValue};
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, ValueProjection};
    use crate::tracing::TracingContext;

    use super::*;

    /// Creates an active batching frame binding the named axis `"i"` over an eager parent whose operation family
    /// contains every operation the collective batching rule may bind (notably constants and broadcasts for `Mean`).
    fn batching_context(
        axis_size: usize,
    ) -> BatchingContext<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy> {
        BatchingContext::new(EagerContext::new(), axis_size).with_axis_name("i".to_string())
    }

    /// Creates a two-axis manual mesh together with the invariant and varying (over `"m"`) types of one local scalar.
    fn mesh_scalar_types() -> (LogicalMesh, ArrayType, ArrayType) {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("m", 4, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("outer", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 0);
        let invariant = ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap();
        let varying = ArrayType::scalar(DataType::F32)
            .with_sharding(sharding.with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        (mesh, invariant, varying)
    }

    /// Binds `"m"` as a checked manual mesh axis of `mesh` for [`TracingContext::trace_with_named_axes`].
    fn checked_mesh_axis(mesh: &LogicalMesh) -> Vec<(String, NamedAxis)> {
        let size = mesh.axis_size("m").unwrap();
        vec![("m".to_string(), NamedAxis::Mesh { axis: 0, size, mesh: mesh.clone() })]
    }

    #[test]
    fn test_parallel_reduce_with_mesh() {
        let (mesh, _, _) = mesh_scalar_types();
        let operation = ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Sum);
        assert_eq!(operation.mesh(), None);
        assert_eq!(operation.to_string(), "parallel_sum [axis_name=\"m\"]");
        let operation = operation.with_mesh(mesh.clone());
        assert_eq!(operation.mesh(), Some(&mesh));
        assert_eq!(operation.to_string(), "parallel_sum [axis_name=\"m\", mesh=['m'=4:manual, 'outer'=2:manual]]");
        let operation = ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Max).with_mesh(mesh);
        assert_eq!(operation.to_string(), "parallel_max [axis_name=\"m\", mesh=['m'=4:manual, 'outer'=2:manual]]");
    }

    #[test]
    fn test_parallel_reduce_mesh_type_inference() {
        let (mesh, invariant, varying) = mesh_scalar_types();
        let sharding = invariant.sharding().unwrap().clone();
        for kind in [ParallelReductionKind::Sum, ParallelReductionKind::Max] {
            let name = kind.name();
            let operation = ParallelReduceOperation::new("m".to_string(), kind).with_mesh(mesh.clone());
            assert_eq!(operation.infer_output_types(&[varying.clone()], &[]), Ok(vec![invariant.clone()]));
            assert_eq!(operation.fold(&[varying.clone()], &[]), Ok(None));
            let input = invariant
                .clone()
                .with_sharding(sharding.clone().with_varying_manual_axes(["m", "outer"]).unwrap())
                .unwrap();
            let expected = invariant
                .clone()
                .with_sharding(sharding.clone().with_varying_manual_axes(["outer"]).unwrap())
                .unwrap();
            assert_eq!(operation.infer_output_types(&[input.clone()], &[]), Ok(vec![expected]));
            // Ordinary and grouped reductions preserve the input variation.
            assert_eq!(
                ParallelReduceOperation::new("m".to_string(), kind).infer_output_types(&[input.clone()], &[]),
                Ok(vec![input.clone()])
            );
            let grouped =
                ParallelReduceOperation::grouped("m".to_string(), kind, 4, vec![vec![0, 1], vec![2, 3]]).unwrap();
            assert_eq!(grouped.mesh(), None);
            assert_eq!(grouped.infer_output_types(&[input.clone()], &[]), Ok(vec![input]));
            assert_eq!(
                grouped.with_mesh(mesh.clone()).infer_output_types(&[varying.clone()], &[]),
                Err(TypeError::invalid(format!("`{name}` over a manual mesh axis must not use axis index groups"))),
            );
            assert_eq!(
                operation.infer_output_types(&[invariant.clone()], &[]),
                Err(TypeError::invalid(format!(
                    "`{name}` input must vary over manual axis `m`; pass an invariant value through `parallel_vary` \
                     first so that every copy is counted"
                ))),
            );
            assert_eq!(
                operation.infer_output_types(&[ArrayType::scalar(DataType::F32)], &[]),
                Err(TypeError::invalid(format!("`{name}` input must carry a mesh containing manual axis `m`"))),
            );
            for sharding in [
                sharding.clone().with_unreduced_axes(["m"]).unwrap(),
                sharding.clone().with_reduced_axes(["m"]).unwrap(),
            ] {
                let input = ArrayType::scalar(DataType::F32).with_sharding(sharding).unwrap();
                assert_eq!(
                    operation.infer_output_types(&[input], &[]),
                    Err(TypeError::invalid(format!("`{name}` axis `m` must not carry reduction state"))),
                );
            }
            let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Manual).unwrap()]).unwrap();
            assert_eq!(
                ParallelReduceOperation::new("m".to_string(), kind)
                    .with_mesh(other_mesh)
                    .infer_output_types(&[varying.clone()], &[]),
                Err(TypeError::invalid(format!("`{name}` input mesh does not match the operation mesh"))),
            );
            let explicit = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
            assert_eq!(
                ParallelReduceOperation::new("m".to_string(), kind)
                    .with_mesh(explicit)
                    .infer_output_types(&[varying.clone()], &[]),
                Err(TypeError::invalid(format!("`{name}` mesh axis `m` must be manual"))),
            );
            // Reduced state along other axes is preserved.
            let reduced = sharding.clone().with_reduced_axes(["outer"]).unwrap();
            let expected = ArrayType::scalar(DataType::F32).with_sharding(reduced.clone()).unwrap();
            let input = expected.clone().with_sharding(reduced.with_varying_manual_axes(["m"]).unwrap()).unwrap();
            assert_eq!(operation.infer_output_types(&[input], &[]), Ok(vec![expected]));
        }
        // A sum commutes with summing partial contributions along other unreduced axes; a maximum does not.
        let unreduced = sharding.clone().with_unreduced_axes(["outer"]).unwrap();
        let expected = ArrayType::scalar(DataType::F32).with_sharding(unreduced.clone()).unwrap();
        let input = expected.clone().with_sharding(unreduced.with_varying_manual_axes(["m"]).unwrap()).unwrap();
        assert_eq!(
            ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Sum)
                .with_mesh(mesh.clone())
                .infer_output_types(&[input.clone()], &[]),
            Ok(vec![expected])
        );
        assert_eq!(
            ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Max)
                .with_mesh(mesh.clone())
                .infer_output_types(&[input], &[]),
            Err(TypeError::invalid("`parallel_max` does not support unreduced inputs")),
        );
        assert_eq!(
            ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Mean)
                .with_mesh(mesh)
                .infer_output_types(&[varying], &[]),
            Err(TypeError::invalid(
                "`parallel_mean` over a manual mesh axis is staged as a `parallel_sum` divided by the axis size",
            )),
        );
    }

    #[test]
    fn test_parallel_reduce_mesh_interpretation() {
        let (mesh, _, _) = mesh_scalar_types();
        assert!(matches!(
            ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Sum).with_mesh(mesh).interpret(
                &EagerContext::<Array>::new(), &EmptyRegionDriver, &[Array::scalar(2.0_f32).unwrap()],
            ),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`parallel_sum` over manual mesh axis `m` requires an execution backend owning the mesh",
        ));
    }

    #[test]
    fn test_parallel_reduce_mesh_partial_evaluation() {
        let (mesh, invariant, varying) = mesh_scalar_types();
        let operation = ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Sum).with_mesh(mesh);
        // A known input under an eager parent residualizes the operation, which has no per-item value.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(varying.clone());
        let output = builder.add_instruction(operation.clone(), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(Array::from_elements(varying.clone(), &[2.0_f32]).unwrap())])
            .unwrap();
        assert_eq!(evaluation.program().output_types(), vec![invariant.clone()]);
        assert_eq!(evaluation.program().instructions().len(), 1);
        assert!(matches!(evaluation.program().instructions()[0].operation(), ArrayOperation::ParallelReduce(operation)
            if operation.mesh().is_some()));
        assert!(evaluation.outputs()[0].is_unknown());

        let eager = PartialEvaluationContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let input = PartialEvaluationValue::known(ArrayIrValue::Array(
            Array::from_elements(varying.clone(), &[2.0_f32]).unwrap(),
        ));
        let outputs = ArrayIrOperation::Array(ArrayOperation::ParallelReduce(operation.clone()))
            .partially_evaluate(&eager, &EmptyRegionDriver, &[input])
            .unwrap();
        assert!(outputs[0].is_unknown());
        assert_eq!(outputs[0].r#type().as_ref(), &invariant.clone().into());

        // A known input under a staging parent stays known: the operation is staged into the parent trace.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = PartialEvaluationValue::known(trace.input(varying));
        let context = PartialEvaluationContext::new(trace);
        let outputs = operation.partially_evaluate(&context, &EmptyRegionDriver, &[input]).unwrap();
        assert!(outputs[0].is_known());
        assert_eq!(outputs[0].r#type().as_ref(), &invariant);
    }

    #[test]
    fn test_parallel_reduce_mesh_batching() {
        let (mesh, _, _) = mesh_scalar_types();
        let operation =
            ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Sum).with_mesh(mesh.clone());
        // A batch level binding an unrelated axis forwards the operation to its parent, preserving mapped and ragged
        // axes.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let packed = trace.input(packed_type);
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let input = ArrayBatch::new(packed, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents, length, vec![0])])
            .unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace, 2).with_axis_name("i".to_string());
        let outputs =
            operation.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)).unwrap().into_parts().0;
        assert_eq!(outputs[0].batch_axis(), input.batch_axis());
        assert_eq!(outputs[0].ragged_axes(), input.ragged_axes());
        assert_eq!(outputs[0].value().r#type().sharding().unwrap(), &Sharding::replicated(mesh.clone(), 2),);
        // A batch level binding the operation's own axis cannot stand in for the manual mesh axis.
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(TracingContext::<Array, ArrayOperation<Array>>::new(), 2)
                .with_axis_name("m".to_string());
        assert!(matches!(
            operation.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`parallel_sum` over a manual mesh axis cannot bind a named batch axis",
        ));
    }

    #[test]
    fn test_parallel_reduce_mesh_differentiation() {
        let (mesh, invariant, varying) = mesh_scalar_types();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_reduce("m", ParallelReductionKind::Sum),
            varying,
            checked_mesh_axis(&mesh),
        )
        .unwrap();
        let differentiated = program.to_flat_program().jvp().unwrap();
        assert_eq!(differentiated.output_types(), vec![invariant.clone(), invariant.clone()]);
        assert_eq!(
            differentiated
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["parallel_sum", "parallel_sum"]
        );
        // A structural-zero tangent keeps the operation's output type without staging a collective.
        let zero_tangent = program.to_flat_program().jvp_with_respect_to(&[]).unwrap();
        assert_eq!(zero_tangent.output_types(), vec![invariant.clone(), invariant]);
        assert_eq!(
            zero_tangent
                .instructions()
                .iter()
                .filter(|instruction| instruction.operation().name() == "parallel_sum")
                .count(),
            1
        );
    }

    #[test]
    fn test_parallel_reduce_mesh_transposition() {
        let (mesh, invariant, varying) = mesh_scalar_types();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(varying.clone());
        let operation = ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Sum).with_mesh(mesh);
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(transposed.input_types(), vec![invariant]);
        assert_eq!(transposed.output_types(), vec![varying]);
        assert_eq!(
            transposed
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["parallel_vary"]
        );
        let twice = transposed.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(twice.to_string(), program.to_string());
    }

    #[test]
    fn test_parallel_reduce_max_over_manual_axis() {
        let (mesh, _, varying) = mesh_scalar_types();
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_reduce("m", ParallelReductionKind::Max)?.parallel_vary("m"),
            varying.clone(),
            checked_mesh_axis(&mesh),
        )
        .unwrap();
        assert_eq!(output, varying);
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["parallel_max", "parallel_vary"]
        );
        assert!(matches!(program.instructions()[0].operation(), ArrayOperation::ParallelReduce(operation)
            if operation.mesh() == Some(&mesh)));
    }

    #[test]
    fn test_parallel_reduce_sum_over_manual_axis() {
        let (mesh, invariant, varying) = mesh_scalar_types();
        // An invariant input is first made varying so that every copy is counted.
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_reduce("m", ParallelReductionKind::Sum),
            invariant.clone(),
            checked_mesh_axis(&mesh),
        )
        .unwrap();
        assert_eq!(output, invariant);
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["parallel_vary", "parallel_sum"]
        );
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_reduce("m", ParallelReductionKind::Sum),
            varying,
            checked_mesh_axis(&mesh),
        )
        .unwrap();
        assert_eq!(output, invariant);
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["parallel_sum"]
        );
        assert!(matches!(program.instructions()[0].operation(), ArrayOperation::ParallelReduce(operation)
            if operation.mesh() == Some(&mesh)));
    }

    #[test]
    fn test_parallel_reduce_constant_and_mean_over_manual_axis() {
        let (mesh, invariant, _) = mesh_scalar_types();
        // A value without a sharding is first placed on the mesh, then made varying.
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_reduce("m", ParallelReductionKind::Sum),
            ArrayType::scalar(DataType::F32),
            checked_mesh_axis(&mesh),
        )
        .unwrap();
        assert_eq!(output, invariant);
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["broadcast", "parallel_vary", "parallel_sum"]
        );
        // A mean is the mesh-form sum divided by the axis size.
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_reduce("m", ParallelReductionKind::Mean),
            invariant.clone(),
            checked_mesh_axis(&mesh),
        )
        .unwrap();
        assert_eq!(output, invariant);
        let names = program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>();
        assert_eq!(names.iter().filter(|name| **name == "parallel_sum").count(), 1);
        assert!(!names.contains(&"parallel_mean"));
        assert_eq!(names.last(), Some(&"div"));
        assert!(matches!(program.instructions()[1].operation(), ArrayOperation::ParallelReduce(operation)
            if operation.kind() == ParallelReductionKind::Sum && operation.mesh() == Some(&mesh)));
    }

    #[test]
    fn test_parallel_sum_and_max_mask_ragged_padding_and_reduce_extents_with_max() -> Result<(), BatchingError> {
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        for kind in [ParallelReductionKind::Sum, ParallelReductionKind::Max] {
            let trace = TraceContext::new();
            let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
            let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
            let batch_extent = trace.input(DimensionType::from(items.clone()).into());
            let packed = trace.input(
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(items.clone()), Dimension::Static(3)]),
                )
                .into(),
            );
            let extents =
                trace.input(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(items)])).into());
            let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                ProjectedContext::new(trace),
                batch_extent,
            )
            .with_axis_name("i".to_string());
            let input = ArrayBatch::new(packed.into_projected()?, BatchAxis::new(0))?
                .with_ragged_axes(vec![RaggedAxis::new(1, extents.into_projected()?, length.clone(), vec![0])])?;

            let output = ParallelReduceOperation::new("i".to_string(), kind)
                .batch(&context, &EmptyRegionDriver, &[input])?
                .into_parts()
                .0
                .remove(0);

            assert_eq!(output.batch_axis(), BatchAxis::replicated());
            assert_eq!(output.ragged_axes().len(), 1);
            assert_eq!(output.ragged_axes()[0].axis(), 0);
            assert_eq!(output.ragged_axes()[0].dimension(), &length);
            assert!(output.ragged_axes()[0].extent_axes().is_empty());
            assert_eq!(output.ragged_axes()[0].extents().r#type().rank(), 0);

            let operation_names = context
                .parent()
                .parent()
                .builder()
                .borrow()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>();
            let expected_identity_operation = if kind == ParallelReductionKind::Sum { "zero_like" } else { "constant" };
            assert!(operation_names.contains(&expected_identity_operation));
            assert!(operation_names.contains(&"select"));
            assert_eq!(operation_names.iter().filter(|name| name.starts_with("reduce_")).count(), 2);
        }
        Ok(())
    }

    #[test]
    fn test_parallel_sum_and_max_numerically_mask_ragged_padding() -> Result<(), BatchingError> {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let parent = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(parent),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("i".to_string());

        let sum_input = ArrayBatch::new(Array::matrix(2, 3, vec![1.0_f32, 100.0, 100.0, 2.0, 3.0, 100.0]).unwrap(), 0)?
            .with_ragged_axes(vec![RaggedAxis::new(
                1,
                Array::vector(vec![1_i32, 2]).unwrap(),
                length.clone(),
                vec![0],
            )])?;
        let sum = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Sum)
            .batch(&context, &EmptyRegionDriver, &[sum_input])?
            .into_parts()
            .0
            .remove(0);
        assert_eq!(sum.batch_axis(), BatchAxis::replicated());
        assert_eq!(sum.value(), &Array::vector(vec![3.0_f32, 3.0, 0.0]).unwrap());
        assert_eq!(sum.ragged_axes()[0].extents(), &Array::scalar(2_i32).unwrap());

        let max_input =
            ArrayBatch::new(Array::matrix(2, 3, vec![-5.0_f32, 100.0, 100.0, -3.0, -4.0, 100.0]).unwrap(), 0)?
                .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 2]).unwrap(), length, vec![0])])?;
        let maximum = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Max)
            .batch(&context, &EmptyRegionDriver, &[max_input])?
            .into_parts()
            .0
            .remove(0);
        assert_eq!(maximum.batch_axis(), BatchAxis::replicated());
        assert_eq!(maximum.value(), &Array::vector(vec![-3.0_f32, -4.0, f32::NEG_INFINITY]).unwrap());
        assert_eq!(maximum.ragged_axes()[0].extents(), &Array::scalar(2_i32).unwrap());
        Ok(())
    }

    #[test]
    fn test_parallel_reduce_passes_through_replicated_input() {
        let input = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let outputs = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Sum)
            .batch(&batching_context(3), &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().to_f64s(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parallel_reduce_over_unbound_axis_is_rejected() {
        use crate::arrays::{Array, ArrayOperation};
        use crate::axes::AxisError;
        use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer};
        use crate::contexts::EagerContext;

        // The batch binds the axis `"i"`, but the collective names `"j"`, which no enclosing transform binds. Rather
        // than silently acting as identity (the pre-validation behavior), staging the collective fails fast with
        // `AxisError::UnboundAxisName`, matching JAX's error for a collective over an unbound axis name. The error
        // rides the `ProgramError::Custom` channel as a `BatchingError::Axis` and is re-typed at the public `batch`
        // boundary, so the surfaced error is exactly that variant.
        let result: Result<Array, BatchingError> = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                item.parallel_reduce("j", ParallelReductionKind::Sum)
            },
            Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("i"),
        );
        assert_eq!(result.unwrap_err(), BatchingError::Axis(AxisError::UnboundAxisName { name: "j".to_string() }));
    }

    #[test]
    fn test_grouped_reduction_collective_validates_and_renders_partition() {
        let operation = ParallelReduceOperation::grouped(
            "x".to_string(),
            ParallelReductionKind::Mean,
            4,
            vec![vec![0, 2], vec![3, 1]],
        )
        .unwrap();
        assert_eq!(operation.axis_size(), Some(4));
        assert_eq!(operation.group_size(), Some(2));
        assert_eq!(operation.axis_index_groups(), Some([vec![0, 2], vec![3, 1]].as_slice()));
        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::F32)], &[]).unwrap(),
            vec![ArrayType::scalar(DataType::F32)],
        );
        assert_eq!(
            operation.to_string(),
            "parallel_mean [axis_name=\"x\", axis_size=4, axis_index_groups=[[0, 2], [3, 1]]]",
        );

        assert!(matches!(
            ParallelReduceOperation::grouped(
                "x".to_string(),
                ParallelReductionKind::Sum,
                4,
                vec![vec![0, 1], vec![1, 2]]
            ),
            Err(TypeError::Invalid { .. }),
        ));
    }

    #[test]
    fn test_parallel_reduce_sum_reduces_along_the_batch_axis() {
        // Mapped input shape [3] at axis 0: per-item scalar. A `Sum` reduction collapses the batch axis to a
        // replicated scalar holding the total.
        let input = {
            let value = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let outputs = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Sum)
            .batch(&batching_context(3), &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().to_f64s(), vec![6.0]);
    }

    #[test]
    fn test_parallel_reduce_max_reduces_along_the_batch_axis() {
        let input = {
            let value = Array::vector(vec![1.0, 4.0, 2.0]).unwrap();
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let outputs = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Max)
            .batch(&batching_context(3), &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().to_f64s(), vec![4.0]);
    }

    #[test]
    fn test_parallel_reduce_mean_divides_by_batch_size() {
        // Per-item scalar input of shape [3] mapped at axis 0. A `Mean` reduction returns the mean of the three batch
        // items as a replicated scalar, exercising the `1 / N` factor that distinguishes it from `Sum`. The batching
        // frame binds the axis name `"data"` to show the rule matches on the collective's own axis name rather than a
        // fixture default.
        let input = {
            let value = Array::vector(vec![2.0, 4.0, 6.0]).unwrap();
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3)
            .with_axis_name("data".to_string());
        let outputs = ParallelReduceOperation::new("data".to_string(), ParallelReductionKind::Mean)
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        let values = outputs[0].value().to_f64s();
        assert_eq!(values.len(), 1);
        let delta = (values[0] - 4.0).abs();
        assert!(delta < 1e-9, "expected parallel_mean = 4.0, got {}", values[0]);
    }

    #[test]
    fn test_parallel_mean_batching_rejects_ragged_operands_without_a_denominator_definition() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![1.0_f32; 6]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 3]).unwrap(), variable, vec![0])])
            .unwrap();

        assert_eq!(
            ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Mean).batch(
                &batching_context(2),
                &EmptyRegionDriver,
                &[input]
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_mean` does not define a denominator for bounded ragged inputs".to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_reduce_mean_value_and_grad_through_vmap_carries_the_inverse_batch_size() {
        use crate::arrays::Array;
        use crate::batching::BatchAxisSpecification;

        // `g(x) = parallel_mean_i(x)`: the vmapped `parallel_mean` over the mapped axis `"i"` consumes that axis,
        // producing the replicated mean `M = (1/N)·Σ_j x_j`. Reverse mode pulls the scalar ones cotangent back
        // through the self-adjoint `parallel_mean`, which carries the `1/N` factor, so `∂g/∂x_i = 1/N` for every
        // input. With `x = [1, 2, 3]` (so `N = 3`) the value is `2` and the gradient is `[1/3, 1/3, 1/3]`,
        // witnessing the `1/N` scaling that distinguishes `parallel_mean` from `parallel_sum`.
        let (value, gradient) = differentiate_at(Array::vector(vec![1.0, 2.0, 3.0]).unwrap())
            .value_and_gradient(|x| {
                let mean = batch(
                    |item| item.parallel_reduce("i", ParallelReductionKind::Mean),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("i"),
                )
                .unwrap();
                mean
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![2.0]);
        assert_eq!(gradient.to_f64s(), vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    }

    #[test]
    fn test_nested_batch_named_axes_route_collectives_to_matching_level() {
        // The inner `parallel_sum` targets the *outer* named axis, so each inner batch item must reduce over the
        // outer batch items: column sums of [[1, 2], [3, 4]].
        let x = Array::from_elements::<f64>(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
            &[1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let output: Array = batch(
            |row| {
                Ok(batch(
                    |scalar| scalar.parallel_reduce("outer", ParallelReductionKind::Sum),
                    row,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("inner"),
                )?)
            },
            x,
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("outer"),
        )
        .unwrap();

        assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])));
        assert_eq!(output.to_f64s(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_parallel_reduce_sum_value_and_grad_through_vmap_re_sums_the_cotangent() {
        use crate::arrays::Array;
        use crate::batching::BatchAxisSpecification;

        // `g(x) = parallel_sum_i(x)`: the vmapped `parallel_sum` over the mapped axis `"i"` consumes that axis,
        // producing the replicated total `S = Σ_j x_j`. Reverse mode pulls the scalar ones cotangent back through
        // the self-adjoint `parallel_sum`, which re-broadcasts the cotangent across the batch items, giving
        // `∂g/∂x_i = 1` for every input. With `x = [1, 2, 3]` the value is `6` and the gradient is `[1, 1, 1]`.
        let (value, gradient) = differentiate_at(Array::vector(vec![1.0, 2.0, 3.0]).unwrap())
            .value_and_gradient(|x| {
                let total = batch(
                    |item| item.parallel_reduce("i", ParallelReductionKind::Sum),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("i"),
                )
                .unwrap();
                total
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![6.0]);
        assert_eq!(gradient.to_f64s(), vec![1.0, 1.0, 1.0]);
    }
}
