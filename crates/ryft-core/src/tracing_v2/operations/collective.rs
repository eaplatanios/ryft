use std::fmt::{Debug, Display};
use std::ops::Mul;

use crate::axes::{AxisError, NamedAxes};
use crate::batching::BatchingError;
use crate::contexts::Domain;
use crate::contexts::{Context, EagerContext};
use crate::differentiation::TransposableOperation;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::constants::{Fill, FillOperation, IotaOperation};
use crate::operations::{Operation, OperationFormatter};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::DifferentiationDual;
use crate::tracing_v2::differentiation::DifferentiableOperation;
use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
use crate::types::{ArrayType, DataType, TypeError, Typed};

/// Kind of collective performed by a [`CollectiveOperation`].
///
/// Collectives operate on a named axis, resolved against the active [`NamedAxes`] environment. When an enclosing
/// `batch` level binds the name (a [`NamedAxis::Batched`](crate::axes::NamedAxis) axis), the matching
/// [`BatchingContext`](crate::batching::BatchingContext) consumes the mapped batch axis at trace time.
/// When a `shard_map` manual region binds the name to a device mesh axis (a
/// [`NamedAxis::Mesh`](crate::axes::NamedAxis) axis), the collective stays in the staged body and lowers to a
/// cross-device `all_reduce` over that mesh axis. The operations described here mirror JAX's
/// `jax.lax.{psum, pmean, pmax}` family.
///
/// `PSum`/`PMean`/`PMax` reduce the named axis away, producing a result that is identical across all batch items or
/// device shards (replicated). `AllGather`-style gather variants are deferred until the machinery for shape-extending
/// collectives lands.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CollectiveKind {
    /// Sum reduction across the named axis (`jax.lax.psum`).
    PSum,

    /// Mean reduction across the named axis (`jax.lax.pmean`).
    PMean,

    /// Maximum reduction across the named axis (`jax.lax.pmax`).
    PMax,
}

impl CollectiveKind {
    /// Returns the canonical operation name suffix for this kind.
    pub fn name(self) -> &'static str {
        match self {
            Self::PSum => "psum",
            Self::PMean => "pmean",
            Self::PMax => "pmax",
        }
    }

    /// Returns the [`ReductionKind`] used to collapse the named axis.
    pub fn reduction_kind(self) -> ReductionKind {
        match self {
            Self::PSum | Self::PMean => ReductionKind::Sum,
            Self::PMax => ReductionKind::Max,
        }
    }
}

impl Display for CollectiveKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Value-level entry point for staging a collective operation.
///
/// The staged operation references an enclosing named-axis binder by name and the name is validated against the active
/// [`NamedAxes`] environment at staging time: an unbound name fails fast rather than silently acting as identity. A
/// name bound by an enclosing `batch` level is collapsed at trace time by
/// [`BatchableOperation::batch`](crate::batching::BatchableOperation::batch) (which reduces the mapped batch axis),
/// while a name bound to a device mesh axis by a `shard_map` manual region stays in the staged body program and lowers
/// to a cross-device `all_reduce` over that mesh axis.
pub trait Collective: Sized {
    /// Stages a collective of the given kind referencing axis `axis_name`, validating that the name is bound by an
    /// enclosing transform. Returns [`AxisError::UnboundAxisName`] (surfaced as
    /// [`BatchingError::Axis`](crate::batching::BatchingError::Axis) riding a [`ProgramError::Custom`] payload) when no
    /// enclosing binder binds `axis_name`.
    fn collective(&self, axis_name: &str, kind: CollectiveKind) -> Result<Self, ProgramError>;
}

/// Any context-carrying value applies a collective by validating the axis name against the active [`NamedAxes`]
/// environment and binding a [`CollectiveOperation`] through its own context: a staged tracer records the operation,
/// a batching tracer resolves the named axis against the batching context stack, and a JVP dual forwards to the
/// primal-side resolution. An unbound name fails fast with [`AxisError::UnboundAxisName`] rather than silently acting
/// as identity.
impl<V: Value> Collective for V
where
    V::DispatchDomain: Context + NamedAxes,
    <V::DispatchDomain as Domain>::Operation: From<CollectiveOperation>,
{
    fn collective(&self, axis_name: &str, kind: CollectiveKind) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        if context.named_axis(axis_name).is_none() {
            return Err(BatchingError::Axis(AxisError::UnboundAxisName { name: axis_name.to_string() }).into());
        }
        let mut outputs = context.bind(CollectiveOperation::new(axis_name.to_string(), kind), &[self.clone()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Primitive representing one named-axis collective operation.
///
/// [`CollectiveOperation`] is identity at the per-item level (the named axis does not exist in
/// per-item semantics) and collapses the mapped axis when invoked inside a
/// [`BatchingContext`](crate::batching::BatchingContext) whose
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches this collective's axis name. Under
/// nested `batch` levels, the traced batching rule below owns that decision: a matching level consumes the mapped
/// batch axis, while a non-matching level forwards the collective untouched to its parent context via
/// [`forward_collective_to_parent`], where the next level repeats the same name resolution. The value-level rule has
/// no level metadata to match against and always reduces the mapped axis, which corresponds to eager batching with a
/// single level.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CollectiveOperation {
    /// Axis name referenced by this collective. Matches the `axis_name` argument of an enclosing
    /// [`BatchingContext::new`](crate::batching::BatchingContext::new) call.
    axis_name: String,

    /// Kind of collective.
    kind: CollectiveKind,
}

impl CollectiveOperation {
    /// Creates a new [`CollectiveOperation`] with the supplied axis name and kind.
    #[inline]
    pub fn new(axis_name: String, kind: CollectiveKind) -> Self {
        Self { axis_name, kind }
    }

    /// Returns the axis name referenced by this collective.
    #[inline]
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }

    /// Returns the kind of collective.
    #[inline]
    pub fn kind(&self) -> CollectiveKind {
        self.kind
    }
}

impl Display for CollectiveOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for CollectiveOperation {
    #[inline]
    fn name(&self) -> &'static str {
        match self.kind {
            CollectiveKind::PSum => "psum",
            CollectiveKind::PMean => "pmean",
            CollectiveKind::PMax => "pmax",
        }
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        // The per-item operation is identity; the named axis only exists physically inside an
        // enclosing `BatchingContext` where the batching rule will collapse it.
        Ok(vec![input_types[0].clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("axis_name", format_args!("{:?}", self.axis_name)))
    }
}

impl<V: Value<Type = ArrayType>, C> InterpretableOperation<V, C> for CollectiveOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // Per-item interpretation is identity: the named axis does not exist in per-item semantics, so reducing across
        // it is a no-op. Staging a collective over an unbound axis is now rejected up front (see [`Collective`]), so an
        // enclosing binder always collapses the mapped axis through the batching rule before this per-item fallback
        // matters; it remains defined for a program interpreted directly, where a collective simply passes its operand
        // through per item.
        Ok(vec![inputs[0].clone()])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for CollectiveOperation where
    C::Operation: From<CollectiveOperation>
{
}

/// Re-stages this collective of the same axis name and kind on a single tracer operand, returning its single output.
///
/// Both the forward-mode (`jvp`) and the transpose rules below re-stage the collective on a tracer (the primal, the
/// tangent, or the output cotangent), which is exactly one operation with one input and one output.
fn stage_collective<C>(
    context: &C,
    axis_name: &str,
    kind: CollectiveKind,
    operand: &C::Value,
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayType>,
    C::Operation: From<CollectiveOperation>,
{
    let mut outputs =
        context.bind(CollectiveOperation::new(axis_name.to_string(), kind), std::slice::from_ref(operand))?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

/// Forward-mode (JVP) rule for [`CollectiveOperation`]. `PSum`/`PMean` are linear and self-adjoint, so the tangent is
/// the same collective applied to the operand tangent: `tangent_out = collective(input.tangent())`. A structural-zero
/// operand tangent is preserved as-is rather than staging a collective on a zero, keeping `collective(zero)` out of the
/// tangent program. `PMax` is non-linear and reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation)
/// error.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for CollectiveOperation
where
    C::Operation: Clone + From<CollectiveOperation>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        if matches!(self.kind, CollectiveKind::PMax) {
            return Err(ProgramError::UnsupportedOperation {
                message: "pmax differentiation is not yet supported".to_string(),
            });
        }
        let primal = stage_collective(context, &self.axis_name, self.kind, inputs[0].primal())?;
        // A collective of a structural zero stays a structural zero, keeping `collective(zero)` out of the tangent
        // program.
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
            MaybeZero::Value(tangent) => {
                MaybeZero::Value(stage_collective(context, &self.axis_name, self.kind, tangent)?)
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for [`CollectiveOperation`]. `psum`/`pmean` are self-adjoint, so the operand cotangent is the same
/// collective applied to the output cotangent. The single operand is linear (its [`PartialValue`] is
/// [`Unknown`](PartialValue::Unknown)); a known operand contributes no cotangent and so receives a structural zero.
/// `PMax` reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V, O> TransposableOperation<V, O> for CollectiveOperation
where
    V: Value<Type = ArrayType>,
    O: Clone + Operation<ArrayType> + From<CollectiveOperation>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        if matches!(self.kind, CollectiveKind::PMax) {
            return Err(ProgramError::UnsupportedOperation {
                message: "pmax transpose is not yet supported".to_string(),
            });
        }
        // A known (non-linear) operand contributes no cotangent.
        if inputs[0].is_known() {
            return Ok(vec![MaybeZero::Zero(inputs[0].r#type().into_owned())]);
        }
        match &outputs[0] {
            MaybeZero::Value(cotangent) => {
                let contribution = stage_collective(context, &self.axis_name, self.kind, cotangent)?;
                Ok(vec![MaybeZero::Value(contribution)])
            }
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().into_owned())]),
        }
    }
}

/// Value-level batching rule for eager backends, where the reduced value already carries its concrete data and a
/// `PMean`'s `1 / N` factor can be synthesized directly through the eager context's [`Fill`] capability.
///
/// Both this and the traced [`BatchingContext`](crate::batching::BatchingContext) rule below share
/// `collective_reduce_batch`; they differ only in which context produces the `PMean` factor.
impl<V, O> crate::batching::BatchableOperation<V, EagerContext<V, O>> for CollectiveOperation
where
    V: Value<Type = ArrayType> + Reduce + Mul<Output = V>,
    EagerContext<V, O>: Fill<f64, V>,
    O: Operation<ArrayType>,
    CollectiveOperation: InterpretableOperation<V, EagerContext<V, O>>,
{
    fn batch(
        &self,
        context: &EagerContext<V, O>,
        inputs: &[crate::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::batching::ArrayBatch<V>>, crate::batching::BatchingError> {
        collective_reduce_batch(self.kind, inputs, |factor_type, inverse_axis_size| {
            context.fill(&factor_type, inverse_axis_size)
        })
    }
}

/// Traced batching rule for [`Tracer`] values inside a [`BatchingContext`](
/// crate::batching::BatchingContext). This rule owns named-axis resolution: when the context's
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches this collective's axis name, the
/// mapped batch axis is consumed; otherwise the collective targets an outer `batch` level and is forwarded untouched
/// to the parent context via [`forward_collective_to_parent`].
///
/// The consuming arm shares `collective_reduce_batch` with the eager rule above but stages a `PMean`'s `1 / N`
/// rank-0 fill into the reduced value's own parent context (via [`StagingContext::stage_operation`]) instead of
/// synthesizing it through the [`Type`](crate::types::Type)-driven [`Fill`], which a [`Tracer`] cannot implement.
impl<C> crate::batching::BatchableOperation<<C as Domain>::Value, crate::batching::BatchingContext<C>>
    for CollectiveOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<CollectiveOperation> + From<FillOperation<ArrayType, f64>>,
    <C as Domain>::Value: Reduce + Mul<Output = <C as Domain>::Value>,
{
    fn batch(
        &self,
        context: &crate::batching::BatchingContext<C>,
        inputs: &[crate::batching::ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<crate::batching::ArrayBatch<<C as Domain>::Value>>, crate::batching::BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            let parent_operation = C::Operation::from(CollectiveOperation::new(self.axis_name.clone(), self.kind));
            return forward_collective_to_parent(context, parent_operation, inputs);
        }
        collective_reduce_batch(self.kind, inputs, |factor_type, inverse_axis_size| {
            // The `1 / N` rank-0 factor binds into the batching context's parent — interpreted eagerly under an eager
            // parent, staged into the enclosing trace under a staging parent.
            context
                .parent()
                .bind(FillOperation::new(factor_type, inverse_axis_size), &[])?
                .into_iter()
                .next()
                .ok_or(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }.into())
        })
    }
}

/// Re-stages a collective that targets a different (outer) named axis into the batching context's parent.
///
/// Under nested `batch` levels, a collective is consumed by the level whose
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches its axis name and must pass through
/// every inner level untouched: each inner batch item participates in the outer collective independently, so the
/// operands' mapped axes are preserved as-is on the forwarded outputs. The parent may itself be another
/// [`BatchingContext`](crate::batching::BatchingContext) — whose own rule dispatch repeats this name
/// resolution at the next level — or an ordinary tracing context. Batching rules for custom collective-like
/// operations should use this helper for their "not my axis" arm.
pub fn forward_collective_to_parent<C>(
    context: &crate::batching::BatchingContext<C>,
    parent_operation: C::Operation,
    inputs: &[crate::batching::ArrayBatch<<C as Domain>::Value>],
) -> Result<Vec<crate::batching::ArrayBatch<<C as Domain>::Value>>, crate::batching::BatchingError>
where
    C: Context<Type = ArrayType>,
{
    let parent_input_values: Vec<<C as Domain>::Value> = inputs.iter().map(|batch| batch.value().clone()).collect();
    let parent_outputs = context.parent().bind(parent_operation, parent_input_values.as_slice())?;
    check_count!("output", parent_outputs, inputs.len(), ProgramError);
    parent_outputs
        .into_iter()
        .zip(inputs.iter())
        .map(|(parent_value, input_batch)| {
            let physical_type = parent_value.r#type().into_owned();
            crate::batching::ArrayBatch::new(physical_type, parent_value, input_batch.batch_axis())
        })
        .collect()
}

/// Shared reduce-and-optionally-mean skeleton for [`CollectiveOperation`] batching, used by both the eager and traced
/// rules above. It collapses the mapped batch axis with the kind's [`ReductionKind`] and, for `PMean`, scales the
/// replicated result by `1 / N` using a `make_pmean_factor`-produced rank-0 factor (relying on implicit rank-0
/// broadcasting in the multiplication). Outside a matching batching context (no mapped axis), it is an identity
/// pass-through. The two callers differ only in `make_pmean_factor`: eager backends synthesize the factor directly,
/// while traced contexts stage it into their owning program.
fn collective_reduce_batch<V, MakePMeanFactor>(
    kind: CollectiveKind,
    inputs: &[crate::batching::ArrayBatch<V>],
    make_pmean_factor: MakePMeanFactor,
) -> Result<Vec<crate::batching::ArrayBatch<V>>, crate::batching::BatchingError>
where
    V: Value<Type = ArrayType> + Reduce + Mul<Output = V>,
    MakePMeanFactor: FnOnce(ArrayType, f64) -> Result<V, ProgramError>,
{
    check_count!("input", inputs, 1, ProgramError);
    let input = &inputs[0];
    let Some(batch_axis) = input.batch_axis().axis() else {
        // Outside any matching batching context: identity pass-through.
        return Ok(vec![input.clone()]);
    };
    // Reduce along the mapped batch axis with the corresponding reduction kind. The output is replicated: every
    // batch item sees the same reduced value, matching JAX's `psum`/`pmean`/`pmax` broadcast semantics.
    let mut output_value = input.value().clone().reduce(&[batch_axis], kind.reduction_kind());
    if matches!(kind, CollectiveKind::PMean) {
        // PMean divides the summed value by the batch size, which must be statically known to scale by `1 / N`.
        let inverse_axis_size = 1.0 / pmean_batch_size(input)? as f64;
        let factor_type = pmean_factor_type(output_value.r#type().data_type());
        output_value = make_pmean_factor(factor_type, inverse_axis_size)? * output_value;
    }
    let output_type = output_value.r#type().into_owned();
    Ok(vec![crate::batching::ArrayBatch::new(output_type, output_value, None)?])
}

/// Returns the static batch size for a `PMean` over the mapped batch axis of `input`, erroring when
/// the batch size is dynamic (a mean cannot be scaled by `1 / N` without a static `N`).
fn pmean_batch_size<V: Value<Type = ArrayType>>(
    input: &crate::batching::ArrayBatch<V>,
) -> Result<usize, crate::batching::BatchingError> {
    input.batch_size()?.ok_or_else(|| {
        crate::batching::BatchingError::UnsupportedOperation {
            message: "pmean requires a static batch size; the staged batch axis is dynamic".to_string(),
        }
        .into()
    })
}

/// Builds the rank-0 [`ArrayType`] of `data_type` used to hold a `PMean`'s `1 / N` factor.
fn pmean_factor_type(data_type: DataType) -> ArrayType {
    ArrayType::new(data_type, crate::types::Shape::scalar())
}

/// Canonical operation name for [`AxisIndexOperation`].
pub const AXIS_INDEX_OPERATION_NAME: &'static str = "axis_index";

/// Nullary primitive that produces the current batch item's or device shard's index along a named axis as a scalar
/// [`DataType::U64`] value — the ryft analogue of JAX's
/// [`axis_index`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.axis_index.html).
///
/// The [`AxisIndex`](crate::axes::AxisIndex) reader stages this operation uniformly for every axis kind; resolution
/// then depends on the enclosing binder. A *batched* axis is consumed by this operation's staged batching rule at the
/// `batch` level that binds it, which materializes the per-item index as an [`iota`](crate::operations::constants::Iota)
/// over the known batch size — so an `AxisIndexOperation` for a batched axis never survives into a staged body. A
/// *device mesh* axis has no such trace-time binder: its per-device coordinate is known only at execution time, so the
/// operation stays in the staged body and lowers inside a `shard_map` manual region to `partition_id`-based coordinate
/// arithmetic. Only mesh uses therefore reach interpretation, which is why this operation is *not* eagerly
/// interpretable and, having no operands, is [partially evaluated](PartiallyEvaluatableOperation) by residualizing
/// rather than folding (folding a nullary operation would try to interpret it).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AxisIndexOperation {
    /// Name of the device mesh axis whose per-shard index this operation produces.
    axis_name: String,
}

impl AxisIndexOperation {
    /// Creates a new [`AxisIndexOperation`] referencing the mesh axis `axis_name`.
    #[inline]
    pub fn new(axis_name: String) -> Self {
        Self { axis_name }
    }

    /// Returns the mesh axis name referenced by this operation.
    #[inline]
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }
}

impl Display for AxisIndexOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for AxisIndexOperation {
    #[inline]
    fn name(&self) -> &'static str {
        AXIS_INDEX_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![ArrayType::scalar(DataType::U64)])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, AXIS_INDEX_OPERATION_NAME)?
            .bracketed(|operation| operation.field("axis_name", format_args!("{:?}", self.axis_name)))
    }
}

impl<V: Value<Type = ArrayType>, C> InterpretableOperation<V, C> for AxisIndexOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        // A mesh axis index is a per-device coordinate that only exists during sharded execution; there is no eager
        // value to produce. It is lowered inside a `shard_map` manual region and never interpreted directly.
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "axis_index for the device mesh axis '{}' has no eager value; it is only defined inside a shard_map \
                manual region",
                self.axis_name,
            ),
        })
    }
}

/// Partial evaluation always residualizes an [`AxisIndexOperation`]: its value depends on the executing device, so it
/// is never a foldable constant even though it has no (known) inputs.
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for AxisIndexOperation
where
    C::Operation: From<AxisIndexOperation>,
{
    fn partially_evaluate(
        &self,
        evaluator: &mut crate::partial::PartialEvaluator<C>,
        inputs: &[crate::partial::PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<crate::partial::PartialEvaluationValue<C::Value>>, ProgramError> {
        evaluator.residualize(self.clone(), inputs)
    }
}

/// Forward-mode rule for [`AxisIndexOperation`]: the mesh index is an integer constant with no tangent, so the primal
/// is replayed and paired with a typed zero tangent.
impl<C: Context> DifferentiableOperation<C> for AxisIndexOperation
where
    C::Operation: Clone + From<AxisIndexOperation>,
    AxisIndexOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        crate::tracing_v2::differentiation::replay_zero_tangent(context, self.clone(), inputs)
    }
}

/// Transpose rule for [`AxisIndexOperation`]: it is a nullary constant, so it contributes no operand cotangents.
impl<V: Value<Type = ArrayType>, O: Operation<ArrayType>> TransposableOperation<V, O> for AxisIndexOperation {
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("output", outputs, 1, ProgramError);
        Ok(Vec::new())
    }
}

/// Eager batching rule for [`AxisIndexOperation`]. A batched axis index needs the batch size to materialize its
/// [`iota`](crate::operations::constants::Iota), but a nullary operation receives no batched operand to read that size
/// from, and eager batching carries no named-axis binder to consult. A batched axis index is therefore only defined
/// under staged batching (see the rule below); this eager rule exists to satisfy the operation family's eager batching
/// witness.
impl<V, O> crate::batching::BatchableOperation<V, EagerContext<V, O>> for AxisIndexOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType>,
{
    fn batch(
        &self,
        _context: &EagerContext<V, O>,
        _inputs: &[crate::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::batching::ArrayBatch<V>>, crate::batching::BatchingError> {
        Err(crate::batching::BatchingError::UnsupportedOperation {
            message: format!(
                "axis_index for axis '{}' is not defined under eager batching; it requires a staged batch or mesh \
                binder",
                self.axis_name,
            ),
        })
    }
}

/// Staged batching rule for [`AxisIndexOperation`], mirroring the collective rule above and, like it, deciding purely
/// from the context's [`axis_name`](crate::batching::BatchingContext::axis_name). When this level's axis
/// name matches, this level binds the axis, so the per-item index is materialized as the length-`size`
/// [`iota`](crate::operations::constants::Iota)`(0)` mapped on the level's batch axis (the size is this level's
/// [`axis_size`](crate::batching::BatchingContext::axis_size)). Otherwise the axis is bound elsewhere — an
/// outer `batch` level or a device mesh — so the operation is re-staged into the parent context (which repeats the same
/// resolution, ultimately materializing at the binding `batch` level or surviving into the staged body for a mesh axis)
/// and presented as replicated across this level's batch items. The name is validated by the
/// [`AxisIndex`](crate::axes::AxisIndex) reader before staging, so no name lookup is needed here.
impl<C> crate::batching::BatchableOperation<<C as Domain>::Value, crate::batching::BatchingContext<C>>
    for AxisIndexOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<IotaOperation<ArrayType>> + From<AxisIndexOperation>,
{
    fn batch(
        &self,
        context: &crate::batching::BatchingContext<C>,
        _inputs: &[crate::batching::ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<crate::batching::ArrayBatch<<C as Domain>::Value>>, crate::batching::BatchingError> {
        if context.axis_name() == Some(self.axis_name.as_str()) {
            // This level binds the axis: the per-item index is the length-`size` `iota(0)`, bound into the parent and
            // mapped on this level's batch axis (position 0). The mapped physical `[size]` dimension is then stripped
            // back to the per-item scalar `u64`.
            let size = context.axis_size();
            let physical_type =
                ArrayType::new(DataType::U64, crate::types::Shape::new(vec![crate::types::Size::Static(size)]));
            let mut index_vector = context.parent().bind(IotaOperation::new(physical_type.clone(), 0), &[])?;
            check_count!("output", index_vector, 1, ProgramError);
            Ok(vec![crate::batching::ArrayBatch::new(physical_type, index_vector.remove(0), Some(0))?])
        } else {
            // The axis is bound by an outer `batch` level or a device mesh: re-bind into the parent, which repeats the
            // resolution, and present the forwarded index as replicated across this level.
            let mut outputs = context.parent().bind(AxisIndexOperation::new(self.axis_name.clone()), &[])?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(vec![crate::batching::ArrayBatch::replicated(outputs.remove(0))])
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::batching::ArrayBatch;
    use crate::batching::BatchAxis;
    use crate::batching::BatchableOperation;
    use crate::contexts::EagerContext;
    use crate::tests::TestArray;

    use super::*;

    #[test]
    fn test_collective_psum_reduces_along_the_batch_axis() {
        // Mapped input shape [3] at axis 0: per-item scalar. PSum collapses the batch axis to a
        // replicated scalar holding the total.
        let input = {
            let value = TestArray::vector(vec![1.0, 2.0, 3.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let context = EagerContext::<TestArray, CollectiveOperation>::new();
        let outputs =
            CollectiveOperation::new("i".to_string(), CollectiveKind::PSum).batch(&context, &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values(), &[6.0]);
    }

    #[test]
    fn test_collective_pmax_reduces_along_the_batch_axis() {
        let input = {
            let value = TestArray::vector(vec![1.0, 4.0, 2.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let context = EagerContext::<TestArray, CollectiveOperation>::new();
        let outputs =
            CollectiveOperation::new("i".to_string(), CollectiveKind::PMax).batch(&context, &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values(), &[4.0]);
    }

    #[test]
    fn test_collective_passes_through_replicated_input() {
        let input = ArrayBatch::replicated(TestArray::vector(vec![1.0, 2.0, 3.0]));
        let context = EagerContext::<TestArray, CollectiveOperation>::new();
        let outputs =
            CollectiveOperation::new("i".to_string(), CollectiveKind::PSum).batch(&context, &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_collective_over_unbound_axis_is_rejected() {
        use crate::axes::AxisError;
        use crate::batching::Batch;
        use crate::batching::BatchingTracer;
        use crate::batching::{BatchAxis, BatchAxisSpecification};
        use crate::contexts::EagerContext;
        use crate::tests::TestArray;
        use crate::tracing_v2::ArrayOperation;

        // The batch binds the axis `"i"`, but the collective names `"j"`, which no enclosing transform binds. Rather
        // than silently acting as identity (the pre-validation behavior), staging the collective fails fast with
        // `AxisError::UnboundAxisName`, matching JAX's error for a collective over an unbound axis name. The error
        // rides the `ProgramError::Custom` channel as a `BatchingError::Axis` and is re-typed at the public `batch`
        // boundary, so the surfaced error is exactly that variant.
        let result: Result<TestArray, BatchingError> = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |item: BatchingTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                    item.collective("j", CollectiveKind::PSum)
                },
                TestArray::vector(vec![1.0, 2.0, 3.0]),
                BatchAxis::new(0),
                BatchAxis::replicated(),
                BatchAxisSpecification::named("i"),
            );
        assert_eq!(result.unwrap_err(), BatchingError::Axis(AxisError::UnboundAxisName { name: "j".to_string() }));
    }

    #[test]
    fn test_collective_psum_value_and_grad_through_vmap_re_sums_the_cotangent() {
        use crate::batching::Batch;
        use crate::batching::BatchAxisSpecification;
        use crate::contexts::EagerContext;
        use crate::tests::TestArray;
        use crate::tracing_v2::ArrayOperation;
        use crate::tracing_v2::{NestedTracer, value_and_grad};

        // `g(x) = psum_i(x)`: the vmapped `psum` over the mapped axis `"i"` consumes that axis, producing the
        // replicated total `S = Σ_j x_j`. Reverse mode pulls the scalar ones cotangent back through the
        // self-adjoint `psum`, which re-broadcasts the cotangent across the batch items, giving `∂g/∂x_i = 1`
        // for every input. With `x = [1, 2, 3]` the value is `6` and the gradient is `[1, 1, 1]`.
        let (value, gradient) = value_and_grad(
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
            |x: NestedTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                let context = x.context().clone();
                let total: NestedTracer<EagerContext<TestArray, ArrayOperation<TestArray>>> = Batch::batch(
                    &context,
                    |item| item.collective("i", CollectiveKind::PSum),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("i"),
                )
                .unwrap();
                total
            },
            TestArray::vector(vec![1.0, 2.0, 3.0]),
        )
        .unwrap();
        assert_eq!(value.values, vec![6.0]);
        assert_eq!(gradient.values, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_collective_pmean_value_and_grad_through_vmap_carries_the_inverse_batch_size() {
        use crate::batching::Batch;
        use crate::batching::BatchAxisSpecification;
        use crate::contexts::EagerContext;
        use crate::tests::TestArray;
        use crate::tracing_v2::ArrayOperation;
        use crate::tracing_v2::{NestedTracer, value_and_grad};

        // `g(x) = pmean_i(x)`: the vmapped `pmean` over the mapped axis `"i"` consumes that axis, producing the
        // replicated mean `M = (1/N)·Σ_j x_j`. Reverse mode pulls the scalar ones cotangent back through the
        // self-adjoint `pmean`, which carries the `1/N` factor, so `∂g/∂x_i = 1/N` for every input. With `x =
        // [1, 2, 3]` (so `N = 3`) the value is `2` and the gradient is `[1/3, 1/3, 1/3]`, witnessing the `1/N`
        // scaling that distinguishes `pmean` from `psum`.
        let (value, gradient) = value_and_grad(
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
            |x: NestedTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                let context = x.context().clone();
                let mean: NestedTracer<EagerContext<TestArray, ArrayOperation<TestArray>>> = Batch::batch(
                    &context,
                    |item| item.collective("i", CollectiveKind::PMean),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("i"),
                )
                .unwrap();
                mean
            },
            TestArray::vector(vec![1.0, 2.0, 3.0]),
        )
        .unwrap();
        assert_eq!(value.values, vec![2.0]);
        assert_eq!(gradient.values, vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    }
}
