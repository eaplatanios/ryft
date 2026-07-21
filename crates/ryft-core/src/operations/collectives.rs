//! Contains the named-axis [`CollectiveOperation`], which reduces a value across a named axis (`psum` / `pmean` /
//! `pmax`), together with its interpretation, partial-evaluation, batching, forward-mode differentiation, and
//! transposition rules. These are the analogues of
//! [JAX's parallel operators](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators) `jax.lax.psum`,
//! `jax.lax.pmean`, and `jax.lax.pmax`.
//!
//! Collectives reference an enclosing named-axis binder by name, validated against the active
//! [`NamedAxes`](crate::axes::NamedAxes) environment at staging time. A name bound by an enclosing `batch` level is
//! resolved at trace time by the operations' batching rules, which collapse or materialize the mapped batch axis at
//! the binding level, while a name bound to a device mesh axis by a `shard_map` manual region stays in the staged
//! body and lowers to cross-device collectives over that mesh axis.

use std::fmt::Display;
use std::ops::Mul;

use crate::axes::{AxisError, NamedAxes, NamedAxis};
use crate::backends::scalars::Scalar;
use crate::batching::{ArrayBatch, BatchableOperation, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::constants::{FillOperation, ZeroLike};
use crate::operations::manipulation::slicing::resized_output_sharding;
use crate::operations::manipulation::{Broadcast, Concatenate, Reshape, Slice, Transpose};
use crate::operations::math::{Reduce, ReductionKind};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::sharding::ShardingDimension;
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, DataType, Shape, Size};

// TODO(eaplatanios): Review this module.

/// Kind of collective performed by a [`CollectiveOperation`].
///
/// Collectives operate on a named axis, resolved against the active [`NamedAxes`] environment. When an enclosing
/// `batch` level binds the name (a [`NamedAxis::Batched`](crate::axes::NamedAxis) axis), the matching
/// [`BatchingContext`] consumes the mapped batch axis at trace time.
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

/// Primitive representing one named-axis collective operation.
///
/// [`CollectiveOperation`] is identity at the per-item level (the named axis does not exist in
/// per-item semantics) and collapses the mapped axis when invoked inside a
/// [`BatchingContext`] whose
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches this collective's axis name. Under
/// nested `batch` levels, the batching rule below owns that decision: a matching level consumes the mapped
/// batch axis, while a non-matching level forwards the collective untouched to its parent context via
/// [`forward_collective_to_parent`], where the next level repeats the same name resolution.
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

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
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

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for CollectiveOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // Per-item interpretation is identity: the named axis does not exist in per-item semantics, so reducing across
        // it is a no-op. Staging a collective over an unbound axis is now rejected up front (see `Collective`), so an
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

/// Batching rule for [`CollectiveOperation`]. This rule owns named-axis resolution: when the active context's
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches this collective's axis name, the mapped batch
/// axis is consumed; otherwise the collective targets an outer `batch` level (or a device mesh axis) and is forwarded
/// untouched to the parent context via [`forward_collective_to_parent`].
///
/// The consuming arm collapses the mapped axis through `collective_reduce_batch` and binds a `PMean`'s `1 / N`
/// rank-0 fill into the parent context — interpreted eagerly under an eager parent and staged into the enclosing
/// trace under a staging parent — so one rule serves eager and staged batching alike.
impl<C> BatchableOperation<C> for CollectiveOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<CollectiveOperation> + From<FillOperation<ArrayType, Scalar>>,
    <C as Domain>::Value: Reduce + Mul<Output = <C as Domain>::Value>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            let parent_operation = C::Operation::from(CollectiveOperation::new(self.axis_name.clone(), self.kind));
            return forward_collective_to_parent(context, parent_operation, inputs);
        }
        collective_reduce_batch(self.kind, inputs, |factor_type, inverse_axis_size| {
            // The `1 / N` rank-0 factor binds into the batching context's parent — interpreted eagerly under an eager
            // parent, staged into the enclosing trace under a staging parent.
            context
                .parent()
                .bind(FillOperation::new(factor_type, Scalar::from(inverse_axis_size)), Vec::new(), &[])?
                .into_iter()
                .next()
                .ok_or(ProgramError::InvalidOutputCount { expected: 1, actual: 0 })
        })
    }
}

/// Re-stages a collective that targets a different (outer) named axis into the batching context's parent.
///
/// Under nested `batch` levels, a collective is consumed by the level whose
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches its axis name and must pass through
/// every inner level untouched: each inner batch item participates in the outer collective independently, so the
/// operands' mapped axes are preserved as-is on the forwarded outputs. The parent may itself be another
/// [`BatchingContext`] — whose own rule dispatch repeats this name
/// resolution at the next level — or an ordinary tracing context. Batching rules for custom collective-like
/// operations should use this helper for their "not my axis" arm.
pub fn forward_collective_to_parent<C>(
    context: &BatchingContext<C>,
    parent_operation: C::Operation,
    inputs: &[ArrayBatch<<C as Domain>::Value>],
) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
{
    let parent_input_values: Vec<<C as Domain>::Value> = inputs.iter().map(|batch| batch.value().clone()).collect();
    let parent_outputs = context.parent().bind(parent_operation, Vec::new(), &parent_input_values)?;
    check_count!("output", parent_outputs, inputs.len(), ProgramError);
    parent_outputs
        .into_iter()
        .zip(inputs.iter())
        .map(|(parent_value, input_batch)| {
            let physical_type = parent_value.r#type().into_owned();
            ArrayBatch::new(physical_type, parent_value, input_batch.batch_axis())
        })
        .collect()
}

/// Shared reduce-and-optionally-mean skeleton for [`CollectiveOperation`] batching. It collapses the mapped batch
/// axis with the kind's [`ReductionKind`] and, for `PMean`, scales the replicated result by `1 / N` using a
/// `make_pmean_factor`-produced rank-0 factor (relying on implicit rank-0 broadcasting in the multiplication).
/// Outside a matching batching context (no mapped axis), it is an identity pass-through.
fn collective_reduce_batch<V, MakePMeanFactor>(
    kind: CollectiveKind,
    inputs: &[ArrayBatch<V>],
    make_pmean_factor: MakePMeanFactor,
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    V: Value<Type = ArrayType> + Reduce + Mul<Output = V>,
    MakePMeanFactor: FnOnce(ArrayType, f64) -> Result<V, ProgramError>,
{
    check_count!("input", inputs, 1, ProgramError);
    let input = &inputs[0];
    let Some(batch_axis) = input.batch_axis_position() else {
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
    Ok(vec![ArrayBatch::new(output_type, output_value, None)?])
}

/// Returns the static batch size for a `PMean` over the mapped batch axis of `input`, erroring when
/// the batch size is dynamic (a mean cannot be scaled by `1 / N` without a static `N`).
fn pmean_batch_size<V: Value<Type = ArrayType>>(input: &ArrayBatch<V>) -> Result<usize, BatchingError> {
    input.batch_size()?.ok_or_else(|| BatchingError::UnsupportedOperation {
        message: "pmean requires a static batch size; the staged batch axis is dynamic".to_string(),
    })
}

/// Builds the rank-0 [`ArrayType`] of `data_type` used to hold a `PMean`'s `1 / N` factor.
fn pmean_factor_type(data_type: DataType) -> ArrayType {
    ArrayType::new(data_type, Shape::scalar())
}

impl_differentiable_operation! {
    CollectiveOperation,
    /// Forward-mode (JVP) rule for [`CollectiveOperation`]. `PSum`/`PMean` are linear and self-adjoint, so the tangent
    /// is the same collective applied to the operand tangent: `tangent_out = collective(input.tangent())`. A
    /// structural-zero operand tangent is preserved as-is rather than staging a collective on a zero, keeping
    /// `collective(zero)` out of the tangent program. `PMax` is non-linear and reports an
    /// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<CollectiveOperation>,
    {
        |operation, context, _driver, inputs| {
            check_count!("input", inputs, 1, ProgramError);
            if matches!(operation.kind, CollectiveKind::PMax) {
                return Err(ProgramError::UnsupportedOperation {
                    message: "pmax differentiation is not yet supported".to_string(),
                }
                .into());
            }
            let primal =
                stage_collective(context, &operation.axis_name, operation.kind, inputs[0].primal())?;
            // A collective of a structural zero stays a structural zero, keeping `collective(zero)` out of the
            // tangent program.
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
                MaybeZero::Value(tangent) => {
                    MaybeZero::Value(stage_collective(context, &operation.axis_name, operation.kind, tangent)?)
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    /// Transpose rule for [`CollectiveOperation`]. `psum`/`pmean` are self-adjoint, so the operand cotangent is the
    /// same collective applied to the output cotangent. The single operand is linear (its [`PartialValue`] is
    /// [`Unknown`](PartialValue::Unknown)); a known operand contributes no cotangent and so receives a structural zero.
    /// `PMax` reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<ArrayType> + From<CollectiveOperation>,
    {
        |operation, context, _driver, inputs, outputs| {
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            if matches!(operation.kind, CollectiveKind::PMax) {
                return Err(ProgramError::UnsupportedOperation {
                    message: "pmax transpose is not yet supported".to_string(),
                }
                .into());
            }
            // A known (non-linear) operand contributes no cotangent.
            if inputs[0].is_known() {
                return Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]);
            }
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    let contribution = stage_collective(context, &operation.axis_name, operation.kind, cotangent)?;
                    Ok(vec![MaybeZero::Value(contribution)])
                }
                MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]),
            }
        }
    },
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
    let mut outputs = context.bind(
        CollectiveOperation::new(axis_name.to_string(), kind),
        Vec::new(),
        std::slice::from_ref(operand),
    )?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

/// Value-level entry point for staging a collective operation.
///
/// The staged operation references an enclosing named-axis binder by name and the name is validated against the active
/// [`NamedAxes`] environment at staging time: an unbound name fails fast rather than silently acting as identity. A
/// name bound by an enclosing `batch` level is collapsed at trace time by
/// [`BatchableOperation::batch`] (which reduces the mapped batch axis),
/// while a name bound to a device mesh axis by a `shard_map` manual region stays in the staged body program and lowers
/// to a cross-device `all_reduce` over that mesh axis.
pub trait Collective: Sized {
    /// Stages a collective of the given kind referencing axis `axis_name`, validating that the name is bound by an
    /// enclosing transform. Returns [`AxisError::UnboundAxisName`] (surfaced as
    /// [`BatchingError::Axis`] riding a [`ProgramError::Custom`] payload) when no
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
        let mut outputs =
            context.bind(CollectiveOperation::new(axis_name.to_string(), kind), Vec::new(), &[self.clone()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Resolves the size of the named axis bound by the active [`NamedAxes`] environment, failing fast with
/// [`AxisError::UnboundAxisName`] when no enclosing binder binds `axis_name`. The shape-changing collective
/// capabilities bake the resolved size into their operation payloads at staging time, because their output shapes
/// depend on it while [`Operation::infer_output_types`] only sees input types.
fn resolve_named_axis_size<C: NamedAxes>(context: &C, axis_name: &str) -> Result<usize, ProgramError> {
    match context.named_axis(axis_name) {
        Some(NamedAxis::Batched { size }) => Ok(size),
        Some(NamedAxis::Mesh { size, .. }) => Ok(size),
        None => Err(BatchingError::Axis(AxisError::UnboundAxisName { name: axis_name.to_string() }).into()),
    }
}

/// Validates the shared operand contract of the shape-changing collectives (exactly one operand with no unreduced
/// axes) and returns the operand's static dimensions.
fn shape_changing_collective_dimensions(
    operation_name: &str,
    input_types: &[ArrayType],
) -> Result<Vec<usize>, TypeError> {
    check_count!("input", input_types, 1, TypeError);
    if !input_types[0].unreduced_axes().is_empty() {
        return Err(TypeError { message: format!("'{operation_name}' does not support unreduced operands") });
    }
    let Some(shape) = input_types[0].static_shape() else {
        return Err(TypeError { message: format!("'{operation_name}' does not support dynamically shaped operands") });
    };
    Ok(shape.dimensions().to_vec())
}

/// Builds a shape-changing collective's output type from its operand and resized dimensions, carrying the operand
/// sharding through with the same per-dimension placement (the dimension count never changes).
fn shape_changing_collective_output_type(
    operation_name: &'static str,
    input_type: &ArrayType,
    output_dimensions: Vec<usize>,
) -> Result<ArrayType, TypeError> {
    let output_sizes = output_dimensions.into_iter().map(Size::Static).collect::<Vec<_>>();
    let sharding = resized_output_sharding(input_type, output_sizes.as_slice(), operation_name)?;
    let mut output_type = ArrayType::new(input_type.data_type(), Shape::new(output_sizes));
    output_type.sharding = sharding;
    Ok(output_type)
}

/// Interprets a shape-changing collective outside any binder: only the degenerate single-participant axis
/// (`axis_size == 1`) has defined per-item semantics (the identity), and any larger axis reports an error because
/// the other participants do not exist per item.
fn interpret_degenerate_collective<V: Clone>(
    operation_name: &str,
    axis_name: &str,
    axis_size: usize,
    inputs: &[V],
) -> Result<Vec<V>, ProgramError> {
    check_count!("input", inputs, 1, ProgramError);
    if axis_size != 1 {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "cannot interpret '{operation_name}' over axis '{axis_name}' of size {axis_size} without an \
                 enclosing binder",
            ),
        });
    }
    Ok(vec![inputs[0].clone()])
}

/// Shared operand preparation for the shape-changing collective batching rules at a matching `batch` level. It
/// validates the single-operand contract, realigns the mapped batch axis to the leading physical axis, and returns
/// the packed physical value together with its static dimensions (whose leading entry is the batch size).
///
/// A replicated operand is first materialized as `axis_size` identical batch items via [`ArrayBatch::broadcast`],
/// which yields the degenerate collective-of-a-replicated-value semantics for free: an `all_gather` of a replicated
/// value concatenates `axis_size` copies, a `psum_scatter` scatters the `axis_size`-fold sum, and so on. A mapped
/// batch axis whose size disagrees with the operation's resolved `axis_size` reports an error; the staging
/// capabilities resolve both from the same binder, so a mismatch indicates a hand-constructed operation.
fn shape_changing_collective_batch_operand<V>(
    operation_name: &str,
    axis_name: &str,
    axis_size: usize,
    inputs: &[ArrayBatch<V>],
) -> Result<(V, Vec<usize>), BatchingError>
where
    V: Value<Type = ArrayType> + Broadcast + Transpose,
{
    check_count!("input", inputs, 1, ProgramError);
    let input = if inputs[0].batch_axis().is_replicated() {
        inputs[0].broadcast(0, axis_size, ShardingDimension::Replicated)?
    } else {
        inputs[0].move_axis(0)?
    };
    // The operand is mapped at axis 0 by construction (a replicated operand was just broadcast onto a mapped axis),
    // so a missing batch size is impossible here; a dynamic batch axis errors through `batch_size` itself.
    let batch_size = input.batch_size()?.unwrap();
    if batch_size != axis_size {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "'{operation_name}' over axis '{axis_name}' resolved axis size {axis_size} but the mapped batch \
                 axis has size {batch_size}",
            ),
        });
    }
    let Some(shape) = input.value().r#type().static_shape() else {
        return Err(BatchingError::UnsupportedOperation {
            message: format!("'{operation_name}' batching requires statically shaped operands"),
        });
    };
    let dimensions = shape.dimensions().to_vec();
    Ok((input.into_value(), dimensions))
}

/// Implements the shared structure of the shape-changing collectives: the operation struct with its accessors, the
/// `Display`/`Operation` implementations (with payload-dependent output-shape inference provided as a closure over
/// the operand dimensions), degenerate interpretation, default partial evaluation, the linear forward-mode rule
/// (the tangent rides the same collective), and the value-level staging capability that resolves the named axis size
/// from the active [`NamedAxes`] environment. The batching rules are hand-written below the macro invocations
/// because each collective materializes the mapped batch axis differently.
macro_rules! shape_changing_collective {
    (
        $(#[$operation_documentation:meta])*
        operation = $operation:ident,
        name = $operation_name:ident = $name_literal:literal,
        $(#[$capability_documentation:meta])*
        capability = $capability:ident::$method:ident,
        fields = { $($(#[$field_documentation:meta])* $field:ident: $field_type:ty),* $(,)? },
        infer = |$infer_self:ident, $dimensions:ident| $infer:block $(,)?
    ) => {
        /// Canonical operation name for the operation.
        pub const $operation_name: &str = $name_literal;

        $(#[$operation_documentation])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $operation {
            /// Axis name referenced by this collective.
            axis_name: String,

            /// Number of participants along the named axis, resolved from the active [`NamedAxes`] environment when
            /// the operation is staged.
            axis_size: usize,

            $($(#[$field_documentation])* $field: $field_type,)*
        }

        impl $operation {
            /// Creates a new operation over the named axis with the provided resolved axis size.
            #[inline]
            pub fn new(axis_name: String, axis_size: usize, $($field: $field_type),*) -> Self {
                Self { axis_name, axis_size, $($field),* }
            }

            /// Returns the axis name referenced by this collective.
            #[inline]
            pub fn axis_name(&self) -> &str {
                &self.axis_name
            }

            /// Returns the number of participants along the named axis.
            #[inline]
            pub fn axis_size(&self) -> usize {
                self.axis_size
            }
        }

        impl Display for $operation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                Operation::<ArrayType>::render(self, formatter, 0)
            }
        }

        impl Operation<ArrayType> for $operation {
            #[inline]
            fn name(&self) -> &'static str {
                $operation_name
            }

            fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
                OperationFormatter::new(formatter, indentation, $operation_name)?.bracketed(|operation| {
                    operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
                    operation.field("axis_size", &self.axis_size)?;
                    $(operation.field(stringify!($field), format_args!("{:?}", &self.$field))?;)*
                    Ok(())
                })
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                let $dimensions = shape_changing_collective_dimensions($name_literal, input_types)?;
                let $infer_self = self;
                let output_dimensions: Vec<usize> = $infer?;
                Ok(vec![shape_changing_collective_output_type(
                    $operation_name,
                    &input_types[0],
                    output_dimensions,
                )?])
            }
        }

        impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for $operation {
            fn interpret<D: InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, ProgramError> {
                interpret_degenerate_collective($name_literal, &self.axis_name, self.axis_size, inputs)
            }
        }

        /// Partial evaluation defers to the default fold-or-residualize behavior of
        /// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
        impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for $operation where
            C::Operation: From<$operation>
        {
        }

        /// Forward-mode rule: the collective is linear, so the tangent rides the same collective. Structural-zero
        /// tangents stay symbolic, retyped to the output tangent type (the collective changes shapes).
        impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for $operation
        where
            C::Operation: From<$operation>,
        {
            fn jvp<D: DifferentiationDriver<C>>(
                &self,
                context: &C,
                _driver: &D,
                inputs: &[DifferentiationDual<C::Value>],
            ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
                check_count!("input", inputs, 1, ProgramError);
                let mut primal_outputs =
                    context.bind(self.clone(), Vec::new(), std::slice::from_ref(inputs[0].primal()))?;
                check_count!("output", primal_outputs, 1, ProgramError);
                let primal = primal_outputs.remove(0);
                let tangent = match inputs[0].tangent() {
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                    MaybeZero::Value(tangent) => {
                        let mut tangent_outputs =
                            context.bind(self.clone(), Vec::new(), std::slice::from_ref(tangent))?;
                        check_count!("output", tangent_outputs, 1, ProgramError);
                        MaybeZero::Value(tangent_outputs.remove(0))
                    }
                };
                Ok(vec![DifferentiationDual::new(primal, tangent)?])
            }
        }

        $(#[$capability_documentation])*
        pub trait $capability: Sized {
            /// Stages this collective over axis `axis_name`, resolving the axis size from the active [`NamedAxes`]
            /// environment and returning an [`AxisError::UnboundAxisName`] error when no enclosing binder binds the
            /// name.
            fn $method(&self, axis_name: &str, $($field: $field_type),*) -> Result<Self, ProgramError>;
        }

        impl<V: Value<Type = ArrayType>> $capability for V
        where
            V::DispatchDomain: Context + NamedAxes,
            <V::DispatchDomain as Domain>::Operation: From<$operation>,
        {
            fn $method(&self, axis_name: &str, $($field: $field_type),*) -> Result<Self, ProgramError> {
                let context = self.dispatch_domain();
                let axis_size = resolve_named_axis_size(&context, axis_name)?;
                let mut outputs = context.bind(
                    $operation::new(axis_name.to_string(), axis_size, $($field),*),
                    Vec::new(),
                    std::slice::from_ref(self),
                )?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(outputs.remove(0))
            }
        }
    };
}

shape_changing_collective! {
    /// [`Operation`] that concatenates every participant's operand along `concat_axis` across the named axis, so
    /// every participant receives the full concatenation — the analogue of
    /// [JAX's `all_gather`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_gather.html) with `tiled = True`
    /// and [StableHLO's `all_gather`](https://openxla.org/stablehlo/spec#all_gather). The output extends
    /// `concat_axis` by the axis size; all other dimensions are unchanged. The collective is linear and its
    /// transpose is [`PSumScatterOperation`] over the same axis and dimension. A matching `batch` level consumes the
    /// mapped batch axis by merging it item-major into `concat_axis`, replicating the gathered value across the
    /// batch items.
    operation = AllGatherOperation,
    name = ALL_GATHER_OPERATION_NAME = "all_gather",
    /// Value-level entry point for staging an [`AllGatherOperation`]. Refer to its documentation for the semantics
    /// and transform rules.
    capability = AllGather::all_gather,
    fields = {
        /// Axis of the operand along which the participants' values are concatenated.
        concat_axis: usize,
    },
    infer = |operation, dimensions| {
        let mut output_dimensions = dimensions;
        let Some(dimension) = output_dimensions.get_mut(operation.concat_axis) else {
            return Err(TypeError {
                message: format!(
                    "'all_gather' concat axis {} is out of bounds for rank {}",
                    operation.concat_axis,
                    output_dimensions.len(),
                ),
            });
        };
        *dimension *= operation.axis_size;
        Ok::<_, TypeError>(output_dimensions)
    },
}

impl AllGatherOperation {
    /// Returns the axis of the operand along which the participants' values are concatenated.
    #[inline]
    pub fn concat_axis(&self) -> usize {
        self.concat_axis
    }
}

shape_changing_collective! {
    /// [`Operation`] that sums every participant's operand across the named axis and scatters the result: each
    /// participant receives its own chunk of the sum along `scatter_axis` — the analogue of
    /// [JAX's `psum_scatter`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.psum_scatter.html) with
    /// `tiled = True` and [StableHLO's `reduce_scatter`](https://openxla.org/stablehlo/spec#reduce_scatter) with a
    /// sum reduction. The output shrinks `scatter_axis` by the axis size (the dimension must be divisible by it).
    /// The collective is linear and its transpose is [`AllGatherOperation`] over the same axis and dimension. A
    /// matching `batch` level consumes the mapped batch axis by summing over it and re-mapping the chunks of
    /// `scatter_axis` onto it, so batch item `i` receives chunk `i` of the sum.
    operation = PSumScatterOperation,
    name = PSUM_SCATTER_OPERATION_NAME = "psum_scatter",
    /// Value-level entry point for staging a [`PSumScatterOperation`]. Refer to its documentation for the semantics
    /// and transform rules.
    capability = PSumScatter::psum_scatter,
    fields = {
        /// Axis of the operand along which the summed result is scattered across the participants.
        scatter_axis: usize,
    },
    infer = |operation, dimensions| {
        let mut output_dimensions = dimensions;
        let Some(dimension) = output_dimensions.get_mut(operation.scatter_axis) else {
            return Err(TypeError {
                message: format!(
                    "'psum_scatter' scatter axis {} is out of bounds for rank {}",
                    operation.scatter_axis,
                    output_dimensions.len(),
                ),
            });
        };
        if *dimension % operation.axis_size != 0 {
            return Err(TypeError {
                message: format!(
                    "'psum_scatter' scatter axis {} size {} is not divisible by axis size {}",
                    operation.scatter_axis,
                    *dimension,
                    operation.axis_size,
                ),
            });
        }
        *dimension /= operation.axis_size;
        Ok::<_, TypeError>(output_dimensions)
    },
}

impl PSumScatterOperation {
    /// Returns the axis of the operand along which the summed result is scattered across the participants.
    #[inline]
    pub fn scatter_axis(&self) -> usize {
        self.scatter_axis
    }
}

shape_changing_collective! {
    /// [`Operation`] that sends every participant's operand to another participant along the named axis according
    /// to explicit `(source, target)` pairs — the analogue of
    /// [JAX's `ppermute`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ppermute.html) and
    /// [StableHLO's `collective_permute`](https://openxla.org/stablehlo/spec#collective_permute). Participants that
    /// no pair targets receive zeros. The output shape is unchanged. The collective is linear and its transpose is
    /// the permutation with every pair inverted. A matching `batch` level consumes the mapped batch axis by
    /// reassembling it in target order from per-item slices, with zero slices at untargeted positions.
    operation = PpermuteOperation,
    name = PPERMUTE_OPERATION_NAME = "ppermute",
    /// Value-level entry point for staging a [`PpermuteOperation`]. Refer to its documentation for the semantics
    /// and transform rules.
    capability = Ppermute::ppermute,
    fields = {
        /// Pairs of `(source, target)` positions along the named axis: the value of participant `source` is sent to
        /// participant `target`.
        source_target_pairs: Vec<(usize, usize)>,
    },
    infer = |operation, dimensions| {
        let mut seen_sources = std::collections::BTreeSet::new();
        let mut seen_targets = std::collections::BTreeSet::new();
        for (source, target) in &operation.source_target_pairs {
            if *source >= operation.axis_size || *target >= operation.axis_size {
                return Err(TypeError {
                    message: format!(
                        "'ppermute' pair ({source}, {target}) is out of bounds for axis size {}",
                        operation.axis_size,
                    ),
                });
            }
            if !seen_sources.insert(*source) || !seen_targets.insert(*target) {
                return Err(TypeError {
                    message: format!(
                        "'ppermute' pairs must have unique sources and targets but ({source}, {target}) repeats one",
                    ),
                });
            }
        }
        Ok::<_, TypeError>(dimensions)
    },
}

impl PpermuteOperation {
    /// Returns the `(source, target)` pairs of participant positions along the named axis.
    #[inline]
    pub fn source_target_pairs(&self) -> &[(usize, usize)] {
        self.source_target_pairs.as_slice()
    }
}

shape_changing_collective! {
    /// [`Operation`] that exchanges chunks between the participants along the named axis: every participant splits
    /// its operand into `axis_size` chunks along `split_axis` and receives the participants' chunks concatenated
    /// along `concat_axis` — the analogue of
    /// [JAX's `all_to_all`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_to_all.html) and
    /// [StableHLO's `all_to_all`](https://openxla.org/stablehlo/spec#all_to_all). The output shrinks `split_axis`
    /// by the axis size (the dimension must be divisible by it) and extends `concat_axis` by it. The collective is
    /// linear and its transpose is the exchange with the split and concatenation axes swapped. A matching `batch`
    /// level consumes the mapped batch axis with a reshape/transpose block exchange: batch item `i` receives every
    /// item's chunk `i` of `split_axis`, concatenated item-major along `concat_axis`.
    operation = AllToAllOperation,
    name = ALL_TO_ALL_OPERATION_NAME = "all_to_all",
    /// Value-level entry point for staging an [`AllToAllOperation`]. Refer to its documentation for the semantics
    /// and transform rules.
    capability = AllToAll::all_to_all,
    fields = {
        /// Axis of the operand that is split into one chunk per participant.
        split_axis: usize,

        /// Axis of the output along which the received chunks are concatenated.
        concat_axis: usize,
    },
    infer = |operation, dimensions| {
        let mut output_dimensions = dimensions;
        let rank = output_dimensions.len();
        if operation.split_axis >= rank || operation.concat_axis >= rank {
            return Err(TypeError {
                message: format!(
                    "'all_to_all' split axis {} or concat axis {} is out of bounds for rank {rank}",
                    operation.split_axis,
                    operation.concat_axis,
                ),
            });
        }
        if output_dimensions[operation.split_axis] % operation.axis_size != 0 {
            return Err(TypeError {
                message: format!(
                    "'all_to_all' split axis {} size {} is not divisible by axis size {}",
                    operation.split_axis,
                    output_dimensions[operation.split_axis],
                    operation.axis_size,
                ),
            });
        }
        output_dimensions[operation.split_axis] /= operation.axis_size;
        output_dimensions[operation.concat_axis] *= operation.axis_size;
        Ok::<_, TypeError>(output_dimensions)
    },
}

impl AllToAllOperation {
    /// Returns the axis of the operand that is split into one chunk per participant.
    #[inline]
    pub fn split_axis(&self) -> usize {
        self.split_axis
    }

    /// Returns the axis of the output along which the received chunks are concatenated.
    #[inline]
    pub fn concat_axis(&self) -> usize {
        self.concat_axis
    }
}

/// Batching rule for [`AllGatherOperation`]. A matching `batch` level consumes the mapped batch axis by
/// materializing the gather: the batch axis is transposed to sit immediately before the per-item `concat_axis` and
/// merged into it, laying the gathered chunks out item-major (item 0's chunk first), which matches the tiled
/// StableHLO `all_gather` ordering. Every batch item sees the same gathered value, so the output is replicated. A
/// non-matching level forwards the collective untouched to the parent context via [`forward_collective_to_parent`].
impl<C> BatchableOperation<C> for AllGatherOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<AllGatherOperation>,
    <C as Domain>::Value: Broadcast + Reshape + Transpose,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            return forward_collective_to_parent(context, C::Operation::from(self.clone()), inputs);
        }
        let (value, dimensions) = shape_changing_collective_batch_operand(
            ALL_GATHER_OPERATION_NAME,
            &self.axis_name,
            self.axis_size,
            inputs,
        )?;
        let per_item_rank = dimensions.len() - 1;
        if self.concat_axis >= per_item_rank {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "'all_gather' concat axis {} is out of bounds for rank {per_item_rank}",
                    self.concat_axis,
                ),
            });
        }
        // The physical layout is `[b, d_0, ..., d_{r-1}]`. Moving the leading batch axis to position `concat_axis`
        // places it immediately before the per-item `concat_axis` dimension (which shifts one position left as the
        // batch axis moves past it), so the row-major merge of `(b, d_c)` into `b * d_c` concatenates the batch
        // items item-major along `concat_axis`.
        let moved = value.move_axis(0, self.concat_axis)?;
        let mut output_dimensions = dimensions[1..].to_vec();
        output_dimensions[self.concat_axis] *= dimensions[0];
        let gathered = moved.reshape(Shape::new(output_dimensions.into_iter().map(Size::Static).collect()))?;
        Ok(vec![ArrayBatch::replicated(gathered)])
    }
}

/// Batching rule for [`PSumScatterOperation`]. A matching `batch` level consumes the mapped batch axis by summing
/// over it and re-mapping the chunks of the per-item `scatter_axis` onto it: the sum's `scatter_axis` is split into
/// `(b, d_s / b)` chunks and the new chunk axis becomes the output batch axis, so batch item `i` receives chunk `i`
/// of the sum. A non-matching level forwards the collective untouched to the parent context via
/// [`forward_collective_to_parent`].
impl<C> BatchableOperation<C> for PSumScatterOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<PSumScatterOperation>,
    <C as Domain>::Value: Broadcast + Reduce + Reshape + Transpose,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            return forward_collective_to_parent(context, C::Operation::from(self.clone()), inputs);
        }
        let (value, dimensions) = shape_changing_collective_batch_operand(
            PSUM_SCATTER_OPERATION_NAME,
            &self.axis_name,
            self.axis_size,
            inputs,
        )?;
        let batch_size = dimensions[0];
        let per_item_rank = dimensions.len() - 1;
        if self.scatter_axis >= per_item_rank {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "'psum_scatter' scatter axis {} is out of bounds for rank {per_item_rank}",
                    self.scatter_axis,
                ),
            });
        }
        let scatter_dimension = dimensions[self.scatter_axis + 1];
        if scatter_dimension % batch_size != 0 {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "'psum_scatter' scatter axis {} size {scatter_dimension} is not divisible by axis size \
                     {batch_size}",
                    self.scatter_axis,
                ),
            });
        }
        // Sum over the mapped batch axis, split the per-item `scatter_axis` into `(b, d_s / b)` chunks (the
        // row-major split makes the leading factor index the chunks), and map the chunk axis at the front.
        let summed = value.reduce(&[0], ReductionKind::Sum);
        let mut split_dimensions = dimensions[1..].to_vec();
        split_dimensions[self.scatter_axis] = batch_size;
        split_dimensions.insert(self.scatter_axis + 1, scatter_dimension / batch_size);
        let split = summed.reshape(Shape::new(split_dimensions.into_iter().map(Size::Static).collect()))?;
        let scattered = split.move_axis(self.scatter_axis, 0)?;
        let physical_type = scattered.r#type().into_owned();
        Ok(vec![ArrayBatch::new(physical_type, scattered, Some(0))?])
    }
}

/// Batching rule for [`PpermuteOperation`]. A matching `batch` level consumes the mapped batch axis by reassembling
/// it in target order: for each position `t` along the batch axis, the output receives the slice of the source item
/// that sends to `t`, or a zero slice when no pair targets `t`. A non-matching level forwards the collective
/// untouched to the parent context via [`forward_collective_to_parent`].
impl<C> BatchableOperation<C> for PpermuteOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<PpermuteOperation>,
    <C as Domain>::Value: Broadcast + Concatenate + Slice + Transpose + ZeroLike,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            return forward_collective_to_parent(context, C::Operation::from(self.clone()), inputs);
        }
        let (value, dimensions) =
            shape_changing_collective_batch_operand(PPERMUTE_OPERATION_NAME, &self.axis_name, self.axis_size, inputs)?;
        let batch_size = dimensions[0];
        // Map each target position along the batch axis to the source item that sends to it; positions that no pair
        // targets receive zeros. Pair uniqueness is enforced by output type inference, so it is not revalidated here.
        let mut sources = vec![None; batch_size];
        for (source, target) in &self.source_target_pairs {
            if *source >= batch_size || *target >= batch_size {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "'ppermute' pair ({source}, {target}) is out of bounds for axis size {batch_size}",
                    ),
                });
            }
            sources[*target] = Some(*source);
        }
        // Slice each item `[i, i + 1)` from the leading batch axis and concatenate the slices back in target order.
        let rank = dimensions.len();
        let strides = vec![1; rank];
        let slice_item = |item: usize| -> Result<<C as Domain>::Value, ProgramError> {
            let mut start_indices = vec![0; rank];
            let mut limit_indices = dimensions.clone();
            start_indices[0] = item;
            limit_indices[0] = item + 1;
            value.slice(&start_indices, &limit_indices, &strides)
        };
        let mut zero_item = None;
        let mut items = Vec::with_capacity(batch_size);
        for source in sources {
            match source {
                Some(source) => items.push(slice_item(source)?),
                None => {
                    if zero_item.is_none() {
                        zero_item = Some(slice_item(0)?.zero_like());
                    }
                    // The zero slice was materialized right above when absent.
                    items.push(zero_item.clone().unwrap());
                }
            }
        }
        let permuted = Concatenate::concatenate(&items, 0)?;
        let physical_type = permuted.r#type().into_owned();
        Ok(vec![ArrayBatch::new(physical_type, permuted, Some(0))?])
    }
}

/// Batching rule for [`AllToAllOperation`]. A matching `batch` level consumes the mapped batch axis with a
/// reshape/transpose block exchange: the per-item `split_axis` is split into `(b, d_p / b)` chunks, the chunk axis
/// is swapped with the leading batch axis (so the batch axis indexes the *receiving* item), and the sender axis is
/// then merged item-major into the per-item `concat_axis` — batch item `i` receives every item's chunk `i`,
/// concatenated along `concat_axis`. A non-matching level forwards the collective untouched to the parent context
/// via [`forward_collective_to_parent`].
impl<C> BatchableOperation<C> for AllToAllOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<AllToAllOperation>,
    <C as Domain>::Value: Broadcast + Reshape + Transpose,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            return forward_collective_to_parent(context, C::Operation::from(self.clone()), inputs);
        }
        let (value, dimensions) = shape_changing_collective_batch_operand(
            ALL_TO_ALL_OPERATION_NAME,
            &self.axis_name,
            self.axis_size,
            inputs,
        )?;
        let batch_size = dimensions[0];
        let per_item_rank = dimensions.len() - 1;
        if self.split_axis >= per_item_rank || self.concat_axis >= per_item_rank {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "'all_to_all' split axis {} or concat axis {} is out of bounds for rank {per_item_rank}",
                    self.split_axis, self.concat_axis,
                ),
            });
        }
        let split_dimension = dimensions[self.split_axis + 1];
        if split_dimension % batch_size != 0 {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "'all_to_all' split axis {} size {split_dimension} is not divisible by axis size {batch_size}",
                    self.split_axis,
                ),
            });
        }
        // Split the per-item `split_axis` (physical position `split_axis + 1`) into `(b, d_p / b)` so its leading
        // factor indexes the chunks, then swap the chunk axis with the leading sender axis: afterwards the leading
        // axis indexes the *receiving* item and the axis at `split_axis + 1` indexes the sender.
        let mut split_dimensions = dimensions.clone();
        split_dimensions[self.split_axis + 1] = batch_size;
        split_dimensions.insert(self.split_axis + 2, split_dimension / batch_size);
        let split = value.reshape(Shape::new(split_dimensions.into_iter().map(Size::Static).collect()))?;
        let exchanged = split.swap_axes(0, self.split_axis + 1)?;
        // Move the sender axis to sit immediately before the per-item `concat_axis` and merge it in row-major, which
        // concatenates the received chunks sender-major along `concat_axis`. When `split_axis == concat_axis` the
        // sender axis already sits immediately before the chunk axis and the merge restores the per-item split-axis
        // size (shrunk by the split, extended back by the concatenation).
        let moved = exchanged.move_axis(self.split_axis + 1, self.concat_axis + 1)?;
        let mut output_dimensions = dimensions;
        output_dimensions[self.split_axis + 1] /= batch_size;
        output_dimensions[self.concat_axis + 1] *= batch_size;
        let received = moved.reshape(Shape::new(output_dimensions.into_iter().map(Size::Static).collect()))?;
        let physical_type = received.r#type().into_owned();
        Ok(vec![ArrayBatch::new(physical_type, received, Some(0))?])
    }
}

/// Transpose rule for [`AllGatherOperation`]: a tiled all-gather is the adjoint of a sum-scatter over the same axis
/// and dimension, so the operand cotangent is a [`PSumScatterOperation`] of the output cotangent.
impl<V, O> TransposableOperation<V, O> for AllGatherOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<PSumScatterOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        transpose_shape_changing_collective(
            context,
            inputs,
            outputs,
            PSumScatterOperation::new(self.axis_name.clone(), self.axis_size, self.concat_axis),
        )
    }
}

/// Transpose rule for [`PSumScatterOperation`]: a sum-scatter is the adjoint of a tiled all-gather over the same
/// axis and dimension, so the operand cotangent is an [`AllGatherOperation`] of the output cotangent.
impl<V, O> TransposableOperation<V, O> for PSumScatterOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<AllGatherOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        transpose_shape_changing_collective(
            context,
            inputs,
            outputs,
            AllGatherOperation::new(self.axis_name.clone(), self.axis_size, self.scatter_axis),
        )
    }
}

/// Transpose rule for [`PpermuteOperation`]: sending along `(source, target)` pulls cotangents back along
/// `(target, source)`, so the operand cotangent is the permutation with every pair inverted.
impl<V, O> TransposableOperation<V, O> for PpermuteOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<PpermuteOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        let inverted_pairs =
            self.source_target_pairs.iter().map(|(source, target)| (*target, *source)).collect::<Vec<_>>();
        transpose_shape_changing_collective(
            context,
            inputs,
            outputs,
            PpermuteOperation::new(self.axis_name.clone(), self.axis_size, inverted_pairs),
        )
    }
}

/// Transpose rule for [`AllToAllOperation`]: the chunk exchange is its own adjoint with the split and concatenation
/// axes swapped.
impl<V, O> TransposableOperation<V, O> for AllToAllOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<AllToAllOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        transpose_shape_changing_collective(
            context,
            inputs,
            outputs,
            AllToAllOperation::new(self.axis_name.clone(), self.axis_size, self.concat_axis, self.split_axis),
        )
    }
}

/// Stages the adjoint collective of a linear shape-changing collective on the output cotangent: a known operand
/// receives a structural zero, a zero output cotangent stays symbolic, and a live cotangent rides the provided
/// adjoint operation.
fn transpose_shape_changing_collective<V, O, A>(
    context: &mut TracingContext<V, O>,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    adjoint: A,
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError>
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<A>,
    A: Operation<ArrayType>,
{
    check_count!("input", inputs, 1, ProgramError);
    check_count!("output", outputs, 1, ProgramError);
    if inputs[0].is_known() {
        return Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]);
    }
    match &outputs[0] {
        MaybeZero::Value(cotangent) => {
            let mut contributions = context.bind(O::from(adjoint), Vec::new(), std::slice::from_ref(cotangent))?;
            check_count!("output", contributions, 1, ProgramError);
            Ok(vec![MaybeZero::Value(contributions.remove(0))])
        }
        MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]),
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::{
        ArrayBatch, BatchAxis, BatchAxisSpecification, BatchableOperation, BatchingContext, BatchingError, batch,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::value_and_gradient;
    use crate::types::{Shape, Size};

    use super::*;

    /// Creates an active batching frame binding the named axis `"i"` over an eager parent whose operation family
    /// contains every operation the collective batching rule may bind (notably `FillOperation` for `PMean`).
    fn batching_context(axis_size: usize) -> BatchingContext<EagerContext<Array, ArrayOperation<Array>>> {
        BatchingContext::new(EagerContext::new(), axis_size).with_axis_name("i".to_string())
    }

    #[test]
    fn test_collective_psum_reduces_along_the_batch_axis() {
        // Mapped input shape [3] at axis 0: per-item scalar. PSum collapses the batch axis to a
        // replicated scalar holding the total.
        let input = {
            let value = Array::vector(vec![1.0, 2.0, 3.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = CollectiveOperation::new("i".to_string(), CollectiveKind::PSum)
            .batch(&batching_context(3), &crate::EmptyRegionDriver, &[input])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values(), &[6.0]);
    }

    #[test]
    fn test_collective_pmax_reduces_along_the_batch_axis() {
        let input = {
            let value = Array::vector(vec![1.0, 4.0, 2.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = CollectiveOperation::new("i".to_string(), CollectiveKind::PMax)
            .batch(&batching_context(3), &crate::EmptyRegionDriver, &[input])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values(), &[4.0]);
    }

    #[test]
    fn test_collective_pmean_divides_by_batch_size() {
        // Per-item scalar input of shape [3] mapped at axis 0. PMean returns the mean of the three batch items as a
        // replicated scalar, exercising the `1 / N` factor that distinguishes it from PSum. The batching frame binds
        // the axis name `"data"` to show the rule matches on the collective's own axis name rather than a fixture
        // default.
        let input = {
            let value = Array::vector(vec![2.0, 4.0, 6.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3)
            .with_axis_name("data".to_string());
        let outputs = CollectiveOperation::new("data".to_string(), CollectiveKind::PMean)
            .batch(&context, &crate::EmptyRegionDriver, &[input])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        let values = outputs[0].value().to_f64s();
        assert_eq!(values.len(), 1);
        let delta = (values[0] - 4.0).abs();
        assert!(delta < 1e-9, "expected pmean = 4.0, got {}", values[0]);
    }

    #[test]
    fn test_collective_passes_through_replicated_input() {
        let input = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]));
        let outputs = CollectiveOperation::new("i".to_string(), CollectiveKind::PSum)
            .batch(&batching_context(3), &crate::EmptyRegionDriver, &[input])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_collective_over_unbound_axis_is_rejected() {
        use crate::axes::AxisError;
        use crate::backends::arrays::{Array, ArrayOperation};
        use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer};
        use crate::contexts::EagerContext;

        // The batch binds the axis `"i"`, but the collective names `"j"`, which no enclosing transform binds. Rather
        // than silently acting as identity (the pre-validation behavior), staging the collective fails fast with
        // `AxisError::UnboundAxisName`, matching JAX's error for a collective over an unbound axis name. The error
        // rides the `ProgramError::Custom` channel as a `BatchingError::Axis` and is re-typed at the public `batch`
        // boundary, so the surfaced error is exactly that variant.
        let result: Result<Array, BatchingError> = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                item.collective("j", CollectiveKind::PSum)
            },
            Array::vector(vec![1.0, 2.0, 3.0]),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("i"),
        );
        assert_eq!(result.unwrap_err(), BatchingError::Axis(AxisError::UnboundAxisName { name: "j".to_string() }));
    }

    #[test]
    fn test_collective_psum_value_and_grad_through_vmap_re_sums_the_cotangent() {
        use crate::backends::arrays::{Array, ArrayOperation};
        use crate::batching::BatchAxisSpecification;
        use crate::contexts::EagerContext;
        use crate::differentiation::LinearizationTracer;

        // `g(x) = psum_i(x)`: the vmapped `psum` over the mapped axis `"i"` consumes that axis, producing the
        // replicated total `S = Σ_j x_j`. Reverse mode pulls the scalar ones cotangent back through the
        // self-adjoint `psum`, which re-broadcasts the cotangent across the batch items, giving `∂g/∂x_i = 1`
        // for every input. With `x = [1, 2, 3]` the value is `6` and the gradient is `[1, 1, 1]`.
        let (value, gradient) = value_and_gradient(
            |x: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let total: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = batch(
                    |item| item.collective("i", CollectiveKind::PSum),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("i"),
                )
                .unwrap();
                total
            },
            Array::vector(vec![1.0, 2.0, 3.0]),
        )
        .unwrap();
        assert_eq!(value.to_f64s(), vec![6.0]);
        assert_eq!(gradient.to_f64s(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_collective_pmean_value_and_grad_through_vmap_carries_the_inverse_batch_size() {
        use crate::backends::arrays::{Array, ArrayOperation};
        use crate::batching::BatchAxisSpecification;
        use crate::contexts::EagerContext;
        use crate::differentiation::LinearizationTracer;

        // `g(x) = pmean_i(x)`: the vmapped `pmean` over the mapped axis `"i"` consumes that axis, producing the
        // replicated mean `M = (1/N)·Σ_j x_j`. Reverse mode pulls the scalar ones cotangent back through the
        // self-adjoint `pmean`, which carries the `1/N` factor, so `∂g/∂x_i = 1/N` for every input. With `x =
        // [1, 2, 3]` (so `N = 3`) the value is `2` and the gradient is `[1/3, 1/3, 1/3]`, witnessing the `1/N`
        // scaling that distinguishes `pmean` from `psum`.
        let (value, gradient) = value_and_gradient(
            |x: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let mean: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = batch(
                    |item| item.collective("i", CollectiveKind::PMean),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("i"),
                )
                .unwrap();
                mean
            },
            Array::vector(vec![1.0, 2.0, 3.0]),
        )
        .unwrap();
        assert_eq!(value.to_f64s(), vec![2.0]);
        assert_eq!(gradient.to_f64s(), vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    }

    #[test]
    fn test_nested_batch_named_axes_route_collectives_to_matching_level() {
        // The inner `psum` targets the *outer* named axis, so each inner batch item must reduce over the
        // outer batch items: column sums of [[1, 2], [3, 4]].
        let x = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)])),
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let output: Array = batch(
            |row| {
                Ok(batch(
                    |scalar| scalar.collective("outer", CollectiveKind::PSum),
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

        assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),);
        assert_eq!(output.to_f64s(), vec![4.0, 6.0]);
    }

    /// Returns the static `f32` vector type of the provided length used by the shape-changing collective tests.
    fn f32_vector(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(length)]))
    }

    #[test]
    fn test_all_gather_type_inference() {
        use crate::macros::check_operation_type_inference;

        let operation = AllGatherOperation::new("x".to_string(), 4, 0);
        assert_eq!(operation.axis_name(), "x");
        assert_eq!(operation.axis_size(), 4);
        assert_eq!(operation.concat_axis(), 0);
        assert_eq!(operation.name(), ALL_GATHER_OPERATION_NAME);
        assert_eq!(operation.to_string(), "all_gather [axis_name=\"x\", axis_size=4, concat_axis=0]");
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    input_types = [f32_vector(2)],
                    output_types = [f32_vector(8)],
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "'all_gather' concat axis 0 is out of bounds for rank 0",
                },
                {
                    input_types = [ArrayType::new(DataType::F32, Shape::new(vec![Size::Dynamic(None)]))],
                    error = "'all_gather' does not support dynamically shaped operands",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = AllGatherOperation::new("x".to_string(), 4, 0),
            input_types = [f32_vector(2)],
        );
    }

    #[test]
    fn test_psum_scatter_type_inference() {
        use crate::macros::check_operation_type_inference;

        check_operation_type_inference!(
            operation = PSumScatterOperation::new("x".to_string(), 4, 0),
            cases = [
                {
                    input_types = [f32_vector(8)],
                    output_types = [f32_vector(2)],
                },
                {
                    input_types = [f32_vector(6)],
                    error = "'psum_scatter' scatter axis 0 size 6 is not divisible by axis size 4",
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "'psum_scatter' scatter axis 0 is out of bounds for rank 0",
                },
            ],
        );
    }

    #[test]
    fn test_ppermute_type_inference() {
        use crate::macros::check_operation_type_inference;

        check_operation_type_inference!(
            operation = PpermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]),
            cases = [{
                input_types = [f32_vector(3)],
                output_types = [f32_vector(3)],
            }],
        );
        check_operation_type_inference!(
            operation = PpermuteOperation::new("x".to_string(), 2, vec![(0, 2)]),
            cases = [{
                input_types = [f32_vector(3)],
                error = "'ppermute' pair (0, 2) is out of bounds for axis size 2",
            }],
        );
        check_operation_type_inference!(
            operation = PpermuteOperation::new("x".to_string(), 2, vec![(0, 1), (0, 0)]),
            cases = [{
                input_types = [f32_vector(3)],
                error = "'ppermute' pairs must have unique sources and targets but (0, 0) repeats one",
            }],
        );
    }

    #[test]
    fn test_all_to_all_type_inference() {
        use crate::macros::check_operation_type_inference;

        let matrix = |rows: usize, columns: usize| {
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(rows), Size::Static(columns)]))
        };
        check_operation_type_inference!(
            operation = AllToAllOperation::new("x".to_string(), 4, 0, 1),
            cases = [
                {
                    input_types = [matrix(8, 3)],
                    output_types = [matrix(2, 12)],
                },
                {
                    input_types = [matrix(6, 3)],
                    error = "'all_to_all' split axis 0 size 6 is not divisible by axis size 4",
                },
            ],
        );
    }

    #[test]
    fn test_all_gather_interpretation_requires_an_enclosing_binder() {
        use crate::interpretation::InterpretableOperation;

        // A single-participant axis is degenerate: the gather concatenates exactly one operand, so interpretation is
        // the identity.
        let outputs = AllGatherOperation::new("x".to_string(), 1, 0)
            .interpret(
                &EagerContext::<Array, ArrayOperation<Array>>::new(),
                &crate::EmptyRegionDriver,
                &[Array::vector(vec![1.0, 2.0])],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].values(), &[1.0, 2.0]);

        // Any larger axis has no per-item semantics: the other participants do not exist outside an enclosing binder.
        let error = AllGatherOperation::new("x".to_string(), 2, 0)
            .interpret(
                &EagerContext::<Array, ArrayOperation<Array>>::new(),
                &crate::EmptyRegionDriver,
                &[Array::vector(vec![1.0, 2.0])],
            )
            .unwrap_err();
        assert!(matches!(
            error,
            ProgramError::UnsupportedOperation { message }
                if message == "cannot interpret 'all_gather' over axis 'x' of size 2 without an enclosing binder",
        ));
    }

    #[test]
    fn test_all_gather_over_unbound_axis_is_rejected() {
        use crate::batching::BatchingTracer;

        // The batch binds only the axis `"i"`, but the `all_gather` names `"x"`, which no enclosing transform binds.
        // Axis-size resolution fails fast at staging time with `AxisError::UnboundAxisName` rather than silently
        // acting as identity.
        let result: Result<Array, BatchingError> = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>>| item.all_gather("x", 0),
            Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("i"),
        );
        assert_eq!(result.unwrap_err(), BatchingError::Axis(AxisError::UnboundAxisName { name: "x".to_string() }));
    }

    #[test]
    fn test_all_gather_over_batched_axis_materializes_the_gather() {
        use crate::batching::BatchingTracer;

        // The batch binds the axis `"x"` that the `all_gather` names, so the matching batching rule consumes the
        // mapped axis: every item receives the item-major concatenation of all items along `concat_axis`,
        // replicated across the batch. With items `[1, 2]` and `[3, 4]` the gathered value is `[1, 2, 3, 4]`,
        // matching the verified cross-device `shard_map` execution semantics of the tiled StableHLO `all_gather`.
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>>| item.all_gather("x", 0),
            Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)])));
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_all_gather_of_replicated_input_concatenates_copies() {
        // A replicated operand at a matching level is first materialized as `axis_size` identical batch items, so
        // the gather degenerates to the item-major concatenation of that many copies of the shared value.
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let outputs = AllGatherOperation::new("x".to_string(), 2, 0)
            .batch(&context, &crate::EmptyRegionDriver, &[ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]))])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values(), &[1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_psum_scatter_over_batched_axis_sums_and_scatters() {
        use crate::batching::BatchingTracer;

        // The batch binds the axis `"x"` that the `psum_scatter` names, so the matching batching rule sums over the
        // mapped axis and re-maps the chunks of `scatter_axis` onto it. With items `[1, 2, 3, 4]` and
        // `[10, 20, 30, 40]` the sum is `[11, 22, 33, 44]`, so item 0 receives `[11, 22]` and item 1 receives
        // `[33, 44]`, matching the verified cross-device `shard_map` execution semantics of StableHLO's
        // `reduce_scatter`.
        let x = Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>>| item.psum_scatter("x", 0),
            x,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(output.to_f64s(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_ppermute_over_batched_axis_permutes_the_items() {
        use crate::batching::BatchingTracer;

        // The rotation `[(0, 1), (1, 0)]` swaps the two batch items: item 0 receives item 1's `[3, 4]` and item 1
        // receives item 0's `[1, 2]`.
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>>| item.ppermute("x", vec![(0, 1), (1, 0)]),
            Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(output.to_f64s(), vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_ppermute_over_batched_axis_zeros_untargeted_items() {
        use crate::batching::BatchingTracer;

        // With the single pair `(0, 1)`, item 1 receives item 0's `[1, 2]` while no pair targets item 0, so it
        // receives zeros, matching JAX's `ppermute` semantics for untargeted participants.
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>>| item.ppermute("x", vec![(0, 1)]),
            Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(output.to_f64s(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_all_to_all_over_batched_axis_exchanges_chunks() {
        use crate::batching::BatchingTracer;

        // Block exchange with `split_axis == concat_axis == 0`: each item splits its vector into two chunks and
        // receives its own chunk index from every item, concatenated item-major. With items `[1, 2, 3, 4]` and
        // `[5, 6, 7, 8]`, item 0 receives `[1, 2, 5, 6]` and item 1 receives `[3, 4, 7, 8]`, matching the verified
        // cross-device `shard_map` execution semantics of StableHLO's `all_to_all`.
        let x = Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>>| item.all_to_all("x", 0, 0),
            x,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(4)])),
        );
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_all_to_all_over_batched_axis_with_distinct_axes_exchanges_chunks() {
        use crate::batching::BatchingTracer;

        // Distinct split and concatenation axes over per-item `[2, 2]` matrices: each item splits its rows across
        // the items and receives its own row index from every item, concatenated item-major along the columns. With
        // item 0 = `[[1, 2], [3, 4]]` and item 1 = `[[5, 6], [7, 8]]`, item 0 receives `[[1, 2, 5, 6]]` and item 1
        // receives `[[3, 4, 7, 8]]` (per-item shape `[1, 4]`).
        let x = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)])),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>>| item.all_to_all("x", 0, 1),
            x,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(1), Size::Static(4)])),
        );
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_all_gather_transposes_to_psum_scatter() {
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        // A tiled all-gather is the adjoint of a sum-scatter over the same axis and dimension, so the pullback stages
        // a `psum_scatter` on the output cotangent with the gather's concat axis as its scatter axis.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(f32_vector(2));
        let output = builder
            .add_instruction(AllGatherOperation::new("x".to_string(), 2, 0), Vec::new(), vec![input])
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc::indoc! {r#"
                lambda %0:f32[4] .
                let %1:f32[2] = psum_scatter [axis_name="x", axis_size=2, scatter_axis=0] %0
                in (%1)
            "#}
            .trim_end(),
        );
    }

    #[test]
    fn test_ppermute_transposes_to_inverted_pairs() {
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        // Sending along `(source, target)` pulls cotangents back along `(target, source)`, so the pullback stages the
        // permutation with every pair inverted.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(f32_vector(2));
        let output = builder
            .add_instruction(PpermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]), Vec::new(), vec![input])
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc::indoc! {r#"
                lambda %0:f32[2] .
                let %1:f32[2] = ppermute [axis_name="x", axis_size=2, source_target_pairs=[(1, 0), (0, 1)]] %0
                in (%1)
            "#}
            .trim_end(),
        );
    }
}
