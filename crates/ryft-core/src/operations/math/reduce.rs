use std::fmt::Display;
use std::ops::{Div, Mul};

use crate::arrays::LinearResiduals;
use crate::arrays::{
    ArrayIrType, ArrayType, DataType, Dimension, DimensionOperation, DimensionType, DimensionValue, Shape, Sharding,
    StaticShape,
};
use crate::batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver,
    BatchingError, InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, ProjectedContext};
use crate::differentiation::elementwise::ElementwiseDerivativeAlignment;
use crate::differentiation::forward::jvp_projected_operation;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    LinearCallOperation, MemberDifferentiableOperation, TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::compare::{Compare, CompareOperation, ComparisonDirection};
use crate::operations::constants::{ConstantOperation, Fill};
use crate::operations::dimensions::{DimensionMulOperation, DimensionSizeOperation, DimensionToScalarOperation};
use crate::operations::manipulation::{
    BroadcastOperation, ConvertElementTypeOperation, LegacyBroadcast, LegacyBroadcastOperation,
};
use crate::operations::math::{DivOperation, MulOperation};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::operations::{Operation, OperationFormatter, OperationProjection};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::{MaybeZero, ProgramError, Value, ValueProjection};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this module.

/// Kind of reduction performed by a [`ReduceOperation`].
///
/// Reductions collapse one or more axes of an input array by combining their elements with a
/// binary associative-commutative operator that defines an identity element. Each kind corresponds
/// to one such operator/identity pair and lowers to the equivalent `stablehlo.reduce` body in the
/// XLA backend.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReductionKind {
    /// Numeric sum reduction. The identity is `0` and the combiner is addition.
    Sum,

    /// Numeric mean reduction: a [`Sum`](Self::Sum) divided by the product of reduced extents.
    /// The numeric data type must support division.
    Mean,

    /// Maximum reduction. Boolean inputs use disjunction, real numeric inputs propagate NaNs and order negative zero
    /// below positive zero, and complex inputs compare lexicographically by `(real, imaginary)`. The identity is the
    /// data type's smallest value under that ordering.
    Max,

    /// Minimum reduction. Boolean inputs use conjunction, real numeric inputs propagate NaNs and order negative zero
    /// below positive zero, and complex inputs compare lexicographically by `(real, imaginary)`. The identity is the
    /// data type's largest value under that ordering.
    Min,

    /// Boolean disjunction reduction (logical-OR). The identity is `false` and the combiner is OR.
    /// Inputs must have [`DataType::Boolean`].
    Any,

    /// Boolean conjunction reduction (logical-AND). The identity is `true` and the combiner is
    /// AND. Inputs must have [`DataType::Boolean`].
    All,
}

impl ReductionKind {
    /// Returns the canonical operation name suffix for this kind.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Max => "max",
            Self::Min => "min",
            Self::Any => "any",
            Self::All => "all",
        }
    }

    /// Returns `true` when this kind requires [`DataType::Boolean`] inputs.
    pub fn requires_boolean(self) -> bool {
        matches!(self, Self::Any | Self::All)
    }
}

impl Display for ReductionKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Returns the output [`ArrayType`] produced by reducing `input` along `axes` with `kind`.
///
/// Validates that:
///   - `axes` are unique and within `0..rank(input)`;
///   - `kind` matches the input data type (Boolean for Any/All, Boolean or numeric for Max/Min, and numeric for
///     Sum/Mean; structural-zero supports every kind).
///
/// The reduced axes are removed from the output shape; non-reduced axes keep their order. The output [`Sharding`]
/// follows JAX's reduction sharding rule (`_reduce_op_sharding_rule` in `jax/_src/lax/lax.py`): the reduced axes'
/// per-dimension [`ShardingDimension`](crate::arrays::ShardingDimension) entries are deleted while the remaining
/// entries keep their order, and the reduction-state and manual-axis sets pass through unchanged. A reduced axis that
/// is sharded is *not* an error here — the backend partitioner owns the cross-shard reduction; use
/// [`ReduceOperation::with_output_sharding`] to request an unreduced output that defers it. The
/// [`Layout`](crate::arrays::Layout) is dropped (it is rank-specific) and the [`Memory`](crate::arrays::Memory)
/// placement is preserved.
pub fn reduce_abstract(
    input: &ArrayType,
    axes: &[usize],
    kind: ReductionKind,
    op: &'static str,
) -> Result<ArrayType, TypeError> {
    let rank = input.rank();
    let mut reduce_mask = vec![false; rank];
    for axis in axes {
        if *axis >= rank {
            return Err(TypeError::invalid(format!("'{op}' axis {axis} is out of bounds for rank {rank}")));
        }
        if reduce_mask[*axis] {
            return Err(TypeError::invalid(format!("'{op}' contains duplicate axis {axis}")));
        }
        reduce_mask[*axis] = true;
    }

    let data_type = input.data_type();
    let (requirement, supports_kind) = if kind.requires_boolean() {
        ("Boolean", data_type.is_boolean())
    } else if matches!(kind, ReductionKind::Max | ReductionKind::Min) {
        ("Boolean or numeric", data_type.is_boolean() || data_type.is_numeric() || data_type == DataType::Zero)
    } else {
        ("numeric", data_type.is_numeric() || data_type == DataType::Zero)
    };
    if !supports_kind {
        return Err(TypeError::invalid(format!(
            "'{op}' kind {kind} requires {requirement} inputs but got {data_type}"
        )));
    }

    let dimensions = input
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| (!reduce_mask[axis]).then_some(size.clone()))
        .collect::<Vec<_>>();
    let sharding = reduce_sharding(input.sharding(), &reduce_mask, op)?;
    ArrayType::new(data_type, Shape::new(dimensions))
        .with_memory(input.memory())
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))
}

/// Computes the output [`Sharding`] for a reduction whose reduced axes are marked in `reduce_mask`. The reduced
/// axes' per-dimension entries are deleted; the remaining entries keep their order and the reduction-state and
/// manual-axis sets pass through unchanged. Refer to the documentation of [`reduce_abstract`] for the full rule.
fn reduce_sharding(
    sharding: Option<&Sharding>,
    reduce_mask: &[bool],
    op: &'static str,
) -> Result<Option<Sharding>, TypeError> {
    sharding
        .map(|sharding| {
            let dimensions = sharding
                .dimensions()
                .iter()
                .enumerate()
                .filter_map(|(axis, dimension)| (!reduce_mask[axis]).then(|| dimension.clone()))
                .collect::<Vec<_>>();
            Sharding::new(sharding.mesh().clone(), dimensions)
                .and_then(|output| output.with_unreduced_axes(sharding.unreduced_axes().clone()))
                .and_then(|output| output.with_reduced_axes(sharding.reduced_axes().clone()))
                .and_then(|output| output.with_varying_manual_axes(sharding.varying_manual_axes().clone()))
                .map_err(|error| TypeError::invalid(format!("'{op}' output sharding construction failed: {error}")))
        })
        .transpose()
}

/// Primitive representing one N-dimensional axis-collapsing reduction.
///
/// [`ReduceOperation`] collapses the input array along `axes` using the operator/identity pair
/// described by [`kind`](Self::kind). The output rank is the input rank minus the number of
/// reduced axes; non-reduced axes keep their relative order. Lowers to StableHLO's
/// `stablehlo.reduce` op in the XLA backend.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReduceOperation {
    /// Axes to reduce.
    axes: Vec<usize>,

    /// Kind of reduction.
    kind: ReductionKind,

    /// Optional requested output [`Sharding`]. Refer to the documentation of [`Self::with_output_sharding`].
    output_sharding: Option<Sharding>,
}

impl ReduceOperation {
    /// Creates a new [`ReduceOperation`] reducing along `axes` with the supplied `kind`. The input
    /// shape is not part of the operation payload: it is recoverable from the staged input types
    /// wherever a rule needs it.
    #[inline]
    pub fn new(axes: Vec<usize>, kind: ReductionKind) -> Self {
        Self { axes, kind, output_sharding: None }
    }

    /// Attaches a requested output [`Sharding`] to this operation, mirroring the `out_sharding` parameter of JAX's
    /// `reduce_sum`. It is honored only by [`ReductionKind::Sum`] (the other kinds reject it), and it is the only way
    /// to produce an output with unreduced axes — per-shard partial sums whose cross-device reduction is delayed.
    /// When set, type inference validates the requested sharding (rank, mesh, no auto axes, and — for an unreduced
    /// request — JAX's `_reduce_sum_unreduced_rule`: every requested unreduced axis must be an `Explicit` axis that
    /// sharded one of the summed-over dimensions or was already unreduced on the operand) and uses it for the output.
    #[inline]
    pub fn with_output_sharding(mut self, output_sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns the axes reduced by this operation.
    #[inline]
    pub fn axes(&self) -> &[usize] {
        self.axes.as_slice()
    }

    /// Returns the kind of reduction.
    #[inline]
    pub fn kind(&self) -> ReductionKind {
        self.kind
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
}

impl Display for ReduceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReduceOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        match self.kind {
            ReductionKind::Sum => "reduce_sum",
            ReductionKind::Mean => "reduce_mean",
            ReductionKind::Max => "reduce_max",
            ReductionKind::Min => "reduce_min",
            ReductionKind::Any => "reduce_any",
            ReductionKind::All => "reduce_all",
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let output = reduce_abstract(&input_types[0], self.axes.as_slice(), self.kind, self.name())?;
        let Some(output_sharding) = &self.output_sharding else {
            return Ok(vec![output]);
        };
        validate_reduce_output_sharding(&input_types[0], self.axes.as_slice(), self.kind, output_sharding, &output)?;
        Ok(vec![
            output
                .with_sharding(output_sharding.clone())
                .map_err(|error| TypeError::invalid(error.to_string()))?,
        ])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("axes", format_args!("{:?}", self.axes))?;
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

/// Validates a requested `output_sharding` for a reduction, mirroring JAX's `_reduce_sum_unreduced_rule`. Only
/// [`ReductionKind::Sum`] may carry a requested output sharding. The sharding must match the reduced output's rank
/// and (when the operand carries a sharding) its mesh, and may not reference [`MeshAxisType::Auto`] axes. When the
/// request is unreduced, every requested unreduced axis must be an [`MeshAxisType::Explicit`] axis that either
/// sharded one of the summed-over dimensions or was already unreduced on the operand — the axes whose cross-device
/// reduction the request is deferring.
fn validate_reduce_output_sharding(
    input: &ArrayType,
    axes: &[usize],
    kind: ReductionKind,
    output_sharding: &Sharding,
    reduced_output: &ArrayType,
) -> Result<(), TypeError> {
    use crate::arrays::{MeshAxisType, ShardingDimension};

    if kind != ReductionKind::Sum {
        return Err(TypeError::invalid(format!(
            "{} does not support a requested output sharding (only reduce_sum does)",
            kind.name()
        )));
    }
    if output_sharding.rank() != reduced_output.rank() {
        return Err(TypeError::invalid(format!(
            "reduce_sum output sharding rank ({}) does not match the output rank ({})",
            output_sharding.rank(),
            reduced_output.rank(),
        )));
    }
    if let Some(input_sharding) = input.sharding()
        && output_sharding.mesh() != input_sharding.mesh()
    {
        return Err(TypeError::invalid("reduce_sum output sharding must use the same mesh as the operand"));
    }
    let mut referenced_axes: Vec<&String> = output_sharding.unreduced_axes().iter().collect();
    referenced_axes.extend(output_sharding.reduced_axes());
    for dimension in output_sharding.dimensions() {
        if let ShardingDimension::Sharded(axis_names) = dimension {
            referenced_axes.extend(axis_names);
        }
    }
    if referenced_axes
        .iter()
        .any(|name| output_sharding.mesh().axis_type(name) == Some(MeshAxisType::Auto))
    {
        return Err(TypeError::invalid("reduce_sum output sharding cannot reference auto mesh axes"));
    }

    if !output_sharding.unreduced_axes().is_empty() {
        // The axes whose reduction the request defers: the Explicit axes that sharded the summed-over dimensions,
        // together with the axes the operand was already unreduced over.
        let mut reducible_axes: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        if let Some(input_sharding) = input.sharding() {
            for axis in axes {
                if let ShardingDimension::Sharded(axis_names) = &input_sharding.dimensions()[*axis] {
                    reducible_axes.extend(
                        axis_names
                            .iter()
                            .filter(|name| input_sharding.mesh().axis_type(name) == Some(MeshAxisType::Explicit))
                            .map(String::as_str),
                    );
                }
            }
            reducible_axes.extend(input_sharding.unreduced_axes().iter().map(String::as_str));
        }
        if !output_sharding.unreduced_axes().iter().all(|name| reducible_axes.contains(name.as_str())) {
            return Err(TypeError::invalid(
                "reduce_sum output sharding unreduced axes must be among the explicit axes sharding the \
                          reduced dimensions or the operand's unreduced axes",
            ));
        }
    }
    Ok(())
}

impl<C: Domain<Type = ArrayType, Value: Reduce>> InterpretableOperation<C> for ReduceOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // The requested output sharding flows through the capability method so that interpretation over staging
        // values (e.g., during program batching) preserves it; concrete values ignore it.
        Ok(vec![match &self.output_sharding {
            Some(output_sharding) => {
                inputs[0].clone().reduce_with_output_sharding(self.axes.as_slice(), self.kind, output_sharding)
            }
            None => inputs[0].clone().reduce(self.axes.as_slice(), self.kind),
        }])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ReduceOperation where
    C::Operation: From<ReduceOperation>
{
}

/// Batching rule for [`ReduceOperation`]: the reduced axes are expressed in the per-item coordinate system, so the
/// rule lifts them past the inserted batch dimension with `lift_reduce_axes` and re-interprets the lifted reduction
/// over the physical batched value, with a requested output sharding gaining the mapped axis's sharding at the new
/// output batch axis position (mirroring the dot batching rule).
impl<C: Context<Type = ArrayType>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for ReduceOperation
where
    ReduceOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        // Validates that a mapped batch axis has a static size before lifting.
        ArrayBatch::common_batch_size(inputs)?;
        let Some(batch_axis) = inputs[0].batch_axis_position() else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        let (lifted_axes, output_axis) = lift_reduce_axes(self.axes.as_slice(), batch_axis);
        // A requested output sharding gains the mapped axis's sharding at the new output batch axis, mirroring the
        // dot batch rule.
        let lifted_output_sharding = match &self.output_sharding {
            Some(output_sharding) => Some(
                output_sharding
                    .with_inserted_dimension(output_axis, ArrayBatch::sharding_for_inputs(inputs)?)
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?,
            ),
            None => None,
        };
        let lifted_op = ReduceOperation::new(lifted_axes, self.kind).with_output_sharding(lifted_output_sharding);
        lifted_op.interpret_with_batch_axes(context, inputs, &[BatchAxis::from_position(output_axis)])
    }
}

/// Lifts a reduce's `axes` through one batching level inserted at `batch_axis`.
///
/// Returns the rewritten axes and the output batch axis position. Each user axis `i` shifts to
/// `i + 1` when `i >= batch_axis`. The output batch axis is `batch_axis` minus the number of
/// reduced axes that lie strictly below it (because those axes get dropped in the output).
///
/// The axes are expressed in the per-item coordinate system and therefore cannot name the inserted
/// batch dimension. In particular, a user axis equal to `batch_axis` shifts past the physical batch
/// dimension rather than reducing it.
pub fn lift_reduce_axes(axes: &[usize], batch_axis: usize) -> (Vec<usize>, usize) {
    let mut lifted = Vec::with_capacity(axes.len());
    let mut axes_below_batch = 0usize;
    for axis in axes {
        if *axis < batch_axis {
            lifted.push(*axis);
            axes_below_batch += 1;
        } else {
            lifted.push(*axis + 1);
        }
    }
    let output_batch_axis = batch_axis - axes_below_batch;
    (lifted, output_batch_axis)
}

/// Forward-mode rule for [`ReduceOperation`]. The additive reductions ([`Sum`](ReductionKind::Sum) /
/// [`Mean`](ReductionKind::Mean)) are linear in the operand, so the tangent is the same reduction applied to the
/// operand tangent. [`Max`](ReductionKind::Max) / [`Min`](ReductionKind::Min) route their tangent through a
/// primal-domain argmax mask: the tangent of `reduce_max(x)` along the reduced axes is `reduce_sum(mask * Δx)`, where
/// `mask` equals `1` exactly at the per-reduction extremal positions (ties split evenly, matching the JAX convention).
/// The mask is staged capture-free as ordinary primal operations — a `compare` of the operand primal against the
/// broadcast-back reduced value, followed by an ordinary `mul` against the operand tangent — so no residual factor is
/// captured. [`Any`](ReductionKind::Any) / [`All`](ReductionKind::All) are Boolean reductions with no tangent and are
/// rejected with [`UnsupportedOperation`](ProgramError::UnsupportedOperation). The shared all-zero fast path handles a
/// zero operand tangent before this rule is consulted, so the operand tangent reaching every supported case is live.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for ReduceOperation
where
    C::Operation: From<ReduceOperation>
        + From<LegacyBroadcastOperation>
        + From<CompareOperation<ArrayType>>
        + From<DivOperation<ArrayType>>
        + From<MulOperation<ArrayType>>,
    C::Value: Reduce
        + LegacyBroadcast
        + Compare<C::Value>
        + Div<Output = C::Value>
        + ElementwiseDerivativeAlignment<ArrayType>
        + Mul<Output = C::Value>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        match self.kind() {
            ReductionKind::Sum | ReductionKind::Mean => {
                let reduce = |value: &C::Value| match self.output_sharding() {
                    Some(output_sharding) => {
                        value.reduce_with_output_sharding(self.axes(), self.kind(), output_sharding)
                    }
                    None => value.reduce(self.axes(), self.kind()),
                };
                let primal = reduce(inputs[0].primal());
                let tangent = match inputs[0].tangent() {
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                    MaybeZero::Value(tangent) => MaybeZero::Value(reduce(tangent)),
                };
                Ok(vec![DifferentiationDual::new(primal, tangent)?])
            }
            kind @ (ReductionKind::Max | ReductionKind::Min) => {
                // Stage the argmax mask from the operand primal capture-free: `compare` the operand primal against the
                // broadcast-back reduced value (an ordinary `compare`/`broadcast`), convert it to the tangent type,
                // normalize it by the number of ties, and route the operand tangent through that normalized mask.
                let primal_input = inputs[0].primal();
                let primal = primal_input.reduce(self.axes(), kind);
                let input_type = primal_input.r#type().into_owned();
                let output_axes = output_to_input_axis_map(input_type.rank(), self.axes());
                let broadcast_primal = primal.legacy_broadcast(input_type, output_axes.as_slice())?;
                let mask = primal_input.compare(&broadcast_primal, ComparisonDirection::Equal)?;
                let tangent = match inputs[0].tangent() {
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                    MaybeZero::Value(input_tangent) => {
                        let numeric_mask = mask.align_tangent(input_tangent.r#type().as_ref())?;
                        let tie_count = numeric_mask.clone().reduce(self.axes(), ReductionKind::Sum);
                        let masked_tangent = numeric_mask * input_tangent.clone();
                        MaybeZero::Value(masked_tangent.reduce(self.axes(), ReductionKind::Sum) / tie_count)
                    }
                };
                Ok(vec![DifferentiationDual::new(primal, tangent)?])
            }
            kind => Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "array operation `reduce with kind {kind:?}` is not supported by the forward-mode \
                     linearization slice; any and all are not differentiable",
                ),
            }
            .into()),
        }
    }
}

/// Parent-context JVP rule for [`ReduceOperation`]. Fully static reductions delegate to the homogeneous projected
/// rule. Dynamically shaped numeric reductions retain their exact input extents as ordinary residual values so their
/// transpose can broadcast cotangents back to the runtime input shape. Maximum and minimum additionally retain the
/// normalized extremum mask, while mean computes its divisor from the retained reduced-axis extents.
impl<C> MemberDifferentiableOperation<C> for ReduceOperation
where
    C: Context<Type = ArrayIrType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<BroadcastOperation>
        + From<DimensionSizeOperation>
        + From<DimensionToScalarOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + OperationProjection<ArrayType>
        + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: DifferentiableOperation<ProjectedContext<C, ArrayType>>
        + From<CompareOperation<ArrayType>>
        + From<ConvertElementTypeOperation<ArrayType>>
        + From<DivOperation<ArrayType>>
        + From<MulOperation<ArrayType>>
        + From<ReduceOperation>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let [operand] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };
        let operand_type = <&ArrayType>::try_from(operand.primal().r#type().as_ref())?.clone();
        if operand_type.shape().dimensions().iter().all(|dimension| matches!(dimension, Dimension::Static(_)))
            || matches!(self.kind(), ReductionKind::Any | ReductionKind::All)
        {
            let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
            return jvp_projected_operation(context, &operation, inputs);
        }

        let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
        let primal = context.bind(operation, Vec::new(), std::slice::from_ref(operand.primal()))?.remove(0);
        let tangent = match operand.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
            MaybeZero::Value(operand_tangent) => {
                let mut residuals = LinearResiduals::new();
                let operand_shape = residuals.retain_shape(context, operand.primal())?;
                match self.kind() {
                    ReductionKind::Max | ReductionKind::Min => {
                        let input_extents = operand_shape.dimensions(context, residuals.values())?;
                        let output_axes = output_to_input_axis_map(operand_type.rank(), self.axes());
                        let mut broadcast_inputs = Vec::with_capacity(1 + input_extents.len());
                        broadcast_inputs.push(primal.clone());
                        broadcast_inputs.extend(input_extents.iter().cloned());
                        let broadcast_primal = context
                            .bind(
                                BroadcastOperation::new(output_axes.clone()),
                                Vec::new(),
                                broadcast_inputs.as_slice(),
                            )?
                            .remove(0);
                        let mask = context
                            .bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                    CompareOperation::new(ComparisonDirection::Equal),
                                ),
                                Vec::new(),
                                &[operand.primal().clone(), broadcast_primal],
                            )?
                            .remove(0);
                        let numeric_mask = context
                            .bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                    ConvertElementTypeOperation::new(operand_type.tangent().data_type()),
                                ),
                                Vec::new(),
                                &[mask],
                            )?
                            .remove(0);
                        let tie_count = context
                            .bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                    ReduceOperation::new(self.axes().to_vec(), ReductionKind::Sum),
                                ),
                                Vec::new(),
                                std::slice::from_ref(&numeric_mask),
                            )?
                            .remove(0);
                        let mut tie_broadcast_inputs = Vec::with_capacity(1 + input_extents.len());
                        tie_broadcast_inputs.push(tie_count);
                        tie_broadcast_inputs.extend(input_extents);
                        let broadcast_tie_count = context
                            .bind(
                                BroadcastOperation::new(output_axes.clone()),
                                Vec::new(),
                                tie_broadcast_inputs.as_slice(),
                            )?
                            .remove(0);
                        let normalized_mask = context
                            .bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(DivOperation::new()),
                                Vec::new(),
                                &[numeric_mask, broadcast_tie_count],
                            )?
                            .remove(0);
                        let mask_index = residuals.retain(normalized_mask);
                        let forward_axes = self.axes().to_vec();
                        let transpose_shape = operand_shape.clone();
                        let transpose_output_axes = output_axes.clone();
                        let transpose_target_type = operand_type.cotangent();
                        let tangent = LinearCallOperation::stage(
                            context,
                            residuals.into_values(),
                            vec![operand_tangent.clone()],
                            move |residuals, linear_inputs| {
                                let forward_context = linear_inputs[0].dispatch_domain();
                                let masked_tangent = forward_context
                                    .bind(
                                        <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                            MulOperation::new(),
                                        ),
                                        Vec::new(),
                                        &[residuals[mask_index].clone(), linear_inputs[0].clone()],
                                    )?
                                    .remove(0);
                                forward_context.bind(
                                    <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                        ReduceOperation::new(forward_axes.clone(), ReductionKind::Sum),
                                    ),
                                    Vec::new(),
                                    &[masked_tangent],
                                )
                            },
                            move |residuals, output_cotangents| {
                                let transpose_context = output_cotangents[0].dispatch_domain();
                                let input_extents = transpose_shape.dimensions(&transpose_context, residuals)?;
                                let mut broadcast_inputs = Vec::with_capacity(1 + input_extents.len());
                                broadcast_inputs.push(output_cotangents[0].clone());
                                broadcast_inputs.extend(input_extents);
                                let broadcasted = transpose_context
                                    .bind(
                                        BroadcastOperation::new(transpose_output_axes.clone())
                                            .with_output_sharding(transpose_target_type.sharding().cloned()),
                                        Vec::new(),
                                        broadcast_inputs.as_slice(),
                                    )?
                                    .remove(0);
                                transpose_context.bind(
                                    <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                        MulOperation::new(),
                                    ),
                                    Vec::new(),
                                    &[residuals[mask_index].clone(), broadcasted],
                                )
                            },
                        )?
                        .remove(0);
                        MaybeZero::Value(tangent)
                    }
                    kind @ (ReductionKind::Sum | ReductionKind::Mean) => {
                        let forward_operation = self.clone();
                        let transpose_operand_type = operand_type.cotangent();
                        let transpose_axes = self.axes().to_vec();
                        let transpose_output_axes =
                            output_to_input_axis_map(transpose_operand_type.rank(), &transpose_axes);
                        let tangent = LinearCallOperation::stage(
                            context,
                            residuals.into_values(),
                            vec![operand_tangent.clone()],
                            move |_, linear_inputs| {
                                linear_inputs[0].dispatch_domain().bind(
                                    <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                        forward_operation,
                                    ),
                                    Vec::new(),
                                    std::slice::from_ref(&linear_inputs[0]),
                                )
                            },
                            move |residuals, output_cotangents| {
                                let transpose_context = output_cotangents[0].dispatch_domain();
                                let input_extents = operand_shape.dimensions(&transpose_context, residuals)?;
                                let mut broadcast_inputs = Vec::with_capacity(1 + input_extents.len());
                                broadcast_inputs.push(output_cotangents[0].clone());
                                broadcast_inputs.extend(input_extents.iter().cloned());
                                let broadcasted = transpose_context
                                    .bind(
                                        BroadcastOperation::new(transpose_output_axes.clone())
                                            .with_output_sharding(transpose_operand_type.sharding().cloned()),
                                        Vec::new(),
                                        broadcast_inputs.as_slice(),
                                    )?
                                    .remove(0);
                                if kind == ReductionKind::Sum {
                                    return Ok(vec![broadcasted]);
                                }

                                let mut element_count = transpose_context
                                    .bind(
                                        DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(1)?)),
                                        Vec::new(),
                                        &[],
                                    )?
                                    .remove(0);
                                for axis in &transpose_axes {
                                    let left_type =
                                        <&DimensionType>::try_from(element_count.r#type().as_ref())?.clone();
                                    let right_type =
                                        <&DimensionType>::try_from(input_extents[*axis].r#type().as_ref())?.clone();
                                    element_count = transpose_context
                                        .bind(
                                            DimensionOperation::Mul(DimensionMulOperation::new(
                                                &left_type,
                                                &right_type,
                                            )?),
                                            Vec::new(),
                                            &[element_count, input_extents[*axis].clone()],
                                        )?
                                        .remove(0);
                                }
                                let element_count = transpose_context
                                    .bind(DimensionToScalarOperation, Vec::new(), &[element_count])?
                                    .remove(0);
                                let element_count = transpose_context
                                    .bind(
                                        <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                            ConvertElementTypeOperation::new(transpose_operand_type.data_type()),
                                        ),
                                        Vec::new(),
                                        &[element_count],
                                    )?
                                    .remove(0);
                                transpose_context.bind(
                                    <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                        DivOperation::new(),
                                    ),
                                    Vec::new(),
                                    &[broadcasted, element_count],
                                )
                            },
                        )?
                        .remove(0);
                        MaybeZero::Value(tangent)
                    }
                    ReductionKind::Any | ReductionKind::All => unreachable!("Boolean reductions delegated above"),
                }
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Transpose (vector-Jacobian product) for a [`ReduceOperation`].
///
/// For a `Sum` reduction, the cotangent of the input is the output cotangent broadcast back to
/// the input shape — singleton-broadcasting over each reduced axis. For a `Mean` reduction, the
/// same broadcast-back result is additionally scaled by `1 / N` where `N` is the product of the
/// reduced axis extents. `Max`/`Min` would need an argmax-style gather to route the cotangent
/// only to the element that produced the reduction's output, and `Any`/`All` are not
/// differentiable.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for ReduceOperation
where
    O: Operation<Type = ArrayType>
        + From<LegacyBroadcastOperation>
        + From<ConstantOperation<crate::backends::arrays::Array>>
        + From<MulOperation<ArrayType>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        let input_type = inputs[0].r#type();
        let input_shape = input_type.shape();
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(input_type.cotangent())]),
            MaybeZero::Value(cotangent) => match self.kind {
                ReductionKind::Sum | ReductionKind::Mean => {
                    let output_type = input_type.cotangent();
                    let output_axes = output_to_input_axis_map(input_shape.rank(), &self.axes);
                    let broadcasted = cotangent.legacy_broadcast(output_type, output_axes.as_slice())?;
                    let cotangent_input = match self.kind {
                        ReductionKind::Sum => broadcasted,
                        ReductionKind::Mean => {
                            let reduced_extents = self
                                .axes
                                .iter()
                                .map(|axis| {
                                    input_shape.dimension(*axis).value().ok_or(TypeError::invalid(format!(
                                        "mean transpose requires static reduced extents but axis {axis} of \
                                            {input_shape} is dynamic",
                                    )))
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let element_count = if reduced_extents.contains(&0) {
                                0
                            } else {
                                reduced_extents.iter().try_fold(1usize, |count, extent| {
                                    count.checked_mul(*extent).ok_or_else(|| {
                                        TypeError::invalid(format!(
                                            "mean transpose reduced element count overflows usize for input shape \
                                             {input_shape}",
                                        ))
                                    })
                                })?
                            };
                            let inverse_count = 1.0 / element_count as f64;
                            // Stage a rank-zero literal holding `1 / N` and rely on implicit rank-zero broadcasting in
                            // the subsequent multiplication to scale the broadcast-back cotangent to the input shape.
                            let factor_type = ArrayType::new(cotangent.r#type().data_type(), Shape::scalar());
                            let factor = context.fill(&factor_type, inverse_count)?;
                            factor * broadcasted
                        }
                        _ => unreachable!("outer match handled the only two supported kinds"),
                    };
                    Ok(vec![MaybeZero::Value(cotangent_input)])
                }
                other => Err(TypeError::invalid(format!(
                    "reduce transpose for {other} is not yet supported; only Sum and Mean are wired \
                        (Max/Min need argmax-style gather; Any/All are not differentiable)"
                ))
                .into()),
            },
        }
    }
}

/// Builds the `output_axes` vector that maps a reduced output's axes back to the
/// corresponding input axes. Output axis `j` corresponds to the `j`-th non-reduced input axis;
/// the returned vector lists those input-axis indices in order.
fn output_to_input_axis_map(input_rank: usize, reduced_axes: &[usize]) -> Vec<usize> {
    let mut reduce_mask = vec![false; input_rank];
    for axis in reduced_axes {
        reduce_mask[*axis] = true;
    }
    (0..input_rank).filter(|axis| !reduce_mask[*axis]).collect()
}

/// Value-level reduction capability.
///
/// [`Reduce`] is the receiver-style entry point for staging or executing a [`ReduceOperation`]: it reduces the
/// receiver along `axes` using the operator/identity pair described by `kind`, returning a value whose rank is
/// `self.rank() - axes.len()`.
pub trait Reduce: Sized {
    /// Reduces `self` along `axes` using the operator selected by `kind`.
    fn reduce(&self, axes: &[usize], kind: ReductionKind) -> Self;

    /// Reduces `self` along `axes` using `kind`, requesting `output_sharding` for the result (refer to the
    /// documentation of [`ReduceOperation::with_output_sharding`]). The default implementation ignores the requested
    /// sharding and delegates to [`Self::reduce`], which is correct for concrete (single-device) values, for which a
    /// sharding only describes distribution metadata; staging implementations override this to attach the requested
    /// sharding to the staged operation.
    fn reduce_with_output_sharding(&self, axes: &[usize], kind: ReductionKind, output_sharding: &Sharding) -> Self {
        let _ = output_sharding;
        self.reduce(axes, kind)
    }
}

/// Any context-carrying value reduces by binding a [`ReduceOperation`] through its own context. The
/// `From<ReduceOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Reduce for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<ReduceOperation>,
{
    #[inline]
    fn reduce(&self, axes: &[usize], kind: ReductionKind) -> Self {
        if axes.is_empty() {
            return self.clone();
        }
        self.dispatch_domain()
            .bind(ReduceOperation::new(axes.to_vec(), kind), Vec::new(), &[self.clone()])
            .expect("`reduce` operation failed")
            .remove(0)
    }

    #[inline]
    fn reduce_with_output_sharding(&self, axes: &[usize], kind: ReductionKind, output_sharding: &Sharding) -> Self {
        self.dispatch_domain()
            .bind(
                ReduceOperation::new(axes.to_vec(), kind).with_output_sharding(output_sharding.clone()),
                Vec::new(),
                &[self.clone()],
            )
            .expect("`reduce` operation failed")
            .remove(0)
    }
}

/// Reduction evaluation helper that operates on a flat row-major payload and shape.
///
/// Returns `(reduced_values, reduced_shape)`. `axes` may be in any order; duplicates are not
/// permitted (callers should validate beforehand). The `combiner` function applies the reduction
/// operator and `identity` returns the initial accumulator value for each output cell.
///
/// # Parameters
///
///   - `values`: Row-major input payload.
///   - `shape`: Input shape.
///   - `axes`: Axes to reduce.
///   - `identity`: Initial accumulator value for each output element.
///   - `combiner`: Binary reduction operator.
pub fn reduce_evaluate<T: Clone>(
    values: &[T],
    shape: &StaticShape,
    axes: &[usize],
    identity: impl Fn() -> T,
    combiner: impl Fn(T, T) -> T,
) -> (Vec<T>, StaticShape) {
    let rank = shape.rank();
    let mut reduce_mask = vec![false; rank];
    for axis in axes {
        reduce_mask[*axis] = true;
    }
    let output_shape = StaticShape::new(
        shape
            .dimensions()
            .iter()
            .enumerate()
            .filter_map(|(axis, size)| if reduce_mask[axis] { None } else { Some(*size) })
            .collect(),
    );
    let output_element_count: usize = output_shape.dimensions().iter().product();
    let mut output = (0..output_element_count).map(|_| identity()).collect::<Vec<_>>();
    if output_element_count == 0 {
        return (output, output_shape);
    }

    let input_strides = shape.row_major_strides();
    let output_strides = output_shape.row_major_strides();

    let mut input_index = vec![0usize; rank];
    let input_element_count: usize = shape.dimensions().iter().product();
    if input_element_count == 0 {
        return (output, output_shape);
    }

    loop {
        let mut input_flat = 0usize;
        let mut output_flat = 0usize;
        let mut output_axis = 0usize;
        for (axis, position) in input_index.iter().enumerate() {
            input_flat += position * input_strides[axis];
            if !reduce_mask[axis] {
                output_flat += position * output_strides[output_axis];
                output_axis += 1;
            }
        }
        output[output_flat] = combiner(output[output_flat].clone(), values[input_flat].clone());

        let mut position = rank;
        let mut carry = true;
        while position > 0 && carry {
            position -= 1;
            input_index[position] += 1;
            if input_index[position] < shape[position] {
                carry = false;
            } else {
                input_index[position] = 0;
            }
        }
        if carry {
            return (output, output_shape);
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::contexts::StagingContext;
    use crate::differentiation::{jvp, value_and_gradient};
    use crate::macros::check_operation_batching;
    use crate::programs::types::Typed;

    use super::*;

    fn array_type(dimensions: &[usize], data_type: DataType) -> ArrayType {
        ArrayType::new(data_type, Shape::new(dimensions.iter().copied().map(Dimension::Static).collect()))
    }

    #[test]
    fn test_reduce_abstract_drops_reduced_axes_and_keeps_remaining_order() {
        let input = array_type(&[2, 3, 4], DataType::F64);
        assert_eq!(
            reduce_abstract(&input, &[1], ReductionKind::Sum, "reduce_sum"),
            Ok(array_type(&[2, 4], DataType::F64))
        );
        assert_eq!(
            reduce_abstract(&input, &[0, 2], ReductionKind::Max, "reduce_max"),
            Ok(array_type(&[3], DataType::F64))
        );
    }

    #[test]
    fn test_reduce_abstract_drops_sharded_reduced_axis_entries() {
        use crate::arrays::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        // Reducing over a sharded dimension deletes its entry without error (the partitioner owns the collective);
        // the surviving dimension keeps its sharding and the reduced manual axis set passes through.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = array_type(&[2, 3], DataType::F64)
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["r"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            reduce_abstract(&input, &[0], ReductionKind::Sum, "reduce_sum"),
            Ok(array_type(&[3], DataType::F64)
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::replicated()])
                        .unwrap()
                        .with_reduced_axes(["r"])
                        .unwrap(),
                )
                .unwrap()),
        );
    }

    #[test]
    fn test_reduce_abstract_propagates_dynamic_dimensions() {
        // Dynamic dimensions flow through reduce inference: reduced axes are dropped whether they are static or
        // dynamic, and the remaining dynamic dimensions are preserved in order.
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let width = DimensionVariable::new("width", DimensionBounds::non_negative(Some(4)).unwrap());
        let input = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(batch.clone()),
                Dimension::Static(3),
                Dimension::Dynamic(width.clone()),
            ]),
        );
        assert_eq!(
            reduce_abstract(&input, &[1], ReductionKind::Sum, "reduce_sum"),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Dynamic(width)]),)),
        );
        assert_eq!(
            reduce_abstract(&input, &[0, 2], ReductionKind::Sum, "reduce_sum"),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))),
        );
    }

    #[test]
    fn test_reduce_abstract_rejects_out_of_bounds_and_duplicate_axes() {
        let input = array_type(&[2, 3], DataType::F64);
        assert!(reduce_abstract(&input, &[2], ReductionKind::Sum, "reduce_sum").is_err());
        assert!(reduce_abstract(&input, &[0, 0], ReductionKind::Sum, "reduce_sum").is_err());
    }

    #[test]
    fn test_reduce_abstract_enforces_reduction_data_types() {
        let numeric = array_type(&[2, 3], DataType::F64);
        assert!(reduce_abstract(&numeric, &[1], ReductionKind::Any, "reduce_any").is_err());
        let boolean = array_type(&[2, 3], DataType::Boolean);
        assert!(reduce_abstract(&boolean, &[1], ReductionKind::Sum, "reduce_sum").is_err());
        assert_eq!(
            reduce_abstract(&boolean, &[1], ReductionKind::Any, "reduce_any"),
            Ok(array_type(&[2], DataType::Boolean))
        );
        assert_eq!(
            reduce_abstract(&boolean, &[1], ReductionKind::Max, "reduce_max"),
            Ok(array_type(&[2], DataType::Boolean)),
        );
        let token = array_type(&[2, 3], DataType::Token);
        assert_eq!(
            reduce_abstract(&token, &[1], ReductionKind::Sum, "reduce_sum"),
            Err(TypeError::invalid("'reduce_sum' kind sum requires numeric inputs but got token".to_string())),
        );
        // The structural-zero element type represents an already-known zero tangent and remains closed under numeric
        // reductions even though it has no numeric payload bytes.
        let zero = array_type(&[2, 3], DataType::Zero);
        assert_eq!(
            reduce_abstract(&zero, &[1], ReductionKind::Sum, "reduce_sum"),
            Ok(array_type(&[2], DataType::Zero)),
        );
    }

    #[test]
    fn test_reduce_abstract_accepts_lexicographic_complex_extrema() {
        // Complex minimum and maximum use JAX's lexicographic `(real, imaginary)` ordering.
        let complex = array_type(&[2, 3], DataType::C64);
        assert_eq!(
            reduce_abstract(&complex, &[1], ReductionKind::Max, "reduce_max"),
            Ok(array_type(&[2], DataType::C64)),
        );
        assert_eq!(
            reduce_abstract(&complex, &[1], ReductionKind::Min, "reduce_min"),
            Ok(array_type(&[2], DataType::C64)),
        );
        assert_eq!(
            reduce_abstract(&complex, &[1], ReductionKind::Sum, "reduce_sum"),
            Ok(array_type(&[2], DataType::C64)),
        );
    }

    #[test]
    fn test_reduce_operation_infer_output_types_follows_the_input_type() {
        // The operation carries no input shape; the output type is derived from the actual staged
        // input type, and out-of-range axes are rejected against it.
        let operation = ReduceOperation::new(vec![1], ReductionKind::Sum);
        let input = array_type(&[3, 2], DataType::F64);
        assert_eq!(operation.infer_output_types(&[input], &[]), Ok(vec![array_type(&[3], DataType::F64)]));
        assert_eq!(
            operation.infer_output_types(&[array_type(&[3], DataType::F64)], &[]),
            Err(TypeError::invalid("'reduce_sum' axis 1 is out of bounds for rank 1".to_string())),
        );
    }

    #[test]
    fn test_reduce_sum_output_sharding_requests_unreduced_output() {
        use crate::arrays::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
        use crate::programs::operations::Operation;

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        // Input: dimension 0 sharded over `x`, dimension 1 replicated; reducing over the `x`-sharded dimension 0.
        let input = array_type(&[2, 3], DataType::F64)
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        // A matching unreduced output (deferring the `x` reduction) is accepted.
        let unreduced = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let operation = ReduceOperation::new(vec![0], ReductionKind::Sum).with_output_sharding(unreduced.clone());
        assert_eq!(operation.output_sharding(), Some(&unreduced));
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&input), &[]),
            Ok(vec![array_type(&[3], DataType::F64).with_sharding(unreduced.clone()).unwrap()]),
        );
        // The output sharding renders only when present.
        assert!(operation.to_string().contains(&format!("output_sharding={unreduced}")));
        assert!(!ReduceOperation::new(vec![0], ReductionKind::Sum).to_string().contains("output_sharding="));

        // Requesting an unreduced axis that did not shard a summed-over dimension is rejected.
        let wrong = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["y"])
            .unwrap();
        assert_eq!(
            ReduceOperation::new(vec![0], ReductionKind::Sum)
                .with_output_sharding(wrong)
                .infer_output_types(std::slice::from_ref(&input), &[]),
            Err(TypeError::invalid(
                "reduce_sum output sharding unreduced axes must be among the explicit axes sharding the \
                          reduced dimensions or the operand's unreduced axes"
                    .to_string()
            )),
        );

        // Only reduce_sum accepts a requested output sharding.
        assert_eq!(
            ReduceOperation::new(vec![0], ReductionKind::Max)
                .with_output_sharding(unreduced)
                .infer_output_types(std::slice::from_ref(&input), &[]),
            Err(TypeError::invalid(
                "max does not support a requested output sharding (only reduce_sum does)".to_string()
            )),
        );
    }

    #[test]
    fn test_reduce_operation_interprets_sum_over_axis() {
        use crate::arrays::{
            Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, Sharding, ShardingDimension, StridedLayout,
        };

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = array_type(&[2, 3], DataType::F64)
            .with_layout(Layout::Strided(StridedLayout::new(vec![24, 8])))
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        let input = Array::from_f64s(input_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let outputs = ReduceOperation::new(vec![1], ReductionKind::Sum)
            .interpret(&crate::EagerContext::<Array>::new(), &crate::EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        let output = outputs.into_iter().next().unwrap();
        // The payload kernel and abstract rule must agree on the complete result type: reduction projects sharding,
        // preserves memory placement, and clears the rank-specific layout.
        assert_eq!(
            output.r#type().as_ref(),
            &array_type(&[2], DataType::F64)
                .with_memory(Memory::Host { pinned: true })
                .with_sharding(output_sharding)
                .unwrap(),
        );
        assert_eq!(output.to_f64s(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_reduce_with_output_sharding_stages_through_the_capability() {
        use std::rc::Rc;

        use crate::arrays::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
        use crate::parameters::Placeholder;
        use crate::tracing::TracingContext;

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = array_type(&[2, 3], DataType::F64)
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let unreduced = Sharding::new(mesh, vec![ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();

        // Staging `reduce_with_output_sharding` on a tracer must carry the requested sharding through the capability,
        // the staged `ReduceOperation`, and the `ArrayOperation::Reduce` variant into the built program.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let builder = context.builder().clone();
        let input_atom = builder.borrow_mut().add_input(input_type);
        let output = context.tracer(input_atom, None).reduce_with_output_sharding(&[0], ReductionKind::Sum, &unreduced);
        let output_atom = output.atom_id().unwrap();
        drop(output);
        drop(context);

        let program = Rc::try_unwrap(builder)
            .expect("staging should not retain the builder")
            .into_inner()
            .build::<Vec<Array>, Vec<Array>>(vec![output_atom], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(program.to_string().contains(&format!("output_sharding={unreduced}")));

        // Linearization must preserve the requested sharding on both applications of the linear reduction: the
        // primal reduction and the same reduction applied to the tangent. Otherwise differentiation silently turns
        // a requested per-shard partial sum into the default reduced result.
        let linearization = program.linearize().unwrap();
        let expected_output_type = array_type(&[3], DataType::F64).with_sharding(unreduced.clone()).unwrap();
        assert_eq!(linearization.primal().output_types()[0], expected_output_type);
        assert_eq!(linearization.tangent().output_types()[0], expected_output_type);
        assert!(linearization.primal().to_string().contains(&format!("output_sharding={unreduced}")));
        assert!(linearization.tangent().to_string().contains(&format!("output_sharding={unreduced}")));
    }

    #[test]
    fn test_reduce_operation_batches_replicated_input_as_pass_through() {
        check_operation_batching!(
            @exact,
            operation = ReduceOperation::new(vec![1], ReductionKind::Sum),
            axis_size = 2,
            cases = [{
                inputs = [(@replicated, Array::matrix(2, 3, vec![1.0; 6]))],
                outputs = [(@replicated, Array::vector(vec![3.0, 3.0]))],
            }],
        );
    }

    #[test]
    fn test_reduce_operation_batches_along_non_batch_axis() {
        // Physical input is [3 batch items, 2 rows, 3 cols] mapped at axis 0. Per-item reduce over
        // axis 1 (the "cols" axis from the per-item view; physically axis 2 after batching).
        check_operation_batching!(
            @exact,
            operation = ReduceOperation::new(vec![1], ReductionKind::Sum),
            axis_size = 3,
            cases = [{
                inputs = [(@mapped(
                    axis = 0
                ), Array::from_f64s(
                    array_type(&[3, 2, 3], DataType::F64),
                    (0..18).map(|index| index as f64).collect(),
                ))],
                outputs = [(@mapped(
                    axis = 0
                ), Array::matrix(3, 2, vec![3.0, 12.0, 21.0, 30.0, 39.0, 48.0]))],
            }],
        );
    }

    #[test]
    fn test_reduce_operation_batches_a_per_item_axis_at_the_physical_batch_position() {
        check_operation_batching!(
            @exact,
            operation = ReduceOperation::new(vec![0], ReductionKind::Sum),
            axis_size = 3,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::matrix(3, 2, vec![1.0; 6]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![2.0, 2.0, 2.0]))],
            }],
        );
    }

    #[test]
    fn test_lift_reduce_axes_shifts_axes_above_batch_and_keeps_axes_below() {
        // Per-item reduce over axes [0, 2] of a rank-3 input. Batching at axis 1 inserts a new
        // dimension at position 1, so per-item axis 0 stays at 0, per-item axis 2 shifts to 3.
        // Output batch axis is at position 1 - 1 = 0 (one reduced axis was below the batch axis).
        assert_eq!(lift_reduce_axes(&[0, 2], 1), (vec![0, 3], 0));
        // Reducing only above the batch axis leaves the batch axis position unchanged.
        assert_eq!(lift_reduce_axes(&[2], 0), (vec![3], 0));
        // A per-item axis at the physical batch position shifts past the inserted batch dimension.
        assert_eq!(lift_reduce_axes(&[0, 1], 1), (vec![0, 2], 0));
    }

    #[test]
    fn test_reduce_extrema_derivatives_split_ties_evenly() {
        for kind in [ReductionKind::Max, ReductionKind::Min] {
            let input = Array::vector(vec![1.0, 1.0]);
            let (primal, tangent) =
                jvp(|input| Ok(input.reduce(&[0], kind)), input.clone(), Array::vector(vec![1.0, 3.0])).unwrap();
            assert_eq!(primal.to_f64s(), vec![1.0]);
            assert_eq!(tangent.to_f64s(), vec![2.0]);

            let (primal, gradient) =
                value_and_gradient(|input| Ok::<_, ProgramError>(input.reduce(&[0], kind)), input).unwrap();
            assert_eq!(primal.to_f64s(), vec![1.0]);
            assert_abs_diff_eq!(gradient.to_f64s()[0], 0.5, epsilon = 1e-9);
            assert_abs_diff_eq!(gradient.to_f64s()[1], 0.5, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_reduce_mean_transpose_divides_by_axis_size() {
        // Mean over a length-4 axis: transpose maps a unit cotangent to a broadcast-back
        // cotangent of `1 / 4` at every input position.
        use std::rc::Rc;

        use crate::parameters::Placeholder;
        use crate::tracing::TracingContext;

        let input_shape = Shape::new(vec![Dimension::Static(4)]);
        let input_type = ArrayType::new(DataType::F64, input_shape.clone());
        let cotangent_type = ArrayType::scalar(DataType::F64);
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let transpose_builder = context.builder().clone();
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(cotangent_type);
        let output_cotangent = context.tracer(output_cotangent_atom, None);
        let contribution = ReduceOperation::new(vec![0], ReductionKind::Mean)
            .transpose(
                &mut context,
                &crate::programs::regions::EmptyRegionDriver,
                &[PartialValue::Unknown(input_type)],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap()
            .into_iter()
            .next()
            .expect("transpose should return one contribution");
        let MaybeZero::Value(contribution) = contribution else {
            panic!("transpose should produce one cotangent contribution");
        };
        let contribution_atom = contribution.atom_id().unwrap();
        drop(contribution);
        drop(context);
        let transpose_builder = Rc::try_unwrap(transpose_builder)
            .expect("transpose builder should not have outstanding linear terms")
            .into_inner();
        let transpose_program =
            transpose_builder.build::<Array, Array>(vec![contribution_atom], Placeholder, Placeholder).unwrap();
        let result = transpose_program.interpret(Array::scalar(1.0)).unwrap();
        assert_eq!(result.r#type().shape(), &input_shape);
        for value in result.to_f64s() {
            let delta = (value - 0.25).abs();
            assert!(delta < 1e-9, "expected ≈ 0.25, got {value}");
        }
    }

    #[test]
    fn test_reduce_mean_transpose_checks_reduced_element_count() {
        use crate::differentiation::DifferentiationError;
        use crate::partial::PartialValue;
        use crate::programs::ProgramError;
        use crate::programs::atoms::MaybeZero;
        use crate::programs::types::TypeError;
        use crate::tracing::TracingContext;

        let input_shape = Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2)]);
        let input_type = ArrayType::new(DataType::F64, input_shape.clone());
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = {
            let atom = context.builder().borrow_mut().add_input(ArrayType::scalar(DataType::F64));
            context.tracer(atom, None)
        };

        assert!(matches!(
            ReduceOperation::new(vec![0, 1], ReductionKind::Mean).transpose(
                &mut context,
                &crate::programs::regions::EmptyRegionDriver,
                &[PartialValue::Unknown(input_type)],
                &[MaybeZero::Value(output_cotangent)],
            ),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!(
                    "mean transpose reduced element count overflows usize for input shape {input_shape}",
                ),
        ));
    }

    #[test]
    fn test_reduce_mean_transpose_accepts_zero_reduced_element_count_without_overflow() {
        use crate::partial::PartialValue;
        use crate::programs::atoms::MaybeZero;
        use crate::tracing::TracingContext;

        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2), Dimension::Static(0)]),
        );
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = {
            let atom = context.builder().borrow_mut().add_input(ArrayType::scalar(DataType::F64));
            context.tracer(atom, None)
        };

        let contributions = ReduceOperation::new(vec![0, 1, 2], ReductionKind::Mean)
            .transpose(
                &mut context,
                &crate::programs::regions::EmptyRegionDriver,
                &[PartialValue::Unknown(input_type.clone())],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap();
        assert_eq!(contributions.len(), 1);
        assert_eq!(contributions[0].r#type().as_ref(), &input_type);
    }

    #[test]
    fn test_output_to_input_axis_map_handles_reduced_and_kept_axes() {
        // Input rank 3, reduce axis 1: output axes [0, 1] map back to input axes [0, 2].
        assert_eq!(super::output_to_input_axis_map(3, &[1]), vec![0, 2]);
        // Input rank 3, reduce axes [0, 2]: output axis [0] maps back to input axis [1].
        assert_eq!(super::output_to_input_axis_map(3, &[0, 2]), vec![1]);
        // Input rank 4, reduce axes [1, 3]: output axes [0, 1] map back to input axes [0, 2].
        assert_eq!(super::output_to_input_axis_map(4, &[1, 3]), vec![0, 2]);
        // No reduction: identity map.
        assert_eq!(super::output_to_input_axis_map(3, &[]), vec![0, 1, 2]);
    }

    #[test]
    fn test_reduce_evaluate_combines_along_specified_axes() {
        let values: Vec<f64> = (1..=24).map(|index| index as f64).collect();
        let (reduced, shape) = reduce_evaluate(
            values.as_slice(),
            &StaticShape::new(vec![2, 3, 4]),
            &[1],
            || 0.0,
            |acc, value| acc + value,
        );
        assert_eq!(shape, StaticShape::new(vec![2, 4]));
        // Row 0 sums across axis 1: [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]
        // Row 1 sums across axis 1: [13+17+21, 14+18+22, 15+19+23, 16+20+24] = [51, 54, 57, 60]
        assert_eq!(reduced, vec![15.0, 18.0, 21.0, 24.0, 51.0, 54.0, 57.0, 60.0]);
    }
}
