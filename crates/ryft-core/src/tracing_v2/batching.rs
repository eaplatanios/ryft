use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

use crate::ElementwiseOperation;
use crate::axes::{NamedAxes, NamedAxis};
use crate::batching::{
    ArrayBatch, BatchAxis, BatchAxisSpecification, BatchableOperation, BatchingError, ProgramBatchingOutputAxesPolicy,
};
use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, Domain, StagingContext, ValueResolution};
use crate::interpretation::InterpretableOperation;
use crate::macros::{check_builders, check_count};
use crate::operations::Operation;
use crate::operations::manipulation::{Broadcast, BroadcastOperation, Transpose, TransposeOperation};
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{DomainTracer, DomainTracingContext, Tracer, TracerState, TracingContext};
use crate::tracing_v2::differentiation::{DifferentiationContext, replay_via_bind};
use crate::types::{ArrayType, Size, Typed};

// TODO(eaplatanios): Review this module.

/// Blanket [`BatchableOperation`] impl for any [`ElementwiseOperation`].
///
/// Every [`ElementwiseOperation`] automatically gets the standard elementwise batching rule, so per-op
/// [`BatchableOperation`] impls do not have to be written for elementwise primitives (`Add`, `Sub`, `Mul`, `Div`,
/// `Neg`, `Sin`, `Cos`, `Select`, `ZeroLike`, `OneLike`, …). Ops with non-trivial axis arithmetic (`Dot`,
/// `Transpose`, `Reshape`, …) and the [`ArrayOperation`](crate::tracing_v2::ArrayOperation) operation enum (whose
/// impls live with the enum in [`operations::primitive`](crate::tracing_v2::operations::primitive)) keep their
/// explicit impls; coherence is preserved because none of those types implement [`ElementwiseOperation`].
impl<
    O: Clone + InterpretableOperation<ArrayType, V, C> + ElementwiseOperation,
    V: Value<ArrayType> + Broadcast + Transpose,
    C,
> BatchableOperation<V, C> for O
{
    #[inline]
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        apply_elementwise_batch(context, self, inputs)
    }
}

/// Applies a lifted operation to `inputs` via [`InterpretableOperation::interpret`] and packages
/// each output value with the corresponding entry of `output_axes`.
///
/// `output_axes` must have one entry per output produced by `lifted_op` on these inputs. This function is public so
/// that backend-owned operation enums (e.g., in `ryft-xla`) can implement [`BatchableOperation::batch`] for their
/// extension operations using the same application path as the built-in rules.
///
/// `context` supplies the value interpretation context directly. Active batching callers pass
/// [`BatchingContext::parent_context`] instead of recovering the context from input operands. This keeps lifted
/// interpretation well-defined when every operand is a symbolic zero and therefore carries no payload context.
pub fn apply_with_axes<V: Value<ArrayType>, C, O: InterpretableOperation<ArrayType, V, C>>(
    context: &C,
    lifted_op: &O,
    inputs: &[ArrayBatch<V>],
    output_axes: &[BatchAxis],
) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
    if inputs.is_empty() {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    }
    let input_values: Vec<V> = inputs.iter().map(|input| input.value().clone()).collect();
    let output_values = lifted_op.interpret(context, input_values.as_slice())?;
    check_count!("output", output_values, output_axes.len(), ProgramError);
    output_values
        .into_iter()
        .zip(output_axes.iter().copied())
        .map(|(value, axis)| ArrayBatch::new(value.r#type().into_owned(), value, axis))
        .collect()
}

/// Generic value-level batching helper for pure elementwise operations. Matches JAX's
/// `defbroadcasting` behavior: replicated inputs are broadcast to the common batched physical
/// shape before applying the operation, so each value-level primitive only ever sees inputs that
/// agree on shape at the boundary. This is the canonical implementation of
/// [`BatchableOperation::batch`] for elementwise primitives.
///
/// Inputs whose mapped batch axis is at a different physical position from the first batched
/// input are realigned with an inserted [`TransposeOperation`] before broadcasting, matching
/// JAX's `matchaxis` policy. The canonical axis position is the first batched input's axis.
pub(crate) fn apply_elementwise_batch<
    V: Value<ArrayType> + Broadcast + Transpose,
    C,
    O: InterpretableOperation<ArrayType, V, C>,
>(
    context: &C,
    operation: &O,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
    let unbatched_types: Vec<ArrayType> =
        inputs.iter().map(|input| input.unbatched_type()).collect::<Result<Vec<_>, _>>()?;
    let original_batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis().axis()).collect();
    // The elementwise rule broadcasts replicated operands uniformly, so an all-replicated input set (batch size
    // `None`) never reaches the axis arithmetic in the broadcasting branch below; `0` is an inert placeholder.
    let axis_size = ArrayBatch::common_batch_size(inputs)?.unwrap_or(0);
    let common_axis = original_batch_axes.iter().copied().flatten().next();
    let aligned_inputs: Vec<ArrayBatch<V>> = match common_axis {
        None => inputs.to_vec(),
        Some(target) => inputs.iter().map(|input| align_batch_axis(input, target)).collect::<Result<_, _>>()?,
    };
    // Realignment only moves each mapped axis to `common_axis`; the unbatched types are unchanged.
    let batch_axes: Vec<Option<usize>> = original_batch_axes.iter().map(|axis| axis.and(common_axis)).collect();
    let broadcasted_inputs = match common_axis {
        None => aligned_inputs,
        Some(batch_axis) => {
            // Mirroring JAX's `defbroadcasting` policy, every operand whose unbatched shape is narrower than the
            // common unbatched shape of all operands (trailing-aligned) is broadcast to that common shape with the
            // batch axis at `batch_axis`. When the operands' unbatched shapes are not broadcast-compatible, the
            // operands are left at their batch-axis-inserted physical shapes so the operation surfaces its own shape
            // error against the original shapes. Realignment preserves the unbatched types, so they are reused here.
            let common_unbatched = Broadcastable::broadcasted(unbatched_types.as_slice()).ok();
            // The common per-item shape only contributes its shape — each operand keeps its own data type (e.g., a
            // Boolean select condition broadcast against numeric branches stays Boolean) — and the mapped batch axis
            // is inserted at `batch_axis`.
            let broadcasted_physical_type = |unbatched_type: &ArrayType| -> Result<ArrayType, ProgramError> {
                let mut target = common_unbatched.as_ref().unwrap_or(unbatched_type).clone();
                target.data_type = unbatched_type.data_type();
                Ok(target.with_inserted_dimension(batch_axis, Size::Static(axis_size))?)
            };
            // Maps the operand's unbatched dimension `index` (trailing-aligned within the common unbatched shape) to
            // its position in the broadcast target, accounting for the batch axis insertion.
            let target_position = |unbatched_rank: usize, target_rank: usize, index: usize| {
                let position = (target_rank - unbatched_rank) + index;
                if position < batch_axis { position } else { position + 1 }
            };
            aligned_inputs
                .iter()
                .zip(batch_axes.iter())
                .zip(unbatched_types.iter())
                .map(|((input, axis), unbatched_type)| {
                    let physical_type = broadcasted_physical_type(unbatched_type)?;
                    if physical_type == *input.r#type() {
                        return Ok(input.clone());
                    }
                    let target_rank = physical_type.rank() - 1;
                    let output_axes: Vec<usize> = match axis {
                        // Mapped operand with a narrower unbatched shape: keep the batch axis fixed and trailing-align
                        // the remaining dimensions.
                        Some(_) => (0..input.r#type().rank())
                            .map(|dimension| match dimension.cmp(&batch_axis) {
                                std::cmp::Ordering::Equal => batch_axis,
                                std::cmp::Ordering::Less => {
                                    target_position(unbatched_type.rank(), target_rank, dimension)
                                }
                                std::cmp::Ordering::Greater => {
                                    target_position(unbatched_type.rank(), target_rank, dimension - 1)
                                }
                            })
                            .collect(),
                        None => (0..unbatched_type.rank())
                            .map(|dimension| target_position(unbatched_type.rank(), target_rank, dimension))
                            .collect(),
                    };
                    let broadcasted = input.value().clone().broadcast(physical_type.clone(), output_axes.as_slice())?;
                    ArrayBatch::new(physical_type, broadcasted, Some(batch_axis))
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    // Elementwise semantics are preserved by adding the batch dimension, so the lifted operation is the original one
    // applied to the batch-carrying physical operands. Its output count is inferred from those physical types, and
    // every output takes the common batch axis. For broadcast-incompatible per-item shapes this inference surfaces the
    // operation's own shape error, matching the shapes `apply_with_axes` would then interpret.
    let physical_input_types: Vec<ArrayType> =
        broadcasted_inputs.iter().map(|input| input.r#type().into_owned()).collect();
    let output_count = operation.infer_output_types(physical_input_types.as_slice())?.len();
    let output_axes = vec![BatchAxis::from(common_axis); output_count];
    apply_with_axes(context, operation, &broadcasted_inputs, &output_axes)
}

/// Realigns a batched input by moving its mapped batch axis to `target_axis`.
///
/// Identity case (already at `target_axis`, or unbatched) returns the input unchanged. Otherwise
/// stages a [`TransposeOperation`] via the receiver's [`Transpose`] impl and returns a new
/// [`ArrayBatch`] whose physical type and value reflect the realigned axis.
///
/// # Parameters
///
///   - `input`: Batched input to realign.
///   - `target_axis`: Desired position of the mapped batch axis in the output.
pub(crate) fn align_batch_axis<V: Value<ArrayType> + Transpose>(
    input: &ArrayBatch<V>,
    target_axis: usize,
) -> Result<ArrayBatch<V>, BatchingError> {
    let Some(current_axis) = input.batch_axis().axis() else {
        return Ok(input.clone());
    };
    if current_axis == target_axis {
        return Ok(input.clone());
    }
    let rank = input.r#type().rank();
    let permutation = move_axis_permutation(rank, current_axis, target_axis);
    let permuted_value = input.value().clone().transpose(permutation)?;
    let permuted_type = permuted_value.r#type().into_owned();
    ArrayBatch::new(permuted_type, permuted_value, Some(target_axis))
}

/// Broadcasts a replicated `operand` to gain a singleton batch axis at `target_axis`.
///
/// This is the canonical building block for mixed batched/unbatched primitive rules (e.g.,
/// [`DotOperation::batch`](crate::tracing_v2::operations::dot::DotOperation)) and for lifting
/// replicated residuals during linearization: it inserts a new axis at `target_axis` in the
/// operand's type, broadcasts the value to that shape, and returns the result as a batched
/// [`ArrayBatch`]. Elementwise rules instead broadcast replicated operands to the full common
/// batched shape inside [`apply_elementwise_batch`].
///
/// Returns an error when called on an already-batched input — callers are expected to dispatch
/// the replicated case explicitly.
///
/// # Parameters
///
///   - `operand`: Replicated input to lift.
///   - `target_axis`: Position of the inserted batch axis in the output.
///   - `axis_size`: Size of the inserted batch axis.
pub(crate) fn broadcast_to_batched<V: Value<ArrayType> + Broadcast>(
    operand: &ArrayBatch<V>,
    target_axis: usize,
    axis_size: usize,
) -> Result<ArrayBatch<V>, BatchingError> {
    if !operand.batch_axis().is_replicated() {
        return Err(BatchingError::MisalignedBatchAxes {
            message: "broadcast_to_batched expects a replicated operand but received a batched value".to_string(),
        }
        .into());
    }
    let per_item_type = operand.unbatched_type()?;
    let physical_type = per_item_type.with_inserted_dimension(target_axis, Size::Static(axis_size))?;
    let output_axes: Vec<usize> = (0..per_item_type.rank()).map(|i| if i < target_axis { i } else { i + 1 }).collect();
    let broadcasted = operand.value().clone().broadcast(physical_type.clone(), output_axes.as_slice())?;
    ArrayBatch::new(physical_type, broadcasted, Some(target_axis))
}

// TODO(eaplatanios): Should this be moved to `Program::batched`?
/// Batches a captured program into a standalone program over batch-carrying physical types.
///
/// This is the batching analog of symbolic program linearization: staged higher-order
/// batching rules use it to batch captured programs *without* concretizing any batch-item values, so that batched
/// control-flow and custom-derivative structure can be staged back into the enclosing trace. Unlike linearization,
/// batching does not split value spaces — the batched replay stays in one tracer space — so the packaging is one
/// fresh replay: the program is replayed through a [`BatchingContext`] over a fresh [`TracingContext`] trace (with
/// the capture parameter pinned to `V`, so bounds written against it match their obligations syntactically),
/// lifting every instruction through its [`BatchableOperation`] rule, and the resulting staged program is extracted
/// together with the requested output-axis policy.
///
/// Inputs whose `input_batch_axes[i]` is mapped at position `k` consume the original logical input type with a mapped
/// batch axis of size `axis_size` inserted at `k`, while replicated inputs enter at their original logical types.
/// [`ProgramBatchingOutputAxesPolicy::Natural`] keeps the mapped axes produced by the batching rules. This is what
/// staged control-flow batching needs, because branch/body outputs are normalized to the surrounding operation's
/// signature afterward. [`ProgramBatchingOutputAxesPolicy::AlignAllTo`] imposes a canonical output axis, which is what
/// custom-derivative re-wrapping needs so independently batched primal/JVP/forward/backward programs have mutually
/// consistent signatures.
///
/// # Parameters
///
///   - `program`: Captured flat program over per-item (logical) input and output types.
///   - `axis_size`: Size of the mapped batch axis.
///   - `input_batch_axes`: [`BatchAxis`] per program input (mapped at a position or replicated).
///   - `output_axes_policy`: Policy for packaging the batched program outputs.
pub fn batch_program<V, O>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    axis_size: usize,
    input_batch_axes: &[BatchAxis],
    output_axes_policy: ProgramBatchingOutputAxesPolicy,
) -> Result<(Program<ArrayType, V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>), BatchingError>
where
    V: Value<ArrayType> + 'static,
    O: Clone + Operation<ArrayType> + 'static,
    O: BatchableOperation<
            Tracer<TracingContext<ArrayType, V, O, V>>,
            BatchingContext<TracingContext<ArrayType, V, O, V>>,
        >,
    O: From<TransposeOperation> + From<BroadcastOperation>,
{
    let logical_input_types = program.input_types();
    let input_count = logical_input_types.len();
    check_count!("input", input_batch_axes, input_count, ProgramError);
    let parent_context: TracingContext<ArrayType, V, O, V> = TracingContext::new();
    let builder = parent_context.builder().clone();
    // Keep every tracer and context that holds a clone of `builder` inside this scope so that recovering the
    // builder below is a real ownership check.
    let (output_atom_ids, output_axes) = {
        let batching_context = BatchingContext::new(parent_context, axis_size);
        let mut input_values = Vec::with_capacity(input_count);
        for (logical_type, axis) in logical_input_types.iter().zip(input_batch_axes.iter()) {
            let physical_type = match axis.axis() {
                Some(position) => logical_type.with_inserted_dimension(position, Size::Static(axis_size))?,
                None => logical_type.clone(),
            };
            let atom = builder.borrow_mut().add_input(physical_type);
            input_values.push(batching_context.batched_value(atom, logical_type.clone(), *axis));
        }
        let output_values = batching_context.stage_program(program, input_values)?;
        let mut output_atom_ids = Vec::with_capacity(output_values.len());
        let mut output_axes = Vec::with_capacity(output_values.len());
        for output_value in output_values {
            match output_axes_policy {
                ProgramBatchingOutputAxesPolicy::Natural => {
                    let atom = output_value.atom_id()?;
                    output_axes.push(output_value.meta().batch_axis());
                    output_atom_ids.push(atom);
                }
                ProgramBatchingOutputAxesPolicy::AlignAllTo(target_axis) => {
                    let atom = output_value.atom_id()?;
                    let axis = output_value.meta().batch_axis().axis();
                    let logical_type = output_value.r#type().into_owned();
                    let physical_type = match axis {
                        Some(k) => logical_type.with_inserted_dimension(k, Size::Static(axis_size))?,
                        None => logical_type,
                    };
                    let parent_batch = ArrayBatch::new(
                        physical_type.clone(),
                        batching_context.parent_context().tracer(atom, Some(physical_type)),
                        axis,
                    )?;
                    let aligned_batch = match axis {
                        Some(axis) if axis == target_axis => parent_batch,
                        Some(_) => align_batch_axis(&parent_batch, target_axis)?,
                        None => broadcast_to_batched(&parent_batch, target_axis, axis_size)?,
                    };
                    output_atom_ids.push(aligned_batch.into_value().atom_id()?);
                    output_axes.push(BatchAxis::new(target_axis));
                }
            }
        }
        Ok::<_, ProgramError>((output_atom_ids, output_axes))
    }?;
    let output_count = output_atom_ids.len();
    let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let batched_program = builder
        .build(output_atom_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?
        .into_simplified()?;
    Ok((batched_program, output_axes))
}

/// Trace context that introduces exactly one batch dimension at a chosen axis.
///
/// [`BatchingContext`] is the active context for one level of `batch`: it runs the user's function
/// against logical per-item [`ArrayType`]s while leaving the runtime value type of the staged
/// program equal to the parent context's value type. Operations staged through this context are
/// lifted through their [`BatchableOperation`] rules at bind time. The lifted operation
/// is then staged into the parent context, so nested transforms compose by wrapping contexts
/// rather than by making each active transform pretend to be a backend domain.
///
/// Nested `batch` composes by repeated context wrapping:
/// `BatchingContext<BatchingContext<C>>` is a two-level batching trace, and the staged program's
/// value type remains `C::Value` regardless of the nesting depth. Each level owns its own
/// `axis_size` and optional `axis_name`, while primitive binds recursively pass through every
/// parent context in order.
#[derive(Debug)]
pub struct BatchingContext<C: Context<Type = ArrayType>> {
    /// Parent trace context wrapped by this batching level.
    parent_context: C,

    /// Size of the batch axis this level introduces.
    axis_size: usize,

    /// Optional human-readable name for this batched axis. Collectives such as `psum`, `pmean`, and
    /// `pmax` can address this axis by name from inside the batched function body.
    axis_name: Option<String>,
}

impl<C: StagingContext<Type = ArrayType>> BatchingContext<C> {
    /// Creates a new anonymous [`BatchingContext`] that wraps `parent_context` with the supplied batch size.
    #[inline]
    pub fn new(parent_context: C, axis_size: usize) -> Self {
        Self::with_axis_name(parent_context, axis_size, None)
    }

    /// Creates a new [`BatchingContext`] with an optionally named batched axis. Collectives such as `psum`,
    /// `pmean`, and `pmax` can address a named axis from inside the batched function body.
    #[inline]
    pub fn with_axis_name(parent_context: C, axis_size: usize, axis_name: Option<String>) -> Self {
        Self { parent_context, axis_size, axis_name }
    }
}

impl<C> BatchingContext<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: BatchableOperation<Tracer<C, C::Meta>, Self>,
{
    /// Creates a live [`BatchingTracer`] referring to `atom` in the parent builder, carrying the given logical
    /// (per-item) type and mapped batch axis at this batching level.
    ///
    /// This is the axis-carrying counterpart of [`StagingContext::tracer`]: callers that have already staged an atom
    /// at its physical type and know where its mapped batch axis sits use this to attach that axis to the flowing
    /// value as the head of its [`Meta`](StagingContext::Meta) stack. The tail (the parent context's per-level axes)
    /// is left replicated here, which is correct for a fresh program input that has no enclosing batched value;
    /// an enclosing nested-`batch` level instead prepends its axis onto the *incoming* value's existing stack
    /// directly (see [`BatchContext::batch`]).
    ///
    /// # Parameters
    ///
    ///   - `atom`: Staged atom in the parent builder.
    ///   - `logical_type`: Per-item (unbatched) type the value reports inside the batched body.
    ///   - `batch_axis`: Mapped batch axis carried by the value ([`BatchAxis::replicated`] when replicated).
    #[inline]
    pub fn batched_value(
        &self,
        atom: AtomId,
        logical_type: ArrayType,
        batch_axis: impl Into<BatchAxis>,
    ) -> BatchingTracer<C> {
        Tracer::new_with_meta(
            self.clone(),
            TracerState::Live(atom),
            logical_type,
            BatchingMeta::new(batch_axis.into(), C::Meta::default()),
        )
    }
}

impl<C: StagingContext<Type = ArrayType>> BatchingContext<C> {
    /// Interprets a captured flat program while staging batched primitive calls into the parent context.
    ///
    /// This only requires the program's own operation family `O` to be batchable; it deliberately does not require the
    /// enclosing context's [`Operation`](Domain::Operation) to be batchable, so higher-order batching rules can replay
    /// a captured sub-program through a [`BatchingContext`] whose [`Context`] impl is not yet in scope.
    pub(crate) fn interpret_program<O>(
        &self,
        program: &Program<ArrayType, C::Constant, O, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: Vec<ArrayBatch<C::Value>>,
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
    where
        O: BatchableOperation<Tracer<C, C::Meta>, Self>,
    {
        program.interpret_with(
            inputs,
            |_, constant| Ok(ArrayBatch::replicated(self.parent_context.constant(constant.clone()))),
            |instruction, instruction_inputs| instruction.operation().batch(self, instruction_inputs),
        )
    }
}

impl<C: Context<Type = ArrayType>> BatchingContext<C> {
    /// Returns the parent [`Context`] this batching context wraps. Batching rules use this to stage operations
    /// directly at the parent level — for example, [`forward_collective_to_parent`](
    /// crate::tracing_v2::operations::collective::forward_collective_to_parent) re-stages a collective that targets
    /// an outer named axis.
    #[inline]
    pub fn parent_context(&self) -> &C {
        &self.parent_context
    }

    /// Returns this batch level's named axis, if the enclosing `batch` call named one. Batching rules for
    /// collective-like operations match their own axis name against this to decide whether to consume the mapped
    /// batch axis at this level or forward the operation to [`BatchingContext::parent_context`].
    #[inline]
    pub fn axis_name(&self) -> Option<&str> {
        self.axis_name.as_deref()
    }

    /// Returns this batch level's batch size.
    #[inline]
    pub fn axis_size(&self) -> usize {
        self.axis_size
    }
}

impl<C: Context<Type = ArrayType>> Clone for BatchingContext<C> {
    fn clone(&self) -> Self {
        Self {
            parent_context: self.parent_context.clone(),
            axis_size: self.axis_size,
            axis_name: self.axis_name.clone(),
        }
    }
}

impl<C> Domain for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: BatchableOperation<Tracer<C, C::Meta>, Self>,
{
    type Type = ArrayType;
    type Value = BatchingTracer<C>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

/// A batching level binds the axis it introduces: a lookup for this level's [`axis_name`](Self::axis_name) resolves to
/// [`NamedAxis::Batched`] with this level's batch size, and any other name delegates to the parent context. Because
/// nested `batch` composes by context wrapping, the delegation chain naturally shadows outer bindings with inner ones.
impl<C> NamedAxes for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType> + NamedAxes,
    C::Operation: BatchableOperation<Tracer<C, C::Meta>, Self>,
{
    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        if self.axis_name.as_deref() == Some(name) {
            Some(NamedAxis::Batched { size: self.axis_size })
        } else {
            self.parent_context.named_axis(name)
        }
    }
}

impl<C> Context for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: BatchableOperation<Tracer<C, C::Meta>, Self>,
{
    /// Lifts a constant payload into this batching context by recording it as a replicated [`BatchingTracer`].
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<BatchingTracer<C>, ProgramError> {
        Ok(self.constant(constant))
    }

    /// Binding in a batching context routes through [`StagingContext::stage_operation`], which lifts the operation over
    /// each input's mapped batch axis through the operation's [`BatchableOperation`] rule.
    #[inline]
    fn bind<P: Into<Self::Operation>>(
        &self,
        operation: P,
        inputs: &[BatchingTracer<C>],
    ) -> Result<Vec<BatchingTracer<C>>, ProgramError> {
        let operation = operation.into();
        self.stage_operation(operation, inputs)
    }

    #[inline]
    fn resolve(&self, value: &BatchingTracer<C>) -> ValueResolution<C::Constant> {
        if !Rc::ptr_eq(self.builder(), value.context().builder()) {
            return ValueResolution::Opaque;
        }
        let Ok(atom_id) = value.atom_id() else {
            return ValueResolution::Opaque;
        };
        match self.builder().borrow().atoms().get(atom_id.index()).and_then(|atom| atom.as_constant()) {
            Some(constant) => ValueResolution::Concrete(constant.clone()),
            None => ValueResolution::Staged(atom_id),
        }
    }
}

impl<C> StagingContext for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: BatchableOperation<Tracer<C, C::Meta>, Self>,
{
    type Meta = BatchingMeta<C::Meta>;

    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Constant, Self::Operation>>> {
        self.parent_context.builder()
    }

    fn stage_operation<P: Into<Self::Operation>, I: std::borrow::Borrow<BatchingTracer<C>>>(
        &self,
        operation: P,
        inputs: &[I],
    ) -> Result<Vec<BatchingTracer<C>>, ProgramError> {
        let operation = operation.into();
        check_builders!(self.builder(), [inputs.iter().map(|input| input.borrow().context().builder())])
            .map_err(|error| self.error(error))?;
        if self.builder().borrow().error.is_some() {
            let input_types = inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect::<Vec<_>>();
            let output_types = operation.infer_output_types(input_types.as_slice())?;
            return Ok(output_types
                .into_iter()
                .map(|r#type| Tracer::new(self.clone(), TracerState::Poison, r#type))
                .collect());
        }

        // Build parent-level input batches. Each `ArrayBatch` wraps the same atom as a *parent* trace value at the
        // parent-physical (= this level's physical) type, with this level's mapped batch axis. This level's axis for
        // each input is the head of the value's `Meta` cons-stack (`input.meta().batch_axis()`), and the parent value
        // the rule dispatches through carries the *tail* of that stack (`input.meta().parent()`), which is exactly the
        // parent context's own per-level axes — so when the parent is itself a `BatchingContext` (nested `batch`), that
        // parent's `stage_operation` reads *its* axis straight off the value in hand, with no side table. The rule's
        // body (`operation.batch(...)`) then dispatches through the parent value's primitive impls, staging directly
        // into the parent context; multi-op staging (e.g., batch-varying `Condition` lowering to two branches + a
        // per-item `Select`) emerges automatically.
        let mut parent_input_batches: Vec<ArrayBatch<C::Value>> = Vec::with_capacity(inputs.len());
        for input in inputs {
            let input = input.borrow();
            let atom = match input.atom_id() {
                Ok(atom) => atom,
                Err(error) => return Err(self.error(error)),
            };
            let logical_type = input.r#type().into_owned();
            let axis = input.meta().batch_axis().axis();
            let parent_physical_type = match axis {
                Some(k) => logical_type.with_inserted_dimension(k, Size::Static(self.axis_size))?,
                None => logical_type,
            };
            let parent_value = Tracer::new_with_meta(
                self.parent_context.clone(),
                TracerState::Live(atom),
                parent_physical_type.clone(),
                input.meta().parent().clone(),
            );
            parent_input_batches.push(ArrayBatch::new(parent_physical_type, parent_value, axis)?);
        }
        let output_batches = operation.batch(self, parent_input_batches.as_slice())?;

        let mut output_values = Vec::with_capacity(output_batches.len());
        for output_batch in output_batches {
            let axis = output_batch.batch_axis().axis();
            let parent_value = output_batch.into_value();
            let parent_physical_type = parent_value.r#type().into_owned();
            let atom = parent_value.atom_id()?;
            let logical_type = match axis {
                Some(k) => parent_physical_type.without_dimension(k)?.0,
                None => parent_physical_type,
            };
            // The output value's `Meta` stack is this level's output axis (head) on top of the rule's parent output
            // value's stack (tail), so the outer levels' axes for this freshly staged atom are carried through.
            output_values.push(Tracer::new_with_meta(
                self.clone(),
                TracerState::Live(atom),
                logical_type,
                BatchingMeta::new(BatchAxis::from(axis), parent_value.meta().clone()),
            ));
        }
        Ok(output_values)
    }
}

impl<C> DifferentiationContext for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType> + DifferentiationContext + Domain<Type = ArrayType>,
    C: DifferentiationContext<Tangent = <C as Domain>::Value>,
    BatchingContext<C>:
        StagingContext<Type = ArrayType, Constant = <C as Domain>::Constant, Operation = <C as Domain>::Operation>,
{
    type Tangent = BatchingTracer<C>;

    #[inline]
    fn validate_primal(&self, primal: &Self::Value) -> Result<(), ProgramError> {
        check_builders!(self.builder(), primal.context().builder()).map_err(|error| self.error(error))
    }

    /// Differentiation through a batching context is only available when the parent context is itself a staging
    /// context whose tangent is its own staged value, so primal values are always tracers and concretizing
    /// extractions on them cannot succeed.
    #[inline]
    fn supports_primal_concretization(&self) -> bool {
        false
    }
}

/// Value flowing through a [`BatchingContext<C>`]: the unified [`Tracer`] specialized to carry a [`BatchAxis`] as its
/// metadata. The batch axis rides on the value itself, so the per-operation [`BatchableOperation`] rules route the
/// mapped batch axis through [`StagingContext::stage_operation`] from the value in hand. Its capability impls
/// (arithmetic, `Broadcast`, `Dot`, `Reduce`, `Select`, …) are the shared `Tracer<C, C::Meta>` impls, so batching needs
/// no bespoke value-level operation impls of its own.
///
/// The carried [`ArrayType`] is the *logical* (per-item, unbatched) type, matching what the staged value reports to
/// the batched function body; the physical type with the mapped axis inserted is reconstructed from the value's
/// [`BatchAxis`] when it is handed to a batching rule.
///
/// Per-level batching metadata carried by a [`BatchingTracer`]: a recursive cons-stack whose head is *this* batching
/// level's mapped [`BatchAxis`] and whose tail is the parent context's metadata.
///
/// For an outer plain trace the tail is `()`; for a nested `vmap` it is itself another [`BatchingMeta`], so the stack
/// grows by one axis per enclosing `batch`. Carrying the axis on the value this way is what lets every level of a
/// nested `batch` recover its own batch axis without any side table — `batch` over `batch` simply prepends one more
/// axis onto the incoming value's stack.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchingMeta<Meta> {
    /// Mapped batch axis introduced at this batching level ([`BatchAxis::replicated`] when this level does not map the
    /// value).
    batch_axis: BatchAxis,

    /// Parent context's metadata: the tail of the cons-stack (`()` for an outer plain trace, another [`BatchingMeta`]
    /// for a nested `vmap`).
    parent: Meta,
}

impl<Meta> BatchingMeta<Meta> {
    /// Creates a [`BatchingMeta`] pairing this level's `batch_axis` with the `parent` context's metadata tail.
    #[inline]
    pub fn new(batch_axis: BatchAxis, parent: Meta) -> Self {
        Self { batch_axis, parent }
    }

    /// Returns this batching level's mapped batch axis (the head of the cons-stack).
    #[inline]
    pub fn batch_axis(&self) -> BatchAxis {
        self.batch_axis
    }

    /// Returns the parent context's metadata (the tail of the cons-stack).
    #[inline]
    pub fn parent(&self) -> &Meta {
        &self.parent
    }
}

/// The [`Meta`](StagingContext::Meta) is a [`BatchingMeta`] cons-stack mirroring the context nesting: the head
/// [`BatchAxis`] is *this* batching level's mapped batch axis, and the tail `C::Meta` is the parent context's metadata
/// (itself another [`BatchingMeta`] stack for a nested `vmap`). This is what lets every level of a nested `batch`
/// recover its own batch axis for a value without any side table — `batch` over `batch` simply prepends one more axis
/// onto the incoming value's stack.
pub type BatchingTracer<C> = Tracer<BatchingContext<C>, BatchingMeta<<C as StagingContext>::Meta>>;

/// Batch-carrying batching value selected by an ordinary backend [`Domain`]. This is the [`BatchingTracer`] flowing
/// through the [`BatchingContext`] that wraps a fresh trace over `D`'s constant and operation families.
///
/// The parent trace is a plain [`TracingContext`] whose [`StagingContext::Meta`] is `()`, so the per-level cons-stack
/// bottoms out at `BatchingMeta<()>`. This is spelled concretely (rather than via the [`BatchingTracer`] alias) so the
/// `()` tail is written directly instead of as a `<TracingContext<…> as StagingContext>::Meta` projection, which the
/// trait solver does not always normalize to `()` when the same alias appears in several positions of a generic
/// signature.
pub type DomainBatchingValue<D> = Tracer<
    BatchingContext<TracingContext<ArrayType, <D as Domain>::Constant, <D as Domain>::Operation>>,
    BatchingMeta<()>,
>;

/// Extension trait that exposes [`Batch::batch`] as a method on any [`Domain`] whose `Type` is
/// [`ArrayType`].
///
/// `domain.batch(f, input, in_axes, out_axes, axis)` is the concrete-value batching entry point; it mirrors how
/// `jvp` sits on [`DifferentiationContext`]. Already-traced values use
/// the active context's [`BatchContext::batch`] path so nested transforms compose through context wrapping.
pub trait Batch: Domain<Type = ArrayType> {
    /// Maps a traced function over array axes selected per leaf by `in_axes` and places each output's
    /// mapped axis at the position requested by `out_axes`.
    ///
    /// `in_axes` is any [`Parameterized`] value over [`BatchAxis`] leaves; it is broadcast into the input's parameter
    /// structure via [`Parameterized::broadcast_to_parameter_structure`], so a single [`BatchAxis`] maps every leaf
    /// the same way (the typed counterpart of JAX's `in_axes=0`), a value whose structure matches the input gives a
    /// per-leaf axis, and a smaller compatible structure prefix-broadcasts to fill the input. Each leaf is either
    /// [`BatchAxis::new(k)`](BatchAxis::new) (the input is mapped on axis `k` of its physical type) or
    /// [`BatchAxis::replicated`] (the input is replicated / broadcast across the batch). When at least one input is
    /// mapped, the batch size is inferred from those inputs; the `axis` parameter accepts anything convertible to a
    /// [`BatchAxisSpecification`] and can supply an explicit batch size to either pin the inferred size or drive a
    /// fully-broadcast `batch` whose batch size would otherwise be unobservable, as well as an axis name that
    /// collectives (`psum`, `pmean`, `pmax`) inside the batched function can address. The `out_axes` value (broadcast
    /// the same way over the output structure) selects where the mapped axis lands in each output:
    /// [`BatchAxis::new(k)`](BatchAxis::new) requests position `k` (an explicit transpose is staged when the
    /// natural output axis differs), and [`BatchAxis::replicated`] declares the corresponding output to be replicated
    /// (e.g., a value produced from broadcast inputs without staging any per-item work).
    ///
    /// This is the concrete-value entry point. Already-traced values use [`BatchContext::batch`] on
    /// their active context.
    fn batch<F, I, O, InputBatchAxes, OutputBatchAxes>(
        &self,
        function: F,
        input: I,
        in_axes: InputBatchAxes,
        out_axes: OutputBatchAxes,
        axis: impl Into<BatchAxisSpecification>,
    ) -> Result<O::To<Self::Value>, BatchingError>
    where
        Self: Context,
        InputBatchAxes: Parameterized<BatchAxis>,
        OutputBatchAxes: Parameterized<BatchAxis>,
        I: Parameterized<
                Self::Value,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<BatchAxis>
                            + ParameterizedFamily<DomainTracer<Self>>
                            + ParameterizedFamily<DomainBatchingValue<Self>>,
            >,
        O: Parameterized<
                DomainBatchingValue<Self>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Value>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<BatchAxis>
                            + ParameterizedFamily<DomainTracer<Self>>
                            + ParameterizedFamily<DomainBatchingValue<Self>>,
            >,
        I::To<BatchAxis>: Parameterized<BatchAxis, ParameterStructure = I::ParameterStructure>,
        O::To<BatchAxis>: Parameterized<BatchAxis, ParameterStructure = O::ParameterStructure>,
        I::To<DomainTracer<Self>>: Parameterized<
                DomainTracer<Self>,
                ParameterStructure = I::ParameterStructure,
                To<ArrayType> = I::To<ArrayType>,
                To<Self::Value> = I,
                To<Self::Constant> = I::To<Self::Constant>,
                To<BatchAxis> = I::To<BatchAxis>,
                To<DomainBatchingValue<Self>> = I::To<DomainBatchingValue<Self>>,
            >,
        O::To<DomainTracer<Self>>: Parameterized<
                DomainTracer<Self>,
                ParameterStructure = O::ParameterStructure,
                To<ArrayType> = O::To<ArrayType>,
                To<Self::Value> = O::To<Self::Value>,
                To<Self::Constant> = O::To<Self::Constant>,
                To<BatchAxis> = O::To<BatchAxis>,
                To<DomainBatchingValue<Self>> = O,
            >,
        I::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Value> = I,
                To<Self::Constant> = I::To<Self::Constant>,
                To<DomainTracer<Self>> = I::To<DomainTracer<Self>>,
                To<DomainBatchingValue<Self>> = I::To<DomainBatchingValue<Self>>,
            >,
        O::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Value> = O::To<Self::Value>,
                To<Self::Constant> = O::To<Self::Constant>,
                To<DomainTracer<Self>> = O::To<DomainTracer<Self>>,
                To<DomainBatchingValue<Self>> = O,
            >,
        Self::Operation: Clone
            + From<TransposeOperation>
            + BatchableOperation<DomainTracer<Self>, BatchingContext<DomainTracingContext<Self>>>,
        F: FnOnce(I::To<DomainBatchingValue<Self>>) -> Result<O, ProgramError>,
    {
        let structure = input.parameter_structure();
        let input_values = input.into_parameters().collect::<Vec<_>>();
        let parent_context: DomainTracingContext<Self> = TracingContext::new();
        let builder = parent_context.builder().clone();
        let mut input_tracers = Vec::with_capacity(input_values.len());
        for value in input_values.iter() {
            let physical_type = value.r#type().into_owned();
            let atom = builder.borrow_mut().add_input(physical_type.clone());
            input_tracers.push(parent_context.tracer(atom, Some(physical_type)));
        }
        let traced_input = I::To::<DomainTracer<Self>>::from_parameters(structure.clone(), input_tracers)?;
        // Batching rules ride up the `ProgramError`-typed staging kernel as `ProgramError::Custom` payloads; the
        // `From<ProgramError>` conversions behind the `?` operators below re-type them so the public `batch` surfaces
        // a transform-owned `BatchingError`, mirroring how `value_and_grad` surfaces a `DifferentiationError`.
        let traced_output: O::To<DomainTracer<Self>> =
            BatchContext::batch(&parent_context, function, traced_input, in_axes, out_axes, axis)?;
        if let Some(error) = builder.borrow_mut().error.take() {
            return Err(error.into());
        }
        let output_structure = traced_output.parameter_structure();
        let output_atom_ids = traced_output.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
        drop(traced_output);
        drop(parent_context);

        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program: Program<ArrayType, Self::Constant, Self::Operation, I::To<Self::Constant>, O::To<Self::Constant>> =
            builder.build(output_atom_ids, structure, output_structure.clone())?;
        // The replay folds through `self` directly: an eager domain interprets each instruction immediately,
        // while a staging context stages it into its enclosing trace.
        let output_values = replay_via_bind(self, &program, input_values)?;
        Ok(O::To::<Self::Value>::from_parameters(output_structure, output_values)?)
    }
}

impl<D: Domain<Type = ArrayType>> Batch for D {}

/// Extension trait that exposes batching as a method on active array contexts.
///
/// This is the already-traced counterpart of [`Batch`]. It wraps the receiver in a [`BatchingContext`] and routes all
/// primitive binds through the current transform stack, so `batch` composes with tracing, JVP, VJP, and other context
/// wrappers through the same [`StagingContext::stage_operation`] path.
///
/// The receiver flows its own [`Tracer`] metadata (its [`StagingContext::Meta`]), so this trait is implemented for
/// plain tracing contexts ([`Meta`](StagingContext::Meta)` = ()`), for batching contexts
/// ([`Meta`](StagingContext::Meta)` = `[`BatchingMeta`], which is what makes nested `vmap` work), and for any other
/// staging context.
pub trait BatchContext: StagingContext<Type = ArrayType> {
    /// Maps a traced function over per-leaf array axes inside this active context. The `in_axes` and `out_axes`
    /// parameters are [`Parameterized`] values over [`BatchAxis`] leaves that are broadcast into the input and output
    /// parameter structures (see [`Batch::batch`] for the broadcasting semantics), and the `axis` parameter accepts
    /// anything convertible to a [`BatchAxisSpecification`] (an optional explicit batch size and an optional axis
    /// name).
    fn batch<F, I, O, InputBatchAxes, OutputBatchAxes>(
        &self,
        function: F,
        input: I,
        in_axes: InputBatchAxes,
        out_axes: OutputBatchAxes,
        axis: impl Into<BatchAxisSpecification>,
    ) -> Result<O::To<Tracer<Self, Self::Meta>>, ProgramError>
    where
        InputBatchAxes: Parameterized<BatchAxis>,
        OutputBatchAxes: Parameterized<BatchAxis>,
        Self::Operation:
            Clone + From<TransposeOperation> + BatchableOperation<Tracer<Self, Self::Meta>, BatchingContext<Self>>,
        I: Parameterized<
                Tracer<Self, Self::Meta>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<BatchAxis>
                            + ParameterizedFamily<Tracer<Self, Self::Meta>>
                            + ParameterizedFamily<BatchingTracer<Self>>,
            >,
        O: Parameterized<
                BatchingTracer<Self>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<BatchAxis>
                            + ParameterizedFamily<Tracer<Self, Self::Meta>>
                            + ParameterizedFamily<BatchingTracer<Self>>,
            >,
        I::To<BatchAxis>: Parameterized<BatchAxis, ParameterStructure = I::ParameterStructure>,
        O::To<BatchAxis>: Parameterized<BatchAxis, ParameterStructure = O::ParameterStructure>,
        I::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Constant> = I::To<Self::Constant>,
                To<BatchingTracer<Self>> = I::To<BatchingTracer<Self>>,
            >,
        O::To<ArrayType>: Parameterized<
                ArrayType,
                To<Tracer<Self, Self::Meta>> = O::To<Tracer<Self, Self::Meta>>,
                To<BatchingTracer<Self>> = O,
            >,
        F: FnOnce(I::To<BatchingTracer<Self>>) -> Result<O, ProgramError>,
    {
        let axis = axis.into();
        let parent_context = self.clone();
        let input_structure = input.parameter_structure();
        let input_tracers = input.into_parameters().collect::<Vec<_>>();
        // Broadcast the caller's `in_axes` into the input parameter structure: a single `BatchAxis` leaf fills every
        // input leaf (JAX's `in_axes=0`), a matching structure gives one axis per leaf, and a smaller compatible
        // structure prefix-broadcasts. A structure that cannot fill the input surfaces as a `ParameterError`.
        let in_axes_values = in_axes
            .broadcast_to_parameter_structure::<I::To<BatchAxis>>(input_structure.clone())?
            .into_parameters()
            .collect::<Vec<_>>();
        if input_tracers.is_empty() && axis.size().is_none() {
            return Err(BatchingError::EmptyBatch.into());
        }

        let mut resolved_axis_size = axis.size();
        let mut inputs_with_axes = Vec::with_capacity(input_tracers.len());
        for (tracer, axis) in input_tracers.into_iter().zip(in_axes_values.iter().copied()) {
            let parent_physical_type = tracer.r#type().into_owned();
            match axis.axis() {
                Some(batch_axis) => {
                    let (per_item_type, dimension) = parent_physical_type.without_dimension(batch_axis)?;
                    let Some(size) = dimension.value() else {
                        return Err(
                            BatchingError::DynamicBatchAxis { r#type: parent_physical_type, axis: batch_axis }.into()
                        );
                    };
                    match resolved_axis_size {
                        Some(existing_size) if existing_size != size => {
                            return Err(
                                BatchingError::MismatchedBatchSizes { expected: existing_size, actual: size }.into()
                            );
                        }
                        Some(_) => {}
                        None => resolved_axis_size = Some(size),
                    }
                    inputs_with_axes.push((tracer, BatchAxis::new(batch_axis), per_item_type));
                }
                None => {
                    inputs_with_axes.push((tracer, BatchAxis::replicated(), parent_physical_type));
                }
            }
        }
        let resolved_axis_size = resolved_axis_size.ok_or(BatchingError::EmptyBatch)?;

        let batching_context =
            BatchingContext::with_axis_name(parent_context.clone(), resolved_axis_size, axis.name().map(String::from));
        let parent_builder = parent_context.builder().clone();

        let mut batched_input_values = Vec::with_capacity(inputs_with_axes.len());
        for (parent_tracer, axis, logical_type) in inputs_with_axes.iter() {
            let atom = parent_tracer.atom_id()?;
            // Prepend this level's mapped axis onto the *incoming* value's existing `Meta` stack: the head is this
            // level's axis and the tail is the outer value's stack, so a value already mapped by an enclosing `batch`
            // keeps every outer level's axis. A fresh program input simply rides the parent context's default stack.
            batched_input_values.push(Tracer::new_with_meta(
                batching_context.clone(),
                TracerState::Live(atom),
                logical_type.clone(),
                BatchingMeta::new(*axis, parent_tracer.meta().clone()),
            ));
        }
        let batched_input = I::To::<BatchingTracer<Self>>::from_parameters(input_structure, batched_input_values)?;
        let batched_output =
            function(batched_input).map_err(|error| parent_builder.borrow_mut().error.take().unwrap_or(error))?;
        parent_builder.borrow_mut().error.take().map_or(Ok(()), Err)?;

        let output_structure = batched_output.parameter_structure();
        // Each output's mapped batch axis at this level is the head of its `Meta` stack (`value.meta().batch_axis()`),
        // and the tail (`value.meta().parent()`) is the parent context's per-level axes for that staged atom. Carrying
        // that tail into the re-wrapped parent value is what threads an enclosing `batch`'s axis through this one for
        // nested `vmap`: the outer driver then reads its own axis straight off this value's stack head with no side
        // table.
        let outputs = batched_output
            .parameters()
            .map(|value| Ok((value.atom_id()?, value.meta().batch_axis().axis(), value.meta().parent().clone())))
            .collect::<Result<Vec<_>, ProgramError>>()?;
        // Broadcast the caller's `out_axes` into the output parameter structure, mirroring the `in_axes` handling
        // above: a single `BatchAxis` leaf applies to every output, a matching structure gives one axis per leaf.
        let out_axes_values = out_axes
            .broadcast_to_parameter_structure::<O::To<BatchAxis>>(output_structure.clone())?
            .into_parameters()
            .collect::<Vec<_>>();
        drop(batched_output);
        drop(batching_context);

        let parent_outputs = outputs
            .into_iter()
            .zip(out_axes_values.iter().map(|axis| axis.axis()))
            .map(
                |((atom, current_axis, parent_meta), expected_axis)| -> Result<Tracer<Self, Self::Meta>, ProgramError> {
                    let physical_type = parent_context.builder().borrow().atoms()[atom.index()].r#type().into_owned();
                    let parent_tracer = Tracer::new_with_meta(
                        parent_context.clone(),
                        TracerState::Live(atom),
                        physical_type,
                        parent_meta,
                    );
                    match (current_axis, expected_axis) {
                        (None, None) => Ok(parent_tracer),
                        // The output's mapped-axis presence disagrees with the caller's `out_axes` declaration.
                        // Collapsing a mapped output requires an explicit reduction inside the batched function, and
                        // materializing a missing axis requires an explicit broadcast; position-only disagreements are
                        // instead repaired with the staged transpose in the arm below.
                        (None, Some(_)) | (Some(_), None) => {
                            Err(BatchingError::MismatchedOutputAxes { expected: expected_axis, actual: current_axis }
                                .into())
                        }
                        (Some(current), Some(expected)) if current == expected => Ok(parent_tracer),
                        (Some(current), Some(expected)) => {
                            let rank = parent_tracer.r#type().as_ref().rank();
                            let permutation = move_axis_permutation(rank, current, expected);
                            parent_tracer.transpose(permutation)
                        }
                    }
                },
            )
            .collect::<Result<Vec<_>, ProgramError>>()?;

        Ok(O::To::<Tracer<Self, Self::Meta>>::from_parameters(output_structure, parent_outputs)?)
    }
}

impl<C> BatchContext for C where C: StagingContext<Type = ArrayType> {}

/// Returns the axis permutation that moves dimension `from` to position `to`, shifting the other
/// dimensions to preserve their relative order. Returns the identity permutation when
/// `from == to`.
pub(crate) fn move_axis_permutation(rank: usize, from: usize, to: usize) -> Vec<usize> {
    let mut permutation: Vec<usize> = (0..rank).collect();
    let axis = permutation.remove(from);
    permutation.insert(to, axis);
    permutation
}
