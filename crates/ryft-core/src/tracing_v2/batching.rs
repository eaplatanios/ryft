use std::fmt::Debug;
use std::rc::Rc;

use crate::batching::{
    ArrayBatch, BatchAxis, BatchAxisSpecification, BatchableOperation, BatchingContext, BatchingError, BatchingMeta,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::macros::{check_builders, check_count};
use crate::operations::Operation;
use crate::operations::manipulation::{BroadcastOperation, Transpose, TransposeOperation};
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::{AtomId, Program, ProgramError, Value};
use crate::tracing::{DomainTracer, DomainTracingContext, Tracer, TracerState, TracingContext};
use crate::tracing_v2::differentiation::{DifferentiationContext, replay_via_bind};
use crate::types::{ArrayType, Size, Typed};

// TODO(eaplatanios): Review this module.

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
/// [`ProgramBatchingOutputAxesPolicy::Natural`] keeps the mapped axes produced by the batching rules (the discovery
/// pass of staged control-flow batching). [`ProgramBatchingOutputAxesPolicy::AlignEachTo`] instantiates each output
/// at a requested axis while the outputs are still live tracers, which is how staged `condition` branches agree on
/// one output layout and the staged `while` fixpoint keeps body outputs on the loop-invariant state axes.
/// [`ProgramBatchingOutputAxesPolicy::AlignAllTo`] imposes one canonical output axis, which is what custom-derivative
/// re-wrapping needs so independently batched primal/JVP/forward/backward programs have mutually consistent
/// signatures.
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
            input_values.push(Tracer::batched(batching_context.clone(), atom, logical_type.clone(), *axis));
        }
        let output_values = batching_context.stage_program(program, input_values)?;
        // Resolve the policy into one alignment target per output: `None` keeps the natural axis, and a mapped target
        // forces the output to carry its batch axis at that position (a replicated `AlignEachTo` entry is a lower
        // bound, not an equality constraint, mirroring JAX's `instantiate`).
        let output_targets: Vec<Option<usize>> = match &output_axes_policy {
            ProgramBatchingOutputAxesPolicy::Natural => vec![None; output_values.len()],
            ProgramBatchingOutputAxesPolicy::AlignAllTo(target_axis) => vec![Some(*target_axis); output_values.len()],
            ProgramBatchingOutputAxesPolicy::AlignEachTo(targets) => {
                check_count!("output", output_values, targets.len(), ProgramError);
                targets.iter().map(|target| target.axis()).collect()
            }
        };
        let mut output_atom_ids = Vec::with_capacity(output_values.len());
        let mut output_axes = Vec::with_capacity(output_values.len());
        for (output_value, target) in output_values.into_iter().zip(output_targets) {
            let atom = output_value.atom_id()?;
            let natural_axis = output_value.meta().batch_axis();
            // Untargeted (or already-aligned) output: keep the axis the batching rules produced.
            let Some(target_axis) = target else {
                output_axes.push(natural_axis);
                output_atom_ids.push(atom);
                continue;
            };
            if natural_axis.axis() == Some(target_axis) {
                output_axes.push(natural_axis);
                output_atom_ids.push(atom);
                continue;
            }
            // Move a mapped output to the target axis, or broadcast a replicated output across the batch, staging the
            // axis-adjusting operation into the batched program while its outputs are still live tracers.
            let logical_type = output_value.r#type().into_owned();
            let physical_type = match natural_axis.axis() {
                Some(axis) => logical_type.with_inserted_dimension(axis, Size::Static(axis_size))?,
                None => logical_type,
            };
            let parent_batch = ArrayBatch::new(
                physical_type.clone(),
                batching_context.parent_context().tracer(atom, Some(physical_type)),
                natural_axis,
            )?;
            let aligned_batch = match natural_axis.axis() {
                Some(_) => parent_batch.move_axis(target_axis)?,
                None => parent_batch.broadcast(target_axis, axis_size)?,
            };
            output_atom_ids.push(aligned_batch.into_value().atom_id()?);
            output_axes.push(BatchAxis::new(target_axis));
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

impl<C> Tracer<BatchingContext<C>, BatchingMeta<C::Meta>>
where
    C: StagingContext<Type = ArrayType, Operation: BatchableOperation<Tracer<C, C::Meta>, BatchingContext<C>>>,
{
    /// Creates a live [`BatchingTracer`] for `atom_id` in `context`'s parent builder, carrying the given logical
    /// (per-item) `r#type` and mapped `batch_axis` at this batching level. This is the axis-carrying counterpart of
    /// [`Tracer::new`]/[`Tracer::new_with_meta`]: callers that have already staged an atom at its physical type and
    /// know where its mapped batch axis sits use this to attach that axis to the flowing value as the head of its
    /// [`Meta`](StagingContext::Meta) stack. The tail (the parent context's per-level axes) is left replicated here,
    /// which is correct for a fresh program input that has no enclosing batched value; an enclosing nested-`batch`
    /// level instead prepends its axis onto the *incoming* value's existing stack directly (see
    /// [`BatchContext::batch`]).
    ///
    /// # Parameters
    ///
    ///   - `context`: [`BatchingContext`] level this value flows through.
    ///   - `atom_id`: Staged atom in the parent builder.
    ///   - `r#type`: Per-item (unbatched) type the value reports inside the batched body.
    ///   - `batch_axis`: Mapped batch axis carried by the value ([`BatchAxis::replicated`] when replicated).
    #[inline]
    pub fn batched<A: Into<BatchAxis>>(
        context: BatchingContext<C>,
        atom_id: AtomId,
        r#type: ArrayType,
        batch_axis: A,
    ) -> Self {
        Self::new_with_meta(
            context,
            TracerState::Live(atom_id),
            r#type,
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
            |_, constant| Ok(ArrayBatch::replicated(self.parent_context().constant(constant.clone()))),
            |instruction, instruction_inputs| instruction.operation().batch(self, instruction_inputs),
        )
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
                        (None, Some(_)) | (Some(_), None) => Err(BatchingError::MismatchedOutputAxes {
                            expected: BatchAxis::from(expected_axis),
                            actual: BatchAxis::from(current_axis),
                        }
                        .into()),
                        (Some(current), Some(expected)) if current == expected => Ok(parent_tracer),
                        (Some(current), Some(expected)) => parent_tracer.move_axis(current, expected),
                    }
                },
            )
            .collect::<Result<Vec<_>, ProgramError>>()?;

        Ok(O::To::<Tracer<Self, Self::Meta>>::from_parameters(output_structure, parent_outputs)?)
    }
}

impl<C> BatchContext for C where C: StagingContext<Type = ArrayType> {}
