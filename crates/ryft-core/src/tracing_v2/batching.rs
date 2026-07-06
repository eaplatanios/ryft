use std::fmt::Debug;
use std::rc::Rc;

use crate::batching::{
    ArrayBatch, BatchAxis, BatchAxisSpecification, BatchableOperation, BatchingContext, BatchingError, BatchingTracer,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::control_flow::SelectOperation;
use crate::operations::logical::AndOperation;
use crate::operations::manipulation::{
    BroadcastOperation, ReshapeOperation, SliceOperation, Transpose, TransposeOperation, UpdateSliceOperation,
};
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::{Program, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::differentiation::DifferentiationContext;
use crate::tracing_v2::operations::reduce::ReduceOperation;
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
    program: &Program<V, O, Vec<V>, Vec<V>>,
    axis_size: usize,
    input_batch_axes: &[BatchAxis],
    output_axes_policy: ProgramBatchingOutputAxesPolicy,
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>), BatchingError>
where
    V: Value<Type = ArrayType> + 'static,
    O: Clone + Operation<ArrayType> + 'static,
    O: BatchableOperation<Tracer<TracingContext<V, O, V>>, BatchingContext<TracingContext<V, O, V>>>,
    O: From<TransposeOperation>
        + From<BroadcastOperation>
        + From<SliceOperation>
        + From<UpdateSliceOperation>
        + From<ReshapeOperation>
        + From<SelectOperation>
        + From<AndOperation>
        + From<ReduceOperation>,
{
    let logical_input_types = program.input_types();
    let input_count = logical_input_types.len();
    check_count!("input", input_batch_axes, input_count, ProgramError);
    let parent: TracingContext<V, O, V> = TracingContext::new();
    let builder = parent.builder().clone();
    // Keep every tracer and context that holds a clone of `builder` inside this scope so that recovering the
    // builder below is a real ownership check.
    let (output_atom_ids, output_axes) = {
        let batching_context = BatchingContext::new(parent, axis_size, None);
        let mut input_values = Vec::with_capacity(input_count);
        for (logical_type, axis) in logical_input_types.iter().zip(input_batch_axes.iter()) {
            let physical_type = match axis.axis() {
                Some(position) => logical_type.with_inserted_dimension(position, Size::Static(axis_size))?,
                None => logical_type.clone(),
            };
            let atom = builder.borrow_mut().add_input(physical_type.clone());
            let parent_value = batching_context.parent().tracer(atom, Some(physical_type.clone()));
            input_values.push(ArrayBatch::new(physical_type, parent_value, *axis)?);
        }
        let output_batches = batching_context.interpret_program(program, input_values)?;
        // Resolve the policy into one alignment target per output: `None` keeps the natural axis, and a mapped target
        // forces the output to carry its batch axis at that position (a replicated `AlignEachTo` entry is a lower
        // bound, not an equality constraint, mirroring JAX's `instantiate`).
        let output_targets: Vec<Option<usize>> = match &output_axes_policy {
            ProgramBatchingOutputAxesPolicy::Natural => vec![None; output_batches.len()],
            ProgramBatchingOutputAxesPolicy::AlignAllTo(target_axis) => vec![Some(*target_axis); output_batches.len()],
            ProgramBatchingOutputAxesPolicy::AlignEachTo(targets) => {
                check_count!("output", output_batches, targets.len(), ProgramError);
                targets.iter().map(|target| target.axis()).collect()
            }
        };
        let mut output_atom_ids = Vec::with_capacity(output_batches.len());
        let mut output_axes = Vec::with_capacity(output_batches.len());
        for (output_batch, target) in output_batches.into_iter().zip(output_targets) {
            let natural_axis = output_batch.batch_axis();
            // Untargeted (or already-aligned) output: keep the axis the batching rules produced.
            let Some(target_axis) = target else {
                output_atom_ids.push(output_batch.value().atom_id()?);
                output_axes.push(natural_axis);
                continue;
            };
            if natural_axis.axis() == Some(target_axis) {
                output_atom_ids.push(output_batch.value().atom_id()?);
                output_axes.push(natural_axis);
                continue;
            }
            // Move a mapped output to the target axis, or broadcast a replicated output across the batch, staging the
            // axis-adjusting operation into the batched program while its outputs are still live tracers.
            let aligned_batch = match natural_axis.axis() {
                Some(_) => output_batch.move_axis(target_axis)?,
                None => output_batch.broadcast(target_axis, axis_size)?,
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

impl<C: Context<Type = ArrayType>> BatchingContext<C> {
    /// Replays a captured flat program by binding each instruction's [`BatchableOperation`] rule against this batching
    /// context, threading the batch-carrying inputs through. Constants are lifted in the parent context and replicated
    /// across the batch. Higher-order batching rules use this to batch a captured sub-program without concretizing any
    /// batch-item values, so batched control-flow structure composes into the enclosing computation (executing under an
    /// eager parent, staging under a live trace).
    ///
    /// This only requires the program's own operation family `O` to be batchable; it deliberately does not require the
    /// enclosing context's [`Operation`](Domain::Operation) to be batchable, so higher-order batching rules can replay
    /// a captured sub-program through a [`BatchingContext`] whose [`Context`] impl is not yet in scope.
    pub(crate) fn interpret_program<O>(
        &self,
        program: &Program<C::Constant, O, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: Vec<ArrayBatch<C::Value>>,
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
    where
        O: BatchableOperation<C::Value, Self>,
    {
        program.interpret_with(
            inputs,
            |_, constant| Ok(ArrayBatch::replicated(self.parent().lift(constant.clone())?)),
            |instruction, instruction_inputs| instruction.operation().batch(self, instruction_inputs),
        )
    }
}

impl<C> DifferentiationContext for BatchingContext<C>
where
    C: Context<Type = ArrayType> + DifferentiationContext<Tangent = <C as Domain>::Value>,
    C::Operation: BatchableOperation<<C as Domain>::Value, Self>,
{
    type Tangent = BatchingTracer<C>;

    /// A batched primal is valid exactly when the parent context accepts the value it packs.
    #[inline]
    fn validate_primal(&self, primal: &Self::Value) -> Result<(), ProgramError> {
        self.parent().validate_primal(primal.batch().value())
    }

    /// Concretizing a batched primal is possible exactly when the parent context supports concretizing the value it
    /// packs (never under a staging parent, always under an eager one).
    #[inline]
    fn supports_primal_concretization(&self) -> bool {
        self.parent().supports_primal_concretization()
    }
}

/// Extension trait that exposes batching (`vmap`) as a method on any array [`Context`] — the single `batch` entry
/// point, mirroring how `jvp` sits on [`DifferentiationContext`].
///
/// `context.batch(function, input, in_axes, out_axes, axis)` wraps the receiver in a [`BatchingContext`] and runs
/// `function` on concrete [`BatchingTracer`] values, so every primitive bind inside the closure flows through the
/// receiver's own context. This composes uniformly across the whole stack: an eager backend [`Context`] interprets
/// each batched bind immediately (concrete-value `vmap`), an active [`StagingContext`] stages it into the enclosing
/// trace, and a [`BatchingContext`] receiver nests `vmap` inside `vmap` — each level's [`BatchingTracer`] carries its
/// own batch axis, so nested maps thread through with no side table.
pub trait BatchContext: Context<Type = ArrayType> {
    /// Maps `function` over array axes selected per leaf by `in_axes` and places each output's mapped axis at the
    /// position requested by `out_axes`.
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
    /// [`BatchAxis::new(k)`](BatchAxis::new) requests position `k` (an explicit transpose is staged when the natural
    /// output axis differs), and [`BatchAxis::replicated`] declares the corresponding output to be replicated (e.g., a
    /// value produced from broadcast inputs without staging any per-item work).
    fn batch<F, I, O, InputBatchAxes, OutputBatchAxes>(
        &self,
        function: F,
        input: I,
        in_axes: InputBatchAxes,
        out_axes: OutputBatchAxes,
        axis: impl Into<BatchAxisSpecification>,
    ) -> Result<O::To<Self::Value>, BatchingError>
    where
        InputBatchAxes: Parameterized<BatchAxis>,
        OutputBatchAxes: Parameterized<BatchAxis>,
        Self::Value: Transpose,
        Self::Operation: Clone + From<TransposeOperation> + BatchableOperation<Self::Value, BatchingContext<Self>>,
        I: Parameterized<
                Self::Value,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<BatchAxis>
                            + ParameterizedFamily<Self::Value>
                            + ParameterizedFamily<BatchingTracer<Self>>,
            >,
        O: Parameterized<
                BatchingTracer<Self>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<BatchAxis>
                            + ParameterizedFamily<Self::Value>
                            + ParameterizedFamily<BatchingTracer<Self>>,
            >,
        I::To<BatchAxis>: Parameterized<BatchAxis, ParameterStructure = I::ParameterStructure>,
        O::To<BatchAxis>: Parameterized<BatchAxis, ParameterStructure = O::ParameterStructure>,
        I::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Constant> = I::To<Self::Constant>,
                To<BatchingTracer<Self>> = I::To<BatchingTracer<Self>>,
            >,
        O::To<ArrayType>: Parameterized<ArrayType, To<Self::Value> = O::To<Self::Value>, To<BatchingTracer<Self>> = O>,
        F: FnOnce(I::To<BatchingTracer<Self>>) -> Result<O, ProgramError>,
    {
        let axis = axis.into();
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
            let physical_type = tracer.r#type().into_owned();
            match axis.axis() {
                Some(batch_axis) => {
                    let Some(size) = physical_type.dimension(batch_axis as isize).value() else {
                        return Err(BatchingError::DynamicBatchAxis { r#type: physical_type, axis: batch_axis }.into());
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
                    inputs_with_axes.push((tracer, physical_type, BatchAxis::new(batch_axis)));
                }
                None => {
                    inputs_with_axes.push((tracer, physical_type, BatchAxis::replicated()));
                }
            }
        }
        let resolved_axis_size = resolved_axis_size.ok_or(BatchingError::EmptyBatch)?;

        let batching_context = BatchingContext::new(self.clone(), resolved_axis_size, axis.name().map(String::from));

        // Pack each input parent value with its mapped batch axis at its physical type. A value already produced by an
        // enclosing `batch` keeps that level's axis (its own [`BatchingTracer`] carries it), so nested maps thread
        // through with no side table; a fresh input simply flows the receiver's own value representation.
        let mut batched_input_values = Vec::with_capacity(inputs_with_axes.len());
        for (parent_value, physical_type, batch_axis) in inputs_with_axes {
            let batch = ArrayBatch::new(physical_type, parent_value, batch_axis)?;
            batched_input_values.push(BatchingTracer::new(batching_context.clone(), batch));
        }
        let batched_input = I::To::<BatchingTracer<Self>>::from_parameters(input_structure, batched_input_values)?;
        // Binds inside the closure fold through the receiver directly: an eager context interprets each immediately,
        // while a staging context stages it into the enclosing trace, whose own drain surfaces any deferred error.
        let batched_output = function(batched_input)?;

        let output_structure = batched_output.parameter_structure();
        // Broadcast the caller's `out_axes` into the output parameter structure, mirroring the `in_axes` handling
        // above: a single `BatchAxis` leaf applies to every output, a matching structure gives one axis per leaf.
        let out_axes_values = out_axes
            .broadcast_to_parameter_structure::<O::To<BatchAxis>>(output_structure.clone())?
            .into_parameters()
            .collect::<Vec<_>>();

        // Realign each output's packed batch axis to the caller's `out_axes` and unwrap the parent tracer, which
        // already carries any enclosing level's metadata, so nested `vmap` threads through with no side table.
        let parent_outputs = batched_output
            .into_parameters()
            .zip(out_axes_values.iter().map(|axis| axis.axis()))
            .map(|(output, expected_axis)| -> Result<Self::Value, ProgramError> {
                let batch = output.into_batch();
                let natural_axis = batch.batch_axis().axis();
                match (natural_axis, expected_axis) {
                    (None, None) => Ok(batch.into_value()),
                    // The output's mapped-axis presence disagrees with the caller's `out_axes` declaration. Collapsing
                    // a mapped output requires an explicit reduction inside the batched function, and materializing a
                    // missing axis requires an explicit broadcast; position-only disagreements are repaired with the
                    // staged transpose in the final arm.
                    (None, Some(_)) | (Some(_), None) => Err(BatchingError::MismatchedOutputAxes {
                        expected: BatchAxis::from(expected_axis),
                        actual: BatchAxis::from(natural_axis),
                    }
                    .into()),
                    (Some(current), Some(expected)) if current == expected => Ok(batch.into_value()),
                    (Some(_), Some(expected)) => Ok(batch.move_axis(expected)?.into_value()),
                }
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;

        Ok(O::To::<Self::Value>::from_parameters(output_structure, parent_outputs)?)
    }
}

impl<C> BatchContext for C where C: Context<Type = ArrayType> {}
