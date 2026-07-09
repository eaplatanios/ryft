use std::fmt::Display;

use crate::batching::ArrayBatch;
use crate::batching::BatchableOperation;
use crate::batching::BatchingContext;
use crate::batching::BatchingError;
use crate::contexts::{Context, Domain, EagerContext, StagingContext};
use crate::differentiation::DifferentiationDual;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableProgramOperation, DifferentiableType, TransposableOperation,
    TransposableProgramOperation,
};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::constants::{Zero, ZeroOperation};
use crate::operations::control_flow::ScanOperation;
use crate::operations::control_flow::scan::{ScanTypeSemantics, stacked_scan_type};
use crate::operations::manipulation::{
    Reshape, ReshapeOperation, Slice, SliceOperation, UpdateSlice, UpdateSliceOperation,
};
use crate::parameters::Placeholder;
use crate::partial::PartialValue;
use crate::programs::{Atom, AtomId, Instruction, MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, DataType, Shape, Size, Type, Typed};

/// Renders a compact comma-separated list of capture-like payloads.
pub(crate) fn render_factor_list<C: Display>(factors: &[C]) -> String {
    let mut rendered = String::from("[");
    for (index, factor) in factors.iter().enumerate() {
        if index > 0 {
            rendered.push_str(", ");
        }
        rendered.push_str(&factor.to_string());
    }
    rendered.push(']');
    rendered
}

/// Capture-free forward-mode (JVP) rule for [`ScanOperation`], staging **one fused** jvp `scan` with doubled
/// carries and doubled scanned inputs as an ordinary primal-enum `scan` operation over the shared builder.
///
/// The rule builds the body's fused jvp program through [`DifferentiableProgramOperation::jvp_program`] and permutes
/// its doubled signature into scan order, giving a fused body
/// `[primal_carries..., tangent_carries..., primal_slices..., tangent_slices...] ->
/// [primal_next_carries..., tangent_next_carries..., primal_outputs..., tangent_outputs...]`, and stages one scan
/// with `2 * carry_count` carries over the operand primals and tangents. Pure forward mode therefore runs a single
/// loop pass and stores **no** per-iteration residual stacks — the JAX jvp-of-`scan` shape.
///
/// The primal/tangent separation that reverse mode needs is deferred to partial evaluation: the known-ness split of
/// [`Program::linearize`](crate::Program::linearize) marks the primal halves known and the tangent halves unknown,
/// and the scan known-ness split (ryft's `_scan_partial_eval` analogue) separates the fused scan into a known
/// primal scan — stacking exactly the per-iteration known→unknown edges the tangent side consumes — and a residual
/// tangent scan over `[tangent_carries..., tangent_slices..., edge_slices...]`, the transposable linear-scan shape.
/// Residual stacks therefore exist only when linearization actually demands them.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C>
    for ScanOperation<C::Constant, C::Operation>
where
    C::Operation: Clone
        + From<ZeroOperation<ArrayType>>
        + From<ScanOperation<C::Constant, C::Operation>>
        + DifferentiableProgramOperation<C::Constant, C::Operation>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        let carry_count = self.carry_count();
        let length = self.length();
        let reverse = self.reverse();
        let unroll = self.unroll();
        let body_input_count = self.body().input_types().len();
        let body_output_count = self.body().output_types().len();
        check_count!("input", inputs, body_input_count, ProgramError);

        // Build the fused jvp body over `[primal_body_inputs..., tangent_body_inputs...]` and permute its doubled
        // signature into scan order (carries lead scanned inputs on both sides).
        let mut fused_body = C::Operation::jvp_program(self.body())?;
        fused_body.input_ids = permute_doubled_scan_signature(fused_body.input_ids, body_input_count, carry_count);
        fused_body.output_ids = permute_doubled_scan_signature(fused_body.output_ids, body_output_count, carry_count);

        // Stage the fused scan with doubled carries over
        // `[primal_carry_inits..., tangent_carry_inits..., primal_stacks..., tangent_stacks...]`.
        let fused_scan = ScanOperation::<C::Constant, C::Operation>::new(fused_body, 2 * carry_count, length)?
            .with_reverse(reverse)
            .with_unroll(unroll)?;
        // The fused scan takes every carry and scanned tangent as a real program input, so materialize structural
        // zeros at this sub-program boundary.
        let mut operands = Vec::with_capacity(2 * body_input_count);
        operands.extend(inputs[..carry_count].iter().map(|input| input.primal().clone()));
        for input in &inputs[..carry_count] {
            operands.push(input.tangent().clone().materialize(context)?);
        }
        operands.extend(inputs[carry_count..].iter().map(|input| input.primal().clone()));
        for input in &inputs[carry_count..] {
            operands.push(input.tangent().clone().materialize(context)?);
        }
        let outputs = context.bind(C::Operation::from(fused_scan), &operands)?;
        check_count!("output", outputs, 2 * body_output_count, ProgramError);

        // The fused scan's outputs are `[primal_final_carries..., tangent_final_carries..., primal_stacked...,
        // tangent_stacked...]`; zip the matching halves back into `DifferentiationDual`s in the original output order.
        let scanned_output_count = body_output_count - carry_count;
        let mut jvp_outputs = Vec::with_capacity(body_output_count);
        for index in 0..carry_count {
            jvp_outputs.push(DifferentiationDual::new(outputs[index].clone(), outputs[carry_count + index].clone()));
        }
        for index in 0..scanned_output_count {
            jvp_outputs.push(DifferentiationDual::new(
                outputs[2 * carry_count + index].clone(),
                outputs[2 * carry_count + scanned_output_count + index].clone(),
            ));
        }
        Ok(jvp_outputs)
    }
}

/// Permutes one side of a fused jvp body's doubled scan signature from jvp order
/// (`[primal_entries..., tangent_entries...]`, each of length `half`) into scan order, where carries lead the
/// scanned entries on both the primal and tangent halves:
/// `[primal_carries..., tangent_carries..., primal_scanned..., tangent_scanned...]`.
fn permute_doubled_scan_signature(ids: Vec<AtomId>, half: usize, carry_count: usize) -> Vec<AtomId> {
    let mut permuted = Vec::with_capacity(ids.len());
    permuted.extend_from_slice(&ids[..carry_count]);
    permuted.extend_from_slice(&ids[half..half + carry_count]);
    permuted.extend_from_slice(&ids[carry_count..half]);
    permuted.extend_from_slice(&ids[half + carry_count..]);
    permuted
}

/// Extracts slice `iteration` of a stacked batch along its *logical* leading axis and drops that axis.
///
/// The logical leading axis is the scan length axis: physical axis `1` when the batch axis sits at physical axis
/// `0`, and physical axis `0` otherwise. The iteration batch keeps the input's batch axis, decremented when it sat
/// after the dropped axis.
fn read_scan_iteration_batch<V>(stack: &ArrayBatch<V>, iteration: usize) -> Result<ArrayBatch<V>, BatchingError>
where
    V: Value<Type = ArrayType> + Slice + Reshape,
{
    let stack_axis = match stack.batch_axis().axis() {
        Some(0) => 1,
        _ => 0,
    };
    let stack_type = stack.r#type().into_owned();
    let dimensions = stack_type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| {
            dimension.value().ok_or_else(|| {
                BatchingError::UnsupportedOperation {
                    message: format!("scan batching requires static stacked input types but got {stack_type}"),
                }
                .into()
            })
        })
        .collect::<Result<Vec<usize>, ProgramError>>()?;
    let mut start_indices = vec![0; dimensions.len()];
    start_indices[stack_axis] = iteration;
    let mut limit_indices = dimensions.clone();
    limit_indices[stack_axis] = iteration + 1;
    let unit_strides = vec![1; dimensions.len()];
    let iteration_value =
        stack
            .value()
            .clone()
            .slice(start_indices.as_slice(), limit_indices.as_slice(), unit_strides.as_slice())?;
    let iteration_dimensions = dimensions
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != stack_axis)
        .map(|(_, &dimension)| Size::Static(dimension))
        .collect::<Vec<_>>();
    let iteration_value = iteration_value.reshape(Shape::new(iteration_dimensions))?;
    let iteration_type = iteration_value.r#type().into_owned();
    let batch_axis = stack.batch_axis().axis().map(|axis| if axis > stack_axis { axis - 1 } else { axis });
    ArrayBatch::new(iteration_type, iteration_value, batch_axis)
}

/// Per-output stacking state used by [`batch_scan_with_interpreter`]: the accumulator batch holding the iterations
/// written so far, together with the batch axis every iteration must agree on.
struct ScanOutputAccumulator<V: Typed<Type = ArrayType>> {
    /// Stacked accumulator; its leading physical axis is the scan length axis.
    accumulator: V,

    /// Batch axis of the per-item values written into the accumulator, if the output is batch-varying.
    batch_axis: Option<usize>,
}

/// Drives one batched scan loop over `[carry..., stacked_xs...]` input batches, delegating each iteration's body
/// evaluation to `interpret_iteration` and allocating stacked output accumulators through `allocate_zero`.
///
/// Per-iteration slices of the stacked inputs are read along their *logical* leading axis (see
/// [`read_scan_iteration_batch`]) so the batch axis threads through untouched, and the per-iteration outputs are
/// stacked along a fresh leading physical axis, shifting each output's batch axis right by one. The visit order
/// reverses when `reverse` is `true` while output slice `i` stays aligned with input slice `i`, exactly like the
/// unbatched scan loop.
pub(crate) fn batch_scan_with_interpreter<V, AllocateZeroFn, InterpretIterationFn>(
    carry_count: usize,
    length: usize,
    reverse: bool,
    y_slice_types: &[ArrayType],
    inputs: &[ArrayBatch<V>],
    mut allocate_zero: AllocateZeroFn,
    mut interpret_iteration: InterpretIterationFn,
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    V: Value<Type = ArrayType> + Slice + UpdateSlice + Reshape,
    AllocateZeroFn: FnMut(&ArrayType) -> Result<V, ProgramError>,
    InterpretIterationFn: FnMut(usize, Vec<ArrayBatch<V>>) -> Result<Vec<ArrayBatch<V>>, BatchingError>,
{
    let (initial_carries, stacks) = inputs.split_at(carry_count);
    let mut carries = initial_carries.to_vec();
    let mut accumulators: Vec<Option<ScanOutputAccumulator<V>>> = (0..y_slice_types.len()).map(|_| None).collect();
    let mut iterations: Vec<usize> = (0..length).collect();
    if reverse {
        iterations.reverse();
    }
    for iteration in iterations {
        let mut iteration_inputs = carries.clone();
        for stack in stacks {
            iteration_inputs.push(read_scan_iteration_batch(stack, iteration)?);
        }
        let mut iteration_outputs = interpret_iteration(iteration, iteration_inputs)?;
        check_count!("output", iteration_outputs, carry_count + y_slice_types.len(), ProgramError);
        let iteration_ys = iteration_outputs.split_off(carry_count);
        carries = iteration_outputs;
        for (accumulator, iteration_y) in accumulators.iter_mut().zip(iteration_ys.into_iter()) {
            let batch_axis = iteration_y.batch_axis().axis();
            let iteration_type = iteration_y.r#type().into_owned();
            let accumulator = match accumulator {
                Some(accumulator) => {
                    if accumulator.batch_axis != batch_axis {
                        return Err(BatchingError::MisalignedBatchAxes {
                            message: format!(
                                "scan body produced stacked output iterations at mismatched batch axes ({:?} vs \
                                 {batch_axis:?})",
                                accumulator.batch_axis,
                            ),
                        }
                        .into());
                    }
                    accumulator
                }
                None => accumulator.insert(ScanOutputAccumulator {
                    accumulator: allocate_zero(&stacked_scan_type(&iteration_type, length))?,
                    batch_axis,
                }),
            };
            let mut expanded_dimensions = Vec::with_capacity(iteration_type.rank() + 1);
            expanded_dimensions.push(Size::Static(1));
            expanded_dimensions.extend(iteration_type.shape().dimensions().iter().cloned());
            let expanded = iteration_y.into_value().reshape(Shape::new(expanded_dimensions))?;
            let mut start_indices = vec![0; iteration_type.rank() + 1];
            start_indices[0] = iteration;
            accumulator.accumulator = accumulator.accumulator.update_slice(&expanded, start_indices.as_slice())?;
        }
    }
    let mut outputs = carries;
    for (accumulator, y_slice_type) in accumulators.into_iter().zip(y_slice_types.iter()) {
        match accumulator {
            Some(ScanOutputAccumulator { accumulator, batch_axis }) => {
                let stacked_type = accumulator.r#type().into_owned();
                outputs.push(ArrayBatch::new(stacked_type, accumulator, batch_axis.map(|axis| axis + 1))?);
            }
            None => {
                // A zero-length scan writes no iterations, so each stacked output is the replicated empty stack of
                // the body's per-iteration output type.
                let stacked_type = stacked_scan_type(y_slice_type, length);
                outputs.push(ArrayBatch::replicated(allocate_zero(&stacked_type)?));
            }
        }
    }
    Ok(outputs)
}

impl<V, O> BatchableOperation<V, EagerContext<V, O>> for ScanOperation<V, O>
where
    V: Value<Type = ArrayType> + Slice + UpdateSlice + Reshape,
    EagerContext<V, O>: Zero<V>,
    O: BatchableOperation<V, EagerContext<V, O>>,
{
    fn batch(
        &self,
        context: &EagerContext<V, O>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        let y_slice_types = self.body().output_types().split_off(self.carry_count());
        batch_scan_with_interpreter(
            self.carry_count(),
            self.length(),
            self.reverse(),
            y_slice_types.as_slice(),
            inputs,
            |stacked_type| context.zero(stacked_type),
            |_, iteration_inputs| {
                self.body().interpret_with(
                    iteration_inputs,
                    |_, constant| Ok(ArrayBatch::replicated(constant.clone())),
                    |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
                )
            },
        )
    }
}

impl<C, O> BatchableOperation<<C as Domain>::Value, BatchingContext<C>> for ScanOperation<C::Constant, O>
where
    C: Context<Type = ArrayType> + Zero<<C as Domain>::Value>,
    C::Constant: Value<Type = ArrayType>,
    <C as Domain>::Value: Slice + UpdateSlice + Reshape,
    C::Operation:
        From<ZeroOperation<ArrayType>> + From<SliceOperation> + From<UpdateSliceOperation> + From<ReshapeOperation>,
    O: BatchableOperation<<C as Domain>::Value, BatchingContext<C>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        let y_slice_types = self.body().output_types().split_off(self.carry_count());
        batch_scan_with_interpreter(
            self.carry_count(),
            self.length(),
            self.reverse(),
            y_slice_types.as_slice(),
            inputs,
            |stacked_type| context.parent().zero(stacked_type),
            |_, iteration_inputs| context.interpret_program(self.body(), iteration_inputs),
        )
    }
}

/// Type-family transposition semantics for [`ScanOperation`], with the scan's value, body-operation, capture,
/// payload, and staging-target parameters riding as trait inputs and the type family as the implementing type
/// (mirroring [`ScanPayload`](crate::operations::control_flow::scan::ScanPayload)) so that the [`ArrayType`] and
/// [`DataType`] rules stay coherent without the operation struct naming its type family as a parameter. The
/// [`ArrayType`] rule pins the staging target to the scan's own body operation family `O`, while the [`DataType`]
/// rule keeps an independent `Target` because a scalar linear scan never inlines its body into the pullback.
pub(crate) trait ScanTransposition<V, O, F, Payload, Target>: Type
where
    V: Value<Type = Self>,
    Target: Operation<Self>,
{
    /// Applies the type family's `scan` transpose rule; refer to the documentation of
    /// [`TransposableOperation::transpose`] for the contract.
    fn transpose_scan(
        operation: &ScanOperation<V, O, F, Payload>,
        context: &mut TracingContext<V, Target>,
        inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, ProgramError>;
}

/// Transpose rule for array scans, covering both scan forms that reach a reverse pass.
///
/// A *captured* linear scan (non-empty [`captures`](ScanOperation::captures)) is transposed whole: linear-scan
/// transposition is total because the body pushforward maps `[carry..., x_slice...]` to `[carry..., y_slice...]`, so
/// its program transpose maps `[carry_cotangent..., y_slice_cotangent...]` to
/// `[carry_cotangent..., x_slice_cotangent...]` — the same scan-body signature with the same carry count. Flipping
/// `reverse` pairs cotangent iteration `i` with residual stack iteration `i` exactly when the forward scan consumed
/// them, so the same residual stacks (and the lowering-only unroll factor) carry over verbatim.
///
/// A capture-free scan is a *primal* operand-form scan whose known residual stacks ride as ordinary operands, so it is
/// forwarded to the partition-aware [`transpose_primal_scan`] rule instead. Both forms recurse into the body through
/// the [`TransposableProgramOperation`] fixed-point witness, keeping the scan-local fixed point owned by the operation
/// family with no recursive [`TransposableOperation`] obligation on `O`.
impl<V, F, O, Payload> ScanTransposition<V, O, F, Payload, O> for ArrayType
where
    V: Value<Type = ArrayType>,
    F: Value<Type = ArrayType>,
    O: TransposableProgramOperation<V> + From<ZeroOperation<ArrayType>> + From<ScanOperation<V, O, F, Payload>>,
{
    fn transpose_scan(
        operation: &ScanOperation<V, O, F, Payload>,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        // A scan with only zero output cotangents is a zero linear map, so every input cotangent is zero.
        if outputs.iter().all(MaybeZero::is_zero) {
            return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect());
        }
        if operation.captures().is_empty() {
            return transpose_primal_scan(operation, context, inputs, outputs);
        }
        let body = operation.body();
        let carry_count = operation.carry_count();
        let length = operation.length();
        let transposed_body =
            <O as TransposableProgramOperation<V>>::transpose_program(body, &vec![true; body.input_ids().len()])?;
        let transposed = ScanOperation::<V, O, F, Payload>::new_with_payload(transposed_body, carry_count, length)?
            .with_reverse(!operation.reverse())
            .with_unroll(operation.unroll())?
            .with_captures(operation.captures().to_vec());
        let mut output_types = body.output_types();
        let y_slice_types = output_types.split_off(carry_count);
        output_types.extend(y_slice_types.iter().map(|slice_type| stacked_scan_type(slice_type, length)));
        check_count!("output", outputs, output_types.len(), ProgramError);
        let materialized = outputs
            .iter()
            .map(|cotangent| cotangent.clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        let cotangents = context.stage_operation(O::from(transposed), materialized.as_slice())?;
        check_count!("output", cotangents, inputs.len(), ProgramError);
        Ok(cotangents.into_iter().map(MaybeZero::Value).collect())
    }
}

/// Partition-aware transpose rule for a *primal* [`ScanOperation`], used when the direct reverse transposes a
/// tangent program in the primal operation family `O` rather than re-keying it into the linear family. This is the
/// operand-form counterpart of the captured-stack linear-scan transpose rule above: the per-iteration residuals are
/// ordinary *scanned operands* (known residual stacks supplied through `operand_values`) instead of capture payloads,
/// so the rule reads them from the pullback and threads them back through as known scanned operands of a transposed
/// scan with the same scan-loop geometry.
///
/// The operands mirror the body's inputs one-to-one as `[carries..., scanned_inputs...]`, and each operand is
/// independently linear (a tangent the reverse accumulates) or known (a residual stack the pullback reads). The
/// forward typically marks the carry-and-scanned tangents linear and the residual stacks known, but the linear
/// operands need not form a leading run: vmapping a bounded `while` threads a non-differentiable Boolean mask as a
/// known *carry*, so a known operand can sit among the linear carries. This rule therefore:
///
///   1. Transposes the body with [`TransposableProgramOperation::transpose_program`] under each
///      input's own linearity. The transposed body maps every body output's cotangent followed by every known body
///      input's runtime value to every *linear* body input's cotangent:
///      `[carry_output_cotangent..., y_slice_cotangent..., known_input_value...] -> [linear_input_cotangent...]`.
///   2. Restores the reversed scan's carry-output arity, which
///      [`Program::transpose_with_respect_to`](crate::Program::transpose_with_respect_to) erases for known
///      carries (a known carry is not a linear input, so it contributes no carry cotangent output). Each known carry's
///      cotangent output is re-inserted as a structural zero of that carry output's cotangent type, mirroring how the
///      split restores pruned tangent outputs, so the reversed body produces one carry cotangent per carry followed
///      by one scanned-output cotangent per linear scanned input.
///   3. Re-stages a primal [`ScanOperation`] over the restored body with flipped [`reverse`](ScanOperation::reverse)
///      and the same carry count, length, and (lowering-only) unroll factor, over `[outputs...,
///      known_input_value_stacks...]`. The known-input-value stacks are the residual stacks for known scanned inputs
///      and typed zero stacks for known carries (whose per-iteration values were threaded as a carry rather than
///      residualized, and which a transposed body only reads when the linear computation depends on them). Flipping
///      `reverse` pairs cotangent iteration `i` with residual stack iteration `i` exactly when the forward scan
///      consumed them, making reverse mode through the scan total with no array-reversal operation.
///
/// The returned cotangents place the reversed scan's carry cotangents at the carry-operand positions, its
/// scanned-output cotangents at the linear scanned-operand positions, and a structural [`MaybeZero::Zero`] at the
/// known scanned-operand positions, which carry no cotangent. The body recursion happens through the
/// [`TransposableProgramOperation`] fixed-point witness in the same operation family, so it introduces no recursive
/// [`TransposableOperation`] obligation on `O`.
///
/// # Parameters
///
///   - `operation`: Primal scan staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `inputs`: Per-operand [`PartialValue`] knowledge, mirroring the body inputs as `[carries..., scanned_inputs...]`.
///     A linear operand is [`Unknown`](PartialValue::Unknown); a known operand is
///     [`Known`](PartialValue::Known) of the residual-stack tracer the pullback reads.
///   - `outputs`: Symbolic cotangents for the scan's outputs.
pub fn transpose_primal_scan<V, O, F, Payload>(
    operation: &ScanOperation<V, O, F, Payload>,
    context: &mut TracingContext<V, O>,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError>
where
    V: Value<Type = ArrayType>,
    F: Value<Type = ArrayType>,
    O: TransposableProgramOperation<V> + From<ZeroOperation<ArrayType>> + From<ScanOperation<V, O, F, Payload>>,
{
    // A scan with only zero output cotangents is a zero linear map, so every operand cotangent is zero.
    if outputs.iter().all(MaybeZero::is_zero) {
        return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect());
    }

    // Operand layout is `[carries..., scanned_inputs...]`, mirroring the body's input order one-to-one, where each
    // operand is independently linear (a tangent the reverse must accumulate) or known (a residual stack the pullback
    // reads). Linear operands need not form a leading run: vmapping a bounded `while` threads a non-differentiable
    // Boolean mask as a known *carry*, so a known operand can sit among the linear carries. The leading `carry_count`
    // operands are the carries and the rest are scanned inputs.
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let body = operation.body();
    let carry_count = operation.carry_count();
    let length = operation.length();
    check_count!("input", operand_linear, body.input_types().len(), ProgramError);
    if carry_count > operand_linear.len() {
        return Err(ProgramError::MalformedProgram(format!(
            "scan transpose found carry count {carry_count} exceeding its {} operands",
            operand_linear.len(),
        )));
    }

    // Transpose the body with each input's own linearity. The transposed body maps the cotangent of every body output
    // followed by every known body input's runtime value to the cotangent of every *linear* body input only:
    // `[carry_output_cotangent..., y_slice_cotangent..., known_input_value...] -> [linear_input_cotangent...]`, in body
    // order on each side.
    let mut transposed_body = O::transpose_program(body, operand_linear.as_slice())?;

    // `transpose_with_respect_to` emits one cotangent output per linear input, so a known carry contributes no carry
    // output and the transposed body has fewer carry outputs than the reversed scan's `carry_count` requires. Restore
    // the carry-output arity exactly as the split restores pruned tangent outputs: walk the carry positions, taking
    // the next linear-carry cotangent where the carry is linear and inserting a fresh structural zero of that carry
    // output's cotangent type where it is known. The trailing transposed outputs (the linear scanned-input cotangents)
    // are carried over unchanged, so the reversed body produces `[carry_cotangent..., scanned_input_cotangent...]`.
    let linear_carry_count = operand_linear[..carry_count].iter().filter(|&&linear| linear).count();
    if linear_carry_count != carry_count {
        let trailing_outputs = transposed_body.output_ids.split_off(linear_carry_count);
        let mut linear_carry_outputs = transposed_body.output_ids.split_off(0).into_iter();
        for (carry_index, &carry_is_linear) in operand_linear[..carry_count].iter().enumerate() {
            if carry_is_linear {
                transposed_body.output_ids.push(linear_carry_outputs.next().unwrap());
            } else {
                // A differentiable carry's cotangent slot carries its cotangent dual; a non-differentiable carry (the
                // `float0` analogue) has no cotangent space, so its slot carries only structural zeros typed by the
                // carry's own primal type. Either way the slot type matches the reversed scan's carry slot type fed by
                // the scan output cotangent below.
                let output_type = &body.output_types()[carry_index];
                let cotangent_type = output_type.cotangent().unwrap_or_else(|| output_type.clone());
                let zero_output = AtomId::new(transposed_body.atoms.len());
                transposed_body.atoms.push(Atom::Variable(cotangent_type.clone()));
                transposed_body.instructions.push(Instruction::new(
                    O::from(ZeroOperation::new(cotangent_type)),
                    Vec::new(),
                    vec![zero_output],
                ));
                transposed_body.output_ids.push(zero_output);
            }
        }
        transposed_body.output_ids.extend(trailing_outputs);
        transposed_body.output_structure = vec![Placeholder; transposed_body.output_ids.len()];
    }

    let transposed = ScanOperation::<V, O, F, Payload>::new_with_payload(transposed_body, carry_count, length)?
        .with_reverse(!operation.reverse())
        .with_unroll(operation.unroll())?;

    // Stage the reversed scan over `[outputs..., known_input_value_stacks...]`, matching the transposed
    // body's input order. The output cotangents are typed by the *scan operation's* outputs, not the body's
    // per-iteration outputs: the leading carries keep their per-iteration shape while each trailing y-slice output is
    // stacked along the scan length. Using the body's per-iteration y-slice types here would materialize a zero
    // cotangent for a dead y-output with the un-stacked slice type, desyncing the reversed scan's operand signature.
    let mut output_types = body.output_types();
    let y_slice_types = output_types.split_off(carry_count);
    output_types.extend(y_slice_types.iter().map(|slice_type| stacked_scan_type(slice_type, length)));
    check_count!("output", outputs, output_types.len(), ProgramError);
    let mut operands = Vec::with_capacity(output_types.len() + operand_linear.len());
    for cotangent in outputs {
        operands.push(cotangent.clone().materialize(context)?);
    }

    // Append one scanned operand per known body input, in body order, to feed the transposed body's known-value
    // inputs. A known *scanned* input is a residual stack read from the pullback; a known *carry* has no stored stack
    // (its per-iteration values were threaded as a carry, not residualized), but `transpose_with_respect_to` only exposes
    // its value as a known input when the linear computation actually reads it, so a known carry that survives here is
    // unused by the transposed body and a typed zero stack of its stacked carry type satisfies the operand signature.
    // A known intermediate (a known operand with no pullback value) is one the partial-evaluation split never leaves in
    // a tangent program, so its absence is a malformed program.
    for (index, &linear) in operand_linear.iter().enumerate() {
        if linear {
            continue;
        }
        if index < carry_count {
            let stacked_type = stacked_scan_type(&body.input_types()[index], length);
            let mut zeros = context.stage_nullary_operation(ZeroOperation::new(stacked_type))?;
            check_count!("output", zeros, 1, ProgramError);
            operands.push(zeros.remove(0));
        } else {
            // A known scanned operand is a residual stack; the dispatch guarantees it carries its pullback value.
            let residual = inputs[index].as_known().ok_or_else(|| {
                ProgramError::MalformedProgram(format!("scan transpose operand {index} has no known residual value"))
            })?;
            operands.push(residual.clone());
        }
    }

    // The reversed scan outputs one carry cotangent per carry and one stacked scanned-output cotangent per *linear*
    // scanned input.
    let linear_scanned_count = operand_linear[carry_count..].iter().filter(|&&linear| linear).count();
    let scan_cotangents = context.stage_operation(O::from(transposed), operands.as_slice())?;
    check_count!("output", scan_cotangents, carry_count + linear_scanned_count, ProgramError);

    // Reassemble one cotangent per operand. The reversed scan outputs `[carry_cotangent..., scanned_input_cotangent...]`,
    // the carry cotangents (including the re-inserted zeros for known carries) leading the scanned-input cotangents over
    // the *linear* scanned inputs. Every carry operand precedes every scanned operand, so a single sequential drain
    // hands each carry operand the next carry cotangent and each linear scanned operand the next scanned-input
    // cotangent in turn; known scanned operands carry a structural zero (they are residual stacks, which carry no
    // cotangent).
    let mut scan_cotangents = scan_cotangents.into_iter();
    let cotangents = operand_linear
        .iter()
        .zip(inputs)
        .enumerate()
        .map(|(index, (&linear, input))| {
            if index < carry_count || linear {
                MaybeZero::Value(scan_cotangents.next().unwrap())
            } else {
                MaybeZero::Zero(input.r#type().into_owned())
            }
        })
        .collect();
    Ok(cotangents)
}

impl<V, F, O, Payload, Target> ScanTransposition<V, O, F, Payload, Target> for DataType
where
    V: Value<Type = DataType>,
    F: Value<Type = DataType>,
    O: TransposableProgramOperation<V>,
    Target: Operation<DataType> + From<ZeroOperation<DataType>> + From<ScanOperation<V, O, F, Payload>>,
{
    fn transpose_scan(
        operation: &ScanOperation<V, O, F, Payload>,
        context: &mut TracingContext<V, Target>,
        inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, ProgramError> {
        if outputs.iter().all(MaybeZero::is_zero) {
            return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect());
        }
        if !operation.captures().is_empty() {
            return Err(ProgramError::UnsupportedOperation {
                message: "scalar linear scan transposition with residual stacks requires a scalar stack representation"
                    .to_string(),
            });
        }
        let body = operation.body();
        let output_types = body.output_types();
        check_count!("output", outputs, output_types.len(), ProgramError);
        let transposed_body =
            <O as TransposableProgramOperation<V>>::transpose_program(body, &vec![true; body.input_ids().len()])?;
        let transposed = ScanOperation::<V, O, F, Payload>::new_with_payload(
            transposed_body,
            operation.carry_count(),
            operation.length(),
        )?
        .with_reverse(!operation.reverse())
        .with_unroll(operation.unroll())?
        .with_captures(operation.captures().to_vec());
        let materialized = outputs
            .iter()
            .map(|cotangent| cotangent.clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        let cotangents = context.stage_operation(Target::from(transposed), materialized.as_slice())?;
        check_count!("output", cotangents, inputs.len(), ProgramError);
        Ok(cotangents.into_iter().map(MaybeZero::Value).collect())
    }
}

/// Transpose rule for [`ScanOperation`], dispatching to the scan's type family through [`ScanTransposition`]: array
/// scans transpose captured linear scans whole and forward operand-form primal scans to [`transpose_primal_scan`],
/// and scalar scans transpose capture-free carry-only linear scans.
impl<V, F, O, Payload, Target> TransposableOperation<V, Target> for ScanOperation<V, O, F, Payload>
where
    V: Value,
    V::Type: ScanTypeSemantics + ScanTransposition<V, O, F, Payload, Target>,
    F: Value<Type = V::Type>,
    O: Operation<V::Type>,
    Target: Operation<V::Type>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, Target>,
        inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, ProgramError> {
        <V::Type>::transpose_scan(self, context, inputs, outputs)
    }
}

#[cfg(test)]
mod tests {

    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::operations::arithmetic::MulOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder};
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::tracing_v2::differentiation::ForwardModeDifferentiate;

    use crate::types::DataType;

    use super::*;
    use crate::batching::BatchAxis;

    type TestOperation = ArrayOperation<TestArray>;
    type TestEagerContext = EagerContext<TestArray, TestOperation>;
    type TestScanOperation = ScanOperation<TestArray, TestOperation>;

    /// Builds a cumulative-product body program that maps `[carry, x]` to `[carry * x, carry * x]`.
    fn product_body() -> Program<TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<TestArray, TestOperation>::new();
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let product = builder.add_instruction(MulOperation, vec![carry, x]).unwrap()[0];
        builder
            .build(vec![product, product], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds the `f64` array type with the provided static dimensions.
    fn f64_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(dimensions.iter().copied().map(Size::Static).collect()))
    }

    /// Builds a cumulative-product [`ScanOperation`] whose nested depth matches `lengths`.
    fn product_scan_with_lengths(lengths: &[usize]) -> ScanOperation<TestArray, TestOperation> {
        assert!(!lengths.is_empty());
        if lengths.len() == 1 {
            return TestScanOperation::new(product_body(), 1, lengths[0]).unwrap();
        }
        let inner_scan = product_scan_with_lengths(&lengths[1..]);
        let mut builder = ProgramBuilder::<TestArray, TestOperation>::new();
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let xs = builder.add_input(f64_type(&lengths[1..]));
        let outputs = builder
            .add_instruction(TestOperation::Scan(Box::new(inner_scan)), vec![carry, xs])
            .unwrap()
            .to_vec();
        let body = builder.build(outputs, vec![Placeholder, Placeholder], vec![Placeholder, Placeholder]).unwrap();
        TestScanOperation::new(body, 1, lengths[0]).unwrap()
    }

    /// Builds the cumulative-product [`ScanOperation`] over three iterations used by the differentiation tests.
    fn product_scan() -> ScanOperation<TestArray, TestOperation> {
        product_scan_with_lengths(&[3])
    }

    /// The fused JVP rule stages exactly one scan with doubled carries and **no** per-iteration residual stacks:
    /// pure forward mode pays a single loop pass and no reverse-mode storage. Residual stacks appear only when
    /// [`Program::linearize`] actually splits the fused program (its known scan then stacks the known→unknown
    /// edges), which the trailing assertion pins.
    #[test]
    fn test_scan_jvp_stages_one_fused_scan_with_no_residual_stacks() {
        use crate::contexts::Domain;
        use crate::tracing::DomainTracer;
        use crate::types::{Shape, Size};

        let scan = product_scan();
        let (_, program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
            |(init, xs): (
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
            )| {
                let mut outputs = init.context().stage_operation(TestOperation::Scan(Box::new(scan)), &[&init, &xs])?;
                let ys = outputs.remove(1);
                Ok((outputs.remove(0), ys))
            },
            (ArrayType::scalar(DataType::F64), ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]))),
        )
        .unwrap();
        let program = program.to_flat_program();

        let jvp = program.jvp().unwrap().into_simplified().unwrap();
        let scans = jvp
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.operation() {
                TestOperation::Scan(operation) => Some(operation),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(scans.len(), 1);
        assert_eq!(scans[0].carry_count(), 2);
        // The fused body is `[primal_carry, tangent_carry, primal_x, tangent_x] ->
        // [primal_carry', tangent_carry', primal_y, tangent_y]`: doubled arity and nothing else.
        assert_eq!(scans[0].body().input_types().len(), 4);
        assert_eq!(scans[0].body().output_types().len(), 4);

        // Linearizing the same program is what materializes residual stacks, as known-scan edges.
        let linearization = program.linearize().unwrap();
        assert!(linearization.residual_count() >= 1);
    }

    #[test]
    fn test_scan_jvp_propagates_tangents_through_linear_scan() {
        // Cumulative product over `xs = [2, 3, 4]` starting at `init = 1`: the final carry is 24 and the running
        // products are `[2, 6, 24]`. A unit tangent on `init` propagates as `d(init * x0 * x1 * x2)/d(init) = 24`
        // on the final carry and `[2, 6, 24]` on the stacked outputs.
        let scan = product_scan();
        let ((carry, ys), (carry_tangent, ys_tangent)) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().bind(TestOperation::Scan(Box::new(scan)), &[init.clone(), xs.clone()])?;
                    let ys = outputs.remove(1);
                    Ok((outputs.remove(0), ys))
                },
                (TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])),
                (TestArray::scalar(1.0), TestArray::vector(vec![0.0, 0.0, 0.0])),
            )
            .unwrap();
        assert_eq!(carry.values, vec![24.0]);
        assert_eq!(ys.values, vec![2.0, 6.0, 24.0]);
        assert_eq!(carry_tangent.values, vec![24.0]);
        assert_eq!(ys_tangent.values, vec![2.0, 6.0, 24.0]);

        // A unit tangent on `xs[1]` propagates as `d(init * x0 * x1 * x2)/d(x1) = init * x0 * x2 = 8` on the final
        // carry and `[0, 2, 8]` on the stacked outputs (`y0` does not depend on `x1`).
        let scan = product_scan();
        let ((carry, _), (carry_tangent, ys_tangent)) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().bind(TestOperation::Scan(Box::new(scan)), &[init.clone(), xs.clone()])?;
                    let ys = outputs.remove(1);
                    Ok((outputs.remove(0), ys))
                },
                (TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])),
                (TestArray::scalar(0.0), TestArray::vector(vec![0.0, 1.0, 0.0])),
            )
            .unwrap();
        assert_eq!(carry.values, vec![24.0]);
        assert_eq!(carry_tangent.values, vec![8.0]);
        assert_eq!(ys_tangent.values, vec![0.0, 2.0, 8.0]);
    }

    #[test]
    fn test_scan_jvp_supports_nested_scans_in_linear_scan_bodies() {
        // Nested scans differentiate by recursively replaying the inner linear scan inside each outer scan iteration.
        // The final carry is the product of every element, and a unit tangent on the initial carry follows the same
        // cumulative-product path through both scan levels.
        let scan = product_scan_with_lengths(&[2, 3]);
        let ((carry, ys), (carry_tangent, ys_tangent)) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().bind(TestOperation::Scan(Box::new(scan)), &[init.clone(), xs.clone()])?;
                    let ys = outputs.remove(1);
                    Ok((outputs.remove(0), ys))
                },
                (TestArray::scalar(1.0), TestArray::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0])),
                (TestArray::scalar(1.0), TestArray::matrix(2, 3, vec![0.0; 6])),
            )
            .unwrap();
        assert_eq!(carry.values, vec![5040.0]);
        assert_eq!(ys.values, vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]);
        assert_eq!(carry_tangent.values, vec![5040.0]);
        assert_eq!(ys_tangent.values, vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]);
    }

    #[test]
    fn test_scan_jvp_supports_three_nested_scans_in_linear_scan_bodies() {
        // Three levels catches the recursive fixed point that failed for nested scan bodies: the middle scan's
        // linear body contains another scan whose body also has scan-local residual references.
        let scan = product_scan_with_lengths(&[2, 2, 2]);
        let xs_type = f64_type(&[2, 2, 2]);
        let ((carry, ys), (carry_tangent, ys_tangent)) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().bind(TestOperation::Scan(Box::new(scan)), &[init.clone(), xs.clone()])?;
                    let ys = outputs.remove(1);
                    Ok((outputs.remove(0), ys))
                },
                (TestArray::scalar(1.0), TestArray::new(xs_type.clone(), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
                (TestArray::scalar(1.0), TestArray::new(xs_type, vec![0.0; 8])),
            )
            .unwrap();
        assert_eq!(carry.values, vec![362880.0]);
        assert_eq!(ys.values, vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0]);
        assert_eq!(carry_tangent.values, vec![362880.0]);
        assert_eq!(ys_tangent.values, vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0]);
    }

    #[test]
    fn test_scan_batching_lifts_batched_carries() {
        // Batching a scan whose carry is mapped at axis 0 threads the batch axis through every iteration: each
        // batch item runs its own cumulative product over the shared `xs = [2, 3, 4]`, and the stacked outputs
        // gain the scan axis in front of the batch axis.
        let scan = product_scan();
        let context = TestEagerContext::new();
        let carries = {
            let value = TestArray::vector(vec![1.0, 2.0, 3.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let stacked_inputs = ArrayBatch::replicated(TestArray::vector(vec![2.0, 3.0, 4.0]));
        let outputs = scan.batch(&context, &[carries, stacked_inputs]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![24.0, 48.0, 72.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].r#type().shape().dimensions(), &[Size::Static(3), Size::Static(3)]);
        assert_eq!(outputs[1].value().values, vec![2.0, 4.0, 6.0, 6.0, 12.0, 18.0, 24.0, 48.0, 72.0]);
    }

    #[test]
    fn test_scan_batching_lifts_batched_stacked_inputs() {
        // Batching a scan whose stacked input is mapped at axis 0 reads each iteration's slice along the logical
        // leading axis (physical axis 1 when the batch axis sits at 0), so every batch item scans its own row.
        let scan = product_scan();
        let context = TestEagerContext::new();
        let carries = ArrayBatch::replicated(TestArray::scalar(1.0));
        let stacked_inputs = {
            let value = TestArray::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = scan.batch(&context, &[carries, stacked_inputs]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![24.0, 210.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].r#type().shape().dimensions(), &[Size::Static(3), Size::Static(2)]);
        assert_eq!(outputs[1].value().values, vec![2.0, 5.0, 6.0, 30.0, 24.0, 210.0]);

        // A trailing batch axis (physical `[3, 2]` with the batch axis at 1) reads the same logical iterations, so the
        // outputs are identical.
        let scan = product_scan();
        let context = TestEagerContext::new();
        let carries = ArrayBatch::replicated(TestArray::scalar(1.0));
        let stacked_inputs = {
            let value = TestArray::matrix(3, 2, vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(1))
        }
        .unwrap();
        let outputs = scan.batch(&context, &[carries, stacked_inputs]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![24.0, 210.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].value().values, vec![2.0, 5.0, 6.0, 30.0, 24.0, 210.0]);
    }

    #[test]
    fn test_scan_batching_threads_batched_carries_and_inputs() {
        // Batching both operands pairs batch item `i` of the carries with batch item `i` of the stacked inputs.
        let scan = product_scan();
        let context = TestEagerContext::new();
        let carries = {
            let value = TestArray::vector(vec![1.0, 10.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let stacked_inputs = {
            let value = TestArray::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = scan.batch(&context, &[carries, stacked_inputs]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![24.0, 2100.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].value().values, vec![2.0, 50.0, 6.0, 300.0, 24.0, 2100.0]);
    }

    #[test]
    fn test_scan_batching_respects_reverse_visit_order() {
        // A reversed batched scan visits the logical iterations from the back while keeping output iteration `i`
        // aligned with input iteration `i`: the reversed cumulative product over `[2, 3, 4]` is `[24, 12, 4]` per
        // batch item.
        let scan = product_scan().with_reverse(true);
        let context = TestEagerContext::new();
        let carries = ArrayBatch::replicated(TestArray::scalar(1.0));
        let stacked_inputs = {
            let value = TestArray::matrix(2, 3, vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = scan.batch(&context, &[carries, stacked_inputs]).unwrap();
        assert_eq!(outputs[0].value().values, vec![24.0, 24.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].value().values, vec![24.0, 24.0, 12.0, 12.0, 4.0, 4.0]);
    }
}
