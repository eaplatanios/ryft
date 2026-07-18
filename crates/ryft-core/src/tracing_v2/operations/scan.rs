use std::fmt::Display;

use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::macros::check_count;
use crate::operations::constants::{Zero, ZeroOperation};
use crate::operations::control_flow::ScanOperation;
use crate::operations::control_flow::scan::{ScanTypeSemantics, stacked_scan_type};
use crate::operations::manipulation::{
    Reshape, ReshapeOperation, Slice, SliceOperation, UpdateSlice, UpdateSliceOperation,
};
use crate::parameters::Placeholder;
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::types::{Type, Typed};
use crate::programs::{MaybeZero, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, DataType, Shape, Size};

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
/// The rule builds the body's fused jvp program through its instruction-scoped differentiation driver and permutes
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
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for ScanOperation<C::Constant>
where
    C::Operation: From<ZeroOperation<ArrayType>> + From<ScanOperation<C::Constant>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The rule requests all nested-computation work through its driver (region 0 is the body), which keeps
        // its bounds free of the operation family's own semantic traits.
        let carry_count = self.carry_count();
        let length = self.length();
        let reverse = self.reverse();
        let unroll = self.unroll();
        let fused_body = driver.jvp_program(driver.region(0)?)?;
        let body_input_count = fused_body.input_types().len() / 2;
        let body_output_count = fused_body.output_types().len() / 2;
        check_count!("input", inputs, body_input_count, ProgramError);

        // The fused jvp body is over `[primal_body_inputs..., tangent_body_inputs...]`; permute its doubled
        // signature into scan order (carries lead scanned inputs on both sides).
        let fused_body = permute_doubled_scan_body(fused_body, body_input_count, body_output_count, carry_count)?;

        // Stage the fused scan with doubled carries over
        // `[primal_carry_inits..., tangent_carry_inits..., primal_stacks..., tangent_stacks...]`.
        let fused_scan = ScanOperation::<C::Constant>::new(2 * carry_count, length)
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
        let outputs = context.bind(C::Operation::from(fused_scan), vec![fused_body], &operands)?;
        check_count!("output", outputs, 2 * body_output_count, ProgramError);

        // The fused scan's outputs are `[primal_final_carries..., tangent_final_carries..., primal_stacked...,
        // tangent_stacked...]`; zip the matching halves back into `DifferentiationDual`s in the original output order.
        let scanned_output_count = body_output_count - carry_count;
        let mut jvp_outputs = Vec::with_capacity(body_output_count);
        for index in 0..carry_count {
            jvp_outputs.push(DifferentiationDual::new(outputs[index].clone(), outputs[carry_count + index].clone())?);
        }
        for index in 0..scanned_output_count {
            jvp_outputs.push(DifferentiationDual::new(
                outputs[2 * carry_count + index].clone(),
                outputs[2 * carry_count + scanned_output_count + index].clone(),
            )?);
        }
        Ok(jvp_outputs)
    }
}

/// Returns the permutation that converts one side of a fused JVP body's doubled scan signature from JVP order
/// (`[primal_entries..., tangent_entries...]`, each of length `half`) into scan order, where carries lead the
/// scanned entries on both the primal and tangent halves:
/// `[primal_carries..., tangent_carries..., primal_scanned..., tangent_scanned...]`.
fn doubled_scan_signature_permutation(half: usize, carry_count: usize) -> Result<Vec<usize>, ProgramError> {
    if carry_count > half {
        return Err(ProgramError::MalformedProgram(format!(
            "scan carry count {carry_count} exceeds fused body half-signature size {half}",
        )));
    }
    let mut permutation = Vec::with_capacity(2 * half);
    permutation.extend(0..carry_count);
    permutation.extend(half..half + carry_count);
    permutation.extend(carry_count..half);
    permutation.extend(half + carry_count..2 * half);
    Ok(permutation)
}

/// Rebuilds `program` with a new public boundary order. `input_order` and `output_order` list old boundary positions
/// in the desired new order.
fn reorder_program_boundary<V, O>(
    program: Program<V, O, Vec<V>, Vec<V>>,
    input_order: &[usize],
    output_order: &[usize],
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value,
    O: Operation<V::Type>,
{
    fn inverse_order(order: &[usize], length: usize, label: &str) -> Result<Vec<usize>, ProgramError> {
        if order.len() != length {
            return Err(ProgramError::MalformedProgram(format!(
                "{label} permutation has length {} but boundary has length {length}",
                order.len(),
            )));
        }
        let mut inverse = vec![None; length];
        for (new_position, &old_position) in order.iter().enumerate() {
            let Some(slot) = inverse.get_mut(old_position) else {
                return Err(ProgramError::MalformedProgram(format!(
                    "{label} permutation references out-of-range position {old_position}",
                )));
            };
            if slot.is_some() {
                return Err(ProgramError::MalformedProgram(format!(
                    "{label} permutation references position {old_position} more than once",
                )));
            }
            *slot = Some(new_position);
        }
        inverse
            .into_iter()
            .enumerate()
            .map(|(old_position, new_position)| {
                new_position.ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "{label} permutation does not reference position {old_position}",
                    ))
                })
            })
            .collect()
    }

    let input_types = program.input_types();
    let output_count = program.output_count();
    let inverse_input_order = inverse_order(input_order, input_types.len(), "input")?;
    let _ = inverse_order(output_order, output_count, "output")?;
    let reordered_input_types = input_order.iter().map(|&index| input_types[index].clone()).collect::<Vec<_>>();
    let mut builder = ProgramBuilder::new();
    let inputs = reordered_input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
    let original_inputs = inverse_input_order.iter().map(|&new_position| inputs[new_position]).collect::<Vec<_>>();
    let outputs = builder.splice_program(&program, original_inputs.as_slice())?;
    let reordered_outputs = output_order.iter().map(|&index| outputs[index]).collect::<Vec<_>>();
    builder.build(reordered_outputs, vec![Placeholder; input_order.len()], vec![Placeholder; output_order.len()])
}

/// Rebuilds a fused JVP scan body so its doubled boundary uses scan order instead of JVP order.
fn permute_doubled_scan_body<V, O>(
    program: Program<V, O, Vec<V>, Vec<V>>,
    input_half: usize,
    output_half: usize,
    carry_count: usize,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value,
    O: Operation<V::Type>,
{
    let input_order = doubled_scan_signature_permutation(input_half, carry_count)?;
    let output_order = doubled_scan_signature_permutation(output_half, carry_count)?;
    reorder_program_boundary(program, input_order.as_slice(), output_order.as_slice())
}

/// Rebuilds a transposed scan body so it produces one carry cotangent output per carry, inserting typed zero
/// instructions for carries that were known and therefore omitted by transposition.
fn restore_known_carry_outputs<V, O>(
    program: Program<V, O, Vec<V>, Vec<V>>,
    body_output_types: &[ArrayType],
    operand_linear: &[bool],
    carry_count: usize,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>>,
{
    let linear_carry_count = operand_linear[..carry_count].iter().filter(|&&linear| linear).count();
    let input_types = program.input_types();
    let input_count = input_types.len();
    let mut builder = ProgramBuilder::new();
    let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
    let mut outputs = builder.splice_program(&program, inputs.as_slice())?;
    check_count!("output", outputs, program.output_count(), ProgramError);
    let trailing_outputs = outputs.split_off(linear_carry_count);
    let mut linear_carry_outputs = outputs.into_iter();
    let mut restored_outputs = Vec::with_capacity(carry_count + trailing_outputs.len());
    for (carry_index, &carry_is_linear) in operand_linear[..carry_count].iter().enumerate() {
        if carry_is_linear {
            restored_outputs.push(linear_carry_outputs.next().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "scan transpose missing linear carry cotangent output {carry_index}",
                ))
            })?);
        } else {
            // A differentiable carry's cotangent boundary carries its cotangent dual; a non-differentiable carry uses
            // the first-class zero-space type returned by `cotangent`.
            let output_type = &body_output_types[carry_index];
            let cotangent_type = output_type.cotangent();
            let zero = builder.add_instruction(ZeroOperation::new(cotangent_type), Vec::new(), Vec::new())?;
            check_count!("output", zero, 1, ProgramError);
            restored_outputs.push(zero[0]);
        }
    }
    restored_outputs.extend(trailing_outputs);
    let output_count = restored_outputs.len();
    builder.build(restored_outputs, vec![Placeholder; input_count], vec![Placeholder; output_count])
}

/// Maps a stacked scan input's physical batch axis to the corresponding per-iteration batch axis after removing the
/// logical leading scan dimension.
fn scan_iteration_batch_axis(batch_axis: BatchAxis) -> BatchAxis {
    match batch_axis.axis() {
        Some(0) => BatchAxis::new(0),
        Some(axis) => BatchAxis::new(axis - 1),
        None => BatchAxis::replicated(),
    }
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
    let stack_axis = match stack.batch_axis_position() {
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
    ArrayBatch::new(iteration_type, iteration_value, scan_iteration_batch_axis(stack.batch_axis()))
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
            let batch_axis = iteration_y.batch_axis_position();
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
                        });
                    }
                    accumulator
                }
                None => accumulator.insert(ScanOutputAccumulator {
                    // Unlike the scan operation's unbatched signature helper, this physical accumulator must retain
                    // the iteration value's mapped-dimension placement. The newly inserted scan dimension itself is
                    // replicated.
                    accumulator: allocate_zero(&iteration_type.with_inserted_dimension(0, Size::Static(length))?)?,
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
                outputs.push(ArrayBatch::new(
                    stacked_type,
                    accumulator,
                    BatchAxis::from_optional_position(batch_axis.map(|axis| axis + 1)),
                )?);
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

/// Batching rule for [`ScanOperation`]: the scan loop is replayed per iteration through
/// `batch_scan_with_interpreter`, with each body instruction re-entering this operation family's batching rules
/// against the same active context. Constants lift and stacked-output accumulators seed (via the parent's [`Zero`])
/// through `context.parent()`, so an eager parent runs the batched scan operationally while a staging parent stages
/// the per-iteration work into the enclosing trace.
impl<C> BatchableOperation<C> for ScanOperation<C::Constant>
where
    C: Context<Type = ArrayType> + Zero<<C as Domain>::Value>,
    <C as Domain>::Value: Slice + UpdateSlice + Reshape,
    C::Operation:
        From<ZeroOperation<ArrayType>> + From<SliceOperation> + From<UpdateSliceOperation> + From<ReshapeOperation>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        let body = driver.region(0)?;
        if self.length() == 0 {
            check_count!("input", inputs, body.input_types().len(), ProgramError);

            // No iteration executes, but batching the body structurally still determines which per-iteration outputs
            // are mapped and where their physical batch dimensions live. Stacked inputs lose their logical leading
            // scan dimension before entering the body, so their batch axes must be adjusted in the same way as an
            // actual iteration slice.
            let mut iteration_input_axes =
                inputs[..self.carry_count()].iter().map(ArrayBatch::batch_axis).collect::<Vec<_>>();
            iteration_input_axes
                .extend(inputs[self.carry_count()..].iter().map(|input| scan_iteration_batch_axis(input.batch_axis())));
            let (batched_body, output_axes) = driver.batch_program(
                context,
                body,
                iteration_input_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::Natural,
            )?;
            let output_types = batched_body.output_types();
            check_count!("output", output_axes, output_types.len(), ProgramError);
            if output_types.len() < self.carry_count() {
                return Err(ProgramError::MalformedProgram(format!(
                    "scan body has {} outputs but carry count is {}",
                    output_types.len(),
                    self.carry_count(),
                ))
                .into());
            }

            // A zero-length scan returns its initial carries unchanged. Its stacked outputs are empty arrays whose
            // physical element types and batch axes come from the structurally batched body. Inserting the leading
            // scan dimension shifts every mapped output axis right by one while preserving its placement metadata.
            let mut outputs = inputs[..self.carry_count()].to_vec();
            for (output_type, output_axis) in
                output_types.into_iter().zip(output_axes.into_iter()).skip(self.carry_count())
            {
                let stacked_type = output_type.with_inserted_dimension(0, Size::Static(0))?;
                let stacked_axis = match output_axis.axis() {
                    Some(axis) => BatchAxis::new(axis + 1),
                    None => BatchAxis::replicated(),
                };
                let stacked_value = context.parent().zero(&stacked_type)?;
                outputs.push(ArrayBatch::new(stacked_type, stacked_value, stacked_axis)?);
            }
            return Ok(outputs);
        }

        let y_slice_types = body.output_types().split_off(self.carry_count());
        batch_scan_with_interpreter(
            self.carry_count(),
            self.length(),
            self.reverse(),
            y_slice_types.as_slice(),
            inputs,
            |stacked_type| context.parent().zero(stacked_type),
            |_, iteration_inputs| driver.batch_region(context, 0, iteration_inputs),
        )
    }
}

/// Type-family transposition semantics for [`ScanOperation`], with the scan's value, body-operation, capture,
/// payload, and staging-target parameters riding as trait inputs and the type family as the implementing type
/// (mirroring [`ScanPayload`](crate::operations::control_flow::scan::ScanPayload)) so that the [`ArrayType`] and
/// [`DataType`] rules stay coherent without the operation struct naming its type family as a parameter. The
/// [`ArrayType`] rule pins the staging target to the scan's own body operation family `O`, while the [`DataType`]
/// rule keeps an independent `Target` because a scalar linear scan never inlines its body into the pullback.
pub(crate) trait ScanTransposition<V, F, Target>: Type
where
    V: Value<Type = Self>,
    F: Value<Type = Self>,
    Target: Operation<Self>,
{
    /// Applies the type family's `scan` transpose rule using the loop's driver; refer to the documentation of
    /// [`TransposableOperation::transpose`] for the contract.
    fn transpose_scan<D: TranspositionDriver<V, Target>>(
        operation: &ScanOperation<F>,
        context: &mut TracingContext<V, Target>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, DifferentiationError>;
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
/// the instruction-scoped driver's transposition requests, keeping the scan-local recursion owned by the operation
/// family with no recursive [`TransposableOperation`] obligation on `O`.
impl<V, F, Target> ScanTransposition<V, F, Target> for ArrayType
where
    V: Value<Type = ArrayType>,
    F: Value<Type = ArrayType>,
    Target: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<ScanOperation<F>>,
{
    fn transpose_scan<D: TranspositionDriver<V, Target>>(
        operation: &ScanOperation<F>,
        context: &mut TracingContext<V, Target>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, DifferentiationError> {
        // The rule requests all nested-computation work through its driver (region 0 is the body), which keeps
        // its bounds free of the operation family's own semantic traits.
        //
        // A scan with only zero output cotangents is a zero linear map, so every input cotangent is zero.
        if outputs.iter().all(MaybeZero::is_zero) {
            return Ok(inputs
                .iter()
                .map(|input| {
                    let input_type = input.r#type();
                    MaybeZero::Zero(input_type.cotangent())
                })
                .collect());
        }
        if operation.captures().is_empty() {
            return transpose_primal_scan(operation, context, driver, inputs, outputs)
                .map_err(DifferentiationError::from);
        }
        let body = driver.region(0)?;
        let carry_count = operation.carry_count();
        let length = operation.length();
        let transposed_body = driver.transpose_program(body, &vec![true; body.input_ids().len()])?;
        let transposed = ScanOperation::<F>::new(carry_count, length)
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
        let cotangents =
            context.stage_operation(Target::from(transposed), vec![transposed_body], materialized.as_slice())?;
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
///   1. Transposes the body through its instruction-scoped driver under each
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
/// instruction-scoped driver's transposition requests in the same operation family, so it introduces no recursive
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
pub fn transpose_primal_scan<V, O, F, D: TranspositionDriver<V, O>>(
    operation: &ScanOperation<F>,
    context: &mut TracingContext<V, O>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError>
where
    V: Value<Type = ArrayType>,
    F: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<ScanOperation<F>>,
{
    // A scan with only zero output cotangents is a zero linear map, so every operand cotangent is zero.
    if outputs.iter().all(MaybeZero::is_zero) {
        return Ok(inputs
            .iter()
            .map(|input| {
                let input_type = input.r#type();
                MaybeZero::Zero(input_type.cotangent())
            })
            .collect());
    }

    // Operand layout is `[carries..., scanned_inputs...]`, mirroring the body's input order one-to-one, where each
    // operand is independently linear (a tangent the reverse must accumulate) or known (a residual stack the pullback
    // reads). Linear operands need not form a leading run: vmapping a bounded `while` threads a non-differentiable
    // Boolean mask as a known *carry*, so a known operand can sit among the linear carries. The leading `carry_count`
    // operands are the carries and the rest are scanned inputs.
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let body = driver.region(0)?;
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
    let mut transposed_body =
        driver.transpose_program(body, operand_linear.as_slice()).map_err(|error| match error {
            crate::differentiation::DifferentiationError::Program(error) => error,
            error => ProgramError::UnsupportedOperation { message: error.to_string() },
        })?;

    // `transpose_with_respect_to` emits one cotangent output per linear input, so a known carry contributes no carry
    // output and the transposed body has fewer carry outputs than the reversed scan's `carry_count` requires. Restore
    // the carry-output arity exactly as the split restores pruned tangent outputs: walk the carry positions, taking
    // the next linear-carry cotangent where the carry is linear and inserting a fresh structural zero of that carry
    // output's cotangent type where it is known. The trailing transposed outputs (the linear scanned-input cotangents)
    // are carried over unchanged, so the reversed body produces `[carry_cotangent..., scanned_input_cotangent...]`.
    let linear_carry_count = operand_linear[..carry_count].iter().filter(|&&linear| linear).count();
    if linear_carry_count != carry_count {
        transposed_body = restore_known_carry_outputs(
            transposed_body,
            body.output_types().as_slice(),
            operand_linear.as_slice(),
            carry_count,
        )?;
    }

    let transposed = ScanOperation::<F>::new(carry_count, length)
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
    let scan_cotangents = context.stage_operation(O::from(transposed), vec![transposed_body], operands.as_slice())?;
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
                let input_type = input.r#type();
                MaybeZero::Zero(input_type.cotangent())
            }
        })
        .collect();
    Ok(cotangents)
}

impl<V, F, Target> ScanTransposition<V, F, Target> for DataType
where
    V: Value<Type = DataType>,
    F: Value<Type = DataType>,
    Target: Operation<DataType> + From<ZeroOperation<DataType>> + From<ScanOperation<F>>,
{
    fn transpose_scan<D: TranspositionDriver<V, Target>>(
        operation: &ScanOperation<F>,
        context: &mut TracingContext<V, Target>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, DifferentiationError> {
        if outputs.iter().all(MaybeZero::is_zero) {
            return Ok(inputs
                .iter()
                .map(|input| {
                    let input_type = input.r#type();
                    MaybeZero::Zero(input_type.cotangent())
                })
                .collect());
        }
        if !operation.captures().is_empty() {
            return Err(ProgramError::UnsupportedOperation {
                message: "scalar linear scan transposition with residual stacks requires a scalar stack representation"
                    .to_string(),
            }
            .into());
        }
        let body = driver.region(0)?;
        let output_types = body.output_types();
        check_count!("output", outputs, output_types.len(), ProgramError);
        let transposed_body = driver.transpose_program(body, &vec![true; body.input_ids().len()])?;
        let transposed = ScanOperation::<F>::new(operation.carry_count(), operation.length())
            .with_reverse(!operation.reverse())
            .with_unroll(operation.unroll())?
            .with_captures(operation.captures().to_vec());
        let materialized = outputs
            .iter()
            .map(|cotangent| cotangent.clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        let cotangents =
            context.stage_operation(Target::from(transposed), vec![transposed_body], materialized.as_slice())?;
        check_count!("output", cotangents, inputs.len(), ProgramError);
        Ok(cotangents.into_iter().map(MaybeZero::Value).collect())
    }
}

/// Transpose rule for [`ScanOperation`], dispatching to the scan's type family through the crate-private
/// `ScanTransposition` trait: array
/// scans transpose captured linear scans whole and forward operand-form primal scans to [`transpose_primal_scan`],
/// and scalar scans transpose capture-free carry-only linear scans.
impl<V, F, Target> TransposableOperation<V, Target> for ScanOperation<F>
where
    V: Value,
    V::Type: ScanTypeSemantics + ScanTransposition<V, F, Target>,
    F: Value<Type = V::Type>,
    Target: Operation<V::Type>,
{
    fn transpose<D: TranspositionDriver<V, Target>>(
        &self,
        context: &mut TracingContext<V, Target>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, DifferentiationError> {
        <V::Type>::transpose_scan(self, context, driver, inputs, outputs)
    }
}

#[cfg(test)]
mod tests {
    use crate::batching::BatchingContext;
    use pretty_assertions::assert_eq;

    use crate::batching::{BatchAxis, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::differentiation::ForwardModeDifferentiate;
    use crate::operations::math::MulOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::DataType;

    use crate::tracing::Trace;

    use super::*;

    type TestOperation = ArrayOperation<TestArray>;
    type TestEagerContext = EagerContext<TestArray, TestOperation>;
    type TestScanOperation = ScanOperation<TestArray>;

    /// Builds a cumulative-product body program that maps `[carry, x]` to `[carry * x, carry * x]`.
    fn product_body() -> Program<TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>> {
        product_body_with_type(ArrayType::scalar(DataType::F64))
    }

    /// Builds a cumulative-product body over `r#type` that maps `[carry, x]` to
    /// `[carry * x, carry * x]`.
    fn product_body_with_type(r#type: ArrayType) -> Program<TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<TestArray, TestOperation>::new();
        let carry = builder.add_input(r#type.clone());
        let x = builder.add_input(r#type);
        let product = builder.add_instruction(MulOperation, Vec::new(), vec![carry, x]).unwrap()[0];
        builder
            .build(vec![product, product], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds a body for zero-length scan tests whose first stacked result follows the carry's mapped axis and whose
    /// second stacked result is a replicated constant.
    fn zero_length_body(r#type: ArrayType) -> Program<TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<TestArray, TestOperation>::new();
        let carry = builder.add_input(r#type.clone());
        let _x = builder.add_input(r#type.clone());
        let constant = builder.add_constant(TestArray::new(r#type, vec![7.0]));
        builder
            .build(
                vec![carry, carry, constant],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder, Placeholder],
            )
            .unwrap()
    }

    /// Builds the `f64` array type with the provided static dimensions.
    fn f64_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(dimensions.iter().copied().map(Size::Static).collect()))
    }

    /// Builds a cumulative-product [`ScanOperation`] whose nested depth matches `lengths`, returning the
    /// payload-free operation together with its body region program.
    fn product_scan_with_lengths(
        lengths: &[usize],
    ) -> (ScanOperation<TestArray>, Program<TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>>) {
        assert!(!lengths.is_empty());
        if lengths.len() == 1 {
            return (TestScanOperation::new(1, lengths[0]), product_body());
        }
        let (inner_scan, inner_body) = product_scan_with_lengths(&lengths[1..]);
        let mut builder = ProgramBuilder::<TestArray, TestOperation>::new();
        let inner_body_region = builder.import_region(inner_body.entry_region_ref());
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let xs = builder.add_input(f64_type(&lengths[1..]));
        let outputs = builder
            .add_instruction(TestOperation::Scan(inner_scan), vec![inner_body_region], vec![carry, xs])
            .unwrap()
            .to_vec();
        let body = builder.build(outputs, vec![Placeholder, Placeholder], vec![Placeholder, Placeholder]).unwrap();
        (TestScanOperation::new(1, lengths[0]), body)
    }

    /// Builds the cumulative-product [`ScanOperation`] over three iterations used by the differentiation tests,
    /// returning the payload-free operation together with its body region program.
    fn product_scan() -> (ScanOperation<TestArray>, Program<TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>>) {
        product_scan_with_lengths(&[3])
    }

    /// Batches `scan` through the public [`BatchingContext::bind`] path with `body` as an owned attached region.
    fn batch_scan(
        context: &BatchingContext<TestEagerContext>,
        scan: ScanOperation<TestArray>,
        body: Program<TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>>,
        inputs: Vec<ArrayBatch<TestArray>>,
    ) -> Vec<ArrayBatch<TestArray>> {
        let tracer_inputs =
            inputs.into_iter().map(|input| BatchingTracer::new(context.clone(), input)).collect::<Vec<_>>();
        context
            .bind(TestOperation::Scan(scan), [body], tracer_inputs.as_slice())
            .unwrap()
            .into_iter()
            .map(|output| output.batch().clone())
            .collect()
    }

    #[test]
    fn test_reorder_program_boundary_supports_nullary_programs() {
        let mut builder = ProgramBuilder::<TestArray, TestOperation>::new();
        let first = builder.add_constant(TestArray::scalar(1.0));
        let second = builder.add_constant(TestArray::scalar(2.0));
        let program = builder.build(vec![first, second], Vec::new(), vec![Placeholder, Placeholder]).unwrap();

        let reordered = reorder_program_boundary(program, &[], &[1, 0]).unwrap();

        assert_eq!(reordered.input_count(), 0);
        assert_eq!(
            reordered.outputs().map(|output| output.as_constant().unwrap().values.clone()).collect::<Vec<_>>(),
            vec![vec![2.0], vec![1.0]],
        );
    }

    /// The fused JVP rule stages exactly one scan with doubled carries and **no** per-iteration residual stacks:
    /// pure forward mode pays a single loop pass and no reverse-mode storage. Residual stacks appear only when
    /// [`Program::linearize`] directly differentiates over partial evaluation (its known scan then stacks the
    /// known→unknown edges), which the trailing assertion pins.
    #[test]
    fn test_scan_jvp_stages_one_fused_scan_with_no_residual_stacks() {
        use crate::tracing::DomainTracer;
        use crate::types::{Shape, Size};

        let (scan, scan_body) = product_scan();
        let (_, program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
            |(init, xs): (
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
            )| {
                let mut outputs = init.context().stage_operation(
                    TestOperation::Scan(scan),
                    vec![scan_body.clone()],
                    &[&init, &xs],
                )?;
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
                TestOperation::Scan(operation) => Some((operation, instruction)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(scans.len(), 1);
        let (fused_scan, fused_instruction) = scans[0];
        assert_eq!(fused_scan.carry_count(), 2);
        // The fused body is `[primal_carry, tangent_carry, primal_x, tangent_x] ->
        // [primal_carry', tangent_carry', primal_y, tangent_y]`: doubled arity and nothing else.
        let fused_body = jvp.region_ref(fused_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(fused_body.input_types().len(), 4);
        assert_eq!(fused_body.output_types().len(), 4);

        // Linearizing the same program is what materializes residual stacks, as known-scan edges.
        let linearization = program.linearize().unwrap();
        assert!(linearization.residual_count() >= 1);
    }

    #[test]
    fn test_scan_jvp_propagates_tangents_through_linear_scan() {
        // Cumulative product over `xs = [2, 3, 4]` starting at `init = 1`: the final carry is 24 and the running
        // products are `[2, 6, 24]`. A unit tangent on `init` propagates as `d(init * x0 * x1 * x2)/d(init) = 24`
        // on the final carry and `[2, 6, 24]` on the stacked outputs.
        let (scan, scan_body) = product_scan();
        let ((carry, ys), (carry_tangent, ys_tangent)) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                move |(init, xs)| {
                    let mut outputs = init.context().bind(
                        TestOperation::Scan(scan),
                        vec![scan_body.clone()],
                        &[init.clone(), xs.clone()],
                    )?;
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
        let (scan, scan_body) = product_scan();
        let ((carry, _), (carry_tangent, ys_tangent)) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                move |(init, xs)| {
                    let mut outputs = init.context().bind(
                        TestOperation::Scan(scan),
                        vec![scan_body.clone()],
                        &[init.clone(), xs.clone()],
                    )?;
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
        let (scan, scan_body) = product_scan_with_lengths(&[2, 3]);
        let ((carry, ys), (carry_tangent, ys_tangent)) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                move |(init, xs)| {
                    let mut outputs = init.context().bind(
                        TestOperation::Scan(scan),
                        vec![scan_body.clone()],
                        &[init.clone(), xs.clone()],
                    )?;
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
        let (scan, scan_body) = product_scan_with_lengths(&[2, 2, 2]);
        let xs_type = f64_type(&[2, 2, 2]);
        let ((carry, ys), (carry_tangent, ys_tangent)) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                move |(init, xs)| {
                    let mut outputs = init.context().bind(
                        TestOperation::Scan(scan),
                        vec![scan_body.clone()],
                        &[init.clone(), xs.clone()],
                    )?;
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
        let (scan, scan_body) = product_scan();
        let context = BatchingContext::new(TestEagerContext::new(), 3);
        let carries = {
            let value = TestArray::vector(vec![1.0, 2.0, 3.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let stacked_inputs = ArrayBatch::replicated(TestArray::vector(vec![2.0, 3.0, 4.0]));
        let outputs = batch_scan(&context, scan, scan_body, vec![carries, stacked_inputs]);
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
        let (scan, scan_body) = product_scan();
        let context = BatchingContext::new(TestEagerContext::new(), 2);
        let carries = ArrayBatch::replicated(TestArray::scalar(1.0));
        let stacked_inputs = {
            let value = TestArray::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = batch_scan(&context, scan, scan_body, vec![carries, stacked_inputs]);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![24.0, 210.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].r#type().shape().dimensions(), &[Size::Static(3), Size::Static(2)]);
        assert_eq!(outputs[1].value().values, vec![2.0, 5.0, 6.0, 30.0, 24.0, 210.0]);

        // A trailing batch axis (physical `[3, 2]` with the batch axis at 1) reads the same logical iterations, so the
        // outputs are identical.
        let (scan, scan_body) = product_scan();
        let context = BatchingContext::new(TestEagerContext::new(), 2);
        let carries = ArrayBatch::replicated(TestArray::scalar(1.0));
        let stacked_inputs = {
            let value = TestArray::matrix(3, 2, vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(1))
        }
        .unwrap();
        let outputs = batch_scan(&context, scan, scan_body, vec![carries, stacked_inputs]);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![24.0, 210.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].value().values, vec![2.0, 5.0, 6.0, 30.0, 24.0, 210.0]);
    }

    #[test]
    fn test_scan_batching_threads_batched_carries_and_inputs() {
        // Batching both operands pairs batch item `i` of the carries with batch item `i` of the stacked inputs.
        let (scan, scan_body) = product_scan();
        let context = BatchingContext::new(TestEagerContext::new(), 2);
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
        let outputs = batch_scan(&context, scan, scan_body, vec![carries, stacked_inputs]);
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
        let (scan, scan_body) = product_scan();
        let scan = scan.with_reverse(true);
        let context = BatchingContext::new(TestEagerContext::new(), 2);
        let carries = ArrayBatch::replicated(TestArray::scalar(1.0));
        let stacked_inputs = {
            let value = TestArray::matrix(2, 3, vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = batch_scan(&context, scan, scan_body, vec![carries, stacked_inputs]);
        assert_eq!(outputs[0].value().values, vec![24.0, 24.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].value().values, vec![24.0, 24.0, 12.0, 12.0, 4.0, 4.0]);
    }

    #[test]
    fn test_scan_batching_preserves_stacked_output_batch_placement() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let logical_type =
                ArrayType::scalar(DataType::F64).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
            let carry_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                .unwrap()
                .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                .unwrap();
            let carry_type = f64_type(&[2]).with_sharding(carry_sharding.clone()).unwrap();
            let carries =
                ArrayBatch::new(carry_type.clone(), TestArray::new(carry_type, vec![1.0, 2.0]), BatchAxis::new(0))
                    .unwrap();
            let stack_type = f64_type(&[3]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
            let stacked_inputs = ArrayBatch::replicated(TestArray::new(stack_type, vec![2.0, 3.0, 4.0]));
            let context =
                BatchingContext::new(TestEagerContext::new(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));

            let outputs = batch_scan(
                &context,
                TestScanOperation::new(1, 3),
                product_body_with_type(logical_type),
                vec![carries, stacked_inputs],
            );

            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type().sharding().unwrap().dimensions(), carry_sharding.dimensions());
            assert_eq!(outputs[0].value().values, vec![24.0, 48.0]);
            assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
            assert_eq!(outputs[1].r#type().shape().dimensions(), &[Size::Static(3), Size::Static(2)]);
            assert_eq!(
                outputs[1].r#type().sharding().unwrap().dimensions(),
                &[ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
            );
            assert_eq!(outputs[1].value().values, vec![2.0, 4.0, 6.0, 12.0, 24.0, 48.0]);
        }
    }

    #[test]
    fn test_zero_length_scan_batching_infers_mapped_and_replicated_outputs_eagerly() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let logical_type =
                ArrayType::scalar(DataType::F64).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
            let carry_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                .unwrap()
                .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                .unwrap();
            let carry_type = f64_type(&[2]).with_sharding(carry_sharding.clone()).unwrap();
            let carries =
                ArrayBatch::new(carry_type.clone(), TestArray::new(carry_type, vec![1.0, 2.0]), BatchAxis::new(0))
                    .unwrap();
            let stack_type = f64_type(&[0]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
            let stacked_inputs = ArrayBatch::replicated(TestArray::new(stack_type, Vec::new()));
            let context =
                BatchingContext::new(TestEagerContext::new(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));

            let outputs = batch_scan(
                &context,
                TestScanOperation::new(1, 0),
                zero_length_body(logical_type),
                vec![carries, stacked_inputs],
            );

            assert_eq!(outputs.len(), 3);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type().shape().dimensions(), &[Size::Static(2)]);
            assert_eq!(outputs[0].r#type().sharding().unwrap().dimensions(), carry_sharding.dimensions());
            assert_eq!(outputs[0].value().values, vec![1.0, 2.0]);
            assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
            assert_eq!(outputs[1].r#type().shape().dimensions(), &[Size::Static(0), Size::Static(2)]);
            assert_eq!(
                outputs[1].r#type().sharding().unwrap().dimensions(),
                &[ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
            );
            assert!(outputs[1].value().values.is_empty());
            assert_eq!(outputs[2].batch_axis(), BatchAxis::replicated());
            assert_eq!(outputs[2].r#type().shape().dimensions(), &[Size::Static(0)]);
            assert_eq!(outputs[2].r#type().sharding().unwrap().dimensions(), &[ShardingDimension::replicated()],);
            assert!(outputs[2].value().values.is_empty());
        }
    }

    #[test]
    fn test_zero_length_scan_batching_infers_mapped_and_replicated_outputs_while_tracing() {
        use std::rc::Rc;

        use crate::tracing::TracingContext;

        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let logical_type =
                ArrayType::scalar(DataType::F64).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
            let carry_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                .unwrap()
                .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                .unwrap();
            let carry_type = f64_type(&[2]).with_sharding(carry_sharding.clone()).unwrap();
            let stack_type = f64_type(&[0]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
            let parent = TracingContext::<TestArray, TestOperation>::new();
            let builder = parent.builder().clone();
            let carry_atom = builder.borrow_mut().add_input(carry_type.clone());
            let stack_atom = builder.borrow_mut().add_input(stack_type.clone());
            let context = BatchingContext::new(parent.clone(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));
            let carries = ArrayBatch::new(carry_type, parent.tracer(carry_atom, None), BatchAxis::new(0)).unwrap();
            let stacked_inputs = ArrayBatch::replicated(parent.tracer(stack_atom, None));
            let tracer_inputs =
                [BatchingTracer::new(context.clone(), carries), BatchingTracer::new(context.clone(), stacked_inputs)];
            let outputs = context
                .bind(
                    TestOperation::Scan(TestScanOperation::new(1, 0)),
                    [zero_length_body(logical_type)],
                    &tracer_inputs,
                )
                .unwrap();
            let output_axes = outputs.iter().map(BatchingTracer::batch_axis).collect::<Vec<_>>();
            let output_atoms =
                outputs.iter().map(|output| output.batch().value().atom_id().unwrap()).collect::<Vec<_>>();
            drop(outputs);
            drop(tracer_inputs);
            drop(context);
            drop(parent);

            let builder = Rc::try_unwrap(builder).expect("batching should not retain the tracing builder").into_inner();
            let program = builder
                .build::<Vec<TestArray>, Vec<TestArray>>(
                    output_atoms,
                    vec![Placeholder, Placeholder],
                    vec![Placeholder, Placeholder, Placeholder],
                )
                .unwrap();
            let output_types = program.output_types();

            assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(1), BatchAxis::replicated()]);
            assert_eq!(output_types[0].shape().dimensions(), &[Size::Static(2)]);
            assert_eq!(output_types[0].sharding().unwrap().dimensions(), carry_sharding.dimensions());
            assert_eq!(output_types[1].shape().dimensions(), &[Size::Static(0), Size::Static(2)]);
            assert_eq!(
                output_types[1].sharding().unwrap().dimensions(),
                &[ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
            );
            assert_eq!(output_types[2].shape().dimensions(), &[Size::Static(0)]);
            assert_eq!(output_types[2].sharding().unwrap().dimensions(), &[ShardingDimension::replicated()],);
        }
    }
}
