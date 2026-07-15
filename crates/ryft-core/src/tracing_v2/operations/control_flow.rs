use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingError, ProgramBatchingOutputAxesPolicy};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationError, TransposableOperation,
};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::constants::{One, OneOperation, Zero, ZeroOperation};
use crate::operations::control_flow::scan::stacked_scan_type;
use crate::operations::control_flow::{
    ConditionOperation, ScanOperation, Select, SelectOperation, WhileOperation, WhileTypeSemantics,
};
use crate::operations::logical::AndOperation;
use crate::operations::manipulation::{
    Broadcast, BroadcastOperation, DynamicUpdateSlice, DynamicUpdateSliceOperation, Transpose, TransposeOperation,
};
use crate::operations::math::{Add, AddOperation};
use crate::operations::{BooleanLike, Operation};
use crate::parameters::Placeholder;
use crate::partial::PartialValue;
use crate::payloads::{Captured, Input};
use crate::programs::{MaybeZero, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::batching::{BatchingContext, BatchingDriver};
use crate::differentiation::{DifferentiationDriver, DifferentiationDual, TranspositionDriver};
use crate::interpretation::InterpretationDriver;
use crate::tracing_v2::operations::custom_derivatives::CustomVjpResidual;
use crate::tracing_v2::operations::reduce::{Reduce, ReduceOperation, ReductionKind};
use crate::tracing_v2::unroll::unroll_concretizable_whiles;
use crate::types::{ArrayType, DataType, TypeError, Typed};

impl<V: Value<Type = ArrayType> + BooleanLike> BooleanLike for ArrayBatch<V> {
    /// Returns an [`ArrayBatch`] that wraps the Boolean reinterpretation of the carried value (via the value's own
    /// [`BooleanLike::as_boolean`]) under the same batch axis.
    fn as_boolean(&self) -> Self {
        // This unwrap is safe because `as_boolean` preserves structural metadata, so the batch axis that was valid
        // for this batch remains in bounds for the reinterpreted value.
        let value = self.value().as_boolean();
        Self::new(value.r#type().into_owned(), value, self.batch_axis()).unwrap()
    }

    fn boolean(&self) -> Result<bool, ProgramError> {
        if let Some(axis) = self.batch_axis().axis() {
            return Err(ProgramError::Concretization {
                message: format!("cannot extract a concrete boolean from a value batched along axis {axis}"),
            });
        }
        self.value().boolean()
    }
}

/// Partition-aware transpose rule for a *primal* input-predicate [`ConditionOperation`], used when the direct
/// reverse transposes a tangent program in the primal operation family `O` rather than re-keying it into the linear
/// family. This is the operand-form counterpart of the captured-predicate
/// [`ConditionOperation`](ConditionOperation::new_captured) transpose rule: the predicate and the per-branch residuals
/// are ordinary *operands* (known values supplied through `operand_values`) instead of captures, so the rule reads them
/// from the pullback and threads them back through as known operands of a transposed condition.
///
/// The forward stages the tangent condition over `[predicate, branch_tangents..., residuals...]` with the
/// predicate and the joined residual set marked known and the branch tangents marked linear, and with both branches
/// already joined to the same input signature `[branch_tangents..., residuals...]` and output signature
/// `[branch_tangent_outputs...]`. This rule therefore:
///
///   1. Splits the operands by `operand_linear` into the known predicate (operand `0`), the leading linear run of
///      branch tangents, and the trailing known residuals.
///   2. Transposes each branch through the driver's region-transposition request, marking the branch tangent inputs
///      linear and the residual inputs known. Each transposed branch maps
///      `[branch_tangent_output_cotangents..., residuals...]` to `[branch_tangent_input_cotangents...]`; because both
///      branches shared the joined signature, their transposes share it too and form a well-typed condition.
///   3. Re-stages a primal input-predicate [`ConditionOperation`] selecting between the two transposed branches by the
///      same known predicate, over `[predicate, outputs..., residuals...]`. Its outputs are the branch-tangent
///      input cotangents.
///
/// The returned cotangents place those branch-tangent cotangents at the linear-operand positions and a structural
/// [`MaybeZero::Zero`] at the predicate and residual positions, which carry no cotangent. The branch recursion happens
/// through the instruction-scoped driver in the same operation family, so it introduces no
/// recursive [`TransposableOperation`] obligation on `O`.
///
/// # Parameters
///
///   - `operation`: Primal input-predicate condition staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `inputs`: Per-operand [`PartialValue`] knowledge. The [`Unknown`](PartialValue::Unknown) entries are the branch
///     tangents; the [`Known`](PartialValue::Known) entries carry the predicate and residual tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the condition's outputs.
pub fn transpose_primal_condition<V, O, D: TranspositionDriver<V, O>>(
    context: &mut TracingContext<V, O>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError>
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<ConditionOperation<V>>,
{
    // A condition with no live output cotangents is a zero linear map, so every operand cotangent is zero.
    if outputs.iter().all(MaybeZero::is_zero) {
        return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect());
    }

    // The rule operates on the attached branch regions through its driver (region 0 is the `true` branch and region 1
    // the `false` branch), which keeps its bounds free of the operation family's own semantic traits.
    let true_branch = driver.region(0)?;

    // Operand layout is `[predicate(known), branch_tangents(linear)..., residuals(known)...]`. The branch tangents are
    // exactly the linear operands, and the residuals are the trailing known operands after the predicate and tangents.
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let branch_input_count = true_branch.input_types().len();
    let branch_tangent_count = operand_linear.iter().filter(|&&linear| linear).count();
    let residual_count = branch_input_count.checked_sub(branch_tangent_count).ok_or_else(|| {
        ProgramError::MalformedProgram(format!(
            "condition transpose found {branch_tangent_count} linear operands but its branches take only \
             {branch_input_count} inputs",
        ))
    })?;
    check_count!("input", operand_linear, 1 + branch_tangent_count + residual_count, ProgramError);

    // The predicate is operand `0` and the residuals are the trailing operands; both are known values read from the
    // pullback. The dispatch guarantees a `Known` operand carries its pullback value, so each tracer is read directly.
    let read_known = |index: usize| -> Result<Tracer<TracingContext<V, O>>, ProgramError> {
        inputs[index]
            .as_known()
            .ok_or_else(|| {
                ProgramError::MalformedProgram(format!("condition transpose operand {index} has no known value"))
            })
            .cloned()
    };
    let predicate = read_known(0)?;
    let residuals = (1 + branch_tangent_count..inputs.len()).map(read_known).collect::<Result<Vec<_>, _>>()?;

    // Transpose each branch with the branch tangents marked linear and the residual inputs marked known. Each
    // transposed branch maps `[branch_output_cotangents..., residuals...]` to `[branch_tangent_cotangents...]`.
    let mut branch_linear = vec![true; branch_tangent_count];
    branch_linear.extend(std::iter::repeat(false).take(residual_count));
    let transposed_true = driver.transpose_program(driver.region(0)?, branch_linear.as_slice())?;
    let transposed_false = driver.transpose_program(driver.region(1)?, branch_linear.as_slice())?;
    let transposed_condition = ConditionOperation::new();

    // Stage the transposed condition over `[predicate, outputs..., residuals...]`. Its outputs are the
    // branch-tangent input cotangents.
    let output_types = true_branch.output_types();
    check_count!("output", outputs, output_types.len(), ProgramError);
    let mut operands = Vec::with_capacity(1 + output_types.len() + residuals.len());
    operands.push(predicate);
    for cotangent in outputs {
        operands.push(cotangent.clone().materialize(context)?);
    }
    operands.extend(residuals);
    let branch_cotangents =
        context.bind(O::from(transposed_condition), vec![transposed_true, transposed_false], operands.as_slice())?;
    check_count!("output", branch_cotangents, branch_tangent_count, ProgramError);

    // Reassemble one cotangent per operand: the predicate and residuals carry structural zeros, while the branch
    // tangents receive the transposed condition's outputs in order.
    let mut branch_cotangents = branch_cotangents.into_iter().map(MaybeZero::Value);
    let cotangents = operand_linear
        .iter()
        .zip(inputs)
        .map(
            |(&linear, input)| {
                if linear { branch_cotangents.next().unwrap() } else { MaybeZero::Zero(input.r#type().into_owned()) }
            },
        )
        .collect();
    Ok(cotangents)
}

/// Builds the augmented condition and body programs of the bounded staged while loop (see the [`WhileOperation`] JVP
/// rule below). The body replays the original body through its tracing context before adding the augmented-state
/// operations, while the condition structurally relocates the original condition into a builder with the extended
/// input boundary.
///
/// The augmented loop state is `[original_state..., counter (i64 scalar), residual_stacks..., mask_stack]`:
///
///   - The body runs the residual-extended primal body (which outputs `[next_state..., residuals...]`) on the
///     original state slots, then *stores* instead of returning each per-iteration residual: residual `k` is
///     broadcast to `[1, …]` and written into stack `k` at batch index `counter` via `dynamic_update_slice`, a scalar
///     Boolean `one` (true) is written into the Boolean `[bound]` mask stack at batch index `counter`, and the counter
///     advances by an i64 `one`. Because the enclosing while keeps `iteration_bound = bound`, the counter is always
///     strictly below `bound` whenever the body runs, so the writes can never clamp.
///   - The condition is the original loop condition extended with ignored extra-state inputs.
///
/// Returns the extended condition, the augmented body, and the `[bound, …]` residual stack types.
fn build_bounded_while_programs<V, O>(
    condition: &Program<V, O, Vec<V>, Vec<V>>,
    primal_body: &Program<V, O, Vec<V>, Vec<V>>,
    residual_types: &[ArrayType],
    bound: usize,
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Program<V, O, Vec<V>, Vec<V>>, Vec<ArrayType>), ProgramError>
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType>
        + From<ZeroOperation<ArrayType>>
        + From<OneOperation<ArrayType>>
        + From<AddOperation>
        + From<BroadcastOperation>
        + From<DynamicUpdateSliceOperation>,
{
    let state_count = condition.input_types().len();
    let counter_type = ArrayType::scalar(DataType::I64);
    let boolean_scalar_type = ArrayType::scalar(DataType::Boolean);
    let mask_stack_type = stacked_scan_type(&boolean_scalar_type, bound);
    for residual_type in residual_types {
        if residual_type.static_shape().is_none() {
            return Err(TypeError {
                message: format!(
                    "jvp of a bounded while loop requires statically shaped body residuals but got {residual_type}",
                ),
            }
            .into());
        }
    }
    let stack_types = residual_types
        .iter()
        .map(|residual_type| stacked_scan_type(residual_type, bound))
        .collect::<Vec<_>>();

    let body_input_types = primal_body
        .input_types()
        .into_iter()
        .chain(std::iter::once(counter_type.clone()))
        .chain(stack_types.iter().cloned())
        .chain(std::iter::once(mask_stack_type.clone()))
        .collect::<Vec<_>>();
    let body = TracingContext::<V, O>::trace(
        |inputs| {
            let context = inputs[0].context().clone();
            let original_input_count = primal_body.input_ids().len();
            let mut extra_inputs = inputs[original_input_count..].iter();
            let counter_input = extra_inputs.next().cloned().ok_or_else(|| {
                ProgramError::MalformedProgram("bounded while body adapter is missing the counter input".to_string())
            })?;
            let stack_inputs = extra_inputs.by_ref().take(stack_types.len()).cloned().collect::<Vec<_>>();
            check_count!("input", stack_inputs, stack_types.len(), ProgramError);
            let mask_input = extra_inputs.next().cloned().ok_or_else(|| {
                ProgramError::MalformedProgram("bounded while body adapter is missing the mask input".to_string())
            })?;
            let mut body_outputs =
                primal_body.interpret_in_context(&context, inputs[..original_input_count].to_vec())?;
            let residual_outputs = body_outputs.split_off(state_count);
            check_count!("output", residual_outputs, residual_types.len(), ProgramError);
            let zero_index = if residual_types.iter().any(|residual_type| residual_type.rank() > 0) {
                Some(context.zero(&counter_type)?)
            } else {
                None
            };
            let mut next_stacks = Vec::with_capacity(stack_types.len());
            for ((residual_output, residual_type), stack_input) in
                residual_outputs.iter().zip(residual_types).zip(stack_inputs.iter())
            {
                let batch_item_type = stacked_scan_type(residual_type, 1);
                let output_axes = (1..=residual_type.rank()).collect::<Vec<_>>();
                let expanded = residual_output.broadcast(batch_item_type, output_axes.as_slice())?;
                let mut start_indices = vec![counter_input.clone()];
                if let Some(zero_index) = &zero_index {
                    start_indices.extend((0..residual_type.rank()).map(|_| zero_index.clone()));
                }
                next_stacks.push(stack_input.dynamic_update_slice(&expanded, start_indices.as_slice())?);
            }
            let true_scalar = context.one(&boolean_scalar_type)?;
            let true_item_type = stacked_scan_type(&boolean_scalar_type, 1);
            let true_item = true_scalar.broadcast(true_item_type, &[])?;
            let next_mask = mask_input.dynamic_update_slice(&true_item, std::slice::from_ref(&counter_input))?;
            let one_i64 = context.one(&counter_type)?;
            let next_counter = Add::add(&counter_input, &one_i64)?;
            body_outputs.push(next_counter);
            body_outputs.extend(next_stacks);
            body_outputs.push(next_mask);
            Ok(body_outputs)
        },
        body_input_types,
    )?
    .1;

    // Condition: the original loop condition extended with ignored extra-state inputs.
    let condition_input_types = condition
        .input_types()
        .into_iter()
        .chain(std::iter::once(counter_type))
        .chain(stack_types.iter().cloned())
        .chain(std::iter::once(mask_stack_type))
        .collect::<Vec<_>>();
    let condition_input_count = condition_input_types.len();
    let mut condition_builder = ProgramBuilder::new();
    let condition_inputs = condition_input_types
        .into_iter()
        .map(|r#type| condition_builder.add_input(r#type))
        .collect::<Vec<_>>();
    let condition_outputs = condition_builder.splice_program(condition, &condition_inputs[..state_count])?;
    let condition_output_count = condition_outputs.len();
    let extended_condition = condition_builder.build(
        condition_outputs,
        vec![Placeholder; condition_input_count],
        vec![Placeholder; condition_output_count],
    )?;
    Ok((extended_condition, body, stack_types))
}

/// Capture-free forward-mode (JVP) rule for [`ConditionOperation`], staging **one fused** jvp `condition` as an
/// ordinary primal-enum operation over the shared builder.
///
/// The rule builds each branch's fused jvp program through its instruction-scoped differentiation driver — both
/// branches share a signature, so the doubled `[primal_operands..., tangent_operands...] ->
/// [primal_outputs..., tangent_outputs...]` signatures also match with no joining or padding — and stages one
/// `condition` over the predicate primal followed by the operand primals and tangents. Pure forward mode therefore
/// stages a single conditional and no residual plumbing.
///
/// The primal/tangent separation that reverse mode needs is deferred to partial evaluation: under the known-ness
/// split of [`Program::linearize`](crate::Program::linearize) the predicate is a known (symbolic) primal, so the
/// condition composite split (ryft's `_cond_partial_eval` analogue) separates the fused conditional into a known
/// primal condition — producing each branch's known→unknown edges with typed zero-padding for the peer's slots —
/// and a residual tangent condition over the operand tangents and those edges.
///
/// The predicate is the first operand and carries no tangent (Boolean predicates have no tangent space); the fused
/// conditional selects the same branch for both halves because they share the same primal predicate edge.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for ConditionOperation<C::Constant>
where
    C::Operation: From<ZeroOperation<ArrayType>> + From<ConditionOperation<C::Constant>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The rule requests all nested-computation work through its driver (region 0 is the `true` branch
        // and region 1 the `false` branch); the true branch's boundary is materialized for the arity checks.
        let true_branch = driver.region(0)?;
        check_count!("input", inputs, true_branch.input_types().len() + 1, ProgramError);
        let predicate_primal = inputs[0].primal().clone();
        let operands = &inputs[1..];
        let output_count = true_branch.output_ids().len();

        // Build both fused jvp branches and stage one fused conditional over the predicate primal followed by the
        // operand primals and tangents.
        let fused_true = driver.jvp_program(true_branch)?;
        let fused_false = driver.jvp_program(driver.region(1)?)?;
        let fused_condition = ConditionOperation::new();
        let mut condition_operands = Vec::with_capacity(2 * operands.len() + 1);
        condition_operands.push(predicate_primal);
        condition_operands.extend(operands.iter().map(|operand| operand.primal().clone()));
        // The fused branches take every operand tangent as a real program input, so materialize structural zeros.
        for operand in operands {
            condition_operands.push(operand.tangent().clone().materialize(context)?);
        }
        let outputs = context.bind(fused_condition, vec![fused_true, fused_false], &condition_operands)?;
        check_count!("output", outputs, 2 * output_count, ProgramError);

        // The fused conditional's outputs are the primal outputs followed by the tangent outputs; zip the halves
        // back into `DifferentiationDual`s in the original output order.
        let (primal_outputs, tangent_outputs) = outputs.split_at(output_count);
        Ok(primal_outputs
            .iter()
            .cloned()
            .zip(tangent_outputs.iter().cloned())
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect())
    }
}

/// Type-family forward-mode (JVP) semantics for [`WhileOperation`], with the differentiation context riding as a
/// trait input and the type family as the implementing type (mirroring the partial-evaluation dispatch in the
/// `while` module) so that the [`ArrayType`] and [`DataType`] rules stay coherent without the operation struct
/// naming its type family as a parameter, and so that each family implementation carries exactly the capability
/// bounds its rule needs.
pub(crate) trait WhileJvp<C>: WhileTypeSemantics
where
    C: Context<Type = Self>,
{
    /// Applies the type family's `while` forward-mode rule over the loop's materialized condition and body region
    /// programs; refer to the documentation of [`DifferentiableOperation::jvp`] for the contract. The scoped
    /// `driver` serves the rule's nested forward-mode and linearization requests over rebuilt body forms,
    /// keeping the rule free of operation-family semantic bounds.
    fn jvp_while<D: DifferentiationDriver<C>>(
        operation: &WhileOperation,
        condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        body: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>;
}

/// Capture-free forward-mode (JVP) rule for the bounded [`WhileOperation`], staging an augmented primal `while`
/// and one masked length-`bound` tangent `scan` as ordinary primal-enum operations over the shared builder.
///
/// In the bounded regime the rule keeps every per-iteration residual and the validity mask as plain primal operand
/// edges: they leave the augmented primal while as ordinary stacked outputs and re-enter the tangent scan as ordinary
/// stacked scanned inputs, so no symbolic capture is ever introduced. The enclosing partial-evaluation split then
/// discovers the residual operand edges structurally, exactly as it does for the scan and condition rules.
///
/// **The unbounded case is rejected.** This staged rule is only reached when the context is not
/// [eager](Context::is_eager) (eager contexts run the loop directly through
/// [`jvp_while_eagerly`], with no bound needed), and without a semantic
/// [`iteration_bound`](crate::operations::control_flow::WhileOperation::with_iteration_bound) there is no statically
/// shaped residual stack and no transposable form, so the rule reports
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) and the non-transposable unbounded-while boundary is
/// preserved (the capture-based rule's unbounded regime stages a non-transposable doubled-state loop, which has
/// no capture-free counterpart here).
///
/// For a bound `B`, the rule linearizes the body capture-free through its instruction-scoped driver,
/// giving a primal body `[state] -> [next_state, residuals...]` and a tangent body
/// `[state_tangent, residuals...] -> [next_state_tangent]` together with the residual count. It then:
///
///   1. Builds the augmented primal `while` over the state `[original_state..., counter (i64 scalar), residual_stacks
///      (one zero-initialized [B, ...] stack per residual), mask_stack (a false-initialized Boolean [B] stack)]`
///      with `build_bounded_while_programs` from the residual-extended primal body, keeping `iteration_bound = B` so
///      the per-item writes can never clamp, and stages it over the operand primals followed by the staged counter and
///      stack zeros. Its outputs split into the original state outputs (the primal outputs), the dropped counter, the
///      stacked residual outputs, and the mask stack.
///   2. Stages a length-`B` tangent [`ScanOperation`] whose body is the tangent body extended so each per-iteration
///      output is wrapped in a [`SelectOperation`] over that state element's mask item, choosing the pushforward output
///      on valid batch items and the carried tangent input on batch items beyond the actual trip count. Because
///      [`SelectOperation`] requires a shape-congruent condition, the Boolean `[B]` mask stack is broadcast to a
///      `[B, ...state_shape]` stack per state element outside the loop, and each broadcast stack is appended as an
///      extra scanned input, so iteration `item` reads its own shape-congruent mask slice. The scan body input order is
///      therefore `[state_tangent..., residual_slice..., mask_slice...]`, with the leading `state_count` carry tangents
///      linear and the trailing residual and mask slices treated as scanned (known) operand edges.
///   3. Pairs each primal output tracer with its tangent output tracer into a [`DifferentiationDual`].
///
/// Reverse mode is total with no while-specific transpose code: the staged tangent scan re-keys through the existing
/// scan re-key path into a captured-stack linear scan whose body re-keys the per-iteration `select` over its mask-item
/// capture, and the single outer transpose flips the scan direction and transposes the body — the masked pushforward
/// side receives a zero cotangent on inactive batch items while the carried side receives the full cotangent, so
/// cotangents pass through inactive batch items unchanged exactly like the capture-based rule.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> WhileJvp<C> for ArrayType
where
    C::Value: BooleanLike,
    C::Operation: From<ZeroOperation<ArrayType>>
        + From<OneOperation<ArrayType>>
        + From<AddOperation>
        + From<BroadcastOperation>
        + From<DynamicUpdateSliceOperation>
        + From<SelectOperation>
        + From<ReduceOperation>
        + From<AndOperation>
        + From<WhileOperation>
        + From<ScanOperation<C::Constant>>,
    for<'operation> &'operation WhileOperation: TryFrom<&'operation C::Operation>,
{
    fn jvp_while<D: DifferentiationDriver<C>>(
        operation: &WhileOperation,
        condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        body: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let state_types = body.input_types();
        let state_count = state_types.len();
        check_count!("input", inputs, state_count, ProgramError);

        // A batched (per-item) predicate cannot thread the bounded rule's augmented differentiation state through the
        // predicate-prefix contract (the scalar iteration counter and the `[bound, ...]` residual stacks are not
        // predicate-prefixed), so the loop is first rewritten into its scalar-predicate masked normal form over
        // `[state..., active_mask]` (see `masked_while_programs`) and differentiated recursively — the masked loop's
        // forward mode is this same rule. The initial mask is the condition replayed on the operand primals, carried
        // with a zero tangent since a Boolean mask has no derivative.
        let predicate_type = condition.output_types()[0].clone();
        if predicate_type.rank() > 0 {
            let (masked_condition, masked_body) = masked_while_programs(condition, body)?;
            let masked_while = WhileOperation::new().with_iteration_bound(operation.iteration_bound())?;
            let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let mut initial_mask = condition.interpret_in_context(context, primal_operands)?;
            check_count!("output", initial_mask, 1, ProgramError);
            let mut extended_inputs = inputs.to_vec();
            extended_inputs.push(DifferentiationDual::new(initial_mask.remove(0), MaybeZero::Zero(predicate_type)));
            // The masked loop's condition and body are freshly built region programs, so the recursive `jvp` is
            // requested through the instruction-scoped driver over them.
            let mut outputs = driver.jvp_operation(
                &C::Operation::from(masked_while),
                vec![masked_condition, masked_body],
                context,
                extended_inputs.as_slice(),
            )?;
            check_count!("output", outputs, state_count + 1, ProgramError);
            outputs.truncate(state_count);
            return Ok(outputs);
        }

        // An unbounded while loop has no statically shaped residual stack and no transposable form, so it has no
        // capture-free forward-mode rule.
        let Some(bound) = operation.iteration_bound() else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "operation `{}` has no capture-free forward-mode linearization rule unless it carries an \
                     iteration bound; an unbounded while loop has no forward-mode rule",
                    Operation::<ArrayType>::name(operation),
                ),
            }
            .into());
        };

        // Linearize the body capture-free. The primal body produces `[next_state..., residuals...]` and the
        // tangent body consumes `[state_tangent..., residuals...]`; the residual count is the number of trailing
        // outputs of the primal body beyond the loop state.
        let (primal_program, tangent_program, residual_count) =
            driver.linearize_program(driver.region(1)?)?.into_parts();
        let residual_types = primal_program.output_types().split_off(state_count);

        // Build and bind the augmented primal while over `[state..., counter, residual_stacks..., mask_stack]`, with
        // the counter starting at zero and the stacks (including the Boolean mask, whose zero is false) starting at
        // typed zeros staged in the shared builder.
        let counter_type = ArrayType::scalar(DataType::I64);
        let boolean_scalar_type = ArrayType::scalar(DataType::Boolean);
        let mask_stack_type = stacked_scan_type(&boolean_scalar_type, bound);
        let (extended_condition, augmented_body, stack_types) =
            build_bounded_while_programs(condition, &primal_program, residual_types.as_slice(), bound)?;
        let augmented_while = WhileOperation::new().with_iteration_bound(bound)?;
        let mut primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let zero_state_types =
            std::iter::once(&counter_type).chain(stack_types.iter()).chain(std::iter::once(&mask_stack_type));
        for zero_state_type in zero_state_types {
            let mut zeros = context.bind(ZeroOperation::new(zero_state_type.clone()), Vec::new(), &[])?;
            check_count!("output", zeros, 1, ProgramError);
            primal_operands.push(zeros.remove(0));
        }
        let mut while_outputs = context.bind(
            C::Operation::from(augmented_while),
            vec![extended_condition, augmented_body],
            &primal_operands,
        )?;
        check_count!("output", while_outputs, state_count + 2 + stack_types.len(), ProgramError);
        let mask_stack = while_outputs.pop().unwrap();
        let residual_stacks = while_outputs.split_off(state_count + 1);
        // Drop the internal iteration counter output; the rule's primal outputs are the original loop state.
        while_outputs.truncate(state_count);
        let primal_outputs = while_outputs;

        // Only *differentiable* state elements receive a masked per-iteration update. A non-differentiable state
        // element (the `float0` analogue, such as batching's Boolean active-mask carry) has a structural-zero tangent,
        // so masking it with `select(mask_item, pushforward, carried)` would be an all-known select that contributes
        // no linear computation. Following JAX's structure, such an element instead passes its pushforward tangent
        // through directly, so the tangent body stays genuinely linear (no all-known select) and reverse mode does no
        // dead work. Mask stacks and mask items are therefore produced only for differentiable elements, keeping the
        // scanned mask operands and the body's appended mask-item inputs aligned.
        let element_is_differentiable =
            state_types.iter().map(|state_type| state_type.cotangent().is_some()).collect::<Vec<_>>();

        // Broadcast the Boolean `[B]` mask stack to a shape-congruent `[B, ...state_shape]` stack per differentiable
        // state element, so each per-iteration select reads a mask slice that matches that element's shape (select
        // requires a shape-congruent condition). Scalar state elements reuse the `[B]` mask stack directly.
        let mut mask_stacks = Vec::new();
        for (state_type, &is_differentiable) in state_types.iter().zip(element_is_differentiable.iter()) {
            if !is_differentiable {
                continue;
            }
            if state_type.rank() == 0 {
                mask_stacks.push(mask_stack.clone());
                continue;
            }
            let condition_type = ArrayType::new(DataType::Boolean, state_type.shape().clone());
            let stacked_condition_type = stacked_scan_type(&condition_type, bound);
            let mut broadcasted = context.bind(
                C::Operation::from(BroadcastOperation::new(stacked_condition_type, vec![0])),
                Vec::new(),
                std::slice::from_ref(&mask_stack),
            )?;
            check_count!("output", broadcasted, 1, ProgramError);
            mask_stacks.push(broadcasted.remove(0));
        }

        // Build the masked tangent scan body: the tangent body extended so each *differentiable* per-iteration output
        // is selected against that state element's mask item, with the mask items appended as extra scanned inputs
        // after the residual slices. A non-differentiable state element's output is its pushforward tangent unchanged.
        // The body input order `[state_tangent..., residual_slice..., mask_slice...]` keeps the leading `state_count`
        // carry tangents linear so the reverse re-key folds the residual and mask slices into scan-local captures.
        check_count!("input", tangent_program.input_ids(), state_count + residual_count, ProgramError);
        check_count!("output", tangent_program.output_ids(), state_count, ProgramError);
        let mask_item_types = state_types
            .iter()
            .zip(element_is_differentiable.iter())
            .filter_map(|(state_type, &is_differentiable)| {
                is_differentiable.then(|| ArrayType::new(DataType::Boolean, state_type.shape().clone()))
            })
            .collect::<Vec<_>>();
        let scan_body_input_types =
            tangent_program.input_types().into_iter().chain(mask_item_types).collect::<Vec<_>>();
        let scan_body = if scan_body_input_types.is_empty() {
            tangent_program
        } else {
            TracingContext::<C::Constant, C::Operation>::trace(
                |inputs| {
                    let context = inputs[0].context().clone();
                    let tangent_input_count = tangent_program.input_ids().len();
                    let carried_inputs = inputs[..state_count].to_vec();
                    let mut mask_items = inputs[tangent_input_count..].iter();
                    let pushforward_outputs =
                        tangent_program.interpret_in_context(&context, inputs[..tangent_input_count].to_vec())?;
                    check_count!("output", pushforward_outputs, state_count, ProgramError);
                    let mut masked_outputs = Vec::with_capacity(state_count);
                    for ((pushforward_output, carried_input), &is_differentiable) in
                        pushforward_outputs.into_iter().zip(carried_inputs).zip(element_is_differentiable.iter())
                    {
                        if !is_differentiable {
                            masked_outputs.push(pushforward_output);
                            continue;
                        }
                        let mask_item = mask_items.next().cloned().ok_or_else(|| {
                            ProgramError::MalformedProgram(
                                "masked tangent scan body adapter is missing a mask input".to_string(),
                            )
                        })?;
                        masked_outputs.push(Select::select(&mask_item, &pushforward_output, &carried_input)?);
                    }
                    Ok(masked_outputs)
                },
                scan_body_input_types,
            )?
            .1
        };

        // Stage the length-`bound` tangent scan over the carry tangents followed by the stacked residuals and then the
        // per-differentiable-state-element mask stacks. Iteration `item` reads residual slice `item` and mask slice
        // `item`.
        let tangent_scan = ScanOperation::<C::Constant>::new(state_count, bound);
        // The tangent scan takes every carry tangent as a real program input, so materialize structural zeros.
        let mut tangent_operands = inputs
            .iter()
            .map(|input| input.tangent().clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        tangent_operands.extend(residual_stacks);
        tangent_operands.extend(mask_stacks);
        let tangent_outputs = context.bind(C::Operation::from(tangent_scan), vec![scan_body], &tangent_operands)?;
        check_count!("output", tangent_outputs, state_count, ProgramError);

        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect())
    }
}

/// Forward-mode (JVP) rule for the scalar [`WhileOperation`]: the scalar `while` loop carries a tangent that is not
/// expressible as primal-enum operand arithmetic (there is no scalar residual-stack representation backing the
/// masked tangent scan the bounded array rule stages), so the rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<C: Context<Type = DataType>> WhileJvp<C> for DataType {
    fn jvp_while<D: DifferentiationDriver<C>>(
        operation: &WhileOperation,
        _condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        _body: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        _context: &C,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "operation `{}` has no capture-free forward-mode linearization rule",
                Operation::<DataType>::name(operation),
            ),
        }
        .into())
    }
}

/// Runs a `while` loop's forward-mode rule directly at concrete duals for an
/// [eager](Context::is_eager) context, returning `None` when the loop's predicate does
/// not concretize to one scalar Boolean (e.g., a batched per-item predicate) and the caller must therefore fall back
/// to the type family's staged strategy.
///
/// Each iteration evaluates the condition on the concrete primal carries, unrolls any nested data-dependent `while`
/// in the body at those carries (through the same value-level rewrite the reverse-mode pre-pass uses), fuses the
/// unrolled body into its JVP program through the differentiation driver's program-taking request, and replays that
/// fused program once over `[primal_carries ++ tangent_carries]` to advance both halves. Data-dependent trip counts
/// therefore need no iteration bound — this is the analogue of
/// [JAX's `jvp` through an eagerly executed loop](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html) — while a
/// semantic [`iteration_bound`](WhileOperation::with_iteration_bound) truncates the loop once it is reached, matching
/// the bounded-`while` truncation semantics. Body effects fire while the loop runs (the correct all-known placement),
/// once during the nested-`while` unroll interpretation and once during the fused replay, exactly as they did on the
/// reverse-mode pre-pass path.
fn jvp_while_eagerly<C, D: DifferentiationDriver<C>>(
    operation: &WhileOperation,
    condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    body: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    context: &C,
    driver: &D,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Option<Vec<DifferentiationDual<C::Value>>>, ProgramError>
where
    C: Context + Zero<C::Value>,
    C::Value: BooleanLike,
    C::Operation: From<ZeroOperation<C::Type>>,
    for<'operation> &'operation WhileOperation: TryFrom<&'operation C::Operation>,
{
    let state_count = inputs.len();
    let mut primal_carries = Vec::with_capacity(state_count);
    let mut tangent_carries = Vec::with_capacity(state_count);
    for input in inputs {
        primal_carries.push(input.primal().clone());
        tangent_carries.push(input.tangent().clone().materialize(context)?);
    }

    let mut completed_iterations = 0;
    loop {
        if operation.iteration_bound().is_some_and(|bound| completed_iterations >= bound) {
            break;
        }

        // Concretize the condition on the current concrete primal carries to decide whether another iteration runs.
        let mut condition_outputs = condition.interpret_in_context(context, primal_carries.clone())?;
        check_count!("output", condition_outputs, 1, ProgramError);
        let predicate = match condition_outputs.remove(0).boolean() {
            Ok(predicate) => predicate,
            // The predicate does not concretize to one scalar Boolean — e.g., a batched per-item predicate, whose
            // items stop on different iterations, has no single trip decision. Report the loop as non-concretizable
            // so the caller falls back to the type family's staged strategy; nothing has been advanced yet on the
            // first iteration. The predicate type is loop-invariant, so a later-iteration failure cannot occur once
            // the first concretization succeeds, and any such error is surfaced.
            Err(_) if completed_iterations == 0 => return Ok(None),
            Err(error) => return Err(error),
        };
        if !predicate {
            break;
        }

        // Advance one iteration: unroll nested data-dependent loops at the current concrete carries, fuse the
        // straight-line body into its JVP program, and replay it over both carry halves.
        let body = unroll_concretizable_whiles(context, body.clone(), primal_carries.clone())?;
        let fused_body = driver.jvp_program(body.entry_region_ref())?;
        let mut combined_carries = primal_carries;
        combined_carries.extend(tangent_carries);
        let mut outputs = fused_body.interpret_in_context(context, combined_carries)?;
        check_count!("output", outputs, 2 * state_count, ProgramError);
        tangent_carries = outputs.split_off(state_count);
        primal_carries = outputs;
        completed_iterations += 1;
    }

    Ok(Some(
        primal_carries
            .into_iter()
            .zip(tangent_carries)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect(),
    ))
}

/// Forward-mode (JVP) rule for [`WhileOperation`]. An [eager](Context::is_eager) context
/// runs the loop directly at the concrete duals (see the crate-private `jvp_while_eagerly`), so eager forward mode is total over
/// data-dependent `while` loops with no iteration bound. Staging contexts — and eager contexts whose loop
/// predicate is batched and therefore has no single trip decision — dispatch to the loop's type family through
/// `WhileJvp` type family: array loops stage the hybrid bounded rule documented on that trait's [`ArrayType`] implementation,
/// and scalar loops report an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<C> DifferentiableOperation<C> for WhileOperation
where
    C: Context + Zero<C::Value>,
    C::Type: WhileJvp<C>,
    C::Value: BooleanLike,
    C::Operation: From<ZeroOperation<C::Type>>,
    for<'operation> &'operation WhileOperation: TryFrom<&'operation C::Operation>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The rule requests all nested-computation work through its driver (region 0 is the condition and
        // region 1 the body), which keeps its bounds free of the operation family's own semantic traits.
        let condition = driver.region(0)?.to_program();
        let body = driver.region(1)?.to_program();
        if context.is_eager()
            && let Some(outputs) = jvp_while_eagerly(self, &condition, &body, context, driver, inputs)?
        {
            return Ok(outputs);
        }
        <C::Type>::jvp_while(self, &condition, &body, context, driver, inputs)
    }
}

impl<V: Value, O> TransposableOperation<V, O> for WhileOperation
where
    V::Type: WhileTypeSemantics,
    O: Operation<V::Type>,
{
    /// Rejects transposition. This rule is only reachable for *unbounded* staged while loops — the doubled-state
    /// linear loop staged by the [`WhileOperation`] JVP rule, which recomputes primal state *forward* through
    /// the iterations, so transposing it would have to run that recomputation backwards, which a while loop cannot
    /// express. Two paths avoid it entirely: concretizing domains unroll the loop into a straight-line pushforward
    /// that transposes (so eager reverse-mode differentiation through unbounded while loops works), and bounded
    /// loops ([`WhileOperation::with_iteration_bound`]) never stage a linear `while` — their tangent side is a
    /// masked linear scan whose transpose is total, so reverse mode through staged bounded loops flows through the
    /// scan transpose without reaching this rule.
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: "while does not support transposition (reverse-mode differentiation through staged unbounded \
                      while loops is not supported; eager differentiation unrolls the loop instead, and loops built \
                      with `with_iteration_bound` stage a transposable masked scan)"
                .to_string(),
        }
        .into())
    }
}

/// Batches a condition whose predicate is *batch-varying* by interpreting both branches over the operand inputs via
/// `interpret_program` and merging their outputs per batch item via
/// [`Select`](crate::operations::control_flow::Select). Each output's batch axis is joined across the branches
/// (erroring when the branches disagree on a mapped position and defaulting to the predicate's axis when both branch
/// outputs are replicated). The predicate must carry a mapped batch axis; the replicated case is the caller's
/// structural staging path.
pub(crate) fn batch_condition_with_interpreter<V, F>(
    predicate_batch: &ArrayBatch<V>,
    operand_inputs: &[ArrayBatch<V>],
    mut batch_branch: F,
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    V: Value<Type = ArrayType> + BooleanLike + crate::operations::control_flow::Select<Condition = V>,
    F: FnMut(usize, Vec<ArrayBatch<V>>) -> Result<Vec<ArrayBatch<V>>, BatchingError>,
{
    let predicate_axis = predicate_batch.batch_axis_position().unwrap();
    let true_outputs = batch_branch(0, operand_inputs.to_vec())?;
    let false_outputs = batch_branch(1, operand_inputs.to_vec())?;
    check_count!("output", true_outputs, false_outputs.len(), ProgramError);
    true_outputs
        .into_iter()
        .zip(false_outputs)
        .map(|(true_output, false_output)| -> Result<ArrayBatch<V>, BatchingError> {
            let output_axis = match (true_output.batch_axis_position(), false_output.batch_axis_position()) {
                (Some(left), Some(right)) if left != right => {
                    return Err(BatchingError::MisalignedBatchAxes {
                        message: format!(
                            "condition branches produced batch-varying outputs at mismatched axes \
                            ({left} vs {right})",
                        ),
                    });
                }
                (Some(axis), _) | (_, Some(axis)) => axis,
                (None, None) => predicate_axis,
            };
            let selected = V::select(predicate_batch.value(), true_output.value(), false_output.value())?;
            let output_type = selected.r#type().into_owned();
            ArrayBatch::new(output_type, selected, BatchAxis::from_position(output_axis))
        })
        .collect()
}

/// Batching rule for [`ConditionOperation`]. The rule builds batched condition *structure* and binds it into the
/// parent context — interpreted eagerly under an eager parent and staged into the enclosing trace under a staging
/// parent:
///
///   - **Replicated predicate.** Both branch programs are batched at the operand batch axes via
///     [`Program::batched`](crate::Program::batched) (the batching analog of symbolic program
///     linearization), their per-output batch axes are
///     normalized to a common layout by appending staged axis-moving operations at the branch tails when they
///     disagree (a transpose for a mismatched axis, a broadcast for a replicated output paired with a batched
///     one), and one [`ConditionOperation`] over the batched branches is bound into the parent context with the
///     unbatched predicate passed through as its scalar Boolean operand. A staging parent therefore keeps one
///     `condition` operation whose branches run whole batches per batch item, while an eager parent concretizes the
///     predicate and interprets the chosen batched branch.
///   - **Batch-varying predicate.** Both branches are interpreted over the operand inputs and merged per batch item
///     via [`Select`]: every per-item primitive re-enters this operation
///     family's batching rules against the same active context, so the multi-operation rewrite composes for eager
///     and staging parents alike.
impl<C, O> BatchableOperation<C> for ConditionOperation<C::Constant>
where
    C: Context<Type = ArrayType, Operation = O>,
    <C as Domain>::Value: BooleanLike + Select<Condition = <C as Domain>::Value>,
    O: Operation<ArrayType>
        + From<TransposeOperation>
        + From<BroadcastOperation>
        + From<SelectOperation>
        + From<ConditionOperation<C::Constant>>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        let Some((predicate_batch, operand_inputs)) = inputs.split_first() else {
            return Err(BatchingError::UnsupportedOperation {
                message: "cannot batch a condition operation with no predicate input".to_string(),
            });
        };
        if !predicate_batch.batch_axis().is_replicated() {
            // Batch-varying predicate: batch both branches item-agnostically through the region access and merge
            // their outputs per batch item via `Select`.
            return batch_condition_with_interpreter(predicate_batch, operand_inputs, |index, region_inputs| {
                driver.batch_region(context, index, region_inputs)
            });
        }

        // Replicated (abstract) predicate: batch both branches at the operand batch axes with natural output axes to
        // discover which outputs each branch batches (the discovery programs are discarded), join the two answers into
        // one output layout — preferring the true branch's natural axis when both are batched — and re-batch each
        // branch instantiated at the joined targets so the branch signatures agree. This is the two-pass shape of
        // JAX's `_cond_batching_rule` (`batch_jaxpr` with `instantiate=out_bat`).
        let operand_axes = operand_inputs.iter().map(|input| input.batch_axis()).collect::<Vec<_>>();
        let true_region = driver.region(0)?;
        let false_region = driver.region(1)?;
        let (_, true_axes) = driver.batch_program(
            context,
            true_region,
            operand_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let (_, false_axes) = driver.batch_program(
            context,
            false_region,
            operand_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        check_count!("output", false_axes, true_axes.len(), ProgramError);
        let output_axes: Vec<BatchAxis> = true_axes
            .iter()
            .zip(false_axes.iter())
            .map(|(true_axis, false_axis)| if true_axis.is_replicated() { *false_axis } else { *true_axis })
            .collect();
        let (batched_true_branch, _) = driver.batch_program(
            context,
            true_region,
            operand_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::AlignEachTo(output_axes.clone()),
        )?;
        let (batched_false_branch, _) = driver.batch_program(
            context,
            false_region,
            operand_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::AlignEachTo(output_axes.clone()),
        )?;

        // Stage one condition over the batched branches with the unbatched predicate passed through.
        let batched_condition = ConditionOperation::new();
        let mut staged_inputs = Vec::with_capacity(inputs.len());
        staged_inputs.push(predicate_batch.value().clone());
        staged_inputs.extend(operand_inputs.iter().map(|input| input.value().clone()));
        let outputs = context.parent().bind(
            batched_condition,
            vec![batched_true_branch, batched_false_branch],
            &staged_inputs,
        )?;
        check_count!("output", outputs, output_axes.len(), ProgramError);
        outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| {
                let physical_type = output.r#type().into_owned();
                ArrayBatch::new(physical_type, output, axis)
            })
            .collect()
    }
}

/// Batching rule for [`WhileOperation`]. The rule builds batched loop *structure* and binds it into the parent
/// context — interpreted eagerly under an eager parent (whose relaxed-predicate interpretation owns the per-item
/// masked semantics) and staged into the enclosing trace under a staging parent:
///
///   1. Every batched state input is realigned to batch axis `0` in the parent context, and the body is batched at
///      the state batch axes via [`Program::batched`](crate::Program::batched),
///      iterating the axes to a fixed point: a while loop's state types are loop-invariant, so a replicated state
///      element whose update depends on a batched element *becomes* batched, and the rule widens that element's
///      input axis and re-batches until the body is axis-invariant (the iteration count is bounded by the state
///      count because every non-final pass widens at least one element). Each pass instantiates the body outputs at
///      the current state axes ([`ProgramBatchingOutputAxesPolicy::AlignEachTo`], mirroring JAX's
///      `instantiate=carry_bat`), so the converged body is already aligned to the loop-invariant state layout, and
///      widened parent inputs gain their batch axis through staged broadcasts.
///   2. The condition is batched at the stabilized axes. When its predicate output stays *replicated*, one
///      [`WhileOperation`] over the batched condition and body is bound into the parent directly, preserving any
///      semantic [`iteration_bound`](WhileOperation::with_iteration_bound) (so bounded loops stay reverse-capable
///      under `batch`).
///   3. When the predicate output is *batched* (per-item termination), every state element is widened to a batched
///      element, the condition is re-batched with its predicate output instantiated at axis `0`, and one
///      [`WhileOperation`] is bound directly with that batched predicate (mirroring JAX's
///      `_while_loop_batching_rule`). The predicate's `[axis_size]` shape is a prefix of every widened state shape,
///      so the bound loop satisfies the relaxed predicate contract and its consumers own the masked semantics:
///      eager interpretation continues while any per-item predicate is true and freezes finished items, and the XLA
///      lowering reduces the predicate with `or` and masks carry updates with a broadcast select. The iteration
///      bound is preserved (batch items share masked iterations, so capping the loop matches per-item truncation
///      exactly).
impl<C, O> BatchableOperation<C> for WhileOperation
where
    C: Context<Type = ArrayType, Operation = O>,
    <C as Domain>::Value: Broadcast + Transpose,
    O: Operation<ArrayType>
        + From<TransposeOperation>
        + From<BroadcastOperation>
        + From<ReduceOperation>
        + From<SelectOperation>
        + From<AndOperation>
        + From<WhileOperation>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        // The rule requests all nested-computation work through its region access (region 0 is the condition and
        // region 1 the body), which keeps its bounds free of the operation family's own semantic traits.
        let state_count = inputs.len();
        let axis_size = context.axis_size();
        let condition_region = driver.region(0)?;
        let body_region = driver.region(1)?;

        // Realign every batched state input to batch axis 0 in the parent context, so the loop-invariance fixed
        // point below only ever distinguishes replicated (`None`) from batched-at-0 (`Some(0)`) state elements.
        let mut state = inputs.iter().map(|input| input.move_axis(0)).collect::<Result<Vec<_>, _>>()?;
        let mut state_axes = state.iter().map(|input| input.batch_axis()).collect::<Vec<_>>();

        // Iterate the body's batch axes to a fixed point: a replicated state element whose update is batched
        // becomes batched. Every non-final pass widens at least one of the `state_count` elements, so the loop
        // stabilizes within `state_count + 1` passes by construction; the trailing error guards the contract that
        // separately implemented batching rules report widening monotonically. Each pass instantiates the body's
        // outputs at the current state axes (JAX's `instantiate=carry_bat`), so the body that stabilizes the fixed
        // point is already aligned to the loop-invariant state layout and needs no further normalization.
        let mut batched_body = None;
        for _ in 0..=state_count {
            let (candidate_body, body_axes) = driver.batch_program(
                context,
                body_region,
                state_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::AlignEachTo(state_axes.clone()),
            )?;
            check_count!("output", body_axes, state_count, ProgramError);
            let mut widened = false;
            for (state_axis, body_axis) in state_axes.iter_mut().zip(body_axes.iter()) {
                if state_axis.is_replicated() && !body_axis.is_replicated() {
                    *state_axis = BatchAxis::new(0);
                    widened = true;
                }
            }
            if !widened {
                batched_body = Some(candidate_body);
                break;
            }
        }
        let Some(mut batched_body) = batched_body else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "while loop batching failed to stabilize the loop state batch axes within {state_count} \
                     widening passes",
                ),
            });
        };

        // Batch the condition at the stabilized axes; a batched predicate output means per-item termination, in
        // which case every state element participates in per-item masking and is therefore widened to a batched
        // element before the masked loop structure is built.
        let (mut batched_condition, mut condition_axes) = driver.batch_program(
            context,
            condition_region,
            state_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        check_count!("output", condition_axes, 1, ProgramError);
        let batch_varying = !condition_axes[0].is_replicated();
        if batch_varying && state_axes.iter().any(|axis| axis.is_replicated()) {
            state_axes = vec![BatchAxis::new(0); state_count];
            let (widened_body, body_axes) = driver.batch_program(
                context,
                body_region,
                state_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::AlignEachTo(state_axes.clone()),
            )?;
            check_count!("output", body_axes, state_count, ProgramError);
            batched_body = widened_body;
            (batched_condition, condition_axes) = driver.batch_program(
                context,
                condition_region,
                state_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::Natural,
            )?;
            check_count!("output", condition_axes, 1, ProgramError);
        }

        // Widen the parent state values whose elements became batched (their batch axis is materialized through a
        // staged broadcast); the batched body's outputs are already aligned to the state axes by the fixed point.
        for (element, state_axis) in state.iter_mut().zip(state_axes.iter()) {
            if !state_axis.is_replicated() && element.batch_axis().is_replicated() {
                *element = element.broadcast(0, axis_size)?;
            }
        }
        let state_values = state.iter().map(|element| element.value().clone()).collect::<Vec<_>>();

        // Replicated predicate: stage one while over the batched condition and body directly.
        if !batch_varying {
            let batched_while = WhileOperation::new().with_iteration_bound(self.iteration_bound())?;
            let outputs = context.parent().bind(batched_while, vec![batched_condition, batched_body], &state_values)?;
            check_count!("output", outputs, state_count, ProgramError);
            return outputs
                .into_iter()
                .zip(state_axes)
                .map(|(output, axis)| {
                    let physical_type = output.r#type().into_owned();
                    ArrayBatch::new(physical_type, output, axis)
                })
                .collect();
        }

        // Batch-varying predicate (per-item termination): re-batch the condition with its predicate output
        // instantiated at axis 0 and stage the while directly with that batched predicate, mirroring JAX's
        // `_while_loop_batching_rule` (which re-batches the cond jaxpr with the predicate at dimension 0 and binds
        // `while_p` directly). The predicate's `[axis_size]` shape is a prefix of every (widened) state shape, so the
        // staged loop satisfies the relaxed predicate contract, and the loop's consumers own the masked semantics:
        // eager interpretation continues while any per-item predicate is true and freezes finished items through
        // `WhilePredicate::mask_select`, and the XLA lowering
        // reduces the predicate with `or` and masks carry updates with a broadcast select.
        let (batched_condition, condition_axes) = driver.batch_program(
            context,
            condition_region,
            state_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::AlignEachTo(vec![BatchAxis::new(0)]),
        )?;
        check_count!("output", condition_axes, 1, ProgramError);
        let batched_while = WhileOperation::new().with_iteration_bound(self.iteration_bound())?;
        let outputs = context.parent().bind(batched_while, vec![batched_condition, batched_body], &state_values)?;
        check_count!("output", outputs, state_count, ProgramError);
        outputs
            .into_iter()
            .map(|output| {
                let physical_type = output.r#type().into_owned();
                ArrayBatch::new(physical_type, output, Some(0))
            })
            .collect()
    }
}

/// Rewrites a while loop's condition and body into the scalar-predicate *masked form* over the augmented state
/// `[state..., active_mask]`: the masked condition reduces the mask with a Boolean `any` over every predicate axis,
/// and the masked body replays the original body for candidate updates, selects per state element between the
/// candidate and the carried state under the (broadcast) mask, recomputes the per-item predicate on the new state,
/// and ANDs it into the mask. The bounded while forward-mode rule uses this normal form for batched-predicate loops,
/// whose counter- and stack-augmented differentiation state is not predicate-prefixed and therefore needs the loop's
/// masking made explicit as program data hanging off a scalar predicate.
fn masked_while_programs<V, O>(
    condition: &Program<V, O, Vec<V>, Vec<V>>,
    body: &Program<V, O, Vec<V>, Vec<V>>,
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Program<V, O, Vec<V>, Vec<V>>), ProgramError>
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType>
        + From<ReduceOperation>
        + From<SelectOperation>
        + From<AndOperation>
        + From<BroadcastOperation>,
{
    let state_types = body.input_types();
    let state_count = state_types.len();
    let mask_type = condition.output_types()[0].clone();
    let mask_axes: Vec<usize> = (0..mask_type.rank()).collect();
    let mut masked_state_types = state_types.clone();
    masked_state_types.push(mask_type.clone());

    // Masked condition: `any(active_mask)` over every predicate axis, ignoring the state inputs.
    let (_, masked_condition) = TracingContext::<V, O>::trace(
        |inputs| Ok(vec![inputs[state_count].reduce(mask_axes.as_slice(), ReductionKind::Any)]),
        masked_state_types.clone(),
    )?;

    // Masked body: candidate updates from the replayed body, per-element masked selection between the candidate
    // update and the carried state, the per-item predicate recomputed on the new state, and the mask narrowed via
    // AND.
    let (_, masked_body) = TracingContext::<V, O>::trace(
        |inputs| {
            let (mask, state) = inputs.split_last().unwrap();
            let trace_context = mask.context().clone();
            let candidates = body.interpret_in_context(&trace_context, state.to_vec())?;
            check_count!("output", candidates, state_count, ProgramError);
            let mut next_state = Vec::with_capacity(state_count);
            for ((candidate, carried), state_type) in candidates.iter().zip(state).zip(state_types.iter()) {
                // The mask broadcasts to each state element's shape so the selection is per predicate item; a state
                // element already shaped like the mask reuses it directly.
                let element_mask_type = ArrayType::new(DataType::Boolean, state_type.shape().clone());
                let element_mask = if element_mask_type == mask_type {
                    mask.clone()
                } else {
                    mask.broadcast(element_mask_type, mask_axes.as_slice())?
                };
                next_state.push(Select::select(&element_mask, candidate, carried)?);
            }
            let next_predicate = condition.interpret_in_context(&trace_context, next_state.clone())?;
            check_count!("output", next_predicate, 1, ProgramError);
            let mut outputs = next_state;
            outputs.push(mask.clone() & next_predicate.into_iter().next().unwrap());
            Ok(outputs)
        },
        masked_state_types,
    )?;
    Ok((masked_condition, masked_body))
}

impl<F, C> InterpretableOperation<C> for ConditionOperation<F, Captured>
where
    C: Domain<Type = ArrayType, Value: BooleanLike>,
    F: Value<Type = ArrayType> + CustomVjpResidual<C::Value>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let predicate = self.predicate().residual_value()?;
        driver.interpret_region(context, if predicate.boolean()? { 0 } else { 1 }, inputs.to_vec())
    }
}

/// Transpose rule for the captured-predicate conditional. The predicate is a residual of the primal computation rather
/// than a linear operand, so it has no cotangent and is carried verbatim into a transposed condition over the
/// transposed branch programs, selected by the same predicate. Branch transposition goes through
/// the instruction-scoped driver, keeping the recursion owned by the transposition driver.
impl<V, F, O> TransposableOperation<V, O> for ConditionOperation<F, Captured>
where
    V: Value<Type = ArrayType>,
    F: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<ConditionOperation<F, Captured>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        // The rule operates on the attached branch regions through its driver (region 0 is the `true` branch and
        // region 1 the `false` branch).
        let true_branch = driver.region(0)?;
        // A condition with no outputs (or only zero output cotangents) is a zero linear map, so every input
        // cotangent is zero. Note that `all` is trivially true for an empty cotangent slice.
        if outputs.iter().all(MaybeZero::is_zero) {
            return Ok(true_branch
                .input_types()
                .into_iter()
                .map(|input_type| MaybeZero::Zero(input_type.clone()))
                .collect());
        }
        let transposed_condition = ConditionOperation::new_captured(self.predicate().clone());
        let branch_linear = vec![true; true_branch.input_ids().len()];
        let transposed_true = driver.transpose_program(true_branch, branch_linear.as_slice())?;
        let transposed_false = driver.transpose_program(driver.region(1)?, branch_linear.as_slice())?;
        let materialized = outputs
            .iter()
            .map(|cotangent| cotangent.clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        let cotangents =
            context.bind(transposed_condition, vec![transposed_true, transposed_false], materialized.as_slice())?;
        check_count!("output", cotangents, true_branch.input_types().len(), ProgramError);
        Ok(cotangents.into_iter().map(MaybeZero::Value).collect())
    }
}

/// Partition-aware transpose rule for a *primal* input-predicate [`ConditionOperation`], forwarding to
/// [`transpose_primal_condition`]. The predicate and the per-branch residuals ride as ordinary known operands, and the
/// branch recursion happens through the instruction-scoped driver's transposition requests, so instantiating this
/// implementation for a closed operation enum introduces no recursive [`TransposableOperation`] obligation on `O`.
impl<V, O> TransposableOperation<V, O> for ConditionOperation<V, Input>
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<ConditionOperation<V>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        transpose_primal_condition(context, driver, inputs, outputs).map_err(DifferentiationError::from)
    }
}

#[cfg(test)]
mod tests {
    use crate::regions::RegionInterface;
    use std::borrow::Cow;

    use crate::macros::check_types;
    use std::fmt::Display;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_macros::Parameter;

    use crate::backends::scalars::Scalar;
    use crate::contexts::{Context, Domain, EagerContext, StagingContext};
    use crate::interpretation::InterpretableOperation;
    use crate::operations::compare::CompareOperation;
    use crate::operations::constants::{One, OneLike, OneLikeOperation, Zero, ZeroLike, ZeroLikeOperation};
    use crate::operations::math::{ADD_OPERATION_NAME, AddOperation, MulOperation, SUB_OPERATION_NAME, SubOperation};
    use crate::parameters::{Parameter, Placeholder};
    use crate::programs::{Program, ProgramBuilder, Value};
    use crate::tracing::DomainTracingContext;
    use crate::tracing_v2::operations::reduce::ReduceOperation;
    use crate::tracing_v2::{ArrayOperation, ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::types::{DataType, Shape, Size, TypeError};

    use super::*;
    use crate::batching::BatchAxis;

    #[derive(Clone, Debug, Parameter, PartialEq)]
    enum TestValue {
        Bool(bool),
        Number(f64),
    }

    impl Display for TestValue {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Bool(value) => Display::fmt(value, formatter),
                Self::Number(value) => Display::fmt(value, formatter),
            }
        }
    }

    impl Typed for TestValue {
        type Type = ArrayType;

        fn r#type(&self) -> Cow<'_, ArrayType> {
            match self {
                Self::Bool(_) => Cow::Owned(ArrayType::scalar(DataType::Boolean)),
                Self::Number(_) => Cow::Owned(ArrayType::scalar(DataType::F64)),
            }
        }
    }

    impl Value for TestValue {
        type DispatchDomain = EagerContext<Self>;
        type ExecutionDomain = EagerContext<Self>;

        fn dispatch_domain(&self) -> EagerContext<Self> {
            EagerContext::new()
        }

        fn execution_domain(&self) -> EagerContext<Self> {
            EagerContext::new()
        }
    }

    impl ZeroLike for TestValue {
        fn zero_like(&self) -> Self {
            match self {
                Self::Bool(_) => Self::Bool(false),
                Self::Number(_) => Self::Number(0.0),
            }
        }
    }

    impl OneLike for TestValue {
        fn one_like(&self) -> Self {
            match self {
                Self::Bool(_) => Self::Bool(true),
                Self::Number(_) => Self::Number(1.0),
            }
        }
    }

    impl<O: Operation<ArrayType>> Zero<TestValue> for EagerContext<TestValue, O> {
        fn zero(&self, value_type: &ArrayType) -> Result<TestValue, ProgramError> {
            match value_type.data_type() {
                DataType::Boolean => Ok(TestValue::Bool(false)),
                DataType::F64 => Ok(TestValue::Number(0.0)),
                _ => Err(crate::types::TypeError {
                    message: format!("test value cannot synthesize zero for {value_type}"),
                }
                .into()),
            }
        }
    }

    impl<O: Operation<ArrayType>> One<TestValue> for EagerContext<TestValue, O> {
        fn one(&self, value_type: &ArrayType) -> Result<TestValue, ProgramError> {
            match value_type.data_type() {
                DataType::Boolean => Ok(TestValue::Bool(true)),
                DataType::F64 => Ok(TestValue::Number(1.0)),
                _ => Err(crate::types::TypeError {
                    message: format!("test value cannot synthesize one for {value_type}"),
                }
                .into()),
            }
        }
    }

    impl BooleanLike for TestValue {
        fn as_boolean(&self) -> Self {
            match self {
                Self::Bool(value) => Self::Bool(*value),
                Self::Number(value) => Self::Bool(*value != 0.0),
            }
        }

        fn boolean(&self) -> Result<bool, ProgramError> {
            match self {
                Self::Bool(value) => Ok(*value),
                value => Err(ProgramError::Concretization {
                    message: format!(
                        "cannot extract a concrete boolean from a value of type {}; expected bool[]",
                        value.r#type(),
                    ),
                }),
            }
        }
    }

    // `TestValue` predicates are scalar, so the scalar `WhilePredicate` defaults apply.
    impl crate::operations::control_flow::WhilePredicate for TestValue {}

    #[derive(Clone, Debug)]
    enum TestOperation {
        Add,
        Sub,
        IsPositive,
        Condition(ConditionOperation<TestValue>),
        While(WhileOperation),
    }

    impl Display for TestOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.name())
        }
    }

    impl Operation<ArrayType> for TestOperation {
        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Add => ADD_OPERATION_NAME,
                Self::Sub => SUB_OPERATION_NAME,
                Self::IsPositive => "is_positive",
                Self::Condition(condition) => condition.name(),
                Self::While(while_operation) => Operation::<ArrayType>::name(while_operation),
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayType],
            region_interfaces: &[RegionInterface<ArrayType>],
        ) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Add | Self::Sub => {
                    check_count!("input", input_types, 2, TypeError);
                    check_types!(self.name(), &input_types[..1], &input_types[1..]);
                    Ok(vec![input_types[0].clone()])
                }
                Self::IsPositive => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![ArrayType::scalar(DataType::Boolean)])
                }
                Self::Condition(condition) => condition.infer_output_types(input_types, region_interfaces),
                Self::While(while_operation) => while_operation.infer_output_types(input_types, region_interfaces),
            }
        }

        fn region_names(&self) -> &'static [&'static str] {
            match self {
                Self::Condition(condition) => condition.region_names(),
                Self::While(while_operation) => Operation::<ArrayType>::region_names(while_operation),
                _ => &[],
            }
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Condition(condition) => condition.render(formatter, indentation),
                Self::While(while_operation) => Operation::<ArrayType>::render(while_operation, formatter, indentation),
                _ => Display::fmt(self, formatter),
            }
        }
    }

    impl<C: Domain<Type = ArrayType, Value = TestValue>> InterpretableOperation<C> for TestOperation
    where
        C: crate::operations::constants::Constant<TestValue, TestValue>,
    {
        fn interpret<D: InterpretationDriver<C>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[TestValue],
        ) -> Result<Vec<TestValue>, ProgramError> {
            match self {
                Self::Add => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left + right)]),
                    _ => Err(TypeError { message: ("'add' expected numeric inputs").into() }.into()),
                },
                Self::Sub => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left - right)]),
                    _ => Err(TypeError { message: ("sub expected numeric inputs").into() }.into()),
                },
                Self::IsPositive => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Bool(*value > 0.0)]),
                    _ => Err(TypeError { message: ("is_positive expected a numeric input").into() }.into()),
                },
                Self::Condition(condition) => condition.interpret(context, driver, inputs),
                Self::While(while_operation) => while_operation.interpret(context, driver, inputs),
            }
        }
    }

    fn add_one_branch() -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_constant(TestValue::Number(1.0));
        let output = builder.add_instruction(TestOperation::Add, Vec::new(), vec![input, one]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn subtract_one_branch() -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_constant(TestValue::Number(1.0));
        let output = builder.add_instruction(TestOperation::Sub, Vec::new(), vec![input, one]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn identity_array_branch() -> Program<TestValue, ArrayOperation<TestValue>, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, ArrayOperation<TestValue>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_condition_interprets_true_and_false_branches() {
        let condition_regions = vec![add_one_branch(), subtract_one_branch()];
        let condition = ConditionOperation::<TestValue>::new();
        let context = EagerContext::<TestValue, TestOperation>::new();

        assert_eq!(
            context.bind(
                TestOperation::Condition(condition.clone()),
                condition_regions.clone(),
                &[TestValue::Bool(true), TestValue::Number(3.0)],
            ),
            Ok(vec![TestValue::Number(4.0)]),
        );
        assert_eq!(
            context.bind(
                TestOperation::Condition(condition),
                condition_regions,
                &[TestValue::Bool(false), TestValue::Number(3.0)],
            ),
            Ok(vec![TestValue::Number(2.0)]),
        );
    }

    #[test]
    fn test_condition_program_rendering_includes_nested_branches() {
        let condition_regions = vec![add_one_branch(), subtract_one_branch()];
        let condition = ConditionOperation::new();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let regions = condition_regions
            .iter()
            .map(|region| builder.import_region(region.entry_region_ref()))
            .collect::<Vec<_>>();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(TestOperation::Condition(condition), regions, vec![predicate, input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition %0 %1 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = const
                            %2:f64[] = add %0 %1
                        in (%2)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = const
                            %2:f64[] = sub %0 %1
                        in (%2)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_condition_rejects_branch_output_mismatch() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestOperation::IsPositive, Vec::new(), vec![input]).unwrap()[0];
        let bool_branch = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let condition = ConditionOperation::<TestValue>::new();
        let branch_interface =
            |program: &Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>| program.interface();
        assert!(
            condition
                .infer_output_types(
                    &[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F64)],
                    &[branch_interface(&add_one_branch()), branch_interface(&bool_branch)],
                )
                .is_err()
        );
    }

    #[test]
    fn test_while_interprets_until_condition_is_false() {
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition_input = condition_builder.add_input(ArrayType::scalar(DataType::F64));
        let condition_output = condition_builder
            .add_instruction(TestOperation::IsPositive, Vec::new(), vec![condition_input])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new();

        assert_eq!(
            EagerContext::<TestValue, TestOperation>::new().bind(
                TestOperation::While(while_operation),
                vec![condition, subtract_one_branch()],
                &[TestValue::Number(3.0)],
            ),
            Ok(vec![TestValue::Number(0.0)]),
        );
    }

    #[test]
    fn test_while_program_rendering_includes_condition_and_body() {
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition_input = condition_builder.add_input(ArrayType::scalar(DataType::F64));
        let condition_output = condition_builder
            .add_instruction(TestOperation::IsPositive, Vec::new(), vec![condition_input])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(subtract_one_branch().entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(TestOperation::While(while_operation), vec![condition_region, body_region], vec![input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = while %0 [
                    condition={
                        lambda %0:f64[] .
                        let %1:bool[] = is_positive %0
                        in (%1)
                    },
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = const
                            %2:f64[] = sub %0 %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_operation_condition_infers_output_types() {
        let condition_regions = vec![identity_array_branch(), identity_array_branch()];
        let condition = ConditionOperation::<TestArray>::new();
        let operation = ArrayOperation::Condition(condition);
        let region_interfaces = condition_regions.iter().map(Program::interface).collect::<Vec<_>>();

        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F64)],
                region_interfaces.as_slice(),
            ),
            Ok(vec![ArrayType::scalar(DataType::F64)]),
        );
    }

    use crate::operations::compare::ComparisonDirection;
    use crate::tests::TestArray;

    /// Test array operation enum used by the while tests below.
    type TestArrayOperation = ArrayOperation<TestArray>;

    /// Eager interpreting domain over [`TestArray`] values that reports no support for primal concretization. Hybrid
    /// rules (in particular the while JVP rule) therefore take their staged, non-concretizing paths while every
    /// primal bind still computes concrete values, which lets the tests below interpret linear while bodies
    /// numerically without abstract tracers.
    #[derive(Copy, Clone, Debug)]
    struct StagedDispatchTestArrayDomain;

    impl Domain for StagedDispatchTestArrayDomain {
        type Type = ArrayType;
        type Value = TestArray;
        type Constant = TestArray;
        type Operation = TestArrayOperation;
    }

    impl Context for StagedDispatchTestArrayDomain {
        fn lift(&self, constant: TestArray) -> Result<TestArray, ProgramError> {
            Ok(constant)
        }

        fn bind<P: Into<Self::Operation>, D: crate::BindingRegionDriver<Self::Constant, Self::Operation>>(
            &self,
            operation: P,
            driver: D,
            inputs: &[Self::Value],
        ) -> Result<Vec<Self::Value>, ProgramError> {
            // Region-carrying binds route through the eager context's own bind, which grants application-scoped region
            // access.
            crate::EagerContext::<TestArray, Self::Operation>::new().bind(operation, driver, inputs)
        }

        fn resolve(&self, value: &TestArray) -> crate::ValueResolution<TestArray> {
            crate::ValueResolution::Concrete(value.clone())
        }

        fn is_eager(&self) -> bool {
            false
        }
    }

    /// Eager-domain context capabilities, delegating to the zero-state [`crate::EagerContext`] exactly like
    /// [`EagerContext<TestArray, ArrayOperation<TestArray>>`](crate::tests::EagerContext<TestArray, ArrayOperation<TestArray>>)'s.
    impl crate::operations::constants::Zero<TestArray> for StagedDispatchTestArrayDomain {
        fn zero(&self, r#type: &ArrayType) -> Result<TestArray, ProgramError> {
            crate::operations::constants::Zero::zero(&crate::EagerContext::<TestArray>::new(), r#type)
        }
    }

    impl crate::operations::constants::One<TestArray> for StagedDispatchTestArrayDomain {
        fn one(&self, r#type: &ArrayType) -> Result<TestArray, ProgramError> {
            crate::operations::constants::One::one(&crate::EagerContext::<TestArray>::new(), r#type)
        }
    }

    impl crate::operations::constants::Fill<Scalar, TestArray> for StagedDispatchTestArrayDomain {
        fn fill(&self, r#type: &ArrayType, value: Scalar) -> Result<TestArray, ProgramError> {
            crate::operations::constants::Fill::fill(&crate::EagerContext::<TestArray>::new(), r#type, value)
        }
    }

    impl crate::operations::constants::Iota<TestArray> for StagedDispatchTestArrayDomain {
        fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<TestArray, ProgramError> {
            crate::operations::constants::Iota::iota(&crate::EagerContext::<TestArray>::new(), r#type, dimension)
        }
    }

    impl<Payload> crate::operations::constants::Constant<TestArray, TestArray, Payload> for StagedDispatchTestArrayDomain {
        fn constant(&self, value: TestArray) -> Result<TestArray, ProgramError> {
            Ok(value)
        }
    }

    /// Builds the `state < threshold` while condition program over one scalar state element.
    fn scalar_threshold_condition(
        threshold: f64,
    ) -> Program<TestArray, TestArrayOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let threshold = builder.add_constant(TestArray::scalar(threshold));
        let predicate = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![state, threshold])
            .unwrap()[0];
        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the `while (x < threshold) { x = 2 * x }` loop with the provided semantic iteration bound.
    fn bounded_doubling_while_operation(
        threshold: f64,
        bound: usize,
    ) -> (WhileOperation, Vec<Program<TestArray, TestArrayOperation, Vec<TestArray>, Vec<TestArray>>>) {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let state = builder.add_input(scalar_f64);
        let two = builder.add_constant(TestArray::scalar(2.0));
        let doubled = builder.add_instruction(MulOperation, Vec::new(), vec![state, two]).unwrap()[0];
        let body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let operation = WhileOperation::new().with_iteration_bound(bound).unwrap();
        (operation, vec![scalar_threshold_condition(threshold), body])
    }

    /// Builds the `while (x < threshold) { x = x * x }` loop with the provided semantic iteration bound. Squaring
    /// captures the loop state itself as a loop-varying residual, so differentiating this loop exercises the
    /// per-iteration residual stacks of the bounded staged path.
    fn bounded_squaring_while_operation(
        threshold: f64,
        bound: usize,
    ) -> (WhileOperation, Vec<Program<TestArray, TestArrayOperation, Vec<TestArray>, Vec<TestArray>>>) {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let state = builder.add_input(scalar_f64);
        let squared = builder.add_instruction(MulOperation, Vec::new(), vec![state, state]).unwrap()[0];
        let body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let operation = WhileOperation::new().with_iteration_bound(bound).unwrap();
        (operation, vec![scalar_threshold_condition(threshold), body])
    }

    #[test]
    fn test_bounded_while_value_and_grad_computes_gradient_through_staged_masked_scan() {
        // The headline bounded-while capability: end-to-end reverse mode through a *staged* while loop.
        // `f(x) = while (x < 8, iteration_bound = 5) { x = 2 * x }` at `x = 1` runs three iterations (`x` visits 1,
        // 2, 4), so the actual trip count 3 is strictly below the bound 5 and the two trailing batch items matter:
        // their mask entries are false, so they must pass tangents through unchanged in the forward scan and cotangents
        // through unchanged in the transposed scan. Locally `f(x) = 8 x`: value 8, gradient 8.
        let (while_operation, while_regions) = bounded_doubling_while_operation(8.0, 5);
        let (output, pullback) = StagedDispatchTestArrayDomain
            .vjp(
                move |x| {
                    let mut outputs = x.context().bind(
                        TestArrayOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.values, vec![8.0]);

        // The pullback contains the transposed (reversed) linear scan and no while loop, and every cotangent seed
        // scales the hand-computed gradient 8. The direct-transpose pullback consumes `[cotangent ++ residuals]`.
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("scan"), "{rendered_pullback}");
        assert!(rendered_pullback.contains("reverse=true"), "{rendered_pullback}");
        assert!(!rendered_pullback.contains("while"), "{rendered_pullback}");
        let pullback_inputs = |cotangent: TestArray| {
            let mut inputs = vec![cotangent];
            inputs.extend(residuals.iter().cloned());
            inputs
        };
        assert_eq!(
            pullback
                .interpret(pullback_inputs(TestArray::scalar(1.0)))
                .map(|cotangents| cotangents[0].values.clone()),
            Ok(vec![8.0]),
        );
        assert_eq!(
            pullback
                .interpret(pullback_inputs(TestArray::scalar(2.0)))
                .map(|cotangents| cotangents[0].values.clone()),
            Ok(vec![16.0]),
        );

        // `value_and_gradient` composes the same machinery end to end.
        let (while_operation, while_regions) = bounded_doubling_while_operation(8.0, 5);
        let (value, gradient) = StagedDispatchTestArrayDomain
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .bind(TestArrayOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![8.0]);
        assert_eq!(gradient.values, vec![8.0]);
    }

    #[test]
    fn test_bounded_while_value_and_grad_stores_loop_varying_residual_stacks() {
        // The store-instead-of-recompute proof: `while (x < 100, iteration_bound = 4) { x = x * x }` at `x = 2`
        // squares three times (`x` visits 2, 4, 16 → 256, trip count 3 < bound 4), and the product rule references
        // the *per-iteration* state as a loop-varying residual, so the gradient depends on the stored stack batch
        // items `[2, 4, 16, 0]` — including the zero batch item beyond the trip count, which the mask must keep inert
        // in both directions. Locally `f(x) = x⁸`: value 256 and gradient `8 x⁷ = 1024`.
        let (while_operation, while_regions) = bounded_squaring_while_operation(100.0, 4);
        let (output, pullback) = StagedDispatchTestArrayDomain
            .vjp(
                move |x| {
                    let mut outputs = x.context().bind(
                        TestArrayOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(2.0),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.values, vec![256.0]);
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("reverse=true"), "{rendered_pullback}");
        let mut pullback_inputs = vec![TestArray::scalar(1.0)];
        pullback_inputs.extend(residuals);
        assert_eq!(
            pullback.interpret(pullback_inputs).map(|cotangents| cotangents[0].values.clone()),
            Ok(vec![1024.0]),
        );

        // The eager-domain reverse-mode entry point produces the same value and gradient numbers.
        let (while_operation, while_regions) = bounded_squaring_while_operation(100.0, 4);
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .bind(TestArrayOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(2.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![256.0]);
        assert_eq!(gradient.values, vec![1024.0]);
    }

    #[test]
    fn test_bounded_while_value_and_grad_supports_vector_state() {
        // Vector-state coverage for the bounded staged path: the residual stacks gain trailing axes (written at
        // `[counter, 0]` through the staged zero index) and the per-item select conditions come from a broadcast of
        // the Boolean `[bound]` mask stack to `[bound, 2]`, staged outside the loop. The loop
        // `while (sum(x) < 20, iteration_bound = 4) { x = x * x }` at `x = [1.5, 2]` squares twice (sums visit 3.5
        // and 6.25 before reaching 21.0625), so `f(x) = sum(x⁴)` locally: value `1.5⁴ + 2⁴ = 21.0625` and gradient
        // `4 x³ = [13.5, 32]`, with trip count 2 strictly below the bound 4.
        use crate::tracing_v2::operations::reduce::ReductionKind;

        let vector_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]));
        let mut condition_builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let condition_state = condition_builder.add_input(vector_f64.clone());
        let summed = condition_builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![condition_state])
            .unwrap()[0];
        let threshold = condition_builder.add_constant(TestArray::scalar(20.0));
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![summed, threshold])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let body_state = body_builder.add_input(vector_f64.clone());
        let squared = body_builder.add_instruction(MulOperation, Vec::new(), vec![body_state, body_state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new().with_iteration_bound(4).unwrap();
        let while_regions = vec![condition, body];

        let (value, gradient) = StagedDispatchTestArrayDomain
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .bind(TestArrayOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                        .unwrap();
                    let state = outputs.remove(0);
                    let mut outputs = state
                        .context()
                        .bind(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), &[state.clone()])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::vector(vec![1.5, 2.0]),
            )
            .unwrap();
        assert_eq!(value.values, vec![21.0625]);
        assert_eq!(gradient.values, vec![13.5, 32.0]);
    }

    #[test]
    fn test_bounded_while_eager_value_and_grad_matches_staged_numbers() {
        // The eager-domain entry point differentiates the same bounded loop to identical numbers: the loop exits
        // through its condition after three iterations, well below the bound of five.
        let (while_operation, while_regions) = bounded_doubling_while_operation(8.0, 5);
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .bind(TestArrayOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![8.0]);
        assert_eq!(gradient.values, vec![8.0]);
    }

    #[test]
    fn test_bounded_while_truncation_differentiates_consistently_across_paths() {
        // A loop whose condition never turns false truncates at the bound by definition: with bound 3 the doubling
        // loop computes `f(x) = 8 x`, so at `x = 2` the value is 16 and the gradient is 8 — identical between plain
        // interpretation, the eager-domain entry point, and the staged dispatch domain (where every mask batch
        // item is true).
        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        let outputs = crate::EagerContext::<TestArray, TestArrayOperation>::new()
            .bind(TestArrayOperation::While(while_operation), while_regions, &[TestArray::scalar(2.0)])
            .unwrap();
        assert_eq!(outputs[0].values, vec![16.0]);

        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .bind(TestArrayOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(2.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![16.0]);
        assert_eq!(gradient.values, vec![8.0]);

        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        let (value, gradient) = StagedDispatchTestArrayDomain
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .bind(TestArrayOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(2.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![16.0]);
        assert_eq!(gradient.values, vec![8.0]);
    }

    /// Builds the per-item countdown loop `while (x > 0) { x = x - 1 }` over one scalar state element.
    fn countdown_while_operation()
    -> (WhileOperation, Vec<Program<TestArray, TestArrayOperation, Vec<TestArray>, Vec<TestArray>>>) {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let zero = condition_builder.add_instruction(ZeroLikeOperation, Vec::new(), vec![condition_state]).unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![condition_state, zero],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let body_state = body_builder.add_input(scalar_f64);
        let one = body_builder.add_instruction(OneLikeOperation, Vec::new(), vec![body_state]).unwrap()[0];
        let next = body_builder.add_instruction(SubOperation, Vec::new(), vec![body_state, one]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![next], vec![Placeholder], vec![Placeholder])
            .unwrap();
        (WhileOperation::new(), vec![condition, body])
    }

    /// Stages `while_operation` over one batched item (mapped at axis 0 with `batch_size` batch items) under tracing
    /// and returns the staged batched program for structural and numeric assertions.
    fn batch_while_under_tracing(
        while_operation: WhileOperation,
        while_regions: Vec<Program<TestArray, TestArrayOperation, Vec<TestArray>, Vec<TestArray>>>,
        batch_size: usize,
    ) -> Program<TestArray, TestArrayOperation, TestArray, TestArray> {
        use crate::batching::Batch;
        let parent = DomainTracingContext::<EagerContext<TestArray, ArrayOperation<TestArray>>>::new();
        let builder = parent.builder().clone();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(batch_size)]));
        let input_atom = builder.borrow_mut().add_input(input_type);
        let input_tracer = parent.tracer(input_atom, None);
        let output = Batch::batch(
            &parent,
            |item| {
                let mut outputs = item.context().bind(
                    TestArrayOperation::While(while_operation),
                    while_regions.clone(),
                    &[item.clone()],
                )?;
                Ok(outputs.remove(0))
            },
            input_tracer,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        builder
            .borrow()
            .clone()
            .build::<TestArray, TestArray>(vec![output_atom], Placeholder, Placeholder)
            .unwrap()
    }

    #[test]
    fn test_batch_stages_batched_predicate_while_for_batch_varying_predicates_under_tracing() {
        // vmap-under-tracing of the per-item countdown loop: the predicate `x > 0` is per batch item, so the staged
        // batching rule stages exactly one `while` whose condition returns the batched `bool[3]` predicate directly
        // (the relaxed predicate contract, mirroring JAX's `_while_loop_batching_rule`) instead of unrolling (the
        // body's single `sub` appears exactly once in the staged trace) and without building any masking program data
        // (no `reduce_any` in the staged form; interpretation and lowering own the masked semantics). Batch items
        // [3, 1, 2] terminate after 3, 1, and 2 iterations, and inactive batch items carry their final state,
        // matching the eager operational path batch item for batch item.
        let (countdown_operation, countdown_regions) = countdown_while_operation();
        let program = batch_while_under_tracing(countdown_operation, countdown_regions, 3);
        let rendered = program.to_string();
        assert_eq!(rendered.matches("= while").count(), 1, "{rendered}");
        assert!(!rendered.contains("reduce_any"), "{rendered}");
        assert!(rendered.contains("%2:bool[3] = compare"), "{rendered}");
        assert_eq!(rendered.matches("sub").count(), 1, "{rendered}");
        let output = program.interpret(TestArray::vector(vec![3.0, 1.0, 2.0])).unwrap();
        assert_eq!(output.values, vec![0.0, 0.0, 0.0]);

        // The semantic iteration bound is preserved on the staged batched-predicate while: every batch item performs
        // at most two body applications, so batch item 0 truncates at 1.0 — the numbers of the eager operational
        // bounded path.
        let (countdown_operation, countdown_regions) = countdown_while_operation();
        let program =
            batch_while_under_tracing(countdown_operation.with_iteration_bound(2).unwrap(), countdown_regions, 3);
        let rendered = program.to_string();
        assert!(rendered.contains("iteration_bound=2"), "{rendered}");
        let output = program.interpret(TestArray::vector(vec![3.0, 1.0, 2.0])).unwrap();
        assert_eq!(output.values, vec![1.0, 0.0, 0.0]);
    }

    /// Builds the `while (counter > 0) { (counter, value) = (counter - 1, value + value) }` loop whose predicate
    /// depends only on the counter state element.
    fn counter_doubling_while_operation()
    -> (WhileOperation, Vec<Program<TestArray, TestArrayOperation, Vec<TestArray>, Vec<TestArray>>>) {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let condition_counter = condition_builder.add_input(scalar_f64.clone());
        condition_builder.add_input(scalar_f64.clone());
        let zero =
            condition_builder.add_instruction(ZeroLikeOperation, Vec::new(), vec![condition_counter]).unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![condition_counter, zero],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let body_counter = body_builder.add_input(scalar_f64.clone());
        let body_value = body_builder.add_input(scalar_f64);
        let one = body_builder.add_instruction(OneLikeOperation, Vec::new(), vec![body_counter]).unwrap()[0];
        let next_counter = body_builder.add_instruction(SubOperation, Vec::new(), vec![body_counter, one]).unwrap()[0];
        let doubled = body_builder.add_instruction(AddOperation, Vec::new(), vec![body_value, body_value]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![next_counter, doubled],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        (WhileOperation::new(), vec![condition, body])
    }

    #[test]
    fn test_batch_stages_plain_while_for_replicated_predicates_under_tracing() {
        use crate::batching::Batch;

        // vmap-under-tracing of a loop whose predicate depends only on a replicated counter: the staged batching
        // rule batches the condition and body at the state batch axes and stages one plain `while` — no mask
        // machinery (`reduce_any` / per-element `select`) appears in the staged program. Two iterations double the
        // batched value twice: [1, 2, 3] -> [4, 8, 12], with the replicated counter ending at 0.
        let parent = DomainTracingContext::<EagerContext<TestArray, ArrayOperation<TestArray>>>::new();
        let builder = parent.builder().clone();
        let counter_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let value_atom =
            builder.borrow_mut().add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])));
        let counter_tracer = parent.tracer(counter_atom, None);
        let value_tracer = parent.tracer(value_atom, None);
        let (counter_output, value_output) = Batch::batch(
            &parent,
            |(counter, value)| {
                let (while_operation, while_regions) = counter_doubling_while_operation();
                let mut outputs = counter.context().bind(
                    TestArrayOperation::While(while_operation),
                    while_regions,
                    &[counter.clone(), value.clone()],
                )?;
                let value_output = outputs.remove(1);
                Ok((outputs.remove(0), value_output))
            },
            (counter_tracer, value_tracer),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            None,
        )
        .unwrap();
        let output_atoms = vec![counter_output.atom_id().unwrap(), value_output.atom_id().unwrap()];
        let program = builder
            .borrow()
            .clone()
            .build::<(TestArray, TestArray), (TestArray, TestArray)>(
                output_atoms,
                (Placeholder, Placeholder),
                (Placeholder, Placeholder),
            )
            .unwrap();
        let rendered = program.to_string();
        assert_eq!(rendered.matches("= while").count(), 1, "{rendered}");
        assert!(!rendered.contains("reduce_any"), "{rendered}");
        assert!(!rendered.contains("select"), "{rendered}");
        let (counter_output, value_output) =
            program.interpret((TestArray::scalar(2.0), TestArray::vector(vec![1.0, 2.0, 3.0]))).unwrap();
        assert_eq!(counter_output.values, vec![0.0]);
        assert_eq!(value_output.values, vec![4.0, 8.0, 12.0]);
    }

    #[test]
    fn test_jvp_of_batched_bounded_while_under_tracing_composes_with_masked_scan() {
        use crate::batching::{Batch, BatchableOperation, BatchingTracer};

        // F5 x F6 composition: jvp of a *vmapped bounded* while under the non-concretizing staged dispatch domain.
        // Batching stages one masked bounded while (the predicate `x < 8` is per batch item and the iteration bound 5
        // survives the staged rewrite), so the while JVP rule takes the bounded staged path: stored residual
        // stacks plus a masked linear scan on the tangent side. Batch items [1, 5, 9] double 3, 1, and 0 times, so the
        // primal is [8, 10, 9] and the per-item tangent scale is 2^iterations = [8, 2, 1].
        fn batched_bounded_while<V>(x: V) -> Result<V, ProgramError>
        where
            V: Value<Type = ArrayType> + crate::operations::manipulation::Transpose,
            V::DispatchDomain:
                Context<Type = ArrayType, Value = V, Constant = TestArray, Operation = TestArrayOperation>,
            TestArrayOperation: BatchableOperation<V::DispatchDomain>
                + crate::batching::BatchableOperation<
                    crate::TracingContext<
                        <V::DispatchDomain as crate::Domain>::Constant,
                        <V::DispatchDomain as crate::Domain>::Operation,
                    >,
                > + From<crate::operations::manipulation::TransposeOperation>
                + From<crate::operations::manipulation::BroadcastOperation>,
        {
            let context = x.dispatch_domain();
            let mapped = Batch::batch(
                &context,
                |item: BatchingTracer<V::DispatchDomain>| {
                    let batching_context = item.context().clone();
                    let (while_operation, while_regions) = bounded_doubling_while_operation(8.0, 5);
                    let mut outputs = batching_context.bind(while_operation, while_regions, &[item])?;
                    Ok(outputs.remove(0))
                },
                x,
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )?;
            Ok(mapped)
        }
        let (primal, tangent) = StagedDispatchTestArrayDomain
            .jvp(batched_bounded_while, TestArray::vector(vec![1.0, 5.0, 9.0]), TestArray::vector(vec![1.0, 1.0, 1.0]))
            .unwrap();
        assert_eq!(primal.values, vec![8.0, 10.0, 9.0]);
        assert_eq!(tangent.values, vec![8.0, 2.0, 1.0]);

        // The plain eager domain produces the same numbers...
        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(batched_bounded_while, TestArray::vector(vec![1.0, 5.0, 9.0]), TestArray::vector(vec![1.0, 1.0, 1.0]))
            .unwrap();
        assert_eq!(primal.values, vec![8.0, 10.0, 9.0]);
        assert_eq!(tangent.values, vec![8.0, 2.0, 1.0]);

        // ... and reverse mode composes through the masked linear scan: the pullback contains the reversed scan
        // and no while loop, and the per-item gradients match the tangent scales.
        let (output, pullback) = StagedDispatchTestArrayDomain
            .vjp(batched_bounded_while, TestArray::vector(vec![1.0, 5.0, 9.0]))
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.values, vec![8.0, 10.0, 9.0]);
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("scan"), "{rendered_pullback}");
        assert!(rendered_pullback.contains("reverse=true"), "{rendered_pullback}");
        assert!(!rendered_pullback.contains("while"), "{rendered_pullback}");
        let mut pullback_inputs = vec![TestArray::vector(vec![1.0, 1.0, 1.0])];
        pullback_inputs.extend(residuals);
        let cotangents = pullback.interpret(pullback_inputs).unwrap();
        assert_eq!(cotangents[0].values, vec![8.0, 2.0, 1.0]);
    }

    #[test]
    fn test_unbounded_while_staged_jvp_reports_unsupported_operation() {
        // Phase 0 boundary pin for the first-class-program-regions plan: an unbounded while loop has no staged
        // forward-mode rule (the eager path unrolls instead), so no while-produced residual can ever reach a staged
        // linearization boundary through this path. The lazy residual-origin design relies on this rejection.
        fn unbounded_while<V>(x: V) -> Result<V, ProgramError>
        where
            V: Value<Type = ArrayType>,
            V::DispatchDomain:
                Context<Type = ArrayType, Value = V, Constant = TestArray, Operation = TestArrayOperation>,
        {
            let context = x.dispatch_domain();
            let (while_operation, while_regions) = countdown_while_operation();
            let mut outputs = context.bind(while_operation, while_regions, &[x])?;
            Ok(outputs.remove(0))
        }
        assert!(matches!(
            StagedDispatchTestArrayDomain.jvp(unbounded_while, TestArray::scalar(4.0), TestArray::scalar(1.0)),
            Err(crate::differentiation::DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message,
            })) if message
                == "operation `while` has no capture-free forward-mode linearization rule unless it carries an \
                    iteration bound; an unbounded while loop has no forward-mode rule",
        ));
    }
}
