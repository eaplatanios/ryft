use crate::batching::BatchingContext;
use crate::batching::BatchingError;
use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchableProgramOperation, ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain, EagerContext, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableProgramOperation, DifferentiableType, LinearizableProgramOperation,
    TransposableOperation, TransposableProgramOperation,
};
use crate::interpretation::{InterpretableOperation, InterpretableProgramOperation};
use crate::macros::check_count;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::{OneOperation, Zero, ZeroOperation};
use crate::operations::control_flow::scan::stacked_scan_type;
use crate::operations::control_flow::{ConditionOperation, ScanOperation, Select, SelectOperation, WhileOperation};
use crate::operations::logical::AndOperation;
use crate::operations::manipulation::{
    Broadcast, BroadcastOperation, DynamicUpdateSliceOperation, Transpose, TransposeOperation,
};
use crate::operations::{BooleanLike, Operation};
use crate::parameters::Placeholder;
use crate::partial::PartialValue;
use crate::payloads::{Captured, Input};
use crate::programs::{Atom, AtomId, Instruction, MaybeZero, Program, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::DifferentiationDual;
use crate::operations::control_flow::MaybeWhile;
use crate::tracing_v2::differentiation::Linearization;
use crate::tracing_v2::operations::custom_derivatives::CustomVjpResidual;
use crate::tracing_v2::operations::reduce::{Reduce, ReduceOperation, ReductionKind};
use crate::tracing_v2::unroll::unroll_concretizable_whiles;
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

impl<V: Value<Type = ArrayType> + BooleanLike> BooleanLike for ArrayBatch<V> {
    /// Returns an [`ArrayBatch`] that wraps the Boolean reinterpretation of the carried value (via the value's own
    /// [`BooleanLike::as_boolean`]) under the same batch axis.
    fn as_boolean(&self) -> Self {
        match self.batch_axis().axis() {
            // This unwrap is safe because `as_boolean` preserves structural metadata, so the batch axis that was
            // valid for this batch remains in bounds for the reinterpreted value.
            Some(axis) => {
                let value = self.value().as_boolean();
                Self::new(value.r#type().into_owned(), value, Some(axis)).unwrap()
            }
            None => Self::replicated(self.value().as_boolean()),
        }
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
///   2. Transposes each branch with
///      [`TransposableProgramOperation::transpose_program`], marking
///      the branch tangent inputs linear and the residual inputs known. Each transposed branch maps
///      `[branch_tangent_output_cotangents..., residuals...]` to `[branch_tangent_input_cotangents...]`; because both
///      branches shared the joined signature, their transposes share it too and form a well-typed condition.
///   3. Re-stages a primal input-predicate [`ConditionOperation`] selecting between the two transposed branches by the
///      same known predicate, over `[predicate, outputs..., residuals...]`. Its outputs are the branch-tangent
///      input cotangents.
///
/// The returned cotangents place those branch-tangent cotangents at the linear-operand positions and a structural
/// [`MaybeZero::Zero`] at the predicate and residual positions, which carry no cotangent. The branch recursion happens
/// through the [`TransposableProgramOperation`] fixed-point witness in the same operation family, so it introduces no
/// recursive [`TransposableOperation`] obligation on `O`.
///
/// # Parameters
///
///   - `operation`: Primal input-predicate condition staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `inputs`: Per-operand [`PartialValue`] knowledge. The [`Unknown`](PartialValue::Unknown) entries are the branch
///     tangents; the [`Known`](PartialValue::Known) entries carry the predicate and residual tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the condition's outputs.
pub fn transpose_primal_condition<V, O>(
    operation: &ConditionOperation<V, O>,
    context: &mut TracingContext<V, O>,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError>
where
    V: Value<Type = ArrayType>,
    O: TransposableProgramOperation<V> + From<ZeroOperation<ArrayType>> + From<ConditionOperation<V, O>>,
{
    // A condition with no live output cotangents is a zero linear map, so every operand cotangent is zero.
    if outputs.iter().all(MaybeZero::is_zero) {
        return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect());
    }

    // Operand layout is `[predicate(known), branch_tangents(linear)..., residuals(known)...]`. The branch tangents are
    // exactly the linear operands, and the residuals are the trailing known operands after the predicate and tangents.
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let branch_input_count = operation.true_branch().input_types().len();
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
    let transposed_true = O::transpose_program(operation.true_branch(), branch_linear.as_slice())?;
    let transposed_false = O::transpose_program(operation.false_branch(), branch_linear.as_slice())?;
    let transposed_condition = ConditionOperation::new(transposed_true, transposed_false)?;

    // Stage the transposed condition over `[predicate, outputs..., residuals...]`. Its outputs are the
    // branch-tangent input cotangents.
    let output_types = operation.true_branch().output_types();
    check_count!("output", outputs, output_types.len(), ProgramError);
    let mut operands = Vec::with_capacity(1 + output_types.len() + residuals.len());
    operands.push(predicate);
    for cotangent in outputs {
        operands.push(cotangent.clone().materialize(context)?);
    }
    operands.extend(residuals);
    let branch_cotangents = context.stage_operation(O::from(transposed_condition), operands.as_slice())?;
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

/// Appends one fresh variable atom to a built `program` by direct program-field extension (every appended atom is a
/// fresh variable, so the [`Program`] invariants that [`ProgramBuilder`] would have established are preserved) and
/// returns its id.
fn append_program_variable<V: Value<Type = ArrayType>, O>(
    program: &mut Program<V, O, Vec<V>, Vec<V>>,
    r#type: ArrayType,
) -> AtomId {
    let id = AtomId::new(program.atoms.len());
    program.atoms.push(Atom::Variable(r#type));
    id
}

/// Appends one instruction with a single fresh output atom to a built `program` by direct program-field extension
/// (the appended instruction reads existing atoms and writes a fresh variable, so the [`Program`] invariants that
/// [`ProgramBuilder`] would have established are preserved) and returns the output id.
fn append_program_instruction<V: Value<Type = ArrayType>, O>(
    program: &mut Program<V, O, Vec<V>, Vec<V>>,
    operation: O,
    inputs: Vec<AtomId>,
    output_type: ArrayType,
) -> AtomId {
    let output = append_program_variable(program, output_type);
    program.instructions.push(Instruction::new(operation, inputs, vec![output]));
    output
}

/// Builds the augmented condition and body programs of the bounded staged while loop (see the [`WhileOperation`] JVP
/// rule below) by direct program-field extension: appended
/// input atoms and instructions reference existing atoms or fresh variables, so every [`Program`] invariant that
/// [`ProgramBuilder`] would have established is preserved.
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
    O: Clone
        + Operation<ArrayType>
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

    // Body: append the extra loop-state inputs, store each residual into its stack at batch index `counter`, mark
    // batch index `counter` valid in the mask stack, and advance the counter.
    let mut body = primal_body.clone();
    let counter_input = append_program_variable(&mut body, counter_type.clone());
    body.input_ids.push(counter_input);
    let stack_inputs = stack_types
        .iter()
        .map(|stack_type| {
            let stack_input = append_program_variable(&mut body, stack_type.clone());
            body.input_ids.push(stack_input);
            stack_input
        })
        .collect::<Vec<_>>();
    let mask_input = append_program_variable(&mut body, mask_stack_type.clone());
    body.input_ids.push(mask_input);
    body.input_structure = vec![Placeholder; body.input_ids.len()];
    let residual_outputs = body.output_ids.split_off(state_count);
    check_count!("output", residual_outputs, residual_types.len(), ProgramError);
    let zero_index = residual_types.iter().any(|residual_type| residual_type.rank() > 0).then(|| {
        append_program_instruction(
            &mut body,
            O::from(ZeroOperation::new(counter_type.clone())),
            vec![],
            counter_type.clone(),
        )
    });
    let mut next_stacks = Vec::with_capacity(stack_types.len());
    for ((residual_output, residual_type), (stack_input, stack_type)) in
        residual_outputs.iter().zip(residual_types).zip(stack_inputs.iter().zip(stack_types.iter()))
    {
        let batch_item_type = stacked_scan_type(residual_type, 1);
        let output_axes = (1..=residual_type.rank()).collect::<Vec<_>>();
        let expanded = append_program_instruction(
            &mut body,
            O::from(BroadcastOperation::new(batch_item_type.clone(), output_axes)),
            vec![*residual_output],
            batch_item_type,
        );
        let mut write_inputs = vec![*stack_input, expanded, counter_input];
        // These unwraps are safe because `zero_index` is staged whenever some residual has rank at least one.
        write_inputs.extend((0..residual_type.rank()).map(|_| zero_index.unwrap()));
        next_stacks.push(append_program_instruction(
            &mut body,
            O::from(DynamicUpdateSliceOperation),
            write_inputs,
            stack_type.clone(),
        ));
    }
    let true_scalar = append_program_instruction(
        &mut body,
        O::from(OneOperation::new(boolean_scalar_type.clone())),
        vec![],
        boolean_scalar_type.clone(),
    );
    let true_item_type = stacked_scan_type(&boolean_scalar_type, 1);
    let true_item = append_program_instruction(
        &mut body,
        O::from(BroadcastOperation::new(true_item_type.clone(), vec![])),
        vec![true_scalar],
        true_item_type,
    );
    let next_mask = append_program_instruction(
        &mut body,
        O::from(DynamicUpdateSliceOperation),
        vec![mask_input, true_item, counter_input],
        mask_stack_type.clone(),
    );
    let one_i64 = append_program_instruction(
        &mut body,
        O::from(OneOperation::new(counter_type.clone())),
        vec![],
        counter_type.clone(),
    );
    let next_counter = append_program_instruction(
        &mut body,
        O::from(AddOperation),
        vec![counter_input, one_i64],
        counter_type.clone(),
    );
    body.output_ids.push(next_counter);
    body.output_ids.extend(next_stacks);
    body.output_ids.push(next_mask);
    body.output_structure = vec![Placeholder; body.output_ids.len()];

    // Condition: the original loop condition extended with ignored extra-state inputs.
    let mut extended_condition = condition.clone();
    let extra_state_types = std::iter::once(counter_type)
        .chain(stack_types.iter().cloned())
        .chain(std::iter::once(mask_stack_type));
    for extra_state_type in extra_state_types {
        let extra_input = append_program_variable(&mut extended_condition, extra_state_type);
        extended_condition.input_ids.push(extra_input);
    }
    extended_condition.input_structure = vec![Placeholder; extended_condition.input_ids.len()];
    Ok((extended_condition, body, stack_types))
}

/// Capture-free forward-mode (JVP) rule for [`ConditionOperation`], staging **one fused** jvp `condition` as an
/// ordinary primal-enum operation over the shared builder.
///
/// The rule builds each branch's fused jvp program through [`DifferentiableProgramOperation::jvp_program`] — both
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
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C>
    for ConditionOperation<C::Constant, C::Operation>
where
    C::Operation: Clone
        + From<ZeroOperation<ArrayType>>
        + From<ConditionOperation<C::Constant, C::Operation>>
        + DifferentiableProgramOperation<C::Constant, C::Operation>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, self.true_branch().input_types().len() + 1, ProgramError);
        let predicate_primal = inputs[0].primal().clone();
        let operands = &inputs[1..];
        let output_count = self.true_branch().output_ids().len();

        // Build both fused jvp branches and stage one fused conditional over the predicate primal followed by the
        // operand primals and tangents.
        let fused_true = C::Operation::jvp_program(self.true_branch())?;
        let fused_false = C::Operation::jvp_program(self.false_branch())?;
        let fused_condition = ConditionOperation::new(fused_true, fused_false)?;
        let mut condition_operands = Vec::with_capacity(2 * operands.len() + 1);
        condition_operands.push(predicate_primal);
        condition_operands.extend(operands.iter().map(|operand| operand.primal().clone()));
        // The fused branches take every operand tangent as a real program input, so materialize structural zeros.
        for operand in operands {
            condition_operands.push(operand.tangent().clone().materialize(context)?);
        }
        let outputs = context.bind(fused_condition, &condition_operands)?;
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
pub(crate) trait WhileJvp<C>: Type
where
    C: Context<Type = Self, Operation: Clone>,
{
    /// Applies the type family's `while` forward-mode rule; refer to the documentation of
    /// [`DifferentiableOperation::jvp`] for the contract.
    fn jvp_while(
        operation: &WhileOperation<C::Constant, C::Operation>,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError>;
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
/// For a bound `B`, the rule linearizes the body capture-free through [`LinearizableProgramOperation`],
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
    C::Constant: Value<Type = ArrayType>,
    C::Operation: Clone
        + From<ZeroOperation<ArrayType>>
        + From<OneOperation<ArrayType>>
        + From<AddOperation>
        + From<BroadcastOperation>
        + From<DynamicUpdateSliceOperation>
        + From<SelectOperation>
        + From<ReduceOperation>
        + From<AndOperation>
        + From<WhileOperation<C::Constant, C::Operation>>
        + From<ScanOperation<C::Constant, C::Operation>>
        + MaybeWhile<C::Constant, C::Operation>
        // The body is linearized through `LinearizableProgramOperation`, while the augmented bounded `while` staged
        // below is itself forward-differentiated via a recursive `WhileOperation::jvp`, which needs the fused
        // `DifferentiableProgramOperation` witness.
        + DifferentiableProgramOperation<C::Constant, C::Operation>
        + LinearizableProgramOperation<C::Constant, C::Operation>,
{
    fn jvp_while(
        operation: &WhileOperation<C::Constant, C::Operation>,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        let state_types = operation.state_types();
        let state_count = state_types.len();
        check_count!("input", inputs, state_count, ProgramError);

        // A batched (per-item) predicate cannot thread the bounded rule's augmented differentiation state through the
        // predicate-prefix contract (the scalar iteration counter and the `[bound, ...]` residual stacks are not
        // predicate-prefixed), so the loop is first rewritten into its scalar-predicate masked normal form over
        // `[state..., active_mask]` (see `masked_while_programs`) and differentiated recursively — the masked loop's
        // forward mode is this same rule. The initial mask is the condition replayed on the operand primals, carried
        // with a zero tangent since a Boolean mask has no derivative.
        let predicate_type = operation.condition().output_types()[0].clone();
        if predicate_type.rank() > 0 {
            let (masked_condition, masked_body) = masked_while_programs(operation.condition(), operation.body())?;
            let masked_while = WhileOperation::<C::Constant, C::Operation>::new(masked_condition, masked_body)?
                .with_iteration_bound(operation.iteration_bound())?;
            let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let mut initial_mask = operation.condition().interpret_in_context(context, primal_operands)?;
            check_count!("output", initial_mask, 1, ProgramError);
            let mut extended_inputs = inputs.to_vec();
            extended_inputs.push(DifferentiationDual::new(initial_mask.remove(0), MaybeZero::Zero(predicate_type)));
            let mut outputs = masked_while.jvp(context, extended_inputs.as_slice())?;
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
                    operation.name(),
                ),
            });
        };

        // Linearize the body capture-free. The primal body produces `[next_state..., residuals...]` and the
        // tangent body consumes `[state_tangent..., residuals...]`; the residual count is the number of trailing
        // outputs of the primal body beyond the loop state.
        let Linearization { primal_program, tangent_program, residual_count, .. } =
            C::Operation::linearize_program(operation.body())?;
        let residual_types = primal_program.output_types().split_off(state_count);

        // Build and bind the augmented primal while over `[state..., counter, residual_stacks..., mask_stack]`, with
        // the counter starting at zero and the stacks (including the Boolean mask, whose zero is false) starting at
        // typed zeros staged in the shared builder.
        let counter_type = ArrayType::scalar(DataType::I64);
        let boolean_scalar_type = ArrayType::scalar(DataType::Boolean);
        let mask_stack_type = stacked_scan_type(&boolean_scalar_type, bound);
        let (extended_condition, augmented_body, stack_types) =
            build_bounded_while_programs(operation.condition(), &primal_program, residual_types.as_slice(), bound)?;
        let augmented_while = WhileOperation::<C::Constant, C::Operation>::new(extended_condition, augmented_body)?
            .with_iteration_bound(bound)?;
        let mut primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let zero_state_types =
            std::iter::once(&counter_type).chain(stack_types.iter()).chain(std::iter::once(&mask_stack_type));
        for zero_state_type in zero_state_types {
            let mut zeros = context.bind(ZeroOperation::new(zero_state_type.clone()), &[])?;
            check_count!("output", zeros, 1, ProgramError);
            primal_operands.push(zeros.remove(0));
        }
        let mut while_outputs = context.bind(C::Operation::from(augmented_while), &primal_operands)?;
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
        let mut scan_body = tangent_program;
        check_count!("input", scan_body.input_ids, state_count + residual_count, ProgramError);
        check_count!("output", scan_body.output_ids, state_count, ProgramError);
        let carried_inputs = scan_body.input_ids[..state_count].to_vec();
        let pushforward_outputs = scan_body.output_ids.clone();
        let mut masked_outputs = Vec::with_capacity(state_count);
        for ((pushforward_output, carried_input), (state_type, &is_differentiable)) in pushforward_outputs
            .into_iter()
            .zip(carried_inputs)
            .zip(state_types.iter().zip(element_is_differentiable.iter()))
        {
            if !is_differentiable {
                masked_outputs.push(pushforward_output);
                continue;
            }
            let condition_type = ArrayType::new(DataType::Boolean, state_type.shape().clone());
            let mask_item = append_program_variable(&mut scan_body, condition_type);
            scan_body.input_ids.push(mask_item);
            let select_output = AtomId::new(scan_body.atoms.len());
            scan_body.atoms.push(Atom::Variable(state_type.clone()));
            scan_body.instructions.push(Instruction::new(
                C::Operation::from(SelectOperation),
                vec![mask_item, pushforward_output, carried_input],
                vec![select_output],
            ));
            masked_outputs.push(select_output);
        }
        scan_body.output_ids = masked_outputs;
        scan_body.input_structure = vec![Placeholder; scan_body.input_ids.len()];
        scan_body.output_structure = vec![Placeholder; state_count];

        // Stage the length-`bound` tangent scan over the carry tangents followed by the stacked residuals and then the
        // per-differentiable-state-element mask stacks. Iteration `item` reads residual slice `item` and mask slice
        // `item`.
        let tangent_scan = ScanOperation::<C::Constant, C::Operation>::new(scan_body, state_count, bound)?;
        // The tangent scan takes every carry tangent as a real program input, so materialize structural zeros.
        let mut tangent_operands = inputs
            .iter()
            .map(|input| input.tangent().clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        tangent_operands.extend(residual_stacks);
        tangent_operands.extend(mask_stacks);
        let tangent_outputs = context.bind(C::Operation::from(tangent_scan), &tangent_operands)?;
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
impl<C: Context<Type = DataType>> WhileJvp<C> for DataType
where
    C::Operation: Clone,
{
    fn jvp_while(
        operation: &WhileOperation<C::Constant, C::Operation>,
        _context: &C,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("operation `{}` has no capture-free forward-mode linearization rule", operation.name()),
        })
    }
}

/// Runs a `while` loop's forward-mode rule directly at concrete duals for an
/// [eager](Context::is_eager) context, returning `None` when the loop's predicate does
/// not concretize to one scalar Boolean (e.g., a batched per-item predicate) and the caller must therefore fall back
/// to the type family's staged strategy.
///
/// Each iteration evaluates the condition on the concrete primal carries, unrolls any nested data-dependent `while`
/// in the body at those carries (through the same value-level rewrite the reverse-mode pre-pass uses), fuses the
/// unrolled body into its JVP program through the [`DifferentiableProgramOperation`] fixed point, and replays that
/// fused program once over `[primal_carries ++ tangent_carries]` to advance both halves. Data-dependent trip counts
/// therefore need no iteration bound — this is the analogue of
/// [JAX's `jvp` through an eagerly executed loop](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html) — while a
/// semantic [`iteration_bound`](WhileOperation::with_iteration_bound) truncates the loop once it is reached, matching
/// the bounded-`while` truncation semantics. Body effects fire while the loop runs (the correct all-known placement),
/// once during the nested-`while` unroll interpretation and once during the fused replay, exactly as they did on the
/// reverse-mode pre-pass path.
fn jvp_while_eagerly<C>(
    operation: &WhileOperation<C::Constant, C::Operation>,
    context: &C,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Option<Vec<DifferentiationDual<C::Value>>>, ProgramError>
where
    C: Context + Zero<C::Value>,
    C::Value: BooleanLike,
    C::Constant: Value<Type = C::Type>,
    C::Operation: Clone
        + MaybeWhile<C::Constant, C::Operation>
        + From<ZeroOperation<C::Type>>
        + DifferentiableProgramOperation<C::Constant, C::Operation>,
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
        let mut condition_outputs = operation.condition().interpret_in_context(context, primal_carries.clone())?;
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
        let body = unroll_concretizable_whiles(context, operation.body().clone(), primal_carries.clone())?;
        let fused_body = C::Operation::jvp_program(&body)?;
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
/// runs the loop directly at the concrete duals (see [`jvp_while_eagerly`]), so eager forward mode is total over
/// data-dependent `while` loops with no iteration bound. Staging contexts — and eager contexts whose loop
/// predicate is batched and therefore has no single trip decision — dispatch to the loop's type family through
/// [`WhileJvp`]: array loops stage the hybrid bounded rule documented on that trait's [`ArrayType`] implementation,
/// and scalar loops report an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<C> DifferentiableOperation<C> for WhileOperation<C::Constant, C::Operation>
where
    C: Context<Operation: Clone> + Zero<C::Value>,
    C::Type: WhileJvp<C>,
    C::Value: BooleanLike,
    C::Constant: Value<Type = C::Type>,
    C::Operation: MaybeWhile<C::Constant, C::Operation>
        + From<ZeroOperation<C::Type>>
        + DifferentiableProgramOperation<C::Constant, C::Operation>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        if context.is_eager()
            && let Some(outputs) = jvp_while_eagerly(self, context, inputs)?
        {
            return Ok(outputs);
        }
        <C::Type>::jvp_while(self, context, inputs)
    }
}

impl<V: Value, O, Payload> TransposableOperation<V, O> for WhileOperation<V, O, Payload>
where
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
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: "while does not support transposition (reverse-mode differentiation through staged unbounded \
                      while loops is not supported; eager differentiation unrolls the loop instead, and loops built \
                      with `with_iteration_bound` stage a transposable masked scan)"
                .to_string(),
        })
    }
}

/// Batches a condition over `true_branch` and `false_branch` by reading the predicate from the first input.
///
/// A replicated predicate is concretized via [`BooleanLike::boolean`] and selects one branch to interpret over the
/// remaining operand inputs. A batch-varying predicate interprets both branches over the operand inputs and merges
/// their outputs per batch item via [`Select`](crate::operations::control_flow::Select).
pub(crate) fn batch_condition_with_interpreter<VOperation, V, O, F>(
    true_branch: &Program<VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    false_branch: &Program<VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    inputs: &[ArrayBatch<V>],
    mut interpret_program: F,
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    VOperation: Value<Type = ArrayType>,
    V: Value<Type = ArrayType> + BooleanLike + crate::operations::control_flow::Select<Condition = V>,
    O: Operation<ArrayType>,
    F: FnMut(
        &Program<VOperation, O, Vec<VOperation>, Vec<VOperation>>,
        Vec<ArrayBatch<V>>,
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError>,
{
    let Some((predicate_batch, operand_inputs)) = inputs.split_first() else {
        return Err(BatchingError::UnsupportedOperation {
            message: "cannot batch a condition operation with no predicate input".to_string(),
        }
        .into());
    };
    match predicate_batch.batch_axis().axis() {
        None => {
            let predicate = predicate_batch.value().boolean()?;
            let branch = if predicate { true_branch } else { false_branch };
            interpret_program(branch, operand_inputs.to_vec())
        }
        Some(predicate_axis) => {
            let true_outputs = interpret_program(true_branch, operand_inputs.to_vec())?;
            let false_outputs = interpret_program(false_branch, operand_inputs.to_vec())?;
            check_count!("output", true_outputs, false_outputs.len(), ProgramError);
            true_outputs
                .into_iter()
                .zip(false_outputs)
                .map(|(true_output, false_output)| -> Result<ArrayBatch<V>, BatchingError> {
                    let output_axis = match (true_output.batch_axis().axis(), false_output.batch_axis().axis()) {
                        (Some(left), Some(right)) if left != right => {
                            return Err(BatchingError::MisalignedBatchAxes {
                                message: format!(
                                    "condition branches produced batch-varying outputs at mismatched axes \
                                    ({left} vs {right})",
                                ),
                            }
                            .into());
                        }
                        (Some(axis), _) | (_, Some(axis)) => axis,
                        (None, None) => predicate_axis,
                    };
                    let selected = V::select(predicate_batch.value(), true_output.value(), false_output.value())?;
                    let output_type = selected.r#type().into_owned();
                    ArrayBatch::new(output_type, selected, Some(output_axis))
                })
                .collect()
        }
    }
}

impl<V, O> BatchableOperation<V, EagerContext<V, O>> for ConditionOperation<V, O>
where
    V: Value<Type = ArrayType> + BooleanLike + crate::operations::control_flow::Select<Condition = V>,
    O: BatchableOperation<V, EagerContext<V, O>>,
{
    fn batch(
        &self,
        context: &EagerContext<V, O>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        batch_condition_with_interpreter(self.true_branch(), self.false_branch(), inputs, |program, program_inputs| {
            program.interpret_with(
                program_inputs,
                |_, constant: &V| Ok(ArrayBatch::replicated(constant.clone())),
                |instruction: &Instruction<O>, instruction_inputs| {
                    instruction.operation().batch(context, instruction_inputs)
                },
            )
        })
    }
}

/// Staged batching for [`ConditionOperation`] under tracing contexts. Primal values in a [`BatchingContext`] over a
/// staging context are always tracers, so a replicated predicate can never be concretized to pick one branch the
/// way the value-level rule above does. Instead of erroring, this rule *stages batched condition structure*:
///
///   - **Replicated predicate.** Both branch programs are batched at the operand batch axes via
///     [`Program::batched`](crate::Program::batched) (the batching analog of symbolic program
///     linearization), their per-output batch axes are
///     normalized to a common layout by appending staged axis-moving operations at the branch tails when they
///     disagree (a transpose for a mismatched axis, a broadcast for a replicated output paired with a batched
///     one), and one [`ConditionOperation`] over the batched branches is staged into the parent context with the
///     unbatched predicate passed through as its scalar Boolean operand. The staged trace therefore keeps one
///     `condition` operation whose branches run whole batches per batch item.
///   - **Batch-varying predicate.** Both branches are interpreted over the operand inputs and merged per batch item
///     via [`Select`](crate::operations::control_flow::Select), exactly like the value-level rule: every per-item
///     primitive stages through the tracers, so the multi-operation rewrite composes under tracing already.
impl<C, O> BatchableOperation<<C as Domain>::Value, BatchingContext<C>> for ConditionOperation<C::Constant, O>
where
    C: Context<Type = ArrayType, Operation = O>,
    C::Constant: Value<Type = ArrayType>,
    <C as Domain>::Value: BooleanLike + Select<Condition = <C as Domain>::Value>,
    O: BatchableOperation<<C as Domain>::Value, BatchingContext<C>>
        + BatchableProgramOperation<C::Constant>
        + From<TransposeOperation>
        + From<BroadcastOperation>
        + From<SelectOperation>
        + From<ConditionOperation<C::Constant, O>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        let Some((predicate_batch, operand_inputs)) = inputs.split_first() else {
            return Err(BatchingError::UnsupportedOperation {
                message: "cannot batch a condition operation with no predicate input".to_string(),
            }
            .into());
        };
        if !predicate_batch.batch_axis().is_replicated() {
            // Batch-varying predicate: interpret both branches and merge their outputs per batch item via `Select`.
            return batch_condition_with_interpreter(
                self.true_branch(),
                self.false_branch(),
                inputs,
                |program, program_inputs| context.interpret_program(program, program_inputs),
            );
        }

        // Replicated (abstract) predicate: batch both branches at the operand batch axes with natural output axes to
        // discover which outputs each branch batches (the discovery programs are discarded), join the two answers into
        // one output layout — preferring the true branch's natural axis when both are batched — and re-batch each
        // branch instantiated at the joined targets so the branch signatures agree. This is the two-pass shape of
        // JAX's `_cond_batching_rule` (`batch_jaxpr` with `instantiate=out_bat`).
        let axis_size = context.axis_size();
        let operand_axes = operand_inputs.iter().map(|input| input.batch_axis()).collect::<Vec<_>>();
        let (_, true_axes) = O::batch_program(
            self.true_branch(),
            axis_size,
            operand_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let (_, false_axes) = O::batch_program(
            self.false_branch(),
            axis_size,
            operand_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        check_count!("output", false_axes, true_axes.len(), ProgramError);
        let output_axes: Vec<BatchAxis> = true_axes
            .iter()
            .zip(false_axes.iter())
            .map(|(true_axis, false_axis)| if true_axis.is_replicated() { *false_axis } else { *true_axis })
            .collect();
        let (batched_true_branch, _) = O::batch_program(
            self.true_branch(),
            axis_size,
            operand_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::AlignEachTo(output_axes.clone()),
        )?;
        let (batched_false_branch, _) = O::batch_program(
            self.false_branch(),
            axis_size,
            operand_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::AlignEachTo(output_axes.clone()),
        )?;

        // Stage one condition over the batched branches with the unbatched predicate passed through.
        let batched_condition = ConditionOperation::new(batched_true_branch, batched_false_branch)?;
        let mut staged_inputs = Vec::with_capacity(inputs.len());
        staged_inputs.push(predicate_batch.value().clone());
        staged_inputs.extend(operand_inputs.iter().map(|input| input.value().clone()));
        let outputs = context.parent().bind(batched_condition, &staged_inputs)?;
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

pub(crate) fn batch_while_with_interpreter<VOperation, V, O, Payload, F>(
    while_operation: &WhileOperation<VOperation, O, Payload>,
    inputs: &[ArrayBatch<V>],
    mut interpret_program: F,
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    VOperation: Value<Type = ArrayType>,
    V: Value<Type = ArrayType>
        + BooleanLike
        + crate::tracing_v2::operations::reduce::Reduce
        + std::ops::BitAnd<Output = V>
        + crate::operations::control_flow::Select<Condition = V>
        + crate::operations::manipulation::Broadcast,
    O: Operation<ArrayType>,
    F: FnMut(
        &Program<VOperation, O, Vec<VOperation>, Vec<VOperation>>,
        Vec<ArrayBatch<V>>,
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError>,
{
    // Run the condition once on the initial state to discover whether the predicate is
    // replicated or batch-varying. The two cases diverge from here: replicated takes the
    // original eager-loop path; batch-varying threads a per-item mask through every iteration
    // and runs the body until no batch item is still active. Both loops respect the semantic
    // iteration bound: a bounded while runs at most `bound` body applications by definition.
    let mut state = inputs.to_vec();
    let initial_condition_outputs = interpret_program(while_operation.condition(), state.clone())?;
    check_count!("output", initial_condition_outputs, 1, ProgramError);
    let initial_predicate = initial_condition_outputs.into_iter().next().unwrap();
    if initial_predicate.batch_axis().is_replicated() {
        if !initial_predicate.value().boolean()? {
            return Ok(state);
        }
        state = interpret_program(while_operation.body(), state)?;
        return run_replicated_while_loop::<VOperation, V, O, F>(
            while_operation.condition(),
            while_operation.body(),
            state,
            // One body application already ran above, so the loop helper receives the remaining budget.
            while_operation.iteration_bound().map(|bound| bound - 1),
            &mut interpret_program,
        );
    }
    // Batch-varying path: the predicate carries a batch axis. Track a per-item mask, mask
    // state updates per batch item via `Select`, and exit once `any(mask)` is false.
    run_batch_varying_while_loop::<VOperation, V, O, F>(
        while_operation.condition(),
        while_operation.body(),
        state,
        initial_predicate,
        while_operation.iteration_bound(),
        &mut interpret_program,
    )
}

impl<V, O, Payload> BatchableOperation<V, EagerContext<V, O>> for WhileOperation<V, O, Payload>
where
    V: Value<Type = ArrayType>
        + BooleanLike
        + crate::tracing_v2::operations::reduce::Reduce
        + std::ops::BitAnd<Output = V>
        + crate::operations::control_flow::Select<Condition = V>
        + crate::operations::manipulation::Broadcast,
    O: BatchableOperation<V, EagerContext<V, O>>,
{
    fn batch(
        &self,
        context: &EagerContext<V, O>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        batch_while_with_interpreter(self, inputs, |program, program_inputs| {
            program.interpret_with(
                program_inputs,
                |_, constant| Ok(ArrayBatch::replicated(constant.clone())),
                |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
            )
        })
    }
}

/// Staged batching for [`WhileOperation`] under tracing contexts. Primal values in a [`BatchingContext`] over a
/// staging context are always tracers, so the loop cannot be driven operationally the way the value-level rule above
/// drives it (per-iteration predicate extraction would concretize tracers). Instead, this rule *stages batched loop
/// structure*:
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
///      [`WhileOperation`] over the batched condition and body is staged directly, preserving any semantic
///      [`iteration_bound`](WhileOperation::with_iteration_bound) (so bounded loops stay reverse-capable under
///      `batch`).
///   3. When the predicate output is *batched* (per-item termination), every state element is widened to a batched
///      element and the masked loop the value-level rule runs operationally is traced as ordinary tracer-valued
///      functions over the augmented state `[state..., active_mask]`: the masked condition reduces the mask with a
///      batch-axis `any`, and the masked body splices the batched body (via
///      [`StagingContext::stage_program`]), selects per state element between the candidate update and the carried
///      state under the (broadcast) mask, recomputes the per-item predicate on the new state, and ANDs it into the
///      mask. The initial mask is the batched condition staged once over the initial state in the parent context,
///      and the iteration bound is preserved (batch items share masked iterations, so capping the staged loop
///      matches per-item truncation exactly, like the operational rule).
impl<C, O> BatchableOperation<<C as Domain>::Value, BatchingContext<C>> for WhileOperation<C::Constant, O>
where
    C: Context<Type = ArrayType, Operation = O>,
    C::Constant: Value<Type = ArrayType>,
    <C as Domain>::Value: Broadcast + Transpose,
    O: Clone
        + BatchableProgramOperation<C::Constant>
        + From<TransposeOperation>
        + From<BroadcastOperation>
        + From<ReduceOperation>
        + From<SelectOperation>
        + From<AndOperation>
        + From<WhileOperation<C::Constant, O>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        let state_count = self.state_types().len();
        check_count!("input", inputs, state_count, ProgramError);
        let axis_size = context.axis_size();

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
            let (candidate_body, body_axes) = O::batch_program(
                self.body(),
                axis_size,
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
            }
            .into());
        };

        // Batch the condition at the stabilized axes; a batched predicate output means per-item termination, in
        // which case every state element participates in per-item masking and is therefore widened to a batched
        // element before the masked loop structure is built.
        let (mut batched_condition, mut condition_axes) = O::batch_program(
            self.condition(),
            axis_size,
            state_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        check_count!("output", condition_axes, 1, ProgramError);
        let batch_varying = !condition_axes[0].is_replicated();
        if batch_varying && state_axes.iter().any(|axis| axis.is_replicated()) {
            state_axes = vec![BatchAxis::new(0); state_count];
            let (widened_body, body_axes) = O::batch_program(
                self.body(),
                axis_size,
                state_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::AlignEachTo(state_axes.clone()),
            )?;
            check_count!("output", body_axes, state_count, ProgramError);
            batched_body = widened_body;
            (batched_condition, condition_axes) = O::batch_program(
                self.condition(),
                axis_size,
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
            let batched_while =
                WhileOperation::new(batched_condition, batched_body)?.with_iteration_bound(self.iteration_bound())?;
            let outputs = context.parent().bind(batched_while, &state_values)?;
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
        // [`WhilePredicate::mask_select`](crate::operations::control_flow::WhilePredicate), and the XLA lowering
        // reduces the predicate with `or` and masks carry updates with a broadcast select.
        let (batched_condition, condition_axes) = O::batch_program(
            self.condition(),
            axis_size,
            state_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::AlignEachTo(vec![BatchAxis::new(0)]),
        )?;
        check_count!("output", condition_axes, 1, ProgramError);
        let batched_while =
            WhileOperation::new(batched_condition, batched_body)?.with_iteration_bound(self.iteration_bound())?;
        let outputs = context.parent().bind(batched_while, &state_values)?;
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
/// and the masked body splices the original body for candidate updates, selects per state element between the
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
    O: Clone
        + Operation<ArrayType>
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

    // Masked body: candidate updates from the spliced body, per-element masked selection between the candidate
    // update and the carried state, the per-item predicate recomputed on the new state, and the mask narrowed via
    // AND.
    let (_, masked_body) = TracingContext::<V, O>::trace(
        |inputs| {
            let (mask, state) = inputs.split_last().unwrap();
            let trace_context = mask.context().clone();
            let candidates = trace_context.stage_program(body, state.to_vec())?;
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
            let next_predicate = trace_context.stage_program(condition, next_state.clone())?;
            check_count!("output", next_predicate, 1, ProgramError);
            let mut outputs = next_state;
            outputs.push(mask.clone() & next_predicate.into_iter().next().unwrap());
            Ok(outputs)
        },
        masked_state_types,
    )?;
    Ok((masked_condition, masked_body))
}

/// Eager loop that drives a [`WhileOperation`] whose condition program produces a replicated
/// scalar Boolean predicate. Each iteration runs the body when the predicate is `true` and exits
/// when it becomes `false` or once the remaining iteration budget (the semantic iteration bound
/// minus any body applications the caller already performed) is exhausted. This is the original
/// simple loop preserved for the replicated case.
fn run_replicated_while_loop<VOperation, V, O, F>(
    condition: &Program<VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    body: &Program<VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    mut state: Vec<ArrayBatch<V>>,
    mut remaining_iterations: Option<usize>,
    interpret_program: &mut F,
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    VOperation: Value<Type = ArrayType>,
    V: Value<Type = ArrayType> + BooleanLike,
    F: FnMut(
        &Program<VOperation, O, Vec<VOperation>, Vec<VOperation>>,
        Vec<ArrayBatch<V>>,
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError>,
{
    loop {
        if remaining_iterations == Some(0) {
            return Ok(state);
        }
        let condition_outputs = interpret_program(condition, state.clone())?;
        check_count!("output", condition_outputs, 1, ProgramError);
        let predicate_batch = &condition_outputs[0];
        if !predicate_batch.batch_axis().is_replicated() {
            return Err(BatchingError::UnsupportedOperation {
                message: "while loop condition produced a batch-varying predicate mid-iteration after starting \
                    replicated; this is not yet supported"
                    .to_string(),
            }
            .into());
        }
        if !predicate_batch.value().boolean()? {
            return Ok(state);
        }
        state = interpret_program(body, state)?;
        remaining_iterations = remaining_iterations.map(|remaining| remaining - 1);
    }
}

/// Eager loop that drives a [`WhileOperation`] whose condition program produces a batch-varying
/// predicate (one Boolean per mapped batch item). Each iteration:
///
///   1. Updates the per-item active mask by AND-ing with the current per-item predicate.
///   2. Stops when no batch item is still active (`any(mask) == false`).
///   3. Runs the body to produce candidate updated state.
///   4. Masks state updates per batch item via [`Select`](crate::operations::control_flow::Select)
///      so inactive batch items retain their prior state forever.
///
/// This implementation requires a value type that supports [`Reduce`](
/// crate::tracing_v2::operations::reduce::Reduce) (for the `any` aggregation),
/// [`BitAnd`](std::ops::BitAnd) (for `mask & current`),
/// [`Select`](crate::operations::control_flow::Select), and
/// [`Broadcast`](crate::operations::manipulation::Broadcast) — the same
/// primitives every staged value type already needs for the rest of the operation enum.
fn run_batch_varying_while_loop<VOperation, V, O, F>(
    condition: &Program<VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    body: &Program<VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    mut state: Vec<ArrayBatch<V>>,
    initial_predicate: ArrayBatch<V>,
    iteration_bound: Option<usize>,
    interpret_program: &mut F,
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    VOperation: Value<Type = ArrayType>,
    V: Value<Type = ArrayType>
        + BooleanLike
        + crate::tracing_v2::operations::reduce::Reduce
        + std::ops::BitAnd<Output = V>
        + crate::operations::control_flow::Select<Condition = V>
        + crate::operations::manipulation::Broadcast,
    F: FnMut(
        &Program<VOperation, O, Vec<VOperation>, Vec<VOperation>>,
        Vec<ArrayBatch<V>>,
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError>,
{
    let predicate_axis = initial_predicate.batch_axis().axis().ok_or_else(|| BatchingError::MisalignedBatchAxes {
        message: "batch-varying while batching requires a batched initial predicate".to_string(),
    })?;
    let mut active_mask = initial_predicate;
    let mut remaining_iterations = iteration_bound;
    loop {
        // The semantic iteration bound applies per batch item, and every batch item shares the same masked
        // iterations, so capping the shared loop at `bound` body applications matches the per-item truncation
        // semantics exactly.
        if remaining_iterations == Some(0) || !batch_varying_any_active(&active_mask, predicate_axis)? {
            return Ok(state);
        }
        let body_outputs = interpret_program(body, state.clone())?;
        check_count!("output", body_outputs, state.len(), ProgramError);
        state = state
            .into_iter()
            .zip(body_outputs)
            .map(|(prior, candidate)| mask_state_element(&active_mask, predicate_axis, candidate, prior))
            .collect::<Result<Vec<_>, _>>()?;
        let next_condition_outputs = interpret_program(condition, state.clone())?;
        check_count!("output", next_condition_outputs, 1, ProgramError);
        let next_predicate = next_condition_outputs.into_iter().next().unwrap();
        if next_predicate.batch_axis().is_replicated() {
            return Err(BatchingError::UnsupportedOperation {
                message: "while loop predicate became replicated mid-iteration after starting batch-varying; \
                    this is not yet supported"
                    .to_string(),
            }
            .into());
        }
        active_mask = combine_active_mask(active_mask, next_predicate)?;
        remaining_iterations = remaining_iterations.map(|remaining| remaining - 1);
    }
}

/// Returns `true` when at least one batch item of `mask` is active by reducing along `predicate_axis`
/// and extracting the resulting scalar Boolean.
fn batch_varying_any_active<
    V: Value<Type = ArrayType> + BooleanLike + crate::tracing_v2::operations::reduce::Reduce,
>(
    mask: &ArrayBatch<V>,
    predicate_axis: usize,
) -> Result<bool, ProgramError> {
    let reduced = mask
        .value()
        .clone()
        .reduce(&[predicate_axis], crate::tracing_v2::operations::reduce::ReductionKind::Any);
    reduced.boolean()
}

/// Combines the prior `active_mask` with the current `next_predicate` via logical AND. Both must
/// be batched on the same physical axis; the result inherits that axis.
fn combine_active_mask<V: Value<Type = ArrayType> + std::ops::BitAnd<Output = V>>(
    active_mask: ArrayBatch<V>,
    next_predicate: ArrayBatch<V>,
) -> Result<ArrayBatch<V>, BatchingError> {
    let axis = active_mask.batch_axis();
    let combined = active_mask.into_value() & next_predicate.into_value();
    let combined_type = combined.r#type().into_owned();
    ArrayBatch::new(combined_type, combined, axis)
}

/// Builds the masked update for one state element by broadcasting the per-item mask to the
/// element's physical shape and selecting between the candidate body output and the prior state
/// per batch item.
fn mask_state_element<V>(
    active_mask: &ArrayBatch<V>,
    predicate_axis: usize,
    candidate: ArrayBatch<V>,
    prior: ArrayBatch<V>,
) -> Result<ArrayBatch<V>, BatchingError>
where
    V: Value<Type = ArrayType>
        + crate::operations::control_flow::Select<Condition = V>
        + crate::operations::manipulation::Broadcast,
{
    let candidate_axis = candidate.batch_axis().axis().or(prior.batch_axis().axis()).ok_or_else(|| {
        BatchingError::UnsupportedOperation {
            message: "batch-varying while body produced a replicated state element; this is not yet supported"
                .to_string(),
        }
    })?;
    let candidate_type = candidate.r#type().into_owned();
    let mask_type = active_mask.r#type().into_owned();
    let mask_output_axes: Vec<usize> = (0..mask_type.rank())
        .map(|i| {
            if i == predicate_axis {
                candidate_axis
            } else if i < predicate_axis {
                // mask axes left of the predicate axis carry over to the candidate left of `candidate_axis`.
                i
            } else {
                // mask axes right of the predicate axis carry over to the candidate right of `candidate_axis`.
                i + (candidate_type.rank() - mask_type.rank())
            }
        })
        .collect();
    let mask_output_type = ArrayType::new(mask_type.data_type(), candidate_type.shape().clone());
    let broadcasted_mask = active_mask.value().broadcast(mask_output_type, mask_output_axes.as_slice())?;
    let selected = V::select(&broadcasted_mask, &candidate.into_value(), &prior.into_value())?;
    let selected_type = selected.r#type().into_owned();
    ArrayBatch::new(selected_type, selected, Some(candidate_axis))
}

impl<V, F, O, C> InterpretableOperation<V, C> for ConditionOperation<V, O, F, Captured>
where
    V: Value<Type = ArrayType> + BooleanLike,
    F: CustomVjpResidual<V>,
    O: InterpretableProgramOperation<V, C, V>,
{
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice())?;
        let predicate = self.predicate().residual_value()?;
        let branch = if predicate.boolean()? { self.true_branch() } else { self.false_branch() };
        O::interpret_program(context, branch, inputs.to_vec())
    }
}

/// Transpose rule for the captured-predicate conditional. The predicate is a residual of the primal computation rather
/// than a linear operand, so it has no cotangent and is carried verbatim into a transposed condition over the
/// transposed branch programs, selected by the same predicate. Branch transposition goes through
/// [`TransposableProgramOperation`], keeping the branch fixed point owned by the operation family.
impl<V, F, O> TransposableOperation<V, O> for ConditionOperation<V, O, F, Captured>
where
    V: Value<Type = ArrayType>,
    F: Value<Type = ArrayType>,
    O: Operation<ArrayType>
        + TransposableProgramOperation<V>
        + From<ZeroOperation<ArrayType>>
        + From<ConditionOperation<V, O, F, Captured>>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        // A condition with no outputs (or only zero output cotangents) is a zero linear map, so every input
        // cotangent is zero. Note that `all` is trivially true for an empty cotangent slice.
        if outputs.iter().all(MaybeZero::is_zero) {
            return Ok(self
                .true_branch()
                .input_types()
                .into_iter()
                .map(|input_type| MaybeZero::Zero(input_type.clone()))
                .collect());
        }
        let transposed_condition = ConditionOperation::new_captured(
            self.predicate().clone(),
            <O as TransposableProgramOperation<V>>::transpose_program(
                self.true_branch(),
                &vec![true; self.true_branch().input_ids().len()],
            )?,
            <O as TransposableProgramOperation<V>>::transpose_program(
                self.false_branch(),
                &vec![true; self.false_branch().input_ids().len()],
            )?,
        )?;
        let materialized = outputs
            .iter()
            .map(|cotangent| cotangent.clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        let cotangents = context.stage_operation(transposed_condition, materialized.as_slice())?;
        check_count!("output", cotangents, self.true_branch().input_types().len(), ProgramError);
        Ok(cotangents.into_iter().map(MaybeZero::Value).collect())
    }
}

/// Partition-aware transpose rule for a *primal* input-predicate [`ConditionOperation`], forwarding to
/// [`transpose_primal_condition`]. The predicate and the per-branch residuals ride as ordinary known operands, and the
/// branch recursion happens through the [`TransposableProgramOperation`] fixed-point witness, so instantiating this
/// implementation for a closed operation enum introduces no recursive [`TransposableOperation`] obligation on `O`.
impl<V, O> TransposableOperation<V, O> for ConditionOperation<V, O, V, Input>
where
    V: Value<Type = ArrayType>,
    O: TransposableProgramOperation<V> + From<ZeroOperation<ArrayType>> + From<ConditionOperation<V, O>>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        transpose_primal_condition(self, context, inputs, outputs)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::macros::check_types;
    use std::fmt::Display;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_macros::Parameter;

    use crate::contexts::{Context, Domain, EagerContext};
    use crate::interpretation::{InterpretableOperation, InterpretableProgramOperation};
    use crate::operations::arithmetic::{
        ADD_OPERATION_NAME, AddOperation, MulOperation, SUB_OPERATION_NAME, SubOperation,
    };
    use crate::operations::compare::CompareOperation;
    use crate::operations::constants::{One, OneLike, OneLikeOperation, Zero, ZeroLike, ZeroLikeOperation};
    use crate::parameters::{Parameter, Placeholder};
    use crate::programs::{Program, ProgramBuilder, Value};
    use crate::tracing::DomainTracingContext;
    use crate::tracing_v2::operations::reduce::ReduceOperation;
    use crate::tracing_v2::{ArrayOperation, Differentiate};
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
        Condition(Box<ConditionOperation<TestValue, TestOperation>>),
        While(Box<WhileOperation<TestValue, TestOperation>>),
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
                Self::While(while_operation) => while_operation.name(),
            }
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
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
                Self::Condition(condition) => condition.infer_output_types(input_types),
                Self::While(while_operation) => while_operation.infer_output_types(input_types),
            }
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Condition(condition) => condition.render(formatter, indentation),
                Self::While(while_operation) => while_operation.render(formatter, indentation),
                _ => Display::fmt(self, formatter),
            }
        }
    }

    impl<C> InterpretableOperation<TestValue, C> for TestOperation
    where
        C: crate::operations::constants::Constant<TestValue, TestValue>,
    {
        fn interpret(&self, context: &C, inputs: &[TestValue]) -> Result<Vec<TestValue>, ProgramError> {
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
                Self::Condition(condition) => condition.interpret(context, inputs),
                Self::While(while_operation) => while_operation.interpret(context, inputs),
            }
        }
    }

    impl<C> InterpretableProgramOperation<TestValue, C> for TestOperation
    where
        C: crate::operations::constants::Constant<TestValue, TestValue>,
        TestOperation: InterpretableOperation<TestValue, C>,
    {
        fn interpret_program(
            context: &C,
            program: &Program<TestValue, Self, Vec<TestValue>, Vec<TestValue>>,
            input: Vec<TestValue>,
        ) -> Result<Vec<TestValue>, ProgramError> {
            program.interpret_with(
                input,
                |_, constant| context.constant(constant.clone()),
                |instruction, inputs| instruction.operation().interpret(context, inputs),
            )
        }
    }

    fn add_one_branch() -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_constant(TestValue::Number(1.0));
        let output = builder.add_instruction(TestOperation::Add, vec![input, one]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn subtract_one_branch() -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_constant(TestValue::Number(1.0));
        let output = builder.add_instruction(TestOperation::Sub, vec![input, one]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn identity_array_branch() -> Program<TestValue, ArrayOperation<TestValue>, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, ArrayOperation<TestValue>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_condition_interprets_true_and_false_branches() {
        let condition = ConditionOperation::new(add_one_branch(), subtract_one_branch()).unwrap();

        assert_eq!(
            condition.interpret(&EagerContext::<TestValue>::new(), &[TestValue::Bool(true), TestValue::Number(3.0)]),
            Ok(vec![TestValue::Number(4.0)]),
        );
        assert_eq!(
            condition.interpret(&EagerContext::<TestValue>::new(), &[TestValue::Bool(false), TestValue::Number(3.0)]),
            Ok(vec![TestValue::Number(2.0)]),
        );
    }

    #[test]
    fn test_condition_program_rendering_includes_nested_branches() {
        let condition = ConditionOperation::new(add_one_branch(), subtract_one_branch()).unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(TestOperation::Condition(Box::new(condition)), vec![predicate, input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition [
                    true_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = const
                            %2:f64[] = add %0 %1
                        in (%2)
                    },
                    false_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = const
                            %2:f64[] = sub %0 %1
                        in (%2)
                    },
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_condition_rejects_branch_output_mismatch() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestOperation::IsPositive, vec![input]).unwrap()[0];
        let bool_branch = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert!(ConditionOperation::new(add_one_branch(), bool_branch).is_err());
    }

    #[test]
    fn test_while_interprets_until_condition_is_false() {
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition_input = condition_builder.add_input(ArrayType::scalar(DataType::F64));
        let condition_output =
            condition_builder.add_instruction(TestOperation::IsPositive, vec![condition_input]).unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation: WhileOperation<TestValue, TestOperation> =
            WhileOperation::new(condition, subtract_one_branch()).unwrap();

        assert_eq!(
            while_operation.interpret(&EagerContext::<TestValue>::new(), &[TestValue::Number(3.0)]),
            Ok(vec![TestValue::Number(0.0)]),
        );
    }

    #[test]
    fn test_while_program_rendering_includes_condition_and_body() {
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition_input = condition_builder.add_input(ArrayType::scalar(DataType::F64));
        let condition_output =
            condition_builder.add_instruction(TestOperation::IsPositive, vec![condition_input]).unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation: WhileOperation<TestValue, TestOperation> =
            WhileOperation::new(condition, subtract_one_branch()).unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestOperation::While(Box::new(while_operation)), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = while [
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
                ] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_operation_condition_infers_output_types() {
        let condition = ConditionOperation::new(identity_array_branch(), identity_array_branch()).unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));

        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F64)]),
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

        fn bind<P: Into<Self::Operation>>(
            &self,
            operation: P,
            inputs: &[Self::Value],
        ) -> Result<Vec<Self::Value>, ProgramError> {
            let operation = operation.into();
            operation.interpret(&crate::EagerContext::<TestArray, Self::Operation>::new(), inputs)
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

    impl crate::operations::constants::Fill<f64, TestArray> for StagedDispatchTestArrayDomain {
        fn fill(&self, r#type: &ArrayType, value: f64) -> Result<TestArray, ProgramError> {
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
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), vec![state, threshold])
            .unwrap()[0];
        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the `while (x < threshold) { x = 2 * x }` loop with the provided semantic iteration bound.
    fn bounded_doubling_while_operation(threshold: f64, bound: usize) -> WhileOperation<TestArray, TestArrayOperation> {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let state = builder.add_input(scalar_f64);
        let two = builder.add_constant(TestArray::scalar(2.0));
        let doubled = builder.add_instruction(MulOperation, vec![state, two]).unwrap()[0];
        let body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        WhileOperation::new(scalar_threshold_condition(threshold), body)
            .unwrap()
            .with_iteration_bound(bound)
            .unwrap()
    }

    /// Builds the `while (x < threshold) { x = x * x }` loop with the provided semantic iteration bound. Squaring
    /// captures the loop state itself as a loop-varying residual, so differentiating this loop exercises the
    /// per-iteration residual stacks of the bounded staged path.
    fn bounded_squaring_while_operation(threshold: f64, bound: usize) -> WhileOperation<TestArray, TestArrayOperation> {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let state = builder.add_input(scalar_f64);
        let squared = builder.add_instruction(MulOperation, vec![state, state]).unwrap()[0];
        let body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        WhileOperation::new(scalar_threshold_condition(threshold), body)
            .unwrap()
            .with_iteration_bound(bound)
            .unwrap()
    }

    #[test]
    fn test_bounded_while_value_and_grad_computes_gradient_through_staged_masked_scan() {
        // The headline bounded-while capability: end-to-end reverse mode through a *staged* while loop.
        // `f(x) = while (x < 8, iteration_bound = 5) { x = 2 * x }` at `x = 1` runs three iterations (`x` visits 1,
        // 2, 4), so the actual trip count 3 is strictly below the bound 5 and the two trailing batch items matter:
        // their mask entries are false, so they must pass tangents through unchanged in the forward scan and cotangents
        // through unchanged in the transposed scan. Locally `f(x) = 8 x`: value 8, gradient 8.
        let while_operation = bounded_doubling_while_operation(8.0, 5);
        let (output, pullback, residuals) = StagedDispatchTestArrayDomain
            .vjp(
                move |x| {
                    let mut outputs =
                        x.context().stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
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
        let while_operation = bounded_doubling_while_operation(8.0, 5);
        let (value, gradient) = StagedDispatchTestArrayDomain
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
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
        let while_operation = bounded_squaring_while_operation(100.0, 4);
        let (output, pullback, residuals) = StagedDispatchTestArrayDomain
            .vjp(
                move |x| {
                    let mut outputs =
                        x.context().stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(2.0),
            )
            .unwrap();
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
        let while_operation = bounded_squaring_while_operation(100.0, 4);
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
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
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), vec![condition_state])
            .unwrap()[0];
        let threshold = condition_builder.add_constant(TestArray::scalar(20.0));
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), vec![summed, threshold])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let body_state = body_builder.add_input(vector_f64.clone());
        let squared = body_builder.add_instruction(MulOperation, vec![body_state, body_state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new(condition, body).unwrap().with_iteration_bound(4).unwrap();

        let (value, gradient) = StagedDispatchTestArrayDomain
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
                        .unwrap();
                    let state = outputs.remove(0);
                    let mut outputs = state
                        .context()
                        .stage_operation(ReduceOperation::new(vec![0], ReductionKind::Sum), &[&state])
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
        let while_operation = bounded_doubling_while_operation(8.0, 5);
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
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
        let while_operation = bounded_doubling_while_operation(f64::INFINITY, 3);
        let outputs = while_operation
            .interpret(&crate::EagerContext::<TestArray>::new(), &[TestArray::scalar(2.0)])
            .unwrap();
        assert_eq!(outputs[0].values, vec![16.0]);

        let while_operation = bounded_doubling_while_operation(f64::INFINITY, 3);
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(2.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![16.0]);
        assert_eq!(gradient.values, vec![8.0]);

        let while_operation = bounded_doubling_while_operation(f64::INFINITY, 3);
        let (value, gradient) = StagedDispatchTestArrayDomain
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
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
    fn countdown_while_operation() -> WhileOperation<TestArray, TestArrayOperation> {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let zero = condition_builder.add_instruction(ZeroLikeOperation, vec![condition_state]).unwrap()[0];
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![condition_state, zero])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let body_state = body_builder.add_input(scalar_f64);
        let one = body_builder.add_instruction(OneLikeOperation, vec![body_state]).unwrap()[0];
        let next = body_builder.add_instruction(SubOperation, vec![body_state, one]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![next], vec![Placeholder], vec![Placeholder])
            .unwrap();
        WhileOperation::new(condition, body).unwrap()
    }

    /// Stages `while_operation` over one batched item (mapped at axis 0 with `batch_size` batch items) under tracing
    /// and returns the staged batched program for structural and numeric assertions.
    fn batch_while_under_tracing(
        while_operation: WhileOperation<TestArray, TestArrayOperation>,
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
                let mut outputs =
                    item.context().bind(TestArrayOperation::While(Box::new(while_operation)), &[item.clone()])?;
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
        let program = batch_while_under_tracing(countdown_while_operation(), 3);
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
        let program = batch_while_under_tracing(countdown_while_operation().with_iteration_bound(2).unwrap(), 3);
        let rendered = program.to_string();
        assert!(rendered.contains("iteration_bound=2"), "{rendered}");
        let output = program.interpret(TestArray::vector(vec![3.0, 1.0, 2.0])).unwrap();
        assert_eq!(output.values, vec![1.0, 0.0, 0.0]);
    }

    /// Builds the `while (counter > 0) { (counter, value) = (counter - 1, value + value) }` loop whose predicate
    /// depends only on the counter state element.
    fn counter_doubling_while_operation() -> WhileOperation<TestArray, TestArrayOperation> {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let condition_counter = condition_builder.add_input(scalar_f64.clone());
        condition_builder.add_input(scalar_f64.clone());
        let zero = condition_builder.add_instruction(ZeroLikeOperation, vec![condition_counter]).unwrap()[0];
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![condition_counter, zero])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestArray, TestArrayOperation>::new();
        let body_counter = body_builder.add_input(scalar_f64.clone());
        let body_value = body_builder.add_input(scalar_f64);
        let one = body_builder.add_instruction(OneLikeOperation, vec![body_counter]).unwrap()[0];
        let next_counter = body_builder.add_instruction(SubOperation, vec![body_counter, one]).unwrap()[0];
        let doubled = body_builder.add_instruction(AddOperation, vec![body_value, body_value]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![next_counter, doubled],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        WhileOperation::new(condition, body).unwrap()
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
                let while_operation = counter_doubling_while_operation();
                let mut outputs = counter
                    .context()
                    .bind(TestArrayOperation::While(Box::new(while_operation)), &[counter.clone(), value.clone()])?;
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
        use crate::batching::{Batch, BatchableOperation, BatchingContext, BatchingTracer};

        // F5 x F6 composition: jvp of a *vmapped bounded* while under the non-concretizing staged dispatch domain.
        // Batching stages one masked bounded while (the predicate `x < 8` is per batch item and the iteration bound 5
        // survives the staged rewrite), so the while JVP rule takes the bounded staged path: stored residual
        // stacks plus a masked linear scan on the tangent side. Batch items [1, 5, 9] double 3, 1, and 0 times, so the
        // primal is [8, 10, 9] and the per-item tangent scale is 2^iterations = [8, 2, 1].
        fn batched_bounded_while<V>(x: V) -> Result<V, ProgramError>
        where
            V: Value<Type = ArrayType> + crate::operations::manipulation::Transpose,
            V::DispatchDomain: Context<Type = ArrayType, Value = V, Operation = TestArrayOperation>,
            TestArrayOperation: BatchableOperation<V, BatchingContext<V::DispatchDomain>>,
        {
            let context = x.dispatch_domain();
            let mapped = Batch::batch(
                &context,
                |item: BatchingTracer<V::DispatchDomain>| {
                    let batching_context = item.context().clone();
                    let mut outputs = batching_context.bind(bounded_doubling_while_operation(8.0, 5), &[item])?;
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
        let (output, pullback, residuals) = StagedDispatchTestArrayDomain
            .vjp(batched_bounded_while, TestArray::vector(vec![1.0, 5.0, 9.0]))
            .unwrap();
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
}
