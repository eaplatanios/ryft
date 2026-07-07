//! Eager unroll-then-fuse pre-pass.
//!
//! Eager differentiation domains (whose
//! [`is_eager`](Context::is_eager) returns `true`) can
//! differentiate unbounded, data-dependent `while` loops by unrolling them at the concrete primal
//! state. Reverse mode needs that unrolling at the *program* level: transposition consumes a primal [`Program`], so
//! given a traced primal [`Program`] and the concrete input values at which it is being differentiated,
//! [`unroll_concretizable_whiles`] rewrites it into an equivalent straight-line [`Program`] with every concretizable
//! `while` unrolled, which the capture-free path then consumes unchanged: an unrolled straight-line primal program
//! produces a control-flow-free tangent program that the existing partitioned transposition handles. (Forward mode
//! needs no pre-pass: the `while` forward-mode rule runs data-dependent loops directly at the concrete duals.)
//!
//! The rewrite is a dual-table pass over the source program's atoms. Each source atom is threaded with both (a) its
//! concrete value, used to evaluate `while` conditions and drive trip counts, and (b) the [`AtomId`] of the
//! corresponding atom in the program being built, used to emit the straight-line program. The two tables ride together
//! as a single `(value, atom)` pair through [`Program::interpret_with`], which performs the per-atom bookkeeping:
//!
//!   - A non-`while` instruction is added verbatim to the builder over its operands' new atoms (preserving the
//!     operation, so tangents survive — the rewrite does not fold operations to constants), and is also interpreted
//!     concretely through [`Context::bind`] on its operands' concrete values.
//!   - A `while` instruction is unrolled: the condition is concretized on the current concrete carry, and while it is
//!     true the body is recursively rewritten into the builder over the current carry's new atoms while being
//!     interpreted concretely on the current carry's values; the final carry becomes the loop's results.
//!
//! Nested `while`s fall out of the recursion: an inner `while` is just an instruction encountered while recursively
//! rewriting the outer body, so it is unrolled by the same pass. This is why the body is rewritten recursively rather
//! than copied with [`add_program`](crate::ProgramBuilder::add_program), which would relocate a nested `while`
//! verbatim and leave it un-unrolled.

use std::cell::RefCell;

use crate::contexts::{Context, Domain};
use crate::macros::check_count;
use crate::operations::BooleanLike;
use crate::operations::control_flow::{MaybeWhile, WhileParts};
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError};

/// Rewrites `program` into an equivalent straight-line [`Program`] with every concretizable `while` loop unrolled at
/// the concrete `input_values`, leaving all other instructions unchanged.
///
/// This is the eager unroll-then-fuse pre-pass. It is value-level (`Program -> Program` over the domain's operation and
/// value types) and carries no recursive trait obligation, so it composes with the front ends without growing the
/// trait-solver obligation graph. The returned program has the same input and output parameter structures as `program`,
/// so it slots in transparently before the JVP / linearization build.
///
/// The rewrite runs only when the context
/// [is eager](Context::is_eager): only eager domains,
/// whose primal values are concrete, can evaluate a `while` condition with [`BooleanLike::boolean`] and decide a
/// data-dependent trip count. For symbolic and staging domains the program is returned unchanged, keeping their
/// staged (masked-scan / linear-loop) strategies.
///
/// # Parameters
///
///   - `context`: Context whose [`lift`](Context::lift) and [`bind`](Context::bind) supply the
///     concrete value semantics used to evaluate `while` conditions and drive trip counts.
///   - `program`: Traced primal program to rewrite, with flat [`Vec`]-parameterized inputs and outputs.
///   - `input_values`: Concrete input values aligned with `program`'s input atoms, at which the loops are unrolled.
pub(crate) fn unroll_concretizable_whiles<C>(
    context: &C,
    program: Program<
        <C as Domain>::Constant,
        <C as Domain>::Operation,
        Vec<<C as Domain>::Constant>,
        Vec<<C as Domain>::Constant>,
    >,
    input_values: Vec<<C as Domain>::Value>,
) -> Result<
    Program<
        <C as Domain>::Constant,
        <C as Domain>::Operation,
        Vec<<C as Domain>::Constant>,
        Vec<<C as Domain>::Constant>,
    >,
    ProgramError,
>
where
    C: Context,
    <C as Domain>::Value: BooleanLike,
    <C as Domain>::Constant: Clone,
    <C as Domain>::Operation: Clone + MaybeWhile<<C as Domain>::Constant, <C as Domain>::Operation>,
{
    if !context.is_eager() {
        return Ok(program);
    }
    let program = &program;
    let mut builder = ProgramBuilder::new();
    let input_atoms = program.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
    let input_pairs = input_values.into_iter().zip(input_atoms).collect::<Vec<_>>();

    // A `RefCell` lets the borrow-checker accept the two interpretation closures, which never run concurrently, each
    // taking a short-lived mutable borrow of the builder.
    let builder = RefCell::new(builder);
    let output_pairs = rewrite_program_into(context, &builder, program, input_pairs)?;

    // The rewritten program keeps `program`'s flat input and output parameter structures: it has the same inputs, and
    // one output per source output.
    let output_atoms = output_pairs.into_iter().map(|(_, atom)| atom).collect::<Vec<_>>();
    builder
        .into_inner()
        .build(output_atoms, program.input_structure().clone(), program.output_structure().clone())
}

/// Rewrites `program` into `builder` under the dual `(value, atom)` table, returning the program's outputs as
/// `(value, atom)` pairs. This is the recursive core shared by the top-level rewrite and the per-iteration unrolling of
/// each `while` body, so nested `while`s are unrolled by recursing here on their enclosing body.
///
/// # Parameters
///
///   - `context`: Eager context supplying the concrete value semantics.
///   - `builder`: Builder accumulating the straight-line program, borrowed through a [`RefCell`] so the interpretation
///     closures can each take a short-lived mutable borrow.
///   - `program`: Sub-program to rewrite (the whole primal program at the top level, or a `while` body when recursing).
///   - `input_pairs`: `(value, atom)` pairs feeding `program`'s inputs, aligned with its input atoms in input order.
fn rewrite_program_into<C>(
    context: &C,
    builder: &RefCell<ProgramBuilder<<C as Domain>::Constant, <C as Domain>::Operation>>,
    program: &Program<
        <C as Domain>::Constant,
        <C as Domain>::Operation,
        Vec<<C as Domain>::Constant>,
        Vec<<C as Domain>::Constant>,
    >,
    input_pairs: Vec<(<C as Domain>::Value, AtomId)>,
) -> Result<Vec<(<C as Domain>::Value, AtomId)>, ProgramError>
where
    C: Context,
    <C as Domain>::Value: BooleanLike,
    <C as Domain>::Constant: Clone,
    <C as Domain>::Operation: Clone + MaybeWhile<<C as Domain>::Constant, <C as Domain>::Operation>,
{
    program.interpret_with::<(<C as Domain>::Value, AtomId), ProgramError, _, _>(
        input_pairs,
        // Lift each live program constant once: materialize its concrete value and emit a matching constant atom.
        |_, constant| Ok((context.lift(constant.clone())?, builder.borrow_mut().add_constant(constant.clone()))),
        |instruction, input_pairs| {
            let operation = instruction.operation();
            if let Some(parts) = operation.as_while()
                && let Some(outputs) = unroll_while_into(context, builder, &parts, input_pairs.to_vec())?
            {
                return Ok(outputs);
            }

            // Non-`while` instruction: emit it verbatim over the operands' new atoms and interpret it concretely over
            // the operands' values, zipping the result atoms and values back into paired outputs.
            let mut input_values = Vec::with_capacity(input_pairs.len());
            let mut input_atoms = Vec::with_capacity(input_pairs.len());
            for (value, atom) in input_pairs {
                input_values.push(value.clone());
                input_atoms.push(*atom);
            }
            let output_atoms = builder.borrow_mut().add_instruction(operation.clone(), input_atoms)?.to_vec();
            let output_values = context.bind(operation.clone(), &input_values)?;
            check_count!("output", output_values, output_atoms.len(), ProgramError);
            Ok(output_values.into_iter().zip(output_atoms).collect())
        },
    )
}

/// Unrolls one `while` loop, returning the final loop-carried state as `(value, atom)` pairs, or `None` when the
/// loop's predicate does not concretize to one scalar Boolean (e.g., a batched per-item predicate) and the loop must
/// therefore be kept staged verbatim by the caller.
///
/// The condition is concretized on the current concrete carry through [`BooleanLike::boolean`]; while it is true the
/// body is rewritten into `builder` over the current carry's atoms (via [`rewrite_program_into`], so a nested `while`
/// inside the body is itself unrolled) while being interpreted concretely on the current carry's values to advance it.
/// A semantic iteration bound truncates the loop once it is reached, even while the condition still produces true,
/// matching the bounded-`while` truncation semantics.
///
/// # Parameters
///
///   - `context`: Eager context supplying the concrete value semantics.
///   - `builder`: Builder accumulating the straight-line program.
///   - `parts`: Borrowed condition program, body program, and iteration bound of the `while` loop being unrolled.
///   - `carry`: `(value, atom)` pairs for the loop's initial state, aligned with the condition and body input atoms.
fn unroll_while_into<C>(
    context: &C,
    builder: &RefCell<ProgramBuilder<<C as Domain>::Constant, <C as Domain>::Operation>>,
    parts: &WhileParts<'_, <C as Domain>::Constant, <C as Domain>::Operation>,
    mut carry: Vec<(<C as Domain>::Value, AtomId)>,
) -> Result<Option<Vec<(<C as Domain>::Value, AtomId)>>, ProgramError>
where
    C: Context,
    <C as Domain>::Value: BooleanLike,
    <C as Domain>::Constant: Clone,
    <C as Domain>::Operation: Clone + MaybeWhile<<C as Domain>::Constant, <C as Domain>::Operation>,
{
    let mut completed_iterations = 0;
    loop {
        let truncated = parts.iteration_bound.is_some_and(|bound| completed_iterations >= bound);
        if truncated {
            return Ok(Some(carry));
        }

        // Concretize the condition on the current concrete carry to decide whether another iteration runs.
        let condition_values = carry.iter().map(|(value, _)| value.clone()).collect::<Vec<_>>();
        let condition_outputs = parts.condition.interpret_with(
            condition_values,
            |_, constant| context.lift(constant.clone()),
            |instruction, inputs| context.bind(instruction.operation().clone(), inputs),
        )?;
        check_count!("output", condition_outputs, 1, ProgramError);
        let predicate = match condition_outputs[0].boolean() {
            Ok(predicate) => predicate,
            // The predicate does not concretize to one scalar Boolean — e.g., a batched per-item predicate, whose
            // items stop on different iterations, has no single trip decision to unroll against. Report the loop as
            // non-unrollable so the caller keeps it staged verbatim; nothing has been emitted for it yet on the
            // first iteration. The predicate type is loop-invariant, so a later-iteration failure cannot occur once
            // the first concretization succeeds, and any such error is surfaced.
            Err(_) if completed_iterations == 0 => return Ok(None),
            Err(error) => return Err(error),
        };
        if !predicate {
            return Ok(Some(carry));
        }

        // Run one body iteration: rewrite it into the builder over the carry atoms (unrolling any nested `while`) and
        // advance the carry to the body's outputs.
        carry = rewrite_program_into(context, builder, parts.body, carry)?;
        completed_iterations += 1;
    }
}
