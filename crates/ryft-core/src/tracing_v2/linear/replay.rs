//! Replay helpers used while turning traced programs into linear programs.
//!
//! These helpers sit below the public autodiff APIs and above the raw staged IR. They replay an
//! existing traced program on specialized leaf wrappers so linearization can reuse the same program
//! body without re-running the original user closure.

use super::*;

/// Replays one staged program under caller-supplied leaf semantics.
///
/// This is the generic engine behind several internal replay modes: ordinary interpretation,
/// linearized JIT replay, and symbolic linearization all use the same atom-walking logic while
/// customizing how constants are lifted and how primitive instructions are applied.
fn replay_program_with<ProgramInput, ProgramOutput, V, O, R, LiftConstant, ApplyOp>(
    program: &Program<ArrayType, V, O, ProgramInput, ProgramOutput>,
    inputs: Vec<R>,
    lift_constant: LiftConstant,
    apply_op: ApplyOp,
) -> Result<Vec<R>, TracingError>
where
    ProgramInput: Parameterized<V>,
    ProgramOutput: Parameterized<V>,
    V: Traceable<ArrayType>,
    O: Clone,
    R: Clone,
    LiftConstant: Fn(&V, &[R]) -> Result<R, TracingError>,
    ApplyOp: Fn(&O, Vec<R>) -> Result<Vec<R>, TracingError>,
{
    let mut values = vec![None; program.atoms.len()];
    for (atom_id, value) in program.input_ids.iter().copied().zip(inputs.iter().cloned()) {
        values[atom_id.index] = Some(value);
    }

    let mut instruction_by_first_output = vec![None; program.atoms.len()];
    for (instruction_index, instruction) in program.instructions.iter().enumerate() {
        if let Some(first_output) = instruction.outputs.first() {
            instruction_by_first_output[first_output.index] = Some(instruction_index);
        }
    }
    let mut input_atom_flags = vec![false; program.atoms.len()];
    for input_atom in program.input_ids.iter().copied() {
        input_atom_flags[input_atom.index] = true;
    }

    for atom_index in 0..program.atoms.len() {
        let atom = &program.atoms[atom_index];
        match atom {
            Atom::Constant(value) => {
                let seed_inputs = inputs.iter().cloned().chain(values.iter().flatten().cloned()).collect::<Vec<_>>();
                if seed_inputs.is_empty() {
                    return Err(TracingError::EmptyParameterizedValue);
                }
                values[atom_index] = Some(lift_constant(value, seed_inputs.as_slice())?);
            }
            Atom::Variable(_) if input_atom_flags[atom_index] => {}
            Atom::Variable(_) => {
                let Some(instruction_index) = instruction_by_first_output[atom_index] else {
                    continue;
                };
                let instruction = &program.instructions[instruction_index];
                let input_values = instruction
                    .inputs
                    .iter()
                    .map(|input| values[input.index].clone().ok_or(TracingError::UnboundAtomId { id: *input }))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = apply_op(&instruction.operation, input_values)?;
                for (output_atom, output_value) in instruction.outputs.iter().copied().zip(outputs) {
                    values[output_atom.index] = Some(output_value);
                }
            }
        }
    }

    program
        .output_ids
        .iter()
        .map(|output| values[output.index].clone().ok_or(TracingError::UnboundAtomId { id: *output }))
        .collect()
}

/// Replays a staged program on traced dual leaves inside an outer JIT scope.
///
/// This is the key helper that lets higher-order transforms symbolically replay an already-traced
/// body while preserving both its primal outputs and its staged tangent propagation.
pub(crate) fn replay_program_linearized_jit<'engine, ProgramInput, ProgramOutput, V, O, L, E>(
    program: &Program<ArrayType, V, O, ProgramInput, ProgramOutput>,
    inputs: Vec<LinearizedTracedValue<'engine, E>>,
) -> Result<Vec<LinearizedTracedValue<'engine, E>>, TracingError>
where
    ProgramInput: Parameterized<V>,
    ProgramOutput: Parameterized<V>,
    V: Traceable<ArrayType> + ZeroLike,
    L: Clone + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
    O: InterpretableOp<ArrayType, LinearizedTracedValue<'engine, E>> + Clone + 'static,
{
    replay_program_with(program, inputs, super::program::lift_linearized_traced_constant::<V, O, L, E>, |op, values| {
        InterpretableOp::<ArrayType, LinearizedTracedValue<'engine, E>>::interpret(op, &values)
    })
}

/// Builds a staged linear program by replaying a traced primal program on symbolic dual inputs.
///
/// In the overall architecture, this is the traced-program analogue of
/// [`super::program::linearize_program`]: instead of consuming concrete primals and producing a
/// linear program immediately, it works inside an outer JIT trace and stages the resulting
/// pushforward symbolically.
pub(crate) fn linearize_traced_program<'engine, V, O, L, E>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    primals: Vec<Tracer<'engine, E>>,
) -> Result<(Vec<Tracer<'engine, E>>, TracedLinearProgram<'engine, E>), TracingError>
where
    V: Traceable<ArrayType> + ZeroLike,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
    O: InterpretableOp<ArrayType, LinearizedTracedValue<'engine, E>> + Clone,
{
    let zero = primals.first().map(ZeroLike::zero_like).ok_or(TracingError::EmptyParameterizedValue)?;
    let input_count = primals.len();
    let builder = Rc::new(RefCell::new(ProgramBuilder::<
        LinearPrimitiveOp<ArrayType, Tracer<'engine, E>>,
        ArrayType,
        Tracer<'engine, E>,
    >::new()));
    let traced_input = primals
        .into_iter()
        .map(|primal| {
            let atom = builder.borrow_mut().add_input(&primal);
            Linearized { primal, tangent: LinearTerm::from_staged_parts(atom, builder.clone()) }
        })
        .collect::<Vec<_>>();
    let traced_output = replay_program_linearized_jit::<_, _, _, O, L, E>(program, traced_input)?;
    let primal_outputs = traced_output.iter().map(|output| output.primal.clone()).collect::<Vec<_>>();
    let tangent_outputs = traced_output.iter().map(|output| output.tangent.atom()).collect::<Vec<_>>();
    drop(traced_output);
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TracingError::InternalInvariantViolation("linearization builder escaped the tracing scope"));
        }
    };
    let program = builder
        .build::<Vec<Tracer<'engine, E>>, Vec<Tracer<'engine, E>>>(
            tangent_outputs,
            vec![Placeholder; input_count],
            vec![Placeholder; primal_outputs.len()],
        )
        .simplify()?;
    Ok((primal_outputs.clone(), LinearProgram::from_program(program, zero)))
}
