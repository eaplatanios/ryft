use super::*;

use crate::tracing_v2::operations::{SupportsZero, TranspositionContext};

/// Converts a staged primal program into a staged pushforward linear map.
///
/// This is the reusable IR-level form of forward-mode differentiation. Instead of evaluating the
/// JVP immediately, it builds a staged [`Program`] over linear operations that can be replayed
/// later on arbitrary tangent inputs at the same primal point.
#[doc(hidden)]
pub fn linearize_program<Input, Output, V, E, O>(
    engine: &E,
    program: &Program<ArrayType, V, O, Input, Output>,
    input_primals: Vec<V>,
) -> Result<Program<ArrayType, V, E::LinearOperation, Input, Output>, TracingError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    E: DifferentiableEngine<Type = ArrayType, Value = V> + ?Sized,
    O: Clone + DifferentiableOperation<E>,
{
    fn tangent_for_atom<V, Input, Output, ProgramOperation, LinearOperation>(
        _program: &Program<ArrayType, V, ProgramOperation, Input, Output>,
        primal_values: &[Option<V>],
        builder: &Rc<RefCell<ProgramBuilder<ArrayType, V, LinearOperation>>>,
        tangents: &mut [Option<LinearTerm<ArrayType, V, LinearOperation>>],
        atom_id: AtomId,
    ) -> Result<LinearTerm<ArrayType, V, LinearOperation>, TracingError>
    where
        V: Traceable<ArrayType> + ZeroLike,
        Input: Parameterized<V>,
        Output: Parameterized<V>,
        ProgramOperation: Clone + Operation<ArrayType>,
        LinearOperation: Clone + Operation<ArrayType>,
    {
        if let Some(term) = tangents[atom_id.index].clone() {
            return Ok(term);
        }
        let primal = primal_values[atom_id.index].as_ref().ok_or(TracingError::UnboundAtomId { id: atom_id })?;
        let tangent_atom = builder.borrow_mut().add_constant(primal.zero_like());
        let tangent = LinearTerm::from_staged_parts(tangent_atom, builder.clone());
        tangents[atom_id.index] = Some(tangent.clone());
        Ok(tangent)
    }

    let program = program;
    if input_primals.len() != program.input_ids.len() {
        return Err(TracingError::InvalidInputCount { expected: program.input_ids.len(), got: input_primals.len() });
    }
    let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, V, E::LinearOperation>::new()));
    let mut primals: Vec<Option<V>> = vec![None; program.atoms.len()];
    let mut tangents: Vec<Option<LinearTerm<ArrayType, V, E::LinearOperation>>> = vec![None; program.atoms.len()];
    for (input_atom, input_primal) in program.input_ids.iter().copied().zip(input_primals.into_iter()) {
        let tangent_atom = builder.borrow_mut().add_input(input_primal.r#type().into_owned());
        tangents[input_atom.index] = Some(LinearTerm::from_staged_parts(tangent_atom, builder.clone()));
        primals[input_atom.index] = Some(input_primal);
    }
    for (atom_index, atom) in program.atoms.iter().enumerate() {
        let atom_id = AtomId { index: atom_index };
        if let Atom::Constant(value) = atom {
            primals[atom_id.index] = Some(value.clone());
        }
    }

    for instruction in &program.instructions {
        let input_duals = instruction
            .inputs
            .iter()
            .copied()
            .map(|input_atom| {
                Ok(JvpTracer {
                    primal: primals[input_atom.index].clone().ok_or(TracingError::UnboundAtomId { id: input_atom })?,
                    tangent: tangent_for_atom(
                        program,
                        primals.as_slice(),
                        &builder,
                        tangents.as_mut_slice(),
                        input_atom,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, TracingError>>()?;
        let output_duals = instruction.operation.jvp(engine, input_duals.as_slice())?;
        if output_duals.len() != instruction.outputs.len() {
            return Err(TracingError::InvalidOutputCount {
                expected: instruction.outputs.len(),
                got: output_duals.len(),
            });
        }
        for (output_atom, output_dual) in instruction.outputs.iter().copied().zip(output_duals.into_iter()) {
            primals[output_atom.index] = Some(output_dual.primal);
            tangents[output_atom.index] = Some(output_dual.tangent);
        }
    }

    let output_tangents = program
        .output_ids
        .iter()
        .copied()
        .map(|output_atom| {
            tangent_for_atom(program, primals.as_slice(), &builder, tangents.as_mut_slice(), output_atom)
                .map(|term| term.atom)
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(tangents);
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TracingError::EscapedProgramBuilder);
        }
    };
    builder
        .build(output_tangents, program.input_structure.clone(), program.output_structure.clone())?
        .simplified()
}

fn transpose_linear_program_with_context<V, Input, Output, O>(
    context: &mut TranspositionContext<'_, ArrayType, V, O>,
    program: &Program<ArrayType, V, O, Input, Output>,
) -> Result<Program<ArrayType, V, O, Output, Input>, TracingError>
where
    V: Traceable<ArrayType>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: Clone
        + InterpretableOperation<ArrayType, V>
        + LinearOperation<ArrayType, V, O>
        + SupportsAdd<ArrayType, V>
        + SupportsZero<ArrayType, V>,
{
    fn accumulate<V, O>(
        builder: &Rc<RefCell<ProgramBuilder<ArrayType, V, O>>>,
        adjoints: &mut [Option<AtomId>],
        atom: AtomId,
        contribution: AtomId,
    ) -> Result<(), TracingError>
    where
        V: Traceable<ArrayType>,
        O: SupportsAdd<ArrayType, V> + Operation<ArrayType> + Clone,
    {
        adjoints[atom.index] = Some(match adjoints[atom.index] {
            Some(existing) => {
                let mut builder_borrow = builder.borrow_mut();
                let abstract_value = builder_borrow.atoms[existing.index].r#type().into_owned();
                let output = builder_borrow.add_variable(abstract_value);
                builder_borrow.instructions.push(Instruction {
                    operation: O::add_operation(),
                    inputs: vec![existing, contribution],
                    outputs: vec![output],
                });
                output
            }
            None => contribution,
        });
        Ok(())
    }

    fn stage_zero<V, O>(builder: &Rc<RefCell<ProgramBuilder<ArrayType, V, O>>>, r#type: ArrayType) -> AtomId
    where
        V: Traceable<ArrayType>,
        O: SupportsZero<ArrayType, V> + Operation<ArrayType> + Clone,
    {
        let mut builder_borrow = builder.borrow_mut();
        let output = builder_borrow.add_variable(r#type.clone());
        builder_borrow.instructions.push(Instruction {
            operation: <O as SupportsZero<ArrayType, V>>::zero_operation(r#type),
            inputs: vec![],
            outputs: vec![output],
        });
        output
    }

    let builder = context.builder().clone();
    let mut output_cotangent_inputs = Vec::with_capacity(program.output_ids.len());
    for output in program.output_ids.iter() {
        let output_atom = program.atoms.get(output.index).ok_or(TracingError::UnboundAtomId { id: *output })?;
        let cotangent_input = builder.borrow_mut().add_input(output_atom.r#type().into_owned());
        output_cotangent_inputs.push(cotangent_input);
    }

    let mut adjoints = vec![None; program.atoms.len()];
    for (cotangent, output) in output_cotangent_inputs.into_iter().zip(program.output_ids.iter().copied()) {
        accumulate(&builder, adjoints.as_mut_slice(), output, cotangent)?;
    }

    for instruction in program.instructions.iter().rev() {
        if instruction.outputs.iter().all(|output| adjoints[output.index].is_none()) {
            continue;
        }
        let instruction_output_cotangents =
            instruction.outputs.iter().map(|output| adjoints[output.index]).collect::<Vec<_>>();
        let input_cotangents = instruction.operation.transpose(context, instruction_output_cotangents.as_slice())?;
        for (input, contribution) in instruction.inputs.iter().copied().zip(input_cotangents) {
            if let Some(contribution) = contribution {
                accumulate(&builder, adjoints.as_mut_slice(), input, contribution)?;
            }
        }
    }

    let outputs = program
        .input_ids
        .iter()
        .copied()
        .map(|input| match adjoints[input.index] {
            Some(adjoint) => adjoint,
            None => stage_zero::<V, O>(&builder, program.atoms[input.index].r#type().into_owned()),
        })
        .collect::<Vec<_>>();
    drop(builder);
    let builder = context.take_builder()?;
    builder
        .build(outputs, program.output_structure.clone(), program.input_structure.clone())?
        .simplified()
}

impl<V, O> TranspositionContext<'_, ArrayType, V, O>
where
    V: Traceable<ArrayType>,
    O: Clone
        + InterpretableOperation<ArrayType, V>
        + LinearOperation<ArrayType, V, O>
        + SupportsAdd<ArrayType, V>
        + SupportsZero<ArrayType, V>,
{
    /// Transposes one nested linear program into a fresh sibling builder.
    pub(crate) fn transpose_nested_program<Input, Output>(
        &mut self,
        program: &Program<ArrayType, V, O, Input, Output>,
    ) -> Result<Program<ArrayType, V, O, Output, Input>, TracingError>
    where
        Input: Parameterized<V, ParameterStructure: Clone>,
        Output: Parameterized<V, ParameterStructure: Clone>,
    {
        let parent_builder = self.replace_builder(Rc::new(RefCell::new(ProgramBuilder::new())));
        let result = transpose_linear_program_with_context(self, program);
        let _nested_builder = self.replace_builder(parent_builder);
        result
    }
}

/// Transposes a linear pushforward program into its reverse-mode pullback.
///
/// This is the IR-level core of reverse-mode AD in `tracing_v2`. Higher-level helpers such as
/// [`vjp`](super::vjp) and [`grad`](super::grad) build on this operation after first producing a
/// forward linear program. `output_examples` carries one representative value per primal output
/// and is only used to validate the output count; cotangent input types come from the program's
/// own atom metadata. Disconnected primal inputs are emitted as
/// [`LinearPrimitiveOperation::Zero`](crate::tracing_v2::LinearPrimitiveOperation::Zero) ops, which
/// the value type's [`Zero<ArrayType>`](crate::tracing_v2::operations::constants::Zero)
/// implementation evaluates at interpretation time.
pub fn transpose_linear_program_with_output_examples<V, Input, Output, O>(
    program: &Program<ArrayType, V, O, Input, Output>,
    output_examples: &[V],
) -> Result<Program<ArrayType, V, O, Output, Input>, TracingError>
where
    V: Traceable<ArrayType>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: Clone
        + InterpretableOperation<ArrayType, V>
        + LinearOperation<ArrayType, V, O>
        + SupportsAdd<ArrayType, V>
        + SupportsZero<ArrayType, V>,
{
    let expected_output_count = program.output_ids.len();
    if output_examples.len() != expected_output_count {
        return Err(TracingError::InvalidInputCount { expected: expected_output_count, got: output_examples.len() });
    }
    let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, V, O>::new()));
    let mut context = TranspositionContext::new(builder);
    transpose_linear_program_with_context(&mut context, program)
}

/// Transposes a traced linear program using an explicit outer tracing builder.
///
/// This is the traced analogue of [`transpose_linear_program_with_output_examples`]. The transpose
/// program itself is staged in a fresh linear-program builder, then any
/// [`LinearPrimitiveOperation::Zero`](crate::tracing_v2::LinearPrimitiveOperation::Zero) op
/// produced for a disconnected primal input is materialized into a
/// [`Tracer`](crate::tracing_v2::Tracer) constant whose underlying outer-trace atom holds
/// `engine.zero(t)`. After this materialization the returned pullback contains no `Zero` ops, so
/// the standard interpret path applies.
#[allow(private_bounds)]
pub fn transpose_traced_linear_program<'engine, Input, Output, V, O, E, TracingOperation>(
    engine: &'engine E,
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, V, TracingOperation>>>,
    program: &Program<ArrayType, Tracer<'engine, E, TracingOperation>, O, Input, Output>,
) -> Result<Program<ArrayType, Tracer<'engine, E, TracingOperation>, O, Output, Input>, TracingError>
where
    V: Traceable<ArrayType>,
    Input: Parameterized<Tracer<'engine, E, TracingOperation>, ParameterStructure: Clone>,
    Output: Parameterized<Tracer<'engine, E, TracingOperation>, ParameterStructure: Clone>,
    O: Clone
        + InterpretableOperation<ArrayType, Tracer<'engine, E, TracingOperation>>
        + LinearOperation<ArrayType, Tracer<'engine, E, TracingOperation>, O>
        + SupportsAdd<ArrayType, Tracer<'engine, E, TracingOperation>>
        + SupportsZero<ArrayType, Tracer<'engine, E, TracingOperation>>,
    E: Engine<Type = ArrayType, Value = V> + ?Sized + 'static,
    TracingOperation: Clone + Operation<ArrayType>,
{
    let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, Tracer<'engine, E, TracingOperation>, O>::new()));
    let mut context = TranspositionContext::new(builder);
    let pullback = transpose_linear_program_with_context(&mut context, program)?;
    materialize_tracer_zero_ops(pullback, engine, tracing_builder)
}

/// Walks a linear program and replaces every
/// [`LinearPrimitiveOperation::Zero`](crate::tracing_v2::LinearPrimitiveOperation::Zero) op with a
/// [`Tracer`] constant atom backed by an outer-trace zero.
///
/// This is the post-processing step that the traced reverse-mode pipeline runs after transposition
/// so the returned pullback is interpretable by code that holds [`Tracer`] inputs. Tracer values
/// cannot satisfy [`Zero<ArrayType>`](crate::tracing_v2::operations::constants::Zero) statically,
/// so traced pullbacks must be materialized away from `Zero` ops before being interpreted.
fn materialize_tracer_zero_ops<'engine, Input, Output, V, O, E, TracingOperation>(
    program: Program<ArrayType, Tracer<'engine, E, TracingOperation>, O, Input, Output>,
    engine: &'engine E,
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, V, TracingOperation>>>,
) -> Result<Program<ArrayType, Tracer<'engine, E, TracingOperation>, O, Input, Output>, TracingError>
where
    V: Traceable<ArrayType>,
    Input: Parameterized<Tracer<'engine, E, TracingOperation>, ParameterStructure: Clone>,
    Output: Parameterized<Tracer<'engine, E, TracingOperation>, ParameterStructure: Clone>,
    O: Clone
        + InterpretableOperation<ArrayType, Tracer<'engine, E, TracingOperation>>
        + LinearOperation<ArrayType, Tracer<'engine, E, TracingOperation>, O>
        + SupportsAdd<ArrayType, Tracer<'engine, E, TracingOperation>>
        + SupportsZero<ArrayType, Tracer<'engine, E, TracingOperation>>,
    E: Engine<Type = ArrayType, Value = V> + ?Sized + 'static,
    TracingOperation: Clone + Operation<ArrayType>,
{
    let mut builder = ProgramBuilder::<ArrayType, Tracer<'engine, E, TracingOperation>, O>::new();
    builder.atoms = program.atoms.clone();
    builder.input_ids = program.input_ids.clone();
    let mut atom_remapping: Vec<Option<AtomId>> = vec![None; builder.atoms.len()];
    let mut rewritten_instructions = Vec::with_capacity(program.instructions.len());
    for instruction in &program.instructions {
        if let Some(zero_type) = instruction.operation.as_zero()
            && instruction.outputs.len() == 1
            && instruction.inputs.is_empty()
        {
            let zero_value = engine.zero(zero_type)?;
            let outer_atom = tracing_builder.borrow_mut().add_constant(zero_value);
            let zero_tracer = Tracer::from_staged_parts(outer_atom, zero_type.clone(), tracing_builder.clone(), engine);
            let constant_atom = builder.add_constant(zero_tracer);
            atom_remapping[instruction.outputs[0].index] = Some(constant_atom);
        } else {
            let inputs = instruction
                .inputs
                .iter()
                .map(|atom| atom_remapping[atom.index].unwrap_or(*atom))
                .collect::<Vec<_>>();
            rewritten_instructions.push(Instruction {
                operation: instruction.operation.clone(),
                inputs,
                outputs: instruction.outputs.clone(),
            });
        }
    }
    builder.instructions = rewritten_instructions;
    let outputs = program
        .output_ids
        .iter()
        .map(|atom| atom_remapping[atom.index].unwrap_or(*atom))
        .collect::<Vec<_>>();
    builder
        .build(outputs, program.input_structure.clone(), program.output_structure.clone())?
        .simplified()
}
