use super::*;

/// Applies one primitive's semantic transpose rule while transposing a staged linear program.
///
/// This helper is the local handshake between IR-level transposition and primitive-level reverse
/// rules. The surrounding transpose pass deals in atom ids, but [`LinearOperation::transpose`] is
/// expressed in terms of [`LinearTerm`] values so primitive implementations can stage new linear
/// instructions directly.
///
/// # Parameters
///   - `op`: primitive whose transpose rule should be applied.
///   - `builder`: transpose-program builder that owns the staged cotangent atoms created while
///     constructing the pullback program.
///   - `output_cotangents`: transpose-builder atom ids for the already-staged cotangents of
///     the primitive outputs.
fn transpose<V, O>(
    op: &O,
    builder: &Rc<RefCell<ProgramBuilder<ArrayType, V, O>>>,
    output_cotangents: &[AtomId],
) -> Result<Vec<Option<AtomId>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: CoreLinearProgramOperation<V> + Clone,
{
    let cotangent_terms = output_cotangents
        .iter()
        .map(|cotangent| LinearTerm::from_staged_parts(*cotangent, builder.clone()))
        .collect::<Vec<_>>();
    Ok(op
        .transpose(cotangent_terms.as_slice())?
        .into_iter()
        .map(|term| term.map(|term| term.atom))
        .collect())
}

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
    let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, V, E::LinearOperation>::new(Vec::new())));
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
        .into_typed::<Input, Output>(program.input_structure.clone())
        .build(output_tangents, program.output_structure.clone())?
        .simplified()
}

/// Transposes a linear pushforward program into its reverse-mode pullback.
///
/// This is the IR-level core of reverse-mode AD in `tracing_v2`. Higher-level helpers such as
/// [`vjp`](super::vjp) and [`grad`](super::grad) build on this operation after first producing a
/// forward linear program. `engine` is used to synthesize zero cotangents for disconnected primal
/// inputs.
#[allow(private_bounds)]
pub fn transpose_linear_program<V, Input, Output, O, E>(
    engine: &E,
    program: &Program<ArrayType, V, O, Input, Output>,
) -> Result<Program<ArrayType, V, O, Output, Input>, TracingError>
where
    V: Traceable<ArrayType>,
    E: Engine<Type = ArrayType, Value = V> + ?Sized,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: CoreLinearProgramOperation<V> + LinearAddOperation<ArrayType, V> + Clone,
{
    transpose_linear_program_with_factories(
        program,
        |builder: &Rc<RefCell<ProgramBuilder<ArrayType, V, O>>>, output_type, _| {
            Ok(builder.borrow_mut().add_input(output_type.clone()))
        },
        |builder: &Rc<RefCell<ProgramBuilder<ArrayType, V, O>>>, input_type| {
            Ok(builder.borrow_mut().add_constant(engine.zero(input_type)?))
        },
    )
}

fn transpose_linear_program_with_factories<V, Input, Output, O, F, G>(
    program: &Program<ArrayType, V, O, Input, Output>,
    mut make_output_cotangent_input: F,
    mut make_missing_input_cotangent: G,
) -> Result<Program<ArrayType, V, O, Output, Input>, TracingError>
where
    V: Traceable<ArrayType>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    F: FnMut(&Rc<RefCell<ProgramBuilder<ArrayType, V, O>>>, &ArrayType, usize) -> Result<AtomId, TracingError>,
    G: FnMut(&Rc<RefCell<ProgramBuilder<ArrayType, V, O>>>, &ArrayType) -> Result<AtomId, TracingError>,
    O: CoreLinearProgramOperation<V> + LinearAddOperation<ArrayType, V> + Clone,
{
    fn accumulate<V, O, BuilderInput, BuilderOutput>(
        builder: &Rc<RefCell<ProgramBuilder<ArrayType, V, O, BuilderInput, BuilderOutput>>>,
        adjoints: &mut [Option<AtomId>],
        atom: AtomId,
        contribution: AtomId,
    ) -> Result<(), TracingError>
    where
        V: Traceable<ArrayType>,
        O: LinearAddOperation<ArrayType, V> + Operation<ArrayType> + Clone,
        BuilderInput: Parameterized<V>,
        BuilderOutput: Parameterized<V>,
    {
        adjoints[atom.index] = Some(match adjoints[atom.index] {
            Some(existing) => {
                let mut builder_borrow = builder.borrow_mut();
                let abstract_value = builder_borrow.atoms[existing.index].r#type().into_owned();
                let output = builder_borrow.add_variable(abstract_value);
                builder_borrow.instructions.push(Instruction {
                    operation: O::linear_add_op(),
                    inputs: vec![existing, contribution],
                    outputs: vec![output],
                });
                output
            }
            None => contribution,
        });
        Ok(())
    }

    let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, V, O>::new(Vec::new())));
    let mut output_cotangent_inputs = Vec::with_capacity(program.output_ids.len());
    for (output_index, output) in program.output_ids.iter().enumerate() {
        let output_atom = program.atoms.get(output.index).ok_or(TracingError::UnboundAtomId { id: *output })?;
        let cotangent_input = make_output_cotangent_input(&builder, &output_atom.r#type(), output_index)?;
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
        let instruction_output_cotangents = instruction
            .outputs
            .iter()
            .copied()
            .map(|output| {
                Ok(match adjoints[output.index] {
                    Some(adjoint) => adjoint,
                    None => make_missing_input_cotangent(&builder, &program.atoms[output.index].r#type())?,
                })
            })
            .collect::<Result<Vec<_>, TracingError>>()?;
        let input_cotangents = transpose(&instruction.operation, &builder, instruction_output_cotangents.as_slice())?;
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
        .map(|input| {
            Ok(match adjoints[input.index] {
                Some(adjoints) => adjoints,
                None => make_missing_input_cotangent(&builder, &program.atoms[input.index].r#type())?,
            })
        })
        .collect::<Result<Vec<_>, TracingError>>()?;
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TracingError::EscapedProgramBuilder);
        }
    };
    builder
        .into_typed::<Output, Input>(program.output_structure.clone())
        .build(outputs, program.input_structure.clone())?
        .simplified()
}

/// Transposes a linear program using concrete output examples to seed the cotangent inputs.
///
/// This variant is useful when the linear program's leaf type cannot be synthesized from bare
/// [`ArrayType`] metadata alone, but the caller still has representative output values available.
/// It plays the same architectural role as [`transpose_linear_program`], but swaps metadata-driven
/// cotangent synthesis for exemplar-driven synthesis. `engine` is still used to create zero
/// cotangents for disconnected primal inputs.
#[allow(private_bounds)]
pub fn transpose_linear_program_with_output_examples<V, Input, Output, O, E>(
    engine: &E,
    program: &Program<ArrayType, V, O, Input, Output>,
    output_examples: &[V],
) -> Result<Program<ArrayType, V, O, Output, Input>, TracingError>
where
    V: Traceable<ArrayType>,
    E: Engine<Type = ArrayType, Value = V> + ?Sized,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: CoreLinearProgramOperation<V> + LinearAddOperation<ArrayType, V> + Clone,
{
    let expected_output_count = program.output_ids.len();
    if output_examples.len() != expected_output_count {
        return Err(TracingError::InvalidInputCount { expected: expected_output_count, got: output_examples.len() });
    }
    transpose_linear_program_with_factories(
        program,
        |builder: &Rc<RefCell<ProgramBuilder<ArrayType, V, O>>>, _, output_index| {
            Ok(builder.borrow_mut().add_input(output_examples[output_index].r#type().into_owned()))
        },
        |builder: &Rc<RefCell<ProgramBuilder<ArrayType, V, O>>>, input_type| {
            Ok(builder.borrow_mut().add_constant(engine.zero(input_type)?))
        },
    )
}

/// Transposes a traced linear program using an explicit outer tracing builder.
///
/// This is the traced analogue of [`transpose_linear_program`]. The transpose program itself is
/// still staged in a fresh linear-program builder, but disconnected primal inputs need zero
/// cotangents represented as traced leaves in the enclosing outer trace. `tracing_builder` is that
/// outer traced-program builder.
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
    O: CoreLinearProgramOperation<Tracer<'engine, E, TracingOperation>>
        + LinearAddOperation<ArrayType, Tracer<'engine, E, TracingOperation>>
        + Clone,
    E: Engine<Type = ArrayType, Value = V> + ?Sized + 'static,
    TracingOperation: Clone + Operation<ArrayType>,
{
    transpose_linear_program_with_factories(
        program,
        |builder: &Rc<RefCell<ProgramBuilder<ArrayType, Tracer<'engine, E, TracingOperation>, O>>>, output_type, _| {
            Ok(builder.borrow_mut().add_input(output_type.clone()))
        },
        |builder: &Rc<RefCell<ProgramBuilder<ArrayType, Tracer<'engine, E, TracingOperation>, O>>>, input_type| {
            let zero_type = input_type.clone();
            let zero_atom = tracing_builder.borrow_mut().add_constant(engine.zero(input_type)?);
            let zero_tracer = Tracer::from_staged_parts(zero_atom, zero_type, tracing_builder.clone(), engine);
            Ok(builder.borrow_mut().add_constant(zero_tracer))
        },
    )
}
