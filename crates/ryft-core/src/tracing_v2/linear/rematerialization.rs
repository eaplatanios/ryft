use super::*;

fn build_traced_gradient_program<'engine, E, Input, V>(
    engine: &'engine E,
    input_structure: Input::ParameterStructure,
    traced_program: &Program<ArrayType, V, E::Operation, Vec<V>, Vec<V>>,
) -> Result<Program<ArrayType, V, E::Operation, Input, Input>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
    V: Value<ArrayType> + Differentiable<ArrayType, Tangent = V> + One<ArrayType>,
    E::Operation: InterpretableOperation<ArrayType, V> + TracedLinearizableOperation<'engine, E> + 'static,
    LinearPrimitiveOperation<Tracer<'engine, E>>: Clone
        + InterpretableOperation<ArrayType, Tracer<'engine, E>>
        + LinearOperation<ArrayType, Tracer<'engine, E>, LinearPrimitiveOperation<Tracer<'engine, E>>>,
    Input: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
{
    let traced_primal_builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, V, E::Operation>::new()));
    let traced_primals = traced_program
        .input_ids
        .iter()
        .map(|input_atom| {
            let input_type = traced_program.atoms[input_atom.index].r#type().into_owned();
            let atom = traced_primal_builder.borrow_mut().add_input(input_type.clone());
            Tracer::from_staged_parts(atom, input_type, traced_primal_builder.clone(), engine)
        })
        .collect::<Vec<_>>();
    let (_, traced_gradient) = reverse_mode_scalar_traced_program::<V, E>(
        engine,
        traced_primal_builder.clone(),
        traced_program,
        traced_primals,
    )?;
    if let Some(tracing_error) = traced_primal_builder.borrow_mut().error.take() {
        return Err(tracing_error);
    }
    let gradient_output_atoms =
        traced_gradient.into_iter().map(|output| output.atom_id()).collect::<Result<Vec<_>, _>>()?;
    let traced_primal_builder = match Rc::try_unwrap(traced_primal_builder) {
        Ok(traced_primal_builder) => traced_primal_builder.into_inner(),
        Err(_) => {
            return Err(TracingError::EscapedProgramBuilder);
        }
    };
    traced_primal_builder
        .build(gradient_output_atoms, input_structure.clone(), input_structure)?
        .simplified()
}

/// Compiles a reverse-mode gradient function into a reusable staged program.
///
/// Unlike [`grad`](super::grad), which returns concrete gradient values at one primal point, this
/// function returns a staged [`Program`] whose inputs are primals and whose outputs are gradients.
/// In the larger architecture, it is the "compile the whole reverse-mode pipeline" entry point:
/// the traced forward pass, the linearization, and the pullback application are all baked into one
/// reusable artifact.
#[allow(private_bounds)]
pub fn compile_grad<'engine, E, F, Input, V>(
    _engine: &'engine E,
    function: F,
    example_primals: Input,
) -> Result<Program<ArrayType, V, E::Operation, Input, Input>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
    V: Value<ArrayType> + Differentiable<ArrayType, Tangent = V> + One<ArrayType>,
    E::Operation: InterpretableOperation<ArrayType, V> + TracedLinearizableOperation<'engine, E>,
    LinearPrimitiveOperation<DifferentiableTracer<'engine, E>>: Clone
        + InterpretableOperation<ArrayType, DifferentiableTracer<'engine, E>>
        + LinearOperation<
            ArrayType,
            DifferentiableTracer<'engine, E>,
            LinearPrimitiveOperation<DifferentiableTracer<'engine, E>>,
        >,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    Vec<V>: Parameterized<V, ParameterStructure = Vec<Placeholder>>,
    Input: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Input::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    Input::To<ArrayType>:
        Parameterized<ArrayType, To<DifferentiableTracer<'engine, E>> = Input::To<DifferentiableTracer<'engine, E>>>,
    V::To<ArrayType>: Parameterized<ArrayType, To<DifferentiableTracer<'engine, E>> = DifferentiableTracer<'engine, E>>,
    F: Fn(Input::To<DifferentiableTracer<'engine, E>>) -> DifferentiableTracer<'engine, E>,
{
    let input_structure = example_primals.parameter_structure();
    let staged_input_types = Input::To::<ArrayType>::from_parameters(
        input_structure.clone(),
        example_primals.parameters().map(|primal| primal.r#type().into_owned()).collect::<Vec<_>>(),
    )?;
    let (_, traced_program) = trace_flat_program_from_input_types::<Input::To<ArrayType>, V::To<ArrayType>, V, E, _>(
        _engine,
        |staged_input| Ok(function(staged_input)),
        staged_input_types,
    )?;
    build_traced_gradient_program(_engine, input_structure, &traced_program)
}

/// Policy controlling how forward-pass intermediates are handled during reverse-mode differentiation.
///
/// Rematerialization is the place where `tracing_v2` exposes a memory-vs-recomputation choice to
/// callers. The policy does not change the mathematical gradient; it changes where the staged
/// reverse-mode program inserts rematerialization boundaries, which in turn affects what a backend
/// may save versus recompute during the backward pass.
#[derive(Clone, Debug)]
pub enum RematerializationPolicy {
    /// Save all forward-pass intermediates (maximum memory, no recomputation).
    SaveAll,

    /// Recompute all forward-pass intermediates from inputs (minimum memory, maximum recomputation).
    RecomputeAll,

    /// Save intermediates every `segment_size` instructions, recomputing within each segment.
    ///
    /// With a program of N instructions, setting `segment_size` to approximately the square root of
    /// N gives O(sqrt(N)) memory usage. A `segment_size` of zero or one degenerates to
    /// [`SaveAll`](RematerializationPolicy::SaveAll) since each segment contains at most one
    /// instruction.
    Checkpoint {
        /// Number of instructions per rematerialization segment.
        segment_size: usize,
    },
}

/// Compiles a reverse-mode gradient function with an explicit rematerialization policy.
///
/// This generalizes [`compile_grad`] by letting the caller control how forward-pass intermediates are handled
/// during the backward pass:
///
///   - [`RematerializationPolicy::SaveAll`]: identical to [`compile_grad`] - no rematerialization boundaries are
///     inserted, so the XLA compiler decides which intermediates to save.
///   - [`RematerializationPolicy::RecomputeAll`]: the entire forward body is wrapped in a single
///     [`rematerialize`] boundary, forcing the backward pass to recompute all intermediates from inputs.
///   - [`RematerializationPolicy::Checkpoint`]: the forward body is partitioned into segments of at most
///     `segment_size` instructions, each wrapped in its own [`rematerialize`] boundary. Intermediates at segment
///     boundaries are saved while within-segment intermediates are recomputed.
#[allow(private_bounds)]
pub fn compile_grad_with_policy<'engine, E, F, Input, V>(
    engine: &'engine E,
    function: F,
    example_primals: Input,
    policy: RematerializationPolicy,
) -> Result<Program<ArrayType, V, E::Operation, Input, Input>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
    V: Value<ArrayType> + Differentiable<ArrayType, Tangent = V> + One<ArrayType>,
    E::Operation: InterpretableOperation<ArrayType, V>
        + TracedLinearizableOperation<'engine, E>
        + SupportsRematerialize<ArrayType, V, E::LinearOperation>,
    LinearPrimitiveOperation<DifferentiableTracer<'engine, E>>: Clone
        + InterpretableOperation<ArrayType, DifferentiableTracer<'engine, E>>
        + LinearOperation<
            ArrayType,
            DifferentiableTracer<'engine, E>,
            LinearPrimitiveOperation<DifferentiableTracer<'engine, E>>,
        >,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    Vec<V>: Parameterized<V, ParameterStructure = Vec<Placeholder>>,
    Input: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Input::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    Input::To<ArrayType>:
        Parameterized<ArrayType, To<DifferentiableTracer<'engine, E>> = Input::To<DifferentiableTracer<'engine, E>>>,
    V::To<ArrayType>: Parameterized<ArrayType, To<DifferentiableTracer<'engine, E>> = DifferentiableTracer<'engine, E>>,
    F: Fn(Input::To<DifferentiableTracer<'engine, E>>) -> DifferentiableTracer<'engine, E>,
{
    match policy {
        RematerializationPolicy::SaveAll => compile_grad(engine, &function, example_primals),
        RematerializationPolicy::RecomputeAll => compile_grad_segmented(engine, &function, example_primals, None),
        RematerializationPolicy::Checkpoint { segment_size } => {
            if segment_size <= 1 {
                return compile_grad(engine, &function, example_primals);
            }
            compile_grad_segmented(engine, &function, example_primals, Some(segment_size))
        }
    }
}

/// Compiles a gradient function with rematerialization boundaries inserted via program segmentation.
///
/// When `segment_size` is `None`, the entire program is wrapped in a single [`RematerializeOperation`]
/// (equivalent to [`RematerializationPolicy::RecomputeAll`]). When `Some(s)`, the program is
/// partitioned into segments of at most `s` instructions, each wrapped in its own [`RematerializeOperation`].
///
/// Internally, this replicates the flow of `grad` for [`Tracer`]-level inputs - trace, linearize,
/// transpose, stage pullback - but inserts a segmentation step between tracing and linearization so
/// that the differentiation transform sees and respects the rematerialization boundaries.
fn compile_grad_segmented<'engine, E, F, Input, V>(
    engine: &'engine E,
    function: &F,
    example_primals: Input,
    segment_size: Option<usize>,
) -> Result<Program<ArrayType, V, E::Operation, Input, Input>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
    V: Value<ArrayType> + Differentiable<ArrayType, Tangent = V> + One<ArrayType>,
    E::Operation: InterpretableOperation<ArrayType, V>
        + TracedLinearizableOperation<'engine, E>
        + SupportsRematerialize<ArrayType, V, E::LinearOperation>,
    LinearPrimitiveOperation<DifferentiableTracer<'engine, E>>: Clone
        + InterpretableOperation<ArrayType, DifferentiableTracer<'engine, E>>
        + LinearOperation<
            ArrayType,
            DifferentiableTracer<'engine, E>,
            LinearPrimitiveOperation<DifferentiableTracer<'engine, E>>,
        >,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    Vec<V>: Parameterized<V, ParameterStructure = Vec<Placeholder>>,
    Input: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Input::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    Input::To<ArrayType>:
        Parameterized<ArrayType, To<DifferentiableTracer<'engine, E>> = Input::To<DifferentiableTracer<'engine, E>>>,
    V::To<ArrayType>: Parameterized<ArrayType, To<DifferentiableTracer<'engine, E>> = DifferentiableTracer<'engine, E>>,
    F: Fn(Input::To<DifferentiableTracer<'engine, E>>) -> DifferentiableTracer<'engine, E>,
{
    let input_structure = example_primals.parameter_structure();
    let staged_input_types = Input::To::<ArrayType>::from_parameters(
        input_structure.clone(),
        example_primals.parameters().map(|primal| primal.r#type().into_owned()).collect::<Vec<_>>(),
    )?;
    let (_, traced_program) = trace_flat_program_from_input_types::<Input::To<ArrayType>, V::To<ArrayType>, V, E, _>(
        engine,
        |staged_input| Ok(function(staged_input)),
        staged_input_types,
    )?;
    let segmented_program = match segment_size {
        None => wrap_program_in_rematerialize::<E, V, E::Operation>(&traced_program)?,
        Some(size) => segment_program::<E, V, E::Operation>(&traced_program, size)?,
    };
    build_traced_gradient_program(engine, input_structure, &segmented_program)
}

/// Partitions a program's instructions into segments of at most `segment_size`, wrapping each
/// segment in a
/// [`RematerializeOperation`].
///
/// Given a program with N instructions and a segment size S, this produces a new program with at
/// most `ceil(N / S)` instructions. Each instruction is a [`RematerializeOperation`] whose body
/// sub-program contains the original instructions from that segment. Atoms crossing segment
/// boundaries become inputs/outputs of the
/// respective sub-programs.
///
/// The segmented program is semantically equivalent to the original: calling it on the same inputs produces
/// the same outputs. The difference is visible only during differentiation, where each [`RematerializeOperation`]
/// boundary forces recomputation of within-segment intermediates rather than saving them.
fn segment_program<E, V, O>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    segment_size: usize,
) -> Result<Program<ArrayType, V, O, Vec<V>, Vec<V>>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType> + Differentiable<ArrayType, Tangent = V>,
    O: Clone
        + Operation<ArrayType>
        + InterpretableOperation<ArrayType, V>
        + SupportsRematerialize<ArrayType, V, E::LinearOperation>,
{
    let program = program;
    let instructions = program.instructions.as_slice();

    // If the program has fewer instructions than a single segment, no segmentation is needed - wrap the
    // whole thing in a single RematerializeOperation.
    if instructions.len() <= segment_size {
        return wrap_program_in_rematerialize::<E, V, O>(program);
    }

    // Divide instructions into segments.
    let segments: Vec<&[Instruction<O>]> = instructions.chunks(segment_size).collect();

    // Build a mapping from atom ID to which instruction produces it (if any).
    let mut atom_producer: Vec<Option<usize>> = vec![None; program.atoms.len()];
    for (instruction_index, instruction) in instructions.iter().enumerate() {
        for &output_atom in &instruction.outputs {
            atom_producer[output_atom.index] = Some(instruction_index);
        }
    }

    // Build a set tracking which atoms are consumed after a given instruction index.
    // For each atom, track all instruction indices that consume it.
    let mut atom_consumers: Vec<Vec<usize>> = vec![Vec::new(); program.atoms.len()];
    for (instruction_index, instruction) in instructions.iter().enumerate() {
        for &input_atom in &instruction.inputs {
            atom_consumers[input_atom.index].push(instruction_index);
        }
    }
    // Also mark program outputs as "consumed" at instruction_count (sentinel for "after all instructions").
    let sentinel = instructions.len();
    for &output_atom in &program.output_ids {
        atom_consumers[output_atom.index].push(sentinel);
    }

    // Build the outer program.
    let input_atoms = program.input_ids.as_slice();
    let mut outer_builder: ProgramBuilder<ArrayType, V, O> = ProgramBuilder::new();

    // Map from original atom IDs to outer-program atom IDs.
    let mut atom_mapping: Vec<Option<AtomId>> = vec![None; program.atoms.len()];

    // Register program inputs in the outer builder.
    for &input_atom in input_atoms {
        let input_type = program
            .atoms
            .get(input_atom.index)
            .ok_or(TracingError::UnboundAtomId { id: input_atom })?
            .r#type()
            .into_owned();
        let outer_atom = outer_builder.add_input(input_type);
        atom_mapping[input_atom.index] = Some(outer_atom);
    }

    // Register constants that are used by instructions (they might be referenced across segments).
    for (atom_index, atom) in program.atoms.iter().enumerate() {
        let atom_id = AtomId { index: atom_index };
        if let Atom::Constant(value) = atom {
            let outer_atom = outer_builder.add_constant(value.clone());
            atom_mapping[atom_id.index] = Some(outer_atom);
        }
    }

    // Process each segment.
    let mut instruction_offset = 0;
    for segment in &segments {
        let segment_start = instruction_offset;
        let segment_end = instruction_offset + segment.len();

        // Identify boundary inputs: atoms consumed by this segment that are produced outside it
        // (by previous segments or program inputs/constants).
        let mut boundary_input_atoms: Vec<AtomId> = Vec::new();
        let mut boundary_input_set = std::collections::HashSet::new();
        for instruction in *segment {
            for &input_atom in &instruction.inputs {
                // If this atom is produced by an instruction outside this segment (or is an input/constant).
                let produced_in_segment = atom_producer[input_atom.index]
                    .map_or(false, |producer_idx| producer_idx >= segment_start && producer_idx < segment_end);
                if !produced_in_segment && boundary_input_set.insert(input_atom) {
                    boundary_input_atoms.push(input_atom);
                }
            }
        }

        // Identify boundary outputs: atoms produced by this segment that are consumed outside it
        // (by later segments or as program outputs).
        let mut boundary_output_atoms: Vec<AtomId> = Vec::new();
        let mut boundary_output_set = std::collections::HashSet::new();
        for instruction in *segment {
            for &output_atom in &instruction.outputs {
                let consumed_outside = atom_consumers[output_atom.index]
                    .iter()
                    .any(|&consumer_idx| consumer_idx < segment_start || consumer_idx >= segment_end);
                if consumed_outside && boundary_output_set.insert(output_atom) {
                    boundary_output_atoms.push(output_atom);
                }
            }
        }

        // Build the sub-program for this segment.
        let sub_program = build_segment_sub_program(program, *segment, &boundary_input_atoms, &boundary_output_atoms)?;

        // Build the RematerializeOperation.
        let input_types: Vec<_> = boundary_input_atoms
            .iter()
            .map(|&atom_id| {
                program
                    .atoms
                    .get(atom_id.index)
                    .ok_or(TracingError::UnboundAtomId { id: atom_id })
                    .map(|atom| atom.r#type().into_owned())
            })
            .collect::<Result<_, _>>()?;
        let output_types: Vec<_> = boundary_output_atoms
            .iter()
            .map(|&atom_id| {
                program
                    .atoms
                    .get(atom_id.index)
                    .ok_or(TracingError::UnboundAtomId { id: atom_id })
                    .map(|atom| atom.r#type().into_owned())
            })
            .collect::<Result<_, _>>()?;

        let body = FlatTracedRematerialize::from_parts(input_types.clone(), output_types.clone(), sub_program);
        let remat_op = RematerializeOperation::new(body);

        // Add the RematerializeOperation instruction to the outer builder.
        let outer_inputs: Vec<AtomId> = boundary_input_atoms
            .iter()
            .map(|&orig_atom| atom_mapping[orig_atom.index].ok_or(TracingError::UnboundAtomId { id: orig_atom }))
            .collect::<Result<_, _>>()?;
        let outer_outputs =
            output_types.into_iter().map(|r#type| outer_builder.add_variable(r#type)).collect::<Vec<_>>();
        outer_builder.instructions.push(Instruction {
            operation: O::rematerialize_operation(remat_op),
            inputs: outer_inputs,
            outputs: outer_outputs.clone(),
        });

        // Map the boundary output atoms to their outer-program counterparts.
        for (orig_atom, outer_atom) in boundary_output_atoms.iter().zip(outer_outputs.iter()) {
            atom_mapping[orig_atom.index] = Some(*outer_atom);
        }

        instruction_offset = segment_end;
    }

    // Wire up the program outputs.
    let outer_outputs: Vec<AtomId> = program
        .output_ids
        .iter()
        .map(|&orig_atom| atom_mapping[orig_atom.index].ok_or(TracingError::UnboundAtomId { id: orig_atom }))
        .collect::<Result<_, _>>()?;

    let outer_program = outer_builder.build(
        outer_outputs,
        flat_leaf_parameter_structure(input_atoms.len()),
        flat_leaf_parameter_structure(program.output_ids.len()),
    )?;
    Ok(outer_program)
}

/// Wraps an entire program in a single [`RematerializeOperation`] boundary.
fn wrap_program_in_rematerialize<E, V, O>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
) -> Result<Program<ArrayType, V, O, Vec<V>, Vec<V>>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType> + Differentiable<ArrayType, Tangent = V>,
    O: Clone + Operation<ArrayType> + SupportsRematerialize<ArrayType, V, E::LinearOperation>,
{
    let program = program;
    let input_types: Vec<_> = program
        .input_ids
        .iter()
        .map(|&atom_id| {
            program
                .atoms
                .get(atom_id.index)
                .ok_or(TracingError::UnboundAtomId { id: atom_id })
                .map(|atom| atom.r#type().into_owned())
        })
        .collect::<Result<_, _>>()?;
    let output_types: Vec<_> = program
        .output_ids
        .iter()
        .map(|&atom_id| {
            program
                .atoms
                .get(atom_id.index)
                .ok_or(TracingError::UnboundAtomId { id: atom_id })
                .map(|atom| atom.r#type().into_owned())
        })
        .collect::<Result<_, _>>()?;

    let body = FlatTracedRematerialize::from_parts(input_types.clone(), output_types.clone(), program.clone());
    let remat_op = RematerializeOperation::new(body);

    let mut outer_builder: ProgramBuilder<ArrayType, V, O> = ProgramBuilder::new();
    let outer_inputs: Vec<AtomId> =
        input_types.iter().cloned().map(|input_type| outer_builder.add_input(input_type)).collect();

    let outer_outputs = output_types.into_iter().map(|r#type| outer_builder.add_variable(r#type)).collect::<Vec<_>>();
    outer_builder.instructions.push(Instruction {
        operation: O::rematerialize_operation(remat_op),
        inputs: outer_inputs.clone(),
        outputs: outer_outputs.clone(),
    });

    let outer_program = outer_builder.build(
        outer_outputs,
        flat_leaf_parameter_structure(outer_inputs.len()),
        flat_leaf_parameter_structure(program.output_ids.len()),
    )?;
    Ok(outer_program)
}

/// Builds a sub-program for a single segment of instructions.
///
/// The sub-program takes the boundary input atoms as its inputs and produces the boundary output atoms as its
/// outputs. Internal atoms (produced and consumed entirely within the segment) are handled as internal constants
/// and instructions within the sub-program.
fn build_segment_sub_program<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    segment_instructions: &[Instruction<O>],
    boundary_input_atoms: &[AtomId],
    boundary_output_atoms: &[AtomId],
) -> Result<Program<ArrayType, V, O, Vec<V>, Vec<V>>, TracingError> {
    let mut sub_builder: ProgramBuilder<ArrayType, V, O> = ProgramBuilder::new();

    // Map from original atom IDs to sub-program atom IDs.
    let mut sub_atom_mapping: std::collections::HashMap<AtomId, AtomId> = std::collections::HashMap::new();

    // Register boundary inputs as sub-program inputs.
    for &input_atom in boundary_input_atoms {
        let input_type = program
            .atoms
            .get(input_atom.index)
            .ok_or(TracingError::UnboundAtomId { id: input_atom })?
            .r#type()
            .into_owned();
        let sub_atom = sub_builder.add_input(input_type);
        sub_atom_mapping.insert(input_atom, sub_atom);
    }

    // Register constants used by instructions in this segment.
    for instruction in segment_instructions {
        for &input_atom in &instruction.inputs {
            if sub_atom_mapping.contains_key(&input_atom) {
                continue;
            }
            let atom = program.atoms.get(input_atom.index).ok_or(TracingError::UnboundAtomId { id: input_atom })?;
            if let Atom::Constant(value) = atom {
                let sub_atom = sub_builder.add_constant(value.clone());
                sub_atom_mapping.insert(input_atom, sub_atom);
            }
        }
    }

    // Add instructions to the sub-program.
    for instruction in segment_instructions {
        let sub_inputs: Vec<AtomId> = instruction
            .inputs
            .iter()
            .map(|&orig_atom| {
                sub_atom_mapping.get(&orig_atom).copied().ok_or(TracingError::UnboundAtomId { id: orig_atom })
            })
            .collect::<Result<_, _>>()?;

        let output_abstracts: Vec<_> = instruction
            .outputs
            .iter()
            .map(|&atom_id| {
                program
                    .atoms
                    .get(atom_id.index)
                    .ok_or(TracingError::UnboundAtomId { id: atom_id })
                    .map(|atom| atom.r#type().into_owned())
            })
            .collect::<Result<_, _>>()?;
        let sub_outputs =
            output_abstracts.into_iter().map(|r#type| sub_builder.add_variable(r#type)).collect::<Vec<_>>();
        sub_builder.instructions.push(Instruction {
            operation: instruction.operation.clone(),
            inputs: sub_inputs,
            outputs: sub_outputs.clone(),
        });

        for (orig_atom, sub_atom) in instruction.outputs.iter().zip(sub_outputs.iter()) {
            sub_atom_mapping.insert(*orig_atom, *sub_atom);
        }
    }

    // Wire up boundary outputs.
    let sub_outputs: Vec<AtomId> = boundary_output_atoms
        .iter()
        .map(|&orig_atom| {
            sub_atom_mapping.get(&orig_atom).copied().ok_or(TracingError::UnboundAtomId { id: orig_atom })
        })
        .collect::<Result<_, _>>()?;

    let sub_program = sub_builder.build(
        sub_outputs,
        flat_leaf_parameter_structure(boundary_input_atoms.len()),
        flat_leaf_parameter_structure(boundary_output_atoms.len()),
    )?;
    Ok(sub_program)
}

#[cfg(test)]
mod tests {
    use crate::parameters::Placeholder;
    use crate::tracing::ProgramBuilder;
    use crate::tracing_v2::{PrimitiveOperation, engines::ArrayScalarEngine};

    use super::*;

    #[test]
    fn test_build_traced_gradient_program_handles_nullary_scalar_programs() {
        let engine = ArrayScalarEngine::<f64>::new();
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let output_atom = builder.add_constant(3.0f64);
        let traced_program = builder.build(vec![output_atom], Vec::<Placeholder>::new(), vec![Placeholder]).unwrap();

        let gradient_program = build_traced_gradient_program(&engine, (), &traced_program).unwrap();

        assert_eq!(gradient_program.interpret(()).unwrap(), ());
    }
}
