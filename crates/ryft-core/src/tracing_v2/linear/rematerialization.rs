use super::*;

/// Compiles a reverse-mode gradient function into a reusable staged program.
///
/// Unlike [`grad`](super::grad), which returns concrete gradient values at a single primal point,
/// this function returns a [`Program`] that takes primal inputs and produces gradient
/// outputs symbolically. The compiled program embeds both the forward residual computation and the
/// backward pass, so it can be replayed at arbitrary primal points without re-tracing.
///
/// This is analogous to JAX's `jit(grad(f))`.
#[allow(private_bounds)]
pub fn compile_grad<E, F, Input, V>(
    _engine: &E,
    function: F,
    example_primals: Input,
) -> Result<Program<ArrayType, V, Input, Input, E::TracingOperation>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: Value<ArrayType> + ZeroLike + OneLike,
    E::TracingOperation: InterpretableOp<ArrayType, V>
        + InterpretableOp<ArrayType, LinearizedTracedValue<V, E::TracingOperation, E::LinearOperation, E>>
        + Op<ArrayType>,
    LinearProgramOpRef<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>:
        CoreLinearProgramOp<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>,
    Vec<V>: Parameterized<V, ParameterStructure = Vec<Placeholder>>,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>,
    Input::To<ArrayType>: Parameterized<
            ArrayType,
            To<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>> = Input::To<
                Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>,
            >,
        >,
    V::To<ArrayType>: Parameterized<
            ArrayType,
            To<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>> = Tracer<
                ArrayType,
                V,
                E::TracingOperation,
                E::LinearOperation,
                E,
            >,
        >,
    F: Fn(
        Input::To<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>,
    ) -> Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>,
{
    let input_structure = example_primals.parameter_structure();
    let (_, compiled) = interpret_and_trace(
        _engine,
        |primals: Input::To<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>| {
            let traced_primals = primals.into_parameters().collect::<Vec<_>>();
            let staged_input_types = Input::To::<ArrayType>::from_parameters(
                input_structure.clone(),
                traced_primals.iter().map(|traced_primal| traced_primal.tpe().into_owned()).collect::<Vec<_>>(),
            )?;
            let (_, traced_program) =
                trace_flat_program_from_input_types::<
                    Input::To<ArrayType>,
                    V::To<ArrayType>,
                    V,
                    E::TracingOperation,
                    E::LinearOperation,
                    E,
                    _,
                >(|staged_input| Ok(function(staged_input)), &traced_primals, staged_input_types)?;
            let (_, traced_gradient) =
                reverse_mode_scalar_traced_program::<V, E::TracingOperation, E::LinearOperation, E>(
                    &traced_program,
                    traced_primals,
                )?;
            Input::To::<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>::from_parameters(
                input_structure.clone(),
                traced_gradient,
            )
            .map_err(TraceError::from)
        },
        example_primals,
    )?;
    Ok(compiled)
}

/// Policy controlling how forward-pass intermediates are handled during reverse-mode differentiation.
///
/// This trades off memory usage against recomputation cost. [`SaveAll`](RematerializationPolicy::SaveAll) is the
/// default: all intermediates are saved, giving fast backward passes at the cost of high memory.
/// [`RecomputeAll`](RematerializationPolicy::RecomputeAll) saves nothing, recomputing everything from inputs.
/// [`Checkpoint`](RematerializationPolicy::Checkpoint) is the classic middle ground, saving intermediates at
/// regular intervals and recomputing within each segment.
#[derive(Clone, Debug)]
pub enum RematerializationPolicy {
    /// Save all forward-pass intermediates (maximum memory, no recomputation).
    SaveAll,

    /// Recompute all forward-pass intermediates from inputs (minimum memory, maximum recomputation).
    RecomputeAll,

    /// Save intermediates every `segment_size` equations, recomputing within each segment.
    ///
    /// With a program of N equations, setting `segment_size` to approximately the square root of N gives O(sqrt(N))
    /// memory usage. A `segment_size` of zero or one degenerates to [`SaveAll`](RematerializationPolicy::SaveAll)
    /// since each segment contains at most one equation.
    Checkpoint {
        /// Number of equations per rematerialization segment.
        segment_size: usize,
    },
}

/// Compiles a reverse-mode gradient function with an explicit rematerialization policy.
///
/// This generalizes [`compile_grad`] by letting the caller control how forward-pass intermediates are handled
/// during the backward pass:
///
///   - [`RematerializationPolicy::SaveAll`]: identical to [`compile_grad`] â€” no rematerialization boundaries are
///     inserted, so the XLA compiler decides which intermediates to save.
///   - [`RematerializationPolicy::RecomputeAll`]: the entire forward body is wrapped in a single
///     [`rematerialize`] boundary, forcing the backward pass to recompute all intermediates from inputs.
///   - [`RematerializationPolicy::Checkpoint`]: the forward body is partitioned into segments of at most
///     `segment_size` equations, each wrapped in its own [`rematerialize`] boundary. Intermediates at segment
///     boundaries are saved while within-segment intermediates are recomputed.
#[allow(private_bounds)]
pub fn compile_grad_with_policy<E, F, Input, V>(
    engine: &E,
    function: F,
    example_primals: Input,
    policy: RematerializationPolicy,
) -> Result<Program<ArrayType, V, Input, Input, E::TracingOperation>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: Value<ArrayType> + ZeroLike + OneLike,
    E::TracingOperation: InterpretableOp<ArrayType, V>
        + InterpretableOp<ArrayType, LinearizedTracedValue<V, E::TracingOperation, E::LinearOperation, E>>
        + RematerializeTracingOperation<ArrayType, V, E::LinearOperation>
        + Op<ArrayType>,
    LinearProgramOpRef<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>:
        CoreLinearProgramOp<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>,
    Vec<V>: Parameterized<V, ParameterStructure = Vec<Placeholder>>,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>,
    Input::To<ArrayType>: Parameterized<
            ArrayType,
            To<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>> = Input::To<
                Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>,
            >,
        >,
    V::To<ArrayType>: Parameterized<
            ArrayType,
            To<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>> = Tracer<
                ArrayType,
                V,
                E::TracingOperation,
                E::LinearOperation,
                E,
            >,
        >,
    F: Fn(
        Input::To<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>,
    ) -> Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>,
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
/// When `segment_size` is `None`, the entire program is wrapped in a single [`RematerializeOp`]
/// (equivalent to [`RematerializationPolicy::RecomputeAll`]). When `Some(s)`, the program is
/// partitioned into segments of at most `s` equations, each wrapped in its own [`RematerializeOp`].
///
/// Internally, this replicates the flow of `grad` for [`Tracer`]-level inputs â€” trace, linearize,
/// transpose, stage pullback â€” but inserts a segmentation step between tracing and linearization so
/// that the differentiation transform sees and respects the rematerialization boundaries.
fn compile_grad_segmented<E, F, Input, V>(
    engine: &E,
    function: &F,
    example_primals: Input,
    segment_size: Option<usize>,
) -> Result<Program<ArrayType, V, Input, Input, E::TracingOperation>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: Value<ArrayType> + ZeroLike + OneLike,
    E::TracingOperation: InterpretableOp<ArrayType, V>
        + InterpretableOp<ArrayType, LinearizedTracedValue<V, E::TracingOperation, E::LinearOperation, E>>
        + RematerializeTracingOperation<ArrayType, V, E::LinearOperation>
        + Op<ArrayType>,
    LinearProgramOpRef<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>:
        CoreLinearProgramOp<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>,
    Vec<V>: Parameterized<V, ParameterStructure = Vec<Placeholder>>,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>,
    Input::To<ArrayType>: Parameterized<
            ArrayType,
            To<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>> = Input::To<
                Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>,
            >,
        >,
    V::To<ArrayType>: Parameterized<
            ArrayType,
            To<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>> = Tracer<
                ArrayType,
                V,
                E::TracingOperation,
                E::LinearOperation,
                E,
            >,
        >,
    F: Fn(
        Input::To<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>,
    ) -> Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>,
{
    let input_structure = example_primals.parameter_structure();
    let (_, compiled) = interpret_and_trace(
        engine,
        |primals: Input::To<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>| {
            let traced_primals = primals.into_parameters().collect::<Vec<_>>();

            // Step 1: Trace the function at the base V level to get a program.
            let staged_input_types = Input::To::<ArrayType>::from_parameters(
                input_structure.clone(),
                traced_primals.iter().map(|traced_primal| traced_primal.tpe().into_owned()).collect::<Vec<_>>(),
            )?;
            let (_, traced_program) =
                trace_flat_program_from_input_types::<
                    Input::To<ArrayType>,
                    V::To<ArrayType>,
                    V,
                    E::TracingOperation,
                    E::LinearOperation,
                    E,
                    _,
                >(|staged_input| Ok(function(staged_input)), &traced_primals, staged_input_types)?;

            // Step 2: Segment the traced program to insert rematerialization boundaries.
            let segmented_program = match segment_size {
                None => wrap_program_in_rematerialize(engine, &traced_program)?,
                Some(size) => segment_program(engine, &traced_program, size)?,
            };

            // Step 3: Linearize and transpose the segmented program to produce the pullback.
            // `linearize_traced_program` replays the program at the Tracer level (staging
            // both forward and backward equations in the outer JIT builder) and returns the
            // primal outputs alongside the linear pushforward map.
            let (_, traced_gradient) =
                reverse_mode_scalar_traced_program::<V, E::TracingOperation, E::LinearOperation, E>(
                    &segmented_program,
                    traced_primals,
                )?;
            Input::To::<Tracer<ArrayType, V, E::TracingOperation, E::LinearOperation, E>>::from_parameters(
                input_structure.clone(),
                traced_gradient,
            )
            .map_err(TraceError::from)
        },
        example_primals,
    )?;
    Ok(compiled)
}

/// Partitions a program's equations into segments of at most `segment_size`, wrapping each segment in a
/// [`RematerializeOp`].
///
/// Given a program with N equations and a segment size S, this produces a new program with at most
/// `ceil(N / S)` equations. Each equation is a [`RematerializeOp`] whose body sub-program contains the
/// original equations from that segment. Atoms crossing segment boundaries become inputs/outputs of the
/// respective sub-programs.
///
/// The segmented program is semantically equivalent to the original: calling it on the same inputs produces
/// the same outputs. The difference is visible only during differentiation, where each [`RematerializeOp`]
/// boundary forces recomputation of within-segment intermediates rather than saving them.
fn segment_program<E, V>(
    engine: &E,
    program: &Program<ArrayType, V, Vec<V>, Vec<V>, E::TracingOperation>,
    segment_size: usize,
) -> Result<Program<ArrayType, V, Vec<V>, Vec<V>, E::TracingOperation>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType>,
    E::TracingOperation:
        InterpretableOp<ArrayType, V> + RematerializeTracingOperation<ArrayType, V, E::LinearOperation> + Op<ArrayType>,
{
    let program = program;
    let representative_values = program.representative_atom_values(engine)?;
    let representative_inputs = program.representative_input_values(engine)?;
    let equations = program.equations();

    // If the program has fewer equations than a single segment, no segmentation is needed â€” wrap the
    // whole thing in a single RematerializeOp.
    if equations.len() <= segment_size {
        return wrap_program_in_rematerialize(engine, program);
    }

    // Divide equations into segments.
    let segments: Vec<&[Equation<E::TracingOperation>]> = equations.chunks(segment_size).collect();

    // Build a mapping from atom ID to which equation produces it (if any).
    let mut atom_producer: Vec<Option<usize>> = vec![None; program.atom_count()];
    for (equation_index, equation) in equations.iter().enumerate() {
        for &output_atom in &equation.outputs {
            atom_producer[output_atom] = Some(equation_index);
        }
    }

    // Build a set tracking which atoms are consumed after a given equation index.
    // For each atom, track all equation indices that consume it.
    let mut atom_consumers: Vec<Vec<usize>> = vec![Vec::new(); program.atom_count()];
    for (equation_index, equation) in equations.iter().enumerate() {
        for &input_atom in &equation.inputs {
            atom_consumers[input_atom].push(equation_index);
        }
    }
    // Also mark program outputs as "consumed" at equation_count (sentinel for "after all equations").
    let sentinel = equations.len();
    for &output_atom in program.outputs() {
        atom_consumers[output_atom].push(sentinel);
    }

    // Build the outer program.
    let input_atoms = program.input_atoms();
    let mut outer_builder: ProgramBuilder<E::TracingOperation, ArrayType, V> = ProgramBuilder::new();

    // Map from original atom IDs to outer-program atom IDs.
    let mut atom_mapping: Vec<Option<AtomId>> = vec![None; program.atom_count()];

    // Register program inputs in the outer builder.
    for (&input_atom, representative_input) in input_atoms.iter().zip(representative_inputs.iter()) {
        let outer_atom = outer_builder.add_input(representative_input);
        atom_mapping[input_atom] = Some(outer_atom);
    }

    // Register constants that are used by equations (they might be referenced across segments).
    for (atom_id, atom) in program.atoms_iter() {
        if let Atom::Constant { value } = atom {
            let outer_atom = outer_builder.add_constant(value.clone());
            atom_mapping[atom_id] = Some(outer_atom);
        }
    }

    // Process each segment.
    let mut equation_offset = 0;
    for segment in &segments {
        let segment_start = equation_offset;
        let segment_end = equation_offset + segment.len();

        // Identify boundary inputs: atoms consumed by this segment that are produced outside it
        // (by previous segments or program inputs/constants).
        let mut boundary_input_atoms: Vec<AtomId> = Vec::new();
        let mut boundary_input_set = std::collections::HashSet::new();
        for equation in *segment {
            for &input_atom in &equation.inputs {
                // If this atom is produced by an equation outside this segment (or is an input/constant).
                let produced_in_segment = atom_producer[input_atom]
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
        for equation in *segment {
            for &output_atom in &equation.outputs {
                let consumed_outside = atom_consumers[output_atom]
                    .iter()
                    .any(|&consumer_idx| consumer_idx < segment_start || consumer_idx >= segment_end);
                if consumed_outside && boundary_output_set.insert(output_atom) {
                    boundary_output_atoms.push(output_atom);
                }
            }
        }

        // Build the sub-program for this segment.
        let sub_program = build_segment_sub_program(
            program,
            representative_values.as_slice(),
            *segment,
            &boundary_input_atoms,
            &boundary_output_atoms,
        )?;

        // Build the RematerializeOp.
        let input_types: Vec<_> = boundary_input_atoms
            .iter()
            .map(|&atom_id| {
                program
                    .atom(atom_id)
                    .ok_or(TraceError::UnboundAtomId { id: atom_id })
                    .map(|atom| atom.tpe().into_owned())
            })
            .collect::<Result<_, _>>()?;
        let output_types: Vec<_> = boundary_output_atoms
            .iter()
            .map(|&atom_id| {
                program
                    .atom(atom_id)
                    .ok_or(TraceError::UnboundAtomId { id: atom_id })
                    .map(|atom| atom.tpe().into_owned())
            })
            .collect::<Result<_, _>>()?;

        let body = FlatTracedRematerialize::from_parts(input_types.clone(), output_types.clone(), sub_program);
        let remat_op = RematerializeOp::new(body);

        // Add the RematerializeOp equation to the outer builder.
        let outer_inputs: Vec<AtomId> = boundary_input_atoms
            .iter()
            .map(|&orig_atom| atom_mapping[orig_atom].ok_or(TraceError::UnboundAtomId { id: orig_atom }))
            .collect::<Result<_, _>>()?;
        let outer_outputs = outer_builder.add_equation_prevalidated(
            E::TracingOperation::rematerialize_op(remat_op),
            outer_inputs,
            output_types,
        );

        // Map the boundary output atoms to their outer-program counterparts.
        for (orig_atom, outer_atom) in boundary_output_atoms.iter().zip(outer_outputs.iter()) {
            atom_mapping[*orig_atom] = Some(*outer_atom);
        }

        equation_offset = segment_end;
    }

    // Wire up the program outputs.
    let outer_outputs: Vec<AtomId> = program
        .outputs()
        .iter()
        .map(|&orig_atom| atom_mapping[orig_atom].ok_or(TraceError::UnboundAtomId { id: orig_atom }))
        .collect::<Result<_, _>>()?;

    let outer_program = outer_builder.build::<Vec<V>, Vec<V>>(
        outer_outputs,
        flat_leaf_parameter_structure(input_atoms.len()),
        flat_leaf_parameter_structure(program.outputs().len()),
    );
    Ok(outer_program)
}

/// Wraps an entire program in a single [`RematerializeOp`] boundary.
fn wrap_program_in_rematerialize<E, V>(
    engine: &E,
    program: &Program<ArrayType, V, Vec<V>, Vec<V>, E::TracingOperation>,
) -> Result<Program<ArrayType, V, Vec<V>, Vec<V>, E::TracingOperation>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType>,
    E::TracingOperation: RematerializeTracingOperation<ArrayType, V, E::LinearOperation>,
{
    let program = program;
    let representative_inputs = program.representative_input_values(engine)?;
    let input_types: Vec<_> = program
        .input_atoms()
        .iter()
        .map(|&atom_id| {
            program
                .atom(atom_id)
                .ok_or(TraceError::UnboundAtomId { id: atom_id })
                .map(|atom| atom.tpe().into_owned())
        })
        .collect::<Result<_, _>>()?;
    let output_types: Vec<_> = program
        .outputs()
        .iter()
        .map(|&atom_id| {
            program
                .atom(atom_id)
                .ok_or(TraceError::UnboundAtomId { id: atom_id })
                .map(|atom| atom.tpe().into_owned())
        })
        .collect::<Result<_, _>>()?;

    let body = FlatTracedRematerialize::from_parts(input_types.clone(), output_types.clone(), program.clone());
    let remat_op = RematerializeOp::new(body);

    let mut outer_builder: ProgramBuilder<E::TracingOperation, ArrayType, V> = ProgramBuilder::new();
    let outer_inputs: Vec<AtomId> = representative_inputs
        .iter()
        .map(|representative_input| outer_builder.add_input(representative_input))
        .collect();

    let outer_outputs = outer_builder.add_equation_prevalidated(
        E::TracingOperation::rematerialize_op(remat_op),
        outer_inputs.clone(),
        output_types,
    );

    let outer_program = outer_builder.build::<Vec<V>, Vec<V>>(
        outer_outputs,
        flat_leaf_parameter_structure(outer_inputs.len()),
        flat_leaf_parameter_structure(program.outputs().len()),
    );
    Ok(outer_program)
}

/// Builds a sub-program for a single segment of equations.
///
/// The sub-program takes the boundary input atoms as its inputs and produces the boundary output atoms as its
/// outputs. Internal atoms (produced and consumed entirely within the segment) are handled as internal constants
/// and equations within the sub-program.
fn build_segment_sub_program<V: Traceable<ArrayType>, O: Clone>(
    program: &Program<ArrayType, V, Vec<V>, Vec<V>, O>,
    representative_values: &[V],
    segment_equations: &[Equation<O>],
    boundary_input_atoms: &[AtomId],
    boundary_output_atoms: &[AtomId],
) -> Result<Program<ArrayType, V, Vec<V>, Vec<V>, O>, TraceError> {
    let mut sub_builder: ProgramBuilder<O, ArrayType, V> = ProgramBuilder::new();

    // Map from original atom IDs to sub-program atom IDs.
    let mut sub_atom_mapping: std::collections::HashMap<AtomId, AtomId> = std::collections::HashMap::new();

    // Register boundary inputs as sub-program inputs.
    for &input_atom in boundary_input_atoms {
        let sub_atom = sub_builder.add_input(&representative_values[input_atom]);
        sub_atom_mapping.insert(input_atom, sub_atom);
    }

    // Register constants used by equations in this segment.
    for equation in segment_equations {
        for &input_atom in &equation.inputs {
            if sub_atom_mapping.contains_key(&input_atom) {
                continue;
            }
            let atom = program.atom(input_atom).ok_or(TraceError::UnboundAtomId { id: input_atom })?;
            if let Atom::Constant { value } = atom {
                let sub_atom = sub_builder.add_constant(value.clone());
                sub_atom_mapping.insert(input_atom, sub_atom);
            }
        }
    }

    // Add equations to the sub-program.
    for equation in segment_equations {
        let sub_inputs: Vec<AtomId> = equation
            .inputs
            .iter()
            .map(|&orig_atom| {
                sub_atom_mapping.get(&orig_atom).copied().ok_or(TraceError::UnboundAtomId { id: orig_atom })
            })
            .collect::<Result<_, _>>()?;

        let output_abstracts: Vec<_> = equation
            .outputs
            .iter()
            .map(|&atom_id| {
                program
                    .atom(atom_id)
                    .ok_or(TraceError::UnboundAtomId { id: atom_id })
                    .map(|atom| atom.tpe().into_owned())
            })
            .collect::<Result<_, _>>()?;
        let sub_outputs = sub_builder.add_equation_prevalidated(equation.op.clone(), sub_inputs, output_abstracts);

        for (orig_atom, sub_atom) in equation.outputs.iter().zip(sub_outputs.iter()) {
            sub_atom_mapping.insert(*orig_atom, *sub_atom);
        }
    }

    // Wire up boundary outputs.
    let sub_outputs: Vec<AtomId> = boundary_output_atoms
        .iter()
        .map(|&orig_atom| sub_atom_mapping.get(&orig_atom).copied().ok_or(TraceError::UnboundAtomId { id: orig_atom }))
        .collect::<Result<_, _>>()?;

    let sub_program = sub_builder.build::<Vec<V>, Vec<V>>(
        sub_outputs,
        flat_leaf_parameter_structure(boundary_input_atoms.len()),
        flat_leaf_parameter_structure(boundary_output_atoms.len()),
    );
    Ok(sub_program)
}
