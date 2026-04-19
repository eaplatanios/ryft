use super::*;

fn replay_program_graph_with<GraphInput, GraphOutput, V, O, R, LiftConstant, ApplyOp>(
    graph: &Graph<O, ArrayType, V, GraphInput, GraphOutput>,
    inputs: Vec<R>,
    lift_constant: LiftConstant,
    apply_op: ApplyOp,
) -> Result<Vec<R>, TraceError>
where
    GraphInput: Parameterized<V>,
    GraphOutput: Parameterized<V>,
    V: Traceable<ArrayType>,
    O: Clone,
    R: Clone,
    LiftConstant: Fn(&V, &[R]) -> Result<R, TraceError>,
    ApplyOp: Fn(&O, Vec<R>) -> Result<Vec<R>, TraceError>,
{
    let mut values = vec![None; graph.atom_count()];
    for (atom_id, value) in graph.input_atoms().iter().copied().zip(inputs.iter().cloned()) {
        values[atom_id] = Some(value);
    }

    let mut equation_by_first_output = vec![None; graph.atom_count()];
    for (equation_index, equation) in graph.equations().iter().enumerate() {
        if let Some(first_output) = equation.outputs.first() {
            equation_by_first_output[*first_output] = Some(equation_index);
        }
    }

    for atom_id in 0..graph.atom_count() {
        let atom = graph.atom(atom_id).expect("atom IDs should be dense");
        match atom {
            Atom::Input { .. } => {}
            Atom::Constant { value } => {
                let seed_inputs = inputs.iter().cloned().chain(values.iter().flatten().cloned()).collect::<Vec<_>>();
                if seed_inputs.is_empty() {
                    return Err(TraceError::EmptyParameterizedValue);
                }
                values[atom_id] = Some(lift_constant(value, seed_inputs.as_slice())?);
            }
            Atom::Derived { .. } => {
                let Some(equation_index) = equation_by_first_output[atom_id] else {
                    continue;
                };
                let equation = &graph.equations()[equation_index];
                let input_values = equation
                    .inputs
                    .iter()
                    .map(|input| values[*input].clone().ok_or(TraceError::UnboundAtomId { id: *input }))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = apply_op(&equation.op, input_values)?;
                for (output_atom, output_value) in equation.outputs.iter().copied().zip(outputs) {
                    values[output_atom] = Some(output_value);
                }
            }
        }
    }

    graph
        .outputs()
        .iter()
        .map(|output| values[*output].clone().ok_or(TraceError::UnboundAtomId { id: *output }))
        .collect()
}

pub(crate) fn replay_program_graph_linearized_jit<GraphInput, GraphOutput, V, O, L>(
    graph: &Graph<O, ArrayType, V, GraphInput, GraphOutput>,
    inputs: Vec<LinearizedTracedValue<V, O, L>>,
) -> Result<Vec<LinearizedTracedValue<V, O, L>>, TraceError>
where
    GraphInput: Parameterized<V>,
    GraphOutput: Parameterized<V>,
    V: Traceable<ArrayType> + ZeroLike,
    L: Clone + 'static,
    O: InterpretableOp<ArrayType, LinearizedTracedValue<V, O, L>> + Clone,
{
    replay_program_graph_with(
        graph,
        inputs,
        super::program::lift_linearized_traced_constant::<V, O, L>,
        |op, values| InterpretableOp::<ArrayType, LinearizedTracedValue<V, O, L>>::interpret(op, &values),
    )
}

pub(crate) fn linearize_traced_program<V, O, L>(
    program: &Program<ArrayType, V, Vec<V>, Vec<V>, O>,
    primals: Vec<JitTracer<ArrayType, V, O, L>>,
) -> Result<(Vec<JitTracer<ArrayType, V, O, L>>, TracedLinearProgram<V, O, L>), TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + 'static,
    O: InterpretableOp<ArrayType, LinearizedTracedValue<V, O, L>> + Clone,
{
    let zero = primals.first().map(ZeroLike::zero_like).ok_or(TraceError::EmptyParameterizedValue)?;
    let input_count = primals.len();
    let builder = Rc::new(RefCell::new(LinearProgramBuilder::<
        JitTracer<ArrayType, V, O, L>,
        LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>,
    >::new()));
    let traced_input = primals
        .into_iter()
        .map(|primal| {
            let atom = builder.borrow_mut().add_input(&primal);
            Linearized { primal, tangent: LinearTerm::from_staged_parts(atom, builder.clone()) }
        })
        .collect::<Vec<_>>();
    let traced_output = replay_program_graph_linearized_jit::<_, _, _, O, L>(program.graph(), traced_input)?;
    let primal_outputs = traced_output.iter().map(|output| output.primal.clone()).collect::<Vec<_>>();
    let tangent_outputs = traced_output.iter().map(|output| output.tangent.atom()).collect::<Vec<_>>();
    drop(traced_output);
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TraceError::InternalInvariantViolation("linearization builder escaped the tracing scope"));
        }
    };
    let program =
        Program::from_graph(builder.build::<Vec<JitTracer<ArrayType, V, O, L>>, Vec<JitTracer<ArrayType, V, O, L>>>(
            tangent_outputs,
            vec![Placeholder; input_count],
            vec![Placeholder; primal_outputs.len()],
        ))
        .simplify()?;
    Ok((primal_outputs.clone(), LinearProgram::from_program(program, zero)))
}
