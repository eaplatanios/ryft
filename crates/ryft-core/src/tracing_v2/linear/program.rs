use super::*;

/// Staged linear map produced by [`jvp_program`](super::jvp_program) or [`vjp`](super::vjp).
pub struct LinearProgram<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: Clone = LinearProgramOpRef<V>,
> {
    program: Program<T, V, Input, Output, O>,
    zero: V,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<
    T: Type + Display,
    V: Clone + Traceable<T>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: Clone,
> Clone for LinearProgram<T, V, Input, Output, O>
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), zero: self.zero.clone(), marker: PhantomData }
    }
}

impl<T: Type + Display, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>, O: Clone>
    LinearProgram<T, V, Input, Output, O>
{
    #[inline]
    pub fn from_program(program: Program<T, V, Input, Output, O>, zero: V) -> Self {
        Self { program, zero, marker: PhantomData }
    }

    /// Returns the staged graph backing this linear program.
    #[inline]
    pub fn program(&self) -> &Program<T, V, Input, Output, O> {
        &self.program
    }

    /// Applies the linear program to a concrete input tangent or cotangent.
    pub fn call(&self, input: Input) -> Result<Output, TraceError>
    where
        O: InterpretableOp<T, V>,
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.program.call(input)
    }
}

impl<V: Traceable<ArrayType>, Input: Parameterized<V>, Output: Parameterized<V>, O: Clone>
    LinearProgram<ArrayType, V, Input, Output, O>
{
    /// Transposes the linear program, turning a pushforward into a pullback.
    #[allow(private_bounds)]
    pub fn transpose(&self) -> Result<LinearProgram<ArrayType, V, Output, Input, O>, TraceError>
    where
        V: ZeroLike,
        O: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V> + Clone,
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        transpose_linear_program(self)
    }
}

impl<V: Traceable<ArrayType>, Input: Parameterized<V>, Output: Parameterized<V>> Display
    for LinearProgram<ArrayType, V, Input, Output>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.program, formatter)
    }
}

/// Applies one primitive's semantic transpose rule while transposing a staged linear program.
///
/// The transpose builder's cotangent atoms are wrapped as [`LinearTerm<ArrayType, V>`] so primitive rules can
/// emit staged cotangent contributions directly.
///
/// The returned atom ids are staged cotangent contributions in the transpose builder, aligned with
/// the forward primitive inputs.
///
/// # Parameters
///   - `op`: primitive whose transpose rule should be applied.
///   - `builder`: transpose-program builder that owns the staged cotangent atoms created while
///     constructing the pullback program.
///   - `output_cotangents`: transpose-builder atom ids for the already-staged cotangents of
///     the primitive outputs.
fn transpose<V, O>(
    op: &O,
    builder: &Rc<RefCell<LinearProgramBuilder<V, O>>>,
    output_cotangents: &[AtomId],
) -> Result<Vec<Option<AtomId>>, TraceError>
where
    V: Traceable<ArrayType>,
    O: CoreLinearProgramOp<V> + Clone,
{
    let cotangent_terms = output_cotangents
        .iter()
        .map(|cotangent| LinearTerm::from_staged_parts(*cotangent, builder.clone()))
        .collect::<Vec<_>>();
    Ok(op
        .transpose(cotangent_terms.as_slice())?
        .into_iter()
        .map(|term| term.map(|term| term.atom()))
        .collect())
}

pub(crate) fn linearize_program<Input, Output, V, O, L>(
    engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
    program: &Program<ArrayType, V, Input, Output, O>,
    input_primals: Vec<V>,
) -> Result<LinearProgram<ArrayType, V, Input, Output, L>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    L: Clone + Op<ArrayType>,
    O: Clone + DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V, L>, O, L>,
{
    fn tangent_for_atom<V, Input, Output, GraphOperation, LinearOperation>(
        _graph: &Graph<GraphOperation, ArrayType, V, Input, Output>,
        primal_values: &[Option<V>],
        builder: &Rc<RefCell<LinearProgramBuilder<V, LinearOperation>>>,
        tangents: &mut [Option<LinearTerm<ArrayType, V, LinearOperation>>],
        atom_id: AtomId,
    ) -> Result<LinearTerm<ArrayType, V, LinearOperation>, TraceError>
    where
        V: Traceable<ArrayType> + ZeroLike,
        Input: Parameterized<V>,
        Output: Parameterized<V>,
        GraphOperation: Clone + Op<ArrayType>,
        LinearOperation: Clone + Op<ArrayType>,
    {
        if let Some(term) = tangents[atom_id].clone() {
            return Ok(term);
        }
        let primal = primal_values[atom_id].as_ref().ok_or(TraceError::UnboundAtomId { id: atom_id })?;
        let tangent_atom = builder.borrow_mut().add_constant(primal.zero_like());
        let tangent = LinearTerm::from_staged_parts(tangent_atom, builder.clone());
        tangents[atom_id] = Some(tangent.clone());
        Ok(tangent)
    }

    let graph = program.graph();
    if input_primals.len() != graph.input_atoms().len() {
        return Err(TraceError::InvalidInputCount { expected: graph.input_atoms().len(), got: input_primals.len() });
    }
    let zero = input_primals.first().map(ZeroLike::zero_like).ok_or(TraceError::EmptyParameterizedValue)?;
    let builder = Rc::new(RefCell::new(LinearProgramBuilder::<V, L>::new()));
    let mut primals: Vec<Option<V>> = vec![None; graph.atom_count()];
    let mut tangents: Vec<Option<LinearTerm<ArrayType, V, L>>> = vec![None; graph.atom_count()];
    for (input_atom, input_primal) in graph.input_atoms().iter().copied().zip(input_primals.into_iter()) {
        let tangent_atom = builder.borrow_mut().add_input(&input_primal.zero_like());
        tangents[input_atom] = Some(LinearTerm::from_staged_parts(tangent_atom, builder.clone()));
        primals[input_atom] = Some(input_primal);
    }
    for (atom_id, atom) in graph.atoms_iter() {
        if let Atom::Constant { value } = atom {
            primals[atom_id] = Some(value.clone());
        }
    }

    for equation in graph.equations() {
        let input_duals = equation
            .inputs
            .iter()
            .copied()
            .map(|input_atom| {
                Ok(JvpTracer {
                    primal: primals[input_atom].clone().ok_or(TraceError::UnboundAtomId { id: input_atom })?,
                    tangent: tangent_for_atom(
                        graph,
                        primals.as_slice(),
                        &builder,
                        tangents.as_mut_slice(),
                        input_atom,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, TraceError>>()?;
        let output_duals = DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V, L>, O, L>::jvp(
            &equation.op,
            engine,
            input_duals.as_slice(),
        )?;
        if output_duals.len() != equation.outputs.len() {
            return Err(TraceError::InvalidOutputCount { expected: equation.outputs.len(), got: output_duals.len() });
        }
        for (output_atom, output_dual) in equation.outputs.iter().copied().zip(output_duals.into_iter()) {
            primals[output_atom] = Some(output_dual.primal);
            tangents[output_atom] = Some(output_dual.tangent);
        }
    }

    let output_tangents = graph
        .outputs()
        .iter()
        .copied()
        .map(|output_atom| {
            tangent_for_atom(graph, primals.as_slice(), &builder, tangents.as_mut_slice(), output_atom)
                .map(|term| term.atom())
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(tangents);
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TraceError::InternalInvariantViolation("linearization builder escaped the tracing scope"));
        }
    };
    Ok(LinearProgram {
        program: Program::from_graph(builder.build::<Input, Output>(
            output_tangents,
            graph.input_structure().clone(),
            graph.output_structure().clone(),
        ))
        .simplify()?,
        zero,
        marker: PhantomData,
    })
}

#[allow(private_bounds)]
pub fn transpose_linear_program<V, Input, Output, O>(
    program: &LinearProgram<ArrayType, V, Input, Output, O>,
) -> Result<LinearProgram<ArrayType, V, Output, Input, O>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V> + Clone,
{
    let zero = program.zero.zero_like();
    transpose_linear_program_with_output_inputs(program, |builder: &mut LinearProgramBuilder<V, O>, _, _| {
        Ok(builder.add_input(&zero))
    })
}

fn transpose_linear_program_with_output_inputs<V, Input, Output, O, F>(
    program: &LinearProgram<ArrayType, V, Input, Output, O>,
    mut make_output_cotangent_input: F,
) -> Result<LinearProgram<ArrayType, V, Output, Input, O>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    F: FnMut(&mut LinearProgramBuilder<V, O>, &ArrayType, usize) -> Result<AtomId, TraceError>,
    O: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V> + Clone,
{
    fn accumulate<V, O>(
        builder: &Rc<RefCell<LinearProgramBuilder<V, O>>>,
        adjoints: &mut [Option<AtomId>],
        atom: AtomId,
        contribution: AtomId,
    ) -> Result<(), TraceError>
    where
        V: Traceable<ArrayType>,
        O: LinearAddOperation<ArrayType, V> + Op<ArrayType> + Clone,
    {
        adjoints[atom] = Some(match adjoints[atom] {
            Some(existing) => {
                let mut builder_borrow = builder.borrow_mut();
                let abstract_value =
                    builder_borrow.atom(existing).expect("adjoint atom should exist").tpe().into_owned();
                builder_borrow.add_equation_prevalidated(
                    O::linear_add_op(),
                    vec![existing, contribution],
                    vec![abstract_value],
                )[0]
            }
            None => contribution,
        });
        Ok(())
    }

    let graph = program.program.graph();
    let builder = Rc::new(RefCell::new(LinearProgramBuilder::<V, O>::new()));
    let mut output_cotangent_inputs = Vec::with_capacity(graph.outputs().len());
    for (output_index, output) in graph.outputs().iter().enumerate() {
        let output_atom = graph.atom(*output).ok_or(TraceError::UnboundAtomId { id: *output })?;
        let cotangent_input = make_output_cotangent_input(&mut builder.borrow_mut(), &output_atom.tpe(), output_index)?;
        output_cotangent_inputs.push(cotangent_input);
    }

    let mut adjoints = vec![None; graph.atom_count()];
    for (cotangent, output) in output_cotangent_inputs.into_iter().zip(graph.outputs().iter().copied()) {
        accumulate(&builder, adjoints.as_mut_slice(), output, cotangent)?;
    }

    for equation in graph.equations().iter().rev() {
        let equation_output_cotangents =
            equation.outputs.iter().map(|output| adjoints[*output]).collect::<Option<Vec<_>>>();
        let Some(equation_output_cotangents) = equation_output_cotangents else {
            continue;
        };
        let input_cotangents = transpose(&equation.op, &builder, equation_output_cotangents.as_slice())?;
        for (input, contribution) in equation.inputs.iter().copied().zip(input_cotangents) {
            if let Some(contribution) = contribution {
                accumulate(&builder, adjoints.as_mut_slice(), input, contribution)?;
            }
        }
    }

    let zero_atom = builder.borrow_mut().add_constant(program.zero.clone());
    let outputs = graph
        .input_atoms()
        .iter()
        .copied()
        .map(|input| adjoints[input].unwrap_or(zero_atom))
        .collect::<Vec<_>>();
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TraceError::InternalInvariantViolation(
                "transpose builder should not have outstanding linear terms",
            ));
        }
    };
    Ok(LinearProgram {
        program: Program::from_graph(builder.build::<Output, Input>(
            outputs,
            graph.output_structure().clone(),
            graph.input_structure().clone(),
        ))
        .simplify()?,
        zero: program.zero.clone(),
        marker: PhantomData,
    })
}

/// Transposes a linear program using concrete output examples to seed the cotangent inputs.
///
/// This variant is useful when the linear program's leaf type cannot be synthesized from bare
/// [`ArrayType`] metadata alone, but the caller still has representative output values available.
#[allow(private_bounds)]
pub fn transpose_linear_program_with_output_examples<V, Input, Output, O>(
    program: &LinearProgram<ArrayType, V, Input, Output, O>,
    output_examples: &[V],
) -> Result<LinearProgram<ArrayType, V, Output, Input, O>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V> + Clone,
{
    let expected_output_count = program.program().graph().outputs().len();
    if output_examples.len() != expected_output_count {
        return Err(TraceError::InvalidInputCount { expected: expected_output_count, got: output_examples.len() });
    }
    transpose_linear_program_with_output_inputs(program, |builder: &mut LinearProgramBuilder<V, O>, _, output_index| {
        Ok(builder.add_input(&output_examples[output_index].zero_like()))
    })
}

fn lift_traced_constant<V, O: Clone, L: Clone>(
    constant: &V,
    inputs: &[JitTracer<ArrayType, V, O, L>],
) -> Result<JitTracer<ArrayType, V, O, L>, TraceError>
where
    V: Traceable<ArrayType>,
{
    let exemplar = inputs.first().ok_or(TraceError::EmptyParameterizedValue)?;
    let atom = exemplar.builder_handle().borrow_mut().add_constant(constant.clone());
    Ok(JitTracer::from_staged_parts(
        atom,
        exemplar.builder_handle(),
        exemplar.staging_error_handle(),
        exemplar.engine(),
    ))
}

pub(crate) fn lift_linearized_traced_constant<V, O: Clone + 'static, L: Clone + 'static>(
    constant: &V,
    inputs: &[LinearizedTracedValue<V, O, L>],
) -> Result<LinearizedTracedValue<V, O, L>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
{
    let exemplar = inputs.first().ok_or(TraceError::EmptyParameterizedValue)?;
    let primal = lift_traced_constant(constant, std::slice::from_ref(&exemplar.primal))?;
    let tangent_atom = exemplar.tangent.builder_handle().borrow_mut().add_constant(primal.zero_like());
    let tangent = LinearTerm::from_staged_parts(tangent_atom, exemplar.tangent.builder_handle());
    Ok(Linearized { primal, tangent })
}
