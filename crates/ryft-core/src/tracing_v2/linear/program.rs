use super::*;

/// Staged linear map produced by [`jvp_program`](super::jvp_program) or [`vjp`](super::vjp).
///
/// [`LinearProgram`] is the reusable artifact that sits between first-order tracing and
/// higher-order autodiff. A pushforward produced by forward-mode linearization and a pullback
/// produced by reverse-mode transposition are represented by the same type; only the chosen input
/// and output structures differ.
pub struct LinearProgram<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: Clone + Operation<T> = LinearPrimitiveOperation<ArrayType, V>,
> {
    /// Underlying staged program that replays the linear map.
    program: Program<T, V, O, Input, Output>,

    /// Representative additive identity used when transpose logic must synthesize missing inputs.
    zero: V,

    /// Phantom marker tying the linear program to its structured input/output parameter families.
    marker: PhantomData<fn(Input) -> Output>,
}

impl<T: Type + Display, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>, O: Clone + Operation<T>>
    Clone for LinearProgram<T, V, Input, Output, O>
where
    Input::ParameterStructure: Clone,
    Output::ParameterStructure: Clone,
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), zero: self.zero.clone(), marker: PhantomData }
    }
}

impl<T: Type + Display, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>, O: Clone + Operation<T>>
    LinearProgram<T, V, Input, Output, O>
{
    /// Wraps an already-built staged program as a linear map.
    ///
    /// Callers use this when a helper has already constructed the linear IR and just needs to tag
    /// it with the representative zero required by later transpose logic.
    #[inline]
    pub fn from_program(program: Program<T, V, O, Input, Output>, zero: V) -> Self {
        Self { program, zero, marker: PhantomData }
    }

    /// Returns the staged program backing this linear program.
    ///
    /// This is useful when a downstream helper needs to inspect or retag the underlying IR while
    /// preserving the linear-map interpretation at the API boundary.
    #[inline]
    pub fn program(&self) -> &Program<T, V, O, Input, Output> {
        &self.program
    }

    /// Applies the linear program to a concrete input tangent or cotangent.
    ///
    /// Conceptually this is no different from replaying an ordinary [`Program`], but the
    /// documentation deliberately uses linear language because these programs represent derived
    /// linear maps rather than original user functions.
    pub fn call(&self, input: Input) -> Result<Output, TracingError>
    where
        O: InterpretableOperation<T, V>,
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.program.interpret(input)
    }
}

impl<V: Traceable<ArrayType>, Input: Parameterized<V>, Output: Parameterized<V>, O: Clone + Operation<ArrayType>>
    LinearProgram<ArrayType, V, Input, Output, O>
{
    /// Transposes the linear program, turning a pushforward into a pullback.
    #[allow(private_bounds)]
    pub fn transpose(&self) -> Result<LinearProgram<ArrayType, V, Output, Input, O>, TracingError>
    where
        V: ZeroLike,
        O: CoreLinearProgramOperation<V> + LinearAddOperation<ArrayType, V> + Clone,
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
        .map(|term| term.map(|term| term.atom()))
        .collect())
}

/// Converts a staged primal program into a staged pushforward linear map.
///
/// This is the reusable IR-level form of forward-mode differentiation. Instead of evaluating the
/// JVP immediately, it builds a [`LinearProgram`] that can be replayed later on arbitrary tangent
/// inputs at the same primal point.
pub(crate) fn linearize_program<Input, Output, V, O, L>(
    engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
    program: &Program<ArrayType, V, O, Input, Output>,
    input_primals: Vec<V>,
) -> Result<LinearProgram<ArrayType, V, Input, Output, L>, TracingError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    L: Clone + Operation<ArrayType>,
    O: Clone + DifferentiableOperation<ArrayType, V, LinearTerm<ArrayType, V, L>, O, L>,
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
    let zero = input_primals.first().map(ZeroLike::zero_like).ok_or(TracingError::EmptyParameterizedValue)?;
    let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, V, L>::new()));
    let mut primals: Vec<Option<V>> = vec![None; program.atoms.len()];
    let mut tangents: Vec<Option<LinearTerm<ArrayType, V, L>>> = vec![None; program.atoms.len()];
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
        let output_duals = DifferentiableOperation::<ArrayType, V, LinearTerm<ArrayType, V, L>, O, L>::jvp(
            &instruction.operation,
            engine,
            input_duals.as_slice(),
        )?;
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
                .map(|term| term.atom())
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(tangents);
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TracingError::InternalInvariantViolation("linearization builder escaped the tracing scope"));
        }
    };
    Ok(LinearProgram {
        program: builder
            .build::<Input, Output>(output_tangents, program.input_structure.clone(), program.output_structure.clone())
            .simplified()?,
        zero,
        marker: PhantomData,
    })
}

/// Transposes a linear pushforward program into its reverse-mode pullback.
///
/// This is the IR-level core of reverse-mode AD in `tracing_v2`. Higher-level helpers such as
/// [`vjp`](super::vjp) and [`grad`](super::grad) build on this operation after first producing a
/// forward linear program.
#[allow(private_bounds)]
pub fn transpose_linear_program<V, Input, Output, O>(
    program: &LinearProgram<ArrayType, V, Input, Output, O>,
) -> Result<LinearProgram<ArrayType, V, Output, Input, O>, TracingError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: CoreLinearProgramOperation<V> + LinearAddOperation<ArrayType, V> + Clone,
{
    let zero = program.zero.zero_like();
    transpose_linear_program_with_output_inputs(program, |builder: &mut ProgramBuilder<ArrayType, V, O>, _, _| {
        Ok(builder.add_input(zero.r#type().into_owned()))
    })
}

fn transpose_linear_program_with_output_inputs<V, Input, Output, O, F>(
    program: &LinearProgram<ArrayType, V, Input, Output, O>,
    mut make_output_cotangent_input: F,
) -> Result<LinearProgram<ArrayType, V, Output, Input, O>, TracingError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    F: FnMut(&mut ProgramBuilder<ArrayType, V, O>, &ArrayType, usize) -> Result<AtomId, TracingError>,
    O: CoreLinearProgramOperation<V> + LinearAddOperation<ArrayType, V> + Clone,
{
    fn accumulate<V, O>(
        builder: &Rc<RefCell<ProgramBuilder<ArrayType, V, O>>>,
        adjoints: &mut [Option<AtomId>],
        atom: AtomId,
        contribution: AtomId,
    ) -> Result<(), TracingError>
    where
        V: Traceable<ArrayType>,
        O: LinearAddOperation<ArrayType, V> + Operation<ArrayType> + Clone,
    {
        adjoints[atom.index] = Some(match adjoints[atom.index] {
            Some(existing) => {
                let mut builder_borrow = builder.borrow_mut();
                let abstract_value =
                    builder_borrow.atom(existing).expect("adjoint atom should exist").r#type().into_owned();
                builder_borrow.add_instruction_prevalidated(
                    O::linear_add_op(),
                    vec![existing, contribution],
                    vec![abstract_value],
                )[0]
            }
            None => contribution,
        });
        Ok(())
    }

    let linear_body = &program.program;
    let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, V, O>::new()));
    let mut output_cotangent_inputs = Vec::with_capacity(linear_body.output_ids.len());
    for (output_index, output) in linear_body.output_ids.iter().enumerate() {
        let output_atom = linear_body.atoms.get(output.index).ok_or(TracingError::UnboundAtomId { id: *output })?;
        let cotangent_input =
            make_output_cotangent_input(&mut builder.borrow_mut(), &output_atom.r#type(), output_index)?;
        output_cotangent_inputs.push(cotangent_input);
    }

    let mut adjoints = vec![None; linear_body.atoms.len()];
    for (cotangent, output) in output_cotangent_inputs.into_iter().zip(linear_body.output_ids.iter().copied()) {
        accumulate(&builder, adjoints.as_mut_slice(), output, cotangent)?;
    }

    for instruction in linear_body.instructions.iter().rev() {
        let instruction_output_cotangents =
            instruction.outputs.iter().map(|output| adjoints[output.index]).collect::<Option<Vec<_>>>();
        let Some(instruction_output_cotangents) = instruction_output_cotangents else {
            continue;
        };
        let input_cotangents = transpose(&instruction.operation, &builder, instruction_output_cotangents.as_slice())?;
        for (input, contribution) in instruction.inputs.iter().copied().zip(input_cotangents) {
            if let Some(contribution) = contribution {
                accumulate(&builder, adjoints.as_mut_slice(), input, contribution)?;
            }
        }
    }

    let zero_atom = builder.borrow_mut().add_constant(program.zero.clone());
    let outputs = linear_body
        .input_ids
        .iter()
        .copied()
        .map(|input| adjoints[input.index].unwrap_or(zero_atom))
        .collect::<Vec<_>>();
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TracingError::InternalInvariantViolation(
                "transpose builder should not have outstanding linear terms",
            ));
        }
    };
    Ok(LinearProgram {
        program: builder
            .build::<Output, Input>(outputs, linear_body.output_structure.clone(), linear_body.input_structure.clone())
            .simplified()?,
        zero: program.zero.clone(),
        marker: PhantomData,
    })
}

/// Transposes a linear program using concrete output examples to seed the cotangent inputs.
///
/// This variant is useful when the linear program's leaf type cannot be synthesized from bare
/// [`ArrayType`] metadata alone, but the caller still has representative output values available.
/// It plays the same architectural role as [`transpose_linear_program`], but swaps metadata-driven
/// cotangent synthesis for exemplar-driven synthesis.
#[allow(private_bounds)]
pub fn transpose_linear_program_with_output_examples<V, Input, Output, O>(
    program: &LinearProgram<ArrayType, V, Input, Output, O>,
    output_examples: &[V],
) -> Result<LinearProgram<ArrayType, V, Output, Input, O>, TracingError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: CoreLinearProgramOperation<V> + LinearAddOperation<ArrayType, V> + Clone,
{
    let expected_output_count = program.program().output_ids.len();
    if output_examples.len() != expected_output_count {
        return Err(TracingError::InvalidInputCount { expected: expected_output_count, got: output_examples.len() });
    }
    transpose_linear_program_with_output_inputs(
        program,
        |builder: &mut ProgramBuilder<ArrayType, V, O>, _, output_index| {
            Ok(builder.add_input(output_examples[output_index].r#type().into_owned()))
        },
    )
}

fn lift_traced_constant<'engine, V, O: Clone + Operation<ArrayType>, L: Clone, E>(
    constant: &V,
    inputs: &[Tracer<'engine, E>],
) -> Result<Tracer<'engine, E>, TracingError>
where
    V: Traceable<ArrayType>,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
{
    let exemplar = inputs.first().ok_or(TracingError::EmptyParameterizedValue)?;
    let builder = exemplar.builder.clone();
    let atom = builder.borrow_mut().add_constant(constant.clone());
    Ok(Tracer::from_engine(atom, builder, exemplar.engine))
}

pub(crate) fn lift_linearized_traced_constant<
    'engine,
    V,
    O: Clone + Operation<ArrayType> + 'static,
    L: Clone + Operation<ArrayType> + 'static,
    E,
>(
    constant: &V,
    inputs: &[LinearizedTracedValue<'engine, E>],
) -> Result<LinearizedTracedValue<'engine, E>, TracingError>
where
    V: Traceable<ArrayType> + ZeroLike,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
{
    let exemplar = inputs.first().ok_or(TracingError::EmptyParameterizedValue)?;
    let primal = lift_traced_constant::<V, O, L, E>(constant, std::slice::from_ref(&exemplar.primal))?;
    let tangent_atom = exemplar.tangent.builder_handle().borrow_mut().add_constant(primal.zero_like());
    let tangent = LinearTerm::from_staged_parts(tangent_atom, exemplar.tangent.builder_handle());
    Ok(Linearized { primal, tangent })
}
