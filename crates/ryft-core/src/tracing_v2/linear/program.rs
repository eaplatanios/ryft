use super::*;

use crate::tracing_v2::JvpContext;
use crate::tracing_v2::operations::{SupportsZero, TranspositionContext};

/// Converts a staged primal program into a staged pushforward linear map.
///
/// This is the reusable IR-level form of forward-mode differentiation. Instead of evaluating the
/// JVP immediately, it builds a staged [`Program`] over linear operations that can be replayed
/// later on arbitrary tangent inputs at the same primal point.
#[doc(hidden)]
pub fn linearize_program<Input, Output, V, E, O>(
    engine: &E,
    program: &Program<E::Type, V, O, Input, Output>,
    input_primals: Vec<V>,
) -> Result<Program<E::Type, V, E::LinearOperation, Input, Output>, TracingError>
where
    V: Differentiable<E::Type, Tangent = V> + Zero<E::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    E: DifferentiableEngine<Value = V> + ?Sized,
    O: Clone + DifferentiableOperation<E>,
{
    fn tangent_for_atom<T, V, LinearOperation>(
        primal_values: &[Option<V>],
        builder: &Rc<RefCell<ProgramBuilder<T, V, LinearOperation>>>,
        tangents: &mut [Option<AtomId>],
        atom_id: AtomId,
    ) -> Result<AtomId, TracingError>
    where
        T: Type,
        V: Differentiable<T, Tangent = V> + Zero<T>,
        LinearOperation: Clone + Operation<T>,
    {
        if let Some(atom) = tangents[atom_id.index] {
            return Ok(atom);
        }
        let primal = primal_values[atom_id.index].as_ref().ok_or(TracingError::UnboundAtomId { id: atom_id })?;
        let atom = builder.borrow_mut().add_constant(<V as Zero<T>>::zero(primal.r#type().as_ref())?);
        tangents[atom_id.index] = Some(atom);
        Ok(atom)
    }

    let program = program;
    if input_primals.len() != program.input_ids.len() {
        return Err(TracingError::InvalidInputCount { expected: program.input_ids.len(), got: input_primals.len() });
    }
    let builder = Rc::new(RefCell::new(ProgramBuilder::<E::Type, V, E::LinearOperation>::new()));
    let mut primals: Vec<Option<V>> = vec![None; program.atoms.len()];
    let mut tangents: Vec<Option<AtomId>> = vec![None; program.atoms.len()];
    for (input_atom, input_primal) in program.input_ids.iter().copied().zip(input_primals.into_iter()) {
        let tangent_atom = builder.borrow_mut().add_input(input_primal.r#type().into_owned());
        tangents[input_atom.index] = Some(tangent_atom);
        primals[input_atom.index] = Some(input_primal);
    }
    for (atom_index, atom) in program.atoms.iter().enumerate() {
        let atom_id = AtomId { index: atom_index };
        if let Atom::Constant(value) = atom {
            primals[atom_id.index] = Some(value.clone());
        }
    }

    let mut context = JvpContext::<'_, V, E::LinearOperation, E::Type>::new(builder.clone());
    for instruction in &program.instructions {
        let input_duals = instruction
            .inputs
            .iter()
            .copied()
            .map(|input_atom| {
                Ok(JvpTracer {
                    primal: primals[input_atom.index].clone().ok_or(TracingError::UnboundAtomId { id: input_atom })?,
                    tangent: tangent_for_atom::<E::Type, V, E::LinearOperation>(
                        primals.as_slice(),
                        &builder,
                        tangents.as_mut_slice(),
                        input_atom,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, TracingError>>()?;
        let output_duals = instruction.operation.jvp(engine, &mut context, input_duals.as_slice())?;
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
            tangent_for_atom::<E::Type, V, E::LinearOperation>(
                primals.as_slice(),
                &builder,
                tangents.as_mut_slice(),
                output_atom,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(context);
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

fn transpose_linear_program_with_context<T, V, Input, Output, O>(
    context: &mut TranspositionContext<'_, T, V, O>,
    program: &Program<T, V, O, Input, Output>,
) -> Result<Program<T, V, O, Output, Input>, TracingError>
where
    T: Type,
    V: Traceable<T>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: Clone + LinearOperation<T, V, O> + SupportsAdd<T, V> + SupportsZero<T, V>,
{
    fn accumulate<T, V, O>(
        builder: &Rc<RefCell<ProgramBuilder<T, V, O>>>,
        adjoints: &mut [Option<AtomId>],
        atom: AtomId,
        contribution: AtomId,
    ) -> Result<(), TracingError>
    where
        T: Type,
        V: Traceable<T>,
        O: SupportsAdd<T, V> + Operation<T> + Clone,
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

    fn stage_zero<T, V, O>(builder: &Rc<RefCell<ProgramBuilder<T, V, O>>>, r#type: T) -> AtomId
    where
        T: Type,
        V: Traceable<T>,
        O: SupportsZero<T, V> + Operation<T> + Clone,
    {
        let mut builder_borrow = builder.borrow_mut();
        let output = builder_borrow.add_variable(r#type.clone());
        builder_borrow.instructions.push(Instruction {
            operation: <O as SupportsZero<T, V>>::zero_operation(r#type),
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
        accumulate::<T, V, O>(&builder, adjoints.as_mut_slice(), output, cotangent)?;
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
                accumulate::<T, V, O>(&builder, adjoints.as_mut_slice(), input, contribution)?;
            }
        }
    }

    let outputs = program
        .input_ids
        .iter()
        .copied()
        .map(|input| match adjoints[input.index] {
            Some(adjoint) => adjoint,
            None => stage_zero::<T, V, O>(&builder, program.atoms[input.index].r#type().into_owned()),
        })
        .collect::<Vec<_>>();
    drop(builder);
    let builder = context.take_builder()?;
    builder
        .build(outputs, program.output_structure.clone(), program.input_structure.clone())?
        .simplified()
}

impl<T, V, O> TranspositionContext<'_, T, V, O>
where
    T: Type,
    V: Traceable<T>,
    O: Clone + LinearOperation<T, V, O> + SupportsAdd<T, V> + SupportsZero<T, V>,
{
    /// Transposes one nested linear program into a fresh sibling builder.
    pub(crate) fn transpose_nested_program<Input, Output>(
        &mut self,
        program: &Program<T, V, O, Input, Output>,
    ) -> Result<Program<T, V, O, Output, Input>, TracingError>
    where
        Input: Parameterized<V>,
        Output: Parameterized<V>,
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
pub fn transpose_linear_program_with_output_examples<T, V, Input, Output, O>(
    program: &Program<T, V, O, Input, Output>,
    output_examples: &[V],
) -> Result<Program<T, V, O, Output, Input>, TracingError>
where
    T: Type,
    V: Traceable<T>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: Clone + LinearOperation<T, V, O> + SupportsAdd<T, V> + SupportsZero<T, V>,
{
    let expected_output_count = program.output_ids.len();
    if output_examples.len() != expected_output_count {
        return Err(TracingError::InvalidInputCount { expected: expected_output_count, got: output_examples.len() });
    }
    let builder = Rc::new(RefCell::new(ProgramBuilder::<T, V, O>::new()));
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
pub fn transpose_traced_linear_program<'engine, Input, Output, V, O, E>(
    tracing_context: TracingContext<'engine, E>,
    program: &Program<E::Type, Tracer<'engine, E>, O, Input, Output>,
) -> Result<Program<E::Type, Tracer<'engine, E>, O, Output, Input>, TracingError>
where
    V: Traceable<E::Type>,
    Input: Parameterized<Tracer<'engine, E>>,
    Output: Parameterized<Tracer<'engine, E>>,
    O: Clone
        + LinearOperation<E::Type, Tracer<'engine, E>, O>
        + SupportsAdd<E::Type, Tracer<'engine, E>>
        + SupportsZero<E::Type, Tracer<'engine, E>>,
    E: TracingEngine<Value = V> + ?Sized + 'static,
{
    let builder = Rc::new(RefCell::new(ProgramBuilder::<E::Type, Tracer<'engine, E>, O>::new()));
    let mut context = TranspositionContext::new(builder);
    let pullback = transpose_linear_program_with_context(&mut context, program)?;
    materialize_tracer_zero_ops(pullback, tracing_context)
}

/// Walks a linear program and replaces every
/// [`LinearPrimitiveOperation::Zero`](crate::tracing_v2::LinearPrimitiveOperation::Zero) op with a
/// [`Tracer`] constant atom backed by an outer-trace zero.
///
/// This is the post-processing step that the traced reverse-mode pipeline runs after transposition
/// so the returned pullback is interpretable by code that holds [`Tracer`] inputs. Tracer values
/// cannot satisfy [`Zero<ArrayType>`](crate::tracing_v2::operations::constants::Zero) statically,
/// so traced pullbacks must be materialized away from `Zero` ops before being interpreted.
fn materialize_tracer_zero_ops<'engine, Input, Output, V, O, E>(
    program: Program<E::Type, Tracer<'engine, E>, O, Input, Output>,
    tracing_context: TracingContext<'engine, E>,
) -> Result<Program<E::Type, Tracer<'engine, E>, O, Input, Output>, TracingError>
where
    V: Traceable<E::Type>,
    Input: Parameterized<Tracer<'engine, E>>,
    Output: Parameterized<Tracer<'engine, E>>,
    O: Clone
        + LinearOperation<E::Type, Tracer<'engine, E>, O>
        + SupportsAdd<E::Type, Tracer<'engine, E>>
        + SupportsZero<E::Type, Tracer<'engine, E>>,
    E: TracingEngine<Value = V> + ?Sized + 'static,
{
    let mut builder = ProgramBuilder::<E::Type, Tracer<'engine, E>, O>::new();
    builder.atoms = program.atoms.clone();
    builder.input_ids = program.input_ids.clone();
    let mut atom_remapping: Vec<Option<AtomId>> = vec![None; builder.atoms.len()];
    let mut rewritten_instructions = Vec::with_capacity(program.instructions.len());
    for instruction in &program.instructions {
        if let Some(zero_type) = instruction.operation.as_zero()
            && instruction.outputs.len() == 1
            && instruction.inputs.is_empty()
        {
            let zero_value = tracing_context.engine.zero(zero_type)?;
            let outer_atom = tracing_context.builder.borrow_mut().add_constant(zero_value);
            let zero_tracer = tracing_context.tracer(outer_atom, Some(zero_type.clone()));
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
