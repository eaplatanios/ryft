use super::*;
use crate::tracing_v2::{JvpContext, LinearizationEngine};

/// Operation-level capability required by traced program linearization.
///
/// [`linearize_traced_program`] runs while an outer trace is already active. Its primal values are
/// therefore [`Tracer`] leaves, while tangent values are atom ids in a fresh linear builder. This
/// trait is the exact semantic contract needed for one staged operation carrier to participate in
/// that pass.
#[doc(hidden)]
pub trait TracedLinearizableOperation<'engine, E>: Clone + Operation<ArrayType>
where
    E: TracingEngine<Type = ArrayType> + ?Sized,
{
    /// Applies this operation's JVP rule to traced primals inside the active linearization pass.
    fn jvp_traced_linearization(
        &self,
        engine: &LinearizationEngine<'engine, E>,
        context: &mut JvpContext<'_, Tracer<'engine, E>, LinearPrimitiveOperation<Tracer<'engine, E>>>,
        inputs: &[JvpTracer<Tracer<'engine, E>, AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, E>, AtomId>>, TracingError>;
}

/// Builds a staged linear program by replaying a traced primal program on symbolic dual inputs.
///
/// In the overall architecture, this is the traced-program analogue of
/// [`super::program::linearize_program`]: instead of consuming concrete primals and producing a
/// linear program immediately, it works inside an outer JIT trace and stages the resulting
/// pushforward symbolically.
#[doc(hidden)]
pub fn linearize_traced_program<'engine, V, E>(
    engine: &'engine E,
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, V, E::Operation>>>,
    program: &Program<ArrayType, V, E::Operation, Vec<V>, Vec<V>>,
    primals: Vec<Tracer<'engine, E>>,
) -> Result<(Vec<Tracer<'engine, E>>, TracedLinearProgram<'engine, E>), TracingError>
where
    V: Traceable<ArrayType>,
    E: TracingEngine<Type = ArrayType, Value = V> + ?Sized + 'static,
    E::Operation: TracedLinearizableOperation<'engine, E> + 'static,
{
    fn tangent_for_atom<'engine, V, E>(
        primal_values: &[Option<Tracer<'engine, E>>],
        builder: &Rc<
            RefCell<ProgramBuilder<ArrayType, Tracer<'engine, E>, LinearPrimitiveOperation<Tracer<'engine, E>>>>,
        >,
        tangents: &mut [Option<AtomId>],
        atom_id: AtomId,
    ) -> Result<AtomId, TracingError>
    where
        V: Traceable<ArrayType>,
        E: TracingEngine<Type = ArrayType, Value = V> + ?Sized,
    {
        if let Some(atom) = tangents[atom_id.index] {
            return Ok(atom);
        }
        let primal = primal_values[atom_id.index].as_ref().ok_or(TracingError::UnboundAtomId { id: atom_id })?;
        let atom = builder.borrow_mut().add_constant(primal.zero_like());
        tangents[atom_id.index] = Some(atom);
        Ok(atom)
    }

    let input_count = primals.len();
    if input_count != program.input_ids.len() {
        return Err(TracingError::InvalidInputCount { expected: program.input_ids.len(), got: input_count });
    }
    let builder = Rc::new(RefCell::new(ProgramBuilder::<
        ArrayType,
        Tracer<'engine, E>,
        LinearPrimitiveOperation<Tracer<'engine, E>>,
    >::new()));
    let linearization_engine = LinearizationEngine::new(engine, tracing_builder);
    let mut primal_values: Vec<Option<Tracer<'engine, E>>> = vec![None; program.atoms.len()];
    let mut tangents: Vec<Option<AtomId>> = vec![None; program.atoms.len()];

    for (input_atom, primal) in program.input_ids.iter().copied().zip(primals.into_iter()) {
        let tangent = builder.borrow_mut().add_input(primal.r#type().into_owned());
        primal_values[input_atom.index] = Some(primal);
        tangents[input_atom.index] = Some(tangent);
    }
    for (atom_index, atom) in program.atoms.iter().enumerate() {
        if let Atom::Constant(value) = atom {
            primal_values[atom_index] = Some(linearization_engine.lift_constant(value.clone()));
        }
    }

    let mut context = JvpContext::new(builder.clone());
    for instruction in &program.instructions {
        let input_duals = instruction
            .inputs
            .iter()
            .copied()
            .map(|input_atom| {
                Ok(JvpTracer {
                    primal: primal_values[input_atom.index]
                        .clone()
                        .ok_or(TracingError::UnboundAtomId { id: input_atom })?,
                    tangent: tangent_for_atom::<V, E>(
                        primal_values.as_slice(),
                        &builder,
                        tangents.as_mut_slice(),
                        input_atom,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, TracingError>>()?;
        let output_duals = instruction.operation.jvp_traced_linearization(
            &linearization_engine,
            &mut context,
            input_duals.as_slice(),
        )?;
        if output_duals.len() != instruction.outputs.len() {
            return Err(TracingError::InvalidOutputCount {
                expected: instruction.outputs.len(),
                got: output_duals.len(),
            });
        }
        for (output_atom, output_dual) in instruction.outputs.iter().copied().zip(output_duals.into_iter()) {
            primal_values[output_atom.index] = Some(output_dual.primal);
            tangents[output_atom.index] = Some(output_dual.tangent);
        }
    }

    let primal_outputs = program
        .output_ids
        .iter()
        .copied()
        .map(|output| primal_values[output.index].clone().ok_or(TracingError::UnboundAtomId { id: output }))
        .collect::<Result<Vec<_>, _>>()?;
    let tangent_outputs = program
        .output_ids
        .iter()
        .copied()
        .map(|output| tangent_for_atom::<V, E>(primal_values.as_slice(), &builder, tangents.as_mut_slice(), output))
        .collect::<Result<Vec<_>, _>>()?;
    drop(context);
    drop(tangents);
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TracingError::EscapedProgramBuilder);
        }
    };
    let program = builder
        .build(tangent_outputs, vec![Placeholder; input_count], vec![Placeholder; primal_outputs.len()])?
        .simplified()?;
    Ok((primal_outputs.clone(), program))
}
