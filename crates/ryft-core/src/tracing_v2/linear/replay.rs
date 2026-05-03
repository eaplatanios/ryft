use crate::macros::check_count;
use crate::tracing::engines::TracingContext;
use crate::tracing_v2::JvpContext;
use crate::types::Type;

use super::*;

impl<'engine, E: DifferentiableTracingEngine> TracingContext<'engine, E> {
    /// Builds a staged linear program by replaying a traced primal program on symbolic dual inputs.
    ///
    /// In the overall architecture, this is the traced-program analogue of
    /// [`Program::linearize`](crate::tracing::Program::linearize): instead of consuming concrete primals and producing
    /// a linear program immediately, it works inside this outer trace and stages the resulting pushforward
    /// symbolically. The traced primal `program` may have any parameterized input and output structure; the returned
    /// pushforward program is flat because downstream traced AD passes operate on flat tangent leaves.
    ///
    /// # Parameters
    ///
    ///   - `program`: Traced primal program to replay in JVP form.
    ///   - `primals`: Traced primal leaves aligned with `program`'s input atoms.
    pub fn linearize<
        T: Type,
        Input: Parameterized<V>,
        Output: Parameterized<V>,
        V: Traceable<T> + Differentiable<T, Tangent = V>,
        O: DifferentiableOperation<TracingContext<'engine, E>> + Operation<T>,
    >(
        &self,
        program: &Program<T, V, O, Input, Output>,
        primals: Vec<Tracer<'engine, E>>,
    ) -> Result<
        (
            Vec<Tracer<'engine, E>>,
            Program<
                T,
                Tracer<'engine, E>,
                <E as DifferentiableTracingEngine>::LinearOperationCarrier<'engine>,
                Vec<Tracer<'engine, E>>,
                Vec<Tracer<'engine, E>>,
            >,
        ),
        TracingError,
    >
    where
        E: DifferentiableTracingEngine<Type = T, Value = V> + 'static,
        E::OperationCarrier: SupportsZeroLike<T, V> + SupportsAdd<T, V> + 'static,
        AddOperation: InterpretableOperation<T, Tracer<'engine, E>>,
    {
        fn tangent_for_atom<'engine, T, V, E>(
            primal_values: &[Option<Tracer<'engine, E>>],
            builder: &Rc<
                RefCell<
                    ProgramBuilder<
                        T,
                        Tracer<'engine, E>,
                        <E as DifferentiableTracingEngine>::LinearOperationCarrier<'engine>,
                    >,
                >,
            >,
            tangents: &mut [Option<AtomId>],
            atom_id: AtomId,
        ) -> Result<AtomId, TracingError>
        where
            T: Type,
            V: Traceable<T>,
            E: DifferentiableTracingEngine<Type = T, Value = V>,
            E::OperationCarrier: SupportsZeroLike<T, V>,
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
            T,
            Tracer<'engine, E>,
            <E as DifferentiableTracingEngine>::LinearOperationCarrier<'engine>,
        >::new()));
        let mut primal_values: Vec<Option<Tracer<'engine, E>>> = vec![None; program.atoms.len()];
        let mut tangents: Vec<Option<AtomId>> = vec![None; program.atoms.len()];

        for (input_atom, primal) in program.input_ids.iter().copied().zip(primals.into_iter()) {
            let tangent = builder.borrow_mut().add_input(primal.r#type().into_owned());
            primal_values[input_atom.index] = Some(primal);
            tangents[input_atom.index] = Some(tangent);
        }
        for (atom_index, atom) in program.atoms.iter().enumerate() {
            if let Atom::Constant(value) = atom {
                primal_values[atom_index] = Some(self.constant(value.clone()));
            }
        }

        let mut context = JvpContext::new(self, builder.clone());
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
                        tangent: tangent_for_atom::<T, V, E>(
                            primal_values.as_slice(),
                            &builder,
                            tangents.as_mut_slice(),
                            input_atom,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, TracingError>>()?;
            let output_duals = instruction.operation.jvp(&mut context, input_duals.as_slice())?;
            check_count!("output", output_duals, instruction.outputs.len(), TracingError);
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
            .map(|output| {
                tangent_for_atom::<T, V, E>(primal_values.as_slice(), &builder, tangents.as_mut_slice(), output)
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
        let program = builder
            .build(tangent_outputs, vec![Placeholder; input_count], vec![Placeholder; primal_outputs.len()])?
            .simplified()?;
        Ok((primal_outputs, program))
    }
}
