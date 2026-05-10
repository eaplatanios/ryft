use crate::macros::check_count;
use crate::tracing::domains::{RuntimeDomain, Tracer, TracingContext, TracingDomain};
use crate::tracing::{Atom, AtomId};
use crate::tracing_v2::JvpContext;

use super::*;

type TracedLinearOperationCarrier<'domain, D> =
    <<D as TracingDomain>::OperationCarrier as DifferentiableTracingOperationCarrier<D>>::LinearOperationCarrier<
        'domain,
    >;

impl<'domain, D> TracingContext<'domain, D>
where
    D: DifferentiableTracingDomain + RuntimeDomain + 'domain,
    D::Value: Differentiable<D::Type> + 'domain,
    D::OperationCarrier: DifferentiableOperation<TracingContext<'domain, D>>
        + SupportsZeroLike<D::Type, D::Value>
        + SupportsAdd<D::Type, D::Value>
        + 'domain,
    AddOperation: InterpretableOperation<D::Type, Tracer<'domain, D>>,
{
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
    pub fn linearize<Input: Parameterized<D::Value>, Output: Parameterized<D::Value>>(
        &self,
        program: &Program<D::Type, D::Value, D::OperationCarrier, Input, Output>,
        primals: Vec<Tracer<'domain, D>>,
    ) -> Result<
        (
            Vec<Tracer<'domain, D>>,
            Program<
                D::Type,
                Tracer<'domain, D>,
                TracedLinearOperationCarrier<'domain, D>,
                Vec<Tracer<'domain, D>>,
                Vec<Tracer<'domain, D>>,
            >,
        ),
        TracingError,
    > {
        fn tangent_for_atom<'jvp, 'domain, D>(
            primal_values: &[Option<Tracer<'domain, D>>],
            context: &JvpContext<'jvp, TracingContext<'domain, D>>,
            tangents: &mut [Option<Tracer<'jvp, TracingContext<'domain, D>>>],
            atom_id: AtomId,
        ) -> Result<Tracer<'jvp, TracingContext<'domain, D>>, TracingError>
        where
            D: DifferentiableTracingDomain + RuntimeDomain + 'domain,
            D::Value: Differentiable<D::Type> + 'domain,
            D::OperationCarrier: SupportsAdd<D::Type, D::Value> + 'domain,
            AddOperation: InterpretableOperation<D::Type, Tracer<'domain, D>>,
        {
            if let Some(tangent) = &tangents[atom_id.index] {
                return Ok(tangent.clone());
            }
            let primal = primal_values[atom_id.index].as_ref().ok_or(TracingError::UnboundAtomId { id: atom_id })?;
            let tangent = context.add_constant(primal.tangent_type()?);
            tangents[atom_id.index] = Some(tangent.clone());
            Ok(tangent)
        }

        let input_count = primals.len();
        if input_count != program.input_ids.len() {
            return Err(TracingError::InvalidInputCount { expected: program.input_ids.len(), got: input_count });
        }
        let builder = Rc::new(RefCell::new(ProgramBuilder::<
            D::Type,
            Tracer<'domain, D>,
            TracedLinearOperationCarrier<'domain, D>,
        >::new()));
        let mut primal_values: Vec<Option<Tracer<'domain, D>>> = vec![None; program.atoms.len()];
        let mut tangents: Vec<Option<Tracer<'_, TracingContext<'domain, D>>>> = vec![None; program.atoms.len()];
        let mut context = JvpContext::new(self, builder.clone());

        for (input_atom, primal) in program.input_ids.iter().copied().zip(primals.into_iter()) {
            let tangent = context.linear_context.input(primal.r#type().into_owned());
            primal_values[input_atom.index] = Some(primal);
            tangents[input_atom.index] = Some(tangent);
        }
        for (atom_index, atom) in program.atoms.iter().enumerate() {
            if let Atom::Constant(value) = atom {
                primal_values[atom_index] = Some(self.constant(value.clone()));
            }
        }

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
                        tangent: tangent_for_atom::<D>(
                            primal_values.as_slice(),
                            &context,
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
            .map(|output| tangent_for_atom::<D>(primal_values.as_slice(), &context, tangents.as_mut_slice(), output))
            .collect::<Result<Vec<_>, _>>()?;
        let tangent_output_atoms = tangent_outputs.iter().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
        drop(tangent_outputs);
        drop(context);
        drop(tangents);
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => {
                return Err(TracingError::EscapedProgramBuilder);
            }
        };
        let program = builder
            .build(tangent_output_atoms, vec![Placeholder; input_count], vec![Placeholder; primal_outputs.len()])?
            .simplified()?;
        Ok((primal_outputs, program))
    }
}
