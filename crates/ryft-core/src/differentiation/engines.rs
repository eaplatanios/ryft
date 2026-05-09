use std::cell::RefCell;
use std::rc::Rc;

use crate::differentiation::{LinearOperation, TranspositionContext};
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::constants::SupportsZero;
use crate::parameters::Parameterized;
use crate::tracing::engines::{Tracer, TracingContext, TracingEngine};
use crate::tracing::{AtomId, Instruction, Program, ProgramBuilder, TracingError};

impl<'engine, E: TracingEngine> TracingContext<'engine, E> {
    /// Transposes a traced linear [`Program`] using this [`TracingContext`] for zero materialization.
    ///
    /// The transpose program itself is staged in a fresh linear-program builder. Any zero operation
    /// produced for a disconnected primal input is then replaced with a traced constant whose
    /// underlying outer-trace value is synthesized through [`Engine::zero`](crate::tracing::engines::Engine::zero).
    pub fn transpose<Input: Parameterized<Tracer<'engine, E>>, Output: Parameterized<Tracer<'engine, E>>, O>(
        &self,
        program: &Program<E::Type, Tracer<'engine, E>, O, Input, Output>,
    ) -> Result<Program<E::Type, Tracer<'engine, E>, O, Output, Input>, TracingError>
    where
        E: 'static,
        O: Clone
            + LinearOperation<E::Type, Tracer<'engine, E>, O>
            + SupportsZero<E::Type, Tracer<'engine, E>>
            + SupportsAdd<E::Type, Tracer<'engine, E>>,
    {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<E::Type, Tracer<'engine, E>, O>::new()));
        let mut context = TranspositionContext::new(builder);
        let pullback = context.transpose(program)?;

        let mut builder = ProgramBuilder::<E::Type, Tracer<'engine, E>, O>::new();
        builder.atoms = pullback.atoms.clone();
        builder.input_ids = pullback.input_ids.clone();
        let mut atom_remapping: Vec<Option<AtomId>> = vec![None; builder.atoms.len()];
        let mut rewritten_instructions = Vec::with_capacity(pullback.instructions.len());
        for instruction in &pullback.instructions {
            if let Some(zero_operation) = instruction.operation.as_zero_operation()
                && instruction.outputs.len() == 1
                && instruction.inputs.is_empty()
            {
                // Zero ops in traced pullbacks have no inputs from which interpretation can recover a tracing
                // context, so materialize each one as a constant in this outer trace and remap its uses.
                let zero_tracer = self.constant(self.engine.zero(&zero_operation.r#type)?);
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
        let outputs = pullback
            .output_ids
            .iter()
            .map(|atom| atom_remapping[atom.index].unwrap_or(*atom))
            .collect::<Vec<_>>();
        builder
            .build(outputs, pullback.input_structure.clone(), pullback.output_structure.clone())?
            .simplified()
    }
}
