use std::cell::RefCell;
use std::rc::Rc;

use crate::differentiation::{LinearOperation, TranspositionContext};
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::constants::SupportsZero;
use crate::parameters::Parameterized;
use crate::tracing::engines::{Tracer, TracingContext, TracingEngine};
use crate::tracing::{Instruction, Program, ProgramBuilder, TracingError};

impl<'engine, E: TracingEngine> TracingContext<'engine, E> {
    /// Transposes the provided linear [`Program`]. The transposed program is first staged as an ordinary linear
    /// [`Program`]. This is used for transforming _pushforward_ functions into _pullback_ functions during automatic
    /// differentiation. When a primal input is disconnected from the outputs, transposition represents its cotangent
    /// as a [`ZeroOperation`](crate::operations::ZeroOperation). Such an input-free operation cannot recover the
    /// surrounding [`TracingContext`] during later interpretation, and so this method replaces each standalone zero
    /// operation with a constant [`Tracer`] created in this [`TracingContext`]. The concrete zero value stored in that
    /// tracer is synthesized through [`Engine::zero`](crate::tracing::Engine::zero), while the final pullback still
    /// receives and returns traced cotangent values.
    pub fn transpose<
        Input: Parameterized<Tracer<'engine, E>>,
        Output: Parameterized<Tracer<'engine, E>>,
        O: Clone
            + LinearOperation<E::Type, Tracer<'engine, E>, O>
            + SupportsZero<E::Type, Tracer<'engine, E>>
            + SupportsAdd<E::Type, Tracer<'engine, E>>,
    >(
        &self,
        program: &Program<E::Type, Tracer<'engine, E>, O, Input, Output>,
    ) -> Result<Program<E::Type, Tracer<'engine, E>, O, Output, Input>, TracingError> {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<E::Type, Tracer<'engine, E>, O>::new()));
        let mut context = TranspositionContext::new(builder);
        let pullback = context.transpose(program)?;
        let mut builder = ProgramBuilder::<E::Type, Tracer<'engine, E>, O>::new();
        builder.atoms = pullback.atoms.clone();
        builder.input_ids = pullback.input_ids.clone();
        let mut atom_remapping = vec![None; builder.atoms.len()];
        let mut rewritten_instructions = Vec::with_capacity(pullback.instructions.len());
        for instruction in &pullback.instructions {
            if let Some(zero_operation) = instruction.operation.as_zero_operation()
                && instruction.outputs.len() == 1
                && instruction.inputs.is_empty()
            {
                // Zero operations in traced pullbacks have no inputs from which interpretation can recover a tracing
                // context, and so we materialize each one as a constant in this outer tracing context and remap its
                // uses.
                let zero = builder.add_constant(self.constant(self.engine.zero(&zero_operation.r#type)?));
                atom_remapping[instruction.outputs[0].index] = Some(zero);
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
            .into_simplified()
    }
}
