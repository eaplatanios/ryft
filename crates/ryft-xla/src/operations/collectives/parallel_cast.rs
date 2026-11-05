use ryft_core::macros::check_count;
use ryft_core::{ArrayType, Operation as CoreOperation, ParallelCastOperation, ProgramError};
use ryft_mlir::ValueRef;

use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for ParallelCastOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        check_count!("input", input_values, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        // A cast emits no communication, but its state and adjoint belong to a checked physical binding. Inference
        // validates the source state and guarantees this mesh and axis exist before the owning binder is checked.
        let inferred = self.infer_output_types(input_types, &[]).map_err(ProgramError::from)?;
        let mesh = input_types[0].sharding().unwrap().mesh();
        lowerer.collective_state.check_manual_collective(
            self.name(),
            self.axis_name(),
            mesh.axis_size(self.axis_name()).unwrap(),
            Some(mesh),
            output_types,
            |_| Ok(inferred),
        )?;
        Ok(vec![input_values[0]])
    }
}
