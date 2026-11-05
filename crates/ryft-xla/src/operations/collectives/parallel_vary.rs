use ryft_core::macros::check_count;
use ryft_core::{ArrayType, Operation as CoreOperation, ParallelVaryOperation, ProgramError};
use ryft_mlir::ValueRef;

use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for ParallelVaryOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        check_count!("input", input_values, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        // Weakening invariance emits no communication, but its sum adjoint belongs to a checked physical binding.
        // Inference guarantees this input mesh and axis exist before the owning binder is checked.
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
