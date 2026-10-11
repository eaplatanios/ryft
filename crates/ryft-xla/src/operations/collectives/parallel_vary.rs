use ryft_core::macros::check_count;
use ryft_core::{ArrayType, ParallelVaryOperation};
use ryft_mlir::ValueRef;

use crate::operations::collectives::mesh_axis_replica_groups;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for ParallelVaryOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        // The forward map only weakens invariance. Resolve its axis even though it emits no communication.
        mesh_axis_replica_groups(&lowerer.collective_state, self.axis_name())?;
        Ok(vec![input_values[0]])
    }
}
