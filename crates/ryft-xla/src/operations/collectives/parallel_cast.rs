use ryft_core::macros::check_count;
use ryft_core::{ArrayType, ParallelCastOperation};
use ryft_mlir::ValueRef;

use crate::operations::collectives::mesh_axis_replica_groups;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for ParallelCastOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        check_count!("input", input_values, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        // A cast changes only the state that the types record, so it emits no communication. Its axis is still
        // resolved, so that a cast over an axis that the enclosing manual region does not bind fails to lower.
        mesh_axis_replica_groups(&lowerer.collective_state, self.axis_name())?;
        Ok(vec![input_values[0]])
    }
}
