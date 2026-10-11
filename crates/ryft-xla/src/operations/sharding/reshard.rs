use ryft_core::{ArrayType, ReshardOperation};
use ryft_mlir::ValueRef;

use crate::operations::sharding::constrain_sharding::lower_sharding_constraint;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for ReshardOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_sharding_constraint(
            input_values,
            self.sharding(),
            &lowerer.collective_state.bound_manual_axes,
            &mut lowerer.block,
            lowerer.location,
        )
    }
}
