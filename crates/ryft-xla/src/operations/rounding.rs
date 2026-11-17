use ryft_core::{ArrayType, CeilOperation, FloorOperation, RoundOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for FloorOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let result = lowerer.block().append_operation(stable_hlo::floor(input_values[0], lowerer.location())?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

impl LowerableOperation for CeilOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let result = lowerer.block().append_operation(stable_hlo::ceil(input_values[0], lowerer.location())?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

impl LowerableOperation for RoundOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let result = lowerer
            .block()
            .append_operation(stable_hlo::round_with_nearest_even_tie_break(input_values[0], lowerer.location())?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{Ceil, DataType, Floor, Round};

    use crate::sessions::tests::{domain_mesh, session_with_client};
    use crate::tests::{execution_client, replicated_array, values_from_bytes, values_to_bytes};

    // Signed zeros, quarters, and ties around zero and one, compared by encoding so that the sign of a zero result and
    // the rounding direction at ties are both observable.
    const F32_INPUTS: [f32; 8] = [-0f32, 0.0, -0.25, 0.25, -0.5, 0.5, 1.5, 2.5];

    // The F8E8M0FNU encodings of 0.25, 0.5, 1.0, and NaN.
    const F8E8M0FNU_INPUTS: [u8; 4] = [125, 126, 127, 255];

    #[test]
    fn test_floor_operation_lower() {
        let client = execution_client();
        let session = session_with_client(client);
        let mesh = domain_mesh(client, "x", 1);
        let input = replicated_array(&session, &mesh, DataType::F32, &[8], &values_to_bytes(&F32_INPUTS));
        assert_eq!(
            values_from_bytes::<u32>(&input.floor().unwrap().copy_to_host().unwrap()),
            [-0f32, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 2.0].map(f32::to_bits),
        );

        // Floats with only exponent bits cannot represent zero, so rounding down to zero produces NaN instead of
        // failing.
        let input = replicated_array(&session, &mesh, DataType::F8E8M0FNU, &[4], &F8E8M0FNU_INPUTS);
        assert_eq!(input.floor().unwrap().copy_to_host().unwrap(), [255, 255, 127, 255]);
    }

    #[test]
    fn test_ceil_operation_lower() {
        let client = execution_client();
        let session = session_with_client(client);
        let mesh = domain_mesh(client, "x", 1);
        let input = replicated_array(&session, &mesh, DataType::F32, &[8], &values_to_bytes(&F32_INPUTS));
        assert_eq!(
            values_from_bytes::<u32>(&input.ceil().unwrap().copy_to_host().unwrap()),
            [-0f32, 0.0, -0.0, 1.0, -0.0, 1.0, 2.0, 3.0].map(f32::to_bits),
        );

        // Rounding up stays representable in the format with only exponent bits, while NaN remains NaN.
        let input = replicated_array(&session, &mesh, DataType::F8E8M0FNU, &[4], &F8E8M0FNU_INPUTS);
        assert_eq!(input.ceil().unwrap().copy_to_host().unwrap(), [127, 127, 127, 255]);
    }

    #[test]
    fn test_round_operation_lower() {
        let client = execution_client();
        let session = session_with_client(client);
        let mesh = domain_mesh(client, "x", 1);
        let input = replicated_array(&session, &mesh, DataType::F32, &[8], &values_to_bytes(&F32_INPUTS));
        assert_eq!(
            values_from_bytes::<u32>(&input.round().unwrap().copy_to_host().unwrap()),
            [-0f32, 0.0, -0.0, 0.0, -0.0, 0.0, 2.0, 2.0].map(f32::to_bits),
        );

        // Floats with only exponent bits cannot represent zero, so rounding to zero produces NaN instead of failing.
        let input = replicated_array(&session, &mesh, DataType::F8E8M0FNU, &[4], &F8E8M0FNU_INPUTS);
        assert_eq!(input.round().unwrap().copy_to_host().unwrap(), [255, 255, 127, 255]);
    }
}
