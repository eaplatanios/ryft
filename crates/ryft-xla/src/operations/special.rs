use ryft_core::{ArrayType, DataType, ErfOperation};
use ryft_mlir::dialects::{chlo, stable_hlo};
use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::experimental::lowering::types::{lower_f64_constant_splat, lower_tensor_type};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

// [`ErfOperation`] lowers to `chlo.erf`, which the XLA compiler legalizes to a rational polynomial approximation
// over StableHLO operations during compilation.
impl LowerableOperation for ErfOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let mut input = input_values[0];
        if output_types[0].data_type() == DataType::F8E8M0FNU {
            // Both smallest encodings have erf outputs that round to 0x01 under the exponent-only conversion
            // contract. Raise 0x00 to 0x01 before CHLO's f32 expansion, where 2^-127 can otherwise flush to zero
            // and produce NaN on conversion back. Integer comparison preserves this distinction even with FTZ.
            let byte_type = output_types[0].clone().with_data_type(DataType::U8);
            let byte_tensor_type = lower_tensor_type(&byte_type, lowerer.context, lowerer.location)?;
            let bits = lowerer.block.append_operation(stable_hlo::bitcast_convert(
                input,
                byte_tensor_type,
                lowerer.location,
            )?)?;
            let minimum = lower_f64_constant_splat(
                1.0,
                &byte_type,
                byte_tensor_type,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let normalized = lowerer.block.append_operation(stable_hlo::maximum(
                bits.result(0).unwrap().as_ref(),
                minimum,
                lowerer.location,
            )?)?;
            input = lowerer
                .block
                .append_operation(stable_hlo::bitcast_convert(
                    normalized.result(0).unwrap().as_ref(),
                    lower_tensor_type(&output_types[0], lowerer.context, lowerer.location)?,
                    lowerer.location,
                )?)?
                .result(0)
                .unwrap()
                .as_ref();
        }
        let result = lowerer.block.append_operation(chlo::erf(input, lowerer.location)?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}
