use ryft_core::{ArrayType, DataType, ErfOperation};
use ryft_mlir::dialects::{chlo, stable_hlo};
use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::experimental::lowering::types::{lower_f64_constant_splat, lower_tensor_type};
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

// [`ErfOperation`] lowers to `chlo.erf`, which the XLA compiler legalizes to a rational polynomial approximation
// over StableHLO operations during compilation.
impl LowerableOperation for ErfOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{Array as CoreArray, ArrayType, DataType, Erf};

    use crate::Session;
    use crate::domains::tests::domain_mesh;
    use crate::tests::{execution_client, replicated_array, values_from_bytes, values_to_bytes};

    #[test]
    fn test_erf_operation_lower() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // The grid covers every approximation regime of the reference implementation on both signs (the series near
        // zero, the primary interval, the interval around one, and both tails of the complementary function), plus
        // zero and the saturated regime at |x| ≥ 6.
        let values = [
            -6.5f32, -4.0, -3.0, -2.0, -1.5, -1.0, -0.9, -0.5, -0.1, -1e-3, -1e-10, 0.0, 1e-10, 1e-3, 0.1, 0.5, 0.9,
            1.0, 1.2, 1.25, 2.0, 2.9, 3.0, 4.0, 6.5,
        ];
        let input = replicated_array(&domain, &mesh, DataType::F32, &[values.len()], &values_to_bytes(&values));
        let device_values = values_from_bytes::<f32>(&input.erf().unwrap().copy_to_host().unwrap());
        let reference_values = CoreArray::vector(values.to_vec()).unwrap().erf().unwrap().to_f64s();
        assert_eq!(device_values.len(), reference_values.len());
        for ((input, device_value), reference_value) in
            values.iter().zip(device_values.iter()).zip(reference_values.iter())
        {
            // XLA legalizes `chlo.erf` to a polynomial approximation in `f32`, while the reference rounds an `f64`
            // evaluation. Agreement is therefore checked within an `f32` relative tolerance, with an absolute floor
            // for the inputs near zero.
            let tolerance = 1e-6f64.max(1e-6 * f64::abs(*reference_value));
            assert!(
                (f64::from(*device_value) - reference_value).abs() < tolerance,
                "erf({input}) disagrees: XLA computed {device_value} but the reference backend computed \
                 {reference_value}",
            );
        }

        // The approximation saturates exactly at both extremes.
        assert_eq!(device_values[0], -1.0);
        assert_eq!(device_values[values.len() - 1], 1.0);
    }

    #[test]
    fn test_erf_operation_lower_f8e8m0fnu() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Cover every exponent-only encoding, including the smallest input that the lowering raises and NaN.
        let bytes = (0..=255u8).collect::<Vec<_>>();
        let input = replicated_array(&domain, &mesh, DataType::F8E8M0FNU, &[256], &bytes);
        let output = input.erf().unwrap().copy_to_host().unwrap();
        assert_eq!(&output[..3], &[1, 1, 2]);
        let reference =
            CoreArray::new(ArrayType::new_static(DataType::F8E8M0FNU, [256]), bytes).unwrap().erf().unwrap();
        assert_eq!(output, reference.logical_bytes());
    }
}
