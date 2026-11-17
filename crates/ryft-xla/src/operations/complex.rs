use ryft_core::{ArrayType, ComplexOperation, ConjugateOperation, DataType, ImaginaryOperation, RealOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::Accuracy;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, TensorTypeRef,
    Value as MlirValue, ValueRef,
};

use crate::lowering::types::{lower_f64_constant_splat, lower_tensor_type};
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for ComplexOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let result = lowerer.block().append_operation(stable_hlo::complex(
            input_values[0],
            input_values[1],
            lowerer.location(),
        )?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

// StableHLO has no conjugation operation, so `conjugate` lowers to the `complex(real(z), negate(imag(z)))`
// composition, which is the same decomposition that JAX's `conj` lowering uses.
impl LowerableOperation for ConjugateOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let real = lowerer.block().append_operation(stable_hlo::real(input_values[0], lowerer.location())?)?;
        let imaginary = lowerer.block().append_operation(stable_hlo::imag(input_values[0], lowerer.location())?)?;
        let negated = lowerer
            .block()
            .append_operation(stable_hlo::negate(imaginary.result(0).unwrap().as_ref(), lowerer.location())?)?;
        let result = lowerer.block().append_operation(stable_hlo::complex(
            real.result(0).unwrap().as_ref(),
            negated.result(0).unwrap().as_ref(),
            lowerer.location(),
        )?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

impl LowerableOperation for RealOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let result = lowerer.block().append_operation(stable_hlo::real(input_values[0], lowerer.location())?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

impl LowerableOperation for ImaginaryOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let result = lowerer.block().append_operation(stable_hlo::imag(input_values[0], lowerer.location())?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

/// Brings the imaginary component of the complex `value` onto the principal branch `[-pi, pi]`, which is where the
/// prefixes of a complex log-sum-exp `cumulative` that are raw inputs must land to agree with its combined prefixes.
/// Phases that already lie on the branch are kept exactly as they are, and others are reduced through `atan2` of
/// their sine and cosine rather than through a floating-point remainder by `2π`, which cannot reduce huge arguments
/// accurately.
pub(crate) fn lower_principal_phase_to_mlir<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let part_type = output_type.clone().with_data_type(if output_type.data_type() == DataType::C64 {
        DataType::F32
    } else {
        DataType::F64
    });
    let part_tensor_type = lower_tensor_type(&part_type, context, location)?;
    let real = block.append_operation(stable_hlo::real(value, location)?)?.result(0).unwrap().as_ref();
    let imaginary = block.append_operation(stable_hlo::imag(value, location)?)?.result(0).unwrap().as_ref();
    let sine = block
        .append_operation(stable_hlo::sine(imaginary, Accuracy::Default, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let cosine = block
        .append_operation(stable_hlo::cosine(imaginary, Accuracy::Default, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let reduced = block.append_operation(stable_hlo::atan2(sine, cosine, location)?)?.result(0).unwrap().as_ref();
    let imaginary = lower_select_principal_phase_to_mlir(
        imaginary,
        reduced,
        &part_type,
        part_tensor_type,
        block,
        context,
        location,
    )?;
    Ok(block.append_operation(stable_hlo::complex(real, imaginary, location)?)?.result(0).unwrap().as_ref())
}

/// Selects the real `phase` where it already lies on the principal branch `[-pi, pi]`, and the `reduced` phase
/// elsewhere.
pub(crate) fn lower_select_principal_phase_to_mlir<'b, 'c: 'b, 't: 'c>(
    phase: ValueRef<'b, 'c, 't>,
    reduced: ValueRef<'b, 'c, 't>,
    part_type: &ArrayType,
    part_tensor_type: TensorTypeRef<'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let pi = lower_f64_constant_splat(std::f64::consts::PI, part_type, part_tensor_type, block, context, location)?;
    let negative_pi = block.append_operation(stable_hlo::negate(pi, location)?)?.result(0).unwrap().as_ref();
    let above_lower = block
        .append_operation(stable_hlo::compare(
            phase,
            negative_pi,
            stable_hlo::ComparisonDirection::GreaterThanOrEqual,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    let below_upper = block
        .append_operation(stable_hlo::compare(
            phase,
            pi,
            stable_hlo::ComparisonDirection::LessThanOrEqual,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    let principal = block
        .append_operation(stable_hlo::and(above_lower, below_upper, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    Ok(block
        .append_operation(stable_hlo::select(principal, phase, reduced, location)?)?
        .result(0)
        .unwrap()
        .as_ref())
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayOperation, Conjugate, Placeholder, Real, differentiate_at};

    use crate::ArrayConstant;
    use crate::lowering::tests::{ProgramBuilder, lowered_unary_module, to_mlir_module_for_plain_program};
    use crate::sessions::tests::{domain_mesh, session_with_client};
    use crate::tests::{execution_client, replicated_array, values_from_bytes, values_to_bytes};

    use super::*;

    #[test]
    fn test_complex_operation_lower() {
        let mut builder = ProgramBuilder::new();
        let real = builder.add_input(ArrayType::new_static(DataType::F32, [2]));
        let imaginary = builder.add_input(ArrayType::new_static(DataType::F32, [2]));
        let output = builder
            .add_instruction(ArrayOperation::Complex(ComplexOperation::new()), Vec::new(), vec![real, imaginary], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xcomplex<f32>> {
                    %0 = stablehlo.complex %arg0, %arg1 : tensor<2xcomplex<f32>>
                    return %0 : tensor<2xcomplex<f32>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_conjugate_operation_lower() {
        assert_eq!(
            lowered_unary_module(ArrayOperation::Conjugate(ConjugateOperation::new()), DataType::C64, vec![2]).unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
                    %0 = stablehlo.real %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
                    %1 = stablehlo.imag %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
                    %2 = stablehlo.negate %1 : tensor<2xf32>
                    %3 = stablehlo.complex %0, %2 : tensor<2xcomplex<f32>>
                    return %3 : tensor<2xcomplex<f32>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_conjugate_operation_lower_differentiation() {
        let client = execution_client();
        let session = session_with_client(client);
        let mesh = domain_mesh(client, "x", 1);
        let z = Complex::new(0.7f32, -0.3);
        let input = replicated_array(&session, &mesh, DataType::C64, &[], &values_to_bytes(&[z]));

        // The gradient of the real-valued |z|² = Re(z · conj(z)) is 2 · conj(z). Its pullback lowers `conjugate`
        // together with `real` and `complex`.
        let gradient =
            differentiate_at(input).gradient(|z| (z.clone() * z.conjugate().unwrap()).real().unwrap()).unwrap();
        assert_eq!(values_from_bytes::<Complex<f32>>(&gradient.copy_to_host().unwrap()), [(z + z).conj()]);
    }

    #[test]
    fn test_real_operation_lower() {
        assert_eq!(
            lowered_unary_module(ArrayOperation::Real(RealOperation::new()), DataType::C64, vec![2]).unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
                    %0 = stablehlo.real %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_imaginary_operation_lower() {
        assert_eq!(
            lowered_unary_module(ArrayOperation::Imaginary(ImaginaryOperation::new()), DataType::C64, vec![2]).unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
                    %0 = stablehlo.imag %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "},
        );
    }
}
