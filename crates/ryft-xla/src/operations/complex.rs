use ryft_core::{ArrayType, ComplexOperation, ConjugateOperation, DataType, ImaginaryOperation, RealOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::Accuracy;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, TensorTypeRef,
    Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::types::{lower_f64_constant_splat, lower_tensor_type};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for ComplexOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result =
            lowerer
                .block
                .append_operation(stable_hlo::complex(input_values[0], input_values[1], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.complex should return one result").as_ref()])
    }
}

// StableHLO has no conjugation operation, so `conjugate` lowers to the `complex(real(z), negate(imag(z)))`
// composition (the same decomposition JAX's `conj` lowering uses).
impl LowerableOperation for ConjugateOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let real = lowerer.block.append_operation(stable_hlo::real(input_values[0], lowerer.location)?)?;
        let imaginary = lowerer.block.append_operation(stable_hlo::imag(input_values[0], lowerer.location)?)?;
        let negated = lowerer.block.append_operation(stable_hlo::negate(
            imaginary.result(0).expect("stablehlo.imag should return one result").as_ref(),
            lowerer.location,
        )?)?;
        let result = lowerer.block.append_operation(stable_hlo::complex(
            real.result(0).expect("stablehlo.real should return one result").as_ref(),
            negated.result(0).expect("stablehlo.negate should return one result").as_ref(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.complex should return one result").as_ref()])
    }
}

impl LowerableOperation for RealOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::real(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.real should return one result").as_ref()])
    }
}

impl LowerableOperation for ImaginaryOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::imag(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.imag should return one result").as_ref()])
    }
}

/// Brings the imaginary component of the complex `value` onto the principal branch `[-pi, pi]`, which is where the
/// raw-input prefixes of a complex log-sum-exp `cumulative` must land to agree with its combined prefixes. Phases that
/// already lie on the branch are kept exactly as they are, and others are reduced through `atan2` of their sine and
/// cosine rather than through a floating-point remainder by `2π`, which cannot reduce huge arguments accurately.
pub(crate) fn lower_principal_phase_to_mlir<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
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
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
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
