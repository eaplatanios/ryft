use ryft_core::macros::check_count;
use ryft_core::{ArrayIrType, ArrayType, CompareOperation, ComparisonDirection, DataType, ProgramError};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, FloatTypeRef, IntegerTypeRef, Location, LocationRef, Operation as MlirOperation, TensorTypeRef,
    Type, TypeRef, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::types::lower_signed_one_bit_to_mlir;
use crate::operations::normalize_binary_elementwise_operands;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, ToMlir, XlaLowerer};

/// Returns the promoted numeric operand descriptor for one comparison result descriptor.
pub(crate) fn comparison_operand_type(
    input_types: &[ArrayType],
    output_types: &[ArrayType],
) -> Result<ArrayType, LoweringError> {
    check_count!("input", input_types, 2, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let data_type = DataType::promoted(&[input_types[0].data_type(), input_types[1].data_type()]).map_err(|error| {
        LoweringError::UnsupportedOp { op: format!("cannot normalize comparison operands: {error}") }
    })?;
    Ok(output_types[0].clone().with_data_type(data_type))
}

impl ToMlir for ComparisonDirection {
    type Output<'c, 't: 'c> = stable_hlo::ComparisonDirection;

    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(&self, _location: L) -> Result<Self::Output<'c, 't>, ryft_mlir::Error> {
        Ok(match self {
            ComparisonDirection::Equal => stable_hlo::ComparisonDirection::Equal,
            ComparisonDirection::NotEqual => stable_hlo::ComparisonDirection::NotEqual,
            ComparisonDirection::LessThan => stable_hlo::ComparisonDirection::LessThan,
            ComparisonDirection::LessThanOrEqual => stable_hlo::ComparisonDirection::LessThanOrEqual,
            ComparisonDirection::GreaterThan => stable_hlo::ComparisonDirection::GreaterThan,
            ComparisonDirection::GreaterThanOrEqual => stable_hlo::ComparisonDirection::GreaterThanOrEqual,
        })
    }
}

/// Lowers an [`ArrayOperation::Compare`](ryft_core::ArrayOperation::Compare)-style dispatch to
/// `stablehlo.compare`. The resulting value has the broadcasted shape of the inputs and Boolean
/// element type. The comparison semantic is routed based on the LHS value's element type
/// (Float / Signed / Unsigned).
pub(crate) fn lower_compare_to_mlir<'b, 'c: 'b, 't: 'c>(
    direction: ComparisonDirection,
    lhs: ValueRef<'b, 'c, 't>,
    rhs: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let direction = direction.to_mlir(location)?;
    let lhs_type = lhs.r#type()?;
    let comparison_type = comparison_type_for_mlir_type(lhs_type)?;
    let result = block.append_operation(stable_hlo::compare(lhs, rhs, direction, comparison_type, location)?)?;
    Ok(result.result(0).expect("stablehlo.compare should return one result").as_ref())
}

/// Picks the right StableHLO comparison semantic based on the element type of an MLIR value.
///
/// Tensor values are unwrapped to their element type; non-tensor scalar types are inspected
/// directly. Float-family types route to [`stable_hlo::ComparisonType::Float`]; explicitly
/// unsigned integers and one-bit predicate carriers route to [`stable_hlo::ComparisonType::Unsigned`]; other
/// integers route to [`stable_hlo::ComparisonType::Signed`]. Logical signed one-bit inputs must be widened before
/// calling this function because their MLIR carrier alone cannot distinguish them from Boolean inputs.
fn comparison_type_for_mlir_type<'c, 't>(r#type: TypeRef<'c, 't>) -> Result<stable_hlo::ComparisonType, LoweringError> {
    let element_type = if let Some(tensor) = r#type.cast::<TensorTypeRef>() {
        tensor.element_type().map_err(|error| LoweringError::MlirError(error))?
    } else {
        r#type
    };
    if element_type.is::<FloatTypeRef>() {
        return Ok(stable_hlo::ComparisonType::Float);
    }
    if let Some(integer) = element_type.cast::<IntegerTypeRef>() {
        if integer.is_unsigned() || integer.bit_width() == 1 {
            return Ok(stable_hlo::ComparisonType::Unsigned);
        }
        return Ok(stable_hlo::ComparisonType::Signed);
    }
    // Default: treat as float for unknown element types (matches StableHLO's lenient handling
    // of non-integer non-float numeric types like complex).
    Ok(stable_hlo::ComparisonType::Float)
}

impl LowerableOperation for CompareOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let mut operand_type = comparison_operand_type(input_types, output_types)?;
        // StableHLO treats i1 as a predicate, even when its logical dtype is signed. Recover each
        // signed input before promotion, and compare two I1 inputs in the same signed byte carrier.
        let mut inputs = [input_values[0], input_values[1]];
        for (input, input_type) in inputs.iter_mut().zip(input_types) {
            if input_type.data_type() == DataType::I1 {
                *input = lower_signed_one_bit_to_mlir(*input, &mut lowerer.block, lowerer.context, lowerer.location)?;
            }
        }
        if operand_type.data_type() == DataType::I1 {
            operand_type = operand_type.with_data_type(DataType::I8);
        }
        let [left, right] = normalize_binary_elementwise_operands(
            &inputs,
            &[operand_type],
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let value = lower_compare_to_mlir(self.direction(), left, right, &mut lowerer.block, lowerer.location)?;
        Ok(vec![value])
    }
}

impl LowerableOperation for CompareOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let location = lowerer.location;
        let block = &mut lowerer.block;
        let [left, right] = input_values else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() }.into());
        };
        // Dimension atoms already lower to scalar `i64` SSA values, so the ordinary comparison lowering can
        // consume them directly without a data gateway or host-side shape reconstruction.
        Ok(vec![lower_compare_to_mlir(self.direction(), *left, *right, block, location)?])
    }
}
