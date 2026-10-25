use ryft_core::macros::check_count;
use ryft_core::{ArrayType, Dimension};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::Accuracy;
use ryft_mlir::{
    Block, Context as MlirContext, Location, Operation as MlirOperation, TensorTypeRef, Type, Value as MlirValue,
    ValueRef,
};

use crate::LoweringError;
use crate::experimental::lowering::types::{lower_restore_dynamic_dimensions, lower_tensor_type, physical_bound_type};
use crate::operations::constants::lower_layout_constraint;

/// StableHLO lowering of arithmetic operations.
pub(crate) mod arithmetic;
/// Runtime assertions: their StableHLO lowering to the `ryft.assert` XLA FFI custom call and the host-side handler that
/// reports failed assertions.
pub(crate) mod assertions;
/// StableHLO lowering of dot-product attention operations.
pub(crate) mod attention;
/// StableHLO lowering of collective operations.
pub(crate) mod collectives;
/// StableHLO lowering of comparison operations.
pub(crate) mod comparisons;
/// StableHLO lowering of complex-number operations.
pub(crate) mod complex;
/// StableHLO lowering of Ryft composites: composite calls inline only their primal program, and composite transposes
/// are rejected because transposition must replace them before lowering.
pub(crate) mod composites;
/// StableHLO lowering of constant and constructor operations (e.g., `zero`, `one`, `iota`, and literal constants).
///
/// Eager evaluation and staging allow unbounded dimensions, but executable zero, one, iota, and like constructors need
/// finite allocation bounds. Their dynamic extents must fit signed 32-bit size carriers: a bound reserves storage,
/// while the explicit dimension inputs or exemplar sizes determine the runtime shape. Explicit constructor layouts
/// support dense minor-to-major axis permutations; byte-strided layouts and physical tiles are rejected.
pub(crate) mod constants;
/// StableHLO lowering of control flow operations.
pub(crate) mod control_flow;
/// StableHLO lowering of cumulative (prefix-scan) operations.
pub(crate) mod cumulative;
/// StableHLO lowering of custom calls.
pub(crate) mod custom_call;
/// Host-callback debugging support: the StableHLO lowering of `print` to the `ryft.print` XLA FFI custom call, its
/// handler, and its capturable print sink.
pub mod debugging;
/// StableHLO lowering of differentiation-specific operations (e.g., `rematerialize` and `linear_call`).
pub(crate) mod differentiation;
/// StableHLO lowering of first-class dimension operations.
pub(crate) mod dimensions;
/// StableHLO lowering of generalized dot products, including grouped (ragged) dots.
pub mod dot;
/// StableHLO lowering of exponential and logarithmic operations.
pub(crate) mod exponential;
/// StableHLO lowering of extrema operations (`min`, `max`, and `clamp`).
pub(crate) mod extrema;
/// Backend-owned `jit_call` operation: lowering of staged calls to jitted XLA programs, including call-site
/// deduplication.
pub mod jit_call;
/// StableHLO lowering of logical operations.
pub(crate) mod logical;
/// StableHLO lowering of array manipulation operations.
pub(crate) mod manipulation;
/// StableHLO lowering of quantized operations such as scaled dot products.
pub(crate) mod quantization;
/// StableHLO lowering of random number generation operations.
pub(crate) mod random;
/// StableHLO lowering of reduction operations.
pub(crate) mod reductions;
/// Rejection of unresolved reference operations, which reference discharge must remove before StableHLO lowering.
pub(crate) mod references;
/// StableHLO lowering of rounding operations.
pub(crate) mod rounding;
/// StableHLO and Shardy lowering of sharding operations.
pub mod sharding;
/// StableHLO lowering of sorting operations.
pub(crate) mod sorting;
/// StableHLO lowering of special functions.
pub(crate) mod special;
/// StableHLO lowering of value tagging.
pub(crate) mod tagging;
/// StableHLO lowering of trigonometric and hyperbolic operations.
pub(crate) mod trigonometric;

/// Converts and broadcasts one implicitly compatible elementwise operand to the exact StableHLO result tensor type.
pub(crate) fn normalize_elementwise_operand<'b, 'c: 'b, 't: 'c, B, L>(
    input: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let input_type = input.r#type()?;
    let input_tensor_type = input_type.cast::<TensorTypeRef>().ok_or_else(|| LoweringError::UnsupportedOp {
        op: format!("elementwise operand has non-tensor MLIR type `{input_type}`"),
    })?;
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    let output_element_type = output_tensor_type.element_type()?;
    let mut input = input;
    if input_tensor_type.element_type()? != output_element_type {
        let dimensions = input_tensor_type.dimensions().collect::<Vec<_>>();
        let converted_type = context.tensor_type(output_element_type, dimensions.as_slice(), None, location)?;
        let converted = block.append_operation(stable_hlo::convert(input, converted_type, location)?)?;
        input = converted.result(0).expect("stablehlo.convert should return one result").as_ref();
    }
    if input_tensor_type.dimensions().ne(output_tensor_type.dimensions()) {
        let output_rank = output_tensor_type.rank();
        let input_rank = input_tensor_type.rank();
        let first_dimension = output_rank.checked_sub(input_rank).ok_or_else(|| LoweringError::UnsupportedOp {
            op: format!("cannot broadcast rank-{input_rank} elementwise operand to rank-{output_rank} output"),
        })?;
        let dimensions = (first_dimension..output_rank).collect::<Vec<_>>();
        let broadcast = block.append_operation(stable_hlo::broadcast(
            input,
            output_tensor_type,
            dimensions.as_slice(),
            location,
        )?)?;
        input = broadcast.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref();
    }
    Ok(input)
}

/// Returns the StableHLO result accuracy that requests the provided [`ryft_core::Accuracy`] from the backend.
pub(crate) fn lower_accuracy(accuracy: ryft_core::Accuracy) -> Accuracy {
    match accuracy {
        ryft_core::Accuracy::Default => Accuracy::Default,
        ryft_core::Accuracy::Highest => Accuracy::Highest,
        ryft_core::Accuracy::Tolerance(tolerance) => Accuracy::Tolerance {
            absolute_tolerance: tolerance.absolute(),
            relative_tolerance: tolerance.relative(),
            units_of_least_precision: tolerance.units_of_least_precision(),
        },
    }
}

/// Normalizes both operands of one implicitly broadcasting binary elementwise operation to its exact result type.
pub(crate) fn normalize_binary_elementwise_operands<'b, 'c: 'b, 't: 'c, B, L>(
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<[ValueRef<'b, 'c, 't>; 2], LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    check_count!("input", input_values, 2, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let output_type = &output_types[0];
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    let multiple_dynamic_axes = output_type
        .shape()
        .dimensions()
        .iter()
        .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        .count()
        > 1;
    let mut inputs = [input_values[0], input_values[1]];
    for index in 0..2 {
        let input_tensor_type = input_values[index].r#type()?.cast::<TensorTypeRef>();
        let peer_tensor_type = input_values[1 - index].r#type()?.cast::<TensorTypeRef>();
        if multiple_dynamic_axes
            && input_tensor_type.is_some_and(|r#type| r#type.rank() == 0)
            && peer_tensor_type.is_some_and(|r#type| r#type.dimensions().eq(output_tensor_type.dimensions()))
        {
            // Native broadcast accepts at most one bounded dynamic result axis. For scalar arithmetic over
            // e.g. `[rows, columns]`, broadcast to physical capacity and restore both logical extents from the
            // array input. The scalar itself cannot supply these sizes, and inventing the maximum sizes would
            // incorrectly make padding observable. Other broadcasting cases use the ordinary path below.
            let scalar = normalize_elementwise_operand(
                input_values[index],
                &ArrayType::scalar(output_type.data_type()),
                block,
                context,
                location,
            )?;
            let physical_type = physical_bound_type(output_type)?;
            let broadcast = block.append_operation(stable_hlo::broadcast(
                scalar,
                lower_tensor_type(&physical_type, context, location)?,
                &[],
                location,
            )?)?;
            let result = lower_layout_constraint(
                broadcast.result(0).unwrap().as_ref(),
                &physical_type,
                &mut block.as_ref(),
                location.as_ref(),
            )?;
            let sources = (0..output_type.rank()).map(|axis| (input_values[1 - index], axis)).collect::<Vec<_>>();
            inputs[index] = lower_restore_dynamic_dimensions(
                result,
                output_type,
                &sources,
                &mut block.as_ref(),
                context,
                location.as_ref(),
            )?;
        } else {
            inputs[index] = normalize_elementwise_operand(input_values[index], output_type, block, context, location)?;
        }
    }
    Ok(inputs)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AndOperation, ArrayOperation, Atan2Operation, CompareOperation, ComparisonDirection, DataType, DivOperation,
        OrOperation, Placeholder, SelectOperation, Shape, SinOperation, XorOperation,
    };

    use crate::experimental::lowering::tests::{PlainXlaProgram, XlaProgramBuilder, lowered_unary_module};
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::{XlaArrayConstant, XlaConstant, XlaOperation};
    use crate::to_mlir_module_for_program;

    use super::*;

    fn xla_elementwise_normalization_program() -> PlainXlaProgram {
        let mut builder = XlaProgramBuilder::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::F32));
        let left = builder
            .add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(1)])));
        let right = builder
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(4)])));
        let condition = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let boolean_vector =
            builder.add_input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)])));
        let divide = builder.add_instruction(DivOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let atan2 = builder.add_instruction(Atan2Operation::new(), Vec::new(), vec![right, left], None).unwrap()[0];
        let compare = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![left, right],
                None,
            )
            .unwrap()[0];
        let select = builder
            .add_instruction(SelectOperation::new(), Vec::new(), vec![condition, scalar, right], None)
            .unwrap()[0];
        let and = builder
            .add_instruction(AndOperation::new(), Vec::new(), vec![condition, boolean_vector], None)
            .unwrap()[0];
        let or = builder
            .add_instruction(OrOperation::new(), Vec::new(), vec![boolean_vector, condition], None)
            .unwrap()[0];
        let xor = builder
            .add_instruction(XorOperation::new(), Vec::new(), vec![condition, boolean_vector], None)
            .unwrap()[0];
        builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![divide, atan2, compare, select, and, or, xor],
                vec![Placeholder; 5],
                vec![Placeholder; 7],
            )
            .unwrap()
    }

    fn assert_elementwise_operands_are_normalized(stablehlo: &str) {
        for operation in ["divide", "atan2", "compare", "select", "and", "or", "xor"] {
            assert!(stablehlo.contains(&format!("stablehlo.{operation}")), "{stablehlo}");
        }
        assert!(stablehlo.contains("tensor<3x4xf64>"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<3x4xi1>"), "{stablehlo}");
        assert!(stablehlo.matches("stablehlo.convert").count() >= 4, "{stablehlo}");
        assert!(stablehlo.matches("stablehlo.broadcast_in_dim").count() >= 8, "{stablehlo}");
    }

    #[test]
    fn test_plain_elementwise_lowering_normalizes_all_implicit_operands() {
        let program = xla_elementwise_normalization_program();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_elementwise_operands_are_normalized(&stablehlo);
    }

    #[test]
    fn test_traced_elementwise_lowering_normalizes_all_implicit_operands() {
        let plain_program = xla_elementwise_normalization_program();
        let input_types = plain_program.input_types();
        let output_types = plain_program.output_types();
        let program = plain_program.into_unprojected::<XlaConstant, XlaOperation>().unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_elementwise_operands_are_normalized(&stablehlo);
    }

    #[test]
    fn test_elementwise_lowering_normalizes_zero_sized_operands() {
        let mut builder = XlaProgramBuilder::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::F32));
        let empty = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)])));
        let output = builder.add_instruction(DivOperation::new(), Vec::new(), vec![scalar, empty], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert!(stablehlo.contains("stablehlo.convert"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.broadcast_in_dim"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.divide"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<0xf64>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_result_accuracies() {
        // The default accuracy keeps the default StableHLO result accuracy, while other accuracies request theirs.
        assert_eq!(
            lowered_unary_module(ArrayOperation::Sin(SinOperation::new()), DataType::F32, vec![2]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.sine %arg0 : tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "#},
        );
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Tanh(ryft_core::TanhOperation::new().with_accuracy(ryft_core::Accuracy::Highest)),
                DataType::F32,
                vec![2],
            )
            .unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.tanh %arg0 {result_accuracy = #stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "#},
        );
        let tolerance = ryft_core::Tolerance::new(1e-6, 0.0, 2).unwrap();
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Exp(
                    ryft_core::ExpOperation::new().with_accuracy(ryft_core::Accuracy::Tolerance(tolerance)),
                ),
                DataType::F32,
                vec![2],
            )
            .unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.exponential %arg0 {result_accuracy = #stablehlo.result_accuracy<atol = 9.9999999999999995E-7, ulps = 2, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "#},
        );
    }
}
