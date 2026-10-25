//! Lowering of [`ConditionOperation`] to [StableHLO `if`](https://openxla.org/stablehlo/spec#if).
//!
//! A condition lowers only from the [`ArrayIrType`] operation family, which stores its two branch regions; the
//! homogeneous [`ArrayType`] payload carries no regions and is rejected. The predicate selects between two
//! `stablehlo.if` regions that take no arguments and capture the lowered branch inputs implicitly. Each branch input is
//! converted from its actual type to the shared declared type of its branch input, and an entry guard checks the input
//! relationships that the actual input types leave unproven before either branch runs. Every ordered effect class that
//! either branch uses is threaded through both branches as one trailing token result, so both branches return the same
//! result types, and both branches normalize their results to the refined output types of the condition. Results whose
//! types place them over explicit mesh axes are constrained to that placement.

use ryft_core::{ArrayIrType, ArrayType, ConditionOperation, Operation, ProgramError};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::experimental::lowering::{current_or_new_token, token_threaded_effects};
use crate::operations::control_flow::{
    lower_control_flow_region, lower_explicit_sharding_constraints, lower_refined_region_input, lower_region_guard,
    lowered_effect_classes, set_threaded_effect_tokens,
};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for ConditionOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "higher-order operation `{}` must be stored directly in the enclosing backend operation family",
                self.name(),
            ),
        }
        .into())
    }
}

impl LowerableOperation for ConditionOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        // A condition lowers to one `stablehlo.if` on its predicate, which is the first of its `1 + n` inputs, followed
        // by the `n` inputs of both branches. The branch regions take no arguments: each one is inlined over the
        // lowered branch inputs, which it captures implicitly, and returns the condition's `output_types` followed by
        // one trailing token for each ordered effect class that either branch threads. The lowering proceeds in order:
        //
        //   1. Each branch input is converted from its actual type to the shared declared type of its branch input.
        //   2. An entry guard checks the input relationships that the actual input types leave unproven, before the
        //      branch tokens are captured, so that the effects of either branch are ordered after the check.
        //   3. Each branch is lowered with the entry tokens of the threaded effect classes and normalizes its results
        //      to the condition's refined `output_types`, so both branches return identical result types.
        //   4. The trailing token results of the `stablehlo.if` become the current tokens of the enclosing scope, and
        //      each result whose type places it over an explicit mesh axis is constrained to that placement.
        //
        // The program that attaches the condition validated its region and input counts, so a mismatch here is an
        // internal error, reported as the operation's own region contract and input count would report it.
        self.validate_region_count(lowerer.regions.len())?;
        let (true_branch, false_branch) = (&lowerer.regions[0], &lowerer.regions[1]);
        let branch_input_types = true_branch.input_types();
        let expected = branch_input_types.len() + 1;
        for actual in [input_values.len(), input_types.len()] {
            if actual != expected {
                return Err(ProgramError::InvalidInputCount { expected, actual }.into());
            }
        }
        // Branch inputs only need to refine the branch input types, so each one is converted to the lowered type of its
        // branch input first.
        let branch_inputs = input_values[1..]
            .iter()
            .zip(&input_types[1..])
            .zip(branch_input_types.iter())
            .map(|((value, input_type), branch_input_type)| {
                lower_refined_region_input(
                    *value,
                    input_type,
                    branch_input_type,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Both branches declare the same inputs, so guarding their shared entry before the conditional checks exactly
        // what eager entry validation checks for the selected branch. The guard runs before the branch tokens are
        // captured.
        lower_region_guard(
            self.name(),
            &branch_input_types,
            &input_types[1..],
            &input_values[1..],
            &mut lowerer.effect_tokens,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        // Each ordered class used by either branch is captured and returned independently. Both branches carry the
        // union so their result signatures agree, returning an entry token unchanged when that branch is pure for the
        // class.
        let threaded_effects = lowered_effect_classes(true_branch)?.union(lowered_effect_classes(false_branch)?);
        let mut entry_effect_tokens = lowerer.effect_tokens;
        for effect in token_threaded_effects(threaded_effects) {
            let token = current_or_new_token(effect, &mut lowerer.effect_tokens, &mut lowerer.block, lowerer.location)?;
            entry_effect_tokens.set(effect, token);
        }
        // The declared branch outputs may differ (e.g., `f64[extent]` and `f64[3]`) while agreeing after input
        // refinement, so each branch normalizes its results to the condition's output types.
        let [true_branch_region, false_branch_region] = [true_branch, false_branch].map(|branch| {
            lower_control_flow_region(
                branch,
                &branch_inputs,
                output_types,
                lowerer.context,
                lowerer.location,
                &lowerer.captured_values,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                entry_effect_tokens,
                threaded_effects,
            )
        });
        let operation = lowerer.block.append_operation(stable_hlo::r#if(
            input_values[0],
            true_branch_region?.into(),
            false_branch_region?.into(),
            lowerer.location,
        )?)?;
        let output_count = output_types.len();
        set_threaded_effect_tokens(&mut lowerer.effect_tokens, threaded_effects, |offset| {
            operation.result(output_count + offset).unwrap().as_ref()
        });
        let outputs = (0..output_count).map(|index| operation.result(index).unwrap().as_ref()).collect();
        lower_explicit_sharding_constraints(
            outputs,
            output_types,
            &lowerer.collective_state.bound_manual_axes,
            &mut lowerer.block,
            lowerer.location,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayOperation, ArrayType, AssertOperation, BatchAxis, BatchedProgram, CompareOperation,
        ComparisonDirection, CompositeTransposeOperation, ConditionOperation, DataType, Dimension, DimensionBounds,
        DimensionSizeOperation, DimensionType, DimensionVariable, DynamicBroadcastOperation, GuardOperation,
        MeshAxisType, MulOperation, NegOperation, OneLikeOperation, OneOperation, Placeholder, PrintOperation,
        ProgramBatchingOutputAxesPolicy, ReduceOperation, ReductionKind, ScanOperation, Shape, Sharding,
        ShardingDimension, SubOperation, TransposedProgramCotangent, WhileOperation, ZeroOperation,
    };
    use ryft_mlir::{Context as MlirContext, Location};
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::experimental::lowering::tests::{
        MixedValue, dynamic_dimension, execute_mixed_program, ragged_dot_cpu_compilation_options,
        unproject_plain_program, xla_identity_branch, xla_neg_branch,
    };
    use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder as ArrayIrXlaProgramBuilder};
    use crate::operations::assertions::ensure_assertion_handler_registered;
    use crate::operations::control_flow::tests::{
        compare, execute_on_two_devices, lowered_mixed_module, positive, two_device_mesh,
    };
    use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};
    use crate::tests::execution_client;
    use crate::to_mlir_module_for_program;

    use super::*;

    #[test]
    fn test_condition_lowering() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F32);
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_program(unproject_plain_program(xla_neg_branch(input_type.clone())));
        let false_region = builder.import_program(unproject_plain_program(xla_identity_branch(input_type.clone())));
        let predicate = builder.add_input(predicate_type.clone().into());
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &(predicate_type, input_type.clone()),
            &input_type,
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32> {
                    %0 = "stablehlo.if"(%arg0) ({
                      %1 = stablehlo.negate %arg1 : tensor<f32>
                      stablehlo.return %1 : tensor<f32>
                    }, {
                      stablehlo.return %arg1 : tensor<f32>
                    }) : (tensor<i1>) -> tensor<f32>
                    return %0 : tensor<f32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_condition_threads_the_io_dependency_of_a_guard_out_of_its_branches() {
        // Each branch's only effect is an entry guard, which joins its assertion token into the ordered I/O chain so
        // that later I/O follows the check. The conditional must thread that chain out of its branches, so the print
        // after it consumes an I/O token that the conditional returns.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new_static(DataType::F64, [2]);
        let branch = || {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let dimension = builder.add_input(m.clone().into());
            let array = builder.add_input(array_type.clone().into());
            let declared: Vec<ArrayIrType> = vec![
                n.clone().into(),
                ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()])).into(),
            ];
            builder
                .add_instruction(GuardOperation::new("branch", declared), Vec::new(), vec![dimension, array], None)
                .unwrap();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![array], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_region(branch().entry_region_ref());
        let false_region = builder.import_region(branch().entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let dimension = builder.add_input(m.clone().into());
        let array = builder.add_input(array_type.clone().into());
        let selected = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, dimension, array],
                None,
            )
            .unwrap()[0];
        let printed = builder
            .add_instruction(PrintOperation::<ArrayType>::new("value"), Vec::new(), vec![selected], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![printed], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::I64), array_type.clone()],
            &[array_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<2xf64>) -> tensor<2xf64> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2:3 = "stablehlo.if"(%arg0) ({
                      %c = stablehlo.constant dense<2> : tensor<i64>
                      %4 = stablehlo.compare EQ, %c, %arg1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %5 = stablehlo.custom_call @ryft.assert(%4, %arg1, %c, %0) {api_version = 4 : i32, backend_config = {actor = "guard", kind = "generic", label_0 = "expected", label_1 = "actual", label_count = "2", message = "input dimension `n` must have one extent at the entry of `branch`"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                      %6 = stablehlo.after_all %1, %5 : !stablehlo.token
                      %7:3 = stablehlo.optimization_barrier %arg1, %arg2, %5 : tensor<i64>, tensor<2xf64>, !stablehlo.token
                      stablehlo.return %arg2, %5, %6 : tensor<2xf64>, !stablehlo.token, !stablehlo.token
                    }, {
                      %c = stablehlo.constant dense<2> : tensor<i64>
                      %4 = stablehlo.compare EQ, %c, %arg1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %5 = stablehlo.custom_call @ryft.assert(%4, %arg1, %c, %0) {api_version = 4 : i32, backend_config = {actor = "guard", kind = "generic", label_0 = "expected", label_1 = "actual", label_count = "2", message = "input dimension `n` must have one extent at the entry of `branch`"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                      %6 = stablehlo.after_all %1, %5 : !stablehlo.token
                      %7:3 = stablehlo.optimization_barrier %arg1, %arg2, %5 : tensor<i64>, tensor<2xf64>, !stablehlo.token
                      stablehlo.return %arg2, %5, %6 : tensor<2xf64>, !stablehlo.token, !stablehlo.token
                    }) : (tensor<i1>) -> (tensor<2xf64>, !stablehlo.token, !stablehlo.token)
                    %3 = stablehlo.custom_call @ryft.print(%2#0, %2#2) {api_version = 4 : i32, backend_config = {label = "value"}, has_side_effect = true} : (tensor<2xf64>, !stablehlo.token) -> !stablehlo.token
                    return %2#0 : tensor<2xf64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_condition_threads_the_io_dependency_of_guards_that_lowering_synthesizes() {
        // The outer branches are pure in the source program, but each contains an inner conditional whose mixed entry
        // (a symbolic `dimension<m>` and an exact `f64[2]` entering `[dimension<n>, f64[n]]`) gains an entry guard
        // during lowering. The outer conditional must thread that guard's I/O dependency out of its branches too, so
        // the print after it consumes an I/O token that the outer conditional returns.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new_static(DataType::F64, [2]);
        let inner_branch = || {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(n.clone().into());
            let array =
                builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()])).into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![array], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let outer_branch = || {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let true_region = builder.import_region(inner_branch().entry_region_ref());
            let false_region = builder.import_region(inner_branch().entry_region_ref());
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
            let dimension = builder.add_input(m.clone().into());
            let array = builder.add_input(array_type.clone().into());
            let selected = builder
                .add_instruction(
                    XlaOperation::Condition(ConditionOperation::new()),
                    vec![true_region, false_region],
                    vec![predicate, dimension, array],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![selected], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        assert!(outer_branch().effects().classes().is_empty());
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_region(outer_branch().entry_region_ref());
        let false_region = builder.import_region(outer_branch().entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let dimension = builder.add_input(m.clone().into());
        let array = builder.add_input(array_type.clone().into());
        let selected = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, predicate, dimension, array],
                None,
            )
            .unwrap()[0];
        let printed = builder
            .add_instruction(PrintOperation::<ArrayType>::new("after"), Vec::new(), vec![selected], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![printed], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::I64), array_type.clone()],
            &[array_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<2xf64>) -> tensor<2xf64> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2:3 = "stablehlo.if"(%arg0) ({
                      %c = stablehlo.constant dense<2> : tensor<i32>
                      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                      %4 = stablehlo.pad %arg2, %cst, low = [0], high = [5], interior = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<7xf64>
                      %5 = stablehlo.set_dimension_size %4, %c, dim = 0 : (tensor<7xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<7>>
                      %c_0 = stablehlo.constant dense<2> : tensor<i64>
                      %6 = stablehlo.compare EQ, %c_0, %arg1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %7 = stablehlo.custom_call @ryft.assert(%6, %arg1, %c_0, %0) {api_version = 4 : i32, backend_config = {actor = "guard", kind = "generic", label_0 = "expected", label_1 = "actual", label_count = "2", message = "input dimension `n` must have one extent at the entry of `condition`"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                      %8 = stablehlo.after_all %1, %7 : !stablehlo.token
                      %9 = "stablehlo.if"(%arg0) ({
                        %c_1 = stablehlo.constant dense<7> : tensor<i32>
                        %10 = stablehlo.set_dimension_size %5, %c_1, dim = 0 : (tensor<?xf64, #stablehlo.bounds<7>>, tensor<i32>) -> tensor<7xf64>
                        %11 = stablehlo.slice %10 [0:2] : (tensor<7xf64>) -> tensor<2xf64>
                        stablehlo.return %11 : tensor<2xf64>
                      }, {
                        %c_1 = stablehlo.constant dense<7> : tensor<i32>
                        %10 = stablehlo.set_dimension_size %5, %c_1, dim = 0 : (tensor<?xf64, #stablehlo.bounds<7>>, tensor<i32>) -> tensor<7xf64>
                        %11 = stablehlo.slice %10 [0:2] : (tensor<7xf64>) -> tensor<2xf64>
                        stablehlo.return %11 : tensor<2xf64>
                      }) : (tensor<i1>) -> tensor<2xf64>
                      stablehlo.return %9, %7, %8 : tensor<2xf64>, !stablehlo.token, !stablehlo.token
                    }, {
                      %c = stablehlo.constant dense<2> : tensor<i32>
                      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                      %4 = stablehlo.pad %arg2, %cst, low = [0], high = [5], interior = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<7xf64>
                      %5 = stablehlo.set_dimension_size %4, %c, dim = 0 : (tensor<7xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<7>>
                      %c_0 = stablehlo.constant dense<2> : tensor<i64>
                      %6 = stablehlo.compare EQ, %c_0, %arg1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %7 = stablehlo.custom_call @ryft.assert(%6, %arg1, %c_0, %0) {api_version = 4 : i32, backend_config = {actor = "guard", kind = "generic", label_0 = "expected", label_1 = "actual", label_count = "2", message = "input dimension `n` must have one extent at the entry of `condition`"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                      %8 = stablehlo.after_all %1, %7 : !stablehlo.token
                      %9 = "stablehlo.if"(%arg0) ({
                        %c_1 = stablehlo.constant dense<7> : tensor<i32>
                        %10 = stablehlo.set_dimension_size %5, %c_1, dim = 0 : (tensor<?xf64, #stablehlo.bounds<7>>, tensor<i32>) -> tensor<7xf64>
                        %11 = stablehlo.slice %10 [0:2] : (tensor<7xf64>) -> tensor<2xf64>
                        stablehlo.return %11 : tensor<2xf64>
                      }, {
                        %c_1 = stablehlo.constant dense<7> : tensor<i32>
                        %10 = stablehlo.set_dimension_size %5, %c_1, dim = 0 : (tensor<?xf64, #stablehlo.bounds<7>>, tensor<i32>) -> tensor<7xf64>
                        %11 = stablehlo.slice %10 [0:2] : (tensor<7xf64>) -> tensor<2xf64>
                        stablehlo.return %11 : tensor<2xf64>
                      }) : (tensor<i1>) -> tensor<2xf64>
                      stablehlo.return %9, %7, %8 : tensor<2xf64>, !stablehlo.token, !stablehlo.token
                    }) : (tensor<i1>) -> (tensor<2xf64>, !stablehlo.token, !stablehlo.token)
                    %3 = stablehlo.custom_call @ryft.print(%2#0, %2#2) {api_version = 4 : i32, backend_config = {label = "after"}, has_side_effect = true} : (tensor<2xf64>, !stablehlo.token) -> !stablehlo.token
                    return %2#0 : tensor<2xf64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_condition_lowering_keeps_constant_predicates() {
        // A condition whose predicate input is fed by a staged constant still lowers to `stablehlo.if`; folding the
        // constant predicate away is the backend's job (StableHLO canonicalization and XLA's conditional
        // simplification), not ryft's.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F32);
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_program(unproject_plain_program(xla_neg_branch(input_type.clone())));
        let false_region = builder.import_program(unproject_plain_program(xla_identity_branch(input_type.clone())));
        let input = builder.add_input(input_type.clone().into());
        let predicate =
            builder.add_instruction(OneOperation::new(predicate_type), Vec::new(), vec![], None).unwrap()[0];
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &input_type, &input_type, "main", None, None).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
                    %c = stablehlo.constant dense<true> : tensor<i1>
                    %0 = "stablehlo.if"(%c) ({
                      %1 = stablehlo.negate %arg0 : tensor<f32>
                      stablehlo.return %1 : tensor<f32>
                    }, {
                      stablehlo.return %arg0 : tensor<f32>
                    }) : (tensor<i1>) -> tensor<f32>
                    return %0 : tensor<f32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_condition_lowering_forwards_dimension_extents() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::from(extent.clone());
        let dynamic_vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let scalar_type = ArrayType::scalar(DataType::F32);
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let output_type = dynamic_vector_type.clone();

        let branch = |negate: bool| {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let branch_extent = builder.add_input(extent_type.clone().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let scalar = if negate {
                builder.add_instruction(NegOperation::new(), Vec::new(), vec![scalar], None).unwrap()[0]
            } else {
                scalar
            };
            let output = builder
                .add_instruction(
                    DynamicBroadcastOperation::new(Vec::new()),
                    Vec::new(),
                    vec![scalar, branch_extent],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };

        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_region(branch(false).entry_region_ref());
        let false_region = builder.import_region(branch(true).entry_region_ref());
        let vector = builder.add_input(dynamic_vector_type.clone().into());
        let predicate = builder.add_input(predicate_type.clone().into());
        let scalar = builder.add_input(scalar_type.clone().into());
        let extent = builder
            .add_instruction(
                DimensionSizeOperation::new(&dynamic_vector_type, 0).unwrap(),
                Vec::new(),
                vec![vector],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, extent, scalar],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![dynamic_vector_type, predicate_type, scalar_type],
            &vec![output_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<i1>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<8>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg3, dim = 0 : (tensor<8xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %3 = "stablehlo.if"(%arg1) ({
                      %6 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<8xf32>
                      %7 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
                      %8 = stablehlo.set_dimension_size %6, %7, dim = 0 : (tensor<8xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                      stablehlo.return %8 : tensor<?xf32, #stablehlo.bounds<8>>
                    }, {
                      %6 = stablehlo.negate %arg2 : tensor<f32>
                      %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<8xf32>
                      %8 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
                      %9 = stablehlo.set_dimension_size %7, %8, dim = 0 : (tensor<8xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                      stablehlo.return %9 : tensor<?xf32, #stablehlo.bounds<8>>
                    }) : (tensor<i1>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %4 = stablehlo.get_dimension_size %3, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
                    return %3, %5 : tensor<?xf32, #stablehlo.bounds<8>>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_condition_lowering_returns_branch_effect_tokens() {
        // A print inside one condition branch makes the lowered `stablehlo.if` return the branch's final effect
        // token as one extra trailing result: both branches capture the entry token implicitly, the effectful branch
        // returns its print custom call's token, and the pure branch returns the entry token unchanged.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F64);
        let mut true_builder = ArrayIrXlaProgramBuilder::new();
        let true_input = true_builder.add_input(input_type.clone().into());
        let printed = true_builder
            .add_instruction(PrintOperation::<ArrayType>::new("taken"), Vec::new(), vec![true_input], None)
            .unwrap()[0];
        let negated = true_builder.add_instruction(NegOperation::new(), Vec::new(), vec![printed], None).unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![negated], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_program(unproject_plain_program(xla_identity_branch(input_type.clone())));
        let predicate = builder.add_input(predicate_type.clone().into());
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![predicate_type, input_type.clone()];
        let output_types = vec![input_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:2 = "stablehlo.if"(%arg0) ({
                      %2 = stablehlo.custom_call @ryft.print(%arg1, %0) {api_version = 4 : i32, backend_config = {label = "taken"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      %3 = stablehlo.negate %arg1 : tensor<f64>
                      stablehlo.return %3, %2 : tensor<f64>, !stablehlo.token
                    }, {
                      stablehlo.return %arg1, %0 : tensor<f64>, !stablehlo.token
                    }) : (tensor<i1>) -> (tensor<f64>, !stablehlo.token)
                    return %1#0 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_condition_lowering_threads_the_union_of_branch_effect_classes() {
        // Each branch uses a different ordered class. StableHLO requires identical branch result signatures, so both
        // branches return the union's two tokens while forwarding the untouched entry token for the unused class.
        let scalar_type = ArrayType::scalar(DataType::I64);
        let true_branch = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone().into());
            let predicate = builder
                .add_instruction(
                    XlaOperation::Array(ArrayOperation::Compare(CompareOperation::<ArrayType>::new(
                        ComparisonDirection::Equal,
                    ))),
                    Vec::new(),
                    vec![input, input],
                    None,
                )
                .unwrap()[0];
            builder
                .add_instruction(
                    AssertOperation::<ArrayIrType>::new("scalar must equal itself")
                        .with_labels(vec!["scalar".to_owned()]),
                    Vec::new(),
                    vec![predicate, input],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone().into());
            let output = builder
                .add_instruction(PrintOperation::<ArrayType>::new("false"), Vec::new(), vec![input], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![ArrayType::scalar(DataType::Boolean), scalar_type.clone()],
            &vec![scalar_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i64>) -> tensor<i64> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2:3 = "stablehlo.if"(%arg0) ({
                      %3 = stablehlo.compare EQ, %arg1, %arg1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %4 = stablehlo.custom_call @ryft.assert(%3, %arg1, %0) {api_version = 4 : i32, backend_config = {actor = "assert", kind = "generic", label_0 = "scalar", label_count = "1", message = "scalar must equal itself"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                      stablehlo.return %arg1, %4, %1 : tensor<i64>, !stablehlo.token, !stablehlo.token
                    }, {
                      %3 = stablehlo.custom_call @ryft.print(%arg1, %1) {api_version = 4 : i32, backend_config = {label = "false"}, has_side_effect = true} : (tensor<i64>, !stablehlo.token) -> !stablehlo.token
                      stablehlo.return %arg1, %0, %3 : tensor<i64>, !stablehlo.token, !stablehlo.token
                    }) : (tensor<i1>) -> (tensor<i64>, !stablehlo.token, !stablehlo.token)
                    return %2#0 : tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_condition_execution_transposed_without_cotangent_outputs_runs_backward_effects() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let branch = |label| {
            let mut backward = ArrayIrXlaProgramBuilder::new();
            let seed = backward.add_input(scalar_type.clone().into());
            backward
                .add_instruction(PrintOperation::<ArrayType>::new(label), Vec::new(), vec![seed], None)
                .unwrap();
            let zero =
                backward.add_instruction(ZeroOperation::new(scalar_type.clone()), Vec::new(), vec![], None).unwrap()[0];
            let backward = backward
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![zero], vec![Placeholder], vec![Placeholder])
                .unwrap();
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let backward = builder.import_region(backward.entry_region_ref());
            let input = builder.add_input(scalar_type.clone().into());
            let output = builder
                .add_instruction(
                    CompositeTransposeOperation::<XlaConstant, XlaOperation>::from_backward_region(
                        0,
                        vec![scalar_type.clone().into()],
                        vec![scalar_type.clone().into()],
                    ),
                    vec![backward],
                    vec![input],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_branch = builder.import_region(branch("true backward").entry_region_ref());
        let false_branch = builder.import_region(branch("false backward").entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone().into());
        let input = builder.add_input(scalar_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.entry_region_ref().transpose(&[1], &[], &[]).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Zero]);
        let program = transposed.into_program();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f64[], %1:bool[] .
                let () = condition %1 %0 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = print [label=true backward] %0
                            %2:f64[] = zero [type=f64[]]
                        in ()
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = print [label=false backward] %0
                            %2:f64[] = zero [type=f64[]]
                        in ()
                    },
                ]
                in ()"#},
        );
        let module = to_mlir_module_for_program(
            program.as_ref(),
            &[],
            &vec![scalar_type, predicate_type],
            &Vec::<ArrayType>::new(),
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<i1>) {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = "stablehlo.if"(%arg1) ({
                      %2 = stablehlo.custom_call @ryft.print(%arg0, %0) {api_version = 4 : i32, backend_config = {label = "true backward"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      stablehlo.return %2 : !stablehlo.token
                    }, {
                      %2 = stablehlo.custom_call @ryft.print(%arg0, %0) {api_version = 4 : i32, backend_config = {label = "false backward"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      stablehlo.return %2 : !stablehlo.token
                    }) : (tensor<i1>) -> !stablehlo.token
                    return
                  }
                }
                "#},
        );
        let client = execution_client();
        ensure_print_handler_registered(&client).unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        for (predicate, expected) in [(true, "true backward: 7.0"), (false, "false backward: 7.0")] {
            let inputs = vec![
                ExecutionInput {
                    buffer: Arc::new(
                        client.buffer(&7f64.to_ne_bytes(), BufferType::F64, &[], None, device.clone(), None).unwrap(),
                    ),
                    donatable: false,
                },
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(&[u8::from(predicate)], BufferType::Predicate, &[], None, device.clone(), None)
                            .unwrap(),
                    ),
                    donatable: false,
                },
            ];
            let (outputs, lines) = with_captured_prints(|| {
                executable
                    .execute(
                        vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                        Vec::new(),
                        0,
                        None,
                        Some(file!()),
                        None,
                        None,
                    )
                    .unwrap()
                    .block_until_ready()
                    .unwrap()
            });
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].outputs.len(), 0);
            assert_eq!(lines, vec![expected]);
        }
    }

    #[test]
    fn test_condition_without_outputs_executes_selected_effects_in_order_on_cpu() {
        // An empty data-flow boundary must retain the selected branch's effects and sequence later effects after it.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let branch = |label| {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone().into());
            builder
                .add_instruction(PrintOperation::<ArrayType>::new(label), Vec::new(), vec![input], None)
                .unwrap();
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![], vec![Placeholder], vec![]).unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_branch = builder.import_region(branch("true").entry_region_ref());
        let false_branch = builder.import_region(branch("false").entry_region_ref());
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let predicate = builder.add_input(predicate_type.clone().into());
        let input = builder.add_input(scalar_type.clone().into());
        let outputs = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                vec![predicate, input],
                None,
            )
            .unwrap();
        assert_eq!(outputs.len(), 0);
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("after"), Vec::new(), vec![input], None)
            .unwrap();
        let program =
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![], vec![Placeholder; 2], vec![]).unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![predicate_type, scalar_type],
            &Vec::<ArrayType>::new(),
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<f64>) {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = "stablehlo.if"(%arg0) ({
                      %3 = stablehlo.custom_call @ryft.print(%arg1, %0) {api_version = 4 : i32, backend_config = {label = "true"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      stablehlo.return %3 : !stablehlo.token
                    }, {
                      %3 = stablehlo.custom_call @ryft.print(%arg1, %0) {api_version = 4 : i32, backend_config = {label = "false"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      stablehlo.return %3 : !stablehlo.token
                    }) : (tensor<i1>) -> !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.print(%arg1, %1) {api_version = 4 : i32, backend_config = {label = "after"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                    return
                  }
                }
            "#},
        );
        let client = execution_client();
        ensure_print_handler_registered(&client).unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        for (predicate, expected) in [(true, "true: 7.0"), (false, "false: 7.0")] {
            let inputs = vec![
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(&[u8::from(predicate)], BufferType::Predicate, &[], None, device.clone(), None)
                            .unwrap(),
                    ),
                    donatable: false,
                },
                ExecutionInput {
                    buffer: Arc::new(
                        client.buffer(&7f64.to_ne_bytes(), BufferType::F64, &[], None, device.clone(), None).unwrap(),
                    ),
                    donatable: false,
                },
            ];
            let (outputs, lines) = with_captured_prints(|| {
                executable
                    .execute(
                        vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                        Vec::new(),
                        0,
                        None,
                        Some(file!()),
                        None,
                        None,
                    )
                    .unwrap()
                    .block_until_ready()
                    .unwrap()
            });
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].outputs.len(), 0);
            assert_eq!(lines, vec![expected, "after: 7.0"]);
        }
    }

    #[test]
    fn test_condition_converts_a_refined_static_branch_input_to_its_bounded_branch_input_type() {
        // A branch input only needs to refine its branch input type, so a static `f64[3]` input feeds a branch input
        // with the bounded dynamic type `f64[rows]`. The output takes the static extent that the inputs establish for
        // `rows`, so the bounded branch result is sliced back to `f64[3]`. The taken branch computes `a + b`.
        let static_type = ArrayType::new_static(DataType::F64, [3]);
        let branch_input_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("rows", Some(5))]));
        let branch = |add: bool| {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let first = builder.add_input(branch_input_type.clone().into());
            let second = builder.add_input(branch_input_type.clone().into());
            let output = if add {
                builder
                    .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![first, second], None)
                    .unwrap()[0]
            } else {
                first
            };
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_branch = builder.import_region(branch(true).entry_region_ref());
        let false_branch = builder.import_region(branch(false).entry_region_ref());
        let predicate = builder.add_constant(XlaConstant::Boolean(true));
        let dynamic_input = builder.add_input(branch_input_type.clone().into());
        let static_input = builder.add_input(static_type.into());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                vec![predicate, dynamic_input, static_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(program.output_types(), vec![ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]))]);
        assert_eq!(
            execute_mixed_program(
                &execution_client(),
                &program,
                &[
                    MixedValue::Array(vec![1.0, 2.0, 3.0, 0.0], vec![4]),
                    MixedValue::Array(vec![10.0, 20.0, 30.0], vec![3]),
                ],
                &[3],
            ),
            Ok(vec![MixedValue::Array(vec![11.0, 22.0, 33.0], vec![3])]),
        );
    }

    #[test]
    fn test_condition_normalizes_retained_branch_signatures_before_lowering_on_cpu() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![extent_type.dimension().clone()]));
        let static_type = ArrayType::new_static(DataType::F64, [3]);
        let client = execution_client();
        ensure_assertion_handler_registered(&client).unwrap();
        let mut builder = ArrayIrXlaProgramBuilder::new();
        builder.add_input(extent_type.clone().into());
        let input = builder.add_input(dynamic_type.clone().into());
        let true_branch = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ArrayIrXlaProgramBuilder::new();
        builder.add_input(extent_type.clone().into());
        builder.add_input(dynamic_type.into());
        let output = builder
            .add_instruction(ZeroOperation::new(static_type.clone().into()), Vec::new(), Vec::new(), None)
            .unwrap()[0];
        let false_branch = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        for predicate_value in [false, true] {
            // The broad first-class dimension and static array share an extent relationship that structural
            // specialization cannot replace. Direct construction retains both declared branch programs, whose declared
            // outputs (`f64[extent]` and `f64[3]`) agree only after input refinement.
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let regions = vec![
                builder.import_region(true_branch.entry_region_ref()),
                builder.import_region(false_branch.entry_region_ref()),
            ];
            let predicate = builder.add_constant(XlaConstant::Boolean(predicate_value));
            let extent = builder.add_input(extent_type.clone().into());
            let input = builder.add_input(static_type.clone().into());
            let output = builder
                .add_instruction(ConditionOperation::new(), regions, vec![predicate, extent, input], None)
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            assert_eq!(program.output_types(), vec![ArrayIrType::Array(static_type.clone())]);
            let expected = if predicate_value { vec![1.0, 2.0, 3.0] } else { vec![0.0; 3] };
            assert_eq!(
                execute_mixed_program(
                    &client,
                    &program,
                    &[MixedValue::Dimension(3), MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3])],
                    &[],
                ),
                Ok(vec![MixedValue::Array(expected, vec![3])]),
            );

            // Lowering retypes each branch input independently, so only the entry guard rejects a dimension that
            // disagrees with the static array extent, before either branch runs.
            let error = execute_mixed_program(
                &client,
                &program,
                &[MixedValue::Dimension(4), MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3])],
                &[],
            )
            .unwrap_err()
            .to_string();
            // The runtime appends its own context to the assertion message.
            assert!(
                error.contains("input dimension `extent` must have one extent at the entry of `condition`"),
                "{error}",
            );
        }
    }

    #[test]
    fn test_condition_lowering_rejects_malformed_instructions() {
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let block = context.block_with_no_arguments();
        let branch = unproject_plain_program(xla_identity_branch(ArrayType::scalar(DataType::F32)));

        // The plain array family cannot carry branch regions, so conditions lower only from the array IR family.
        let mut lowerer = XlaLowerer::new(block.as_ref(), &context, location).with_regions(vec![branch.clone(); 2]);
        let expected_message = concat!(
            "higher-order operation `condition` must be stored directly ",
            "in the enclosing backend operation family",
        );
        assert!(matches!(
            ConditionOperation::<ArrayType>::new().lower(&mut lowerer, &[], &[], &[]),
            Err(LoweringError::Tracing(ProgramError::UnsupportedOperation { message })) if message == expected_message,
        ));

        // An array IR condition needs exactly one region per branch.
        let mut lowerer = XlaLowerer::new(block.as_ref(), &context, location).with_regions(vec![branch.clone()]);
        assert!(matches!(
            ConditionOperation::<ArrayIrType>::new().lower(&mut lowerer, &[], &[], &[]),
            Err(LoweringError::Tracing(ProgramError::MalformedProgram(message)))
                if message == "operation `condition` declares 2 region slots but 1 regions were attached",
        ));

        // It also needs the predicate followed by one input per branch input.
        let mut lowerer = XlaLowerer::new(block.as_ref(), &context, location).with_regions(vec![branch; 2]);
        assert!(matches!(
            ConditionOperation::<ArrayIrType>::new().lower(&mut lowerer, &[], &[], &[]),
            Err(LoweringError::Tracing(ProgramError::InvalidInputCount { expected: 2, actual: 0 })),
        ));
    }

    #[test]
    fn test_condition_execution_bounded_dynamic_and_zero_space_outputs() {
        // Both branches return a bounded dynamic `f64[extent]` and a zero-space `f64[0]`. The dynamic result keeps its
        // runtime extent at every extent that its bound admits, including zero and the largest one, and the zero-space
        // result stays empty whichever branch runs.
        let extent_type = DimensionType::new("extent", DimensionBounds::non_negative(Some(9)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let empty_type = ArrayType::new_static(DataType::F64, [0]);
        let branch = |negate: bool| {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let extent = builder.add_input(extent_type.clone().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let (scalar, empty) = if negate {
                let scalar = builder.add_instruction(NegOperation::new(), Vec::new(), vec![scalar], None).unwrap()[0];
                let empty = builder
                    .add_instruction(ZeroOperation::new(empty_type.clone()), Vec::new(), Vec::new(), None)
                    .unwrap()[0];
                (scalar, empty)
            } else {
                let empty = builder
                    .add_instruction(OneOperation::new(empty_type.clone()), Vec::new(), Vec::new(), None)
                    .unwrap()[0];
                (scalar, empty)
            };
            let vector = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, extent], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![vector, empty],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_region(branch(false).entry_region_ref());
        let false_region = builder.import_region(branch(true).entry_region_ref());
        let selector = builder.add_input(scalar_type.clone().into());
        let extent = builder.add_input(extent_type.into());
        let scalar = builder.add_input(scalar_type.into());
        let predicate = positive(&mut builder, selector);
        let outputs = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, extent, scalar],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f64[], %1:dimension<extent ∈ [0, 9)>, %2:f64[] .
                let %3:f64[] = zero_like %0
                    %4:bool[] = compare [direction=GreaterThan] %0 %3
                    %5:f64[extent], %6:f64[0] = condition %4 %1 %2 [
                        true={
                            lambda %0:dimension<extent ∈ [0, 9)>, %1:f64[] .
                            let %2:f64[0] = one [type=f64[0]]
                                %3:f64[extent] = broadcast [output_axes=[]] %1 %0
                            in (%3, %2)
                        },
                        false={
                            lambda %0:dimension<extent ∈ [0, 9)>, %1:f64[] .
                            let %2:f64[] = neg %1
                                %3:f64[0] = zero [type=f64[0]]
                                %4:f64[extent] = broadcast [output_axes=[]] %2 %0
                            in (%4, %3)
                        },
                    ]
                in (%5, %6)"#},
        );
        assert_eq!(
            lowered_mixed_module(&program),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<i64>, %arg2: tensor<f64>) -> (tensor<?xf64, #stablehlo.bounds<8>>, tensor<0xf64>, tensor<i64>) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.compare GT, %arg0, %cst, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
                    %1:2 = "stablehlo.if"(%0) ({
                      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                      %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<0xf64>
                      %5 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<8xf64>
                      %6 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
                      %7 = stablehlo.set_dimension_size %5, %6, dim = 0 : (tensor<8xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<8>>
                      stablehlo.return %7, %4 : tensor<?xf64, #stablehlo.bounds<8>>, tensor<0xf64>
                    }, {
                      %4 = stablehlo.negate %arg2 : tensor<f64>
                      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                      %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<0xf64>
                      %6 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f64>) -> tensor<8xf64>
                      %7 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
                      %8 = stablehlo.set_dimension_size %6, %7, dim = 0 : (tensor<8xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<8>>
                      stablehlo.return %8, %5 : tensor<?xf64, #stablehlo.bounds<8>>, tensor<0xf64>
                    }) : (tensor<i1>) -> (tensor<?xf64, #stablehlo.bounds<8>>, tensor<0xf64>)
                    %2 = stablehlo.get_dimension_size %1#0, dim = 0 : (tensor<?xf64, #stablehlo.bounds<8>>) -> tensor<i32>
                    %3 = stablehlo.convert %2 : (tensor<i32>) -> tensor<i64>
                    return %1#0, %1#1, %3 : tensor<?xf64, #stablehlo.bounds<8>>, tensor<0xf64>, tensor<i64>
                  }
                }
            "#},
        );
        let client = execution_client();
        for (selector, value) in [(1.0, 7.0), (-1.0, -7.0)] {
            for extent in [0, 1, 8] {
                assert_eq!(
                    execute_mixed_program(
                        &client,
                        &program,
                        &[
                            MixedValue::Array(vec![selector], vec![]),
                            MixedValue::Dimension(extent),
                            MixedValue::Array(vec![7.0], vec![]),
                        ],
                        &[],
                    ),
                    Ok(vec![
                        MixedValue::Array(vec![value; extent as usize], vec![extent as u64]),
                        MixedValue::Array(Vec::new(), vec![0]),
                        MixedValue::Dimension(extent),
                    ]),
                );
            }
        }
    }

    #[test]
    fn test_condition_execution_returns_equal_bounded_dynamic_results() {
        // Each branch returns one bounded dynamic value twice, and the program returns both condition results. Each
        // result must keep its own runtime extent at the executable boundary.
        let extent_type = DimensionType::new("extent", DimensionBounds::non_negative(Some(9)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let branch = |negate: bool| {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let extent = builder.add_input(extent_type.clone().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let scalar = if negate {
                builder.add_instruction(NegOperation::new(), Vec::new(), vec![scalar], None).unwrap()[0]
            } else {
                scalar
            };
            let vector = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, extent], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![vector, vector],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_region(branch(false).entry_region_ref());
        let false_region = builder.import_region(branch(true).entry_region_ref());
        let selector = builder.add_input(scalar_type.clone().into());
        let extent = builder.add_input(extent_type.into());
        let scalar = builder.add_input(scalar_type.into());
        let predicate = positive(&mut builder, selector);
        let outputs = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, extent, scalar],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f64[], %1:dimension<extent ∈ [0, 9)>, %2:f64[] .
                let %3:f64[] = zero_like %0
                    %4:bool[] = compare [direction=GreaterThan] %0 %3
                    %5:f64[extent], %6:f64[extent] = condition %4 %1 %2 [
                        true={
                            lambda %0:dimension<extent ∈ [0, 9)>, %1:f64[] .
                            let %2:f64[extent] = broadcast [output_axes=[]] %1 %0
                            in (%2, %2)
                        },
                        false={
                            lambda %0:dimension<extent ∈ [0, 9)>, %1:f64[] .
                            let %2:f64[] = neg %1
                                %3:f64[extent] = broadcast [output_axes=[]] %2 %0
                            in (%3, %3)
                        },
                    ]
                in (%5, %6)"#},
        );
        assert_eq!(
            lowered_mixed_module(&program),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<i64>, %arg2: tensor<f64>) -> (tensor<?xf64, #stablehlo.bounds<8>>, tensor<?xf64, #stablehlo.bounds<8>>, tensor<i64>, tensor<i64>) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.compare GT, %arg0, %cst, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
                    %1:2 = "stablehlo.if"(%0) ({
                      %7 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<8xf64>
                      %8 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
                      %9 = stablehlo.set_dimension_size %7, %8, dim = 0 : (tensor<8xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<8>>
                      stablehlo.return %9, %9 : tensor<?xf64, #stablehlo.bounds<8>>, tensor<?xf64, #stablehlo.bounds<8>>
                    }, {
                      %7 = stablehlo.negate %arg2 : tensor<f64>
                      %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f64>) -> tensor<8xf64>
                      %9 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
                      %10 = stablehlo.set_dimension_size %8, %9, dim = 0 : (tensor<8xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<8>>
                      stablehlo.return %10, %10 : tensor<?xf64, #stablehlo.bounds<8>>, tensor<?xf64, #stablehlo.bounds<8>>
                    }) : (tensor<i1>) -> (tensor<?xf64, #stablehlo.bounds<8>>, tensor<?xf64, #stablehlo.bounds<8>>)
                    %2:2 = stablehlo.optimization_barrier %1#0, %1#1 : tensor<?xf64, #stablehlo.bounds<8>>, tensor<?xf64, #stablehlo.bounds<8>>
                    %3 = stablehlo.get_dimension_size %1#0, dim = 0 : (tensor<?xf64, #stablehlo.bounds<8>>) -> tensor<i32>
                    %4 = stablehlo.convert %3 : (tensor<i32>) -> tensor<i64>
                    %5 = stablehlo.get_dimension_size %1#1, dim = 0 : (tensor<?xf64, #stablehlo.bounds<8>>) -> tensor<i32>
                    %6 = stablehlo.convert %5 : (tensor<i32>) -> tensor<i64>
                    return %2#0, %2#1, %4, %6 : tensor<?xf64, #stablehlo.bounds<8>>, tensor<?xf64, #stablehlo.bounds<8>>, tensor<i64>, tensor<i64>
                  }
                }
            "#},
        );
        let client = execution_client();
        for (selector, extent, value) in [(1.0, 3, 7.0), (-1.0, 5, -7.0)] {
            assert_eq!(
                execute_mixed_program(
                    &client,
                    &program,
                    &[
                        MixedValue::Array(vec![selector], vec![]),
                        MixedValue::Dimension(extent),
                        MixedValue::Array(vec![7.0], vec![]),
                    ],
                    &[],
                ),
                Ok(vec![
                    MixedValue::Array(vec![value; extent as usize], vec![extent as u64]),
                    MixedValue::Array(vec![value; extent as usize], vec![extent as u64]),
                    MixedValue::Dimension(extent),
                    MixedValue::Dimension(extent),
                ]),
            );
        }
    }

    #[test]
    fn test_condition_execution_skips_assertions_of_the_untaken_branch() {
        // The `false` branch asserts that the selector is positive, which fails exactly when that branch is selected.
        // The assertion must not run when the `true` branch is selected.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let true_branch = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let selector = builder.add_input(scalar_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![selector], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let selector = builder.add_input(scalar_type.clone().into());
            let predicate = positive(&mut builder, selector);
            builder
                .add_instruction(
                    AssertOperation::<ArrayIrType>::new("selector must be positive")
                        .with_labels(vec!["selector".to_owned()]),
                    Vec::new(),
                    vec![predicate, selector],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![selector], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let selector = builder.add_input(scalar_type.into());
        let predicate = positive(&mut builder, selector);
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, selector],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f64[] .
                let %1:f64[] = zero_like %0
                    %2:bool[] = compare [direction=GreaterThan] %0 %1
                    %3:f64[] = condition %2 %0 [
                        true={
                            lambda %0:f64[] .
                            in (%0)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = zero_like %0
                                %2:bool[] = compare [direction=GreaterThan] %0 %1
                                () = assert [message="selector must be positive", labels=["selector"]] %2 %0
                            in (%0)
                        },
                    ]
                in (%3)"#},
        );
        assert_eq!(
            lowered_mixed_module(&program),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.compare GT, %arg0, %cst, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.after_all  : !stablehlo.token
                    %3:3 = "stablehlo.if"(%0) ({
                      stablehlo.return %arg0, %1, %2 : tensor<f64>, !stablehlo.token, !stablehlo.token
                    }, {
                      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                      %4 = stablehlo.compare GT, %arg0, %cst_0, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
                      %5 = stablehlo.custom_call @ryft.assert(%4, %arg0, %1) {api_version = 4 : i32, backend_config = {actor = "assert", kind = "generic", label_0 = "selector", label_count = "1", message = "selector must be positive"}, has_side_effect = true} : (tensor<i1>, tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      stablehlo.return %arg0, %5, %2 : tensor<f64>, !stablehlo.token, !stablehlo.token
                    }) : (tensor<i1>) -> (tensor<f64>, !stablehlo.token, !stablehlo.token)
                    return %3#0 : tensor<f64>
                  }
                }
            "#},
        );
        let client = execution_client();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(vec![2.0], vec![])], &[]),
            Ok(vec![MixedValue::Array(vec![2.0], vec![])]),
        );
        let error = execute_mixed_program(&client, &program, &[MixedValue::Array(vec![-2.0], vec![])], &[])
            .unwrap_err()
            .to_string();
        // The runtime appends its own context to the assertion message.
        assert!(error.contains("assertion failed: selector must be positive; selector=-2"), "{error}");
    }

    #[test]
    fn test_condition_execution_nested_in_while() {
        // A loop body selects one of two printing branches per iteration. The loop must thread the ordered I/O token
        // through the condition, so that the prints appear once per iteration and in iteration order.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let print_branch = |label| {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let index = builder.add_input(scalar_type.clone().into());
            let index = builder
                .add_instruction(PrintOperation::<ArrayType>::new(label), Vec::new(), vec![index], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![index], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let loop_condition = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let index = builder.add_input(scalar_type.clone().into());
            let limit = builder.add_input(scalar_type.clone().into());
            let predicate = builder
                .add_instruction(compare(ComparisonDirection::LessThan), Vec::new(), vec![index, limit], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let loop_body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let first_region = builder.import_region(print_branch("first").entry_region_ref());
            let later_region = builder.import_region(print_branch("later").entry_region_ref());
            let index = builder.add_input(scalar_type.clone().into());
            let limit = builder.add_input(scalar_type.clone().into());
            let one = builder
                .add_instruction(OneLikeOperation::<ArrayType>::new(), Vec::new(), vec![index], None)
                .unwrap()[0];
            let is_first = builder
                .add_instruction(compare(ComparisonDirection::LessThan), Vec::new(), vec![index, one], None)
                .unwrap()[0];
            let index = builder
                .add_instruction(
                    XlaOperation::Condition(ConditionOperation::new()),
                    vec![first_region, later_region],
                    vec![is_first, index],
                    None,
                )
                .unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![index, one], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![next, limit],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let condition_region = builder.import_region(loop_condition.entry_region_ref());
        let body_region = builder.import_region(loop_body.entry_region_ref());
        let index = builder.add_input(scalar_type.clone().into());
        let limit = builder.add_input(scalar_type.into());
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![index, limit],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = while %0 %1 [
                    condition={
                        lambda %0:f64[], %1:f64[] .
                        let %2:bool[] = compare [direction=LessThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = one_like %0
                            %3:bool[] = compare [direction=LessThan] %0 %2
                            %4:f64[] = condition %3 %0 [
                                true={
                                    lambda %0:f64[] .
                                    let %1:f64[] = print [label=first] %0
                                    in (%1)
                                },
                                false={
                                    lambda %0:f64[] .
                                    let %1:f64[] = print [label=later] %0
                                    in (%1)
                                },
                            ]
                            %5:f64[] = add %4 %2
                        in (%5, %1)
                    },
                ]
                in (%2)"#},
        );
        assert_eq!(
            lowered_mixed_module(&program),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:3 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %arg1, %iterArg_1 = %0) : tensor<f64>, tensor<f64>, !stablehlo.token
                    cond {
                      %2 = stablehlo.compare LT, %iterArg, %iterArg_0, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
                      stablehlo.return %2 : tensor<i1>
                    } do {
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                      %2 = stablehlo.compare LT, %iterArg, %cst, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
                      %3:2 = "stablehlo.if"(%2) ({
                        %5 = stablehlo.custom_call @ryft.print(%iterArg, %iterArg_1) {api_version = 4 : i32, backend_config = {label = "first"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                        stablehlo.return %iterArg, %5 : tensor<f64>, !stablehlo.token
                      }, {
                        %5 = stablehlo.custom_call @ryft.print(%iterArg, %iterArg_1) {api_version = 4 : i32, backend_config = {label = "later"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                        stablehlo.return %iterArg, %5 : tensor<f64>, !stablehlo.token
                      }) : (tensor<i1>) -> (tensor<f64>, !stablehlo.token)
                      %4 = stablehlo.add %3#0, %cst : tensor<f64>
                      stablehlo.return %4, %iterArg_0, %3#1 : tensor<f64>, tensor<f64>, !stablehlo.token
                    }
                    return %1#0 : tensor<f64>
                  }
                }
            "#},
        );
        let client = execution_client();
        ensure_print_handler_registered(&client).unwrap();
        let (outputs, lines) = with_captured_prints(|| {
            execute_mixed_program(
                &client,
                &program,
                &[MixedValue::Array(vec![0.0], vec![]), MixedValue::Array(vec![3.0], vec![])],
                &[],
            )
        });
        assert_eq!(outputs, Ok(vec![MixedValue::Array(vec![3.0], vec![])]));
        assert_eq!(lines, vec!["first: 0.0", "later: 1.0", "later: 2.0"]);
    }

    #[test]
    fn test_condition_execution_nested_in_scan() {
        // A scan body adds positive items to its carry and subtracts the others, printing each item from the selected
        // branch. The scan must thread the ordered I/O token through the condition, so that the prints appear in item
        // order.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let branch = |label, add: bool| {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let carry = builder.add_input(scalar_type.clone().into());
            let item = builder.add_input(scalar_type.clone().into());
            let item = builder
                .add_instruction(PrintOperation::<ArrayType>::new(label), Vec::new(), vec![item], None)
                .unwrap()[0];
            let carry = if add {
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, item], None).unwrap()[0]
            } else {
                builder.add_instruction(SubOperation::new(), Vec::new(), vec![carry, item], None).unwrap()[0]
            };
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![carry], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let true_region = builder.import_region(branch("positive", true).entry_region_ref());
            let false_region = builder.import_region(branch("nonpositive", false).entry_region_ref());
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let item = builder.add_input(scalar_type.clone().into());
            let predicate = positive(&mut builder, item);
            let carry = builder
                .add_instruction(
                    XlaOperation::Condition(ConditionOperation::new()),
                    vec![true_region, false_region],
                    vec![predicate, carry, item],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![carry, carry],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.into());
        let items = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let outputs = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(1, 3)),
                vec![body_region],
                vec![initial, items],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[], %3:f64[3] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = zero_like %2
                            %4:bool[] = compare [direction=GreaterThan] %2 %3
                            %5:f64[] = condition %4 %1 %2 [
                                true={
                                    lambda %0:f64[], %1:f64[] .
                                    let %2:f64[] = print [label=positive] %1
                                        %3:f64[] = add %0 %2
                                    in (%3)
                                },
                                false={
                                    lambda %0:f64[], %1:f64[] .
                                    let %2:f64[] = print [label=nonpositive] %1
                                        %3:f64[] = sub %0 %2
                                    in (%3)
                                },
                            ]
                        in (%5, %5)
                    },
                ]
                in (%2, %3)"#},
        );
        assert_eq!(
            lowered_mixed_module(&program),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<3xf64>) -> (tensor<f64>, tensor<3xf64>) {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2:5 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %arg1, %iterArg_2 = %0, %iterArg_3 = %1) : tensor<i64>, tensor<f64>, tensor<3xf64>, tensor<3xf64>, !stablehlo.token
                    cond {
                      %c_4 = stablehlo.constant dense<3> : tensor<i64>
                      %3 = stablehlo.compare LT, %iterArg, %c_4, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %3 : tensor<i1>
                    } do {
                      %3 = stablehlo.dynamic_slice %iterArg_1, %iterArg, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
                      %4 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
                      %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                      %5 = stablehlo.compare GT, %4, %cst_4, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
                      %6:2 = "stablehlo.if"(%5) ({
                        %10 = stablehlo.custom_call @ryft.print(%4, %iterArg_3) {api_version = 4 : i32, backend_config = {label = "positive"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                        %11 = stablehlo.add %iterArg_0, %4 : tensor<f64>
                        stablehlo.return %11, %10 : tensor<f64>, !stablehlo.token
                      }, {
                        %10 = stablehlo.custom_call @ryft.print(%4, %iterArg_3) {api_version = 4 : i32, backend_config = {label = "nonpositive"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                        %11 = stablehlo.subtract %iterArg_0, %4 : tensor<f64>
                        stablehlo.return %11, %10 : tensor<f64>, !stablehlo.token
                      }) : (tensor<i1>) -> (tensor<f64>, !stablehlo.token)
                      %7 = stablehlo.reshape %6#0 : (tensor<f64>) -> tensor<1xf64>
                      %8 = stablehlo.dynamic_update_slice %iterArg_2, %7, %iterArg : (tensor<3xf64>, tensor<1xf64>, tensor<i64>) -> tensor<3xf64>
                      %c_5 = stablehlo.constant dense<1> : tensor<i64>
                      %9 = stablehlo.add %iterArg, %c_5 : tensor<i64>
                      stablehlo.return %9, %6#0, %iterArg_1, %8, %6#1 : tensor<i64>, tensor<f64>, tensor<3xf64>, tensor<3xf64>, !stablehlo.token
                    }
                    return %2#1, %2#3 : tensor<f64>, tensor<3xf64>
                  }
                }
            "#},
        );
        let client = execution_client();
        ensure_print_handler_registered(&client).unwrap();
        let (outputs, lines) = with_captured_prints(|| {
            execute_mixed_program(
                &client,
                &program,
                &[MixedValue::Array(vec![0.0], vec![]), MixedValue::Array(vec![1.0, -2.0, 3.0], vec![3])],
                &[],
            )
        });
        assert_eq!(
            outputs,
            Ok(vec![MixedValue::Array(vec![6.0], vec![]), MixedValue::Array(vec![1.0, 3.0, 6.0], vec![3])]),
        );
        assert_eq!(lines, vec!["positive: 1.0", "nonpositive: -2.0", "positive: 3.0"]);
    }

    #[test]
    fn test_condition_execution_batched() {
        // The condition negates its input when the selector is positive and doubles it otherwise.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let branch = |negate: bool| {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone().into());
            let output = if negate {
                builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0]
            } else {
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0]
            };
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_region(branch(true).entry_region_ref());
        let false_region = builder.import_region(branch(false).entry_region_ref());
        let selector = builder.add_input(scalar_type.clone().into());
        let input = builder.add_input(scalar_type.into());
        let predicate = positive(&mut builder, selector);
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let client = execution_client();
        let batch_type = DimensionType::from(Dimension::Static(3));

        // A replicated predicate selects one branch for the whole batch, so the batched program keeps the condition.
        let (batched, output_axes) = program
            .batched_with_threaded_extent(
                batch_type.clone(),
                ShardingDimension::replicated(),
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {r#"
                lambda %0:dimension<3>, %1:f64[], %2:f64[3] .
                let %3:f64[] = zero_like %1
                    %4:bool[] = compare [direction=GreaterThan] %1 %3
                    %5:f64[3] = condition %4 %0 %2 [
                        true={
                            lambda %0:dimension<3>, %1:f64[3] .
                            let %2:f64[3] = neg %1
                            in (%2)
                        },
                        false={
                            lambda %0:dimension<3>, %1:f64[3] .
                            let %2:f64[3] = add %1 %1
                            in (%2)
                        },
                    ]
                in (%0, %5)"#},
        );
        for (selector, expected) in [(1.0, vec![-1.0, -2.0, -3.0]), (-1.0, vec![2.0, 4.0, 6.0])] {
            assert_eq!(
                execute_mixed_program(
                    &client,
                    &batched,
                    &[
                        MixedValue::Dimension(3),
                        MixedValue::Array(vec![selector], vec![]),
                        MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]),
                    ],
                    &[],
                ),
                Ok(vec![MixedValue::Dimension(3), MixedValue::Array(expected, vec![3])]),
            );
        }

        // A mapped predicate selects a branch per batch item, so both branches run on every item and their results are
        // selected per item.
        let (batched, output_axes) = program
            .batched_with_threaded_extent(
                batch_type,
                ShardingDimension::replicated(),
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {r#"
                lambda %0:dimension<3>, %1:f64[3], %2:f64[3] .
                let %3:f64[3] = zero_like %1
                    %4:bool[3] = compare [direction=GreaterThan] %1 %3
                    %5:f64[3] = stop_gradient %2
                    %6:f64[3] = select %4 %2 %5
                    %7:f64[3] = neg %6
                    %8:f64[3] = select %4 %5 %2
                    %9:f64[3] = add %8 %8
                    %10:f64[3] = select %4 %7 %9
                in (%0, %10)"#},
        );
        assert_eq!(
            execute_mixed_program(
                &client,
                &batched,
                &[
                    MixedValue::Dimension(3),
                    MixedValue::Array(vec![1.0, -1.0, 1.0], vec![3]),
                    MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]),
                ],
                &[],
            ),
            Ok(vec![MixedValue::Dimension(3), MixedValue::Array(vec![-1.0, 4.0, -3.0], vec![3])]),
        );
    }

    #[test]
    fn test_condition_execution_differentiated() {
        // The `true` branch broadcasts the sum of `x` back to the extent of `x`, so its tangent needs that extent as a
        // residual `dimension<n>`, while the `false` branch squares `x`, so its tangent needs `x` as a residual. Each
        // branch fills the residual slots of its peer with placeholders: the `false` branch captures the extent from
        // its known input `x`, and the `true` branch builds a zero of the extent of `x`.
        let n = DimensionType::new("n", DimensionBounds::non_negative(Some(5)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![n.dimension().clone()]));
        let true_branch = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let sum = builder
                .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![input], None)
                .unwrap()[0];
            let extent = builder
                .add_instruction(DimensionSizeOperation::new(&vector_type, 0).unwrap(), Vec::new(), vec![input], None)
                .unwrap()[0];
            let output = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![sum, extent], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let selector = builder.add_input(scalar_type.clone().into());
        let input = builder.add_input(vector_type.clone().into());
        let predicate = positive(&mut builder, selector);
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let linearization = program.linearize_with_respect_to(&[1]).unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {r#"
                lambda %0:f64[], %1:f64[n] .
                let %2:f64[] = zero_like %0
                    %3:bool[] = compare [direction=GreaterThan] %0 %2
                    %4:f64[n], %5:dimension<n ∈ [0, 5)>, %6:dimension<n ∈ [0, 5)>, %7:f64[n] = condition %3 %1 [
                        true={
                            lambda %0:f64[n] .
                            let %1:f64[] = reduce [kind=sum, axes=[0]] %0
                                %2:dimension<n ∈ [0, 5)> = dimension_size [axis=0] %0
                                %3:f64[n] = broadcast [output_axes=[]] %1 %2
                                %4:dimension<n ∈ [0, 5)> = dimension_size [axis=0] %0
                                %5:dimension<n ∈ [0, 5)> = dimension_size [axis=0] %0
                                %6:f64[n] = zero [type=f64[n]] %5
                            in (%3, %4, %2, %6)
                        },
                        false={
                            lambda %0:f64[n] .
                            let %1:f64[n] = mul %0 %0
                                %2:dimension<n ∈ [0, 5)> = dimension_size [axis=0] %0
                                %3:dimension<n ∈ [0, 5)> = dimension_size [axis=0] %0
                            in (%1, %2, %3, %0)
                        },
                    ]
                in (%4, %3, %5, %6, %7)"#},
        );
        let pullback = linearization.pullback().unwrap().into_program();
        assert_eq!(
            pullback.to_string(),
            indoc! {r#"
                lambda %0:f64[n], %1:bool[], %2:dimension<n ∈ [0, 5)>, %3:dimension<n ∈ [0, 5)>, %4:f64[n] .
                let %5:f64[n] = condition %1 %0 %2 %3 %4 [
                    true={
                        lambda %0:f64[n], %1:dimension<n ∈ [0, 5)>, %2:dimension<n ∈ [0, 5)>, %3:f64[n] .
                        let %4:f64[] = reduce [kind=sum, axes=[0]] %0
                            %5:f64[n] = linear_call [residual_count=1] %1 %4 [
                                forward={
                                    lambda %0:dimension<n ∈ [0, 5)>, %1:f64[] .
                                    let %2:f64[n] = broadcast [output_axes=[]] %1 %0
                                    in (%2)
                                },
                                transpose={
                                    lambda %0:dimension<n ∈ [0, 5)>, %1:f64[n] .
                                    let %2:f64[] = reduce [kind=sum, axes=[0]] %1
                                    in (%2)
                                },
                            ]
                        in (%5)
                    },
                    false={
                        lambda %0:f64[n], %1:dimension<n ∈ [0, 5)>, %2:dimension<n ∈ [0, 5)>, %3:f64[n] .
                        let %4:f64[n] = mul %3 %0
                            %5:f64[n] = mul %3 %0
                            %6:f64[n] = add %4 %5
                        in (%6)
                    },
                ]
                in (%5)"#},
        );

        // The primal and the pullback run in one program, which keeps the boolean predicate residual inside it.
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let selector = builder.add_input(scalar_type.into());
        let input = builder.add_input(vector_type.clone().into());
        let cotangent = builder.add_input(vector_type.into());
        let primal_outputs = builder.splice_program(linearization.primal(), &[selector, input]).unwrap();
        let mut pullback_inputs = vec![cotangent];
        pullback_inputs.extend_from_slice(&primal_outputs[1..]);
        let gradient = builder.splice_program(&pullback, &pullback_inputs).unwrap();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![primal_outputs[0], gradient[0]],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f64[], %1:f64[n], %2:f64[n] .
                let %3:f64[] = zero_like %0
                    %4:bool[] = compare [direction=GreaterThan] %0 %3
                    %5:f64[n], %6:dimension<n ∈ [0, 5)>, %7:dimension<n ∈ [0, 5)>, %8:f64[n] = condition %4 %1 [
                        true={
                            lambda %0:f64[n] .
                            let %1:f64[] = reduce [kind=sum, axes=[0]] %0
                                %2:dimension<n ∈ [0, 5)> = dimension_size [axis=0] %0
                                %3:f64[n] = broadcast [output_axes=[]] %1 %2
                                %4:dimension<n ∈ [0, 5)> = dimension_size [axis=0] %0
                                %5:dimension<n ∈ [0, 5)> = dimension_size [axis=0] %0
                                %6:f64[n] = zero [type=f64[n]] %5
                            in (%3, %4, %2, %6)
                        },
                        false={
                            lambda %0:f64[n] .
                            let %1:f64[n] = mul %0 %0
                                %2:dimension<n ∈ [0, 5)> = dimension_size [axis=0] %0
                                %3:dimension<n ∈ [0, 5)> = dimension_size [axis=0] %0
                            in (%1, %2, %3, %0)
                        },
                    ]
                    %9:f64[n] = condition %4 %2 %6 %7 %8 [
                        true={
                            lambda %0:f64[n], %1:dimension<n ∈ [0, 5)>, %2:dimension<n ∈ [0, 5)>, %3:f64[n] .
                            let %4:f64[] = reduce [kind=sum, axes=[0]] %0
                                %5:f64[n] = linear_call [residual_count=1] %1 %4 [
                                    forward={
                                        lambda %0:dimension<n ∈ [0, 5)>, %1:f64[] .
                                        let %2:f64[n] = broadcast [output_axes=[]] %1 %0
                                        in (%2)
                                    },
                                    transpose={
                                        lambda %0:dimension<n ∈ [0, 5)>, %1:f64[n] .
                                        let %2:f64[] = reduce [kind=sum, axes=[0]] %1
                                        in (%2)
                                    },
                                ]
                            in (%5)
                        },
                        false={
                            lambda %0:f64[n], %1:dimension<n ∈ [0, 5)>, %2:dimension<n ∈ [0, 5)>, %3:f64[n] .
                            let %4:f64[n] = mul %3 %0
                                %5:f64[n] = mul %3 %0
                                %6:f64[n] = add %4 %5
                            in (%6)
                        },
                    ]
                in (%5, %9)"#},
        );
        let client = execution_client();
        for (selector, expected_value, expected_gradient) in
            [(1.0, vec![6.0, 6.0, 6.0], vec![3.0, 3.0, 3.0]), (-1.0, vec![1.0, 4.0, 9.0], vec![2.0, 4.0, 6.0])]
        {
            assert_eq!(
                execute_mixed_program(
                    &client,
                    &program,
                    &[
                        MixedValue::Array(vec![selector], vec![]),
                        MixedValue::Array(vec![1.0, 2.0, 3.0, 0.0], vec![4]),
                        MixedValue::Array(vec![1.0, 1.0, 1.0, 0.0], vec![4]),
                    ],
                    &[3, 3],
                ),
                Ok(vec![
                    MixedValue::Array(expected_value, vec![3]),
                    MixedValue::Array(expected_gradient, vec![3]),
                    MixedValue::Dimension(3),
                    MixedValue::Dimension(3),
                ]),
            );
        }
    }

    #[test]
    fn test_condition_lowering_constrains_explicit_output_shardings() {
        // The `true` branch negates a vector sharded over the explicit mesh axis `x`, and the `false` branch forwards
        // it. The result of the `stablehlo.if` is constrained to the placement that its type states.
        let (client, mesh) = two_device_mesh(MeshAxisType::Explicit);
        let selector_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let vector_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let selector_type = ArrayType::scalar(DataType::F32).with_sharding(selector_sharding.clone()).unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [8]).with_sharding(vector_sharding.clone()).unwrap();
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_program(unproject_plain_program(xla_neg_branch(vector_type.clone())));
        let false_region = builder.import_program(unproject_plain_program(xla_identity_branch(vector_type.clone())));
        let selector = builder.add_input(selector_type.clone().into());
        let input = builder.add_input(vector_type.clone().into());
        let predicate = positive(&mut builder, selector);
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f32[][sharding={mesh<['x'=2:explicit]>, []}], %1:f32[8][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                let %2:f32[][sharding={mesh<['x'=2:explicit]>, []}] = zero_like %0
                    %3:bool[][sharding={mesh<['x'=2:explicit]>, []}] = compare [direction=GreaterThan] %0 %2
                    %4:f32[8][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = condition %3 %1 [
                        true={
                            lambda %0:f32[8][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                            let %1:f32[8][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = neg %0
                            in (%1)
                        },
                        false={
                            lambda %0:f32[8][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                            in (%0)
                        },
                    ]
                in (%4)"#},
        );
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![selector_type.clone(), vector_type.clone()],
            &vec![vector_type.clone()],
            "main",
            Some(&[selector_sharding, vector_sharding.clone()]),
            Some(&[vector_sharding]),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={"x"}>}, %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = sdy.sharding_constraint %cst <@mesh, [], replicated={"x"}> : tensor<f32>
                    %1 = stablehlo.compare GT, %arg0, %0, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                    %2 = "stablehlo.if"(%1) ({
                      %4 = stablehlo.negate %arg1 : tensor<8xf32>
                      stablehlo.return %4 : tensor<8xf32>
                    }, {
                      stablehlo.return %arg1 : tensor<8xf32>
                    }) : (tensor<i1>) -> tensor<8xf32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [{"x"}]> : tensor<8xf32>
                    return %3 : tensor<8xf32>
                  }
                }
            "#},
        );
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for (selector, expected) in [
            (1.0, vec![vec![vec![-1.0, -2.0, -3.0, -4.0]], vec![vec![-5.0, -6.0, -7.0, -8.0]]]),
            (-1.0, vec![vec![vec![1.0, 2.0, 3.0, 4.0]], vec![vec![5.0, 6.0, 7.0, 8.0]]]),
        ] {
            assert_eq!(
                execute_on_two_devices(
                    &client,
                    &mesh,
                    &module,
                    &[(&selector_type, &[selector]), (&vector_type, &values)],
                ),
                expected,
                "selector {selector}",
            );
        }
    }
}
