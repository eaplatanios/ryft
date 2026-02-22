use crate::{
    DetachedOp, DetachedRegion, DialectHandle, Location, Operation, OperationBuilder, RegionRef, Value, ValueRef,
    mlir_op, mlir_op_trait,
};

/// StableHLO [`Operation`] that performs conditional branching across multiple branches. This operation enables
/// multi-way conditional execution by evaluating an index operand and executing the corresponding branch
/// [`Region`](crate::Region). Each branch is a [`Region`](crate::Region) in this operation that computes and returns
/// a list of values. All branches must return the same number and type of results, and must use
/// [`stable_hlo::return`](crate::dialects::stable_hlo::return) for their return operation.
///
/// More formally, `result = selected_branch()` where:
///
///   - `selected_branch = branches[index]`, if `0 <= index < size(branches)`, and
///   - `selected_branch = branches[-1]`, otherwise.
///
/// And where `index` is a scalar integer tensor and the only input/operand of this operation.
///
/// # Example
///
/// The following is an example of a [`CaseOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %index: 1
/// %result = "stablehlo.case"(%index) ({
///   %c = stablehlo.constant dense<0> : tensor<2xi64>
///   stablehlo.return %c : tensor<2xi64>
/// }, {
///   %c = stablehlo.constant dense<1> : tensor<2xi64>
///   stablehlo.return %c : tensor<2xi64>
/// }) : (tensor<i32>) -> tensor<2xi64>
/// // %result: [1, 1]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#case) for more information.
pub trait CaseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns an [`Iterator`] over the branches of this [`CaseOperation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`](crate::Context) because that
    /// would make it impossible to perform mutating operations on that context (e.g., from within [`Pass`](crate::Pass)es) while
    /// iterating over the contents of that iterator.
    fn branches(&self) -> impl Iterator<Item = RegionRef<'o, 'c, 't>> {
        self.regions()
    }
}

mlir_op!(Case);
mlir_op_trait!(Case, SingleBlockRegions);
mlir_op_trait!(Case, ZeroSuccessors);

/// Constructs a new detached/owned [`CaseOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`CaseOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn case<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    index: V,
    branches: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> DetachedCaseOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.case", location)
        .add_operand(index)
        .add_regions(branches)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::case`")
}

/// StableHLO [`Operation`] that performs binary conditional execution. It receives as input (i.e., its sole operand)
/// a scalar boolean `predicate` tensor (i.e., a [`Value`] with type `tensor<i1>`) and depending on its value, executes
/// one of the two [`Region`](crate::Region)s it contains. If [`IfOperation::predicate`] is `true`, then it executes
/// [`IfOperation::true_branch`], and if it is `false`, it executes [`IfOperation::false_branch`].
/// Both branches must return the same number and type of results, and must use
/// [`stable_hlo::return`](crate::dialects::stable_hlo::return) for their return operation.
///
/// # Example
///
/// The following is an example of an [`IfOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %predicate: true
/// %result = "stablehlo.if"(%predicate) ({
///   %c = stablehlo.constant dense<0> : tensor<2xi64>
///   stablehlo.return %c : tensor<2xi64>
/// }, {
///   %c = stablehlo.constant dense<1> : tensor<2xi64>
///   stablehlo.return %c : tensor<2xi64>
/// }) : (tensor<i1>) -> tensor<2xi64>
/// // %result: [0, 0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#if) for more information.
pub trait IfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the predicate [`Value`] of this [`IfOperation`] (i.e., its only input/operand).
    fn predicate(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the `true` branch of this [`IfOperation`].
    fn true_branch(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns the `false` branch of this [`IfOperation`].
    fn false_branch(&self) -> RegionRef<'o, 'c, 't> {
        self.region(1).unwrap()
    }
}

mlir_op!(If);
mlir_op_trait!(If, SingleBlockRegions);
mlir_op_trait!(If, ZeroSuccessors);

/// Constructs a new detached/owned [`IfOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`IfOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn r#if<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    predicate: V,
    true_branch: DetachedRegion<'c, 't>,
    false_branch: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedIfOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.if", location)
        .add_operand(predicate)
        .add_region(true_branch)
        .add_region(false_branch)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::if`")
}

/// StableHLO [`Operation`] that implements iterative looping with a condition check. This operation enables repeated
/// execution of a body [`Region`](crate::Region) while a condition [`Region`](crate::Region) evaluates to `true`. The
/// operation takes initial values as operands, which are passed to both the condition and the body regions. The body
/// region produces updated values that are fed back into the next iteration's condition check, and the loop continues
/// until the condition evaluates to `false`.
///
/// More formally, the semantics can be expressed using Python-like syntax as follows:
///
/// ```python
/// internal_state = operands
/// while cond(*internal_state):
///   internal_state = body(*internal_state)
/// results = internal_state
/// ```
///
/// The number and type of the operands and the results of this operation match the number and type of the arguments
/// to both the [`WhileOperation::condition`] and [`WhileOperation::body`] regions, and also the results of the
/// [`WhileOperation::body`] region. Furthermore, the [`WhileOperation::condition`] returns a single scalar boolean
/// tensor (i.e., a [`Value`] with type `tensor<i1>`).
///
/// # Example
///
/// The following is an example of a [`WhileOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %initial_i: 0
/// // %initial_sum: 0
/// %result_i, %result_sum = stablehlo.while(%arg0 = %initial_i, %arg1 = %initial_sum) : tensor<i64>, tensor<i64>
/// cond {
///   %c = stablehlo.constant dense<10> : tensor<i64>
///   %1 = stablehlo.compare  LT, %arg0, %c,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
///   stablehlo.return %1 : tensor<i1>
/// } do {
///   %c = stablehlo.constant dense<1> : tensor<i64>
///   %1 = stablehlo.add %arg0, %c : tensor<i64>
///   %2 = stablehlo.add %arg1, %arg0 : tensor<i64>
///   stablehlo.return %1, %2 : tensor<i64>, tensor<i64>
/// }
/// // %result_i: 10
/// // %result_sum: 55
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#while) for more information.
pub trait WhileOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the condition [`Region`](crate::Region) of this [`WhileOperation`] that evaluates the loop predicate.
    fn condition(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns the body [`Region`](crate::Region) of this [`WhileOperation`] that executes the loop body.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.region(1).unwrap()
    }
}

mlir_op!(While);
mlir_op_trait!(While, SingleBlockRegions);
mlir_op_trait!(While, ZeroSuccessors);

/// Constructs a new detached/owned [`WhileOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`WhileOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn r#while<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    operands: &[V],
    condition: DetachedRegion<'c, 't>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedWhileOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.while", location)
        .add_operands(operands)
        .add_region(condition)
        .add_region(body)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::while`")
}

/// StableHLO [`Operation`] that produces a result tensor where each element is selected from
/// [`SelectOperation::on_true`] or [`SelectOperation::on_false`], based on the value of [`SelectOperation::predicate`].
/// More formally, `result[index] = on_true[index] if predicate[index] else on_false[index]`, using Python notation.
/// If not all inputs have the same shape, then [broadcasting semantics](https://openxla.org/xla/broadcasting) apply.
///
/// For quantized types, this operation first dequantizes the inputs, applies the select operation
/// as described above, and then quantizes the result.
///
/// # Example
///
/// The following is an example of a [`SelectOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %predicate: [[false, true], [true, false]]
/// // %on_true: [[1, 2], [3, 4]]
/// // %on_false: [[5, 6], [7, 8]]
/// %result = stablehlo.select %predicate, %on_true, %on_false : tensor<2x2xi1>, tensor<2x2xi32>
/// // %result: [[5, 2], [3, 8]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#select) for more information.
pub trait SelectOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `predicate` input of this [`SelectOperation`].
    fn predicate(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the `on_true` input of this [`SelectOperation`].
    fn on_true(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns the `on_false` input of this [`SelectOperation`].
    fn on_false(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(2).unwrap()
    }
}

mlir_op!(Select);
mlir_op_trait!(Select, OneResult);
mlir_op_trait!(Select, ZeroRegions);
mlir_op_trait!(Select, ZeroSuccessors);

/// Constructs a new detached/owned [`SelectOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`SelectOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn select<
    'predicate,
    'on_true,
    'on_false,
    'c: 'predicate + 'on_true + 'on_false,
    't: 'c,
    P: Value<'predicate, 'c, 't>,
    T: Value<'on_true, 'c, 't>,
    F: Value<'on_false, 'c, 't>,
    L: Location<'c, 't>,
>(
    predicate: P,
    on_true: T,
    on_false: F,
    location: L,
) -> DetachedSelectOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.select", location)
        .add_operand(predicate)
        .add_operand(on_true)
        .add_operand(on_false)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::select`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::{func, stable_hlo};
    use crate::{Block, Context, Operation, Region, Size, Value};

    use super::{CaseOperation, IfOperation, SelectOperation, WhileOperation, case, r#if, select, r#while};

    #[test]
    fn test_case() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let index_type = context.tensor_type(i32_type, &[], None, location).unwrap();
        let result_type = context.tensor_type(i64_type, &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(index_type, location)]);

            // Create the first branch region.
            let mut branch_0_region = context.region();
            let mut branch_0_block = context.block_with_no_arguments();
            let value_0_op = branch_0_block.append_operation(stable_hlo::constant(
                context.dense_i64_elements_attribute(result_type, &[0, 0]).unwrap(),
                location,
            ));
            branch_0_block.append_operation(stable_hlo::r#return(&[value_0_op.result(0).unwrap()], location));
            branch_0_region.append_block(branch_0_block);

            // Create the second branch region.
            let mut branch_1_region = context.region();
            let mut branch_1_block = context.block_with_no_arguments();
            let value_1_op = branch_1_block.append_operation(stable_hlo::constant(
                context.dense_i64_elements_attribute(result_type, &[1, 1]).unwrap(),
                location,
            ));
            branch_1_block.append_operation(stable_hlo::r#return(&[value_1_op.result(0).unwrap()], location));
            branch_1_region.append_block(branch_1_block);

            // Create the `case` operation.
            let index = block.argument(0).unwrap();
            let op = case(index, vec![branch_0_region.into(), branch_1_region.into()], location);
            assert_eq!(op.branches().count(), 2);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.regions().count(), 2);
            let op = block.append_operation(op);

            // Create a function that contains that `case` operation and returns its output.
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "case_test",
                func::FuncAttributes {
                    arguments: vec![index_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @case_test(%arg0: tensor<i32>) -> tensor<2xi64> {
                    %0 = \"stablehlo.case\"(%arg0) ({
                      %c = stablehlo.constant dense<0> : tensor<2xi64>
                      stablehlo.return %c : tensor<2xi64>
                    }, {
                      %c = stablehlo.constant dense<1> : tensor<2xi64>
                      stablehlo.return %c : tensor<2xi64>
                    }) : (tensor<i32>) -> tensor<2xi64>
                    return %0 : tensor<2xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_if() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i64_type = context.signless_integer_type(64);
        let pred_type = context.tensor_type(i1_type, &[], None, location).unwrap();
        let result_type = context.tensor_type(i64_type, &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(pred_type, location)]);

            // Create the `true` branch region.
            let mut true_branch_region = context.region();
            let mut true_branch_block = context.block_with_no_arguments();
            let true_value_op = true_branch_block.append_operation(stable_hlo::constant(
                context.dense_i64_elements_attribute(result_type, &[0, 0]).unwrap(),
                location,
            ));
            true_branch_block.append_operation(stable_hlo::r#return(&[true_value_op.result(0).unwrap()], location));
            true_branch_region.append_block(true_branch_block);

            // Create the `false` branch region.
            let mut false_branch_region = context.region();
            let mut false_branch_block = context.block_with_no_arguments();
            let false_value_op = false_branch_block.append_operation(stable_hlo::constant(
                context.dense_i64_elements_attribute(result_type, &[1, 1]).unwrap(),
                location,
            ));
            false_branch_block.append_operation(stable_hlo::r#return(&[false_value_op.result(0).unwrap()], location));
            false_branch_region.append_block(false_branch_block);

            // Create the `if` operation.
            let predicate = block.argument(0).unwrap();
            let if_op = r#if(predicate, true_branch_region.into(), false_branch_region.into(), location);
            assert_eq!(if_op.predicate(), predicate);
            assert_eq!(if_op.true_branch().blocks().count(), 1);
            assert_eq!(if_op.false_branch().blocks().count(), 1);
            assert_eq!(if_op.operands().count(), 1);
            assert_eq!(if_op.results().count(), 1);
            assert_eq!(if_op.regions().count(), 2);
            let if_op = block.append_operation(if_op);

            // Create a function that contains that `if` operation and returns its output.
            block.append_operation(func::r#return(&[if_op.result(0).unwrap()], location));
            func::func(
                "if_test",
                func::FuncAttributes {
                    arguments: vec![pred_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @if_test(%arg0: tensor<i1>) -> tensor<2xi64> {
                    %0 = \"stablehlo.if\"(%arg0) ({
                      %c = stablehlo.constant dense<0> : tensor<2xi64>
                      stablehlo.return %c : tensor<2xi64>
                    }, {
                      %c = stablehlo.constant dense<1> : tensor<2xi64>
                      stablehlo.return %c : tensor<2xi64>
                    }) : (tensor<i1>) -> tensor<2xi64>
                    return %0 : tensor<2xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_while() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let counter_type = context.tensor_type(i64_type, &[], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(counter_type, location), (counter_type, location)]);

            // Create the condition region.
            let mut condition_region = context.region();
            let mut condition_block = context.block(&[(counter_type, location), (counter_type, location)]);
            let limit_value_op = condition_block.append_operation(stable_hlo::constant(
                context.dense_i64_elements_attribute(counter_type, &[10]).unwrap(),
                location,
            ));
            let compare_op = condition_block.append_operation(stable_hlo::compare(
                condition_block.argument(0).unwrap(),
                limit_value_op.result(0).unwrap(),
                stable_hlo::ComparisonDirection::LessThan,
                stable_hlo::ComparisonType::Signed,
                location,
            ));
            let cond_result = compare_op.result(0).unwrap().as_value_ref();
            condition_block.append_operation(stable_hlo::r#return(&[cond_result], location));
            condition_region.append_block(condition_block);

            // Create the body region.
            let mut body_region = context.region();
            let mut body_block = context.block(&[(counter_type, location), (counter_type, location)]);
            let one_value_op = body_block.append_operation(stable_hlo::constant(
                context.dense_i64_elements_attribute(counter_type, &[1]).unwrap(),
                location,
            ));
            let one_value = one_value_op.result(0).unwrap().as_value_ref();
            let new_i_op =
                body_block.append_operation(stable_hlo::add(body_block.argument(0).unwrap(), one_value, location));
            let new_i = new_i_op.result(0).unwrap();
            let new_sum_op = stable_hlo::add(body_block.argument(1).unwrap(), one_value, location);
            let new_sum_block = body_block.append_operation(new_sum_op);
            let new_sum = new_sum_block.result(0).unwrap();
            body_block.append_operation(stable_hlo::r#return(&[new_i, new_sum], location));
            body_region.append_block(body_block);

            // Create the `while` operation.
            let while_op = r#while(
                block.arguments().collect::<Vec<_>>().as_slice(),
                condition_region.into(),
                body_region.into(),
                location,
            );
            assert_eq!(while_op.condition().blocks().count(), 1);
            assert_eq!(while_op.body().blocks().count(), 1);
            assert_eq!(while_op.operands().count(), 2);
            assert_eq!(while_op.results().count(), 2);
            assert_eq!(while_op.regions().count(), 2);
            let while_block = block.append_operation(while_op);

            // Create a function that contains that `while` operation and returns its outputs.
            block.append_operation(func::r#return(
                &[while_block.result(0).unwrap(), while_block.result(1).unwrap()],
                location,
            ));
            func::func(
                "while_test",
                func::FuncAttributes {
                    arguments: vec![counter_type.into(), counter_type.into()],
                    results: vec![counter_type.into(), counter_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @while_test(%arg0: tensor<i64>, %arg1: tensor<i64>) -> (tensor<i64>, tensor<i64>) {
                    %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %arg1) : tensor<i64>, tensor<i64>
                    cond {
                      %c = stablehlo.constant dense<10> : tensor<i64>
                      %1 = stablehlo.compare  LT, %iterArg, %c,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %1 : tensor<i1>
                    } do {
                      %c = stablehlo.constant dense<1> : tensor<i64>
                      %1 = stablehlo.add %iterArg, %c : tensor<i64>
                      %2 = stablehlo.add %iterArg_0, %c : tensor<i64>
                      stablehlo.return %1, %2 : tensor<i64>, tensor<i64>
                    }
                    return %0#0, %0#1 : tensor<i64>, tensor<i64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_select() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let predicate_type = context
            .tensor_type(context.signless_integer_type(1), &[Size::Static(3), Size::Static(4)], None, location)
            .unwrap();
        let value_type = context
            .tensor_type(context.signless_integer_type(32), &[Size::Static(3), Size::Static(4)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block =
                context.block(&[(predicate_type, location), (value_type, location), (value_type, location)]);
            let predicate = block.argument(0).unwrap();
            let on_true = block.argument(1).unwrap();
            let on_false = block.argument(2).unwrap();
            let op = select(predicate, on_true, on_false, location);
            assert_eq!(op.predicate(), predicate);
            assert_eq!(op.on_true(), on_true);
            assert_eq!(op.on_false(), on_false);
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), value_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "select_test",
                func::FuncAttributes {
                    arguments: vec![predicate_type.into(), value_type.into(), value_type.into()],
                    results: vec![value_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @select_test(\
                    %arg0: tensor<3x4xi1>, \
                    %arg1: tensor<3x4xi32>, \
                    %arg2: tensor<3x4xi32>\
                  ) -> tensor<3x4xi32> {
                    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<3x4xi1>, tensor<3x4xi32>
                    return %0 : tensor<3x4xi32>
                  }
                }
            "}
        );
    }
}
