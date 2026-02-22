use crate::{
    DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Value, ValueRef, mlir_op, mlir_op_trait,
};

/// StableHLO [`Operation`] that performs element-wise left shift of two tensors. The operation shifts the bits of
/// [`ShiftLeftOperation::lhs`] to the left by the number of positions specified by [`ShiftLeftOperation::rhs`] for
/// corresponding elements from the two input tensors and produces a resulting tensor. The output shape is determined
/// by [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input tensors.
///
/// # Example
///
/// The following is an example of a [`ShiftLeftOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [-1, 0, 1]
/// // %rhs: [1, 2, 3]
/// %result = stablehlo.shift_left %lhs, %rhs : tensor<3xi64>
/// // %result: [-2, 0, 8]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#shift_left) for more information.
pub trait ShiftLeftOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`ShiftLeftOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`ShiftLeftOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(ShiftLeft);
mlir_op_trait!(ShiftLeft, OneResult);
mlir_op_trait!(ShiftLeft, ZeroRegions);
mlir_op_trait!(ShiftLeft, ZeroSuccessors);

/// Constructs a new detached/owned [`ShiftLeftOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ShiftLeftOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn shift_left<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    location: L,
) -> DetachedShiftLeftOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.shift_left", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::shift_left`")
}

/// StableHLO [`Operation`] that performs element-wise arithmetic right shift of two tensors. The operation shifts
/// the bits of [`ShiftRightArithmeticOperation::lhs`] to the right by the number of positions specified by
/// [`ShiftRightArithmeticOperation::rhs`] for corresponding elements from the two input tensors and produces
/// a resulting tensor. This is an arithmetic shift that preserves the sign bit of the
/// [`ShiftRightArithmeticOperation::lhs`] values. The output shape is determined by
/// [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input tensors.
///
/// # Example
///
/// The following is an example of a [`ShiftRightArithmeticOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [-1, 0, 8]
/// // %rhs: [1, 2, 3]
/// %result = stablehlo.shift_right_arithmetic %lhs, %rhs : tensor<3xi32>
/// // %result: [-1, 0, 1]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#shift_right_arithmetic)
/// for more information.
pub trait ShiftRightArithmeticOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`ShiftRightArithmeticOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`ShiftRightArithmeticOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(ShiftRightArithmetic);
mlir_op_trait!(ShiftRightArithmetic, OneResult);
mlir_op_trait!(ShiftRightArithmetic, ZeroRegions);
mlir_op_trait!(ShiftRightArithmetic, ZeroSuccessors);

/// Constructs a new detached/owned [`ShiftRightArithmeticOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ShiftRightArithmeticOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn shift_right_arithmetic<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    location: L,
) -> DetachedShiftRightArithmeticOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.shift_right_arithmetic", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::shift_right_arithmetic`")
}

/// StableHLO [`Operation`] that performs element-wise logical right shift of two tensors. The operation shifts
/// the bits of [`ShiftRightLogicalOperation::lhs`] to the right by the number of positions specified by
/// [`ShiftRightLogicalOperation::rhs`] for corresponding elements from the two input tensors and produces
/// a resulting tensor. This is a logical shift that does not provide any special treatment for the sign bit
/// of the [`ShiftRightLogicalOperation::lhs`] values. The output shape is determined by
/// [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input tensors.
///
/// # Example
///
/// The following is an example of a [`ShiftRightLogicalOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [-1, 0, 8]
/// // %rhs: [1, 2, 3]
/// %result = stablehlo.shift_right_logical %lhs, %rhs : tensor<3xi32>
/// // %result: [9223372036854775807, 0, 1]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#shift_right_logical)
/// for more information.
pub trait ShiftRightLogicalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`ShiftRightLogicalOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`ShiftRightLogicalOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(ShiftRightLogical);
mlir_op_trait!(ShiftRightLogical, OneResult);
mlir_op_trait!(ShiftRightLogical, ZeroRegions);
mlir_op_trait!(ShiftRightLogical, ZeroSuccessors);

/// Constructs a new detached/owned [`ShiftRightLogicalOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ShiftRightLogicalOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn shift_right_logical<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    location: L,
) -> DetachedShiftRightLogicalOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.shift_right_logical", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::shift_right_logical`")
}

/// StableHLO [`Operation`] that counts the number of leading zero bits in each element of the input tensor.
/// The output tensor has the same shape as the input tensor.
///
/// # Example
///
/// The following is an example of a [`CountLeadingZerosOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [[0, 1], [128, -1]]
/// %result = stablehlo.count_leading_zeros %operand : tensor<2x2xi64>
/// // %result: [[64, 63], [56, 0]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#count_leading_zeros)
/// for more information.
pub trait CountLeadingZerosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(CountLeadingZeros);
mlir_op_trait!(CountLeadingZeros, OneOperand);
mlir_op_trait!(CountLeadingZeros, OneResult);
mlir_op_trait!(CountLeadingZeros, ZeroRegions);
mlir_op_trait!(CountLeadingZeros, ZeroSuccessors);

/// Constructs a new detached/owned [`CountLeadingZerosOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`CountLeadingZerosOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn count_leading_zeros<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedCountLeadingZerosOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.count_leading_zeros", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::count_leading_zeros`")
}

/// StableHLO [`Operation`] that counts the number of bits set to `1` in each element of the input tensor.
/// The output tensor has the same shape as the input tensor.
///
/// # Example
///
/// The following is an example of a [`PopulationCountOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [0, 1, 2, 127]
/// %result = stablehlo.popcnt %operand : tensor<4xi64>
/// // %result: [0, 1, 1, 7]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#popcnt) for more information.
pub trait PopulationCountOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(PopulationCount);
mlir_op_trait!(PopulationCount, OneOperand);
mlir_op_trait!(PopulationCount, OneResult);
mlir_op_trait!(PopulationCount, ZeroRegions);
mlir_op_trait!(PopulationCount, ZeroSuccessors);

/// Constructs a new detached/owned [`PopulationCountOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`PopulationCountOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn population_count<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedPopulationCountOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.popcnt", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::population_count`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, OneOperand, OneResult, Operation, Size, Value};

    use super::{
        ShiftLeftOperation, ShiftRightArithmeticOperation, ShiftRightLogicalOperation, count_leading_zeros,
        population_count, shift_left, shift_right_arithmetic, shift_right_logical,
    };

    #[test]
    fn test_shift_left() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = shift_left(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "shift_left_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), tensor_type.into()],
                    results: vec![tensor_type.into()],
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
                  func.func @shift_left_test(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi32> {
                    %0 = stablehlo.shift_left %arg0, %arg1 : tensor<3xi32>
                    return %0 : tensor<3xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_shift_right_arithmetic() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = shift_right_arithmetic(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "shift_right_arithmetic_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), tensor_type.into()],
                    results: vec![tensor_type.into()],
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
                  func.func @shift_right_arithmetic_test(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi32> {
                    %0 = stablehlo.shift_right_arithmetic %arg0, %arg1 : tensor<3xi32>
                    return %0 : tensor<3xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_shift_right_logical() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = shift_right_logical(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "shift_right_logical_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), tensor_type.into()],
                    results: vec![tensor_type.into()],
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
                  func.func @shift_right_logical_test(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi32> {
                    %0 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<3xi32>
                    return %0 : tensor<3xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_count_leading_zeros() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let tensor_type = context.tensor_type(i8_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = count_leading_zeros(input, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "count_leading_zeros_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into()],
                    results: vec![tensor_type.into()],
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
                  func.func @count_leading_zeros_test(%arg0: tensor<2x2xi8>) -> tensor<2x2xi8> {
                    %0 = stablehlo.count_leading_zeros %arg0 : tensor<2x2xi8>
                    return %0 : tensor<2x2xi8>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_population_count() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let tensor_type = context.tensor_type(i8_type, &[Size::Static(4)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = population_count(input, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "popcnt_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into()],
                    results: vec![tensor_type.into()],
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
                  func.func @popcnt_test(%arg0: tensor<4xi8>) -> tensor<4xi8> {
                    %0 = stablehlo.popcnt %arg0 : tensor<4xi8>
                    return %0 : tensor<4xi8>
                  }
                }
            "},
        );
    }
}
