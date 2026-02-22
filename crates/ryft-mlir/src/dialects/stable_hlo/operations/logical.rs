use crate::{
    DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Value, ValueRef, mlir_op, mlir_op_trait,
};

/// StableHLO [`Operation`] that performs an element-wise logical `not` operation on boolean tensors and a bitwise `not`
/// operation on integer tensors.
///
/// # Example
///
/// The following are examples of [`NotOperation`]s represented using their [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [[1, 2], [3, 4]]
/// %result = stablehlo.not %operand : tensor<2x2xi8>
/// // %result: [[-2, -3], [-4, -5]]
///
/// // %operand: [true, false]
/// %result = stablehlo.not %operand : tensor<2xi1>
/// // %result: [false, true]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#not) for more information.
pub trait NotOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Not);
mlir_op_trait!(Not, OneOperand);
mlir_op_trait!(Not, OneResult);
mlir_op_trait!(Not, ZeroRegions);
mlir_op_trait!(Not, ZeroSuccessors);

/// Constructs a new detached/owned [`NotOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`NotOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn not<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedNotOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.not", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::not`")
}

/// StableHLO [`Operation`] that performs an element-wise logical `and` operation on boolean tensors and a bitwise `and`
/// operation on integer tensors. The output shape is determined by [broadcasting](https://openxla.org/xla/broadcasting)
/// the shapes of the two input tensors.
///
/// # Example
///
/// The following is an example of an [`AndOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [[1, 2], [3, 4]]
/// // %rhs: [[5, 6], [7, 8]]
/// %result = stablehlo.and %lhs, %rhs : tensor<2x2xi32>
/// // %result: [[1, 2], [3, 0]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#and) for more information.
pub trait AndOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`AndOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`AndOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(And);
mlir_op_trait!(And, OneResult);
mlir_op_trait!(And, ZeroRegions);
mlir_op_trait!(And, ZeroSuccessors);

/// Constructs a new detached/owned [`AndOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`AndOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn and<
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
) -> DetachedAndOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.and", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::and`")
}

/// StableHLO [`Operation`] that performs an element-wise logical `or` operation on boolean tensors and a bitwise `or`
/// operation on integer tensors. The output shape is determined by [broadcasting](https://openxla.org/xla/broadcasting)
/// the shapes of the two input tensors.
///
/// # Example
///
/// The following are examples of [`OrOperation`]s represented using their [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [[1, 2], [3, 4]]
/// // %rhs: [[5, 6], [7, 8]]
/// %result = stablehlo.or %lhs, %rhs : tensor<2x2xi32>
/// // %result: [[5, 6], [7, 12]]
///
/// // %lhs: [[false, false], [true, true]]
/// // %rhs: [[false, true], [false, true]]
/// %result = stablehlo.or %lhs, %rhs : tensor<2x2xi1>
/// // %result: [[false, true], [true, true]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#or) for more information.
pub trait OrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`OrOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`OrOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(Or);
mlir_op_trait!(Or, OneResult);
mlir_op_trait!(Or, ZeroRegions);
mlir_op_trait!(Or, ZeroSuccessors);

/// Constructs a new detached/owned [`OrOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`OrOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn or<
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
) -> DetachedOrOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.or", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::or`")
}

/// StableHLO [`Operation`] that performs an element-wise logical `xor` operation on boolean tensors and a bitwise `xor`
/// operation on integer tensors. The output shape is determined by [broadcasting](https://openxla.org/xla/broadcasting)
/// the shapes of the two input tensors.
///
/// # Example
///
/// The following are examples of [`XorOperation`]s represented using their [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [[1, 2], [3, 4]]
/// // %rhs: [[5, 6], [7, 8]]
/// %result = stablehlo.xor %lhs, %rhs : tensor<2x2xi32>
/// // %result: [[4, 4], [4, 12]]
///
/// // %lhs: [[false, false], [true, true]]
/// // %rhs: [[false, true], [false, true]]
/// %result = stablehlo.xor %lhs, %rhs : tensor<2x2xi1>
/// // %result: [[false, true], [true, false]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#xor) for more information.
pub trait XorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`XorOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`XorOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(Xor);
mlir_op_trait!(Xor, OneResult);
mlir_op_trait!(Xor, ZeroRegions);
mlir_op_trait!(Xor, ZeroSuccessors);

/// Constructs a new detached/owned [`XorOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`XorOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn xor<
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
) -> DetachedXorOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.xor", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::xor`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, OneOperand, OneResult, Operation, Size, Value};

    use super::{AndOperation, OrOperation, XorOperation, and, not, or, xor};

    #[test]
    fn test_not() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let tensor_type = context.tensor_type(i8_type, &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = not(input, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "not_test",
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
                  func.func @not_test(%arg0: tensor<2xi8>) -> tensor<2xi8> {
                    %0 = stablehlo.not %arg0 : tensor<2xi8>
                    return %0 : tensor<2xi8>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_and() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = and(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "and_test",
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
                  func.func @and_test(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
                    %0 = stablehlo.and %arg0, %arg1 : tensor<2x2xi32>
                    return %0 : tensor<2x2xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_or() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = or(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "or_test",
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
                  func.func @or_test(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
                    %0 = stablehlo.or %arg0, %arg1 : tensor<2x2xi32>
                    return %0 : tensor<2x2xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_xor() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = xor(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "xor_test",
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
                  func.func @xor_test(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
                    %0 = stablehlo.xor %arg0, %arg1 : tensor<2x2xi32>
                    return %0 : tensor<2x2xi32>
                  }
                }
            "},
        );
    }
}
