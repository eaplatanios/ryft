use crate::{
    DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Value, ValueRef, mlir_op, mlir_op_trait,
};

/// StableHLO [`Operation`] that performs element-wise sign extraction on a tensor. The operation computes the sign
/// of each element in the input tensor and produces a resulting tensor with the same shape. The specific semantics
/// depend on the element type:
///
///   - **Integers**: Returns -1 for negative values, 0 for zero, and 1 for positive values.
///   - **Floating-Point**: Implements the IEEE-754 `sign` function. For NaN, returns NaN.
///   - **Complex Numbers**: Computes the complex sign (i.e., `x / |x|` for non-zero `x`, 0 for zero, and NaN
///     if either the real or the imaginary parts of the input number is NaN-valued).
///   - **Quantized Types**: Dequantizes the input, applies `sign`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`SignOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // Logical values: +NaN, -1.0, -0.0, +0.0, 1.0
/// // %operand: [0x7FFFFFFFFFFFFFFF, -1.0, -0.0, 0.0, 1.0]
/// %result = stablehlo.sign %operand : tensor<5xf64>
/// // Logical values: +NaN, -1.0, -0.0, +0.0, 1.0
/// // %result: [0x7FFFFFFFFFFFFFFF, -1.0, -0.0, 0.0, 1.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#sign) for more information.
pub trait SignOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Sign);
mlir_op_trait!(Sign, OneOperand);
mlir_op_trait!(Sign, OneResult);
mlir_op_trait!(Sign, ZeroRegions);

/// Constructs a new detached/owned [`SignOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`SignOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn sign<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedSignOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.sign", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::sign`")
}

/// StableHLO [`Operation`] that performs element-wise absolute value computation on a tensor. The operation computes
/// the absolute value of each element in the input tensor and produces a resulting tensor with the same shape. The
/// specific semantics depend on the element type:
///
///   - **Signed Integers**: Computes the integer modulus (absolute value).
///   - **Floating-Point**: Applies the IEEE-754 `abs` function, which returns the magnitude without the sign bit.
///   - **Complex Numbers**: Computes the complex modulus (magnitude; i.e., `sqrt(real^2 + imag^2)`). Note that for
///     complex inputs, the resulting element type is the underlying floating-point type rather than the complex type.
///   - **Quantized Types**: Dequantizes the input, applies `abs`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of an [`AbsOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [-2, 0, 2]
/// %result = stablehlo.abs %operand : tensor<3xi32>
/// // %result: [2, 0, 2]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#abs) for more information.
pub trait AbsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Abs);
mlir_op_trait!(Abs, OneOperand);
mlir_op_trait!(Abs, OneResult);
mlir_op_trait!(Abs, ZeroRegions);
mlir_op_trait!(Abs, ZeroSuccessors);

/// Constructs a new detached/owned [`AbsOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`AbsOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn abs<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedAbsOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.abs", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::abs`")
}

/// StableHLO [`Operation`] that performs element-wise negation on a tensor. The operation negates each element in the
/// input tensor and produces a resulting tensor with the same shape. The specific semantics depend on the element type:
///
///   - **Signed Integers**: Computes integer negation.
///   - **Unsigned Integers**: Bitwise casts to the signed integer type, negates, and bitwise casts back.
///   - **Floating-Point**: Implements the IEEE-754 `negate` function.
///   - **Complex Numbers**: Computes the complex negation.
///   - **Quantized Types**: Dequantizes the input, applies `negate`, and then requantizes the result.
///
/// # Example
///
/// The following are examples of [`NegateOperation`]s represented using their [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [0, -2, 2]
/// %result = stablehlo.negate %operand : tensor<3xi32>
/// // %result: [0, 2, -2]
///
/// // %operand: (2.5, 0.0)
/// %result = stablehlo.negate %operand : tensor<1xcomplex<f32>>
/// // %result: [-2.5, -0.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#negate) for more information.
pub trait NegateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Negate);
mlir_op_trait!(Negate, OneOperand);
mlir_op_trait!(Negate, OneResult);
mlir_op_trait!(Negate, ZeroRegions);
mlir_op_trait!(Negate, ZeroSuccessors);

/// Constructs a new detached/owned [`NegateOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`NegateOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn negate<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedNegateOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.negate", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::negate`")
}

/// StableHLO [`Operation`] that performs element-wise addition of two tensors. The output shape is determined by
/// [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input tensors. The specific semantics
/// depend on the element type:
///
///   - **Booleans**: Performs logical OR.
///   - **Integers**: Performs standard integer addition.
///   - **Floating-Point**: Applies the IEEE-754 `addition` operation.
///   - **Complex Numbers**: Computes complex addition.
///   - **Quantized Types**: Dequantizes the inputs, applies `addition`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of an [`AddOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [[1, 2], [3, 4]]
/// // %rhs: [[5, 6], [7, 8]]
/// %result = stablehlo.add %lhs, %rhs : tensor<2x2xi32>
/// // %result: [[6, 8], [10, 12]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#add) for more information.
pub trait AddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`AddOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`AddOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(Add);
mlir_op_trait!(Add, OneResult);
mlir_op_trait!(Add, ZeroRegions);
mlir_op_trait!(Add, ZeroSuccessors);

/// Constructs a new detached/owned [`AddOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`AddOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn add<
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
) -> DetachedAddOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.add", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::add`")
}

/// StableHLO [`Operation`] that performs element-wise subtraction of two tensors. The output shape is determined by
/// [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input tensors. The specific semantics
/// depend on the element type:
///
///   - **Integers**: Performs standard integer subtraction.
///   - **Floating-Point**: Applies the IEEE-754 `subtraction` operation.
///   - **Complex Numbers**: Computes complex subtraction.
///   - **Quantized Types**: Dequantizes the inputs, applies `subtraction`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`SubtractOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [[6, 8], [10, 12]]
/// // %rhs: [[5, 6], [7, 8]]
/// %result = stablehlo.subtract %lhs, %rhs : tensor<2x2xi32>
/// // %result: [[1, 2], [3, 4]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#subtract) for more information.
pub trait SubtractOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`SubtractOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`SubtractOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(Subtract);
mlir_op_trait!(Subtract, OneResult);
mlir_op_trait!(Subtract, ZeroRegions);
mlir_op_trait!(Subtract, ZeroSuccessors);

/// Constructs a new detached/owned [`SubtractOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`SubtractOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn subtract<
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
) -> DetachedSubtractOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.subtract", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::subtract`")
}

/// StableHLO [`Operation`] that performs element-wise multiplication of two tensors. The output shape is determined by
/// [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input tensors. The specific semantics
/// depend on the element type:
///
///   - **Booleans**: Performs logical AND.
///   - **Integers**: Performs standard integer multiplication.
///   - **Floating-Point**: Applies the IEEE-754 `multiplication` operation.
///   - **Complex Numbers**: Computes complex multiplication.
///   - **Quantized Types**: Dequantizes the inputs, applies multiplication, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`MultiplyOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [[1, 2], [3, 4]]
/// // %rhs: [[5, 6], [7, 8]]
/// %result = stablehlo.multiply %lhs, %rhs : tensor<2x2xi32>
/// // %result: [[5, 12], [21, 32]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#multiply) for more information.
pub trait MultiplyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`MultiplyOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`MultiplyOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(Multiply);
mlir_op_trait!(Multiply, OneResult);
mlir_op_trait!(Multiply, ZeroRegions);
mlir_op_trait!(Multiply, ZeroSuccessors);

/// Constructs a new detached/owned [`MultiplyOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`MultiplyOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn multiply<
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
) -> DetachedMultiplyOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.multiply", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::multiply`")
}

/// StableHLO [`Operation`] that performs element-wise division of two tensors. The output shape is determined by
/// [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input tensors. The specific semantics
/// depend on the element type:
///
///   - **Integers**: Performs integer division (truncated towards zero).
///   - **Floating-Point**: Applies the IEEE-754 `division` operation.
///   - **Complex Numbers**: Computes complex division.
///   - **Quantized Types**: Dequantizes the inputs, applies `division`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`DivideOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [17.1, -17.1, 17.1, -17.1]
/// // %rhs: [3.0, 3.0, -3.0, -3.0]
/// %result = stablehlo.divide %lhs, %rhs : tensor<4xf32>
/// // %result: [5.66666651, -5.66666651, -5.66666651, 5.66666651]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#divide) for more information.
pub trait DivideOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`DivideOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`DivideOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(Divide);
mlir_op_trait!(Divide, OneResult);
mlir_op_trait!(Divide, ZeroRegions);
mlir_op_trait!(Divide, ZeroSuccessors);

/// Constructs a new detached/owned [`DivideOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DivideOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn divide<
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
) -> DetachedDivideOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.divide", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::divide`")
}

/// StableHLO [`Operation`] that performs element-wise remainder of dividing two tensors. The output shape is determined
/// by [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input tensors. The specific semantics
/// depend on the element type:
///
///   - **Integers**: Performs integer remainder (sign follows dividend).
///   - **Floating-Point**: Applies the IEEE-754 `remainder` operation.
///   - **Quantized Types**: Dequantizes the inputs, applies `remainder`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`RemainderOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [17, -17, 17, -17]
/// // %rhs: [3, 3, -3, -3]
/// %result = stablehlo.remainder %lhs, %rhs : tensor<4xi64>
/// // %result: [2, -2, 2, -2]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#remainder) for more information.
pub trait RemainderOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`RemainderOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`RemainderOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(Remainder);
mlir_op_trait!(Remainder, OneResult);
mlir_op_trait!(Remainder, ZeroRegions);
mlir_op_trait!(Remainder, ZeroSuccessors);

/// Constructs a new detached/owned [`RemainderOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`RemainderOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn remainder<
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
) -> DetachedRemainderOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.remainder", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::remainder`")
}

/// StableHLO [`Operation`] that performs element-wise check for finiteness on a tensor. The operation checks whether
/// each element in the input tensor is finite (i.e., not infinity and not NaN) and produces a resulting boolean tensor
/// with the same shape.
///
/// # Example
///
/// The following is an example of an [`IsFiniteOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [0.0, 1.0, inf, -inf, nan]
/// %result = stablehlo.is_finite %operand : (tensor<5xf32>) -> tensor<5xi1>
/// // %result: [true, true, false, false, false]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#is_finite) for more information.
pub trait IsFiniteOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(IsFinite);
mlir_op_trait!(IsFinite, OneOperand);
mlir_op_trait!(IsFinite, OneResult);
mlir_op_trait!(IsFinite, ZeroRegions);
mlir_op_trait!(IsFinite, ZeroSuccessors);

/// Constructs a new detached/owned [`IsFiniteOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`IsFiniteOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn is_finite<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedIsFiniteOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.is_finite", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::is_finite`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, OneOperand, OneResult, Operation, Size, Type, Value};

    use super::{
        AddOperation, DivideOperation, MultiplyOperation, RemainderOperation, SubtractOperation, abs, add, divide,
        is_finite, multiply, negate, remainder, sign, subtract,
    };

    #[test]
    fn test_sign() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = sign(input, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "sign_test",
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
                  func.func @sign_test(%arg0: tensor<3xi32>) -> tensor<3xi32> {
                    %0 = stablehlo.sign %arg0 : tensor<3xi32>
                    return %0 : tensor<3xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_abs() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();
        let complex_type = context.complex_type(f64_type);
        let input_tensor_type = context.tensor_type(complex_type, &[Size::Static(4)], None, location).unwrap();
        let output_tensor_type = context.tensor_type(f64_type, &[Size::Static(4)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = abs(input, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), output_tensor_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "abs_test",
                func::FuncAttributes {
                    arguments: vec![input_tensor_type.into()],
                    results: vec![output_tensor_type.into()],
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
                  func.func @abs_test(%arg0: tensor<4xcomplex<f64>>) -> tensor<4xf64> {
                    %0 = stablehlo.abs %arg0 : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
                    return %0 : tensor<4xf64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_negate() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type =
            context.tensor_type(context.signless_integer_type(32), &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = negate(input, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "negate_test",
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
                  func.func @negate_test(%arg0: tensor<3xi32>) -> tensor<3xi32> {
                    %0 = stablehlo.negate %arg0 : tensor<3xi32>
                    return %0 : tensor<3xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_add() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type = context
            .tensor_type(context.signless_integer_type(32), &[Size::Static(2), Size::Static(2)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = add(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "add_test",
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
                  func.func @add_test(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xi32>
                    return %0 : tensor<2x2xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_subtract() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type = context
            .tensor_type(context.signless_integer_type(32), &[Size::Static(2), Size::Static(2)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = subtract(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "subtract_test",
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
                  func.func @subtract_test(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
                    %0 = stablehlo.subtract %arg0, %arg1 : tensor<2x2xi32>
                    return %0 : tensor<2x2xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_multiply() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type = context
            .tensor_type(context.signless_integer_type(32), &[Size::Static(2), Size::Static(2)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = multiply(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "multiply_test",
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
                  func.func @multiply_test(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
                    %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x2xi32>
                    return %0 : tensor<2x2xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_divide() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type = context
            .tensor_type(context.float32_type(), &[Size::Static(3), Size::Dynamic], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = divide(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "divide_test",
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
                  func.func @divide_test(%arg0: tensor<3x?xf32>, %arg1: tensor<3x?xf32>) -> tensor<3x?xf32> {
                    %0 = stablehlo.divide %arg0, %arg1 : tensor<3x?xf32>
                    return %0 : tensor<3x?xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_remainder() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type =
            context.tensor_type(context.signless_integer_type(32), &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = remainder(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "remainder_test",
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
                  func.func @remainder_test(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi32> {
                    %0 = stablehlo.remainder %arg0, %arg1 : tensor<3xi32>
                    return %0 : tensor<3xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_is_finite() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let input_tensor_type =
            context.tensor_type(context.float32_type(), &[Size::Static(5)], None, location).unwrap();
        let output_tensor_type =
            context.tensor_type(context.signless_integer_type(1), &[Size::Static(5)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = is_finite(input, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), output_tensor_type.as_type_ref());
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "is_finite_test",
                func::FuncAttributes {
                    arguments: vec![input_tensor_type.into()],
                    results: vec![output_tensor_type.into()],
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
                  func.func @is_finite_test(%arg0: tensor<5xf32>) -> tensor<5xi1> {
                    %0 = stablehlo.is_finite %arg0 : (tensor<5xf32>) -> tensor<5xi1>
                    return %0 : tensor<5xi1>
                  }
                }
            "},
        );
    }
}
