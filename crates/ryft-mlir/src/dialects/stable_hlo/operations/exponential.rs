use crate::{
    DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Value, ValueRef, mlir_op, mlir_op_trait,
};

use super::{Accuracy, HasAccuracy, RESULT_ACCURACY_ATTRIBUTE};

/// StableHLO [`Operation`] that performs element-wise exponentiation of two tensors. The operation computes the value
/// of [`PowerOperation::lhs`] raised to the power of [`PowerOperation::rhs`] for corresponding elements from the two
/// input tensors to produce its output tensor. The output shape is determined by
/// [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input tensors.
/// The specific semantics of this operation depend on the element type:
///
///   - **Integers**: Performs integer exponentiation.
///   - **Floating-Point**: Applies the IEEE-754 `pow` operation.
///   - **Complex Numbers**: Computes complex exponentiation.
///   - **Quantized Types**: Dequantizes the inputs, applies `pow`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`PowerOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [-2.0, -0.0, -36.0, 5.0, 3.0, 10000.0]
/// // %rhs: [2.0, 2.0, 1.1, 2.0, -1.0, 10.0]
/// %result = stablehlo.power %lhs, %rhs : tensor<6xf64>
/// // %result: [4.0, 0.0, -nan, 25.0, 0.333333343, inf]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#power) for more information.
pub trait PowerOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`PowerOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`PowerOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(Power);
mlir_op_trait!(Power, OneResult);
mlir_op_trait!(Power, ZeroRegions);
mlir_op_trait!(Power, ZeroSuccessors);

/// Constructs a new detached/owned [`PowerOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`PowerOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn power<
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
) -> DetachedPowerOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.power", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::power`")
}

/// StableHLO [`Operation`] that computes the square root of each element of a tensor.
/// The specific semantics of this operation depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `squareRoot` function.
///   - **Complex Numbers**: Computes the complex square root.
///   - **Quantized Types**: Dequantizes the input, applies `squareRoot`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`SqrtOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [[0.0, 1.0], [4.0, 9.0]]
/// %result = stablehlo.sqrt %operand : tensor<2x2xf32>
/// // %result: [[0.0, 1.0], [2.0, 3.0]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#sqrt) for more information.
pub trait SqrtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Sqrt);
mlir_op_trait!(Sqrt, OneOperand);
mlir_op_trait!(Sqrt, OneResult);
mlir_op_trait!(Sqrt, ZeroRegions);
mlir_op_trait!(Sqrt, ZeroSuccessors);
mlir_op_trait!(Sqrt, @local HasAccuracy);

/// Constructs a new detached/owned [`SqrtOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`SqrtOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn sqrt<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    accuracy: Accuracy,
    location: L,
) -> DetachedSqrtOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.sqrt", location)
        .add_operand(input)
        .add_attribute(RESULT_ACCURACY_ATTRIBUTE, location.context().stable_hlo_accuracy(accuracy))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::sqrt`")
}

/// StableHLO [`Operation`] that computes the reciprocal square root (i.e., `1 / sqrt(x)`) of each element of a tensor.
/// The specific semantics of this operation depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `rSqrt` function.
///   - **Complex Numbers**: Computes the complex reciprocal square root.
///   - **Quantized Types**: Dequantizes the input, applies `rSqrt`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`RsqrtOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [[1.0, 4.0], [9.0, 25.0]]
/// %result = stablehlo.rsqrt %operand : tensor<2x2xf32>
/// // %result: [[1.0, 0.5], [0.33333343, 0.2]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#rsqrt) for more information.
pub trait RsqrtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Rsqrt);
mlir_op_trait!(Rsqrt, OneOperand);
mlir_op_trait!(Rsqrt, OneResult);
mlir_op_trait!(Rsqrt, ZeroRegions);
mlir_op_trait!(Rsqrt, ZeroSuccessors);
mlir_op_trait!(Rsqrt, @local HasAccuracy);

/// Constructs a new detached/owned [`RsqrtOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`RsqrtOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn rsqrt<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    accuracy: Accuracy,
    location: L,
) -> DetachedRsqrtOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.rsqrt", location)
        .add_operand(input)
        .add_attribute(RESULT_ACCURACY_ATTRIBUTE, location.context().stable_hlo_accuracy(accuracy))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::rsqrt`")
}

/// StableHLO [`Operation`] that computes the cubic root of each element of a tensor.
/// The specific semantics of this operation depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `rootn(x, 3)` function, which computes the cubic root.
///   - **Complex Numbers**: Computes the complex cubic root.
///   - **Quantized Types**: Dequantizes the input, applies `rootn(x, 3)`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`CbrtOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [0.0, 1.0, 8.0, 27.0]
/// %result = stablehlo.cbrt %operand : tensor<4xf64>
/// // %result: [0.0, 1.0, 2.0, 3.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#cbrt) for more information.
pub trait CbrtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Cbrt);
mlir_op_trait!(Cbrt, OneOperand);
mlir_op_trait!(Cbrt, OneResult);
mlir_op_trait!(Cbrt, ZeroRegions);
mlir_op_trait!(Cbrt, ZeroSuccessors);
mlir_op_trait!(Cbrt, @local HasAccuracy);

/// Constructs a new detached/owned [`CbrtOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`CbrtOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn cbrt<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    accuracy: Accuracy,
    location: L,
) -> DetachedCbrtOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.cbrt", location)
        .add_operand(input)
        .add_attribute(RESULT_ACCURACY_ATTRIBUTE, location.context().stable_hlo_accuracy(accuracy))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::cbrt`")
}

/// StableHLO [`Operation`] that computes the exponential of each element of a tensor.
/// The specific semantics of this operation depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `exp` function.
///   - **Complex Numbers**: Computes the complex exponential.
///   - **Quantized Types**: Dequantizes the input, applies `exp`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of an [`ExponentialOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [[0.0, 1.0], [2.0, 3.0]]
/// %result = stablehlo.exponential %operand : tensor<2x2xf64>
/// // %result: [[1.0, 2.7182818284590451], [7.3890560989306504, 20.085536923187668]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#exponential)
/// for more information.
pub trait ExponentialOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Exponential);
mlir_op_trait!(Exponential, OneOperand);
mlir_op_trait!(Exponential, OneResult);
mlir_op_trait!(Exponential, ZeroRegions);
mlir_op_trait!(Exponential, ZeroSuccessors);
mlir_op_trait!(Exponential, @local HasAccuracy);

/// Constructs a new detached/owned [`ExponentialOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ExponentialOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn exponential<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    accuracy: Accuracy,
    location: L,
) -> DetachedExponentialOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.exponential", location)
        .add_operand(input)
        .add_attribute(RESULT_ACCURACY_ATTRIBUTE, location.context().stable_hlo_accuracy(accuracy))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::exponential`")
}

/// StableHLO [`Operation`] that computes the exponential of the input minus one (i.e., `exp(x) - 1`) for each element
/// of a tensor. The specific semantics of this operation depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `expm1` function.
///   - **Complex Numbers**: Computes the complex exponential minus one.
///   - **Quantized Types**: Dequantizes the input, applies `expm1`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of an [`ExponentialMinusOneOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [0.0, 1.0]
/// %result = stablehlo.exponential_minus_one %operand : tensor<2xf64>
/// // %result: [0.0, 1.71828187]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#exponential_minus_one)
/// for more information.
pub trait ExponentialMinusOneOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(ExponentialMinusOne);
mlir_op_trait!(ExponentialMinusOne, OneOperand);
mlir_op_trait!(ExponentialMinusOne, OneResult);
mlir_op_trait!(ExponentialMinusOne, ZeroRegions);
mlir_op_trait!(ExponentialMinusOne, ZeroSuccessors);
mlir_op_trait!(ExponentialMinusOne, @local HasAccuracy);

/// Constructs a new detached/owned [`ExponentialMinusOneOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ExponentialMinusOneOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn exponential_minus_one<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    accuracy: Accuracy,
    location: L,
) -> DetachedExponentialMinusOneOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.exponential_minus_one", location)
        .add_operand(input)
        .add_attribute(RESULT_ACCURACY_ATTRIBUTE, location.context().stable_hlo_accuracy(accuracy))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::exponential_minus_one`")
}

/// StableHLO [`Operation`] that computes the logarithm of each element of a tensor.
/// The specific semantics of this operation depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `log` function.
///   - **Complex Numbers**: Computes the complex logarithm.
///   - **Quantized Types**: Dequantizes the input, applies `log`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`LogOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [[1.0, 2.0], [3.0, 4.0]]
/// %result = stablehlo.log %operand : tensor<2x2xf64>
/// // %result: [[0.0, 0.69314718055994529], [1.0986122886681098, 1.3862943611198906]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#log) for more information.
pub trait LogOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Log);
mlir_op_trait!(Log, OneOperand);
mlir_op_trait!(Log, OneResult);
mlir_op_trait!(Log, ZeroRegions);
mlir_op_trait!(Log, ZeroSuccessors);
mlir_op_trait!(Log, @local HasAccuracy);

/// Constructs a new detached/owned [`LogOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`LogOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn log<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    accuracy: Accuracy,
    location: L,
) -> DetachedLogOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.log", location)
        .add_operand(input)
        .add_attribute(RESULT_ACCURACY_ATTRIBUTE, location.context().stable_hlo_accuracy(accuracy))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::log`")
}

/// StableHLO [`Operation`] that computes the logarithm of one plus the input (i.e., `log(1 + x)`) for each element
/// of a tensor. The specific semantics of this operation depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `log1p` function.
///   - **Complex Numbers**: Computes the complex logarithm of one plus the input.
///   - **Quantized Types**: Dequantizes the input, applies `log1p`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`LogPlusOneOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [0.0, -0.999, 7.0, 6.38905621, 15.0]
/// %result = stablehlo.log_plus_one %operand : tensor<5xf64>
/// // %result: [0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#log_plus_one)
/// for more information.
pub trait LogPlusOneOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(LogPlusOne);
mlir_op_trait!(LogPlusOne, OneOperand);
mlir_op_trait!(LogPlusOne, OneResult);
mlir_op_trait!(LogPlusOne, ZeroRegions);
mlir_op_trait!(LogPlusOne, ZeroSuccessors);
mlir_op_trait!(LogPlusOne, @local HasAccuracy);

/// Constructs a new detached/owned [`LogPlusOneOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`LogPlusOneOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn log_plus_one<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    accuracy: Accuracy,
    location: L,
) -> DetachedLogPlusOneOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.log_plus_one", location)
        .add_operand(input)
        .add_attribute(RESULT_ACCURACY_ATTRIBUTE, location.context().stable_hlo_accuracy(accuracy))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::log_plus_one`")
}

/// StableHLO [`Operation`] that computes the logistic/sigmoid (i.e., `1 / (1 + exp(-x))`) for each element
/// of a tensor. The specific semantics of this operation depend on the element type:
///
///   - **Floating-Point**: Computes the IEEE-754 logistic function (sigmoid), `division(1, addition(1, exp(-x)))`.
///   - **Complex Numbers**: Computes the complex logistic function.
///   - **Quantized Types**: Dequantizes the input, applies the logistic function, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`LogisticOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [[0.0, 1.0], [2.0, 3.0]]
/// %result = stablehlo.logistic %operand : tensor<2x2xf64>
/// // %result: [[0.5, 0.73105858], [0.88079708, 0.95257413]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#logistic) for more information.
pub trait LogisticOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Logistic);
mlir_op_trait!(Logistic, OneOperand);
mlir_op_trait!(Logistic, OneResult);
mlir_op_trait!(Logistic, ZeroRegions);
mlir_op_trait!(Logistic, ZeroSuccessors);
mlir_op_trait!(Logistic, @local HasAccuracy);

/// Constructs a new detached/owned [`LogisticOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`LogisticOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn logistic<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    accuracy: Accuracy,
    location: L,
) -> DetachedLogisticOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.logistic", location)
        .add_operand(input)
        .add_attribute(RESULT_ACCURACY_ATTRIBUTE, location.context().stable_hlo_accuracy(accuracy))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::logistic`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, OneOperand, OneResult, Operation, Size, Value};

    use super::{
        Accuracy, HasAccuracy, PowerOperation, cbrt, exponential, exponential_minus_one, log, log_plus_one, logistic,
        power, rsqrt, sqrt,
    };

    #[test]
    fn test_power() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Dynamic, Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = power(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "power_test",
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
                  func.func @power_test(%arg0: tensor<?x3xf32>, %arg1: tensor<?x3xf32>) -> tensor<?x3xf32> {
                    %0 = stablehlo.power %arg0, %arg1 : tensor<?x3xf32>
                    return %0 : tensor<?x3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_sqrt() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = sqrt(input, Accuracy::Highest, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "sqrt_test",
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
                  func.func @sqrt_test(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.sqrt %arg0 {\
                      result_accuracy = #stablehlo.result_accuracy<\
                        mode = #stablehlo.result_accuracy_mode<HIGHEST>\
                      >\
                    } : tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_rsqrt() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = rsqrt(input, Accuracy::Default, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.accuracy(), Accuracy::Default);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "rsqrt_test",
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
                  func.func @rsqrt_test(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.rsqrt %arg0 : tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_cbrt() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();
        let tensor_type = context.tensor_type(f64_type, &[Size::Static(4)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = cbrt(input, Accuracy::Default, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.accuracy(), Accuracy::Default);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "cbrt_test",
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
                  func.func @cbrt_test(%arg0: tensor<4xf64>) -> tensor<4xf64> {
                    %0 = stablehlo.cbrt %arg0 : tensor<4xf64>
                    return %0 : tensor<4xf64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_exponential() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = exponential(input, Accuracy::Default, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.accuracy(), Accuracy::Default);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "exponential_test",
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
                  func.func @exponential_test(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.exponential %arg0 : tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_exponential_minus_one() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = exponential_minus_one(input, Accuracy::Default, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.accuracy(), Accuracy::Default);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "exponential_minus_one_test",
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
                  func.func @exponential_minus_one_test(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.exponential_minus_one %arg0 : tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_log() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = log(
                input,
                Accuracy::Tolerance { absolute_tolerance: 1e-5, relative_tolerance: 1e-3, units_of_least_precision: 3 },
                location,
            );
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "log_test",
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
                  func.func @log_test(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.log %arg0 {\
                      result_accuracy = #stablehlo.result_accuracy<\
                        atol = 1.000000e-05, \
                        rtol = 1.000000e-03, \
                        ulps = 3, \
                        mode = #stablehlo.result_accuracy_mode<TOLERANCE>\
                      >\
                    } : tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_log_plus_one() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = log_plus_one(input, Accuracy::Default, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.accuracy(), Accuracy::Default);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "log_plus_one_test",
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
                  func.func @log_plus_one_test(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.log_plus_one %arg0 : tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_logistic() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = logistic(input, Accuracy::Default, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.accuracy(), Accuracy::Default);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "logistic_test",
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
                  func.func @logistic_test(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.logistic %arg0 : tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "},
        );
    }
}
