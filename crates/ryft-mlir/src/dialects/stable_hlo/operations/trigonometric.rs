use crate::{
    DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Value, ValueRef, mlir_op, mlir_op_trait,
};

use super::{Accuracy, HasAccuracy, RESULT_ACCURACY_ATTRIBUTE};

/// StableHLO [`Operation`] that performs element-wise sine computation on a tensor.
/// The specific semantics depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `sin` function.
///   - **Complex Numbers**: Computes the complex sine.
///   - **Quantized Types**: Dequantizes the input, applies `sin`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`SineOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [
/// //            [0.0, 1.57079632],       // [0, π/2]
/// //            [3.14159265, 4.71238898] // [π, 3π/2]
/// //           ]
/// %result = stablehlo.sine %operand : tensor<2x2xf32>
/// // %result: [[0.0, 1.0], [0.0, -1.0]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#sine) for more information.
pub trait SineOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Sine);
mlir_op_trait!(Sine, OneOperand);
mlir_op_trait!(Sine, OneResult);
mlir_op_trait!(Sine, ZeroRegions);
mlir_op_trait!(Sine, ZeroSuccessors);
mlir_op_trait!(Sine, @local HasAccuracy);

/// Constructs a new detached/owned [`SineOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`SineOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn sine<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    accuracy: Accuracy,
    location: L,
) -> DetachedSineOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.sine", location)
        .add_operand(input)
        .add_attribute(RESULT_ACCURACY_ATTRIBUTE, location.context().stable_hlo_accuracy(accuracy))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::sine`")
}

/// StableHLO [`Operation`] that performs element-wise cosine computation on a tensor.
/// The specific semantics depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `cos` function.
///   - **Complex Numbers**: Computes the complex cosine.
///   - **Quantized Types**: Dequantizes the input, applies `cosine`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`CosineOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [
/// //            [0.0, 1.57079632],       // [0, π/2]
/// //            [3.14159265, 4.71238898] // [π, 3π/2]
/// //           ]
/// %result = stablehlo.cosine %operand : tensor<2x2xf32>
/// // %result: [[1.0, 0.0], [-1.0, 0.0]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#cosine) for more information.
pub trait CosineOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Cosine);
mlir_op_trait!(Cosine, OneOperand);
mlir_op_trait!(Cosine, OneResult);
mlir_op_trait!(Cosine, ZeroRegions);
mlir_op_trait!(Cosine, ZeroSuccessors);
mlir_op_trait!(Cosine, @local HasAccuracy);

/// Constructs a new detached/owned [`CosineOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`CosineOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn cosine<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    accuracy: Accuracy,
    location: L,
) -> DetachedCosineOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.cosine", location)
        .add_operand(input)
        .add_attribute(RESULT_ACCURACY_ATTRIBUTE, location.context().stable_hlo_accuracy(accuracy))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::cosine`")
}

/// StableHLO [`Operation`] that performs element-wise tangent computation on a tensor.
/// The specific semantics depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `tan` function.
///   - **Complex Numbers**: Computes the complex tangent.
///   - **Quantized Types**: Dequantizes the input, applies `tan`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`TanOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [
/// //            [0.0, 1.57079632],       // [0, π/2]
/// //            [3.14159265, 4.71238898] // [π, 3π/2]
/// //           ]
/// %result = stablehlo.tan %operand : tensor<2x2xf64>
/// // %result: [
/// //           [0.0, 1.63312e+16],
/// //           [0.0, 5.44375e+15]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#tangent) for more information.
pub trait TanOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Tan);
mlir_op_trait!(Tan, OneOperand);
mlir_op_trait!(Tan, OneResult);
mlir_op_trait!(Tan, ZeroRegions);
mlir_op_trait!(Tan, ZeroSuccessors);
mlir_op_trait!(Tan, @local HasAccuracy);

/// Constructs a new detached/owned [`TanOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`TanOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn tan<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    accuracy: Accuracy,
    location: L,
) -> DetachedTanOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.tan", location)
        .add_operand(input)
        .add_attribute(RESULT_ACCURACY_ATTRIBUTE, location.context().stable_hlo_accuracy(accuracy))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::tan`")
}

/// StableHLO [`Operation`] that performs element-wise hyperbolic tangent computation on a tensor.
/// The specific semantics depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `tanh` function.
///   - **Complex Numbers**: Computes the complex hyperbolic tangent.
///   - **Quantized Types**: Dequantizes the input, applies `tanh`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`TanhOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [-1.0, 0.0, 1.0]
/// %result = stablehlo.tanh %operand : tensor<3xf32>
/// // %result: [-0.76159416, 0.0, 0.76159416]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#tanh) for more information.
pub trait TanhOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Tanh);
mlir_op_trait!(Tanh, OneOperand);
mlir_op_trait!(Tanh, OneResult);
mlir_op_trait!(Tanh, ZeroRegions);
mlir_op_trait!(Tanh, ZeroSuccessors);
mlir_op_trait!(Tanh, @local HasAccuracy);

/// Constructs a new detached/owned [`TanhOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`TanhOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn tanh<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    accuracy: Accuracy,
    location: L,
) -> DetachedTanhOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.tanh", location)
        .add_operand(input)
        .add_attribute(RESULT_ACCURACY_ATTRIBUTE, location.context().stable_hlo_accuracy(accuracy))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::tanh`")
}

/// StableHLO [`Operation`] that performs element-wise arc tangent of two values computation on two tensors. The
/// operation computes the arc tangent using the signs of both input values to determine the correct quadrant. The
/// output shape is determined by [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input
/// tensors. The specific semantics of the operation depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `atan2` operation.
///   - **Complex Numbers**: Computes the complex arc tangent.
///   - **Quantized Types**: Dequantizes the inputs, applies `atan2`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of an [`Atan2Operation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [0.0, 1.0, -1.0]
/// // %rhs: [0.0, 0.0, 0.0]
/// %result = stablehlo.atan2 %lhs, %rhs : tensor<4xf64>
/// // %result: [0.0, 1.57079637, -1.57079637] // [0.0, π/2, -π/2]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#atan2) for more information.
pub trait Atan2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`Atan2Operation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`Atan2Operation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(Atan2);
mlir_op_trait!(Atan2, OneResult);
mlir_op_trait!(Atan2, ZeroRegions);
mlir_op_trait!(Atan2, ZeroSuccessors);

/// Constructs a new detached/owned [`Atan2Operation`] at the specified [`Location`]. Refer to the
/// documentation of [`Atan2Operation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn atan2<
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
) -> DetachedAtan2Operation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.atan2", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::atan2`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, OneOperand, OneResult, Operation, Size, Value};

    use super::{Accuracy, Atan2Operation, HasAccuracy, atan2, cosine, sine, tan, tanh};

    #[test]
    fn test_sine() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = sine(input, Accuracy::Default, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.accuracy(), Accuracy::Default);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "sine_test",
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
                  func.func @sine_test(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.sine %arg0 : tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_cosine() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = cosine(input, Accuracy::Default, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.accuracy(), Accuracy::Default);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "cosine_test",
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
                  func.func @cosine_test(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.cosine %arg0 : tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_tan() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();
        let tensor_type = context.tensor_type(f64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = tan(input, Accuracy::Default, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.accuracy(), Accuracy::Default);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "tan_test",
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
                  func.func @tan_test(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
                    %0 = stablehlo.tan %arg0 : tensor<2x2xf64>
                    return %0 : tensor<2x2xf64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_tanh() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = tanh(input, Accuracy::Default, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.accuracy(), Accuracy::Default);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "tanh_test",
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
                  func.func @tanh_test(%arg0: tensor<3xf32>) -> tensor<3xf32> {
                    %0 = stablehlo.tanh %arg0 : tensor<3xf32>
                    return %0 : tensor<3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_atan2() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();
        let tensor_type = context.tensor_type(f64_type, &[Size::Static(4)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = atan2(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "atan2_test",
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
                  func.func @atan2_test(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
                    %0 = stablehlo.atan2 %arg0, %arg1 : tensor<4xf64>
                    return %0 : tensor<4xf64>
                  }
                }
            "},
        );
    }
}
