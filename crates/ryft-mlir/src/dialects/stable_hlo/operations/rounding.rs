use crate::{DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Value, mlir_op, mlir_op_trait};

/// StableHLO [`Operation`] that rounds up the elements of its input tensor.
/// The specific semantics depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `roundToIntegralTowardPositive` function, which rounds towards
///     positive infinity (i.e., returns the smallest integer value that is greater than or equal to the input).
///   - **Quantized Types**: Dequantizes the input, applies `roundToIntegralTowardPositive`, and then requantizes
///     the result.
///
/// # Example
///
/// The following is an example of a [`CeilOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [-0.8166, -0.2530, 0.2530, 0.8166, 2.0]
/// %result = stablehlo.ceil %operand : tensor<5xf32>
/// // %result: [-0.0, -0.0, 1.0, 1.0, 2.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#ceil) for more information.
pub trait CeilOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Ceil);
mlir_op_trait!(Ceil, OneOperand);
mlir_op_trait!(Ceil, OneResult);
mlir_op_trait!(Ceil, ZeroRegions);
mlir_op_trait!(Ceil, ZeroSuccessors);

/// Constructs a new detached/owned [`CeilOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`CeilOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn ceil<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedCeilOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.ceil", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::ceil`")
}

/// StableHLO [`Operation`] that rounds down the elements of its input tensor.
/// The specific semantics depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `roundToIntegralTowardNegative` function, which rounds towards
///     negative infinity (i.e., returns the largest integer value that is less than or equal to the input).
///   - **Quantized Types**: Dequantizes the input, applies `roundToIntegralTowardNegative`, and then requantizes
///     the result.
///
/// # Example
///
/// The following is an example of a [`FloorOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [-0.8166, -0.2530, 0.2530, 0.8166, 2.0]
/// %result = stablehlo.floor %operand : tensor<5xf32>
/// // %result: [-1.0, -1.0, 0.0, 0.0, 2.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#floor) for more information.
pub trait FloorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Floor);
mlir_op_trait!(Floor, OneOperand);
mlir_op_trait!(Floor, OneResult);
mlir_op_trait!(Floor, ZeroRegions);
mlir_op_trait!(Floor, ZeroSuccessors);

/// Constructs a new detached/owned [`FloorOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`FloorOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn floor<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedFloorOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.floor", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::floor`")
}

/// StableHLO [`Operation`] that rounds the elements of its input tensor to their nearest integer, breaking ties by
/// rounding _away_ from zero. The specific semantics depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `roundToIntegralTiesToAway` function.
///   - **Quantized Types**: Dequantizes the input, applies `roundToIntegralTiesToAway`,
///     and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`RoundWithAwayFromZeroTieBreakOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [-2.5, 0.4, 0.5, 0.6, 2.5]
/// %result = stablehlo.round_nearest_afz %operand : tensor<5xf64>
/// // %result: [-3.0, 0.0, 1.0, 1.0, 3.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#round_nearest_afz)
/// for more information.
pub trait RoundWithAwayFromZeroTieBreakOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(RoundWithAwayFromZeroTieBreak);
mlir_op_trait!(RoundWithAwayFromZeroTieBreak, OneOperand);
mlir_op_trait!(RoundWithAwayFromZeroTieBreak, OneResult);
mlir_op_trait!(RoundWithAwayFromZeroTieBreak, ZeroRegions);
mlir_op_trait!(RoundWithAwayFromZeroTieBreak, ZeroSuccessors);

/// Constructs a new detached/owned [`RoundWithAwayFromZeroTieBreakOperation`] at the specified [`Location`]. Refer to
/// the documentation of [`RoundWithAwayFromZeroTieBreakOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn round_with_away_from_zero_tie_break<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedRoundWithAwayFromZeroTieBreakOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.round_nearest_afz", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::round_nearest_afz`")
}

/// StableHLO [`Operation`] that rounds the elements of its input tensor to their nearest integer, breaking ties by
/// rounding to the nearest even integer. The specific semantics depend on the element type:
///
///   - **Floating-Point**: Implements the IEEE-754 `roundToIntegralTiesToEven` function.
///   - **Quantized Types**: Dequantizes the input, applies `roundToIntegralTiesToEven`,
///     and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`RoundWithNearestEvenTieBreakOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [-2.5, 0.4, 0.5, 0.6, 2.5]
/// %result = stablehlo.round_nearest_even %operand : tensor<6xf32>
/// // %result: [-2.0, 0.0, 0.0, 1.0, 2.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#round_nearest_even)
/// for more information.
pub trait RoundWithNearestEvenTieBreakOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(RoundWithNearestEvenTieBreak);
mlir_op_trait!(RoundWithNearestEvenTieBreak, OneOperand);
mlir_op_trait!(RoundWithNearestEvenTieBreak, OneResult);
mlir_op_trait!(RoundWithNearestEvenTieBreak, ZeroRegions);
mlir_op_trait!(RoundWithNearestEvenTieBreak, ZeroSuccessors);

/// Constructs a new detached/owned [`RoundWithNearestEvenTieBreakOperation`] at the specified [`Location`]. Refer to
/// the documentation of [`RoundWithNearestEvenTieBreakOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn round_with_nearest_even_tie_break<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedRoundWithNearestEvenTieBreakOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.round_nearest_even", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::round_nearest_even`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, OneOperand, OneResult, Operation, Size, Type, Value};

    use super::{ceil, floor, round_with_away_from_zero_tie_break, round_with_nearest_even_tie_break};

    #[test]
    fn test_ceil() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(5)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input_value = block.argument(0).unwrap();
            let op = ceil(input_value, location);
            assert_eq!(op.input(), input_value);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "ceil_test",
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
                  func.func @ceil_test(%arg0: tensor<5xf32>) -> tensor<5xf32> {
                    %0 = stablehlo.ceil %arg0 : tensor<5xf32>
                    return %0 : tensor<5xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_floor() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(5)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input_value = block.argument(0).unwrap();
            let op = floor(input_value, location);
            assert_eq!(op.input(), input_value);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "floor_test",
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
                  func.func @floor_test(%arg0: tensor<5xf32>) -> tensor<5xf32> {
                    %0 = stablehlo.floor %arg0 : tensor<5xf32>
                    return %0 : tensor<5xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_round_with_away_from_zero_tie_break() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(5)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input_value = block.argument(0).unwrap();
            let op = round_with_away_from_zero_tie_break(input_value, location);
            assert_eq!(op.input(), input_value);
            assert_eq!(op.output().r#type(), tensor_type.as_ref());
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "round_nearest_afz_test",
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
                  func.func @round_nearest_afz_test(%arg0: tensor<5xf32>) -> tensor<5xf32> {
                    %0 = stablehlo.round_nearest_afz %arg0 : tensor<5xf32>
                    return %0 : tensor<5xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_round_with_nearest_even_tie_break() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(6)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input_value = block.argument(0).unwrap();
            let op = round_with_nearest_even_tie_break(input_value, location);
            assert_eq!(op.input(), input_value);
            assert_eq!(op.output().r#type(), tensor_type.as_ref());
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "round_nearest_even_test",
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
                  func.func @round_nearest_even_test(%arg0: tensor<6xf32>) -> tensor<6xf32> {
                    %0 = stablehlo.round_nearest_even %arg0 : tensor<6xf32>
                    return %0 : tensor<6xf32>
                  }
                }
            "},
        );
    }
}
