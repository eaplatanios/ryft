use crate::{
    Attribute, DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Value, ValueRef, mlir_enum_attribute,
    mlir_op, mlir_op_trait,
};

mlir_enum_attribute!(
    rust_name = ComparisonDirection,
    mlir_name = ComparisonDirection,
    description = "StableHLO comparison direction",
    variants = {
        Equal => "EQ",
        NotEqual => "NE",
        GreaterThanOrEqual => "GE",
        GreaterThan => "GT",
        LessThanOrEqual => "LE",
        LessThan => "LT",
    },
    rust_prefix = stable_hlo,
    mlir_prefix = stablehlo,
    mlir_dialect_handle_constructor = stable_hlo,
);

mlir_enum_attribute!(
    rust_name = ComparisonType,
    mlir_name = ComparisonType,
    description = "StableHLO comparison type",
    variants = {
        Float => "FLOAT",
        TotalOrder => "TOTALORDER",
        Signed => "SIGNED",
        Unsigned => "UNSIGNED",
    },
    rust_prefix = stable_hlo,
    mlir_prefix = stablehlo,
    mlir_dialect_handle_constructor = stable_hlo,
);

/// Name of the [`Attribute`] that is used to store [`CompareOperation::comparison_direction`].
pub const COMPARISON_DIRECTION_ATTRIBUTE: &'static str = "comparison_direction";

/// Name of the [`Attribute`] that is used to store [`CompareOperation::comparison_type`].
pub const COMPARISON_TYPE_ATTRIBUTE: &'static str = "compare_type";

/// StableHLO [`Operation`] that performs element-wise comparison of two tensors. The operation compares
/// corresponding elements from the two input tensors according to [`CompareOperation::comparison_direction`] and
/// [`CompareOperation::comparison_type`], producing a boolean result tensor. The output tensor shape is determined by
/// [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input tensors.
///
/// # Comparison Directions
///
/// [`CompareOperation::comparison_direction`] determines how elements of the two tensors are compared:
///
///   - [`ComparisonDirection::Equal`]: Returns `true` if `lhs` equals `rhs`.
///   - [`ComparisonDirection::NotEqual`]: Returns `true` if `lhs` does not equal `rhs`.
///   - [`ComparisonDirection::GreaterThanOrEqual`]: Returns `true` if `lhs` is greater than or equal to `rhs`.
///   - [`ComparisonDirection::GreaterThan`]: Returns `true` if `lhs` is strictly greater than `rhs`.
///   - [`ComparisonDirection::LessThanOrEqual`]: Returns `true` if `lhs` is less than or equal to `rhs`.
///   - [`ComparisonDirection::LessThan`]: Returns `true` if `lhs` is strictly less than `rhs`.
///
/// # Comparison Types
///
/// [`CompareOperation::comparison_type`] determines how the comparison is performed based on the type of the elements
/// of the two tensors that are being compared:
///
///   - [`ComparisonType::Float`]: For floating-point types, implements IEEE-754 quiet comparison semantics.
///     The comparison properly handles NaN values according to the IEEE-754 specification.
///   - [`ComparisonType::TotalOrder`]: For floating-point types, uses total ordering semantics combining `totalOrder`
///     and `compareQuietEqual` from IEEE-754. This provides a consistent ordering for all values, including NaN.
///   - [`ComparisonType::Signed`]: For signed integer types, performs signed integer comparison.
///   - [`ComparisonType::Unsigned`]: For unsigned integer and boolean types, performs unsigned integer comparison.
///
/// # Type Support
///
/// [`CompareOperation`] supports the following element [`Type`](crate::Type)s:
///
///   - **Boolean**: Element-wise boolean comparison. Note that boolean types are represented as signless
///     [`IntegerTypeRef`](crate::IntegerTypeRef)s with a bit width of 1.
///   - **Integer**: Element-wise integer comparison (signed or unsigned).
///   - **Floating-Point**: Element-wise floating-point comparison with configurable semantics
///     ([`ComparisonType::Float`] or [`ComparisonType::TotalOrder`]).
///   - **Complex Numbers**: Compares complex numbers using lexicographic ordering of `(real, imaginary)` pairs.
///   - **Quantized Types**: Dequantizes the inputs, applies the comparison, and produces a boolean result tensor.
///
/// # Example
///
/// The following is an example of a [`CompareOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [1.0, 3.0]
/// // %rhs: [1.1, 2.9]
/// %result = stablehlo.compare  GE, %arg0, %arg1,  TOTALORDER : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
/// // %result: [false, true]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#compare) for more information.
pub trait CompareOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`CompareOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`CompareOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns the [`ComparisonDirection`] of this [`CompareOperation`].
    fn comparison_direction(&self) -> ComparisonDirection {
        self.as_operation_ref()
            .attribute(COMPARISON_DIRECTION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ComparisonDirectionAttributeRef>())
            .map(|attribute| attribute.value())
            .expect(&format!("`{COMPARISON_DIRECTION_ATTRIBUTE}` attribute not found or had incorrect type"))
    }

    /// Returns the [`ComparisonType`] of this [`CompareOperation`].
    fn comparison_type(&self) -> ComparisonType {
        self.as_operation_ref()
            .attribute(COMPARISON_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ComparisonTypeAttributeRef>())
            .map(|attribute| attribute.value())
            .expect(&format!("`{COMPARISON_TYPE_ATTRIBUTE}` attribute not found or had incorrect type"))
    }
}

mlir_op!(Compare);
mlir_op_trait!(Compare, OneResult);
mlir_op_trait!(Compare, ZeroRegions);
mlir_op_trait!(Compare, ZeroSuccessors);

/// Constructs a new detached/owned [`CompareOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`CompareOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn compare<
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
    comparison_direction: ComparisonDirection,
    comparison_type: ComparisonType,
    location: L,
) -> DetachedCompareOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.compare", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .add_attribute(
            COMPARISON_DIRECTION_ATTRIBUTE,
            location.context().stable_hlo_comparison_direction(comparison_direction),
        )
        .add_attribute(COMPARISON_TYPE_ATTRIBUTE, location.context().stable_hlo_comparison_type(comparison_type))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::compare`")
}

/// StableHLO [`Operation`] that clamps each element of [`ClampOperation::input`] between a [`ClampOperation::min`] and
/// [`ClampOperation::max`] to produce an output tensor. The output tensor shape is determined by
/// [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the three input tensors.
///
/// # Examples
///
/// ```mlir
/// %result = stablehlo.clamp %min, %operand, %max : tensor<3xi32>
/// ```
///
/// Refer to the [StableHLO specification](https://openxla.org/stablehlo/spec#clamp) for more information.
pub trait ClampOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the minimum bound of this [`ClampOperation`].
    fn min(&self) -> ValueRef<'o, 'c, 't> {
        Operation::operand(self, 0).unwrap()
    }

    /// Returns the input (i.e., value to be clamped) of this [`ClampOperation`].
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        Operation::operand(self, 1).unwrap()
    }

    /// Returns the maximum bound of this [`ClampOperation`].
    fn max(&self) -> ValueRef<'o, 'c, 't> {
        Operation::operand(self, 2).unwrap()
    }
}

mlir_op!(Clamp);
mlir_op_trait!(Clamp, OneResult);
mlir_op_trait!(Clamp, ZeroRegions);
mlir_op_trait!(Clamp, ZeroSuccessors);

/// Constructs a new detached/owned [`ClampOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ClampOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn clamp<
    'min,
    'input,
    'max,
    'c: 'min + 'input + 'max,
    't: 'c,
    Min: Value<'min, 'c, 't>,
    Input: Value<'input, 'c, 't>,
    Max: Value<'max, 'c, 't>,
    L: Location<'c, 't>,
>(
    min: Min,
    input: Input,
    max: Max,
    location: L,
) -> DetachedClampOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.clamp", location)
        .add_operand(min)
        .add_operand(input)
        .add_operand(max)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::clamp`")
}

/// StableHLO [`Operation`] that computes the element-wise maximum between two tensors. The operation computes the
/// maximum value between corresponding elements from the two input tensors to produce its output tensor. The output
/// tensor shape is determined by [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input
/// tensors. The specific semantics of the comparison depend on the element type:
///
///   - **Integers**: Performs standard maximum comparison.
///   - **Floating-Point**: Applies the IEEE-754 `maximum` operation (propagates NaN).
///   - **Complex Numbers**: Computes lexicographic ordering on `(real, imag)` pairs.
///   - **Quantized Types**: Dequantizes the inputs, applies `maximum`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`MaximumOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [1.0, 5.0, 3.0]
/// // %rhs: [2.0, 4.0, 6.0]
/// %result = stablehlo.maximum %lhs, %rhs : tensor<3xf32>
/// // %result: [2.0, 5.0, 6.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#maximum) for more information.
pub trait MaximumOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`MaximumOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`MaximumOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(Maximum);
mlir_op_trait!(Maximum, OneResult);
mlir_op_trait!(Maximum, ZeroRegions);
mlir_op_trait!(Maximum, ZeroSuccessors);

/// Constructs a new detached/owned [`MaximumOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`MaximumOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn maximum<
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
) -> DetachedMaximumOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.maximum", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::maximum`")
}

/// StableHLO [`Operation`] that computes the element-wise minimum between two tensors. The operation computes the
/// minimum value between corresponding elements from the two input tensors to produce its output tensor. The output
/// tensor shape is determined by [broadcasting](https://openxla.org/xla/broadcasting) the shapes of the two input
/// tensors. The specific semantics of the comparison depend on the element type:
///
///   - **Integers**: Performs standard minimum comparison.
///   - **Floating-Point**: Applies the IEEE-754 `minimum` operation (propagates NaN).
///   - **Complex Numbers**: Computes lexicographic ordering on `(real, imag)` pairs.
///   - **Quantized Types**: Dequantizes the inputs, applies `minimum`, and then requantizes the result.
///
/// # Example
///
/// The following is an example of a [`MinimumOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [1.0, 5.0, 3.0]
/// // %rhs: [2.0, 4.0, 6.0]
/// %result = stablehlo.minimum %lhs, %rhs : tensor<3xf32>
/// // %result: [1.0, 4.0, 3.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#minimum) for more information.
pub trait MinimumOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`MinimumOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`MinimumOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(Minimum);
mlir_op_trait!(Minimum, OneResult);
mlir_op_trait!(Minimum, ZeroRegions);
mlir_op_trait!(Minimum, ZeroSuccessors);

/// Constructs a new detached/owned [`MinimumOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`MinimumOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn minimum<
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
) -> DetachedMinimumOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.minimum", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::minimum`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};
    use crate::dialects::func;
    use crate::{Attribute, Block, Context, OneResult, Operation, Size, Value};

    use super::{
        ClampOperation, CompareOperation, ComparisonDirection, ComparisonType, MaximumOperation, MinimumOperation,
        clamp, compare, maximum, minimum,
    };

    #[test]
    fn test_comparison_direction_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_comparison_direction(ComparisonDirection::NotEqual);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), ComparisonDirection::NotEqual);
    }

    #[test]
    fn test_comparison_direction_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_comparison_direction(ComparisonDirection::NotEqual);
        let attribute_2 = context.stable_hlo_comparison_direction(ComparisonDirection::NotEqual);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_comparison_direction(ComparisonDirection::LessThanOrEqual);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_comparison_direction(ComparisonDirection::NotEqual);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_comparison_direction_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_comparison_direction(ComparisonDirection::NotEqual);
        test_attribute_display_and_debug(attribute, "#stablehlo<comparison_direction NE>");
    }

    #[test]
    fn test_comparison_direction_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_comparison_direction(ComparisonDirection::NotEqual);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_comparison_type_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_comparison_type(ComparisonType::TotalOrder);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), ComparisonType::TotalOrder);
    }

    #[test]
    fn test_comparison_type_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_comparison_type(ComparisonType::TotalOrder);
        let attribute_2 = context.stable_hlo_comparison_type(ComparisonType::TotalOrder);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_comparison_type(ComparisonType::Float);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_comparison_type(ComparisonType::TotalOrder);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_comparison_type_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_comparison_type(ComparisonType::TotalOrder);
        test_attribute_display_and_debug(attribute, "#stablehlo<comparison_type TOTALORDER>");
    }

    #[test]
    fn test_comparison_type_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_comparison_type(ComparisonType::TotalOrder);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_compare() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let i1_type = context.signless_integer_type(1);
        let input_tensor_type = context.tensor_type(f32_type, &[Size::Static(3)], None, location).unwrap();
        let output_tensor_type = context.tensor_type(i1_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location), (input_tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = compare(lhs, rhs, ComparisonDirection::GreaterThanOrEqual, ComparisonType::TotalOrder, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), output_tensor_type);
            assert_eq!(op.comparison_direction(), ComparisonDirection::GreaterThanOrEqual);
            assert_eq!(op.comparison_type(), ComparisonType::TotalOrder);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "compare_test",
                func::FuncAttributes {
                    arguments: vec![input_tensor_type.into(), input_tensor_type.into()],
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
                  func.func @compare_test(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xi1> {
                    %0 = stablehlo.compare  GE, %arg0, %arg1,  TOTALORDER : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
                    return %0 : tensor<3xi1>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_clamp() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location), (tensor_type, location)]);
            let min = block.argument(0).unwrap();
            let input = block.argument(1).unwrap();
            let max = block.argument(2).unwrap();
            let op = clamp(min, input, max, location);
            assert_eq!(op.min(), min);
            assert_eq!(op.input(), input);
            assert_eq!(op.max(), max);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "clamp_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), tensor_type.into(), tensor_type.into()],
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
                  func.func @clamp_test(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<3xi32> {
                    %0 = stablehlo.clamp %arg0, %arg1, %arg2 : tensor<3xi32>
                    return %0 : tensor<3xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_maximum() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = maximum(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "maximum_test",
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
                  func.func @maximum_test(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
                    %0 = stablehlo.maximum %arg0, %arg1 : tensor<3xf32>
                    return %0 : tensor<3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_minimum() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = minimum(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), tensor_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "minimum_test",
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
                  func.func @minimum_test(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
                    %0 = stablehlo.minimum %arg0, %arg1 : tensor<3xf32>
                    return %0 : tensor<3xf32>
                  }
                }
            "},
        );
    }
}
