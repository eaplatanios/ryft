use crate::{
    Attribute, DetachedOp, DialectHandle, IntegerAttributeRef, Location, Operation, OperationBuilder, Type, Value,
    mlir_op, mlir_op_trait,
};

/// StableHLO [`Operation`] that performs a bitcast conversion on a tensor. The operation reinterprets the bit
/// representation of the input tensor as a different element type without changing the underlying data. The
/// relationship between the shapes and element types of the input and output tensors depends on the bit widths
/// of their element types:
///
///   - **Equal Bit Widths:** The shape remains unchanged (e.g., `tensor<2xf32>` to `tensor<2xi32>`).
///   - **Smaller Destination Type:** An additional dimension is added at the end to accommodate multiple
///     smaller elements packed from each source element (e.g., `tensor<2xf64>` to `tensor<2x4xf16>`).
///   - **Larger Destination Type:** The last dimension is removed as multiple source elements are consolidated
///     into each destination element (e.g., `tensor<2x4xf16>` to `tensor<2xf64>`).
///
/// The following also hold for this operation:
///
///   - The total number of bits is preserved across the conversion:
///     `size(operand) * element_bit_width(operand) = size(result) * element_bit_width(result)`.
///   - If `element_bit_width(operand) = element_bit_width(result)`, then `shape(operand) = shape(result)`.
///     Otherwise, shape adjustments follow the rules described above.
///   - If `is_complex(operand) || is_complex(result)`, then `is_complex(operand) && is_complex(result)`.
///   - Given `E = is_quantized(operand) ? storage_type(operand) : element_type(operand)`,
///     `E' = is_quantized(result) ? storage_type(result) : element_type(result)`, and `R = rank(operand)`:
///       - If `num_bits(E') = num_bits(E)`, `shape(result) = shape(operand)`.
///       - If `num_bits(E') < num_bits(E)`:
///           - `rank(result) = R + 1`,
///           - `dim(result, i) = dim(operand, i)` for all `0 <= i < R`, and
///           - `dim(result, R) * num_bits(E') = num_bits(E)`.
///       - If `num_bits(E') > num_bits(E)`:
///           - `rank(result) = R - 1`,
///           - `dim(result, i) = dim(operand, i)` for all `0 <= i < R`, and
///           - `dim(operand, R - 1) * num_bits(E) = num_bits(E')`.
///
/// # Example
///
/// The following is an example of a [`BitcastConvertOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: 0x0123456789ABCDEF
/// %result = stablehlo.bitcast_convert %operand : (tensor<f64>) -> tensor<4xf16>
/// // %result: [0xCDEF, 0x89AB, 0x4567, 0x0123] // little-endian representation
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#bitcast_convert)
/// for more information.
pub trait BitcastConvertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(BitcastConvert);
mlir_op_trait!(BitcastConvert, OneResult);
mlir_op_trait!(BitcastConvert, ZeroRegions);
mlir_op_trait!(BitcastConvert, ZeroSuccessors);

/// Constructs a new detached/owned [`BitcastConvertOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`BitcastConvertOperation`] for more information on the operation semantics and constraints
/// on the output type.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn bitcast_convert<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: V,
    output_type: T,
    location: L,
) -> DetachedBitcastConvertOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.bitcast_convert", location)
        .add_operand(input)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::bitcast_convert`")
}

/// StableHLO [`Operation`] that performs element-wise type conversion on a tensor. The operation transforms values from
/// one element type to another while preserving the tensor's shape. The specific conversion semantics depend on the
/// source and destination element types:
///
///   - For **boolean-to-any-supported-type** conversions, the value `false` is converted to `0`, and the value `true`
///     is converted to `1`. For **any-supported-type-to-boolean** conversions, a `0` value is converted to `false`,
///     and non-zero values are converted to `true`.
///   - For conversions involving **integer-to-integer**, **integer-to-floating-point** or
///     **floating-point-to-floating-point**, if the source value can be exactly represented in the destination type,
///     the result value is that exact representation. Otherwise, the behavior is
///     [TBD](https://github.com/openxla/stablehlo/issues/180).
///   - For conversions involving **floating-point-to-integer**, the fractional part is truncated. If the truncated
///     value cannot be represented in the destination type, the behavior is
///     [TBD](https://github.com/openxla/stablehlo/issues/180).
///   - Conversions involving **complex-to-complex** follow the behavior of **floating-point-to-floating-point**
///     conversions for converting real and imaginary parts. For **complex-to-any-other-type** and
///     **any-other-type-to-complex** conversions, the source imaginary value is ignored or the destination
///     imaginary value is zeroed, respectively. The conversion of the real part follows the floating-point conversions.
///
/// In principle, this operation could express dequantization (conversion from quantized tensors to regular tensors),
/// quantization (conversion from regular tensors to quantized tensors), and requantization (conversion between
/// quantized tensors), but at the moment we have dedicated operations for that; [`uniform_dequantize`] for the first
/// use case and [`uniform_quantize`] for the second and third use cases. In the future, these operations may be merged
/// into [`convert`] (see relevant [issue](https://github.com/openxla/stablehlo/issues/1576) for more information).
///
/// # Example
///
/// The following is an example of a [`ConvertOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [-1, 0, 1]
/// %result = stablehlo.convert %operand : (tensor<3xi32>) -> tensor<3xcomplex<f64>>
/// // %result: [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#convert) for more information.
pub trait ConvertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Convert);
mlir_op_trait!(Convert, OneResult);
mlir_op_trait!(Convert, ZeroRegions);
mlir_op_trait!(Convert, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`ConvertOperation`] for more information on the operation semantics and constraints on the output type.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn convert<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: V,
    output_type: T,
    location: L,
) -> DetachedConvertOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.convert", location)
        .add_operand(input)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::convert`")
}

/// StableHLO [`Operation`] that converts a quantized tensor to a floating-point tensor.
/// The operation dequantizes each element using the uniform quantization formula:
///
/// ```text
/// result[i] = scale * (operand[i] - zero_point)
/// ```
///
/// where `scale` and `zero_point` are the quantization parameters encoded in the quantized tensor type. This operation
/// supports both per-tensor and per-axis quantization schemes, similar to [`UniformQuantizeOperation`]:
///
///   - **Per-Tensor Quantization**: A single scale and zero point are used for all elements in the tensor.
///     This quantization information is encoded in the tensor type as a suffix that looks like this:
///     `x!quant.uniform<i8:f32, 0.5:-10>>`, where in this case the source data type is `i8`, the target data
///     type is `f32`, the scale is `0.5` and the zero point is `-10`.
///   - **Per-Axis Quantization**: Different scales and zero points are applied to slices along a specified dimension.
///     This quantization information is encoded in the tensor type as a suffix that looks like this:
///     `x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>`, where in this case the source data type is `i8`, the target
///     data type is `f32`, the quantization dimension is dimension `0` and along the first slice of that dimension we
///     use scale `0.1` and zero point `-30`, and along the second slice of that dimension we use scale `0.5` and zero
///     point `-20`.
///
/// # Example
///
/// The following is an example of a [`UniformDequantizeOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [10, 10]
/// %result = stablehlo.uniform_dequantize %operand
///   : (tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>) -> tensor<2xf32>
/// // %result: [4.0, 15.0]
/// ```
///
/// Refer to the [official StableHLO quantization documentation](https://openxla.org/stablehlo/quantization)
/// for more information.
pub trait UniformDequantizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(UniformDequantize);
mlir_op_trait!(UniformDequantize, OneResult);
mlir_op_trait!(UniformDequantize, ZeroRegions);
mlir_op_trait!(UniformDequantize, ZeroSuccessors);

/// Constructs a new detached/owned [`UniformDequantizeOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`UniformDequantizeOperation`] for more information on the operation semantics and constraints
/// on the output type.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn uniform_dequantize<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: V,
    output_type: T,
    location: L,
) -> DetachedUniformDequantizeOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.uniform_dequantize", location)
        .add_operand(input)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::uniform_dequantize`")
}

/// StableHLO [`Operation`] that converts a floating-point tensor to a quantized tensor.
/// The operation quantizes each element using the uniform quantization formula:
///
/// ```text
/// result[i] = round(operand[i] / scale) + zero_point
/// ```
///
/// where `scale` and `zero_point` are the quantization parameters encoded in the result tensor type. The quantized
/// values are clamped to the range of the quantized storage type. This operation supports both per-tensor and
/// per-axis quantization schemes, similar to [`UniformDequantizeOperation`]:
///
///   - **Per-Tensor Quantization**: A single scale and zero point are used for all elements in the tensor.
///     This quantization information is encoded in the tensor type as a suffix that looks like this:
///     `x!quant.uniform<i8:f32, 0.5:-10>>`, where in this case the source data type is `i8`, the target data
///     type is `f32`, the scale is `0.5` and the zero point is `-10`.
///   - **Per-Axis Quantization**: Different scales and zero points are applied to slices along a specified dimension.
///     This quantization information is encoded in the tensor type as a suffix that looks like this:
///     `x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>`, where in this case the source data type is `i8`, the target
///     data type is `f32`, the quantization dimension is dimension `0` and along the first slice of that dimension
///     we use scale `0.1` and zero point `-30`, and along the second slice of that dimension we use scale `0.5` and
///     zero point `-20`.
///
/// # Example
///
/// The following is an example of a [`UniformQuantizeOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [2.5, 7.5, 12.5]
/// %result = stablehlo.uniform_quantize %operand : (tensor<3xf32>) -> tensor<3x!quant.uniform<i8:f32, 0.5:5>>
/// // %result: [10, 20, 30]
/// ```
///
/// Refer to the [official StableHLO quantization documentation](https://openxla.org/stablehlo/quantization)
/// for more information.
pub trait UniformQuantizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(UniformQuantize);
mlir_op_trait!(UniformQuantize, OneResult);
mlir_op_trait!(UniformQuantize, ZeroRegions);
mlir_op_trait!(UniformQuantize, ZeroSuccessors);

/// Constructs a new detached/owned [`UniformQuantizeOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`UniformQuantizeOperation`] for more information on the operation semantics and constraints
/// on the output type.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn uniform_quantize<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: V,
    output_type: T,
    location: L,
) -> DetachedUniformQuantizeOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.uniform_quantize", location)
        .add_operand(input)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::uniform_quantize`")
}

/// Name of the [`Attribute`] that is used to store [`ReducePrecisionOperation::exponent_bits`].
pub const REDUCE_PRECISION_EXPONENT_BITS_ATTRIBUTE: &str = "exponent_bits";

/// Name of the [`Attribute`] that is used to store [`ReducePrecisionOperation::mantissa_bits`].
pub const REDUCE_PRECISION_MANTISSA_BITS_ATTRIBUTE: &str = "mantissa_bits";

/// StableHLO [`Operation`] that performs element-wise reduction of floating-point precision.
///
/// This operation simulates the effect of converting floating-point values to a lower-precision format and back,
/// without actually changing the element type. This is useful for:
///
///   - Testing numerical stability of models under reduced precision.
///   - Simulating lower-precision hardware (e.g., `bfloat16`, `float16`) on higher-precision hardware.
///   - Understanding the effects of quantization on model accuracy.
///
/// Formally, this operation does the following:
///
///   - The mantissa bits of the original value are updated to round the original value to the nearest value
///     representable with [`ReducePrecisionOperation::mantissa_bits`] using `roundToIntegralTiesToEven` semantics.
///   - Then, if [`ReducePrecisionOperation::mantissa_bits`] is smaller than the number of mantissa bits of the
///     original value, the mantissa bits are truncated to [`ReducePrecisionOperation::mantissa_bits`] bits.
///   - Then, if the exponent bits of the intermediate result do not fit into the range provided by
///     [`ReducePrecisionOperation::exponent_bits`], the intermediate result overflows to infinity using the original
///     sign or underflows to zero using the original sign.
///   - For quantized types, it dequantizes the input, applies the precision reduction operation as described above,
///     and then quantizes the result.
///
/// # Attributes
///
/// - `exponent_bits`: Positive integer specifying the number of exponent bits in the target format.
/// - `mantissa_bits`: Non-negative integer specifying the number of mantissa bits in the target format.
///
/// # Example
///
/// The following is an example of a [`ReducePrecisionOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [1.0, 2.0, 3.0]
/// %result = stablehlo.reduce_precision %operand, format = e5m10 : tensor<3xf64>
/// // %result: values rounded to e5m10 precision (5 exponent bits, 10 mantissa bits)
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#reduce_precision)
/// for more information.
pub trait ReducePrecisionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the number of exponent bits in the target precision of this [`ReducePrecisionOperation`].
    fn exponent_bits(&self) -> u32 {
        self.attribute(REDUCE_PRECISION_EXPONENT_BITS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as u32)
            .unwrap_or_else(|| {
                panic!(
                    "invalid '{REDUCE_PRECISION_EXPONENT_BITS_ATTRIBUTE}' attribute in `stable_hlo::reduce_precision`"
                )
            })
    }

    /// Returns the number of mantissa bits in the target precision of this [`ReducePrecisionOperation`].
    fn mantissa_bits(&self) -> u32 {
        self.attribute(REDUCE_PRECISION_MANTISSA_BITS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as u32)
            .unwrap_or_else(|| {
                panic!(
                    "invalid '{REDUCE_PRECISION_MANTISSA_BITS_ATTRIBUTE}' attribute in `stable_hlo::reduce_precision`"
                )
            })
    }
}

mlir_op!(ReducePrecision);
mlir_op_trait!(ReducePrecision, OneResult);
mlir_op_trait!(ReducePrecision, ZeroRegions);
mlir_op_trait!(ReducePrecision, ZeroSuccessors);

/// Constructs a new detached/owned [`ReducePrecisionOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DetachedReducePrecisionOperation`] for more information on the operation semantics and constraints
/// on the output type.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn reduce_precision<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    exponent_bits: u32,
    mantissa_bits: u32,
    location: L,
) -> DetachedReducePrecisionOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.reduce_precision", location)
        .add_operand(input)
        .add_attribute(
            REDUCE_PRECISION_EXPONENT_BITS_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), exponent_bits as i64),
        )
        .add_attribute(
            REDUCE_PRECISION_MANTISSA_BITS_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), mantissa_bits as i64),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::reduce_precision`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, DialectHandle, Operation, Size, Value};

    use super::{
        ReducePrecisionOperation, bitcast_convert, convert, reduce_precision, uniform_dequantize, uniform_quantize,
    };

    #[test]
    fn test_bitcast_convert() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();
        let i8_type = context.signless_integer_type(8);
        let input_tensor_type = context.tensor_type(f64_type, &[Size::Static(8)], None, location).unwrap();
        let output_tensor_type =
            context.tensor_type(i8_type, &[Size::Static(8), Size::Static(8)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location)]);
            let input_value = block.argument(0).unwrap();
            let bitcast_convert_op = bitcast_convert(input_value, output_tensor_type, location);
            assert_eq!(bitcast_convert_op.operands().count(), 1);
            assert_eq!(bitcast_convert_op.results().count(), 1);
            assert_eq!(bitcast_convert_op.result(0).unwrap().r#type(), output_tensor_type);
            let bitcast_convert_block = block.append_operation(bitcast_convert_op);
            block.append_operation(func::r#return(&[bitcast_convert_block.result(0).unwrap()], location));
            func::func(
                "bitcast_convert_test",
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
                  func.func @bitcast_convert_test(%arg0: tensor<8xf64>) -> tensor<8x8xi8> {
                    %0 = stablehlo.bitcast_convert %arg0 : (tensor<8xf64>) -> tensor<8x8xi8>
                    return %0 : tensor<8x8xi8>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_convert() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f64_type = context.float64_type();
        let complex_type = context.complex_type(f64_type);
        let input_tensor_type = context.tensor_type(i32_type, &[Size::Static(3)], None, location).unwrap();
        let output_tensor_type = context.tensor_type(complex_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location)]);
            let input_value = block.argument(0).unwrap();
            let convert_op = convert(input_value, output_tensor_type, location);
            assert_eq!(convert_op.operands().count(), 1);
            assert_eq!(convert_op.results().count(), 1);
            assert_eq!(convert_op.result(0).unwrap().r#type(), output_tensor_type);
            let convert_block = block.append_operation(convert_op);
            block.append_operation(func::r#return(&[convert_block.result(0).unwrap()], location));
            func::func(
                "convert_test",
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
                  func.func @convert_test(%arg0: tensor<3xi32>) -> tensor<3xcomplex<f64>> {
                    %0 = stablehlo.convert %arg0 : (tensor<3xi32>) -> tensor<3xcomplex<f64>>
                    return %0 : tensor<3xcomplex<f64>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_uniform_dequantize() {
        let context = Context::new();
        context.load_dialect(DialectHandle::tensor());
        context.load_dialect(DialectHandle::quant());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let input_quantized_type = context
            .parse_type("tensor<3x!quant.uniform<i8:f32, 0.5:5>>")
            .expect("failed to parse quantized type");
        let output_tensor_type = context.tensor_type(f32_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_quantized_type, location)]);
            let input_value = block.argument(0).unwrap();
            let uniform_dequantize_op = uniform_dequantize(input_value, output_tensor_type, location);
            assert_eq!(uniform_dequantize_op.operands().count(), 1);
            assert_eq!(uniform_dequantize_op.results().count(), 1);
            assert_eq!(uniform_dequantize_op.result(0).unwrap().r#type(), output_tensor_type,);
            let uniform_dequantize_block = block.append_operation(uniform_dequantize_op);
            block.append_operation(func::r#return(&[uniform_dequantize_block.result(0).unwrap()], location));
            func::func(
                "uniform_dequantize_test",
                func::FuncAttributes {
                    arguments: vec![input_quantized_type.into()],
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
                  func.func @uniform_dequantize_test(\
                    %arg0: tensor<3x!quant.uniform<i8:f32, 5.000000e-01:5>>\
                  ) -> tensor<3xf32> {
                    %0 = stablehlo.uniform_dequantize %arg0 \
                      : (tensor<3x!quant.uniform<i8:f32, 5.000000e-01:5>>) -> tensor<3xf32>
                    return %0 : tensor<3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_uniform_quantize() {
        let context = Context::new();
        context.load_dialect(DialectHandle::tensor());
        context.load_dialect(DialectHandle::quant());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let input_tensor_type = context.tensor_type(f32_type, &[Size::Static(3)], None, location).unwrap();
        let output_quantized_type = context
            .parse_type("tensor<3x!quant.uniform<i8:f32, 0.5:5>>")
            .expect("failed to parse quantized type");
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location)]);
            let input_value = block.argument(0).unwrap();
            let uniform_quantize_op = uniform_quantize(input_value, output_quantized_type, location);
            assert_eq!(uniform_quantize_op.operands().count(), 1);
            assert_eq!(uniform_quantize_op.results().count(), 1);
            assert_eq!(uniform_quantize_op.result(0).unwrap().r#type(), output_quantized_type);
            let uniform_quantize_block = block.append_operation(uniform_quantize_op);
            block.append_operation(func::r#return(&[uniform_quantize_block.result(0).unwrap()], location));
            func::func(
                "uniform_quantize_test",
                func::FuncAttributes {
                    arguments: vec![input_tensor_type.into()],
                    results: vec![output_quantized_type.into()],
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
                  func.func @uniform_quantize_test(\
                    %arg0: tensor<3xf32>\
                  ) -> tensor<3x!quant.uniform<i8:f32, 5.000000e-01:5>> {
                    %0 = stablehlo.uniform_quantize %arg0 \
                      : (tensor<3xf32>) -> tensor<3x!quant.uniform<i8:f32, 5.000000e-01:5>>
                    return %0 : tensor<3x!quant.uniform<i8:f32, 5.000000e-01:5>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_reduce_precision() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();
        let input_tensor_type = context.tensor_type(f64_type, &[Size::Static(6)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location)]);
            let input_value = block.argument(0).unwrap();
            let reduce_precision_op = reduce_precision(input_value, 5, 10, location);
            assert_eq!(reduce_precision_op.operands().count(), 1);
            assert_eq!(reduce_precision_op.results().count(), 1);
            assert_eq!(reduce_precision_op.exponent_bits(), 5);
            assert_eq!(reduce_precision_op.mantissa_bits(), 10);
            assert_eq!(reduce_precision_op.result(0).unwrap().r#type(), input_tensor_type);
            let reduce_precision_block = block.append_operation(reduce_precision_op);
            block.append_operation(func::r#return(&[reduce_precision_block.result(0).unwrap()], location));
            func::func(
                "reduce_precision_test",
                func::FuncAttributes {
                    arguments: vec![input_tensor_type.into()],
                    results: vec![input_tensor_type.into()],
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
                  func.func @reduce_precision_test(%arg0: tensor<6xf64>) -> tensor<6xf64> {
                    %0 = stablehlo.reduce_precision %arg0, format = e5m10 : tensor<6xf64>
                    return %0 : tensor<6xf64>
                  }
                }
            "},
        );
    }
}
