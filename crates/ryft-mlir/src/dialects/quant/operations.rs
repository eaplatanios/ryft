use crate::{DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Type, Value, mlir_op, mlir_op_trait};

/// Quant [`Operation`] that casts from an expressed floating-point scalar/tensor type to a quantized type.
/// This operation is represented in MLIR as `quant.qcast`. Conceptually, it applies the quantization parameters
/// encoded in the result type to map expressed input values to quantized values. The scalar/tensor shape is
/// preserved and only the element representation changes.
///
/// # Example
///
/// The following is an example of a [`QCastOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = quant.qcast %input : tensor<?xf32> to tensor<?x!quant.uniform<i8:f32, 2.0>>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/QuantDialect/#quantqcast-quantquantizecastop)
/// for more information.
pub trait QCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(QCast);
mlir_op_trait!(QCast, OneOperand);
mlir_op_trait!(QCast, OneResult);
mlir_op_trait!(QCast, ZeroRegions);
mlir_op_trait!(QCast, ZeroSuccessors);

/// Constructs a new detached/owned [`QCastOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`QCastOperation`] for more information on the operation semantics and constraints on the output type.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn qcast<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: V,
    output_type: T,
    location: L,
) -> DetachedQCastOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::quant());
    OperationBuilder::new("quant.qcast", location)
        .add_operand(input)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `quant::qcast`")
}

/// Quant [`Operation`] that casts from a quantized scalar/tensor type to an expressed floating-point type.
/// This operation is represented in MLIR as `quant.dcast`. Conceptually, it applies the dequantization mapping
/// encoded in the input quantized type to recover expressed floating-point values. The scalar/tensor shape is
/// preserved and only the element representation changes.
///
/// # Example
///
/// The following is an example of a [`DCastOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = quant.dcast %input : tensor<?x!quant.uniform<i8:f32, 2.0>> to tensor<?xf32>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/QuantDialect/#quantdcast-quantdequantizecastop)
/// for more information.
pub trait DCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(DCast);
mlir_op_trait!(DCast, OneOperand);
mlir_op_trait!(DCast, OneResult);
mlir_op_trait!(DCast, ZeroRegions);
mlir_op_trait!(DCast, ZeroSuccessors);

/// Constructs a new detached/owned [`DCastOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`DCastOperation`] for more information on the operation semantics and constraints on the output type.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn dcast<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: V,
    output_type: T,
    location: L,
) -> DetachedDCastOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::quant());
    OperationBuilder::new("quant.dcast", location)
        .add_operand(input)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `quant::dcast`")
}

/// Quant [`Operation`] that casts between a quantized scalar/tensor type and its storage scalar/tensor type.
/// This operation is represented in MLIR as `quant.scast`. It converts between the quantized and storage
/// representations without changing the logical scalar/tensor shape, and is commonly used when lowering
/// quantized IR to integer storage computations.
///
/// # Example
///
/// The following is an example of an [`SCastOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = quant.scast %input : tensor<?x!quant.uniform<i8:f32, 2.0>> to tensor<?xi8>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/QuantDialect/#quantscast-quantstoragecastop)
/// for more information.
pub trait SCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(SCast);
mlir_op_trait!(SCast, OneOperand);
mlir_op_trait!(SCast, OneResult);
mlir_op_trait!(SCast, ZeroRegions);
mlir_op_trait!(SCast, ZeroSuccessors);

/// Constructs a new detached/owned [`SCastOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`SCastOperation`] for more information on the operation semantics and constraints on the output type.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn scast<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: V,
    output_type: T,
    location: L,
) -> DetachedSCastOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::quant());
    OperationBuilder::new("quant.scast", location)
        .add_operand(input)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `quant::scast`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, DialectHandle, Operation, Size, Value};

    use super::{dcast, qcast, scast};

    #[test]
    fn test_qcast() {
        let context = Context::new();
        context.load_dialect(DialectHandle::quant());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let input_tensor_type = context.tensor_type(f32_type, &[Size::Static(3)], None, location).unwrap();
        let output_quantized_type = context.parse_type("tensor<3x!quant.uniform<i8:f32, 0.5:5>>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location)]);
            let input_value = block.argument(0).unwrap();
            let qcast_op = qcast(input_value, output_quantized_type, location);
            assert_eq!(qcast_op.operands().count(), 1);
            assert_eq!(qcast_op.results().count(), 1);
            assert_eq!(qcast_op.result(0).unwrap().r#type(), output_quantized_type);
            let qcast_block = block.append_operation(qcast_op);
            block.append_operation(func::r#return(&[qcast_block.result(0).unwrap()], location));
            func::func(
                "qcast_test",
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
                  func.func @qcast_test(%arg0: tensor<3xf32>) -> tensor<3x!quant.uniform<i8:f32, 5.000000e-01:5>> {
                    %0 = quant.qcast %arg0 : tensor<3xf32> to tensor<3x!quant.uniform<i8:f32, 5.000000e-01:5>>
                    return %0 : tensor<3x!quant.uniform<i8:f32, 5.000000e-01:5>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_dcast() {
        let context = Context::new();
        context.load_dialect(DialectHandle::quant());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let input_quantized_type = context.parse_type("tensor<3x!quant.uniform<i8:f32, 0.5:5>>").unwrap();
        let output_tensor_type = context.tensor_type(f32_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_quantized_type, location)]);
            let input_value = block.argument(0).unwrap();
            let dcast_op = dcast(input_value, output_tensor_type, location);
            assert_eq!(dcast_op.operands().count(), 1);
            assert_eq!(dcast_op.results().count(), 1);
            assert_eq!(dcast_op.result(0).unwrap().r#type(), output_tensor_type);
            let dcast_block = block.append_operation(dcast_op);
            block.append_operation(func::r#return(&[dcast_block.result(0).unwrap()], location));
            func::func(
                "dcast_test",
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
                  func.func @dcast_test(%arg0: tensor<3x!quant.uniform<i8:f32, 5.000000e-01:5>>) -> tensor<3xf32> {
                    %0 = quant.dcast %arg0 : tensor<3x!quant.uniform<i8:f32, 5.000000e-01:5>> to tensor<3xf32>
                    return %0 : tensor<3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_scast() {
        let context = Context::new();
        context.load_dialect(DialectHandle::quant());
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let input_quantized_type = context.parse_type("tensor<3x!quant.uniform<i8:f32, 0.5:5>>").unwrap();
        let output_tensor_type = context.tensor_type(i8_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_quantized_type, location)]);
            let input_value = block.argument(0).unwrap();
            let scast_op = scast(input_value, output_tensor_type, location);
            assert_eq!(scast_op.operands().count(), 1);
            assert_eq!(scast_op.results().count(), 1);
            assert_eq!(scast_op.result(0).unwrap().r#type(), output_tensor_type);
            let scast_block = block.append_operation(scast_op);
            block.append_operation(func::r#return(&[scast_block.result(0).unwrap()], location));
            func::func(
                "scast_test",
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
                  func.func @scast_test(%arg0: tensor<3x!quant.uniform<i8:f32, 5.000000e-01:5>>) -> tensor<3xi8> {
                    %0 = quant.scast %arg0 : tensor<3x!quant.uniform<i8:f32, 5.000000e-01:5>> to tensor<3xi8>
                    return %0 : tensor<3xi8>
                  }
                }
            "},
        );
    }
}
