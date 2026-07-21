use crate::{DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, Value, mlir_op, mlir_op_trait};

/// CHLO [`Operation`] that performs element-wise error function computation on a tensor of floating-point element
/// type, where `erf(x) = 2/√π · ∫₀ˣ e^{-t²} dt`. The XLA compiler legalizes this operation to a rational polynomial
/// approximation over StableHLO operations during lowering.
///
/// # Example
///
/// The following is an example of an [`ErfOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [-1.0, 0.0, 1.0]
/// %result = chlo.erf %operand : tensor<3xf32> -> tensor<3xf32>
/// // %result: [-0.842700793, 0.0, 0.842700793]
/// ```
///
/// Refer to the [official CHLO specification](https://openxla.org/stablehlo/generated/chlo#chloerf_chloerfop)
/// for more information.
pub trait ErfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Erf);
mlir_op_trait!(Erf, OneOperand);
mlir_op_trait!(Erf, OneResult);
mlir_op_trait!(Erf, ZeroRegions);
mlir_op_trait!(Erf, ZeroSuccessors);

/// Constructs a new detached/owned [`ErfOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ErfOperation`] for more information on the operation semantics.
pub fn erf<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedErfOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::chlo()?)?;
    OperationBuilder::new("chlo.erf", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `chlo::erf`"))
        })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, OneOperand, Operation, Size};

    use super::erf;

    #[test]
    fn test_erf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let op = erf(input, location).unwrap();
                assert_eq!(op.input().unwrap(), input);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "erf_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @erf_test(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = chlo.erf %arg0 : tensor<2x2xf32> -> tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "},
        );
    }
}
