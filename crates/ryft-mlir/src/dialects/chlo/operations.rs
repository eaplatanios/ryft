use crate::macros::{mlir_op, mlir_op_trait};
use crate::{
    Attribute, DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, Type, Value, ValueRef,
};

use super::attributes::{Precision, PrecisionAttributeRef, RaggedDotDimensionsAttributeRef};

/// Name of the [`RaggedDotOperation::dimensions`] attribute.
pub const RAGGED_DOT_DIMENSIONS_ATTRIBUTE: &str = "ragged_dot_dimension_numbers";

/// Name of the [`RaggedDotOperation::precision`] attribute.
pub const RAGGED_DOT_PRECISION_ATTRIBUTE: &str = "precision_config";

/// CHLO [`Operation`] that computes a grouped generalized dot product using explicit group sizes.
pub trait RaggedDotOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the LHS operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the RHS operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the integer group-size operand.
    fn group_sizes(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the grouped-dot dimension-number attribute.
    fn dimensions(&self) -> Result<RaggedDotDimensionsAttributeRef<'c, 't>, Error> {
        self.attribute(RAGGED_DOT_DIMENSIONS_ATTRIBUTE)?
            .and_then(|attribute| attribute.cast())
            .ok_or_else(|| Error::invalid_argument("missing or invalid CHLO ragged-dot dimension numbers"))
    }

    /// Returns the optional operand precision configuration.
    fn precision(&self) -> Result<Option<(Precision, Precision)>, Error> {
        if !self.has_attribute(RAGGED_DOT_PRECISION_ATTRIBUTE) {
            return Ok(None);
        }
        let attribute = self.array_attribute(RAGGED_DOT_PRECISION_ATTRIBUTE)?;
        let mut elements = attribute.elements();
        let lhs = elements
            .next()
            .transpose()?
            .and_then(|element| element.cast::<PrecisionAttributeRef>())
            .ok_or_else(|| Error::invalid_argument("invalid `precision_config` attribute in `chlo.ragged_dot`"))?
            .value()?;
        let rhs = elements
            .next()
            .transpose()?
            .and_then(|element| element.cast::<PrecisionAttributeRef>())
            .ok_or_else(|| Error::invalid_argument("invalid `precision_config` attribute in `chlo.ragged_dot`"))?
            .value()?;
        if elements.next().transpose()?.is_some() {
            return Err(Error::invalid_argument("invalid `precision_config` attribute in `chlo.ragged_dot`"));
        }
        Ok(Some((lhs, rhs)))
    }
}

mlir_op!(RaggedDot);
mlir_op_trait!(RaggedDot, OneResult);
mlir_op_trait!(RaggedDot, ZeroRegions);
mlir_op_trait!(RaggedDot, ZeroSuccessors);

/// Constructs a detached [`RaggedDotOperation`] at `location`.
pub fn ragged_dot<
    'lhs,
    'rhs,
    'groups,
    'c: 'lhs + 'rhs + 'groups,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    Groups: Value<'groups, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    group_sizes: Groups,
    dimensions: RaggedDotDimensionsAttributeRef<'c, 't>,
    precision: Option<(Precision, Precision)>,
    result_type: T,
    location: L,
) -> Result<DetachedRaggedDotOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::chlo()?)?;
    let mut builder = OperationBuilder::new("chlo.ragged_dot", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .add_operand(group_sizes)
        .add_attribute(RAGGED_DOT_DIMENSIONS_ATTRIBUTE, dimensions);
    if let Some((lhs_precision, rhs_precision)) = precision {
        builder = builder.add_attribute(
            RAGGED_DOT_PRECISION_ATTRIBUTE,
            context.array_attribute(&[context.chlo_precision(lhs_precision)?, context.chlo_precision(rhs_precision)?]),
        );
    }
    builder.add_result(result_type).build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `chlo::ragged_dot`"))
    })
}

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

    use crate::dialects::chlo::attributes::Precision;
    use crate::dialects::func;
    use crate::{Block, Context, OneOperand, Operation, Size};

    use super::{RaggedDotOperation, erf, ragged_dot};

    #[test]
    fn test_ragged_dot() {
        let context = Context::new();
        let location = context.unknown_location();
        let dimensions = context.chlo_ragged_dot_dimensions(&[], &[], &[1], &[1], &[0], &[0]).unwrap();

        let lhs_type = context
            .tensor_type(context.float32_type(), &[Size::Static(4), Size::Static(2)], None, location)
            .unwrap();
        let rhs_type = context
            .tensor_type(context.float32_type(), &[Size::Static(2), Size::Static(2), Size::Static(1)], None, location)
            .unwrap();
        let group_sizes_type =
            context.tensor_type(context.signless_integer_type(32), &[Size::Static(2)], None, location).unwrap();
        let result_type = context
            .tensor_type(context.float32_type(), &[Size::Static(4), Size::Static(1)], None, location)
            .unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block =
                    context.block(&[(lhs_type, location), (rhs_type, location), (group_sizes_type, location)]);
                let operation = ragged_dot(
                    block.argument(0).unwrap(),
                    block.argument(1).unwrap(),
                    block.argument(2).unwrap(),
                    dimensions,
                    Some((Precision::Default, Precision::Default)),
                    result_type,
                    location,
                )
                .unwrap();
                assert_eq!(operation.operands().collect::<Result<Vec<_>, _>>().unwrap().len(), 3);
                assert_eq!(operation.results().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
                assert_eq!(operation.lhs().unwrap(), block.argument(0).unwrap());
                assert_eq!(operation.rhs().unwrap(), block.argument(1).unwrap());
                assert_eq!(operation.group_sizes().unwrap(), block.argument(2).unwrap());
                assert_eq!(operation.dimensions().unwrap(), dimensions);
                assert_eq!(operation.precision().unwrap(), Some((Precision::Default, Precision::Default)));
                let operation = block.append_operation(operation).unwrap();
                block.append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "ragged_dot_test",
                    func::FuncAttributes {
                        arguments: vec![lhs_type.into(), rhs_type.into(), group_sizes_type.into()],
                        results: vec![result_type.into()],
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
                  func.func @ragged_dot_test(%arg0: tensor<4x2xf32>, %arg1: tensor<2x2x1xf32>, \
                      %arg2: tensor<2xi32>) -> tensor<4x1xf32> {
                    %0 = \"chlo.ragged_dot\"(%arg0, %arg1, %arg2) <{precision_config = \
                        [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>], \
                        ragged_dot_dimension_numbers = #chlo.ragged_dot<lhs_contracting_dimensions = [1], \
                        rhs_contracting_dimensions = [1], lhs_ragged_dimensions = [0], \
                        rhs_group_dimensions = [0]>}> : (tensor<4x2xf32>, tensor<2x2x1xf32>, tensor<2xi32>) \
                        -> tensor<4x1xf32>
                    return %0 : tensor<4x1xf32>
                  }
                }
            "},
        );
    }

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
