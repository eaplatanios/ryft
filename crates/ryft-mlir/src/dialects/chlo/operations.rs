use crate::macros::{mlir_op, mlir_op_trait};
use crate::{
    Attribute, DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, Type, Value, ValueRef,
};

use super::attributes::{Precision, PrecisionAttributeRef, RaggedDotDimensionsAttributeRef};

/// Name of the [`RaggedDotOperation::dimensions`] attribute.
pub const RAGGED_DOT_DIMENSIONS_ATTRIBUTE: &str = "ragged_dot_dimension_numbers";

/// Name of the [`RaggedDotOperation::precision`] attribute.
pub const RAGGED_DOT_PRECISION_ATTRIBUTE: &str = "precision_config";

/// CHLO [`Operation`] that computes a generalized dot product over one ragged LHS dimension. The integer
/// [`RaggedDotOperation::group_sizes`] operand partitions that dimension into groups, while
/// [`RaggedDotOperation::dimensions`] identifies the batch, contracting, ragged, and optional RHS group dimensions.
/// The operation supports three modes, depending on the role of the LHS ragged dimension:
///
///   - Non-contracting: `[b, m, k]`, `[g, b, k, n]`, `[b, g]` produce `[b, m, n]`; the RHS has group dimension `g`.
///   - Contracting: `[b, m, k]`, `[b, k, n]`, `[b, g]` produce `[g, b, m, n]`.
///   - Batching: `[b, m, k]`, `[b, k, n]`, `[g]` produce `[b, m, n]`.
///
/// Here `b`, `k`, and `g` denote batch, contracting, and group dimensions, respectively. The optional
/// [`RaggedDotOperation::precision`] configuration supplies one precision value for each data operand.
///
/// Refer to the
/// [official CHLO specification](https://openxla.org/stablehlo/generated/chlo#chloragged_dot_chloraggeddotop)
/// for the complete shape and type constraints.
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

/// Constructs a detached [`RaggedDotOperation`] at `location`. Refer to the documentation of [`RaggedDotOperation`]
/// for the supported modes and shape constraints.
///
/// # Parameters
///
///   - `lhs`: Left data operand containing exactly one ragged dimension.
///   - `rhs`: Right data operand, optionally containing one group dimension.
///   - `group_sizes`: Integer tensor containing the ragged group sizes.
///   - `dimensions`: Batch, contracting, ragged, and group dimension assignments.
///   - `precision`: Optional per-operand precision values, ordered as LHS then RHS.
///   - `result_type`: Result tensor type implied by the selected ragged-dot mode.
///   - `location`: Source location to attach to the operation.
pub fn ragged_dot<
    'lhs,
    'rhs,
    'groups,
    'c: 'lhs + 'rhs + 'groups,
    't: 'c,
    T: Type<'c, 't>,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    Groups: Value<'groups, 'c, 't>,
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

/// Name of the attribute storing the number of entries selected by [`TopKOperation`].
pub const TOP_K_COUNT_ATTRIBUTE: &str = "k";

/// Name of the stable ordering attribute of [`TopKOperation`].
pub const TOP_K_IS_STABLE_ATTRIBUTE: &str = "is_stable";

/// CHLO [`Operation`] that selects the largest [`TopKOperation::k`] entries along the last dimension of an input
/// tensor, returning their values in descending order and their zero-based `i32` indices along that dimension.
/// Both results preserve the input shape except that the last dimension has size `k`, which must not exceed the
/// corresponding input dimension. The values retain the input element type. When [`TopKOperation::is_stable`] is
/// true (the default), equal values retain their input order; otherwise, their relative order is unspecified.
///
/// # Example
///
/// The following is an example of a [`TopKOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [2.0, 9.0, 9.0, 4.0]
/// %values, %indices = chlo.top_k(%operand, k = 3) : tensor<4xf32> -> (tensor<3xf32>, tensor<3xi32>)
/// // %values: [9.0, 9.0, 4.0]
/// // %indices: [1, 2, 3]
/// ```
///
/// Refer to the [official CHLO specification](https://openxla.org/stablehlo/generated/chlo#chlotop_k_chlotopkop)
/// for more information.
pub trait TopKOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the number of entries selected along the last dimension.
    fn k(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(TOP_K_COUNT_ATTRIBUTE)?.signed_value())
    }

    /// Returns whether ties retain their input order, defaulting to true when the attribute is absent.
    fn is_stable(&self) -> Result<bool, Error> {
        if self.has_attribute(TOP_K_IS_STABLE_ATTRIBUTE) {
            Ok(self.boolean_attribute(TOP_K_IS_STABLE_ATTRIBUTE)?.value())
        } else {
            Ok(true)
        }
    }
}

mlir_op!(TopK);
mlir_op_trait!(TopK, ZeroRegions);
mlir_op_trait!(TopK, ZeroSuccessors);

/// Constructs a new detached/owned [`TopKOperation`] at the specified [`Location`], inferring the value and index
/// tensor types from `input` and `k`. Refer to the documentation of [`TopKOperation`] for more information on the
/// operation semantics.
///
/// # Parameters
///
///   - `input`: Tensor whose last dimension is searched for the largest entries.
///   - `k`: Number of entries to select, at most the size of the last input dimension.
///   - `is_stable`: Whether equal values retain their input order.
///   - `location`: Source location to attach to the operation.
pub fn top_k<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    k: usize,
    is_stable: bool,
    location: L,
) -> Result<DetachedTopKOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::chlo()?)?;
    let k = i64::try_from(k).map_err(|_| Error::invalid_argument("`k` exceeds the signed 64-bit range"))?;
    OperationBuilder::new("chlo.top_k", location)
        .add_operand(input)
        .add_attribute(TOP_K_COUNT_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), k))
        .add_attribute(TOP_K_IS_STABLE_ATTRIBUTE, context.boolean_attribute(is_stable))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `chlo::top_k`"))
        })
}

/// CHLO [`Operation`] that multiplies two integer tensors element-wise and returns the most significant `N` bits
/// of each full `2N`-bit product, where `N` is the operand element bit width. Both operands and the result have
/// matching shapes and integer element types.
///
/// # Example
///
/// The following is an example of a [`MulhiOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [65536, 131072, 7]
/// // %rhs: [65536, 65536, 9]
/// %result = chlo.mulhi %lhs, %rhs : tensor<3xi32>, tensor<3xi32> -> tensor<3xi32>
/// // %result: [1, 2, 0]
/// ```
///
/// Refer to the [official CHLO specification](https://openxla.org/stablehlo/generated/chlo#chlomulhi_chlomulhiop)
/// for more information.
pub trait MulhiOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Mulhi);
mlir_op_trait!(Mulhi, OneResult);
mlir_op_trait!(Mulhi, ZeroRegions);
mlir_op_trait!(Mulhi, ZeroSuccessors);

/// Constructs a new detached/owned [`MulhiOperation`] at the specified [`Location`], using the left operand's tensor
/// type for the result. Refer to the documentation of [`MulhiOperation`] for more information on the operation
/// semantics.
pub fn mulhi<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    lhs: V,
    rhs: V,
    location: L,
) -> Result<DetachedMulhiOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::chlo()?)?;
    OperationBuilder::new("chlo.mulhi", location)
        .add_result(lhs.r#type()?)
        .add_operand(lhs)
        .add_operand(rhs)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `chlo::mulhi`"))
        })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::chlo::attributes::Precision;
    use crate::dialects::func;
    use crate::{Attribute, Block, Context, Error, OneOperand, Operation, Size};

    use super::*;

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

                let mut operation_without_precision = ragged_dot(
                    block.argument(0).unwrap(),
                    block.argument(1).unwrap(),
                    block.argument(2).unwrap(),
                    dimensions,
                    None,
                    result_type,
                    location,
                )
                .unwrap();
                assert_eq!(operation_without_precision.precision().unwrap(), None);

                operation_without_precision.set_attribute(
                    RAGGED_DOT_PRECISION_ATTRIBUTE,
                    context.array_attribute(&[context.chlo_precision(Precision::Default).unwrap()]),
                );
                assert!(matches!(
                    operation_without_precision.precision(),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid `precision_config` attribute in `chlo.ragged_dot`",
                ));

                operation_without_precision.set_attribute(
                    RAGGED_DOT_PRECISION_ATTRIBUTE,
                    context.array_attribute(&[
                        context.chlo_precision(Precision::Default).unwrap().as_ref(),
                        context.string_attribute("rhs").as_ref(),
                    ]),
                );
                assert!(matches!(
                    operation_without_precision.precision(),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid `precision_config` attribute in `chlo.ragged_dot`",
                ));

                operation_without_precision.set_attribute(
                    RAGGED_DOT_PRECISION_ATTRIBUTE,
                    context.array_attribute(&[context.string_attribute("lhs"), context.string_attribute("rhs")]),
                );
                assert!(matches!(
                    operation_without_precision.precision(),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid `precision_config` attribute in `chlo.ragged_dot`",
                ));

                operation_without_precision.set_attribute(
                    RAGGED_DOT_PRECISION_ATTRIBUTE,
                    context.array_attribute(&[
                        context.chlo_precision(Precision::Default).unwrap(),
                        context.chlo_precision(Precision::High).unwrap(),
                        context.chlo_precision(Precision::Highest).unwrap(),
                    ]),
                );
                assert!(matches!(
                    operation_without_precision.precision(),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid `precision_config` attribute in `chlo.ragged_dot`",
                ));

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

    #[test]
    fn test_top_k() {
        let context = Context::new();
        let location = context.unknown_location();
        let input_type = context
            .tensor_type(context.float32_type(), &[Size::Static(2), Size::Static(8)], None, location)
            .unwrap();
        let block = context.block(&[(input_type, location)]);
        for is_stable in [true, false] {
            let operation = top_k(block.argument(0).unwrap(), 3, is_stable, location).unwrap();
            assert!(operation.verify());
            assert_eq!(operation.k(), Ok(3));
            assert_eq!(operation.is_stable(), Ok(is_stable));
            assert_eq!(operation.result_count(), 2);
            assert_eq!(operation.result(0).unwrap().r#type().unwrap().to_string(), "tensor<2x3xf32>");
            assert_eq!(operation.result(1).unwrap().r#type().unwrap().to_string(), "tensor<2x3xi32>");
        }
        assert!(top_k(block.argument(0).unwrap(), 9, true, location).is_err());
        let mut operation = top_k(block.argument(0).unwrap(), 3, false, location).unwrap();
        assert!(operation.remove_attribute(TOP_K_IS_STABLE_ATTRIBUTE));
        assert_eq!(operation.is_stable(), Ok(true));
        assert!(operation.verify());
    }

    #[test]
    fn test_mulhi() {
        let context = Context::new();
        let location = context.unknown_location();
        let input_type =
            context.tensor_type(context.signless_integer_type(32), &[Size::Static(4)], None, location).unwrap();
        let mut block = context.block(&[(input_type, location), (input_type, location)]);
        let operation = mulhi(block.argument(0).unwrap(), block.argument(1).unwrap(), location).unwrap();
        assert!(operation.verify());
        assert_eq!(operation.operand_count(), 2);
        assert_eq!(operation.result(0).unwrap().r#type().unwrap(), input_type.as_ref());
        let operation = block.append_operation(operation).unwrap();
        block.append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap()).unwrap();
        let function = func::func(
            "test_mulhi",
            func::FuncAttributes {
                arguments: vec![input_type.into(), input_type.into()],
                results: vec![input_type.into()],
                ..Default::default()
            },
            block.try_into().unwrap(),
            location,
        )
        .unwrap();
        assert!(function.verify());
        let parsed = context.parse_operation_from_bytes(function.bytecode(), "mulhi.mlir").unwrap();
        assert!(parsed.verify());
        assert_eq!(parsed.to_string(), function.to_string());
    }
}
