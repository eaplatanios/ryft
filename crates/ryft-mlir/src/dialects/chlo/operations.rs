use crate::macros::{mlir_op, mlir_op_trait};
use crate::{
    Attribute, Block, DetachedOp, DetachedRegion, DialectHandle, Error, Location, Operation, OperationBuilder,
    OperationResultRef, Region, SingleBlock, Type, Value, ValueRef,
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
        .add_operand(lhs)?
        .add_operand(rhs)?
        .add_operand(group_sizes)?
        .add_attribute(RAGGED_DOT_DIMENSIONS_ATTRIBUTE, dimensions)?;
    if let Some((lhs_precision, rhs_precision)) = precision {
        builder = builder.add_attribute(
            RAGGED_DOT_PRECISION_ATTRIBUTE,
            context.array_attribute(&[context.chlo_precision(lhs_precision)?, context.chlo_precision(rhs_precision)?]),
        )?;
    }
    builder.add_result(result_type)?.build().and_then(|operation| unsafe {
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
        .add_operand(input)?
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
        .add_operand(input)?
        .add_attribute(TOP_K_COUNT_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), k))?
        .add_attribute(TOP_K_IS_STABLE_ATTRIBUTE, context.boolean_attribute(is_stable))?
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `chlo::top_k`"))
        })
}

/// Name of the attribute storing the sizes of the input and initial value operand segments of [`ScanOperation`].
pub const SCAN_OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Name of the attribute storing the sizes of the output and carry result segments of [`ScanOperation`].
pub const SCAN_RESULT_SEGMENT_SIZES_ATTRIBUTE: &str = "resultSegmentSizes";

/// Name of the [`ScanOperation::dimension`] attribute.
pub const SCAN_DIMENSION_ATTRIBUTE: &str = "dimension";

/// Name of the [`ScanOperation::scan_dimension_size`] attribute.
pub const SCAN_DIMENSION_SIZE_ATTRIBUTE: &str = "scan_dim_size";

/// Name of the [`ScanOperation::is_reverse`] attribute.
pub const SCAN_IS_REVERSE_ATTRIBUTE: &str = "is_reverse";

/// Name of the [`ScanOperation::is_associative`] attribute.
pub const SCAN_IS_ASSOCIATIVE_ATTRIBUTE: &str = "is_associative";

/// CHLO [`Operation`] that scans its [`ScanOperation::inputs`] along [`ScanOperation::dimension`], threading a set
/// of carries through the single-block body region (i.e., [`SingleBlock::body`]). The carries start at
/// [`ScanOperation::initial_values`] and, at each position along the scan dimension, the body receives one slice of
/// every input (i.e., the input with the scan dimension removed) followed by the current carries, and returns one
/// slice of every output followed by the updated carries:
///
/// ```text
/// ^bb0(input_slice_0, ..., input_slice_n, carry_0, ..., carry_m):
///   return output_slice_0, ..., output_slice_k, new_carry_0, ..., new_carry_m
/// ```
///
/// The operation stacks the output slices along the scan dimension to form [`ScanOperation::outputs`] and returns
/// the carries produced by the last step as [`ScanOperation::carries`]. Each body argument must be compatible with
/// the corresponding input slice or initial value type, the number of carries equals the number of initial values,
/// and at least one input or output must be present. The result types are inferred from the body terminator: each
/// output has the type of its returned slice with the scan dimension size inserted at [`ScanOperation::dimension`]
/// and each carry has the type of its returned value.
///
/// All inputs must have the same scan dimension size, which may also be stated explicitly using
/// [`ScanOperation::scan_dimension_size`] (e.g., to provide a static output size when all inputs are dynamic along
/// the scan dimension). When [`ScanOperation::is_reverse`] is true, the scan visits positions from last to first.
/// [`ScanOperation::is_associative`] optionally declares whether the body computes an associative reduction, which
/// allows compilers to use parallel (e.g., work-efficient tree) implementations instead of a sequential loop. This
/// operation currently has no decomposition into StableHLO.
///
/// # Example
///
/// The following is an example of a [`ScanOperation`] that computes a cumulative sum along the second dimension,
/// represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
/// // %init: [0.0, 0.0]
/// %output, %carry = chlo.scan(%input) inits (%init) dimension=1  {
/// ^bb0(%input: tensor<2xf32>, %carry: tensor<2xf32>):
///   %0 = stablehlo.add %input, %carry : tensor<2xf32>
///   stablehlo.return %0, %0 : tensor<2xf32>, tensor<2xf32>
/// } : (tensor<2x3xf32>, tensor<2xf32>) -> (tensor<2x3xf32>, tensor<2xf32>)
/// // %output: [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]]
/// // %carry: [6.0, 15.0]
/// ```
///
/// Refer to the [official CHLO specification](https://openxla.org/stablehlo/generated/chlo#chloscan_chloscanop)
/// and the [XLA `Scan` semantics](https://openxla.org/xla/operation_semantics#scan) for more information.
pub trait ScanOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + SingleBlock<'o, 'c, 't> {
    /// Returns an [`Iterator`] over the inputs that are scanned along [`ScanOperation::dimension`].
    fn inputs(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(SCAN_OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        Ok(range.map(|index| self.operand_value(index)))
    }

    /// Returns an [`Iterator`] over the initial values of the carries.
    fn initial_values(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(SCAN_OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        Ok(range.map(|index| self.operand_value(index)))
    }

    /// Returns an [`Iterator`] over the outputs, which stack the per-step output slices along
    /// [`ScanOperation::dimension`].
    fn outputs(&self) -> Result<impl Iterator<Item = Result<OperationResultRef<'o, 'c, 't>, Error>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(SCAN_RESULT_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        Ok(range.map(|index| self.result(index)))
    }

    /// Returns an [`Iterator`] over the final carries produced by the last scan step.
    fn carries(&self) -> Result<impl Iterator<Item = Result<OperationResultRef<'o, 'c, 't>, Error>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(SCAN_RESULT_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        Ok(range.map(|index| self.result(index)))
    }

    /// Returns the dimension of the inputs along which the scan is performed.
    fn dimension(&self) -> Result<usize, Error> {
        usize::try_from(self.integer_attribute(SCAN_DIMENSION_ATTRIBUTE)?.signed_value())
            .map_err(|_| Error::invalid_argument("invalid `dimension` attribute in `chlo.scan`"))
    }

    /// Returns the explicitly specified size of the scan dimension, if one is present.
    fn scan_dimension_size(&self) -> Result<Option<usize>, Error> {
        if !self.has_attribute(SCAN_DIMENSION_SIZE_ATTRIBUTE) {
            return Ok(None);
        }
        usize::try_from(self.integer_attribute(SCAN_DIMENSION_SIZE_ATTRIBUTE)?.signed_value())
            .map(Some)
            .map_err(|_| Error::invalid_argument("invalid `scan_dim_size` attribute in `chlo.scan`"))
    }

    /// Returns whether the scan visits positions from last to first, defaulting to false when the attribute is
    /// absent.
    fn is_reverse(&self) -> Result<bool, Error> {
        if self.has_attribute(SCAN_IS_REVERSE_ATTRIBUTE) {
            Ok(self.boolean_attribute(SCAN_IS_REVERSE_ATTRIBUTE)?.value())
        } else {
            Ok(false)
        }
    }

    /// Returns whether the body is declared to compute an associative reduction, if that property is specified.
    fn is_associative(&self) -> Result<Option<bool>, Error> {
        if self.has_attribute(SCAN_IS_ASSOCIATIVE_ATTRIBUTE) {
            Ok(Some(self.boolean_attribute(SCAN_IS_ASSOCIATIVE_ATTRIBUTE)?.value()))
        } else {
            Ok(None)
        }
    }
}

mlir_op!(Scan);
mlir_op_trait!(Scan, IsolatedFromAbove);
mlir_op_trait!(Scan, OneRegion);
mlir_op_trait!(Scan, SingleBlock);
mlir_op_trait!(Scan, SingleBlockRegions);
mlir_op_trait!(Scan, ZeroSuccessors);

/// Constructs a new detached/owned [`ScanOperation`] at the specified [`Location`], inferring the output and carry
/// types from the terminator of `body`. The number of outputs is the number of terminator operands minus the number
/// of `initial_values`. Refer to the documentation of [`ScanOperation`] for more information on the operation
/// semantics and on the required body signature.
///
/// # Parameters
///
///   - `inputs`: Ranked tensors that are scanned along `dimension`, all with the same scan dimension size.
///   - `initial_values`: Initial values of the carries, one for each carry that the body threads through the scan.
///   - `dimension`: Dimension of the inputs along which the scan is performed.
///   - `scan_dimension_size`: Optional explicit size of the scan dimension, which must match every input with a
///     static size along `dimension`.
///   - `is_reverse`: Whether the scan visits positions from last to first. The attribute is only attached when this
///     is true because false is its default value.
///   - `is_associative`: Optional declaration of whether `body` computes an associative reduction.
///   - `body`: Single-block region whose arguments are the input slices followed by the carries and whose
///     terminator returns the output slices followed by the updated carries.
///   - `location`: Source location to attach to the operation.
pub fn scan<
    'input,
    'initial_value,
    'c: 'input + 'initial_value,
    't: 'c,
    Input: Value<'input, 'c, 't>,
    InitialValue: Value<'initial_value, 'c, 't>,
    L: Location<'c, 't>,
>(
    inputs: &[Input],
    initial_values: &[InitialValue],
    dimension: usize,
    scan_dimension_size: Option<usize>,
    is_reverse: bool,
    is_associative: Option<bool>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedScanOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::chlo()?)?;

    // Result type inference dereferences the body terminator without checking that it exists, so we validate it here
    // before building the operation. The terminator operand count also determines the size of the output segment.
    let terminator_operand_count = body
        .blocks()?
        .next()
        .transpose()?
        .map(|block| block.terminator())
        .transpose()?
        .flatten()
        .map(|terminator| terminator.operand_count())
        .ok_or_else(|| Error::invalid_argument("the body of `chlo::scan` must end with a terminator"))?;
    let output_count = terminator_operand_count.checked_sub(initial_values.len()).ok_or_else(|| {
        Error::invalid_argument("the body of `chlo::scan` must return at least one value for each initial value")
    })?;
    let segment_size = |size: usize| {
        i32::try_from(size).map_err(|_| Error::invalid_argument("`chlo::scan` segment size exceeds the `i32` range"))
    };
    let operand_segment_sizes = [segment_size(inputs.len())?, segment_size(initial_values.len())?];
    let result_segment_sizes = [segment_size(output_count)?, segment_size(initial_values.len())?];
    let dimension =
        i64::try_from(dimension).map_err(|_| Error::invalid_argument("`dimension` exceeds the signed 64-bit range"))?;

    let mut builder = OperationBuilder::new("chlo.scan", location)
        .add_operands(inputs)?
        .add_operands(initial_values)?
        .add_attribute(
            SCAN_OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&operand_segment_sizes)?,
        )?
        .add_attribute(SCAN_RESULT_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&result_segment_sizes)?)?
        .add_attribute(
            SCAN_DIMENSION_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), dimension),
        )?;
    if let Some(scan_dimension_size) = scan_dimension_size {
        let scan_dimension_size = i64::try_from(scan_dimension_size)
            .map_err(|_| Error::invalid_argument("`scan_dimension_size` exceeds the signed 64-bit range"))?;
        builder = builder.add_attribute(
            SCAN_DIMENSION_SIZE_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), scan_dimension_size),
        )?;
    }
    if is_reverse {
        builder = builder.add_attribute(SCAN_IS_REVERSE_ATTRIBUTE, context.boolean_attribute(true))?;
    }
    if let Some(is_associative) = is_associative {
        builder = builder.add_attribute(SCAN_IS_ASSOCIATIVE_ATTRIBUTE, context.boolean_attribute(is_associative))?;
    }
    builder.add_region(body)?.enable_result_type_inference().build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `chlo::scan`"))
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
        .add_result(lhs.r#type()?)?
        .add_operand(lhs)?
        .add_operand(rhs)?
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
    use crate::dialects::{func, stable_hlo};
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
    fn test_scan() {
        let context = Context::new();
        let location = context.unknown_location();
        let input_type = context
            .tensor_type(context.float32_type(), &[Size::Static(4), Size::Static(8)], None, location)
            .unwrap();
        let carry_type = context.tensor_type(context.float32_type(), &[Size::Static(4)], None, location).unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(input_type, location), (carry_type, location)]);
                let input = block.argument(0).unwrap();
                let initial_value = block.argument(1).unwrap();
                let mut body_block = context.block(&[(carry_type, location), (carry_type, location)]);
                let sum = stable_hlo::add(body_block.argument(0).unwrap(), body_block.argument(1).unwrap(), location)
                    .unwrap();
                let sum = body_block.append_operation(sum).unwrap();
                body_block
                    .append_operation(
                        stable_hlo::r#return(&[sum.result(0).unwrap(), sum.result(0).unwrap()], location).unwrap(),
                    )
                    .unwrap();
                let operation =
                    scan(&[input], &[initial_value], 1, None, false, None, body_block.try_into().unwrap(), location)
                        .unwrap();
                assert_eq!(operation.inputs().unwrap().collect::<Result<Vec<_>, _>>().unwrap(), vec![input]);
                assert_eq!(
                    operation.initial_values().unwrap().collect::<Result<Vec<_>, _>>().unwrap(),
                    vec![initial_value],
                );
                let outputs = operation.outputs().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
                let carries = operation.carries().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
                assert_eq!(outputs.len(), 1);
                assert_eq!(carries.len(), 1);
                assert_eq!(outputs[0].r#type().unwrap(), input_type.as_ref());
                assert_eq!(carries[0].r#type().unwrap(), carry_type.as_ref());
                assert_eq!(operation.dimension(), Ok(1));
                assert_eq!(operation.scan_dimension_size(), Ok(None));
                assert_eq!(operation.is_reverse(), Ok(false));
                assert_eq!(operation.is_associative(), Ok(None));
                assert_eq!(operation.body().unwrap().argument_count(), 2);
                let operation = block.append_operation(operation).unwrap();
                block
                    .append_operation(
                        func::r#return(&[operation.result(0).unwrap(), operation.result(1).unwrap()], location)
                            .unwrap(),
                    )
                    .unwrap();
                func::func(
                    "scan_test",
                    func::FuncAttributes {
                        arguments: vec![input_type.into(), carry_type.into()],
                        results: vec![input_type.into(), carry_type.into()],
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
                  func.func @scan_test(%arg0: tensor<4x8xf32>, %arg1: tensor<4xf32>) \
                      -> (tensor<4x8xf32>, tensor<4xf32>) {
                    %0:2 = chlo.scan(%arg0) inits (%arg1) dimension=1  {
                    ^bb0(%input: tensor<4xf32>, %carry: tensor<4xf32>):
                      %1 = stablehlo.add %input, %carry : tensor<4xf32>
                      stablehlo.return %1, %1 : tensor<4xf32>, tensor<4xf32>
                    } : (tensor<4x8xf32>, tensor<4xf32>) -> (tensor<4x8xf32>, tensor<4xf32>)
                    return %0#0, %0#1 : tensor<4x8xf32>, tensor<4xf32>
                  }
                }
            "},
        );

        // Check the optional attributes and the body validation.
        let block = context.block(&[(input_type, location), (carry_type, location)]);
        let mut body_block = context.block(&[(carry_type, location), (carry_type, location)]);
        let sum = stable_hlo::add(body_block.argument(0).unwrap(), body_block.argument(1).unwrap(), location).unwrap();
        let sum = body_block.append_operation(sum).unwrap();
        body_block
            .append_operation(
                stable_hlo::r#return(&[sum.result(0).unwrap(), sum.result(0).unwrap()], location).unwrap(),
            )
            .unwrap();
        let operation = scan(
            &[block.argument(0).unwrap()],
            &[block.argument(1).unwrap()],
            1,
            Some(8),
            true,
            Some(true),
            body_block.try_into().unwrap(),
            location,
        )
        .unwrap();
        assert!(operation.verify());
        assert_eq!(operation.scan_dimension_size(), Ok(Some(8)));
        assert_eq!(operation.is_reverse(), Ok(true));
        assert_eq!(operation.is_associative(), Ok(Some(true)));
        assert!(matches!(
            scan(
                &[block.argument(0).unwrap()],
                &[block.argument(1).unwrap()],
                1,
                None,
                false,
                None,
                context.block(&[(carry_type, location), (carry_type, location)]).try_into().unwrap(),
                location,
            ),
            Err(Error::InvalidArgument { message, .. })
                if message == "the body of `chlo::scan` must end with a terminator",
        ));
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
