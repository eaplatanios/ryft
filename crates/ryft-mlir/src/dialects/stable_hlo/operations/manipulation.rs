use ryft_xla_sys::bindings::{MlirAttribute, stablehloGatherDimensionNumbersGet, stablehloScatterDimensionNumbersGet};

use crate::{
    Attribute, BooleanAttributeRef, Context, DenseInteger64ArrayAttributeRef, DetachedOp, DetachedRegion,
    DialectHandle, IntegerAttributeRef, Location, OneRegion, Operation, OperationBuilder, RegionRef, ShapedType, Size,
    TensorTypeRef, Type, Value, ValueRef, mlir_attribute_field, mlir_op, mlir_op_trait, mlir_subtype_trait_impls,
};

use super::{HasPadding, PADDING_ATTRIBUTE};

/// Name of the [`Attribute`] that is used to store [`GetDimensionSizeOperation::dimension`].
pub const GET_DIMENSION_SIZE_DIMENSION_ATTRIBUTE: &str = "dimension";

/// StableHLO [`Operation`] that returns the size of dimension [`GetDimensionSizeOperation::dimension`] of its input
/// tensor as an `i32` scalar [`TensorTypeRef`].
///
/// # Example
///
/// The following is an example of a [`GetDimensionSizeOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [[1, 2, 3], [4, 5, 6]]
/// %output = stablehlo.get_dimension_size %input, dim = 1 : (tensor<2x3xi64>) -> tensor<i32>
/// // %output: 3
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#get_dimension_size)
/// for more information.
pub trait GetDimensionSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the dimension whose size this [`GetDimensionSizeOperation`] extracts.
    fn dimension(&self) -> usize {
        self.attribute(GET_DIMENSION_SIZE_DIMENSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{GET_DIMENSION_SIZE_DIMENSION_ATTRIBUTE}' attribute in `stable_hlo::get_dimension_size`"))
            .signless_value() as usize
    }
}

mlir_op!(GetDimensionSize);
mlir_op_trait!(GetDimensionSize, OneOperand);
mlir_op_trait!(GetDimensionSize, OneResult);
mlir_op_trait!(GetDimensionSize, ZeroRegions);
mlir_op_trait!(GetDimensionSize, ZeroSuccessors);

/// Constructs a new detached/owned [`GetDimensionSizeOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`GetDimensionSizeOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn get_dimension_size<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    dimension: usize,
    location: L,
) -> DetachedGetDimensionSizeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.get_dimension_size", location)
        .add_operand(input)
        .add_attribute(
            GET_DIMENSION_SIZE_DIMENSION_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), dimension as i64),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::get_dimension_size`")
}

/// Name of the [`Attribute`] that is used to store [`TransposeOperation::permutation`].
pub const TRANSPOSE_PERMUTATION_ATTRIBUTE: &str = "permutation";

/// StableHLO [`Operation`] that permutes the dimensions of its input tensor according to
/// [`TransposeOperation::permutation`]. More formally, `result[result_index] = operand[operand_index]` where
/// `result_index[d] = operand_index[permutation[d]]`.
///
/// # Example
///
/// The following is an example of a [`TransposeOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [
/// //          [[1,2], [3,4], [5,6]],
/// //          [[7,8], [9,10], [11,12]]
/// //         ]
/// %output = "tablehlo.transpose %input, dims = [2, 1, 0] : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
/// // %output: [
/// //           [[1,7], [3,9], [5,11]],
/// //           [[2,8], [4,10], [6,12]]
/// //          ]
/// ```
///
/// Refer to the [StableHLO specification](https://openxla.org/stablehlo/spec#transpose) for more information.
pub trait TransposeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the dimension permutation of this [`TransposeOperation`].
    fn permutation(&self) -> Vec<usize> {
        self.attribute(TRANSPOSE_PERMUTATION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
            .unwrap_or_else(|| panic!("invalid '{TRANSPOSE_PERMUTATION_ATTRIBUTE}' attribute in `stable_hlo::transpose`"))
    }
}

mlir_op!(Transpose);
mlir_op_trait!(Transpose, OneOperand);
mlir_op_trait!(Transpose, OneResult);
mlir_op_trait!(Transpose, ZeroRegions);
mlir_op_trait!(Transpose, ZeroSuccessors);

/// Constructs a new detached/owned [`TransposeOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`TransposeOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn transpose<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    permutation: &[usize],
    location: L,
) -> DetachedTransposeOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.transpose", location)
        .add_operand(input)
        .add_attribute(
            TRANSPOSE_PERMUTATION_ATTRIBUTE,
            location
                .context()
                .dense_i64_array_attribute(permutation.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::transpose`")
}

/// StableHLO [`Operation`] that reshapes its input tensor while keeping the number of elements it contains fixed.
///
/// # Example
///
/// The following is an example of a [`ReshapeOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [[1, 2, 3], [4, 5, 6]]
/// %output = stablehlo.reshape %input : (tensor<2x3xi32>) -> tensor<3x2xi32>
/// // %output: [[1, 2], [3, 4], [5, 6]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#reshape)
/// for more information.
pub trait ReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the target shape of this [`ReshapeOperation`].
    fn shape(&self) -> Vec<Size> {
        self.result(0).unwrap().r#type().cast::<TensorTypeRef>().unwrap().dimensions().collect()
    }
}

mlir_op!(Reshape);
mlir_op_trait!(Reshape, OneOperand);
mlir_op_trait!(Reshape, OneResult);
mlir_op_trait!(Reshape, ZeroRegions);
mlir_op_trait!(Reshape, ZeroSuccessors);

/// Constructs a new detached/owned [`ReshapeOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ReshapeOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn reshape<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    shape: &[usize],
    location: L,
) -> DetachedReshapeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let element_type = input.r#type().cast::<TensorTypeRef>().unwrap().element_type();
    OperationBuilder::new("stablehlo.reshape", location)
        .add_operand(input)
        .add_result(
            context
                .tensor_type(
                    element_type,
                    shape.iter().map(|size| Size::Static(*size)).collect::<Vec<_>>().as_slice(),
                    None,
                    location,
                )
                .unwrap(),
        )
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::reshape`")
}

/// StableHLO [`Operation`] that reshapes its input tensor while keeping the number of elements it contains fixed.
/// Semantically, this operation is equivalent to [`ReshapeOperation`] except for the fact that the output shape is
/// not statically known and is instead provided dynamically via its second input/operand.
///
/// # Example
///
/// The following is an example of a [`DynamicReshapeOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [[1, 2, 3], [4, 5, 6]]
/// // %shape: [3, 2]
/// %output = stablehlo.dynamic_reshape %input, %shape : (tensor<2x3xi64>, tensor<2xi64>) -> tensor<?x?xi64>
/// // %output: [[1, 2], [3, 4], [5, 6]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#dynamic_reshape)
/// for more information.
pub trait DynamicReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input of this [`DynamicReshapeOperation`].
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the target shape of this [`DynamicReshapeOperation`].
    fn shape(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(DynamicReshape);
mlir_op_trait!(DynamicReshape, OneResult);
mlir_op_trait!(DynamicReshape, ZeroRegions);
mlir_op_trait!(DynamicReshape, ZeroSuccessors);

/// Constructs a new detached/owned [`DynamicReshapeOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DynamicReshapeOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn dynamic_reshape<
    'input,
    'shape,
    'c: 'input + 'shape,
    't: 'c,
    Input: Value<'input, 'c, 't>,
    Shape: Value<'shape, 'c, 't>,
    L: Location<'c, 't>,
>(
    input: Input,
    shape: Shape,
    location: L,
) -> DetachedDynamicReshapeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let input_type = input
        .r#type()
        .cast::<TensorTypeRef>()
        .expect("invalid arguments to `stable_hlo::dynamic_reshape`; the input is not a tensor");
    let element_type = input_type.element_type();
    let output_shape = input_type.dimensions().map(|_| Size::Dynamic).collect::<Vec<_>>();
    OperationBuilder::new("stablehlo.dynamic_reshape", location)
        .add_operand(input)
        .add_operand(shape)
        .add_result(context.tensor_type(element_type, output_shape.as_slice(), None, location).unwrap())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::dynamic_reshape`")
}

/// Name of the [`Attribute`] that is used to store [`BroadcastOperation::dimensions`]
/// and [`DynamicBroadcastOperation::dimensions`].
pub const BROADCAST_DIMENSIONS_ATTRIBUTE: &str = "broadcast_dimensions";

/// StableHLO [`Operation`] that expands the dimensions and/or rank of an input tensor by duplicating data
/// in the operand tensor. The operation produces a result tensor with a potentially higher rank and/or larger
/// dimensions than the input.
///
/// [`BroadcastOperation::dimensions`] specifies how the dimensions of the operand map to the dimensions of the result.
/// For each dimension `d` in the operand, `broadcast_dimensions[d]` indicates which dimension in the result corresponds
/// to dimension `d` in the operand. The size of `broadcast_dimensions` must thus also equal the rank of the operand.
///
/// The semantics of broadcasting are implemented as `result[result_index] = operand[operand_index]`,
/// where `operand_index[d]` is computed based on whether the operand's dimension `d` has size 1:
/// 
///   - If `dim(operand, d) = 1`, then `operand_index[d] = 0` (broadcast across this dimension).
///   - Otherwise, `operand_index[d] = result_index[broadcast_dimensions[d]]` (direct mapping).
/// 
/// This is computed over all values of `d` in `axes(operand)`.
///
/// To better understand the semantics of this operation, let us consider an operand with type `tensor<1x3xi32>`,
/// an output type of `tensor<2x3x2xi32>`, and the following broadcast dimensions `[2, 1]`. In this case, we have
/// the following:
///
///   1. Dimension `0` in the operand maps to dimension `2` in the output (and is broadcasted because we need to map a
///      size `1` dimension to a size `2` dimension).
///   2. Dimension `1` in the operand maps to dimension `1` in the output (and no broadcasting takes place since the
///      sizes are both `3`).
///   3. Dimension `0` of the result is a new dimension, as it not mapped from any dimensions in the operand) and the
///      tensor is broadcasted along this dimension, resulting in `result[0, :, :] == result[1, :, :]`.
///
/// # Example
///
/// The following is an example of a [`BroadcastOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [[1, 2, 3]]
/// %output = stablehlo.broadcast_in_dim %input, dims = [2, 1] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
/// // %output: [
/// //   [[1, 1], [2, 2], [3, 3]],
/// //   [[1, 1], [2, 2], [3, 3]]
/// // ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#broadcast_in_dim)
/// for more information.
pub trait BroadcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the broadcast dimensions of this [`BroadcastOperation`], which specify
    /// how the dimensions of the operand map to the dimensions of the result.
    fn dimensions(&self) -> Vec<usize> {
        self.attribute(BROADCAST_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{BROADCAST_DIMENSIONS_ATTRIBUTE}' attribute in `stable_hlo::broadcast`"))
            .values()
            .map(|value| value as usize)
            .collect()
    }
}

mlir_op!(Broadcast);
mlir_op_trait!(Broadcast, OneOperand);
mlir_op_trait!(Broadcast, OneResult);
mlir_op_trait!(Broadcast, ZeroRegions);
mlir_op_trait!(Broadcast, ZeroSuccessors);

/// Constructs a new detached/owned [`BroadcastOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`BroadcastOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn broadcast<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: V,
    output_type: T,
    dimensions: &[usize],
    location: L,
) -> DetachedBroadcastOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.broadcast_in_dim", location)
        .add_operand(input)
        .add_attribute(
            BROADCAST_DIMENSIONS_ATTRIBUTE,
            context
                .dense_i64_array_attribute(dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        )
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::broadcast`")
}

/// Name of the [`Attribute`] that is used to store [`DynamicBroadcastOperation::known_expanding_dimensions`].
pub const DYNAMIC_BROADCAST_KNOWN_EXPANDING_DIMENSIONS_ATTRIBUTE: &str = "known_expanding_dimensions";

/// Name of the [`Attribute`] that is used to store [`DynamicBroadcastOperation::known_non_expanding_dimensions`].
pub const DYNAMIC_BROADCAST_KNOWN_NON_EXPANDING_DIMENSIONS_ATTRIBUTE: &str = "known_nonexpanding_dimensions";

/// StableHLO [`Operation`] that expands the dimensions and/or rank of an input tensor by duplicating data
/// in the operand tensor. Semantically, this operation is equivalent to [`BroadcastOperation`] except for the fact
/// that the output shape is not statically known and is instead provided dynamically via its second input/operand.
///
/// This operation also has optional attributes to express static knowledge about the expanding behavior of dimensions
/// (i.e., [`DynamicBroadcastOperation::known_expanding_dimensions`] and
/// [`DynamicBroadcastOperation::known_non_expanding_dimensions`]). If not specified, then all dimensions are assumed
/// to be possibly expanding. The sets of dimensions that are known to be expanding and the set of dimensions that are
/// known to be non-expanding must be disjoint, and they must be a subset of the input tensor's dimensions.
///
/// # Example
///
/// The following is an example of a [`DynamicBroadcastOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [[1, 2, 3]]
/// %input = stablehlo.constant dense<[[1, 2, 3]]> : tensor<1x3xi64>
/// %shape = stablehlo.constant dense<[2, 3, 2]> : tensor<3xi64>
/// %output = stablehlo.dynamic_broadcast_in_dim %input, %shape, dims = [2, 1] {
///   known_expanding_dimensions = array<i64: 0>,
///   known_nonexpanding_dimensions = array<i64: 1>
/// } : (tensor<1x3xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
/// // %output: [
/// //            [
/// //             [1, 1],
/// //             [2, 2],
/// //             [3, 3]
/// //            ],
/// //            [
/// //             [1, 1],
/// //             [2, 2],
/// //             [3, 3]
/// //            ]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#dynamic_broadcast_in_dim)
/// for more information.
pub trait DynamicBroadcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input of this [`DynamicBroadcastOperation`].
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the target shape of this [`DynamicBroadcastOperation`].
    fn shape(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns the broadcast dimensions of this [`DynamicBroadcastOperation`], which specify
    /// how the dimensions of the operand map to the dimensions of the result.
    fn dimensions(&self) -> Vec<usize> {
        self.attribute(BROADCAST_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{BROADCAST_DIMENSIONS_ATTRIBUTE}' attribute in `stable_hlo::dynamic_broadcast`"))
            .values()
            .map(|value| value as usize)
            .collect()
    }

    /// Returns the known expanding dimensions of this [`DynamicBroadcastOperation`].
    fn known_expanding_dimensions(&self) -> Option<Vec<usize>> {
        self.attribute(DYNAMIC_BROADCAST_KNOWN_EXPANDING_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
    }

    /// Returns the known non-expanding dimensions of this [`DynamicBroadcastOperation`].
    fn known_non_expanding_dimensions(&self) -> Option<Vec<usize>> {
        self.attribute(DYNAMIC_BROADCAST_KNOWN_NON_EXPANDING_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
    }
}

mlir_op!(DynamicBroadcast);
mlir_op_trait!(DynamicBroadcast, OneResult);
mlir_op_trait!(DynamicBroadcast, ZeroRegions);
mlir_op_trait!(DynamicBroadcast, ZeroSuccessors);

/// Constructs a new detached/owned [`DynamicBroadcastOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DynamicBroadcastOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn dynamic_broadcast<
    'input,
    'shape,
    'c: 'input + 'shape,
    't: 'c,
    Input: Value<'input, 'c, 't>,
    Shape: Value<'shape, 'c, 't>,
    L: Location<'c, 't>,
>(
    input: Input,
    shape: Shape,
    dimensions: &[usize],
    known_expanding_dimensions: Option<&[usize]>,
    known_non_expanding_dimensions: Option<&[usize]>,
    location: L,
) -> DetachedDynamicBroadcastOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let input_type = input
        .r#type()
        .cast::<TensorTypeRef>()
        .expect("invalid arguments to `stable_hlo::dynamic_broadcast`; `input` is not a tensor");
    let element_type = input_type.element_type();
    let output_rank = shape
        .r#type()
        .cast::<TensorTypeRef>()
        .expect("invalid arguments to `stable_hlo::dynamic_broadcast`; `shape` is not a tensor")
        .dimension(0)
        .value()
        .expect(
            "invalid arguments to `stable_hlo::dynamic_broadcast`; \
            `shape` must be one-dimensional and have a statically known length",
        );
    let output_shape = (0..output_rank).map(|_| Size::Dynamic).collect::<Vec<_>>();
    let mut builder = OperationBuilder::new("stablehlo.dynamic_broadcast_in_dim", location)
        .add_operand(input)
        .add_operand(shape)
        .add_attribute(
            BROADCAST_DIMENSIONS_ATTRIBUTE,
            context
                .dense_i64_array_attribute(dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        );
    if let Some(known_expanding_dimensions) = known_expanding_dimensions {
        builder = builder.add_attribute(
            DYNAMIC_BROADCAST_KNOWN_EXPANDING_DIMENSIONS_ATTRIBUTE,
            context
                .dense_i64_array_attribute(
                    known_expanding_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice(),
                )
                .unwrap(),
        );
    }
    if let Some(known_non_expanding_dimensions) = known_non_expanding_dimensions {
        builder = builder.add_attribute(
            DYNAMIC_BROADCAST_KNOWN_NON_EXPANDING_DIMENSIONS_ATTRIBUTE,
            context
                .dense_i64_array_attribute(
                    known_non_expanding_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice(),
                )
                .unwrap(),
        );
    }
    builder
        .add_result(context.tensor_type(element_type, output_shape.as_slice(), None, location).unwrap())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::dynamic_broadcast`")
}

/// Name of the [`Attribute`] that is used to store [`PadOperation::edge_padding_low`].
pub const EDGE_PADDING_LOW_ATTRIBUTE: &str = "edge_padding_low";

/// Name of the [`Attribute`] that is used to store [`PadOperation::edge_padding_high`].
pub const EDGE_PADDING_HIGH_ATTRIBUTE: &str = "edge_padding_high";

/// Name of the [`Attribute`] that is used to store [`PadOperation::interior_padding`].
pub const INTERIOR_PADDING_ATTRIBUTE: &str = "interior_padding";

/// StableHLO [`Operation`] that expands a tensor by adding padding around and between its elements.
///
/// [`PadOperation::edge_padding_low`] and [`PadOperation::edge_padding_high`] specify the amount of padding added at
/// the low-end (next to index 0) and the high-end (next to the highest index) of each dimension respectively. The
/// amount of padding can be negative, where the absolute value of negative padding indicates the number of elements
/// to remove from the specified dimension.
///
/// [`PadOperation::interior_padding`] specifies the amount of padding added between any two elements in each dimension
/// which may not be negative. Interior padding occurs before edge padding such that negative edge padding will remove
/// elements from the interior-padded operand.
///
/// More formally, `result[result_index]` is defined as `operand[operand_index]` if
/// `result_index = edge_padding_low + operand_index * (interior_padding + 1)`, and
/// [`PadOperation::padding_value`] otherwise.
///
/// # Example
///
/// The following is an example of a [`PadOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [
/// //          [1, 2, 3],
/// //          [4, 5, 6]
/// //         ]
/// // %padding_value: 0
/// %result = stablehlo.pad
///   %input,
///   %padding_value,
///   low = [0, 1],
///   high = [2, 1],
///   interior = [1, 2]
/// : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
/// // %result: [
/// //           [0, 1, 0, 0, 2, 0, 0, 3, 0],
/// //           [0, 0, 0, 0, 0, 0, 0, 0, 0],
/// //           [0, 4, 0, 0, 5, 0, 0, 6, 0],
/// //           [0, 0, 0, 0, 0, 0, 0, 0, 0],
/// //           [0, 0, 0, 0, 0, 0, 0, 0, 0]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#pad) for more information.
pub trait PadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input of this [`PadOperation`].
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the padding value of this [`PadOperation`].
    fn padding_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns the edge "low" padding amounts for this [`PadOperation`].
    fn edge_padding_low(&self) -> Vec<i64> {
        self.attribute(EDGE_PADDING_LOW_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{EDGE_PADDING_LOW_ATTRIBUTE}' attribute in `stable_hlo::pad`"))
            .values()
            .collect()
    }

    /// Returns the edge "high" padding amounts for this [`PadOperation`].
    fn edge_padding_high(&self) -> Vec<i64> {
        self.attribute(EDGE_PADDING_HIGH_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{EDGE_PADDING_HIGH_ATTRIBUTE}' attribute in `stable_hlo::pad`"))
            .values()
            .collect()
    }

    /// Returns the interior padding amounts for this [`PadOperation`].
    fn interior_padding(&self) -> Vec<usize> {
        self.attribute(INTERIOR_PADDING_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{INTERIOR_PADDING_ATTRIBUTE}' attribute in `stable_hlo::pad`"))
            .values()
            .map(|value| value as usize)
            .collect()
    }
}

mlir_op!(Pad);
mlir_op_trait!(Pad, OneResult);
mlir_op_trait!(Pad, ZeroRegions);
mlir_op_trait!(Pad, ZeroSuccessors);

/// Constructs a new detached/owned [`PadOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`PadOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn pad<
    'input,
    'padding_value,
    'c: 'input + 'padding_value,
    't: 'c,
    Input: Value<'input, 'c, 't>,
    PaddingValue: Value<'padding_value, 'c, 't>,
    L: Location<'c, 't>,
>(
    input: Input,
    padding_value: PaddingValue,
    edge_padding_low: &[i64],
    edge_padding_high: &[i64],
    interior_padding: &[usize],
    location: L,
) -> DetachedPadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.pad", location)
        .add_operand(input)
        .add_operand(padding_value)
        .add_attribute(EDGE_PADDING_LOW_ATTRIBUTE, context.dense_i64_array_attribute(edge_padding_low).unwrap())
        .add_attribute(EDGE_PADDING_HIGH_ATTRIBUTE, context.dense_i64_array_attribute(edge_padding_high).unwrap())
        .add_attribute(
            INTERIOR_PADDING_ATTRIBUTE,
            context
                .dense_i64_array_attribute(interior_padding.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::pad`")
}

/// StableHLO [`Operation`] that expands a tensor by adding padding around and between its elements. Semantically,
/// this operation is equivalent to [`PadOperation`] except for the fact that [`PadOperation::edge_padding_low`],
/// [`PadOperation::edge_padding_high`], and [`PadOperation::interior_padding`] are not statically known and are
/// instead provided dynamically via three additional inputs/operands.
///
/// # Example
///
/// The following is an example of a [`DynamicPadOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [
/// //          [1, 2, 3],
/// //          [4, 5, 6]
/// //         ]
/// // %padding_value: 0
/// // %edge_padding_low: [0, 1]
/// // %edge_padding_high: [2, 1]
/// // %interior_padding: [1, 2]
/// %result = stablehlo.dynamic_pad %input, %padding_value, %edge_padding_low, %edge_padding_high, %interior_padding
///   : (tensor<2x3xi64>, tensor<i64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<5x9xi64>
/// // %result: [
/// //           [0, 1, 0, 0, 2, 0, 0, 3, 0],
/// //           [0, 0, 0, 0, 0, 0, 0, 0, 0],
/// //           [0, 4, 0, 0, 5, 0, 0, 6, 0],
/// //           [0, 0, 0, 0, 0, 0, 0, 0, 0],
/// //           [0, 0, 0, 0, 0, 0, 0, 0, 0]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#dynamic_pad)
/// for more information.
pub trait DynamicPadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input of this [`DynamicPadOperation`].
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the padding value of this [`DynamicPadOperation`].
    fn padding_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns the edge "low" padding amounts for this [`DynamicPadOperation`].
    fn edge_padding_low(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(2).unwrap()
    }

    /// Returns the edge "high" padding amounts for this [`DynamicPadOperation`].
    fn edge_padding_high(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(3).unwrap()
    }

    /// Returns the interior padding amounts for this [`DynamicPadOperation`].
    fn interior_padding(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(4).unwrap()
    }
}

mlir_op!(DynamicPad);
mlir_op_trait!(DynamicPad, OneResult);
mlir_op_trait!(DynamicPad, ZeroRegions);
mlir_op_trait!(DynamicPad, ZeroSuccessors);

/// Constructs a new detached/owned [`DynamicPadOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DynamicPadOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn dynamic_pad<
    'input,
    'padding_value,
    'edge_padding_low,
    'edge_padding_high,
    'interior_padding,
    'c: 'input + 'padding_value + 'edge_padding_low + 'edge_padding_high + 'interior_padding,
    't: 'c,
    Input: Value<'input, 'c, 't>,
    PaddingValue: Value<'padding_value, 'c, 't>,
    EdgePaddingLow: Value<'edge_padding_low, 'c, 't>,
    EdgePaddingHigh: Value<'edge_padding_high, 'c, 't>,
    InteriorPadding: Value<'interior_padding, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: Input,
    padding_value: PaddingValue,
    edge_padding_low: EdgePaddingLow,
    edge_padding_high: EdgePaddingHigh,
    interior_padding: InteriorPadding,
    output_type: T,
    location: L,
) -> DetachedDynamicPadOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.dynamic_pad", location)
        .add_operand(input)
        .add_operand(padding_value)
        .add_operand(edge_padding_low)
        .add_operand(edge_padding_high)
        .add_operand(interior_padding)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::dynamic_pad`")
}

/// Name of the [`Attribute`] that is used to store [`ConcatenateOperation::dimension`].
pub const CONCATENATE_DIMENSION_ATTRIBUTE: &str = "dimension";

/// StableHLO [`Operation`] that concatenates multiple input tensors along a specified dimension,
/// in the same order as provided in the inputs to this operation.
///
/// # Example
///
/// The following is an example of a [`ConcatenateOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input_0: [[1, 2], [3, 4], [5, 6]]
/// // %input_1: [[7, 8]]
/// %result = stablehlo.concatenate %input_0, %input_1, dim = 0
///   : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
/// // %result: [[1, 2], [3, 4], [5, 6], [7, 8]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#concatenate)
/// for more information.
pub trait ConcatenateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns an [`Iterator`] over the inputs of this [`Operation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn inputs<'r>(&'r self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        self.operands()
    }

    /// Returns the dimension along which the input tensors will be concatenated in this [`ConcatenateOperation`].
    fn dimension(&self) -> usize {
        self.attribute(CONCATENATE_DIMENSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{CONCATENATE_DIMENSION_ATTRIBUTE}' attribute in `stable_hlo::concatenate`"))
            .signless_value() as usize
    }
}

mlir_op!(Concatenate);
mlir_op_trait!(Concatenate, OneResult);
mlir_op_trait!(Concatenate, ZeroRegions);
mlir_op_trait!(Concatenate, ZeroSuccessors);

/// Constructs a new detached/owned [`ConcatenateOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ConcatenateOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn concatenate<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    inputs: &[V],
    dimension: usize,
    location: L,
) -> DetachedConcatenateOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.concatenate", location)
        .add_operands(inputs)
        .add_attribute(
            CONCATENATE_DIMENSION_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), dimension as i64),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::concatenate`")
}

/// Name of the [`Attribute`] that is used to store [`SliceOperation::start_indices`].
pub const SLICE_START_INDICES_ATTRIBUTE: &str = "start_indices";

/// Name of the [`Attribute`] that is used to store [`SliceOperation::limit_indices`].
pub const SLICE_LIMIT_INDICES_ATTRIBUTE: &str = "limit_indices";

/// Name of the [`Attribute`] that is used to store [`SliceOperation::strides`].
pub const SLICE_STRIDES_ATTRIBUTE: &str = "strides";

/// StableHLO [`Operation`] that extracts a slice from its input/operand tensor using statically-specified
/// [`SliceOperation::start_indices`], [`SliceOperation::limit_indices`], and [`SliceOperation::strides`].
/// start indices, limit indices, and strides.
///
/// # Example
///
/// The following is an example of a [`SliceOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [
/// //          [0, 0, 0, 0],
/// //          [0, 0, 1, 1],
/// //          [0, 0, 1, 1]
/// //         ]
/// %result = stablehlo.slice %input [1:3, 2:4] : (tensor<3x4xi64>) -> tensor<2x2xi64>
/// // % result: [
/// //            [1, 1],
/// //            [1, 1]
/// //           ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#slice)
/// for more information.
pub trait SliceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the start indices for this [`SliceOperation`].
    fn start_indices(&self) -> Vec<usize> {
        self.attribute(SLICE_START_INDICES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{SLICE_START_INDICES_ATTRIBUTE}' attribute in `stable_hlo::slice`"))
            .values()
            .map(|value| value as usize)
            .collect()
    }

    /// Returns the limit indices for this [`SliceOperation`].
    fn limit_indices(&self) -> Vec<usize> {
        self.attribute(SLICE_LIMIT_INDICES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{SLICE_LIMIT_INDICES_ATTRIBUTE}' attribute in `stable_hlo::slice`"))
            .values()
            .map(|value| value as usize)
            .collect()
    }

    /// Returns the strides for this [`SliceOperation`].
    fn strides(&self) -> Vec<usize> {
        self.attribute(SLICE_STRIDES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{SLICE_STRIDES_ATTRIBUTE}' attribute in `stable_hlo::slice`"))
            .values()
            .map(|value| value as usize)
            .collect()
    }
}

mlir_op!(Slice);
mlir_op_trait!(Slice, OneOperand);
mlir_op_trait!(Slice, OneResult);
mlir_op_trait!(Slice, ZeroRegions);
mlir_op_trait!(Slice, ZeroSuccessors);

/// Constructs a new detached/owned [`SliceOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`SliceOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn slice<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    start_indices: &[usize],
    limit_indices: &[usize],
    strides: &[usize],
    location: L,
) -> DetachedSliceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.slice", location)
        .add_operand(input)
        .add_attribute(
            SLICE_START_INDICES_ATTRIBUTE,
            context
                .dense_i64_array_attribute(start_indices.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        )
        .add_attribute(
            SLICE_LIMIT_INDICES_ATTRIBUTE,
            context
                .dense_i64_array_attribute(limit_indices.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        )
        .add_attribute(
            SLICE_STRIDES_ATTRIBUTE,
            context
                .dense_i64_array_attribute(strides.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::slice`")
}

/// Name of the [`Attribute`] that is used to store [`DynamicSliceOperation::slice_sizes`].
pub const DYNAMIC_SLICE_SLICE_SIZES_ATTRIBUTE: &str = "slice_sizes";

/// StableHLO [`Operation`] that extracts a slice from a tensor using dynamically-computed start indices.
/// [`DynamicSliceOperation::start_indices`] contains the start indices of the slice for each dimension, subject to
/// potential adjustment, and [`DynamicSliceOperation::slice_sizes`] contains the sizes of the slice for each dimension.
/// More formally, `result[result_index] = operand[operand_index]`, where:
///
/// ```text
/// adjusted_start_indices = clamp(0, start_indices, shape(operand) - slice_sizes)
/// operand_index = adjusted_start_indices + result_index
/// ```
///
/// # Example
///
/// The following is an example of a [`DynamicSliceOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [
/// //          [0, 0, 1, 1],
/// //          [0, 0, 1, 1],
/// //          [0, 0, 0, 0],
/// //          [0, 0, 0, 0]
/// //         ]
/// // %start_indices_0: -1
/// // %start_indices_1: 3
/// %result = stablehlo.dynamic_slice %input, %start_indices_0, %start_indices_1, sizes = [2, 2]
///   : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x2xi32>
/// // %result: [
/// //           [1, 1],
/// //           [1, 1]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#dynamic_slice)
/// for more information.
pub trait DynamicSliceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input of this [`DynamicSliceOperation`].
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the start indices of this [`DynamicSliceOperation`].
    fn start_indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (1..self.operand_count()).flat_map(|index| self.operand(index)).collect()
    }

    /// Returns the slice sizes for this [`DynamicSliceOperation`].
    fn slice_sizes(&self) -> Vec<usize> {
        self.attribute(DYNAMIC_SLICE_SLICE_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{DYNAMIC_SLICE_SLICE_SIZES_ATTRIBUTE}' attribute in `stable_hlo::dynamic_slice`"))
            .values()
            .map(|value| value as usize)
            .collect()
    }
}

mlir_op!(DynamicSlice);
mlir_op_trait!(DynamicSlice, OneResult);
mlir_op_trait!(DynamicSlice, ZeroRegions);
mlir_op_trait!(DynamicSlice, ZeroSuccessors);

/// Constructs a new detached/owned [`DynamicSliceOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DynamicSliceOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn dynamic_slice<'v, 'i, 'c: 'v + 'i, 't: 'c, V: Value<'v, 'c, 't>, I: Value<'i, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    start_indices: &[I],
    slice_sizes: &[usize],
    location: L,
) -> DetachedDynamicSliceOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.dynamic_slice", location)
        .add_operand(input)
        .add_operands(start_indices)
        .add_attribute(
            DYNAMIC_SLICE_SLICE_SIZES_ATTRIBUTE,
            location
                .context()
                .dense_i64_array_attribute(slice_sizes.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::dynamic_slice`")
}

/// StableHLO [`Operation`] that produces a result tensor which is equal to [`DynamicUpdateSliceOperation::input`]
/// except that the slice starting at [`DynamicUpdateSliceOperation::start_indices`] is updated with the values in
/// [`DynamicUpdateSliceOperation::update`]. More formally, `result[result_index]` is defined as:
///
///   - `update[update_index]` if `0 <= update_index < shape(update)`, where
///     `adjusted_start_indices = clamp(0, start_indices, shape(operand) - shape(update))`, and
///     `update_index = result_index - adjusted_start_indices`, and
///   - `operand[result_index]`, otherwise.
///
/// # Example
///
/// The following is an example of a [`DynamicUpdateSliceOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [
/// //          [1, 1, 0, 0],
/// //          [1, 1, 0, 0],
/// //          [1, 1, 1, 1],
/// //          [1, 1, 1, 1]
/// //         ]
/// // %update: [
/// //           [1, 1],
/// //           [1, 1]
/// //          ]
/// // %start_indices_0: -1
/// // %start_indices_1: 3
/// %result = stablehlo.dynamic_update_slice %input, %update, %start_indices_0, %start_indices_1
///   : (tensor<4x4xi32>, tensor<2x2xi32>, tensor<i64>, tensor<i64>) -> tensor<4x4xi32>
/// // %result: [
/// //           [1, 1, 1, 1],
/// //           [1, 1, 1, 1],
/// //           [1, 1, 1, 1],
/// //           [1, 1, 1, 1]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#dynamic_update_slice)
/// for more information.
pub trait DynamicUpdateSliceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input of this [`DynamicUpdateSliceOperation`].
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the update value of this [`DynamicUpdateSliceOperation`].
    fn update(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns the start indices of this [`DynamicUpdateSliceOperation`].
    fn start_indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (2..self.operand_count()).flat_map(|index| self.operand(index)).collect()
    }
}

mlir_op!(DynamicUpdateSlice);
mlir_op_trait!(DynamicUpdateSlice, OneResult);
mlir_op_trait!(DynamicUpdateSlice, ZeroRegions);
mlir_op_trait!(DynamicUpdateSlice, ZeroSuccessors);

/// Constructs a new detached/owned [`DynamicUpdateSliceOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DynamicUpdateSliceOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn dynamic_update_slice<
    'v,
    'u,
    'i,
    'c: 'v + 'u + 'i,
    't: 'c,
    V: Value<'v, 'c, 't>,
    U: Value<'u, 'c, 't>,
    I: Value<'i, 'c, 't>,
    L: Location<'c, 't>,
>(
    operand: V,
    update: U,
    start_indices: &[I],
    location: L,
) -> DetachedDynamicUpdateSliceOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    let mut operands = vec![operand.as_ref(), update.as_ref()];
    operands.extend(start_indices.iter().map(|v| v.as_ref()));
    OperationBuilder::new("stablehlo.dynamic_update_slice", location)
        .add_operands(&operands)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::dynamic_update_slice`")
}

/// StableHLO [`Attribute`] that models the dimension information in [`GatherOperation`]s.
#[derive(Copy, Clone)]
pub struct GatherDimensionsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> GatherDimensionsAttributeRef<'c, 't> {
    mlir_attribute_field!(offset_dimensions, GatherDimensionNumbersGetOffsetDims, [usize], mlir_prefix = stablehlo);

    mlir_attribute_field!(
        collapsed_slice_dimensions,
        GatherDimensionNumbersGetCollapsedSliceDims,
        [usize],
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        operand_batching_dimensions,
        GatherDimensionNumbersGetOperandBatchingDims,
        [usize],
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        start_indices_batching_dimensions,
        GatherDimensionNumbersGetStartIndicesBatchingDims,
        [usize],
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(start_index_map, GatherDimensionNumbersGetStartIndexMap, [usize], mlir_prefix = stablehlo);

    mlir_attribute_field!(
        index_vector_dimension,
        GatherDimensionNumbersGetIndexVectorDim,
        usize,
        mlir_prefix = stablehlo,
    );
}

mlir_subtype_trait_impls!(
    GatherDimensionsAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = GatherDimensionNumbers,
    mlir_prefix = stablehlo,
);

impl<'t> Context<'t> {
    /// Creates a new StableHLO [`GatherDimensionsAttributeRef`] owned by this [`Context`].
    pub fn stable_hlo_gather_dimensions<'c>(
        &'c self,
        offset_dimensions: &[usize],
        collapsed_slice_dimensions: &[usize],
        operand_batching_dimensions: &[usize],
        start_indices_batching_dimensions: &[usize],
        start_index_map: &[usize],
        index_vector_dimension: usize,
    ) -> GatherDimensionsAttributeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        let offset_dimensions = offset_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let collapsed_slice_dimensions = collapsed_slice_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let operand_batching_dimensions = operand_batching_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let start_indices_batching_dimensions =
            start_indices_batching_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let start_index_map = start_index_map.iter().map(|v| *v as i64).collect::<Vec<_>>();
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            GatherDimensionsAttributeRef::from_c_api(
                stablehloGatherDimensionNumbersGet(
                    *self.handle.borrow(),
                    offset_dimensions.len().cast_signed(),
                    offset_dimensions.as_ptr(),
                    collapsed_slice_dimensions.len().cast_signed(),
                    collapsed_slice_dimensions.as_ptr(),
                    operand_batching_dimensions.len().cast_signed(),
                    operand_batching_dimensions.as_ptr(),
                    start_indices_batching_dimensions.len().cast_signed(),
                    start_indices_batching_dimensions.as_ptr(),
                    start_index_map.len().cast_signed(),
                    start_index_map.as_ptr(),
                    index_vector_dimension as i64,
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Name of the [`Attribute`] that is used to store [`GatherOperation::dimensions`].
pub const GATHER_DIMENSIONS_ATTRIBUTE: &str = "dimension_numbers";

/// Name of the [`Attribute`] that is used to store [`GatherOperation::slice_sizes`].
pub const GATHER_SLICE_SIZES_ATTRIBUTE: &str = "slice_sizes";

/// Name of the [`Attribute`] that is used to store [`GatherOperation::indices_are_sorted`].
pub const GATHER_INDICES_ARE_SORTED_ATTRIBUTE: &str = "indices_are_sorted";

/// StableHLO [`Operation`] that gathers slices from [`GatherOperation::input`] at indices specified by
/// [`GatherOperation::start_indices`]. The exact semantics are more involved and are also controllable by
/// [`GatherOperation::dimensions`], [`GatherOperation::slice_sizes`], and [`GatherOperation::indices_are_sorted`].
///
/// # Example
///
/// The following is an example of a [`GatherOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [
/// //          [
/// //           [[1, 2], [3, 4], [5, 6], [7, 8]],
/// //           [[9, 10],[11, 12], [13, 14], [15, 16]],
/// //           [[17, 18], [19, 20], [21, 22], [23, 24]]
/// //          ],
/// //          [
/// //           [[25, 26], [27, 28], [29, 30], [31, 32]],
/// //           [[33, 34], [35, 36], [37, 38], [39, 40]],
/// //           [[41, 42], [43, 44], [45, 46], [47, 48]]
/// //          ]
/// //         ]
/// // %start_indices: [
/// //                  [
/// //                   [[0, 0], [1, 0], [2, 1]],
/// //                   [[0, 1], [1, 1], [0, 9]]
/// //                  ],
/// //                  [
/// //                   [[0, 0], [2, 1], [2, 2]],
/// //                   [[1, 2], [0, 1], [1, 0]]
/// //                  ]
/// //                 ]
/// %result = "stablehlo.gather"(%input, %start_indices) <{
///   dimension_numbers = #stablehlo.gather<
///     offset_dims = [3, 4],
///     collapsed_slice_dims = [1],
///     operand_batching_dims = [0],
///     start_indices_batching_dims = [1],
///     start_index_map = [2, 1],
///     index_vector_dim = 3
///   >,
///   indices_are_sorted = false,
///   slice_sizes = array<i64: 1, 1, 2, 2>
/// }> : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi32>
/// // %result: [
/// //           [
/// //            [
/// //             [[1, 2], [3, 4]],
/// //             [[3, 4], [5, 6]],
/// //             [[13, 14], [15, 16]]
/// //            ],
/// //            [
/// //             [[33, 34], [35, 36]],
/// //             [[35, 36], [37, 38]],
/// //             [[41, 42], [43, 44]]
/// //            ]
/// //           ],
/// //           [
/// //            [
/// //             [[1, 2], [3, 4]],
/// //             [[13, 14], [15, 16]],
/// //             [[21, 22], [23, 24]]
/// //            ],
/// //            [
/// //             [[43, 44], [45, 46]],
/// //             [[33, 34], [35, 36]],
/// //             [[27, 28], [29, 30]]
/// //            ]
/// //           ]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#gather)
/// for more information.
pub trait GatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input of this [`GatherOperation`].
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the start indices of this [`GatherOperation`].
    fn start_indices(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns the dimensions configuration of this [`GatherOperation`].
    fn dimensions(&self) -> GatherDimensionsAttributeRef<'c, 't> {
        self.attribute(GATHER_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<GatherDimensionsAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{GATHER_DIMENSIONS_ATTRIBUTE}' attribute in `stable_hlo::gather`"))
    }

    /// Returns the slice sizes for this [`GatherOperation`].
    fn slice_sizes(&self) -> Vec<usize> {
        self.attribute(GATHER_SLICE_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{GATHER_SLICE_SIZES_ATTRIBUTE}' attribute in `stable_hlo::gather`"))
            .values()
            .map(|value| value as usize)
            .collect()
    }

    /// Returns whether the indices are sorted for this [`GatherOperation`].
    fn indices_are_sorted(&self) -> bool {
        self.attribute(GATHER_INDICES_ARE_SORTED_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or(false)
    }
}

mlir_op!(Gather);
mlir_op_trait!(Gather, OneResult);
mlir_op_trait!(Gather, ZeroRegions);
mlir_op_trait!(Gather, ZeroSuccessors);

/// Constructs a new detached/owned [`GatherOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`GatherOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn gather<'v, 'i, 'c: 'v + 'i, 't: 'c, V: Value<'v, 'c, 't>, I: Value<'i, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    start_indices: I,
    dimensions: GatherDimensionsAttributeRef<'c, 't>,
    slice_sizes: &[usize],
    indices_are_sorted: bool,
    location: L,
) -> DetachedGatherOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.gather", location)
        .add_operand(input)
        .add_operand(start_indices)
        .add_attribute(GATHER_DIMENSIONS_ATTRIBUTE, dimensions)
        .add_attribute(
            GATHER_SLICE_SIZES_ATTRIBUTE,
            context
                .dense_i64_array_attribute(slice_sizes.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        )
        .add_attribute(GATHER_INDICES_ARE_SORTED_ATTRIBUTE, context.boolean_attribute(indices_are_sorted))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::gather`")
}

/// StableHLO [`Operation`] that gathers slices, same as [`GatherOperation`], with the only difference being that in
/// this operation [`GatherOperation::slice_sizes`] is dynamically-computed and provided as an additional input/operand.
///
/// # Example
///
/// The following is an example of a [`DynamicGatherOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [
/// //          [[1, 2], [3, 4], [5, 6], [7, 8]],
/// //          [[9, 10],[11, 12], [13, 14], [15, 16]],
/// //          [[17, 18], [19, 20], [21, 22], [23, 24]]
/// //         ]
/// // %start_indices: [
/// //                  [[0, 0], [1, 0], [2, 1]],
/// //                  [[0, 1], [1, 1], [0, 2]]
/// //                 ]
/// // %slice_sizes: [1, 2, 2]
/// %result = "stablehlo.dynamic_gather"(%input, %start_indices, %slice_sizes) <{
///   dimension_numbers = #stablehlo.gather<
///     offset_dims = [2, 3],
///     collapsed_slice_dims = [0],
///     start_index_map = [1, 0],
///     index_vector_dim = 2
///   >,
///   indices_are_sorted = false
/// }> : (tensor<3x4x2xi64>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x2x2xi64>
/// // %result: [
/// //            [
/// //              [[1, 2], [3, 4]],
/// //              [[3, 4], [5, 6]],
/// //              [[13, 14], [15, 16]]
/// //            ],
/// //            [
/// //              [[9, 10], [11, 12]],
/// //              [[11, 12], [13, 14]],
/// //              [[17, 18], [19, 20]]
/// //            ]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#dynamic_gather)
/// for more information.
pub trait DynamicGatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input of this [`DynamicGatherOperation`].
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the start indices of this [`DynamicGatherOperation`].
    fn start_indices(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns the slice sizes for this [`DynamicGatherOperation`].
    fn slice_sizes(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(2).unwrap()
    }

    /// Returns the dimensions configuration of this [`DynamicGatherOperation`].
    fn dimensions(&self) -> GatherDimensionsAttributeRef<'c, 't> {
        self.attribute(GATHER_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<GatherDimensionsAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{GATHER_DIMENSIONS_ATTRIBUTE}' attribute in `stable_hlo::dynamic_gather`"))
    }

    /// Returns whether the indices are sorted for this [`DynamicGatherOperation`].
    fn indices_are_sorted(&self) -> bool {
        self.attribute(GATHER_INDICES_ARE_SORTED_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or(false)
    }
}

mlir_op!(DynamicGather);
mlir_op_trait!(DynamicGather, OneResult);
mlir_op_trait!(DynamicGather, ZeroRegions);
mlir_op_trait!(DynamicGather, ZeroSuccessors);

/// Constructs a new detached/owned [`DynamicGatherOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DynamicGatherOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn dynamic_gather<
    'v,
    'i,
    's,
    'c: 'v + 'i + 's,
    't: 'c,
    V: Value<'v, 'c, 't>,
    I: Value<'i, 'c, 't>,
    S: Value<'s, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V,
    start_indices: I,
    slice_sizes: S,
    dimensions: GatherDimensionsAttributeRef<'c, 't>,
    output_type: T,
    indices_are_sorted: bool,
    location: L,
) -> DetachedDynamicGatherOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.dynamic_gather", location)
        .add_operand(input)
        .add_operand(start_indices)
        .add_operand(slice_sizes)
        .add_attribute(GATHER_DIMENSIONS_ATTRIBUTE, dimensions)
        .add_attribute(GATHER_INDICES_ARE_SORTED_ATTRIBUTE, context.boolean_attribute(indices_are_sorted))
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::dynamic_gather`")
}

/// StableHLO [`Attribute`] that models the dimension information in [`ScatterOperation`]s.
#[derive(Copy, Clone)]
pub struct ScatterDimensionsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ScatterDimensionsAttributeRef<'c, 't> {
    mlir_attribute_field!(
        update_window_dimensions,
        ScatterDimensionNumbersGetUpdateWindowDims,
        [usize],
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        inserted_window_dimensions,
        ScatterDimensionNumbersGetInsertedWindowDims,
        [usize],
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        input_batching_dimensions,
        ScatterDimensionNumbersGetInputBatchingDims,
        [usize],
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        scatter_indices_batching_dimensions,
        ScatterDimensionNumbersGetScatterIndicesBatchingDims,
        [usize],
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        scattered_dimensions_to_operand_dimensions,
        ScatterDimensionNumbersGetScatteredDimsToOperandDims,
        [usize],
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(index_vector_dimension, DimensionNumbersGetIndexVectorDim, usize, mlir_prefix = stablehlo);
}

mlir_subtype_trait_impls!(
    ScatterDimensionsAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = ScatterDimensionNumbers,
    mlir_prefix = stablehlo,
);

impl<'t> Context<'t> {
    /// Creates a new StableHLO [`ScatterDimensionsAttributeRef`] owned by this [`Context`].
    pub fn stable_hlo_scatter_dimensions<'c>(
        &'c self,
        update_window_dimensions: &[usize],
        inserted_window_dimensions: &[usize],
        input_batching_dimensions: &[usize],
        scatter_indices_batching_dimensions: &[usize],
        scattered_dimensions_to_operand_dimensions: &[usize],
        index_vector_dimension: usize,
    ) -> ScatterDimensionsAttributeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        let update_window_dimensions = update_window_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let inserted_window_dimensions = inserted_window_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let input_batching_dimensions = input_batching_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let scatter_indices_batching_dimensions =
            scatter_indices_batching_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let scattered_dimensions_to_operand_dimensions =
            scattered_dimensions_to_operand_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            ScatterDimensionsAttributeRef::from_c_api(
                stablehloScatterDimensionNumbersGet(
                    *self.handle.borrow(),
                    update_window_dimensions.len().cast_signed(),
                    update_window_dimensions.as_ptr(),
                    inserted_window_dimensions.len().cast_signed(),
                    inserted_window_dimensions.as_ptr(),
                    input_batching_dimensions.len().cast_signed(),
                    input_batching_dimensions.as_ptr(),
                    scatter_indices_batching_dimensions.len().cast_signed(),
                    scatter_indices_batching_dimensions.as_ptr(),
                    scattered_dimensions_to_operand_dimensions.len().cast_signed(),
                    scattered_dimensions_to_operand_dimensions.as_ptr(),
                    index_vector_dimension as i64,
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Name of the [`Attribute`] that is used to store [`ScatterOperation::dimensions`].
pub const SCATTER_DIMENSIONS_ATTRIBUTE: &str = "scatter_dimension_numbers";

/// Name of the [`Attribute`] that is used to store [`ScatterOperation::indices_are_sorted`].
pub const SCATTER_INDICES_ARE_SORTED_ATTRIBUTE: &str = "indices_are_sorted";

/// Name of the [`Attribute`] that is used to store [`ScatterOperation::unique_indices`].
pub const SCATTER_UNIQUE_INDICES_ATTRIBUTE: &str = "unique_indices";

/// StableHLO [`Operation`] that produces result tensors which are equal to its input tensors except that several
/// slices specified by [`ScatterOperation::scatter_indices`] are updated with [`ScatterOperation::updates`] using
/// [`ScatterOperation::computation`] to potentially combine values.
///
/// # Example
///
/// The following is an example of a [`ScatterOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [
/// //          [
/// //           [[1, 2], [3, 4], [5, 6], [7, 8]],
/// //           [[9, 10],[11, 12], [13, 14], [15, 16]],
/// //           [[17, 18], [19, 20], [21, 22], [23, 24]]
/// //          ],
/// //          [
/// //           [[25, 26], [27, 28], [29, 30], [31, 32]],
/// //           [[33, 34], [35, 36], [37, 38], [39, 40]],
/// //           [[41, 42], [43, 44], [45, 46], [47, 48]]
/// //          ]
/// //         ]
/// // %scatter_indices: [
/// //                    [
/// //                     [[0, 0], [1, 0], [2, 1]],
/// //                     [[0, 1], [1, 1], [0, 9]]
/// //                    ],
/// //                    [
/// //                     [[0, 0], [2, 1], [2, 2]],
/// //                     [[1, 2], [0, 1], [1, 0]]
/// //                    ]
/// //                   ]
/// // %update: [
/// //           [
/// //            [[1, 1], [1, 1], [1, 1]],
/// //            [[1, 1], [1, 1], [1, 1]]
/// //           ],
/// //           [
/// //            [[1, 1], [1, 1], [1, 1]],
/// //            [[1, 1], [1, 1], [1, 1]]
/// //           ]
/// //          ]
/// %result = "stablehlo.scatter"(%input, %scatter_indices, %update) <{
///   indices_are_sorted = false,
///   scatter_dimension_numbers = #stablehlo.scatter<
///     update_window_dims = [3, 4],
///     inserted_window_dims = [1],
///     input_batching_dims = [0],
///     scatter_indices_batching_dims = [1],
///     scatter_dims_to_operand_dims = [2, 1],
///     index_vector_dim = 3
///   >,
///   unique_indices = false
/// }> ({
///   ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
///     %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
///     stablehlo.return %0 : tensor<i64>
/// }) : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
/// // %result: [
/// //           [
/// //            [[3, 4], [6, 7], [6, 7], [7, 8]],
/// //            [[9, 10],[11, 12], [15, 16], [17, 18]],
/// //            [[17, 18], [19, 20], [22, 23], [24, 25]]
/// //           ],
/// //           [
/// //            [[25, 26], [28, 29], [30, 31], [31, 32]],
/// //            [[35, 36], [38, 39], [38, 39], [39, 40]],
/// //            [[41, 42], [44, 45], [46, 47], [47, 48]]
/// //           ]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#scatter)
/// for more information.
pub trait ScatterOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> {
    /// Returns the inputs of this [`ScatterOperation`].
    fn inputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let input_count = (self.operand_count() - 1) / 2;
        (0..input_count).flat_map(|index| self.operand(index)).collect()
    }

    /// Returns the scatter indices of this [`ScatterOperation`].
    fn scatter_indices(&self) -> ValueRef<'o, 'c, 't> {
        let input_count = (self.operand_count() - 1) / 2;
        self.operand(input_count).unwrap()
    }

    /// Returns the updates of this [`ScatterOperation`].
    fn updates(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let input_count = (self.operand_count() - 1) / 2;
        (input_count + 1..self.operand_count()).flat_map(|index| self.operand(index)).collect()
    }

    /// Returns the dimensions configuration of this [`ScatterOperation`].
    fn dimensions(&self) -> ScatterDimensionsAttributeRef<'c, 't> {
        self.attribute(SCATTER_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ScatterDimensionsAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{SCATTER_DIMENSIONS_ATTRIBUTE}' attribute in `stable_hlo::scatter`"))
    }

    /// Returns whether the indices are assumed to be sorted for this [`ScatterOperation`].
    fn indices_are_sorted(&self) -> bool {
        self.attribute(SCATTER_INDICES_ARE_SORTED_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or(false)
    }

    /// Returns whether the indices are assumed to be unique for this [`ScatterOperation`].
    fn unique_indices(&self) -> bool {
        self.attribute(SCATTER_UNIQUE_INDICES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or(false)
    }

    /// Returns a reference to the [`Region`](crate::Region) that contains the update computation
    /// used by this [`ScatterOperation`].
    fn computation(&self) -> RegionRef<'o, 'c, 't> {
        self.body_region()
    }
}

mlir_op!(Scatter);
mlir_op_trait!(Scatter, OneRegion);
mlir_op_trait!(Scatter, SingleBlock);
mlir_op_trait!(Scatter, SingleBlockRegions);
mlir_op_trait!(Scatter, ZeroSuccessors);

/// Constructs a new detached/owned [`ScatterOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ScatterOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
#[allow(clippy::too_many_arguments)]
pub fn scatter<
    'input,
    'indices,
    'update,
    'c: 'input + 'indices + 'update,
    't: 'c,
    Input: Value<'input, 'c, 't>,
    Indices: Value<'indices, 'c, 't>,
    Update: Value<'update, 'c, 't>,
    L: Location<'c, 't>,
>(
    inputs: &[Input],
    scatter_indices: Indices,
    updates: &[Update],
    dimensions: ScatterDimensionsAttributeRef<'c, 't>,
    computation: DetachedRegion<'c, 't>,
    indices_are_sorted: bool,
    unique_indices: bool,
    location: L,
) -> DetachedScatterOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.scatter", location)
        .add_operands(inputs)
        .add_operand(scatter_indices)
        .add_operands(updates)
        .add_attribute(SCATTER_DIMENSIONS_ATTRIBUTE, dimensions)
        .add_attribute(SCATTER_INDICES_ARE_SORTED_ATTRIBUTE, context.boolean_attribute(indices_are_sorted))
        .add_attribute(SCATTER_UNIQUE_INDICES_ATTRIBUTE, context.boolean_attribute(unique_indices))
        .add_region(computation)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::scatter`")
}

/// Name of the [`Attribute`] that is used to store [`SelectAndScatterOperation::window_dimensions`].
pub const SELECT_AND_SCATTER_WINDOW_DIMENSIONS_ATTRIBUTE: &str = "window_dimensions";

/// Name of the [`Attribute`] that is used to store [`SelectAndScatterOperation::window_strides`].
pub const SELECT_AND_SCATTER_WINDOW_STRIDES_ATTRIBUTE: &str = "window_strides";

/// StableHLO [`Operation`] that scatters values from [`SelectAndScatterOperation::source`]
/// using [`SelectAndScatterOperation::scatter`] based on the outcome of
/// [`reduce_window`](crate::dialects::stable_hlo::reduce_window) on [`SelectAndScatterOperation::input`] using
/// [`SelectAndScatterOperation::select`] to produce an output tensor.
///
/// # Example
///
/// The following is an example of a [`SelectAndScatterOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input: [[1, 5], [2, 5], [3, 6], [4, 4]]
/// // %source: [[5, 6], [7, 8]]
/// // %initial_value: 0
/// %result = "stablehlo.select_and_scatter"(%input, %source, %initial_value) <{
///   padding = dense<[[0, 1], [0, 0]]> : tensor<2x2xi64>,
///   window_dimensions = array<i64: 3, 1>,
///   window_strides = array<i64: 2, 1>
/// }> ({
///   ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
///     %0 = stablehlo.compare  GE, %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
///     stablehlo.return %0 : tensor<i1>
/// }, {
///   ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
///     %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
///     stablehlo.return %0 : tensor<i64>
/// }) : (tensor<4x2xi64>, tensor<2x2xi64>, tensor<i64>) -> tensor<4x2xi64>
/// // %result: [[0, 0], [0, 0], [5, 14], [7, 0]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#select_and_scatter)
/// for more information.
pub trait SelectAndScatterOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input of this [`SelectAndScatterOperation`].
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the source of this [`SelectAndScatterOperation`].
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns the initial value of this [`SelectAndScatterOperation`].
    fn initial_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(2).unwrap()
    }

    /// Returns the window dimensions for this [`SelectAndScatterOperation`], if specified.
    fn window_dimensions(&self) -> Option<Vec<usize>> {
        self.attribute(SELECT_AND_SCATTER_WINDOW_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
    }

    /// Returns the window strides for this [`SelectAndScatterOperation`], if specified.
    fn window_strides(&self) -> Option<Vec<usize>> {
        self.attribute(SELECT_AND_SCATTER_WINDOW_STRIDES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
    }

    /// Returns a reference to the [`Region`](crate::Region) that contains the select computation used
    /// by this [`SelectAndScatterOperation`].
    fn select(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns a reference to the [`Region`](crate::Region) that contains the scatter computation used
    /// by this [`SelectAndScatterOperation`].
    fn scatter(&self) -> RegionRef<'o, 'c, 't> {
        self.region(1).unwrap()
    }
}

mlir_op!(SelectAndScatter);
mlir_op_trait!(SelectAndScatter, OneResult);
mlir_op_trait!(SelectAndScatter, ZeroSuccessors);
mlir_op_trait!(SelectAndScatter, @local HasPadding);

/// Constructs a new detached/owned [`SelectAndScatterOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`SelectAndScatterOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
#[allow(clippy::too_many_arguments)]
pub fn select_and_scatter<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    source: V,
    initial_value: V,
    window_dimensions: Option<&[usize]>,
    window_strides: Option<&[usize]>,
    padding: Option<&[(usize, usize)]>,
    select: DetachedRegion<'c, 't>,
    scatter: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedSelectAndScatterOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let mut builder = OperationBuilder::new("stablehlo.select_and_scatter", location)
        .add_operand(input)
        .add_operand(source)
        .add_operand(initial_value);
    if let Some(window_dimensions) = window_dimensions {
        builder = builder.add_attribute(
            SELECT_AND_SCATTER_WINDOW_DIMENSIONS_ATTRIBUTE,
            context
                .dense_i64_array_attribute(window_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        );
    }
    if let Some(window_strides) = window_strides {
        builder = builder.add_attribute(
            SELECT_AND_SCATTER_WINDOW_STRIDES_ATTRIBUTE,
            context
                .dense_i64_array_attribute(window_strides.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        );
    }
    if let Some(padding) = padding {
        builder = builder.add_attribute(PADDING_ATTRIBUTE, context.stable_hlo_padding(padding, location));
    }
    builder
        .add_region(select)
        .add_region(scatter)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::select_and_scatter`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};
    use crate::dialects::{func, stable_hlo};
    use crate::{Attribute, Block, Context, Operation, Region, Size, Value};

    use super::{
        BroadcastOperation, ConcatenateOperation, DynamicBroadcastOperation, DynamicGatherOperation,
        DynamicPadOperation, DynamicSliceOperation, DynamicUpdateSliceOperation, GatherOperation,
        GetDimensionSizeOperation, HasPadding, PadOperation, ReshapeOperation, ScatterOperation,
        SelectAndScatterOperation, SliceOperation, TransposeOperation, broadcast, concatenate, dynamic_broadcast,
        dynamic_gather, dynamic_pad, dynamic_reshape, dynamic_slice, dynamic_update_slice, gather, get_dimension_size,
        pad, reshape, scatter, select_and_scatter, slice, transpose,
    };

    #[test]
    fn test_get_dimension_size() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let i32_type = context.signless_integer_type(32);
        let input_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(3)], None, location).unwrap();
        let output_type = context.tensor_type(i32_type, &[], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location)]);
            let input_value = block.argument(0).unwrap();
            let op = get_dimension_size(input_value, 1, location);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), output_type);
            assert_eq!(op.dimension(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "get_dimension_size_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @get_dimension_size_test(%arg0: tensor<2x3xi64>) -> tensor<i32> {
                    %0 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<2x3xi64>) -> tensor<i32>
                    return %0 : tensor<i32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_transpose() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let input_type = context
            .tensor_type(i32_type, &[Size::Static(3), Size::Static(4), Size::Static(5)], None, location)
            .unwrap();
        let output_type = context
            .tensor_type(i32_type, &[Size::Static(5), Size::Static(3), Size::Static(4)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location)]);
            let input = block.argument(0).unwrap();
            let op = transpose(input, &[2, 0, 1], location);
            assert_eq!(op.permutation(), vec![2, 0, 1]);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), output_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "transpose_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @transpose_test(%arg0: tensor<3x4x5xi32>) -> tensor<5x3x4xi32> {
                    %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<3x4x5xi32>) -> tensor<5x3x4xi32>
                    return %0 : tensor<5x3x4xi32>
                  }
                }
            "}
        );
    }

    #[test]
    fn test_reshape() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let input_type = context.tensor_type(i32_type, &[Size::Static(1), Size::Static(6)], None, location).unwrap();
        let output_type = context.tensor_type(i32_type, &[Size::Static(2), Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location)]);
            let input = block.argument(0).unwrap();
            let op = reshape(input, &[2, 3], location);
            assert_eq!(op.shape(), vec![Size::Static(2), Size::Static(3)]);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), output_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "reshape_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @reshape_test(%arg0: tensor<1x6xi32>) -> tensor<2x3xi32> {
                    %0 = stablehlo.reshape %arg0 : (tensor<1x6xi32>) -> tensor<2x3xi32>
                    return %0 : tensor<2x3xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_dynamic_reshape() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let input_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(3)], None, location).unwrap();
        let shape_type = context.tensor_type(i64_type, &[Size::Static(2)], None, location).unwrap();
        let output_type = context.tensor_type(i64_type, &[Size::Dynamic, Size::Dynamic], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (shape_type, location)]);
            let input = block.argument(0).unwrap();
            let output_shape = block.argument(1).unwrap();
            let op = dynamic_reshape(input, output_shape, location);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), output_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "dynamic_reshape_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), shape_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @dynamic_reshape_test(%arg0: tensor<2x3xi64>, %arg1: tensor<2xi64>) -> tensor<?x?xi64> {
                    %0 = stablehlo.dynamic_reshape %arg0, %arg1 : (tensor<2x3xi64>, tensor<2xi64>) -> tensor<?x?xi64>
                    return %0 : tensor<?x?xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_broadcast() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let input_type = context.tensor_type(i32_type, &[Size::Static(1), Size::Static(3)], None, location).unwrap();
        let output_type = context
            .tensor_type(i32_type, &[Size::Static(2), Size::Static(3), Size::Static(2)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location)]);
            let input = block.argument(0).unwrap();
            let op = broadcast(input, output_type, &[2, 1], location);
            assert_eq!(op.dimensions(), vec![2, 1]);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), output_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "broadcast_in_dim_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @broadcast_in_dim_test(%arg0: tensor<1x3xi32>) -> tensor<2x3x2xi32> {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2, 1] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
                    return %0 : tensor<2x3x2xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_dynamic_broadcast() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let input_type = context.tensor_type(i32_type, &[Size::Static(1), Size::Static(3)], None, location).unwrap();
        let shape_type = context.tensor_type(i64_type, &[Size::Static(3)], None, location).unwrap();
        let output_type = context
            .tensor_type(i32_type, &[Size::Dynamic, Size::Dynamic, Size::Dynamic], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (shape_type, location)]);
            let input_value = block.argument(0).unwrap();
            let shape_value = block.argument(1).unwrap();
            let op = dynamic_broadcast(input_value, shape_value, &[2, 1], Some(&[1]), Some(&[0]), location);
            assert_eq!(op.dimensions(), vec![2, 1]);
            assert_eq!(op.known_expanding_dimensions(), Some(vec![1]));
            assert_eq!(op.known_non_expanding_dimensions(), Some(vec![0]));
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "dynamic_broadcast_in_dim_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), shape_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @dynamic_broadcast_in_dim_test(\
                    %arg0: tensor<1x3xi32>, \
                    %arg1: tensor<3xi64>\
                  ) -> tensor<?x?x?xi32> {
                    %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [2, 1] {\
                      known_expanding_dimensions = array<i64: 1>, \
                      known_nonexpanding_dimensions = array<i64: 0>\
                    } : (tensor<1x3xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
                    return %0 : tensor<?x?x?xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_pad() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let input_type = context.tensor_type(i32_type, &[Size::Static(2), Size::Static(3)], None, location).unwrap();
        let padding_value_type = context.tensor_type(i32_type, &[], None, location).unwrap();
        let output_type = context.tensor_type(i32_type, &[Size::Static(5), Size::Static(5)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (padding_value_type, location)]);
            let input = block.argument(0).unwrap();
            let padding_value = block.argument(1).unwrap();
            let op = pad(input, padding_value, &[0, 1], &[2, 1], &[1, 0], location);
            assert_eq!(op.input(), input);
            assert_eq!(op.padding_value(), padding_value);
            assert_eq!(op.edge_padding_low(), vec![0, 1]);
            assert_eq!(op.edge_padding_high(), vec![2, 1]);
            assert_eq!(op.interior_padding(), vec![1, 0]);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "pad_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), padding_value_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @pad_test(%arg0: tensor<2x3xi32>, %arg1: tensor<i32>) -> tensor<5x5xi32> {
                    %0 = stablehlo.pad \
                      %arg0, \
                      %arg1, \
                      low = [0, 1], \
                      high = [2, 1], \
                      interior = [1, 0] \
                    : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x5xi32>
                    return %0 : tensor<5x5xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_dynamic_pad() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let input_type = context.tensor_type(i32_type, &[Size::Static(2), Size::Static(3)], None, location).unwrap();
        let padding_value_type = context.tensor_type(i32_type, &[], None, location).unwrap();
        let padding_specification_type = context.tensor_type(i64_type, &[Size::Static(2)], None, location).unwrap();
        let output_type = context.tensor_type(i32_type, &[Size::Static(4), Size::Static(8)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (input_type, location),
                (padding_value_type, location),
                (padding_specification_type, location),
                (padding_specification_type, location),
                (padding_specification_type, location),
            ]);
            let input = block.argument(0).unwrap();
            let padding_value = block.argument(1).unwrap();
            let edge_padding_low = block.argument(2).unwrap();
            let edge_padding_high = block.argument(3).unwrap();
            let interior_padding = block.argument(4).unwrap();
            let op = dynamic_pad(
                input,
                padding_value,
                edge_padding_low,
                edge_padding_high,
                interior_padding,
                output_type,
                location,
            );
            assert_eq!(op.input(), input);
            assert_eq!(op.padding_value(), padding_value);
            assert_eq!(op.edge_padding_low(), edge_padding_low);
            assert_eq!(op.edge_padding_high(), edge_padding_high);
            assert_eq!(op.interior_padding(), interior_padding);
            assert_eq!(op.operands().count(), 5);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "dynamic_pad_test",
                func::FuncAttributes {
                    arguments: vec![
                        input_type.into(),
                        padding_value_type.into(),
                        padding_specification_type.into(),
                        padding_specification_type.into(),
                        padding_specification_type.into(),
                    ],
                    results: vec![output_type.into()],
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
                  func.func @dynamic_pad_test(\
                    %arg0: tensor<2x3xi32>, \
                    %arg1: tensor<i32>, \
                    %arg2: tensor<2xi64>, \
                    %arg3: tensor<2xi64>, \
                    %arg4: tensor<2xi64>\
                  ) -> tensor<4x8xi32> {
                    %0 = stablehlo.dynamic_pad %arg0, %arg1, %arg2, %arg3, %arg4 \
                      : (tensor<2x3xi32>, tensor<i32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<4x8xi32>
                    return %0 : tensor<4x8xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_concatenate() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let input_0_type = context.tensor_type(i64_type, &[Size::Static(3), Size::Static(2)], None, location).unwrap();
        let input_1_type = context.tensor_type(i64_type, &[Size::Static(1), Size::Static(2)], None, location).unwrap();
        let output_type = context.tensor_type(i64_type, &[Size::Static(4), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_0_type, location), (input_1_type, location)]);
            let input0_value = block.argument(0).unwrap();
            let input1_value = block.argument(1).unwrap();
            let op = concatenate(&[input0_value, input1_value], 0, location);
            assert_eq!(op.dimension(), 0);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), output_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "concatenate_test",
                func::FuncAttributes {
                    arguments: vec![input_0_type.into(), input_1_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @concatenate_test(%arg0: tensor<3x2xi64>, %arg1: tensor<1x2xi64>) -> tensor<4x2xi64> {
                    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 \
                      : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
                    return %0 : tensor<4x2xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_slice() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let input_type = context
            .tensor_type(i32_type, &[Size::Static(3), Size::Static(4), Size::Static(5)], None, location)
            .unwrap();
        let output_type = context
            .tensor_type(i32_type, &[Size::Static(2), Size::Static(2), Size::Static(3)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location)]);
            let input = block.argument(0).unwrap();
            let op = slice(input, &[0, 1, 0], &[2, 3, 5], &[1, 1, 2], location);
            assert_eq!(op.start_indices(), vec![0, 1, 0]);
            assert_eq!(op.limit_indices(), vec![2, 3, 5]);
            assert_eq!(op.strides(), vec![1, 1, 2]);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), output_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "slice_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @slice_test(%arg0: tensor<3x4x5xi32>) -> tensor<2x2x3xi32> {
                    %0 = stablehlo.slice %arg0 [0:2, 1:3, 0:5:2] : (tensor<3x4x5xi32>) -> tensor<2x2x3xi32>
                    return %0 : tensor<2x2x3xi32>
                  }
                }
            "}
        );
    }

    #[test]
    fn test_dynamic_slice() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let input_type = context
            .tensor_type(i32_type, &[Size::Static(3), Size::Static(4), Size::Static(5)], None, location)
            .unwrap();
        let index_type = context.tensor_type(i64_type, &[], None, location).unwrap();
        let output_type = context
            .tensor_type(i32_type, &[Size::Static(1), Size::Static(2), Size::Static(3)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (input_type, location),
                (index_type, location),
                (index_type, location),
                (index_type, location),
            ]);
            let input = block.argument(0).unwrap();
            let start_index_0 = block.argument(1).unwrap();
            let start_index_1 = block.argument(2).unwrap();
            let start_index_2 = block.argument(3).unwrap();
            let op = dynamic_slice(input, &[start_index_0, start_index_1, start_index_2], &[1, 2, 3], location);
            assert_eq!(op.input(), input);
            assert_eq!(op.start_indices(), vec![start_index_0, start_index_1, start_index_2]);
            assert_eq!(op.slice_sizes(), vec![1, 2, 3]);
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), output_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "dynamic_slice_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), index_type.into(), index_type.into(), index_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @dynamic_slice_test(\
                    %arg0: tensor<3x4x5xi32>, \
                    %arg1: tensor<i64>, \
                    %arg2: tensor<i64>, \
                    %arg3: tensor<i64>\
                  ) -> tensor<1x2x3xi32> {
                    %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [1, 2, 3] \
                      : (tensor<3x4x5xi32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x2x3xi32>
                    return %0 : tensor<1x2x3xi32>
                  }
                }
            "}
        );
    }

    #[test]
    fn test_dynamic_update_slice() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let input_type = context
            .tensor_type(i32_type, &[Size::Static(3), Size::Static(4), Size::Static(5)], None, location)
            .unwrap();
        let update_type = context
            .tensor_type(i32_type, &[Size::Static(1), Size::Static(2), Size::Static(3)], None, location)
            .unwrap();
        let index_type = context.tensor_type(i64_type, &[], None, location).unwrap();
        let output_type = context
            .tensor_type(i32_type, &[Size::Static(3), Size::Static(4), Size::Static(5)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (input_type, location),
                (update_type, location),
                (index_type, location),
                (index_type, location),
                (index_type, location),
            ]);
            let input = block.argument(0).unwrap();
            let update = block.argument(1).unwrap();
            let start_index_0 = block.argument(2).unwrap();
            let start_index_1 = block.argument(3).unwrap();
            let start_index_2 = block.argument(4).unwrap();
            let op = dynamic_update_slice(input, update, &[start_index_0, start_index_1, start_index_2], location);
            assert_eq!(op.input(), input);
            assert_eq!(op.update(), update);
            assert_eq!(op.start_indices(), vec![start_index_0, start_index_1, start_index_2]);
            assert_eq!(op.operands().count(), 5);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), output_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "dynamic_update_slice_test",
                func::FuncAttributes {
                    arguments: vec![
                        input_type.into(),
                        update_type.into(),
                        index_type.into(),
                        index_type.into(),
                        index_type.into(),
                    ],
                    results: vec![output_type.into()],
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
                  func.func @dynamic_update_slice_test(\
                    %arg0: tensor<3x4x5xi32>, \
                    %arg1: tensor<1x2x3xi32>, \
                    %arg2: tensor<i64>, \
                    %arg3: tensor<i64>, \
                    %arg4: tensor<i64>\
                  ) -> tensor<3x4x5xi32> {
                    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4 \
                      : (tensor<3x4x5xi32>, tensor<1x2x3xi32>, tensor<i64>, tensor<i64>, tensor<i64>) \
                      -> tensor<3x4x5xi32>
                    return %0 : tensor<3x4x5xi32>
                  }
                }
            "}
        );
    }

    #[test]
    fn test_gather_dimensions_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_gather_dimensions(&[0, 1], &[2, 3], &[4], &[5], &[6, 7], 8);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.offset_dimensions(), vec![0, 1]);
        assert_eq!(attribute.collapsed_slice_dimensions(), vec![2, 3]);
        assert_eq!(attribute.operand_batching_dimensions(), vec![4]);
        assert_eq!(attribute.start_indices_batching_dimensions(), vec![5]);
        assert_eq!(attribute.start_index_map(), vec![6, 7]);
        assert_eq!(attribute.index_vector_dimension(), 8);
    }

    #[test]
    fn test_gather_dimensions_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_gather_dimensions(&[0, 1], &[2, 3], &[4], &[5], &[6, 7], 8);
        let attribute_2 = context.stable_hlo_gather_dimensions(&[0, 1], &[2, 3], &[4], &[5], &[6, 7], 8);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_scatter_dimensions(&[], &[0, 1], &[4], &[], &[6, 7], 1);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_gather_dimensions(&[0, 1], &[2, 3], &[4], &[5], &[6, 7], 8);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_gather_dimensions_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_gather_dimensions(&[0, 1], &[2, 3], &[4], &[5], &[6, 7], 8);
        test_attribute_display_and_debug(
            attribute,
            "#stablehlo.gather<\
              offset_dims = [0, 1], \
              collapsed_slice_dims = [2, 3], \
              operand_batching_dims = [4], \
              start_indices_batching_dims = [5], \
              start_index_map = [6, 7], \
              index_vector_dim = 8\
            >",
        );
    }

    #[test]
    fn test_gather_dimensions_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_gather_dimensions(&[0, 1], &[2, 3], &[4], &[5], &[6, 7], 8);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_gather() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let input_type = context
            .tensor_type(i64_type, &[Size::Static(3), Size::Static(4), Size::Static(2)], None, location)
            .unwrap();
        let indices_type = context
            .tensor_type(i64_type, &[Size::Static(2), Size::Static(3), Size::Static(2)], None, location)
            .unwrap();
        let output_type = context
            .tensor_type(
                i64_type,
                &[Size::Static(2), Size::Static(3), Size::Static(2), Size::Static(2)],
                None,
                location,
            )
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (indices_type, location)]);
            let operand = block.argument(0).unwrap();
            let start_indices = block.argument(1).unwrap();
            let dimension_numbers = context.stable_hlo_gather_dimensions(&[2, 3], &[0], &[], &[], &[1, 0], 2);
            let op = gather(operand, start_indices, dimension_numbers, &[1, 2, 2], false, location);
            assert_eq!(op.dimensions(), dimension_numbers);
            assert_eq!(op.slice_sizes(), vec![1, 2, 2]);
            assert_eq!(op.indices_are_sorted(), false);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "gather_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), indices_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @gather_test(\
                    %arg0: tensor<3x4x2xi64>, \
                    %arg1: tensor<2x3x2xi64>\
                  ) -> tensor<2x3x2x2xi64> {
                    %0 = \"stablehlo.gather\"(%arg0, %arg1) <{\
                      dimension_numbers = #stablehlo.gather<\
                        offset_dims = [2, 3], \
                        collapsed_slice_dims = [0], \
                        start_index_map = [1, 0], \
                        index_vector_dim = 2\
                      >, \
                      indices_are_sorted = false, \
                      slice_sizes = array<i64: 1, 2, 2>\
                    }> : (tensor<3x4x2xi64>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi64>
                    return %0 : tensor<2x3x2x2xi64>
                  }
                }
            "}
        );
    }

    #[test]
    fn test_dynamic_gather() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let operand_tensor_type = context
            .tensor_type(i64_type, &[Size::Static(3), Size::Static(4), Size::Static(2)], None, location)
            .unwrap();
        let indices_tensor_type = context
            .tensor_type(i64_type, &[Size::Static(2), Size::Static(3), Size::Static(2)], None, location)
            .unwrap();
        let slice_sizes_tensor_type = context.tensor_type(i64_type, &[Size::Static(3)], None, location).unwrap();
        let output_tensor_type = context
            .tensor_type(
                i64_type,
                &[Size::Static(2), Size::Static(3), Size::Static(2), Size::Static(2)],
                None,
                location,
            )
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (operand_tensor_type, location),
                (indices_tensor_type, location),
                (slice_sizes_tensor_type, location),
            ]);
            let input = block.argument(0).unwrap();
            let start_indices = block.argument(1).unwrap();
            let slice_sizes = block.argument(2).unwrap();
            let dimensions = context.stable_hlo_gather_dimensions(&[2, 3], &[0], &[], &[], &[1, 0], 2);
            let op = dynamic_gather(input, start_indices, slice_sizes, dimensions, output_tensor_type, true, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.start_indices(), start_indices);
            assert_eq!(op.slice_sizes(), slice_sizes);
            assert_eq!(op.dimensions(), dimensions);
            assert_eq!(op.indices_are_sorted(), true);
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op_block = block.append_operation(op);
            block.append_operation(func::r#return(&[op_block.result(0).unwrap()], location));
            func::func(
                "dynamic_gather_test",
                func::FuncAttributes {
                    arguments: vec![
                        operand_tensor_type.into(),
                        indices_tensor_type.into(),
                        slice_sizes_tensor_type.into(),
                    ],
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
                  func.func @dynamic_gather_test(\
                    %arg0: tensor<3x4x2xi64>, \
                    %arg1: tensor<2x3x2xi64>, \
                    %arg2: tensor<3xi64>\
                  ) -> tensor<2x3x2x2xi64> {
                    %0 = \"stablehlo.dynamic_gather\"(%arg0, %arg1, %arg2) <{\
                      dimension_numbers = #stablehlo.gather<\
                        offset_dims = [2, 3], \
                        collapsed_slice_dims = [0], \
                        start_index_map = [1, 0], \
                        index_vector_dim = 2\
                      >, \
                      indices_are_sorted = true\
                    }> : (tensor<3x4x2xi64>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x2x2xi64>
                    return %0 : tensor<2x3x2x2xi64>
                  }
                }
            "}
        );
    }

    #[test]
    fn test_scatter_dimensions_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_scatter_dimensions(&[0, 1], &[2], &[3], &[4], &[5, 6], 7);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.dialect().namespace().unwrap(), "stablehlo");
        assert_eq!(attribute.update_window_dimensions(), vec![0, 1]);
        assert_eq!(attribute.inserted_window_dimensions(), vec![2]);
        assert_eq!(attribute.input_batching_dimensions(), vec![3]);
        assert_eq!(attribute.scatter_indices_batching_dimensions(), vec![4]);
        assert_eq!(attribute.scattered_dimensions_to_operand_dimensions(), vec![5, 6]);
        assert_eq!(attribute.index_vector_dimension(), 7);
    }

    #[test]
    fn test_scatter_dimensions_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_scatter_dimensions(&[0, 1], &[2], &[3], &[4], &[5, 6], 7);
        let attribute_2 = context.stable_hlo_scatter_dimensions(&[0, 1], &[2], &[3], &[4], &[5, 6], 7);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_scatter_dimensions(&[], &[1], &[0], &[4], &[], 1);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_scatter_dimensions(&[0, 1], &[2], &[3], &[4], &[5, 6], 7);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_scatter_dimensions_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_scatter_dimensions(&[0, 1], &[2], &[3], &[4], &[5, 6], 7);
        test_attribute_display_and_debug(
            attribute,
            "#stablehlo.scatter<\
              update_window_dims = [0, 1], \
              inserted_window_dims = [2], \
              input_batching_dims = [3], \
              scatter_indices_batching_dims = [4], \
              scatter_dims_to_operand_dims = [5, 6], \
              index_vector_dim = 7\
            >",
        );
    }

    #[test]
    fn test_scatter_dimensions_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_scatter_dimensions(&[0, 1], &[2], &[3], &[4], &[5, 6], 7);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_scatter() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let i32_scalar_type = context.tensor_type(i32_type, &[], None, location).unwrap();
        let input_type = context
            .tensor_type(i32_type, &[Size::Static(3), Size::Static(4), Size::Static(2)], None, location)
            .unwrap();
        let indices_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let update_type = context
            .tensor_type(i32_type, &[Size::Static(2), Size::Static(2), Size::Static(2)], None, location)
            .unwrap();
        let output_type = context
            .tensor_type(i32_type, &[Size::Static(3), Size::Static(4), Size::Static(2)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (indices_type, location), (update_type, location)]);
            let mut region = context.region();
            let mut inner_block = context.block(&[(i32_scalar_type, location), (i32_scalar_type, location)]);
            let current = inner_block.argument(0).unwrap();
            let update = inner_block.argument(1).unwrap();
            let add_op = stable_hlo::add(current, update, location);
            let add_op = inner_block.append_operation(add_op);
            let return_op = stable_hlo::r#return(&[add_op.result(0).unwrap()], location);
            inner_block.append_operation(return_op);
            region.append_block(inner_block);
            let dimensions = context.stable_hlo_scatter_dimensions(&[1, 2], &[0], &[], &[], &[0, 1], 1);
            let op = scatter(
                &[block.argument(0).unwrap()],
                block.argument(1).unwrap(),
                &[block.argument(2).unwrap()],
                dimensions,
                region.into(),
                false,
                false,
                location,
            );
            assert_eq!(op.inputs(), vec![block.argument(0).unwrap()]);
            assert_eq!(op.scatter_indices(), block.argument(1).unwrap());
            assert_eq!(op.updates(), vec![block.argument(2).unwrap()]);
            assert_eq!(op.dimensions(), dimensions);
            assert_eq!(op.indices_are_sorted(), false);
            assert_eq!(op.unique_indices(), false);
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "scatter_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), indices_type.into(), update_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @scatter_test(\
                    %arg0: tensor<3x4x2xi32>, \
                    %arg1: tensor<2x2xi64>, \
                    %arg2: tensor<2x2x2xi32>\
                  ) -> tensor<3x4x2xi32> {
                    %0 = \"stablehlo.scatter\"(%arg0, %arg1, %arg2) <{\
                    indices_are_sorted = false, \
                    scatter_dimension_numbers = #stablehlo.scatter<\
                      update_window_dims = [1, 2], \
                      inserted_window_dims = [0], \
                      scatter_dims_to_operand_dims = [0, 1], \
                      index_vector_dim = 1\
                    >, \
                    unique_indices = false\
                  }> ({
                    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
                      %1 = stablehlo.add %arg3, %arg4 : tensor<i32>
                      stablehlo.return %1 : tensor<i32>
                    }) : (tensor<3x4x2xi32>, tensor<2x2xi64>, tensor<2x2x2xi32>) -> tensor<3x4x2xi32>
                    return %0 : tensor<3x4x2xi32>
                  }
                }
            "}
        );
    }

    #[test]
    fn test_select_and_scatter() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i32_scalar_type = context.tensor_type(i32_type, &[], None, location).unwrap();
        let input_type = context.tensor_type(i32_type, &[Size::Static(4), Size::Static(6)], None, location).unwrap();
        let source_type = context.tensor_type(i32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let output_type = context.tensor_type(i32_type, &[Size::Static(4), Size::Static(6)], None, location).unwrap();
        module.body().append_operation({
            let mut block =
                context.block(&[(input_type, location), (source_type, location), (i32_scalar_type, location)]);
            let input = block.argument(0).unwrap();
            let source = block.argument(1).unwrap();
            let initial_value = block.argument(2).unwrap();

            // Create the `select` region.
            let mut select_region = context.region();
            let mut select_block = context.block(&[(i32_scalar_type, location), (i32_scalar_type, location)]);
            let compare_op = stable_hlo::compare(
                select_block.argument(0).unwrap(),
                select_block.argument(1).unwrap(),
                stable_hlo::ComparisonDirection::GreaterThanOrEqual,
                stable_hlo::ComparisonType::Signed,
                location,
            );
            let compare_op = select_block.append_operation(compare_op);
            let return_op = stable_hlo::r#return(&[compare_op.result(0).unwrap()], location);
            select_block.append_operation(return_op);
            select_region.append_block(select_block);

            // Create the `scatter` region.
            let mut scatter_region = context.region();
            let mut scatter_block = context.block(&[(i32_scalar_type, location), (i32_scalar_type, location)]);
            let add_op =
                stable_hlo::add(scatter_block.argument(0).unwrap(), scatter_block.argument(1).unwrap(), location);
            let add_op = scatter_block.append_operation(add_op);
            let return_op = stable_hlo::r#return(&[add_op.result(0).unwrap()], location);
            scatter_block.append_operation(return_op);
            scatter_region.append_block(scatter_block);

            // Create the `select_and_scatter` operation.
            let op = select_and_scatter(
                input,
                source,
                initial_value,
                Some(&[2, 3]),
                Some(&[2, 3]),
                Some(&[(0, 1), (0, 0)]),
                select_region.into(),
                scatter_region.into(),
                location,
            );
            assert_eq!(op.input(), input);
            assert_eq!(op.source(), source);
            assert_eq!(op.initial_value(), initial_value);
            assert_eq!(op.window_dimensions(), Some(vec![2, 3]));
            assert_eq!(op.window_strides(), Some(vec![2, 3]));
            assert_eq!(op.padding(), Some(vec![(0, 1), (0, 0)]));
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), output_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "select_and_scatter_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), source_type.into(), i32_scalar_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @select_and_scatter_test(\
                    %arg0: tensor<4x6xi32>, \
                    %arg1: tensor<2x2xi32>, \
                    %arg2: tensor<i32>\
                  ) -> tensor<4x6xi32> {
                    %0 = \"stablehlo.select_and_scatter\"(%arg0, %arg1, %arg2) <{\
                      padding = dense<[[0, 1], [0, 0]]> : tensor<2x2xi64>, \
                      window_dimensions = array<i64: 2, 3>, \
                      window_strides = array<i64: 2, 3>\
                    }> ({
                    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
                      %1 = stablehlo.compare  GE, %arg3, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
                      stablehlo.return %1 : tensor<i1>
                    }, {
                    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
                      %1 = stablehlo.add %arg3, %arg4 : tensor<i32>
                      stablehlo.return %1 : tensor<i32>
                    }) : (tensor<4x6xi32>, tensor<2x2xi32>, tensor<i32>) -> tensor<4x6xi32>
                    return %0 : tensor<4x6xi32>
                  }
                }
            "}
        );
    }
}
