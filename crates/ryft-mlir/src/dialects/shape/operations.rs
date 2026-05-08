use std::collections::HashMap;

use crate::{
    Attribute, AttributeRef, DenseIntegerElementsAttributeRef, DetachedOp, DetachedRegion, DialectHandle,
    DictionaryAttributeRef, Error, FUNCTION_TYPE_ATTRIBUTE, HasCallableArgumentAndResultAttributes, Location,
    Operation, OperationBuilder, RegionRef, SYMBOL_NAME_ATTRIBUTE, SYMBOL_VISIBILITY_ATTRIBUTE, StringAttributeRef,
    SymbolVisibility, TryIntoWithContext, Type, TypeAndAttributes, Value, ValueRef, mlir_op, mlir_op_trait,
};

/// Operation trait for `shape.add`.
pub trait AddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand size or index.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right-hand size or index.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(Add);
mlir_op_trait!(Add, AlwaysSpeculatable);
mlir_op_trait!(Add, NoMemoryEffect);
mlir_op_trait!(Add, OneResult);
mlir_op_trait!(Add, Pure);
mlir_op_trait!(Add, ZeroRegions);
mlir_op_trait!(Add, ZeroSuccessors);

/// Constructs a new detached/owned [`AddOperation`] at the specified [`Location`].
pub fn add<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    result_type: T,
    location: L,
) -> Result<DetachedAddOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.add", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::add`"))
        })
}

/// Name of the optional Shape error message attribute.
pub const ERROR_ATTRIBUTE: &str = "error";

/// Operation trait for `shape.broadcast`.
pub trait BroadcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input shapes or extent tensors.
    fn shapes(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }

    /// Returns the optional error message.
    fn error(&self) -> Result<Option<StringAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(ERROR_ATTRIBUTE) { self.string_attribute(ERROR_ATTRIBUTE).map(Some) } else { Ok(None) }
    }
}

mlir_op!(Broadcast);
mlir_op_trait!(Broadcast, AlwaysSpeculatable);
mlir_op_trait!(Broadcast, NoMemoryEffect);
mlir_op_trait!(Broadcast, OneResult);
mlir_op_trait!(Broadcast, Pure);
mlir_op_trait!(Broadcast, ZeroRegions);
mlir_op_trait!(Broadcast, ZeroSuccessors);

/// Constructs a new detached/owned [`BroadcastOperation`] at the specified [`Location`].
pub fn broadcast<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    shapes: &[ValueRef<'v, 'c, 't>],
    result_type: T,
    error: Option<StringAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedBroadcastOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    let mut builder = OperationBuilder::new("shape.broadcast", location).add_operands(shapes).add_result(result_type);
    if let Some(error) = error {
        builder = builder.add_attribute(ERROR_ATTRIBUTE, error);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::broadcast`"))
    })
}

/// Name of the Shape constant shape attribute.
pub const SHAPE_ATTRIBUTE: &str = "shape";

/// Operation trait for `shape.const_shape`.
pub trait ConstShapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the dense shape attribute.
    fn shape(&self) -> Result<DenseIntegerElementsAttributeRef<'c, 't>, Error> {
        self.dense_integer_elements_attribute(SHAPE_ATTRIBUTE)
    }
}

mlir_op!(ConstShape);
mlir_op_trait!(ConstShape, AlwaysSpeculatable);
mlir_op_trait!(ConstShape, ConstantLike);
mlir_op_trait!(ConstShape, NoMemoryEffect);
mlir_op_trait!(ConstShape, OneResult);
mlir_op_trait!(ConstShape, Pure);
mlir_op_trait!(ConstShape, ZeroOperands);
mlir_op_trait!(ConstShape, ZeroRegions);
mlir_op_trait!(ConstShape, ZeroSuccessors);

/// Constructs a new detached/owned [`ConstShapeOperation`] at the specified [`Location`].
pub fn const_shape<'c, 't: 'c, A: Attribute<'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    shape: A,
    result_type: T,
    location: L,
) -> Result<DetachedConstShapeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.const_shape", location)
        .add_attribute(SHAPE_ATTRIBUTE, shape)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::const_shape`"))
        })
}

/// Name of the Shape scalar value attribute.
pub const VALUE_ATTRIBUTE: &str = "value";

/// Operation trait for `shape.const_size`.
pub trait ConstSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the constant size value.
    fn value(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(VALUE_ATTRIBUTE)?.signless_value())
    }
}

mlir_op!(ConstSize);
mlir_op_trait!(ConstSize, AlwaysSpeculatable);
mlir_op_trait!(ConstSize, ConstantLike);
mlir_op_trait!(ConstSize, NoMemoryEffect);
mlir_op_trait!(ConstSize, OneResult);
mlir_op_trait!(ConstSize, Pure);
mlir_op_trait!(ConstSize, ZeroOperands);
mlir_op_trait!(ConstSize, ZeroRegions);
mlir_op_trait!(ConstSize, ZeroSuccessors);

/// Constructs a new detached/owned [`ConstSizeOperation`] at the specified [`Location`].
pub fn const_size<'c, 't: 'c, L: Location<'c, 't>>(
    value: i64,
    location: L,
) -> Result<DetachedConstSizeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.const_size", location)
        .add_attribute(VALUE_ATTRIBUTE, context.integer_attribute(context.index_type(), value))
        .add_result(context.shape_size_type()?)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::const_size`"))
        })
}

/// Operation trait for `shape.div`.
pub trait DivOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the dividend.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the divisor.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(Div);
mlir_op_trait!(Div, AlwaysSpeculatable);
mlir_op_trait!(Div, NoMemoryEffect);
mlir_op_trait!(Div, OneResult);
mlir_op_trait!(Div, Pure);
mlir_op_trait!(Div, ZeroRegions);
mlir_op_trait!(Div, ZeroSuccessors);

/// Constructs a new detached/owned [`DivOperation`] at the specified [`Location`].
pub fn div<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    result_type: T,
    location: L,
) -> Result<DetachedDivOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.div", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::div`"))
        })
}

/// Operation trait for `shape.shape_eq`.
pub trait ShapeEqOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the compared shapes or extent tensors.
    fn shapes(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }
}

mlir_op!(ShapeEq);
mlir_op_trait!(ShapeEq, AlwaysSpeculatable);
mlir_op_trait!(ShapeEq, NoMemoryEffect);
mlir_op_trait!(ShapeEq, OneResult);
mlir_op_trait!(ShapeEq, Pure);
mlir_op_trait!(ShapeEq, ZeroRegions);
mlir_op_trait!(ShapeEq, ZeroSuccessors);

/// Constructs a new detached/owned [`ShapeEqOperation`] at the specified [`Location`].
pub fn shape_eq<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    shapes: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedShapeEqOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.shape_eq", location)
        .add_operands(shapes)
        .add_result(context.signless_integer_type(1))
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::shape_eq`"))
        })
}

/// Operation trait for `shape.from_extents`.
pub trait FromExtentsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the extents forming the shape.
    fn extents(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }
}

mlir_op!(FromExtents);
mlir_op_trait!(FromExtents, AlwaysSpeculatable);
mlir_op_trait!(FromExtents, NoMemoryEffect);
mlir_op_trait!(FromExtents, OneResult);
mlir_op_trait!(FromExtents, Pure);
mlir_op_trait!(FromExtents, ZeroRegions);
mlir_op_trait!(FromExtents, ZeroSuccessors);

/// Constructs a new detached/owned [`FromExtentsOperation`] at the specified [`Location`].
pub fn from_extents<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    extents: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedFromExtentsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.from_extents", location)
        .add_operands(extents)
        .add_result(context.shape_type()?)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::from_extents`"))
        })
}

/// Operation trait for `shape.from_extent_tensor`.
pub trait FromExtentTensorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input extent tensor.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(FromExtentTensor);
mlir_op_trait!(FromExtentTensor, AlwaysSpeculatable);
mlir_op_trait!(FromExtentTensor, NoMemoryEffect);
mlir_op_trait!(FromExtentTensor, OneResult);
mlir_op_trait!(FromExtentTensor, Pure);
mlir_op_trait!(FromExtentTensor, ZeroRegions);
mlir_op_trait!(FromExtentTensor, ZeroSuccessors);

/// Constructs a new detached/owned [`FromExtentTensorOperation`] at the specified [`Location`].
pub fn from_extent_tensor<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedFromExtentTensorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.from_extent_tensor", location)
        .add_operand(input)
        .add_result(context.shape_type()?)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::from_extent_tensor`"))
        })
}

/// Operation trait for `shape.is_broadcastable`.
pub trait IsBroadcastableOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the checked shapes or extent tensors.
    fn shapes(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }
}

mlir_op!(IsBroadcastable);
mlir_op_trait!(IsBroadcastable, OneResult);
mlir_op_trait!(IsBroadcastable, ZeroRegions);
mlir_op_trait!(IsBroadcastable, ZeroSuccessors);

/// Constructs a new detached/owned [`IsBroadcastableOperation`] at the specified [`Location`].
pub fn is_broadcastable<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    shapes: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedIsBroadcastableOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.is_broadcastable", location)
        .add_operands(shapes)
        .add_result(context.signless_integer_type(1))
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::is_broadcastable`"))
        })
}

/// Operation trait for `shape.rank`.
pub trait RankOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input shape or extent tensor.
    fn shape(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(Rank);
mlir_op_trait!(Rank, AlwaysSpeculatable);
mlir_op_trait!(Rank, NoMemoryEffect);
mlir_op_trait!(Rank, OneResult);
mlir_op_trait!(Rank, Pure);
mlir_op_trait!(Rank, ZeroRegions);
mlir_op_trait!(Rank, ZeroSuccessors);

/// Constructs a new detached/owned [`RankOperation`] at the specified [`Location`].
pub fn rank<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    shape: V,
    result_type: T,
    location: L,
) -> Result<DetachedRankOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.rank", location)
        .add_operand(shape)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::rank`"))
        })
}

/// Operation trait for `shape.to_extent_tensor`.
pub trait ToExtentTensorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input shape.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(ToExtentTensor);
mlir_op_trait!(ToExtentTensor, AlwaysSpeculatable);
mlir_op_trait!(ToExtentTensor, NoMemoryEffect);
mlir_op_trait!(ToExtentTensor, OneResult);
mlir_op_trait!(ToExtentTensor, Pure);
mlir_op_trait!(ToExtentTensor, ZeroRegions);
mlir_op_trait!(ToExtentTensor, ZeroSuccessors);

/// Constructs a new detached/owned [`ToExtentTensorOperation`] at the specified [`Location`].
pub fn to_extent_tensor<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: V,
    result_type: T,
    location: L,
) -> Result<DetachedToExtentTensorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.to_extent_tensor", location)
        .add_operand(input)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::to_extent_tensor`"))
        })
}

/// Operation trait for `shape.dim`.
pub trait DimOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the shaped input value.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the dimension index.
    fn index(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(Dim);
mlir_op_trait!(Dim, AlwaysSpeculatable);
mlir_op_trait!(Dim, NoMemoryEffect);
mlir_op_trait!(Dim, OneResult);
mlir_op_trait!(Dim, Pure);
mlir_op_trait!(Dim, ZeroRegions);
mlir_op_trait!(Dim, ZeroSuccessors);

/// Constructs a new detached/owned [`DimOperation`] at the specified [`Location`].
pub fn dim<
    'value,
    'index,
    'c: 'value + 'index,
    't: 'c,
    V: Value<'value, 'c, 't>,
    I: Value<'index, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    value: V,
    index: I,
    result_type: T,
    location: L,
) -> Result<DetachedDimOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.dim", location)
        .add_operands(&[value.as_ref(), index.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::dim`"))
        })
}

/// Operation trait for `shape.get_extent`.
pub trait GetExtentOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input shape or extent tensor.
    fn shape(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the dimension index.
    fn dimension(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(GetExtent);
mlir_op_trait!(GetExtent, AlwaysSpeculatable);
mlir_op_trait!(GetExtent, NoMemoryEffect);
mlir_op_trait!(GetExtent, OneResult);
mlir_op_trait!(GetExtent, Pure);
mlir_op_trait!(GetExtent, ZeroRegions);
mlir_op_trait!(GetExtent, ZeroSuccessors);

/// Constructs a new detached/owned [`GetExtentOperation`] at the specified [`Location`].
pub fn get_extent<
    'shape,
    'dimension,
    'c: 'shape + 'dimension,
    't: 'c,
    S: Value<'shape, 'c, 't>,
    D: Value<'dimension, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    shape: S,
    dimension: D,
    result_type: T,
    location: L,
) -> Result<DetachedGetExtentOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.get_extent", location)
        .add_operands(&[shape.as_ref(), dimension.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::get_extent`"))
        })
}

/// Operation trait for `shape.index_to_size`.
pub trait IndexToSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the index input.
    fn argument(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(IndexToSize);
mlir_op_trait!(IndexToSize, AlwaysSpeculatable);
mlir_op_trait!(IndexToSize, NoMemoryEffect);
mlir_op_trait!(IndexToSize, OneResult);
mlir_op_trait!(IndexToSize, Pure);
mlir_op_trait!(IndexToSize, ZeroRegions);
mlir_op_trait!(IndexToSize, ZeroSuccessors);

/// Constructs a new detached/owned [`IndexToSizeOperation`] at the specified [`Location`].
pub fn index_to_size<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    argument: V,
    location: L,
) -> Result<DetachedIndexToSizeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.index_to_size", location)
        .add_operand(argument)
        .add_result(context.shape_size_type()?)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::index_to_size`"))
        })
}

/// Operation trait for `shape.max`.
pub trait MaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand shape or size.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right-hand shape or size.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(Max);
mlir_op_trait!(Max, AlwaysSpeculatable);
mlir_op_trait!(Max, NoMemoryEffect);
mlir_op_trait!(Max, OneResult);
mlir_op_trait!(Max, Pure);
mlir_op_trait!(Max, ZeroRegions);
mlir_op_trait!(Max, ZeroSuccessors);

/// Constructs a new detached/owned [`MaxOperation`] at the specified [`Location`].
pub fn max<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    result_type: T,
    location: L,
) -> Result<DetachedMaxOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.max", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::max`"))
        })
}

/// Operation trait for `shape.meet`.
pub trait MeetOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the first shape or size.
    fn first_argument(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the second shape or size.
    fn second_argument(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the optional error message.
    fn error(&self) -> Result<Option<StringAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(ERROR_ATTRIBUTE) { self.string_attribute(ERROR_ATTRIBUTE).map(Some) } else { Ok(None) }
    }
}

mlir_op!(Meet);
mlir_op_trait!(Meet, OneResult);
mlir_op_trait!(Meet, ZeroRegions);
mlir_op_trait!(Meet, ZeroSuccessors);

/// Constructs a new detached/owned [`MeetOperation`] at the specified [`Location`].
pub fn meet<
    'first,
    'second,
    'c: 'first + 'second,
    't: 'c,
    A0: Value<'first, 'c, 't>,
    A1: Value<'second, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    first_argument: A0,
    second_argument: A1,
    result_type: T,
    error: Option<StringAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedMeetOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    let mut builder = OperationBuilder::new("shape.meet", location)
        .add_operands(&[first_argument.as_ref(), second_argument.as_ref()])
        .add_result(result_type);
    if let Some(error) = error {
        builder = builder.add_attribute(ERROR_ATTRIBUTE, error);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::meet`"))
    })
}

/// Operation trait for `shape.min`.
pub trait MinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand shape or size.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right-hand shape or size.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(Min);
mlir_op_trait!(Min, AlwaysSpeculatable);
mlir_op_trait!(Min, NoMemoryEffect);
mlir_op_trait!(Min, OneResult);
mlir_op_trait!(Min, Pure);
mlir_op_trait!(Min, ZeroRegions);
mlir_op_trait!(Min, ZeroSuccessors);

/// Constructs a new detached/owned [`MinOperation`] at the specified [`Location`].
pub fn min<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    result_type: T,
    location: L,
) -> Result<DetachedMinOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.min", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::min`"))
        })
}

/// Operation trait for `shape.mul`.
pub trait MulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand size or index.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right-hand size or index.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(Mul);
mlir_op_trait!(Mul, AlwaysSpeculatable);
mlir_op_trait!(Mul, NoMemoryEffect);
mlir_op_trait!(Mul, OneResult);
mlir_op_trait!(Mul, Pure);
mlir_op_trait!(Mul, ZeroRegions);
mlir_op_trait!(Mul, ZeroSuccessors);

/// Constructs a new detached/owned [`MulOperation`] at the specified [`Location`].
pub fn mul<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    result_type: T,
    location: L,
) -> Result<DetachedMulOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.mul", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::mul`"))
        })
}

/// Operation trait for `shape.num_elements`.
pub trait NumElementsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input shape or extent tensor.
    fn shape(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(NumElements);
mlir_op_trait!(NumElements, AlwaysSpeculatable);
mlir_op_trait!(NumElements, NoMemoryEffect);
mlir_op_trait!(NumElements, OneResult);
mlir_op_trait!(NumElements, Pure);
mlir_op_trait!(NumElements, ZeroRegions);
mlir_op_trait!(NumElements, ZeroSuccessors);

/// Constructs a new detached/owned [`NumElementsOperation`] at the specified [`Location`].
pub fn num_elements<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    shape: V,
    result_type: T,
    location: L,
) -> Result<DetachedNumElementsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.num_elements", location)
        .add_operand(shape)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::num_elements`"))
        })
}

/// Operation trait for `shape.reduce`.
pub trait ReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input shape or extent tensor.
    fn shape(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the initial reduction values.
    fn initial_values(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values().skip(1)
    }

    /// Returns the reduction body region.
    fn region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.as_ref().region(0)
    }
}

mlir_op!(Reduce);
mlir_op_trait!(Reduce, OneRegion);
mlir_op_trait!(Reduce, SingleBlock);
mlir_op_trait!(Reduce, SingleBlockRegions);
mlir_op_trait!(Reduce, ZeroSuccessors);

/// Constructs a new detached/owned [`ReduceOperation`] at the specified [`Location`].
pub fn reduce<'v, 'c: 'v, 't: 'c, S: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    shape: S,
    initial_values: &[ValueRef<'v, 'c, 't>],
    result_types: &[crate::TypeRef<'c, 't>],
    region: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedReduceOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.reduce", location)
        .add_operand(shape)
        .add_operands(initial_values)
        .add_results(result_types)
        .add_region(region)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::reduce`"))
        })
}

/// Operation trait for `shape.shape_of`.
pub trait ShapeOfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the shaped input or value-shape input.
    fn argument(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(ShapeOf);
mlir_op_trait!(ShapeOf, AlwaysSpeculatable);
mlir_op_trait!(ShapeOf, NoMemoryEffect);
mlir_op_trait!(ShapeOf, OneResult);
mlir_op_trait!(ShapeOf, Pure);
mlir_op_trait!(ShapeOf, ZeroRegions);
mlir_op_trait!(ShapeOf, ZeroSuccessors);

/// Constructs a new detached/owned [`ShapeOfOperation`] at the specified [`Location`].
pub fn shape_of<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    argument: V,
    result_type: T,
    location: L,
) -> Result<DetachedShapeOfOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.shape_of", location)
        .add_operand(argument)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::shape_of`"))
        })
}

/// Operation trait for `shape.value_of`.
pub trait ValueOfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value-shape input.
    fn argument(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(ValueOf);
mlir_op_trait!(ValueOf, AlwaysSpeculatable);
mlir_op_trait!(ValueOf, NoMemoryEffect);
mlir_op_trait!(ValueOf, OneResult);
mlir_op_trait!(ValueOf, Pure);
mlir_op_trait!(ValueOf, ZeroRegions);
mlir_op_trait!(ValueOf, ZeroSuccessors);

/// Constructs a new detached/owned [`ValueOfOperation`] at the specified [`Location`].
pub fn value_of<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    argument: V,
    result_type: T,
    location: L,
) -> Result<DetachedValueOfOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.value_of", location)
        .add_operand(argument)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::value_of`"))
        })
}

/// Operation trait for `shape.size_to_index`.
pub trait SizeToIndexOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the size input.
    fn argument(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(SizeToIndex);
mlir_op_trait!(SizeToIndex, AlwaysSpeculatable);
mlir_op_trait!(SizeToIndex, NoMemoryEffect);
mlir_op_trait!(SizeToIndex, OneResult);
mlir_op_trait!(SizeToIndex, Pure);
mlir_op_trait!(SizeToIndex, ZeroRegions);
mlir_op_trait!(SizeToIndex, ZeroSuccessors);

/// Constructs a new detached/owned [`SizeToIndexOperation`] at the specified [`Location`].
pub fn size_to_index<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    argument: V,
    location: L,
) -> Result<DetachedSizeToIndexOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.size_to_index", location)
        .add_operand(argument)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::size_to_index`"))
        })
}

/// Operation trait for `shape.value_as_shape`.
pub trait ValueAsShapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn argument(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(ValueAsShape);
mlir_op_trait!(ValueAsShape, AlwaysSpeculatable);
mlir_op_trait!(ValueAsShape, NoMemoryEffect);
mlir_op_trait!(ValueAsShape, OneResult);
mlir_op_trait!(ValueAsShape, Pure);
mlir_op_trait!(ValueAsShape, ZeroRegions);
mlir_op_trait!(ValueAsShape, ZeroSuccessors);

/// Constructs a new detached/owned [`ValueAsShapeOperation`] at the specified [`Location`].
pub fn value_as_shape<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    argument: V,
    result_type: T,
    location: L,
) -> Result<DetachedValueAsShapeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.value_as_shape", location)
        .add_operand(argument)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::value_as_shape`"))
        })
}

/// Operation trait for `shape.with_shape`.
pub trait WithOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the operand value.
    fn operand(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the replacement shape.
    fn shape(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(With);
mlir_op_trait!(With, AlwaysSpeculatable);
mlir_op_trait!(With, NoMemoryEffect);
mlir_op_trait!(With, OneResult);
mlir_op_trait!(With, Pure);
mlir_op_trait!(With, ZeroRegions);
mlir_op_trait!(With, ZeroSuccessors);

/// Constructs a new detached/owned [`WithOperation`] at the specified [`Location`].
pub fn with_shape<
    'operand,
    'shape,
    'c: 'operand + 'shape,
    't: 'c,
    O: Value<'operand, 'c, 't>,
    S: Value<'shape, 'c, 't>,
    L: Location<'c, 't>,
>(
    operand: O,
    shape: S,
    location: L,
) -> Result<DetachedWithOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.with_shape", location)
        .add_operands(&[operand.as_ref(), shape.as_ref()])
        .add_result(context.shape_value_shape_type()?)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::with_shape`"))
        })
}

/// Operation trait for `shape.yield`.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the yielded values.
    fn values(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, AlwaysSpeculatable);
mlir_op_trait!(Yield, NoMemoryEffect);
mlir_op_trait!(Yield, Pure);
mlir_op_trait!(Yield, ReturnLike);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Constructs a new detached/owned [`YieldOperation`] at the specified [`Location`].
pub fn r#yield<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedYieldOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.yield", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::yield`"))
        })
}

/// Operation trait for `shape.debug_print`.
pub trait DebugPrintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the printed input.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(DebugPrint);
mlir_op_trait!(DebugPrint, OneResult);
mlir_op_trait!(DebugPrint, ZeroRegions);
mlir_op_trait!(DebugPrint, ZeroSuccessors);

/// Constructs a new detached/owned [`DebugPrintOperation`] at the specified [`Location`].
pub fn debug_print<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedDebugPrintOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.debug_print", location)
        .add_operand(input.as_ref())
        .add_result(input.r#type()?)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::debug_print`"))
        })
}

/// Operation trait for `shape.split_at`.
pub trait SplitAtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input shape or extent tensor.
    fn operand(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the split index.
    fn index(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the leading portion of the split shape.
    fn head(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0).map(|result| result.as_ref())
    }

    /// Returns the trailing portion of the split shape.
    fn tail(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.as_ref().result(1).map(|result| result.as_ref())
    }
}

mlir_op!(SplitAt);
mlir_op_trait!(SplitAt, AlwaysSpeculatable);
mlir_op_trait!(SplitAt, NoMemoryEffect);
mlir_op_trait!(SplitAt, Pure);
mlir_op_trait!(SplitAt, ZeroRegions);
mlir_op_trait!(SplitAt, ZeroSuccessors);

/// Constructs a new detached/owned [`SplitAtOperation`] at the specified [`Location`].
pub fn split_at<
    'operand,
    'index,
    'c: 'operand + 'index,
    't: 'c,
    O: Value<'operand, 'c, 't>,
    I: Value<'index, 'c, 't>,
    L: Location<'c, 't>,
>(
    operand: O,
    index: I,
    result_types: [crate::TypeRef<'c, 't>; 2],
    location: L,
) -> Result<DetachedSplitAtOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.split_at", location)
        .add_operands(&[operand.as_ref(), index.as_ref()])
        .add_results(&result_types)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::split_at`"))
        })
}

/// Operation trait for `shape.concat`.
pub trait ConcatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand shape.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right-hand shape.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(Concat);
mlir_op_trait!(Concat, AlwaysSpeculatable);
mlir_op_trait!(Concat, NoMemoryEffect);
mlir_op_trait!(Concat, OneResult);
mlir_op_trait!(Concat, Pure);
mlir_op_trait!(Concat, ZeroRegions);
mlir_op_trait!(Concat, ZeroSuccessors);

/// Constructs a new detached/owned [`ConcatOperation`] at the specified [`Location`].
pub fn concat<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    result_type: T,
    location: L,
) -> Result<DetachedConcatOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.concat", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::concat`"))
        })
}

/// Operation trait for `shape.any`.
pub trait AnyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input shapes or extent tensors.
    fn inputs(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }
}

mlir_op!(Any);
mlir_op_trait!(Any, AlwaysSpeculatable);
mlir_op_trait!(Any, NoMemoryEffect);
mlir_op_trait!(Any, OneResult);
mlir_op_trait!(Any, Pure);
mlir_op_trait!(Any, ZeroRegions);
mlir_op_trait!(Any, ZeroSuccessors);

/// Constructs a new detached/owned [`AnyOperation`] at the specified [`Location`].
pub fn any<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    result_type: T,
    location: L,
) -> Result<DetachedAnyOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.any", location)
        .add_operands(inputs)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::any`"))
        })
}

/// Operation trait for `shape.assuming_all`.
pub trait AssumingAllOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input witnesses.
    fn inputs(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }
}

mlir_op!(AssumingAll);
mlir_op_trait!(AssumingAll, AlwaysSpeculatable);
mlir_op_trait!(AssumingAll, NoMemoryEffect);
mlir_op_trait!(AssumingAll, OneResult);
mlir_op_trait!(AssumingAll, Pure);
mlir_op_trait!(AssumingAll, ZeroRegions);
mlir_op_trait!(AssumingAll, ZeroSuccessors);

/// Constructs a new detached/owned [`AssumingAllOperation`] at the specified [`Location`].
pub fn assuming_all<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedAssumingAllOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.assuming_all", location)
        .add_operands(inputs)
        .add_result(context.shape_witness_type()?)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::assuming_all`"))
        })
}

/// Operation trait for `shape.assuming`.
pub trait AssumingOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the witness controlling the assumption.
    fn witness(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the body region executed under the assumption.
    fn region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.as_ref().region(0)
    }
}

mlir_op!(Assuming);
mlir_op_trait!(Assuming, OneRegion);
mlir_op_trait!(Assuming, SingleBlock);
mlir_op_trait!(Assuming, SingleBlockRegions);
mlir_op_trait!(Assuming, ZeroSuccessors);

/// Constructs a new detached/owned [`AssumingOperation`] at the specified [`Location`].
pub fn assuming<'v, 'c: 'v, 't: 'c, W: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    witness: W,
    result_types: &[crate::TypeRef<'c, 't>],
    region: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedAssumingOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.assuming", location)
        .add_operand(witness)
        .add_results(result_types)
        .add_region(region)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::assuming`"))
        })
}

/// Operation trait for `shape.assuming_yield`.
pub trait AssumingYieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the yielded values.
    fn values(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }
}

mlir_op!(AssumingYield);
mlir_op_trait!(AssumingYield, AlwaysSpeculatable);
mlir_op_trait!(AssumingYield, NoMemoryEffect);
mlir_op_trait!(AssumingYield, Pure);
mlir_op_trait!(AssumingYield, ReturnLike);
mlir_op_trait!(AssumingYield, ZeroRegions);
mlir_op_trait!(AssumingYield, ZeroSuccessors);

/// Constructs a new detached/owned [`AssumingYieldOperation`] at the specified [`Location`].
pub fn assuming_yield<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedAssumingYieldOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.assuming_yield", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::assuming_yield`"))
        })
}

/// Operation trait for `shape.cstr_broadcastable`.
pub trait CstrBroadcastableOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the checked shapes or extent tensors.
    fn shapes(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }
}

mlir_op!(CstrBroadcastable);
mlir_op_trait!(CstrBroadcastable, OneResult);
mlir_op_trait!(CstrBroadcastable, ZeroRegions);
mlir_op_trait!(CstrBroadcastable, ZeroSuccessors);

/// Constructs a new detached/owned [`CstrBroadcastableOperation`] at the specified [`Location`].
pub fn cstr_broadcastable<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    shapes: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedCstrBroadcastableOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.cstr_broadcastable", location)
        .add_operands(shapes)
        .add_result(context.shape_witness_type()?)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::cstr_broadcastable`"))
        })
}

/// Operation trait for `shape.cstr_eq`.
pub trait CstrEqOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the compared shapes.
    fn shapes(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }
}

mlir_op!(CstrEq);
mlir_op_trait!(CstrEq, OneResult);
mlir_op_trait!(CstrEq, ZeroRegions);
mlir_op_trait!(CstrEq, ZeroSuccessors);

/// Constructs a new detached/owned [`CstrEqOperation`] at the specified [`Location`].
pub fn cstr_eq<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    shapes: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedCstrEqOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.cstr_eq", location)
        .add_operands(shapes)
        .add_result(context.shape_witness_type()?)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::cstr_eq`"))
        })
}

/// Name of the Shape witness passing attribute.
pub const PASSING_ATTRIBUTE: &str = "passing";

/// Operation trait for `shape.const_witness`.
pub trait ConstWitnessOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns whether this witness is statically known to pass.
    fn passing(&self) -> Result<bool, Error> {
        Ok(self.boolean_attribute(PASSING_ATTRIBUTE)?.value())
    }
}

mlir_op!(ConstWitness);
mlir_op_trait!(ConstWitness, AlwaysSpeculatable);
mlir_op_trait!(ConstWitness, ConstantLike);
mlir_op_trait!(ConstWitness, NoMemoryEffect);
mlir_op_trait!(ConstWitness, OneResult);
mlir_op_trait!(ConstWitness, Pure);
mlir_op_trait!(ConstWitness, ZeroOperands);
mlir_op_trait!(ConstWitness, ZeroRegions);
mlir_op_trait!(ConstWitness, ZeroSuccessors);

/// Constructs a new detached/owned [`ConstWitnessOperation`] at the specified [`Location`].
pub fn const_witness<'c, 't: 'c, L: Location<'c, 't>>(
    passing: bool,
    location: L,
) -> Result<DetachedConstWitnessOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.const_witness", location)
        .add_attribute(PASSING_ATTRIBUTE, context.boolean_attribute(passing))
        .add_result(context.shape_witness_type()?)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::const_witness`"))
        })
}

/// Name of the Shape constraint message attribute.
pub const MESSAGE_ATTRIBUTE: &str = "msg";

/// Operation trait for `shape.cstr_require`.
pub trait CstrRequireOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the required predicate.
    fn predicate(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the diagnostic message.
    fn message(&self) -> Result<StringAttributeRef<'c, 't>, Error> {
        self.string_attribute(MESSAGE_ATTRIBUTE)
    }
}

mlir_op!(CstrRequire);
mlir_op_trait!(CstrRequire, OneResult);
mlir_op_trait!(CstrRequire, ZeroRegions);
mlir_op_trait!(CstrRequire, ZeroSuccessors);

/// Constructs a new detached/owned [`CstrRequireOperation`] at the specified [`Location`].
pub fn cstr_require<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    predicate: V,
    message: StringAttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedCstrRequireOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.cstr_require", location)
        .add_operand(predicate)
        .add_attribute(MESSAGE_ATTRIBUTE, message)
        .add_result(context.shape_witness_type()?)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::cstr_require`"))
        })
}

/// Name of the Shape function-library mapping attribute.
pub const MAPPING_ATTRIBUTE: &str = "mapping";

/// Operation trait for `shape.function_library`.
pub trait FunctionLibraryOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the mapping from operation names to shape functions.
    fn mapping(&self) -> Result<DictionaryAttributeRef<'c, 't>, Error> {
        self.dictionary_attribute(MAPPING_ATTRIBUTE)
    }

    /// Returns the body region.
    fn body_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }
}

mlir_op!(FunctionLibrary);
mlir_op_trait!(FunctionLibrary, AffineScope);
mlir_op_trait!(FunctionLibrary, OneRegion);
mlir_op_trait!(FunctionLibrary, SingleBlock);
mlir_op_trait!(FunctionLibrary, SingleBlockRegions);
mlir_op_trait!(FunctionLibrary, Symbol);
mlir_op_trait!(FunctionLibrary, SymbolTable);
mlir_op_trait!(FunctionLibrary, ZeroSuccessors);

/// Constructs a new detached/owned [`FunctionLibraryOperation`] at the specified [`Location`].
pub fn function_library<'c, 't: 'c, N: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    name: N,
    mapping: DictionaryAttributeRef<'c, 't>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedFunctionLibraryOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.function_library", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, name.try_into_with_context(context)?)
        .add_attribute(MAPPING_ATTRIBUTE, mapping)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::function_library`"))
        })
}

/// Operation trait for `shape.func`.
pub trait FuncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Func);
mlir_op_trait!(Func, AffineScope);
mlir_op_trait!(Func, AutomaticAllocationScope);
mlir_op_trait!(Func, Callable);
mlir_op_trait!(Func, Function);
mlir_op_trait!(Func, HasCallableArgumentAndResultAttributes);
mlir_op_trait!(Func, IsolatedFromAbove);
mlir_op_trait!(Func, Symbol);

/// Structured representation of the attributes attached to a [`FuncOperation`].
pub struct FuncAttributes<'c, 't, 's> {
    /// Function arguments and their optional attributes.
    pub arguments: Vec<TypeAndAttributes<'c, 't, 's>>,

    /// Function results and their optional attributes.
    pub results: Vec<TypeAndAttributes<'c, 't, 's>>,

    /// Function symbol visibility.
    pub visibility: SymbolVisibility,

    /// Additional attributes attached to the function.
    pub other_attributes: HashMap<&'c str, AttributeRef<'c, 't>>,
}

impl Default for FuncAttributes<'_, '_, '_> {
    fn default() -> Self {
        Self {
            arguments: Vec::new(),
            results: Vec::new(),
            visibility: SymbolVisibility::Public,
            other_attributes: HashMap::new(),
        }
    }
}

/// Constructs a new detached/owned [`FuncOperation`] at the specified [`Location`].
pub fn func<'c, 't: 'c, 's, N: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    name: N,
    attributes: FuncAttributes<'c, 't, 's>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedFuncOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    let mut builder = OperationBuilder::new("shape.func", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, name.try_into_with_context(context)?);
    builder = builder.add_attribute(
        FUNCTION_TYPE_ATTRIBUTE,
        context.type_attribute(context.function_type(
            &attributes.arguments.iter().map(|argument| argument.r#type).collect::<Vec<_>>(),
            &attributes.results.iter().map(|result| result.r#type).collect::<Vec<_>>(),
        )),
    );
    if attributes.arguments.iter().any(|argument| argument.attributes.is_some()) {
        builder = DetachedFuncOperation::<'c, 't>::add_callable_argument_attributes(
            builder,
            attributes.arguments.iter().map(|argument| &argument.attributes),
        )?;
    }
    if attributes.results.iter().any(|result| result.attributes.is_some()) {
        builder = DetachedFuncOperation::<'c, 't>::add_callable_result_attributes(
            builder,
            attributes.results.iter().map(|result| &result.attributes),
        )?;
    }
    if attributes.visibility != SymbolVisibility::default() {
        builder = builder
            .add_attribute(SYMBOL_VISIBILITY_ATTRIBUTE, context.symbol_visibility_attribute(attributes.visibility));
    }
    for (attribute_name, attribute) in &attributes.other_attributes {
        builder = builder.add_attribute(*attribute_name, *attribute);
    }
    builder.add_region(body).build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::func`"))
    })
}

/// Operation trait for `shape.return`.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the returned values.
    fn values(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }
}

mlir_op!(Return);
mlir_op_trait!(Return, AlwaysSpeculatable);
mlir_op_trait!(Return, NoMemoryEffect);
mlir_op_trait!(Return, Pure);
mlir_op_trait!(Return, ReturnLike);
mlir_op_trait!(Return, ZeroRegions);
mlir_op_trait!(Return, ZeroSuccessors);

/// Constructs a new detached/owned [`ReturnOperation`] at the specified [`Location`].
pub fn r#return<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedReturnOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shape()?)?;
    OperationBuilder::new("shape.return", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shape::return`"))
        })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func as func_dialect;
    use crate::{Block, Context, DenseIntegerElementsAttributeRef, OneResult, Region, Size, TryFromWithContext, Type};

    use super::*;

    #[test]
    fn test_size_arithmetic_operations() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let size_type = context.shape_size_type().unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();

                let size_2_op = const_size(2, location).unwrap();
                assert_eq!(size_2_op.value().unwrap(), 2);
                assert_eq!(size_2_op.value().unwrap(), 2);
                let size_2 = block.append_operation(size_2_op).unwrap();
                let size_2 = size_2.as_ref().result(0).unwrap().as_ref();

                let size_3_op = const_size(3, location).unwrap();
                assert_eq!(size_3_op.value().unwrap(), 3);
                let size_3 = block.append_operation(size_3_op).unwrap();
                let size_3 = size_3.as_ref().result(0).unwrap().as_ref();

                let add_op = add(size_2, size_3, size_type, location).unwrap();
                assert_eq!(add_op.lhs().unwrap(), size_2);
                assert_eq!(add_op.rhs().unwrap(), size_3);
                assert_eq!(add_op.output_type().unwrap(), size_type);
                let add_op = block.append_operation(add_op).unwrap();
                let add_value = add_op.as_ref().result(0).unwrap().as_ref();

                let div_op = div(size_3, size_2, size_type, location).unwrap();
                assert_eq!(div_op.lhs().unwrap(), size_3);
                assert_eq!(div_op.rhs().unwrap(), size_2);
                let div_op = block.append_operation(div_op).unwrap();
                let div_value = div_op.as_ref().result(0).unwrap().as_ref();

                let max_op = max(add_value, div_value, size_type, location).unwrap();
                assert_eq!(max_op.lhs().unwrap(), add_value);
                assert_eq!(max_op.rhs().unwrap(), div_value);
                let max_op = block.append_operation(max_op).unwrap();
                let max_value = max_op.as_ref().result(0).unwrap().as_ref();

                let meet_error = context.string_attribute("size mismatch");
                let meet_op = meet(max_value, size_3, size_type, Some(meet_error), location).unwrap();
                assert_eq!(meet_op.first_argument().unwrap(), max_value);
                assert_eq!(meet_op.second_argument().unwrap(), size_3);
                assert_eq!(meet_op.error().unwrap().unwrap().string().as_str().unwrap(), "size mismatch");
                let meet_op = block.append_operation(meet_op).unwrap();
                let meet_value = meet_op.as_ref().result(0).unwrap().as_ref();

                let min_op = min(meet_value, size_2, size_type, location).unwrap();
                assert_eq!(min_op.lhs().unwrap(), meet_value);
                assert_eq!(min_op.rhs().unwrap(), size_2);
                let min_op = block.append_operation(min_op).unwrap();
                let min_value = min_op.as_ref().result(0).unwrap().as_ref();

                let mul_op = mul(min_value, size_3, size_type, location).unwrap();
                assert_eq!(mul_op.lhs().unwrap(), min_value);
                assert_eq!(mul_op.rhs().unwrap(), size_3);
                let mul_op = block.append_operation(mul_op).unwrap();
                let mul_value = mul_op.as_ref().result(0).unwrap().as_ref();

                let size_to_index_op = size_to_index(mul_value, location).unwrap();
                assert_eq!(size_to_index_op.argument().unwrap(), mul_value);
                let size_to_index_op = block.append_operation(size_to_index_op).unwrap();
                let index_value = size_to_index_op.as_ref().result(0).unwrap().as_ref();

                let index_to_size_op = index_to_size(index_value, location).unwrap();
                assert_eq!(index_to_size_op.argument().unwrap(), index_value);
                let index_to_size_op = block.append_operation(index_to_size_op).unwrap();
                let result = index_to_size_op.as_ref().result(0).unwrap().as_ref();

                block.append_operation(func_dialect::r#return(&[result], location).unwrap()).unwrap();
                func_dialect::func(
                    "shape_size_arithmetic_test",
                    func_dialect::FuncAttributes { results: vec![size_type.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap(), "{module}");
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @shape_size_arithmetic_test() -> !shape.size {
                    %c2 = shape.const_size 2
                    %c3 = shape.const_size 3
                    %0 = shape.add %c2, %c3 : !shape.size, !shape.size -> !shape.size
                    %1 = shape.div %c3, %c2 : !shape.size, !shape.size -> !shape.size
                    %2 = shape.max %0, %1 : !shape.size, !shape.size -> !shape.size
                    %3 = shape.meet %2, %c3, error = \"size mismatch\" : !shape.size, !shape.size -> !shape.size
                    %4 = shape.min %3, %c2 : !shape.size, !shape.size -> !shape.size
                    %5 = shape.mul %4, %c3 : !shape.size, !shape.size -> !shape.size
                    %6 = shape.size_to_index %5 : !shape.size
                    %7 = shape.index_to_size %6
                    return %7 : !shape.size
                  }
                }
            "},
        );
    }

    #[test]
    fn test_shape_construction_and_constraint_operations() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let shape_type = context.shape_type().unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();

                let size_2 = block.append_operation(const_size(2, location).unwrap()).unwrap();
                let size_2 = size_2.as_ref().result(0).unwrap().as_ref();
                let size_3 = block.append_operation(const_size(3, location).unwrap()).unwrap();
                let size_3 = size_3.as_ref().result(0).unwrap().as_ref();

                let shape_0_attribute =
                    DenseIntegerElementsAttributeRef::try_from_with_context(&[2usize, 3][..], &context).unwrap();
                let shape_0 = const_shape(shape_0_attribute, shape_type, location).unwrap();
                assert_eq!(shape_0.shape().unwrap(), shape_0_attribute);
                assert_eq!(shape_0.output_type().unwrap(), shape_type);
                let shape_0 = block.append_operation(shape_0).unwrap();
                let shape_0 = shape_0.as_ref().result(0).unwrap().as_ref();

                let shape_1_attribute =
                    DenseIntegerElementsAttributeRef::try_from_with_context(&[1usize, 3][..], &context).unwrap();
                let shape_1 = const_shape(shape_1_attribute, shape_type, location).unwrap();
                assert_eq!(shape_1.shape().unwrap(), shape_1_attribute);
                let shape_1 = block.append_operation(shape_1).unwrap();
                let shape_1 = shape_1.as_ref().result(0).unwrap().as_ref();

                let from_extents_op = from_extents(&[size_2, size_3], location).unwrap();
                assert_eq!(from_extents_op.extents().collect::<Result<Vec<_>, _>>().unwrap(), vec![size_2, size_3]);
                let from_extents_op = block.append_operation(from_extents_op).unwrap();
                let from_extents_value = from_extents_op.as_ref().result(0).unwrap().as_ref();

                let broadcast_error = context.string_attribute("cannot broadcast");
                let broadcast_op =
                    broadcast(&[shape_0, from_extents_value], shape_type, Some(broadcast_error), location).unwrap();
                assert_eq!(
                    broadcast_op.shapes().collect::<Result<Vec<_>, _>>().unwrap(),
                    vec![shape_0, from_extents_value],
                );
                assert_eq!(broadcast_op.error().unwrap().unwrap().string().as_str().unwrap(), "cannot broadcast");
                let broadcast_op = block.append_operation(broadcast_op).unwrap();
                let broadcast_value = broadcast_op.as_ref().result(0).unwrap().as_ref();

                let shape_eq_op = shape_eq(&[broadcast_value, shape_1], location).unwrap();
                assert_eq!(
                    shape_eq_op.shapes().collect::<Result<Vec<_>, _>>().unwrap(),
                    vec![broadcast_value, shape_1],
                );
                let shape_eq_op = block.append_operation(shape_eq_op).unwrap();
                let shape_eq_value = shape_eq_op.as_ref().result(0).unwrap().as_ref();

                let is_broadcastable_op = is_broadcastable(&[shape_0, shape_1], location).unwrap();
                assert_eq!(
                    is_broadcastable_op.shapes().collect::<Result<Vec<_>, _>>().unwrap(),
                    vec![shape_0, shape_1],
                );
                let is_broadcastable_op = block.append_operation(is_broadcastable_op).unwrap();
                let is_broadcastable_value = is_broadcastable_op.as_ref().result(0).unwrap().as_ref();

                let concat_op = concat(shape_0, shape_1, shape_type, location).unwrap();
                assert_eq!(concat_op.lhs().unwrap(), shape_0);
                assert_eq!(concat_op.rhs().unwrap(), shape_1);
                let concat_op = block.append_operation(concat_op).unwrap();
                let concat_value = concat_op.as_ref().result(0).unwrap().as_ref();

                let any_op = any(&[shape_0, shape_1, concat_value], shape_type, location).unwrap();
                assert_eq!(
                    any_op.inputs().collect::<Result<Vec<_>, _>>().unwrap(),
                    vec![shape_0, shape_1, concat_value],
                );
                let any_op = block.append_operation(any_op).unwrap();
                let any_value = any_op.as_ref().result(0).unwrap().as_ref();

                let assuming_all_inputs = {
                    let cstr_broadcastable_op = cstr_broadcastable(&[shape_0, shape_1], location).unwrap();
                    assert_eq!(
                        cstr_broadcastable_op.shapes().collect::<Result<Vec<_>, _>>().unwrap(),
                        vec![shape_0, shape_1],
                    );
                    let cstr_broadcastable_op = block.append_operation(cstr_broadcastable_op).unwrap();
                    let cstr_broadcastable_value = cstr_broadcastable_op.as_ref().result(0).unwrap().as_ref();

                    let cstr_eq_op = cstr_eq(&[shape_0, shape_1, from_extents_value], location).unwrap();
                    assert_eq!(
                        cstr_eq_op.shapes().collect::<Result<Vec<_>, _>>().unwrap(),
                        vec![shape_0, shape_1, from_extents_value],
                    );
                    let cstr_eq_op = block.append_operation(cstr_eq_op).unwrap();
                    let cstr_eq_value = cstr_eq_op.as_ref().result(0).unwrap().as_ref();

                    let const_witness_op = const_witness(true, location).unwrap();
                    assert!(const_witness_op.passing().unwrap());
                    let const_witness_op = block.append_operation(const_witness_op).unwrap();
                    let const_witness_value = const_witness_op.as_ref().result(0).unwrap().as_ref();

                    vec![cstr_broadcastable_value, cstr_eq_value, const_witness_value]
                };

                let assuming_all_op = assuming_all(&assuming_all_inputs, location).unwrap();
                assert_eq!(assuming_all_op.inputs().collect::<Result<Vec<_>, _>>().unwrap(), assuming_all_inputs);
                let assuming_all_op = block.append_operation(assuming_all_op).unwrap();
                let assuming_all_value = assuming_all_op.as_ref().result(0).unwrap().as_ref();

                let cstr_require_op =
                    cstr_require(shape_eq_value, context.string_attribute("shapes must match"), location).unwrap();
                assert_eq!(cstr_require_op.predicate().unwrap(), shape_eq_value);
                assert_eq!(cstr_require_op.message().unwrap().string().as_str().unwrap(), "shapes must match");
                let cstr_require_op = block.append_operation(cstr_require_op).unwrap();
                let cstr_require_value = cstr_require_op.as_ref().result(0).unwrap().as_ref();

                block
                    .append_operation(
                        func_dialect::r#return(
                            &[is_broadcastable_value, any_value, assuming_all_value, cstr_require_value],
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                func_dialect::func(
                    "shape_constraints_test",
                    func_dialect::FuncAttributes {
                        results: vec![
                            context.signless_integer_type(1).into(),
                            shape_type.into(),
                            context.shape_witness_type().unwrap().as_ref().into(),
                            context.shape_witness_type().unwrap().as_ref().into(),
                        ],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap(), "{module}");
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @shape_constraints_test() -> (i1, !shape.shape, !shape.witness, !shape.witness) {
                    %c2 = shape.const_size 2
                    %c3 = shape.const_size 3
                    %0 = shape.const_shape [2, 3] : !shape.shape
                    %1 = shape.const_shape [1, 3] : !shape.shape
                    %2 = shape.from_extents %c2, %c3 : !shape.size, !shape.size
                    %3 = shape.broadcast %0, %2 {error = \"cannot broadcast\"} : !shape.shape, !shape.shape -> !shape.shape
                    %4 = shape.shape_eq %3, %1 : !shape.shape, !shape.shape
                    %5 = shape.is_broadcastable %0, %1 : !shape.shape, !shape.shape
                    %6 = shape.concat %0, %1 : !shape.shape, !shape.shape -> !shape.shape
                    %7 = shape.any %0, %1, %6 : !shape.shape, !shape.shape, !shape.shape -> !shape.shape
                    %8 = shape.cstr_broadcastable %0, %1 : !shape.shape, !shape.shape
                    %9 = shape.cstr_eq %0, %1, %2 : !shape.shape, !shape.shape, !shape.shape
                    %10 = shape.const_witness true
                    %11 = shape.assuming_all %8, %9, %10
                    %12 = shape.cstr_require %4, \"shapes must match\"
                    return %5, %7, %11, %12 : i1, !shape.shape, !shape.witness, !shape.witness
                  }
                }
            "},
        );
    }

    #[test]
    fn test_extent_tensor_and_value_operations() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let shape_type = context.shape_type().unwrap();
        let size_type = context.shape_size_type().unwrap();
        let value_shape_type = context.shape_value_shape_type().unwrap();
        let extent_tensor_type = context.tensor_type(index_type, &[Size::Dynamic], None, location).unwrap();
        let shaped_tensor_type = context.tensor_type(index_type, &[Size::Static(2)], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (extent_tensor_type.as_ref(), location),
                    (shaped_tensor_type.as_ref(), location),
                    (index_type.as_ref(), location),
                    (value_shape_type.as_ref(), location),
                ]);
                let extent_tensor = block.argument(0).unwrap();
                let tensor_value = block.argument(1).unwrap();
                let index = block.argument(2).unwrap();
                let value_shape = block.argument(3).unwrap();

                let from_extent_tensor_op = from_extent_tensor(extent_tensor, location).unwrap();
                assert_eq!(from_extent_tensor_op.input().unwrap(), extent_tensor);
                let from_extent_tensor_op = block.append_operation(from_extent_tensor_op).unwrap();
                let from_extent_tensor_value = from_extent_tensor_op.as_ref().result(0).unwrap().as_ref();

                let rank_op = rank(from_extent_tensor_value, size_type, location).unwrap();
                assert_eq!(rank_op.shape().unwrap(), from_extent_tensor_value);
                let rank_op = block.append_operation(rank_op).unwrap();
                let rank_value = rank_op.as_ref().result(0).unwrap().as_ref();

                let to_extent_tensor_op =
                    to_extent_tensor(from_extent_tensor_value, extent_tensor_type, location).unwrap();
                assert_eq!(to_extent_tensor_op.input().unwrap(), from_extent_tensor_value);
                let to_extent_tensor_op = block.append_operation(to_extent_tensor_op).unwrap();
                let to_extent_tensor_value = to_extent_tensor_op.as_ref().result(0).unwrap().as_ref();

                let dim_op = dim(tensor_value, index, size_type, location).unwrap();
                assert_eq!(dim_op.value().unwrap(), tensor_value);
                assert_eq!(dim_op.index().unwrap(), index);
                let dim_op = block.append_operation(dim_op).unwrap();
                let dim_value = dim_op.as_ref().result(0).unwrap().as_ref();

                let get_extent_op = get_extent(from_extent_tensor_value, index, size_type, location).unwrap();
                assert_eq!(get_extent_op.shape().unwrap(), from_extent_tensor_value);
                assert_eq!(get_extent_op.dimension().unwrap(), index);
                let get_extent_op = block.append_operation(get_extent_op).unwrap();
                let get_extent_value = get_extent_op.as_ref().result(0).unwrap().as_ref();

                let num_elements_op = num_elements(from_extent_tensor_value, size_type, location).unwrap();
                assert_eq!(num_elements_op.shape().unwrap(), from_extent_tensor_value);
                let num_elements_op = block.append_operation(num_elements_op).unwrap();
                let num_elements_value = num_elements_op.as_ref().result(0).unwrap().as_ref();

                let shape_of_op = shape_of(tensor_value, shape_type, location).unwrap();
                assert_eq!(shape_of_op.argument().unwrap(), tensor_value);
                let shape_of_op = block.append_operation(shape_of_op).unwrap();
                let shape_of_value = shape_of_op.as_ref().result(0).unwrap().as_ref();

                let value_of_op = value_of(value_shape, shaped_tensor_type, location).unwrap();
                assert_eq!(value_of_op.argument().unwrap(), value_shape);
                let value_of_op = block.append_operation(value_of_op).unwrap();
                let value_of_value = value_of_op.as_ref().result(0).unwrap().as_ref();

                let value_as_shape_op = value_as_shape(extent_tensor, shape_type, location).unwrap();
                assert_eq!(value_as_shape_op.argument().unwrap(), extent_tensor);
                let value_as_shape_op = block.append_operation(value_as_shape_op).unwrap();
                let value_as_shape_value = value_as_shape_op.as_ref().result(0).unwrap().as_ref();

                let with_shape_op = with_shape(tensor_value, shape_of_value, location).unwrap();
                assert_eq!(with_shape_op.operand_value(0).unwrap(), tensor_value);
                assert_eq!(with_shape_op.shape().unwrap(), shape_of_value);
                let with_shape_op = block.append_operation(with_shape_op).unwrap();
                let with_shape_value = with_shape_op.as_ref().result(0).unwrap().as_ref();

                let debug_print_op = debug_print(from_extent_tensor_value, location).unwrap();
                assert_eq!(debug_print_op.input().unwrap(), from_extent_tensor_value);
                let debug_print_op = block.append_operation(debug_print_op).unwrap();
                let debug_print_value = debug_print_op.as_ref().result(0).unwrap().as_ref();

                let split_at_op =
                    split_at(from_extent_tensor_value, index, [shape_type.as_ref(), shape_type.as_ref()], location)
                        .unwrap();
                assert_eq!(split_at_op.operand_value(0).unwrap(), from_extent_tensor_value);
                assert_eq!(split_at_op.index().unwrap(), index);
                assert_eq!(split_at_op.operand_value(0).unwrap(), from_extent_tensor_value);
                assert_eq!(split_at_op.index().unwrap(), index);
                let split_at_op = block.append_operation(split_at_op).unwrap();
                let split_at_head = split_at_op.as_ref().result(0).unwrap().as_ref();
                let split_at_tail = split_at_op.result(1).unwrap().as_ref();

                block
                    .append_operation(
                        func_dialect::r#return(
                            &[
                                rank_value,
                                to_extent_tensor_value,
                                dim_value,
                                get_extent_value,
                                num_elements_value,
                                value_of_value,
                                value_as_shape_value,
                                with_shape_value,
                                debug_print_value,
                                split_at_head,
                                split_at_tail,
                            ],
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                func_dialect::func(
                    "shape_extent_tensor_test",
                    func_dialect::FuncAttributes {
                        arguments: vec![
                            extent_tensor_type.into(),
                            shaped_tensor_type.into(),
                            index_type.into(),
                            value_shape_type.into(),
                        ],
                        results: vec![
                            size_type.into(),
                            extent_tensor_type.into(),
                            size_type.into(),
                            size_type.into(),
                            size_type.into(),
                            shaped_tensor_type.into(),
                            shape_type.into(),
                            value_shape_type.into(),
                            shape_type.into(),
                            shape_type.into(),
                            shape_type.into(),
                        ],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap(), "{module}");
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @shape_extent_tensor_test(%arg0: tensor<?xindex>, %arg1: tensor<2xindex>, %arg2: index, %arg3: !shape.value_shape) -> (!shape.size, tensor<?xindex>, !shape.size, !shape.size, !shape.size, tensor<2xindex>, !shape.shape, !shape.value_shape, !shape.shape, !shape.shape, !shape.shape) {
                    %0 = shape.from_extent_tensor %arg0 : tensor<?xindex>
                    %1 = shape.rank %0 : !shape.shape -> !shape.size
                    %2 = shape.to_extent_tensor %0 : !shape.shape -> tensor<?xindex>
                    %3 = shape.dim %arg1, %arg2 : tensor<2xindex>, index -> !shape.size
                    %4 = shape.get_extent %0, %arg2 : !shape.shape, index -> !shape.size
                    %5 = shape.num_elements %0 : !shape.shape -> !shape.size
                    %6 = shape.shape_of %arg1 : tensor<2xindex> -> !shape.shape
                    %7 = shape.value_of %arg3 : tensor<2xindex>
                    %8 = shape.value_as_shape %arg0 : tensor<?xindex> -> !shape.shape
                    %9 = shape.with_shape %arg1, %6 : tensor<2xindex>, !shape.shape
                    %10 = \"shape.debug_print\"(%0) : (!shape.shape) -> !shape.shape
                    %head, %tail = \"shape.split_at\"(%0, %arg2) : (!shape.shape, index) -> (!shape.shape, !shape.shape)
                    return %1, %2, %3, %4, %5, %7, %8, %9, %10, %head, %tail : !shape.size, tensor<?xindex>, !shape.size, !shape.size, !shape.size, tensor<2xindex>, !shape.shape, !shape.value_shape, !shape.shape, !shape.shape, !shape.shape
                  }
                }
            "},
        );
    }

    #[test]
    fn test_reduce_and_assuming_operations() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let shape_type = context.shape_type().unwrap();
        let size_type = context.shape_size_type().unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(shape_type.as_ref(), location)]);
                let shape = block.argument(0).unwrap();
                let initial_value = block.append_operation(const_size(1, location).unwrap()).unwrap();
                let initial_value = initial_value.as_ref().result(0).unwrap().as_ref();

                let mut reduce_block = context.block(&[
                    (index_type.as_ref(), location),
                    (size_type.as_ref(), location),
                    (size_type.as_ref(), location),
                ]);
                let dimension = reduce_block.argument(1).unwrap();
                let accumulator = reduce_block.argument(2).unwrap();
                let updated_accumulator =
                    reduce_block.append_operation(mul(dimension, accumulator, size_type, location).unwrap()).unwrap();
                let updated_accumulator = updated_accumulator.as_ref().result(0).unwrap().as_ref();
                let yield_op = r#yield(&[updated_accumulator], location).unwrap();
                assert_eq!(yield_op.values().collect::<Result<Vec<_>, _>>().unwrap(), vec![updated_accumulator]);
                reduce_block.append_operation(yield_op).unwrap();

                let reduce_op =
                    reduce(shape, &[initial_value], &[size_type.as_ref()], reduce_block.try_into().unwrap(), location)
                        .unwrap();
                assert_eq!(reduce_op.shape().unwrap(), shape);
                assert_eq!(reduce_op.initial_values().collect::<Result<Vec<_>, _>>().unwrap(), vec![initial_value],);
                assert_eq!(reduce_op.as_ref().region(0).unwrap().blocks().unwrap().count(), 1);
                let reduce_op = block.append_operation(reduce_op).unwrap();
                let reduce_value = reduce_op.as_ref().result(0).unwrap().as_ref();

                let witness = block.append_operation(const_witness(true, location).unwrap()).unwrap();
                let witness = witness.as_ref().result(0).unwrap().as_ref();
                let mut assuming_block = context.block_with_no_arguments();
                let assumed_value = assuming_block.append_operation(const_size(7, location).unwrap()).unwrap();
                let assumed_value = assumed_value.as_ref().result(0).unwrap().as_ref();
                let assuming_yield_op = assuming_yield(&[assumed_value], location).unwrap();
                assert_eq!(assuming_yield_op.values().collect::<Result<Vec<_>, _>>().unwrap(), vec![assumed_value]);
                assuming_block.append_operation(assuming_yield_op).unwrap();

                let assuming_op =
                    assuming(witness, &[size_type.as_ref()], assuming_block.try_into().unwrap(), location).unwrap();
                assert_eq!(assuming_op.witness().unwrap(), witness);
                assert_eq!(assuming_op.as_ref().region(0).unwrap().blocks().unwrap().count(), 1);
                assert_eq!(assuming_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let assuming_op = block.append_operation(assuming_op).unwrap();
                let assuming_value = assuming_op.as_ref().result(0).unwrap().as_ref();

                block
                    .append_operation(func_dialect::r#return(&[reduce_value, assuming_value], location).unwrap())
                    .unwrap();
                func_dialect::func(
                    "shape_reduce_and_assuming_test",
                    func_dialect::FuncAttributes {
                        arguments: vec![shape_type.into()],
                        results: vec![size_type.into(), size_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap(), "{module}");
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @shape_reduce_and_assuming_test(%arg0: !shape.shape) -> (!shape.size, !shape.size) {
                    %c1 = shape.const_size 1
                    %0 = shape.reduce(%arg0, %c1) : !shape.shape -> !shape.size {
                    ^bb0(%arg1: index, %arg2: !shape.size, %arg3: !shape.size):
                      %3 = shape.mul %arg2, %arg3 : !shape.size, !shape.size -> !shape.size
                      shape.yield %3 : !shape.size
                    }
                    %1 = shape.const_witness true
                    %2 = shape.assuming %1 -> (!shape.size) {
                      %c7 = shape.const_size 7
                      shape.assuming_yield %c7 : !shape.size
                    }
                    return %0, %2 : !shape.size, !shape.size
                  }
                }
            "},
        );
    }

    #[test]
    fn test_function_library_func_and_return_operations() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let shape_type = context.shape_type().unwrap();
        let value_shape_type = context.shape_value_shape_type().unwrap();

        let mut function_body = context.block(&[(value_shape_type.as_ref(), location)]);
        let value_shape = function_body.argument(0).unwrap();
        let shape_of_op = shape_of(value_shape, shape_type, location).unwrap();
        assert_eq!(shape_of_op.argument().unwrap(), value_shape);
        let shape_of_op = function_body.append_operation(shape_of_op).unwrap();
        let shape = shape_of_op.as_ref().result(0).unwrap().as_ref();
        let return_op = r#return(&[shape], location).unwrap();
        assert_eq!(return_op.values().collect::<Result<Vec<_>, _>>().unwrap(), vec![shape]);
        function_body.append_operation(return_op).unwrap();

        let shape_func = func(
            "same_result_shape",
            FuncAttributes {
                arguments: vec![value_shape_type.into()],
                results: vec![shape_type.into()],
                ..Default::default()
            },
            function_body.try_into().unwrap(),
            location,
        )
        .unwrap();
        assert_eq!(shape_func.name().as_str(), Ok("shape.func"));

        let mut library_body = context.block_with_no_arguments();
        library_body.append_operation(shape_func).unwrap();
        let mapping = context.dictionary_attribute(&[]);
        let library = function_library("shape_lib", mapping, library_body.try_into().unwrap(), location).unwrap();
        assert_eq!(library.mapping().unwrap(), mapping);
        assert_eq!(library.body_region().unwrap().blocks().unwrap().count(), 1);
        module.body().unwrap().append_operation(library).unwrap();

        assert!(module.verify().unwrap(), "{module}");
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  shape.function_library @shape_lib {
                    func @same_result_shape(%arg0: !shape.value_shape) -> !shape.shape {
                      %0 = shape_of %arg0 : !shape.value_shape -> !shape.shape
                      return %0 : !shape.shape
                    }
                  } mapping {}
                }
            "},
        );
    }
}
