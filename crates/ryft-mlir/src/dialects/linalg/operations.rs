use ryft_xla_sys::bindings::mlirLinalgFillBuiltinNamedOpRegion;

use crate::{
    ArrayAttributeRef, Attribute, AttributeRef, DenseInteger32ArrayAttributeRef, DenseInteger64ArrayAttributeRef,
    DenseIntegerElementsAttributeRef, DetachedOp, DetachedRegion, DialectHandle, Location, Operation, OperationBuilder,
    RegionRef, StringAttributeRef, TypeRef, Value, ValueRef, mlir_op, mlir_op_trait,
};

use super::{
    ElementwiseKind, ElementwiseKindAttributeRef, TypeFn, TypeFnAttributeRef, WinogradConv2DFmr,
    WinogradConv2DFmrAttributeRef,
};

/// Name of the attribute storing operand segment sizes for variadic Linalg operations.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Common API for Linalg named structured operations.
pub trait LinalgNamedStructuredOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns this operation's input operands.
    fn inputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        (0..segment_sizes[0]).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns this operation's output operands.
    fn outputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        let start = segment_sizes[0];
        let end = start + segment_sizes[1];
        (start..end).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns this operation's result tensors.
    fn result_tensors(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (0..self.result_count()).map(|index| self.result(index).unwrap().as_ref()).collect()
    }

    /// Returns this operation's payload region.
    fn region(&self) -> RegionRef<'o, 'c, 't> {
        Operation::region(self, 0).unwrap()
    }
}

/// Operation trait for `linalg.yield`.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the values yielded to the enclosing Linalg operation.
    fn values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (0..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, ReturnLike);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Constructs a new detached/owned [`YieldOperation`] at the specified [`Location`].
pub fn r#yield<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedYieldOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::linalg());
    OperationBuilder::new("linalg.yield", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::yield`")
}

/// Name of the Linalg index dimension attribute.
pub const DIM_ATTRIBUTE: &str = "dim";

/// Operation trait for `linalg.index`.
pub trait IndexOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the accessed loop dimension.
    fn dimension(&self) -> i64 {
        self.attribute(DIM_ATTRIBUTE)
            .unwrap()
            .cast::<crate::IntegerAttributeRef>()
            .unwrap()
            .signless_value()
    }
}

mlir_op!(Index);
mlir_op_trait!(Index, OneResult);
mlir_op_trait!(Index, ZeroOperands);
mlir_op_trait!(Index, ZeroRegions);
mlir_op_trait!(Index, ZeroSuccessors);

/// Constructs a new detached/owned [`IndexOperation`] at the specified [`Location`].
pub fn index<'c, 't: 'c, L: Location<'c, 't>>(dimension: i64, location: L) -> DetachedIndexOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    OperationBuilder::new("linalg.index", location)
        .add_attribute(DIM_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), dimension))
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::index`")
}

/// Name of the Linalg indexing maps attribute.
pub const INDEXING_MAPS_ATTRIBUTE: &str = "indexing_maps";

/// Name of the Linalg iterator types attribute.
pub const ITERATOR_TYPES_ATTRIBUTE: &str = "iterator_types";

/// Name of the optional Linalg documentation attribute.
pub const DOC_ATTRIBUTE: &str = "doc";

/// Name of the optional Linalg library call attribute.
pub const LIBRARY_CALL_ATTRIBUTE: &str = "library_call";

/// Operation trait for `linalg.generic`.
pub trait GenericOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns this operation's input operands.
    fn inputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        (0..segment_sizes[0]).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns this operation's output operands.
    fn outputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        let start = segment_sizes[0];
        let end = start + segment_sizes[1];
        (start..end).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns this operation's indexing maps.
    fn indexing_maps(&self) -> ArrayAttributeRef<'c, 't> {
        self.attribute(INDEXING_MAPS_ATTRIBUTE).unwrap().cast().unwrap()
    }

    /// Returns this operation's iterator types.
    fn iterator_types(&self) -> ArrayAttributeRef<'c, 't> {
        self.attribute(ITERATOR_TYPES_ATTRIBUTE).unwrap().cast().unwrap()
    }

    /// Returns the optional documentation attribute.
    fn doc(&self) -> Option<StringAttributeRef<'c, 't>> {
        self.attribute(DOC_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the optional external library call attribute.
    fn library_call(&self) -> Option<StringAttributeRef<'c, 't>> {
        self.attribute(LIBRARY_CALL_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }
}

mlir_op!(Generic);
mlir_op_trait!(Generic, OneRegion);
mlir_op_trait!(Generic, ZeroSuccessors);

/// Constructs a new detached/owned [`GenericOperation`] at the specified [`Location`].
pub fn generic<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    indexing_maps: ArrayAttributeRef<'c, 't>,
    iterator_types: ArrayAttributeRef<'c, 't>,
    doc: Option<StringAttributeRef<'c, 't>>,
    library_call: Option<StringAttributeRef<'c, 't>>,
    region: DetachedRegion<'c, 't>,
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedGenericOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.generic", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_attribute(INDEXING_MAPS_ATTRIBUTE, indexing_maps)
        .add_attribute(ITERATOR_TYPES_ATTRIBUTE, iterator_types)
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(region);
    if let Some(doc) = doc {
        builder = builder.add_attribute(DOC_ATTRIBUTE, doc);
    }
    if let Some(library_call) = library_call {
        builder = builder.add_attribute(LIBRARY_CALL_ATTRIBUTE, library_call);
    }
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::generic`")
}

/// Operation trait for `linalg.map`.
pub trait MapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns this operation's inputs.
    fn inputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (0..self.operand_count() - 1).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns this operation's initial output.
    fn init(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(self.operand_count() - 1).unwrap()
    }
}

mlir_op!(Map);
mlir_op_trait!(Map, OneRegion);
mlir_op_trait!(Map, ZeroSuccessors);

/// Constructs a new detached/owned [`MapOperation`] at the specified [`Location`].
pub fn map<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    init: ValueRef<'v, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    region: DetachedRegion<'c, 't>,
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedMapOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::linalg());
    let mut builder = OperationBuilder::new("linalg.map", location)
        .add_operands(inputs)
        .add_operand(init)
        .add_results(result_types)
        .add_region(region);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::map`")
}

/// Name of the Linalg dimensions attribute.
pub const DIMENSIONS_ATTRIBUTE: &str = "dimensions";

/// Operation trait for `linalg.reduce`.
pub trait ReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns this operation's inputs.
    fn inputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let half = self.operand_count() / 2;
        (0..half).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns this operation's initial values.
    fn inits(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let half = self.operand_count() / 2;
        (half..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns this operation's reduction dimensions.
    fn dimensions(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(DIMENSIONS_ATTRIBUTE).unwrap().cast().unwrap()
    }
}

mlir_op!(Reduce);
mlir_op_trait!(Reduce, OneRegion);
mlir_op_trait!(Reduce, ZeroSuccessors);

/// Constructs a new detached/owned [`ReduceOperation`] at the specified [`Location`].
pub fn reduce<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    inits: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    dimensions: &[i64],
    region: DetachedRegion<'c, 't>,
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedReduceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let mut builder = OperationBuilder::new("linalg.reduce", location)
        .add_attribute(DIMENSIONS_ATTRIBUTE, context.dense_i64_array_attribute(dimensions).unwrap())
        .add_operands(inputs)
        .add_operands(inits)
        .add_results(result_types)
        .add_region(region);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::reduce`")
}

/// Name of the Linalg permutation attribute.
pub const PERMUTATION_ATTRIBUTE: &str = "permutation";

/// Operation trait for `linalg.transpose`.
pub trait TransposeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the initial output operand.
    fn init(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the permutation attribute.
    fn permutation(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(PERMUTATION_ATTRIBUTE).unwrap().cast().unwrap()
    }
}

mlir_op!(Transpose);
mlir_op_trait!(Transpose, OneRegion);
mlir_op_trait!(Transpose, ZeroSuccessors);

/// Constructs a new detached/owned [`TransposeOperation`] at the specified [`Location`].
pub fn transpose<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    init: ValueRef<'v, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    permutation: &[i64],
    region: DetachedRegion<'c, 't>,
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedTransposeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let mut builder = OperationBuilder::new("linalg.transpose", location)
        .add_attribute(PERMUTATION_ATTRIBUTE, context.dense_i64_array_attribute(permutation).unwrap())
        .add_operands(&[input, init])
        .add_results(result_types)
        .add_region(region);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::transpose`")
}

/// Operation trait for `linalg.broadcast`.
pub trait BroadcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the initial output operand.
    fn init(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the broadcast dimensions.
    fn dimensions(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(DIMENSIONS_ATTRIBUTE).unwrap().cast().unwrap()
    }
}

mlir_op!(Broadcast);
mlir_op_trait!(Broadcast, OneRegion);
mlir_op_trait!(Broadcast, ZeroSuccessors);

/// Constructs a new detached/owned [`BroadcastOperation`] at the specified [`Location`].
pub fn broadcast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    init: ValueRef<'v, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    dimensions: &[i64],
    region: DetachedRegion<'c, 't>,
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedBroadcastOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let mut builder = OperationBuilder::new("linalg.broadcast", location)
        .add_attribute(DIMENSIONS_ATTRIBUTE, context.dense_i64_array_attribute(dimensions).unwrap())
        .add_operands(&[input, init])
        .add_results(result_types)
        .add_region(region);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::broadcast`")
}

/// Name of the Linalg elementwise kind attribute.
pub const KIND_ATTRIBUTE: &str = "kind";

/// Operation trait for `linalg.elementwise`.
pub trait ElementwiseOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {
    /// Returns the elementwise operation kind.
    fn kind(&self) -> ElementwiseKind {
        self.attribute(KIND_ATTRIBUTE).unwrap().cast::<ElementwiseKindAttributeRef>().unwrap().value()
    }

    /// Returns the optional explicit indexing maps.
    fn indexing_maps(&self) -> Option<ArrayAttributeRef<'c, 't>> {
        self.attribute(INDEXING_MAPS_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }
}

mlir_op!(Elementwise);
mlir_op_trait!(Elementwise, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Elementwise, OneRegion);
mlir_op_trait!(Elementwise, ZeroSuccessors);

/// Constructs a new detached/owned [`ElementwiseOperation`] at the specified [`Location`].
pub fn elementwise<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    kind: ElementwiseKind,
    indexing_maps: Option<ArrayAttributeRef<'c, 't>>,
    region: DetachedRegion<'c, 't>,
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedElementwiseOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.elementwise", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_attribute(KIND_ATTRIBUTE, context.linalg_elementwise_kind_attribute(kind))
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(region);
    if let Some(indexing_maps) = indexing_maps {
        builder = builder.add_attribute(INDEXING_MAPS_ATTRIBUTE, indexing_maps);
    }
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::elementwise`")
}

/// Name of the Linalg numeric cast attribute.
pub const CAST_ATTRIBUTE: &str = "cast";

/// Common API for Linalg operations with numeric cast behavior.
pub trait LinalgCastedOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the numeric cast behavior.
    fn cast(&self) -> TypeFn {
        self.attribute(CAST_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeFnAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or(TypeFn::CastSigned)
    }
}

/// Operation trait for Linalg core structured operations with `inputs` and `outputs`.
pub trait LinalgCoreStructuredOperation<'o, 'c: 'o, 't: 'c>:
    LinalgNamedStructuredOperation<'o, 'c, 't> + LinalgCastedOperation<'o, 'c, 't>
{
    /// Returns the optional explicit indexing maps.
    fn indexing_maps(&self) -> Option<ArrayAttributeRef<'c, 't>> {
        self.attribute(INDEXING_MAPS_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }
}

/// Operation trait for `linalg.matmul`.
pub trait MatmulOperation<'o, 'c: 'o, 't: 'c>: LinalgCoreStructuredOperation<'o, 'c, 't> {}

mlir_op!(Matmul);
mlir_op_trait!(Matmul, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Matmul, @local LinalgCastedOperation);
mlir_op_trait!(Matmul, @local LinalgCoreStructuredOperation);
mlir_op_trait!(Matmul, OneRegion);
mlir_op_trait!(Matmul, ZeroSuccessors);

/// Constructs a new detached/owned [`MatmulOperation`] at the specified [`Location`].
pub fn matmul<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    region: DetachedRegion<'c, 't>,
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedMatmulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.matmul", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(region);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::matmul`")
}

/// Operation trait for `linalg.contract`.
pub trait ContractOperation<'o, 'c: 'o, 't: 'c>: LinalgCoreStructuredOperation<'o, 'c, 't> {}

mlir_op!(Contract);
mlir_op_trait!(Contract, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Contract, @local LinalgCastedOperation);
mlir_op_trait!(Contract, @local LinalgCoreStructuredOperation);
mlir_op_trait!(Contract, OneRegion);
mlir_op_trait!(Contract, ZeroSuccessors);

/// Constructs a new detached/owned [`ContractOperation`] at the specified [`Location`].
pub fn contract<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    region: DetachedRegion<'c, 't>,
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedContractOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.contract", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(region);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::contract`")
}

/// Operation trait for `linalg.batch_matmul`.
pub trait BatchMatmulOperation<'o, 'c: 'o, 't: 'c>: LinalgCoreStructuredOperation<'o, 'c, 't> {}

mlir_op!(BatchMatmul);
mlir_op_trait!(BatchMatmul, @local LinalgNamedStructuredOperation);
mlir_op_trait!(BatchMatmul, @local LinalgCastedOperation);
mlir_op_trait!(BatchMatmul, @local LinalgCoreStructuredOperation);
mlir_op_trait!(BatchMatmul, OneRegion);
mlir_op_trait!(BatchMatmul, ZeroSuccessors);

/// Constructs a new detached/owned [`BatchMatmulOperation`] at the specified [`Location`].
pub fn batch_matmul<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    region: DetachedRegion<'c, 't>,
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedBatchMatmulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.batch_matmul", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(region);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::batch_matmul`")
}

/// Operation trait for `linalg.batch_reduce_matmul`.
pub trait BatchReduceMatmulOperation<'o, 'c: 'o, 't: 'c>: LinalgCoreStructuredOperation<'o, 'c, 't> {}

mlir_op!(BatchReduceMatmul);
mlir_op_trait!(BatchReduceMatmul, @local LinalgNamedStructuredOperation);
mlir_op_trait!(BatchReduceMatmul, @local LinalgCastedOperation);
mlir_op_trait!(BatchReduceMatmul, @local LinalgCoreStructuredOperation);
mlir_op_trait!(BatchReduceMatmul, OneRegion);
mlir_op_trait!(BatchReduceMatmul, ZeroSuccessors);

/// Constructs a new detached/owned [`BatchReduceMatmulOperation`] at the specified [`Location`].
pub fn batch_reduce_matmul<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    region: DetachedRegion<'c, 't>,
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedBatchReduceMatmulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.batch_reduce_matmul", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(region);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::batch_reduce_matmul`")
}

/// Name of the Linalg dimension attribute.
pub const DIMENSION_ATTRIBUTE: &str = "dimension";

/// Operation trait for `linalg.softmax`.
pub trait SoftmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the output operand.
    fn output_operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the softmax dimension.
    fn dimension(&self) -> i64 {
        self.attribute(DIMENSION_ATTRIBUTE)
            .unwrap()
            .cast::<crate::IntegerAttributeRef>()
            .unwrap()
            .signless_value()
    }
}

mlir_op!(Softmax);
mlir_op_trait!(Softmax, ZeroRegions);
mlir_op_trait!(Softmax, ZeroSuccessors);

/// Constructs a new detached/owned [`SoftmaxOperation`] at the specified [`Location`].
pub fn softmax<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    output: ValueRef<'v, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    dimension: i64,
    location: L,
) -> DetachedSoftmaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    OperationBuilder::new("linalg.softmax", location)
        .add_attribute(DIMENSION_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), dimension))
        .add_operands(&[input, output])
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::softmax`")
}

/// Name of the Linalg Winograd FMR attribute.
pub const FMR_ATTRIBUTE: &str = "fmr";

/// Common API for Linalg Winograd transform operations.
pub trait LinalgWinogradTransformOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the transform input operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the output operand.
    fn output_operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the Winograd FMR value.
    fn fmr(&self) -> WinogradConv2DFmr {
        self.attribute(FMR_ATTRIBUTE).unwrap().cast::<WinogradConv2DFmrAttributeRef>().unwrap().value()
    }
}

/// Operation trait for `linalg.winograd_filter_transform`.
pub trait WinogradFilterTransformOperation<'o, 'c: 'o, 't: 'c>: LinalgWinogradTransformOperation<'o, 'c, 't> {}

mlir_op!(WinogradFilterTransform);
mlir_op_trait!(WinogradFilterTransform, @local LinalgWinogradTransformOperation);
mlir_op_trait!(WinogradFilterTransform, ZeroRegions);
mlir_op_trait!(WinogradFilterTransform, ZeroSuccessors);

/// Constructs a new detached/owned [`WinogradFilterTransformOperation`] at the specified [`Location`].
pub fn winograd_filter_transform<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    output: ValueRef<'v, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    fmr: WinogradConv2DFmr,
    location: L,
) -> DetachedWinogradFilterTransformOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    OperationBuilder::new("linalg.winograd_filter_transform", location)
        .add_attribute(FMR_ATTRIBUTE, context.linalg_winograd_conv_2d_fmr_attribute(fmr))
        .add_operands(&[input, output])
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::winograd_filter_transform`")
}

/// Operation trait for `linalg.winograd_input_transform`.
pub trait WinogradInputTransformOperation<'o, 'c: 'o, 't: 'c>: LinalgWinogradTransformOperation<'o, 'c, 't> {}

mlir_op!(WinogradInputTransform);
mlir_op_trait!(WinogradInputTransform, @local LinalgWinogradTransformOperation);
mlir_op_trait!(WinogradInputTransform, ZeroRegions);
mlir_op_trait!(WinogradInputTransform, ZeroSuccessors);

/// Constructs a new detached/owned [`WinogradInputTransformOperation`] at the specified [`Location`].
pub fn winograd_input_transform<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    output: ValueRef<'v, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    fmr: WinogradConv2DFmr,
    location: L,
) -> DetachedWinogradInputTransformOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    OperationBuilder::new("linalg.winograd_input_transform", location)
        .add_attribute(FMR_ATTRIBUTE, context.linalg_winograd_conv_2d_fmr_attribute(fmr))
        .add_operands(&[input, output])
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::winograd_input_transform`")
}

/// Operation trait for `linalg.winograd_output_transform`.
pub trait WinogradOutputTransformOperation<'o, 'c: 'o, 't: 'c>: LinalgWinogradTransformOperation<'o, 'c, 't> {}

mlir_op!(WinogradOutputTransform);
mlir_op_trait!(WinogradOutputTransform, @local LinalgWinogradTransformOperation);
mlir_op_trait!(WinogradOutputTransform, ZeroRegions);
mlir_op_trait!(WinogradOutputTransform, ZeroSuccessors);

/// Constructs a new detached/owned [`WinogradOutputTransformOperation`] at the specified [`Location`].
pub fn winograd_output_transform<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    output: ValueRef<'v, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    fmr: WinogradConv2DFmr,
    location: L,
) -> DetachedWinogradOutputTransformOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    OperationBuilder::new("linalg.winograd_output_transform", location)
        .add_attribute(FMR_ATTRIBUTE, context.linalg_winograd_conv_2d_fmr_attribute(fmr))
        .add_operands(&[input, output])
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::winograd_output_transform`")
}

/// Name of the Linalg outer-dimension permutation attribute.
pub const OUTER_DIMS_PERM_ATTRIBUTE: &str = "outer_dims_perm";

/// Name of the Linalg inner-dimension positions attribute.
pub const INNER_DIMS_POS_ATTRIBUTE: &str = "inner_dims_pos";

/// Name of the Linalg static inner tiles attribute.
pub const STATIC_INNER_TILES_ATTRIBUTE: &str = "static_inner_tiles";

/// Operation trait for `linalg.pack`.
pub trait PackOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination operand.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional padding value.
    fn padding_value(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        let start = segment_sizes.iter().take(2).sum::<usize>();
        (start..start + segment_sizes[2]).map(|index| self.operand_value(index).unwrap()).next()
    }

    /// Returns the dynamic inner tile operands.
    fn inner_tiles(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        let start = segment_sizes.iter().take(3).sum::<usize>();
        let end = start + segment_sizes[3];
        (start..end).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the outer-dimension permutation attribute.
    fn outer_dims_perm(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(OUTER_DIMS_PERM_ATTRIBUTE).unwrap().cast().unwrap()
    }

    /// Returns the inner-dimension positions attribute.
    fn inner_dims_pos(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(INNER_DIMS_POS_ATTRIBUTE).unwrap().cast().unwrap()
    }

    /// Returns the static inner tile sizes attribute.
    fn static_inner_tiles(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(STATIC_INNER_TILES_ATTRIBUTE).unwrap().cast().unwrap()
    }
}

mlir_op!(Pack);
mlir_op_trait!(Pack, ZeroRegions);
mlir_op_trait!(Pack, ZeroSuccessors);

/// Constructs a new detached/owned [`PackOperation`] at the specified [`Location`].
pub fn pack<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    padding_value: Option<ValueRef<'v, 'c, 't>>,
    inner_tiles: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    outer_dims_perm: &[i64],
    inner_dims_pos: &[i64],
    static_inner_tiles: &[i64],
    location: L,
) -> DetachedPackOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [1, 1, i32::from(padding_value.is_some()), inner_tiles.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pack", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_attribute(OUTER_DIMS_PERM_ATTRIBUTE, context.dense_i64_array_attribute(outer_dims_perm).unwrap())
        .add_attribute(INNER_DIMS_POS_ATTRIBUTE, context.dense_i64_array_attribute(inner_dims_pos).unwrap())
        .add_attribute(STATIC_INNER_TILES_ATTRIBUTE, context.dense_i64_array_attribute(static_inner_tiles).unwrap())
        .add_operands(&[source, destination]);
    if let Some(padding_value) = padding_value {
        builder = builder.add_operand(padding_value);
    }
    builder
        .add_operands(inner_tiles)
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::pack`")
}

/// Operation trait for `linalg.unpack`.
pub trait UnpackOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination operand.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the dynamic inner tile operands.
    fn inner_tiles(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute"))
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        let start = segment_sizes.iter().take(2).sum::<usize>();
        let end = start + segment_sizes[2];
        (start..end).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the outer-dimension permutation attribute.
    fn outer_dims_perm(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(OUTER_DIMS_PERM_ATTRIBUTE).unwrap().cast().unwrap()
    }

    /// Returns the inner-dimension positions attribute.
    fn inner_dims_pos(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(INNER_DIMS_POS_ATTRIBUTE).unwrap().cast().unwrap()
    }

    /// Returns the static inner tile sizes attribute.
    fn static_inner_tiles(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(STATIC_INNER_TILES_ATTRIBUTE).unwrap().cast().unwrap()
    }
}

mlir_op!(Unpack);
mlir_op_trait!(Unpack, ZeroRegions);
mlir_op_trait!(Unpack, ZeroSuccessors);

/// Constructs a new detached/owned [`UnpackOperation`] at the specified [`Location`].
pub fn unpack<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    inner_tiles: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    outer_dims_perm: &[i64],
    inner_dims_pos: &[i64],
    static_inner_tiles: &[i64],
    location: L,
) -> DetachedUnpackOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [1, 1, inner_tiles.len() as i32];
    OperationBuilder::new("linalg.unpack", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_attribute(OUTER_DIMS_PERM_ATTRIBUTE, context.dense_i64_array_attribute(outer_dims_perm).unwrap())
        .add_attribute(INNER_DIMS_POS_ATTRIBUTE, context.dense_i64_array_attribute(inner_dims_pos).unwrap())
        .add_attribute(STATIC_INNER_TILES_ATTRIBUTE, context.dense_i64_array_attribute(static_inner_tiles).unwrap())
        .add_operands(&[source, destination])
        .add_operands(inner_tiles)
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `linalg::unpack`")
}

/// Operation trait for `linalg.copy`.
pub trait CopyOperation<'o, 'c: 'o, 't: 'c>:
    LinalgNamedStructuredOperation<'o, 'c, 't> + LinalgCastedOperation<'o, 'c, 't>
{
}

mlir_op!(Copy);
mlir_op_trait!(Copy, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Copy, @local LinalgCastedOperation);
mlir_op_trait!(Copy, OneRegion);
mlir_op_trait!(Copy, ZeroSuccessors);

/// Constructs a new detached/owned [`CopyOperation`] at the specified [`Location`].
pub fn copy<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedCopyOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.copy", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::copy`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::copy` operation cast")
}

/// Operation trait for `linalg.exp`.
pub trait ExpOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Exp);
mlir_op_trait!(Exp, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Exp, OneRegion);
mlir_op_trait!(Exp, ZeroSuccessors);

/// Constructs a new detached/owned [`ExpOperation`] at the specified [`Location`].
pub fn exp<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedExpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.exp", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::exp`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::exp` operation cast")
}

/// Operation trait for `linalg.log`.
pub trait LogOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Log);
mlir_op_trait!(Log, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Log, OneRegion);
mlir_op_trait!(Log, ZeroSuccessors);

/// Constructs a new detached/owned [`LogOperation`] at the specified [`Location`].
pub fn log<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedLogOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.log", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::log`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::log` operation cast")
}

/// Operation trait for `linalg.abs`.
pub trait AbsOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Abs);
mlir_op_trait!(Abs, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Abs, OneRegion);
mlir_op_trait!(Abs, ZeroSuccessors);

/// Constructs a new detached/owned [`AbsOperation`] at the specified [`Location`].
pub fn abs<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedAbsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.abs", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::abs`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::abs` operation cast")
}

/// Operation trait for `linalg.ceil`.
pub trait CeilOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Ceil);
mlir_op_trait!(Ceil, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Ceil, OneRegion);
mlir_op_trait!(Ceil, ZeroSuccessors);

/// Constructs a new detached/owned [`CeilOperation`] at the specified [`Location`].
pub fn ceil<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedCeilOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.ceil", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::ceil`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::ceil` operation cast")
}

/// Operation trait for `linalg.floor`.
pub trait FloorOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Floor);
mlir_op_trait!(Floor, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Floor, OneRegion);
mlir_op_trait!(Floor, ZeroSuccessors);

/// Constructs a new detached/owned [`FloorOperation`] at the specified [`Location`].
pub fn floor<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedFloorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.floor", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::floor`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::floor` operation cast")
}

/// Operation trait for `linalg.negf`.
pub trait NegFOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(NegF);
mlir_op_trait!(NegF, @local LinalgNamedStructuredOperation);
mlir_op_trait!(NegF, OneRegion);
mlir_op_trait!(NegF, ZeroSuccessors);

/// Constructs a new detached/owned [`NegFOperation`] at the specified [`Location`].
pub fn negf<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedNegFOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.negf", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::negf`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::negf` operation cast")
}

/// Operation trait for `linalg.reciprocal`.
pub trait ReciprocalOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Reciprocal);
mlir_op_trait!(Reciprocal, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Reciprocal, OneRegion);
mlir_op_trait!(Reciprocal, ZeroSuccessors);

/// Constructs a new detached/owned [`ReciprocalOperation`] at the specified [`Location`].
pub fn reciprocal<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedReciprocalOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.reciprocal", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::reciprocal`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::reciprocal` operation cast")
}

/// Operation trait for `linalg.round`.
pub trait RoundOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Round);
mlir_op_trait!(Round, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Round, OneRegion);
mlir_op_trait!(Round, ZeroSuccessors);

/// Constructs a new detached/owned [`RoundOperation`] at the specified [`Location`].
pub fn round<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedRoundOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.round", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::round`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::round` operation cast")
}

/// Operation trait for `linalg.sqrt`.
pub trait SqrtOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Sqrt);
mlir_op_trait!(Sqrt, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Sqrt, OneRegion);
mlir_op_trait!(Sqrt, ZeroSuccessors);

/// Constructs a new detached/owned [`SqrtOperation`] at the specified [`Location`].
pub fn sqrt<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedSqrtOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.sqrt", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::sqrt`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::sqrt` operation cast")
}

/// Operation trait for `linalg.rsqrt`.
pub trait RsqrtOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Rsqrt);
mlir_op_trait!(Rsqrt, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Rsqrt, OneRegion);
mlir_op_trait!(Rsqrt, ZeroSuccessors);

/// Constructs a new detached/owned [`RsqrtOperation`] at the specified [`Location`].
pub fn rsqrt<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedRsqrtOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.rsqrt", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::rsqrt`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::rsqrt` operation cast")
}

/// Operation trait for `linalg.square`.
pub trait SquareOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Square);
mlir_op_trait!(Square, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Square, OneRegion);
mlir_op_trait!(Square, ZeroSuccessors);

/// Constructs a new detached/owned [`SquareOperation`] at the specified [`Location`].
pub fn square<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedSquareOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.square", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::square`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::square` operation cast")
}

/// Operation trait for `linalg.tanh`.
pub trait TanhOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Tanh);
mlir_op_trait!(Tanh, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Tanh, OneRegion);
mlir_op_trait!(Tanh, ZeroSuccessors);

/// Constructs a new detached/owned [`TanhOperation`] at the specified [`Location`].
pub fn tanh<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedTanhOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.tanh", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::tanh`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::tanh` operation cast")
}

/// Operation trait for `linalg.erf`.
pub trait ErfOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Erf);
mlir_op_trait!(Erf, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Erf, OneRegion);
mlir_op_trait!(Erf, ZeroSuccessors);

/// Constructs a new detached/owned [`ErfOperation`] at the specified [`Location`].
pub fn erf<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedErfOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.erf", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::erf`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::erf` operation cast")
}

/// Operation trait for `linalg.add`.
pub trait AddOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Add);
mlir_op_trait!(Add, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Add, OneRegion);
mlir_op_trait!(Add, ZeroSuccessors);

/// Constructs a new detached/owned [`AddOperation`] at the specified [`Location`].
pub fn add<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedAddOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.add", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::add`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::add` operation cast")
}

/// Operation trait for `linalg.sub`.
pub trait SubOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Sub);
mlir_op_trait!(Sub, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Sub, OneRegion);
mlir_op_trait!(Sub, ZeroSuccessors);

/// Constructs a new detached/owned [`SubOperation`] at the specified [`Location`].
pub fn sub<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedSubOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.sub", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::sub`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::sub` operation cast")
}

/// Operation trait for `linalg.mul`.
pub trait MulOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Mul);
mlir_op_trait!(Mul, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Mul, OneRegion);
mlir_op_trait!(Mul, ZeroSuccessors);

/// Constructs a new detached/owned [`MulOperation`] at the specified [`Location`].
pub fn mul<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedMulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.mul", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::mul`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::mul` operation cast")
}

/// Operation trait for `linalg.div`.
pub trait DivOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Div);
mlir_op_trait!(Div, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Div, OneRegion);
mlir_op_trait!(Div, ZeroSuccessors);

/// Constructs a new detached/owned [`DivOperation`] at the specified [`Location`].
pub fn div<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDivOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.div", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::div`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::div` operation cast")
}

/// Operation trait for `linalg.div_unsigned`.
pub trait DivUnsignedOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(DivUnsigned);
mlir_op_trait!(DivUnsigned, @local LinalgNamedStructuredOperation);
mlir_op_trait!(DivUnsigned, OneRegion);
mlir_op_trait!(DivUnsigned, ZeroSuccessors);

/// Constructs a new detached/owned [`DivUnsignedOperation`] at the specified [`Location`].
pub fn div_unsigned<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDivUnsignedOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.div_unsigned", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::div_unsigned`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::div_unsigned` operation cast")
}

/// Operation trait for `linalg.max`.
pub trait MaxOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Max);
mlir_op_trait!(Max, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Max, OneRegion);
mlir_op_trait!(Max, ZeroSuccessors);

/// Constructs a new detached/owned [`MaxOperation`] at the specified [`Location`].
pub fn max<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedMaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.max", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::max`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::max` operation cast")
}

/// Operation trait for `linalg.min`.
pub trait MinOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Min);
mlir_op_trait!(Min, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Min, OneRegion);
mlir_op_trait!(Min, ZeroSuccessors);

/// Constructs a new detached/owned [`MinOperation`] at the specified [`Location`].
pub fn min<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedMinOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.min", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::min`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::min` operation cast")
}

/// Operation trait for `linalg.powf`.
pub trait PowFOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(PowF);
mlir_op_trait!(PowF, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PowF, OneRegion);
mlir_op_trait!(PowF, ZeroSuccessors);

/// Constructs a new detached/owned [`PowFOperation`] at the specified [`Location`].
pub fn powf<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPowFOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.powf", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::powf`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::powf` operation cast")
}

/// Operation trait for `linalg.select`.
pub trait SelectOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Select);
mlir_op_trait!(Select, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Select, OneRegion);
mlir_op_trait!(Select, ZeroSuccessors);

/// Constructs a new detached/owned [`SelectOperation`] at the specified [`Location`].
pub fn select<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedSelectOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.select", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::select`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::select` operation cast")
}

/// Operation trait for `linalg.quantized_matmul`.
pub trait QuantizedMatmulOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(QuantizedMatmul);
mlir_op_trait!(QuantizedMatmul, @local LinalgNamedStructuredOperation);
mlir_op_trait!(QuantizedMatmul, OneRegion);
mlir_op_trait!(QuantizedMatmul, ZeroSuccessors);

/// Constructs a new detached/owned [`QuantizedMatmulOperation`] at the specified [`Location`].
pub fn quantized_matmul<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedQuantizedMatmulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.quantized_matmul", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::quantized_matmul`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::quantized_matmul` operation cast")
}

/// Operation trait for `linalg.mmt4d`.
pub trait Mmt4DOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Mmt4D);
mlir_op_trait!(Mmt4D, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Mmt4D, OneRegion);
mlir_op_trait!(Mmt4D, ZeroSuccessors);

/// Constructs a new detached/owned [`Mmt4DOperation`] at the specified [`Location`].
pub fn mmt4d<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedMmt4DOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.mmt4d", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::mmt4d`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::mmt4d` operation cast")
}

/// Operation trait for `linalg.batch_mmt4d`.
pub trait BatchMmt4DOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(BatchMmt4D);
mlir_op_trait!(BatchMmt4D, @local LinalgNamedStructuredOperation);
mlir_op_trait!(BatchMmt4D, OneRegion);
mlir_op_trait!(BatchMmt4D, ZeroSuccessors);

/// Constructs a new detached/owned [`BatchMmt4DOperation`] at the specified [`Location`].
pub fn batch_mmt4d<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedBatchMmt4DOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.batch_mmt4d", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::batch_mmt4d`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::batch_mmt4d` operation cast")
}

/// Operation trait for `linalg.quantized_batch_matmul`.
pub trait QuantizedBatchMatmulOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(QuantizedBatchMatmul);
mlir_op_trait!(QuantizedBatchMatmul, @local LinalgNamedStructuredOperation);
mlir_op_trait!(QuantizedBatchMatmul, OneRegion);
mlir_op_trait!(QuantizedBatchMatmul, ZeroSuccessors);

/// Constructs a new detached/owned [`QuantizedBatchMatmulOperation`] at the specified [`Location`].
pub fn quantized_batch_matmul<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedQuantizedBatchMatmulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.quantized_batch_matmul", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::quantized_batch_matmul`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::quantized_batch_matmul` operation cast")
}

/// Operation trait for `linalg.matvec`.
pub trait MatvecOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Matvec);
mlir_op_trait!(Matvec, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Matvec, OneRegion);
mlir_op_trait!(Matvec, ZeroSuccessors);

/// Constructs a new detached/owned [`MatvecOperation`] at the specified [`Location`].
pub fn matvec<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedMatvecOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.matvec", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::matvec`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::matvec` operation cast")
}

/// Operation trait for `linalg.vecmat`.
pub trait VecmatOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Vecmat);
mlir_op_trait!(Vecmat, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Vecmat, OneRegion);
mlir_op_trait!(Vecmat, ZeroSuccessors);

/// Constructs a new detached/owned [`VecmatOperation`] at the specified [`Location`].
pub fn vecmat<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedVecmatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.vecmat", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::vecmat`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::vecmat` operation cast")
}

/// Operation trait for `linalg.batch_matvec`.
pub trait BatchMatvecOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(BatchMatvec);
mlir_op_trait!(BatchMatvec, @local LinalgNamedStructuredOperation);
mlir_op_trait!(BatchMatvec, OneRegion);
mlir_op_trait!(BatchMatvec, ZeroSuccessors);

/// Constructs a new detached/owned [`BatchMatvecOperation`] at the specified [`Location`].
pub fn batch_matvec<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedBatchMatvecOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.batch_matvec", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::batch_matvec`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::batch_matvec` operation cast")
}

/// Operation trait for `linalg.batch_vecmat`.
pub trait BatchVecmatOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(BatchVecmat);
mlir_op_trait!(BatchVecmat, @local LinalgNamedStructuredOperation);
mlir_op_trait!(BatchVecmat, OneRegion);
mlir_op_trait!(BatchVecmat, ZeroSuccessors);

/// Constructs a new detached/owned [`BatchVecmatOperation`] at the specified [`Location`].
pub fn batch_vecmat<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedBatchVecmatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.batch_vecmat", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::batch_vecmat`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::batch_vecmat` operation cast")
}

/// Operation trait for `linalg.dot`.
pub trait DotOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Dot);
mlir_op_trait!(Dot, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Dot, OneRegion);
mlir_op_trait!(Dot, ZeroSuccessors);

/// Constructs a new detached/owned [`DotOperation`] at the specified [`Location`].
pub fn dot<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDotOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.dot", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::dot`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::dot` operation cast")
}

/// Operation trait for `linalg.conv_1d`.
pub trait Conv1DOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Conv1D);
mlir_op_trait!(Conv1D, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv1D, OneRegion);
mlir_op_trait!(Conv1D, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv1DOperation`] at the specified [`Location`].
pub fn conv_1d<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv1DOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_1d", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_1d`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_1d` operation cast")
}

/// Operation trait for `linalg.conv_2d`.
pub trait Conv2DOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Conv2D);
mlir_op_trait!(Conv2D, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv2D, OneRegion);
mlir_op_trait!(Conv2D, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv2DOperation`] at the specified [`Location`].
pub fn conv_2d<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv2DOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_2d", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_2d`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_2d` operation cast")
}

/// Operation trait for `linalg.conv_3d`.
pub trait Conv3DOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Conv3D);
mlir_op_trait!(Conv3D, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv3D, OneRegion);
mlir_op_trait!(Conv3D, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv3DOperation`] at the specified [`Location`].
pub fn conv_3d<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv3DOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_3d", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_3d`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_3d` operation cast")
}

/// Name of the Linalg strides attribute.
pub const STRIDES_ATTRIBUTE: &str = "strides";

/// Name of the Linalg dilations attribute.
pub const DILATIONS_ATTRIBUTE: &str = "dilations";

/// Common API for Linalg named structured operations with stride and dilation attributes.
pub trait LinalgStridedDilatedOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {
    /// Returns the optional stride values.
    fn strides(&self) -> Option<DenseIntegerElementsAttributeRef<'c, 't>> {
        self.attribute(STRIDES_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the optional dilation values.
    fn dilations(&self) -> Option<DenseIntegerElementsAttributeRef<'c, 't>> {
        self.attribute(DILATIONS_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }
}

/// Operation trait for `linalg.conv_1d_nwc_wcf`.
pub trait Conv1DNwcWcfOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv1DNwcWcf);
mlir_op_trait!(Conv1DNwcWcf, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv1DNwcWcf, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv1DNwcWcf, OneRegion);
mlir_op_trait!(Conv1DNwcWcf, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv1DNwcWcfOperation`] at the specified [`Location`].
pub fn conv_1d_nwc_wcf<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv1DNwcWcfOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_1d_nwc_wcf", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_1d_nwc_wcf`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_1d_nwc_wcf` operation cast")
}

/// Operation trait for `linalg.conv_1d_ncw_fcw`.
pub trait Conv1DNcwFcwOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv1DNcwFcw);
mlir_op_trait!(Conv1DNcwFcw, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv1DNcwFcw, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv1DNcwFcw, OneRegion);
mlir_op_trait!(Conv1DNcwFcw, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv1DNcwFcwOperation`] at the specified [`Location`].
pub fn conv_1d_ncw_fcw<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv1DNcwFcwOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_1d_ncw_fcw", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_1d_ncw_fcw`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_1d_ncw_fcw` operation cast")
}

/// Operation trait for `linalg.conv_2d_nhwc_hwcf`.
pub trait Conv2DNhwcHwcfOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv2DNhwcHwcf);
mlir_op_trait!(Conv2DNhwcHwcf, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv2DNhwcHwcf, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv2DNhwcHwcf, OneRegion);
mlir_op_trait!(Conv2DNhwcHwcf, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv2DNhwcHwcfOperation`] at the specified [`Location`].
pub fn conv_2d_nhwc_hwcf<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv2DNhwcHwcfOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_2d_nhwc_hwcf", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_2d_nhwc_hwcf`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_2d_nhwc_hwcf` operation cast")
}

/// Operation trait for `linalg.conv_2d_nhwc_fhwc`.
pub trait Conv2DNhwcFhwcOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv2DNhwcFhwc);
mlir_op_trait!(Conv2DNhwcFhwc, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv2DNhwcFhwc, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv2DNhwcFhwc, OneRegion);
mlir_op_trait!(Conv2DNhwcFhwc, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv2DNhwcFhwcOperation`] at the specified [`Location`].
pub fn conv_2d_nhwc_fhwc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv2DNhwcFhwcOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_2d_nhwc_fhwc", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_2d_nhwc_fhwc`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_2d_nhwc_fhwc` operation cast")
}

/// Operation trait for `linalg.conv_2d_nhwc_hwcf_q`.
pub trait Conv2DNhwcHwcfQOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv2DNhwcHwcfQ);
mlir_op_trait!(Conv2DNhwcHwcfQ, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv2DNhwcHwcfQ, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv2DNhwcHwcfQ, OneRegion);
mlir_op_trait!(Conv2DNhwcHwcfQ, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv2DNhwcHwcfQOperation`] at the specified [`Location`].
pub fn conv_2d_nhwc_hwcf_q<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv2DNhwcHwcfQOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_2d_nhwc_hwcf_q", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_2d_nhwc_hwcf_q`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_2d_nhwc_hwcf_q` operation cast")
}

/// Operation trait for `linalg.conv_2d_nhwc_fhwc_q`.
pub trait Conv2DNhwcFhwcQOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv2DNhwcFhwcQ);
mlir_op_trait!(Conv2DNhwcFhwcQ, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv2DNhwcFhwcQ, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv2DNhwcFhwcQ, OneRegion);
mlir_op_trait!(Conv2DNhwcFhwcQ, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv2DNhwcFhwcQOperation`] at the specified [`Location`].
pub fn conv_2d_nhwc_fhwc_q<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv2DNhwcFhwcQOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_2d_nhwc_fhwc_q", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_2d_nhwc_fhwc_q`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_2d_nhwc_fhwc_q` operation cast")
}

/// Operation trait for `linalg.conv_2d_nchw_fchw_q`.
pub trait Conv2DNchwFchwQOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv2DNchwFchwQ);
mlir_op_trait!(Conv2DNchwFchwQ, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv2DNchwFchwQ, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv2DNchwFchwQ, OneRegion);
mlir_op_trait!(Conv2DNchwFchwQ, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv2DNchwFchwQOperation`] at the specified [`Location`].
pub fn conv_2d_nchw_fchw_q<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv2DNchwFchwQOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_2d_nchw_fchw_q", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_2d_nchw_fchw_q`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_2d_nchw_fchw_q` operation cast")
}

/// Operation trait for `linalg.conv_2d_nchw_fchw`.
pub trait Conv2DNchwFchwOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv2DNchwFchw);
mlir_op_trait!(Conv2DNchwFchw, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv2DNchwFchw, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv2DNchwFchw, OneRegion);
mlir_op_trait!(Conv2DNchwFchw, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv2DNchwFchwOperation`] at the specified [`Location`].
pub fn conv_2d_nchw_fchw<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv2DNchwFchwOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_2d_nchw_fchw", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_2d_nchw_fchw`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_2d_nchw_fchw` operation cast")
}

/// Operation trait for `linalg.conv_2d_ngchw_fgchw`.
pub trait Conv2DNgchwFgchwOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv2DNgchwFgchw);
mlir_op_trait!(Conv2DNgchwFgchw, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv2DNgchwFgchw, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv2DNgchwFgchw, OneRegion);
mlir_op_trait!(Conv2DNgchwFgchw, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv2DNgchwFgchwOperation`] at the specified [`Location`].
pub fn conv_2d_ngchw_fgchw<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv2DNgchwFgchwOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_2d_ngchw_fgchw", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_2d_ngchw_fgchw`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_2d_ngchw_fgchw` operation cast")
}

/// Operation trait for `linalg.conv_2d_ngchw_gfchw`.
pub trait Conv2DNgchwGfchwOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv2DNgchwGfchw);
mlir_op_trait!(Conv2DNgchwGfchw, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv2DNgchwGfchw, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv2DNgchwGfchw, OneRegion);
mlir_op_trait!(Conv2DNgchwGfchw, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv2DNgchwGfchwOperation`] at the specified [`Location`].
pub fn conv_2d_ngchw_gfchw<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv2DNgchwGfchwOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_2d_ngchw_gfchw", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_2d_ngchw_gfchw`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_2d_ngchw_gfchw` operation cast")
}

/// Operation trait for `linalg.conv_2d_nhwgc_gfhwc`.
pub trait Conv2DNhwgcGfhwcOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv2DNhwgcGfhwc);
mlir_op_trait!(Conv2DNhwgcGfhwc, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv2DNhwgcGfhwc, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv2DNhwgcGfhwc, OneRegion);
mlir_op_trait!(Conv2DNhwgcGfhwc, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv2DNhwgcGfhwcOperation`] at the specified [`Location`].
pub fn conv_2d_nhwgc_gfhwc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv2DNhwgcGfhwcOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_2d_nhwgc_gfhwc", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_2d_nhwgc_gfhwc`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_2d_nhwgc_gfhwc` operation cast")
}

/// Operation trait for `linalg.conv_2d_nhwgc_gfhwc_q`.
pub trait Conv2DNhwgcGfhwcQOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv2DNhwgcGfhwcQ);
mlir_op_trait!(Conv2DNhwgcGfhwcQ, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv2DNhwgcGfhwcQ, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv2DNhwgcGfhwcQ, OneRegion);
mlir_op_trait!(Conv2DNhwgcGfhwcQ, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv2DNhwgcGfhwcQOperation`] at the specified [`Location`].
pub fn conv_2d_nhwgc_gfhwc_q<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv2DNhwgcGfhwcQOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_2d_nhwgc_gfhwc_q", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_2d_nhwgc_gfhwc_q`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_2d_nhwgc_gfhwc_q` operation cast")
}

/// Operation trait for `linalg.conv_2d_ngchw_gfchw_q`.
pub trait Conv2DNgchwGfchwQOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv2DNgchwGfchwQ);
mlir_op_trait!(Conv2DNgchwGfchwQ, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv2DNgchwGfchwQ, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv2DNgchwGfchwQ, OneRegion);
mlir_op_trait!(Conv2DNgchwGfchwQ, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv2DNgchwGfchwQOperation`] at the specified [`Location`].
pub fn conv_2d_ngchw_gfchw_q<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv2DNgchwGfchwQOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_2d_ngchw_gfchw_q", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_2d_ngchw_gfchw_q`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_2d_ngchw_gfchw_q` operation cast")
}

/// Operation trait for `linalg.conv_3d_ndhwc_dhwcf`.
pub trait Conv3DNdhwcDhwcfOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv3DNdhwcDhwcf);
mlir_op_trait!(Conv3DNdhwcDhwcf, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv3DNdhwcDhwcf, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv3DNdhwcDhwcf, OneRegion);
mlir_op_trait!(Conv3DNdhwcDhwcf, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv3DNdhwcDhwcfOperation`] at the specified [`Location`].
pub fn conv_3d_ndhwc_dhwcf<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv3DNdhwcDhwcfOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_3d_ndhwc_dhwcf", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_3d_ndhwc_dhwcf`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_3d_ndhwc_dhwcf` operation cast")
}

/// Operation trait for `linalg.conv_3d_ndhwc_dhwcf_q`.
pub trait Conv3DNdhwcDhwcfQOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv3DNdhwcDhwcfQ);
mlir_op_trait!(Conv3DNdhwcDhwcfQ, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv3DNdhwcDhwcfQ, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv3DNdhwcDhwcfQ, OneRegion);
mlir_op_trait!(Conv3DNdhwcDhwcfQ, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv3DNdhwcDhwcfQOperation`] at the specified [`Location`].
pub fn conv_3d_ndhwc_dhwcf_q<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv3DNdhwcDhwcfQOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_3d_ndhwc_dhwcf_q", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_3d_ndhwc_dhwcf_q`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_3d_ndhwc_dhwcf_q` operation cast")
}

/// Operation trait for `linalg.conv_3d_ncdhw_fcdhw`.
pub trait Conv3DNcdhwFcdhwOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(Conv3DNcdhwFcdhw);
mlir_op_trait!(Conv3DNcdhwFcdhw, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Conv3DNcdhwFcdhw, @local LinalgStridedDilatedOperation);
mlir_op_trait!(Conv3DNcdhwFcdhw, OneRegion);
mlir_op_trait!(Conv3DNcdhwFcdhw, ZeroSuccessors);

/// Constructs a new detached/owned [`Conv3DNcdhwFcdhwOperation`] at the specified [`Location`].
pub fn conv_3d_ncdhw_fcdhw<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedConv3DNcdhwFcdhwOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.conv_3d_ncdhw_fcdhw", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::conv_3d_ncdhw_fcdhw`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::conv_3d_ncdhw_fcdhw` operation cast")
}

/// Operation trait for `linalg.depthwise_conv_1d_nwc_wc`.
pub trait DepthwiseConv1DNwcWcOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(DepthwiseConv1DNwcWc);
mlir_op_trait!(DepthwiseConv1DNwcWc, @local LinalgNamedStructuredOperation);
mlir_op_trait!(DepthwiseConv1DNwcWc, @local LinalgStridedDilatedOperation);
mlir_op_trait!(DepthwiseConv1DNwcWc, OneRegion);
mlir_op_trait!(DepthwiseConv1DNwcWc, ZeroSuccessors);

/// Constructs a new detached/owned [`DepthwiseConv1DNwcWcOperation`] at the specified [`Location`].
pub fn depthwise_conv_1d_nwc_wc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDepthwiseConv1DNwcWcOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.depthwise_conv_1d_nwc_wc", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::depthwise_conv_1d_nwc_wc`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::depthwise_conv_1d_nwc_wc` operation cast")
}

/// Operation trait for `linalg.depthwise_conv_1d_ncw_cw`.
pub trait DepthwiseConv1DNcwCwOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(DepthwiseConv1DNcwCw);
mlir_op_trait!(DepthwiseConv1DNcwCw, @local LinalgNamedStructuredOperation);
mlir_op_trait!(DepthwiseConv1DNcwCw, @local LinalgStridedDilatedOperation);
mlir_op_trait!(DepthwiseConv1DNcwCw, OneRegion);
mlir_op_trait!(DepthwiseConv1DNcwCw, ZeroSuccessors);

/// Constructs a new detached/owned [`DepthwiseConv1DNcwCwOperation`] at the specified [`Location`].
pub fn depthwise_conv_1d_ncw_cw<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDepthwiseConv1DNcwCwOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.depthwise_conv_1d_ncw_cw", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::depthwise_conv_1d_ncw_cw`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::depthwise_conv_1d_ncw_cw` operation cast")
}

/// Operation trait for `linalg.depthwise_conv_1d_nwc_wcm`.
pub trait DepthwiseConv1DNwcWcmOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(DepthwiseConv1DNwcWcm);
mlir_op_trait!(DepthwiseConv1DNwcWcm, @local LinalgNamedStructuredOperation);
mlir_op_trait!(DepthwiseConv1DNwcWcm, @local LinalgStridedDilatedOperation);
mlir_op_trait!(DepthwiseConv1DNwcWcm, OneRegion);
mlir_op_trait!(DepthwiseConv1DNwcWcm, ZeroSuccessors);

/// Constructs a new detached/owned [`DepthwiseConv1DNwcWcmOperation`] at the specified [`Location`].
pub fn depthwise_conv_1d_nwc_wcm<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDepthwiseConv1DNwcWcmOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.depthwise_conv_1d_nwc_wcm", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::depthwise_conv_1d_nwc_wcm`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::depthwise_conv_1d_nwc_wcm` operation cast")
}

/// Operation trait for `linalg.depthwise_conv_2d_nhwc_hwc`.
pub trait DepthwiseConv2DNhwcHwcOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(DepthwiseConv2DNhwcHwc);
mlir_op_trait!(DepthwiseConv2DNhwcHwc, @local LinalgNamedStructuredOperation);
mlir_op_trait!(DepthwiseConv2DNhwcHwc, @local LinalgStridedDilatedOperation);
mlir_op_trait!(DepthwiseConv2DNhwcHwc, OneRegion);
mlir_op_trait!(DepthwiseConv2DNhwcHwc, ZeroSuccessors);

/// Constructs a new detached/owned [`DepthwiseConv2DNhwcHwcOperation`] at the specified [`Location`].
pub fn depthwise_conv_2d_nhwc_hwc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDepthwiseConv2DNhwcHwcOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.depthwise_conv_2d_nhwc_hwc", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::depthwise_conv_2d_nhwc_hwc`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::depthwise_conv_2d_nhwc_hwc` operation cast")
}

/// Operation trait for `linalg.depthwise_conv_2d_nchw_chw`.
pub trait DepthwiseConv2DNchwChwOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(DepthwiseConv2DNchwChw);
mlir_op_trait!(DepthwiseConv2DNchwChw, @local LinalgNamedStructuredOperation);
mlir_op_trait!(DepthwiseConv2DNchwChw, @local LinalgStridedDilatedOperation);
mlir_op_trait!(DepthwiseConv2DNchwChw, OneRegion);
mlir_op_trait!(DepthwiseConv2DNchwChw, ZeroSuccessors);

/// Constructs a new detached/owned [`DepthwiseConv2DNchwChwOperation`] at the specified [`Location`].
pub fn depthwise_conv_2d_nchw_chw<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDepthwiseConv2DNchwChwOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.depthwise_conv_2d_nchw_chw", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::depthwise_conv_2d_nchw_chw`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::depthwise_conv_2d_nchw_chw` operation cast")
}

/// Operation trait for `linalg.depthwise_conv_2d_nhwc_hwc_q`.
pub trait DepthwiseConv2DNhwcHwcQOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(DepthwiseConv2DNhwcHwcQ);
mlir_op_trait!(DepthwiseConv2DNhwcHwcQ, @local LinalgNamedStructuredOperation);
mlir_op_trait!(DepthwiseConv2DNhwcHwcQ, @local LinalgStridedDilatedOperation);
mlir_op_trait!(DepthwiseConv2DNhwcHwcQ, OneRegion);
mlir_op_trait!(DepthwiseConv2DNhwcHwcQ, ZeroSuccessors);

/// Constructs a new detached/owned [`DepthwiseConv2DNhwcHwcQOperation`] at the specified [`Location`].
pub fn depthwise_conv_2d_nhwc_hwc_q<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDepthwiseConv2DNhwcHwcQOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.depthwise_conv_2d_nhwc_hwc_q", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::depthwise_conv_2d_nhwc_hwc_q`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::depthwise_conv_2d_nhwc_hwc_q` operation cast")
}

/// Operation trait for `linalg.depthwise_conv_2d_nhwc_hwcm`.
pub trait DepthwiseConv2DNhwcHwcmOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(DepthwiseConv2DNhwcHwcm);
mlir_op_trait!(DepthwiseConv2DNhwcHwcm, @local LinalgNamedStructuredOperation);
mlir_op_trait!(DepthwiseConv2DNhwcHwcm, @local LinalgStridedDilatedOperation);
mlir_op_trait!(DepthwiseConv2DNhwcHwcm, OneRegion);
mlir_op_trait!(DepthwiseConv2DNhwcHwcm, ZeroSuccessors);

/// Constructs a new detached/owned [`DepthwiseConv2DNhwcHwcmOperation`] at the specified [`Location`].
pub fn depthwise_conv_2d_nhwc_hwcm<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDepthwiseConv2DNhwcHwcmOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.depthwise_conv_2d_nhwc_hwcm", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::depthwise_conv_2d_nhwc_hwcm`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::depthwise_conv_2d_nhwc_hwcm` operation cast")
}

/// Operation trait for `linalg.depthwise_conv_2d_nhwc_hwcm_q`.
pub trait DepthwiseConv2DNhwcHwcmQOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(DepthwiseConv2DNhwcHwcmQ);
mlir_op_trait!(DepthwiseConv2DNhwcHwcmQ, @local LinalgNamedStructuredOperation);
mlir_op_trait!(DepthwiseConv2DNhwcHwcmQ, @local LinalgStridedDilatedOperation);
mlir_op_trait!(DepthwiseConv2DNhwcHwcmQ, OneRegion);
mlir_op_trait!(DepthwiseConv2DNhwcHwcmQ, ZeroSuccessors);

/// Constructs a new detached/owned [`DepthwiseConv2DNhwcHwcmQOperation`] at the specified [`Location`].
pub fn depthwise_conv_2d_nhwc_hwcm_q<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDepthwiseConv2DNhwcHwcmQOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.depthwise_conv_2d_nhwc_hwcm_q", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::depthwise_conv_2d_nhwc_hwcm_q`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::depthwise_conv_2d_nhwc_hwcm_q` operation cast")
}

/// Operation trait for `linalg.depthwise_conv_3d_ndhwc_dhwc`.
pub trait DepthwiseConv3DNdhwcDhwcOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(DepthwiseConv3DNdhwcDhwc);
mlir_op_trait!(DepthwiseConv3DNdhwcDhwc, @local LinalgNamedStructuredOperation);
mlir_op_trait!(DepthwiseConv3DNdhwcDhwc, @local LinalgStridedDilatedOperation);
mlir_op_trait!(DepthwiseConv3DNdhwcDhwc, OneRegion);
mlir_op_trait!(DepthwiseConv3DNdhwcDhwc, ZeroSuccessors);

/// Constructs a new detached/owned [`DepthwiseConv3DNdhwcDhwcOperation`] at the specified [`Location`].
pub fn depthwise_conv_3d_ndhwc_dhwc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDepthwiseConv3DNdhwcDhwcOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.depthwise_conv_3d_ndhwc_dhwc", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::depthwise_conv_3d_ndhwc_dhwc`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::depthwise_conv_3d_ndhwc_dhwc` operation cast")
}

/// Operation trait for `linalg.depthwise_conv_3d_ncdhw_cdhw`.
pub trait DepthwiseConv3DNcdhwCdhwOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(DepthwiseConv3DNcdhwCdhw);
mlir_op_trait!(DepthwiseConv3DNcdhwCdhw, @local LinalgNamedStructuredOperation);
mlir_op_trait!(DepthwiseConv3DNcdhwCdhw, @local LinalgStridedDilatedOperation);
mlir_op_trait!(DepthwiseConv3DNcdhwCdhw, OneRegion);
mlir_op_trait!(DepthwiseConv3DNcdhwCdhw, ZeroSuccessors);

/// Constructs a new detached/owned [`DepthwiseConv3DNcdhwCdhwOperation`] at the specified [`Location`].
pub fn depthwise_conv_3d_ncdhw_cdhw<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDepthwiseConv3DNcdhwCdhwOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.depthwise_conv_3d_ncdhw_cdhw", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::depthwise_conv_3d_ncdhw_cdhw`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::depthwise_conv_3d_ncdhw_cdhw` operation cast")
}

/// Operation trait for `linalg.depthwise_conv_3d_ndhwc_dhwcm`.
pub trait DepthwiseConv3DNdhwcDhwcmOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(DepthwiseConv3DNdhwcDhwcm);
mlir_op_trait!(DepthwiseConv3DNdhwcDhwcm, @local LinalgNamedStructuredOperation);
mlir_op_trait!(DepthwiseConv3DNdhwcDhwcm, @local LinalgStridedDilatedOperation);
mlir_op_trait!(DepthwiseConv3DNdhwcDhwcm, OneRegion);
mlir_op_trait!(DepthwiseConv3DNdhwcDhwcm, ZeroSuccessors);

/// Constructs a new detached/owned [`DepthwiseConv3DNdhwcDhwcmOperation`] at the specified [`Location`].
pub fn depthwise_conv_3d_ndhwc_dhwcm<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedDepthwiseConv3DNdhwcDhwcmOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.depthwise_conv_3d_ndhwc_dhwcm", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::depthwise_conv_3d_ndhwc_dhwcm`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::depthwise_conv_3d_ndhwc_dhwcm` operation cast")
}

/// Operation trait for `linalg.pooling_nhwc_sum`.
pub trait PoolingNhwcSumOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNhwcSum);
mlir_op_trait!(PoolingNhwcSum, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNhwcSum, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNhwcSum, OneRegion);
mlir_op_trait!(PoolingNhwcSum, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNhwcSumOperation`] at the specified [`Location`].
pub fn pooling_nhwc_sum<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNhwcSumOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_nhwc_sum", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_nhwc_sum`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_nhwc_sum` operation cast")
}

/// Operation trait for `linalg.pooling_nchw_sum`.
pub trait PoolingNchwSumOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNchwSum);
mlir_op_trait!(PoolingNchwSum, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNchwSum, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNchwSum, OneRegion);
mlir_op_trait!(PoolingNchwSum, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNchwSumOperation`] at the specified [`Location`].
pub fn pooling_nchw_sum<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNchwSumOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_nchw_sum", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_nchw_sum`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_nchw_sum` operation cast")
}

/// Operation trait for `linalg.pooling_nhwc_max`.
pub trait PoolingNhwcMaxOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNhwcMax);
mlir_op_trait!(PoolingNhwcMax, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNhwcMax, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNhwcMax, OneRegion);
mlir_op_trait!(PoolingNhwcMax, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNhwcMaxOperation`] at the specified [`Location`].
pub fn pooling_nhwc_max<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNhwcMaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_nhwc_max", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_nhwc_max`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_nhwc_max` operation cast")
}

/// Operation trait for `linalg.pooling_nhwc_max_unsigned`.
pub trait PoolingNhwcMaxUnsignedOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNhwcMaxUnsigned);
mlir_op_trait!(PoolingNhwcMaxUnsigned, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNhwcMaxUnsigned, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNhwcMaxUnsigned, OneRegion);
mlir_op_trait!(PoolingNhwcMaxUnsigned, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNhwcMaxUnsignedOperation`] at the specified [`Location`].
pub fn pooling_nhwc_max_unsigned<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNhwcMaxUnsignedOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_nhwc_max_unsigned", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_nhwc_max_unsigned`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_nhwc_max_unsigned` operation cast")
}

/// Operation trait for `linalg.pooling_nchw_max`.
pub trait PoolingNchwMaxOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNchwMax);
mlir_op_trait!(PoolingNchwMax, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNchwMax, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNchwMax, OneRegion);
mlir_op_trait!(PoolingNchwMax, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNchwMaxOperation`] at the specified [`Location`].
pub fn pooling_nchw_max<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNchwMaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_nchw_max", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_nchw_max`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_nchw_max` operation cast")
}

/// Operation trait for `linalg.pooling_nhwc_min`.
pub trait PoolingNhwcMinOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNhwcMin);
mlir_op_trait!(PoolingNhwcMin, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNhwcMin, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNhwcMin, OneRegion);
mlir_op_trait!(PoolingNhwcMin, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNhwcMinOperation`] at the specified [`Location`].
pub fn pooling_nhwc_min<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNhwcMinOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_nhwc_min", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_nhwc_min`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_nhwc_min` operation cast")
}

/// Operation trait for `linalg.pooling_nhwc_min_unsigned`.
pub trait PoolingNhwcMinUnsignedOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNhwcMinUnsigned);
mlir_op_trait!(PoolingNhwcMinUnsigned, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNhwcMinUnsigned, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNhwcMinUnsigned, OneRegion);
mlir_op_trait!(PoolingNhwcMinUnsigned, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNhwcMinUnsignedOperation`] at the specified [`Location`].
pub fn pooling_nhwc_min_unsigned<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNhwcMinUnsignedOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_nhwc_min_unsigned", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_nhwc_min_unsigned`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_nhwc_min_unsigned` operation cast")
}

/// Operation trait for `linalg.pooling_nwc_sum`.
pub trait PoolingNwcSumOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNwcSum);
mlir_op_trait!(PoolingNwcSum, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNwcSum, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNwcSum, OneRegion);
mlir_op_trait!(PoolingNwcSum, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNwcSumOperation`] at the specified [`Location`].
pub fn pooling_nwc_sum<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNwcSumOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_nwc_sum", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_nwc_sum`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_nwc_sum` operation cast")
}

/// Operation trait for `linalg.pooling_ncw_sum`.
pub trait PoolingNcwSumOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNcwSum);
mlir_op_trait!(PoolingNcwSum, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNcwSum, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNcwSum, OneRegion);
mlir_op_trait!(PoolingNcwSum, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNcwSumOperation`] at the specified [`Location`].
pub fn pooling_ncw_sum<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNcwSumOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_ncw_sum", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_ncw_sum`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_ncw_sum` operation cast")
}

/// Operation trait for `linalg.pooling_nwc_max`.
pub trait PoolingNwcMaxOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNwcMax);
mlir_op_trait!(PoolingNwcMax, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNwcMax, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNwcMax, OneRegion);
mlir_op_trait!(PoolingNwcMax, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNwcMaxOperation`] at the specified [`Location`].
pub fn pooling_nwc_max<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNwcMaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_nwc_max", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_nwc_max`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_nwc_max` operation cast")
}

/// Operation trait for `linalg.pooling_nwc_max_unsigned`.
pub trait PoolingNwcMaxUnsignedOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNwcMaxUnsigned);
mlir_op_trait!(PoolingNwcMaxUnsigned, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNwcMaxUnsigned, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNwcMaxUnsigned, OneRegion);
mlir_op_trait!(PoolingNwcMaxUnsigned, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNwcMaxUnsignedOperation`] at the specified [`Location`].
pub fn pooling_nwc_max_unsigned<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNwcMaxUnsignedOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_nwc_max_unsigned", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_nwc_max_unsigned`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_nwc_max_unsigned` operation cast")
}

/// Operation trait for `linalg.pooling_ncw_max`.
pub trait PoolingNcwMaxOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNcwMax);
mlir_op_trait!(PoolingNcwMax, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNcwMax, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNcwMax, OneRegion);
mlir_op_trait!(PoolingNcwMax, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNcwMaxOperation`] at the specified [`Location`].
pub fn pooling_ncw_max<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNcwMaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_ncw_max", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_ncw_max`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_ncw_max` operation cast")
}

/// Operation trait for `linalg.pooling_nwc_min`.
pub trait PoolingNwcMinOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNwcMin);
mlir_op_trait!(PoolingNwcMin, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNwcMin, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNwcMin, OneRegion);
mlir_op_trait!(PoolingNwcMin, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNwcMinOperation`] at the specified [`Location`].
pub fn pooling_nwc_min<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNwcMinOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_nwc_min", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_nwc_min`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_nwc_min` operation cast")
}

/// Operation trait for `linalg.pooling_nwc_min_unsigned`.
pub trait PoolingNwcMinUnsignedOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNwcMinUnsigned);
mlir_op_trait!(PoolingNwcMinUnsigned, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNwcMinUnsigned, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNwcMinUnsigned, OneRegion);
mlir_op_trait!(PoolingNwcMinUnsigned, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNwcMinUnsignedOperation`] at the specified [`Location`].
pub fn pooling_nwc_min_unsigned<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNwcMinUnsignedOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_nwc_min_unsigned", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_nwc_min_unsigned`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_nwc_min_unsigned` operation cast")
}

/// Operation trait for `linalg.pooling_ndhwc_sum`.
pub trait PoolingNdhwcSumOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNdhwcSum);
mlir_op_trait!(PoolingNdhwcSum, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNdhwcSum, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNdhwcSum, OneRegion);
mlir_op_trait!(PoolingNdhwcSum, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNdhwcSumOperation`] at the specified [`Location`].
pub fn pooling_ndhwc_sum<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNdhwcSumOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_ndhwc_sum", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_ndhwc_sum`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_ndhwc_sum` operation cast")
}

/// Operation trait for `linalg.pooling_ndhwc_max`.
pub trait PoolingNdhwcMaxOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNdhwcMax);
mlir_op_trait!(PoolingNdhwcMax, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNdhwcMax, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNdhwcMax, OneRegion);
mlir_op_trait!(PoolingNdhwcMax, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNdhwcMaxOperation`] at the specified [`Location`].
pub fn pooling_ndhwc_max<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNdhwcMaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_ndhwc_max", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_ndhwc_max`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_ndhwc_max` operation cast")
}

/// Operation trait for `linalg.pooling_ndhwc_min`.
pub trait PoolingNdhwcMinOperation<'o, 'c: 'o, 't: 'c>: LinalgStridedDilatedOperation<'o, 'c, 't> {}

mlir_op!(PoolingNdhwcMin);
mlir_op_trait!(PoolingNdhwcMin, @local LinalgNamedStructuredOperation);
mlir_op_trait!(PoolingNdhwcMin, @local LinalgStridedDilatedOperation);
mlir_op_trait!(PoolingNdhwcMin, OneRegion);
mlir_op_trait!(PoolingNdhwcMin, ZeroSuccessors);

/// Constructs a new detached/owned [`PoolingNdhwcMinOperation`] at the specified [`Location`].
pub fn pooling_ndhwc_min<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedPoolingNdhwcMinOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.pooling_ndhwc_min", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::pooling_ndhwc_min`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::pooling_ndhwc_min` operation cast")
}

/// Operation trait for `linalg.fill`.
pub trait FillOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(Fill);
mlir_op_trait!(Fill, @local LinalgNamedStructuredOperation);
mlir_op_trait!(Fill, OneRegion);
mlir_op_trait!(Fill, ZeroSuccessors);

/// Constructs a new detached/owned [`FillOperation`] at the specified [`Location`].
pub fn fill<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedFillOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.fill", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::fill`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::fill` operation cast")
}

/// Operation trait for `linalg.fill_rng_2d`.
pub trait FillRng2DOperation<'o, 'c: 'o, 't: 'c>: LinalgNamedStructuredOperation<'o, 'c, 't> {}

mlir_op!(FillRng2D);
mlir_op_trait!(FillRng2D, @local LinalgNamedStructuredOperation);
mlir_op_trait!(FillRng2D, OneRegion);
mlir_op_trait!(FillRng2D, ZeroSuccessors);

/// Constructs a new detached/owned [`FillRng2DOperation`] at the specified [`Location`].
pub fn fill_rng_2d<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    outputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    location: L,
) -> DetachedFillRng2DOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::linalg());
    let segment_sizes = [inputs.len() as i32, outputs.len() as i32];
    let mut builder = OperationBuilder::new("linalg.fill_rng_2d", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_operands(inputs)
        .add_operands(outputs)
        .add_results(result_types)
        .add_region(context.region());
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    let operation = builder.build().expect("invalid arguments to `linalg::fill_rng_2d`");
    unsafe { mlirLinalgFillBuiltinNamedOpRegion(operation.to_c_api()) };
    unsafe { operation.cast() }.expect("invalid `linalg::fill_rng_2d` operation cast")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::{func, linalg::TypeFn};
    use crate::{Attribute, Block, Context, Operation, Size, Type, Value, VectorTypeDimension};

    use super::{
        CAST_ATTRIBUTE, DILATIONS_ATTRIBUTE, LinalgCastedOperation, LinalgNamedStructuredOperation,
        LinalgStridedDilatedOperation, STRIDES_ATTRIBUTE, add, conv_1d_nwc_wcf, copy,
    };

    #[test]
    fn test_linalg_add_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap().as_ref();
            let rhs = block.argument(1).unwrap().as_ref();
            let output = block.argument(2).unwrap().as_ref();
            let result_types = [tensor_type.as_ref()];
            let op = add(&[lhs, rhs], &[output], &result_types, &[], location);
            assert_eq!(op.inputs(), vec![lhs, rhs]);
            assert_eq!(op.outputs(), vec![output]);
            assert_eq!(op.result_tensors()[0].r#type(), tensor_type.as_ref());
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "linalg_add_test",
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
                  func.func @linalg_add_test(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> tensor<3xf32> {
                    %0 = linalg.add ins(%arg0, %arg1 : tensor<3xf32>, tensor<3xf32>) outs(%arg2 : tensor<3xf32>) -> tensor<3xf32>
                    return %0 : tensor<3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_linalg_copy_operation_cast_attribute() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let input = block.argument(0).unwrap().as_ref();
            let output = block.argument(1).unwrap().as_ref();
            let result_types = [tensor_type.as_ref()];
            let cast = context.linalg_type_fn_attribute(TypeFn::CastUnsigned);
            let attributes = [(CAST_ATTRIBUTE, cast.as_ref())];
            let op = copy(&[input], &[output], &result_types, &attributes, location);
            assert_eq!(LinalgCastedOperation::cast(&op), TypeFn::CastUnsigned);
            assert_eq!(op.inputs(), vec![input]);
            assert_eq!(op.outputs(), vec![output]);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "linalg_copy_test",
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
                  func.func @linalg_copy_test(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
                    %0 = linalg.copy {cast = #linalg.type_fn<cast_unsigned>} ins(%arg0 : tensor<3xf32>) outs(%arg1 : tensor<3xf32>) -> tensor<3xf32>
                    return %0 : tensor<3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_linalg_conv_1d_nwc_wcf_operation_stride_and_dilation_attributes() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let input_type = context
            .tensor_type(context.float32_type(), &[Size::Static(1), Size::Static(9), Size::Static(2)], None, location)
            .unwrap();
        let filter_type = context
            .tensor_type(context.float32_type(), &[Size::Static(3), Size::Static(2), Size::Static(4)], None, location)
            .unwrap();
        let output_type = context
            .tensor_type(context.float32_type(), &[Size::Static(1), Size::Static(4), Size::Static(4)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (filter_type, location), (output_type, location)]);
            let input = block.argument(0).unwrap().as_ref();
            let filter = block.argument(1).unwrap().as_ref();
            let output = block.argument(2).unwrap().as_ref();
            let result_types = [output_type.as_ref()];
            let attribute_type = context
                .vector_type(context.signless_integer_type(64), &[VectorTypeDimension::Fixed(1)], location)
                .unwrap();
            let strides = context.dense_i64_elements_attribute(attribute_type, &[2]).unwrap();
            let dilations = context.dense_i64_elements_attribute(attribute_type, &[1]).unwrap();
            let attributes = [(STRIDES_ATTRIBUTE, strides.as_ref()), (DILATIONS_ATTRIBUTE, dilations.as_ref())];
            let op = conv_1d_nwc_wcf(&[input, filter], &[output], &result_types, &attributes, location);
            assert_eq!(
                unsafe { LinalgStridedDilatedOperation::strides(&op).unwrap().i64_elements().collect::<Vec<_>>() },
                vec![2],
            );
            assert_eq!(
                unsafe { LinalgStridedDilatedOperation::dilations(&op).unwrap().i64_elements().collect::<Vec<_>>() },
                vec![1],
            );
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "linalg_conv_1d_nwc_wcf_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), filter_type.into(), output_type.into()],
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
                  func.func @linalg_conv_1d_nwc_wcf_test(%arg0: tensor<1x9x2xf32>, %arg1: tensor<3x2x4xf32>, %arg2: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
                    %0 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%arg0, %arg1 : tensor<1x9x2xf32>, tensor<3x2x4xf32>) outs(%arg2 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
                    return %0 : tensor<1x4x4xf32>
                  }
                }
            "},
        );
    }
}
