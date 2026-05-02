use crate::{
    AffineMap, AffineMapAttributeRef, ArrayAttributeRef, Attribute, DenseInteger32ArrayAttributeRef, DetachedOp,
    DetachedRegion, DialectHandle, IntegerAttributeRef, Location, OneResult, Operation, OperationBuilder, RegionRef,
    Type, TypeRef, Value, ValueRef, mlir_op, mlir_op_trait,
};

use super::{
    CoordinateTranslationDirection, CoordinateTranslationDirectionAttributeRef, SortKind, SortKindAttributeRef,
    SparseTensorEncodingAttributeRef, StorageSpecifierKind, StorageSpecifierKindAttributeRef,
};

/// Operation trait for `sparse_tensor.new`.
pub trait NewOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source value used to materialize the sparse tensor.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the materialized sparse tensor.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(New);
mlir_op_trait!(New, AlwaysSpeculatable);
mlir_op_trait!(New, NoMemoryEffect);
mlir_op_trait!(New, OneResult);
mlir_op_trait!(New, Pure);
mlir_op_trait!(New, ZeroRegions);
mlir_op_trait!(New, ZeroSuccessors);

/// Constructs a new detached/owned [`NewOperation`] at the specified [`Location`].
pub fn new<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    result_type: T,
    location: L,
) -> DetachedNewOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.new", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::new`")
}

/// Operation trait for `sparse_tensor.assemble`.
pub trait AssembleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the level storage tensors.
    fn levels(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (0..self.operand_count() - 1).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the values tensor.
    fn values(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(self.operand_count() - 1).unwrap()
    }

    /// Returns the assembled sparse tensor.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Assemble);
mlir_op_trait!(Assemble, AlwaysSpeculatable);
mlir_op_trait!(Assemble, NoMemoryEffect);
mlir_op_trait!(Assemble, OneResult);
mlir_op_trait!(Assemble, Pure);
mlir_op_trait!(Assemble, ZeroRegions);
mlir_op_trait!(Assemble, ZeroSuccessors);

/// Constructs a new detached/owned [`AssembleOperation`] at the specified [`Location`].
pub fn assemble<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    levels: &[ValueRef<'v, 'c, 't>],
    values: ValueRef<'v, 'c, 't>,
    result_type: T,
    location: L,
) -> DetachedAssembleOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.assemble", location)
        .add_operands(levels)
        .add_operand(values)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::assemble`")
}

/// Operation trait for `sparse_tensor.disassemble`.
pub trait DisassembleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the sparse tensor being disassembled.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the output level buffers.
    fn output_levels(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let level_count = (self.result_count() - 2) / 2;
        (1..1 + level_count).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the output values buffer.
    fn output_values(&self) -> ValueRef<'o, 'c, 't> {
        let level_count = (self.result_count() - 2) / 2;
        self.operand_value(1 + level_count).unwrap()
    }

    /// Returns the copied level buffers.
    fn returned_levels(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let level_count = (self.result_count() - 2) / 2;
        (0..level_count).map(|index| self.result(index).unwrap().as_ref()).collect()
    }

    /// Returns the copied values buffer.
    fn returned_values(&self) -> ValueRef<'o, 'c, 't> {
        let level_count = (self.result_count() - 2) / 2;
        self.result(level_count).unwrap().as_ref()
    }

    /// Returns the occupied lengths of the returned level buffers.
    fn level_lengths(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let level_count = (self.result_count() - 2) / 2;
        (level_count + 1..level_count + 1 + level_count)
            .map(|index| self.result(index).unwrap().as_ref())
            .collect()
    }

    /// Returns the occupied length of the returned values buffer.
    fn values_length(&self) -> ValueRef<'o, 'c, 't> {
        self.result(self.result_count() - 1).unwrap().as_ref()
    }
}

mlir_op!(Disassemble);
mlir_op_trait!(Disassemble, AlwaysSpeculatable);
mlir_op_trait!(Disassemble, NoMemoryEffect);
mlir_op_trait!(Disassemble, Pure);
mlir_op_trait!(Disassemble, ZeroRegions);
mlir_op_trait!(Disassemble, ZeroSuccessors);

/// Constructs a new detached/owned [`DisassembleOperation`] at the specified [`Location`].
pub fn disassemble<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    output_levels: &[ValueRef<'v, 'c, 't>],
    output_values: ValueRef<'v, 'c, 't>,
    returned_level_types: &[TypeRef<'c, 't>],
    returned_values_type: TypeRef<'c, 't>,
    level_length_types: &[TypeRef<'c, 't>],
    values_length_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedDisassembleOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.disassemble", location)
        .add_operand(tensor)
        .add_operands(output_levels)
        .add_operand(output_values)
        .add_results(returned_level_types)
        .add_result(returned_values_type)
        .add_results(level_length_types)
        .add_result(values_length_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::disassemble`")
}

/// Operation trait for `sparse_tensor.convert`.
pub trait ConvertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source tensor.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the converted tensor.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Convert);
mlir_op_trait!(Convert, AlwaysSpeculatable);
mlir_op_trait!(Convert, NoMemoryEffect);
mlir_op_trait!(Convert, OneResult);
mlir_op_trait!(Convert, Pure);
mlir_op_trait!(Convert, ZeroRegions);
mlir_op_trait!(Convert, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertOperation`] at the specified [`Location`].
pub fn convert<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    result_type: T,
    location: L,
) -> DetachedConvertOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.convert", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::convert`")
}

/// Operation trait for `sparse_tensor.reinterpret_map`.
pub trait ReinterpretMapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source tensor.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor with reinterpreted dimension and level maps.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(ReinterpretMap);
mlir_op_trait!(ReinterpretMap, NoMemoryEffect);
mlir_op_trait!(ReinterpretMap, OneResult);
mlir_op_trait!(ReinterpretMap, ZeroRegions);
mlir_op_trait!(ReinterpretMap, ZeroSuccessors);

/// Constructs a new detached/owned [`ReinterpretMapOperation`] at the specified [`Location`].
pub fn reinterpret_map<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    result_type: T,
    location: L,
) -> DetachedReinterpretMapOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.reinterpret_map", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::reinterpret_map`")
}

/// Name of the sparse tensor level attribute.
pub const LEVEL_ATTRIBUTE: &str = "level";

/// Operation trait for `sparse_tensor.positions`.
pub trait PositionsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being queried.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the storage level whose positions buffer is queried.
    fn level(&self) -> i64 {
        self.attribute(LEVEL_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
            .unwrap_or_else(|| panic!("invalid '{LEVEL_ATTRIBUTE}' attribute in `sparse_tensor.positions`"))
    }

    /// Returns the positions buffer.
    fn positions(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Positions);
mlir_op_trait!(Positions, AlwaysSpeculatable);
mlir_op_trait!(Positions, NoMemoryEffect);
mlir_op_trait!(Positions, OneResult);
mlir_op_trait!(Positions, Pure);
mlir_op_trait!(Positions, ZeroRegions);
mlir_op_trait!(Positions, ZeroSuccessors);

/// Constructs a new detached/owned [`PositionsOperation`] at the specified [`Location`].
pub fn positions<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    level: i64,
    result_type: T,
    location: L,
) -> DetachedPositionsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.positions", location)
        .add_operand(tensor)
        .add_attribute(LEVEL_ATTRIBUTE, context.integer_attribute(context.index_type(), level))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::positions`")
}

/// Operation trait for `sparse_tensor.coordinates`.
pub trait CoordinatesOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being queried.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the storage level whose coordinates buffer is queried.
    fn level(&self) -> i64 {
        self.attribute(LEVEL_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
            .unwrap_or_else(|| panic!("invalid '{LEVEL_ATTRIBUTE}' attribute in `sparse_tensor.coordinates`"))
    }

    /// Returns the coordinates buffer.
    fn coordinates(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Coordinates);
mlir_op_trait!(Coordinates, AlwaysSpeculatable);
mlir_op_trait!(Coordinates, NoMemoryEffect);
mlir_op_trait!(Coordinates, OneResult);
mlir_op_trait!(Coordinates, Pure);
mlir_op_trait!(Coordinates, ZeroRegions);
mlir_op_trait!(Coordinates, ZeroSuccessors);

/// Constructs a new detached/owned [`CoordinatesOperation`] at the specified [`Location`].
pub fn coordinates<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    level: i64,
    result_type: T,
    location: L,
) -> DetachedCoordinatesOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.coordinates", location)
        .add_operand(tensor)
        .add_attribute(LEVEL_ATTRIBUTE, context.integer_attribute(context.index_type(), level))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::coordinates`")
}

/// Operation trait for `sparse_tensor.coordinates_buffer`.
pub trait CoordinatesBufferOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being queried.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the linear coordinates buffer.
    fn coordinates_buffer(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(CoordinatesBuffer);
mlir_op_trait!(CoordinatesBuffer, AlwaysSpeculatable);
mlir_op_trait!(CoordinatesBuffer, NoMemoryEffect);
mlir_op_trait!(CoordinatesBuffer, OneResult);
mlir_op_trait!(CoordinatesBuffer, Pure);
mlir_op_trait!(CoordinatesBuffer, ZeroRegions);
mlir_op_trait!(CoordinatesBuffer, ZeroSuccessors);

/// Constructs a new detached/owned [`CoordinatesBufferOperation`] at the specified [`Location`].
pub fn coordinates_buffer<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    result_type: T,
    location: L,
) -> DetachedCoordinatesBufferOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.coordinates_buffer", location)
        .add_operand(tensor)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::coordinates_buffer`")
}

/// Operation trait for `sparse_tensor.values`.
pub trait ValuesOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being queried.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the values buffer.
    fn values(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Values);
mlir_op_trait!(Values, AlwaysSpeculatable);
mlir_op_trait!(Values, NoMemoryEffect);
mlir_op_trait!(Values, OneResult);
mlir_op_trait!(Values, Pure);
mlir_op_trait!(Values, ZeroRegions);
mlir_op_trait!(Values, ZeroSuccessors);

/// Constructs a new detached/owned [`ValuesOperation`] at the specified [`Location`].
pub fn values<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    result_type: T,
    location: L,
) -> DetachedValuesOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.values", location)
        .add_operand(tensor)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::values`")
}

/// Operation trait for `sparse_tensor.number_of_entries`.
pub trait NumberOfEntriesOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being queried.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the number of stored entries.
    fn entry_count(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(NumberOfEntries);
mlir_op_trait!(NumberOfEntries, AlwaysSpeculatable);
mlir_op_trait!(NumberOfEntries, NoMemoryEffect);
mlir_op_trait!(NumberOfEntries, OneResult);
mlir_op_trait!(NumberOfEntries, Pure);
mlir_op_trait!(NumberOfEntries, ZeroRegions);
mlir_op_trait!(NumberOfEntries, ZeroSuccessors);

/// Constructs a new detached/owned [`NumberOfEntriesOperation`] at the specified [`Location`].
pub fn number_of_entries<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedNumberOfEntriesOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.number_of_entries", location)
        .add_operand(tensor)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::number_of_entries`")
}

/// Name of the sparse tensor dimension attribute.
pub const DIMENSION_ATTRIBUTE: &str = "dimension";

/// Operation trait for `sparse_tensor.concatenate`.
pub trait ConcatenateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the input tensors being concatenated.
    fn inputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the dimension along which the inputs are concatenated.
    fn dimension(&self) -> i64 {
        self.attribute(DIMENSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
            .unwrap_or_else(|| panic!("invalid '{DIMENSION_ATTRIBUTE}' attribute in `sparse_tensor.concatenate`"))
    }

    /// Returns the concatenated tensor.
    fn concatenated(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Concatenate);
mlir_op_trait!(Concatenate, AlwaysSpeculatable);
mlir_op_trait!(Concatenate, NoMemoryEffect);
mlir_op_trait!(Concatenate, OneResult);
mlir_op_trait!(Concatenate, Pure);
mlir_op_trait!(Concatenate, ZeroRegions);
mlir_op_trait!(Concatenate, ZeroSuccessors);

/// Constructs a new detached/owned [`ConcatenateOperation`] at the specified [`Location`].
pub fn concatenate<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    dimension: i64,
    result_type: T,
    location: L,
) -> DetachedConcatenateOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.concatenate", location)
        .add_operands(inputs)
        .add_attribute(DIMENSION_ATTRIBUTE, context.integer_attribute(context.index_type(), dimension))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::concatenate`")
}

/// Name of the sparse tensor slice dimension attribute.
pub const DIM_ATTRIBUTE: &str = "dim";

/// Operation trait for `sparse_tensor.slice.offset`.
pub trait SliceOffsetOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor slice being queried.
    fn slice(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor dimension being queried.
    fn dimension(&self) -> i64 {
        self.attribute(DIM_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
            .unwrap_or_else(|| panic!("invalid '{DIM_ATTRIBUTE}' attribute in `sparse_tensor.slice.offset`"))
    }

    /// Returns the slice offset.
    fn offset(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(SliceOffset);
mlir_op_trait!(SliceOffset, AlwaysSpeculatable);
mlir_op_trait!(SliceOffset, NoMemoryEffect);
mlir_op_trait!(SliceOffset, OneResult);
mlir_op_trait!(SliceOffset, Pure);
mlir_op_trait!(SliceOffset, ZeroRegions);
mlir_op_trait!(SliceOffset, ZeroSuccessors);

/// Constructs a new detached/owned [`SliceOffsetOperation`] at the specified [`Location`].
pub fn slice_offset<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    slice: ValueRef<'v, 'c, 't>,
    dimension: i64,
    location: L,
) -> DetachedSliceOffsetOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.slice.offset", location)
        .add_operand(slice)
        .add_attribute(DIM_ATTRIBUTE, context.integer_attribute(context.index_type(), dimension))
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::slice_offset`")
}

/// Operation trait for `sparse_tensor.slice.stride`.
pub trait SliceStrideOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor slice being queried.
    fn slice(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor dimension being queried.
    fn dimension(&self) -> i64 {
        self.attribute(DIM_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
            .unwrap_or_else(|| panic!("invalid '{DIM_ATTRIBUTE}' attribute in `sparse_tensor.slice.stride`"))
    }

    /// Returns the slice stride.
    fn stride(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(SliceStride);
mlir_op_trait!(SliceStride, AlwaysSpeculatable);
mlir_op_trait!(SliceStride, NoMemoryEffect);
mlir_op_trait!(SliceStride, OneResult);
mlir_op_trait!(SliceStride, Pure);
mlir_op_trait!(SliceStride, ZeroRegions);
mlir_op_trait!(SliceStride, ZeroSuccessors);

/// Constructs a new detached/owned [`SliceStrideOperation`] at the specified [`Location`].
pub fn slice_stride<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    slice: ValueRef<'v, 'c, 't>,
    dimension: i64,
    location: L,
) -> DetachedSliceStrideOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.slice.stride", location)
        .add_operand(slice)
        .add_attribute(DIM_ATTRIBUTE, context.integer_attribute(context.index_type(), dimension))
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::slice_stride`")
}

/// Operation trait for `sparse_tensor.storage_specifier.init`.
pub trait StorageSpecifierInitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the optional source storage specifier.
    fn source(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(0)
    }

    /// Returns the initialized storage specifier.
    fn specifier(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(StorageSpecifierInit);
mlir_op_trait!(StorageSpecifierInit, AlwaysSpeculatable);
mlir_op_trait!(StorageSpecifierInit, NoMemoryEffect);
mlir_op_trait!(StorageSpecifierInit, OneResult);
mlir_op_trait!(StorageSpecifierInit, Pure);
mlir_op_trait!(StorageSpecifierInit, ZeroRegions);
mlir_op_trait!(StorageSpecifierInit, ZeroSuccessors);

/// Constructs a new detached/owned [`StorageSpecifierInitOperation`] at the specified [`Location`].
pub fn storage_specifier_init<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: Option<ValueRef<'v, 'c, 't>>,
    result_type: T,
    location: L,
) -> DetachedStorageSpecifierInitOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    let mut builder = OperationBuilder::new("sparse_tensor.storage_specifier.init", location).add_result(result_type);
    if let Some(source) = source {
        builder = builder.add_operand(source);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::storage_specifier_init`")
}

/// Name of the sparse tensor storage-specifier kind attribute.
pub const SPECIFIER_KIND_ATTRIBUTE: &str = "specifierKind";

/// Operation trait for `sparse_tensor.storage_specifier.get`.
pub trait StorageSpecifierGetOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the storage specifier being queried.
    fn specifier(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the storage-specifier field kind.
    fn specifier_kind(&self) -> StorageSpecifierKindAttributeRef<'c, 't> {
        self.attribute(SPECIFIER_KIND_ATTRIBUTE).and_then(|attribute| attribute.cast()).unwrap_or_else(|| {
            panic!("invalid '{SPECIFIER_KIND_ATTRIBUTE}' attribute in `sparse_tensor.storage_specifier.get`")
        })
    }

    /// Returns the optional storage level associated with the queried field.
    fn level(&self) -> Option<i64> {
        self.attribute(LEVEL_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
    }

    /// Returns the queried metadata value.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(StorageSpecifierGet);
mlir_op_trait!(StorageSpecifierGet, AlwaysSpeculatable);
mlir_op_trait!(StorageSpecifierGet, NoMemoryEffect);
mlir_op_trait!(StorageSpecifierGet, OneResult);
mlir_op_trait!(StorageSpecifierGet, Pure);
mlir_op_trait!(StorageSpecifierGet, ZeroRegions);
mlir_op_trait!(StorageSpecifierGet, ZeroSuccessors);

/// Constructs a new detached/owned [`StorageSpecifierGetOperation`] at the specified [`Location`].
pub fn storage_specifier_get<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    specifier: ValueRef<'v, 'c, 't>,
    specifier_kind: StorageSpecifierKind,
    level: Option<i64>,
    location: L,
) -> DetachedStorageSpecifierGetOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    let mut builder = OperationBuilder::new("sparse_tensor.storage_specifier.get", location)
        .add_operand(specifier)
        .add_attribute(SPECIFIER_KIND_ATTRIBUTE, context.sparse_tensor_storage_specifier_kind_attribute(specifier_kind))
        .add_result(context.index_type());
    if let Some(level) = level {
        builder = builder.add_attribute(LEVEL_ATTRIBUTE, context.integer_attribute(context.index_type(), level));
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::storage_specifier_get`")
}

/// Operation trait for `sparse_tensor.storage_specifier.set`.
pub trait StorageSpecifierSetOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the storage specifier being updated.
    fn specifier(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the storage-specifier field kind.
    fn specifier_kind(&self) -> StorageSpecifierKindAttributeRef<'c, 't> {
        self.attribute(SPECIFIER_KIND_ATTRIBUTE).and_then(|attribute| attribute.cast()).unwrap_or_else(|| {
            panic!("invalid '{SPECIFIER_KIND_ATTRIBUTE}' attribute in `sparse_tensor.storage_specifier.set`")
        })
    }

    /// Returns the optional storage level associated with the updated field.
    fn level(&self) -> Option<i64> {
        self.attribute(LEVEL_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
    }

    /// Returns the value written into the storage specifier.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the updated storage specifier.
    fn result(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(StorageSpecifierSet);
mlir_op_trait!(StorageSpecifierSet, AlwaysSpeculatable);
mlir_op_trait!(StorageSpecifierSet, NoMemoryEffect);
mlir_op_trait!(StorageSpecifierSet, OneResult);
mlir_op_trait!(StorageSpecifierSet, Pure);
mlir_op_trait!(StorageSpecifierSet, ZeroRegions);
mlir_op_trait!(StorageSpecifierSet, ZeroSuccessors);

/// Constructs a new detached/owned [`StorageSpecifierSetOperation`] at the specified [`Location`].
pub fn storage_specifier_set<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    specifier: ValueRef<'v, 'c, 't>,
    specifier_kind: StorageSpecifierKind,
    level: Option<i64>,
    value: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedStorageSpecifierSetOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    let mut builder = OperationBuilder::new("sparse_tensor.storage_specifier.set", location)
        .add_operand(specifier)
        .add_operand(value)
        .add_attribute(SPECIFIER_KIND_ATTRIBUTE, context.sparse_tensor_storage_specifier_kind_attribute(specifier_kind))
        .add_result(specifier.r#type());
    if let Some(level) = level {
        builder = builder.add_attribute(LEVEL_ATTRIBUTE, context.integer_attribute(context.index_type(), level));
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::storage_specifier_set`")
}

/// Operation trait for `sparse_tensor.lvl`.
pub trait LevelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being queried.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the level index operand.
    fn index(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the level size.
    fn size(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Level);
mlir_op_trait!(Level, NoMemoryEffect);
mlir_op_trait!(Level, OneResult);
mlir_op_trait!(Level, ZeroRegions);
mlir_op_trait!(Level, ZeroSuccessors);

/// Constructs a new detached/owned [`LevelOperation`] at the specified [`Location`].
pub fn level<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    index: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedLevelOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.lvl", location)
        .add_operand(source)
        .add_operand(index)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::level`")
}

/// Name of the sparse tensor coordinate-translation direction attribute.
pub const DIRECTION_ATTRIBUTE: &str = "direction";

/// Name of the sparse tensor encoding attribute used by coordinate translation.
pub const ENCODER_ATTRIBUTE: &str = "encoder";

/// Operation trait for `sparse_tensor.crd_translate`.
pub trait CoordinateTranslateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input coordinates.
    fn input_coordinates(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the translation direction.
    fn direction(&self) -> CoordinateTranslationDirectionAttributeRef<'c, 't> {
        self.attribute(DIRECTION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast())
            .unwrap_or_else(|| panic!("invalid '{DIRECTION_ATTRIBUTE}' attribute in `sparse_tensor.crd_translate`"))
    }

    /// Returns the sparse tensor encoding that defines the coordinate maps.
    fn encoder(&self) -> SparseTensorEncodingAttributeRef<'c, 't> {
        self.attribute(ENCODER_ATTRIBUTE)
            .and_then(|attribute| attribute.cast())
            .unwrap_or_else(|| panic!("invalid '{ENCODER_ATTRIBUTE}' attribute in `sparse_tensor.crd_translate`"))
    }

    /// Returns the translated coordinates.
    fn output_coordinates(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (0..self.result_count()).map(|index| self.result(index).unwrap().as_ref()).collect()
    }
}

mlir_op!(CoordinateTranslate);
mlir_op_trait!(CoordinateTranslate, AlwaysSpeculatable);
mlir_op_trait!(CoordinateTranslate, NoMemoryEffect);
mlir_op_trait!(CoordinateTranslate, Pure);
mlir_op_trait!(CoordinateTranslate, ZeroRegions);
mlir_op_trait!(CoordinateTranslate, ZeroSuccessors);

/// Constructs a new detached/owned [`CoordinateTranslateOperation`] at the specified [`Location`].
pub fn coordinate_translate<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    input_coordinates: &[ValueRef<'v, 'c, 't>],
    direction: CoordinateTranslationDirection,
    encoder: SparseTensorEncodingAttributeRef<'c, 't>,
    output_count: usize,
    location: L,
) -> DetachedCoordinateTranslateOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    let output_types = (0..output_count).map(|_| context.index_type()).collect::<Vec<_>>();
    OperationBuilder::new("sparse_tensor.crd_translate", location)
        .add_operands(input_coordinates)
        .add_attribute(DIRECTION_ATTRIBUTE, context.sparse_tensor_coordinate_translation_direction_attribute(direction))
        .add_attribute(ENCODER_ATTRIBUTE, encoder)
        .add_results(&output_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::coordinate_translate`")
}

/// Name of the sparse tensor in-bounds insertion attribute.
pub const INBOUNDS_ATTRIBUTE: &str = "inbounds";

/// Operation trait for `sparse_tensor.push_back`.
pub trait PushBackOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the current logical size of the input buffer.
    fn current_size(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the input buffer.
    fn input_buffer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the value being appended.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the optional append count.
    fn count(&self) -> Option<ValueRef<'o, 'c, 't>> {
        if self.operand_count() > 3 { self.operand_value(3) } else { None }
    }

    /// Returns whether the operation is marked as in-bounds.
    fn inbounds(&self) -> bool {
        self.has_attribute(INBOUNDS_ATTRIBUTE)
    }

    /// Returns the output buffer.
    fn output_buffer(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the new logical size of the output buffer.
    fn new_size(&self) -> ValueRef<'o, 'c, 't> {
        self.result(1).unwrap().as_ref()
    }
}

mlir_op!(PushBack);
mlir_op_trait!(PushBack, ZeroRegions);
mlir_op_trait!(PushBack, ZeroSuccessors);

/// Constructs a new detached/owned [`PushBackOperation`] at the specified [`Location`].
pub fn push_back<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    current_size: ValueRef<'v, 'c, 't>,
    input_buffer: ValueRef<'v, 'c, 't>,
    value: ValueRef<'v, 'c, 't>,
    count: Option<ValueRef<'v, 'c, 't>>,
    inbounds: bool,
    location: L,
) -> DetachedPushBackOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    let mut builder = OperationBuilder::new("sparse_tensor.push_back", location)
        .add_operand(current_size)
        .add_operand(input_buffer)
        .add_operand(value)
        .add_result(input_buffer.r#type())
        .add_result(current_size.r#type());
    if let Some(count) = count {
        builder = builder.add_operand(count);
    }
    if inbounds {
        builder = builder.add_attribute(INBOUNDS_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::push_back`")
}

/// Operation trait for `sparse_tensor.expand`.
pub trait ExpandOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the sparse tensor being expanded.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the expanded values buffer.
    fn values(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the boolean filled buffer.
    fn filled(&self) -> ValueRef<'o, 'c, 't> {
        self.result(1).unwrap().as_ref()
    }

    /// Returns the added-coordinates buffer.
    fn added(&self) -> ValueRef<'o, 'c, 't> {
        self.result(2).unwrap().as_ref()
    }

    /// Returns the number of added coordinates.
    fn count(&self) -> ValueRef<'o, 'c, 't> {
        self.result(3).unwrap().as_ref()
    }
}

mlir_op!(Expand);
mlir_op_trait!(Expand, ZeroRegions);
mlir_op_trait!(Expand, ZeroSuccessors);

/// Constructs a new detached/owned [`ExpandOperation`] at the specified [`Location`].
pub fn expand<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    values_type: TypeRef<'c, 't>,
    filled_type: TypeRef<'c, 't>,
    added_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedExpandOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.expand", location)
        .add_operand(tensor)
        .add_result(values_type)
        .add_result(filled_type)
        .add_result(added_type)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::expand`")
}

/// Operation trait for `sparse_tensor.compress`.
pub trait CompressOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the expanded values buffer.
    fn values(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the boolean filled buffer.
    fn filled(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the added-coordinates buffer.
    fn added(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the number of added coordinates.
    fn count(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns the sparse tensor being compressed into.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(4).unwrap()
    }

    /// Returns the outer level coordinates.
    fn level_coordinates(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(5).collect()
    }

    /// Returns the updated sparse tensor.
    fn result(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Compress);
mlir_op_trait!(Compress, OneResult);
mlir_op_trait!(Compress, ZeroRegions);
mlir_op_trait!(Compress, ZeroSuccessors);

/// Constructs a new detached/owned [`CompressOperation`] at the specified [`Location`].
pub fn compress<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    values: ValueRef<'v, 'c, 't>,
    filled: ValueRef<'v, 'c, 't>,
    added: ValueRef<'v, 'c, 't>,
    count: ValueRef<'v, 'c, 't>,
    tensor: ValueRef<'v, 'c, 't>,
    level_coordinates: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedCompressOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.compress", location)
        .add_operand(values)
        .add_operand(filled)
        .add_operand(added)
        .add_operand(count)
        .add_operand(tensor)
        .add_operands(level_coordinates)
        .add_result(tensor.r#type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::compress`")
}

/// Name of the sparse tensor insertion-finalization attribute.
pub const HAS_INSERTS_ATTRIBUTE: &str = "hasInserts";

/// Operation trait for `sparse_tensor.load`.
pub trait LoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being loaded.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns whether underlying insertions must be finalized.
    fn has_inserts(&self) -> bool {
        self.has_attribute(HAS_INSERTS_ATTRIBUTE)
    }

    /// Returns the loaded tensor.
    fn result(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Load);
mlir_op_trait!(Load, OneResult);
mlir_op_trait!(Load, ZeroRegions);
mlir_op_trait!(Load, ZeroSuccessors);

/// Constructs a new detached/owned [`LoadOperation`] at the specified [`Location`].
pub fn load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    has_inserts: bool,
    location: L,
) -> DetachedLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    let mut builder = OperationBuilder::new("sparse_tensor.load", location)
        .add_operand(tensor)
        .add_result(tensor.r#type());
    if has_inserts {
        builder = builder.add_attribute(HAS_INSERTS_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::load`")
}

/// Operation trait for `sparse_tensor.out`.
pub trait OutOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the sparse tensor being output.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the output destination.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }
}

mlir_op!(Out);
mlir_op_trait!(Out, ZeroRegions);
mlir_op_trait!(Out, ZeroSuccessors);

/// Constructs a new detached/owned [`OutOperation`] at the specified [`Location`].
pub fn out<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedOutOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.out", location)
        .add_operand(tensor)
        .add_operand(destination)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::out`")
}

/// Name of the sparse tensor sort permutation-map attribute.
pub const PERMUTATION_MAP_ATTRIBUTE: &str = "perm_map";

/// Name of the sparse tensor sort payload-count attribute.
pub const NY_ATTRIBUTE: &str = "ny";

/// Name of the sparse tensor sorting algorithm attribute.
pub const ALGORITHM_ATTRIBUTE: &str = "algorithm";

/// Operation trait for `sparse_tensor.sort`.
pub trait SortOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the number of entries being sorted.
    fn count(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the coordinate and value buffer being sorted.
    fn coordinates_and_values(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the payload buffers being sorted jointly.
    fn payloads(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(2).collect()
    }

    /// Returns the coordinate permutation map.
    fn permutation_map(&self) -> AffineMap<'c, 't> {
        self.attribute(PERMUTATION_MAP_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<AffineMapAttributeRef>())
            .map(|attribute| attribute.affine_map())
            .unwrap_or_else(|| panic!("invalid '{PERMUTATION_MAP_ATTRIBUTE}' attribute in `sparse_tensor.sort`"))
    }

    /// Returns the optional number of payload values carried by the packed buffer.
    fn payload_count(&self) -> Option<i64> {
        self.attribute(NY_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
    }

    /// Returns the sorting algorithm.
    fn algorithm(&self) -> SortKindAttributeRef<'c, 't> {
        self.attribute(ALGORITHM_ATTRIBUTE)
            .and_then(|attribute| attribute.cast())
            .unwrap_or_else(|| panic!("invalid '{ALGORITHM_ATTRIBUTE}' attribute in `sparse_tensor.sort`"))
    }
}

mlir_op!(Sort);
mlir_op_trait!(Sort, ZeroRegions);
mlir_op_trait!(Sort, ZeroSuccessors);

/// Constructs a new detached/owned [`SortOperation`] at the specified [`Location`].
pub fn sort<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    count: ValueRef<'v, 'c, 't>,
    coordinates_and_values: ValueRef<'v, 'c, 't>,
    payloads: &[ValueRef<'v, 'c, 't>],
    permutation_map: AffineMap<'c, 't>,
    payload_count: Option<i64>,
    algorithm: SortKind,
    location: L,
) -> DetachedSortOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    let mut builder = OperationBuilder::new("sparse_tensor.sort", location)
        .add_operand(count)
        .add_operand(coordinates_and_values)
        .add_operands(payloads)
        .add_attribute(PERMUTATION_MAP_ATTRIBUTE, context.affine_map_attribute(permutation_map))
        .add_attribute(ALGORITHM_ATTRIBUTE, context.sparse_tensor_sort_kind_attribute(algorithm));
    if let Some(payload_count) = payload_count {
        builder = builder.add_attribute(NY_ATTRIBUTE, context.integer_attribute(context.index_type(), payload_count));
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::sort`")
}

/// Operation trait for `sparse_tensor.reorder_coo`.
pub trait ReorderCooOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the input COO tensor.
    fn input_coo(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the sorting algorithm.
    fn algorithm(&self) -> SortKindAttributeRef<'c, 't> {
        self.attribute(ALGORITHM_ATTRIBUTE)
            .and_then(|attribute| attribute.cast())
            .unwrap_or_else(|| panic!("invalid '{ALGORITHM_ATTRIBUTE}' attribute in `sparse_tensor.reorder_coo`"))
    }

    /// Returns the reordered COO tensor.
    fn result_coo(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(ReorderCoo);
mlir_op_trait!(ReorderCoo, AlwaysSpeculatable);
mlir_op_trait!(ReorderCoo, NoMemoryEffect);
mlir_op_trait!(ReorderCoo, OneResult);
mlir_op_trait!(ReorderCoo, Pure);
mlir_op_trait!(ReorderCoo, ZeroRegions);
mlir_op_trait!(ReorderCoo, ZeroSuccessors);

/// Constructs a new detached/owned [`ReorderCooOperation`] at the specified [`Location`].
pub fn reorder_coo<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    input_coo: ValueRef<'v, 'c, 't>,
    algorithm: SortKind,
    result_type: T,
    location: L,
) -> DetachedReorderCooOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.reorder_coo", location)
        .add_operand(input_coo)
        .add_attribute(ALGORITHM_ATTRIBUTE, context.sparse_tensor_sort_kind_attribute(algorithm))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::reorder_coo`")
}

/// Name of the sparse tensor binary left-identity attribute.
pub const LEFT_IDENTITY_ATTRIBUTE: &str = "left_identity";

/// Name of the sparse tensor binary right-identity attribute.
pub const RIGHT_IDENTITY_ATTRIBUTE: &str = "right_identity";

/// Operation trait for `sparse_tensor.binary`.
pub trait BinaryOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the left-hand input value.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the right-hand input value.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns whether the left-only case is the identity function.
    fn left_identity(&self) -> bool {
        self.has_attribute(LEFT_IDENTITY_ATTRIBUTE)
    }

    /// Returns whether the right-only case is the identity function.
    fn right_identity(&self) -> bool {
        self.has_attribute(RIGHT_IDENTITY_ATTRIBUTE)
    }

    /// Returns the overlap region.
    fn overlap_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns the left-only region.
    fn left_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(1).unwrap()
    }

    /// Returns the right-only region.
    fn right_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(2).unwrap()
    }

    /// Returns the binary operation result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        OneResult::output(self)
    }
}

mlir_op!(Binary);
mlir_op_trait!(Binary, AlwaysSpeculatable);
mlir_op_trait!(Binary, NoMemoryEffect);
mlir_op_trait!(Binary, OneResult);
mlir_op_trait!(Binary, Pure);
mlir_op_trait!(Binary, ZeroSuccessors);

/// Constructs a new detached/owned [`BinaryOperation`] at the specified [`Location`].
pub fn binary<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    lhs: ValueRef<'v, 'c, 't>,
    rhs: ValueRef<'v, 'c, 't>,
    result_type: T,
    left_identity: bool,
    right_identity: bool,
    overlap_region: DetachedRegion<'c, 't>,
    left_region: DetachedRegion<'c, 't>,
    right_region: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedBinaryOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    let mut builder = OperationBuilder::new("sparse_tensor.binary", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .add_result(result_type)
        .add_region(overlap_region)
        .add_region(left_region)
        .add_region(right_region);
    if left_identity {
        builder = builder.add_attribute(LEFT_IDENTITY_ATTRIBUTE, context.unit_attribute());
    }
    if right_identity {
        builder = builder.add_attribute(RIGHT_IDENTITY_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::binary`")
}

/// Operation trait for `sparse_tensor.unary`.
pub trait UnaryOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the present-value region.
    fn present_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns the absent-value region.
    fn absent_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(1).unwrap()
    }

    /// Returns the unary operation result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        OneResult::output(self)
    }
}

mlir_op!(Unary);
mlir_op_trait!(Unary, AlwaysSpeculatable);
mlir_op_trait!(Unary, NoMemoryEffect);
mlir_op_trait!(Unary, OneResult);
mlir_op_trait!(Unary, Pure);
mlir_op_trait!(Unary, ZeroSuccessors);

/// Constructs a new detached/owned [`UnaryOperation`] at the specified [`Location`].
pub fn unary<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    result_type: T,
    present_region: DetachedRegion<'c, 't>,
    absent_region: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedUnaryOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.unary", location)
        .add_operand(input)
        .add_result(result_type)
        .add_region(present_region)
        .add_region(absent_region)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::unary`")
}

/// Operation trait for `sparse_tensor.reduce`.
pub trait ReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the running reduction value.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the next input value.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the reduction identity value.
    fn identity(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the reduction region.
    fn region(&self) -> RegionRef<'o, 'c, 't> {
        Operation::region(self, 0).unwrap()
    }

    /// Returns the reduction result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        OneResult::output(self)
    }
}

mlir_op!(Reduce);
mlir_op_trait!(Reduce, AlwaysSpeculatable);
mlir_op_trait!(Reduce, NoMemoryEffect);
mlir_op_trait!(Reduce, OneRegion);
mlir_op_trait!(Reduce, OneResult);
mlir_op_trait!(Reduce, Pure);
mlir_op_trait!(Reduce, ZeroSuccessors);

/// Constructs a new detached/owned [`ReduceOperation`] at the specified [`Location`].
pub fn reduce<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    lhs: ValueRef<'v, 'c, 't>,
    rhs: ValueRef<'v, 'c, 't>,
    identity: ValueRef<'v, 'c, 't>,
    region: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedReduceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.reduce", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .add_operand(identity)
        .add_result(lhs.r#type())
        .add_region(region)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::reduce`")
}

/// Operation trait for `sparse_tensor.select`.
pub trait SelectOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the predicate region.
    fn region(&self) -> RegionRef<'o, 'c, 't> {
        Operation::region(self, 0).unwrap()
    }

    /// Returns the selected value.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        OneResult::output(self)
    }
}

mlir_op!(Select);
mlir_op_trait!(Select, AlwaysSpeculatable);
mlir_op_trait!(Select, NoMemoryEffect);
mlir_op_trait!(Select, OneRegion);
mlir_op_trait!(Select, OneResult);
mlir_op_trait!(Select, Pure);
mlir_op_trait!(Select, ZeroSuccessors);

/// Constructs a new detached/owned [`SelectOperation`] at the specified [`Location`].
pub fn select<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    region: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedSelectOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.select", location)
        .add_operand(input)
        .add_result(input.r#type())
        .add_region(region)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::select`")
}

/// Operation trait for `sparse_tensor.yield`.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the yielded values.
    fn values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, AlwaysSpeculatable);
mlir_op_trait!(Yield, IsTerminator);
mlir_op_trait!(Yield, NoMemoryEffect);
mlir_op_trait!(Yield, Pure);
mlir_op_trait!(Yield, ReturnLike);
mlir_op_trait!(Yield, SingleBlockRegions);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Constructs a new detached/owned [`YieldOperation`] at the specified [`Location`].
pub fn r#yield<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedYieldOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.yield", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::yield`")
}

/// Name of the sparse tensor foreach order attribute.
pub const ORDER_ATTRIBUTE: &str = "order";

/// Operation trait for `sparse_tensor.foreach`.
pub trait ForeachOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor being iterated over.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the initial loop-carried values.
    fn initial_values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the optional dense-tensor traversal order.
    fn order(&self) -> Option<AffineMap<'c, 't>> {
        self.attribute(ORDER_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<AffineMapAttributeRef>())
            .map(|attribute| attribute.affine_map())
    }

    /// Returns the final loop-carried values.
    fn final_values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (0..self.result_count()).map(|index| self.result(index).unwrap().as_ref()).collect()
    }

    /// Returns the foreach body region.
    fn body_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(Foreach);
mlir_op_trait!(Foreach, OneRegion);
mlir_op_trait!(Foreach, SingleBlock);
mlir_op_trait!(Foreach, SingleBlockRegions);
mlir_op_trait!(Foreach, ZeroSuccessors);

/// Constructs a new detached/owned [`ForeachOperation`] at the specified [`Location`].
pub fn foreach<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    initial_values: &[ValueRef<'v, 'c, 't>],
    order: Option<AffineMap<'c, 't>>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedForeachOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    let result_types = initial_values.iter().map(|value| value.r#type()).collect::<Vec<_>>();
    let mut builder = OperationBuilder::new("sparse_tensor.foreach", location)
        .add_operand(tensor)
        .add_operands(initial_values)
        .add_results(&result_types)
        .add_region(body);
    if let Some(order) = order {
        builder = builder.add_attribute(ORDER_ATTRIBUTE, context.affine_map_attribute(order));
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::foreach`")
}

/// Operation trait for `sparse_tensor.extract_iteration_space`.
pub trait ExtractIterationSpaceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor defining the iteration space.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional parent iterator.
    fn parent_iterator(&self) -> Option<ValueRef<'o, 'c, 't>> {
        if self.operand_count() > 1 { self.operand_value(1) } else { None }
    }

    /// Returns the inclusive lower storage level of the iteration space.
    fn lower_level(&self) -> i64 {
        self.attribute(LOWER_LEVEL_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
            .unwrap_or_else(|| {
                panic!("invalid '{LOWER_LEVEL_ATTRIBUTE}' attribute in `sparse_tensor.extract_iteration_space`")
            })
    }

    /// Returns the exclusive upper storage level of the iteration space.
    fn upper_level(&self) -> i64 {
        self.attribute(UPPER_LEVEL_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
            .unwrap_or_else(|| {
                panic!("invalid '{UPPER_LEVEL_ATTRIBUTE}' attribute in `sparse_tensor.extract_iteration_space`")
            })
    }

    /// Returns the extracted iteration space.
    fn iteration_space(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

/// Name of the sparse tensor iteration-space lower-level attribute.
pub const LOWER_LEVEL_ATTRIBUTE: &str = "loLvl";

/// Name of the sparse tensor iteration-space upper-level attribute.
pub const UPPER_LEVEL_ATTRIBUTE: &str = "hiLvl";

mlir_op!(ExtractIterationSpace);
mlir_op_trait!(ExtractIterationSpace, AlwaysSpeculatable);
mlir_op_trait!(ExtractIterationSpace, NoMemoryEffect);
mlir_op_trait!(ExtractIterationSpace, OneResult);
mlir_op_trait!(ExtractIterationSpace, Pure);
mlir_op_trait!(ExtractIterationSpace, ZeroRegions);
mlir_op_trait!(ExtractIterationSpace, ZeroSuccessors);

/// Constructs a new detached/owned [`ExtractIterationSpaceOperation`] at the specified [`Location`].
pub fn extract_iteration_space<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    parent_iterator: Option<ValueRef<'v, 'c, 't>>,
    lower_level: i64,
    upper_level: i64,
    result_type: T,
    location: L,
) -> DetachedExtractIterationSpaceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    let mut builder = OperationBuilder::new("sparse_tensor.extract_iteration_space", location)
        .add_operand(tensor)
        .add_attribute(LOWER_LEVEL_ATTRIBUTE, context.integer_attribute(context.index_type(), lower_level))
        .add_attribute(UPPER_LEVEL_ATTRIBUTE, context.integer_attribute(context.index_type(), upper_level))
        .add_result(result_type);
    if let Some(parent_iterator) = parent_iterator {
        builder = builder.add_operand(parent_iterator);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::extract_iteration_space`")
}

/// Operation trait for `sparse_tensor.extract_value`.
pub trait ExtractValueOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being read.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the sparse iterator pointing at the stored value.
    fn iterator(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the extracted value.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(ExtractValue);
mlir_op_trait!(ExtractValue, AlwaysSpeculatable);
mlir_op_trait!(ExtractValue, NoMemoryEffect);
mlir_op_trait!(ExtractValue, OneResult);
mlir_op_trait!(ExtractValue, Pure);
mlir_op_trait!(ExtractValue, ZeroRegions);
mlir_op_trait!(ExtractValue, ZeroSuccessors);

/// Constructs a new detached/owned [`ExtractValueOperation`] at the specified [`Location`].
pub fn extract_value<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    iterator: ValueRef<'v, 'c, 't>,
    result_type: T,
    location: L,
) -> DetachedExtractValueOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.extract_value", location)
        .add_operand(tensor)
        .add_operand(iterator)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::extract_value`")
}

/// Name of the sparse tensor coordinate-used-levels bitset attribute.
pub const COORDINATE_USED_LEVELS_ATTRIBUTE: &str = "crdUsedLvls";

/// Operation trait for `sparse_tensor.iterate`.
pub trait IterateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the sparse iteration space.
    fn iteration_space(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the initial loop-carried values.
    fn initial_values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the bitset of coordinate levels used by the region.
    fn coordinate_used_levels(&self) -> u64 {
        self.attribute(COORDINATE_USED_LEVELS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.unsigned_value())
            .unwrap_or_else(|| {
                panic!("invalid '{COORDINATE_USED_LEVELS_ATTRIBUTE}' attribute in `sparse_tensor.iterate`")
            })
    }

    /// Returns the final loop-carried values.
    fn final_values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (0..self.result_count()).map(|index| self.result(index).unwrap().as_ref()).collect()
    }

    /// Returns the loop body region.
    fn body_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(Iterate);
mlir_op_trait!(Iterate, OneRegion);
mlir_op_trait!(Iterate, SingleBlock);
mlir_op_trait!(Iterate, SingleBlockRegions);
mlir_op_trait!(Iterate, ZeroSuccessors);

/// Constructs a new detached/owned [`IterateOperation`] at the specified [`Location`].
pub fn iterate<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    iteration_space: ValueRef<'v, 'c, 't>,
    initial_values: &[ValueRef<'v, 'c, 't>],
    coordinate_used_levels: u64,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedIterateOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    let result_types = initial_values.iter().map(|value| value.r#type()).collect::<Vec<_>>();
    OperationBuilder::new("sparse_tensor.iterate", location)
        .add_operand(iteration_space)
        .add_operands(initial_values)
        .add_attribute(
            COORDINATE_USED_LEVELS_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), coordinate_used_levels as i64),
        )
        .add_results(&result_types)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::iterate`")
}

/// Name of the sparse tensor coiterate case bitset array attribute.
pub const CASES_ATTRIBUTE: &str = "cases";

/// Name of the operand segment-size attribute used by SparseTensor operations with multiple variadic operand groups.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Operation trait for `sparse_tensor.coiterate`.
pub trait CoIterateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the sparse iteration spaces being co-iterated.
    fn iteration_spaces(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `sparse_tensor.coiterate`")
            });
        (0..segment_sizes[0] as usize).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the initial loop-carried values.
    fn initial_values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `sparse_tensor.coiterate`")
            });
        let start = segment_sizes[0] as usize;
        let end = start + segment_sizes[1] as usize;
        (start..end).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the bitset of coordinate levels used by the regions.
    fn coordinate_used_levels(&self) -> u64 {
        self.attribute(COORDINATE_USED_LEVELS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.unsigned_value())
            .unwrap_or_else(|| {
                panic!("invalid '{COORDINATE_USED_LEVELS_ATTRIBUTE}' attribute in `sparse_tensor.coiterate`")
            })
    }

    /// Returns the case bitsets in region order.
    fn cases(&self) -> Vec<u64> {
        self.attribute(CASES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
            .map(|attribute| {
                attribute
                    .elements()
                    .map(|element| element.cast::<IntegerAttributeRef>().unwrap().unsigned_value())
                    .collect()
            })
            .unwrap_or_else(|| panic!("invalid '{CASES_ATTRIBUTE}' attribute in `sparse_tensor.coiterate`"))
    }

    /// Returns the final loop-carried values.
    fn final_values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (0..self.result_count()).map(|index| self.result(index).unwrap().as_ref()).collect()
    }

    /// Returns the case regions.
    fn case_regions(&self) -> Vec<RegionRef<'o, 'c, 't>> {
        (0..self.region_count()).map(|index| self.region(index).unwrap()).collect()
    }
}

mlir_op!(CoIterate);
mlir_op_trait!(CoIterate, SingleBlockRegions);
mlir_op_trait!(CoIterate, ZeroSuccessors);

/// Constructs a new detached/owned [`CoIterateOperation`] at the specified [`Location`].
pub fn coiterate<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    iteration_spaces: &[ValueRef<'v, 'c, 't>],
    initial_values: &[ValueRef<'v, 'c, 't>],
    coordinate_used_levels: u64,
    cases: &[u64],
    case_regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> DetachedCoIterateOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    let result_types = initial_values.iter().map(|value| value.r#type()).collect::<Vec<_>>();
    let cases = cases
        .iter()
        .map(|case| context.integer_attribute(context.signless_integer_type(64), *case as i64))
        .collect::<Vec<_>>();
    OperationBuilder::new("sparse_tensor.coiterate", location)
        .add_operands(iteration_spaces)
        .add_operands(initial_values)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[iteration_spaces.len() as i32, initial_values.len() as i32])
                .unwrap(),
        )
        .add_attribute(
            COORDINATE_USED_LEVELS_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), coordinate_used_levels as i64),
        )
        .add_attribute(CASES_ATTRIBUTE, context.array_attribute(&cases))
        .add_results(&result_types)
        .add_regions(case_regions)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::coiterate`")
}

/// Operation trait for `sparse_tensor.print`.
pub trait PrintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the sparse tensor being printed.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(Print);
mlir_op_trait!(Print, ZeroRegions);
mlir_op_trait!(Print, ZeroSuccessors);

/// Constructs a new detached/owned [`PrintOperation`] at the specified [`Location`].
pub fn print<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedPrintOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.print", location)
        .add_operand(tensor)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::print`")
}

/// Operation trait for `sparse_tensor.has_runtime_library`.
pub trait HasRuntimeLibraryOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns whether the sparse tensor runtime library is enabled.
    fn enabled(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(HasRuntimeLibrary);
mlir_op_trait!(HasRuntimeLibrary, OneResult);
mlir_op_trait!(HasRuntimeLibrary, ZeroOperands);
mlir_op_trait!(HasRuntimeLibrary, ZeroRegions);
mlir_op_trait!(HasRuntimeLibrary, ZeroSuccessors);

/// Constructs a new detached/owned [`HasRuntimeLibraryOperation`] at the specified [`Location`].
pub fn has_runtime_library<'c, 't: 'c, L: Location<'c, 't>>(location: L) -> DetachedHasRuntimeLibraryOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor());
    OperationBuilder::new("sparse_tensor.has_runtime_library", location)
        .add_result(context.signless_integer_type(1))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `sparse_tensor::has_runtime_library`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context};

    use super::*;

    #[test]
    fn test_has_runtime_library() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let result_type = context.signless_integer_type(1);
        module.body().append_operation({
            let mut block = context.block(&[(result_type, location)]);
            let op = has_runtime_library(location);
            assert_eq!(op.enabled().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "has_runtime_library_test",
                func::FuncAttributes {
                    arguments: vec![result_type.into()],
                    results: vec![result_type.into()],
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
                  func.func @has_runtime_library_test(%arg0: i1) -> i1 {
                    %0 = sparse_tensor.has_runtime_library : i1
                    return %0 : i1
                  }
                }
            "},
        );
    }
}
