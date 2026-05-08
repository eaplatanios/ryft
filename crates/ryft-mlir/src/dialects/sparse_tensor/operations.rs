use crate::{
    AffineMap, Attribute, DetachedOp, DetachedRegion, DialectHandle, Error, IntegerAttributeRef, Location, OneResult,
    Operation, OperationBuilder, RegionRef, Type, TypeRef, Value, ValueRef, mlir_op, mlir_op_trait,
};

use super::{
    CoordinateTranslationDirection, CoordinateTranslationDirectionAttributeRef, SortKind, SortKindAttributeRef,
    SparseTensorEncodingAttributeRef, StorageSpecifierKind, StorageSpecifierKindAttributeRef,
};

/// Operation trait for `sparse_tensor.new`.
pub trait NewOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source value used to materialize the sparse tensor.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the materialized sparse tensor.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedNewOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.new", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::new`"))
        })
}

/// Operation trait for `sparse_tensor.assemble`.
pub trait AssembleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the level storage tensors.
    fn levels(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let level_count = self.operand_count().checked_sub(1).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing values operand in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })?;
        (0..level_count).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the values tensor.
    fn values(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let index = self.operand_count().checked_sub(1).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing values operand in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })?;
        self.operand_value(index)
    }

    /// Returns the assembled sparse tensor.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedAssembleOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.assemble", location)
        .add_operands(levels)
        .add_operand(values)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::assemble`"))
        })
}

/// Operation trait for `sparse_tensor.disassemble`.
pub trait DisassembleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the sparse tensor being disassembled.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the output level buffers.
    fn output_levels(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let level_count = self.result_count().checked_sub(2).ok_or_else(|| {
            Error::invalid_argument(format!(
                "invalid result count in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })? / 2;
        (1..1 + level_count).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the output values buffer.
    fn output_values(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let level_count = self.result_count().checked_sub(2).ok_or_else(|| {
            Error::invalid_argument(format!(
                "invalid result count in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })? / 2;
        self.operand_value(1 + level_count)
    }

    /// Returns the copied level buffers.
    fn returned_levels(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let level_count = self.result_count().checked_sub(2).ok_or_else(|| {
            Error::invalid_argument(format!(
                "invalid result count in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })? / 2;
        (0..level_count).map(|index| Ok(self.result(index)?.as_ref())).collect()
    }

    /// Returns the copied values buffer.
    fn returned_values(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let level_count = self.result_count().checked_sub(2).ok_or_else(|| {
            Error::invalid_argument(format!(
                "invalid result count in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })? / 2;
        Ok(self.result(level_count)?.as_ref())
    }

    /// Returns the occupied lengths of the returned level buffers.
    fn level_lengths(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let level_count = self.result_count().checked_sub(2).ok_or_else(|| {
            Error::invalid_argument(format!(
                "invalid result count in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })? / 2;
        (level_count + 1..level_count + 1 + level_count)
            .map(|index| Ok(self.result(index)?.as_ref()))
            .collect()
    }

    /// Returns the occupied length of the returned values buffer.
    fn values_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let index = self.result_count().checked_sub(1).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing values length result in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })?;
        Ok(self.result(index)?.as_ref())
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
) -> Result<DetachedDisassembleOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.disassemble", location)
        .add_operand(tensor)
        .add_operands(output_levels)
        .add_operand(output_values)
        .add_results(returned_level_types)
        .add_result(returned_values_type)
        .add_results(level_length_types)
        .add_result(values_length_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::disassemble`"))
        })
}

/// Operation trait for `sparse_tensor.convert`.
pub trait ConvertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source tensor.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the converted tensor.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedConvertOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.convert", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::convert`"))
        })
}

/// Operation trait for `sparse_tensor.reinterpret_map`.
pub trait ReinterpretMapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source tensor.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the tensor with reinterpreted dimension and level maps.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedReinterpretMapOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.reinterpret_map", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::reinterpret_map`"))
        })
}

/// Name of the sparse tensor level attribute.
pub const LEVEL_ATTRIBUTE: &str = "level";

/// Operation trait for `sparse_tensor.positions`.
pub trait PositionsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being queried.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the storage level whose positions buffer is queried.
    fn level(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(LEVEL_ATTRIBUTE)?.signless_value())
    }

    /// Returns the positions buffer.
    fn positions(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedPositionsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.positions", location)
        .add_operand(tensor)
        .add_attribute(LEVEL_ATTRIBUTE, context.integer_attribute(context.index_type(), level))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::positions`"))
        })
}

/// Operation trait for `sparse_tensor.coordinates`.
pub trait CoordinatesOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being queried.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the storage level whose coordinates buffer is queried.
    fn level(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(LEVEL_ATTRIBUTE)?.signless_value())
    }

    /// Returns the coordinates buffer.
    fn coordinates(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedCoordinatesOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.coordinates", location)
        .add_operand(tensor)
        .add_attribute(LEVEL_ATTRIBUTE, context.integer_attribute(context.index_type(), level))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::coordinates`"))
        })
}

/// Operation trait for `sparse_tensor.coordinates_buffer`.
pub trait CoordinatesBufferOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being queried.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the linear coordinates buffer.
    fn coordinates_buffer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedCoordinatesBufferOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.coordinates_buffer", location)
        .add_operand(tensor)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::coordinates_buffer`"))
        })
}

/// Operation trait for `sparse_tensor.values`.
pub trait ValuesOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being queried.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the values buffer.
    fn values(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedValuesOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.values", location)
        .add_operand(tensor)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::values`"))
        })
}

/// Operation trait for `sparse_tensor.number_of_entries`.
pub trait NumberOfEntriesOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being queried.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the number of stored entries.
    fn entry_count(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedNumberOfEntriesOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.number_of_entries", location)
        .add_operand(tensor)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::number_of_entries`"))
        })
}

/// Name of the sparse tensor dimension attribute.
pub const DIMENSION_ATTRIBUTE: &str = "dimension";

/// Operation trait for `sparse_tensor.concatenate`.
pub trait ConcatenateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the input tensors being concatenated.
    fn inputs(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns the dimension along which the inputs are concatenated.
    fn dimension(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(DIMENSION_ATTRIBUTE)?.signless_value())
    }

    /// Returns the concatenated tensor.
    fn concatenated(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedConcatenateOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.concatenate", location)
        .add_operands(inputs)
        .add_attribute(DIMENSION_ATTRIBUTE, context.integer_attribute(context.index_type(), dimension))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::concatenate`"))
        })
}

/// Name of the sparse tensor slice dimension attribute.
pub const DIM_ATTRIBUTE: &str = "dim";

/// Operation trait for `sparse_tensor.slice.offset`.
pub trait SliceOffsetOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor slice being queried.
    fn slice(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the tensor dimension being queried.
    fn dimension(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(DIM_ATTRIBUTE)?.signless_value())
    }

    /// Returns the slice offset.
    fn offset(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedSliceOffsetOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.slice.offset", location)
        .add_operand(slice)
        .add_attribute(DIM_ATTRIBUTE, context.integer_attribute(context.index_type(), dimension))
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::slice_offset`"))
        })
}

/// Operation trait for `sparse_tensor.slice.stride`.
pub trait SliceStrideOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor slice being queried.
    fn slice(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the tensor dimension being queried.
    fn dimension(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(DIM_ATTRIBUTE)?.signless_value())
    }

    /// Returns the slice stride.
    fn stride(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedSliceStrideOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.slice.stride", location)
        .add_operand(slice)
        .add_attribute(DIM_ATTRIBUTE, context.integer_attribute(context.index_type(), dimension))
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::slice_stride`"))
        })
}

/// Operation trait for `sparse_tensor.storage_specifier.init`.
pub trait StorageSpecifierInitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the optional source storage specifier.
    fn source(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() == 0 { Ok(None) } else { self.operand_value(0).map(Some) }
    }

    /// Returns the initialized storage specifier.
    fn specifier(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedStorageSpecifierInitOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    let mut builder = OperationBuilder::new("sparse_tensor.storage_specifier.init", location).add_result(result_type);
    if let Some(source) = source {
        builder = builder.add_operand(source);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::storage_specifier_init`"))
    })
}

/// Name of the sparse tensor storage-specifier kind attribute.
pub const SPECIFIER_KIND_ATTRIBUTE: &str = "specifierKind";

/// Operation trait for `sparse_tensor.storage_specifier.get`.
pub trait StorageSpecifierGetOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the storage specifier being queried.
    fn specifier(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the storage-specifier field kind.
    fn specifier_kind(&self) -> Result<StorageSpecifierKindAttributeRef<'c, 't>, Error> {
        self.attribute(SPECIFIER_KIND_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                SPECIFIER_KIND_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the optional storage level associated with the queried field.
    fn level(&self) -> Result<Option<i64>, Error> {
        if self.has_attribute(LEVEL_ATTRIBUTE) {
            self.integer_attribute(LEVEL_ATTRIBUTE).map(|attribute| Some(attribute.signless_value()))
        } else {
            Ok(None)
        }
    }

    /// Returns the queried metadata value.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedStorageSpecifierGetOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    let mut builder = OperationBuilder::new("sparse_tensor.storage_specifier.get", location)
        .add_operand(specifier)
        .add_attribute(
            SPECIFIER_KIND_ATTRIBUTE,
            context.sparse_tensor_storage_specifier_kind_attribute(specifier_kind)?,
        )
        .add_result(context.index_type());
    if let Some(level) = level {
        builder = builder.add_attribute(LEVEL_ATTRIBUTE, context.integer_attribute(context.index_type(), level));
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::storage_specifier_get`"))
    })
}

/// Operation trait for `sparse_tensor.storage_specifier.set`.
pub trait StorageSpecifierSetOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the storage specifier being updated.
    fn specifier(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the storage-specifier field kind.
    fn specifier_kind(&self) -> Result<StorageSpecifierKindAttributeRef<'c, 't>, Error> {
        self.attribute(SPECIFIER_KIND_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                SPECIFIER_KIND_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the optional storage level associated with the updated field.
    fn level(&self) -> Result<Option<i64>, Error> {
        if self.has_attribute(LEVEL_ATTRIBUTE) {
            self.integer_attribute(LEVEL_ATTRIBUTE).map(|attribute| Some(attribute.signless_value()))
        } else {
            Ok(None)
        }
    }

    /// Returns the value written into the storage specifier.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the updated storage specifier.
    fn result(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedStorageSpecifierSetOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    let mut builder = OperationBuilder::new("sparse_tensor.storage_specifier.set", location)
        .add_operand(specifier)
        .add_operand(value)
        .add_attribute(
            SPECIFIER_KIND_ATTRIBUTE,
            context.sparse_tensor_storage_specifier_kind_attribute(specifier_kind)?,
        )
        .add_result(specifier.r#type()?);
    if let Some(level) = level {
        builder = builder.add_attribute(LEVEL_ATTRIBUTE, context.integer_attribute(context.index_type(), level));
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::storage_specifier_set`"))
    })
}

/// Operation trait for `sparse_tensor.lvl`.
pub trait LevelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being queried.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the level index operand.
    fn index(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the level size.
    fn size(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedLevelOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.lvl", location)
        .add_operand(source)
        .add_operand(index)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::level`"))
        })
}

/// Name of the sparse tensor coordinate-translation direction attribute.
pub const DIRECTION_ATTRIBUTE: &str = "direction";

/// Name of the sparse tensor encoding attribute used by coordinate translation.
pub const ENCODER_ATTRIBUTE: &str = "encoder";

/// Operation trait for `sparse_tensor.crd_translate`.
pub trait CoordinateTranslateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input coordinates.
    fn input_coordinates(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns the translation direction.
    fn direction(&self) -> Result<CoordinateTranslationDirectionAttributeRef<'c, 't>, Error> {
        self.attribute(DIRECTION_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                DIRECTION_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the sparse tensor encoding that defines the coordinate maps.
    fn encoder(&self) -> Result<SparseTensorEncodingAttributeRef<'c, 't>, Error> {
        self.attribute(ENCODER_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                ENCODER_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the translated coordinates.
    fn output_coordinates(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (0..self.result_count()).map(|index| Ok(self.result(index)?.as_ref())).collect()
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
) -> Result<DetachedCoordinateTranslateOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    let output_types = (0..output_count).map(|_| context.index_type()).collect::<Vec<_>>();
    OperationBuilder::new("sparse_tensor.crd_translate", location)
        .add_operands(input_coordinates)
        .add_attribute(
            DIRECTION_ATTRIBUTE,
            context.sparse_tensor_coordinate_translation_direction_attribute(direction)?,
        )
        .add_attribute(ENCODER_ATTRIBUTE, encoder)
        .add_results(&output_types)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::coordinate_translate`"))
        })
}

/// Name of the sparse tensor in-bounds insertion attribute.
pub const INBOUNDS_ATTRIBUTE: &str = "inbounds";

/// Operation trait for `sparse_tensor.push_back`.
pub trait PushBackOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the current logical size of the input buffer.
    fn current_size(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the input buffer.
    fn input_buffer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the value being appended.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the optional append count.
    fn count(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() > 3 { self.operand_value(3).map(Some) } else { Ok(None) }
    }

    /// Returns whether the operation is marked as in-bounds.
    fn inbounds(&self) -> bool {
        self.has_attribute(INBOUNDS_ATTRIBUTE)
    }

    /// Returns the output buffer.
    fn output_buffer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the new logical size of the output buffer.
    fn new_size(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(1)?.as_ref())
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
) -> Result<DetachedPushBackOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    let mut builder = OperationBuilder::new("sparse_tensor.push_back", location)
        .add_operand(current_size)
        .add_operand(input_buffer)
        .add_operand(value)
        .add_result(input_buffer.r#type()?)
        .add_result(current_size.r#type()?);
    if let Some(count) = count {
        builder = builder.add_operand(count);
    }
    if inbounds {
        builder = builder.add_attribute(INBOUNDS_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::push_back`"))
    })
}

/// Operation trait for `sparse_tensor.expand`.
pub trait ExpandOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the sparse tensor being expanded.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the expanded values buffer.
    fn values(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the boolean filled buffer.
    fn filled(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(1)?.as_ref())
    }

    /// Returns the added-coordinates buffer.
    fn added(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(2)?.as_ref())
    }

    /// Returns the number of added coordinates.
    fn count(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(3)?.as_ref())
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
) -> Result<DetachedExpandOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.expand", location)
        .add_operand(tensor)
        .add_result(values_type)
        .add_result(filled_type)
        .add_result(added_type)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::expand`"))
        })
}

/// Operation trait for `sparse_tensor.compress`.
pub trait CompressOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the expanded values buffer.
    fn values(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the boolean filled buffer.
    fn filled(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the added-coordinates buffer.
    fn added(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the number of added coordinates.
    fn count(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns the sparse tensor being compressed into.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(4)
    }

    /// Returns the outer level coordinates.
    fn level_coordinates(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().skip(5).collect()
    }

    /// Returns the updated sparse tensor.
    fn result(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedCompressOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.compress", location)
        .add_operand(values)
        .add_operand(filled)
        .add_operand(added)
        .add_operand(count)
        .add_operand(tensor)
        .add_operands(level_coordinates)
        .add_result(tensor.r#type()?)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::compress`"))
        })
}

/// Name of the sparse tensor insertion-finalization attribute.
pub const HAS_INSERTS_ATTRIBUTE: &str = "hasInserts";

/// Operation trait for `sparse_tensor.load`.
pub trait LoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being loaded.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns whether underlying insertions must be finalized.
    fn has_inserts(&self) -> bool {
        self.has_attribute(HAS_INSERTS_ATTRIBUTE)
    }

    /// Returns the loaded tensor.
    fn result(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedLoadOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    let mut builder = OperationBuilder::new("sparse_tensor.load", location)
        .add_operand(tensor)
        .add_result(tensor.r#type()?);
    if has_inserts {
        builder = builder.add_attribute(HAS_INSERTS_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::load`"))
    })
}

/// Operation trait for `sparse_tensor.out`.
pub trait OutOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the sparse tensor being output.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the output destination.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
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
) -> Result<DetachedOutOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.out", location)
        .add_operand(tensor)
        .add_operand(destination)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::out`"))
        })
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
    fn count(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the coordinate and value buffer being sorted.
    fn coordinates_and_values(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the payload buffers being sorted jointly.
    fn payloads(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().skip(2).collect()
    }

    /// Returns the coordinate permutation map.
    fn permutation_map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(PERMUTATION_MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the optional number of payload values carried by the packed buffer.
    fn payload_count(&self) -> Result<Option<i64>, Error> {
        if self.has_attribute(NY_ATTRIBUTE) {
            self.integer_attribute(NY_ATTRIBUTE).map(|attribute| Some(attribute.signless_value()))
        } else {
            Ok(None)
        }
    }

    /// Returns the sorting algorithm.
    fn algorithm(&self) -> Result<SortKindAttributeRef<'c, 't>, Error> {
        self.attribute(ALGORITHM_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                ALGORITHM_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
) -> Result<DetachedSortOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    let mut builder = OperationBuilder::new("sparse_tensor.sort", location)
        .add_operand(count)
        .add_operand(coordinates_and_values)
        .add_operands(payloads)
        .add_attribute(PERMUTATION_MAP_ATTRIBUTE, context.affine_map_attribute(permutation_map))
        .add_attribute(ALGORITHM_ATTRIBUTE, context.sparse_tensor_sort_kind_attribute(algorithm)?);
    if let Some(payload_count) = payload_count {
        builder = builder.add_attribute(NY_ATTRIBUTE, context.integer_attribute(context.index_type(), payload_count));
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::sort`"))
    })
}

/// Operation trait for `sparse_tensor.reorder_coo`.
pub trait ReorderCooOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the input COO tensor.
    fn input_coo(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the sorting algorithm.
    fn algorithm(&self) -> Result<SortKindAttributeRef<'c, 't>, Error> {
        self.attribute(ALGORITHM_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                ALGORITHM_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the reordered COO tensor.
    fn result_coo(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedReorderCooOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.reorder_coo", location)
        .add_operand(input_coo)
        .add_attribute(ALGORITHM_ATTRIBUTE, context.sparse_tensor_sort_kind_attribute(algorithm)?)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::reorder_coo`"))
        })
}

/// Name of the sparse tensor binary left-identity attribute.
pub const LEFT_IDENTITY_ATTRIBUTE: &str = "left_identity";

/// Name of the sparse tensor binary right-identity attribute.
pub const RIGHT_IDENTITY_ATTRIBUTE: &str = "right_identity";

/// Operation trait for `sparse_tensor.binary`.
pub trait BinaryOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the left-hand input value.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right-hand input value.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
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
    fn overlap_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns the left-only region.
    fn left_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(1)
    }

    /// Returns the right-only region.
    fn right_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(2)
    }

    /// Returns the binary operation result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedBinaryOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
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
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::binary`"))
    })
}

/// Operation trait for `sparse_tensor.unary`.
pub trait UnaryOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the present-value region.
    fn present_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns the absent-value region.
    fn absent_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(1)
    }

    /// Returns the unary operation result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedUnaryOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.unary", location)
        .add_operand(input)
        .add_result(result_type)
        .add_region(present_region)
        .add_region(absent_region)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::unary`"))
        })
}

/// Operation trait for `sparse_tensor.reduce`.
pub trait ReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the running reduction value.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the next input value.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the reduction identity value.
    fn identity(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the reduction region.
    fn region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.as_ref().region(0)
    }

    /// Returns the reduction result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedReduceOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.reduce", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .add_operand(identity)
        .add_result(lhs.r#type()?)
        .add_region(region)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::reduce`"))
        })
}

/// Operation trait for `sparse_tensor.select`.
pub trait SelectOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the predicate region.
    fn region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.as_ref().region(0)
    }

    /// Returns the selected value.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedSelectOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.select", location)
        .add_operand(input)
        .add_result(input.r#type()?)
        .add_region(region)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::select`"))
        })
}

/// Operation trait for `sparse_tensor.yield`.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the yielded values.
    fn values(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
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
) -> Result<DetachedYieldOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.yield", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::yield`"))
        })
}

/// Name of the sparse tensor foreach order attribute.
pub const ORDER_ATTRIBUTE: &str = "order";

/// Operation trait for `sparse_tensor.foreach`.
pub trait ForeachOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor being iterated over.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the initial loop-carried values.
    fn initial_values(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the optional dense-tensor traversal order.
    fn order(&self) -> Result<Option<AffineMap<'c, 't>>, Error> {
        if self.has_attribute(ORDER_ATTRIBUTE) {
            self.affine_map_attribute(ORDER_ATTRIBUTE)?.affine_map().map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the final loop-carried values.
    fn final_values(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (0..self.result_count()).map(|index| Ok(self.result(index)?.as_ref())).collect()
    }

    /// Returns the foreach body region.
    fn body_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
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
) -> Result<DetachedForeachOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    let result_types = initial_values.iter().map(|value| value.r#type()).collect::<Result<Vec<_>, _>>()?;
    let mut builder = OperationBuilder::new("sparse_tensor.foreach", location)
        .add_operand(tensor)
        .add_operands(initial_values)
        .add_results(&result_types)
        .add_region(body);
    if let Some(order) = order {
        builder = builder.add_attribute(ORDER_ATTRIBUTE, context.affine_map_attribute(order));
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::foreach`"))
    })
}

/// Operation trait for `sparse_tensor.extract_iteration_space`.
pub trait ExtractIterationSpaceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor defining the iteration space.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional parent iterator.
    fn parent_iterator(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() > 1 { self.operand_value(1).map(Some) } else { Ok(None) }
    }

    /// Returns the inclusive lower storage level of the iteration space.
    fn lower_level(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(LOWER_LEVEL_ATTRIBUTE)?.signless_value())
    }

    /// Returns the exclusive upper storage level of the iteration space.
    fn upper_level(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(UPPER_LEVEL_ATTRIBUTE)?.signless_value())
    }

    /// Returns the extracted iteration space.
    fn iteration_space(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedExtractIterationSpaceOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    let mut builder = OperationBuilder::new("sparse_tensor.extract_iteration_space", location)
        .add_operand(tensor)
        .add_attribute(LOWER_LEVEL_ATTRIBUTE, context.integer_attribute(context.index_type(), lower_level))
        .add_attribute(UPPER_LEVEL_ATTRIBUTE, context.integer_attribute(context.index_type(), upper_level))
        .add_result(result_type);
    if let Some(parent_iterator) = parent_iterator {
        builder = builder.add_operand(parent_iterator);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::extract_iteration_space`"))
    })
}

/// Operation trait for `sparse_tensor.extract_value`.
pub trait ExtractValueOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the sparse tensor being read.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the sparse iterator pointing at the stored value.
    fn iterator(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the extracted value.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedExtractValueOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.extract_value", location)
        .add_operand(tensor)
        .add_operand(iterator)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::extract_value`"))
        })
}

/// Name of the sparse tensor coordinate-used-levels bitset attribute.
pub const COORDINATE_USED_LEVELS_ATTRIBUTE: &str = "crdUsedLvls";

/// Operation trait for `sparse_tensor.iterate`.
pub trait IterateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the sparse iteration space.
    fn iteration_space(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the initial loop-carried values.
    fn initial_values(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the bitset of coordinate levels used by the region.
    fn coordinate_used_levels(&self) -> Result<u64, Error> {
        Ok(self.integer_attribute(COORDINATE_USED_LEVELS_ATTRIBUTE)?.unsigned_value())
    }

    /// Returns the final loop-carried values.
    fn final_values(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (0..self.result_count()).map(|index| Ok(self.result(index)?.as_ref())).collect()
    }

    /// Returns the loop body region.
    fn body_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
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
) -> Result<DetachedIterateOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    let result_types = initial_values.iter().map(|value| value.r#type()).collect::<Result<Vec<_>, _>>()?;
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
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::iterate`"))
        })
}

/// Name of the sparse tensor coiterate case bitset array attribute.
pub const CASES_ATTRIBUTE: &str = "cases";

/// Name of the operand segment-size attribute used by SparseTensor operations with multiple variadic operand groups.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Operation trait for `sparse_tensor.coiterate`.
pub trait CoIterateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the sparse iteration spaces being co-iterated.
    fn iteration_spaces(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let iteration_space_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        (0..iteration_space_count).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the initial loop-carried values.
    fn initial_values(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let start = self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        let initial_value_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        let end = start + initial_value_count;
        (start..end).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the bitset of coordinate levels used by the regions.
    fn coordinate_used_levels(&self) -> Result<u64, Error> {
        Ok(self.integer_attribute(COORDINATE_USED_LEVELS_ATTRIBUTE)?.unsigned_value())
    }

    /// Returns the case bitsets in region order.
    fn cases(&self) -> Result<Vec<u64>, Error> {
        self.array_attribute(CASES_ATTRIBUTE)?
            .elements()
            .map(|element| {
                element?
                    .cast::<IntegerAttributeRef<'c, 't>>()
                    .map(|attribute| attribute.unsigned_value())
                    .ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{CASES_ATTRIBUTE}` attribute in `{}`",
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
            })
            .collect()
    }

    /// Returns the final loop-carried values.
    fn final_values(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (0..self.result_count()).map(|index| Ok(self.result(index)?.as_ref())).collect()
    }

    /// Returns the case regions.
    fn case_regions(&self) -> Result<Vec<RegionRef<'o, 'c, 't>>, Error> {
        (0..self.region_count()).map(|index| self.region(index)).collect()
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
) -> Result<DetachedCoIterateOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    let result_types = initial_values.iter().map(|value| value.r#type()).collect::<Result<Vec<_>, _>>()?;
    let cases = cases
        .iter()
        .map(|case| context.integer_attribute(context.signless_integer_type(64), *case as i64))
        .collect::<Vec<_>>();
    OperationBuilder::new("sparse_tensor.coiterate", location)
        .add_operands(iteration_spaces)
        .add_operands(initial_values)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[iteration_spaces.len() as i32, initial_values.len() as i32])?,
        )
        .add_attribute(
            COORDINATE_USED_LEVELS_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), coordinate_used_levels as i64),
        )
        .add_attribute(CASES_ATTRIBUTE, context.array_attribute(&cases))
        .add_results(&result_types)
        .add_regions(case_regions)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::coiterate`"))
        })
}

/// Operation trait for `sparse_tensor.print`.
pub trait PrintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the sparse tensor being printed.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(Print);
mlir_op_trait!(Print, ZeroRegions);
mlir_op_trait!(Print, ZeroSuccessors);

/// Constructs a new detached/owned [`PrintOperation`] at the specified [`Location`].
pub fn print<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedPrintOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.print", location)
        .add_operand(tensor)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::print`"))
        })
}

/// Operation trait for `sparse_tensor.has_runtime_library`.
pub trait HasRuntimeLibraryOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns whether the sparse tensor runtime library is enabled.
    fn enabled(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(HasRuntimeLibrary);
mlir_op_trait!(HasRuntimeLibrary, OneResult);
mlir_op_trait!(HasRuntimeLibrary, ZeroOperands);
mlir_op_trait!(HasRuntimeLibrary, ZeroRegions);
mlir_op_trait!(HasRuntimeLibrary, ZeroSuccessors);

/// Constructs a new detached/owned [`HasRuntimeLibraryOperation`] at the specified [`Location`].
pub fn has_runtime_library<'c, 't: 'c, L: Location<'c, 't>>(
    location: L,
) -> Result<DetachedHasRuntimeLibraryOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::sparse_tensor()?)?;
    OperationBuilder::new("sparse_tensor.has_runtime_library", location)
        .add_result(context.signless_integer_type(1))
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `sparse_tensor::has_runtime_library`"))
        })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::dialects::sparse_tensor::{LevelFormat, LevelProperty, LevelType};
    use crate::{Attribute, Block, Context, Operation, Region, Size, Type, Value, ValueRef};

    use super::*;

    #[test]
    fn test_new() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let source_type = context.signless_integer_type(64);
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(source_type, location)]);
                let source = block.argument(0).unwrap().as_ref();
                let op = new(source, tensor_type, location).unwrap();
                assert_eq!(op.source().unwrap(), source);
                assert_eq!(op.source().unwrap(), source);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "new_test",
                    func::FuncAttributes {
                        arguments: vec![source_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @new_test(%arg0: i64) -> tensor<?xf64, #sparse> {
                    %0 = sparse_tensor.new %arg0 : i64 to tensor<?xf64, #sparse>
                    return %0 : tensor<?xf64, #sparse>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_assemble() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f64_type = context.float64_type();
        let level_tensor_type = context.tensor_type(index_type, &[Size::Dynamic], None, location).unwrap();
        let values_tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], None, location).unwrap();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Static(4)], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (level_tensor_type.as_ref(), location),
                    (level_tensor_type.as_ref(), location),
                    (values_tensor_type.as_ref(), location),
                ]);
                let positions = block.argument(0).unwrap().as_ref();
                let coordinates = block.argument(1).unwrap().as_ref();
                let values = block.argument(2).unwrap().as_ref();
                let op = assemble(&[positions, coordinates], values, tensor_type, location).unwrap();
                assert_eq!(op.levels().unwrap(), vec![positions, coordinates]);
                assert_eq!(op.values().unwrap(), values);
                assert_eq!(op.levels().unwrap(), vec![positions, coordinates]);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "assemble_test",
                    func::FuncAttributes {
                        arguments: vec![level_tensor_type.into(), level_tensor_type.into(), values_tensor_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @assemble_test(%arg0: tensor<?xindex>, %arg1: tensor<?xindex>, \
                    %arg2: tensor<?xf64>) -> tensor<4xf64, #sparse> {
                    %0 = sparse_tensor.assemble (%arg0, %arg1), %arg2 : \
                      (tensor<?xindex>, tensor<?xindex>), tensor<?xf64> to tensor<4xf64, #sparse>
                    return %0 : tensor<4xf64, #sparse>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_disassemble() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f64_type = context.float64_type();
        let level_tensor_type = context.tensor_type(index_type, &[Size::Dynamic], None, location).unwrap();
        let values_tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], None, location).unwrap();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (tensor_type.as_ref(), location),
                    (level_tensor_type.as_ref(), location),
                    (level_tensor_type.as_ref(), location),
                    (values_tensor_type.as_ref(), location),
                ]);
                let tensor = block.argument(0).unwrap().as_ref();
                let output_positions = block.argument(1).unwrap().as_ref();
                let output_coordinates = block.argument(2).unwrap().as_ref();
                let output_values = block.argument(3).unwrap().as_ref();
                let returned_level_types = [level_tensor_type.as_ref(), level_tensor_type.as_ref()];
                let level_length_types = [index_type.as_ref(), index_type.as_ref()];
                let op = disassemble(
                    tensor,
                    &[output_positions, output_coordinates],
                    output_values,
                    &returned_level_types,
                    values_tensor_type.as_ref(),
                    &level_length_types,
                    index_type.as_ref(),
                    location,
                )
                .unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.output_levels().unwrap(), vec![output_positions, output_coordinates]);
                assert_eq!(op.output_values().unwrap(), output_values);
                assert_eq!(op.returned_levels().unwrap().len(), 2);
                assert_eq!(op.level_lengths().unwrap().len(), 2);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 6);
                let results = op
                    .results()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap()
                    .into_iter()
                    .map(|result| result.as_ref())
                    .collect::<Vec<_>>();
                block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&results, location).unwrap()).unwrap();
                func::func(
                    "disassemble_test",
                    func::FuncAttributes {
                        arguments: vec![
                            tensor_type.into(),
                            level_tensor_type.into(),
                            level_tensor_type.into(),
                            values_tensor_type.into(),
                        ],
                        results: vec![
                            level_tensor_type.into(),
                            level_tensor_type.into(),
                            values_tensor_type.into(),
                            index_type.into(),
                            index_type.into(),
                            index_type.into(),
                        ],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @disassemble_test(%arg0: tensor<?xf64, #sparse>, %arg1: tensor<?xindex>, \
                    %arg2: tensor<?xindex>, %arg3: tensor<?xf64>) -> (tensor<?xindex>, tensor<?xindex>, \
                    tensor<?xf64>, index, index, index) {
                    %ret_levels:2, %ret_values, %lvl_lens:2, %val_len = sparse_tensor.disassemble %arg0 : tensor<?xf64, #sparse> \
                      out_lvls(%arg1, %arg2 : tensor<?xindex>, tensor<?xindex>) \
                      out_vals(%arg3 : tensor<?xf64>) -> (tensor<?xindex>, tensor<?xindex>), tensor<?xf64>, \
                      (index, index), index
                    return %ret_levels#0, %ret_levels#1, %ret_values, %lvl_lens#0, %lvl_lens#1, %val_len : tensor<?xindex>, tensor<?xindex>, \
                      tensor<?xf64>, index, index, index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_convert() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let dense_tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], None, location).unwrap();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let sparse_tensor_type =
            context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(dense_tensor_type, location)]);
                let source = block.argument(0).unwrap().as_ref();
                let op = convert(source, sparse_tensor_type, location).unwrap();
                assert_eq!(op.source().unwrap(), source);
                assert_eq!(op.source().unwrap(), source);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "convert_test",
                    func::FuncAttributes {
                        arguments: vec![dense_tensor_type.into()],
                        results: vec![sparse_tensor_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @convert_test(%arg0: tensor<?xf64>) -> tensor<?xf64, #sparse> {
                    %0 = sparse_tensor.convert %arg0 : tensor<?xf64> to tensor<?xf64, #sparse>
                    return %0 : tensor<?xf64, #sparse>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_reinterpret_map() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let source = block.argument(0).unwrap().as_ref();
                let op = reinterpret_map(source, tensor_type, location).unwrap();
                assert_eq!(op.source().unwrap(), source);
                assert_eq!(op.source().unwrap(), source);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "reinterpret_map_test",
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @reinterpret_map_test(%arg0: tensor<?xf64, #sparse>) -> tensor<?xf64, #sparse> {
                    %0 = sparse_tensor.reinterpret_map %arg0 : tensor<?xf64, #sparse> to tensor<?xf64, #sparse>
                    return %0 : tensor<?xf64, #sparse>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_positions() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        let memref_type = context.mem_ref_type(context.index_type(), &[Size::Dynamic], None, None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let op = positions(tensor, 0, memref_type, location).unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.level().unwrap(), 0);
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "positions_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![memref_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @positions_test(%arg0: tensor<?xf64, #sparse>) -> memref<?xindex> {
                    %0 = sparse_tensor.positions %arg0 {level = 0 : index} : tensor<?xf64, #sparse> to memref<?xindex>
                    return %0 : memref<?xindex>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_coordinates() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        let memref_type = context.mem_ref_type(context.index_type(), &[Size::Dynamic], None, None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let op = coordinates(tensor, 0, memref_type, location).unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.level().unwrap(), 0);
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "coordinates_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![memref_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @coordinates_test(%arg0: tensor<?xf64, #sparse>) -> memref<?xindex> {
                    %0 = sparse_tensor.coordinates %arg0 {level = 0 : index} : tensor<?xf64, #sparse> to memref<?xindex>
                    return %0 : memref<?xindex>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_coordinates_buffer() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[
                    LevelType::new(LevelFormat::Compressed, &[LevelProperty::NonUnique]),
                    LevelType::from(LevelFormat::Singleton),
                ],
                Some(context.identity_affine_map(2)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context
            .tensor_type(f64_type, &[Size::Dynamic, Size::Dynamic], Some(encoding.as_ref()), location)
            .unwrap();
        let memref_type = context.mem_ref_type(context.index_type(), &[Size::Dynamic], None, None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let op = coordinates_buffer(tensor, memref_type, location).unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "coordinates_buffer_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![memref_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton) }>
                module {
                  func.func @coordinates_buffer_test(%arg0: tensor<?x?xf64, #sparse>) -> memref<?xindex> {
                    %0 = sparse_tensor.coordinates_buffer %arg0 : tensor<?x?xf64, #sparse> to memref<?xindex>
                    return %0 : memref<?xindex>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_values() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        let memref_type = context.mem_ref_type(f64_type, &[Size::Dynamic], None, None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let op = values(tensor, memref_type, location).unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "values_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![memref_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @values_test(%arg0: tensor<?xf64, #sparse>) -> memref<?xf64> {
                    %0 = sparse_tensor.values %arg0 : tensor<?xf64, #sparse> to memref<?xf64>
                    return %0 : memref<?xf64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_number_of_entries() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let index_type = context.index_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let op = number_of_entries(tensor, location).unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "number_of_entries_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![index_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @number_of_entries_test(%arg0: tensor<?xf64, #sparse>) -> index {
                    %0 = sparse_tensor.number_of_entries %arg0 : tensor<?xf64, #sparse>
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_concatenate() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let input_type = context.tensor_type(f64_type, &[Size::Static(4)], Some(encoding.as_ref()), location).unwrap();
        let result_type = context.tensor_type(f64_type, &[Size::Static(8)], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(input_type, location), (input_type, location)]);
                let lhs = block.argument(0).unwrap().as_ref();
                let rhs = block.argument(1).unwrap().as_ref();
                let op = concatenate(&[lhs, rhs], 0, result_type, location).unwrap();
                assert_eq!(op.inputs().unwrap(), vec![lhs, rhs]);
                assert_eq!(op.dimension().unwrap(), 0);
                assert_eq!(op.dimension().unwrap(), 0);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "concatenate_test",
                    func::FuncAttributes {
                        arguments: vec![input_type.into(), input_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @concatenate_test(%arg0: tensor<4xf64, #sparse>, %arg1: tensor<4xf64, #sparse>) \
                    -> tensor<8xf64, #sparse> {
                    %0 = sparse_tensor.concatenate %arg0, %arg1 {dimension = 0 : index} : \
                      tensor<4xf64, #sparse>, tensor<4xf64, #sparse> to tensor<8xf64, #sparse>
                    return %0 : tensor<8xf64, #sparse>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_slice_offset() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let index_type = context.index_type();
        let slice = context.sparse_tensor_dim_slice_attribute(0, 4, 1).unwrap();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[slice],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Static(4)], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let op = slice_offset(tensor, 0, location).unwrap();
                assert_eq!(op.slice().unwrap(), tensor);
                assert_eq!(op.dimension().unwrap(), 0);
                assert_eq!(op.slice().unwrap(), tensor);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "slice_offset_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![index_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0 : #sparse_tensor<slice(0, 4, 1)>) -> \
                  (d0 : compressed) }>
                module {
                  func.func @slice_offset_test(%arg0: tensor<4xf64, #sparse>) -> index {
                    %0 = sparse_tensor.slice.offset %arg0 at 0 : tensor<4xf64, #sparse>
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_slice_stride() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let index_type = context.index_type();
        let slice = context.sparse_tensor_dim_slice_attribute(0, 4, 1).unwrap();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[slice],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Static(4)], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let op = slice_stride(tensor, 0, location).unwrap();
                assert_eq!(op.slice().unwrap(), tensor);
                assert_eq!(op.dimension().unwrap(), 0);
                assert_eq!(op.slice().unwrap(), tensor);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "slice_stride_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![index_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0 : #sparse_tensor<slice(0, 4, 1)>) -> \
                  (d0 : compressed) }>
                module {
                  func.func @slice_stride_test(%arg0: tensor<4xf64, #sparse>) -> index {
                    %0 = sparse_tensor.slice.stride %arg0 at 0 : tensor<4xf64, #sparse>
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_storage_specifier_init() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let specifier_type = context.sparse_tensor_storage_specifier_type(encoding).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = storage_specifier_init(None, specifier_type, location).unwrap();
                assert_eq!(op.source().unwrap(), None);
                assert_eq!(op.specifier().unwrap().r#type().unwrap(), specifier_type);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "storage_specifier_init_test",
                    func::FuncAttributes { results: vec![specifier_type.into()], ..Default::default() },
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @storage_specifier_init_test() -> !sparse_tensor.storage_specifier<#sparse> {
                    %0 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#sparse>
                    return %0 : !sparse_tensor.storage_specifier<#sparse>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_storage_specifier_get() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let specifier_type = context.sparse_tensor_storage_specifier_type(encoding).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(specifier_type, location)]);
                let specifier = block.argument(0).unwrap().as_ref();
                let op =
                    storage_specifier_get(specifier, StorageSpecifierKind::CoordinateMemorySize, Some(0), location)
                        .unwrap();
                assert_eq!(op.specifier().unwrap(), specifier);
                assert_eq!(op.specifier_kind().unwrap().value().unwrap(), StorageSpecifierKind::CoordinateMemorySize);
                assert_eq!(op.level().unwrap(), Some(0));
                assert_eq!(op.specifier().unwrap(), specifier);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "storage_specifier_get_test",
                    func::FuncAttributes {
                        arguments: vec![specifier_type.into()],
                        results: vec![index_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @storage_specifier_get_test(%arg0: !sparse_tensor.storage_specifier<#sparse>) -> index {
                    %0 = sparse_tensor.storage_specifier.get %arg0 crd_mem_sz at 0 : \
                      !sparse_tensor.storage_specifier<#sparse>
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_storage_specifier_set() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let specifier_type = context.sparse_tensor_storage_specifier_type(encoding).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(specifier_type.as_ref(), location), (index_type.as_ref(), location)]);
                let specifier = block.argument(0).unwrap().as_ref();
                let value = block.argument(1).unwrap().as_ref();
                let op = storage_specifier_set(
                    specifier,
                    StorageSpecifierKind::CoordinateMemorySize,
                    Some(0),
                    value,
                    location,
                )
                .unwrap();
                assert_eq!(op.specifier().unwrap(), specifier);
                assert_eq!(op.specifier_kind().unwrap().value().unwrap(), StorageSpecifierKind::CoordinateMemorySize);
                assert_eq!(op.level().unwrap(), Some(0));
                assert_eq!(op.value().unwrap(), value);
                assert_eq!(op.specifier().unwrap(), specifier);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "storage_specifier_set_test",
                    func::FuncAttributes {
                        arguments: vec![specifier_type.into(), index_type.into()],
                        results: vec![specifier_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @storage_specifier_set_test(%arg0: !sparse_tensor.storage_specifier<#sparse>, \
                    %arg1: index) -> !sparse_tensor.storage_specifier<#sparse> {
                    %0 = sparse_tensor.storage_specifier.set %arg0 crd_mem_sz at 0 with %arg1 : \
                      !sparse_tensor.storage_specifier<#sparse>
                    return %0 : !sparse_tensor.storage_specifier<#sparse>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_level() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type.as_ref(), location), (index_type.as_ref(), location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let index = block.argument(1).unwrap().as_ref();
                let op = level(tensor, index, location).unwrap();
                assert_eq!(op.source().unwrap(), tensor);
                assert_eq!(op.index().unwrap(), index);
                assert_eq!(op.source().unwrap(), tensor);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "level_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @level_test(%arg0: tensor<?xf64, #sparse>, %arg1: index) -> index {
                    %0 = sparse_tensor.lvl %arg0, %arg1 : tensor<?xf64, #sparse>
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_coordinate_translate() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location)]);
                let coordinate = block.argument(0).unwrap().as_ref();
                let op = coordinate_translate(
                    &[coordinate],
                    CoordinateTranslationDirection::DimensionToLevel,
                    encoding,
                    1,
                    location,
                )
                .unwrap();
                assert_eq!(op.input_coordinates().unwrap(), vec![coordinate]);
                assert_eq!(op.direction().unwrap().value().unwrap(), CoordinateTranslationDirection::DimensionToLevel);
                assert_eq!(op.encoder().unwrap(), encoding);
                assert_eq!(op.output_coordinates().unwrap().len(), 1);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "coordinate_translate_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into()],
                        results: vec![index_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @coordinate_translate_test(%arg0: index) -> index {
                    %0 = sparse_tensor.crd_translate dim_to_lvl[%arg0] as #sparse : index
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_push_back() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f64_type = context.float64_type();
        let memref_type = context.mem_ref_type(f64_type, &[Size::Dynamic], None, None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (index_type.as_ref(), location),
                    (memref_type.as_ref(), location),
                    (f64_type.as_ref(), location),
                    (index_type.as_ref(), location),
                ]);
                let current_size = block.argument(0).unwrap().as_ref();
                let input_buffer = block.argument(1).unwrap().as_ref();
                let value = block.argument(2).unwrap().as_ref();
                let count = block.argument(3).unwrap().as_ref();
                let op = push_back(current_size, input_buffer, value, Some(count), true, location).unwrap();
                assert_eq!(op.current_size().unwrap(), current_size);
                assert_eq!(op.input_buffer().unwrap(), input_buffer);
                assert_eq!(op.value().unwrap(), value);
                assert_eq!(op.count().unwrap(), Some(count));
                assert!(op.inbounds());
                assert_eq!(op.output_buffer().unwrap().r#type().unwrap(), memref_type);
                assert_eq!(op.new_size().unwrap().r#type().unwrap(), index_type);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                let results = op
                    .results()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap()
                    .into_iter()
                    .map(|result| result.as_ref())
                    .collect::<Vec<_>>();
                block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&results, location).unwrap()).unwrap();
                func::func(
                    "push_back_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), memref_type.into(), f64_type.into(), index_type.into()],
                        results: vec![memref_type.into(), index_type.into()],
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
                  func.func @push_back_test(%arg0: index, %arg1: memref<?xf64>, %arg2: f64, %arg3: index) \
                    -> (memref<?xf64>, index) {
                    %outBuffer, %newSize = sparse_tensor.push_back inbounds %arg0, %arg1, %arg2, %arg3 : \
                      index, memref<?xf64>, f64, index
                    return %outBuffer, %newSize : memref<?xf64>, index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_expand() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f64_type = context.float64_type();
        let i1_type = context.signless_integer_type(1);
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        let values_type = context.mem_ref_type(f64_type, &[Size::Dynamic], None, None, location).unwrap();
        let filled_type = context.mem_ref_type(i1_type, &[Size::Dynamic], None, None, location).unwrap();
        let added_type = context.mem_ref_type(index_type, &[Size::Dynamic], None, None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let op =
                    expand(tensor, values_type.as_ref(), filled_type.as_ref(), added_type.as_ref(), location).unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.values().unwrap().r#type().unwrap(), values_type);
                assert_eq!(op.filled().unwrap().r#type().unwrap(), filled_type);
                assert_eq!(op.added().unwrap().r#type().unwrap(), added_type);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                let results = op
                    .results()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap()
                    .into_iter()
                    .map(|result| result.as_ref())
                    .collect::<Vec<_>>();
                block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&results, location).unwrap()).unwrap();
                func::func(
                    "expand_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![values_type.into(), filled_type.into(), added_type.into(), index_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @expand_test(%arg0: tensor<?xf64, #sparse>) \
                    -> (memref<?xf64>, memref<?xi1>, memref<?xindex>, index) {
                    %values, %filled, %added, %count = sparse_tensor.expand %arg0 : tensor<?xf64, #sparse> \
                      to memref<?xf64>, memref<?xi1>, memref<?xindex>
                    return %values, %filled, %added, %count : memref<?xf64>, memref<?xi1>, memref<?xindex>, index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_compress() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f64_type = context.float64_type();
        let i1_type = context.signless_integer_type(1);
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        let values_type = context.mem_ref_type(f64_type, &[Size::Dynamic], None, None, location).unwrap();
        let filled_type = context.mem_ref_type(i1_type, &[Size::Dynamic], None, None, location).unwrap();
        let added_type = context.mem_ref_type(index_type, &[Size::Dynamic], None, None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (values_type.as_ref(), location),
                    (filled_type.as_ref(), location),
                    (added_type.as_ref(), location),
                    (index_type.as_ref(), location),
                    (tensor_type.as_ref(), location),
                ]);
                let values = block.argument(0).unwrap().as_ref();
                let filled = block.argument(1).unwrap().as_ref();
                let added = block.argument(2).unwrap().as_ref();
                let count = block.argument(3).unwrap().as_ref();
                let tensor = block.argument(4).unwrap().as_ref();
                let op = compress(values, filled, added, count, tensor, &[], location).unwrap();
                assert_eq!(op.values().unwrap(), values);
                assert_eq!(op.filled().unwrap(), filled);
                assert_eq!(op.added().unwrap(), added);
                assert_eq!(op.count().unwrap(), count);
                assert_eq!(op.tensor().unwrap(), tensor);
                assert!(op.level_coordinates().unwrap().is_empty());
                assert_eq!(op.as_ref().result(0).unwrap().r#type().unwrap(), tensor_type);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 5);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "compress_test",
                    func::FuncAttributes {
                        arguments: vec![
                            values_type.into(),
                            filled_type.into(),
                            added_type.into(),
                            index_type.into(),
                            tensor_type.into(),
                        ],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @compress_test(%arg0: memref<?xf64>, %arg1: memref<?xi1>, %arg2: memref<?xindex>, \
                    %arg3: index, %arg4: tensor<?xf64, #sparse>) -> tensor<?xf64, #sparse> {
                    %0 = sparse_tensor.compress %arg0, %arg1, %arg2, %arg3 into %arg4[] : \
                      memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<?xf64, #sparse>
                    return %0 : tensor<?xf64, #sparse>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_load() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let op = load(tensor, true, location).unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert!(op.has_inserts());
                assert_eq!(op.as_ref().result(0).unwrap().r#type().unwrap(), tensor_type);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "load_test",
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @load_test(%arg0: tensor<?xf64, #sparse>) -> tensor<?xf64, #sparse> {
                    %0 = sparse_tensor.load %arg0 hasInserts : tensor<?xf64, #sparse>
                    return %0 : tensor<?xf64, #sparse>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_out() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let destination_type = context.signless_integer_type(64);
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block =
                    context.block(&[(tensor_type.as_ref(), location), (destination_type.as_ref(), location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let destination = block.argument(1).unwrap().as_ref();
                let op = out(tensor, destination, location).unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.destination().unwrap(), destination);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[] as &[ValueRef], location).unwrap()).unwrap();
                func::func(
                    "out_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into(), destination_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @out_test(%arg0: tensor<?xf64, #sparse>, %arg1: i64) {
                    sparse_tensor.out %arg0, %arg1 : tensor<?xf64, #sparse>, i64
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_sort() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f64_type = context.float64_type();
        let coordinates_type = context.mem_ref_type(index_type, &[Size::Dynamic], None, None, location).unwrap();
        let payload_type = context.mem_ref_type(f64_type, &[Size::Dynamic], None, None, location).unwrap();
        let permutation_map = context.identity_affine_map(1);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (index_type.as_ref(), location),
                    (coordinates_type.as_ref(), location),
                    (payload_type.as_ref(), location),
                ]);
                let count = block.argument(0).unwrap().as_ref();
                let coordinates = block.argument(1).unwrap().as_ref();
                let payload = block.argument(2).unwrap().as_ref();
                let op = sort(count, coordinates, &[payload], permutation_map, Some(1), SortKind::QuickSort, location)
                    .unwrap();
                assert_eq!(op.count().unwrap(), count);
                assert_eq!(op.coordinates_and_values().unwrap(), coordinates);
                assert_eq!(op.payloads().unwrap(), vec![payload]);
                assert_eq!(op.permutation_map().unwrap(), permutation_map);
                assert_eq!(op.payload_count().unwrap(), Some(1));
                assert_eq!(op.algorithm().unwrap().value().unwrap(), SortKind::QuickSort);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[] as &[ValueRef], location).unwrap()).unwrap();
                func::func(
                    "sort_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), coordinates_type.into(), payload_type.into()],
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
                #map = affine_map<(d0) -> (d0)>
                module {
                  func.func @sort_test(%arg0: index, %arg1: memref<?xindex>, %arg2: memref<?xf64>) {
                    sparse_tensor.sort quick_sort %arg0, %arg1 jointly %arg2 {ny = 1 : index, perm_map = #map} \
                      : memref<?xindex> jointly memref<?xf64>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_reorder_coo() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[
                    LevelType::new(LevelFormat::Compressed, &[LevelProperty::NonUnique]),
                    LevelType::from(LevelFormat::Singleton),
                ],
                Some(context.identity_affine_map(2)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context
            .tensor_type(f64_type, &[Size::Dynamic, Size::Dynamic], Some(encoding.as_ref()), location)
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap().as_ref();
                let op = reorder_coo(input, SortKind::QuickSort, tensor_type, location).unwrap();
                assert_eq!(op.input_coo().unwrap(), input);
                assert_eq!(op.algorithm().unwrap().value().unwrap(), SortKind::QuickSort);
                assert_eq!(op.input_coo().unwrap(), input);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "reorder_coo_test",
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
                #sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton) }>
                module {
                  func.func @reorder_coo_test(%arg0: tensor<?x?xf64, #sparse>) -> tensor<?x?xf64, #sparse> {
                    %0 = sparse_tensor.reorder_coo quick_sort %arg0 : tensor<?x?xf64, #sparse> to tensor<?x?xf64, #sparse>
                    return %0 : tensor<?x?xf64, #sparse>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_binary() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f64_type, location), (f64_type, location)]);
                let lhs = block.argument(0).unwrap().as_ref();
                let rhs = block.argument(1).unwrap().as_ref();
                let mut overlap_region = context.region();
                let mut overlap_block = context.block(&[(f64_type, location), (f64_type, location)]);
                let overlap_lhs = overlap_block.argument(0).unwrap().as_ref();
                overlap_block.append_operation(r#yield(&[overlap_lhs], location).unwrap()).unwrap();
                overlap_region.append_block(overlap_block).unwrap();
                let op = binary(
                    lhs,
                    rhs,
                    f64_type,
                    true,
                    true,
                    overlap_region.into(),
                    context.region().into(),
                    context.region().into(),
                    location,
                )
                .unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert!(op.left_identity());
                assert!(op.right_identity());
                assert_eq!(op.overlap_region().unwrap().blocks().unwrap().count(), 1);
                assert_eq!(op.left_region().unwrap().blocks().unwrap().count(), 0);
                assert_eq!(op.right_region().unwrap().blocks().unwrap().count(), 0);
                assert_eq!(op.overlap_region().unwrap().blocks().unwrap().count(), 1);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "binary_test",
                    func::FuncAttributes {
                        arguments: vec![f64_type.into(), f64_type.into()],
                        results: vec![f64_type.into()],
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
                  func.func @binary_test(%arg0: f64, %arg1: f64) -> f64 {
                    %0 = sparse_tensor.binary %arg0, %arg1 : f64, f64 to f64
                     overlap = {
                    ^bb0(%arg2: f64, %arg3: f64):
                      sparse_tensor.yield %arg2 : f64
                    }
                     left = identity
                     right = identity
                    return %0 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_unary() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f64_type, location)]);
                let input = block.argument(0).unwrap().as_ref();
                let mut present_region = context.region();
                let mut present_block = context.block(&[(f64_type, location)]);
                let present_value = present_block.argument(0).unwrap().as_ref();
                present_block.append_operation(r#yield(&[present_value], location).unwrap()).unwrap();
                present_region.append_block(present_block).unwrap();
                let absent_region = context.region();
                let op = unary(input, f64_type, present_region.into(), absent_region.into(), location).unwrap();
                assert_eq!(op.input().unwrap(), input);
                assert_eq!(op.present_region().unwrap().blocks().unwrap().count(), 1);
                assert_eq!(op.absent_region().unwrap().blocks().unwrap().count(), 0);
                assert_eq!(op.input().unwrap(), input);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "unary_test",
                    func::FuncAttributes {
                        arguments: vec![f64_type.into()],
                        results: vec![f64_type.into()],
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
                  func.func @unary_test(%arg0: f64) -> f64 {
                    %0 = sparse_tensor.unary %arg0 : f64 to f64
                     present = {
                    ^bb0(%arg1: f64):
                      sparse_tensor.yield %arg1 : f64
                    }
                     absent = {
                    }
                    return %0 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_reduce() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f64_type, location), (f64_type, location), (f64_type, location)]);
                let lhs = block.argument(0).unwrap().as_ref();
                let rhs = block.argument(1).unwrap().as_ref();
                let identity = block.argument(2).unwrap().as_ref();
                let mut region = context.region();
                let mut region_block = context.block(&[(f64_type, location), (f64_type, location)]);
                let region_lhs = region_block.argument(0).unwrap().as_ref();
                region_block.append_operation(r#yield(&[region_lhs], location).unwrap()).unwrap();
                region.append_block(region_block).unwrap();
                let op = reduce(lhs, rhs, identity, region.into(), location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.identity().unwrap(), identity);
                assert_eq!(op.as_ref().region(0).unwrap().blocks().unwrap().count(), 1);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "reduce_test",
                    func::FuncAttributes {
                        arguments: vec![f64_type.into(), f64_type.into(), f64_type.into()],
                        results: vec![f64_type.into()],
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
                  func.func @reduce_test(%arg0: f64, %arg1: f64, %arg2: f64) -> f64 {
                    %0 = sparse_tensor.reduce %arg0, %arg1, %arg2 : f64 {
                    ^bb0(%arg3: f64, %arg4: f64):
                      sparse_tensor.yield %arg3 : f64
                    }
                    return %0 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_select() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let i1_type = context.signless_integer_type(1);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f64_type.as_ref(), location), (i1_type.as_ref(), location)]);
                let input = block.argument(0).unwrap().as_ref();
                let predicate = block.argument(1).unwrap().as_ref();
                let mut region = context.region();
                let mut region_block = context.block(&[(f64_type, location)]);
                region_block.append_operation(r#yield(&[predicate], location).unwrap()).unwrap();
                region.append_block(region_block).unwrap();
                let op = select(input, region.into(), location).unwrap();
                assert_eq!(op.input().unwrap(), input);
                assert_eq!(op.as_ref().region(0).unwrap().blocks().unwrap().count(), 1);
                assert_eq!(op.input().unwrap(), input);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "select_test",
                    func::FuncAttributes {
                        arguments: vec![f64_type.into(), i1_type.into()],
                        results: vec![f64_type.into()],
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
                  func.func @select_test(%arg0: f64, %arg1: i1) -> f64 {
                    %0 = sparse_tensor.select %arg0 : f64 {
                    ^bb0(%arg2: f64):
                      sparse_tensor.yield %arg1 : i1
                    }
                    return %0 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_yield() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f64_type, location), (f64_type, location), (f64_type, location)]);
                let lhs = block.argument(0).unwrap().as_ref();
                let rhs = block.argument(1).unwrap().as_ref();
                let identity = block.argument(2).unwrap().as_ref();
                let mut region = context.region();
                let mut region_block = context.block(&[(f64_type, location), (f64_type, location)]);
                let yielded = region_block.argument(0).unwrap().as_ref();
                let yield_op = r#yield(&[yielded], location).unwrap();
                assert_eq!(yield_op.values().unwrap(), vec![yielded]);
                assert_eq!(yield_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(yield_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                region_block.append_operation(yield_op).unwrap();
                region.append_block(region_block).unwrap();
                let op = reduce(lhs, rhs, identity, region.into(), location).unwrap();
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "yield_test",
                    func::FuncAttributes {
                        arguments: vec![f64_type.into(), f64_type.into(), f64_type.into()],
                        results: vec![f64_type.into()],
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
                  func.func @yield_test(%arg0: f64, %arg1: f64, %arg2: f64) -> f64 {
                    %0 = sparse_tensor.reduce %arg0, %arg1, %arg2 : f64 {
                    ^bb0(%arg3: f64, %arg4: f64):
                      sparse_tensor.yield %arg3 : f64
                    }
                    return %0 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_foreach() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f64_type = context.float64_type();
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type.as_ref(), location), (i64_type.as_ref(), location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let initial = block.argument(1).unwrap().as_ref();
                let mut body = context.region();
                let mut body_block = context.block(&[
                    (index_type.as_ref(), location),
                    (f64_type.as_ref(), location),
                    (i64_type.as_ref(), location),
                ]);
                let carried = body_block.argument(2).unwrap().as_ref();
                body_block.append_operation(r#yield(&[carried], location).unwrap()).unwrap();
                body.append_block(body_block).unwrap();
                let op = foreach(tensor, &[initial], None, body.into(), location).unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.initial_values().unwrap(), vec![initial]);
                assert_eq!(op.order().unwrap(), None);
                assert_eq!(op.final_values().unwrap().len(), 1);
                assert_eq!(op.body_region().unwrap().blocks().unwrap().count(), 1);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "foreach_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into(), i64_type.into()],
                        results: vec![i64_type.into()],
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
                  func.func @foreach_test(%arg0: tensor<?xf64>, %arg1: i64) -> i64 {
                    %0 = sparse_tensor.foreach in %arg0 init(%arg1) : tensor<?xf64>, i64 -> i64 do {
                    ^bb0(%arg2: index, %arg3: f64, %arg4: i64):
                      sparse_tensor.yield %arg4 : i64
                    }
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_extract_iteration_space() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        let iteration_space_type = context.sparse_tensor_iteration_space_type(encoding, 0, 1).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let op = extract_iteration_space(tensor, None, 0, 1, iteration_space_type, location).unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.parent_iterator().unwrap(), None);
                assert_eq!(op.lower_level().unwrap(), 0);
                assert_eq!(op.upper_level().unwrap(), 1);
                assert_eq!(op.lower_level().unwrap(), 0);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "extract_iteration_space_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![iteration_space_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @extract_iteration_space_test(%arg0: tensor<?xf64, #sparse>) \
                    -> !sparse_tensor.iter_space<#sparse, lvls = 0> {
                    %0 = sparse_tensor.extract_iteration_space %arg0 lvls = 0 : tensor<?xf64, #sparse> \
                      -> !sparse_tensor.iter_space<#sparse, lvls = 0>
                    return %0 : !sparse_tensor.iter_space<#sparse, lvls = 0>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_extract_value() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        let iterator_type = context.sparse_tensor_iterator_type(encoding, 0, 1).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type.as_ref(), location), (iterator_type.as_ref(), location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let iterator = block.argument(1).unwrap().as_ref();
                let op = extract_value(tensor, iterator, f64_type, location).unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.iterator().unwrap(), iterator);
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "extract_value_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into(), iterator_type.into()],
                        results: vec![f64_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @extract_value_test(%arg0: tensor<?xf64, #sparse>, \
                    %arg1: !sparse_tensor.iterator<#sparse, lvls = 0>) -> f64 {
                    %0 = sparse_tensor.extract_value %arg0 at %arg1 : tensor<?xf64, #sparse>, \
                      !sparse_tensor.iterator<#sparse, lvls = 0>
                    return %0 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_iterate() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let iteration_space_type = context.sparse_tensor_iteration_space_type(encoding, 0, 1).unwrap();
        let iterator_type = context.sparse_tensor_iterator_type(encoding, 0, 1).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block =
                    context.block(&[(iteration_space_type.as_ref(), location), (index_type.as_ref(), location)]);
                let iteration_space = block.argument(0).unwrap().as_ref();
                let initial = block.argument(1).unwrap().as_ref();
                let mut body = context.region();
                let mut body_block = context.block(&[
                    (index_type.as_ref(), location),
                    (index_type.as_ref(), location),
                    (iterator_type.as_ref(), location),
                ]);
                let carried = body_block.argument(0).unwrap().as_ref();
                body_block.append_operation(r#yield(&[carried], location).unwrap()).unwrap();
                body.append_block(body_block).unwrap();
                let op = iterate(iteration_space, &[initial], 1, body.into(), location).unwrap();
                assert_eq!(op.iteration_space().unwrap(), iteration_space);
                assert_eq!(op.initial_values().unwrap(), vec![initial]);
                assert_eq!(op.coordinate_used_levels().unwrap(), 1);
                assert_eq!(op.final_values().unwrap().len(), 1);
                assert_eq!(op.body_region().unwrap().blocks().unwrap().count(), 1);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "iterate_test",
                    func::FuncAttributes {
                        arguments: vec![iteration_space_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @iterate_test(%arg0: !sparse_tensor.iter_space<#sparse, lvls = 0>, \
                    %arg1: index) -> index {
                    %0 = sparse_tensor.iterate %arg4 in %arg0 at(%arg3) iter_args(%arg2 = %arg1) : \
                      !sparse_tensor.iter_space<#sparse, lvls = 0>  -> index {
                      sparse_tensor.yield %arg2 : index
                    }
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_coiterate() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let iteration_space_type = context.sparse_tensor_iteration_space_type(encoding, 0, 1).unwrap();
        let iterator_type = context.sparse_tensor_iterator_type(encoding, 0, 1).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (iteration_space_type.as_ref(), location),
                    (iteration_space_type.as_ref(), location),
                    (index_type.as_ref(), location),
                ]);
                let lhs_space = block.argument(0).unwrap().as_ref();
                let rhs_space = block.argument(1).unwrap().as_ref();
                let initial = block.argument(2).unwrap().as_ref();
                let mut case_region = context.region();
                let mut case_block = context.block(&[
                    (index_type.as_ref(), location),
                    (index_type.as_ref(), location),
                    (iterator_type.as_ref(), location),
                ]);
                let carried = case_block.argument(0).unwrap().as_ref();
                case_block.append_operation(r#yield(&[carried], location).unwrap()).unwrap();
                case_region.append_block(case_block).unwrap();
                let op = coiterate(&[lhs_space, rhs_space], &[initial], 1, &[1], vec![case_region.into()], location)
                    .unwrap();
                assert_eq!(op.iteration_spaces().unwrap(), vec![lhs_space, rhs_space]);
                assert_eq!(op.initial_values().unwrap(), vec![initial]);
                assert_eq!(op.coordinate_used_levels().unwrap(), 1);
                assert_eq!(op.cases().unwrap(), vec![1]);
                assert_eq!(op.final_values().unwrap().len(), 1);
                assert_eq!(op.case_regions().unwrap().len(), 1);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "coiterate_test",
                    func::FuncAttributes {
                        arguments: vec![iteration_space_type.into(), iteration_space_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @coiterate_test(%arg0: !sparse_tensor.iter_space<#sparse, lvls = 0>, \
                    %arg1: !sparse_tensor.iter_space<#sparse, lvls = 0>, %arg2: index) -> index {
                    %0 = sparse_tensor.coiterate (%arg0, %arg1) at(%arg4) iter_args(%arg3 = %arg2) : \
                      (!sparse_tensor.iter_space<#sparse, lvls = 0>, !sparse_tensor.iter_space<#sparse, lvls = 0>) \
                      -> index
                    case %arg5, _ {
                      sparse_tensor.yield %arg3 : index
                    }
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_print() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f64_type = context.float64_type();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let tensor_type = context.tensor_type(f64_type, &[Size::Dynamic], Some(encoding.as_ref()), location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let tensor = block.argument(0).unwrap().as_ref();
                let op = print(tensor, location).unwrap();
                assert_eq!(op.tensor().unwrap(), tensor);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[] as &[ValueRef], location).unwrap()).unwrap();
                func::func(
                    "print_test",
                    func::FuncAttributes { arguments: vec![tensor_type.into()], ..Default::default() },
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
                #sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
                module {
                  func.func @print_test(%arg0: tensor<?xf64, #sparse>) {
                    sparse_tensor.print %arg0 : tensor<?xf64, #sparse>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_has_runtime_library() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let result_type = context.signless_integer_type(1);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = has_runtime_library(location).unwrap();
                assert_eq!(op.enabled().unwrap().r#type().unwrap(), result_type);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return(&[op.as_ref().result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "has_runtime_library_test",
                    func::FuncAttributes { results: vec![result_type.into()], ..Default::default() },
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
                  func.func @has_runtime_library_test() -> i1 {
                    %0 = sparse_tensor.has_runtime_library
                    return %0 : i1
                  }
                }
            "},
        );
    }
}
