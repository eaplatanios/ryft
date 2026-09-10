//! Typed constructors and accessors for Bufferization operations on ranked built-in tensors and memrefs.
//!
//! Constructors check operand kinds and local shape constraints before creating detached operations. Verify the
//! containing module to check the remaining MLIR invariants. `restrict`, `writable`, and `read_only` describe promises
//! to the bufferization analysis; these constructors cannot prove those promises from a single operation.
//!
//! The underlying dialect also supports unranked and custom tensor-like/buffer-like types in some operations. These
//! convenience constructors accept the ranked built-in types represented by [`TensorTypeRef`] and [`MemRefTypeRef`].
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/BufferizationOps/)
//! for more information.

use crate::macros::{mlir_op, mlir_op_trait};
use crate::{
    AttributeRef, DetachedOp, DialectHandle, Error, IndexTypeRef, Location, MemRefTypeRef, OneResult, Operation,
    OperationBuilder, ShapedType, Size, TensorTypeRef, Type, TypeRef, Value, ValueRef,
};

/// Name of the [`Attribute`](crate::Attribute) that records optional and variadic operand segment sizes.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Name of the [`Attribute`](crate::Attribute) that selects the memory space for [`AllocTensorOperation`].
pub const MEMORY_SPACE_ATTRIBUTE: &str = "memory_space";

/// Bufferization [`Operation`] that creates a fresh tensor allocation, optionally initialized from
/// [`AllocTensorOperation::copy`]. Without `copy`, the tensor contents are undefined. A copy supplies the dynamic
/// sizes, so explicit dynamic size operands must be absent. Otherwise, provide one index operand per dynamic result
/// dimension, in dimension order. [`AllocTensorOperation::size_hint`] estimates the number of nonzero sparse elements
/// and must be between one and the dense element count at execution time. [`AllocTensorOperation::memory_space`]
/// selects the allocation's memory space. Otherwise, it is inferred from `copy` or uses the bufferization default.
///
/// # Example
///
/// The following is an example of an [`AllocTensorOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %tensor = bufferization.alloc_tensor(%size) : tensor<?xf32>
/// ```
///
/// Refer to the [official MLIR documentation](
/// https://mlir.llvm.org/docs/Dialects/BufferizationOps/#bufferizationalloc_tensor-bufferizationalloctensorop)
/// for more information.
pub trait AllocTensorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the dynamic dimension operands.
    fn dynamic_sizes(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the optional tensor copied into the allocation.
    fn copy(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .next()
            .map(|index| self.operand_value(index))
            .transpose()
    }

    /// Returns the optional sparse allocation size hint.
    fn size_hint(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .next()
            .map(|index| self.operand_value(index))
            .transpose()
    }

    /// Returns the optional memory-space attribute.
    fn memory_space(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute(MEMORY_SPACE_ATTRIBUTE)
    }

    /// Returns the allocated tensor.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(AllocTensor);
mlir_op_trait!(AllocTensor, OneResult);
mlir_op_trait!(AllocTensor, ZeroRegions);
mlir_op_trait!(AllocTensor, ZeroSuccessors);

/// Constructs a new detached/owned [`AllocTensorOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`AllocTensorOperation`] for more information on the operation semantics.
///
/// # Parameters
///
///   - `dynamic_sizes`: One index value per dynamic result dimension, or an empty slice when `copy` is present.
///   - `copy`: Optional tensor whose type and initial contents match the result.
///   - `size_hint`: Optional index value estimating the number of nonzero elements for a sparse allocation.
///   - `memory_space`: Optional allocation memory space; otherwise inferred from `copy` or the bufferization default.
///   - `result_type`: Ranked tensor type to allocate.
///   - `location`: Source location to attach to the operation.
pub fn alloc_tensor<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    dynamic_sizes: &[ValueRef<'v, 'c, 't>],
    copy: Option<ValueRef<'v, 'c, 't>>,
    size_hint: Option<ValueRef<'v, 'c, 't>>,
    memory_space: Option<AttributeRef<'c, 't>>,
    result_type: TensorTypeRef<'c, 't>,
    location: L,
) -> Result<DetachedAllocTensorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::bufferization()?)?;
    if copy.is_some() && !dynamic_sizes.is_empty() {
        return Err(Error::invalid_argument(
            "expected `bufferization.alloc_tensor` with `copy` to have no dynamic size operands",
        ));
    }
    if copy.is_none() && dynamic_sizes.len() != result_type.dimensions().filter(Size::is_dynamic).count() {
        return Err(Error::invalid_argument(
            "expected one dynamic size operand per dynamic result dimension of `bufferization.alloc_tensor`",
        ));
    }
    for dynamic_size in dynamic_sizes {
        if !dynamic_size.r#type()?.is::<IndexTypeRef>() {
            return Err(Error::invalid_argument(
                "expected dynamic size operands of `bufferization.alloc_tensor` to have index type",
            ));
        }
    }
    if let Some(copy) = copy
        && validate_tensor(copy, "bufferization.alloc_tensor")? != result_type
    {
        return Err(Error::invalid_argument(
            "expected `copy` and result of `bufferization.alloc_tensor` to have the same type",
        ));
    }
    if let Some(size_hint) = size_hint
        && !size_hint.r#type()?.is::<IndexTypeRef>()
    {
        return Err(Error::invalid_argument("expected `size_hint` of `bufferization.alloc_tensor` to have index type"));
    }
    let segment_sizes = [
        i32::try_from(dynamic_sizes.len())
            .map_err(|_| Error::invalid_argument("too many dynamic sizes for `bufferization.alloc_tensor`"))?,
        i32::from(copy.is_some()),
        i32::from(size_hint.is_some()),
    ];
    let mut builder = OperationBuilder::new("bufferization.alloc_tensor", location)
        .add_operands(dynamic_sizes)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes)?)
        .add_result(result_type);
    if let Some(copy) = copy {
        builder = builder.add_operand(copy);
    }
    if let Some(size_hint) = size_hint {
        builder = builder.add_operand(size_hint);
    }
    if let Some(memory_space) = memory_space {
        builder = builder.add_attribute(MEMORY_SPACE_ATTRIBUTE, memory_space);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `bufferization::alloc_tensor`"))
    })
}

/// Bufferization [`Operation`] that clones a memref view. A valid lowering may alias the source and result instead
/// of copying the data. Mutating either after cloning has undefined behavior; this operation is not a promise of an
/// independent writable allocation.
///
/// # Example
///
/// The following is an example of a [`CloneOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = bufferization.clone %input : memref<4xf32> to memref<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation](
/// https://mlir.llvm.org/docs/Dialects/BufferizationOps/#bufferizationclone-bufferizationcloneop)
/// for more information.
pub trait CloneOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the cloned memref.
    fn output_memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(Clone);
mlir_op_trait!(Clone, OneResult);
mlir_op_trait!(Clone, ZeroRegions);
mlir_op_trait!(Clone, ZeroSuccessors);

/// Constructs a new detached/owned [`CloneOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`CloneOperation`] for more information on the operation semantics.
///
/// # Parameters
///
///   - `input`: Ranked memref to clone.
///   - `output_type`: Ranked memref type of the cloned view.
///   - `location`: Source location to attach to the operation.
pub fn clone<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    output_type: MemRefTypeRef<'c, 't>,
    location: L,
) -> Result<DetachedCloneOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::bufferization()?)?;
    validate_memref(input, "bufferization.clone")?;
    OperationBuilder::new("bufferization.clone", location)
        .add_operand(input)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `bufferization::clone`"))
        })
}

/// Name of the unit [`Attribute`](crate::Attribute) that carries the aliasing restriction promise.
pub const RESTRICT_ATTRIBUTE: &str = "restrict";

/// Name of the unit [`Attribute`](crate::Attribute) that marks a buffer as writable during bufferization.
pub const WRITABLE_ATTRIBUTE: &str = "writable";

/// Bufferization [`Operation`] that materializes a source tensor in a specified destination. A tensor destination
/// produces an updated tensor with exactly the destination's type, including its encoding. A memref destination
/// produces no result and requires `writable`. The source and destination must have equal runtime shapes and element
/// types; compatible dynamic dimensions are accepted during construction.
///
/// `restrict` and `writable` are valid only for memref destinations and have the semantics documented on
/// [`ToTensorOperation`]. A tensor destination requires both flags to be false.
///
/// # Example
///
/// The following is an example of a [`MaterializeInDestinationOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// bufferization.materialize_in_destination %source in restrict writable %destination :
///   (tensor<4xf32>, memref<4xf32>) -> ()
/// ```
///
/// Refer to the [official MLIR documentation](
/// https://mlir.llvm.org/docs/Dialects/BufferizationOps/#bufferizationmaterialize_in_destination-bufferizationmaterializeindestinationop)
/// for more information.
pub trait MaterializeInDestinationOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source tensor.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the tensor or memref destination.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the optional updated destination tensor.
    fn result_tensor(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() == 0 { Ok(None) } else { self.result(0).map(|result| Some(result.as_ref())) }
    }

    /// Returns whether the destination carries the `restrict` promise.
    fn is_restrict(&self) -> bool {
        self.has_attribute(RESTRICT_ATTRIBUTE)
    }

    /// Returns whether the destination is writable.
    fn is_writable(&self) -> bool {
        self.has_attribute(WRITABLE_ATTRIBUTE)
    }
}

mlir_op!(MaterializeInDestination);
mlir_op_trait!(MaterializeInDestination, ZeroRegions);
mlir_op_trait!(MaterializeInDestination, ZeroSuccessors);

/// Constructs a new detached/owned [`MaterializeInDestinationOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`MaterializeInDestinationOperation`] for more information on the operation semantics.
///
/// # Parameters
///
///   - `source`: Ranked tensor whose contents are materialized.
///   - `destination`: Ranked tensor or memref with the same runtime shape and element type as `source`.
///   - `restrict`: Whether the memref destination carries the aliasing promise described on [`ToTensorOperation`].
///   - `writable`: Must be true for memref destinations and false for tensor destinations.
///   - `location`: Source location to attach to the operation.
pub fn materialize_in_destination<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    restrict: bool,
    writable: bool,
    location: L,
) -> Result<DetachedMaterializeInDestinationOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::bufferization()?)?;
    let source_type = validate_tensor(source, "bufferization.materialize_in_destination")?;
    let destination_type = destination.r#type()?;
    let result_type = if let Some(destination_type) = destination_type.cast::<TensorTypeRef>() {
        if restrict || writable {
            return Err(Error::invalid_argument(
                "expected `restrict` and `writable` to be absent for tensor destination of \
                 `bufferization.materialize_in_destination`",
            ));
        }
        validate_shape_and_element_type(
            source_type.dimensions(),
            source_type.element_type()?,
            destination_type.dimensions(),
            destination_type.element_type()?,
            false,
            "bufferization.materialize_in_destination",
        )?;
        Some(destination_type)
    } else if let Some(destination_type) = destination_type.cast::<MemRefTypeRef>() {
        if !writable {
            return Err(Error::invalid_argument(
                "expected memref destination of `bufferization.materialize_in_destination` to be writable",
            ));
        }
        validate_shape_and_element_type(
            source_type.dimensions(),
            source_type.element_type()?,
            destination_type.dimensions(),
            destination_type.element_type()?,
            false,
            "bufferization.materialize_in_destination",
        )?;
        None
    } else {
        return Err(Error::invalid_argument(
            "expected tensor or memref destination for `bufferization.materialize_in_destination`",
        ));
    };
    let mut builder = OperationBuilder::new("bufferization.materialize_in_destination", location)
        .add_operand(source)
        .add_operand(destination);
    if restrict {
        builder = builder.add_attribute(RESTRICT_ATTRIBUTE, context.unit_attribute());
    }
    if writable {
        builder = builder.add_attribute(WRITABLE_ATTRIBUTE, context.unit_attribute());
    }
    if let Some(result_type) = result_type {
        builder = builder.add_result(result_type);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `bufferization::materialize_in_destination`"))
    })
}

/// Bufferization [`Operation`] that releases the storage underlying a tensor. The tensor must own the storage being
/// released; using it or an alias after deallocation is invalid.
///
/// # Example
///
/// The following is an example of a [`DeallocTensorOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// bufferization.dealloc_tensor %tensor : tensor<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation](
/// https://mlir.llvm.org/docs/Dialects/BufferizationOps/#bufferizationdealloc_tensor-bufferizationdealloctensorop)
/// for more information.
pub trait DeallocTensorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor whose storage is released.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(DeallocTensor);
mlir_op_trait!(DeallocTensor, ZeroRegions);
mlir_op_trait!(DeallocTensor, ZeroSuccessors);

/// Constructs a new detached/owned [`DeallocTensorOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DeallocTensorOperation`] for more information on the operation semantics.
pub fn dealloc_tensor<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedDeallocTensorOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::bufferization()?)?;
    validate_tensor(tensor, "bufferization.dealloc_tensor")?;
    OperationBuilder::new("bufferization.dealloc_tensor", location)
        .add_operand(tensor)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `bufferization::dealloc_tensor`"))
        })
}

/// Bufferization [`Operation`] that exposes a memref as a tensor with the same shape and element type. `restrict`
/// promises that this result is the only route through which tensor IR accesses the buffer or its aliases. Violating
/// this promise can lead to incorrect bufferization. One-Shot Bufferize requires this flag. `writable` allows writes
/// through the tensor to bufferize in place when other dependencies permit it. Without it, writes are bufferized out
/// of place to preserve the original buffer.
///
/// This operation does not itself copy the buffer. Together with [`ToBufferOperation`], it bridges tensor and buffer
/// representations during type conversion.
///
/// # Example
///
/// The following is an example of a [`ToTensorOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %tensor = bufferization.to_tensor %buffer restrict writable : memref<4xf32> to tensor<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation](
/// https://mlir.llvm.org/docs/Dialects/BufferizationOps/#bufferizationto_tensor-bufferizationtotensorop)
/// for more information.
pub trait ToTensorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source buffer.
    fn buffer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the tensor view.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }

    /// Returns whether the operation carries the exclusive tensor-access promise.
    fn is_restrict(&self) -> bool {
        self.has_attribute(RESTRICT_ATTRIBUTE)
    }

    /// Returns whether writes through the tensor are allowed to bufferize in place.
    fn is_writable(&self) -> bool {
        self.has_attribute(WRITABLE_ATTRIBUTE)
    }
}

mlir_op!(ToTensor);
mlir_op_trait!(ToTensor, OneResult);
mlir_op_trait!(ToTensor, ZeroRegions);
mlir_op_trait!(ToTensor, ZeroSuccessors);

/// Constructs a new detached/owned [`ToTensorOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ToTensorOperation`] for more information on the operation semantics.
///
/// # Parameters
///
///   - `buffer`: Ranked memref exposed as a tensor.
///   - `result_type`: Ranked tensor type with the same shape and element type as `buffer`.
///   - `restrict`: Whether the result is the only tensor-IR access to the buffer or its aliases.
///   - `writable`: Whether writes through the tensor may bufferize in place.
///   - `location`: Source location to attach to the operation.
pub fn to_tensor<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    buffer: ValueRef<'v, 'c, 't>,
    result_type: TensorTypeRef<'c, 't>,
    restrict: bool,
    writable: bool,
    location: L,
) -> Result<DetachedToTensorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::bufferization()?)?;
    let buffer_type = validate_memref(buffer, "bufferization.to_tensor")?;
    validate_shape_and_element_type(
        buffer_type.dimensions(),
        buffer_type.element_type()?,
        result_type.dimensions(),
        result_type.element_type()?,
        true,
        "bufferization.to_tensor",
    )?;
    let mut builder = OperationBuilder::new("bufferization.to_tensor", location)
        .add_operand(buffer)
        .add_result(result_type);
    if restrict {
        builder = builder.add_attribute(RESTRICT_ATTRIBUTE, context.unit_attribute());
    }
    if writable {
        builder = builder.add_attribute(WRITABLE_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `bufferization::to_tensor`"))
    })
}

/// Name of the unit [`Attribute`](crate::Attribute) that promises a buffer and its aliases will not be written.
pub const READ_ONLY_ATTRIBUTE: &str = "read_only";

/// Bufferization [`Operation`] that exposes the future buffer of a tensor. The result must have the same shape and
/// element type as the tensor. `read_only` promises that neither the returned buffer nor its aliases will be written.
/// This operation bridges type conversion and does not itself copy the tensor.
///
/// # Example
///
/// The following is an example of a [`ToBufferOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %buffer = bufferization.to_buffer %tensor read_only : tensor<4xf32> to memref<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation](
/// https://mlir.llvm.org/docs/Dialects/BufferizationOps/#bufferizationto_buffer-bufferizationtobufferop)
/// for more information.
pub trait ToBufferOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source tensor.
    fn tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the future buffer.
    fn buffer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }

    /// Returns whether the operation promises that the buffer and its aliases will not be written.
    fn is_read_only(&self) -> bool {
        self.has_attribute(READ_ONLY_ATTRIBUTE)
    }
}

mlir_op!(ToBuffer);
mlir_op_trait!(ToBuffer, AlwaysSpeculatable);
mlir_op_trait!(ToBuffer, NoMemoryEffect);
mlir_op_trait!(ToBuffer, OneResult);
mlir_op_trait!(ToBuffer, Pure);
mlir_op_trait!(ToBuffer, ZeroRegions);
mlir_op_trait!(ToBuffer, ZeroSuccessors);

/// Constructs a new detached/owned [`ToBufferOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ToBufferOperation`] for more information on the operation semantics.
///
/// # Parameters
///
///   - `tensor`: Ranked tensor whose future buffer is exposed.
///   - `result_type`: Ranked memref type with the same shape and element type as `tensor`.
///   - `read_only`: Whether the returned buffer and its aliases are promised never to be written.
///   - `location`: Source location to attach to the operation.
pub fn to_buffer<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    result_type: MemRefTypeRef<'c, 't>,
    read_only: bool,
    location: L,
) -> Result<DetachedToBufferOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::bufferization()?)?;
    let tensor_type = validate_tensor(tensor, "bufferization.to_buffer")?;
    validate_shape_and_element_type(
        tensor_type.dimensions(),
        tensor_type.element_type()?,
        result_type.dimensions(),
        result_type.element_type()?,
        true,
        "bufferization.to_buffer",
    )?;
    let mut builder = OperationBuilder::new("bufferization.to_buffer", location)
        .add_operand(tensor)
        .add_result(result_type);
    if read_only {
        builder = builder.add_attribute(READ_ONLY_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `bufferization::to_buffer`"))
    })
}

/// Bufferization [`Operation`] that conditionally releases buffers while retaining specified aliases. Each memref has
/// a corresponding `i1` condition. A buffer is deallocated only when its condition is true and it does not alias a
/// retained buffer. Aliases among the candidates are handled without double deallocation. Each retained memref produces
/// an updated ownership condition, in the same order as the retained operands. Candidate memrefs must refer to the
/// original allocations; retained memrefs may be arbitrary views. An updated condition is `true` when any candidate
/// alias of that retained memref has a `true` condition, and `false` when there is no such candidate.
///
/// # Example
///
/// The following is an example of a [`DeallocOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %ownership = bufferization.dealloc (%buffer : memref<4xf32>) if (%condition) retain (%view : memref<4xf32>)
/// ```
///
/// Refer to the [official MLIR documentation](
/// https://mlir.llvm.org/docs/Dialects/BufferizationOps/#bufferizationdealloc-bufferizationdeallocop)
/// for more information.
pub trait DeallocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the memrefs considered for deallocation.
    fn memrefs(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the deallocation conditions.
    fn conditions(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the retained memrefs.
    fn retained(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the updated ownership conditions for the retained memrefs.
    fn updated_conditions(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.results().map(|result| result.map(|result| result.as_ref())).collect()
    }
}

mlir_op!(Dealloc);
mlir_op_trait!(Dealloc, ZeroRegions);
mlir_op_trait!(Dealloc, ZeroSuccessors);

/// Constructs a new detached/owned [`DeallocOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DeallocOperation`] for more information on the operation semantics.
///
/// # Parameters
///
///   - `memrefs`: Original allocations considered for deallocation.
///   - `conditions`: One `i1` ownership condition per candidate memref, in the same order.
///   - `retained`: Memrefs whose aliases must remain allocated; each produces an updated ownership condition.
///   - `location`: Source location to attach to the operation.
pub fn dealloc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    memrefs: &[ValueRef<'v, 'c, 't>],
    conditions: &[ValueRef<'v, 'c, 't>],
    retained: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedDeallocOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::bufferization()?)?;
    if memrefs.len() != conditions.len() {
        return Err(Error::invalid_argument(
            "expected equal numbers of `memrefs` and `conditions` for `bufferization.dealloc`",
        ));
    }
    for memref in memrefs.iter().chain(retained) {
        validate_memref(*memref, "bufferization.dealloc")?;
    }
    let condition_type = context.signless_integer_type(1);
    for condition in conditions {
        if condition.r#type()? != condition_type.as_ref() {
            return Err(Error::invalid_argument("expected `i1` conditions for `bufferization.dealloc`"));
        }
    }
    let segment_sizes = [
        i32::try_from(memrefs.len())
            .map_err(|_| Error::invalid_argument("too many memrefs for `bufferization.dealloc`"))?,
        i32::try_from(conditions.len())
            .map_err(|_| Error::invalid_argument("too many conditions for `bufferization.dealloc`"))?,
        i32::try_from(retained.len())
            .map_err(|_| Error::invalid_argument("too many retained memrefs for `bufferization.dealloc`"))?,
    ];
    OperationBuilder::new("bufferization.dealloc", location)
        .add_operands(memrefs)
        .add_operands(conditions)
        .add_operands(retained)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes)?)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `bufferization::dealloc`"))
        })
}

/// Requires a ranked built-in tensor operand and returns its type.
fn validate_tensor<'v, 'c: 'v, 't: 'c>(
    value: ValueRef<'v, 'c, 't>,
    operation_name: &str,
) -> Result<TensorTypeRef<'c, 't>, Error> {
    let r#type = value.r#type()?;
    r#type
        .cast::<TensorTypeRef>()
        .ok_or_else(|| Error::invalid_argument(format!("expected ranked tensor operand for `{operation_name}`")))
}

/// Requires a ranked built-in memref operand and returns its type.
fn validate_memref<'v, 'c: 'v, 't: 'c>(
    value: ValueRef<'v, 'c, 't>,
    operation_name: &str,
) -> Result<MemRefTypeRef<'c, 't>, Error> {
    let r#type = value.r#type()?;
    r#type
        .cast::<MemRefTypeRef>()
        .ok_or_else(|| Error::invalid_argument(format!("expected ranked memref operand for `{operation_name}`")))
}

/// Checks element types and either exact dimension descriptors or compatible runtime shapes.
/// Dynamic dimensions are wildcards only for materialization, whose runtime shape equality remains a caller obligation.
fn validate_shape_and_element_type<'c, 't: 'c>(
    mut source_dimensions: impl Iterator<Item = Size>,
    source_element_type: TypeRef<'c, 't>,
    mut destination_dimensions: impl Iterator<Item = Size>,
    destination_element_type: TypeRef<'c, 't>,
    require_exact_shape: bool,
    operation_name: &str,
) -> Result<(), Error> {
    let shapes_match = source_dimensions.all(|source| {
        destination_dimensions.next().is_some_and(|destination| {
            source == destination || (!require_exact_shape && (source == Size::Dynamic || destination == Size::Dynamic))
        })
    }) && destination_dimensions.next().is_none();
    if !shapes_match {
        return Err(Error::invalid_argument(format!(
            "expected source and destination of `{operation_name}` to have compatible shapes",
        )));
    }
    if source_element_type != destination_element_type {
        return Err(Error::invalid_argument(format!(
            "expected source and destination of `{operation_name}` to have the same element type",
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Attribute, Block, Context, Operation, Size};

    use super::*;

    #[test]
    fn test_alloc_tensor() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Dynamic], None, location).unwrap();
        let mut block = context.block(&[(context.index_type(), location)]);
        assert!(matches!(
            alloc_tensor(&[], None, None, None, tensor_type, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected one dynamic size operand per dynamic result dimension of \
                    `bufferization.alloc_tensor`",
        ));
        let operation =
            alloc_tensor(&[block.argument(0).unwrap().into()], None, None, None, tensor_type, location).unwrap();
        assert_eq!(operation.dynamic_sizes().unwrap(), vec![block.argument(0).unwrap()]);
        assert_eq!(operation.copy(), Ok(None));
        assert_eq!(operation.size_hint(), Ok(None));
        assert_eq!(operation.memory_space(), Ok(None));
        assert_eq!(operation.tensor().unwrap().r#type().unwrap(), tensor_type);
        // Copy operands supply the dynamic shape, and optional operands retain their segment positions.
        let copy_block = context.block(&[(tensor_type.as_ref(), location), (context.index_type().as_ref(), location)]);
        let copy = copy_block.argument(0).unwrap().as_ref();
        let hint = copy_block.argument(1).unwrap().as_ref();
        let memory_space = context.integer_attribute(context.signless_integer_type(64), 1).as_ref();
        let copied = alloc_tensor(&[], Some(copy), Some(hint), Some(memory_space), tensor_type, location).unwrap();
        assert!(copied.verify());
        assert_eq!(copied.dynamic_sizes(), Ok(Vec::new()));
        assert_eq!(copied.copy(), Ok(Some(copy)));
        assert_eq!(copied.size_hint(), Ok(Some(hint)));
        assert_eq!(copied.memory_space(), Ok(Some(memory_space)));
        assert_eq!(copied.tensor().unwrap().r#type().unwrap(), tensor_type);
        assert!(matches!(
            alloc_tensor(&[hint], Some(copy), None, None, tensor_type, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected `bufferization.alloc_tensor` with `copy` to have no dynamic size operands",
        ));
        assert!(matches!(
            alloc_tensor(&[copy], None, None, None, tensor_type, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected dynamic size operands of `bufferization.alloc_tensor` to have index type",
        ));
        assert!(matches!(
            alloc_tensor(&[], Some(copy), Some(copy), None, tensor_type, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected `size_hint` of `bufferization.alloc_tensor` to have index type",
        ));
        let other_type = context.tensor_type(context.float64_type(), &[Size::Dynamic], None, location).unwrap();
        assert!(matches!(
            alloc_tensor(&[], Some(copy), None, None, other_type, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected `copy` and result of `bufferization.alloc_tensor` to have the same type",
        ));
        let operation = block.append_operation(operation).unwrap();
        block.append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap()).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                func::func(
                    "test_alloc_tensor",
                    func::FuncAttributes {
                        arguments: vec![context.index_type().into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @test_alloc_tensor(%arg0: index) -> tensor<?xf32> {
                    %0 = bufferization.alloc_tensor(%arg0) : tensor<?xf32>
                    return %0 : tensor<?xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_clone() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let memref_type =
            context.mem_ref_type(context.float32_type(), &[Size::Static(4)], None, None, location).unwrap();
        let mut block = context.block(&[(memref_type, location)]);
        let operation = clone(block.argument(0).unwrap().into(), memref_type, location).unwrap();
        assert_eq!(operation.input().unwrap(), block.argument(0).unwrap());
        assert_eq!(operation.output_memref().unwrap().r#type().unwrap(), memref_type);
        let invalid = context.block(&[(context.index_type(), location)]);
        assert!(matches!(
            clone(invalid.argument(0).unwrap().into(), memref_type, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected ranked memref operand for `bufferization.clone`",
        ));
        let operation = block.append_operation(operation).unwrap();
        block.append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap()).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                func::func(
                    "test_clone",
                    func::FuncAttributes {
                        arguments: vec![memref_type.into()],
                        results: vec![memref_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @test_clone(%arg0: memref<4xf32>) -> memref<4xf32> {
                    %0 = bufferization.clone %arg0 : memref<4xf32> to memref<4xf32>
                    return %0 : memref<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_materialize_in_destination() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(4)], None, location).unwrap();
        let memref_type =
            context.mem_ref_type(context.float32_type(), &[Size::Static(4)], None, None, location).unwrap();
        let wrong_shape_type =
            context.mem_ref_type(context.float32_type(), &[Size::Static(8)], None, None, location).unwrap();
        let wrong_element_type =
            context.mem_ref_type(context.float64_type(), &[Size::Static(4)], None, None, location).unwrap();
        let invalid_block = context.block(&[
            (tensor_type.as_ref(), location),
            (wrong_shape_type.as_ref(), location),
            (wrong_element_type.as_ref(), location),
        ]);
        assert!(matches!(
            materialize_in_destination(
                invalid_block.argument(0).unwrap().into(),
                invalid_block.argument(1).unwrap().into(),
                false,
                true,
                location,
            ),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected source and destination of `bufferization.materialize_in_destination` to have \
                    compatible shapes",
        ));
        assert!(matches!(
            materialize_in_destination(
                invalid_block.argument(0).unwrap().into(),
                invalid_block.argument(2).unwrap().into(),
                false,
                true,
                location,
            ),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected source and destination of `bufferization.materialize_in_destination` to have \
                    the same element type",
        ));
        let mut block = context.block(&[(tensor_type.as_ref(), location), (memref_type.as_ref(), location)]);
        assert!(matches!(
            materialize_in_destination(
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                false,
                false,
                location,
            ),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected memref destination of `bufferization.materialize_in_destination` to be writable",
        ));
        let operation = materialize_in_destination(
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
            true,
            true,
            location,
        )
        .unwrap();
        assert_eq!(operation.source().unwrap(), block.argument(0).unwrap());
        assert_eq!(operation.destination().unwrap(), block.argument(1).unwrap());
        assert_eq!(operation.result_tensor(), Ok(None));
        assert!(operation.is_restrict());
        assert!(operation.is_writable());
        block.append_operation(operation).unwrap();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                func::func(
                    "test_materialize",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into(), memref_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @test_materialize(%arg0: tensor<4xf32>, %arg1: memref<4xf32>) {
                    bufferization.materialize_in_destination %arg0 in restrict writable %arg1 : \
                        (tensor<4xf32>, memref<4xf32>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_materialize_in_destination_tensor() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(4)], None, location).unwrap();
        // Tensor destinations return an updated tensor and cannot carry buffer-only promises.
        let mut tensor_block = context.block(&[(tensor_type, location), (tensor_type, location)]);
        let source = tensor_block.argument(0).unwrap().as_ref();
        let destination = tensor_block.argument(1).unwrap().as_ref();
        let operation = materialize_in_destination(source, destination, false, false, location).unwrap();
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(operation.destination().unwrap(), destination);
        assert_eq!(operation.result_tensor().unwrap(), Some(operation.result(0).unwrap().as_ref()));
        assert_eq!(operation.result_tensor().unwrap().unwrap().r#type().unwrap(), tensor_type);
        assert!(!operation.is_restrict());
        assert!(!operation.is_writable());
        assert!(matches!(
            materialize_in_destination(source, destination, true, false, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected `restrict` and `writable` to be absent for tensor destination of \
                    `bufferization.materialize_in_destination`",
        ));
        assert!(matches!(
            materialize_in_destination(source, destination, false, true, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected `restrict` and `writable` to be absent for tensor destination of \
                    `bufferization.materialize_in_destination`",
        ));
        let dynamic_type = context.tensor_type(context.float32_type(), &[Size::Dynamic], None, location).unwrap();
        let dynamic_block = context.block(&[(dynamic_type, location)]);
        let dynamic_destination = dynamic_block.argument(0).unwrap().as_ref();
        let dynamic = materialize_in_destination(source, dynamic_destination, false, false, location).unwrap();
        assert!(dynamic.verify());
        assert_eq!(dynamic.result_tensor().unwrap().unwrap().r#type().unwrap(), dynamic_type);
        let result = tensor_block.append_operation(operation).unwrap().result(0).unwrap();
        tensor_block.append_operation(func::r#return(&[result], location).unwrap()).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                func::func(
                    "test_materialize_tensor",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into(), tensor_type.into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    tensor_block.try_into().unwrap(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @test_materialize_tensor(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = bufferization.materialize_in_destination %arg0 in %arg1 : \
                        (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_dealloc_tensor() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(4)], None, location).unwrap();
        let mut block = context.block(&[(tensor_type, location)]);
        let operation = dealloc_tensor(block.argument(0).unwrap().into(), location).unwrap();
        assert_eq!(operation.tensor().unwrap(), block.argument(0).unwrap());
        let invalid = context.block(&[(context.index_type(), location)]);
        assert!(matches!(
            dealloc_tensor(invalid.argument(0).unwrap().into(), location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected ranked tensor operand for `bufferization.dealloc_tensor`",
        ));
        block.append_operation(operation).unwrap();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                func::func(
                    "test_dealloc_tensor",
                    func::FuncAttributes { arguments: vec![tensor_type.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @test_dealloc_tensor(%arg0: tensor<4xf32>) {
                    bufferization.dealloc_tensor %arg0 : tensor<4xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_tensor() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(4)], None, location).unwrap();
        let memref_type =
            context.mem_ref_type(context.float32_type(), &[Size::Static(4)], None, None, location).unwrap();
        let wrong_shape_type = context.tensor_type(context.float32_type(), &[Size::Static(8)], None, location).unwrap();
        let wrong_element_type =
            context.tensor_type(context.float64_type(), &[Size::Static(4)], None, location).unwrap();
        let mut block = context.block(&[(memref_type, location)]);
        assert!(matches!(
            to_tensor(block.argument(0).unwrap().into(), wrong_shape_type, true, true, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected source and destination of `bufferization.to_tensor` to have compatible shapes",
        ));
        assert!(matches!(
            to_tensor(block.argument(0).unwrap().into(), wrong_element_type, true, true, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected source and destination of `bufferization.to_tensor` to have the same element \
                    type",
        ));
        let dynamic_type = context.tensor_type(context.float32_type(), &[Size::Dynamic], None, location).unwrap();
        assert!(matches!(
            to_tensor(block.argument(0).unwrap().into(), dynamic_type, true, true, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected source and destination of `bufferization.to_tensor` to have compatible shapes",
        ));
        let wrong_rank = context
            .tensor_type(context.float32_type(), &[Size::Static(4), Size::Static(1)], None, location)
            .unwrap();
        assert!(matches!(
            to_tensor(block.argument(0).unwrap().into(), wrong_rank, true, true, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected source and destination of `bufferization.to_tensor` to have compatible shapes",
        ));
        let operation = to_tensor(block.argument(0).unwrap().into(), tensor_type, true, true, location).unwrap();
        assert_eq!(operation.buffer().unwrap(), block.argument(0).unwrap());
        assert!(operation.is_restrict());
        assert!(operation.is_writable());
        assert_eq!(operation.tensor().unwrap().r#type().unwrap(), tensor_type);
        let without_promises =
            to_tensor(block.argument(0).unwrap().into(), tensor_type, false, false, location).unwrap();
        assert!(without_promises.verify());
        assert!(!without_promises.is_restrict());
        assert!(!without_promises.is_writable());
        let operation = block.append_operation(operation).unwrap();
        block.append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap()).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                func::func(
                    "test_to_tensor",
                    func::FuncAttributes {
                        arguments: vec![memref_type.into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @test_to_tensor(%arg0: memref<4xf32>) -> tensor<4xf32> {
                    %0 = bufferization.to_tensor %arg0 restrict writable : memref<4xf32> to tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_buffer() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(4)], None, location).unwrap();
        let memref_type =
            context.mem_ref_type(context.float32_type(), &[Size::Static(4)], None, None, location).unwrap();
        let wrong_shape_type =
            context.mem_ref_type(context.float32_type(), &[Size::Static(8)], None, None, location).unwrap();
        let wrong_element_type =
            context.mem_ref_type(context.float64_type(), &[Size::Static(4)], None, None, location).unwrap();
        let mut block = context.block(&[(tensor_type, location)]);
        assert!(matches!(
            to_buffer(block.argument(0).unwrap().into(), wrong_shape_type, true, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected source and destination of `bufferization.to_buffer` to have compatible shapes",
        ));
        assert!(matches!(
            to_buffer(block.argument(0).unwrap().into(), wrong_element_type, true, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected source and destination of `bufferization.to_buffer` to have the same element \
                    type",
        ));
        let operation = to_buffer(block.argument(0).unwrap().into(), memref_type, true, location).unwrap();
        assert_eq!(operation.tensor().unwrap(), block.argument(0).unwrap());
        assert!(operation.is_read_only());
        assert_eq!(operation.buffer().unwrap().r#type().unwrap(), memref_type);
        let without_promises = to_buffer(block.argument(0).unwrap().into(), memref_type, false, location).unwrap();
        assert!(without_promises.verify());
        assert!(!without_promises.is_read_only());
        let operation = block.append_operation(operation).unwrap();
        block.append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap()).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                func::func(
                    "test_to_buffer",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![memref_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @test_to_buffer(%arg0: tensor<4xf32>) -> memref<4xf32> {
                    %0 = bufferization.to_buffer %arg0 read_only : tensor<4xf32> to memref<4xf32>
                    return %0 : memref<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_dealloc() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let memref_type =
            context.mem_ref_type(context.float32_type(), &[Size::Static(4)], None, None, location).unwrap();
        let condition_type = context.signless_integer_type(1);
        let mut block = context.block(&[
            (memref_type.as_ref(), location),
            (condition_type.as_ref(), location),
            (memref_type.as_ref(), location),
        ]);
        let operation = dealloc(
            &[block.argument(0).unwrap().into()],
            &[block.argument(1).unwrap().into()],
            &[block.argument(2).unwrap().into()],
            location,
        )
        .unwrap();
        assert_eq!(operation.memrefs().unwrap(), vec![block.argument(0).unwrap()]);
        assert_eq!(operation.conditions().unwrap(), vec![block.argument(1).unwrap()]);
        assert_eq!(operation.retained().unwrap(), vec![block.argument(2).unwrap()]);
        assert_eq!(operation.updated_conditions().unwrap(), vec![operation.result(0).unwrap()]);
        assert_eq!(operation.updated_conditions().unwrap()[0].r#type().unwrap(), condition_type);
        let empty = dealloc(&[], &[], &[], location).unwrap();
        assert!(empty.verify());
        assert_eq!(empty.memrefs(), Ok(Vec::new()));
        assert_eq!(empty.conditions(), Ok(Vec::new()));
        assert_eq!(empty.retained(), Ok(Vec::new()));
        assert_eq!(empty.updated_conditions(), Ok(Vec::new()));
        assert!(matches!(
            dealloc(&[block.argument(0).unwrap().into()], &[], &[], location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected equal numbers of `memrefs` and `conditions` for `bufferization.dealloc`",
        ));
        assert!(matches!(
            dealloc(&[block.argument(0).unwrap().into()], &[block.argument(0).unwrap().into()], &[], location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected `i1` conditions for `bufferization.dealloc`",
        ));
        let operation = block.append_operation(operation).unwrap();
        block.append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap()).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                func::func(
                    "test_dealloc",
                    func::FuncAttributes {
                        arguments: vec![memref_type.into(), condition_type.into(), memref_type.into()],
                        results: vec![condition_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @test_dealloc(%arg0: memref<4xf32>, %arg1: i1, %arg2: memref<4xf32>) -> i1 {
                    %0 = bufferization.dealloc (%arg0 : memref<4xf32>) if (%arg1) retain (%arg2 : memref<4xf32>)
                    return %0 : i1
                  }
                }
            "},
        );
    }
}
