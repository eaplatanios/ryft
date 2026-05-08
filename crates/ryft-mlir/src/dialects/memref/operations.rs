use crate::dialects::arith::{AtomicRmwKind, AtomicRmwKindAttributeRef};
use crate::{
    AffineMap, ArrayAttributeRef, Attribute, AttributeRef, DetachedOp, DetachedRegion, DialectHandle, Error,
    FlatSymbolRefAttributeRef, Location, MemRefTypeRef, OneRegion, OneResult, Operation, OperationBuilder, RegionRef,
    SYMBOL_NAME_ATTRIBUTE, SYMBOL_VISIBILITY_ATTRIBUTE, Size, StringAttributeRef, Symbol, SymbolVisibility,
    TryIntoWithContext, Type, TypeRef, Value, ValueRef, mlir_op, mlir_op_trait,
};

/// An index entry that is either known statically or provided by an SSA value at runtime.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StaticOrDynamicIndex<'v, 'c: 'v, 't: 'c> {
    /// Statically known index value.
    Static(i64),

    /// Dynamically provided index value.
    Dynamic(ValueRef<'v, 'c, 't>),
}

impl<'v, 'c: 'v, 't: 'c> StaticOrDynamicIndex<'v, 'c, 't> {
    /// Returns the static value, if this entry is statically known.
    pub fn static_value(&self) -> Option<i64> {
        match self {
            Self::Static(value) => Some(*value),
            Self::Dynamic(_) => None,
        }
    }

    /// Returns the dynamic SSA value, if this entry is dynamically provided.
    pub fn dynamic_value(&self) -> Option<ValueRef<'v, 'c, 't>> {
        match self {
            Self::Static(_) => None,
            Self::Dynamic(value) => Some(*value),
        }
    }
}

/// Name of the [`Attribute`] that stores byte alignment requirements for MemRef operations.
pub const ALIGNMENT_ATTRIBUTE: &str = "alignment";

/// Operation trait for the `memref.assume_alignment` operation.
pub trait AssumeAlignmentOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the memref whose alignment is being assumed.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the assumed byte alignment.
    fn alignment(&self) -> Result<i32, Error> {
        i32::try_from(self.integer_attribute(ALIGNMENT_ATTRIBUTE)?.signless_value())
            .map_err(|_| Error::invalid_argument("invalid `alignment` attribute in `memref::assume_alignment`"))
    }
}

mlir_op!(AssumeAlignment);
mlir_op_trait!(AssumeAlignment, AlwaysSpeculatable);
mlir_op_trait!(AssumeAlignment, NoMemoryEffect);
mlir_op_trait!(AssumeAlignment, OneResult);
mlir_op_trait!(AssumeAlignment, Pure);
mlir_op_trait!(AssumeAlignment, ZeroRegions);
mlir_op_trait!(AssumeAlignment, ZeroSuccessors);

/// Constructs a new detached [`AssumeAlignmentOperation`].
pub fn assume_alignment<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    memref: ValueRef<'v, 'c, 't>,
    alignment: i32,
    location: L,
) -> Result<DetachedAssumeAlignmentOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.assume_alignment", location)
        .add_operand(memref)
        .add_attribute(
            ALIGNMENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), alignment.into()),
        )
        .add_result(memref.r#type()?)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::assume_alignment`"))
        })
}

/// Operation trait for the `memref.distinct_objects` operation.
pub trait DistinctObjectsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input memrefs asserted to be mutually non-aliasing.
    fn inputs(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns the memref values carrying the non-aliasing assumption.
    fn outputs(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.results().map(|result| result.map(|result| result.as_ref())).collect()
    }
}

mlir_op!(DistinctObjects);
mlir_op_trait!(DistinctObjects, AlwaysSpeculatable);
mlir_op_trait!(DistinctObjects, NoMemoryEffect);
mlir_op_trait!(DistinctObjects, Pure);
mlir_op_trait!(DistinctObjects, ZeroRegions);
mlir_op_trait!(DistinctObjects, ZeroSuccessors);

/// Constructs a new detached [`DistinctObjectsOperation`].
pub fn distinct_objects<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedDistinctObjectsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let result_types = operands.iter().map(|operand| operand.r#type()).collect::<Result<Vec<_>, _>>()?;
    OperationBuilder::new("memref.distinct_objects", location)
        .add_operands(operands)
        .add_results(&result_types)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::distinct_objects`"))
        })
}

/// Name of the [`Attribute`] that stores operand segment sizes for variadic MemRef operations.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Operation trait shared by `memref.alloc` and `memref.alloca`.
pub trait AllocLikeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the dynamic dimension operands.
    fn dynamic_sizes(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the symbolic operands bound to the memref layout map.
    fn symbol_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Result<Option<i64>, Error> {
        if self.has_attribute(ALIGNMENT_ATTRIBUTE) {
            self.integer_attribute(ALIGNMENT_ATTRIBUTE).map(|attribute| Some(attribute.signless_value()))
        } else {
            Ok(None)
        }
    }

    /// Returns the allocated memref result.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

/// Operation trait for the `memref.alloc` operation.
pub trait AllocOperation<'o, 'c: 'o, 't: 'c>: AllocLikeOperation<'o, 'c, 't> {}

mlir_op!(Alloc);
mlir_op_trait!(Alloc, OneResult);
mlir_op_trait!(Alloc, ZeroRegions);
mlir_op_trait!(Alloc, ZeroSuccessors);
mlir_op_trait!(Alloc, @local AllocLikeOperation);

/// Constructs a new detached [`AllocOperation`].
pub fn alloc<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    dynamic_sizes: &[ValueRef<'v, 'c, 't>],
    symbol_operands: &[ValueRef<'v, 'c, 't>],
    memref_type: T,
    alignment: Option<i64>,
    location: L,
) -> Result<DetachedAllocOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let mut builder = OperationBuilder::new("memref.alloc", location)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[dynamic_sizes.len() as i32, symbol_operands.len() as i32])?,
        )
        .add_operands(dynamic_sizes)
        .add_operands(symbol_operands)
        .add_result(memref_type);
    if let Some(alignment) = alignment {
        builder = builder.add_attribute(
            ALIGNMENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), alignment),
        );
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::alloc`"))
    })
}

/// Operation trait for the `memref.realloc` operation.
pub trait ReallocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref being reallocated.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional dynamic result size.
    fn dynamic_result_size(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() <= 1 { Ok(None) } else { self.operand_value(1).map(Some) }
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Result<Option<i64>, Error> {
        if self.has_attribute(ALIGNMENT_ATTRIBUTE) {
            self.integer_attribute(ALIGNMENT_ATTRIBUTE).map(|attribute| Some(attribute.signless_value()))
        } else {
            Ok(None)
        }
    }

    /// Returns the reallocated memref.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(Realloc);
mlir_op_trait!(Realloc, OneResult);
mlir_op_trait!(Realloc, ZeroRegions);
mlir_op_trait!(Realloc, ZeroSuccessors);

/// Constructs a new detached [`ReallocOperation`].
pub fn realloc<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    dynamic_result_size: Option<ValueRef<'v, 'c, 't>>,
    result_type: T,
    alignment: Option<i64>,
    location: L,
) -> Result<DetachedReallocOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let mut builder = OperationBuilder::new("memref.realloc", location).add_operand(source).add_result(result_type);
    if let Some(dynamic_result_size) = dynamic_result_size {
        builder = builder.add_operand(dynamic_result_size);
    }
    if let Some(alignment) = alignment {
        builder = builder.add_attribute(
            ALIGNMENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), alignment),
        );
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::realloc`"))
    })
}

/// Operation trait for the `memref.alloca` operation.
pub trait AllocaOperation<'o, 'c: 'o, 't: 'c>: AllocLikeOperation<'o, 'c, 't> {}

mlir_op!(Alloca);
mlir_op_trait!(Alloca, OneResult);
mlir_op_trait!(Alloca, ZeroRegions);
mlir_op_trait!(Alloca, ZeroSuccessors);
mlir_op_trait!(Alloca, @local AllocLikeOperation);

/// Constructs a new detached [`AllocaOperation`].
pub fn alloca<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    dynamic_sizes: &[ValueRef<'v, 'c, 't>],
    symbol_operands: &[ValueRef<'v, 'c, 't>],
    memref_type: T,
    alignment: Option<i64>,
    location: L,
) -> Result<DetachedAllocaOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let mut builder = OperationBuilder::new("memref.alloca", location)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[dynamic_sizes.len() as i32, symbol_operands.len() as i32])?,
        )
        .add_operands(dynamic_sizes)
        .add_operands(symbol_operands)
        .add_result(memref_type);
    if let Some(alignment) = alignment {
        builder = builder.add_attribute(
            ALIGNMENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), alignment),
        );
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::alloca`"))
    })
}

/// Operation trait for the `memref.alloca_scope` operation.
pub trait AllocaScopeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> {
    /// Returns the alloca scope body.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.body_region()
    }
}

mlir_op!(AllocaScope);
mlir_op_trait!(AllocaScope, AutomaticAllocationScope);
mlir_op_trait!(AllocaScope, NoRegionArguments);
mlir_op_trait!(AllocaScope, OneRegion);
mlir_op_trait!(AllocaScope, SingleBlockRegions);
mlir_op_trait!(AllocaScope, ZeroSuccessors);

/// Constructs a new detached [`AllocaScopeOperation`].
pub fn alloca_scope<'c, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    result_types: &[T],
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedAllocaScopeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.alloca_scope", location)
        .add_results(result_types)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::alloca_scope`"))
        })
}

/// Operation trait for the `memref.alloca_scope.return` operation.
pub trait AllocaScopeReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the values yielded from the alloca scope.
    fn values(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }
}

mlir_op!(AllocaScopeReturn);
mlir_op_trait!(AllocaScopeReturn, AlwaysSpeculatable);
mlir_op_trait!(AllocaScopeReturn, NoMemoryEffect);
mlir_op_trait!(AllocaScopeReturn, Pure);
mlir_op_trait!(AllocaScopeReturn, ReturnLike);
mlir_op_trait!(AllocaScopeReturn, ZeroRegions);
mlir_op_trait!(AllocaScopeReturn, ZeroSuccessors);

/// Constructs a new detached [`AllocaScopeReturnOperation`].
pub fn alloca_scope_return<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedAllocaScopeReturnOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.alloca_scope.return", location).add_operands(values).build().and_then(
        |operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::alloca_scope_return`"))
        },
    )
}

/// Operation trait for the `memref.cast` operation.
pub trait CastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the destination memref.
    fn dest(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(Cast);
mlir_op_trait!(Cast, AlwaysSpeculatable);
mlir_op_trait!(Cast, MemRefsNormalizable);
mlir_op_trait!(Cast, NoMemoryEffect);
mlir_op_trait!(Cast, OneResult);
mlir_op_trait!(Cast, Pure);
mlir_op_trait!(Cast, ZeroRegions);
mlir_op_trait!(Cast, ZeroSuccessors);

/// Constructs a new detached [`CastOperation`].
pub fn cast<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    dest_type: T,
    location: L,
) -> Result<DetachedCastOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.cast", location)
        .add_operand(source)
        .add_result(dest_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::cast`"))
        })
}

/// Operation trait for the `memref.copy` operation.
pub trait CopyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the target memref.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(Copy);
mlir_op_trait!(Copy, ZeroRegions);
mlir_op_trait!(Copy, ZeroSuccessors);

/// Constructs a new detached [`CopyOperation`].
pub fn copy<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    target: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedCopyOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.copy", location)
        .add_operand(source)
        .add_operand(target)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::copy`"))
        })
}

/// Operation trait for the `memref.dealloc` operation.
pub trait DeallocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the memref being deallocated.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(Dealloc);
mlir_op_trait!(Dealloc, MemRefsNormalizable);
mlir_op_trait!(Dealloc, ZeroRegions);
mlir_op_trait!(Dealloc, ZeroSuccessors);

/// Constructs a new detached [`DeallocOperation`].
pub fn dealloc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    memref: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedDeallocOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.dealloc", location)
        .add_operand(memref)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::dealloc`"))
        })
}

/// Operation trait for the `memref.dim` operation.
pub trait DimOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the dimension index.
    fn index(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(Dim);
mlir_op_trait!(Dim, MemRefsNormalizable);
mlir_op_trait!(Dim, NoMemoryEffect);
mlir_op_trait!(Dim, OneResult);
mlir_op_trait!(Dim, ZeroRegions);
mlir_op_trait!(Dim, ZeroSuccessors);

/// Constructs a new detached [`DimOperation`].
pub fn dim<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    index: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedDimOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.dim", location)
        .add_operand(source)
        .add_operand(index)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::dim`"))
        })
}

/// Operation trait for the `memref.dma_start` operation.
pub trait DmaStartOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the rank of the source memref.
    fn source_rank(&self) -> Result<usize, Error> {
        self.source()?.r#type()?.cast::<MemRefTypeRef>().map(|r#type| r#type.rank()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "invalid source memref type in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the source memref indices.
    fn source_indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let source_rank = self.source_rank()?;
        (1..1 + source_rank).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the destination memref.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let source_rank = self.source_rank()?;
        self.operand_value(1 + source_rank)
    }

    /// Returns the rank of the destination memref.
    fn destination_rank(&self) -> Result<usize, Error> {
        self.destination()?.r#type()?.cast::<MemRefTypeRef>().map(|r#type| r#type.rank()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "invalid destination memref type in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the destination memref indices.
    fn destination_indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let source_rank = self.source_rank()?;
        let destination_rank = self.destination_rank()?;
        let start = 2 + source_rank;
        (start..start + destination_rank).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the number of elements transferred by this DMA operation.
    fn num_elements(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let source_rank = self.source_rank()?;
        let destination_rank = self.destination_rank()?;
        self.operand_value(2 + source_rank + destination_rank)
    }

    /// Returns the tag memref used to synchronize with the matching `memref.dma_wait`.
    fn tag(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let source_rank = self.source_rank()?;
        let destination_rank = self.destination_rank()?;
        self.operand_value(3 + source_rank + destination_rank)
    }

    /// Returns the rank of the tag memref.
    fn tag_rank(&self) -> Result<usize, Error> {
        self.tag()?.r#type()?.cast::<MemRefTypeRef>().map(|r#type| r#type.rank()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "invalid tag memref type in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the tag memref indices.
    fn tag_indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let source_rank = self.source_rank()?;
        let destination_rank = self.destination_rank()?;
        let tag_rank = self.tag_rank()?;
        let start = 4 + source_rank + destination_rank;
        (start..start + tag_rank).map(|index| self.operand_value(index)).collect()
    }

    /// Returns `true` if this DMA operation has explicit stride operands.
    fn is_strided(&self) -> Result<bool, Error> {
        let source_rank = self.source_rank()?;
        let destination_rank = self.destination_rank()?;
        let tag_rank = self.tag_rank()?;
        Ok(self.operand_count() != 4 + source_rank + destination_rank + tag_rank)
    }

    /// Returns the optional stride operand.
    fn stride(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.is_strided()? { self.operand_value(self.operand_count() - 2).map(Some) } else { Ok(None) }
    }

    /// Returns the optional elements-per-stride operand.
    fn elements_per_stride(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.is_strided()? { self.operand_value(self.operand_count() - 1).map(Some) } else { Ok(None) }
    }
}

mlir_op!(DmaStart);
mlir_op_trait!(DmaStart, ZeroRegions);
mlir_op_trait!(DmaStart, ZeroSuccessors);

/// Constructs a new detached [`DmaStartOperation`].
pub fn dma_start<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    source_indices: &[ValueRef<'v, 'c, 't>],
    destination: ValueRef<'v, 'c, 't>,
    destination_indices: &[ValueRef<'v, 'c, 't>],
    num_elements: ValueRef<'v, 'c, 't>,
    tag: ValueRef<'v, 'c, 't>,
    tag_indices: &[ValueRef<'v, 'c, 't>],
    stride: Option<ValueRef<'v, 'c, 't>>,
    elements_per_stride: Option<ValueRef<'v, 'c, 't>>,
    location: L,
) -> Result<DetachedDmaStartOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let builder = OperationBuilder::new("memref.dma_start", location)
        .add_operand(source)
        .add_operands(source_indices)
        .add_operand(destination)
        .add_operands(destination_indices)
        .add_operand(num_elements)
        .add_operand(tag)
        .add_operands(tag_indices);
    let builder = match (stride, elements_per_stride) {
        (Some(stride), Some(elements_per_stride)) => builder.add_operand(stride).add_operand(elements_per_stride),
        (None, None) => builder,
        _ => {
            return Err(Error::invalid_argument("`memref::dma_start` requires either both stride operands or neither"));
        }
    };
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::dma_start`"))
    })
}

/// Operation trait for the `memref.dma_wait` operation.
pub trait DmaWaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tag memref.
    fn tag(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the rank of the tag memref.
    fn tag_rank(&self) -> Result<usize, Error> {
        self.tag()?.r#type()?.cast::<MemRefTypeRef>().map(|r#type| r#type.rank()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "invalid tag memref type in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the tag memref indices.
    fn tag_indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let tag_rank = self.tag_rank()?;
        (1..1 + tag_rank).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the number of elements associated with the DMA operation.
    fn num_elements(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let tag_rank = self.tag_rank()?;
        self.operand_value(1 + tag_rank)
    }
}

mlir_op!(DmaWait);
mlir_op_trait!(DmaWait, ZeroRegions);
mlir_op_trait!(DmaWait, ZeroSuccessors);

/// Constructs a new detached [`DmaWaitOperation`].
pub fn dma_wait<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tag: ValueRef<'v, 'c, 't>,
    tag_indices: &[ValueRef<'v, 'c, 't>],
    num_elements: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedDmaWaitOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.dma_wait", location)
        .add_operand(tag)
        .add_operands(tag_indices)
        .add_operand(num_elements)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::dma_wait`"))
        })
}

/// Operation trait for the `memref.extract_aligned_pointer_as_index` operation.
pub trait ExtractAlignedPointerAsIndexOperation<'o, 'c: 'o, 't: 'c>:
    Operation<'o, 'c, 't> + OneResult<'o, 'c, 't>
{
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the extracted aligned pointer represented as an index value.
    fn aligned_pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(ExtractAlignedPointerAsIndex);
mlir_op_trait!(ExtractAlignedPointerAsIndex, AlwaysSpeculatable);
mlir_op_trait!(ExtractAlignedPointerAsIndex, NoMemoryEffect);
mlir_op_trait!(ExtractAlignedPointerAsIndex, OneResult);
mlir_op_trait!(ExtractAlignedPointerAsIndex, Pure);
mlir_op_trait!(ExtractAlignedPointerAsIndex, ZeroRegions);
mlir_op_trait!(ExtractAlignedPointerAsIndex, ZeroSuccessors);

/// Constructs a new detached [`ExtractAlignedPointerAsIndexOperation`].
pub fn extract_aligned_pointer_as_index<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedExtractAlignedPointerAsIndexOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.extract_aligned_pointer_as_index", location)
        .add_operand(source)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| {
                Error::invalid_argument("invalid arguments to `memref::extract_aligned_pointer_as_index`")
            })
        })
}

/// Operation trait for the `memref.extract_strided_metadata` operation.
pub trait ExtractStridedMetadataOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the rank of the source memref.
    fn source_rank(&self) -> Result<usize, Error> {
        self.source()?.r#type()?.cast::<MemRefTypeRef>().map(|r#type| r#type.rank()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "invalid source memref type in `{}`",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the extracted zero-rank base buffer.
    fn base_buffer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0).map(|result| result.as_ref())
    }

    /// Returns the extracted offset.
    fn offset(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.as_ref().result(1).map(|result| result.as_ref())
    }

    /// Returns the extracted dynamic size results.
    fn sizes(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let rank = self.source_rank()?;
        (2..2 + rank).map(|index| self.as_ref().result(index).map(|result| result.as_ref())).collect()
    }

    /// Returns the extracted dynamic stride results.
    fn strides(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let rank = self.source_rank()?;
        (2 + rank..2 + 2 * rank)
            .map(|index| self.as_ref().result(index).map(|result| result.as_ref()))
            .collect()
    }
}

mlir_op!(ExtractStridedMetadata);
mlir_op_trait!(ExtractStridedMetadata, AlwaysSpeculatable);
mlir_op_trait!(ExtractStridedMetadata, NoMemoryEffect);
mlir_op_trait!(ExtractStridedMetadata, Pure);
mlir_op_trait!(ExtractStridedMetadata, ZeroRegions);
mlir_op_trait!(ExtractStridedMetadata, ZeroSuccessors);

/// Constructs a new detached [`ExtractStridedMetadataOperation`].
pub fn extract_strided_metadata<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    base_buffer_type: T,
    location: L,
) -> Result<DetachedExtractStridedMetadataOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let rank = source
        .r#type()?
        .cast::<MemRefTypeRef>()
        .ok_or_else(|| Error::invalid_argument("invalid source memref type in `memref::extract_strided_metadata`"))?
        .rank();
    let index_types = vec![context.index_type(); 1 + 2 * rank];
    OperationBuilder::new("memref.extract_strided_metadata", location)
        .add_operand(source)
        .add_result(base_buffer_type)
        .add_results(&index_types)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::extract_strided_metadata`"))
        })
}

/// Operation trait for the `memref.generic_atomic_rmw` operation.
pub trait GenericAtomicRmwOperation<'o, 'c: 'o, 't: 'c>:
    Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> + OneResult<'o, 'c, 't>
{
    /// Returns the memref being read and updated atomically.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the index operands used to access the memref.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (1..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the atomic read-modify-write body.
    fn atomic_body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.body_region()
    }

    /// Returns the latest stored value.
    fn latest_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(GenericAtomicRmw);
mlir_op_trait!(GenericAtomicRmw, OneRegion);
mlir_op_trait!(GenericAtomicRmw, OneResult);
mlir_op_trait!(GenericAtomicRmw, SingleBlockRegions);
mlir_op_trait!(GenericAtomicRmw, ZeroSuccessors);

/// Constructs a new detached [`GenericAtomicRmwOperation`].
pub fn generic_atomic_rmw<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    memref: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    result_type: T,
    atomic_body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedGenericAtomicRmwOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.generic_atomic_rmw", location)
        .add_operand(memref)
        .add_operands(indices)
        .add_result(result_type)
        .add_region(atomic_body)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::generic_atomic_rmw`"))
        })
}

/// Operation trait for the `memref.atomic_yield` operation.
pub trait AtomicYieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the yielded value.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(AtomicYield);
mlir_op_trait!(AtomicYield, AlwaysSpeculatable);
mlir_op_trait!(AtomicYield, NoMemoryEffect);
mlir_op_trait!(AtomicYield, Pure);
mlir_op_trait!(AtomicYield, ReturnLike);
mlir_op_trait!(AtomicYield, ZeroRegions);
mlir_op_trait!(AtomicYield, ZeroSuccessors);

/// Constructs a new detached [`AtomicYieldOperation`].
pub fn atomic_yield<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedAtomicYieldOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.atomic_yield", location)
        .add_operand(value)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::atomic_yield`"))
        })
}

/// Name of the [`Attribute`] that stores the referenced global symbol name.
pub const NAME_ATTRIBUTE: &str = "name";

/// Operation trait for the `memref.get_global` operation.
pub trait GetGlobalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the referenced global symbol.
    fn name(&self) -> Result<FlatSymbolRefAttributeRef<'c, 't>, Error> {
        self.flat_symbol_ref_attribute(NAME_ATTRIBUTE)
    }

    /// Returns the retrieved global memref.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(GetGlobal);
mlir_op_trait!(GetGlobal, AlwaysSpeculatable);
mlir_op_trait!(GetGlobal, NoMemoryEffect);
mlir_op_trait!(GetGlobal, OneResult);
mlir_op_trait!(GetGlobal, Pure);
mlir_op_trait!(GetGlobal, ZeroOperands);
mlir_op_trait!(GetGlobal, ZeroRegions);
mlir_op_trait!(GetGlobal, ZeroSuccessors);

/// Constructs a new detached [`GetGlobalOperation`].
pub fn get_global<
    'c,
    't: 'c,
    N: TryIntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    name: N,
    result_type: T,
    location: L,
) -> Result<DetachedGetGlobalOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.get_global", location)
        .add_attribute(NAME_ATTRIBUTE, name.try_into_with_context(context)?)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::get_global`"))
        })
}

/// Name of the [`Attribute`] that stores the type of a MemRef global.
pub const TYPE_ATTRIBUTE: &str = "type";

/// Name of the [`Attribute`] that stores the initial value of a MemRef global.
pub const INITIAL_VALUE_ATTRIBUTE: &str = "initial_value";

/// Name of the [`Attribute`] that marks a MemRef global as constant.
pub const CONSTANT_ATTRIBUTE: &str = "constant";

/// Operation trait for the `memref.global` operation.
pub trait GlobalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + Symbol<'o, 'c, 't> {
    /// Returns the declared global memref type.
    fn r#type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.type_attribute(TYPE_ATTRIBUTE)?.r#type()
    }

    /// Returns the optional initial value.
    fn initial_value(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute(INITIAL_VALUE_ATTRIBUTE)
    }

    /// Returns `true` if this global is constant.
    fn is_constant(&self) -> bool {
        self.has_attribute(CONSTANT_ATTRIBUTE)
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Result<Option<i64>, Error> {
        if self.has_attribute(ALIGNMENT_ATTRIBUTE) {
            self.integer_attribute(ALIGNMENT_ATTRIBUTE).map(|attribute| Some(attribute.signless_value()))
        } else {
            Ok(None)
        }
    }
}

mlir_op!(Global);
mlir_op_trait!(Global, Symbol);
mlir_op_trait!(Global, ZeroOperands);
mlir_op_trait!(Global, ZeroRegions);
mlir_op_trait!(Global, ZeroSuccessors);

/// Constructs a new detached [`GlobalOperation`].
pub fn global<
    'c,
    't: 'c,
    N: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    name: N,
    visibility: SymbolVisibility,
    r#type: T,
    initial_value: Option<AttributeRef<'c, 't>>,
    is_constant: bool,
    alignment: Option<i64>,
    location: L,
) -> Result<DetachedGlobalOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let mut builder = OperationBuilder::new("memref.global", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, name.try_into_with_context(context)?)
        .add_attribute(TYPE_ATTRIBUTE, context.type_attribute(r#type));
    if visibility != SymbolVisibility::default() {
        builder = builder.add_attribute(SYMBOL_VISIBILITY_ATTRIBUTE, context.symbol_visibility_attribute(visibility));
    }
    if let Some(initial_value) = initial_value {
        builder = builder.add_attribute(INITIAL_VALUE_ATTRIBUTE, initial_value);
    }
    if is_constant {
        builder = builder.add_attribute(CONSTANT_ATTRIBUTE, context.unit_attribute());
    }
    if let Some(alignment) = alignment {
        builder = builder.add_attribute(
            ALIGNMENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), alignment),
        );
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::global`"))
    })
}

/// Name of the [`Attribute`] that marks a load or store as non-temporal.
pub const NONTEMPORAL_ATTRIBUTE: &str = "nontemporal";

/// Operation trait for the `memref.load` operation.
pub trait LoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the memref being loaded from.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the index operands used to access the memref.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (1..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }

    /// Returns `true` if this load is marked as non-temporal.
    fn nontemporal(&self) -> Result<bool, Error> {
        if self.has_attribute(NONTEMPORAL_ATTRIBUTE) {
            self.boolean_attribute(NONTEMPORAL_ATTRIBUTE).map(|attribute| attribute.value())
        } else {
            Ok(false)
        }
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Result<Option<i64>, Error> {
        if self.has_attribute(ALIGNMENT_ATTRIBUTE) {
            self.integer_attribute(ALIGNMENT_ATTRIBUTE).map(|attribute| Some(attribute.signless_value()))
        } else {
            Ok(None)
        }
    }
}

mlir_op!(Load);
mlir_op_trait!(Load, MemRefsNormalizable);
mlir_op_trait!(Load, OneResult);
mlir_op_trait!(Load, ZeroRegions);
mlir_op_trait!(Load, ZeroSuccessors);

/// Constructs a new detached [`LoadOperation`].
pub fn load<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    memref: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    result_type: T,
    nontemporal: bool,
    alignment: Option<i64>,
    location: L,
) -> Result<DetachedLoadOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let mut builder = OperationBuilder::new("memref.load", location)
        .add_operand(memref)
        .add_operands(indices)
        .add_result(result_type);
    if nontemporal {
        builder = builder.add_attribute(NONTEMPORAL_ATTRIBUTE, context.boolean_attribute(true));
    }
    if let Some(alignment) = alignment {
        builder = builder.add_attribute(
            ALIGNMENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), alignment),
        );
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::load`"))
    })
}

/// Operation trait for the `memref.memory_space_cast` operation.
pub trait MemorySpaceCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the destination memref.
    fn dest(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(MemorySpaceCast);
mlir_op_trait!(MemorySpaceCast, AlwaysSpeculatable);
mlir_op_trait!(MemorySpaceCast, MemRefsNormalizable);
mlir_op_trait!(MemorySpaceCast, NoMemoryEffect);
mlir_op_trait!(MemorySpaceCast, OneResult);
mlir_op_trait!(MemorySpaceCast, Pure);
mlir_op_trait!(MemorySpaceCast, ZeroRegions);
mlir_op_trait!(MemorySpaceCast, ZeroSuccessors);

/// Constructs a new detached [`MemorySpaceCastOperation`].
pub fn memory_space_cast<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    dest_type: T,
    location: L,
) -> Result<DetachedMemorySpaceCastOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.memory_space_cast", location)
        .add_operand(source)
        .add_result(dest_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::memory_space_cast`"))
        })
}

/// Name of the [`Attribute`] that stores whether a prefetch is for a write.
pub const IS_WRITE_ATTRIBUTE: &str = "isWrite";

/// Name of the [`Attribute`] that stores the prefetch locality hint.
pub const LOCALITY_HINT_ATTRIBUTE: &str = "localityHint";

/// Name of the [`Attribute`] that stores whether a prefetch targets the data cache.
pub const IS_DATA_CACHE_ATTRIBUTE: &str = "isDataCache";

/// Operation trait for the `memref.prefetch` operation.
pub trait PrefetchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the memref being prefetched.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the index operands used to access the memref.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (1..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }

    /// Returns `true` if the prefetch is for a write.
    fn is_write(&self) -> Result<bool, Error> {
        Ok(self.boolean_attribute(IS_WRITE_ATTRIBUTE)?.value())
    }

    /// Returns the locality hint in the range `0..=3`.
    fn locality_hint(&self) -> Result<i32, Error> {
        i32::try_from(self.integer_attribute(LOCALITY_HINT_ATTRIBUTE)?.signless_value())
            .map_err(|_| Error::invalid_argument("invalid `localityHint` attribute in `memref::prefetch`"))
    }

    /// Returns `true` if the prefetch targets the data cache.
    fn is_data_cache(&self) -> Result<bool, Error> {
        Ok(self.boolean_attribute(IS_DATA_CACHE_ATTRIBUTE)?.value())
    }
}

mlir_op!(Prefetch);
mlir_op_trait!(Prefetch, ZeroRegions);
mlir_op_trait!(Prefetch, ZeroSuccessors);

/// Constructs a new detached [`PrefetchOperation`].
pub fn prefetch<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    memref: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    is_write: bool,
    locality_hint: i32,
    is_data_cache: bool,
    location: L,
) -> Result<DetachedPrefetchOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.prefetch", location)
        .add_operand(memref)
        .add_operands(indices)
        .add_attribute(IS_WRITE_ATTRIBUTE, context.boolean_attribute(is_write))
        .add_attribute(
            LOCALITY_HINT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), locality_hint.into()),
        )
        .add_attribute(IS_DATA_CACHE_ATTRIBUTE, context.boolean_attribute(is_data_cache))
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::prefetch`"))
        })
}

/// Name of the [`Attribute`] that stores static offset entries for view-like operations.
pub const STATIC_OFFSETS_ATTRIBUTE: &str = "static_offsets";

/// Name of the [`Attribute`] that stores static size entries for view-like operations.
pub const STATIC_SIZES_ATTRIBUTE: &str = "static_sizes";

/// Name of the [`Attribute`] that stores static stride entries for view-like operations.
pub const STATIC_STRIDES_ATTRIBUTE: &str = "static_strides";

/// Operation trait for the `memref.reinterpret_cast` operation.
pub trait ReinterpretCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the mixed static or dynamic offset.
    fn offset(&self) -> Result<StaticOrDynamicIndex<'o, 'c, 't>, Error> {
        let dynamic_offsets = self
            .dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .map(|index| self.operand_value(index))
            .collect::<Result<Vec<_>, _>>()?;
        let static_offsets = Vec::<i64>::from(self.dense_integer_64_array_attribute(STATIC_OFFSETS_ATTRIBUTE)?);
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_offsets = dynamic_offsets.into_iter();
        static_offsets
            .into_iter()
            .map(|index| {
                if index == dynamic_index {
                    dynamic_offsets.next().map(StaticOrDynamicIndex::Dynamic).ok_or_else(|| {
                        Error::invalid_argument("missing dynamic offset operand in `memref::reinterpret_cast`")
                    })
                } else {
                    Ok(StaticOrDynamicIndex::Static(index))
                }
            })
            .next()
            .ok_or_else(|| Error::invalid_argument("missing reinterpret-cast offset"))?
    }

    /// Returns the mixed static and dynamic sizes.
    fn sizes(&self) -> Result<Vec<StaticOrDynamicIndex<'o, 'c, 't>>, Error> {
        let dynamic_sizes = self
            .dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .map(|index| self.operand_value(index))
            .collect::<Result<Vec<_>, _>>()?;
        let static_sizes = Vec::<i64>::from(self.dense_integer_64_array_attribute(STATIC_SIZES_ATTRIBUTE)?);
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_sizes = dynamic_sizes.into_iter();
        static_sizes
            .into_iter()
            .map(|index| {
                if index == dynamic_index {
                    dynamic_sizes.next().map(StaticOrDynamicIndex::Dynamic).ok_or_else(|| {
                        Error::invalid_argument("missing dynamic size operand in `memref::reinterpret_cast`")
                    })
                } else {
                    Ok(StaticOrDynamicIndex::Static(index))
                }
            })
            .collect()
    }

    /// Returns the mixed static and dynamic strides.
    fn strides(&self) -> Result<Vec<StaticOrDynamicIndex<'o, 'c, 't>>, Error> {
        let dynamic_strides = self
            .dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 3)?
            .map(|index| self.operand_value(index))
            .collect::<Result<Vec<_>, _>>()?;
        let static_strides = Vec::<i64>::from(self.dense_integer_64_array_attribute(STATIC_STRIDES_ATTRIBUTE)?);
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_strides = dynamic_strides.into_iter();
        static_strides
            .into_iter()
            .map(|index| {
                if index == dynamic_index {
                    dynamic_strides.next().map(StaticOrDynamicIndex::Dynamic).ok_or_else(|| {
                        Error::invalid_argument("missing dynamic stride operand in `memref::reinterpret_cast`")
                    })
                } else {
                    Ok(StaticOrDynamicIndex::Static(index))
                }
            })
            .collect()
    }

    /// Returns the reinterpreted memref.
    fn reinterpreted(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(ReinterpretCast);
mlir_op_trait!(ReinterpretCast, AlwaysSpeculatable);
mlir_op_trait!(ReinterpretCast, MemRefsNormalizable);
mlir_op_trait!(ReinterpretCast, NoMemoryEffect);
mlir_op_trait!(ReinterpretCast, OneResult);
mlir_op_trait!(ReinterpretCast, Pure);
mlir_op_trait!(ReinterpretCast, ZeroRegions);
mlir_op_trait!(ReinterpretCast, ZeroSuccessors);

/// Constructs a new detached [`ReinterpretCastOperation`].
pub fn reinterpret_cast<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    offset: StaticOrDynamicIndex<'v, 'c, 't>,
    sizes: &[StaticOrDynamicIndex<'v, 'c, 't>],
    strides: &[StaticOrDynamicIndex<'v, 'c, 't>],
    result_type: T,
    location: L,
) -> Result<DetachedReinterpretCastOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
    let static_offsets = [offset.static_value().unwrap_or(dynamic_index)];
    let dynamic_offsets = offset.dynamic_value().into_iter().collect::<Vec<_>>();
    let static_sizes = sizes.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_sizes = sizes.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    let static_strides = strides.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_strides = strides.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    OperationBuilder::new("memref.reinterpret_cast", location)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[
                1,
                dynamic_offsets.len() as i32,
                dynamic_sizes.len() as i32,
                dynamic_strides.len() as i32,
            ])?,
        )
        .add_attribute(STATIC_OFFSETS_ATTRIBUTE, context.dense_i64_array_attribute(&static_offsets)?)
        .add_attribute(STATIC_SIZES_ATTRIBUTE, context.dense_i64_array_attribute(&static_sizes)?)
        .add_attribute(STATIC_STRIDES_ATTRIBUTE, context.dense_i64_array_attribute(&static_strides)?)
        .add_operand(source)
        .add_operands(&dynamic_offsets)
        .add_operands(&dynamic_sizes)
        .add_operands(&dynamic_strides)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::reinterpret_cast`"))
        })
}

/// Operation trait for the `memref.rank` operation.
pub trait RankOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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

/// Constructs a new detached [`RankOperation`].
pub fn rank<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    memref: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedRankOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.rank", location)
        .add_operand(memref)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::rank`"))
        })
}

/// Operation trait for the `memref.reshape` operation.
pub trait ReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the memref containing the dynamic shape.
    fn shape(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the reshaped memref.
    fn reshaped(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(Reshape);
mlir_op_trait!(Reshape, AlwaysSpeculatable);
mlir_op_trait!(Reshape, NoMemoryEffect);
mlir_op_trait!(Reshape, OneResult);
mlir_op_trait!(Reshape, Pure);
mlir_op_trait!(Reshape, ZeroRegions);
mlir_op_trait!(Reshape, ZeroSuccessors);

/// Constructs a new detached [`ReshapeOperation`].
pub fn reshape<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    shape: ValueRef<'v, 'c, 't>,
    result_type: T,
    location: L,
) -> Result<DetachedReshapeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.reshape", location)
        .add_operand(source)
        .add_operand(shape)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::reshape`"))
        })
}

/// Name of the [`Attribute`] that stores reassociation groups for MemRef reshape operations.
pub const REASSOCIATION_ATTRIBUTE: &str = "reassociation";

/// Name of the [`Attribute`] that stores static output shape entries for `memref.expand_shape`.
pub const STATIC_OUTPUT_SHAPE_ATTRIBUTE: &str = "static_output_shape";

/// Operation trait shared by `memref.expand_shape` and `memref.collapse_shape`.
pub trait ReassociativeReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the reassociation groups.
    fn reassociation(&self) -> Result<ArrayAttributeRef<'c, 't>, Error> {
        self.array_attribute(REASSOCIATION_ATTRIBUTE)
    }

    /// Returns the resulting memref.
    fn reshaped(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

/// Operation trait for the `memref.expand_shape` operation.
pub trait ExpandShapeOperation<'o, 'c: 'o, 't: 'c>: ReassociativeReshapeOperation<'o, 'c, 't> {
    /// Returns the mixed static and dynamic output shape entries.
    fn output_shape(&self) -> Result<Vec<StaticOrDynamicIndex<'o, 'c, 't>>, Error> {
        let dynamic_shape =
            (1..self.operand_count()).map(|index| self.operand_value(index)).collect::<Result<Vec<_>, _>>()?;
        let static_shape = Vec::<i64>::from(self.dense_integer_64_array_attribute(STATIC_OUTPUT_SHAPE_ATTRIBUTE)?);
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_shape = dynamic_shape.into_iter();
        static_shape
            .into_iter()
            .map(|index| {
                if index == dynamic_index {
                    dynamic_shape.next().map(StaticOrDynamicIndex::Dynamic).ok_or_else(|| {
                        Error::invalid_argument("missing dynamic output shape operand in `memref::expand_shape`")
                    })
                } else {
                    Ok(StaticOrDynamicIndex::Static(index))
                }
            })
            .collect()
    }
}

mlir_op!(ExpandShape);
mlir_op_trait!(ExpandShape, AlwaysSpeculatable);
mlir_op_trait!(ExpandShape, NoMemoryEffect);
mlir_op_trait!(ExpandShape, OneResult);
mlir_op_trait!(ExpandShape, Pure);
mlir_op_trait!(ExpandShape, ZeroRegions);
mlir_op_trait!(ExpandShape, ZeroSuccessors);
mlir_op_trait!(ExpandShape, @local ReassociativeReshapeOperation);

/// Constructs a new detached [`ExpandShapeOperation`].
pub fn expand_shape<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    reassociation: &[&[i64]],
    output_shape: &[StaticOrDynamicIndex<'v, 'c, 't>],
    result_type: T,
    location: L,
) -> Result<DetachedExpandShapeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
    let index_attribute_type = context.signless_integer_type(64);
    let reassociation = reassociation
        .iter()
        .map(|group| {
            let group = group
                .iter()
                .map(|index| context.integer_attribute(index_attribute_type, *index))
                .collect::<Vec<_>>();
            context.array_attribute(&group).as_ref()
        })
        .collect::<Vec<_>>();
    let static_output_shape =
        output_shape.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_output_shape = output_shape.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    OperationBuilder::new("memref.expand_shape", location)
        .add_operand(source)
        .add_operands(&dynamic_output_shape)
        .add_attribute(REASSOCIATION_ATTRIBUTE, context.array_attribute(&reassociation))
        .add_attribute(STATIC_OUTPUT_SHAPE_ATTRIBUTE, context.dense_i64_array_attribute(&static_output_shape)?)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::expand_shape`"))
        })
}

/// Operation trait for the `memref.collapse_shape` operation.
pub trait CollapseShapeOperation<'o, 'c: 'o, 't: 'c>: ReassociativeReshapeOperation<'o, 'c, 't> {}

mlir_op!(CollapseShape);
mlir_op_trait!(CollapseShape, AlwaysSpeculatable);
mlir_op_trait!(CollapseShape, NoMemoryEffect);
mlir_op_trait!(CollapseShape, OneResult);
mlir_op_trait!(CollapseShape, Pure);
mlir_op_trait!(CollapseShape, ZeroRegions);
mlir_op_trait!(CollapseShape, ZeroSuccessors);
mlir_op_trait!(CollapseShape, @local ReassociativeReshapeOperation);

/// Constructs a new detached [`CollapseShapeOperation`].
pub fn collapse_shape<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    reassociation: &[&[i64]],
    result_type: T,
    location: L,
) -> Result<DetachedCollapseShapeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let index_attribute_type = context.signless_integer_type(64);
    let reassociation = reassociation
        .iter()
        .map(|group| {
            let group = group
                .iter()
                .map(|index| context.integer_attribute(index_attribute_type, *index))
                .collect::<Vec<_>>();
            context.array_attribute(&group).as_ref()
        })
        .collect::<Vec<_>>();
    OperationBuilder::new("memref.collapse_shape", location)
        .add_operand(source)
        .add_attribute(REASSOCIATION_ATTRIBUTE, context.array_attribute(&reassociation))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::collapse_shape`"))
        })
}

/// Operation trait for the `memref.store` operation.
pub trait StoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the stored value.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the memref being stored into.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the index operands used to access the memref.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (2..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }

    /// Returns `true` if this store is marked as non-temporal.
    fn nontemporal(&self) -> Result<bool, Error> {
        if self.has_attribute(NONTEMPORAL_ATTRIBUTE) {
            self.boolean_attribute(NONTEMPORAL_ATTRIBUTE).map(|attribute| attribute.value())
        } else {
            Ok(false)
        }
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Result<Option<i64>, Error> {
        if self.has_attribute(ALIGNMENT_ATTRIBUTE) {
            self.integer_attribute(ALIGNMENT_ATTRIBUTE).map(|attribute| Some(attribute.signless_value()))
        } else {
            Ok(None)
        }
    }
}

mlir_op!(Store);
mlir_op_trait!(Store, MemRefsNormalizable);
mlir_op_trait!(Store, ZeroRegions);
mlir_op_trait!(Store, ZeroSuccessors);

/// Constructs a new detached [`StoreOperation`].
pub fn store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'v, 'c, 't>,
    memref: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    nontemporal: bool,
    alignment: Option<i64>,
    location: L,
) -> Result<DetachedStoreOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let mut builder = OperationBuilder::new("memref.store", location)
        .add_operand(value)
        .add_operand(memref)
        .add_operands(indices);
    if nontemporal {
        builder = builder.add_attribute(NONTEMPORAL_ATTRIBUTE, context.boolean_attribute(true));
    }
    if let Some(alignment) = alignment {
        builder = builder.add_attribute(
            ALIGNMENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), alignment),
        );
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::store`"))
    })
}

/// Operation trait for the `memref.subview` operation.
pub trait SubViewOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the mixed static and dynamic offsets.
    fn offsets(&self) -> Result<Vec<StaticOrDynamicIndex<'o, 'c, 't>>, Error> {
        let dynamic_offsets = self
            .dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .map(|index| self.operand_value(index))
            .collect::<Result<Vec<_>, _>>()?;
        let static_offsets = Vec::<i64>::from(self.dense_integer_64_array_attribute(STATIC_OFFSETS_ATTRIBUTE)?);
        let dynamic_size = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_offsets = dynamic_offsets.into_iter();
        static_offsets
            .into_iter()
            .map(|index| {
                if index == dynamic_size {
                    dynamic_offsets
                        .next()
                        .map(StaticOrDynamicIndex::Dynamic)
                        .ok_or_else(|| Error::invalid_argument("missing dynamic offset operand in `memref::subview`"))
                } else {
                    Ok(StaticOrDynamicIndex::Static(index))
                }
            })
            .collect()
    }

    /// Returns the mixed static and dynamic sizes.
    fn sizes(&self) -> Result<Vec<StaticOrDynamicIndex<'o, 'c, 't>>, Error> {
        let dynamic_sizes = self
            .dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .map(|index| self.operand_value(index))
            .collect::<Result<Vec<_>, _>>()?;
        let static_sizes = Vec::<i64>::from(self.dense_integer_64_array_attribute(STATIC_SIZES_ATTRIBUTE)?);
        let dynamic_size = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_sizes = dynamic_sizes.into_iter();
        static_sizes
            .into_iter()
            .map(|index| {
                if index == dynamic_size {
                    dynamic_sizes
                        .next()
                        .map(StaticOrDynamicIndex::Dynamic)
                        .ok_or_else(|| Error::invalid_argument("missing dynamic size operand in `memref::subview`"))
                } else {
                    Ok(StaticOrDynamicIndex::Static(index))
                }
            })
            .collect()
    }

    /// Returns the mixed static and dynamic strides.
    fn strides(&self) -> Result<Vec<StaticOrDynamicIndex<'o, 'c, 't>>, Error> {
        let dynamic_strides = self
            .dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 3)?
            .map(|index| self.operand_value(index))
            .collect::<Result<Vec<_>, _>>()?;
        let static_strides = Vec::<i64>::from(self.dense_integer_64_array_attribute(STATIC_STRIDES_ATTRIBUTE)?);
        let dynamic_size = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_strides = dynamic_strides.into_iter();
        static_strides
            .into_iter()
            .map(|index| {
                if index == dynamic_size {
                    dynamic_strides
                        .next()
                        .map(StaticOrDynamicIndex::Dynamic)
                        .ok_or_else(|| Error::invalid_argument("missing dynamic stride operand in `memref::subview`"))
                } else {
                    Ok(StaticOrDynamicIndex::Static(index))
                }
            })
            .collect()
    }

    /// Returns the resulting subview memref.
    fn subview(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(SubView);
mlir_op_trait!(SubView, AlwaysSpeculatable);
mlir_op_trait!(SubView, NoMemoryEffect);
mlir_op_trait!(SubView, OneResult);
mlir_op_trait!(SubView, Pure);
mlir_op_trait!(SubView, ZeroRegions);
mlir_op_trait!(SubView, ZeroSuccessors);

/// Constructs a new detached [`SubViewOperation`].
pub fn subview<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    offsets: &[StaticOrDynamicIndex<'v, 'c, 't>],
    sizes: &[StaticOrDynamicIndex<'v, 'c, 't>],
    strides: &[StaticOrDynamicIndex<'v, 'c, 't>],
    result_type: T,
    location: L,
) -> Result<DetachedSubViewOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
    let static_offsets = offsets.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_offsets = offsets.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    let static_sizes = sizes.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_sizes = sizes.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    let static_strides = strides.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_strides = strides.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    OperationBuilder::new("memref.subview", location)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[
                1,
                dynamic_offsets.len() as i32,
                dynamic_sizes.len() as i32,
                dynamic_strides.len() as i32,
            ])?,
        )
        .add_attribute(STATIC_OFFSETS_ATTRIBUTE, context.dense_i64_array_attribute(&static_offsets)?)
        .add_attribute(STATIC_SIZES_ATTRIBUTE, context.dense_i64_array_attribute(&static_sizes)?)
        .add_attribute(STATIC_STRIDES_ATTRIBUTE, context.dense_i64_array_attribute(&static_strides)?)
        .add_operand(source)
        .add_operands(&dynamic_offsets)
        .add_operands(&dynamic_sizes)
        .add_operands(&dynamic_strides)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::subview`"))
        })
}

/// Name of the [`Attribute`] that stores the affine-map permutation for `memref.transpose`.
pub const PERMUTATION_ATTRIBUTE: &str = "permutation";

/// Operation trait for the `memref.transpose` operation.
pub trait TransposeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the input memref.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the affine-map permutation.
    fn permutation(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(PERMUTATION_ATTRIBUTE)?.affine_map()
    }

    /// Returns the transposed memref.
    fn transposed(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(Transpose);
mlir_op_trait!(Transpose, AlwaysSpeculatable);
mlir_op_trait!(Transpose, NoMemoryEffect);
mlir_op_trait!(Transpose, OneResult);
mlir_op_trait!(Transpose, Pure);
mlir_op_trait!(Transpose, ZeroRegions);
mlir_op_trait!(Transpose, ZeroSuccessors);

/// Constructs a new detached [`TransposeOperation`].
pub fn transpose<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    permutation: AffineMap<'c, 't>,
    result_type: T,
    location: L,
) -> Result<DetachedTransposeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.transpose", location)
        .add_operand(input)
        .add_attribute(PERMUTATION_ATTRIBUTE, context.affine_map_attribute(permutation))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::transpose`"))
        })
}

/// Operation trait for the `memref.view` operation.
pub trait ViewOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source byte buffer.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the dynamic byte-shift operand.
    fn byte_shift(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the dynamic size operands.
    fn sizes(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (2..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the viewed memref.
    fn viewed(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(View);
mlir_op_trait!(View, AlwaysSpeculatable);
mlir_op_trait!(View, NoMemoryEffect);
mlir_op_trait!(View, OneResult);
mlir_op_trait!(View, Pure);
mlir_op_trait!(View, ZeroRegions);
mlir_op_trait!(View, ZeroSuccessors);

/// Constructs a new detached [`ViewOperation`].
pub fn view<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    byte_shift: ValueRef<'v, 'c, 't>,
    sizes: &[ValueRef<'v, 'c, 't>],
    result_type: T,
    location: L,
) -> Result<DetachedViewOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.view", location)
        .add_operand(source)
        .add_operand(byte_shift)
        .add_operands(sizes)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::view`"))
        })
}

/// Name of the [`Attribute`] that stores an atomic read-modify-write kind.
pub const KIND_ATTRIBUTE: &str = "kind";

/// Operation trait for the `memref.atomic_rmw` operation.
pub trait AtomicRmwOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the atomic read-modify-write kind.
    fn kind(&self) -> Result<AtomicRmwKind, Error> {
        self.attribute(KIND_ATTRIBUTE)?
            .and_then(|attribute| attribute.cast::<AtomicRmwKindAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    KIND_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns the value applied by the read-modify-write operation.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the memref being read and updated atomically.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the index operands used to access the memref.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (2..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the latest stored value.
    fn latest_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.output()
    }
}

mlir_op!(AtomicRmw);
mlir_op_trait!(AtomicRmw, OneResult);
mlir_op_trait!(AtomicRmw, ZeroRegions);
mlir_op_trait!(AtomicRmw, ZeroSuccessors);

/// Constructs a new detached [`AtomicRmwOperation`].
pub fn atomic_rmw<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    kind: AtomicRmwKind,
    value: ValueRef<'v, 'c, 't>,
    memref: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedAtomicRmwOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref()?)?;
    OperationBuilder::new("memref.atomic_rmw", location)
        .add_attribute(KIND_ATTRIBUTE, context.arith_atomic_rmw_kind_attribute(kind)?)
        .add_operand(value)
        .add_operand(memref)
        .add_operands(indices)
        .add_result(value.r#type()?)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `memref::atomic_rmw`"))
        })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, Operation, Symbol, Type, Value};

    use super::*;

    #[test]
    fn test_dma_start_and_dma_wait() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let i32_type = context.signless_integer_type(32);
        let index_type = context.index_type();
        let destination_memory_space = context.integer_attribute(index_type, 1);
        let tag_memory_space = context.integer_attribute(index_type, 2);
        let source_type = context.mem_ref_type(f32_type, &[Size::Static(16)], None, None, location).unwrap();
        let destination_type = context
            .mem_ref_type(f32_type, &[Size::Static(16)], None, Some(destination_memory_space.as_ref()), location)
            .unwrap();
        let tag_type = context
            .mem_ref_type(i32_type, &[Size::Static(1)], None, Some(tag_memory_space.as_ref()), location)
            .unwrap();

        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (source_type.as_ref(), location),
                    (destination_type.as_ref(), location),
                    (tag_type.as_ref(), location),
                    (index_type.as_ref(), location),
                    (index_type.as_ref(), location),
                ]);
                let source = block.argument(0).unwrap().as_ref();
                let destination = block.argument(1).unwrap().as_ref();
                let tag = block.argument(2).unwrap().as_ref();
                let num_elements = block.argument(3).unwrap().as_ref();
                let index = block.argument(4).unwrap().as_ref();

                let dma_start_op = dma_start(
                    source,
                    &[index],
                    destination,
                    &[index],
                    num_elements,
                    tag,
                    &[index],
                    None,
                    None,
                    location,
                )
                .unwrap();
                assert_eq!(dma_start_op.source().unwrap(), source);
                assert_eq!(dma_start_op.source_indices().unwrap(), vec![index]);
                assert_eq!(dma_start_op.destination().unwrap(), destination);
                assert_eq!(dma_start_op.destination_indices().unwrap(), vec![index]);
                assert_eq!(dma_start_op.num_elements().unwrap(), num_elements);
                assert_eq!(dma_start_op.tag().unwrap(), tag);
                assert_eq!(dma_start_op.tag_indices().unwrap(), vec![index]);
                assert!(!dma_start_op.is_strided().unwrap());
                assert_eq!(dma_start_op.stride().unwrap(), None);
                assert_eq!(dma_start_op.elements_per_stride().unwrap(), None);
                block.append_operation(dma_start_op).unwrap();

                let dma_wait_op = dma_wait(tag, &[index], num_elements, location).unwrap();
                assert_eq!(dma_wait_op.tag().unwrap(), tag);
                assert_eq!(dma_wait_op.tag_indices().unwrap(), vec![index]);
                assert_eq!(dma_wait_op.num_elements().unwrap(), num_elements);
                block.append_operation(dma_wait_op).unwrap();

                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "memref_dma",
                    func::FuncAttributes {
                        arguments: vec![
                            source_type.into(),
                            destination_type.into(),
                            tag_type.into(),
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
                module {
                  func.func @memref_dma(%arg0: memref<16xf32>, %arg1: memref<16xf32, 1 : index>, %arg2: memref<1xi32, 2 : index>, %arg3: index, %arg4: index) {
                    memref.dma_start %arg0[%arg4], %arg1[%arg4], %arg3, %arg2[%arg4] : memref<16xf32>, memref<16xf32, 1 : index>, memref<1xi32, 2 : index>
                    memref.dma_wait %arg2[%arg4], %arg3 : memref<1xi32, 2 : index>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_alloc_load_store_dealloc_and_prefetch() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f32_type = context.float32_type();
        let memref_type = context.mem_ref_type(f32_type, &[Size::Dynamic], None, None, location).unwrap();

        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type.as_ref(), location), (f32_type.as_ref(), location)]);
                let dynamic_size = block.argument(0).unwrap().as_ref();
                let value = block.argument(1).unwrap().as_ref();

                let alloc_op = alloc(&[dynamic_size], &[], memref_type, Some(64), location).unwrap();
                assert_eq!(alloc_op.dynamic_sizes().unwrap(), vec![dynamic_size]);
                assert_eq!(alloc_op.symbol_operands().unwrap(), Vec::<ValueRef>::new());
                assert_eq!(alloc_op.alignment().unwrap(), Some(64));
                assert_eq!(alloc_op.memref().unwrap().r#type().unwrap(), memref_type);
                let alloc_op = block.append_operation(alloc_op).unwrap();
                let memref = alloc_op.result(0).unwrap().as_ref();

                let store_op = store(value, memref, &[dynamic_size], true, Some(4), location).unwrap();
                assert_eq!(store_op.value().unwrap(), value);
                assert_eq!(store_op.memref().unwrap(), memref);
                assert_eq!(store_op.indices().unwrap(), vec![dynamic_size]);
                assert_eq!(store_op.nontemporal().unwrap(), true);
                assert_eq!(store_op.alignment().unwrap(), Some(4));
                block.append_operation(store_op).unwrap();

                let prefetch_op = prefetch(memref, &[dynamic_size], false, 3, true, location).unwrap();
                assert_eq!(prefetch_op.memref().unwrap(), memref);
                assert_eq!(prefetch_op.indices().unwrap(), vec![dynamic_size]);
                assert_eq!(prefetch_op.is_write().unwrap(), false);
                assert_eq!(prefetch_op.locality_hint().unwrap(), 3);
                assert_eq!(prefetch_op.is_data_cache().unwrap(), true);
                block.append_operation(prefetch_op).unwrap();

                let load_op = load(memref, &[dynamic_size], f32_type, false, None, location).unwrap();
                assert_eq!(load_op.memref().unwrap(), memref);
                assert_eq!(load_op.indices().unwrap(), vec![dynamic_size]);
                assert_eq!(load_op.output_type().unwrap(), f32_type);
                assert_eq!(load_op.nontemporal().unwrap(), false);
                assert_eq!(load_op.alignment().unwrap(), None);
                let load_op = block.append_operation(load_op).unwrap();

                let dealloc_op = dealloc(memref, location).unwrap();
                assert_eq!(dealloc_op.memref().unwrap(), memref);
                block.append_operation(dealloc_op).unwrap();

                block
                    .append_operation(func::r#return(&[load_op.result(0).unwrap().as_ref()], location).unwrap())
                    .unwrap();
                func::func(
                    "memref_access",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @memref_access(%arg0: index, %arg1: f32) -> f32 {
                    %alloc = memref.alloc(%arg0) {alignment = 64 : i64} : memref<?xf32>
                    memref.store %arg1, %alloc[%arg0] {alignment = 4 : i64, nontemporal = true} : memref<?xf32>
                    memref.prefetch %alloc[%arg0], read, locality<3>, data : memref<?xf32>
                    %0 = memref.load %alloc[%arg0] : memref<?xf32>
                    memref.dealloc %alloc : memref<?xf32>
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_cast_dim_rank_and_pointer_operations() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f32_type = context.float32_type();
        let source_type =
            context.mem_ref_type(f32_type, &[Size::Static(4), Size::Dynamic], None, None, location).unwrap();
        let cast_type = context.mem_ref_type(f32_type, &[Size::Dynamic, Size::Dynamic], None, None, location).unwrap();

        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(source_type.as_ref(), location), (index_type.as_ref(), location)]);
                let source = block.argument(0).unwrap().as_ref();
                let index = block.argument(1).unwrap().as_ref();

                let assume_alignment_op = assume_alignment(source, 16, location).unwrap();
                assert_eq!(assume_alignment_op.memref().unwrap(), source);
                assert_eq!(assume_alignment_op.alignment().unwrap(), 16);
                assert_eq!(assume_alignment_op.output_type().unwrap(), source_type);
                let assume_alignment_op = block.append_operation(assume_alignment_op).unwrap();
                let aligned = assume_alignment_op.result(0).unwrap().as_ref();

                let cast_op = cast(aligned, cast_type, location).unwrap();
                assert_eq!(cast_op.source().unwrap(), aligned);
                assert_eq!(cast_op.source().unwrap(), aligned);
                let cast_op = block.append_operation(cast_op).unwrap();
                let cast_memref = cast_op.result(0).unwrap().as_ref();

                let dim_op = dim(source, index, location).unwrap();
                assert_eq!(dim_op.source().unwrap(), source);
                assert_eq!(dim_op.index().unwrap(), index);
                assert_eq!(dim_op.output_type().unwrap(), index_type);
                let dim_op = block.append_operation(dim_op).unwrap();

                let rank_op = rank(cast_memref, location).unwrap();
                assert_eq!(rank_op.memref().unwrap(), cast_memref);
                assert_eq!(rank_op.output_type().unwrap(), index_type);
                let rank_op = block.append_operation(rank_op).unwrap();

                let pointer_op = extract_aligned_pointer_as_index(cast_memref, location).unwrap();
                assert_eq!(pointer_op.source().unwrap(), cast_memref);
                assert_eq!(pointer_op.source().unwrap(), cast_memref);
                let pointer_op = block.append_operation(pointer_op).unwrap();

                block
                    .append_operation(
                        func::r#return(
                            &[
                                cast_memref,
                                dim_op.result(0).unwrap().as_ref(),
                                rank_op.result(0).unwrap().as_ref(),
                                pointer_op.result(0).unwrap().as_ref(),
                            ],
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                func::func(
                    "memref_shape",
                    func::FuncAttributes {
                        arguments: vec![source_type.into(), index_type.into()],
                        results: vec![cast_type.into(), index_type.into(), index_type.into(), index_type.into()],
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
                  func.func @memref_shape(%arg0: memref<4x?xf32>, %arg1: index) -> (memref<?x?xf32>, index, index, index) {
                    %assume_align = memref.assume_alignment %arg0, 16 : memref<4x?xf32>
                    %cast = memref.cast %assume_align : memref<4x?xf32> to memref<?x?xf32>
                    %dim = memref.dim %arg0, %arg1 : memref<4x?xf32>
                    %0 = memref.rank %cast : memref<?x?xf32>
                    %intptr = memref.extract_aligned_pointer_as_index %cast : memref<?x?xf32> -> index
                    return %cast, %dim, %0, %intptr : memref<?x?xf32>, index, index, index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_metadata_reinterpret_expand_collapse_transpose_and_view() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f32_type = context.float32_type();
        let i8_type = context.signless_integer_type(8);
        let source_type =
            context.mem_ref_type(f32_type, &[Size::Dynamic, Size::Dynamic], None, None, location).unwrap();
        let base_buffer_type = context.mem_ref_type(f32_type, &[], None, None, location).unwrap();
        let reinterpret_type = context
            .parse_type("memref<?x?xf32, strided<[?, ?], offset: ?>>")
            .unwrap()
            .cast::<MemRefTypeRef>()
            .unwrap();
        let expand_source_type =
            context.mem_ref_type(f32_type, &[Size::Dynamic, Size::Static(32)], None, None, location).unwrap();
        let expanded_type = context
            .mem_ref_type(f32_type, &[Size::Dynamic, Size::Dynamic, Size::Static(32)], None, None, location)
            .unwrap();
        let transposed_type =
            context.parse_type("memref<?x?xf32, strided<[1, ?]>>").unwrap().cast::<MemRefTypeRef>().unwrap();
        let byte_buffer_type = context.mem_ref_type(i8_type, &[Size::Static(2048)], None, None, location).unwrap();
        let view_type =
            context.mem_ref_type(f32_type, &[Size::Dynamic, Size::Static(4)], None, None, location).unwrap();

        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (source_type.as_ref(), location),
                    (expand_source_type.as_ref(), location),
                    (byte_buffer_type.as_ref(), location),
                    (index_type.as_ref(), location),
                ]);
                let source = block.argument(0).unwrap().as_ref();
                let reshape_source = block.argument(1).unwrap().as_ref();
                let byte_buffer = block.argument(2).unwrap().as_ref();
                let dynamic_size = block.argument(3).unwrap().as_ref();

                let metadata_op = extract_strided_metadata(source, base_buffer_type, location).unwrap();
                assert_eq!(metadata_op.source().unwrap(), source);
                assert_eq!(metadata_op.source().unwrap(), source);
                assert_eq!(metadata_op.base_buffer().unwrap().r#type().unwrap(), base_buffer_type);
                assert_eq!(metadata_op.sizes().unwrap().len(), 2);
                assert_eq!(metadata_op.strides().unwrap().len(), 2);
                let metadata_op = block.append_operation(metadata_op).unwrap();
                let base_buffer = metadata_op.result(0).unwrap().as_ref();
                let offset = metadata_op.result(1).unwrap().as_ref();
                let size_0 = metadata_op.result(2).unwrap().as_ref();
                let size_1 = metadata_op.result(3).unwrap().as_ref();
                let stride_0 = metadata_op.result(4).unwrap().as_ref();
                let stride_1 = metadata_op.result(5).unwrap().as_ref();

                let reinterpret_op = reinterpret_cast(
                    base_buffer,
                    StaticOrDynamicIndex::Dynamic(offset),
                    &[StaticOrDynamicIndex::Dynamic(size_0), StaticOrDynamicIndex::Dynamic(size_1)],
                    &[StaticOrDynamicIndex::Dynamic(stride_0), StaticOrDynamicIndex::Dynamic(stride_1)],
                    reinterpret_type,
                    location,
                )
                .unwrap();
                assert_eq!(reinterpret_op.source().unwrap(), base_buffer);
                assert_eq!(reinterpret_op.offset().unwrap(), StaticOrDynamicIndex::Dynamic(offset));
                assert_eq!(
                    reinterpret_op.sizes().unwrap(),
                    vec![StaticOrDynamicIndex::Dynamic(size_0), StaticOrDynamicIndex::Dynamic(size_1)],
                );
                assert_eq!(
                    reinterpret_op.strides().unwrap(),
                    vec![StaticOrDynamicIndex::Dynamic(stride_0), StaticOrDynamicIndex::Dynamic(stride_1)],
                );
                assert_eq!(reinterpret_op.reinterpreted().unwrap().r#type().unwrap(), reinterpret_type);
                let reinterpret_op = block.append_operation(reinterpret_op).unwrap();

                let expand_op = expand_shape(
                    reshape_source,
                    &[&[0, 1], &[2]],
                    &[
                        StaticOrDynamicIndex::Dynamic(dynamic_size),
                        StaticOrDynamicIndex::Dynamic(dynamic_size),
                        StaticOrDynamicIndex::Static(32),
                    ],
                    expanded_type,
                    location,
                )
                .unwrap();
                assert_eq!(expand_op.source().unwrap(), reshape_source);
                assert_eq!(expand_op.reassociation().unwrap().len(), 2);
                assert_eq!(
                    expand_op.output_shape().unwrap(),
                    vec![
                        StaticOrDynamicIndex::Dynamic(dynamic_size),
                        StaticOrDynamicIndex::Dynamic(dynamic_size),
                        StaticOrDynamicIndex::Static(32),
                    ],
                );
                assert_eq!(expand_op.reshaped().unwrap().r#type().unwrap(), expanded_type);
                let expand_op = block.append_operation(expand_op).unwrap();

                let collapse_op = collapse_shape(
                    expand_op.result(0).unwrap().as_ref(),
                    &[&[0, 1], &[2]],
                    expand_source_type,
                    location,
                )
                .unwrap();
                assert_eq!(collapse_op.source().unwrap(), expand_op.result(0).unwrap().as_ref());
                assert_eq!(collapse_op.reassociation().unwrap().len(), 2);
                assert_eq!(collapse_op.source().unwrap(), expand_op.result(0).unwrap().as_ref());
                let collapse_op = block.append_operation(collapse_op).unwrap();

                let permutation = context.permutation_affine_map(2, &[1, 0]);
                let transpose_op = transpose(source, permutation, transposed_type, location).unwrap();
                assert_eq!(transpose_op.input().unwrap(), source);
                assert_eq!(transpose_op.permutation().unwrap(), permutation);
                assert_eq!(transpose_op.input().unwrap(), source);
                let transpose_op = block.append_operation(transpose_op).unwrap();

                let view_op = view(byte_buffer, dynamic_size, &[dynamic_size], view_type, location).unwrap();
                assert_eq!(view_op.source().unwrap(), byte_buffer);
                assert_eq!(view_op.byte_shift().unwrap(), dynamic_size);
                assert_eq!(view_op.sizes().unwrap(), vec![dynamic_size]);
                assert_eq!(view_op.source().unwrap(), byte_buffer);
                let view_op = block.append_operation(view_op).unwrap();

                block
                    .append_operation(
                        func::r#return(
                            &[
                                reinterpret_op.result(0).unwrap().as_ref(),
                                collapse_op.result(0).unwrap().as_ref(),
                                transpose_op.result(0).unwrap().as_ref(),
                                view_op.result(0).unwrap().as_ref(),
                            ],
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                func::func(
                    "memref_views",
                    func::FuncAttributes {
                        arguments: vec![
                            source_type.into(),
                            expand_source_type.into(),
                            byte_buffer_type.into(),
                            index_type.into(),
                        ],
                        results: vec![
                            reinterpret_type.into(),
                            expand_source_type.into(),
                            transposed_type.into(),
                            view_type.into(),
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
                module {
                  func.func @memref_views(%arg0: memref<?x?xf32>, %arg1: memref<?x32xf32>, %arg2: memref<2048xi8>, %arg3: index) -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x32xf32>, memref<?x?xf32, strided<[1, ?]>>, memref<?x4xf32>) {
                    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
                    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%offset], sizes: [%sizes#0, %sizes#1], strides: [%strides#0, %strides#1] : memref<f32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    %expand_shape = memref.expand_shape %arg1 [[0, 1], [2]] output_shape [%arg3, %arg3, 32] : memref<?x32xf32> into memref<?x?x32xf32>
                    %collapse_shape = memref.collapse_shape %expand_shape [[0, 1], [2]] : memref<?x?x32xf32> into memref<?x32xf32>
                    %transpose = memref.transpose %arg0 (d0, d1) -> (d1, d0) : memref<?x?xf32> to memref<?x?xf32, strided<[1, ?]>>
                    %view = memref.view %arg2[%arg3][%arg3] : memref<2048xi8> to memref<?x4xf32>
                    return %reinterpret_cast, %collapse_shape, %transpose, %view : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x32xf32>, memref<?x?xf32, strided<[1, ?]>>, memref<?x4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_global_and_get_global() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let memref_type =
            context.mem_ref_type(context.float32_type(), &[Size::Static(4)], None, None, location).unwrap();

        let global_op = global(
            "weights",
            SymbolVisibility::Private,
            memref_type,
            Some(context.unit_attribute().as_ref()),
            false,
            Some(64),
            location,
        )
        .unwrap();
        assert_eq!(global_op.symbol_name().unwrap().unwrap().as_str().unwrap(), "weights");
        assert_eq!(global_op.symbol_visibility().unwrap(), SymbolVisibility::Private);
        assert_eq!(global_op.r#type().unwrap(), memref_type);
        assert!(global_op.initial_value().unwrap().is_some());
        assert_eq!(global_op.is_constant(), false);
        assert_eq!(global_op.alignment().unwrap(), Some(64));
        module.body().unwrap().append_operation(global_op).unwrap();

        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let get_global_op = get_global("weights", memref_type, location).unwrap();
                assert_eq!(
                    GetGlobalOperation::name(&get_global_op).unwrap(),
                    context.flat_symbol_ref_attribute("weights")
                );
                assert_eq!(
                    GetGlobalOperation::name(&get_global_op).unwrap(),
                    context.flat_symbol_ref_attribute("weights")
                );
                let get_global_op = block.append_operation(get_global_op).unwrap();
                block
                    .append_operation(func::r#return(&[get_global_op.result(0).unwrap().as_ref()], location).unwrap())
                    .unwrap();
                func::func(
                    "get_weights",
                    func::FuncAttributes { results: vec![memref_type.into()], ..Default::default() },
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
                  memref.global \"private\" @weights : memref<4xf32> = uninitialized {alignment = 64 : i64}
                  func.func @get_weights() -> memref<4xf32> {
                    %0 = memref.get_global @weights : memref<4xf32>
                    return %0 : memref<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_distinct_copy_realloc_and_memory_space_cast() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f32_type = context.float32_type();
        let memref_type = context.mem_ref_type(f32_type, &[Size::Dynamic], None, None, location).unwrap();
        let memory_space = context.integer_attribute(context.signless_integer_type(64), 3);
        let memory_space_type = context
            .mem_ref_type(f32_type, &[Size::Dynamic], None, Some(memory_space.as_ref()), location)
            .unwrap();

        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (index_type.as_ref(), location),
                    (memref_type.as_ref(), location),
                    (memref_type.as_ref(), location),
                ]);
                let dynamic_size = block.argument(0).unwrap().as_ref();
                let source = block.argument(1).unwrap().as_ref();
                let target = block.argument(2).unwrap().as_ref();

                let distinct_op = distinct_objects(&[source, target], location).unwrap();
                assert_eq!(distinct_op.inputs().unwrap(), vec![source, target]);
                assert_eq!(
                    distinct_op.outputs().unwrap().iter().map(|value| value.r#type().unwrap()).collect::<Vec<_>>(),
                    vec![memref_type.as_ref(), memref_type.as_ref()]
                );
                let distinct_op = block.append_operation(distinct_op).unwrap();
                let distinct_source = distinct_op.result(0).unwrap().as_ref();
                let distinct_target = distinct_op.result(1).unwrap().as_ref();

                let copy_op = copy(distinct_source, distinct_target, location).unwrap();
                assert_eq!(copy_op.source().unwrap(), distinct_source);
                assert_eq!(copy_op.target().unwrap(), distinct_target);
                block.append_operation(copy_op).unwrap();

                let realloc_op =
                    realloc(distinct_source, Some(dynamic_size), memref_type, Some(128), location).unwrap();
                assert_eq!(realloc_op.source().unwrap(), distinct_source);
                assert_eq!(realloc_op.dynamic_result_size().unwrap(), Some(dynamic_size));
                assert_eq!(realloc_op.alignment().unwrap(), Some(128));
                assert_eq!(realloc_op.memref().unwrap().r#type().unwrap(), memref_type);
                let realloc_op = block.append_operation(realloc_op).unwrap();

                let memory_space_cast_op = memory_space_cast(distinct_target, memory_space_type, location).unwrap();
                assert_eq!(memory_space_cast_op.source().unwrap(), distinct_target);
                assert_eq!(memory_space_cast_op.source().unwrap(), distinct_target);
                let memory_space_cast_op = block.append_operation(memory_space_cast_op).unwrap();

                block
                    .append_operation(
                        func::r#return(
                            &[
                                distinct_target,
                                realloc_op.result(0).unwrap().as_ref(),
                                memory_space_cast_op.result(0).unwrap().as_ref(),
                            ],
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                func::func(
                    "memref_misc",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), memref_type.into(), memref_type.into()],
                        results: vec![memref_type.into(), memref_type.into(), memory_space_type.into()],
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
                  func.func @memref_misc(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<?xf32>) -> (memref<?xf32>, memref<?xf32>, memref<?xf32, 3>) {
                    %0:2 = memref.distinct_objects %arg1, %arg2 : memref<?xf32>, memref<?xf32>
                    memref.copy %0#0, %0#1 : memref<?xf32> to memref<?xf32>
                    %1 = memref.realloc %0#0(%arg0) {alignment = 128 : i64} : memref<?xf32> to memref<?xf32>
                    %memspacecast = memref.memory_space_cast %0#1 : memref<?xf32> to memref<?xf32, 3>
                    return %0#1, %1, %memspacecast : memref<?xf32>, memref<?xf32>, memref<?xf32, 3>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_alloca_scope_and_reshape() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let i32_type = context.signless_integer_type(32);
        let source_type =
            context.mem_ref_type(f32_type, &[Size::Static(4), Size::Static(1)], None, None, location).unwrap();
        let shape_type = context.mem_ref_type(i32_type, &[Size::Static(1)], None, None, location).unwrap();
        let alloca_type = context.mem_ref_type(f32_type, &[Size::Static(4)], None, None, location).unwrap();
        let result_type = context.mem_ref_type(f32_type, &[Size::Static(4)], None, None, location).unwrap();

        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(source_type.as_ref(), location), (shape_type.as_ref(), location)]);
                let source = block.argument(0).unwrap().as_ref();
                let shape = block.argument(1).unwrap().as_ref();

                let mut scope_block = context.block_with_no_arguments();
                let alloca_op = alloca(&[], &[], alloca_type, Some(16), location).unwrap();
                assert_eq!(alloca_op.dynamic_sizes().unwrap(), Vec::<ValueRef>::new());
                assert_eq!(alloca_op.symbol_operands().unwrap(), Vec::<ValueRef>::new());
                assert_eq!(alloca_op.alignment().unwrap(), Some(16));
                assert_eq!(alloca_op.memref().unwrap().r#type().unwrap(), alloca_type);
                scope_block.append_operation(alloca_op).unwrap();

                let empty_values = Vec::<ValueRef>::new();
                let scope_return_op = alloca_scope_return(&empty_values, location).unwrap();
                assert_eq!(scope_return_op.values().collect::<Result<Vec<_>, _>>().unwrap(), empty_values);
                scope_block.append_operation(scope_return_op).unwrap();

                let empty_result_types = Vec::<TypeRef>::new();
                let alloca_scope_op =
                    alloca_scope(&empty_result_types, scope_block.try_into().unwrap(), location).unwrap();
                assert_eq!(alloca_scope_op.result_count(), 0);
                assert_eq!(alloca_scope_op.region_count(), 1);
                block.append_operation(alloca_scope_op).unwrap();

                let reshape_op = reshape(source, shape, result_type, location).unwrap();
                assert_eq!(reshape_op.source().unwrap(), source);
                assert_eq!(reshape_op.shape().unwrap(), shape);
                assert_eq!(reshape_op.source().unwrap(), source);
                let reshape_op = block.append_operation(reshape_op).unwrap();

                block
                    .append_operation(func::r#return(&[reshape_op.result(0).unwrap().as_ref()], location).unwrap())
                    .unwrap();
                func::func(
                    "memref_reshape",
                    func::FuncAttributes {
                        arguments: vec![source_type.into(), shape_type.into()],
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
                  func.func @memref_reshape(%arg0: memref<4x1xf32>, %arg1: memref<1xi32>) -> memref<4xf32> {
                    memref.alloca_scope  {
                      %alloca = memref.alloca() {alignment = 16 : i64} : memref<4xf32>
                    }
                    %reshape = memref.reshape %arg0(%arg1) : (memref<4x1xf32>, memref<1xi32>) -> memref<4xf32>
                    return %reshape : memref<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_subview() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let source_type =
            context.mem_ref_type(f32_type, &[Size::Static(8), Size::Static(16)], None, None, location).unwrap();
        let result_layout = context.strided_layout_attribute(18, &[16, 1]);
        let result_type = context
            .mem_ref_type(f32_type, &[Size::Static(4), Size::Static(8)], Some(result_layout.as_ref()), None, location)
            .unwrap();

        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(source_type, location)]);
                let source = block.argument(0).unwrap().as_ref();
                let offsets = [StaticOrDynamicIndex::Static(1), StaticOrDynamicIndex::Static(2)];
                let sizes = [StaticOrDynamicIndex::Static(4), StaticOrDynamicIndex::Static(8)];
                let strides = [StaticOrDynamicIndex::Static(1), StaticOrDynamicIndex::Static(1)];
                let subview_op = subview(source, &offsets, &sizes, &strides, result_type, location).unwrap();
                assert_eq!(subview_op.source().unwrap(), source);
                assert_eq!(subview_op.offsets().unwrap(), offsets);
                assert_eq!(subview_op.sizes().unwrap(), sizes);
                assert_eq!(subview_op.strides().unwrap(), strides);
                assert_eq!(subview_op.source().unwrap(), source);
                let subview_op = block.append_operation(subview_op).unwrap();
                block
                    .append_operation(func::r#return(&[subview_op.result(0).unwrap().as_ref()], location).unwrap())
                    .unwrap();
                func::func(
                    "memref_subview",
                    func::FuncAttributes {
                        arguments: vec![source_type.into()],
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
                  func.func @memref_subview(%arg0: memref<8x16xf32>) -> memref<4x8xf32, strided<[16, 1], offset: 18>> {
                    %subview = memref.subview %arg0[1, 2] [4, 8] [1, 1] : memref<8x16xf32> to memref<4x8xf32, strided<[16, 1], offset: 18>>
                    return %subview : memref<4x8xf32, strided<[16, 1], offset: 18>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_atomic_rmw_operations() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f32_type = context.float32_type();
        let memref_type = context.mem_ref_type(f32_type, &[Size::Static(10)], None, None, location).unwrap();

        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (memref_type.as_ref(), location),
                    (index_type.as_ref(), location),
                    (f32_type.as_ref(), location),
                ]);
                let memref = block.argument(0).unwrap().as_ref();
                let index = block.argument(1).unwrap().as_ref();
                let value = block.argument(2).unwrap().as_ref();

                let atomic_rmw_op = atomic_rmw(AtomicRmwKind::AddFloat, value, memref, &[index], location).unwrap();
                assert_eq!(atomic_rmw_op.kind().unwrap(), AtomicRmwKind::AddFloat);
                assert_eq!(atomic_rmw_op.value().unwrap(), value);
                assert_eq!(atomic_rmw_op.memref().unwrap(), memref);
                assert_eq!(atomic_rmw_op.indices().unwrap(), vec![index]);
                assert_eq!(atomic_rmw_op.kind().unwrap(), AtomicRmwKind::AddFloat);
                let atomic_rmw_op = block.append_operation(atomic_rmw_op).unwrap();

                let mut atomic_block = context.block(&[(f32_type.as_ref(), location)]);
                let current_value = atomic_block.argument(0).unwrap().as_ref();
                let atomic_yield_op = atomic_yield(current_value, location).unwrap();
                assert_eq!(atomic_yield_op.value().unwrap(), current_value);
                atomic_block.append_operation(atomic_yield_op).unwrap();

                let generic_atomic_rmw_op =
                    generic_atomic_rmw(memref, &[index], f32_type, atomic_block.try_into().unwrap(), location).unwrap();
                assert_eq!(generic_atomic_rmw_op.memref().unwrap(), memref);
                assert_eq!(generic_atomic_rmw_op.indices().unwrap(), vec![index]);
                assert_eq!(generic_atomic_rmw_op.memref().unwrap(), memref);
                assert_eq!(generic_atomic_rmw_op.region_count(), 1);
                let generic_atomic_rmw_op = block.append_operation(generic_atomic_rmw_op).unwrap();

                block
                    .append_operation(
                        func::r#return(
                            &[
                                atomic_rmw_op.result(0).unwrap().as_ref(),
                                generic_atomic_rmw_op.result(0).unwrap().as_ref(),
                            ],
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                func::func(
                    "memref_atomics",
                    func::FuncAttributes {
                        arguments: vec![memref_type.into(), index_type.into(), f32_type.into()],
                        results: vec![f32_type.into(), f32_type.into()],
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
                  func.func @memref_atomics(%arg0: memref<10xf32>, %arg1: index, %arg2: f32) -> (f32, f32) {
                    %0 = memref.atomic_rmw addf %arg2, %arg0[%arg1] : (f32, memref<10xf32>) -> f32
                    %1 = memref.generic_atomic_rmw %arg0[%arg1] : memref<10xf32> {
                    ^bb0(%arg3: f32):
                      memref.atomic_yield %arg3 : f32
                    }
                    return %0, %1 : f32, f32
                  }
                }
            "},
        );
    }
}
