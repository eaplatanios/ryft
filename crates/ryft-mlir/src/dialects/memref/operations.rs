use crate::{
    Attribute, AttributeRef, BooleanAttributeRef, DenseInteger32ArrayAttributeRef, DenseInteger64ArrayAttributeRef,
    DetachedOp, DetachedRegion, DialectHandle, FlatSymbolRefAttributeRef, IntegerAttributeRef, IntoWithContext,
    Location, OneRegion, OneResult, Operation, OperationBuilder, RegionRef, Size, StringAttributeRef, Symbol,
    SymbolVisibility, Type, TypeAttributeRef, TypeRef, Value, ValueRef, mlir_op, mlir_op_trait,
};
use crate::{SYMBOL_NAME_ATTRIBUTE, SYMBOL_VISIBILITY_ATTRIBUTE};

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
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the assumed byte alignment.
    fn alignment(&self) -> i32 {
        self.attribute(ALIGNMENT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as i32)
            .unwrap_or_else(|| panic!("invalid '{ALIGNMENT_ATTRIBUTE}' attribute in `memref::assume_alignment`"))
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
) -> DetachedAssumeAlignmentOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    OperationBuilder::new("memref.assume_alignment", location)
        .add_operand(memref)
        .add_attribute(
            ALIGNMENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), alignment.into()),
        )
        .add_result(memref.r#type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::assume_alignment`")
}

/// Operation trait for the `memref.distinct_objects` operation.
pub trait DistinctObjectsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input memrefs asserted to be mutually non-aliasing.
    fn inputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the memref values carrying the non-aliasing assumption.
    fn outputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.results().map(|result| result.as_ref()).collect()
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
) -> DetachedDistinctObjectsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    let result_types = operands.iter().map(|operand| operand.r#type()).collect::<Vec<_>>();
    OperationBuilder::new("memref.distinct_objects", location)
        .add_operands(operands)
        .add_results(&result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::distinct_objects`")
}

/// Name of the [`Attribute`] that stores operand segment sizes for variadic MemRef operations.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Operation trait shared by `memref.alloc` and `memref.alloca`.
pub trait AllocLikeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the dynamic dimension operands.
    fn dynamic_sizes(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = operand_segment_sizes(self, self.name().as_str().unwrap());
        (0..sizes[0] as usize).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the symbolic operands bound to the memref layout map.
    fn symbol_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = operand_segment_sizes(self, self.name().as_str().unwrap());
        let start = sizes[0] as usize;
        let end = start + sizes[1] as usize;
        (start..end).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Option<i64> {
        self.attribute(ALIGNMENT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
    }

    /// Returns the allocated memref result.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

fn operand_segment_sizes<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
    operation_name: &str,
) -> Vec<i32> {
    operation
        .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
        .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
        .map(|attribute| attribute.values().collect())
        .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `{operation_name}`"))
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
) -> DetachedAllocOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    let mut builder = OperationBuilder::new("memref.alloc", location)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[dynamic_sizes.len() as i32, symbol_operands.len() as i32])
                .unwrap(),
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
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::alloc`")
}

/// Operation trait for the `memref.realloc` operation.
pub trait ReallocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref being reallocated.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional dynamic result size.
    fn dynamic_result_size(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(1)
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Option<i64> {
        self.attribute(ALIGNMENT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
    }

    /// Returns the reallocated memref.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
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
) -> DetachedReallocOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
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
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::realloc`")
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
) -> DetachedAllocaOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    let mut builder = OperationBuilder::new("memref.alloca", location)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[dynamic_sizes.len() as i32, symbol_operands.len() as i32])
                .unwrap(),
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
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::alloca`")
}

/// Operation trait for the `memref.alloca_scope` operation.
pub trait AllocaScopeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> {
    /// Returns the alloca scope body.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
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
) -> DetachedAllocaScopeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    OperationBuilder::new("memref.alloca_scope", location)
        .add_results(result_types)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::alloca_scope`")
}

/// Operation trait for the `memref.alloca_scope.return` operation.
pub trait AllocaScopeReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the values yielded from the alloca scope.
    fn values(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
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
) -> DetachedAllocaScopeReturnOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    OperationBuilder::new("memref.alloca_scope.return", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::alloca_scope_return`")
}

/// Operation trait for the `memref.cast` operation.
pub trait CastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination memref.
    fn dest(&self) -> ValueRef<'o, 'c, 't> {
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
) -> DetachedCastOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    OperationBuilder::new("memref.cast", location)
        .add_operand(source)
        .add_result(dest_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::cast`")
}

/// Operation trait for the `memref.copy` operation.
pub trait CopyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the target memref.
    fn target(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
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
) -> DetachedCopyOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    OperationBuilder::new("memref.copy", location)
        .add_operand(source)
        .add_operand(target)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::copy`")
}

/// Operation trait for the `memref.dealloc` operation.
pub trait DeallocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the memref being deallocated.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
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
) -> DetachedDeallocOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    OperationBuilder::new("memref.dealloc", location)
        .add_operand(memref)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::dealloc`")
}

/// Operation trait for the `memref.dim` operation.
pub trait DimOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the dimension index.
    fn index(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
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
) -> DetachedDimOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    OperationBuilder::new("memref.dim", location)
        .add_operand(source)
        .add_operand(index)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::dim`")
}

/// Operation trait for the `memref.extract_aligned_pointer_as_index` operation.
pub trait ExtractAlignedPointerAsIndexOperation<'o, 'c: 'o, 't: 'c>:
    Operation<'o, 'c, 't> + OneResult<'o, 'c, 't>
{
    /// Returns the source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the extracted aligned pointer represented as an index value.
    fn aligned_pointer(&self) -> ValueRef<'o, 'c, 't> {
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
) -> DetachedExtractAlignedPointerAsIndexOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    OperationBuilder::new("memref.extract_aligned_pointer_as_index", location)
        .add_operand(source)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::extract_aligned_pointer_as_index`")
}

/// Name of the [`Attribute`] that stores the referenced global symbol name.
pub const NAME_ATTRIBUTE: &str = "name";

/// Operation trait for the `memref.get_global` operation.
pub trait GetGlobalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the referenced global symbol.
    fn name(&self) -> FlatSymbolRefAttributeRef<'c, 't> {
        self.attribute(NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<FlatSymbolRefAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{NAME_ATTRIBUTE}' attribute in `memref::get_global`"))
    }

    /// Returns the retrieved global memref.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
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
    N: IntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    name: N,
    result_type: T,
    location: L,
) -> DetachedGetGlobalOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    OperationBuilder::new("memref.get_global", location)
        .add_attribute(NAME_ATTRIBUTE, name.into_with_context(context))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::get_global`")
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
    fn r#type(&self) -> TypeRef<'c, 't> {
        self.attribute(TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>())
            .map(|attribute| attribute.r#type())
            .unwrap_or_else(|| panic!("invalid '{TYPE_ATTRIBUTE}' attribute in `memref::global`"))
    }

    /// Returns the optional initial value.
    fn initial_value(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute(INITIAL_VALUE_ATTRIBUTE)
    }

    /// Returns `true` if this global is constant.
    fn is_constant(&self) -> bool {
        self.has_attribute(CONSTANT_ATTRIBUTE)
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Option<i64> {
        self.attribute(ALIGNMENT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
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
    N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
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
) -> DetachedGlobalOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    let mut builder = OperationBuilder::new("memref.global", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, name.into_with_context(context))
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
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::global`")
}

/// Name of the [`Attribute`] that marks a load or store as non-temporal.
pub const NONTEMPORAL_ATTRIBUTE: &str = "nontemporal";

/// Operation trait for the `memref.load` operation.
pub trait LoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the memref being loaded from.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the index operands used to access the memref.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (1..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns `true` if this load is marked as non-temporal.
    fn nontemporal(&self) -> bool {
        self.attribute(NONTEMPORAL_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or(false)
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Option<i64> {
        self.attribute(ALIGNMENT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
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
) -> DetachedLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
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
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::load`")
}

/// Operation trait for the `memref.memory_space_cast` operation.
pub trait MemorySpaceCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination memref.
    fn dest(&self) -> ValueRef<'o, 'c, 't> {
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
) -> DetachedMemorySpaceCastOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    OperationBuilder::new("memref.memory_space_cast", location)
        .add_operand(source)
        .add_result(dest_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::memory_space_cast`")
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
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the index operands used to access the memref.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (1..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns `true` if the prefetch is for a write.
    fn is_write(&self) -> bool {
        self.attribute(IS_WRITE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| panic!("invalid '{IS_WRITE_ATTRIBUTE}' attribute in `memref::prefetch`"))
    }

    /// Returns the locality hint in the range `0..=3`.
    fn locality_hint(&self) -> i32 {
        self.attribute(LOCALITY_HINT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as i32)
            .unwrap_or_else(|| panic!("invalid '{LOCALITY_HINT_ATTRIBUTE}' attribute in `memref::prefetch`"))
    }

    /// Returns `true` if the prefetch targets the data cache.
    fn is_data_cache(&self) -> bool {
        self.attribute(IS_DATA_CACHE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| panic!("invalid '{IS_DATA_CACHE_ATTRIBUTE}' attribute in `memref::prefetch`"))
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
) -> DetachedPrefetchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
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
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::prefetch`")
}

/// Operation trait for the `memref.rank` operation.
pub trait RankOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
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
) -> DetachedRankOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    OperationBuilder::new("memref.rank", location)
        .add_operand(memref)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::rank`")
}

/// Operation trait for the `memref.reshape` operation.
pub trait ReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the memref containing the dynamic shape.
    fn shape(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the reshaped memref.
    fn reshaped(&self) -> ValueRef<'o, 'c, 't> {
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
) -> DetachedReshapeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
    OperationBuilder::new("memref.reshape", location)
        .add_operand(source)
        .add_operand(shape)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::reshape`")
}

/// Operation trait for the `memref.store` operation.
pub trait StoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the stored value.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the memref being stored into.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the index operands used to access the memref.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (2..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns `true` if this store is marked as non-temporal.
    fn nontemporal(&self) -> bool {
        self.attribute(NONTEMPORAL_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or(false)
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Option<i64> {
        self.attribute(ALIGNMENT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
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
) -> DetachedStoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
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
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::store`")
}

/// Name of the [`Attribute`] that stores static offset entries for view-like operations.
pub const STATIC_OFFSETS_ATTRIBUTE: &str = "static_offsets";

/// Name of the [`Attribute`] that stores static size entries for view-like operations.
pub const STATIC_SIZES_ATTRIBUTE: &str = "static_sizes";

/// Name of the [`Attribute`] that stores static stride entries for view-like operations.
pub const STATIC_STRIDES_ATTRIBUTE: &str = "static_strides";

/// Operation trait for the `memref.subview` operation.
pub trait SubViewOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the mixed static and dynamic offsets.
    fn offsets(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let sizes = operand_segment_sizes(self, "memref::subview");
        let dynamic_start = 1;
        let dynamic_end = dynamic_start + sizes[1] as usize;
        let dynamic_offsets =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_offsets = self
            .attribute(STATIC_OFFSETS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<i64>>())
            .unwrap_or_else(|| panic!("invalid '{STATIC_OFFSETS_ATTRIBUTE}' attribute in `memref::subview`"));
        let dynamic_size = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_offsets = dynamic_offsets.into_iter();
        static_offsets
            .into_iter()
            .map(|index| {
                (index == dynamic_size)
                    .then(|| dynamic_offsets.next().expect("missing dynamic offset operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns the mixed static and dynamic sizes.
    fn sizes(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = operand_segment_sizes(self, "memref::subview");
        let dynamic_start = 1 + segment_sizes[1] as usize;
        let dynamic_end = dynamic_start + segment_sizes[2] as usize;
        let dynamic_sizes =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_sizes = self
            .attribute(STATIC_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<i64>>())
            .unwrap_or_else(|| panic!("invalid '{STATIC_SIZES_ATTRIBUTE}' attribute in `memref::subview`"));
        let dynamic_size = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_sizes = dynamic_sizes.into_iter();
        static_sizes
            .into_iter()
            .map(|index| {
                (index == dynamic_size)
                    .then(|| dynamic_sizes.next().expect("missing dynamic size operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns the mixed static and dynamic strides.
    fn strides(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = operand_segment_sizes(self, "memref::subview");
        let dynamic_start = 1 + segment_sizes[1] as usize + segment_sizes[2] as usize;
        let dynamic_end = dynamic_start + segment_sizes[3] as usize;
        let dynamic_strides =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_strides = self
            .attribute(STATIC_STRIDES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<i64>>())
            .unwrap_or_else(|| panic!("invalid '{STATIC_STRIDES_ATTRIBUTE}' attribute in `memref::subview`"));
        let dynamic_size = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_strides = dynamic_strides.into_iter();
        static_strides
            .into_iter()
            .map(|index| {
                (index == dynamic_size)
                    .then(|| dynamic_strides.next().expect("missing dynamic stride operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns the resulting subview memref.
    fn subview(&self) -> ValueRef<'o, 'c, 't> {
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
) -> DetachedSubViewOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::memref());
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
            context
                .dense_i32_array_attribute(&[
                    1,
                    dynamic_offsets.len() as i32,
                    dynamic_sizes.len() as i32,
                    dynamic_strides.len() as i32,
                ])
                .unwrap(),
        )
        .add_attribute(STATIC_OFFSETS_ATTRIBUTE, context.dense_i64_array_attribute(&static_offsets).unwrap())
        .add_attribute(STATIC_SIZES_ATTRIBUTE, context.dense_i64_array_attribute(&static_sizes).unwrap())
        .add_attribute(STATIC_STRIDES_ATTRIBUTE, context.dense_i64_array_attribute(&static_strides).unwrap())
        .add_operand(source)
        .add_operands(&dynamic_offsets)
        .add_operands(&dynamic_sizes)
        .add_operands(&dynamic_strides)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `memref::subview`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, Operation, Symbol, Type, Value};

    use super::*;

    #[test]
    fn test_alloc_load_store_dealloc_and_prefetch() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        let f32_type = context.float32_type();
        let memref_type = context.mem_ref_type(f32_type, &[Size::Dynamic], None, None, location).unwrap();

        module.body().append_operation({
            let mut block = context.block(&[(index_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let dynamic_size = block.argument(0).unwrap().as_ref();
            let value = block.argument(1).unwrap().as_ref();

            let alloc_op = alloc(&[dynamic_size], &[], memref_type, Some(64), location);
            assert_eq!(alloc_op.dynamic_sizes(), vec![dynamic_size]);
            assert_eq!(alloc_op.symbol_operands(), Vec::<ValueRef>::new());
            assert_eq!(alloc_op.alignment(), Some(64));
            assert_eq!(alloc_op.memref().r#type(), memref_type);
            let alloc_op = block.append_operation(alloc_op);
            let memref = alloc_op.result(0).unwrap().as_ref();

            let store_op = store(value, memref, &[dynamic_size], true, Some(4), location);
            assert_eq!(store_op.value(), value);
            assert_eq!(store_op.memref(), memref);
            assert_eq!(store_op.indices(), vec![dynamic_size]);
            assert_eq!(store_op.nontemporal(), true);
            assert_eq!(store_op.alignment(), Some(4));
            block.append_operation(store_op);

            let prefetch_op = prefetch(memref, &[dynamic_size], false, 3, true, location);
            assert_eq!(prefetch_op.memref(), memref);
            assert_eq!(prefetch_op.indices(), vec![dynamic_size]);
            assert_eq!(prefetch_op.is_write(), false);
            assert_eq!(prefetch_op.locality_hint(), 3);
            assert_eq!(prefetch_op.is_data_cache(), true);
            block.append_operation(prefetch_op);

            let load_op = load(memref, &[dynamic_size], f32_type, false, None, location);
            assert_eq!(load_op.memref(), memref);
            assert_eq!(load_op.indices(), vec![dynamic_size]);
            assert_eq!(load_op.output_type(), f32_type);
            assert_eq!(load_op.nontemporal(), false);
            assert_eq!(load_op.alignment(), None);
            let load_op = block.append_operation(load_op);

            let dealloc_op = dealloc(memref, location);
            assert_eq!(dealloc_op.memref(), memref);
            block.append_operation(dealloc_op);

            block.append_operation(func::r#return(&[load_op.result(0).unwrap().as_ref()], location));
            func::func(
                "memref_access",
                func::FuncAttributes {
                    arguments: vec![index_type.into(), f32_type.into()],
                    results: vec![f32_type.into()],
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
        let module = context.module(location);
        let index_type = context.index_type();
        let f32_type = context.float32_type();
        let source_type =
            context.mem_ref_type(f32_type, &[Size::Static(4), Size::Dynamic], None, None, location).unwrap();
        let cast_type = context.mem_ref_type(f32_type, &[Size::Dynamic, Size::Dynamic], None, None, location).unwrap();

        module.body().append_operation({
            let mut block = context.block(&[(source_type.as_ref(), location), (index_type.as_ref(), location)]);
            let source = block.argument(0).unwrap().as_ref();
            let index = block.argument(1).unwrap().as_ref();

            let assume_alignment_op = assume_alignment(source, 16, location);
            assert_eq!(assume_alignment_op.memref(), source);
            assert_eq!(assume_alignment_op.alignment(), 16);
            assert_eq!(assume_alignment_op.output_type(), source_type);
            let assume_alignment_op = block.append_operation(assume_alignment_op);
            let aligned = assume_alignment_op.result(0).unwrap().as_ref();

            let cast_op = cast(aligned, cast_type, location);
            assert_eq!(cast_op.source(), aligned);
            assert_eq!(cast_op.dest().r#type(), cast_type);
            let cast_op = block.append_operation(cast_op);
            let cast_memref = cast_op.result(0).unwrap().as_ref();

            let dim_op = dim(source, index, location);
            assert_eq!(dim_op.source(), source);
            assert_eq!(dim_op.index(), index);
            assert_eq!(dim_op.output_type(), index_type);
            let dim_op = block.append_operation(dim_op);

            let rank_op = rank(cast_memref, location);
            assert_eq!(rank_op.memref(), cast_memref);
            assert_eq!(rank_op.output_type(), index_type);
            let rank_op = block.append_operation(rank_op);

            let pointer_op = extract_aligned_pointer_as_index(cast_memref, location);
            assert_eq!(pointer_op.source(), cast_memref);
            assert_eq!(pointer_op.aligned_pointer().r#type(), index_type);
            let pointer_op = block.append_operation(pointer_op);

            block.append_operation(func::r#return(
                &[
                    cast_memref,
                    dim_op.result(0).unwrap().as_ref(),
                    rank_op.result(0).unwrap().as_ref(),
                    pointer_op.result(0).unwrap().as_ref(),
                ],
                location,
            ));
            func::func(
                "memref_shape",
                func::FuncAttributes {
                    arguments: vec![source_type.into(), index_type.into()],
                    results: vec![cast_type.into(), index_type.into(), index_type.into(), index_type.into()],
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
    fn test_global_and_get_global() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
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
        );
        assert_eq!(global_op.symbol_name().unwrap().as_str().unwrap(), "weights");
        assert_eq!(global_op.symbol_visibility(), SymbolVisibility::Private);
        assert_eq!(global_op.r#type(), memref_type);
        assert!(global_op.initial_value().is_some());
        assert_eq!(global_op.is_constant(), false);
        assert_eq!(global_op.alignment(), Some(64));
        module.body().append_operation(global_op);

        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let get_global_op = get_global("weights", memref_type, location);
            assert_eq!(GetGlobalOperation::name(&get_global_op), context.flat_symbol_ref_attribute("weights"));
            assert_eq!(get_global_op.memref().r#type(), memref_type);
            let get_global_op = block.append_operation(get_global_op);
            block.append_operation(func::r#return(&[get_global_op.result(0).unwrap().as_ref()], location));
            func::func(
                "get_weights",
                func::FuncAttributes { results: vec![memref_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });

        assert!(module.verify());
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
        let module = context.module(location);
        let index_type = context.index_type();
        let f32_type = context.float32_type();
        let memref_type = context.mem_ref_type(f32_type, &[Size::Dynamic], None, None, location).unwrap();
        let memory_space = context.integer_attribute(context.signless_integer_type(64), 3);
        let memory_space_type = context
            .mem_ref_type(f32_type, &[Size::Dynamic], None, Some(memory_space.as_ref()), location)
            .unwrap();

        module.body().append_operation({
            let mut block = context.block(&[
                (index_type.as_ref(), location),
                (memref_type.as_ref(), location),
                (memref_type.as_ref(), location),
            ]);
            let dynamic_size = block.argument(0).unwrap().as_ref();
            let source = block.argument(1).unwrap().as_ref();
            let target = block.argument(2).unwrap().as_ref();

            let distinct_op = distinct_objects(&[source, target], location);
            assert_eq!(distinct_op.inputs(), vec![source, target]);
            assert_eq!(
                distinct_op.outputs().iter().map(|value| value.r#type()).collect::<Vec<_>>(),
                vec![memref_type.as_ref(), memref_type.as_ref()]
            );
            let distinct_op = block.append_operation(distinct_op);
            let distinct_source = distinct_op.result(0).unwrap().as_ref();
            let distinct_target = distinct_op.result(1).unwrap().as_ref();

            let copy_op = copy(distinct_source, distinct_target, location);
            assert_eq!(copy_op.source(), distinct_source);
            assert_eq!(copy_op.target(), distinct_target);
            block.append_operation(copy_op);

            let realloc_op = realloc(distinct_source, Some(dynamic_size), memref_type, Some(128), location);
            assert_eq!(realloc_op.source(), distinct_source);
            assert_eq!(realloc_op.dynamic_result_size(), Some(dynamic_size));
            assert_eq!(realloc_op.alignment(), Some(128));
            assert_eq!(realloc_op.memref().r#type(), memref_type);
            let realloc_op = block.append_operation(realloc_op);

            let memory_space_cast_op = memory_space_cast(distinct_target, memory_space_type, location);
            assert_eq!(memory_space_cast_op.source(), distinct_target);
            assert_eq!(memory_space_cast_op.dest().r#type(), memory_space_type);
            let memory_space_cast_op = block.append_operation(memory_space_cast_op);

            block.append_operation(func::r#return(
                &[
                    distinct_target,
                    realloc_op.result(0).unwrap().as_ref(),
                    memory_space_cast_op.result(0).unwrap().as_ref(),
                ],
                location,
            ));
            func::func(
                "memref_misc",
                func::FuncAttributes {
                    arguments: vec![index_type.into(), memref_type.into(), memref_type.into()],
                    results: vec![memref_type.into(), memref_type.into(), memory_space_type.into()],
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
        let module = context.module(location);
        let f32_type = context.float32_type();
        let i32_type = context.signless_integer_type(32);
        let source_type =
            context.mem_ref_type(f32_type, &[Size::Static(4), Size::Static(1)], None, None, location).unwrap();
        let shape_type = context.mem_ref_type(i32_type, &[Size::Static(1)], None, None, location).unwrap();
        let alloca_type = context.mem_ref_type(f32_type, &[Size::Static(4)], None, None, location).unwrap();
        let result_type = context.mem_ref_type(f32_type, &[Size::Static(4)], None, None, location).unwrap();

        module.body().append_operation({
            let mut block = context.block(&[(source_type.as_ref(), location), (shape_type.as_ref(), location)]);
            let source = block.argument(0).unwrap().as_ref();
            let shape = block.argument(1).unwrap().as_ref();

            let mut scope_block = context.block_with_no_arguments();
            let alloca_op = alloca(&[], &[], alloca_type, Some(16), location);
            assert_eq!(alloca_op.dynamic_sizes(), Vec::<ValueRef>::new());
            assert_eq!(alloca_op.symbol_operands(), Vec::<ValueRef>::new());
            assert_eq!(alloca_op.alignment(), Some(16));
            assert_eq!(alloca_op.memref().r#type(), alloca_type);
            scope_block.append_operation(alloca_op);

            let empty_values = Vec::<ValueRef>::new();
            let scope_return_op = alloca_scope_return(&empty_values, location);
            assert_eq!(scope_return_op.values().collect::<Vec<_>>(), empty_values);
            scope_block.append_operation(scope_return_op);

            let empty_result_types = Vec::<TypeRef>::new();
            let alloca_scope_op = alloca_scope(&empty_result_types, scope_block.into(), location);
            assert_eq!(alloca_scope_op.result_count(), 0);
            assert_eq!(alloca_scope_op.region_count(), 1);
            block.append_operation(alloca_scope_op);

            let reshape_op = reshape(source, shape, result_type, location);
            assert_eq!(reshape_op.source(), source);
            assert_eq!(reshape_op.shape(), shape);
            assert_eq!(reshape_op.reshaped().r#type(), result_type);
            let reshape_op = block.append_operation(reshape_op);

            block.append_operation(func::r#return(&[reshape_op.result(0).unwrap().as_ref()], location));
            func::func(
                "memref_reshape",
                func::FuncAttributes {
                    arguments: vec![source_type.into(), shape_type.into()],
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
        let module = context.module(location);
        let f32_type = context.float32_type();
        let source_type =
            context.mem_ref_type(f32_type, &[Size::Static(8), Size::Static(16)], None, None, location).unwrap();
        let result_layout = context.strided_layout_attribute(18, &[16, 1]);
        let result_type = context
            .mem_ref_type(f32_type, &[Size::Static(4), Size::Static(8)], Some(result_layout.as_ref()), None, location)
            .unwrap();

        module.body().append_operation({
            let mut block = context.block(&[(source_type, location)]);
            let source = block.argument(0).unwrap().as_ref();
            let offsets = [StaticOrDynamicIndex::Static(1), StaticOrDynamicIndex::Static(2)];
            let sizes = [StaticOrDynamicIndex::Static(4), StaticOrDynamicIndex::Static(8)];
            let strides = [StaticOrDynamicIndex::Static(1), StaticOrDynamicIndex::Static(1)];
            let subview_op = subview(source, &offsets, &sizes, &strides, result_type, location);
            assert_eq!(subview_op.source(), source);
            assert_eq!(subview_op.offsets(), offsets);
            assert_eq!(subview_op.sizes(), sizes);
            assert_eq!(subview_op.strides(), strides);
            assert_eq!(subview_op.subview().r#type(), result_type);
            let subview_op = block.append_operation(subview_op);
            block.append_operation(func::r#return(&[subview_op.result(0).unwrap().as_ref()], location));
            func::func(
                "memref_subview",
                func::FuncAttributes {
                    arguments: vec![source_type.into()],
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
                  func.func @memref_subview(%arg0: memref<8x16xf32>) -> memref<4x8xf32, strided<[16, 1], offset: 18>> {
                    %subview = memref.subview %arg0[1, 2] [4, 8] [1, 1] : memref<8x16xf32> to memref<4x8xf32, strided<[16, 1], offset: 18>>
                    return %subview : memref<4x8xf32, strided<[16, 1], offset: 18>>
                  }
                }
            "},
        );
    }
}
