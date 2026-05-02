use crate::{
    Attribute, AttributeRef, Block, BlockRef, DenseInteger32ArrayAttributeRef, DetachedOp, DetachedRegion,
    DialectHandle, Location, Operation, OperationBuilder, TypeRef, Value, ValueRef, mlir_op,
};

/// Canonical MLIR operation name for [`AliasOperation`].
pub const ALIAS_OPERATION_NAME: &str = "llvm.mlir.alias";

/// Operation trait for `llvm.mlir.alias`.
pub trait AliasOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ALIAS_OPERATION_NAME
    }

    /// Returns the `alias_type` attribute.
    fn alias_type(&self) -> AttributeRef<'c, 't> {
        self.attribute("alias_type").unwrap()
    }

    /// Returns the `sym_name` attribute.
    fn sym_name(&self) -> AttributeRef<'c, 't> {
        self.attribute("sym_name").unwrap()
    }

    /// Returns the `linkage` attribute.
    fn linkage(&self) -> AttributeRef<'c, 't> {
        self.attribute("linkage").unwrap()
    }

    /// Returns whether the `dso_local` unit attribute is present.
    fn dso_local(&self) -> bool {
        self.has_attribute("dso_local")
    }

    /// Returns whether the `thread_local_` unit attribute is present.
    fn thread_local_(&self) -> bool {
        self.has_attribute("thread_local_")
    }

    /// Returns the optional `unnamed_addr` attribute.
    fn unnamed_addr(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("unnamed_addr")
    }

    /// Returns the optional `visibility_` attribute.
    fn visibility_(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("visibility_")
    }
}

mlir_op!(Alias);

/// Constructs a new detached `llvm.mlir.alias` operation.
pub fn alias<'c, 't: 'c, L: Location<'c, 't>>(
    alias_type: AttributeRef<'c, 't>,
    sym_name: AttributeRef<'c, 't>,
    linkage: AttributeRef<'c, 't>,
    dso_local: bool,
    thread_local_: bool,
    unnamed_addr: Option<AttributeRef<'c, 't>>,
    visibility_: Option<AttributeRef<'c, 't>>,
    initializer: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedAliasOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ALIAS_OPERATION_NAME, location);
    builder = builder.add_attribute("alias_type", alias_type);
    builder = builder.add_attribute("sym_name", sym_name);
    builder = builder.add_attribute("linkage", linkage);
    if dso_local {
        builder = builder.add_attribute("dso_local", context.unit_attribute());
    }
    if thread_local_ {
        builder = builder.add_attribute("thread_local_", context.unit_attribute());
    }
    if let Some(unnamed_addr) = unnamed_addr {
        builder = builder.add_attribute("unnamed_addr", unnamed_addr);
    }
    if let Some(visibility_) = visibility_ {
        builder = builder.add_attribute("visibility_", visibility_);
    }
    builder = builder.add_region(initializer);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::alias`")
}

/// Canonical MLIR operation name for [`AtomicCmpXchgOperation`].
pub const ATOMIC_CMP_XCHG_OPERATION_NAME: &str = "llvm.cmpxchg";

/// Operation trait for `llvm.cmpxchg`.
pub trait AtomicCmpXchgOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ATOMIC_CMP_XCHG_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `compare_value` operand.
    fn compare_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `new_value` operand.
    fn new_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `success_ordering` attribute.
    fn success_ordering(&self) -> AttributeRef<'c, 't> {
        self.attribute("success_ordering").unwrap()
    }

    /// Returns the `failure_ordering` attribute.
    fn failure_ordering(&self) -> AttributeRef<'c, 't> {
        self.attribute("failure_ordering").unwrap()
    }

    /// Returns the optional `syncscope` attribute.
    fn syncscope(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("syncscope")
    }

    /// Returns the optional `alignment` attribute.
    fn alignment(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("alignment")
    }

    /// Returns whether the `weak` unit attribute is present.
    fn weak(&self) -> bool {
        self.has_attribute("weak")
    }

    /// Returns whether the `volatile_` unit attribute is present.
    fn is_volatile(&self) -> bool {
        self.has_attribute("volatile_")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(AtomicCmpXchg);

/// Constructs a new detached `llvm.cmpxchg` operation.
pub fn cmpxchg<'c, 't: 'c, V1: Value<'c, 'c, 't>, V2: Value<'c, 'c, 't>, V3: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    pointer: V1,
    compare_value: V2,
    new_value: V3,
    result_type: TypeRef<'c, 't>,
    success_ordering: AttributeRef<'c, 't>,
    failure_ordering: AttributeRef<'c, 't>,
    syncscope: Option<AttributeRef<'c, 't>>,
    alignment: Option<AttributeRef<'c, 't>>,
    weak: bool,
    is_volatile: bool,
    location: L,
) -> DetachedAtomicCmpXchgOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ATOMIC_CMP_XCHG_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(compare_value);
    builder = builder.add_operand(new_value);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("success_ordering", success_ordering);
    builder = builder.add_attribute("failure_ordering", failure_ordering);
    if let Some(syncscope) = syncscope {
        builder = builder.add_attribute("syncscope", syncscope);
    }
    if let Some(alignment) = alignment {
        builder = builder.add_attribute("alignment", alignment);
    }
    if weak {
        builder = builder.add_attribute("weak", context.unit_attribute());
    }
    if is_volatile {
        builder = builder.add_attribute("volatile_", context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::cmpxchg`")
}

/// Canonical MLIR operation name for [`AtomicRmwOperation`].
pub const ATOMIC_RMW_OPERATION_NAME: &str = "llvm.atomicrmw";

/// Operation trait for `llvm.atomicrmw`.
pub trait AtomicRmwOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ATOMIC_RMW_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `bin_op` attribute.
    fn bin_op(&self) -> AttributeRef<'c, 't> {
        self.attribute("bin_op").unwrap()
    }

    /// Returns the `ordering` attribute.
    fn ordering(&self) -> AttributeRef<'c, 't> {
        self.attribute("ordering").unwrap()
    }

    /// Returns the optional `syncscope` attribute.
    fn syncscope(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("syncscope")
    }

    /// Returns the optional `alignment` attribute.
    fn alignment(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("alignment")
    }

    /// Returns whether the `volatile_` unit attribute is present.
    fn is_volatile(&self) -> bool {
        self.has_attribute("volatile_")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(AtomicRmw);

/// Constructs a new detached `llvm.atomicrmw` operation.
pub fn atomicrmw<'c, 't: 'c, V1: Value<'c, 'c, 't>, V2: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    pointer: V1,
    value: V2,
    result_type: TypeRef<'c, 't>,
    bin_op: AttributeRef<'c, 't>,
    ordering: AttributeRef<'c, 't>,
    syncscope: Option<AttributeRef<'c, 't>>,
    alignment: Option<AttributeRef<'c, 't>>,
    is_volatile: bool,
    location: L,
) -> DetachedAtomicRmwOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ATOMIC_RMW_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("bin_op", bin_op);
    builder = builder.add_attribute("ordering", ordering);
    if let Some(syncscope) = syncscope {
        builder = builder.add_attribute("syncscope", syncscope);
    }
    if let Some(alignment) = alignment {
        builder = builder.add_attribute("alignment", alignment);
    }
    if is_volatile {
        builder = builder.add_attribute("volatile_", context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::atomicrmw`")
}

/// Canonical MLIR operation name for [`BlockAddressOperation`].
pub const BLOCK_ADDRESS_OPERATION_NAME: &str = "llvm.blockaddress";

/// Operation trait for `llvm.blockaddress`.
pub trait BlockAddressOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        BLOCK_ADDRESS_OPERATION_NAME
    }

    /// Returns the `block_addr` attribute.
    fn block_addr(&self) -> AttributeRef<'c, 't> {
        self.attribute("block_addr").unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(BlockAddress);

/// Constructs a new detached `llvm.blockaddress` operation.
pub fn blockaddress<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    block_addr: AttributeRef<'c, 't>,
    location: L,
) -> DetachedBlockAddressOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(BLOCK_ADDRESS_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("block_addr", block_addr);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::blockaddress`")
}

/// Canonical MLIR operation name for [`BlockTagOperation`].
pub const BLOCK_TAG_OPERATION_NAME: &str = "llvm.blocktag";

/// Operation trait for `llvm.blocktag`.
pub trait BlockTagOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        BLOCK_TAG_OPERATION_NAME
    }

    /// Returns the `tag` attribute.
    fn tag(&self) -> AttributeRef<'c, 't> {
        self.attribute("tag").unwrap()
    }
}

mlir_op!(BlockTag);

/// Constructs a new detached `llvm.blocktag` operation.
pub fn blocktag<'c, 't: 'c, L: Location<'c, 't>>(
    tag: AttributeRef<'c, 't>,
    location: L,
) -> DetachedBlockTagOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(BLOCK_TAG_OPERATION_NAME, location);
    builder = builder.add_attribute("tag", tag);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::blocktag`")
}

/// Canonical MLIR operation name for [`CallIntrinsicOperation`].
pub const CALL_INTRINSIC_OPERATION_NAME: &str = "llvm.call_intrinsic";

/// Operation trait for `llvm.call_intrinsic`.
pub trait CallIntrinsicOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CALL_INTRINSIC_OPERATION_NAME
    }

    /// Returns the `arguments` operands.
    fn arguments(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let argument_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[0])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.call_intrinsic`"));
        self.operand_values().take(argument_count as usize).collect()
    }

    /// Returns the `op_bundle_operands` operands.
    fn op_bundle_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let argument_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[0])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.call_intrinsic`"));
        self.operand_values().skip(argument_count as usize).collect()
    }

    /// Returns the `intrin` attribute.
    fn intrin(&self) -> AttributeRef<'c, 't> {
        self.attribute("intrin").unwrap()
    }

    /// Returns the optional `fastmath_flags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmath_flags")
    }

    /// Returns the optional `op_bundle_sizes` attribute.
    fn op_bundle_sizes(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("op_bundle_sizes")
    }

    /// Returns the optional `op_bundle_tags` attribute.
    fn op_bundle_tags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("op_bundle_tags")
    }

    /// Returns the optional `arg_attrs` attribute.
    fn arg_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("arg_attrs")
    }

    /// Returns the optional `res_attrs` attribute.
    fn res_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("res_attrs")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(CallIntrinsic);

/// Constructs a new detached `llvm.call_intrinsic` operation.
pub fn call_intrinsic<'c, 't: 'c, L: Location<'c, 't>>(
    arguments: &[ValueRef<'c, 'c, 't>],
    op_bundle_operands: &[ValueRef<'c, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    intrin: AttributeRef<'c, 't>,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    op_bundle_sizes: Option<AttributeRef<'c, 't>>,
    op_bundle_tags: Option<AttributeRef<'c, 't>>,
    arg_attrs: Option<AttributeRef<'c, 't>>,
    res_attrs: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedCallIntrinsicOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CALL_INTRINSIC_OPERATION_NAME, location);
    builder = builder.add_operands(arguments);
    builder = builder.add_operands(op_bundle_operands);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("intrin", intrin);
    builder = builder.add_attribute(
        "operand_segment_sizes",
        context
            .dense_i32_array_attribute(&[arguments.len() as i32, op_bundle_operands.len() as i32])
            .unwrap(),
    );
    builder = builder.add_attribute(
        "op_bundle_sizes",
        op_bundle_sizes.unwrap_or_else(|| context.dense_i32_array_attribute(&[]).unwrap().as_ref()),
    );
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmath_flags", fastmath_flags);
    }
    if let Some(op_bundle_tags) = op_bundle_tags {
        builder = builder.add_attribute("op_bundle_tags", op_bundle_tags);
    }
    if let Some(arg_attrs) = arg_attrs {
        builder = builder.add_attribute("arg_attrs", arg_attrs);
    }
    if let Some(res_attrs) = res_attrs {
        builder = builder.add_attribute("res_attrs", res_attrs);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::call_intrinsic`")
}

/// Canonical MLIR operation name for [`CallOperation`].
pub const CALL_OPERATION_NAME: &str = "llvm.call";

/// Operation trait for `llvm.call`.
pub trait CallOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CALL_OPERATION_NAME
    }

    /// Returns the `callee_operands` operands.
    fn callee_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(0).collect()
    }

    /// Returns the optional `callee` attribute.
    fn callee(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("callee")
    }

    /// Returns the optional `var_callee_type` attribute.
    fn var_callee_type(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("var_callee_type")
    }

    /// Returns the optional `fastmath_flags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmath_flags")
    }

    /// Returns the optional `calling_convention` attribute.
    fn calling_convention(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("CConv")
    }

    /// Returns the `op_bundle_operands` operands.
    fn op_bundle_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let callee_operand_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[0])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.call`"));
        self.operand_values().skip(callee_operand_count as usize).collect()
    }

    /// Returns the `op_bundle_sizes` attribute.
    fn op_bundle_sizes(&self) -> AttributeRef<'c, 't> {
        self.attribute("op_bundle_sizes").unwrap()
    }

    /// Returns the optional `op_bundle_tags` attribute.
    fn op_bundle_tags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("op_bundle_tags")
    }

    /// Returns the optional `arg_attrs` attribute.
    fn arg_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("arg_attrs")
    }

    /// Returns the optional `res_attrs` attribute.
    fn res_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("res_attrs")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Call);

/// Constructs a new detached `llvm.call` operation.
pub fn call<'c, 't: 'c, L: Location<'c, 't>>(
    callee_operands: &[ValueRef<'c, 'c, 't>],
    op_bundle_operands: &[ValueRef<'c, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    var_callee_type: Option<AttributeRef<'c, 't>>,
    callee: Option<AttributeRef<'c, 't>>,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    calling_convention: Option<AttributeRef<'c, 't>>,
    op_bundle_sizes: Option<AttributeRef<'c, 't>>,
    op_bundle_tags: Option<AttributeRef<'c, 't>>,
    arg_attrs: Option<AttributeRef<'c, 't>>,
    res_attrs: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedCallOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CALL_OPERATION_NAME, location);
    builder = builder.add_operands(callee_operands);
    builder = builder.add_operands(op_bundle_operands);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute(
        "operand_segment_sizes",
        context
            .dense_i32_array_attribute(&[callee_operands.len() as i32, op_bundle_operands.len() as i32])
            .unwrap(),
    );
    builder = builder.add_attribute(
        "op_bundle_sizes",
        op_bundle_sizes.unwrap_or_else(|| context.dense_i32_array_attribute(&[]).unwrap().as_ref()),
    );
    if let Some(var_callee_type) = var_callee_type {
        builder = builder.add_attribute("var_callee_type", var_callee_type);
    }
    if let Some(callee) = callee {
        builder = builder.add_attribute("callee", callee);
    }
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmath_flags", fastmath_flags);
    }
    if let Some(calling_convention) = calling_convention {
        builder = builder.add_attribute("CConv", calling_convention);
    }
    if let Some(op_bundle_tags) = op_bundle_tags {
        builder = builder.add_attribute("op_bundle_tags", op_bundle_tags);
    }
    if let Some(arg_attrs) = arg_attrs {
        builder = builder.add_attribute("arg_attrs", arg_attrs);
    }
    if let Some(res_attrs) = res_attrs {
        builder = builder.add_attribute("res_attrs", res_attrs);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::call`")
}

/// Canonical MLIR operation name for [`ComdatOperation`].
pub const COMDAT_OPERATION_NAME: &str = "llvm.comdat";

/// Operation trait for `llvm.comdat`.
pub trait ComdatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        COMDAT_OPERATION_NAME
    }

    /// Returns the `sym_name` attribute.
    fn sym_name(&self) -> AttributeRef<'c, 't> {
        self.attribute("sym_name").unwrap()
    }
}

mlir_op!(Comdat);

/// Constructs a new detached `llvm.comdat` operation.
pub fn comdat<'c, 't: 'c, L: Location<'c, 't>>(
    sym_name: AttributeRef<'c, 't>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedComdatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(COMDAT_OPERATION_NAME, location);
    builder = builder.add_attribute("sym_name", sym_name);
    builder = builder.add_region(body);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::comdat`")
}

/// Canonical MLIR operation name for [`ComdatSelectorOperation`].
pub const COMDAT_SELECTOR_OPERATION_NAME: &str = "llvm.comdat_selector";

/// Operation trait for `llvm.comdat_selector`.
pub trait ComdatSelectorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        COMDAT_SELECTOR_OPERATION_NAME
    }

    /// Returns the `sym_name` attribute.
    fn sym_name(&self) -> AttributeRef<'c, 't> {
        self.attribute("sym_name").unwrap()
    }

    /// Returns the `comdat` attribute.
    fn comdat(&self) -> AttributeRef<'c, 't> {
        self.attribute("comdat").unwrap()
    }
}

mlir_op!(ComdatSelector);

/// Constructs a new detached `llvm.comdat_selector` operation.
pub fn comdat_selector<'c, 't: 'c, L: Location<'c, 't>>(
    sym_name: AttributeRef<'c, 't>,
    comdat: AttributeRef<'c, 't>,
    location: L,
) -> DetachedComdatSelectorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(COMDAT_SELECTOR_OPERATION_NAME, location);
    builder = builder.add_attribute("sym_name", sym_name);
    builder = builder.add_attribute("comdat", comdat);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::comdat_selector`")
}

/// Canonical MLIR operation name for [`CondBrOperation`].
pub const COND_BR_OPERATION_NAME: &str = "llvm.cond_br";

/// Operation trait for `llvm.cond_br`.
pub trait CondBrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        COND_BR_OPERATION_NAME
    }

    /// Returns the `condition` operand.
    fn condition(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `true_destination_operands` operands.
    fn true_destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let true_destination_operand_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.cond_br`"));
        self.operand_values().skip(1).take(true_destination_operand_count as usize).collect()
    }

    /// Returns the `false_destination_operands` operands.
    fn false_destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let true_destination_operand_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.cond_br`"));
        self.operand_values().skip(1 + true_destination_operand_count as usize).collect()
    }

    /// Returns the true destination block.
    fn true_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(0).unwrap()
    }

    /// Returns the false destination block.
    fn false_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(1).unwrap()
    }

    /// Returns the optional `branch_weights` attribute.
    fn branch_weights(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("branch_weights")
    }

    /// Returns the optional `loop_annotation` attribute.
    fn loop_annotation(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("loop_annotation")
    }
}

mlir_op!(CondBr);

/// Constructs a new detached `llvm.cond_br` operation.
pub fn cond_br<
    'b,
    'c: 'b,
    't: 'c,
    V1: Value<'c, 'c, 't>,
    B1: Block<'b, 'c, 't>,
    B2: Block<'b, 'c, 't>,
    L: Location<'c, 't>,
>(
    condition: V1,
    true_destination: &B1,
    false_destination: &B2,
    true_destination_operands: &[ValueRef<'c, 'c, 't>],
    false_destination_operands: &[ValueRef<'c, 'c, 't>],
    branch_weights: Option<AttributeRef<'c, 't>>,
    loop_annotation: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedCondBrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(COND_BR_OPERATION_NAME, location);
    builder = builder.add_operand(condition);
    builder = builder.add_operands(true_destination_operands);
    builder = builder.add_operands(false_destination_operands);
    builder = builder.add_successor(true_destination);
    builder = builder.add_successor(false_destination);
    builder = builder.add_attribute(
        "operand_segment_sizes",
        context
            .dense_i32_array_attribute(&[
                1,
                true_destination_operands.len() as i32,
                false_destination_operands.len() as i32,
            ])
            .unwrap(),
    );
    if let Some(branch_weights) = branch_weights {
        builder = builder.add_attribute("branch_weights", branch_weights);
    }
    if let Some(loop_annotation) = loop_annotation {
        builder = builder.add_attribute("loop_annotation", loop_annotation);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::cond_br`")
}

/// Canonical MLIR operation name for [`DsoLocalEquivalentOperation`].
pub const DSO_LOCAL_EQUIVALENT_OPERATION_NAME: &str = "llvm.dso_local_equivalent";

/// Operation trait for `llvm.dso_local_equivalent`.
pub trait DsoLocalEquivalentOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        DSO_LOCAL_EQUIVALENT_OPERATION_NAME
    }

    /// Returns the `function_name` attribute.
    fn function_name(&self) -> AttributeRef<'c, 't> {
        self.attribute("function_name").unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(DsoLocalEquivalent);

/// Constructs a new detached `llvm.dso_local_equivalent` operation.
pub fn dso_local_equivalent<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    function_name: AttributeRef<'c, 't>,
    location: L,
) -> DetachedDsoLocalEquivalentOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(DSO_LOCAL_EQUIVALENT_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("function_name", function_name);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::dso_local_equivalent`")
}

/// Canonical MLIR operation name for [`ExtractElementOperation`].
pub const EXTRACT_ELEMENT_OPERATION_NAME: &str = "llvm.extractelement";

/// Operation trait for `llvm.extractelement`.
pub trait ExtractElementOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EXTRACT_ELEMENT_OPERATION_NAME
    }

    /// Returns the `vector` operand.
    fn vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `position` operand.
    fn position(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ExtractElement);

/// Constructs a new detached `llvm.extractelement` operation.
pub fn extractelement<'c, 't: 'c, V1: Value<'c, 'c, 't>, V2: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    vector: V1,
    position: V2,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedExtractElementOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(EXTRACT_ELEMENT_OPERATION_NAME, location);
    builder = builder.add_operand(vector);
    builder = builder.add_operand(position);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::extractelement`")
}

/// Canonical MLIR operation name for [`ExtractValueOperation`].
pub const EXTRACT_VALUE_OPERATION_NAME: &str = "llvm.extractvalue";

/// Operation trait for `llvm.extractvalue`.
pub trait ExtractValueOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EXTRACT_VALUE_OPERATION_NAME
    }

    /// Returns the `container` operand.
    fn container(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `position` attribute.
    fn position(&self) -> AttributeRef<'c, 't> {
        self.attribute("position").unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ExtractValue);

/// Constructs a new detached `llvm.extractvalue` operation.
pub fn extractvalue<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    container: V1,
    result_type: TypeRef<'c, 't>,
    position: AttributeRef<'c, 't>,
    location: L,
) -> DetachedExtractValueOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(EXTRACT_VALUE_OPERATION_NAME, location);
    builder = builder.add_operand(container);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("position", position);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::extractvalue`")
}

/// Canonical MLIR operation name for [`FcmpOperation`].
pub const FCMP_OPERATION_NAME: &str = "llvm.fcmp";

/// Operation trait for `llvm.fcmp`.
pub trait FcmpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FCMP_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `predicate` attribute.
    fn predicate(&self) -> AttributeRef<'c, 't> {
        self.attribute("predicate").unwrap()
    }

    /// Returns the optional `fastmath_flags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmath_flags")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Fcmp);

/// Constructs a new detached `llvm.fcmp` operation.
pub fn fcmp<'c, 't: 'c, V1: Value<'c, 'c, 't>, V2: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    lhs: V1,
    rhs: V2,
    result_type: TypeRef<'c, 't>,
    predicate: AttributeRef<'c, 't>,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedFcmpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FCMP_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("predicate", predicate);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmath_flags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::fcmp`")
}

/// Canonical MLIR operation name for [`FenceOperation`].
pub const FENCE_OPERATION_NAME: &str = "llvm.fence";

/// Operation trait for `llvm.fence`.
pub trait FenceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FENCE_OPERATION_NAME
    }

    /// Returns the `ordering` attribute.
    fn ordering(&self) -> AttributeRef<'c, 't> {
        self.attribute("ordering").unwrap()
    }

    /// Returns the optional `syncscope` attribute.
    fn syncscope(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("syncscope")
    }
}

mlir_op!(Fence);

/// Constructs a new detached `llvm.fence` operation.
pub fn fence<'c, 't: 'c, L: Location<'c, 't>>(
    ordering: AttributeRef<'c, 't>,
    syncscope: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedFenceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FENCE_OPERATION_NAME, location);
    builder = builder.add_attribute("ordering", ordering);
    if let Some(syncscope) = syncscope {
        builder = builder.add_attribute("syncscope", syncscope);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::fence`")
}

/// Canonical MLIR operation name for [`FreezeOperation`].
pub const FREEZE_OPERATION_NAME: &str = "llvm.freeze";

/// Operation trait for `llvm.freeze`.
pub trait FreezeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FREEZE_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Freeze);

/// Constructs a new detached `llvm.freeze` operation.
pub fn freeze<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    value: V1,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedFreezeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FREEZE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::freeze`")
}

/// Canonical MLIR operation name for [`GepOperation`].
pub const GEP_OPERATION_NAME: &str = "llvm.getelementptr";

/// Operation trait for `llvm.getelementptr`.
pub trait GepOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        GEP_OPERATION_NAME
    }

    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `dynamic_indices` operands.
    fn dynamic_indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the `raw_constant_indices` attribute.
    fn raw_constant_indices(&self) -> AttributeRef<'c, 't> {
        self.attribute("rawConstantIndices").unwrap()
    }

    /// Returns the `elem_type` attribute.
    fn elem_type(&self) -> AttributeRef<'c, 't> {
        self.attribute("elem_type").unwrap()
    }

    /// Returns the optional `no_wrap_flags` attribute.
    fn no_wrap_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("noWrapFlags")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Gep);

/// Constructs a new detached `llvm.getelementptr` operation.
pub fn getelementptr<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    base: V1,
    dynamic_indices: &[ValueRef<'c, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    raw_constant_indices: AttributeRef<'c, 't>,
    elem_type: AttributeRef<'c, 't>,
    no_wrap_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedGepOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(GEP_OPERATION_NAME, location);
    builder = builder.add_operand(base);
    builder = builder.add_operands(dynamic_indices);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("rawConstantIndices", raw_constant_indices);
    builder = builder.add_attribute("elem_type", elem_type);
    if let Some(no_wrap_flags) = no_wrap_flags {
        builder = builder.add_attribute("noWrapFlags", no_wrap_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::getelementptr`")
}

/// Canonical MLIR operation name for [`GlobalCtorsOperation`].
pub const GLOBAL_CTORS_OPERATION_NAME: &str = "llvm.mlir.global_ctors";

/// Operation trait for `llvm.mlir.global_ctors`.
pub trait GlobalCtorsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        GLOBAL_CTORS_OPERATION_NAME
    }

    /// Returns the `ctors` attribute.
    fn ctors(&self) -> AttributeRef<'c, 't> {
        self.attribute("ctors").unwrap()
    }

    /// Returns the `priorities` attribute.
    fn priorities(&self) -> AttributeRef<'c, 't> {
        self.attribute("priorities").unwrap()
    }

    /// Returns the `data` attribute.
    fn data(&self) -> AttributeRef<'c, 't> {
        self.attribute("data").unwrap()
    }
}

mlir_op!(GlobalCtors);

/// Constructs a new detached `llvm.mlir.global_ctors` operation.
pub fn global_ctors<'c, 't: 'c, L: Location<'c, 't>>(
    ctors: AttributeRef<'c, 't>,
    priorities: AttributeRef<'c, 't>,
    data: AttributeRef<'c, 't>,
    location: L,
) -> DetachedGlobalCtorsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(GLOBAL_CTORS_OPERATION_NAME, location);
    builder = builder.add_attribute("ctors", ctors);
    builder = builder.add_attribute("priorities", priorities);
    builder = builder.add_attribute("data", data);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::global_ctors`")
}

/// Canonical MLIR operation name for [`GlobalDtorsOperation`].
pub const GLOBAL_DTORS_OPERATION_NAME: &str = "llvm.mlir.global_dtors";

/// Operation trait for `llvm.mlir.global_dtors`.
pub trait GlobalDtorsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        GLOBAL_DTORS_OPERATION_NAME
    }

    /// Returns the `dtors` attribute.
    fn dtors(&self) -> AttributeRef<'c, 't> {
        self.attribute("dtors").unwrap()
    }

    /// Returns the `priorities` attribute.
    fn priorities(&self) -> AttributeRef<'c, 't> {
        self.attribute("priorities").unwrap()
    }

    /// Returns the `data` attribute.
    fn data(&self) -> AttributeRef<'c, 't> {
        self.attribute("data").unwrap()
    }
}

mlir_op!(GlobalDtors);

/// Constructs a new detached `llvm.mlir.global_dtors` operation.
pub fn global_dtors<'c, 't: 'c, L: Location<'c, 't>>(
    dtors: AttributeRef<'c, 't>,
    priorities: AttributeRef<'c, 't>,
    data: AttributeRef<'c, 't>,
    location: L,
) -> DetachedGlobalDtorsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(GLOBAL_DTORS_OPERATION_NAME, location);
    builder = builder.add_attribute("dtors", dtors);
    builder = builder.add_attribute("priorities", priorities);
    builder = builder.add_attribute("data", data);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::global_dtors`")
}

/// Canonical MLIR operation name for [`GlobalOperation`].
pub const GLOBAL_OPERATION_NAME: &str = "llvm.mlir.global";

/// Operation trait for `llvm.mlir.global`.
pub trait GlobalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        GLOBAL_OPERATION_NAME
    }

    /// Returns the `global_type` attribute.
    fn global_type(&self) -> AttributeRef<'c, 't> {
        self.attribute("global_type").unwrap()
    }

    /// Returns whether the `constant` unit attribute is present.
    fn constant(&self) -> bool {
        self.has_attribute("constant")
    }

    /// Returns the `sym_name` attribute.
    fn sym_name(&self) -> AttributeRef<'c, 't> {
        self.attribute("sym_name").unwrap()
    }

    /// Returns the `linkage` attribute.
    fn linkage(&self) -> AttributeRef<'c, 't> {
        self.attribute("linkage").unwrap()
    }

    /// Returns whether the `dso_local` unit attribute is present.
    fn dso_local(&self) -> bool {
        self.has_attribute("dso_local")
    }

    /// Returns whether the `thread_local_` unit attribute is present.
    fn thread_local_(&self) -> bool {
        self.has_attribute("thread_local_")
    }

    /// Returns whether the `externally_initialized` unit attribute is present.
    fn externally_initialized(&self) -> bool {
        self.has_attribute("externally_initialized")
    }

    /// Returns the optional `value` attribute.
    fn value(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("value")
    }

    /// Returns the optional `alignment` attribute.
    fn alignment(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("alignment")
    }

    /// Returns the optional `addr_space` attribute.
    fn addr_space(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("addr_space")
    }

    /// Returns the optional `unnamed_addr` attribute.
    fn unnamed_addr(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("unnamed_addr")
    }

    /// Returns the optional `section` attribute.
    fn section(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("section")
    }

    /// Returns the optional `comdat` attribute.
    fn comdat(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("comdat")
    }

    /// Returns the optional `dbg_exprs` attribute.
    fn dbg_exprs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("dbg_exprs")
    }

    /// Returns the optional `visibility_` attribute.
    fn visibility_(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("visibility_")
    }

    /// Returns the optional `target_specific_attrs` attribute.
    fn target_specific_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("target_specific_attrs")
    }
}

mlir_op!(Global);

/// Constructs a new detached `llvm.mlir.global` operation.
pub fn global<'c, 't: 'c, L: Location<'c, 't>>(
    global_type: AttributeRef<'c, 't>,
    constant: bool,
    sym_name: AttributeRef<'c, 't>,
    linkage: AttributeRef<'c, 't>,
    dso_local: bool,
    thread_local_: bool,
    externally_initialized: bool,
    value: Option<AttributeRef<'c, 't>>,
    alignment: Option<AttributeRef<'c, 't>>,
    addr_space: Option<AttributeRef<'c, 't>>,
    unnamed_addr: Option<AttributeRef<'c, 't>>,
    section: Option<AttributeRef<'c, 't>>,
    comdat: Option<AttributeRef<'c, 't>>,
    dbg_exprs: Option<AttributeRef<'c, 't>>,
    visibility_: Option<AttributeRef<'c, 't>>,
    target_specific_attrs: Option<AttributeRef<'c, 't>>,
    initializer: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedGlobalOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(GLOBAL_OPERATION_NAME, location);
    builder = builder.add_attribute("global_type", global_type);
    if constant {
        builder = builder.add_attribute("constant", context.unit_attribute());
    }
    builder = builder.add_attribute("sym_name", sym_name);
    builder = builder.add_attribute("linkage", linkage);
    if dso_local {
        builder = builder.add_attribute("dso_local", context.unit_attribute());
    }
    if thread_local_ {
        builder = builder.add_attribute("thread_local_", context.unit_attribute());
    }
    if externally_initialized {
        builder = builder.add_attribute("externally_initialized", context.unit_attribute());
    }
    if let Some(value) = value {
        builder = builder.add_attribute("value", value);
    }
    if let Some(alignment) = alignment {
        builder = builder.add_attribute("alignment", alignment);
    }
    if let Some(addr_space) = addr_space {
        builder = builder.add_attribute("addr_space", addr_space);
    }
    if let Some(unnamed_addr) = unnamed_addr {
        builder = builder.add_attribute("unnamed_addr", unnamed_addr);
    }
    if let Some(section) = section {
        builder = builder.add_attribute("section", section);
    }
    if let Some(comdat) = comdat {
        builder = builder.add_attribute("comdat", comdat);
    }
    if let Some(dbg_exprs) = dbg_exprs {
        builder = builder.add_attribute("dbg_exprs", dbg_exprs);
    }
    if let Some(visibility_) = visibility_ {
        builder = builder.add_attribute("visibility_", visibility_);
    }
    if let Some(target_specific_attrs) = target_specific_attrs {
        builder = builder.add_attribute("target_specific_attrs", target_specific_attrs);
    }
    builder = builder.add_region(initializer);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::global`")
}

/// Canonical MLIR operation name for [`IcmpOperation`].
pub const ICMP_OPERATION_NAME: &str = "llvm.icmp";

/// Operation trait for `llvm.icmp`.
pub trait IcmpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ICMP_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `predicate` attribute.
    fn predicate(&self) -> AttributeRef<'c, 't> {
        self.attribute("predicate").unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Icmp);

/// Constructs a new detached `llvm.icmp` operation.
pub fn icmp<'c, 't: 'c, V1: Value<'c, 'c, 't>, V2: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    lhs: V1,
    rhs: V2,
    result_type: TypeRef<'c, 't>,
    predicate: AttributeRef<'c, 't>,
    location: L,
) -> DetachedIcmpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ICMP_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("predicate", predicate);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::icmp`")
}

/// Canonical MLIR operation name for [`IfuncOperation`].
pub const IFUNC_OPERATION_NAME: &str = "llvm.mlir.ifunc";

/// Operation trait for `llvm.mlir.ifunc`.
pub trait IfuncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        IFUNC_OPERATION_NAME
    }

    /// Returns the `sym_name` attribute.
    fn sym_name(&self) -> AttributeRef<'c, 't> {
        self.attribute("sym_name").unwrap()
    }

    /// Returns the `i_func_type` attribute.
    fn i_func_type(&self) -> AttributeRef<'c, 't> {
        self.attribute("i_func_type").unwrap()
    }

    /// Returns the `resolver` attribute.
    fn resolver(&self) -> AttributeRef<'c, 't> {
        self.attribute("resolver").unwrap()
    }

    /// Returns the `resolver_type` attribute.
    fn resolver_type(&self) -> AttributeRef<'c, 't> {
        self.attribute("resolver_type").unwrap()
    }

    /// Returns the `linkage` attribute.
    fn linkage(&self) -> AttributeRef<'c, 't> {
        self.attribute("linkage").unwrap()
    }

    /// Returns whether the `dso_local` unit attribute is present.
    fn dso_local(&self) -> bool {
        self.has_attribute("dso_local")
    }

    /// Returns the optional `address_space` attribute.
    fn address_space(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("address_space")
    }

    /// Returns the optional `unnamed_addr` attribute.
    fn unnamed_addr(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("unnamed_addr")
    }

    /// Returns the optional `visibility_` attribute.
    fn visibility_(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("visibility_")
    }
}

mlir_op!(Ifunc);

/// Constructs a new detached `llvm.mlir.ifunc` operation.
pub fn ifunc<'c, 't: 'c, L: Location<'c, 't>>(
    sym_name: AttributeRef<'c, 't>,
    i_func_type: AttributeRef<'c, 't>,
    resolver: AttributeRef<'c, 't>,
    resolver_type: AttributeRef<'c, 't>,
    linkage: AttributeRef<'c, 't>,
    dso_local: bool,
    address_space: Option<AttributeRef<'c, 't>>,
    unnamed_addr: Option<AttributeRef<'c, 't>>,
    visibility_: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedIfuncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(IFUNC_OPERATION_NAME, location);
    builder = builder.add_attribute("sym_name", sym_name);
    builder = builder.add_attribute("i_func_type", i_func_type);
    builder = builder.add_attribute("resolver", resolver);
    builder = builder.add_attribute("resolver_type", resolver_type);
    builder = builder.add_attribute("linkage", linkage);
    if dso_local {
        builder = builder.add_attribute("dso_local", context.unit_attribute());
    }
    if let Some(address_space) = address_space {
        builder = builder.add_attribute("address_space", address_space);
    }
    if let Some(unnamed_addr) = unnamed_addr {
        builder = builder.add_attribute("unnamed_addr", unnamed_addr);
    }
    if let Some(visibility_) = visibility_ {
        builder = builder.add_attribute("visibility_", visibility_);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::ifunc`")
}

/// Canonical MLIR operation name for [`IndirectBrOperation`].
pub const INDIRECT_BR_OPERATION_NAME: &str = "llvm.indirectbr";

/// Operation trait for `llvm.indirectbr`.
pub trait IndirectBrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INDIRECT_BR_OPERATION_NAME
    }

    /// Returns the `address` operand.
    fn address(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `successor_operands` operands.
    fn successor_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the destination blocks.
    fn destinations(&self) -> Vec<BlockRef<'o, 'c, 't>> {
        self.successors().collect()
    }

    /// Returns the `indbr_operand_segments` attribute.
    fn indbr_operand_segments(&self) -> AttributeRef<'c, 't> {
        self.attribute("indbr_operand_segments").unwrap()
    }
}

mlir_op!(IndirectBr);

/// Constructs a new detached `llvm.indirectbr` operation.
pub fn indirectbr<'b, 'c: 'b, 't: 'c, V1: Value<'c, 'c, 't>, B: Block<'b, 'c, 't>, L: Location<'c, 't>>(
    address: V1,
    destinations: &[&B],
    successor_operands: &[ValueRef<'c, 'c, 't>],
    indbr_operand_segments: AttributeRef<'c, 't>,
    location: L,
) -> DetachedIndirectBrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INDIRECT_BR_OPERATION_NAME, location);
    builder = builder.add_operand(address);
    builder = builder.add_operands(successor_operands);
    builder = builder.add_successors(destinations);
    builder = builder.add_attribute("indbr_operand_segments", indbr_operand_segments);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::indirectbr`")
}

/// Canonical MLIR operation name for [`InlineAsmOperation`].
pub const INLINE_ASM_OPERATION_NAME: &str = "llvm.inline_asm";

/// Operation trait for `llvm.inline_asm`.
pub trait InlineAsmOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INLINE_ASM_OPERATION_NAME
    }

    /// Returns the `operands` operands.
    fn operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(0).collect()
    }

    /// Returns the `asm_string` attribute.
    fn asm_string(&self) -> AttributeRef<'c, 't> {
        self.attribute("asm_string").unwrap()
    }

    /// Returns the `constraints` attribute.
    fn constraints(&self) -> AttributeRef<'c, 't> {
        self.attribute("constraints").unwrap()
    }

    /// Returns whether the `has_side_effects` unit attribute is present.
    fn has_side_effects(&self) -> bool {
        self.has_attribute("has_side_effects")
    }

    /// Returns whether the `is_align_stack` unit attribute is present.
    fn is_align_stack(&self) -> bool {
        self.has_attribute("is_align_stack")
    }

    /// Returns the optional `tail_call_kind` attribute.
    fn tail_call_kind(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("tail_call_kind")
    }

    /// Returns the optional `asm_dialect` attribute.
    fn asm_dialect(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("asm_dialect")
    }

    /// Returns the optional `operand_attrs` attribute.
    fn operand_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("operand_attrs")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(InlineAsm);

/// Constructs a new detached `llvm.inline_asm` operation.
pub fn inline_asm<'c, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'c, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    asm_string: AttributeRef<'c, 't>,
    constraints: AttributeRef<'c, 't>,
    has_side_effects: bool,
    is_align_stack: bool,
    tail_call_kind: Option<AttributeRef<'c, 't>>,
    asm_dialect: Option<AttributeRef<'c, 't>>,
    operand_attrs: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedInlineAsmOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INLINE_ASM_OPERATION_NAME, location);
    builder = builder.add_operands(operands);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("asm_string", asm_string);
    builder = builder.add_attribute("constraints", constraints);
    if has_side_effects {
        builder = builder.add_attribute("has_side_effects", context.unit_attribute());
    }
    if is_align_stack {
        builder = builder.add_attribute("is_align_stack", context.unit_attribute());
    }
    if let Some(tail_call_kind) = tail_call_kind {
        builder = builder.add_attribute("tail_call_kind", tail_call_kind);
    }
    if let Some(asm_dialect) = asm_dialect {
        builder = builder.add_attribute("asm_dialect", asm_dialect);
    }
    if let Some(operand_attrs) = operand_attrs {
        builder = builder.add_attribute("operand_attrs", operand_attrs);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::inline_asm`")
}

/// Canonical MLIR operation name for [`InsertElementOperation`].
pub const INSERT_ELEMENT_OPERATION_NAME: &str = "llvm.insertelement";

/// Operation trait for `llvm.insertelement`.
pub trait InsertElementOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INSERT_ELEMENT_OPERATION_NAME
    }

    /// Returns the `vector` operand.
    fn vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `position` operand.
    fn position(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(InsertElement);

/// Constructs a new detached `llvm.insertelement` operation.
pub fn insert_element<
    'c,
    't: 'c,
    V1: Value<'c, 'c, 't>,
    V2: Value<'c, 'c, 't>,
    V3: Value<'c, 'c, 't>,
    L: Location<'c, 't>,
>(
    vector: V1,
    value: V2,
    position: V3,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedInsertElementOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INSERT_ELEMENT_OPERATION_NAME, location);
    builder = builder.add_operand(vector);
    builder = builder.add_operand(value);
    builder = builder.add_operand(position);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::insert_element`")
}

/// Canonical MLIR operation name for [`InsertValueOperation`].
pub const INSERT_VALUE_OPERATION_NAME: &str = "llvm.insertvalue";

/// Operation trait for `llvm.insertvalue`.
pub trait InsertValueOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INSERT_VALUE_OPERATION_NAME
    }

    /// Returns the `container` operand.
    fn container(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `position` attribute.
    fn position(&self) -> AttributeRef<'c, 't> {
        self.attribute("position").unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(InsertValue);

/// Constructs a new detached `llvm.insertvalue` operation.
pub fn insert_value<'c, 't: 'c, V1: Value<'c, 'c, 't>, V2: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    container: V1,
    value: V2,
    result_type: TypeRef<'c, 't>,
    position: AttributeRef<'c, 't>,
    location: L,
) -> DetachedInsertValueOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INSERT_VALUE_OPERATION_NAME, location);
    builder = builder.add_operand(container);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("position", position);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::insert_value`")
}

/// Canonical MLIR operation name for [`InvokeOperation`].
pub const INVOKE_OPERATION_NAME: &str = "llvm.invoke";

/// Operation trait for `llvm.invoke`.
pub trait InvokeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INVOKE_OPERATION_NAME
    }

    /// Returns the `callee_operands` operands.
    fn callee_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.invoke`"));
        self.operand_values().take(segment_sizes[0] as usize).collect()
    }

    /// Returns the `normal_destination_operands` operands.
    fn normal_destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.invoke`"));
        self.operand_values().skip(segment_sizes[0] as usize).take(segment_sizes[1] as usize).collect()
    }

    /// Returns the `unwind_destination_operands` operands.
    fn unwind_destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.invoke`"));
        self.operand_values()
            .skip((segment_sizes[0] + segment_sizes[1]) as usize)
            .take(segment_sizes[2] as usize)
            .collect()
    }

    /// Returns the `op_bundle_operands` operands.
    fn op_bundle_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.invoke`"));
        self.operand_values()
            .skip((segment_sizes[0] + segment_sizes[1] + segment_sizes[2]) as usize)
            .collect()
    }

    /// Returns the normal destination block.
    fn normal_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(0).unwrap()
    }

    /// Returns the unwind destination block.
    fn unwind_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(1).unwrap()
    }

    /// Returns the optional `var_callee_type` attribute.
    fn var_callee_type(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("var_callee_type")
    }

    /// Returns the optional `callee` attribute.
    fn callee(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("callee")
    }

    /// Returns the optional `arg_attrs` attribute.
    fn arg_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("arg_attrs")
    }

    /// Returns the optional `res_attrs` attribute.
    fn res_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("res_attrs")
    }

    /// Returns the optional `branch_weights` attribute.
    fn branch_weights(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("branch_weights")
    }

    /// Returns the optional `calling_convention` attribute.
    fn calling_convention(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("CConv")
    }

    /// Returns the optional `op_bundle_sizes` attribute.
    fn op_bundle_sizes(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("op_bundle_sizes")
    }

    /// Returns the optional `op_bundle_tags` attribute.
    fn op_bundle_tags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("op_bundle_tags")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Invoke);

/// Constructs a new detached `llvm.invoke` operation.
pub fn invoke<'b, 'c: 'b, 't: 'c, B1: Block<'b, 'c, 't>, B2: Block<'b, 'c, 't>, L: Location<'c, 't>>(
    callee_operands: &[ValueRef<'c, 'c, 't>],
    normal_destination: &B1,
    normal_destination_operands: &[ValueRef<'c, 'c, 't>],
    unwind_destination: &B2,
    unwind_destination_operands: &[ValueRef<'c, 'c, 't>],
    op_bundle_operands: &[ValueRef<'c, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    var_callee_type: Option<AttributeRef<'c, 't>>,
    callee: Option<AttributeRef<'c, 't>>,
    arg_attrs: Option<AttributeRef<'c, 't>>,
    res_attrs: Option<AttributeRef<'c, 't>>,
    branch_weights: Option<AttributeRef<'c, 't>>,
    calling_convention: Option<AttributeRef<'c, 't>>,
    op_bundle_sizes: Option<AttributeRef<'c, 't>>,
    op_bundle_tags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedInvokeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INVOKE_OPERATION_NAME, location);
    builder = builder.add_operands(callee_operands);
    builder = builder.add_operands(normal_destination_operands);
    builder = builder.add_operands(unwind_destination_operands);
    builder = builder.add_operands(op_bundle_operands);
    builder = builder.add_successor(normal_destination);
    builder = builder.add_successor(unwind_destination);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute(
        "operand_segment_sizes",
        context
            .dense_i32_array_attribute(&[
                callee_operands.len() as i32,
                normal_destination_operands.len() as i32,
                unwind_destination_operands.len() as i32,
                op_bundle_operands.len() as i32,
            ])
            .unwrap(),
    );
    builder = builder.add_attribute(
        "op_bundle_sizes",
        op_bundle_sizes.unwrap_or_else(|| context.dense_i32_array_attribute(&[]).unwrap().as_ref()),
    );
    if let Some(var_callee_type) = var_callee_type {
        builder = builder.add_attribute("var_callee_type", var_callee_type);
    }
    if let Some(callee) = callee {
        builder = builder.add_attribute("callee", callee);
    }
    if let Some(arg_attrs) = arg_attrs {
        builder = builder.add_attribute("arg_attrs", arg_attrs);
    }
    if let Some(res_attrs) = res_attrs {
        builder = builder.add_attribute("res_attrs", res_attrs);
    }
    if let Some(branch_weights) = branch_weights {
        builder = builder.add_attribute("branch_weights", branch_weights);
    }
    if let Some(calling_convention) = calling_convention {
        builder = builder.add_attribute("CConv", calling_convention);
    }
    if let Some(op_bundle_tags) = op_bundle_tags {
        builder = builder.add_attribute("op_bundle_tags", op_bundle_tags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::invoke`")
}

/// Canonical MLIR operation name for [`LlvmFuncOperation`].
pub const LLVM_FUNC_OPERATION_NAME: &str = "llvm.func";

/// Operation trait for `llvm.func`.
pub trait LlvmFuncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LLVM_FUNC_OPERATION_NAME
    }

    /// Returns the `sym_name` attribute.
    fn sym_name(&self) -> AttributeRef<'c, 't> {
        self.attribute("sym_name").unwrap()
    }

    /// Returns the optional `sym_visibility` attribute.
    fn sym_visibility(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("sym_visibility")
    }

    /// Returns the `function_type` attribute.
    fn function_type(&self) -> AttributeRef<'c, 't> {
        self.attribute("function_type").unwrap()
    }

    /// Returns the optional `linkage` attribute.
    fn linkage(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("linkage")
    }
}

mlir_op!(LlvmFunc);

/// Constructs a new detached `llvm.func` operation.
pub fn func<'c, 't: 'c, L: Location<'c, 't>>(
    sym_name: AttributeRef<'c, 't>,
    sym_visibility: Option<AttributeRef<'c, 't>>,
    function_type: AttributeRef<'c, 't>,
    linkage: Option<AttributeRef<'c, 't>>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedLlvmFuncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LLVM_FUNC_OPERATION_NAME, location);
    builder = builder.add_attribute("sym_name", sym_name);
    if let Some(sym_visibility) = sym_visibility {
        builder = builder.add_attribute("sym_visibility", sym_visibility);
    }
    builder = builder.add_attribute("function_type", function_type);
    if let Some(linkage) = linkage {
        builder = builder.add_attribute("linkage", linkage);
    }
    builder = builder.add_region(body);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::func`")
}

/// Canonical MLIR operation name for [`LandingpadOperation`].
pub const LANDINGPAD_OPERATION_NAME: &str = "llvm.landingpad";

/// Operation trait for `llvm.landingpad`.
pub trait LandingpadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LANDINGPAD_OPERATION_NAME
    }

    /// Returns the `clauses` operands.
    fn clauses(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(0).collect()
    }

    /// Returns whether the `cleanup` unit attribute is present.
    fn cleanup(&self) -> bool {
        self.has_attribute("cleanup")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Landingpad);

/// Constructs a new detached `llvm.landingpad` operation.
pub fn landingpad<'c, 't: 'c, L: Location<'c, 't>>(
    clauses: &[ValueRef<'c, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    cleanup: bool,
    location: L,
) -> DetachedLandingpadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LANDINGPAD_OPERATION_NAME, location);
    builder = builder.add_operands(clauses);
    builder = builder.add_result(result_type);
    if cleanup {
        builder = builder.add_attribute("cleanup", context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::landingpad`")
}

/// Canonical MLIR operation name for [`LinkerOptionsOperation`].
pub const LINKER_OPTIONS_OPERATION_NAME: &str = "llvm.linker_options";

/// Operation trait for `llvm.linker_options`.
pub trait LinkerOptionsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LINKER_OPTIONS_OPERATION_NAME
    }

    /// Returns the `options` attribute.
    fn options(&self) -> AttributeRef<'c, 't> {
        self.attribute("options").unwrap()
    }
}

mlir_op!(LinkerOptions);

/// Constructs a new detached `llvm.linker_options` operation.
pub fn linker_options<'c, 't: 'c, L: Location<'c, 't>>(
    options: AttributeRef<'c, 't>,
    location: L,
) -> DetachedLinkerOptionsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LINKER_OPTIONS_OPERATION_NAME, location);
    builder = builder.add_attribute("options", options);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::linker_options`")
}

/// Canonical MLIR operation name for [`ModuleFlagsOperation`].
pub const MODULE_FLAGS_OPERATION_NAME: &str = "llvm.module_flags";

/// Operation trait for `llvm.module_flags`.
pub trait ModuleFlagsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MODULE_FLAGS_OPERATION_NAME
    }

    /// Returns the `flags` attribute.
    fn flags(&self) -> AttributeRef<'c, 't> {
        self.attribute("flags").unwrap()
    }
}

mlir_op!(ModuleFlags);

/// Constructs a new detached `llvm.module_flags` operation.
pub fn module_flags<'c, 't: 'c, L: Location<'c, 't>>(
    flags: AttributeRef<'c, 't>,
    location: L,
) -> DetachedModuleFlagsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MODULE_FLAGS_OPERATION_NAME, location);
    builder = builder.add_attribute("flags", flags);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::module_flags`")
}

/// Canonical MLIR operation name for [`NamedMetadataOperation`].
pub const NAMED_METADATA_OPERATION_NAME: &str = "llvm.named_metadata";

/// Operation trait for `llvm.named_metadata`.
pub trait NamedMetadataOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        NAMED_METADATA_OPERATION_NAME
    }

    /// Returns the `metadata_name` attribute.
    fn metadata_name(&self) -> AttributeRef<'c, 't> {
        self.attribute("metadata_name").unwrap()
    }

    /// Returns the `nodes` attribute.
    fn nodes(&self) -> AttributeRef<'c, 't> {
        self.attribute("nodes").unwrap()
    }
}

mlir_op!(NamedMetadata);

/// Constructs a new detached `llvm.named_metadata` operation.
pub fn named_metadata<'c, 't: 'c, L: Location<'c, 't>>(
    metadata_name: AttributeRef<'c, 't>,
    nodes: AttributeRef<'c, 't>,
    location: L,
) -> DetachedNamedMetadataOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(NAMED_METADATA_OPERATION_NAME, location);
    builder = builder.add_attribute("metadata_name", metadata_name);
    builder = builder.add_attribute("nodes", nodes);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::named_metadata`")
}

/// Canonical MLIR operation name for [`NoneTokenOperation`].
pub const NONE_TOKEN_OPERATION_NAME: &str = "llvm.mlir.none";

/// Operation trait for `llvm.mlir.none`.
pub trait NoneTokenOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        NONE_TOKEN_OPERATION_NAME
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(NoneToken);

/// Constructs a new detached `llvm.mlir.none` operation.
pub fn none<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedNoneTokenOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(NONE_TOKEN_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::none`")
}

/// Canonical MLIR operation name for [`ResumeOperation`].
pub const RESUME_OPERATION_NAME: &str = "llvm.resume";

/// Operation trait for `llvm.resume`.
pub trait ResumeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        RESUME_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(Resume);

/// Constructs a new detached `llvm.resume` operation.
pub fn resume<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    value: V1,
    location: L,
) -> DetachedResumeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(RESUME_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::resume`")
}

/// Canonical MLIR operation name for [`ShuffleVectorOperation`].
pub const SHUFFLE_VECTOR_OPERATION_NAME: &str = "llvm.shufflevector";

/// Operation trait for `llvm.shufflevector`.
pub trait ShuffleVectorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SHUFFLE_VECTOR_OPERATION_NAME
    }

    /// Returns the `first_vector` operand.
    fn first_vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `second_vector` operand.
    fn second_vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` attribute.
    fn mask(&self) -> AttributeRef<'c, 't> {
        self.attribute("mask").unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ShuffleVector);

/// Constructs a new detached `llvm.shufflevector` operation.
pub fn shufflevector<'c, 't: 'c, V1: Value<'c, 'c, 't>, V2: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    first_vector: V1,
    second_vector: V2,
    result_type: TypeRef<'c, 't>,
    mask: AttributeRef<'c, 't>,
    location: L,
) -> DetachedShuffleVectorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SHUFFLE_VECTOR_OPERATION_NAME, location);
    builder = builder.add_operand(first_vector);
    builder = builder.add_operand(second_vector);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("mask", mask);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::shufflevector`")
}

/// Canonical MLIR operation name for [`SwitchOperation`].
pub const SWITCH_OPERATION_NAME: &str = "llvm.switch";

/// Operation trait for `llvm.switch`.
pub trait SwitchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SWITCH_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `default_operands` operands.
    fn default_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let default_operand_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.switch`"));
        self.operand_values().skip(1).take(default_operand_count as usize).collect()
    }

    /// Returns the `case_operands` operands.
    fn case_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let default_operand_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.switch`"));
        self.operand_values().skip(1 + default_operand_count as usize).collect()
    }

    /// Returns the default destination block.
    fn default_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(0).unwrap()
    }

    /// Returns the case destination blocks.
    fn case_destinations(&self) -> Vec<BlockRef<'o, 'c, 't>> {
        self.successors().skip(1).collect()
    }

    /// Returns the optional `case_values` attribute.
    fn case_values(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("case_values")
    }

    /// Returns the `case_operand_segments` attribute.
    fn case_operand_segments(&self) -> AttributeRef<'c, 't> {
        self.attribute("case_operand_segments").unwrap()
    }

    /// Returns the optional `branch_weights` attribute.
    fn branch_weights(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("branch_weights")
    }
}

mlir_op!(Switch);

/// Constructs a new detached `llvm.switch` operation.
pub fn switch<'b, 'c: 'b, 't: 'c, V1: Value<'c, 'c, 't>, B: Block<'b, 'c, 't>, L: Location<'c, 't>>(
    value: V1,
    default_destination: &B,
    default_operands: &[ValueRef<'c, 'c, 't>],
    case_destinations: &[&B],
    case_operands: &[ValueRef<'c, 'c, 't>],
    case_values: Option<AttributeRef<'c, 't>>,
    case_operand_segments: AttributeRef<'c, 't>,
    branch_weights: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedSwitchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SWITCH_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operands(default_operands);
    builder = builder.add_operands(case_operands);
    builder = builder.add_successor(default_destination);
    builder = builder.add_successors(case_destinations);
    if let Some(case_values) = case_values {
        builder = builder.add_attribute("case_values", case_values);
    }
    builder = builder.add_attribute("case_operand_segments", case_operand_segments);
    builder = builder.add_attribute(
        "operand_segment_sizes",
        context
            .dense_i32_array_attribute(&[1, default_operands.len() as i32, case_operands.len() as i32])
            .unwrap(),
    );
    if let Some(branch_weights) = branch_weights {
        builder = builder.add_attribute("branch_weights", branch_weights);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::switch`")
}

/// Canonical MLIR operation name for [`VaArgOperation`].
pub const VA_ARG_OPERATION_NAME: &str = "llvm.va_arg";

/// Operation trait for `llvm.va_arg`.
pub trait VaArgOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VA_ARG_OPERATION_NAME
    }

    /// Returns the `argument` operand.
    fn argument(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VaArg);

/// Constructs a new detached `llvm.va_arg` operation.
pub fn va_arg<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    argument: V1,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedVaArgOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VA_ARG_OPERATION_NAME, location);
    builder = builder.add_operand(argument);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::va_arg`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::llvm::Linkage;
    use crate::{Block, BlockRef, Context, Operation, Region, Type, TypeRef, ValueRef, dialects::func};

    use super::super::core::{address_of, constant, r#return as llvm_return};
    use super::*;

    #[test]
    fn test_alias() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation(global(
            context.type_attribute(i32_type).as_ref(),
            false,
            context.string_attribute("target").as_ref(),
            context.llvm_linkage_attribute(Linkage::External).as_ref(),
            false,
            false,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            context.region(),
            location,
        ));
        let mut initializer = context.region();
        let mut block = context.block_with_no_arguments();
        let address = block.append_operation(address_of("target", pointer_type, location));
        block.append_operation(llvm_return(Some(address.result(0).unwrap().into()), location));
        initializer.append_block(block);
        module.body().append_operation({
            let op = alias(
                context.type_attribute(pointer_type).as_ref(),
                context.string_attribute("target_alias").as_ref(),
                context.llvm_linkage_attribute(Linkage::Internal).as_ref(),
                false,
                false,
                None,
                None,
                initializer,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.mlir.alias");
            assert_eq!(op.alias_type(), context.type_attribute(pointer_type).as_ref());
            assert_eq!(op.sym_name(), context.string_attribute("target_alias").as_ref());
            assert_eq!(op.linkage(), context.llvm_linkage_attribute(Linkage::Internal).as_ref());
            assert_eq!(op.regions().count(), 1);
            op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.mlir.global external @target() {addr_space = 0 : i32} : i32
                  llvm.mlir.alias internal @target_alias : !llvm.ptr {
                    %0 = llvm.mlir.addressof @target : !llvm.ptr
                    llvm.return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_cmpxchg() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        let result_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false);
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (i32_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let op = cmpxchg(
                block.argument(0).unwrap(),
                block.argument(1).unwrap(),
                block.argument(2).unwrap(),
                result_type.as_ref(),
                context.integer_attribute(context.signless_integer_type(64), 4).as_ref(),
                context.integer_attribute(context.signless_integer_type(64), 2).as_ref(),
                Some(context.string_attribute("singlethread").as_ref()),
                Some(context.integer_attribute(context.signless_integer_type(64), 4).as_ref()),
                true,
                true,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.cmpxchg");
            assert_eq!(op.pointer(), block.argument(0).unwrap());
            assert_eq!(op.compare_value(), block.argument(1).unwrap());
            assert_eq!(op.new_value(), block.argument(2).unwrap());
            assert_eq!(op.output_type(), result_type);
            assert!(op.weak());
            assert!(op.is_volatile());
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_cmpxchg_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i32_type.into(), i32_type.into()],
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
                  func.func @llvm_cmpxchg_test(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = llvm.cmpxchg weak volatile %arg0, %arg1, %arg2 syncscope(\"singlethread\") acquire monotonic {alignment = 4 : i64} : !llvm.ptr, i32
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_atomicrmw() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let op = atomicrmw(
                block.argument(0).unwrap(),
                block.argument(1).unwrap(),
                i32_type.as_ref(),
                context.integer_attribute(context.signless_integer_type(64), 1).as_ref(),
                context.integer_attribute(context.signless_integer_type(64), 2).as_ref(),
                None,
                Some(context.integer_attribute(context.signless_integer_type(64), 4).as_ref()),
                true,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.atomicrmw");
            assert_eq!(op.pointer(), block.argument(0).unwrap());
            assert_eq!(op.value(), block.argument(1).unwrap());
            assert_eq!(op.output_type(), i32_type);
            assert!(op.is_volatile());
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_atomicrmw_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i32_type.into()],
                    results: vec![i32_type.into()],
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
                  func.func @llvm_atomicrmw_test(%arg0: !llvm.ptr, %arg1: i32) -> i32 {
                    %0 = llvm.atomicrmw volatile add %arg0, %arg1 monotonic {alignment = 4 : i64} : !llvm.ptr, i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_blockaddress() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        context.load_dialect(DialectHandle::llvm());
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let tag = context.parse_attribute("#llvm.blocktag<id = 1>").unwrap();
            block.append_operation(blocktag(tag, location));
            block.append_operation(llvm_return(None, location));
            super::func(
                context.string_attribute("target").as_ref(),
                None,
                context
                    .type_attribute(context.llvm_function_type(context.llvm_void_type(), &[] as &[TypeRef], false))
                    .as_ref(),
                Some(context.llvm_linkage_attribute(Linkage::Internal).as_ref()),
                block.into(),
                location,
            )
        });
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let block_addr = context.parse_attribute("#llvm.blockaddress<function = @target, tag = <id = 1>>").unwrap();
            let op = blockaddress(pointer_type.as_ref(), block_addr, location);
            assert_eq!(op.operation_name(), "llvm.blockaddress");
            assert_eq!(op.block_addr(), block_addr);
            assert_eq!(op.output_type(), pointer_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_blockaddress_test",
                func::FuncAttributes { arguments: vec![], results: vec![pointer_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.func internal @target() {
                    llvm.blocktag <id = 1>
                    llvm.return
                  }
                  func.func @llvm_blockaddress_test() -> !llvm.ptr {
                    %0 = llvm.blockaddress <function = @target, tag = <id = 1>> : !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_blocktag() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        context.load_dialect(DialectHandle::llvm());
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let tag = context.parse_attribute("#llvm.blocktag<id = 1>").unwrap();
            let op = blocktag(tag, location);
            assert_eq!(op.operation_name(), "llvm.blocktag");
            assert_eq!(op.tag(), tag);
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<ValueRef, _>(&[], location));
            func::func(
                "llvm_blocktag_test",
                func::FuncAttributes { arguments: vec![], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_blocktag_test() {
                    llvm.blocktag <id = 1>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_call_intrinsic() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let op = call_intrinsic(
                &[block.argument(0).unwrap().into()],
                &[],
                f32_type.as_ref(),
                context.string_attribute("llvm.sqrt").as_ref(),
                None,
                None,
                None,
                None,
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.call_intrinsic");
            assert_eq!(op.arguments(), vec![block.argument(0).unwrap()]);
            assert!(op.op_bundle_operands().is_empty());
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_call_intrinsic_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
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
                  func.func @llvm_call_intrinsic_test(%arg0: f32) -> f32 {
                    %0 = llvm.call_intrinsic \"llvm.sqrt\"(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_call() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let function_type = context.llvm_function_type(i32_type, &[i32_type], false);
        module.body().append_operation(super::func(
            context.string_attribute("callee").as_ref(),
            None,
            context.type_attribute(function_type).as_ref(),
            Some(context.llvm_linkage_attribute(Linkage::External).as_ref()),
            context.region(),
            location,
        ));
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let op = call(
                &[block.argument(0).unwrap().into()],
                &[],
                i32_type.as_ref(),
                None,
                Some(context.flat_symbol_ref_attribute("callee").as_ref()),
                None,
                None,
                None,
                None,
                None,
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.call");
            assert_eq!(op.callee_operands(), vec![block.argument(0).unwrap()]);
            assert!(op.op_bundle_operands().is_empty());
            assert_eq!(op.output_type(), i32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_call_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
                    results: vec![i32_type.into()],
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
                  llvm.func @callee(i32) -> i32
                  func.func @llvm_call_test(%arg0: i32) -> i32 {
                    %0 = llvm.call @callee(%arg0) : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_comdat() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let mut body = context.region();
        let mut block = context.block_with_no_arguments();
        let selector = comdat_selector(
            context.string_attribute("any").as_ref(),
            context.integer_attribute(context.signless_integer_type(64), 0).as_ref(),
            location,
        );
        assert_eq!(selector.operation_name(), "llvm.comdat_selector");
        assert_eq!(selector.sym_name(), context.string_attribute("any").as_ref());
        block.append_operation(selector);
        body.append_block(block);
        module.body().append_operation({
            let op = comdat(context.string_attribute("__llvm_comdat").as_ref(), body, location);
            assert_eq!(op.operation_name(), "llvm.comdat");
            assert_eq!(op.sym_name(), context.string_attribute("__llvm_comdat").as_ref());
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 0);
            assert_eq!(op.regions().count(), 1);
            op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.comdat @__llvm_comdat {
                    llvm.comdat_selector @any any
                  }
                }
            "},
        );
    }

    #[test]
    fn test_comdat_selector() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let mut body = context.region();
        let mut block = context.block_with_no_arguments();
        let comdat_kind = context.integer_attribute(context.signless_integer_type(64), 0);
        let selector = comdat_selector(context.string_attribute("any").as_ref(), comdat_kind.as_ref(), location);
        assert_eq!(selector.operation_name(), "llvm.comdat_selector");
        assert_eq!(selector.sym_name(), context.string_attribute("any").as_ref());
        assert_eq!(selector.comdat(), comdat_kind.as_ref());
        block.append_operation(selector);
        body.append_block(block);
        module
            .body()
            .append_operation(comdat(context.string_attribute("__llvm_comdat").as_ref(), body, location));
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.comdat @__llvm_comdat {
                    llvm.comdat_selector @any any
                  }
                }
            "},
        );
    }

    #[test]
    fn test_cond_br() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut entry_block = context.block(&[
                (i1_type.as_ref(), location),
                (i32_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let mut true_block = context.block(&[(i32_type.as_ref(), location)]);
            let mut false_block = context.block(&[(i32_type.as_ref(), location)]);
            let condition = entry_block.argument(0).unwrap();
            let true_value = entry_block.argument(1).unwrap();
            let false_value = entry_block.argument(2).unwrap();
            let op = cond_br(
                condition,
                &true_block,
                &false_block,
                &[true_value.into()],
                &[false_value.into()],
                Some(context.dense_i32_array_attribute(&[13, 21]).unwrap().as_ref()),
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.cond_br");
            assert_eq!(op.condition(), condition);
            assert_eq!(op.true_destination(), BlockRef::from(&true_block));
            assert_eq!(op.false_destination(), BlockRef::from(&false_block));
            assert_eq!(op.true_destination_operands(), vec![true_value]);
            assert_eq!(op.false_destination_operands(), vec![false_value]);
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 0);
            assert_eq!(op.successors().count(), 2);
            entry_block.append_operation(op);
            true_block.append_operation(func::r#return(&[true_block.argument(0).unwrap()], location));
            false_block.append_operation(func::r#return(&[false_block.argument(0).unwrap()], location));
            let mut region = context.region();
            region.append_block(entry_block);
            region.append_block(true_block);
            region.append_block(false_block);
            func::func(
                "llvm_cond_br_test",
                func::FuncAttributes {
                    arguments: vec![i1_type.into(), i32_type.into(), i32_type.into()],
                    results: vec![i32_type.into()],
                    ..Default::default()
                },
                region,
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_cond_br_test(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
                    llvm.cond_br %arg0 weights([13, 21]), ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
                  ^bb1(%0: i32):  // pred: ^bb0
                    return %0 : i32
                  ^bb2(%1: i32):  // pred: ^bb0
                    return %1 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_dso_local_equivalent() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let function_type = context.llvm_function_type(context.llvm_void_type(), &[] as &[TypeRef], false);
        module.body().append_operation(super::func(
            context.string_attribute("callee").as_ref(),
            None,
            context.type_attribute(function_type).as_ref(),
            Some(context.llvm_linkage_attribute(Linkage::External).as_ref()),
            context.region(),
            location,
        ));
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = dso_local_equivalent(
                pointer_type.as_ref(),
                context.flat_symbol_ref_attribute("callee").as_ref(),
                location,
            );
            assert_eq!(op.operation_name(), "llvm.dso_local_equivalent");
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_dso_local_equivalent_test",
                func::FuncAttributes { arguments: vec![], results: vec![pointer_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.func @callee()
                  func.func @llvm_dso_local_equivalent_test() -> !llvm.ptr {
                    %0 = llvm.dso_local_equivalent @callee : !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_extractelement() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_type, location), (i32_type.as_ref(), location)]);
            let op =
                extractelement(block.argument(0).unwrap(), block.argument(1).unwrap(), i32_type.as_ref(), location);
            assert_eq!(op.operation_name(), "llvm.extractelement");
            assert_eq!(op.vector(), block.argument(0).unwrap());
            assert_eq!(op.position(), block.argument(1).unwrap());
            assert_eq!(op.output_type(), i32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_extractelement_test",
                func::FuncAttributes {
                    arguments: vec![vector_type.into(), i32_type.into()],
                    results: vec![i32_type.into()],
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
                  func.func @llvm_extractelement_test(%arg0: vector<4xi32>, %arg1: i32) -> i32 {
                    %0 = llvm.extractelement %arg0[%arg1 : i32] : vector<4xi32>
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_extractvalue() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let struct_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), i32_type.as_ref()], false);
        module.body().append_operation({
            let mut block = context.block(&[(struct_type.as_ref(), location)]);
            let position = context.dense_i64_array_attribute(&[1]).unwrap();
            let op = extractvalue(block.argument(0).unwrap(), i32_type.as_ref(), position.as_ref(), location);
            assert_eq!(op.operation_name(), "llvm.extractvalue");
            assert_eq!(op.container(), block.argument(0).unwrap());
            assert_eq!(op.position(), position);
            assert_eq!(op.output_type(), i32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_extractvalue_test",
                func::FuncAttributes {
                    arguments: vec![struct_type.into()],
                    results: vec![i32_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            concat!(
                "module {\n",
                "  func.func @llvm_extractvalue_test(%arg0: !llvm.struct<(i32, i32)>) -> i32 {\n",
                "    %0 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i32)> \n",
                "    return %0 : i32\n",
                "  }\n",
                "}\n",
            ),
        );
    }

    #[test]
    fn test_fcmp() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let i1_type = context.signless_integer_type(1);
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let op = fcmp(
                block.argument(0).unwrap(),
                block.argument(1).unwrap(),
                i1_type.as_ref(),
                context.integer_attribute(context.signless_integer_type(64), 1).as_ref(),
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.fcmp");
            assert_eq!(op.lhs(), block.argument(0).unwrap());
            assert_eq!(op.rhs(), block.argument(1).unwrap());
            assert_eq!(op.output_type(), i1_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_fcmp_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into()],
                    results: vec![i1_type.into()],
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
                  func.func @llvm_fcmp_test(%arg0: f32, %arg1: f32) -> i1 {
                    %0 = llvm.fcmp \"oeq\" %arg0, %arg1 : f32
                    return %0 : i1
                  }
                }
            "},
        );
    }

    #[test]
    fn test_fence() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = fence(
                context.integer_attribute(context.signless_integer_type(64), 7).as_ref(),
                Some(context.string_attribute("singlethread").as_ref()),
                location,
            );
            assert_eq!(op.operation_name(), "llvm.fence");
            assert!(op.syncscope().is_some());
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<ValueRef, _>(&[], location));
            func::func(
                "llvm_fence_test",
                func::FuncAttributes { arguments: vec![], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_fence_test() {
                    llvm.fence syncscope(\"singlethread\") seq_cst
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_freeze() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let input = block.append_operation(constant(context.integer_attribute(i32_type, 42), i32_type, location));
            let op = freeze(input.result(0).unwrap(), i32_type.as_ref(), location);
            assert_eq!(op.operation_name(), "llvm.freeze");
            assert_eq!(op.value(), input.result(0).unwrap());
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.output_type(), i32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_freeze_test",
                func::FuncAttributes { arguments: vec![], results: vec![i32_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_freeze_test() -> i32 {
                    %0 = llvm.mlir.constant(42 : i32) : i32
                    %1 = llvm.freeze %0 : i32
                    return %1 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_getelementptr() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let op = getelementptr(
                block.argument(0).unwrap(),
                &[block.argument(1).unwrap().into()],
                pointer_type.as_ref(),
                context.dense_i32_array_attribute(&[i32::MIN]).unwrap().as_ref(),
                context.type_attribute(i32_type).as_ref(),
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.getelementptr");
            assert_eq!(op.base(), block.argument(0).unwrap());
            assert_eq!(op.dynamic_indices(), vec![block.argument(1).unwrap()]);
            assert_eq!(op.output_type(), pointer_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_getelementptr_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i32_type.into()],
                    results: vec![pointer_type.into()],
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
                  func.func @llvm_getelementptr_test(%arg0: !llvm.ptr, %arg1: i32) -> !llvm.ptr {
                    %0 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr, i32) -> !llvm.ptr, i32
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_global_ctors() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let function_type = context.llvm_function_type(context.llvm_void_type(), &[] as &[TypeRef], false);
        let mut ctor_body = context.region();
        let mut ctor_block = context.block_with_no_arguments();
        ctor_block.append_operation(super::super::core::r#return(None, location));
        ctor_body.append_block(ctor_block);
        module.body().append_operation(super::func(
            context.string_attribute("ctor").as_ref(),
            None,
            context.type_attribute(function_type).as_ref(),
            Some(context.llvm_linkage_attribute(Linkage::External).as_ref()),
            ctor_body,
            location,
        ));
        module.body().append_operation({
            let ctors = context.array_attribute(&[context.flat_symbol_ref_attribute("ctor").as_ref()]);
            let priorities =
                context.array_attribute(&[context.integer_attribute(context.signless_integer_type(32), 0).as_ref()]);
            let data = context.array_attribute(&[context.llvm_zero_attribute().as_ref()]);
            let op = global_ctors(ctors.as_ref(), priorities.as_ref(), data.as_ref(), location);
            assert_eq!(op.operation_name(), "llvm.mlir.global_ctors");
            assert_eq!(op.ctors(), ctors);
            assert_eq!(op.priorities(), priorities);
            assert_eq!(op.data(), data);
            op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.func @ctor() {
                    llvm.return
                  }
                  llvm.mlir.global_ctors ctors = [@ctor], priorities = [0 : i32], data = [#llvm.zero]
                }
            "},
        );
    }

    #[test]
    fn test_global_dtors() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let function_type = context.llvm_function_type(context.llvm_void_type(), &[] as &[TypeRef], false);
        let mut dtor_body = context.region();
        let mut dtor_block = context.block_with_no_arguments();
        dtor_block.append_operation(super::super::core::r#return(None, location));
        dtor_body.append_block(dtor_block);
        module.body().append_operation(super::func(
            context.string_attribute("dtor").as_ref(),
            None,
            context.type_attribute(function_type).as_ref(),
            Some(context.llvm_linkage_attribute(Linkage::External).as_ref()),
            dtor_body,
            location,
        ));
        module.body().append_operation({
            let dtors = context.array_attribute(&[context.flat_symbol_ref_attribute("dtor").as_ref()]);
            let priorities =
                context.array_attribute(&[context.integer_attribute(context.signless_integer_type(32), 0).as_ref()]);
            let data = context.array_attribute(&[context.llvm_zero_attribute().as_ref()]);
            let op = global_dtors(dtors.as_ref(), priorities.as_ref(), data.as_ref(), location);
            assert_eq!(op.operation_name(), "llvm.mlir.global_dtors");
            assert_eq!(op.dtors(), dtors);
            assert_eq!(op.priorities(), priorities);
            assert_eq!(op.data(), data);
            op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.func @dtor() {
                    llvm.return
                  }
                  llvm.mlir.global_dtors dtors = [@dtor], priorities = [0 : i32], data = [#llvm.zero]
                }
            "},
        );
    }

    #[test]
    fn test_global() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let value = context.integer_attribute(i32_type, 42);
            let op = global(
                context.type_attribute(i32_type).as_ref(),
                true,
                context.string_attribute("value").as_ref(),
                context.llvm_linkage_attribute(Linkage::Internal).as_ref(),
                false,
                false,
                false,
                Some(value.as_ref()),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                context.region(),
                location,
            );
            assert_eq!(op.operation_name(), "llvm.mlir.global");
            assert_eq!(op.global_type(), context.type_attribute(i32_type).as_ref());
            assert!(op.constant());
            assert_eq!(op.value(), Some(value.as_ref()));
            op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.mlir.global internal constant @value(42 : i32) {addr_space = 0 : i32} : i32
                }
            "},
        );
    }

    #[test]
    fn test_icmp() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let op = icmp(
                block.argument(0).unwrap(),
                block.argument(1).unwrap(),
                i1_type.as_ref(),
                context.integer_attribute(context.signless_integer_type(64), 0).as_ref(),
                location,
            );
            assert_eq!(op.operation_name(), "llvm.icmp");
            assert_eq!(op.lhs(), block.argument(0).unwrap());
            assert_eq!(op.rhs(), block.argument(1).unwrap());
            assert_eq!(op.output_type(), i1_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_icmp_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![i1_type.into()],
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
                  func.func @llvm_icmp_test(%arg0: i32, %arg1: i32) -> i1 {
                    %0 = llvm.icmp \"eq\" %arg0, %arg1 : i32
                    return %0 : i1
                  }
                }
            "},
        );
    }

    #[test]
    fn test_ifunc() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let implementation_type = context.llvm_function_type(context.llvm_void_type(), &[] as &[TypeRef], false);
        let resolver_type = context.llvm_function_type(pointer_type, &[] as &[TypeRef], false);
        module.body().append_operation(super::func(
            context.string_attribute("implementation").as_ref(),
            None,
            context.type_attribute(implementation_type).as_ref(),
            Some(context.llvm_linkage_attribute(Linkage::External).as_ref()),
            context.region(),
            location,
        ));
        let mut resolver_body = context.region();
        let mut resolver_block = context.block_with_no_arguments();
        let address = resolver_block.append_operation(address_of("implementation", pointer_type, location));
        resolver_block.append_operation(llvm_return(Some(address.result(0).unwrap().into()), location));
        resolver_body.append_block(resolver_block);
        module.body().append_operation(super::func(
            context.string_attribute("resolver").as_ref(),
            None,
            context.type_attribute(resolver_type).as_ref(),
            Some(context.llvm_linkage_attribute(Linkage::Internal).as_ref()),
            resolver_body,
            location,
        ));
        module.body().append_operation({
            let op = ifunc(
                context.string_attribute("selected").as_ref(),
                context.type_attribute(implementation_type).as_ref(),
                context.flat_symbol_ref_attribute("resolver").as_ref(),
                context.type_attribute(pointer_type).as_ref(),
                context.llvm_linkage_attribute(Linkage::Internal).as_ref(),
                true,
                None,
                None,
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.mlir.ifunc");
            assert_eq!(op.sym_name(), context.string_attribute("selected").as_ref());
            assert_eq!(op.i_func_type(), context.type_attribute(implementation_type).as_ref());
            assert_eq!(op.resolver(), context.flat_symbol_ref_attribute("resolver").as_ref());
            assert!(op.dso_local());
            op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.func @implementation()
                  llvm.func internal @resolver() -> !llvm.ptr {
                    %0 = llvm.mlir.addressof @implementation : !llvm.ptr
                    llvm.return %0 : !llvm.ptr
                  }
                  llvm.mlir.ifunc internal @selected : !llvm.func<void ()>, !llvm.ptr @resolver {dso_local}
                }
            "},
        );
    }

    #[test]
    fn test_indirectbr() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut entry_block = context.block(&[(pointer_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let mut destination_block = context.block(&[(i32_type.as_ref(), location)]);
            let address = entry_block.argument(0).unwrap();
            let forwarded = entry_block.argument(1).unwrap();
            let op = indirectbr(
                address,
                &[&destination_block],
                &[forwarded.into()],
                context.dense_i32_array_attribute(&[1]).unwrap().as_ref(),
                location,
            );
            assert_eq!(op.operation_name(), "llvm.indirectbr");
            assert_eq!(op.address(), address);
            assert_eq!(op.destinations(), vec![BlockRef::from(&destination_block)]);
            assert_eq!(op.successor_operands(), vec![forwarded]);
            entry_block.append_operation(op);
            destination_block.append_operation(func::r#return(&[destination_block.argument(0).unwrap()], location));
            let mut region = context.region();
            region.append_block(entry_block);
            region.append_block(destination_block);
            func::func(
                "llvm_indirectbr_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i32_type.into()],
                    results: vec![i32_type.into()],
                    ..Default::default()
                },
                region,
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_indirectbr_test(%arg0: !llvm.ptr, %arg1: i32) -> i32 {
                    llvm.indirectbr %arg0 : !llvm.ptr, [
                    ^bb1(%arg1 : i32)
                    ]
                  ^bb1(%0: i32):  // pred: ^bb0
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_inline_asm() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let op = inline_asm(
                &[block.argument(0).unwrap().into()],
                i32_type.as_ref(),
                context.string_attribute("mov $0, $1").as_ref(),
                context.string_attribute("=r,r").as_ref(),
                true,
                false,
                None,
                None,
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.inline_asm");
            assert_eq!(InlineAsmOperation::operands(&op), vec![block.argument(0).unwrap()]);
            assert!(op.has_side_effects());
            assert_eq!(op.output_type(), i32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_inline_asm_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
                    results: vec![i32_type.into()],
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
                  func.func @llvm_inline_asm_test(%arg0: i32) -> i32 {
                    %0 = llvm.inline_asm has_side_effects \"mov $0, $1\", \"=r,r\" %arg0 : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_insert_element() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block =
                context.block(&[(vector_type, location), (i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let op = insert_element(
                block.argument(0).unwrap(),
                block.argument(1).unwrap(),
                block.argument(2).unwrap(),
                vector_type,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.insertelement");
            assert_eq!(op.vector(), block.argument(0).unwrap());
            assert_eq!(op.value(), block.argument(1).unwrap());
            assert_eq!(op.position(), block.argument(2).unwrap());
            assert_eq!(op.output_type(), vector_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_insert_element_test",
                func::FuncAttributes {
                    arguments: vec![vector_type.into(), i32_type.into(), i32_type.into()],
                    results: vec![vector_type.into()],
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
                  func.func @llvm_insert_element_test(%arg0: vector<4xi32>, %arg1: i32, %arg2: i32) -> vector<4xi32> {
                    %0 = llvm.insertelement %arg1, %arg0[%arg2 : i32] : vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_insert_value() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let struct_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), i32_type.as_ref()], false);
        module.body().append_operation({
            let mut block = context.block(&[(struct_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let position = context.dense_i64_array_attribute(&[1]).unwrap();
            let op = insert_value(
                block.argument(0).unwrap(),
                block.argument(1).unwrap(),
                struct_type.as_ref(),
                position.as_ref(),
                location,
            );
            assert_eq!(op.operation_name(), "llvm.insertvalue");
            assert_eq!(op.container(), block.argument(0).unwrap());
            assert_eq!(op.value(), block.argument(1).unwrap());
            assert_eq!(op.position(), position);
            assert_eq!(op.output_type(), struct_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_insert_value_test",
                func::FuncAttributes {
                    arguments: vec![struct_type.into(), i32_type.into()],
                    results: vec![struct_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            concat!(
                "module {\n",
                "  func.func @llvm_insert_value_test(%arg0: !llvm.struct<(i32, i32)>, %arg1: i32) -> !llvm.struct<(i32, i32)> {\n",
                "    %0 = llvm.insertvalue %arg1, %arg0[1] : !llvm.struct<(i32, i32)> \n",
                "    return %0 : !llvm.struct<(i32, i32)>\n",
                "  }\n",
                "}\n",
            ),
        );
    }

    #[test]
    fn test_invoke() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        let exception_type = context.llvm_literal_struct_type(&[pointer_type.as_ref(), i8_type.as_ref()], false);
        let function_type = context.llvm_function_type(i32_type, &[i32_type], false);
        module.body().append_operation(super::func(
            context.string_attribute("callee").as_ref(),
            None,
            context.type_attribute(function_type).as_ref(),
            Some(context.llvm_linkage_attribute(Linkage::External).as_ref()),
            context.region(),
            location,
        ));
        module.body().append_operation({
            let mut entry_block = context.block(&[(i32_type.as_ref(), location)]);
            let mut normal_block = context.block(&[(i32_type.as_ref(), location)]);
            let mut unwind_block = context.block(&[(i32_type.as_ref(), location)]);
            let argument = entry_block.argument(0).unwrap();
            let op = invoke(
                &[argument.into()],
                &normal_block,
                &[argument.into()],
                &unwind_block,
                &[argument.into()],
                &[],
                i32_type.as_ref(),
                None,
                Some(context.flat_symbol_ref_attribute("callee").as_ref()),
                None,
                None,
                None,
                None,
                None,
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.invoke");
            assert_eq!(op.callee_operands(), vec![argument]);
            assert_eq!(op.normal_destination(), BlockRef::from(&normal_block));
            assert_eq!(op.unwind_destination(), BlockRef::from(&unwind_block));
            assert_eq!(op.normal_destination_operands(), vec![argument]);
            assert_eq!(op.unwind_destination_operands(), vec![argument]);
            assert!(op.op_bundle_operands().is_empty());
            assert_eq!(op.output_type(), i32_type);
            entry_block.append_operation(op);
            normal_block.append_operation(func::r#return(&[normal_block.argument(0).unwrap()], location));
            unwind_block.append_operation(landingpad(&[], exception_type.as_ref(), true, location));
            unwind_block.append_operation(func::r#return(&[unwind_block.argument(0).unwrap()], location));
            let mut region = context.region();
            region.append_block(entry_block);
            region.append_block(normal_block);
            region.append_block(unwind_block);
            func::func(
                "llvm_invoke_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
                    results: vec![i32_type.into()],
                    ..Default::default()
                },
                region,
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.func @callee(i32) -> i32
                  func.func @llvm_invoke_test(%arg0: i32) -> i32 {
                    %0 = llvm.invoke @callee(%arg0) to ^bb1(%arg0 : i32) unwind ^bb2(%arg0 : i32) : (i32) -> i32
                  ^bb1(%1: i32):  // pred: ^bb0
                    return %1 : i32
                  ^bb2(%2: i32):  // pred: ^bb0
                    %3 = llvm.landingpad cleanup : !llvm.struct<(ptr, i8)>
                    return %2 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_func() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let function_type = context.llvm_function_type(i32_type, &[i32_type], false);
        let mut body = context.region();
        let mut block = context.block(&[(i32_type.as_ref(), location)]);
        block.append_operation(super::super::core::r#return(Some(block.argument(0).unwrap().into()), location));
        body.append_block(block);
        module.body().append_operation({
            let op = super::func(
                context.string_attribute("identity").as_ref(),
                None,
                context.type_attribute(function_type).as_ref(),
                Some(context.llvm_linkage_attribute(Linkage::Internal).as_ref()),
                body,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.func");
            assert_eq!(op.sym_name(), context.string_attribute("identity").as_ref());
            assert_eq!(op.function_type(), context.type_attribute(function_type).as_ref());
            assert!(op.linkage().is_some());
            op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.func internal @identity(%arg0: i32) -> i32 {
                    llvm.return %arg0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_landingpad() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let pointer_type = context.llvm_pointer_type(0);
        let result_type = context.llvm_literal_struct_type(&[pointer_type.as_ref(), i8_type.as_ref()], false);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = landingpad(&[], result_type.as_ref(), true, location);
            assert_eq!(op.operation_name(), "llvm.landingpad");
            assert!(op.cleanup());
            assert_eq!(op.output_type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_landingpad_test",
                func::FuncAttributes { arguments: vec![], results: vec![result_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_landingpad_test() -> !llvm.struct<(ptr, i8)> {
                    %0 = llvm.landingpad cleanup : !llvm.struct<(ptr, i8)>
                    return %0 : !llvm.struct<(ptr, i8)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_linker_options() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let options = context.array_attribute(&[
                context.string_attribute("framework").as_ref(),
                context.string_attribute("Accelerate").as_ref(),
            ]);
            let op = linker_options(options.as_ref(), location);
            assert_eq!(op.operation_name(), "llvm.linker_options");
            assert_eq!(op.options(), options);
            op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.linker_options [\"framework\", \"Accelerate\"]
                }
            "},
        );
    }

    #[test]
    fn test_module_flags() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let flags = context.array_attribute(&[] as &[AttributeRef]);
            let op = module_flags(flags.as_ref(), location);
            assert_eq!(op.operation_name(), "llvm.module_flags");
            assert_eq!(op.flags(), flags);
            op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.module_flags []
                }
            "},
        );
    }

    #[test]
    fn test_named_metadata() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let nodes = context.array_attribute(&[] as &[AttributeRef]);
            let op = named_metadata(context.string_attribute("llvm.ident").as_ref(), nodes.as_ref(), location);
            assert_eq!(op.operation_name(), "llvm.named_metadata");
            assert_eq!(op.metadata_name(), context.string_attribute("llvm.ident").as_ref());
            assert_eq!(op.nodes(), nodes);
            op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.named_metadata \"llvm.ident\" []
                }
            "},
        );
    }

    #[test]
    fn test_none() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let token_type = context.llvm_token_type();
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = none(token_type.as_ref(), location);
            assert_eq!(op.operation_name(), "llvm.mlir.none");
            assert_eq!(op.output_type(), token_type);
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_none_test",
                func::FuncAttributes { arguments: vec![], results: vec![token_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_none_test() -> !llvm.token {
                    %0 = llvm.mlir.none : !llvm.token
                    return %0 : !llvm.token
                  }
                }
            "},
        );
    }

    #[test]
    fn test_resume() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let pointer_type = context.llvm_pointer_type(0);
        let exception_type = context.llvm_literal_struct_type(&[pointer_type.as_ref(), i8_type.as_ref()], false);
        module.body().append_operation({
            let mut block = context.block(&[(exception_type.as_ref(), location)]);
            let op = resume(block.argument(0).unwrap(), location);
            assert_eq!(op.operation_name(), "llvm.resume");
            assert_eq!(op.value(), block.argument(0).unwrap());
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            func::func(
                "llvm_resume_test",
                func::FuncAttributes { arguments: vec![exception_type.into()], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_resume_test(%arg0: !llvm.struct<(ptr, i8)>) {
                    llvm.resume %arg0 : !llvm.struct<(ptr, i8)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_shufflevector() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let vector_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_type, location), (vector_type, location)]);
            let mask = context.dense_i32_array_attribute(&[0, 1, 4, 5]).unwrap();
            let op = shufflevector(
                block.argument(0).unwrap(),
                block.argument(1).unwrap(),
                vector_type,
                mask.as_ref(),
                location,
            );
            assert_eq!(op.operation_name(), "llvm.shufflevector");
            assert_eq!(op.first_vector(), block.argument(0).unwrap());
            assert_eq!(op.second_vector(), block.argument(1).unwrap());
            assert_eq!(op.mask(), mask);
            assert_eq!(op.output_type(), vector_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_shufflevector_test",
                func::FuncAttributes {
                    arguments: vec![vector_type.into(), vector_type.into()],
                    results: vec![vector_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            concat!(
                "module {\n",
                "  func.func @llvm_shufflevector_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi32> {\n",
                "    %0 = llvm.shufflevector %arg0, %arg1 [0, 1, 4, 5] : vector<4xi32> \n",
                "    return %0 : vector<4xi32>\n",
                "  }\n",
                "}\n",
            ),
        );
    }

    #[test]
    fn test_switch() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut entry_block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let mut default_block = context.block(&[(i32_type.as_ref(), location)]);
            let mut case_block = context.block(&[(i32_type.as_ref(), location)]);
            let value = entry_block.argument(0).unwrap();
            let forwarded = entry_block.argument(1).unwrap();
            let case_values = context
                .dense_elements_attribute(
                    context.tensor_type(i32_type, &[crate::Size::Static(1)], None, location).unwrap(),
                    &[context.integer_attribute(i32_type, 7)],
                )
                .unwrap();
            let op = switch(
                value,
                &default_block,
                &[forwarded.into()],
                &[&case_block],
                &[forwarded.into()],
                Some(case_values.as_ref()),
                context.dense_i32_array_attribute(&[1]).unwrap().as_ref(),
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.switch");
            assert_eq!(op.value(), value);
            assert_eq!(op.default_destination(), BlockRef::from(&default_block));
            assert_eq!(op.case_destinations(), vec![BlockRef::from(&case_block)]);
            assert_eq!(op.default_operands(), vec![forwarded]);
            assert_eq!(op.case_operands(), vec![forwarded]);
            entry_block.append_operation(op);
            default_block.append_operation(func::r#return(&[default_block.argument(0).unwrap()], location));
            case_block.append_operation(func::r#return(&[case_block.argument(0).unwrap()], location));
            let mut region = context.region();
            region.append_block(entry_block);
            region.append_block(default_block);
            region.append_block(case_block);
            func::func(
                "llvm_switch_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![i32_type.into()],
                    ..Default::default()
                },
                region,
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_switch_test(%arg0: i32, %arg1: i32) -> i32 {
                    llvm.switch %arg0 : i32, ^bb1(%arg1 : i32) [
                      7: ^bb2(%arg1 : i32)
                    ]
                  ^bb1(%0: i32):  // pred: ^bb0
                    return %0 : i32
                  ^bb2(%1: i32):  // pred: ^bb0
                    return %1 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_va_arg() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let op = va_arg(block.argument(0).unwrap(), i32_type.as_ref(), location);
            assert_eq!(op.operation_name(), "llvm.va_arg");
            assert_eq!(op.argument(), block.argument(0).unwrap());
            assert_eq!(op.output_type(), i32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_va_arg_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into()],
                    results: vec![i32_type.into()],
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
                  func.func @llvm_va_arg_test(%arg0: !llvm.ptr) -> i32 {
                    %0 = llvm.va_arg %arg0 : (!llvm.ptr) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }
}
