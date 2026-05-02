use crate::{
    AttributeRef, DetachedOp, DialectHandle, Location, Operation, OperationBuilder, TypeRef, Value, ValueRef, mlir_op,
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
pub fn mlir_alias<'c, 't: 'c, L: Location<'c, 't>>(
    alias_type: AttributeRef<'c, 't>,
    sym_name: AttributeRef<'c, 't>,
    linkage: AttributeRef<'c, 't>,
    dso_local: bool,
    thread_local_: bool,
    unnamed_addr: Option<AttributeRef<'c, 't>>,
    visibility_: Option<AttributeRef<'c, 't>>,
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
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::mlir_alias`")
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
    location: L,
) -> DetachedAtomicCmpXchgOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ATOMIC_CMP_XCHG_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(compare_value);
    builder = builder.add_operand(new_value);
    builder = builder.add_result(result_type);
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
    location: L,
) -> DetachedAtomicRmwOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ATOMIC_RMW_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
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
        self.operand_values().skip(0).collect()
    }

    /// Returns the `op_bundle_operands` operands.
    fn op_bundle_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(0).collect()
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
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmath_flags", fastmath_flags);
    }
    if let Some(op_bundle_sizes) = op_bundle_sizes {
        builder = builder.add_attribute("op_bundle_sizes", op_bundle_sizes);
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

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Call);

/// Constructs a new detached `llvm.call` operation.
pub fn call<'c, 't: 'c, L: Location<'c, 't>>(
    callee_operands: &[ValueRef<'c, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    callee: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedCallOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CALL_OPERATION_NAME, location);
    builder = builder.add_operands(callee_operands);
    builder = builder.add_result(result_type);
    if let Some(callee) = callee {
        builder = builder.add_attribute("callee", callee);
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
    location: L,
) -> DetachedComdatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(COMDAT_OPERATION_NAME, location);
    builder = builder.add_attribute("sym_name", sym_name);
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
        self.operand_values().skip(1).collect()
    }

    /// Returns the `false_destination_operands` operands.
    fn false_destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
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
pub fn cond_br<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    condition: V1,
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
        self.attribute("raw_constant_indices").unwrap()
    }

    /// Returns the `elem_type` attribute.
    fn elem_type(&self) -> AttributeRef<'c, 't> {
        self.attribute("elem_type").unwrap()
    }

    /// Returns the optional `no_wrap_flags` attribute.
    fn no_wrap_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("no_wrap_flags")
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
    builder = builder.add_attribute("raw_constant_indices", raw_constant_indices);
    builder = builder.add_attribute("elem_type", elem_type);
    if let Some(no_wrap_flags) = no_wrap_flags {
        builder = builder.add_attribute("no_wrap_flags", no_wrap_flags);
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
pub fn mlir_global_ctors<'c, 't: 'c, L: Location<'c, 't>>(
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
        .expect("invalid arguments to `llvm::mlir_global_ctors`")
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
pub fn mlir_global_dtors<'c, 't: 'c, L: Location<'c, 't>>(
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
        .expect("invalid arguments to `llvm::mlir_global_dtors`")
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
pub fn mlir_global<'c, 't: 'c, L: Location<'c, 't>>(
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
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::mlir_global`")
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
pub fn mlir_ifunc<'c, 't: 'c, L: Location<'c, 't>>(
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
        .expect("invalid arguments to `llvm::mlir_ifunc`")
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

    /// Returns the `indbr_operand_segments` attribute.
    fn indbr_operand_segments(&self) -> AttributeRef<'c, 't> {
        self.attribute("indbr_operand_segments").unwrap()
    }
}

mlir_op!(IndirectBr);

/// Constructs a new detached `llvm.indirectbr` operation.
pub fn indirectbr<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    address: V1,
    successor_operands: &[ValueRef<'c, 'c, 't>],
    indbr_operand_segments: AttributeRef<'c, 't>,
    location: L,
) -> DetachedIndirectBrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INDIRECT_BR_OPERATION_NAME, location);
    builder = builder.add_operand(address);
    builder = builder.add_operands(successor_operands);
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
    location: L,
) -> DetachedInlineAsmOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INLINE_ASM_OPERATION_NAME, location);
    builder = builder.add_operands(operands);
    builder = builder.add_result(result_type);
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
        self.operand_values().skip(0).collect()
    }

    /// Returns the `normal_destination_operands` operands.
    fn normal_destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(0).collect()
    }

    /// Returns the `unwind_destination_operands` operands.
    fn unwind_destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(0).collect()
    }

    /// Returns the `op_bundle_operands` operands.
    fn op_bundle_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(0).collect()
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
pub fn invoke<'c, 't: 'c, L: Location<'c, 't>>(
    callee_operands: &[ValueRef<'c, 'c, 't>],
    normal_destination_operands: &[ValueRef<'c, 'c, 't>],
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
    builder = builder.add_result(result_type);
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
    if let Some(op_bundle_sizes) = op_bundle_sizes {
        builder = builder.add_attribute("op_bundle_sizes", op_bundle_sizes);
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
pub fn mlir_none<'c, 't: 'c, L: Location<'c, 't>>(
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
        .expect("invalid arguments to `llvm::mlir_none`")
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
        self.operand_values().skip(1).collect()
    }

    /// Returns the `case_operands` operands.
    fn case_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
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
pub fn switch<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    value: V1,
    default_operands: &[ValueRef<'c, 'c, 't>],
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
    if let Some(case_values) = case_values {
        builder = builder.add_attribute("case_values", case_values);
    }
    builder = builder.add_attribute("case_operand_segments", case_operand_segments);
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

    use crate::{Block, Context, Operation, Type, dialects::func};

    use super::super::core::constant;
    use super::*;

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
}
