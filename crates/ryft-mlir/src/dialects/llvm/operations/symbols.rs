use crate::{
    AttributeRef, DetachedOp, DetachedRegion, DialectHandle, Location, Operation, OperationBuilder, TypeRef, mlir_op,
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::Attribute;
    use crate::dialects::llvm::Linkage;
    use crate::dialects::llvm::operations::core::{address_of, r#return as llvm_return};
    use crate::{Block, Context, Operation, Region, Type, TypeRef, dialects::func};

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
}
