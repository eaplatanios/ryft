use crate::{
    AttributeRef, DetachedOp, DetachedRegion, DialectHandle, Error, Location, Operation, OperationBuilder, TypeRef,
    mlir_op,
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
    fn alias_type(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("alias_type")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "alias_type",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `sym_name` attribute.
    fn sym_name(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("sym_name")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "sym_name",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `linkage` attribute.
    fn linkage(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("linkage")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "linkage",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
    fn unnamed_addr(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("unnamed_addr")
    }

    /// Returns the optional `visibility_` attribute.
    fn visibility_(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
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
) -> Result<DetachedAliasOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
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
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::alias`"))
    })
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
    fn sym_name(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("sym_name")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "sym_name",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(Comdat);

/// Constructs a new detached `llvm.comdat` operation.
pub fn comdat<'c, 't: 'c, L: Location<'c, 't>>(
    sym_name: AttributeRef<'c, 't>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedComdatOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(COMDAT_OPERATION_NAME, location);
    builder = builder.add_attribute("sym_name", sym_name);
    builder = builder.add_region(body);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::comdat`"))
    })
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
    fn sym_name(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("sym_name")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "sym_name",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `comdat` attribute.
    fn comdat(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("comdat")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "comdat",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(ComdatSelector);

/// Constructs a new detached `llvm.comdat_selector` operation.
pub fn comdat_selector<'c, 't: 'c, L: Location<'c, 't>>(
    sym_name: AttributeRef<'c, 't>,
    comdat: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedComdatSelectorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(COMDAT_SELECTOR_OPERATION_NAME, location);
    builder = builder.add_attribute("sym_name", sym_name);
    builder = builder.add_attribute("comdat", comdat);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::comdat_selector`"))
    })
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
    fn function_name(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("function_name")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "function_name",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(DsoLocalEquivalent);

/// Constructs a new detached `llvm.dso_local_equivalent` operation.
pub fn dso_local_equivalent<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    function_name: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedDsoLocalEquivalentOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(DSO_LOCAL_EQUIVALENT_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("function_name", function_name);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::dso_local_equivalent`"))
    })
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
    fn ctors(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("ctors")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "ctors",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `priorities` attribute.
    fn priorities(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("priorities")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "priorities",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `data` attribute.
    fn data(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("data")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "data",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(GlobalCtors);

/// Constructs a new detached `llvm.mlir.global_ctors` operation.
pub fn global_ctors<'c, 't: 'c, L: Location<'c, 't>>(
    ctors: AttributeRef<'c, 't>,
    priorities: AttributeRef<'c, 't>,
    data: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedGlobalCtorsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(GLOBAL_CTORS_OPERATION_NAME, location);
    builder = builder.add_attribute("ctors", ctors);
    builder = builder.add_attribute("priorities", priorities);
    builder = builder.add_attribute("data", data);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::global_ctors`"))
    })
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
    fn dtors(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("dtors")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "dtors",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `priorities` attribute.
    fn priorities(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("priorities")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "priorities",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `data` attribute.
    fn data(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("data")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "data",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(GlobalDtors);

/// Constructs a new detached `llvm.mlir.global_dtors` operation.
pub fn global_dtors<'c, 't: 'c, L: Location<'c, 't>>(
    dtors: AttributeRef<'c, 't>,
    priorities: AttributeRef<'c, 't>,
    data: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedGlobalDtorsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(GLOBAL_DTORS_OPERATION_NAME, location);
    builder = builder.add_attribute("dtors", dtors);
    builder = builder.add_attribute("priorities", priorities);
    builder = builder.add_attribute("data", data);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::global_dtors`"))
    })
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
    fn global_type(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("global_type")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "global_type",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns whether the `constant` unit attribute is present.
    fn constant(&self) -> bool {
        self.has_attribute("constant")
    }

    /// Returns the `sym_name` attribute.
    fn sym_name(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("sym_name")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "sym_name",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `linkage` attribute.
    fn linkage(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("linkage")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "linkage",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
    fn value(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("value")
    }

    /// Returns the optional `alignment` attribute.
    fn alignment(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("alignment")
    }

    /// Returns the optional `addr_space` attribute.
    fn addr_space(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("addr_space")
    }

    /// Returns the optional `unnamed_addr` attribute.
    fn unnamed_addr(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("unnamed_addr")
    }

    /// Returns the optional `section` attribute.
    fn section(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("section")
    }

    /// Returns the optional `comdat` attribute.
    fn comdat(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("comdat")
    }

    /// Returns the optional `dbg_exprs` attribute.
    fn dbg_exprs(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("dbg_exprs")
    }

    /// Returns the optional `visibility_` attribute.
    fn visibility_(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("visibility_")
    }

    /// Returns the optional `target_specific_attrs` attribute.
    fn target_specific_attrs(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
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
) -> Result<DetachedGlobalOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
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
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::global`"))
    })
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
    fn sym_name(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("sym_name")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "sym_name",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `i_func_type` attribute.
    fn i_func_type(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("i_func_type")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "i_func_type",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `resolver` attribute.
    fn resolver(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("resolver")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "resolver",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `resolver_type` attribute.
    fn resolver_type(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("resolver_type")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "resolver_type",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `linkage` attribute.
    fn linkage(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("linkage")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "linkage",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns whether the `dso_local` unit attribute is present.
    fn dso_local(&self) -> bool {
        self.has_attribute("dso_local")
    }

    /// Returns the optional `address_space` attribute.
    fn address_space(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("address_space")
    }

    /// Returns the optional `unnamed_addr` attribute.
    fn unnamed_addr(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("unnamed_addr")
    }

    /// Returns the optional `visibility_` attribute.
    fn visibility_(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
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
) -> Result<DetachedIfuncOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
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
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::ifunc`"))
    })
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
    fn sym_name(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("sym_name")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "sym_name",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the optional `sym_visibility` attribute.
    fn sym_visibility(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("sym_visibility")
    }

    /// Returns the `function_type` attribute.
    fn function_type(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("function_type")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "function_type",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the optional `linkage` attribute.
    fn linkage(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
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
) -> Result<DetachedLlvmFuncOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
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
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::func`"))
    })
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
    fn options(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("options")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "options",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(LinkerOptions);

/// Constructs a new detached `llvm.linker_options` operation.
pub fn linker_options<'c, 't: 'c, L: Location<'c, 't>>(
    options: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedLinkerOptionsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(LINKER_OPTIONS_OPERATION_NAME, location);
    builder = builder.add_attribute("options", options);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::linker_options`"))
    })
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
    fn flags(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("flags")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "flags",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(ModuleFlags);

/// Constructs a new detached `llvm.module_flags` operation.
pub fn module_flags<'c, 't: 'c, L: Location<'c, 't>>(
    flags: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedModuleFlagsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MODULE_FLAGS_OPERATION_NAME, location);
    builder = builder.add_attribute("flags", flags);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::module_flags`"))
    })
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
    fn metadata_name(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("metadata_name")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "metadata_name",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `nodes` attribute.
    fn nodes(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("nodes")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "nodes",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(NamedMetadata);

/// Constructs a new detached `llvm.named_metadata` operation.
pub fn named_metadata<'c, 't: 'c, L: Location<'c, 't>>(
    metadata_name: AttributeRef<'c, 't>,
    nodes: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedNamedMetadataOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(NAMED_METADATA_OPERATION_NAME, location);
    builder = builder.add_attribute("metadata_name", metadata_name);
    builder = builder.add_attribute("nodes", nodes);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::named_metadata`"))
    })
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
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation(
                global(
                    context.type_attribute(i32_type).as_ref(),
                    false,
                    context.string_attribute("target").as_ref(),
                    context.llvm_linkage_attribute(Linkage::External).unwrap().as_ref(),
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
                )
                .unwrap(),
            )
            .unwrap();
        let mut initializer = context.region();
        let mut block = context.block_with_no_arguments();
        let address = block.append_operation(address_of("target", pointer_type, location).unwrap()).unwrap();
        block
            .append_operation(llvm_return(Some(address.result(0).unwrap().into()), location).unwrap())
            .unwrap();
        initializer.append_block(block).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let op = alias(
                    context.type_attribute(pointer_type).as_ref(),
                    context.string_attribute("target_alias").as_ref(),
                    context.llvm_linkage_attribute(Linkage::Internal).unwrap().as_ref(),
                    false,
                    false,
                    None,
                    None,
                    initializer,
                    location,
                )
                .unwrap();
                assert_eq!(op.operation_name(), "llvm.mlir.alias");
                assert_eq!(op.alias_type().unwrap(), context.type_attribute(pointer_type).as_ref());
                assert_eq!(op.sym_name().unwrap(), context.string_attribute("target_alias").as_ref());
                assert_eq!(op.linkage().unwrap(), context.llvm_linkage_attribute(Linkage::Internal).unwrap().as_ref());
                assert_eq!(op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                op
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let mut body = context.region();
        let mut block = context.block_with_no_arguments();
        let selector = comdat_selector(
            context.string_attribute("any").as_ref(),
            context.integer_attribute(context.signless_integer_type(64), 0).as_ref(),
            location,
        )
        .unwrap();
        assert_eq!(selector.operation_name(), "llvm.comdat_selector");
        assert_eq!(selector.sym_name().unwrap(), context.string_attribute("any").as_ref());
        block.append_operation(selector).unwrap();
        body.append_block(block).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let op = comdat(context.string_attribute("__llvm_comdat").as_ref(), body, location).unwrap();
                assert_eq!(op.operation_name(), "llvm.comdat");
                assert_eq!(op.sym_name().unwrap(), context.string_attribute("__llvm_comdat").as_ref());
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                op
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let mut body = context.region();
        let mut block = context.block_with_no_arguments();
        let comdat_kind = context.integer_attribute(context.signless_integer_type(64), 0);
        let selector =
            comdat_selector(context.string_attribute("any").as_ref(), comdat_kind.as_ref(), location).unwrap();
        assert_eq!(selector.operation_name(), "llvm.comdat_selector");
        assert_eq!(selector.sym_name().unwrap(), context.string_attribute("any").as_ref());
        assert_eq!(selector.comdat().unwrap(), comdat_kind.as_ref());
        block.append_operation(selector).unwrap();
        body.append_block(block).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(comdat(context.string_attribute("__llvm_comdat").as_ref(), body, location).unwrap())
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let function_type =
            context.llvm_function_type(context.llvm_void_type().unwrap(), &[] as &[TypeRef], false).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                super::func(
                    context.string_attribute("callee").as_ref(),
                    None,
                    context.type_attribute(function_type).as_ref(),
                    Some(context.llvm_linkage_attribute(Linkage::External).unwrap().as_ref()),
                    context.region(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = dso_local_equivalent(
                    pointer_type.as_ref(),
                    context.flat_symbol_ref_attribute("callee").as_ref(),
                    location,
                )
                .unwrap();
                assert_eq!(op.operation_name(), "llvm.dso_local_equivalent");
                assert_eq!(op.output_type().unwrap(), pointer_type);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_dso_local_equivalent_test",
                    func::FuncAttributes {
                        arguments: vec![],
                        results: vec![pointer_type.into()],
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
        let module = context.module(location).unwrap();
        let function_type =
            context.llvm_function_type(context.llvm_void_type().unwrap(), &[] as &[TypeRef], false).unwrap();
        let mut ctor_body = context.region();
        let mut ctor_block = context.block_with_no_arguments();
        ctor_block.append_operation(super::super::core::r#return(None, location).unwrap()).unwrap();
        ctor_body.append_block(ctor_block).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                super::func(
                    context.string_attribute("ctor").as_ref(),
                    None,
                    context.type_attribute(function_type).as_ref(),
                    Some(context.llvm_linkage_attribute(Linkage::External).unwrap().as_ref()),
                    ctor_body,
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let ctors = context.array_attribute(&[context.flat_symbol_ref_attribute("ctor").as_ref()]);
                let priorities = context
                    .array_attribute(&[context.integer_attribute(context.signless_integer_type(32), 0).as_ref()]);
                let data = context.array_attribute(&[context.llvm_zero_attribute().unwrap().as_ref()]);
                let op = global_ctors(ctors.as_ref(), priorities.as_ref(), data.as_ref(), location).unwrap();
                assert_eq!(op.operation_name(), "llvm.mlir.global_ctors");
                assert_eq!(op.ctors().unwrap(), ctors);
                assert_eq!(op.priorities().unwrap(), priorities);
                assert_eq!(op.data().unwrap(), data);
                op
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let function_type =
            context.llvm_function_type(context.llvm_void_type().unwrap(), &[] as &[TypeRef], false).unwrap();
        let mut dtor_body = context.region();
        let mut dtor_block = context.block_with_no_arguments();
        dtor_block.append_operation(super::super::core::r#return(None, location).unwrap()).unwrap();
        dtor_body.append_block(dtor_block).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                super::func(
                    context.string_attribute("dtor").as_ref(),
                    None,
                    context.type_attribute(function_type).as_ref(),
                    Some(context.llvm_linkage_attribute(Linkage::External).unwrap().as_ref()),
                    dtor_body,
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let dtors = context.array_attribute(&[context.flat_symbol_ref_attribute("dtor").as_ref()]);
                let priorities = context
                    .array_attribute(&[context.integer_attribute(context.signless_integer_type(32), 0).as_ref()]);
                let data = context.array_attribute(&[context.llvm_zero_attribute().unwrap().as_ref()]);
                let op = global_dtors(dtors.as_ref(), priorities.as_ref(), data.as_ref(), location).unwrap();
                assert_eq!(op.operation_name(), "llvm.mlir.global_dtors");
                assert_eq!(op.dtors().unwrap(), dtors);
                assert_eq!(op.priorities().unwrap(), priorities);
                assert_eq!(op.data().unwrap(), data);
                op
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let value = context.integer_attribute(i32_type, 42);
                let op = global(
                    context.type_attribute(i32_type).as_ref(),
                    true,
                    context.string_attribute("value").as_ref(),
                    context.llvm_linkage_attribute(Linkage::Internal).unwrap().as_ref(),
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
                )
                .unwrap();
                assert_eq!(op.operation_name(), "llvm.mlir.global");
                assert_eq!(op.global_type().unwrap(), context.type_attribute(i32_type).as_ref());
                assert!(op.constant());
                assert_eq!(op.value().unwrap(), Some(value.as_ref()));
                op
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let implementation_type =
            context.llvm_function_type(context.llvm_void_type().unwrap(), &[] as &[TypeRef], false).unwrap();
        let resolver_type = context.llvm_function_type(pointer_type, &[] as &[TypeRef], false).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                super::func(
                    context.string_attribute("implementation").as_ref(),
                    None,
                    context.type_attribute(implementation_type).as_ref(),
                    Some(context.llvm_linkage_attribute(Linkage::External).unwrap().as_ref()),
                    context.region(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        let mut resolver_body = context.region();
        let mut resolver_block = context.block_with_no_arguments();
        let address = resolver_block
            .append_operation(address_of("implementation", pointer_type, location).unwrap())
            .unwrap();
        resolver_block
            .append_operation(llvm_return(Some(address.result(0).unwrap().into()), location).unwrap())
            .unwrap();
        resolver_body.append_block(resolver_block).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                super::func(
                    context.string_attribute("resolver").as_ref(),
                    None,
                    context.type_attribute(resolver_type).as_ref(),
                    Some(context.llvm_linkage_attribute(Linkage::Internal).unwrap().as_ref()),
                    resolver_body,
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let op = ifunc(
                    context.string_attribute("selected").as_ref(),
                    context.type_attribute(implementation_type).as_ref(),
                    context.flat_symbol_ref_attribute("resolver").as_ref(),
                    context.type_attribute(pointer_type).as_ref(),
                    context.llvm_linkage_attribute(Linkage::Internal).unwrap().as_ref(),
                    true,
                    None,
                    None,
                    None,
                    location,
                )
                .unwrap();
                assert_eq!(op.operation_name(), "llvm.mlir.ifunc");
                assert_eq!(op.sym_name().unwrap(), context.string_attribute("selected").as_ref());
                assert_eq!(op.i_func_type().unwrap(), context.type_attribute(implementation_type).as_ref());
                assert_eq!(op.resolver().unwrap(), context.flat_symbol_ref_attribute("resolver").as_ref());
                assert!(op.dso_local());
                op
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type = context.llvm_function_type(i32_type, &[i32_type], false).unwrap();
        let mut body = context.region();
        let mut block = context.block(&[(i32_type.as_ref(), location)]);
        block
            .append_operation(super::super::core::r#return(Some(block.argument(0).unwrap().into()), location).unwrap())
            .unwrap();
        body.append_block(block).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let op = super::func(
                    context.string_attribute("identity").as_ref(),
                    None,
                    context.type_attribute(function_type).as_ref(),
                    Some(context.llvm_linkage_attribute(Linkage::Internal).unwrap().as_ref()),
                    body,
                    location,
                )
                .unwrap();
                assert_eq!(op.operation_name(), "llvm.func");
                assert_eq!(op.sym_name().unwrap(), context.string_attribute("identity").as_ref());
                assert_eq!(op.function_type().unwrap(), context.type_attribute(function_type).as_ref());
                assert!(op.linkage().unwrap().is_some());
                op
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let options = context.array_attribute(&[
                    context.string_attribute("framework").as_ref(),
                    context.string_attribute("Accelerate").as_ref(),
                ]);
                let op = linker_options(options.as_ref(), location).unwrap();
                assert_eq!(op.operation_name(), "llvm.linker_options");
                assert_eq!(op.options().unwrap(), options);
                op
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let flags = context.array_attribute(&[] as &[AttributeRef]);
                let op = module_flags(flags.as_ref(), location).unwrap();
                assert_eq!(op.operation_name(), "llvm.module_flags");
                assert_eq!(op.flags().unwrap(), flags);
                op
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let nodes = context.array_attribute(&[] as &[AttributeRef]);
                let op =
                    named_metadata(context.string_attribute("llvm.ident").as_ref(), nodes.as_ref(), location).unwrap();
                assert_eq!(op.operation_name(), "llvm.named_metadata");
                assert_eq!(op.metadata_name().unwrap(), context.string_attribute("llvm.ident").as_ref());
                assert_eq!(op.nodes().unwrap(), nodes);
                op
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
