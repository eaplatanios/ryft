use crate::{
    AttributeRef, DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Type, TypeRef, Value, ValueRef,
    mlir_op,
};

/// Canonical MLIR operation name for [`AnnotationOperation`].
pub const ANNOTATION_OPERATION_NAME: &str = "llvm.intr.annotation";

/// Operation trait for `llvm.intr.annotation`.
pub trait AnnotationOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ANNOTATION_OPERATION_NAME
    }

    /// Returns the `integer` operand.
    fn integer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `annotation` operand.
    fn annotation(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `file_name` operand.
    fn file_name(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `line` operand.
    fn line(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Annotation);

/// Constructs a new detached `llvm.intr.annotation` operation.
pub fn intr_annotation<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    V3: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    integer: V0,
    annotation: V1,
    file_name: V2,
    line: V3,
    result_type: T0,
    location: L,
) -> DetachedAnnotationOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ANNOTATION_OPERATION_NAME, location);
    builder = builder.add_operand(integer);
    builder = builder.add_operand(annotation);
    builder = builder.add_operand(file_name);
    builder = builder.add_operand(line);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_annotation`")
}

/// Canonical MLIR operation name for [`DbgDeclareOperation`].
pub const DBG_DECLARE_OPERATION_NAME: &str = "llvm.intr.dbg.declare";

/// Operation trait for `llvm.intr.dbg.declare`.
pub trait DbgDeclareOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        DBG_DECLARE_OPERATION_NAME
    }

    /// Returns the `address` operand.
    fn address(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `varInfo` attribute.
    fn var_info(&self) -> AttributeRef<'c, 't> {
        self.attribute("varInfo").unwrap()
    }

    /// Returns the `locationExpr` attribute.
    fn location_expr(&self) -> AttributeRef<'c, 't> {
        self.attribute("locationExpr").unwrap()
    }
}

mlir_op!(DbgDeclare);

/// Constructs a new detached `llvm.intr.dbg.declare` operation.
pub fn intr_dbg_declare<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    address: V0,
    var_info: AttributeRef<'c, 't>,
    location_expr: AttributeRef<'c, 't>,
    location: L,
) -> DetachedDbgDeclareOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(DBG_DECLARE_OPERATION_NAME, location);
    builder = builder.add_operand(address);
    builder = builder.add_attribute("varInfo", var_info);
    builder = builder.add_attribute("locationExpr", location_expr);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_dbg_declare`")
}

/// Canonical MLIR operation name for [`DbgLabelOperation`].
pub const DBG_LABEL_OPERATION_NAME: &str = "llvm.intr.dbg.label";

/// Operation trait for `llvm.intr.dbg.label`.
pub trait DbgLabelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        DBG_LABEL_OPERATION_NAME
    }

    /// Returns the `label` attribute.
    fn label(&self) -> AttributeRef<'c, 't> {
        self.attribute("label").unwrap()
    }
}

mlir_op!(DbgLabel);

/// Constructs a new detached `llvm.intr.dbg.label` operation.
pub fn intr_dbg_label<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    label: AttributeRef<'c, 't>,
    location: L,
) -> DetachedDbgLabelOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(DBG_LABEL_OPERATION_NAME, location);
    builder = builder.add_attribute("label", label);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_dbg_label`")
}

/// Canonical MLIR operation name for [`DbgValueOperation`].
pub const DBG_VALUE_OPERATION_NAME: &str = "llvm.intr.dbg.value";

/// Operation trait for `llvm.intr.dbg.value`.
pub trait DbgValueOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        DBG_VALUE_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `varInfo` attribute.
    fn var_info(&self) -> AttributeRef<'c, 't> {
        self.attribute("varInfo").unwrap()
    }

    /// Returns the `locationExpr` attribute.
    fn location_expr(&self) -> AttributeRef<'c, 't> {
        self.attribute("locationExpr").unwrap()
    }
}

mlir_op!(DbgValue);

/// Constructs a new detached `llvm.intr.dbg.value` operation.
pub fn intr_dbg_value<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    value: V0,
    var_info: AttributeRef<'c, 't>,
    location_expr: AttributeRef<'c, 't>,
    location: L,
) -> DetachedDbgValueOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(DBG_VALUE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_attribute("varInfo", var_info);
    builder = builder.add_attribute("locationExpr", location_expr);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_dbg_value`")
}

/// Canonical MLIR operation name for [`DebugTrapOperation`].
pub const DEBUG_TRAP_OPERATION_NAME: &str = "llvm.intr.debugtrap";

/// Operation trait for `llvm.intr.debugtrap`.
pub trait DebugTrapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        DEBUG_TRAP_OPERATION_NAME
    }
}

mlir_op!(DebugTrap);

/// Constructs a new detached `llvm.intr.debugtrap` operation.
pub fn intr_debug_trap<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(location: L) -> DetachedDebugTrapOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let builder = OperationBuilder::new(DEBUG_TRAP_OPERATION_NAME, location);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_debug_trap`")
}

/// Canonical MLIR operation name for [`EhTypeidForOperation`].
pub const EH_TYPEID_FOR_OPERATION_NAME: &str = "llvm.intr.eh.typeid.for";

/// Operation trait for `llvm.intr.eh.typeid.for`.
pub trait EhTypeIdForOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EH_TYPEID_FOR_OPERATION_NAME
    }

    /// Returns the `type_info` operand.
    fn type_info(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(EhTypeIdFor);

/// Constructs a new detached `llvm.intr.eh.typeid.for` operation.
pub fn intr_eh_type_id_for<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    type_info: V0,
    result_type: T0,
    location: L,
) -> DetachedEhTypeIdForOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(EH_TYPEID_FOR_OPERATION_NAME, location);
    builder = builder.add_operand(type_info);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_eh_type_id_for`")
}

/// Canonical MLIR operation name for [`FakeUseOperation`].
pub const FAKE_USE_OPERATION_NAME: &str = "llvm.intr.fake.use";

/// Operation trait for `llvm.intr.fake.use`.
pub trait FakeUseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FAKE_USE_OPERATION_NAME
    }

    /// Returns the variadic arguments.
    fn arguments(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(FakeUse);

/// Constructs a new detached `llvm.intr.fake.use` operation.
pub fn intr_fake_use<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedFakeUseOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FAKE_USE_OPERATION_NAME, location);
    builder = builder.add_operands(arguments);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_fake_use`")
}

/// Canonical MLIR operation name for [`TrapOperation`].
pub const TRAP_OPERATION_NAME: &str = "llvm.intr.trap";

/// Operation trait for `llvm.intr.trap`.
pub trait TrapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TRAP_OPERATION_NAME
    }
}

mlir_op!(Trap);

/// Constructs a new detached `llvm.intr.trap` operation.
pub fn intr_trap<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(location: L) -> DetachedTrapOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let builder = OperationBuilder::new(TRAP_OPERATION_NAME, location);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_trap`")
}

/// Canonical MLIR operation name for [`UbsanTrapOperation`].
pub const UBSAN_TRAP_OPERATION_NAME: &str = "llvm.intr.ubsantrap";

/// Operation trait for `llvm.intr.ubsantrap`.
pub trait UbsanTrapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        UBSAN_TRAP_OPERATION_NAME
    }

    /// Returns the `failureKind` attribute.
    fn failure_kind(&self) -> AttributeRef<'c, 't> {
        self.attribute("failureKind").unwrap()
    }
}

mlir_op!(UbsanTrap);

/// Constructs a new detached `llvm.intr.ubsantrap` operation.
pub fn intr_ubsan_trap<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    failure_kind: AttributeRef<'c, 't>,
    location: L,
) -> DetachedUbsanTrapOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(UBSAN_TRAP_OPERATION_NAME, location);
    builder = builder.add_attribute("failureKind", failure_kind);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ubsan_trap`")
}

/// Canonical MLIR operation name for [`VarAnnotationOperation`].
pub const VAR_ANNOTATION_OPERATION_NAME: &str = "llvm.intr.var.annotation";

/// Operation trait for `llvm.intr.var.annotation`.
pub trait VarAnnotationOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VAR_ANNOTATION_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `annotation` operand.
    fn annotation(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `file_name` operand.
    fn file_name(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `line` operand.
    fn line(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns the `attribute` operand.
    fn attribute(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(4).unwrap()
    }
}

mlir_op!(VarAnnotation);

/// Constructs a new detached `llvm.intr.var.annotation` operation.
pub fn intr_var_annotation<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    V3: Value<'v, 'c, 't>,
    V4: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    annotation: V1,
    file_name: V2,
    line: V3,
    attribute: V4,
    location: L,
) -> DetachedVarAnnotationOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VAR_ANNOTATION_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(annotation);
    builder = builder.add_operand(file_name);
    builder = builder.add_operand(line);
    builder = builder.add_operand(attribute);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_var_annotation`")
}
