use crate::{
    AttributeRef, DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, Type, TypeRef, Value,
    ValueRef, mlir_op,
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
    fn integer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `annotation` operand.
    fn annotation(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `file_name` operand.
    fn file_name(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `line` operand.
    fn line(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedAnnotationOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(ANNOTATION_OPERATION_NAME, location);
    builder = builder.add_operand(integer);
    builder = builder.add_operand(annotation);
    builder = builder.add_operand(file_name);
    builder = builder.add_operand(line);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_annotation`"))
    })
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
    fn address(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `varInfo` attribute.
    fn var_info(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("varInfo")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "varInfo",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `locationExpr` attribute.
    fn location_expr(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("locationExpr")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "locationExpr",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(DbgDeclare);

/// Constructs a new detached `llvm.intr.dbg.declare` operation.
pub fn intr_dbg_declare<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    address: V0,
    var_info: AttributeRef<'c, 't>,
    location_expr: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedDbgDeclareOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(DBG_DECLARE_OPERATION_NAME, location);
    builder = builder.add_operand(address);
    builder = builder.add_attribute("varInfo", var_info);
    builder = builder.add_attribute("locationExpr", location_expr);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_dbg_declare`"))
    })
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
    fn label(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("label")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "label",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(DbgLabel);

/// Constructs a new detached `llvm.intr.dbg.label` operation.
pub fn intr_dbg_label<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    label: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedDbgLabelOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(DBG_LABEL_OPERATION_NAME, location);
    builder = builder.add_attribute("label", label);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_dbg_label`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `varInfo` attribute.
    fn var_info(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("varInfo")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "varInfo",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `locationExpr` attribute.
    fn location_expr(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("locationExpr")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "locationExpr",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(DbgValue);

/// Constructs a new detached `llvm.intr.dbg.value` operation.
pub fn intr_dbg_value<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    value: V0,
    var_info: AttributeRef<'c, 't>,
    location_expr: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedDbgValueOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(DBG_VALUE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_attribute("varInfo", var_info);
    builder = builder.add_attribute("locationExpr", location_expr);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_dbg_value`"))
    })
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
pub fn intr_debug_trap<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    location: L,
) -> Result<DetachedDebugTrapOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let builder = OperationBuilder::new(DEBUG_TRAP_OPERATION_NAME, location);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_debug_trap`"))
    })
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
    fn type_info(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(EhTypeIdFor);

/// Constructs a new detached `llvm.intr.eh.typeid.for` operation.
pub fn intr_eh_type_id_for<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    type_info: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedEhTypeIdForOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(EH_TYPEID_FOR_OPERATION_NAME, location);
    builder = builder.add_operand(type_info);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_eh_type_id_for`"))
    })
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
    fn arguments(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }
}

mlir_op!(FakeUse);

/// Constructs a new detached `llvm.intr.fake.use` operation.
pub fn intr_fake_use<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedFakeUseOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(FAKE_USE_OPERATION_NAME, location);
    builder = builder.add_operands(arguments);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_fake_use`"))
    })
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
pub fn intr_trap<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(location: L) -> Result<DetachedTrapOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let builder = OperationBuilder::new(TRAP_OPERATION_NAME, location);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_trap`"))
    })
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
    fn failure_kind(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("failureKind")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "failureKind",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(UbsanTrap);

/// Constructs a new detached `llvm.intr.ubsantrap` operation.
pub fn intr_ubsan_trap<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    failure_kind: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedUbsanTrapOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(UBSAN_TRAP_OPERATION_NAME, location);
    builder = builder.add_attribute("failureKind", failure_kind);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_ubsan_trap`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `annotation` operand.
    fn annotation(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `file_name` operand.
    fn file_name(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `line` operand.
    fn line(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns the `attribute` operand.
    fn attribute(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(4)
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
) -> Result<DetachedVarAnnotationOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VAR_ANNOTATION_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(annotation);
    builder = builder.add_operand(file_name);
    builder = builder.add_operand(line);
    builder = builder.add_operand(attribute);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_var_annotation`"))
    })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Attribute, Block, Context, DialectHandle, Operation, Type};

    use super::*;

    #[test]
    fn test_intr_annotation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_annotation(arg_0, arg_1, arg_2, arg_3, i32_type, location).unwrap();
                assert_eq!(op.integer().unwrap(), arg_0);
                assert_eq!(op.annotation().unwrap(), arg_1);
                assert_eq!(op.file_name().unwrap(), arg_2);
                assert_eq!(op.line().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.annotation");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_annotation_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), pointer_type.into(), pointer_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_annotation_test(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.annotation\"(%arg0, %arg1, %arg2, %arg3) : (i32, !llvm.ptr, !llvm.ptr, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_dbg_declare() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let var_info = context.parse_attribute(r#"#llvm.di_local_variable<scope = #llvm.di_file<"file.c" in "/tmp">, name = "x", file = #llvm.di_file<"file.c" in "/tmp">, line = 1>"#).unwrap();
        let location_expr = context.llvm_di_expression_attribute(&[]).unwrap().as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_dbg_declare(arg_0, var_info, location_expr, location).unwrap();
                assert_eq!(op.address().unwrap(), arg_0);
                assert_eq!(op.var_info().unwrap(), var_info);
                assert_eq!(op.location_expr().unwrap(), location_expr);
                assert_eq!(op.operation_name(), "llvm.intr.dbg.declare");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_dbg_declare_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
                        results: vec![],
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
                #di_file = #llvm.di_file<\"file.c\" in \"/tmp\">
                #di_local_variable = #llvm.di_local_variable<scope = #di_file, name = \"x\", file = #di_file, line = 1>
                module {
                  func.func @llvm_intr_dbg_declare_test(%arg0: !llvm.ptr) {
                    llvm.intr.dbg.declare #di_local_variable = %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_dbg_label() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let label = context.parse_attribute(r#"#llvm.di_label<scope = #llvm.di_file<"file.c" in "/tmp">, name = "label", file = #llvm.di_file<"file.c" in "/tmp">, line = 1>"#).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = intr_dbg_label(label, location).unwrap();
                assert_eq!(op.label().unwrap(), label);
                assert_eq!(op.operation_name(), "llvm.intr.dbg.label");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_dbg_label_test",
                    func::FuncAttributes { arguments: vec![], results: vec![], ..Default::default() },
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
                #di_file = #llvm.di_file<\"file.c\" in \"/tmp\">
                #di_label = #llvm.di_label<scope = #di_file, name = \"label\", file = #di_file, line = 1>
                module {
                  func.func @llvm_intr_dbg_label_test() {
                    llvm.intr.dbg.label #di_label
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_dbg_value() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let var_info = context.parse_attribute(r#"#llvm.di_local_variable<scope = #llvm.di_file<"file.c" in "/tmp">, name = "x", file = #llvm.di_file<"file.c" in "/tmp">, line = 1>"#).unwrap();
        let location_expr = context.llvm_di_expression_attribute(&[]).unwrap().as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_dbg_value(arg_0, var_info, location_expr, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.var_info().unwrap(), var_info);
                assert_eq!(op.location_expr().unwrap(), location_expr);
                assert_eq!(op.operation_name(), "llvm.intr.dbg.value");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_dbg_value_test",
                    func::FuncAttributes { arguments: vec![i32_type.into()], results: vec![], ..Default::default() },
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
                #di_file = #llvm.di_file<\"file.c\" in \"/tmp\">
                #di_local_variable = #llvm.di_local_variable<scope = #di_file, name = \"x\", file = #di_file, line = 1>
                module {
                  func.func @llvm_intr_dbg_value_test(%arg0: i32) {
                    llvm.intr.dbg.value #di_local_variable = %arg0 : i32
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_debug_trap() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = intr_debug_trap(location).unwrap();
                assert_eq!(op.operation_name(), "llvm.intr.debugtrap");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_debugtrap_test",
                    func::FuncAttributes { arguments: vec![], results: vec![], ..Default::default() },
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
                  func.func @llvm_intr_debugtrap_test() {
                    llvm.intr.debugtrap
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_eh_type_id_for() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_eh_type_id_for(arg_0, i32_type, location).unwrap();
                assert_eq!(op.type_info().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.eh.typeid.for");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_eh_typeid_for_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_eh_typeid_for_test(%arg0: !llvm.ptr) -> i32 {
                    %0 = llvm.intr.eh.typeid.for %arg0 : (!llvm.ptr) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_fake_use() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_fake_use(&[arg_0.into(), arg_1.into()], location).unwrap();
                assert_eq!(op.arguments().unwrap(), vec![arg_0, arg_1]);
                assert_eq!(op.operation_name(), "llvm.intr.fake.use");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_fake_use_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![],
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
                  func.func @llvm_intr_fake_use_test(%arg0: i32, %arg1: i32) {
                    llvm.intr.fake.use %arg0, %arg1 : i32, i32
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_trap() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = intr_trap(location).unwrap();
                assert_eq!(op.operation_name(), "llvm.intr.trap");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_trap_test",
                    func::FuncAttributes { arguments: vec![], results: vec![], ..Default::default() },
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
                  func.func @llvm_intr_trap_test() {
                    llvm.intr.trap
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ubsan_trap() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i8_type = context.signless_integer_type(8);
        let failure_kind = context.integer_attribute(i8_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = intr_ubsan_trap(failure_kind, location).unwrap();
                assert_eq!(op.failure_kind().unwrap(), failure_kind);
                assert_eq!(op.operation_name(), "llvm.intr.ubsantrap");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_ubsantrap_test",
                    func::FuncAttributes { arguments: vec![], results: vec![], ..Default::default() },
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
                  func.func @llvm_intr_ubsantrap_test() {
                    llvm.intr.ubsantrap <{failureKind = 1 : i8}>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_var_annotation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (pointer_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let arg_4 = block.argument(4).unwrap();
                let op = intr_var_annotation(arg_0, arg_1, arg_2, arg_3, arg_4, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.annotation().unwrap(), arg_1);
                assert_eq!(op.file_name().unwrap(), arg_2);
                assert_eq!(op.line().unwrap(), arg_3);
                assert_eq!(VarAnnotationOperation::attribute(&op).unwrap(), arg_4);
                assert_eq!(op.operation_name(), "llvm.intr.var.annotation");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 5);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_var_annotation_test",
                    func::FuncAttributes {
                        arguments: vec![
                            pointer_type.into(),
                            pointer_type.into(),
                            pointer_type.into(),
                            i32_type.into(),
                            pointer_type.into(),
                        ],
                        results: vec![],
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
                  func.func @llvm_intr_var_annotation_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: !llvm.ptr) {
                    \"llvm.intr.var.annotation\"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
                    return
                  }
                }
            "},
        );
    }
}
