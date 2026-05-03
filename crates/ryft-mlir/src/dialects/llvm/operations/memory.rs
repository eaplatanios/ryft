use crate::{
    AttributeRef, DetachedOp, DialectHandle, Location, Operation, OperationBuilder, TypeRef, Value, ValueRef, mlir_op,
};

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
pub fn get_element_ptr<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
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
        .expect("invalid arguments to `llvm::get_element_ptr`")
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

    use crate::dialects::func;
    use crate::{Attribute, Block, Context, Operation, Type};

    use super::*;

    #[test]
    fn test_get_element_ptr() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let op = get_element_ptr(
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
