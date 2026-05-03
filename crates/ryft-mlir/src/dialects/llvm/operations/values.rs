use crate::{DetachedOp, DialectHandle, Location, Operation, OperationBuilder, TypeRef, Value, ValueRef, mlir_op};

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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::dialects::llvm::operations::core::constant;
    use crate::{Block, Context, Operation, Type};

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
}
