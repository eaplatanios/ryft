use crate::{
    DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, TypeRef, Value, ValueRef, mlir_op,
};

/// Canonical MLIR operation name for [`FreezeOperation`].
pub const FREEZE_OPERATION_NAME: &str = "llvm.freeze";

/// Operation trait for `llvm.freeze`.
pub trait FreezeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FREEZE_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Freeze);

/// Constructs a new detached `llvm.freeze` operation.
pub fn freeze<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    value: V1,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedFreezeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(FREEZE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::freeze`"))
    })
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
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(NoneToken);

/// Constructs a new detached `llvm.mlir.none` operation.
pub fn none<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedNoneTokenOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(NONE_TOKEN_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::none`"))
    })
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
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let input = block
                    .append_operation(constant(context.integer_attribute(i32_type, 42), i32_type, location).unwrap())
                    .unwrap();
                let op = freeze(input.result(0).unwrap(), i32_type.as_ref(), location).unwrap();
                assert_eq!(op.operation_name(), "llvm.freeze");
                assert_eq!(op.value().unwrap(), input.result(0).unwrap());
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_freeze_test",
                    func::FuncAttributes { arguments: vec![], results: vec![i32_type.into()], ..Default::default() },
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
        let module = context.module(location).unwrap();
        let token_type = context.token_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = none(token_type.as_ref(), location).unwrap();
                assert_eq!(op.operation_name(), "llvm.mlir.none");
                assert_eq!(op.output_type().unwrap(), token_type);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                // Note that token values cannot cross function boundaries (i.e., `func.return` does not have the
                // `TokenConsumerTrait`), and so the produced token is not returned from the function.
                block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[] as &[ValueRef], location).unwrap()).unwrap();
                func::func(
                    "llvm_none_test",
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
                  func.func @llvm_none_test() {
                    %0 = llvm.mlir.none : token
                    return
                  }
                }
            "},
        );
    }
}
