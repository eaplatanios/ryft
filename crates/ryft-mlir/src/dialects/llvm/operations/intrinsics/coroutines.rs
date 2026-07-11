use crate::{
    DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, Type, TypeRef, Value, ValueRef, mlir_op,
};

/// Canonical MLIR operation name for [`CoroAlignOperation`].
pub const CORO_ALIGN_OPERATION_NAME: &str = "llvm.intr.coro.align";

/// Operation trait for `llvm.intr.coro.align`.
pub trait CoroAlignOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CORO_ALIGN_OPERATION_NAME
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(CoroAlign);

/// Constructs a new detached `llvm.intr.coro.align` operation.
pub fn intr_coro_align<'v, 'c: 'v, 't: 'c, T0: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T0,
    location: L,
) -> Result<DetachedCoroAlignOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(CORO_ALIGN_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_coro_align`"))
    })
}

/// Canonical MLIR operation name for [`CoroBeginOperation`].
pub const CORO_BEGIN_OPERATION_NAME: &str = "llvm.intr.coro.begin";

/// Operation trait for `llvm.intr.coro.begin`.
pub trait CoroBeginOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CORO_BEGIN_OPERATION_NAME
    }

    /// Returns the `token` operand.
    fn token(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `memory` operand.
    fn memory(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(CoroBegin);

/// Constructs a new detached `llvm.intr.coro.begin` operation.
pub fn intr_coro_begin<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    token: V0,
    memory: V1,
    result_type: T0,
    location: L,
) -> Result<DetachedCoroBeginOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(CORO_BEGIN_OPERATION_NAME, location);
    builder = builder.add_operand(token);
    builder = builder.add_operand(memory);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_coro_begin`"))
    })
}

/// Canonical MLIR operation name for [`CoroEndOperation`].
pub const CORO_END_OPERATION_NAME: &str = "llvm.intr.coro.end";

/// Operation trait for `llvm.intr.coro.end`.
pub trait CoroEndOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CORO_END_OPERATION_NAME
    }

    /// Returns the `handle` operand.
    fn handle(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `unwind` operand.
    fn unwind(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `return_values` operand.
    fn return_values(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(CoroEnd);

/// Constructs a new detached `llvm.intr.coro.end` operation.
pub fn intr_coro_end<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    handle: V0,
    unwind: V1,
    return_values: V2,
    result_type: T0,
    location: L,
) -> Result<DetachedCoroEndOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(CORO_END_OPERATION_NAME, location);
    builder = builder.add_operand(handle);
    builder = builder.add_operand(unwind);
    builder = builder.add_operand(return_values);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_coro_end`"))
    })
}

/// Canonical MLIR operation name for [`CoroFreeOperation`].
pub const CORO_FREE_OPERATION_NAME: &str = "llvm.intr.coro.free";

/// Operation trait for `llvm.intr.coro.free`.
pub trait CoroFreeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CORO_FREE_OPERATION_NAME
    }

    /// Returns the `id` operand.
    fn id(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `handle` operand.
    fn handle(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(CoroFree);

/// Constructs a new detached `llvm.intr.coro.free` operation.
pub fn intr_coro_free<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    id: V0,
    handle: V1,
    result_type: T0,
    location: L,
) -> Result<DetachedCoroFreeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(CORO_FREE_OPERATION_NAME, location);
    builder = builder.add_operand(id);
    builder = builder.add_operand(handle);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_coro_free`"))
    })
}

/// Canonical MLIR operation name for [`CoroIdOperation`].
pub const CORO_ID_OPERATION_NAME: &str = "llvm.intr.coro.id";

/// Operation trait for `llvm.intr.coro.id`.
pub trait CoroIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CORO_ID_OPERATION_NAME
    }

    /// Returns the `alignment` operand.
    fn alignment(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `promise` operand.
    fn promise(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `coroutine_address` operand.
    fn coroutine_address(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `function_addresses` operand.
    fn function_addresses(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(CoroId);

/// Constructs a new detached `llvm.intr.coro.id` operation.
pub fn intr_coro_id<
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
    alignment: V0,
    promise: V1,
    coroutine_address: V2,
    function_addresses: V3,
    result_type: T0,
    location: L,
) -> Result<DetachedCoroIdOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(CORO_ID_OPERATION_NAME, location);
    builder = builder.add_operand(alignment);
    builder = builder.add_operand(promise);
    builder = builder.add_operand(coroutine_address);
    builder = builder.add_operand(function_addresses);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_coro_id`"))
    })
}

/// Canonical MLIR operation name for [`CoroPromiseOperation`].
pub const CORO_PROMISE_OPERATION_NAME: &str = "llvm.intr.coro.promise";

/// Operation trait for `llvm.intr.coro.promise`.
pub trait CoroPromiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CORO_PROMISE_OPERATION_NAME
    }

    /// Returns the `handle` operand.
    fn handle(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `alignment` operand.
    fn alignment(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `from` operand.
    fn from(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(CoroPromise);

/// Constructs a new detached `llvm.intr.coro.promise` operation.
pub fn intr_coro_promise<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    handle: V0,
    alignment: V1,
    from: V2,
    result_type: T0,
    location: L,
) -> Result<DetachedCoroPromiseOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(CORO_PROMISE_OPERATION_NAME, location);
    builder = builder.add_operand(handle);
    builder = builder.add_operand(alignment);
    builder = builder.add_operand(from);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_coro_promise`"))
    })
}

/// Canonical MLIR operation name for [`CoroResumeOperation`].
pub const CORO_RESUME_OPERATION_NAME: &str = "llvm.intr.coro.resume";

/// Operation trait for `llvm.intr.coro.resume`.
pub trait CoroResumeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CORO_RESUME_OPERATION_NAME
    }

    /// Returns the `handle` operand.
    fn handle(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(CoroResume);

/// Constructs a new detached `llvm.intr.coro.resume` operation.
pub fn intr_coro_resume<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    handle: V0,
    location: L,
) -> Result<DetachedCoroResumeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(CORO_RESUME_OPERATION_NAME, location);
    builder = builder.add_operand(handle);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_coro_resume`"))
    })
}

/// Canonical MLIR operation name for [`CoroSaveOperation`].
pub const CORO_SAVE_OPERATION_NAME: &str = "llvm.intr.coro.save";

/// Operation trait for `llvm.intr.coro.save`.
pub trait CoroSaveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CORO_SAVE_OPERATION_NAME
    }

    /// Returns the `handle` operand.
    fn handle(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(CoroSave);

/// Constructs a new detached `llvm.intr.coro.save` operation.
pub fn intr_coro_save<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    handle: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedCoroSaveOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(CORO_SAVE_OPERATION_NAME, location);
    builder = builder.add_operand(handle);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_coro_save`"))
    })
}

/// Canonical MLIR operation name for [`CoroSizeOperation`].
pub const CORO_SIZE_OPERATION_NAME: &str = "llvm.intr.coro.size";

/// Operation trait for `llvm.intr.coro.size`.
pub trait CoroSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CORO_SIZE_OPERATION_NAME
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(CoroSize);

/// Constructs a new detached `llvm.intr.coro.size` operation.
pub fn intr_coro_size<'v, 'c: 'v, 't: 'c, T0: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T0,
    location: L,
) -> Result<DetachedCoroSizeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(CORO_SIZE_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_coro_size`"))
    })
}

/// Canonical MLIR operation name for [`CoroSuspendOperation`].
pub const CORO_SUSPEND_OPERATION_NAME: &str = "llvm.intr.coro.suspend";

/// Operation trait for `llvm.intr.coro.suspend`.
pub trait CoroSuspendOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CORO_SUSPEND_OPERATION_NAME
    }

    /// Returns the `save` operand.
    fn save(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `final_suspend` operand.
    fn final_suspend(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(CoroSuspend);

/// Constructs a new detached `llvm.intr.coro.suspend` operation.
pub fn intr_coro_suspend<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    save: V0,
    final_suspend: V1,
    result_type: T0,
    location: L,
) -> Result<DetachedCoroSuspendOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(CORO_SUSPEND_OPERATION_NAME, location);
    builder = builder.add_operand(save);
    builder = builder.add_operand(final_suspend);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_coro_suspend`"))
    })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, DialectHandle, Operation, Type};

    use super::*;

    #[test]
    fn test_intr_coro_align() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = intr_coro_align(i64_type.as_ref(), location).unwrap();
                assert_eq!(op.output_type().unwrap(), i64_type);
                assert_eq!(op.operation_name(), "llvm.intr.coro.align");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_coro_align_test",
                    func::FuncAttributes { arguments: vec![], results: vec![i64_type.into()], ..Default::default() },
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
                  func.func @llvm_intr_coro_align_test() -> i64 {
                    %0 = llvm.intr.coro.align : i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_begin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let token_type = context.token_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(token_type.as_ref(), location), (pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_coro_begin(arg_0, arg_1, pointer_type, location).unwrap();
                assert_eq!(op.token().unwrap(), arg_0);
                assert_eq!(op.memory().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), pointer_type);
                assert_eq!(op.operation_name(), "llvm.intr.coro.begin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_coro_begin_test",
                    func::FuncAttributes {
                        arguments: vec![token_type.into(), pointer_type.into()],
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
                  func.func @llvm_intr_coro_begin_test(%arg0: token, %arg1: !llvm.ptr) -> !llvm.ptr {
                    %0 = llvm.intr.coro.begin %arg0, %arg1 : (token, !llvm.ptr) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_end() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let token_type = context.token_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (pointer_type.as_ref(), location),
                    (i1_type.as_ref(), location),
                    (token_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_coro_end(arg_0, arg_1, arg_2, i1_type, location).unwrap();
                assert_eq!(op.handle().unwrap(), arg_0);
                assert_eq!(op.unwind().unwrap(), arg_1);
                assert_eq!(op.return_values().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), i1_type);
                assert_eq!(op.operation_name(), "llvm.intr.coro.end");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_coro_end_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), i1_type.into(), token_type.into()],
                        results: vec![i1_type.into()],
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
                  func.func @llvm_intr_coro_end_test(%arg0: !llvm.ptr, %arg1: i1, %arg2: token) -> i1 {
                    %0 = llvm.intr.coro.end %arg0, %arg1, %arg2 : (!llvm.ptr, i1, token) -> i1
                    return %0 : i1
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_free() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let token_type = context.token_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(token_type.as_ref(), location), (pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_coro_free(arg_0, arg_1, pointer_type, location).unwrap();
                assert_eq!(op.id().unwrap(), arg_0);
                assert_eq!(op.handle().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), pointer_type);
                assert_eq!(op.operation_name(), "llvm.intr.coro.free");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_coro_free_test",
                    func::FuncAttributes {
                        arguments: vec![token_type.into(), pointer_type.into()],
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
                  func.func @llvm_intr_coro_free_test(%arg0: token, %arg1: !llvm.ptr) -> !llvm.ptr {
                    %0 = llvm.intr.coro.free %arg0, %arg1 : (token, !llvm.ptr) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_id() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let token_type = context.token_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_coro_id(arg_0, arg_1, arg_2, arg_3, token_type, location).unwrap();
                assert_eq!(op.alignment().unwrap(), arg_0);
                assert_eq!(op.promise().unwrap(), arg_1);
                assert_eq!(op.coroutine_address().unwrap(), arg_2);
                assert_eq!(op.function_addresses().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), token_type);
                assert_eq!(op.operation_name(), "llvm.intr.coro.id");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_coro_id_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), pointer_type.into(), pointer_type.into(), pointer_type.into()],
                        results: vec![token_type.into()],
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
                  func.func @llvm_intr_coro_id_test(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) -> token {
                    %0 = llvm.intr.coro.id %arg0, %arg1, %arg2, %arg3 : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> token
                    return %0 : token
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_promise() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (pointer_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                    (i1_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_coro_promise(arg_0, arg_1, arg_2, pointer_type, location).unwrap();
                assert_eq!(op.handle().unwrap(), arg_0);
                assert_eq!(op.alignment().unwrap(), arg_1);
                assert_eq!(op.from().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), pointer_type);
                assert_eq!(op.operation_name(), "llvm.intr.coro.promise");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_coro_promise_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), i32_type.into(), i1_type.into()],
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
                  func.func @llvm_intr_coro_promise_test(%arg0: !llvm.ptr, %arg1: i32, %arg2: i1) -> !llvm.ptr {
                    %0 = llvm.intr.coro.promise %arg0, %arg1, %arg2 : (!llvm.ptr, i32, i1) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_resume() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_coro_resume(arg_0, location).unwrap();
                assert_eq!(op.handle().unwrap(), arg_0);
                assert_eq!(op.operation_name(), "llvm.intr.coro.resume");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_coro_resume_test",
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
                module {
                  func.func @llvm_intr_coro_resume_test(%arg0: !llvm.ptr) {
                    llvm.intr.coro.resume %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_save() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let token_type = context.token_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_coro_save(arg_0, token_type, location).unwrap();
                assert_eq!(op.handle().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), token_type);
                assert_eq!(op.operation_name(), "llvm.intr.coro.save");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_coro_save_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
                        results: vec![token_type.into()],
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
                  func.func @llvm_intr_coro_save_test(%arg0: !llvm.ptr) -> token {
                    %0 = llvm.intr.coro.save %arg0 : (!llvm.ptr) -> token
                    return %0 : token
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_size() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = intr_coro_size(i64_type.as_ref(), location).unwrap();
                assert_eq!(op.output_type().unwrap(), i64_type);
                assert_eq!(op.operation_name(), "llvm.intr.coro.size");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_coro_size_test",
                    func::FuncAttributes { arguments: vec![], results: vec![i64_type.into()], ..Default::default() },
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
                  func.func @llvm_intr_coro_size_test() -> i64 {
                    %0 = llvm.intr.coro.size : i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_suspend() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i8_type = context.signless_integer_type(8);
        let token_type = context.token_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(token_type.as_ref(), location), (i1_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_coro_suspend(arg_0, arg_1, i8_type, location).unwrap();
                assert_eq!(op.save().unwrap(), arg_0);
                assert_eq!(op.final_suspend().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i8_type);
                assert_eq!(op.operation_name(), "llvm.intr.coro.suspend");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_coro_suspend_test",
                    func::FuncAttributes {
                        arguments: vec![token_type.into(), i1_type.into()],
                        results: vec![i8_type.into()],
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
                  func.func @llvm_intr_coro_suspend_test(%arg0: token, %arg1: i1) -> i8 {
                    %0 = llvm.intr.coro.suspend %arg0, %arg1 : i8
                    return %0 : i8
                  }
                }
            "},
        );
    }
}
