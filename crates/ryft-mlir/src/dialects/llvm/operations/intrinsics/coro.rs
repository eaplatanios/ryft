use crate::{
    DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Type, TypeRef, Value, ValueRef, mlir_op,
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
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(CoroAlign);

/// Constructs a new detached `llvm.intr.coro.align` operation.
pub fn intr_coro_align<'v, 'c: 'v, 't: 'c, T0: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T0,
    location: L,
) -> DetachedCoroAlignOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CORO_ALIGN_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_coro_align`")
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
    fn token(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `memory` operand.
    fn memory(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
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
) -> DetachedCoroBeginOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CORO_BEGIN_OPERATION_NAME, location);
    builder = builder.add_operand(token);
    builder = builder.add_operand(memory);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_coro_begin`")
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
    fn handle(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `unwind` operand.
    fn unwind(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `return_values` operand.
    fn return_values(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
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
) -> DetachedCoroEndOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CORO_END_OPERATION_NAME, location);
    builder = builder.add_operand(handle);
    builder = builder.add_operand(unwind);
    builder = builder.add_operand(return_values);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_coro_end`")
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
    fn id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `handle` operand.
    fn handle(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
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
) -> DetachedCoroFreeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CORO_FREE_OPERATION_NAME, location);
    builder = builder.add_operand(id);
    builder = builder.add_operand(handle);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_coro_free`")
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
    fn alignment(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `promise` operand.
    fn promise(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `coroutine_address` operand.
    fn coroutine_address(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `function_addresses` operand.
    fn function_addresses(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
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
) -> DetachedCoroIdOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CORO_ID_OPERATION_NAME, location);
    builder = builder.add_operand(alignment);
    builder = builder.add_operand(promise);
    builder = builder.add_operand(coroutine_address);
    builder = builder.add_operand(function_addresses);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_coro_id`")
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
    fn handle(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `alignment` operand.
    fn alignment(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `from` operand.
    fn from(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
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
) -> DetachedCoroPromiseOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CORO_PROMISE_OPERATION_NAME, location);
    builder = builder.add_operand(handle);
    builder = builder.add_operand(alignment);
    builder = builder.add_operand(from);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_coro_promise`")
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
    fn handle(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(CoroResume);

/// Constructs a new detached `llvm.intr.coro.resume` operation.
pub fn intr_coro_resume<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    handle: V0,
    location: L,
) -> DetachedCoroResumeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CORO_RESUME_OPERATION_NAME, location);
    builder = builder.add_operand(handle);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_coro_resume`")
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
    fn handle(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(CoroSave);

/// Constructs a new detached `llvm.intr.coro.save` operation.
pub fn intr_coro_save<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    handle: V0,
    result_type: T0,
    location: L,
) -> DetachedCoroSaveOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CORO_SAVE_OPERATION_NAME, location);
    builder = builder.add_operand(handle);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_coro_save`")
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
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(CoroSize);

/// Constructs a new detached `llvm.intr.coro.size` operation.
pub fn intr_coro_size<'v, 'c: 'v, 't: 'c, T0: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T0,
    location: L,
) -> DetachedCoroSizeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CORO_SIZE_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_coro_size`")
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
    fn save(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `final_suspend` operand.
    fn final_suspend(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
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
) -> DetachedCoroSuspendOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CORO_SUSPEND_OPERATION_NAME, location);
    builder = builder.add_operand(save);
    builder = builder.add_operand(final_suspend);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_coro_suspend`")
}
