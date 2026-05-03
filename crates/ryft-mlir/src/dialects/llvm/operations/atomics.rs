use crate::{
    AttributeRef, DetachedOp, DialectHandle, Location, Operation, OperationBuilder, TypeRef, Value, ValueRef, mlir_op,
};

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

    /// Returns the `success_ordering` attribute.
    fn success_ordering(&self) -> AttributeRef<'c, 't> {
        self.attribute("success_ordering").unwrap()
    }

    /// Returns the `failure_ordering` attribute.
    fn failure_ordering(&self) -> AttributeRef<'c, 't> {
        self.attribute("failure_ordering").unwrap()
    }

    /// Returns the optional `syncscope` attribute.
    fn syncscope(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("syncscope")
    }

    /// Returns the optional `alignment` attribute.
    fn alignment(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("alignment")
    }

    /// Returns whether the `weak` unit attribute is present.
    fn weak(&self) -> bool {
        self.has_attribute("weak")
    }

    /// Returns whether the `volatile_` unit attribute is present.
    fn is_volatile(&self) -> bool {
        self.has_attribute("volatile_")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(AtomicCmpXchg);

/// Constructs a new detached `llvm.cmpxchg` operation.
pub fn cmp_xchg<
    'c,
    't: 'c,
    V1: Value<'c, 'c, 't>,
    V2: Value<'c, 'c, 't>,
    V3: Value<'c, 'c, 't>,
    L: Location<'c, 't>,
>(
    pointer: V1,
    compare_value: V2,
    new_value: V3,
    result_type: TypeRef<'c, 't>,
    success_ordering: AttributeRef<'c, 't>,
    failure_ordering: AttributeRef<'c, 't>,
    syncscope: Option<AttributeRef<'c, 't>>,
    alignment: Option<AttributeRef<'c, 't>>,
    weak: bool,
    is_volatile: bool,
    location: L,
) -> DetachedAtomicCmpXchgOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ATOMIC_CMP_XCHG_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(compare_value);
    builder = builder.add_operand(new_value);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("success_ordering", success_ordering);
    builder = builder.add_attribute("failure_ordering", failure_ordering);
    if let Some(syncscope) = syncscope {
        builder = builder.add_attribute("syncscope", syncscope);
    }
    if let Some(alignment) = alignment {
        builder = builder.add_attribute("alignment", alignment);
    }
    if weak {
        builder = builder.add_attribute("weak", context.unit_attribute());
    }
    if is_volatile {
        builder = builder.add_attribute("volatile_", context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::cmp_xchg`")
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

    /// Returns the `bin_op` attribute.
    fn bin_op(&self) -> AttributeRef<'c, 't> {
        self.attribute("bin_op").unwrap()
    }

    /// Returns the `ordering` attribute.
    fn ordering(&self) -> AttributeRef<'c, 't> {
        self.attribute("ordering").unwrap()
    }

    /// Returns the optional `syncscope` attribute.
    fn syncscope(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("syncscope")
    }

    /// Returns the optional `alignment` attribute.
    fn alignment(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("alignment")
    }

    /// Returns whether the `volatile_` unit attribute is present.
    fn is_volatile(&self) -> bool {
        self.has_attribute("volatile_")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(AtomicRmw);

/// Constructs a new detached `llvm.atomicrmw` operation.
pub fn atomic_rmw<'c, 't: 'c, V1: Value<'c, 'c, 't>, V2: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    pointer: V1,
    value: V2,
    result_type: TypeRef<'c, 't>,
    bin_op: AttributeRef<'c, 't>,
    ordering: AttributeRef<'c, 't>,
    syncscope: Option<AttributeRef<'c, 't>>,
    alignment: Option<AttributeRef<'c, 't>>,
    is_volatile: bool,
    location: L,
) -> DetachedAtomicRmwOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ATOMIC_RMW_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("bin_op", bin_op);
    builder = builder.add_attribute("ordering", ordering);
    if let Some(syncscope) = syncscope {
        builder = builder.add_attribute("syncscope", syncscope);
    }
    if let Some(alignment) = alignment {
        builder = builder.add_attribute("alignment", alignment);
    }
    if is_volatile {
        builder = builder.add_attribute("volatile_", context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::atomic_rmw`")
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Attribute, Block, Context, Operation, Type, ValueRef};

    use super::*;

    #[test]
    fn test_cmp_xchg() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        let result_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false);
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (i32_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let op = cmp_xchg(
                block.argument(0).unwrap(),
                block.argument(1).unwrap(),
                block.argument(2).unwrap(),
                result_type.as_ref(),
                context.integer_attribute(context.signless_integer_type(64), 4).as_ref(),
                context.integer_attribute(context.signless_integer_type(64), 2).as_ref(),
                Some(context.string_attribute("singlethread").as_ref()),
                Some(context.integer_attribute(context.signless_integer_type(64), 4).as_ref()),
                true,
                true,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.cmpxchg");
            assert_eq!(op.pointer(), block.argument(0).unwrap());
            assert_eq!(op.compare_value(), block.argument(1).unwrap());
            assert_eq!(op.new_value(), block.argument(2).unwrap());
            assert_eq!(op.output_type(), result_type);
            assert!(op.weak());
            assert!(op.is_volatile());
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_cmpxchg_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i32_type.into(), i32_type.into()],
                    results: vec![result_type.into()],
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
                  func.func @llvm_cmpxchg_test(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = llvm.cmpxchg weak volatile %arg0, %arg1, %arg2 syncscope(\"singlethread\") acquire monotonic {alignment = 4 : i64} : !llvm.ptr, i32
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_atomic_rmw() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let op = atomic_rmw(
                block.argument(0).unwrap(),
                block.argument(1).unwrap(),
                i32_type.as_ref(),
                context.integer_attribute(context.signless_integer_type(64), 1).as_ref(),
                context.integer_attribute(context.signless_integer_type(64), 2).as_ref(),
                None,
                Some(context.integer_attribute(context.signless_integer_type(64), 4).as_ref()),
                true,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.atomicrmw");
            assert_eq!(op.pointer(), block.argument(0).unwrap());
            assert_eq!(op.value(), block.argument(1).unwrap());
            assert_eq!(op.output_type(), i32_type);
            assert!(op.is_volatile());
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_atomicrmw_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i32_type.into()],
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
                  func.func @llvm_atomicrmw_test(%arg0: !llvm.ptr, %arg1: i32) -> i32 {
                    %0 = llvm.atomicrmw volatile add %arg0, %arg1 monotonic {alignment = 4 : i64} : !llvm.ptr, i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_fence() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = fence(
                context.integer_attribute(context.signless_integer_type(64), 7).as_ref(),
                Some(context.string_attribute("singlethread").as_ref()),
                location,
            );
            assert_eq!(op.operation_name(), "llvm.fence");
            assert!(op.syncscope().is_some());
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<ValueRef, _>(&[], location));
            func::func(
                "llvm_fence_test",
                func::FuncAttributes { arguments: vec![], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_fence_test() {
                    llvm.fence syncscope(\"singlethread\") seq_cst
                    return
                  }
                }
            "},
        );
    }
}
