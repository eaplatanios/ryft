use crate::{
    Attribute, AttributeRef, Block, BlockRef, DetachedOp, DialectHandle, FlatSymbolRefAttributeRef,
    IntegerAttributeRef, Location, OneResult, Operation, OperationBuilder, StringRef, Type, TypeAttributeRef, TypeRef,
    Value, ValueRef, mlir_binary_op, mlir_generic_unary_op, mlir_op, mlir_op_trait, mlir_unary_op,
};

/// Name of the attribute that stores an `llvm.mlir.constant` value.
pub const VALUE_ATTRIBUTE: &str = "value";

/// Name of the attribute that stores an LLVM operation's source element type.
pub const ELEMENT_TYPE_ATTRIBUTE: &str = "elem_type";

/// Name of the attribute that stores byte alignment requirements.
pub const ALIGNMENT_ATTRIBUTE: &str = "alignment";

/// Name of the attribute that marks volatile LLVM memory operations.
pub const VOLATILE_ATTRIBUTE: &str = "volatile_";

/// Name of the attribute that marks `llvm.alloca` allocations as `inalloca`.
pub const INALLOCA_ATTRIBUTE: &str = "inalloca";

/// Name of the attribute that stores an `llvm.mlir.addressof` referenced symbol.
pub const GLOBAL_NAME_ATTRIBUTE: &str = "global_name";

mlir_binary_op!(llvm, add);
mlir_binary_op!(llvm, sub);
mlir_binary_op!(llvm, mul);
mlir_binary_op!(llvm, udiv);
mlir_binary_op!(llvm, sdiv);
mlir_binary_op!(llvm, urem);
mlir_binary_op!(llvm, srem);
mlir_binary_op!(llvm, and);
mlir_binary_op!(llvm, or);
mlir_binary_op!(llvm, xor);
mlir_binary_op!(llvm, shl);
mlir_binary_op!(llvm, lshr);
mlir_binary_op!(llvm, ashr);
mlir_binary_op!(llvm, fadd);
mlir_binary_op!(llvm, fsub);
mlir_binary_op!(llvm, fmul);
mlir_binary_op!(llvm, fdiv);
mlir_binary_op!(llvm, frem);

mlir_unary_op!(llvm, fneg);

mlir_generic_unary_op!(llvm, bitcast);
mlir_generic_unary_op!(llvm, addrspacecast);
mlir_generic_unary_op!(llvm, inttoptr);
mlir_generic_unary_op!(llvm, ptrtoint);
mlir_generic_unary_op!(llvm, ptrtoaddr);
mlir_generic_unary_op!(llvm, sext);
mlir_generic_unary_op!(llvm, zext);
mlir_generic_unary_op!(llvm, trunc);
mlir_generic_unary_op!(llvm, sitofp);
mlir_generic_unary_op!(llvm, uitofp);
mlir_generic_unary_op!(llvm, fptosi);
mlir_generic_unary_op!(llvm, fptoui);
mlir_generic_unary_op!(llvm, fpext);
mlir_generic_unary_op!(llvm, fptrunc);

/// Name of the attribute that stores LLVM comparison predicates.
pub const PREDICATE_ATTRIBUTE: &str = "predicate";

/// Operation trait for `llvm.icmp`.
pub trait IcmpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the integer comparison predicate.
    fn predicate(&self) -> AttributeRef<'c, 't> {
        self.attribute(PREDICATE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("invalid '{PREDICATE_ATTRIBUTE}' attribute in `llvm::icmp`"))
    }

    /// Returns the left-hand side operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the right-hand side operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }
}

mlir_op!(Icmp);
mlir_op_trait!(Icmp, OneResult);
mlir_op_trait!(Icmp, ZeroRegions);
mlir_op_trait!(Icmp, ZeroSuccessors);

/// Constructs a new detached [`IcmpOperation`].
pub fn icmp<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    P: Attribute<'c, 't>,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    predicate: P,
    lhs: Lhs,
    rhs: Rhs,
    result_type: T,
    location: L,
) -> DetachedIcmpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    OperationBuilder::new("llvm.icmp", location)
        .add_attribute(PREDICATE_ATTRIBUTE, predicate)
        .add_operand(lhs)
        .add_operand(rhs)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::icmp`")
}

/// Name of the attribute that stores LLVM fast-math flags.
pub const FASTMATH_FLAGS_ATTRIBUTE: &str = "fastmathFlags";

/// Operation trait for `llvm.fcmp`.
pub trait FcmpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the floating-point comparison predicate.
    fn predicate(&self) -> AttributeRef<'c, 't> {
        self.attribute(PREDICATE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("invalid '{PREDICATE_ATTRIBUTE}' attribute in `llvm::fcmp`"))
    }

    /// Returns the left-hand side operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the right-hand side operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional fast-math flags.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute(FASTMATH_FLAGS_ATTRIBUTE)
    }
}

mlir_op!(Fcmp);
mlir_op_trait!(Fcmp, OneResult);
mlir_op_trait!(Fcmp, ZeroRegions);
mlir_op_trait!(Fcmp, ZeroSuccessors);

/// Constructs a new detached [`FcmpOperation`].
pub fn fcmp<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    P: Attribute<'c, 't>,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    predicate: P,
    lhs: Lhs,
    rhs: Rhs,
    result_type: T,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedFcmpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new("llvm.fcmp", location)
        .add_attribute(PREDICATE_ATTRIBUTE, predicate)
        .add_operand(lhs)
        .add_operand(rhs)
        .add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute(FASTMATH_FLAGS_ATTRIBUTE, fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::fcmp`")
}

/// Operation trait for `llvm.select`.
pub trait SelectOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the condition operand.
    fn condition(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the value selected when the condition is true.
    fn true_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the value selected when the condition is false.
    fn false_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }
}

mlir_op!(Select);
mlir_op_trait!(Select, OneResult);
mlir_op_trait!(Select, ZeroRegions);
mlir_op_trait!(Select, ZeroSuccessors);

/// Constructs a new detached [`SelectOperation`].
pub fn select<
    'condition,
    'true_value,
    'false_value,
    'c: 'condition + 'true_value + 'false_value,
    't: 'c,
    C: Value<'condition, 'c, 't>,
    T: Value<'true_value, 'c, 't>,
    F: Value<'false_value, 'c, 't>,
    L: Location<'c, 't>,
>(
    condition: C,
    true_value: T,
    false_value: F,
    location: L,
) -> DetachedSelectOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    OperationBuilder::new("llvm.select", location)
        .add_operand(condition)
        .add_operand(true_value)
        .add_operand(false_value)
        .add_result(true_value.r#type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::select`")
}

/// Operation trait for `llvm.mlir.constant`.
pub trait ConstantOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the constant value attribute.
    fn value(&self) -> AttributeRef<'c, 't> {
        self.attribute(VALUE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("invalid '{VALUE_ATTRIBUTE}' attribute in `llvm::constant`"))
    }
}

mlir_op!(Constant);
mlir_op_trait!(Constant, ConstantLike);
mlir_op_trait!(Constant, OneResult);
mlir_op_trait!(Constant, ZeroOperands);
mlir_op_trait!(Constant, ZeroRegions);
mlir_op_trait!(Constant, ZeroSuccessors);

/// Constructs a new detached [`ConstantOperation`].
pub fn constant<'c, 't: 'c, A: Attribute<'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    value: A,
    result_type: T,
    location: L,
) -> DetachedConstantOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    OperationBuilder::new("llvm.mlir.constant", location)
        .add_attribute(VALUE_ATTRIBUTE, value)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::constant`")
}

/// Operation trait shared by LLVM no-operand constant-like operations that produce one value.
pub trait ValueConstantOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {}

/// Operation trait for `llvm.mlir.undef`.
pub trait UndefOperation<'o, 'c: 'o, 't: 'c>: ValueConstantOperation<'o, 'c, 't> {}

mlir_op!(Undef);
mlir_op_trait!(Undef, ConstantLike);
mlir_op_trait!(Undef, OneResult);
mlir_op_trait!(Undef, ZeroOperands);
mlir_op_trait!(Undef, ZeroRegions);
mlir_op_trait!(Undef, ZeroSuccessors);
mlir_op_trait!(Undef, @local ValueConstantOperation);

/// Constructs a new detached [`UndefOperation`].
pub fn undef<'c, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T,
    location: L,
) -> DetachedUndefOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    OperationBuilder::new("llvm.mlir.undef", location)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::undef`")
}

/// Operation trait for `llvm.mlir.poison`.
pub trait PoisonOperation<'o, 'c: 'o, 't: 'c>: ValueConstantOperation<'o, 'c, 't> {}

mlir_op!(Poison);
mlir_op_trait!(Poison, ConstantLike);
mlir_op_trait!(Poison, OneResult);
mlir_op_trait!(Poison, ZeroOperands);
mlir_op_trait!(Poison, ZeroRegions);
mlir_op_trait!(Poison, ZeroSuccessors);
mlir_op_trait!(Poison, @local ValueConstantOperation);

/// Constructs a new detached [`PoisonOperation`].
pub fn poison<'c, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T,
    location: L,
) -> DetachedPoisonOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    OperationBuilder::new("llvm.mlir.poison", location)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::poison`")
}

/// Operation trait for `llvm.mlir.zero`.
pub trait ZeroOperation<'o, 'c: 'o, 't: 'c>: ValueConstantOperation<'o, 'c, 't> {}

mlir_op!(Zero);
mlir_op_trait!(Zero, ConstantLike);
mlir_op_trait!(Zero, OneResult);
mlir_op_trait!(Zero, ZeroOperands);
mlir_op_trait!(Zero, ZeroRegions);
mlir_op_trait!(Zero, ZeroSuccessors);
mlir_op_trait!(Zero, @local ValueConstantOperation);

/// Constructs a new detached [`ZeroOperation`].
pub fn zero<'c, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T,
    location: L,
) -> DetachedZeroOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    OperationBuilder::new("llvm.mlir.zero", location)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::zero`")
}

/// Operation trait for `llvm.alloca`.
pub trait AllocaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the runtime array size operand.
    fn array_size(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the allocated element type.
    fn element_type(&self) -> TypeRef<'c, 't> {
        self.attribute(ELEMENT_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>())
            .map(|attribute| attribute.r#type())
            .unwrap_or_else(|| panic!("invalid '{ELEMENT_TYPE_ATTRIBUTE}' attribute in `llvm::alloca`"))
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Option<i64> {
        self.attribute(ALIGNMENT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
    }

    /// Returns `true` if this allocation uses LLVM `inalloca` argument-passing storage.
    fn inalloca(&self) -> bool {
        self.has_attribute(INALLOCA_ATTRIBUTE)
    }
}

mlir_op!(Alloca);
mlir_op_trait!(Alloca, OneResult);
mlir_op_trait!(Alloca, ZeroRegions);
mlir_op_trait!(Alloca, ZeroSuccessors);

/// Constructs a new detached [`AllocaOperation`].
pub fn alloca<
    'size,
    'c: 'size,
    't: 'c,
    S: Value<'size, 'c, 't>,
    E: Type<'c, 't>,
    R: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    array_size: S,
    element_type: E,
    result_type: R,
    alignment: Option<i64>,
    inalloca: bool,
    location: L,
) -> DetachedAllocaOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new("llvm.alloca", location)
        .add_operand(array_size)
        .add_attribute(ELEMENT_TYPE_ATTRIBUTE, context.type_attribute(element_type))
        .add_result(result_type);
    if let Some(alignment) = alignment {
        builder = builder.add_attribute(
            ALIGNMENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), alignment),
        );
    }
    if inalloca {
        builder = builder.add_attribute(INALLOCA_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::alloca`")
}

/// Operation trait for `llvm.load`.
pub trait LoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the pointer operand being loaded from.
    fn address(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Option<i64> {
        self.attribute(ALIGNMENT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
    }

    /// Returns `true` if this load is volatile.
    fn is_volatile(&self) -> bool {
        self.has_attribute(VOLATILE_ATTRIBUTE)
    }
}

mlir_op!(Load);
mlir_op_trait!(Load, OneResult);
mlir_op_trait!(Load, ZeroRegions);
mlir_op_trait!(Load, ZeroSuccessors);

/// Constructs a new detached [`LoadOperation`].
pub fn load<'address, 'c: 'address, 't: 'c, A: Value<'address, 'c, 't>, R: Type<'c, 't>, L: Location<'c, 't>>(
    address: A,
    result_type: R,
    alignment: Option<i64>,
    is_volatile: bool,
    location: L,
) -> DetachedLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new("llvm.load", location).add_operand(address).add_result(result_type);
    if let Some(alignment) = alignment {
        builder = builder.add_attribute(
            ALIGNMENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), alignment),
        );
    }
    if is_volatile {
        builder = builder.add_attribute(VOLATILE_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::load`")
}

/// Operation trait for `llvm.store`.
pub trait StoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value being stored.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the pointer operand being stored to.
    fn address(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional byte alignment.
    fn alignment(&self) -> Option<i64> {
        self.attribute(ALIGNMENT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
    }

    /// Returns `true` if this store is volatile.
    fn is_volatile(&self) -> bool {
        self.has_attribute(VOLATILE_ATTRIBUTE)
    }
}

mlir_op!(Store);
mlir_op_trait!(Store, ZeroRegions);
mlir_op_trait!(Store, ZeroSuccessors);

/// Constructs a new detached [`StoreOperation`].
pub fn store<
    'value,
    'address,
    'c: 'value + 'address,
    't: 'c,
    V: Value<'value, 'c, 't>,
    A: Value<'address, 'c, 't>,
    L: Location<'c, 't>,
>(
    value: V,
    address: A,
    alignment: Option<i64>,
    is_volatile: bool,
    location: L,
) -> DetachedStoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new("llvm.store", location).add_operand(value).add_operand(address);
    if let Some(alignment) = alignment {
        builder = builder.add_attribute(
            ALIGNMENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), alignment),
        );
    }
    if is_volatile {
        builder = builder.add_attribute(VOLATILE_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::store`")
}

/// Operation trait for `llvm.return`.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional value being returned.
    fn argument(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(0)
    }
}

mlir_op!(Return);
mlir_op_trait!(Return, IsTerminator);
mlir_op_trait!(Return, ReturnLike);
mlir_op_trait!(Return, SingleBlockRegions);
mlir_op_trait!(Return, ZeroRegions);
mlir_op_trait!(Return, ZeroSuccessors);

/// Constructs a new detached [`ReturnOperation`].
pub fn r#return<'argument, 'c: 'argument, 't: 'c, L: Location<'c, 't>>(
    argument: Option<ValueRef<'argument, 'c, 't>>,
    location: L,
) -> DetachedReturnOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new("llvm.return", location);
    if let Some(argument) = argument {
        builder = builder.add_operand(argument);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::return`")
}

/// Operation trait for `llvm.br`.
pub trait BranchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the destination block.
    fn destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(0).unwrap()
    }

    /// Returns the operands forwarded to the destination block.
    fn destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(Branch);
mlir_op_trait!(Branch, IsTerminator);
mlir_op_trait!(Branch, SingleBlockRegions);
mlir_op_trait!(Branch, ZeroRegions);

/// Constructs a new detached [`BranchOperation`].
pub fn br<'b, 'v, 'c: 'b + 'v, 't: 'c, B: Block<'b, 'c, 't>, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    destination: &B,
    operands: &[V],
    location: L,
) -> DetachedBranchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    OperationBuilder::new("llvm.br", location)
        .add_operands(operands)
        .add_successor(destination)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::br`")
}

/// Operation trait for `llvm.unreachable`.
pub trait UnreachableOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Unreachable);
mlir_op_trait!(Unreachable, IsTerminator);
mlir_op_trait!(Unreachable, SingleBlockRegions);
mlir_op_trait!(Unreachable, ZeroOperands);
mlir_op_trait!(Unreachable, ZeroRegions);
mlir_op_trait!(Unreachable, ZeroSuccessors);

/// Constructs a new detached [`UnreachableOperation`].
pub fn unreachable<'c, 't: 'c, L: Location<'c, 't>>(location: L) -> DetachedUnreachableOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    OperationBuilder::new("llvm.unreachable", location)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::unreachable`")
}

/// Operation trait for `llvm.mlir.addressof`.
pub trait AddressOfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the referenced symbol name.
    fn global_name(&self) -> StringRef<'c> {
        self.attribute(GLOBAL_NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<FlatSymbolRefAttributeRef>())
            .map(|attribute| attribute.reference())
            .unwrap_or_else(|| panic!("invalid '{GLOBAL_NAME_ATTRIBUTE}' attribute in `llvm::address_of`"))
    }
}

mlir_op!(AddressOf);
mlir_op_trait!(AddressOf, ConstantLike);
mlir_op_trait!(AddressOf, OneResult);
mlir_op_trait!(AddressOf, ZeroOperands);
mlir_op_trait!(AddressOf, ZeroRegions);
mlir_op_trait!(AddressOf, ZeroSuccessors);

/// Constructs a new detached [`AddressOfOperation`].
pub fn address_of<'c, 't: 'c, S: Into<StringRef<'c>>, T: Type<'c, 't>, L: Location<'c, 't>>(
    global_name: S,
    result_type: T,
    location: L,
) -> DetachedAddressOfOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    OperationBuilder::new("llvm.mlir.addressof", location)
        .add_attribute(GLOBAL_NAME_ATTRIBUTE, context.flat_symbol_ref_attribute(global_name.into()))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::address_of`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::{
        Block, Context, OneOperand, Operation, OperationBuilder, Region, Type, ValueRef,
        dialects::{func, llvm::attributes::Linkage},
    };

    use super::*;

    macro_rules! llvm_binary_operation_test {
        (
            $test_name:ident,
            $constructor:ident,
            $function_name:literal,
            $value_type:ident($($value_type_arguments:tt)*),
            $expected:literal $(,)?
        ) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let module = context.module(location);
                let value_type = context.$value_type($($value_type_arguments)*);
                module.body().append_operation({
                    let mut block = context.block(&[(value_type, location), (value_type, location)]);
                    let lhs = block.argument(0).unwrap();
                    let rhs = block.argument(1).unwrap();
                    let op = $constructor(lhs, rhs, location);
                    assert_eq!(op.lhs(), lhs);
                    assert_eq!(op.rhs(), rhs);
                    assert_eq!(op.output_type(), value_type);
                    assert_eq!(op.operands().count(), 2);
                    assert_eq!(op.results().count(), 1);
                    assert_eq!(op.regions().count(), 0);
                    assert_eq!(op.successors().count(), 0);
                    let op = block.append_operation(op);
                    block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
                    func::func(
                        $function_name,
                        func::FuncAttributes {
                            arguments: vec![value_type.into(), value_type.into()],
                            results: vec![value_type.into()],
                            ..Default::default()
                        },
                        block.into(),
                        location,
                    )
                });
                assert!(module.verify());
                assert_eq!(module.to_string(), indoc! { $expected });
            }
        };
    }

    macro_rules! llvm_unary_operation_test {
        (
            $test_name:ident,
            $constructor:ident,
            $function_name:literal,
            $value_type:ident($($value_type_arguments:tt)*),
            $expected:literal $(,)?
        ) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let module = context.module(location);
                let value_type = context.$value_type($($value_type_arguments)*);
                module.body().append_operation({
                    let mut block = context.block(&[(value_type, location)]);
                    let input = block.argument(0).unwrap();
                    let op = $constructor(input, location);
                    assert_eq!(op.input(), input);
                    assert_eq!(op.output_type(), value_type);
                    assert_eq!(op.operands().count(), 1);
                    assert_eq!(op.results().count(), 1);
                    assert_eq!(op.regions().count(), 0);
                    assert_eq!(op.successors().count(), 0);
                    let op = block.append_operation(op);
                    block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
                    func::func(
                        $function_name,
                        func::FuncAttributes {
                            arguments: vec![value_type.into()],
                            results: vec![value_type.into()],
                            ..Default::default()
                        },
                        block.into(),
                        location,
                    )
                });
                assert!(module.verify());
                assert_eq!(module.to_string(), indoc! { $expected });
            }
        };
    }

    macro_rules! llvm_generic_unary_operation_test {
        (
            $test_name:ident,
            $constructor:ident,
            $function_name:literal,
            $input_type:ident($($input_type_arguments:tt)*),
            $output_type:ident($($output_type_arguments:tt)*),
            $expected:literal,
        ) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let module = context.module(location);
                let input_type = context.$input_type($($input_type_arguments)*);
                let output_type = context.$output_type($($output_type_arguments)*);
                module.body().append_operation({
                    let mut block = context.block(&[(input_type, location)]);
                    let input = block.argument(0).unwrap();
                    let op = $constructor(input, output_type, location);
                    assert_eq!(op.input(), input);
                    assert_eq!(op.output_type(), output_type);
                    assert_eq!(op.operands().count(), 1);
                    assert_eq!(op.results().count(), 1);
                    assert_eq!(op.regions().count(), 0);
                    assert_eq!(op.successors().count(), 0);
                    let op = block.append_operation(op);
                    block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
                    func::func(
                        $function_name,
                        func::FuncAttributes {
                            arguments: vec![input_type.into()],
                            results: vec![output_type.into()],
                            ..Default::default()
                        },
                        block.into(),
                        location,
                    )
                });
                assert!(module.verify());
                assert_eq!(module.to_string(), indoc! { $expected });
            }
        };
    }

    llvm_binary_operation_test!(
        test_add,
        add,
        "llvm_add_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_add_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.add %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_sub,
        sub,
        "llvm_sub_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_sub_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.sub %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_mul,
        mul,
        "llvm_mul_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_mul_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.mul %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_udiv,
        udiv,
        "llvm_udiv_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_udiv_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.udiv %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_sdiv,
        sdiv,
        "llvm_sdiv_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_sdiv_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.sdiv %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_urem,
        urem,
        "llvm_urem_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_urem_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.urem %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_srem,
        srem,
        "llvm_srem_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_srem_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.srem %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_and,
        and,
        "llvm_and_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_and_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.and %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_or,
        or,
        "llvm_or_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_or_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.or %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_xor,
        xor,
        "llvm_xor_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_xor_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.xor %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_shl,
        shl,
        "llvm_shl_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_shl_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.shl %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_lshr,
        lshr,
        "llvm_lshr_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_lshr_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.lshr %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_ashr,
        ashr,
        "llvm_ashr_test",
        signless_integer_type(32),
        "
        module {
          func.func @llvm_ashr_test(%arg0: i32, %arg1: i32) -> i32 {
            %0 = llvm.ashr %arg0, %arg1 : i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_fadd,
        fadd,
        "llvm_fadd_test",
        float32_type(),
        "
        module {
          func.func @llvm_fadd_test(%arg0: f32, %arg1: f32) -> f32 {
            %0 = llvm.fadd %arg0, %arg1 : f32
            return %0 : f32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_fsub,
        fsub,
        "llvm_fsub_test",
        float32_type(),
        "
        module {
          func.func @llvm_fsub_test(%arg0: f32, %arg1: f32) -> f32 {
            %0 = llvm.fsub %arg0, %arg1 : f32
            return %0 : f32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_fmul,
        fmul,
        "llvm_fmul_test",
        float32_type(),
        "
        module {
          func.func @llvm_fmul_test(%arg0: f32, %arg1: f32) -> f32 {
            %0 = llvm.fmul %arg0, %arg1 : f32
            return %0 : f32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_fdiv,
        fdiv,
        "llvm_fdiv_test",
        float32_type(),
        "
        module {
          func.func @llvm_fdiv_test(%arg0: f32, %arg1: f32) -> f32 {
            %0 = llvm.fdiv %arg0, %arg1 : f32
            return %0 : f32
          }
        }
        ",
    );

    llvm_binary_operation_test!(
        test_frem,
        frem,
        "llvm_frem_test",
        float32_type(),
        "
        module {
          func.func @llvm_frem_test(%arg0: f32, %arg1: f32) -> f32 {
            %0 = llvm.frem %arg0, %arg1 : f32
            return %0 : f32
          }
        }
        ",
    );

    llvm_unary_operation_test!(
        test_fneg,
        fneg,
        "llvm_fneg_test",
        float32_type(),
        "
        module {
          func.func @llvm_fneg_test(%arg0: f32) -> f32 {
            %0 = llvm.fneg %arg0 : f32
            return %0 : f32
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_bitcast,
        bitcast,
        "llvm_bitcast_test",
        signless_integer_type(32),
        float32_type(),
        "
        module {
          func.func @llvm_bitcast_test(%arg0: i32) -> f32 {
            %0 = llvm.bitcast %arg0 : i32 to f32
            return %0 : f32
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_addrspacecast,
        addrspacecast,
        "llvm_addrspacecast_test",
        llvm_pointer_type(0),
        llvm_pointer_type(3),
        "
        module {
          func.func @llvm_addrspacecast_test(%arg0: !llvm.ptr) -> !llvm.ptr<3> {
            %0 = llvm.addrspacecast %arg0 : !llvm.ptr to !llvm.ptr<3>
            return %0 : !llvm.ptr<3>
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_inttoptr,
        inttoptr,
        "llvm_inttoptr_test",
        signless_integer_type(64),
        llvm_pointer_type(0),
        "
        module {
          func.func @llvm_inttoptr_test(%arg0: i64) -> !llvm.ptr {
            %0 = llvm.inttoptr %arg0 : i64 to !llvm.ptr
            return %0 : !llvm.ptr
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_ptrtoint,
        ptrtoint,
        "llvm_ptrtoint_test",
        llvm_pointer_type(0),
        signless_integer_type(64),
        "
        module {
          func.func @llvm_ptrtoint_test(%arg0: !llvm.ptr) -> i64 {
            %0 = llvm.ptrtoint %arg0 : !llvm.ptr to i64
            return %0 : i64
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_ptrtoaddr,
        ptrtoaddr,
        "llvm_ptrtoaddr_test",
        llvm_pointer_type(0),
        signless_integer_type(64),
        "
        module {
          func.func @llvm_ptrtoaddr_test(%arg0: !llvm.ptr) -> i64 {
            %0 = llvm.ptrtoaddr %arg0 : !llvm.ptr to i64
            return %0 : i64
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_sext,
        sext,
        "llvm_sext_test",
        signless_integer_type(32),
        signless_integer_type(64),
        "
        module {
          func.func @llvm_sext_test(%arg0: i32) -> i64 {
            %0 = llvm.sext %arg0 : i32 to i64
            return %0 : i64
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_zext,
        zext,
        "llvm_zext_test",
        signless_integer_type(32),
        signless_integer_type(64),
        "
        module {
          func.func @llvm_zext_test(%arg0: i32) -> i64 {
            %0 = llvm.zext %arg0 : i32 to i64
            return %0 : i64
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_trunc,
        trunc,
        "llvm_trunc_test",
        signless_integer_type(64),
        signless_integer_type(32),
        "
        module {
          func.func @llvm_trunc_test(%arg0: i64) -> i32 {
            %0 = llvm.trunc %arg0 : i64 to i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_sitofp,
        sitofp,
        "llvm_sitofp_test",
        signless_integer_type(32),
        float32_type(),
        "
        module {
          func.func @llvm_sitofp_test(%arg0: i32) -> f32 {
            %0 = llvm.sitofp %arg0 : i32 to f32
            return %0 : f32
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_uitofp,
        uitofp,
        "llvm_uitofp_test",
        signless_integer_type(32),
        float32_type(),
        "
        module {
          func.func @llvm_uitofp_test(%arg0: i32) -> f32 {
            %0 = llvm.uitofp %arg0 : i32 to f32
            return %0 : f32
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_fptosi,
        fptosi,
        "llvm_fptosi_test",
        float32_type(),
        signless_integer_type(32),
        "
        module {
          func.func @llvm_fptosi_test(%arg0: f32) -> i32 {
            %0 = llvm.fptosi %arg0 : f32 to i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_fptoui,
        fptoui,
        "llvm_fptoui_test",
        float32_type(),
        signless_integer_type(32),
        "
        module {
          func.func @llvm_fptoui_test(%arg0: f32) -> i32 {
            %0 = llvm.fptoui %arg0 : f32 to i32
            return %0 : i32
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_fpext,
        fpext,
        "llvm_fpext_test",
        float32_type(),
        float64_type(),
        "
        module {
          func.func @llvm_fpext_test(%arg0: f32) -> f64 {
            %0 = llvm.fpext %arg0 : f32 to f64
            return %0 : f64
          }
        }
        ",
    );

    llvm_generic_unary_operation_test!(
        test_fptrunc,
        fptrunc,
        "llvm_fptrunc_test",
        float64_type(),
        float32_type(),
        "
        module {
          func.func @llvm_fptrunc_test(%arg0: f64) -> f32 {
            %0 = llvm.fptrunc %arg0 : f64 to f32
            return %0 : f32
          }
        }
        ",
    );

    #[test]
    fn test_icmp() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        context.load_dialect(DialectHandle::llvm());
        let input_type = context.signless_integer_type(32);
        let output_type = context.signless_integer_type(1);
        let predicate = context.integer_attribute(context.signless_integer_type(64), 4);
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (input_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = icmp(predicate, lhs, rhs, output_type, location);
            assert_eq!(op.predicate(), predicate);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output_type(), output_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_icmp_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), input_type.into()],
                    results: vec![output_type.into()],
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
              func.func @llvm_icmp_test(%arg0: i32, %arg1: i32) -> i1 {
                %0 = llvm.icmp \"sgt\" %arg0, %arg1 : i32
                return %0 : i1
              }
            }
            "}
        );
    }

    #[test]
    fn test_fcmp() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        context.load_dialect(DialectHandle::llvm());
        let input_type = context.float32_type();
        let output_type = context.signless_integer_type(1);
        let predicate = context.integer_attribute(context.signless_integer_type(64), 2);
        let fastmath_flags = context.parse_attribute("#llvm.fastmath<none>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (input_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = fcmp(predicate, lhs, rhs, output_type, Some(fastmath_flags), location);
            assert_eq!(op.predicate(), predicate);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.fastmath_flags().unwrap(), fastmath_flags);
            assert_eq!(op.output_type(), output_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_fcmp_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), input_type.into()],
                    results: vec![output_type.into()],
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
              func.func @llvm_fcmp_test(%arg0: f32, %arg1: f32) -> i1 {
                %0 = llvm.fcmp \"ogt\" %arg0, %arg1 : f32
                return %0 : i1
              }
            }
            "}
        );
    }

    #[test]
    fn test_select() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i1_type, location), (i32_type, location), (i32_type, location)]);
            let condition = block.argument(0).unwrap();
            let true_value = block.argument(1).unwrap();
            let false_value = block.argument(2).unwrap();
            let op = select(condition, true_value, false_value, location);
            assert_eq!(op.condition(), condition);
            assert_eq!(op.true_value(), true_value);
            assert_eq!(op.false_value(), false_value);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_select_test",
                func::FuncAttributes {
                    arguments: vec![i1_type.into(), i32_type.into(), i32_type.into()],
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
                  func.func @llvm_select_test(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
                    %0 = llvm.select %arg0, %arg1, %arg2 : i1, i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_constant() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = constant(context.integer_attribute(i32_type, 42), i32_type, location);
            assert_eq!(op.value(), context.integer_attribute(i32_type, 42));
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_constant_test",
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
                  func.func @llvm_constant_test() -> i32 {
                    %0 = llvm.mlir.constant(42 : i32) : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_undef() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = undef(i32_type, location);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_undef_test",
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
                  func.func @llvm_undef_test() -> i32 {
                    %0 = llvm.mlir.undef : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_poison() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = poison(i32_type, location);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_poison_test",
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
                  func.func @llvm_poison_test() -> i32 {
                    %0 = llvm.mlir.poison : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_zero() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = zero(i32_type, location);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_zero_test",
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
                  func.func @llvm_zero_test() -> i32 {
                    %0 = llvm.mlir.zero : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_alloca() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location)]);
            let array_size = block.argument(0).unwrap();
            let op = alloca(array_size, i32_type, pointer_type, Some(16), true, location);
            assert_eq!(op.array_size(), array_size);
            assert_eq!(op.element_type(), i32_type);
            assert_eq!(op.alignment(), Some(16));
            assert!(op.inalloca());
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_alloca_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
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
                  func.func @llvm_alloca_test(%arg0: i32) -> !llvm.ptr {
                    %0 = llvm.alloca inalloca %arg0 x i32 {alignment = 16 : i64} : (i32) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_load() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type, location)]);
            let pointer = block.argument(0).unwrap();
            let op = load(pointer, i32_type, Some(4), true, location);
            assert_eq!(op.address(), pointer);
            assert_eq!(op.alignment(), Some(4));
            assert!(op.is_volatile());
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_load_test",
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
                  func.func @llvm_load_test(%arg0: !llvm.ptr) -> i32 {
                    %0 = llvm.load volatile %arg0 {alignment = 4 : i64} : !llvm.ptr -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_store() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (pointer_type.as_ref(), location)]);
            let value = block.argument(0).unwrap();
            let pointer = block.argument(1).unwrap();
            let op = store(value, pointer, Some(4), true, location);
            assert_eq!(op.value(), value);
            assert_eq!(op.address(), pointer);
            assert_eq!(op.alignment(), Some(4));
            assert!(op.is_volatile());
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 0);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<ValueRef, _>(&[], location));
            func::func(
                "llvm_store_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), pointer_type.into()],
                    results: vec![],
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
                  func.func @llvm_store_test(%arg0: i32, %arg1: !llvm.ptr) {
                    llvm.store volatile %arg0, %arg1 {alignment = 4 : i64} : i32, !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_return() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location)]);
            let argument = block.argument(0).unwrap();
            let op = r#return(Some(argument.into()), location);
            assert_eq!(op.argument(), Some(argument.into()));
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
            block.append_operation(op);
            func::func(
                "llvm_return_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
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
                  func.func @llvm_return_test(%arg0: i32) -> i32 {
                    llvm.return %arg0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_br() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut entry_block = context.block(&[(i32_type, location)]);
            let mut target_block = context.block(&[(i32_type, location)]);
            let argument = entry_block.argument(0).unwrap();
            let op = br(&target_block, &[argument], location);
            assert_eq!(op.destination(), BlockRef::from(&target_block));
            assert_eq!(op.destination_operands(), vec![argument]);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 1);
            entry_block.append_operation(op);
            target_block.append_operation(func::r#return(&[target_block.argument(0).unwrap()], location));
            let mut region = context.region();
            region.append_block(entry_block);
            region.append_block(target_block);
            func::func(
                "llvm_br_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
                    results: vec![i32_type.into()],
                    ..Default::default()
                },
                region,
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_br_test(%arg0: i32) -> i32 {
                    llvm.br ^bb1(%arg0 : i32)
                  ^bb1(%0: i32):  // pred: ^bb0
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_unreachable() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = unreachable(location);
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 0);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
            block.append_operation(op);
            func::func(
                "llvm_unreachable_test",
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
                  func.func @llvm_unreachable_test() {
                    llvm.unreachable
                  }
                }
            "},
        );
    }

    #[test]
    fn test_address_of() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let input_types: [TypeRef; 0] = [];
        module.body().append_operation(
            OperationBuilder::new("llvm.func", location)
                .add_attribute("sym_name", context.string_attribute("global"))
                .add_attribute(
                    "function_type",
                    context.type_attribute(context.llvm_function_type(pointer_type, &input_types, false)),
                )
                .add_attribute("linkage", context.llvm_linkage_attribute(Linkage::External))
                .add_region(context.region())
                .build()
                .expect("invalid `llvm.func` declaration"),
        );
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = address_of("global", pointer_type, location);
            assert_eq!(op.global_name().as_str(), Ok("global"));
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_address_of_test",
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
                  llvm.func @global() -> !llvm.ptr
                  func.func @llvm_address_of_test() -> !llvm.ptr {
                    %0 = llvm.mlir.addressof @global : !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }
}
