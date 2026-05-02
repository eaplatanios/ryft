use crate::{
    ArrayAttributeRef, Attribute, AttributeRef, DenseInteger64ArrayAttributeRef, DetachedOp, DetachedRegion,
    DialectHandle, FlatSymbolRefAttributeRef, FunctionTypeRef, IntoWithContext, Location, Operation, OperationBuilder,
    RegionRef, StringAttributeRef, StringRef, Type, TypeAttributeRef, TypeRef, Value, ValueRef, mlir_op, mlir_op_trait,
};

use super::{CmpPredicate, CmpPredicateAttributeRef};

/// Name of the `emitc.file` id attribute.
pub const FILE_ID_ATTRIBUTE: &str = "id";

/// Operation trait for `emitc.file`.
pub trait FileOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the file identifier used by `mlir-translate-file-id`.
    fn id(&self) -> StringRef<'c> {
        self.attribute(FILE_ID_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{FILE_ID_ATTRIBUTE}' attribute in `emitc::file`"))
    }

    /// Returns the file body region.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(File);
mlir_op_trait!(File, IsolatedFromAbove);
mlir_op_trait!(File, NoRegionArguments);
mlir_op_trait!(File, OneRegion);
mlir_op_trait!(File, SymbolTable);
mlir_op_trait!(File, ZeroSuccessors);

/// Constructs a new detached [`FileOperation`].
pub fn file<'c, 't: 'c, I: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    id: I,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedFileOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.file", location)
        .add_attribute(FILE_ID_ATTRIBUTE, id.into_with_context(context))
        .add_region(body)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::file`")
}

/// Common API for single-operand Emit-C expression operations.
pub trait UnaryExpressionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns this operation's operand.
    fn operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

/// Common API for two-operand Emit-C expression operations.
pub trait BinaryExpressionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand-side operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the right-hand-side operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

/// Operation trait for `emitc.address_of`.
pub trait AddressOfOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(AddressOf);
mlir_op_trait!(AddressOf, OneOperand);
mlir_op_trait!(AddressOf, OneResult);
mlir_op_trait!(AddressOf, ZeroRegions);
mlir_op_trait!(AddressOf, ZeroSuccessors);
mlir_op_trait!(AddressOf, @local UnaryExpressionOperation);

/// Constructs a new detached [`AddressOfOperation`].
pub fn address_of<
    'reference,
    'c: 'reference,
    't: 'c,
    Reference: Value<'reference, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    reference: Reference,
    result_type: ResultType,
    location: L,
) -> DetachedAddressOfOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.address_of", location)
        .add_operand(reference)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::address_of`")
}

/// Operation trait for `emitc.add`.
pub trait AddOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(Add);
mlir_op_trait!(Add, OneResult);
mlir_op_trait!(Add, ZeroRegions);
mlir_op_trait!(Add, ZeroSuccessors);
mlir_op_trait!(Add, @local BinaryExpressionOperation);

/// Constructs a new detached [`AddOperation`].
pub fn add<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    result_type: ResultType,
    location: L,
) -> DetachedAddOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.add", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::add`")
}

/// Name of the `emitc.apply` operator attribute.
pub const APPLICABLE_OPERATOR_ATTRIBUTE: &str = "applicableOperator";

/// Operation trait for `emitc.apply`.
pub trait ApplyOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {
    /// Returns the operator applied by this deprecated operation.
    fn applicable_operator(&self) -> StringRef<'c> {
        self.attribute(APPLICABLE_OPERATOR_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{APPLICABLE_OPERATOR_ATTRIBUTE}' attribute in `emitc::apply`"))
    }
}

mlir_op!(Apply);
mlir_op_trait!(Apply, OneOperand);
mlir_op_trait!(Apply, OneResult);
mlir_op_trait!(Apply, ZeroRegions);
mlir_op_trait!(Apply, ZeroSuccessors);
mlir_op_trait!(Apply, @local UnaryExpressionOperation);

/// Constructs a new detached [`ApplyOperation`].
pub fn apply<
    'operand,
    'c: 'operand,
    't: 'c,
    Operand: Value<'operand, 'c, 't>,
    ResultType: Type<'c, 't>,
    O: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    applicable_operator: O,
    operand: Operand,
    result_type: ResultType,
    location: L,
) -> DetachedApplyOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.apply", location)
        .add_attribute(APPLICABLE_OPERATOR_ATTRIBUTE, applicable_operator.into_with_context(context))
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::apply`")
}

/// Operation trait for `emitc.bitwise_and`.
pub trait BitwiseAndOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(BitwiseAnd);
mlir_op_trait!(BitwiseAnd, OneResult);
mlir_op_trait!(BitwiseAnd, ZeroRegions);
mlir_op_trait!(BitwiseAnd, ZeroSuccessors);
mlir_op_trait!(BitwiseAnd, @local BinaryExpressionOperation);

/// Constructs a new detached [`BitwiseAndOperation`].
pub fn bitwise_and<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    result_type: ResultType,
    location: L,
) -> DetachedBitwiseAndOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.bitwise_and", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::bitwise_and`")
}

/// Operation trait for `emitc.bitwise_left_shift`.
pub trait BitwiseLeftShiftOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(BitwiseLeftShift);
mlir_op_trait!(BitwiseLeftShift, OneResult);
mlir_op_trait!(BitwiseLeftShift, ZeroRegions);
mlir_op_trait!(BitwiseLeftShift, ZeroSuccessors);
mlir_op_trait!(BitwiseLeftShift, @local BinaryExpressionOperation);

/// Constructs a new detached [`BitwiseLeftShiftOperation`].
pub fn bitwise_left_shift<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    result_type: ResultType,
    location: L,
) -> DetachedBitwiseLeftShiftOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.bitwise_left_shift", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::bitwise_left_shift`")
}

/// Operation trait for `emitc.bitwise_not`.
pub trait BitwiseNotOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(BitwiseNot);
mlir_op_trait!(BitwiseNot, OneOperand);
mlir_op_trait!(BitwiseNot, OneResult);
mlir_op_trait!(BitwiseNot, ZeroRegions);
mlir_op_trait!(BitwiseNot, ZeroSuccessors);
mlir_op_trait!(BitwiseNot, @local UnaryExpressionOperation);

/// Constructs a new detached [`BitwiseNotOperation`].
pub fn bitwise_not<
    'operand,
    'c: 'operand,
    't: 'c,
    Operand: Value<'operand, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    operand: Operand,
    result_type: ResultType,
    location: L,
) -> DetachedBitwiseNotOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.bitwise_not", location)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::bitwise_not`")
}

/// Operation trait for `emitc.bitwise_or`.
pub trait BitwiseOrOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(BitwiseOr);
mlir_op_trait!(BitwiseOr, OneResult);
mlir_op_trait!(BitwiseOr, ZeroRegions);
mlir_op_trait!(BitwiseOr, ZeroSuccessors);
mlir_op_trait!(BitwiseOr, @local BinaryExpressionOperation);

/// Constructs a new detached [`BitwiseOrOperation`].
pub fn bitwise_or<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    result_type: ResultType,
    location: L,
) -> DetachedBitwiseOrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.bitwise_or", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::bitwise_or`")
}

/// Operation trait for `emitc.bitwise_right_shift`.
pub trait BitwiseRightShiftOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(BitwiseRightShift);
mlir_op_trait!(BitwiseRightShift, OneResult);
mlir_op_trait!(BitwiseRightShift, ZeroRegions);
mlir_op_trait!(BitwiseRightShift, ZeroSuccessors);
mlir_op_trait!(BitwiseRightShift, @local BinaryExpressionOperation);

/// Constructs a new detached [`BitwiseRightShiftOperation`].
pub fn bitwise_right_shift<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    result_type: ResultType,
    location: L,
) -> DetachedBitwiseRightShiftOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.bitwise_right_shift", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::bitwise_right_shift`")
}

/// Operation trait for `emitc.bitwise_xor`.
pub trait BitwiseXorOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(BitwiseXor);
mlir_op_trait!(BitwiseXor, OneResult);
mlir_op_trait!(BitwiseXor, ZeroRegions);
mlir_op_trait!(BitwiseXor, ZeroSuccessors);
mlir_op_trait!(BitwiseXor, @local BinaryExpressionOperation);

/// Constructs a new detached [`BitwiseXorOperation`].
pub fn bitwise_xor<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    result_type: ResultType,
    location: L,
) -> DetachedBitwiseXorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.bitwise_xor", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::bitwise_xor`")
}

/// Name of the `emitc.call_opaque` callee attribute.
pub const CALLEE_ATTRIBUTE: &str = "callee";

/// Name of the `emitc.call_opaque` argument-order attribute.
pub const ARGS_ATTRIBUTE: &str = "args";

/// Name of the `emitc.call_opaque` template-arguments attribute.
pub const TEMPLATE_ARGS_ATTRIBUTE: &str = "template_args";

/// Operation trait for `emitc.call_opaque`.
pub trait CallOpaqueOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the opaque callee name.
    fn callee(&self) -> StringRef<'c> {
        self.attribute(CALLEE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{CALLEE_ATTRIBUTE}' attribute in `emitc::call_opaque`"))
    }

    /// Returns the optional argument-order attribute.
    fn args(&self) -> Option<ArrayAttributeRef<'c, 't>> {
        self.attribute(ARGS_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the optional template-arguments attribute.
    fn template_args(&self) -> Option<ArrayAttributeRef<'c, 't>> {
        self.attribute(TEMPLATE_ARGS_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the call operands.
    fn arguments(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the call results.
    fn outputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.results().map(|result| result.as_ref()).collect()
    }
}

mlir_op!(CallOpaque);
mlir_op_trait!(CallOpaque, ZeroRegions);
mlir_op_trait!(CallOpaque, ZeroSuccessors);

/// Constructs a new detached [`CallOpaqueOperation`].
pub fn call_opaque<
    'operand,
    'c: 'operand,
    't: 'c,
    C: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    callee: C,
    operands: &[ValueRef<'operand, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    args: Option<ArrayAttributeRef<'c, 't>>,
    template_args: Option<ArrayAttributeRef<'c, 't>>,
    location: L,
) -> DetachedCallOpaqueOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let mut builder = OperationBuilder::new("emitc.call_opaque", location)
        .add_attribute(CALLEE_ATTRIBUTE, callee.into_with_context(context))
        .add_operands(operands)
        .add_results(result_types);
    if let Some(args) = args {
        builder = builder.add_attribute(ARGS_ATTRIBUTE, args);
    }
    if let Some(template_args) = template_args {
        builder = builder.add_attribute(TEMPLATE_ARGS_ATTRIBUTE, template_args);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::call_opaque`")
}

/// Operation trait for `emitc.cast`.
pub trait CastOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(Cast);
mlir_op_trait!(Cast, OneOperand);
mlir_op_trait!(Cast, OneResult);
mlir_op_trait!(Cast, ZeroRegions);
mlir_op_trait!(Cast, ZeroSuccessors);
mlir_op_trait!(Cast, @local UnaryExpressionOperation);

/// Constructs a new detached [`CastOperation`].
pub fn cast<
    'source,
    'c: 'source,
    't: 'c,
    Source: Value<'source, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    source: Source,
    result_type: ResultType,
    location: L,
) -> DetachedCastOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.cast", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::cast`")
}

/// Name of the `emitc.cmp` predicate attribute.
pub const CMP_PREDICATE_ATTRIBUTE: &str = "predicate";

/// Operation trait for `emitc.cmp`.
pub trait CmpOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {
    /// Returns the comparison predicate.
    fn predicate(&self) -> CmpPredicate {
        self.attribute(CMP_PREDICATE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<CmpPredicateAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| panic!("invalid '{CMP_PREDICATE_ATTRIBUTE}' attribute in `emitc::cmp`"))
    }
}

mlir_op!(Cmp);
mlir_op_trait!(Cmp, OneResult);
mlir_op_trait!(Cmp, ZeroRegions);
mlir_op_trait!(Cmp, ZeroSuccessors);
mlir_op_trait!(Cmp, @local BinaryExpressionOperation);

/// Constructs a new detached [`CmpOperation`].
pub fn cmp<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    predicate: CmpPredicate,
    lhs: Lhs,
    rhs: Rhs,
    result_type: ResultType,
    location: L,
) -> DetachedCmpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.cmp", location)
        .add_attribute(CMP_PREDICATE_ATTRIBUTE, context.emit_c_cmp_predicate_attribute(predicate))
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::cmp`")
}

/// Name of the `emitc.constant` and `emitc.variable` value attribute.
pub const VALUE_ATTRIBUTE: &str = "value";

/// Operation trait for `emitc.constant`.
pub trait ConstantOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the constant value attribute.
    fn value(&self) -> AttributeRef<'c, 't> {
        self.attribute(VALUE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("invalid '{VALUE_ATTRIBUTE}' attribute in `emitc::constant`"))
    }

    /// Returns this operation's result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(Constant);
mlir_op_trait!(Constant, ConstantLike);
mlir_op_trait!(Constant, OneResult);
mlir_op_trait!(Constant, ZeroOperands);
mlir_op_trait!(Constant, ZeroRegions);
mlir_op_trait!(Constant, ZeroSuccessors);

/// Constructs a new detached [`ConstantOperation`].
pub fn constant<'c, 't: 'c, A: Attribute<'c, 't>, ResultType: Type<'c, 't>, L: Location<'c, 't>>(
    value: A,
    result_type: ResultType,
    location: L,
) -> DetachedConstantOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.constant", location)
        .add_attribute(VALUE_ATTRIBUTE, value)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::constant`")
}

/// Operation trait for `emitc.dereference`.
pub trait DereferenceOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(Dereference);
mlir_op_trait!(Dereference, OneOperand);
mlir_op_trait!(Dereference, OneResult);
mlir_op_trait!(Dereference, ZeroRegions);
mlir_op_trait!(Dereference, ZeroSuccessors);
mlir_op_trait!(Dereference, @local UnaryExpressionOperation);

/// Constructs a new detached [`DereferenceOperation`].
pub fn dereference<
    'pointer,
    'c: 'pointer,
    't: 'c,
    Pointer: Value<'pointer, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    pointer: Pointer,
    result_type: ResultType,
    location: L,
) -> DetachedDereferenceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.dereference", location)
        .add_operand(pointer)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::dereference`")
}

/// Operation trait for `emitc.div`.
pub trait DivOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(Div);
mlir_op_trait!(Div, OneResult);
mlir_op_trait!(Div, ZeroRegions);
mlir_op_trait!(Div, ZeroSuccessors);
mlir_op_trait!(Div, @local BinaryExpressionOperation);

/// Constructs a new detached [`DivOperation`].
pub fn div<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    result_type: ResultType,
    location: L,
) -> DetachedDivOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.div", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::div`")
}

/// Name of the `emitc.expression` no-inline attribute.
pub const DO_NOT_INLINE_ATTRIBUTE: &str = "do_not_inline";

/// Operation trait for `emitc.expression`.
pub trait ExpressionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the operands passed as block arguments to the expression body.
    fn definitions(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns whether this expression must not be emitted inline at its use.
    fn do_not_inline(&self) -> bool {
        self.has_attribute(DO_NOT_INLINE_ATTRIBUTE)
    }

    /// Returns this operation's result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the expression body region.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(Expression);
mlir_op_trait!(Expression, HasOnlyGraphRegion);
mlir_op_trait!(Expression, IsolatedFromAbove);
mlir_op_trait!(Expression, OneRegion);
mlir_op_trait!(Expression, OneResult);
mlir_op_trait!(Expression, ZeroSuccessors);

/// Constructs a new detached [`ExpressionOperation`].
pub fn expression<'definition, 'c: 'definition, 't: 'c, L: Location<'c, 't>>(
    definitions: &[ValueRef<'definition, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    region: DetachedRegion<'c, 't>,
    do_not_inline: bool,
    location: L,
) -> DetachedExpressionOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let mut builder = OperationBuilder::new("emitc.expression", location)
        .add_operands(definitions)
        .add_result(result_type)
        .add_region(region);
    if do_not_inline {
        builder = builder.add_attribute(DO_NOT_INLINE_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::expression`")
}

/// Operation trait for `emitc.for`.
pub trait ForOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the lower bound.
    fn lower_bound(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the upper bound.
    fn upper_bound(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the loop step.
    fn step(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the loop body region.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(For);
mlir_op_trait!(For, OneRegion);
mlir_op_trait!(For, ZeroSuccessors);

/// Constructs a new detached [`ForOperation`].
pub fn r#for<
    'lower,
    'upper,
    'step,
    'c: 'lower + 'upper + 'step,
    't: 'c,
    Lower: Value<'lower, 'c, 't>,
    Upper: Value<'upper, 'c, 't>,
    Step: Value<'step, 'c, 't>,
    L: Location<'c, 't>,
>(
    lower_bound: Lower,
    upper_bound: Upper,
    step: Step,
    region: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedForOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.for", location)
        .add_operands(&[lower_bound.as_ref(), upper_bound.as_ref(), step.as_ref()])
        .add_region(region)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::for`")
}

/// Operation trait for `emitc.call`.
pub trait CallOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the callee symbol.
    fn callee(&self) -> FlatSymbolRefAttributeRef<'c, 't> {
        self.attribute(CALLEE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast())
            .unwrap_or_else(|| panic!("invalid '{CALLEE_ATTRIBUTE}' attribute in `emitc::call`"))
    }

    /// Returns the call operands.
    fn arguments(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the call results.
    fn outputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.results().map(|result| result.as_ref()).collect()
    }
}

mlir_op!(Call);
mlir_op_trait!(Call, ZeroRegions);
mlir_op_trait!(Call, ZeroSuccessors);

/// Constructs a new detached [`CallOperation`].
pub fn call<
    'operand,
    'c: 'operand,
    't: 'c,
    C: IntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    callee: C,
    operands: &[ValueRef<'operand, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    arg_attrs: Option<ArrayAttributeRef<'c, 't>>,
    res_attrs: Option<ArrayAttributeRef<'c, 't>>,
    location: L,
) -> DetachedCallOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let mut builder = OperationBuilder::new("emitc.call", location)
        .add_attribute(CALLEE_ATTRIBUTE, callee.into_with_context(context))
        .add_operands(operands)
        .add_results(result_types);
    if let Some(arg_attrs) = arg_attrs {
        builder = builder.add_attribute(ARG_ATTRS_ATTRIBUTE, arg_attrs);
    }
    if let Some(res_attrs) = res_attrs {
        builder = builder.add_attribute(RES_ATTRS_ATTRIBUTE, res_attrs);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::call`")
}

/// Operation trait for `emitc.declare_func`.
pub trait DeclareFuncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the declared function symbol.
    fn symbol_name(&self) -> FlatSymbolRefAttributeRef<'c, 't> {
        self.attribute(SYMBOL_NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast())
            .unwrap_or_else(|| panic!("invalid '{SYMBOL_NAME_ATTRIBUTE}' attribute in `emitc::declare_func`"))
    }
}

mlir_op!(DeclareFunc);
mlir_op_trait!(DeclareFunc, ZeroRegions);
mlir_op_trait!(DeclareFunc, ZeroSuccessors);

/// Constructs a new detached [`DeclareFuncOperation`].
pub fn declare_func<'c, 't: 'c, S: IntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    symbol_name: S,
    location: L,
) -> DetachedDeclareFuncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.declare_func", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, symbol_name.into_with_context(context))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::declare_func`")
}

/// Name of MLIR symbol-name attributes.
pub const SYMBOL_NAME_ATTRIBUTE: &str = "sym_name";

/// Name of MLIR function-type attributes.
pub const FUNCTION_TYPE_ATTRIBUTE: &str = "function_type";

/// Name of Emit-C specifier array attributes.
pub const SPECIFIERS_ATTRIBUTE: &str = "specifiers";

/// Name of function argument attribute arrays.
pub const ARG_ATTRS_ATTRIBUTE: &str = "arg_attrs";

/// Name of function result attribute arrays.
pub const RES_ATTRS_ATTRIBUTE: &str = "res_attrs";

/// Operation trait for `emitc.func`.
pub trait FuncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the function symbol name.
    fn symbol_name(&self) -> StringRef<'c> {
        self.attribute(SYMBOL_NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{SYMBOL_NAME_ATTRIBUTE}' attribute in `emitc::func`"))
    }

    /// Returns the function type.
    fn function_type(&self) -> FunctionTypeRef<'c, 't> {
        self.attribute(FUNCTION_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>())
            .and_then(|attribute| attribute.r#type().cast())
            .unwrap_or_else(|| panic!("invalid '{FUNCTION_TYPE_ATTRIBUTE}' attribute in `emitc::func`"))
    }

    /// Returns optional C/C++ specifiers such as `static` or `inline`.
    fn specifiers(&self) -> Option<ArrayAttributeRef<'c, 't>> {
        self.attribute(SPECIFIERS_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the function body region.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(Func);
mlir_op_trait!(Func, AutomaticAllocationScope);
mlir_op_trait!(Func, IsolatedFromAbove);
mlir_op_trait!(Func, OneRegion);
mlir_op_trait!(Func, ZeroSuccessors);

/// Constructs a new detached [`FuncOperation`].
pub fn func<'c, 't: 'c, N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    name: N,
    function_type: FunctionTypeRef<'c, 't>,
    body: DetachedRegion<'c, 't>,
    specifiers: Option<ArrayAttributeRef<'c, 't>>,
    arg_attrs: Option<ArrayAttributeRef<'c, 't>>,
    res_attrs: Option<ArrayAttributeRef<'c, 't>>,
    location: L,
) -> DetachedFuncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let mut builder = OperationBuilder::new("emitc.func", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, name.into_with_context(context))
        .add_attribute(FUNCTION_TYPE_ATTRIBUTE, context.type_attribute(function_type))
        .add_region(body);
    if let Some(specifiers) = specifiers {
        builder = builder.add_attribute(SPECIFIERS_ATTRIBUTE, specifiers);
    }
    if let Some(arg_attrs) = arg_attrs {
        builder = builder.add_attribute(ARG_ATTRS_ATTRIBUTE, arg_attrs);
    }
    if let Some(res_attrs) = res_attrs {
        builder = builder.add_attribute(RES_ATTRS_ATTRIBUTE, res_attrs);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::func`")
}

/// Operation trait for `emitc.return`.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional returned value.
    fn value(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(0)
    }
}

mlir_op!(Return);
mlir_op_trait!(Return, ReturnLike);
mlir_op_trait!(Return, ZeroRegions);
mlir_op_trait!(Return, ZeroSuccessors);

/// Constructs a new detached [`ReturnOperation`].
pub fn r#return<'value, 'c: 'value, 't: 'c, L: Location<'c, 't>>(
    value: Option<ValueRef<'value, 'c, 't>>,
    location: L,
) -> DetachedReturnOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let mut builder = OperationBuilder::new("emitc.return", location);
    if let Some(value) = value {
        builder = builder.add_operand(value);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::return`")
}

/// Name of the `emitc.include` include attribute.
pub const INCLUDE_ATTRIBUTE: &str = "include";

/// Name of the `emitc.include` standard-include marker.
pub const IS_STANDARD_INCLUDE_ATTRIBUTE: &str = "is_standard_include";

/// Operation trait for `emitc.include`.
pub trait IncludeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the included path or header name.
    fn include(&self) -> StringRef<'c> {
        self.attribute(INCLUDE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{INCLUDE_ATTRIBUTE}' attribute in `emitc::include`"))
    }

    /// Returns whether this is a standard include rendered with angle brackets.
    fn is_standard_include(&self) -> bool {
        self.has_attribute(IS_STANDARD_INCLUDE_ATTRIBUTE)
    }
}

mlir_op!(Include);
mlir_op_trait!(Include, ZeroRegions);
mlir_op_trait!(Include, ZeroSuccessors);

/// Constructs a new detached [`IncludeOperation`].
pub fn include<'c, 't: 'c, I: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    include: I,
    is_standard_include: bool,
    location: L,
) -> DetachedIncludeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let mut builder = OperationBuilder::new("emitc.include", location)
        .add_attribute(INCLUDE_ATTRIBUTE, include.into_with_context(context));
    if is_standard_include {
        builder = builder.add_attribute(IS_STANDARD_INCLUDE_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::include`")
}

/// Operation trait for `emitc.literal`.
pub trait LiteralOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the literal source spelling.
    fn value(&self) -> StringRef<'c> {
        self.attribute(VALUE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{VALUE_ATTRIBUTE}' attribute in `emitc::literal`"))
    }

    /// Returns this operation's result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(Literal);
mlir_op_trait!(Literal, OneResult);
mlir_op_trait!(Literal, ZeroOperands);
mlir_op_trait!(Literal, ZeroRegions);
mlir_op_trait!(Literal, ZeroSuccessors);

/// Constructs a new detached [`LiteralOperation`].
pub fn literal<
    'c,
    't: 'c,
    V: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    value: V,
    result_type: ResultType,
    location: L,
) -> DetachedLiteralOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.literal", location)
        .add_attribute(VALUE_ATTRIBUTE, value.into_with_context(context))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::literal`")
}

/// Operation trait for `emitc.logical_and`.
pub trait LogicalAndOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(LogicalAnd);
mlir_op_trait!(LogicalAnd, OneResult);
mlir_op_trait!(LogicalAnd, ZeroRegions);
mlir_op_trait!(LogicalAnd, ZeroSuccessors);
mlir_op_trait!(LogicalAnd, @local BinaryExpressionOperation);

/// Constructs a new detached [`LogicalAndOperation`].
pub fn logical_and<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    location: L,
) -> DetachedLogicalAndOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let result_type = context.signless_integer_type(1);
    OperationBuilder::new("emitc.logical_and", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::logical_and`")
}

/// Operation trait for `emitc.logical_not`.
pub trait LogicalNotOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(LogicalNot);
mlir_op_trait!(LogicalNot, OneOperand);
mlir_op_trait!(LogicalNot, OneResult);
mlir_op_trait!(LogicalNot, ZeroRegions);
mlir_op_trait!(LogicalNot, ZeroSuccessors);
mlir_op_trait!(LogicalNot, @local UnaryExpressionOperation);

/// Constructs a new detached [`LogicalNotOperation`].
pub fn logical_not<'operand, 'c: 'operand, 't: 'c, Operand: Value<'operand, 'c, 't>, L: Location<'c, 't>>(
    operand: Operand,
    location: L,
) -> DetachedLogicalNotOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let result_type = context.signless_integer_type(1);
    OperationBuilder::new("emitc.logical_not", location)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::logical_not`")
}

/// Operation trait for `emitc.logical_or`.
pub trait LogicalOrOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(LogicalOr);
mlir_op_trait!(LogicalOr, OneResult);
mlir_op_trait!(LogicalOr, ZeroRegions);
mlir_op_trait!(LogicalOr, ZeroSuccessors);
mlir_op_trait!(LogicalOr, @local BinaryExpressionOperation);

/// Constructs a new detached [`LogicalOrOperation`].
pub fn logical_or<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    location: L,
) -> DetachedLogicalOrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let result_type = context.signless_integer_type(1);
    OperationBuilder::new("emitc.logical_or", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::logical_or`")
}

/// Operation trait for `emitc.load`.
pub trait LoadOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(Load);
mlir_op_trait!(Load, OneOperand);
mlir_op_trait!(Load, OneResult);
mlir_op_trait!(Load, ZeroRegions);
mlir_op_trait!(Load, ZeroSuccessors);
mlir_op_trait!(Load, @local UnaryExpressionOperation);

/// Constructs a new detached [`LoadOperation`].
pub fn load<
    'operand,
    'c: 'operand,
    't: 'c,
    Operand: Value<'operand, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    operand: Operand,
    result_type: ResultType,
    location: L,
) -> DetachedLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.load", location)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::load`")
}

/// Operation trait for `emitc.mul`.
pub trait MulOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(Mul);
mlir_op_trait!(Mul, OneResult);
mlir_op_trait!(Mul, ZeroRegions);
mlir_op_trait!(Mul, ZeroSuccessors);
mlir_op_trait!(Mul, @local BinaryExpressionOperation);

/// Constructs a new detached [`MulOperation`].
pub fn mul<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    result_type: ResultType,
    location: L,
) -> DetachedMulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.mul", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::mul`")
}

/// Operation trait for `emitc.rem`.
pub trait RemOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(Rem);
mlir_op_trait!(Rem, OneResult);
mlir_op_trait!(Rem, ZeroRegions);
mlir_op_trait!(Rem, ZeroSuccessors);
mlir_op_trait!(Rem, @local BinaryExpressionOperation);

/// Constructs a new detached [`RemOperation`].
pub fn rem<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    result_type: ResultType,
    location: L,
) -> DetachedRemOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.rem", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::rem`")
}

/// Operation trait for `emitc.sub`.
pub trait SubOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(Sub);
mlir_op_trait!(Sub, OneResult);
mlir_op_trait!(Sub, ZeroRegions);
mlir_op_trait!(Sub, ZeroSuccessors);
mlir_op_trait!(Sub, @local BinaryExpressionOperation);

/// Constructs a new detached [`SubOperation`].
pub fn sub<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    result_type: ResultType,
    location: L,
) -> DetachedSubOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.sub", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::sub`")
}

/// Name of member-selection attributes.
pub const MEMBER_ATTRIBUTE: &str = "member";

/// Operation trait for `emitc.member`.
pub trait MemberOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {
    /// Returns the selected member name.
    fn member(&self) -> StringRef<'c> {
        self.attribute(MEMBER_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{MEMBER_ATTRIBUTE}' attribute in `emitc::member`"))
    }
}

mlir_op!(Member);
mlir_op_trait!(Member, OneOperand);
mlir_op_trait!(Member, OneResult);
mlir_op_trait!(Member, ZeroRegions);
mlir_op_trait!(Member, ZeroSuccessors);
mlir_op_trait!(Member, @local UnaryExpressionOperation);

/// Constructs a new detached [`MemberOperation`].
pub fn member<
    'operand,
    'c: 'operand,
    't: 'c,
    Operand: Value<'operand, 'c, 't>,
    ResultType: Type<'c, 't>,
    M: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    member: M,
    operand: Operand,
    result_type: ResultType,
    location: L,
) -> DetachedMemberOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.member", location)
        .add_attribute(MEMBER_ATTRIBUTE, member.into_with_context(context))
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::member`")
}

/// Operation trait for `emitc.member_of_ptr`.
pub trait MemberOfPtrOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {
    /// Returns the selected member name.
    fn member(&self) -> StringRef<'c> {
        self.attribute(MEMBER_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{MEMBER_ATTRIBUTE}' attribute in `emitc::member_of_ptr`"))
    }
}

mlir_op!(MemberOfPtr);
mlir_op_trait!(MemberOfPtr, OneOperand);
mlir_op_trait!(MemberOfPtr, OneResult);
mlir_op_trait!(MemberOfPtr, ZeroRegions);
mlir_op_trait!(MemberOfPtr, ZeroSuccessors);
mlir_op_trait!(MemberOfPtr, @local UnaryExpressionOperation);

/// Constructs a new detached [`MemberOfPtrOperation`].
pub fn member_of_ptr<
    'operand,
    'c: 'operand,
    't: 'c,
    Operand: Value<'operand, 'c, 't>,
    ResultType: Type<'c, 't>,
    M: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    member: M,
    operand: Operand,
    result_type: ResultType,
    location: L,
) -> DetachedMemberOfPtrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.member_of_ptr", location)
        .add_attribute(MEMBER_ATTRIBUTE, member.into_with_context(context))
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::member_of_ptr`")
}

/// Operation trait for `emitc.conditional`.
pub trait ConditionalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the condition operand.
    fn condition(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the value used when the condition is true.
    fn true_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the value used when the condition is false.
    fn false_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(Conditional);
mlir_op_trait!(Conditional, OneResult);
mlir_op_trait!(Conditional, ZeroRegions);
mlir_op_trait!(Conditional, ZeroSuccessors);

/// Constructs a new detached [`ConditionalOperation`].
pub fn conditional<
    'condition,
    'true_value,
    'false_value,
    'c: 'condition + 'true_value + 'false_value,
    't: 'c,
    Condition: Value<'condition, 'c, 't>,
    TrueValue: Value<'true_value, 'c, 't>,
    FalseValue: Value<'false_value, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    condition: Condition,
    true_value: TrueValue,
    false_value: FalseValue,
    result_type: ResultType,
    location: L,
) -> DetachedConditionalOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.conditional", location)
        .add_operands(&[condition.as_ref(), true_value.as_ref(), false_value.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::conditional`")
}

/// Operation trait for `emitc.unary_minus`.
pub trait UnaryMinusOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(UnaryMinus);
mlir_op_trait!(UnaryMinus, OneOperand);
mlir_op_trait!(UnaryMinus, OneResult);
mlir_op_trait!(UnaryMinus, ZeroRegions);
mlir_op_trait!(UnaryMinus, ZeroSuccessors);
mlir_op_trait!(UnaryMinus, @local UnaryExpressionOperation);

/// Constructs a new detached [`UnaryMinusOperation`].
pub fn unary_minus<
    'operand,
    'c: 'operand,
    't: 'c,
    Operand: Value<'operand, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    operand: Operand,
    result_type: ResultType,
    location: L,
) -> DetachedUnaryMinusOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.unary_minus", location)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::unary_minus`")
}

/// Operation trait for `emitc.unary_plus`.
pub trait UnaryPlusOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {}

mlir_op!(UnaryPlus);
mlir_op_trait!(UnaryPlus, OneOperand);
mlir_op_trait!(UnaryPlus, OneResult);
mlir_op_trait!(UnaryPlus, ZeroRegions);
mlir_op_trait!(UnaryPlus, ZeroSuccessors);
mlir_op_trait!(UnaryPlus, @local UnaryExpressionOperation);

/// Constructs a new detached [`UnaryPlusOperation`].
pub fn unary_plus<
    'operand,
    'c: 'operand,
    't: 'c,
    Operand: Value<'operand, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    operand: Operand,
    result_type: ResultType,
    location: L,
) -> DetachedUnaryPlusOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.unary_plus", location)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::unary_plus`")
}

/// Operation trait for `emitc.variable`.
pub trait VariableOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the initial value attribute.
    fn value(&self) -> AttributeRef<'c, 't> {
        self.attribute(VALUE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("invalid '{VALUE_ATTRIBUTE}' attribute in `emitc::variable`"))
    }

    /// Returns this operation's allocated result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(Variable);
mlir_op_trait!(Variable, OneResult);
mlir_op_trait!(Variable, ZeroOperands);
mlir_op_trait!(Variable, ZeroRegions);
mlir_op_trait!(Variable, ZeroSuccessors);

/// Constructs a new detached [`VariableOperation`].
pub fn variable<'c, 't: 'c, A: Attribute<'c, 't>, ResultType: Type<'c, 't>, L: Location<'c, 't>>(
    value: A,
    result_type: ResultType,
    location: L,
) -> DetachedVariableOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.variable", location)
        .add_attribute(VALUE_ATTRIBUTE, value)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::variable`")
}

/// Name of the `emitc.global` type attribute.
pub const TYPE_ATTRIBUTE: &str = "type";

/// Name of the `emitc.global` initial value attribute.
pub const INITIAL_VALUE_ATTRIBUTE: &str = "initial_value";

/// Name of the `emitc.global` external linkage marker.
pub const EXTERN_SPECIFIER_ATTRIBUTE: &str = "extern_specifier";

/// Name of the `emitc.global` internal linkage marker.
pub const STATIC_SPECIFIER_ATTRIBUTE: &str = "static_specifier";

/// Name of the `emitc.global` constant marker.
pub const CONST_SPECIFIER_ATTRIBUTE: &str = "const_specifier";

/// Operation trait for `emitc.global`.
pub trait GlobalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the global symbol name.
    fn symbol_name(&self) -> StringRef<'c> {
        self.attribute(SYMBOL_NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{SYMBOL_NAME_ATTRIBUTE}' attribute in `emitc::global`"))
    }

    /// Returns the global variable type.
    fn r#type(&self) -> TypeRef<'c, 't> {
        self.attribute(TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>())
            .map(|attribute| attribute.r#type())
            .unwrap_or_else(|| panic!("invalid '{TYPE_ATTRIBUTE}' attribute in `emitc::global`"))
    }

    /// Returns the optional initial value.
    fn initial_value(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute(INITIAL_VALUE_ATTRIBUTE)
    }

    /// Returns whether the global has external linkage.
    fn extern_specifier(&self) -> bool {
        self.has_attribute(EXTERN_SPECIFIER_ATTRIBUTE)
    }

    /// Returns whether the global has internal linkage.
    fn static_specifier(&self) -> bool {
        self.has_attribute(STATIC_SPECIFIER_ATTRIBUTE)
    }

    /// Returns whether the global is constant.
    fn const_specifier(&self) -> bool {
        self.has_attribute(CONST_SPECIFIER_ATTRIBUTE)
    }
}

mlir_op!(Global);
mlir_op_trait!(Global, Symbol);
mlir_op_trait!(Global, ZeroOperands);
mlir_op_trait!(Global, ZeroRegions);
mlir_op_trait!(Global, ZeroSuccessors);

/// Constructs a new detached [`GlobalOperation`].
pub fn global<
    'c,
    't: 'c,
    N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    GlobalType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    symbol_name: N,
    r#type: GlobalType,
    initial_value: Option<AttributeRef<'c, 't>>,
    extern_specifier: bool,
    static_specifier: bool,
    const_specifier: bool,
    location: L,
) -> DetachedGlobalOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let mut builder = OperationBuilder::new("emitc.global", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, symbol_name.into_with_context(context))
        .add_attribute(TYPE_ATTRIBUTE, context.type_attribute(r#type));
    if let Some(initial_value) = initial_value {
        builder = builder.add_attribute(INITIAL_VALUE_ATTRIBUTE, initial_value);
    }
    if extern_specifier {
        builder = builder.add_attribute(EXTERN_SPECIFIER_ATTRIBUTE, context.unit_attribute());
    }
    if static_specifier {
        builder = builder.add_attribute(STATIC_SPECIFIER_ATTRIBUTE, context.unit_attribute());
    }
    if const_specifier {
        builder = builder.add_attribute(CONST_SPECIFIER_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::global`")
}

/// Name of the symbol-reference attribute used by `emitc.get_global`.
pub const NAME_ATTRIBUTE: &str = "name";

/// Operation trait for `emitc.get_global`.
pub trait GetGlobalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the referenced global name.
    fn name(&self) -> FlatSymbolRefAttributeRef<'c, 't> {
        self.attribute(NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast())
            .unwrap_or_else(|| panic!("invalid '{NAME_ATTRIBUTE}' attribute in `emitc::get_global`"))
    }

    /// Returns this operation's result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(GetGlobal);
mlir_op_trait!(GetGlobal, OneResult);
mlir_op_trait!(GetGlobal, ZeroOperands);
mlir_op_trait!(GetGlobal, ZeroRegions);
mlir_op_trait!(GetGlobal, ZeroSuccessors);

/// Constructs a new detached [`GetGlobalOperation`].
pub fn get_global<
    'c,
    't: 'c,
    N: IntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    name: N,
    result_type: ResultType,
    location: L,
) -> DetachedGetGlobalOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.get_global", location)
        .add_attribute(NAME_ATTRIBUTE, name.into_with_context(context))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::get_global`")
}

/// Operation trait for `emitc.verbatim`.
pub trait VerbatimOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the verbatim format string.
    fn value(&self) -> StringRef<'c> {
        self.attribute(VALUE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{VALUE_ATTRIBUTE}' attribute in `emitc::verbatim`"))
    }

    /// Returns the format arguments.
    fn format_arguments(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(Verbatim);
mlir_op_trait!(Verbatim, ZeroRegions);
mlir_op_trait!(Verbatim, ZeroSuccessors);

/// Constructs a new detached [`VerbatimOperation`].
pub fn verbatim<
    'argument,
    'c: 'argument,
    't: 'c,
    V: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    value: V,
    format_arguments: &[ValueRef<'argument, 'c, 't>],
    location: L,
) -> DetachedVerbatimOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.verbatim", location)
        .add_attribute(VALUE_ATTRIBUTE, value.into_with_context(context))
        .add_operands(format_arguments)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::verbatim`")
}

/// Operation trait for `emitc.assign`.
pub trait AssignOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the lvalue being assigned.
    fn variable(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the assigned value.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }
}

mlir_op!(Assign);
mlir_op_trait!(Assign, ZeroRegions);
mlir_op_trait!(Assign, ZeroSuccessors);

/// Constructs a new detached [`AssignOperation`].
pub fn assign<
    'variable,
    'value,
    'c: 'variable + 'value,
    't: 'c,
    Variable: Value<'variable, 'c, 't>,
    AssignedValue: Value<'value, 'c, 't>,
    L: Location<'c, 't>,
>(
    variable: Variable,
    value: AssignedValue,
    location: L,
) -> DetachedAssignOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.assign", location)
        .add_operands(&[variable.as_ref(), value.as_ref()])
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::assign`")
}

/// Operation trait for `emitc.yield`.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional yielded value.
    fn value(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(0)
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Constructs a new detached [`YieldOperation`].
pub fn r#yield<'value, 'c: 'value, 't: 'c, L: Location<'c, 't>>(
    value: Option<ValueRef<'value, 'c, 't>>,
    location: L,
) -> DetachedYieldOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let mut builder = OperationBuilder::new("emitc.yield", location);
    if let Some(value) = value {
        builder = builder.add_operand(value);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::yield`")
}

/// Operation trait for `emitc.if`.
pub trait IfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the condition.
    fn condition(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the then region.
    fn then_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns the else region.
    fn else_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(1).unwrap()
    }
}

mlir_op!(If);
mlir_op_trait!(If, ZeroSuccessors);

/// Constructs a new detached [`IfOperation`].
pub fn r#if<'condition, 'c: 'condition, 't: 'c, Condition: Value<'condition, 'c, 't>, L: Location<'c, 't>>(
    condition: Condition,
    then_region: DetachedRegion<'c, 't>,
    else_region: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedIfOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.if", location)
        .add_operand(condition)
        .add_regions(vec![then_region, else_region])
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::if`")
}

/// Operation trait for `emitc.subscript`.
pub trait SubscriptOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the indexed value.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
    }

    /// Returns this operation's result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(Subscript);
mlir_op_trait!(Subscript, OneResult);
mlir_op_trait!(Subscript, ZeroRegions);
mlir_op_trait!(Subscript, ZeroSuccessors);

/// Constructs a new detached [`SubscriptOperation`].
pub fn subscript<
    'value,
    'index,
    'c: 'value + 'index,
    't: 'c,
    IndexedValue: Value<'value, 'c, 't>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    value: IndexedValue,
    indices: &[ValueRef<'index, 'c, 't>],
    result_type: ResultType,
    location: L,
) -> DetachedSubscriptOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.subscript", location)
        .add_operand(value)
        .add_operands(indices)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::subscript`")
}

/// Name of the `emitc.switch` cases attribute.
pub const CASES_ATTRIBUTE: &str = "cases";

/// Operation trait for `emitc.switch`.
pub trait SwitchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the switched value.
    fn argument(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the case values.
    fn cases(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(CASES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast())
            .unwrap_or_else(|| panic!("invalid '{CASES_ATTRIBUTE}' attribute in `emitc::switch`"))
    }

    /// Returns the default region.
    fn default_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns the case regions.
    fn case_regions(&self) -> Vec<RegionRef<'o, 'c, 't>> {
        (1..self.region_count()).map(|index| self.region(index).unwrap()).collect()
    }
}

mlir_op!(Switch);
mlir_op_trait!(Switch, ZeroSuccessors);

/// Constructs a new detached [`SwitchOperation`].
pub fn switch<'argument, 'c: 'argument, 't: 'c, Argument: Value<'argument, 'c, 't>, L: Location<'c, 't>>(
    argument: Argument,
    cases: &[i64],
    default_region: DetachedRegion<'c, 't>,
    case_regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> DetachedSwitchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let mut regions = vec![default_region];
    regions.extend(case_regions);
    OperationBuilder::new("emitc.switch", location)
        .add_operand(argument)
        .add_attribute(CASES_ATTRIBUTE, context.dense_i64_array_attribute(cases).unwrap())
        .add_regions(regions)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::switch`")
}

/// Name of the `emitc.class` final specifier.
pub const FINAL_SPECIFIER_ATTRIBUTE: &str = "final_specifier";

/// Operation trait for `emitc.class`.
pub trait ClassOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the class symbol name.
    fn symbol_name(&self) -> StringRef<'c> {
        self.attribute(SYMBOL_NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{SYMBOL_NAME_ATTRIBUTE}' attribute in `emitc::class`"))
    }

    /// Returns whether this class has a `final` specifier.
    fn final_specifier(&self) -> bool {
        self.has_attribute(FINAL_SPECIFIER_ATTRIBUTE)
    }

    /// Returns the class body.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(Class);
mlir_op_trait!(Class, AutomaticAllocationScope);
mlir_op_trait!(Class, IsolatedFromAbove);
mlir_op_trait!(Class, OneRegion);
mlir_op_trait!(Class, Symbol);
mlir_op_trait!(Class, SymbolTable);
mlir_op_trait!(Class, ZeroSuccessors);

/// Constructs a new detached [`ClassOperation`].
pub fn class<'c, 't: 'c, N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    symbol_name: N,
    final_specifier: bool,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedClassOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let mut builder = OperationBuilder::new("emitc.class", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, symbol_name.into_with_context(context))
        .add_region(body);
    if final_specifier {
        builder = builder.add_attribute(FINAL_SPECIFIER_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::class`")
}

/// Operation trait for `emitc.field`.
pub trait FieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the field symbol name.
    fn symbol_name(&self) -> StringRef<'c> {
        self.attribute(SYMBOL_NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{SYMBOL_NAME_ATTRIBUTE}' attribute in `emitc::field`"))
    }

    /// Returns the field type.
    fn r#type(&self) -> TypeRef<'c, 't> {
        self.attribute(TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>())
            .map(|attribute| attribute.r#type())
            .unwrap_or_else(|| panic!("invalid '{TYPE_ATTRIBUTE}' attribute in `emitc::field`"))
    }

    /// Returns the optional initial value.
    fn initial_value(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute(INITIAL_VALUE_ATTRIBUTE)
    }
}

mlir_op!(Field);
mlir_op_trait!(Field, Symbol);
mlir_op_trait!(Field, ZeroOperands);
mlir_op_trait!(Field, ZeroRegions);
mlir_op_trait!(Field, ZeroSuccessors);

/// Constructs a new detached [`FieldOperation`].
pub fn field<
    'c,
    't: 'c,
    N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    FieldType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    symbol_name: N,
    r#type: FieldType,
    initial_value: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedFieldOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    let mut builder = OperationBuilder::new("emitc.field", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, symbol_name.into_with_context(context))
        .add_attribute(TYPE_ATTRIBUTE, context.type_attribute(r#type));
    if let Some(initial_value) = initial_value {
        builder = builder.add_attribute(INITIAL_VALUE_ATTRIBUTE, initial_value);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::field`")
}

/// Name of the `emitc.get_field` field-name attribute.
pub const FIELD_NAME_ATTRIBUTE: &str = "field_name";

/// Operation trait for `emitc.get_field`.
pub trait GetFieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the referenced field name.
    fn field_name(&self) -> FlatSymbolRefAttributeRef<'c, 't> {
        self.attribute(FIELD_NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast())
            .unwrap_or_else(|| panic!("invalid '{FIELD_NAME_ATTRIBUTE}' attribute in `emitc::get_field`"))
    }

    /// Returns this operation's result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(GetField);
mlir_op_trait!(GetField, OneResult);
mlir_op_trait!(GetField, ZeroOperands);
mlir_op_trait!(GetField, ZeroRegions);
mlir_op_trait!(GetField, ZeroSuccessors);

/// Constructs a new detached [`GetFieldOperation`].
pub fn get_field<
    'c,
    't: 'c,
    N: IntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    field_name: N,
    result_type: ResultType,
    location: L,
) -> DetachedGetFieldOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.get_field", location)
        .add_attribute(FIELD_NAME_ATTRIBUTE, field_name.into_with_context(context))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::get_field`")
}

/// Operation trait for `emitc.do`.
pub trait DoOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the body region.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns the condition region.
    fn condition(&self) -> RegionRef<'o, 'c, 't> {
        self.region(1).unwrap()
    }
}

mlir_op!(Do);
mlir_op_trait!(Do, ZeroSuccessors);

/// Constructs a new detached [`DoOperation`].
pub fn r#do<'c, 't: 'c, L: Location<'c, 't>>(
    body: DetachedRegion<'c, 't>,
    condition: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedDoOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c());
    OperationBuilder::new("emitc.do", location)
        .add_regions(vec![body, condition])
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `emit_c::do`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::{Block, Context, Operation, Region, Type, TypeRef, Value, ValueRef};

    use super::*;

    #[test]
    fn test_func_return_and_constant_module() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let function_type = context.function_type::<TypeRef, _>(&[], &[i32_type]);

        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let lhs = block.append_operation(constant(context.integer_attribute(i32_type, 1), i32_type, location));
            let rhs = block.append_operation(constant(context.integer_attribute(i32_type, 2), i32_type, location));
            let op = add(lhs.result(0).unwrap(), rhs.result(0).unwrap(), i32_type, location);
            assert_eq!(op.lhs(), lhs.result(0).unwrap().as_ref());
            assert_eq!(op.rhs(), rhs.result(0).unwrap().as_ref());

            let op = block.append_operation(op);
            let return_op = r#return(Some(op.result(0).unwrap().as_ref()), location);
            assert_eq!(return_op.value(), Some(op.result(0).unwrap().as_ref()));
            block.append_operation(return_op);

            let op = func("main", function_type, block.into(), None, None, None, location);
            assert_eq!(op.symbol_name().as_str().unwrap(), "main");
            assert_eq!(op.function_type(), function_type);
            assert!(op.specifiers().is_none());
            op
        });

        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @main() -> i32 {
                    %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
                    %1 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
                    %2 = add %0, %1 : (i32, i32) -> i32
                    return %2 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_expression_operation_constructors_and_accessors() {
        let context = Context::new();
        let location = context.unknown_location();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        let lvalue_i32_type = context.emit_c_lvalue_type(i32_type);
        let pointer_i32_type = context.emit_c_pointer_type(i32_type);
        let array_i32_type = context.emit_c_array_type(i32_type, &[4]);
        let opaque_box_type = context.emit_c_opaque_type("Box");
        let lvalue_box_type = context.emit_c_lvalue_type(opaque_box_type);
        let pointer_box_type = context.emit_c_pointer_type(opaque_box_type);
        let lvalue_pointer_box_type = context.emit_c_lvalue_type(pointer_box_type);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref(), i1_type.as_ref()], &[]);
        let module = context.module(location);
        let mut block = context.block(&[
            (i32_type.as_ref(), location),
            (i32_type.as_ref(), location),
            (i1_type.as_ref(), location),
        ]);
        let lhs = block.argument(0).unwrap();
        let rhs = block.argument(1).unwrap();
        let condition = block.argument(2).unwrap();

        let op = variable(context.integer_attribute(i32_type, 0), lvalue_i32_type, location);
        assert_eq!(op.value().to_string(), "0 : i32");
        assert_eq!(VariableOperation::output(&op).r#type(), lvalue_i32_type.as_ref());
        let op = block.append_operation(op);
        let lvalue = op.result(0).unwrap().as_ref();

        let op = literal("values", array_i32_type, location);
        assert_eq!(op.value().as_str().unwrap(), "values");
        assert_eq!(LiteralOperation::output(&op).r#type(), array_i32_type.as_ref());
        let op = block.append_operation(op);
        let array = op.result(0).unwrap().as_ref();

        let op = variable(context.emit_c_opaque_attribute("box"), lvalue_box_type, location);
        assert_eq!(VariableOperation::output(&op).r#type(), lvalue_box_type.as_ref());
        let op = block.append_operation(op);
        let box_lvalue = op.result(0).unwrap().as_ref();

        let op = address_of(box_lvalue, pointer_box_type, location);
        assert_eq!(UnaryExpressionOperation::operand(&op), box_lvalue);
        assert_eq!(UnaryExpressionOperation::output(&op).r#type(), pointer_box_type.as_ref());
        block.append_operation(op);

        let op = variable(context.emit_c_opaque_attribute("box_ptr"), lvalue_pointer_box_type, location);
        assert_eq!(VariableOperation::output(&op).r#type(), lvalue_pointer_box_type.as_ref());
        let op = block.append_operation(op);
        let box_pointer_lvalue = op.result(0).unwrap().as_ref();

        let op = address_of(lvalue, pointer_i32_type, location);
        assert_eq!(UnaryExpressionOperation::operand(&op), lvalue);
        assert_eq!(UnaryExpressionOperation::output(&op).r#type(), pointer_i32_type.as_ref());
        let op = block.append_operation(op);
        let pointer = op.result(0).unwrap().as_ref();

        let op = apply("&", lvalue, pointer_i32_type, location);
        assert_eq!(op.applicable_operator().as_str().unwrap(), "&");
        assert_eq!(UnaryExpressionOperation::operand(&op), lvalue);
        assert_eq!(UnaryExpressionOperation::output(&op).r#type(), pointer_i32_type.as_ref());
        block.append_operation(op);

        let op = bitwise_and(lhs, rhs, i32_type, location);
        assert_eq!(op.lhs(), lhs);
        assert_eq!(op.rhs(), rhs);
        assert_eq!(BinaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = bitwise_left_shift(lhs, rhs, i32_type, location);
        assert_eq!(op.lhs(), lhs);
        assert_eq!(op.rhs(), rhs);
        assert_eq!(BinaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = bitwise_not(lhs, i32_type, location);
        assert_eq!(UnaryExpressionOperation::operand(&op), lhs);
        assert_eq!(UnaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = bitwise_or(lhs, rhs, i32_type, location);
        assert_eq!(op.lhs(), lhs);
        assert_eq!(op.rhs(), rhs);
        assert_eq!(BinaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = bitwise_right_shift(lhs, rhs, i32_type, location);
        assert_eq!(op.lhs(), lhs);
        assert_eq!(op.rhs(), rhs);
        assert_eq!(BinaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = bitwise_xor(lhs, rhs, i32_type, location);
        assert_eq!(op.lhs(), lhs);
        assert_eq!(op.rhs(), rhs);
        assert_eq!(BinaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = cast(lhs, context.emit_c_opaque_type("int"), location);
        assert_eq!(UnaryExpressionOperation::operand(&op), lhs);
        assert_eq!(UnaryExpressionOperation::output(&op).r#type(), context.emit_c_opaque_type("int").as_ref());
        block.append_operation(op);

        let op = cmp(CmpPredicate::LessThan, lhs, rhs, i1_type, location);
        assert_eq!(op.predicate(), CmpPredicate::LessThan);
        assert_eq!(op.lhs(), lhs);
        assert_eq!(op.rhs(), rhs);
        assert_eq!(BinaryExpressionOperation::output(&op).r#type(), i1_type.as_ref());
        block.append_operation(op);

        let op = constant(context.integer_attribute(i32_type, 3), i32_type, location);
        assert_eq!(op.value().to_string(), "3 : i32");
        assert_eq!(ConstantOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = dereference(pointer, lvalue_i32_type, location);
        assert_eq!(UnaryExpressionOperation::operand(&op), pointer);
        assert_eq!(UnaryExpressionOperation::output(&op).r#type(), lvalue_i32_type.as_ref());
        block.append_operation(op);

        let op = div(lhs, rhs, i32_type, location);
        assert_eq!(op.lhs(), lhs);
        assert_eq!(op.rhs(), rhs);
        assert_eq!(BinaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = logical_and(condition, condition, location);
        assert_eq!(op.lhs(), condition);
        assert_eq!(op.rhs(), condition);
        assert_eq!(BinaryExpressionOperation::output(&op).r#type(), i1_type.as_ref());
        block.append_operation(op);

        let op = logical_not(condition, location);
        assert_eq!(UnaryExpressionOperation::operand(&op), condition);
        assert_eq!(UnaryExpressionOperation::output(&op).r#type(), i1_type.as_ref());
        block.append_operation(op);

        let op = logical_or(condition, condition, location);
        assert_eq!(op.lhs(), condition);
        assert_eq!(op.rhs(), condition);
        assert_eq!(BinaryExpressionOperation::output(&op).r#type(), i1_type.as_ref());
        block.append_operation(op);

        let op = load(lvalue, i32_type, location);
        assert_eq!(UnaryExpressionOperation::operand(&op), lvalue);
        assert_eq!(UnaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = mul(lhs, rhs, i32_type, location);
        assert_eq!(op.lhs(), lhs);
        assert_eq!(op.rhs(), rhs);
        assert_eq!(BinaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = rem(lhs, rhs, i32_type, location);
        assert_eq!(op.lhs(), lhs);
        assert_eq!(op.rhs(), rhs);
        assert_eq!(BinaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = sub(lhs, rhs, i32_type, location);
        assert_eq!(op.lhs(), lhs);
        assert_eq!(op.rhs(), rhs);
        assert_eq!(BinaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = member("field", box_lvalue, lvalue_i32_type, location);
        assert_eq!(op.member().as_str().unwrap(), "field");
        assert_eq!(UnaryExpressionOperation::operand(&op), box_lvalue);
        assert_eq!(UnaryExpressionOperation::output(&op).r#type(), lvalue_i32_type.as_ref());
        block.append_operation(op);

        let op = member_of_ptr("field", box_pointer_lvalue, lvalue_i32_type, location);
        assert_eq!(op.member().as_str().unwrap(), "field");
        assert_eq!(UnaryExpressionOperation::operand(&op), box_pointer_lvalue);
        assert_eq!(UnaryExpressionOperation::output(&op).r#type(), lvalue_i32_type.as_ref());
        block.append_operation(op);

        let op = conditional(condition, lhs, rhs, i32_type, location);
        assert_eq!(op.condition(), condition);
        assert_eq!(op.true_value(), lhs);
        assert_eq!(op.false_value(), rhs);
        assert_eq!(ConditionalOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = unary_minus(lhs, i32_type, location);
        assert_eq!(UnaryExpressionOperation::operand(&op), lhs);
        assert_eq!(UnaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = unary_plus(lhs, i32_type, location);
        assert_eq!(UnaryExpressionOperation::operand(&op), lhs);
        assert_eq!(UnaryExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        block.append_operation(op);

        let op = subscript(array, &[lhs.as_ref()], lvalue_i32_type, location);
        assert_eq!(op.value(), array);
        assert_eq!(op.indices(), vec![lhs.as_ref()]);
        assert_eq!(SubscriptOperation::output(&op).r#type(), lvalue_i32_type.as_ref());
        block.append_operation(op);

        let op = assign(lvalue, lhs, location);
        assert_eq!(op.variable(), lvalue);
        assert_eq!(op.value(), lhs);
        block.append_operation(op);

        let op = verbatim("value = {}", &[lhs.as_ref()], location);
        assert_eq!(op.value().as_str().unwrap(), "value = {}");
        assert_eq!(op.format_arguments(), vec![lhs.as_ref()]);
        block.append_operation(op);

        let mut expression_region = context.region();
        let mut expression_block = context.block_with_no_arguments();
        let expression_value =
            expression_block.append_operation(constant(context.integer_attribute(i32_type, 4), i32_type, location));
        let op = r#yield(Some(expression_value.result(0).unwrap().as_ref()), location);
        assert_eq!(op.value(), Some(expression_value.result(0).unwrap().as_ref()));
        expression_block.append_operation(op);
        expression_region.append_block(expression_block);
        let op = expression(&[lhs.as_ref()], i32_type.as_ref(), expression_region.into(), true, location);
        assert_eq!(op.definitions(), vec![lhs.as_ref()]);
        assert!(op.do_not_inline());
        assert_eq!(ExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        assert_eq!(op.body().blocks().count(), 1);

        block.append_operation(r#return(Option::<ValueRef<'_, '_, '_>>::None, location));
        module
            .body()
            .append_operation(func("expression_ops", function_type, block.into(), None, None, None, location));
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @expression_ops(%arg0: i32, %arg1: i32, %arg2: i1) {
                    %0 = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
                    %1 = literal "values" : !emitc.array<4xi32>
                    %2 = "emitc.variable"() <{value = #emitc.opaque<"box">}> : () -> !emitc.lvalue<!emitc.opaque<"Box">>
                    %3 = address_of %2 : !emitc.lvalue<!emitc.opaque<"Box">>
                    %4 = "emitc.variable"() <{value = #emitc.opaque<"box_ptr">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"Box">>>
                    %5 = address_of %0 : !emitc.lvalue<i32>
                    %6 = apply "&"(%0) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
                    %7 = bitwise_and %arg0, %arg1 : (i32, i32) -> i32
                    %8 = bitwise_left_shift %arg0, %arg1 : (i32, i32) -> i32
                    %9 = bitwise_not %arg0 : (i32) -> i32
                    %10 = bitwise_or %arg0, %arg1 : (i32, i32) -> i32
                    %11 = bitwise_right_shift %arg0, %arg1 : (i32, i32) -> i32
                    %12 = bitwise_xor %arg0, %arg1 : (i32, i32) -> i32
                    %13 = cast %arg0 : i32 to !emitc.opaque<"int">
                    %14 = cmp lt, %arg0, %arg1 : (i32, i32) -> i1
                    %15 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
                    %16 = dereference %5 : !emitc.ptr<i32>
                    %17 = div %arg0, %arg1 : (i32, i32) -> i32
                    %18 = logical_and %arg2, %arg2 : i1, i1
                    %19 = logical_not %arg2 : i1
                    %20 = logical_or %arg2, %arg2 : i1, i1
                    %21 = load %0 : <i32>
                    %22 = mul %arg0, %arg1 : (i32, i32) -> i32
                    %23 = rem %arg0, %arg1 : (i32, i32) -> i32
                    %24 = sub %arg0, %arg1 : (i32, i32) -> i32
                    %25 = "emitc.member"(%2) <{member = "field"}> : (!emitc.lvalue<!emitc.opaque<"Box">>) -> !emitc.lvalue<i32>
                    %26 = "emitc.member_of_ptr"(%4) <{member = "field"}> : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"Box">>>) -> !emitc.lvalue<i32>
                    %27 = conditional %arg2, %arg0, %arg1 : i32
                    %28 = unary_minus %arg0 : (i32) -> i32
                    %29 = unary_plus %arg0 : (i32) -> i32
                    %30 = subscript %1[%arg0] : (!emitc.array<4xi32>, i32) -> !emitc.lvalue<i32>
                    assign %arg0 : i32 to %0 : <i32>
                    verbatim "value = {}" args %arg0 : i32
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_call_symbol_and_region_operation_constructors_and_accessors() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let block = context.block(&[(i32_type.as_ref(), location)]);
        let argument = block.argument(0).unwrap();

        let empty_array_attribute = context.array_attribute::<AttributeRef>(&[]);
        let op = call_opaque(
            "opaque",
            &[argument.as_ref()],
            &[i32_type.as_ref()],
            Some(empty_array_attribute),
            Some(empty_array_attribute),
            location,
        );
        assert_eq!(op.callee().as_str().unwrap(), "opaque");
        assert!(op.args().unwrap().is_empty());
        assert!(op.template_args().unwrap().is_empty());
        assert_eq!(op.arguments(), vec![argument.as_ref()]);
        assert_eq!(op.outputs().len(), 1);
        assert_eq!(op.outputs()[0].r#type(), i32_type.as_ref());

        let op = call("callee", &[argument.as_ref()], &[i32_type.as_ref()], None, None, location);
        assert_eq!(op.callee().reference().as_str().unwrap(), "callee");
        assert_eq!(op.arguments(), vec![argument.as_ref()]);
        assert_eq!(op.outputs().len(), 1);
        assert_eq!(op.outputs()[0].r#type(), i32_type.as_ref());

        let op = declare_func("external", location);
        assert_eq!(op.symbol_name().reference().as_str().unwrap(), "external");

        let op = include("stdint.h", true, location);
        assert_eq!(op.include().as_str().unwrap(), "stdint.h");
        assert!(op.is_standard_include());

        let op = include("local.h", false, location);
        assert_eq!(op.include().as_str().unwrap(), "local.h");
        assert!(!op.is_standard_include());

        let op = literal("nullptr", context.emit_c_opaque_type("int *"), location);
        assert_eq!(op.value().as_str().unwrap(), "nullptr");
        assert_eq!(LiteralOperation::output(&op).r#type(), context.emit_c_opaque_type("int *").as_ref());

        let op = expression(
            &[argument.as_ref()],
            i32_type.as_ref(),
            context.block_with_no_arguments().into(),
            true,
            location,
        );
        assert_eq!(op.definitions(), vec![argument.as_ref()]);
        assert!(op.do_not_inline());
        assert_eq!(ExpressionOperation::output(&op).r#type(), i32_type.as_ref());
        assert_eq!(op.body().blocks().count(), 1);

        let op = expression(
            &[argument.as_ref()],
            i32_type.as_ref(),
            context.block_with_no_arguments().into(),
            false,
            location,
        );
        assert_eq!(op.definitions(), vec![argument.as_ref()]);
        assert!(!op.do_not_inline());
        assert_eq!(ExpressionOperation::output(&op).r#type(), i32_type.as_ref());

        let op = r#for(argument, argument, argument, context.block_with_no_arguments().into(), location);
        assert_eq!(op.lower_bound(), argument);
        assert_eq!(op.upper_bound(), argument);
        assert_eq!(op.step(), argument);
        assert_eq!(op.body().blocks().count(), 1);

        let op = file("generated.cc", context.block_with_no_arguments().into(), location);
        assert_eq!(op.id().as_str().unwrap(), "generated.cc");
        assert_eq!(op.body().blocks().count(), 1);

        let initial_value = context.integer_attribute(i32_type, 7).as_ref();
        let op = global("counter", i32_type, Some(initial_value), false, true, true, location);
        assert_eq!(op.symbol_name().as_str().unwrap(), "counter");
        assert_eq!(op.r#type(), i32_type.as_ref());
        assert_eq!(op.initial_value(), Some(initial_value));
        assert!(!op.extern_specifier());
        assert!(op.static_specifier());
        assert!(op.const_specifier());

        let op = global("external_counter", i32_type, None, true, false, false, location);
        assert_eq!(op.symbol_name().as_str().unwrap(), "external_counter");
        assert_eq!(op.r#type(), i32_type.as_ref());
        assert!(op.initial_value().is_none());
        assert!(op.extern_specifier());
        assert!(!op.static_specifier());
        assert!(!op.const_specifier());

        let lvalue_i32_type = context.emit_c_lvalue_type(i32_type);
        let op = get_global("counter", lvalue_i32_type, location);
        assert_eq!(GetGlobalOperation::name(&op).reference().as_str().unwrap(), "counter");
        assert_eq!(GetGlobalOperation::output(&op).r#type(), lvalue_i32_type.as_ref());

        let op = r#if(
            argument,
            context.block_with_no_arguments().into(),
            context.block_with_no_arguments().into(),
            location,
        );
        assert_eq!(op.condition(), argument);
        assert_eq!(op.then_region().blocks().count(), 1);
        assert_eq!(op.else_region().blocks().count(), 1);

        let op = switch(
            argument,
            &[1, 2],
            context.block_with_no_arguments().into(),
            vec![context.block_with_no_arguments().into(), context.block_with_no_arguments().into()],
            location,
        );
        assert_eq!(op.argument(), argument);
        assert_eq!(op.cases().values().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(op.default_region().blocks().count(), 1);
        assert_eq!(op.case_regions().len(), 2);

        let op = field("value", i32_type, Some(initial_value), location);
        assert_eq!(op.symbol_name().as_str().unwrap(), "value");
        assert_eq!(op.r#type(), i32_type.as_ref());
        assert_eq!(op.initial_value(), Some(initial_value));

        let op = field("empty", i32_type, None, location);
        assert_eq!(op.symbol_name().as_str().unwrap(), "empty");
        assert_eq!(op.r#type(), i32_type.as_ref());
        assert!(op.initial_value().is_none());

        let op = class("Box", true, context.block_with_no_arguments().into(), location);
        assert_eq!(op.symbol_name().as_str().unwrap(), "Box");
        assert!(op.final_specifier());
        assert_eq!(op.body().blocks().count(), 1);

        let op = class("OpenBox", false, context.block_with_no_arguments().into(), location);
        assert_eq!(op.symbol_name().as_str().unwrap(), "OpenBox");
        assert!(!op.final_specifier());
        assert_eq!(op.body().blocks().count(), 1);

        let op = get_field("value", i32_type, location);
        assert_eq!(op.field_name().reference().as_str().unwrap(), "value");
        assert_eq!(GetFieldOperation::output(&op).r#type(), i32_type.as_ref());

        let op = r#do(context.block_with_no_arguments().into(), context.block_with_no_arguments().into(), location);
        assert_eq!(op.body().blocks().count(), 1);
        assert_eq!(op.condition().blocks().count(), 1);

        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[]);
        let op = func("empty", function_type, context.block_with_no_arguments().into(), None, None, None, location);
        assert_eq!(op.symbol_name().as_str().unwrap(), "empty");
        assert_eq!(op.function_type(), function_type);
        assert!(op.specifiers().is_none());

        let specifiers = context.array_attribute(&[context.string_attribute("static")]);
        let op = func(
            "static_empty",
            function_type,
            context.block_with_no_arguments().into(),
            Some(specifiers),
            None,
            None,
            location,
        );
        assert_eq!(op.symbol_name().as_str().unwrap(), "static_empty");
        assert_eq!(op.function_type(), function_type);
        assert_eq!(op.specifiers().unwrap().len(), 1);

        let op = r#return(Option::<ValueRef<'_, '_, '_>>::None, location);
        assert!(op.value().is_none());

        let op = r#yield(Option::<ValueRef<'_, '_, '_>>::None, location);
        assert!(op.value().is_none());

        let module = context.module(location);
        module.body().append_operation(include("stdint.h", true, location));
        module.body().append_operation(include("local.h", false, location));
        module.body().append_operation(declare_func("empty", location));
        module
            .body()
            .append_operation(global("counter", i32_type, Some(initial_value), false, true, true, location));
        let mut body = context.block_with_no_arguments();
        body.append_operation(r#return(Option::<ValueRef<'_, '_, '_>>::None, location));
        module
            .body()
            .append_operation(func("empty", function_type, body.into(), None, None, None, location));
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.include <"stdint.h">
                  emitc.include "local.h"
                  emitc.declare_func @empty
                  emitc.global static const @counter : i32 = 7
                  emitc.func @empty() {
                    return
                  }
                }
            "#},
        );
    }
}
