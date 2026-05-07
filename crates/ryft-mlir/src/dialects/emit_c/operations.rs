use crate::{
    ArrayAttributeRef, Attribute, AttributeRef, DenseInteger64ArrayAttributeRef, DetachedOp, DetachedRegion,
    DialectHandle, Error, FlatSymbolRefAttributeRef, FunctionTypeRef, Location, Operation, OperationBuilder, RegionRef,
    StringAttributeRef, StringRef, TryIntoWithContext, Type, TypeRef, Value, ValueRef, mlir_op, mlir_op_trait,
};

use super::{CmpPredicate, CmpPredicateAttributeRef};

/// Name of the `emitc.file` id attribute.
pub const FILE_ID_ATTRIBUTE: &str = "id";

/// Operation trait for `emitc.file`.
pub trait FileOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the file identifier used by `mlir-translate-file-id`.
    fn id(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(FILE_ID_ATTRIBUTE)?.string())
    }

    /// Returns the file body region.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }
}

mlir_op!(File);
mlir_op_trait!(File, IsolatedFromAbove);
mlir_op_trait!(File, NoRegionArguments);
mlir_op_trait!(File, OneRegion);
mlir_op_trait!(File, SymbolTable);
mlir_op_trait!(File, ZeroSuccessors);

/// Constructs a new detached [`FileOperation`].
pub fn file<'c, 't: 'c, I: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    id: I,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedFileOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.file", location)
        .add_attribute(FILE_ID_ATTRIBUTE, id.try_into_with_context(context)?)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::file`"))
        })
}

/// Common API for single-operand Emit-C expression operations.
pub trait UnaryExpressionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns this operation's operand.
    fn operand(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }
}

/// Common API for two-operand Emit-C expression operations.
pub trait BinaryExpressionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand-side operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right-hand-side operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
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
) -> Result<DetachedAddressOfOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.address_of", location)
        .add_operand(reference)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::address_of`"))
        })
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
) -> Result<DetachedAddOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.add", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::add`"))
        })
}

/// Name of the `emitc.apply` operator attribute.
pub const APPLICABLE_OPERATOR_ATTRIBUTE: &str = "applicableOperator";

/// Operation trait for `emitc.apply`.
pub trait ApplyOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {
    /// Returns the operator applied by this deprecated operation.
    fn applicable_operator(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(APPLICABLE_OPERATOR_ATTRIBUTE)?.string())
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
    O: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    applicable_operator: O,
    operand: Operand,
    result_type: ResultType,
    location: L,
) -> Result<DetachedApplyOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.apply", location)
        .add_attribute(APPLICABLE_OPERATOR_ATTRIBUTE, applicable_operator.try_into_with_context(context)?)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::apply`"))
        })
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
) -> Result<DetachedBitwiseAndOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.bitwise_and", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::bitwise_and`"))
        })
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
) -> Result<DetachedBitwiseLeftShiftOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.bitwise_left_shift", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::bitwise_left_shift`"))
        })
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
) -> Result<DetachedBitwiseNotOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.bitwise_not", location)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::bitwise_not`"))
        })
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
) -> Result<DetachedBitwiseOrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.bitwise_or", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::bitwise_or`"))
        })
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
) -> Result<DetachedBitwiseRightShiftOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.bitwise_right_shift", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::bitwise_right_shift`"))
        })
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
) -> Result<DetachedBitwiseXorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.bitwise_xor", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::bitwise_xor`"))
        })
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
    fn callee(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(CALLEE_ATTRIBUTE)?.string())
    }

    /// Returns the optional argument-order attribute.
    fn args(&self) -> Result<Option<ArrayAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(ARGS_ATTRIBUTE) { self.array_attribute(ARGS_ATTRIBUTE).map(Some) } else { Ok(None) }
    }

    /// Returns the optional template-arguments attribute.
    fn template_args(&self) -> Result<Option<ArrayAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(TEMPLATE_ARGS_ATTRIBUTE) {
            self.array_attribute(TEMPLATE_ARGS_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the call operands.
    fn arguments(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns the call results.
    fn outputs(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.results().map(|result| result.map(|result| result.as_ref())).collect()
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
    C: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    callee: C,
    operands: &[ValueRef<'operand, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    args: Option<ArrayAttributeRef<'c, 't>>,
    template_args: Option<ArrayAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedCallOpaqueOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let mut builder = OperationBuilder::new("emitc.call_opaque", location)
        .add_attribute(CALLEE_ATTRIBUTE, callee.try_into_with_context(context)?)
        .add_operands(operands)
        .add_results(result_types);
    if let Some(args) = args {
        builder = builder.add_attribute(ARGS_ATTRIBUTE, args);
    }
    if let Some(template_args) = template_args {
        builder = builder.add_attribute(TEMPLATE_ARGS_ATTRIBUTE, template_args);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::call_opaque`"))
    })
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
) -> Result<DetachedCastOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.cast", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::cast`"))
        })
}

/// Name of the `emitc.cmp` predicate attribute.
pub const CMP_PREDICATE_ATTRIBUTE: &str = "predicate";

/// Operation trait for `emitc.cmp`.
pub trait CmpOperation<'o, 'c: 'o, 't: 'c>: BinaryExpressionOperation<'o, 'c, 't> {
    /// Returns the comparison predicate.
    fn predicate(&self) -> Result<CmpPredicate, Error> {
        self.attribute(CMP_PREDICATE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<CmpPredicateAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    CMP_PREDICATE_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>")
                ))
            })?
            .value()
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
) -> Result<DetachedCmpOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.cmp", location)
        .add_attribute(CMP_PREDICATE_ATTRIBUTE, context.emit_c_cmp_predicate_attribute(predicate)?)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::cmp`"))
        })
}

/// Name of the `emitc.constant` and `emitc.variable` value attribute.
pub const VALUE_ATTRIBUTE: &str = "value";

/// Operation trait for `emitc.constant`.
pub trait ConstantOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the constant value attribute.
    fn value(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute(VALUE_ATTRIBUTE).and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                VALUE_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>")
            ))
        })
    }

    /// Returns this operation's result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
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
) -> Result<DetachedConstantOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.constant", location)
        .add_attribute(VALUE_ATTRIBUTE, value)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::constant`"))
        })
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
) -> Result<DetachedDereferenceOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.dereference", location)
        .add_operand(pointer)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::dereference`"))
        })
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
) -> Result<DetachedDivOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.div", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::div`"))
        })
}

/// Name of the `emitc.expression` no-inline attribute.
pub const DO_NOT_INLINE_ATTRIBUTE: &str = "do_not_inline";

/// Operation trait for `emitc.expression`.
pub trait ExpressionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the operands passed as block arguments to the expression body.
    fn definitions(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns whether this expression must not be emitted inline at its use.
    fn do_not_inline(&self) -> bool {
        self.has_attribute(DO_NOT_INLINE_ATTRIBUTE)
    }

    /// Returns this operation's result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the expression body region.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
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
) -> Result<DetachedExpressionOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let mut builder = OperationBuilder::new("emitc.expression", location)
        .add_operands(definitions)
        .add_result(result_type)
        .add_region(region);
    if do_not_inline {
        builder = builder.add_attribute(DO_NOT_INLINE_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::expression`"))
    })
}

/// Operation trait for `emitc.for`.
pub trait ForOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the lower bound.
    fn lower_bound(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the upper bound.
    fn upper_bound(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the loop step.
    fn step(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the loop body region.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
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
) -> Result<DetachedForOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.for", location)
        .add_operands(&[lower_bound.as_ref(), upper_bound.as_ref(), step.as_ref()])
        .add_region(region)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::for`"))
        })
}

/// Operation trait for `emitc.call`.
pub trait CallOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the callee symbol.
    fn callee(&self) -> Result<FlatSymbolRefAttributeRef<'c, 't>, Error> {
        self.flat_symbol_ref_attribute(CALLEE_ATTRIBUTE)
    }

    /// Returns the call operands.
    fn arguments(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns the call results.
    fn outputs(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.results().map(|result| result.map(|result| result.as_ref())).collect()
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
    C: TryIntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    callee: C,
    operands: &[ValueRef<'operand, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    arg_attrs: Option<ArrayAttributeRef<'c, 't>>,
    res_attrs: Option<ArrayAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedCallOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let mut builder = OperationBuilder::new("emitc.call", location)
        .add_attribute(CALLEE_ATTRIBUTE, callee.try_into_with_context(context)?)
        .add_operands(operands)
        .add_results(result_types);
    if let Some(arg_attrs) = arg_attrs {
        builder = builder.add_attribute(ARG_ATTRS_ATTRIBUTE, arg_attrs);
    }
    if let Some(res_attrs) = res_attrs {
        builder = builder.add_attribute(RES_ATTRS_ATTRIBUTE, res_attrs);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::call`"))
    })
}

/// Operation trait for `emitc.declare_func`.
pub trait DeclareFuncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the declared function symbol.
    fn symbol_name(&self) -> Result<FlatSymbolRefAttributeRef<'c, 't>, Error> {
        self.flat_symbol_ref_attribute(SYMBOL_NAME_ATTRIBUTE)
    }
}

mlir_op!(DeclareFunc);
mlir_op_trait!(DeclareFunc, ZeroRegions);
mlir_op_trait!(DeclareFunc, ZeroSuccessors);

/// Constructs a new detached [`DeclareFuncOperation`].
pub fn declare_func<
    'c,
    't: 'c,
    S: TryIntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    symbol_name: S,
    location: L,
) -> Result<DetachedDeclareFuncOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.declare_func", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, symbol_name.try_into_with_context(context)?)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::declare_func`"))
        })
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
    fn symbol_name(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(SYMBOL_NAME_ATTRIBUTE)?.string())
    }

    /// Returns the function type.
    fn function_type(&self) -> Result<FunctionTypeRef<'c, 't>, Error> {
        self.type_attribute(FUNCTION_TYPE_ATTRIBUTE)?
            .r#type()?
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid `function_type` attribute in `emitc.func`"))
    }

    /// Returns optional C/C++ specifiers such as `static` or `inline`.
    fn specifiers(&self) -> Result<Option<ArrayAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(SPECIFIERS_ATTRIBUTE) {
            self.array_attribute(SPECIFIERS_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the function body region.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }
}

mlir_op!(Func);
mlir_op_trait!(Func, AutomaticAllocationScope);
mlir_op_trait!(Func, IsolatedFromAbove);
mlir_op_trait!(Func, OneRegion);
mlir_op_trait!(Func, ZeroSuccessors);

/// Constructs a new detached [`FuncOperation`].
pub fn func<'c, 't: 'c, N: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    name: N,
    function_type: FunctionTypeRef<'c, 't>,
    body: DetachedRegion<'c, 't>,
    specifiers: Option<ArrayAttributeRef<'c, 't>>,
    arg_attrs: Option<ArrayAttributeRef<'c, 't>>,
    res_attrs: Option<ArrayAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedFuncOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let mut builder = OperationBuilder::new("emitc.func", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, name.try_into_with_context(context)?)
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
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::func`"))
    })
}

/// Operation trait for `emitc.return`.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional returned value.
    fn value(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() == 0 { Ok(None) } else { self.operand_value(0).map(Some) }
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
) -> Result<DetachedReturnOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let mut builder = OperationBuilder::new("emitc.return", location);
    if let Some(value) = value {
        builder = builder.add_operand(value);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::return`"))
    })
}

/// Name of the `emitc.include` include attribute.
pub const INCLUDE_ATTRIBUTE: &str = "include";

/// Name of the `emitc.include` standard-include marker.
pub const IS_STANDARD_INCLUDE_ATTRIBUTE: &str = "is_standard_include";

/// Operation trait for `emitc.include`.
pub trait IncludeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the included path or header name.
    fn include(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(INCLUDE_ATTRIBUTE)?.string())
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
pub fn include<'c, 't: 'c, I: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    include: I,
    is_standard_include: bool,
    location: L,
) -> Result<DetachedIncludeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let mut builder = OperationBuilder::new("emitc.include", location)
        .add_attribute(INCLUDE_ATTRIBUTE, include.try_into_with_context(context)?);
    if is_standard_include {
        builder = builder.add_attribute(IS_STANDARD_INCLUDE_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::include`"))
    })
}

/// Operation trait for `emitc.literal`.
pub trait LiteralOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the literal source spelling.
    fn value(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(VALUE_ATTRIBUTE)?.string())
    }

    /// Returns this operation's result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
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
    V: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    value: V,
    result_type: ResultType,
    location: L,
) -> Result<DetachedLiteralOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.literal", location)
        .add_attribute(VALUE_ATTRIBUTE, value.try_into_with_context(context)?)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::literal`"))
        })
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
) -> Result<DetachedLogicalAndOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let result_type = context.signless_integer_type(1);
    OperationBuilder::new("emitc.logical_and", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::logical_and`"))
        })
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
) -> Result<DetachedLogicalNotOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let result_type = context.signless_integer_type(1);
    OperationBuilder::new("emitc.logical_not", location)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::logical_not`"))
        })
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
) -> Result<DetachedLogicalOrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let result_type = context.signless_integer_type(1);
    OperationBuilder::new("emitc.logical_or", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::logical_or`"))
        })
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
) -> Result<DetachedLoadOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.load", location)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::load`"))
        })
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
) -> Result<DetachedMulOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.mul", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::mul`"))
        })
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
) -> Result<DetachedRemOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.rem", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::rem`"))
        })
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
) -> Result<DetachedSubOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.sub", location)
        .add_operands(&[lhs.as_ref(), rhs.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::sub`"))
        })
}

/// Name of member-selection attributes.
pub const MEMBER_ATTRIBUTE: &str = "member";

/// Operation trait for `emitc.member`.
pub trait MemberOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {
    /// Returns the selected member name.
    fn member(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(MEMBER_ATTRIBUTE)?.string())
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
    M: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    member: M,
    operand: Operand,
    result_type: ResultType,
    location: L,
) -> Result<DetachedMemberOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.member", location)
        .add_attribute(MEMBER_ATTRIBUTE, member.try_into_with_context(context)?)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::member`"))
        })
}

/// Operation trait for `emitc.member_of_ptr`.
pub trait MemberOfPtrOperation<'o, 'c: 'o, 't: 'c>: UnaryExpressionOperation<'o, 'c, 't> {
    /// Returns the selected member name.
    fn member(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(MEMBER_ATTRIBUTE)?.string())
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
    M: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    member: M,
    operand: Operand,
    result_type: ResultType,
    location: L,
) -> Result<DetachedMemberOfPtrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.member_of_ptr", location)
        .add_attribute(MEMBER_ATTRIBUTE, member.try_into_with_context(context)?)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::member_of_ptr`"))
        })
}

/// Operation trait for `emitc.conditional`.
pub trait ConditionalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the condition operand.
    fn condition(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the value used when the condition is true.
    fn true_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the value used when the condition is false.
    fn false_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
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
) -> Result<DetachedConditionalOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.conditional", location)
        .add_operands(&[condition.as_ref(), true_value.as_ref(), false_value.as_ref()])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::conditional`"))
        })
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
) -> Result<DetachedUnaryMinusOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.unary_minus", location)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::unary_minus`"))
        })
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
) -> Result<DetachedUnaryPlusOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.unary_plus", location)
        .add_operand(operand)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::unary_plus`"))
        })
}

/// Operation trait for `emitc.variable`.
pub trait VariableOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the initial value attribute.
    fn value(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute(VALUE_ATTRIBUTE).and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                VALUE_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>")
            ))
        })
    }

    /// Returns this operation's allocated result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
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
) -> Result<DetachedVariableOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.variable", location)
        .add_attribute(VALUE_ATTRIBUTE, value)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::variable`"))
        })
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
    fn symbol_name(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(SYMBOL_NAME_ATTRIBUTE)?.string())
    }

    /// Returns the global variable type.
    fn r#type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.type_attribute(TYPE_ATTRIBUTE)?.r#type()
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
    N: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
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
) -> Result<DetachedGlobalOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let mut builder = OperationBuilder::new("emitc.global", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, symbol_name.try_into_with_context(context)?)
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
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::global`"))
    })
}

/// Name of the symbol-reference attribute used by `emitc.get_global`.
pub const NAME_ATTRIBUTE: &str = "name";

/// Operation trait for `emitc.get_global`.
pub trait GetGlobalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the referenced global name.
    fn name(&self) -> Result<FlatSymbolRefAttributeRef<'c, 't>, Error> {
        self.flat_symbol_ref_attribute(NAME_ATTRIBUTE)
    }

    /// Returns this operation's result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
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
    N: TryIntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    name: N,
    result_type: ResultType,
    location: L,
) -> Result<DetachedGetGlobalOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.get_global", location)
        .add_attribute(NAME_ATTRIBUTE, name.try_into_with_context(context)?)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::get_global`"))
        })
}

/// Operation trait for `emitc.verbatim`.
pub trait VerbatimOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the verbatim format string.
    fn value(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(VALUE_ATTRIBUTE)?.string())
    }

    /// Returns the format arguments.
    fn format_arguments(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
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
    V: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    value: V,
    format_arguments: &[ValueRef<'argument, 'c, 't>],
    location: L,
) -> Result<DetachedVerbatimOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.verbatim", location)
        .add_attribute(VALUE_ATTRIBUTE, value.try_into_with_context(context)?)
        .add_operands(format_arguments)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::verbatim`"))
        })
}

/// Operation trait for `emitc.assign`.
pub trait AssignOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the lvalue being assigned.
    fn variable(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the assigned value.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
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
) -> Result<DetachedAssignOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.assign", location)
        .add_operands(&[variable.as_ref(), value.as_ref()])
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::assign`"))
        })
}

/// Operation trait for `emitc.yield`.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional yielded value.
    fn value(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() == 0 { Ok(None) } else { self.operand_value(0).map(Some) }
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Constructs a new detached [`YieldOperation`].
pub fn r#yield<'value, 'c: 'value, 't: 'c, L: Location<'c, 't>>(
    value: Option<ValueRef<'value, 'c, 't>>,
    location: L,
) -> Result<DetachedYieldOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let mut builder = OperationBuilder::new("emitc.yield", location);
    if let Some(value) = value {
        builder = builder.add_operand(value);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::yield`"))
    })
}

/// Operation trait for `emitc.if`.
pub trait IfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the condition.
    fn condition(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the then region.
    fn then_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns the else region.
    fn else_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(1)
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
) -> Result<DetachedIfOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.if", location)
        .add_operand(condition)
        .add_regions(vec![then_region, else_region])
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::if`"))
        })
}

/// Operation trait for `emitc.subscript`.
pub trait SubscriptOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the indexed value.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the index operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().skip(1).collect()
    }

    /// Returns this operation's result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
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
) -> Result<DetachedSubscriptOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.subscript", location)
        .add_operand(value)
        .add_operands(indices)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::subscript`"))
        })
}

/// Name of the `emitc.switch` cases attribute.
pub const CASES_ATTRIBUTE: &str = "cases";

/// Operation trait for `emitc.switch`.
pub trait SwitchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the switched value.
    fn argument(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the case values.
    fn cases(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(CASES_ATTRIBUTE)
    }

    /// Returns the default region.
    fn default_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns the case regions.
    fn case_regions(&self) -> Result<Vec<RegionRef<'o, 'c, 't>>, Error> {
        (1..self.region_count()).map(|index| self.region(index)).collect()
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
) -> Result<DetachedSwitchOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let mut regions = vec![default_region];
    regions.extend(case_regions);
    OperationBuilder::new("emitc.switch", location)
        .add_operand(argument)
        .add_attribute(CASES_ATTRIBUTE, context.dense_i64_array_attribute(cases)?)
        .add_regions(regions)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::switch`"))
        })
}

/// Name of the `emitc.class` final specifier.
pub const FINAL_SPECIFIER_ATTRIBUTE: &str = "final_specifier";

/// Operation trait for `emitc.class`.
pub trait ClassOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the class symbol name.
    fn symbol_name(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(SYMBOL_NAME_ATTRIBUTE)?.string())
    }

    /// Returns whether this class has a `final` specifier.
    fn final_specifier(&self) -> bool {
        self.has_attribute(FINAL_SPECIFIER_ATTRIBUTE)
    }

    /// Returns the class body.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
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
pub fn class<'c, 't: 'c, N: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    symbol_name: N,
    final_specifier: bool,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedClassOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let mut builder = OperationBuilder::new("emitc.class", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, symbol_name.try_into_with_context(context)?)
        .add_region(body);
    if final_specifier {
        builder = builder.add_attribute(FINAL_SPECIFIER_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::class`"))
    })
}

/// Operation trait for `emitc.field`.
pub trait FieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the field symbol name.
    fn symbol_name(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(SYMBOL_NAME_ATTRIBUTE)?.string())
    }

    /// Returns the field type.
    fn r#type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.type_attribute(TYPE_ATTRIBUTE)?.r#type()
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
    N: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    FieldType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    symbol_name: N,
    r#type: FieldType,
    initial_value: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedFieldOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    let mut builder = OperationBuilder::new("emitc.field", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, symbol_name.try_into_with_context(context)?)
        .add_attribute(TYPE_ATTRIBUTE, context.type_attribute(r#type));
    if let Some(initial_value) = initial_value {
        builder = builder.add_attribute(INITIAL_VALUE_ATTRIBUTE, initial_value);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::field`"))
    })
}

/// Name of the `emitc.get_field` field-name attribute.
pub const FIELD_NAME_ATTRIBUTE: &str = "field_name";

/// Operation trait for `emitc.get_field`.
pub trait GetFieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the referenced field name.
    fn field_name(&self) -> Result<FlatSymbolRefAttributeRef<'c, 't>, Error> {
        self.flat_symbol_ref_attribute(FIELD_NAME_ATTRIBUTE)
    }

    /// Returns this operation's result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
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
    N: TryIntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    ResultType: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    field_name: N,
    result_type: ResultType,
    location: L,
) -> Result<DetachedGetFieldOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.get_field", location)
        .add_attribute(FIELD_NAME_ATTRIBUTE, field_name.try_into_with_context(context)?)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::get_field`"))
        })
}

/// Operation trait for `emitc.do`.
pub trait DoOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the body region.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns the condition region.
    fn condition(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(1)
    }
}

mlir_op!(Do);
mlir_op_trait!(Do, ZeroSuccessors);

/// Constructs a new detached [`DoOperation`].
pub fn r#do<'c, 't: 'c, L: Location<'c, 't>>(
    body: DetachedRegion<'c, 't>,
    condition: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedDoOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::emit_c()?);
    OperationBuilder::new("emitc.do", location).add_regions(vec![body, condition]).build().and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `emit_c::do`"))
        },
    )
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::{Block, Context, Operation, Region, Type, TypeRef, Value, ValueRef};

    use super::*;

    #[test]
    fn test_file() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let mut body = context.block_with_no_arguments();
        body.append_operation(include("stdint.h", true, location).unwrap()).unwrap();
        let op = file("generated.cc", body.try_into().unwrap(), location).unwrap();
        assert_eq!(op.id().unwrap().as_str().unwrap(), "generated.cc");
        assert_eq!(op.body().unwrap().blocks().count(), 1);
        module.body().unwrap().append_operation(op).unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.file "generated.cc" {
                    include <"stdint.h">
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_address_of() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let lvalue_i32_type = context.emit_c_lvalue_type(i32_type).unwrap();
        let pointer_i32_type = context.emit_c_pointer_type(i32_type).unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[pointer_i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let variable_op = block
                    .append_operation(
                        variable(context.integer_attribute(i32_type, 0), lvalue_i32_type, location).unwrap(),
                    )
                    .unwrap();
                let op =
                    address_of(variable_op.as_ref().result(0).unwrap().as_ref(), pointer_i32_type, location).unwrap();
                assert_eq!(op.operand_value(0).unwrap(), variable_op.as_ref().result(0).unwrap().as_ref());
                assert_eq!(op.operand_value(0).unwrap(), variable_op.as_ref().result(0).unwrap().as_ref());
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("address_of", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @address_of() -> !emitc.ptr<i32> {
                    %0 = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
                    %1 = address_of %0 : !emitc.lvalue<i32>
                    return %1 : !emitc.ptr<i32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_add() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = add(lhs, rhs, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("add", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @add(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = add %arg0, %arg1 : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_apply() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let lvalue_i32_type = context.emit_c_lvalue_type(i32_type).unwrap();
        let pointer_i32_type = context.emit_c_pointer_type(i32_type).unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[pointer_i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let variable_op = block
                    .append_operation(
                        variable(context.integer_attribute(i32_type, 0), lvalue_i32_type, location).unwrap(),
                    )
                    .unwrap();
                let variable_value = variable_op.as_ref().result(0).unwrap().as_ref();
                let op = apply("&", variable_value, pointer_i32_type, location).unwrap();
                assert_eq!(op.applicable_operator().unwrap().as_str().unwrap(), "&");
                assert_eq!(op.operand_value(0).unwrap(), variable_value);
                assert_eq!(op.applicable_operator().unwrap().as_str().unwrap(), "&");
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("apply", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @apply() -> !emitc.ptr<i32> {
                    %0 = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
                    %1 = apply "&"(%0) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
                    return %1 : !emitc.ptr<i32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_bitwise_and() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = bitwise_and(lhs, rhs, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("bitwise_and", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @bitwise_and(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = bitwise_and %arg0, %arg1 : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_bitwise_left_shift() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = bitwise_left_shift(lhs, rhs, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("bitwise_left_shift", function_type, block.try_into().unwrap(), None, None, None, location)
                    .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @bitwise_left_shift(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = bitwise_left_shift %arg0, %arg1 : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_bitwise_not() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type = context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let operand = block.argument(0).unwrap();
                let op = bitwise_not(operand, i32_type, location).unwrap();
                assert_eq!(op.operand_value(0).unwrap(), operand);
                assert_eq!(op.operand_value(0).unwrap(), operand);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("bitwise_not", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @bitwise_not(%arg0: i32) -> i32 {
                    %0 = bitwise_not %arg0 : (i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_bitwise_or() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = bitwise_or(lhs, rhs, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("bitwise_or", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @bitwise_or(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = bitwise_or %arg0, %arg1 : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_bitwise_right_shift() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = bitwise_right_shift(lhs, rhs, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("bitwise_right_shift", function_type, block.try_into().unwrap(), None, None, None, location)
                    .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @bitwise_right_shift(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = bitwise_right_shift %arg0, %arg1 : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_bitwise_xor() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = bitwise_xor(lhs, rhs, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("bitwise_xor", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @bitwise_xor(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = bitwise_xor %arg0, %arg1 : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_call_opaque() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type = context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let argument = block.argument(0).unwrap();
                let op =
                    call_opaque("opaque", &[argument.as_ref()], &[i32_type.as_ref()], None, None, location).unwrap();
                assert_eq!(op.callee().unwrap().as_str().unwrap(), "opaque");
                assert!(op.args().unwrap().is_none());
                assert!(op.template_args().unwrap().is_none());
                assert_eq!(op.arguments().unwrap(), vec![argument.as_ref()]);
                assert_eq!(op.outputs().unwrap().len(), 1);
                assert_eq!(op.outputs().unwrap()[0].r#type().unwrap(), i32_type.as_ref());
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("call_opaque", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();

        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @call_opaque(%arg0: i32) -> i32 {
                    %0 = call_opaque "opaque"(%arg0) : (i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_cast() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let opaque_int_type = context.emit_c_opaque_type("int").unwrap();
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref()], &[opaque_int_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let source = block.argument(0).unwrap();
                let op = cast(source, opaque_int_type, location).unwrap();
                assert_eq!(op.operand_value(0).unwrap(), source);
                assert_eq!(op.operand_value(0).unwrap(), source);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("cast", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @cast(%arg0: i32) -> !emitc.opaque<"int"> {
                    %0 = cast %arg0 : i32 to !emitc.opaque<"int">
                    return %0 : !emitc.opaque<"int">
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_cmp() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i1_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = cmp(CmpPredicate::LessThan, lhs, rhs, i1_type, location).unwrap();
                assert_eq!(op.predicate().unwrap(), CmpPredicate::LessThan);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.predicate().unwrap(), CmpPredicate::LessThan);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("cmp", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @cmp(%arg0: i32, %arg1: i32) -> i1 {
                    %0 = cmp lt, %arg0, %arg1 : (i32, i32) -> i1
                    return %0 : i1
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_constant() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = constant(context.integer_attribute(i32_type, 3), i32_type, location).unwrap();
                assert_eq!(op.value().unwrap().to_string(), "3 : i32");
                assert_eq!(op.value().unwrap().to_string(), "3 : i32");
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("constant", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @constant() -> i32 {
                    %0 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_dereference() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let lvalue_i32_type = context.emit_c_lvalue_type(i32_type).unwrap();
        let pointer_i32_type = context.emit_c_pointer_type(i32_type).unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let variable_op = block
                    .append_operation(
                        variable(context.integer_attribute(i32_type, 0), lvalue_i32_type, location).unwrap(),
                    )
                    .unwrap();
                let address_op = block
                    .append_operation(
                        address_of(variable_op.as_ref().result(0).unwrap().as_ref(), pointer_i32_type, location)
                            .unwrap(),
                    )
                    .unwrap();
                let op =
                    dereference(address_op.as_ref().result(0).unwrap().as_ref(), lvalue_i32_type, location).unwrap();
                assert_eq!(op.operand_value(0).unwrap(), address_op.as_ref().result(0).unwrap().as_ref());
                assert_eq!(op.operand_value(0).unwrap(), address_op.as_ref().result(0).unwrap().as_ref());
                let op = block.append_operation(op).unwrap();
                let load_op = block
                    .append_operation(load(op.as_ref().result(0).unwrap().as_ref(), i32_type, location).unwrap())
                    .unwrap();
                block
                    .append_operation(r#return(Some(load_op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("dereference", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @dereference() -> i32 {
                    %0 = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
                    %1 = address_of %0 : !emitc.lvalue<i32>
                    %2 = dereference %1 : !emitc.ptr<i32>
                    %3 = load %2 : <i32>
                    return %3 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_div() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = div(lhs, rhs, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("div", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @div(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = div %arg0, %arg1 : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_expression() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let mut region = context.region();
                let mut body = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let body_lhs = body.argument(0).unwrap();
                let body_rhs = body.argument(1).unwrap();
                let add_op = body.append_operation(add(body_lhs, body_rhs, i32_type, location).unwrap()).unwrap();
                body.append_operation(r#yield(Some(add_op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                region.append_block(body).unwrap();
                let op = expression(&[lhs.as_ref(), rhs.as_ref()], i32_type.as_ref(), region.into(), true, location)
                    .unwrap();
                assert_eq!(op.definitions().unwrap(), vec![lhs.as_ref(), rhs.as_ref()]);
                assert!(op.do_not_inline());
                assert_eq!(op.output().unwrap().r#type().unwrap(), i32_type.as_ref());
                assert_eq!(op.body().unwrap().blocks().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("expression", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();

        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @expression(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = expression %arg0, %arg1 : (i32, i32) -> i32 {
                      %1 = add %arg0, %arg1 : (i32, i32) -> i32
                      yield %1 : i32
                    }
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_for() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref(), i32_type.as_ref()], &[]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let lower_bound = block.argument(0).unwrap();
                let upper_bound = block.argument(1).unwrap();
                let step = block.argument(2).unwrap();
                let mut body = context.block(&[(i32_type.as_ref(), location)]);
                body.append_operation(r#yield(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap()).unwrap();
                let op = r#for(lower_bound, upper_bound, step, body.try_into().unwrap(), location).unwrap();
                assert_eq!(op.lower_bound().unwrap(), lower_bound);
                assert_eq!(op.upper_bound().unwrap(), upper_bound);
                assert_eq!(op.step().unwrap(), step);
                assert_eq!(op.body().unwrap().blocks().count(), 1);
                block.append_operation(op).unwrap();
                block.append_operation(r#return(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap()).unwrap();
                func("for", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @for(%arg0: i32, %arg1: i32, %arg2: i32) {
                    for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
                    }
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_call() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let callee_type = context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref()], &[i32_type.as_ref()]);
        let caller_type = context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let argument = block.argument(0).unwrap();
                block.append_operation(r#return(Some(argument.as_ref()), location).unwrap()).unwrap();
                func("callee", callee_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let argument = block.argument(0).unwrap();
                let op = call("callee", &[argument.as_ref()], &[i32_type.as_ref()], None, None, location).unwrap();
                assert_eq!(op.callee().unwrap().reference().as_str().unwrap(), "callee");
                assert_eq!(op.arguments().unwrap(), vec![argument.as_ref()]);
                assert_eq!(op.outputs().unwrap().len(), 1);
                assert_eq!(op.outputs().unwrap()[0].r#type().unwrap(), i32_type.as_ref());
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("caller", caller_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @callee(%arg0: i32) -> i32 {
                    return %arg0 : i32
                  }
                  emitc.func @caller(%arg0: i32) -> i32 {
                    %0 = call @callee(%arg0) : (i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_declare_func() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[]);
        let op = declare_func("declared", location).unwrap();
        assert_eq!(op.symbol_name().unwrap().reference().as_str().unwrap(), "declared");
        module.body().unwrap().append_operation(op).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                block.append_operation(r#return(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap()).unwrap();
                func("declared", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.declare_func @declared
                  emitc.func @declared() {
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_func() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[]);
        let specifiers = context.array_attribute(&[context.string_attribute("static")]);
        let mut block = context.block_with_no_arguments();
        block.append_operation(r#return(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap()).unwrap();
        let op =
            func("empty", function_type, block.try_into().unwrap(), Some(specifiers), None, None, location).unwrap();
        assert_eq!(op.symbol_name().unwrap().as_str().unwrap(), "empty");
        assert_eq!(op.function_type().unwrap(), function_type);
        assert_eq!(op.specifiers().unwrap().unwrap().len(), 1);
        assert_eq!(op.body().unwrap().blocks().count(), 1);
        module.body().unwrap().append_operation(op).unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @empty() attributes {specifiers = ["static"]} {
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_return() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let constant_op = block
                    .append_operation(constant(context.integer_attribute(i32_type, 1), i32_type, location).unwrap())
                    .unwrap();
                let op = r#return(Some(constant_op.as_ref().result(0).unwrap().as_ref()), location).unwrap();
                assert_eq!(op.value().unwrap(), Some(constant_op.as_ref().result(0).unwrap().as_ref()));
                block.append_operation(op).unwrap();
                func("return", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();

        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @return() -> i32 {
                    %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_include() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let op = include("stdint.h", true, location).unwrap();
        assert_eq!(op.include().unwrap().as_str().unwrap(), "stdint.h");
        assert!(op.is_standard_include());
        module.body().unwrap().append_operation(op).unwrap();

        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.include <"stdint.h">
                }
            "#},
        );
    }

    #[test]
    fn test_literal() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let opaque_pointer_type = context.emit_c_opaque_type("int *").unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[opaque_pointer_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = literal("nullptr", opaque_pointer_type, location).unwrap();
                assert_eq!(op.value().unwrap().as_str().unwrap(), "nullptr");
                assert_eq!(op.value().unwrap().as_str().unwrap(), "nullptr");
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("literal", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @literal() -> !emitc.opaque<"int *"> {
                    %0 = literal "nullptr" : !emitc.opaque<"int *">
                    return %0 : !emitc.opaque<"int *">
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_logical_and() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i1_type.as_ref(), i1_type.as_ref()], &[i1_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i1_type.as_ref(), location), (i1_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = logical_and(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("logical_and", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @logical_and(%arg0: i1, %arg1: i1) -> i1 {
                    %0 = logical_and %arg0, %arg1 : i1, i1
                    return %0 : i1
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_logical_not() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let function_type = context.function_type::<TypeRef, TypeRef>(&[i1_type.as_ref()], &[i1_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i1_type.as_ref(), location)]);
                let operand = block.argument(0).unwrap();
                let op = logical_not(operand, location).unwrap();
                assert_eq!(op.operand_value(0).unwrap(), operand);
                assert_eq!(op.operand_value(0).unwrap(), operand);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("logical_not", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @logical_not(%arg0: i1) -> i1 {
                    %0 = logical_not %arg0 : i1
                    return %0 : i1
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_logical_or() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i1_type.as_ref(), i1_type.as_ref()], &[i1_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i1_type.as_ref(), location), (i1_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = logical_or(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("logical_or", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @logical_or(%arg0: i1, %arg1: i1) -> i1 {
                    %0 = logical_or %arg0, %arg1 : i1, i1
                    return %0 : i1
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_load() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let lvalue_i32_type = context.emit_c_lvalue_type(i32_type).unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let variable_op = block
                    .append_operation(
                        variable(context.integer_attribute(i32_type, 0), lvalue_i32_type, location).unwrap(),
                    )
                    .unwrap();
                let variable_value = variable_op.as_ref().result(0).unwrap().as_ref();
                let op = load(variable_value, i32_type, location).unwrap();
                assert_eq!(op.operand_value(0).unwrap(), variable_value);
                assert_eq!(op.operand_value(0).unwrap(), variable_value);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("load", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @load() -> i32 {
                    %0 = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
                    %1 = load %0 : <i32>
                    return %1 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_mul() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = mul(lhs, rhs, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("mul", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @mul(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = mul %arg0, %arg1 : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_rem() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = rem(lhs, rhs, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("rem", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @rem(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = rem %arg0, %arg1 : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_sub() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = sub(lhs, rhs, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("sub", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @sub(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = sub %arg0, %arg1 : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_member() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let opaque_box_type = context.emit_c_opaque_type("Box").unwrap();
        let lvalue_box_type = context.emit_c_lvalue_type(opaque_box_type).unwrap();
        let lvalue_i32_type = context.emit_c_lvalue_type(i32_type).unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let variable_op = block
                    .append_operation(
                        variable(context.emit_c_opaque_attribute("box").unwrap(), lvalue_box_type, location).unwrap(),
                    )
                    .unwrap();
                let variable_value = variable_op.as_ref().result(0).unwrap().as_ref();
                let op = member("field", variable_value, lvalue_i32_type, location).unwrap();
                assert_eq!(op.member().unwrap().as_str().unwrap(), "field");
                assert_eq!(op.operand_value(0).unwrap(), variable_value);
                assert_eq!(op.member().unwrap().as_str().unwrap(), "field");
                let op = block.append_operation(op).unwrap();
                let load_op = block
                    .append_operation(load(op.as_ref().result(0).unwrap().as_ref(), i32_type, location).unwrap())
                    .unwrap();
                block
                    .append_operation(r#return(Some(load_op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("member", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @member() -> i32 {
                    %0 = "emitc.variable"() <{value = #emitc.opaque<"box">}> : () -> !emitc.lvalue<!emitc.opaque<"Box">>
                    %1 = "emitc.member"(%0) <{member = "field"}> : (!emitc.lvalue<!emitc.opaque<"Box">>) -> !emitc.lvalue<i32>
                    %2 = load %1 : <i32>
                    return %2 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_member_of_ptr() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let opaque_box_type = context.emit_c_opaque_type("Box").unwrap();
        let pointer_box_type = context.emit_c_pointer_type(opaque_box_type).unwrap();
        let lvalue_pointer_box_type = context.emit_c_lvalue_type(pointer_box_type).unwrap();
        let lvalue_i32_type = context.emit_c_lvalue_type(i32_type).unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let variable_op = block
                    .append_operation(
                        variable(
                            context.emit_c_opaque_attribute("box_ptr").unwrap(),
                            lvalue_pointer_box_type,
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                let variable_value = variable_op.as_ref().result(0).unwrap().as_ref();
                let op = member_of_ptr("field", variable_value, lvalue_i32_type, location).unwrap();
                assert_eq!(op.member().unwrap().as_str().unwrap(), "field");
                assert_eq!(op.operand_value(0).unwrap(), variable_value);
                assert_eq!(op.member().unwrap().as_str().unwrap(), "field");
                let op = block.append_operation(op).unwrap();
                let load_op = block
                    .append_operation(load(op.as_ref().result(0).unwrap().as_ref(), i32_type, location).unwrap())
                    .unwrap();
                block
                    .append_operation(r#return(Some(load_op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("member_of_ptr", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @member_of_ptr() -> i32 {
                    %0 = "emitc.variable"() <{value = #emitc.opaque<"box_ptr">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"Box">>>
                    %1 = "emitc.member_of_ptr"(%0) <{member = "field"}> : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"Box">>>) -> !emitc.lvalue<i32>
                    %2 = load %1 : <i32>
                    return %2 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_conditional() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        let function_type = context.function_type::<TypeRef, TypeRef>(
            &[i1_type.as_ref(), i32_type.as_ref(), i32_type.as_ref()],
            &[i32_type.as_ref()],
        );
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i1_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let condition = block.argument(0).unwrap();
                let true_value = block.argument(1).unwrap();
                let false_value = block.argument(2).unwrap();
                let op = conditional(condition, true_value, false_value, i32_type, location).unwrap();
                assert_eq!(op.condition().unwrap(), condition);
                assert_eq!(op.true_value().unwrap(), true_value);
                assert_eq!(op.false_value().unwrap(), false_value);
                assert_eq!(op.condition().unwrap(), condition);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("conditional", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @conditional(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
                    %0 = conditional %arg0, %arg1, %arg2 : i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_unary_minus() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type = context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let operand = block.argument(0).unwrap();
                let op = unary_minus(operand, i32_type, location).unwrap();
                assert_eq!(op.operand_value(0).unwrap(), operand);
                assert_eq!(op.operand_value(0).unwrap(), operand);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("unary_minus", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @unary_minus(%arg0: i32) -> i32 {
                    %0 = unary_minus %arg0 : (i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_unary_plus() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type = context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let operand = block.argument(0).unwrap();
                let op = unary_plus(operand, i32_type, location).unwrap();
                assert_eq!(op.operand_value(0).unwrap(), operand);
                assert_eq!(op.operand_value(0).unwrap(), operand);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("unary_plus", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @unary_plus(%arg0: i32) -> i32 {
                    %0 = unary_plus %arg0 : (i32) -> i32
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_variable() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let lvalue_i32_type = context.emit_c_lvalue_type(i32_type).unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = variable(context.integer_attribute(i32_type, 0), lvalue_i32_type, location).unwrap();
                assert_eq!(op.value().unwrap().to_string(), "0 : i32");
                assert_eq!(op.value().unwrap().to_string(), "0 : i32");
                block.append_operation(op).unwrap();
                block.append_operation(r#return(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap()).unwrap();
                func("variable", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @variable() {
                    %0 = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_global() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let initial_value = context.integer_attribute(i32_type, 7).as_ref();
        let op = global("counter", i32_type, Some(initial_value), false, true, true, location).unwrap();
        assert_eq!(op.symbol_name().unwrap().as_str().unwrap(), "counter");
        assert_eq!(op.r#type().unwrap(), i32_type.as_ref());
        assert_eq!(op.initial_value(), Some(initial_value));
        assert!(!op.extern_specifier());
        assert!(op.static_specifier());
        assert!(op.const_specifier());
        module.body().unwrap().append_operation(op).unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.global static const @counter : i32 = 7
                }
            "#},
        );
    }

    #[test]
    fn test_get_global() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let lvalue_i32_type = context.emit_c_lvalue_type(i32_type).unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation(global("counter", i32_type, None, true, false, false, location).unwrap())
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = get_global("counter", lvalue_i32_type, location).unwrap();
                assert_eq!(GetGlobalOperation::name(&op).unwrap().reference().as_str().unwrap(), "counter");
                assert_eq!(GetGlobalOperation::name(&op).unwrap().reference().as_str().unwrap(), "counter");
                let op = block.append_operation(op).unwrap();
                let load_op = block
                    .append_operation(load(op.as_ref().result(0).unwrap().as_ref(), i32_type, location).unwrap())
                    .unwrap();
                block
                    .append_operation(r#return(Some(load_op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("get_global", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.global extern @counter : i32
                  emitc.func @get_global() -> i32 {
                    %0 = get_global @counter : !emitc.lvalue<i32>
                    %1 = load %0 : <i32>
                    return %1 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_verbatim() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type = context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref()], &[]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let argument = block.argument(0).unwrap();
                let op = verbatim("value = {}", &[argument.as_ref()], location).unwrap();
                assert_eq!(op.value().unwrap().as_str().unwrap(), "value = {}");
                assert_eq!(op.format_arguments().unwrap(), vec![argument.as_ref()]);
                block.append_operation(op).unwrap();
                block.append_operation(r#return(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap()).unwrap();
                func("verbatim", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @verbatim(%arg0: i32) {
                    verbatim "value = {}" args %arg0 : i32
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_assign() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let lvalue_i32_type = context.emit_c_lvalue_type(i32_type).unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref()], &[]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let value = block.argument(0).unwrap();
                let variable_op = block
                    .append_operation(
                        variable(context.integer_attribute(i32_type, 0), lvalue_i32_type, location).unwrap(),
                    )
                    .unwrap();
                let variable_value = variable_op.as_ref().result(0).unwrap().as_ref();
                let op = assign(variable_value, value, location).unwrap();
                assert_eq!(op.variable().unwrap(), variable_value);
                assert_eq!(op.value().unwrap(), value);
                block.append_operation(op).unwrap();
                block.append_operation(r#return(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap()).unwrap();
                func("assign", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @assign(%arg0: i32) {
                    %0 = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
                    assign %arg0 : i32 to %0 : <i32>
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_yield() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref(), i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let mut region = context.region();
                let mut body = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let body_lhs = body.argument(0).unwrap();
                let body_rhs = body.argument(1).unwrap();
                let add_op = body.append_operation(add(body_lhs, body_rhs, i32_type, location).unwrap()).unwrap();
                let op = r#yield(Some(add_op.as_ref().result(0).unwrap().as_ref()), location).unwrap();
                assert_eq!(op.value().unwrap(), Some(add_op.as_ref().result(0).unwrap().as_ref()));
                body.append_operation(op).unwrap();
                region.append_block(body).unwrap();
                let expression_op = block
                    .append_operation(
                        expression(&[lhs.as_ref(), rhs.as_ref()], i32_type.as_ref(), region.into(), true, location)
                            .unwrap(),
                    )
                    .unwrap();
                block
                    .append_operation(
                        r#return(Some(expression_op.as_ref().result(0).unwrap().as_ref()), location).unwrap(),
                    )
                    .unwrap();
                func("yield", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @yield(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = expression %arg0, %arg1 : (i32, i32) -> i32 {
                      %1 = add %arg0, %arg1 : (i32, i32) -> i32
                      yield %1 : i32
                    }
                    return %0 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_if() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let function_type = context.function_type::<TypeRef, TypeRef>(&[i1_type.as_ref()], &[]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i1_type.as_ref(), location)]);
                let condition = block.argument(0).unwrap();
                let mut then_region = context.block_with_no_arguments();
                then_region
                    .append_operation(r#yield(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap())
                    .unwrap();
                let mut else_region = context.block_with_no_arguments();
                else_region
                    .append_operation(r#yield(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap())
                    .unwrap();
                let op = r#if(condition, then_region.try_into().unwrap(), else_region.try_into().unwrap(), location)
                    .unwrap();
                assert_eq!(op.condition().unwrap(), condition);
                assert_eq!(op.then_region().unwrap().blocks().count(), 1);
                assert_eq!(op.else_region().unwrap().blocks().count(), 1);
                block.append_operation(op).unwrap();
                block.append_operation(r#return(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap()).unwrap();
                func("if", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @if(%arg0: i1) {
                    if %arg0 {
                    } else {
                    }
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_subscript() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let array_i32_type = context.emit_c_array_type(i32_type, &[4]).unwrap();
        let lvalue_i32_type = context.emit_c_lvalue_type(i32_type).unwrap();
        let function_type = context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref()], &[i32_type.as_ref()]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let index = block.argument(0).unwrap();
                let array_op = block.append_operation(literal("values", array_i32_type, location).unwrap()).unwrap();
                let array_value = array_op.as_ref().result(0).unwrap().as_ref();
                let op = subscript(array_value, &[index.as_ref()], lvalue_i32_type, location).unwrap();
                assert_eq!(op.value().unwrap(), array_value);
                assert_eq!(op.indices().unwrap(), vec![index.as_ref()]);
                assert_eq!(op.output().unwrap().r#type().unwrap(), lvalue_i32_type.as_ref());
                let op = block.append_operation(op).unwrap();
                let load_op = block
                    .append_operation(load(op.as_ref().result(0).unwrap().as_ref(), i32_type, location).unwrap())
                    .unwrap();
                block
                    .append_operation(r#return(Some(load_op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("subscript", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @subscript(%arg0: i32) -> i32 {
                    %0 = literal "values" : !emitc.array<4xi32>
                    %1 = subscript %0[%arg0] : (!emitc.array<4xi32>, i32) -> !emitc.lvalue<i32>
                    %2 = load %1 : <i32>
                    return %2 : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_switch() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type = context.function_type::<TypeRef, TypeRef>(&[i32_type.as_ref()], &[]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let argument = block.argument(0).unwrap();
                let mut default_region = context.block_with_no_arguments();
                default_region
                    .append_operation(r#yield(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap())
                    .unwrap();
                let mut case_region = context.block_with_no_arguments();
                case_region
                    .append_operation(r#yield(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap())
                    .unwrap();
                let op = switch(
                    argument,
                    &[1],
                    default_region.try_into().unwrap(),
                    vec![case_region.try_into().unwrap()],
                    location,
                )
                .unwrap();
                assert_eq!(op.argument().unwrap(), argument);
                assert_eq!(op.cases().unwrap().values().collect::<Vec<_>>(), vec![1]);
                assert_eq!(op.default_region().unwrap().blocks().count(), 1);
                assert_eq!(op.case_regions().unwrap().len(), 1);
                block.append_operation(op).unwrap();
                block.append_operation(r#return(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap()).unwrap();
                func("switch", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            // The MLIR printer currently emits a trailing space after the `emitc.switch` argument type.
            concat!(
                "module {\n",
                "  emitc.func @switch(%arg0: i32) {\n",
                "    switch %arg0 : i32 \n",
                "    case 1 {\n",
                "      yield\n",
                "    }\n",
                "    default {\n",
                "    }\n",
                "    return\n",
                "  }\n",
                "}\n",
            ),
        );
    }

    #[test]
    fn test_class() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let mut body = context.block_with_no_arguments();
        body.append_operation(field("value", context.signless_integer_type(32), None, location).unwrap())
            .unwrap();
        let op = class("Box", true, body.try_into().unwrap(), location).unwrap();
        assert_eq!(op.symbol_name().unwrap().as_str().unwrap(), "Box");
        assert!(op.final_specifier());
        assert_eq!(op.body().unwrap().blocks().count(), 1);
        module.body().unwrap().append_operation(op).unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.class final @Box {
                    emitc.field @value : i32
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_field() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let initial_value = context.integer_attribute(i32_type, 7).as_ref();
        let mut body = context.block_with_no_arguments();
        let op = field("value", i32_type, Some(initial_value), location).unwrap();
        assert_eq!(op.symbol_name().unwrap().as_str().unwrap(), "value");
        assert_eq!(op.r#type().unwrap(), i32_type.as_ref());
        assert_eq!(op.initial_value(), Some(initial_value));
        body.append_operation(op).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(class("Box", false, body.try_into().unwrap(), location).unwrap())
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.class @Box {
                    emitc.field @value : i32 = 7
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_get_field() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[i32_type.as_ref()]);
        let mut class_body = context.block_with_no_arguments();
        class_body.append_operation(field("value", i32_type, None, location).unwrap()).unwrap();
        class_body
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = get_field("value", i32_type, location).unwrap();
                assert_eq!(op.field_name().unwrap().reference().as_str().unwrap(), "value");
                assert_eq!(op.field_name().unwrap().reference().as_str().unwrap(), "value");
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(r#return(Some(op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                func("get_field", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation(class("Box", false, class_body.try_into().unwrap(), location).unwrap())
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.class @Box {
                    emitc.field @value : i32
                    emitc.func @get_field() -> i32 {
                      %0 = get_field @value : i32
                      return %0 : i32
                    }
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_do() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let function_type = context.function_type::<TypeRef, TypeRef>(&[], &[]);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let mut body = context.block_with_no_arguments();
                body.append_operation(verbatim("side_effect()", &[], location).unwrap()).unwrap();

                let mut condition = context.block_with_no_arguments();
                let mut expression_region = context.region();
                let mut expression_body = context.block_with_no_arguments();
                let literal_op = expression_body.append_operation(literal("true", i1_type, location).unwrap()).unwrap();
                expression_body
                    .append_operation(r#yield(Some(literal_op.as_ref().result(0).unwrap().as_ref()), location).unwrap())
                    .unwrap();
                expression_region.append_block(expression_body).unwrap();
                let expression_op = condition
                    .append_operation(
                        expression(&[], i1_type.as_ref(), expression_region.into(), false, location).unwrap(),
                    )
                    .unwrap();
                condition
                    .append_operation(
                        r#yield(Some(expression_op.as_ref().result(0).unwrap().as_ref()), location).unwrap(),
                    )
                    .unwrap();

                let op = r#do(body.try_into().unwrap(), condition.try_into().unwrap(), location).unwrap();
                assert_eq!(op.body().unwrap().blocks().count(), 1);
                assert_eq!(op.condition().unwrap().blocks().count(), 1);
                block.append_operation(op).unwrap();
                block.append_operation(r#return(Option::<ValueRef<'_, '_, '_>>::None, location).unwrap()).unwrap();
                func("do", function_type, block.try_into().unwrap(), None, None, None, location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  emitc.func @do() {
                    do {
                      verbatim "side_effect()"
                    } while {
                      %0 = expression  : () -> i1 {
                        %1 = literal "true" : i1
                        yield %1 : i1
                      }
                      yield %0 : i1
                    }
                    return
                  }
                }
            "#},
        );
    }
}
