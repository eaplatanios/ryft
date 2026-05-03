use crate::{
    AttributeRef, DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Type, TypeRef, Value, ValueRef,
    mlir_op,
};

/// Canonical MLIR operation name for [`AcosOperation`].
pub const ACOS_OPERATION_NAME: &str = "llvm.intr.acos";

/// Operation trait for `llvm.intr.acos`.
pub trait AcosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ACOS_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Acos);

/// Constructs a new detached `llvm.intr.acos` operation.
pub fn intr_acos<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedAcosOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ACOS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_acos`")
}

/// Canonical MLIR operation name for [`AsinOperation`].
pub const ASIN_OPERATION_NAME: &str = "llvm.intr.asin";

/// Operation trait for `llvm.intr.asin`.
pub trait AsinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ASIN_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Asin);

/// Constructs a new detached `llvm.intr.asin` operation.
pub fn intr_asin<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedAsinOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ASIN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_asin`")
}

/// Canonical MLIR operation name for [`Atan2Operation`].
pub const ATAN2_OPERATION_NAME: &str = "llvm.intr.atan2";

/// Operation trait for `llvm.intr.atan2`.
pub trait Atan2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ATAN2_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Atan2);

/// Constructs a new detached `llvm.intr.atan2` operation.
pub fn intr_atan2<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedAtan2Operation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ATAN2_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_atan2`")
}

/// Canonical MLIR operation name for [`AtanOperation`].
pub const ATAN_OPERATION_NAME: &str = "llvm.intr.atan";

/// Operation trait for `llvm.intr.atan`.
pub trait AtanOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ATAN_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Atan);

/// Constructs a new detached `llvm.intr.atan` operation.
pub fn intr_atan<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedAtanOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ATAN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_atan`")
}

/// Canonical MLIR operation name for [`AbsOperation`].
pub const ABS_OPERATION_NAME: &str = "llvm.intr.abs";

/// Operation trait for `llvm.intr.abs`.
pub trait AbsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ABS_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `is_int_min_poison` attribute.
    fn is_int_min_poison(&self) -> AttributeRef<'c, 't> {
        self.attribute("is_int_min_poison").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Abs);

/// Constructs a new detached `llvm.intr.abs` operation.
pub fn intr_abs<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    is_int_min_poison: AttributeRef<'c, 't>,
    location: L,
) -> DetachedAbsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ABS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("is_int_min_poison", is_int_min_poison);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_abs`")
}

/// Canonical MLIR operation name for [`AssumeOperation`].
pub const ASSUME_OPERATION_NAME: &str = "llvm.intr.assume";

/// Operation trait for `llvm.intr.assume`.
pub trait AssumeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ASSUME_OPERATION_NAME
    }

    /// Returns the `condition` operand.
    fn condition(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(Assume);

/// Constructs a new detached `llvm.intr.assume` operation.
pub fn intr_assume<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    condition: V0,
    location: L,
) -> DetachedAssumeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ASSUME_OPERATION_NAME, location);
    builder = builder.add_operand(condition);
    builder = builder.add_attribute("op_bundle_sizes", context.dense_i32_array_attribute(&[]).unwrap());
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_assume`")
}

/// Canonical MLIR operation name for [`BitReverseOperation`].
pub const BIT_REVERSE_OPERATION_NAME: &str = "llvm.intr.bitreverse";

/// Operation trait for `llvm.intr.bitreverse`.
pub trait BitReverseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        BIT_REVERSE_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(BitReverse);

/// Constructs a new detached `llvm.intr.bitreverse` operation.
pub fn intr_bit_reverse<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> DetachedBitReverseOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(BIT_REVERSE_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_bit_reverse`")
}

/// Canonical MLIR operation name for [`ByteSwapOperation`].
pub const BYTE_SWAP_OPERATION_NAME: &str = "llvm.intr.bswap";

/// Operation trait for `llvm.intr.bswap`.
pub trait ByteSwapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        BYTE_SWAP_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ByteSwap);

/// Constructs a new detached `llvm.intr.bswap` operation.
pub fn intr_bswap<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> DetachedByteSwapOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(BYTE_SWAP_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_bswap`")
}

/// Canonical MLIR operation name for [`CopySignOperation`].
pub const COPY_SIGN_OPERATION_NAME: &str = "llvm.intr.copysign";

/// Operation trait for `llvm.intr.copysign`.
pub trait CopySignOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        COPY_SIGN_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(CopySign);

/// Constructs a new detached `llvm.intr.copysign` operation.
pub fn intr_copy_sign<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedCopySignOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(COPY_SIGN_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_copy_sign`")
}

/// Canonical MLIR operation name for [`CosOperation`].
pub const COS_OPERATION_NAME: &str = "llvm.intr.cos";

/// Operation trait for `llvm.intr.cos`.
pub trait CosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        COS_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Cos);

/// Constructs a new detached `llvm.intr.cos` operation.
pub fn intr_cos<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedCosOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(COS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_cos`")
}

/// Canonical MLIR operation name for [`CoshOperation`].
pub const COSH_OPERATION_NAME: &str = "llvm.intr.cosh";

/// Operation trait for `llvm.intr.cosh`.
pub trait CoshOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        COSH_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Cosh);

/// Constructs a new detached `llvm.intr.cosh` operation.
pub fn intr_cosh<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedCoshOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(COSH_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_cosh`")
}

/// Canonical MLIR operation name for [`CountLeadingZerosOperation`].
pub const COUNT_LEADING_ZEROS_OPERATION_NAME: &str = "llvm.intr.ctlz";

/// Operation trait for `llvm.intr.ctlz`.
pub trait CountLeadingZerosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        COUNT_LEADING_ZEROS_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `is_zero_poison` attribute.
    fn is_zero_poison(&self) -> AttributeRef<'c, 't> {
        self.attribute("is_zero_poison").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(CountLeadingZeros);

/// Constructs a new detached `llvm.intr.ctlz` operation.
pub fn intr_ctlz<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    is_zero_poison: AttributeRef<'c, 't>,
    location: L,
) -> DetachedCountLeadingZerosOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(COUNT_LEADING_ZEROS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("is_zero_poison", is_zero_poison);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ctlz`")
}

/// Canonical MLIR operation name for [`CountTrailingZerosOperation`].
pub const COUNT_TRAILING_ZEROS_OPERATION_NAME: &str = "llvm.intr.cttz";

/// Operation trait for `llvm.intr.cttz`.
pub trait CountTrailingZerosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        COUNT_TRAILING_ZEROS_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `is_zero_poison` attribute.
    fn is_zero_poison(&self) -> AttributeRef<'c, 't> {
        self.attribute("is_zero_poison").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(CountTrailingZeros);

/// Constructs a new detached `llvm.intr.cttz` operation.
pub fn intr_count_trailing_zeros<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    is_zero_poison: AttributeRef<'c, 't>,
    location: L,
) -> DetachedCountTrailingZerosOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(COUNT_TRAILING_ZEROS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("is_zero_poison", is_zero_poison);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_count_trailing_zeros`")
}

/// Canonical MLIR operation name for [`CtPopOperation`].
pub const CT_POP_OPERATION_NAME: &str = "llvm.intr.ctpop";

/// Operation trait for `llvm.intr.ctpop`.
pub trait CtPopOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CT_POP_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(CtPop);

/// Constructs a new detached `llvm.intr.ctpop` operation.
pub fn intr_ct_pop<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> DetachedCtPopOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CT_POP_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ct_pop`")
}

/// Canonical MLIR operation name for [`Exp10Operation`].
pub const EXP10_OPERATION_NAME: &str = "llvm.intr.exp10";

/// Operation trait for `llvm.intr.exp10`.
pub trait Exp10Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EXP10_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Exp10);

/// Constructs a new detached `llvm.intr.exp10` operation.
pub fn intr_exp10<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedExp10Operation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(EXP10_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_exp10`")
}

/// Canonical MLIR operation name for [`Exp2Operation`].
pub const EXP2_OPERATION_NAME: &str = "llvm.intr.exp2";

/// Operation trait for `llvm.intr.exp2`.
pub trait Exp2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EXP2_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Exp2);

/// Constructs a new detached `llvm.intr.exp2` operation.
pub fn intr_exp2<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedExp2Operation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(EXP2_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_exp2`")
}

/// Canonical MLIR operation name for [`ExpOperation`].
pub const EXP_OPERATION_NAME: &str = "llvm.intr.exp";

/// Operation trait for `llvm.intr.exp`.
pub trait ExpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EXP_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Exp);

/// Constructs a new detached `llvm.intr.exp` operation.
pub fn intr_exp<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedExpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(EXP_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_exp`")
}

/// Canonical MLIR operation name for [`ExpectOperation`].
pub const EXPECT_OPERATION_NAME: &str = "llvm.intr.expect";

/// Operation trait for `llvm.intr.expect`.
pub trait ExpectOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EXPECT_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `expected` operand.
    fn expected(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Expect);

/// Constructs a new detached `llvm.intr.expect` operation.
pub fn intr_expect<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    expected: V1,
    result_type: T0,
    location: L,
) -> DetachedExpectOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(EXPECT_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(expected);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_expect`")
}

/// Canonical MLIR operation name for [`ExpectWithProbabilityOperation`].
pub const EXPECT_WITH_PROBABILITY_OPERATION_NAME: &str = "llvm.intr.expect.with.probability";

/// Operation trait for `llvm.intr.expect.with.probability`.
pub trait ExpectWithProbabilityOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EXPECT_WITH_PROBABILITY_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `expected` operand.
    fn expected(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `prob` attribute.
    fn prob(&self) -> AttributeRef<'c, 't> {
        self.attribute("prob").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ExpectWithProbability);

/// Constructs a new detached `llvm.intr.expect.with.probability` operation.
pub fn intr_expect_with_probability<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    expected: V1,
    result_type: T0,
    prob: AttributeRef<'c, 't>,
    location: L,
) -> DetachedExpectWithProbabilityOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(EXPECT_WITH_PROBABILITY_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(expected);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("prob", prob);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_expect_with_probability`")
}

/// Canonical MLIR operation name for [`FabsOperation`].
pub const FABS_OPERATION_NAME: &str = "llvm.intr.fabs";

/// Operation trait for `llvm.intr.fabs`.
pub trait FabsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FABS_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Fabs);

/// Constructs a new detached `llvm.intr.fabs` operation.
pub fn intr_fabs<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedFabsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FABS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_fabs`")
}

/// Canonical MLIR operation name for [`FceilOperation`].
pub const FCEIL_OPERATION_NAME: &str = "llvm.intr.ceil";

/// Operation trait for `llvm.intr.ceil`.
pub trait FceilOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FCEIL_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Fceil);

/// Constructs a new detached `llvm.intr.ceil` operation.
pub fn intr_ceil<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedFceilOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FCEIL_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ceil`")
}

/// Canonical MLIR operation name for [`FfloorOperation`].
pub const FFLOOR_OPERATION_NAME: &str = "llvm.intr.floor";

/// Operation trait for `llvm.intr.floor`.
pub trait FfloorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FFLOOR_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Ffloor);

/// Constructs a new detached `llvm.intr.floor` operation.
pub fn intr_floor<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedFfloorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FFLOOR_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_floor`")
}

/// Canonical MLIR operation name for [`FmaOperation`].
pub const FMA_OPERATION_NAME: &str = "llvm.intr.fma";

/// Operation trait for `llvm.intr.fma`.
pub trait FmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FMA_OPERATION_NAME
    }

    /// Returns the `first` operand.
    fn first(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `second` operand.
    fn second(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `third` operand.
    fn third(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Fma);

/// Constructs a new detached `llvm.intr.fma` operation.
pub fn intr_fma<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    first: V0,
    second: V1,
    third: V2,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedFmaOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FMA_OPERATION_NAME, location);
    builder = builder.add_operand(first);
    builder = builder.add_operand(second);
    builder = builder.add_operand(third);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_fma`")
}

/// Canonical MLIR operation name for [`FmulAddOperation`].
pub const FMUL_ADD_OPERATION_NAME: &str = "llvm.intr.fmuladd";

/// Operation trait for `llvm.intr.fmuladd`.
pub trait FmulAddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FMUL_ADD_OPERATION_NAME
    }

    /// Returns the `first` operand.
    fn first(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `second` operand.
    fn second(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `third` operand.
    fn third(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(FmulAdd);

/// Constructs a new detached `llvm.intr.fmuladd` operation.
pub fn intr_fmuladd<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    first: V0,
    second: V1,
    third: V2,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedFmulAddOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FMUL_ADD_OPERATION_NAME, location);
    builder = builder.add_operand(first);
    builder = builder.add_operand(second);
    builder = builder.add_operand(third);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_fmuladd`")
}

/// Canonical MLIR operation name for [`FtruncOperation`].
pub const FTRUNC_OPERATION_NAME: &str = "llvm.intr.trunc";

/// Operation trait for `llvm.intr.trunc`.
pub trait FtruncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FTRUNC_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Ftrunc);

/// Constructs a new detached `llvm.intr.trunc` operation.
pub fn intr_trunc<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedFtruncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FTRUNC_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_trunc`")
}

/// Canonical MLIR operation name for [`FractionExpOperation`].
pub const FRACTION_EXP_OPERATION_NAME: &str = "llvm.intr.frexp";

/// Operation trait for `llvm.intr.frexp`.
pub trait FractionExpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FRACTION_EXP_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(FractionExp);

/// Constructs a new detached `llvm.intr.frexp` operation.
pub fn intr_frexp<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedFractionExpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FRACTION_EXP_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_frexp`")
}

/// Canonical MLIR operation name for [`FshlOperation`].
pub const FSHL_OPERATION_NAME: &str = "llvm.intr.fshl";

/// Operation trait for `llvm.intr.fshl`.
pub trait FshlOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FSHL_OPERATION_NAME
    }

    /// Returns the `first` operand.
    fn first(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `second` operand.
    fn second(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `third` operand.
    fn third(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Fshl);

/// Constructs a new detached `llvm.intr.fshl` operation.
pub fn intr_fshl<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    first: V0,
    second: V1,
    third: V2,
    result_type: T0,
    location: L,
) -> DetachedFshlOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FSHL_OPERATION_NAME, location);
    builder = builder.add_operand(first);
    builder = builder.add_operand(second);
    builder = builder.add_operand(third);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_fshl`")
}

/// Canonical MLIR operation name for [`FshrOperation`].
pub const FSHR_OPERATION_NAME: &str = "llvm.intr.fshr";

/// Operation trait for `llvm.intr.fshr`.
pub trait FshrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FSHR_OPERATION_NAME
    }

    /// Returns the `first` operand.
    fn first(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `second` operand.
    fn second(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `third` operand.
    fn third(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Fshr);

/// Constructs a new detached `llvm.intr.fshr` operation.
pub fn intr_fshr<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    first: V0,
    second: V1,
    third: V2,
    result_type: T0,
    location: L,
) -> DetachedFshrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FSHR_OPERATION_NAME, location);
    builder = builder.add_operand(first);
    builder = builder.add_operand(second);
    builder = builder.add_operand(third);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_fshr`")
}

/// Canonical MLIR operation name for [`IsConstantOperation`].
pub const IS_CONSTANT_OPERATION_NAME: &str = "llvm.intr.is.constant";

/// Operation trait for `llvm.intr.is.constant`.
pub trait IsConstantOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        IS_CONSTANT_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(IsConstant);

/// Constructs a new detached `llvm.intr.is.constant` operation.
pub fn intr_is_constant<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    location: L,
) -> DetachedIsConstantOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(IS_CONSTANT_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_is_constant`")
}

/// Canonical MLIR operation name for [`IsFpclassOperation`].
pub const IS_FPCLASS_OPERATION_NAME: &str = "llvm.intr.is.fpclass";

/// Operation trait for `llvm.intr.is.fpclass`.
pub trait IsFpclassOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        IS_FPCLASS_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `bit` attribute.
    fn bit(&self) -> AttributeRef<'c, 't> {
        self.attribute("bit").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(IsFpclass);

/// Constructs a new detached `llvm.intr.is.fpclass` operation.
pub fn intr_is_fpclass<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    bit: AttributeRef<'c, 't>,
    location: L,
) -> DetachedIsFpclassOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(IS_FPCLASS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("bit", bit);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_is_fpclass`")
}

/// Canonical MLIR operation name for [`LlrintOperation`].
pub const LLRINT_OPERATION_NAME: &str = "llvm.intr.llrint";

/// Operation trait for `llvm.intr.llrint`.
pub trait LlrintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LLRINT_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Llrint);

/// Constructs a new detached `llvm.intr.llrint` operation.
pub fn intr_llrint<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    location: L,
) -> DetachedLlrintOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LLRINT_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_llrint`")
}

/// Canonical MLIR operation name for [`LlroundOperation`].
pub const LLROUND_OPERATION_NAME: &str = "llvm.intr.llround";

/// Operation trait for `llvm.intr.llround`.
pub trait LlroundOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LLROUND_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Llround);

/// Constructs a new detached `llvm.intr.llround` operation.
pub fn intr_llround<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    location: L,
) -> DetachedLlroundOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LLROUND_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_llround`")
}

/// Canonical MLIR operation name for [`LoadExpOperation`].
pub const LOAD_EXP_OPERATION_NAME: &str = "llvm.intr.ldexp";

/// Operation trait for `llvm.intr.ldexp`.
pub trait LoadExpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LOAD_EXP_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `power` operand.
    fn power(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(LoadExp);

/// Constructs a new detached `llvm.intr.ldexp` operation.
pub fn intr_ldexp<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    power: V1,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedLoadExpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LOAD_EXP_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(power);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ldexp`")
}

/// Canonical MLIR operation name for [`Log10Operation`].
pub const LOG10_OPERATION_NAME: &str = "llvm.intr.log10";

/// Operation trait for `llvm.intr.log10`.
pub trait Log10Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LOG10_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Log10);

/// Constructs a new detached `llvm.intr.log10` operation.
pub fn intr_log10<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedLog10Operation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LOG10_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_log10`")
}

/// Canonical MLIR operation name for [`Log2Operation`].
pub const LOG2_OPERATION_NAME: &str = "llvm.intr.log2";

/// Operation trait for `llvm.intr.log2`.
pub trait Log2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LOG2_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Log2);

/// Constructs a new detached `llvm.intr.log2` operation.
pub fn intr_log2<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedLog2Operation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LOG2_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_log2`")
}

/// Canonical MLIR operation name for [`LogOperation`].
pub const LOG_OPERATION_NAME: &str = "llvm.intr.log";

/// Operation trait for `llvm.intr.log`.
pub trait LogOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LOG_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Log);

/// Constructs a new detached `llvm.intr.log` operation.
pub fn intr_log<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedLogOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LOG_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_log`")
}

/// Canonical MLIR operation name for [`LrintOperation`].
pub const LRINT_OPERATION_NAME: &str = "llvm.intr.lrint";

/// Operation trait for `llvm.intr.lrint`.
pub trait LrintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LRINT_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Lrint);

/// Constructs a new detached `llvm.intr.lrint` operation.
pub fn intr_lrint<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    location: L,
) -> DetachedLrintOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LRINT_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_lrint`")
}

/// Canonical MLIR operation name for [`LroundOperation`].
pub const LROUND_OPERATION_NAME: &str = "llvm.intr.lround";

/// Operation trait for `llvm.intr.lround`.
pub trait LroundOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LROUND_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Lround);

/// Constructs a new detached `llvm.intr.lround` operation.
pub fn intr_lround<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    location: L,
) -> DetachedLroundOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LROUND_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_lround`")
}

/// Canonical MLIR operation name for [`MaxNumOperation`].
pub const MAX_NUM_OPERATION_NAME: &str = "llvm.intr.maxnum";

/// Operation trait for `llvm.intr.maxnum`.
pub trait MaxNumOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MAX_NUM_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(MaxNum);

/// Constructs a new detached `llvm.intr.maxnum` operation.
pub fn intr_maxnum<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedMaxNumOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MAX_NUM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_maxnum`")
}

/// Canonical MLIR operation name for [`MaximumOperation`].
pub const MAXIMUM_OPERATION_NAME: &str = "llvm.intr.maximum";

/// Operation trait for `llvm.intr.maximum`.
pub trait MaximumOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MAXIMUM_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Maximum);

/// Constructs a new detached `llvm.intr.maximum` operation.
pub fn intr_maximum<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedMaximumOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MAXIMUM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_maximum`")
}

/// Canonical MLIR operation name for [`MinNumOperation`].
pub const MIN_NUM_OPERATION_NAME: &str = "llvm.intr.minnum";

/// Operation trait for `llvm.intr.minnum`.
pub trait MinNumOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MIN_NUM_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(MinNum);

/// Constructs a new detached `llvm.intr.minnum` operation.
pub fn intr_min_num<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedMinNumOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MIN_NUM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_min_num`")
}

/// Canonical MLIR operation name for [`MinimumOperation`].
pub const MINIMUM_OPERATION_NAME: &str = "llvm.intr.minimum";

/// Operation trait for `llvm.intr.minimum`.
pub trait MinimumOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MINIMUM_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Minimum);

/// Constructs a new detached `llvm.intr.minimum` operation.
pub fn intr_minimum<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedMinimumOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MINIMUM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_minimum`")
}

/// Canonical MLIR operation name for [`NearbyintOperation`].
pub const NEARBY_INT_OPERATION_NAME: &str = "llvm.intr.nearbyint";

/// Operation trait for `llvm.intr.nearbyint`.
pub trait NearbyIntOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        NEARBY_INT_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(NearbyInt);

/// Constructs a new detached `llvm.intr.nearbyint` operation.
pub fn intr_nearby_int<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedNearbyIntOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(NEARBY_INT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_nearby_int`")
}

/// Canonical MLIR operation name for [`PowIOperation`].
pub const POW_I_OPERATION_NAME: &str = "llvm.intr.powi";

/// Operation trait for `llvm.intr.powi`.
pub trait PowIOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        POW_I_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `power` operand.
    fn power(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(PowI);

/// Constructs a new detached `llvm.intr.powi` operation.
pub fn intr_powi<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    power: V1,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedPowIOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(POW_I_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(power);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_powi`")
}

/// Canonical MLIR operation name for [`PowOperation`].
pub const POW_OPERATION_NAME: &str = "llvm.intr.pow";

/// Operation trait for `llvm.intr.pow`.
pub trait PowOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        POW_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Pow);

/// Constructs a new detached `llvm.intr.pow` operation.
pub fn intr_pow<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedPowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(POW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_pow`")
}

/// Canonical MLIR operation name for [`RintOperation`].
pub const RINT_OPERATION_NAME: &str = "llvm.intr.rint";

/// Operation trait for `llvm.intr.rint`.
pub trait RintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        RINT_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Rint);

/// Constructs a new detached `llvm.intr.rint` operation.
pub fn intr_rint<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedRintOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(RINT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_rint`")
}

/// Canonical MLIR operation name for [`RoundEvenOperation`].
pub const ROUND_EVEN_OPERATION_NAME: &str = "llvm.intr.roundeven";

/// Operation trait for `llvm.intr.roundeven`.
pub trait RoundEvenOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ROUND_EVEN_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(RoundEven);

/// Constructs a new detached `llvm.intr.roundeven` operation.
pub fn intr_round_even<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedRoundEvenOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ROUND_EVEN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_round_even`")
}

/// Canonical MLIR operation name for [`RoundOperation`].
pub const ROUND_OPERATION_NAME: &str = "llvm.intr.round";

/// Operation trait for `llvm.intr.round`.
pub trait RoundOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ROUND_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Round);

/// Constructs a new detached `llvm.intr.round` operation.
pub fn intr_round<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedRoundOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ROUND_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_round`")
}

/// Canonical MLIR operation name for [`SaddSatOperation`].
pub const SADD_SAT_OPERATION_NAME: &str = "llvm.intr.sadd.sat";

/// Operation trait for `llvm.intr.sadd.sat`.
pub trait SaddSatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SADD_SAT_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(SaddSat);

/// Constructs a new detached `llvm.intr.sadd.sat` operation.
pub fn intr_sadd_sat<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedSaddSatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SADD_SAT_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_sadd_sat`")
}

/// Canonical MLIR operation name for [`SaddWithOverflowOperation`].
pub const SADD_WITH_OVERFLOW_OPERATION_NAME: &str = "llvm.intr.sadd.with.overflow";

/// Operation trait for `llvm.intr.sadd.with.overflow`.
pub trait SaddWithOverflowOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SADD_WITH_OVERFLOW_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(SaddWithOverflow);

/// Constructs a new detached `llvm.intr.sadd.with.overflow` operation.
pub fn intr_sadd_with_overflow<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedSaddWithOverflowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SADD_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_sadd_with_overflow`")
}

/// Canonical MLIR operation name for [`ScmpOperation`].
pub const SCMP_OPERATION_NAME: &str = "llvm.intr.scmp";

/// Operation trait for `llvm.intr.scmp`.
pub trait ScmpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SCMP_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Scmp);

/// Constructs a new detached `llvm.intr.scmp` operation.
pub fn intr_scmp<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedScmpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SCMP_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_scmp`")
}

/// Canonical MLIR operation name for [`SmaxOperation`].
pub const SMAX_OPERATION_NAME: &str = "llvm.intr.smax";

/// Operation trait for `llvm.intr.smax`.
pub trait SmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SMAX_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Smax);

/// Constructs a new detached `llvm.intr.smax` operation.
pub fn intr_smax<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedSmaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SMAX_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_smax`")
}

/// Canonical MLIR operation name for [`SminOperation`].
pub const SMIN_OPERATION_NAME: &str = "llvm.intr.smin";

/// Operation trait for `llvm.intr.smin`.
pub trait SminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SMIN_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Smin);

/// Constructs a new detached `llvm.intr.smin` operation.
pub fn intr_smin<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedSminOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SMIN_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_smin`")
}

/// Canonical MLIR operation name for [`SmulWithOverflowOperation`].
pub const SMUL_WITH_OVERFLOW_OPERATION_NAME: &str = "llvm.intr.smul.with.overflow";

/// Operation trait for `llvm.intr.smul.with.overflow`.
pub trait SmulWithOverflowOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SMUL_WITH_OVERFLOW_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(SmulWithOverflow);

/// Constructs a new detached `llvm.intr.smul.with.overflow` operation.
pub fn intr_smul_with_overflow<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedSmulWithOverflowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SMUL_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_smul_with_overflow`")
}

/// Canonical MLIR operation name for [`SsaCopyOperation`].
pub const SSA_COPY_OPERATION_NAME: &str = "llvm.intr.ssa.copy";

/// Operation trait for `llvm.intr.ssa.copy`.
pub trait SsaCopyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SSA_COPY_OPERATION_NAME
    }

    /// Returns the `operand` operand.
    fn operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(SsaCopy);

/// Constructs a new detached `llvm.intr.ssa.copy` operation.
pub fn intr_ssa_copy<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    operand: V0,
    result_type: T0,
    location: L,
) -> DetachedSsaCopyOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SSA_COPY_OPERATION_NAME, location);
    builder = builder.add_operand(operand);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ssa_copy`")
}

/// Canonical MLIR operation name for [`SshlSatOperation`].
pub const SSHL_SAT_OPERATION_NAME: &str = "llvm.intr.sshl.sat";

/// Operation trait for `llvm.intr.sshl.sat`.
pub trait SshlSatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SSHL_SAT_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(SshlSat);

/// Constructs a new detached `llvm.intr.sshl.sat` operation.
pub fn intr_sshl_sat<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedSshlSatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SSHL_SAT_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_sshl_sat`")
}

/// Canonical MLIR operation name for [`SsubSatOperation`].
pub const SSUB_SAT_OPERATION_NAME: &str = "llvm.intr.ssub.sat";

/// Operation trait for `llvm.intr.ssub.sat`.
pub trait SsubSatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SSUB_SAT_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(SsubSat);

/// Constructs a new detached `llvm.intr.ssub.sat` operation.
pub fn intr_ssub_sat<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedSsubSatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SSUB_SAT_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ssub_sat`")
}

/// Canonical MLIR operation name for [`SsubWithOverflowOperation`].
pub const SSUB_WITH_OVERFLOW_OPERATION_NAME: &str = "llvm.intr.ssub.with.overflow";

/// Operation trait for `llvm.intr.ssub.with.overflow`.
pub trait SsubWithOverflowOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SSUB_WITH_OVERFLOW_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(SsubWithOverflow);

/// Constructs a new detached `llvm.intr.ssub.with.overflow` operation.
pub fn intr_ssub_with_overflow<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedSsubWithOverflowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SSUB_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ssub_with_overflow`")
}

/// Canonical MLIR operation name for [`SinOperation`].
pub const SIN_OPERATION_NAME: &str = "llvm.intr.sin";

/// Operation trait for `llvm.intr.sin`.
pub trait SinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SIN_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Sin);

/// Constructs a new detached `llvm.intr.sin` operation.
pub fn intr_sin<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedSinOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SIN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_sin`")
}

/// Canonical MLIR operation name for [`SincosOperation`].
pub const SINCOS_OPERATION_NAME: &str = "llvm.intr.sincos";

/// Operation trait for `llvm.intr.sincos`.
pub trait SincosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SINCOS_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Sincos);

/// Constructs a new detached `llvm.intr.sincos` operation.
pub fn intr_sincos<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedSincosOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SINCOS_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_sincos`")
}

/// Canonical MLIR operation name for [`SinhOperation`].
pub const SINH_OPERATION_NAME: &str = "llvm.intr.sinh";

/// Operation trait for `llvm.intr.sinh`.
pub trait SinhOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SINH_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Sinh);

/// Constructs a new detached `llvm.intr.sinh` operation.
pub fn intr_sinh<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedSinhOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SINH_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_sinh`")
}

/// Canonical MLIR operation name for [`SqrtOperation`].
pub const SQRT_OPERATION_NAME: &str = "llvm.intr.sqrt";

/// Operation trait for `llvm.intr.sqrt`.
pub trait SqrtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SQRT_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Sqrt);

/// Constructs a new detached `llvm.intr.sqrt` operation.
pub fn intr_sqrt<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedSqrtOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SQRT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_sqrt`")
}

/// Canonical MLIR operation name for [`StepVectorOperation`].
pub const STEP_VECTOR_OPERATION_NAME: &str = "llvm.intr.stepvector";

/// Operation trait for `llvm.intr.stepvector`.
pub trait StepVectorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        STEP_VECTOR_OPERATION_NAME
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(StepVector);

/// Constructs a new detached `llvm.intr.stepvector` operation.
pub fn intr_stepvector<'v, 'c: 'v, 't: 'c, T0: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T0,
    location: L,
) -> DetachedStepVectorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(STEP_VECTOR_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_stepvector`")
}

/// Canonical MLIR operation name for [`TanOperation`].
pub const TAN_OPERATION_NAME: &str = "llvm.intr.tan";

/// Operation trait for `llvm.intr.tan`.
pub trait TanOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TAN_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Tan);

/// Constructs a new detached `llvm.intr.tan` operation.
pub fn intr_tan<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedTanOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(TAN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_tan`")
}

/// Canonical MLIR operation name for [`TanhOperation`].
pub const TANH_OPERATION_NAME: &str = "llvm.intr.tanh";

/// Operation trait for `llvm.intr.tanh`.
pub trait TanhOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TANH_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Tanh);

/// Constructs a new detached `llvm.intr.tanh` operation.
pub fn intr_tanh<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedTanhOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(TANH_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_tanh`")
}

/// Canonical MLIR operation name for [`UaddSatOperation`].
pub const UADD_SAT_OPERATION_NAME: &str = "llvm.intr.uadd.sat";

/// Operation trait for `llvm.intr.uadd.sat`.
pub trait UaddSatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        UADD_SAT_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(UaddSat);

/// Constructs a new detached `llvm.intr.uadd.sat` operation.
pub fn intr_uadd_sat<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedUaddSatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(UADD_SAT_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_uadd_sat`")
}

/// Canonical MLIR operation name for [`UaddWithOverflowOperation`].
pub const UADD_WITH_OVERFLOW_OPERATION_NAME: &str = "llvm.intr.uadd.with.overflow";

/// Operation trait for `llvm.intr.uadd.with.overflow`.
pub trait UaddWithOverflowOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        UADD_WITH_OVERFLOW_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(UaddWithOverflow);

/// Constructs a new detached `llvm.intr.uadd.with.overflow` operation.
pub fn intr_uadd_with_overflow<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedUaddWithOverflowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(UADD_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_uadd_with_overflow`")
}

/// Canonical MLIR operation name for [`UcmpOperation`].
pub const UCMP_OPERATION_NAME: &str = "llvm.intr.ucmp";

/// Operation trait for `llvm.intr.ucmp`.
pub trait UcmpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        UCMP_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Ucmp);

/// Constructs a new detached `llvm.intr.ucmp` operation.
pub fn intr_ucmp<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedUcmpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(UCMP_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ucmp`")
}

/// Canonical MLIR operation name for [`UmaxOperation`].
pub const UMAX_OPERATION_NAME: &str = "llvm.intr.umax";

/// Operation trait for `llvm.intr.umax`.
pub trait UmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        UMAX_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Umax);

/// Constructs a new detached `llvm.intr.umax` operation.
pub fn intr_umax<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedUmaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(UMAX_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_umax`")
}

/// Canonical MLIR operation name for [`UminOperation`].
pub const UMIN_OPERATION_NAME: &str = "llvm.intr.umin";

/// Operation trait for `llvm.intr.umin`.
pub trait UminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        UMIN_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Umin);

/// Constructs a new detached `llvm.intr.umin` operation.
pub fn intr_umin<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedUminOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(UMIN_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_umin`")
}

/// Canonical MLIR operation name for [`UmulWithOverflowOperation`].
pub const UMUL_WITH_OVERFLOW_OPERATION_NAME: &str = "llvm.intr.umul.with.overflow";

/// Operation trait for `llvm.intr.umul.with.overflow`.
pub trait UmulWithOverflowOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        UMUL_WITH_OVERFLOW_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(UmulWithOverflow);

/// Constructs a new detached `llvm.intr.umul.with.overflow` operation.
pub fn intr_umul_with_overflow<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedUmulWithOverflowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(UMUL_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_umul_with_overflow`")
}

/// Canonical MLIR operation name for [`UshlSatOperation`].
pub const USHL_SAT_OPERATION_NAME: &str = "llvm.intr.ushl.sat";

/// Operation trait for `llvm.intr.ushl.sat`.
pub trait UshlSatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        USHL_SAT_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(UshlSat);

/// Constructs a new detached `llvm.intr.ushl.sat` operation.
pub fn intr_ushl_sat<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedUshlSatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(USHL_SAT_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ushl_sat`")
}

/// Canonical MLIR operation name for [`UsubSatOperation`].
pub const USUB_SAT_OPERATION_NAME: &str = "llvm.intr.usub.sat";

/// Operation trait for `llvm.intr.usub.sat`.
pub trait UsubSatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        USUB_SAT_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(UsubSat);

/// Constructs a new detached `llvm.intr.usub.sat` operation.
pub fn intr_usub_sat<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedUsubSatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(USUB_SAT_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_usub_sat`")
}

/// Canonical MLIR operation name for [`UsubWithOverflowOperation`].
pub const USUB_WITH_OVERFLOW_OPERATION_NAME: &str = "llvm.intr.usub.with.overflow";

/// Operation trait for `llvm.intr.usub.with.overflow`.
pub trait UsubWithOverflowOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        USUB_WITH_OVERFLOW_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(UsubWithOverflow);

/// Constructs a new detached `llvm.intr.usub.with.overflow` operation.
pub fn intr_usub_with_overflow<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    result_type: T0,
    location: L,
) -> DetachedUsubWithOverflowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(USUB_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_usub_with_overflow`")
}
